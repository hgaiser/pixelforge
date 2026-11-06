//! GPU-accelerated color format conversion using Vulkan compute shaders.
//!
//! This module provides efficient color space conversion on the GPU,
//! converting from RGB/BGR formats to YUV for video encoding.
//!
//! The output remains on the GPU as a `vk::Image` to avoid unnecessary
//! CPU round-trips when used with a GPU-based video encoder.

mod pipeline;

use crate::encoder::ColorDescription;
use crate::encoder::input::SourceOwnership;
use crate::error::{PixelForgeError, Result};
use crate::sync::TimelinePoint;
use crate::video::TimelineChain;
use crate::vulkan::VideoContext;
use ash::vk;

/// A colour space, used for both ends of a conversion:
/// [`RgbSource::color`](crate::RgbSource::color) for the input, and the
/// stream's colour description for the encoded output.
///
/// The two linear spaces can only be an input. Video files have no way to say
/// "linear", so nothing could be decoded correctly; see
/// [`ColorSpec::is_encodable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ColorSpec {
    /// Ordinary SDR content, the usual desktop and game output.
    ///
    /// sRGB and BT.709 cover the same colours and differ only in the curve
    /// applied to them, which is why the linear one below is named for BT.709
    /// rather than for sRGB.
    Srgb = 0,
    /// scRGB: the same colours as `Srgb`, in linear light.
    ///
    /// Values above 1.0 are allowed and mean brighter than SDR white, which is
    /// how this carries HDR. What a `VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT`
    /// swapchain gives you.
    Bt709Linear = 1,
    /// Like `Bt709Linear`, but with the wider HDR colours.
    Bt2020Linear = 2,
    /// HDR10: wide gamut, PQ encoded. What a
    /// `VK_COLOR_SPACE_HDR10_ST2084_EXT` swapchain gives you.
    Bt2020Pq = 3,
}

impl ColorSpec {
    /// How bright this space's white is, in nits.
    ///
    /// Only used when encoding to HDR, which needs real brightness values
    /// rather than the relative ones SDR content carries. Getting it wrong
    /// makes the picture too bright or too dim, so the spaces differ here:
    /// scRGB defines its white as 80 nits, the others use 203.
    ///
    /// `None` for `Bt2020Pq`, which already carries real brightness values.
    /// Override it with
    /// [`RgbSource::with_reference_white_nits`](crate::RgbSource::with_reference_white_nits).
    pub fn reference_white_nits(&self) -> Option<f32> {
        match self {
            Self::Srgb | Self::Bt2020Linear => Some(203.0),
            Self::Bt709Linear => Some(80.0),
            Self::Bt2020Pq => None,
        }
    }

    /// Whether video can be encoded in this space.
    ///
    /// False for the two linear spaces: a video file has no way to record that
    /// it holds linear light, so a player would get it wrong.
    pub fn is_encodable(&self) -> bool {
        matches!(self, Self::Srgb | Self::Bt2020Pq)
    }

    /// What to tell a player about video in this space.
    ///
    /// `None` when [`Self::is_encodable`] is false. The range is set
    /// separately.
    pub fn color_description(&self) -> Option<ColorDescription> {
        match self {
            // sRGB and BT.709 differ slightly in their curves, but video
            // always labels this case BT.709 and players treat the two the
            // same, so labelling it anything else would be surprising.
            Self::Srgb => Some(ColorDescription::bt709()),
            Self::Bt2020Pq => Some(ColorDescription::bt2020_pq()),
            Self::Bt709Linear | Self::Bt2020Linear => None,
        }
    }
}

/// Luma and chroma quantization range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorRange {
    /// Studio range: luma 16..235, chroma 16..240 at 8-bit.
    Limited,
    /// Full range: 0..255 at 8-bit.
    Full,
}

impl ColorRange {
    /// Whether this is [`ColorRange::Full`].
    pub fn is_full(&self) -> bool {
        matches!(self, Self::Full)
    }
}

/// Supported input pixel formats for color conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum RgbFormat {
    /// BGRx (32-bit, blue first, x = unused).
    BGRx,
    /// RGBx (32-bit, red first, x = unused).
    RGBx,
    /// BGRA (32-bit, blue first, alpha last).
    BGRA,
    /// RGBA (32-bit, red first, alpha last).
    RGBA,
    /// ABGR2101010 (packed 10-bit per channel, 2-bit alpha).
    /// Maps to DRM_FORMAT_ABGR2101010 / VK_FORMAT_A2B10G10R10_UNORM_PACK32.
    ABGR2101010,
    /// RGBA16F (64-bit, 16-bit float per channel).
    /// Maps to DRM_FORMAT_ABGR16161616F / VK_FORMAT_R16G16B16A16_SFLOAT.
    ///
    /// The converter treats FP16 data the same as other formats: the
    /// [`ColorSpec`] says what the samples mean, not the pixel format.
    /// [`ColorSpec::Bt709Linear`] is the natural pairing for FP16 from an
    /// `EXTENDED_SRGB_LINEAR_EXT` swapchain; FP16 that is already PQ-encoded
    /// (via the gamescope WSI layer, say) is [`ColorSpec::Bt2020Pq`].
    RGBA16F,
}

impl RgbFormat {
    /// Bytes per pixel for this format.
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            RgbFormat::BGRx
            | RgbFormat::RGBx
            | RgbFormat::BGRA
            | RgbFormat::RGBA
            | RgbFormat::ABGR2101010 => 4,
            RgbFormat::RGBA16F => 8,
        }
    }

    /// Vulkan format for creating image views of this input format.
    pub fn vk_format(&self) -> vk::Format {
        match self {
            RgbFormat::BGRx | RgbFormat::BGRA => vk::Format::B8G8R8A8_UNORM,
            RgbFormat::RGBx | RgbFormat::RGBA => vk::Format::R8G8B8A8_UNORM,
            RgbFormat::ABGR2101010 => vk::Format::A2B10G10R10_UNORM_PACK32,
            RgbFormat::RGBA16F => vk::Format::R16G16B16A16_SFLOAT,
        }
    }
}

/// Supported output YUV formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputFormat {
    /// NV12: Y plane followed by interleaved UV (4:2:0), 8-bit.
    NV12,
    /// I420: Y plane, U plane, V plane (4:2:0), 8-bit. The shader has the
    /// mode, and no encoder input takes it.
    #[allow(dead_code)]
    I420,
    /// YUV444 8-bit: 2-plane semi-planar (Y plane + interleaved UV) at full resolution.
    YUV444,
    /// P010: Y plane followed by interleaved UV (4:2:0), 10-bit in 16-bit words.
    P010,
    /// YUV444 10-bit: 2-plane semi-planar (Y plane + interleaved UV) in 16-bit words.
    YUV444P10,
}

impl OutputFormat {
    /// Calculate output size in bytes for given dimensions.
    ///
    /// The returned size is always a multiple of 4, since the compute shader writes
    /// to a `uint[]` buffer and `vkCmdFillBuffer` requires 4-byte aligned sizes.
    pub fn output_size(&self, width: u32, height: u32) -> usize {
        let pixel_count = (width * height) as usize;
        let raw = match self {
            OutputFormat::NV12 | OutputFormat::I420 => pixel_count * 3 / 2,
            OutputFormat::YUV444 => {
                // Y plane (aligned to 4 bytes) + UV interleaved plane.
                crate::align4(pixel_count) + pixel_count * 2
            }
            // 10-bit formats use 2 bytes per sample.
            OutputFormat::P010 => pixel_count * 3, // Y (2 bytes) + UV (1 byte each, half res)
            OutputFormat::YUV444P10 => {
                // Y plane (2 bytes/sample, aligned to 4 bytes) + UV interleaved (4 bytes/pixel).
                crate::align4(pixel_count * 2) + pixel_count * 4
            }
        };
        crate::align4(raw)
    }

    /// Returns true if this is a 10-bit format.
    #[cfg(test)]
    pub fn is_10bit(&self) -> bool {
        matches!(self, OutputFormat::P010 | OutputFormat::YUV444P10)
    }

    /// Bytes per sample for this format.
    #[cfg(test)]
    pub fn bytes_per_sample(&self) -> usize {
        if self.is_10bit() { 2 } else { 1 }
    }
}

/// Configuration for the color converter.
///
/// Colour is two decisions: [`source`](Self::source) for what the input pixels
/// are, [`target`](Self::target) for what the encoded stream should be.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub(crate) struct ColorConverterConfig {
    /// Input frame width.
    pub width: u32,
    /// Input frame height.
    pub height: u32,
    /// Input pixel format.
    pub input_format: RgbFormat,
    /// Output YUV format.
    pub output_format: OutputFormat,
    /// What the input pixels already are.
    pub source: ColorSpec,
    /// What the encoded stream should be. Also picks the YUV matrix.
    ///
    /// Must be a space video can be encoded in, see
    /// [`ColorSpec::is_encodable`].
    pub target: ColorSpec,
    /// How bright the source's white is, in nits, overriding the source
    /// space's default. Only used when encoding to HDR.
    pub reference_white_nits: Option<f32>,
    /// Whether the shader writes full or limited range.
    ///
    /// The stream has to say the same thing, which is what
    /// [`color_description`](Self::color_description) is for.
    pub range: ColorRange,
}

impl ColorConverterConfig {
    /// Create a new configuration.
    pub fn new(
        width: u32,
        height: u32,
        input_format: RgbFormat,
        output_format: OutputFormat,
        source_color: ColorSpec,
        target_color: ColorSpec,
        color_range: ColorRange,
    ) -> Self {
        Self {
            width,
            height,
            input_format,
            output_format,
            source: source_color,
            target: target_color,
            range: color_range,
            reference_white_nits: None,
        }
    }

    /// The value the shader gets: the override, or the source's default.
    fn effective_reference_white_nits(&self) -> f32 {
        self.reference_white_nits
            .or_else(|| self.source.reference_white_nits())
            .unwrap_or(0.0)
    }
    /// The stream colour this configuration converts to.
    #[cfg(test)]
    pub fn color_description(&self) -> Option<ColorDescription> {
        Some(
            self.target
                .color_description()?
                .with_full_range(self.range.is_full()),
        )
    }

    /// Whether the shader can get from this source to this target.
    ///
    /// Anything that would need tone mapping or a forward gamma encode is not
    /// implemented, and is rejected rather than silently passed through.
    pub(crate) fn conversion_supported(&self) -> bool {
        match self.target {
            // Anything can be converted to HDR.
            ColorSpec::Bt2020Pq => true,
            // Converting to SDR would need tone mapping or a gamma curve.
            ColorSpec::Srgb => matches!(self.source, ColorSpec::Srgb),
            // Cannot be encoded at all, see `ColorSpec::is_encodable`.
            ColorSpec::Bt709Linear | ColorSpec::Bt2020Linear => false,
        }
    }

    /// Error explaining why [`Self::conversion_supported`] said no.
    pub(crate) fn unsupported_conversion(&self) -> PixelForgeError {
        let reason = if !self.target.is_encodable() {
            "video cannot be encoded in a linear space, so it can only be a source"
        } else {
            "converting to SDR would need tone mapping or a gamma curve applied, \
             which the shader does not do"
        };
        PixelForgeError::InvalidInput(format!(
            "no conversion from {:?} to {:?}: {reason}",
            self.source, self.target
        ))
    }
}

/// GPU-based color format converter.
///
/// Uses Vulkan compute shaders to convert RGB/BGR formats to YUV.
/// with minimal latency and high throughput.
pub(crate) struct ColorConverter {
    context: VideoContext,
    config: ColorConverterConfig,

    // Compute pipeline resources.
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    // Sampler for texelFetch on the source image.
    sampler: vk::Sampler,

    // The view the current conversion reads its source through. A view is
    // made per conversion rather than cached by image handle: a driver may
    // give a new image a destroyed one's handle, and a cache would then read
    // the new image through a view of the old one.
    src_view: Option<vk::ImageView>,

    // Output buffer (compute shader writes here).
    output_buffer: vk::Buffer,
    output_memory: vk::DeviceMemory,
    output_buffer_size: usize,

    // Binds both descriptors inside the command buffer each frame.
    push_descriptor: ash::khr::push_descriptor::Device,

    // Command resources.
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    /// Whether a conversion was submitted and `fence` not yet waited on. The
    /// next conversion reuses the command buffer and output buffer, and
    /// destroys this one's source view, so it waits for this one first.
    in_flight: bool,
    /// Signalled by each conversion, for [`Self::convert_with`] to hand out.
    timeline: TimelineChain,
}

impl ColorConverter {
    /// Create a new color converter.
    ///
    /// Fails if the shader has no path from the configured
    /// [`source`](ColorConverterConfig::source) to its
    /// [`target`](ColorConverterConfig::target).
    pub fn new(context: VideoContext, config: ColorConverterConfig) -> Result<Self> {
        if !config.conversion_supported() {
            return Err(config.unsupported_conversion());
        }
        pipeline::create_converter(context, config)
    }

    /// The stream colour this converter produces.
    #[cfg(test)]
    pub fn color_description(&self) -> ColorDescription {
        self.config
            .color_description()
            .expect("a converter cannot be built with a target that has no description")
    }

    /// Get the converter configuration.
    pub fn config(&self) -> &ColorConverterConfig {
        &self.config
    }

    /// The buffer the shader writes the YUV planes to.
    #[cfg(test)]
    pub fn output_buffer(&self) -> vk::Buffer {
        self.output_buffer
    }

    /// Whether [`Self::reconfigure`] can switch this converter to `config`:
    /// the same size and output, which its buffers are made for.
    pub fn can_become(&self, config: &ColorConverterConfig) -> bool {
        (config.width, config.height, config.output_format)
            == (
                self.config.width,
                self.config.height,
                self.config.output_format,
            )
    }

    /// Switch to `config` for subsequent conversions: its input format,
    /// source, target, range and reference white. They all reach the shader
    /// as push constants, and the source view is made per conversion, so
    /// nothing is made again.
    ///
    /// Returns an error, and leaves the converter untouched, if the shader
    /// cannot do that conversion or [`Self::can_become`] is false.
    pub fn reconfigure(&mut self, config: ColorConverterConfig) -> Result<()> {
        if !self.can_become(&config) {
            return Err(PixelForgeError::InvalidInput(
                "a converter cannot change its size or output format".to_string(),
            ));
        }
        if !config.conversion_supported() {
            return Err(config.unsupported_conversion());
        }
        self.config = config;
        Ok(())
    }

    /// Build buffer-to-image copy regions for multi-planar YUV formats.
    ///
    /// For multi-planar formats like NV12, I420, and YUV444, we need separate.
    /// copy regions for each plane with the appropriate aspect mask.
    fn build_buffer_to_image_copy_regions(&self) -> Vec<vk::BufferImageCopy> {
        match self.config.output_format {
            OutputFormat::NV12 => {
                // NV12: Y plane (PLANE_0) followed by interleaved UV plane (PLANE_1)
                let y_size = (self.config.width * self.config.height) as u64;
                vec![
                    // Y plane.
                    vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_0,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                    // UV plane (interleaved, half resolution)
                    vk::BufferImageCopy {
                        buffer_offset: y_size,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_1,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width / 2,
                            height: self.config.height / 2,
                            depth: 1,
                        },
                    },
                ]
            }
            OutputFormat::I420 => {
                // I420: Y plane, U plane, V plane (all separate)
                let y_size = (self.config.width * self.config.height) as u64;
                let uv_size = y_size / 4;
                vec![
                    // Y plane.
                    vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_0,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                    // U plane.
                    vk::BufferImageCopy {
                        buffer_offset: y_size,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_1,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width / 2,
                            height: self.config.height / 2,
                            depth: 1,
                        },
                    },
                    // V plane.
                    vk::BufferImageCopy {
                        buffer_offset: y_size + uv_size,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_2,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width / 2,
                            height: self.config.height / 2,
                            depth: 1,
                        },
                    },
                ]
            }
            OutputFormat::YUV444 => {
                // YUV444 8-bit 2-plane: Y plane at full resolution, UV interleaved at full resolution.
                // Align Y plane size to 4 bytes for VkBufferImageCopy::bufferOffset compliance.
                let y_size =
                    crate::align4((self.config.width * self.config.height) as usize) as u64;
                vec![
                    // Y plane.
                    vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_0,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                    // UV plane (interleaved, full resolution).
                    vk::BufferImageCopy {
                        buffer_offset: y_size,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_1,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                ]
            }
            OutputFormat::P010 => {
                // P010: 10-bit NV12 with 16-bit samples.
                // Y plane: full resolution, 2 bytes per sample.
                // UV plane: half resolution, interleaved, 2 bytes per component.
                let y_size = (self.config.width * self.config.height * 2) as u64;
                vec![
                    // Y plane (16-bit samples).
                    vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_0,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                    // UV plane (interleaved, half resolution, 16-bit per component).
                    vk::BufferImageCopy {
                        buffer_offset: y_size,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_1,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width / 2,
                            height: self.config.height / 2,
                            depth: 1,
                        },
                    },
                ]
            }
            OutputFormat::YUV444P10 => {
                // YUV444 10-bit: 2-plane format (Y plane, UV interleaved).
                // Align Y plane size to 4 bytes for VkBufferImageCopy::bufferOffset compliance.
                let y_size =
                    crate::align4((self.config.width * self.config.height * 2) as usize) as u64;
                vec![
                    // Y plane (16-bit samples).
                    vk::BufferImageCopy {
                        buffer_offset: 0,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_0,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                    // UV plane (interleaved, full resolution, 16-bit per component).
                    vk::BufferImageCopy {
                        buffer_offset: y_size,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_1,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                        image_extent: vk::Extent3D {
                            width: self.config.width,
                            height: self.config.height,
                            depth: 1,
                        },
                    },
                ]
            }
        }
    }

    /// Convert `src_image` into `target_image`, the encoder's input image,
    /// which is left in `VIDEO_ENCODE_SRC_KHR` layout.
    ///
    /// The conversion waits for `waits` on the GPU, acquires and releases the
    /// source as `ownership` says, signals `extra_signals` with its own
    /// timeline, and returns the point it signals when done. The next call
    /// waits for this conversion to finish before recording, since every
    /// call reuses the same command buffer.
    pub(crate) fn convert_with(
        &mut self,
        src_image: vk::Image,
        ownership: SourceOwnership,
        target_image: vk::Image,
        waits: &[vk::SemaphoreSubmitInfo<'static>],
        extra_signals: &[vk::SemaphoreSubmitInfo<'static>],
    ) -> Result<TimelinePoint> {
        // Before anything is reused, and before the previous view goes.
        self.wait_idle()?;
        let src_view = self.create_src_view(src_image)?;

        let device = self.context.device();

        // Reset and record command buffer.
        unsafe {
            device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(PixelForgeError::vk("vkResetCommandBuffer"))?;

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .map_err(PixelForgeError::vk("vkBeginCommandBuffer"))?;

            // --- Phase 1: Transition source image for shader read ---

            let compute_family = self.context.compute_queue_family();
            let src_barrier = [ownership.acquire_barrier(
                src_image,
                compute_family,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ,
            )];
            self.context.sync2().cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&src_barrier),
            );

            // --- Phase 2: Run compute shader (reads source image directly) ---

            // Clear output buffer to zero before compute shader runs.
            let output_size = self
                .config
                .output_format
                .output_size(self.config.width, self.config.height);
            device.cmd_fill_buffer(
                self.command_buffer,
                self.output_buffer,
                0,
                output_size as vk::DeviceSize,
                0,
            );

            // Barrier: fill buffer write -> shader read/write
            let fill_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .buffer(self.output_buffer)
                .size(vk::WHOLE_SIZE);

            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[fill_barrier],
                &[],
            );

            // Bind pipeline.
            device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );

            let image_info = [vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(src_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(self.output_buffer)
                .offset(0)
                .range(self.output_buffer_size as vk::DeviceSize)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_info),
                vk::WriteDescriptorSet::default()
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_info),
            ];
            self.push_descriptor.cmd_push_descriptor_set(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &writes,
            );

            // Push constants: width, height, input_format, output_format,
            // source, target, range, reference_white_nits.
            let push_constants: [u32; 8] = [
                self.config.width,
                self.config.height,
                self.config.input_format as u32,
                self.config.output_format as u32,
                self.config.source as u32,
                self.config.target as u32,
                self.config.range.is_full() as u32,
                self.config.effective_reference_white_nits().to_bits(),
            ];
            let push_constants_bytes: &[u8] = std::slice::from_raw_parts(
                push_constants.as_ptr() as *const u8,
                std::mem::size_of_val(&push_constants),
            );
            device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constants_bytes,
            );

            // Dispatch workgroups (8x8 workgroup size)
            let workgroup_x = self.config.width.div_ceil(8);
            let workgroup_y = self.config.height.div_ceil(8);
            device.cmd_dispatch(self.command_buffer, workgroup_x, workgroup_y, 1);

            // --- Phase 3: Copy output buffer to target image (encoder's input) ---

            // Memory barrier: buffer write -> buffer read
            let output_buffer_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .buffer(self.output_buffer)
                .size(vk::WHOLE_SIZE);

            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[output_buffer_barrier],
                &[],
            );

            // Transition target image (encoder's input) to TRANSFER_DST layout.
            // The encoder's init clears the input image and leaves it in
            // VIDEO_ENCODE_SRC_KHR, so we must use that (not UNDEFINED) as the
            // old layout on every frame, including the first.
            let target_barrier_to_transfer = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(target_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[target_barrier_to_transfer],
            );

            // Copy buffer to target image - use per-plane copies for multi-planar formats.
            let copy_regions = self.build_buffer_to_image_copy_regions();

            device.cmd_copy_buffer_to_image(
                self.command_buffer,
                self.output_buffer,
                target_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy_regions,
            );

            // Transition target image to VIDEO_ENCODE_SRC_KHR layout for encoding.
            let target_barrier_to_encode = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
                .image(target_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[target_barrier_to_encode],
            );

            // Transition the source back to GENERAL for reuse, handing it back
            // to its owner if it has one.
            let src_back = [ownership.release_barrier(
                src_image,
                compute_family,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
            )];
            self.context.sync2().cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&src_back),
            );

            device
                .end_command_buffer(self.command_buffer)
                .map_err(PixelForgeError::vk("vkEndCommandBuffer"))?;
        }

        let (semaphore, value) = self.timeline.pending_signal();
        let mut signals = vec![
            vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore)
                .value(value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        ];
        signals.extend_from_slice(extra_signals);
        let command_buffers =
            [vk::CommandBufferSubmitInfo::default().command_buffer(self.command_buffer)];
        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(waits)
            .command_buffer_infos(&command_buffers)
            .signal_semaphore_infos(&signals);

        unsafe {
            device
                .reset_fences(&[self.fence])
                .map_err(PixelForgeError::vk("vkResetFences"))?;
            self.context
                .sync2()
                .queue_submit2(self.context.compute_queue(), &[submit_info], self.fence)
                .map_err(PixelForgeError::vk("vkQueueSubmit2"))?;
        }
        self.timeline.commit();
        self.in_flight = true;

        Ok(TimelinePoint::new(semaphore, value))
    }

    /// Wait on the CPU for the last conversion submitted, if it is still
    /// running, and destroy the view it read its source through. A conversion
    /// that failed before submitting leaves its view for this to destroy too.
    fn wait_idle(&mut self) -> Result<()> {
        if self.in_flight {
            unsafe {
                self.context
                    .device()
                    .wait_for_fences(&[self.fence], true, u64::MAX)
            }
            .map_err(PixelForgeError::vk("vkWaitForFences"))?;
            self.in_flight = false;
        }
        self.destroy_src_view();
        Ok(())
    }

    /// Create the view this conversion reads `src_image` through. It lives
    /// until the conversion is done, see [`Self::wait_idle`].
    fn create_src_view(&mut self, src_image: vk::Image) -> Result<vk::ImageView> {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(src_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(self.config.input_format.vk_format())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe { self.context.device().create_image_view(&view_info, None) }
            .map_err(PixelForgeError::vk("vkCreateImageView"))?;

        self.src_view = Some(view);
        Ok(view)
    }

    /// Destroy the source view, once nothing on the GPU reads through it.
    fn destroy_src_view(&mut self) {
        if let Some(view) = self.src_view.take() {
            unsafe { self.context.device().destroy_image_view(view, None) };
        }
    }
}

impl Drop for ColorConverter {
    fn drop(&mut self) {
        // Nothing below may be destroyed while a conversion still uses it.
        let _ = self.wait_idle();
        unsafe {
            let device = self.context.device();
            // Not destroyed here: an encode may still be waiting on a point
            // this converter handed out, and that wait outlives the
            // conversion it is waiting for.
            self.context.retire_semaphore(self.timeline.semaphore());

            // Destroy sampler.
            device.destroy_sampler(self.sampler, None);

            // Destroy output buffer and its memory.
            device.destroy_buffer(self.output_buffer, None);
            device.free_memory(self.output_memory, None);

            // Destroy pipeline resources.
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            // Destroy command resources.
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}

#[cfg(test)]
mod device_tests;

#[cfg(test)]
mod tests {
    use super::*;

    // ========================
    // RgbFormat tests.
    // ========================

    #[test]
    fn test_input_format_bytes_per_pixel() {
        assert_eq!(RgbFormat::BGRx.bytes_per_pixel(), 4);
        assert_eq!(RgbFormat::RGBx.bytes_per_pixel(), 4);
        assert_eq!(RgbFormat::BGRA.bytes_per_pixel(), 4);
        assert_eq!(RgbFormat::RGBA.bytes_per_pixel(), 4);
        assert_eq!(RgbFormat::ABGR2101010.bytes_per_pixel(), 4);
        assert_eq!(RgbFormat::RGBA16F.bytes_per_pixel(), 8);
    }

    #[test]
    fn test_input_format_enum_values() {
        // Verify enum values match shader expectations.
        assert_eq!(RgbFormat::BGRx as u32, 0);
        assert_eq!(RgbFormat::RGBx as u32, 1);
        assert_eq!(RgbFormat::BGRA as u32, 2);
        assert_eq!(RgbFormat::RGBA as u32, 3);
        assert_eq!(RgbFormat::ABGR2101010 as u32, 4);
        assert_eq!(RgbFormat::RGBA16F as u32, 5);
    }

    #[test]
    fn test_input_format_vk_format() {
        assert_eq!(RgbFormat::BGRx.vk_format(), vk::Format::B8G8R8A8_UNORM);
        assert_eq!(RgbFormat::BGRA.vk_format(), vk::Format::B8G8R8A8_UNORM);
        assert_eq!(RgbFormat::RGBx.vk_format(), vk::Format::R8G8B8A8_UNORM);
        assert_eq!(RgbFormat::RGBA.vk_format(), vk::Format::R8G8B8A8_UNORM);
        assert_eq!(
            RgbFormat::ABGR2101010.vk_format(),
            vk::Format::A2B10G10R10_UNORM_PACK32
        );
        assert_eq!(
            RgbFormat::RGBA16F.vk_format(),
            vk::Format::R16G16B16A16_SFLOAT
        );
    }

    // ========================
    // ColorSpec / ColorSpec / ColorRange tests.
    // ========================

    #[test]
    fn spec_discriminants_match_the_shader() {
        // The shader's SPEC_* defines. One set, used for both push constants.
        assert_eq!(ColorSpec::Srgb as u32, 0);
        assert_eq!(ColorSpec::Bt709Linear as u32, 1);
        assert_eq!(ColorSpec::Bt2020Linear as u32, 2);
        assert_eq!(ColorSpec::Bt2020Pq as u32, 3);
    }

    #[test]
    fn range_matches_the_shader() {
        // The shader's `range == 0u` test.
        assert!(!ColorRange::Limited.is_full());
        assert!(ColorRange::Full.is_full());
    }

    #[test]
    fn only_encodable_specs_describe_a_stream() {
        // A decoder can be told about these two.
        assert_eq!(
            ColorSpec::Srgb.color_description(),
            Some(ColorDescription::bt709())
        );
        assert_eq!(
            ColorSpec::Bt2020Pq.color_description(),
            Some(ColorDescription::bt2020_pq())
        );
        assert!(ColorSpec::Srgb.is_encodable());
        assert!(ColorSpec::Bt2020Pq.is_encodable());

        // Linear light cannot be encoded, and these say so.
        for spec in [ColorSpec::Bt709Linear, ColorSpec::Bt2020Linear] {
            assert!(!spec.is_encodable(), "{spec:?}");
            assert_eq!(spec.color_description(), None, "{spec:?}");
        }
    }

    #[test]
    fn scrgb_white_is_not_the_srgb_reference() {
        // scRGB's white is 80 nits, not the 203 the other spaces use. Using
        // 203 for it makes HDR output far too bright.
        assert_eq!(ColorSpec::Bt709Linear.reference_white_nits(), Some(80.0));
        assert_eq!(ColorSpec::Srgb.reference_white_nits(), Some(203.0));
        assert_eq!(ColorSpec::Bt2020Linear.reference_white_nits(), Some(203.0));
    }

    #[test]
    fn pq_has_no_reference_white() {
        // Already carries real brightness values, so there is nothing to ask.
        assert_eq!(ColorSpec::Bt2020Pq.reference_white_nits(), None);
    }

    #[test]
    fn the_override_replaces_the_standard_figure() {
        let base = ColorConverterConfig::new(
            64,
            64,
            RgbFormat::BGRA,
            OutputFormat::P010,
            ColorSpec::Bt709Linear,
            ColorSpec::Bt2020Pq,
            ColorRange::Full,
        );
        // Unset, the source's own figure is what the shader gets.
        assert_eq!(base.reference_white_nits, None);
        assert_eq!(base.effective_reference_white_nits(), 80.0);

        let mut overridden = base.clone();
        overridden.reference_white_nits = Some(203.0);
        assert_eq!(overridden.effective_reference_white_nits(), 203.0);
        // The space itself is untouched; the override lives on the conversion.
        assert_eq!(overridden.source, ColorSpec::Bt709Linear);
        assert_eq!(overridden.source.reference_white_nits(), Some(80.0));
    }

    #[test]
    fn a_pq_source_needs_no_reference_white() {
        // Nothing reads it, so an absent figure has to be harmless rather than
        // a panic or a NaN reaching the shader.
        let config = ColorConverterConfig::new(
            64,
            64,
            RgbFormat::BGRA,
            OutputFormat::P010,
            ColorSpec::Bt2020Pq,
            ColorSpec::Bt2020Pq,
            ColorRange::Full,
        );
        assert_eq!(config.effective_reference_white_nits(), 0.0);
    }

    // ========================
    // color_description() tests.
    // ========================

    fn config_for(target: ColorSpec, range: ColorRange) -> ColorConverterConfig {
        config_from(ColorSpec::Srgb, target, range)
    }

    fn config_from(
        source: ColorSpec,
        target: ColorSpec,
        range: ColorRange,
    ) -> ColorConverterConfig {
        ColorConverterConfig::new(
            64,
            64,
            RgbFormat::BGRA,
            OutputFormat::NV12,
            source,
            target,
            range,
        )
    }

    #[test]
    fn description_follows_the_target() {
        let sdr = config_for(ColorSpec::Srgb, ColorRange::Limited)
            .color_description()
            .expect("sRGB is encodable");
        assert_eq!(sdr, ColorDescription::bt709());
        assert!(!sdr.is_hdr());

        let hdr = config_for(ColorSpec::Bt2020Pq, ColorRange::Limited)
            .color_description()
            .expect("PQ is encodable");
        assert_eq!(hdr, ColorDescription::bt2020_pq());
        assert!(hdr.is_hdr());
    }

    #[test]
    fn description_follows_the_range() {
        // The drift this whole split exists to prevent: what the shader
        // quantizes to and what the VUI claims are now one decision.
        for target in [ColorSpec::Srgb, ColorSpec::Bt2020Pq] {
            for range in [ColorRange::Limited, ColorRange::Full] {
                let config = config_for(target, range);
                assert_eq!(
                    config
                        .color_description()
                        .expect("both targets are encodable")
                        .full_range,
                    config.range.is_full(),
                    "{target:?} / {range:?}"
                );
            }
        }
    }

    #[test]
    fn description_ignores_the_source() {
        // The source says nothing about the encoded stream, so it must not
        // reach the declaration.
        let from_srgb = config_from(ColorSpec::Srgb, ColorSpec::Bt2020Pq, ColorRange::Full);
        let from_scrgb = config_from(
            ColorSpec::Bt709Linear,
            ColorSpec::Bt2020Pq,
            ColorRange::Full,
        );
        let from_pq = config_from(ColorSpec::Bt2020Pq, ColorSpec::Bt2020Pq, ColorRange::Full);
        assert_eq!(
            from_srgb.color_description(),
            from_scrgb.color_description()
        );
        assert_eq!(from_srgb.color_description(), from_pq.color_description());
    }

    // ========================
    // Supported conversion tests.
    // ========================

    #[test]
    fn every_source_can_reach_pq() {
        for source in [
            ColorSpec::Srgb,
            ColorSpec::Bt709Linear,
            ColorSpec::Bt2020Linear,
            ColorSpec::Bt2020Pq,
        ] {
            let config = config_from(source, ColorSpec::Bt2020Pq, ColorRange::Full);
            assert!(config.conversion_supported(), "{source:?}");
        }
    }

    #[test]
    fn only_srgb_can_reach_sdr() {
        // Everything else would need a forward gamma encode or tone mapping,
        // and the shader does neither. Rejected beats silently passed through.
        let ok = config_from(ColorSpec::Srgb, ColorSpec::Srgb, ColorRange::Limited);
        assert!(ok.conversion_supported());

        for source in [
            ColorSpec::Bt709Linear,
            ColorSpec::Bt2020Linear,
            ColorSpec::Bt2020Pq,
        ] {
            let config = config_from(source, ColorSpec::Srgb, ColorRange::Limited);
            assert!(!config.conversion_supported(), "{source:?}");
        }
    }

    // ========================
    // OutputFormat tests.
    // ========================

    #[test]
    fn test_output_format_size_nv12() {
        // NV12: Y plane + interleaved UV at half resolution = 1.5 * pixel_count.
        assert_eq!(OutputFormat::NV12.output_size(8, 8), 96); // 64 * 1.5 = 96
        assert_eq!(OutputFormat::NV12.output_size(16, 16), 384); // 256 * 1.5 = 384
        assert_eq!(
            OutputFormat::NV12.output_size(1920, 1080),
            1920 * 1080 * 3 / 2
        );
    }

    #[test]
    fn test_output_format_size_i420() {
        // I420: Y plane + U plane (quarter) + V plane (quarter) = 1.5 * pixel_count.
        assert_eq!(OutputFormat::I420.output_size(8, 8), 96);
        assert_eq!(OutputFormat::I420.output_size(16, 16), 384);
        assert_eq!(
            OutputFormat::I420.output_size(1920, 1080),
            1920 * 1080 * 3 / 2
        );
    }

    #[test]
    fn test_output_format_size_yuv444() {
        // YUV444: Full resolution Y + U + V = 3 * pixel_count.
        assert_eq!(OutputFormat::YUV444.output_size(8, 8), 192); // 64 * 3 = 192
        assert_eq!(OutputFormat::YUV444.output_size(16, 16), 768); // 256 * 3 = 768
        assert_eq!(
            OutputFormat::YUV444.output_size(1920, 1080),
            1920 * 1080 * 3
        );
    }

    #[test]
    fn test_output_format_size_standard_resolutions() {
        // Common video resolutions.
        let resolutions = [
            (320, 240),   // QVGA
            (640, 480),   // VGA
            (1280, 720),  // HD
            (1920, 1080), // Full HD
            (3840, 2160), // 4K
        ];

        for (width, height) in resolutions {
            let pixels = (width * height) as usize;

            // NV12 and I420 should be 1.5x pixels.
            assert_eq!(
                OutputFormat::NV12.output_size(width, height),
                pixels * 3 / 2
            );
            assert_eq!(
                OutputFormat::I420.output_size(width, height),
                pixels * 3 / 2
            );

            // YUV444 should be 3x pixels.
            assert_eq!(OutputFormat::YUV444.output_size(width, height), pixels * 3);
        }
    }

    #[test]
    fn test_output_format_enum_values() {
        // Verify enum values match shader expectations.
        assert_eq!(OutputFormat::NV12 as u32, 0);
        assert_eq!(OutputFormat::I420 as u32, 1);
        assert_eq!(OutputFormat::YUV444 as u32, 2);
    }

    // ========================
    // ColorConverterConfig tests.
    // ========================

    #[test]
    fn test_config_clone() {
        let config = ColorConverterConfig::new(
            1920,
            1080,
            RgbFormat::BGRx,
            OutputFormat::NV12,
            ColorSpec::Srgb,
            ColorSpec::Srgb,
            ColorRange::Full,
        );

        let cloned = config.clone();
        assert_eq!(cloned.width, 1920);
        assert_eq!(cloned.height, 1080);
        assert_eq!(cloned.input_format, RgbFormat::BGRx);
        assert_eq!(cloned.output_format, OutputFormat::NV12);
        assert_eq!(cloned.target, ColorSpec::Srgb);
        assert!(cloned.range.is_full());
    }

    #[test]
    fn test_config_debug() {
        let config = ColorConverterConfig::new(
            640,
            480,
            RgbFormat::RGBA,
            OutputFormat::I420,
            ColorSpec::Srgb,
            ColorSpec::Srgb,
            ColorRange::Full,
        );

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("640"));
        assert!(debug_str.contains("480"));
        assert!(debug_str.contains("RGBA"));
        assert!(debug_str.contains("I420"));
    }

    // ========================
    // ColorConverter Vulkan tests (require hardware)
    // ========================
    // These tests require Vulkan support and are gated behind a feature.
    // or will be skipped if Vulkan initialization fails.

    /// Helper to create a Vulkan context for testing.
    /// Returns None if Vulkan is not available.
    fn create_test_context() -> Option<VideoContext> {
        use crate::VideoContextBuilder;

        VideoContextBuilder::new()
            .app_name("ColorConverter Test")
            .enable_validation(false)
            .build()
            .ok()
    }

    #[test]
    fn test_converter_creation() {
        let Some(context) = create_test_context() else {
            eprintln!("Skipping test_converter_creation: Vulkan not available");
            return;
        };

        let config = ColorConverterConfig::new(
            64,
            64,
            RgbFormat::BGRx,
            OutputFormat::NV12,
            ColorSpec::Srgb,
            ColorSpec::Srgb,
            ColorRange::Full,
        );

        let result = ColorConverter::new(context, config);
        assert!(
            result.is_ok(),
            "Failed to create ColorConverter: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_converter_creation_various_formats() {
        let Some(context) = create_test_context() else {
            eprintln!("Skipping test: Vulkan not available");
            return;
        };

        let input_formats = [
            RgbFormat::BGRx,
            RgbFormat::RGBx,
            RgbFormat::BGRA,
            RgbFormat::RGBA,
        ];
        let output_formats = [OutputFormat::NV12, OutputFormat::I420, OutputFormat::YUV444];

        for input_format in &input_formats {
            for output_format in &output_formats {
                let config = ColorConverterConfig::new(
                    32,
                    32,
                    *input_format,
                    *output_format,
                    ColorSpec::Srgb,
                    ColorSpec::Srgb,
                    ColorRange::Full,
                );

                let result = ColorConverter::new(context.clone(), config);
                assert!(
                    result.is_ok(),
                    "Failed to create ColorConverter with {:?} -> {:?}: {:?}",
                    input_format,
                    output_format,
                    result.err()
                );
            }
        }
    }
}
