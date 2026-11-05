//! Image utilities for creating and uploading video frames.
//!
//! This module provides [`YuvUploader`], a helper for uploading YUV data from the CPU
//! to the GPU. The image it creates is a transfer-only staging image (not a Vulkan Video
//! image) and must be copied into an encoder's input image before encoding.
//! Use [`YuvUploader::upload`] to upload to the internal image, then submit
//! [`YuvUploader::source`] with [`Frame::image`](crate::Frame::image).
//! Alternatively, use [`YuvUploader::upload_to`] to upload directly into an
//! encoder's [`InputSlot`] (from
//! [`Encoder::input_slot`](crate::encoder::Encoder::input_slot)).

use crate::encoder::{BitDepth, ChromaFormat, Encoder, Handoff, InputSlot, Source, SourceImage};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;
use ash::vk;

/// How the YUV samples handed to [`YuvUploader::upload`] are laid out in
/// memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YuvLayout {
    /// 8-bit 4:2:0 planes: Y, then U, then V (yuv420p). For a 10-bit image,
    /// each sample becomes the high 8 bits of the 10.
    I420,
    /// 8-bit 4:2:0: a Y plane, then an interleaved UV plane. For an 8-bit
    /// image only.
    Nv12,
    /// 10-bit 4:2:0 planes: Y, then U, then V, each sample a little-endian
    /// 16-bit word with the value in its low 10 bits (yuv420p10le). For a
    /// 10-bit image only.
    I420P10,
    /// 8-bit 4:4:4 planes: Y, then U, then V (yuv444p). For a 10-bit image,
    /// each sample becomes the high 8 bits of the 10.
    I444,
}

impl YuvLayout {
    /// The size of one `width` by `height` frame in this layout, in bytes.
    pub fn frame_size(&self, width: u32, height: u32) -> usize {
        let luma = (width * height) as usize;
        match self {
            Self::I420 | Self::Nv12 => luma * 3 / 2,
            Self::I420P10 | Self::I444 => luma * 3,
        }
    }
}

/// A GPU image for staging YUV frame data before encoding.
///
/// This struct owns a transfer-only Vulkan image (no video profile, no
/// `VIDEO_ENCODE_SRC_KHR` usage) with NV12/P010 format (YUV420) or 2-plane
/// semi-planar YUV444 format. It provides methods to upload YUV data from the
/// CPU, which can then be copied into an encoder's input image for encoding.
///
/// The image is **not** directly usable as a Vulkan Video encode source. To
/// encode, either:
/// - [`upload`](Self::upload) to this image and submit [`Self::source`] with
///   [`Frame::image`](crate::Frame::image), which copies it into the encoder's
///   input image, or
/// - [`upload_to`](Self::upload_to) the encoder's input image directly,
///   through an [`InputSlot`].
pub struct YuvUploader {
    /// Keeps track of the frames that read `image`.
    source: SourceImage,
    context: VideoContext,
    image: vk::Image,
    /// Current layout of `self.image`.
    image_layout: vk::ImageLayout,
    memory: vk::DeviceMemory,
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    staging_size: usize,
    /// Kept for automatic cleanup via Drop.
    #[allow(dead_code)]
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    width: u32,
    height: u32,
    bit_depth: BitDepth,
    chroma_format: ChromaFormat,
}

impl YuvUploader {
    /// A staging image for the frames of `encoder`: its size, chroma format
    /// and bit depth.
    ///
    /// An error for an encoder with an RGB [`Source`]: this uploads YUV.
    pub fn for_encoder(encoder: &Encoder) -> Result<Self> {
        let config = encoder.config();
        if config.source != Source::Yuv {
            return Err(PixelForgeError::InvalidInput(
                "a YuvUploader uploads YUV, and the encoder's source is RGB".to_string(),
            ));
        }
        Self::new(
            encoder.context().clone(),
            config.dimensions.width,
            config.dimensions.height,
            config.bit_depth,
            config.chroma_format,
        )
    }

    /// Create a new staging image for uploading YUV frame data.
    ///
    /// Creates a transfer-only image suitable for staging YUV data before
    /// copying it into an encoder's input image. The image has no video
    /// profile and uses `TRANSFER_DST | TRANSFER_SRC` usage flags.
    ///
    /// # Arguments
    /// * `context` - The Vulkan video context
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `bit_depth` - Bit depth for the image (8-bit or 10-bit)
    /// * `chroma_format` - Chroma format (YUV420 or YUV444)
    pub fn new(
        context: VideoContext,
        width: u32,
        height: u32,
        bit_depth: BitDepth,
        chroma_format: ChromaFormat,
    ) -> Result<Self> {
        let device = context.device();

        // Select format based on chroma format and bit depth.
        // Use 2-plane semi-planar formats for both YUV420 and YUV444.
        let format = match (chroma_format, bit_depth) {
            (ChromaFormat::Yuv420, BitDepth::Eight) => vk::Format::G8_B8R8_2PLANE_420_UNORM,
            (ChromaFormat::Yuv420, BitDepth::Ten) => {
                vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
            }
            (ChromaFormat::Yuv444, BitDepth::Eight) => vk::Format::G8_B8R8_2PLANE_444_UNORM,
            (ChromaFormat::Yuv444, BitDepth::Ten) => {
                vk::Format::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16
            }
            _ => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "Unsupported chroma format / bit depth combination: {:?} / {:?}",
                    chroma_format, bit_depth
                )));
            }
        };

        // Create the image.
        // This image is used purely for staging (buffer→image copy on the transfer queue),
        // so it only needs TRANSFER_DST | TRANSFER_SRC and no video profile pNext.
        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.create_image(&image_create_info, None) }
            .map_err(PixelForgeError::vk("vkCreateImage"))?;

        // Get memory requirements and allocate.
        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };

        let memory_type_index = context
            .find_memory_type(
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .ok_or_else(|| {
                PixelForgeError::NoSuitableDevice("No suitable memory type for image".to_string())
            })?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { device.allocate_memory(&alloc_info, None) }
            .map_err(PixelForgeError::vk("vkAllocateMemory"))?;

        unsafe { device.bind_image_memory(image, memory, 0) }
            .map_err(PixelForgeError::vk("vkBindImageMemory"))?;

        // Create staging buffer for uploads.
        // Size depends on chroma format, bit depth, and alignment padding between planes.
        let bytes_per_sample_init = match bit_depth {
            BitDepth::Eight => 1usize,
            BitDepth::Ten => 2,
        };
        let luma_pixels = (width * height) as usize;
        let staging_size = match chroma_format {
            ChromaFormat::Yuv444 => {
                // Y plane (aligned to 4 bytes) + UV interleaved plane.
                let y_bytes = luma_pixels * bytes_per_sample_init;
                crate::align4(y_bytes) + luma_pixels * 2 * bytes_per_sample_init
            }
            _ => chroma_format.frame_size(width, height) * bytes_per_sample_init,
        };
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(staging_size as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let staging_buffer = unsafe { device.create_buffer(&buffer_create_info, None) }
            .map_err(PixelForgeError::vk("vkCreateBuffer"))?;

        let staging_mem_requirements =
            unsafe { device.get_buffer_memory_requirements(staging_buffer) };

        let staging_memory_type = context
            .find_memory_type(
                staging_mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .ok_or_else(|| {
                PixelForgeError::NoSuitableDevice(
                    "No suitable memory type for staging buffer".to_string(),
                )
            })?;

        let staging_alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(staging_mem_requirements.size)
            .memory_type_index(staging_memory_type);

        let staging_memory = unsafe { device.allocate_memory(&staging_alloc_info, None) }
            .map_err(PixelForgeError::vk("vkAllocateMemory"))?;

        unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0) }
            .map_err(PixelForgeError::vk("vkBindBufferMemory"))?;

        // Create command pool and buffer for transfers.
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(context.transfer_queue_family())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(PixelForgeError::vk("vkCreateCommandPool"))?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info) }
            .map_err(PixelForgeError::vk("vkAllocateCommandBuffers"))?;
        let command_buffer = command_buffers[0];

        // Create fence for synchronization.
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { device.create_fence(&fence_info, None) }
            .map_err(PixelForgeError::vk("vkCreateFence"))?;

        let source = SourceImage::new(
            &context,
            image,
            Handoff::Device {
                layout: vk::ImageLayout::GENERAL,
            },
        )?;
        let mut input = Self {
            source,
            context,
            image,
            image_layout: vk::ImageLayout::UNDEFINED,
            memory,
            staging_buffer,
            staging_memory,
            staging_size,
            command_pool,
            command_buffer,
            fence,
            width,
            height,
            bit_depth,
            chroma_format,
        };
        // Start from zeroed planes in GENERAL, so the image is a valid source
        // before the first upload.
        unsafe {
            let device = input.context.device();
            let ptr = device
                .map_memory(
                    staging_memory,
                    0,
                    staging_size as vk::DeviceSize,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(PixelForgeError::vk("vkMapMemory"))?;
            std::ptr::write_bytes(ptr as *mut u8, 0, staging_size);
            device.unmap_memory(staging_memory);
        }
        input.copy_staging_to_image()?;
        Ok(input)
    }

    /// Upload a frame of `layout` data to this image.
    ///
    /// Submit it afterwards with [`Frame::image`](crate::Frame::image) and
    /// [`Self::source`]. This waits until the frames submitted from the image
    /// before are done with it.
    pub fn upload(&mut self, layout: YuvLayout, data: &[u8]) -> Result<()> {
        self.write_staging(layout, data)?;
        self.copy_staging_to_image()
    }

    /// Upload a frame of `layout` data into an encoder's input image, so the
    /// encode reads it with no further copy.
    ///
    /// Submit the slot with [`Frame::input`](crate::Frame::input) afterwards.
    pub fn upload_to(&mut self, slot: &InputSlot, layout: YuvLayout, data: &[u8]) -> Result<()> {
        self.write_staging(layout, data)?;
        self.copy_staging_to_image_internal(
            slot.image(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
        )
    }

    /// Convert `data` into the staging buffer, in the image's format.
    fn write_staging(&mut self, layout: YuvLayout, data: &[u8]) -> Result<()> {
        let fits = matches!(
            (layout, self.chroma_format, self.bit_depth),
            (YuvLayout::I420, ChromaFormat::Yuv420, _)
                | (YuvLayout::Nv12, ChromaFormat::Yuv420, BitDepth::Eight)
                | (YuvLayout::I420P10, ChromaFormat::Yuv420, BitDepth::Ten)
                | (YuvLayout::I444, ChromaFormat::Yuv444, _)
        );
        if !fits {
            return Err(PixelForgeError::InvalidInput(format!(
                "{layout:?} data cannot be uploaded to a {:?} {:?}-bit image",
                self.chroma_format, self.bit_depth
            )));
        }
        let expected_size = layout.frame_size(self.width, self.height);
        if data.len() < expected_size {
            return Err(PixelForgeError::InvalidInput(format!(
                "{layout:?} data too small: expected {expected_size} bytes, got {}",
                data.len()
            )));
        }

        let luma = (self.width * self.height) as usize;
        let chroma = match self.chroma_format {
            ChromaFormat::Yuv444 => luma,
            _ => luma / 4,
        };
        let ten_bit = self.bit_depth == BitDepth::Ten;
        let luma_bytes = if ten_bit { luma * 2 } else { luma };
        // The UV plane of a 4:4:4 image starts 4-byte aligned, as
        // `copy_staging_to_image_internal` copies it.
        let uv_offset = match self.chroma_format {
            ChromaFormat::Yuv444 => crate::align4(luma_bytes),
            _ => luma_bytes,
        };

        let device = self.context.device();
        let ptr = unsafe {
            device.map_memory(
                self.staging_memory,
                0,
                self.staging_size as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(PixelForgeError::vk("vkMapMemory"))?;
        // SAFETY: the mapping covers `staging_size` bytes and nothing else
        // uses it until it is unmapped below.
        let dst = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, self.staging_size) };
        let put16 = |dst: &mut [u8], index: usize, value: u16| {
            dst[index * 2..index * 2 + 2].copy_from_slice(&value.to_ne_bytes());
        };

        match layout {
            YuvLayout::Nv12 => dst[..expected_size].copy_from_slice(&data[..expected_size]),
            YuvLayout::I420 | YuvLayout::I444 => {
                let (y, rest) = data.split_at(luma);
                let (u, v) = rest.split_at(chroma);
                if ten_bit {
                    // An 8-bit sample in the high byte is the same value in
                    // P010's 10 high bits.
                    for (i, &s) in y.iter().enumerate() {
                        put16(dst, i, (s as u16) << 8);
                    }
                    let uv = &mut dst[uv_offset..];
                    for i in 0..chroma {
                        put16(uv, i * 2, (u[i] as u16) << 8);
                        put16(uv, i * 2 + 1, (v[i] as u16) << 8);
                    }
                } else {
                    dst[..luma].copy_from_slice(y);
                    let uv = &mut dst[uv_offset..];
                    for i in 0..chroma {
                        uv[i * 2] = u[i];
                        uv[i * 2 + 1] = v[i];
                    }
                }
            }
            YuvLayout::I420P10 => {
                // The source has the 10 bits in the low bits of each sample,
                // P010 in the high bits.
                let sample = |i: usize| u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]) << 6;
                for i in 0..luma {
                    put16(dst, i, sample(i));
                }
                let uv = &mut dst[uv_offset..];
                for i in 0..chroma {
                    put16(uv, i * 2, sample(luma + i));
                    put16(uv, i * 2 + 1, sample(luma + chroma + i));
                }
            }
        }

        unsafe { device.unmap_memory(self.staging_memory) };
        Ok(())
    }

    fn copy_staging_to_image(&mut self) -> Result<()> {
        // A frame submitted before may still be reading the image.
        self.source.wait_released()?;
        let old_layout = self.image_layout;
        self.copy_staging_to_image_internal(self.image, old_layout, vk::ImageLayout::GENERAL)?;
        self.image_layout = vk::ImageLayout::GENERAL;
        Ok(())
    }

    fn copy_staging_to_image_internal(
        &mut self,
        target_image: vk::Image,
        old_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) -> Result<()> {
        let device = self.context.device();
        let width = self.width;
        let height = self.height;

        // Calculate plane sizes based on bit depth.
        // For 8-bit: 1 byte per sample
        // For 10-bit: 2 bytes per sample (16-bit per sample)
        let bytes_per_sample = match self.bit_depth {
            BitDepth::Eight => 1,
            BitDepth::Ten => 2,
        };
        // For YUV444, align Y plane size to 4 bytes so the UV plane buffer offset
        // meets VkBufferImageCopy::bufferOffset alignment requirements.
        // YUV420 dimensions are always even (required for 4:2:0), so alignment is
        // naturally satisfied and must not pad to stay consistent with upload functions.
        let y_plane_bytes = (width * height) as usize * bytes_per_sample;
        let y_plane_size_bytes = match self.chroma_format {
            ChromaFormat::Yuv444 => crate::align4(y_plane_bytes) as vk::DeviceSize,
            _ => y_plane_bytes as vk::DeviceSize,
        };

        unsafe {
            device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
        }
        .map_err(PixelForgeError::vk("vkResetCommandBuffer"))?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(self.command_buffer, &begin_info) }
            .map_err(PixelForgeError::vk("vkBeginCommandBuffer"))?;

        // Transition image to transfer destination.
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(target_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

        unsafe {
            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        // Build copy regions based on chroma format.
        let copy_regions = match self.chroma_format {
            ChromaFormat::Yuv420 => {
                // NV12/P010: Y plane (PLANE_0) + interleaved UV plane (PLANE_1)
                vec![
                    // Y plane.
                    vk::BufferImageCopy::default()
                        .buffer_offset(0)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_0,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .image_extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        }),
                    // UV plane (interleaved, half resolution).
                    vk::BufferImageCopy::default()
                        .buffer_offset(y_plane_size_bytes)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_1,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .image_extent(vk::Extent3D {
                            width: width / 2,
                            height: height / 2,
                            depth: 1,
                        }),
                ]
            }
            ChromaFormat::Yuv444 => {
                // 2-plane semi-planar YUV444: Y (PLANE_0), UV interleaved (PLANE_1) at full resolution.
                vec![
                    // Y plane.
                    vk::BufferImageCopy::default()
                        .buffer_offset(0)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_0,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .image_extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        }),
                    // UV plane (interleaved, full resolution for 444).
                    vk::BufferImageCopy::default()
                        .buffer_offset(y_plane_size_bytes)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_1,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .image_extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        }),
                ]
            }
            _ => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "Unsupported chroma format: {:?}",
                    self.chroma_format
                )));
            }
        };

        unsafe {
            device.cmd_copy_buffer_to_image(
                self.command_buffer,
                self.staging_buffer,
                target_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy_regions,
            );
        }

        // Transition image to final layout (ready for its intended use)
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(final_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(target_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ);

        unsafe {
            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        unsafe { device.end_command_buffer(self.command_buffer) }
            .map_err(PixelForgeError::vk("vkEndCommandBuffer"))?;

        // Submit and wait.
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.command_buffer));

        unsafe { device.queue_submit(self.context.transfer_queue(), &[submit_info], self.fence) }
            .map_err(PixelForgeError::vk("vkQueueSubmit"))?;

        unsafe { device.wait_for_fences(&[self.fence], true, u64::MAX) }
            .map_err(PixelForgeError::vk("vkWaitForFences"))?;

        unsafe { device.reset_fences(&[self.fence]) }
            .map_err(PixelForgeError::vk("vkResetFences"))?;

        Ok(())
    }

    /// Get the underlying Vulkan image handle.
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// The image as a source to encode from, see
    /// [`Frame::image`](crate::Frame::image). The uploads wait until the
    /// frames that read it are done with it.
    pub fn source(&self) -> &SourceImage {
        &self.source
    }

    /// Get the image dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl Drop for YuvUploader {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe {
            // Wait for all GPU work to finish before destroying resources.
            let _ = device.device_wait_idle();
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_buffer(self.staging_buffer, None);
            device.free_memory(self.staging_memory, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}
