//! DMA-BUF input and sync_file fences.
//!
//! An encoder can read a DMA-BUF straight into its encode, with no copy in
//! pixelforge, when the device takes the buffer's DRM format modifier for that.
//! [`Encoder::input_formats`] says which modifiers those are, so a caller can
//! negotiate one with its producer. [`Encoder::allocate_input`] makes such a
//! buffer for a producer that writes into buffers the consumer gives it, and
//! [`Encoder::import`] takes one a producer made. Either gives an
//! [`ExternalImage`] to hand to [`Frame::external`](crate::Frame::external).
//!
//! A DMA-BUF is shared with another process, so each frame acquires it from
//! the foreign queue family before reading it and releases it back after.
//! Its producer's write fence and the encoder's read fence can travel as
//! sync_files, see [`Frame::wait_sync_file`](crate::Frame::wait_sync_file).

use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};

use ash::vk;
use ash::vk::TaggedStructure;

use super::api::Encoder;
use super::input::{FrameInput, FrameSync, SourceOwnership, Uses};
use super::source::{Conversion, Source};
use super::{EncodeFuture, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::sync::TimelinePoint;
use crate::video::{create_video_image_view, get_video_format};
use crate::vulkan::VideoContext;

/// A DRM fourcc code, as `drm_fourcc.h` builds them.
const fn fourcc(code: &[u8; 4]) -> u32 {
    (code[0] as u32) | (code[1] as u32) << 8 | (code[2] as u32) << 16 | (code[3] as u32) << 24
}

/// DRM fourcc codes for the formats an encoder reads.
pub mod drm_format {
    use super::fourcc;

    /// 8-bit 4:2:0, Y plane then interleaved CbCr.
    pub const NV12: u32 = fourcc(b"NV12");
    /// 10-bit 4:2:0 in 16-bit words, Y plane then interleaved CbCr.
    pub const P010: u32 = fourcc(b"P010");
    /// 8-bit 4:4:4, Y plane then interleaved CbCr.
    pub const NV24: u32 = fourcc(b"NV24");
    /// 32-bit, B G R X in memory.
    pub const XRGB8888: u32 = fourcc(b"XR24");
    /// 32-bit, B G R A in memory.
    pub const ARGB8888: u32 = fourcc(b"AR24");
    /// 32-bit, R G B X in memory.
    pub const XBGR8888: u32 = fourcc(b"XB24");
    /// 32-bit, R G B A in memory.
    pub const ABGR8888: u32 = fourcc(b"AB24");
    /// 10 bits per colour, 2 of alpha, red in the low bits.
    pub const ABGR2101010: u32 = fourcc(b"AB30");
    /// 10 bits per colour, red in the low bits.
    pub const XBGR2101010: u32 = fourcc(b"XB30");
    /// 16-bit float per channel, red first.
    pub const ABGR16161616F: u32 = fourcc(b"AB4H");
}

/// The Vulkan format a DRM fourcc holds, for the formats an encoder reads.
pub fn vk_format_of(drm: u32) -> Option<vk::Format> {
    use drm_format::*;
    Some(match drm {
        NV12 => vk::Format::G8_B8R8_2PLANE_420_UNORM,
        P010 => vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
        NV24 => vk::Format::G8_B8R8_2PLANE_444_UNORM,
        XRGB8888 | ARGB8888 => vk::Format::B8G8R8A8_UNORM,
        XBGR8888 | ABGR8888 => vk::Format::R8G8B8A8_UNORM,
        XBGR2101010 | ABGR2101010 => vk::Format::A2B10G10R10_UNORM_PACK32,
        ABGR16161616F => vk::Format::R16G16B16A16_SFLOAT,
        _ => return None,
    })
}

/// The DRM fourcc for a Vulkan format an encoder reads. For the RGB formats,
/// the one with alpha: a buffer without it is taken too.
fn drm_format_of(format: vk::Format) -> Option<u32> {
    use drm_format::*;
    Some(match format {
        vk::Format::G8_B8R8_2PLANE_420_UNORM => NV12,
        vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 => P010,
        vk::Format::G8_B8R8_2PLANE_444_UNORM => NV24,
        vk::Format::B8G8R8A8_UNORM => ARGB8888,
        vk::Format::R8G8B8A8_UNORM => ABGR8888,
        vk::Format::A2B10G10R10_UNORM_PACK32 => ABGR2101010,
        vk::Format::R16G16B16A16_SFLOAT => ABGR16161616F,
        _ => return None,
    })
}

/// One plane of a [`DmaBuf`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmaBufPlane {
    /// Where the plane starts in the buffer, in bytes.
    pub offset: u64,
    /// Bytes from one row to the next.
    pub stride: u64,
}

/// A DMA-BUF holding one picture: one buffer, every plane in it at its own
/// offset.
#[derive(Debug)]
pub struct DmaBuf {
    /// The buffer.
    pub fd: OwnedFd,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// DRM fourcc of the pixel format, see [`drm_format`].
    pub format: u32,
    /// DRM format modifier.
    pub modifier: u64,
    /// The memory planes, as many as the modifier has.
    pub planes: Vec<DmaBufPlane>,
}

/// What an encoder can do with buffers of one DRM format modifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputModifier {
    /// The DRM format modifier.
    pub modifier: u64,
    /// How many memory planes a buffer with it has.
    pub plane_count: u32,
    /// The encoder reads such a buffer with no copy in pixelforge: the
    /// encode itself, or the colour converter's shader where it converts.
    pub zero_copy: bool,
    /// [`Encoder::import`] takes such a buffer, reading it directly where
    /// `zero_copy` says so and copying it otherwise.
    pub importable: bool,
    /// [`Encoder::allocate_input`] can make such a buffer: read directly
    /// where `zero_copy` says so, copied otherwise.
    pub exportable: bool,
}

/// The buffers an encoder takes, from [`Encoder::input_formats`].
#[derive(Debug, Clone)]
pub struct InputFormats {
    /// The DRM fourcc of the pixel format, see [`drm_format`].
    pub format: u32,
    /// The same format, as Vulkan names it.
    pub vk_format: vk::Format,
    /// Every modifier the device knows for the format and the encoder can do
    /// something with. Offer the `zero_copy` ones to a producer first.
    pub modifiers: Vec<InputModifier>,
}

/// How an [`ExternalImage`] is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Access {
    /// By the encode, made with its video profile.
    Encode,
    /// By a copy into the encoder's input image.
    Copy,
    /// By the colour converter's shader.
    Sample,
}

/// Which modifier a new image gets.
enum ModifierChoice<'a> {
    /// The driver picks one of these, for an image to export.
    OneOf(&'a [u64]),
    /// This one, with these plane layouts, for an imported buffer.
    Exactly(u64, &'a [vk::SubresourceLayout]),
}

/// What the device says about one modifier for one use.
struct ModifierSupport {
    importable: bool,
    exportable: bool,
}

/// A DMA-BUF an [`Encoder`] can read, made by [`Encoder::import`] or
/// [`Encoder::allocate_input`]. Hand it to
/// [`Frame::external`](crate::Frame::external).
///
/// Only the encoder that made it can read it. Dropping it waits until its last
/// frame is done with it.
pub struct ExternalImage {
    context: VideoContext,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: Option<vk::ImageView>,
    access: Access,
    extent: vk::Extent2D,
    format: u32,
    modifier: u64,
    plane_count: u32,
    /// The encoder it belongs to, and its conversion's generation.
    owner: (u64, u64),
    /// The frames that read it.
    uses: Uses,
}

impl std::fmt::Debug for ExternalImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalImage")
            .field("extent", &self.extent)
            .field("format", &self.format)
            .field("modifier", &self.modifier)
            .field("access", &self.access)
            .finish()
    }
}

impl ExternalImage {
    /// Whether the encoder reads this image directly, with no copy.
    pub fn is_zero_copy(&self) -> bool {
        self.access != Access::Copy
    }

    /// Whether a frame from this image ends up in the encoder's input image,
    /// copied or converted, rather than read where it is.
    pub(crate) fn fills_input_image(&self) -> bool {
        self.access != Access::Encode
    }

    /// The points after which every frame submitted so far is done with the
    /// buffer, so its producer can write it again. Empty before the first
    /// frame. For a producer that takes a sync_file instead, see
    /// [`Frame::signal_sync_file`](crate::Frame::signal_sync_file).
    pub fn released(&self) -> Vec<TimelinePoint> {
        self.uses.points()
    }

    /// The DRM format modifier.
    pub fn modifier(&self) -> u64 {
        self.modifier
    }

    /// The image, for a caller that also uses it with Vulkan itself.
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// Export the image as a DMA-BUF, for a producer to write into. Each call
    /// gives a new fd for the same memory.
    pub fn export(&self) -> Result<DmaBuf> {
        let device = self.context.device();
        let memory_fd = ash::khr::external_memory_fd::Device::load(self.context.instance(), device);
        let info = vk::MemoryGetFdInfoKHR::default()
            .memory(self.memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let fd = unsafe { memory_fd.get_memory_fd(&info) }
            .map_err(|e| PixelForgeError::ResourceCreation(format!("DMA-BUF export: {e}")))?;
        // SAFETY: vkGetMemoryFdKHR hands over a new fd the caller owns.
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        let planes = (0..self.plane_count)
            .map(|plane| {
                let layout = unsafe {
                    device.get_image_subresource_layout(
                        self.image,
                        vk::ImageSubresource {
                            aspect_mask: memory_plane(plane),
                            mip_level: 0,
                            array_layer: 0,
                        },
                    )
                };
                DmaBufPlane {
                    offset: layout.offset,
                    stride: layout.row_pitch,
                }
            })
            .collect();
        Ok(DmaBuf {
            fd,
            width: self.extent.width,
            height: self.extent.height,
            format: self.format,
            modifier: self.modifier,
            planes,
        })
    }

    /// The ownership transfers of the next frame that reads this image.
    fn ownership(&self) -> SourceOwnership {
        let foreign = self.context.external().foreign_family();
        SourceOwnership {
            old_layout: if self.uses.was_read() {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::UNDEFINED
            },
            acquire_from: Some(foreign),
            release_to: Some(foreign),
            final_layout: vk::ImageLayout::GENERAL,
        }
    }
}

impl Drop for ExternalImage {
    fn drop(&mut self) {
        let _ = self.uses.wait(&self.context);
        let device = self.context.device();
        unsafe {
            if let Some(view) = self.view {
                device.destroy_image_view(view, None);
            }
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

fn memory_plane(plane: u32) -> vk::ImageAspectFlags {
    match plane {
        0 => vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
        1 => vk::ImageAspectFlags::MEMORY_PLANE_1_EXT,
        2 => vk::ImageAspectFlags::MEMORY_PLANE_2_EXT,
        _ => vk::ImageAspectFlags::MEMORY_PLANE_3_EXT,
    }
}

/// Whether a DMA-BUF with `modifier` is safe to use for `access` on this
/// driver, past what the driver itself reports.
///
/// NVIDIA's driver (seen on 615.78.08) reports more than works, in two ways,
/// both of which give a wrong picture rather than an error:
///
/// - An image made with a video profile reads a block-linear buffer as all
///   black, even one the encoder allocated and exported itself. Imported
///   without the profile, the same buffer reads correctly. So the encode never
///   reads a DMA-BUF directly there; it is copied.
/// - A two-plane YUV buffer with a block-linear modifier loses its chroma plane
///   between the image that wrote it and one that imports it, even with no
///   pixelforge code involved. LINEAR survives, so it is the only modifier
///   offered for YUV there. Single-plane RGB is not affected.
///
/// `PIXELFORGE_NO_DMABUF_QUIRKS=1` trusts the driver anyway, to try a newer
/// one.
fn quirks_allow(context: &VideoContext, format: vk::Format, modifier: u64, access: Access) -> bool {
    const NVIDIA: u32 = 0x10de;
    const LINEAR: u64 = 0;
    if context.device_properties().vendor_id != NVIDIA
        || std::env::var_os("PIXELFORGE_NO_DMABUF_QUIRKS").is_some_and(|v| v == "1")
    {
        return true;
    }
    let multi_planar = matches!(
        format,
        vk::Format::G8_B8R8_2PLANE_420_UNORM
            | vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
            | vk::Format::G8_B8R8_2PLANE_444_UNORM
    );
    access != Access::Encode && (!multi_planar || modifier == LINEAR)
}

/// What an encoder reads and how, worked out from its conversion.
struct InputPlan {
    vk_format: vk::Format,
    zero_copy: Access,
    /// A way to read buffers the zero-copy path does not take, if any.
    fallback: Option<Access>,
}

impl Encoder {
    fn input_plan(&self) -> Result<InputPlan> {
        let config = self.config();
        let rgb_format = match config.source {
            Source::Rgb(rgb) => Some(rgb.format.vk_format()),
            Source::Yuv => None,
        };
        Ok(match self.conversion() {
            Conversion::None => {
                if config.pixel_format == PixelFormat::Yuv422 {
                    return Err(PixelForgeError::InvalidInput(
                        "DMA-BUF input: no 4:2:2 encode".to_string(),
                    ));
                }
                InputPlan {
                    vk_format: get_video_format(config.pixel_format, config.bit_depth),
                    zero_copy: Access::Encode,
                    fallback: Some(Access::Copy),
                }
            }
            Conversion::Encoder => InputPlan {
                vk_format: rgb_format.expect("an encoder conversion has an RGB source"),
                zero_copy: Access::Encode,
                fallback: Some(Access::Copy),
            },
            Conversion::Shader => InputPlan {
                vk_format: rgb_format.expect("a shader conversion has an RGB source"),
                zero_copy: Access::Sample,
                fallback: None,
            },
        })
    }

    fn require_dmabuf(&self) -> Result<()> {
        if self.context().external().dmabuf {
            Ok(())
        } else {
            Err(PixelForgeError::NoSuitableDevice(
                "DMA-BUF input needs VK_EXT_image_drm_format_modifier, \
                 VK_EXT_external_memory_dma_buf and VK_KHR_external_memory_fd"
                    .to_string(),
            ))
        }
    }

    /// The image usage `access` needs, on an image to `export` or not.
    ///
    /// An image made to be exported also gets `TRANSFER_DST`, the usage a
    /// producer writing into it needs. A driver may lay a buffer out
    /// differently by usage, and then follow its own layout over the one an
    /// importer gives it: NVIDIA's (615) pads LINEAR rows for `TRANSFER_DST`
    /// and not for `TRANSFER_SRC` alone, and a producer importing the
    /// unpadded export then writes every row in the wrong place.
    fn usage(access: Access, export: bool) -> vk::ImageUsageFlags {
        let usage = match access {
            Access::Encode => vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR,
            Access::Copy => vk::ImageUsageFlags::TRANSFER_SRC,
            Access::Sample => vk::ImageUsageFlags::SAMPLED,
        };
        if export {
            usage | vk::ImageUsageFlags::TRANSFER_DST
        } else {
            usage
        }
    }

    /// What the device says about `modifier` for `access`, on an image to
    /// `export` or one imported, or `None` if it cannot make such an image at
    /// all.
    fn modifier_support(
        &self,
        format: vk::Format,
        modifier: u64,
        access: Access,
        export: bool,
    ) -> Option<ModifierSupport> {
        let context = self.context();
        if !quirks_allow(context, format, modifier, access) {
            return None;
        }
        let profile = self.inner().common().profile.info();
        let profiles = std::slice::from_ref(profile);
        let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(profiles);
        let mut external = vk::PhysicalDeviceExternalImageFormatInfo::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut drm = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
            .drm_format_modifier(modifier)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let mut info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(format)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(Self::usage(access, export))
            .push(&mut external)
            .push(&mut drm);
        if access == Access::Encode {
            info = info.push(&mut profile_list);
        }
        let mut external_props = vk::ExternalImageFormatProperties::default();
        let mut props = vk::ImageFormatProperties2::default().push(&mut external_props);
        unsafe {
            context
                .instance()
                .get_physical_device_image_format_properties2(
                    context.physical_device(),
                    &info,
                    &mut props,
                )
        }
        .ok()?;
        let max = props.image_format_properties.max_extent;
        let size = self.config().dimensions;
        if max.width < size.width || max.height < size.height {
            return None;
        }
        let features = external_props
            .external_memory_properties
            .external_memory_features;
        Some(ModifierSupport {
            importable: features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE),
            exportable: features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE),
        })
    }

    /// The modifiers the device knows for `format`, with their plane counts.
    fn known_modifiers(&self, format: vk::Format) -> Vec<(u64, u32)> {
        let context = self.context();
        let query = |list: &mut vk::DrmFormatModifierPropertiesListEXT| {
            let mut props = vk::FormatProperties2::default().push(list);
            unsafe {
                context.instance().get_physical_device_format_properties2(
                    context.physical_device(),
                    format,
                    &mut props,
                )
            };
        };
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
        query(&mut list);
        let mut entries = vec![
            vk::DrmFormatModifierPropertiesEXT::default();
            list.drm_format_modifier_count as usize
        ];
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default()
            .drm_format_modifier_properties(&mut entries);
        query(&mut list);
        let count = list.drm_format_modifier_count as usize;
        entries
            .iter()
            .take(count)
            .map(|e| (e.drm_format_modifier, e.drm_format_modifier_plane_count))
            .collect()
    }

    /// The DRM format this encoder takes as a DMA-BUF, and what it can do
    /// with each modifier the device knows for it.
    ///
    /// The format follows the [`Source`](crate::encoder::Source): the
    /// encoder's YUV for a YUV source, the RGB format for an RGB one. A
    /// producer that can choose should be offered the `zero_copy` modifiers
    /// first.
    pub fn input_formats(&self) -> Result<InputFormats> {
        self.require_dmabuf()?;
        let plan = self.input_plan()?;
        let format = drm_format_of(plan.vk_format).ok_or_else(|| {
            PixelForgeError::InvalidInput(format!(
                "DMA-BUF input: no DRM format for {:?}",
                plan.vk_format
            ))
        })?;
        let modifiers = self
            .known_modifiers(plan.vk_format)
            .into_iter()
            .filter_map(|(modifier, plane_count)| {
                let support = |access: Access, export: bool| {
                    self.modifier_support(plan.vk_format, modifier, access, export)
                };
                let imports = |access| support(access, false).is_some_and(|s| s.importable);
                let exports = |access| support(access, true).is_some_and(|s| s.exportable);
                let direct_import = imports(plan.zero_copy);
                let direct_export = exports(plan.zero_copy);
                let zero_copy = direct_import || direct_export;
                let importable = direct_import || plan.fallback.is_some_and(imports);
                let exportable = direct_export || plan.fallback.is_some_and(exports);
                (importable || exportable).then_some(InputModifier {
                    modifier,
                    plane_count,
                    zero_copy,
                    importable,
                    exportable,
                })
            })
            .collect();
        Ok(InputFormats {
            format,
            vk_format: plan.vk_format,
            modifiers,
        })
    }

    /// Make an image with one of `modifiers`, which a producer can write into
    /// once [exported](ExternalImage::export).
    ///
    /// For producers that write into buffers the consumer supplies, such as
    /// `ext-image-copy-capture`. The image has the encoder's display size,
    /// which is what such a protocol asks for. `modifiers` are the ones the
    /// producer takes. The driver picks among those the encoder reads directly
    /// (`zero_copy` in [`Self::input_formats`]); only if there are none, among
    /// those it can copy from, and then each frame is copied
    /// ([`ExternalImage::is_zero_copy`] says which). It fails if none of
    /// `modifiers` can be allocated either way.
    pub fn allocate_input(&self, modifiers: &[u64]) -> Result<ExternalImage> {
        self.require_dmabuf()?;
        let plan = self.input_plan()?;
        let format = drm_format_of(plan.vk_format).ok_or_else(|| {
            PixelForgeError::InvalidInput(format!(
                "DMA-BUF input: no DRM format for {:?}",
                plan.vk_format
            ))
        })?;
        let exportable_for = |access: Access| -> Vec<u64> {
            modifiers
                .iter()
                .copied()
                .filter(|&m| {
                    self.modifier_support(plan.vk_format, m, access, true)
                        .is_some_and(|s| s.exportable)
                })
                .collect()
        };
        let direct = exportable_for(plan.zero_copy);
        let (access, usable) = if !direct.is_empty() {
            (plan.zero_copy, direct)
        } else if let Some(fallback) = plan.fallback {
            (fallback, exportable_for(fallback))
        } else {
            (plan.zero_copy, Vec::new())
        };
        if usable.is_empty() {
            return Err(PixelForgeError::NoSuitableDevice(format!(
                "DMA-BUF input: none of the modifiers {modifiers:x?} can be allocated for this \
                 encoder"
            )));
        }
        self.make_external_image(
            plan.vk_format,
            format,
            access,
            ModifierChoice::OneOf(&usable),
            None,
        )
    }

    /// Import a producer's DMA-BUF for this encoder to read.
    ///
    /// The buffer is checked here, once: a modifier the encoder reads directly
    /// gives an image it does ([`ExternalImage::is_zero_copy`]); one it can only
    /// copy from gives an image each frame copies; any other fails. The format
    /// must be the one [`Self::input_formats`] names, or an RGB format that
    /// differs from it only in whether the fourth channel is alpha. The size
    /// must be at least the encoder's.
    pub fn import(&self, buffer: &DmaBuf) -> Result<ExternalImage> {
        self.require_dmabuf()?;
        let plan = self.input_plan()?;
        if vk_format_of(buffer.format) != Some(plan.vk_format) {
            return Err(PixelForgeError::InvalidInput(format!(
                "DMA-BUF import: format {:#x} is not {:?}, which this encoder reads",
                buffer.format, plan.vk_format
            )));
        }
        let size = self.config().dimensions;
        if buffer.width < size.width || buffer.height < size.height {
            return Err(PixelForgeError::InvalidInput(format!(
                "DMA-BUF import: {}x{} is smaller than the encoder's {}x{}",
                buffer.width, buffer.height, size.width, size.height
            )));
        }
        let can_import = |access: Access| {
            self.modifier_support(plan.vk_format, buffer.modifier, access, false)
                .is_some_and(|s| s.importable)
        };
        let access = if can_import(plan.zero_copy) {
            plan.zero_copy
        } else if let Some(fallback) = plan.fallback.filter(|&a| can_import(a)) {
            fallback
        } else {
            return Err(PixelForgeError::NoSuitableDevice(format!(
                "DMA-BUF import: the encoder can neither read nor copy modifier {:#x}",
                buffer.modifier
            )));
        };
        let layouts: Vec<vk::SubresourceLayout> = buffer
            .planes
            .iter()
            .map(|p| vk::SubresourceLayout {
                offset: p.offset,
                size: 0,
                row_pitch: p.stride,
                array_pitch: 0,
                depth_pitch: 0,
            })
            .collect();
        self.make_external_image(
            plan.vk_format,
            buffer.format,
            access,
            ModifierChoice::Exactly(buffer.modifier, &layouts),
            Some(buffer),
        )
    }

    /// Create the image and its memory: imported from `import`, or allocated
    /// for export without it.
    fn make_external_image(
        &self,
        vk_format: vk::Format,
        drm: u32,
        access: Access,
        modifier: ModifierChoice<'_>,
        import: Option<&DmaBuf>,
    ) -> Result<ExternalImage> {
        let context = self.context();
        let device = context.device();
        let extent = match import {
            Some(buffer) => vk::Extent2D {
                width: buffer.width,
                height: buffer.height,
            },
            None => vk::Extent2D {
                width: self.config().dimensions.width,
                height: self.config().dimensions.height,
            },
        };
        let usage = Self::usage(access, import.is_none());
        let profile = self.inner().common().profile.info();
        let profiles = std::slice::from_ref(profile);
        let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(profiles);
        let mut external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let (mut list, mut explicit) = match modifier {
            ModifierChoice::OneOf(modifiers) => (
                Some(
                    vk::ImageDrmFormatModifierListCreateInfoEXT::default()
                        .drm_format_modifiers(modifiers),
                ),
                None,
            ),
            ModifierChoice::Exactly(modifier, layouts) => (
                None,
                Some(
                    vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
                        .drm_format_modifier(modifier)
                        .plane_layouts(layouts),
                ),
            ),
        };
        let mut info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push(&mut external);
        if let Some(list) = list.as_mut() {
            info = info.push(list);
        }
        if let Some(explicit) = explicit.as_mut() {
            info = info.push(explicit);
        }
        if access == Access::Encode {
            info = info.push(&mut profile_list);
        }
        let image = unsafe { device.create_image(&info, None) }
            .map_err(|e| PixelForgeError::ResourceCreation(format!("DMA-BUF image: {e}")))?;

        let memory = match self.bind_external_memory(image, import) {
            Ok(memory) => memory,
            Err(e) => {
                unsafe { device.destroy_image(image, None) };
                return Err(e);
            }
        };

        let modifier_fn =
            ash::ext::image_drm_format_modifier::Device::load(context.instance(), device);
        let mut modifier_props = vk::ImageDrmFormatModifierPropertiesEXT::default();
        let modifier = unsafe {
            modifier_fn.get_image_drm_format_modifier_properties(image, &mut modifier_props)
        }
        .map(|()| modifier_props.drm_format_modifier)
        .unwrap_or_else(|_| import.map_or(0, |b| b.modifier));
        let plane_count = self
            .known_modifiers(vk_format)
            .into_iter()
            .find(|&(m, _)| m == modifier)
            .map_or(1, |(_, planes)| planes);

        let view = if access == Access::Encode {
            match create_video_image_view(context, image, vk_format, 0, usage) {
                Ok(view) => Some(view),
                Err(e) => {
                    unsafe {
                        device.destroy_image(image, None);
                        device.free_memory(memory, None);
                    }
                    return Err(e);
                }
            }
        } else {
            None
        };

        tracing::debug!(
            "DMA-BUF {}: {}x{} {vk_format:?}, modifier {modifier:#x}, {plane_count} planes, \
             read by {access:?}",
            if import.is_some() {
                "imported"
            } else {
                "allocated"
            },
            extent.width,
            extent.height,
        );
        Ok(ExternalImage {
            context: context.clone(),
            image,
            memory,
            view,
            access,
            extent,
            format: drm,
            modifier,
            plane_count,
            owner: (self.id(), self.generation()),
            uses: Uses::default(),
        })
    }

    /// Give `image` dedicated memory, imported from `import`'s fd or
    /// allocated to be exported.
    fn bind_external_memory(
        &self,
        image: vk::Image,
        import: Option<&DmaBuf>,
    ) -> Result<vk::DeviceMemory> {
        let context = self.context();
        let device = context.device();
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let handle = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;

        let memory = match import {
            Some(buffer) => {
                let memory_fd =
                    ash::khr::external_memory_fd::Device::load(context.instance(), device);
                let mut fd_props = vk::MemoryFdPropertiesKHR::default();
                unsafe {
                    memory_fd.get_memory_fd_properties(handle, buffer.fd.as_raw_fd(), &mut fd_props)
                }
                .map_err(|e| PixelForgeError::ResourceCreation(format!("DMA-BUF fd: {e}")))?;
                let type_index = context
                    .find_memory_type(
                        requirements.memory_type_bits & fd_props.memory_type_bits,
                        vk::MemoryPropertyFlags::empty(),
                    )
                    .ok_or_else(|| {
                        PixelForgeError::ResourceCreation(
                            "DMA-BUF import: no memory type fits".to_string(),
                        )
                    })?;
                // Vulkan owns the fd once the import is attempted. The spec
                // gives it back on failure, but at least one driver (NVIDIA
                // 615) closes it anyway, so it is never closed here: a leaked
                // fd on a failed import is better than closing whatever file
                // took its number.
                let fd = buffer
                    .fd
                    .try_clone()
                    .map_err(|e| PixelForgeError::ResourceCreation(format!("DMA-BUF fd: {e}")))?;
                let raw = fd.into_raw_fd();
                let mut import_info = vk::ImportMemoryFdInfoKHR::default()
                    .handle_type(handle)
                    .fd(raw);
                let info = vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(type_index)
                    .push(&mut import_info)
                    .push(&mut dedicated);
                unsafe { device.allocate_memory(&info, None) }.map_err(|e| {
                    PixelForgeError::ResourceCreation(format!("DMA-BUF import: {e}"))
                })?
            }
            None => {
                let type_index = context
                    .find_memory_type(
                        requirements.memory_type_bits,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    )
                    .or_else(|| {
                        context.find_memory_type(
                            requirements.memory_type_bits,
                            vk::MemoryPropertyFlags::empty(),
                        )
                    })
                    .ok_or_else(|| {
                        PixelForgeError::ResourceCreation(
                            "DMA-BUF allocation: no memory type fits".to_string(),
                        )
                    })?;
                let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(handle);
                let info = vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(type_index)
                    .push(&mut export)
                    .push(&mut dedicated);
                unsafe { device.allocate_memory(&info, None) }.map_err(|e| {
                    PixelForgeError::ResourceCreation(format!("DMA-BUF allocation: {e}"))
                })?
            }
        };
        if let Err(e) = unsafe { device.bind_image_memory(image, memory, 0) } {
            unsafe { device.free_memory(memory, None) };
            return Err(PixelForgeError::ResourceCreation(format!(
                "DMA-BUF bind: {e}"
            )));
        }
        Ok(memory)
    }

    /// Encode from an [`ExternalImage`] this encoder made.
    pub(crate) fn submit_external(
        &mut self,
        external: &ExternalImage,
        mut sync: FrameSync,
    ) -> Result<EncodeFuture> {
        if external.owner != (self.id(), self.generation()) {
            return Err(PixelForgeError::InvalidInput(
                "this ExternalImage was made by another encoder, or before a reconfigure \
                 changed the source or the conversion; import or allocate it again"
                    .to_string(),
            ));
        }
        // The previous frame's release has to come before this frame's
        // acquire, even when both run on the same queue.
        sync.waits
            .extend(external.uses.points().iter().map(TimelinePoint::wait_info));
        let ownership = external.ownership();
        let future = match external.access {
            Access::Encode => self.encode_input(
                FrameInput::Direct {
                    image: external.image,
                    view: external.view.expect("an encode-read image has a view"),
                    extent: external.extent,
                    ownership,
                },
                sync,
            )?,
            Access::Copy => self.encode_input(
                FrameInput::Copy {
                    image: external.image,
                    ownership,
                },
                sync,
            )?,
            Access::Sample => self.convert_and_encode(external.image, ownership, sync)?,
        };
        external.uses.record(future.source_released());
        Ok(future)
    }
}

/// Per-slot binary semaphores for sync_file fences.
///
/// A slot's semaphores are used again only once the slot is free, which is
/// after every submission of its last frame has completed.
pub(crate) struct SyncFiles {
    context: VideoContext,
    fd_fn: ash::khr::external_semaphore_fd::Device,
    acquire: Vec<vk::Semaphore>,
    release: Vec<vk::Semaphore>,
}

impl SyncFiles {
    pub(crate) fn new(context: &VideoContext) -> Result<Self> {
        if !context.external().sync_file {
            return Err(PixelForgeError::NoSuitableDevice(
                "sync_file fences need VK_KHR_external_semaphore_fd".to_string(),
            ));
        }
        let device = context.device();
        let slots = crate::encoder::pipeline::ENCODE_PIPELINE_DEPTH;
        let mut files = Self {
            context: context.clone(),
            fd_fn: ash::khr::external_semaphore_fd::Device::load(context.instance(), device),
            acquire: Vec::with_capacity(slots),
            release: Vec::with_capacity(slots),
        };
        for _ in 0..slots {
            let semaphore =
                unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                    .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
            files.acquire.push(semaphore);
            let mut export = vk::ExportSemaphoreCreateInfo::default()
                .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
            let semaphore = unsafe {
                device.create_semaphore(&vk::SemaphoreCreateInfo::default().push(&mut export), None)
            }
            .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
            files.release.push(semaphore);
        }
        Ok(files)
    }

    /// Put `fence` into the slot's acquire semaphore, and return the wait on
    /// it.
    pub(crate) fn import_wait(
        &mut self,
        slot: usize,
        fence: OwnedFd,
    ) -> Result<vk::SemaphoreSubmitInfo<'static>> {
        let semaphore = self.acquire[slot];
        let raw = fence.into_raw_fd();
        let info = vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(semaphore)
            .flags(vk::SemaphoreImportFlags::TEMPORARY)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
            .fd(raw);
        // Never closed here on failure, as for memory imports.
        if let Err(e) = unsafe { self.fd_fn.import_semaphore_fd(&info) } {
            return Err(PixelForgeError::Synchronization(format!(
                "sync_file import: {e}"
            )));
        }
        Ok(vk::SemaphoreSubmitInfo::default()
            .semaphore(semaphore)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS))
    }

    /// The signal of the slot's release semaphore.
    pub(crate) fn release_signal(&self, slot: usize) -> vk::SemaphoreSubmitInfo<'static> {
        vk::SemaphoreSubmitInfo::default()
            .semaphore(self.release[slot])
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
    }

    /// The slot's release semaphore as a sync_file, once a submission that
    /// signals it was made. `None` when it has already signalled, which the
    /// driver may report as fd -1.
    pub(crate) fn export_release(&self, slot: usize) -> Result<Option<OwnedFd>> {
        let info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(self.release[slot])
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        let fd = unsafe { self.fd_fn.get_semaphore_fd(&info) }
            .map_err(|e| PixelForgeError::Synchronization(format!("sync_file export: {e}")))?;
        if fd < 0 {
            return Ok(None);
        }
        // SAFETY: vkGetSemaphoreFdKHR hands over a new fd the caller owns.
        Ok(Some(unsafe { OwnedFd::from_raw_fd(fd) }))
    }
}

impl Drop for SyncFiles {
    fn drop(&mut self) {
        // A submission may still wait on or signal these.
        for &semaphore in self.acquire.iter().chain(&self.release) {
            self.context.retire_semaphore(semaphore);
        }
    }
}
