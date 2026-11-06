//! Planning a decode session: from what a stream needs and what the device
//! offers for its profile to the numbers a session is built from.
//!
//! The codec reads its parameter sets and fills a [`StreamNeeds`]. Everything
//! after that (capabilities, DPB slots, picture format, image flags) is the
//! same for every codec and lives here.

use ash::vk;
use ash::vk::TaggedStructure;
use tracing::debug;

use crate::decoder::common::DecoderCommon;
use crate::encoder::{BitDepth, ChromaFormat};
use crate::error::{PixelForgeError, Result};
use crate::video::{align_up, get_video_format, query_supported_video_formats};

/// What a stream asks of a decode session, as the codec reads it from the
/// stream's own parameter sets.
pub(crate) struct StreamNeeds {
    /// The codec's name, for messages: "H.264".
    pub codec: &'static str,
    /// Coded dimensions before alignment to the device's picture granularity.
    pub width: u32,
    pub height: u32,
    pub chroma_format: ChromaFormat,
    pub bit_depth: BitDepth,
    /// Reference pictures the stream keeps in its DPB at once.
    pub references: u32,
    /// Reference pictures a single picture may use. Never more than
    /// `references`.
    pub active_references: u32,
    /// The most DPB slots the codec can address.
    pub slot_limit: u32,
    /// Pictures the stream may hold back for display-order reordering.
    pub reorder_depth: usize,
    /// Frames the caller may hold at once (`DecodeConfig::output_depth`).
    pub output_depth: usize,
    /// The stream applies film grain (AV1), which needs an output image apart
    /// from the DPB slot for every slot.
    pub film_grain: bool,
}

/// What a codec decided the session should look like.
///
/// [`DecoderCommon::plan_session`] derives these from the stream's needs and
/// the device caps; [`DecoderCommon::create_session`] turns them into Vulkan
/// objects.
pub(crate) struct SessionPlan {
    /// The codec's name, for messages.
    pub codec: &'static str,
    /// Coded dimensions, already aligned to the device's picture granularity.
    pub coded_width: u32,
    pub coded_height: u32,
    pub picture_format: vk::Format,
    pub bit_depth: BitDepth,
    pub chroma_format: ChromaFormat,
    /// DPB slots: references, the picture being decoded, and `spare_slots`.
    pub slot_count: usize,
    /// How many of `slot_count` are spare capacity for pictures pinned past the
    /// decode that produced them: those held for display-order reordering, and
    /// those handed to the caller. Zero means every emitted picture has to be
    /// copied out instead of pinned in place.
    pub spare_slots: usize,
    /// References the driver may use for a single picture.
    pub max_active_references: u32,
    /// Driver decodes straight into the DPB image (`DPB_AND_OUTPUT_COINCIDE`).
    pub coincide: bool,
    /// A decoded picture may be handed to the caller in place, pinned in its
    /// DPB slot, rather than copied out. See
    /// [`DecodeSession::pinnable`](crate::decoder::common::DecodeSession::pinnable).
    pub pinnable: bool,
    /// `dpb_usage` includes `SAMPLED`, so handed-out frames can be sampled.
    pub sampleable: bool,
    /// Picture images are created with `MUTABLE_FORMAT`, so a consumer can view
    /// their planes separately. See
    /// [`DecodeSession::plane_views`](crate::decoder::common::DecodeSession::plane_views).
    pub plane_views: bool,
    /// Creation flags for picture images, from what the driver reported.
    pub image_flags: vk::ImageCreateFlags,
    /// Formats views of a picture image may use. Empty unless `plane_views`.
    pub view_formats: Vec<vk::Format>,
    /// One layered DPB image rather than one image per slot.
    pub use_layered_dpb: bool,
    pub dpb_usage: vk::ImageUsageFlags,
    /// Create a film grain output image for every DPB slot.
    pub film_grain: bool,
    /// The Std header version the device reports for this profile.
    pub std_header_version: vk::ExtensionProperties,
    /// Bitstream buffer alignments the device requires, each at least one.
    pub bitstream_offset_alignment: u64,
    pub bitstream_size_alignment: u64,
}

/// The fields of `VkVideoCapabilitiesKHR` a plan uses, copied out of the
/// query's pointer chain.
struct Caps {
    flags: vk::VideoCapabilityFlagsKHR,
    min_bitstream_buffer_offset_alignment: u64,
    min_bitstream_buffer_size_alignment: u64,
    picture_access_granularity: vk::Extent2D,
    max_coded_extent: vk::Extent2D,
    max_dpb_slots: u32,
    max_active_reference_pictures: u32,
    std_header_version: vk::ExtensionProperties,
}

impl DecoderCommon {
    /// Plan a session for a stream with `needs`, decoded with `profile_info`.
    ///
    /// `codec_caps` is the codec's capability struct (for example
    /// `VkVideoDecodeH264CapabilitiesKHR`), which the query must chain even
    /// when the codec reads nothing from it.
    pub fn plan_session<'a, C>(
        &self,
        profile_info: &vk::VideoProfileInfoKHR,
        codec_caps: &'a mut C,
        needs: &StreamNeeds,
    ) -> Result<SessionPlan>
    where
        C: vk::Extends<vk::VideoCapabilitiesKHR<'a>> + TaggedStructure<'a>,
    {
        let codec = needs.codec;

        // Query capabilities for this exact profile. The chain borrows the
        // caps structs, so scope it and copy the results out.
        let mut decode_caps = vk::VideoDecodeCapabilitiesKHR::default();
        let caps = {
            let mut chain = vk::VideoCapabilitiesKHR::default()
                .push(codec_caps)
                .push(&mut decode_caps);
            query_decode_caps(self, profile_info, &mut chain)?;
            Caps {
                flags: chain.flags,
                min_bitstream_buffer_offset_alignment: chain.min_bitstream_buffer_offset_alignment,
                min_bitstream_buffer_size_alignment: chain.min_bitstream_buffer_size_alignment,
                picture_access_granularity: chain.picture_access_granularity,
                max_coded_extent: chain.max_coded_extent,
                max_dpb_slots: chain.max_dpb_slots,
                max_active_reference_pictures: chain.max_active_reference_pictures,
                std_header_version: chain.std_header_version,
            }
        };
        let decode_flags = decode_caps.flags;

        let granularity = caps.picture_access_granularity;
        let coded_width = align_up(needs.width, granularity.width.max(1));
        let coded_height = align_up(needs.height, granularity.height.max(1));
        let max_extent = caps.max_coded_extent;
        if coded_width > max_extent.width || coded_height > max_extent.height {
            return Err(PixelForgeError::InvalidInput(format!(
                "{codec} decode: stream is {}x{}, device maximum is {}x{}",
                coded_width, coded_height, max_extent.width, max_extent.height
            )));
        }

        let coincide =
            decode_flags.contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE);
        debug!("{codec} decode capability flags: {:?}", decode_flags);
        let use_layered_dpb = !caps
            .flags
            .contains(vk::VideoCapabilityFlagsKHR::SEPARATE_REFERENCE_IMAGES);

        // The stream needs its references plus the current picture. On top of
        // that come the spare slots, which is what lets a picture stay pinned
        // in place past the decode that produced it instead of being copied
        // out: `reorder_depth` of them for pictures waiting their turn in
        // display order, and `output_depth` for frames the caller is holding.
        // The device caps bound the total.
        let required = (needs.references + 1).max(2);
        let slot_limit = caps.max_dpb_slots.min(needs.slot_limit);
        if required > slot_limit {
            return Err(PixelForgeError::InvalidInput(format!(
                "{codec} decode: stream needs {} DPB slots, device supports {}",
                required, caps.max_dpb_slots
            )));
        }
        // Spare slots only buy something when pictures can be pinned. Without
        // that they are handed out as copies, which needs no DPB slot at all,
        // so asking for spares would just reserve memory nothing uses.
        let pinnable = coincide && self.context.has_unified_image_layouts();
        let wanted_spare = if pinnable {
            needs.reorder_depth + needs.output_depth
        } else {
            0
        };
        let slot_count = (required + wanted_spare as u32).min(slot_limit) as usize;
        let spare_slots = slot_count - required as usize;
        if spare_slots < wanted_spare {
            debug!(
                "{codec} decode: {} of {} spare DPB slots available \
                 (reorder depth {} plus {} output slots; the stream's references \
                 need {} of the device's {}); pictures beyond that are copied",
                spare_slots,
                wanted_spare,
                needs.reorder_depth,
                needs.output_depth,
                required,
                caps.max_dpb_slots
            );
        }

        // Only the reference pictures are ever active at once: the current
        // picture and the output reservation are not references.
        let active_references = needs.active_references.max(1).min(required - 1);
        if active_references > caps.max_active_reference_pictures {
            return Err(PixelForgeError::InvalidInput(format!(
                "{codec} decode: stream needs {} active reference pictures, device supports {}",
                active_references, caps.max_active_reference_pictures
            )));
        }

        // Pick a picture format the device actually supports for decode.
        //
        // Ask for SAMPLED as well: a coincident DPB image is what the caller
        // receives, and SAMPLED is what lets them read it in a shader instead
        // of copying it out first. Fall back to the plain usage if the device
        // reports no sampleable format, which costs the caller a copy but keeps
        // decoding working. With a distinct output image nothing asks, since
        // those pictures reach the caller as pool copies, which are sampleable
        // whatever the device says about video images.
        let base_usage = if coincide {
            vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
                | vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                | vk::ImageUsageFlags::TRANSFER_SRC
        } else {
            vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
        };
        let sampled_usage = base_usage | vk::ImageUsageFlags::SAMPLED;
        let sampled_formats = if coincide {
            query_supported_video_formats(&self.context, profile_info, sampled_usage)
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let (dpb_usage, sampleable, supported) = if sampled_formats.is_empty() {
            let formats = query_supported_video_formats(&self.context, profile_info, base_usage)?;
            (base_usage, false, formats)
        } else {
            (sampled_usage, true, sampled_formats)
        };
        let preferred = get_video_format(needs.chroma_format, needs.bit_depth);
        let chosen = supported
            .iter()
            .find(|f| f.format == preferred)
            .or_else(|| supported.first())
            .copied()
            .ok_or_else(|| {
                PixelForgeError::NoSuitableDevice(format!(
                    "{codec} decode: no supported picture format for this profile"
                ))
            })?;
        let picture_format = chosen.format;

        // MUTABLE_FORMAT lets a consumer view the picture's planes separately,
        // which is the only way to read a multi-planar image without a
        // sampler-YCbCr conversion, and the only way at all for a renderer
        // whose shaders come from a toolchain that cannot express one (naga and
        // therefore wgpu, among others). It is opportunistic: the driver
        // enumerates which creation flags it accepts for this profile, format
        // and usage, and DPB-only images on both RADV and ANV accept none.
        //
        // Naming the plane formats keeps a driver from having to assume the
        // image might be reinterpreted as anything, which is what would
        // otherwise cost it compression.
        let plane_views = chosen
            .create_flags
            .contains(vk::ImageCreateFlags::MUTABLE_FORMAT);
        let image_flags = if plane_views {
            vk::ImageCreateFlags::MUTABLE_FORMAT
        } else {
            vk::ImageCreateFlags::empty()
        };
        let view_formats: Vec<vk::Format> = if plane_views {
            let mut list = vec![picture_format];
            list.extend(crate::video::plane_view_formats(picture_format));
            list
        } else {
            Vec::new()
        };

        Ok(SessionPlan {
            codec,
            coded_width,
            coded_height,
            picture_format,
            bit_depth: needs.bit_depth,
            chroma_format: needs.chroma_format,
            slot_count,
            spare_slots,
            max_active_references: active_references,
            coincide,
            pinnable,
            sampleable,
            plane_views,
            image_flags,
            view_formats,
            use_layered_dpb,
            dpb_usage,
            film_grain: needs.film_grain,
            std_header_version: caps.std_header_version,
            bitstream_offset_alignment: caps.min_bitstream_buffer_offset_alignment.max(1),
            bitstream_size_alignment: caps.min_bitstream_buffer_size_alignment.max(1),
        })
    }
}

/// Query decode capabilities for a profile.
///
/// `caps` is supplied by the caller already chained with the codec-specific
/// capability struct, since only the codec knows which one applies.
fn query_decode_caps(
    common: &DecoderCommon,
    profile_info: &vk::VideoProfileInfoKHR,
    caps: &mut vk::VideoCapabilitiesKHR,
) -> Result<()> {
    let context = &common.context;
    let video_queue_instance =
        ash::khr::video_queue::Instance::load(context.entry(), context.instance());
    let result = unsafe {
        (video_queue_instance
            .fp()
            .get_physical_device_video_capabilities_khr)(
            context.physical_device(),
            profile_info,
            caps,
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::NoSuitableDevice(format!(
            "decode profile not supported: {:?}",
            result
        )));
    }
    Ok(())
}
