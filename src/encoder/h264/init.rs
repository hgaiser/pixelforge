use super::{H264Encoder, MB_SIZE};

use crate::encoder::dpb::{DecodedPictureBuffer, DecodedPictureBufferTrait, DpbConfig};
use crate::encoder::gop::GopStructure;
use crate::encoder::resources::{
    align_up, allocate_session_memory, clear_input_image, create_bitstream_buffer,
    create_command_resources, create_dpb_images, create_image, get_video_format, lcm,
    map_bitstream_buffer, query_supported_video_formats, ClearImageParams,
    MIN_BITSTREAM_BUFFER_SIZE,
};
use crate::encoder::ColorDescription;
use crate::encoder::PixelFormat;
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;
use ash::vk;
use std::ptr;
use tracing::{debug, info};

impl H264Encoder {
    /// Create a new H.264 encoder.
    pub fn new(context: VideoContext, config: crate::encoder::EncodeConfig) -> Result<Self> {
        // B-frames are not yet supported.
        if config.b_frame_count > 0 {
            panic!(
                "B-frame encoding is not yet supported. Set b_frame_count=0 in encoder config. \
                 Got b_frame_count={}",
                config.b_frame_count
            );
        }

        let width = config.dimensions.width;
        let height = config.dimensions.height;

        info!(
            "Creating H.264 encoder: requested {}x{}, pixel_format={:?}",
            width, height, config.pixel_format
        );

        // Load video queue extension functions.
        let video_queue_fn =
            ash::khr::video_queue::Device::load(context.instance(), context.device());
        let video_encode_fn =
            ash::khr::video_encode_queue::Device::load(context.instance(), context.device());

        // Get chroma subsampling from pixel format via `From` impl.
        let chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR = config.pixel_format.into();

        let luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR = config.bit_depth.into();
        let chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR = config.bit_depth.into();

        // Select H.264 profile based on pixel format.
        // - High profile for YUV420
        // - High 4:4:4 Predictive profile for YUV444
        let profile_idc = match config.pixel_format {
            PixelFormat::Yuv444 => {
                ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE
            }
            _ => ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH,
        };

        // Preferred input format based on pixel format and bit depth.
        // Note: the DPB format may differ and must be queried separately.
        let preferred_src_format = get_video_format(config.pixel_format, config.bit_depth);

        // Create H.264 encode profile.
        let mut h264_profile_info =
            vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(profile_idc);

        let mut profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(luma_bit_depth)
            .chroma_bit_depth(chroma_bit_depth);
        profile_info.p_next =
            (&mut h264_profile_info as *mut vk::VideoEncodeH264ProfileInfoKHR).cast();

        // Query encode capabilities for the selected profile and use them to derive a safe
        // coded extent and DPB limits. This mirrors vk_video_samples and avoids device loss
        // when the implementation requires larger picture access granularity (commonly for 4:4:4).
        let video_queue_instance =
            ash::khr::video_queue::Instance::load(context.entry(), context.instance());
        let mut h264_capabilities = vk::VideoEncodeH264CapabilitiesKHR::default();
        let mut encode_capabilities = vk::VideoEncodeCapabilitiesKHR {
            p_next: (&mut h264_capabilities as *mut vk::VideoEncodeH264CapabilitiesKHR).cast(),
            ..Default::default()
        };
        let mut capabilities = vk::VideoCapabilitiesKHR {
            p_next: (&mut encode_capabilities as *mut vk::VideoEncodeCapabilitiesKHR).cast(),
            ..Default::default()
        };

        let result = unsafe {
            (video_queue_instance
                .fp()
                .get_physical_device_video_capabilities_khr)(
                context.physical_device(),
                &profile_info,
                &mut capabilities,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(PixelForgeError::NoSuitableDevice(format!(
                "Failed to query Vulkan Video encode capabilities for requested H.264 profile: {:?}",
                result
            )));
        }

        debug!(
            "H.264 capabilities: maxLevelIdc={}, maxSliceCount={}, maxPPictureL0ReferenceCount={}, maxBPictureL0ReferenceCount={}, maxL1ReferenceCount={}, maxTemporalLayerCount={}, prefersGopRemainingFrames={}, requiresGopRemainingFrames={}, stdSyntaxFlags={:#010x}",
            h264_capabilities.max_level_idc,
            h264_capabilities.max_slice_count,
            h264_capabilities.max_p_picture_l0_reference_count,
            h264_capabilities.max_b_picture_l0_reference_count,
            h264_capabilities.max_l1_reference_count,
            h264_capabilities.max_temporal_layer_count,
            h264_capabilities.prefers_gop_remaining_frames,
            h264_capabilities.requires_gop_remaining_frames,
            h264_capabilities.std_syntax_flags.as_raw(),
        );
        debug!(
            "Encode capabilities: encodeInputPictureGranularity={}x{}, supportedEncodeFeedbackFlags={:#010x}, maxQualityLevels={}",
            encode_capabilities.encode_input_picture_granularity.width,
            encode_capabilities.encode_input_picture_granularity.height,
            encode_capabilities.supported_encode_feedback_flags.as_raw(),
            encode_capabilities.max_quality_levels,
        );
        debug!(
            "Video capabilities: flags={:#010x}, minBitstreamBufferOffsetAlignment={}, minBitstreamBufferSizeAlignment={}, minCodedExtent={}x{}, maxCodedExtent={}x{}, maxDpbSlots={}, maxActiveReferencePictures={}, pictureAccessGranularity={}x{}",
            capabilities.flags.as_raw(),
            capabilities.min_bitstream_buffer_offset_alignment,
            capabilities.min_bitstream_buffer_size_alignment,
            capabilities.min_coded_extent.width,
            capabilities.min_coded_extent.height,
            capabilities.max_coded_extent.width,
            capabilities.max_coded_extent.height,
            capabilities.max_dpb_slots,
            capabilities.max_active_reference_pictures,
            capabilities.picture_access_granularity.width,
            capabilities.picture_access_granularity.height,
        );

        // Query quality level properties to get the driver's preferred settings.
        let video_encode_instance =
            ash::khr::video_encode_queue::Instance::load(context.entry(), context.instance());
        let mut h264_quality_level_properties =
            vk::VideoEncodeH264QualityLevelPropertiesKHR::default();
        let mut quality_level_properties = vk::VideoEncodeQualityLevelPropertiesKHR {
            p_next: (&mut h264_quality_level_properties
                as *mut vk::VideoEncodeH264QualityLevelPropertiesKHR)
                .cast(),
            ..Default::default()
        };
        let quality_level_info = vk::PhysicalDeviceVideoEncodeQualityLevelInfoKHR::default()
            .video_profile(&profile_info)
            .quality_level(0);
        let ql_result = unsafe {
            (video_encode_instance
                .fp()
                .get_physical_device_video_encode_quality_level_properties_khr)(
                context.physical_device(),
                &quality_level_info,
                &mut quality_level_properties,
            )
        };
        let preferred_entropy_cabac = if ql_result == vk::Result::SUCCESS {
            debug!(
                "H.264 quality level 0: preferredStdEntropyCodingModeFlag={}, preferredMaxL0ReferenceCount={}, preferredMaxL1ReferenceCount={}",
                h264_quality_level_properties.preferred_std_entropy_coding_mode_flag,
                h264_quality_level_properties.preferred_max_l0_reference_count,
                h264_quality_level_properties.preferred_max_l1_reference_count,
            );
            h264_quality_level_properties.preferred_std_entropy_coding_mode_flag != 0
        } else {
            debug!(
                "Failed to query quality level properties: {:?}, defaulting to CABAC",
                ql_result
            );
            true
        };

        let gran_w = capabilities.picture_access_granularity.width.max(1);
        let gran_h = capabilities.picture_access_granularity.height.max(1);
        let align_w = lcm(MB_SIZE, gran_w);
        let align_h = lcm(MB_SIZE, gran_h);

        let mut aligned_width = align_up(width, align_w);
        let mut aligned_height = align_up(height, align_h);

        aligned_width = align_up(
            aligned_width.max(capabilities.min_coded_extent.width),
            align_w,
        );
        aligned_height = align_up(
            aligned_height.max(capabilities.min_coded_extent.height),
            align_h,
        );

        if aligned_width > capabilities.max_coded_extent.width
            || aligned_height > capabilities.max_coded_extent.height
        {
            return Err(PixelForgeError::InvalidInput(format!(
                "Requested coded extent {}x{} (aligned to {}x{} with granularity {}x{}) exceeds device max {}x{} for this profile",
                width,
                height,
                aligned_width,
                aligned_height,
                gran_w,
                gran_h,
                capabilities.max_coded_extent.width,
                capabilities.max_coded_extent.height
            )));
        }

        info!(
            "Using coded extent {}x{} (granularity {}x{}, min {}x{}, max {}x{})",
            aligned_width,
            aligned_height,
            gran_w,
            gran_h,
            capabilities.min_coded_extent.width,
            capabilities.min_coded_extent.height,
            capabilities.max_coded_extent.width,
            capabilities.max_coded_extent.height
        );

        // Query supported formats separately for SRC and DPB usage (vk_video_samples-style).
        // Using an unsupported DPB format is a common cause of device loss, especially for 4:4:4.
        let supported_src_formats = query_supported_video_formats(
            &context,
            &profile_info,
            vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR,
        )?;
        let supported_dpb_formats = query_supported_video_formats(
            &context,
            &profile_info,
            vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR,
        )?;

        if supported_src_formats.is_empty() {
            return Err(PixelForgeError::NoSuitableDevice(
                "No supported Vulkan Video SRC formats returned for this profile".to_string(),
            ));
        }
        info!("Supported SRC formats: {:?}", supported_src_formats);
        if supported_dpb_formats.is_empty() {
            return Err(PixelForgeError::NoSuitableDevice(
                "No supported Vulkan Video DPB formats returned for this profile".to_string(),
            ));
        }
        info!("Supported DPB formats: {:?}", supported_dpb_formats);

        // For input uploads, we currently require the preferred 2-plane formats.
        let picture_format = if supported_src_formats.contains(&preferred_src_format) {
            preferred_src_format
        } else {
            return Err(PixelForgeError::NoSuitableDevice(format!(
                "Preferred input format {:?} is not supported for VIDEO_ENCODE_SRC_KHR. Supported: {:?}",
                preferred_src_format, supported_src_formats
            )));
        };

        // DPB format can differ from the input format; prefer matching when possible.
        let reference_picture_format = supported_dpb_formats
            .iter()
            .copied()
            .find(|f| *f == picture_format)
            .unwrap_or(supported_dpb_formats[0]);

        debug!(
            "Selected Vulkan Video formats: picture_format={:?}, reference_picture_format={:?} (preferred_src={:?})",
            picture_format,
            reference_picture_format,
            preferred_src_format
        );

        // Create video session.
        // Use the STD header version reported by the driver capabilities.
        let std_header_version = capabilities.std_header_version;

        // Calculate required DPB slots and active references.
        let max_dpb_slots_supported = capabilities.max_dpb_slots as usize;
        let max_active_reference_pictures_supported =
            capabilities.max_active_reference_pictures as usize;

        if max_dpb_slots_supported < 2 {
            return Err(PixelForgeError::NoSuitableDevice(format!(
                "Device reports max_dpb_slots={} for this profile; need at least 2",
                max_dpb_slots_supported
            )));
        }

        // Target number of active reference pictures.
        // H.264 L0 list can theoretically handle more, but we clamp to config and device limits.
        let mut target_active_refs = (config.max_reference_frames as usize)
            .min(max_active_reference_pictures_supported)
            .min(32);

        // Ensure we have at least 1 active ref if supported.
        if target_active_refs < 1 && max_active_reference_pictures_supported >= 1 {
            target_active_refs = 1;
        }

        // Calculate required DPB slots.
        let requested_dpb_slots = if config.b_frame_count > 0 {
            // For B-frames: Active Refs + B-frame buffer + Setup slot + Margin
            target_active_refs + config.b_frame_count as usize + 2
        } else {
            // For P-frames: Active Refs + Setup slot
            // We use target_active_refs + 1 (setup), and maybe +1 for safety if parallel operations occur.
            target_active_refs + 1
        };

        let dpb_slot_count = requested_dpb_slots
            .min(max_dpb_slots_supported)
            .min(crate::encoder::dpb::MAX_DPB_SLOTS);

        // Finalize active reference count based on what we actually allocated.
        // We need at least 1 slot for the current setup frame.
        let max_active_reference_pictures =
            target_active_refs.min(dpb_slot_count.saturating_sub(1)); // Ensure room for setup

        debug!(
            "Allocating {} DPB slots (requested {}, device max {}), max_active_reference_pictures={} (target {}, device max {})",
            dpb_slot_count,
            requested_dpb_slots,
            max_dpb_slots_supported,
            max_active_reference_pictures,
            target_active_refs,
            max_active_reference_pictures_supported
        );

        let encode_queue_family = context.video_encode_queue_family().ok_or_else(|| {
            PixelForgeError::NoSuitableDevice("No video encode queue family available".to_string())
        })?;

        let session_create_info = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(encode_queue_family)
            .flags(vk::VideoSessionCreateFlagsKHR::empty())
            .video_profile(&profile_info)
            .picture_format(picture_format)
            .max_coded_extent(vk::Extent2D {
                width: aligned_width,
                height: aligned_height,
            })
            .reference_picture_format(reference_picture_format)
            .max_dpb_slots(dpb_slot_count as u32)
            .max_active_reference_pictures(max_active_reference_pictures as u32)
            .std_header_version(&std_header_version);

        let mut session = vk::VideoSessionKHR::null();
        let result = unsafe {
            (video_queue_fn.fp().create_video_session_khr)(
                context.device().handle(),
                &session_create_info,
                ptr::null(),
                &mut session,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(PixelForgeError::VideoSessionCreation(format!(
                "{:?}",
                result
            )));
        }

        // Query and allocate session memory.
        let session_memory = allocate_session_memory(&context, session, &video_queue_fn)?;

        // Create SPS and PPS.
        let pic_width_in_mbs = aligned_width / 16;
        let pic_height_in_map_units = aligned_height / 16;

        // Cropping offsets are expressed in units that depend on chroma subsampling.
        // For progressive frames (frame_mbs_only_flag=1):
        // - 4:2:0 => crop_unit_x=2, crop_unit_y=2
        // - 4:4:4 => crop_unit_x=1, crop_unit_y=1
        let (crop_unit_x, crop_unit_y) = match config.pixel_format {
            PixelFormat::Yuv420 => (2u32, 2u32),
            PixelFormat::Yuv444 => (1u32, 1u32),
            _ => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "Unsupported pixel format for H.264: {:?}",
                    config.pixel_format
                )));
            }
        };

        let coded_width = pic_width_in_mbs * 16;
        let coded_height = pic_height_in_map_units * 16;
        let crop_right_pixels = coded_width.saturating_sub(width);
        let crop_bottom_pixels = coded_height.saturating_sub(height);

        if !crop_right_pixels.is_multiple_of(crop_unit_x) {
            return Err(PixelForgeError::InvalidInput(format!(
                "Width {} is not representable for {:?} with coded width {} (crop_unit_x={}): crop delta {} must be divisible by crop unit",
                width, config.pixel_format, coded_width, crop_unit_x, crop_right_pixels
            )));
        }
        if !crop_bottom_pixels.is_multiple_of(crop_unit_y) {
            return Err(PixelForgeError::InvalidInput(format!(
                "Height {} is not representable for {:?} with coded height {} (crop_unit_y={}): crop delta {} must be divisible by crop unit",
                height, config.pixel_format, coded_height, crop_unit_y, crop_bottom_pixels
            )));
        }

        let color_desc = config
            .color_description
            .unwrap_or(ColorDescription::bt709());

        // Create profile info for images/buffers.
        let mut h264_profile_for_resources =
            vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(profile_idc);
        let mut profile_for_resources = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(luma_bit_depth)
            .chroma_bit_depth(chroma_bit_depth);
        profile_for_resources.p_next =
            (&mut h264_profile_for_resources as *mut vk::VideoEncodeH264ProfileInfoKHR).cast();

        // Determine DPB mode: use layered DPB when the driver does not advertise
        // support for separate reference images (required for AMD RADV).
        let supports_separate_dpb = capabilities
            .flags
            .contains(vk::VideoCapabilityFlagsKHR::SEPARATE_REFERENCE_IMAGES);
        let use_layered_dpb = !supports_separate_dpb;
        if use_layered_dpb {
            info!("Using layered DPB (driver does not support separate reference images)");
        }

        // Create DPB images (shared across slots).
        let (dpb_images, dpb_image_memories, dpb_image_views) = create_dpb_images(
            &context,
            aligned_width,
            aligned_height,
            reference_picture_format,
            dpb_slot_count,
            &profile_for_resources,
            use_layered_dpb,
        )?;

        // Create command pool and shared upload resources. Encode command
        // buffers (one per slot) are allocated below from `command_pool`.
        let upload_queue_family = context.transfer_queue_family();
        let cmd_resources =
            create_command_resources(&context, encode_queue_family, upload_queue_family)?;
        let command_pool = cmd_resources.command_pool;
        let upload_command_pool = cmd_resources.upload_command_pool;
        let upload_command_buffer = cmd_resources.upload_command_buffer;
        let upload_fence = cmd_resources.upload_fence;

        // Allocate ENCODE_PIPELINE_DEPTH-1 additional encode command buffers.
        let extra_buffers_needed = super::ENCODE_PIPELINE_DEPTH.saturating_sub(1) as u32;
        let extra_encode_buffers: Vec<vk::CommandBuffer> = if extra_buffers_needed > 0 {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(extra_buffers_needed);
            unsafe { context.device().allocate_command_buffers(&alloc_info) }
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?
        } else {
            Vec::new()
        };

        // Build per-slot resources.
        let mut slots: Vec<super::EncodeSlot> = Vec::with_capacity(super::ENCODE_PIPELINE_DEPTH);
        for slot_idx in 0..super::ENCODE_PIPELINE_DEPTH {
            let (input_image, input_image_memory, input_image_view) = create_image(
                &context,
                aligned_width,
                aligned_height,
                picture_format,
                false,
                &profile_for_resources,
            )?;

            let (bitstream_buffer, bitstream_buffer_memory) = create_bitstream_buffer(
                &context,
                MIN_BITSTREAM_BUFFER_SIZE,
                &profile_for_resources,
            )?;
            let bitstream_buffer_ptr =
                map_bitstream_buffer(&context, bitstream_buffer_memory, MIN_BITSTREAM_BUFFER_SIZE)?;

            clear_input_image(
                &context,
                &ClearImageParams {
                    command_buffer: upload_command_buffer,
                    fence: upload_fence,
                    queue: context.transfer_queue(),
                    image: input_image,
                    width: aligned_width,
                    height: aligned_height,
                    pixel_format: config.pixel_format,
                    bit_depth: config.bit_depth,
                },
            )?;

            let encode_command_buffer = if slot_idx == 0 {
                cmd_resources.encode_command_buffer
            } else {
                extra_encode_buffers[slot_idx - 1]
            };

            let encode_fence = if slot_idx == 0 {
                cmd_resources.encode_fence
            } else {
                let signaled = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
                unsafe { context.device().create_fence(&signaled, None) }
                    .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?
            };

            // Per-slot single-query pool.
            let mut h264_profile_info_query =
                vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(profile_idc);
            let mut profile_info_query = vk::VideoProfileInfoKHR::default()
                .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
                .chroma_subsampling(chroma_subsampling)
                .luma_bit_depth(luma_bit_depth)
                .chroma_bit_depth(chroma_bit_depth);
            profile_info_query.p_next =
                (&mut h264_profile_info_query as *mut vk::VideoEncodeH264ProfileInfoKHR).cast();
            let mut encode_feedback_create =
                vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR::default().encode_feedback_flags(
                    vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET
                        | vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN,
                );
            encode_feedback_create.p_next =
                (&mut profile_info_query as *mut vk::VideoProfileInfoKHR).cast();
            let mut query_pool_create_info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR)
                .query_count(1);
            query_pool_create_info.p_next = (&mut encode_feedback_create
                as *mut vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR)
                .cast();
            let query_pool = unsafe {
                context
                    .device()
                    .create_query_pool(&query_pool_create_info, None)
            }
            .map_err(|e| PixelForgeError::QueryPool(e.to_string()))?;

            slots.push(super::EncodeSlot {
                input_image,
                input_image_memory,
                input_image_view,
                input_image_layout: vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
                bitstream_buffer,
                bitstream_buffer_memory,
                bitstream_buffer_ptr,
                encode_command_buffer,
                encode_fence,
                query_pool,
                in_flight: false,
                pending_metadata: None,
            });
        }

        // Create DPB and GOP structure.
        // The DPB size should match the actual number of allocated DPB slots.
        let mut dpb = DecodedPictureBuffer::new();
        let dpb_config = DpbConfig {
            dpb_size: dpb_slot_count as u32,
            max_num_ref_frames: if config.b_frame_count > 0 { 2 } else { 1 },
            use_multiple_references: config.b_frame_count > 0,
            max_long_term_refs: 0,
            log2_max_frame_num_minus4: 4,         // max_frame_num = 256
            log2_max_pic_order_cnt_lsb_minus4: 4, // max_poc_lsb = 256
            num_temporal_layers: 1,
        };
        dpb.h264.sequence_start(dpb_config);

        let mut gop = if config.b_frame_count > 0 {
            GopStructure::new(config.gop_size, config.b_frame_count, config.gop_size)
        } else {
            GopStructure::new_ip_only(config.gop_size)
        };

        // Set GOP parameters to match SPS values.
        // log2_max_frame_num_minus4 = 4, so max_frame_num = 2^8 = 256
        gop.set_max_frame_num(4);
        // log2_max_pic_order_cnt_lsb_minus4 = 4, so max_poc_lsb = 2^8 = 256
        gop.set_max_poc_lsb(4);

        info!("H.264 encoder created successfully");

        let mut encoder = Self {
            context,
            config: config.clone(),
            dpb,
            gop,
            aligned_width,
            aligned_height,
            video_queue_fn,
            video_encode_fn,
            session,
            session_params: vk::VideoSessionParametersKHR::null(),
            session_memory,
            input_frame_num: 0,
            encode_frame_num: 0,
            frame_num_syntax: 0,
            idr_pic_id: 0,
            slots,
            current_slot: 0,
            dpb_images,
            dpb_image_memories,
            dpb_image_views,
            dpb_slot_count,
            use_layered_dpb,
            dpb_slot_active: vec![false; dpb_slot_count],
            current_dpb_slot: 0,
            l0_references: Vec::new(),
            active_reference_count: max_active_reference_pictures as u32,
            command_pool,
            upload_command_pool,
            upload_command_buffer,
            upload_fence,
            pending_wait_semaphore: None,
            sps_written: false,
            // has_reference: false, // removed
            // reference_frame_num: 0, // removed
            // reference_poc: 0, // removed
            has_backward_reference: false,
            backward_reference_frame_num: 0,
            backward_reference_poc: 0,
            backward_reference_dpb_slot: 2,
            profile_idc,
            preferred_entropy_cabac,
        };

        encoder.session_params = encoder.create_session_params(&color_desc)?;

        Ok(encoder)
    }
}
