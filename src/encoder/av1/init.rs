use super::{AV1Encoder, MIN_BITSTREAM_BUFFER_SIZE, SUPERBLOCK_SIZE};

use crate::encoder::gop::GopStructure;
use crate::encoder::resources::{
    allocate_session_memory, clear_input_image, clear_rgb_input_image, create_bitstream_buffer,
    create_command_resources, create_dpb_images, create_image, get_video_format, make_codec_name,
    map_bitstream_buffer, query_supported_video_formats, rgb_conversion_model,
    rgb_conversion_range, rgb_input_format, ClearImageParams,
};
use crate::encoder::{ColorDescription, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;
use ash::vk;
use std::ptr;
use tracing::{debug, info, warn};

impl AV1Encoder {
    /// Create a new AV1 encoder.
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

        warn!(
            "AV1 encoding is experimental. On NVIDIA GPUs, P-frames cannot reference other \
             P-frames, causing all P-frames to reference the I-frame instead. This leads to \
             progressively larger frame sizes over time. Consider using H.264 or HEVC until \
             this is resolved."
        );

        info!(
            "Creating AV1 encoder: requested {}x{}, pixel_format={:?}",
            width, height, config.pixel_format
        );

        // Load video queue extension functions.
        let video_queue_fn =
            ash::khr::video_queue::Device::load(context.instance(), context.device());
        let video_encode_fn =
            ash::khr::video_encode_queue::Device::load(context.instance(), context.device());

        if config.use_rgb_input && !context.supports_rgb_direct_encode() {
            return Err(PixelForgeError::NoSuitableDevice(
                "EncodeConfig::use_rgb_input requires VK_VALVE_video_encode_rgb_conversion, \
                 which this device does not support."
                    .to_string(),
            ));
        }
        let use_rgb_input = config.use_rgb_input;

        // Get chroma subsampling from pixel format.
        let chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR = config.pixel_format.into();
        let luma_bit_depth: vk::VideoComponentBitDepthFlagsKHR = config.bit_depth.into();
        let chroma_bit_depth: vk::VideoComponentBitDepthFlagsKHR = config.bit_depth.into();

        // AV1 profile selection based on chroma subsampling (not bit depth).
        // Main profile: 8/10-bit, 4:2:0 only.
        // High profile: 8/10-bit, 4:2:0 and 4:4:4.
        let profile = match config.pixel_format {
            PixelFormat::Yuv420 => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN,
            _ => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_HIGH,
        };

        // Preferred input format based on pixel format and bit depth.
        let preferred_src_format = get_video_format(config.pixel_format, config.bit_depth);

        // Create AV1 encode profile. When RGB-direct is enabled we chain
        // VkVideoEncodeProfileRgbConversionInfoVALVE inside av1_profile_info
        // so all downstream uses (capability query, session create, image
        // creation, query pool) see a profile that matches.
        let mut rgb_conv_profile_info = vk::VideoEncodeProfileRgbConversionInfoVALVE::default()
            .perform_encode_rgb_conversion(true);
        let mut av1_profile_info = vk::VideoEncodeAV1ProfileInfoKHR::default().std_profile(profile);
        if use_rgb_input {
            av1_profile_info.p_next = (&mut rgb_conv_profile_info
                as *mut vk::VideoEncodeProfileRgbConversionInfoVALVE)
                .cast();
        }

        let mut profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_AV1)
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(luma_bit_depth)
            .chroma_bit_depth(chroma_bit_depth);
        profile_info.p_next =
            (&mut av1_profile_info as *mut vk::VideoEncodeAV1ProfileInfoKHR).cast();

        // Query encode capabilities.
        let video_queue_instance =
            ash::khr::video_queue::Instance::load(context.entry(), context.instance());
        let mut av1_capabilities = vk::VideoEncodeAV1CapabilitiesKHR::default();
        let mut encode_capabilities = vk::VideoEncodeCapabilitiesKHR {
            p_next: (&mut av1_capabilities as *mut vk::VideoEncodeAV1CapabilitiesKHR).cast(),
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
                "Failed to query Vulkan Video encode capabilities for AV1: {:?}",
                result
            )));
        }

        // Helper functions for alignment calculations.
        let gcd = |mut a: u32, mut b: u32| {
            while b != 0 {
                let tmp = a % b;
                a = b;
                b = tmp;
            }
            a
        };
        let lcm = |a: u32, b: u32| {
            if a == 0 || b == 0 {
                0
            } else {
                a / gcd(a, b) * b
            }
        };
        let align_up = |value: u32, alignment: u32| {
            if alignment <= 1 {
                value
            } else {
                value.div_ceil(alignment) * alignment
            }
        };

        let gran_w = capabilities.picture_access_granularity.width.max(1);
        let gran_h = capabilities.picture_access_granularity.height.max(1);
        let align_w = lcm(SUPERBLOCK_SIZE, gran_w);
        let align_h = lcm(SUPERBLOCK_SIZE, gran_h);

        let mut aligned_width = align_up(width, align_w);
        let mut aligned_height = align_up(height, align_h);

        aligned_width = aligned_width.max(capabilities.min_coded_extent.width);
        aligned_height = aligned_height.max(capabilities.min_coded_extent.height);

        if aligned_width > capabilities.max_coded_extent.width
            || aligned_height > capabilities.max_coded_extent.height
        {
            return Err(PixelForgeError::InvalidInput(format!(
                "Requested coded extent {}x{} (aligned to {}x{}) exceeds device max {}x{}",
                width,
                height,
                aligned_width,
                aligned_height,
                capabilities.max_coded_extent.width,
                capabilities.max_coded_extent.height
            )));
        }

        info!(
            "Using coded extent {}x{} (granularity {}x{})",
            aligned_width, aligned_height, gran_w, gran_h
        );

        // Query supported formats.
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
                "No supported Vulkan Video SRC formats for AV1".to_string(),
            ));
        }
        if supported_dpb_formats.is_empty() {
            return Err(PixelForgeError::NoSuitableDevice(
                "No supported Vulkan Video DPB formats for AV1".to_string(),
            ));
        }

        info!("Supported SRC formats: {:?}", supported_src_formats);
        info!("Supported DPB formats: {:?}", supported_dpb_formats);

        let picture_format = if use_rgb_input {
            let rgb_fmt = rgb_input_format(config.bit_depth);
            if !supported_src_formats.contains(&rgb_fmt) {
                return Err(PixelForgeError::NoSuitableDevice(format!(
                    "RGB-direct encode requested but driver does not advertise {:?} as a \
                     VIDEO_ENCODE_SRC_KHR format for this AV1 profile. Supported: {:?}",
                    rgb_fmt, supported_src_formats
                )));
            }
            rgb_fmt
        } else if supported_src_formats.contains(&preferred_src_format) {
            preferred_src_format
        } else {
            return Err(PixelForgeError::NoSuitableDevice(format!(
                "Preferred input format {:?} not supported for AV1",
                preferred_src_format
            )));
        };

        let reference_picture_format = if use_rgb_input {
            supported_dpb_formats
                .iter()
                .copied()
                .find(|f| *f == preferred_src_format)
                .unwrap_or(supported_dpb_formats[0])
        } else {
            supported_dpb_formats
                .iter()
                .copied()
                .find(|f| *f == picture_format)
                .unwrap_or(supported_dpb_formats[0])
        };

        debug!(
            "Selected formats: picture={:?}, reference={:?}",
            picture_format, reference_picture_format
        );

        // Get encode queue family.
        let encode_queue_family = context.video_encode_queue_family().ok_or_else(|| {
            PixelForgeError::NoSuitableDevice("No video encode queue family available".to_string())
        })?;

        // Create video session.
        let std_header_version = vk::ExtensionProperties {
            extension_name: make_codec_name(b"VK_STD_vulkan_video_codec_av1_encode"),
            spec_version: vk::make_api_version(0, 1, 0, 0),
        };

        // Calculate DPB slots and active references.
        let max_dpb_slots_supported = capabilities.max_dpb_slots as usize;
        let max_active_reference_pictures_supported =
            capabilities.max_active_reference_pictures as usize;

        if max_dpb_slots_supported < 2 {
            return Err(PixelForgeError::NoSuitableDevice(format!(
                "Device reports max_dpb_slots={}, need at least 2",
                max_dpb_slots_supported
            )));
        }

        let mut target_active_refs = (config.max_reference_frames as usize)
            .min(max_active_reference_pictures_supported)
            .min(8); // AV1 supports up to 8 reference frames

        if target_active_refs < 1 && max_active_reference_pictures_supported >= 1 {
            target_active_refs = 1;
        }

        // AV1 typically needs: active refs + 1 for current frame being setup
        let requested_dpb_slots = (target_active_refs + 1).min(max_dpb_slots_supported);

        info!(
            "DPB configuration: slots={}, active_refs={} (max_supported: slots={}, refs={})",
            requested_dpb_slots,
            target_active_refs,
            max_dpb_slots_supported,
            max_active_reference_pictures_supported
        );

        let color_desc = config
            .color_description
            .unwrap_or(ColorDescription::bt709());

        let mut session_rgb_conv_info =
            vk::VideoEncodeSessionRgbConversionCreateInfoVALVE::default()
                .rgb_model(rgb_conversion_model(&color_desc))
                .rgb_range(rgb_conversion_range(&color_desc))
                .x_chroma_offset(vk::VideoEncodeRgbChromaOffsetFlagsVALVE::COSITED_EVEN)
                .y_chroma_offset(vk::VideoEncodeRgbChromaOffsetFlagsVALVE::MIDPOINT);

        let mut session_create_info = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(encode_queue_family)
            .video_profile(&profile_info)
            .picture_format(picture_format)
            .max_coded_extent(vk::Extent2D {
                width: aligned_width,
                height: aligned_height,
            })
            .reference_picture_format(reference_picture_format)
            .max_dpb_slots(requested_dpb_slots as u32)
            .max_active_reference_pictures(target_active_refs as u32)
            .std_header_version(&std_header_version);
        if use_rgb_input {
            session_create_info.p_next = (&mut session_rgb_conv_info
                as *mut vk::VideoEncodeSessionRgbConversionCreateInfoVALVE)
                .cast();
        }

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

        // Allocate session memory.
        let session_memory = allocate_session_memory(&context, session, &video_queue_fn)?;

        // Create DPB images (shared across slots).
        let (dpb_images, dpb_image_memories, dpb_image_views) = create_dpb_images(
            &context,
            aligned_width,
            aligned_height,
            reference_picture_format,
            requested_dpb_slots,
            &profile_info,
            false,
        )?;

        let bitstream_buffer_size = MIN_BITSTREAM_BUFFER_SIZE.max(width as usize * height as usize);

        // Shared command pool / upload resources.
        let upload_queue_family = context.transfer_queue_family();
        let cmd_resources =
            create_command_resources(&context, encode_queue_family, upload_queue_family)?;
        let command_pool = cmd_resources.command_pool;
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
                &profile_info,
            )?;

            let (bitstream_buffer, bitstream_buffer_memory) =
                create_bitstream_buffer(&context, bitstream_buffer_size, &profile_info)?;
            let bitstream_buffer_ptr =
                map_bitstream_buffer(&context, bitstream_buffer_memory, bitstream_buffer_size)?;

            if use_rgb_input {
                clear_rgb_input_image(
                    &context,
                    upload_command_buffer,
                    upload_fence,
                    context.transfer_queue(),
                    input_image,
                )?;
            } else {
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
            }

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
            let mut query_feedback_info = vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR::default()
                .encode_feedback_flags(
                    vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET
                        | vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN,
                );
            query_feedback_info.p_next = (&profile_info as *const vk::VideoProfileInfoKHR).cast();
            let mut query_pool_create_info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR)
                .query_count(1);
            query_pool_create_info.p_next = (&query_feedback_info
                as *const vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR)
                .cast();
            let query_pool = unsafe {
                context
                    .device()
                    .create_query_pool(&query_pool_create_info, None)
                    .map_err(|e| PixelForgeError::QueryPool(e.to_string()))?
            };

            slots.push(super::EncodeSlot {
                input_image,
                input_image_memory,
                input_image_view,
                input_image_layout: vk::ImageLayout::UNDEFINED,
                bitstream_buffer,
                bitstream_buffer_memory,
                bitstream_buffer_size,
                bitstream_buffer_ptr,
                encode_command_buffer,
                encode_fence,
                query_pool,
                in_flight: false,
                pending_metadata: None,
            });
        }

        // Initialize GOP structure.
        let gop = GopStructure::new(config.gop_size, config.b_frame_count, config.gop_size);

        let mut encoder = Self {
            context,
            config,
            gop,
            video_queue_fn,
            video_encode_fn,
            session,
            session_params: vk::VideoSessionParametersKHR::null(),
            session_memory,
            input_frame_num: 0,
            encode_frame_num: 0,
            frame_num: 0,
            order_hint: 0,
            slots,
            current_slot: 0,
            dpb_images,
            dpb_image_memories,
            dpb_image_views,
            dpb_slot_count: requested_dpb_slots,
            dpb_slot_active: vec![false; requested_dpb_slots],
            command_pool,
            upload_command_pool: cmd_resources.upload_command_pool,
            upload_command_buffer,
            upload_fence,
            header_data: None,
            current_dpb_slot: 0,
            references: Vec::new(),
        };

        encoder.session_params = encoder.create_session_params(&color_desc)?;

        Ok(encoder)
    }
}
