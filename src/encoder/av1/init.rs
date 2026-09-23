use super::{Av1, MIN_BITSTREAM_BUFFER_SIZE, SUPERBLOCK_SIZE};

use crate::encoder::codec::{
    CodecEncoder, CommonInitRequest, RgbConversionCaps, build_encoder_common, query_video_caps,
    rgb_conversion_profile,
};
use crate::encoder::{ColorDescription, EncodeConfig, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;
use ash::vk;
use ash::vk::TaggedStructure;
use tracing::info;

impl Av1 {
    /// Create a new AV1 encoder.
    pub fn create(context: VideoContext, config: EncodeConfig) -> Result<CodecEncoder<Self>> {
        // Reachable from safe caller code via `with_b_frames`, so report it
        // rather than panicking inside the library.
        if config.b_frame_count != 0 {
            return Err(PixelForgeError::InvalidInput(format!(
                "AV1 encode: B-frames are not yet supported; set b_frame_count = 0 (got {})",
                config.b_frame_count
            )));
        }

        info!(
            "Creating AV1 encoder: {}x{}, pixel_format={:?}",
            config.dimensions.width, config.dimensions.height, config.pixel_format
        );

        // AV1 profile from chroma subsampling: Main is 4:2:0, High adds 4:4:4.
        let profile = match config.pixel_format {
            PixelFormat::Yuv420 => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN,
            _ => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_HIGH,
        };

        let chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR = config.pixel_format.into();
        let bit_depth: vk::VideoComponentBitDepthFlagsKHR = config.bit_depth.into();

        // Get encoder tuning mode, usage and content hints.
        let encode_usage_hint: vk::VideoEncodeUsageFlagsKHR = config.encode_usage_hint.into();
        let encode_content_hint: vk::VideoEncodeContentFlagsKHR = config.encode_content_hint.into();
        let encoder_tuning_mode: vk::VideoEncodeTuningModeKHR = config.encoder_tuning_mode.into();

        let mut video_encode_usage_info = vk::VideoEncodeUsageInfoKHR::default()
            .video_usage_hints(encode_usage_hint)
            .video_content_hints(encode_content_hint)
            .tuning_mode(encoder_tuning_mode);

        let mut av1_profile_info = vk::VideoEncodeAV1ProfileInfoKHR::default().std_profile(profile);
        let mut rgb_profile = rgb_conversion_profile();
        let mut profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_AV1)
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(bit_depth)
            .chroma_bit_depth(bit_depth)
            .push(&mut av1_profile_info)
            .push(&mut video_encode_usage_info);
        if config.rgb_input.is_some() {
            profile_info = profile_info.push(&mut rgb_profile);
        }

        let bitstream_buffer_size = MIN_BITSTREAM_BUFFER_SIZE
            .max(config.dimensions.width as usize * config.dimensions.height as usize);

        // Query device capabilities with the AV1-specific capability struct
        // chained in (required by the driver).
        let mut av1_caps = vk::VideoEncodeAV1CapabilitiesKHR::default();
        let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
        let mut capabilities = vk::VideoCapabilitiesKHR::default()
            .push(&mut encode_caps)
            .push(&mut av1_caps);
        let mut rgb_caps = vk::VideoEncodeRgbConversionCapabilitiesVALVE::default();
        if config.rgb_input.is_some() {
            capabilities = capabilities.push(&mut rgb_caps);
        }
        let caps = query_video_caps(&context, &profile_info, &mut capabilities)?;
        let rgb_caps = config.rgb_input.map(|_| RgbConversionCaps::from(&rgb_caps));

        let init = build_encoder_common(&CommonInitRequest {
            context: &context,
            config: &config,
            profile_info: &profile_info,
            caps: &caps,
            align_unit: SUPERBLOCK_SIZE,
            // AV1 has NUM_REF_FRAMES = 8 DPB slots, and `refresh_frame_flags`
            // is an 8-bit syntax element. `build_encoder_common` adds one setup
            // slot on top of the active count, so cap active refs at 7 to keep
            // every DPB slot index in 0..=7 (the range the [u8; 8] order-hint
            // array and `1 << slot` refresh mask can represent).
            max_active_refs_cap: 7,
            bitstream_buffer_size,
            // Allow the shared infrastructure to fall back to a layered DPB
            // when the driver does not support separate reference images
            // (required for AMD RADV).
            allow_layered_dpb: true,
            rgb_caps,
        })?;
        let active_reference_count = init.active_reference_count;
        let common = init.common;

        let codec = Av1 {
            frame_num: 0,
            order_hint: 0,
            header_data: None,
            references: Vec::new(),
            active_reference_count,
        };

        let mut encoder = CodecEncoder { common, codec };
        let color_desc = config
            .color_description
            .unwrap_or(ColorDescription::bt709());
        encoder.common.session_params = encoder
            .codec
            .build_session_params(&encoder.common, &color_desc)?;

        info!("AV1 encoder created successfully");
        Ok(encoder)
    }
}
