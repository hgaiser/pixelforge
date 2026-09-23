use super::{CTB_SIZE, H265};

use crate::encoder::codec::{
    CodecEncoder, CommonInitRequest, RgbConversionCaps, build_encoder_common, query_video_caps,
    rgb_conversion_profile,
};
use crate::encoder::dpb::{DecodedPictureBuffer, DecodedPictureBufferTrait, DpbConfig};
use crate::encoder::resources::MIN_BITSTREAM_BUFFER_SIZE;
use crate::encoder::{BitDepth, ColorDescription, EncodeConfig, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;
use ash::vk;
use ash::vk::TaggedStructure;
use tracing::info;

impl H265 {
    /// Create a new H.265/HEVC encoder.
    pub fn create(context: VideoContext, config: EncodeConfig) -> Result<CodecEncoder<Self>> {
        // Reachable from safe caller code via `with_b_frames`, so report it
        // rather than panicking inside the library.
        if config.b_frame_count != 0 {
            return Err(PixelForgeError::InvalidInput(format!(
                "H.265 encode: B-frames are not yet supported; set b_frame_count = 0 (got {})",
                config.b_frame_count
            )));
        }

        info!(
            "Creating H.265 encoder: {}x{}, pixel_format={:?}",
            config.dimensions.width, config.dimensions.height, config.pixel_format
        );

        let profile_idc = match (config.pixel_format, config.bit_depth) {
            (PixelFormat::Yuv420, BitDepth::Eight) => {
                ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN
            }
            (PixelFormat::Yuv420, BitDepth::Ten) => {
                ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10
            }
            (PixelFormat::Yuv444, _) => {
                ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS
            }
            _ => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "Unsupported pixel format / bit depth for H.265: {:?} / {:?}",
                    config.pixel_format, config.bit_depth
                )));
            }
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

        let mut h265_profile_info =
            vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(profile_idc);
        let mut rgb_profile = rgb_conversion_profile();
        let mut profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H265)
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(bit_depth)
            .chroma_bit_depth(bit_depth)
            .push(&mut h265_profile_info)
            .push(&mut video_encode_usage_info);
        if config.rgb_input.is_some() {
            profile_info = profile_info.push(&mut rgb_profile);
        }

        // Query device capabilities with the H.265-specific capability struct
        // chained in (required by the driver).
        let mut h265_caps = vk::VideoEncodeH265CapabilitiesKHR::default();
        let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
        let mut capabilities = vk::VideoCapabilitiesKHR::default()
            .push(&mut encode_caps)
            .push(&mut h265_caps);
        let mut rgb_caps = vk::VideoEncodeRgbConversionCapabilitiesVALVE::default();
        if config.rgb_input.is_some() {
            capabilities = capabilities.push(&mut rgb_caps);
        }
        let caps = query_video_caps(&context, &profile_info, &mut capabilities)?;
        let rgb_caps = config.rgb_input.map(|_| RgbConversionCaps::from(&rgb_caps));

        // Scale the bitstream buffer by resolution so a single frame can't overflow (matches AV1).
        let bitstream_buffer_size = MIN_BITSTREAM_BUFFER_SIZE
            .max(config.dimensions.width as usize * config.dimensions.height as usize);

        let init = build_encoder_common(&CommonInitRequest {
            context: &context,
            config: &config,
            profile_info: &profile_info,
            caps: &caps,
            align_unit: CTB_SIZE,
            max_active_refs_cap: 15,
            bitstream_buffer_size,
            allow_layered_dpb: true,
            rgb_caps,
        })?;
        let active_reference_count = init.active_reference_count;
        let common = init.common;

        let mut dpb = DecodedPictureBuffer::new();
        dpb.h265.sequence_start(DpbConfig {
            dpb_size: common.dpb_slot_count as u32,
            max_num_ref_frames: if config.b_frame_count > 0 { 2 } else { 1 },
            use_multiple_references: config.b_frame_count > 0,
            max_long_term_refs: 0,
            log2_max_frame_num_minus4: 0,
            log2_max_pic_order_cnt_lsb_minus4: 4,
            num_temporal_layers: 1,
        });

        let codec = H265 {
            dpb,
            header_data: None,
            has_backward_reference: false,
            backward_reference_poc: 0,
            backward_reference_dpb_slot: 2,
            l0_references: Vec::new(),
            active_reference_count,
            profile_idc,
        };

        let mut encoder = CodecEncoder { common, codec };
        let color_desc = config
            .color_description
            .unwrap_or(ColorDescription::bt709());
        encoder.common.session_params = encoder
            .codec
            .build_session_params(&encoder.common, &color_desc)?;

        info!("H.265 encoder created successfully");
        Ok(encoder)
    }

    /// Profile IDC (used when rebuilding parameter sets).
    pub(super) fn profile_idc(&self) -> u32 {
        self.profile_idc
    }
}
