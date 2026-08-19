//! AV1 sequence-header generation and OBU retrieval.

use super::Av1;

use crate::encoder::codec::EncoderCommon;
use crate::encoder::resources::get_encoded_session_params;
use crate::encoder::{BitDepth, ColorDescription, PixelFormat};
use crate::error::{PixelForgeError, Result};
use ash::vk;
use ash::vk::TaggedStructure;
use std::ptr;

impl Av1 {
    /// Build the AV1 sequence header and create Vulkan video session parameters.
    pub(super) fn build_session_params(
        &self,
        common: &EncoderCommon,
        desc: &ColorDescription,
    ) -> Result<vk::VideoSessionParametersKHR> {
        let config = &common.config;
        let width = config.dimensions.width;
        let height = config.dimensions.height;

        let frame_width_bits = 32 - (width - 1).leading_zeros();
        let frame_height_bits = 32 - (height - 1).leading_zeros();

        let av1_color_primaries = match desc.color_primaries {
            9 => ash::vk::native::StdVideoAV1ColorPrimaries_STD_VIDEO_AV1_COLOR_PRIMARIES_BT_2020,
            _ => ash::vk::native::StdVideoAV1ColorPrimaries_STD_VIDEO_AV1_COLOR_PRIMARIES_BT_709,
        };
        let av1_transfer = match desc.transfer_characteristics {
            16 => ash::vk::native::StdVideoAV1TransferCharacteristics_STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_SMPTE_2084,
            _ => ash::vk::native::StdVideoAV1TransferCharacteristics_STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_BT_709,
        };
        let av1_matrix = match desc.matrix_coefficients {
            9 => ash::vk::native::StdVideoAV1MatrixCoefficients_STD_VIDEO_AV1_MATRIX_COEFFICIENTS_BT_2020_NCL,
            _ => ash::vk::native::StdVideoAV1MatrixCoefficients_STD_VIDEO_AV1_MATRIX_COEFFICIENTS_BT_709,
        };
        let av1_full_range = if desc.full_range { 1 } else { 0 };

        let color_config_flags = ash::vk::native::StdVideoAV1ColorConfigFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoAV1ColorConfigFlags::new_bitfield_1(
                0,              // mono_chrome
                av1_full_range, // color_range
                0,              // separate_uv_delta_q
                1,              // color_description_present_flag
                0,              // reserved
            ),
        };

        let bit_depth = match config.bit_depth {
            BitDepth::Eight => 8,
            BitDepth::Ten => 10,
        };

        let (subsampling_x, subsampling_y) = match config.pixel_format {
            PixelFormat::Yuv420 => (1u8, 1u8),
            PixelFormat::Yuv444 => (0u8, 0u8),
            _ => (1u8, 1u8),
        };

        let color_config = ash::vk::native::StdVideoAV1ColorConfig {
            flags: color_config_flags,
            BitDepth: bit_depth,
            subsampling_x,
            subsampling_y,
            reserved1: 0,
            color_primaries: av1_color_primaries,
            transfer_characteristics: av1_transfer,
            matrix_coefficients: av1_matrix,
            chroma_sample_position: ash::vk::native::StdVideoAV1ChromaSamplePosition_STD_VIDEO_AV1_CHROMA_SAMPLE_POSITION_UNKNOWN,
        };

        let profile = match config.pixel_format {
            PixelFormat::Yuv420 => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN,
            _ => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_HIGH,
        };

        // Minimal sequence-header flags (avoid driver issues).
        let seq_flags = ash::vk::native::StdVideoAV1SequenceHeaderFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoAV1SequenceHeaderFlags::new_bitfield_1(
                0, // still_picture
                0, // reduced_still_picture_header
                0, // use_128x128_superblock
                0, // enable_filter_intra
                0, // enable_intra_edge_filter
                0, // enable_interintra_compound
                0, // enable_masked_compound
                0, // enable_warped_motion
                0, // enable_dual_filter
                1, // enable_order_hint
                0, // enable_jnt_comp
                0, // enable_ref_frame_mvs
                0, // frame_id_numbers_present_flag
                0, // enable_superres
                1, // enable_cdef
                0, // enable_restoration
                0, // film_grain_params_present
                0, // timing_info_present_flag
                0, // initial_display_delay_present_flag
                0, // reserved
            ),
        };

        let av1_sequence_header = ash::vk::native::StdVideoAV1SequenceHeader {
            flags: seq_flags,
            seq_profile: profile,
            frame_width_bits_minus_1: (frame_width_bits - 1) as u8,
            frame_height_bits_minus_1: (frame_height_bits - 1) as u8,
            max_frame_width_minus_1: (width - 1) as u16,
            max_frame_height_minus_1: (height - 1) as u16,
            delta_frame_id_length_minus_2: 0,
            additional_frame_id_length_minus_1: 0,
            order_hint_bits_minus_1: 7,
            seq_force_integer_mv: 0,
            seq_force_screen_content_tools: 0,
            reserved1: [0; 5],
            pColorConfig: &color_config,
            pTimingInfo: ptr::null(),
        };

        let decoder_model_info = ash::vk::native::StdVideoEncodeAV1DecoderModelInfo {
            buffer_delay_length_minus_1: 0,
            buffer_removal_time_length_minus_1: 0,
            frame_presentation_time_length_minus_1: 0,
            reserved1: 0,
            num_units_in_decoding_tick: 0,
        };

        let operating_point = ash::vk::native::StdVideoEncodeAV1OperatingPointInfo {
            flags: ash::vk::native::StdVideoEncodeAV1OperatingPointInfoFlags {
                _bitfield_align_1: [],
                _bitfield_1:
                    ash::vk::native::StdVideoEncodeAV1OperatingPointInfoFlags::new_bitfield_1(
                        0, 0, 0, 0,
                    ),
            },
            operating_point_idc: 0,
            seq_level_idx: 5, // Level 3.1
            seq_tier: 0,
            initial_display_delay_minus_1: 0,
            decoder_buffer_delay: 0,
            encoder_buffer_delay: 0,
        };

        let mut av1_session_params_create_info =
            vk::VideoEncodeAV1SessionParametersCreateInfoKHR::default()
                .std_sequence_header(&av1_sequence_header)
                .std_decoder_model_info(&decoder_model_info)
                .std_operating_points(std::slice::from_ref(&operating_point));

        let mut quality_info = vk::VideoEncodeQualityLevelInfoKHR::default()
            .quality_level(common.config.encode_quality_level);
        let session_params_create_info = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(common.session)
            .push(&mut quality_info)
            .push(&mut av1_session_params_create_info);

        unsafe {
            common
                .video_queue_fn
                .create_video_session_parameters(&session_params_create_info, None)
                .map_err(|e| {
                    PixelForgeError::SessionParametersCreation(format!(
                        "Failed to create AV1 session parameters: {:?}",
                        e
                    ))
                })
        }
    }

    /// Retrieve the driver-generated sequence-header OBU for key frames.
    pub(super) fn build_header(&self, common: &EncoderCommon) -> Result<Vec<u8>> {
        let get_info = vk::VideoEncodeSessionParametersGetInfoKHR {
            video_session_parameters: common.session_params,
            ..Default::default()
        };
        let mut feedback = vk::VideoEncodeSessionParametersFeedbackInfoKHR::default();
        get_encoded_session_params(
            &common.context,
            &common.video_encode_fn,
            &get_info,
            &mut feedback,
        )
    }
}
