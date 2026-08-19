//! H.265 parameter-set generation: VPS/SPS/PPS construction and header retrieval.

use super::H265;

use crate::encoder::codec::EncoderCommon;
use crate::encoder::resources::get_encoded_session_params;
use crate::encoder::{BitDepth, ColorDescription, PixelFormat};
use crate::error::{PixelForgeError, Result};
use ash::vk;
use ash::vk::TaggedStructure;
use std::ptr;

impl H265 {
    /// Build VPS/SPS/PPS and create Vulkan video session parameters.
    pub(super) fn build_session_params(
        &self,
        common: &EncoderCommon,
        desc: &ColorDescription,
    ) -> Result<vk::VideoSessionParametersKHR> {
        let config = &common.config;

        let ctb_log2_size_y: u8 = 5;
        let min_cb_log2_size_y: u8 = 4;
        let log2_min_transform_block_size: u8 = 2;
        let log2_max_transform_block_size: u8 = 5;

        let pic_width_in_luma_samples = common.aligned_width;
        let pic_height_in_luma_samples = common.aligned_height;

        let (sub_width_c, sub_height_c) = match config.pixel_format {
            PixelFormat::Yuv420 => (2u32, 2u32),
            PixelFormat::Yuv444 => (1u32, 1u32),
            _ => (2u32, 2u32),
        };
        let conf_win_right_offset = (common.aligned_width - config.dimensions.width) / sub_width_c;
        let conf_win_bottom_offset =
            (common.aligned_height - config.dimensions.height) / sub_height_c;
        let conformance_window_flag = conf_win_right_offset > 0 || conf_win_bottom_offset > 0;

        let profile_tier_level = ash::vk::native::StdVideoH265ProfileTierLevel {
            flags: ash::vk::native::StdVideoH265ProfileTierLevelFlags {
                _bitfield_align_1: [],
                _bitfield_1: ash::vk::native::StdVideoH265ProfileTierLevelFlags::new_bitfield_1(
                    0, 1, 0, 0, 1,
                ),
                __bindgen_padding_0: [0; 3],
            },
            general_profile_idc: self.profile_idc(),
            general_level_idc: ash::vk::native::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_1,
        };

        let dec_pic_buf_mgr = ash::vk::native::StdVideoH265DecPicBufMgr {
            max_latency_increase_plus1: [0; 7],
            max_dec_pic_buffering_minus1: [(common.dpb_slot_count - 1) as u8, 0, 0, 0, 0, 0, 0],
            max_num_reorder_pics: [0; 7],
        };

        let short_term_ref_pic_set = ash::vk::native::StdVideoH265ShortTermRefPicSet {
            flags: ash::vk::native::StdVideoH265ShortTermRefPicSetFlags {
                _bitfield_align_1: [],
                _bitfield_1: ash::vk::native::StdVideoH265ShortTermRefPicSetFlags::new_bitfield_1(
                    0, 0,
                ),
                __bindgen_padding_0: [0; 3],
            },
            delta_idx_minus1: 0,
            use_delta_flag: 0,
            abs_delta_rps_minus1: 0,
            used_by_curr_pic_flag: 0,
            used_by_curr_pic_s0_flag: 1,
            used_by_curr_pic_s1_flag: 0,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
            num_negative_pics: 1,
            num_positive_pics: 0,
            delta_poc_s0_minus1: [0; 16],
            delta_poc_s1_minus1: [0; 16],
        };

        let long_term_ref_pics_sps = ash::vk::native::StdVideoH265LongTermRefPicsSps {
            used_by_curr_pic_lt_sps_flag: 0,
            lt_ref_pic_poc_lsb_sps: [0; 32],
        };

        let sps_flags = ash::vk::native::StdVideoH265SpsFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoH265SpsFlags::new_bitfield_1(
                1,
                0,
                if conformance_window_flag { 1 } else { 0 },
                1,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
        };

        let vui_flags = ash::vk::native::StdVideoH265SpsVuiFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoH265SpsVuiFlags::new_bitfield_1(
                1,
                0,
                0,
                1,
                if desc.full_range { 1 } else { 0 },
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            __bindgen_padding_0: 0,
        };

        let vui = ash::vk::native::StdVideoH265SequenceParameterSetVui {
            flags: vui_flags,
            aspect_ratio_idc:
                ash::vk::native::StdVideoH265AspectRatioIdc_STD_VIDEO_H265_ASPECT_RATIO_IDC_SQUARE,
            sar_width: 0,
            sar_height: 0,
            video_format: 5,
            colour_primaries: desc.color_primaries,
            transfer_characteristics: desc.transfer_characteristics,
            matrix_coeffs: desc.matrix_coefficients,
            chroma_sample_loc_type_top_field: 0,
            chroma_sample_loc_type_bottom_field: 0,
            reserved1: 0,
            reserved2: 0,
            def_disp_win_left_offset: 0,
            def_disp_win_right_offset: 0,
            def_disp_win_top_offset: 0,
            def_disp_win_bottom_offset: 0,
            vui_num_units_in_tick: 0,
            vui_time_scale: 0,
            vui_num_ticks_poc_diff_one_minus1: 0,
            min_spatial_segmentation_idc: 0,
            reserved3: 0,
            max_bytes_per_pic_denom: 0,
            max_bits_per_min_cu_denom: 0,
            log2_max_mv_length_horizontal: 0,
            log2_max_mv_length_vertical: 0,
            pHrdParameters: ptr::null(),
        };

        let bit_depth_minus8: u8 = match config.bit_depth {
            BitDepth::Eight => 0,
            BitDepth::Ten => 2,
        };

        let chroma_format_idc = match config.pixel_format {
            PixelFormat::Yuv420 => {
                ash::vk::native::StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_420
            }
            PixelFormat::Yuv444 => {
                ash::vk::native::StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_444
            }
            _ => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "Unsupported pixel format for H.265: {:?}",
                    config.pixel_format
                )));
            }
        };

        let sps = ash::vk::native::StdVideoH265SequenceParameterSet {
            flags: sps_flags,
            chroma_format_idc,
            pic_width_in_luma_samples,
            pic_height_in_luma_samples,
            sps_video_parameter_set_id: 0,
            sps_max_sub_layers_minus1: 0,
            sps_seq_parameter_set_id: 0,
            bit_depth_luma_minus8: bit_depth_minus8,
            bit_depth_chroma_minus8: bit_depth_minus8,
            log2_max_pic_order_cnt_lsb_minus4: 4,
            log2_min_luma_coding_block_size_minus3: min_cb_log2_size_y - 3,
            log2_diff_max_min_luma_coding_block_size: ctb_log2_size_y - min_cb_log2_size_y,
            log2_min_luma_transform_block_size_minus2: log2_min_transform_block_size - 2,
            log2_diff_max_min_luma_transform_block_size: log2_max_transform_block_size
                - log2_min_transform_block_size,
            max_transform_hierarchy_depth_inter: (ctb_log2_size_y - log2_min_transform_block_size)
                .max(1),
            max_transform_hierarchy_depth_intra: 3,
            num_short_term_ref_pic_sets: 0,
            num_long_term_ref_pics_sps: 0,
            pcm_sample_bit_depth_luma_minus1: 7,
            pcm_sample_bit_depth_chroma_minus1: 7,
            log2_min_pcm_luma_coding_block_size_minus3: min_cb_log2_size_y - 3,
            log2_diff_max_min_pcm_luma_coding_block_size: ctb_log2_size_y - min_cb_log2_size_y,
            reserved1: 0,
            reserved2: 0,
            palette_max_size: 0,
            delta_palette_max_predictor_size: 0,
            motion_vector_resolution_control_idc: 0,
            sps_num_palette_predictor_initializers_minus1: 0,
            conf_win_left_offset: 0,
            conf_win_right_offset,
            conf_win_top_offset: 0,
            conf_win_bottom_offset,
            pProfileTierLevel: ptr::null(),
            pDecPicBufMgr: ptr::null(),
            pScalingLists: ptr::null(),
            pShortTermRefPicSet: ptr::null(),
            pLongTermRefPicsSps: ptr::null(),
            pSequenceParameterSetVui: &vui,
            pPredictorPaletteEntries: ptr::null(),
        };

        let profile_tier_level_boxed = Box::new(profile_tier_level);
        let dec_pic_buf_mgr_boxed = Box::new(dec_pic_buf_mgr);
        let short_term_ref_pic_set_boxed = Box::new(short_term_ref_pic_set);
        let long_term_ref_pics_sps_boxed = Box::new(long_term_ref_pics_sps);

        let mut sps_with_ptrs = sps;
        sps_with_ptrs.pProfileTierLevel = profile_tier_level_boxed.as_ref();
        sps_with_ptrs.pDecPicBufMgr = dec_pic_buf_mgr_boxed.as_ref();
        sps_with_ptrs.pShortTermRefPicSet = short_term_ref_pic_set_boxed.as_ref();
        sps_with_ptrs.pLongTermRefPicsSps = long_term_ref_pics_sps_boxed.as_ref();

        let vps_flags = ash::vk::native::StdVideoH265VpsFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoH265VpsFlags::new_bitfield_1(1, 1, 0, 0),
            __bindgen_padding_0: [0; 3],
        };

        let vps = ash::vk::native::StdVideoH265VideoParameterSet {
            flags: vps_flags,
            vps_video_parameter_set_id: 0,
            vps_max_sub_layers_minus1: 0,
            reserved1: 0,
            reserved2: 0,
            vps_num_units_in_tick: 0,
            vps_time_scale: 0,
            vps_num_ticks_poc_diff_one_minus1: 0,
            reserved3: 0,
            pDecPicBufMgr: dec_pic_buf_mgr_boxed.as_ref(),
            pHrdParameters: ptr::null(),
            pProfileTierLevel: profile_tier_level_boxed.as_ref(),
        };

        let pps_flags = ash::vk::native::StdVideoH265PpsFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoH265PpsFlags::new_bitfield_1(
                0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0,
            ),
        };

        let pps = ash::vk::native::StdVideoH265PictureParameterSet {
            flags: pps_flags,
            pps_pic_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            sps_video_parameter_set_id: 0,
            num_extra_slice_header_bits: 0,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            init_qp_minus26: 0,
            diff_cu_qp_delta_depth: 0,
            pps_cb_qp_offset: 0,
            pps_cr_qp_offset: 0,
            pps_beta_offset_div2: 0,
            pps_tc_offset_div2: 0,
            log2_parallel_merge_level_minus2: 0,
            log2_max_transform_skip_block_size_minus2: 0,
            diff_cu_chroma_qp_offset_depth: 0,
            chroma_qp_offset_list_len_minus1: 0,
            cb_qp_offset_list: [0; 6],
            cr_qp_offset_list: [0; 6],
            log2_sao_offset_scale_luma: 0,
            log2_sao_offset_scale_chroma: 0,
            pps_act_y_qp_offset_plus5: 0,
            pps_act_cb_qp_offset_plus5: 0,
            pps_act_cr_qp_offset_plus3: 0,
            pps_num_palette_predictor_initializers: 0,
            luma_bit_depth_entry_minus8: bit_depth_minus8,
            chroma_bit_depth_entry_minus8: bit_depth_minus8,
            num_tile_columns_minus1: 0,
            num_tile_rows_minus1: 0,
            reserved1: 0,
            reserved2: 0,
            column_width_minus1: [0; 19],
            row_height_minus1: [0; 21],
            reserved3: 0,
            pScalingLists: ptr::null(),
            pPredictorPaletteEntries: ptr::null(),
        };

        let vps_array = [vps];
        let sps_array = [sps_with_ptrs];
        let pps_array = [pps];

        let h265_add_info = vk::VideoEncodeH265SessionParametersAddInfoKHR::default()
            .std_vp_ss(&vps_array)
            .std_sp_ss(&sps_array)
            .std_pp_ss(&pps_array);

        let mut h265_params_create_info =
            vk::VideoEncodeH265SessionParametersCreateInfoKHR::default()
                .max_std_vps_count(1)
                .max_std_sps_count(1)
                .max_std_pps_count(1)
                .parameters_add_info(&h265_add_info);

        let mut quality_level_info = vk::VideoEncodeQualityLevelInfoKHR::default()
            .quality_level(common.config.encode_quality_level);

        let params_create_info = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(common.session)
            .push(&mut h265_params_create_info)
            .push(&mut quality_level_info);

        let mut session_params = vk::VideoSessionParametersKHR::null();
        let result = unsafe {
            (common
                .video_queue_fn
                .fp()
                .create_video_session_parameters_khr)(
                common.device().handle(),
                &params_create_info,
                ptr::null(),
                &mut session_params,
            )
        };
        if result != vk::Result::SUCCESS {
            return Err(PixelForgeError::SessionParametersCreation(format!(
                "{:?}",
                result
            )));
        }
        Ok(session_params)
    }

    /// Retrieve the encoded VPS/SPS/PPS NAL units to prepend to an IDR frame.
    pub(super) fn build_header(&self, common: &EncoderCommon) -> Result<Vec<u8>> {
        let mut h265_get_info = vk::VideoEncodeH265SessionParametersGetInfoKHR::default()
            .write_std_vps(true)
            .write_std_sps(true)
            .write_std_pps(true)
            .std_vps_id(0)
            .std_sps_id(0)
            .std_pps_id(0);
        let get_info = vk::VideoEncodeSessionParametersGetInfoKHR::default()
            .video_session_parameters(common.session_params)
            .push(&mut h265_get_info);
        let mut h265_feedback = vk::VideoEncodeH265SessionParametersFeedbackInfoKHR::default();
        let mut feedback =
            vk::VideoEncodeSessionParametersFeedbackInfoKHR::default().push(&mut h265_feedback);
        get_encoded_session_params(
            &common.context,
            &common.video_encode_fn,
            &get_info,
            &mut feedback,
        )
    }
}
