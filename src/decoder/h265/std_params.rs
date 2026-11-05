//! The `StdVideoH265*` parameter sets a decode session is created with, built
//! from the parsed VPS, SPS and PPS.

use ash::vk::native::{
    StdVideoH265DecPicBufMgr, StdVideoH265LongTermRefPicsSps, StdVideoH265PictureParameterSet,
    StdVideoH265ProfileTierLevel, StdVideoH265ScalingLists, StdVideoH265SequenceParameterSet,
    StdVideoH265SequenceParameterSetVui, StdVideoH265ShortTermRefPicSet,
    StdVideoH265VideoParameterSet,
};

use super::parser::{DecPicBufMgr, ProfileTierLevel, Vps};
use super::pps::Pps;
use super::rps::StRefPicSetSyntax;
use super::scaling::ScalingLists;
use super::sps::{Sps, Vui};

/// The three parameter sets, and everything they point at.
///
/// Each sub-structure is boxed, so the pointers to it stay valid when this
/// value moves. Keep it alive until the session parameters are created.
pub(super) struct StdParameterSets {
    pub vps: StdVideoH265VideoParameterSet,
    pub sps: StdVideoH265SequenceParameterSet,
    pub pps: StdVideoH265PictureParameterSet,
    _vps_ptl: Box<StdVideoH265ProfileTierLevel>,
    _vps_dpbm: Box<StdVideoH265DecPicBufMgr>,
    _sps_ptl: Box<StdVideoH265ProfileTierLevel>,
    _sps_dpbm: Box<StdVideoH265DecPicBufMgr>,
    _sps_scaling: Option<Box<StdVideoH265ScalingLists>>,
    _st_rps: Vec<StdVideoH265ShortTermRefPicSet>,
    _lt_rps: Box<StdVideoH265LongTermRefPicsSps>,
    _vui: Option<Box<StdVideoH265SequenceParameterSetVui>>,
    _pps_scaling: Option<Box<StdVideoH265ScalingLists>>,
}

// SAFETY: the raw pointers only point into the boxes this value owns.
unsafe impl Send for StdParameterSets {}

fn ptr<T>(value: &Option<Box<T>>) -> *const T {
    value.as_deref().map_or(std::ptr::null(), |v| v as *const T)
}

impl StdParameterSets {
    pub(super) fn new(vps: &Vps, sps: &Sps, pps: &Pps) -> Self {
        let vps_ptl = Box::new(profile_tier_level(&vps.profile_tier_level));
        let vps_dpbm = Box::new(dec_pic_buf_mgr(&vps.dec_pic_buf_mgr));
        let sps_ptl = Box::new(profile_tier_level(&sps.profile_tier_level));
        let sps_dpbm = Box::new(dec_pic_buf_mgr(&sps.dec_pic_buf_mgr));
        // With scaling lists enabled but none coded, the default lists apply,
        // and the driver is handed them explicitly.
        let sps_scaling = sps.scaling_list_enabled_flag.then(|| {
            Box::new(scaling_lists(
                sps.scaling_lists
                    .as_ref()
                    .unwrap_or(&ScalingLists::default()),
            ))
        });
        let st_rps: Vec<_> = sps
            .short_term_ref_pic_sets
            .iter()
            .map(|set| short_term_ref_pic_set(&set.syntax))
            .collect();
        let lt_rps = Box::new(long_term_ref_pics(sps));
        let vui = sps.vui.as_ref().map(|v| Box::new(vui(v)));
        let pps_scaling = pps
            .scaling_lists
            .as_ref()
            .filter(|_| pps.pps_scaling_list_data_present_flag)
            .map(|l| Box::new(scaling_lists(l)));

        let std_vps = video_parameter_set(vps, &vps_ptl, &vps_dpbm);
        let mut std_sps = sequence_parameter_set(sps);
        std_sps.pProfileTierLevel = &*sps_ptl;
        std_sps.pDecPicBufMgr = &*sps_dpbm;
        std_sps.pScalingLists = ptr(&sps_scaling);
        std_sps.pShortTermRefPicSet = if st_rps.is_empty() {
            std::ptr::null()
        } else {
            st_rps.as_ptr()
        };
        std_sps.pLongTermRefPicsSps = &*lt_rps;
        std_sps.pSequenceParameterSetVui = ptr(&vui);
        let mut std_pps = picture_parameter_set(pps, sps.sps_video_parameter_set_id);
        std_pps.pScalingLists = ptr(&pps_scaling);

        Self {
            vps: std_vps,
            sps: std_sps,
            pps: std_pps,
            _vps_ptl: vps_ptl,
            _vps_dpbm: vps_dpbm,
            _sps_ptl: sps_ptl,
            _sps_dpbm: sps_dpbm,
            _sps_scaling: sps_scaling,
            _st_rps: st_rps,
            _lt_rps: lt_rps,
            _vui: vui,
            _pps_scaling: pps_scaling,
        }
    }
}

fn profile_tier_level(ptl: &ProfileTierLevel) -> StdVideoH265ProfileTierLevel {
    // SAFETY: plain C data, for which all-zero is valid.
    let mut out: StdVideoH265ProfileTierLevel = unsafe { std::mem::zeroed() };
    out.flags
        .set_general_tier_flag(ptl.general_tier_flag as u32);
    out.flags
        .set_general_progressive_source_flag(ptl.general_progressive_source_flag as u32);
    out.flags
        .set_general_interlaced_source_flag(ptl.general_interlaced_source_flag as u32);
    out.flags
        .set_general_non_packed_constraint_flag(ptl.general_non_packed_constraint_flag as u32);
    out.flags
        .set_general_frame_only_constraint_flag(ptl.general_frame_only_constraint_flag as u32);
    // The Std profile enum uses the coded values.
    out.general_profile_idc = ptl.general_profile_idc as u32;
    out.general_level_idc = level_idc(ptl.general_level_idc);
    out
}

/// `StdVideoH265LevelIdc` for a coded `general_level_idc`, which is thirty
/// times the level.
fn level_idc(general_level_idc: u8) -> ash::vk::native::StdVideoH265LevelIdc {
    use ash::vk::native::*;
    match general_level_idc {
        0..=30 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_1_0,
        31..=60 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_0,
        61..=63 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_1,
        64..=90 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_0,
        91..=93 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_1,
        94..=120 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_0,
        121..=123 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_1,
        124..=150 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_0,
        151..=153 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_1,
        154..=156 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_2,
        157..=180 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_0,
        181..=183 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_1,
        _ => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_2,
    }
}

fn dec_pic_buf_mgr(m: &DecPicBufMgr) -> StdVideoH265DecPicBufMgr {
    StdVideoH265DecPicBufMgr {
        max_latency_increase_plus1: m.max_latency_increase_plus1,
        max_dec_pic_buffering_minus1: m.max_dec_pic_buffering_minus1,
        max_num_reorder_pics: m.max_num_reorder_pics,
    }
}

fn scaling_lists(l: &ScalingLists) -> StdVideoH265ScalingLists {
    StdVideoH265ScalingLists {
        ScalingList4x4: l.scaling_list_4x4,
        ScalingList8x8: l.scaling_list_8x8,
        ScalingList16x16: l.scaling_list_16x16,
        ScalingList32x32: l.scaling_list_32x32,
        ScalingListDCCoef16x16: l.scaling_list_dc_coef_16x16,
        ScalingListDCCoef32x32: l.scaling_list_dc_coef_32x32,
    }
}

fn short_term_ref_pic_set(s: &StRefPicSetSyntax) -> StdVideoH265ShortTermRefPicSet {
    // SAFETY: plain C data, for which all-zero is valid.
    let mut out: StdVideoH265ShortTermRefPicSet = unsafe { std::mem::zeroed() };
    out.flags
        .set_inter_ref_pic_set_prediction_flag(s.inter_ref_pic_set_prediction_flag as u32);
    out.flags.set_delta_rps_sign(s.delta_rps_sign as u32);
    out.delta_idx_minus1 = s.delta_idx_minus1;
    // Vulkan holds these flag arrays as 16 bits. Inter-RPS prediction can code
    // a 17th entry, which only a reference set of 16 pictures reaches.
    out.use_delta_flag = s.use_delta_flag as u16;
    out.abs_delta_rps_minus1 = s.abs_delta_rps_minus1 as u16;
    out.used_by_curr_pic_flag = s.used_by_curr_pic_flag as u16;
    out.used_by_curr_pic_s0_flag = s.used_by_curr_pic_s0_flag;
    out.used_by_curr_pic_s1_flag = s.used_by_curr_pic_s1_flag;
    out.num_negative_pics = s.num_negative_pics;
    out.num_positive_pics = s.num_positive_pics;
    out.delta_poc_s0_minus1 = s.delta_poc_s0_minus1;
    out.delta_poc_s1_minus1 = s.delta_poc_s1_minus1;
    out
}

fn long_term_ref_pics(sps: &Sps) -> StdVideoH265LongTermRefPicsSps {
    let mut lsb = [0u32; 32];
    for (dst, src) in lsb
        .iter_mut()
        .zip(&sps.long_term_ref_pics.lt_ref_pic_poc_lsb_sps)
    {
        *dst = *src;
    }
    StdVideoH265LongTermRefPicsSps {
        used_by_curr_pic_lt_sps_flag: sps.long_term_ref_pics.used_by_curr_pic_lt_sps_flag,
        lt_ref_pic_poc_lsb_sps: lsb,
    }
}

fn vui(v: &Vui) -> StdVideoH265SequenceParameterSetVui {
    // SAFETY: plain C data, for which all-zero is valid.
    let mut out: StdVideoH265SequenceParameterSetVui = unsafe { std::mem::zeroed() };
    let f = &mut out.flags;
    f.set_aspect_ratio_info_present_flag(v.aspect_ratio_info_present_flag as u32);
    f.set_overscan_info_present_flag(v.overscan_info_present_flag as u32);
    f.set_overscan_appropriate_flag(v.overscan_appropriate_flag as u32);
    f.set_video_signal_type_present_flag(v.video_signal_type_present_flag as u32);
    f.set_video_full_range_flag(v.video_full_range_flag as u32);
    f.set_colour_description_present_flag(v.colour_description_present_flag as u32);
    f.set_chroma_loc_info_present_flag(v.chroma_loc_info_present_flag as u32);
    f.set_neutral_chroma_indication_flag(v.neutral_chroma_indication_flag as u32);
    f.set_field_seq_flag(v.field_seq_flag as u32);
    f.set_frame_field_info_present_flag(v.frame_field_info_present_flag as u32);
    f.set_default_display_window_flag(v.default_display_window_flag as u32);
    f.set_vui_timing_info_present_flag(v.vui_timing_info_present_flag as u32);
    f.set_vui_poc_proportional_to_timing_flag(v.vui_poc_proportional_to_timing_flag as u32);
    // The HRD parameters are skipped when parsing, so none are handed over;
    // decoding does not use them.
    f.set_vui_hrd_parameters_present_flag(0);
    f.set_bitstream_restriction_flag(v.bitstream_restriction_flag as u32);
    f.set_tiles_fixed_structure_flag(v.tiles_fixed_structure_flag as u32);
    f.set_motion_vectors_over_pic_boundaries_flag(v.motion_vectors_over_pic_boundaries_flag as u32);
    f.set_restricted_ref_pic_lists_flag(v.restricted_ref_pic_lists_flag as u32);
    out.aspect_ratio_idc = v.aspect_ratio_idc as u32;
    out.sar_width = v.sar_width;
    out.sar_height = v.sar_height;
    out.video_format = v.video_format;
    out.colour_primaries = v.colour_primaries;
    out.transfer_characteristics = v.transfer_characteristics;
    out.matrix_coeffs = v.matrix_coeffs;
    out.chroma_sample_loc_type_top_field = v.chroma_sample_loc_type_top_field;
    out.chroma_sample_loc_type_bottom_field = v.chroma_sample_loc_type_bottom_field;
    out.def_disp_win_left_offset = v.def_disp_win_left_offset;
    out.def_disp_win_right_offset = v.def_disp_win_right_offset;
    out.def_disp_win_top_offset = v.def_disp_win_top_offset;
    out.def_disp_win_bottom_offset = v.def_disp_win_bottom_offset;
    out.vui_num_units_in_tick = v.vui_num_units_in_tick;
    out.vui_time_scale = v.vui_time_scale;
    out.vui_num_ticks_poc_diff_one_minus1 = v.vui_num_ticks_poc_diff_one_minus1;
    out.min_spatial_segmentation_idc = v.min_spatial_segmentation_idc;
    out.max_bytes_per_pic_denom = v.max_bytes_per_pic_denom;
    out.max_bits_per_min_cu_denom = v.max_bits_per_min_cu_denom;
    out.log2_max_mv_length_horizontal = v.log2_max_mv_length_horizontal;
    out.log2_max_mv_length_vertical = v.log2_max_mv_length_vertical;
    out
}

fn video_parameter_set(
    vps: &Vps,
    ptl: &StdVideoH265ProfileTierLevel,
    dpbm: &StdVideoH265DecPicBufMgr,
) -> StdVideoH265VideoParameterSet {
    // SAFETY: plain C data, for which all-zero is valid.
    let mut out: StdVideoH265VideoParameterSet = unsafe { std::mem::zeroed() };
    out.flags
        .set_vps_temporal_id_nesting_flag(vps.vps_temporal_id_nesting_flag as u32);
    out.flags.set_vps_sub_layer_ordering_info_present_flag(
        vps.vps_sub_layer_ordering_info_present_flag as u32,
    );
    if let Some(t) = &vps.timing {
        out.flags.set_vps_timing_info_present_flag(1);
        out.flags
            .set_vps_poc_proportional_to_timing_flag(t.poc_proportional_to_timing_flag as u32);
        out.vps_num_units_in_tick = t.num_units_in_tick;
        out.vps_time_scale = t.time_scale;
        out.vps_num_ticks_poc_diff_one_minus1 = t.num_ticks_poc_diff_one_minus1;
    }
    out.vps_video_parameter_set_id = vps.vps_video_parameter_set_id;
    out.vps_max_sub_layers_minus1 = vps.vps_max_sub_layers_minus1;
    out.pDecPicBufMgr = dpbm;
    out.pProfileTierLevel = ptl;
    out
}

fn sequence_parameter_set(sps: &Sps) -> StdVideoH265SequenceParameterSet {
    // SAFETY: plain C data, for which all-zero is valid.
    let mut out: StdVideoH265SequenceParameterSet = unsafe { std::mem::zeroed() };
    let f = &mut out.flags;
    let r = &sps.range_extension;
    f.set_sps_temporal_id_nesting_flag(sps.sps_temporal_id_nesting_flag as u32);
    f.set_separate_colour_plane_flag(sps.separate_colour_plane_flag as u32);
    f.set_conformance_window_flag(sps.conformance_window_flag as u32);
    f.set_sps_sub_layer_ordering_info_present_flag(
        sps.sps_sub_layer_ordering_info_present_flag as u32,
    );
    f.set_scaling_list_enabled_flag(sps.scaling_list_enabled_flag as u32);
    f.set_sps_scaling_list_data_present_flag(sps.sps_scaling_list_data_present_flag as u32);
    f.set_amp_enabled_flag(sps.amp_enabled_flag as u32);
    f.set_sample_adaptive_offset_enabled_flag(sps.sample_adaptive_offset_enabled_flag as u32);
    f.set_pcm_enabled_flag(sps.pcm_enabled_flag as u32);
    f.set_pcm_loop_filter_disabled_flag(sps.pcm_loop_filter_disabled_flag as u32);
    f.set_long_term_ref_pics_present_flag(sps.long_term_ref_pics_present_flag as u32);
    f.set_sps_temporal_mvp_enabled_flag(sps.sps_temporal_mvp_enabled_flag as u32);
    f.set_strong_intra_smoothing_enabled_flag(sps.strong_intra_smoothing_enabled_flag as u32);
    f.set_vui_parameters_present_flag(sps.vui.is_some() as u32);
    f.set_sps_extension_present_flag(sps.sps_extension_present_flag as u32);
    f.set_sps_range_extension_flag(sps.sps_range_extension_flag as u32);
    f.set_transform_skip_rotation_enabled_flag(r.transform_skip_rotation_enabled_flag as u32);
    f.set_transform_skip_context_enabled_flag(r.transform_skip_context_enabled_flag as u32);
    f.set_implicit_rdpcm_enabled_flag(r.implicit_rdpcm_enabled_flag as u32);
    f.set_explicit_rdpcm_enabled_flag(r.explicit_rdpcm_enabled_flag as u32);
    f.set_extended_precision_processing_flag(r.extended_precision_processing_flag as u32);
    f.set_intra_smoothing_disabled_flag(r.intra_smoothing_disabled_flag as u32);
    f.set_high_precision_offsets_enabled_flag(r.high_precision_offsets_enabled_flag as u32);
    f.set_persistent_rice_adaptation_enabled_flag(r.persistent_rice_adaptation_enabled_flag as u32);
    f.set_cabac_bypass_alignment_enabled_flag(r.cabac_bypass_alignment_enabled_flag as u32);
    f.set_sps_scc_extension_flag(sps.sps_scc_extension_flag as u32);
    out.chroma_format_idc = sps.chroma_format_idc as u32;
    out.pic_width_in_luma_samples = sps.pic_width_in_luma_samples;
    out.pic_height_in_luma_samples = sps.pic_height_in_luma_samples;
    out.sps_video_parameter_set_id = sps.sps_video_parameter_set_id;
    out.sps_max_sub_layers_minus1 = sps.sps_max_sub_layers_minus1;
    out.sps_seq_parameter_set_id = sps.sps_seq_parameter_set_id;
    out.bit_depth_luma_minus8 = sps.bit_depth_luma_minus8;
    out.bit_depth_chroma_minus8 = sps.bit_depth_chroma_minus8;
    out.log2_max_pic_order_cnt_lsb_minus4 = sps.log2_max_pic_order_cnt_lsb_minus4;
    out.log2_min_luma_coding_block_size_minus3 = sps.log2_min_luma_coding_block_size_minus3;
    out.log2_diff_max_min_luma_coding_block_size = sps.log2_diff_max_min_luma_coding_block_size;
    out.log2_min_luma_transform_block_size_minus2 = sps.log2_min_luma_transform_block_size_minus2;
    out.log2_diff_max_min_luma_transform_block_size =
        sps.log2_diff_max_min_luma_transform_block_size;
    out.max_transform_hierarchy_depth_inter = sps.max_transform_hierarchy_depth_inter;
    out.max_transform_hierarchy_depth_intra = sps.max_transform_hierarchy_depth_intra;
    out.num_short_term_ref_pic_sets = sps.short_term_ref_pic_sets.len() as u8;
    out.num_long_term_ref_pics_sps = sps.long_term_ref_pics.lt_ref_pic_poc_lsb_sps.len() as u8;
    out.pcm_sample_bit_depth_luma_minus1 = sps.pcm_sample_bit_depth_luma_minus1;
    out.pcm_sample_bit_depth_chroma_minus1 = sps.pcm_sample_bit_depth_chroma_minus1;
    out.log2_min_pcm_luma_coding_block_size_minus3 = sps.log2_min_pcm_luma_coding_block_size_minus3;
    out.log2_diff_max_min_pcm_luma_coding_block_size =
        sps.log2_diff_max_min_pcm_luma_coding_block_size;
    out.conf_win_left_offset = sps.conf_win_left_offset;
    out.conf_win_right_offset = sps.conf_win_right_offset;
    out.conf_win_top_offset = sps.conf_win_top_offset;
    out.conf_win_bottom_offset = sps.conf_win_bottom_offset;
    out
}

fn picture_parameter_set(pps: &Pps, vps_id: u8) -> StdVideoH265PictureParameterSet {
    // SAFETY: plain C data, for which all-zero is valid.
    let mut out: StdVideoH265PictureParameterSet = unsafe { std::mem::zeroed() };
    let f = &mut out.flags;
    let r = &pps.range_extension;
    f.set_dependent_slice_segments_enabled_flag(pps.dependent_slice_segments_enabled_flag as u32);
    f.set_output_flag_present_flag(pps.output_flag_present_flag as u32);
    f.set_sign_data_hiding_enabled_flag(pps.sign_data_hiding_enabled_flag as u32);
    f.set_cabac_init_present_flag(pps.cabac_init_present_flag as u32);
    f.set_constrained_intra_pred_flag(pps.constrained_intra_pred_flag as u32);
    f.set_transform_skip_enabled_flag(pps.transform_skip_enabled_flag as u32);
    f.set_cu_qp_delta_enabled_flag(pps.cu_qp_delta_enabled_flag as u32);
    f.set_pps_slice_chroma_qp_offsets_present_flag(
        pps.pps_slice_chroma_qp_offsets_present_flag as u32,
    );
    f.set_weighted_pred_flag(pps.weighted_pred_flag as u32);
    f.set_weighted_bipred_flag(pps.weighted_bipred_flag as u32);
    f.set_transquant_bypass_enabled_flag(pps.transquant_bypass_enabled_flag as u32);
    f.set_tiles_enabled_flag(pps.tiles_enabled_flag as u32);
    f.set_entropy_coding_sync_enabled_flag(pps.entropy_coding_sync_enabled_flag as u32);
    f.set_uniform_spacing_flag(pps.uniform_spacing_flag as u32);
    f.set_loop_filter_across_tiles_enabled_flag(pps.loop_filter_across_tiles_enabled_flag as u32);
    f.set_pps_loop_filter_across_slices_enabled_flag(
        pps.pps_loop_filter_across_slices_enabled_flag as u32,
    );
    f.set_deblocking_filter_control_present_flag(pps.deblocking_filter_control_present_flag as u32);
    f.set_deblocking_filter_override_enabled_flag(
        pps.deblocking_filter_override_enabled_flag as u32,
    );
    f.set_pps_deblocking_filter_disabled_flag(pps.pps_deblocking_filter_disabled_flag as u32);
    f.set_pps_scaling_list_data_present_flag(pps.pps_scaling_list_data_present_flag as u32);
    f.set_lists_modification_present_flag(pps.lists_modification_present_flag as u32);
    f.set_slice_segment_header_extension_present_flag(
        pps.slice_segment_header_extension_present_flag as u32,
    );
    f.set_pps_extension_present_flag(pps.pps_extension_present_flag as u32);
    f.set_cross_component_prediction_enabled_flag(r.cross_component_prediction_enabled_flag as u32);
    f.set_chroma_qp_offset_list_enabled_flag(r.chroma_qp_offset_list_enabled_flag as u32);
    f.set_pps_range_extension_flag(pps.pps_range_extension_flag as u32);
    out.pps_pic_parameter_set_id = pps.pps_pic_parameter_set_id;
    out.pps_seq_parameter_set_id = pps.pps_seq_parameter_set_id;
    out.sps_video_parameter_set_id = vps_id;
    out.num_extra_slice_header_bits = pps.num_extra_slice_header_bits;
    out.num_ref_idx_l0_default_active_minus1 = pps.num_ref_idx_l0_default_active_minus1;
    out.num_ref_idx_l1_default_active_minus1 = pps.num_ref_idx_l1_default_active_minus1;
    out.init_qp_minus26 = pps.init_qp_minus26;
    out.diff_cu_qp_delta_depth = pps.diff_cu_qp_delta_depth;
    out.pps_cb_qp_offset = pps.pps_cb_qp_offset;
    out.pps_cr_qp_offset = pps.pps_cr_qp_offset;
    out.pps_beta_offset_div2 = pps.pps_beta_offset_div2;
    out.pps_tc_offset_div2 = pps.pps_tc_offset_div2;
    out.log2_parallel_merge_level_minus2 = pps.log2_parallel_merge_level_minus2;
    out.log2_max_transform_skip_block_size_minus2 = r.log2_max_transform_skip_block_size_minus2;
    out.diff_cu_chroma_qp_offset_depth = r.diff_cu_chroma_qp_offset_depth;
    out.chroma_qp_offset_list_len_minus1 = r.chroma_qp_offset_list_len_minus1;
    out.cb_qp_offset_list = r.cb_qp_offset_list;
    out.cr_qp_offset_list = r.cr_qp_offset_list;
    out.log2_sao_offset_scale_luma = r.log2_sao_offset_scale_luma;
    out.log2_sao_offset_scale_chroma = r.log2_sao_offset_scale_chroma;
    out.num_tile_columns_minus1 = pps.num_tile_columns_minus1;
    out.num_tile_rows_minus1 = pps.num_tile_rows_minus1;
    for (dst, src) in out
        .column_width_minus1
        .iter_mut()
        .zip(&pps.column_width_minus1)
    {
        *dst = *src;
    }
    for (dst, src) in out.row_height_minus1.iter_mut().zip(&pps.row_height_minus1) {
        *dst = *src;
    }
    out
}
