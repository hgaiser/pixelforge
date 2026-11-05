//! The H.265 sequence parameter set (clause 7.3.2.2) and its VUI (Annex E).
//!
//! Field names follow the standard, which are also the names
//! `StdVideoH265SequenceParameterSet` and its sub-structures use, so filling
//! the Vulkan structures is a field-by-field copy.

use super::parser::{
    DecPicBufMgr, MAX_SUB_LAYERS, ProfileTierLevel, Reader, invalid, parse_dec_pic_buf_mgr,
    parse_profile_tier_level, skip_hrd_parameters,
};
use super::rps::{StRefPicSet, parse_st_ref_pic_set};
use super::scaling::{ScalingLists, parse_scaling_list_data};
use crate::error::Result;

/// Most short-term reference picture sets an SPS may code.
pub(crate) const MAX_SPS_ST_REF_PIC_SETS: usize = 64;
/// Most long-term reference pictures an SPS may list.
pub(crate) const MAX_SPS_LT_REF_PICS: usize = 32;

/// `vui_parameters()` (clause E.2.1), laid out like
/// `StdVideoH265SequenceParameterSetVui`.
///
/// HRD parameters are parsed past but not kept; `vui_hrd_parameters_present_flag`
/// records whether there were any.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Vui {
    pub aspect_ratio_info_present_flag: bool,
    pub aspect_ratio_idc: u8,
    pub sar_width: u16,
    pub sar_height: u16,
    pub overscan_info_present_flag: bool,
    pub overscan_appropriate_flag: bool,
    pub video_signal_type_present_flag: bool,
    pub video_format: u8,
    pub video_full_range_flag: bool,
    pub colour_description_present_flag: bool,
    pub colour_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coeffs: u8,
    pub chroma_loc_info_present_flag: bool,
    pub chroma_sample_loc_type_top_field: u8,
    pub chroma_sample_loc_type_bottom_field: u8,
    pub neutral_chroma_indication_flag: bool,
    pub field_seq_flag: bool,
    pub frame_field_info_present_flag: bool,
    pub default_display_window_flag: bool,
    pub def_disp_win_left_offset: u16,
    pub def_disp_win_right_offset: u16,
    pub def_disp_win_top_offset: u16,
    pub def_disp_win_bottom_offset: u16,
    pub vui_timing_info_present_flag: bool,
    pub vui_num_units_in_tick: u32,
    pub vui_time_scale: u32,
    pub vui_poc_proportional_to_timing_flag: bool,
    pub vui_num_ticks_poc_diff_one_minus1: u32,
    pub vui_hrd_parameters_present_flag: bool,
    pub bitstream_restriction_flag: bool,
    pub tiles_fixed_structure_flag: bool,
    pub motion_vectors_over_pic_boundaries_flag: bool,
    pub restricted_ref_pic_lists_flag: bool,
    pub min_spatial_segmentation_idc: u16,
    pub max_bytes_per_pic_denom: u8,
    pub max_bits_per_min_cu_denom: u8,
    pub log2_max_mv_length_horizontal: u8,
    pub log2_max_mv_length_vertical: u8,
}

/// Parse `vui_parameters()`.
fn parse_vui(r: &mut Reader, sps_max_sub_layers_minus1: u8) -> Result<Vui> {
    let mut vui = Vui {
        aspect_ratio_info_present_flag: r.flag()?,
        ..Default::default()
    };
    if vui.aspect_ratio_info_present_flag {
        vui.aspect_ratio_idc = r.bits(8)? as u8;
        // EXTENDED_SAR.
        if vui.aspect_ratio_idc == 255 {
            vui.sar_width = r.bits(16)? as u16;
            vui.sar_height = r.bits(16)? as u16;
        }
    }
    vui.overscan_info_present_flag = r.flag()?;
    if vui.overscan_info_present_flag {
        vui.overscan_appropriate_flag = r.flag()?;
    }
    vui.video_signal_type_present_flag = r.flag()?;
    if vui.video_signal_type_present_flag {
        vui.video_format = r.bits(3)? as u8;
        vui.video_full_range_flag = r.flag()?;
        vui.colour_description_present_flag = r.flag()?;
        if vui.colour_description_present_flag {
            vui.colour_primaries = r.bits(8)? as u8;
            vui.transfer_characteristics = r.bits(8)? as u8;
            vui.matrix_coeffs = r.bits(8)? as u8;
        }
    }
    vui.chroma_loc_info_present_flag = r.flag()?;
    if vui.chroma_loc_info_present_flag {
        vui.chroma_sample_loc_type_top_field = r.ue_max(5, "chroma_sample_loc_type")? as u8;
        vui.chroma_sample_loc_type_bottom_field = r.ue_max(5, "chroma_sample_loc_type")? as u8;
    }
    vui.neutral_chroma_indication_flag = r.flag()?;
    vui.field_seq_flag = r.flag()?;
    vui.frame_field_info_present_flag = r.flag()?;
    vui.default_display_window_flag = r.flag()?;
    if vui.default_display_window_flag {
        vui.def_disp_win_left_offset = r.ue_max(u16::MAX as u32, "def_disp_win offset")? as u16;
        vui.def_disp_win_right_offset = r.ue_max(u16::MAX as u32, "def_disp_win offset")? as u16;
        vui.def_disp_win_top_offset = r.ue_max(u16::MAX as u32, "def_disp_win offset")? as u16;
        vui.def_disp_win_bottom_offset = r.ue_max(u16::MAX as u32, "def_disp_win offset")? as u16;
    }
    vui.vui_timing_info_present_flag = r.flag()?;
    if vui.vui_timing_info_present_flag {
        vui.vui_num_units_in_tick = r.bits(32)?;
        vui.vui_time_scale = r.bits(32)?;
        vui.vui_poc_proportional_to_timing_flag = r.flag()?;
        if vui.vui_poc_proportional_to_timing_flag {
            vui.vui_num_ticks_poc_diff_one_minus1 = r.ue()?;
        }
        vui.vui_hrd_parameters_present_flag = r.flag()?;
        if vui.vui_hrd_parameters_present_flag {
            skip_hrd_parameters(r, true, sps_max_sub_layers_minus1)?;
        }
    }
    vui.bitstream_restriction_flag = r.flag()?;
    if vui.bitstream_restriction_flag {
        vui.tiles_fixed_structure_flag = r.flag()?;
        vui.motion_vectors_over_pic_boundaries_flag = r.flag()?;
        vui.restricted_ref_pic_lists_flag = r.flag()?;
        vui.min_spatial_segmentation_idc = r.ue_max(4095, "min_spatial_segmentation_idc")? as u16;
        vui.max_bytes_per_pic_denom = r.ue_max(16, "max_bytes_per_pic_denom")? as u8;
        vui.max_bits_per_min_cu_denom = r.ue_max(16, "max_bits_per_min_cu_denom")? as u8;
        vui.log2_max_mv_length_horizontal = r.ue_max(16, "log2_max_mv_length_horizontal")? as u8;
        vui.log2_max_mv_length_vertical = r.ue_max(15, "log2_max_mv_length_vertical")? as u8;
    }
    Ok(vui)
}

/// `sps_range_extension()` (clause 7.3.2.2.2).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SpsRangeExtension {
    pub transform_skip_rotation_enabled_flag: bool,
    pub transform_skip_context_enabled_flag: bool,
    pub implicit_rdpcm_enabled_flag: bool,
    pub explicit_rdpcm_enabled_flag: bool,
    pub extended_precision_processing_flag: bool,
    pub intra_smoothing_disabled_flag: bool,
    pub high_precision_offsets_enabled_flag: bool,
    pub persistent_rice_adaptation_enabled_flag: bool,
    pub cabac_bypass_alignment_enabled_flag: bool,
}

/// Long-term reference pictures listed in the SPS, as
/// `StdVideoH265LongTermRefPicsSps` carries them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct LongTermRefPicsSps {
    /// `lt_ref_pic_poc_lsb_sps[i]`, `num_long_term_ref_pics_sps` entries.
    pub lt_ref_pic_poc_lsb_sps: Vec<u32>,
    /// `used_by_curr_pic_lt_sps_flag[i]`, bit `i`.
    pub used_by_curr_pic_lt_sps_flag: u32,
}

/// Parsed H.265 sequence parameter set.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Sps {
    pub sps_video_parameter_set_id: u8,
    pub sps_max_sub_layers_minus1: u8,
    pub sps_temporal_id_nesting_flag: bool,
    pub profile_tier_level: ProfileTierLevel,
    pub sps_seq_parameter_set_id: u8,
    pub chroma_format_idc: u8,
    pub separate_colour_plane_flag: bool,
    pub pic_width_in_luma_samples: u32,
    pub pic_height_in_luma_samples: u32,
    pub conformance_window_flag: bool,
    pub conf_win_left_offset: u32,
    pub conf_win_right_offset: u32,
    pub conf_win_top_offset: u32,
    pub conf_win_bottom_offset: u32,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    pub sps_sub_layer_ordering_info_present_flag: bool,
    /// Filled for every sub-layer up to `sps_max_sub_layers_minus1`.
    pub dec_pic_buf_mgr: DecPicBufMgr,
    pub log2_min_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_luma_coding_block_size: u8,
    pub log2_min_luma_transform_block_size_minus2: u8,
    pub log2_diff_max_min_luma_transform_block_size: u8,
    pub max_transform_hierarchy_depth_inter: u8,
    pub max_transform_hierarchy_depth_intra: u8,
    pub scaling_list_enabled_flag: bool,
    pub sps_scaling_list_data_present_flag: bool,
    /// The lists in effect for the sequence: the coded ones when
    /// `sps_scaling_list_data_present_flag` is set, the defaults when only
    /// `scaling_list_enabled_flag` is, `None` when scaling lists are off (flat
    /// scaling). A PPS may override them.
    pub scaling_lists: Option<ScalingLists>,
    pub amp_enabled_flag: bool,
    pub sample_adaptive_offset_enabled_flag: bool,
    pub pcm_enabled_flag: bool,
    pub pcm_sample_bit_depth_luma_minus1: u8,
    pub pcm_sample_bit_depth_chroma_minus1: u8,
    pub log2_min_pcm_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_pcm_luma_coding_block_size: u8,
    pub pcm_loop_filter_disabled_flag: bool,
    /// The `num_short_term_ref_pic_sets` sets coded in the SPS.
    pub short_term_ref_pic_sets: Vec<StRefPicSet>,
    pub long_term_ref_pics_present_flag: bool,
    pub long_term_ref_pics: LongTermRefPicsSps,
    pub sps_temporal_mvp_enabled_flag: bool,
    pub strong_intra_smoothing_enabled_flag: bool,
    pub vui_parameters_present_flag: bool,
    pub vui: Option<Vui>,
    pub sps_extension_present_flag: bool,
    pub sps_range_extension_flag: bool,
    pub sps_multilayer_extension_flag: bool,
    pub sps_3d_extension_flag: bool,
    pub sps_scc_extension_flag: bool,
    pub sps_extension_4bits: u8,
    pub range_extension: SpsRangeExtension,
}

impl Sps {
    /// `num_short_term_ref_pic_sets`.
    pub fn num_short_term_ref_pic_sets(&self) -> usize {
        self.short_term_ref_pic_sets.len()
    }

    /// `num_long_term_ref_pics_sps`.
    pub fn num_long_term_ref_pics_sps(&self) -> usize {
        self.long_term_ref_pics.lt_ref_pic_poc_lsb_sps.len()
    }

    /// `MaxPicOrderCntLsb`.
    pub fn max_pic_order_cnt_lsb(&self) -> u32 {
        1 << (self.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4)
    }

    /// `CtbLog2SizeY`.
    pub fn ctb_log2_size(&self) -> u32 {
        self.log2_min_luma_coding_block_size_minus3 as u32
            + 3
            + self.log2_diff_max_min_luma_coding_block_size as u32
    }

    /// `PicWidthInCtbsY`.
    pub fn pic_width_in_ctbs(&self) -> u32 {
        self.pic_width_in_luma_samples
            .div_ceil(1 << self.ctb_log2_size())
    }

    /// `PicHeightInCtbsY`.
    pub fn pic_height_in_ctbs(&self) -> u32 {
        self.pic_height_in_luma_samples
            .div_ceil(1 << self.ctb_log2_size())
    }

    /// `PicSizeInCtbsY`.
    pub fn pic_size_in_ctbs(&self) -> u32 {
        self.pic_width_in_ctbs() * self.pic_height_in_ctbs()
    }

    /// `ChromaArrayType`: 0 for monochrome or separately coded colour planes.
    pub fn chroma_array_type(&self) -> u8 {
        if self.separate_colour_plane_flag {
            0
        } else {
            self.chroma_format_idc
        }
    }

    /// `HighestTid` when every sub-layer is decoded.
    pub fn highest_tid(&self) -> usize {
        self.sps_max_sub_layers_minus1 as usize
    }

    /// Pictures the DPB must hold (`sps_max_dec_pic_buffering_minus1 + 1` of
    /// the highest sub-layer), the current picture included.
    pub fn max_dec_pic_buffering(&self) -> usize {
        self.dec_pic_buf_mgr.max_dec_pic_buffering_minus1[self.highest_tid()] as usize + 1
    }

    /// `sps_max_num_reorder_pics` of the highest sub-layer: how many pictures
    /// can precede any picture in decoding order and follow it in output
    /// order, which is the depth a reorder buffer needs.
    pub fn reorder_depth(&self) -> usize {
        self.dec_pic_buf_mgr.max_num_reorder_pics[self.highest_tid()] as usize
    }

    /// `SpsMaxLatencyPictures` of the highest sub-layer, or `None` when
    /// `sps_max_latency_increase_plus1` is 0 (no limit). The output process
    /// uses only the reorder depth, so this is for tests.
    #[cfg(test)]
    pub fn max_latency_pictures(&self) -> Option<usize> {
        let tid = self.highest_tid();
        let plus1 = self.dec_pic_buf_mgr.max_latency_increase_plus1[tid] as usize;
        (plus1 != 0).then(|| self.dec_pic_buf_mgr.max_num_reorder_pics[tid] as usize + plus1 - 1)
    }

    /// Display (conformance-window cropped) dimensions.
    ///
    /// The offsets are in chroma sample units (clause 7.4.3.2.1).
    pub fn display_dimensions(&self) -> (u32, u32) {
        let (sub_width_c, sub_height_c) = match self.chroma_array_type() {
            1 => (2, 2),
            2 => (2, 1),
            _ => (1, 1),
        };
        let mut width = self.pic_width_in_luma_samples;
        let mut height = self.pic_height_in_luma_samples;
        if self.conformance_window_flag {
            width = width.saturating_sub(
                sub_width_c * (self.conf_win_left_offset + self.conf_win_right_offset),
            );
            height = height.saturating_sub(
                sub_height_c * (self.conf_win_top_offset + self.conf_win_bottom_offset),
            );
        }
        (width, height)
    }
}

/// Parse an SPS NAL payload (EBSP, after the 2-byte NAL header).
pub(crate) fn parse_sps(payload: &[u8]) -> Result<Sps> {
    let mut r = Reader::new(payload);
    let mut sps = Sps {
        sps_video_parameter_set_id: r.bits(4)? as u8,
        sps_max_sub_layers_minus1: r.bits(3)? as u8,
        sps_temporal_id_nesting_flag: r.flag()?,
        ..Default::default()
    };
    if sps.sps_max_sub_layers_minus1 as usize >= MAX_SUB_LAYERS {
        return Err(invalid("SPS: sps_max_sub_layers_minus1 is 7"));
    }
    sps.profile_tier_level = parse_profile_tier_level(&mut r, sps.sps_max_sub_layers_minus1)?;
    sps.sps_seq_parameter_set_id = r.ue_max(15, "sps_seq_parameter_set_id")? as u8;
    sps.chroma_format_idc = r.ue_max(3, "chroma_format_idc")? as u8;
    if sps.chroma_format_idc == 3 {
        sps.separate_colour_plane_flag = r.flag()?;
    }
    sps.pic_width_in_luma_samples = r.ue_max(16888, "pic_width_in_luma_samples")?;
    sps.pic_height_in_luma_samples = r.ue_max(16888, "pic_height_in_luma_samples")?;
    if sps.pic_width_in_luma_samples == 0 || sps.pic_height_in_luma_samples == 0 {
        return Err(invalid("SPS: zero picture size"));
    }
    sps.conformance_window_flag = r.flag()?;
    if sps.conformance_window_flag {
        sps.conf_win_left_offset = r.ue()?;
        sps.conf_win_right_offset = r.ue()?;
        sps.conf_win_top_offset = r.ue()?;
        sps.conf_win_bottom_offset = r.ue()?;
    }
    sps.bit_depth_luma_minus8 = r.ue_max(8, "bit_depth_luma_minus8")? as u8;
    sps.bit_depth_chroma_minus8 = r.ue_max(8, "bit_depth_chroma_minus8")? as u8;
    sps.log2_max_pic_order_cnt_lsb_minus4 =
        r.ue_max(12, "log2_max_pic_order_cnt_lsb_minus4")? as u8;
    sps.sps_sub_layer_ordering_info_present_flag = r.flag()?;
    sps.dec_pic_buf_mgr = parse_dec_pic_buf_mgr(
        &mut r,
        sps.sps_max_sub_layers_minus1,
        sps.sps_sub_layer_ordering_info_present_flag,
    )?;

    sps.log2_min_luma_coding_block_size_minus3 =
        r.ue_max(3, "log2_min_luma_coding_block_size_minus3")? as u8;
    sps.log2_diff_max_min_luma_coding_block_size =
        r.ue_max(3, "log2_diff_max_min_luma_coding_block_size")? as u8;
    if sps.ctb_log2_size() > 6 {
        return Err(invalid("SPS: CTB larger than 64x64"));
    }
    let min_cb = 1u32 << (sps.log2_min_luma_coding_block_size_minus3 + 3);
    if !sps.pic_width_in_luma_samples.is_multiple_of(min_cb)
        || !sps.pic_height_in_luma_samples.is_multiple_of(min_cb)
    {
        return Err(invalid(
            "SPS: picture size is not a multiple of the minimum coding block",
        ));
    }
    sps.log2_min_luma_transform_block_size_minus2 =
        r.ue_max(3, "log2_min_luma_transform_block_size_minus2")? as u8;
    sps.log2_diff_max_min_luma_transform_block_size =
        r.ue_max(3, "log2_diff_max_min_luma_transform_block_size")? as u8;
    sps.max_transform_hierarchy_depth_inter =
        r.ue_max(4, "max_transform_hierarchy_depth_inter")? as u8;
    sps.max_transform_hierarchy_depth_intra =
        r.ue_max(4, "max_transform_hierarchy_depth_intra")? as u8;

    sps.scaling_list_enabled_flag = r.flag()?;
    if sps.scaling_list_enabled_flag {
        sps.sps_scaling_list_data_present_flag = r.flag()?;
        sps.scaling_lists = Some(if sps.sps_scaling_list_data_present_flag {
            parse_scaling_list_data(&mut r)?
        } else {
            ScalingLists::default()
        });
    }

    sps.amp_enabled_flag = r.flag()?;
    sps.sample_adaptive_offset_enabled_flag = r.flag()?;
    sps.pcm_enabled_flag = r.flag()?;
    if sps.pcm_enabled_flag {
        sps.pcm_sample_bit_depth_luma_minus1 = r.bits(4)? as u8;
        sps.pcm_sample_bit_depth_chroma_minus1 = r.bits(4)? as u8;
        sps.log2_min_pcm_luma_coding_block_size_minus3 =
            r.ue_max(2, "log2_min_pcm_luma_coding_block_size_minus3")? as u8;
        sps.log2_diff_max_min_pcm_luma_coding_block_size =
            r.ue_max(2, "log2_diff_max_min_pcm_luma_coding_block_size")? as u8;
        sps.pcm_loop_filter_disabled_flag = r.flag()?;
    }

    let num_sets = r.ue_max(
        MAX_SPS_ST_REF_PIC_SETS as u32,
        "num_short_term_ref_pic_sets",
    )? as usize;
    let max_refs = sps.max_dec_pic_buffering() - 1;
    for i in 0..num_sets {
        let set =
            parse_st_ref_pic_set(&mut r, i, num_sets, &sps.short_term_ref_pic_sets, max_refs)?;
        sps.short_term_ref_pic_sets.push(set);
    }

    sps.long_term_ref_pics_present_flag = r.flag()?;
    if sps.long_term_ref_pics_present_flag {
        let count = r.ue_max(MAX_SPS_LT_REF_PICS as u32, "num_long_term_ref_pics_sps")?;
        let lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
        for i in 0..count {
            sps.long_term_ref_pics
                .lt_ref_pic_poc_lsb_sps
                .push(r.bits(lsb_bits)?);
            sps.long_term_ref_pics.used_by_curr_pic_lt_sps_flag |= (r.flag()? as u32) << i;
        }
    }
    sps.sps_temporal_mvp_enabled_flag = r.flag()?;
    sps.strong_intra_smoothing_enabled_flag = r.flag()?;
    sps.vui_parameters_present_flag = r.flag()?;
    if sps.vui_parameters_present_flag {
        sps.vui = Some(parse_vui(&mut r, sps.sps_max_sub_layers_minus1)?);
    }

    sps.sps_extension_present_flag = r.flag()?;
    if sps.sps_extension_present_flag {
        sps.sps_range_extension_flag = r.flag()?;
        sps.sps_multilayer_extension_flag = r.flag()?;
        sps.sps_3d_extension_flag = r.flag()?;
        sps.sps_scc_extension_flag = r.flag()?;
        sps.sps_extension_4bits = r.bits(4)? as u8;
    }
    if sps.sps_range_extension_flag {
        sps.range_extension = SpsRangeExtension {
            transform_skip_rotation_enabled_flag: r.flag()?,
            transform_skip_context_enabled_flag: r.flag()?,
            implicit_rdpcm_enabled_flag: r.flag()?,
            explicit_rdpcm_enabled_flag: r.flag()?,
            extended_precision_processing_flag: r.flag()?,
            intra_smoothing_disabled_flag: r.flag()?,
            high_precision_offsets_enabled_flag: r.flag()?,
            persistent_rice_adaptation_enabled_flag: r.flag()?,
            cabac_bypass_alignment_enabled_flag: r.flag()?,
        };
    }
    // The multilayer, 3D and SCC extensions follow. Nothing after them matters
    // to a base-layer decoder, and their flags are recorded above, so parsing
    // stops here.
    Ok(sps)
}
