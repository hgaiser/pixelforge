//! The H.265 picture parameter set (clause 7.3.2.3).
//!
//! Field names follow the standard, which are also the names
//! `StdVideoH265PictureParameterSet` uses.

use super::parser::Reader;
use super::scaling::{ScalingLists, parse_scaling_list_data};
use crate::error::Result;

/// Tile columns `StdVideoH265PictureParameterSet::column_width_minus1` holds.
pub(crate) const MAX_TILE_COLUMNS: usize = 20;
/// Tile rows `StdVideoH265PictureParameterSet::row_height_minus1` holds.
pub(crate) const MAX_TILE_ROWS: usize = 22;

/// `pps_range_extension()` (clause 7.3.2.3.2).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PpsRangeExtension {
    pub log2_max_transform_skip_block_size_minus2: u8,
    pub cross_component_prediction_enabled_flag: bool,
    pub chroma_qp_offset_list_enabled_flag: bool,
    pub diff_cu_chroma_qp_offset_depth: u8,
    pub chroma_qp_offset_list_len_minus1: u8,
    pub cb_qp_offset_list: [i8; 6],
    pub cr_qp_offset_list: [i8; 6],
    pub log2_sao_offset_scale_luma: u8,
    pub log2_sao_offset_scale_chroma: u8,
}

/// Parsed H.265 picture parameter set.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Pps {
    pub pps_pic_parameter_set_id: u8,
    pub pps_seq_parameter_set_id: u8,
    pub dependent_slice_segments_enabled_flag: bool,
    pub output_flag_present_flag: bool,
    pub num_extra_slice_header_bits: u8,
    pub sign_data_hiding_enabled_flag: bool,
    pub cabac_init_present_flag: bool,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub init_qp_minus26: i8,
    pub constrained_intra_pred_flag: bool,
    pub transform_skip_enabled_flag: bool,
    pub cu_qp_delta_enabled_flag: bool,
    pub diff_cu_qp_delta_depth: u8,
    pub pps_cb_qp_offset: i8,
    pub pps_cr_qp_offset: i8,
    pub pps_slice_chroma_qp_offsets_present_flag: bool,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_flag: bool,
    pub transquant_bypass_enabled_flag: bool,
    pub tiles_enabled_flag: bool,
    pub entropy_coding_sync_enabled_flag: bool,
    pub num_tile_columns_minus1: u8,
    pub num_tile_rows_minus1: u8,
    pub uniform_spacing_flag: bool,
    /// `column_width_minus1[i]`, coded only without uniform spacing; see
    /// `Pps::tile_column_widths` for the widths either way.
    pub column_width_minus1: Vec<u16>,
    /// `row_height_minus1[i]`, coded only without uniform spacing.
    pub row_height_minus1: Vec<u16>,
    pub loop_filter_across_tiles_enabled_flag: bool,
    pub pps_loop_filter_across_slices_enabled_flag: bool,
    pub deblocking_filter_control_present_flag: bool,
    pub deblocking_filter_override_enabled_flag: bool,
    pub pps_deblocking_filter_disabled_flag: bool,
    pub pps_beta_offset_div2: i8,
    pub pps_tc_offset_div2: i8,
    pub pps_scaling_list_data_present_flag: bool,
    /// The coded lists when `pps_scaling_list_data_present_flag` is set; they
    /// replace the SPS's for pictures using this PPS.
    pub scaling_lists: Option<ScalingLists>,
    pub lists_modification_present_flag: bool,
    pub log2_parallel_merge_level_minus2: u8,
    pub slice_segment_header_extension_present_flag: bool,
    pub pps_extension_present_flag: bool,
    pub pps_range_extension_flag: bool,
    pub pps_multilayer_extension_flag: bool,
    pub pps_3d_extension_flag: bool,
    pub pps_scc_extension_flag: bool,
    pub pps_extension_4bits: u8,
    pub range_extension: PpsRangeExtension,
}

// The driver takes the coded tile syntax; the derived sizes are for tests.
#[cfg(test)]
impl Pps {
    /// Tile column widths in CTBs (equation 6-3), for a picture
    /// `pic_width_in_ctbs` wide. One column when tiles are off.
    pub fn tile_column_widths(&self, pic_width_in_ctbs: u32) -> Vec<u32> {
        tile_sizes(
            self.tiles_enabled_flag,
            self.uniform_spacing_flag,
            self.num_tile_columns_minus1 as u32,
            &self.column_width_minus1,
            pic_width_in_ctbs,
        )
    }
}

/// Equations 6-3 and 6-4, which have the same shape for columns and rows.
#[cfg(test)]
fn tile_sizes(
    tiles: bool,
    uniform: bool,
    count_minus1: u32,
    explicit_minus1: &[u16],
    total: u32,
) -> Vec<u32> {
    if !tiles {
        return vec![total];
    }
    let count = count_minus1 + 1;
    if uniform {
        return (0..count)
            .map(|i| ((i + 1) * total) / count - (i * total) / count)
            .collect();
    }
    let mut sizes: Vec<u32> = explicit_minus1.iter().map(|w| *w as u32 + 1).collect();
    let used: u32 = sizes.iter().sum();
    // The last one takes whatever is left.
    sizes.push(total.saturating_sub(used));
    sizes
}

/// Parse a PPS NAL payload (EBSP, after the 2-byte NAL header).
pub(crate) fn parse_pps(payload: &[u8]) -> Result<Pps> {
    let mut r = Reader::new(payload);
    let mut pps = Pps {
        pps_pic_parameter_set_id: r.ue_max(63, "pps_pic_parameter_set_id")? as u8,
        pps_seq_parameter_set_id: r.ue_max(15, "pps_seq_parameter_set_id")? as u8,
        dependent_slice_segments_enabled_flag: r.flag()?,
        output_flag_present_flag: r.flag()?,
        num_extra_slice_header_bits: r.bits(3)? as u8,
        sign_data_hiding_enabled_flag: r.flag()?,
        cabac_init_present_flag: r.flag()?,
        ..Default::default()
    };
    pps.num_ref_idx_l0_default_active_minus1 =
        r.ue_max(14, "num_ref_idx_l0_default_active_minus1")? as u8;
    pps.num_ref_idx_l1_default_active_minus1 =
        r.ue_max(14, "num_ref_idx_l1_default_active_minus1")? as u8;
    // The lower bound depends on the bit depth (-(26 + QpBdOffsetY)), which
    // the PPS cannot see; -26 - 48 covers 16-bit video.
    pps.init_qp_minus26 = r.se_range(-74, 25, "init_qp_minus26")? as i8;
    pps.constrained_intra_pred_flag = r.flag()?;
    pps.transform_skip_enabled_flag = r.flag()?;
    pps.cu_qp_delta_enabled_flag = r.flag()?;
    if pps.cu_qp_delta_enabled_flag {
        pps.diff_cu_qp_delta_depth = r.ue_max(3, "diff_cu_qp_delta_depth")? as u8;
    }
    pps.pps_cb_qp_offset = r.se_range(-12, 12, "pps_cb_qp_offset")? as i8;
    pps.pps_cr_qp_offset = r.se_range(-12, 12, "pps_cr_qp_offset")? as i8;
    pps.pps_slice_chroma_qp_offsets_present_flag = r.flag()?;
    pps.weighted_pred_flag = r.flag()?;
    pps.weighted_bipred_flag = r.flag()?;
    pps.transquant_bypass_enabled_flag = r.flag()?;
    pps.tiles_enabled_flag = r.flag()?;
    pps.entropy_coding_sync_enabled_flag = r.flag()?;

    if pps.tiles_enabled_flag {
        pps.num_tile_columns_minus1 =
            r.ue_max(MAX_TILE_COLUMNS as u32 - 1, "num_tile_columns_minus1")? as u8;
        pps.num_tile_rows_minus1 =
            r.ue_max(MAX_TILE_ROWS as u32 - 1, "num_tile_rows_minus1")? as u8;
        pps.uniform_spacing_flag = r.flag()?;
        if !pps.uniform_spacing_flag {
            for _ in 0..pps.num_tile_columns_minus1 {
                pps.column_width_minus1
                    .push(r.ue_max(u16::MAX as u32, "column_width_minus1")? as u16);
            }
            for _ in 0..pps.num_tile_rows_minus1 {
                pps.row_height_minus1
                    .push(r.ue_max(u16::MAX as u32, "row_height_minus1")? as u16);
            }
        }
        pps.loop_filter_across_tiles_enabled_flag = r.flag()?;
    }

    pps.pps_loop_filter_across_slices_enabled_flag = r.flag()?;
    pps.deblocking_filter_control_present_flag = r.flag()?;
    if pps.deblocking_filter_control_present_flag {
        pps.deblocking_filter_override_enabled_flag = r.flag()?;
        pps.pps_deblocking_filter_disabled_flag = r.flag()?;
        if !pps.pps_deblocking_filter_disabled_flag {
            pps.pps_beta_offset_div2 = r.se_range(-6, 6, "pps_beta_offset_div2")? as i8;
            pps.pps_tc_offset_div2 = r.se_range(-6, 6, "pps_tc_offset_div2")? as i8;
        }
    }
    pps.pps_scaling_list_data_present_flag = r.flag()?;
    if pps.pps_scaling_list_data_present_flag {
        pps.scaling_lists = Some(parse_scaling_list_data(&mut r)?);
    }
    pps.lists_modification_present_flag = r.flag()?;
    pps.log2_parallel_merge_level_minus2 = r.ue_max(4, "log2_parallel_merge_level_minus2")? as u8;
    pps.slice_segment_header_extension_present_flag = r.flag()?;

    pps.pps_extension_present_flag = r.flag()?;
    if pps.pps_extension_present_flag {
        pps.pps_range_extension_flag = r.flag()?;
        pps.pps_multilayer_extension_flag = r.flag()?;
        pps.pps_3d_extension_flag = r.flag()?;
        pps.pps_scc_extension_flag = r.flag()?;
        pps.pps_extension_4bits = r.bits(4)? as u8;
    }
    if pps.pps_range_extension_flag {
        let ext = &mut pps.range_extension;
        if pps.transform_skip_enabled_flag {
            ext.log2_max_transform_skip_block_size_minus2 =
                r.ue_max(3, "log2_max_transform_skip_block_size_minus2")? as u8;
        }
        ext.cross_component_prediction_enabled_flag = r.flag()?;
        ext.chroma_qp_offset_list_enabled_flag = r.flag()?;
        if ext.chroma_qp_offset_list_enabled_flag {
            ext.diff_cu_chroma_qp_offset_depth =
                r.ue_max(3, "diff_cu_chroma_qp_offset_depth")? as u8;
            ext.chroma_qp_offset_list_len_minus1 =
                r.ue_max(5, "chroma_qp_offset_list_len_minus1")? as u8;
            for i in 0..=ext.chroma_qp_offset_list_len_minus1 as usize {
                ext.cb_qp_offset_list[i] = r.se_range(-12, 12, "cb_qp_offset_list")? as i8;
                ext.cr_qp_offset_list[i] = r.se_range(-12, 12, "cr_qp_offset_list")? as i8;
            }
        }
        ext.log2_sao_offset_scale_luma = r.ue_max(6, "log2_sao_offset_scale_luma")? as u8;
        ext.log2_sao_offset_scale_chroma = r.ue_max(6, "log2_sao_offset_scale_chroma")? as u8;
    }
    // As in the SPS, the remaining extensions are recorded by flag only.
    Ok(pps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_tiles_spread_the_remainder() {
        // 10 CTBs over 3 columns: 3, 3, 4 (equation 6-3).
        assert_eq!(tile_sizes(true, true, 2, &[], 10), vec![3, 3, 4]);
        assert_eq!(tile_sizes(false, false, 0, &[], 10), vec![10]);
        // Explicit widths: the last column takes the rest.
        assert_eq!(tile_sizes(true, false, 2, &[1, 3], 10), vec![2, 4, 4]);
    }
}
