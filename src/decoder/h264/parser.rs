//! Minimal H.264 Annex B parsing: NAL unit splitting plus SPS, PPS and
//! slice-header syntax.
//!
//! This is intentionally *not* a full H.264 parser. Vulkan Video decode
//! consumes raw slice NAL units and parses the reference-list machinery in the
//! driver; the host only needs enough syntax to
//! - populate `StdVideoH264SequenceParameterSet` / `StdVideoH264PictureParameterSet`,
//! - compute the picture order count (POC) and frame_num of each picture, and
//! - drive DPB slot management (IDR detection, reference flags).
//!
//! Accordingly the slice-header parser stops right after the POC syntax
//! elements; nothing later in the header is needed on the host side.

use crate::decoder::annexb;
use crate::decoder::bitreader::BitReader;
use crate::error::{PixelForgeError, Result};

/// H.264 NAL unit types this parser cares about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NalType {
    /// Coded slice of a non-IDR picture (type 1).
    Slice,
    /// Coded slice of an IDR picture (type 5).
    IdrSlice,
    /// Supplemental enhancement information (type 6).
    Sei,
    /// Sequence parameter set (type 7).
    Sps,
    /// Picture parameter set (type 8).
    Pps,
    /// Access unit delimiter (type 9).
    Aud,
    /// Anything else.
    Other(u8),
}

impl NalType {
    fn from_raw(raw: u8) -> Self {
        match raw {
            1 => NalType::Slice,
            5 => NalType::IdrSlice,
            6 => NalType::Sei,
            7 => NalType::Sps,
            8 => NalType::Pps,
            9 => NalType::Aud,
            other => NalType::Other(other),
        }
    }

    pub fn is_slice(self) -> bool {
        matches!(self, NalType::Slice | NalType::IdrSlice)
    }
}

/// A single NAL unit within an Annex B stream.
#[derive(Debug, Clone, Copy)]
pub struct NalUnit<'a> {
    pub nal_type: NalType,
    /// `nal_ref_idc`: non-zero means this NAL is part of a reference picture.
    pub ref_idc: u8,
    /// The complete NAL unit (header byte + EBSP payload), without start code.
    pub data: &'a [u8],
}

impl<'a> NalUnit<'a> {
    /// EBSP payload after the 1-byte NAL header.
    pub fn payload(&self) -> &'a [u8] {
        &self.data[1..]
    }
}

/// Iterate over NAL units in an Annex B byte stream (3- or 4-byte start codes).
pub fn iter_nal_units(data: &[u8]) -> impl Iterator<Item = NalUnit<'_>> {
    iter_nal_units_with_offsets(data).map(|(_, nal)| nal)
}

/// Like [`iter_nal_units`], but also yields each NAL's start-code offset within
/// `data`. Used to slice the stream on access-unit boundaries.
pub(crate) fn iter_nal_units_with_offsets(
    data: &[u8],
) -> impl Iterator<Item = (usize, NalUnit<'_>)> {
    annexb::iter_nal_units(data).map(|(offset, nal)| {
        let header = nal[0];
        (
            offset,
            NalUnit {
                nal_type: NalType::from_raw(header & 0x1F),
                ref_idc: (header >> 5) & 0x3,
                data: nal,
            },
        )
    })
}

/// Parsed H.264 sequence parameter set (the subset Vulkan decode needs).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sps {
    pub profile_idc: u8,
    pub constraint_set_flags: u8,
    pub level_idc: u8,
    pub sps_id: u8,
    pub chroma_format_idc: u8,
    pub separate_colour_plane_flag: bool,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub qpprime_y_zero_transform_bypass_flag: bool,
    pub seq_scaling_matrix_present_flag: bool,
    pub log2_max_frame_num_minus4: u8,
    pub pic_order_cnt_type: u8,
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    pub delta_pic_order_always_zero_flag: bool,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    pub offsets_for_ref_frame: Vec<i32>,
    pub max_num_ref_frames: u8,
    pub gaps_in_frame_num_value_allowed_flag: bool,
    pub pic_width_in_mbs_minus1: u32,
    pub pic_height_in_map_units_minus1: u32,
    pub frame_mbs_only_flag: bool,
    pub mb_adaptive_frame_field_flag: bool,
    pub direct_8x8_inference_flag: bool,
    pub frame_cropping_flag: bool,
    pub frame_crop_left_offset: u32,
    pub frame_crop_right_offset: u32,
    pub frame_crop_top_offset: u32,
    pub frame_crop_bottom_offset: u32,
    /// `max_num_reorder_frames` from the VUI bitstream restriction, if present.
    ///
    /// The number of frames that may precede any frame in decode order yet
    /// follow it in display order — i.e. the reorder buffer depth. `None` when
    /// the stream does not signal it, in which case the decoder falls back to a
    /// conservative bound.
    pub max_num_reorder_frames: Option<u32>,
}

impl Sps {
    /// Coded width in pixels (before cropping).
    pub fn coded_width(&self) -> u32 {
        (self.pic_width_in_mbs_minus1 + 1) * 16
    }

    /// Coded height in pixels (before cropping).
    pub fn coded_height(&self) -> u32 {
        let map_units = self.pic_height_in_map_units_minus1 + 1;
        let frame_height_in_mbs = if self.frame_mbs_only_flag {
            map_units
        } else {
            map_units * 2
        };
        frame_height_in_mbs * 16
    }

    /// Display (cropped) dimensions.
    pub fn display_dimensions(&self) -> (u32, u32) {
        let (mut width, mut height) = (self.coded_width(), self.coded_height());
        if self.frame_cropping_flag {
            let (crop_x, crop_y) = match self.chroma_format_idc {
                0 => (1, 1),
                1 => (2, 2),
                2 => (2, 1),
                _ => (1, 1),
            };
            let crop_y = crop_y * if self.frame_mbs_only_flag { 1 } else { 2 };
            width = width.saturating_sub(
                crop_x * (self.frame_crop_left_offset + self.frame_crop_right_offset),
            );
            height = height.saturating_sub(
                crop_y * (self.frame_crop_top_offset + self.frame_crop_bottom_offset),
            );
        }
        (width, height)
    }

    pub fn max_frame_num(&self) -> u32 {
        1 << (self.log2_max_frame_num_minus4 as u32 + 4)
    }

    pub fn max_pic_order_cnt_lsb(&self) -> u32 {
        1 << (self.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4)
    }
}

fn skip_scaling_list(r: &mut BitReader, size: usize) -> Result<()> {
    let mut last_scale = 8i32;
    let mut next_scale = 8i32;
    for _ in 0..size {
        if next_scale != 0 {
            let delta = r.se()?;
            next_scale = (last_scale + delta + 256) % 256;
        }
        if next_scale != 0 {
            last_scale = next_scale;
        }
    }
    Ok(())
}

/// Parse an SPS NAL payload (EBSP, after the NAL header byte).
pub fn parse_sps(payload: &[u8]) -> Result<Sps> {
    let mut r = BitReader::new(payload);
    let mut sps = Sps {
        profile_idc: r.bits(8)? as u8,
        constraint_set_flags: r.bits(8)? as u8,
        level_idc: r.bits(8)? as u8,
        ..Default::default()
    };
    sps.sps_id = r.ue()? as u8;

    // High-profile family carries chroma/bit-depth syntax.
    sps.chroma_format_idc = 1;
    if matches!(
        sps.profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        sps.chroma_format_idc = r.ue()? as u8;
        if sps.chroma_format_idc == 3 {
            sps.separate_colour_plane_flag = r.flag()?;
        }
        sps.bit_depth_luma_minus8 = r.ue()? as u8;
        sps.bit_depth_chroma_minus8 = r.ue()? as u8;
        sps.qpprime_y_zero_transform_bypass_flag = r.flag()?;
        sps.seq_scaling_matrix_present_flag = r.flag()?;
        if sps.seq_scaling_matrix_present_flag {
            let count = if sps.chroma_format_idc == 3 { 12 } else { 8 };
            for i in 0..count {
                if r.flag()? {
                    skip_scaling_list(&mut r, if i < 6 { 16 } else { 64 })?;
                }
            }
        }
    }

    sps.log2_max_frame_num_minus4 = r.ue()? as u8;
    sps.pic_order_cnt_type = r.ue()? as u8;
    match sps.pic_order_cnt_type {
        0 => {
            sps.log2_max_pic_order_cnt_lsb_minus4 = r.ue()? as u8;
        }
        1 => {
            sps.delta_pic_order_always_zero_flag = r.flag()?;
            sps.offset_for_non_ref_pic = r.se()?;
            sps.offset_for_top_to_bottom_field = r.se()?;
            let num = r.ue()?;
            for _ in 0..num {
                sps.offsets_for_ref_frame.push(r.se()?);
            }
        }
        2 => {}
        other => {
            return Err(PixelForgeError::InvalidInput(format!(
                "H.264 SPS: invalid pic_order_cnt_type {}",
                other
            )));
        }
    }

    sps.max_num_ref_frames = r.ue()? as u8;
    sps.gaps_in_frame_num_value_allowed_flag = r.flag()?;
    sps.pic_width_in_mbs_minus1 = r.ue()?;
    sps.pic_height_in_map_units_minus1 = r.ue()?;
    sps.frame_mbs_only_flag = r.flag()?;
    if !sps.frame_mbs_only_flag {
        sps.mb_adaptive_frame_field_flag = r.flag()?;
    }
    sps.direct_8x8_inference_flag = r.flag()?;
    sps.frame_cropping_flag = r.flag()?;
    if sps.frame_cropping_flag {
        sps.frame_crop_left_offset = r.ue()?;
        sps.frame_crop_right_offset = r.ue()?;
        sps.frame_crop_top_offset = r.ue()?;
        sps.frame_crop_bottom_offset = r.ue()?;
    }
    // The only VUI field the decoder needs is max_num_reorder_frames, which
    // sits at the very end (in the bitstream restriction). Everything before it
    // must be parsed to get there; a malformed/truncated VUI just leaves the
    // value unset and the decoder falls back to a conservative bound.
    if r.flag().unwrap_or(false) {
        sps.max_num_reorder_frames = parse_vui_reorder(&mut r).ok().flatten();
    }

    Ok(sps)
}

/// Parse `vui_parameters()` (Annex E.1.1) far enough to recover
/// `max_num_reorder_frames`. Every field before it is parsed only to advance.
fn parse_vui_reorder(r: &mut BitReader) -> Result<Option<u32>> {
    if r.flag()? {
        // aspect_ratio_info_present_flag
        let aspect_ratio_idc = r.bits(8)?;
        // Extended_SAR
        if aspect_ratio_idc == 255 {
            let _sar_width = r.bits(16)?;
            let _sar_height = r.bits(16)?;
        }
    }
    if r.flag()? {
        // overscan_appropriate_flag
        let _overscan_appropriate_flag = r.flag()?;
    }
    if r.flag()? {
        // video_signal_type_present_flag
        let _video_format = r.bits(3)?;
        let _video_full_range_flag = r.flag()?;
        if r.flag()? {
            // colour_description_present_flag
            let _colour_primaries = r.bits(8)?;
            let _transfer_characteristics = r.bits(8)?;
            let _matrix_coefficients = r.bits(8)?;
        }
    }
    if r.flag()? {
        // chroma_loc_info_present_flag
        let _top = r.ue()?;
        let _bottom = r.ue()?;
    }
    if r.flag()? {
        // timing_info_present_flag
        let _num_units_in_tick = r.bits(32)?;
        let _time_scale = r.bits(32)?;
        let _fixed_frame_rate_flag = r.flag()?;
    }
    let nal_hrd = r.flag()?;
    if nal_hrd {
        parse_hrd(r)?;
    }
    let vcl_hrd = r.flag()?;
    if vcl_hrd {
        parse_hrd(r)?;
    }
    if nal_hrd || vcl_hrd {
        let _low_delay_hrd_flag = r.flag()?;
    }
    let _pic_struct_present_flag = r.flag()?;

    if r.flag()? {
        // bitstream_restriction_flag
        let _motion_vectors_over_pic_boundaries_flag = r.flag()?;
        let _max_bytes_per_pic_denom = r.ue()?;
        let _max_bits_per_mb_denom = r.ue()?;
        let _log2_max_mv_length_horizontal = r.ue()?;
        let _log2_max_mv_length_vertical = r.ue()?;
        let max_num_reorder_frames = r.ue()?;
        let _max_dec_frame_buffering = r.ue()?;
        return Ok(Some(max_num_reorder_frames));
    }

    Ok(None)
}

/// Parse `hrd_parameters()` (Annex E.1.2). Consumed only to advance past it.
fn parse_hrd(r: &mut BitReader) -> Result<()> {
    let cpb_cnt_minus1 = r.ue()?;
    let _bit_rate_scale = r.bits(4)?;
    let _cpb_size_scale = r.bits(4)?;
    for _ in 0..=cpb_cnt_minus1 {
        let _bit_rate_value_minus1 = r.ue()?;
        let _cpb_size_value_minus1 = r.ue()?;
        let _cbr_flag = r.flag()?;
    }
    let _initial_cpb_removal_delay_length_minus1 = r.bits(5)?;
    let _cpb_removal_delay_length_minus1 = r.bits(5)?;
    let _dpb_output_delay_length_minus1 = r.bits(5)?;
    let _time_offset_length = r.bits(5)?;
    Ok(())
}

/// Parsed H.264 picture parameter set (the subset Vulkan decode needs).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Pps {
    pub pps_id: u8,
    pub sps_id: u8,
    pub entropy_coding_mode_flag: bool,
    pub bottom_field_pic_order_in_frame_present_flag: bool,
    pub num_slice_groups_minus1: u32,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub weighted_pred_flag: bool,
    pub weighted_bipred_idc: u8,
    pub pic_init_qp_minus26: i8,
    pub pic_init_qs_minus26: i8,
    pub chroma_qp_index_offset: i8,
    pub deblocking_filter_control_present_flag: bool,
    pub constrained_intra_pred_flag: bool,
    pub redundant_pic_cnt_present_flag: bool,
    pub transform_8x8_mode_flag: bool,
    pub pic_scaling_matrix_present_flag: bool,
    pub second_chroma_qp_index_offset: i8,
}

/// Parse a PPS NAL payload (EBSP, after the NAL header byte).
pub fn parse_pps(payload: &[u8], sps_chroma_format_idc: impl Fn(u8) -> Option<u8>) -> Result<Pps> {
    let mut r = BitReader::new(payload);
    let mut pps = Pps {
        pps_id: r.ue()? as u8,
        sps_id: r.ue()? as u8,
        ..Default::default()
    };
    pps.entropy_coding_mode_flag = r.flag()?;
    pps.bottom_field_pic_order_in_frame_present_flag = r.flag()?;
    pps.num_slice_groups_minus1 = r.ue()?;
    if pps.num_slice_groups_minus1 > 0 {
        // FMO is exotic and not supported by any Vulkan Video implementation.
        return Err(PixelForgeError::InvalidInput(
            "H.264 PPS: slice groups (FMO) are not supported".to_string(),
        ));
    }
    pps.num_ref_idx_l0_default_active_minus1 = r.ue()? as u8;
    pps.num_ref_idx_l1_default_active_minus1 = r.ue()? as u8;
    pps.weighted_pred_flag = r.flag()?;
    pps.weighted_bipred_idc = r.bits(2)? as u8;
    pps.pic_init_qp_minus26 = r.se()? as i8;
    pps.pic_init_qs_minus26 = r.se()? as i8;
    pps.chroma_qp_index_offset = r.se()? as i8;
    pps.deblocking_filter_control_present_flag = r.flag()?;
    pps.constrained_intra_pred_flag = r.flag()?;
    pps.redundant_pic_cnt_present_flag = r.flag()?;

    // Optional trailing fields (present in High profile streams). Detect by
    // attempting to read; `more_rbsp_data` is approximated by whether reads
    // succeed and the stop bit hasn't been consumed. We conservatively try and
    // fall back to defaults on end-of-data.
    pps.second_chroma_qp_index_offset = pps.chroma_qp_index_offset;
    if let Ok(transform_8x8) = r.flag() {
        // Distinguish real data from the RBSP stop bit: the stop bit is a `1`
        // followed only by zero padding. If everything after this flag fails to
        // parse, treat it as the stop bit.
        let mut tail = || -> Result<(bool, bool, i8)> {
            let scaling = r.flag()?;
            if scaling {
                let chroma_idc = sps_chroma_format_idc(pps.sps_id).unwrap_or(1);
                let count = 6 + if chroma_idc == 3 { 6 } else { 2 };
                for i in 0..count {
                    if r.flag()? {
                        skip_scaling_list(&mut r, if i < 6 { 16 } else { 64 })?;
                    }
                }
            }
            let second_offset = r.se()? as i8;
            Ok((transform_8x8, scaling, second_offset))
        };
        if let Ok((t8, scaling, second)) = tail() {
            pps.transform_8x8_mode_flag = t8;
            pps.pic_scaling_matrix_present_flag = scaling;
            pps.second_chroma_qp_index_offset = second;
        }
    }

    Ok(pps)
}

/// H.264 slice types (values already reduced modulo 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    P,
    B,
    I,
    Sp,
    Si,
}

/// A memory management control operation (clause 7.4.3.3).
///
/// Operand names follow the standard. All picture numbers are frame-based:
/// field decoding is rejected before any of this is reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mmco {
    /// 1: mark a short-term reference as unused.
    ForgetShort { difference_of_pic_nums_minus1: u32 },
    /// 2: mark a long-term reference as unused.
    ForgetLong { long_term_pic_num: u32 },
    /// 3: turn a short-term reference into a long-term one.
    ShortToLong {
        difference_of_pic_nums_minus1: u32,
        long_term_frame_idx: u32,
    },
    /// 4: set the upper bound on long-term frame indices.
    MaxLongTermIdx { max_long_term_frame_idx_plus1: u32 },
    /// 5: mark every reference unused and reset frame_num/POC.
    ForgetAll,
    /// 6: mark the current picture as a long-term reference.
    CurrentToLong { long_term_frame_idx: u32 },
}

/// `dec_ref_pic_marking()` (clause 7.3.3.3).
///
/// Only present when `nal_ref_idc != 0`; a default value therefore means "this
/// picture is not a reference and marks nothing".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RefPicMarking {
    /// IDR only: the IDR itself becomes a long-term reference.
    pub long_term_reference_flag: bool,
    /// IDR only. Retained for completeness; output order is not driven by it.
    pub no_output_of_prior_pics_flag: bool,
    /// Non-IDR: explicit marking is in use, so `ops` replaces the sliding
    /// window. When false the sliding-window process applies (clause 8.2.5.3).
    pub adaptive: bool,
    /// The MMCO ops, in bitstream order. Terminating op 0 is not stored.
    pub ops: Vec<Mmco>,
}

/// The slice-header fields needed for picture-level decode bookkeeping.
///
/// Parsing runs as far as `dec_ref_pic_marking()`, since reference marking is
/// the decoder's responsibility; the syntax in between is parsed only to reach
/// it. Everything after is consumed by the driver from the raw slice NAL.
#[derive(Debug, Clone)]
pub struct SliceHeader {
    pub first_mb_in_slice: u32,
    pub slice_type: SliceType,
    pub pps_id: u8,
    pub frame_num: u32,
    pub field_pic_flag: bool,
    pub idr_pic_id: u16,
    pub pic_order_cnt_lsb: u32,
    pub delta_pic_order_cnt: [i32; 2],
    pub marking: RefPicMarking,
}

/// Parse the leading portion of a slice header.
pub fn parse_slice_header(nal: &NalUnit, sps: &Sps, pps: &Pps) -> Result<SliceHeader> {
    let mut r = BitReader::new(nal.payload());
    let first_mb_in_slice = r.ue()?;
    let slice_type_raw = r.ue()?;
    let slice_type = match slice_type_raw % 5 {
        0 => SliceType::P,
        1 => SliceType::B,
        2 => SliceType::I,
        3 => SliceType::Sp,
        4 => SliceType::Si,
        _ => unreachable!(),
    };
    let pps_id = r.ue()? as u8;

    if sps.separate_colour_plane_flag {
        let _colour_plane_id = r.bits(2)?;
    }

    let frame_num = r.bits(sps.log2_max_frame_num_minus4 as u32 + 4)?;

    // Field coding is rejected before decode; bottom_field_flag is parsed only
    // to stay bit-aligned, never stored.
    let mut field_pic_flag = false;
    if !sps.frame_mbs_only_flag {
        field_pic_flag = r.flag()?;
        if field_pic_flag {
            let _bottom_field_flag = r.flag()?;
        }
    }

    let mut idr_pic_id = 0u16;
    if nal.nal_type == NalType::IdrSlice {
        idr_pic_id = r.ue()? as u16;
    }

    let mut pic_order_cnt_lsb = 0u32;
    let mut delta_pic_order_cnt = [0i32; 2];
    match sps.pic_order_cnt_type {
        0 => {
            pic_order_cnt_lsb = r.bits(sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4)?;
            if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
                // Bottom-field POC delta: parsed for alignment, unused for frames.
                let _delta_pic_order_cnt_bottom = r.se()?;
            }
        }
        1 if !sps.delta_pic_order_always_zero_flag => {
            delta_pic_order_cnt[0] = r.se()?;
            if pps.bottom_field_pic_order_in_frame_present_flag && !field_pic_flag {
                delta_pic_order_cnt[1] = r.se()?;
            }
        }
        _ => {}
    }

    // From here the only field we need is dec_ref_pic_marking(), but it is not
    // at a fixed offset: everything before it must be parsed to find it.
    if pps.redundant_pic_cnt_present_flag {
        let _redundant_pic_cnt = r.ue()?;
    }
    if slice_type == SliceType::B {
        let _direct_spatial_mv_pred_flag = r.flag()?;
    }

    let mut num_ref_idx_l0_active_minus1 = pps.num_ref_idx_l0_default_active_minus1 as u32;
    let mut num_ref_idx_l1_active_minus1 = pps.num_ref_idx_l1_default_active_minus1 as u32;
    if matches!(slice_type, SliceType::P | SliceType::Sp | SliceType::B) {
        if r.flag()? {
            num_ref_idx_l0_active_minus1 = r.ue()?;
            if slice_type == SliceType::B {
                num_ref_idx_l1_active_minus1 = r.ue()?;
            }
        }
        // Clause 7.4.3: the value is at most 31. A larger one means we have
        // lost bit alignment, and would otherwise drive a huge parse loop.
        if num_ref_idx_l0_active_minus1 > 31 || num_ref_idx_l1_active_minus1 > 31 {
            return Err(PixelForgeError::InvalidInput(
                "H.264 decode: num_ref_idx_active_minus1 out of range".to_string(),
            ));
        }
    }

    parse_ref_pic_list_modification(&mut r, slice_type)?;

    let chroma_array_type = if sps.separate_colour_plane_flag {
        0
    } else {
        sps.chroma_format_idc
    };
    let weighted = match slice_type {
        SliceType::P | SliceType::Sp => pps.weighted_pred_flag,
        SliceType::B => pps.weighted_bipred_idc == 1,
        _ => false,
    };
    if weighted {
        parse_pred_weight_table(
            &mut r,
            chroma_array_type,
            num_ref_idx_l0_active_minus1,
            num_ref_idx_l1_active_minus1,
            slice_type == SliceType::B,
        )?;
    }

    let marking = if nal.ref_idc != 0 {
        parse_dec_ref_pic_marking(&mut r, nal.nal_type == NalType::IdrSlice)?
    } else {
        RefPicMarking::default()
    };

    Ok(SliceHeader {
        first_mb_in_slice,
        slice_type,
        pps_id,
        frame_num,
        field_pic_flag,
        idr_pic_id,
        pic_order_cnt_lsb,
        delta_pic_order_cnt,
        marking,
    })
}

/// `ref_pic_list_modification()` (clause 7.3.3.1).
///
/// The reordered lists themselves are rebuilt by the driver from the slice
/// data, so the syntax is parsed only to advance past it.
fn parse_ref_pic_list_modification(r: &mut BitReader, slice_type: SliceType) -> Result<()> {
    let modification_list = |r: &mut BitReader| -> Result<()> {
        if !r.flag()? {
            return Ok(());
        }
        loop {
            match r.ue()? {
                // abs_diff_pic_num_minus1 / long_term_pic_num
                0..=2 => {
                    let _ = r.ue()?;
                }
                3 => return Ok(()),
                other => {
                    return Err(PixelForgeError::InvalidInput(format!(
                        "H.264 decode: invalid modification_of_pic_nums_idc {}",
                        other
                    )));
                }
            }
        }
    };

    if !matches!(slice_type, SliceType::I | SliceType::Si) {
        modification_list(r)?;
    }
    if slice_type == SliceType::B {
        modification_list(r)?;
    }
    Ok(())
}

/// `pred_weight_table()` (clause 7.3.3.2). Parsed only to advance past it.
fn parse_pred_weight_table(
    r: &mut BitReader,
    chroma_array_type: u8,
    num_ref_idx_l0_active_minus1: u32,
    num_ref_idx_l1_active_minus1: u32,
    is_b: bool,
) -> Result<()> {
    let _luma_log2_weight_denom = r.ue()?;
    if chroma_array_type != 0 {
        let _chroma_log2_weight_denom = r.ue()?;
    }

    let weights = |r: &mut BitReader, count: u32| -> Result<()> {
        for _ in 0..=count {
            if r.flag()? {
                let _luma_weight = r.se()?;
                let _luma_offset = r.se()?;
            }
            if chroma_array_type != 0 && r.flag()? {
                for _ in 0..2 {
                    let _chroma_weight = r.se()?;
                    let _chroma_offset = r.se()?;
                }
            }
        }
        Ok(())
    };

    weights(r, num_ref_idx_l0_active_minus1)?;
    if is_b {
        weights(r, num_ref_idx_l1_active_minus1)?;
    }
    Ok(())
}

/// `dec_ref_pic_marking()` (clause 7.3.3.3).
fn parse_dec_ref_pic_marking(r: &mut BitReader, is_idr: bool) -> Result<RefPicMarking> {
    let mut marking = RefPicMarking::default();
    if is_idr {
        marking.no_output_of_prior_pics_flag = r.flag()?;
        marking.long_term_reference_flag = r.flag()?;
        return Ok(marking);
    }

    marking.adaptive = r.flag()?;
    if !marking.adaptive {
        return Ok(marking);
    }

    loop {
        let op = r.ue()?;
        let op = match op {
            0 => return Ok(marking),
            1 => Mmco::ForgetShort {
                difference_of_pic_nums_minus1: r.ue()?,
            },
            2 => Mmco::ForgetLong {
                long_term_pic_num: r.ue()?,
            },
            3 => Mmco::ShortToLong {
                difference_of_pic_nums_minus1: r.ue()?,
                long_term_frame_idx: r.ue()?,
            },
            4 => Mmco::MaxLongTermIdx {
                max_long_term_frame_idx_plus1: r.ue()?,
            },
            5 => Mmco::ForgetAll,
            6 => Mmco::CurrentToLong {
                long_term_frame_idx: r.ue()?,
            },
            other => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "H.264 decode: invalid memory_management_control_operation {}",
                    other
                )));
            }
        };
        marking.ops.push(op);

        // A conforming stream cannot mark more pictures than the DPB can hold;
        // this bounds the loop if the bitstream is corrupt.
        if marking.ops.len() > 64 {
            return Err(PixelForgeError::InvalidInput(
                "H.264 decode: runaway dec_ref_pic_marking".to_string(),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nal_iteration() {
        // 4-byte start code + SPS header byte, 3-byte start code + PPS header byte.
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0xAA, // SPS (type 7, ref_idc 3)
            0x00, 0x00, 0x01, 0x68, 0xBB, // PPS (type 8, ref_idc 3)
            0x00, 0x00, 0x01, 0x65, 0xCC, // IDR slice (type 5)
        ];
        let nals: Vec<_> = iter_nal_units(&data).collect();
        assert_eq!(nals.len(), 3);
        assert_eq!(nals[0].nal_type, NalType::Sps);
        assert_eq!(nals[0].ref_idc, 3);
        assert_eq!(nals[0].data, &[0x67, 0xAA]);
        assert_eq!(nals[1].nal_type, NalType::Pps);
        assert_eq!(nals[2].nal_type, NalType::IdrSlice);
        assert!(nals[2].nal_type.is_slice());
    }

    #[test]
    fn test_parse_x264_sps_pps() {
        // Verbatim SPS/PPS from an x264-encoded 320x240 High-profile stream
        // (the same encoder settings as tests/data/bframes.264). Note the
        // 0x000003 emulation-prevention sequences in the VUI.
        let sps_payload = [
            0x64, 0x00, 0x0d, 0xac, 0xd9, 0x41, 0x41, 0xfa, 0x10, 0x00, 0x00, 0x03, 0x00, 0x10,
            0x00, 0x00, 0x03, 0x03, 0xc8, 0xf1, 0x42, 0x99, 0x60,
        ];
        let sps = parse_sps(&sps_payload).unwrap();
        assert_eq!(sps.profile_idc, 100);
        assert_eq!(sps.level_idc, 13);
        assert_eq!(sps.chroma_format_idc, 1);
        assert_eq!(sps.coded_width(), 320);
        assert_eq!(sps.coded_height(), 240);
        assert_eq!(sps.display_dimensions(), (320, 240));
        assert!(sps.frame_mbs_only_flag);

        // Matching PPS: 68 eb e3 cb 22 c0
        let pps_payload = [0xeb, 0xe3, 0xcb, 0x22, 0xc0];
        let pps = parse_pps(&pps_payload, |_| Some(sps.chroma_format_idc)).unwrap();
        assert_eq!(pps.pps_id, 0);
        assert_eq!(pps.sps_id, 0);
        assert!(pps.entropy_coding_mode_flag); // CABAC
    }
}
