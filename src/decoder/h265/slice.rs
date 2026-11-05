//! The leading part of the H.265 slice segment header (clause 7.3.6.1).
//!
//! Parsing stops after `slice_temporal_mvp_enabled_flag`. Everything the host
//! needs is before that point: the picture order count, the short-term and
//! long-term reference picture set syntax, and the bit count Vulkan wants for
//! the inline short-term set. What follows (SAO, reference list sizes and
//! modification, weighted prediction, QP and deblocking overrides, entry
//! points) is parsed by the driver from the slice NAL unit itself.

use super::parser::{NalType, NalUnit, Reader, ceil_log2, invalid};
use super::pps::Pps;
use super::rps::{MAX_ST_REFS, StRefPicSet, parse_st_ref_pic_set};
use super::sps::Sps;
use crate::error::Result;

/// H.265 slice types (Table 7-7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum SliceType {
    B,
    P,
    #[default]
    I,
}

/// One long-term reference picture entry of the slice header, resolved
/// against the SPS where it was an index into `lt_ref_pic_poc_lsb_sps`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct LongTermRef {
    /// `lt_idx_sps[i]` for an entry taken from the SPS, `None` for one coded
    /// in the slice header.
    pub lt_idx_sps: Option<u8>,
    /// `PocLsbLt[i]`.
    pub poc_lsb_lt: u32,
    /// `UsedByCurrPicLt[i]`.
    pub used_by_curr_pic_lt: bool,
    pub delta_poc_msb_present_flag: bool,
    /// `delta_poc_msb_cycle_lt[i]` as coded.
    pub delta_poc_msb_cycle_lt: u32,
    /// `DeltaPocMsbCycleLt[i]` (equation 7-52), which accumulates the coded
    /// values within each of the two groups (SPS entries, then slice entries).
    pub delta_poc_msb_cycle_lt_accumulated: u32,
}

/// The slice segment header fields the host needs.
///
/// A dependent slice segment (`dependent_slice_segment_flag`) codes only the
/// fields up to `slice_segment_address`; the rest of its header is that of the
/// preceding independent segment, and is left at defaults here.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SliceHeader {
    pub first_slice_segment_in_pic_flag: bool,
    /// Coded only for IRAP pictures.
    pub no_output_of_prior_pics_flag: bool,
    pub slice_pic_parameter_set_id: u8,
    pub dependent_slice_segment_flag: bool,
    pub slice_segment_address: u32,
    pub slice_type: SliceType,
    /// 1 when not coded (`output_flag_present_flag` = 0).
    pub pic_output_flag: bool,
    pub colour_plane_id: u8,
    /// 0 for IDR pictures, which do not code it.
    pub slice_pic_order_cnt_lsb: u32,
    pub short_term_ref_pic_set_sps_flag: bool,
    pub short_term_ref_pic_set_idx: u8,
    /// The set coded in the slice header, when
    /// `short_term_ref_pic_set_sps_flag` is 0 (and the picture is not IDR).
    pub st_ref_pic_set: Option<StRefPicSet>,
    /// RBSP bits the inline `st_ref_pic_set()` occupied: Vulkan's
    /// `NumBitsForSTRefPicSetInSlice`. 0 when the set came from the SPS.
    pub num_bits_for_st_ref_pic_set: u32,
    pub num_long_term_sps: u8,
    pub num_long_term_pics: u8,
    /// `num_long_term_sps + num_long_term_pics` entries, SPS ones first.
    pub long_term_refs: Vec<LongTermRef>,
    pub slice_temporal_mvp_enabled_flag: bool,
}

impl SliceHeader {
    /// The short-term set in effect: the inline one, or the SPS set
    /// `short_term_ref_pic_set_idx` selects. `None` for an IDR picture, whose
    /// reference picture set is empty.
    pub fn short_term_ref_pic_set<'a>(&'a self, sps: &'a Sps) -> Option<&'a StRefPicSet> {
        if self.short_term_ref_pic_set_sps_flag {
            sps.short_term_ref_pic_sets
                .get(self.short_term_ref_pic_set_idx as usize)
        } else {
            self.st_ref_pic_set.as_ref()
        }
    }

    /// Vulkan's `NumDeltaPocsOfRefRpsIdx`: `NumDeltaPocs[RefRpsIdx]` when the
    /// slice header codes its own set with inter RPS prediction, else 0.
    pub fn num_delta_pocs_of_ref_rps_idx(&self, sps: &Sps) -> u8 {
        let Some(set) = self.st_ref_pic_set.as_ref() else {
            return 0;
        };
        set.ref_rps_idx
            .and_then(|idx| sps.short_term_ref_pic_sets.get(idx))
            .map_or(0, |reference| reference.rps.num_delta_pocs() as u8)
    }
}

/// The PPS a slice segment refers to, read without parsing the rest of the
/// header, so the caller can find the PPS (and through it the SPS) first.
pub(crate) fn slice_pps_id(nal: &NalUnit) -> Result<u8> {
    let mut r = Reader::new(nal.payload());
    let _first_slice_segment_in_pic_flag = r.flag()?;
    if nal.nal_type().is_irap() {
        let _no_output_of_prior_pics_flag = r.flag()?;
    }
    Ok(r.ue_max(63, "slice_pic_parameter_set_id")? as u8)
}

/// Parse the leading part of a slice segment header; see the module
/// documentation for where it stops.
pub(crate) fn parse_slice_header(nal: &NalUnit, sps: &Sps, pps: &Pps) -> Result<SliceHeader> {
    let nal_type = nal.nal_type();
    if !nal_type.is_vcl() {
        return Err(invalid("slice header parse of a non-VCL NAL unit"));
    }
    let mut r = Reader::new(nal.payload());
    let mut h = SliceHeader {
        first_slice_segment_in_pic_flag: r.flag()?,
        pic_output_flag: true,
        ..Default::default()
    };
    if nal_type.is_irap() {
        h.no_output_of_prior_pics_flag = r.flag()?;
    }
    h.slice_pic_parameter_set_id = r.ue_max(63, "slice_pic_parameter_set_id")? as u8;
    if h.slice_pic_parameter_set_id != pps.pps_pic_parameter_set_id
        || pps.pps_seq_parameter_set_id != sps.sps_seq_parameter_set_id
    {
        return Err(invalid("slice parsed against the wrong PPS or SPS"));
    }

    if !h.first_slice_segment_in_pic_flag {
        if pps.dependent_slice_segments_enabled_flag {
            h.dependent_slice_segment_flag = r.flag()?;
        }
        let pic_size = sps.pic_size_in_ctbs();
        h.slice_segment_address = r.bits(ceil_log2(pic_size))?;
        if h.slice_segment_address >= pic_size {
            return Err(invalid("slice_segment_address outside the picture"));
        }
    }
    if h.dependent_slice_segment_flag {
        return Ok(h);
    }

    // slice_reserved_flag[i].
    r.skip(pps.num_extra_slice_header_bits as u32)?;
    h.slice_type = match r.ue()? {
        0 => SliceType::B,
        1 => SliceType::P,
        2 => SliceType::I,
        other => return Err(invalid(format_args!("invalid slice_type {other}"))),
    };
    if pps.output_flag_present_flag {
        h.pic_output_flag = r.flag()?;
    }
    if sps.separate_colour_plane_flag {
        h.colour_plane_id = r.bits(2)? as u8;
    }
    if nal_type.is_idr() {
        return Ok(h);
    }

    let lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
    h.slice_pic_order_cnt_lsb = r.bits(lsb_bits)?;
    h.short_term_ref_pic_set_sps_flag = r.flag()?;
    let num_sets = sps.num_short_term_ref_pic_sets();
    if !h.short_term_ref_pic_set_sps_flag {
        let start = r.position();
        let set = parse_st_ref_pic_set(
            &mut r,
            num_sets,
            num_sets,
            &sps.short_term_ref_pic_sets,
            sps.max_dec_pic_buffering() - 1,
        )?;
        h.num_bits_for_st_ref_pic_set = (r.position() - start) as u32;
        h.st_ref_pic_set = Some(set);
    } else {
        if num_sets == 0 {
            return Err(invalid(
                "slice selects an SPS short-term set, but the SPS has none",
            ));
        }
        if num_sets > 1 {
            h.short_term_ref_pic_set_idx = r.bits(ceil_log2(num_sets as u32))? as u8;
        }
        if h.short_term_ref_pic_set_idx as usize >= num_sets {
            return Err(invalid("short_term_ref_pic_set_idx out of range"));
        }
    }

    if sps.long_term_ref_pics_present_flag {
        parse_long_term_refs(&mut r, &mut h, sps)?;
    }
    if sps.sps_temporal_mvp_enabled_flag {
        h.slice_temporal_mvp_enabled_flag = r.flag()?;
    }
    Ok(h)
}

/// The long-term part of the slice header, from `num_long_term_sps` to the
/// last `delta_poc_msb_cycle_lt`.
fn parse_long_term_refs(r: &mut Reader, h: &mut SliceHeader, sps: &Sps) -> Result<()> {
    let num_sps_entries = sps.num_long_term_ref_pics_sps();
    if num_sps_entries > 0 {
        h.num_long_term_sps = r.ue_max(num_sps_entries as u32, "num_long_term_sps")? as u8;
    }
    h.num_long_term_pics = r.ue_max(
        MAX_ST_REFS as u32 - h.num_long_term_sps as u32,
        "num_long_term_pics",
    )? as u8;

    let lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4;
    let idx_bits = ceil_log2(num_sps_entries as u32);
    let total = h.num_long_term_sps as usize + h.num_long_term_pics as usize;
    let mut accumulated = 0u32;
    for i in 0..total {
        let mut entry = LongTermRef::default();
        if i < h.num_long_term_sps as usize {
            let idx = if num_sps_entries > 1 {
                r.bits(idx_bits)?
            } else {
                0
            };
            let lsb = sps
                .long_term_ref_pics
                .lt_ref_pic_poc_lsb_sps
                .get(idx as usize)
                .ok_or_else(|| invalid("lt_idx_sps out of range"))?;
            entry.lt_idx_sps = Some(idx as u8);
            entry.poc_lsb_lt = *lsb;
            entry.used_by_curr_pic_lt =
                sps.long_term_ref_pics.used_by_curr_pic_lt_sps_flag & (1 << idx) != 0;
        } else {
            entry.poc_lsb_lt = r.bits(lsb_bits)?;
            entry.used_by_curr_pic_lt = r.flag()?;
        }
        entry.delta_poc_msb_present_flag = r.flag()?;
        if entry.delta_poc_msb_present_flag {
            entry.delta_poc_msb_cycle_lt = r.ue()?;
        }
        // Equation 7-52: the sum restarts at the first slice-coded entry.
        if i == 0 || i == h.num_long_term_sps as usize {
            accumulated = entry.delta_poc_msb_cycle_lt;
        } else {
            accumulated = accumulated.saturating_add(entry.delta_poc_msb_cycle_lt);
        }
        entry.delta_poc_msb_cycle_lt_accumulated = accumulated;
        h.long_term_refs.push(entry);
    }
    Ok(())
}

/// Whether `nal` starts a new coded picture: a slice segment of a known type
/// with `first_slice_segment_in_pic_flag` set, which is the first bit after
/// the NAL header. Reserved VCL types are ignored, as clause 7.4.2.2 requires.
pub(crate) fn starts_picture(nal: &NalUnit) -> bool {
    let nal_type = nal.nal_type();
    nal_type.is_vcl()
        && !matches!(nal_type, NalType::Other(_))
        && nal.payload().first().is_some_and(|b| b & 0x80 != 0)
}
