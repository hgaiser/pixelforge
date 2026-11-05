//! H.265 Annex B parsing: NAL unit splitting and headers, the bit reader the
//! other syntax parsers share, `profile_tier_level()`, `hrd_parameters()` and
//! the video parameter set.
//!
//! The SPS, PPS and slice segment header live in their own modules
//! ([`super::sps`], [`super::pps`], [`super::slice`]); this one holds what they
//! have in common.
//!
//! As with H.264, this is not a full parser. Vulkan Video consumes the raw
//! slice segment NAL units and parses the slice data in the driver; the host
//! needs only enough syntax to fill the `StdVideoH265*` parameter sets, derive
//! the picture order count and drive the reference picture set.

use crate::decoder::annexb;
use crate::decoder::bitreader::BitReader;
use crate::error::{PixelForgeError, Result};

/// Highest number of temporal sub-layers (`sps_max_sub_layers_minus1` is at
/// most 6).
pub(crate) const MAX_SUB_LAYERS: usize = 7;

/// Build the error every parser in this module returns for bad syntax.
pub(crate) fn invalid(message: impl std::fmt::Display) -> PixelForgeError {
    PixelForgeError::InvalidInput(format!("H.265: {message}"))
}

/// `Ceil(Log2(x))`, the width of the fixed-length index fields.
pub(crate) fn ceil_log2(x: u32) -> u32 {
    if x <= 1 {
        0
    } else {
        32 - (x - 1).leading_zeros()
    }
}

/// H.265 NAL unit types (Table 7-1).
///
/// The reserved and unspecified values are kept as [`NalType::Other`] with
/// their raw value, so the range helpers still classify them correctly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NalType {
    /// 0: trailing picture, sub-layer non-reference.
    TrailN,
    /// 1: trailing picture, sub-layer reference.
    TrailR,
    /// 2: temporal sub-layer access, sub-layer non-reference.
    TsaN,
    /// 3: temporal sub-layer access, sub-layer reference.
    TsaR,
    /// 4: step-wise temporal sub-layer access, sub-layer non-reference.
    StsaN,
    /// 5: step-wise temporal sub-layer access, sub-layer reference.
    StsaR,
    /// 6: random access decodable leading picture, sub-layer non-reference.
    RadlN,
    /// 7: random access decodable leading picture, sub-layer reference.
    RadlR,
    /// 8: random access skipped leading picture, sub-layer non-reference.
    RaslN,
    /// 9: random access skipped leading picture, sub-layer reference.
    RaslR,
    /// 16: broken link access, may have RASL and RADL leading pictures.
    BlaWLp,
    /// 17: broken link access, may have RADL leading pictures only.
    BlaWRadl,
    /// 18: broken link access, no leading pictures.
    BlaNLp,
    /// 19: instantaneous decoding refresh, may have RADL leading pictures.
    IdrWRadl,
    /// 20: instantaneous decoding refresh, no leading pictures.
    IdrNLp,
    /// 21: clean random access.
    Cra,
    /// 32: video parameter set.
    Vps,
    /// 33: sequence parameter set.
    Sps,
    /// 34: picture parameter set.
    Pps,
    /// 35: access unit delimiter.
    Aud,
    /// 36: end of sequence.
    Eos,
    /// 37: end of bitstream.
    Eob,
    /// 38: filler data.
    Fd,
    /// 39: prefix supplemental enhancement information.
    SeiPrefix,
    /// 40: suffix supplemental enhancement information.
    SeiSuffix,
    /// Any reserved or unspecified type, by raw value.
    Other(u8),
}

impl NalType {
    pub fn from_raw(raw: u8) -> Self {
        match raw {
            0 => NalType::TrailN,
            1 => NalType::TrailR,
            2 => NalType::TsaN,
            3 => NalType::TsaR,
            4 => NalType::StsaN,
            5 => NalType::StsaR,
            6 => NalType::RadlN,
            7 => NalType::RadlR,
            8 => NalType::RaslN,
            9 => NalType::RaslR,
            16 => NalType::BlaWLp,
            17 => NalType::BlaWRadl,
            18 => NalType::BlaNLp,
            19 => NalType::IdrWRadl,
            20 => NalType::IdrNLp,
            21 => NalType::Cra,
            32 => NalType::Vps,
            33 => NalType::Sps,
            34 => NalType::Pps,
            35 => NalType::Aud,
            36 => NalType::Eos,
            37 => NalType::Eob,
            38 => NalType::Fd,
            39 => NalType::SeiPrefix,
            40 => NalType::SeiSuffix,
            other => NalType::Other(other),
        }
    }

    /// The `nal_unit_type` value.
    pub fn raw(self) -> u8 {
        match self {
            NalType::TrailN => 0,
            NalType::TrailR => 1,
            NalType::TsaN => 2,
            NalType::TsaR => 3,
            NalType::StsaN => 4,
            NalType::StsaR => 5,
            NalType::RadlN => 6,
            NalType::RadlR => 7,
            NalType::RaslN => 8,
            NalType::RaslR => 9,
            NalType::BlaWLp => 16,
            NalType::BlaWRadl => 17,
            NalType::BlaNLp => 18,
            NalType::IdrWRadl => 19,
            NalType::IdrNLp => 20,
            NalType::Cra => 21,
            NalType::Vps => 32,
            NalType::Sps => 33,
            NalType::Pps => 34,
            NalType::Aud => 35,
            NalType::Eos => 36,
            NalType::Eob => 37,
            NalType::Fd => 38,
            NalType::SeiPrefix => 39,
            NalType::SeiSuffix => 40,
            NalType::Other(raw) => raw,
        }
    }

    /// A coded slice segment (types 0..=31), including the reserved ones.
    pub fn is_vcl(self) -> bool {
        self.raw() < 32
    }

    /// An intra random access point picture: BLA, IDR, CRA or a reserved IRAP
    /// type (16..=23).
    pub fn is_irap(self) -> bool {
        (16..=23).contains(&self.raw())
    }

    /// An IDR picture, with or without RADL leading pictures.
    pub fn is_idr(self) -> bool {
        matches!(self, NalType::IdrWRadl | NalType::IdrNLp)
    }

    /// A broken link access picture.
    pub fn is_bla(self) -> bool {
        matches!(self, NalType::BlaWLp | NalType::BlaWRadl | NalType::BlaNLp)
    }

    /// A random access skipped leading picture.
    pub fn is_rasl(self) -> bool {
        matches!(self, NalType::RaslN | NalType::RaslR)
    }

    /// A random access decodable leading picture.
    pub fn is_radl(self) -> bool {
        matches!(self, NalType::RadlN | NalType::RadlR)
    }

    /// A sub-layer non-reference picture: the even types up to 14
    /// (`TRAIL_N`, `TSA_N`, `STSA_N`, `RADL_N`, `RASL_N` and the reserved
    /// `RSV_VCL_N10/12/14`).
    ///
    /// Such a picture is never referenced by a picture of the same temporal
    /// sub-layer, but a higher sub-layer may still reference it.
    pub fn is_sub_layer_non_reference(self) -> bool {
        let raw = self.raw();
        raw <= 14 && raw.is_multiple_of(2)
    }
}

/// The two-byte NAL unit header (clause 7.3.1.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NalHeader {
    pub nal_type: NalType,
    /// `nuh_layer_id`. Anything above 0 belongs to a layer this decoder does
    /// not decode (scalable or multiview extensions) and should be dropped.
    pub layer_id: u8,
    /// `TemporalId`, which is `nuh_temporal_id_plus1 - 1`.
    pub temporal_id: u8,
}

/// Parse the NAL unit header from the bytes after the start code.
pub(crate) fn parse_nal_header(nal: &[u8]) -> Result<NalHeader> {
    if nal.len() < 2 {
        return Err(invalid("NAL unit shorter than its header"));
    }
    if nal[0] & 0x80 != 0 {
        return Err(invalid("forbidden_zero_bit is set"));
    }
    let temporal_id_plus1 = nal[1] & 0x7;
    if temporal_id_plus1 == 0 {
        return Err(invalid("nuh_temporal_id_plus1 is 0"));
    }
    Ok(NalHeader {
        nal_type: NalType::from_raw((nal[0] >> 1) & 0x3F),
        layer_id: ((nal[0] & 1) << 5) | (nal[1] >> 3),
        temporal_id: temporal_id_plus1 - 1,
    })
}

/// A single NAL unit within an Annex B stream.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NalUnit<'a> {
    pub header: NalHeader,
    /// The complete NAL unit (2-byte header + EBSP payload), without start code.
    pub data: &'a [u8],
}

impl<'a> NalUnit<'a> {
    pub fn nal_type(&self) -> NalType {
        self.header.nal_type
    }

    /// EBSP payload after the 2-byte NAL header.
    pub fn payload(&self) -> &'a [u8] {
        &self.data[2..]
    }
}

/// Iterate over NAL units in an Annex B byte stream (3- or 4-byte start codes).
///
/// NAL units whose header is malformed (too short, forbidden bit set) are
/// skipped rather than ending the iteration.
pub(crate) fn iter_nal_units(data: &[u8]) -> impl Iterator<Item = NalUnit<'_>> {
    iter_nal_units_with_offsets(data).map(|(_, nal)| nal)
}

/// Like [`iter_nal_units`], but also yields each NAL's start-code offset within
/// `data`.
pub(crate) fn iter_nal_units_with_offsets(
    data: &[u8],
) -> impl Iterator<Item = (usize, NalUnit<'_>)> {
    annexb::iter_nal_units(data).filter_map(|(offset, nal)| {
        let header = parse_nal_header(nal).ok()?;
        Some((offset, NalUnit { header, data: nal }))
    })
}

/// [`BitReader`] plus a count of the RBSP bits consumed.
///
/// The count excludes emulation prevention bytes, which is what Vulkan's
/// `NumBitsForSTRefPicSetInSlice` is measured in. `ue(v)`/`se(v)` are
/// reimplemented here on top of [`BitReader::bit`] so every bit is counted.
pub(crate) struct Reader<'a> {
    inner: BitReader<'a>,
    position: usize,
}

impl<'a> Reader<'a> {
    pub fn new(payload: &'a [u8]) -> Self {
        Self {
            inner: BitReader::new(payload),
            position: 0,
        }
    }

    /// RBSP bits consumed so far.
    pub fn position(&self) -> usize {
        self.position
    }

    pub fn bit(&mut self) -> Result<u32> {
        let bit = self.inner.bit()?;
        self.position += 1;
        Ok(bit)
    }

    /// `u(n)`, n <= 32.
    pub fn bits(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        let mut v = 0u64;
        for _ in 0..n {
            v = (v << 1) | self.bit()? as u64;
        }
        Ok(v as u32)
    }

    /// `u(1)` as a boolean.
    pub fn flag(&mut self) -> Result<bool> {
        Ok(self.bit()? != 0)
    }

    /// Skip `n` bits.
    pub fn skip(&mut self, n: u32) -> Result<()> {
        for _ in 0..n {
            self.bit()?;
        }
        Ok(())
    }

    /// `ue(v)`.
    pub fn ue(&mut self) -> Result<u32> {
        let mut leading_zeros = 0u32;
        while self.bit()? == 0 {
            leading_zeros += 1;
            if leading_zeros > 31 {
                return Err(invalid("invalid Exp-Golomb code"));
            }
        }
        if leading_zeros == 0 {
            return Ok(0);
        }
        let suffix = self.bits(leading_zeros)? as u64;
        Ok(((1u64 << leading_zeros) - 1 + suffix) as u32)
    }

    /// `ue(v)` that must not exceed `max`; `what` names the syntax element.
    pub fn ue_max(&mut self, max: u32, what: &str) -> Result<u32> {
        let v = self.ue()?;
        if v > max {
            return Err(invalid(format_args!("{what} {v} exceeds {max}")));
        }
        Ok(v)
    }

    /// `se(v)`.
    pub fn se(&mut self) -> Result<i32> {
        let code = self.ue()? as i64;
        // Mapping: 0 -> 0, 1 -> 1, 2 -> -1, 3 -> 2, 4 -> -2, ...
        let value = (code + 1) / 2;
        Ok(if code % 2 == 0 { -value } else { value } as i32)
    }

    /// `se(v)` that must lie within `min..=max`.
    pub fn se_range(&mut self, min: i32, max: i32, what: &str) -> Result<i32> {
        let v = self.se()?;
        if v < min || v > max {
            return Err(invalid(format_args!("{what} {v} outside {min}..={max}")));
        }
        Ok(v)
    }
}

/// The general part of `profile_tier_level()` (clause 7.3.3), which is what
/// `StdVideoH265ProfileTierLevel` carries. Sub-layer profiles and levels are
/// parsed past but not kept.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ProfileTierLevel {
    pub general_profile_space: u8,
    pub general_tier_flag: bool,
    /// 1 = Main, 2 = Main 10, 3 = Main Still Picture, 4 = format range
    /// extensions, and so on.
    pub general_profile_idc: u8,
    /// `general_profile_compatibility_flag[j]`, bit 31 - j.
    pub general_profile_compatibility_flags: u32,
    pub general_progressive_source_flag: bool,
    pub general_interlaced_source_flag: bool,
    pub general_non_packed_constraint_flag: bool,
    pub general_frame_only_constraint_flag: bool,
    /// Raw `general_level_idc`: 30 times the level number (level 4.1 is 123).
    /// `StdVideoH265LevelIdc` is an enumeration instead, so it needs mapping.
    pub general_level_idc: u8,
}

/// Parse `profile_tier_level(1, max_sub_layers_minus1)`.
pub(crate) fn parse_profile_tier_level(
    r: &mut Reader,
    max_sub_layers_minus1: u8,
) -> Result<ProfileTierLevel> {
    let mut ptl = ProfileTierLevel {
        general_profile_space: r.bits(2)? as u8,
        general_tier_flag: r.flag()?,
        general_profile_idc: r.bits(5)? as u8,
        general_profile_compatibility_flags: r.bits(32)?,
        general_progressive_source_flag: r.flag()?,
        general_interlaced_source_flag: r.flag()?,
        general_non_packed_constraint_flag: r.flag()?,
        general_frame_only_constraint_flag: r.flag()?,
        ..Default::default()
    };
    // 43 bits of profile-specific constraint flags, then general_inbld_flag
    // (or a reserved bit). None of them matter to the decoder.
    r.skip(44)?;
    ptl.general_level_idc = r.bits(8)? as u8;

    let sub_layers = max_sub_layers_minus1 as usize;
    let mut profile_present = [false; MAX_SUB_LAYERS];
    let mut level_present = [false; MAX_SUB_LAYERS];
    for i in 0..sub_layers {
        profile_present[i] = r.flag()?;
        level_present[i] = r.flag()?;
    }
    if sub_layers > 0 {
        // reserved_zero_2bits for each unused sub-layer slot up to 8.
        r.skip(2 * (8 - sub_layers as u32))?;
    }
    for i in 0..sub_layers {
        if profile_present[i] {
            // Profile space, tier, profile idc, 32 compatibility flags, the
            // four source flags and 44 constraint/reserved bits.
            r.skip(2 + 1 + 5 + 32 + 4 + 44)?;
        }
        if level_present[i] {
            r.skip(8)?;
        }
    }
    Ok(ptl)
}

/// Per-sub-layer DPB sizing, as `StdVideoH265DecPicBufMgr` carries it.
///
/// Every entry is filled: when the bitstream only codes the highest
/// sub-layer (`*_sub_layer_ordering_info_present_flag` = 0), the lower ones
/// are inferred equal to it (clause 7.4.3.2.1).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct DecPicBufMgr {
    pub max_dec_pic_buffering_minus1: [u8; MAX_SUB_LAYERS],
    pub max_num_reorder_pics: [u8; MAX_SUB_LAYERS],
    pub max_latency_increase_plus1: [u32; MAX_SUB_LAYERS],
}

/// Parse the sub-layer ordering loop shared by the VPS and SPS.
pub(crate) fn parse_dec_pic_buf_mgr(
    r: &mut Reader,
    max_sub_layers_minus1: u8,
    info_present: bool,
) -> Result<DecPicBufMgr> {
    let mut mgr = DecPicBufMgr::default();
    let highest = max_sub_layers_minus1 as usize;
    let first = if info_present { 0 } else { highest };
    for i in first..=highest {
        // MaxDpbSize is at most 16, so max_dec_pic_buffering_minus1 <= 15.
        mgr.max_dec_pic_buffering_minus1[i] = r.ue_max(15, "max_dec_pic_buffering_minus1")? as u8;
        mgr.max_num_reorder_pics[i] = r.ue_max(
            mgr.max_dec_pic_buffering_minus1[i] as u32,
            "max_num_reorder_pics",
        )? as u8;
        mgr.max_latency_increase_plus1[i] = r.ue()?;
    }
    for i in 0..first {
        mgr.max_dec_pic_buffering_minus1[i] = mgr.max_dec_pic_buffering_minus1[highest];
        mgr.max_num_reorder_pics[i] = mgr.max_num_reorder_pics[highest];
        mgr.max_latency_increase_plus1[i] = mgr.max_latency_increase_plus1[highest];
    }
    Ok(mgr)
}

/// Parse `hrd_parameters()` (clause E.2.2) only to advance past it.
///
/// Vulkan decode does not use HRD information, and it is large, so nothing is
/// kept; but it sits in the middle of the VPS and the VUI, so it has to be
/// parsed exactly.
pub(crate) fn skip_hrd_parameters(
    r: &mut Reader,
    common_inf_present: bool,
    max_sub_layers_minus1: u8,
) -> Result<()> {
    let mut nal_hrd = false;
    let mut vcl_hrd = false;
    let mut sub_pic_hrd_params_present = false;
    if common_inf_present {
        nal_hrd = r.flag()?;
        vcl_hrd = r.flag()?;
        if nal_hrd || vcl_hrd {
            sub_pic_hrd_params_present = r.flag()?;
            if sub_pic_hrd_params_present {
                // tick_divisor_minus2, du_cpb_removal_delay_increment_length_minus1,
                // sub_pic_cpb_params_in_pic_timing_sei_flag,
                // dpb_output_delay_du_length_minus1.
                r.skip(8 + 5 + 1 + 5)?;
            }
            // bit_rate_scale, cpb_size_scale.
            r.skip(4 + 4)?;
            if sub_pic_hrd_params_present {
                // cpb_size_du_scale.
                r.skip(4)?;
            }
            // initial_cpb_removal_delay_length_minus1,
            // au_cpb_removal_delay_length_minus1, dpb_output_delay_length_minus1.
            r.skip(5 + 5 + 5)?;
        }
    }

    for _ in 0..=max_sub_layers_minus1 {
        let fixed_pic_rate_general = r.flag()?;
        // Inferred to be 1 when fixed_pic_rate_general_flag is 1.
        let fixed_pic_rate_within_cvs = fixed_pic_rate_general || r.flag()?;
        let mut low_delay_hrd = false;
        if fixed_pic_rate_within_cvs {
            let _elemental_duration_in_tc_minus1 = r.ue()?;
        } else {
            low_delay_hrd = r.flag()?;
        }
        let mut cpb_cnt_minus1 = 0;
        if !low_delay_hrd {
            cpb_cnt_minus1 = r.ue_max(31, "cpb_cnt_minus1")?;
        }
        for present in [nal_hrd, vcl_hrd] {
            if !present {
                continue;
            }
            // sub_layer_hrd_parameters().
            for _ in 0..=cpb_cnt_minus1 {
                let _bit_rate_value_minus1 = r.ue()?;
                let _cpb_size_value_minus1 = r.ue()?;
                if sub_pic_hrd_params_present {
                    let _cpb_size_du_value_minus1 = r.ue()?;
                    let _bit_rate_du_value_minus1 = r.ue()?;
                }
                let _cbr_flag = r.flag()?;
            }
        }
    }
    Ok(())
}

/// Timing information from a VPS (`vps_timing_info_present_flag`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct VpsTiming {
    pub num_units_in_tick: u32,
    pub time_scale: u32,
    pub poc_proportional_to_timing_flag: bool,
    pub num_ticks_poc_diff_one_minus1: u32,
}

/// Parsed video parameter set (clause 7.3.2.1): the fields
/// `StdVideoH265VideoParameterSet` needs. HRD parameters are parsed past and
/// dropped, and the VPS extension is not parsed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Vps {
    pub vps_video_parameter_set_id: u8,
    pub vps_base_layer_internal_flag: bool,
    pub vps_base_layer_available_flag: bool,
    pub vps_max_layers_minus1: u8,
    pub vps_max_sub_layers_minus1: u8,
    pub vps_temporal_id_nesting_flag: bool,
    pub profile_tier_level: ProfileTierLevel,
    pub vps_sub_layer_ordering_info_present_flag: bool,
    pub dec_pic_buf_mgr: DecPicBufMgr,
    pub vps_max_layer_id: u8,
    pub vps_num_layer_sets_minus1: u32,
    /// Present when `vps_timing_info_present_flag` is set.
    pub timing: Option<VpsTiming>,
    pub vps_num_hrd_parameters: u32,
    pub vps_extension_flag: bool,
}

/// Parse a VPS NAL payload (EBSP, after the 2-byte NAL header).
pub(crate) fn parse_vps(payload: &[u8]) -> Result<Vps> {
    let mut r = Reader::new(payload);
    let mut vps = Vps {
        vps_video_parameter_set_id: r.bits(4)? as u8,
        vps_base_layer_internal_flag: r.flag()?,
        vps_base_layer_available_flag: r.flag()?,
        vps_max_layers_minus1: r.bits(6)? as u8,
        vps_max_sub_layers_minus1: r.bits(3)? as u8,
        vps_temporal_id_nesting_flag: r.flag()?,
        ..Default::default()
    };
    if vps.vps_max_sub_layers_minus1 as usize >= MAX_SUB_LAYERS {
        return Err(invalid("VPS: vps_max_sub_layers_minus1 is 7"));
    }
    let _vps_reserved_0xffff_16bits = r.bits(16)?;
    vps.profile_tier_level = parse_profile_tier_level(&mut r, vps.vps_max_sub_layers_minus1)?;
    vps.vps_sub_layer_ordering_info_present_flag = r.flag()?;
    vps.dec_pic_buf_mgr = parse_dec_pic_buf_mgr(
        &mut r,
        vps.vps_max_sub_layers_minus1,
        vps.vps_sub_layer_ordering_info_present_flag,
    )?;
    vps.vps_max_layer_id = r.bits(6)? as u8;
    vps.vps_num_layer_sets_minus1 = r.ue_max(1023, "vps_num_layer_sets_minus1")?;
    for _ in 1..=vps.vps_num_layer_sets_minus1 {
        // layer_id_included_flag[i][j] for j in 0..=vps_max_layer_id.
        r.skip(vps.vps_max_layer_id as u32 + 1)?;
    }
    if r.flag()? {
        let mut timing = VpsTiming {
            num_units_in_tick: r.bits(32)?,
            time_scale: r.bits(32)?,
            poc_proportional_to_timing_flag: r.flag()?,
            num_ticks_poc_diff_one_minus1: 0,
        };
        if timing.poc_proportional_to_timing_flag {
            timing.num_ticks_poc_diff_one_minus1 = r.ue()?;
        }
        vps.timing = Some(timing);
        vps.vps_num_hrd_parameters =
            r.ue_max(vps.vps_num_layer_sets_minus1 + 1, "vps_num_hrd_parameters")?;
        for i in 0..vps.vps_num_hrd_parameters {
            let _hrd_layer_set_idx = r.ue()?;
            // cprms_present_flag[0] is inferred to be 1.
            let cprms_present = i == 0 || r.flag()?;
            skip_hrd_parameters(&mut r, cprms_present, vps.vps_max_sub_layers_minus1)?;
        }
    }
    vps.vps_extension_flag = r.flag()?;
    Ok(vps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nal_iteration_and_header() {
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0xAA, // VPS, layer 0, tid 0
            0x00, 0x00, 0x01, 0x26, 0x01, 0xBB, // IDR_W_RADL
            0x00, 0x00, 0x01, 0x02, 0x03, 0xCC, // TRAIL_R, tid 2
            0x00, 0x00, 0x01, 0x80, 0x01, // forbidden bit set: skipped
            0x00, 0x00, 0x01, 0x4F, 0xF9, 0xDD, // SEI prefix, nuh_layer_id 63
        ];
        let nals: Vec<_> = iter_nal_units(&data).collect();
        assert_eq!(nals.len(), 4);
        assert_eq!(nals[0].nal_type(), NalType::Vps);
        assert_eq!(nals[0].payload(), &[0xAA]);
        assert_eq!(nals[1].nal_type(), NalType::IdrWRadl);
        assert!(nals[1].nal_type().is_irap() && nals[1].nal_type().is_idr());
        assert_eq!(nals[2].nal_type(), NalType::TrailR);
        assert_eq!(nals[2].header.temporal_id, 2);
        // 0x4F 0xF9: type 39, layer id (1 << 5) | 31 = 63, tid 0.
        assert_eq!(nals[3].nal_type(), NalType::SeiPrefix);
        assert_eq!(nals[3].header.layer_id, 63);
    }

    #[test]
    fn nal_type_classes() {
        for raw in 0..64u8 {
            let t = NalType::from_raw(raw);
            assert_eq!(t.raw(), raw, "round trip of {raw}");
            assert_eq!(t.is_vcl(), raw < 32);
            assert_eq!(t.is_irap(), (16..=23).contains(&raw));
            assert_eq!(t.is_idr(), raw == 19 || raw == 20);
            assert_eq!(t.is_rasl(), raw == 8 || raw == 9);
            assert_eq!(t.is_radl(), raw == 6 || raw == 7);
        }
        assert!(NalType::TrailN.is_sub_layer_non_reference());
        assert!(NalType::RaslN.is_sub_layer_non_reference());
        assert!(!NalType::TrailR.is_sub_layer_non_reference());
        assert!(!NalType::Cra.is_sub_layer_non_reference());
        assert!(NalType::Other(14).is_sub_layer_non_reference());
    }

    #[test]
    fn reader_counts_rbsp_bits_not_emulation_bytes() {
        // 00 00 03 01: the 03 is not part of the RBSP.
        let data = [0x00, 0x00, 0x03, 0x01];
        let mut r = Reader::new(&data);
        assert_eq!(r.bits(24).unwrap(), 1);
        assert_eq!(r.position(), 24);
    }

    #[test]
    fn reader_exp_golomb() {
        // ue: 1 -> 0, 010 -> 1, 011 -> 2; se: 00100 -> code 3 -> +2,
        // 00101 -> code 4 -> -2.
        let data = [0b1010_0110, 0b0100_0010, 0b1000_0000];
        let mut r = Reader::new(&data);
        assert_eq!(r.ue().unwrap(), 0);
        assert_eq!(r.ue().unwrap(), 1);
        assert_eq!(r.ue().unwrap(), 2);
        assert_eq!(r.se().unwrap(), 2);
        assert_eq!(r.se().unwrap(), -2);
        assert_eq!(r.position(), 17);
    }

    #[test]
    fn ceil_log2_values() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(9), 4);
    }
}
