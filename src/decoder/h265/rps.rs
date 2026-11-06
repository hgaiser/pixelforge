//! `st_ref_pic_set()` (clause 7.3.7) and its semantics (clause 7.4.8).
//!
//! Each set is kept twice: as the syntax that was coded, which is what
//! `StdVideoH265ShortTermRefPicSet` carries, and as the derived
//! `DeltaPocS0`/`DeltaPocS1`/`UsedByCurrPicS0`/`UsedByCurrPicS1` lists, which
//! is what the reference picture set derivation (clause 8.3.2) consumes. With
//! inter RPS prediction the two differ: the syntax is a delta against another
//! set, and only the derived form says which pictures are meant.

use super::parser::{Reader, invalid};
use crate::error::Result;

/// The most entries a short-term set can have (`MaxDpbSize - 1` is at most 15,
/// and Vulkan sizes its arrays at 16).
pub(crate) const MAX_ST_REFS: usize = 16;

/// `st_ref_pic_set()` as coded.
///
/// Bit masks hold one flag per entry, bit `j` for entry `j`, as the Vulkan
/// structure does. Only the fields of the branch that was coded are set: the
/// inter prediction fields when `inter_ref_pic_set_prediction_flag` is 1, the
/// explicit lists otherwise.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct StRefPicSetSyntax {
    pub inter_ref_pic_set_prediction_flag: bool,
    /// Coded only in a slice header; 0 (inferred) in the SPS.
    pub delta_idx_minus1: u32,
    pub delta_rps_sign: bool,
    pub abs_delta_rps_minus1: u32,
    /// `used_by_curr_pic_flag[j]` for j in `0..=NumDeltaPocs[RefRpsIdx]`.
    ///
    /// That can be 17 entries, one more than the `u16` Vulkan uses holds; a
    /// set that needs bit 16 cannot be described to Vulkan exactly.
    pub used_by_curr_pic_flag: u32,
    /// `use_delta_flag[j]`, with the inferred 1s filled in.
    pub use_delta_flag: u32,
    pub num_negative_pics: u8,
    pub num_positive_pics: u8,
    pub delta_poc_s0_minus1: [u16; MAX_ST_REFS],
    pub delta_poc_s1_minus1: [u16; MAX_ST_REFS],
    pub used_by_curr_pic_s0_flag: u16,
    pub used_by_curr_pic_s1_flag: u16,
}

/// The derived form of a short-term reference picture set (equations 7-61
/// to 7-64): POC deltas relative to the current picture, closest first.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct StRps {
    /// `DeltaPocS0`, all negative, in decreasing order.
    pub delta_poc_s0: Vec<i32>,
    /// `UsedByCurrPicS0`.
    pub used_s0: Vec<bool>,
    /// `DeltaPocS1`, all positive, in increasing order.
    pub delta_poc_s1: Vec<i32>,
    /// `UsedByCurrPicS1`.
    pub used_s1: Vec<bool>,
}

impl StRps {
    pub fn num_negative_pics(&self) -> usize {
        self.delta_poc_s0.len()
    }

    pub fn num_positive_pics(&self) -> usize {
        self.delta_poc_s1.len()
    }

    /// `NumDeltaPocs`.
    pub fn num_delta_pocs(&self) -> usize {
        self.delta_poc_s0.len() + self.delta_poc_s1.len()
    }
}

/// One short-term reference picture set, coded and derived.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct StRefPicSet {
    pub syntax: StRefPicSetSyntax,
    pub rps: StRps,
    /// `RefRpsIdx` when inter predicted, which Vulkan's
    /// `NumDeltaPocsOfRefRpsIdx` is derived from.
    pub ref_rps_idx: Option<usize>,
}

/// Parse `st_ref_pic_set(st_rps_idx)`.
///
/// `previous` holds the sets already parsed from the SPS (all of them, when
/// this is the set in a slice header, where `st_rps_idx` equals
/// `num_short_term_ref_pic_sets`). `max_refs` bounds the set's size; it is
/// `sps_max_dec_pic_buffering_minus1` of the highest sub-layer.
pub(crate) fn parse_st_ref_pic_set(
    r: &mut Reader,
    st_rps_idx: usize,
    num_short_term_ref_pic_sets: usize,
    previous: &[StRefPicSet],
    max_refs: usize,
) -> Result<StRefPicSet> {
    let mut set = StRefPicSet::default();
    let syntax = &mut set.syntax;
    if st_rps_idx != 0 {
        syntax.inter_ref_pic_set_prediction_flag = r.flag()?;
    }

    if syntax.inter_ref_pic_set_prediction_flag {
        if st_rps_idx == num_short_term_ref_pic_sets {
            syntax.delta_idx_minus1 = r.ue_max(st_rps_idx as u32 - 1, "delta_idx_minus1")?;
        }
        syntax.delta_rps_sign = r.flag()?;
        syntax.abs_delta_rps_minus1 = r.ue_max(32767, "abs_delta_rps_minus1")?;
        let ref_rps_idx = st_rps_idx - (syntax.delta_idx_minus1 as usize + 1);
        let reference = &previous
            .get(ref_rps_idx)
            .ok_or_else(|| invalid("st_ref_pic_set references a missing set"))?
            .rps;
        set.ref_rps_idx = Some(ref_rps_idx);

        let num_delta_pocs = reference.num_delta_pocs();
        for j in 0..=num_delta_pocs {
            let used = r.flag()?;
            // use_delta_flag is inferred to be 1 when not present.
            let use_delta = used || r.flag()?;
            syntax.used_by_curr_pic_flag |= (used as u32) << j;
            syntax.use_delta_flag |= (use_delta as u32) << j;
        }
        let delta_rps =
            (1 - 2 * syntax.delta_rps_sign as i32) * (syntax.abs_delta_rps_minus1 as i32 + 1);
        set.rps = predict_rps(
            reference,
            delta_rps,
            syntax.used_by_curr_pic_flag,
            syntax.use_delta_flag,
        );
        if set.rps.num_delta_pocs() > max_refs.min(MAX_ST_REFS) {
            return Err(invalid("predicted st_ref_pic_set has too many entries"));
        }
        return Ok(set);
    }

    let limit = max_refs.min(MAX_ST_REFS) as u32;
    syntax.num_negative_pics = r.ue_max(limit, "num_negative_pics")? as u8;
    syntax.num_positive_pics =
        r.ue_max(limit - syntax.num_negative_pics as u32, "num_positive_pics")? as u8;

    let mut poc = 0i32;
    for i in 0..syntax.num_negative_pics as usize {
        let minus1 = r.ue_max(32767, "delta_poc_s0_minus1")?;
        let used = r.flag()?;
        syntax.delta_poc_s0_minus1[i] = minus1 as u16;
        syntax.used_by_curr_pic_s0_flag |= (used as u16) << i;
        poc -= minus1 as i32 + 1;
        set.rps.delta_poc_s0.push(poc);
        set.rps.used_s0.push(used);
    }
    poc = 0;
    for i in 0..syntax.num_positive_pics as usize {
        let minus1 = r.ue_max(32767, "delta_poc_s1_minus1")?;
        let used = r.flag()?;
        syntax.delta_poc_s1_minus1[i] = minus1 as u16;
        syntax.used_by_curr_pic_s1_flag |= (used as u16) << i;
        poc += minus1 as i32 + 1;
        set.rps.delta_poc_s1.push(poc);
        set.rps.used_s1.push(used);
    }
    Ok(set)
}

/// Equations 7-61 and 7-62: derive a set from `reference` shifted by
/// `delta_rps`, keeping the entries `use_delta` selects.
///
/// The flags are indexed as the syntax indexes them: the reference's negative
/// entries first, then its positive entries, then one for the reference
/// picture itself (at `NumDeltaPocs[RefRpsIdx]`).
fn predict_rps(reference: &StRps, delta_rps: i32, used: u32, use_delta: u32) -> StRps {
    let flag = |mask: u32, j: usize| mask & (1 << j) != 0;
    let num_negative = reference.num_negative_pics();
    let num_delta = reference.num_delta_pocs();
    let mut out = StRps::default();

    // Negative pictures, closest first.
    for j in (0..reference.num_positive_pics()).rev() {
        let d_poc = reference.delta_poc_s1[j] + delta_rps;
        if d_poc < 0 && flag(use_delta, num_negative + j) {
            out.delta_poc_s0.push(d_poc);
            out.used_s0.push(flag(used, num_negative + j));
        }
    }
    if delta_rps < 0 && flag(use_delta, num_delta) {
        out.delta_poc_s0.push(delta_rps);
        out.used_s0.push(flag(used, num_delta));
    }
    for j in 0..num_negative {
        let d_poc = reference.delta_poc_s0[j] + delta_rps;
        if d_poc < 0 && flag(use_delta, j) {
            out.delta_poc_s0.push(d_poc);
            out.used_s0.push(flag(used, j));
        }
    }

    // Positive pictures, closest first.
    for j in (0..num_negative).rev() {
        let d_poc = reference.delta_poc_s0[j] + delta_rps;
        if d_poc > 0 && flag(use_delta, j) {
            out.delta_poc_s1.push(d_poc);
            out.used_s1.push(flag(used, j));
        }
    }
    if delta_rps > 0 && flag(use_delta, num_delta) {
        out.delta_poc_s1.push(delta_rps);
        out.used_s1.push(flag(used, num_delta));
    }
    for j in 0..reference.num_positive_pics() {
        let d_poc = reference.delta_poc_s1[j] + delta_rps;
        if d_poc > 0 && flag(use_delta, num_negative + j) {
            out.delta_poc_s1.push(d_poc);
            out.used_s1.push(flag(used, num_negative + j));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rps(s0: &[(i32, bool)], s1: &[(i32, bool)]) -> StRps {
        StRps {
            delta_poc_s0: s0.iter().map(|e| e.0).collect(),
            used_s0: s0.iter().map(|e| e.1).collect(),
            delta_poc_s1: s1.iter().map(|e| e.0).collect(),
            used_s1: s1.iter().map(|e| e.1).collect(),
        }
    }

    /// The reference set {-1, +1, +3} predicted with deltaRps = -2 gives
    /// {-1 - 2, 1 - 2, 3 - 2} plus the reference picture itself at -2, which
    /// sorted closest first is S0 = {-1, -2, -3}, S1 = {+1}.
    #[test]
    fn inter_prediction_shifts_and_resorts() {
        let reference = rps(&[(-1, true)], &[(1, true), (3, true)]);
        // Four flags: S0[0], S1[0], S1[1], then the reference picture.
        let all = 0b1111;
        let out = predict_rps(&reference, -2, all, all);
        assert_eq!(out.delta_poc_s0, vec![-1, -2, -3]);
        assert_eq!(out.delta_poc_s1, vec![1]);
        assert!(out.used_s0.iter().chain(&out.used_s1).all(|u| *u));
    }

    /// use_delta_flag drops entries, and used_by_curr_pic_flag follows each
    /// entry to its new position.
    #[test]
    fn inter_prediction_honours_flags() {
        let reference = rps(&[(-1, true), (-3, true)], &[(2, true)]);
        // deltaRps = +1: S0[0] -> 0 (dropped, never 0), S0[1] -> -2,
        // S1[0] -> 3, the reference picture -> +1.
        // Drop S1[0] with use_delta_flag; mark the reference picture unused.
        let use_delta = 0b1011;
        let used = 0b0011;
        let out = predict_rps(&reference, 1, used, use_delta);
        assert_eq!(out.delta_poc_s0, vec![-2]);
        assert_eq!(out.used_s0, vec![true]);
        assert_eq!(out.delta_poc_s1, vec![1]);
        assert_eq!(out.used_s1, vec![false]);
    }
}
