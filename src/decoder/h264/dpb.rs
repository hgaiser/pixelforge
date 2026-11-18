//! Decode-side H.264 picture order count and reference picture management.
//!
//! The encoder's DPB (`crate::encoder::dpb`) decides *which* pictures to
//! reference; a decoder instead reconstructs the state the encoder implied from
//! the bitstream syntax. That makes this a separate, much smaller piece of
//! bookkeeping: compute POC, hand the driver the current reference set, and
//! retire pictures.
//!
//! Retirement follows whichever process the bitstream asks for: the sliding
//! window (clause 8.2.5.3), or the explicit MMCO commands in
//! `dec_ref_pic_marking()` (clause 8.2.5.4). Both are required -- x264 emits
//! MMCO whenever B-pyramid is enabled, and applying the sliding window in its
//! place silently decodes the wrong pictures.

use std::sync::Arc;

use crate::decoder::frames::SlotPins;
use crate::decoder::h264::parser::{Mmco, NalType, RefPicMarking, SliceHeader, Sps};
use crate::error::{PixelForgeError, Result};

/// Maximum DPB slots addressable by the H.264 spec (16 frames + 1 current).
pub(crate) const MAX_DPB_SLOTS: usize = 17;

/// A picture currently held in the DPB as a reference.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RefPicture {
    /// DPB slot index this picture lives in.
    pub slot: u8,
    /// `frame_num` from the slice header.
    pub frame_num: u16,
    /// Wrapped frame number used for sliding-window ordering (`FrameNumWrap`).
    ///
    /// For frame pictures this is also `PicNum`, which is what MMCO operands
    /// are expressed in, so it must be refreshed against the current picture
    /// before any marking is applied (clause 8.2.4.1).
    pub frame_num_wrap: i32,
    /// Picture order count (top and bottom; identical for frame pictures).
    pub poc: [i32; 2],
    /// Whether this is a long-term reference.
    pub long_term: bool,
    /// `LongTermFrameIdx`, meaningful only when `long_term` is set. For frame
    /// pictures this doubles as `LongTermPicNum`.
    pub long_term_frame_idx: u32,
}

/// Per-picture POC and reference state derived from the bitstream.
#[derive(Debug, Clone)]
pub(crate) struct PictureState {
    /// Picture order count of the current picture.
    pub poc: i32,
    /// `frame_num` of the current picture.
    pub frame_num: u16,
    /// Whether the current picture is an IDR.
    pub is_idr: bool,
    /// Whether the current picture is used for reference (`nal_ref_idc != 0`).
    pub is_reference: bool,
    /// Whether the current picture contains only intra slices.
    pub is_intra: bool,
    /// `idr_pic_id` (only meaningful for IDR pictures).
    pub idr_pic_id: u16,
    /// `dec_ref_pic_marking()` from the slice header, driving how this picture
    /// retires earlier references once decoding completes.
    ///
    /// A picture carrying MMCO 5 already has `poc` and `frame_num` rebased to
    /// zero by [`DecodeDpb::begin_picture`].
    pub marking: RefPicMarking,
}

/// Decode-side DPB: POC state machine plus the reference picture set.
pub(crate) struct DecodeDpb {
    /// Slots reserved by decoded frames the caller still holds. Never handed
    /// out for a new picture, however the reference rules mark them.
    pins: Arc<SlotPins>,
    /// Reference pictures, most recently decoded last.
    refs: Vec<RefPicture>,
    /// Which slots are occupied (by a reference or the in-flight picture).
    slot_used: [bool; MAX_DPB_SLOTS],
    /// Number of slots the session was created with.
    slot_count: usize,
    /// `MaxLongTermFrameIdx`, or `None` for "no long-term frame indices"
    /// (the initial state and the state after an IDR without
    /// `long_term_reference_flag`). Set by MMCO 4.
    max_long_term_frame_idx: Option<u32>,

    // --- POC type 0 state (8.2.1.1) ---
    prev_poc_msb: i32,
    prev_poc_lsb: i32,
    // --- POC type 1/2 state (8.2.1.2, 8.2.1.3) ---
    prev_frame_num_offset: i32,
    prev_frame_num: u32,
}

impl DecodeDpb {
    pub fn new(slot_count: usize, pins: Arc<SlotPins>) -> Self {
        Self {
            pins,
            refs: Vec::new(),
            slot_used: [false; MAX_DPB_SLOTS],
            slot_count: slot_count.min(MAX_DPB_SLOTS),
            max_long_term_frame_idx: None,
            prev_poc_msb: 0,
            prev_poc_lsb: 0,
            prev_frame_num_offset: 0,
            prev_frame_num: 0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.slot_count, self.pins.clone());
    }

    /// Current reference pictures, in the order they were decoded.
    pub fn references(&self) -> &[RefPicture] {
        &self.refs
    }

    /// Compute the POC and reference state for the picture about to be decoded.
    ///
    /// Implements the frame-picture subset of clause 8.2.1. Must be called
    /// exactly once per picture, before [`Self::allocate_slot`].
    pub fn begin_picture(
        &mut self,
        nal_type: NalType,
        ref_idc: u8,
        header: &SliceHeader,
        sps: &Sps,
        is_intra: bool,
    ) -> Result<PictureState> {
        if header.field_pic_flag {
            return Err(PixelForgeError::InvalidInput(
                "H.264 decode: field/interlaced pictures are not supported".to_string(),
            ));
        }

        let is_idr = nal_type == NalType::IdrSlice;
        let is_reference = ref_idc != 0;

        if is_idr {
            // An IDR empties the DPB and resets all POC state.
            self.refs.clear();
            self.slot_used = [false; MAX_DPB_SLOTS];
            self.prev_poc_msb = 0;
            self.prev_poc_lsb = 0;
            self.prev_frame_num_offset = 0;
            self.prev_frame_num = 0;
        }

        let mut poc = match sps.pic_order_cnt_type {
            0 => self.compute_poc_type0(header, sps, is_idr, is_reference),
            1 => self.compute_poc_type1(header, sps, is_idr, is_reference),
            2 => self.compute_poc_type2(header, sps, is_idr, is_reference),
            other => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "H.264 decode: unsupported pic_order_cnt_type {}",
                    other
                )));
            }
        };

        self.prev_frame_num = header.frame_num;

        // MMCO 5 makes the current picture behave like the start of a new
        // sequence: it is inferred to have frame_num 0, its POC is rebased so
        // that PicOrderCnt(CurrPic) becomes 0, and the POC predictors reset
        // (clauses 8.2.1 and 8.2.5.4).
        let mmco5 = is_reference && header.marking.ops.contains(&Mmco::ForgetAll);
        let mut frame_num = header.frame_num as u16;
        if mmco5 {
            poc = 0;
            frame_num = 0;
            self.prev_poc_msb = 0;
            self.prev_poc_lsb = 0;
            self.prev_frame_num_offset = 0;
            self.prev_frame_num = 0;
        }

        Ok(PictureState {
            poc,
            frame_num,
            is_idr,
            is_reference,
            is_intra,
            idr_pic_id: header.idr_pic_id,
            marking: header.marking.clone(),
        })
    }

    /// POC type 0 (8.2.1.1): explicit LSB in the slice header, MSB tracked here.
    fn compute_poc_type0(
        &mut self,
        header: &SliceHeader,
        sps: &Sps,
        is_idr: bool,
        is_reference: bool,
    ) -> i32 {
        let max_lsb = sps.max_pic_order_cnt_lsb() as i32;
        let lsb = header.pic_order_cnt_lsb as i32;

        let poc_msb = if is_idr {
            0
        } else if lsb < self.prev_poc_lsb && (self.prev_poc_lsb - lsb) >= max_lsb / 2 {
            self.prev_poc_msb + max_lsb
        } else if lsb > self.prev_poc_lsb && (lsb - self.prev_poc_lsb) > max_lsb / 2 {
            self.prev_poc_msb - max_lsb
        } else {
            self.prev_poc_msb
        };

        // Only reference pictures update the prev* state.
        if is_reference {
            self.prev_poc_msb = poc_msb;
            self.prev_poc_lsb = lsb;
        }

        poc_msb + lsb
    }

    /// Frame number offset shared by POC types 1 and 2 (handles frame_num wrap).
    fn frame_num_offset(&mut self, header: &SliceHeader, sps: &Sps, is_idr: bool) -> i32 {
        let max_frame_num = sps.max_frame_num() as i32;
        let offset = if is_idr {
            0
        } else if (self.prev_frame_num as i32) > (header.frame_num as i32) {
            self.prev_frame_num_offset + max_frame_num
        } else {
            self.prev_frame_num_offset
        };
        self.prev_frame_num_offset = offset;
        offset
    }

    /// POC type 1 (8.2.1.2): POC derived from a repeating cycle in the SPS.
    fn compute_poc_type1(
        &mut self,
        header: &SliceHeader,
        sps: &Sps,
        is_idr: bool,
        is_reference: bool,
    ) -> i32 {
        let frame_num_offset = self.frame_num_offset(header, sps, is_idr);
        let num_in_cycle = sps.offsets_for_ref_frame.len() as i32;

        let abs_frame_num = if num_in_cycle != 0 {
            let n = frame_num_offset + header.frame_num as i32;
            if !is_reference && n > 0 { n - 1 } else { n }
        } else {
            0
        };

        let mut expected_poc = 0i32;
        if abs_frame_num > 0 {
            let cycle: i32 = sps.offsets_for_ref_frame.iter().sum();
            let poc_cycle_cnt = (abs_frame_num - 1) / num_in_cycle;
            let frame_num_in_cycle = (abs_frame_num - 1) % num_in_cycle;
            expected_poc = poc_cycle_cnt * cycle;
            for i in 0..=frame_num_in_cycle {
                expected_poc += sps.offsets_for_ref_frame[i as usize];
            }
        }
        if !is_reference {
            expected_poc += sps.offset_for_non_ref_pic;
        }

        expected_poc + header.delta_pic_order_cnt[0]
    }

    /// POC type 2 (8.2.1.3): POC is a direct function of frame_num (decode order).
    fn compute_poc_type2(
        &mut self,
        header: &SliceHeader,
        sps: &Sps,
        is_idr: bool,
        is_reference: bool,
    ) -> i32 {
        let frame_num_offset = self.frame_num_offset(header, sps, is_idr);
        if is_idr {
            return 0;
        }
        let temp = frame_num_offset + header.frame_num as i32;
        if is_reference { 2 * temp } else { 2 * temp - 1 }
    }

    /// Reserve a DPB slot for the current picture, if one is available.
    ///
    /// The current picture always needs a slot, even when it is not a
    /// reference, because Vulkan decodes into a DPB resource. Slots pinned by
    /// a handed-out frame are skipped: the caller is still reading them.
    /// `None` means every slot is either a live reference or pinned.
    pub fn try_allocate_slot(&mut self) -> Option<u8> {
        for slot in 0..self.slot_count {
            if !self.slot_used[slot] && !self.pins.is_pinned(slot as u8) {
                self.slot_used[slot] = true;
                return Some(slot as u8);
            }
        }
        None
    }

    /// Whether any slot is held by a frame the caller has not dropped, which
    /// means waiting could free one.
    pub fn has_pinned_slots(&self) -> bool {
        self.pins.any_pinned()
    }

    /// Release a slot that was allocated for a non-reference picture.
    pub fn release_slot(&mut self, slot: u8) {
        self.slot_used[slot as usize] = false;
    }

    /// Retire references and insert the just-decoded picture (clause 8.2.5).
    ///
    /// Dispatches to the explicit marking process when the slice header carries
    /// MMCO commands, and to the sliding window otherwise. If
    /// `state.is_reference` is false the picture's slot is released and the
    /// reference set is untouched.
    pub fn end_picture(&mut self, slot: u8, state: &PictureState, sps: &Sps) {
        if !state.is_reference {
            self.release_slot(slot);
            return;
        }

        // PicNum is derived relative to the current picture and MMCO operands
        // are expressed in it, so this must happen before any marking.
        self.refresh_pic_nums(state.frame_num as i32, sps);

        if state.is_idr {
            // begin_picture already emptied the DPB.
            self.refs.clear();
            let long_term = state.marking.long_term_reference_flag;
            self.max_long_term_frame_idx = long_term.then_some(0);
            self.push_current(slot, state, long_term, 0);
            return;
        }

        if state.marking.adaptive {
            self.apply_mmco(slot, state);
        } else {
            self.sliding_window(sps);
            self.push_current(slot, state, false, 0);
        }
    }

    /// Recompute `FrameNumWrap` (== `PicNum` for frames) for every retained
    /// reference, relative to the current picture (clause 8.2.4.1).
    fn refresh_pic_nums(&mut self, current_frame_num: i32, sps: &Sps) {
        let max_frame_num = sps.max_frame_num() as i32;
        for r in &mut self.refs {
            r.frame_num_wrap = if (r.frame_num as i32) > current_frame_num {
                r.frame_num as i32 - max_frame_num
            } else {
                r.frame_num as i32
            };
        }
    }

    /// Sliding-window marking (clause 8.2.5.3): evict the short-term reference
    /// with the smallest `FrameNumWrap` until there is room for the current
    /// picture.
    fn sliding_window(&mut self, sps: &Sps) {
        let max_refs = (sps.max_num_ref_frames as usize).max(1);
        while self.refs.len() >= max_refs {
            let victim = self
                .refs
                .iter()
                .enumerate()
                .filter(|(_, r)| !r.long_term)
                .min_by_key(|(_, r)| r.frame_num_wrap)
                .map(|(i, _)| i);
            match victim {
                Some(i) => {
                    let removed = self.refs.remove(i);
                    self.slot_used[removed.slot as usize] = false;
                }
                // Only long-term refs remain; nothing to slide out.
                None => break,
            }
        }
    }

    /// Explicit marking (clause 8.2.5.4). Encoders that use B-pyramid rely on
    /// this to retire a B-reference at a point the sliding window would not.
    fn apply_mmco(&mut self, slot: u8, state: &PictureState) {
        // CurrPicNum == frame_num for a frame picture (clause 8.2.4.1).
        let curr_pic_num = state.frame_num as i32;
        let mut current_long_term_idx = None;

        for op in &state.marking.ops {
            match *op {
                Mmco::ForgetShort {
                    difference_of_pic_nums_minus1,
                } => {
                    let pic_num_x = curr_pic_num - (difference_of_pic_nums_minus1 as i32 + 1);
                    self.remove_refs(|r| !r.long_term && r.frame_num_wrap == pic_num_x);
                }
                Mmco::ForgetLong { long_term_pic_num } => {
                    self.remove_refs(|r| r.long_term && r.long_term_frame_idx == long_term_pic_num);
                }
                Mmco::ShortToLong {
                    difference_of_pic_nums_minus1,
                    long_term_frame_idx,
                } => {
                    let pic_num_x = curr_pic_num - (difference_of_pic_nums_minus1 as i32 + 1);
                    // A long-term index is unique: whatever holds it is displaced.
                    self.remove_refs(|r| {
                        r.long_term && r.long_term_frame_idx == long_term_frame_idx
                    });
                    if let Some(r) = self
                        .refs
                        .iter_mut()
                        .find(|r| !r.long_term && r.frame_num_wrap == pic_num_x)
                    {
                        r.long_term = true;
                        r.long_term_frame_idx = long_term_frame_idx;
                    }
                }
                Mmco::MaxLongTermIdx {
                    max_long_term_frame_idx_plus1,
                } => {
                    // 0 means "no long-term frame indices", so every long-term
                    // reference goes; otherwise those above the bound go.
                    self.max_long_term_frame_idx = max_long_term_frame_idx_plus1.checked_sub(1);
                    let max = self.max_long_term_frame_idx;
                    self.remove_refs(|r| {
                        r.long_term && max.is_none_or(|m| r.long_term_frame_idx > m)
                    });
                }
                Mmco::ForgetAll => {
                    for r in std::mem::take(&mut self.refs) {
                        self.slot_used[r.slot as usize] = false;
                    }
                    self.max_long_term_frame_idx = None;
                }
                Mmco::CurrentToLong {
                    long_term_frame_idx,
                } => {
                    self.remove_refs(|r| {
                        r.long_term && r.long_term_frame_idx == long_term_frame_idx
                    });
                    current_long_term_idx = Some(long_term_frame_idx);
                }
            }
        }

        match current_long_term_idx {
            Some(idx) => self.push_current(slot, state, true, idx),
            None => self.push_current(slot, state, false, 0),
        }
    }

    /// Drop every reference matching `pred`, freeing its slot.
    fn remove_refs(&mut self, pred: impl Fn(&RefPicture) -> bool) {
        let slot_used = &mut self.slot_used;
        self.refs.retain(|r| {
            let drop_it = pred(r);
            if drop_it {
                slot_used[r.slot as usize] = false;
            }
            !drop_it
        });
    }

    /// Insert the just-decoded picture into the reference set.
    fn push_current(
        &mut self,
        slot: u8,
        state: &PictureState,
        long_term: bool,
        long_term_frame_idx: u32,
    ) {
        self.refs.push(RefPicture {
            slot,
            frame_num: state.frame_num,
            frame_num_wrap: state.frame_num as i32,
            poc: [state.poc, state.poc],
            long_term,
            long_term_frame_idx,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::h264::parser::{SliceHeader, SliceType};

    fn sps(poc_type: u8) -> Sps {
        Sps {
            pic_order_cnt_type: poc_type,
            log2_max_frame_num_minus4: 0,         // MaxFrameNum = 16
            log2_max_pic_order_cnt_lsb_minus4: 0, // MaxPocLsb = 16
            max_num_ref_frames: 2,
            frame_mbs_only_flag: true,
            ..Default::default()
        }
    }

    fn header(frame_num: u32, poc_lsb: u32) -> SliceHeader {
        SliceHeader {
            first_mb_in_slice: 0,
            slice_type: SliceType::P,
            pps_id: 0,
            frame_num,
            field_pic_flag: false,
            idr_pic_id: 0,
            pic_order_cnt_lsb: poc_lsb,
            delta_pic_order_cnt: [0, 0],
            marking: RefPicMarking::default(),
        }
    }

    /// A header carrying explicit reference marking.
    fn header_with_mmco(frame_num: u32, poc_lsb: u32, ops: Vec<Mmco>) -> SliceHeader {
        SliceHeader {
            marking: RefPicMarking {
                adaptive: true,
                ops,
                ..Default::default()
            },
            ..header(frame_num, poc_lsb)
        }
    }

    /// Decode one reference frame picture and return its slot.
    fn decode_ref(dpb: &mut DecodeDpb, sps: &Sps, header: &SliceHeader) -> u8 {
        let state = dpb
            .begin_picture(NalType::Slice, 2, header, sps, false)
            .unwrap();
        let slot = dpb.try_allocate_slot().unwrap();
        dpb.end_picture(slot, &state, sps);
        slot
    }

    /// The (slot, poc, long_term) triples currently held as references.
    fn ref_summary(dpb: &DecodeDpb) -> Vec<(u8, i32, bool)> {
        dpb.references()
            .iter()
            .map(|r| (r.slot, r.poc[0], r.long_term))
            .collect()
    }

    #[test]
    fn test_poc_type0_wraps() {
        let sps = sps(0);
        let mut dpb = DecodeDpb::new(4, Arc::default());
        // IDR at POC 0.
        let s = dpb
            .begin_picture(NalType::IdrSlice, 3, &header(0, 0), &sps, true)
            .unwrap();
        assert_eq!(s.poc, 0);
        // POC lsb increments by 2 per frame: 2, 4, ... 14, then wraps to 0.
        for i in 1..8 {
            let s = dpb
                .begin_picture(NalType::Slice, 2, &header(i, i * 2), &sps, false)
                .unwrap();
            assert_eq!(s.poc, (i * 2) as i32);
        }
        // lsb wraps 14 -> 0: MSB must advance by MaxPocLsb (16).
        let s = dpb
            .begin_picture(NalType::Slice, 2, &header(8, 0), &sps, false)
            .unwrap();
        assert_eq!(s.poc, 16);
        let s = dpb
            .begin_picture(NalType::Slice, 2, &header(9, 2), &sps, false)
            .unwrap();
        assert_eq!(s.poc, 18);
    }

    #[test]
    fn test_poc_type2_is_decode_order() {
        let sps = sps(2);
        let mut dpb = DecodeDpb::new(4, Arc::default());
        let s = dpb
            .begin_picture(NalType::IdrSlice, 3, &header(0, 0), &sps, true)
            .unwrap();
        assert_eq!(s.poc, 0);
        let s = dpb
            .begin_picture(NalType::Slice, 2, &header(1, 0), &sps, false)
            .unwrap();
        assert_eq!(s.poc, 2);
        let s = dpb
            .begin_picture(NalType::Slice, 2, &header(2, 0), &sps, false)
            .unwrap();
        assert_eq!(s.poc, 4);
    }

    #[test]
    fn test_sliding_window_evicts_oldest() {
        let sps = sps(0); // max_num_ref_frames = 2
        let mut dpb = DecodeDpb::new(4, Arc::default());

        // IDR.
        let s = dpb
            .begin_picture(NalType::IdrSlice, 3, &header(0, 0), &sps, true)
            .unwrap();
        let slot = dpb.try_allocate_slot().unwrap();
        dpb.end_picture(slot, &s, &sps);
        assert_eq!(dpb.references().len(), 1);

        // Two more reference frames: window fills, then evicts the IDR.
        for i in 1..=2u32 {
            let s = dpb
                .begin_picture(NalType::Slice, 2, &header(i, i * 2), &sps, false)
                .unwrap();
            let slot = dpb.try_allocate_slot().unwrap();
            dpb.end_picture(slot, &s, &sps);
        }
        assert_eq!(dpb.references().len(), 2);
        let frame_nums: Vec<u16> = dpb.references().iter().map(|r| r.frame_num).collect();
        assert_eq!(frame_nums, vec![1, 2]); // frame_num 0 (the IDR) slid out.
    }

    /// MMCO 1 retires a specific short-term reference that the sliding window
    /// would have kept. This is what B-pyramid streams rely on.
    #[test]
    fn test_mmco1_forgets_specific_short_term_ref() {
        let mut sps = sps(0);
        sps.max_num_ref_frames = 4; // Sliding window would not evict anything.
        let mut dpb = DecodeDpb::new(6, Arc::default());

        let idr = dpb
            .begin_picture(NalType::IdrSlice, 3, &header(0, 0), &sps, true)
            .unwrap();
        let idr_slot = dpb.try_allocate_slot().unwrap();
        dpb.end_picture(idr_slot, &idr, &sps);
        let s1 = decode_ref(&mut dpb, &sps, &header(1, 2));
        decode_ref(&mut dpb, &sps, &header(2, 4));
        assert_eq!(dpb.references().len(), 3);

        // From frame_num 3, difference_of_pic_nums_minus1 = 1 targets
        // PicNum 3 - (1 + 1) = 1, i.e. the frame_num 1 picture.
        let marked = header_with_mmco(
            3,
            6,
            vec![Mmco::ForgetShort {
                difference_of_pic_nums_minus1: 1,
            }],
        );
        decode_ref(&mut dpb, &sps, &marked);

        let frame_nums: Vec<u16> = dpb.references().iter().map(|r| r.frame_num).collect();
        assert_eq!(frame_nums, vec![0, 2, 3], "frame_num 1 should be retired");
        // Its slot must be reusable now.
        assert_eq!(dpb.try_allocate_slot().unwrap(), s1);
    }

    /// MMCO 6 marks the current picture long-term; MMCO 2 later drops it by
    /// LongTermPicNum. Long-term refs are immune to the sliding window.
    #[test]
    fn test_mmco6_and_mmco2_long_term_lifecycle() {
        let sps = sps(0); // max_num_ref_frames = 2
        let mut dpb = DecodeDpb::new(6, Arc::default());

        let idr = dpb
            .begin_picture(NalType::IdrSlice, 3, &header(0, 0), &sps, true)
            .unwrap();
        let slot = dpb.try_allocate_slot().unwrap();
        dpb.end_picture(slot, &idr, &sps);

        // Current picture becomes long-term index 0.
        let marked = header_with_mmco(
            1,
            2,
            vec![Mmco::CurrentToLong {
                long_term_frame_idx: 0,
            }],
        );
        decode_ref(&mut dpb, &sps, &marked);
        assert_eq!(ref_summary(&dpb), vec![(0, 0, false), (1, 2, true)]);

        // A long-term reference survives sliding-window pressure that would
        // otherwise evict the oldest picture.
        decode_ref(&mut dpb, &sps, &header(2, 4));
        decode_ref(&mut dpb, &sps, &header(3, 6));
        assert!(
            dpb.references().iter().any(|r| r.long_term),
            "long-term ref must not slide out"
        );

        // MMCO 2 drops it explicitly by LongTermPicNum.
        let marked = header_with_mmco(
            4,
            8,
            vec![Mmco::ForgetLong {
                long_term_pic_num: 0,
            }],
        );
        decode_ref(&mut dpb, &sps, &marked);
        assert!(
            !dpb.references().iter().any(|r| r.long_term),
            "long-term ref should be gone"
        );
    }

    /// MMCO 3 converts a short-term reference into a long-term one in place.
    #[test]
    fn test_mmco3_converts_short_to_long_term() {
        let mut sps = sps(0);
        sps.max_num_ref_frames = 4;
        let mut dpb = DecodeDpb::new(6, Arc::default());

        let idr = dpb
            .begin_picture(NalType::IdrSlice, 3, &header(0, 0), &sps, true)
            .unwrap();
        let slot = dpb.try_allocate_slot().unwrap();
        dpb.end_picture(slot, &idr, &sps);
        decode_ref(&mut dpb, &sps, &header(1, 2));

        // From frame_num 2, target PicNum 2 - (0 + 1) = 1.
        let marked = header_with_mmco(
            2,
            4,
            vec![Mmco::ShortToLong {
                difference_of_pic_nums_minus1: 0,
                long_term_frame_idx: 3,
            }],
        );
        decode_ref(&mut dpb, &sps, &marked);

        let converted = dpb
            .references()
            .iter()
            .find(|r| r.frame_num == 1)
            .expect("frame_num 1 retained");
        assert!(converted.long_term);
        assert_eq!(converted.long_term_frame_idx, 3);
    }

    /// MMCO 4 bounds LongTermFrameIdx; a bound of 0 (plus1 == 0) clears them all.
    #[test]
    fn test_mmco4_bounds_long_term_indices() {
        let mut sps = sps(0);
        sps.max_num_ref_frames = 4;
        let mut dpb = DecodeDpb::new(6, Arc::default());

        let idr = dpb
            .begin_picture(NalType::IdrSlice, 3, &header(0, 0), &sps, true)
            .unwrap();
        let slot = dpb.try_allocate_slot().unwrap();
        dpb.end_picture(slot, &idr, &sps);

        let marked = header_with_mmco(
            1,
            2,
            vec![Mmco::CurrentToLong {
                long_term_frame_idx: 2,
            }],
        );
        decode_ref(&mut dpb, &sps, &marked);
        assert!(dpb.references().iter().any(|r| r.long_term_frame_idx == 2));

        // max_long_term_frame_idx_plus1 = 2 => MaxLongTermFrameIdx = 1, so
        // index 2 is out of range and must go.
        let marked = header_with_mmco(
            2,
            4,
            vec![Mmco::MaxLongTermIdx {
                max_long_term_frame_idx_plus1: 2,
            }],
        );
        decode_ref(&mut dpb, &sps, &marked);
        assert!(
            !dpb.references().iter().any(|r| r.long_term),
            "long-term index above the bound must be retired"
        );
    }

    /// MMCO 5 empties the DPB and rebases frame_num/POC to zero.
    #[test]
    fn test_mmco5_resets_dpb_and_poc() {
        let sps = sps(0);
        let mut dpb = DecodeDpb::new(6, Arc::default());

        let idr = dpb
            .begin_picture(NalType::IdrSlice, 3, &header(0, 0), &sps, true)
            .unwrap();
        let slot = dpb.try_allocate_slot().unwrap();
        dpb.end_picture(slot, &idr, &sps);
        decode_ref(&mut dpb, &sps, &header(1, 2));
        assert_eq!(dpb.references().len(), 2);

        let marked = header_with_mmco(2, 4, vec![Mmco::ForgetAll]);
        let state = dpb
            .begin_picture(NalType::Slice, 2, &marked, &sps, false)
            .unwrap();
        // The picture is rebased rather than keeping poc 4 / frame_num 2.
        assert_eq!(state.poc, 0);
        assert_eq!(state.frame_num, 0);
        let slot = dpb.try_allocate_slot().unwrap();
        dpb.end_picture(slot, &state, &sps);

        // Everything prior is gone; only the rebased picture remains.
        assert_eq!(ref_summary(&dpb), vec![(slot, 0, false)]);

        // POC prediction continues from the reset state.
        let next = dpb
            .begin_picture(NalType::Slice, 2, &header(1, 2), &sps, false)
            .unwrap();
        assert_eq!(next.poc, 2);
    }

    /// Marking is skipped entirely for a picture that is not a reference.
    #[test]
    fn test_non_reference_picture_ignores_marking() {
        let sps = sps(0);
        let mut dpb = DecodeDpb::new(4, Arc::default());
        let idr = dpb
            .begin_picture(NalType::IdrSlice, 3, &header(0, 0), &sps, true)
            .unwrap();
        let slot = dpb.try_allocate_slot().unwrap();
        dpb.end_picture(slot, &idr, &sps);

        // nal_ref_idc == 0, so dec_ref_pic_marking() is not even present.
        let marked = header_with_mmco(1, 2, vec![Mmco::ForgetAll]);
        let state = dpb
            .begin_picture(NalType::Slice, 0, &marked, &sps, false)
            .unwrap();
        let slot = dpb.try_allocate_slot().unwrap();
        dpb.end_picture(slot, &state, &sps);
        assert_eq!(dpb.references().len(), 1, "IDR must survive");
    }

    #[test]
    fn test_non_reference_picture_releases_slot() {
        let sps = sps(0);
        let mut dpb = DecodeDpb::new(4, Arc::default());
        let s = dpb
            .begin_picture(NalType::IdrSlice, 3, &header(0, 0), &sps, true)
            .unwrap();
        let slot = dpb.try_allocate_slot().unwrap();
        dpb.end_picture(slot, &s, &sps);

        // nal_ref_idc == 0 => disposable picture.
        let s = dpb
            .begin_picture(NalType::Slice, 0, &header(1, 2), &sps, false)
            .unwrap();
        assert!(!s.is_reference);
        let slot = dpb.try_allocate_slot().unwrap();
        dpb.end_picture(slot, &s, &sps);
        assert_eq!(dpb.references().len(), 1);
        // The slot is reusable immediately.
        assert_eq!(dpb.try_allocate_slot().unwrap(), slot);
    }

    #[test]
    fn reset_restores_initial_dpb_and_keeps_pins() {
        let sps = sps(1);
        let pins = Arc::new(SlotPins::default());
        let mut dpb = DecodeDpb::new(4, pins.clone());
        let slot = decode_ref(&mut dpb, &sps, &header(15, 0));
        pins.pin(slot);
        decode_ref(
            &mut dpb,
            &sps,
            &header_with_mmco(
                1,
                0,
                vec![Mmco::MaxLongTermIdx {
                    max_long_term_frame_idx_plus1: 2,
                }],
            ),
        );
        assert_eq!(dpb.prev_frame_num_offset, 16);
        assert_eq!(dpb.max_long_term_frame_idx, Some(1));

        dpb.reset();
        assert!(dpb.references().is_empty());
        assert_eq!(dpb.prev_frame_num_offset, 0);
        assert_eq!(dpb.prev_frame_num, 0);
        assert_eq!(dpb.max_long_term_frame_idx, None);
        assert_eq!(dpb.try_allocate_slot(), Some(1));
    }

    #[test]
    fn test_idr_flushes_dpb() {
        let sps = sps(0);
        let mut dpb = DecodeDpb::new(4, Arc::default());
        for i in 0..2u32 {
            let nal = if i == 0 {
                NalType::IdrSlice
            } else {
                NalType::Slice
            };
            let s = dpb
                .begin_picture(nal, 3, &header(i, i * 2), &sps, i == 0)
                .unwrap();
            let slot = dpb.try_allocate_slot().unwrap();
            dpb.end_picture(slot, &s, &sps);
        }
        assert_eq!(dpb.references().len(), 2);

        let s = dpb
            .begin_picture(NalType::IdrSlice, 3, &header(0, 0), &sps, true)
            .unwrap();
        assert_eq!(s.poc, 0);
        assert!(dpb.references().is_empty());
        // All slots freed by the IDR.
        assert_eq!(dpb.try_allocate_slot().unwrap(), 0);
    }

    #[test]
    fn test_field_pictures_rejected() {
        let mut sps = sps(0);
        sps.frame_mbs_only_flag = false;
        let mut dpb = DecodeDpb::new(4, Arc::default());
        let mut h = header(0, 0);
        h.field_pic_flag = true;
        assert!(
            dpb.begin_picture(NalType::IdrSlice, 3, &h, &sps, true)
                .is_err()
        );
    }

    /// A slot a handed-out frame is reading must not be decoded over, even once
    /// the reference rules are done with it.
    #[test]
    fn pinned_slots_are_not_reallocated() {
        let pins = Arc::new(SlotPins::default());
        let mut dpb = DecodeDpb::new(2, pins.clone());

        // Decode a non-reference picture and hand it to the caller.
        let handed_out = dpb.try_allocate_slot().unwrap();
        pins.pin(handed_out);
        dpb.release_slot(handed_out);
        assert!(dpb.has_pinned_slots());

        // The DPB has no use for the slot any more, but the caller is reading it.
        let next = dpb.try_allocate_slot().unwrap();
        assert_ne!(next, handed_out);
        assert_eq!(dpb.try_allocate_slot(), None, "both slots are busy");

        // Dropping the frame releases the pin.
        pins.release(handed_out);
        assert!(!dpb.has_pinned_slots());
        assert_eq!(dpb.try_allocate_slot(), Some(handed_out));
    }

    /// Waiting when nothing is pinned would never wake, so it must not wait.
    #[test]
    fn waiting_with_nothing_pinned_returns_immediately() {
        let pins = SlotPins::default();
        pins.wait_for_release();
    }

    /// Each session owns its pins, so a frame outliving a rebuild releases into
    /// the set it was decoded under and cannot free a slot the *current*
    /// session has reserved for someone else.
    #[test]
    fn pins_do_not_leak_across_sessions() {
        let old_pins = Arc::new(SlotPins::default());
        let mut old_dpb = DecodeDpb::new(2, old_pins.clone());
        let held = old_dpb.try_allocate_slot().unwrap();
        old_pins.pin(held);

        // A rebuild: new session, new pin set, and a frame from the old one is
        // still alive.
        let new_pins = Arc::new(SlotPins::default());
        let mut new_dpb = DecodeDpb::new(2, new_pins.clone());
        let reused = new_dpb.try_allocate_slot().unwrap();
        assert_eq!(reused, held, "the new session should reuse the slot number");
        new_pins.pin(reused);

        // The old frame is dropped, releasing into the old set.
        old_pins.release(held);

        assert!(
            new_pins.is_pinned(reused),
            "a stale frame must not release a slot the current session pinned"
        );
    }
}
