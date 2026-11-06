//! Decode-side H.265 picture order count and reference picture set handling.
//!
//! H.265 has no sliding window and no marking commands: every slice header
//! names the complete set of pictures to keep (the reference picture set), and
//! anything not named is dropped *before* the current picture is decoded. This
//! module derives the picture order count (clause 8.3.1), resolves the
//! reference picture set against the pictures held (clause 8.3.2), and keeps
//! the DPB slots the way the H.264 decoder's `DecodeDpb` does, so the session
//! code can drive both alike.
//!
//! Random access follows `NoRaslOutputFlag`: an IDR or BLA picture, the first
//! picture of the stream and the first after an end of sequence all start a
//! new coded video sequence, and the RASL pictures that lead such a CRA are
//! skipped, since they reference pictures the decoder never saw.
//! `HandleCraAsBlaFlag` is always 0, and unavailable reference pictures
//! (clause 8.3.3) are never generated: the only pictures that would need them
//! are those RASL pictures, and they are not decoded.

use std::sync::Arc;

use super::parser::{NalHeader, NalType, invalid};
use super::slice::SliceHeader;
use super::sps::Sps;
use crate::decoder::frames::SlotPins;
use crate::error::Result;

/// DPB slots this module can address: `MaxDpbSize` (16) pictures, which
/// includes the current one, plus one so a picture held for output never
/// blocks the next decode.
pub(crate) const MAX_DPB_SLOTS: usize = 17;

/// Entries in each of Vulkan's `RefPicSet*` arrays
/// (`STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE`).
pub(crate) const VULKAN_REF_PIC_SET_LIST_SIZE: usize = 8;

/// The value Vulkan's `RefPicSet*` arrays use for "no reference picture"
/// (`STD_VIDEO_H265_NO_REFERENCE_PICTURE`).
pub(crate) const NO_REFERENCE_PICTURE: u8 = 0xFF;

/// A picture currently held in the DPB as a reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RefPicture {
    /// DPB slot index this picture lives in.
    pub slot: u8,
    /// `PicOrderCntVal`.
    pub poc: i32,
    /// Marked "used for long-term reference"; otherwise short-term.
    pub long_term: bool,
}

/// One entry of a reference picture set list: the POC the slice header asks
/// for, and the slot of the picture that has it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RpsEntry {
    pub poc: i32,
    /// `None` is the standard's "no reference picture".
    pub slot: Option<u8>,
}

/// The five lists of clause 8.3.2, resolved to DPB slots.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RefPicSet {
    /// `RefPicSetStCurrBefore`: short-term, used by the current picture,
    /// preceding it in output order.
    pub st_curr_before: Vec<RpsEntry>,
    /// `RefPicSetStCurrAfter`: short-term, used, following in output order.
    pub st_curr_after: Vec<RpsEntry>,
    /// `RefPicSetStFoll`: short-term, kept for later pictures only.
    pub st_foll: Vec<RpsEntry>,
    /// `RefPicSetLtCurr`: long-term, used by the current picture.
    pub lt_curr: Vec<RpsEntry>,
    /// `RefPicSetLtFoll`: long-term, kept for later pictures only.
    pub lt_foll: Vec<RpsEntry>,
}

impl RefPicSet {
    /// Whether a picture the current one predicts from is not in the DPB.
    ///
    /// Missing entries in the two `Foll` lists do not matter to the current
    /// picture and are not counted (the standard ignores them too).
    pub fn missing_current(&self) -> bool {
        self.st_curr_before
            .iter()
            .chain(&self.st_curr_after)
            .chain(&self.lt_curr)
            .any(|e| e.slot.is_none())
    }

    /// A list in the form Vulkan's `RefPicSetStCurrBefore`, `RefPicSetStCurrAfter`
    /// and `RefPicSetLtCurr` take: slot indices, padded with
    /// [`NO_REFERENCE_PICTURE`].
    pub fn vulkan_list(list: &[RpsEntry]) -> Result<[u8; VULKAN_REF_PIC_SET_LIST_SIZE]> {
        if list.len() > VULKAN_REF_PIC_SET_LIST_SIZE {
            return Err(invalid(
                "more than 8 pictures in one reference picture set list",
            ));
        }
        let mut out = [NO_REFERENCE_PICTURE; VULKAN_REF_PIC_SET_LIST_SIZE];
        for (dst, entry) in out.iter_mut().zip(list) {
            *dst = entry.slot.unwrap_or(NO_REFERENCE_PICTURE);
        }
        Ok(out)
    }
}

/// Why a picture is not to be decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SkipReason {
    /// No IRAP picture has started a coded video sequence yet (the stream
    /// starts mid-sequence, or after an end of sequence). The caller should
    /// ask for a keyframe.
    AwaitingIrap,
    /// A RASL picture led by a CRA that started decoding: it references
    /// pictures from before the CRA, which the decoder never had. Skipping it
    /// is normal and not an error.
    RaslWithoutReferences,
}

/// Per-picture state derived from the bitstream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PictureState {
    /// `PicOrderCntVal`.
    pub poc: i32,
    pub nal_type: NalType,
    pub temporal_id: u8,
    /// `NoRaslOutputFlag`. Meaningful for IRAP pictures only.
    pub no_rasl_output_flag: bool,
    /// An IRAP picture with `NoRaslOutputFlag`: decoding can start here, and
    /// nothing after it in decoding order references anything before it.
    pub is_keyframe: bool,
    /// `PicOutputFlag`: whether the picture is to be output at all.
    pub pic_output_flag: bool,
    /// `NoOutputOfPriorPicsFlag` (clause C.5.2.2), set only on a keyframe.
    ///
    /// On a keyframe every picture still waiting for output is either output
    /// first (false) or discarded (true). It is inferred to be true for a CRA,
    /// so a caller that flushes its output at an end of sequence NAL unit,
    /// before the CRA that follows it arrives, loses nothing.
    pub no_output_of_prior_pics: bool,
    /// Whether the decoded picture is kept as a reference afterwards. False
    /// only for a sub-layer non-reference picture of the highest sub-layer,
    /// which nothing can reference.
    pub is_reference: bool,
    /// The reference picture set, resolved to slots. Empty for IDR pictures.
    pub rps: RefPicSet,
    /// Whether a picture the current one predicts from is missing (see
    /// [`RefPicSet::missing_current`]). Decoding it anyway produces corrupt
    /// output; the caller would normally skip it and ask for a keyframe.
    pub missing_reference: bool,
}

/// What [`DecodeDpb::begin_picture`] decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PictureDecision {
    Decode(PictureState),
    Skip(SkipReason),
}

/// Decode-side DPB: POC state, the reference pictures held and their slots.
pub(crate) struct DecodeDpb {
    /// Slots reserved by decoded frames the caller still holds. Never handed
    /// out for a new picture, however the reference rules mark them.
    pins: Arc<SlotPins>,
    /// Reference pictures, in decoding order.
    refs: Vec<RefPicture>,
    /// Which slots are occupied (by a reference or the in-flight picture).
    slot_used: [bool; MAX_DPB_SLOTS],
    /// Number of slots the session was created with.
    slot_count: usize,
    /// The next picture is the first of the stream or follows an end of
    /// sequence, so only an IRAP can start decoding.
    at_sequence_start: bool,
    /// `NoRaslOutputFlag` of the last IRAP, which decides whether the RASL
    /// pictures associated with it are decodable.
    irap_no_rasl_output: bool,
    /// `PicOrderCntVal` of `prevTid0Pic` (clause 8.3.1).
    prev_tid0_poc: i32,
}

impl DecodeDpb {
    pub fn new(slot_count: usize, pins: Arc<SlotPins>) -> Self {
        Self {
            pins,
            refs: Vec::new(),
            slot_used: [false; MAX_DPB_SLOTS],
            slot_count: slot_count.min(MAX_DPB_SLOTS),
            at_sequence_start: true,
            irap_no_rasl_output: true,
            prev_tid0_poc: 0,
        }
    }

    /// Current reference pictures, in the order they were decoded.
    ///
    /// After [`Self::begin_picture`] these are exactly the pictures in the
    /// current reference picture set, `Foll` lists included; all of them must
    /// stay active in the Vulkan session for later pictures.
    pub fn references(&self) -> &[RefPicture] {
        &self.refs
    }

    /// Note an end of sequence NAL unit: the next picture must be an IRAP, and
    /// starts a new coded video sequence with `NoRaslOutputFlag` = 1.
    ///
    /// Also the way to resynchronise after an error: the next CRA is then
    /// treated as a random access point and its RASL pictures are skipped.
    pub fn end_of_sequence(&mut self) {
        self.at_sequence_start = true;
    }

    /// Derive POC and the reference picture set for the picture about to be
    /// decoded, and drop every reference the set does not name.
    ///
    /// `slice` is the header of the picture's first slice segment. Must be
    /// called once per picture, before [`Self::try_allocate_slot`]; a skipped
    /// picture changes no state, and needs no slot.
    pub fn begin_picture(
        &mut self,
        nal: &NalHeader,
        slice: &SliceHeader,
        sps: &Sps,
    ) -> Result<PictureDecision> {
        let nal_type = nal.nal_type;
        if !nal_type.is_vcl() || slice.dependent_slice_segment_flag {
            return Err(invalid("begin_picture needs an independent slice segment"));
        }

        let irap = nal_type.is_irap();
        if !irap && self.at_sequence_start {
            return Ok(PictureDecision::Skip(SkipReason::AwaitingIrap));
        }
        // HandleCraAsBlaFlag is 0, so a CRA starts a new sequence only where
        // it is the first picture (clause 8.1.3).
        let no_rasl_output =
            irap && (nal_type.is_idr() || nal_type.is_bla() || self.at_sequence_start);
        if irap {
            self.irap_no_rasl_output = no_rasl_output;
        }
        if nal_type.is_rasl() && self.irap_no_rasl_output {
            return Ok(PictureDecision::Skip(SkipReason::RaslWithoutReferences));
        }
        self.at_sequence_start = false;

        let poc = self.derive_poc(nal, slice, sps, irap && no_rasl_output);

        if irap && no_rasl_output {
            // A new coded video sequence: nothing before it is referenced.
            self.drop_refs(|_| true);
        }
        let rps = if nal_type.is_idr() {
            RefPicSet::default()
        } else {
            self.apply_rps(poc, slice, sps)?
        };
        let missing_reference = rps.missing_current();

        let keyframe = irap && no_rasl_output;
        Ok(PictureDecision::Decode(PictureState {
            poc,
            nal_type,
            temporal_id: nal.temporal_id,
            no_rasl_output_flag: no_rasl_output,
            is_keyframe: keyframe,
            pic_output_flag: slice.pic_output_flag,
            no_output_of_prior_pics: keyframe
                && (nal_type == NalType::Cra || slice.no_output_of_prior_pics_flag),
            is_reference: !(nal_type.is_sub_layer_non_reference()
                && nal.temporal_id as usize >= sps.highest_tid()),
            rps,
            missing_reference,
        }))
    }

    /// Clause 8.3.1. Updates `prevTid0Pic` when the current picture qualifies.
    fn derive_poc(
        &mut self,
        nal: &NalHeader,
        slice: &SliceHeader,
        sps: &Sps,
        new_sequence: bool,
    ) -> i32 {
        let max_lsb = sps.max_pic_order_cnt_lsb() as i32;
        let lsb = slice.slice_pic_order_cnt_lsb as i32;
        let msb = if new_sequence {
            0
        } else {
            let prev_lsb = self.prev_tid0_poc & (max_lsb - 1);
            let prev_msb = self.prev_tid0_poc - prev_lsb;
            if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
                prev_msb + max_lsb
            } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
                prev_msb - max_lsb
            } else {
                prev_msb
            }
        };
        let poc = msb + lsb;

        // prevTid0Pic: TemporalId 0 and not a RASL, RADL or sub-layer
        // non-reference picture.
        let t = nal.nal_type;
        if nal.temporal_id == 0 && !t.is_rasl() && !t.is_radl() && !t.is_sub_layer_non_reference() {
            self.prev_tid0_poc = poc;
        }
        poc
    }

    /// Clause 8.3.2 for a non-IDR picture: build the five lists, mark the
    /// long-term pictures, and drop every reference left out.
    fn apply_rps(&mut self, poc: i32, slice: &SliceHeader, sps: &Sps) -> Result<RefPicSet> {
        let st = &slice
            .short_term_ref_pic_set(sps)
            .ok_or_else(|| invalid("non-IDR picture without a short-term reference picture set"))?
            .rps;
        let max_lsb = sps.max_pic_order_cnt_lsb() as i32;
        let mut keep = vec![false; self.refs.len()];
        let mut set = RefPicSet::default();

        // Long-term pictures first: they may be found among the short-term
        // ones, and are then no longer candidates for the short-term lists.
        for lt in &slice.long_term_refs {
            let mut poc_lt = lt.poc_lsb_lt as i32;
            if lt.delta_poc_msb_present_flag {
                poc_lt += poc
                    - lt.delta_poc_msb_cycle_lt_accumulated as i32 * max_lsb
                    - (poc & (max_lsb - 1));
            }
            let found = self.refs.iter().position(|r| {
                if lt.delta_poc_msb_present_flag {
                    r.poc == poc_lt
                } else {
                    r.poc & (max_lsb - 1) == poc_lt
                }
            });
            let slot = found.map(|i| {
                keep[i] = true;
                self.refs[i].long_term = true;
                self.refs[i].slot
            });
            let entry = RpsEntry { poc: poc_lt, slot };
            if lt.used_by_curr_pic_lt {
                set.lt_curr.push(entry);
            } else {
                set.lt_foll.push(entry);
            }
        }

        let short_term = |target: i32, keep: &mut [bool]| {
            let found = self
                .refs
                .iter()
                .position(|r| !r.long_term && r.poc == target);
            let slot = found.map(|i| {
                keep[i] = true;
                self.refs[i].slot
            });
            RpsEntry { poc: target, slot }
        };
        for (delta, used) in st.delta_poc_s0.iter().zip(&st.used_s0) {
            let entry = short_term(poc + delta, &mut keep);
            if *used {
                set.st_curr_before.push(entry);
            } else {
                set.st_foll.push(entry);
            }
        }
        for (delta, used) in st.delta_poc_s1.iter().zip(&st.used_s1) {
            let entry = short_term(poc + delta, &mut keep);
            if *used {
                set.st_curr_after.push(entry);
            } else {
                set.st_foll.push(entry);
            }
        }

        let mut index = 0;
        self.drop_refs(|_| {
            let drop_it = !keep[index];
            index += 1;
            drop_it
        });
        Ok(set)
    }

    /// Drop every reference matching `pred`, freeing its slot.
    fn drop_refs(&mut self, mut pred: impl FnMut(&RefPicture) -> bool) {
        let slot_used = &mut self.slot_used;
        self.refs.retain(|r| {
            let drop_it = pred(r);
            if drop_it {
                slot_used[r.slot as usize] = false;
            }
            !drop_it
        });
    }

    /// Reserve a DPB slot for the current picture, if one is available.
    ///
    /// Slots pinned by a handed-out frame are skipped: the caller is still
    /// reading them. `None` means every slot is either a reference or pinned.
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

    /// Release a slot allocated for a picture that will not be kept.
    pub fn release_slot(&mut self, slot: u8) {
        self.slot_used[slot as usize] = false;
    }

    /// Record the just-decoded picture. It becomes a short-term reference
    /// (clause 8.3.2, last paragraph) unless nothing can ever reference it, in
    /// which case its slot is released.
    pub fn end_picture(&mut self, slot: u8, state: &PictureState) {
        if !state.is_reference {
            self.release_slot(slot);
            return;
        }
        self.refs.push(RefPicture {
            slot,
            poc: state.poc,
            long_term: false,
        });
    }
}

#[cfg(test)]
#[path = "dpb_tests.rs"]
mod tests;
