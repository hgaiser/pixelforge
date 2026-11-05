//! Reference frame state (AV1 specification sections 7.8, 7.20 and 7.21).
//!
//! AV1 keeps eight reference slots. Each slot remembers, besides the picture
//! itself, the frame's size, order hints and the coding-tool state that later
//! headers inherit from it (loop filter deltas, segmentation features, global
//! motion and film grain). The header parser reads this table; the reference
//! frame update process writes it once a frame is decoded.

use super::consts::*;
use super::frame_header::{FrameHeader, FrameType};
use super::params::FilmGrainParams;
use super::sequence::SequenceHeader;

/// Everything one reference slot remembers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefSlot {
    /// `RefValid`.
    pub valid: bool,
    /// `RefFrameId`.
    pub frame_id: u32,
    /// `RefFrameType`.
    pub frame_type: FrameType,
    /// `RefShowableFrame`: whether `show_existing_frame` may show it.
    pub showable_frame: bool,
    pub upscaled_width: u32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub mi_cols: u32,
    pub mi_rows: u32,
    pub bit_depth: u8,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    /// `RefOrderHint`.
    pub order_hint: u8,
    /// `SavedOrderHints`, indexed by reference frame name.
    pub saved_order_hints: [u8; TOTAL_REFS_PER_FRAME],
    /// The frame's `RefFrameSignBias` as a bitmask indexed by reference
    /// frame name, as Vulkan's reference info takes it.
    pub ref_frame_sign_bias: u8,
    /// Kept for Vulkan's reference info.
    pub disable_frame_end_update_cdf: bool,
    /// Kept for Vulkan's reference info.
    pub segmentation_enabled: bool,
    /// `SavedLoopFilterRefDeltas`.
    pub loop_filter_ref_deltas: [i8; TOTAL_REFS_PER_FRAME],
    /// `SavedLoopFilterModeDeltas`.
    pub loop_filter_mode_deltas: [i8; 2],
    /// `SavedFeatureEnabled`.
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    /// `SavedFeatureData`.
    pub feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
    /// `SavedGmParams`.
    pub gm_params: [[i32; 6]; TOTAL_REFS_PER_FRAME],
    /// The saved film grain parameters (`load_grain_params()`).
    pub film_grain: FilmGrainParams,
    /// The Vulkan DPB slot holding this frame's picture, which the parser
    /// never looks at: the decoder fills it in through
    /// [`Av1Parser::commit_frame`](super::Av1Parser::commit_frame).
    pub dpb_slot: Option<u8>,
}

impl Default for RefSlot {
    fn default() -> Self {
        Self {
            valid: false,
            frame_id: 0,
            frame_type: FrameType::Key,
            showable_frame: false,
            upscaled_width: 0,
            frame_width: 0,
            frame_height: 0,
            render_width: 0,
            render_height: 0,
            mi_cols: 0,
            mi_rows: 0,
            bit_depth: 0,
            subsampling_x: false,
            subsampling_y: false,
            order_hint: 0,
            saved_order_hints: [0; TOTAL_REFS_PER_FRAME],
            ref_frame_sign_bias: 0,
            disable_frame_end_update_cdf: false,
            segmentation_enabled: false,
            loop_filter_ref_deltas: DEFAULT_LOOP_FILTER_REF_DELTAS,
            loop_filter_mode_deltas: [0; 2],
            feature_enabled: [[false; SEG_LVL_MAX]; MAX_SEGMENTS],
            feature_data: [[0; SEG_LVL_MAX]; MAX_SEGMENTS],
            gm_params: [DEFAULT_GM_PARAMS; TOTAL_REFS_PER_FRAME],
            film_grain: FilmGrainParams::default(),
            dpb_slot: None,
        }
    }
}

/// The eight reference slots.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RefState {
    pub slots: [RefSlot; NUM_REF_FRAMES],
}

impl RefState {
    /// The reference frame update process (7.20): store the decoded frame
    /// in every slot `refresh_frame_flags` selects, with `dpb_slot` as the
    /// picture's location.
    ///
    /// For a key frame shown with `show_existing_frame`, the loading process
    /// (7.21) makes the frame the current one first, so every slot becomes a
    /// copy of the shown slot, DPB location included.
    pub fn update(&mut self, seq: &SequenceHeader, header: &FrameHeader, dpb_slot: Option<u8>) {
        if header.show_existing_frame {
            if header.frame_type == FrameType::Key {
                let mut shown = self.slots[header.frame_to_show_map_idx as usize].clone();
                // A key frame may be shown this way at most once.
                shown.showable_frame = false;
                for i in 0..NUM_REF_FRAMES {
                    if (header.refresh_frame_flags >> i) & 1 != 0 {
                        self.slots[i] = shown.clone();
                    }
                }
            }
            return;
        }

        let size = &header.frame_size;
        let slot = RefSlot {
            valid: true,
            frame_id: header.current_frame_id,
            frame_type: header.frame_type,
            showable_frame: header.showable_frame,
            upscaled_width: size.upscaled_width,
            frame_width: size.frame_width,
            frame_height: size.frame_height,
            render_width: size.render_width,
            render_height: size.render_height,
            mi_cols: size.mi_cols,
            mi_rows: size.mi_rows,
            bit_depth: seq.color_config.bit_depth,
            subsampling_x: seq.color_config.subsampling_x,
            subsampling_y: seq.color_config.subsampling_y,
            order_hint: header.order_hint,
            saved_order_hints: header.order_hints,
            ref_frame_sign_bias: header.ref_frame_sign_bias_mask(),
            disable_frame_end_update_cdf: header.disable_frame_end_update_cdf,
            segmentation_enabled: header.segmentation.segmentation_enabled,
            loop_filter_ref_deltas: header.loop_filter.loop_filter_ref_deltas,
            loop_filter_mode_deltas: header.loop_filter.loop_filter_mode_deltas,
            feature_enabled: header.segmentation.feature_enabled,
            feature_data: header.segmentation.feature_data,
            gm_params: header.global_motion.gm_params,
            film_grain: header.film_grain.clone(),
            dpb_slot,
        };
        for i in 0..NUM_REF_FRAMES {
            if (header.refresh_frame_flags >> i) & 1 != 0 {
                self.slots[i] = slot.clone();
            }
        }
    }

    /// Whether any slot still refers to DPB slot `dpb_slot`, which tells the
    /// decoder when a picture can be reused.
    pub fn dpb_slot_in_use(&self, dpb_slot: u8) -> bool {
        self.slots.iter().any(|s| s.dpb_slot == Some(dpb_slot))
    }
}

/// `get_relative_dist(a, b)` (7.12.3).
pub fn relative_dist(seq: &SequenceHeader, a: u8, b: u8) -> i32 {
    if !seq.enable_order_hint || seq.order_hint_bits == 0 {
        return 0;
    }
    let diff = a as i32 - b as i32;
    let m = 1 << (seq.order_hint_bits - 1);
    (diff & (m - 1)) - (diff & m)
}

/// The set_frame_refs process (7.8): derive all seven `ref_frame_idx` from
/// the two coded with `frame_refs_short_signaling`.
pub fn set_frame_refs(
    seq: &SequenceHeader,
    refs: &RefState,
    order_hint: u8,
    last_frame_idx: u8,
    gold_frame_idx: u8,
) -> [u8; REFS_PER_FRAME] {
    let mut ref_frame_idx = [-1i32; REFS_PER_FRAME];
    ref_frame_idx[0] = last_frame_idx as i32;
    ref_frame_idx[GOLDEN_FRAME - LAST_FRAME] = gold_frame_idx as i32;
    let mut used = [false; NUM_REF_FRAMES];
    used[last_frame_idx as usize] = true;
    used[gold_frame_idx as usize] = true;

    let cur_frame_hint = 1i32 << (seq.order_hint_bits.max(1) - 1);
    let shifted: [i32; NUM_REF_FRAMES] = std::array::from_fn(|i| {
        cur_frame_hint + relative_dist(seq, refs.slots[i].order_hint, order_hint)
    });

    // Latest or earliest unused backward (future) reference, or latest
    // forward (past) one.
    let find = |used: &[bool; NUM_REF_FRAMES], backward: bool, latest: bool| -> Option<usize> {
        let mut best: Option<(usize, i32)> = None;
        for (i, &hint) in shifted.iter().enumerate() {
            let side = if backward {
                hint >= cur_frame_hint
            } else {
                hint < cur_frame_hint
            };
            if used[i] || !side {
                continue;
            }
            let better = match best {
                None => true,
                Some((_, b)) if latest => hint >= b,
                Some((_, b)) => hint < b,
            };
            if better {
                best = Some((i, hint));
            }
        }
        best.map(|(i, _)| i)
    };

    if let Some(r) = find(&used, true, true) {
        ref_frame_idx[ALTREF_FRAME - LAST_FRAME] = r as i32;
        used[r] = true;
    }
    if let Some(r) = find(&used, true, false) {
        ref_frame_idx[BWDREF_FRAME - LAST_FRAME] = r as i32;
        used[r] = true;
    }
    if let Some(r) = find(&used, true, false) {
        ref_frame_idx[ALTREF2_FRAME - LAST_FRAME] = r as i32;
        used[r] = true;
    }
    const REF_FRAME_LIST: [usize; REFS_PER_FRAME - 2] = [
        LAST2_FRAME,
        LAST3_FRAME,
        BWDREF_FRAME,
        ALTREF2_FRAME,
        ALTREF_FRAME,
    ];
    for ref_frame in REF_FRAME_LIST {
        if ref_frame_idx[ref_frame - LAST_FRAME] < 0
            && let Some(r) = find(&used, false, true)
        {
            ref_frame_idx[ref_frame - LAST_FRAME] = r as i32;
            used[r] = true;
        }
    }

    // Whatever is left gets the earliest frame of all.
    let mut earliest: Option<(usize, i32)> = None;
    for (i, &hint) in shifted.iter().enumerate() {
        if earliest.is_none_or(|(_, e)| hint < e) {
            earliest = Some((i, hint));
        }
    }
    let fallback = earliest.map_or(0, |(i, _)| i as i32);
    ref_frame_idx.map(|idx| if idx < 0 { fallback } else { idx } as u8)
}

/// The frame id check of `mark_ref_frames(idLen)` (5.9.4): invalidate the
/// slots whose frame id cannot precede `current_frame_id`.
pub fn mark_ref_frames(seq: &SequenceHeader, refs: &mut RefState, current_frame_id: u32) {
    let id_len = seq.frame_id_length();
    let diff_len = seq.delta_frame_id_length_minus_2 as u32 + 2;
    for slot in refs.slots.iter_mut() {
        let id = slot.frame_id;
        let stale = if current_frame_id > (1 << diff_len) {
            id > current_frame_id || id < current_frame_id - (1 << diff_len)
        } else {
            id > current_frame_id && id < (1 << id_len) + current_frame_id - (1 << diff_len)
        };
        if stale {
            slot.valid = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq_with_order_hints(bits: u8) -> SequenceHeader {
        SequenceHeader {
            enable_order_hint: true,
            order_hint_bits: bits,
            ..Default::default()
        }
    }

    #[test]
    fn relative_dist_wraps() {
        let seq = seq_with_order_hints(7);
        assert_eq!(relative_dist(&seq, 5, 3), 2);
        assert_eq!(relative_dist(&seq, 3, 5), -2);
        // 127 is one before 0 with 7-bit hints.
        assert_eq!(relative_dist(&seq, 127, 0), -1);
        assert_eq!(relative_dist(&seq, 0, 127), 1);
        assert_eq!(relative_dist(&SequenceHeader::default(), 9, 1), 0);
    }

    #[test]
    fn set_frame_refs_orders_past_and_future() {
        // Current frame 10; slots hold 2, 4, 6, 8 (past) and 12, 14, 16 and
        // 9 (one more past frame).
        let seq = seq_with_order_hints(7);
        let mut refs = RefState::default();
        for (slot, hint) in [2u8, 4, 6, 8, 12, 14, 16, 9].iter().enumerate() {
            refs.slots[slot].order_hint = *hint;
        }
        // LAST is slot 3 (8), GOLDEN slot 0 (2).
        let idx = set_frame_refs(&seq, &refs, 10, 3, 0);
        assert_eq!(idx[0], 3, "LAST as coded");
        assert_eq!(idx[GOLDEN_FRAME - LAST_FRAME], 0, "GOLDEN as coded");
        assert_eq!(
            idx[ALTREF_FRAME - LAST_FRAME],
            6,
            "ALTREF: latest future (16)"
        );
        assert_eq!(
            idx[BWDREF_FRAME - LAST_FRAME],
            4,
            "BWDREF: earliest future (12)"
        );
        assert_eq!(
            idx[ALTREF2_FRAME - LAST_FRAME],
            5,
            "ALTREF2: next future (14)"
        );
        assert_eq!(
            idx[LAST2_FRAME - LAST_FRAME],
            7,
            "LAST2: latest past left (9)"
        );
        assert_eq!(idx[LAST3_FRAME - LAST_FRAME], 2, "LAST3: next past (6)");
    }

    #[test]
    fn set_frame_refs_fills_gaps_with_the_earliest_frame() {
        // Every slot in the past and only three distinct ones: the rest of
        // the list falls back to the earliest frame.
        let seq = seq_with_order_hints(7);
        let mut refs = RefState::default();
        for (slot, hint) in [5u8, 3, 1, 1, 1, 1, 1, 1].iter().enumerate() {
            refs.slots[slot].order_hint = *hint;
        }
        let idx = set_frame_refs(&seq, &refs, 6, 0, 1);
        assert_eq!(idx[0], 0);
        assert_eq!(idx[GOLDEN_FRAME - LAST_FRAME], 1);
        // No future frames: BWDREF/ALTREF2/ALTREF take the remaining past
        // frames in latest-first order, then the earliest frame (slot 2).
        assert!(idx.iter().all(|&i| i < 8));
        assert_eq!(idx[LAST2_FRAME - LAST_FRAME], 7);
    }

    #[test]
    fn mark_ref_frames_drops_ids_from_the_future() {
        let seq = SequenceHeader {
            frame_id_numbers_present_flag: true,
            delta_frame_id_length_minus_2: 2,      // diffLen 4
            additional_frame_id_length_minus_1: 3, // idLen 8
            ..Default::default()
        };
        let mut refs = RefState::default();
        for (i, slot) in refs.slots.iter_mut().enumerate() {
            slot.valid = true;
            slot.frame_id = [30, 20, 14, 31, 40, 25, 15, 29][i];
        }
        // Current id 30: valid ids are 14..=30.
        mark_ref_frames(&seq, &mut refs, 30);
        let valid: Vec<bool> = refs.slots.iter().map(|s| s.valid).collect();
        assert_eq!(valid, [true, true, true, false, false, true, true, true]);
    }
}
