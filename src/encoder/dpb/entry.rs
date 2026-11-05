//! DPB entry structures for individual pictures in the buffer.

use super::{MAX_DPB_SIZE, PictureType};

/// Reference marking state for a picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MarkingState {
    /// Unused for reference.
    #[default]
    Unused,
    /// Used for short-term reference.
    ShortTerm,
    /// Used for long-term reference.
    LongTerm,
}

impl MarkingState {
    /// Returns true if this entry is used for reference.
    pub fn is_reference(&self) -> bool {
        !matches!(self, Self::Unused)
    }

    /// Returns true if this is a short-term reference.
    pub fn is_short_term(&self) -> bool {
        matches!(self, Self::ShortTerm)
    }

    /// Returns true if this is a long-term reference.
    pub fn is_long_term(&self) -> bool {
        matches!(self, Self::LongTerm)
    }
}

/// DPB entry state (empty or in use).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DpbState {
    /// Slot is empty.
    #[default]
    Empty,
    /// Slot contains a frame.
    InUse,
}

/// A single entry in the Decoded Picture Buffer.
#[derive(Debug, Clone)]
pub struct DpbEntry {
    /// State of this DPB slot.
    pub state: DpbState,
    /// Reference marking for top field / frame.
    pub marking: MarkingState,
    /// Reference marking for bottom field (H.264 field coding).
    pub bottom_field_marking: MarkingState,
    /// Whether this picture is needed for output.
    pub output: bool,
    /// Whether this frame is corrupted (for error resilience).
    pub corrupted: bool,
    /// Picture Order Count.
    pub pic_order_cnt: i32,
    /// Top field order count (H.264).
    pub top_foc: i32,
    /// Bottom field order count (H.264).
    pub bottom_foc: i32,
    /// Frame number (H.264 frame_num syntax element).
    pub frame_num: u32,
    /// Frame number wrapped for POC calculation.
    pub frame_num_wrap: i32,
    /// Top field picture number (H.264).
    pub top_pic_num: i32,
    /// Bottom field picture number (H.264).
    pub bottom_pic_num: i32,
    /// Long-term frame index.
    pub long_term_frame_idx: i32,
    /// Top field long-term picture number.
    pub top_long_term_pic_num: i32,
    /// Bottom field long-term picture number.
    pub bottom_long_term_pic_num: i32,
    /// POC values of reference pictures (for H.265 RPS).
    pub ref_pic_order_cnt: [i32; MAX_DPB_SIZE],
    /// Bitmask indicating which refs are long-term.
    pub long_term_ref_pic: u32,
    /// Internal unique frame ID.
    pub frame_id: u64,
    /// Temporal layer ID.
    pub temporal_id: i32,
    /// Timestamp of this picture.
    pub timestamp: u64,
    /// Picture type.
    pub pic_type: PictureType,
    /// Whether this entry represents a non-existing frame (gap handling).
    pub not_existing: bool,
}

impl Default for DpbEntry {
    fn default() -> Self {
        Self::new()
    }
}

impl DpbEntry {
    /// Create a new empty DPB entry.
    pub fn new() -> Self {
        Self {
            state: DpbState::Empty,
            marking: MarkingState::Unused,
            bottom_field_marking: MarkingState::Unused,
            output: false,
            corrupted: false,
            pic_order_cnt: 0,
            top_foc: 0,
            bottom_foc: 0,
            frame_num: 0,
            frame_num_wrap: 0,
            top_pic_num: 0,
            bottom_pic_num: 0,
            long_term_frame_idx: -1,
            top_long_term_pic_num: 0,
            bottom_long_term_pic_num: 0,
            ref_pic_order_cnt: [0; MAX_DPB_SIZE],
            long_term_ref_pic: 0,
            frame_id: 0,
            temporal_id: 0,
            timestamp: 0,
            pic_type: PictureType::I,
            not_existing: false,
        }
    }

    /// Reset this entry to empty state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Check if this entry is a reference picture.
    pub fn is_reference(&self) -> bool {
        self.marking.is_reference() || self.bottom_field_marking.is_reference()
    }

    /// Check if this entry is a short-term reference.
    pub fn is_short_term_reference(&self) -> bool {
        self.marking.is_short_term() || self.bottom_field_marking.is_short_term()
    }

    /// Check if this entry is a long-term reference.
    pub fn is_long_term_reference(&self) -> bool {
        self.marking.is_long_term() || self.bottom_field_marking.is_long_term()
    }

    /// Mark as unused for reference.
    pub fn mark_unused(&mut self) {
        self.marking = MarkingState::Unused;
        self.bottom_field_marking = MarkingState::Unused;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpb_entry_default() {
        let entry = DpbEntry::new();
        assert_eq!(entry.state, DpbState::Empty);
        assert_eq!(entry.marking, MarkingState::Unused);
        assert!(!entry.is_reference());
    }

    #[test]
    fn test_marking_state() {
        let mut entry = DpbEntry::new();

        entry.marking = MarkingState::ShortTerm;
        entry.bottom_field_marking = MarkingState::ShortTerm;
        assert!(entry.is_short_term_reference());
        assert!(entry.is_reference());
        assert!(!entry.is_long_term_reference());

        entry.marking = MarkingState::LongTerm;
        entry.bottom_field_marking = MarkingState::LongTerm;
        assert!(entry.is_long_term_reference());
        assert!(entry.is_reference());
        assert!(!entry.is_short_term_reference());

        entry.mark_unused();
        assert!(!entry.is_reference());
    }
}
