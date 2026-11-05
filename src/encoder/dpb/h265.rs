//! H.265/HEVC-specific DPB implementation.
//!
//! This module keeps track of the H.265 reference pictures: which DPB slot
//! holds each picture, its POC, and whether it is still a reference.
use super::entry::{DpbEntry, DpbState, MarkingState};
use super::types::{DpbConfig, PictureStartInfo};
use super::{DecodedPictureBufferTrait, MAX_DPB_SIZE, PictureType};

/// H.265-specific DPB implementation.
#[derive(Debug)]
pub struct DpbH265 {
    /// DPB entries.
    entries: [DpbEntry; MAX_DPB_SIZE],
    /// Current DPB slot index (-1 if none).
    current_slot: i8,
    /// Maximum DPB size.
    max_dpb_size: i8,
    /// Number of POC values in stCurrBefore.
    num_poc_st_curr_before: i8,
    /// Number of POC values in stCurrAfter.
    num_poc_st_curr_after: i8,
    /// Number of POC values in stFoll.
    num_poc_st_foll: i8,
    /// Number of POC values in ltCurr.
    num_poc_lt_curr: i8,
    /// Number of POC values in ltFoll.
    num_poc_lt_foll: i8,
    /// Last IDR timestamp.
    last_idr_timestamp: u64,
    /// Long-term flags bitmask.
    long_term_flags: u32,
    /// Maximum POC LSB value.
    max_poc_lsb: i32,
    /// Number of temporal layers.
    num_temporal_layers: u32,
}

impl Default for DpbH265 {
    fn default() -> Self {
        Self::new()
    }
}

impl DpbH265 {
    /// Create a new H.265 DPB.
    pub fn new() -> Self {
        Self {
            entries: std::array::from_fn(|_| DpbEntry::new()),
            current_slot: -1,
            max_dpb_size: 0,
            num_poc_st_curr_before: 0,
            num_poc_st_curr_after: 0,
            num_poc_st_foll: 0,
            num_poc_lt_curr: 0,
            num_poc_lt_foll: 0,
            last_idr_timestamp: 0,
            long_term_flags: 0,
            max_poc_lsb: 16,
            num_temporal_layers: 1,
        }
    }

    /// DPB bumping process.
    fn dpb_bumping(&mut self) {
        // Find picture with smallest POC that needs output.
        let mut min_poc = i32::MAX;
        let mut min_idx: Option<usize> = None;

        for i in 0..self.max_dpb_size as usize {
            let entry = &self.entries[i];
            if entry.state == DpbState::InUse && entry.output && entry.pic_order_cnt < min_poc {
                min_poc = entry.pic_order_cnt;
                min_idx = Some(i);
            }
        }

        if let Some(idx) = min_idx {
            self.entries[idx].output = false;
            if self.entries[idx].marking == MarkingState::Unused {
                self.entries[idx].state = DpbState::Empty;
            }
        }
    }
}

impl DecodedPictureBufferTrait for DpbH265 {
    fn sequence_start(&mut self, config: DpbConfig) {
        // Reset all entries.
        for entry in &mut self.entries {
            entry.reset();
        }

        self.max_dpb_size = std::cmp::min(config.dpb_size as i8, MAX_DPB_SIZE as i8);
        self.max_poc_lsb = config.max_pic_order_cnt_lsb();
        self.num_temporal_layers = config.num_temporal_layers;
        self.current_slot = -1;
        self.num_poc_st_curr_before = 0;
        self.num_poc_st_curr_after = 0;
        self.num_poc_st_foll = 0;
        self.num_poc_lt_curr = 0;
        self.num_poc_lt_foll = 0;
        self.long_term_flags = 0;
    }

    fn picture_start(&mut self, info: PictureStartInfo) -> i8 {
        let is_irap = info.pic_type == PictureType::Idr || info.pic_type == PictureType::I;
        let no_rasl_output_flag = info.pic_type == PictureType::Idr;

        if is_irap && no_rasl_output_flag {
            if info.no_output_of_prior_pics_flag {
                // Clear all DPB entries.
                for entry in &mut self.entries[..self.max_dpb_size as usize] {
                    entry.state = DpbState::Empty;
                    entry.mark_unused();
                }
            } else {
                // Flush DPB.
                self.flush();
            }
        } else {
            // Remove entries that are unused and not needed for output.
            for entry in &mut self.entries[..self.max_dpb_size as usize] {
                if entry.marking == MarkingState::Unused && !entry.output {
                    entry.state = DpbState::Empty;
                }
            }
            while self.is_full() {
                self.dpb_bumping();
            }
        }

        // Find empty slot.
        self.current_slot = -1;
        for i in 0..self.max_dpb_size as usize {
            if self.entries[i].state == DpbState::Empty {
                self.current_slot = i as i8;
                break;
            }
        }

        if self.current_slot < 0 {
            // Force bumping.
            self.dpb_bumping();
            for i in 0..self.max_dpb_size as usize {
                if self.entries[i].state == DpbState::Empty {
                    self.current_slot = i as i8;
                    break;
                }
            }
        }

        if self.current_slot >= 0 {
            // Collect reference POCs and long-term flags first.
            let mut ref_pocs = [0i32; MAX_DPB_SIZE];
            let mut long_term_mask = 0u32;
            for (i, entry) in self
                .entries
                .iter()
                .enumerate()
                .take(self.max_dpb_size as usize)
            {
                ref_pocs[i] = entry.pic_order_cnt;
                if entry.marking == MarkingState::LongTerm {
                    long_term_mask |= 1 << i;
                }
            }

            let entry = &mut self.entries[self.current_slot as usize];
            entry.state = DpbState::InUse;
            entry.frame_id = info.frame_id;
            entry.pic_order_cnt = info.pic_order_cnt;
            entry.output = info.pic_output_flag;
            entry.corrupted = false;
            entry.temporal_id = info.temporal_id as i32;
            entry.pic_type = info.pic_type;

            // Store reference POCs.
            entry.ref_pic_order_cnt = ref_pocs;
            entry.long_term_ref_pic = long_term_mask;

            if is_irap && no_rasl_output_flag {
                self.last_idr_timestamp = info.timestamp;
            }
        }

        self.current_slot
    }

    fn picture_end(&mut self, is_reference: bool) {
        if self.current_slot < 0 {
            return;
        }

        // For temporal SVC, unmark refs with same temporal ID.
        if self.num_temporal_layers > 1 {
            let current_temporal_id = self.entries[self.current_slot as usize].temporal_id;
            for i in 0..self.max_dpb_size as usize {
                let entry = &mut self.entries[i];
                if entry.state == DpbState::InUse
                    && entry.marking != MarkingState::Unused
                    && entry.temporal_id == current_temporal_id
                    && i != self.current_slot as usize
                {
                    entry.mark_unused();
                }
            }
        }

        let entry = &mut self.entries[self.current_slot as usize];
        entry.marking = if is_reference {
            MarkingState::ShortTerm
        } else {
            MarkingState::Unused
        };

        // Apply sliding window to remove oldest reference if we exceed max refs.
        if is_reference {
            let num_short_term = self.num_short_term_refs();
            let num_long_term = self.num_long_term_refs();
            let max_refs = (self.max_dpb_size - 1) as u32; // Reserve one slot for current

            if (num_short_term + num_long_term) > max_refs && num_short_term > 0 {
                // Find short-term ref with lowest POC (oldest in display order)
                let mut oldest_idx: Option<usize> = None;
                let mut lowest_poc = i32::MAX;

                for i in 0..self.max_dpb_size as usize {
                    let e = &self.entries[i];
                    if e.is_short_term_reference()
                        && i as i8 != self.current_slot
                        && e.pic_order_cnt < lowest_poc
                    {
                        lowest_poc = e.pic_order_cnt;
                        oldest_idx = Some(i);
                    }
                }

                if let Some(idx) = oldest_idx {
                    self.entries[idx].mark_unused();
                    if !self.entries[idx].output {
                        self.entries[idx].state = DpbState::Empty;
                    }
                }
            }
        }
    }

    fn is_full(&self) -> bool {
        let mut count = 0;
        for i in 0..self.max_dpb_size as usize {
            if self.entries[i].state == DpbState::InUse {
                count += 1;
            }
        }
        count >= self.max_dpb_size as usize
    }

    fn is_empty(&self) -> bool {
        for i in 0..self.max_dpb_size as usize {
            if self.entries[i].state == DpbState::InUse {
                return false;
            }
        }
        true
    }

    fn flush(&mut self) {
        // Mark all as unused.
        for entry in &mut self.entries[..self.max_dpb_size as usize] {
            entry.mark_unused();
        }
        // Empty slots not needed for output.
        for entry in &mut self.entries[..self.max_dpb_size as usize] {
            if entry.state == DpbState::InUse
                && !entry.output
                && entry.marking == MarkingState::Unused
            {
                entry.state = DpbState::Empty;
            }
        }
        // Bump until empty.
        while !self.is_empty() {
            self.dpb_bumping();
        }
    }

    fn num_short_term_refs(&self) -> u32 {
        let mut count = 0;
        for i in 0..self.max_dpb_size as usize {
            if self.entries[i].is_short_term_reference() {
                count += 1;
            }
        }
        count
    }

    fn num_long_term_refs(&self) -> u32 {
        let mut count = 0;
        for i in 0..self.max_dpb_size as usize {
            if self.entries[i].is_long_term_reference() {
                count += 1;
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h265_dpb_sequence_start() {
        let mut dpb = DpbH265::new();
        dpb.sequence_start(DpbConfig {
            dpb_size: 4,
            ..Default::default()
        });

        assert_eq!(dpb.max_dpb_size, 4);
        assert!(dpb.is_empty());
    }

    #[test]
    fn test_h265_dpb_picture_flow() {
        let mut dpb = DpbH265::new();
        dpb.sequence_start(DpbConfig {
            dpb_size: 4,
            ..Default::default()
        });

        // Add IDR.
        let slot = dpb.picture_start(PictureStartInfo {
            frame_id: 0,
            pic_order_cnt: 0,
            pic_type: PictureType::Idr,
            is_reference: true,
            ..Default::default()
        });
        assert_eq!(slot, 0);
        dpb.picture_end(true);

        assert!(!dpb.is_empty());
        assert_eq!(dpb.num_short_term_refs(), 1);

        // Add P frame.
        let slot = dpb.picture_start(PictureStartInfo {
            frame_id: 1,
            pic_order_cnt: 2,
            pic_type: PictureType::P,
            is_reference: true,
            ..Default::default()
        });
        assert_eq!(slot, 1);
        dpb.picture_end(true);

        assert_eq!(dpb.num_short_term_refs(), 2);
    }
}
