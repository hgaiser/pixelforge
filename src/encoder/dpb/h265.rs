//! H.265/HEVC-specific DPB implementation.
//!
//! This module implements the H.265 DPB according to the specification,
//! including:
//! - Reference Picture Set (RPS) management
//! - Short-term and long-term reference picture handling
//! - Reference picture list construction for P and B frames
//! - Temporal layer support
//! - CRA (Clean Random Access) handling

// Many loops in this module use index-based iteration because they access.
// multiple arrays with the same index or use the index for bit operations.
#![allow(clippy::needless_range_loop)]

use super::entry::{DpbEntry, DpbState, MarkingState};
use super::reference_lists::{H265ReferenceListBuilder, ReferenceList};
use super::types::{DpbConfig, PictureStartInfo, RefPicSet, ShortTermRefPicSet};
use super::{DecodedPictureBufferTrait, PictureType, MAX_DPB_SIZE, MAX_REF_LIST_SIZE};

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
    /// POC of CRA picture (for refresh pending).
    pic_order_cnt_cra: i32,
    /// Whether a refresh is pending.
    refresh_pending: bool,
    /// Long-term flags bitmask.
    long_term_flags: u32,
    /// Whether to use multiple references.
    use_multiple_refs: bool,
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
            pic_order_cnt_cra: 0,
            refresh_pending: false,
            long_term_flags: 0,
            use_multiple_refs: true,
            max_poc_lsb: 16,
            num_temporal_layers: 1,
        }
    }

    /// Get reference to the entries array.
    pub fn entries(&self) -> &[DpbEntry; MAX_DPB_SIZE] {
        &self.entries
    }

    /// Get the maximum DPB size.
    pub fn dpb_size(&self) -> i8 {
        self.max_dpb_size
    }

    /// Get the maximum POC LSB value.
    pub fn max_poc_lsb(&self) -> i32 {
        self.max_poc_lsb
    }

    /// Reference picture marking for H.265.
    ///
    /// Marks pictures as unused based on picture type and long-term flags.
    pub fn reference_picture_marking(
        &mut self,
        current_poc: i32,
        pic_type: PictureType,
        long_term_refs_present: bool,
    ) {
        if pic_type == PictureType::Idr {
            // IDR: mark all as unused.
            for entry in &mut self.entries[..self.max_dpb_size as usize] {
                entry.mark_unused();
            }
            return;
        }

        // Handle CRA refresh pending.
        if self.refresh_pending && current_poc > self.pic_order_cnt_cra {
            for entry in &mut self.entries[..self.max_dpb_size as usize] {
                if entry.pic_order_cnt != self.pic_order_cnt_cra {
                    entry.mark_unused();
                }
            }
            self.refresh_pending = false;
        }

        // CRA picture triggers refresh.
        if pic_type == PictureType::I {
            self.refresh_pending = true;
            self.pic_order_cnt_cra = current_poc;
        }

        if !self.use_multiple_refs {
            return;
        }

        // Balance short-term and long-term references.
        let mut num_short_term = 0;
        let mut num_long_term = 0;
        let mut num_corrupted = 0;
        let mut oldest_st_idx: Option<usize> = None;
        let mut oldest_st_poc = u32::MAX;
        let mut oldest_lt_idx: Option<usize> = None;
        let mut oldest_lt_poc = u32::MAX;
        let mut oldest_corrupted_idx: Option<usize> = None;
        let mut oldest_corrupted_poc = u32::MAX;

        for i in 0..self.max_dpb_size as usize {
            let entry = &self.entries[i];
            if entry.state != DpbState::InUse {
                continue;
            }

            if entry.marking == MarkingState::ShortTerm && !entry.corrupted {
                num_short_term += 1;
                if (entry.pic_order_cnt as u32) < oldest_st_poc {
                    oldest_st_poc = entry.pic_order_cnt as u32;
                    oldest_st_idx = Some(i);
                }
            }

            if entry.marking == MarkingState::LongTerm && !entry.corrupted {
                num_long_term += 1;
                if (entry.pic_order_cnt as u32) < oldest_lt_poc {
                    oldest_lt_poc = entry.pic_order_cnt as u32;
                    oldest_lt_idx = Some(i);
                }
            }

            if entry.corrupted {
                num_corrupted += 1;
                if (entry.pic_order_cnt as u32) < oldest_corrupted_poc {
                    oldest_corrupted_poc = entry.pic_order_cnt as u32;
                    oldest_corrupted_idx = Some(i);
                }
            }
        }

        let total_refs = num_short_term + num_long_term + num_corrupted;

        if !long_term_refs_present {
            if total_refs > (self.max_dpb_size as usize - 1) {
                // Remove oldest corrupted, or oldest short-term, or oldest long-term.
                if num_corrupted > 0 && oldest_corrupted_poc < oldest_st_poc {
                    if let Some(idx) = oldest_corrupted_idx {
                        self.entries[idx].mark_unused();
                    }
                } else if num_short_term > 0 {
                    if let Some(idx) = oldest_st_idx {
                        self.entries[idx].mark_unused();
                    }
                } else if num_long_term > 0 {
                    if let Some(idx) = oldest_lt_idx {
                        self.entries[idx].mark_unused();
                    }
                }
            }
        } else {
            // Balance LTR and STR (max 50% LTR)
            let max_allowed_ltr = total_refs / 2;

            if total_refs > (self.max_dpb_size as usize - 1) {
                if num_corrupted > 0 && oldest_corrupted_poc < oldest_st_poc {
                    if let Some(idx) = oldest_corrupted_idx {
                        self.entries[idx].mark_unused();
                    }
                } else if num_long_term > max_allowed_ltr {
                    if let Some(idx) = oldest_lt_idx {
                        self.entries[idx].mark_unused();
                    }
                } else if num_short_term > 0 {
                    if let Some(idx) = oldest_st_idx {
                        self.entries[idx].mark_unused();
                    }
                }
            }
        }
    }

    /// Apply the Reference Picture Set to mark pictures.
    ///
    /// This categorizes reference pictures into:
    /// - stCurrBefore: short-term refs with POC < current
    /// - stCurrAfter: short-term refs with POC > current
    /// - ltCurr: long-term refs used by current
    /// - stFoll: short-term refs not used by current
    /// - ltFoll: long-term refs not used by current
    pub fn apply_reference_picture_set(
        &mut self,
        current_poc: i32,
        pic_type: PictureType,
        short_term_rps: &ShortTermRefPicSet,
        num_long_term_pics: u32,
        long_term_poc_lsb: &[u32],
        used_by_curr_pic_lt: u32,
    ) -> RefPicSet {
        let mut ref_pic_set = RefPicSet::new();

        if pic_type == PictureType::Idr {
            // IDR has no references.
            self.num_poc_st_curr_before = 0;
            self.num_poc_st_curr_after = 0;
            self.num_poc_st_foll = 0;
            self.num_poc_lt_curr = 0;
            self.num_poc_lt_foll = 0;
            return ref_pic_set;
        }

        // Calculate delta POC values.
        let mut delta_poc_s0 = [0i32; MAX_DPB_SIZE];
        let mut delta_poc_s1 = [0i32; MAX_DPB_SIZE];

        // S0: negative delta POCs.
        for i in 0..short_term_rps.num_negative_pics as usize {
            delta_poc_s0[i] = if i == 0 {
                -(short_term_rps.delta_poc_s0_minus1[i] as i32 + 1)
            } else {
                delta_poc_s0[i - 1] - (short_term_rps.delta_poc_s0_minus1[i] as i32 + 1)
            };
        }

        // S1: positive delta POCs.
        for i in 0..short_term_rps.num_positive_pics as usize {
            delta_poc_s1[i] = if i == 0 {
                short_term_rps.delta_poc_s1_minus1[i] as i32 + 1
            } else {
                delta_poc_s1[i - 1] + short_term_rps.delta_poc_s1_minus1[i] as i32 + 1
            };
        }

        // Build POC lists.
        let mut poc_st_curr_before = [0i32; MAX_REF_LIST_SIZE];
        let mut poc_st_curr_after = [0i32; MAX_REF_LIST_SIZE];
        let mut poc_st_foll = [0i32; MAX_REF_LIST_SIZE];
        let mut poc_lt_curr = [0i32; MAX_REF_LIST_SIZE];

        let mut j = 0usize;
        let mut k = 0usize;

        // Process S0 (negative)
        for i in 0..short_term_rps.num_negative_pics as usize {
            let poc = current_poc + delta_poc_s0[i];
            if (short_term_rps.used_by_curr_pic_s0_flag >> i) & 1 != 0 {
                poc_st_curr_before[j] = poc;
                j += 1;
            } else {
                poc_st_foll[k] = poc;
                k += 1;
            }
        }
        self.num_poc_st_curr_before = j as i8;

        j = 0;
        // Process S1 (positive)
        for i in 0..short_term_rps.num_positive_pics as usize {
            let poc = current_poc + delta_poc_s1[i];
            if (short_term_rps.used_by_curr_pic_s1_flag >> i) & 1 != 0 {
                poc_st_curr_after[j] = poc;
                j += 1;
            } else {
                poc_st_foll[k] = poc;
                k += 1;
            }
        }
        self.num_poc_st_curr_after = j as i8;
        self.num_poc_st_foll = k as i8;

        // Process long-term references.
        j = 0;
        k = 0;
        for i in 0..num_long_term_pics as usize {
            let poc_lt = long_term_poc_lsb[i] as i32;
            if (used_by_curr_pic_lt >> i) & 1 != 0 {
                poc_lt_curr[j] = poc_lt;
                j += 1;
            } else {
                k += 1;
            }
        }
        self.num_poc_lt_curr = j as i8;
        self.num_poc_lt_foll = k as i8;

        // Map POCs to DPB indices.
        // stCurrBefore
        for i in 0..self.num_poc_st_curr_before as usize {
            ref_pic_set.st_curr_before[i] = -1;
            for d in 0..self.max_dpb_size as usize {
                let entry = &self.entries[d];
                if entry.state == DpbState::InUse
                    && entry.marking == MarkingState::ShortTerm
                    && entry.pic_order_cnt == poc_st_curr_before[i]
                {
                    ref_pic_set.st_curr_before[i] = d as i8;
                    break;
                }
            }
        }
        ref_pic_set.num_st_curr_before = self.num_poc_st_curr_before as u8;

        // stCurrAfter
        for i in 0..self.num_poc_st_curr_after as usize {
            ref_pic_set.st_curr_after[i] = -1;
            for d in 0..self.max_dpb_size as usize {
                let entry = &self.entries[d];
                if entry.state == DpbState::InUse
                    && entry.marking == MarkingState::ShortTerm
                    && entry.pic_order_cnt == poc_st_curr_after[i]
                {
                    ref_pic_set.st_curr_after[i] = d as i8;
                    break;
                }
            }
        }
        ref_pic_set.num_st_curr_after = self.num_poc_st_curr_after as u8;

        // ltCurr
        for i in 0..self.num_poc_lt_curr as usize {
            ref_pic_set.lt_curr[i] = -1;
            let mask = self.max_poc_lsb - 1;
            for d in 0..self.max_dpb_size as usize {
                let entry = &self.entries[d];
                if entry.state == DpbState::InUse
                    && entry.marking != MarkingState::Unused
                    && (entry.pic_order_cnt & mask) == poc_lt_curr[i]
                {
                    ref_pic_set.lt_curr[i] = d as i8;
                    // Mark as long-term.
                    break;
                }
            }
        }
        ref_pic_set.num_lt_curr = self.num_poc_lt_curr as u8;

        // Mark pictures not in RPS as unused.
        let mut in_use = [false; MAX_DPB_SIZE];
        for i in 0..self.num_poc_st_curr_before as usize {
            if ref_pic_set.st_curr_before[i] >= 0 {
                in_use[ref_pic_set.st_curr_before[i] as usize] = true;
            }
        }
        for i in 0..self.num_poc_st_curr_after as usize {
            if ref_pic_set.st_curr_after[i] >= 0 {
                in_use[ref_pic_set.st_curr_after[i] as usize] = true;
            }
        }
        for i in 0..self.num_poc_lt_curr as usize {
            if ref_pic_set.lt_curr[i] >= 0 {
                in_use[ref_pic_set.lt_curr[i] as usize] = true;
            }
        }
        // stFoll and ltFoll would also be marked as in_use

        for i in 0..self.max_dpb_size as usize {
            if !in_use[i] && self.entries[i].marking != MarkingState::Unused {
                self.entries[i].mark_unused();
            }
        }

        ref_pic_set
    }

    /// Initialize the short-term RPS for a P/B frame.
    ///
    /// This builds the short-term reference picture set based on current DPB state.
    pub fn initialize_short_term_rps(
        &self,
        current_poc: i32,
        pic_type: PictureType,
        current_temporal_id: i32,
        num_ref_l0: u32,
        num_ref_l1: u32,
    ) -> ShortTermRefPicSet {
        let mut rps = ShortTermRefPicSet::default();

        if pic_type == PictureType::Idr {
            return rps;
        }

        // Collect short-term references.
        let mut negative_refs: Vec<(i32, i32)> = Vec::new(); // (POC, delta_poc)
        let mut positive_refs: Vec<(i32, i32)> = Vec::new();

        for i in 0..self.max_dpb_size as usize {
            let entry = &self.entries[i];
            if entry.state != DpbState::InUse {
                continue;
            }
            if entry.marking != MarkingState::ShortTerm {
                continue;
            }
            if entry.corrupted {
                continue;
            }
            if entry.temporal_id > current_temporal_id {
                continue;
            }

            let delta_poc = entry.pic_order_cnt - current_poc;
            if delta_poc < 0 {
                negative_refs.push((entry.pic_order_cnt, delta_poc));
            } else if delta_poc > 0 && self.use_multiple_refs {
                positive_refs.push((entry.pic_order_cnt, delta_poc));
            }
        }

        // Sort negative refs by POC descending (closest to current first)
        negative_refs.sort_by_key(|r| std::cmp::Reverse(r.0));

        // Sort positive refs by POC ascending (closest to current first)
        positive_refs.sort_by_key(|r| r.0);

        // Limit to DPB size - 1.
        let max_refs = (self.max_dpb_size - 1) as usize;
        while negative_refs.len() + positive_refs.len() > max_refs {
            if !negative_refs.is_empty() {
                negative_refs.pop();
            } else if !positive_refs.is_empty() {
                positive_refs.pop();
            }
        }

        // Build the RPS structure.
        rps.num_negative_pics = negative_refs.len() as u8;
        rps.num_positive_pics = positive_refs.len() as u8;

        let mut prev_delta = 0i32;
        for (i, (_, delta_poc)) in negative_refs.iter().enumerate() {
            rps.delta_poc_s0_minus1[i] = (prev_delta - delta_poc - 1) as u8;
            // Mark as used if within num_ref_l0.
            if i < num_ref_l0 as usize {
                rps.used_by_curr_pic_s0_flag |= 1 << i;
            }
            prev_delta = *delta_poc;
        }

        prev_delta = 0;
        for (i, (_, delta_poc)) in positive_refs.iter().enumerate() {
            rps.delta_poc_s1_minus1[i] = (*delta_poc - prev_delta - 1) as u8;
            // Mark as used if within num_ref_l1.
            if i < num_ref_l1 as usize {
                rps.used_by_curr_pic_s1_flag |= 1 << i;
            }
            prev_delta = *delta_poc;
        }

        rps
    }

    /// Setup reference picture lists L0 and L1.
    pub fn setup_reference_lists(
        &self,
        pic_type: PictureType,
        ref_pic_set: &RefPicSet,
        num_ref_l0: u32,
        num_ref_l1: u32,
    ) -> (ReferenceList, ReferenceList) {
        H265ReferenceListBuilder::build_ref_lists(
            &self.entries,
            self.max_dpb_size,
            pic_type,
            &ref_pic_set.st_curr_before,
            &ref_pic_set.st_curr_after,
            &ref_pic_set.lt_curr,
            num_ref_l0,
            num_ref_l1,
            self.use_multiple_refs,
        )
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

    /// Get an entry by index.
    pub fn get_entry(&self, index: usize) -> Option<&DpbEntry> {
        if index < MAX_DPB_SIZE && self.entries[index].state == DpbState::InUse {
            Some(&self.entries[index])
        } else {
            None
        }
    }

    /// Get mutable entry by index.
    pub fn get_entry_mut(&mut self, index: usize) -> Option<&mut DpbEntry> {
        if index < MAX_DPB_SIZE {
            Some(&mut self.entries[index])
        } else {
            None
        }
    }

    /// Get the long-term flags.
    pub fn long_term_flags(&self) -> u32 {
        self.long_term_flags
    }
}

impl DecodedPictureBufferTrait for DpbH265 {
    fn sequence_start(&mut self, config: DpbConfig) {
        // Reset all entries.
        for entry in &mut self.entries {
            entry.reset();
        }

        self.max_dpb_size = std::cmp::min(config.dpb_size as i8, MAX_DPB_SIZE as i8);
        self.use_multiple_refs = config.use_multiple_references;
        self.max_poc_lsb = config.max_pic_order_cnt_lsb();
        self.num_temporal_layers = config.num_temporal_layers;
        self.current_slot = -1;
        self.num_poc_st_curr_before = 0;
        self.num_poc_st_curr_after = 0;
        self.num_poc_st_foll = 0;
        self.num_poc_lt_curr = 0;
        self.num_poc_lt_foll = 0;
        self.refresh_pending = false;
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
            for i in 0..self.max_dpb_size as usize {
                ref_pocs[i] = self.entries[i].pic_order_cnt;
                if self.entries[i].marking == MarkingState::LongTerm {
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

    fn current_slot(&self) -> i8 {
        self.current_slot
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

    #[test]
    fn test_h265_short_term_rps() {
        let mut dpb = DpbH265::new();
        dpb.sequence_start(DpbConfig {
            dpb_size: 4,
            use_multiple_references: true,
            ..Default::default()
        });

        // Add IDR at POC 0.
        dpb.picture_start(PictureStartInfo {
            frame_id: 0,
            pic_order_cnt: 0,
            pic_type: PictureType::Idr,
            is_reference: true,
            ..Default::default()
        });
        dpb.picture_end(true);

        // Initialize RPS for P-frame at POC 2.
        let rps = dpb.initialize_short_term_rps(2, PictureType::P, 0, 1, 0);
        assert_eq!(rps.num_negative_pics, 1);
        assert_eq!(rps.num_positive_pics, 0);
    }
}
