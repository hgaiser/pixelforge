//! Reference picture list construction and management.

// Loops in this module use index-based iteration because they need the DPB slot index.
#![allow(clippy::needless_range_loop)]

use super::entry::{DpbEntry, DpbState, MarkingState};
use super::{PictureType, MAX_DPB_SIZE, MAX_REF_LIST_SIZE};

/// A reference picture for building reference lists.
#[derive(Debug, Clone, Copy)]
pub struct ReferencePicture {
    /// DPB slot index.
    pub dpb_index: i8,
    /// Picture Order Count.
    pub poc: i32,
    /// Frame number.
    pub frame_num: u32,
    /// Whether this is a long-term reference.
    pub is_long_term: bool,
    /// Long-term picture number (if long-term).
    pub long_term_pic_num: i32,
    /// Picture number (short-term).
    pub pic_num: i32,
}

impl Default for ReferencePicture {
    fn default() -> Self {
        Self {
            dpb_index: -1,
            poc: 0,
            frame_num: 0,
            is_long_term: false,
            long_term_pic_num: 0,
            pic_num: 0,
        }
    }
}

/// Reference picture list (L0 or L1).
#[derive(Debug, Clone)]
pub struct ReferenceList {
    /// Reference pictures in this list.
    pub refs: [ReferencePicture; MAX_REF_LIST_SIZE],
    /// Number of active references in this list.
    pub count: u8,
    /// Number of active references minus 1 (for slice header).
    pub num_ref_idx_active_minus1: u8,
}

impl Default for ReferenceList {
    fn default() -> Self {
        Self::new()
    }
}

impl ReferenceList {
    /// Create an empty reference list.
    pub fn new() -> Self {
        Self {
            refs: [ReferencePicture::default(); MAX_REF_LIST_SIZE],
            count: 0,
            num_ref_idx_active_minus1: 0,
        }
    }

    /// Clear the reference list.
    pub fn clear(&mut self) {
        self.count = 0;
        self.num_ref_idx_active_minus1 = 0;
        for r in &mut self.refs {
            r.dpb_index = -1;
        }
    }

    /// Add a reference to the list.
    pub fn add(&mut self, reference: ReferencePicture) {
        if (self.count as usize) < MAX_REF_LIST_SIZE {
            self.refs[self.count as usize] = reference;
            self.count += 1;
        }
    }

    /// Get the DPB indices of references in this list.
    pub fn dpb_indices(&self) -> Vec<i8> {
        self.refs[..self.count as usize]
            .iter()
            .map(|r| r.dpb_index)
            .collect()
    }

    /// Check if a DPB index is in this list.
    pub fn contains_dpb_index(&self, dpb_index: i8) -> bool {
        self.refs[..self.count as usize]
            .iter()
            .any(|r| r.dpb_index == dpb_index)
    }
}

/// Reference list builder for H.264.
pub struct H264ReferenceListBuilder;

impl H264ReferenceListBuilder {
    /// Initialize reference picture list for P-frames (L0 only).
    ///
    /// For P-frames, L0 is initialized with short-term references sorted by.
    /// descending PicNum, followed by long-term references sorted by ascending
    /// LongTermPicNum.
    pub fn init_ref_list_p_frame(
        dpb: &[DpbEntry; MAX_DPB_SIZE],
        dpb_size: i8,
        current_frame_num: u32,
        max_frame_num: u32,
    ) -> ReferenceList {
        let mut list = ReferenceList::new();
        let mut short_term_refs: Vec<ReferencePicture> = Vec::new();
        let mut long_term_refs: Vec<ReferencePicture> = Vec::new();

        // Collect short-term and long-term references.
        for i in 0..dpb_size as usize {
            let entry = &dpb[i];
            if entry.state != DpbState::InUse {
                continue;
            }

            if entry.marking == MarkingState::ShortTerm
                || entry.bottom_field_marking == MarkingState::ShortTerm
            {
                // Calculate frame_num_wrap.
                let frame_num_wrap = if entry.frame_num > current_frame_num {
                    entry.frame_num as i32 - max_frame_num as i32
                } else {
                    entry.frame_num as i32
                };

                short_term_refs.push(ReferencePicture {
                    dpb_index: i as i8,
                    poc: entry.pic_order_cnt,
                    frame_num: entry.frame_num,
                    is_long_term: false,
                    long_term_pic_num: 0,
                    pic_num: frame_num_wrap,
                });
            }

            if entry.marking == MarkingState::LongTerm
                || entry.bottom_field_marking == MarkingState::LongTerm
            {
                long_term_refs.push(ReferencePicture {
                    dpb_index: i as i8,
                    poc: entry.pic_order_cnt,
                    frame_num: entry.frame_num,
                    is_long_term: true,
                    long_term_pic_num: entry.long_term_frame_idx,
                    pic_num: 0,
                });
            }
        }

        // Sort short-term by descending PicNum.
        short_term_refs.sort_by_key(|r| std::cmp::Reverse(r.pic_num));

        // Sort long-term by ascending LongTermPicNum.
        long_term_refs.sort_by_key(|r| r.long_term_pic_num);

        // Build L0: short-term first, then long-term.
        for r in short_term_refs {
            list.add(r);
        }
        for r in long_term_refs {
            list.add(r);
        }

        if list.count > 0 {
            list.num_ref_idx_active_minus1 = list.count - 1;
        }

        list
    }

    /// Initialize reference picture lists for B-frames (L0 and L1).
    ///
    /// For B-frames:
    /// - L0: Short-term refs with POC < current sorted by descending POC,
    ///   then POC > current sorted by ascending POC, then long-term refs.
    /// - L1: Short-term refs with POC > current sorted by ascending POC,
    ///   then POC < current sorted by descending POC, then long-term refs.
    pub fn init_ref_lists_b_frame(
        dpb: &[DpbEntry; MAX_DPB_SIZE],
        dpb_size: i8,
        current_poc: i32,
    ) -> (ReferenceList, ReferenceList) {
        let mut list0 = ReferenceList::new();
        let mut list1 = ReferenceList::new();

        let mut refs_before: Vec<ReferencePicture> = Vec::new();
        let mut refs_after: Vec<ReferencePicture> = Vec::new();
        let mut long_term_refs: Vec<ReferencePicture> = Vec::new();

        // Collect references categorized by POC.
        for i in 0..dpb_size as usize {
            let entry = &dpb[i];
            if entry.state != DpbState::InUse {
                continue;
            }

            if entry.marking == MarkingState::ShortTerm
                || entry.bottom_field_marking == MarkingState::ShortTerm
            {
                let ref_pic = ReferencePicture {
                    dpb_index: i as i8,
                    poc: entry.pic_order_cnt,
                    frame_num: entry.frame_num,
                    is_long_term: false,
                    long_term_pic_num: 0,
                    pic_num: entry.top_pic_num,
                };

                if entry.pic_order_cnt < current_poc {
                    refs_before.push(ref_pic);
                } else if entry.pic_order_cnt > current_poc {
                    refs_after.push(ref_pic);
                }
            }

            if entry.marking == MarkingState::LongTerm
                || entry.bottom_field_marking == MarkingState::LongTerm
            {
                long_term_refs.push(ReferencePicture {
                    dpb_index: i as i8,
                    poc: entry.pic_order_cnt,
                    frame_num: entry.frame_num,
                    is_long_term: true,
                    long_term_pic_num: entry.long_term_frame_idx,
                    pic_num: 0,
                });
            }
        }

        // Sort refs_before by descending POC (closest to current first)
        refs_before.sort_by_key(|r| std::cmp::Reverse(r.poc));

        // Sort refs_after by ascending POC (closest to current first)
        refs_after.sort_by_key(|r| r.poc);

        // Sort long-term by ascending LongTermPicNum.
        long_term_refs.sort_by_key(|r| r.long_term_pic_num);

        // Build L0: refs_before, refs_after, long_term.
        for r in &refs_before {
            list0.add(*r);
        }
        for r in &refs_after {
            list0.add(*r);
        }
        for r in &long_term_refs {
            list0.add(*r);
        }

        // Build L1: refs_after, refs_before, long_term.
        for r in &refs_after {
            list1.add(*r);
        }
        for r in &refs_before {
            list1.add(*r);
        }
        for r in &long_term_refs {
            list1.add(*r);
        }

        if list0.count > 0 {
            list0.num_ref_idx_active_minus1 = list0.count.saturating_sub(1);
        }
        if list1.count > 0 {
            list1.num_ref_idx_active_minus1 = list1.count.saturating_sub(1);
        }

        // H.264 spec: If L0 and L1 have same entries in same order, swap first two in L1.
        if list0.count > 1
            && list1.count > 1
            && list0.count == list1.count
            && Self::lists_identical(&list0, &list1)
        {
            list1.refs.swap(0, 1);
        }

        (list0, list1)
    }

    /// Check if two reference lists have identical entries.
    fn lists_identical(l0: &ReferenceList, l1: &ReferenceList) -> bool {
        if l0.count != l1.count {
            return false;
        }
        for i in 0..l0.count as usize {
            if l0.refs[i].dpb_index != l1.refs[i].dpb_index {
                return false;
            }
        }
        true
    }
}

/// Reference list builder for H.265.
pub struct H265ReferenceListBuilder;

impl H265ReferenceListBuilder {
    /// Build reference picture lists L0 and L1 from the Reference Picture Set.
    ///
    /// For P-frames: L0 only (from stCurrBefore, stCurrAfter, ltCurr).
    /// For B-frames: L0 and L1 with different ordering.
    #[allow(clippy::too_many_arguments)]
    pub fn build_ref_lists(
        dpb: &[DpbEntry; MAX_DPB_SIZE],
        dpb_size: i8,
        pic_type: PictureType,
        st_curr_before: &[i8],
        st_curr_after: &[i8],
        lt_curr: &[i8],
        num_ref_l0: u32,
        num_ref_l1: u32,
        use_multiple_refs: bool,
    ) -> (ReferenceList, ReferenceList) {
        let mut list0 = ReferenceList::new();
        let mut list1 = ReferenceList::new();

        let num_poc_st_curr_before = st_curr_before.iter().filter(|&&x| x >= 0).count();
        let num_poc_st_curr_after = st_curr_after.iter().filter(|&&x| x >= 0).count();
        let num_poc_lt_curr = lt_curr.iter().filter(|&&x| x >= 0).count();
        let num_poc_total_curr = num_poc_st_curr_before + num_poc_st_curr_after + num_poc_lt_curr;

        // Set active reference counts.
        list0.num_ref_idx_active_minus1 = if num_ref_l0 > 0 {
            (num_ref_l0 - 1) as u8
        } else {
            0
        };
        list1.num_ref_idx_active_minus1 = if num_ref_l1 > 0 {
            (num_ref_l1 - 1) as u8
        } else {
            0
        };

        // Adjust based on available references.
        if use_multiple_refs {
            if (list0.num_ref_idx_active_minus1 as usize + 1) > num_poc_st_curr_before {
                list0.num_ref_idx_active_minus1 = num_poc_st_curr_before.saturating_sub(1) as u8;
            }
            if pic_type == PictureType::B
                && (list1.num_ref_idx_active_minus1 as usize + 1) > num_poc_st_curr_after
            {
                list1.num_ref_idx_active_minus1 = num_poc_st_curr_after.saturating_sub(1) as u8;
            }
        }

        // Build L0 for P and B frames.
        if pic_type == PictureType::P || pic_type == PictureType::B {
            let num_rps_curr_temp_list0 = std::cmp::max(
                list0.num_ref_idx_active_minus1 as usize + 1,
                num_poc_total_curr,
            );

            let mut r_idx = 0;
            while r_idx < num_rps_curr_temp_list0 {
                // Add stCurrBefore.
                for &dpb_idx in st_curr_before.iter().filter(|&&x| x >= 0) {
                    if r_idx >= num_rps_curr_temp_list0 {
                        break;
                    }
                    if dpb_idx >= 0 && (dpb_idx as usize) < dpb_size as usize {
                        let entry = &dpb[dpb_idx as usize];
                        list0.add(ReferencePicture {
                            dpb_index: dpb_idx,
                            poc: entry.pic_order_cnt,
                            frame_num: entry.frame_num,
                            is_long_term: false,
                            long_term_pic_num: 0,
                            pic_num: entry.top_pic_num,
                        });
                    }
                    r_idx += 1;
                }
                // Add stCurrAfter.
                for &dpb_idx in st_curr_after.iter().filter(|&&x| x >= 0) {
                    if r_idx >= num_rps_curr_temp_list0 {
                        break;
                    }
                    if dpb_idx >= 0 && (dpb_idx as usize) < dpb_size as usize {
                        let entry = &dpb[dpb_idx as usize];
                        list0.add(ReferencePicture {
                            dpb_index: dpb_idx,
                            poc: entry.pic_order_cnt,
                            frame_num: entry.frame_num,
                            is_long_term: false,
                            long_term_pic_num: 0,
                            pic_num: entry.top_pic_num,
                        });
                    }
                    r_idx += 1;
                }
                // Add ltCurr.
                for &dpb_idx in lt_curr.iter().filter(|&&x| x >= 0) {
                    if r_idx >= num_rps_curr_temp_list0 {
                        break;
                    }
                    if dpb_idx >= 0 && (dpb_idx as usize) < dpb_size as usize {
                        let entry = &dpb[dpb_idx as usize];
                        list0.add(ReferencePicture {
                            dpb_index: dpb_idx,
                            poc: entry.pic_order_cnt,
                            frame_num: entry.frame_num,
                            is_long_term: true,
                            long_term_pic_num: entry.long_term_frame_idx,
                            pic_num: 0,
                        });
                    }
                    r_idx += 1;
                }

                // If we still need more refs, we'll loop and repeat.
                if r_idx >= num_rps_curr_temp_list0 {
                    break;
                }
            }
        }

        // Build L1 for B frames.
        if pic_type == PictureType::B {
            let num_rps_curr_temp_list1 = std::cmp::max(
                list1.num_ref_idx_active_minus1 as usize + 1,
                num_poc_total_curr,
            );

            let mut r_idx = 0;
            while r_idx < num_rps_curr_temp_list1 {
                // Add stCurrAfter first (future references)
                for &dpb_idx in st_curr_after.iter().filter(|&&x| x >= 0) {
                    if r_idx >= num_rps_curr_temp_list1 {
                        break;
                    }
                    if dpb_idx >= 0 && (dpb_idx as usize) < dpb_size as usize {
                        let entry = &dpb[dpb_idx as usize];
                        list1.add(ReferencePicture {
                            dpb_index: dpb_idx,
                            poc: entry.pic_order_cnt,
                            frame_num: entry.frame_num,
                            is_long_term: false,
                            long_term_pic_num: 0,
                            pic_num: entry.top_pic_num,
                        });
                    }
                    r_idx += 1;
                }
                // Add stCurrBefore.
                for &dpb_idx in st_curr_before.iter().filter(|&&x| x >= 0) {
                    if r_idx >= num_rps_curr_temp_list1 {
                        break;
                    }
                    if dpb_idx >= 0 && (dpb_idx as usize) < dpb_size as usize {
                        let entry = &dpb[dpb_idx as usize];
                        list1.add(ReferencePicture {
                            dpb_index: dpb_idx,
                            poc: entry.pic_order_cnt,
                            frame_num: entry.frame_num,
                            is_long_term: false,
                            long_term_pic_num: 0,
                            pic_num: entry.top_pic_num,
                        });
                    }
                    r_idx += 1;
                }
                // Add ltCurr.
                for &dpb_idx in lt_curr.iter().filter(|&&x| x >= 0) {
                    if r_idx >= num_rps_curr_temp_list1 {
                        break;
                    }
                    if dpb_idx >= 0 && (dpb_idx as usize) < dpb_size as usize {
                        let entry = &dpb[dpb_idx as usize];
                        list1.add(ReferencePicture {
                            dpb_index: dpb_idx,
                            poc: entry.pic_order_cnt,
                            frame_num: entry.frame_num,
                            is_long_term: true,
                            long_term_pic_num: entry.long_term_frame_idx,
                            pic_num: 0,
                        });
                    }
                    r_idx += 1;
                }

                if r_idx >= num_rps_curr_temp_list1 {
                    break;
                }
            }
        }

        (list0, list1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_list_add() {
        let mut list = ReferenceList::new();
        assert_eq!(list.count, 0);

        list.add(ReferencePicture {
            dpb_index: 0,
            poc: 0,
            ..Default::default()
        });
        assert_eq!(list.count, 1);

        list.add(ReferencePicture {
            dpb_index: 1,
            poc: 2,
            ..Default::default()
        });
        assert_eq!(list.count, 2);
    }

    #[test]
    fn test_reference_list_dpb_indices() {
        let mut list = ReferenceList::new();
        list.add(ReferencePicture {
            dpb_index: 2,
            ..Default::default()
        });
        list.add(ReferencePicture {
            dpb_index: 5,
            ..Default::default()
        });

        let indices = list.dpb_indices();
        assert_eq!(indices, vec![2, 5]);
    }
}
