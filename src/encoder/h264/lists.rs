//! The initial reference picture lists of an H.264 B-frame.
//!
//! The hardware writes the slice header from the lists it is given, and it
//! does not reorder them. A decoder builds its own initial lists from the
//! pictures it holds (clause 8.2.4.2.3), so the lists the encoder uses have
//! to be those, cut to the active counts.

/// The initial `RefPicList0` and `RefPicList1` of a B-frame with POC
/// `current`, as indices into `pocs`, the POCs of the short-term references
/// held.
///
/// List 0 is the references shown before the frame, nearest first, then
/// those shown after it, nearest first. List 1 is the other way around. If
/// that makes both lists the same, with more than one entry, the first two
/// entries of list 1 change places.
pub(super) fn b_frame_lists(pocs: &[i32], current: i32) -> (Vec<usize>, Vec<usize>) {
    let mut before: Vec<usize> = (0..pocs.len()).filter(|&i| pocs[i] < current).collect();
    let mut after: Vec<usize> = (0..pocs.len()).filter(|&i| pocs[i] > current).collect();
    before.sort_by_key(|&i| std::cmp::Reverse(pocs[i]));
    after.sort_by_key(|&i| pocs[i]);

    let l0: Vec<usize> = before.iter().chain(&after).copied().collect();
    let mut l1: Vec<usize> = after.iter().chain(&before).copied().collect();
    if l1.len() > 1 && l1 == l0 {
        l1.swap(0, 1);
    }
    (l0, l1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_first_on_each_side() {
        // References in decode order, newest first: P6, P3, I0. Current B4.
        let pocs = [12, 6, 0];
        let (l0, l1) = b_frame_lists(&pocs, 8);
        assert_eq!(l0, [1, 2, 0]);
        assert_eq!(l1, [0, 1, 2]);
    }

    #[test]
    fn references_on_both_sides_interleave_by_distance() {
        // A pyramid: I0, P8, B4 held, current B2.
        let pocs = [0, 16, 8];
        let (l0, l1) = b_frame_lists(&pocs, 4);
        assert_eq!(l0, [0, 2, 1]);
        assert_eq!(l1, [2, 1, 0]);
    }

    #[test]
    fn equal_lists_swap_the_first_two_of_list_1() {
        // Nothing shown after the frame: both lists would be the same.
        let pocs = [0, 2, 4];
        let (l0, l1) = b_frame_lists(&pocs, 6);
        assert_eq!(l0, [2, 1, 0]);
        assert_eq!(l1, [1, 2, 0]);
    }

    #[test]
    fn one_reference_is_not_swapped() {
        let (l0, l1) = b_frame_lists(&[0], 2);
        assert_eq!(l0, [0]);
        assert_eq!(l1, [0]);
    }
}
