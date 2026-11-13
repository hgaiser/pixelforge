//! The references an encoder holds in its DPB, and the rules for which ones
//! it keeps: the same for each codec.
//!
//! The window counts the IDR and P-frames (level 0). The B-frames of a
//! pyramid that are references are held on top of it, until a later
//! reference frame of their level or of a level above makes them of no more
//! use (see [`obsolete_b_references`]).

use super::gop::obsolete_b_references;

/// One reference frame in the DPB.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Reference<T> {
    pub dpb_slot: u8,
    /// The POC: twice the frames shown since the last IDR.
    pub poc: i32,
    /// Display order (`pts`) since the encoder was made.
    pub display_order: u64,
    /// Encode order since the encoder was made.
    pub encode_index: u64,
    /// 0 for an IDR or P-frame, and the pyramid level of a B-frame.
    pub level: u8,
    /// What only one codec needs to know about the reference.
    pub codec: T,
}

/// The references held, newest first in encode order.
pub(crate) struct ReferenceSet<T> {
    references: Vec<Reference<T>>,
    /// How many IDR and P-frames stay.
    window: usize,
}

impl<T: Copy> ReferenceSet<T> {
    pub fn new(window: u32) -> Self {
        Self {
            references: Vec::new(),
            window: window.max(1) as usize,
        }
    }

    /// The references, newest first in encode order.
    pub fn as_slice(&self) -> &[Reference<T>] {
        &self.references
    }

    pub fn is_empty(&self) -> bool {
        self.references.is_empty()
    }

    pub fn len(&self) -> usize {
        self.references.len()
    }

    /// The newest `count` references, in encode order.
    pub fn newest(&self, count: usize) -> &[Reference<T>] {
        &self.references[..self.references.len().min(count)]
    }

    /// Drop all references, for an IDR.
    pub fn clear(&mut self) {
        self.references.clear();
    }

    /// Before a reference frame of `level` is encoded: drop the B-frame
    /// references it makes of no more use, and return them.
    pub fn drop_obsolete(&mut self, level: u8) -> Vec<Reference<T>> {
        let levels: Vec<u8> = self.references.iter().map(|r| r.level).collect();
        obsolete_b_references(&levels, level)
            .into_iter()
            .rev()
            .map(|index| self.references.remove(index))
            .collect()
    }

    /// The reference that [`Self::insert`] drops for a new reference frame
    /// of `level`, if any: the oldest IDR or P-frame, once the window is full.
    pub fn window_victim(&self, level: u8) -> Option<&Reference<T>> {
        if level > 0 || self.references.iter().filter(|r| r.level == 0).count() < self.window {
            return None;
        }
        self.references.iter().rev().find(|r| r.level == 0)
    }

    /// Add the frame just encoded as the newest reference, and drop the
    /// oldest IDR or P-frame if that makes the window too full.
    pub fn insert(&mut self, reference: Reference<T>) {
        let victim = self
            .window_victim(reference.level)
            .map(|victim| victim.encode_index);
        self.references.retain(|r| Some(r.encode_index) != victim);
        self.references.insert(0, reference);
    }

    /// A DPB slot that no reference holds, out of `slot_count`: `preferred`
    /// if that one is free, or else the lowest free one.
    pub fn free_slot(&self, slot_count: usize, preferred: Option<u8>) -> u8 {
        let free = |slot: &u8| !self.references.iter().any(|r| r.dpb_slot == *slot);
        preferred
            .filter(|slot| (*slot as usize) < slot_count && free(slot))
            .or_else(|| (0..slot_count as u8).find(free))
            .expect("the DPB has a slot for the frame being encoded")
    }

    /// The references shown before the frame with POC `poc`, and those
    /// shown after it, each nearest first.
    pub fn around(&self, poc: i32) -> (Vec<Reference<T>>, Vec<Reference<T>>) {
        let mut before: Vec<Reference<T>> = self
            .references
            .iter()
            .filter(|r| r.poc < poc)
            .copied()
            .collect();
        let mut after: Vec<Reference<T>> = self
            .references
            .iter()
            .filter(|r| r.poc > poc)
            .copied()
            .collect();
        before.sort_by_key(|r| std::cmp::Reverse(r.poc));
        after.sort_by_key(|r| r.poc);
        (before, after)
    }

    /// Reference frame invalidation: drop the references a client that lost
    /// the frame shown at `first_lost_display_order`, and every frame after
    /// it, cannot decode. Returns them.
    ///
    /// A client loses packets in decode order. So what it lost begins with
    /// the first frame encoded that is shown at or after the lost one, and
    /// every reference encoded from there on is lost too. Without B-frames
    /// that is the references shown at or after the lost frame. With
    /// B-frames it can include a B-frame shown before it, which is the safe
    /// side.
    ///
    /// A frame shown at or after the lost one that is not a reference was
    /// encoded after a reference that is (the one after its group), and
    /// that reference is held: the window keeps every P-frame encoded since.
    pub fn invalidate(&mut self, first_lost_display_order: u64) -> Vec<Reference<T>> {
        let Some(first_lost) = self
            .references
            .iter()
            .filter(|r| r.display_order >= first_lost_display_order)
            .map(|r| r.encode_index)
            .min()
        else {
            return Vec::new();
        };
        let (lost, kept) = self
            .references
            .drain(..)
            .partition(|r| r.encode_index >= first_lost);
        self.references = kept;
        lost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(encode_index: u64, display_order: u64, level: u8) -> Reference<()> {
        Reference {
            dpb_slot: encode_index as u8,
            poc: 2 * display_order as i32,
            display_order,
            encode_index,
            level,
            codec: (),
        }
    }

    fn shown(set: &ReferenceSet<()>) -> Vec<u64> {
        set.as_slice().iter().map(|r| r.display_order).collect()
    }

    #[test]
    fn the_window_counts_p_frames_only() {
        let mut set = ReferenceSet::new(2);
        set.insert(reference(0, 0, 0));
        set.insert(reference(1, 4, 0));
        set.insert(reference(2, 2, 1));
        assert_eq!(shown(&set), [2, 4, 0]);
        assert!(set.window_victim(1).is_none());
        assert_eq!(set.window_victim(0).unwrap().display_order, 0);

        assert_eq!(set.drop_obsolete(0).len(), 1);
        set.insert(reference(5, 8, 0));
        assert_eq!(shown(&set), [8, 4]);
    }

    #[test]
    fn free_slot_prefers_the_slot_asked_for() {
        let mut set = ReferenceSet::new(4);
        set.insert(reference(0, 0, 0));
        set.insert(reference(2, 1, 0));
        assert_eq!(set.free_slot(4, None), 1);
        assert_eq!(set.free_slot(4, Some(3)), 3);
        assert_eq!(set.free_slot(4, Some(2)), 1);
    }

    #[test]
    fn around_sorts_nearest_first() {
        let mut set = ReferenceSet::new(4);
        for (e, d) in [(0, 0), (1, 8), (2, 4)] {
            set.insert(reference(e, d, 0));
        }
        let (before, after) = set.around(2 * 5);
        let d = |list: Vec<Reference<()>>| list.iter().map(|r| r.display_order).collect::<Vec<_>>();
        assert_eq!(d(before), [4, 0]);
        assert_eq!(d(after), [8]);
    }

    #[test]
    fn invalidation_without_b_frames_drops_what_is_shown_from_the_loss() {
        let mut set = ReferenceSet::new(4);
        for i in 0..4 {
            set.insert(reference(i, i, 0));
        }
        let lost: Vec<u64> = set.invalidate(2).iter().map(|r| r.display_order).collect();
        assert_eq!(lost, [3, 2]);
        assert_eq!(shown(&set), [1, 0]);
        assert!(set.invalidate(7).is_empty());
    }

    #[test]
    fn invalidation_with_b_frames_drops_what_is_encoded_from_the_loss() {
        // I0 P4 B2 b1 b3, then b3 is lost: P4 is the first frame encoded
        // that is shown at or after it, so P4 and B2 go, and I0 stays.
        let mut set = ReferenceSet::new(2);
        set.insert(reference(0, 0, 0));
        set.insert(reference(1, 4, 0));
        set.insert(reference(2, 2, 1));
        set.invalidate(3);
        assert_eq!(shown(&set), [0]);
    }
}
