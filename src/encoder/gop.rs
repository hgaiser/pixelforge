//! GOP (Group of Pictures) planning: frame types, and the order frames are
//! encoded in.
//!
//! Frames come in in display order. Without B-frames each one is encoded at
//! once, as an IDR or a P-frame. With B-frames, a group of `b_frames` frames
//! waits for the reference frame shown after it: that one is encoded first,
//! as a P-frame, and then the group, as B-frames that predict from both.
//!
//! The planner is codec-agnostic and has no Vulkan in it. It holds the frames
//! that wait, as whatever the caller gives it, and hands them back in encode
//! order with their [`GopPosition`].

use super::BFrameReferences;

/// Frame type in GOP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GopFrameType {
    /// IDR frame.
    Idr,
    /// P frame (predicted).
    P,
    /// B frame (predicted from both directions).
    B,
}

impl GopFrameType {
    /// Whether this is an IDR frame.
    pub fn is_idr(&self) -> bool {
        matches!(self, Self::Idr)
    }
}

/// Position in the GOP.
#[derive(Debug, Clone)]
pub struct GopPosition {
    /// Frame type.
    pub frame_type: GopFrameType,
    /// POC (Picture Order Count): twice the frames shown since the last IDR.
    ///
    /// It does not wrap. Writing its least significant bits into the slice
    /// header is the driver's job, and the list and RPS code sorts and
    /// subtracts whole values.
    pub pic_order_cnt: i32,
    /// Whether this frame is a reference.
    pub is_reference: bool,
    /// How deep in a B-frame pyramid the frame is: 0 for IDR and P-frames, 1
    /// for the B-frames of a flat group and for the middle B-frame of a
    /// pyramid, and one more for each split below that.
    pub pyramid_level: u8,
}

/// The held references that are of no more use once a frame of
/// `new_level` is encoded, as indices into `levels`, the pyramid levels of
/// the held references.
///
/// A B-frame that is a reference is used only by the frames between the two
/// references it was predicted from. In encode order those all come before
/// the next frame of its own level or of a level above it. So a reference
/// frame replaces every B-frame reference of its level and of the levels
/// below it, and the references the frame itself needs all stay.
pub fn obsolete_b_references(levels: &[u8], new_level: u8) -> Vec<usize> {
    (0..levels.len())
        .filter(|&i| levels[i] > 0 && levels[i] >= new_level)
        .collect()
}

/// A B-frame of a group, in encode order: its index in the group's display
/// order, whether it is a reference, and its pyramid level.
type GroupEntry = (usize, bool, u8);

/// The encode order of a group of `count` B-frames.
///
/// Flat: display order, none a reference. Pyramid: the middle frame first, as
/// a reference, then each half the same way. A range of fewer than three
/// frames has no middle frame that frames on both sides use, so it is flat.
fn b_frame_order(count: usize, references: BFrameReferences) -> Vec<GroupEntry> {
    fn split(lo: usize, len: usize, level: u8, out: &mut Vec<GroupEntry>) {
        if len < 3 {
            out.extend((lo..lo + len).map(|i| (i, false, level)));
            return;
        }
        let mid = (len - 1) / 2;
        out.push((lo + mid, true, level));
        split(lo, mid, level + 1, out);
        split(lo + mid + 1, len - mid - 1, level + 1, out);
    }

    let mut out = Vec::with_capacity(count);
    match references {
        BFrameReferences::Flat => out.extend((0..count).map(|i| (i, false, 1))),
        BFrameReferences::Pyramid => split(0, count, 1, &mut out),
    }
    out
}

/// GOP planner: takes frames in display order and gives them back in encode
/// order.
pub struct GopPlanner<T> {
    /// IDR period in frames, or 0 for no periodic IDR.
    idr_period: u32,
    b_frames: u32,
    references: BFrameReferences,
    /// Display position of the next frame since the last IDR, or `None`
    /// before the first frame.
    next_position: Option<u32>,
    /// Flag to force next frame to be an IDR.
    force_idr: bool,
    /// The frames that wait for the reference frame after them, in display
    /// order, with their display position since the last IDR.
    waiting: Vec<(T, u32)>,
    /// See [`Self::b_reference_count`].
    b_reference_count: usize,
    /// See [`Self::reorder_delay`].
    reorder_delay: u32,
}

/// The reference pattern of the B-frames, as the rate control is told it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferencePattern {
    /// No B-frame is a reference.
    Flat,
    /// Each B-frame reference is in the middle of the two references it
    /// predicts from, which only `2^n - 1` B-frames give.
    Dyadic,
}

impl<T> GopPlanner<T> {
    /// A planner with an IDR every `idr_period` frames (0 for none) and
    /// `b_frames` B-frames between two reference frames.
    pub fn new(idr_period: u32, b_frames: u32, references: BFrameReferences) -> Self {
        let order = b_frame_order(b_frames as usize, references);
        Self {
            idr_period,
            b_frames,
            references,
            next_position: None,
            force_idr: false,
            waiting: Vec::new(),
            b_reference_count: b_reference_count(&order),
            reorder_delay: reorder_delay(&order),
        }
    }

    /// Take the next frame in display order, and give back the frames that
    /// can be encoded now, in encode order. Empty while `frame` waits.
    pub fn push(&mut self, frame: T) -> Vec<(T, GopPosition)> {
        let position = match self.next_position {
            Some(position)
                if !self.force_idr && (self.idr_period == 0 || position < self.idr_period) =>
            {
                position
            }
            _ => {
                // A group never crosses an IDR: what waits is closed first.
                let mut out = self.close();
                self.force_idr = false;
                self.next_position = Some(1);
                out.push((
                    frame,
                    GopPosition {
                        frame_type: GopFrameType::Idr,
                        pic_order_cnt: 0,
                        is_reference: true,
                        pyramid_level: 0,
                    },
                ));
                return out;
            }
        };
        self.next_position = Some(position + 1);
        self.waiting.push((frame, position));
        // The frame before a periodic IDR is a reference frame, so the GOP
        // is closed.
        let before_idr = self.idr_period > 0 && position + 1 == self.idr_period;
        if self.waiting.len() > self.b_frames as usize || before_idr {
            self.close()
        } else {
            Vec::new()
        }
    }

    /// Encode the frames that wait now, as a shorter group: the last one
    /// becomes the P-frame, and the others its B-frames. Empty if no frame
    /// waits.
    pub fn close(&mut self) -> Vec<(T, GopPosition)> {
        let Some((anchor, anchor_position)) = self.waiting.pop() else {
            return Vec::new();
        };
        let mut out = Vec::with_capacity(self.waiting.len() + 1);
        out.push((
            anchor,
            GopPosition {
                frame_type: GopFrameType::P,
                pic_order_cnt: poc(anchor_position),
                is_reference: true,
                pyramid_level: 0,
            },
        ));
        let mut group: Vec<Option<(T, u32)>> = self.waiting.drain(..).map(Some).collect();
        for (index, is_reference, level) in b_frame_order(group.len(), self.references) {
            let (frame, position) = group[index].take().expect("each frame once");
            out.push((
                frame,
                GopPosition {
                    frame_type: GopFrameType::B,
                    pic_order_cnt: poc(position),
                    is_reference,
                    pyramid_level: level,
                },
            ));
        }
        out
    }

    /// Request that the next frame be an IDR frame. The frames that wait are
    /// encoded before it, by the next [`Self::push`] or by [`Self::close`].
    pub fn request_idr(&mut self) {
        self.force_idr = true;
    }

    /// Drop the frames that wait, and make the next frame an IDR. For when
    /// the reference state is lost.
    pub fn abandon(&mut self) -> Vec<T> {
        self.force_idr = true;
        self.waiting.drain(..).map(|(frame, _)| frame).collect()
    }

    /// Make the oldest frame that waits an IDR now, and keep the others
    /// waiting after it. Returns the IDR, or nothing if no frame waits, and
    /// then the next frame is an IDR. For when the reference state is lost:
    /// the frames that wait are then never encoded from it.
    pub fn restart(&mut self) -> Vec<(T, GopPosition)> {
        self.force_idr = true;
        let waiting: Vec<T> = self.waiting.drain(..).map(|(frame, _)| frame).collect();
        let mut out = Vec::new();
        for frame in waiting {
            out.extend(self.push(frame));
        }
        out
    }

    /// How many B-frames are held as references at once at most: one for
    /// each pyramid level that has references. 0 for flat groups.
    pub fn b_reference_count(&self) -> usize {
        self.b_reference_count
    }

    /// The largest number of frames that are encoded before a frame and
    /// shown after it: how far decode order runs ahead of display order. 0
    /// without B-frames, 1 for a flat group, more for a pyramid.
    pub fn reorder_delay(&self) -> u32 {
        self.reorder_delay
    }

    /// The reference pattern to tell the rate control, or `None` for a
    /// pyramid that is not dyadic.
    pub fn reference_pattern(&self) -> Option<ReferencePattern> {
        if self.b_reference_count == 0 {
            Some(ReferencePattern::Flat)
        } else if (self.b_frames + 1).is_power_of_two() {
            Some(ReferencePattern::Dyadic)
        } else {
            None
        }
    }
}

/// See [`GopPlanner::b_reference_count`]; `order` is the encode order of a
/// whole group.
fn b_reference_count(order: &[GroupEntry]) -> usize {
    order
        .iter()
        .filter(|&&(_, is_reference, _)| is_reference)
        .map(|&(_, _, level)| level as usize)
        .max()
        .unwrap_or(0)
}

/// See [`GopPlanner::reorder_delay`]; `order` is the encode order of a
/// whole group.
fn reorder_delay(order: &[GroupEntry]) -> u32 {
    let count = order.len();
    if count == 0 {
        return 0;
    }
    // Display positions in encode order: the reference frame after the
    // group, then the group. A shorter group never delays more.
    let mut encode_order = vec![count];
    encode_order.extend(order.iter().map(|&(index, _, _)| index));
    encode_order
        .iter()
        .enumerate()
        .map(|(e, &shown)| encode_order[..e].iter().filter(|&&d| d > shown).count())
        .max()
        .unwrap_or(0) as u32
}

/// The POC of the frame at `position` since the last IDR.
fn poc(position: u32) -> i32 {
    2 * position as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Push `frames`, numbered in display order, and return what came out,
    /// as (display number, type, POC, reference).
    fn run(
        planner: &mut GopPlanner<u32>,
        frames: std::ops::Range<u32>,
    ) -> Vec<(u32, GopFrameType, i32, bool)> {
        let mut out = Vec::new();
        for frame in frames {
            out.extend(planner.push(frame));
        }
        out.into_iter()
            .map(|(f, p)| (f, p.frame_type, p.pic_order_cnt, p.is_reference))
            .collect()
    }

    fn types(out: &[(u32, GopFrameType, i32, bool)]) -> String {
        out.iter()
            .map(|(f, t, _, r)| {
                let c = match t {
                    GopFrameType::Idr => "I",
                    GopFrameType::P => "P",
                    GopFrameType::B if *r => "B",
                    GopFrameType::B => "b",
                };
                format!("{c}{f}")
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn test_ip_only_gop() {
        let mut gop = GopPlanner::new(30, 0, BFrameReferences::Flat);

        // Each frame comes out at once.
        let pos = gop.push(0);
        assert_eq!(pos.len(), 1);
        assert_eq!(pos[0].1.frame_type, GopFrameType::Idr);
        assert!(pos[0].1.is_reference);

        for i in 1..30u32 {
            let pos = gop.push(i);
            assert_eq!(pos.len(), 1);
            assert_eq!(
                pos[0].1.frame_type,
                GopFrameType::P,
                "Frame {i} should be P"
            );
            assert_eq!(pos[0].1.pic_order_cnt, 2 * i as i32);
            assert!(pos[0].1.is_reference);
        }

        // Frame 30 should be IDR (new GOP)
        let pos = gop.push(30);
        assert_eq!(pos[0].1.frame_type, GopFrameType::Idr);
        assert_eq!(pos[0].1.pic_order_cnt, 0);
        assert_eq!(gop.reorder_delay(), 0);
    }

    #[test]
    fn poc_does_not_wrap_without_idr() {
        let mut gop = GopPlanner::new(0, 0, BFrameReferences::Flat);
        let out = run(&mut gop, 0..600);
        assert_eq!(out[599].2, 1198);
    }

    #[test]
    fn flat_groups() {
        let mut gop = GopPlanner::new(0, 2, BFrameReferences::Flat);
        let out = run(&mut gop, 0..7);
        assert_eq!(types(&out), "I0 P3 b1 b2 P6 b4 b5");
        assert_eq!(out[2].2, 2);
        assert_eq!(gop.reorder_delay(), 1);
    }

    #[test]
    fn pyramid_groups() {
        let mut gop = GopPlanner::new(0, 3, BFrameReferences::Pyramid);
        let out = run(&mut gop, 0..9);
        assert_eq!(types(&out), "I0 P4 B2 b1 b3 P8 B6 b5 b7");
        assert_eq!(gop.reorder_delay(), 2);

        let mut gop = GopPlanner::new(0, 7, BFrameReferences::Pyramid);
        let out = run(&mut gop, 0..9);
        assert_eq!(types(&out), "I0 P8 B4 B2 b1 b3 B6 b5 b7");
        assert_eq!(gop.reorder_delay(), 3);
    }

    #[test]
    fn short_pyramid_is_flat() {
        let mut gop = GopPlanner::new(0, 2, BFrameReferences::Pyramid);
        assert_eq!(types(&run(&mut gop, 0..4)), "I0 P3 b1 b2");
        assert_eq!(gop.reorder_delay(), 1);
    }

    #[test]
    fn pyramid_levels() {
        let mut gop = GopPlanner::new(0, 7, BFrameReferences::Pyramid);
        let mut out = Vec::new();
        for frame in 0..9 {
            out.extend(gop.push(frame));
        }
        let levels: Vec<(u32, u8)> = out.iter().map(|(f, p)| (*f, p.pyramid_level)).collect();
        assert_eq!(
            levels,
            [
                (0, 0),
                (8, 0),
                (4, 1),
                (2, 2),
                (1, 3),
                (3, 3),
                (6, 2),
                (5, 3),
                (7, 3)
            ]
        );
    }

    #[test]
    fn b_reference_count_is_the_depth_of_the_references() {
        let count = |b_frames, references| {
            GopPlanner::<u32>::new(0, b_frames, references).b_reference_count()
        };
        for b_frames in 0..=7 {
            assert_eq!(count(b_frames, BFrameReferences::Flat), 0);
        }
        let pyramid: Vec<usize> = (0..=7)
            .map(|b_frames| count(b_frames, BFrameReferences::Pyramid))
            .collect();
        assert_eq!(pyramid, [0, 0, 0, 1, 1, 1, 2, 2]);
    }

    /// Encode a long stream, dropping references by `obsolete_b_references`
    /// and by a sliding window of two P-frames: every B-frame must still
    /// find the nearest reference frame on each side, and the B-frame
    /// references held never exceed `b_reference_count`.
    #[test]
    fn obsolete_references_are_never_needed() {
        for b_frames in 0..=7 {
            let mut gop = GopPlanner::new(0, b_frames, BFrameReferences::Pyramid);
            let limit = gop.b_reference_count();
            let mut out = Vec::new();
            for frame in 0..60u32 {
                out.extend(gop.push(frame));
            }
            // (display number, level) of the held references.
            let mut held: Vec<(u32, u8)> = Vec::new();
            // Every frame encoded so far that is a reference.
            let mut all: Vec<u32> = Vec::new();
            for (frame, position) in out {
                if position.is_reference {
                    let levels: Vec<u8> = held.iter().map(|h| h.1).collect();
                    for index in obsolete_b_references(&levels, position.pyramid_level)
                        .into_iter()
                        .rev()
                    {
                        held.remove(index);
                    }
                }
                if position.frame_type == GopFrameType::B {
                    let before = all.iter().filter(|&&r| r < frame).max().unwrap();
                    let after = all.iter().filter(|&&r| r > frame).min().unwrap();
                    assert!(held.iter().any(|h| h.0 == *before), "b={b_frames} {frame}");
                    assert!(held.iter().any(|h| h.0 == *after), "b={b_frames} {frame}");
                }
                if position.is_reference {
                    held.push((frame, position.pyramid_level));
                    all.push(frame);
                    while held.iter().filter(|h| h.1 == 0).count() > 2 {
                        let oldest = held.iter().position(|h| h.1 == 0).unwrap();
                        held.remove(oldest);
                    }
                    assert!(held.iter().filter(|h| h.1 > 0).count() <= limit);
                }
            }
        }
    }

    #[test]
    fn group_never_crosses_a_periodic_idr() {
        // 8 frames per GOP, groups of 3: the last group is short.
        let mut gop = GopPlanner::new(8, 2, BFrameReferences::Flat);
        let out = run(&mut gop, 0..9);
        assert_eq!(types(&out), "I0 P3 b1 b2 P6 b4 b5 P7 I8");
        // POC restarts at the IDR.
        assert_eq!(out[8].2, 0);
    }

    #[test]
    fn request_idr_closes_the_group() {
        let mut gop = GopPlanner::new(0, 3, BFrameReferences::Flat);
        let mut out = run(&mut gop, 0..3);
        assert_eq!(types(&out), "I0");
        gop.request_idr();
        // The waiting frames are closed when asked, before the IDR.
        out = gop
            .close()
            .into_iter()
            .map(|(f, p)| (f, p.frame_type, p.pic_order_cnt, p.is_reference))
            .collect();
        assert_eq!(types(&out), "P2 b1");
        let out = run(&mut gop, 3..5);
        assert_eq!(types(&out), "I3");
        assert_eq!(out[0].2, 0);
    }

    #[test]
    fn request_idr_on_push_closes_first() {
        let mut gop = GopPlanner::new(0, 3, BFrameReferences::Flat);
        run(&mut gop, 0..3);
        gop.request_idr();
        assert_eq!(types(&run(&mut gop, 3..4)), "P2 b1 I3");
    }

    #[test]
    fn close_with_each_count_waiting() {
        for waiting in 0..=3u32 {
            let mut gop = GopPlanner::new(0, 3, BFrameReferences::Flat);
            run(&mut gop, 0..5);
            run(&mut gop, 5..5 + waiting);
            let out = gop.close();
            assert_eq!(out.len(), waiting as usize);
            if waiting > 0 {
                assert_eq!(out[0].0, 4 + waiting);
                assert_eq!(out[0].1.frame_type, GopFrameType::P);
            }
            assert!(gop.close().is_empty());
        }
    }

    #[test]
    fn restart_makes_the_oldest_waiting_frame_the_idr() {
        let mut gop = GopPlanner::new(0, 3, BFrameReferences::Flat);
        run(&mut gop, 0..4);
        let out: Vec<_> = gop
            .restart()
            .into_iter()
            .map(|(f, p)| (f, p.frame_type, p.pic_order_cnt, p.is_reference))
            .collect();
        assert_eq!(types(&out), "I1");
        assert_eq!(out[0].2, 0);
        // The others wait after it, with POCs from it.
        let out = run(&mut gop, 4..6);
        assert_eq!(types(&out), "P5 b2 b3 b4");
        assert_eq!(out[0].2, 8);

        // Nothing waits: the next frame is the IDR.
        assert!(gop.restart().is_empty());
        assert_eq!(types(&run(&mut gop, 6..7)), "I6");
    }

    #[test]
    fn only_full_pyramids_are_dyadic() {
        let pattern = |b_frames, references| {
            GopPlanner::<u32>::new(0, b_frames, references).reference_pattern()
        };
        assert_eq!(
            pattern(3, BFrameReferences::Flat),
            Some(ReferencePattern::Flat)
        );
        assert_eq!(
            pattern(2, BFrameReferences::Pyramid),
            Some(ReferencePattern::Flat)
        );
        assert_eq!(
            pattern(3, BFrameReferences::Pyramid),
            Some(ReferencePattern::Dyadic)
        );
        assert_eq!(pattern(5, BFrameReferences::Pyramid), None);
        assert_eq!(
            pattern(7, BFrameReferences::Pyramid),
            Some(ReferencePattern::Dyadic)
        );
    }

    #[test]
    fn abandon_drops_the_group() {
        let mut gop = GopPlanner::new(0, 3, BFrameReferences::Flat);
        run(&mut gop, 0..3);
        assert_eq!(gop.abandon(), vec![1, 2]);
        assert_eq!(types(&run(&mut gop, 3..4)), "I3");
    }

    #[test]
    fn every_frame_once_and_decode_order_valid() {
        for b_frames in 0..=7 {
            for references in [BFrameReferences::Flat, BFrameReferences::Pyramid] {
                for idr_period in [0, 1, 5, 16] {
                    let mut gop = GopPlanner::new(idr_period, b_frames, references);
                    let mut out = run(&mut gop, 0..50);
                    out.extend(
                        gop.close()
                            .into_iter()
                            .map(|(f, p)| (f, p.frame_type, p.pic_order_cnt, p.is_reference)),
                    );
                    let mut shown: Vec<u32> = out.iter().map(|o| o.0).collect();
                    shown.sort_unstable();
                    assert_eq!(shown, (0..50).collect::<Vec<_>>());

                    // A B-frame comes after a reference frame shown after it,
                    // and the decode order runs ahead by no more than the delay.
                    let delay = gop.reorder_delay() as usize;
                    for (e, (f, t, _, _)) in out.iter().enumerate() {
                        let ahead = out[..e].iter().filter(|o| o.0 > *f).count();
                        assert!(ahead <= delay, "b={b_frames} {references:?} frame {f}");
                        if *t == GopFrameType::B {
                            assert!(out[..e].iter().any(|o| o.0 > *f && o.3));
                        }
                    }
                    // dts = encode index - delay never passes pts.
                    for (e, (f, _, _, _)) in out.iter().enumerate() {
                        assert!(e as i64 - delay as i64 <= *f as i64);
                    }
                }
            }
        }
    }
}
