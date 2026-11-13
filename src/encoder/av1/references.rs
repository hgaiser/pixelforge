//! Which references an AV1 frame predicts from, under which reference names,
//! and the packet that shows a hidden frame.
//!
//! AV1 has no B-frame type. A frame that predicts from both directions is an
//! inter frame with compound prediction from a past reference name (`LAST`
//! to `GOLDEN`) and a future one (`BWDREF` to `ALTREF`). The future reference
//! was encoded before it as a hidden frame (`show_frame = 0`), and a frame
//! header with `show_existing_frame = 1` shows that frame when its turn
//! comes.

/// The first reference name that predicts from the future, `BWDREF`. The
/// names before it (`LAST`, `LAST2`, `LAST3`, `GOLDEN`) predict from the past.
const FIRST_FUTURE_NAME: usize = 4;

/// The references of one frame.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct FrameReferences {
    /// The DPB slot each of the 7 reference names predicts from, or -1.
    pub names: [i32; 7],
    /// The DPB slot the names that are not predicted from point at in the
    /// frame header, which needs a slot for every name: one for the past
    /// names and one for the future names.
    pub fill: [i8; 2],
}

/// The references of a frame that predicts from both directions.
///
/// `past` and `future` are the DPB slots of the references shown before the
/// frame and after it, each nearest first. Up to `past_count` of the past
/// ones take the past names in `mask`, and the nearest future one takes the
/// first future name in `mask`, where bit 0 of `mask` is `LAST`. `None` if
/// a side has no reference or no name.
pub(super) fn bidirectional_references(
    past: &[u8],
    future: &[u8],
    mask: u32,
    past_count: usize,
) -> Option<FrameReferences> {
    let allowed = |name: &usize| mask & (1 << name) != 0;
    let mut names = [-1i32; 7];
    let past_names = (0..FIRST_FUTURE_NAME).filter(allowed);
    let mut assigned = 0;
    for (name, &slot) in past_names.zip(past.iter().take(past_count)) {
        names[name] = i32::from(slot);
        assigned += 1;
    }
    let future_name = (FIRST_FUTURE_NAME..7).find(allowed)?;
    names[future_name] = i32::from(*future.first()?);
    if assigned == 0 {
        return None;
    }
    Some(FrameReferences {
        names,
        fill: [past[0] as i8, future[0] as i8],
    })
}

/// The slot each reference name has in the frame header: its own, or the
/// fill of its side.
pub(super) fn ref_frame_idx(references: &FrameReferences) -> [i8; 7] {
    let mut idx = [0i8; 7];
    for (name, slot) in idx.iter_mut().enumerate() {
        *slot = match references.names[name] {
            -1 => references.fill[(name >= FIRST_FUTURE_NAME) as usize],
            slot => slot as i8,
        };
    }
    idx
}

/// A temporal unit that shows the frame in frame buffer `slot`: a temporal
/// delimiter and a frame header with `show_existing_frame = 1`.
///
/// The header is `show_existing_frame` (1 bit), `frame_to_show_map_idx` (3
/// bits) and the trailing bits. Nothing else is in it, because the sequence
/// header has no decoder model and no frame ids.
pub(super) fn show_existing_frame(slot: u8) -> Vec<u8> {
    debug_assert!(slot < 8);
    vec![
        // OBU_TEMPORAL_DELIMITER with a size field, of size 0.
        0x12,
        0x00,
        // OBU_FRAME_HEADER with a size field, of size 1.
        0x1A,
        0x01,
        0x80 | (slot & 7) << 4 | 0x08,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_references_take_the_first_names() {
        // Every name allowed: LAST and LAST2 from the past, BWDREF from the
        // future.
        let references = bidirectional_references(&[3, 1], &[2, 5], 0x7F, 2).unwrap();
        assert_eq!(references.names, [3, 1, -1, -1, 2, -1, -1]);
        assert_eq!(references.fill, [3, 2]);
        assert_eq!(ref_frame_idx(&references), [3, 1, 3, 3, 2, 2, 2]);
    }

    #[test]
    fn only_names_in_the_mask_are_used() {
        // LAST and ALTREF only.
        let references = bidirectional_references(&[3, 1], &[2], 0b100_0001, 2).unwrap();
        assert_eq!(references.names, [3, -1, -1, -1, -1, -1, 2]);
    }

    #[test]
    fn the_past_count_limits_the_past_names() {
        let references = bidirectional_references(&[3, 1, 0], &[2], 0x7F, 1).unwrap();
        assert_eq!(references.names, [3, -1, -1, -1, 2, -1, -1]);
    }

    #[test]
    fn both_sides_are_needed() {
        assert_eq!(bidirectional_references(&[], &[2], 0x7F, 1), None);
        assert_eq!(bidirectional_references(&[3], &[], 0x7F, 1), None);
        assert_eq!(bidirectional_references(&[3], &[2], 0x0F, 1), None);
        assert_eq!(bidirectional_references(&[3], &[2], 0x70, 1), None);
    }

    #[test]
    fn the_show_packet_is_what_the_decoder_reads() {
        use crate::decoder::av1::obu::{ObuType, iter_obus};
        let packet = show_existing_frame(5);
        let obus: Vec<_> = iter_obus(&packet).collect::<Result<_, _>>().unwrap();
        assert_eq!(obus.len(), 2);
        assert_eq!(obus[0].header.obu_type, ObuType::TemporalDelimiter);
        assert_eq!(obus[1].header.obu_type, ObuType::FrameHeader);
        // show_existing_frame = 1, frame_to_show_map_idx = 5, trailing bits.
        assert_eq!(obus[1].payload, [0b1101_1000]);
    }
}
