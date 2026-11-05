//! DPB tests on synthetic slice headers, for the cases the fixtures do not
//! reach: POC wrap-around with `prevTid0Pic`, long-term pictures, missing
//! references, BLA and end of sequence.

use super::*;
use crate::decoder::h265::parser::DecPicBufMgr;
use crate::decoder::h265::rps::{StRefPicSet, StRps};
use crate::decoder::h265::slice::LongTermRef;

/// MaxPicOrderCntLsb = 16, a 6-picture DPB, `sub_layers` temporal layers.
fn sps(sub_layers: u8) -> Sps {
    let mgr = DecPicBufMgr {
        max_dec_pic_buffering_minus1: [5; 7],
        ..Default::default()
    };
    Sps {
        log2_max_pic_order_cnt_lsb_minus4: 0,
        sps_max_sub_layers_minus1: sub_layers - 1,
        dec_pic_buf_mgr: mgr,
        ..Default::default()
    }
}

fn nal(nal_type: NalType, temporal_id: u8) -> NalHeader {
    NalHeader {
        nal_type,
        layer_id: 0,
        temporal_id,
    }
}

/// A slice header with an inline short-term set of (delta, used) pairs.
fn slice(lsb: u32, s0: &[(i32, bool)], s1: &[(i32, bool)]) -> SliceHeader {
    SliceHeader {
        first_slice_segment_in_pic_flag: true,
        pic_output_flag: true,
        slice_pic_order_cnt_lsb: lsb,
        st_ref_pic_set: Some(StRefPicSet {
            rps: StRps {
                delta_poc_s0: s0.iter().map(|e| e.0).collect(),
                used_s0: s0.iter().map(|e| e.1).collect(),
                delta_poc_s1: s1.iter().map(|e| e.0).collect(),
                used_s1: s1.iter().map(|e| e.1).collect(),
            },
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Begin a picture that must be decoded, allocate its slot and finish it.
fn decode(
    dpb: &mut DecodeDpb,
    nal: NalHeader,
    slice: &SliceHeader,
    sps: &Sps,
) -> (PictureState, u8) {
    let state = match dpb.begin_picture(&nal, slice, sps).unwrap() {
        PictureDecision::Decode(state) => state,
        PictureDecision::Skip(reason) => panic!("unexpected skip: {reason:?}"),
    };
    let slot = dpb.try_allocate_slot().expect("free slot");
    dpb.end_picture(slot, &state);
    (state, slot)
}

fn skip_reason(dpb: &mut DecodeDpb, nal: NalHeader, slice: &SliceHeader, sps: &Sps) -> SkipReason {
    match dpb.begin_picture(&nal, slice, sps).unwrap() {
        PictureDecision::Skip(reason) => reason,
        PictureDecision::Decode(state) => panic!("unexpected decode of POC {}", state.poc),
    }
}

fn pocs(dpb: &DecodeDpb) -> Vec<(i32, bool)> {
    dpb.references()
        .iter()
        .map(|r| (r.poc, r.long_term))
        .collect()
}

fn idr() -> SliceHeader {
    SliceHeader {
        first_slice_segment_in_pic_flag: true,
        pic_output_flag: true,
        ..Default::default()
    }
}

/// The POC MSB is predicted from `prevTid0Pic` only: a picture of a higher
/// sub-layer, or a sub-layer non-reference one, sitting between two TemporalId
/// 0 pictures must not move the prediction, or the LSB wrap is misread.
#[test]
fn poc_wraps_against_prev_tid0_pic() {
    for (between, tid) in [
        (NalType::TsaR, 1),
        (NalType::TrailN, 0),
        (NalType::RadlR, 0),
    ] {
        let sps = sps(2);
        let mut dpb = DecodeDpb::new(6, Arc::default());
        let keep_all = |lsb, deltas: &[i32]| {
            let s0: Vec<_> = deltas.iter().map(|d| (*d, true)).collect();
            slice(lsb, &s0, &[])
        };
        let mut poc = |nal_type, tid, slice: &SliceHeader| {
            decode(&mut dpb, nal(nal_type, tid), slice, &sps).0.poc
        };
        assert_eq!(poc(NalType::IdrNLp, 0, &idr()), 0);
        assert_eq!(poc(NalType::TrailR, 0, &keep_all(8, &[-8])), 8);
        // LSB 0 after 8 is a wrap: 16.
        assert_eq!(poc(NalType::TrailR, 0, &keep_all(0, &[-8])), 16);
        // LSB 12 is just before 16, not after it.
        assert_eq!(poc(between, tid, &keep_all(12, &[-4])), 12, "{between:?}");
        // Still predicted from 16 (LSB 0): LSB 8 is 24. Predicting from 12
        // instead would give 8.
        assert_eq!(
            poc(NalType::TrailR, 0, &keep_all(8, &[-8])),
            24,
            "{between:?}"
        );
    }
}

/// A picture with POC below the last TemporalId 0 picture's, across the
/// wrap, gets a negative MSB step.
#[test]
fn poc_steps_back_across_the_wrap() {
    let sps = sps(1);
    let mut dpb = DecodeDpb::new(6, Arc::default());
    decode(&mut dpb, nal(NalType::IdrWRadl, 0), &idr(), &sps);
    // A RADL picture with LSB 15 right after an IDR is POC -1.
    let (state, _) = decode(
        &mut dpb,
        nal(NalType::RadlN, 0),
        &slice(15, &[], &[(1, true)]),
        &sps,
    );
    assert_eq!(state.poc, -1);
    assert!(!state.missing_reference);
}

/// The set names what stays: pictures left out are dropped before the current
/// picture is decoded, and their slots come free for it.
#[test]
fn rps_keeps_what_it_names_and_maps_slots() {
    let sps = sps(1);
    let mut dpb = DecodeDpb::new(3, Arc::default());
    let (_, slot0) = decode(&mut dpb, nal(NalType::IdrNLp, 0), &idr(), &sps);
    let (s1, slot1) = decode(
        &mut dpb,
        nal(NalType::TrailR, 0),
        &slice(1, &[(-1, true)], &[]),
        &sps,
    );
    assert_eq!(
        s1.rps.st_curr_before,
        vec![RpsEntry {
            poc: 0,
            slot: Some(slot0)
        }]
    );

    // POC 2 uses 1 and keeps 0 for later.
    let (s2, slot2) = decode(
        &mut dpb,
        nal(NalType::TrailR, 0),
        &slice(2, &[(-1, true), (-2, false)], &[]),
        &sps,
    );
    assert_eq!(
        s2.rps.st_curr_before,
        vec![RpsEntry {
            poc: 1,
            slot: Some(slot1)
        }]
    );
    assert_eq!(
        s2.rps.st_foll,
        vec![RpsEntry {
            poc: 0,
            slot: Some(slot0)
        }]
    );
    assert_eq!(pocs(&dpb), vec![(0, false), (1, false), (2, false)]);

    // All three slots are taken. POC 3 keeps only 2, so 0 and 1 go before it
    // needs a slot, and it gets one of theirs.
    let (s3, slot3) = decode(
        &mut dpb,
        nal(NalType::TrailR, 0),
        &slice(3, &[(-1, true)], &[]),
        &sps,
    );
    assert_eq!(
        s3.rps.st_curr_before,
        vec![RpsEntry {
            poc: 2,
            slot: Some(slot2)
        }]
    );
    assert!(slot3 == slot0 || slot3 == slot1);
    assert_eq!(pocs(&dpb), vec![(2, false), (3, false)]);

    // A B picture between them: StCurrAfter.
    let (b, _) = decode(
        &mut dpb,
        nal(NalType::TrailN, 0),
        &slice(5, &[(-2, true)], &[(1, false)]),
        &sps,
    );
    assert_eq!(b.rps.st_curr_before[0].poc, 3);
    assert_eq!(b.rps.st_foll[0].poc, 6);
    assert!(b.rps.st_foll[0].slot.is_none(), "POC 6 is not decoded yet");
    assert!(!b.missing_reference, "a missing Foll entry is harmless");
}

/// A picture predicting from something the DPB does not hold says so.
#[test]
fn missing_reference_is_reported() {
    let sps = sps(1);
    let mut dpb = DecodeDpb::new(4, Arc::default());
    decode(&mut dpb, nal(NalType::IdrNLp, 0), &idr(), &sps);
    let (state, _) = decode(
        &mut dpb,
        nal(NalType::TrailR, 0),
        &slice(4, &[(-2, true), (-4, true)], &[]),
        &sps,
    );
    assert!(state.missing_reference);
    assert_eq!(state.rps.st_curr_before[0], RpsEntry { poc: 2, slot: None });
    assert_eq!(state.rps.st_curr_before[1].poc, 0);
    assert!(state.rps.st_curr_before[1].slot.is_some());
}

/// Long-term entries pick a picture by LSB (or by full POC with the MSB
/// present), mark it long-term, take it out of reach of the short-term lists,
/// and drop it once no set names it.
#[test]
fn long_term_pictures() {
    let sps = sps(1);
    let mut dpb = DecodeDpb::new(6, Arc::default());
    let lt = |poc_lsb_lt, used, msb: Option<u32>| LongTermRef {
        poc_lsb_lt,
        used_by_curr_pic_lt: used,
        delta_poc_msb_present_flag: msb.is_some(),
        delta_poc_msb_cycle_lt: msb.unwrap_or(0),
        delta_poc_msb_cycle_lt_accumulated: msb.unwrap_or(0),
        lt_idx_sps: None,
    };
    let with_lt = |lsb, s0: &[(i32, bool)], lts: Vec<LongTermRef>| SliceHeader {
        long_term_refs: lts,
        ..slice(lsb, s0, &[])
    };

    let (_, slot0) = decode(&mut dpb, nal(NalType::IdrNLp, 0), &idr(), &sps);
    decode(
        &mut dpb,
        nal(NalType::TrailR, 0),
        &slice(1, &[(-1, true)], &[]),
        &sps,
    );
    let (s2, _) = decode(
        &mut dpb,
        nal(NalType::TrailR, 0),
        &with_lt(2, &[(-1, true)], vec![lt(0, true, None)]),
        &sps,
    );
    assert_eq!(
        s2.rps.lt_curr,
        vec![RpsEntry {
            poc: 0,
            slot: Some(slot0)
        }]
    );
    assert_eq!(pocs(&dpb), vec![(0, true), (1, false), (2, false)]);

    // Kept for later only; and by full POC with the MSB (cycle 0 is this
    // picture's MSB, so POC 0 itself).
    let (s3, _) = decode(
        &mut dpb,
        nal(NalType::TrailR, 0),
        &with_lt(3, &[(-1, true)], vec![lt(0, false, Some(0))]),
        &sps,
    );
    assert_eq!(
        s3.rps.lt_foll,
        vec![RpsEntry {
            poc: 0,
            slot: Some(slot0)
        }]
    );
    assert!(s3.rps.lt_curr.is_empty());

    // A long-term picture is not a short-term candidate any more.
    let (s4, _) = decode(
        &mut dpb,
        nal(NalType::TrailR, 0),
        &with_lt(4, &[(-1, true), (-4, true)], vec![lt(0, false, None)]),
        &sps,
    );
    assert_eq!(s4.rps.st_curr_before[1], RpsEntry { poc: 0, slot: None });
    assert!(s4.missing_reference);

    // Named by nothing: gone, slot free.
    decode(
        &mut dpb,
        nal(NalType::TrailR, 0),
        &slice(5, &[(-1, true)], &[]),
        &sps,
    );
    assert!(dpb.references().iter().all(|r| r.poc != 0 && !r.long_term));
}

/// The MSB cycle reaches a long-term picture a whole LSB period back.
#[test]
fn long_term_msb_cycle() {
    let sps = sps(1);
    let mut dpb = DecodeDpb::new(6, Arc::default());
    let (_, slot0) = decode(&mut dpb, nal(NalType::IdrNLp, 0), &idr(), &sps);
    let keep0 = |lsb| SliceHeader {
        long_term_refs: vec![LongTermRef {
            poc_lsb_lt: 0,
            used_by_curr_pic_lt: true,
            delta_poc_msb_present_flag: true,
            delta_poc_msb_cycle_lt: 1,
            delta_poc_msb_cycle_lt_accumulated: 1,
            lt_idx_sps: None,
        }],
        ..slice(lsb, &[], &[])
    };
    let lt_only = |lsb| SliceHeader {
        long_term_refs: vec![LongTermRef {
            poc_lsb_lt: 0,
            used_by_curr_pic_lt: false,
            delta_poc_msb_present_flag: false,
            ..Default::default()
        }],
        ..slice(lsb, &[], &[])
    };
    // Walk up to POC 16 keeping POC 0 long-term by LSB.
    for lsb in [4, 8, 12] {
        decode(&mut dpb, nal(NalType::TrailR, 0), &lt_only(lsb), &sps);
    }
    // POC 16 has LSB 0, as POC 0 does; MSB cycle 1 makes it unambiguous.
    let (state, _) = decode(&mut dpb, nal(NalType::TrailR, 0), &keep0(0), &sps);
    assert_eq!(state.poc, 16);
    assert_eq!(
        state.rps.lt_curr,
        vec![RpsEntry {
            poc: 0,
            slot: Some(slot0)
        }]
    );
}

/// A stream starting at a CRA: RASL pictures are skipped, RADL ones decode.
#[test]
fn cra_first_skips_rasl() {
    let sps = sps(1);
    let mut dpb = DecodeDpb::new(6, Arc::default());
    assert_eq!(
        skip_reason(&mut dpb, nal(NalType::TrailR, 0), &slice(3, &[], &[]), &sps),
        SkipReason::AwaitingIrap
    );
    // The CRA's own set keeps an earlier picture for its RASL pictures.
    let (cra, cra_slot) = decode(
        &mut dpb,
        nal(NalType::Cra, 0),
        &slice(8, &[(-2, false)], &[]),
        &sps,
    );
    assert_eq!(cra.poc, 8, "the MSB starts at 0");
    assert!(cra.is_keyframe && cra.no_rasl_output_flag);
    assert!(cra.no_output_of_prior_pics, "inferred for a CRA");
    assert!(!cra.missing_reference);
    assert_eq!(
        skip_reason(
            &mut dpb,
            nal(NalType::RaslN, 0),
            &slice(6, &[(-2, true)], &[(2, true)]),
            &sps
        ),
        SkipReason::RaslWithoutReferences
    );
    let (radl, _) = decode(
        &mut dpb,
        nal(NalType::RadlN, 0),
        &slice(7, &[], &[(1, true)]),
        &sps,
    );
    assert_eq!(radl.poc, 7);
    assert_eq!(
        radl.rps.st_curr_after,
        vec![RpsEntry {
            poc: 8,
            slot: Some(cra_slot)
        }]
    );
    let (trail, _) = decode(
        &mut dpb,
        nal(NalType::TrailR, 0),
        &slice(9, &[(-1, true)], &[]),
        &sps,
    );
    assert!(!trail.is_keyframe && !trail.missing_reference);
}

/// A CRA mid-stream is an ordinary picture, and its RASL pictures decode.
#[test]
fn cra_mid_stream_keeps_rasl() {
    let sps = sps(1);
    let mut dpb = DecodeDpb::new(6, Arc::default());
    decode(&mut dpb, nal(NalType::IdrNLp, 0), &idr(), &sps);
    decode(
        &mut dpb,
        nal(NalType::TrailR, 0),
        &slice(2, &[(-2, true)], &[]),
        &sps,
    );
    let (cra, _) = decode(
        &mut dpb,
        nal(NalType::Cra, 0),
        &slice(4, &[(-2, false)], &[]),
        &sps,
    );
    assert!(!cra.is_keyframe && !cra.no_rasl_output_flag);
    let (rasl, _) = decode(
        &mut dpb,
        nal(NalType::RaslN, 0),
        &slice(3, &[(-1, true)], &[(1, true)]),
        &sps,
    );
    assert_eq!(rasl.poc, 3);
    assert!(!rasl.missing_reference);
}

/// After an end of sequence the next picture must be an IRAP, which starts
/// over: MSB 0, nothing kept, RASL pictures skipped.
#[test]
fn end_of_sequence_restarts() {
    let sps = sps(1);
    let mut dpb = DecodeDpb::new(6, Arc::default());
    decode(&mut dpb, nal(NalType::IdrNLp, 0), &idr(), &sps);
    for lsb in [6u32, 12, 2] {
        decode(
            &mut dpb,
            nal(NalType::TrailR, 0),
            &slice(lsb, &[(-6, true)], &[]),
            &sps,
        );
    }
    assert_eq!(pocs(&dpb).last(), Some(&(18, false)));

    dpb.end_of_sequence();
    assert_eq!(
        skip_reason(
            &mut dpb,
            nal(NalType::TrailR, 0),
            &slice(4, &[(-1, true)], &[]),
            &sps
        ),
        SkipReason::AwaitingIrap
    );
    let (cra, _) = decode(
        &mut dpb,
        nal(NalType::Cra, 0),
        &slice(4, &[(-2, false)], &[]),
        &sps,
    );
    assert_eq!(cra.poc, 4, "not 20: the MSB restarts");
    assert!(cra.is_keyframe);
    assert_eq!(pocs(&dpb), vec![(4, false)]);
    assert_eq!(
        skip_reason(
            &mut dpb,
            nal(NalType::RaslR, 0),
            &slice(3, &[(-1, true)], &[(1, true)]),
            &sps
        ),
        SkipReason::RaslWithoutReferences
    );
}

/// A BLA picture starts a new sequence wherever it is.
#[test]
fn bla_starts_a_new_sequence() {
    let sps = sps(1);
    let mut dpb = DecodeDpb::new(6, Arc::default());
    decode(&mut dpb, nal(NalType::IdrNLp, 0), &idr(), &sps);
    decode(
        &mut dpb,
        nal(NalType::TrailR, 0),
        &slice(5, &[(-5, true)], &[]),
        &sps,
    );
    let mut header = slice(9, &[(-4, false)], &[]);
    header.no_output_of_prior_pics_flag = false;
    let (bla, _) = decode(&mut dpb, nal(NalType::BlaWLp, 0), &header, &sps);
    assert_eq!(bla.poc, 9);
    assert!(bla.is_keyframe);
    assert!(
        !bla.no_output_of_prior_pics,
        "taken from the flag for a BLA"
    );
    assert_eq!(bla.rps.st_foll, vec![RpsEntry { poc: 5, slot: None }]);
    assert_eq!(pocs(&dpb), vec![(9, false)]);
    assert_eq!(
        skip_reason(
            &mut dpb,
            nal(NalType::RaslN, 0),
            &slice(8, &[], &[(1, true)]),
            &sps
        ),
        SkipReason::RaslWithoutReferences
    );
}

/// An IDR clears the DPB; its own reference picture set is empty.
#[test]
fn idr_clears_everything() {
    let sps = sps(1);
    let mut dpb = DecodeDpb::new(3, Arc::default());
    decode(&mut dpb, nal(NalType::IdrNLp, 0), &idr(), &sps);
    decode(
        &mut dpb,
        nal(NalType::TrailR, 0),
        &slice(1, &[(-1, true)], &[]),
        &sps,
    );
    let mut header = idr();
    header.no_output_of_prior_pics_flag = true;
    let (state, slot) = decode(&mut dpb, nal(NalType::IdrWRadl, 0), &header, &sps);
    assert_eq!(state.poc, 0);
    assert_eq!(state.rps, RefPicSet::default());
    assert!(state.no_output_of_prior_pics);
    assert_eq!(slot, 0, "every slot was freed");
    assert_eq!(pocs(&dpb), vec![(0, false)]);
}

/// A sub-layer non-reference picture of the highest sub-layer is never
/// referenced, so its slot is released at once; in a lower sub-layer it is
/// kept, since higher sub-layers may predict from it.
#[test]
fn sub_layer_non_reference_pictures() {
    let one_layer = sps(1);
    let mut dpb = DecodeDpb::new(4, Arc::default());
    decode(&mut dpb, nal(NalType::IdrNLp, 0), &idr(), &one_layer);
    let (state, slot) = decode(
        &mut dpb,
        nal(NalType::TrailN, 0),
        &slice(1, &[(-1, true)], &[]),
        &one_layer,
    );
    assert!(!state.is_reference);
    assert_eq!(dpb.try_allocate_slot(), Some(slot), "released");

    let two_layers = sps(2);
    let mut dpb = DecodeDpb::new(4, Arc::default());
    decode(&mut dpb, nal(NalType::IdrNLp, 0), &idr(), &two_layers);
    let (state, _) = decode(
        &mut dpb,
        nal(NalType::TrailN, 0),
        &slice(2, &[(-2, true)], &[]),
        &two_layers,
    );
    assert!(state.is_reference);
    let (state, _) = decode(
        &mut dpb,
        nal(NalType::TsaN, 1),
        &slice(1, &[(-1, true)], &[(1, true)]),
        &two_layers,
    );
    assert!(!state.is_reference);
    assert!(!state.missing_reference);
}

/// pic_output_flag passes through.
#[test]
fn pic_output_flag_passes_through() {
    let sps = sps(1);
    let mut dpb = DecodeDpb::new(4, Arc::default());
    decode(&mut dpb, nal(NalType::IdrNLp, 0), &idr(), &sps);
    let mut header = slice(1, &[(-1, true)], &[]);
    header.pic_output_flag = false;
    let (state, _) = decode(&mut dpb, nal(NalType::TrailR, 0), &header, &sps);
    assert!(!state.pic_output_flag);
}

#[test]
fn vulkan_lists_are_padded() {
    let list = [
        RpsEntry {
            poc: 1,
            slot: Some(3),
        },
        RpsEntry { poc: 2, slot: None },
    ];
    let out = RefPicSet::vulkan_list(&list).unwrap();
    assert_eq!(
        out,
        [3, NO_REFERENCE_PICTURE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
    );
    let too_long = [RpsEntry {
        poc: 0,
        slot: Some(0),
    }; 9];
    assert!(RefPicSet::vulkan_list(&too_long).is_err());
}

/// A slot a handed-out frame is reading must not be decoded over, even once
/// the reference picture set has dropped it.
#[test]
fn pinned_slots_are_not_reallocated() {
    let sps = sps(1);
    let pins = Arc::new(SlotPins::default());
    let mut dpb = DecodeDpb::new(2, pins.clone());
    let (_, idr_slot) = decode(&mut dpb, nal(NalType::IdrNLp, 0), &idr(), &sps);
    pins.pin(idr_slot);
    assert!(dpb.has_pinned_slots());

    // POC 1 drops the IDR, but the caller still holds it.
    let (_, slot1) = decode(&mut dpb, nal(NalType::TrailR, 0), &slice(1, &[], &[]), &sps);
    assert_ne!(slot1, idr_slot);
    let state = match dpb
        .begin_picture(
            &nal(NalType::TrailR, 0),
            &slice(2, &[(-1, true)], &[]),
            &sps,
        )
        .unwrap()
    {
        PictureDecision::Decode(state) => state,
        PictureDecision::Skip(_) => unreachable!(),
    };
    assert_eq!(dpb.try_allocate_slot(), None, "one reference, one pinned");

    pins.release(idr_slot);
    assert!(!dpb.has_pinned_slots());
    let slot = dpb.try_allocate_slot().unwrap();
    assert_eq!(slot, idr_slot);
    dpb.end_picture(slot, &state);
}

/// Malformed input is an error, not a panic.
#[test]
fn rejects_bad_input() {
    let sps = sps(1);
    let mut dpb = DecodeDpb::new(4, Arc::default());
    decode(&mut dpb, nal(NalType::IdrNLp, 0), &idr(), &sps);
    // A non-IDR picture without any short-term set.
    let header = SliceHeader {
        slice_pic_order_cnt_lsb: 1,
        first_slice_segment_in_pic_flag: true,
        ..Default::default()
    };
    assert!(
        dpb.begin_picture(&nal(NalType::TrailR, 0), &header, &sps)
            .is_err()
    );
    // Not a slice.
    assert!(
        dpb.begin_picture(&nal(NalType::Sps, 0), &idr(), &sps)
            .is_err()
    );
}
