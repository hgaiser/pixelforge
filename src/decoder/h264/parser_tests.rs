//! Parser tests against real x264-generated streams.
//!
//! The fixtures in `tests/data` are 320x240, 30-frame streams produced by
//! x264. Between them they cover the Baseline and High profiles, POC types 0
//! and 2, and streams with and without B-frames.
//!
//! `bpyramid` (`bframes=3:b-pyramid=normal:weightb=1:direct=temporal`) and
//! `bstrict` (`bframes=3:b-pyramid=strict`) come from
//! `ffmpeg -c:v libx264 -crf 40 -x264-params b-adapt=0:scenecut=0:<params>` on
//! testsrc2. `novui` and `novui_poc2` are `bframes` and `zerolatency` with the
//! VUI removed from the SPS by `tests/data/strip_vui.py`, so they carry no
//! `max_num_reorder_frames`.
//!
//! These tests exercise the host-side parsing and DPB bookkeeping only; they
//! need no GPU and so run everywhere.

use super::parser::{
    Mmco, NalType, ReorderSource, iter_nal_units, parse_pps, parse_slice_header, parse_sps,
};

const BASELINE: &[u8] = include_bytes!("../../../tests/data/base.264");
const ZEROLATENCY: &[u8] = include_bytes!("../../../tests/data/zerolatency.264");
const BFRAMES: &[u8] = include_bytes!("../../../tests/data/bframes.264");
/// 4 slices per picture: distinguishes "split per slice" from "split per picture".
const MULTISLICE: &[u8] = include_bytes!("../../../tests/data/multislice.264");
const BPYRAMID: &[u8] = include_bytes!("../../../tests/data/bpyramid.264");
const BSTRICT: &[u8] = include_bytes!("../../../tests/data/bstrict.264");
const NOVUI: &[u8] = include_bytes!("../../../tests/data/novui.264");
const NOVUI_POC2: &[u8] = include_bytes!("../../../tests/data/novui_poc2.264");

/// Every fixture: one SPS, one PPS, 30 slices, first slice is an IDR.
#[test]
fn parses_stream_structure() {
    for (name, data) in [
        ("baseline", BASELINE),
        ("zerolatency", ZEROLATENCY),
        ("bframes", BFRAMES),
    ] {
        let nals: Vec<_> = iter_nal_units(data).collect();
        let sps_count = nals.iter().filter(|n| n.nal_type == NalType::Sps).count();
        let pps_count = nals.iter().filter(|n| n.nal_type == NalType::Pps).count();
        let slices: Vec<_> = nals.iter().filter(|n| n.nal_type.is_slice()).collect();

        assert_eq!(sps_count, 1, "{name}: expected one SPS");
        assert_eq!(pps_count, 1, "{name}: expected one PPS");
        assert_eq!(slices.len(), 30, "{name}: expected 30 slices");
        assert_eq!(
            slices[0].nal_type,
            NalType::IdrSlice,
            "{name}: stream must open with an IDR"
        );
        // An IDR is always a reference picture.
        assert_ne!(slices[0].ref_idc, 0, "{name}: IDR must have ref_idc != 0");
    }
}

/// The SPS decodes to the dimensions and profile x264 was asked for.
#[test]
fn parses_sps_geometry_and_profile() {
    for (name, data, expected_profile) in [
        ("baseline", BASELINE, 66u8),
        ("zerolatency", ZEROLATENCY, 100),
        ("bframes", BFRAMES, 100),
    ] {
        let sps_nal = iter_nal_units(data)
            .find(|n| n.nal_type == NalType::Sps)
            .expect("stream has an SPS");
        let sps = parse_sps(sps_nal.payload()).expect("SPS parses");

        assert_eq!(sps.profile_idc, expected_profile, "{name}: profile_idc");
        assert_eq!(sps.sps_id, 0, "{name}: sps_id");
        assert_eq!(sps.chroma_format_idc, 1, "{name}: 4:2:0");
        assert_eq!(sps.bit_depth_luma_minus8, 0, "{name}: 8-bit luma");
        assert!(sps.frame_mbs_only_flag, "{name}: progressive");
        // 320x240 is exactly 20x15 macroblocks, so no cropping is needed.
        assert_eq!(sps.coded_width(), 320, "{name}: coded width");
        assert_eq!(sps.coded_height(), 240, "{name}: coded height");
        assert_eq!(sps.display_dimensions(), (320, 240), "{name}: display size");
    }
}

/// A stream whose dimensions aren't macroblock-aligned must report cropping.
#[test]
fn parses_cropped_dimensions() {
    // The fixtures are all MB-aligned, so verify the crop arithmetic against a
    // synthetic SPS: 4:2:0 crop offsets are in chroma units (2 luma samples).
    let sps_nal = iter_nal_units(BFRAMES)
        .find(|n| n.nal_type == NalType::Sps)
        .unwrap();
    let mut sps = parse_sps(sps_nal.payload()).unwrap();
    assert!(!sps.frame_cropping_flag);

    // Crop 1920x1080-style: 68 rows of padding -> 240 - 8 = 232.
    sps.frame_cropping_flag = true;
    sps.frame_crop_bottom_offset = 4; // 4 * 2 = 8 luma rows
    sps.frame_crop_right_offset = 2; // 2 * 2 = 4 luma columns
    assert_eq!(sps.display_dimensions(), (316, 232));
    assert_eq!(sps.coded_width(), 320, "cropping must not alter coded size");
}

/// The PPS parses, and its entropy mode matches the profile x264 used.
#[test]
fn parses_pps() {
    for (name, data, expect_cabac) in [
        // Baseline profile cannot use CABAC.
        ("baseline", BASELINE, false),
        ("zerolatency", ZEROLATENCY, true),
        ("bframes", BFRAMES, true),
    ] {
        let sps_nal = iter_nal_units(data)
            .find(|n| n.nal_type == NalType::Sps)
            .unwrap();
        let sps = parse_sps(sps_nal.payload()).unwrap();
        let pps_nal = iter_nal_units(data)
            .find(|n| n.nal_type == NalType::Pps)
            .unwrap();
        let pps =
            parse_pps(pps_nal.payload(), |_| Some(sps.chroma_format_idc)).expect("PPS parses");

        assert_eq!(pps.pps_id, 0, "{name}: pps_id");
        assert_eq!(pps.sps_id, 0, "{name}: references sps 0");
        assert_eq!(
            pps.entropy_coding_mode_flag, expect_cabac,
            "{name}: entropy coding mode"
        );
        assert_eq!(pps.num_slice_groups_minus1, 0, "{name}: no FMO");
    }
}

/// Slice headers parse across the whole stream, and frame_num advances the way
/// the reference structure implies.
#[test]
fn parses_all_slice_headers() {
    for (name, data) in [
        ("baseline", BASELINE),
        ("zerolatency", ZEROLATENCY),
        ("bframes", BFRAMES),
    ] {
        let sps_nal = iter_nal_units(data)
            .find(|n| n.nal_type == NalType::Sps)
            .unwrap();
        let sps = parse_sps(sps_nal.payload()).unwrap();
        let pps_nal = iter_nal_units(data)
            .find(|n| n.nal_type == NalType::Pps)
            .unwrap();
        let pps = parse_pps(pps_nal.payload(), |_| Some(sps.chroma_format_idc)).unwrap();

        let mut headers = Vec::new();
        for nal in iter_nal_units(data).filter(|n| n.nal_type.is_slice()) {
            let header = parse_slice_header(&nal, &sps, &pps)
                .unwrap_or_else(|e| panic!("{name}: slice header must parse: {e}"));
            assert_eq!(header.pps_id, 0, "{name}: slice references pps 0");
            assert!(!header.field_pic_flag, "{name}: progressive stream");
            assert!(
                header.frame_num < sps.max_frame_num(),
                "{name}: frame_num must fit in log2_max_frame_num"
            );
            headers.push((header, nal.ref_idc));
        }

        assert_eq!(headers.len(), 30, "{name}: all slices parsed");
        // The IDR resets frame_num to 0.
        assert_eq!(headers[0].0.frame_num, 0, "{name}: IDR has frame_num 0");

        // frame_num only advances after a *reference* picture. Verify that
        // invariant holds across the entire stream.
        for pair in headers.windows(2) {
            let (prev, prev_ref_idc) = &pair[0];
            let (next, _) = &pair[1];
            let expected = if *prev_ref_idc != 0 {
                (prev.frame_num + 1) % sps.max_frame_num()
            } else {
                prev.frame_num
            };
            assert_eq!(
                next.frame_num, expected,
                "{name}: frame_num must advance only after reference pictures"
            );
        }
    }
}

/// x264's B-pyramid retires its B-references with explicit MMCO commands rather
/// than leaving them to the sliding window. Parsing `dec_ref_pic_marking()` is
/// therefore not optional: ignoring it silently corrupts the DPB, and the
/// decoded pictures with it.
///
/// This pins both halves: the B-pyramid fixtures really do carry MMCO, and the
/// streams without B-references really do not.
#[test]
fn parses_explicit_reference_marking() {
    fn markings(data: &[u8]) -> Vec<(bool, Vec<Mmco>)> {
        let sps_nal = iter_nal_units(data)
            .find(|n| n.nal_type == NalType::Sps)
            .unwrap();
        let sps = parse_sps(sps_nal.payload()).unwrap();
        let pps_nal = iter_nal_units(data)
            .find(|n| n.nal_type == NalType::Pps)
            .unwrap();
        let pps = parse_pps(pps_nal.payload(), |_| Some(sps.chroma_format_idc)).unwrap();

        iter_nal_units(data)
            .filter(|n| n.nal_type.is_slice())
            .map(|nal| {
                let h = parse_slice_header(&nal, &sps, &pps).unwrap();
                (h.marking.adaptive, h.marking.ops.clone())
            })
            .collect()
    }

    // B-pyramid: at least one picture marks a short-term reference unused.
    for (name, data) in [("bframes", BFRAMES), ("multislice", MULTISLICE)] {
        let markings = markings(data);
        assert!(
            markings.iter().any(|(adaptive, _)| *adaptive),
            "{name}: B-pyramid stream must use adaptive reference marking"
        );
        assert!(
            markings
                .iter()
                .flat_map(|(_, ops)| ops)
                .any(|op| matches!(op, Mmco::ForgetShort { .. })),
            "{name}: expected an MMCO 1 retiring a short-term reference"
        );
    }

    // No B-references: pure sliding window, so no MMCO anywhere.
    for (name, data) in [("baseline", BASELINE), ("zerolatency", ZEROLATENCY)] {
        for (adaptive, ops) in markings(data) {
            assert!(
                !adaptive && ops.is_empty(),
                "{name}: expected sliding-window marking only"
            );
        }
    }
}

/// The B-frame stream really does contain disposable (non-reference) pictures,
/// and the zerolatency one does not. This is what makes the two fixtures
/// distinct tests rather than duplicates.
#[test]
fn fixtures_differ_in_reference_structure() {
    let disposable = |data: &[u8]| {
        iter_nal_units(data)
            .filter(|n| n.nal_type.is_slice())
            .filter(|n| n.ref_idc == 0)
            .count()
    };

    assert!(
        disposable(BFRAMES) > 0,
        "the bframes fixture should contain non-reference B-frames"
    );
    assert_eq!(
        disposable(ZEROLATENCY),
        0,
        "a zerolatency stream should have every picture as a reference"
    );
}

/// POC type 0 streams carry an explicit pic_order_cnt_lsb; POC type 2 streams
/// derive POC from decode order and carry none.
#[test]
fn poc_type_matches_stream_structure() {
    let sps_of = |data: &[u8]| {
        let nal = iter_nal_units(data)
            .find(|n| n.nal_type == NalType::Sps)
            .unwrap();
        parse_sps(nal.payload()).unwrap()
    };

    // These exact values are cross-checked against ffprobe, which reports
    // has_b_frames=0 for the first two fixtures and 2 for the third.
    //
    // x264 picks POC type 2 when no reordering is needed (POC == decode
    // order), and type 0 when B-frames force display order to diverge. The two
    // cases exercise both POC paths in the decoder's DPB.
    assert_eq!(
        sps_of(BASELINE).pic_order_cnt_type,
        2,
        "a stream without B-frames should use POC type 2"
    );
    assert_eq!(
        sps_of(ZEROLATENCY).pic_order_cnt_type,
        2,
        "a zerolatency stream should use POC type 2"
    );
    assert_eq!(
        sps_of(BFRAMES).pic_order_cnt_type,
        0,
        "a stream with B-frames must carry explicit POC (type 0)"
    );
}

/// Level and reference-frame counts match what ffprobe reports for these files.
#[test]
fn parses_level_and_ref_frames() {
    let sps_of = |data: &[u8]| {
        let nal = iter_nal_units(data)
            .find(|n| n.nal_type == NalType::Sps)
            .unwrap();
        parse_sps(nal.payload()).unwrap()
    };

    for (name, data, expected_refs) in [
        ("baseline", BASELINE, 3u8),
        ("zerolatency", ZEROLATENCY, 3),
        ("bframes", BFRAMES, 4),
    ] {
        let sps = sps_of(data);
        // ffprobe reports level=13 (i.e. level 1.3) for all three fixtures.
        assert_eq!(sps.level_idc, 13, "{name}: level_idc");
        assert_eq!(sps.max_num_ref_frames, expected_refs, "{name}: ref frames");
        // The decoder sizes its DPB from this; it must fit the spec maximum.
        assert!(sps.max_num_ref_frames <= 16, "{name}: ref frames in range");
    }
}

/// Reassemble the coded pictures of a whole stream from their start offsets,
/// which is what `picture_starts` exists to provide.
fn split_stream(stream: &[u8]) -> Vec<&[u8]> {
    let starts = super::picture_starts(stream);
    let ends = starts
        .iter()
        .skip(1)
        .copied()
        .chain(std::iter::once(stream.len()));
    starts
        .iter()
        .copied()
        .zip(ends)
        .map(|(start, end)| &stream[start..end])
        .collect()
}

/// Feeding a stream in fixed-size chunks, the way a socket delivers it, must
/// find exactly the same picture boundaries as having the whole buffer at once.
///
/// This drives the same carry-over rule `decode_byte_stream` uses: append the
/// chunk, decode everything up to `complete_prefix`, hold the rest. Chunk sizes
/// are chosen to cut inside start codes, inside slice headers and across whole
/// pictures.
#[test]
fn byte_stream_framing_finds_the_same_pictures() {
    for (name, data) in [
        ("baseline", BASELINE),
        ("zerolatency", ZEROLATENCY),
        ("bframes", BFRAMES),
        ("multislice", MULTISLICE),
    ] {
        let expected: Vec<&[u8]> = split_stream(data);

        for chunk_size in [1usize, 2, 3, 7, 64, 1024, 100_000] {
            let mut carry: Vec<u8> = Vec::new();
            let mut emitted: Vec<u8> = Vec::new();
            let mut units = 0usize;

            for chunk in data.chunks(chunk_size) {
                carry.extend_from_slice(chunk);
                let split = super::complete_prefix(&carry);
                if split > 0 {
                    units += super::picture_starts(&carry[..split]).len();
                    emitted.extend_from_slice(&carry[..split]);
                    carry.drain(..split);
                }
            }
            // End of stream ends the final picture, which is what `flush` does.
            if !carry.is_empty() {
                units += super::picture_starts(&carry).len();
                emitted.extend_from_slice(&carry);
            }

            assert_eq!(
                units,
                expected.len(),
                "{name}: chunk size {chunk_size} found {units} pictures, expected {}",
                expected.len()
            );
            // Nothing may be dropped or duplicated on the way through.
            assert_eq!(
                emitted, data,
                "{name}: chunk size {chunk_size} did not reproduce the stream"
            );
        }
    }
}

/// `split_stream` must split a stream into exactly one unit per coded picture,
/// losing no bytes and keeping parameter sets with the picture they configure.
#[test]
fn splits_stream_into_coded_frames() {
    for (name, data) in [
        ("baseline", BASELINE),
        ("zerolatency", ZEROLATENCY),
        ("bframes", BFRAMES),
        ("multislice", MULTISLICE),
    ] {
        let units = split_stream(data);

        // Each fixture is 30 coded pictures.
        assert_eq!(units.len(), 30, "{name}: one access unit per picture");

        // Every unit holds exactly one picture-start slice.
        for (i, unit) in units.iter().enumerate() {
            let picture_starts = iter_nal_units(unit)
                .filter(|n| n.nal_type.is_slice())
                .filter(|n| n.payload().first().is_some_and(|b| b & 0x80 != 0))
                .count();
            assert_eq!(
                picture_starts, 1,
                "{name}: unit {i} must contain exactly one picture"
            );
        }

        // The split must be lossless and contiguous: concatenating the units
        // reproduces the stream from the first unit's start onward.
        let total: usize = units.iter().map(|u| u.len()).sum();
        let first_offset = data.len() - total;
        let rejoined: Vec<u8> = units.concat();
        assert_eq!(
            rejoined,
            &data[first_offset..],
            "{name}: units must tile the stream without gaps or overlap"
        );

        // The SPS and PPS must travel with the first picture, or the decoder
        // could not configure a session before the IDR.
        let first = units[0];
        assert!(
            iter_nal_units(first).any(|n| n.nal_type == NalType::Sps),
            "{name}: first access unit must carry the SPS"
        );
        assert!(
            iter_nal_units(first).any(|n| n.nal_type == NalType::Pps),
            "{name}: first access unit must carry the PPS"
        );
        assert!(
            iter_nal_units(first).any(|n| n.nal_type == NalType::IdrSlice),
            "{name}: first access unit must be the IDR"
        );
    }
}

/// A picture split across several slice NALs must still form a single access
/// unit. Without this fixture, splitting per slice and splitting per picture
/// are indistinguishable, since x264 emits one slice per picture by default.
#[test]
fn multislice_pictures_form_one_access_unit() {
    let slice_nals = iter_nal_units(MULTISLICE)
        .filter(|n| n.nal_type.is_slice())
        .count();
    let units = split_stream(MULTISLICE);

    // The fixture really is multi-slice: 4 slices per picture, 30 pictures.
    assert_eq!(slice_nals, 120, "fixture should hold 120 slice NALs");
    assert_eq!(units.len(), 30, "but only 30 access units");

    // Every unit carries all 4 slices of its picture.
    for (i, unit) in units.iter().enumerate() {
        let slices = iter_nal_units(unit)
            .filter(|n| n.nal_type.is_slice())
            .count();
        assert_eq!(slices, 4, "access unit {i} should hold all 4 slices");
    }
}

#[test]
fn vui_reorder_matches_streams() {
    // (stream, expected max_num_reorder_frames present and value)
    for (name, data, expect) in [
        ("baseline", BASELINE, Some(0u32)),
        ("zerolatency", ZEROLATENCY, Some(0)),
        ("bframes", BFRAMES, Some(2)),
        ("multislice", MULTISLICE, Some(2)),
    ] {
        let sps_nal = iter_nal_units(data)
            .find(|n| n.nal_type == NalType::Sps)
            .unwrap();
        let sps = parse_sps(sps_nal.payload()).unwrap();
        // Only assert when the stream signals it; if None, skip (fallback path).
        if let Some(v) = expect {
            assert_eq!(sps.max_num_reorder_frames, Some(v), "{name}: reorder depth");
        }
    }
}

/// Without a VUI the reorder depth is inferred: none where the stream cannot
/// reorder, and the level's DPB size otherwise.
#[test]
fn reorder_depth_without_vui() {
    let sps_of = |data: &[u8]| {
        let nal = iter_nal_units(data)
            .find(|n| n.nal_type == NalType::Sps)
            .unwrap();
        parse_sps(nal.payload()).unwrap()
    };

    for (name, data, expect) in [
        ("bframes", BFRAMES, (2, ReorderSource::Vui)),
        ("bpyramid", BPYRAMID, (2, ReorderSource::Vui)),
        ("bstrict", BSTRICT, (2, ReorderSource::Vui)),
        ("zerolatency", ZEROLATENCY, (0, ReorderSource::Vui)),
        // Level 1.3 at 320x240: MaxDpbMbs 2376 over 300 macroblocks.
        ("novui", NOVUI, (7, ReorderSource::Level)),
        ("novui_poc2", NOVUI_POC2, (0, ReorderSource::PocType)),
    ] {
        let sps = sps_of(data);
        assert_eq!(sps.reorder_depth(), expect, "{name}");
    }

    // The rest of the SPS is untouched by removing the VUI.
    let (with, without) = (sps_of(BFRAMES), sps_of(NOVUI));
    assert_eq!(without.max_num_reorder_frames, None);
    assert_eq!(
        super::parser::Sps {
            max_num_reorder_frames: with.max_num_reorder_frames,
            ..without
        },
        with
    );

    // A Baseline stream has no B-frames, whatever its POC type and level.
    let mut baseline = sps_of(BASELINE);
    baseline.max_num_reorder_frames = None;
    baseline.pic_order_cnt_type = 0;
    assert_eq!(baseline.reorder_depth(), (0, ReorderSource::Profile));

    // MaxDpbFrames is capped at 16, and follows the picture size.
    let mut sps = sps_of(NOVUI);
    sps.level_idc = 51;
    assert_eq!(sps.reorder_depth(), (16, ReorderSource::Level));
    sps.level_idc = 41;
    sps.pic_width_in_mbs_minus1 = 119;
    sps.pic_height_in_map_units_minus1 = 67;
    assert_eq!(sps.reorder_depth(), (4, ReorderSource::Level));
}

/// The pyramid fixtures use B-frames as references, each with its own way of
/// retiring them.
#[test]
fn pyramid_fixtures_have_reference_b_frames() {
    for (name, data) in [("bpyramid", BPYRAMID), ("bstrict", BSTRICT)] {
        let mut sps = None;
        let mut pps = None;
        let mut reference_b = 0;
        let mut plain_b = 0;
        for nal in iter_nal_units(data) {
            match nal.nal_type {
                NalType::Sps => sps = Some(parse_sps(nal.payload()).unwrap()),
                NalType::Pps => pps = Some(parse_pps(nal.payload(), |_| Some(1)).unwrap()),
                t if t.is_slice() => {
                    let header =
                        parse_slice_header(&nal, sps.as_ref().unwrap(), pps.as_ref().unwrap())
                            .unwrap();
                    if header.slice_type == super::parser::SliceType::B {
                        if nal.ref_idc != 0 {
                            reference_b += 1;
                        } else {
                            plain_b += 1;
                        }
                    }
                }
                _ => {}
            }
        }
        assert!(reference_b > 0, "{name}: B-frames used as references");
        assert!(plain_b > 0, "{name}: non-reference B-frames");
    }
}
