//! Parser and DPB tests against real H.265 streams.
//!
//! The fixtures in `tests/data/*.265` are 176x144 (one is 172x140) testsrc2
//! clips of 10 to 30 frames. All but one come from libx265 4.3 through
//! `ffmpeg -c:v libx265 -crf 45 -x265-params log-level=error:<params>`:
//!
//! | fixture         | x265 params                                                       |
//! |-----------------|-------------------------------------------------------------------|
//! | `opengop`       | `open-gop=1:keyint=12:min-keyint=12:scenecut=0:bframes=3:b-pyramid=1` |
//! | `closedgop`     | `open-gop=0:keyint=10:min-keyint=10:scenecut=0:bframes=3:radl=2:wpp=0` |
//! | `slices`        | `slices=4:keyint=15:bframes=2` (3 slices of 3 CTB rows)           |
//! | `main10`        | `keyint=15:bframes=2`, `-pix_fmt yuv420p10le`, 172x140            |
//! | `temporal`      | `temporal-layers=3:bframes=3:b-pyramid=1:keyint=20:scaling-list=default` |
//! | `scaling`       | `scaling-list=<custom HM-format file>:bframes=2`                  |
//!
//! `nvenc` is `hevc_nvenc -bf 3 -b_ref_mode middle -g 10 -rc constqp -qp 45`:
//! unlike x265 it codes a short-term set in the SPS and has VUI HRD
//! parameters. `cra_start` is `opengop` cut at its second CRA (keeping the
//! parameter sets), so it starts with a CRA whose RASL pictures reference
//! pictures that are not there.
//!
//! The expected values below were read from `ffmpeg -bsf:v trace_headers`,
//! which prints every syntax element with its bit position, and the frame
//! counts from `ffprobe -count_frames` (which skips those RASL pictures too).
//! A one-off comparison of every parsed VPS/SPS/PPS field and every slice
//! header field against the trace found no differences; the assertions here
//! keep the parts that matter.

use std::collections::HashMap;
use std::sync::Arc;

use super::dpb::{DecodeDpb, PictureDecision, RefPicSet, SkipReason};
use super::parser::{NalType, NalUnit, Vps, iter_nal_units, parse_vps};
use super::pps::{Pps, parse_pps};
use super::slice::{SliceHeader, parse_slice_header, slice_pps_id, starts_picture};
use super::sps::{Sps, parse_sps};

const OPENGOP: &[u8] = include_bytes!("../../../tests/data/opengop.265");
const CLOSEDGOP: &[u8] = include_bytes!("../../../tests/data/closedgop.265");
const SLICES: &[u8] = include_bytes!("../../../tests/data/slices.265");
const MAIN10: &[u8] = include_bytes!("../../../tests/data/main10.265");
const NVENC: &[u8] = include_bytes!("../../../tests/data/nvenc.265");
const CRA_START: &[u8] = include_bytes!("../../../tests/data/cra_start.265");
const TEMPORAL: &[u8] = include_bytes!("../../../tests/data/temporal.265");
const SCALING: &[u8] = include_bytes!("../../../tests/data/scaling.265");

/// Every fixture, with the number of pictures it holds and the number ffmpeg
/// outputs.
const ALL: &[(&str, &[u8], usize, usize)] = &[
    ("opengop", OPENGOP, 30, 30),
    ("closedgop", CLOSEDGOP, 30, 30),
    ("slices", SLICES, 20, 20),
    ("main10", MAIN10, 20, 20),
    ("nvenc", NVENC, 20, 20),
    ("cra_start", CRA_START, 8, 6),
    ("temporal", TEMPORAL, 20, 20),
    ("scaling", SCALING, 10, 10),
];

/// One slice segment with the parameter sets in effect for it.
struct Segment<'a> {
    nal: NalUnit<'a>,
    header: SliceHeader,
    sps: Sps,
}

/// The parameter sets of a stream and all of its slice segments, parsed.
struct Parsed<'a> {
    vps: Vec<Vps>,
    sps: Vec<Sps>,
    pps: Vec<Pps>,
    segments: Vec<Segment<'a>>,
}

fn parse_stream<'a>(name: &str, data: &'a [u8]) -> Parsed<'a> {
    let mut parsed = Parsed {
        vps: Vec::new(),
        sps: Vec::new(),
        pps: Vec::new(),
        segments: Vec::new(),
    };
    let mut sps_by_id: HashMap<u8, Sps> = HashMap::new();
    let mut pps_by_id: HashMap<u8, Pps> = HashMap::new();
    for nal in iter_nal_units(data) {
        assert_eq!(nal.header.layer_id, 0, "{name}: base layer only");
        match nal.nal_type() {
            NalType::Vps => parsed
                .vps
                .push(parse_vps(nal.payload()).unwrap_or_else(|e| panic!("{name}: VPS: {e}"))),
            NalType::Sps => {
                let sps = parse_sps(nal.payload()).unwrap_or_else(|e| panic!("{name}: SPS: {e}"));
                sps_by_id.insert(sps.sps_seq_parameter_set_id, sps.clone());
                parsed.sps.push(sps);
            }
            NalType::Pps => {
                let pps = parse_pps(nal.payload()).unwrap_or_else(|e| panic!("{name}: PPS: {e}"));
                pps_by_id.insert(pps.pps_pic_parameter_set_id, pps.clone());
                parsed.pps.push(pps);
            }
            t if t.is_vcl() => {
                let pps = pps_by_id[&slice_pps_id(&nal).unwrap()].clone();
                let sps = sps_by_id[&pps.pps_seq_parameter_set_id].clone();
                let header = parse_slice_header(&nal, &sps, &pps)
                    .unwrap_or_else(|e| panic!("{name}: slice header: {e}"));
                assert_eq!(
                    header.first_slice_segment_in_pic_flag,
                    starts_picture(&nal),
                    "{name}: starts_picture agrees with the parsed header"
                );
                parsed.segments.push(Segment { nal, header, sps });
            }
            _ => {}
        }
    }
    parsed
}

impl Parsed<'_> {
    /// The first segment of each picture.
    fn pictures(&self) -> impl Iterator<Item = &Segment<'_>> {
        self.segments
            .iter()
            .filter(|s| s.header.first_slice_segment_in_pic_flag)
    }
}

/// What the DPB made of one picture.
struct Outcome {
    nal_type: NalType,
    /// `None` when skipped.
    poc: Option<i32>,
    skip: Option<SkipReason>,
    rps: RefPicSet,
    missing_reference: bool,
    is_keyframe: bool,
}

/// Run every picture of `parsed` through a [`DecodeDpb`] with exactly
/// `sps_max_dec_pic_buffering` slots, which a conforming stream never
/// overflows.
fn run_dpb(name: &str, parsed: &Parsed) -> Vec<Outcome> {
    let first_sps = &parsed.sps[0];
    let mut dpb = DecodeDpb::new(first_sps.max_dec_pic_buffering(), Arc::default());
    let mut outcomes = Vec::new();
    for segment in parsed.pictures() {
        let decision = dpb
            .begin_picture(&segment.nal.header, &segment.header, &segment.sps)
            .unwrap_or_else(|e| panic!("{name}: begin_picture: {e}"));
        let state = match decision {
            PictureDecision::Skip(reason) => {
                outcomes.push(Outcome {
                    nal_type: segment.nal.nal_type(),
                    poc: None,
                    skip: Some(reason),
                    rps: RefPicSet::default(),
                    missing_reference: false,
                    is_keyframe: false,
                });
                continue;
            }
            PictureDecision::Decode(state) => state,
        };
        // Every slot the reference picture set names holds that picture, and
        // the DPB holds nothing else.
        let rps = &state.rps;
        let named: Vec<_> = [
            &rps.st_curr_before,
            &rps.st_curr_after,
            &rps.st_foll,
            &rps.lt_curr,
            &rps.lt_foll,
        ]
        .into_iter()
        .flatten()
        .filter_map(|e| e.slot.map(|slot| (slot, e.poc)))
        .collect();
        for &(slot, poc) in &named {
            assert!(
                dpb.references()
                    .iter()
                    .any(|r| r.slot == slot && r.poc == poc),
                "{name}: slot {slot} holds POC {poc}"
            );
        }
        assert_eq!(
            named.len(),
            dpb.references().len(),
            "{name}: DPB holds the RPS"
        );
        let slot = dpb
            .try_allocate_slot()
            .unwrap_or_else(|| panic!("{name}: DPB overflow at POC {}", state.poc));
        dpb.end_picture(slot, &state);
        outcomes.push(Outcome {
            nal_type: state.nal_type,
            poc: Some(state.poc),
            skip: None,
            rps: state.rps,
            missing_reference: state.missing_reference,
            is_keyframe: state.is_keyframe,
        });
    }
    outcomes
}

/// Every NAL unit of every fixture parses, and the picture structure is what
/// the encoder was asked for.
#[test]
fn parses_every_nal_unit() {
    for &(name, data, pictures, _) in ALL {
        let parsed = parse_stream(name, data);
        assert!(!parsed.vps.is_empty(), "{name}: has a VPS");
        assert!(!parsed.sps.is_empty(), "{name}: has an SPS");
        assert!(!parsed.pps.is_empty(), "{name}: has a PPS");
        assert_eq!(parsed.pictures().count(), pictures, "{name}: pictures");
        let first = parsed.pictures().next().unwrap();
        assert!(first.nal.nal_type().is_irap(), "{name}: opens with an IRAP");
    }
    // Three slices per picture, at CTB rows 0, 1 and 2.
    let parsed = parse_stream("slices", SLICES);
    assert_eq!(parsed.segments.len(), 60);
    let addresses: Vec<u32> = parsed.segments[..3]
        .iter()
        .map(|s| s.header.slice_segment_address)
        .collect();
    assert_eq!(addresses, vec![0, 3, 6]);
    // Every slice of a picture carries the same POC LSB and slice type.
    for picture in parsed.segments.chunks(3) {
        for s in &picture[1..] {
            assert!(!s.header.first_slice_segment_in_pic_flag);
            assert_eq!(
                s.header.slice_pic_order_cnt_lsb,
                picture[0].header.slice_pic_order_cnt_lsb
            );
            assert_eq!(s.header.slice_type, picture[0].header.slice_type);
        }
    }
}

/// SPS and VPS values against trace_headers.
#[test]
fn parses_sps_and_vps() {
    for &(name, data, _, _) in ALL {
        let parsed = parse_stream(name, data);
        let sps = &parsed.sps[0];
        let vps = &parsed.vps[0];
        assert_eq!(sps.chroma_format_idc, 1, "{name}: 4:2:0");
        let display = if name == "main10" {
            (172, 140)
        } else {
            (176, 144)
        };
        assert_eq!(sps.display_dimensions(), display, "{name}: display size");
        assert_eq!(
            sps.log2_max_pic_order_cnt_lsb_minus4, 4,
            "{name}: 8-bit POC LSB"
        );
        assert_eq!(sps.max_pic_order_cnt_lsb(), 256);
        assert!(!sps.long_term_ref_pics_present_flag, "{name}: no long-term");
        assert_eq!(
            vps.profile_tier_level, sps.profile_tier_level,
            "{name}: VPS and SPS agree on the profile"
        );
        assert_eq!(vps.vps_max_sub_layers_minus1, sps.sps_max_sub_layers_minus1);
        assert_eq!(
            vps.dec_pic_buf_mgr, sps.dec_pic_buf_mgr,
            "{name}: DPB sizes"
        );
        if name != "nvenc" {
            // x265 always codes every short-term set in the slice header.
            assert_eq!(sps.num_short_term_ref_pic_sets(), 0, "{name}");
            assert_eq!(sps.max_dec_pic_buffering(), 5, "{name}");
            assert_eq!(sps.reorder_depth(), 2, "{name}");
            assert_eq!(sps.profile_tier_level.general_level_idc, 60, "{name}");
            assert_eq!(
                (
                    sps.pic_width_in_luma_samples,
                    sps.pic_height_in_luma_samples
                ),
                (176, 144),
                "{name}: coded size"
            );
            assert_eq!(sps.ctb_log2_size(), 6, "{name}: 64x64 CTBs");
            assert_eq!(sps.pic_size_in_ctbs(), 9, "{name}: 3x3 CTBs");
            assert!(sps.sps_temporal_mvp_enabled_flag, "{name}: TMVP");
        }
    }

    let main = &parse_stream("opengop", OPENGOP).sps[0];
    assert_eq!(main.profile_tier_level.general_profile_idc, 1, "Main");
    assert_eq!(main.bit_depth_luma_minus8, 0);
    assert!(!main.conformance_window_flag);
    assert_eq!(main.display_dimensions(), (176, 144));
    assert!(main.sample_adaptive_offset_enabled_flag);
    assert!(main.strong_intra_smoothing_enabled_flag);
    assert!(!main.amp_enabled_flag);
    assert_eq!(main.max_latency_pictures(), Some(5));
    let vui = main.vui.as_ref().expect("x265 writes a VUI");
    assert_eq!(vui.aspect_ratio_idc, 1);
    assert_eq!(vui.video_format, 5);
    assert_eq!((vui.vui_num_units_in_tick, vui.vui_time_scale), (1, 30));

    // Main 10 at 172x140: coded 176x144 with a 2-sample (4 luma) crop.
    let main10 = &parse_stream("main10", MAIN10).sps[0];
    assert_eq!(main10.profile_tier_level.general_profile_idc, 2);
    assert_eq!(
        (main10.bit_depth_luma_minus8, main10.bit_depth_chroma_minus8),
        (2, 2)
    );
    assert!(main10.conformance_window_flag);
    assert_eq!(
        (main10.conf_win_right_offset, main10.conf_win_bottom_offset),
        (2, 2)
    );
    assert_eq!(main10.display_dimensions(), (172, 140));

    // Three temporal sub-layers; every one signals the same DPB sizes.
    let parsed = parse_stream("temporal", TEMPORAL);
    let (sps, vps) = (&parsed.sps[0], &parsed.vps[0]);
    assert_eq!(sps.sps_max_sub_layers_minus1, 2);
    assert_eq!(sps.highest_tid(), 2);
    assert!(!sps.sps_temporal_id_nesting_flag && !vps.vps_temporal_id_nesting_flag);
    assert!(vps.vps_sub_layer_ordering_info_present_flag);
    assert_eq!(
        vps.dec_pic_buf_mgr.max_dec_pic_buffering_minus1[..3],
        [4, 4, 4]
    );
    assert_eq!(vps.dec_pic_buf_mgr.max_num_reorder_pics[..3], [2, 2, 2]);
    assert_eq!(
        vps.dec_pic_buf_mgr.max_latency_increase_plus1[..3],
        [4, 4, 4]
    );
    assert_eq!(sps.reorder_depth(), 2);
}

/// NVENC's SPS: one short-term set, and a VUI with HRD parameters that must be
/// skipped exactly to reach the bitstream restriction and extension flags.
#[test]
fn parses_nvenc_sps_rps_and_hrd() {
    let parsed = parse_stream("nvenc", NVENC);
    let sps = &parsed.sps[0];
    // 32x32 CTBs, coded at 192x160 and cropped by 8 chroma samples a side.
    assert_eq!(
        (
            sps.pic_width_in_luma_samples,
            sps.pic_height_in_luma_samples
        ),
        (192, 160)
    );
    assert_eq!(
        (sps.conf_win_right_offset, sps.conf_win_bottom_offset),
        (8, 8)
    );
    assert_eq!(sps.ctb_log2_size(), 5);
    assert_eq!(sps.pic_size_in_ctbs(), 30);
    assert_eq!(sps.max_dec_pic_buffering(), 6);
    assert_eq!(sps.max_latency_pictures(), None);
    assert!(sps.amp_enabled_flag && !sps.sps_temporal_mvp_enabled_flag);
    assert_eq!(sps.profile_tier_level.general_level_idc, 90);
    assert_eq!(sps.num_short_term_ref_pic_sets(), 1);
    let set = &sps.short_term_ref_pic_sets[0];
    assert!(!set.syntax.inter_ref_pic_set_prediction_flag);
    assert_eq!(set.syntax.num_negative_pics, 5);
    assert_eq!(set.syntax.num_positive_pics, 0);
    assert_eq!(set.syntax.used_by_curr_pic_s0_flag, 0b00011);
    assert_eq!(set.rps.delta_poc_s0, vec![-1, -2, -3, -4, -5]);
    assert_eq!(set.rps.used_s0, vec![true, true, false, false, false]);

    let vui = sps.vui.as_ref().unwrap();
    assert!(vui.vui_hrd_parameters_present_flag);
    assert!(!sps.sps_extension_present_flag);

    // Each slice codes its own set at index 1, so the inter prediction flag
    // is coded (as 0) and counts towards NumBitsForSTRefPicSetInSlice.
    for s in parsed.pictures().filter(|s| !s.nal.nal_type().is_idr()) {
        let inline = s.header.st_ref_pic_set.as_ref().expect("inline set");
        assert!(!inline.syntax.inter_ref_pic_set_prediction_flag);
        assert_eq!(s.header.num_delta_pocs_of_ref_rps_idx(sps), 0);
    }
}

/// PPS values against trace_headers.
#[test]
fn parses_pps() {
    for &(name, data, _, _) in ALL {
        let pps = &parse_stream(name, data).pps[0];
        assert_eq!(pps.pps_pic_parameter_set_id, 0, "{name}");
        assert_eq!(pps.pps_seq_parameter_set_id, 0, "{name}");
        assert!(!pps.tiles_enabled_flag, "{name}: x265 has no tiles");
        assert_eq!(pps.tile_column_widths(3), vec![3], "{name}");
        assert!(!pps.lists_modification_present_flag, "{name}");
        assert!(!pps.pps_extension_present_flag, "{name}");
    }
    let x265 = &parse_stream("opengop", OPENGOP).pps[0];
    assert!(
        x265.entropy_coding_sync_enabled_flag,
        "WPP is x265's default"
    );
    assert!(x265.sign_data_hiding_enabled_flag);
    assert!(x265.cu_qp_delta_enabled_flag);
    assert_eq!(x265.diff_cu_qp_delta_depth, 1);
    assert!(x265.weighted_pred_flag);
    assert!(!x265.weighted_bipred_flag);
    assert!(x265.pps_loop_filter_across_slices_enabled_flag);
    assert!(!x265.deblocking_filter_control_present_flag);
    assert_eq!(x265.log2_parallel_merge_level_minus2, 0);
    let closed = &parse_stream("closedgop", CLOSEDGOP).pps[0];
    assert!(!closed.entropy_coding_sync_enabled_flag, "wpp=0");
}

/// Explicit SPS scaling lists, against the values rebuilt from the trace's
/// `scaling_list_delta_coef`s, including lists predicted from the previous
/// matrix and the DC coefficients of the 16x16 and 32x32 lists.
#[test]
fn parses_scaling_lists() {
    let sps = &parse_stream("scaling", SCALING).sps[0];
    assert!(sps.scaling_list_enabled_flag && sps.sps_scaling_list_data_present_flag);
    let lists = sps.scaling_lists.as_ref().unwrap();
    // Coded order: the HM file's raster 4x4 starts 12, 14, 19, 24 / 22, ...
    // and the up-right diagonal scan visits (0,0), (1,0), (0,1), (2,0), ...
    assert_eq!(
        lists.scaling_list_4x4[0][..8],
        [12, 22, 14, 36, 25, 19, 18, 41]
    );
    assert_eq!(
        lists.scaling_list_8x8[0][..8],
        [19, 44, 21, 40, 46, 27, 34, 38]
    );
    assert_eq!(
        lists.scaling_list_16x16[0][..8],
        [22, 50, 27, 44, 49, 33, 40, 48]
    );
    assert_eq!(
        lists.scaling_list_32x32[1][..8],
        [32, 53, 35, 49, 58, 35, 45, 52]
    );
    let sums: Vec<u32> = (0..6)
        .map(|m| lists.scaling_list_16x16[m].iter().map(|&c| c as u32).sum())
        .collect();
    assert_eq!(sums, vec![2424, 2476, 2537, 2592, 2708, 2708]);
    // matrixId 5 is predicted from 4 at every size (delta 1).
    assert_eq!(lists.scaling_list_4x4[5], lists.scaling_list_4x4[4]);
    assert_eq!(lists.scaling_list_8x8[5], lists.scaling_list_8x8[4]);
    assert_eq!(lists.scaling_list_16x16[5], lists.scaling_list_16x16[4]);
    assert_eq!(lists.scaling_list_dc_coef_16x16, [23, 27, 27, 31, 28, 28]);
    assert_eq!(lists.scaling_list_dc_coef_32x32, [32, 33]);

    // scaling-list=default: enabled, no data, so the Table 7-6 defaults.
    let sps = &parse_stream("temporal", TEMPORAL).sps[0];
    assert!(sps.scaling_list_enabled_flag && !sps.sps_scaling_list_data_present_flag);
    let defaults = sps.scaling_lists.as_ref().unwrap();
    assert_eq!(defaults.scaling_list_4x4[0], [16; 16]);
    assert_eq!(defaults.scaling_list_8x8[0][63], 115, "intra default");
    assert_eq!(defaults.scaling_list_32x32[1][63], 91, "inter default");
    assert_eq!(defaults.scaling_list_dc_coef_32x32, [16, 16]);
}

/// Per picture, in decoding order: nal_unit_type, slice_pic_order_cnt_lsb,
/// and the bits of the inline st_ref_pic_set() (from trace_headers' bit
/// positions: the element after short_term_ref_pic_set_sps_flag up to
/// slice_temporal_mvp_enabled_flag).
#[test]
fn slice_headers_match_trace() {
    /// Name, stream, nal_unit_type, POC LSB and RPS bits per picture.
    type Case<'a> = (&'a str, &'a [u8], &'a [u8], &'a [u32], &'a [u32]);
    let cases: &[Case] = &[
        (
            "opengop",
            OPENGOP,
            &[
                20, 1, 1, 1, 0, 0, 1, 1, 0, 0, 21, 9, 8, 1, 1, 0, 1, 1, 0, 1, 1, 0, 21, 9, 8, 1, 1,
                0, 0, 1,
            ],
            &[
                0, 1, 5, 3, 2, 4, 9, 7, 6, 8, 12, 11, 10, 15, 14, 13, 18, 17, 16, 21, 20, 19, 24,
                23, 22, 28, 26, 25, 27, 29,
            ],
            &[
                0, 6, 12, 16, 16, 18, 22, 24, 18, 20, 22, 24, 16, 8, 12, 12, 16, 20, 14, 18, 20,
                14, 18, 20, 14, 10, 14, 14, 14, 16,
            ],
        ),
        (
            "closedgop",
            CLOSEDGOP,
            &[
                20, 1, 1, 1, 0, 0, 1, 0, 19, 7, 6, 1, 1, 0, 1, 0, 1, 0, 19, 7, 6, 1, 1, 0, 1, 0, 1,
                0, 1, 0,
            ],
            &[
                0, 1, 5, 3, 2, 4, 7, 6, 0, 255, 254, 3, 2, 1, 5, 4, 7, 6, 0, 255, 254, 3, 2, 1, 5,
                4, 7, 6, 9, 8,
            ],
            &[
                0, 6, 12, 16, 16, 18, 20, 20, 0, 6, 8, 8, 12, 12, 16, 18, 20, 18, 0, 6, 8, 8, 12,
                12, 16, 18, 20, 18, 20, 20,
            ],
        ),
        (
            "nvenc",
            NVENC,
            &[19, 1, 1, 0, 0, 1, 1, 0, 0, 1, 19, 1, 1, 0, 0, 1, 1, 0, 0, 1],
            &[0, 4, 2, 1, 3, 8, 6, 5, 7, 9, 0, 4, 2, 1, 3, 8, 6, 5, 7, 9],
            &[
                0, 11, 15, 15, 15, 21, 25, 25, 25, 25, 0, 11, 15, 15, 15, 21, 25, 25, 25, 25,
            ],
        ),
        (
            "temporal",
            TEMPORAL,
            &[20, 1, 3, 2, 2, 1, 3, 2, 2, 1, 3, 2, 2, 1, 3, 2, 2, 1, 1, 2],
            &[
                0, 4, 2, 1, 3, 8, 6, 5, 7, 12, 10, 9, 11, 16, 14, 13, 15, 19, 18, 17,
            ],
            &[
                0, 10, 14, 14, 14, 16, 20, 20, 22, 24, 28, 20, 22, 24, 28, 20, 22, 22, 26, 18,
            ],
        ),
    ];
    for &(name, data, types, lsbs, bits) in cases {
        let parsed = parse_stream(name, data);
        let pictures: Vec<_> = parsed.pictures().collect();
        let got: Vec<u8> = pictures.iter().map(|s| s.nal.nal_type().raw()).collect();
        assert_eq!(got, types, "{name}: nal_unit_type");
        let got: Vec<u32> = pictures
            .iter()
            .map(|s| s.header.slice_pic_order_cnt_lsb)
            .collect();
        assert_eq!(got, lsbs, "{name}: slice_pic_order_cnt_lsb");
        let got: Vec<u32> = pictures
            .iter()
            .map(|s| s.header.num_bits_for_st_ref_pic_set)
            .collect();
        assert_eq!(got, bits, "{name}: NumBitsForSTRefPicSetInSlice");
        for s in &pictures {
            assert!(
                s.header.pic_output_flag,
                "{name}: no output_flag_present_flag"
            );
            assert!(
                !s.header.short_term_ref_pic_set_sps_flag,
                "{name}: every encoder here codes the set inline"
            );
        }
    }

    // Temporal sub-layers: TemporalId per picture, and TSA types.
    let parsed = parse_stream("temporal", TEMPORAL);
    let tids: Vec<u8> = parsed
        .pictures()
        .map(|s| s.nal.header.temporal_id)
        .collect();
    assert_eq!(
        tids,
        vec![0, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1]
    );
    let types: Vec<NalType> = parsed.pictures().map(|s| s.nal.nal_type()).collect();
    assert_eq!(types[2], NalType::TsaR);
    assert_eq!(types[3], NalType::TsaN);
}

/// POC in decoding order, derived by the DPB, against the trace's LSBs and
/// IRAP structure. closedgop's RADL pictures have LSB 255/254 after an IDR,
/// which is POC -1/-2.
#[test]
fn poc_sequences() {
    let cases: &[(&str, &[u8], &[i32])] = &[
        (
            "opengop",
            OPENGOP,
            &[
                0, 1, 5, 3, 2, 4, 9, 7, 6, 8, 12, 11, 10, 15, 14, 13, 18, 17, 16, 21, 20, 19, 24,
                23, 22, 28, 26, 25, 27, 29,
            ],
        ),
        (
            "closedgop",
            CLOSEDGOP,
            &[
                0, 1, 5, 3, 2, 4, 7, 6, 0, -1, -2, 3, 2, 1, 5, 4, 7, 6, 0, -1, -2, 3, 2, 1, 5, 4,
                7, 6, 9, 8,
            ],
        ),
        (
            "nvenc",
            NVENC,
            &[0, 4, 2, 1, 3, 8, 6, 5, 7, 9, 0, 4, 2, 1, 3, 8, 6, 5, 7, 9],
        ),
        (
            "temporal",
            TEMPORAL,
            &[
                0, 4, 2, 1, 3, 8, 6, 5, 7, 12, 10, 9, 11, 16, 14, 13, 15, 19, 18, 17,
            ],
        ),
    ];
    for &(name, data, expected) in cases {
        let outcomes = run_dpb(name, &parse_stream(name, data));
        let pocs: Vec<i32> = outcomes.iter().map(|o| o.poc.unwrap()).collect();
        assert_eq!(pocs, expected, "{name}: POC in decoding order");
    }

    // In every fixture, each coded video sequence covers a contiguous range of
    // POCs, which is what a display-order sort of the output needs.
    for &(name, data, _, output) in ALL {
        let outcomes = run_dpb(name, &parse_stream(name, data));
        let decoded: Vec<_> = outcomes.iter().filter(|o| o.poc.is_some()).collect();
        assert_eq!(decoded.len(), output, "{name}: pictures output, as ffprobe");
        let mut sequences: Vec<Vec<i32>> = Vec::new();
        for o in decoded {
            if o.is_keyframe {
                sequences.push(Vec::new());
            }
            sequences.last_mut().unwrap().push(o.poc.unwrap());
        }
        for mut pocs in sequences {
            pocs.sort();
            assert!(
                pocs.windows(2).all(|w| w[1] == w[0] + 1),
                "{name}: contiguous POCs {pocs:?}"
            );
        }
    }
}

/// Every reference the slice headers name exists, in a DPB no larger than the
/// SPS allows, except where random access left them behind.
#[test]
fn reference_picture_sets_resolve() {
    for &(name, data, _, _) in ALL {
        let outcomes = run_dpb(name, &parse_stream(name, data));
        for (i, o) in outcomes.iter().enumerate() {
            if o.skip.is_some() {
                continue;
            }
            assert!(
                !o.missing_reference,
                "{name}: picture {i} misses a reference"
            );
            let entries = o
                .rps
                .st_curr_before
                .iter()
                .chain(&o.rps.st_curr_after)
                .chain(&o.rps.st_foll);
            for e in entries {
                if name == "cra_start" && i == 0 {
                    // The CRA keeps pictures for its RASL pictures, which
                    // this cut does not have.
                    assert!(e.slot.is_none(), "{name}: nothing before the CRA");
                } else {
                    assert!(e.slot.is_some(), "{name}: picture {i}: POC {} held", e.poc);
                }
            }
            if o.nal_type.is_irap() {
                assert!(
                    o.rps.st_curr_before.is_empty() && o.rps.st_curr_after.is_empty(),
                    "{name}: an IRAP predicts from nothing"
                );
            }
            // B pictures reference both directions, P pictures only the past.
            let b = o.rps.st_curr_after.iter().all(|e| e.poc > o.poc.unwrap());
            let p = o.rps.st_curr_before.iter().all(|e| e.poc < o.poc.unwrap());
            assert!(b && p, "{name}: picture {i} list ordering");
        }
    }
}

/// A stream cut at a CRA: the CRA starts decoding, its two RASL pictures are
/// skipped, and everything after it decodes from the CRA alone.
#[test]
fn cra_start_skips_rasl() {
    let outcomes = run_dpb("cra_start", &parse_stream("cra_start", CRA_START));
    let summary: Vec<(NalType, Option<i32>, Option<SkipReason>)> = outcomes
        .iter()
        .map(|o| (o.nal_type, o.poc, o.skip))
        .collect();
    let rasl = Some(SkipReason::RaslWithoutReferences);
    assert_eq!(
        summary[..4],
        [
            (NalType::Cra, Some(24), None),
            (NalType::RaslR, None, rasl),
            (NalType::RaslN, None, rasl),
            (NalType::TrailR, Some(28), None),
        ]
    );
    assert!(outcomes[0].is_keyframe);
    // POC 28 is a P picture predicting from the CRA only.
    assert_eq!(outcomes[3].rps.st_curr_before.len(), 1);
    assert_eq!(outcomes[3].rps.st_curr_before[0].poc, 24);

    // In the uncut stream the same CRA is not a keyframe, and its RASL
    // pictures decode.
    let outcomes = run_dpb("opengop", &parse_stream("opengop", OPENGOP));
    let cra = outcomes.iter().position(|o| o.poc == Some(24)).unwrap();
    assert_eq!(outcomes[cra].nal_type, NalType::Cra);
    assert!(!outcomes[cra].is_keyframe);
    assert_eq!(outcomes[cra + 1].poc, Some(23));
    assert!(!outcomes[cra + 1].missing_reference);
}
