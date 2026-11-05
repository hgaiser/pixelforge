//! Parser tests against real encoder output.
//!
//! The fixtures in `tests/data/av1` are 320x240 (`aom_mono`: 176x144)
//! streams of 12 to 15 frames in the low-overhead OBU format, from libaom
//! (through ffmpeg's `libaom-av1` and `aomenc --obu`) and SVT-AV1. Each
//! `<name>.expected` holds what ffmpeg's `trace_headers` bitstream filter
//! printed for that stream, reduced to one `F` line per frame header and one
//! `T` line per tile group, each a list of `syntax_element=value`. The
//! script that reduced them, `tests/data/av1/extract_trace.py`, also lists
//! the encoder settings; `header_bits` is the position after the header's last
//! syntax element minus that of its first.
//!
//! | stream            | what it covers                                         |
//! |-------------------|--------------------------------------------------------|
//! | `aom_altref`      | hidden alt-refs, `show_existing_frame`, skip mode      |
//! | `aom_tiles`       | 2x2 uniform tiles, two-byte tile sizes                 |
//! | `aom_tilegroups`  | frame header OBUs with two tile groups each, 128x128   |
//! |                   | superblocks, quantizer matrices, segmentation, delta   |
//! |                   | lf, loop restoration, decoder model and removal times  |
//! | `aom_10bit`       | 10-bit Main profile                                    |
//! | `aom_grain`       | film grain (libaom denoiser)                           |
//! | `svt_grain`       | SVT-AV1 with film grain, `update_grain = 0` reuse      |
//! | `aom_errres`      | error resilient frames, frame ids, `ref_order_hint`,   |
//! |                   | S-frames                                               |
//! | `aom_superres`    | superres                                               |
//! | `aom_resize`      | frame size override, `frame_size_with_refs`, render    |
//! |                   | size                                                   |
//! | `aom_mono`        | monochrome, loop restoration                           |
//! | `aom_annexb`      | Annex B framing, which must be rejected                |
//!
//! No encoder at hand produces `frame_refs_short_signaling`, non-uniform
//! tile spacing or redundant frame headers; `refs.rs` and `tile.rs` test
//! those paths on synthetic input.

use super::frame_header::FrameHeader;
use super::obu::{ObuType, iter_obus, temporal_unit_starts};
use super::parser::{Av1Event, Av1Frame, Av1Parser};
use super::sequence::SequenceHeader;
use super::std_info::{
    StdPictureInfo, StdSequenceHeader, reference_name_slot_indices, std_reference_info,
};

macro_rules! fixture {
    ($name:literal) => {
        (
            $name,
            include_bytes!(concat!("../../../tests/data/av1/", $name, ".obu")).as_slice(),
            include_str!(concat!("../../../tests/data/av1/", $name, ".expected")),
        )
    };
}

const FIXTURES: [(&str, &[u8], &str); 10] = [
    fixture!("aom_altref"),
    fixture!("aom_tiles"),
    fixture!("aom_tilegroups"),
    fixture!("aom_10bit"),
    fixture!("aom_grain"),
    fixture!("svt_grain"),
    fixture!("aom_errres"),
    fixture!("aom_superres"),
    fixture!("aom_resize"),
    fixture!("aom_mono"),
];

const ANNEX_B: &[u8] = include_bytes!("../../../tests/data/av1/aom_annexb.obu");

fn fixture(name: &str) -> &'static [u8] {
    FIXTURES.iter().find(|f| f.0 == name).unwrap().1
}

/// Everything parsing a whole stream produced.
struct Parsed {
    sequence: SequenceHeader,
    frames: Vec<Av1Frame>,
    /// `(payload_offset, end)` of every tile group carrying OBU.
    tile_group_obus: Vec<(usize, usize)>,
    /// Frames shown, by `show_frame` or `show_existing_frame`.
    shown: usize,
}

/// Parse a stream, committing every frame, and check what holds for every
/// stream: every OBU parses and every frame is complete.
fn parse_stream(name: &str, data: &[u8]) -> Parsed {
    let mut parser = Av1Parser::new();
    let mut frames = Vec::new();
    let mut tile_group_obus = Vec::new();
    let mut shown = 0;
    let mut next_dpb_slot = 0u8;
    for obu in iter_obus(data) {
        let obu = obu.unwrap_or_else(|e| panic!("{name}: OBU framing: {e}"));
        if matches!(obu.header.obu_type, ObuType::TileGroup | ObuType::Frame) {
            tile_group_obus.push((
                obu.payload_offset(),
                obu.payload_offset() + obu.payload.len(),
            ));
        }
        let event = parser
            .parse_obu(&obu)
            .unwrap_or_else(|e| panic!("{name}: OBU at byte {}: {e}", obu.offset));
        match event {
            Some(Av1Event::Frame(frame)) => {
                let h = &frame.header;
                if h.show_frame || h.show_existing_frame {
                    shown += 1;
                }
                if !h.show_existing_frame {
                    check_vulkan_conversion(&parser, h);
                }
                // A stand-in for the DPB slot the decoder would pick.
                parser.commit_frame(Some(next_dpb_slot)).unwrap();
                next_dpb_slot = (next_dpb_slot + 1) % 16;
                frames.push(*frame);
            }
            Some(Av1Event::MissingReferences(slots)) => {
                panic!("{name}: missing references {slots:#010b}")
            }
            Some(Av1Event::SequenceHeader) | None => {}
        }
    }
    Parsed {
        sequence: parser
            .sequence_header()
            .expect("stream has a sequence header")
            .clone(),
        frames,
        tile_group_obus,
        shown,
    }
}

/// Build the Vulkan structures for a frame about to be decoded, as the
/// decoder will, and spot-check that they carry the parsed values.
fn check_vulkan_conversion(parser: &Av1Parser, h: &FrameHeader) {
    let seq = StdSequenceHeader::new(parser.sequence_header().unwrap());
    let std_seq = seq.get();
    // SAFETY: the pointer targets are owned by `seq`, which is alive.
    let color = unsafe { &*std_seq.pColorConfig };
    assert_eq!(
        color.BitDepth,
        parser.sequence_header().unwrap().color_config.bit_depth
    );

    let pic = StdPictureInfo::new(h);
    let info = pic.get();
    assert_eq!(info.OrderHint, h.order_hint);
    assert_eq!(info.refresh_frame_flags, h.refresh_frame_flags);
    // SAFETY: as above, owned by `pic`.
    let tile_info = unsafe { &*info.pTileInfo };
    let quant = unsafe { &*info.pQuantization };
    assert_eq!(quant.base_q_idx, h.quantization.base_q_idx);
    assert_eq!(tile_info.TileCols, h.tile_info.tile_cols);
    let cols =
        unsafe { std::slice::from_raw_parts(tile_info.pMiColStarts, tile_info.TileCols as usize) };
    assert_eq!(cols, &h.tile_info.mi_col_starts[..cols.len()]);

    let indices = reference_name_slot_indices(h, &parser.refs().slots);
    if !h.frame_is_intra() {
        for (i, &idx) in h.ref_frame_idx.iter().enumerate() {
            let slot = &parser.refs().slots[idx as usize];
            assert!(slot.valid);
            assert_eq!(indices[i], slot.dpb_slot.unwrap() as i32);
            assert_eq!(std_reference_info(slot).OrderHint, h.order_hints[i + 1]);
        }
    } else {
        assert!(indices.iter().all(|&i| i == -1));
    }
}

/// The `syntax_element=value` pairs of one line.
type Elements = Vec<(String, i64)>;

/// The `F` (frame header) and `T` (tile group) lines of an `.expected` file.
fn parse_expected(text: &str) -> (Vec<Elements>, Vec<Elements>) {
    let mut frames = Vec::new();
    let mut tile_groups = Vec::new();
    for line in text.lines() {
        let mut words = line.split_whitespace();
        let kind = words.next().unwrap();
        let pairs: Elements = words
            .map(|w| {
                let (k, v) = w.split_once('=').unwrap();
                (k.to_string(), v.parse().unwrap())
            })
            .collect();
        match kind {
            "F" => frames.push(pairs),
            "T" => tile_groups.push(pairs),
            other => panic!("unknown line kind {other}"),
        }
    }
    (frames, tile_groups)
}

/// The parsed value of the syntax element `trace_headers` calls `key`.
/// `return_coded` is the traced value, returned where the parser keeps a
/// derived value that cannot be compared with it.
fn value_of(f: &Av1Frame, key: &str, return_coded: i64) -> i64 {
    let h = &f.header;
    let (name, index) = match key.split_once('[') {
        Some((name, rest)) => (
            name,
            Some(rest.trim_end_matches(']').parse::<usize>().unwrap()),
        ),
        None => (key, None),
    };
    let b = |v: bool| v as i64;
    let lr = &h.loop_restoration;
    match (name, index) {
        ("obu", None) => match f.header_obu_type {
            ObuType::FrameHeader => 3,
            ObuType::Frame => 6,
            ObuType::RedundantFrameHeader => 7,
            other => panic!("header in {other:?}"),
        },
        ("show_existing_frame", None) => b(h.show_existing_frame),
        ("frame_to_show_map_idx", None) => h.frame_to_show_map_idx as i64,
        ("frame_type", None) => h.frame_type as i64,
        ("show_frame", None) => b(h.show_frame),
        ("showable_frame", None) => b(h.showable_frame),
        ("error_resilient_mode", None) => b(h.error_resilient_mode),
        ("disable_cdf_update", None) => b(h.disable_cdf_update),
        ("allow_screen_content_tools", None) => b(h.allow_screen_content_tools),
        // Intra frames code force_integer_mv but always use 1, which is the
        // value kept; only an inter frame's coded value is comparable.
        ("force_integer_mv", None) if h.frame_is_intra() => return_coded,
        ("force_integer_mv", None) => b(h.force_integer_mv),
        ("current_frame_id", None) => h.current_frame_id as i64,
        ("frame_size_override_flag", None) => b(h.frame_size_override_flag),
        ("order_hint", None) => h.order_hint as i64,
        ("primary_ref_frame", None) => h.primary_ref_frame as i64,
        ("refresh_frame_flags", None) => h.refresh_frame_flags as i64,
        ("ref_frame_idx", Some(i)) => h.ref_frame_idx[i] as i64,
        ("frame_width_minus_1", None) => h.frame_size.upscaled_width as i64 - 1,
        ("frame_height_minus_1", None) => h.frame_size.frame_height as i64 - 1,
        ("render_width_minus_1", None) => h.frame_size.render_width as i64 - 1,
        ("render_height_minus_1", None) => h.frame_size.render_height as i64 - 1,
        ("use_superres", None) => b(h.frame_size.use_superres),
        ("coded_denom", None) => h.frame_size.coded_denom as i64,
        ("allow_intrabc", None) => b(h.allow_intrabc),
        ("allow_high_precision_mv", None) => b(h.allow_high_precision_mv),
        ("is_filter_switchable", None) => b(h.is_filter_switchable),
        ("interpolation_filter", None) => h.interpolation_filter as i64,
        ("use_ref_frame_mvs", None) => b(h.use_ref_frame_mvs),
        ("disable_frame_end_update_cdf", None) => b(h.disable_frame_end_update_cdf),
        ("tile_cols_log2", None) => h.tile_info.tile_cols_log2 as i64,
        ("tile_rows_log2", None) => h.tile_info.tile_rows_log2 as i64,
        ("context_update_tile_id", None) => h.tile_info.context_update_tile_id as i64,
        ("tile_size_bytes_minus1", None) => h.tile_info.tile_size_bytes_minus_1 as i64,
        ("base_q_idx", None) => h.quantization.base_q_idx as i64,
        ("using_qmatrix", None) => b(h.quantization.using_qmatrix),
        ("qm_y", None) => h.quantization.qm_y as i64,
        ("qm_u", None) => h.quantization.qm_u as i64,
        ("qm_v", None) => h.quantization.qm_v as i64,
        ("segmentation_enabled", None) => b(h.segmentation.segmentation_enabled),
        ("segmentation_update_map", None) => b(h.segmentation.segmentation_update_map),
        ("segmentation_temporal_update", None) => b(h.segmentation.segmentation_temporal_update),
        ("segmentation_update_data", None) => b(h.segmentation.segmentation_update_data),
        ("delta_q_present", None) => b(h.delta.delta_q_present),
        ("delta_q_res", None) => h.delta.delta_q_res as i64,
        ("delta_lf_present", None) => b(h.delta.delta_lf_present),
        ("delta_lf_res", None) => h.delta.delta_lf_res as i64,
        ("delta_lf_multi", None) => b(h.delta.delta_lf_multi),
        ("loop_filter_level", Some(i)) => h.loop_filter.loop_filter_level[i] as i64,
        ("loop_filter_sharpness", None) => h.loop_filter.loop_filter_sharpness as i64,
        ("loop_filter_delta_enabled", None) => b(h.loop_filter.loop_filter_delta_enabled),
        ("loop_filter_delta_update", None) => b(h.loop_filter.loop_filter_delta_update),
        ("cdef_damping_minus_3", None) => h.cdef.cdef_damping_minus_3 as i64,
        ("cdef_bits", None) => h.cdef.cdef_bits as i64,
        ("cdef_y_pri_strength", Some(i)) => h.cdef.cdef_y_pri_strength[i] as i64,
        ("cdef_y_sec_strength", Some(i)) => h.cdef.cdef_y_sec_strength[i] as i64,
        // trace_headers prints the coded lr_type, before the remapping.
        ("lr_type", Some(i)) => [0, 2, 3, 1][lr.frame_restoration_type[i] as usize],
        // cbs_av1 reads lr_unit_shift as one increment and prints the sum.
        ("lr_unit_shift", None) => lr.lr_unit_shift as i64,
        ("lr_uv_shift", None) => lr.lr_uv_shift as i64,
        ("tx_mode", None) => h.tx_mode as i64,
        ("reference_select", None) => b(h.reference_select),
        ("skip_mode_present", None) => b(h.skip_mode_present),
        ("allow_warped_motion", None) => b(h.allow_warped_motion),
        ("reduced_tx_set", None) => b(h.reduced_tx_set),
        ("apply_grain", None) => b(h.film_grain.apply_grain),
        ("grain_seed", None) => h.film_grain.grain_seed as i64,
        ("update_grain", None) => b(h.film_grain.update_grain),
        ("film_grain_params_ref_idx", None) => h.film_grain.film_grain_params_ref_idx as i64,
        ("num_y_points", None) => h.film_grain.num_y_points as i64,
        ("num_cb_points", None) => h.film_grain.num_cb_points as i64,
        ("num_cr_points", None) => h.film_grain.num_cr_points as i64,
        ("header_bits", None) => h.header_bits as i64,
        _ => panic!("no mapping for trace element {key}"),
    }
}

/// Every syntax element `trace_headers` printed matches the parse, frame by
/// frame and tile group by tile group, including each header's bit length.
#[test]
fn headers_match_trace_headers() {
    for (name, data, expected) in FIXTURES {
        let parsed = parse_stream(name, data);
        let (frames, tile_groups) = parse_expected(expected);
        assert_eq!(parsed.frames.len(), frames.len(), "{name}: frame count");
        for (n, (frame, pairs)) in parsed.frames.iter().zip(&frames).enumerate() {
            for (key, want) in pairs {
                let got = value_of(frame, key, *want);
                assert_eq!(got, *want, "{name}: frame {n}: {key}");
            }
            // skip_mode_present is coded exactly when skip mode is allowed,
            // so its presence checks the SkipModeFrame derivation.
            let skip_coded = pairs.iter().any(|(k, _)| k == "skip_mode_present");
            let h = &frame.header;
            if !h.show_existing_frame {
                assert_eq!(
                    h.skip_mode_frame[0] != 0,
                    skip_coded,
                    "{name}: frame {n}: skipModeAllowed"
                );
            }
        }
        let parsed_groups: Vec<_> = parsed.frames.iter().flat_map(|f| &f.tile_groups).collect();
        assert_eq!(
            parsed_groups.len(),
            tile_groups.len(),
            "{name}: tile groups"
        );
        for (n, (tg, pairs)) in parsed_groups.iter().zip(&tile_groups).enumerate() {
            for (key, want) in pairs {
                let got = match key.as_str() {
                    "obu" => continue,
                    "tile_start_and_end_present_flag" => tg.tile_start_and_end_present_flag as i64,
                    "tg_start" => tg.tg_start as i64,
                    "tg_end" => tg.tg_end as i64,
                    other => panic!("no mapping for {other}"),
                };
                assert_eq!(got, *want, "{name}: tile group {n}: {key}");
            }
        }
    }
}

/// Tiles cover every tile group exactly: each tile starts right after the
/// previous one and its size field, and the last ends at the end of the OBU.
/// The frame header offset points at the OBU that carried the header.
#[test]
fn tiles_and_header_offsets_cover_the_obus() {
    for (name, data, _) in FIXTURES {
        let parsed = parse_stream(name, data);
        let mut obus = parsed.tile_group_obus.iter();
        for (n, frame) in parsed.frames.iter().enumerate() {
            let h = &frame.header;
            let first = data[frame.frame_header_offset];
            let obu_type = (first >> 3) & 0x0f;
            assert!(
                matches!(obu_type, 3 | 6),
                "{name}: frame {n}: header OBU type"
            );
            assert_eq!(
                frame.frame_header_bytes,
                h.header_bits.div_ceil(8)
                    + (frame.header_obu_type == ObuType::FrameHeader && h.header_bits % 8 == 0)
                        as usize,
                "{name}: frame {n}: header bytes vs bits"
            );
            if h.show_existing_frame {
                assert!(frame.tile_groups.is_empty());
                continue;
            }
            assert_eq!(
                frame.tiles().count(),
                h.tile_info.num_tiles(),
                "{name}: frame {n}"
            );
            let size_bytes = h.tile_info.tile_size_bytes() as usize;
            for tg in &frame.tile_groups {
                let &(_, obu_end) = obus.next().expect("a tile group OBU per tile group");
                // Every tile but the last is preceded by its size field.
                for (k, pair) in tg.tiles.windows(2).enumerate() {
                    let field = if k + 2 < tg.tiles.len() {
                        size_bytes
                    } else {
                        0
                    };
                    assert_eq!(pair[0].offset + pair[0].size + field, pair[1].offset);
                }
                let last = tg.tiles.last().unwrap();
                assert_eq!(
                    last.offset + last.size,
                    obu_end,
                    "{name}: frame {n}: last tile"
                );
                assert!(tg.tiles.iter().all(|t| t.size > 0));
            }
        }
        assert!(obus.next().is_none(), "{name}: tile group OBUs left over");
    }
}

/// Each temporal unit shows exactly one frame, and the temporal unit
/// boundaries split the stream into units that parse on their own.
#[test]
fn temporal_units_show_one_frame_each() {
    for (name, data, _) in FIXTURES {
        let parsed = parse_stream(name, data);
        let starts = temporal_unit_starts(data).unwrap();
        assert_eq!(
            starts[0], 0,
            "{name}: stream starts with a temporal delimiter"
        );
        assert_eq!(
            starts.len(),
            parsed.shown,
            "{name}: one shown frame per temporal unit"
        );

        // Feeding the units one at a time gives the same frames.
        let mut parser = Av1Parser::new();
        let mut count = 0;
        let mut bounds = starts.clone();
        bounds.push(data.len());
        for w in bounds.windows(2) {
            for obu in iter_obus(&data[w[0]..w[1]]) {
                if let Some(Av1Event::Frame(_)) = parser.parse_obu(&obu.unwrap()).unwrap() {
                    parser.commit_frame(None).unwrap();
                    count += 1;
                }
            }
        }
        assert_eq!(count, parsed.frames.len(), "{name}: frames per unit");

        // A byte stream cut mid-OBU still yields the complete units.
        let cut = &data[..data.len() - 3];
        assert_eq!(temporal_unit_starts(cut).unwrap(), starts, "{name}");
    }
}

/// The features each fixture exists for are really in it.
#[test]
fn fixtures_cover_their_features() {
    let altref = parse_stream("aom_altref", fixture("aom_altref"));
    assert!(
        altref
            .frames
            .iter()
            .any(|f| !f.header.show_frame && !f.header.show_existing_frame)
    );
    assert!(altref.frames.iter().any(|f| f.header.show_existing_frame));
    assert_eq!(altref.shown, 15);

    let tiles = parse_stream("aom_tiles", fixture("aom_tiles"));
    assert!(tiles.frames.iter().all(|f| f.header.show_existing_frame
        || (f.header.tile_info.tile_cols, f.header.tile_info.tile_rows) == (2, 2)));

    let groups = parse_stream("aom_tilegroups", fixture("aom_tilegroups"));
    assert!(groups.sequence.use_128x128_superblock);
    assert!(groups.sequence.decoder_model_info.is_some());
    assert!(groups.frames.iter().all(|f| f.tile_groups.len() == 2));
    assert!(
        groups
            .frames
            .iter()
            .any(|f| f.header.segmentation.segmentation_enabled)
    );
    assert!(
        groups
            .frames
            .iter()
            .any(|f| f.header.quantization.using_qmatrix)
    );
    assert!(
        groups
            .frames
            .iter()
            .any(|f| f.header.loop_restoration.uses_lr)
    );

    let ten = parse_stream("aom_10bit", fixture("aom_10bit"));
    assert_eq!(ten.sequence.seq_profile, 0);
    assert_eq!(ten.sequence.color_config.bit_depth, 10);

    for name in ["aom_grain", "svt_grain"] {
        let grain = parse_stream(name, fixture(name));
        assert!(grain.sequence.film_grain_params_present);
        assert!(grain.frames.iter().any(|f| f.header.film_grain.apply_grain));
    }
    let errres = parse_stream("aom_errres", fixture("aom_errres"));
    assert!(errres.sequence.frame_id_numbers_present_flag);
    assert!(
        errres
            .frames
            .iter()
            .any(|f| f.header.frame_type == super::frame_header::FrameType::Switch)
    );
    assert!(
        errres
            .frames
            .iter()
            .filter(|f| !f.header.frame_is_intra())
            .all(|f| f.header.error_resilient_mode)
    );

    let superres = parse_stream("aom_superres", fixture("aom_superres"));
    assert!(
        superres
            .frames
            .iter()
            .any(|f| f.header.frame_size.use_superres
                && f.header.frame_size.frame_width < f.header.frame_size.upscaled_width)
    );

    let resize = parse_stream("aom_resize", fixture("aom_resize"));
    assert!(
        resize
            .frames
            .iter()
            .any(|f| f.header.frame_size.frame_width != 320)
    );
    assert!(
        resize
            .frames
            .iter()
            .all(|f| f.header.frame_size.render_width == 320)
    );
    assert!(
        resize
            .frames
            .iter()
            .any(|f| f.header.frame_size.found_ref.is_some())
    );

    let mono = parse_stream("aom_mono", fixture("aom_mono"));
    assert!(mono.sequence.color_config.mono_chrome);
}

/// Annex B input is recognised and rejected, not misparsed.
#[test]
fn rejects_annex_b() {
    let err = iter_obus(ANNEX_B).next().unwrap().unwrap_err();
    assert!(err.to_string().contains("Annex B"), "{err}");
    assert!(temporal_unit_starts(ANNEX_B).is_err());
}

/// Starting after the key frame, or losing a frame, reports missing
/// references instead of failing, and the next key frame recovers.
#[test]
fn missing_references_wait_for_a_key_frame() {
    let data = fixture("aom_altref");
    let starts = temporal_unit_starts(data).unwrap();

    // Skip the first temporal unit's frames, keep its sequence header.
    let mut parser = Av1Parser::new();
    let mut missing = 0;
    let mut frames = 0;
    for obu in iter_obus(data) {
        let obu = obu.unwrap();
        if obu.offset < starts[1] && obu.header.obu_type != ObuType::SequenceHeader {
            continue;
        }
        match parser.parse_obu(&obu).unwrap() {
            Some(Av1Event::MissingReferences(slots)) => {
                assert_ne!(slots, 0);
                missing += 1;
            }
            Some(Av1Event::Frame(_)) => {
                parser.commit_frame(None).unwrap();
                frames += 1;
            }
            _ => {}
        }
    }
    assert!(missing > 0, "frames after the lost key frame are flagged");
    assert_eq!(
        frames, 0,
        "and none of them decode (there is one key frame)"
    );

    // Discarding a frame leaves the references it would have refreshed
    // empty; the stream continues once the slots are reset by a key frame.
    let mut parser = Av1Parser::new();
    let mut seen = 0;
    let mut flagged = false;
    for obu in iter_obus(data) {
        match parser.parse_obu(&obu.unwrap()).unwrap() {
            Some(Av1Event::Frame(_)) => {
                seen += 1;
                if seen == 2 {
                    parser.discard_frame();
                } else {
                    parser.commit_frame(None).unwrap();
                }
            }
            Some(Av1Event::MissingReferences(_)) => flagged = true,
            _ => {}
        }
    }
    assert!(flagged, "a frame needing the discarded one is flagged");
}

/// Corrupt input never panics: flip bits throughout every stream and parse
/// everything, ignoring the errors.
#[test]
fn corrupt_streams_do_not_panic() {
    for (_, data, _) in FIXTURES {
        for step in [7usize, 13, 29, 61] {
            let mut corrupt = data.to_vec();
            let mut i = step;
            while i < corrupt.len() {
                corrupt[i] ^= 1 << (i % 8);
                i += step * 17;
            }
            let mut parser = Av1Parser::new();
            for obu in iter_obus(&corrupt) {
                let Ok(obu) = obu else { break };
                if let Ok(Some(Av1Event::Frame(_))) = parser.parse_obu(&obu) {
                    parser.commit_frame(None).unwrap();
                }
            }
            for len in (0..corrupt.len()).step_by(step * 101) {
                let _ = temporal_unit_starts(&corrupt[..len]);
            }
        }
    }
}
