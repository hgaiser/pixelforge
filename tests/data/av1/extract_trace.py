"""Reduce ffmpeg trace_headers output for an AV1 stream to the per-frame
expectations the parser tests check (src/decoder/av1/stream_tests.rs).

One line per frame header ("F ...") and one per tile group ("T ...").

Usage:
    ffmpeg -i x.obu -c copy -bsf:v trace_headers -f null - 2> x.trace
    python3 extract_trace.py x.trace > x.expected

The fixtures were made from testsrc2 (320x240, 15 frames; aom_mono 176x144)
with ffmpeg n9.0, libaom and SVT-AV1 4.2:
    aom_altref:     ffmpeg -c:v libaom-av1 -cpu-used 8 -lag-in-frames 16 -auto-alt-ref 1 -f obu
    aom_tiles:      ffmpeg -c:v libaom-av1 -cpu-used 8 -tiles 2x2 -row-mt 0 -f obu
    aom_10bit:      ffmpeg -c:v libaom-av1 -cpu-used 8 -pix_fmt yuv420p10le -f obu
    aom_grain:      ffmpeg -c:v libaom-av1 -cpu-used 8 -denoise-noise-level 10 -f obu
    svt_grain:      ffmpeg -c:v libsvtav1 -preset 10 -svtav1-params film-grain=8 -f obu
    aom_errres:     aomenc --cpu-used=8 --limit=15 --obu --error-resilient=1 --sframe-dist=5 --sframe-mode=1
    aom_superres:   aomenc --cpu-used=8 --limit=15 --obu --superres-mode=1 --superres-denominator=12 --superres-kf-denominator=10
    aom_resize:     aomenc --cpu-used=8 --limit=15 --obu --resize-mode=1 --resize-denominator=12 --resize-kf-denominator=16
    aom_tilegroups: aomenc --cpu-used=8 --limit=12 --obu --tile-columns=1 --tile-rows=1 --num-tile-groups=2
                    --sb-size=128 --enable-qm=1 --aq-mode=1 --delta-lf-mode=1 --deltaq-mode=3
                    --lag-in-frames=0 --timing-info=model
    aom_mono:       aomenc --cpu-used=8 --limit=12 --obu --monochrome --enable-restoration=1
    aom_annexb:     aomenc --cpu-used=8 --limit=3 --obu --annexb=1
"""


import re
import sys

KEYS = {
    "show_existing_frame", "frame_to_show_map_idx", "frame_type", "show_frame",
    "showable_frame", "error_resilient_mode", "disable_cdf_update",
    "allow_screen_content_tools", "force_integer_mv", "current_frame_id",
    "frame_size_override_flag", "order_hint", "primary_ref_frame",
    "refresh_frame_flags", "use_superres", "coded_denom", "allow_intrabc",
    "allow_high_precision_mv", "interpolation_filter", "is_filter_switchable",
    "use_ref_frame_mvs", "disable_frame_end_update_cdf", "tile_cols_log2",
    "tile_rows_log2", "context_update_tile_id", "tile_size_bytes_minus1",
    "base_q_idx", "using_qmatrix", "qm_y", "qm_u", "qm_v", "segmentation_enabled",
    "segmentation_update_map", "segmentation_temporal_update",
    "segmentation_update_data", "delta_q_present", "delta_q_res",
    "delta_lf_present", "delta_lf_res", "delta_lf_multi", "loop_filter_sharpness",
    "loop_filter_delta_enabled", "loop_filter_delta_update", "cdef_bits",
    "cdef_damping_minus_3", "tx_mode", "reference_select", "skip_mode_present",
    "allow_warped_motion", "reduced_tx_set", "apply_grain", "grain_seed",
    "update_grain", "film_grain_params_ref_idx", "num_y_points", "num_cb_points",
    "num_cr_points", "frame_width_minus_1", "frame_height_minus_1",
    "render_width_minus_1", "render_height_minus_1", "lr_unit_shift",
    "lr_uv_shift",
}
INDEXED = {"ref_frame_idx", "loop_filter_level", "lr_type",
           "cdef_y_pri_strength", "cdef_y_sec_strength"}

LINE = re.compile(r"^\[trace_headers @ 0x[0-9a-f]+\] (.*)$")
ELEM = re.compile(r"^(\d+)\s+(\S+)\s+([01]+) = (-?\d+)$")


def main(path):
    out = []
    section = None
    obu_type = None
    obu_size = None
    frame = None
    tg = None

    def flush():
        nonlocal frame, tg
        if frame is not None:
            start, end, kv = frame
            kv.append(("header_bits", end - start))
            out.append("F obu=%d " % frame_obu + " ".join("%s=%s" % p for p in kv))
            frame = None
        if tg is not None:
            out.append("T obu=%d " % tg_obu + " ".join("%s=%s" % p for p in tg))
            tg = None

    frame_obu = tg_obu = 0
    for raw in open(path, errors="replace"):
        m = LINE.match(raw.rstrip("\n"))
        if not m:
            continue
        body = m.group(1).strip()
        e = ELEM.match(body)
        if not e:
            if body in ("Frame Header", "Tile Group", "Redundant Frame Header"):
                if body == "Tile Group":
                    flush_frame_only = frame is not None
                    if flush_frame_only:
                        start, end, kv = frame
                        kv.append(("header_bits", end - start))
                        out.append("F obu=%d " % frame_obu + " ".join("%s=%s" % p for p in kv))
                        frame = None
                    tg = []
                    tg_obu = obu_type
                elif body == "Frame Header":
                    frame = [None, None, []]
                    frame_obu = obu_type
                section = body
            elif body == "OBU header":
                flush()
                section = "OBU header"
            else:
                section = body
            continue
        pos, name, bits, value = int(e.group(1)), e.group(2), e.group(3), e.group(4)
        if section == "OBU header":
            if name == "obu_type":
                obu_type = int(value)
            if name == "obu_size":
                obu_size = int(value)
            continue
        if section == "Frame Header" and frame is not None:
            if name in ("zero_bit", "trailing_one_bit", "trailing_zero_bit"):
                continue
            if frame[0] is None:
                frame[0] = pos
            frame[1] = pos + len(bits)
            base = name.split("[")[0]
            if name in KEYS or base in INDEXED:
                frame[2].append((name, value))
        elif section == "Tile Group" and tg is not None:
            if name in ("tile_start_and_end_present_flag", "tg_start", "tg_end"):
                tg.append((name, value))
    flush()
    print("\n".join(out))


main(sys.argv[1])
