//! H.264 and H.265 level verification.
//!
//! Encodes streams that ask for 16 references and checks, with ffmpeg's
//! `trace_headers`, that the SPS gives a DPB that its level allows (H.264
//! A.3.1, H.265 A.4.2), and for H.265 the level that the stream's size and
//! frame rate need. Strict decoders (Apple VideoToolbox) stop on a stream
//! whose DPB breaks its level.
//!
//! Ignored by default: requires a Vulkan Video device and ffmpeg. Run with
//! `cargo test -- --ignored`.

#[allow(dead_code)]
mod common;

use pixelforge::{
    BitDepth, ChromaFormat, Codec, EncodeConfig, Encoder, Frame, RateControl, VideoContextBuilder,
    YuvLayout, YuvUploader,
};
use std::fs::File;
use std::io::Write;
use std::process::Command;

const FRAMES: u64 = 4;

struct Case {
    codec: Codec,
    width: u32,
    height: u32,
    frame_rate: u32,
    /// The level the SPS must give, as its `level_idc` or
    /// `general_level_idc`. `None` where it depends on the highest level the
    /// device encodes.
    level_idc: Option<u32>,
}

const fn case(codec: Codec, width: u32, height: u32, frame_rate: u32, level: Option<u32>) -> Case {
    Case {
        codec,
        width,
        height,
        frame_rate,
        level_idc: level,
    }
}

#[test]
#[ignore = "requires a Vulkan Video device and ffmpeg"]
fn levels() -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();

    let context = VideoContextBuilder::new()
        .app_name("Level verification")
        .enable_validation(cfg!(debug_assertions))
        .build()?;

    let dir = common::scratch_dir("levels");
    // H.265 `general_level_idc` is 30 times the level. 4K at 120 fps needs
    // level 5.2 for its sample rate.
    let cases = [
        case(Codec::H264, 1920, 1080, 120, None),
        case(Codec::H264, 3840, 2160, 60, None),
        case(Codec::H264, 3840, 2160, 120, None),
        case(Codec::H265, 1920, 1080, 120, Some(153)),
        case(Codec::H265, 2736, 1260, 120, Some(153)),
        case(Codec::H265, 3840, 2160, 120, Some(156)),
    ];

    let mut failures = Vec::new();
    for case in &cases {
        let name = format!(
            "{:?} {}x{}@{}",
            case.codec, case.width, case.height, case.frame_rate
        );
        if !context.supports_encode(case.codec) {
            println!("{name}: skipped (encode not supported)");
            continue;
        }
        match run_case(&context, &dir, case) {
            Ok(summary) => println!("{name}: PASS ({summary})"),
            Err(e) => {
                println!("{name}: FAIL: {e}");
                failures.push(format!("{name}: {e}"));
            }
        }
    }
    if !failures.is_empty() {
        return Err(format!("level verification failed: {}", failures.join("; ")).into());
    }
    Ok(())
}

/// Encode `case` and check its SPS. Returns what the SPS gives.
fn run_case(
    context: &pixelforge::VideoContext,
    dir: &std::path::Path,
    case: &Case,
) -> Result<String, Box<dyn std::error::Error>> {
    let output = dir.join(format!(
        "{:?}_{}x{}_{}.bin",
        case.codec, case.width, case.height, case.frame_rate
    ));

    let config = match case.codec {
        Codec::H264 => EncodeConfig::h264(case.width, case.height),
        Codec::H265 => EncodeConfig::h265(case.width, case.height),
        Codec::AV1 => return Err("AV1 has no DPB in its levels".into()),
    }
    .with_rate_control(RateControl::Cqp { qp: 30 })
    .with_chroma_format(ChromaFormat::Yuv420)
    .with_bit_depth(BitDepth::Eight)
    .with_frame_rate(case.frame_rate, 1)
    .with_gop_size(0)
    .with_max_reference_frames(16)
    .with_active_reference_frames(1);
    let mut encoder = Encoder::new(context.clone(), config)?;
    let mut uploader = YuvUploader::for_encoder(&encoder)?;

    let luma = (case.width * case.height) as usize;
    let chroma = (case.width.div_ceil(2) * case.height.div_ceil(2)) as usize;
    let mut file = File::create(&output)?;
    for i in 0..FRAMES {
        // A flat picture whose brightness changes, so each frame predicts.
        let mut yuv = vec![(64 + 32 * i) as u8; luma];
        yuv.resize(luma + 2 * chroma, 128);
        let slot = encoder.input_slot()?;
        uploader.upload_to(&slot, YuvLayout::I420, &yuv)?;
        let packet = pollster::block_on(encoder.submit(Frame::input(slot))?)?;
        file.write_all(&packet.data)?;
    }
    encoder.flush()?;
    drop(file);

    let trace = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "trace", "-i"])
        .arg(&output)
        .args(["-c", "copy", "-bsf:v", "trace_headers", "-f", "null", "-"])
        .output()?;
    let log = String::from_utf8_lossy(&trace.stderr);

    let (level_name, dpb_name) = match case.codec {
        Codec::H264 => ("level_idc", "max_dec_frame_buffering"),
        _ => ("general_level_idc", "sps_max_dec_pic_buffering_minus1[0]"),
    };
    let level_idc = traced_value(&log, level_name).ok_or("no level in the trace")?;
    let dpb = traced_value(&log, dpb_name).ok_or("no DPB size in the trace")?;
    let summary = format!("{level_name} {level_idc}, {dpb_name} {dpb}");

    if let Some(expected) = case.level_idc
        && level_idc != expected
    {
        return Err(format!("{summary}: expected level_idc {expected}").into());
    }
    let allowed = match case.codec {
        Codec::H264 => h264_max_dpb_frames(case.width, case.height, level_idc),
        _ => h265_max_dpb_size(case.width, case.height, level_idc).map(|size| size - 1),
    }
    .ok_or_else(|| format!("{summary}: unknown level"))?;
    if dpb > allowed {
        return Err(format!("{summary}: the level allows {allowed}").into());
    }
    Ok(summary)
}

/// H.264 `MaxDpbFrames` (A.3.1) for a `level_idc` of 10 times the level.
fn h264_max_dpb_frames(width: u32, height: u32, level_idc: u32) -> Option<u32> {
    let max_dpb_mbs = match level_idc {
        40 => 32_768,
        41 => 32_768,
        42 => 34_816,
        50 => 110_400,
        51 | 52 => 184_320,
        60..=62 => 696_320,
        _ => return None,
    };
    let frame_mbs = width.div_ceil(16) * height.div_ceil(16);
    Some((max_dpb_mbs / frame_mbs).min(16))
}

/// H.265 `MaxDpbSize` (A.4.2) for a `general_level_idc` of 30 times the
/// level, with the picture aligned to the encoder's 32x32 coding blocks.
fn h265_max_dpb_size(width: u32, height: u32, level_idc: u32) -> Option<u32> {
    let max_luma_ps: u64 = match level_idc {
        120 | 123 => 2_228_224,
        150 | 153 | 156 => 8_912_896,
        180 | 183 | 186 => 35_651_584,
        _ => return None,
    };
    let luma_ps = u64::from(width.next_multiple_of(32)) * u64::from(height.next_multiple_of(32));
    Some(if luma_ps <= max_luma_ps >> 2 {
        16
    } else if luma_ps <= max_luma_ps >> 1 {
        12
    } else if luma_ps <= (3 * max_luma_ps) >> 2 {
        8
    } else {
        6
    })
}

/// The value of the last `name` in a `trace_headers` log. Its lines end with
/// `name <bits> = <value>`.
fn traced_value(log: &str, name: &str) -> Option<u32> {
    log.lines().rev().find_map(|line| {
        let (left, value) = line.rsplit_once('=')?;
        left.split_whitespace()
            .any(|word| word == name)
            .then(|| value.trim().parse().ok())?
    })
}
