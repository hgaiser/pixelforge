//! Changing the bitrate of a running encoder.
//!
//! Each codec encodes three phases of [`PHASE`] frames with constant bitrate:
//! [`HIGH`], [`LOW`], then [`HIGH`] again, changed with
//! [`Encoder::set_rate_control`]. The low phase has to come out much smaller
//! than the high phases on either side, with no key frame after the first
//! (the change needs no IDR), and ffmpeg has to decode the stream with
//! `-err_detect explode`. Each codec runs without B-frames and with the
//! B-frames the device has.
//!
//! Run it under validation too (`PIXELFORGE_VALIDATION=1`, with
//! `--nocapture`): begin coding has to state the session's rate control as
//! it was before the change, and only the layer reports a mistake there.
//!
//! Ignored by default: requires a Vulkan Video device and ffmpeg. Run with
//! `cargo test --test rate_control -- --ignored`.

#[allow(dead_code)]
mod common;

use pixelforge::{
    BFrameReferences, BFrameSupport, BitDepth, ChromaFormat, Codec, EncodeConfig, Encoder, Frame,
    RateControl, VideoContextBuilder, YuvLayout, YuvUploader,
};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Write};
use std::process::Command;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
/// Frames per phase.
const PHASE: u64 = 40;
/// Frames at the start of a phase not counted, while the rate control
/// settles on the new bitrate.
const SETTLE: u64 = 10;
/// More than the test frames need: the high phases come out below it, at
/// what the content takes.
const HIGH: u32 = 1_000_000;
/// Less than the test frames need, so the low phase comes out at it.
const LOW: u32 = 50_000;
const FRAME_RATE: u32 = 30;
/// The low phase has to be at most this share of each high phase.
const MAX_LOW_SHARE: f64 = 0.5;
/// And at most this many times what [`LOW`] allows per frame.
const MAX_LOW_OVERSHOOT: f64 = 1.5;

#[test]
#[ignore = "requires a Vulkan Video device and ffmpeg"]
fn rate_control_changes() -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();

    let input_path = format!("testdata/test_frames_{WIDTH}x{HEIGHT}_yuv420p.yuv");
    common::ensure_test_data(WIDTH, HEIGHT, "yuv420p", &input_path)?;
    let mut yuv = Vec::new();
    File::open(&input_path)?.read_to_end(&mut yuv)?;

    let dir = common::scratch_dir("rate_control");
    let context = VideoContextBuilder::new()
        .app_name("Rate control changes")
        .enable_validation(cfg!(debug_assertions))
        .build()?;

    let mut failures = Vec::new();
    for codec in [Codec::H264, Codec::H265, Codec::AV1] {
        if !context.supports_encode(codec) {
            println!("{codec:?}: skipped (encode not supported)");
            continue;
        }
        let b_frames = match context.supports_b_frames(codec, ChromaFormat::Yuv420, BitDepth::Eight)
        {
            BFrameSupport::None => None,
            BFrameSupport::Flat => Some(BFrameReferences::Flat),
            BFrameSupport::Pyramid => Some(BFrameReferences::Pyramid),
        };
        for references in std::iter::once(None).chain(b_frames.map(Some)) {
            let name = match references {
                None => format!("{codec:?}"),
                Some(references) => format!("{codec:?} B-frames {references:?}"),
            };
            match run(&context, &dir, codec, references, &yuv) {
                Ok(sizes) => println!(
                    "{name}: PASS, bytes per frame {:.0} -> {:.0} -> {:.0}",
                    sizes[0], sizes[1], sizes[2]
                ),
                Err(e) => {
                    println!("{name}: FAIL: {e}");
                    failures.push(format!("{name}: {e}"));
                }
            }
        }
    }
    if !failures.is_empty() {
        return Err(failures.join("; ").into());
    }
    Ok(())
}

/// Encode the three phases, and return the bytes per frame of each.
fn run(
    context: &pixelforge::VideoContext,
    dir: &std::path::Path,
    codec: Codec,
    b_frames: Option<BFrameReferences>,
    yuv: &[u8],
) -> Result<[f64; 3], Box<dyn std::error::Error>> {
    let config = match codec {
        Codec::H264 => EncodeConfig::h264(WIDTH, HEIGHT),
        Codec::H265 => EncodeConfig::h265(WIDTH, HEIGHT),
        Codec::AV1 => EncodeConfig::av1(WIDTH, HEIGHT),
    }
    .with_rate_control(RateControl::Cbr { bitrate: HIGH })
    .with_frame_rate(FRAME_RATE, 1)
    // No periodic key frame, so any key frame after the first is the change's.
    .with_gop_size(0);
    let config = match b_frames {
        Some(references) => config.with_b_frames(3).with_b_frame_references(references),
        None => config,
    };
    let held = config.b_frames.max(2) as usize;
    let mut encoder = Encoder::new(context.clone(), config)?;
    let mut upload = YuvUploader::for_encoder(&encoder)?;

    // The mode cannot change, and the peak cannot go above the first.
    if encoder
        .set_rate_control(RateControl::Cqp { qp: 30 })
        .is_ok()
    {
        return Err("a change from CBR to CQP was accepted".into());
    }
    if encoder
        .set_rate_control(RateControl::Cbr { bitrate: HIGH + 1 })
        .is_ok()
    {
        return Err("a bitrate above the first was accepted".into());
    }

    let suffix = if b_frames.is_some() { "_b" } else { "" };
    let ext = if codec == Codec::AV1 { "obu" } else { "bin" };
    let path = dir.join(format!("rate_control_{codec:?}{suffix}.{ext}"));
    let mut output = File::create(&path)?;

    let frame_size = (WIDTH * HEIGHT * 3 / 2) as usize;
    let source_frames = (yuv.len() / frame_size) as u64;
    let mut bytes = [0usize; 3];
    let mut later_key_frames = 0;
    let mut pending: VecDeque<pixelforge::EncodeFuture> = VecDeque::new();
    let mut drain = |pending: &mut VecDeque<pixelforge::EncodeFuture>,
                     output: &mut File|
     -> Result<(), Box<dyn std::error::Error>> {
        let packet = pollster::block_on(pending.pop_front().unwrap())?;
        if packet.pts > 0 && packet.is_key_frame() {
            later_key_frames += 1;
        }
        if packet.pts % PHASE >= SETTLE {
            bytes[(packet.pts / PHASE) as usize] += packet.data.len();
        }
        output.write_all(&packet.data)?;
        Ok(())
    };

    for i in 0..3 * PHASE {
        match i {
            i if i == PHASE => encoder.set_rate_control(RateControl::Cbr { bitrate: LOW })?,
            i if i == 2 * PHASE => encoder.set_rate_control(RateControl::Cbr { bitrate: HIGH })?,
            _ => {}
        }
        // Play the source forward and back, so the motion never jumps.
        let index = i % (2 * source_frames);
        let index = if index < source_frames {
            index
        } else {
            2 * source_frames - 1 - index
        };
        let start = index as usize * frame_size;
        let slot = encoder.input_slot()?;
        upload.upload_to(&slot, YuvLayout::I420, &yuv[start..start + frame_size])?;
        pending.push_back(encoder.submit(Frame::input(slot))?);
        while pending.len() > held {
            drain(&mut pending, &mut output)?;
        }
    }
    encoder.flush()?;
    while !pending.is_empty() {
        drain(&mut pending, &mut output)?;
    }
    drop(drain);
    drop(output);

    if encoder.rate_control() != (RateControl::Cbr { bitrate: HIGH }) {
        return Err(format!("rate control reads {:?}", encoder.rate_control()).into());
    }
    if later_key_frames > 0 {
        return Err(format!("{later_key_frames} key frames after the first").into());
    }
    let counted = (PHASE - SETTLE) as f64;
    let sizes = bytes.map(|b| b as f64 / counted);
    if sizes[1] > MAX_LOW_SHARE * sizes[0] || sizes[1] > MAX_LOW_SHARE * sizes[2] {
        return Err(format!(
            "bytes per frame {:.0} -> {:.0} -> {:.0}: the low phase is not below \
             {MAX_LOW_SHARE} of the others",
            sizes[0], sizes[1], sizes[2]
        )
        .into());
    }
    let low_budget = f64::from(LOW) / 8.0 / f64::from(FRAME_RATE);
    if sizes[1] > MAX_LOW_OVERSHOOT * low_budget {
        return Err(format!(
            "the low phase has {:.0} bytes per frame, more than {MAX_LOW_OVERSHOOT} times \
             the {low_budget:.0} its bitrate allows",
            sizes[1]
        )
        .into());
    }

    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-err_detect",
            "explode",
            "-i",
        ])
        .arg(&path)
        .args(["-f", "null", "-"])
        .output()?;
    if !status.status.success() {
        return Err(format!(
            "ffmpeg failed to decode: {}",
            String::from_utf8_lossy(&status.stderr)
        )
        .into());
    }
    std::fs::remove_file(&path).ok();
    Ok(sizes)
}
