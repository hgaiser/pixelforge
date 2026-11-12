//! Reference Frame Invalidation (RFI) verification.
//!
//! Encodes a stream and, partway through, simulates packet loss by calling
//! [`Encoder::invalidate_reference_frames`] for the two most recent references.
//! Because the encoder keeps a window of references (default 4), older ones
//! survive, so the next frame recovers by predicting from a surviving reference
//! instead of emitting a full keyframe.
//!
//! Correctness is checked two ways for each codec:
//!  - the recovery frame must NOT be a keyframe (proving RFI engaged rather than
//!    falling back to an IDR), and
//!  - ffmpeg must decode the whole stream with high PSNR (proving the encoder's
//!    reference re-signaling — H.264 MMCO, H.265 RPS, AV1 ref mapping — keeps the
//!    decoder's DPB in sync with the encoder's).
//!
//! Each codec runs without B-frames, and with a pyramid of three B-frames
//! where the device has one (flat B-frames where it has only those). With
//! B-frames, the recovery is checked by the absence of any keyframe after the
//! first.
//!
//! Ignored by default: requires a Vulkan Video device and ffmpeg. Run with
//! `cargo test -- --ignored`.

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
const FRAMES: u64 = 30;
/// Frame at which we simulate loss and invalidate references.
const INVALIDATE_AT: u64 = 12;
/// Number of most-recent references to invalidate (must stay below the encoder's
/// reference window so older references survive and recovery stays a P-frame).
const INVALIDATE_COUNT: u64 = 2;
/// PSNR below this means the decoder desynced from the encoder after recovery.
const MIN_PSNR: f64 = 30.0;
/// How far below the worst frame before the loss a frame may be. B-frames
/// have a higher QP than the other frames, so this cannot be very small.
const MAX_DROP: f64 = 3.0;

#[test]
#[ignore = "requires a Vulkan Video device and ffmpeg"]
fn rfi() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    println!("PixelForge Reference Frame Invalidation verification\n");

    let input_path = format!("testdata/test_frames_{WIDTH}x{HEIGHT}_yuv420p.yuv");
    common::ensure_test_data(WIDTH, HEIGHT, "yuv420p", &input_path)?;

    let dir = common::scratch_dir("rfi");

    let context = VideoContextBuilder::new()
        .app_name("RFI Verification")
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
        let runs = std::iter::once(None).chain(b_frames.map(Some));
        for references in runs {
            let name = match references {
                None => format!("{codec:?}"),
                Some(references) => format!("{codec:?} B-frames {references:?}"),
            };
            if let Err(e) = run_codec(&context, &dir, codec, references, &input_path) {
                println!("{name}: FAIL: {e}");
                failures.push(format!("{name}: {e}"));
            }
        }
    }

    if !failures.is_empty() {
        return Err(format!("RFI verification failed: {}", failures.join("; ")).into());
    }
    Ok(())
}

fn run_codec(
    context: &pixelforge::VideoContext,
    dir: &std::path::Path,
    codec: Codec,
    b_frames: Option<BFrameReferences>,
    input_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let ext = if codec == Codec::AV1 { "obu" } else { "bin" };
    let suffix = if b_frames.is_some() { "_b" } else { "" };
    let output_filename = dir
        .join(format!("output_rfi_{codec:?}{suffix}.{ext}"))
        .to_string_lossy()
        .into_owned();
    let decoded_filename = dir
        .join(format!("decoded_rfi_{codec:?}{suffix}.yuv"))
        .to_string_lossy()
        .into_owned();

    let config = match codec {
        Codec::H264 => EncodeConfig::h264(WIDTH, HEIGHT),
        Codec::H265 => EncodeConfig::h265(WIDTH, HEIGHT),
        Codec::AV1 => EncodeConfig::av1(WIDTH, HEIGHT),
    }
    .with_rate_control(RateControl::Cqp { qp: 10 })
    .with_chroma_format(ChromaFormat::Yuv420)
    .with_bit_depth(BitDepth::Eight)
    // Infinite GOP so the recovery can only come from RFI, never a periodic IDR.
    .with_gop_size(0);
    let config = match b_frames {
        Some(references) => config.with_b_frames(3).with_b_frame_references(references),
        None => config,
    };
    // At most `b_frames` frames wait for their group, so with more futures
    // pending than that the oldest is ready (this also checks that claim).
    let held = config.b_frames.max(2) as usize;

    let mut encoder = Encoder::new(context.clone(), config)?;
    let mut input_image = YuvUploader::for_encoder(&encoder)?;

    let mut yuv_data = Vec::new();
    File::open(input_path)?.read_to_end(&mut yuv_data)?;
    let frame_size = (WIDTH * HEIGHT * 3 / 2) as usize;

    let mut output_file = File::create(&output_filename)?;
    let mut pending: VecDeque<pixelforge::EncodeFuture> = VecDeque::new();
    let mut recovery_is_key: Option<bool> = None;
    let mut later_key_frames = 0;

    let mut drain_one = |pending: &mut VecDeque<pixelforge::EncodeFuture>,
                         output_file: &mut File,
                         recovery_is_key: &mut Option<bool>|
     -> Result<(), Box<dyn std::error::Error>> {
        let packet = pollster::block_on(pending.pop_front().unwrap())?;
        if packet.pts == INVALIDATE_AT {
            *recovery_is_key = Some(packet.is_key_frame());
        }
        if packet.pts > 0 && packet.is_key_frame() {
            later_key_frames += 1;
        }
        output_file.write_all(&packet.data)?;
        Ok(())
    };

    for i in 0..FRAMES {
        let start = (i as usize) * frame_size;
        let end = start + frame_size;
        if end > yuv_data.len() {
            break;
        }

        // Simulate the client losing the `INVALIDATE_COUNT` most recent frames
        // just before encoding frame `INVALIDATE_AT`.
        if i == INVALIDATE_AT {
            let first_lost = INVALIDATE_AT - INVALIDATE_COUNT;
            encoder.invalidate_reference_frames(first_lost);
        }

        let slot = encoder.input_slot()?;
        input_image.upload_to(&slot, YuvLayout::I420, &yuv_data[start..end])?;
        pending.push_back(encoder.submit(Frame::input(slot))?);
        while pending.len() > held {
            drain_one(&mut pending, &mut output_file, &mut recovery_is_key)?;
        }
    }

    encoder.flush()?;
    while !pending.is_empty() {
        drain_one(&mut pending, &mut output_file, &mut recovery_is_key)?;
    }
    drop(output_file);
    drop(drain_one);

    if later_key_frames > 0 {
        return Err(
            format!("{later_key_frames} keyframes after the first (RFI did not engage)").into(),
        );
    }
    match recovery_is_key {
        Some(true) => {
            return Err(format!(
                "recovery frame {INVALIDATE_AT} fell back to a keyframe (RFI did not engage)"
            )
            .into());
        }
        None => return Err(format!("recovery frame {INVALIDATE_AT} was never produced").into()),
        Some(false) => {}
    }

    let frames = decode_and_psnr(&output_filename, &decoded_filename, input_path)?;
    std::fs::remove_file(&output_filename).ok();
    std::fs::remove_file(&decoded_filename).ok();

    let psnr = frames.iter().sum::<f64>() / frames.len() as f64;
    if psnr < MIN_PSNR {
        return Err(format!(
            "post-recovery PSNR {psnr:.2} dB below {MIN_PSNR} dB (decoder desynced)"
        )
        .into());
    }
    // A decoder that predicts from other pictures than the encoder did drifts
    // from there on, by a few dB at first. The frames before the loss say
    // what the quality is.
    let before = &frames[..INVALIDATE_AT as usize - INVALIDATE_COUNT as usize];
    let worst_before = before.iter().copied().fold(f64::INFINITY, f64::min);
    for (index, frame) in frames.iter().enumerate() {
        if *frame < worst_before - MAX_DROP {
            return Err(format!(
                "frame {index} has PSNR {frame:.2} dB, more than {MAX_DROP} dB below \
                 the worst frame before the loss ({worst_before:.2} dB): the decoder desynced"
            )
            .into());
        }
    }

    println!(
        "{codec:?} {b_frames:?}: PASS — recovered with a P-frame, full-stream PSNR {psnr:.2} dB"
    );
    Ok(())
}

/// Decode the bitstream to raw YUV and return the PSNR of each frame against
/// the source.
fn decode_and_psnr(
    bitstream: &str,
    decoded: &str,
    source: &str,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    // The device may code a frame larger than the display size (AV1's
    // `codedPictureAlignment`); compare only the display size.
    let crop = format!("crop={WIDTH}:{HEIGHT}:0:0");
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            bitstream,
            "-vf",
            &crop,
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            decoded,
        ])
        .output()?;
    if !status.status.success() {
        return Err(format!(
            "ffmpeg decode failed: {}",
            String::from_utf8_lossy(&status.stderr)
        )
        .into());
    }

    let size = format!("{WIDTH}x{HEIGHT}");
    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "info",
            "-s",
            &size,
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "-i",
            source,
            "-s",
            &size,
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "-i",
            decoded,
            "-lavfi",
            "psnr=stats_file=-",
            "-f",
            "null",
            "-",
        ])
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "ffmpeg psnr failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    // One line per frame: `n:1 mse_avg:... psnr_avg:51.23 ...`.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let frames: Vec<f64> = stdout
        .lines()
        .filter_map(|line| {
            let value = line
                .split_whitespace()
                .find_map(|f| f.strip_prefix("psnr_avg:"))?;
            // A frame equal to the source has "inf".
            Some(value.parse::<f64>().unwrap_or(f64::INFINITY).min(100.0))
        })
        .collect();
    if frames.is_empty() {
        return Err(format!("could not parse PSNR: {stdout}").into());
    }
    Ok(frames)
}
