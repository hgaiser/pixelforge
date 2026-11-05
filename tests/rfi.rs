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
//! Ignored by default: requires a Vulkan Video device and ffmpeg. Run with
//! `cargo test -- --ignored`.

#[allow(dead_code)]
mod common;

use pixelforge::{
    BitDepth, ChromaFormat, Codec, EncodeConfig, Encoder, Frame, RateControl, VideoContextBuilder,
    YuvLayout, YuvUploader,
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
        match run_codec(&context, &dir, codec, &input_path) {
            Ok(()) => {}
            Err(e) => {
                println!("{codec:?}: FAIL: {e}");
                failures.push(format!("{codec:?}: {e}"));
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
    input_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let ext = if codec == Codec::AV1 { "obu" } else { "bin" };
    let output_filename = dir
        .join(format!("output_rfi_{codec:?}.{ext}"))
        .to_string_lossy()
        .into_owned();
    let decoded_filename = dir
        .join(format!("decoded_rfi_{codec:?}.yuv"))
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

    let mut encoder = Encoder::new(context.clone(), config)?;
    let mut input_image = YuvUploader::for_encoder(&encoder)?;

    let mut yuv_data = Vec::new();
    File::open(input_path)?.read_to_end(&mut yuv_data)?;
    let frame_size = (WIDTH * HEIGHT * 3 / 2) as usize;

    let mut output_file = File::create(&output_filename)?;
    let mut pending: VecDeque<pixelforge::EncodeFuture> = VecDeque::new();
    let mut recovery_is_key: Option<bool> = None;

    let drain_one = |pending: &mut VecDeque<pixelforge::EncodeFuture>,
                     output_file: &mut File,
                     recovery_is_key: &mut Option<bool>|
     -> Result<(), Box<dyn std::error::Error>> {
        let packet = pollster::block_on(pending.pop_front().unwrap())?;
        if packet.pts == INVALIDATE_AT {
            *recovery_is_key = Some(packet.is_key_frame());
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
        while pending.len() > 2 {
            drain_one(&mut pending, &mut output_file, &mut recovery_is_key)?;
        }
    }

    encoder.flush()?;
    while !pending.is_empty() {
        drain_one(&mut pending, &mut output_file, &mut recovery_is_key)?;
    }
    drop(output_file);

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

    let psnr = decode_and_psnr(&output_filename, &decoded_filename, input_path)?;
    std::fs::remove_file(&output_filename).ok();
    std::fs::remove_file(&decoded_filename).ok();

    if psnr < MIN_PSNR {
        return Err(format!(
            "post-recovery PSNR {psnr:.2} dB below {MIN_PSNR} dB (decoder desynced)"
        )
        .into());
    }

    println!("{codec:?}: PASS — recovered with a P-frame, full-stream PSNR {psnr:.2} dB");
    Ok(())
}

/// Decode the bitstream to raw YUV and return its PSNR against the source.
fn decode_and_psnr(
    bitstream: &str,
    decoded: &str,
    source: &str,
) -> Result<f64, Box<dyn std::error::Error>> {
    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            bitstream,
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
            "psnr",
            "-f",
            "null",
            "-",
        ])
        .output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    let pos = stderr
        .find("average:")
        .ok_or_else(|| format!("could not parse PSNR: {stderr}"))?;
    let rest = &stderr[pos + 8..];
    let end = rest.find(' ').unwrap_or(rest.len());
    Ok(rest[..end].parse()?)
}
