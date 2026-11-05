//! Verify every encoding combination the encoder claims to support.
//!
//! Runs H.264/H.265/AV1 across 8-bit/10-bit and YUV420/YUV444, decodes each
//! result with ffmpeg, and checks the PSNR against the source. Combinations the
//! device does not support are reported and skipped.
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
use std::io::{Read, Write};
use std::process::Command;
use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const FRAMES: u32 = 30;
/// PSNR floor for a combination that actually ran. Observed values are 58+ dB;
/// this only catches a genuine breakage, not implementation differences.
const MIN_PSNR: f64 = 30.0;

#[test]
#[ignore = "requires a Vulkan Video device and ffmpeg"]
fn verify_all() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing.
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer().with_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
            ),
        )
        .init();

    // Ensure test data exists (dimensions encoded in filename to avoid stale data
    // when switching between branches with different WIDTH/HEIGHT constants).
    let yuv420_path = format!("testdata/test_frames_{WIDTH}x{HEIGHT}_yuv420p.yuv");
    let yuv444_path = format!("testdata/test_frames_{WIDTH}x{HEIGHT}_yuv444p.yuv");
    common::ensure_test_data(WIDTH, HEIGHT, "yuv420p", &yuv420_path)?;
    common::ensure_test_data(WIDTH, HEIGHT, "yuv444p", &yuv444_path)?;

    let dir = common::scratch_dir("verify_all");

    let combinations = [
        (Codec::H264, BitDepth::Eight, ChromaFormat::Yuv420),
        (Codec::H264, BitDepth::Eight, ChromaFormat::Yuv444),
        (Codec::H264, BitDepth::Ten, ChromaFormat::Yuv420),
        (Codec::H264, BitDepth::Ten, ChromaFormat::Yuv444),
        (Codec::H265, BitDepth::Eight, ChromaFormat::Yuv420),
        (Codec::H265, BitDepth::Eight, ChromaFormat::Yuv444),
        (Codec::H265, BitDepth::Ten, ChromaFormat::Yuv420),
        (Codec::H265, BitDepth::Ten, ChromaFormat::Yuv444),
        (Codec::AV1, BitDepth::Eight, ChromaFormat::Yuv420),
        (Codec::AV1, BitDepth::Eight, ChromaFormat::Yuv444),
        (Codec::AV1, BitDepth::Ten, ChromaFormat::Yuv420),
        (Codec::AV1, BitDepth::Ten, ChromaFormat::Yuv444),
    ];

    let context = VideoContextBuilder::new()
        .app_name("Verify All")
        .enable_validation(true) // Enable validation for debugging
        .build()?;

    let mut ran = 0usize;
    for (codec, depth, format) in combinations {
        println!("Testing {codec:?} {depth:?} {format:?}...");

        if !context.supports_encode(codec) {
            println!("  Skipping: codec not supported");
            continue;
        }

        // `supports_encode` only checks the codec, not the profile and format,
        // so an encoder-creation failure with NOT_SUPPORTED means this device
        // does not expose that combination.
        match run_test(&context, &dir, codec, depth, format) {
            Ok(psnr) => {
                assert!(
                    psnr >= MIN_PSNR,
                    "{codec:?} {depth:?} {format:?}: PSNR {psnr:.2} dB below the {MIN_PSNR} dB floor"
                );
                println!("  PASS: PSNR = {psnr:.2} dB");
                ran += 1;
            }
            Err(e) if e.to_string().contains("NOT_SUPPORTED") => {
                println!("  SKIP: unsupported on this device: {e}");
            }
            Err(e) => return Err(format!("{codec:?} {depth:?} {format:?}: {e}").into()),
        }
        println!("------------------------------------------------");
    }

    assert!(ran > 0, "no encoding combination ran on this device");
    Ok(())
}

fn run_test(
    context: &pixelforge::VideoContext,
    dir: &std::path::Path,
    codec: Codec,
    depth: BitDepth,
    format: ChromaFormat,
) -> Result<f64, Box<dyn std::error::Error>> {
    // AV1 uses .obu extension for raw OBU streams (with temporal delimiters).
    // H.264/H.265 use .bin for raw Annex B bitstreams.
    let output_ext = if codec == Codec::AV1 { "obu" } else { "bin" };
    let output_filename = dir
        .join(format!(
            "output_{codec:?}_{depth:?}_{format:?}.{output_ext}"
        ))
        .to_string_lossy()
        .into_owned();
    let decoded_filename = dir
        .join(format!("decoded_{codec:?}_{depth:?}_{format:?}.yuv"))
        .to_string_lossy()
        .into_owned();

    // 1. Encode
    {
        let config = match codec {
            Codec::H264 => EncodeConfig::h264(WIDTH, HEIGHT),
            Codec::H265 => EncodeConfig::h265(WIDTH, HEIGHT),
            Codec::AV1 => EncodeConfig::av1(WIDTH, HEIGHT),
        }
        .with_rate_control(RateControl::Cqp { qp: 10 })
        .with_chroma_format(format)
        .with_bit_depth(depth);

        let mut encoder = match Encoder::new(context.clone(), config) {
            Ok(e) => e,
            Err(e) => return Err(format!("Failed to create encoder: {}", e).into()),
        };

        let mut input_image = YuvUploader::for_encoder(&encoder)?;

        let input_path = match format {
            ChromaFormat::Yuv420 => {
                format!("testdata/test_frames_{}x{}_yuv420p.yuv", WIDTH, HEIGHT)
            }
            ChromaFormat::Yuv444 => {
                format!("testdata/test_frames_{}x{}_yuv444p.yuv", WIDTH, HEIGHT)
            }
            _ => return Err("Unsupported format".into()),
        };

        let mut yuv_data = Vec::new();
        File::open(&input_path)?.read_to_end(&mut yuv_data)?;

        let frame_size = match format {
            ChromaFormat::Yuv420 => (WIDTH * HEIGHT * 3 / 2) as usize,
            ChromaFormat::Yuv444 => (WIDTH * HEIGHT * 3) as usize,
            _ => return Err("Unsupported format".into()),
        };

        let mut output_file = File::create(&output_filename)?;

        // Futures for in-flight encodes, drained in submission order.
        let mut pending: std::collections::VecDeque<pixelforge::EncodeFuture> =
            std::collections::VecDeque::new();

        for i in 0..FRAMES {
            let start = (i as usize) * frame_size;
            let end = start + frame_size;
            if end > yuv_data.len() {
                break;
            }
            let frame = &yuv_data[start..end];

            // Upload directly to encoder's input image to avoid cross-queue
            // copy issues (YuvUploader uses the transfer queue, encoder uses the
            // video encode queue which doesn't support transfer ops).
            let slot = encoder.input_slot()?;
            let layout = match format {
                ChromaFormat::Yuv420 => YuvLayout::I420,
                ChromaFormat::Yuv444 => YuvLayout::I444,
                _ => return Err("Unsupported format".into()),
            };
            input_image.upload_to(&slot, layout, frame)?;

            pending.push_back(encoder.submit(Frame::input(slot))?);
            while pending.len() > 2 {
                let packet = pollster::block_on(pending.pop_front().unwrap())?;
                output_file.write_all(&packet.data)?;
            }
        }

        // Barrier, then drain the outstanding futures in submission order.
        encoder.flush()?;
        while let Some(future) = pending.pop_front() {
            let packet = pollster::block_on(future)?;
            output_file.write_all(&packet.data)?;
        }
    }

    // 2. Decode to raw YUV
    // We need to specify the output pixel format for ffmpeg to write rawvideo.
    // For 8-bit: yuv420p or yuv444p
    // For 10-bit: yuv420p10le or yuv444p10le
    // Note: The encoder output is H.264/H.265. ffmpeg should auto-detect input format.
    // But we need to force output format to match what we want to compare against.
    // Actually, we should decode to the SAME format as the input for PSNR comparison.
    // Input was 8-bit yuv420p or yuv444p.
    // Even if we encoded as 10-bit, we fed it 8-bit data (expanded).
    // So we should decode to 8-bit to compare with original 8-bit source.
    // OR, we decode to whatever it is, and let PSNR filter handle format conversion if needed.
    // But PSNR filter needs same resolution and format usually.

    // Let's decode to the input format (8-bit).
    let (input_pix_fmt, input_path) = match format {
        ChromaFormat::Yuv420 => (
            "yuv420p",
            format!("testdata/test_frames_{}x{}_yuv420p.yuv", WIDTH, HEIGHT),
        ),
        ChromaFormat::Yuv444 => (
            "yuv444p",
            format!("testdata/test_frames_{}x{}_yuv444p.yuv", WIDTH, HEIGHT),
        ),
        _ => return Err("Unsupported format".into()),
    };

    let status = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            &output_filename,
            "-pix_fmt",
            input_pix_fmt,
            "-f",
            "rawvideo",
            &decoded_filename,
        ])
        .output()?;

    if !status.status.success() {
        return Err(format!(
            "FFmpeg decode failed: {:?}",
            String::from_utf8_lossy(&status.stderr)
        )
        .into());
    }

    // 3. PSNR
    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "info",
            "-s",
            &format!("{}x{}", WIDTH, HEIGHT),
            "-pix_fmt",
            input_pix_fmt,
            "-f",
            "rawvideo",
            "-i",
            &input_path,
            "-s",
            &format!("{}x{}", WIDTH, HEIGHT),
            "-pix_fmt",
            input_pix_fmt,
            "-f",
            "rawvideo",
            "-i",
            &decoded_filename,
            "-lavfi",
            "psnr",
            "-f",
            "null",
            "-",
        ])
        .output()?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Parse PSNR from stderr. Look for "average:".
    // Output example: "PSNR y:30.12 u:32.34 v:33.45 average:31.23 min:..."

    if let Some(pos) = stderr.find("average:") {
        let rest = &stderr[pos + 8..];
        let end = rest.find(' ').unwrap_or(rest.len());
        let psnr_str = &rest[..end];
        let psnr: f64 = psnr_str.parse()?;

        // Cleanup
        std::fs::remove_file(&output_filename).ok();
        std::fs::remove_file(&decoded_filename).ok();

        Ok(psnr)
    } else {
        Err(format!("Could not parse PSNR from output: {}", stderr).into())
    }
}
