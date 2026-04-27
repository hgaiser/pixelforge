//! Example: Verify all encoding combinations
//!
//! Verifies H.264/H.265/AV1, 8-bit/10-bit, YUV420/YUV444 combinations.
//! Runs PSNR analysis for each combination.

use pixelforge::{
    Codec, EncodeBitDepth, EncodeConfig, Encoder, InputImage, PixelFormat, RateControlMode,
    VideoContextBuilder,
};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::process::Command;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const FRAMES: u32 = 30;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    let yuv420_path = format!("testdata/test_frames_{}x{}_yuv420p.yuv", WIDTH, HEIGHT);
    let yuv444_path = format!("testdata/test_frames_{}x{}_yuv444p.yuv", WIDTH, HEIGHT);
    ensure_test_data("yuv420p", &yuv420_path)?;
    ensure_test_data("yuv444p", &yuv444_path)?;

    let combinations = [
        (Codec::H264, EncodeBitDepth::Eight, PixelFormat::Yuv420),
        (Codec::H264, EncodeBitDepth::Eight, PixelFormat::Yuv444),
        (Codec::H264, EncodeBitDepth::Ten, PixelFormat::Yuv420),
        (Codec::H264, EncodeBitDepth::Ten, PixelFormat::Yuv444),
        (Codec::H265, EncodeBitDepth::Eight, PixelFormat::Yuv420),
        (Codec::H265, EncodeBitDepth::Eight, PixelFormat::Yuv444),
        (Codec::H265, EncodeBitDepth::Ten, PixelFormat::Yuv420),
        (Codec::H265, EncodeBitDepth::Ten, PixelFormat::Yuv444),
        (Codec::AV1, EncodeBitDepth::Eight, PixelFormat::Yuv420),
        (Codec::AV1, EncodeBitDepth::Eight, PixelFormat::Yuv444),
        (Codec::AV1, EncodeBitDepth::Ten, PixelFormat::Yuv420),
        (Codec::AV1, EncodeBitDepth::Ten, PixelFormat::Yuv444),
    ];

    let context = VideoContextBuilder::new()
        .app_name("Verify All")
        .enable_validation(true) // Enable validation for debugging
        .build()?;

    for (codec, depth, format) in combinations {
        println!("Testing {:?} {:?} {:?}...", codec, depth, format);

        if !context.supports_encode(codec) {
            println!("  Skipping: Codec not supported");
            continue;
        }

        // TODO: Check if specific format/depth is supported?
        // The context.supports_encode only checks codec presence.
        // We'll try to create the encoder and see if it fails.

        let result = run_test(&context, codec, depth, format);
        match result {
            Ok(psnr) => println!("  PASS: PSNR = {:.2} dB", psnr),
            Err(e) => println!("  FAIL: {}", e),
        }
        println!("------------------------------------------------");
    }

    Ok(())
}

fn ensure_test_data(pix_fmt: &str, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if Path::new(path).exists() {
        return Ok(());
    }
    println!("Generating {}...", path);
    let status = Command::new("ffmpeg")
        .args([
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc=duration=1:size={}x{}:rate=30", WIDTH, HEIGHT),
            "-pix_fmt",
            pix_fmt,
            "-f",
            "rawvideo",
            "-y",
            path,
        ])
        .output()?;

    if !status.status.success() {
        return Err(format!("Failed to generate test data: {:?}", status).into());
    }
    Ok(())
}

fn run_test(
    context: &pixelforge::VideoContext,
    codec: Codec,
    depth: EncodeBitDepth,
    format: PixelFormat,
) -> Result<f64, Box<dyn std::error::Error>> {
    // AV1 uses .obu extension for raw OBU streams (with temporal delimiters).
    // H.264/H.265 use .bin for raw Annex B bitstreams.
    let output_ext = if codec == Codec::AV1 { "obu" } else { "bin" };
    let output_filename = format!("output_{:?}_{:?}_{:?}.{}", codec, depth, format, output_ext);
    let decoded_filename = format!("decoded_{:?}_{:?}_{:?}.yuv", codec, depth, format);

    // 1. Encode
    {
        let config = match codec {
            Codec::H264 => EncodeConfig::h264(WIDTH, HEIGHT),
            Codec::H265 => EncodeConfig::h265(WIDTH, HEIGHT),
            Codec::AV1 => EncodeConfig::av1(WIDTH, HEIGHT),
        }
        .with_rate_control(RateControlMode::Cqp)
        .with_quality_level(10)
        .with_pixel_format(format)
        .with_bit_depth(depth);

        let mut encoder = match Encoder::new(context.clone(), config) {
            Ok(e) => e,
            Err(e) => return Err(format!("Failed to create encoder: {}", e).into()),
        };

        let mut input_image =
            InputImage::new(context.clone(), codec, WIDTH, HEIGHT, depth, format)?;

        let input_path = match format {
            PixelFormat::Yuv420 => format!("testdata/test_frames_{}x{}_yuv420p.yuv", WIDTH, HEIGHT),
            PixelFormat::Yuv444 => format!("testdata/test_frames_{}x{}_yuv444p.yuv", WIDTH, HEIGHT),
            _ => return Err("Unsupported format".into()),
        };

        let mut yuv_data = Vec::new();
        File::open(&input_path)?.read_to_end(&mut yuv_data)?;

        let frame_size = match format {
            PixelFormat::Yuv420 => (WIDTH * HEIGHT * 3 / 2) as usize,
            PixelFormat::Yuv444 => (WIDTH * HEIGHT * 3) as usize,
            _ => return Err("Unsupported format".into()),
        };

        let mut output_file = File::create(&output_filename)?;

        for i in 0..FRAMES {
            let start = (i as usize) * frame_size;
            let end = start + frame_size;
            if end > yuv_data.len() {
                break;
            }
            let frame = &yuv_data[start..end];

            // Upload directly to encoder's input image to avoid cross-queue
            // copy issues (InputImage uses the transfer queue, encoder uses the
            // video encode queue which doesn't support transfer ops).
            let encoder_image = encoder.input_image();
            match format {
                PixelFormat::Yuv420 => input_image.upload_yuv420_to(encoder_image, frame)?,
                PixelFormat::Yuv444 => input_image.upload_yuv444_to(encoder_image, frame)?,
                _ => return Err("Unsupported format".into()),
            }

            for packet in encoder.encode(encoder_image, None)? {
                output_file.write_all(&packet.data)?;
            }
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
        PixelFormat::Yuv420 => (
            "yuv420p",
            format!("testdata/test_frames_{}x{}_yuv420p.yuv", WIDTH, HEIGHT),
        ),
        PixelFormat::Yuv444 => (
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
