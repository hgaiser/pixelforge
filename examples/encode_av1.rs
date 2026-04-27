//! Example: AV1 Video Encoding
//!
//! Demonstrates AV1 video encoding using PixelForge with Vulkan Video.
//! Loads raw YUV420 frames from `testdata/test_frames.yuv`.

use pixelforge::{
    Codec, EncodeBitDepth, EncodeConfig, Encoder, InputImage, PixelFormat, RateControlMode,
    VideoContextBuilder,
};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};

const TEST_FRAMES_PATH: &str = "testdata/test_frames.yuv";
const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing with RUST_LOG support.
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer().with_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
            ),
        )
        .init();

    println!("PixelForge AV1 Encode Example\n");

    // Load test frames.
    let test_path = Path::new(TEST_FRAMES_PATH);
    if !test_path.exists() {
        eprintln!("Test frames not found at '{TEST_FRAMES_PATH}'");
        eprintln!("Generate with: ffmpeg -f lavfi -i testsrc=duration=0.5:size=320x240:rate=30 -pix_fmt yuv420p -f rawvideo testdata/test_frames.yuv");
        return Ok(());
    }

    let mut yuv_data = Vec::new();
    File::open(test_path)?.read_to_end(&mut yuv_data)?;

    let frame_size = (WIDTH * HEIGHT * 3 / 2) as usize;
    let num_frames = yuv_data.len() / frame_size;
    println!(
        "Input: {num_frames} frames, {WIDTH}x{HEIGHT} YUV420, {} bytes",
        yuv_data.len()
    );

    // Create video context.
    let context = VideoContextBuilder::new()
        .app_name("AV1 Encode Example")
        .enable_validation(cfg!(debug_assertions))
        .require_encode(Codec::AV1)
        .build()?;

    if !context.supports_encode(Codec::AV1) {
        eprintln!("AV1 encode not supported");
        return Ok(());
    }

    // Configure encoder.
    let config = EncodeConfig::av1(WIDTH, HEIGHT)
        .with_rate_control(RateControlMode::Cqp)
        .with_quality_level(26)
        .with_frame_rate(30, 1)
        .with_gop_size(30)
        .with_b_frames(0);

    println!(
        "Config: {:?}, QP={}, GOP={}, B-frames={}\n",
        config.rate_control_mode, config.quality_level, config.gop_size, config.b_frame_count
    );

    // Create input image for uploading frames.
    let mut input_image = InputImage::new(
        context.clone(),
        Codec::AV1,
        WIDTH,
        HEIGHT,
        EncodeBitDepth::Eight,
        PixelFormat::Yuv420,
    )?;
    let mut encoder = Encoder::new(context, config)?;
    let mut output = File::create("output.av1")?;
    let mut total_bytes = 0;

    // Encode frames.
    for i in 0..num_frames {
        let frame = &yuv_data[i * frame_size..(i + 1) * frame_size];

        // Upload YUV420 data to the input image.
        input_image.upload_yuv420(frame)?;

        // Encode the image (passing InputImage's image, which triggers
        // an internal copy to the encoder's input image with proper
        // layout transitions).
        for packet in encoder.encode(input_image.image(), None)? {
            total_bytes += packet.data.len();
            output.write_all(&packet.data)?;
            println!(
                "  pts={:<2} dts={:<2}: {:>5} bytes, {:?}{}",
                packet.pts,
                packet.dts,
                packet.data.len(),
                packet.frame_type,
                if packet.is_key_frame { " [KEY]" } else { "" }
            );
        }
    }

    // Flush remaining frames.
    for packet in encoder.flush()? {
        total_bytes += packet.data.len();
        output.write_all(&packet.data)?;
        println!(
            "  pts={:<2} dts={:<2}: {:>5} bytes, {:?} (flushed)",
            packet.pts,
            packet.dts,
            packet.data.len(),
            packet.frame_type
        );
    }

    let ratio = (num_frames * frame_size) as f64 / total_bytes as f64;
    println!("\nEncoded {num_frames} frames, {total_bytes} bytes, {ratio:.1}:1 compression");
    println!("Output: output.av1");

    Ok(())
}
