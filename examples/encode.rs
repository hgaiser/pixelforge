//! Encode raw YUV420 frames to H.264, H.265 or AV1.
//!
//! Demonstrates encoding with PixelForge using Vulkan Video. Loads raw YUV420
//! frames from `testdata/test_frames.yuv` and writes the bitstream to
//! `output.h264`, `output.h265` or `output.av1` depending on the codec.
//!
//! Usage:
//!   cargo run --example encode -- [h264|h265|av1]
//!
//! The codec defaults to `h264`.

use pixelforge::{
    ChromaFormat, Codec, EncodeConfig, Encoder, Frame, RateControl, VideoContextBuilder, YuvLayout,
    YuvUploader,
};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};

const TEST_FRAMES_PATH: &str = "testdata/test_frames.yuv";
const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

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

    // Pick the codec. H.265 asks for an explicit format; the others take the
    // default, which is also YUV420.
    let codec_arg = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "h264".to_string());
    let (codec, config, output_name) = match codec_arg.as_str() {
        "h264" => (
            Codec::H264,
            EncodeConfig::h264(WIDTH, HEIGHT),
            "output.h264",
        ),
        "h265" => (
            Codec::H265,
            EncodeConfig::h265(WIDTH, HEIGHT).with_chroma_format(ChromaFormat::Yuv420),
            "output.h265",
        ),
        "av1" => (Codec::AV1, EncodeConfig::av1(WIDTH, HEIGHT), "output.av1"),
        other => {
            eprintln!("unknown codec '{other}', expected one of: h264, h265, av1");
            std::process::exit(1);
        }
    };

    println!("PixelForge {codec:?} Encode Example ({output_name})\n");

    // Load test frames.
    let test_path = Path::new(TEST_FRAMES_PATH);
    if !test_path.exists() {
        eprintln!("Test frames not found at '{TEST_FRAMES_PATH}'");
        eprintln!(
            "Generate with: ffmpeg -f lavfi -i testsrc=duration=0.5:size=320x240:rate=30 -pix_fmt yuv420p -f rawvideo testdata/test_frames.yuv"
        );
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
        .app_name("Encode Example")
        .enable_validation(cfg!(debug_assertions))
        .require_encode(codec)
        .build()?;

    if !context.supports_encode(codec) {
        eprintln!("{codec:?} encode not supported");
        return Ok(());
    }

    // Configure encoder.
    let config = config
        .with_rate_control(RateControl::Cqp { qp: 26 })
        .with_frame_rate(30, 1)
        .with_gop_size(30);

    println!(
        "Config: {:?}, GOP={}\n",
        config.rate_control, config.gop_size
    );

    let mut encoder = Encoder::new(context, config)?;
    // Create input image for uploading frames.
    let mut input_image = YuvUploader::for_encoder(&encoder)?;
    let mut output = File::create(output_name)?;
    let mut total_bytes = 0;

    // Each `encode()` returns a future that resolves with that frame's packet.
    // Keep a few in flight (so capture/upload overlaps GPU encode) and drain the
    // oldest once the pipeline is full, preserving submission order.
    let mut pending: std::collections::VecDeque<pixelforge::EncodeFuture> =
        std::collections::VecDeque::new();

    let mut write_packet = |packet: pixelforge::EncodedPacket, total: &mut usize| {
        *total += packet.data.len();
        output.write_all(&packet.data)?;
        println!(
            "  pts={:<2} dts={:<2}: {:>5} bytes, {:?}{}",
            packet.pts,
            packet.dts,
            packet.data.len(),
            packet.frame_type,
            if packet.is_key_frame() { " [KEY]" } else { "" }
        );
        Ok::<(), Box<dyn std::error::Error>>(())
    };

    // Encode frames.
    for i in 0..num_frames {
        let frame = &yuv_data[i * frame_size..(i + 1) * frame_size];

        // Upload YUV420 data to the input image.
        input_image.upload(YuvLayout::I420, frame)?;

        // Submit the frame (async) and keep the pipeline at most ~2 deep.
        pending.push_back(encoder.submit(Frame::image(input_image.source()))?);
        while pending.len() > 2 {
            let packet = pollster::block_on(pending.pop_front().unwrap())?;
            write_packet(packet, &mut total_bytes)?;
        }
    }

    // Flush remaining frames: barrier, then drain the outstanding futures in order.
    encoder.flush()?;
    while let Some(future) = pending.pop_front() {
        let packet = pollster::block_on(future)?;
        write_packet(packet, &mut total_bytes)?;
    }

    let ratio = (num_frames * frame_size) as f64 / total_bytes as f64;
    println!("\nEncoded {num_frames} frames, {total_bytes} bytes, {ratio:.1}:1 compression");
    println!("Output: {output_name}");

    Ok(())
}
