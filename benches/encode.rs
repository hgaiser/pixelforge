//! Benchmark: Encoding timings and throughput.
//!
//! Reports the GPU, frame-latency and wall-latency statistics the encoder
//! returns for every packet, plus compression ratio, across all codecs and
//! tuning modes. Loads raw YUV420 frames from `testdata/test_frames_1080p.yuv`
//! (1920x1080, 60 FPS).
//!
//! Run with:
//!   cargo bench --bench encode

use pixelforge::{
    Codec, EncodeConfig, Encoder, EncoderTuningMode, Frame, RateControl, VideoContextBuilder,
    YuvLayout, YuvUploader,
};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};

use std::time::Duration;

const TEST_FRAMES_PATH: &str = "testdata/test_frames_1080p.yuv";
const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

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

    println!("PixelForge Encode Bench Example\n");

    // Load test frames.
    let test_path = Path::new(TEST_FRAMES_PATH);
    if !test_path.exists() {
        eprintln!("Test frames not found at '{TEST_FRAMES_PATH}'");
        eprintln!(
            "Generate with: ffmpeg -f lavfi -i testsrc=duration=5:size=1920x1080:rate=60 -pix_fmt yuv420p -f rawvideo testdata/test_frames_1080p.yuv"
        );
        return Ok(());
    }

    let mut yuv_data = Vec::new();
    File::open(test_path)?.read_to_end(&mut yuv_data)?;

    let frame_size = (WIDTH * HEIGHT * 3 / 2) as usize;
    let num_frames = yuv_data.len() / frame_size;
    println!(
        "Input: {num_frames} frames, {WIDTH}x{HEIGHT} YUV420, {} bytes\n",
        yuv_data.len()
    );

    // Create video context. A machine without a usable device skips rather than
    // failing, so `cargo test --all-targets`/`cargo bench` stay green on CI.
    let context = match VideoContextBuilder::new()
        .app_name("Encode Bench")
        .enable_validation(cfg!(debug_assertions))
        .build()
    {
        Ok(context) => context,
        Err(e) => {
            eprintln!("skipping encode bench: no usable Vulkan device ({e})");
            return Ok(());
        }
    };

    // Define codecs and tuning modes to test.
    let codecs = [Codec::H264, Codec::H265, Codec::AV1];
    let tuning_modes = [
        EncoderTuningMode::Default,
        EncoderTuningMode::HighQuality,
        EncoderTuningMode::LowLatency,
        EncoderTuningMode::UltraLowLatency,
        EncoderTuningMode::Lossless,
    ];

    // Test each codec and tuning mode combination.
    for &codec in &codecs {
        if !context.supports_encode(codec) {
            println!("=== {:?} not supported, skipping ===\n", codec);
            continue;
        }

        for &tuning_mode in &tuning_modes {
            println!("=== Testing {:?} with {:?} tuning ===", codec, tuning_mode);

            match run_encode_test(
                &context,
                &yuv_data,
                frame_size,
                num_frames,
                codec,
                tuning_mode,
            ) {
                Ok(()) => println!(),
                Err(e) => eprintln!("Error: {}\n", e),
            }
        }
    }

    Ok(())
}

fn run_encode_test(
    context: &pixelforge::VideoContext,
    yuv_data: &[u8],
    frame_size: usize,
    num_frames: usize,
    codec: Codec,
    tuning_mode: EncoderTuningMode,
) -> Result<(), Box<dyn std::error::Error>> {
    // Configure encoder.
    let config = match codec {
        Codec::H264 => EncodeConfig::h264(WIDTH, HEIGHT),
        Codec::H265 => EncodeConfig::h265(WIDTH, HEIGHT),
        Codec::AV1 => EncodeConfig::av1(WIDTH, HEIGHT),
    }
    .with_rate_control(if matches!(tuning_mode, EncoderTuningMode::Lossless) {
        RateControl::default_for(codec)
    } else {
        RateControl::Cbr { bitrate: 8_000_000 }
    })
    .with_frame_rate(60, 1)
    .with_gop_size(120)
    .with_encode_content_hint(pixelforge::EncodeContentHint::Rendered)
    .with_encode_usage_hint(pixelforge::EncodeUsageHint::Streaming)
    .with_encoder_tuning_mode(tuning_mode);

    println!("Config: {:?}, GOP={}", config.rate_control, config.gop_size);

    let mut encoder = Encoder::new(context.clone(), config)?;
    // Create input image for uploading frames.
    let mut input_image = YuvUploader::for_encoder(&encoder)?;
    let mut total_bytes = 0;

    let mut gpu_durations: Vec<Duration> = Vec::new();
    let mut frame_latency_durations: Vec<Duration> = Vec::new();
    let mut wall_latency_durations: Vec<Duration> = Vec::new();

    let mut pending: std::collections::VecDeque<pixelforge::EncodeFuture> =
        std::collections::VecDeque::new();

    let mut write_packet = |packet: pixelforge::EncodedPacket| {
        total_bytes += packet.data.len();
        let stats = packet.stats;
        if let Some(gpu_time_ns) = stats.gpu_time_ns {
            gpu_durations.push(Duration::from_nanos(gpu_time_ns));
        }
        frame_latency_durations.push(Duration::from_nanos(stats.frame_latency_ns));
        wall_latency_durations.push(Duration::from_nanos(stats.wall_latency_ns));
        Ok::<(), Box<dyn std::error::Error>>(())
    };

    // Encode frames
    for i in 0..num_frames {
        let frame = &yuv_data[i * frame_size..(i + 1) * frame_size];

        // Upload YUV420 data to the input image
        input_image.upload(YuvLayout::I420, frame)?;

        pending.push_back(encoder.submit(Frame::image(input_image.source()))?);
        // drain before submitting when at capacity
        while pending.len() >= 2 {
            let packet = pollster::block_on(pending.pop_front().unwrap())?;
            write_packet(packet)?;
        }
    }

    // Flush remaining frames without using them for statistics
    encoder.flush()?;
    while let Some(_future) = pending.pop_front() {}

    let ratio = (num_frames * frame_size) as f64 / total_bytes as f64;
    println!("Encoded {num_frames} frames, {total_bytes} bytes, {ratio:.1}:1 compression");

    // Print GPU timings.
    if !gpu_durations.is_empty() {
        gpu_durations.sort_unstable();
        print_timing_stats("GPU", &gpu_durations);
    }

    // Print frame latency timings.
    if !frame_latency_durations.is_empty() {
        frame_latency_durations.sort_unstable();
        print_timing_stats("Frame Latency", &frame_latency_durations);
    }

    // Print wall latency timings.
    if !wall_latency_durations.is_empty() {
        wall_latency_durations.sort_unstable();
        print_timing_stats("Wall Latency", &wall_latency_durations);
    }

    Ok(())
}

fn print_timing_stats(label: &str, durations: &[Duration]) {
    let min = durations[0];
    let max = *durations.last().unwrap();
    let sum: Duration = durations.iter().sum();
    let avg = sum / durations.len() as u32;
    let p99_index = (durations.len() as f64 * 0.99) as usize;
    let p99 = durations[p99_index.min(durations.len() - 1)];
    let total = sum.as_millis();

    println!(
        "{} Encode timings: Min: {:?}, Max: {:?}, Avg: {:?}, P99: {:?}, Total: {}ms",
        label, min, max, avg, p99, total
    );
}
