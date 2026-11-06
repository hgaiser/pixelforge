//! Encode with pixelforge, then decode the result with pixelforge.
//!
//! The decoder must handle anything this project's encoder can produce. That is
//! not covered by decoding third-party streams: the encoder makes its own
//! choices (notably explicit reference marking), so it needs its own round trip.
//! It is also where B-frames will be exercised once the encoder supports them.
//!
//! The input is planar YUV420 (I420). The decoded output is NV12 in display
//! order. The assertions are that no frame is lost across the round trip, and
//! that the output is byte-identical to ffmpeg's software decode of the same
//! stream. The files are left in the scratch dir for inspection.
//!
//! Ignored by default: requires a Vulkan Video device and ffmpeg. Run with
//! `cargo test -- --ignored`.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Write};

#[allow(dead_code)]
mod common;
use common::{Readback, decode_stream, write_nv12};

use pixelforge::decoder::{DecodeConfig, Decoder};
use pixelforge::{
    Codec, EncodeConfig, Encoder, Frame, RateControl, VideoContextBuilder, YuvLayout, YuvUploader,
};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const GOP: u32 = 30;

#[test]
#[ignore = "requires a Vulkan Video device and ffmpeg"]
fn roundtrip_h264() -> Result<(), Box<dyn std::error::Error>> {
    roundtrip(Codec::H264, "h264")
}

#[test]
#[ignore = "requires a Vulkan Video device and ffmpeg"]
fn roundtrip_h265() -> Result<(), Box<dyn std::error::Error>> {
    roundtrip(Codec::H265, "h265")
}

#[test]
#[ignore = "requires a Vulkan Video device and ffmpeg"]
fn roundtrip_av1() -> Result<(), Box<dyn std::error::Error>> {
    roundtrip(Codec::AV1, "obu")
}

fn roundtrip(codec: Codec, extension: &str) -> Result<(), Box<dyn std::error::Error>> {
    let _ = tracing_subscriber::fmt::try_init();

    let input_path = format!("testdata/test_frames_{WIDTH}x{HEIGHT}_yuv420p.yuv");
    common::ensure_test_data(WIDTH, HEIGHT, "yuv420p", &input_path)?;

    let dir = common::scratch_dir(&format!("roundtrip-{extension}"));
    let bitstream_path = dir.join(format!("out.{extension}"));

    let context = VideoContextBuilder::new()
        .app_name("pixelforge-roundtrip")
        .require_encode(codec)
        .require_decode(codec)
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .build()?;

    // --- Encode ---
    let mut yuv = Vec::new();
    File::open(&input_path)?.read_to_end(&mut yuv)?;
    let frame_size = (WIDTH * HEIGHT * 3 / 2) as usize;
    let frame_count = yuv.len() / frame_size;

    let config = EncodeConfig::new(codec, WIDTH, HEIGHT)
        .with_rate_control(RateControl::Cqp { qp: 26 })
        .with_frame_rate(30, 1)
        .with_gop_size(GOP);

    let mut encoder = Encoder::new(context.clone(), config)?;
    let mut input_image = YuvUploader::for_encoder(&encoder)?;

    let mut bitstream = Vec::new();
    let mut pending: VecDeque<pixelforge::EncodeFuture> = VecDeque::new();
    for i in 0..frame_count {
        input_image.upload(YuvLayout::I420, &yuv[i * frame_size..(i + 1) * frame_size])?;
        pending.push_back(encoder.submit(Frame::image(input_image.source()))?);
        while pending.len() > 2 {
            let packet = pollster::block_on(pending.pop_front().expect("non-empty"))?;
            bitstream.extend_from_slice(&packet.data);
        }
    }
    encoder.flush()?;
    while let Some(future) = pending.pop_front() {
        let packet = pollster::block_on(future)?;
        bitstream.extend_from_slice(&packet.data);
    }
    File::create(&bitstream_path)?.write_all(&bitstream)?;
    println!(
        "Encoded {} frames -> {} bytes (gop={GOP})",
        frame_count,
        bitstream.len()
    );

    // --- Decode it back ---
    // Display order by default, so frames come out ready to write; `flush`
    // drains whatever the reorder buffer still holds at end of stream.
    let yuv_path = dir.join("out.yuv");
    let mut out = Some(File::create(&yuv_path)?);
    let mut readback = Some(Readback::new(&context)?);
    let mut decoder = Decoder::new(context, DecodeConfig::new(codec).with_byte_stream())?;
    let mut decoded_count = 0usize;

    decode_stream(&mut decoder, &bitstream, |frame| {
        write_nv12(&frame, &mut readback, &mut out)?;
        decoded_count += 1;
        Ok(())
    })?;
    println!("Decoded {} frames", decoded_count);

    assert_eq!(
        decoded_count, frame_count,
        "round trip lost frames: encoded {frame_count} but decoded {decoded_count}"
    );
    println!("Wrote decoded frames to {}", yuv_path.display());
    drop(out);

    // --- Compare against ffmpeg's software decoder ---
    let reference_path = dir.join("reference.yuv");
    let mut ffmpeg = std::process::Command::new("ffmpeg");
    ffmpeg.args(["-hide_banner", "-loglevel", "error", "-y"]);
    if codec == Codec::AV1 {
        ffmpeg.args(["-c:v", "libdav1d"]);
    }
    let status = ffmpeg
        .arg("-i")
        .arg(&bitstream_path)
        .args(["-pix_fmt", "nv12", "-f", "rawvideo"])
        .arg(&reference_path)
        .status()?;
    assert!(status.success(), "ffmpeg failed to decode the stream");
    assert!(
        std::fs::read(&reference_path)? == std::fs::read(&yuv_path)?,
        "decoded frames differ from ffmpeg's decode of {}",
        bitstream_path.display()
    );

    Ok(())
}
