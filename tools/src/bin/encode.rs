//! Encode raw YUV420 frames, with every setting that matters for checking the
//! encoder.
//!
//! Usage:
//!   cargo run -p pixelforge-tools --bin encode -- [h264|h265|av1] [options]
//!
//! Options:
//!   --input PATH          raw I420 frames (default testdata/test_frames.yuv)
//!   --size WxH            frame size (default 320x240)
//!   --frames N            encode at most N frames
//!   --qp N                constant QP or q-index (default 26)
//!   --gop N               frames per IDR, 0 for none (default 30)
//!   --b-frames N          B-frames between two reference frames (default 0)
//!   --pyramid             make the middle B-frames references
//!   --quality-level N     the driver's quality level (default 0)
//!   --output PATH         the stream (default output.h264, output.h265 or output.av1)
//!
//! Each packet is printed with its timestamps and type.

use pixelforge::{
    BFrameReferences, Codec, EncodeConfig, Encoder, Frame, RateControl, VideoContextBuilder,
    YuvLayout, YuvUploader,
};
use std::collections::VecDeque;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let mut codec = Codec::H264;
    let mut input = "testdata/test_frames.yuv".to_string();
    let (mut width, mut height) = (320u32, 240u32);
    let mut frames = usize::MAX;
    let mut qp = 26;
    let mut gop = 30;
    let mut b_frames = 0;
    let mut references = BFrameReferences::Flat;
    let mut quality_level = 0;
    let mut output = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or(format!("{arg} takes a value"));
        match arg.as_str() {
            "h264" => codec = Codec::H264,
            "h265" => codec = Codec::H265,
            "av1" => codec = Codec::AV1,
            "--input" => input = value()?,
            "--size" => {
                let size = value()?;
                let (w, h) = size.split_once('x').ok_or("--size takes WxH")?;
                (width, height) = (w.parse()?, h.parse()?);
            }
            "--frames" => frames = value()?.parse()?,
            "--qp" => qp = value()?.parse()?,
            "--gop" => gop = value()?.parse()?,
            "--b-frames" => b_frames = value()?.parse()?,
            "--pyramid" => references = BFrameReferences::Pyramid,
            "--quality-level" => quality_level = value()?.parse()?,
            "--output" => output = Some(value()?),
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    let output = output.unwrap_or_else(|| {
        match codec {
            Codec::H264 => "output.h264",
            Codec::H265 => "output.h265",
            Codec::AV1 => "output.av1",
        }
        .to_string()
    });

    let data = std::fs::read(&input)
        .map_err(|e| format!("{input}: {e} (make test frames with `ffmpeg -f lavfi -i testsrc=duration=1:size=320x240:rate=30 -pix_fmt yuv420p -f rawvideo {input}`)"))?;
    let frame_size = YuvLayout::I420.frame_size(width, height);
    let count = (data.len() / frame_size).min(frames);

    let context = VideoContextBuilder::new()
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .require_encode(codec)
        .build()?;
    let config = EncodeConfig::new(codec, width, height)
        .with_rate_control(RateControl::Cqp { qp })
        .with_gop_size(gop)
        .with_b_frames(b_frames)
        .with_b_frame_references(references)
        .with_quality_level(quality_level);
    println!("{config:?}\n");
    let mut encoder = Encoder::new(context, config)?;
    let mut uploader = YuvUploader::for_encoder(&encoder)?;
    let mut file = std::fs::File::create(&output)?;

    let mut total = 0;
    let mut write = |packet: pixelforge::EncodedPacket| -> std::io::Result<()> {
        total += packet.data.len();
        println!(
            "  pts={:<3} dts={:<3} {:>7} bytes {:?}",
            packet.pts,
            packet.dts,
            packet.data.len(),
            packet.frame_type
        );
        file.write_all(&packet.data)
    };
    let mut pending = VecDeque::new();
    for frame in data.chunks_exact(frame_size).take(count) {
        let slot = encoder.input_slot()?;
        uploader.upload_to(&slot, YuvLayout::I420, frame)?;
        pending.push_back(encoder.submit(Frame::input(slot))?);
        // At most `b_frames` frames wait for their group.
        while pending.len() > b_frames as usize + 1 {
            write(pollster::block_on(pending.pop_front().unwrap())?)?;
        }
    }
    encoder.flush()?;
    for future in pending {
        write(pollster::block_on(future)?)?;
    }
    println!("\n{count} frames, {total} bytes, to {output}");
    Ok(())
}
