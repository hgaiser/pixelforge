//! Encodes a generated sequence, for `scripts/compare_b0.sh`.
//!
//! The script copies this file into `examples/` of two builds and compares
//! what they write. It only uses the API that both builds have.
//!
//! Usage: b0_compare <h264|h265|av1> <cqp|cbr> <gop size> <frames> <output>

use pixelforge::{
    EncodeConfig, Encoder, Frame, RateControl, VideoContextBuilder, YuvLayout, YuvUploader,
};
use std::io::Write;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

/// A frame that moves, so that P-frames are not empty.
fn frame(index: usize) -> Vec<u8> {
    let (w, h) = (WIDTH as usize, HEIGHT as usize);
    let mut data = vec![0u8; w * h * 3 / 2];
    for y in 0..h {
        for x in 0..w {
            data[y * w + x] = ((x + 3 * index) ^ (y + index)) as u8;
        }
    }
    for y in 0..h / 2 {
        for x in 0..w / 2 {
            data[w * h + y * (w / 2) + x] = (x + index) as u8;
            data[w * h + (w / 2) * (h / 2) + y * (w / 2) + x] = (y + 2 * index) as u8;
        }
    }
    data
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        return Err("usage: b0_compare <codec> <cqp|cbr> <gop size> <frames> <output>".into());
    }
    let config = match args[1].as_str() {
        "h264" => EncodeConfig::h264(WIDTH, HEIGHT),
        "h265" => EncodeConfig::h265(WIDTH, HEIGHT),
        "av1" => EncodeConfig::av1(WIDTH, HEIGHT),
        other => return Err(format!("unknown codec {other}").into()),
    };
    let rate_control = match args[2].as_str() {
        "cqp" => RateControl::Cqp { qp: 26 },
        "cbr" => RateControl::Cbr { bitrate: 500_000 },
        other => return Err(format!("unknown rate control {other}").into()),
    };
    let config = config
        .with_rate_control(rate_control)
        .with_frame_rate(30, 1)
        .with_gop_size(args[3].parse()?);
    let frames: usize = args[4].parse()?;

    let context = VideoContextBuilder::new()
        .app_name("b0 compare")
        .require_encode(config.codec)
        .build()?;
    let mut encoder = Encoder::new(context, config)?;
    let mut upload = YuvUploader::for_encoder(&encoder)?;
    let mut output = std::fs::File::create(&args[5])?;
    let mut pending = std::collections::VecDeque::new();
    for index in 0..frames {
        upload.upload(YuvLayout::I420, &frame(index))?;
        if index == frames / 2 {
            encoder.request_idr();
        }
        pending.push_back(encoder.submit(Frame::image(upload.source()))?);
        while pending.len() > 2 {
            output.write_all(&pollster::block_on(pending.pop_front().unwrap())?.data)?;
        }
    }
    encoder.flush()?;
    for future in pending {
        output.write_all(&pollster::block_on(future)?.data)?;
    }
    Ok(())
}
