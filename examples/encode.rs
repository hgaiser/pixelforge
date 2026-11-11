//! Encode frames from CPU memory to a file.
//!
//! Each frame is made here as I420, uploaded into the encoder's input image,
//! and encoded. The packets are written as they come, which gives a raw
//! stream: Annex B for H.264 and H.265, low-overhead OBUs for AV1.
//!
//! Usage:
//!   cargo run --example encode -- [h264|h265|av1]

use pixelforge::{
    Codec, EncodeConfig, EncodedPacket, Encoder, Frame, RateControl, VideoContextBuilder,
    YuvLayout, YuvUploader,
};
use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const FRAMES: u32 = 90;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (codec, path) = match std::env::args().nth(1).as_deref() {
        None | Some("h264") => (Codec::H264, "output.h264"),
        Some("h265") => (Codec::H265, "output.h265"),
        Some("av1") => (Codec::AV1, "output.obu"),
        Some(other) => return Err(format!("unknown codec {other:?}").into()),
    };

    let context = VideoContextBuilder::new().require_encode(codec).build()?;
    let config = EncodeConfig::new(codec, WIDTH, HEIGHT)
        .with_frame_rate(30, 1)
        .with_rate_control(RateControl::Cbr { bitrate: 2_000_000 });
    let mut encoder = Encoder::new(context, config)?;
    let mut uploader = YuvUploader::for_encoder(&encoder)?;

    let mut output = File::create(path)?;
    let mut write = |packet: EncodedPacket| output.write_all(&packet.data);
    let mut pending = VecDeque::new();
    for index in 0..FRAMES {
        // Write the frame straight into the encoder's input image.
        let slot = encoder.input_slot()?;
        uploader.upload_to(&slot, YuvLayout::I420, &make_frame(index))?;
        // This returns before the frame is encoded. The future resolves with
        // the next packet of the stream.
        pending.push_back(encoder.submit(Frame::input(slot))?);
        // Keep a frame in flight, so the CPU and the GPU work at once.
        while pending.len() > 1 {
            write(pollster::block_on(pending.pop_front().unwrap())?)?;
        }
    }
    // Encode what is still held, and take the last packets.
    encoder.flush()?;
    for future in pending {
        write(pollster::block_on(future)?)?;
    }
    println!("Wrote {FRAMES} frames to {path}");
    Ok(())
}

/// Frame `index` of a pattern that moves: I420, a full-size Y plane, then
/// quarter-size U and V planes.
fn make_frame(index: u32) -> Vec<u8> {
    let (width, height) = (WIDTH as usize, HEIGHT as usize);
    let mut frame = vec![128u8; YuvLayout::I420.frame_size(WIDTH, HEIGHT)];
    for y in 0..height {
        for x in 0..width {
            frame[y * width + x] = ((x + y + 4 * index as usize) % 256) as u8;
        }
    }
    frame
}
