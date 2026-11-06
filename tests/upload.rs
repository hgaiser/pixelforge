//! `YuvUploader` uploads: every layout of the same picture is the same frame.
//!
//! Ignored by default: requires a Vulkan Video device. Run with
//! `cargo test -- --ignored`.

#[allow(dead_code)]
mod common;

use pixelforge::{
    BitDepth, Codec, EncodeConfig, Encoder, Frame, VideoContext, VideoContextBuilder, YuvLayout,
    YuvUploader,
};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const FRAMES: usize = 3;

fn context() -> Result<VideoContext, Box<dyn std::error::Error>> {
    common::init_logging();
    Ok(VideoContextBuilder::new()
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .require_encode(Codec::H265)
        .build()?)
}

/// A 4:2:0 picture as 8-bit planes: Y, U and V.
fn planes(frame: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (WIDTH as usize, HEIGHT as usize);
    let y = (0..w * h)
        .map(|i| ((i % w) + (i / w) * 3 + frame * 16) as u8)
        .collect();
    let u = (0..w * h / 4).map(|i| ((i % (w / 2)) * 2) as u8).collect();
    let v = (0..w * h / 4)
        .map(|i| ((i / (w / 2)) * 5 + frame * 8) as u8)
        .collect();
    (y, u, v)
}

/// The picture in `layout`.
fn frame(layout: YuvLayout, index: usize) -> Vec<u8> {
    let (y, u, v) = planes(index);
    match layout {
        YuvLayout::I420 => [y, u, v].concat(),
        YuvLayout::Nv12 => {
            let uv = u.iter().zip(&v).flat_map(|(&u, &v)| [u, v]);
            y.into_iter().chain(uv).collect()
        }
        // The same values, in the high 8 of 10 bits.
        YuvLayout::I420P10 => [y, u, v]
            .concat()
            .into_iter()
            .flat_map(|s| ((s as u16) << 2).to_le_bytes())
            .collect(),
        YuvLayout::I444 => unreachable!("a 4:2:0 test"),
    }
}

/// Encode the frames in `layout`, half uploaded to the input image's own
/// image and half straight into a slot.
fn encode(
    context: &VideoContext,
    depth: BitDepth,
    layout: YuvLayout,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let config = EncodeConfig::h265(WIDTH, HEIGHT).with_bit_depth(depth);
    let mut encoder = Encoder::new(context.clone(), config)?;
    let mut input = YuvUploader::for_encoder(&encoder)?;
    let mut stream = Vec::new();
    for i in 0..FRAMES {
        let data = frame(layout, i);
        let packet = if i % 2 == 0 {
            input.upload(layout, &data)?;
            encoder.submit(Frame::image(input.source()))?
        } else {
            let slot = encoder.input_slot()?;
            input.upload_to(&slot, layout, &data)?;
            encoder.submit(Frame::input(slot))?
        };
        stream.extend(pollster::block_on(packet)?.data);
    }
    Ok(stream)
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn nv12_and_i420_upload_the_same_frame() -> Result<(), Box<dyn std::error::Error>> {
    let context = context()?;
    let i420 = encode(&context, BitDepth::Eight, YuvLayout::I420)?;
    let nv12 = encode(&context, BitDepth::Eight, YuvLayout::Nv12)?;
    assert!(!i420.is_empty());
    assert_eq!(
        i420, nv12,
        "NV12 and I420 of one picture encode differently"
    );
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn ten_bit_and_eight_bit_upload_the_same_frame() -> Result<(), Box<dyn std::error::Error>> {
    let context = context()?;
    let i420 = match encode(&context, BitDepth::Ten, YuvLayout::I420) {
        Ok(stream) => stream,
        Err(e) => {
            eprintln!("skipping: no 10-bit H.265 encoder ({e})");
            return Ok(());
        }
    };
    let i420p10 = encode(&context, BitDepth::Ten, YuvLayout::I420P10)?;
    assert!(!i420.is_empty());
    assert_eq!(
        i420, i420p10,
        "I420P10 and I420 of one picture encode differently"
    );
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn a_layout_must_fit_the_image() -> Result<(), Box<dyn std::error::Error>> {
    let context = context()?;
    let encoder = Encoder::new(context.clone(), EncodeConfig::h265(WIDTH, HEIGHT))?;
    let mut input = YuvUploader::for_encoder(&encoder)?;
    let size = |layout: YuvLayout| layout.frame_size(WIDTH, HEIGHT);
    for layout in [YuvLayout::I420P10, YuvLayout::I444] {
        assert!(
            input.upload(layout, &vec![0; size(layout)]).is_err(),
            "{layout:?} was uploaded to an 8-bit 4:2:0 image"
        );
    }
    assert!(
        input
            .upload(YuvLayout::I420, &vec![0; size(YuvLayout::I420) - 1])
            .is_err(),
        "a short frame was uploaded"
    );
    input.upload(YuvLayout::I420, &vec![0; size(YuvLayout::I420)])?;
    Ok(())
}
