//! The rules `Encoder::submit` holds a caller to.
//!
//! Ignored by default: requires a Vulkan Video device. Run with
//! `cargo test -- --ignored`.

#[allow(dead_code)]
mod common;

use ash::vk;
use pixelforge::{
    BitDepth, ChromaFormat, Codec, ColorSpec, ConversionPreference, EncodeConfig, Encoder, Frame,
    Handoff, RgbFormat, RgbSource, Source, SourceImage, VideoContext, VideoContextBuilder,
    YuvLayout, YuvUploader,
};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

fn context() -> Result<VideoContext, Box<dyn std::error::Error>> {
    common::init_logging();
    Ok(VideoContextBuilder::new()
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .require_encode(Codec::H264)
        .build()?)
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn an_input_slot_is_valid_until_the_next_submission() -> Result<(), Box<dyn std::error::Error>> {
    let context = context()?;
    let mut encoder = Encoder::new(context.clone(), EncodeConfig::h264(WIDTH, HEIGHT))?;
    let input = YuvUploader::for_encoder(&encoder)?;

    let stale = encoder.input_slot()?;
    let packet = encoder.submit(Frame::image(input.source()))?;
    assert!(
        encoder.submit(Frame::input(stale)).is_err(),
        "a slot from before a submission was accepted"
    );
    pollster::block_on(packet)?;

    let slot = encoder.input_slot()?;
    assert!(
        !pollster::block_on(encoder.submit(Frame::input(slot))?)?
            .data
            .is_empty()
    );
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn a_source_image_needs_pixels() -> Result<(), Box<dyn std::error::Error>> {
    let context = context()?;
    let input = YuvUploader::new(
        context.clone(),
        WIDTH,
        HEIGHT,
        BitDepth::Eight,
        ChromaFormat::Yuv420,
    )?;
    for layout in [vk::ImageLayout::UNDEFINED, vk::ImageLayout::PREINITIALIZED] {
        assert!(
            SourceImage::new(&context, input.image(), Handoff::Device { layout }).is_err(),
            "{layout:?} was accepted"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn a_shader_conversion_has_no_input_slot() -> Result<(), Box<dyn std::error::Error>> {
    let context = context()?;
    let encoder = Encoder::new(
        context,
        EncodeConfig::h264(WIDTH, HEIGHT)
            .with_source(Source::Rgb(RgbSource::new(
                RgbFormat::BGRA,
                ColorSpec::Srgb,
            )))
            .with_conversion(ConversionPreference::Shader),
    )?;
    assert!(encoder.input_slot().is_err());
    Ok(())
}

/// `repeat` encodes the last frame again, from the encoder's own input image,
/// as a key frame when one was asked for. Twice in a row covers both cases:
/// the frame in another slot's image, then in the current slot's own.
#[test]
#[ignore = "requires a Vulkan Video device"]
fn repeat_encodes_the_last_frame_again() -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();
    let context = VideoContextBuilder::new()
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .require_encode(Codec::H264)
        .require_decode(Codec::H264)
        .build()?;
    let mut encoder = Encoder::new(context.clone(), EncodeConfig::h264(WIDTH, HEIGHT))?;
    assert!(
        encoder.repeat().is_err(),
        "repeated a frame that was never there"
    );

    let mut input = YuvUploader::for_encoder(&encoder)?;
    let luma_levels = [40u8, 120, 200];
    let plane = (WIDTH * HEIGHT) as usize;
    let mut stream = Vec::new();
    let mut keys = Vec::new();
    for level in luma_levels {
        let mut yuv = vec![128u8; plane * 3 / 2];
        yuv[..plane].fill(level);
        input.upload(YuvLayout::I420, &yuv)?;
        let packet = pollster::block_on(encoder.submit(Frame::image(input.source()))?)?;
        keys.push(packet.is_key_frame());
        stream.extend(packet.data);
    }
    // The source changes, and the repeats must not see it.
    let mut yuv = vec![128u8; plane * 3 / 2];
    yuv[..plane].fill(10);
    input.upload(YuvLayout::I420, &yuv)?;
    for _ in 0..2 {
        encoder.request_idr();
        let packet = pollster::block_on(encoder.repeat()?)?;
        keys.push(packet.is_key_frame());
        stream.extend(packet.data);
    }
    assert_eq!(keys, [true, false, false, true, true]);

    let mut readback = common::Readback::new(&context)?;
    let mut decoder = pixelforge::Decoder::new(
        context.clone(),
        pixelforge::DecodeConfig::new(pixelforge::Codec::H264).with_byte_stream(),
    )?;
    let mut means = Vec::new();
    common::decode_stream(&mut decoder, &stream, |frame| {
        let y = readback.read(&frame)?.y;
        means.push(y.iter().map(|&v| f64::from(v)).sum::<f64>() / y.len() as f64);
        Ok(())
    })?;
    println!("decoded luma means: {means:?}");
    assert_eq!(means.len(), 5, "frame count");
    for (i, mean) in means.iter().enumerate() {
        let want = f64::from(luma_levels[i.min(2)]);
        assert!(
            (mean - want).abs() < 2.0,
            "frame {i}: luma {mean:.1}, want {want}"
        );
    }
    Ok(())
}
