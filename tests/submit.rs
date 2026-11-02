//! The rules `Encoder::submit` holds a caller to.
//!
//! Ignored by default: requires a Vulkan Video device. Run with
//! `cargo test -- --ignored`.

#[allow(dead_code)]
mod common;

use ash::vk;
use pixelforge::{
    Codec, ColorSpec, ConversionPreference, EncodeBitDepth, EncodeConfig, Encoder, Frame, Handoff,
    InputFormat, InputImage, PixelFormat, RgbSource, Source, SourceImage, VideoContext,
    VideoContextBuilder,
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
    let input = InputImage::new(
        context.clone(),
        Codec::H264,
        WIDTH,
        HEIGHT,
        EncodeBitDepth::Eight,
        PixelFormat::Yuv420,
    )?;

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
    let input = InputImage::new(
        context.clone(),
        Codec::H264,
        WIDTH,
        HEIGHT,
        EncodeBitDepth::Eight,
        PixelFormat::Yuv420,
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
                InputFormat::BGRA,
                ColorSpec::Srgb,
            )))
            .with_conversion(ConversionPreference::Shader),
    )?;
    assert!(encoder.input_slot().is_err());
    Ok(())
}
