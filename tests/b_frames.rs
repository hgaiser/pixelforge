//! B-frame configuration: what the device query says against what
//! `Encoder::new` accepts.
//!
//! Ignored by default: requires a Vulkan Video device. Run with
//! `cargo test -- --ignored`.

#[allow(dead_code)]
mod common;

use pixelforge::{
    BFrameReferences, BFrameSupport, BitDepth, ChromaFormat, Codec, EncodeConfig, Encoder,
    MAX_B_FRAMES, PixelForgeError, VideoContextBuilder,
};

/// `supports_b_frames` must agree with what `Encoder::new` accepts, for each
/// codec and kind of B-frames, on every device.
#[test]
#[ignore = "requires a Vulkan Video device"]
fn b_frame_query_matches_encoder_creation() -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();
    let context = VideoContextBuilder::new()
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .build()?;
    for codec in [Codec::H264, Codec::H265, Codec::AV1] {
        if !context.supports_encode(codec) {
            continue;
        }
        let support = context.supports_b_frames(codec, ChromaFormat::Yuv420, BitDepth::Eight);
        eprintln!("{codec:?}: {support:?}");
        for (references, needed) in [
            (BFrameReferences::Flat, BFrameSupport::Flat),
            (BFrameReferences::Pyramid, BFrameSupport::Pyramid),
        ] {
            let config = EncodeConfig::new(codec, 320, 240)
                .with_b_frames(3)
                .with_b_frame_references(references);
            let created = Encoder::new(context.clone(), config);
            assert_eq!(
                created.is_ok(),
                support >= needed,
                "{codec:?} {references:?}: query says {support:?}, creation gives {:?}",
                created.err()
            );
        }
    }
    Ok(())
}

/// More than `MAX_B_FRAMES` is an error on any device, before the device is
/// asked.
#[test]
#[ignore = "requires a Vulkan Video device"]
fn too_many_b_frames_is_an_error() -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();
    let context = VideoContextBuilder::new()
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .require_encode(Codec::H264)
        .build()?;
    let config = EncodeConfig::h264(320, 240).with_b_frames(MAX_B_FRAMES + 1);
    assert!(matches!(
        Encoder::new(context, config),
        Err(PixelForgeError::InvalidInput(_))
    ));
    Ok(())
}
