//! `supports_encode_format` must agree with what an encoder can actually be
//! created for. Needs a video-encode capable GPU, so the tests are ignored by default.

#[allow(dead_code)]
mod common;

use pixelforge::{BitDepth, ChromaFormat, Codec, EncodeConfig, Encoder, VideoContextBuilder};

fn config(codec: Codec, chroma_format: ChromaFormat, bit_depth: BitDepth) -> EncodeConfig {
    match codec {
        Codec::H264 => EncodeConfig::h264(1280, 720),
        Codec::H265 => EncodeConfig::h265(1280, 720),
        Codec::AV1 => EncodeConfig::av1(1280, 720),
    }
    .with_chroma_format(chroma_format)
    .with_bit_depth(bit_depth)
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn query_matches_encoder_creation() {
    common::init_logging();
    let context = VideoContextBuilder::new()
        .app_name("pixelforge-test")
        .build()
        .expect("no Vulkan video device");

    for codec in [Codec::H264, Codec::H265, Codec::AV1] {
        for chroma_format in [ChromaFormat::Yuv420, ChromaFormat::Yuv444] {
            for bit_depth in [BitDepth::Eight, BitDepth::Ten] {
                let supported = context.supports_encode_format(codec, chroma_format, bit_depth);
                let created =
                    Encoder::new(context.clone(), config(codec, chroma_format, bit_depth)).is_ok();
                eprintln!(
                    "{codec:?} {chroma_format:?} {bit_depth:?}: query={supported} encoder={created}"
                );
                assert_eq!(
                    supported, created,
                    "{codec:?} {chroma_format:?} {bit_depth:?}: query and encoder creation disagree"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn eight_bit_420_matches_supports_encode() {
    common::init_logging();
    let context = VideoContextBuilder::new()
        .app_name("pixelforge-test")
        .build()
        .expect("no Vulkan video device");
    for codec in [Codec::H264, Codec::H265, Codec::AV1] {
        assert_eq!(
            context.supports_encode_format(codec, ChromaFormat::Yuv420, BitDepth::Eight),
            context.supports_encode(codec),
            "{codec:?}"
        );
    }
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn unsupported_codec_or_format_is_false() {
    common::init_logging();
    let context = VideoContextBuilder::new()
        .app_name("pixelforge-test")
        .build()
        .expect("no Vulkan video device");
    // The encoder has no 4:2:2 H.265 profile.
    assert!(!context.supports_encode_format(Codec::H265, ChromaFormat::Yuv422, BitDepth::Eight));
    for codec in [Codec::H264, Codec::H265, Codec::AV1] {
        if !context.supports_encode(codec) {
            assert!(!context.supports_encode_format(codec, ChromaFormat::Yuv420, BitDepth::Eight));
        }
    }
}
