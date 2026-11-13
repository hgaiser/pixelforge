//! What this device can encode and decode.
//!
//! Usage:
//!   cargo run --example query_capabilities

use pixelforge::{
    BitDepth, ChromaFormat, Codec, ColorRange, RgbFormat, StreamColor, VideoContextBuilder,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = VideoContextBuilder::new().build()?;
    let name =
        unsafe { std::ffi::CStr::from_ptr(context.device_properties().device_name.as_ptr()) };
    println!("Device: {}\n", name.to_string_lossy());

    for codec in [Codec::H264, Codec::H265, Codec::AV1] {
        println!("{codec:?}");
        println!("  decode: {}", context.supports_decode(codec));
        println!("  encode: {}", context.supports_encode(codec));
        if !context.supports_encode(codec) {
            continue;
        }
        for chroma_format in [ChromaFormat::Yuv420, ChromaFormat::Yuv444] {
            for bit_depth in [BitDepth::Eight, BitDepth::Ten] {
                if !context.supports_encode_format(codec, chroma_format, bit_depth) {
                    continue;
                }
                println!(
                    "  encode {chroma_format:?} {bit_depth:?}: B-frames {:?}, {} quality levels",
                    context.supports_b_frames(codec, chroma_format, bit_depth),
                    context.encode_quality_levels(codec, chroma_format, bit_depth),
                );
            }
        }
        // RGB input that the encoder converts itself, with no shader.
        let sdr = StreamColor::Sdr {
            range: ColorRange::Limited,
        };
        println!(
            "  RGB input converted by the encoder: {}",
            context.supports_rgb_encode_input(codec, RgbFormat::BGRx, sdr)
        );
    }
    Ok(())
}
