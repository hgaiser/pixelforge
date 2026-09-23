//! Encoding RGB input with the encoder's own colour conversion, through
//! `VK_VALVE_video_encode_rgb_conversion`, against the colour converter.
//!
//! When the conversion is nothing but the YUV matrix, the encoder can take the
//! RGB frame as it is and the shader does not run at all. These tests encode
//! the same frames both ways and check the hardware path lands where the
//! shader does.
//!
//! Skipped on devices without the extension, which today is everything but
//! AMD under RADV.

#[allow(dead_code)]
mod common;
use common::source::{SrcImage, create_src_image};
use common::{Readback, decode_stream};

use ash::vk;
use pixelforge::decoder::{DecodeConfig, Decoder};
use pixelforge::{
    Codec, ColorConverter, ColorConverterConfig, ColorRange, ColorSpec, EncodeConfig, Encoder,
    InputFormat, OutputFormat, RateControlMode, VideoContext, VideoContextBuilder,
};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const FRAMES: u32 = 8;

fn context(codec: Codec) -> Result<VideoContext, Box<dyn std::error::Error>> {
    common::init_logging();
    let mut builder = VideoContextBuilder::new()
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .require_encode(codec);
    if codec == Codec::H264 {
        builder = builder.require_decode(Codec::H264);
    }
    Ok(builder.build()?)
}

fn converter_config() -> ColorConverterConfig {
    ColorConverterConfig::new(
        WIDTH,
        HEIGHT,
        InputFormat::BGRA,
        OutputFormat::NV12,
        ColorSpec::Srgb,
        ColorSpec::Srgb,
        ColorRange::Limited,
    )
}

/// A BGRA frame that moves with `n`.
fn moving_frame(n: u32) -> Vec<u8> {
    let mut data = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let (sx, sy) = (x + n * 3, y + n);
            data.extend_from_slice(&[(sy * 3) as u8, (sx * 2 + sy) as u8, (sx ^ sy) as u8, 255]);
        }
    }
    data
}

/// Upload frame `n`. It is left in `GENERAL`, which is where [`Encoder::encode`]
/// expects a source it copies from.
fn source(context: &VideoContext, n: u32) -> Result<SrcImage, Box<dyn std::error::Error>> {
    Ok(unsafe { create_src_image(context, WIDTH, HEIGHT, &moving_frame(n))? })
}

fn encode_config(codec: Codec) -> EncodeConfig {
    let config = match codec {
        Codec::H264 => EncodeConfig::h264(WIDTH, HEIGHT),
        Codec::H265 => EncodeConfig::h265(WIDTH, HEIGHT),
        Codec::AV1 => EncodeConfig::av1(WIDTH, HEIGHT),
    };
    config
        .with_rate_control(RateControlMode::Cqp)
        .with_quality_level(22)
        .with_frame_rate(30, 1)
        .with_b_frames(0)
        .with_color_description(converter_config().color_description().unwrap())
}

/// Encode the frames with the colour converter doing the conversion.
fn encode_with_shader(
    context: &VideoContext,
    codec: Codec,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut encoder = Encoder::new(context.clone(), encode_config(codec))?;
    let mut converter = ColorConverter::new(context.clone(), converter_config())?;
    let mut stream = Vec::new();
    for n in 0..FRAMES {
        let src = source(context, n)?;
        converter.convert(src.image, vk::ImageLayout::GENERAL, encoder.input_image())?;
        let packet = pollster::block_on(encoder.encode(encoder.input_image())?)?;
        stream.extend_from_slice(&packet.data);
    }
    Ok(stream)
}

/// Encode the frames with the encoder doing the conversion.
fn encode_rgb(
    context: &VideoContext,
    codec: Codec,
    format: InputFormat,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut encoder = Encoder::new(context.clone(), encode_config(codec).with_rgb_input(format))?;
    let mut stream = Vec::new();
    for n in 0..FRAMES {
        let src = source(context, n)?;
        let packet = pollster::block_on(encoder.encode(src.image)?)?;
        stream.extend_from_slice(&packet.data);
    }
    Ok(stream)
}

fn decode_all(
    context: &VideoContext,
    stream: &[u8],
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let mut readback = Readback::new(context)?;
    let mut decoder = Decoder::new(context.clone(), DecodeConfig::h264().with_byte_stream())?;
    let mut frames = Vec::new();
    decode_stream(&mut decoder, stream, |frame| {
        frames.push(readback.read(&frame)?.y);
        Ok(())
    })?;
    Ok(frames)
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

/// The format to hand the encoder, or `None` to skip on this device.
fn rgb_format(context: &VideoContext, test: &str) -> Option<InputFormat> {
    let format = converter_config().rgb_encode_input(context);
    if format.is_none() {
        eprintln!("skipping {test}: no VK_VALVE_video_encode_rgb_conversion on this device");
    }
    format
}

#[test]
#[ignore = "requires a Vulkan Video device with VK_VALVE_video_encode_rgb_conversion"]
fn rgb_input_matches_the_converter() -> Result<(), Box<dyn std::error::Error>> {
    let context = context(Codec::H264)?;
    let Some(format) = rgb_format(&context, "rgb_input_matches_the_converter") else {
        return Ok(());
    };
    let shader = decode_all(&context, &encode_with_shader(&context, Codec::H264)?)?;
    let hardware = decode_all(&context, &encode_rgb(&context, Codec::H264, format)?)?;
    assert_eq!(hardware.len(), FRAMES as usize, "frame count");
    let psnrs: Vec<f64> = hardware
        .iter()
        .zip(&shader)
        .map(|(h, s)| psnr(h, s))
        .collect();
    println!("hardware vs shader conversion, luma PSNR per frame: {psnrs:?}");
    // Both apply the same BT.709 matrix to the same frames, then the same
    // encoder settings; what is left is rounding in the two conversions and
    // what the encoder makes of it.
    for (i, p) in psnrs.iter().enumerate() {
        assert!(*p >= 40.0, "frame {i}: {p:.2} dB against the shader path");
    }
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan Video device with VK_VALVE_video_encode_rgb_conversion"]
fn rgb_input_encodes_every_codec() -> Result<(), Box<dyn std::error::Error>> {
    for codec in [Codec::H264, Codec::H265, Codec::AV1] {
        let context = match context(codec) {
            Ok(context) => context,
            Err(e) => {
                eprintln!("{codec:?}: no encoder on this device ({e})");
                continue;
            }
        };
        let Some(format) = rgb_format(&context, "rgb_input_encodes_every_codec") else {
            return Ok(());
        };
        let stream = encode_rgb(&context, codec, format)?;
        assert!(!stream.is_empty(), "{codec:?}: empty stream");
        if let Ok(dir) = std::env::var("PIXELFORGE_RGB_STREAM_DIR") {
            std::fs::write(format!("{dir}/rgb.{codec:?}").to_lowercase(), &stream)?;
        }
    }
    Ok(())
}
