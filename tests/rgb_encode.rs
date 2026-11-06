//! Encoding an RGB [`Source`], through each [`Conversion`] the encoder can
//! pick.
//!
//! With [`Conversion::Shader`] the encoder runs the colour converter itself;
//! the crate's own tests check that it is byte-identical to converting by
//! hand. With [`Conversion::Encoder`] the encoder converts through
//! `VK_VALVE_video_encode_rgb_conversion`; those tests check that it lands
//! where the shader does, and skip on devices without the extension, which
//! today is everything but AMD under RADV.

#[allow(dead_code)]
mod common;
use common::source::{SrcImage, create_src_image, create_src_image_as};
use common::{Readback, decode_stream};

use ash::vk;
use pixelforge::decoder::{DecodeConfig, Decoder};
use pixelforge::{
    BitDepth, Codec, ColorRange, ColorSpec, Conversion, ConversionPreference, EncodeConfig,
    Encoder, Frame, Handoff, RateControl, RgbFormat, RgbSource, Source, SourceImage, StreamColor,
    VideoContext, VideoContextBuilder,
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

/// Upload frame `n`. It is left in `GENERAL`.
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
        .with_rate_control(RateControl::Cqp { qp: 22 })
        .with_frame_rate(30, 1)
        .with_color(StreamColor::default())
}

fn rgb(format: RgbFormat) -> Source {
    Source::Rgb(RgbSource::new(format, ColorSpec::Srgb))
}

/// `src` as a source to encode from, handed over in `GENERAL`.
fn source_image(
    context: &VideoContext,
    src: &SrcImage,
) -> Result<SourceImage, pixelforge::PixelForgeError> {
    SourceImage::new(
        context,
        src.image,
        Handoff::Device {
            layout: vk::ImageLayout::GENERAL,
        },
    )
}

/// Encode the frames with an RGB source and the given conversion.
fn encode_rgb(
    context: &VideoContext,
    codec: Codec,
    format: RgbFormat,
    preference: ConversionPreference,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut encoder = Encoder::new(
        context.clone(),
        encode_config(codec)
            .with_source(rgb(format))
            .with_conversion(preference),
    )?;
    let mut stream = Vec::new();
    for n in 0..FRAMES {
        let src = source(context, n)?;
        let image = source_image(context, &src)?;
        let packet = pollster::block_on(encoder.submit(Frame::image(&image))?)?;
        stream.extend_from_slice(&packet.data);
    }
    Ok(stream)
}

fn decode_all(
    context: &VideoContext,
    stream: &[u8],
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let mut readback = Readback::new(context)?;
    let mut decoder = Decoder::new(
        context.clone(),
        DecodeConfig::new(Codec::H264).with_byte_stream(),
    )?;
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

/// Whether the encoder can convert `format` for `codec` here; prints why a
/// test is skipped when not.
fn encoder_converts(context: &VideoContext, codec: Codec, test: &str) -> bool {
    let ok = context.supports_rgb_encode_input(codec, RgbFormat::BGRA, StreamColor::default());
    if !ok {
        eprintln!("skipping {test} for {codec:?}: the encoder cannot convert RGB here");
    }
    ok
}

/// `Auto` picks the encoder's own conversion exactly where it is available.
#[test]
#[ignore = "requires a Vulkan Video device"]
fn auto_picks_the_encoder_where_it_can() -> Result<(), Box<dyn std::error::Error>> {
    let context = context(Codec::H264)?;
    let encoder = Encoder::new(
        context.clone(),
        encode_config(Codec::H264).with_source(rgb(RgbFormat::BGRA)),
    )?;
    let expected = if context.supports_rgb_encode_input(
        Codec::H264,
        RgbFormat::BGRA,
        StreamColor::default(),
    ) {
        Conversion::Encoder
    } else {
        Conversion::Shader
    };
    assert_eq!(encoder.conversion(), expected);
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan Video device with VK_VALVE_video_encode_rgb_conversion"]
fn rgb_input_matches_the_converter() -> Result<(), Box<dyn std::error::Error>> {
    let context = context(Codec::H264)?;
    if !encoder_converts(&context, Codec::H264, "rgb_input_matches_the_converter") {
        return Ok(());
    }
    let shader = decode_all(
        &context,
        &encode_rgb(
            &context,
            Codec::H264,
            RgbFormat::BGRA,
            ConversionPreference::Shader,
        )?,
    )?;
    let hardware = decode_all(
        &context,
        &encode_rgb(
            &context,
            Codec::H264,
            RgbFormat::BGRA,
            ConversionPreference::Encoder,
        )?,
    )?;
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
        if !encoder_converts(&context, codec, "rgb_input_encodes_every_codec") {
            continue;
        }
        let stream = encode_rgb(
            &context,
            codec,
            RgbFormat::BGRA,
            ConversionPreference::Encoder,
        )?;
        assert!(!stream.is_empty(), "{codec:?}: empty stream");
        if let Ok(dir) = std::env::var("PIXELFORGE_RGB_STREAM_DIR") {
            std::fs::write(format!("{dir}/rgb.{codec:?}").to_lowercase(), &stream)?;
        }
    }
    Ok(())
}

/// `supports_rgb_encode_input` must agree with what an encoder forced to
/// convert can actually be created for, on every device: one that answers
/// yes for something the encoder then refuses would make `Auto` fail, and one
/// that answers no for something that works leaves the shader running for
/// nothing.
#[test]
#[ignore = "requires a Vulkan Video device"]
fn rgb_query_matches_encoder_creation() -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();
    let context = VideoContextBuilder::new()
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .build()?;
    let colors = [
        ("SDR limited", StreamColor::default()),
        (
            "SDR full",
            StreamColor::Sdr {
                range: ColorRange::Full,
            },
        ),
        (
            "HDR10 limited",
            StreamColor::Hdr10 {
                range: ColorRange::Limited,
            },
        ),
    ];
    for codec in [Codec::H264, Codec::H265, Codec::AV1] {
        if !context.supports_encode(codec) {
            continue;
        }
        for format in [
            RgbFormat::BGRA,
            RgbFormat::RGBx,
            RgbFormat::ABGR2101010,
            RgbFormat::RGBA16F,
        ] {
            let bit_depth = if format == RgbFormat::ABGR2101010 {
                BitDepth::Ten
            } else {
                BitDepth::Eight
            };
            for &(name, stream) in &colors {
                let color = if matches!(stream, StreamColor::Hdr10 { .. }) {
                    ColorSpec::Bt2020Pq
                } else {
                    ColorSpec::Srgb
                };
                let supported = context.supports_rgb_encode_input(codec, format, stream);
                let created = Encoder::new(
                    context.clone(),
                    encode_config(codec)
                        .with_bit_depth(bit_depth)
                        .with_color(stream)
                        .with_source(Source::Rgb(RgbSource::new(format, color)))
                        .with_conversion(ConversionPreference::Encoder),
                )
                .is_ok();
                eprintln!("{codec:?} {format:?} {name}: query={supported} encoder={created}");
                assert_eq!(
                    supported, created,
                    "{codec:?} {format:?} {name}: query and encoder creation disagree"
                );
            }
        }
    }
    Ok(())
}

/// A colour change keeps the conversion in step with the label, the next
/// packet is a key frame, and timestamps carry on.
#[test]
#[ignore = "requires a Vulkan Video device"]
fn color_change_follows_into_the_conversion() -> Result<(), Box<dyn std::error::Error>> {
    let context = context(Codec::H264)?;
    let mut encoder = Encoder::new(
        context.clone(),
        encode_config(Codec::H264).with_source(rgb(RgbFormat::BGRA)),
    )?;
    let conversion = encoder.conversion();
    let mut pts = Vec::new();
    let mut keys = Vec::new();
    for n in 0..FRAMES {
        if n == FRAMES / 2 {
            encoder.reconfigure(
                rgb(RgbFormat::BGRA),
                StreamColor::Sdr {
                    range: ColorRange::Full,
                },
            )?;
        }
        let src = source(&context, n)?;
        let image = source_image(&context, &src)?;
        let packet = pollster::block_on(encoder.submit(Frame::image(&image))?)?;
        pts.push(packet.pts);
        keys.push(packet.is_key_frame());
    }
    assert_eq!(pts, (0..u64::from(FRAMES)).collect::<Vec<_>>());
    assert!(keys[0] && keys[FRAMES as usize / 2], "key frames: {keys:?}");
    assert_eq!(
        encoder.config().color,
        StreamColor::Sdr {
            range: ColorRange::Full
        },
        "the colour was not kept"
    );
    // Full range is not something every encoder converts to; where it does
    // not, the shader takes over.
    println!("conversion {conversion:?} -> {:?}", encoder.conversion());
    Ok(())
}

/// A source that changes its pixel layout mid-stream is read in its new
/// layout from the next frame on: the shader switches in place, with no new
/// key frame, and the picture stays the same.
#[test]
#[ignore = "requires a Vulkan Video device"]
fn a_format_change_switches_the_shader_in_place() -> Result<(), Box<dyn std::error::Error>> {
    let context = context(Codec::H264)?;
    let reference = encode_rgb(
        &context,
        Codec::H264,
        RgbFormat::BGRA,
        ConversionPreference::Shader,
    )?;
    let mut encoder = Encoder::new(
        context.clone(),
        encode_config(Codec::H264)
            .with_source(rgb(RgbFormat::BGRA))
            .with_conversion(ConversionPreference::Shader),
    )?;
    let mut stream = Vec::new();
    let mut keys = Vec::new();
    for n in 0..FRAMES {
        let mut pixels = moving_frame(n);
        let mut format = vk::Format::B8G8R8A8_UNORM;
        if n >= FRAMES / 2 {
            encoder.reconfigure(rgb(RgbFormat::RGBA), StreamColor::default())?;
            for pixel in pixels.as_chunks_mut::<4>().0 {
                pixel.swap(0, 2);
            }
            format = vk::Format::R8G8B8A8_UNORM;
        }
        let src = unsafe { create_src_image_as(&context, WIDTH, HEIGHT, format, &pixels)? };
        let image = source_image(&context, &src)?;
        let packet = pollster::block_on(encoder.submit(Frame::image(&image))?)?;
        keys.push(packet.is_key_frame());
        stream.extend_from_slice(&packet.data);
    }
    assert_eq!(encoder.conversion(), Conversion::Shader);
    assert!(
        !keys[FRAMES as usize / 2],
        "the format change made a key frame: {keys:?}"
    );
    let expected = decode_all(&context, &reference)?;
    let actual = decode_all(&context, &stream)?;
    assert_eq!(actual.len(), FRAMES as usize, "frame count");
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        let p = psnr(a, e);
        assert!(
            p >= 40.0,
            "frame {i}: {p:.2} dB against the unchanged source"
        );
    }
    Ok(())
}
