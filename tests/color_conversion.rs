//! What the colour converter writes, checked against the same maths on the CPU.
//!
//! Colour bugs here do not look like failures. They look like a normal picture
//! that is slightly too dark, or the wrong shade. So rather than eyeball it,
//! these tests compute what each conversion should produce and compare.
//!
//! **Quantization.** The shader turns floating-point YUV into integer values.
//! If it truncates instead of rounding, every pixel comes out up to one step
//! too dark. Each sample is scored against both possibilities.
//!
//! **Conversion paths.** Which [`ColorSpec`] pair you pick decides which steps
//! the shader runs: an sRGB decode, a gamut conversion, a PQ encode, or none of
//! them. The same steps are redone on the CPU, picked from the source and
//! target rather than from the shader's branches, so a wrong branch shows up as
//! a large disagreement.
//!
//! Luma only, since chroma is averaged over 2x2 blocks and that averaging would
//! muddy the comparison. The test frame is deliberately not grey: grey converts
//! to exact values and would hide the very errors being looked for.
//!
//! Ignored by default: requires a Vulkan Video device. Run with
//! `cargo test -- --ignored`.

#[allow(dead_code)]
mod common;
use common::source::{SrcImage, create_src_image, host_buffer, one_shot};

use ash::vk;
use pixelforge::{
    Codec, ColorConverter, ColorConverterConfig, ColorDescription, ColorRange, ColorSpec,
    EncodeBitDepth, EncodeConfig, Encoder, InputFormat, OutputFormat, RateControlMode,
    VideoContext, VideoContextBuilder,
};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 64;

/// Deterministic colours that are never grey, so every channel differs and the
/// luma lands between code values often enough to be worth measuring.
fn colour_at(x: u32, y: u32) -> [u8; 3] {
    [
        (x ^ y.wrapping_mul(37)) as u8,
        x.wrapping_mul(7).wrapping_add(y.wrapping_mul(11)) as u8,
        x.wrapping_mul(13).wrapping_add(y.wrapping_mul(3)) as u8,
    ]
}

fn make_frame_bgra() -> Vec<u8> {
    let mut data = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let [r, g, b] = colour_at(x, y);
            data.extend_from_slice(&[b, g, r, 255]);
        }
    }
    data
}

// --- CPU model of each shader stage, same order, same constants. ---

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn bt709_to_bt2020(rgb: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = rgb;
    [
        0.6274 * r + 0.3293 * g + 0.0433 * b,
        0.0691 * r + 0.9195 * g + 0.0114 * b,
        0.0164 * r + 0.0880 * g + 0.8956 * b,
    ]
}

fn linear_to_pq(l: f32) -> f32 {
    // Computed in f64. This is the reference; the shader's own f32 rounding is
    // what the one-code-value tolerance covers.
    const M1: f64 = 0.1593017578125;
    const M2: f64 = 78.84375;
    const C1: f64 = 0.8359375;
    const C2: f64 = 18.8515625;
    const C3: f64 = 18.6875;
    let lm1 = (l.max(0.0) as f64).powf(M1);
    (((C1 + C2 * lm1) / (1.0 + C3 * lm1)).powf(M2)) as f32
}

/// The signal the shader should arrive at for this source and target, then its
/// luma, then its code value. Mirrors `read_rgb` and `rgb_to_yuv`.
fn expected_code(
    rgb: [u8; 3],
    source: ColorSpec,
    target: ColorSpec,
    ten_bit: bool,
    full: bool,
) -> f32 {
    let mut v = [
        rgb[0] as f32 / 255.0,
        rgb[1] as f32 / 255.0,
        rgb[2] as f32 / 255.0,
    ];
    if target == ColorSpec::Bt2020Pq && source != ColorSpec::Bt2020Pq {
        if matches!(source, ColorSpec::Srgb) {
            v = [
                srgb_to_linear(v[0]),
                srgb_to_linear(v[1]),
                srgb_to_linear(v[2]),
            ];
        }
        if !matches!(source, ColorSpec::Bt2020Linear) {
            v = bt709_to_bt2020(v);
        }
        let nits = source.reference_white_nits().unwrap();
        v = [
            linear_to_pq(v[0] * (nits / 10000.0)),
            linear_to_pq(v[1] * (nits / 10000.0)),
            linear_to_pq(v[2] * (nits / 10000.0)),
        ];
    }
    let y = if target == ColorSpec::Bt2020Pq {
        0.2627f32 * v[0] + 0.6780f32 * v[1] + 0.0593f32 * v[2]
    } else {
        0.2126f32 * v[0] + 0.7152f32 * v[1] + 0.0722f32 * v[2]
    }
    .clamp(0.0, 1.0);
    match (ten_bit, full) {
        (false, true) => (y * 255.0).clamp(0.0, 255.0),
        (false, false) => (y * 219.0 + 16.0).clamp(0.0, 255.0),
        (true, true) => (y * 1023.0).clamp(0.0, 1023.0),
        (true, false) => (y * 876.0 + 64.0).clamp(0.0, 1023.0),
    }
}

/// Read the converter's luma plane back as code values.
fn read_luma(
    context: &VideoContext,
    converter: &ColorConverter,
    output_format: OutputFormat,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let sample_bytes = output_format.bytes_per_sample() as u64;
    let plane_bytes = (WIDTH * HEIGHT) as u64 * sample_bytes;
    let device = context.device();
    let (readback, readback_mem) =
        unsafe { host_buffer(context, plane_bytes, vk::BufferUsageFlags::TRANSFER_DST)? };
    unsafe {
        one_shot(context, |cb| {
            let copy = vk::BufferCopy::default().size(plane_bytes);
            device.cmd_copy_buffer(cb, converter.output_buffer(), readback, &[copy]);
        })?;
    }
    let mut bytes = vec![0u8; plane_bytes as usize];
    unsafe {
        let ptr = device.map_memory(readback_mem, 0, plane_bytes, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(ptr as *const u8, bytes.as_mut_ptr(), bytes.len());
        device.unmap_memory(readback_mem);
        device.destroy_buffer(readback, None);
        device.free_memory(readback_mem, None);
    }
    Ok(if sample_bytes == 2 {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|w| (u16::from_le_bytes(*w) >> 6) as u32)
            .collect()
    } else {
        bytes.iter().map(|&b| b as u32).collect()
    })
}

/// One 8-bit and one 10-bit encoder, for the conversion's target image.
///
/// 10-bit needs a Main10 encoder, which not every device exposes, so a missing
/// one skips the 10-bit half rather than failing the test.
fn encoders(
    context: &VideoContext,
) -> Result<(Encoder, Option<Encoder>), Box<dyn std::error::Error>> {
    let eight = Encoder::new(
        context.clone(),
        EncodeConfig::h264(WIDTH, HEIGHT)
            .with_rate_control(RateControlMode::Cqp)
            .with_frame_rate(30, 1)
            .with_b_frames(0),
    )?;
    let ten = Encoder::new(
        context.clone(),
        EncodeConfig::h265(WIDTH, HEIGHT)
            .with_bit_depth(EncodeBitDepth::Ten)
            .with_rate_control(RateControlMode::Cqp)
            .with_frame_rate(30, 1)
            .with_b_frames(0),
    )
    .ok();
    if ten.is_none() {
        println!("no 10-bit H.265 encoder on this device: P010 cases skipped");
    }
    Ok((eight, ten))
}

fn context() -> Result<VideoContext, Box<dyn std::error::Error>> {
    common::init_logging();
    Ok(VideoContextBuilder::new()
        .app_name("pixelforge-color-conversion")
        .require_encode(Codec::H264)
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .build()?)
}

/// Convert one frame and read its luma plane back as code values.
fn convert(
    context: &VideoContext,
    src: &SrcImage,
    encoder: &Encoder,
    source: ColorSpec,
    target: ColorSpec,
    output_format: OutputFormat,
    range: ColorRange,
) -> Result<(Vec<u32>, ColorDescription), Box<dyn std::error::Error>> {
    let config = ColorConverterConfig::new(
        WIDTH,
        HEIGHT,
        InputFormat::BGRA,
        output_format,
        source,
        target,
        range,
    );
    let mut converter = ColorConverter::new(context.clone(), config)?;
    let description = converter.color_description();
    converter.convert(src.image, vk::ImageLayout::GENERAL, encoder.input_image())?;
    Ok((read_luma(context, &converter, output_format)?, description))
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn quantizers_round_rather_than_truncate() -> Result<(), Box<dyn std::error::Error>> {
    let context = context()?;
    let src = unsafe { create_src_image(&context, WIDTH, HEIGHT, &make_frame_bgra())? };
    let (eight, ten) = encoders(&context)?;

    for (output_format, encoder) in [
        (OutputFormat::NV12, Some(&eight)),
        (OutputFormat::P010, ten.as_ref()),
    ] {
        let Some(encoder) = encoder else { continue };
        let ten_bit = output_format.bytes_per_sample() == 2;
        for range in [ColorRange::Limited, ColorRange::Full] {
            let full = range.is_full();
            let (codes, _) = convert(
                &context,
                &src,
                encoder,
                ColorSpec::Srgb,
                ColorSpec::Srgb,
                output_format,
                range,
            )?;

            let mut if_truncating = 0usize;
            let mut if_rounding = 0usize;
            let mut signed_error = 0f64;
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    let got = codes[(y * WIDTH + x) as usize];
                    let ideal = expected_code(
                        colour_at(x, y),
                        ColorSpec::Srgb,
                        ColorSpec::Srgb,
                        ten_bit,
                        full,
                    );
                    if got != ideal as u32 {
                        if_truncating += 1;
                    }
                    if got != ideal.round() as u32 {
                        if_rounding += 1;
                    }
                    signed_error += got as f64 - ideal as f64;
                }
            }
            let samples = (WIDTH * HEIGHT) as f64;
            let bias = signed_error / samples;
            println!(
                "{output_format:?} {range:?}: {if_truncating} mismatches if truncating, \
                 {if_rounding} if rounding, mean signed error {bias:+.4}"
            );

            // Truncation biases every sample low by about half a code value,
            // which is what this is really watching for. The tie tolerance is
            // for the GPU ordering its f32 arithmetic differently from us.
            assert!(
                if_rounding < if_truncating,
                "{output_format:?} {range:?}: looks like it truncates"
            );
            assert!(
                bias.abs() < 0.05,
                "{output_format:?} {range:?}: mean signed error {bias:+.4} is a bias, not noise"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn every_conversion_matches_the_model() -> Result<(), Box<dyn std::error::Error>> {
    let context = context()?;
    let src = unsafe { create_src_image(&context, WIDTH, HEIGHT, &make_frame_bgra())? };
    let (eight, ten) = encoders(&context)?;

    let supported = [
        (ColorSpec::Srgb, ColorSpec::Srgb),
        (ColorSpec::Srgb, ColorSpec::Bt2020Pq),
        (ColorSpec::Bt709Linear, ColorSpec::Bt2020Pq),
        (ColorSpec::Bt2020Linear, ColorSpec::Bt2020Pq),
        (ColorSpec::Bt2020Pq, ColorSpec::Bt2020Pq),
    ];

    for (source, target) in supported {
        for (output_format, encoder) in [
            (OutputFormat::NV12, Some(&eight)),
            (OutputFormat::P010, ten.as_ref()),
        ] {
            let Some(encoder) = encoder else { continue };
            let ten_bit = output_format.bytes_per_sample() == 2;
            let (codes, description) = convert(
                &context,
                &src,
                encoder,
                source,
                target,
                output_format,
                ColorRange::Full,
            )?;

            let mut worst = 0f64;
            let mut sum = 0f64;
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    let got = codes[(y * WIDTH + x) as usize] as f64;
                    let want = expected_code(colour_at(x, y), source, target, ten_bit, true).round()
                        as f64;
                    let d = (got - want).abs();
                    worst = worst.max(d);
                    sum += d;
                }
            }
            let mean = sum / (WIDTH * HEIGHT) as f64;
            println!(
                "{source:?} -> {target:?} {output_format:?}: worst {worst:.1}, mean {mean:.4}, \
                 VUI {}/{}/{} {}",
                description.color_primaries,
                description.transfer_characteristics,
                description.matrix_coefficients,
                if description.full_range { "pc" } else { "tv" },
            );

            // The declaration has to describe the target, not the source.
            assert_eq!(
                Some(description.with_full_range(false)),
                target.color_description(),
                "{source:?} -> {target:?}: declared the wrong space"
            );
            // One code value of slack: the GPU is free to order and fuse these
            // f32 operations differently, which moves samples landing on a tie.
            assert!(
                worst <= 1.0,
                "{source:?} -> {target:?} {output_format:?}: worst {worst:.1} code values off"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn unsupported_conversions_are_refused() -> Result<(), Box<dyn std::error::Error>> {
    let context = context()?;

    let refused = [
        // Converting to SDR: would need a gamma curve, or tone mapping.
        (ColorSpec::Bt709Linear, ColorSpec::Srgb),
        (ColorSpec::Bt2020Linear, ColorSpec::Srgb),
        (ColorSpec::Bt2020Pq, ColorSpec::Srgb),
        // Video cannot be encoded in a linear space.
        (ColorSpec::Srgb, ColorSpec::Bt709Linear),
        (ColorSpec::Srgb, ColorSpec::Bt2020Linear),
    ];

    for (source, target) in refused {
        let config = ColorConverterConfig::new(
            WIDTH,
            HEIGHT,
            InputFormat::BGRA,
            OutputFormat::NV12,
            source,
            target,
            ColorRange::Full,
        );
        assert_eq!(
            config.color_description().is_some(),
            target.is_encodable(),
            "{target:?}: color_description() and is_encodable() disagree"
        );
        let error = ColorConverter::new(context.clone(), config)
            .err()
            .unwrap_or_else(|| panic!("{source:?} -> {target:?} was accepted"));
        println!("{source:?} -> {target:?}: {error}");
    }
    Ok(())
}

/// A source image destroyed and replaced by a new one must be read as the new
/// one, even when the driver hands the new image the old one's handle.
///
/// Drivers do reuse handles, so a converter that remembered anything by image
/// handle would read the second frame through a view of the first, destroyed
/// image. The validation layer gives every object a unique handle, which hides
/// exactly this, so the test means most with validation off.
#[test]
#[ignore = "requires a Vulkan Video device"]
fn a_replaced_source_image_is_read_afresh() -> Result<(), Box<dyn std::error::Error>> {
    let context = context()?;
    let (encoder, _) = encoders(&context)?;
    let first = make_frame_bgra();
    let second: Vec<u8> = first
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|&[b, g, r, a]| [255 - b, 255 - g, 255 - r, a])
        .collect();

    let config = ColorConverterConfig::new(
        WIDTH,
        HEIGHT,
        InputFormat::BGRA,
        OutputFormat::NV12,
        ColorSpec::Srgb,
        ColorSpec::Srgb,
        ColorRange::Limited,
    );
    let convert_with = |converter: &mut ColorConverter, pixels: &[u8]| {
        let src = unsafe { create_src_image(&context, WIDTH, HEIGHT, pixels)? };
        converter.convert(src.image, vk::ImageLayout::GENERAL, encoder.input_image())?;
        let luma = read_luma(&context, converter, OutputFormat::NV12)?;
        drop(src);
        Ok::<_, Box<dyn std::error::Error>>(luma)
    };

    let expected = convert_with(
        &mut ColorConverter::new(context.clone(), config.clone())?,
        &second,
    )?;
    let mut converter = ColorConverter::new(context.clone(), config)?;
    convert_with(&mut converter, &first)?;
    let actual = convert_with(&mut converter, &second)?;
    assert!(
        actual == expected,
        "the second frame came out as something other than the second frame"
    );
    Ok(())
}
