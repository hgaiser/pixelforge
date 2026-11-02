//! What the frames an encoder is given are, and how they become YUV.
//!
//! The caller describes the source once, in [`EncodeConfig::source`]. The
//! encoder then picks the [`Conversion`] that gets it to the stream's YUV, so
//! a caller never chooses between the colour converter and the encoder's own
//! RGB conversion, and always submits frames the same way.

use super::rgb::rgb_input_bit_depth;
use super::{BitDepth, ColorDescription, EncodeConfig, PixelFormat};
use crate::converter::{ColorConverterConfig, ColorRange, ColorSpec, InputFormat, OutputFormat};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;

/// What the frames handed to an [`Encoder`](crate::Encoder) are.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Source {
    /// YUV frames, already in the encoder's pixel format and bit depth.
    #[default]
    Yuv,
    /// RGB frames, which pixelforge converts to YUV.
    Rgb(RgbSource),
}

/// RGB frames, and what their pixels mean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RgbSource {
    /// How the pixels are laid out.
    pub format: InputFormat,
    /// What the pixel values mean.
    pub color: ColorSpec,
    /// How bright the source's white is, in nits, overriding
    /// [`ColorSpec::reference_white_nits`]. Only used when encoding SDR
    /// content to HDR.
    pub reference_white_nits: Option<f32>,
}

impl RgbSource {
    /// RGB frames in `format`, holding colours in `color`.
    pub fn new(format: InputFormat, color: ColorSpec) -> Self {
        Self {
            format,
            color,
            reference_white_nits: None,
        }
    }

    /// Override how bright the source's white is, in nits.
    pub fn with_reference_white_nits(mut self, nits: f32) -> Self {
        self.reference_white_nits = Some(nits);
        self
    }
}

/// Which conversion an encoder should use for an RGB [`Source`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConversionPreference {
    /// The encoder's own conversion where the device can do it, the colour
    /// converter otherwise.
    #[default]
    Auto,
    /// The encoder's own conversion, or fail.
    Encoder,
    /// The colour converter's compute shader, even where the encoder could
    /// convert.
    Shader,
}

/// The conversion an [`Encoder`](crate::Encoder) uses, see
/// [`Encoder::conversion`](crate::Encoder::conversion).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conversion {
    /// The source is already YUV.
    None,
    /// The encoder converts RGB itself, through
    /// `VK_VALVE_video_encode_rgb_conversion`. Its input image is RGB.
    Encoder,
    /// The colour converter converts RGB into the encoder's YUV input image.
    Shader,
}

/// The conversion an encoder will use, and what it needs for it.
pub(crate) struct Plan {
    pub conversion: Conversion,
    /// The RGB format the encoder converts from, for [`Conversion::Encoder`].
    pub rgb_input: Option<InputFormat>,
    /// The colour converter's configuration, for [`Conversion::Shader`].
    pub converter: Option<ColorConverterConfig>,
}

/// The colour space a stream described by `desc` is in, if a conversion
/// can produce it.
pub(crate) fn target_space(desc: &ColorDescription) -> Option<ColorSpec> {
    [ColorSpec::Srgb, ColorSpec::Bt2020Pq]
        .into_iter()
        .find(|space| {
            space.color_description().is_some_and(|d| {
                (
                    d.color_primaries,
                    d.transfer_characteristics,
                    d.matrix_coefficients,
                ) == (
                    desc.color_primaries,
                    desc.transfer_characteristics,
                    desc.matrix_coefficients,
                )
            })
        })
}

/// The YUV the colour converter has to write for this encoder.
fn shader_output(pixel_format: PixelFormat, bit_depth: BitDepth) -> Option<OutputFormat> {
    match (pixel_format, bit_depth) {
        (PixelFormat::Yuv420, BitDepth::Eight) => Some(OutputFormat::NV12),
        (PixelFormat::Yuv420, BitDepth::Ten) => Some(OutputFormat::P010),
        (PixelFormat::Yuv444, BitDepth::Eight) => Some(OutputFormat::YUV444),
        (PixelFormat::Yuv444, BitDepth::Ten) => Some(OutputFormat::YUV444P10),
        _ => None,
    }
}

/// Pick the conversion for `config` on this device.
pub(crate) fn plan(context: &VideoContext, config: &EncodeConfig) -> Result<Plan> {
    let rgb = match config.source {
        Source::Yuv => {
            if config.conversion != ConversionPreference::Auto {
                return Err(PixelForgeError::InvalidInput(format!(
                    "conversion preference {:?} needs an RGB source",
                    config.conversion
                )));
            }
            return Ok(Plan {
                conversion: Conversion::None,
                rgb_input: None,
                converter: None,
            });
        }
        Source::Rgb(rgb) => rgb,
    };

    let desc = config
        .color_description
        .unwrap_or_else(ColorDescription::bt709);
    let target = target_space(&desc).ok_or_else(|| {
        PixelForgeError::InvalidInput(format!(
            "no conversion from RGB produces the colour description {desc:?}; \
             use BT.709 or BT.2020 with PQ"
        ))
    })?;

    let encoder = encoder_conversion(context, config, &rgb, target, &desc);
    let shader = shader_conversion(context, config, &rgb, target, &desc);

    let use_encoder = match config.conversion {
        ConversionPreference::Encoder => {
            encoder?;
            true
        }
        ConversionPreference::Shader => false,
        ConversionPreference::Auto => match &encoder {
            Ok(()) => true,
            Err(e) => {
                tracing::debug!("RGB input: the encoder cannot convert ({e}); using the shader");
                false
            }
        },
    };
    if use_encoder {
        return Ok(Plan {
            conversion: Conversion::Encoder,
            rgb_input: Some(rgb.format),
            converter: None,
        });
    }
    Ok(Plan {
        conversion: Conversion::Shader,
        rgb_input: None,
        converter: Some(shader?),
    })
}

/// Why the encoder cannot do this conversion itself, if it cannot.
fn encoder_conversion(
    context: &VideoContext,
    config: &EncodeConfig,
    rgb: &RgbSource,
    target: ColorSpec,
    desc: &ColorDescription,
) -> Result<()> {
    let no = |reason: String| {
        Err(PixelForgeError::InvalidInput(format!(
            "RGB input: {reason}"
        )))
    };
    if !context.has_video_encode_rgb_conversion() {
        return no("the device has no VK_VALVE_video_encode_rgb_conversion".to_string());
    }
    if rgb.color != target {
        return no(format!(
            "{:?} to {target:?} needs more than a YUV matrix",
            rgb.color
        ));
    }
    match rgb_input_bit_depth(rgb.format) {
        Some(depth) if config.pixel_format == PixelFormat::Yuv420 && config.bit_depth == depth => {}
        Some(depth) => {
            return no(format!(
                "{:?} converts to {depth:?} 4:2:0, the encoder is set up for {:?} {:?}",
                rgb.format, config.bit_depth, config.pixel_format
            ));
        }
        None => return no(format!("{:?} needs tone mapping first", rgb.format)),
    }
    if !context.supports_rgb_encode_input(config.codec, rgb.format, desc) {
        return no(format!(
            "the driver does not convert {:?} to {desc:?} for {:?}",
            rgb.format, config.codec
        ));
    }
    Ok(())
}

/// The colour converter's configuration for this conversion, or why the
/// shader cannot do it.
fn shader_conversion(
    context: &VideoContext,
    config: &EncodeConfig,
    rgb: &RgbSource,
    target: ColorSpec,
    desc: &ColorDescription,
) -> Result<ColorConverterConfig> {
    if !context.has_push_descriptor() {
        return Err(PixelForgeError::NoSuitableDevice(
            "RGB input: the colour converter needs VK_KHR_push_descriptor".to_string(),
        ));
    }
    let output = shader_output(config.pixel_format, config.bit_depth).ok_or_else(|| {
        PixelForgeError::InvalidInput(format!(
            "RGB input: the colour converter does not write {:?} {:?}",
            config.pixel_format, config.bit_depth
        ))
    })?;
    let range = if desc.full_range {
        ColorRange::Full
    } else {
        ColorRange::Limited
    };
    let mut converter = ColorConverterConfig::new(
        config.dimensions.width,
        config.dimensions.height,
        rgb.format,
        output,
        rgb.color,
        target,
        range,
    );
    converter.reference_white_nits = rgb.reference_white_nits;
    if !converter.conversion_supported() {
        return Err(converter.unsupported_conversion());
    }
    Ok(converter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_space_follows_the_description() {
        assert_eq!(
            target_space(&ColorDescription::bt709()),
            Some(ColorSpec::Srgb)
        );
        assert_eq!(
            target_space(&ColorDescription::bt709().with_full_range(true)),
            Some(ColorSpec::Srgb)
        );
        assert_eq!(
            target_space(&ColorDescription::bt2020_pq()),
            Some(ColorSpec::Bt2020Pq)
        );
        let bt601 = ColorDescription {
            matrix_coefficients: 6,
            ..ColorDescription::bt709()
        };
        assert_eq!(target_space(&bt601), None);
    }

    #[test]
    fn shader_output_matches_the_encoder() {
        assert_eq!(
            shader_output(PixelFormat::Yuv420, BitDepth::Ten),
            Some(OutputFormat::P010)
        );
        assert_eq!(shader_output(PixelFormat::Yuv422, BitDepth::Eight), None);
    }
}
