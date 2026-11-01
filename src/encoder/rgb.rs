//! RGB input converted to YUV by the encoder itself, through
//! `VK_VALVE_video_encode_rgb_conversion`.
//!
//! The extension adds one struct to the video profile, which then means "this
//! encoder takes RGB pictures". Everything that names the profile has to name
//! that version of it: the capability and format queries, the input and DPB
//! images, the query pools and the session. The session also says which
//! matrix, range and chroma siting to convert with.

use super::support::report;
use super::{BitDepth, Codec, ColorDescription, EncodeConfig, PixelFormat};
use crate::converter::InputFormat;
use crate::error::{PixelForgeError, Result};
use crate::video::query_supported_video_formats;
use crate::vulkan::VideoContext;
use ash::vk;

/// The profile struct that turns on RGB input.
pub(crate) fn rgb_conversion_profile() -> vk::VideoEncodeProfileRgbConversionInfoVALVE<'static> {
    vk::VideoEncodeProfileRgbConversionInfoVALVE::default().perform_encode_rgb_conversion(true)
}

/// The YUV bit depth an RGB input format converts to, or `None` for a format
/// the hardware conversion does not take.
///
/// 8-bit RGB becomes 8-bit YUV and 10-bit RGB 10-bit YUV. Half floats carry
/// values outside the 0 to 1 range that need tone mapping first, which only
/// the colour converter does.
pub(crate) fn rgb_input_bit_depth(format: InputFormat) -> Option<BitDepth> {
    match format {
        InputFormat::BGRx | InputFormat::BGRA | InputFormat::RGBx | InputFormat::RGBA => {
            Some(BitDepth::Eight)
        }
        InputFormat::ABGR2101010 => Some(BitDepth::Ten),
        InputFormat::RGBA16F => None,
    }
}

/// Refuse an RGB input configuration that cannot work, before any driver is
/// asked about it.
///
/// The device must have the extension, and the encoder must be set up for
/// the YUV the conversion produces: 4:2:0, at 8 bits from 8-bit RGB and 10
/// bits from 10-bit RGB.
pub(crate) fn check_rgb_input(
    context: &VideoContext,
    config: &EncodeConfig,
    format: InputFormat,
) -> Result<()> {
    if !context.has_video_encode_rgb_conversion() {
        return Err(PixelForgeError::NoSuitableDevice(
            "RGB input needs VK_VALVE_video_encode_rgb_conversion, which this device does \
             not have enabled"
                .to_string(),
        ));
    }
    let Some(bit_depth) = rgb_input_bit_depth(format) else {
        return Err(PixelForgeError::InvalidInput(format!(
            "RGB input: {format:?} needs tone mapping first, which only the colour \
             converter does"
        )));
    };
    if config.pixel_format != PixelFormat::Yuv420 || config.bit_depth != bit_depth {
        return Err(PixelForgeError::InvalidInput(format!(
            "RGB input: {format:?} converts to {bit_depth:?} 4:2:0, but the encoder is set \
             up for {:?} {:?}",
            config.bit_depth, config.pixel_format
        )));
    }
    Ok(())
}

/// What the driver's RGB conversion can do for one profile:
/// `VkVideoEncodeRgbConversionCapabilitiesVALVE` without its `pNext`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RgbConversionCaps {
    models: vk::VideoEncodeRgbModelConversionFlagsVALVE,
    ranges: vk::VideoEncodeRgbRangeCompressionFlagsVALVE,
    x_chroma_offsets: vk::VideoEncodeRgbChromaOffsetFlagsVALVE,
    y_chroma_offsets: vk::VideoEncodeRgbChromaOffsetFlagsVALVE,
}

impl From<&vk::VideoEncodeRgbConversionCapabilitiesVALVE<'_>> for RgbConversionCaps {
    fn from(caps: &vk::VideoEncodeRgbConversionCapabilitiesVALVE<'_>) -> Self {
        Self {
            models: caps.rgb_models,
            ranges: caps.rgb_ranges,
            x_chroma_offsets: caps.x_chroma_offsets,
            y_chroma_offsets: caps.y_chroma_offsets,
        }
    }
}

impl RgbConversionCaps {
    /// The session settings that make the hardware produce what `desc`
    /// describes, or why it cannot.
    pub(crate) fn session_info(
        &self,
        desc: &ColorDescription,
    ) -> Result<vk::VideoEncodeSessionRgbConversionCreateInfoVALVE<'static>> {
        use vk::VideoEncodeRgbChromaOffsetFlagsVALVE as Offset;
        use vk::VideoEncodeRgbModelConversionFlagsVALVE as Model;
        use vk::VideoEncodeRgbRangeCompressionFlagsVALVE as Range;

        // The matrix coefficients code the stream carries (H.273 numbering).
        let model = match desc.matrix_coefficients {
            ColorDescription::MATRIX_BT709 => Model::YCBCR_709,
            5 | 6 => Model::YCBCR_601,
            ColorDescription::MATRIX_BT2020_NCL => Model::YCBCR_2020,
            other => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "RGB input: no hardware conversion for matrix coefficients {other}"
                )));
            }
        };
        let range = if desc.full_range {
            Range::FULL_RANGE
        } else {
            Range::NARROW_RANGE
        };
        if !self.models.contains(model) || !self.ranges.contains(range) {
            return Err(PixelForgeError::NoSuitableDevice(format!(
                "RGB input: the driver converts {:?} in {:?}, not {:?} in {:?}",
                self.models, self.ranges, model, range
            )));
        }

        // Where chroma samples sit relative to luma. The stream does not say,
        // and a player that is not told assumes chroma sits on the left luma
        // sample of each pair, and midway between the two rows (H.264 and
        // H.265 chroma location type 0, which most AV1 players follow as well).
        // Asking the hardware for exactly that keeps colour edges where the
        // player will put them. A driver that cannot gets the other option.
        let pick = |supported: Offset, preferred: Offset| {
            if supported.contains(preferred) {
                preferred
            } else if supported.contains(Offset::COSITED_EVEN) {
                Offset::COSITED_EVEN
            } else {
                Offset::MIDPOINT
            }
        };
        Ok(
            vk::VideoEncodeSessionRgbConversionCreateInfoVALVE::default()
                .rgb_model(model)
                .rgb_range(range)
                .x_chroma_offset(pick(self.x_chroma_offsets, Offset::COSITED_EVEN))
                .y_chroma_offset(pick(self.y_chroma_offsets, Offset::MIDPOINT)),
        )
    }
}

impl VideoContext {
    /// Whether an [`Encoder`](crate::Encoder) for `codec` can take `format`
    /// RGB input and convert it to the YUV that `desc` describes itself,
    /// through `VK_VALVE_video_encode_rgb_conversion`.
    ///
    /// Asks the driver everything [`Encoder::new`](crate::Encoder::new) will
    /// need: that the device has the extension, that the codec's RGB input
    /// profile exists, that its conversion does `desc`'s matrix and range,
    /// and that it accepts `format` as an input image. The output is always
    /// 4:2:0, at 8 bits for 8-bit RGB and 10 bits for
    /// [`InputFormat::ABGR2101010`]. Resolution and rate control are not
    /// considered.
    ///
    /// When this is true, build the encoder with
    /// [`EncodeConfig::with_rgb_input`](crate::EncodeConfig::with_rgb_input)
    /// and [`EncodeConfig::with_color_description`](crate::EncodeConfig::with_color_description)
    /// set to `desc`. [`ColorConverterConfig::rgb_encode_input`](crate::ColorConverterConfig::rgb_encode_input)
    /// asks this for a conversion it would otherwise run.
    pub fn supports_rgb_encode_input(
        &self,
        codec: Codec,
        format: InputFormat,
        desc: &ColorDescription,
    ) -> bool {
        if !self.has_video_encode_rgb_conversion() || !self.supports_encode(codec) {
            return false;
        }
        let Some(bit_depth) = rgb_input_bit_depth(format) else {
            return false;
        };

        let mut caps = vk::VideoEncodeRgbConversionCapabilitiesVALVE::default();
        let what = format!("{codec:?} encode from {format:?} RGB");
        let formats = self.query_encode_profile(
            codec,
            PixelFormat::Yuv420,
            bit_depth,
            Some(&mut caps),
            |profile| {
                query_supported_video_formats(
                    self,
                    profile,
                    vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR,
                )
            },
        );
        let formats = match formats {
            Ok(Ok(formats)) => Ok(formats),
            Ok(Err(e)) => {
                tracing::warn!("Failed to query {what} input formats: {e}");
                return false;
            }
            Err(e) => Err(e),
        };
        let Some(formats) = report(&what, formats) else {
            return false;
        };
        if !formats.iter().any(|f| f.format == format.vk_format()) {
            tracing::debug!(
                "{what}: {:?} is not an accepted input format",
                format.vk_format()
            );
            return false;
        }
        if let Err(e) = RgbConversionCaps::from(&caps).session_info(desc) {
            tracing::debug!("{what}: {e}");
            return false;
        }
        true
    }
}
