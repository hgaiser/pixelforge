//! Encoder types, configuration, and shared utilities.
//!
//! This module provides:
//! - Core encoder types and configuration (`EncodeConfig`, `EncodedPacket`, etc.)
//! - GOP structure management (`gop` module) - reusable for H.264/H.265.
//! - Frame reordering for B-frame support (`reorder` module) - reusable for H.264/H.265.

pub mod av1;
pub mod bitwriter;
pub mod dpb;
pub mod gop;
pub mod h264;
pub mod h265;
pub mod reorder;
pub mod resources;

use ash::vk;

// Default encoder configuration constants.

/// Default target bitrate in bits per second (4 Mbps).
pub const DEFAULT_TARGET_BITRATE: u32 = 4_000_000;

/// Default maximum bitrate in bits per second (6 Mbps).
pub const DEFAULT_MAX_BITRATE: u32 = 6_000_000;

/// Default frame rate (frames per second).
pub const DEFAULT_FRAME_RATE: u32 = 30;

/// Default GOP (Group of Pictures) size.
pub const DEFAULT_GOP_SIZE: u32 = 30;

/// Default QP (quantization parameter) for H.264.
pub const DEFAULT_H264_QP: u32 = 26;

/// Default QP (quantization parameter) for H.265.
pub const DEFAULT_H265_QP: u32 = 28;

/// Default maximum number of reference frames.
pub const DEFAULT_MAX_REFERENCE_FRAMES: u32 = 4;

use crate::error::Result;
use crate::vulkan::VideoContext;

/// Video codec types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    /// H.264/AVC codec.
    H264,
    /// H.265/HEVC codec.
    H265,
    /// AV1 codec.
    AV1,
}

/// Pixel format / chroma subsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PixelFormat {
    /// YUV 4:2:0 (half horizontal and vertical chroma resolution).
    #[default]
    Yuv420,
    /// YUV 4:2:2 (half horizontal chroma resolution).
    Yuv422,
    /// YUV 4:4:4 (full chroma resolution).
    Yuv444,
}

impl From<PixelFormat> for vk::VideoChromaSubsamplingFlagsKHR {
    fn from(format: PixelFormat) -> Self {
        match format {
            PixelFormat::Yuv420 => vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
            PixelFormat::Yuv422 => vk::VideoChromaSubsamplingFlagsKHR::TYPE_422,
            PixelFormat::Yuv444 => vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
        }
    }
}

impl PixelFormat {
    /// Calculate frame size in bytes for given dimensions.
    pub fn frame_size(&self, width: u32, height: u32) -> usize {
        let luma_size = (width * height) as usize;
        match self {
            PixelFormat::Yuv420 => luma_size * 3 / 2, // Y + U/4 + V/4
            PixelFormat::Yuv422 => luma_size * 2,     // Y + U/2 + V/2
            PixelFormat::Yuv444 => luma_size * 3,     // Y + U + V
        }
    }
}

/// Bit depth for video encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BitDepth {
    /// 8-bit per component (standard).
    #[default]
    Eight,
    /// 10-bit per component (HDR, Main10 profile).
    Ten,
}

impl From<BitDepth> for vk::VideoComponentBitDepthFlagsKHR {
    fn from(depth: BitDepth) -> Self {
        match depth {
            BitDepth::Eight => vk::VideoComponentBitDepthFlagsKHR::TYPE_8,
            BitDepth::Ten => vk::VideoComponentBitDepthFlagsKHR::TYPE_10,
        }
    }
}

/// Rate control modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RateControlMode {
    /// Disabled rate control - constant QP.
    #[default]
    Disabled,
    /// Constant QP mode.
    Cqp,
    /// Constant bitrate mode.
    Cbr,
    /// Variable bitrate mode.
    Vbr,
}

/// Frame types in encoded stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// Instantaneous Decoder Refresh frame.
    Idr,
    /// Intra-coded frame.
    I,
    /// Predicted frame.
    P,
    /// Bi-predicted frame.
    B,
    /// Unknown frame type.
    Unknown,
}

/// Video dimensions.
#[derive(Debug, Clone, Copy)]
pub struct Dimensions {
    pub width: u32,
    pub height: u32,
}

/// Video signal color description for VUI parameters.
///
/// Describes how color is encoded in the video stream, allowing decoders
/// to correctly interpret the color space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorDescription {
    /// Color primaries (1=BT.709, 9=BT.2020).
    pub color_primaries: u8,
    /// Transfer characteristics (1=BT.709, 16=ST2084/PQ).
    pub transfer_characteristics: u8,
    /// Matrix coefficients (1=BT.709, 9=BT.2020 NCL).
    pub matrix_coefficients: u8,
    /// Full range (true) or limited/TV range (false).
    pub full_range: bool,
}

impl ColorDescription {
    /// BT.709 color description (standard SDR).
    pub fn bt709() -> Self {
        Self {
            color_primaries: 1,
            transfer_characteristics: 1,
            matrix_coefficients: 1,
            full_range: true,
        }
    }

    /// BT.2020 with PQ transfer function (HDR10).
    pub fn bt2020_pq() -> Self {
        Self {
            color_primaries: 9,
            transfer_characteristics: 16,
            matrix_coefficients: 9,
            full_range: false,
        }
    }
}

/// Encode configuration.
#[derive(Debug, Clone)]
#[must_use]
pub struct EncodeConfig {
    /// Video codec to use.
    pub codec: Codec,
    /// Video dimensions.
    pub dimensions: Dimensions,
    /// Pixel format (chroma subsampling).
    pub pixel_format: PixelFormat,
    /// Bit depth per component.
    pub bit_depth: BitDepth,
    /// Rate control mode.
    pub rate_control_mode: RateControlMode,
    /// Target bitrate in bits per second.
    pub target_bitrate: u32,
    /// Maximum bitrate in bits per second.
    pub max_bitrate: u32,
    /// Quality level for CQP mode (QP value).
    pub quality_level: u32,
    /// Frame rate numerator.
    pub frame_rate_numerator: u32,
    /// Frame rate denominator.
    pub frame_rate_denominator: u32,
    /// GOP size (distance between IDR frames).
    pub gop_size: u32,
    /// Number of consecutive B-frames.
    pub b_frame_count: u32,
    /// Maximum number of reference frames.
    pub max_reference_frames: u32,
    /// VBV/HRD virtual buffer size in milliseconds.
    /// Controls how much the encoder can deviate from the target bitrate
    /// on a per-frame basis. Smaller values produce more uniform frame
    /// sizes.
    pub virtual_buffer_size_ms: u32,
    /// Initial VBV buffer fullness in milliseconds.
    /// Controls how much budget the encoder has for IDR/I-frames.
    /// Setting this to 0 constrains IDR frames to the same budget as
    /// P-frames. Setting it equal to `virtual_buffer_size_ms` gives
    /// IDR frames maximum headroom.
    pub initial_virtual_buffer_size_ms: u32,
    /// Color description for VUI signaling.
    /// Defaults to BT.709 (full-range) when `None`.
    pub color_description: Option<ColorDescription>,
    /// When true, configure the encoder to take **RGB** input images directly
    /// and have the hardware perform RGB→YUV conversion inline during encode.
    /// Requires the device to advertise `VK_VALVE_video_encode_rgb_conversion`
    /// (currently only AMD's RADV driver). Trying to enable this on a device
    /// that doesn't support it returns an error from `Encoder::new`.
    ///
    /// When false (the default), the encoder takes its native YUV input
    /// format (NV12 / P010 / 4:4:4 variants) and the caller is responsible
    /// for converting RGB sources beforehand.
    pub use_rgb_input: bool,
}

impl EncodeConfig {
    /// Create a new H.264 encode configuration with default settings.
    pub fn h264(width: u32, height: u32) -> Self {
        assert!(width > 0, "width must be non-zero");
        assert!(height > 0, "height must be non-zero");

        Self {
            codec: Codec::H264,
            dimensions: Dimensions { width, height },
            pixel_format: PixelFormat::Yuv420,
            bit_depth: BitDepth::Eight,
            rate_control_mode: RateControlMode::Disabled,
            target_bitrate: DEFAULT_TARGET_BITRATE,
            max_bitrate: DEFAULT_MAX_BITRATE,
            quality_level: DEFAULT_H264_QP,
            frame_rate_numerator: DEFAULT_FRAME_RATE,
            frame_rate_denominator: 1,
            gop_size: DEFAULT_GOP_SIZE,
            b_frame_count: 0, // Start without B-frames for simplicity.
            max_reference_frames: DEFAULT_MAX_REFERENCE_FRAMES,
            virtual_buffer_size_ms: 1000,
            initial_virtual_buffer_size_ms: 1000,
            color_description: None,
            use_rgb_input: false,
        }
    }

    /// Create a new H.265/HEVC encode configuration with default settings.
    pub fn h265(width: u32, height: u32) -> Self {
        assert!(width > 0, "width must be non-zero");
        assert!(height > 0, "height must be non-zero");

        Self {
            codec: Codec::H265,
            dimensions: Dimensions { width, height },
            pixel_format: PixelFormat::Yuv420,
            bit_depth: BitDepth::Eight,
            rate_control_mode: RateControlMode::Disabled,
            target_bitrate: DEFAULT_TARGET_BITRATE,
            max_bitrate: DEFAULT_MAX_BITRATE,
            quality_level: DEFAULT_H265_QP,
            frame_rate_numerator: DEFAULT_FRAME_RATE,
            frame_rate_denominator: 1,
            gop_size: DEFAULT_GOP_SIZE,
            b_frame_count: 0, // Start without B-frames for simplicity.
            max_reference_frames: DEFAULT_MAX_REFERENCE_FRAMES,
            virtual_buffer_size_ms: 1000,
            initial_virtual_buffer_size_ms: 1000,
            color_description: None,
            use_rgb_input: false,
        }
    }

    /// Create a new AV1 encode configuration with default settings.
    pub fn av1(width: u32, height: u32) -> Self {
        assert!(width > 0, "width must be non-zero");
        assert!(height > 0, "height must be non-zero");

        Self {
            codec: Codec::AV1,
            dimensions: Dimensions { width, height },
            pixel_format: PixelFormat::Yuv420,
            bit_depth: BitDepth::Eight,
            rate_control_mode: RateControlMode::Disabled,
            target_bitrate: DEFAULT_TARGET_BITRATE,
            max_bitrate: DEFAULT_MAX_BITRATE,
            quality_level: 128, // AV1 uses 0-255 QP range
            frame_rate_numerator: DEFAULT_FRAME_RATE,
            frame_rate_denominator: 1,
            gop_size: DEFAULT_GOP_SIZE,
            b_frame_count: 0, // Start without B-frames for simplicity.
            max_reference_frames: DEFAULT_MAX_REFERENCE_FRAMES,
            virtual_buffer_size_ms: 1000,
            initial_virtual_buffer_size_ms: 1000,
            color_description: None,
            use_rgb_input: false,
        }
    }

    /// Set the rate control mode.
    pub fn with_rate_control(mut self, mode: RateControlMode) -> Self {
        self.rate_control_mode = mode;
        self
    }

    /// Set the pixel format (chroma subsampling).
    pub fn with_pixel_format(mut self, format: PixelFormat) -> Self {
        self.pixel_format = format;
        self
    }

    /// Set the bit depth (8 or 10 bit).
    pub fn with_bit_depth(mut self, depth: BitDepth) -> Self {
        self.bit_depth = depth;
        self
    }

    /// Set the quality level (QP for CQP mode).
    pub fn with_quality_level(mut self, level: u32) -> Self {
        self.quality_level = level;
        self
    }

    /// Set the frame rate.
    pub fn with_frame_rate(mut self, numerator: u32, denominator: u32) -> Self {
        self.frame_rate_numerator = numerator;
        self.frame_rate_denominator = denominator;
        self
    }

    /// Set the GOP size.
    pub fn with_gop_size(mut self, size: u32) -> Self {
        self.gop_size = size;
        self
    }

    /// Set the number of B-frames.
    pub fn with_b_frames(mut self, count: u32) -> Self {
        self.b_frame_count = count;
        self
    }

    /// Set the maximum reference frames.
    pub fn with_max_reference_frames(mut self, count: u32) -> Self {
        self.max_reference_frames = count;
        self
    }

    /// Set the target bitrate.
    pub fn with_target_bitrate(mut self, bitrate: u32) -> Self {
        self.target_bitrate = bitrate;
        self
    }

    /// Set the maximum bitrate.
    pub fn with_max_bitrate(mut self, bitrate: u32) -> Self {
        self.max_bitrate = bitrate;
        self
    }

    /// Set the VBV/HRD virtual buffer size in milliseconds.
    /// Smaller values produce more uniform frame sizes at the cost of
    /// quality variation during scene changes.
    pub fn with_virtual_buffer_size_ms(mut self, ms: u32) -> Self {
        self.virtual_buffer_size_ms = ms;
        self
    }

    /// Set the initial VBV buffer fullness in milliseconds.
    /// Use 0 to tightly constrain IDR/I-frame sizes.
    pub fn with_initial_virtual_buffer_size_ms(mut self, ms: u32) -> Self {
        self.initial_virtual_buffer_size_ms = ms;
        self
    }

    /// Set the color description for VUI signaling.
    pub fn with_color_description(mut self, desc: ColorDescription) -> Self {
        self.color_description = Some(desc);
        self
    }

    /// Enable hardware-direct RGB input (`VK_VALVE_video_encode_rgb_conversion`).
    /// See [`EncodeConfig::use_rgb_input`].
    pub fn with_rgb_input(mut self, enable: bool) -> Self {
        self.use_rgb_input = enable;
        self
    }
}

/// Encoded video packet.
#[derive(Debug, Clone)]
pub struct EncodedPacket {
    /// Encoded bitstream data.
    pub data: Vec<u8>,
    /// Frame type.
    pub frame_type: FrameType,
    /// Whether this is a keyframe.
    pub is_key_frame: bool,
    /// Presentation timestamp.
    pub pts: u64,
    /// Decode timestamp.
    pub dts: u64,
}

/// Video encoder supporting multiple codecs.
///
/// The encoder is implemented as an enum to dispatch to codec-specific implementations.
// Allow large_enum_variant: H265Encoder is currently a stub. When fully implemented,
// it will be similar in size to H264Encoder, making the size difference negligible.
#[allow(clippy::large_enum_variant)]
pub enum Encoder {
    /// H.264/AVC encoder.
    H264(self::h264::H264Encoder),
    /// H.265/HEVC encoder.
    H265(self::h265::H265Encoder),
    /// AV1 encoder.
    AV1(self::av1::AV1Encoder),
}

impl Encoder {
    /// Get the internal input image.
    ///
    /// This image can be used as a target for `ColorConverter::convert` to avoid
    /// an intermediate copy.
    pub fn input_image(&self) -> vk::Image {
        match self {
            Encoder::H264(encoder) => encoder.input_image(),
            Encoder::H265(encoder) => encoder.input_image(),
            Encoder::AV1(encoder) => encoder.input_image(),
        }
    }

    /// Create a new encoder.
    pub fn new(context: VideoContext, config: EncodeConfig) -> Result<Self> {
        match config.codec {
            Codec::H264 => Ok(Encoder::H264(self::h264::H264Encoder::new(
                context, config,
            )?)),
            Codec::H265 => Ok(Encoder::H265(self::h265::H265Encoder::new(
                context, config,
            )?)),
            Codec::AV1 => Ok(Encoder::AV1(self::av1::AV1Encoder::new(context, config)?)),
        }
    }

    /// Encode a frame from a GPU image.
    ///
    /// This accepts a source NV12 (YUV420) or planar YUV444 image on the GPU and encodes it directly.
    /// The source image must match the format and dimensions in the encoder configuration.
    ///
    /// Use `InputImage` to create an image from YUV data:
    /// ```no_run
    /// use pixelforge::{InputImage, Encoder, EncodeConfig, EncodeBitDepth, PixelFormat, VideoContext, Codec};
    ///
    /// # fn example(context: VideoContext) -> Result<(), Box<dyn std::error::Error>> {
    /// let config = EncodeConfig::h264(1920, 1080);
    /// let mut encoder = Encoder::new(context.clone(), config)?;
    /// let mut input = InputImage::new(
    ///     context,
    ///     Codec::H264,
    ///     1920,
    ///     1080,
    ///     EncodeBitDepth::Eight,
    ///     PixelFormat::Yuv420,
    /// )?;
    ///
    /// // Upload YUV420 data to the input image
    /// # let yuv_data = vec![0u8; 1920 * 1080 * 3 / 2];
    /// input.upload_yuv420(&yuv_data)?;
    ///
    /// // Encode the image (no GPU wait semaphore needed when uploaded synchronously).
    /// let packets = encoder.encode(input.image())?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn encode(&mut self, src_image: vk::Image) -> Result<Vec<EncodedPacket>> {
        match self {
            Encoder::H264(encoder) => encoder.encode(src_image),
            Encoder::H265(encoder) => encoder.encode(src_image),
            Encoder::AV1(encoder) => encoder.encode(src_image),
        }
    }

    /// Flush the encoder and get remaining packets.
    pub fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
        match self {
            Encoder::H264(encoder) => encoder.flush(),
            Encoder::H265(encoder) => encoder.flush(),
            Encoder::AV1(encoder) => encoder.flush(),
        }
    }

    /// Request that the next frame be an IDR frame.
    pub fn request_idr(&mut self) {
        match self {
            Encoder::H264(encoder) => encoder.request_idr(),
            Encoder::H265(encoder) => encoder.request_idr(),
            Encoder::AV1(encoder) => encoder.request_idr(),
        }
    }

    /// Update the color description (VUI parameters) for the encoder.
    ///
    /// This recreates the video session parameters with an updated SPS/VPS/sequence
    /// header containing the new color description. The next frame will be encoded as
    /// an IDR/key frame with the new parameters.
    pub fn set_color_description(&mut self, desc: ColorDescription) -> Result<()> {
        match self {
            Encoder::H264(encoder) => encoder.set_color_description(desc),
            Encoder::H265(encoder) => encoder.set_color_description(desc),
            Encoder::AV1(encoder) => encoder.set_color_description(desc),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // PixelFormat tests.
    mod pixel_format_tests {
        use super::*;

        #[test]
        fn test_yuv420_frame_size() {
            // YUV420: Y = width * height, U = Y/4, V = Y/4 -> total = Y * 1.5
            let size = PixelFormat::Yuv420.frame_size(1920, 1080);
            let expected = (1920 * 1080) * 3 / 2; // 3110400
            assert_eq!(size, expected);
        }

        #[test]
        fn test_yuv422_frame_size() {
            // YUV422: Y = width * height, U = Y/2, V = Y/2 -> total = Y * 2
            let size = PixelFormat::Yuv422.frame_size(1920, 1080);
            let expected = (1920 * 1080) * 2; // 4147200
            assert_eq!(size, expected);
        }

        #[test]
        fn test_yuv444_frame_size() {
            // YUV444: Y = width * height, U = Y, V = Y -> total = Y * 3
            let size = PixelFormat::Yuv444.frame_size(1920, 1080);
            let expected = (1920 * 1080) * 3; // 6220800
            assert_eq!(size, expected);
        }

        #[test]
        fn test_small_resolution() {
            // Test with small resolution.
            let size = PixelFormat::Yuv420.frame_size(320, 240);
            let expected = (320 * 240) * 3 / 2; // 115200
            assert_eq!(size, expected);
        }

        #[test]
        fn test_4k_resolution() {
            // Test with 4K resolution.
            let size = PixelFormat::Yuv420.frame_size(3840, 2160);
            let expected = (3840 * 2160) * 3 / 2; // 12441600
            assert_eq!(size, expected);
        }

        #[test]
        fn test_default() {
            assert_eq!(PixelFormat::default(), PixelFormat::Yuv420);
        }

        #[test]
        fn test_vk_chroma_subsampling_conversion() {
            let vk_420: vk::VideoChromaSubsamplingFlagsKHR = PixelFormat::Yuv420.into();
            assert_eq!(vk_420, vk::VideoChromaSubsamplingFlagsKHR::TYPE_420);

            let vk_422: vk::VideoChromaSubsamplingFlagsKHR = PixelFormat::Yuv422.into();
            assert_eq!(vk_422, vk::VideoChromaSubsamplingFlagsKHR::TYPE_422);

            let vk_444: vk::VideoChromaSubsamplingFlagsKHR = PixelFormat::Yuv444.into();
            assert_eq!(vk_444, vk::VideoChromaSubsamplingFlagsKHR::TYPE_444);
        }
    }

    // BitDepth tests.
    mod bit_depth_tests {
        use super::*;

        #[test]
        fn test_default() {
            assert_eq!(BitDepth::default(), BitDepth::Eight);
        }

        #[test]
        fn test_vk_bit_depth_conversion() {
            let vk_8: vk::VideoComponentBitDepthFlagsKHR = BitDepth::Eight.into();
            assert_eq!(vk_8, vk::VideoComponentBitDepthFlagsKHR::TYPE_8);

            let vk_10: vk::VideoComponentBitDepthFlagsKHR = BitDepth::Ten.into();
            assert_eq!(vk_10, vk::VideoComponentBitDepthFlagsKHR::TYPE_10);
        }
    }

    // RateControlMode tests.
    mod rate_control_tests {
        use super::*;

        #[test]
        fn test_default() {
            assert_eq!(RateControlMode::default(), RateControlMode::Disabled);
        }
    }

    // EncodeConfig tests.
    mod encode_config_tests {
        use super::*;

        #[test]
        fn test_h264_defaults() {
            let config = EncodeConfig::h264(1920, 1080);

            assert_eq!(config.codec, Codec::H264);
            assert_eq!(config.dimensions.width, 1920);
            assert_eq!(config.dimensions.height, 1080);
            assert_eq!(config.pixel_format, PixelFormat::Yuv420);
            assert_eq!(config.bit_depth, BitDepth::Eight);
            assert_eq!(config.rate_control_mode, RateControlMode::Disabled);
            assert_eq!(config.quality_level, 26);
            assert_eq!(config.gop_size, 30);
            assert_eq!(config.b_frame_count, 0);
            assert_eq!(config.frame_rate_numerator, 30);
            assert_eq!(config.frame_rate_denominator, 1);
        }

        #[test]
        fn test_h265_defaults() {
            let config = EncodeConfig::h265(3840, 2160);

            assert_eq!(config.codec, Codec::H265);
            assert_eq!(config.dimensions.width, 3840);
            assert_eq!(config.dimensions.height, 2160);
            assert_eq!(config.quality_level, 28); // H.265 uses slightly higher QP
        }

        #[test]
        fn test_with_rate_control() {
            let config = EncodeConfig::h264(1920, 1080).with_rate_control(RateControlMode::Cbr);

            assert_eq!(config.rate_control_mode, RateControlMode::Cbr);
        }

        #[test]
        fn test_with_pixel_format() {
            let config = EncodeConfig::h264(1920, 1080).with_pixel_format(PixelFormat::Yuv444);

            assert_eq!(config.pixel_format, PixelFormat::Yuv444);
        }

        #[test]
        fn test_with_bit_depth() {
            let config = EncodeConfig::h265(1920, 1080).with_bit_depth(BitDepth::Ten);

            assert_eq!(config.bit_depth, BitDepth::Ten);
        }

        #[test]
        fn test_with_quality_level() {
            let config = EncodeConfig::h264(1920, 1080).with_quality_level(20);

            assert_eq!(config.quality_level, 20);
        }

        #[test]
        fn test_with_frame_rate() {
            let config = EncodeConfig::h264(1920, 1080).with_frame_rate(60, 1);

            assert_eq!(config.frame_rate_numerator, 60);
            assert_eq!(config.frame_rate_denominator, 1);
        }

        #[test]
        fn test_with_gop_size() {
            let config = EncodeConfig::h264(1920, 1080).with_gop_size(60);

            assert_eq!(config.gop_size, 60);
        }

        #[test]
        fn test_with_b_frames() {
            let config = EncodeConfig::h264(1920, 1080).with_b_frames(2);

            assert_eq!(config.b_frame_count, 2);
        }

        #[test]
        fn test_with_max_reference_frames() {
            let config = EncodeConfig::h264(1920, 1080).with_max_reference_frames(8);

            assert_eq!(config.max_reference_frames, 8);
        }

        #[test]
        fn test_with_target_bitrate() {
            let config = EncodeConfig::h264(1920, 1080).with_target_bitrate(8_000_000);

            assert_eq!(config.target_bitrate, 8_000_000);
        }

        #[test]
        fn test_with_max_bitrate() {
            let config = EncodeConfig::h264(1920, 1080).with_max_bitrate(12_000_000);

            assert_eq!(config.max_bitrate, 12_000_000);
        }

        #[test]
        fn test_av1_defaults() {
            let config = EncodeConfig::av1(2560, 1440);

            assert_eq!(config.codec, Codec::AV1);
            assert_eq!(config.dimensions.width, 2560);
            assert_eq!(config.dimensions.height, 1440);
            assert_eq!(config.pixel_format, PixelFormat::Yuv420);
            assert_eq!(config.bit_depth, BitDepth::Eight);
            assert_eq!(config.rate_control_mode, RateControlMode::Disabled);
            assert_eq!(config.quality_level, 128); // AV1 uses 0-255 QP range
            assert_eq!(config.gop_size, 30);
            assert_eq!(config.b_frame_count, 0);
            assert_eq!(config.frame_rate_numerator, 30);
            assert_eq!(config.frame_rate_denominator, 1);
        }

        #[test]
        fn test_av1_builder_chaining() {
            let config = EncodeConfig::av1(1920, 1080)
                .with_rate_control(RateControlMode::Vbr)
                .with_target_bitrate(8_000_000)
                .with_max_bitrate(12_000_000)
                .with_gop_size(60)
                .with_frame_rate(60, 1)
                .with_quality_level(100)
                .with_max_reference_frames(2);

            assert_eq!(config.codec, Codec::AV1);
            assert_eq!(config.rate_control_mode, RateControlMode::Vbr);
            assert_eq!(config.target_bitrate, 8_000_000);
            assert_eq!(config.max_bitrate, 12_000_000);
            assert_eq!(config.gop_size, 60);
            assert_eq!(config.frame_rate_numerator, 60);
            assert_eq!(config.quality_level, 100);
            assert_eq!(config.max_reference_frames, 2);
        }

        #[test]
        fn test_builder_chaining() {
            let config = EncodeConfig::h264(1920, 1080)
                .with_rate_control(RateControlMode::Vbr)
                .with_target_bitrate(6_000_000)
                .with_max_bitrate(10_000_000)
                .with_gop_size(120)
                .with_b_frames(2)
                .with_frame_rate(60, 1)
                .with_pixel_format(PixelFormat::Yuv420)
                .with_bit_depth(BitDepth::Eight);

            assert_eq!(config.rate_control_mode, RateControlMode::Vbr);
            assert_eq!(config.target_bitrate, 6_000_000);
            assert_eq!(config.max_bitrate, 10_000_000);
            assert_eq!(config.gop_size, 120);
            assert_eq!(config.b_frame_count, 2);
            assert_eq!(config.frame_rate_numerator, 60);
        }
    }

    // FrameType tests.
    mod frame_type_tests {
        use super::*;

        #[test]
        fn test_frame_types() {
            // Just test the enum variants exist and are distinct.
            assert_ne!(FrameType::Idr, FrameType::I);
            assert_ne!(FrameType::I, FrameType::P);
            assert_ne!(FrameType::P, FrameType::B);
            assert_ne!(FrameType::B, FrameType::Unknown);
        }
    }

    // EncodedPacket tests.
    mod encoded_packet_tests {
        use super::*;

        #[test]
        fn test_packet_creation() {
            let packet = EncodedPacket {
                data: vec![0x00, 0x00, 0x00, 0x01, 0x67],
                frame_type: FrameType::Idr,
                is_key_frame: true,
                pts: 0,
                dts: 0,
            };

            assert!(packet.is_key_frame);
            assert_eq!(packet.frame_type, FrameType::Idr);
            assert_eq!(packet.data.len(), 5);
        }
    }

    // Codec tests.
    mod codec_tests {
        use super::*;

        #[test]
        fn test_codec_variants() {
            assert_ne!(Codec::H264, Codec::H265);
            assert_ne!(Codec::H265, Codec::AV1);
        }
    }

    // ColorDescription tests.
    mod color_description_tests {
        use super::*;

        #[test]
        fn test_bt709() {
            let cd = ColorDescription::bt709();
            assert_eq!(cd.color_primaries, 1);
            assert_eq!(cd.transfer_characteristics, 1);
            assert_eq!(cd.matrix_coefficients, 1);
            assert!(cd.full_range);
        }

        #[test]
        fn test_bt2020_pq() {
            let cd = ColorDescription::bt2020_pq();
            assert_eq!(cd.color_primaries, 9);
            assert_eq!(cd.transfer_characteristics, 16);
            assert_eq!(cd.matrix_coefficients, 9);
            assert!(!cd.full_range);
        }
    }
}
