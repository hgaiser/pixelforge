//! Encoder types, configuration, and shared utilities.
//!
//! This module provides:
//! - Core encoder types and configuration (`EncodeConfig`, `EncodedPacket`, etc.)
//! - GOP structure management (`gop` module) - reusable for H.264/H.265.
//! - Frame reordering for B-frame support (`reorder` module) - reusable for H.264/H.265.

pub mod av1;
pub mod bitwriter;
pub(crate) mod codec;
pub mod dpb;
pub mod gop;
pub mod h264;
pub mod h265;
pub(crate) mod pipeline;
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

/// Encode usage hints.
/// Allows encoder to potentially make smarter choices with appropriate usage hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncodeUsageHint {
    /// Default usage - no specific usage hint given to encoder.
    #[default]
    Default,
    /// Transcoding usage - hint that encoding will be done in conjunction with decoding.
    Transcoding,
    /// Streaming usage - hint that the output will be sent over network.
    Streaming,
    /// Recording usage - hint that the output will be used for offline consumption.
    Recording,
    /// Conferencing usage - hint that the output will be used for video conferencing.
    Conferencing,
}

impl From<EncodeUsageHint> for vk::VideoEncodeUsageFlagsKHR {
    fn from(hint: EncodeUsageHint) -> Self {
        match hint {
            EncodeUsageHint::Default => vk::VideoEncodeUsageFlagsKHR::DEFAULT,
            EncodeUsageHint::Transcoding => vk::VideoEncodeUsageFlagsKHR::TRANSCODING,
            EncodeUsageHint::Streaming => vk::VideoEncodeUsageFlagsKHR::STREAMING,
            EncodeUsageHint::Recording => vk::VideoEncodeUsageFlagsKHR::RECORDING,
            EncodeUsageHint::Conferencing => vk::VideoEncodeUsageFlagsKHR::CONFERENCING,
        }
    }
}

/// Encode content hints.
/// Allows encoder to potentially make smarter choices with appropriate content hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncodeContentHint {
    /// Default content - no specific content hint given to encoder.
    #[default]
    Default,
    /// Camera content - hint that the content is from a camera.
    Camera,
    /// Desktop content - hint that the content is from desktop.
    Desktop,
    /// Rendered content - hint that the content is rendered (i.e. game).
    Rendered,
}

impl From<EncodeContentHint> for vk::VideoEncodeContentFlagsKHR {
    fn from(hint: EncodeContentHint) -> Self {
        match hint {
            EncodeContentHint::Default => vk::VideoEncodeContentFlagsKHR::DEFAULT,
            EncodeContentHint::Camera => vk::VideoEncodeContentFlagsKHR::CAMERA,
            EncodeContentHint::Desktop => vk::VideoEncodeContentFlagsKHR::DESKTOP,
            EncodeContentHint::Rendered => vk::VideoEncodeContentFlagsKHR::RENDERED,
        }
    }
}

/// Encoder tuning modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncoderTuningMode {
    /// Default mode - encoder specific default tuning.
    #[default]
    Default,
    /// High-quality mode - focus on quality over encoding speed.
    HighQuality,
    /// Low-latency mode - focus on encoding speed over quality.
    LowLatency,
    /// Ultra-low-latency mode - focus on highest encoding speed with a hit to quality.
    UltraLowLatency,
    /// Lossless mode - tune encoder for lossless output.
    Lossless,
}

impl From<EncoderTuningMode> for vk::VideoEncodeTuningModeKHR {
    fn from(mode: EncoderTuningMode) -> Self {
        match mode {
            EncoderTuningMode::Default => vk::VideoEncodeTuningModeKHR::DEFAULT,
            EncoderTuningMode::HighQuality => vk::VideoEncodeTuningModeKHR::HIGH_QUALITY,
            EncoderTuningMode::LowLatency => vk::VideoEncodeTuningModeKHR::LOW_LATENCY,
            EncoderTuningMode::UltraLowLatency => vk::VideoEncodeTuningModeKHR::ULTRA_LOW_LATENCY,
            EncoderTuningMode::Lossless => vk::VideoEncodeTuningModeKHR::LOSSLESS,
        }
    }
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
#[must_use]
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
    // H.273 code points for the fields above.
    const PRIMARIES_BT709: u8 = 1;
    const PRIMARIES_BT2020: u8 = 9;
    const TRANSFER_BT709: u8 = 1;
    const TRANSFER_ST2084_PQ: u8 = 16;
    const MATRIX_BT709: u8 = 1;
    const MATRIX_BT2020_NCL: u8 = 9;

    /// BT.709 color description (standard SDR, limited range).
    pub fn bt709() -> Self {
        Self {
            color_primaries: Self::PRIMARIES_BT709,
            transfer_characteristics: Self::TRANSFER_BT709,
            matrix_coefficients: Self::MATRIX_BT709,
            full_range: false,
        }
    }

    /// BT.2020 with PQ transfer function (HDR10).
    pub fn bt2020_pq() -> Self {
        Self {
            color_primaries: Self::PRIMARIES_BT2020,
            transfer_characteristics: Self::TRANSFER_ST2084_PQ,
            matrix_coefficients: Self::MATRIX_BT2020_NCL,
            full_range: false,
        }
    }

    /// Set full range (0-255) rather than limited/TV range (16-235).
    pub fn with_full_range(mut self, full_range: bool) -> Self {
        self.full_range = full_range;
        self
    }

    /// Whether this description makes the stream HDR.
    ///
    /// The PQ (ST 2084) transfer function is what decides it; primaries and
    /// luma range do not.
    pub fn is_hdr(&self) -> bool {
        self.transfer_characteristics == Self::TRANSFER_ST2084_PQ
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
    /// Maximum QP for I/IDR frames in rate-controlled modes.
    /// `None` uses the codec default (42 for H.264, 51 for H.265).
    /// Capping it below the default forces keyframes to spend enough bits
    /// even before rate control has adapted to the content, at the cost of
    /// larger keyframes. Values below the codec's minimum QP bound are
    /// raised to it. Ignored by AV1 and by CQP/disabled rate control.
    pub max_qp_i: Option<u32>,
    /// Vulkan video encode quality level
    /// (`VkVideoEncodeQualityLevelInfoKHR`). Higher levels trade encoding
    /// speed for quality. Must be below the device's `maxQualityLevels`
    /// for the video profile.
    pub encode_quality_level: u32,
    /// Color description for VUI signaling.
    /// Defaults to BT.709 (full-range) when `None`.
    pub color_description: Option<ColorDescription>,
    /// Usage hint for encoding.
    pub encode_usage_hint: EncodeUsageHint,
    /// Content hint for encoding.
    pub encode_content_hint: EncodeContentHint,
    /// Encoder tuning mode.
    pub encoder_tuning_mode: EncoderTuningMode,
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
            max_qp_i: None,
            encode_quality_level: 0,
            color_description: None,
            encode_usage_hint: EncodeUsageHint::Default,
            encode_content_hint: EncodeContentHint::Default,
            encoder_tuning_mode: EncoderTuningMode::Default,
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
            max_qp_i: None,
            encode_quality_level: 0,
            color_description: None,
            encode_usage_hint: EncodeUsageHint::Default,
            encode_content_hint: EncodeContentHint::Default,
            encoder_tuning_mode: EncoderTuningMode::Default,
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
            max_qp_i: None,
            encode_quality_level: 0,
            color_description: None,
            encode_usage_hint: EncodeUsageHint::Default,
            encode_content_hint: EncodeContentHint::Default,
            encoder_tuning_mode: EncoderTuningMode::Default,
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

    /// Set the maximum QP for I/IDR frames in rate-controlled modes.
    pub fn with_max_qp_i(mut self, qp: u32) -> Self {
        self.max_qp_i = Some(qp);
        self
    }

    /// Set the Vulkan video encode quality level.
    pub fn with_encode_quality_level(mut self, level: u32) -> Self {
        self.encode_quality_level = level;
        self
    }

    /// Set the color description for VUI signaling.
    pub fn with_color_description(mut self, desc: ColorDescription) -> Self {
        self.color_description = Some(desc);
        self
    }

    /// Set the usage hint for encoding.
    pub fn with_encode_usage_hint(mut self, hint: EncodeUsageHint) -> Self {
        self.encode_usage_hint = hint;
        self
    }

    /// Set the content hint for encoding.
    pub fn with_encode_content_hint(mut self, hint: EncodeContentHint) -> Self {
        self.encode_content_hint = hint;
        self
    }

    /// Set the encoder tuning mode.
    pub fn with_encoder_tuning_mode(mut self, mode: EncoderTuningMode) -> Self {
        self.encoder_tuning_mode = mode;
        self
    }
}

pub use pipeline::EncodeFuture;

/// Statistic about the encoded video packet.
#[derive(Debug, Clone)]
pub struct EncodedPacketStats {
    /// GPU encode time in nanoseconds
    pub gpu_time_ns: u64,
    /// CPU wall time in nanoseconds (submission + fence wait + readback)
    pub frame_latency_ns: u64,
    /// Wall latency in nanoseconds (time between submit and bitstream ready)
    pub wall_latency_ns: u64,
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
    /// Optional stats about the packet
    pub stats: Option<EncodedPacketStats>,
}

/// The codec-erased operations every [`codec::CodecEncoder`] exposes.
///
/// One blanket impl covers all codecs, so [`Encoder`] can hold any of them
/// behind a single boxed pointer instead of an enum that re-dispatches by hand.
trait EncoderApi: Send {
    fn input_image(&self) -> vk::Image;
    fn encode(&mut self, src_image: vk::Image) -> Result<EncodeFuture>;
    fn flush(&mut self) -> Result<()>;
    fn request_idr(&mut self);
    fn invalidate_reference_frames(&mut self, first_lost_display_order: u64);
    fn set_color_description(&mut self, desc: ColorDescription) -> Result<()>;
}

impl<C: codec::VideoCodec> EncoderApi for codec::CodecEncoder<C> {
    fn input_image(&self) -> vk::Image {
        codec::CodecEncoder::input_image(self)
    }
    fn encode(&mut self, src_image: vk::Image) -> Result<EncodeFuture> {
        codec::CodecEncoder::encode(self, src_image)
    }
    fn flush(&mut self) -> Result<()> {
        codec::CodecEncoder::flush(self)
    }
    fn request_idr(&mut self) {
        codec::CodecEncoder::request_idr(self)
    }
    fn invalidate_reference_frames(&mut self, first_lost_display_order: u64) {
        codec::CodecEncoder::invalidate_reference_frames(self, first_lost_display_order)
    }
    fn set_color_description(&mut self, desc: ColorDescription) -> Result<()> {
        codec::CodecEncoder::set_color_description(self, desc)
    }
}

/// Video encoder supporting multiple codecs.
///
/// Constructed via [`Encoder::new`], which selects the codec from the config and
/// boxes the corresponding `codec::CodecEncoder`. All codecs share one generic
/// implementation; this type just erases which one is in use.
pub struct Encoder(Box<dyn EncoderApi>);

impl Encoder {
    /// Create a new encoder for the codec named in `config`.
    pub fn new(context: VideoContext, config: EncodeConfig) -> Result<Self> {
        let inner: Box<dyn EncoderApi> = match config.codec {
            Codec::H264 => Box::new(self::h264::H264::create(context, config)?),
            Codec::H265 => Box::new(self::h265::H265::create(context, config)?),
            Codec::AV1 => Box::new(self::av1::Av1::create(context, config)?),
        };
        Ok(Encoder(inner))
    }

    /// Get the internal input image.
    ///
    /// This image can be used as a target for `ColorConverter::convert` to avoid
    /// an intermediate copy.
    pub fn input_image(&self) -> vk::Image {
        self.0.input_image()
    }

    /// Encode a frame from a GPU image.
    ///
    /// This accepts a source NV12 (YUV420) or planar YUV444 image on the GPU and encodes it directly.
    /// The source image must match the format and dimensions in the encoder configuration.
    ///
    /// Encoding is asynchronous: this submits the frame without blocking and
    /// returns an [`EncodeFuture`] that resolves with the encoded packet once the
    /// GPU finishes and a background readback thread reads it back. The call only
    /// blocks if every pipeline slot is still in flight (backpressure).
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
    /// // Submit the frame and await its packet.
    /// let future = encoder.encode(input.image())?;
    /// let packet = pollster::block_on(future)?;
    /// // ... use packet.data ...
    /// # Ok(())
    /// # }
    /// ```
    pub fn encode(&mut self, src_image: vk::Image) -> Result<EncodeFuture> {
        self.0.encode(src_image)
    }

    /// Wait for all in-flight frames to finish encoding (end-of-stream barrier).
    ///
    /// Packets are delivered through the [`EncodeFuture`]s returned by
    /// [`Encoder::encode`]; await those to obtain them. Once this returns, every
    /// outstanding future has been resolved.
    pub fn flush(&mut self) -> Result<()> {
        self.0.flush()
    }

    /// Request that the next frame be an IDR frame.
    pub fn request_idr(&mut self) {
        self.0.request_idr()
    }

    /// Recover from packet loss without a full IDR (reference frame invalidation).
    ///
    /// When a client reports that it could not decode a range of frames, every
    /// reference picture the encoder still holds from the earliest lost frame
    /// onward is transitively undecodable on the client. This drops those
    /// references so the next P-frame is predicted from the most recent
    /// *surviving* reference instead — a much cheaper recovery than re-sending a
    /// full keyframe. If no reference survives (the loss covers the encoder's
    /// whole reference window), it transparently falls back to forcing an IDR.
    ///
    /// `first_lost_display_order` is the display order — the `pts` reported on
    /// [`EncodedPacket`]s — of the earliest frame the client lost.
    ///
    /// Effective recovery requires the encoder to keep more than one reference
    /// (see [`EncodeConfig::with_max_reference_frames`]); with a single
    /// reference this necessarily falls back to an IDR. This applies uniformly
    /// across H.264, H.265, and AV1 — each keeps a multi-reference window and
    /// re-anchors prediction to the most recent survivor, falling back to an
    /// IDR (AV1: key frame) only when the loss covers the entire window.
    pub fn invalidate_reference_frames(&mut self, first_lost_display_order: u64) {
        self.0.invalidate_reference_frames(first_lost_display_order)
    }

    /// Update the color description (VUI parameters) for the encoder.
    ///
    /// This recreates the video session parameters with an updated SPS/VPS/sequence
    /// header containing the new color description. The next frame will be encoded as
    /// an IDR/key frame with the new parameters.
    pub fn set_color_description(&mut self, desc: ColorDescription) -> Result<()> {
        self.0.set_color_description(desc)
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
                stats: None,
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
            assert!(!cd.full_range);
        }

        #[test]
        fn test_bt2020_pq() {
            let cd = ColorDescription::bt2020_pq();
            assert_eq!(cd.color_primaries, 9);
            assert_eq!(cd.transfer_characteristics, 16);
            assert_eq!(cd.matrix_coefficients, 9);
            assert!(!cd.full_range);
        }

        #[test]
        fn test_with_full_range() {
            let cd = ColorDescription::bt709().with_full_range(true);
            assert!(cd.full_range);
            // Only the range changes; the preset stays intact.
            assert_eq!(cd.with_full_range(false), ColorDescription::bt709());
        }

        #[test]
        fn test_is_hdr() {
            // The luma range doesn't decide it.
            for full_range in [false, true] {
                assert!(
                    ColorDescription::bt2020_pq()
                        .with_full_range(full_range)
                        .is_hdr()
                );
                assert!(
                    !ColorDescription::bt709()
                        .with_full_range(full_range)
                        .is_hdr()
                );
            }

            // Neither do the primaries: PQ on BT.709 primaries is still HDR.
            // 16 is the H.273 code point for ST 2084, written literally so the
            // test pins the constant rather than echoing it.
            let pq_709 = ColorDescription {
                transfer_characteristics: 16,
                ..ColorDescription::bt709()
            };
            assert!(pq_709.is_hdr());
        }
    }
}
