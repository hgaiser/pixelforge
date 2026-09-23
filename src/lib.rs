//! A Vulkan-based video encoding and decoding library for Rust, supporting H.264,
//! H.265 and AV1 encode, and H.264 decode.
//!
//! # Features
//!
//! - **Hardware-accelerated** video encoding and decoding using Vulkan Video extensions.
//! - **Multiple codec support**: H.264/AVC, H.265/HEVC, AV1 encode; H.264 decode.
//! - **Asynchronous pipelines**: both directions submit without waiting.
//!   Encoding hands back an [`EncodeFuture`]; decoding delivers frames through a
//!   [`DecodeSource`] as the GPU finishes with them.
//! - **GPU color conversion**: RGB/BGR → YUV via Vulkan compute shaders (BT.709, BT.2020, sRGB→BT.2020+PQ, scRGB-linear→BT.2020+PQ),
//!   or in the encoder itself where `VK_VALVE_video_encode_rgb_conversion` is available.
//! - **Shared devices**: encode and decode on a Vulkan device your application
//!   created, from Vulkan 1.1 up, on queues you choose, ordered against your own
//!   work with timeline semaphores.
//! - **HDR support**: 10-bit encoding (P010, YUV444P10), PQ transfer function, BT.2020 color space.
//! - **GPU-native API**: Encode directly from Vulkan images (`vk::Image`).
//! - **Flexible configuration**: Rate control (CBR, VBR, CQP), quality levels, GOP settings.
//! - **Multiple input formats**: BGRx, RGBx, BGRA, RGBA, ABGR2101010 (10-bit packed), RGBA16F (FP16).
//! - **Utility helpers**: [`InputImage`] for easy YUV data upload to GPU.
//! - **Optional DMA-BUF support**: Zero-copy image import from external processes (Linux only).
//!
//! > **Note**: B-frame support is not yet implemented. Setting `b_frame_count > 0` will panic.
//!
//! # Supported Codecs
//!
//! | Codec | Encode | Decode |
//! |-------|--------|--------|
//! | H.264/AVC | ✓ | ✓ |
//! | H.265/HEVC | ✓ | |
//! | AV1 | ✓ | |
//!
//! H.264 decoding is verified byte-identical to `ffmpeg -pix_fmt nv12` on AMD
//! (RADV), NVIDIA and Intel (ANV).
//!
//! # Requirements
//!
//! - A GPU with Vulkan video support (e.g., NVIDIA RTX series, AMD RDNA2+, Intel Arc).
//!   Decoding additionally needs a video decode queue; on Intel Arc under Mesa it
//!   currently has to be enabled with `ANV_DEBUG=video-decode,video-encode`.
//!
//! # Installation
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! pixelforge = "0.1"
//! ```
//!
//! ## Optional Features
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `dmabuf` | Enable DMA-BUF support for zero-copy image import from external processes (Linux only). Adds Vulkan extensions: `VK_KHR_external_memory`, `VK_KHR_external_memory_fd`, `VK_EXT_external_memory_dma_buf`, `VK_EXT_image_drm_format_modifier`. |
//!
//! To enable DMA-BUF support:
//!
//! ```toml
//! [dependencies]
//! pixelforge = { version = "0.1", features = ["dmabuf"] }
//! ```
//!
//! # Quick Start
//!
//! ## Query Capabilities
//!
//! ```rust,no_run
//! use pixelforge::{Codec, VideoContextBuilder};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let context = VideoContextBuilder::new()
//!         .app_name("My App")
//!         .build()?;
//!
//!     for codec in [Codec::H264, Codec::H265, Codec::AV1] {
//!         println!("{:?}: encode={}",
//!             codec,
//!             context.supports_encode(codec)
//!         );
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ## Encoding Video
//!
//! ```rust,no_run
//! use pixelforge::{
//!     Codec, EncodeBitDepth, EncodeConfig, Encoder, InputImage, PixelFormat, RateControlMode,
//!     VideoContextBuilder,
//! };
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let context = VideoContextBuilder::new()
//!         .app_name("Encoder Example")
//!         .require_encode(Codec::H264)
//!         .build()?;
//!
//!     let config = EncodeConfig::h264(1920, 1080)
//!         .with_rate_control(RateControlMode::Vbr)
//!         .with_target_bitrate(5_000_000)
//!         .with_frame_rate(30, 1)
//!         .with_gop_size(60);
//!
//!     // Create an InputImage helper for uploading YUV data to the GPU.
//!     let mut input_image = InputImage::new(
//!         context.clone(),
//!         Codec::H264,
//!         1920,
//!         1080,
//!         EncodeBitDepth::Eight,
//!         PixelFormat::Yuv420,
//!     )?;
//!     let mut encoder = Encoder::new(context, config)?;
//!
//!     // For each frame: upload YUV data and encode.
//!     // let yuv_data: &[u8] = ...;  // YUV420 frame data
//!     // input_image.upload_yuv420(yuv_data)?;
//!     // let packets = encoder.encode(input_image.image())?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Decoding Video
//!
//! The decoder is stream-driven: it creates its Vulkan session from the
//! stream's own parameter sets, so nothing has to be configured up front, and a
//! mid-stream resolution change is handled transparently.
//!
//! Bytes go in through a [`DecodeSink`], frames come out
//! of a [`DecodeSource`]. A [`Decoder`] holds both, so
//! one thread can drive the whole thing; [`Decoder::split`](decoder::Decoder::split)
//! separates them for a producer and a consumer on their own threads.
//!
//! Frames come out in presentation order and, where the device supports unified
//! image layouts, without ever being copied: the frame *is* the decoder's own
//! image. Drop each one when done, which returns its storage.
//!
//! ```rust,no_run
//! use pixelforge::{Codec, VideoContextBuilder};
//! use pixelforge::decoder::{DecodeConfig, Decoder, FramePoll};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let context = VideoContextBuilder::new()
//!         .app_name("Decoder Example")
//!         .require_decode(Codec::H264)
//!         .build()?;
//!
//!     // A file can cut anywhere, so let the decoder frame it. Input that
//!     // arrives already framed (RTP, a container) skips `with_byte_stream`.
//!     let config = DecodeConfig::h264().with_byte_stream();
//!     let mut decoder = Decoder::new(context, config)?;
//!     let stream: Vec<u8> = std::fs::read("input.264")?;
//!
//!     for (i, chunk) in stream.chunks(64 * 1024).enumerate() {
//!         // The status says what happened; an `Err` means something is
//!         // actually wrong. Joining a stream partway through is not.
//!         let _status = decoder.decode(chunk, i as u64)?;
//!         // Take what the GPU has finished with; `Pending` just means "not yet".
//!         while let FramePoll::Frame(frame) = decoder.try_next_frame()? {
//!             // `frame.image` is a decoder-owned GPU image, valid until dropped.
//!             let _ = frame.image;
//!         }
//!     }
//!
//!     // End of stream: decodes the trailing picture, emits what reordering
//!     // held back, and closes the source.
//!     decoder.finish()?;
//!     while let Some(frame) = pollster::block_on(decoder.next_frame())? {
//!         let _ = frame.image;
//!     }
//!     Ok(())
//! }
//! ```
//!
//! A live frame reserves a DPB slot, so
//! [`DecodeConfig::with_output_depth`](decoder::DecodeConfig::with_output_depth)
//! bounds how many can be outstanding before the decoder starts copying
//! pictures out instead of handing over its own. Reading a frame back to the
//! CPU is the consumer's job; `examples/common` shows one way.
//!
//! ## Color Conversion (RGB → YUV)
//!
//! PixelForge includes a GPU compute shader for converting RGB input to YUV
//! output. The source color describes the input. The target color describes the stream.
//! Both are a [`ColorSpec`].
//!
//! | `ColorSpec` | | Can be a target |
//! |-------------|-|-----------------|
//! | `Srgb` | Ordinary SDR content | yes |
//! | `Bt709Linear` | scRGB, from an `EXTENDED_SRGB_LINEAR_EXT` swapchain | no |
//! | `Bt2020Linear` | Linear light, wide gamut | no |
//! | `Bt2020Pq` | HDR10, from an `HDR10_ST2084_EXT` swapchain | yes |
//!
//! The linear spaces cannot be a target, because a video file has no way to
//! record that it holds linear light. Any source can be converted to
//! `Bt2020Pq`. Only `Srgb` can be converted to `Srgb`; the others would need
//! tone mapping or a gamma curve applied, which the shader does not do.
//! [`ColorConverter::new`] rejects the combinations it cannot do.
//!
//! Encoding to HDR needs to know how bright the source's white is, since HDR
//! carries real brightness values and SDR does not. Each space has a sensible
//! default, see [`ColorSpec::reference_white_nits`], overridable with
//! [`ColorConverterConfig::with_reference_white_nits`].
//!
//! Supported input formats: BGRx, RGBx, BGRA, RGBA, ABGR2101010 (10-bit packed), RGBA16F (FP16).
//! Supported output formats: NV12 (8-bit), I420 (8-bit), YUV444 (8-bit), P010 (10-bit), YUV444P10 (10-bit).
//!
//! Pass [`ColorConverter::color_description`] to the encoder rather than
//! describing the colours a second time by hand. The two have to agree: if the
//! converter writes full-range pixels and the stream says limited, players
//! stretch the range again and the picture comes out wrong.
//!
//! ```rust,no_run
//! use pixelforge::{
//!     Codec, ColorConverter, ColorConverterConfig, ColorRange, ColorSpec, EncodeConfig,
//!     Encoder, InputFormat, OutputFormat, VideoContextBuilder,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let context = VideoContextBuilder::new()
//!     .app_name("Color Converter")
//!     .require_encode(Codec::H265)
//!     .build()?;
//!
//! // SDR desktop content, encoded as HDR10 for an HDR streaming session.
//! let config = ColorConverterConfig::new(
//!     1920,
//!     1080,
//!     InputFormat::BGRx,
//!     OutputFormat::P010,
//!     ColorSpec::Srgb,
//!     ColorSpec::Bt2020Pq,
//!     ColorRange::Full,
//! );
//!
//! let mut converter = ColorConverter::new(context.clone(), config)?;
//!
//! // The encoder declares exactly what the shader wrote.
//! let encode_config = EncodeConfig::h265(1920, 1080)
//!     .with_color_description(converter.color_description());
//!
//! let mut encoder = Encoder::new(context, encode_config)?;
//! // converter.convert(input_image, layout, encoder.input_image())?;
//! # Ok(())
//! # }
//! ```
//!
//! Where the conversion is nothing but the YUV matrix, the encoder may be able
//! to do it with no shader at all:
//! [`ColorConverterConfig::rgb_encode_input`] says when, and
//! [`EncodeConfig::with_rgb_input`] hands it the RGB frames.
//!
//! ## Encoding on your own device
//!
//! An application that already renders with Vulkan can give pixelforge its own
//! device instead of letting it create a second one. Frames then never leave
//! the device: no external memory, no import, and no CPU wait between the
//! application's work and pixelforge's.
//!
//! [`VideoContextBuilder::encode_device_requirements`] lists the queue
//! families, extensions and features to create the device with, and
//! [`VideoContextBuilder::build_from_existing_encode`] adopts it. A `VkQueue`
//! may not be submitted to from two threads at once, so give pixelforge queues
//! the application does not use, with
//! [`VideoContextBuilder::with_encode_queue`] and friends.
//!
//! [`ColorConverter::convert_async`] and [`Encoder::encode_after`] then order
//! each frame on the GPU: the application's submission signals a
//! [`TimelinePoint`], the conversion waits for it and signals its own, and the
//! encode waits for that.
//!
//! ```rust,no_run
//! # use ash::vk;
//! # use pixelforge::{ColorConverter, Encoder, TimelinePoint};
//! # fn frame(
//! #     converter: &mut ColorConverter,
//! #     encoder: &mut Encoder,
//! #     rendered: vk::Semaphore,
//! #     frame_number: u64,
//! #     image: vk::Image,
//! # ) -> Result<(), Box<dyn std::error::Error>> {
//! // `rendered` reaches `frame_number` once the application has drawn `image`.
//! let converted = converter.convert_async(
//!     image,
//!     vk::ImageLayout::GENERAL,
//!     encoder.input_image(),
//!     &[TimelinePoint::new(rendered, frame_number)],
//! )?;
//! let packet = encoder.encode_after(encoder.input_image(), &[converted])?;
//! # Ok(())
//! # }
//! ```
//!
//! `examples/encode_adopted.rs` builds such a device from scratch.
//!
//! # Benchmarking
//!
//! Run the encode latency benchmark with:
//!
//! ```text
//! cargo bench --bench encode
//! ```
//!
//! # Examples
//!
//! Run the examples with:
//!
//! ```text
//! # Query codec capabilities
//! cargo run --example query_capabilities
//!
//! # Decode H.264 to raw YUV
//! cargo run --example decode -- input.264 output.yuv
//!
//! # Decode on a caller-created Vulkan device
//! cargo run --example decode_adopted -- input.264 output.yuv
//!
//! # Encode on a caller-created Vulkan device
//! cargo run --example encode_adopted -- input.yuv output.h264
//!
//! # Encode, choosing the codec (h264, h265 or av1)
//! cargo run --example encode -- h265
//!
//! # Sample decoded frames through a ycbcr conversion (RGBA output)
//! cargo run --example sample_frame -- input.264 out.rgba
//!
//! # Sample decoded frames through per-plane views (NV12 output)
//! cargo run --example sample_planes -- input.264 out.yuv
//! ```
//!
//! Correctness checks that need a video device and ffmpeg are integration
//! tests, ignored by default:
//!
//! ```text
//! cargo test -- --ignored
//! ```
//!
//! # Shader Development
//!
//! The color conversion shader is precompiled to SPIR-V and embedded at build time.
//! See [shader/README.md](shader/README.md) for details on editing and recompiling shaders.
//!
//! # TODO's
//!
//! 1. [] H.265 and AV1 decoding.
//! 1. [] B-frames support (encode).
//!
//! # Contributing
//!
//! Contributions are welcome! Please feel free to submit a Pull Request.
//!
//! # Acknowledgement
//!
//! This project was heavily inspired by the [vk_video_samples](https://github.com/nvpro-samples/vk_video_samples)
//! repository by NVIDIA, which provided invaluable reference for Vulkan Video encoding.

pub mod converter;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod image;
pub mod sync;
pub(crate) mod video;
pub mod vulkan;

/// Align a byte size up to a multiple of 4.
///
/// Required for `VkBufferImageCopy::bufferOffset` to meet the texel block alignment
/// of multi-component plane formats (e.g. R8G8, R16G16).
pub(crate) const fn align4(size: usize) -> usize {
    (size + 3) & !3
}

pub use converter::{
    ColorConverter, ColorConverterConfig, ColorRange, ColorSpec, InputFormat, OutputFormat,
};
pub use decoder::{
    DecodeConfig, DecodeSink, DecodeSource, DecodeStatus, DecodedFrame, Decoder, FramePoll, Framing,
};
pub use encoder::{
    BitDepth as EncodeBitDepth, Codec, ColorDescription, DEFAULT_FRAME_RATE, DEFAULT_GOP_SIZE,
    DEFAULT_H264_QP, DEFAULT_H265_QP, DEFAULT_MAX_BITRATE, DEFAULT_MAX_REFERENCE_FRAMES,
    DEFAULT_TARGET_BITRATE, EncodeConfig, EncodeContentHint, EncodeFuture, EncodeUsageHint,
    EncodedPacket, Encoder, EncoderTuningMode, FrameType, PixelFormat, RateControlMode,
};
pub use error::PixelForgeError;
pub use image::InputImage;
pub use sync::TimelinePoint;
pub use vulkan::VideoContextBuilder;

/// Re-export VideoContext for convenience.
pub use vulkan::VideoContext;
