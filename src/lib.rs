//! A Vulkan-based video encoding and decoding library for Rust, supporting H.264,
//! H.265 and AV1, for both encode and decode.
//!
//! # Features
//!
//! - **Hardware-accelerated** video encoding and decoding using Vulkan Video extensions.
//! - **Multiple codec support**: H.264/AVC, H.265/HEVC and AV1, encode and decode.
//! - **Asynchronous pipelines**: both directions submit without waiting.
//!   Encoding hands back an [`EncodeFuture`]; decoding delivers frames through a
//!   [`DecodeSource`] as the GPU finishes with them.
//! - **RGB input**: describe the source once and hand the encoder RGB frames.
//!   It converts them in the encoder itself where `VK_VALVE_video_encode_rgb_conversion`
//!   allows, and with a compute shader otherwise (BT.709, BT.2020,
//!   sRGB→BT.2020+PQ, scRGB-linear→BT.2020+PQ).
//! - **Shared devices**: encode and decode on a Vulkan device your application
//!   created, from Vulkan 1.1 up, on queues you choose, ordered against your own
//!   work with timeline semaphores.
//! - **HDR support**: 10-bit encoding (P010, YUV444P10), PQ transfer function, BT.2020 color space.
//! - **GPU-native API**: Encode directly from Vulkan images (`vk::Image`).
//! - **Flexible configuration**: Rate control (CBR, VBR, CQP), quality levels, GOP settings.
//! - **B-frames**: for all three codecs, flat or as a pyramid, where the
//!   device can encode them. Off by default, since they add latency.
//! - **Multiple input formats**: BGRx, RGBx, BGRA, RGBA, ABGR2101010 (10-bit packed), RGBA16F (FP16).
//! - **Utility helpers**: [`YuvUploader`] to upload YUV frames to the GPU,
//!   and [`YuvDownloader`] to copy decoded frames back to the CPU.
//! - **Optional DMA-BUF input**: import a producer's buffers, or allocate buffers
//!   for it, read with no copy where the driver allows, with sync_file fences
//!   (Linux only).
//!
//! # Supported Codecs
//!
//! | Codec | Encode | Decode |
//! |-------|--------|--------|
//! | H.264/AVC | ✓ | ✓ |
//! | H.265/HEVC | ✓ | ✓ |
//! | AV1 | ✓ | ✓ |
//!
//! H.264 decoding is verified byte-identical to `ffmpeg -pix_fmt nv12` on AMD
//! (RADV), NVIDIA and Intel (ANV). H.265 decoding is verified byte-identical to
//! ffmpeg, and AV1 decoding (film grain included) to dav1d, on NVIDIA so far.
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
//! pixelforge = { git = "https://github.com/hgaiser/pixelforge", tag = "v0.11.0" }
//! ```
//!
//! ## Optional Features
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `dmabuf` | DMA-BUF input and sync_file fences (Linux only). Enables, where the device has them, `VK_KHR_external_memory_fd`, `VK_EXT_external_memory_dma_buf`, `VK_EXT_image_drm_format_modifier`, `VK_KHR_external_semaphore_fd` and `VK_EXT_queue_family_foreign`. |
//!
//! To enable DMA-BUF support:
//!
//! ```toml
//! [dependencies]
//! pixelforge = { git = "https://github.com/hgaiser/pixelforge", tag = "v0.11.0", features = ["dmabuf"] }
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
//! use pixelforge::{Codec, EncodeConfig, Encoder, YuvUploader, RateControl, VideoContextBuilder};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let context = VideoContextBuilder::new()
//!         .app_name("Encoder Example")
//!         .require_encode(Codec::H264)
//!         .build()?;
//!
//!     let config = EncodeConfig::h264(1920, 1080)
//!         .with_rate_control(RateControl::Vbr {
//!             target: 5_000_000,
//!             max: 7_500_000,
//!         })
//!         .with_frame_rate(30, 1)
//!         .with_gop_size(60);
//!
//!     let mut encoder = Encoder::new(context, config)?;
//!     // Create a YuvUploader helper for uploading YUV data to the GPU.
//!     let mut input_image = YuvUploader::for_encoder(&encoder)?;
//!
//!     // For each frame: upload YUV data and encode.
//!     // let yuv_data: &[u8] = ...;  // YUV420 frame data
//!     // input_image.upload(YuvLayout::I420, yuv_data)?;
//!     // let packet = encoder.submit(Frame::image(input_image.source()))?;
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
//!     let config = DecodeConfig::new(Codec::H264).with_byte_stream();
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
//! pictures out instead of handing over its own. To get a frame's samples on
//! the CPU (for a file, a test or software processing), copy it with
//! [`YuvDownloader`].
//!
//! ## RGB input
//!
//! An encoder takes RGB frames once it is told what they are, with a
//! [`Source`]. It then converts them to the YUV that the stream's
//! [`StreamColor`] says: [`StreamColor::Sdr`] (BT.709) or
//! [`StreamColor::Hdr10`] (BT.2020 with PQ), each with its range. The source
//! and the stream are described once each, so the conversion and the
//! stream's label cannot disagree, and a colour change through
//! [`Encoder::reconfigure`] changes both.
//!
//! ```rust,no_run
//! use pixelforge::{
//!     Codec, ColorRange, ColorSpec, EncodeConfig, Encoder, RgbFormat, RgbSource, Source,
//!     StreamColor, VideoContextBuilder,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let context = VideoContextBuilder::new()
//!     .require_encode(Codec::H265)
//!     .build()?;
//!
//! // SDR desktop content, encoded as HDR10 for an HDR streaming session.
//! let config = EncodeConfig::h265(1920, 1080)
//!     .with_bit_depth(pixelforge::BitDepth::Ten)
//!     .with_source(Source::Rgb(RgbSource::new(RgbFormat::BGRx, ColorSpec::Srgb)))
//!     .with_color(StreamColor::Hdr10 {
//!         range: ColorRange::Full,
//!     });
//! let mut encoder = Encoder::new(context, config)?;
//! println!("converting with {:?}", encoder.conversion());
//! // let packet = encoder.submit(Frame::image(&rgb_image))?;
//! # Ok(())
//! # }
//! ```
//!
//! [`Encoder::new`] picks the [`Conversion`]:
//!
//! - [`Conversion::Encoder`] where the conversion is nothing but the YUV
//!   matrix, as for ordinary SDR content, and the device has
//!   `VK_VALVE_video_encode_rgb_conversion` with that matrix and range. No
//!   shader runs at all.
//! - [`Conversion::Shader`] otherwise: the colour converter's compute shader
//!   writes the encoder's YUV input.
//!
//! [`EncodeConfig::with_conversion`] can force one, for tests and comparisons.
//!
//! The shader's colour spaces are [`ColorSpec`]s:
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
//! [`Encoder::new`] rejects the combinations nothing can do.
//!
//! Encoding to HDR needs to know how bright the source's white is, since HDR
//! carries real brightness values and SDR does not. Each space has a sensible
//! default, see [`ColorSpec::reference_white_nits`], overridable with
//! [`RgbSource::with_reference_white_nits`].
//!
//! Supported input formats: BGRx, RGBx, BGRA, RGBA, ABGR2101010 (10-bit packed), RGBA16F (FP16).
//!
//! ## Submitting frames
//!
//! [`Encoder::submit`] is the one way to encode a frame. It takes a [`Frame`]:
//! where the pixels are, and what has to happen on the GPU before they can be
//! read. It does not wait on the CPU, except when every pipeline slot is in
//! flight. The pixels are in one of:
//!
//! - an [`InputSlot`], the encoder's own input image, from
//!   [`Encoder::input_slot`]. The encode reads it in place, with no copy.
//! - a [`SourceImage`], an image of the caller's. Pixelforge copies or
//!   converts it into the input image after the call returns, and the
//!   `SourceImage` keeps track of that: write the image again only after
//!   [`SourceImage::released`]. [`YuvUploader`], for frames uploaded from the
//!   CPU, waits for that itself.
//! - a `DmaBufImage`, a DMA-BUF (see below).
//!
//! ## B-frames
//!
//! [`EncodeConfig::with_b_frames`] puts B-frames between two reference
//! frames. A B-frame predicts from the frame shown before it and from the one
//! shown after it, which costs fewer bits, and it costs latency: the encoder
//! has to wait for the later frame. [`VideoContext::supports_b_frames`] says
//! what the device can do for a codec, and [`Encoder::new`] returns an error
//! for more than that.
//!
//! ```no_run
//! # use std::collections::VecDeque;
//! # use pixelforge::{Encoder, Frame, SourceImage};
//! # fn encode(encoder: &mut Encoder, frames: &[SourceImage]) -> Result<(), Box<dyn std::error::Error>> {
//! let b_frames = encoder.config().b_frames as usize;
//! let mut pending = VecDeque::new();
//! for frame in frames {
//!     pending.push_back(encoder.submit(Frame::image(frame))?);
//!     // At most `b_frames` frames wait for the frame after their group, so
//!     // the oldest future is only sure to resolve once more are pending.
//!     // Waiting for it sooner never returns.
//!     while pending.len() > b_frames {
//!         let packet = pollster::block_on(pending.pop_front().unwrap())?;
//!         // Write `packet.data`. The packets come in decode order.
//!     }
//! }
//! // Encode the frames that still wait, then take the rest.
//! encoder.flush()?;
//! for future in pending {
//!     let packet = pollster::block_on(future)?;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! What changes with B-frames:
//!
//! - The future of the k-th submitted frame resolves with the k-th packet of
//!   the stream, in decode order. That is the packet of another frame, and
//!   [`EncodedPacket::pts`] says which.
//! - For H.264 and H.265, [`EncodedPacket::dts`] counts in decode order and
//!   starts below zero, so it is never after `pts`. An AV1 stream has no
//!   reordering of its own: its packets are in display order, with
//!   `dts == pts`, and the frame encoded ahead travels hidden in an earlier
//!   packet.
//! - [`Encoder::request_idr`], [`Encoder::flush`], [`Encoder::repeat`] and
//!   [`Encoder::reconfigure`] encode the frames that wait at once, as a
//!   shorter group.
//! - [`Encoder::invalidate_reference_frames`] drops every reference encoded
//!   from the first lost frame on, also B-frames shown before it, because
//!   the client lost those packets too. If no reference survives, the oldest
//!   frame that waits for its group becomes the IDR at once.
//! - With a constant QP, a B-frame gets a slightly higher QP than the
//!   other frames, and one step more for each level of a pyramid.
//! - A DMA-BUF is copied when it is submitted and never read directly, so
//!   the producer gets its buffer back at once.
//!
//! How much B-frames save depends on the driver. On NVIDIA (615.78.08) they
//! save bits with H.265 and AV1. With H.264 that driver predicts a B-frame
//! from one direction only, and the stream gets larger than without them.
//!
//! ## Encoding on your own device
//!
//! An application that already renders with Vulkan can give pixelforge its own
//! device instead of letting it create a second one. Frames then never leave
//! the device: no external memory, no import, and no CPU wait between the
//! application's work and pixelforge's.
//!
//! [`VideoContextBuilder::device_requirements`] lists the queue
//! families, extensions and features to create the device with, for the
//! codecs asked for with `require_encode` and `require_decode`, and
//! [`VideoContextBuilder::build_from_existing`] adopts it. A `VkQueue`
//! may not be submitted to from two threads at once, so give pixelforge queues
//! the application does not use, with
//! [`VideoContextBuilder::with_encode_queue`] and friends.
//!
//! Each frame is then ordered on the GPU: the application's submission signals
//! a [`TimelinePoint`], the encoder's first submission for the frame waits for
//! it, and the application waits for [`SourceImage::released`] before it draws
//! into the image again.
//!
//! ```rust,no_run
//! # use ash::vk;
//! # use pixelforge::{Encoder, Frame, SourceImage, TimelinePoint};
//! # fn frame(
//! #     encoder: &mut Encoder,
//! #     rendered: vk::Semaphore,
//! #     frame_number: u64,
//! #     source: &SourceImage,
//! # ) -> Result<(), Box<dyn std::error::Error>> {
//! // `rendered` reaches `frame_number` once the application has drawn
//! // `source.image()`.
//! let ready = [TimelinePoint::new(rendered, frame_number)];
//! let packet = encoder.submit(Frame::image(source).wait(&ready))?;
//! // Wait on these before drawing into the image again.
//! let reusable = source.released();
//! # Ok(())
//! # }
//! ```
//!
//! ## DMA-BUF input
//!
//! With the `dmabuf` feature, an encoder reads DMA-BUFs from another process.
//! `Encoder::dmabuf_format` gives the DRM format it wants and, per DRM
//! modifier, whether it reads such a buffer with no copy. Offer those
//! modifiers to the producer first. `Encoder::import` takes a producer's
//! buffer; `Encoder::allocate_input` makes one for a producer that writes
//! into buffers the consumer supplies, such as `ext-image-copy-capture`.
//! Either is checked once and gives a `DmaBufImage`, which
//! `Frame::external` submits. A buffer the encoder cannot read directly is
//! copied on the GPU instead.
//!
//! The producer's write fence and the encoder's read fence can travel as
//! sync_files: `Frame::wait_sync_file` and `Frame::signal_sync_file`.
//!
//! //! # Benchmarking
//!
//! Run the encode latency benchmark with:
//!
//! ```text
//! cargo bench --bench encode
//! ```
//!
//! # Examples
//!
//! ```text
//! # What this device can encode and decode
//! cargo run --example query_capabilities
//!
//! # Encode frames from CPU memory (h264, h265 or av1)
//! cargo run --example encode -- h265
//!
//! # Encode frames that the application draws on the GPU
//! cargo run --example encode_gpu -- h265
//!
//! # Decode a stream to a raw YUV file
//! cargo run --example decode -- input.264 output.yuv
//!
//! # Decode and sample the frames in a shader, with no copy (RGBA output)
//! cargo run --example sample_frame -- input.264 out.rgba
//! ```
//!
//! The `tools` crate in the repository has command-line tools to check the
//! library, with more options: `cargo run -p pixelforge-tools --bin encode`
//! and `--bin decode`. Correctness checks that need a video device are
//! integration tests, ignored by default:
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
//! 1. [] Open GOPs.
//!
//! # Contributing
//!
//! Contributions are welcome! Please feel free to submit a Pull Request.
//!
//! # Acknowledgement
//!
//! This project was heavily inspired by the [vk_video_samples](https://github.com/nvpro-samples/vk_video_samples)
//! repository by NVIDIA, which provided invaluable reference for Vulkan Video encoding.

// The device tests share `tests/common/source.rs`, which names the crate.
#[cfg(test)]
extern crate self as pixelforge;

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

pub use converter::{ColorRange, ColorSpec, RgbFormat};
pub use decoder::{
    DecodeConfig, DecodeSink, DecodeSource, DecodeStatus, DecodedFrame, Decoder, FramePoll, Framing,
};
#[cfg(feature = "dmabuf")]
pub use encoder::external::{DmaBuf, DmaBufFormat, DmaBufImage, DmaBufModifier, DmaBufPlane};
pub use encoder::{
    BFrameReferences, BFrameSupport, BitDepth, ChromaFormat, Codec, ColorDescription, Dimensions,
    EncodeConfig, EncodeContentHint, EncodeFuture, EncodeUsageHint, EncodedPacket, Encoder,
    EncoderTuningMode, Frame, FrameRate, FrameType, MAX_B_FRAMES, RateControl,
};
pub use encoder::{
    Conversion, ConversionPreference, Handoff, InputSlot, RgbSource, Source, SourceImage,
    StreamColor,
};
pub use error::PixelForgeError;
pub use image::{DownloadedFrame, YuvDownloader, YuvLayout, YuvUploader};
pub use sync::TimelinePoint;
pub use vulkan::VideoContextBuilder;

/// Re-export VideoContext for convenience.
pub use vulkan::VideoContext;
