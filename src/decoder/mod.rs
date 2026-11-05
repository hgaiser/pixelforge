//! Hardware-accelerated video decoding using Vulkan Video.
//!
//! The decoder consumes a coded byte stream and produces decoded frames as GPU
//! images (`vk::Image`), optionally with CPU readback. [`Decoder::decode`]
//! takes whatever the caller has: whole coded frames, if they arrive already
//! framed by a container or transport, or an arbitrary slice of a byte stream,
//! which the decoder frames itself using the codec's own rules (Annex B start
//! codes for H.264/H.265, OBU temporal units for AV1). See [`Framing`].
//!
//! # Design
//!
//! - **Asynchronous**: [`DecodeSink::decode`] records and submits without
//!   waiting for the GPU, so the CPU can parse and submit the next picture
//!   while the current one decodes. Frames arrive on the
//!   [`DecodeSource`] as they complete, and the two halves can be driven from
//!   separate threads. Mirrors
//!   [`Encoder::submit`](crate::encoder::Encoder::submit).
//! - **Stream-driven**: the Vulkan video session is created lazily from the
//!   stream's own parameter sets, so no dimensions or profile need to be
//!   configured up front. Mid-stream resolution changes recreate the session
//!   transparently.
//! - **Display order**: frames come out in presentation order; call
//!   [`DecodeSink::finish`] at end of stream to emit what reordering held back.
//!   Streams without B-frames buffer nothing and add no latency.
//! - **Zero-copy output**: where the device supports unified image layouts, a
//!   [`DecodedFrame`] is the decoder's own DPB image, held until the frame is
//!   dropped, and reordering holds images rather than copying pictures out.
//!   Elsewhere pictures are copied into private images instead. Either way the
//!   image is valid for exactly as long as the frame is; see [`DecodedFrame`]
//!   for what holding one costs.
//!
//! # Limitations
//!
//! - **H.264**: progressive streams only (no interlaced/field coding), and no
//!   explicit scaling matrices. Reference management is complete (IDR,
//!   sliding-window, and MMCO), so any stream this crate's encoder produces,
//!   and typical output from x264, NVENC and friends, including B-pyramid,
//!   decodes correctly.
//! - **H.265**: the base layer only; Main and Main 10 (and 4:2:2 or 4:4:4 where
//!   the device has the profile), progressive. The screen content and
//!   multi-layer extensions are rejected. RASL pictures after the CRA that
//!   decoding starts at are skipped, as the standard allows.
//! - **AV1**: low-overhead OBU streams (the format of `.obu` files, IVF and
//!   MP4), not Annex B; operating point 0. Monochrome streams and 12-bit are
//!   rejected. Film grain is applied, which takes a second image for every DPB
//!   slot. A device whose DPB and output images do not coincide is not
//!   supported.

/// Annex B start-code framing, shared by H.264 and H.265.
pub(crate) mod annexb;
/// AV1 decoding.
pub(crate) mod av1;
pub(crate) mod bitreader;
/// The Vulkan video session and the per-decoder state every codec shares.
pub(crate) mod common;
/// The decode flow every codec shares.
pub(crate) mod driver;
/// Frame ownership: pool images, DPB slot pins, display-order reordering.
pub(crate) mod frames;
/// H.264 decoding.
pub(crate) mod h264;
/// H.265 decoding.
pub(crate) mod h265;
pub(crate) mod pipeline;
/// Turning what a stream needs into a session plan.
pub(crate) mod plan;
pub(crate) mod profile;
/// The structures a decode command is recorded from.
pub(crate) mod record;
/// Readback and copies, all of which run on the transfer queue.
pub(crate) mod transfer;

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use ash::vk;
use futures_core::Stream;

use crate::encoder::{BitDepth, ChromaFormat, Codec};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;

/// How the bytes handed to [`Decoder::decode`] are framed.
///
/// The decoder has to know where one coded frame ends and the next begins.
/// Which of these applies is a property of where the bytes come from, not of
/// the codec, so it cannot be detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    /// Every call carries whole coded frames: the end of the data is the end of
    /// a frame (the default).
    ///
    /// This is what a container or a transport gives you, where framing has
    /// already been done: an RTP access unit, an MP4 sample, a WebRTC frame.
    /// Nothing is buffered, so a frame is decoded by the call that delivers it
    /// and this is the lower-latency option.
    FrameAligned,
    /// A continuous byte stream that may cut anywhere, including mid-frame.
    ///
    /// This is what a raw `.264` file or a socket carrying one gives you. The
    /// decoder buffers a trailing partial frame until later bytes complete it,
    /// which costs one frame of latency: a coded frame cannot be known to be
    /// complete until the start of the next one is seen. [`DecodeSink::finish`]
    /// decodes whatever is still buffered, so the last frame is not lost.
    ByteStream,
}

/// How many decoded frames may be outstanding at once, by default.
///
/// Two for the caller to hold while consuming them, matching the encoder's
/// pipeline depth, plus two for the frames the decode pipeline itself has in
/// flight between submitting a picture and delivering it.
///
/// A caller feeding several coded frames per [`Decoder::decode`] call needs
/// more, since every frame that call emits is outstanding at once. See
/// [`DecodeConfig::with_output_depth`].
pub const DEFAULT_OUTPUT_DEPTH: usize = 4;

/// Configuration for creating a [`Decoder`].
#[derive(Debug, Clone)]
pub struct DecodeConfig {
    /// The codec to decode.
    pub codec: Codec,
    /// How many decoded frames the caller may hold at once. Defaults to
    /// [`DEFAULT_OUTPUT_DEPTH`]. See [`DecodeConfig::with_output_depth`].
    pub output_depth: usize,
    /// Queue family that will read decoded frames, if it is not the decoder's
    /// own. See [`DecodeConfig::with_consumer_queue_family`].
    pub consumer_queue_family: Option<u32>,
    /// How input handed to [`Decoder::decode`] is framed. Defaults to
    /// [`Framing::FrameAligned`].
    pub framing: Framing,
}

impl DecodeConfig {
    /// Create a decode configuration for `codec`.
    pub fn new(codec: Codec) -> Self {
        Self {
            codec,
            output_depth: DEFAULT_OUTPUT_DEPTH,
            consumer_queue_family: None,
            framing: Framing::FrameAligned,
        }
    }

    /// How many decoded frames the caller may hold at once.
    ///
    /// A frame is the decoder's own DPB image, so an outstanding frame keeps a
    /// DPB slot reserved. The decoder allocates this many slots beyond what the
    /// stream's references and reorder depth need.
    ///
    /// Exceeding it costs a copy, not an error: pictures past the budget are
    /// copied into private images so decoding can continue. So this is a
    /// performance setting, and the number to match is how many frames are
    /// outstanding at once, which is everything a [`Decoder::decode`] call
    /// emits plus whatever the caller is still holding. Feeding one coded frame
    /// per call keeps that small; feeding 64 KB of byte stream at a time does
    /// not.
    ///
    /// Clamped to what the device's DPB slot limit allows for the stream, so a
    /// device too small for the request degrades to copying rather than
    /// failing. Each slot costs one decoded picture's worth of memory.
    ///
    /// The number to match is how many frames are outstanding at once: every
    /// frame a [`DecodeSink::decode`] call emits, plus whatever has been pulled
    /// from the [`DecodeSource`] and not yet dropped.
    pub fn with_output_depth(mut self, depth: usize) -> Self {
        self.output_depth = depth;
        self
    }

    /// Name the queue family that will read decoded frames.
    ///
    /// A [`DecodedFrame`] is a decoder-owned image, and images are shared
    /// between queue families explicitly. By default the decoder shares its
    /// pictures with its own decode and transfer families only. A renderer
    /// sampling frames from a graphics queue is a *third* family, and reading
    /// an image from a family it was not shared with is undefined behaviour
    /// unless the caller performs a queue family ownership transfer
    /// themselves.
    ///
    /// Passing that family here adds it to the sharing set, so frames can be
    /// sampled directly with no ownership transfer and no copy. Set it whenever
    /// frames are consumed by anything outside the decoder; for a context
    /// adopted from an existing device (see
    /// [`build_from_existing`](crate::vulkan::VideoContextBuilder::build_from_existing))
    /// this is the family the caller already renders on.
    pub fn with_consumer_queue_family(mut self, family: u32) -> Self {
        self.consumer_queue_family = Some(family);
        self
    }

    /// Feed a continuous byte stream that may cut mid-frame, rather than whole
    /// coded frames per call. See [`Framing::ByteStream`].
    pub fn with_byte_stream(mut self) -> Self {
        self.framing = Framing::ByteStream;
        self
    }
}

/// A decoded frame residing in GPU memory.
///
/// # Validity
///
/// [`image`](Self::image) is a decoder-owned GPU image whose storage this frame
/// keeps reserved. It stays valid and unmodified for as long as the frame is
/// alive; dropping the frame hands the storage back to the decoder.
///
/// Two consequences worth planning for:
///
/// - **Hold frames briefly.** The image *is* a DPB slot (no copy), so a live
///   frame reserves one of the [`DecodeConfig::with_output_depth`] spare slots,
///   and once they are all reserved the decoder copies pictures out instead of
///   handing over its own, which still works but gives up the zero-copy path.
///   A frame alive across a session rebuild also keeps that whole image alive.
///
///   If something needs a picture for longer than the moment it is rendered,
///   copy it into an image of your own and drop the frame. Keeping decoded
///   frames as a buffer is the one usage this API is not built for.
/// - **Do not drop a frame while your own GPU work on it is still running.**
///   The drop is what returns the storage, so the image may be reused or
///   destroyed immediately afterwards. Wait for your fence first.
#[derive(Debug)]
pub struct DecodedFrame {
    /// The decoded picture on the GPU.
    ///
    /// The image format is the decoder's picture format (see
    /// [`Decoder::picture_format`]), typically NV12
    /// (`G8_B8R8_2PLANE_420_UNORM`) or P010 for 10-bit streams. The image is
    /// in `VIDEO_DECODE_DPB_KHR` or `VIDEO_DECODE_DST_KHR` layout (see
    /// [`layout`](Self::layout)) and sized to the coded dimensions; only the
    /// `width` x `height` top-left region contains the visible picture.
    ///
    /// Created with `TRANSFER_SRC` and, where the device allows it,
    /// `SAMPLED` (see [`sampleable`](Self::sampleable)). Reading it from a
    /// queue family other than the decoder's own requires
    /// [`DecodeConfig::with_consumer_queue_family`].
    pub image: vk::Image,
    /// An image view over the whole decoded picture.
    pub image_view: vk::ImageView,
    /// Current layout of `image`.
    pub layout: vk::ImageLayout,
    /// Array layer within `image` holding this picture. Always 0 for a
    /// non-layered DPB or a distinct output image; the DPB slot's layer
    /// when the DPB is layered and the driver decoded straight into it.
    pub array_layer: u32,
    /// The chroma format of this frame.
    pub chroma_format: ChromaFormat,
    /// Bit depth of the samples: eight for NV12, ten for P010.
    pub bit_depth: BitDepth,
    /// Which set of decoder images [`image`](Self::image) belongs to.
    ///
    /// The decoder rebuilds its session, and with it every picture image, when
    /// the stream's geometry or parameter sets change. This frame's image
    /// stays valid regardless, for as long as the frame lives; the generation
    /// is not about validity.
    ///
    /// It is about **caching**. Anything keyed on a `vk::Image` handle, such as
    /// the views a renderer builds over decoded frames, outlives the frames it
    /// was built from. Once the last frame of a generation is dropped its
    /// images really are destroyed, and drivers reuse handles freely, so a
    /// handle can come back naming a different image and a cache keyed on the
    /// handle alone will hit and return views of something else. Key on
    /// `(generation, image, array_layer)` and the problem disappears.
    ///
    /// Frames from the copying path carry a generation too, so a consumer never
    /// has to know which path a frame took.
    pub generation: u64,
    /// Visible (cropped) width in pixels.
    pub width: u32,
    /// Visible (cropped) height in pixels.
    pub height: u32,
    /// Coded width in pixels (image width).
    pub coded_width: u32,
    /// Coded height in pixels (image height).
    pub coded_height: u32,
    /// Presentation timestamp, passed through from [`Decoder::decode`].
    pub pts: u64,
    /// Position of this frame in presentation order within the current coded
    /// video sequence: the codec's own ordering value (H.26x picture order
    /// count, AV1 order hint). Only the relative ordering carries meaning, and
    /// it restarts at every keyframe. Frames are already emitted sorted by it.
    pub display_order: i32,
    /// Whether this frame is a keyframe: a random-access point starting a new
    /// coded video sequence (H.264 IDR, H.265 IRAP, AV1 key frame).
    pub is_keyframe: bool,
    /// Whether [`image`](Self::image) was created with `SAMPLED` usage, so a
    /// shader can read it directly rather than copying from it first.
    ///
    /// True on every driver tested so far. It is false only where the device
    /// reports no usable picture format for a sampleable decode image, in which
    /// case the frame can still be copied from (`TRANSFER_SRC`).
    ///
    /// Sampling also needs a view of the caller's own: the format is
    /// multi-planar YCbCr, so it takes a `VkSamplerYcbcrConversion`, and the
    /// image must be transitioned out of [`layout`](Self::layout) first, unless
    /// it is already in `GENERAL`.
    pub sampleable: bool,
    /// Whether views of [`image`](Self::image)'s individual planes are legal:
    /// a view with a `VK_IMAGE_ASPECT_PLANE_i_BIT` aspect and that plane's
    /// compatible single-plane format (`R8_UNORM` for luma and `R8G8_UNORM`
    /// for chroma in NV12, `R10X6_UNORM_PACK16` and
    /// `R10X6G10X6_UNORM_2PACK16` in P010).
    ///
    /// This is the way to read a decoded picture *without* a
    /// `VkSamplerYcbcrConversion`: two ordinary single-plane textures, with the
    /// YUV to RGB matrix left to the shader. Worth caring about because a
    /// combined image sampler with an immutable ycbcr sampler cannot be
    /// expressed by every shader toolchain, naga and so wgpu among them, and a
    /// renderer built on one of those otherwise has to copy every picture.
    ///
    /// The image is created with `MUTABLE_FORMAT` where the driver allows it
    /// for this profile, format and usage, which both RADV and ANV do for
    /// H.264 4:2:0. Frames that came through the copying path always allow it,
    /// since those images carry no video profile. When false, a consumer needs
    /// the conversion, or a copy of their own.
    ///
    /// Views are the consumer's to create and cache; the decoder rotates
    /// through a handful of images, so key a cache on
    /// [`image`](Self::image) and [`array_layer`](Self::array_layer).
    ///
    /// One thing to get right: a view inherits the image's usage, and a decoded
    /// picture's usage includes `VIDEO_DECODE_DST_KHR`, which a single-plane
    /// format cannot satisfy. Chain a `VkImageViewUsageCreateInfo` naming only
    /// what the view is for, typically `SAMPLED`. Skipping it produces an
    /// invalid view, and only on the drivers where the frame really is the
    /// decoder's DPB image, so it passes on hardware that copies. See
    /// `examples/sample_planes`.
    pub plane_views: bool,
    /// Keeps this frame's storage reserved; releases it on drop. `None` for a
    /// frame that borrows the decoder's DPB image (see the validity rules).
    // Held purely for its `Drop`, which is what returns the storage.
    #[allow(dead_code)]
    pub(crate) pin: Option<frames::FramePin>,
}

/// What a [`DecodeSink::decode`] call did with the data it was given.
///
/// None of these is a failure. Joining a stream mid-flight, or losing packets,
/// routinely produces data that cannot be decoded yet, and a caller has to keep
/// going rather than treat it as an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeStatus {
    /// At least one picture was decoded. Its frames will appear on the
    /// [`DecodeSource`] once the GPU has finished with them.
    Decoded,
    /// Nothing to decode yet. The data held no complete coded frame, which is
    /// the normal state of affairs in [`Framing::ByteStream`] when a chunk ends
    /// mid-picture, and of a chunk carrying only parameter sets.
    Buffered,
    /// Nothing can be decoded until a keyframe arrives.
    ///
    /// Pictures were present but reference decode state that does not exist:
    /// parameter sets or reference pictures that were never seen, which is what
    /// joining a stream mid-flight or losing packets looks like. A live client
    /// should ask the sender for a keyframe (an RTP receiver would send a PLI)
    /// and keep feeding; the decoder recovers on its own once one arrives. The
    /// reason is logged at debug level.
    NeedsKeyframe,
}

/// Receiving half of the decoder's frame channel.
pub(crate) type FrameReceiver = futures_channel::mpsc::UnboundedReceiver<Result<DecodedFrame>>;

/// The codec-erased operations every codec decoder exposes.
trait DecoderApi: Send {
    fn decode(&mut self, data: &[u8], pts: u64) -> Result<DecodeStatus>;
    fn finish(&mut self) -> Result<()>;
    fn take_frame_receiver(&mut self) -> Option<FrameReceiver>;
    fn picture_format(&self) -> Option<vk::Format>;
}

/// Video decoder supporting multiple codecs.
///
/// Two halves that can be used together or apart: bytes go in through the sink,
/// frames come out through the source. Held together here for single-threaded
/// use; [`split`](Self::split) separates them so a producer and a consumer can
/// run on their own threads.
///
/// Constructed via [`Decoder::new`], which selects the codec from the config.
/// Mirrors [`Encoder`](crate::encoder::Encoder): all codec decoders share the
/// same generic driving flow and are held behind a single boxed pointer.
pub struct Decoder {
    sink: DecodeSink,
    source: DecodeSource,
}

/// The input half of a [`Decoder`]: coded bytes go in.
///
/// Owns the Vulkan session and everything that submits work, so this is the
/// half that costs something to use. `Send`, so it can be moved onto a producer
/// thread.
pub struct DecodeSink {
    inner: Box<dyn DecoderApi>,
}

/// The output half of a [`Decoder`]: decoded frames come out.
///
/// Just the receiving end of a channel, so it is cheap to move onto whichever
/// thread renders. Frames arrive in presentation order, each one already
/// decoded: the GPU work behind a frame is complete before it is handed over.
pub struct DecodeSource {
    rx: FrameReceiver,
}

impl Decoder {
    /// Create a new decoder for the codec named in `config`.
    ///
    /// Requires a [`VideoContext`] whose device has a video decode queue and
    /// supports the codec (see
    /// [`VideoContextBuilder::require_decode`](crate::vulkan::VideoContextBuilder::require_decode)
    /// and [`VideoContext::supports_decode`]).
    pub fn new(context: VideoContext, config: DecodeConfig) -> Result<Self> {
        if !context.supports_decode(config.codec) {
            return Err(PixelForgeError::NoSuitableDevice(format!(
                "{:?} decode is not supported by the selected device",
                config.codec
            )));
        }
        let mut inner: Box<dyn DecoderApi> = match config.codec {
            Codec::H264 => Box::new(driver::StreamDecoder::new(
                h264::H264Decoder::create(context, &config)?,
                &config,
            )),
            Codec::H265 => Box::new(driver::StreamDecoder::new(
                h265::H265Decoder::create(context, &config)?,
                &config,
            )),
            Codec::AV1 => Box::new(driver::StreamDecoder::new(
                av1::Av1Decoder::create(context, &config)?,
                &config,
            )),
        };
        let rx = inner
            .take_frame_receiver()
            .expect("a fresh decoder still owns its frame channel");
        Ok(Decoder {
            sink: DecodeSink { inner },
            source: DecodeSource { rx },
        })
    }

    /// Separate the two halves so they can be driven from different threads.
    ///
    /// The sink feeds coded bytes and never waits for output; the source pulls
    /// frames and never touches the codec. Splitting is what lets a decode run
    /// ahead of a renderer, or a network reader run ahead of a decode.
    ///
    /// ```no_run
    /// # use pixelforge::decoder::{DecodeConfig, Decoder};
    /// # fn run(decoder: Decoder, chunks: Vec<Vec<u8>>) -> pixelforge::error::Result<()> {
    /// let (mut sink, mut source) = decoder.split();
    ///
    /// std::thread::spawn(move || -> pixelforge::error::Result<()> {
    ///     for (i, chunk) in chunks.iter().enumerate() {
    ///         sink.decode(chunk, i as u64)?;
    ///     }
    ///     sink.finish()
    /// });
    ///
    /// while let Some(frame) = pollster::block_on(source.next_frame())? {
    ///     // ... render, then drop the frame to release its storage ...
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn split(self) -> (DecodeSink, DecodeSource) {
        (self.sink, self.source)
    }

    /// Feed coded bytes. See [`DecodeSink::decode`].
    pub fn decode(&mut self, data: &[u8], pts: u64) -> Result<DecodeStatus> {
        self.sink.decode(data, pts)
    }

    /// End the stream. See [`DecodeSink::finish`].
    pub fn finish(&mut self) -> Result<()> {
        self.sink.finish()
    }

    /// Take the next decoded frame. See [`DecodeSource::next_frame`].
    pub fn next_frame(&mut self) -> NextFrame<'_> {
        self.source.next_frame()
    }

    /// Take a frame if one is ready. See [`DecodeSource::try_next_frame`].
    pub fn try_next_frame(&mut self) -> Result<FramePoll> {
        self.source.try_next_frame()
    }

    /// The Vulkan format of decoded picture images, once known.
    pub fn picture_format(&self) -> Option<vk::Format> {
        self.sink.picture_format()
    }
}

impl DecodeSink {
    /// Feed coded bytes.
    ///
    /// What `data` may contain depends on the configured [`Framing`]. By
    /// default ([`Framing::FrameAligned`]) it must hold whole coded frames:
    /// any number of them, plus whatever non-picture data the codec carries
    /// alongside (H.26x parameter sets and SEI, AV1 sequence headers), ending
    /// on a frame boundary. Feed exactly one coded frame per call for the
    /// lowest latency.
    ///
    /// With [`Framing::ByteStream`] the data may cut anywhere and the decoder
    /// does the framing itself, holding back a trailing partial frame until
    /// later calls complete it.
    ///
    /// `pts` is attached to every frame this data produces.
    ///
    /// Returns as soon as the work is submitted, without waiting for the GPU.
    /// Frames appear on the [`DecodeSource`] as they complete, so this call
    /// yields no frames of its own and the two halves can run independently.
    ///
    /// The [`DecodeStatus`] says what happened to the data. An `Err` means
    /// something actually went wrong; data that merely cannot be decoded yet,
    /// which is the normal case when joining a stream or recovering from loss,
    /// comes back as [`DecodeStatus::NeedsKeyframe`] instead.
    pub fn decode(&mut self, data: &[u8], pts: u64) -> Result<DecodeStatus> {
        self.inner.decode(data, pts)
    }

    /// End the stream: decode whatever is still buffered, emit the frames held
    /// back for reordering, and close the source.
    ///
    /// Call once, after the last [`decode`](Self::decode). Until it is called
    /// [`DecodeSource::next_frame`] never reports the end, since more frames
    /// could always follow.
    pub fn finish(&mut self) -> Result<()> {
        self.inner.finish()
    }

    /// The Vulkan format of decoded picture images, once known.
    ///
    /// `None` until the first parameter set has been consumed (the format is
    /// negotiated from the stream's profile).
    pub fn picture_format(&self) -> Option<vk::Format> {
        self.inner.picture_format()
    }
}

/// The result of asking a [`DecodeSource`] for a frame without waiting.
#[derive(Debug)]
pub enum FramePoll {
    /// A decoded frame.
    Frame(DecodedFrame),
    /// Nothing ready yet. More is still coming.
    Pending,
    /// The stream ended and every frame has been delivered.
    Finished,
}

impl DecodeSource {
    /// Take a frame if one is ready, without waiting.
    ///
    /// For driving both halves from one thread: feed a chunk, drain whatever
    /// has become ready, repeat. [`FramePoll::Pending`] means the GPU has not
    /// finished with the frames in flight yet, not that there are none, so a
    /// caller that keeps feeding will collect them on a later pass.
    ///
    /// Use [`next_frame`](Self::next_frame) instead when a thread has nothing
    /// better to do than wait.
    pub fn try_next_frame(&mut self) -> Result<FramePoll> {
        use futures_channel::mpsc::TryRecvError;
        match self.rx.try_recv() {
            Ok(Ok(frame)) => Ok(FramePoll::Frame(frame)),
            Ok(Err(e)) => Err(e),
            Err(TryRecvError::Empty) => Ok(FramePoll::Pending),
            Err(TryRecvError::Closed) => Ok(FramePoll::Finished),
        }
    }

    /// Wait for the next decoded frame.
    ///
    /// Resolves to `Ok(None)` once the stream has ended, which is after
    /// [`DecodeSink::finish`] and every frame before it. Frames arrive in
    /// presentation order.
    ///
    /// Drop each frame as soon as you are done with it: a live frame holds a
    /// DPB slot, and once the slots run out the decoder copies pictures out
    /// instead of handing over its own images.
    pub fn next_frame(&mut self) -> NextFrame<'_> {
        NextFrame { rx: &mut self.rx }
    }
}

/// The future returned by [`DecodeSource::next_frame`].
pub struct NextFrame<'a> {
    rx: &'a mut FrameReceiver,
}

impl Future for NextFrame<'_> {
    type Output = Result<Option<DecodedFrame>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut *self.rx).poll_next(cx) {
            Poll::Ready(Some(Ok(frame))) => Poll::Ready(Ok(Some(frame))),
            Poll::Ready(Some(Err(e))) => Poll::Ready(Err(e)),
            // The sender is dropped at end of stream, and also if the decoder
            // is torn down; either way there is nothing more to come.
            Poll::Ready(None) => Poll::Ready(Ok(None)),
            Poll::Pending => Poll::Pending,
        }
    }
}
