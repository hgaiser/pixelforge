//! The public [`Encoder`]: the codec encoder, plus whatever turns the
//! configured [`Source`](super::Source) into its YUV.

use ash::vk;

use super::codec::{CodecEncoder, EncoderCommon, VideoCodec};
use super::frame::{Frame, FrameSource};
use super::input::{FrameInput, FrameSync, SourceOwnership};
use super::source::{Conversion, ConversionPreference, plan, target_space};
use super::{Codec, ColorDescription, EncodeConfig, EncodeFuture};
use crate::converter::{ColorConverter, ColorRange};
use crate::error::{PixelForgeError, Result};
use crate::sync::TimelinePoint;
use crate::vulkan::VideoContext;

static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The codec-erased operations every [`CodecEncoder`] exposes.
///
/// One blanket impl covers all codecs, so [`Encoder`] can hold any of them
/// behind a single boxed pointer instead of an enum that re-dispatches by hand.
pub(crate) trait EncoderApi: Send {
    fn common(&self) -> &EncoderCommon;
    fn input_image(&self) -> vk::Image;
    fn encode_frame(&mut self, input: FrameInput, sync: FrameSync) -> Result<EncodeFuture>;
    fn flush(&mut self) -> Result<()>;
    fn request_idr(&mut self);
    fn invalidate_reference_frames(&mut self, first_lost_display_order: u64);
    fn set_color_description(&mut self, desc: ColorDescription) -> Result<()>;
}

impl<C: VideoCodec> EncoderApi for CodecEncoder<C> {
    fn common(&self) -> &EncoderCommon {
        &self.common
    }
    fn input_image(&self) -> vk::Image {
        CodecEncoder::input_image(self)
    }
    fn encode_frame(&mut self, input: FrameInput, sync: FrameSync) -> Result<EncodeFuture> {
        CodecEncoder::encode_frame(self, input, sync)
    }
    fn flush(&mut self) -> Result<()> {
        CodecEncoder::flush(self)
    }
    fn request_idr(&mut self) {
        CodecEncoder::request_idr(self)
    }
    fn invalidate_reference_frames(&mut self, first_lost_display_order: u64) {
        CodecEncoder::invalidate_reference_frames(self, first_lost_display_order)
    }
    fn set_color_description(&mut self, desc: ColorDescription) -> Result<()> {
        CodecEncoder::set_color_description(self, desc)
    }
}

/// Video encoder supporting multiple codecs.
///
/// Constructed via [`Encoder::new`], which selects the codec from the config,
/// and for an RGB [`Source`](super::Source) the [`Conversion`] that turns
/// frames into the stream's YUV. Frames go in through [`Encoder::submit`],
/// whatever the conversion.
pub struct Encoder {
    inner: Box<dyn EncoderApi>,
    context: VideoContext,
    config: EncodeConfig,
    conversion: Conversion,
    converter: Option<ColorConverter>,
    /// Added to every packet's timestamps: the frames encoded before the
    /// codec encoder was last made again.
    pts_offset: u64,
    /// Tells encoders apart, so an image made for one is not read by another.
    id: u64,
    /// Counts the times the codec encoder was made again with a different
    /// conversion. An [`ExternalImage`](super::external::ExternalImage) made
    /// for one cannot be read by another.
    generation: u64,
    #[cfg(feature = "dmabuf")]
    sync_files: Option<super::external::SyncFiles>,
}

impl Encoder {
    /// Create a new encoder for the codec named in `config`.
    ///
    /// For an RGB [`Source`](super::Source), this also picks the
    /// [`Conversion`], see [`Self::conversion`]. It fails if no conversion
    /// gets from the source to [`EncodeConfig::color_description`], or if
    /// [`EncodeConfig::conversion`] asks for one that cannot.
    pub fn new(context: VideoContext, config: EncodeConfig) -> Result<Self> {
        let plan = plan(&context, &config)?;
        if let Some(format) = plan.rgb_input {
            super::rgb::check_rgb_input(&context, &config, format)?;
        }
        let converter = plan
            .converter
            .map(|c| ColorConverter::new(context.clone(), c))
            .transpose()?;
        let inner = create_codec_encoder(&context, &config, plan.rgb_input)?;
        tracing::debug!(
            "{:?} encoder, {:?} source, conversion {:?}",
            config.codec,
            config.source,
            plan.conversion
        );
        Ok(Self {
            inner,
            context,
            config,
            conversion: plan.conversion,
            converter,
            pts_offset: 0,
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            generation: 0,
            #[cfg(feature = "dmabuf")]
            sync_files: None,
        })
    }

    /// The conversion this encoder picked for its source.
    pub fn conversion(&self) -> Conversion {
        self.conversion
    }

    /// The configuration, with any colour description set since.
    pub fn config(&self) -> &EncodeConfig {
        &self.config
    }

    /// Get the current slot's input image.
    ///
    /// For a YUV source, rendering or uploading into this image and passing it
    /// to [`Frame::image`] encodes it in place, with no copy. It is in
    /// `VIDEO_ENCODE_SRC_KHR` layout, and must be left there. This call waits
    /// until the slot's previous encode has finished with it.
    ///
    /// For an RGB source the input image belongs to the conversion: hand the
    /// RGB frames to [`Self::submit`] instead.
    pub fn input_image(&self) -> vk::Image {
        self.inner.input_image()
    }

    /// Encode a frame from a GPU image in `GENERAL` layout, and return once
    /// the image can be written again.
    ///
    /// Like `submit(Frame::image(src_image, vk::ImageLayout::GENERAL))`, but
    /// for an image other than [`Self::input_image`] it also waits on the CPU
    /// for [`EncodeFuture::source_released`]: the copy or conversion out of
    /// it. That keeps the simple loop of writing one image and encoding it
    /// correct. [`Self::submit`] does not wait, for callers that order their
    /// writes on the GPU instead.
    ///
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
        // Read before submitting, since submitting moves to the next slot.
        let in_place =
            self.conversion != Conversion::Shader && src_image == self.inner.input_image();
        let future = self.submit(Frame::image(src_image, vk::ImageLayout::GENERAL))?;
        if !in_place {
            self.context.wait_point(future.source_released())?;
        }
        Ok(future)
    }

    /// Encode a frame once GPU work the caller submitted has finished.
    ///
    /// Short for `submit(Frame::image(src_image, src_layout).wait(wait))`. It
    /// does not wait for the source to be read: write it again only after
    /// [`EncodeFuture::source_released`].
    pub fn encode_after(
        &mut self,
        src_image: vk::Image,
        src_layout: vk::ImageLayout,
        wait: &[TimelinePoint],
    ) -> Result<EncodeFuture> {
        self.submit(Frame::image(src_image, src_layout).wait(wait))
    }

    /// Encode one frame.
    ///
    /// Nothing here waits on the CPU for the GPU, except backpressure: when
    /// every pipeline slot is still in flight, this waits for the oldest. The
    /// frame's first submission waits for the frame's
    /// [`wait`](Frame::wait) points, and the source is read by a copy into
    /// the encoder's input image, by the colour converter, or by the encode
    /// itself, depending on the [`Conversion`] and the source.
    ///
    /// Because the call returns before the source is read, write the source
    /// again only after [`EncodeFuture::source_released`].
    pub fn submit(&mut self, frame: Frame<'_>) -> Result<EncodeFuture> {
        #[cfg_attr(not(feature = "dmabuf"), allow(unused_mut))]
        let mut sync = FrameSync {
            waits: frame.waits.iter().map(TimelinePoint::wait_info).collect(),
            release_signals: Vec::new(),
        };

        #[cfg(feature = "dmabuf")]
        let slot = {
            // The slot's semaphores are free once the slot is.
            let _ = self.inner.input_image();
            let slot = self.inner.common().pipeline.current_index();
            if frame.wait_sync_file.is_some() || frame.signal_sync_file {
                let files = match &mut self.sync_files {
                    Some(files) => files,
                    files @ None => files.insert(super::external::SyncFiles::new(&self.context)?),
                };
                if let Some(fence) = frame.wait_sync_file {
                    sync.waits.push(files.import_wait(slot, fence)?);
                }
                if frame.signal_sync_file {
                    sync.release_signals.push(files.release_signal(slot));
                }
            }
            slot
        };

        let mut future = match frame.source {
            FrameSource::Image { image, layout } => {
                self.submit_image(image, SourceOwnership::caller_image(layout), sync)?
            }
            #[cfg(feature = "dmabuf")]
            FrameSource::External(external) => self.submit_external(external, sync)?,
            #[cfg(not(feature = "dmabuf"))]
            FrameSource::Never(_) => unreachable!(),
        };
        future.pts_offset = self.pts_offset;

        #[cfg(feature = "dmabuf")]
        if frame.signal_sync_file
            && let Some(files) = &self.sync_files
        {
            future.release_sync_file = files.export_release(slot)?;
        }
        Ok(future)
    }

    /// Encode from an image the caller owns.
    pub(crate) fn submit_image(
        &mut self,
        image: vk::Image,
        ownership: SourceOwnership,
        sync: FrameSync,
    ) -> Result<EncodeFuture> {
        match self.conversion {
            Conversion::None | Conversion::Encoder => self
                .inner
                .encode_frame(FrameInput::Copy { image, ownership }, sync),
            Conversion::Shader => self.convert_and_encode(image, ownership, sync),
        }
    }

    /// Run the colour converter into the slot's input image, then encode it.
    pub(crate) fn convert_and_encode(
        &mut self,
        image: vk::Image,
        ownership: SourceOwnership,
        sync: FrameSync,
    ) -> Result<EncodeFuture> {
        let target = self.inner.input_image();
        if image == target {
            return Err(PixelForgeError::InvalidInput(
                "the encoder's input image holds converted YUV; hand it the RGB frame instead"
                    .to_string(),
            ));
        }
        let converter = self
            .converter
            .as_mut()
            .expect("a shader conversion has a converter");
        let converted =
            converter.convert_with(image, ownership, target, &sync.waits, &sync.release_signals)?;
        let mut future = self.inner.encode_frame(
            FrameInput::InPlace,
            FrameSync {
                waits: vec![converted.wait_info()],
                release_signals: Vec::new(),
            },
        )?;
        future.set_released(converted);
        Ok(future)
    }

    #[cfg_attr(not(feature = "dmabuf"), allow(dead_code))]
    pub(crate) fn inner(&self) -> &dyn EncoderApi {
        &*self.inner
    }

    #[cfg_attr(not(feature = "dmabuf"), allow(dead_code))]
    pub(crate) fn context(&self) -> &VideoContext {
        &self.context
    }

    #[cfg_attr(not(feature = "dmabuf"), allow(dead_code))]
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    #[cfg_attr(not(feature = "dmabuf"), allow(dead_code))]
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Encode `input` with the codec encoder, as it is.
    #[cfg_attr(not(feature = "dmabuf"), allow(dead_code))]
    pub(crate) fn encode_input(
        &mut self,
        input: FrameInput,
        sync: FrameSync,
    ) -> Result<EncodeFuture> {
        self.inner.encode_frame(input, sync)
    }

    /// Wait for all in-flight frames to finish encoding (end-of-stream barrier).
    ///
    /// Packets are delivered through the [`EncodeFuture`]s returned by
    /// [`Encoder::submit`]; await those to obtain them. Once this returns,
    /// every outstanding future has been resolved.
    pub fn flush(&mut self) -> Result<()> {
        self.inner.flush()
    }

    /// Request that the next frame be an IDR frame.
    pub fn request_idr(&mut self) {
        self.inner.request_idr()
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
    /// [`EncodedPacket`](super::EncodedPacket)s — of the earliest frame the
    /// client lost.
    ///
    /// Effective recovery requires the encoder to keep more than one reference
    /// (see [`EncodeConfig::with_max_reference_frames`]); with a single
    /// reference this necessarily falls back to an IDR. This applies uniformly
    /// across H.264, H.265, and AV1 — each keeps a multi-reference window and
    /// re-anchors prediction to the most recent survivor, falling back to an
    /// IDR (AV1: key frame) only when the loss covers the entire window.
    pub fn invalidate_reference_frames(&mut self, first_lost_display_order: u64) {
        let local = first_lost_display_order.saturating_sub(self.pts_offset);
        self.inner.invalidate_reference_frames(local)
    }

    /// Change what the stream says about its colours, and what the encoder
    /// makes its YUV to match.
    ///
    /// The next frame is an IDR/key frame carrying the new header. For an RGB
    /// source the conversion follows: the colour converter gets the new target
    /// and range, and where the encoder converts, it is made again with the
    /// new matrix and range, since those are fixed in its video session.
    /// Packet timestamps carry on across that. If the new description needs a
    /// different [`Conversion`], images imported or allocated for the old one
    /// have to be made again.
    ///
    /// On error the encoder is left as it was.
    pub fn set_color_description(&mut self, desc: ColorDescription) -> Result<()> {
        let mut config = self.config.clone();
        config.color_description = Some(desc);
        match self.conversion {
            Conversion::None => self.inner.set_color_description(desc)?,
            Conversion::Shader if self.shader_still_converts(&config) => {
                let target = target_space(&desc).expect("the plan checked the target");
                let range = if desc.full_range {
                    ColorRange::Full
                } else {
                    ColorRange::Limited
                };
                let converter = self
                    .converter
                    .as_mut()
                    .expect("a shader conversion has a converter");
                let (old_target, old_range) = (converter.config().target, converter.config().range);
                converter.set_target(target)?;
                converter.set_range(range);
                if let Err(e) = self.inner.set_color_description(desc) {
                    let _ = converter.set_target(old_target);
                    converter.set_range(old_range);
                    return Err(e);
                }
            }
            Conversion::Shader | Conversion::Encoder => self.rebuild(config.clone())?,
        }
        self.config = config;
        Ok(())
    }

    /// Whether the colour converter this encoder has can produce what
    /// `config` describes, so the encoder can keep it.
    fn shader_still_converts(&self, config: &EncodeConfig) -> bool {
        let mut probe = config.clone();
        probe.conversion = ConversionPreference::Shader;
        plan(&self.context, &probe).is_ok()
    }

    /// Make the codec encoder again for `config`, keeping the timestamps
    /// going.
    fn rebuild(&mut self, config: EncodeConfig) -> Result<()> {
        self.inner.flush()?;
        let encoded = self.inner.common().input_frame_num;
        let mut next = Self::new(self.context.clone(), config)?;
        next.pts_offset = self.pts_offset + encoded;
        next.id = self.id;
        next.generation = if next.conversion == self.conversion {
            self.generation
        } else {
            self.generation + 1
        };
        #[cfg(feature = "dmabuf")]
        {
            next.sync_files = self.sync_files.take();
        }
        *self = next;
        Ok(())
    }
}

fn create_codec_encoder(
    context: &VideoContext,
    config: &EncodeConfig,
    rgb_input: Option<crate::converter::InputFormat>,
) -> Result<Box<dyn EncoderApi>> {
    let context = context.clone();
    let config = config.clone();
    Ok(match config.codec {
        Codec::H264 => Box::new(super::h264::H264::create(context, config, rgb_input)?),
        Codec::H265 => Box::new(super::h265::H265::create(context, config, rgb_input)?),
        Codec::AV1 => Box::new(super::av1::Av1::create(context, config, rgb_input)?),
    })
}
