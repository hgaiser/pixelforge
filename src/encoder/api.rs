//! The public [`Encoder`]: the codec encoder, plus whatever turns the
//! configured [`Source`](super::Source) into its YUV.

use ash::vk;

use super::codec::{CodecEncoder, EncoderCommon, VideoCodec};
use super::frame::{Frame, FrameSource, InputSlot};
use super::input::{FrameInput, FrameSync, SourceOwnership};
use super::source::{Conversion, Source, plan};
use super::source_image::SourceImage;
use super::{Codec, EncodeConfig, EncodeFuture, StreamColor};
use crate::converter::ColorConverter;
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
    fn encode_frame_now(&mut self, input: FrameInput, sync: FrameSync) -> Result<EncodeFuture>;
    fn flush(&mut self) -> Result<()>;
    fn request_idr(&mut self);
    fn invalidate_reference_frames(&mut self, first_lost_display_order: u64);
    fn set_color(&mut self, color: StreamColor) -> Result<()>;
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
    fn encode_frame_now(&mut self, input: FrameInput, sync: FrameSync) -> Result<EncodeFuture> {
        CodecEncoder::encode_frame_now(self, input, sync)
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
    fn set_color(&mut self, color: StreamColor) -> Result<()> {
        CodecEncoder::set_color(self, color)
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
    /// Counts submissions, so an [`InputSlot`] is submitted only before the
    /// next one.
    ticket: u64,
    /// The slot whose input image holds the last frame, for [`Self::repeat`].
    last_frame_slot: Option<usize>,
    /// Counts the times the codec encoder was made again with a different
    /// conversion. A [`DmaBufImage`](super::external::DmaBufImage) made
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
    /// gets from the source to [`EncodeConfig::color`], or if
    /// [`EncodeConfig::conversion`] asks for one that cannot.
    pub fn new(context: VideoContext, config: EncodeConfig) -> Result<Self> {
        config.rate_control.check(config.codec)?;
        super::bframes::check_config(&config)?;
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
            ticket: 0,
            last_frame_slot: None,
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

    /// The encoder's input image for the next frame, to write it in place.
    ///
    /// Write the frame into [`InputSlot::image`] and submit it with
    /// [`Frame::input`]: the encode reads it with no copy. This waits until
    /// the slot's previous encode is done with the image.
    ///
    /// The input image holds what the encode reads: YUV, or RGB where the
    /// encoder converts ([`Conversion::Encoder`]). With
    /// [`Conversion::Shader`] it belongs to the colour converter, and this is
    /// an error: submit the RGB frames from a [`SourceImage`] instead.
    ///
    /// ```no_run
    /// use pixelforge::{EncodeConfig, Encoder, Frame, YuvUploader, VideoContext, YuvLayout};
    ///
    /// # fn example(context: VideoContext) -> Result<(), Box<dyn std::error::Error>> {
    /// let mut encoder = Encoder::new(context, EncodeConfig::h264(1920, 1080))?;
    /// let mut upload = YuvUploader::for_encoder(&encoder)?;
    /// # let yuv_data = vec![0u8; 1920 * 1080 * 3 / 2];
    /// let slot = encoder.input_slot()?;
    /// upload.upload_to(&slot, YuvLayout::I420, &yuv_data)?;
    /// let packet = pollster::block_on(encoder.submit(Frame::input(slot))?)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn input_slot(&self) -> Result<InputSlot> {
        if self.conversion == Conversion::Shader {
            return Err(PixelForgeError::InvalidInput(
                "the input image holds the colour converter's YUV; submit the RGB frame \
                 from a SourceImage instead"
                    .to_string(),
            ));
        }
        Ok(InputSlot {
            image: self.inner.input_image(),
            encoder: self.id,
            ticket: self.ticket,
        })
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
    /// The call returns before the source is read. A [`SourceImage`] or an
    /// `DmaBufImage` keeps track of that itself, see
    /// [`SourceImage::released`].
    pub fn submit(&mut self, frame: Frame<'_>) -> Result<EncodeFuture> {
        if let FrameSource::Input(slot) = &frame.source
            && (slot.encoder, slot.ticket) != (self.id, self.ticket)
        {
            return Err(PixelForgeError::InvalidInput(
                "this input slot is from before the encoder's last submission; \
                 take a new one with input_slot()"
                    .to_string(),
            ));
        }
        self.ticket += 1;
        let fills_input_image = match &frame.source {
            #[cfg(feature = "dmabuf")]
            FrameSource::External(external) => external.fills_input_image(),
            _ => true,
        };
        // The release of a frame written in place is its encode, which with
        // B-frames may come in a later call, after the sync_file for it had
        // to be handed out.
        #[cfg(feature = "dmabuf")]
        if frame.signal_sync_file
            && matches!(frame.source, FrameSource::Input(_))
            && self.config().b_frames > 0
        {
            return Err(PixelForgeError::InvalidInput(
                "with B-frames, a frame written into an input slot has no release sync_file"
                    .to_string(),
            ));
        }
        let frame_slot = self.inner.common().pipeline.next_index();

        #[cfg_attr(not(feature = "dmabuf"), allow(unused_mut))]
        let mut sync = FrameSync {
            waits: frame.waits.iter().map(TimelinePoint::wait_info).collect(),
            release_signals: Vec::new(),
        };

        #[cfg(feature = "dmabuf")]
        let slot = {
            // The slot's semaphores are free once the slot is.
            let slot = frame_slot;
            if frame.wait_sync_file.is_some() || frame.signal_sync_file {
                let files = match &mut self.sync_files {
                    Some(files) => files,
                    files @ None => files.insert(super::external::SyncFiles::new(
                        &self.context,
                        self.inner.common().pipeline.slot_count(),
                    )?),
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
            FrameSource::Input(_) => self.inner.encode_frame(FrameInput::InPlace, sync)?,
            FrameSource::Image(source) => self.submit_source(source, sync)?,
            #[cfg(feature = "dmabuf")]
            FrameSource::External(external) => self.submit_external(external, sync)?,
        };
        future.pts_offset = self.pts_offset;
        self.last_frame_slot = fills_input_image.then_some(frame_slot);

        #[cfg(feature = "dmabuf")]
        if frame.signal_sync_file
            && let Some(files) = &self.sync_files
        {
            future.release_sync_file = files.export_release(slot)?;
        }
        Ok(future)
    }

    /// Encode from a caller's [`SourceImage`].
    fn submit_source(&mut self, source: &SourceImage, mut sync: FrameSync) -> Result<EncodeFuture> {
        if source.image() == self.inner.input_image() {
            return Err(PixelForgeError::InvalidInput(
                "the encoder's input image is submitted with Frame::input".to_string(),
            ));
        }
        sync.waits.extend(
            source
                .pending_release()
                .iter()
                .map(TimelinePoint::wait_info),
        );
        let future = self.submit_image(source.image(), source.ownership(), sync)?;
        source.mark_used(future.source_released());
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
        // This waits until the slot is free, so the conversion never writes
        // into an image that an earlier encode still reads.
        let target = self.inner.input_image();
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

    /// Encode the last frame again: an IDR if one was requested, for a client
    /// that needs a key frame while no new frame arrives.
    ///
    /// The encoder still holds the last frame in an input image, so the
    /// caller's source does not have to stay alive or unchanged. When that
    /// image is another slot's, this waits on the CPU until the encode is
    /// done, because the next frame can be written into it.
    ///
    /// An error if there is no last frame to repeat: nothing was submitted
    /// since the encoder was made or reconfigured into a new encoder, or the
    /// last frame was read straight from a DMA-BUF.
    pub fn repeat(&mut self) -> Result<EncodeFuture> {
        let Some(last) = self.last_frame_slot else {
            return Err(PixelForgeError::InvalidInput(
                "no frame to repeat: none was submitted since the encoder was made, \
                 or the last one was read straight from a DMA-BUF"
                    .to_string(),
            ));
        };
        self.ticket += 1;
        let common = self.inner.common();
        // The encode writes nothing into its own slot's image, so the frame
        // stays where it is for the next repeat.
        let other_slot = last != common.pipeline.next_index();
        let input = if other_slot {
            let slot = common.pipeline.slot(last);
            FrameInput::Direct {
                image: slot.input_image,
                view: slot.input_image_view,
                extent: vk::Extent2D {
                    width: common.aligned_width,
                    height: common.aligned_height,
                },
                ownership: SourceOwnership {
                    old_layout: vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
                    acquire_from: None,
                    release_to: None,
                    final_layout: vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
                },
            }
        } else {
            FrameInput::InPlace
        };
        // Encoded now, with the frames that wait before it: no frame arrives
        // after it to end their group, and its source may be another slot's
        // image.
        let mut future = self.inner.encode_frame_now(input, FrameSync::default())?;
        future.pts_offset = self.pts_offset;
        if other_slot {
            // The next frame may be written into that slot's image.
            self.context.wait_point(future.source_released())?;
        }
        Ok(future)
    }

    #[cfg_attr(not(feature = "dmabuf"), allow(dead_code))]
    pub(crate) fn inner(&self) -> &dyn EncoderApi {
        &*self.inner
    }

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
    ///
    /// The window does not depend on how many references the device can
    /// predict from per frame. A device that predicts from one or two still
    /// keeps the whole window, and recovers from the older frames in it.
    ///
    /// With B-frames, packets come in decode order, so the client lost every
    /// frame encoded after the lost one, also B-frames shown before it. The
    /// encoder drops every reference from the first frame it encoded that is
    /// shown at or after `first_lost_display_order`. The frames that wait for
    /// their group are encoded later from the references that survive. If
    /// none survives, the oldest frame that waits is encoded at once as the
    /// IDR.
    pub fn invalidate_reference_frames(&mut self, first_lost_display_order: u64) {
        let local = first_lost_display_order.saturating_sub(self.pts_offset);
        self.inner.invalidate_reference_frames(local)
    }

    /// Change what the frames are and what the stream says about its
    /// colours, together.
    ///
    /// Both decide the conversion, so they change in one call: there is no
    /// moment where the new source meets the old colour, a pair the
    /// conversion may not be able to do. A call that changes nothing does
    /// nothing, so it is cheap to make for every frame.
    ///
    /// A new colour makes the next frame an IDR/key frame carrying the new
    /// header. The conversion follows:
    ///
    /// - with [`Conversion::Shader`] kept, the colour converter switches in
    ///   place;
    /// - where the encoder converts, or the conversion changes, the codec
    ///   encoder is made again, because its video session fixes the input
    ///   format, matrix and range. Packet timestamps carry on across that.
    ///
    /// A `DmaBufImage` made before a change of source or conversion has to
    /// be made again. On error the encoder is left as it was.
    pub fn reconfigure(&mut self, source: Source, color: StreamColor) -> Result<()> {
        if (source, color) == (self.config.source, self.config.color) {
            return Ok(());
        }
        let mut config = self.config.clone();
        config.source = source;
        config.color = color;
        let next = plan(&self.context, &config)?;
        match (self.conversion, next.conversion, next.converter) {
            (Conversion::None, Conversion::None, _) => self.inner.set_color(color)?,
            (Conversion::Shader, Conversion::Shader, Some(next_converter))
                if self
                    .converter
                    .as_ref()
                    .is_some_and(|c| c.can_become(&next_converter)) =>
            {
                let converter = self
                    .converter
                    .as_mut()
                    .expect("a shader conversion has a converter");
                let old = converter.config().clone();
                converter.reconfigure(next_converter)?;
                if color != self.config.color
                    && let Err(e) = self.inner.set_color(color)
                {
                    let _ = converter.reconfigure(old);
                    return Err(e);
                }
                if source != self.config.source {
                    self.generation += 1;
                }
            }
            _ => self.rebuild(config.clone())?,
        }
        self.config = config;
        Ok(())
    }

    /// Make the codec encoder again for `config`, keeping the timestamps
    /// going.
    fn rebuild(&mut self, config: EncodeConfig) -> Result<()> {
        self.inner.flush()?;
        let encoded = self.inner.common().input_frame_num;
        let mut next = Self::new(self.context.clone(), config)?;
        next.pts_offset = self.pts_offset + encoded;
        next.id = self.id;
        // A slot taken before is not a slot of the new encoder.
        next.ticket = self.ticket + 1;
        next.generation =
            if (next.conversion, next.config.source) == (self.conversion, self.config.source) {
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
    rgb_input: Option<crate::converter::RgbFormat>,
) -> Result<Box<dyn EncoderApi>> {
    let context = context.clone();
    let config = config.clone();
    Ok(match config.codec {
        Codec::H264 => Box::new(super::h264::H264::create(context, config, rgb_input)?),
        Codec::H265 => Box::new(super::h265::H265::create(context, config, rgb_input)?),
        Codec::AV1 => Box::new(super::av1::Av1::create(context, config, rgb_input)?),
    })
}
