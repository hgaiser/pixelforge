//! The decode flow every codec shares: framing the input, waiting for a
//! keyframe, reordering into display order, and the end of the stream.
//!
//! A codec implements [`CodecDecoder`]: how to split its bitstream into coded
//! pictures and how to decode one. [`StreamDecoder`] drives it and implements
//! the codec-erased [`DecoderApi`](crate::decoder::DecoderApi) once for all
//! codecs.

use ash::vk;
use tracing::debug;

use crate::decoder::common::DecoderCommon;
use crate::decoder::frames::{DecodedPicture, ReorderBuffer};
use crate::decoder::{DecodeConfig, DecodeStatus, DecoderApi, FrameReceiver, Framing};
use crate::error::Result;

/// What a codec does; everything else is [`StreamDecoder`]'s.
pub(crate) trait CodecDecoder: Send {
    /// One coded picture, as [`Self::split_units`] gathers it from the input.
    type Unit<'a>;

    /// The codec's name, for messages: "H.264".
    const NAME: &'static str;

    /// The shared Vulkan state, which the codec owns.
    fn common(&self) -> &DecoderCommon;
    fn common_mut(&mut self) -> &mut DecoderCommon;

    /// How many leading bytes of `buffer` hold coded pictures known to be
    /// complete.
    ///
    /// A coded picture is only known to have ended once the next one starts,
    /// so everything from the last picture start onward may still be waiting
    /// for more data. Zero means nothing in `buffer` can be decoded yet.
    fn complete_prefix(buffer: &[u8]) -> usize;

    /// Split `data`, which ends on a picture boundary, into coded pictures,
    /// absorbing any parameter sets on the way.
    ///
    /// Returns the pictures plus, if some data was skipped because state it
    /// needs has not been seen (parameter sets sent before the stream was
    /// joined), the reason a keyframe is needed.
    fn split_units<'a>(&mut self, data: &'a [u8]) -> Result<(Vec<Self::Unit<'a>>, Option<String>)>;

    /// Whether decoding can start at `unit`.
    fn is_keyframe(unit: &Self::Unit<'_>) -> bool;

    /// Decode one coded picture.
    fn decode_unit(&mut self, unit: &Self::Unit<'_>, pts: u64) -> Result<Decoded>;

    /// How many pictures the active stream may hold back for display-order
    /// reordering.
    fn reorder_depth(&self) -> usize;
}

/// What [`CodecDecoder::decode_unit`] did with a coded picture.
pub(crate) enum Decoded {
    /// Decoded, and to be output.
    Shown(DecodedPicture),
    /// Decoded, but not output: an AV1 frame with `show_frame` 0, or an H.265
    /// picture with `pic_output_flag` 0.
    Hidden,
    /// Not decoded, and that is normal: an H.265 RASL picture after the CRA
    /// that started decoding, which references pictures from before it.
    Skipped,
    /// Not decoded, because it references state the decoder does not have:
    /// pictures or parameter sets from before the stream was joined, or lost
    /// since. The reason names what is missing.
    NeedsKeyframe(String),
}

/// Drives a [`CodecDecoder`].
pub(crate) struct StreamDecoder<C: CodecDecoder> {
    codec: C,

    /// Display-order reordering, which holds pictures by pinning their DPB
    /// slots rather than by copying them out.
    reorder: ReorderBuffer,

    /// How input is framed, which decides whether the end of a `decode` call is
    /// also the end of a coded picture.
    framing: Framing,
    /// Bytes of a coded picture that has not been seen to end yet, held over
    /// from earlier `decode` calls. Only ever non-empty in
    /// [`Framing::ByteStream`].
    carry: Vec<u8>,
    /// The `pts` of the call that delivered the first byte of `carry`, so a
    /// picture is timestamped by when it started arriving rather than by which
    /// call happened to complete it.
    carry_pts: u64,

    /// Whether decoding is waiting for a keyframe before it can produce output.
    /// True until the first keyframe is decoded: other pictures reference state
    /// that does not exist yet, so they are skipped and the caller is told to
    /// request a keyframe. This is the normal state when joining a stream.
    awaiting_keyframe: bool,
}

impl<C: CodecDecoder> StreamDecoder<C> {
    pub(crate) fn new(codec: C, config: &DecodeConfig) -> Self {
        Self {
            codec,
            reorder: ReorderBuffer::new(),
            framing: config.framing,
            carry: Vec::new(),
            carry_pts: 0,
            awaiting_keyframe: true,
        }
    }

    /// Framing for input that may cut anywhere.
    ///
    /// A coded picture is only known to have ended once the next one starts, so
    /// everything up to the last picture start is complete and decodable, and
    /// everything from it onward is held over for the next call. `finish` ends
    /// the final picture.
    fn decode_byte_stream(&mut self, data: &[u8], pts: u64) -> Result<DecodeStatus> {
        let carried = self.carry.len();
        let mut buffer = std::mem::take(&mut self.carry);
        buffer.extend_from_slice(data);

        // Nothing before the last picture start can still be waiting for more
        // data. With no picture start at all, nothing is complete yet.
        let split = C::complete_prefix(&buffer);

        let result = if split > 0 {
            self.decode_frames(&buffer[..split], self.carry_pts)
        } else {
            // Not enough bytes to complete a picture yet.
            Ok(DecodeStatus::Buffered)
        };

        // The held-over picture is timestamped by the call its first byte came
        // in on, which is this one only if the split consumed everything the
        // previous calls had left.
        if split >= carried {
            self.carry_pts = pts;
        }
        buffer.drain(..split);
        self.carry = buffer;
        result
    }

    /// Decode input already known to end on a picture boundary.
    ///
    /// Frames go to the consumer as each picture's work is handed to the
    /// completion thread, so a call decoding many pictures does not hold their
    /// frames until it returns.
    fn decode_frames(&mut self, data: &[u8], pts: u64) -> Result<DecodeStatus> {
        // What is missing when a picture cannot be decoded yet; set only while
        // no frame has been produced, so a keyframe later in the same buffer
        // still yields output. Seeded with data skipped during splitting.
        let (units, mut awaiting) = self.codec.split_units(data)?;
        // Whether any picture actually reached the GPU, which decides between
        // reporting a decode and asking for a keyframe.
        let mut decoded_any = false;

        for unit in units {
            let is_keyframe = C::is_keyframe(&unit);

            // A picture before the first keyframe references pictures and
            // state that do not exist yet. Skip it and ask for a keyframe
            // rather than decoding garbage.
            if self.awaiting_keyframe && !is_keyframe {
                awaiting.get_or_insert_with(|| {
                    "no keyframe decoded yet; picture cannot be decoded".to_string()
                });
                continue;
            }

            let ready = match self.codec.decode_unit(&unit, pts)? {
                Decoded::Shown(picture) => {
                    let depth = self.codec.reorder_depth();
                    self.reorder
                        .push(self.codec.common_mut(), &picture, depth)?
                }
                Decoded::Hidden => Vec::new(),
                Decoded::Skipped => continue,
                Decoded::NeedsKeyframe(reason) => {
                    awaiting.get_or_insert(reason);
                    continue;
                }
            };
            decoded_any = true;
            // A decoded keyframe re-establishes a valid decode point.
            if is_keyframe {
                self.awaiting_keyframe = false;
            }
            // The picture's submissions are complete as far as recording goes;
            // hand its slot and frames to the completion thread and move on to
            // the next slot.
            self.codec.common_mut().end_picture(ready);
        }

        if decoded_any {
            return Ok(DecodeStatus::Decoded);
        }
        // Only ask for a keyframe when nothing was decoded: if one arrived
        // later in the same buffer, its pictures take priority.
        if let Some(reason) = awaiting {
            debug!("{} decode: awaiting keyframe: {}", C::NAME, reason);
            return Ok(DecodeStatus::NeedsKeyframe);
        }
        Ok(DecodeStatus::Buffered)
    }
}

impl<C: CodecDecoder> DecoderApi for StreamDecoder<C> {
    fn decode(&mut self, data: &[u8], pts: u64) -> Result<DecodeStatus> {
        match self.framing {
            // The caller guarantees the chunk ends on a picture boundary, so it
            // can go straight to the decoder with no buffering and no copy.
            Framing::FrameAligned => self.decode_frames(data, pts),
            Framing::ByteStream => self.decode_byte_stream(data, pts),
        }
    }

    /// Decode what is still buffered and emit the pictures held for
    /// reordering. Call at end of stream.
    fn finish(&mut self) -> Result<()> {
        // In byte-stream framing the tail is a picture that was never seen to
        // end because nothing followed it. End of stream is what ends it.
        if !self.carry.is_empty() {
            let tail = std::mem::take(&mut self.carry);
            let pts = self.carry_pts;
            self.decode_frames(&tail, pts)?;
        }
        let frames = self.reorder.flush();
        self.codec.common_mut().finish_stream(frames);
        Ok(())
    }

    fn take_frame_receiver(&mut self) -> Option<FrameReceiver> {
        self.codec.common_mut().frames_rx.take()
    }

    fn picture_format(&self) -> Option<vk::Format> {
        self.codec
            .common()
            .session
            .as_ref()
            .map(|s| s.picture_format)
    }
}

impl<C: CodecDecoder> Drop for StreamDecoder<C> {
    fn drop(&mut self) {
        // The reorder pool's images must be freed while the codec's
        // `DecoderCommon` (and its device) is still alive. Fields drop after
        // this, so the codec goes afterwards.
        self.reorder.destroy(self.codec.common());
    }
}
