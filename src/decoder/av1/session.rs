//! The AV1 Vulkan decode session: lazy stream-driven setup and per-frame
//! record/submit.
//!
//! One unit of input is a temporal unit, which shows exactly one frame but can
//! decode more than one: a hidden frame (`show_frame` 0) that a later
//! `show_existing_frame` shows, followed by the frame shown now. Each frame is
//! parsed only once the one before it is committed, since its header depends
//! on the reference state that frame leaves behind.

use ash::vk;
use ash::vk::TaggedStructure;
use tracing::debug;

use super::frame_header::FrameType;
use super::obu::{ObuType, iter_obus, temporal_unit_starts};
use super::parser::{Av1Event, Av1Frame, Av1Parser};
use super::record::{Av1Picture, Av1Record};
use super::sequence::SequenceHeader;
use super::std_info::{
    StdPictureInfo, StdSequenceHeader, reference_name_slot_indices, std_reference_info,
};
use crate::decoder::DecodeConfig;
use crate::decoder::common::DecoderCommon;
use crate::decoder::driver::{CodecDecoder, Decoded};
use crate::decoder::frames::DecodedPicture;
use crate::decoder::pipeline::DECODE_PIPELINE_DEPTH;
use crate::decoder::plan::StreamNeeds;
use crate::decoder::profile::DecodeProfile;
use crate::decoder::record::{DecodeRecord, LinkParams};
use crate::encoder::{BitDepth, ChromaFormat};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;

/// The most DPB slots the AV1 decoder uses: the 8 reference frames, the frame
/// being decoded, and spare slots for frames held for output.
const MAX_DPB_SLOTS: u32 = 17;

/// AV1 decoder.
///
/// Owns only what is specific to AV1: the OBU parser with its reference state.
/// Everything Vulkan lives in [`DecoderCommon`].
pub(crate) struct Av1Decoder {
    common: DecoderCommon,

    /// How many DPB slots to reserve beyond the stream's needs, so the caller
    /// can hold decoded frames while decoding continues.
    output_depth: usize,

    parser: Av1Parser,
    /// The sequence header the current session was built from.
    active: Option<SequenceHeader>,
    /// The Std sequence header the current session parameters were created
    /// from, kept for as long as they exist.
    ///
    /// NVIDIA (615.78.08) does not copy it when the parameters are created:
    /// it reads it again at every decode. Freed, its memory is reused, and the
    /// decode reads whatever is there now. Nothing fails; the chroma of every
    /// inter frame just comes out slightly wrong, which is how this was found.
    /// Boxed, so moving the decoder does not move it.
    std_sequence: Option<Box<StdSequenceHeader>>,
    /// DPB slots in the current session.
    slot_count: usize,
    /// Which DPB slots hold a frame decoded with film grain, whose output is
    /// in the slot's grain image. A `show_existing_frame` of it shows that
    /// image again.
    grain_applied: [bool; MAX_DPB_SLOTS as usize],

    /// Shown frames since the last key frame, which is the display order: AV1
    /// shows frames in the order the stream gives them.
    shown_since_key: i32,

    /// What each pipeline slot's decode command was recorded from, indexed like
    /// the pipeline's slots.
    records: Box<[DecodeRecord<Av1Record>]>,
}

impl Av1Decoder {
    pub(crate) fn create(context: VideoContext, config: &DecodeConfig) -> Result<Self> {
        Ok(Self {
            common: DecoderCommon::new(context, config.consumer_queue_family)?,
            output_depth: config.output_depth,
            parser: Av1Parser::new(),
            active: None,
            std_sequence: None,
            slot_count: 0,
            grain_applied: [false; MAX_DPB_SLOTS as usize],
            shown_since_key: 0,
            records: (0..DECODE_PIPELINE_DEPTH)
                .map(|_| DecodeRecord::new(Av1Picture::default()))
                .collect(),
        })
    }

    /// The decode profile for a sequence header.
    fn profile_for(seq: &SequenceHeader) -> Result<(DecodeProfile, ChromaFormat, BitDepth)> {
        let cc = &seq.color_config;
        if cc.mono_chrome {
            return Err(PixelForgeError::InvalidInput(
                "AV1 decode: monochrome streams are not supported".to_string(),
            ));
        }
        let chroma_format = match (cc.subsampling_x, cc.subsampling_y) {
            (true, true) => ChromaFormat::Yuv420,
            (true, false) => ChromaFormat::Yuv422,
            (false, false) => ChromaFormat::Yuv444,
            (false, true) => {
                return Err(PixelForgeError::InvalidInput(
                    "AV1 decode: invalid chroma subsampling".to_string(),
                ));
            }
        };
        let bit_depth = match cc.bit_depth {
            8 => BitDepth::Eight,
            10 => BitDepth::Ten,
            other => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "AV1 decode: unsupported bit depth {other}"
                )));
            }
        };
        // A session that may apply film grain has to say so in its profile.
        let profile = DecodeProfile::av1(
            seq.seq_profile as ash::vk::native::StdVideoAV1Profile,
            seq.film_grain_params_present,
            chroma_format,
            bit_depth,
        );
        Ok((profile, chroma_format, bit_depth))
    }

    /// Create (or recreate) the Vulkan session for the sequence header in
    /// effect.
    fn activate(&mut self) -> Result<()> {
        let seq = self
            .parser
            .sequence_header()
            .expect("a frame implies a sequence header")
            .clone();
        if self.common.session.is_some() && self.active.as_ref() == Some(&seq) {
            return Ok(());
        }

        let (profile, chroma_format, bit_depth) = Self::profile_for(&seq)?;
        let needs = StreamNeeds {
            codec: "AV1",
            width: seq.max_frame_width(),
            height: seq.max_frame_height(),
            chroma_format,
            bit_depth,
            references: super::consts::NUM_REF_FRAMES as u32,
            active_references: super::consts::REFS_PER_FRAME as u32,
            slot_limit: MAX_DPB_SLOTS,
            // Frames are shown in the order the stream gives them.
            reorder_depth: 0,
            output_depth: self.output_depth,
            film_grain: seq.film_grain_params_present,
        };
        let mut av1_caps = vk::VideoDecodeAV1CapabilitiesKHR::default();
        let plan = self
            .common
            .plan_session(profile.info(), &mut av1_caps, &needs)?;
        if !plan.coincide {
            // A frame shown again by `show_existing_frame` is read from its DPB
            // slot, which only a coincident DPB hands out.
            return Err(PixelForgeError::NoSuitableDevice(
                "AV1 decode: needs a device whose DPB and output images coincide".to_string(),
            ));
        }
        self.slot_count = plan.slot_count;
        let std_seq = Box::new(StdSequenceHeader::new(&seq));
        self.common
            .create_session(&plan, profile, |common, session| {
                Self::build_session_params(common, session, &std_seq)
            })?;
        // The old session parameters are gone by now, so the header they
        // pointed at can go too.
        self.std_sequence = Some(std_seq);
        self.active = Some(seq);
        Ok(())
    }

    fn build_session_params(
        common: &DecoderCommon,
        session: vk::VideoSessionKHR,
        std_seq: &StdSequenceHeader,
    ) -> Result<vk::VideoSessionParametersKHR> {
        let mut av1_create = vk::VideoDecodeAV1SessionParametersCreateInfoKHR::default()
            .std_sequence_header(std_seq.get());
        let create_info = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(session)
            .push(&mut av1_create);
        unsafe {
            common
                .video_queue_fn
                .create_video_session_parameters(&create_info, None)
        }
        .map_err(PixelForgeError::vk("vkCreateVideoSessionParametersKHR"))
    }

    /// Decode every frame of one temporal unit, returning the one it shows.
    fn decode_temporal_unit(&mut self, tu: &[u8], pts: u64) -> Result<Decoded> {
        let mut outcome = Decoded::Hidden;
        let mut missing: Option<String> = None;
        for obu in iter_obus(tu) {
            let obu = obu?;
            // Frame data from before the first sequence header belongs to a
            // stream joined mid-way.
            let is_frame_data = matches!(
                obu.header.obu_type,
                ObuType::Frame
                    | ObuType::FrameHeader
                    | ObuType::RedundantFrameHeader
                    | ObuType::TileGroup
            );
            if is_frame_data && self.parser.sequence_header().is_none() {
                missing.get_or_insert_with(|| "no sequence header received yet".to_string());
                continue;
            }
            match self.parser.parse_obu(&obu)? {
                None | Some(Av1Event::SequenceHeader) => {}
                Some(Av1Event::MissingReferences(slots)) => {
                    missing.get_or_insert_with(|| {
                        format!("frame references empty slots (mask {slots:#04x})")
                    });
                }
                Some(Av1Event::Frame(frame)) => {
                    // The frame shown by the previous one has already been
                    // handed over; close off its pipeline slot first.
                    if let Decoded::Shown(_) = outcome {
                        return Err(PixelForgeError::InvalidInput(
                            "AV1 decode: temporal unit shows more than one frame".to_string(),
                        ));
                    }
                    let result = self.decode_frame(tu, &frame, pts);
                    match result {
                        Ok(Some(picture)) => outcome = Decoded::Shown(picture),
                        Ok(None) => {
                            // A hidden frame: its work is complete as far as
                            // recording goes.
                            self.common.end_picture(Vec::new());
                        }
                        Err(e) => {
                            self.parser.discard_frame();
                            return Err(e);
                        }
                    }
                }
            }
        }
        match (outcome, missing) {
            (Decoded::Hidden, Some(reason)) => Ok(Decoded::NeedsKeyframe(reason)),
            (outcome, _) => Ok(outcome),
        }
    }

    /// Decode (or show) one frame and commit it to the reference state.
    /// Returns the picture to output when the frame is shown.
    fn decode_frame(
        &mut self,
        tu: &[u8],
        frame: &Av1Frame,
        pts: u64,
    ) -> Result<Option<DecodedPicture>> {
        let header = &frame.header;
        self.activate()?;

        if header.show_existing_frame {
            let shown = self.parser.refs().slots[header.frame_to_show_map_idx as usize].clone();
            let slot = shown.dpb_slot.ok_or_else(|| {
                PixelForgeError::InvalidInput(
                    "AV1 decode: show_existing_frame of a slot with no picture".to_string(),
                )
            })?;
            let is_key = shown.frame_type == FrameType::Key;
            if is_key {
                self.shown_since_key = 0;
            }
            let mut picture = self.common.decoded_picture(
                slot,
                shown.upscaled_width,
                shown.frame_height,
                pts,
                self.shown_since_key,
                is_key,
            );
            if self.grain_applied[slot as usize] {
                self.common.use_grain_output(&mut picture);
            }
            self.shown_since_key += 1;
            self.parser.commit_frame(None)?;
            return Ok(Some(picture));
        }

        let slot = self.allocate_slot()?;
        self.common.begin_decode_commands()?;
        // Only the tile data goes to the driver, each tile at the offset the
        // staging gives it.
        let tile_data: Vec<&[u8]> = frame
            .tiles()
            .map(|t| &tu[t.offset..t.offset + t.size])
            .collect();
        let (buffer_range, offsets) = self.common.stage_slices(&tile_data, &[])?;
        let grain = header.film_grain.apply_grain;
        self.common.record_barriers(slot);
        if grain {
            self.common.record_grain_barrier(slot);
        }
        self.record_decode(frame, slot, buffer_range, &offsets, grain)?;
        self.common.submit_decode()?;
        self.grain_applied[slot as usize] = grain;

        let size = &header.frame_size;
        let is_key = header.frame_type == FrameType::Key;
        if is_key && header.show_frame {
            self.shown_since_key = 0;
        }
        let mut picture = self.common.decoded_picture(
            slot,
            size.upscaled_width,
            size.frame_height,
            pts,
            self.shown_since_key,
            is_key && header.show_frame,
        );
        if grain {
            self.common.use_grain_output(&mut picture);
        }
        self.parser.commit_frame(Some(slot))?;
        if !header.show_frame {
            return Ok(None);
        }
        self.shown_since_key += 1;
        Ok(Some(picture))
    }

    /// A DPB slot no reference frame holds and no caller's frame pins.
    fn allocate_slot(&self) -> Result<u8> {
        loop {
            let pins = &self.common.session()?.slot_pins;
            let refs = self.parser.refs();
            if let Some(slot) =
                (0..self.slot_count as u8).find(|&s| !refs.dpb_slot_in_use(s) && !pins.is_pinned(s))
            {
                return Ok(slot);
            }
            if !pins.any_pinned() {
                return Err(PixelForgeError::InvalidInput(
                    "AV1 decode: no free DPB slot".to_string(),
                ));
            }
            pins.wait_for_release();
        }
    }

    /// Fill and link this pipeline slot's record, and record the coding
    /// commands from it.
    fn record_decode(
        &mut self,
        frame: &Av1Frame,
        slot: u8,
        buffer_range: u64,
        offsets: &[u32],
        grain: bool,
    ) -> Result<()> {
        let header = &frame.header;
        let active = self.common.session.as_ref().expect("active");
        let refs = self.parser.refs();

        let tiles: Vec<_> = frame.tiles().collect();
        let record = &mut self.records[self.common.pipeline.current_index()];
        record.reset(offsets);

        // Every picture the frame references, once each: several reference
        // names can point at the same picture.
        let mut pushed = [false; MAX_DPB_SLOTS as usize];
        let named: Vec<usize> = if header.frame_is_intra() {
            Vec::new()
        } else {
            header.ref_frame_idx.iter().map(|&i| i as usize).collect()
        };
        for r in named.iter().map(|&i| &refs.slots[i]).filter(|r| r.valid) {
            let Some(dpb_slot) = r.dpb_slot else { continue };
            if std::mem::replace(&mut pushed[dpb_slot as usize], true) {
                continue;
            }
            record.push_reference(
                dpb_slot,
                active.dpb_entry(dpb_slot).2,
                Some(vk::Extent2D {
                    width: r.upscaled_width,
                    height: r.frame_height,
                }),
                std_reference_info(r),
            );
        }

        // What the slot set up here will hold once this frame is a reference.
        // SAFETY: plain C data, for which all-zero is valid.
        let mut setup_std: ash::vk::native::StdVideoDecodeAV1ReferenceInfo =
            unsafe { std::mem::zeroed() };
        setup_std
            .flags
            .set_disable_frame_end_update_cdf(header.disable_frame_end_update_cdf as u32);
        setup_std
            .flags
            .set_segmentation_enabled(header.segmentation.segmentation_enabled as u32);
        setup_std.frame_type = header.frame_type as u8;
        setup_std.RefFrameSignBias = header.ref_frame_sign_bias_mask();
        setup_std.OrderHint = header.order_hint;
        setup_std.SavedOrderHints = header.order_hints;

        let picture = record.picture_mut();
        picture.std = Some(StdPictureInfo::new(header));
        picture.reference_name_slot_indices = reference_name_slot_indices(header, &refs.slots);
        // The buffer holds tile data alone, as ffmpeg hands it over, so there is
        // no header in it to point at.
        picture.frame_header_offset = 0;
        picture.tile_sizes.clear();
        picture
            .tile_sizes
            .extend(tiles.iter().map(|t| t.size as u32));

        let size = &header.frame_size;
        record.link(&LinkParams {
            extent: vk::Extent2D {
                width: size.upscaled_width,
                height: size.frame_height,
            },
            session: active.session,
            session_params: active.session_params,
            bitstream: self.common.bitstream_buffer(),
            bitstream_range: buffer_range,
            setup_slot: slot,
            setup_view: active.dpb_entry(slot).2,
            // With grain, the output must not be the reference picture, which
            // stays grain-free.
            output_view: if grain {
                Some(active.grain_view(slot))
            } else {
                active.output_image.map(|(_, _, view)| view)
            },
            setup_std,
        });
        self.common.record_coding(record);
        Ok(())
    }
}

impl CodecDecoder for Av1Decoder {
    type Unit<'a> = &'a [u8];

    const NAME: &'static str = "AV1";

    fn common(&self) -> &DecoderCommon {
        &self.common
    }

    fn common_mut(&mut self) -> &mut DecoderCommon {
        &mut self.common
    }

    fn complete_prefix(buffer: &[u8]) -> usize {
        // An error here is input that is not low-overhead OBUs; decoding the
        // buffer reports it.
        temporal_unit_starts(buffer)
            .ok()
            .and_then(|starts| starts.last().copied())
            .unwrap_or(0)
    }

    fn split_units<'a>(&mut self, data: &'a [u8]) -> Result<(Vec<&'a [u8]>, Option<String>)> {
        let mut starts = temporal_unit_starts(data)?;
        // Data before the first temporal delimiter is a temporal unit too,
        // when the caller strips delimiters (as some containers do).
        if starts.first() != Some(&0) {
            starts.insert(0, 0);
        }
        let units = starts
            .iter()
            .zip(starts.iter().skip(1).chain([&data.len()]))
            .map(|(&start, &end)| &data[start..end])
            .filter(|unit| !unit.is_empty())
            .collect();
        Ok((units, None))
    }

    /// AV1 tracks which reference slots hold what itself and reports a frame
    /// it cannot decode, so every temporal unit is offered.
    fn is_keyframe(_unit: &&[u8]) -> bool {
        true
    }

    fn decode_unit(&mut self, unit: &&[u8], pts: u64) -> Result<Decoded> {
        let outcome = self.decode_temporal_unit(unit, pts)?;
        if let Decoded::NeedsKeyframe(reason) = &outcome {
            debug!("AV1 decode: {reason}");
        }
        Ok(outcome)
    }

    fn reorder_depth(&self) -> usize {
        0
    }
}
