//! The H.265 Vulkan decode session: lazy stream-driven setup and per-picture
//! record/submit.

use std::collections::HashMap;

use ash::vk;
use ash::vk::TaggedStructure;
use tracing::debug;

use super::dpb::{DecodeDpb, PictureDecision, PictureState, RefPicSet, SkipReason};
use super::parser::{self, NalHeader, NalType, NalUnit, Vps};
use super::pps::{self, Pps};
use super::record::H265Record;
use super::slice::{self, SliceHeader};
use super::sps::{self, Sps};
use super::std_params::StdParameterSets;
use crate::decoder::DecodeConfig;
use crate::decoder::annexb::START_CODE;
use crate::decoder::common::DecoderCommon;
use crate::decoder::driver::{CodecDecoder, Decoded};
use crate::decoder::pipeline::DECODE_PIPELINE_DEPTH;
use crate::decoder::plan::StreamNeeds;
use crate::decoder::profile::DecodeProfile;
use crate::decoder::record::{DecodeRecord, LinkParams};
use crate::encoder::{BitDepth, ChromaFormat};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;

/// H.265 decoder.
///
/// Owns only what is specific to H.265: the parameter sets seen so far, and the
/// POC / reference picture set state. Everything Vulkan lives in
/// [`DecoderCommon`], and the flow around a picture (framing, keyframes,
/// reordering) is [`StreamDecoder`](crate::decoder::driver::StreamDecoder)'s.
pub(crate) struct H265Decoder {
    common: DecoderCommon,

    /// How many DPB slots to reserve beyond the stream's needs, so the caller
    /// can hold decoded frames while decoding continues.
    output_depth: usize,

    /// Parameter sets seen so far, by id.
    vps_map: HashMap<u8, Vps>,
    sps_map: HashMap<u8, Sps>,
    pps_map: HashMap<u8, Pps>,

    /// The parameter sets the current session parameters were built from, kept
    /// whole rather than by id: a stream may resend an id with new content,
    /// which is how a resolution change is signalled.
    active: Option<(Vps, Sps, Pps)>,

    /// POC and reference picture set state, recreated alongside the session.
    dpb: Option<DecodeDpb>,
    /// An end of sequence NAL unit arrived after the last picture, so the next
    /// one starts a new coded video sequence.
    pending_eos: bool,

    /// What each pipeline slot's decode command was recorded from, indexed like
    /// the pipeline's slots.
    records: Box<[DecodeRecord<H265Record>]>,

    /// `sps_max_num_reorder_pics` of the active SPS's highest sub-layer.
    reorder_depth: usize,
}

/// The slice segments making up one picture, gathered from the input.
pub(crate) struct Picture<'a> {
    nal: NalHeader,
    /// The header of the picture's first slice segment.
    header: SliceHeader,
    segments: Vec<&'a [u8]>,
    /// An end of sequence NAL unit came before this picture.
    after_eos: bool,
    /// The parameter sets in effect when this picture was grouped. Captured
    /// here for the same reason as in the H.264 decoder: a later parameter set
    /// in the same chunk must not reach back.
    vps: Vps,
    sps: Sps,
    pps: Pps,
}

impl H265Decoder {
    pub(crate) fn create(context: VideoContext, config: &DecodeConfig) -> Result<Self> {
        Ok(Self {
            common: DecoderCommon::new(context, config.consumer_queue_family)?,
            output_depth: config.output_depth,
            vps_map: HashMap::new(),
            sps_map: HashMap::new(),
            pps_map: HashMap::new(),
            active: None,
            dpb: None,
            pending_eos: false,
            // SAFETY: the picture info is plain C data, for which all-zero is
            // a valid (if meaningless) value.
            records: (0..DECODE_PIPELINE_DEPTH)
                .map(|_| DecodeRecord::new(unsafe { std::mem::zeroed() }))
                .collect(),
            reorder_depth: 0,
        })
    }

    /// The decode profile for a parsed SPS.
    fn profile_for(sps: &Sps) -> Result<(DecodeProfile, ChromaFormat, BitDepth)> {
        let chroma_format = match (sps.chroma_format_idc, sps.separate_colour_plane_flag) {
            (1, _) => ChromaFormat::Yuv420,
            (2, _) => ChromaFormat::Yuv422,
            (3, false) => ChromaFormat::Yuv444,
            (idc, separate) => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "H.265 decode: unsupported chroma_format_idc {idc} \
                     (separate colour planes: {separate})"
                )));
            }
        };
        let bit_depth = match (sps.bit_depth_luma_minus8, sps.bit_depth_chroma_minus8) {
            (0, 0) => BitDepth::Eight,
            (2, 2) => BitDepth::Ten,
            (luma, chroma) => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "H.265 decode: unsupported bit depth {}/{} (luma/chroma)",
                    luma + 8,
                    chroma + 8
                )));
            }
        };
        if sps.sps_scc_extension_flag || sps.sps_multilayer_extension_flag {
            return Err(PixelForgeError::InvalidInput(
                "H.265 decode: screen content and multi-layer extensions are not supported"
                    .to_string(),
            ));
        }
        // The Std profile enum uses the coded values.
        let profile_idc = sps.profile_tier_level.general_profile_idc as u32;
        let profile = DecodeProfile::h265(profile_idc, chroma_format, bit_depth);
        Ok((profile, chroma_format, bit_depth))
    }

    /// Create (or recreate) the Vulkan session for the given parameter sets.
    fn activate(&mut self, vps: &Vps, sps: &Sps, pps: &Pps) -> Result<()> {
        if self.common.session.is_some()
            && self
                .active
                .as_ref()
                .is_some_and(|(v, s, p)| v == vps && s == sps && p == pps)
        {
            return Ok(());
        }

        let (profile, chroma_format, bit_depth) = Self::profile_for(sps)?;
        // `sps_max_dec_pic_buffering` counts the current picture too.
        let references = sps.max_dec_pic_buffering().saturating_sub(1) as u32;
        let reorder_depth = sps.reorder_depth();
        let needs = StreamNeeds {
            codec: "H.265",
            width: sps.pic_width_in_luma_samples,
            height: sps.pic_height_in_luma_samples,
            chroma_format,
            bit_depth,
            references,
            active_references: references,
            slot_limit: super::dpb::MAX_DPB_SLOTS as u32,
            reorder_depth,
            output_depth: self.output_depth,
            film_grain: false,
        };
        let mut h265_caps = vk::VideoDecodeH265CapabilitiesKHR::default();
        let plan = self
            .common
            .plan_session(profile.info(), &mut h265_caps, &needs)?;
        let slot_count = plan.slot_count;
        self.common
            .create_session(&plan, profile, |common, session| {
                Self::build_session_params(common, session, vps, sps, pps)
            })?;

        self.active = Some((vps.clone(), sps.clone(), pps.clone()));
        self.reorder_depth = reorder_depth;
        let pins = self.common.session()?.slot_pins.clone();
        self.dpb = Some(DecodeDpb::new(slot_count, pins));
        Ok(())
    }

    /// Translate the parsed parameter sets into StdVideo structs and create
    /// session parameters.
    fn build_session_params(
        common: &DecoderCommon,
        session: vk::VideoSessionKHR,
        vps: &Vps,
        sps: &Sps,
        pps: &Pps,
    ) -> Result<vk::VideoSessionParametersKHR> {
        let sets = StdParameterSets::new(vps, sps, pps);
        let vps_array = [sets.vps];
        let sps_array = [sets.sps];
        let pps_array = [sets.pps];
        let add_info = vk::VideoDecodeH265SessionParametersAddInfoKHR::default()
            .std_vp_ss(&vps_array)
            .std_sp_ss(&sps_array)
            .std_pp_ss(&pps_array);
        let mut h265_create = vk::VideoDecodeH265SessionParametersCreateInfoKHR::default()
            .max_std_vps_count(1)
            .max_std_sps_count(1)
            .max_std_pps_count(1)
            .parameters_add_info(&add_info);
        let create_info = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(session)
            .push(&mut h265_create);

        unsafe {
            common
                .video_queue_fn
                .create_video_session_parameters(&create_info, None)
        }
        .map_err(PixelForgeError::vk("vkCreateVideoSessionParametersKHR"))
    }

    /// Group NAL units into pictures, absorbing parameter sets. Returns the
    /// pictures plus, if any slice was skipped for missing parameter sets, the
    /// reason a keyframe is needed.
    fn split_pictures<'a>(&mut self, data: &'a [u8]) -> Result<(Vec<Picture<'a>>, Option<String>)> {
        let mut pictures: Vec<Picture<'a>> = Vec::new();
        let mut awaiting: Option<String> = None;
        // Whether the slices of the picture in progress are being dropped,
        // because its first segment could not be used.
        let mut dropping = false;

        for nal in parser::iter_nal_units(data) {
            // Only the base layer is decoded.
            if nal.header.layer_id != 0 {
                continue;
            }
            match nal.nal_type() {
                NalType::Vps => {
                    let vps = parser::parse_vps(nal.payload())?;
                    self.vps_map.insert(vps.vps_video_parameter_set_id, vps);
                }
                NalType::Sps => {
                    let sps = sps::parse_sps(nal.payload())?;
                    debug!(
                        "H.265 SPS id={} {}x{} profile={}",
                        sps.sps_seq_parameter_set_id,
                        sps.pic_width_in_luma_samples,
                        sps.pic_height_in_luma_samples,
                        sps.profile_tier_level.general_profile_idc
                    );
                    self.sps_map.insert(sps.sps_seq_parameter_set_id, sps);
                }
                NalType::Pps => {
                    let pps = pps::parse_pps(nal.payload())?;
                    self.pps_map.insert(pps.pps_pic_parameter_set_id, pps);
                }
                NalType::Eos => self.pending_eos = true,
                t if t.is_vcl() && !matches!(t, NalType::Other(_)) => {
                    if slice::starts_picture(&nal) {
                        dropping = false;
                        match self.begin_picture(&nal)? {
                            Ok(picture) => pictures.push(picture),
                            Err(reason) => {
                                awaiting.get_or_insert(reason);
                                dropping = true;
                            }
                        }
                    } else if !dropping && let Some(picture) = pictures.last_mut() {
                        picture.segments.push(nal.data);
                    }
                }
                // SEI, AUD, and everything else is not needed by the driver.
                _ => {}
            }
        }
        Ok((pictures, awaiting))
    }

    /// Start a picture at its first slice segment. The inner `Err` is the
    /// reason it cannot be decoded: parameter sets that have not been seen.
    fn begin_picture<'a>(
        &mut self,
        nal: &NalUnit<'a>,
    ) -> Result<std::result::Result<Picture<'a>, String>> {
        let pps_id = slice::slice_pps_id(nal)?;
        let Some(pps) = self.pps_map.get(&pps_id) else {
            return Ok(Err(format!("PPS {pps_id} not yet received")));
        };
        let Some(sps) = self.sps_map.get(&pps.pps_seq_parameter_set_id) else {
            return Ok(Err(format!(
                "SPS {} not yet received",
                pps.pps_seq_parameter_set_id
            )));
        };
        let Some(vps) = self.vps_map.get(&sps.sps_video_parameter_set_id) else {
            return Ok(Err(format!(
                "VPS {} not yet received",
                sps.sps_video_parameter_set_id
            )));
        };
        let header = slice::parse_slice_header(nal, sps, pps)?;
        let after_eos = std::mem::take(&mut self.pending_eos);
        Ok(Ok(Picture {
            nal: nal.header,
            header,
            segments: vec![nal.data],
            after_eos,
            vps: vps.clone(),
            sps: sps.clone(),
            pps: pps.clone(),
        }))
    }

    /// Record and submit the decode of a single picture, without waiting for
    /// it.
    fn decode_picture(&mut self, picture: &Picture, pts: u64) -> Result<Decoded> {
        let sps = &picture.sps;
        let dpb = self.dpb.as_mut().expect("activated above");
        if picture.after_eos {
            dpb.end_of_sequence();
        }
        let state = match dpb.begin_picture(&picture.nal, &picture.header, sps)? {
            PictureDecision::Decode(state) => state,
            PictureDecision::Skip(SkipReason::RaslWithoutReferences) => {
                return Ok(Decoded::Skipped);
            }
            PictureDecision::Skip(SkipReason::AwaitingIrap) => {
                debug!("H.265 decode: picture before the first IRAP picture skipped");
                return Ok(Decoded::Skipped);
            }
        };
        if state.missing_reference {
            // A picture that predicts from one the decoder does not have
            // would decode to garbage, and so would everything after it that
            // predicts from it in turn.
            return Ok(Decoded::NeedsKeyframe(format!(
                "picture POC {} references a picture that is not in the DPB",
                state.poc
            )));
        }

        // Every slot can be busy because a frame the caller still holds pins
        // one; wait for them to drop it, as the encoder waits for a free slot.
        let slot = loop {
            let dpb = self.dpb.as_mut().expect("active");
            if let Some(slot) = dpb.try_allocate_slot() {
                break slot;
            }
            if !dpb.has_pinned_slots() {
                return Err(PixelForgeError::InvalidInput(
                    "H.265 decode: no free DPB slot (stream exceeds negotiated DPB size)"
                        .to_string(),
                ));
            }
            self.common.session()?.slot_pins.wait_for_release();
        };

        // Claim the pipeline slot before staging into its buffer; see the
        // H.264 decoder for why the order matters.
        self.common.begin_decode_commands()?;
        let (buffer_range, offsets) = self.common.stage_slices(&picture.segments, &START_CODE)?;
        self.common.record_barriers(slot);
        self.record_decode(picture, &state, slot, buffer_range, &offsets)?;
        self.common.submit_decode()?;

        let (width, height) = sps.display_dimensions();
        let frame =
            self.common
                .decoded_picture(slot, width, height, pts, state.poc, state.is_keyframe);
        self.dpb.as_mut().expect("active").end_picture(slot, &state);

        Ok(if state.pic_output_flag {
            Decoded::Shown(frame)
        } else {
            Decoded::Hidden
        })
    }

    /// Fill and link this pipeline slot's record, and record the coding
    /// commands from it.
    fn record_decode(
        &mut self,
        picture: &Picture,
        state: &PictureState,
        slot: u8,
        buffer_range: u64,
        offsets: &[u32],
    ) -> Result<()> {
        let active = self.common.session.as_ref().expect("active");

        // The slot's previous decode has finished (`begin_decode_commands`
        // waited for it), so its record can be rewritten.
        let record = &mut self.records[self.common.pipeline.current_index()];
        record.reset(offsets);

        // Every picture the DPB holds, so the driver keeps them all; the lists
        // in the picture info say which the current picture predicts from.
        for r in self.dpb.as_ref().expect("active").references() {
            // SAFETY: plain C data, for which all-zero is valid.
            let mut flags: ash::vk::native::StdVideoDecodeH265ReferenceInfoFlags =
                unsafe { std::mem::zeroed() };
            flags.set_used_for_long_term_reference(r.long_term as u32);
            record.push_reference(
                r.slot,
                active.dpb_entry(r.slot).2,
                None,
                ash::vk::native::StdVideoDecodeH265ReferenceInfo {
                    flags,
                    PicOrderCntVal: r.poc,
                },
            );
        }
        let setup_std = ash::vk::native::StdVideoDecodeH265ReferenceInfo {
            // SAFETY: plain C data, for which all-zero is valid.
            flags: unsafe { std::mem::zeroed() },
            PicOrderCntVal: state.poc,
        };

        *record.picture_mut() = picture_info(picture, state)?;
        record.link(&LinkParams {
            extent: vk::Extent2D {
                width: active.coded_width,
                height: active.coded_height,
            },
            session: active.session,
            session_params: active.session_params,
            bitstream: self.common.bitstream_buffer(),
            bitstream_range: buffer_range,
            setup_slot: slot,
            setup_view: active.dpb_entry(slot).2,
            output_view: active.output_image.map(|(_, _, view)| view),
            setup_std,
        });
        self.common.record_coding(record);
        Ok(())
    }
}

/// The Std picture info for a picture about to be decoded.
fn picture_info(
    picture: &Picture,
    state: &PictureState,
) -> Result<ash::vk::native::StdVideoDecodeH265PictureInfo> {
    let header = &picture.header;
    // SAFETY: plain C data, for which all-zero is valid.
    let mut flags: ash::vk::native::StdVideoDecodeH265PictureInfoFlags =
        unsafe { std::mem::zeroed() };
    flags.set_IrapPicFlag(state.nal_type.is_irap() as u32);
    flags.set_IdrPicFlag(state.nal_type.is_idr() as u32);
    flags.set_IsReference(state.is_reference as u32);
    flags.set_short_term_ref_pic_set_sps_flag(header.short_term_ref_pic_set_sps_flag as u32);
    Ok(ash::vk::native::StdVideoDecodeH265PictureInfo {
        flags,
        sps_video_parameter_set_id: picture.sps.sps_video_parameter_set_id,
        pps_seq_parameter_set_id: picture.pps.pps_seq_parameter_set_id,
        pps_pic_parameter_set_id: picture.pps.pps_pic_parameter_set_id,
        NumDeltaPocsOfRefRpsIdx: header.num_delta_pocs_of_ref_rps_idx(&picture.sps),
        PicOrderCntVal: state.poc,
        NumBitsForSTRefPicSetInSlice: header.num_bits_for_st_ref_pic_set as u16,
        reserved: 0,
        RefPicSetStCurrBefore: RefPicSet::vulkan_list(&state.rps.st_curr_before)?,
        RefPicSetStCurrAfter: RefPicSet::vulkan_list(&state.rps.st_curr_after)?,
        RefPicSetLtCurr: RefPicSet::vulkan_list(&state.rps.lt_curr)?,
    })
}

impl CodecDecoder for H265Decoder {
    type Unit<'a> = Picture<'a>;

    const NAME: &'static str = "H.265";

    fn common(&self) -> &DecoderCommon {
        &self.common
    }

    fn common_mut(&mut self) -> &mut DecoderCommon {
        &mut self.common
    }

    fn complete_prefix(buffer: &[u8]) -> usize {
        super::complete_prefix(buffer)
    }

    fn split_units<'a>(&mut self, data: &'a [u8]) -> Result<(Vec<Picture<'a>>, Option<String>)> {
        self.split_pictures(data)
    }

    fn is_keyframe(picture: &Picture<'_>) -> bool {
        picture.nal.nal_type.is_irap()
    }

    fn decode_unit(&mut self, picture: &Picture<'_>, pts: u64) -> Result<Decoded> {
        // (Re)build the session if the stream's parameter sets changed.
        self.activate(&picture.vps, &picture.sps, &picture.pps)?;
        self.decode_picture(picture, pts)
    }

    fn reorder_depth(&self) -> usize {
        self.reorder_depth
    }
}
