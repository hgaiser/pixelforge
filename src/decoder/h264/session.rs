//! The H.264 Vulkan decode session: lazy stream-driven setup, per-picture
//! record/submit, and CPU readback.

use std::collections::HashMap;

use ash::vk;
use ash::vk::TaggedStructure;
use tracing::debug;

use crate::decoder::DecodeConfig;
use crate::decoder::annexb::START_CODE;
use crate::decoder::common::DecoderCommon;
use crate::decoder::driver::{CodecDecoder, Decoded};
use crate::decoder::frames::DecodedPicture;
use crate::decoder::h264::dpb::{DecodeDpb, PictureState};
use crate::decoder::h264::parser::{
    self, NalType, NalUnit, Pps, SliceHeader, SliceType, Sps, iter_nal_units,
};
use crate::decoder::h264::record::H264Record;
use crate::decoder::pipeline::DECODE_PIPELINE_DEPTH;
use crate::decoder::plan::StreamNeeds;
use crate::decoder::profile::DecodeProfile;
use crate::decoder::record::{DecodeRecord, LinkParams};
use crate::encoder::{BitDepth, ChromaFormat};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;

/// H.264 decoder.
///
/// Owns only what is specific to H.264: the parameter sets seen so far, and the
/// POC / reference-marking state. Everything Vulkan lives in [`DecoderCommon`],
/// and the flow around a picture (framing, keyframes, reordering) is
/// [`StreamDecoder`](crate::decoder::driver::StreamDecoder)'s.
pub(crate) struct H264Decoder {
    common: DecoderCommon,

    /// How many DPB slots to reserve beyond the stream's needs, so the caller
    /// can hold decoded frames while decoding continues.
    output_depth: usize,

    /// Parameter sets seen so far, by id.
    sps_map: HashMap<u8, Sps>,
    pps_map: HashMap<u8, Pps>,

    /// The SPS/PPS the current session parameters were built from, kept whole
    /// rather than by id so that a stream reusing an id for different content,
    /// which is how a resolution change is signalled, is seen as the change it
    /// is.
    active_sps: Option<Sps>,
    active_pps: Option<Pps>,

    /// POC and reference state, recreated alongside the session.
    dpb: Option<DecodeDpb>,

    /// What each pipeline slot's decode command was recorded from, indexed like
    /// the pipeline's slots. Never resized or moved out of: the records point
    /// into themselves.
    records: Box<[DecodeRecord<H264Record>]>,

    /// `max_num_reorder_frames` for the active SPS: how deep the reorder buffer
    /// may hold pictures before emitting. Set by `activate`, which also sizes
    /// the DPB so that many pictures can be pinned rather than copied.
    reorder_depth: usize,
}

/// What `group_slices` did with a slice.
enum Grouped {
    /// Attached to a picture, new or existing.
    Added,
    /// Skipped: the parameter sets it needs have not been seen, so the stream
    /// was joined after they were sent. The reason names which.
    NeedsParameterSets(String),
}

/// The slices making up one picture, gathered from the input.
pub(crate) struct Picture<'a> {
    header: SliceHeader,
    nal_type: NalType,
    ref_idc: u8,
    is_intra: bool,
    slices: Vec<&'a [u8]>,
    /// The parameter sets in effect when this picture was grouped.
    ///
    /// Captured here rather than looked up later, because a chunk can carry a
    /// parameter set *change*: an SPS reusing an id it has already used, which
    /// is how a stream signals a new resolution. Resolving by id after the
    /// whole chunk has been scanned would decode every picture in it against
    /// the last SPS seen, so a resolution change would apply retroactively to
    /// the pictures before it.
    sps: Sps,
    pps: Pps,
}

impl H264Decoder {
    pub(crate) fn create(context: VideoContext, config: &DecodeConfig) -> Result<Self> {
        Ok(Self {
            common: DecoderCommon::new(context, config.consumer_queue_family)?,
            output_depth: config.output_depth,
            sps_map: HashMap::new(),
            pps_map: HashMap::new(),
            active_sps: None,
            active_pps: None,
            dpb: None,
            // SAFETY: the picture info is plain C data, for which all-zero is
            // a valid (if meaningless) value.
            records: (0..DECODE_PIPELINE_DEPTH)
                .map(|_| DecodeRecord::new(unsafe { std::mem::zeroed() }))
                .collect(),
            reorder_depth: 0,
        })
    }

    /// Build the H.264 decode profile for a parsed SPS.
    fn profile_for(sps: &Sps) -> Result<(DecodeProfile, ChromaFormat, BitDepth)> {
        let chroma_format = match sps.chroma_format_idc {
            1 => ChromaFormat::Yuv420,
            3 => ChromaFormat::Yuv444,
            other => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "H.264 decode: unsupported chroma_format_idc {}",
                    other
                )));
            }
        };
        let bit_depth = match sps.bit_depth_luma_minus8 {
            0 => BitDepth::Eight,
            2 => BitDepth::Ten,
            other => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "H.264 decode: unsupported luma bit depth {}",
                    other + 8
                )));
            }
        };
        if sps.seq_scaling_matrix_present_flag {
            return Err(PixelForgeError::InvalidInput(
                "H.264 decode: streams with explicit scaling matrices are not supported"
                    .to_string(),
            ));
        }

        let std_profile = match sps.profile_idc {
            66 => ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_BASELINE,
            77 => ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN,
            100 => ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH,
            244 => {
                ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE
            }
            other => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "H.264 decode: unsupported profile_idc {}",
                    other
                )));
            }
        };

        let profile = DecodeProfile::h264(std_profile, chroma_format, bit_depth);
        Ok((profile, chroma_format, bit_depth))
    }

    /// Create (or recreate) the Vulkan session for the given parameter sets.
    fn activate(&mut self, sps: &Sps, pps: &Pps) -> Result<()> {
        // Reuse the session only when the parameter sets are the same ones, not
        // merely the same ids: an SPS may be resent with different content
        // under an id it has already used.
        if self.common.session.is_some()
            && self.active_sps.as_ref() == Some(sps)
            && self.active_pps.as_ref() == Some(pps)
        {
            return Ok(());
        }

        // `profile_for` rejects scaling matrices in the SPS; a PPS can carry
        // its own, and those are just as unsupported. Without this the session
        // would report pic_scaling_matrix_present_flag = 0 with null scaling
        // lists, and the driver would silently dequantise with a flat matrix.
        if pps.pic_scaling_matrix_present_flag {
            return Err(PixelForgeError::InvalidInput(
                "H.264 decode: streams with explicit scaling matrices are not supported"
                    .to_string(),
            ));
        }

        let (profile, chroma_format, bit_depth) = Self::profile_for(sps)?;

        // The stream needs max_num_ref_frames references plus the current
        // picture. Reorder depth comes from the stream (VUI); without it, bound
        // by the reference count, which is never smaller than the reorder depth
        // in practice.
        let reorder_depth = sps
            .max_num_reorder_frames
            .map(|v| v as usize)
            .unwrap_or(sps.max_num_ref_frames as usize);
        let needs = StreamNeeds {
            codec: "H.264",
            width: sps.coded_width(),
            height: sps.coded_height(),
            chroma_format,
            bit_depth,
            references: sps.max_num_ref_frames as u32,
            active_references: sps.max_num_ref_frames as u32,
            slot_limit: crate::decoder::h264::dpb::MAX_DPB_SLOTS as u32,
            reorder_depth,
            output_depth: self.output_depth,
            film_grain: false,
        };
        let mut h264_caps = vk::VideoDecodeH264CapabilitiesKHR::default();
        let plan = self
            .common
            .plan_session(profile.info(), &mut h264_caps, &needs)?;
        let slot_count = plan.slot_count;
        self.common
            .create_session(&plan, profile, |common, session| {
                Self::build_session_params(common, session, sps, pps)
            })?;

        self.active_sps = Some(sps.clone());
        self.active_pps = Some(pps.clone());
        self.reorder_depth = reorder_depth;
        let pins = self.common.session()?.slot_pins.clone();
        self.dpb = Some(DecodeDpb::new(slot_count, pins));

        Ok(())
    }

    /// Translate the parsed SPS/PPS into StdVideo structs and create session
    /// parameters.
    fn build_session_params(
        common: &DecoderCommon,
        session: vk::VideoSessionKHR,
        sps: &Sps,
        pps: &Pps,
    ) -> Result<vk::VideoSessionParametersKHR> {
        let mut sps_flags: ash::vk::native::StdVideoH264SpsFlags = unsafe { std::mem::zeroed() };
        sps_flags.set_constraint_set0_flag((sps.constraint_set_flags >> 7) as u32 & 1);
        sps_flags.set_constraint_set1_flag((sps.constraint_set_flags >> 6) as u32 & 1);
        sps_flags.set_constraint_set2_flag((sps.constraint_set_flags >> 5) as u32 & 1);
        sps_flags.set_constraint_set3_flag((sps.constraint_set_flags >> 4) as u32 & 1);
        sps_flags.set_constraint_set4_flag((sps.constraint_set_flags >> 3) as u32 & 1);
        sps_flags.set_constraint_set5_flag((sps.constraint_set_flags >> 2) as u32 & 1);
        sps_flags.set_direct_8x8_inference_flag(sps.direct_8x8_inference_flag as u32);
        sps_flags.set_mb_adaptive_frame_field_flag(sps.mb_adaptive_frame_field_flag as u32);
        sps_flags.set_frame_mbs_only_flag(sps.frame_mbs_only_flag as u32);
        sps_flags.set_delta_pic_order_always_zero_flag(sps.delta_pic_order_always_zero_flag as u32);
        sps_flags.set_separate_colour_plane_flag(sps.separate_colour_plane_flag as u32);
        sps_flags.set_gaps_in_frame_num_value_allowed_flag(
            sps.gaps_in_frame_num_value_allowed_flag as u32,
        );
        sps_flags.set_qpprime_y_zero_transform_bypass_flag(
            sps.qpprime_y_zero_transform_bypass_flag as u32,
        );
        sps_flags.set_frame_cropping_flag(sps.frame_cropping_flag as u32);
        sps_flags.set_seq_scaling_matrix_present_flag(0);
        sps_flags.set_vui_parameters_present_flag(0);

        let std_sps = ash::vk::native::StdVideoH264SequenceParameterSet {
            flags: sps_flags,
            profile_idc: std_profile_idc(sps.profile_idc)?,
            level_idc: std_level_idc(sps.level_idc),
            chroma_format_idc: std_chroma_format_idc(sps.chroma_format_idc)?,
            seq_parameter_set_id: sps.sps_id,
            bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
            bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
            log2_max_frame_num_minus4: sps.log2_max_frame_num_minus4,
            pic_order_cnt_type: sps.pic_order_cnt_type as u32,
            offset_for_non_ref_pic: sps.offset_for_non_ref_pic,
            offset_for_top_to_bottom_field: sps.offset_for_top_to_bottom_field,
            log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
            num_ref_frames_in_pic_order_cnt_cycle: sps.offsets_for_ref_frame.len() as u8,
            max_num_ref_frames: sps.max_num_ref_frames,
            reserved1: 0,
            pic_width_in_mbs_minus1: sps.pic_width_in_mbs_minus1,
            pic_height_in_map_units_minus1: sps.pic_height_in_map_units_minus1,
            frame_crop_left_offset: sps.frame_crop_left_offset,
            frame_crop_right_offset: sps.frame_crop_right_offset,
            frame_crop_top_offset: sps.frame_crop_top_offset,
            frame_crop_bottom_offset: sps.frame_crop_bottom_offset,
            reserved2: 0,
            pOffsetForRefFrame: if sps.offsets_for_ref_frame.is_empty() {
                std::ptr::null()
            } else {
                sps.offsets_for_ref_frame.as_ptr()
            },
            pScalingLists: std::ptr::null(),
            pSequenceParameterSetVui: std::ptr::null(),
        };

        let mut pps_flags: ash::vk::native::StdVideoH264PpsFlags = unsafe { std::mem::zeroed() };
        pps_flags.set_transform_8x8_mode_flag(pps.transform_8x8_mode_flag as u32);
        pps_flags.set_redundant_pic_cnt_present_flag(pps.redundant_pic_cnt_present_flag as u32);
        pps_flags.set_constrained_intra_pred_flag(pps.constrained_intra_pred_flag as u32);
        pps_flags.set_deblocking_filter_control_present_flag(
            pps.deblocking_filter_control_present_flag as u32,
        );
        pps_flags.set_weighted_pred_flag(pps.weighted_pred_flag as u32);
        pps_flags.set_bottom_field_pic_order_in_frame_present_flag(
            pps.bottom_field_pic_order_in_frame_present_flag as u32,
        );
        pps_flags.set_entropy_coding_mode_flag(pps.entropy_coding_mode_flag as u32);
        pps_flags.set_pic_scaling_matrix_present_flag(0);

        let std_pps = ash::vk::native::StdVideoH264PictureParameterSet {
            flags: pps_flags,
            seq_parameter_set_id: pps.sps_id,
            pic_parameter_set_id: pps.pps_id,
            num_ref_idx_l0_default_active_minus1: pps.num_ref_idx_l0_default_active_minus1,
            num_ref_idx_l1_default_active_minus1: pps.num_ref_idx_l1_default_active_minus1,
            weighted_bipred_idc: pps.weighted_bipred_idc as u32,
            pic_init_qp_minus26: pps.pic_init_qp_minus26,
            pic_init_qs_minus26: pps.pic_init_qs_minus26,
            chroma_qp_index_offset: pps.chroma_qp_index_offset,
            second_chroma_qp_index_offset: pps.second_chroma_qp_index_offset,
            pScalingLists: std::ptr::null(),
        };

        let sps_array = [std_sps];
        let pps_array = [std_pps];
        let add_info = vk::VideoDecodeH264SessionParametersAddInfoKHR::default()
            .std_sp_ss(&sps_array)
            .std_pp_ss(&pps_array);
        let mut h264_create = vk::VideoDecodeH264SessionParametersCreateInfoKHR::default()
            .max_std_sps_count(1)
            .max_std_pps_count(1)
            .parameters_add_info(&add_info);

        let create_info = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(session)
            .push(&mut h264_create);

        unsafe {
            common
                .video_queue_fn
                .create_video_session_parameters(&create_info, None)
        }
        .map_err(PixelForgeError::vk("vkCreateVideoSessionParametersKHR"))
    }

    /// Split `data` into pictures, absorbing any parameter sets encountered.
    /// Group NALs into pictures. Returns the pictures plus, if any slice was
    /// skipped for missing parameter sets, the reason a keyframe is needed.
    fn split_pictures<'a>(&mut self, data: &'a [u8]) -> Result<(Vec<Picture<'a>>, Option<String>)> {
        // Parameter sets are absorbed into self; slice grouping is delegated to
        // the pure `group_slices` so it can be tested without a device.
        let mut pictures: Vec<Picture<'a>> = Vec::new();
        let mut awaiting: Option<String> = None;

        for nal in iter_nal_units(data) {
            match nal.nal_type {
                NalType::Sps => {
                    let sps = parser::parse_sps(nal.payload())?;
                    debug!(
                        "H.264 SPS id={} {}x{} profile={}",
                        sps.sps_id,
                        sps.coded_width(),
                        sps.coded_height(),
                        sps.profile_idc
                    );
                    self.sps_map.insert(sps.sps_id, sps);
                }
                NalType::Pps => {
                    let sps_map = &self.sps_map;
                    let pps = parser::parse_pps(nal.payload(), |id| {
                        sps_map.get(&id).map(|s| s.chroma_format_idc)
                    })?;
                    self.pps_map.insert(pps.pps_id, pps);
                }
                t if t.is_slice() => {
                    match group_slices(&mut pictures, &nal, &self.sps_map, &self.pps_map)? {
                        Grouped::Added => {}
                        // Missing parameter sets: skip the slice, remember why.
                        Grouped::NeedsParameterSets(reason) => {
                            awaiting.get_or_insert(reason);
                        }
                    }
                }
                // SEI, AUD, and everything else is not needed by the driver.
                _ => {}
            }
        }

        Ok((pictures, awaiting))
    }

    /// Record and submit the decode of a single picture, without waiting for it.
    fn decode_picture(&mut self, picture: &Picture, sps: &Sps, pts: u64) -> Result<DecodedPicture> {
        // --- Host-side bookkeeping: POC, slot, reference set ---
        let state = {
            let dpb = self.dpb.as_mut().expect("activated above");
            dpb.begin_picture(
                picture.nal_type,
                picture.ref_idc,
                &picture.header,
                sps,
                picture.is_intra,
            )?
        };
        // Every slot can be busy for two reasons: the stream keeps that many
        // references (a hard error), or a frame the caller still holds pins one
        // (wait for them to drop it, like the encoder waits for a free slot).
        let slot = loop {
            let dpb = self.dpb.as_mut().expect("active");
            if let Some(slot) = dpb.try_allocate_slot() {
                break slot;
            }
            if !dpb.has_pinned_slots() {
                return Err(PixelForgeError::InvalidInput(
                    "H.264 decode: no free DPB slot (stream exceeds negotiated DPB size)"
                        .to_string(),
                ));
            }
            self.common.session()?.slot_pins.wait_for_release();
        };

        // --- Record and submit ---
        //
        // Claim the pipeline slot first. Staging writes into that slot's
        // buffer, and `ensure_bitstream_capacity` may replace it outright, so
        // both have to happen after the slot's previous submission has
        // completed. Staging before this waits overwrote coded data the GPU was
        // still reading.
        self.common.begin_decode_commands()?;
        let (buffer_range, slice_offsets) =
            self.common.stage_slices(&picture.slices, &START_CODE)?;
        self.common.record_barriers(slot);
        self.record_decode(picture, &state, slot, buffer_range, &slice_offsets)?;
        self.common.submit_decode()?;

        // --- Publish the frame, then retire references ---
        let (width, height) = sps.display_dimensions();
        let frame = self
            .common
            .decoded_picture(slot, width, height, pts, state.poc, state.is_idr);

        self.dpb
            .as_mut()
            .expect("active")
            .end_picture(slot, &state, sps);

        Ok(frame)
    }

    /// Record `vkCmdBeginVideoCoding` / `vkCmdDecodeVideo` / `vkCmdEndVideoCoding`.
    fn record_decode(
        &mut self,
        picture: &Picture,
        state: &PictureState,
        slot: u8,
        buffer_range: u64,
        slice_offsets: &[u32],
    ) -> Result<()> {
        let active = self.common.session.as_ref().expect("active");

        // The slot's previous decode has finished (`begin_decode_commands`
        // waited for it), so its record can be rewritten.
        let record = &mut self.records[self.common.pipeline.current_index()];
        record.reset(slice_offsets);

        // Every reference the driver may use, plus the slot we reconstruct into.
        for r in self.dpb.as_ref().expect("active").references() {
            let mut flags: ash::vk::native::StdVideoDecodeH264ReferenceInfoFlags =
                unsafe { std::mem::zeroed() };
            flags.set_top_field_flag(0);
            flags.set_bottom_field_flag(0);
            flags.set_used_for_long_term_reference(r.long_term as u32);
            flags.set_is_non_existing(0);
            // For a long-term reference this field carries LongTermFrameIdx
            // rather than frame_num; that is how the driver identifies it when
            // building the reference lists.
            let frame_num = if r.long_term {
                r.long_term_frame_idx as u16
            } else {
                r.frame_num
            };
            record.push_reference(
                r.slot,
                active.dpb_entry(r.slot).2,
                None,
                ash::vk::native::StdVideoDecodeH264ReferenceInfo {
                    flags,
                    FrameNum: frame_num,
                    reserved: 0,
                    PicOrderCnt: r.poc,
                },
            );
        }

        // Setup (reconstruction) slot info for the current picture.
        let mut setup_flags: ash::vk::native::StdVideoDecodeH264ReferenceInfoFlags =
            unsafe { std::mem::zeroed() };
        setup_flags.set_top_field_flag(0);
        setup_flags.set_bottom_field_flag(0);
        setup_flags.set_used_for_long_term_reference(0);
        setup_flags.set_is_non_existing(0);
        let setup_std = ash::vk::native::StdVideoDecodeH264ReferenceInfo {
            flags: setup_flags,
            FrameNum: state.frame_num,
            reserved: 0,
            PicOrderCnt: [state.poc, state.poc],
        };

        let mut pic_flags: ash::vk::native::StdVideoDecodeH264PictureInfoFlags =
            unsafe { std::mem::zeroed() };
        pic_flags.set_field_pic_flag(0);
        pic_flags.set_is_intra(state.is_intra as u32);
        pic_flags.set_IdrPicFlag(state.is_idr as u32);
        pic_flags.set_bottom_field_flag(0);
        pic_flags.set_is_reference(state.is_reference as u32);
        pic_flags.set_complementary_field_pair(0);
        let pic_std = ash::vk::native::StdVideoDecodeH264PictureInfo {
            flags: pic_flags,
            seq_parameter_set_id: picture.sps.sps_id,
            pic_parameter_set_id: picture.header.pps_id,
            reserved1: 0,
            reserved2: 0,
            frame_num: state.frame_num,
            idr_pic_id: state.idr_pic_id,
            PicOrderCnt: [state.poc, state.poc],
        };

        *record.picture_mut() = pic_std;

        // The decode destination is the DPB slot itself when the implementation
        // supports coincide, otherwise the separate output image.
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

impl CodecDecoder for H264Decoder {
    type Unit<'a> = Picture<'a>;

    const NAME: &'static str = "H.264";

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
        picture.nal_type == NalType::IdrSlice
    }

    fn decode_unit(&mut self, picture: &Picture<'_>, pts: u64) -> Result<Decoded> {
        // The parameter sets this picture was grouped under, not whatever
        // currently sits at those ids: a later SPS in the same chunk must not
        // reach back and change how this picture is decoded.
        let (sps, pps) = (picture.sps.clone(), picture.pps.clone());

        // (Re)build the session if the stream's geometry or ids changed.
        self.activate(&sps, &pps)?;
        self.decode_picture(picture, &sps, pts).map(Decoded::Shown)
    }

    fn reorder_depth(&self) -> usize {
        self.reorder_depth
    }
}

/// Append a slice NAL to the picture in progress, or begin a new picture.
///
/// Free function (rather than a method) so the grouping logic — which is pure
/// and easy to get wrong for multi-slice streams — can be unit tested without a
/// Vulkan device.
fn group_slices<'a>(
    pictures: &mut Vec<Picture<'a>>,
    nal: &NalUnit<'a>,
    sps_map: &HashMap<u8, Sps>,
    pps_map: &HashMap<u8, Pps>,
) -> Result<Grouped> {
    // Peek the header to find the picture this slice belongs to.
    // Peek just far enough to find which PPS (and hence SPS) applies.
    let mut probe = crate::decoder::bitreader::BitReader::new(nal.payload());
    let _first_mb_in_slice = probe.ue()?;
    let _slice_type = probe.ue()?;
    let pps_id = probe.ue()? as u8;

    // Missing parameter sets: we joined the stream before they were sent. Not a
    // fault — signal that a keyframe (which carries fresh sets) is needed.
    let Some(pps) = pps_map.get(&pps_id) else {
        return Ok(Grouped::NeedsParameterSets(format!(
            "PPS {} not yet received",
            pps_id
        )));
    };
    let Some(sps) = sps_map.get(&pps.sps_id) else {
        return Ok(Grouped::NeedsParameterSets(format!(
            "SPS {} not yet received",
            pps.sps_id
        )));
    };

    let header = parser::parse_slice_header(nal, sps, pps)?;
    let is_intra = matches!(header.slice_type, SliceType::I | SliceType::Si);
    let (sps, pps) = (sps.clone(), pps.clone());

    // Picture boundary detection (clause 7.4.1.2.4, frame-picture subset).
    //
    // `first_mb_in_slice == 0` marks the first slice of a picture and is
    // sufficient for conforming streams; the syntax comparisons additionally
    // catch a picture boundary that a truncated or spliced stream would
    // otherwise hide (e.g. a dropped first slice).
    let starts_new = match pictures.last() {
        None => true,
        Some(prev) => {
            header.first_mb_in_slice == 0
                || prev.header.frame_num != header.frame_num
                || prev.header.pps_id != header.pps_id
                || prev.nal_type != nal.nal_type
                || prev.header.pic_order_cnt_lsb != header.pic_order_cnt_lsb
                || prev.header.idr_pic_id != header.idr_pic_id
        }
    };

    if starts_new {
        pictures.push(Picture {
            header,
            nal_type: nal.nal_type,
            ref_idc: nal.ref_idc,
            is_intra,
            slices: vec![nal.data],
            sps,
            pps,
        });
    } else {
        let picture = pictures.last_mut().expect("checked above");
        // A picture is intra only if *every* slice is intra.
        picture.is_intra &= is_intra;
        picture.slices.push(nal.data);
    }
    Ok(Grouped::Added)
}

fn std_profile_idc(profile_idc: u8) -> Result<ash::vk::native::StdVideoH264ProfileIdc> {
    Ok(match profile_idc {
        66 => ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_BASELINE,
        77 => ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN,
        100 => ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH,
        244 => {
            ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE
        }
        other => {
            return Err(PixelForgeError::InvalidInput(format!(
                "H.264 decode: unsupported profile_idc {}",
                other
            )));
        }
    })
}

fn std_chroma_format_idc(idc: u8) -> Result<ash::vk::native::StdVideoH264ChromaFormatIdc> {
    Ok(match idc {
        0 => {
            ash::vk::native::StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_MONOCHROME
        }
        1 => ash::vk::native::StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_420,
        2 => ash::vk::native::StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_422,
        3 => ash::vk::native::StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_444,
        other => {
            return Err(PixelForgeError::InvalidInput(format!(
                "H.264 decode: invalid chroma_format_idc {}",
                other
            )));
        }
    })
}

/// Map `level_idc` (10 * level) onto the StdVideo enum.
fn std_level_idc(level_idc: u8) -> ash::vk::native::StdVideoH264LevelIdc {
    use ash::vk::native::*;
    match level_idc {
        10 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_0,
        11 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_1,
        12 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_2,
        13 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_3,
        20 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_0,
        21 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_1,
        22 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_2,
        30 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_0,
        31 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_1,
        32 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_2,
        40 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_0,
        41 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_1,
        42 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_2,
        50 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_0,
        51 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_1,
        52 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_2,
        60 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_0,
        61 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_1,
        _ => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A chunk carrying a parameter set *change* must not apply it backwards.
    ///
    /// A stream signals a new resolution by resending an SPS under an id it has
    /// already used. Pictures grouped before that point belong to the old SPS,
    /// and if the parameter sets were resolved by id after the whole chunk had
    /// been scanned, every picture in it would be decoded against the last SPS
    /// seen. That produced a real failure: a 320x240 clip followed by a 640x480
    /// one, fed as one byte stream, decoded entirely at 640x480.
    #[test]
    fn a_later_parameter_set_does_not_reach_back() {
        let first: &[u8] = include_bytes!("../../../tests/data/base.264");
        let second: &[u8] = include_bytes!("../../../tests/data/multislice.264");

        // Both fixtures use SPS id 0, so concatenating them is exactly the
        // id-reuse case, and `group_slices` sees the second SPS partway
        // through.
        let mut stream = first.to_vec();
        stream.extend_from_slice(second);

        let mut sps_map = HashMap::new();
        let mut pps_map = HashMap::new();
        let mut pictures: Vec<Picture> = Vec::new();
        let mut sps_seen = 0usize;

        for nal in parser::iter_nal_units(&stream) {
            match nal.nal_type {
                NalType::Sps => {
                    let sps = parser::parse_sps(nal.payload()).unwrap();
                    sps_seen += 1;
                    sps_map.insert(sps.sps_id, sps);
                }
                NalType::Pps => {
                    let pps = parser::parse_pps(nal.payload(), |id| {
                        sps_map.get(&id).map(|s: &Sps| s.chroma_format_idc)
                    })
                    .unwrap();
                    pps_map.insert(pps.pps_id, pps);
                }
                t if t.is_slice() => {
                    group_slices(&mut pictures, &nal, &sps_map, &pps_map).unwrap();
                }
                _ => {}
            }
        }

        assert!(sps_seen >= 2, "fixture should carry more than one SPS");
        assert_ne!(
            pictures[0].sps, pictures[59].sps,
            "the two fixtures must differ in SPS content, or this test proves nothing"
        );
        assert_eq!(pictures.len(), 60, "30 pictures from each fixture");

        // Every picture must carry the parameter sets that were in effect when
        // it was grouped, so the two halves keep their own.
        let first_sps = &pictures[0].sps;
        let second_sps = &pictures[59].sps;
        assert_eq!(
            first_sps.sps_id, second_sps.sps_id,
            "the fixtures should reuse the same id, or this proves nothing"
        );
        for (i, picture) in pictures.iter().enumerate().take(30) {
            assert_eq!(&picture.sps, first_sps, "picture {i} took a later SPS");
        }
        for (i, picture) in pictures.iter().enumerate().skip(30) {
            assert_eq!(&picture.sps, second_sps, "picture {i} kept an earlier SPS");
        }
    }

    /// Joining a stream after its parameter sets were sent must be reported as
    /// needing a keyframe, not as a failure and not by silently decoding
    /// garbage. This is the ordinary case for a client that connects to a live
    /// stream partway through, and it is what `DecodeStatus::NeedsKeyframe`
    /// exists to say.
    #[test]
    fn slices_without_parameter_sets_ask_for_a_keyframe() {
        let data: &[u8] = include_bytes!("../../../tests/data/bframes.264");

        // Deliberately empty: the SPS and PPS went past before we tuned in.
        let sps_map = HashMap::new();
        let pps_map = HashMap::new();
        let mut pictures: Vec<Picture> = Vec::new();

        let mut slices = 0usize;
        for nal in parser::iter_nal_units(data) {
            if !nal.nal_type.is_slice() {
                continue;
            }
            slices += 1;
            match group_slices(&mut pictures, &nal, &sps_map, &pps_map).unwrap() {
                Grouped::NeedsParameterSets(reason) => {
                    assert!(
                        reason.contains("PPS"),
                        "should name the missing parameter set, got {reason:?}"
                    );
                }
                Grouped::Added => panic!("a slice was grouped with no parameter sets to group it"),
            }
        }

        assert!(slices > 0, "fixture should contain slices");
        assert!(
            pictures.is_empty(),
            "no picture can be formed without parameter sets"
        );
    }

    /// Group the slices of a real multi-slice stream (4 slices per picture,
    /// generated by `x264 --slices 4`). Exercises `group_slices` without a GPU.
    #[test]
    fn groups_multislice_pictures() {
        let data: &[u8] = include_bytes!("../../../tests/data/multislice.264");

        let mut sps_map = HashMap::new();
        let mut pps_map = HashMap::new();
        let mut pictures: Vec<Picture> = Vec::new();

        for nal in parser::iter_nal_units(data) {
            match nal.nal_type {
                NalType::Sps => {
                    let sps = parser::parse_sps(nal.payload()).unwrap();
                    sps_map.insert(sps.sps_id, sps);
                }
                NalType::Pps => {
                    let pps = parser::parse_pps(nal.payload(), |id| {
                        sps_map.get(&id).map(|s: &Sps| s.chroma_format_idc)
                    })
                    .unwrap();
                    pps_map.insert(pps.pps_id, pps);
                }
                t if t.is_slice() => {
                    group_slices(&mut pictures, &nal, &sps_map, &pps_map).unwrap();
                }
                _ => {}
            }
        }

        // 120 slice NALs must group into 30 pictures of 4 slices each.
        assert_eq!(pictures.len(), 30, "expected 30 pictures");
        for (i, picture) in pictures.iter().enumerate() {
            assert_eq!(picture.slices.len(), 4, "picture {i} should have 4 slices");
        }

        // The first picture is the IDR and is intra-coded.
        assert_eq!(pictures[0].nal_type, NalType::IdrSlice);
        assert!(pictures[0].is_intra, "IDR must be intra");
        assert_ne!(pictures[0].ref_idc, 0, "IDR must be a reference");

        // frame_num must be constant within a picture and advance between them.
        assert_eq!(pictures[0].header.frame_num, 0);
        assert!(
            pictures[1].header.frame_num > 0,
            "frame_num must advance after the IDR"
        );
    }
}
