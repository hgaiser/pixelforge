//! AV1 per-frame encode recording: builds the StdVideo* graph for one frame and
//! submits it. AV1 uses single-reference prediction.

use super::references::{FrameReferences, ref_frame_idx};
use super::{Av1, order_hint};

use crate::encoder::codec::{EncodeCommands, EncoderCommon, FramePlan, RateControlPlan};
use crate::encoder::gop::ReferencePattern;
use crate::encoder::resources::{prepare_encode_command_buffer, record_dpb_barriers};
use crate::error::Result;
use ash::vk;
use ash::vk::TaggedStructure;
use tracing::debug;

/// What a constant quantizer index is raised by for a B-frame that is not a
/// reference. About the step that 2 is for an H.264 QP.
const B_FRAME_Q_INDEX_OFFSET: u32 = 8;

impl Av1 {
    pub(super) fn record(&mut self, common: &mut EncoderCommon, plan: &FramePlan) -> Result<()> {
        let is_key_frame = plan.is_idr();
        let is_b = plan.is_b();
        // All frames need a setup reference slot (DPB write) per the Vulkan spec
        // when maxDpbSlots > 0, also a frame that no frame predicts from.
        let has_setup_slot = true;
        let current_dpb_slot = common.current_dpb_slot;
        let order_hint = order_hint(plan.pic_order_cnt());
        let (references, used_references) = self.frame_references(plan)?;

        let slot = common.pipeline.current();
        let command_buffer = slot.encode_command_buffer;
        let query_pool = slot.query_pool;
        let bitstream_buffer = slot.bitstream_buffer;
        let bitstream_buffer_size = slot.bitstream_buffer_size;
        let input_image_view = common.source_view();

        debug!(
            "av1 record: key={}, refs_len={}, dpb_slot={}",
            is_key_frame,
            self.references.len(),
            current_dpb_slot
        );

        let rc = RateControlPlan::for_frame(&common.config, 128, plan, B_FRAME_Q_INDEX_OFFSET, 255);
        let qp = rc.qp;

        unsafe {
            prepare_encode_command_buffer(common.device(), command_buffer, query_pool)?;
        }
        // Only the references this frame predicts from go to the encode. The
        // rest of the window stays in its DPB slots, untouched.
        let ref_dpb_slots: Vec<u8> = used_references.iter().map(|r| r.dpb_slot).collect();
        unsafe {
            record_dpb_barriers(
                common.context.sync2(),
                command_buffer,
                &common.dpb_images,
                common.use_layered_dpb,
                current_dpb_slot,
                &ref_dpb_slots,
                common.dpb_slot_active[current_dpb_slot as usize],
            );
            common.record_source_acquire(command_buffer);
        }

        let frame_type = if is_key_frame {
            ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_KEY
        } else {
            ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_INTER
        };

        // show_frame for all frames; error_resilient_mode on key frames (FFmpeg).
        let mut picture_info_flags = ash::vk::native::StdVideoEncodeAV1PictureInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: Default::default(),
        };
        picture_info_flags.set_show_frame(!plan.hidden as u32);
        if is_key_frame {
            picture_info_flags.set_error_resilient_mode(1);
        } else {
            picture_info_flags.set_showable_frame(1);
        }

        // Without MOTION_VECTOR_SCALING all picture resource codedExtents must
        // match. The device may code the frame larger, to its
        // `codedPictureAlignment`; the sequence header allows that size.
        let frame_extent = vk::Extent2D {
            width: common.config.dimensions.width,
            height: common.config.dimensions.height,
        };

        let setup_picture_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(frame_extent)
            .base_array_layer(0)
            .image_view_binding(common.dpb_image_views[current_dpb_slot as usize]);

        let reference_info_flags = ash::vk::native::StdVideoEncodeAV1ReferenceInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeAV1ReferenceInfoFlags::new_bitfield_1(
                0, 0, 0,
            ),
        };
        let std_reference_info = ash::vk::native::StdVideoEncodeAV1ReferenceInfo {
            flags: reference_info_flags,
            frame_type,
            RefFrameId: current_dpb_slot as u32,
            OrderHint: order_hint as u8,
            reserved1: [0; 3],
            pExtensionHeader: std::ptr::null(),
        };

        let mut setup_av1_dpb_info =
            vk::VideoEncodeAV1DpbSlotInfoKHR::default().std_reference_info(&std_reference_info);
        let mut setup_av1_dpb_info_ref0 = setup_av1_dpb_info;
        let setup_reference_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(current_dpb_slot as i32)
            .picture_resource(&setup_picture_resource)
            .push(&mut setup_av1_dpb_info_ref0);

        // Reference frames for inter frames: declare each distinct DPB slot
        // this frame predicts from, most-recent first. The vectors hold the data so the
        // pointers the slot infos take into them stay stable.
        //
        // `distinct_refs` dedups by DPB slot (a reference can be mapped to more
        // than one AV1 reference name below, but each slot is declared once).
        let mut distinct_refs: Vec<&super::ReferenceInfo> = Vec::new();
        if !is_key_frame {
            for r in &used_references {
                if !distinct_refs.iter().any(|d| d.dpb_slot == r.dpb_slot) {
                    distinct_refs.push(r);
                }
            }
        }

        let mut ref_std_infos = Vec::with_capacity(distinct_refs.len());
        for ref_info in &distinct_refs {
            ref_std_infos.push(ash::vk::native::StdVideoEncodeAV1ReferenceInfo {
                flags: reference_info_flags,
                frame_type: ref_info.codec.frame_type,
                RefFrameId: ref_info.dpb_slot as u32,
                OrderHint: super::order_hint(ref_info.poc) as u8,
                reserved1: [0; 3],
                pExtensionHeader: std::ptr::null(),
            });
        }

        let mut ref_picture_resources = Vec::with_capacity(distinct_refs.len());
        for ref_info in &distinct_refs {
            ref_picture_resources.push(
                vk::VideoPictureResourceInfoKHR::default()
                    .coded_offset(vk::Offset2D { x: 0, y: 0 })
                    .coded_extent(frame_extent)
                    .base_array_layer(0)
                    .image_view_binding(common.dpb_image_views[ref_info.dpb_slot as usize]),
            );
        }

        let mut av1_reference_infos = Vec::with_capacity(distinct_refs.len());
        for std_info in &ref_std_infos {
            av1_reference_infos
                .push(vk::VideoEncodeAV1DpbSlotInfoKHR::default().std_reference_info(std_info));
        }

        let mut reference_slots = Vec::with_capacity(distinct_refs.len());
        for ((ref_info, resource), dpb_info) in distinct_refs
            .iter()
            .zip(ref_picture_resources.iter())
            .zip(av1_reference_infos.iter_mut())
        {
            reference_slots.push(
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(ref_info.dpb_slot as i32)
                    .picture_resource(resource)
                    .push(dpb_info),
            );
        }

        // Quantization (FFmpeg-style defaults).
        let quantization = ash::vk::native::StdVideoAV1Quantization {
            flags: ash::vk::native::StdVideoAV1QuantizationFlags {
                _bitfield_align_1: [],
                _bitfield_1: ash::vk::native::StdVideoAV1QuantizationFlags::new_bitfield_1(0, 0, 0),
            },
            base_q_idx: qp as u8,
            DeltaQYDc: 0,
            DeltaQUDc: 0,
            DeltaQUAc: 0,
            DeltaQVDc: 0,
            DeltaQVAc: 0,
            qm_y: 0,
            qm_u: 0,
            qm_v: 0,
        };

        // CDEF and loop restoration. A lossless frame (quantizer index 0)
        // codes neither. The encoder picks one of four CDEF strengths for each
        // block; the strengths are those of the Vulkan Video samples. NVIDIA
        // (615.78.08) ignores the chroma strengths and writes its own.
        let lossless = rc.is_disabled() && qp == 0;
        let cdef = if lossless {
            ash::vk::native::StdVideoAV1CDEF {
                cdef_damping_minus_3: 0,
                cdef_bits: 0,
                cdef_y_pri_strength: [0; 8],
                cdef_y_sec_strength: [0; 8],
                cdef_uv_pri_strength: [0; 8],
                cdef_uv_sec_strength: [0; 8],
            }
        } else {
            ash::vk::native::StdVideoAV1CDEF {
                cdef_damping_minus_3: 2,
                cdef_bits: 2,
                cdef_y_pri_strength: [0, 2, 4, 9, 0, 0, 0, 0],
                cdef_y_sec_strength: [0; 8],
                cdef_uv_pri_strength: [0; 8],
                cdef_uv_sec_strength: [0; 8],
            }
        };
        // Loop restoration on luma, with 64x64 restoration units
        // (`LoopRestorationSize` is the log2 of the size, less 5). NVIDIA
        // (615.78.08) writes SGRPROJ, whatever type it gets.
        let loop_restoration = ash::vk::native::StdVideoAV1LoopRestoration {
            FrameRestorationType: [
                ash::vk::native::StdVideoAV1FrameRestorationType_STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_SGRPROJ,
                ash::vk::native::StdVideoAV1FrameRestorationType_STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE,
                ash::vk::native::StdVideoAV1FrameRestorationType_STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE,
            ],
            LoopRestorationSize: [1, 1, 1],
        };
        if !lossless {
            picture_info_flags.set_UsesLr(1);
        }

        let loop_filter = ash::vk::native::StdVideoAV1LoopFilter {
            flags: ash::vk::native::StdVideoAV1LoopFilterFlags {
                _bitfield_align_1: [],
                _bitfield_1: ash::vk::native::StdVideoAV1LoopFilterFlags::new_bitfield_1(0, 0, 0),
            },
            loop_filter_level: [0; 4],
            loop_filter_sharpness: 0,
            update_ref_delta: 0,
            loop_filter_ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
            update_mode_delta: 1,
            loop_filter_mode_deltas: [0; 2],
        };

        let (ref_frame_idx, ref_order_hint, primary_ref_frame, refresh_frame_flags) =
            self.calculate_reference_frame_mapping(plan, &references, current_dpb_slot);

        let std_picture_info = ash::vk::native::StdVideoEncodeAV1PictureInfo {
            flags: picture_info_flags,
            frame_type,
            frame_presentation_time: self.frame_num,
            current_frame_id: current_dpb_slot as u32,
            order_hint: order_hint as u8,
            primary_ref_frame,
            refresh_frame_flags,
            coded_denom: 0,
            render_width_minus_1: (common.config.dimensions.width - 1) as u16,
            render_height_minus_1: (common.config.dimensions.height - 1) as u16,
            interpolation_filter: ash::vk::native::StdVideoAV1InterpolationFilter_STD_VIDEO_AV1_INTERPOLATION_FILTER_EIGHTTAP,
            TxMode: ash::vk::native::StdVideoAV1TxMode_STD_VIDEO_AV1_TX_MODE_SELECT,
            delta_q_res: 0,
            delta_lf_res: 0,
            ref_order_hint,
            ref_frame_idx,
            reserved1: [0; 3],
            delta_frame_id_minus_1: [0; 7],
            pTileInfo: std::ptr::null(),
            pQuantization: &quantization,
            pSegmentation: std::ptr::null(),
            pLoopFilter: &loop_filter,
            pCDEF: &cdef,
            pLoopRestoration: if lossless {
                std::ptr::null()
            } else {
                &loop_restoration
            },
            pGlobalMotion: std::ptr::null(),
            pExtensionHeader: std::ptr::null(),
            pBufferRemovalTimes: std::ptr::null(),
        };

        // Map the AV1 reference names this frame predicts from to DPB slots
        // (see `reference_names`); the others stay -1. The names that are set
        // must mirror `ref_frame_idx` from `calculate_reference_frame_mapping`.
        let reference_name_slot_indices = references.names;

        let (prediction_mode, rate_control_group) = if is_key_frame {
            (
                vk::VideoEncodeAV1PredictionModeKHR::INTRA_ONLY,
                vk::VideoEncodeAV1RateControlGroupKHR::INTRA,
            )
        } else if is_b {
            (
                vk::VideoEncodeAV1PredictionModeKHR::BIDIRECTIONAL_COMPOUND,
                vk::VideoEncodeAV1RateControlGroupKHR::BIPREDICTIVE,
            )
        } else {
            (
                vk::VideoEncodeAV1PredictionModeKHR::SINGLE_REFERENCE,
                vk::VideoEncodeAV1RateControlGroupKHR::PREDICTIVE,
            )
        };

        let mut av1_picture_info = vk::VideoEncodeAV1PictureInfoKHR::default()
            .std_picture_info(&std_picture_info)
            .prediction_mode(prediction_mode)
            .rate_control_group(rate_control_group)
            .reference_name_slot_indices(reference_name_slot_indices);
        if rc.is_disabled() {
            av1_picture_info = av1_picture_info.constant_q_index(qp);
        }

        let mut av1_rc_layer_info = vk::VideoEncodeAV1RateControlLayerInfoKHR::default();
        if rc.is_disabled() {
            let q_index = vk::VideoEncodeAV1QIndexKHR {
                intra_q_index: qp,
                predictive_q_index: qp,
                bipredictive_q_index: qp,
            };
            av1_rc_layer_info = av1_rc_layer_info
                .use_min_q_index(true)
                .min_q_index(q_index)
                .use_max_q_index(true)
                .max_q_index(q_index);
        } else {
            av1_rc_layer_info = av1_rc_layer_info
                .use_min_q_index(false)
                .use_max_q_index(false);
        }

        // Begin coding: setup slot (slot_index -1) plus any active reference slots.
        let mut all_reference_slots = Vec::new();
        if has_setup_slot {
            let setup_slot_for_begin = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(-1)
                .picture_resource(&setup_picture_resource)
                .push(&mut setup_av1_dpb_info);
            all_reference_slots.push(setup_slot_for_begin);
        }
        all_reference_slots.extend_from_slice(&reference_slots);

        // Clamp GOP values to at least 1; a value of 0 is undefined in
        // Vulkan and causes undefined behavior on some drivers (RADV).
        let gop_frames = common.config.gop_size.max(1);
        let mut codec_rate_control = vk::VideoEncodeAV1RateControlInfoKHR::default()
            .gop_frame_count(gop_frames)
            .key_frame_period(gop_frames)
            .consecutive_bipredictive_frame_count(0)
            .temporal_layer_count(1);
        let b_frames = common.config.b_frames;
        if b_frames > 0 {
            // The count of B-frames has to be below the GOP length, so a GOP
            // with no end is given the longest length there is.
            let gop_frames = if common.config.gop_size == 0 {
                u32::MAX
            } else {
                common.config.gop_size
            };
            let pattern = match common.gop.reference_pattern() {
                Some(ReferencePattern::Flat) => {
                    vk::VideoEncodeAV1RateControlFlagsKHR::REFERENCE_PATTERN_FLAT
                }
                Some(ReferencePattern::Dyadic) => {
                    vk::VideoEncodeAV1RateControlFlagsKHR::REFERENCE_PATTERN_DYADIC
                }
                None => vk::VideoEncodeAV1RateControlFlagsKHR::empty(),
            };
            codec_rate_control = codec_rate_control
                .gop_frame_count(gop_frames)
                .key_frame_period(gop_frames)
                .consecutive_bipredictive_frame_count(b_frames.min(gop_frames - 1))
                .flags(vk::VideoEncodeAV1RateControlFlagsKHR::REGULAR_GOP | pattern);
        }

        let src_picture_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(frame_extent)
            .base_array_layer(0)
            .image_view_binding(input_image_view);

        let mut encode_info = vk::VideoEncodeInfoKHR::default()
            .src_picture_resource(src_picture_resource)
            .dst_buffer(bitstream_buffer)
            .dst_buffer_offset(0)
            .dst_buffer_range(bitstream_buffer_size as u64);
        if has_setup_slot {
            encode_info = encode_info.setup_reference_slot(&setup_reference_slot);
        }
        if !reference_slots.is_empty() {
            encode_info = encode_info.reference_slots(&reference_slots);
        }
        encode_info = encode_info.push(&mut av1_picture_info);

        // A key frame with a bitrate controller resets it too.
        let first_frame = plan.is_first_frame();
        common.record_encode(EncodeCommands {
            begin_slots: &all_reference_slots,
            encode_info: &encode_info,
            rate_control: &rc,
            codec_rate_control,
            codec_layer: av1_rc_layer_info,
            first_frame,
            reset: first_frame || (is_key_frame && !rc.is_disabled()),
            post_encode_barrier: true,
        })
    }

    /// Build `ref_frame_idx`, `ref_order_hint`, `primary_ref_frame` and
    /// `refresh_frame_flags` for the current frame.
    ///
    /// `ref_frame_idx[name]` is the DPB frame-buffer slot for AV1 reference
    /// `name` (0 = LAST … 6 = ALTREF) and must mirror the
    /// `reference_name_slot_indices` handed to the encoder. `ref_order_hint` is
    /// indexed by DPB slot (not reference name) and carries each stored
    /// reference's order hint, which the decoder needs because the sequence
    /// header enables order hints. `STD_VIDEO_AV1_PRIMARY_REF_NONE` is 7.
    fn calculate_reference_frame_mapping(
        &self,
        plan: &FramePlan,
        references: &FrameReferences,
        current_dpb_slot: u8,
    ) -> ([i8; 7], [u8; 8], u8, u8) {
        const PRIMARY_REF_NONE: u8 = 7;
        if plan.is_idr() {
            // Key frame refreshes all 8 reference slots; all names point to slot 0.
            return ([0i8; 7], [0u8; 8], PRIMARY_REF_NONE, 0xFFu8);
        }
        if self.references.is_empty() {
            return ([0i8; 7], [0u8; 8], PRIMARY_REF_NONE, 0x00u8);
        }

        let mut ref_order_hint = [0u8; 8];
        // The names this frame predicts from match `reference_name_slot_indices`.
        // The bitstream needs a slot for every name, so the others point at
        // a reference on their side, which no block then uses.
        let names = references.names;
        let ref_frame_idx = ref_frame_idx(references);
        // Order hints keyed by DPB slot, for every reference still held.
        for r in self.references.as_slice() {
            ref_order_hint[r.dpb_slot as usize] = super::order_hint(r.poc) as u8;
        }
        // Load the frame context from the most recent reference, under the
        // first name it took, and refresh only the current slot so this frame
        // becomes the newest reference.
        let primary_ref_frame = names
            .iter()
            .position(|&slot| slot >= 0)
            .map_or(PRIMARY_REF_NONE, |name| name as u8);
        // A frame that no frame predicts from goes into no frame buffer.
        let refresh_flags = if plan.is_reference() {
            1u8 << current_dpb_slot
        } else {
            0
        };
        (
            ref_frame_idx,
            ref_order_hint,
            primary_ref_frame,
            refresh_flags,
        )
    }
}
