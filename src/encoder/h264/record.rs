//! H.264 per-frame encode recording: builds the StdVideo* graph for one frame
//! and submits it. This is the part that genuinely differs from the other codecs.

use super::H264;

use crate::encoder::codec::{EncoderCommon, FramePlan, RateControlPlan};
use crate::encoder::pipeline::EncodeFuture;
use crate::encoder::resources::{
    end_timestamp, prepare_encode_command_buffer, record_dpb_barriers, reset_start_timestamp,
};
use crate::error::{PixelForgeError, Result};
use ash::vk;
use ash::vk::TaggedStructure;
use tracing::debug;

impl H264 {
    pub(super) fn record(
        &mut self,
        common: &mut EncoderCommon,
        plan: &FramePlan,
    ) -> Result<EncodeFuture> {
        let is_idr = plan.is_idr();
        let is_reference = plan.is_reference();
        let is_b_frame = plan.is_b_frame();
        let pic_order_cnt = plan.pic_order_cnt();
        let frame_num = self.frame_num_syntax;

        let slot = common.pipeline.current();
        let command_buffer = slot.encode_command_buffer;
        let query_pool = slot.query_pool;
        let timestamp_query_pool = slot.timestamp_query_pool;
        let bitstream_buffer = slot.bitstream_buffer;
        let bitstream_buffer_size = slot.bitstream_buffer_size;
        let input_image_view = slot.input_image_view;

        debug!(
            "h264 record: frame_num={}, poc={}, is_idr={}, refs_len={}, cur_slot={}",
            frame_num,
            pic_order_cnt,
            is_idr,
            self.l0_references.len(),
            common.current_dpb_slot
        );

        let rc = RateControlPlan::new(&common.config, 26);

        // Reference frame invalidation: emit MMCO "unmark short-term" operations
        // for references dropped since the last frame, so the decoder evicts the
        // same pictures we did and its default reference list re-anchors to the
        // surviving reference. Only valid on a non-IDR reference picture (an IDR
        // resets the DPB on its own). Kept alive for the whole `record` call so
        // the pointer handed to the encoder stays valid.
        let ref_pic_marking_ops: Vec<ash::vk::native::StdVideoEncodeH264RefPicMarkingEntry> =
            if !is_idr && is_reference && !self.pending_unmark_frame_nums.is_empty() {
                self.pending_unmark_frame_nums
                    .iter()
                    .map(|&tainted_frame_num| {
                        // CurrPicNum is frame_num for a (non-field) frame. A
                        // short-term ref's PicNumX is its FrameNumWrap; a tainted
                        // frame_num greater than the current one has wrapped past
                        // MaxFrameNum (256 here, log2_max_frame_num_minus4 = 4).
                        let curr = frame_num as i32;
                        let mut pic_num_x = tainted_frame_num as i32;
                        if pic_num_x > curr {
                            pic_num_x -= 256;
                        }
                        let diff = (curr - pic_num_x).max(1);
                        ash::vk::native::StdVideoEncodeH264RefPicMarkingEntry {
                            memory_management_control_operation:
                                ash::vk::native::StdVideoH264MemMgmtControlOp_STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_UNMARK_SHORT_TERM,
                            difference_of_pic_nums_minus1: (diff - 1) as u16,
                            long_term_pic_num: 0,
                            long_term_frame_idx: 0,
                            max_long_term_frame_idx_plus1: 0,
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            };
        let use_adaptive_marking = !ref_pic_marking_ops.is_empty();

        // Prepare command buffer and transition DPB images for encode.
        unsafe {
            prepare_encode_command_buffer(common.device(), command_buffer, query_pool)?;
        }
        let ref_dpb_slots: Vec<u8> = self.l0_references.iter().map(|r| r.dpb_slot).collect();
        unsafe {
            record_dpb_barriers(
                common.device(),
                command_buffer,
                &common.dpb_images,
                common.use_layered_dpb,
                common.current_dpb_slot,
                &ref_dpb_slots,
                common.dpb_slot_active[common.current_dpb_slot as usize],
            );
        }

        let slice_type = if is_idr {
            ash::vk::native::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_I
        } else if is_b_frame {
            ash::vk::native::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_B
        } else {
            ash::vk::native::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_P
        };

        let picture_type = if is_idr {
            ash::vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR
        } else if is_b_frame {
            ash::vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_B
        } else {
            ash::vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P
        };

        // Build StdVideoEncodeH264SliceHeader.
        // num_ref_idx_active_override_flag is only present in P/B/SP slice headers per the H.264
        // spec. For P/B slices we set it to 1 so each slice signals the actual available
        // reference count instead of relying on the PPS default, preventing "Missing reference
        // picture" errors when the DPB is not yet full.
        let use_ref_override = !is_idr as u32;
        let slice_header_flags = ash::vk::native::StdVideoEncodeH264SliceHeaderFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeH264SliceHeaderFlags::new_bitfield_1(
                0,                // direct_spatial_mv_pred_flag
                use_ref_override, // num_ref_idx_active_override_flag
                0,                // reserved
            ),
        };

        let slice_qp_delta = match common.config.rate_control_mode {
            crate::encoder::RateControlMode::Cqp | crate::encoder::RateControlMode::Disabled => {
                (common.config.quality_level as i32 - 26) as i8
            }
            _ => 0,
        };

        let slice_header = ash::vk::native::StdVideoEncodeH264SliceHeader {
            flags: slice_header_flags,
            first_mb_in_slice: 0,
            slice_type,
            slice_alpha_c0_offset_div2: 0,
            slice_beta_offset_div2: 0,
            slice_qp_delta,
            reserved1: 0,
            cabac_init_idc:
                ash::vk::native::StdVideoH264CabacInitIdc_STD_VIDEO_H264_CABAC_INIT_IDC_0,
            disable_deblocking_filter_idc: ash::vk::native::StdVideoH264DisableDeblockingFilterIdc_STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_ENABLED,
            pWeightTable: std::ptr::null(),
        };

        let picture_info_flags = ash::vk::native::StdVideoEncodeH264PictureInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeH264PictureInfoFlags::new_bitfield_1(
                if is_idr { 1 } else { 0 },       // IdrPicFlag
                if is_reference { 1 } else { 0 }, // is_reference
                0,                                // no_output_of_prior_pics_flag
                0,                                // long_term_reference_flag
                use_adaptive_marking as u32,      // adaptive_ref_pic_marking_mode_flag
                0,                                // reserved
            ),
        };

        // For P-frames, we need a reference list.
        // STD_VIDEO_H264_NO_REFERENCE_PICTURE = 0xFF.
        const NO_REFERENCE_PICTURE: u8 = 0xFF;
        let mut ref_list0: [u8; 32] = [NO_REFERENCE_PICTURE; 32];
        let mut ref_list1: [u8; 32] = [NO_REFERENCE_PICTURE; 32];

        let ref_lists_info_flags = ash::vk::native::StdVideoEncodeH264ReferenceListsInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeH264ReferenceListsInfoFlags::new_bitfield_1(
                0, // ref_pic_list_modification_flag_l0
                0, // ref_pic_list_modification_flag_l1
                0, // reserved
            ),
        };

        // Set up reference lists for P-frames and B-frames.
        let (num_ref_l0, num_ref_l1) = if is_b_frame && self.has_backward_reference {
            if let Some(first_ref) = self.l0_references.first() {
                ref_list0[0] = first_ref.dpb_slot;
                ref_list1[0] = self.backward_reference_dpb_slot;
                (1, 1)
            } else {
                (0, 0)
            }
        } else if !is_idr && !self.l0_references.is_empty() {
            let actual_count = self
                .l0_references
                .len()
                .min(self.active_reference_count as usize)
                .min(32);
            for (i, ref_info) in self.l0_references.iter().take(actual_count).enumerate() {
                ref_list0[i] = ref_info.dpb_slot;
            }
            (actual_count, 0)
        } else {
            (0, 0)
        };

        let ref_lists_info = ash::vk::native::StdVideoEncodeH264ReferenceListsInfo {
            flags: ref_lists_info_flags,
            num_ref_idx_l0_active_minus1: if num_ref_l0 > 0 {
                (num_ref_l0 - 1) as u8
            } else {
                0
            },
            num_ref_idx_l1_active_minus1: if num_ref_l1 > 0 {
                (num_ref_l1 - 1) as u8
            } else {
                0
            },
            RefPicList0: ref_list0,
            RefPicList1: ref_list1,
            refList0ModOpCount: 0,
            refList1ModOpCount: 0,
            refPicMarkingOpCount: ref_pic_marking_ops.len() as u8,
            reserved1: [0; 7],
            pRefList0ModOperations: std::ptr::null(),
            pRefList1ModOperations: std::ptr::null(),
            pRefPicMarkingOperations: if ref_pic_marking_ops.is_empty() {
                std::ptr::null()
            } else {
                ref_pic_marking_ops.as_ptr()
            },
        };

        let picture_info = ash::vk::native::StdVideoEncodeH264PictureInfo {
            flags: picture_info_flags,
            seq_parameter_set_id: 0,
            pic_parameter_set_id: 0,
            idr_pic_id: self.idr_pic_id as u16,
            primary_pic_type: picture_type,
            frame_num,
            PicOrderCnt: pic_order_cnt,
            temporal_id: 0,
            reserved1: [0; 3],
            pRefLists: if !is_idr && (!self.l0_references.is_empty() || use_adaptive_marking) {
                &ref_lists_info
            } else {
                std::ptr::null()
            },
        };

        let constant_qp = if rc.is_disabled() { rc.qp as i32 } else { 0 };
        let nalu_slice_entries = [vk::VideoEncodeH264NaluSliceInfoKHR::default()
            .constant_qp(constant_qp)
            .std_slice_header(&slice_header)];

        let mut h264_picture_info = vk::VideoEncodeH264PictureInfoKHR::default()
            .nalu_slice_entries(&nalu_slice_entries)
            .std_picture_info(&picture_info);

        let coded_extent = vk::Extent2D {
            width: common.aligned_width,
            height: common.aligned_height,
        };

        let src_picture_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(coded_extent)
            .base_array_layer(0)
            .image_view_binding(input_image_view);

        let setup_picture_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(coded_extent)
            .base_array_layer(0)
            .image_view_binding(common.dpb_image_views[common.current_dpb_slot as usize]);

        let std_reference_info_flags = ash::vk::native::StdVideoEncodeH264ReferenceInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeH264ReferenceInfoFlags::new_bitfield_1(
                0, // used_for_long_term_reference
                0, // reserved
            ),
        };

        // Vectors hold the data so pointers into them stay stable.
        let mut l0_resources = Vec::with_capacity(self.l0_references.len());
        let mut l0_std_infos = Vec::with_capacity(self.l0_references.len());

        for ref_info in &self.l0_references {
            l0_resources.push(
                vk::VideoPictureResourceInfoKHR::default()
                    .coded_offset(vk::Offset2D { x: 0, y: 0 })
                    .coded_extent(coded_extent)
                    .base_array_layer(0)
                    .image_view_binding(common.dpb_image_views[ref_info.dpb_slot as usize]),
            );
            l0_std_infos.push(ash::vk::native::StdVideoEncodeH264ReferenceInfo {
                flags: std_reference_info_flags,
                primary_pic_type:
                    ash::vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P,
                FrameNum: ref_info.frame_num,
                PicOrderCnt: ref_info.poc,
                long_term_pic_num: 0,
                long_term_frame_idx: 0,
                temporal_id: 0,
            });
        }

        let mut l0_dpb_slot_infos = Vec::with_capacity(l0_std_infos.len());
        for std_info in &l0_std_infos {
            l0_dpb_slot_infos
                .push(vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(std_info));
        }

        let mut l0_slots = Vec::with_capacity(l0_resources.len());
        for (i, (resource, dpb_info)) in l0_resources
            .iter()
            .zip(l0_dpb_slot_infos.iter_mut())
            .enumerate()
        {
            let ref_info = &self.l0_references[i];
            l0_slots.push(
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(ref_info.dpb_slot as i32)
                    .picture_resource(resource)
                    .push(dpb_info),
            );
        }

        // Backward (L1) reference for B-frames.
        let (backward_resource, backward_std_info) = if is_b_frame && self.has_backward_reference {
            let image_view = common.dpb_image_views[self.backward_reference_dpb_slot as usize];
            let resource = vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D { x: 0, y: 0 })
                .coded_extent(coded_extent)
                .base_array_layer(0)
                .image_view_binding(image_view);
            let std_info = ash::vk::native::StdVideoEncodeH264ReferenceInfo {
                flags: std_reference_info_flags,
                primary_pic_type:
                    ash::vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P,
                FrameNum: self.backward_reference_frame_num,
                PicOrderCnt: self.backward_reference_poc,
                long_term_pic_num: 0,
                long_term_frame_idx: 0,
                temporal_id: 0,
            };
            (Some(resource), Some(std_info))
        } else {
            (None, None)
        };

        let mut backward_dpb_info = if let Some(ref std_info) = backward_std_info {
            vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(std_info)
        } else {
            vk::VideoEncodeH264DpbSlotInfoKHR::default()
        };

        let backward_ref_slot = if let Some(ref resource) = backward_resource {
            Some(
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(self.backward_reference_dpb_slot as i32)
                    .picture_resource(resource)
                    .push(&mut backward_dpb_info),
            )
        } else {
            None
        };

        // Reference info for the setup slot (this frame).
        let std_reference_info = ash::vk::native::StdVideoEncodeH264ReferenceInfo {
            flags: std_reference_info_flags,
            primary_pic_type: if is_idr {
                ash::vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR
            } else {
                ash::vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P
            },
            FrameNum: frame_num,
            PicOrderCnt: pic_order_cnt,
            long_term_pic_num: 0,
            long_term_frame_idx: 0,
            temporal_id: 0,
        };

        let mut h264_dpb_slot_info =
            vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(&std_reference_info);
        let setup_reference_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(common.current_dpb_slot as i32)
            .picture_resource(&setup_picture_resource)
            .push(&mut h264_dpb_slot_info);

        let mut h264_begin_dpb_slot_info =
            vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(&std_reference_info);
        let setup_slot_for_begin = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(-1)
            .picture_resource(&setup_picture_resource)
            .push(&mut h264_begin_dpb_slot_info);

        // Reference slots for the encode command.
        let mut encode_ref_slots = Vec::new();
        if is_b_frame && self.has_backward_reference {
            if let Some(l0) = l0_slots.first() {
                encode_ref_slots.push(*l0);
            }
            if let Some(l1) = backward_ref_slot {
                encode_ref_slots.push(l1);
            }
        } else if !is_idr {
            encode_ref_slots.extend_from_slice(&l0_slots);
        }

        let mut encode_info = vk::VideoEncodeInfoKHR::default()
            .dst_buffer(bitstream_buffer)
            .dst_buffer_offset(0)
            .dst_buffer_range(bitstream_buffer_size as vk::DeviceSize)
            .src_picture_resource(src_picture_resource)
            .setup_reference_slot(&setup_reference_slot);
        if !encode_ref_slots.is_empty() {
            encode_info = encode_info.reference_slots(&encode_ref_slots);
        }
        encode_info = encode_info.push(&mut h264_picture_info);

        // Reference slots for begin coding (setup slot is marked inactive, -1).
        let mut reference_slots_for_begin = vec![setup_slot_for_begin];
        if is_b_frame && self.has_backward_reference {
            if let Some(l0) = l0_slots.first() {
                reference_slots_for_begin.push(*l0);
            }
            if let Some(l1) = backward_ref_slot {
                reference_slots_for_begin.push(l1);
            }
        } else if !is_idr {
            reference_slots_for_begin.extend_from_slice(&l0_slots);
        }

        // Rate control.
        let qp_bounds = if rc.is_disabled() { rc.qp as i32 } else { 18 };
        let qp_bounds_max = if rc.is_disabled() { rc.qp as i32 } else { 42 };
        let qp_bounds_max_i = if rc.is_disabled() {
            rc.qp as i32
        } else {
            common
                .config
                .max_qp_i
                .map_or(qp_bounds_max, |qp| (qp as i32).max(qp_bounds))
        };
        let min_qp = vk::VideoEncodeH264QpKHR {
            qp_i: qp_bounds,
            qp_p: qp_bounds,
            qp_b: qp_bounds,
        };
        let max_qp = vk::VideoEncodeH264QpKHR {
            qp_i: qp_bounds_max_i,
            qp_p: qp_bounds_max,
            qp_b: qp_bounds_max,
        };

        let mut h264_rc_layer_info = vk::VideoEncodeH264RateControlLayerInfoKHR::default()
            .use_min_qp(true)
            .min_qp(min_qp)
            .use_max_qp(true)
            .max_qp(max_qp);
        let rc_layer_info = vk::VideoEncodeRateControlLayerInfoKHR::default()
            .average_bitrate(rc.average_bitrate as u64)
            .max_bitrate(rc.max_bitrate as u64)
            .frame_rate_numerator(common.config.frame_rate_numerator)
            .frame_rate_denominator(common.config.frame_rate_denominator)
            .push(&mut h264_rc_layer_info);
        let rc_layers = [rc_layer_info];

        let mut h264_rc_info = vk::VideoEncodeH264RateControlInfoKHR::default()
            .gop_frame_count(common.config.gop_size)
            .idr_period(common.config.gop_size)
            .consecutive_b_frame_count(common.config.b_frame_count);

        let mut rc_info = vk::VideoEncodeRateControlInfoKHR::default().rate_control_mode(rc.mode);
        if !rc.is_disabled() {
            rc_info = rc_info
                .layers(&rc_layers)
                .virtual_buffer_size_in_ms(common.config.virtual_buffer_size_ms)
                .initial_virtual_buffer_size_in_ms(common.config.initial_virtual_buffer_size_ms);
        }

        // Reset and write start timestamp
        reset_start_timestamp(common.device(), command_buffer, timestamp_query_pool);

        // For the first frame, configure rate control via the control command
        // after RESET rather than in begin_coding.
        let is_first_frame = plan.is_first_frame();
        let begin_info = if is_first_frame {
            vk::VideoBeginCodingInfoKHR::default()
                .video_session(common.session)
                .video_session_parameters(common.session_params)
                .reference_slots(&reference_slots_for_begin)
                .push(&mut h264_rc_info)
        } else {
            vk::VideoBeginCodingInfoKHR::default()
                .video_session(common.session)
                .video_session_parameters(common.session_params)
                .reference_slots(&reference_slots_for_begin)
                .push(&mut h264_rc_info)
                .push(&mut rc_info)
        };

        unsafe {
            (common.video_queue_fn.fp().cmd_begin_video_coding_khr)(command_buffer, &begin_info);
        }

        // RESET + RATE_CONTROL + QUALITY_LEVEL in one control command on the first
        // frame (matches FFmpeg; required for AMD RADV).
        if is_first_frame {
            let mut quality_level_info = vk::VideoEncodeQualityLevelInfoKHR::default()
                .quality_level(common.config.encode_quality_level);
            let control_info = vk::VideoCodingControlInfoKHR::default()
                .flags(
                    vk::VideoCodingControlFlagsKHR::RESET
                        | vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL
                        | vk::VideoCodingControlFlagsKHR::ENCODE_QUALITY_LEVEL,
                )
                .push(&mut rc_info)
                .push(&mut h264_rc_info)
                .push(&mut quality_level_info);
            unsafe {
                (common.video_queue_fn.fp().cmd_control_video_coding_khr)(
                    command_buffer,
                    &control_info,
                );
            }
        }

        unsafe {
            let device = common.device();
            device.cmd_begin_query(
                command_buffer,
                query_pool,
                0,
                vk::QueryControlFlags::empty(),
            );
            (common.video_encode_fn.fp().cmd_encode_video_khr)(command_buffer, &encode_info);
            device.cmd_end_query(command_buffer, query_pool, 0);

            // Write end timestamp
            end_timestamp(common.device(), command_buffer, timestamp_query_pool);

            let end_info = vk::VideoEndCodingInfoKHR::default();
            (common.video_queue_fn.fp().cmd_end_video_coding_khr)(command_buffer, &end_info);

            device
                .end_command_buffer(command_buffer)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
        }

        let future = common.submit_frame()?;
        // Clear the unmark queue only after the encode is committed to the GPU.
        // If any fallible step above had failed, the MMCO ops would still live
        // in `pending_unmark_frame_nums` so a retry re-emits them — otherwise the
        // encoder would silently desync from the decoder's reference state.
        if use_adaptive_marking {
            self.pending_unmark_frame_nums.clear();
        }
        Ok(future)
    }
}
