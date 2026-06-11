use super::H264Encoder;

use crate::encoder::gop::{GopFrameType, GopPosition};
use crate::encoder::resources::{
    prepare_encode_command_buffer, record_dpb_barriers, submit_encode_only,
    MIN_BITSTREAM_BUFFER_SIZE,
};
use crate::error::{PixelForgeError, Result};
use ash::vk;
use tracing::debug;

impl H264Encoder {
    /// Records and submits the encode commands for a single frame to the
    /// current slot. Does NOT wait for completion or read the bitstream —
    /// the caller drains the slot's prior in-flight encode before calling
    /// this, and the slot is marked in_flight so a later call can drain the
    /// submission made here.
    pub(super) fn encode_frame_internal(
        &mut self,
        gop_position: &GopPosition,
        frame_num: u32,
        pic_order_cnt: i32,
        is_idr: bool,
    ) -> Result<()> {
        let is_b_frame = gop_position.frame_type == GopFrameType::B;
        let is_reference = gop_position.is_reference;

        debug!(
            "encode_frame_internal: frame_num={}, poc={}, is_idr={}, refs_len={}, current_dpb_slot={}",
            frame_num,
            pic_order_cnt,
            is_idr,
            self.l0_references.len(),
            self.current_dpb_slot
        );

        // Rate control setup.
        let (rc_mode, average_bitrate, max_bitrate, qp) = match self.config.rate_control_mode {
            crate::encoder::RateControlMode::Cqp | crate::encoder::RateControlMode::Disabled => (
                vk::VideoEncodeRateControlModeFlagsKHR::DISABLED,
                0,
                0,
                self.config.quality_level as i32,
            ),
            crate::encoder::RateControlMode::Cbr => (
                vk::VideoEncodeRateControlModeFlagsKHR::CBR,
                self.config.target_bitrate,
                self.config.target_bitrate,
                26, // Default QP for rate control to adjust from
            ),
            crate::encoder::RateControlMode::Vbr => (
                vk::VideoEncodeRateControlModeFlagsKHR::VBR,
                self.config.target_bitrate,
                self.config.max_bitrate,
                26, // Default QP for rate control to adjust from
            ),
        };

        // Prepare command buffer for recording.
        unsafe {
            prepare_encode_command_buffer(
                self.context.device(),
                self.slots[self.current_slot].encode_command_buffer,
                self.slots[self.current_slot].query_pool,
            )?;
        }

        // Transition DPB images for encode.
        let ref_dpb_slots: Vec<u8> = self.l0_references.iter().map(|r| r.dpb_slot).collect();
        unsafe {
            record_dpb_barriers(
                self.context.device(),
                self.slots[self.current_slot].encode_command_buffer,
                &self.dpb_images,
                self.use_layered_dpb,
                self.current_dpb_slot,
                &ref_dpb_slots,
                self.dpb_slot_active[self.current_dpb_slot as usize],
            );
        }

        // Set up H.264 specific encode info.
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

        let slice_qp_delta = match self.config.rate_control_mode {
            crate::encoder::RateControlMode::Cqp | crate::encoder::RateControlMode::Disabled => {
                (self.config.quality_level as i32 - 26) as i8
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

        // Build StdVideoEncodeH264PictureInfo.
        let picture_info_flags = ash::vk::native::StdVideoEncodeH264PictureInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeH264PictureInfoFlags::new_bitfield_1(
                if is_idr { 1 } else { 0 },       // IdrPicFlag
                if is_reference { 1 } else { 0 }, // is_reference
                0,                                // no_output_of_prior_pics_flag
                0,                                // long_term_reference_flag
                0,                                // adaptive_ref_pic_marking_mode_flag
                0,                                // reserved
            ),
        };

        // For P-frames, we need a reference list.
        // STD_VIDEO_H264_NO_REFERENCE_PICTURE = 0xFF.
        const NO_REFERENCE_PICTURE: u8 = 0xFF;
        let mut ref_list0: [u8; 32] = [NO_REFERENCE_PICTURE; 32];
        let mut ref_list1: [u8; 32] = [NO_REFERENCE_PICTURE; 32];

        // Reference list modification operations (not used)
        let _ref_pic_list_mod_flags = ash::vk::native::StdVideoEncodeH264RefPicMarkingEntry {
            memory_management_control_operation:
                ash::vk::native::StdVideoH264MemMgmtControlOp_STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_END,
            difference_of_pic_nums_minus1: 0,
            long_term_pic_num: 0,
            long_term_frame_idx: 0,
            max_long_term_frame_idx_plus1: 0,
        };

        let ref_lists_info_flags = ash::vk::native::StdVideoEncodeH264ReferenceListsInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeH264ReferenceListsInfoFlags::new_bitfield_1(
                0, // ref_pic_list_modification_flag_l0
                0, // ref_pic_list_modification_flag_l1
                0, // reserved
            ),
        };

        // Set up reference lists for P-frames and B-frames.
        // P-frames: L0 only (reference previous frame)
        // B-frames: L0 (forward/past) and L1 (backward/future)
        let (num_ref_l0, num_ref_l1) = if is_b_frame && self.has_backward_reference {
            // B-frame: both L0 and L1 references.
            if let Some(first_ref) = self.l0_references.first() {
                ref_list0[0] = first_ref.dpb_slot;
                ref_list1[0] = self.backward_reference_dpb_slot;
                (1, 1)
            } else {
                (0, 0)
            }
        } else if !is_idr && !self.l0_references.is_empty() {
            // P-frame: L0 reference list using actual available references,
            // clamped to the negotiated active count (which is itself ≤32 at init time).
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
            // IDR: no references.
            (0, 0)
        };

        // num_ref_idx_l0_active_minus1 tells the driver how many entries in RefPicList0
        // are valid and is encoded into the slice header (because we set
        // num_ref_idx_active_override_flag=1 above).
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
            refPicMarkingOpCount: 0,
            reserved1: [0; 7],
            pRefList0ModOperations: std::ptr::null(),
            pRefList1ModOperations: std::ptr::null(),
            pRefPicMarkingOperations: std::ptr::null(),
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
            pRefLists: if !is_idr && !self.l0_references.is_empty() {
                &ref_lists_info
            } else {
                std::ptr::null()
            },
        };

        // Create slice NAL unit entry.
        // constant_qp should only be set when rate control is DISABLED.
        let constant_qp = if rc_mode == vk::VideoEncodeRateControlModeFlagsKHR::DISABLED {
            qp
        } else {
            0
        };
        let nalu_slice_entries = [vk::VideoEncodeH264NaluSliceInfoKHR::default()
            .constant_qp(constant_qp)
            .std_slice_header(&slice_header)];

        // Create H.264 picture info.
        let mut h264_picture_info = vk::VideoEncodeH264PictureInfoKHR::default()
            .nalu_slice_entries(&nalu_slice_entries)
            .std_picture_info(&picture_info);

        // Set up source picture resource.
        let src_picture_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: self.aligned_width,
                height: self.aligned_height,
            })
            .base_array_layer(0)
            .image_view_binding(self.slots[self.current_slot].input_image_view);

        // Set up DPB slot for reconstructed picture (setup slot)
        let setup_picture_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: self.aligned_width,
                height: self.aligned_height,
            })
            .base_array_layer(0)
            .image_view_binding(self.dpb_image_views[self.current_dpb_slot as usize]);

        // Set up reference picture resources and info.

        let std_reference_info_flags = ash::vk::native::StdVideoEncodeH264ReferenceInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeH264ReferenceInfoFlags::new_bitfield_1(
                0, // used_for_long_term_reference
                0, // reserved
            ),
        };

        // We use vectors to hold the data to ensure stable memory addresses for pointers.
        let mut l0_resources = Vec::with_capacity(self.l0_references.len());
        let mut l0_std_infos = Vec::with_capacity(self.l0_references.len());

        // 1. Populate data for L0 references (P-frames and B-frames)
        for ref_info in &self.l0_references {
            l0_resources.push(
                vk::VideoPictureResourceInfoKHR::default()
                    .coded_offset(vk::Offset2D { x: 0, y: 0 })
                    .coded_extent(vk::Extent2D {
                        width: self.aligned_width,
                        height: self.aligned_height,
                    })
                    .base_array_layer(0)
                    .image_view_binding(self.dpb_image_views[ref_info.dpb_slot as usize]),
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

        // 2. Create DPB slot infos for L0.
        // MUST be done after l0_std_infos is fully populated so addresses don't change.
        let mut l0_dpb_slot_infos = Vec::with_capacity(l0_std_infos.len());
        for std_info in &l0_std_infos {
            l0_dpb_slot_infos
                .push(vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(std_info));
        }

        // 3. Create the L0 reference slots.
        let mut l0_slots = Vec::with_capacity(l0_resources.len());
        for (i, (resource, dpb_info)) in l0_resources
            .iter()
            .zip(l0_dpb_slot_infos.iter_mut())
            .enumerate()
        {
            let ref_info = &self.l0_references[i];
            let mut slot = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(ref_info.dpb_slot as i32)
                .picture_resource(resource);
            slot.p_next = (dpb_info as *mut vk::VideoEncodeH264DpbSlotInfoKHR).cast();
            l0_slots.push(slot);
        }

        // 4. Handle Backward Ref (L1) for B-frames
        let (backward_resource, backward_std_info) = if is_b_frame && self.has_backward_reference {
            let image_view = self.dpb_image_views[self.backward_reference_dpb_slot as usize];
            let resource = vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D { x: 0, y: 0 })
                .coded_extent(vk::Extent2D {
                    width: self.aligned_width,
                    height: self.aligned_height,
                })
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
            let mut slot = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(self.backward_reference_dpb_slot as i32)
                .picture_resource(resource);
            slot.p_next = (&mut backward_dpb_info as *mut vk::VideoEncodeH264DpbSlotInfoKHR).cast();
            Some(slot)
        } else {
            None
        };

        // Create H.264 reference info for the setup slot (this frame being encoded)
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

        // Create H.264 DPB slot info for setup.
        let mut h264_dpb_slot_info =
            vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(&std_reference_info);

        let mut setup_reference_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(self.current_dpb_slot as i32)
            .picture_resource(&setup_picture_resource);
        setup_reference_slot.p_next =
            (&mut h264_dpb_slot_info as *mut vk::VideoEncodeH264DpbSlotInfoKHR).cast();

        // Also create a setup slot for begin_info (slotIndex = -1)
        let mut h264_begin_dpb_slot_info =
            vk::VideoEncodeH264DpbSlotInfoKHR::default().std_reference_info(&std_reference_info);

        let mut setup_slot_for_begin = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(-1) // Marked as inactive for begin
            .picture_resource(&setup_picture_resource);
        setup_slot_for_begin.p_next =
            (&mut h264_begin_dpb_slot_info as *mut vk::VideoEncodeH264DpbSlotInfoKHR).cast();

        // Collect final reference slots for VIDEO ENCODE INFO
        let mut encode_ref_slots = Vec::new();
        if is_b_frame && self.has_backward_reference {
            // B-Frames: Use first L0 (forward) + L1 (backward)
            if let Some(l0) = l0_slots.first() {
                encode_ref_slots.push(*l0);
            }
            if let Some(l1) = backward_ref_slot {
                encode_ref_slots.push(l1);
            }
        } else if !is_idr {
            // P-Frames: Use all L0
            encode_ref_slots.extend_from_slice(&l0_slots);
        }

        let mut encode_info = vk::VideoEncodeInfoKHR::default()
            .dst_buffer(self.slots[self.current_slot].bitstream_buffer)
            .dst_buffer_offset(0)
            .dst_buffer_range(MIN_BITSTREAM_BUFFER_SIZE as vk::DeviceSize)
            .src_picture_resource(src_picture_resource)
            .setup_reference_slot(&setup_reference_slot);

        if !encode_ref_slots.is_empty() {
            encode_info = encode_info.reference_slots(&encode_ref_slots);
        }

        encode_info.p_next =
            (&mut h264_picture_info as *mut vk::VideoEncodeH264PictureInfoKHR).cast();

        // Collect reference slots for BEGIN VIDEO CODING
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

        let min_qp_val = if rc_mode == vk::VideoEncodeRateControlModeFlagsKHR::DISABLED
            || self.config.rate_control_mode == crate::encoder::RateControlMode::Cqp
            || self.config.rate_control_mode == crate::encoder::RateControlMode::Disabled
        {
            qp // Clamp to fixed QP for CQP/Disabled simulation
        } else {
            18 // Allow high quality for H.264
        };
        let max_qp_val = if rc_mode == vk::VideoEncodeRateControlModeFlagsKHR::DISABLED
            || self.config.rate_control_mode == crate::encoder::RateControlMode::Cqp
            || self.config.rate_control_mode == crate::encoder::RateControlMode::Disabled
        {
            qp // Clamp to fixed QP for CQP/Disabled simulation
        } else {
            42 // Allow lower quality when needed
        };

        let min_qp = vk::VideoEncodeH264QpKHR {
            qp_i: min_qp_val,
            qp_p: min_qp_val,
            qp_b: min_qp_val,
        };

        let max_qp = vk::VideoEncodeH264QpKHR {
            qp_i: max_qp_val,
            qp_p: max_qp_val,
            qp_b: max_qp_val,
        };

        let mut h264_rc_layer_info = vk::VideoEncodeH264RateControlLayerInfoKHR::default()
            .use_min_qp(true)
            .min_qp(min_qp)
            .use_max_qp(true)
            .max_qp(max_qp);

        let mut rc_layer_info = vk::VideoEncodeRateControlLayerInfoKHR::default()
            .average_bitrate(average_bitrate as u64)
            .max_bitrate(max_bitrate as u64)
            .frame_rate_numerator(self.config.frame_rate_numerator)
            .frame_rate_denominator(self.config.frame_rate_denominator);
        rc_layer_info.p_next =
            (&mut h264_rc_layer_info as *mut vk::VideoEncodeH264RateControlLayerInfoKHR).cast();

        let rc_layers = [rc_layer_info];

        let mut h264_rc_info = vk::VideoEncodeH264RateControlInfoKHR::default()
            .gop_frame_count(self.config.gop_size)
            .idr_period(self.config.gop_size)
            .consecutive_b_frame_count(self.config.b_frame_count);

        let mut rc_info = vk::VideoEncodeRateControlInfoKHR::default().rate_control_mode(rc_mode);

        if rc_mode != vk::VideoEncodeRateControlModeFlagsKHR::DISABLED {
            rc_info = rc_info
                .layers(&rc_layers)
                .virtual_buffer_size_in_ms(self.config.virtual_buffer_size_ms)
                .initial_virtual_buffer_size_in_ms(self.config.initial_virtual_buffer_size_ms);
            rc_info.p_next = &mut h264_rc_info as *mut _ as *mut std::ffi::c_void;
        }

        // Begin video coding.
        // For the first frame, don't include rate control in begin_coding - set it via control command after RESET.
        let is_first_frame = self.encode_frame_num == 0;

        let begin_info = if is_first_frame {
            vk::VideoBeginCodingInfoKHR::default()
                .video_session(self.session)
                .video_session_parameters(self.session_params)
                .reference_slots(&reference_slots_for_begin)
        } else {
            let mut info = vk::VideoBeginCodingInfoKHR::default()
                .video_session(self.session)
                .video_session_parameters(self.session_params)
                .reference_slots(&reference_slots_for_begin);
            info.p_next = (&mut rc_info as *mut vk::VideoEncodeRateControlInfoKHR).cast();
            info
        };

        unsafe {
            (self.video_queue_fn.fp().cmd_begin_video_coding_khr)(
                self.slots[self.current_slot].encode_command_buffer,
                &begin_info,
            );
        }

        // Reset video coding state for the first frame.
        // Combine RESET + RATE_CONTROL + QUALITY_LEVEL into a single control command.
        // This matches FFmpeg's approach and is required for AMD RADV.
        if is_first_frame {
            let mut quality_level_info =
                vk::VideoEncodeQualityLevelInfoKHR::default().quality_level(0);
            quality_level_info.p_next =
                (&mut rc_info as *mut vk::VideoEncodeRateControlInfoKHR).cast();

            let mut control_info = vk::VideoCodingControlInfoKHR::default().flags(
                vk::VideoCodingControlFlagsKHR::RESET
                    | vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL
                    | vk::VideoCodingControlFlagsKHR::ENCODE_QUALITY_LEVEL,
            );
            control_info.p_next =
                (&mut quality_level_info as *mut vk::VideoEncodeQualityLevelInfoKHR).cast();

            unsafe {
                (self.video_queue_fn.fp().cmd_control_video_coding_khr)(
                    self.slots[self.current_slot].encode_command_buffer,
                    &control_info,
                );
            }
        }

        // Begin query.
        unsafe {
            self.context.device().cmd_begin_query(
                self.slots[self.current_slot].encode_command_buffer,
                self.slots[self.current_slot].query_pool,
                0,
                vk::QueryControlFlags::empty(),
            );
        }

        // Encode
        unsafe {
            (self.video_encode_fn.fp().cmd_encode_video_khr)(
                self.slots[self.current_slot].encode_command_buffer,
                &encode_info,
            );
        }

        // End query.
        unsafe {
            self.context.device().cmd_end_query(
                self.slots[self.current_slot].encode_command_buffer,
                self.slots[self.current_slot].query_pool,
                0,
            );
        }

        // End video coding.
        let end_info = vk::VideoEndCodingInfoKHR::default();
        unsafe {
            (self.video_queue_fn.fp().cmd_end_video_coding_khr)(
                self.slots[self.current_slot].encode_command_buffer,
                &end_info,
            );
        }

        // End command buffer.
        unsafe {
            self.context
                .device()
                .end_command_buffer(self.slots[self.current_slot].encode_command_buffer)
        }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

        // Submit, wait, and read bitstream.
        let encode_queue = self.context.video_encode_queue().ok_or_else(|| {
            PixelForgeError::NoSuitableDevice("No video encode queue available".to_string())
        })?;

        unsafe {
            submit_encode_only(
                self.context.device(),
                self.slots[self.current_slot].encode_command_buffer,
                self.slots[self.current_slot].encode_fence,
                encode_queue,
                None,
            )?;
        }

        // Mark DPB slot as active.
        self.dpb_slot_active[self.current_dpb_slot as usize] = true;

        // Mark slot as in-flight; bitstream is drained on the next encode()
        // call that targets this slot.
        self.slots[self.current_slot].in_flight = true;

        Ok(())
    }
}
