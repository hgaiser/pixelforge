//! H.265/HEVC encoder internal encoding implementation.
//!
//! This module handles the actual frame encoding using Vulkan Video.

use super::H265Encoder;

use crate::encoder::gop::{GopFrameType, GopPosition};
use crate::encoder::resources::{
    prepare_encode_command_buffer, record_dpb_barriers, record_post_encode_dpb_barrier,
    submit_encode_only, MIN_BITSTREAM_BUFFER_SIZE,
};
use crate::error::{PixelForgeError, Result};
use ash::vk;
use tracing::debug;

impl H265Encoder {
    /// Records and submits the encode commands for a single frame to the
    /// current slot. Does NOT wait for completion or read the bitstream —
    /// the caller drains the slot's prior in-flight encode before calling
    /// this, and the slot is marked in_flight so a later call can drain the
    /// submission made here.
    pub(super) fn encode_frame_internal(
        &mut self,
        gop_position: &GopPosition,
        pic_order_cnt: i32,
        is_idr: bool,
    ) -> Result<()> {
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

        // Determine picture type.
        let is_b_frame = gop_position.frame_type == GopFrameType::B;
        let is_reference = gop_position.is_reference;
        let slice_type = if is_idr {
            ash::vk::native::StdVideoH265SliceType_STD_VIDEO_H265_SLICE_TYPE_I
        } else if is_b_frame {
            ash::vk::native::StdVideoH265SliceType_STD_VIDEO_H265_SLICE_TYPE_B
        } else {
            ash::vk::native::StdVideoH265SliceType_STD_VIDEO_H265_SLICE_TYPE_P
        };

        let picture_type = if is_idr {
            ash::vk::native::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_IDR
        } else if is_b_frame {
            ash::vk::native::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_B
        } else {
            ash::vk::native::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_P
        };

        // Build StdVideoEncodeH265SliceSegmentHeader.
        let slice_header_flags = ash::vk::native::StdVideoEncodeH265SliceSegmentHeaderFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeH265SliceSegmentHeaderFlags::new_bitfield_1(
                1, // first_slice_segment_in_pic_flag
                0, // dependent_slice_segment_flag
                1, // slice_sao_luma_flag
                1, // slice_sao_chroma_flag
                1, // num_ref_idx_active_override_flag
                0, // mvd_l1_zero_flag
                0, // cabac_init_flag
                0, // cu_chroma_qp_offset_enabled_flag
                0, // deblocking_filter_override_flag
                0, // slice_deblocking_filter_disabled_flag
                0, // collocated_from_l0_flag
                0, // slice_loop_filter_across_slices_enabled_flag
                0, // reserved
            ),
        };

        let slice_header = ash::vk::native::StdVideoEncodeH265SliceSegmentHeader {
            flags: slice_header_flags,
            slice_type,
            slice_segment_address: 0,
            collocated_ref_idx: 0,
            MaxNumMergeCand: 5,
            slice_cb_qp_offset: 0,
            slice_cr_qp_offset: 0,
            slice_beta_offset_div2: 0,
            slice_tc_offset_div2: 0,
            slice_act_y_qp_offset: 0,
            slice_act_cb_qp_offset: 0,
            slice_act_cr_qp_offset: 0,
            slice_qp_delta: 0,
            reserved1: 0,
            pWeightTable: std::ptr::null(),
        };

        // Build short-term reference picture set.
        // For B-frames, we need both negative (past) and positive (future) references.
        // For P-frames, we only need negative references.
        let mut delta_poc_s0_minus1 = [0u16; 16]; // negative refs (past)
        let mut delta_poc_s1_minus1 = [0u16; 16]; // positive refs (future)
        let mut num_negative_pics: u8 = 0;
        let mut num_positive_pics: u8 = 0;
        let mut used_by_curr_pic_s0_flag: u16 = 0;
        let mut used_by_curr_pic_s1_flag: u16 = 0;

        if !is_idr && !self.l0_references.is_empty() {
            // L0 references (negative/past)
            // Calculate max_poc from config (2^(log2_max_pic_order_cnt_lsb_minus4 + 4) * 2).
            // With log2_max_pic_order_cnt_lsb_minus4 = 4, max_poc = 2^8 * 2 = 512.
            let max_poc = 1i32 << 9; // 512

            let mut prev_delta_poc = 0;

            for (i, ref_info) in self.l0_references.iter().enumerate() {
                if i >= 15 {
                    break;
                } // limit to 15 neg pics

                // Calculate delta POC with wraparound handling.
                // delta_poc should be negative (reference is in the past).
                let mut delta_poc = ref_info.poc - pic_order_cnt;
                // If delta_poc is positive and large, it means POC wrapped around.
                // Adjust by subtracting max_poc to get the correct negative delta.
                if delta_poc > max_poc / 2 {
                    delta_poc -= max_poc;
                } else if delta_poc < -max_poc / 2 {
                    delta_poc += max_poc;
                }

                let diff = prev_delta_poc - delta_poc;
                delta_poc_s0_minus1[num_negative_pics as usize] = (diff - 1).max(0) as u16;
                prev_delta_poc = delta_poc;

                used_by_curr_pic_s0_flag |= 1 << num_negative_pics;
                num_negative_pics += 1;
            }

            // For B-frames, add L1 reference (positive/future)
            if is_b_frame && self.has_backward_reference {
                let delta_poc_l1 = self.backward_reference_poc - pic_order_cnt;
                // delta_poc_s1 should be positive
                delta_poc_s1_minus1[0] = (delta_poc_l1 - 1).max(0) as u16;
                num_positive_pics = 1;
                used_by_curr_pic_s1_flag = 1; // First positive reference is used
            }
        }

        let rps_flags = ash::vk::native::StdVideoH265ShortTermRefPicSetFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoH265ShortTermRefPicSetFlags::new_bitfield_1(0, 0),
            __bindgen_padding_0: [0; 3],
        };

        let frame_rps = ash::vk::native::StdVideoH265ShortTermRefPicSet {
            flags: rps_flags,
            delta_idx_minus1: 0,
            use_delta_flag: 0,
            abs_delta_rps_minus1: 0,
            used_by_curr_pic_flag: 0,
            used_by_curr_pic_s0_flag,
            used_by_curr_pic_s1_flag,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
            num_negative_pics,
            num_positive_pics,
            delta_poc_s0_minus1,
            delta_poc_s1_minus1,
        };

        // Empty RPS for IDR frames.
        let empty_rps_flags = ash::vk::native::StdVideoH265ShortTermRefPicSetFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoH265ShortTermRefPicSetFlags::new_bitfield_1(0, 0),
            __bindgen_padding_0: [0; 3],
        };

        let empty_rps = ash::vk::native::StdVideoH265ShortTermRefPicSet {
            flags: empty_rps_flags,
            delta_idx_minus1: 0,
            use_delta_flag: 0,
            abs_delta_rps_minus1: 0,
            used_by_curr_pic_flag: 0,
            used_by_curr_pic_s0_flag: 0,
            used_by_curr_pic_s1_flag: 0,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
            num_negative_pics: 0,
            num_positive_pics: 0,
            delta_poc_s0_minus1: [0; 16],
            delta_poc_s1_minus1: [0; 16],
        };

        // Build picture info flags.
        let picture_info_flags = ash::vk::native::StdVideoEncodeH265PictureInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeH265PictureInfoFlags::new_bitfield_1(
                if is_reference { 1 } else { 0 }, // is_reference
                if is_idr { 1 } else { 0 },       // IrapPicFlag
                0,                                // used_for_long_term_reference
                0,                                // discardable_flag
                0,                                // cross_layer_bla_flag
                1,                                // pic_output_flag
                if is_idr { 1 } else { 0 },       // no_output_of_prior_pics_flag
                0,                                // short_term_ref_pic_set_sps_flag
                0,                                // slice_temporal_mvp_enabled_flag
                0,                                // reserved
            ),
        };

        // Set up reference lists.
        const NO_REFERENCE_PICTURE: u8 = 0xFF;
        let mut ref_list0: [u8; 15] = [NO_REFERENCE_PICTURE; 15];
        let mut ref_list1: [u8; 15] = [NO_REFERENCE_PICTURE; 15];

        let (num_ref_l0, num_ref_l1) = if is_b_frame && self.has_backward_reference {
            // B-frame logic
            if let Some(first_ref) = self.l0_references.first() {
                ref_list0[0] = first_ref.dpb_slot;
                ref_list1[0] = self.backward_reference_dpb_slot;
                (1, 1)
            } else {
                (0, 0)
            }
        } else if !is_idr && !self.l0_references.is_empty() {
            // P-frame logic
            let count = self.l0_references.len();
            for (i, ref_info) in self.l0_references.iter().enumerate() {
                if i < 15 {
                    ref_list0[i] = ref_info.dpb_slot;
                }
            }

            // We use all available references as active
            (count.min(15), 0)
        } else {
            (0, 0)
        };

        let ref_lists_info_flags = ash::vk::native::StdVideoEncodeH265ReferenceListsInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeH265ReferenceListsInfoFlags::new_bitfield_1(
                0, 0, 0,
            ),
        };

        let ref_lists_info = ash::vk::native::StdVideoEncodeH265ReferenceListsInfo {
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
            list_entry_l0: [0; 15],
            list_entry_l1: [0; 15],
        };

        let picture_info = ash::vk::native::StdVideoEncodeH265PictureInfo {
            flags: picture_info_flags,
            pps_pic_parameter_set_id: 0,
            pps_seq_parameter_set_id: 0,
            sps_video_parameter_set_id: 0,
            PicOrderCntVal: pic_order_cnt,
            TemporalId: 0,
            reserved1: [0; 7],
            pRefLists: if !is_idr && !self.l0_references.is_empty() {
                &ref_lists_info
            } else {
                std::ptr::null()
            },
            pShortTermRefPicSet: if is_idr {
                &empty_rps
            } else if !self.l0_references.is_empty() {
                &frame_rps
            } else {
                &empty_rps
            },
            pLongTermRefPics: std::ptr::null(),
            pic_type: picture_type,
            short_term_ref_pic_set_idx: 0,
        };

        // Create slice NAL unit entry.
        let constant_qp = match self.config.rate_control_mode {
            crate::encoder::RateControlMode::Cqp | crate::encoder::RateControlMode::Disabled => {
                self.config.quality_level as i32
            }
            _ => 0,
        };
        let nalu_slice_entries = [vk::VideoEncodeH265NaluSliceSegmentInfoKHR::default()
            .constant_qp(constant_qp)
            .std_slice_segment_header(&slice_header)];

        // Create H.265 picture info.
        let mut h265_picture_info = vk::VideoEncodeH265PictureInfoKHR::default()
            .nalu_slice_segment_entries(&nalu_slice_entries)
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

        // Set up setup picture resource (reconstructed picture)
        let setup_picture_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: self.aligned_width,
                height: self.aligned_height,
            })
            .base_array_layer(0)
            .image_view_binding(self.dpb_image_views[self.current_dpb_slot as usize]);

        // Create reference info for setup slot.
        let std_reference_info_flags = ash::vk::native::StdVideoEncodeH265ReferenceInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeH265ReferenceInfoFlags::new_bitfield_1(
                0, 0, 0,
            ),
        };

        let std_reference_info = ash::vk::native::StdVideoEncodeH265ReferenceInfo {
            flags: std_reference_info_flags,
            PicOrderCntVal: pic_order_cnt,
            TemporalId: 0,
            pic_type: picture_type,
        };

        let mut h265_setup_dpb_slot_info =
            vk::VideoEncodeH265DpbSlotInfoKHR::default().std_reference_info(&std_reference_info);

        let mut setup_slot_info = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(self.current_dpb_slot as i32)
            .picture_resource(&setup_picture_resource);
        setup_slot_info.p_next =
            (&mut h265_setup_dpb_slot_info as *mut vk::VideoEncodeH265DpbSlotInfoKHR).cast();

        // Setup slot for begin - always use -1 to indicate it's not yet active.
        // (it will be written to during encoding)
        let mut h265_begin_dpb_slot_info =
            vk::VideoEncodeH265DpbSlotInfoKHR::default().std_reference_info(&std_reference_info);

        let mut setup_slot_for_begin = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(-1) // Always -1 for setup slot
            .picture_resource(&setup_picture_resource);
        setup_slot_for_begin.p_next =
            (&mut h265_begin_dpb_slot_info as *mut vk::VideoEncodeH265DpbSlotInfoKHR).cast();

        // Set up reference slots.

        // Storage vectors to keep data alive while pointers are used.
        // We need capacity sufficient for all potential references (L0 + L1).
        let mut ref_resources = Vec::with_capacity(16);
        let mut std_ref_infos = Vec::with_capacity(16);
        let mut h265_slot_infos = Vec::with_capacity(16);

        // Final slot lists
        let mut reference_slots = Vec::with_capacity(16);
        let mut reference_slots_for_begin = Vec::with_capacity(17); // +1 for setup slot
        reference_slots_for_begin.push(setup_slot_for_begin);

        let has_l0_ref = !is_idr && !self.l0_references.is_empty();
        let has_l1_ref = is_b_frame && self.has_backward_reference;

        // Phase 1: Populate data storage (resources and std infos)
        if has_l0_ref {
            for ref_info in &self.l0_references {
                let ref_resource = vk::VideoPictureResourceInfoKHR::default()
                    .coded_offset(vk::Offset2D { x: 0, y: 0 })
                    .coded_extent(vk::Extent2D {
                        width: self.aligned_width,
                        height: self.aligned_height,
                    })
                    .base_array_layer(0)
                    .image_view_binding(self.dpb_image_views[ref_info.dpb_slot as usize]);

                ref_resources.push(ref_resource);

                let mut std_info = std_reference_info;
                std_info.PicOrderCntVal = ref_info.poc;
                std_info.pic_type =
                    ash::vk::native::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_P;
                std_ref_infos.push(std_info);
            }
        }

        if has_l1_ref {
            let ref_resource = vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D { x: 0, y: 0 })
                .coded_extent(vk::Extent2D {
                    width: self.aligned_width,
                    height: self.aligned_height,
                })
                .base_array_layer(0)
                .image_view_binding(
                    self.dpb_image_views[self.backward_reference_dpb_slot as usize],
                );

            ref_resources.push(ref_resource);

            let mut std_info = std_reference_info;
            std_info.PicOrderCntVal = self.backward_reference_poc;
            std_info.pic_type =
                ash::vk::native::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_P;
            std_ref_infos.push(std_info);
        }

        // Phase 2: Populate chain structs (referencing std_ref_infos)
        for std_info in &std_ref_infos {
            let h265_ref_slot_info =
                vk::VideoEncodeH265DpbSlotInfoKHR::default().std_reference_info(std_info);
            h265_slot_infos.push(h265_ref_slot_info);
        }

        // Phase 3: Populate reference slots (referencing ref_resources and h265_slot_infos)
        // We know the mapping corresponds 1:1 by index because we pushed in the same order.
        // Re-construct the list of DPB slots to assign correct slot_index.

        let mut stored_indices_count = 0;
        if has_l0_ref {
            for ref_info in &self.l0_references {
                let mut ref_slot = vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(ref_info.dpb_slot as i32)
                    .picture_resource(&ref_resources[stored_indices_count]);

                ref_slot.p_next = (&mut h265_slot_infos[stored_indices_count]
                    as *mut vk::VideoEncodeH265DpbSlotInfoKHR)
                    .cast();

                reference_slots.push(ref_slot);
                reference_slots_for_begin.push(ref_slot);

                stored_indices_count += 1;
            }
        }

        if has_l1_ref {
            let mut ref_slot = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(self.backward_reference_dpb_slot as i32)
                .picture_resource(&ref_resources[stored_indices_count]);

            ref_slot.p_next = (&mut h265_slot_infos[stored_indices_count]
                as *mut vk::VideoEncodeH265DpbSlotInfoKHR)
                .cast();

            reference_slots.push(ref_slot);
            reference_slots_for_begin.push(ref_slot);

            // stored_indices_count += 1; // Not needed anymore
        }

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
                26,
            ),
            crate::encoder::RateControlMode::Vbr => (
                vk::VideoEncodeRateControlModeFlagsKHR::VBR,
                self.config.target_bitrate,
                self.config.max_bitrate,
                26,
            ),
        };

        let min_qp_val = if rc_mode == vk::VideoEncodeRateControlModeFlagsKHR::DISABLED {
            qp
        } else {
            26
        };
        let max_qp_val = if rc_mode == vk::VideoEncodeRateControlModeFlagsKHR::DISABLED {
            qp
        } else {
            51
        };

        let min_qp = vk::VideoEncodeH265QpKHR {
            qp_i: min_qp_val,
            qp_p: min_qp_val,
            qp_b: min_qp_val,
        };

        let max_qp = vk::VideoEncodeH265QpKHR {
            qp_i: max_qp_val,
            qp_p: max_qp_val,
            qp_b: max_qp_val,
        };

        let mut h265_rc_layer_info = vk::VideoEncodeH265RateControlLayerInfoKHR::default()
            .min_qp(min_qp)
            .max_qp(max_qp);

        let mut rc_layer_info = vk::VideoEncodeRateControlLayerInfoKHR::default()
            .average_bitrate(average_bitrate as u64)
            .max_bitrate(max_bitrate as u64)
            .frame_rate_numerator(self.config.frame_rate_numerator)
            .frame_rate_denominator(self.config.frame_rate_denominator);
        rc_layer_info.p_next =
            (&mut h265_rc_layer_info as *mut vk::VideoEncodeH265RateControlLayerInfoKHR).cast();

        let rc_layers = [rc_layer_info];

        let mut h265_rc_info = vk::VideoEncodeH265RateControlInfoKHR::default()
            .gop_frame_count(self.config.gop_size)
            .idr_period(self.config.gop_size)
            .consecutive_b_frame_count(self.config.b_frame_count);

        let mut rc_info = vk::VideoEncodeRateControlInfoKHR::default().rate_control_mode(rc_mode);

        if rc_mode != vk::VideoEncodeRateControlModeFlagsKHR::DISABLED {
            rc_info = rc_info
                .layers(&rc_layers)
                .virtual_buffer_size_in_ms(self.config.virtual_buffer_size_ms)
                .initial_virtual_buffer_size_in_ms(self.config.initial_virtual_buffer_size_ms);
            rc_info.p_next =
                (&mut h265_rc_info as *mut vk::VideoEncodeH265RateControlInfoKHR).cast();
        }

        // Begin video coding.
        let is_first_frame = self.encode_frame_num == 0;

        let begin_coding_info = if is_first_frame {
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
                &begin_coding_info,
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

        // Encode command.
        let mut encode_info = vk::VideoEncodeInfoKHR::default()
            .flags(vk::VideoEncodeFlagsKHR::empty())
            .src_picture_resource(src_picture_resource)
            .setup_reference_slot(&setup_slot_info)
            .reference_slots(&reference_slots)
            .dst_buffer(self.slots[self.current_slot].bitstream_buffer)
            .dst_buffer_offset(0)
            .dst_buffer_range(MIN_BITSTREAM_BUFFER_SIZE as u64);
        encode_info.p_next =
            (&mut h265_picture_info as *mut vk::VideoEncodeH265PictureInfoKHR).cast();

        unsafe {
            self.context.device().cmd_begin_query(
                self.slots[self.current_slot].encode_command_buffer,
                self.slots[self.current_slot].query_pool,
                0,
                vk::QueryControlFlags::empty(),
            );

            (self.video_encode_fn.fp().cmd_encode_video_khr)(
                self.slots[self.current_slot].encode_command_buffer,
                &encode_info,
            );

            self.context.device().cmd_end_query(
                self.slots[self.current_slot].encode_command_buffer,
                self.slots[self.current_slot].query_pool,
                0,
            );
        }

        // Add DPB synchronization barrier after encoding.
        unsafe {
            record_post_encode_dpb_barrier(
                self.context.device(),
                self.slots[self.current_slot].encode_command_buffer,
                &self.dpb_images,
                self.use_layered_dpb,
                self.current_dpb_slot,
            );
        }

        // End video coding.
        let end_coding_info = vk::VideoEndCodingInfoKHR::default();
        unsafe {
            (self.video_queue_fn.fp().cmd_end_video_coding_khr)(
                self.slots[self.current_slot].encode_command_buffer,
                &end_coding_info,
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

        debug!(
            "Submitting frame {} to GPU: idr={}, num_refs={}, cur_slot={}",
            self.encode_frame_num,
            is_idr,
            self.l0_references.len(),
            self.current_dpb_slot
        );

        let gpu_start = std::time::Instant::now();

        unsafe {
            submit_encode_only(
                self.context.device(),
                self.slots[self.current_slot].encode_command_buffer,
                self.slots[self.current_slot].encode_fence,
                encode_queue,
                None,
            )?;
        }

        debug!("Submitted encode (no wait): {:?}", gpu_start.elapsed());

        // Mark DPB slot as active.
        self.dpb_slot_active[self.current_dpb_slot as usize] = true;

        // Mark the slot as in flight; the bitstream is drained at the start
        // of the next encode() call that targets this slot.
        self.slots[self.current_slot].in_flight = true;

        Ok(())
    }
}
