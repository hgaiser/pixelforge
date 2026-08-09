//! H.265 per-frame encode recording: builds the StdVideo* graph for one frame
//! and submits it.

use super::H265;

use crate::encoder::codec::{EncoderCommon, FramePlan, RateControlPlan};
use crate::encoder::pipeline::EncodeFuture;
use crate::encoder::resources::{
    end_timestamp, prepare_encode_command_buffer, record_dpb_barriers,
    record_post_encode_dpb_barrier, reset_start_timestamp,
};
use crate::error::{PixelForgeError, Result};
use ash::vk;
use ash::vk::TaggedStructure;
use tracing::debug;

impl H265 {
    pub(super) fn record(
        &mut self,
        common: &mut EncoderCommon,
        plan: &FramePlan,
    ) -> Result<EncodeFuture> {
        let is_idr = plan.is_idr();
        let is_reference = plan.is_reference();
        let is_b_frame = plan.is_b_frame();
        let pic_order_cnt = plan.pic_order_cnt();

        let slot = common.pipeline.current();
        let command_buffer = slot.encode_command_buffer;
        let query_pool = slot.query_pool;
        let timestamp_query_pool = slot.timestamp_query_pool;
        let bitstream_buffer = slot.bitstream_buffer;
        let bitstream_buffer_size = slot.bitstream_buffer_size;
        let input_image_view = slot.input_image_view;

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

        // Short-term reference picture set: negative (past) refs for P, plus a
        // positive (future) ref for B.
        let mut delta_poc_s0_minus1 = [0u16; 16];
        let mut delta_poc_s1_minus1 = [0u16; 16];
        let mut num_negative_pics: u8 = 0;
        let mut num_positive_pics: u8 = 0;
        let mut used_by_curr_pic_s0_flag: u16 = 0;
        let mut used_by_curr_pic_s1_flag: u16 = 0;

        if !is_idr && !self.l0_references.is_empty() {
            // max_poc = 2^(log2_max_pic_order_cnt_lsb_minus4 + 4) * 2 = 512.
            let max_poc = 1i32 << 9;
            let mut prev_delta_poc = 0;
            for (i, ref_info) in self.l0_references.iter().enumerate() {
                if i >= 15 {
                    break;
                }
                let mut delta_poc = ref_info.poc - pic_order_cnt;
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
            if is_b_frame && self.has_backward_reference {
                let delta_poc_l1 = self.backward_reference_poc - pic_order_cnt;
                delta_poc_s1_minus1[0] = (delta_poc_l1 - 1).max(0) as u16;
                num_positive_pics = 1;
                used_by_curr_pic_s1_flag = 1;
            }
        }

        let frame_rps = ash::vk::native::StdVideoH265ShortTermRefPicSet {
            flags: ash::vk::native::StdVideoH265ShortTermRefPicSetFlags {
                _bitfield_align_1: [],
                _bitfield_1: ash::vk::native::StdVideoH265ShortTermRefPicSetFlags::new_bitfield_1(
                    0, 0,
                ),
                __bindgen_padding_0: [0; 3],
            },
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

        let empty_rps = ash::vk::native::StdVideoH265ShortTermRefPicSet {
            flags: ash::vk::native::StdVideoH265ShortTermRefPicSetFlags {
                _bitfield_align_1: [],
                _bitfield_1: ash::vk::native::StdVideoH265ShortTermRefPicSetFlags::new_bitfield_1(
                    0, 0,
                ),
                __bindgen_padding_0: [0; 3],
            },
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

        const NO_REFERENCE_PICTURE: u8 = 0xFF;
        let mut ref_list0: [u8; 15] = [NO_REFERENCE_PICTURE; 15];
        let mut ref_list1: [u8; 15] = [NO_REFERENCE_PICTURE; 15];

        let (num_ref_l0, num_ref_l1) = if is_b_frame && self.has_backward_reference {
            if let Some(first_ref) = self.l0_references.first() {
                ref_list0[0] = first_ref.dpb_slot;
                ref_list1[0] = self.backward_reference_dpb_slot;
                (1, 1)
            } else {
                (0, 0)
            }
        } else if !is_idr && !self.l0_references.is_empty() {
            let count = self.l0_references.len();
            for (i, ref_info) in self.l0_references.iter().enumerate() {
                if i < 15 {
                    ref_list0[i] = ref_info.dpb_slot;
                }
            }
            (count.min(15), 0)
        } else {
            (0, 0)
        };

        let ref_lists_info = ash::vk::native::StdVideoEncodeH265ReferenceListsInfo {
            flags: ash::vk::native::StdVideoEncodeH265ReferenceListsInfoFlags {
                _bitfield_align_1: [],
                _bitfield_1:
                    ash::vk::native::StdVideoEncodeH265ReferenceListsInfoFlags::new_bitfield_1(
                        0, 0, 0,
                    ),
            },
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

        let rc = RateControlPlan::new(&common.config, 26);
        let constant_qp = if rc.is_disabled() {
            common.config.quality_level as i32
        } else {
            0
        };
        let nalu_slice_entries = [vk::VideoEncodeH265NaluSliceSegmentInfoKHR::default()
            .constant_qp(constant_qp)
            .std_slice_segment_header(&slice_header)];

        let mut h265_picture_info = vk::VideoEncodeH265PictureInfoKHR::default()
            .nalu_slice_segment_entries(&nalu_slice_entries)
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
        let setup_slot_info = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(common.current_dpb_slot as i32)
            .picture_resource(&setup_picture_resource)
            .push(&mut h265_setup_dpb_slot_info);

        let mut h265_begin_dpb_slot_info =
            vk::VideoEncodeH265DpbSlotInfoKHR::default().std_reference_info(&std_reference_info);
        let setup_slot_for_begin = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(-1)
            .picture_resource(&setup_picture_resource)
            .push(&mut h265_begin_dpb_slot_info);

        // Storage to keep pointers alive across the encode call.
        let mut ref_resources = Vec::with_capacity(16);
        let mut std_ref_infos = Vec::with_capacity(16);
        let mut h265_slot_infos = Vec::with_capacity(16);
        let mut reference_slots = Vec::with_capacity(16);
        let mut reference_slots_for_begin = Vec::with_capacity(17);
        reference_slots_for_begin.push(setup_slot_for_begin);

        let has_l0_ref = !is_idr && !self.l0_references.is_empty();
        let has_l1_ref = is_b_frame && self.has_backward_reference;

        if has_l0_ref {
            for ref_info in &self.l0_references {
                ref_resources.push(
                    vk::VideoPictureResourceInfoKHR::default()
                        .coded_offset(vk::Offset2D { x: 0, y: 0 })
                        .coded_extent(coded_extent)
                        .base_array_layer(0)
                        .image_view_binding(common.dpb_image_views[ref_info.dpb_slot as usize]),
                );
                let mut std_info = std_reference_info;
                std_info.PicOrderCntVal = ref_info.poc;
                std_info.pic_type =
                    ash::vk::native::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_P;
                std_ref_infos.push(std_info);
            }
        }
        if has_l1_ref {
            ref_resources.push(
                vk::VideoPictureResourceInfoKHR::default()
                    .coded_offset(vk::Offset2D { x: 0, y: 0 })
                    .coded_extent(coded_extent)
                    .base_array_layer(0)
                    .image_view_binding(
                        common.dpb_image_views[self.backward_reference_dpb_slot as usize],
                    ),
            );
            let mut std_info = std_reference_info;
            std_info.PicOrderCntVal = self.backward_reference_poc;
            std_info.pic_type =
                ash::vk::native::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_P;
            std_ref_infos.push(std_info);
        }

        for std_info in &std_ref_infos {
            h265_slot_infos
                .push(vk::VideoEncodeH265DpbSlotInfoKHR::default().std_reference_info(std_info));
        }

        let mut stored_indices_count = 0;
        let mut slot_infos_iter = h265_slot_infos.iter_mut();
        if has_l0_ref {
            for ref_info in &self.l0_references {
                if let Some(h265_slot_info) = slot_infos_iter.next() {
                    let ref_slot = vk::VideoReferenceSlotInfoKHR::default()
                        .slot_index(ref_info.dpb_slot as i32)
                        .picture_resource(&ref_resources[stored_indices_count])
                        .push(h265_slot_info);
                    reference_slots.push(ref_slot);
                    reference_slots_for_begin.push(ref_slot);
                    stored_indices_count += 1;
                }
            }
        }
        if has_l1_ref && let Some(h265_slot_info) = slot_infos_iter.next() {
            let ref_slot = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(self.backward_reference_dpb_slot as i32)
                .picture_resource(&ref_resources[stored_indices_count])
                .push(h265_slot_info);
            reference_slots.push(ref_slot);
            reference_slots_for_begin.push(ref_slot);
        }

        // Rate control.
        let qp_min = if rc.is_disabled() { rc.qp as i32 } else { 26 };
        let qp_max = if rc.is_disabled() { rc.qp as i32 } else { 51 };
        let qp_max_i = if rc.is_disabled() {
            rc.qp as i32
        } else {
            common
                .config
                .max_qp_i
                .map_or(qp_max, |qp| (qp as i32).max(qp_min))
        };
        let min_qp = vk::VideoEncodeH265QpKHR {
            qp_i: qp_min,
            qp_p: qp_min,
            qp_b: qp_min,
        };
        let max_qp = vk::VideoEncodeH265QpKHR {
            qp_i: qp_max_i,
            qp_p: qp_max,
            qp_b: qp_max,
        };
        let mut h265_rc_layer_info = vk::VideoEncodeH265RateControlLayerInfoKHR::default()
            .min_qp(min_qp)
            .max_qp(max_qp);
        let rc_layer_info = vk::VideoEncodeRateControlLayerInfoKHR::default()
            .average_bitrate(rc.average_bitrate as u64)
            .max_bitrate(rc.max_bitrate as u64)
            .frame_rate_numerator(common.config.frame_rate_numerator)
            .frame_rate_denominator(common.config.frame_rate_denominator)
            .push(&mut h265_rc_layer_info);
        let rc_layers = [rc_layer_info];

        let mut h265_rc_info = vk::VideoEncodeH265RateControlInfoKHR::default()
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

        let is_first_frame = plan.is_first_frame();
        let begin_coding_info = if is_first_frame {
            vk::VideoBeginCodingInfoKHR::default()
                .video_session(common.session)
                .video_session_parameters(common.session_params)
                .reference_slots(&reference_slots_for_begin)
                .push(&mut h265_rc_info)
        } else {
            vk::VideoBeginCodingInfoKHR::default()
                .video_session(common.session)
                .video_session_parameters(common.session_params)
                .reference_slots(&reference_slots_for_begin)
                .push(&mut h265_rc_info)
                .push(&mut rc_info)
        };

        unsafe {
            (common.video_queue_fn.fp().cmd_begin_video_coding_khr)(
                command_buffer,
                &begin_coding_info,
            );
        }

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
                .push(&mut h265_rc_info)
                .push(&mut quality_level_info);
            unsafe {
                (common.video_queue_fn.fp().cmd_control_video_coding_khr)(
                    command_buffer,
                    &control_info,
                );
            }
        }

        let encode_info = vk::VideoEncodeInfoKHR::default()
            .flags(vk::VideoEncodeFlagsKHR::empty())
            .src_picture_resource(src_picture_resource)
            .setup_reference_slot(&setup_slot_info)
            .reference_slots(&reference_slots)
            .dst_buffer(bitstream_buffer)
            .dst_buffer_offset(0)
            .dst_buffer_range(bitstream_buffer_size as u64)
            .push(&mut h265_picture_info);

        debug!(
            "h265 submit frame {}: idr={}, num_refs={}, cur_slot={}",
            plan.encode_index,
            is_idr,
            self.l0_references.len(),
            common.current_dpb_slot
        );

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

            record_post_encode_dpb_barrier(
                device,
                command_buffer,
                &common.dpb_images,
                common.use_layered_dpb,
                common.current_dpb_slot,
            );

            let end_coding_info = vk::VideoEndCodingInfoKHR::default();
            (common.video_queue_fn.fp().cmd_end_video_coding_khr)(command_buffer, &end_coding_info);

            device
                .end_command_buffer(command_buffer)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
        }

        common.submit_frame()
    }
}
