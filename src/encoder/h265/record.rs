//! H.265 per-frame encode recording: builds the StdVideo* graph for one frame
//! and submits it.

use super::H265;

use crate::encoder::codec::{EncodeCommands, EncoderCommon, FramePlan, RateControlPlan};
use crate::encoder::pipeline::EncodeFuture;
use crate::encoder::resources::{prepare_encode_command_buffer, record_dpb_barriers};
use crate::error::Result;
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
        let pic_order_cnt = plan.pic_order_cnt();

        let slot = common.pipeline.current();
        let command_buffer = slot.encode_command_buffer;
        let query_pool = slot.query_pool;
        let bitstream_buffer = slot.bitstream_buffer;
        let bitstream_buffer_size = slot.bitstream_buffer_size;
        let input_image_view = common.source_view();

        unsafe {
            prepare_encode_command_buffer(common.device(), command_buffer, query_pool)?;
        }
        // Only the references this frame predicts from go to the encode. The
        // rest of the window stays in its DPB slots, untouched.
        let used_references = self.used_references();
        let ref_dpb_slots: Vec<u8> = used_references.iter().map(|r| r.dpb_slot).collect();
        unsafe {
            record_dpb_barriers(
                common.context.sync2(),
                command_buffer,
                &common.dpb_images,
                common.use_layered_dpb,
                common.current_dpb_slot,
                &ref_dpb_slots,
                common.dpb_slot_active[common.current_dpb_slot as usize],
            );
            common.record_source_acquire(command_buffer);
        }

        let slice_type = if is_idr {
            ash::vk::native::StdVideoH265SliceType_STD_VIDEO_H265_SLICE_TYPE_I
        } else {
            ash::vk::native::StdVideoH265SliceType_STD_VIDEO_H265_SLICE_TYPE_P
        };
        let picture_type = if is_idr {
            ash::vk::native::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_IDR
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
        let mut num_negative_pics: u8 = 0;
        let mut used_by_curr_pic_s0_flag: u16 = 0;

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
                // Every held reference stays in the set, so the decoder keeps
                // it too, but only the newest are used by this picture.
                if i < used_references.len() {
                    used_by_curr_pic_s0_flag |= 1 << num_negative_pics;
                }
                num_negative_pics += 1;
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
            used_by_curr_pic_s1_flag: 0,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
            num_negative_pics,
            num_positive_pics: 0,
            delta_poc_s0_minus1,
            delta_poc_s1_minus1: [0; 16],
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
        let ref_list1: [u8; 15] = [NO_REFERENCE_PICTURE; 15];

        let num_ref_l0 = if !is_idr && !self.l0_references.is_empty() {
            let count = used_references.len();
            for (i, ref_info) in used_references.iter().enumerate() {
                if i < 15 {
                    ref_list0[i] = ref_info.dpb_slot;
                }
            }
            count.min(15)
        } else {
            0
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
            num_ref_idx_l1_active_minus1: 0,
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
        let constant_qp = if rc.is_disabled() { rc.qp as i32 } else { 0 };
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
            .coded_extent(common.source_coded_extent())
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

        if has_l0_ref {
            for ref_info in used_references {
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

        for std_info in &std_ref_infos {
            h265_slot_infos
                .push(vk::VideoEncodeH265DpbSlotInfoKHR::default().std_reference_info(std_info));
        }

        let mut stored_indices_count = 0;
        let mut slot_infos_iter = h265_slot_infos.iter_mut();
        if has_l0_ref {
            for ref_info in used_references {
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

        // Rate control.
        let qp_min = if rc.is_disabled() { rc.qp as i32 } else { 26 };
        let qp_max = if rc.is_disabled() { rc.qp as i32 } else { 51 };
        let min_qp = vk::VideoEncodeH265QpKHR {
            qp_i: qp_min,
            qp_p: qp_min,
            qp_b: qp_min,
        };
        let max_qp = vk::VideoEncodeH265QpKHR {
            qp_i: qp_max,
            qp_p: qp_max,
            qp_b: qp_max,
        };
        let codec_layer = vk::VideoEncodeH265RateControlLayerInfoKHR::default()
            .min_qp(min_qp)
            .max_qp(max_qp);
        let codec_rate_control = vk::VideoEncodeH265RateControlInfoKHR::default()
            .gop_frame_count(common.config.gop_size)
            .idr_period(common.config.gop_size);

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

        let first_frame = plan.is_first_frame();
        common.record_encode(EncodeCommands {
            begin_slots: &reference_slots_for_begin,
            encode_info: &encode_info,
            rate_control: &rc,
            codec_rate_control,
            codec_layer,
            first_frame,
            reset: first_frame,
            post_encode_barrier: true,
        })
    }
}
