//! H.264 per-frame encode recording: builds the StdVideo* graph for one frame
//! and submits it. This is the part that genuinely differs from the other codecs.

use super::lists::{b_frame_lists, p_frame_list_modifications};
use super::{H264, ReferenceInfo};

use crate::encoder::codec::{EncodeCommands, EncoderCommon, FramePlan, RateControlPlan};
use crate::encoder::resources::{prepare_encode_command_buffer, record_dpb_barriers};
use crate::error::{PixelForgeError, Result};
use ash::vk;
use ash::vk::TaggedStructure;
use tracing::debug;

/// What a constant QP is raised by for a B-frame that is not a reference.
const B_FRAME_QP_OFFSET: u32 = 2;

impl H264 {
    pub(super) fn record(&mut self, common: &mut EncoderCommon, plan: &FramePlan) -> Result<()> {
        let is_idr = plan.is_idr();
        let is_reference = plan.is_reference();
        let is_b = plan.is_b();
        let pic_order_cnt = plan.pic_order_cnt();
        let frame_num = self.frame_num_syntax;

        let slot = common.pipeline.current();
        let command_buffer = slot.encode_command_buffer;
        let query_pool = slot.query_pool;
        let bitstream_buffer = slot.bitstream_buffer;
        let bitstream_buffer_size = slot.bitstream_buffer_size;
        let input_image_view = common.source_view();

        debug!(
            "h264 record: frame_num={}, poc={}, is_idr={}, refs_len={}, cur_slot={}",
            frame_num,
            pic_order_cnt,
            is_idr,
            self.l0_references.len(),
            common.current_dpb_slot
        );

        let rc = RateControlPlan::for_frame(&common.config, 26, plan, B_FRAME_QP_OFFSET, 51);

        // Reference frame invalidation: emit MMCO "unmark short-term" operations
        // for references dropped since the last frame, so the decoder evicts the
        // same pictures we did and its default reference list re-anchors to the
        // surviving reference. Only valid on a non-IDR reference picture (an IDR
        // resets the DPB on its own). Kept alive for the whole `record` call so
        // the pointer handed to the encoder stays valid.
        let mut ref_pic_marking_ops: Vec<ash::vk::native::StdVideoEncodeH264RefPicMarkingEntry> =
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
        // Only the references this frame predicts from go to the encode. The
        // rest of the window stays in its DPB slots, untouched.
        let (list0, list1) = self.reference_lists(plan)?;
        let used_references: Vec<ReferenceInfo> = list0.iter().chain(&list1).copied().collect();
        let used_references = used_references.as_slice();
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
            ash::vk::native::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_I
        } else if is_b {
            ash::vk::native::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_B
        } else {
            ash::vk::native::StdVideoH264SliceType_STD_VIDEO_H264_SLICE_TYPE_P
        };

        let picture_type = if is_idr {
            ash::vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR
        } else if is_b {
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
                is_b as u32,      // direct_spatial_mv_pred_flag
                use_ref_override, // num_ref_idx_active_override_flag
                0,                // reserved
            ),
        };

        let slice_qp_delta = if rc.is_disabled() {
            (rc.qp as i32 - 26) as i8
        } else {
            0
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
        for (entry, reference) in ref_list0.iter_mut().zip(&list0) {
            *entry = reference.dpb_slot;
        }
        for (entry, reference) in ref_list1.iter_mut().zip(&list1) {
            *entry = reference.dpb_slot;
        }

        // A P-frame that makes B-frame references obsolete still has them at
        // the start of the list a decoder builds for it, because the marking
        // only takes effect after the frame. So it names its references.
        let mut list0_modifications: Vec<ash::vk::native::StdVideoEncodeH264RefListModEntry> =
            if !is_idr && !is_b && common.config.b_frames > 0 && use_adaptive_marking {
                let frame_nums: Vec<u32> = list0.iter().map(|r| r.frame_num).collect();
                p_frame_list_modifications(frame_num, &frame_nums, 256)
                    .into_iter()
                    .map(|op| ash::vk::native::StdVideoEncodeH264RefListModEntry {
                        modification_of_pic_nums_idc: if op.add {
                            ash::vk::native::StdVideoH264ModificationOfPicNumsIdc_STD_VIDEO_H264_MODIFICATION_OF_PIC_NUMS_IDC_SHORT_TERM_ADD
                        } else {
                            ash::vk::native::StdVideoH264ModificationOfPicNumsIdc_STD_VIDEO_H264_MODIFICATION_OF_PIC_NUMS_IDC_SHORT_TERM_SUBTRACT
                        },
                        abs_diff_pic_num_minus1: op.abs_diff_pic_num_minus1,
                        long_term_pic_num: 0,
                    })
                    .collect()
            } else {
                Vec::new()
            };

        // The NVIDIA driver (615.78.08) writes the operations it is given
        // and nothing more, so the operation that ends the list is one of
        // them.
        if !list0_modifications.is_empty() {
            list0_modifications.push(ash::vk::native::StdVideoEncodeH264RefListModEntry {
                modification_of_pic_nums_idc:
                    ash::vk::native::StdVideoH264ModificationOfPicNumsIdc_STD_VIDEO_H264_MODIFICATION_OF_PIC_NUMS_IDC_END,
                abs_diff_pic_num_minus1: 0,
                long_term_pic_num: 0,
            });
        }
        // The same driver ends the marking operations by itself only in a
        // P-slice with no list operations. In a B-slice, or with list
        // operations, the operation that ends them has to be given.
        if use_adaptive_marking && (is_b || !list0_modifications.is_empty()) {
            ref_pic_marking_ops.push(ash::vk::native::StdVideoEncodeH264RefPicMarkingEntry {
                memory_management_control_operation:
                    ash::vk::native::StdVideoH264MemMgmtControlOp_STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_END,
                difference_of_pic_nums_minus1: 0,
                long_term_pic_num: 0,
                long_term_frame_idx: 0,
                max_long_term_frame_idx_plus1: 0,
            });
        }

        let ref_lists_info_flags = ash::vk::native::StdVideoEncodeH264ReferenceListsInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeH264ReferenceListsInfoFlags::new_bitfield_1(
                !list0_modifications.is_empty() as u32, // ref_pic_list_modification_flag_l0
                0,                                      // ref_pic_list_modification_flag_l1
                0,                                      // reserved
            ),
        };

        let ref_lists_info = ash::vk::native::StdVideoEncodeH264ReferenceListsInfo {
            flags: ref_lists_info_flags,
            num_ref_idx_l0_active_minus1: list0.len().saturating_sub(1) as u8,
            num_ref_idx_l1_active_minus1: list1.len().saturating_sub(1) as u8,
            RefPicList0: ref_list0,
            RefPicList1: ref_list1,
            refList0ModOpCount: list0_modifications.len() as u8,
            refList1ModOpCount: 0,
            refPicMarkingOpCount: ref_pic_marking_ops.len() as u8,
            reserved1: [0; 7],
            pRefList0ModOperations: if list0_modifications.is_empty() {
                std::ptr::null()
            } else {
                list0_modifications.as_ptr()
            },
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
            pRefLists: if !is_idr && (!used_references.is_empty() || use_adaptive_marking) {
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
            .coded_extent(common.source_coded_extent())
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
        let mut l0_resources = Vec::with_capacity(used_references.len());
        let mut l0_std_infos = Vec::with_capacity(used_references.len());

        for ref_info in used_references {
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
            let ref_info = &used_references[i];
            l0_slots.push(
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(ref_info.dpb_slot as i32)
                    .picture_resource(resource)
                    .push(dpb_info),
            );
        }

        // Reference info for the setup slot (this frame).
        let std_reference_info = ash::vk::native::StdVideoEncodeH264ReferenceInfo {
            flags: std_reference_info_flags,
            primary_pic_type: picture_type,
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
        if !is_idr {
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
        if !is_idr {
            reference_slots_for_begin.extend_from_slice(&l0_slots);
        }

        // Rate control.
        let qp_bounds = if rc.is_disabled() { rc.qp as i32 } else { 18 };
        let qp_bounds_max = if rc.is_disabled() { rc.qp as i32 } else { 42 };
        let min_qp = vk::VideoEncodeH264QpKHR {
            qp_i: qp_bounds,
            qp_p: qp_bounds,
            qp_b: qp_bounds,
        };
        let max_qp = vk::VideoEncodeH264QpKHR {
            qp_i: qp_bounds_max,
            qp_p: qp_bounds_max,
            qp_b: qp_bounds_max,
        };
        let codec_layer = vk::VideoEncodeH264RateControlLayerInfoKHR::default()
            .use_min_qp(true)
            .min_qp(min_qp)
            .use_max_qp(true)
            .max_qp(max_qp);
        let codec_rate_control = self.rate_control_info(common);

        let first_frame = plan.is_first_frame();
        common.record_encode(EncodeCommands {
            begin_slots: &reference_slots_for_begin,
            encode_info: &encode_info,
            rate_control: &rc,
            codec_rate_control,
            codec_layer,
            first_frame,
            reset: first_frame,
            post_encode_barrier: false,
        })?;
        // Clear the unmark queue only after the encode is committed to the GPU.
        // If any fallible step above had failed, the MMCO ops would still live
        // in `pending_unmark_frame_nums` so a retry re-emits them — otherwise the
        // encoder would silently desync from the decoder's reference state.
        if use_adaptive_marking {
            self.pending_unmark_frame_nums.clear();
        }
        Ok(())
    }

    /// The references this frame predicts from, as `RefPicList0` and
    /// `RefPicList1`.
    ///
    /// A P-frame uses the newest references, in decode order, which is the
    /// order a decoder puts them in. A B-frame uses the nearest ones shown
    /// before it and the nearest one shown after it.
    fn reference_lists(
        &self,
        plan: &FramePlan,
    ) -> Result<(Vec<ReferenceInfo>, Vec<ReferenceInfo>)> {
        if plan.is_idr() {
            return Ok((Vec::new(), Vec::new()));
        }
        if !plan.is_b() {
            return Ok((self.used_references().to_vec(), Vec::new()));
        }
        let current = plan.pic_order_cnt();
        let pocs: Vec<i32> = self.l0_references.iter().map(|r| r.poc).collect();
        let before = pocs.iter().filter(|&&poc| poc < current).count();
        let after = pocs.len() - before;
        if before == 0 || after == 0 {
            return Err(PixelForgeError::InvalidInput(format!(
                "H.264 B-frame with POC {current} has {before} references before it and \
                 {after} after it"
            )));
        }
        let (l0, l1) = b_frame_lists(&pocs, current);
        let take = |list: Vec<usize>, count: usize| -> Vec<ReferenceInfo> {
            list.into_iter()
                .take(count)
                .map(|i| self.l0_references[i])
                .collect()
        };
        // Only references on the frame's own side of each list, so that no
        // picture is in both and the count stays within the session's.
        let l0_count = before.min(self.b_l0_reference_count as usize);
        Ok((take(l0, l0_count), take(l1, 1)))
    }

    /// The GOP the rate control is told to expect.
    fn rate_control_info(
        &self,
        common: &EncoderCommon,
    ) -> vk::VideoEncodeH264RateControlInfoKHR<'static> {
        let config = &common.config;
        let info = vk::VideoEncodeH264RateControlInfoKHR::default()
            .gop_frame_count(config.gop_size)
            .idr_period(config.gop_size);
        if config.b_frames == 0 {
            return info;
        }
        // The count of B-frames has to be below the GOP length, so a GOP
        // with no end is given the longest length there is.
        let gop_frame_count = if config.gop_size == 0 {
            u32::MAX
        } else {
            config.gop_size
        };
        info.gop_frame_count(gop_frame_count)
            .consecutive_b_frame_count(config.b_frames.min(gop_frame_count - 1))
            .flags(
                vk::VideoEncodeH264RateControlFlagsKHR::REGULAR_GOP
                    | if common.gop.b_reference_count() > 0 {
                        vk::VideoEncodeH264RateControlFlagsKHR::REFERENCE_PATTERN_DYADIC
                    } else {
                        vk::VideoEncodeH264RateControlFlagsKHR::REFERENCE_PATTERN_FLAT
                    },
            )
    }
}
