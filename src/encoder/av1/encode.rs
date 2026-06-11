use super::AV1Encoder;

use crate::encoder::gop::GopPosition;
use crate::encoder::resources::{
    prepare_encode_command_buffer, record_dpb_barriers, record_post_encode_dpb_barrier,
    submit_encode_only,
};
use crate::error::{PixelForgeError, Result};
use ash::vk;
use tracing::debug;

impl AV1Encoder {
    /// Records and submits the encode commands for a single frame to the
    /// current slot. Does NOT wait for completion — see encoder::h265 for the
    /// pipelining contract.
    pub(super) fn encode_frame_internal(
        &mut self,
        _gop_position: &GopPosition,
        is_key_frame: bool,
    ) -> Result<()> {
        // All frames need a setup reference slot (DPB write) per Vulkan spec when maxDpbSlots > 0.
        let is_reference = true;

        debug!(
            "encode_frame_internal: key={}, ref={}, refs_len={}, dpb_slot={}",
            is_key_frame,
            is_reference,
            self.references.len(),
            self.current_dpb_slot
        );

        // Rate control setup (matches H265 pattern: CQP/Disabled uses DISABLED mode).
        let (rc_mode, average_bitrate, max_bitrate, qp) = match self.config.rate_control_mode {
            crate::encoder::RateControlMode::Cqp | crate::encoder::RateControlMode::Disabled => (
                vk::VideoEncodeRateControlModeFlagsKHR::DISABLED,
                0,
                0,
                self.config.quality_level,
            ),
            crate::encoder::RateControlMode::Cbr => (
                vk::VideoEncodeRateControlModeFlagsKHR::CBR,
                self.config.target_bitrate,
                self.config.target_bitrate,
                128u32,
            ),
            crate::encoder::RateControlMode::Vbr => (
                vk::VideoEncodeRateControlModeFlagsKHR::VBR,
                self.config.target_bitrate,
                self.config.max_bitrate,
                128u32,
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
        let ref_dpb_slots: Vec<u8> = self.references.iter().map(|r| r.dpb_slot).collect();
        unsafe {
            record_dpb_barriers(
                self.context.device(),
                self.slots[self.current_slot].encode_command_buffer,
                &self.dpb_images,
                false, // AV1 does not use layered DPB
                self.current_dpb_slot,
                &ref_dpb_slots,
                self.dpb_slot_active[self.current_dpb_slot as usize],
            );
        }

        // AV1 frame type.
        let frame_type = if is_key_frame {
            ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_KEY
        } else {
            ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_INTER
        };

        // Build picture info flags using ash's accessor methods.
        // show_frame must be set for all frames; error_resilient_mode for key frames (match FFmpeg).
        let mut picture_info_flags = ash::vk::native::StdVideoEncodeAV1PictureInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: Default::default(),
        };
        picture_info_flags.set_show_frame(1);
        if is_key_frame {
            picture_info_flags.set_error_resilient_mode(1);
        } else {
            picture_info_flags.set_showable_frame(1);
        }

        // Frame extent uses display dimensions for all picture resources.
        // Per Vulkan spec, without MOTION_VECTOR_SCALING support, all picture resource
        // codedExtent values must match, and srcPictureResource.codedExtent must equal
        // the sequence header's max_frame_width/height.
        let frame_extent = vk::Extent2D {
            width: self.config.dimensions.width,
            height: self.config.dimensions.height,
        };

        // Setup reconstructed picture (DPB slot for output).
        let setup_picture_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(frame_extent)
            .base_array_layer(0)
            .image_view_binding(self.dpb_image_views[self.current_dpb_slot as usize]);
        // AV1 reference info for the setup slot.
        let reference_info_flags = ash::vk::native::StdVideoEncodeAV1ReferenceInfoFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoEncodeAV1ReferenceInfoFlags::new_bitfield_1(
                0, 0, 0,
            ),
        };

        let std_reference_info = ash::vk::native::StdVideoEncodeAV1ReferenceInfo {
            flags: reference_info_flags,
            frame_type: if is_key_frame {
                ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_KEY
            } else {
                ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_INTER
            },
            RefFrameId: self.current_dpb_slot as u32,
            OrderHint: self.order_hint as u8,
            reserved1: [0; 3],
            pExtensionHeader: std::ptr::null(),
        };

        // AV1 DPB slot info for the setup reference slot (the slot being written).
        let setup_av1_dpb_info =
            vk::VideoEncodeAV1DpbSlotInfoKHR::default().std_reference_info(&std_reference_info);

        let mut setup_reference_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(self.current_dpb_slot as i32)
            .picture_resource(&setup_picture_resource);

        // Attach AV1 DPB info to setup slot's pNext chain.
        setup_reference_slot.p_next =
            (&setup_av1_dpb_info as *const vk::VideoEncodeAV1DpbSlotInfoKHR).cast();

        // Reference frames for inter frames.
        let mut reference_slots = Vec::new();
        let mut av1_reference_infos = Vec::new();
        let mut ref_picture_resources = Vec::new();
        let mut ref_std_infos = Vec::new(); // Store std info to keep it alive

        if !is_key_frame && !self.references.is_empty() {
            // Use the most recent reference frame.
            let ref_info = &self.references[0];

            // Create StdVideoEncodeAV1ReferenceInfo for the reference slot.
            let ref_std_info = ash::vk::native::StdVideoEncodeAV1ReferenceInfo {
                flags: reference_info_flags,
                frame_type: ref_info.frame_type,
                RefFrameId: ref_info.dpb_slot as u32,
                OrderHint: ref_info.order_hint as u8,
                reserved1: [0; 3],
                pExtensionHeader: std::ptr::null(),
            };
            ref_std_infos.push(ref_std_info);

            // Create AV1 DPB slot info for the reference (without pointer first).
            let av1_ref_info = vk::VideoEncodeAV1DpbSlotInfoKHR::default();
            av1_reference_infos.push(av1_ref_info);
            // Now set the pointer after it's in the vector at its final location.
            av1_reference_infos[0] = av1_reference_infos[0].std_reference_info(&ref_std_infos[0]);

            let ref_picture_resource = vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D { x: 0, y: 0 })
                .coded_extent(frame_extent)
                .base_array_layer(0)
                .image_view_binding(self.dpb_image_views[ref_info.dpb_slot as usize]);
            ref_picture_resources.push(ref_picture_resource);

            // Create reference slot (without pNext first).
            let ref_slot = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(ref_info.dpb_slot as i32)
                .picture_resource(&ref_picture_resources[0]);
            reference_slots.push(ref_slot);
            // Now set pNext after it's in the vector at its final location.
            reference_slots[0].p_next =
                (&av1_reference_infos[0] as *const vk::VideoEncodeAV1DpbSlotInfoKHR).cast();
        }

        // AV1 quantization parameters - required structure.
        // Start with a moderate QP that the rate controller can adjust.
        let quantization_flags = ash::vk::native::StdVideoAV1QuantizationFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoAV1QuantizationFlags::new_bitfield_1(
                0, // using_qmatrix
                0, // diff_uv_delta
                0, // reserved
            ),
        };

        let quantization = ash::vk::native::StdVideoAV1Quantization {
            flags: quantization_flags,
            base_q_idx: qp as u8, // Use the same QP as constant_q_index
            DeltaQYDc: 0,
            DeltaQUDc: 0,
            DeltaQUAc: 0,
            DeltaQVDc: 0,
            DeltaQVAc: 0,
            qm_y: 0,
            qm_u: 0,
            qm_v: 0,
        };

        // CDEF (Constrained Directional Enhancement Filter) - required since we enabled it in sequence header.
        // Match FFmpeg's default initialization (all zeros).
        let cdef = ash::vk::native::StdVideoAV1CDEF {
            cdef_damping_minus_3: 0,                       // Match FFmpeg: damping = 3
            cdef_bits: 0,                                  // 1 CDEF strength combination (2^0)
            cdef_y_pri_strength: [0, 0, 0, 0, 0, 0, 0, 0], // Match FFmpeg: all zeros
            cdef_y_sec_strength: [0, 0, 0, 0, 0, 0, 0, 0],
            cdef_uv_pri_strength: [0, 0, 0, 0, 0, 0, 0, 0],
            cdef_uv_sec_strength: [0, 0, 0, 0, 0, 0, 0, 0],
        };

        // Loop filter - deblocking filter parameters.
        // Match FFmpeg's default initialization.
        let loop_filter_flags = ash::vk::native::StdVideoAV1LoopFilterFlags {
            _bitfield_align_1: [],
            _bitfield_1: ash::vk::native::StdVideoAV1LoopFilterFlags::new_bitfield_1(
                0, // loop_filter_delta_enabled
                0, // loop_filter_delta_update
                0, // reserved
            ),
        };

        let loop_filter = ash::vk::native::StdVideoAV1LoopFilter {
            flags: loop_filter_flags,
            loop_filter_level: [0, 0, 0, 0], // Match FFmpeg: disable filter initially
            loop_filter_sharpness: 0,
            update_ref_delta: 0,
            // Match FFmpeg's default_loop_filter_ref_deltas: { 1, 0, 0, 0, -1, 0, -1, -1 }
            loop_filter_ref_deltas: [1, 0, 0, 0, -1, 0, -1, -1],
            update_mode_delta: 1, // Match FFmpeg: set to 1
            loop_filter_mode_deltas: [0; 2],
        };

        // Tile info - FFmpeg has this commented out with "TODO FIX" at line 340.
        // Match FFmpeg: don't provide tile info (set to null).
        // Note: If this causes issues, we may need to re-enable it with proper values.

        // Build ref_frame_idx, ref_order_hint, refresh_frame_flags, and primary_ref_frame.
        //
        // All P frames reference only the KEY frame (stored in buffer 0).
        // P frames set refresh_frame_flags=0x00 (don't update any reference buffer).
        // This avoids P→P references which produce corrupt output on NVIDIA AV1 encoders.
        let (ref_frame_idx, ref_order_hint, primary_ref_frame, refresh_frame_flags) =
            if !is_key_frame && !self.references.is_empty() {
                // All reference names point to buffer 0 (where KEY frame lives).
                let ref_idx = [0i8; 7];
                let ref_info = &self.references[0];
                let mut order_hints = [0u8; 8];
                order_hints[0] = ref_info.order_hint as u8;

                // P frames don't refresh any reference buffer.
                (ref_idx, order_hints, 0u8, 0x00u8)
            } else {
                // KEY frame: refresh all buffers.
                ([0i8; 7], [0u8; 8], 7u8, 0xFFu8)
            };

        // AV1 encode picture info.
        let std_picture_info = ash::vk::native::StdVideoEncodeAV1PictureInfo {
            flags: picture_info_flags,
            frame_type,
            frame_presentation_time: self.frame_num,
            current_frame_id: self.current_dpb_slot as u32, // Match FFmpeg: slot index
            order_hint: self.order_hint as u8,
            primary_ref_frame,
            refresh_frame_flags,
            coded_denom: 0,
            render_width_minus_1: (self.config.dimensions.width - 1) as u16,
            render_height_minus_1: (self.config.dimensions.height - 1) as u16,
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
            pLoopRestoration: std::ptr::null(),
            pGlobalMotion: std::ptr::null(),
            pExtensionHeader: std::ptr::null(),
            pBufferRemovalTimes: std::ptr::null(),
        };

        // Reference name slot indices - maps AV1 reference names to Vulkan DPB slot indices.
        // Only set entries for reference names that appear in pReferenceSlots.
        // For SINGLE_REFERENCE mode, only LAST_FRAME (index 0) is used.
        let mut reference_name_slot_indices = [-1i32; 7];

        if !is_key_frame && !self.references.is_empty() {
            // Map LAST_FRAME to the reference's DPB slot.
            let ref_info = &self.references[0];
            reference_name_slot_indices[0] = ref_info.dpb_slot as i32;
        }

        // Set prediction mode and rate control group based on frame type.
        let (prediction_mode, rate_control_group) = if is_key_frame {
            (
                vk::VideoEncodeAV1PredictionModeKHR::INTRA_ONLY,
                vk::VideoEncodeAV1RateControlGroupKHR::INTRA,
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

        // For DISABLED rate control mode, set constant_q_index on the picture info.
        if rc_mode == vk::VideoEncodeRateControlModeFlagsKHR::DISABLED {
            av1_picture_info = av1_picture_info.constant_q_index(qp);
        }

        // AV1-specific rate control layer info.
        let min_q_index = vk::VideoEncodeAV1QIndexKHR {
            intra_q_index: qp,
            predictive_q_index: qp,
            bipredictive_q_index: qp,
        };
        let max_q_index = vk::VideoEncodeAV1QIndexKHR {
            intra_q_index: qp,
            predictive_q_index: qp,
            bipredictive_q_index: qp,
        };
        let mut av1_rc_layer_info = vk::VideoEncodeAV1RateControlLayerInfoKHR::default()
            .use_min_q_index(true)
            .min_q_index(min_q_index)
            .use_max_q_index(true)
            .max_q_index(max_q_index);

        let mut rc_layer_info = vk::VideoEncodeRateControlLayerInfoKHR::default()
            .average_bitrate(average_bitrate as u64)
            .max_bitrate(max_bitrate as u64)
            .frame_rate_numerator(self.config.frame_rate_numerator)
            .frame_rate_denominator(self.config.frame_rate_denominator);
        rc_layer_info.p_next =
            (&mut av1_rc_layer_info as *mut vk::VideoEncodeAV1RateControlLayerInfoKHR).cast();
        let rc_layers = [rc_layer_info];

        // AV1-specific rate control info.
        let mut av1_rc_info = vk::VideoEncodeAV1RateControlInfoKHR::default()
            .gop_frame_count(self.config.gop_size)
            .key_frame_period(self.config.gop_size)
            .consecutive_bipredictive_frame_count(0)
            .temporal_layer_count(1);

        // Rate control info (matches H265 pattern: only add layers/buffer for non-DISABLED modes).
        let mut rc_info = vk::VideoEncodeRateControlInfoKHR::default().rate_control_mode(rc_mode);

        if rc_mode != vk::VideoEncodeRateControlModeFlagsKHR::DISABLED {
            rc_info = rc_info
                .layers(&rc_layers)
                .virtual_buffer_size_in_ms(self.config.virtual_buffer_size_ms)
                .initial_virtual_buffer_size_in_ms(self.config.initial_virtual_buffer_size_ms);
            rc_info.p_next = (&mut av1_rc_info as *mut vk::VideoEncodeAV1RateControlInfoKHR).cast();
        }

        // Video begin coding info.
        // Include the setup slot (with slot_index -1 to indicate it's not yet active)
        // and any reference slots that will be used for reading during encoding.
        let mut all_reference_slots = Vec::new();

        if is_reference {
            // Build a separate setup slot for begin coding with slot_index = -1.
            // This tells the implementation the slot is being set up, not yet active.
            let mut setup_slot_for_begin = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(-1)
                .picture_resource(&setup_picture_resource);
            setup_slot_for_begin.p_next =
                (&setup_av1_dpb_info as *const vk::VideoEncodeAV1DpbSlotInfoKHR).cast();
            all_reference_slots.push(setup_slot_for_begin);
        }

        // Add reference slots (already active slots we're reading from)
        all_reference_slots.extend_from_slice(&reference_slots);

        debug!(
            "Begin coding: {} reference slots (setup={}, refs={})",
            all_reference_slots.len(),
            if is_reference { 1 } else { 0 },
            reference_slots.len()
        );

        // Begin video coding with rate control info for non-first frames.
        let is_first_frame = self.encode_frame_num == 0;

        let begin_coding_info = if is_first_frame {
            vk::VideoBeginCodingInfoKHR::default()
                .video_session(self.session)
                .video_session_parameters(self.session_params)
                .reference_slots(&all_reference_slots)
        } else {
            let mut info = vk::VideoBeginCodingInfoKHR::default()
                .video_session(self.session)
                .video_session_parameters(self.session_params)
                .reference_slots(&all_reference_slots);
            info.p_next = (&rc_info as *const vk::VideoEncodeRateControlInfoKHR).cast();
            info
        };

        unsafe {
            self.video_queue_fn.cmd_begin_video_coding(
                self.slots[self.current_slot].encode_command_buffer,
                &begin_coding_info,
            );
        }

        // Reset video coding state for the first frame.
        // Combine RESET + RATE_CONTROL + QUALITY_LEVEL into a single control command.
        // This matches the H265 approach and is required for AMD RADV.
        if is_first_frame {
            let mut quality_level_info =
                vk::VideoEncodeQualityLevelInfoKHR::default().quality_level(0);
            quality_level_info.p_next =
                (&rc_info as *const vk::VideoEncodeRateControlInfoKHR).cast();

            let mut control_info = vk::VideoCodingControlInfoKHR::default().flags(
                vk::VideoCodingControlFlagsKHR::RESET
                    | vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL
                    | vk::VideoCodingControlFlagsKHR::ENCODE_QUALITY_LEVEL,
            );
            control_info.p_next =
                (&quality_level_info as *const vk::VideoEncodeQualityLevelInfoKHR).cast();

            unsafe {
                self.video_queue_fn.cmd_control_video_coding(
                    self.slots[self.current_slot].encode_command_buffer,
                    &control_info,
                );
            }
        }

        // Encode info.
        let src_picture_resource = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(frame_extent)
            .base_array_layer(0)
            .image_view_binding(self.slots[self.current_slot].input_image_view);

        let mut encode_info = vk::VideoEncodeInfoKHR::default()
            .src_picture_resource(src_picture_resource)
            .dst_buffer(self.slots[self.current_slot].bitstream_buffer)
            .dst_buffer_offset(0)
            .dst_buffer_range(self.slots[self.current_slot].bitstream_buffer_size as u64);

        if is_reference {
            encode_info = encode_info.setup_reference_slot(&setup_reference_slot);
        }

        if !reference_slots.is_empty() {
            encode_info = encode_info.reference_slots(&reference_slots);
        }

        encode_info.p_next = (&av1_picture_info as *const vk::VideoEncodeAV1PictureInfoKHR).cast();

        // Begin query to capture encode feedback (bitstream size, status).
        unsafe {
            self.context.device().cmd_begin_query(
                self.slots[self.current_slot].encode_command_buffer,
                self.slots[self.current_slot].query_pool,
                0,
                vk::QueryControlFlags::empty(),
            );
        }

        unsafe {
            self.video_encode_fn.cmd_encode_video(
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

        // Add DPB synchronization barrier after encoding.
        unsafe {
            record_post_encode_dpb_barrier(
                self.context.device(),
                self.slots[self.current_slot].encode_command_buffer,
                &self.dpb_images,
                false, // AV1 does not use layered DPB
                self.current_dpb_slot,
            );
        }

        // End video coding.
        let end_coding_info = vk::VideoEndCodingInfoKHR::default();
        unsafe {
            self.video_queue_fn.cmd_end_video_coding(
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
            "Submitting frame {} to GPU: key={}, num_refs={}, cur_slot={}",
            self.encode_frame_num,
            is_key_frame,
            self.references.len(),
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

        // Mark current DPB slot as active.
        self.dpb_slot_active[self.current_dpb_slot as usize] = true;

        // Mark slot as in-flight; bitstream is drained on next encode() call.
        self.slots[self.current_slot].in_flight = true;

        Ok(())
    }
}
