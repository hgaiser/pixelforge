use super::H264Encoder;

use crate::encoder::dpb::{DecodedPictureBufferTrait, DpbConfig, PictureStartInfo, PictureType};
use crate::encoder::gop::{GopFrameType, GopPosition};
use crate::encoder::{ColorDescription, EncodedPacket};
use crate::error::Result;
use crate::PixelForgeError;
use ash::vk;
use tracing::debug;

impl H264Encoder {
    /// Get the internal input image.
    ///
    /// This image can be used as a target for `ColorConverter::convert` to avoid
    /// an intermediate copy.
    pub fn input_image(&self) -> vk::Image {
        self.slots[self.current_slot].input_image
    }

    /// Encode a frame from a GPU image (depth-2 pipelined).
    ///
    /// Submits the frame to the encode queue without waiting, drains the
    /// previous in-flight frame from the slot we are about to overwrite,
    /// and returns *that* drained frame's packet. The first call returns
    /// an empty Vec (pipeline still filling); subsequent calls return one
    /// packet per call. Use `flush()` to drain remaining slots at end of stream.
    ///
    /// # Panics
    ///
    /// The encoder will panic at creation time if B-frames are enabled
    /// (b_frame_count > 0), as B-frame encoding is not yet supported.
    pub fn encode(&mut self, src_image: vk::Image) -> Result<Vec<EncodedPacket>> {
        let prev_packet = self.drain_current_slot()?;

        let gop_position = self.gop.get_next_frame();
        let display_order = self.input_frame_num;
        self.input_frame_num += 1;

        debug!(
            "Encoding frame {} from GPU image: type={:?}, poc={}, slot={}",
            display_order, gop_position.frame_type, gop_position.pic_order_cnt, self.current_slot
        );

        self.upload_from_image(src_image)?;
        self.encode_current_frame(&gop_position, display_order)?;

        self.current_slot = (self.current_slot + 1) % self.slots.len();
        Ok(prev_packet.into_iter().collect())
    }

    fn drain_current_slot(&mut self) -> Result<Option<EncodedPacket>> {
        if !self.slots[self.current_slot].in_flight {
            return Ok(None);
        }
        let bitstream = unsafe {
            crate::encoder::resources::wait_and_read_bitstream(
                self.context.device(),
                self.slots[self.current_slot].encode_fence,
                self.slots[self.current_slot].query_pool,
                self.slots[self.current_slot].bitstream_buffer_ptr,
            )?
        };
        self.slots[self.current_slot].in_flight = false;
        let meta = self.slots[self.current_slot]
            .pending_metadata
            .take()
            .ok_or_else(|| {
                PixelForgeError::CommandBuffer(
                    "Drained slot has bitstream but no metadata; encoder state corrupted"
                        .to_string(),
                )
            })?;
        let mut data = meta.header.unwrap_or_default();
        data.extend_from_slice(&bitstream);
        Ok(Some(EncodedPacket {
            data,
            frame_type: meta.frame_type,
            is_key_frame: meta.is_key_frame,
            pts: meta.pts,
            dts: meta.dts,
        }))
    }

    /// Internal method to encode the current frame already uploaded to input_image.
    fn encode_current_frame(
        &mut self,
        gop_position: &GopPosition,
        display_order: u64,
    ) -> Result<()> {
        let is_idr = gop_position.frame_type.is_idr();
        let is_reference = gop_position.is_reference;
        let is_b_frame = gop_position.frame_type == GopFrameType::B;
        let frame_type = match gop_position.frame_type {
            GopFrameType::Idr | GopFrameType::I => crate::encoder::FrameType::I,
            GopFrameType::P => crate::encoder::FrameType::P,
            GopFrameType::B => crate::encoder::FrameType::B,
        };

        debug!(
            "Encoding frame: display_order={}, type={:?}, idr={}, ref={}",
            display_order, frame_type, is_idr, is_reference
        );

        if is_idr {
            self.frame_num_syntax = 0;
            self.idr_pic_id = (self.idr_pic_id + 1) & 1;
            // Reset DPB by calling sequence_start with new config for IDR.
            let dpb_config = DpbConfig {
                dpb_size: self.dpb_slot_count as u32,
                max_num_ref_frames: self.config.max_reference_frames,
                use_multiple_references: self.config.b_frame_count > 0,
                log2_max_frame_num_minus4: 4,
                log2_max_pic_order_cnt_lsb_minus4: 4,
                ..Default::default()
            };
            self.dpb.h264.sequence_start(dpb_config);
            self.l0_references.clear();
            self.has_backward_reference = false;
        }

        let pic_order_cnt = gop_position.pic_order_cnt;
        let frame_num = self.frame_num_syntax;

        // For IDR frames, capture SPS/PPS header to be prepended at drain time.
        let header = if is_idr {
            let h = self.get_h264_header()?;
            self.sps_written = true;
            Some(h)
        } else {
            None
        };

        // Submit the encode (no wait, no readback). Marks the slot in_flight.
        self.encode_frame_internal(gop_position, frame_num, pic_order_cnt, is_idr)?;

        let dts = self.encode_frame_num;
        self.encode_frame_num += 1;
        if is_reference && !is_b_frame {
            self.frame_num_syntax = (self.frame_num_syntax + 1) % 256;
        }

        // Stash metadata so drain_current_slot() can build the packet later.
        self.slots[self.current_slot].pending_metadata = Some(super::SlotPacketMetadata {
            frame_type,
            is_key_frame: is_idr,
            pts: display_order,
            dts,
            header,
        });

        if is_reference {
            let pic_type = if is_idr {
                PictureType::Idr
            } else if is_b_frame {
                PictureType::B
            } else {
                PictureType::P
            };
            let pic_info = PictureStartInfo {
                frame_id: display_order,
                pic_order_cnt,
                frame_num,
                pic_type,
                is_reference,
                ..Default::default()
            };
            self.dpb.h264.picture_start(pic_info);
            self.dpb.h264.picture_end(is_reference);

            // Update reference tracking for the next P-frame.
            // The current frame becomes the reference for subsequent frames.
            let ref_info = super::ReferenceInfo {
                dpb_slot: self.current_dpb_slot,
                frame_num,
                poc: pic_order_cnt,
            };
            self.l0_references.insert(0, ref_info);
            // Limit to negotiated max active internal references
            while self.l0_references.len() > self.active_reference_count as usize {
                self.l0_references.pop();
            }

            // Find a free DPB slot for the next frame.
            let used_slots: Vec<u8> = self.l0_references.iter().map(|r| r.dpb_slot).collect();
            for i in 0..self.dpb_slot_count as u8 {
                if !used_slots.contains(&i) {
                    self.current_dpb_slot = i;
                    break;
                }
            }
        }

        Ok(())
    }

    /// Flush the encoder and drain any remaining in-flight slots.
    pub fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
        let mut out = Vec::new();
        for offset in 0..self.slots.len() {
            let idx = (self.current_slot + offset) % self.slots.len();
            if !self.slots[idx].in_flight {
                continue;
            }
            let saved_current = self.current_slot;
            self.current_slot = idx;
            if let Some(packet) = self.drain_current_slot()? {
                out.push(packet);
            }
            self.current_slot = saved_current;
        }
        Ok(out)
    }

    /// Request that the next frame be an IDR frame.
    pub fn request_idr(&mut self) {
        self.gop.request_idr();
    }

    /// Retrieve encoded SPS/PPS header data from video session parameters.
    /// This uses vkGetEncodedVideoSessionParametersKHR to get the NAL units.
    fn get_h264_header(&self) -> Result<Vec<u8>> {
        // H.264-specific get info requesting SPS and PPS.
        let mut h264_get_info = vk::VideoEncodeH264SessionParametersGetInfoKHR::default()
            .write_std_sps(true)
            .write_std_pps(true)
            .std_sps_id(0)
            .std_pps_id(0);

        let get_info = vk::VideoEncodeSessionParametersGetInfoKHR {
            video_session_parameters: self.session_params,
            p_next: (&mut h264_get_info as *mut vk::VideoEncodeH264SessionParametersGetInfoKHR)
                .cast(),
            ..Default::default()
        };

        // Some implementations misbehave for a size-only query (pData = NULL), especially with
        // 4:4:4 profiles. vk_video_samples avoids that by providing a preallocated buffer.
        let mut data = vec![0u8; 4096];
        let mut data_size: usize = data.len();
        let mut h264_feedback = vk::VideoEncodeH264SessionParametersFeedbackInfoKHR::default();
        let mut feedback = vk::VideoEncodeSessionParametersFeedbackInfoKHR {
            p_next: (&mut h264_feedback
                as *mut vk::VideoEncodeH264SessionParametersFeedbackInfoKHR)
                .cast(),
            ..Default::default()
        };

        let mut attempts = 0;
        loop {
            attempts += 1;
            let result = unsafe {
                (self
                    .video_encode_fn
                    .fp()
                    .get_encoded_video_session_parameters_khr)(
                    self.context.device().handle(),
                    &get_info,
                    &mut feedback,
                    &mut data_size,
                    data.as_mut_ptr() as *mut std::ffi::c_void,
                )
            };

            match result {
                vk::Result::SUCCESS => {
                    if data_size == 0 {
                        return Err(PixelForgeError::SessionParametersCreation(
                            "Encoded parameters size is 0".to_string(),
                        ));
                    }
                    data.truncate(data_size);
                    return Ok(data);
                }
                vk::Result::INCOMPLETE if attempts < 3 => {
                    // Driver indicates the buffer was too small; resize to the reported required
                    // size (or grow conservatively if the size is not provided).
                    let new_size = data_size.max(data.len() * 2).max(1);
                    data.resize(new_size, 0);
                    data_size = data.len();
                }
                err => {
                    return Err(PixelForgeError::SessionParametersCreation(format!(
                        "Failed to get encoded parameters: {:?}",
                        err
                    )));
                }
            }
        }
    }

    /// Update the color description (VUI parameters) in the encoded stream.
    ///
    /// This recreates the video session parameters with a new SPS containing the
    /// updated VUI color primaries, transfer characteristics, and matrix coefficients.
    /// The next encoded frame will be an IDR with the new SPS/PPS prepended.
    pub fn set_color_description(&mut self, desc: ColorDescription) -> Result<()> {
        // Wait for ALL slot fences before modifying session params. Do NOT reset
        // here; submit_encode_only resets each fence on submit so leaving them
        // signaled lets consecutive set_color_description() calls work safely.
        let fences: Vec<vk::Fence> = self.slots.iter().map(|s| s.encode_fence).collect();
        unsafe {
            self.context
                .device()
                .wait_for_fences(&fences, true, u64::MAX)
                .map_err(|e| {
                    PixelForgeError::Synchronization(format!(
                        "Failed to wait for encode fences: {:?}",
                        e
                    ))
                })?;
        }

        // Save old handle so we can destroy it after successful creation.
        let old_session_params = self.session_params;

        let new_session_params = self.create_session_params(&desc)?;

        // Destroy old session parameters now that the new ones are created.
        unsafe {
            (self
                .video_queue_fn
                .fp()
                .destroy_video_session_parameters_khr)(
                self.context.device().handle(),
                old_session_params,
                std::ptr::null(),
            );
        }

        self.session_params = new_session_params;
        self.config.color_description = Some(desc);
        self.sps_written = false;
        self.gop.request_idr();

        debug!(
            "H.264 color description updated: primaries={}, transfer={}, matrix={}, full_range={}",
            desc.color_primaries,
            desc.transfer_characteristics,
            desc.matrix_coefficients,
            desc.full_range
        );

        Ok(())
    }
}
