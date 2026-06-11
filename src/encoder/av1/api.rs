use super::AV1Encoder;

use crate::encoder::gop::{GopFrameType, GopPosition};
use crate::encoder::{ColorDescription, EncodedPacket};
use crate::error::{PixelForgeError, Result};
use ash::vk;
use tracing::debug;

impl AV1Encoder {
    /// Get the internal input image.
    ///
    /// This image can be used as a target for `ColorConverter::convert` to avoid
    /// an intermediate copy.
    pub fn input_image(&self) -> vk::Image {
        self.slots[self.current_slot].input_image
    }

    /// Encode a frame from a GPU image.
    ///
    /// This accepts a source image on the GPU and encodes it directly without
    /// any CPU-side data copies. The source image must be in the correct format
    /// with the same dimensions as the encoder configuration, and should be in GENERAL layout.
    ///
    /// # Panics
    ///
    /// The encoder will panic at creation time if B-frames are enabled (b_frame_count > 0),
    /// as B-frame encoding is not yet supported.
    pub fn encode(&mut self, src_image: vk::Image) -> Result<Vec<EncodedPacket>> {
        let prev_packet = self.drain_current_slot()?;

        let gop_position = self.gop.get_next_frame();
        let display_order = self.input_frame_num;
        self.input_frame_num += 1;

        debug!(
            "AV1 encode: frame {} type={:?}, slot={}",
            display_order, gop_position.frame_type, self.current_slot
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
        // AV1 always prefixes a Temporal Delimiter OBU; key frames also need
        // the sequence header captured at submit time.
        let mut data = vec![0x12, 0x00];
        if let Some(header) = meta.header {
            data.extend_from_slice(&header);
        }
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
        let is_key_frame =
            gop_position.frame_type.is_idr() || gop_position.frame_type == GopFrameType::I;
        let is_reference = gop_position.is_reference;
        let frame_type = match gop_position.frame_type {
            GopFrameType::Idr | GopFrameType::I => crate::encoder::FrameType::I,
            GopFrameType::P => crate::encoder::FrameType::P,
            GopFrameType::B => crate::encoder::FrameType::B,
        };

        debug!(
            "Encoding frame: display_order={}, type={:?}, key={}, ref={}",
            display_order, frame_type, is_key_frame, is_reference
        );

        if is_key_frame {
            self.frame_num = 0;
            self.order_hint = 0;
            // Reset references for key frames.
            self.references.clear();
            // Reset DPB slot activation tracking on key frame - all slots become inactive.
            for active in &mut self.dpb_slot_active {
                *active = false;
            }
        }

        // For key frames, capture the AV1 Sequence Header OBU to be prepended
        // at drain time. (The Temporal Delimiter prefix is added in
        // drain_current_slot for every frame.)
        let header = if is_key_frame {
            if self.header_data.is_none() {
                let h = self.get_av1_sequence_header()?;
                debug!(
                    "AV1 sequence header ({} bytes): {:02X?}",
                    h.len(),
                    &h[..std::cmp::min(32, h.len())]
                );
                self.header_data = Some(h);
            }
            self.header_data.clone()
        } else {
            None
        };

        // Submit the encode (no wait, no readback). Marks the slot in_flight.
        self.encode_frame_internal(gop_position, is_key_frame)?;

        let encoded_order_hint = self.order_hint;
        let dts = self.encode_frame_num;
        self.encode_frame_num += 1;
        self.frame_num += 1;
        self.order_hint = (self.order_hint + 1) & 0xFF; // 8-bit order hint

        self.slots[self.current_slot].pending_metadata = Some(super::SlotPacketMetadata {
            frame_type,
            is_key_frame,
            pts: display_order,
            dts,
            header,
        });

        // Only KEY frames are stored as references. P frames all reference the KEY frame
        // and don't update any reference buffer, avoiding P→P which produces corrupt output
        // on NVIDIA AV1 encoders.
        if is_key_frame {
            let ref_info = super::ReferenceInfo {
                dpb_slot: self.current_dpb_slot,
                order_hint: encoded_order_hint,
                frame_type: ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_KEY,
            };
            self.references.clear();
            self.references.push(ref_info);

            // KEY frame uses the current DPB slot; pick a different one for P frames.
            let used_slots: Vec<u8> = self.references.iter().map(|r| r.dpb_slot).collect();
            for i in 0..self.dpb_slot_count as u8 {
                if !used_slots.contains(&i) {
                    self.current_dpb_slot = i;
                    break;
                }
            }
        }
        // P frames reuse the same scratch DPB slot (current_dpb_slot stays unchanged
        // between P frames since it's always different from the KEY frame's slot).

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

    /// Request that the next frame be an IDR/key frame.
    pub fn request_idr(&mut self) {
        self.gop.request_idr();
    }

    /// Retrieve encoded AV1 Sequence Header OBU from video session parameters.
    ///
    /// Uses vkGetEncodedVideoSessionParametersKHR to get the driver-generated OBU.
    /// The driver's sequence header must be used because the frame OBUs it produces
    /// reference values from its internal sequence header (not ours).
    fn get_av1_sequence_header(&self) -> Result<Vec<u8>> {
        let get_info = vk::VideoEncodeSessionParametersGetInfoKHR {
            video_session_parameters: self.session_params,
            ..Default::default()
        };

        let mut data = vec![0u8; 4096];
        let mut data_size: usize = data.len();
        let mut feedback = vk::VideoEncodeSessionParametersFeedbackInfoKHR::default();

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
                            "AV1 sequence header size is 0".to_string(),
                        ));
                    }
                    data.truncate(data_size);
                    debug!("Retrieved AV1 sequence header: {} bytes", data.len());
                    return Ok(data);
                }
                vk::Result::INCOMPLETE if attempts < 3 => {
                    let new_size = data_size.max(data.len() * 2).max(1);
                    data.resize(new_size, 0);
                    data_size = data.len();
                }
                err => {
                    return Err(PixelForgeError::SessionParametersCreation(format!(
                        "Failed to get AV1 sequence header: {:?}",
                        err
                    )));
                }
            }
        }
    }

    /// Update the color description in the AV1 sequence header.
    ///
    /// This recreates the video session parameters with a new sequence header
    /// containing the updated color configuration. The next encoded frame will
    /// be a key frame with the new sequence header prepended.
    pub fn set_color_description(&mut self, desc: ColorDescription) -> Result<()> {
        // Wait for ALL slot fences before modifying session params. Do NOT reset
        // here; submit_encode_only resets each fence on submit.
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
            self.video_queue_fn
                .destroy_video_session_parameters(old_session_params, None);
        }

        self.session_params = new_session_params;
        self.config.color_description = Some(desc);
        self.header_data = None; // Invalidate cached sequence header
        self.gop.request_idr();

        debug!(
            "AV1 color description updated: primaries={}, transfer={}, matrix={}, full_range={}",
            desc.color_primaries,
            desc.transfer_characteristics,
            desc.matrix_coefficients,
            desc.full_range
        );

        Ok(())
    }
}
