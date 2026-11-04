//! H.265/HEVC codec: the differences from the generic encoder.
//!
//! Shared machinery lives in `crate::encoder::codec`; this folder holds only
//! H.265's reference tracking (`H265`), its per-frame StdVideo* graph
//! (`record`), and its VPS/SPS/PPS generation (`session_params`).

mod init;
mod record;
mod session_params;

use crate::encoder::ColorDescription;
use crate::encoder::codec::{EncoderCommon, FramePlan, PictureSetup, VideoCodec};
use crate::encoder::dpb::{
    DecodedPictureBuffer, DecodedPictureBufferTrait, DpbConfig, PictureStartInfo, PictureType,
};
use crate::encoder::pipeline::EncodeFuture;
use crate::error::Result;
use ash::vk;

/// H.265 Coding Tree Block (CTB) size in pixels.
pub const CTB_SIZE: u32 = 32;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ReferenceInfo {
    pub dpb_slot: u8,
    pub poc: i32,
    /// Display order (`pts`) of this reference, for reference frame invalidation.
    pub display_order: u64,
}

/// H.265-specific encoder state.
pub(crate) struct H265 {
    dpb: DecodedPictureBuffer,
    /// Cached VPS/SPS/PPS header (invalidated when session params change).
    header_data: Option<Vec<u8>>,
    has_backward_reference: bool,
    backward_reference_poc: i32,
    backward_reference_dpb_slot: u8,
    /// References held in the DPB, most-recent first. Each frame predicts
    /// from the first `active_reference_count`; the rest stay in its
    /// reference picture set, unused, for reference frame invalidation to fall
    /// back on.
    l0_references: Vec<ReferenceInfo>,
    /// How many references stay in the DPB.
    reference_window: u32,
    /// How many of the newest references one frame predicts from.
    active_reference_count: u32,
    profile_idc: u32,
}

impl H265 {
    /// The references this frame predicts from: the newest held ones.
    fn used_references(&self) -> &[ReferenceInfo] {
        let n = self
            .l0_references
            .len()
            .min(self.active_reference_count as usize);
        &self.l0_references[..n]
    }
}

impl VideoCodec for H265 {
    fn begin_picture(
        &mut self,
        common: &mut EncoderCommon,
        plan: &FramePlan,
    ) -> Result<PictureSetup> {
        if plan.is_idr() {
            let dpb_config = DpbConfig {
                dpb_size: common.dpb_slot_count as u32,
                max_num_ref_frames: self.reference_window,
                use_multiple_references: common.config.b_frame_count > 0,
                log2_max_frame_num_minus4: 0,
                log2_max_pic_order_cnt_lsb_minus4: 4,
                ..Default::default()
            };
            self.dpb.h265.sequence_start(dpb_config);
            self.l0_references.clear();
            self.has_backward_reference = false;
            // All DPB slots become inactive at the start of a coded video sequence.
            common.dpb_slot_active.fill(false);
        }

        let header = if plan.is_idr() {
            if self.header_data.is_none() {
                self.header_data = Some(self.build_header(common)?);
            }
            self.header_data.clone()
        } else {
            None
        };

        Ok(PictureSetup {
            frame_type: plan.frame_type(),
            header,
        })
    }

    fn record_picture(
        &mut self,
        common: &mut EncoderCommon,
        plan: &FramePlan,
    ) -> Result<EncodeFuture> {
        self.record(common, plan)
    }

    fn end_picture(&mut self, common: &mut EncoderCommon, plan: &FramePlan) {
        if !plan.is_reference() {
            return;
        }
        let pic_order_cnt = plan.pic_order_cnt();
        let pic_type = if plan.is_idr() {
            PictureType::Idr
        } else if plan.is_b_frame() {
            PictureType::B
        } else {
            PictureType::P
        };
        let pic_info = PictureStartInfo {
            frame_id: plan.display_order,
            pic_order_cnt,
            frame_num: 0,
            pic_type,
            is_reference: true,
            ..Default::default()
        };
        self.dpb.h265.picture_start(pic_info);
        self.dpb.h265.picture_end(true);

        self.l0_references.insert(
            0,
            ReferenceInfo {
                dpb_slot: common.current_dpb_slot,
                poc: pic_order_cnt,
                display_order: plan.display_order,
            },
        );
        while self.l0_references.len() > self.reference_window as usize {
            self.l0_references.pop();
        }

        if !plan.is_b_frame() {
            let used_slots: Vec<u8> = self.l0_references.iter().map(|r| r.dpb_slot).collect();
            for i in 0..common.dpb_slot_count as u8 {
                if !used_slots.contains(&i) {
                    common.current_dpb_slot = i;
                    break;
                }
            }
        }
    }

    fn invalidate_references(
        &mut self,
        _common: &mut EncoderCommon,
        first_lost_display_order: u64,
    ) -> bool {
        // Every reference at or after the first lost frame is transitively
        // undecodable on the client; keep only older survivors. H.265 declares
        // its short-term reference picture set explicitly every frame from
        // `l0_references` (see `record`), so dropping entries here is enough for
        // the decoder to follow — no extra bitstream marking is required.
        self.l0_references
            .retain(|r| r.display_order < first_lost_display_order);
        // A backward (L1) reference may also be tainted; B-frames are the only
        // consumer of it, so simply forget it on invalidation.
        self.has_backward_reference = false;
        !self.l0_references.is_empty()
    }

    fn create_session_params(
        &self,
        common: &EncoderCommon,
        desc: &ColorDescription,
    ) -> Result<vk::VideoSessionParametersKHR> {
        self.build_session_params(common, desc)
    }

    fn invalidate_header_cache(&mut self) {
        self.header_data = None;
    }
}
