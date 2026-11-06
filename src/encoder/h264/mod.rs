//! H.264/AVC codec: the differences from the generic encoder.
//!
//! The shared session/DPB/pipeline machinery lives in `crate::encoder::codec`;
//! this folder holds only what is specific to H.264: its reference-picture
//! tracking and syntax counters (`H264`), the per-frame StdVideo* graph
//! (`record`), and the SPS/PPS generation (`session_params`).

mod init;
mod record;
mod session_params;

use crate::encoder::ColorDescription;
use crate::encoder::codec::{EncoderCommon, FramePlan, PictureSetup, VideoCodec};
use crate::encoder::pipeline::EncodeFuture;
use crate::error::Result;
use ash::vk;

/// H.264 macroblock size in pixels.
pub const MB_SIZE: u32 = 16;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ReferenceInfo {
    pub dpb_slot: u8,
    pub frame_num: u32,
    pub poc: i32,
    /// Display order (`pts`) of this reference, for reference frame invalidation.
    pub display_order: u64,
}

/// H.264-specific encoder state (everything the generic encoder doesn't own).
pub(crate) struct H264 {
    /// `frame_num` syntax element (mod `max_frame_num`).
    frame_num_syntax: u32,
    /// IDR picture id, toggled each IDR.
    idr_pic_id: u32,
    /// References held in the DPB, most-recent first. Each frame predicts
    /// from the first `active_reference_count`; the rest are kept for
    /// reference frame invalidation to fall back on.
    l0_references: Vec<ReferenceInfo>,
    /// How many references stay in the DPB.
    reference_window: u32,
    /// How many of the newest references one frame predicts from.
    active_reference_count: u32,
    /// `frame_num`s of references dropped by reference frame invalidation that
    /// still need to be marked unused-for-reference (via MMCO) in the next
    /// P-frame's slice header, so the decoder's DPB matches the encoder's.
    pending_unmark_frame_nums: Vec<u32>,
    /// Profile IDC (cached for parameter-set recreation).
    profile_idc: u32,
    /// Whether CABAC entropy coding is preferred (from quality-level query).
    preferred_entropy_cabac: bool,
}

impl VideoCodec for H264 {
    fn begin_picture(
        &mut self,
        common: &mut EncoderCommon,
        plan: &FramePlan,
    ) -> Result<PictureSetup> {
        if plan.is_idr() {
            self.frame_num_syntax = 0;
            self.idr_pic_id = (self.idr_pic_id + 1) & 1;
            // The IDR empties the DPB for the new coded video sequence.
            self.l0_references.clear();
            // An IDR resets the decoder's whole DPB, so any deferred
            // unused-reference markings are moot.
            self.pending_unmark_frame_nums.clear();
        }

        let header = if plan.is_idr() {
            Some(self.build_header(common)?)
        } else {
            None
        };

        Ok(PictureSetup { header })
    }

    fn record_picture(
        &mut self,
        common: &mut EncoderCommon,
        plan: &FramePlan,
    ) -> Result<EncodeFuture> {
        self.record(common, plan)
    }

    fn end_picture(&mut self, common: &mut EncoderCommon, plan: &FramePlan) {
        // `frame_num` carried by this frame (pre-increment), reused below for the
        // reference entry even after the syntax counter advances.
        let frame_num = self.frame_num_syntax;
        let pic_order_cnt = plan.pic_order_cnt();

        if plan.is_reference() {
            self.frame_num_syntax = (frame_num + 1) % 256;

            // The current frame becomes a reference for subsequent frames.
            self.l0_references.insert(
                0,
                ReferenceInfo {
                    dpb_slot: common.current_dpb_slot,
                    frame_num,
                    poc: pic_order_cnt,
                    display_order: plan.display_order,
                },
            );
            while self.l0_references.len() > self.reference_window as usize {
                self.l0_references.pop();
            }

            // Pick the next free DPB slot for the reconstructed picture.
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
        // Keep references older than the loss; every newer one is transitively
        // undecodable on the client. H.264 maintains the DPB with an implicit
        // sliding window, which would keep those newer pictures and desync the
        // decoder's reference list from ours — so each dropped reference is
        // queued to be explicitly marked unused-for-reference (MMCO) in the next
        // P-frame (see `record`).
        let mut survivors = Vec::with_capacity(self.l0_references.len());
        for r in self.l0_references.drain(..) {
            if r.display_order < first_lost_display_order {
                survivors.push(r);
            } else {
                self.pending_unmark_frame_nums.push(r.frame_num);
            }
        }
        self.l0_references = survivors;
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
        // H.264 regenerates SPS/PPS on every IDR, so there is nothing cached.
    }
}
