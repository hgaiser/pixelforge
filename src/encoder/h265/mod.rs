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
use crate::encoder::gop::obsolete_b_references;
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
    /// 0 for an IDR or P-frame, and the pyramid level of a B-frame.
    pub level: u8,
}

/// H.265-specific encoder state.
pub(crate) struct H265 {
    /// Cached VPS/SPS/PPS header (invalidated when session params change).
    header_data: Option<Vec<u8>>,
    /// References held in the DPB, most-recent first. Each frame predicts
    /// from the first `active_reference_count`; the rest stay in its
    /// reference picture set, unused, for reference frame invalidation to fall
    /// back on.
    l0_references: Vec<ReferenceInfo>,
    /// How many references stay in the DPB.
    reference_window: u32,
    /// How many of the newest references one frame predicts from.
    active_reference_count: u32,
    /// How many references shown before it a B-frame predicts from. It
    /// predicts from one shown after it.
    b_l0_reference_count: u32,
    profile_idc: u32,
}

impl VideoCodec for H265 {
    fn begin_picture(
        &mut self,
        common: &mut EncoderCommon,
        plan: &FramePlan,
    ) -> Result<PictureSetup> {
        if plan.is_idr() {
            self.l0_references.clear();
            // All DPB slots become inactive at the start of a coded video sequence.
            common.dpb_slot_active.fill(false);
        } else if plan.is_reference() {
            // A reference frame makes the B-frame references of its level
            // and below of no more use. Its reference picture set leaves
            // them out, which is what makes a decoder drop them.
            let levels: Vec<u8> = self.l0_references.iter().map(|r| r.level).collect();
            for index in obsolete_b_references(&levels, plan.pyramid_level())
                .into_iter()
                .rev()
            {
                self.l0_references.remove(index);
            }
        }

        let header = if plan.is_idr() {
            if self.header_data.is_none() {
                self.header_data = Some(self.build_header(common)?);
            }
            self.header_data.clone()
        } else {
            None
        };

        Ok(PictureSetup { header })
    }

    fn record_picture(&mut self, common: &mut EncoderCommon, plan: &FramePlan) -> Result<()> {
        self.record(common, plan)
    }

    fn end_picture(&mut self, common: &mut EncoderCommon, plan: &FramePlan) {
        if !plan.is_reference() {
            return;
        }
        let pic_order_cnt = plan.pic_order_cnt();

        self.l0_references.insert(
            0,
            ReferenceInfo {
                dpb_slot: common.current_dpb_slot,
                poc: pic_order_cnt,
                display_order: plan.display_order,
                level: plan.pyramid_level(),
            },
        );
        // The window counts the P-frames. The B-frame references on top of
        // it go when a later reference frame replaces them.
        while self.l0_references.iter().filter(|r| r.level == 0).count()
            > self.reference_window as usize
        {
            let oldest = self.l0_references.iter().rposition(|r| r.level == 0);
            self.l0_references.remove(oldest.expect("counted above"));
        }

        let used_slots: Vec<u8> = self.l0_references.iter().map(|r| r.dpb_slot).collect();
        for i in 0..common.dpb_slot_count as u8 {
            if !used_slots.contains(&i) {
                common.current_dpb_slot = i;
                break;
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
