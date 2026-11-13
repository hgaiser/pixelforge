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
use crate::encoder::references::{Reference, ReferenceSet};
use crate::error::Result;
use ash::vk;

/// H.265 Coding Tree Block (CTB) size in pixels.
pub const CTB_SIZE: u32 = 32;

/// What H.265 needs to know about a reference besides what every codec
/// does.
#[derive(Clone, Copy, Debug)]
pub(crate) struct H265Reference {
    pub pic_type: ash::vk::native::StdVideoH265PictureType,
}

pub(crate) type ReferenceInfo = Reference<H265Reference>;

/// H.265-specific encoder state.
pub(crate) struct H265 {
    /// Cached VPS/SPS/PPS header (invalidated when session params change).
    header_data: Option<Vec<u8>>,
    /// References held in the DPB. Each frame predicts from the newest
    /// `active_reference_count`; the rest stay in its reference picture set,
    /// unused, for reference frame invalidation to fall back on.
    references: ReferenceSet<H265Reference>,
    /// How many of the newest references one frame predicts from.
    active_reference_count: u32,
    /// How many references shown before it a B-frame predicts from. It
    /// predicts from one shown after it.
    b_l0_reference_count: u32,
    profile_idc: u32,
}

/// The picture type of the frame `plan`.
fn picture_type(plan: &FramePlan) -> ash::vk::native::StdVideoH265PictureType {
    if plan.is_idr() {
        ash::vk::native::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_IDR
    } else if plan.is_b() {
        ash::vk::native::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_B
    } else {
        ash::vk::native::StdVideoH265PictureType_STD_VIDEO_H265_PICTURE_TYPE_P
    }
}

impl VideoCodec for H265 {
    fn begin_picture(
        &mut self,
        common: &mut EncoderCommon,
        plan: &FramePlan,
    ) -> Result<PictureSetup> {
        if plan.is_idr() {
            self.references.clear();
            // All DPB slots become inactive at the start of a coded video sequence.
            common.dpb_slot_active.fill(false);
        } else if plan.is_reference() {
            // A reference frame makes the B-frame references of its level
            // and below of no more use. Its reference picture set leaves
            // them out, which is what makes a decoder drop them.
            self.references.drop_obsolete(plan.pyramid_level());
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
        self.references.insert(Reference {
            dpb_slot: common.current_dpb_slot,
            poc: plan.pic_order_cnt(),
            display_order: plan.display_order,
            encode_index: plan.encode_index,
            level: plan.pyramid_level(),
            codec: H265Reference {
                pic_type: picture_type(plan),
            },
        });
        common.current_dpb_slot = self.references.free_slot(common.dpb_slot_count, None);
    }

    fn invalidate_references(
        &mut self,
        _common: &mut EncoderCommon,
        first_lost_display_order: u64,
    ) -> bool {
        // H.265 declares its short-term reference picture set explicitly every
        // frame from `references` (see `record`), so dropping entries here is
        // enough for the decoder to follow: no extra bitstream marking is
        // required.
        self.references.invalidate(first_lost_display_order);
        !self.references.is_empty()
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
