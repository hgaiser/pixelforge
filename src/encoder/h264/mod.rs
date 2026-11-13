//! H.264/AVC codec: the differences from the generic encoder.
//!
//! The shared session/DPB/pipeline machinery lives in `crate::encoder::codec`;
//! this folder holds only what is specific to H.264: its reference-picture
//! tracking and syntax counters (`H264`), the per-frame StdVideo* graph
//! (`record`), and the SPS/PPS generation (`session_params`).

mod init;
mod level;
mod lists;
mod record;
mod session_params;

use crate::encoder::ColorDescription;
use crate::encoder::codec::{EncoderCommon, FramePlan, PictureSetup, VideoCodec};
use crate::encoder::references::{Reference, ReferenceSet};
use crate::error::Result;
use ash::vk;

/// H.264 macroblock size in pixels.
pub const MB_SIZE: u32 = 16;

/// What H.264 needs to know about a reference besides what every codec
/// does.
#[derive(Clone, Copy, Debug)]
pub(crate) struct H264Reference {
    pub frame_num: u32,
    pub pic_type: ash::vk::native::StdVideoH264PictureType,
}

pub(crate) type ReferenceInfo = Reference<H264Reference>;

/// H.264-specific encoder state (everything the generic encoder doesn't own).
pub(crate) struct H264 {
    /// `frame_num` syntax element (mod `max_frame_num`).
    frame_num_syntax: u32,
    /// IDR picture id, toggled each IDR.
    idr_pic_id: u32,
    /// References held in the DPB. Each frame predicts from the newest
    /// `active_reference_count`; the rest are kept for reference frame
    /// invalidation to fall back on.
    references: ReferenceSet<H264Reference>,
    /// How many references stay in the DPB.
    reference_window: u32,
    /// How many of the newest references one frame predicts from.
    active_reference_count: u32,
    /// How many references shown before it a B-frame predicts from. It
    /// predicts from one shown after it.
    b_l0_reference_count: u32,
    /// `frame_num`s of references that the next reference frame marks
    /// unused-for-reference (via MMCO) in its slice header, so the decoder's
    /// DPB matches the encoder's.
    pending_unmark_frame_nums: Vec<u32>,
    /// Profile IDC (cached for parameter-set recreation).
    profile_idc: u32,
    /// Whether CABAC entropy coding is preferred (from quality-level query).
    preferred_entropy_cabac: bool,
    /// The highest level the device encodes, `VkVideoEncodeH264CapabilitiesKHR::maxLevelIdc`.
    max_level_idc: ash::vk::native::StdVideoH264LevelIdc,
}

/// The picture type of the frame `plan`.
fn picture_type(plan: &FramePlan) -> ash::vk::native::StdVideoH264PictureType {
    if plan.is_idr() {
        ash::vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_IDR
    } else if plan.is_b() {
        ash::vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_B
    } else {
        ash::vk::native::StdVideoH264PictureType_STD_VIDEO_H264_PICTURE_TYPE_P
    }
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
            self.references.clear();
            // An IDR resets the decoder's whole DPB, so any deferred
            // unused-reference markings are moot.
            self.pending_unmark_frame_nums.clear();
        }

        // A reference frame makes the B-frame references of its level and
        // below of no more use. The decoder drops what it is told to, so
        // they are marked in this frame's slice header (see `record`).
        if plan.is_reference() && !plan.is_idr() {
            let level = plan.pyramid_level();
            for dropped in self.references.drop_obsolete(level) {
                self.pending_unmark_frame_nums.push(dropped.codec.frame_num);
            }
            // A frame with marking operations has no sliding window in the
            // decoder: it drops only what it is told to. So the reference
            // that the window drops after this frame is marked as well. The
            // frame still predicts from it: marking comes after decoding.
            if !self.pending_unmark_frame_nums.is_empty()
                && let Some(victim) = self.references.window_victim(level)
            {
                self.pending_unmark_frame_nums.push(victim.codec.frame_num);
            }
        }

        let header = if plan.is_idr() {
            Some(self.build_header(common)?)
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
        // `frame_num` carried by this frame, before the counter advances.
        let frame_num = self.frame_num_syntax;
        self.frame_num_syntax = (frame_num + 1) % 256;
        // The current frame becomes a reference for subsequent frames.
        self.references.insert(Reference {
            dpb_slot: common.current_dpb_slot,
            poc: plan.pic_order_cnt(),
            display_order: plan.display_order,
            encode_index: plan.encode_index,
            level: plan.pyramid_level(),
            codec: H264Reference {
                frame_num,
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
        // H.264 maintains the DPB with an implicit sliding window, which
        // would keep the lost pictures and desync the decoder's reference
        // list from ours. So each dropped reference is queued to be marked
        // unused-for-reference (MMCO) in the next reference frame (see
        // `record`).
        for lost in self.references.invalidate(first_lost_display_order) {
            self.pending_unmark_frame_nums.push(lost.codec.frame_num);
        }
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
        // H.264 regenerates SPS/PPS on every IDR, so there is nothing cached.
    }
}
