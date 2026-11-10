//! AV1 codec: the differences from the generic encoder.
//!
//! Shared machinery lives in `crate::encoder::codec`; this folder holds only
//! AV1's reference tracking (`Av1`), its per-frame StdVideo* graph (`record`),
//! and its sequence-header generation (`session_params`).
//!
//! Inter frames predict from up to `Av1::active_reference_count` recent
//! references, mapped onto the forward AV1 reference names the device allows.
//! A window of `Av1::reference_window` references stays in the DPB (rather than
//! just the ones predicted from), which is what lets reference frame
//! invalidation recover from loss by re-anchoring to an older surviving frame
//! instead of forcing a key frame.

mod init;
mod record;
mod references;
mod session_params;

use crate::encoder::ColorDescription;
use crate::encoder::codec::{EncoderCommon, FramePlan, PictureSetup, VideoCodec};
use crate::encoder::references::{Reference, ReferenceSet};
use crate::error::{PixelForgeError, Result};
use ash::vk;
use references::{FrameReferences, bidirectional_references, show_existing_frame};

/// Minimum bitstream buffer size.
const MIN_BITSTREAM_BUFFER_SIZE: usize = 2 * 1024 * 1024;

/// AV1 superblock size in pixels (64x64, matching use_128x128_superblock=0).
pub const SUPERBLOCK_SIZE: u32 = 64;

/// What AV1 needs to know about a reference besides what every codec does.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Av1Reference {
    pub frame_type: u32,
}

pub(crate) type ReferenceInfo = Reference<Av1Reference>;

/// AV1-specific encoder state (multi-reference prediction).
pub(crate) struct Av1 {
    frame_num: u32,
    /// Cached sequence-header OBU (invalidated when session params change).
    header_data: Option<Vec<u8>>,
    /// References held in the DPB. Each frame predicts from the newest
    /// `active_reference_count`; the rest are kept for reference frame
    /// invalidation to fall back on.
    references: ReferenceSet<Av1Reference>,
    /// How many of the newest references one frame predicts from.
    active_reference_count: u32,
    /// The reference names single-reference prediction may use (bit 0 is
    /// LAST), from `VkVideoEncodeAV1CapabilitiesKHR`.
    single_reference_name_mask: u32,
    /// The reference names prediction from both directions may use.
    bidirectional_reference_name_mask: u32,
    /// How many references shown before it a frame that predicts from both
    /// directions uses. It uses one shown after it.
    bidirectional_past_count: u32,
}

/// The order hint of the frame with POC `poc`: its place in display order
/// since the key frame, in the 8 bits the sequence header gives it.
fn order_hint(poc: i32) -> u32 {
    (poc / 2) as u32 & 0xFF
}

impl Av1 {
    /// The references this frame predicts from: the newest held ones.
    fn used_references(&self) -> &[ReferenceInfo] {
        self.references.newest(self.active_reference_count as usize)
    }

    /// The DPB slot each of the 7 AV1 reference names predicts from, or -1.
    ///
    /// The used references take the names the device allows, most recent
    /// first. A name left at -1 is not predicted from.
    fn reference_names(&self) -> [i32; 7] {
        let mut names = [-1i32; 7];
        let allowed = (0..7).filter(|name| self.single_reference_name_mask & (1 << name) != 0);
        for (name, r) in allowed.zip(self.used_references()) {
            names[name] = r.dpb_slot as i32;
        }
        names
    }

    /// The references `plan` predicts from, under their names, and the
    /// distinct ones among them. None for a key frame.
    ///
    /// A frame that predicts from one direction uses the newest references.
    /// A B-frame uses the nearest ones shown before it and the nearest one
    /// shown after it.
    fn frame_references(&self, plan: &FramePlan) -> Result<(FrameReferences, Vec<ReferenceInfo>)> {
        if plan.is_idr() || self.references.is_empty() {
            let none = FrameReferences {
                names: [-1; 7],
                fill: [0; 2],
            };
            return Ok((none, Vec::new()));
        }
        if !plan.is_b() {
            let newest = self.references.as_slice()[0].dpb_slot as i8;
            let references = FrameReferences {
                names: self.reference_names(),
                fill: [newest; 2],
            };
            return Ok((references, self.used_references().to_vec()));
        }

        let (past, future) = self.references.around(plan.pic_order_cnt());
        let slots = |list: &[ReferenceInfo]| list.iter().map(|r| r.dpb_slot).collect::<Vec<u8>>();
        let references = bidirectional_references(
            &slots(&past),
            &slots(&future),
            self.bidirectional_reference_name_mask,
            self.bidirectional_past_count as usize,
        )
        .ok_or_else(|| {
            PixelForgeError::InvalidInput(format!(
                "AV1 B-frame {} has {} references before it and {} after it",
                plan.display_order,
                past.len(),
                future.len()
            ))
        })?;
        let used = self
            .references
            .as_slice()
            .iter()
            .filter(|r| references.names.contains(&i32::from(r.dpb_slot)))
            .copied()
            .collect();
        Ok((references, used))
    }
}

impl VideoCodec for Av1 {
    const HIDDEN_FRAMES: bool = true;

    fn show_packet(&self, _plan: &FramePlan, dpb_slot: u8) -> Vec<u8> {
        show_existing_frame(dpb_slot)
    }

    fn begin_picture(
        &mut self,
        common: &mut EncoderCommon,
        plan: &FramePlan,
    ) -> Result<PictureSetup> {
        let is_key = plan.is_idr();
        if is_key {
            self.frame_num = 0;
            self.references.clear();
        } else if plan.is_reference() {
            // A reference frame makes the B-frame references of its level
            // and below of no more use. They go out of the list, and their
            // frame buffers are taken by later frames.
            self.references.drop_obsolete(plan.pyramid_level());
        }

        // A hidden frame's bitstream goes in front of the next packet, which
        // has the temporal delimiter.
        if plan.hidden {
            return Ok(PictureSetup { header: None });
        }

        // Every temporal unit starts with a temporal delimiter OBU.
        let mut header = vec![0x12, 0x00];
        if is_key {
            if self.header_data.is_none() {
                self.header_data = Some(self.build_header(common)?);
            }
            if let Some(seq) = &self.header_data {
                header.extend_from_slice(seq);
            }
        }

        Ok(PictureSetup {
            header: Some(header),
        })
    }

    fn record_picture(&mut self, common: &mut EncoderCommon, plan: &FramePlan) -> Result<()> {
        self.record(common, plan)
    }

    fn end_picture(&mut self, common: &mut EncoderCommon, plan: &FramePlan) {
        self.frame_num += 1;
        // A frame that is not a reference goes into no frame buffer, and the
        // next frame takes its DPB slot.
        if !plan.is_reference() {
            return;
        }

        let frame_type = if plan.is_idr() {
            ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_KEY
        } else {
            ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_INTER
        };
        self.references.insert(Reference {
            dpb_slot: common.current_dpb_slot,
            poc: plan.pic_order_cnt(),
            display_order: plan.display_order,
            encode_index: plan.encode_index,
            level: plan.pyramid_level(),
            codec: Av1Reference { frame_type },
        });

        // Round-robin over the frame buffers, skipping the ones in use.
        let next = (common.current_dpb_slot + 1) % common.dpb_slot_count as u8;
        common.current_dpb_slot = self.references.free_slot(common.dpb_slot_count, Some(next));
    }

    fn invalidate_references(
        &mut self,
        _common: &mut EncoderCommon,
        first_lost_display_order: u64,
    ) -> bool {
        // Every frame buffer is declared explicitly per frame from
        // `references` (ref_frame_idx + ref_order_hint), so dropping entries
        // here is enough for the decoder to follow.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_superblock_size() {
        assert_eq!(SUPERBLOCK_SIZE, 64);
    }

    #[test]
    fn test_superblock_alignment() {
        let align = |v: u32| (v + SUPERBLOCK_SIZE - 1) & !(SUPERBLOCK_SIZE - 1);
        assert_eq!(align(1920), 1920);
        assert_eq!(align(1080), 1088);
        assert_eq!(align(2560), 2560);
        assert_eq!(align(1440), 1472);
        assert_eq!(align(1), 64);
    }

    #[test]
    fn order_hint_wraps_at_8_bits() {
        assert_eq!(order_hint(2 * 254), 254);
        assert_eq!(order_hint(2 * 258), 2);
    }
}
