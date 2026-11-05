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
mod session_params;

use crate::encoder::ColorDescription;
use crate::encoder::codec::{EncoderCommon, FramePlan, PictureSetup, VideoCodec};
use crate::encoder::pipeline::EncodeFuture;
use crate::error::Result;
use ash::vk;

/// Minimum bitstream buffer size.
const MIN_BITSTREAM_BUFFER_SIZE: usize = 2 * 1024 * 1024;

/// AV1 superblock size in pixels (64x64, matching use_128x128_superblock=0).
pub const SUPERBLOCK_SIZE: u32 = 64;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ReferenceInfo {
    pub dpb_slot: u8,
    pub order_hint: u32,
    pub frame_type: u32,
    /// Display order (`pts`) of this reference, for reference frame invalidation.
    pub display_order: u64,
}

/// AV1-specific encoder state (multi-reference prediction).
pub(crate) struct Av1 {
    frame_num: u32,
    order_hint: u32,
    /// Cached sequence-header OBU (invalidated when session params change).
    header_data: Option<Vec<u8>>,
    /// References held in the DPB, most-recent first. Each frame predicts
    /// from the first `active_reference_count`; the rest are kept for
    /// reference frame invalidation to fall back on.
    references: Vec<ReferenceInfo>,
    /// How many references stay in the DPB.
    reference_window: u32,
    /// How many of the newest references one frame predicts from.
    active_reference_count: u32,
    /// The reference names single-reference prediction may use (bit 0 is
    /// LAST), from `VkVideoEncodeAV1CapabilitiesKHR`.
    single_reference_name_mask: u32,
}

impl Av1 {
    /// The references this frame predicts from: the newest held ones.
    fn used_references(&self) -> &[ReferenceInfo] {
        let n = self
            .references
            .len()
            .min(self.active_reference_count as usize);
        &self.references[..n]
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
}

impl VideoCodec for Av1 {
    fn begin_picture(
        &mut self,
        common: &mut EncoderCommon,
        plan: &FramePlan,
    ) -> Result<PictureSetup> {
        let is_key = plan.is_idr();
        if is_key {
            self.frame_num = 0;
            self.order_hint = 0;
            self.references.clear();
        }

        // Every temporal unit starts with a Temporal Delimiter OBU; key frames
        // also carry the sequence header (so decoders can initialize).
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
            frame_type: plan.frame_type(),
            header: Some(header),
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
        let is_key = plan.is_idr();
        // Order hint used while recording this frame (pre-increment).
        let encoded_order_hint = self.order_hint;
        self.frame_num += 1;
        self.order_hint = (self.order_hint + 1) & 0xFF;

        let ref_info = ReferenceInfo {
            dpb_slot: common.current_dpb_slot,
            order_hint: encoded_order_hint,
            display_order: plan.display_order,
            frame_type: if is_key {
                ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_KEY
            } else {
                ash::vk::native::StdVideoAV1FrameType_STD_VIDEO_AV1_FRAME_TYPE_INTER
            },
        };

        if is_key {
            self.references.clear();
        }
        self.references.insert(0, ref_info);
        // Keep a sliding window of the most recent references for prediction and
        // loss recovery.
        while self.references.len() > self.reference_window.max(1) as usize {
            self.references.pop();
        }

        // Cycle to the next available DPB slot.
        let used_slots: Vec<u8> = self.references.iter().map(|r| r.dpb_slot).collect();
        let mut next_slot = (common.current_dpb_slot + 1) % common.dpb_slot_count as u8;
        if used_slots.contains(&next_slot) {
            for i in 0..common.dpb_slot_count as u8 {
                if !used_slots.contains(&i) {
                    next_slot = i;
                    break;
                }
            }
        }
        common.current_dpb_slot = next_slot;
    }

    fn invalidate_references(
        &mut self,
        _common: &mut EncoderCommon,
        first_lost_display_order: u64,
    ) -> bool {
        // Drop every reference at or after the first lost frame: those are
        // transitively undecodable on the client. Older survivors are kept, so
        // the next P-frame re-anchors LAST_FRAME to the most recent of them (see
        // `calculate_reference_frame_mapping`). Returns false only when no
        // reference survives, in which case the caller forces a full key frame.
        self.references
            .retain(|r| r.display_order < first_lost_display_order);
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
    fn test_reference_info() {
        let ref_info = ReferenceInfo {
            dpb_slot: 2,
            order_hint: 42,
            frame_type: 0,
            display_order: 0,
        };
        assert_eq!(ref_info.dpb_slot, 2);
        assert_eq!(ref_info.order_hint, 42);
        let copied = ref_info;
        assert_eq!(copied.dpb_slot, ref_info.dpb_slot);
        assert_eq!(copied.order_hint, ref_info.order_hint);
    }

    #[test]
    fn test_order_hint_wrapping() {
        let mut order_hint: u32 = 254;
        for _ in 0..4 {
            order_hint = (order_hint + 1) & 0xFF;
        }
        assert_eq!(order_hint, 2);
    }

    #[test]
    fn test_reference_tracking() {
        let mut references: Vec<ReferenceInfo> = Vec::new();
        let max_refs = 4usize;
        for i in 0..6u8 {
            let ref_info = ReferenceInfo {
                dpb_slot: i % max_refs as u8,
                order_hint: i as u32,
                frame_type: 0,
                display_order: 0,
            };
            references.insert(0, ref_info);
            while references.len() > max_refs {
                references.pop();
            }
        }
        assert_eq!(references.len(), max_refs);
        assert_eq!(references[0].order_hint, 5);
        assert_eq!(references[max_refs - 1].order_hint, 2);
    }

    #[test]
    fn test_key_frame_clears_references() {
        let mut references: Vec<ReferenceInfo> = Vec::new();
        for i in 0..3u8 {
            references.insert(
                0,
                ReferenceInfo {
                    dpb_slot: i,
                    order_hint: i as u32,
                    frame_type: 0,
                    display_order: 0,
                },
            );
        }
        assert_eq!(references.len(), 3);
        references.clear();
        assert!(references.is_empty());
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 0,
                order_hint: 0,
                frame_type: 0,
                display_order: 0,
            },
        );
        assert_eq!(references.len(), 1);
        assert_eq!(references[0].dpb_slot, 0);
        assert_eq!(references[0].order_hint, 0);
    }

    #[test]
    fn test_dpb_slot_reuse() {
        let max_refs = 2usize;
        let dpb_slot_count = 3u8;
        let mut references: Vec<ReferenceInfo> = Vec::new();
        let mut current_dpb_slot: u8 = 0;

        let find_free_slot = |refs: &[ReferenceInfo], slot_count: u8| -> u8 {
            let used: Vec<u8> = refs.iter().map(|r| r.dpb_slot).collect();
            for i in 0..slot_count {
                if !used.contains(&i) {
                    return i;
                }
            }
            0
        };

        assert_eq!(current_dpb_slot, 0);
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 0,
                order_hint: 0,
                frame_type: 0,
                display_order: 0,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 1);

        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 1,
                order_hint: 1,
                frame_type: 0,
                display_order: 0,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 2);

        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 2,
                order_hint: 2,
                frame_type: 0,
                display_order: 0,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        assert_eq!(references.len(), 2);
        assert_eq!(references[0].dpb_slot, 2);
        assert_eq!(references[1].dpb_slot, 1);
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 0);

        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 0,
                order_hint: 3,
                frame_type: 0,
                display_order: 0,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 1);
    }

    #[test]
    fn test_single_reference_slot() {
        let max_refs = 1usize;
        let dpb_slot_count = 2u8;
        let mut references: Vec<ReferenceInfo> = Vec::new();

        let find_free_slot = |refs: &[ReferenceInfo], slot_count: u8| -> u8 {
            let used: Vec<u8> = refs.iter().map(|r| r.dpb_slot).collect();
            for i in 0..slot_count {
                if !used.contains(&i) {
                    return i;
                }
            }
            0
        };

        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 0,
                order_hint: 0,
                frame_type: 0,
                display_order: 0,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        let mut current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 1);

        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 1,
                order_hint: 1,
                frame_type: 0,
                display_order: 0,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        assert_eq!(references.len(), 1);
        assert_eq!(references[0].dpb_slot, 1);
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 0);

        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 0,
                order_hint: 2,
                frame_type: 0,
                display_order: 0,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 1);
    }
}
