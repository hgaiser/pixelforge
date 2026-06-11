//! AV1 encoder implementation using Vulkan Video.
//!
//! This module implements AV1 video encoding using Vulkan Video extensions.

mod api;
mod encode;
mod init;
mod session_params;

use ash::vk;
use tracing::debug;

use crate::encoder::resources::{upload_image_to_input, UploadParams};
use crate::error::Result;

use crate::encoder::gop::GopStructure;
use crate::encoder::EncodeConfig;
use crate::vulkan::VideoContext;

/// Minimum bitstream buffer size.
const MIN_BITSTREAM_BUFFER_SIZE: usize = 2 * 1024 * 1024;

/// AV1 superblock size in pixels (64x64, matching use_128x128_superblock=0 in the sequence header).
pub const SUPERBLOCK_SIZE: u32 = 64;

/// Number of in-flight encode slots. Depth=2 lets frame N+1 begin encoding
/// while frame N is still on the encode hardware.
pub(crate) const ENCODE_PIPELINE_DEPTH: usize = 2;

/// One slot's worth of per-frame encode resources. Mirrors encoder::h265::EncodeSlot.
pub(crate) struct EncodeSlot {
    pub input_image: vk::Image,
    pub input_image_memory: vk::DeviceMemory,
    pub input_image_view: vk::ImageView,
    pub input_image_layout: vk::ImageLayout,

    pub bitstream_buffer: vk::Buffer,
    pub bitstream_buffer_memory: vk::DeviceMemory,
    pub bitstream_buffer_size: usize,
    pub bitstream_buffer_ptr: *mut u8,

    pub encode_command_buffer: vk::CommandBuffer,
    pub encode_fence: vk::Fence,
    pub query_pool: vk::QueryPool,

    pub in_flight: bool,
    pub pending_metadata: Option<SlotPacketMetadata>,
}

/// Metadata stashed at submit-time, returned with the bitstream when this
/// slot's encode is drained.
pub(crate) struct SlotPacketMetadata {
    pub frame_type: crate::encoder::FrameType,
    pub is_key_frame: bool,
    pub pts: u64,
    pub dts: u64,
    /// AV1 sequence header OBU to prepend (Some only for IDR/key frames).
    pub header: Option<Vec<u8>>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ReferenceInfo {
    pub dpb_slot: u8,
    pub order_hint: u32,
    pub frame_type: u32,
}

/// AV1 encoder.
pub struct AV1Encoder {
    context: VideoContext,
    config: EncodeConfig,
    gop: GopStructure,

    // Video session.
    video_queue_fn: ash::khr::video_queue::Device,
    video_encode_fn: ash::khr::video_encode_queue::Device,
    session: vk::VideoSessionKHR,
    session_params: vk::VideoSessionParametersKHR,
    session_memory: Vec<vk::DeviceMemory>,

    // Frame counters.
    input_frame_num: u64,
    encode_frame_num: u64,
    frame_num: u32,
    order_hint: u32,

    /// Per-frame encode slots. See encoder::h265 for invariants.
    pub(crate) slots: Vec<EncodeSlot>,
    pub(crate) current_slot: usize,

    /// DPB images for reference frames.
    dpb_images: Vec<vk::Image>,
    dpb_image_memories: Vec<vk::DeviceMemory>,
    dpb_image_views: Vec<vk::ImageView>,
    /// Number of DPB slots allocated.
    dpb_slot_count: usize,
    /// Whether each DPB slot has been activated (written to at least once).
    dpb_slot_active: Vec<bool>,

    // Command pool (encode command buffers per slot allocated from this pool).
    command_pool: vk::CommandPool,
    upload_command_pool: vk::CommandPool,
    upload_command_buffer: vk::CommandBuffer,
    upload_fence: vk::Fence,

    // Optional semaphore to wait on before encoding (from color converter).

    // Cached AV1 sequence header OBU (retrieved from session parameters).
    header_data: Option<Vec<u8>>,

    // Reference picture tracking.
    /// Current DPB slot to use for setup (the reconstructed picture).
    current_dpb_slot: u8,
    /// Active reference pictures. Ordered from most recent to oldest.
    references: Vec<ReferenceInfo>,
}

impl AV1Encoder {
    /// Upload input frame from a GPU image.
    ///
    /// This copies from a source image directly to the encoder's input image,
    /// avoiding any CPU-side data copies. The source image must match the
    /// encoder's configured pixel format and dimensions, and should be in
    /// GENERAL layout.
    fn upload_from_image(&mut self, src_image: vk::Image) -> Result<()> {
        let slot = &mut self.slots[self.current_slot];
        if src_image == slot.input_image {
            debug!("Source image is the encoder's input image, skipping upload copy");
            return Ok(());
        }

        let params = UploadParams {
            upload_command_buffer: self.upload_command_buffer,
            upload_fence: self.upload_fence,
            src_image,
            dst_image: slot.input_image,
            width: self.config.dimensions.width,
            height: self.config.dimensions.height,
            pixel_format: self.config.pixel_format,
            input_image_layout: slot.input_image_layout,
            upload_queue: self.context.transfer_queue(),
        };

        upload_image_to_input(&self.context, &params)?;

        slot.input_image_layout = vk::ImageLayout::VIDEO_ENCODE_SRC_KHR;

        Ok(())
    }
}

// SAFETY: The raw pointer bitstream_buffer_ptr is only used within the encoder's
// thread and is properly synchronized via Vulkan fences before access
unsafe impl Send for AV1Encoder {}

impl Drop for AV1Encoder {
    fn drop(&mut self) {
        unsafe {
            let device = self.context.device();
            let _ = device.device_wait_idle();

            for slot in &mut self.slots {
                if !slot.bitstream_buffer_ptr.is_null() {
                    device.unmap_memory(slot.bitstream_buffer_memory);
                    slot.bitstream_buffer_ptr = std::ptr::null_mut();
                }
                device.destroy_query_pool(slot.query_pool, None);
                device.destroy_fence(slot.encode_fence, None);
                device.destroy_buffer(slot.bitstream_buffer, None);
                device.free_memory(slot.bitstream_buffer_memory, None);
                device.destroy_image_view(slot.input_image_view, None);
                device.destroy_image(slot.input_image, None);
                device.free_memory(slot.input_image_memory, None);
            }

            device.destroy_fence(self.upload_fence, None);
            device.destroy_command_pool(self.command_pool, None);
            if self.upload_command_pool != self.command_pool {
                device.destroy_command_pool(self.upload_command_pool, None);
            }

            for i in 0..self.dpb_images.len() {
                device.destroy_image_view(self.dpb_image_views[i], None);
                device.destroy_image(self.dpb_images[i], None);
                device.free_memory(self.dpb_image_memories[i], None);
            }

            if self.session_params != vk::VideoSessionParametersKHR::null() {
                self.video_queue_fn
                    .destroy_video_session_parameters(self.session_params, None);
            }
            self.video_queue_fn
                .destroy_video_session(self.session, None);
            for mem in &self.session_memory {
                device.free_memory(*mem, None);
            }
        }
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
        // Dimensions should be aligned up to superblock boundaries.
        let align = |v: u32| (v + SUPERBLOCK_SIZE - 1) & !(SUPERBLOCK_SIZE - 1);

        assert_eq!(align(1920), 1920); // Already aligned
        assert_eq!(align(1080), 1088); // 1080 rounds up to 1088
        assert_eq!(align(2560), 2560); // Already aligned
        assert_eq!(align(1440), 1472); // 1440 rounds up to 1472
        assert_eq!(align(1), 64); // Minimum is one superblock
    }

    #[test]
    fn test_reference_info() {
        let ref_info = ReferenceInfo {
            dpb_slot: 2,
            order_hint: 42,
            frame_type: 0,
        };

        assert_eq!(ref_info.dpb_slot, 2);
        assert_eq!(ref_info.order_hint, 42);

        // Should be Copy + Clone.
        let copied = ref_info;
        assert_eq!(copied.dpb_slot, ref_info.dpb_slot);
        assert_eq!(copied.order_hint, ref_info.order_hint);
    }

    #[test]
    fn test_order_hint_wrapping() {
        // AV1 order hints are 8-bit, wrapping at 256.
        let mut order_hint: u32 = 254;
        for _ in 0..4 {
            order_hint = (order_hint + 1) & 0xFF;
        }
        // 254 -> 255 -> 0 -> 1 -> 2
        assert_eq!(order_hint, 2);
    }

    #[test]
    fn test_reference_tracking() {
        // Simulate building up a reference list like the encoder does.
        let mut references: Vec<ReferenceInfo> = Vec::new();
        let max_refs = 4usize;

        for i in 0..6u8 {
            let ref_info = ReferenceInfo {
                dpb_slot: i % max_refs as u8,
                order_hint: i as u32,
                frame_type: 0,
            };
            references.insert(0, ref_info);
            while references.len() > max_refs {
                references.pop();
            }
        }

        // Should have exactly max_refs entries.
        assert_eq!(references.len(), max_refs);
        // Most recent should be first.
        assert_eq!(references[0].order_hint, 5);
        assert_eq!(references[max_refs - 1].order_hint, 2);
    }

    #[test]
    fn test_key_frame_clears_references() {
        // Simulate the encoder's key frame behavior: references.clear() on IDR.
        let mut references: Vec<ReferenceInfo> = Vec::new();

        // Build up some references (like a sequence of P-frames).
        for i in 0..3u8 {
            references.insert(
                0,
                ReferenceInfo {
                    dpb_slot: i,
                    order_hint: i as u32,
                    frame_type: 0,
                },
            );
        }
        assert_eq!(references.len(), 3);

        // Key frame resets everything.
        references.clear();
        assert!(references.is_empty());

        // First frame after key should start fresh at slot 0.
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 0,
                order_hint: 0,
                frame_type: 0,
            },
        );
        assert_eq!(references.len(), 1);
        assert_eq!(references[0].dpb_slot, 0);
        assert_eq!(references[0].order_hint, 0);
    }

    #[test]
    fn test_dpb_slot_reuse() {
        // Simulate the encoder's DPB slot allocation: after encoding a reference
        // frame, find the first slot not used by any active reference.
        let max_refs = 2usize;
        let dpb_slot_count = 3u8; // active_refs + 1
        let mut references: Vec<ReferenceInfo> = Vec::new();
        let mut current_dpb_slot: u8 = 0;

        // Helper: find next free slot (mirrors api.rs logic).
        let find_free_slot = |refs: &[ReferenceInfo], slot_count: u8| -> u8 {
            let used: Vec<u8> = refs.iter().map(|r| r.dpb_slot).collect();
            for i in 0..slot_count {
                if !used.contains(&i) {
                    return i;
                }
            }
            0 // fallback (shouldn't happen with correct slot_count)
        };

        // Frame 0 (IDR): uses slot 0.
        assert_eq!(current_dpb_slot, 0);
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 0,
                order_hint: 0,
                frame_type: 0,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 1); // slot 0 is used, next free is 1

        // Frame 1 (P): uses slot 1.
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 1,
                order_hint: 1,
                frame_type: 0,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 2); // slots 0,1 used, next free is 2

        // Frame 2 (P): uses slot 2. Now all 3 slots have been touched,
        // but max_refs=2 means the oldest reference (slot 0) gets evicted.
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 2,
                order_hint: 2,
                frame_type: 0,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        // references = [{slot:2, hint:2}, {slot:1, hint:1}] - slot 0 evicted
        assert_eq!(references.len(), 2);
        assert_eq!(references[0].dpb_slot, 2);
        assert_eq!(references[1].dpb_slot, 1);
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 0); // slot 0 is now free again (reuse!)

        // Frame 3 (P): uses recycled slot 0.
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 0,
                order_hint: 3,
                frame_type: 0,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        // references = [{slot:0, hint:3}, {slot:2, hint:2}] - slot 1 evicted
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 1); // slot 1 recycled
    }

    #[test]
    fn test_idr_p_p_idr_cycle() {
        // Full GOP cycle: IDR -> P -> P -> IDR, verifying DPB slot allocation
        // and reference list state at each step.
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

        // IDR frame: clears refs, writes to slot 0.
        references.clear();
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: current_dpb_slot,
                order_hint: 0,
                frame_type: 0,
            },
        );
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);

        assert_eq!(references.len(), 1);
        assert_eq!(references[0].order_hint, 0);
        assert_eq!(current_dpb_slot, 1);

        // P frame 1: writes to slot 1.
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: current_dpb_slot,
                order_hint: 1,
                frame_type: 0,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);

        assert_eq!(references.len(), 2);
        assert_eq!(references[0].order_hint, 1);
        assert_eq!(current_dpb_slot, 2);

        // P frame 2: writes to slot 2, evicts oldest ref (slot 0).
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: current_dpb_slot,
                order_hint: 2,
                frame_type: 0,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);

        assert_eq!(references.len(), 2);
        assert_eq!(references[0].order_hint, 2);
        assert_eq!(current_dpb_slot, 0); // slot 0 recycled

        // Second IDR: everything resets.
        references.clear();
        assert!(references.is_empty());
    }

    #[test]
    fn test_single_reference_slot() {
        // Edge case: only 1 active reference with 2 DPB slots (minimum viable).
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

        // Frame 0: slot 0.
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 0,
                order_hint: 0,
                frame_type: 0,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        let mut current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 1);

        // Frame 1: slot 1. Old ref (slot 0) evicted since max_refs=1.
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 1,
                order_hint: 1,
                frame_type: 0,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        assert_eq!(references.len(), 1);
        assert_eq!(references[0].dpb_slot, 1);
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 0); // ping-pong between 0 and 1

        // Frame 2: slot 0 again.
        references.insert(
            0,
            ReferenceInfo {
                dpb_slot: 0,
                order_hint: 2,
                frame_type: 0,
            },
        );
        while references.len() > max_refs {
            references.pop();
        }
        current_dpb_slot = find_free_slot(&references, dpb_slot_count);
        assert_eq!(current_dpb_slot, 1); // ping-pong back
    }
}
