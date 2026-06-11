//! H.265/HEVC encoder implementation using Vulkan Video.
//!
//! This module implements H.265/HEVC video encoding using Vulkan Video extensions.

mod api;
mod encode;
mod init;
mod session_params;

use ash::vk;
use tracing::debug;

use crate::encoder::dpb::DecodedPictureBuffer;
use crate::encoder::gop::GopStructure;
use crate::encoder::resources::{upload_image_to_input, UploadParams};
use crate::encoder::EncodeConfig;
use crate::error::Result;
use crate::vulkan::VideoContext;

/// H.265 Coding Tree Block (CTB) size in pixels.
pub const CTB_SIZE: u32 = 32;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ReferenceInfo {
    pub dpb_slot: u8,
    pub poc: i32,
}

/// Number of in-flight encode slots. Depth=2 lets frame N+1 begin encoding
/// while frame N is still on the encode hardware, so the per-frame budget
/// becomes 2 × frame_interval (16.6ms at 120fps) instead of 1 ×.
pub(crate) const ENCODE_PIPELINE_DEPTH: usize = 2;

/// One slot's worth of per-frame encode resources. All fields here are
/// duplicated `ENCODE_PIPELINE_DEPTH` times so multiple frames can be
/// in-flight concurrently. See the comment on `slots` below for the rotation
/// invariants.
pub(crate) struct EncodeSlot {
    /// Image the converter writes into (and the encoder reads from) for
    /// this slot's frame.
    pub input_image: vk::Image,
    pub input_image_memory: vk::DeviceMemory,
    pub input_image_view: vk::ImageView,
    /// Tracked layout of `input_image` for safe transitions between frames.
    pub input_image_layout: vk::ImageLayout,

    /// Bitstream destination buffer for this slot's encode.
    pub bitstream_buffer: vk::Buffer,
    pub bitstream_buffer_memory: vk::DeviceMemory,
    /// Persistently-mapped pointer (avoids per-frame map/unmap).
    pub bitstream_buffer_ptr: *mut u8,

    /// Command buffer recorded fresh each time this slot is used.
    pub encode_command_buffer: vk::CommandBuffer,
    /// Signaled when the encode for this slot finishes on the GPU.
    pub encode_fence: vk::Fence,
    /// Single-query pool — one feedback query per encode submission.
    pub query_pool: vk::QueryPool,

    /// `true` after we've submitted to this slot but not yet drained it.
    /// Used to decide whether `input_image()` must wait before returning.
    pub in_flight: bool,

    /// Metadata captured at submission time. The drained bitstream is wrapped
    /// in an `EncodedPacket` using this metadata after the next encode() call
    /// targeting the same slot waits on its fence.
    pub pending_metadata: Option<SlotPacketMetadata>,
}

/// Frame metadata stashed alongside an in-flight encode submission. When the
/// submission is drained on a later `encode()` call, we reconstruct the
/// `EncodedPacket` using these fields plus the freshly-read bitstream.
pub(crate) struct SlotPacketMetadata {
    pub frame_type: crate::encoder::FrameType,
    pub is_key_frame: bool,
    pub pts: u64,
    pub dts: u64,
    /// VPS/SPS/PPS header bytes (Some only for IDR frames).
    pub header: Option<Vec<u8>>,
}

/// H.265 encoder.
pub struct H265Encoder {
    context: VideoContext,
    config: EncodeConfig,
    dpb: DecodedPictureBuffer,
    gop: GopStructure,

    /// Aligned width (to CTB size).
    aligned_width: u32,
    /// Aligned height (to CTB size).
    aligned_height: u32,

    // Video session.
    video_queue_fn: ash::khr::video_queue::Device,
    video_encode_fn: ash::khr::video_encode_queue::Device,
    session: vk::VideoSessionKHR,
    session_params: vk::VideoSessionParametersKHR,
    session_memory: Vec<vk::DeviceMemory>,

    // Frame counters.
    input_frame_num: u64,
    encode_frame_num: u64,

    /// Per-frame slots. Index `current_slot` is the slot we'll use for the
    /// *next* encode submission (and whose `input_image` `input_image()`
    /// returns). When `encode()` runs, it drains that slot's previous
    /// in-flight work (if any), records new commands into it, submits, then
    /// advances `current_slot` for the next frame. With depth=2 the encoder
    /// can keep two frames in flight at once.
    pub(crate) slots: Vec<EncodeSlot>,
    pub(crate) current_slot: usize,

    /// DPB images.
    dpb_images: Vec<vk::Image>,
    dpb_image_memories: Vec<vk::DeviceMemory>,
    dpb_image_views: Vec<vk::ImageView>,
    /// Number of DPB slots allocated.
    dpb_slot_count: usize,
    /// Whether the DPB uses a single layered image (true) or separate images (false).
    use_layered_dpb: bool,

    // Command pool (encode + upload command buffers allocated from these).
    command_pool: vk::CommandPool,
    upload_command_pool: vk::CommandPool,
    upload_command_buffer: vk::CommandBuffer,
    upload_fence: vk::Fence,

    // Parameter sets - cached header data (VPS/SPS/PPS)
    header_data: Option<Vec<u8>>,

    // Reference picture tracking.
    /// Whether we have a backward reference (for B-frames, L1).
    has_backward_reference: bool,
    /// POC of the L1 (backward) reference picture.
    backward_reference_poc: i32,
    /// DPB slot for L1 (backward) reference.
    backward_reference_dpb_slot: u8,
    /// Current DPB slot to use for setup (the reconstructed picture).
    current_dpb_slot: u8,
    /// Active L0 reference pictures (for P-frames).
    l0_references: Vec<ReferenceInfo>,
    /// Number of active reference frames.
    active_reference_count: u32,
    /// H.265 profile IDC (cached from initialization for session parameter recreation).
    profile_idc: u32,

    // DPB slot activation tracking.
    /// Tracks which DPB slots have been activated (used at least once).
    dpb_slot_active: Vec<bool>,
}

impl H265Encoder {
    /// Upload input frame from a GPU image.
    ///
    /// This copies from a source NV12 image directly to the encoder's input image,
    /// avoiding any CPU-side data copies. The source image must be in NV12 format
    /// with the same dimensions as the encoder configuration. The source image
    /// should be in GENERAL layout.
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

        // Update tracked layout.
        slot.input_image_layout = vk::ImageLayout::VIDEO_ENCODE_SRC_KHR;

        Ok(())
    }
}

// SAFETY: The raw pointer bitstream_buffer_ptr is only used within the encoder's
// thread and is properly synchronized via Vulkan fences before access
unsafe impl Send for H265Encoder {}

impl Drop for H265Encoder {
    fn drop(&mut self) {
        unsafe {
            let device = self.context.device();
            // Wait on the queues used by the encoder rather than stalling
            // the entire device.
            let _ = device.queue_wait_idle(self.context.transfer_queue());
            if let Some(q) = self.context.video_encode_queue() {
                let _ = device.queue_wait_idle(q);
            }

            // Destroy per-slot resources first (each slot has its own image,
            // bitstream buffer, fence, query pool, and command buffer that
            // was allocated from `command_pool`). The command buffers are
            // freed implicitly when the pool is destroyed below.
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

            // Shared resources.
            device.destroy_fence(self.upload_fence, None);
            device.destroy_command_pool(self.command_pool, None);
            if self.upload_command_pool != self.command_pool {
                device.destroy_command_pool(self.upload_command_pool, None);
            }

            for view in &self.dpb_image_views {
                device.destroy_image_view(*view, None);
            }
            for image in &self.dpb_images {
                device.destroy_image(*image, None);
            }
            for memory in &self.dpb_image_memories {
                device.free_memory(*memory, None);
            }

            if self.session_params != vk::VideoSessionParametersKHR::null() {
                (self
                    .video_queue_fn
                    .fp()
                    .destroy_video_session_parameters_khr)(
                    device.handle(),
                    self.session_params,
                    std::ptr::null(),
                );
            }
            (self.video_queue_fn.fp().destroy_video_session_khr)(
                device.handle(),
                self.session,
                std::ptr::null(),
            );

            for memory in &self.session_memory {
                device.free_memory(*memory, None);
            }
        }
    }
}
