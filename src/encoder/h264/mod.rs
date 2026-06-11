//! H.264 encoder implementation using Vulkan Video.
//!
//! This module implements H.264 video encoding using Vulkan Video extensions.

mod api;
mod encode;
mod init;
mod session_params;

use ash::vk;
use tracing::debug;

use crate::encoder::resources::{upload_image_to_input, UploadParams};
use crate::error::Result;

use crate::encoder::dpb::DecodedPictureBuffer;
use crate::encoder::gop::GopStructure;
use crate::encoder::EncodeConfig;
use crate::vulkan::VideoContext;

/// H.264 macroblock size in pixels.
pub const MB_SIZE: u32 = 16;

/// Number of in-flight encode slots. Depth=2 lets frame N+1 begin encoding
/// while frame N is still on the encode hardware, so the per-frame budget
/// becomes 2 × frame_interval (16.6ms at 120fps) instead of 1 ×.
pub(crate) const ENCODE_PIPELINE_DEPTH: usize = 2;

/// One slot's worth of per-frame encode resources. Mirrors the H.265 design
/// (see encoder::h265::EncodeSlot).
pub(crate) struct EncodeSlot {
    pub input_image: vk::Image,
    pub input_image_memory: vk::DeviceMemory,
    pub input_image_view: vk::ImageView,
    pub input_image_layout: vk::ImageLayout,

    pub bitstream_buffer: vk::Buffer,
    pub bitstream_buffer_memory: vk::DeviceMemory,
    pub bitstream_buffer_ptr: *mut u8,

    pub encode_command_buffer: vk::CommandBuffer,
    pub encode_fence: vk::Fence,
    pub query_pool: vk::QueryPool,

    pub in_flight: bool,
    pub pending_metadata: Option<SlotPacketMetadata>,
}

/// Metadata stashed at submit-time, returned with the bitstream when this
/// slot's encode is drained on a later encode() call.
pub(crate) struct SlotPacketMetadata {
    pub frame_type: crate::encoder::FrameType,
    pub is_key_frame: bool,
    pub pts: u64,
    pub dts: u64,
    /// SPS/PPS header to prepend (Some only on first IDR).
    pub header: Option<Vec<u8>>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ReferenceInfo {
    pub dpb_slot: u8,
    pub frame_num: u32,
    pub poc: i32,
}

/// H.264 encoder.
pub struct H264Encoder {
    context: VideoContext,
    config: EncodeConfig,
    dpb: DecodedPictureBuffer,
    gop: GopStructure,

    /// Aligned width (macroblock + granularity aligned).
    aligned_width: u32,
    /// Aligned height (macroblock + granularity aligned).
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
    frame_num_syntax: u32,
    idr_pic_id: u32,

    /// Per-frame encode slots. See encoder::h265 for invariants.
    pub(crate) slots: Vec<EncodeSlot>,
    pub(crate) current_slot: usize,

    /// DPB images (up to MAX_DPB_SLOTS for B-frame and long-term reference support).
    dpb_images: Vec<vk::Image>,
    dpb_image_memories: Vec<vk::DeviceMemory>,
    dpb_image_views: Vec<vk::ImageView>,
    /// Number of DPB slots allocated.
    dpb_slot_count: usize,
    /// Whether the DPB uses a single layered image (true) or separate images (false).
    use_layered_dpb: bool,
    /// Tracks which DPB slots have been activated (used at least once).
    dpb_slot_active: Vec<bool>,

    // Command pool (encode command buffers per slot allocated from this pool).
    command_pool: vk::CommandPool,
    upload_command_pool: vk::CommandPool,
    upload_command_buffer: vk::CommandBuffer,
    upload_fence: vk::Fence,

    // SPS/PPS written flag.
    sps_written: bool,

    // Reference picture tracking.
    /// Whether we have a backward reference (for B-frames, L1).
    has_backward_reference: bool,
    /// Frame number of the L1 (backward) reference picture (for B-frames).
    backward_reference_frame_num: u32,
    /// POC of the L1 (backward) reference picture.
    backward_reference_poc: i32,
    /// DPB slot for L1 (backward) reference.
    backward_reference_dpb_slot: u8,
    /// Current DPB slot to use for setup (the reconstructed picture).
    current_dpb_slot: u8,
    /// Active L0 reference pictures (for P-frames). Ordered from most recent to oldest.
    l0_references: Vec<ReferenceInfo>,
    /// Number of active reference frames (as configured/negotiated).
    active_reference_count: u32,
    /// H.264 profile IDC (cached from initialization for session parameter recreation).
    profile_idc: u32,
    /// Whether CABAC entropy coding is preferred (cached from quality level query).
    preferred_entropy_cabac: bool,
}

impl H264Encoder {
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

        slot.input_image_layout = vk::ImageLayout::VIDEO_ENCODE_SRC_KHR;

        Ok(())
    }
}

// SAFETY: The raw pointer bitstream_buffer_ptr is only used within the encoder's
// thread and is properly synchronized via Vulkan fences before access
unsafe impl Send for H264Encoder {}

impl Drop for H264Encoder {
    fn drop(&mut self) {
        unsafe {
            let device = self.context.device();
            let _ = device.queue_wait_idle(self.context.transfer_queue());
            if let Some(q) = self.context.video_encode_queue() {
                let _ = device.queue_wait_idle(q);
            }

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
