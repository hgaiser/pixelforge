//! The codec-generic decoder.
//!
//! Everything that is identical across H.264, H.265 and AV1 lives here: the
//! shared per-decoder state ([`DecoderCommon`]), the Vulkan video session and
//! its DPB images ([`DecodeSession`]), coded-data staging, and the copy that
//! gets a picture out of the DPB where it cannot be handed over in place.
//!
//! A codec supplies only its differences: parsing its bitstream into pictures,
//! reconstructing the reference state the encoder implied, and building the
//! codec-specific `StdVideo*` graph for a picture. Reading a codec's folder
//! shows those differences; the scaffolding is here.
//!
//! There is deliberately no `VideoDecodeCodec` trait yet. The per-picture hook
//! shape is only knowable once a second codec exists to constrain it; H.265 is
//! the forcing function. Until then a codec owns a `DecoderCommon` directly.
//!
//! This mirrors the encoder's [`crate::encoder::codec`] split deliberately, so
//! the two directions read the same way.

use std::sync::Arc;

use ash::vk;

use crate::decoder::DecodedFrame;
use crate::decoder::frames::SlotPins;
use crate::decoder::pipeline::DecodePipeline;
use crate::encoder::{BitDepth, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::video::{
    VideoImageParams, allocate_session_memory, create_command_pool, create_dpb_images,
    create_video_image,
};
use crate::vulkan::VideoContext;

/// Annex B start code prefixed to each slice handed to the driver.
///
/// Three bytes, not four. The leading zero byte of a 4-byte start code is legal
/// Annex B, and RADV decodes either form, but Intel's ANV decodes almost nothing
/// when a slice offset points at that extra zero: it reports no error and
/// returns a picture with a handful of macroblocks in it. ffmpeg's Vulkan
/// decoder uses three bytes for the same reason.
pub(crate) const START_CODE: [u8; 3] = [0, 0, 1];

/// Query decode capabilities for a profile.
///
/// `caps` is supplied by the caller already chained with the codec-specific
/// capability struct, since only the codec knows which one applies.
pub(crate) fn query_decode_caps(
    context: &VideoContext,
    profile_info: &vk::VideoProfileInfoKHR,
    caps: &mut vk::VideoCapabilitiesKHR,
) -> Result<()> {
    let video_queue_instance =
        ash::khr::video_queue::Instance::load(context.entry(), context.instance());
    let result = unsafe {
        (video_queue_instance
            .fp()
            .get_physical_device_video_capabilities_khr)(
            context.physical_device(),
            profile_info,
            caps,
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::CodecNotSupported(format!(
            "decode profile not supported: {:?}",
            result
        )));
    }
    Ok(())
}

/// What a codec decided the session should look like.
///
/// The codec derives these from its own parameter sets and the device caps; the
/// generic layer turns them into Vulkan objects.
pub(crate) struct SessionPlan {
    /// Coded dimensions, already aligned to the device's picture granularity.
    pub coded_width: u32,
    pub coded_height: u32,
    pub picture_format: vk::Format,
    pub bit_depth: BitDepth,
    pub pixel_format: PixelFormat,
    /// DPB slots: references, the picture being decoded, and `spare_slots`.
    pub slot_count: usize,
    /// How many of `slot_count` are spare capacity for pictures pinned past the
    /// decode that produced them: those held for display-order reordering, and
    /// those handed to the caller. Zero means every emitted picture has to be
    /// copied out instead of pinned in place.
    pub spare_slots: usize,
    /// References the driver may use for a single picture, which is
    /// `slot_count` minus the current picture and the output reservation.
    pub max_active_references: u32,
    /// Driver decodes straight into the DPB image (`DPB_AND_OUTPUT_COINCIDE`).
    pub coincide: bool,
    /// A decoded picture may be handed to the caller in place, pinned in its
    /// DPB slot, rather than copied out. See [`DecodeSession::pinnable`].
    pub pinnable: bool,
    /// `dpb_usage` includes `SAMPLED`, so handed-out frames can be sampled.
    pub sampleable: bool,
    /// Picture images are created with `MUTABLE_FORMAT`, so a consumer can view
    /// their planes separately. See [`DecodeSession::plane_views`].
    pub plane_views: bool,
    /// Creation flags for picture images, from what the driver reported.
    pub image_flags: vk::ImageCreateFlags,
    /// Formats views of a picture image may use. Empty unless `plane_views`.
    pub view_formats: Vec<vk::Format>,
    /// One layered DPB image rather than one image per slot.
    pub use_layered_dpb: bool,
    pub dpb_usage: vk::ImageUsageFlags,
}

/// One DPB image and everything that lives and dies with it.
///
/// Held behind an `Arc` shared with every frame decoded into it, so a session
/// rebuild frees only the images nothing is still holding. The rest outlive the
/// session that made them and are destroyed by the last frame to let go, which
/// is what makes a `DecodedFrame`'s image valid for as long as the frame is,
/// full stop, with no exception for a stream that changes resolution.
pub(crate) struct DpbImage {
    context: VideoContext,
    image: vk::Image,
    memory: vk::DeviceMemory,
    /// One view per slot backed by this image: all of them for a layered DPB,
    /// exactly one when each slot has an image of its own.
    views: Vec<vk::ImageView>,
}

impl std::fmt::Debug for DpbImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `VideoContext` is not Debug, and the handles are what matter anyway.
        f.debug_struct("DpbImage")
            .field("image", &self.image)
            .finish_non_exhaustive()
    }
}

impl Drop for DpbImage {
    fn drop(&mut self) {
        // Runs on whichever thread dropped the last frame referencing this
        // image. Safe because the decoder waited for the device to idle before
        // giving the image up, and a caller must not drop a frame while their
        // own work on it is still in flight.
        unsafe {
            let device = self.context.device();
            for &view in &self.views {
                device.destroy_image_view(view, None);
            }
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// The Vulkan video session and the images it decodes into.
///
/// Recreated whenever the stream's parameter sets change in a way that matters
/// (a new resolution, most often).
pub(crate) struct DecodeSession {
    pub session: vk::VideoSessionKHR,
    pub session_memory: Vec<vk::DeviceMemory>,
    pub session_params: vk::VideoSessionParametersKHR,

    /// Coded dimensions the session was created for.
    pub coded_width: u32,
    pub coded_height: u32,
    pub picture_format: vk::Format,
    pub bit_depth: BitDepth,
    pub pixel_format: PixelFormat,

    /// DPB images. One image per slot, or a single layered image.
    /// One entry per slot, or a single shared entry when the DPB is layered.
    pub dpb: Vec<Arc<DpbImage>>,
    /// Whether each slot has been written at least once (governs the
    /// UNDEFINED-vs-DPB old layout in the pre-decode barrier).
    pub dpb_slot_active: Vec<bool>,
    pub use_layered_dpb: bool,

    /// True when the driver decodes into the DPB image directly; false when a
    /// distinct output image is required.
    pub coincide: bool,
    /// Spare DPB slots available for pinning pictures past their decode.
    pub spare_slots: usize,
    /// Whether a picture may be handed out in place instead of being copied.
    ///
    /// Needs two things at once. The driver must decode into the DPB image
    /// (`coincide`), or the picture lives in an output image the next decode
    /// overwrites. And the device must support unified image layouts, or the
    /// picture stays in `VIDEO_DECODE_DPB_KHR` and a consumer would have to
    /// transition it to read it, which is not safe while later pictures are
    /// still using it as a reference. Without both, every picture is copied
    /// into a private image on its way out.
    pub pinnable: bool,
    /// Whether decoded pictures can be sampled without being copied first.
    pub sampleable: bool,
    /// Slots reserved by frames the caller still holds, shared with those
    /// frames so they release on drop.
    ///
    /// Per session, not per decoder: a frame decoded under an earlier session
    /// releases into *that* session's set when it is finally dropped, where it
    /// can no longer free a slot the current session has reserved for someone
    /// else.
    pub slot_pins: Arc<SlotPins>,
    /// Whether a consumer may create per-plane views of a decoded picture.
    ///
    /// True when the picture images were created with `MUTABLE_FORMAT`, which
    /// the driver has to allow for this profile, format and usage. It is what
    /// lets a renderer read the luma and chroma planes as two ordinary
    /// single-plane textures instead of needing a `VkSamplerYcbcrConversion`,
    /// which some shader toolchains cannot express at all.
    pub plane_views: bool,
    /// Distinct decode output image, only when `!coincide`.
    pub output_image: Option<(vk::Image, vk::DeviceMemory, vk::ImageView)>,
}
impl DecodeSession {
    /// The image and array layer holding DPB slot `slot`.
    /// For layered DPBs all slots share `dpb[0]` and are distinguished by
    /// layer; for separate images each slot has its own image at layer 0.
    pub fn dpb_image_for_slot(&self, slot: u8) -> (vk::Image, u32) {
        let (entry, layer, _) = self.dpb_entry(slot);
        (entry.image, layer)
    }

    /// The image backing `slot`, the array layer within it, and its view.
    pub fn dpb_entry(&self, slot: u8) -> (&Arc<DpbImage>, u32, vk::ImageView) {
        if self.use_layered_dpb {
            let entry = &self.dpb[0];
            (entry, slot as u32, entry.views[slot as usize])
        } else {
            let entry = &self.dpb[slot as usize];
            (entry, 0, entry.views[0])
        }
    }
}

/// Per-decoder state shared by every codec.
///
/// Holds the Vulkan video session, the DPB images and the coded-data staging
/// buffer — none of which differ between codecs.
/// Codec-specific state (parameter sets, POC, reference marking) lives in the
/// codec instead.
pub(crate) struct DecoderCommon {
    pub context: VideoContext,
    pub video_queue_fn: ash::khr::video_queue::Device,
    pub video_decode_fn: ash::khr::video_decode_queue::Device,
    pub decode_queue: vk::Queue,
    pub decode_queue_family: u32,

    pub command_pool: vk::CommandPool,

    /// Readback resources. The video decode queue family is a dedicated engine
    /// that does not advertise `TRANSFER_BIT` (true on RADV, among others), so
    /// copies must be recorded and submitted on the transfer queue instead.
    pub transfer_queue: vk::Queue,
    pub transfer_pool: vk::CommandPool,

    /// In-flight decode submissions, their per-picture resources and the
    /// completion thread that resolves their futures.
    pub pipeline: DecodePipeline,

    /// Required alignment of bitstream buffer offsets/ranges.
    pub bitstream_offset_alignment: u64,
    pub bitstream_size_alignment: u64,

    /// The active session, created lazily from the stream's parameter sets.
    pub session: Option<DecodeSession>,

    /// Which set of picture images the decoder is currently handing out.
    ///
    /// Incremented every time a session is created. Copied onto each
    /// [`DecodedFrame`], where it is the only thing that lets a consumer tell
    /// one set of images from the next: a rebuilt session destroys its images,
    /// and drivers reuse `VkImage` handles freely, so a handle alone cannot
    /// identify anything.
    pub generation: u64,

    /// Receiving half of the frame channel, until a `DecodeSource` takes it.
    pub frames_rx: Option<futures_channel::mpsc::UnboundedReceiver<Result<DecodedFrame>>>,

    /// Queue family that reads decoded frames, when it is neither the decode
    /// nor the transfer family. Added to every picture image's sharing set so
    /// a consumer can use frames without a queue family ownership transfer.
    pub consumer_queue_family: Option<u32>,
}

// `bitstream_ptr` is a persistently mapped host-visible allocation owned by
// this struct; it is never aliased or shared across threads.
unsafe impl Send for DecoderCommon {}

impl DecoderCommon {
    pub fn new(context: VideoContext, consumer_queue_family: Option<u32>) -> Result<Self> {
        let decode_queue_family = context.video_decode_queue_family().ok_or_else(|| {
            PixelForgeError::NoSuitableDevice("Device has no video decode queue family".to_string())
        })?;
        let decode_queue = context.video_decode_queue().ok_or_else(|| {
            PixelForgeError::NoSuitableDevice("Device has no video decode queue".to_string())
        })?;

        let video_queue_fn =
            ash::khr::video_queue::Device::load(context.instance(), context.device());
        let video_decode_fn =
            ash::khr::video_decode_queue::Device::load(context.instance(), context.device());

        let command_pool = create_command_pool(&context, decode_queue_family, "decode")?;
        let transfer_pool =
            create_command_pool(&context, context.transfer_queue_family(), "decode transfer")?;
        let (pipeline, frames_rx) = DecodePipeline::new(&context, command_pool, transfer_pool)?;

        Ok(Self {
            transfer_queue: context.transfer_queue(),
            context,
            video_queue_fn,
            video_decode_fn,
            decode_queue,
            decode_queue_family,
            command_pool,
            transfer_pool,
            pipeline,
            bitstream_offset_alignment: 1,
            bitstream_size_alignment: 1,
            session: None,
            generation: 0,
            frames_rx: Some(frames_rx),
            consumer_queue_family,
        })
    }

    /// Every queue family that may touch a decoded picture.
    ///
    /// The decode family writes it, the transfer family copies from it, and a
    /// consumer named by
    /// [`DecodeConfig::with_consumer_queue_family`](crate::decoder::DecodeConfig::with_consumer_queue_family)
    /// reads it. `create_video_image` reduces this to `EXCLUSIVE` when the
    /// families turn out to be the same one.
    pub fn picture_sharing_families(&self) -> Vec<u32> {
        let mut families = vec![
            self.decode_queue_family,
            self.context.transfer_queue_family(),
        ];
        if let Some(consumer) = self.consumer_queue_family {
            families.push(consumer);
        }
        families
    }

    /// The layout DPB pictures live in.
    ///
    /// `GENERAL` when the device supports unified image layouts, which is what
    /// lets a handed-out picture be sampled or copied from while the decoder is
    /// still reading it as a reference: nothing ever has to transition it, so
    /// there is no layout for a consumer and the decode queue to disagree
    /// about. Otherwise the layout video decode requires.
    pub fn picture_layout(&self) -> vk::ImageLayout {
        if self.context.has_unified_image_layouts() {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::VIDEO_DECODE_DPB_KHR
        }
    }

    /// The layout a distinct decode output image lives in. See
    /// [`Self::picture_layout`].
    pub fn output_layout(&self) -> vk::ImageLayout {
        if self.context.has_unified_image_layouts() {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::VIDEO_DECODE_DST_KHR
        }
    }

    /// The active session, or an error if no parameter sets have been seen yet.
    pub fn session(&self) -> Result<&DecodeSession> {
        self.session
            .as_ref()
            .ok_or_else(|| PixelForgeError::InvalidInput("decode: no active session".to_string()))
    }

    /// Create the session, its DPB images and (when needed) a distinct output
    /// image, replacing any existing one.
    ///
    /// `make_params` builds the codec's session parameters once the session
    /// exists, since parameter sets are entirely codec-specific.
    pub fn create_session(
        &mut self,
        plan: &SessionPlan,
        profile_info: &vk::VideoProfileInfoKHR,
        std_header_version: &vk::ExtensionProperties,
        make_params: impl FnOnce(&Self, vk::VideoSessionKHR) -> Result<vk::VideoSessionParametersKHR>,
    ) -> Result<()> {
        self.destroy_session();

        let session_create_info = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(self.decode_queue_family)
            .flags(vk::VideoSessionCreateFlagsKHR::empty())
            .video_profile(profile_info)
            .picture_format(plan.picture_format)
            .max_coded_extent(vk::Extent2D {
                width: plan.coded_width,
                height: plan.coded_height,
            })
            .reference_picture_format(plan.picture_format)
            .max_dpb_slots(plan.slot_count as u32)
            .max_active_reference_pictures(plan.max_active_references)
            .std_header_version(std_header_version);

        let session = unsafe {
            self.video_queue_fn
                .create_video_session(&session_create_info, None)
        }
        .map_err(|e| PixelForgeError::VideoSessionCreation(format!("{:?}", e)))?;

        let session_memory = allocate_session_memory(&self.context, session, &self.video_queue_fn)?;

        // When the DPB doubles as the decode output it may also be copied from,
        // so the transfer queue needs access too, and a consumer reading frames
        // from a third family has to be named or its reads are undefined.
        let families = self.picture_sharing_families();

        let dpb_params = VideoImageParams {
            width: plan.coded_width,
            height: plan.coded_height,
            format: plan.picture_format,
            usage: plan.dpb_usage,
            sharing_families: &families,
            flags: plan.image_flags,
            view_formats: &plan.view_formats,
        };

        let (raw_images, raw_memories, raw_views) = create_dpb_images(
            &self.context,
            &dpb_params,
            profile_info,
            plan.slot_count,
            plan.use_layered_dpb,
        )?;

        // Group the raw handles by the image they belong to, so each can be
        // retained on its own: a layered DPB is one image carrying every slot's
        // view, otherwise it is one image and one view per slot.
        let dpb: Vec<Arc<DpbImage>> = if plan.use_layered_dpb {
            vec![Arc::new(DpbImage {
                context: self.context.clone(),
                image: raw_images[0],
                memory: raw_memories[0],
                views: raw_views,
            })]
        } else {
            raw_images
                .into_iter()
                .zip(raw_memories)
                .zip(raw_views)
                .map(|((image, memory), view)| {
                    Arc::new(DpbImage {
                        context: self.context.clone(),
                        image,
                        memory,
                        views: vec![view],
                    })
                })
                .collect()
        };

        let output_image = if plan.coincide {
            None
        } else {
            let output_params = VideoImageParams {
                usage: vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                ..dpb_params
            };
            Some(create_video_image(
                &self.context,
                &output_params,
                profile_info,
            )?)
        };

        let session_params = make_params(self, session)?;

        // A new set of images, so a new generation. Frames from the previous
        // one keep their old number and a consumer can tell them apart.
        self.generation = self.generation.wrapping_add(1);
        self.session = Some(DecodeSession {
            session,
            session_memory,
            session_params,
            coded_width: plan.coded_width,
            coded_height: plan.coded_height,
            picture_format: plan.picture_format,
            bit_depth: plan.bit_depth,
            pixel_format: plan.pixel_format,
            dpb,
            slot_pins: Arc::new(SlotPins::default()),
            dpb_slot_active: vec![false; plan.slot_count],
            use_layered_dpb: plan.use_layered_dpb,
            coincide: plan.coincide,
            spare_slots: plan.spare_slots,
            pinnable: plan.pinnable,
            sampleable: plan.sampleable,
            plane_views: plan.plane_views,
            output_image,
        });
        Ok(())
    }

    pub fn destroy_session(&mut self) {
        // Let every in-flight decode finish and its future resolve before the
        // session, DPB images and slots they refer to stop existing.
        self.pipeline.wait_all_free();
        let Some(session) = self.session.take() else {
            return;
        };
        // Frames decoded under this session may still be alive. Their images
        // are reference counted, so dropping the session's handles here frees
        // every image nothing is holding and leaves the rest to the last frame
        // that lets go. Nothing dangles, and nothing that is finished lingers.
        if session.slot_pins.any_pinned() {
            tracing::debug!(
                "decode session rebuilt while {} decoded frame(s) are still alive; \
                 their images outlive generation {} until those frames are dropped",
                session.slot_pins.count(),
                self.generation
            );
        }
        unsafe {
            self.context.device().device_wait_idle().ok();
            self.video_queue_fn
                .destroy_video_session_parameters(session.session_params, None);
            if let Some((image, memory, view)) = session.output_image {
                self.context.device().destroy_image_view(view, None);
                self.context.device().destroy_image(image, None);
                self.context.device().free_memory(memory, None);
            }
            self.video_queue_fn
                .destroy_video_session(session.session, None);
            for &memory in &session.session_memory {
                self.context.device().free_memory(memory, None);
            }
        }
    }

    /// Copy a picture's slices into the staging buffer, each prefixed with a
    /// start code, and report the buffer range plus each slice's offset.
    pub fn stage_slices(
        &mut self,
        slices: &[&[u8]],
        profile_info: &vk::VideoProfileInfoKHR,
    ) -> Result<(u64, Vec<u32>)> {
        let total: usize = slices.iter().map(|s| s.len() + START_CODE.len()).sum();
        let aligned_total =
            crate::video::align_up(total as u32, self.bitstream_size_alignment as u32) as usize;
        self.pipeline
            .ensure_bitstream_capacity(&self.context, aligned_total, profile_info)?;
        let dst = self.pipeline.current().bitstream_ptr;

        let mut offsets = Vec::with_capacity(slices.len());
        let mut cursor = 0usize;
        for slice in slices {
            offsets.push(cursor as u32);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    START_CODE.as_ptr(),
                    dst.add(cursor),
                    START_CODE.len(),
                );
                cursor += START_CODE.len();
                std::ptr::copy_nonoverlapping(slice.as_ptr(), dst.add(cursor), slice.len());
                cursor += slice.len();
            }
        }
        // Zero the alignment padding so the driver never reads stale bytes.
        if aligned_total > cursor {
            unsafe {
                std::ptr::write_bytes(dst.add(cursor), 0, aligned_total - cursor);
            }
        }
        Ok((aligned_total as u64, offsets))
    }

    /// Transition the DPB slot this decode writes into, and the output image
    /// (if distinct), into the layouts `vkCmdDecodeVideo` requires.
    pub fn record_barriers(&self, slot: u8) {
        let session = self.session.as_ref().expect("active session");
        let sync2 = self.context.sync2();
        let mut barriers: Vec<vk::ImageMemoryBarrier2> = Vec::new();

        let subresource = |layer: u32| vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: layer,
            layer_count: 1,
        };

        // The slot being written: UNDEFINED on first use, its picture layout
        // afterwards.
        let picture_layout = self.picture_layout();
        let (dst_image, dst_layer) = session.dpb_image_for_slot(slot);
        let old_layout = if session.dpb_slot_active[slot as usize] {
            picture_layout
        } else {
            vk::ImageLayout::UNDEFINED
        };
        barriers.push(
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                .src_access_mask(vk::AccessFlags2::VIDEO_DECODE_READ_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                .dst_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
                .old_layout(old_layout)
                .new_layout(picture_layout)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(dst_image)
                .subresource_range(subresource(dst_layer)),
        );

        // A distinct output image must be in decode-DST layout.
        if let Some((image, _, _)) = session.output_image {
            barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::empty())
                    .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                    .dst_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(self.output_layout())
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(subresource(0)),
            );
        }

        let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        unsafe { sync2.cmd_pipeline_barrier2(self.decode_command_buffer(), &dependency) };
    }

    /// The decode command buffer of the picture currently being recorded.
    pub fn decode_command_buffer(&self) -> vk::CommandBuffer {
        self.pipeline.current().decode_command_buffer
    }

    /// The coded-data staging buffer of the picture currently being recorded.
    pub fn bitstream_buffer(&self) -> vk::Buffer {
        self.pipeline.current().bitstream_buffer
    }

    /// Wait until the next picture's slot is free, then begin recording into it.
    ///
    /// Waiting here is what makes the staging buffer and command buffers safe to
    /// overwrite: the slot is busy until its previous submission has completed.
    pub fn begin_decode_commands(&self) -> Result<()> {
        self.pipeline.wait_current_free();
        let device = self.context.device();
        let command_buffer = self.decode_command_buffer();
        unsafe {
            device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
        }
        Ok(())
    }

    /// Submit the recorded decode without waiting for it.
    ///
    /// The submission is chained onto the decode timeline, so the GPU still runs
    /// decodes in decode order despite the CPU running ahead.
    pub fn submit_decode(&mut self) -> Result<()> {
        let device = self.context.device().clone();
        let queue = self.decode_queue;
        self.pipeline
            .submit_decode(&device, self.context.sync2(), queue)
    }

    /// Close off the current picture and hand `frames` to the completion thread.
    pub fn end_picture(&mut self, frames: Vec<DecodedFrame>) {
        self.pipeline.end_picture(frames);
    }

    /// Deliver the last frames and close the stream.
    pub fn finish_stream(&mut self, frames: Vec<DecodedFrame>) {
        self.pipeline.finish_stream(frames);
    }
}

impl Drop for DecoderCommon {
    fn drop(&mut self) {
        // `destroy_session` drains the pipeline first, so by the time the
        // staging buffers and fences below are freed nothing is in flight.
        self.destroy_session();
        unsafe {
            self.context.device().device_wait_idle().ok();
            let device = self.context.device().clone();
            self.pipeline.destroy(&device);
            self.context
                .device()
                .destroy_command_pool(self.command_pool, None);
            self.context
                .device()
                .destroy_command_pool(self.transfer_pool, None);
        }
    }
}
