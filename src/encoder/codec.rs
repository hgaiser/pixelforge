//! The codec-generic encoder.
//!
//! Everything that is identical across H.264, H.265 and AV1 lives here:
//! the shared per-encoder state ([`EncoderCommon`]), the generic driver
//! ([`CodecEncoder`]) that owns the public `submit`/`flush`/`reconfigure`
//! flow, the shared initialization scaffolding ([`build_encoder_common`]) and the
//! shared rate-control decision ([`RateControlPlan`]).
//!
//! A codec is anything that implements [`VideoCodec`]: a small state struct that
//! plugs its *differences* into the generic flow — building the codec-specific
//! StdVideo* graph for a frame, tracking its reference pictures, and emitting its
//! parameter sets. Reading a codec's folder shows only those differences; the
//! scaffolding is here.

use ash::vk;
use ash::vk::TaggedStructure;

use crate::converter::RgbFormat;
use crate::encoder::gop::{GopFrameType, GopPosition, GopStructure};
use crate::encoder::input::{FrameInput, FrameSync, SourceOwnership};
use crate::encoder::pipeline::{
    Delivery, EncodeFuture, EncodePipeline, PipelineConfig, SlotPacketMetadata,
};
use crate::encoder::profile::EncodeProfile;
use crate::encoder::resources::{
    EncoderTeardown, UploadParams, align_up, allocate_session_memory, create_command_resources,
    create_dpb_images, destroy_encoder_resources, end_timestamp, get_video_format, lcm,
    query_supported_video_formats, record_post_encode_dpb_barrier, reset_start_timestamp,
    upload_image_to_input,
};
use crate::encoder::rgb::RgbConversionCaps;
use crate::encoder::{ColorDescription, EncodeConfig, FrameType, RateControl, StreamColor};
use crate::error::{PixelForgeError, Result};
use crate::sync::TimelinePoint;
use crate::video::{TimelineChain, allocate_command_buffers};
use crate::vulkan::VideoContext;

/// The most DPB slots an encoder uses, for any codec.
pub(crate) const MAX_DPB_SLOTS: usize = 16;

/// Per-encoder state shared by every codec.
///
/// Holds the Vulkan video session, the DPB images, the async [`EncodePipeline`],
/// the GOP structure and the upload path — none of which differ between codecs.
/// Codec-specific state (reference lists, syntax counters, parameter caches)
/// lives in the [`VideoCodec`] implementor instead.
pub(crate) struct EncoderCommon {
    pub context: VideoContext,
    pub config: EncodeConfig,

    pub video_queue_fn: ash::khr::video_queue::Device,
    pub video_encode_fn: ash::khr::video_encode_queue::Device,
    pub session: vk::VideoSessionKHR,
    pub session_params: vk::VideoSessionParametersKHR,
    pub session_memory: Vec<vk::DeviceMemory>,

    /// Coded extent (display dimensions aligned to the codec block size and the
    /// device's picture-access granularity).
    pub aligned_width: u32,
    pub aligned_height: u32,

    /// Async slot rotation + bitstream readback.
    pub pipeline: EncodePipeline,
    /// Frame-type/POC schedule.
    pub gop: GopStructure,
    /// Monotonic display-order counter (presentation order).
    pub input_frame_num: u64,
    /// Monotonic encode-order counter (decode order / DTS; `0` => first frame).
    pub encode_frame_num: u64,

    pub dpb_images: Vec<vk::Image>,
    pub dpb_image_memories: Vec<vk::DeviceMemory>,
    pub dpb_image_views: Vec<vk::ImageView>,
    pub dpb_slot_count: usize,
    /// Whether each DPB slot has been written at least once (governs the
    /// UNDEFINED-vs-DPB old layout in the pre-encode barrier).
    pub dpb_slot_active: Vec<bool>,
    /// Single layered DPB image (true) vs one image per slot (false).
    pub use_layered_dpb: bool,
    /// DPB slot the current frame reconstructs into.
    pub current_dpb_slot: u8,

    pub command_pool: vk::CommandPool,
    pub upload_command_pool: vk::CommandPool,
    pub upload_fence: vk::Fence,
    /// One upload command buffer per pipeline slot. A slot is handed out again
    /// only once its encode is complete, and that encode waited for the
    /// slot's upload, so the buffer is free by then.
    pub upload_command_buffers: Vec<vk::CommandBuffer>,
    /// Signalled by each upload; the encode of the same frame waits on it.
    pub upload_timeline: TimelineChain,
    /// What the encode submission of the frame being encoded waits for.
    pub pending_waits: Vec<vk::SemaphoreSubmitInfo<'static>>,
    /// What the encode submission of the frame being encoded signals besides
    /// its timeline: a release the caller asked for, when the encode is what
    /// reads the source last.
    pub pending_signals: Vec<vk::SemaphoreSubmitInfo<'static>>,
    /// What the encode of the frame being encoded reads.
    pub source: SourceBinding,
    /// Where the bitstream of the frame being encoded goes.
    pub delivery: Delivery,
    /// The point the last encode submission signals. `None` until the frame
    /// being encoded was submitted.
    pub encode_point: Option<TimelinePoint>,
    /// The session's video profile, for images made after creation.
    #[cfg_attr(not(feature = "dmabuf"), allow(dead_code))]
    pub profile: EncodeProfile,
    /// The RGB format the encoder converts from, when it takes RGB input.
    pub rgb_input: Option<RgbFormat>,
}

impl EncoderCommon {
    /// Set up what a frame's encode reads, copying the source into the input
    /// image of `slot` first when it has to. Returns the frame, ready to wait
    /// for its encode, and the point where its source is free again if
    /// something before the encode is the last to read it.
    ///
    /// `sync.waits` go to the frame's first submission and
    /// `sync.release_signals` to the one that is last to read the source:
    /// the copy when there is one, the encode otherwise.
    pub fn prepare_source(
        &mut self,
        slot: usize,
        input: FrameInput,
        sync: FrameSync,
    ) -> Result<(HeldFrame, Option<TimelinePoint>)> {
        let (slot_image, slot_view, slot_layout) = {
            let slot = self.pipeline.slot(slot);
            (
                slot.input_image,
                slot.input_image_view,
                slot.input_image_layout,
            )
        };
        let slot_binding = SourceBinding {
            image: slot_image,
            view: slot_view,
            extent: None,
            ownership: None,
        };
        let mut released = None;
        let (source, waits, signals) = match input {
            FrameInput::InPlace => (slot_binding, sync.waits, sync.release_signals),
            FrameInput::Copy { image, ownership } => {
                let (semaphore, value) = self.upload_timeline.pending_signal();
                let done = vk::SemaphoreSubmitInfo::default()
                    .semaphore(semaphore)
                    .value(value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS);
                let mut signals = vec![done];
                signals.extend(sync.release_signals);
                let params = UploadParams {
                    upload_command_buffer: self.upload_command_buffers[slot],
                    src_image: image,
                    src_ownership: ownership,
                    dst_image: slot_image,
                    width: self.config.dimensions.width,
                    height: self.config.dimensions.height,
                    chroma_format: self.config.chroma_format,
                    input_image_layout: slot_layout,
                    rgb: self.rgb_input.is_some(),
                    upload_queue: self.context.transfer_queue(),
                    waits: &sync.waits,
                    signals: &signals,
                };
                upload_image_to_input(&self.context, &params)?;
                self.upload_timeline.commit();
                self.pipeline.slot_mut(slot).input_image_layout =
                    vk::ImageLayout::VIDEO_ENCODE_SRC_KHR;
                let point = TimelinePoint::new(semaphore, value);
                released = Some(point);
                (slot_binding, vec![point.wait_info()], Vec::new())
            }
            FrameInput::Direct {
                image,
                view,
                extent,
                ownership,
            } => (
                SourceBinding {
                    image,
                    view,
                    extent: Some(extent),
                    ownership: Some(ownership),
                },
                sync.waits,
                sync.release_signals,
            ),
        };
        Ok((
            HeldFrame {
                slot,
                source,
                waits,
                signals,
                submitted: std::time::Instant::now(),
            },
            released,
        ))
    }

    /// Make `frame` the frame being encoded: the codec hooks record into its
    /// slot and read its source from here on.
    pub fn begin_encode(&mut self, frame: HeldFrame, delivery: Delivery) {
        self.pipeline.begin_encode(frame.slot);
        self.source = frame.source;
        self.pending_waits = frame.waits;
        self.pending_signals = frame.signals;
        self.delivery = delivery;
        self.encode_point = None;
    }

    /// The view the current frame's encode reads.
    pub fn source_view(&self) -> vk::ImageView {
        self.source.view
    }

    /// The coded extent of the current frame's source picture.
    ///
    /// The aligned extent for the encoder's own input images, which are made
    /// that large and have their padding zeroed. A source read directly may
    /// be only the display size, as a DMA-BUF from a capture protocol has to
    /// be. Its coded extent is then the display size, which the spec allows:
    /// the stream headers already crop to it, and what the encoder reads
    /// beyond the image is the implementation's padding.
    pub fn source_coded_extent(&self) -> vk::Extent2D {
        let aligned = vk::Extent2D {
            width: self.aligned_width,
            height: self.aligned_height,
        };
        match self.source.extent {
            Some(extent) if extent.width < aligned.width || extent.height < aligned.height => {
                vk::Extent2D {
                    width: self.config.dimensions.width,
                    height: self.config.dimensions.height,
                }
            }
            _ => aligned,
        }
    }

    /// Record the barrier that brings a directly read source into
    /// `VIDEO_ENCODE_SRC_KHR` on the encode queue. Nothing to do for the
    /// encoder's own input images, which are already there.
    ///
    /// # Safety
    ///
    /// `command_buffer` must be recording, before the encode.
    pub unsafe fn record_source_acquire(&self, command_buffer: vk::CommandBuffer) {
        let (Some(ownership), Some(family)) = (
            self.source.ownership,
            self.context.video_encode_queue_family(),
        ) else {
            return;
        };
        let barrier = [ownership.acquire_barrier(
            self.source.image,
            family,
            vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
            vk::PipelineStageFlags2::VIDEO_ENCODE_KHR,
            vk::AccessFlags2::VIDEO_ENCODE_READ_KHR,
        )];
        unsafe {
            self.context.sync2().cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&barrier),
            );
        }
    }

    /// Record the barrier that hands a directly read source back after the
    /// encode, in `GENERAL`.
    ///
    /// # Safety
    ///
    /// `command_buffer` must be recording, after the encode.
    pub unsafe fn record_source_release(&self, command_buffer: vk::CommandBuffer) {
        let (Some(ownership), Some(family)) = (
            self.source.ownership,
            self.context.video_encode_queue_family(),
        ) else {
            return;
        };
        let barrier = [ownership.release_barrier(
            self.source.image,
            family,
            vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
            vk::PipelineStageFlags2::VIDEO_ENCODE_KHR,
        )];
        unsafe {
            self.context.sync2().cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&barrier),
            );
        }
    }

    /// Record the metadata for the packet the current slot will produce.
    pub fn set_pending_metadata(&mut self, metadata: SlotPacketMetadata) {
        self.pipeline.set_pending_metadata(metadata);
    }

    /// Submit the recorded command buffer for the current slot and mark its
    /// DPB slot active. [`Self::encode_point`] is what the submission signals.
    pub fn submit_frame(&mut self) -> Result<()> {
        let encode_queue = self.context.video_encode_queue().ok_or_else(|| {
            PixelForgeError::NoSuitableDevice("No video encode queue available".to_string())
        })?;
        let waits = std::mem::take(&mut self.pending_waits);
        let signals = std::mem::take(&mut self.pending_signals);
        let delivery = std::mem::replace(&mut self.delivery, Delivery::Packet);
        let point = self.pipeline.submit_current(
            self.context.device(),
            self.context.sync2(),
            encode_queue,
            &waits,
            &signals,
            delivery,
        )?;
        self.encode_point = Some(point);
        self.dpb_slot_active[self.current_dpb_slot as usize] = true;
        Ok(())
    }

    /// Record one frame's encode into the current slot's command buffer,
    /// from begin coding to the end of the command buffer, and submit it.
    ///
    /// The codec builds what differs: the reference slots, the encode info
    /// with its picture info, and its rate control structs.
    pub fn record_encode<R, L>(&mut self, cmd: EncodeCommands<'_, R, L>) -> Result<()>
    where
        R: for<'x> vk::Extends<vk::VideoBeginCodingInfoKHR<'x>>
            + for<'x> vk::Extends<vk::VideoCodingControlInfoKHR<'x>>
            + TaggedStructure<'static>
            + Copy,
        L: for<'x> vk::Extends<vk::VideoEncodeRateControlLayerInfoKHR<'x>>
            + TaggedStructure<'static>,
    {
        let slot = self.pipeline.current();
        let command_buffer = slot.encode_command_buffer;
        let query_pool = slot.query_pool;
        let timestamp_query_pool = slot.timestamp_query_pool;
        let rc = cmd.rate_control;

        let mut codec_layer = cmd.codec_layer;
        let rc_layers = [vk::VideoEncodeRateControlLayerInfoKHR::default()
            .average_bitrate(rc.average_bitrate as u64)
            .max_bitrate(rc.max_bitrate as u64)
            .frame_rate_numerator(self.config.frame_rate.numerator)
            .frame_rate_denominator(self.config.frame_rate.denominator)
            .push(&mut codec_layer)];
        let mut rc_info = vk::VideoEncodeRateControlInfoKHR::default().rate_control_mode(rc.mode);
        if !rc.is_disabled() {
            rc_info = rc_info
                .layers(&rc_layers)
                .virtual_buffer_size_in_ms(self.config.virtual_buffer_size_ms)
                .initial_virtual_buffer_size_in_ms(self.config.initial_virtual_buffer_size_ms);
        }

        reset_start_timestamp(self.device(), command_buffer, timestamp_query_pool);

        // Ash links pNext chains in place, so each command needs its own
        // extension structs. The first frame sets rate control with the
        // control command below, after RESET, rather than here.
        let mut begin_codec_rc = cmd.codec_rate_control;
        let mut begin_rc = rc_info;
        let mut begin_info = vk::VideoBeginCodingInfoKHR::default()
            .video_session(self.session)
            .video_session_parameters(self.session_params)
            .reference_slots(cmd.begin_slots)
            .push(&mut begin_codec_rc);
        if !cmd.first_frame {
            begin_info = begin_info.push(&mut begin_rc);
        }
        unsafe {
            self.video_queue_fn
                .cmd_begin_video_coding(command_buffer, &begin_info);
        }

        // RESET + RATE_CONTROL + QUALITY_LEVEL in one control command (matches
        // FFmpeg; required for AMD RADV).
        if cmd.reset {
            let mut codec_rc = cmd.codec_rate_control;
            let mut quality_level_info =
                vk::VideoEncodeQualityLevelInfoKHR::default().quality_level(0);
            let control_info = vk::VideoCodingControlInfoKHR::default()
                .flags(
                    vk::VideoCodingControlFlagsKHR::RESET
                        | vk::VideoCodingControlFlagsKHR::ENCODE_RATE_CONTROL
                        | vk::VideoCodingControlFlagsKHR::ENCODE_QUALITY_LEVEL,
                )
                .push(&mut rc_info)
                .push(&mut codec_rc)
                .push(&mut quality_level_info);
            unsafe {
                self.video_queue_fn
                    .cmd_control_video_coding(command_buffer, &control_info);
            }
        }

        unsafe {
            let device = self.device();
            device.cmd_begin_query(
                command_buffer,
                query_pool,
                0,
                vk::QueryControlFlags::empty(),
            );
            self.video_encode_fn
                .cmd_encode_video(command_buffer, cmd.encode_info);
            device.cmd_end_query(command_buffer, query_pool, 0);
            end_timestamp(device, command_buffer, timestamp_query_pool);

            if cmd.post_encode_barrier {
                record_post_encode_dpb_barrier(
                    device,
                    command_buffer,
                    &self.dpb_images,
                    self.use_layered_dpb,
                    self.current_dpb_slot,
                );
            }

            self.video_queue_fn
                .cmd_end_video_coding(command_buffer, &vk::VideoEndCodingInfoKHR::default());
            self.record_source_release(command_buffer);
            self.device()
                .end_command_buffer(command_buffer)
                .map_err(PixelForgeError::vk("vkEndCommandBuffer"))?;
        }

        self.submit_frame()
    }

    /// The `ash` device handle.
    pub fn device(&self) -> &ash::Device {
        self.context.device()
    }
}

/// What one frame's encode reads: the slot's own input image, or a source
/// read directly.
#[derive(Default)]
pub(crate) struct SourceBinding {
    pub image: vk::Image,
    pub view: vk::ImageView,
    /// The image's size, for a source read directly; `None` for the slot's
    /// input image, which has the aligned extent.
    pub extent: Option<vk::Extent2D>,
    /// The ownership transfers the encode does, for a source read directly.
    pub ownership: Option<SourceOwnership>,
}

/// A submitted frame that waits for its encode: its pipeline slot, which holds
/// the pixels unless the encode reads the source directly, and what the encode
/// submission needs.
pub(crate) struct HeldFrame {
    pub slot: usize,
    pub source: SourceBinding,
    /// What the encode submission waits for.
    pub waits: Vec<vk::SemaphoreSubmitInfo<'static>>,
    /// What the encode submission signals besides its timeline.
    pub signals: Vec<vk::SemaphoreSubmitInfo<'static>>,
    /// When the frame was submitted, for the packet's latency.
    pub submitted: std::time::Instant,
}

/// Everything a single frame needs to know about its schedule, so the codec
/// hooks all see a consistent view.
pub(crate) struct FramePlan {
    pub gop: GopPosition,
    /// Presentation order (PTS).
    pub display_order: u64,
    /// Encode order (DTS); `0` marks the very first encoded frame.
    pub encode_index: u64,
}

impl FramePlan {
    pub fn is_idr(&self) -> bool {
        self.gop.frame_type.is_idr()
    }
    pub fn is_reference(&self) -> bool {
        self.gop.is_reference
    }
    pub fn is_first_frame(&self) -> bool {
        self.encode_index == 0
    }
    pub fn pic_order_cnt(&self) -> i32 {
        self.gop.pic_order_cnt
    }
    /// The stream-level frame type for packet metadata.
    pub fn frame_type(&self) -> FrameType {
        match self.gop.frame_type {
            GopFrameType::Idr => FrameType::Idr,
            GopFrameType::P => FrameType::P,
        }
    }
}

/// The codec header (if any) produced by [`VideoCodec::begin_picture`].
pub(crate) struct PictureSetup {
    /// Codec header to prepend (SPS/PPS, VPS/SPS/PPS, AV1 sequence header).
    /// `Some` only for frames that carry one (typically the IDR/key frame).
    pub header: Option<Vec<u8>>,
}

/// What differs between the codecs in one frame's encode commands, for
/// [`EncoderCommon::record_encode`].
pub(crate) struct EncodeCommands<'a, R, L> {
    /// The slots for begin coding: the setup slot as inactive (index -1),
    /// if the frame is a reference, then the references it uses.
    pub begin_slots: &'a [vk::VideoReferenceSlotInfoKHR<'a>],
    /// The encode, with the codec's picture info chained in.
    pub encode_info: &'a vk::VideoEncodeInfoKHR<'a>,
    pub rate_control: &'a RateControlPlan,
    /// The codec's rate control info, such as
    /// `VideoEncodeH264RateControlInfoKHR`.
    pub codec_rate_control: R,
    /// The codec's rate control layer info, such as
    /// `VideoEncodeH264RateControlLayerInfoKHR`.
    pub codec_layer: L,
    /// Whether this is the session's first frame.
    pub first_frame: bool,
    /// Whether to reset the coding state and set rate control and the
    /// quality level with a control command. At least on the first frame.
    pub reset: bool,
    /// Whether to make the DPB write visible with a barrier after the encode.
    pub post_encode_barrier: bool,
}

/// The resolved rate-control decision for a frame.
///
/// The CQP/CBR/VBR selection logic is identical across codecs; only the default
/// QP the controller starts from differs (H.264/H.265 use 26, AV1 uses 128), so
/// the codec passes that in. Each codec then wires these values into its own
/// `VideoEncode*RateControl*InfoKHR` structs.
pub(crate) struct RateControlPlan {
    pub mode: vk::VideoEncodeRateControlModeFlagsKHR,
    pub average_bitrate: u32,
    pub max_bitrate: u32,
    /// QP/q-index: the configured quantizer for CQP, otherwise the
    /// codec's default starting point for the bitrate controller.
    pub qp: u32,
}

impl RateControlPlan {
    pub fn new(config: &EncodeConfig, controller_default_qp: u32) -> Self {
        match config.rate_control {
            RateControl::Cqp { qp } => Self {
                mode: vk::VideoEncodeRateControlModeFlagsKHR::DISABLED,
                average_bitrate: 0,
                max_bitrate: 0,
                qp,
            },
            RateControl::Cbr { bitrate } => Self {
                mode: vk::VideoEncodeRateControlModeFlagsKHR::CBR,
                average_bitrate: bitrate,
                max_bitrate: bitrate,
                qp: controller_default_qp,
            },
            RateControl::Vbr { target, max } => Self {
                mode: vk::VideoEncodeRateControlModeFlagsKHR::VBR,
                average_bitrate: target,
                max_bitrate: max,
                qp: controller_default_qp,
            },
        }
    }

    pub fn is_disabled(&self) -> bool {
        self.mode == vk::VideoEncodeRateControlModeFlagsKHR::DISABLED
    }
}

/// A video codec plugged into the generic [`CodecEncoder`].
///
/// Implementors are small state structs (reference lists, syntax counters,
/// parameter caches). The generic driver calls these hooks in order around each
/// frame; everything else (slot rotation, upload, readback, teardown) is shared.
pub(crate) trait VideoCodec: Sized + Send {
    /// Per-frame prologue: codec IDR/key-frame resets and header retrieval.
    /// Runs before recording; returns the packet's frame type and header.
    fn begin_picture(
        &mut self,
        common: &mut EncoderCommon,
        plan: &FramePlan,
    ) -> Result<PictureSetup>;

    /// Record the codec-specific encode commands and submit the frame. Builds
    /// the StdVideo* graph on its own stack (so the FFI pointers stay valid until
    /// `cmd_encode_video`) and finishes via [`EncoderCommon::submit_frame`].
    /// An error means that nothing was submitted.
    fn record_picture(&mut self, common: &mut EncoderCommon, plan: &FramePlan) -> Result<()>;

    /// Per-frame epilogue: advance reference lists, syntax counters and the next
    /// DPB slot. Runs after submission.
    fn end_picture(&mut self, common: &mut EncoderCommon, plan: &FramePlan);

    /// Reference frame invalidation: drop every held reference the client can no
    /// longer decode and arrange for the next frame to predict from a surviving
    /// one.
    ///
    /// `first_lost_display_order` is the display order (the `pts` reported on
    /// encoded packets) of the earliest frame the client lost. In a linear P
    /// chain every reference at or after that frame is transitively undecodable,
    /// so all of them are dropped. Returns `true` if a usable reference survives
    /// (the next P-frame can be predicted from it); `false` if none remain and
    /// the caller must fall back to an IDR.
    fn invalidate_references(
        &mut self,
        common: &mut EncoderCommon,
        first_lost_display_order: u64,
    ) -> bool;

    /// Build the codec parameter sets and create Vulkan session parameters.
    /// Used at init and by `set_color`.
    fn create_session_params(
        &self,
        common: &EncoderCommon,
        desc: &ColorDescription,
    ) -> Result<vk::VideoSessionParametersKHR>;

    /// Drop any cached header after the session parameters change.
    fn invalidate_header_cache(&mut self);
}

/// The codec-generic encoder: shared state plus a codec.
///
/// This owns the entire public encode flow; the codec only supplies its
/// differences through [`VideoCodec`].
pub struct CodecEncoder<C: VideoCodec> {
    pub(crate) common: EncoderCommon,
    pub(crate) codec: C,
}

// SAFETY: the only non-Send state is the persistently-mapped bitstream pointer
// inside the pipeline slots, which is synchronized via Vulkan fences and only
// read on the dedicated readback thread (see `pipeline`). The codec state is
// `Send` by the trait bound.
unsafe impl<C: VideoCodec> Send for CodecEncoder<C> {}

impl<C: VideoCodec> CodecEncoder<C> {
    /// The input image of the slot the next frame takes (a
    /// `ColorConverter::convert` target that avoids an intermediate copy).
    pub fn input_image(&self) -> vk::Image {
        self.common.pipeline.input_image()
    }

    /// Encode one frame from `input`, returning a future for its packet.
    /// See [`crate::Encoder::submit`].
    pub fn encode_frame(&mut self, input: FrameInput, sync: FrameSync) -> Result<EncodeFuture> {
        // Waits until the slot we are about to record over was read back.
        let slot = self.common.pipeline.next_index();
        let (frame, released) = self.common.prepare_source(slot, input, sync)?;
        self.common.pipeline.take_slot();
        let mut future = self.common.pipeline.new_future();

        let plan = FramePlan {
            gop: self.common.gop.get_next_frame(),
            display_order: self.common.input_frame_num,
            encode_index: self.common.encode_frame_num,
        };
        self.common.input_frame_num += 1;
        match self.encode_planned(frame, &plan) {
            Ok(point) => {
                future.set_released(released.unwrap_or(point));
                Ok(future)
            }
            Err(e) => {
                self.common.pipeline.release(slot);
                self.common.pipeline.drop_newest_future();
                // The codec may have reset its references for this frame.
                self.common.gop.request_idr();
                Err(e)
            }
        }
    }

    /// Encode `frame` now, as `plan` says. Returns the point its encode
    /// submission signals. An error means that nothing was submitted.
    fn encode_planned(&mut self, frame: HeldFrame, plan: &FramePlan) -> Result<TimelinePoint> {
        let submitted = frame.submitted;
        self.common.begin_encode(frame, Delivery::Packet);

        let setup = self.codec.begin_picture(&mut self.common, plan)?;
        self.common.set_pending_metadata(SlotPacketMetadata {
            frame_type: plan.frame_type(),
            pts: plan.display_order,
            dts: plan.encode_index as i64,
            header: setup.header,
            timestamps: [0; 2],
            now: submitted,
        });

        self.codec.record_picture(&mut self.common, plan)?;
        self.common.encode_frame_num += 1;
        self.codec.end_picture(&mut self.common, plan);
        Ok(self
            .common
            .encode_point
            .take()
            .expect("record_picture submits the frame"))
    }

    /// End-of-stream barrier: wait for all in-flight frames to be read back.
    pub fn flush(&mut self) -> Result<()> {
        self.common.pipeline.flush();
        Ok(())
    }

    /// Force the next frame to be an IDR/key frame.
    pub fn request_idr(&mut self) {
        self.common.gop.request_idr();
    }

    /// Reference frame invalidation: drop references the client lost and predict
    /// the next frame from a surviving reference, falling back to an IDR when no
    /// reference survives. See [`crate::Encoder::invalidate_reference_frames`].
    pub fn invalidate_reference_frames(&mut self, first_lost_display_order: u64) {
        if !self
            .codec
            .invalidate_references(&mut self.common, first_lost_display_order)
        {
            self.common.gop.request_idr();
        }
    }

    /// Rebuild session parameters with a new stream colour; the next frame is
    /// an IDR/key frame carrying the updated header.
    pub fn set_color(&mut self, color: StreamColor) -> Result<()> {
        // Drain in-flight encodes before mutating shared session parameters.
        self.common.pipeline.wait_all_free();
        let desc = color.description();

        let old_session_params = self.common.session_params;
        let new_session_params = self.codec.create_session_params(&self.common, &desc)?;
        unsafe {
            self.common
                .video_queue_fn
                .destroy_video_session_parameters(old_session_params, None);
        }

        self.common.session_params = new_session_params;
        self.common.config.color = color;
        self.codec.invalidate_header_cache();
        self.common.gop.request_idr();
        Ok(())
    }
}

impl<C: VideoCodec> Drop for CodecEncoder<C> {
    fn drop(&mut self) {
        unsafe {
            let common = &mut self.common;
            let device = common.context.device();
            // Wait on just the queues this encoder used, not the whole device.
            let _ = device.queue_wait_idle(common.context.transfer_queue());
            if let Some(q) = common.context.video_encode_queue() {
                let _ = device.queue_wait_idle(q);
            }

            common.pipeline.destroy(device);
            // A caller may still wait on points these handed out.
            common
                .context
                .retire_semaphore(common.pipeline.timeline_semaphore());
            common
                .context
                .retire_semaphore(common.upload_timeline.semaphore());

            destroy_encoder_resources(
                device,
                &common.video_queue_fn,
                &EncoderTeardown {
                    command_pool: common.command_pool,
                    upload_command_pool: common.upload_command_pool,
                    upload_fence: common.upload_fence,
                    dpb_images: &common.dpb_images,
                    dpb_image_views: &common.dpb_image_views,
                    dpb_image_memories: &common.dpb_image_memories,
                    session: common.session,
                    session_params: common.session_params,
                    session_memory: &common.session_memory,
                },
            );
        }
    }
}

/// What a codec passes to [`build_encoder_common`]; the codec owns the parts that
/// genuinely differ (its profile, block size, reference cap), the builder owns
/// the rest.
pub(crate) struct CommonInitRequest<'a> {
    pub context: &'a VideoContext,
    pub config: &'a EncodeConfig,
    /// The codec profile. Moved into [`EncoderCommon`], since images made
    /// later have to name it too.
    pub profile: EncodeProfile,
    /// The RGB format the encoder converts from, when it takes RGB input.
    pub rgb_input: Option<RgbFormat>,
    /// Device capabilities for this profile, resolved by [`query_video_caps`].
    pub caps: &'a DeviceVideoCaps,
    /// Codec block size for coded-extent alignment (macroblock / CTB / superblock).
    pub align_unit: u32,
    /// Upper bound on the reference window the codec's syntax allows.
    pub max_reference_window: usize,
    /// How many references one frame may predict from, from the codec's own
    /// capabilities (`maxPPictureL0ReferenceCount`, `maxSingleReferenceCount`).
    pub max_frame_references: usize,
    pub bitstream_buffer_size: usize,
    /// Whether the codec can use a layered DPB image when the driver lacks
    /// `SEPARATE_REFERENCE_IMAGES` (H.264/H.265 yes; AV1 no).
    pub allow_layered_dpb: bool,
    /// What the hardware RGB conversion supports for this profile, queried
    /// alongside the codec capabilities when `rgb_input` is set.
    pub rgb_caps: Option<RgbConversionCaps>,
}

/// Result of [`build_encoder_common`]: the assembled common state plus the
/// negotiated reference counts the codec needs for its parameter sets.
pub(crate) struct CommonInit {
    pub common: EncoderCommon,
    /// How many reconstructed frames stay in the DPB as references.
    pub reference_window: u32,
    /// How many of the newest of those one frame predicts from.
    pub active_reference_count: u32,
}

/// The device-capability fields the generic init needs, resolved once by the
/// codec. Copied out of the Vulkan query so they outlive its pointer chain.
pub(crate) struct DeviceVideoCaps {
    pub picture_access_granularity: vk::Extent2D,
    pub min_coded_extent: vk::Extent2D,
    pub max_coded_extent: vk::Extent2D,
    pub max_dpb_slots: u32,
    pub max_active_reference_pictures: u32,
    pub flags: vk::VideoCapabilityFlagsKHR,
    pub std_header_version: vk::ExtensionProperties,
}

/// Call `vkGetPhysicalDeviceVideoCapabilitiesKHR` and return its raw result.
pub(crate) fn query_video_capabilities(
    context: &VideoContext,
    profile_info: &vk::VideoProfileInfoKHR,
    capabilities: &mut vk::VideoCapabilitiesKHR,
) -> vk::Result {
    let video_queue_instance =
        ash::khr::video_queue::Instance::load(context.entry(), context.instance());
    unsafe {
        (video_queue_instance
            .fp()
            .get_physical_device_video_capabilities_khr)(
            context.physical_device(),
            profile_info,
            capabilities,
        )
    }
}

/// Run `vkGetPhysicalDeviceVideoCapabilitiesKHR` and resolve the fields the
/// generic init needs.
///
/// The driver *requires* the codec's `VideoEncode*CapabilitiesKHR` chained into
/// `pNext`, so the codec builds the `capabilities` chain (the only codec-specific
/// part) and passes it in already populated with its encode/codec caps structs.
pub(crate) fn query_video_caps(
    context: &VideoContext,
    profile_info: &vk::VideoProfileInfoKHR,
    capabilities: &mut vk::VideoCapabilitiesKHR,
) -> Result<DeviceVideoCaps> {
    let result = query_video_capabilities(context, profile_info, capabilities);
    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::NoSuitableDevice(format!(
            "Failed to query Vulkan Video encode capabilities: {:?}",
            result
        )));
    }
    Ok(DeviceVideoCaps {
        picture_access_granularity: capabilities.picture_access_granularity,
        min_coded_extent: capabilities.min_coded_extent,
        max_coded_extent: capabilities.max_coded_extent,
        max_dpb_slots: capabilities.max_dpb_slots,
        max_active_reference_pictures: capabilities.max_active_reference_pictures,
        flags: capabilities.flags,
        std_header_version: capabilities.std_header_version,
    })
}

/// Query capabilities, create the video session, DPB images, command resources,
/// the encode pipeline and the GOP structure — the ~85% of encoder
/// initialization that is identical across codecs.
pub(crate) fn build_encoder_common(req: CommonInitRequest) -> Result<CommonInit> {
    let context = req.context;
    let config = req.config;
    let profile_info = req.profile.info();
    let width = config.dimensions.width;
    let height = config.dimensions.height;

    let video_queue_fn = ash::khr::video_queue::Device::load(context.instance(), context.device());
    let video_encode_fn =
        ash::khr::video_encode_queue::Device::load(context.instance(), context.device());

    // Capabilities were queried by the codec (which chains its codec-specific
    // capability struct, required by the driver) via [`query_video_caps`].
    let capabilities = req.caps;

    // Align the coded extent to lcm(codec block size, device granularity), then
    // clamp up to the device minimum and re-align.
    let gran_w = capabilities.picture_access_granularity.width.max(1);
    let gran_h = capabilities.picture_access_granularity.height.max(1);
    let align_w = lcm(req.align_unit, gran_w);
    let align_h = lcm(req.align_unit, gran_h);
    let aligned_width = align_up(
        align_up(width, align_w).max(capabilities.min_coded_extent.width),
        align_w,
    );
    let aligned_height = align_up(
        align_up(height, align_h).max(capabilities.min_coded_extent.height),
        align_h,
    );
    if aligned_width > capabilities.max_coded_extent.width
        || aligned_height > capabilities.max_coded_extent.height
    {
        return Err(PixelForgeError::InvalidInput(format!(
            "Requested coded extent {}x{} (aligned to {}x{} with granularity {}x{}) exceeds device max {}x{} for this profile",
            width,
            height,
            aligned_width,
            aligned_height,
            gran_w,
            gran_h,
            capabilities.max_coded_extent.width,
            capabilities.max_coded_extent.height
        )));
    }
    tracing::info!(
        "Using coded extent {}x{} (granularity {}x{}, min {}x{}, max {}x{})",
        aligned_width,
        aligned_height,
        gran_w,
        gran_h,
        capabilities.min_coded_extent.width,
        capabilities.min_coded_extent.height,
        capabilities.max_coded_extent.width,
        capabilities.max_coded_extent.height
    );

    // With RGB input the encoder converts, so the session needs its settings
    // and the input image is RGB; the DPB stays YUV either way.
    // `Encoder::new` has already checked the configuration itself.
    let rgb_session = match req.rgb_input {
        Some(_) => {
            let caps = req.rgb_caps.ok_or_else(|| {
                PixelForgeError::NoSuitableDevice(
                    "RGB input: the driver reported no conversion capabilities".to_string(),
                )
            })?;
            Some(caps.session_info(&config.color.description())?)
        }
        None => None,
    };

    // Pick input (SRC) and reference (DPB) formats.
    let preferred_src_format = get_video_format(config.chroma_format, config.bit_depth);
    let supported_src_formats = query_supported_video_formats(
        context,
        profile_info,
        vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR,
    )?;
    let supported_dpb_formats = query_supported_video_formats(
        context,
        profile_info,
        vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR,
    )?;
    if supported_src_formats.is_empty() {
        return Err(PixelForgeError::NoSuitableDevice(
            "No supported Vulkan Video SRC formats for this profile".to_string(),
        ));
    }
    if supported_dpb_formats.is_empty() {
        return Err(PixelForgeError::NoSuitableDevice(
            "No supported Vulkan Video DPB formats for this profile".to_string(),
        ));
    }
    let picture_format = req
        .rgb_input
        .map_or(preferred_src_format, |format| format.vk_format());
    if !supported_src_formats
        .iter()
        .any(|f| f.format == picture_format)
    {
        return Err(PixelForgeError::NoSuitableDevice(format!(
            "Input format {:?} is not supported for VIDEO_ENCODE_SRC_KHR. Supported: {:?}",
            picture_format, supported_src_formats
        )));
    }
    let reference_picture_format = supported_dpb_formats
        .iter()
        .map(|f| f.format)
        .find(|f| *f == preferred_src_format)
        .unwrap_or(supported_dpb_formats[0].format);

    // Negotiate the reference window and the references each frame uses.
    //
    // The window is how many reconstructed frames stay in the DPB, which is
    // what reference frame invalidation can recover from. Each frame predicts
    // from only the newest `active` of them, so a device that predicts from
    // one or two references still keeps older frames to fall back on. This is
    // the model NVENC documents for `NvEncInvalidateRefFrames`.
    let max_dpb_slots_supported = capabilities.max_dpb_slots as usize;
    let max_active_supported = capabilities.max_active_reference_pictures as usize;
    if max_dpb_slots_supported < 2 {
        return Err(PixelForgeError::NoSuitableDevice(format!(
            "Device reports max_dpb_slots={} for this profile; need at least 2",
            max_dpb_slots_supported
        )));
    }
    // At least 1: `max_dpb_slots_supported` is at least 2, and every codec's
    // window cap and `MAX_DPB_SLOTS` are larger.
    let window_limit = req
        .max_reference_window
        .min(max_dpb_slots_supported - 1)
        .min(MAX_DPB_SLOTS - 1);
    let reference_window = (config.max_reference_frames as usize).clamp(1, window_limit);
    let mut max_active_reference_pictures = reference_window
        .min(max_active_supported)
        .min(req.max_frame_references);
    // `PIXELFORGE_MAX_ACTIVE_REFERENCES=1` predicts from fewer references than
    // the device allows, to reproduce a device with tighter limits.
    if let Some(cap) = std::env::var("PIXELFORGE_MAX_ACTIVE_REFERENCES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        max_active_reference_pictures = max_active_reference_pictures.min(cap);
    }
    if max_active_reference_pictures < 1 && max_active_supported >= 1 {
        max_active_reference_pictures = 1;
    }
    // B-frames are not yet supported, so a reconstructed-frame slot plus the
    // window is enough.
    let dpb_slot_count = reference_window + 1;
    tracing::debug!(
        "Reference window {reference_window}, {max_active_reference_pictures} used per frame \
         (device: {max_dpb_slots_supported} DPB slots, {max_active_supported} active, \
         {} per frame)",
        req.max_frame_references
    );

    let encode_queue_family = context.video_encode_queue_family().ok_or_else(|| {
        PixelForgeError::NoSuitableDevice("No video encode queue family available".to_string())
    })?;

    // Use the driver-reported std header version for this profile.
    let std_header_version = capabilities.std_header_version;
    let mut rgb_session_info = rgb_session.unwrap_or_default();
    let mut session_create_info = vk::VideoSessionCreateInfoKHR::default()
        .queue_family_index(encode_queue_family)
        .flags(vk::VideoSessionCreateFlagsKHR::empty())
        .video_profile(profile_info)
        .picture_format(picture_format)
        .max_coded_extent(vk::Extent2D {
            width: aligned_width,
            height: aligned_height,
        })
        .reference_picture_format(reference_picture_format)
        .max_dpb_slots(dpb_slot_count as u32)
        .max_active_reference_pictures(max_active_reference_pictures as u32)
        .std_header_version(&std_header_version);
    if rgb_session.is_some() {
        session_create_info = session_create_info.push(&mut rgb_session_info);
    }

    let mut session = vk::VideoSessionKHR::null();
    let result = unsafe {
        (video_queue_fn.fp().create_video_session_khr)(
            context.device().handle(),
            &session_create_info,
            std::ptr::null(),
            &mut session,
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::vulkan("vkCreateVideoSessionKHR", result));
    }
    let session_memory = allocate_session_memory(context, session, &video_queue_fn)?;

    // Use a layered DPB only when allowed and the driver lacks separate
    // reference images (AMD RADV).
    let supports_separate_dpb = capabilities
        .flags
        .contains(vk::VideoCapabilityFlagsKHR::SEPARATE_REFERENCE_IMAGES);
    let use_layered_dpb = req.allow_layered_dpb && !supports_separate_dpb;
    if use_layered_dpb {
        tracing::info!("Using layered DPB (driver does not support separate reference images)");
    }

    let (dpb_images, dpb_image_memories, dpb_image_views) = create_dpb_images(
        context,
        aligned_width,
        aligned_height,
        reference_picture_format,
        dpb_slot_count,
        profile_info,
        use_layered_dpb,
    )?;

    let upload_queue_family = context.transfer_queue_family();
    let cmd = create_command_resources(context, encode_queue_family, upload_queue_family)?;

    let pipeline = EncodePipeline::new(&PipelineConfig {
        context,
        aligned_width,
        aligned_height,
        picture_format,
        rgb_input: req.rgb_input.is_some(),
        chroma_format: config.chroma_format,
        bit_depth: config.bit_depth,
        bitstream_buffer_size: req.bitstream_buffer_size,
        profile_info,
        command_pool: cmd.command_pool,
        held_frames: 0,
        upload_command_buffer: cmd.upload_command_buffer,
        upload_fence: cmd.upload_fence,
    })?;

    // B-frames are not yet supported, so an I-P GOP. Match the SPS's POC
    // wrap (no-op for AV1, which keys off order hints).
    let mut gop = GopStructure::new(config.gop_size);
    gop.set_max_poc_lsb(4);

    let upload_command_buffers = allocate_command_buffers(
        context,
        cmd.upload_command_pool,
        pipeline.slot_count() as u32,
    )?;

    let common = EncoderCommon {
        context: context.clone(),
        config: config.clone(),
        video_queue_fn,
        video_encode_fn,
        session,
        session_params: vk::VideoSessionParametersKHR::null(),
        session_memory,
        aligned_width,
        aligned_height,
        pipeline,
        gop,
        input_frame_num: 0,
        encode_frame_num: 0,
        dpb_images,
        dpb_image_memories,
        dpb_image_views,
        dpb_slot_count,
        dpb_slot_active: vec![false; dpb_slot_count],
        use_layered_dpb,
        current_dpb_slot: 0,
        command_pool: cmd.command_pool,
        upload_command_pool: cmd.upload_command_pool,
        upload_fence: cmd.upload_fence,
        upload_command_buffers,
        upload_timeline: TimelineChain::new(context)?,
        pending_waits: Vec::new(),
        pending_signals: Vec::new(),
        source: SourceBinding::default(),
        delivery: Delivery::Packet,
        encode_point: None,
        profile: req.profile,
        rgb_input: req.rgb_input,
    };

    Ok(CommonInit {
        common,
        reference_window: reference_window as u32,
        active_reference_count: max_active_reference_pictures as u32,
    })
}
