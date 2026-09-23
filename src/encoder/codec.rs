//! The codec-generic encoder.
//!
//! Everything that is identical across H.264, H.265 and AV1 lives here:
//! the shared per-encoder state ([`EncoderCommon`]), the generic driver
//! ([`CodecEncoder`]) that owns the public `encode`/`flush`/`set_color_description`
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

use crate::encoder::dpb::MAX_DPB_SLOTS;
use crate::encoder::gop::{GopFrameType, GopPosition, GopStructure};
use crate::encoder::pipeline::{EncodeFuture, EncodePipeline, PipelineConfig, SlotPacketMetadata};
use crate::encoder::resources::{
    EncoderTeardown, UploadParams, align_up, allocate_session_memory, create_command_resources,
    create_dpb_images, destroy_encoder_resources, get_video_format, lcm,
    query_supported_video_formats, upload_image_to_input,
};
use crate::encoder::{ColorDescription, EncodeConfig, FrameType, RateControlMode};
use crate::error::{PixelForgeError, Result};
use crate::sync::TimelinePoint;
use crate::vulkan::VideoContext;

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
    pub upload_command_buffer: vk::CommandBuffer,
    pub upload_fence: vk::Fence,
    /// Caller work this frame's first submission must wait for. The upload
    /// takes them when it copies, otherwise the encode submission does.
    pub pending_waits: Vec<TimelinePoint>,
}

impl EncoderCommon {
    /// Prologue: wait until the slot we are about to record over has been read
    /// back, pull the next GOP position, and snapshot the counters this frame
    /// will encode with.
    pub fn begin_frame(&mut self) -> FramePlan {
        self.pipeline.wait_current_free();
        let gop = self.gop.get_next_frame();
        let display_order = self.input_frame_num;
        self.input_frame_num += 1;
        FramePlan {
            gop,
            display_order,
            encode_index: self.encode_frame_num,
        }
    }

    /// Copy a source image into the current slot's input image. No-op when the
    /// source already *is* the slot image (e.g. converted in place).
    pub fn upload(&mut self, src_image: vk::Image) -> Result<()> {
        let (dst_image, input_image_layout) = {
            let slot = self.pipeline.current();
            (slot.input_image, slot.input_image_layout)
        };
        if src_image == dst_image {
            return Ok(());
        }
        let waits = std::mem::take(&mut self.pending_waits);

        let params = UploadParams {
            upload_command_buffer: self.upload_command_buffer,
            upload_fence: self.upload_fence,
            src_image,
            dst_image,
            width: self.config.dimensions.width,
            height: self.config.dimensions.height,
            pixel_format: self.config.pixel_format,
            input_image_layout,
            rgb: self.config.rgb_input.is_some(),
            upload_queue: self.context.transfer_queue(),
            waits: &waits,
        };
        upload_image_to_input(&self.context, &params)?;
        self.pipeline.current_mut().input_image_layout = vk::ImageLayout::VIDEO_ENCODE_SRC_KHR;
        Ok(())
    }

    /// Record the metadata for the packet the current slot will produce.
    pub fn set_pending_metadata(&mut self, metadata: SlotPacketMetadata) {
        self.pipeline.set_pending_metadata(metadata);
    }

    /// Submit the recorded command buffer for the current slot, mark its DPB
    /// slot active, and return the future that resolves with its packet.
    pub fn submit_frame(&mut self) -> Result<EncodeFuture> {
        let encode_queue = self.context.video_encode_queue().ok_or_else(|| {
            PixelForgeError::NoSuitableDevice("No video encode queue available".to_string())
        })?;
        let waits = std::mem::take(&mut self.pending_waits);
        let future = self.pipeline.submit_current(
            self.context.device(),
            self.context.sync2(),
            encode_queue,
            &waits,
        )?;
        self.dpb_slot_active[self.current_dpb_slot as usize] = true;
        Ok(future)
    }

    /// Epilogue: advance to the next pipeline slot.
    pub fn advance(&mut self) {
        self.pipeline.advance();
    }

    /// The `ash` device handle.
    pub fn device(&self) -> &ash::Device {
        self.context.device()
    }
}

/// Everything a single frame needs to know about its schedule, snapshotted by
/// [`EncoderCommon::begin_frame`] so the codec hooks all see a consistent view.
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
    pub fn is_b_frame(&self) -> bool {
        self.gop.frame_type == GopFrameType::B
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
            GopFrameType::Idr | GopFrameType::I => FrameType::I,
            GopFrameType::P => FrameType::P,
            GopFrameType::B => FrameType::B,
        }
    }
}

/// The codec header (if any) and stream frame type produced by
/// [`VideoCodec::begin_picture`].
pub(crate) struct PictureSetup {
    pub frame_type: FrameType,
    /// Codec header to prepend (SPS/PPS, VPS/SPS/PPS, AV1 sequence header).
    /// `Some` only for frames that carry one (typically the IDR/key frame).
    pub header: Option<Vec<u8>>,
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
    /// QP/q-index: the configured quality level for CQP/Disabled, otherwise the
    /// codec's default starting point for the bitrate controller.
    pub qp: u32,
}

impl RateControlPlan {
    pub fn new(config: &EncodeConfig, controller_default_qp: u32) -> Self {
        match config.rate_control_mode {
            RateControlMode::Cqp | RateControlMode::Disabled => Self {
                mode: vk::VideoEncodeRateControlModeFlagsKHR::DISABLED,
                average_bitrate: 0,
                max_bitrate: 0,
                qp: config.quality_level,
            },
            RateControlMode::Cbr => Self {
                mode: vk::VideoEncodeRateControlModeFlagsKHR::CBR,
                average_bitrate: config.target_bitrate,
                max_bitrate: config.target_bitrate,
                qp: controller_default_qp,
            },
            RateControlMode::Vbr => Self {
                mode: vk::VideoEncodeRateControlModeFlagsKHR::VBR,
                average_bitrate: config.target_bitrate,
                max_bitrate: config.max_bitrate,
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
    fn record_picture(
        &mut self,
        common: &mut EncoderCommon,
        plan: &FramePlan,
    ) -> Result<EncodeFuture>;

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
    /// Used at init and by `set_color_description`.
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
    /// The internal input image for the current slot (a `ColorConverter::convert`
    /// target that avoids an intermediate copy).
    pub fn input_image(&self) -> vk::Image {
        self.common.pipeline.input_image()
    }

    /// Encode one frame once `wait` is reached, returning a future for its
    /// packet. See [`crate::Encoder::encode_after`].
    pub fn encode_after(
        &mut self,
        src_image: vk::Image,
        wait: &[TimelinePoint],
    ) -> Result<EncodeFuture> {
        let plan = self.common.begin_frame();
        self.common.pending_waits.clear();
        self.common.pending_waits.extend_from_slice(wait);
        self.common.upload(src_image)?;

        let setup = self.codec.begin_picture(&mut self.common, &plan)?;
        self.common.set_pending_metadata(SlotPacketMetadata {
            frame_type: setup.frame_type,
            is_key_frame: plan.is_idr(),
            pts: plan.display_order,
            dts: plan.encode_index,
            header: setup.header,
            timestamps: [0; 2],
            now: std::time::Instant::now(),
        });

        let future = self.codec.record_picture(&mut self.common, &plan)?;
        self.common.encode_frame_num += 1;
        self.codec.end_picture(&mut self.common, &plan);
        self.common.advance();
        Ok(future)
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

    /// Rebuild session parameters with a new color description; the next frame is
    /// an IDR/key frame carrying the updated header.
    pub fn set_color_description(&mut self, desc: ColorDescription) -> Result<()> {
        // Drain in-flight encodes before mutating shared session parameters.
        self.common.pipeline.wait_all_free();

        let old_session_params = self.common.session_params;
        let new_session_params = self.codec.create_session_params(&self.common, &desc)?;
        unsafe {
            self.common
                .video_queue_fn
                .destroy_video_session_parameters(old_session_params, None);
        }

        self.common.session_params = new_session_params;
        self.common.config.color_description = Some(desc);
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
    /// The codec profile, with its `VideoEncode*ProfileInfoKHR` already chained.
    pub profile_info: &'a vk::VideoProfileInfoKHR<'a>,
    /// Device capabilities for this profile, resolved by [`query_video_caps`].
    pub caps: &'a DeviceVideoCaps,
    /// Codec block size for coded-extent alignment (macroblock / CTB / superblock).
    pub align_unit: u32,
    /// Upper bound on active reference pictures the codec's syntax allows.
    pub max_active_refs_cap: usize,
    pub bitstream_buffer_size: usize,
    /// Whether the codec can use a layered DPB image when the driver lacks
    /// `SEPARATE_REFERENCE_IMAGES` (H.264/H.265 yes; AV1 no).
    pub allow_layered_dpb: bool,
    /// What the hardware RGB conversion supports for this profile, queried
    /// alongside the codec capabilities when [`EncodeConfig::rgb_input`] is
    /// set.
    pub rgb_caps: Option<RgbConversionCaps>,
}

/// The profile struct that turns on RGB input. When
/// [`EncodeConfig::rgb_input`] is set it goes on the one profile the codec
/// builds, which every capability query, image, query pool and the session
/// then share: they must all name the same profile.
pub(crate) fn rgb_conversion_profile() -> vk::VideoEncodeProfileRgbConversionInfoVALVE<'static> {
    vk::VideoEncodeProfileRgbConversionInfoVALVE::default().perform_encode_rgb_conversion(true)
}

/// `VkVideoEncodeRgbConversionCapabilitiesVALVE`, without its `pNext`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RgbConversionCaps {
    models: vk::VideoEncodeRgbModelConversionFlagsVALVE,
    ranges: vk::VideoEncodeRgbRangeCompressionFlagsVALVE,
    x_chroma_offsets: vk::VideoEncodeRgbChromaOffsetFlagsVALVE,
    y_chroma_offsets: vk::VideoEncodeRgbChromaOffsetFlagsVALVE,
}

impl From<&vk::VideoEncodeRgbConversionCapabilitiesVALVE<'_>> for RgbConversionCaps {
    fn from(caps: &vk::VideoEncodeRgbConversionCapabilitiesVALVE<'_>) -> Self {
        Self {
            models: caps.rgb_models,
            ranges: caps.rgb_ranges,
            x_chroma_offsets: caps.x_chroma_offsets,
            y_chroma_offsets: caps.y_chroma_offsets,
        }
    }
}

impl RgbConversionCaps {
    /// The session settings that make the hardware produce what `desc`
    /// describes, or why it cannot.
    fn session_info(
        &self,
        desc: &ColorDescription,
    ) -> Result<vk::VideoEncodeSessionRgbConversionCreateInfoVALVE<'static>> {
        use vk::VideoEncodeRgbChromaOffsetFlagsVALVE as Offset;
        use vk::VideoEncodeRgbModelConversionFlagsVALVE as Model;
        use vk::VideoEncodeRgbRangeCompressionFlagsVALVE as Range;

        // H.273 matrix coefficients.
        let model = match desc.matrix_coefficients {
            1 => Model::YCBCR_709,
            5 | 6 => Model::YCBCR_601,
            9 => Model::YCBCR_2020,
            other => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "RGB input: no hardware conversion for matrix coefficients {other}"
                )));
            }
        };
        let range = if desc.full_range {
            Range::FULL_RANGE
        } else {
            Range::NARROW_RANGE
        };
        if !self.models.contains(model) || !self.ranges.contains(range) {
            return Err(PixelForgeError::NoSuitableDevice(format!(
                "RGB input: the driver converts {:?} in {:?}, not {:?} in {:?}",
                self.models, self.ranges, model, range
            )));
        }
        // Chroma sited midway between luma samples on both axes, which is what
        // the colour converter's 2x2 average produces, so a stream looks the
        // same whichever path made it. Where the driver cannot, it gets
        // whichever siting it can do.
        let pick = |supported: Offset| {
            if supported.contains(Offset::MIDPOINT) {
                Offset::MIDPOINT
            } else {
                Offset::COSITED_EVEN
            }
        };
        Ok(
            vk::VideoEncodeSessionRgbConversionCreateInfoVALVE::default()
                .rgb_model(model)
                .rgb_range(range)
                .x_chroma_offset(pick(self.x_chroma_offsets))
                .y_chroma_offset(pick(self.y_chroma_offsets)),
        )
    }
}

/// Result of [`build_encoder_common`]: the assembled common state plus the
/// negotiated active-reference count the codec needs for its parameter sets.
pub(crate) struct CommonInit {
    pub common: EncoderCommon,
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
    let video_queue_instance =
        ash::khr::video_queue::Instance::load(context.entry(), context.instance());
    let result = unsafe {
        (video_queue_instance
            .fp()
            .get_physical_device_video_capabilities_khr)(
            context.physical_device(),
            profile_info,
            capabilities,
        )
    };
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
pub(crate) fn build_encoder_common(req: &CommonInitRequest) -> Result<CommonInit> {
    let context = req.context;
    let config = req.config;
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
    let rgb_session = match config.rgb_input {
        Some(_) if !context.has_video_encode_rgb_conversion() => {
            return Err(PixelForgeError::NoSuitableDevice(
                "RGB input needs VK_VALVE_video_encode_rgb_conversion, which this device \
                 does not have enabled"
                    .to_string(),
            ));
        }
        Some(_) => {
            let caps = req.rgb_caps.ok_or_else(|| {
                PixelForgeError::NoSuitableDevice(
                    "RGB input: the driver reported no conversion capabilities".to_string(),
                )
            })?;
            Some(
                caps.session_info(
                    &config
                        .color_description
                        .unwrap_or_else(ColorDescription::bt709),
                )?,
            )
        }
        None => None,
    };

    // Pick input (SRC) and reference (DPB) formats.
    let preferred_src_format = get_video_format(config.pixel_format, config.bit_depth);
    let supported_src_formats = query_supported_video_formats(
        context,
        req.profile_info,
        vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR,
    )?;
    let supported_dpb_formats = query_supported_video_formats(
        context,
        req.profile_info,
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
    let picture_format = config
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

    // Negotiate DPB slots and active references.
    let max_dpb_slots_supported = capabilities.max_dpb_slots as usize;
    let max_active_supported = capabilities.max_active_reference_pictures as usize;
    if max_dpb_slots_supported < 2 {
        return Err(PixelForgeError::NoSuitableDevice(format!(
            "Device reports max_dpb_slots={} for this profile; need at least 2",
            max_dpb_slots_supported
        )));
    }
    let mut target_active_refs = (config.max_reference_frames as usize)
        .min(max_active_supported)
        .min(req.max_active_refs_cap);
    if target_active_refs < 1 && max_active_supported >= 1 {
        target_active_refs = 1;
    }
    // B-frames are not yet supported, so a reconstructed-frame slot plus the
    // active references is enough.
    let dpb_slot_count = (target_active_refs + 1)
        .min(max_dpb_slots_supported)
        .min(MAX_DPB_SLOTS);
    let max_active_reference_pictures = target_active_refs.min(dpb_slot_count.saturating_sub(1));

    let encode_queue_family = context.video_encode_queue_family().ok_or_else(|| {
        PixelForgeError::NoSuitableDevice("No video encode queue family available".to_string())
    })?;

    // Use the driver-reported std header version for this profile.
    let std_header_version = capabilities.std_header_version;
    let mut rgb_session_info = rgb_session.unwrap_or_default();
    let mut session_create_info = vk::VideoSessionCreateInfoKHR::default()
        .queue_family_index(encode_queue_family)
        .flags(vk::VideoSessionCreateFlagsKHR::empty())
        .video_profile(req.profile_info)
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
        return Err(PixelForgeError::VideoSessionCreation(format!(
            "{:?}",
            result
        )));
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
        req.profile_info,
        use_layered_dpb,
    )?;

    let upload_queue_family = context.transfer_queue_family();
    let cmd = create_command_resources(context, encode_queue_family, upload_queue_family)?;

    let pipeline = EncodePipeline::new(&PipelineConfig {
        context,
        aligned_width,
        aligned_height,
        picture_format,
        rgb_input: config.rgb_input.is_some(),
        pixel_format: config.pixel_format,
        bit_depth: config.bit_depth,
        bitstream_buffer_size: req.bitstream_buffer_size,
        profile_info: req.profile_info,
        command_pool: cmd.command_pool,
        upload_command_buffer: cmd.upload_command_buffer,
        upload_fence: cmd.upload_fence,
    })?;

    // B-frames are not yet supported, so an I-P GOP. Set the SPS-matching
    // counters (no-ops for AV1, which keys off order hints).
    let mut gop = GopStructure::new_ip_only(config.gop_size);
    gop.set_max_frame_num(4);
    gop.set_max_poc_lsb(4);

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
        upload_command_buffer: cmd.upload_command_buffer,
        upload_fence: cmd.upload_fence,
        pending_waits: Vec::new(),
    };

    Ok(CommonInit {
        common,
        active_reference_count: max_active_reference_pictures as u32,
    })
}
