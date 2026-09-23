//! Vulkan context and initialization for video encoding.
use crate::encoder::Codec;
use crate::error::{PixelForgeError, Result};
use ash::vk;
use ash::vk::TaggedStructure;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use tracing::{debug, info, warn};

mod encode_device;
mod queues;
use queues::QueueOverrides;
pub use queues::{DeviceQueue, QueueRoles};

/// Route validation-layer messages into `tracing`.
///
/// Without a messenger the validation layer has nowhere to report to and its
/// findings are silently dropped, which makes "no validation errors" impossible
/// to verify. Severities map onto tracing levels so `RUST_LOG` controls the
/// noise: errors and warnings are always worth seeing, the layer's
/// informational chatter sits at debug.
unsafe extern "system" fn debug_utils_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let data = unsafe { &*callback_data };
    let message = if data.p_message.is_null() {
        std::borrow::Cow::Borrowed("(no message)")
    } else {
        unsafe { CStr::from_ptr(data.p_message) }.to_string_lossy()
    };
    let kind = if message_types.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION) {
        "validation"
    } else if message_types.contains(vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE) {
        "performance"
    } else {
        "general"
    };

    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        tracing::error!("Vulkan {kind}: {message}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        warn!("Vulkan {kind}: {message}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
        debug!("Vulkan {kind}: {message}");
    } else {
        tracing::trace!("Vulkan {kind}: {message}");
    }

    // Never abort the offending call; the caller decides what to do about it.
    vk::FALSE
}

/// Builder for creating a VideoContext.
#[must_use]
pub struct VideoContextBuilder {
    app_name: String,
    app_version: (u32, u32, u32),
    enable_validation: bool,
    required_encode_codecs: Vec<Codec>,
    required_decode_codecs: Vec<Codec>,
    /// `None` lets each path decide: a context that creates its own device
    /// enables unified image layouts when the device supports them, and one
    /// that adopts a caller's device assumes they were not enabled.
    unified_image_layouts: Option<bool>,
    queues: QueueOverrides,
}

impl Default for VideoContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoContextBuilder {
    /// Create a new VideoContextBuilder with default settings.
    pub fn new() -> Self {
        Self {
            app_name: "PixelForge".to_string(),
            app_version: (1, 0, 0),
            enable_validation: false,
            required_encode_codecs: Vec::new(),
            required_decode_codecs: Vec::new(),
            unified_image_layouts: None,
            queues: QueueOverrides::default(),
        }
    }

    /// Set the application name.
    pub fn app_name(mut self, name: &str) -> Self {
        self.app_name = name.to_string();
        self
    }

    /// Set the application version.
    pub fn app_version(mut self, major: u32, minor: u32, patch: u32) -> Self {
        self.app_version = (major, minor, patch);
        self
    }

    /// Enable or disable validation layers.
    pub fn enable_validation(mut self, enable: bool) -> Self {
        self.enable_validation = enable;
        self
    }

    /// Require video encode support for a codec.
    pub fn require_encode(mut self, codec: Codec) -> Self {
        self.required_encode_codecs.push(codec);
        self
    }

    /// Require video decode support for a codec.
    pub fn require_decode(mut self, codec: Codec) -> Self {
        self.required_decode_codecs.push(codec);
        self
    }

    /// Submit video encode work to `queue` on an adopted device.
    ///
    /// See [`Self::with_transfer_queue`] for why this matters. Ignored by
    /// [`Self::build`], which creates its own device and owns every queue on
    /// it.
    pub fn with_encode_queue(mut self, queue: DeviceQueue) -> Self {
        self.queues.encode = Some(queue);
        self
    }

    /// Submit video decode work to `queue` on an adopted device.
    ///
    /// See [`Self::with_transfer_queue`] for why this matters. Ignored by
    /// [`Self::build`], which creates its own device and owns every queue on
    /// it.
    pub fn with_decode_queue(mut self, queue: DeviceQueue) -> Self {
        self.queues.decode = Some(queue);
        self
    }

    /// Submit uploads and readback copies to `queue` on an adopted device.
    ///
    /// Without this, an adopted context takes queue 0 of each family it needs,
    /// and on most hardware queue 0 of the transfer and compute families is the
    /// caller's own graphics queue. A `VkQueue` must not be submitted to from
    /// two threads at once, so a caller that keeps submitting to its queue
    /// while pixelforge works on another thread has to either create a spare
    /// queue and give it here, or create its queue with
    /// `VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR` (see
    /// [`DeviceRequirements::internally_synchronized_queues`]).
    ///
    /// The queue's family must support transfer, and `queue.index` must be
    /// below the number of queues the caller created in that family. The
    /// second part cannot be checked. Ignored by [`Self::build`].
    pub fn with_transfer_queue(mut self, queue: DeviceQueue) -> Self {
        self.queues.transfer = Some(queue);
        self
    }

    /// Submit the colour converter's dispatches to `queue` on an adopted
    /// device. See [`Self::with_transfer_queue`]. Ignored by [`Self::build`].
    pub fn with_compute_queue(mut self, queue: DeviceQueue) -> Self {
        self.queues.compute = Some(queue);
        self
    }

    /// Build the VideoContext.
    pub fn build(self) -> Result<VideoContext> {
        VideoContext::new(self)
    }

    /// What a caller-created device must provide for pixelforge to decode on it.
    ///
    /// Use this when you already have your own Vulkan device (e.g. a renderer's)
    /// and want to decode into images that device can use directly, without the
    /// cross-device copy a separate context would require.
    pub fn decode_device_requirements(
        &self,
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<DeviceRequirements> {
        let families = find_decode_queue_families(
            entry,
            instance,
            physical_device,
            &self.required_decode_codecs,
        )?;
        let unified_image_layouts = supports_unified_image_layouts(instance, physical_device);
        let mut extensions = decode_extension_names(&self.required_decode_codecs);
        if unified_image_layouts {
            extensions.push(ash::khr::unified_image_layouts::NAME);
        }
        let queues = QueueRoles {
            encode: None,
            decode: Some(families.decode),
            transfer: families.transfer,
            compute: families.compute,
        };
        Ok(DeviceRequirements {
            queue_families: queues.unique_families(),
            queues,
            extensions,
            features: DeviceFeatures {
                synchronization2: true,
                timeline_semaphore: true,
                ..DeviceFeatures::default()
            },
            unified_image_layouts,
            internally_synchronized_queues: supports_internally_synchronized_queues(
                instance,
                physical_device,
            ),
        })
    }

    /// Declare that the adopted device was created with
    /// `VK_KHR_unified_image_layouts` enabled, including both
    /// `unifiedImageLayouts` and `unifiedImageLayoutsVideo`.
    ///
    /// Only meaningful for [`Self::build_from_existing_decode`]: a context that
    /// creates its own device enables the extension itself when the hardware
    /// has it. Vulkan offers no way to ask a device which features were enabled
    /// on it, so this is a promise pixelforge cannot verify. What it *can*
    /// check, and does, is that the physical device supports both bits; a
    /// declaration on hardware that cannot support them is a clean error rather
    /// than undefined behaviour.
    ///
    /// Getting it wrong the other way, declaring it without having enabled it,
    /// is undefined behaviour. Validation layers report it immediately, so
    /// develop with them on. This is the same kind of promise the adopted path
    /// already relies on for `timelineSemaphore`.
    ///
    /// Worth making: without it every decoded frame is copied into a private
    /// image before the caller sees it. With it, frames are the decoder's own
    /// images, sampleable in place.
    pub fn declare_unified_image_layouts(mut self) -> Self {
        self.unified_image_layouts = Some(true);
        self
    }

    /// Never use unified image layouts, even where the device supports them.
    ///
    /// Exists so the copying fallback can be exercised on hardware that would
    /// otherwise always take the zero-copy path. There is no reason to reach
    /// for this in production.
    pub fn without_unified_image_layouts(mut self) -> Self {
        self.unified_image_layouts = Some(false);
        self
    }

    /// Adopt a caller-created device for decoding, rather than creating one.
    ///
    /// The device must have been created with the queue families and extensions
    /// reported by [`Self::decode_device_requirements`] for the same
    /// `physical_device`, and with the `synchronization2` and
    /// `timelineSemaphore` features enabled (the decoder orders its pipelined
    /// submissions with timeline semaphores). The
    /// resulting context **borrows** `instance` and `device`: dropping it frees
    /// neither, so the caller must keep both alive for at least as long as the
    /// context and anything decoded with it.
    pub fn build_from_existing_decode(
        self,
        entry: ash::Entry,
        instance: ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
    ) -> Result<VideoContext> {
        VideoContext::from_existing_decode(
            self.required_decode_codecs,
            self.unified_image_layouts.unwrap_or(false),
            self.queues,
            entry,
            instance,
            physical_device,
            device,
        )
    }
}

/// Queue families, extensions and features a caller's device must provide for
/// pixelforge to work on it.
///
/// Returned by [`VideoContextBuilder::decode_device_requirements`] and
/// [`VideoContextBuilder::encode_device_requirements`]. Enable every extension
/// in [`Self::extensions`] and every feature set in [`Self::features`]:
/// pixelforge assumes all of them are on, since Vulkan cannot be asked
/// afterwards which were.
///
/// The instance may ask for any Vulkan version from 1.1 up. Pixelforge reaches
/// everything newer through the extensions listed here, so an application that
/// created its instance for 1.1 or 1.2 does not have to raise it.
#[derive(Debug, Clone)]
pub struct DeviceRequirements {
    /// Queue families pixelforge needs a queue created for. Merge these with
    /// your own (deduplicated) when building the device.
    pub queue_families: Vec<u32>,
    /// The family behind each role in [`Self::queue_families`]. An adopted
    /// context takes queue 0 of each unless told otherwise with
    /// [`VideoContextBuilder::with_transfer_queue`] and friends, which is
    /// where a caller that needs its own queue to itself points pixelforge at
    /// a spare one.
    pub queues: QueueRoles,
    /// Device extensions pixelforge needs enabled. Merge with your own.
    pub extensions: Vec<&'static std::ffi::CStr>,
    /// Device features pixelforge needs enabled. Merge with your own.
    pub features: DeviceFeatures,
    /// Whether this device can support unified image layouts for video, which
    /// is what lets decoded frames be sampled with no copy and no layout
    /// transition. When true, `VK_KHR_unified_image_layouts` is included in
    /// [`Self::extensions`]; enable it *and* chain
    /// `VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR` with both
    /// `unifiedImageLayouts` and `unifiedImageLayoutsVideo` set, then tell
    /// pixelforge you did with
    /// [`VideoContextBuilder::declare_unified_image_layouts`].
    ///
    /// Without that declaration the decoder falls back to copying every frame
    /// into a private image, which is correct but costs a full-frame GPU copy
    /// per frame.
    pub unified_image_layouts: bool,
    /// Whether this device supports `VK_KHR_internally_synchronized_queues`.
    ///
    /// A queue created with `VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR`
    /// (and the `internallySynchronizedQueues` feature enabled) may be
    /// submitted to from several threads at once, the driver doing the
    /// locking. That makes it safe to give pixelforge the same queue the
    /// caller uses, which is the way out on hardware with too few queues to
    /// spare one. Pixelforge needs to be told nothing: it is the caller's
    /// queue that has to be created this way. The extension is not in
    /// [`Self::extensions`], since only a caller sharing its queue needs it.
    pub internally_synchronized_queues: bool,
}

/// Device features, as booleans rather than a `pNext` chain.
///
/// A caller merging these into its own device creation may already chain
/// `VkPhysicalDeviceVulkan12Features` or `VkPhysicalDeviceVulkan13Features`,
/// and Vulkan forbids chaining those alongside the per-feature structs they
/// contain. So pixelforge says which bits it needs and leaves it to the caller
/// to set them in whichever structs it uses. The field names match the Vulkan
/// feature names.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeviceFeatures {
    /// `synchronization2`, from `VkPhysicalDeviceSynchronization2Features`.
    pub synchronization2: bool,
    /// `timelineSemaphore`, from `VkPhysicalDeviceTimelineSemaphoreFeatures`.
    pub timeline_semaphore: bool,
    /// `samplerYcbcrConversion`, from
    /// `VkPhysicalDeviceSamplerYcbcrConversionFeatures`.
    pub sampler_ycbcr_conversion: bool,
    /// `ycbcr2plane444Formats`, from
    /// `VkPhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT`.
    pub ycbcr_2plane_444_formats: bool,
    /// `videoEncodeAV1`, from `VkPhysicalDeviceVideoEncodeAV1FeaturesKHR`.
    pub video_encode_av1: bool,
    /// `videoEncodeRgbConversion`, from
    /// `VkPhysicalDeviceVideoEncodeRgbConversionFeaturesVALVE`.
    pub video_encode_rgb_conversion: bool,
}

/// The queue families pixelforge selects for decoding on a given device.
struct DecodeQueueFamilies {
    decode: u32,
    transfer: u32,
    compute: u32,
}

/// The families [`scan_queue_families`] found.
struct ScannedFamilies {
    /// The last family with the requested video flag, if any.
    video: Option<u32>,
    /// `timestampValidBits` of [`Self::video`]'s family.
    video_timestamp_valid_bits: u32,
    transfer: u32,
    compute: u32,
}

/// Find a family with the `video` flag, a transfer family and a compute family
/// on `physical_device`, the same way [`VideoContext::new`] does.
///
/// The transfer family is picked by preference rather than last-wins: a family
/// that also does video is the worst choice, since it contends with the
/// encode or decode work itself, then the graphics family, which the caller
/// most likely renders on, then compute, and best of all a dedicated transfer
/// engine.
fn scan_queue_families(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    video_flag: vk::QueueFlags,
) -> Result<ScannedFamilies> {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    let mut video = None;
    let mut video_timestamp_valid_bits = 0;
    let mut transfer = u32::MAX;
    let mut transfer_score = -1i32;
    let mut compute = u32::MAX;

    for (idx, props) in queue_families.iter().enumerate() {
        let idx = idx as u32;
        let flags = props.queue_flags;

        if flags.contains(video_flag) {
            video = Some(idx);
            video_timestamp_valid_bits = props.timestamp_valid_bits;
        }
        if flags.contains(vk::QueueFlags::TRANSFER) {
            let is_video = flags
                .intersects(vk::QueueFlags::VIDEO_ENCODE_KHR | vk::QueueFlags::VIDEO_DECODE_KHR);
            let score = if is_video {
                0
            } else if flags.contains(vk::QueueFlags::GRAPHICS) {
                1
            } else if flags.contains(vk::QueueFlags::COMPUTE) {
                2
            } else {
                3
            };
            if score > transfer_score {
                transfer_score = score;
                transfer = idx;
            }
        }
        if flags.contains(vk::QueueFlags::COMPUTE) && compute == u32::MAX {
            compute = idx;
        }
    }

    if transfer == u32::MAX {
        return Err(PixelForgeError::NoSuitableDevice(
            "Physical device has no transfer queue family".to_string(),
        ));
    }
    if compute == u32::MAX {
        return Err(PixelForgeError::NoSuitableDevice(
            "Physical device has no compute queue family".to_string(),
        ));
    }
    Ok(ScannedFamilies {
        video,
        video_timestamp_valid_bits,
        transfer,
        compute,
    })
}

/// Select the decode / transfer / compute queue families on `physical_device`,
/// failing if it cannot decode the required codecs.
///
/// Mirrors the selection [`VideoContext::new`] does inline, but scoped to the
/// decode path: a video-decode family, a transfer family (preferring a
/// dedicated engine over one that also does video — see the scoring), and any
/// compute family.
fn find_decode_queue_families(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    decode_codecs: &[Codec],
) -> Result<DecodeQueueFamilies> {
    let ScannedFamilies {
        video: decode,
        transfer,
        compute,
        ..
    } = scan_queue_families(instance, physical_device, vk::QueueFlags::VIDEO_DECODE_KHR)?;

    let decode = decode.ok_or_else(|| {
        PixelForgeError::NoSuitableDevice(
            "Physical device has no video decode queue family".to_string(),
        )
    })?;

    // Confirm the device actually decodes the codecs asked for.
    let available = query_decode_codecs(entry, instance, physical_device);
    for codec in decode_codecs {
        if !available.contains(codec) {
            return Err(PixelForgeError::CodecNotSupported(format!(
                "Physical device does not support decoding {:?}",
                codec
            )));
        }
    }

    Ok(DecodeQueueFamilies {
        decode,
        transfer,
        compute,
    })
}

/// Device extensions required to decode the given codecs.
/// Whether `physical_device` can support unified image layouts *for video*.
///
/// Both the extension and both of its feature bits are needed:
/// `unifiedImageLayouts` alone leaves video picture resources under the old
/// rules, which is exactly the case that matters here.
fn supports_unified_image_layouts(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> bool {
    let exts = match unsafe { instance.enumerate_device_extension_properties(physical_device) } {
        Ok(exts) => exts,
        Err(_) => return false,
    };
    let present = exts.iter().any(|ext| {
        let name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
        name == ash::khr::unified_image_layouts::NAME
    });
    if !present {
        return false;
    }
    let mut features = vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR::default();
    let mut query = vk::PhysicalDeviceFeatures2::default().push(&mut features);
    unsafe { instance.get_physical_device_features2(physical_device, &mut query) };
    features.unified_image_layouts != 0 && features.unified_image_layouts_video != 0
}

/// Whether `physical_device` supports `VK_KHR_internally_synchronized_queues`,
/// extension and feature both.
fn supports_internally_synchronized_queues(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> bool {
    let exts = match unsafe { instance.enumerate_device_extension_properties(physical_device) } {
        Ok(exts) => exts,
        Err(_) => return false,
    };
    let present = exts.iter().any(|ext| {
        let name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
        name == ash::khr::internally_synchronized_queues::NAME
    });
    if !present {
        return false;
    }
    let mut features = vk::PhysicalDeviceInternallySynchronizedQueuesFeaturesKHR::default();
    let mut query = vk::PhysicalDeviceFeatures2::default().push(&mut features);
    unsafe { instance.get_physical_device_features2(physical_device, &mut query) };
    features.internally_synchronized_queues != 0
}

/// Whether `physical_device` has the `videoEncodeRgbConversion` feature. The
/// caller checks for the extension first.
fn supports_rgb_conversion(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> bool {
    let mut features = vk::PhysicalDeviceVideoEncodeRgbConversionFeaturesVALVE::default();
    let mut query = vk::PhysicalDeviceFeatures2::default().push(&mut features);
    unsafe { instance.get_physical_device_features2(physical_device, &mut query) };
    features.video_encode_rgb_conversion != 0
}

fn decode_extension_names(decode_codecs: &[Codec]) -> Vec<&'static std::ffi::CStr> {
    // Synchronization2 and timeline semaphores are core in 1.3 and 1.2, but
    // are listed anyway: enabling a promoted extension is harmless, and on a
    // device created from an instance asking for less it is what makes them
    // available at all.
    let mut names = vec![
        ash::khr::video_queue::NAME,
        ash::khr::video_decode_queue::NAME,
        ash::khr::synchronization2::NAME,
        ash::khr::timeline_semaphore::NAME,
    ];
    if decode_codecs.contains(&Codec::H264) {
        names.push(ash::khr::video_decode_h264::NAME);
    }
    names
}

/// The decode codecs `physical_device` supports.
fn query_decode_codecs(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Vec<Codec> {
    let mut codecs = Vec::new();
    if VideoContext::check_h264_decode_support(entry, instance, physical_device) {
        codecs.push(Codec::H264);
    }
    codecs
}

/// Inner struct holding the actual Vulkan resources.
struct VideoContextInner {
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    /// `vkQueueSubmit2` and `vkCmdPipelineBarrier2` through their KHR names.
    ///
    /// The core entry points exist only when the device was created from an
    /// instance asking for Vulkan 1.3. An adopted device may come from an
    /// application that asked for less, where the core pointers are null and
    /// calling them crashes. The KHR names are valid on any device with
    /// `VK_KHR_synchronization2` enabled, which pixelforge requires either way.
    sync2: ash::khr::synchronization2::Device,
    video_encode_queue_family: Option<u32>,
    video_encode_timestamp_valid_bits: u32,
    video_encode_queue: Option<vk::Queue>,
    video_decode_queue_family: Option<u32>,
    video_decode_queue: Option<vk::Queue>,
    transfer_queue_family: u32,
    transfer_queue: vk::Queue,
    compute_queue_family: u32,
    compute_queue: vk::Queue,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device_properties: vk::PhysicalDeviceProperties,
    supported_encode_codecs: Vec<Codec>,
    supported_decode_codecs: Vec<Codec>,
    has_push_descriptor: bool,
    has_video_encode_rgb_conversion: bool,
    has_unified_image_layouts: bool,
    /// Whether this context created (and therefore must destroy) the device and
    /// instance. A context adopted from a caller's device borrows them and
    /// destroys neither.
    owns_device: bool,
    /// Validation message sink, present only when validation is enabled and the
    /// instance is ours. Destroyed before the instance it belongs to.
    debug_messenger: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl Drop for VideoContextInner {
    fn drop(&mut self) {
        // Only tear down the device/instance we created ourselves. An adopted
        // device is owned by the caller and outlives this context.
        if self.owns_device {
            unsafe {
                self.device.destroy_device(None);
                if let Some((debug_utils, messenger)) = &self.debug_messenger {
                    debug_utils.destroy_debug_utils_messenger(*messenger, None);
                }
                self.instance.destroy_instance(None);
            }
        }
    }
}

/// Holds the Vulkan context for video operations.
///
/// This type is cheaply cloneable - clones share the same underlying Vulkan resources.
#[derive(Clone)]
pub struct VideoContext {
    inner: std::sync::Arc<VideoContextInner>,
}

/// Provide access to inner fields through deref-like accessors.
impl VideoContext {
    /// Get the Vulkan instance.
    pub fn instance(&self) -> &ash::Instance {
        &self.inner.instance
    }

    /// Get the Vulkan device.
    pub fn device(&self) -> &ash::Device {
        &self.inner.device
    }

    pub(crate) fn sync2(&self) -> &ash::khr::synchronization2::Device {
        &self.inner.sync2
    }

    pub(crate) fn video_encode_queue_family(&self) -> Option<u32> {
        self.inner.video_encode_queue_family
    }

    pub(crate) fn video_encode_queue(&self) -> Option<vk::Queue> {
        self.inner.video_encode_queue
    }

    /// Whether the selected video encode queue family supports timestamp
    /// queries, i.e. reports a non-zero `timestampValidBits`. RADV's dedicated
    /// video encode queue reports 0, so `vkCmdWriteTimestamp` is illegal there
    /// (VUID-vkCmdWriteTimestamp-timestampValidBits-00829).
    pub(crate) fn encode_timestamps_supported(&self) -> bool {
        self.inner.video_encode_timestamp_valid_bits > 0
    }

    pub(crate) fn video_decode_queue_family(&self) -> Option<u32> {
        self.inner.video_decode_queue_family
    }

    pub(crate) fn video_decode_queue(&self) -> Option<vk::Queue> {
        self.inner.video_decode_queue
    }

    /// Get the transfer queue family index.
    pub fn transfer_queue_family(&self) -> u32 {
        self.inner.transfer_queue_family
    }

    /// Get the transfer queue.
    pub fn transfer_queue(&self) -> vk::Queue {
        self.inner.transfer_queue
    }

    /// Get the compute queue family index.
    pub fn compute_queue_family(&self) -> u32 {
        self.inner.compute_queue_family
    }

    /// Get the compute queue.
    pub fn compute_queue(&self) -> vk::Queue {
        self.inner.compute_queue
    }

    pub(crate) fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.inner.memory_properties
    }

    /// Get the physical device handle.
    ///
    /// This can be used to query device capabilities and properties.
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.inner.physical_device
    }

    /// Get the physical device properties.
    ///
    /// Contains information about the GPU such as device name, limits, and supported Vulkan version.
    pub fn device_properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.inner.device_properties
    }

    /// Whether `VK_KHR_push_descriptor` is enabled, which the colour converter
    /// needs.
    pub fn has_push_descriptor(&self) -> bool {
        self.inner.has_push_descriptor
    }

    /// Whether `VK_VALVE_video_encode_rgb_conversion` is enabled, extension
    /// and feature both: encoders can then take RGB input and convert it
    /// themselves, see
    /// [`EncodeConfig::with_rgb_input`](crate::EncodeConfig::with_rgb_input).
    pub fn has_video_encode_rgb_conversion(&self) -> bool {
        self.inner.has_video_encode_rgb_conversion
    }

    /// Whether `VK_KHR_unified_image_layouts` is enabled, with its video bit.
    ///
    /// When it is, `VK_IMAGE_LAYOUT_GENERAL` is valid for every use including
    /// video picture resources, so a decoded picture never has to change layout
    /// to be sampled, copied from, or used as a reference.
    pub fn has_unified_image_layouts(&self) -> bool {
        self.inner.has_unified_image_layouts
    }
}

impl VideoContext {
    fn new(builder: VideoContextBuilder) -> Result<Self> {
        // Load Vulkan.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| PixelForgeError::InstanceCreation(e.to_string()))?;

        // Create instance.
        let app_name = CString::new(builder.app_name.clone()).expect("Invalid app name");
        let engine_name = CString::new("PixelForge").expect("Invalid engine name");

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(
                0,
                builder.app_version.0,
                builder.app_version.1,
                builder.app_version.2,
            ))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_3);

        let mut enable_validation = builder.enable_validation;
        if enable_validation {
            let available_layers = unsafe { entry.enumerate_instance_layer_properties() }
                .map_err(|e| PixelForgeError::InstanceCreation(e.to_string()))?;
            let validation_layer_name = c"VK_LAYER_KHRONOS_validation";
            let has_validation_layer = available_layers.iter().any(|layer| {
                let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                name == validation_layer_name
            });
            if !has_validation_layer {
                warn!("Validation layer requested but not available");
                enable_validation = false;
            }
        }

        let mut layer_names: Vec<*const c_char> = Vec::new();
        let validation_layer = c"VK_LAYER_KHRONOS_validation";
        if enable_validation {
            layer_names.push(validation_layer.as_ptr());
        }

        // Enable VK_EXT_validation_features if validation is enabled to allow configuration.
        let mut instance_extensions: Vec<*const c_char> = Vec::new();
        if enable_validation {
            let validation_layer_name = c"VK_LAYER_KHRONOS_validation";
            let available_exts = unsafe {
                entry.enumerate_instance_extension_properties(Some(validation_layer_name))
            }
            .map_err(|e| PixelForgeError::InstanceCreation(e.to_string()))?;
            let validation_features_name = c"VK_EXT_validation_features";
            let has_validation_features = available_exts.iter().any(|ext| {
                let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                name == validation_features_name
            });
            if has_validation_features {
                instance_extensions.push(validation_features_name.as_ptr());
            } else {
                warn!("VK_EXT_validation_features requested but not available");
            }
        }

        // VK_EXT_debug_utils carries the messenger the layer reports through.
        let mut has_debug_utils = false;
        if enable_validation {
            let available_exts = unsafe { entry.enumerate_instance_extension_properties(None) }
                .map_err(|e| PixelForgeError::InstanceCreation(e.to_string()))?;
            has_debug_utils = available_exts.iter().any(|ext| {
                let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                name == ash::ext::debug_utils::NAME
            });
            if has_debug_utils {
                instance_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
            } else {
                warn!(
                    "VK_EXT_debug_utils not available; validation layer messages will not be \
                     reported"
                );
            }
        }

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&instance_extensions);

        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|e| PixelForgeError::InstanceCreation(e.to_string()))?;

        info!("Created Vulkan instance");

        let debug_messenger = if has_debug_utils {
            let debug_utils = ash::ext::debug_utils::Instance::load(&entry, &instance);
            let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_utils_callback));
            let messenger = unsafe { debug_utils.create_debug_utils_messenger(&create_info, None) }
                .map_err(|e| PixelForgeError::InstanceCreation(e.to_string()))?;
            info!("Validation layer messages are routed to tracing");
            Some((debug_utils, messenger))
        } else {
            None
        };

        // Find physical device with video support.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| PixelForgeError::NoSuitableDevice(e.to_string()))?;

        let mut selected_device = None;
        let mut selected_device_exts = None;
        let mut video_encode_queue_family = None;
        let mut video_encode_timestamp_valid_bits = 0u32;
        let mut video_decode_queue_family = None;
        let mut transfer_queue_family = u32::MAX;
        let mut compute_queue_family = u32::MAX;
        let mut supported_encode_codecs = Vec::new();
        let mut supported_decode_codecs = Vec::new();

        let has_extension =
            |extensions: &[vk::ExtensionProperties], name: &std::ffi::CStr| -> bool {
                extensions.iter().any(|ext| {
                    let ext_name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
                    ext_name == name
                })
            };

        for physical_device in physical_devices {
            let props = unsafe { instance.get_physical_device_properties(physical_device) };
            let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                .to_string_lossy()
                .to_string();
            debug!("Checking device: {}", device_name);

            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

            // Find queue families.
            let mut encode_queue = None;
            let mut encode_ts_bits = 0u32;
            let mut decode_queue = None;
            let mut transfer_q = u32::MAX;
            let mut transfer_score = -1i32;
            let mut compute_q = u32::MAX;

            for (idx, props) in queue_families.iter().enumerate() {
                debug!(
                    "Queue family {}: flags={:?}, count={}",
                    idx, props.queue_flags, props.queue_count
                );
                let flags = props.queue_flags;

                // Check for video encode queue.
                if flags.contains(vk::QueueFlags::VIDEO_ENCODE_KHR) {
                    encode_queue = Some(idx as u32);
                    encode_ts_bits = props.timestamp_valid_bits;
                    debug!("Found video encode queue at family {}", idx);
                }

                // Check for video decode queue.
                if flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR) {
                    decode_queue = Some(idx as u32);
                    debug!("Found video decode queue at family {}", idx);
                }

                // Pick the transfer queue by preference, not by last-wins. The
                // readback copy runs here, so a family that also does video
                // (encode or decode) is the worst choice: it contends with our
                // own encode/decode work and, on some drivers, shares a single
                // VkQueue that would then need external synchronization. Among
                // the rest, prefer the least capable family: a dedicated DMA
                // engine first, then compute+transfer, and only then the
                // graphics queue (which a downstream app typically drives for
                // rendering and present).
                if flags.contains(vk::QueueFlags::TRANSFER) {
                    let is_video = flags.intersects(
                        vk::QueueFlags::VIDEO_ENCODE_KHR | vk::QueueFlags::VIDEO_DECODE_KHR,
                    );
                    let score = if is_video {
                        0 // shares a video engine; last resort
                    } else if flags.contains(vk::QueueFlags::GRAPHICS) {
                        1 // the universal graphics queue
                    } else if flags.contains(vk::QueueFlags::COMPUTE) {
                        2 // compute + transfer
                    } else {
                        3 // dedicated transfer engine
                    };
                    if score > transfer_score {
                        transfer_score = score;
                        transfer_q = idx as u32;
                    }
                }

                // Check for compute queue (prefer dedicated compute, otherwise graphics+compute).
                if flags.contains(vk::QueueFlags::COMPUTE) && compute_q == u32::MAX {
                    compute_q = idx as u32;
                    debug!("Found compute queue at family {}", idx);
                }
            }

            // Get list of available device extensions
            let available_extensions = match unsafe {
                instance.enumerate_device_extension_properties(physical_device)
            } {
                Ok(exts) => exts,
                Err(e) => {
                    warn!(
                        "Failed to enumerate device extension properties for {}: {}. Skipping device.",
                        device_name, e
                    );
                    continue;
                }
            };

            // Check codec support for encoding.
            let mut encode_codecs = Vec::new();
            if let Some(eq) = encode_queue {
                // Only check codec support if the extension exists
                if has_extension(&available_extensions, ash::khr::video_encode_h264::NAME)
                    && Self::check_h264_encode_support(&entry, &instance, physical_device, eq)
                {
                    encode_codecs.push(Codec::H264);
                    debug!("Device {} supports H.264 encode", device_name);
                }
                if has_extension(&available_extensions, ash::khr::video_encode_h265::NAME)
                    && Self::check_h265_encode_support(&entry, &instance, physical_device, eq)
                {
                    encode_codecs.push(Codec::H265);
                    debug!("Device {} supports H.265 encode", device_name);
                }
                if has_extension(&available_extensions, ash::khr::video_encode_av1::NAME)
                    && Self::check_av1_encode_support(&entry, &instance, physical_device, eq)
                {
                    encode_codecs.push(Codec::AV1);
                    debug!("Device {} supports AV1 encode", device_name);
                }
            }

            // Check codec support for decoding.
            let mut decode_codecs = Vec::new();
            if decode_queue.is_some() {
                let available_extensions =
                    unsafe { instance.enumerate_device_extension_properties(physical_device) }
                        .unwrap_or_default();
                let has_extension = |name: &std::ffi::CStr| -> bool {
                    available_extensions.iter().any(|ext| {
                        let ext_name =
                            unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
                        ext_name == name
                    })
                };

                if has_extension(ash::khr::video_decode_h264::NAME)
                    && Self::check_h264_decode_support(&entry, &instance, physical_device)
                {
                    decode_codecs.push(Codec::H264);
                    debug!("Device {} supports H.264 decode", device_name);
                }
            }

            // Check if all required encode/decode codecs are supported.
            let encode_supported = builder
                .required_encode_codecs
                .iter()
                .all(|codec| encode_codecs.contains(codec));
            let decode_supported = builder
                .required_decode_codecs
                .iter()
                .all(|codec| decode_codecs.contains(codec));

            // We need at least one video queue, and compute support.
            let has_video_support = encode_queue.is_some() || decode_queue.is_some();
            let has_compute_support = compute_q != u32::MAX;

            if has_video_support && encode_supported && decode_supported && has_compute_support {
                selected_device = Some(physical_device);
                selected_device_exts = Some(available_extensions);
                video_encode_queue_family = encode_queue;
                video_encode_timestamp_valid_bits = encode_ts_bits;
                video_decode_queue_family = decode_queue;
                transfer_queue_family = if transfer_q != u32::MAX {
                    transfer_q
                } else {
                    encode_queue.unwrap_or(0)
                };
                compute_queue_family = compute_q;
                supported_encode_codecs = encode_codecs;
                supported_decode_codecs = decode_codecs;
                info!("Selected device: {}", device_name);
                break;
            } else {
                warn!(
                    "Device {} skipped: video_support={}, encode_supported={}, decode_supported={}, compute_support={}",
                    device_name,
                    has_video_support,
                    encode_supported,
                    decode_supported,
                    has_compute_support
                );
                if !has_video_support {
                    warn!("  - No queue with VIDEO_ENCODE_KHR flag found");
                }
                if !encode_supported {
                    warn!(
                        "  - Required encode codecs not supported: {:?}",
                        builder.required_encode_codecs
                    );
                    warn!("  - Available encode codecs: {:?}", encode_codecs);
                }
                if !decode_supported {
                    warn!(
                        "  - Required decode codecs not supported: {:?}",
                        builder.required_decode_codecs
                    );
                    warn!("  - Available decode codecs: {:?}", decode_codecs);
                }
            }
        }

        let physical_device = selected_device.ok_or_else(|| {
            PixelForgeError::NoSuitableDevice(
                "No device with required video support found. Ensure your GPU drivers support Vulkan Video extensions (VK_KHR_video_queue, VK_KHR_video_encode_queue, etc.).".to_string(),
            )
        })?;

        // Get device properties and memory properties.
        let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // Create logical device with video extensions.
        let queue_priorities = [1.0f32];

        // Build queue create infos - collect unique families.
        let mut unique_families = Vec::new();
        if let Some(encode_family) = video_encode_queue_family {
            unique_families.push(encode_family);
        }
        if let Some(decode_family) = video_decode_queue_family
            && !unique_families.contains(&decode_family)
        {
            unique_families.push(decode_family);
        }
        if !unique_families.contains(&transfer_queue_family) {
            unique_families.push(transfer_queue_family);
        }
        if !unique_families.contains(&compute_queue_family) {
            unique_families.push(compute_queue_family);
        }

        let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = unique_families
            .iter()
            .map(|&family| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&queue_priorities)
            })
            .collect();

        // Required device extensions for video encoding.
        let mut extension_names = vec![
            ash::khr::video_queue::NAME.as_ptr(),
            ash::khr::synchronization2::NAME.as_ptr(),
        ];

        // External memory extensions for DMA-BUF support (optional, enabled with "dmabuf" feature).
        #[cfg(feature = "dmabuf")]
        {
            extension_names.push(ash::khr::external_memory::NAME.as_ptr());
            extension_names.push(ash::khr::external_memory_fd::NAME.as_ptr());
            extension_names.push(ash::ext::external_memory_dma_buf::NAME.as_ptr());
            extension_names.push(ash::ext::image_drm_format_modifier::NAME.as_ptr());
        }

        let mut push_ext = |name: *const std::ffi::c_char| {
            if !extension_names.contains(&name) {
                extension_names.push(name);
            }
        };
        if video_encode_queue_family.is_some() {
            push_ext(ash::khr::video_encode_queue::NAME.as_ptr());

            if supported_encode_codecs.contains(&Codec::H264) {
                push_ext(ash::khr::video_encode_h264::NAME.as_ptr());
            }
            if supported_encode_codecs.contains(&Codec::H265) {
                push_ext(ash::khr::video_encode_h265::NAME.as_ptr());
            }
            if supported_encode_codecs.contains(&Codec::AV1) {
                push_ext(ash::khr::video_encode_av1::NAME.as_ptr());
            }
        }
        if video_decode_queue_family.is_some() && !supported_decode_codecs.is_empty() {
            push_ext(ash::khr::video_decode_queue::NAME.as_ptr());

            if supported_decode_codecs.contains(&Codec::H264) {
                push_ext(ash::khr::video_decode_h264::NAME.as_ptr());
            }
        }

        // The colour converter binds its two descriptors with push descriptors.
        let has_push_descriptor = selected_device_exts
            .as_ref()
            .is_some_and(|exts| has_extension(exts, ash::khr::push_descriptor::NAME));
        if has_push_descriptor {
            push_ext(ash::khr::push_descriptor::NAME.as_ptr());
        } else {
            warn!("VK_KHR_push_descriptor not available; the colour converter will be unavailable");
        }

        // Lets an encoder take RGB input and convert it to YUV itself, so a
        // plain matrix conversion needs no shader at all. Extension and feature
        // both, or the profile struct is parsed and ignored.
        let has_rgb_conversion_ext = video_encode_queue_family.is_some()
            && selected_device_exts.as_ref().is_some_and(|exts| {
                has_extension(exts, ash::valve::video_encode_rgb_conversion::NAME)
            });
        let has_rgb_conversion =
            has_rgb_conversion_ext && supports_rgb_conversion(&instance, physical_device);
        let mut rgb_conversion_features =
            vk::PhysicalDeviceVideoEncodeRgbConversionFeaturesVALVE::default()
                .video_encode_rgb_conversion(true);
        if has_rgb_conversion {
            push_ext(ash::valve::video_encode_rgb_conversion::NAME.as_ptr());
            debug!("Video encode RGB conversion available");
        }

        // Enable synchronization2 feature.
        let mut sync2_features =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);

        // VK_KHR_unified_image_layouts: makes VK_IMAGE_LAYOUT_GENERAL valid
        // everywhere, and its `video` bit extends that to video picture
        // resources. Both bits are what let a decoded picture be sampled
        // without ever being transitioned out of the layout the decoder needs.
        let has_unified_layouts_ext = if let Some(device_exts) = &selected_device_exts {
            has_extension(device_exts, ash::khr::unified_image_layouts::NAME)
        } else {
            false
        };
        let mut supported_unified_layouts =
            vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR::default();
        if has_unified_layouts_ext {
            let mut query =
                vk::PhysicalDeviceFeatures2::default().push(&mut supported_unified_layouts);
            unsafe {
                instance.get_physical_device_features2(physical_device, &mut query);
            }
        }
        let has_unified_layouts = builder.unified_image_layouts != Some(false)
            && has_unified_layouts_ext
            && supported_unified_layouts.unified_image_layouts != 0
            && supported_unified_layouts.unified_image_layouts_video != 0;
        if has_unified_layouts {
            debug!("Unified image layouts enabled: decoded frames need no copy to be read");
        } else {
            debug!(
                "Unified image layouts unavailable (extension {}, unifiedImageLayouts {}, \
                 unifiedImageLayoutsVideo {}); decoded frames will be copied out",
                has_unified_layouts_ext,
                supported_unified_layouts.unified_image_layouts,
                supported_unified_layouts.unified_image_layouts_video
            );
        }
        let mut unified_layout_features =
            vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR::default()
                .unified_image_layouts(true)
                .unified_image_layouts_video(true);
        if has_unified_layouts {
            push_ext(ash::khr::unified_image_layouts::NAME.as_ptr());
        }

        let mut supported_timeline_features =
            vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
        let mut timeline_feature_query =
            vk::PhysicalDeviceFeatures2::default().push(&mut supported_timeline_features);
        unsafe {
            instance.get_physical_device_features2(physical_device, &mut timeline_feature_query);
        }
        if supported_timeline_features.timeline_semaphore == 0 {
            return Err(PixelForgeError::NoSuitableDevice(
                "Timeline semaphores are required for pipelined video encode synchronization"
                    .to_string(),
            ));
        }
        let mut timeline_features =
            vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);

        // Enable sampler YCbCr conversion feature (required for YUV image views with SAMPLED flag).
        let mut supported_ycbcr = vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default();
        let mut ycbcr_query = vk::PhysicalDeviceFeatures2::default().push(&mut supported_ycbcr);
        unsafe {
            instance.get_physical_device_features2(physical_device, &mut ycbcr_query);
        }
        let has_ycbcr_conversion = supported_ycbcr.sampler_ycbcr_conversion != 0;
        let mut ycbcr_features = vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default()
            .sampler_ycbcr_conversion(true);

        let has_ycbcr_2plane_444_ext = if let Some(device_exts) = selected_device_exts {
            has_extension(&device_exts, ash::ext::ycbcr_2plane_444_formats::NAME)
        } else {
            false
        };

        let mut ycbcr_2plane_444_features =
            vk::PhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT::default()
                .ycbcr2plane444_formats(true);

        if has_ycbcr_2plane_444_ext {
            // Add the 2-plane 444 formats extension.
            push_ext(ash::ext::ycbcr_2plane_444_formats::NAME.as_ptr());
        }

        // Enable AV1 video encode feature only if AV1 is supported.
        // Only include AV1 features in the pNext chain when AV1 is actually supported,
        // to avoid chaining unknown feature structs on devices without AV1.
        let mut av1_encode_features =
            vk::PhysicalDeviceVideoEncodeAV1FeaturesKHR::default().video_encode_av1(true);

        // Log all extensions being enabled
        debug!("Enabling {} device extensions:", extension_names.len());
        for ext_name_ptr in &extension_names {
            let ext_name = unsafe { std::ffi::CStr::from_ptr(*ext_name_ptr) };
            debug!("  - {}", ext_name.to_string_lossy());
        }

        let mut device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extension_names)
            .push(&mut sync2_features)
            .push(&mut timeline_features);

        if has_unified_layouts {
            device_create_info = device_create_info.push(&mut unified_layout_features);
        }

        if supported_encode_codecs.contains(&Codec::AV1) {
            device_create_info = device_create_info.push(&mut av1_encode_features);
        }

        if has_ycbcr_conversion {
            device_create_info = device_create_info.push(&mut ycbcr_features);
        }
        if has_rgb_conversion {
            device_create_info = device_create_info.push(&mut rgb_conversion_features);
        }
        // Enable YCbCr 2-plane 444 formats feature (required for YUV444 encoding with NVIDIA).
        // Its extension is enabled above whenever present, so the feature has
        // to follow it unconditionally: an extension without its feature bit
        // leaves the 4:4:4 formats unusable.
        if has_ycbcr_2plane_444_ext {
            device_create_info = device_create_info.push(&mut ycbcr_2plane_444_features);
        }

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
            .map_err(|e| PixelForgeError::DeviceCreation(e.to_string()))?;

        // Get queues.
        let video_encode_queue =
            video_encode_queue_family.map(|family| unsafe { device.get_device_queue(family, 0) });
        let video_decode_queue =
            video_decode_queue_family.map(|family| unsafe { device.get_device_queue(family, 0) });
        let transfer_queue = unsafe { device.get_device_queue(transfer_queue_family, 0) };
        let compute_queue = unsafe { device.get_device_queue(compute_queue_family, 0) };

        if let Some(family) = video_encode_queue_family {
            info!("Video encode queue family: {}", family);
        }
        if let Some(family) = video_decode_queue_family {
            info!("Video decode queue family: {}", family);
        }
        info!("Transfer queue family: {}", transfer_queue_family);
        info!("Compute queue family: {}", compute_queue_family);
        info!("Created Vulkan device with video support");

        Ok(Self {
            inner: std::sync::Arc::new(VideoContextInner {
                entry,
                sync2: ash::khr::synchronization2::Device::load(&instance, &device),
                instance,
                physical_device,
                device,
                video_encode_queue_family,
                video_encode_timestamp_valid_bits,
                video_encode_queue,
                video_decode_queue_family,
                video_decode_queue,
                transfer_queue_family,
                transfer_queue,
                compute_queue_family,
                compute_queue,
                memory_properties,
                device_properties,
                supported_encode_codecs,
                supported_decode_codecs,
                has_push_descriptor,
                has_video_encode_rgb_conversion: has_rgb_conversion,
                has_unified_image_layouts: has_unified_layouts,
                owns_device: true,
                debug_messenger,
            }),
        })
    }

    /// Adopt a caller-created device for decoding.
    ///
    /// See [`VideoContextBuilder::build_from_existing_decode`], the public entry
    /// point. The returned context borrows `instance` and `device` and destroys
    /// neither on drop.
    fn from_existing_decode(
        required_decode_codecs: Vec<Codec>,
        declared_unified_image_layouts: bool,
        queue_overrides: QueueOverrides,
        entry: ash::Entry,
        instance: ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
    ) -> Result<VideoContext> {
        let families = find_decode_queue_families(
            &entry,
            &instance,
            physical_device,
            &required_decode_codecs,
        )?;

        let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let decode = queues::resolve(
            &instance,
            physical_device,
            "decode",
            queue_overrides.decode,
            families.decode,
            vk::QueueFlags::VIDEO_DECODE_KHR,
        )?;
        let transfer = queues::resolve(
            &instance,
            physical_device,
            "transfer",
            queue_overrides.transfer,
            families.transfer,
            vk::QueueFlags::TRANSFER,
        )?;
        let compute = queues::resolve(
            &instance,
            physical_device,
            "compute",
            queue_overrides.compute,
            families.compute,
            vk::QueueFlags::COMPUTE,
        )?;

        let video_decode_queue = unsafe { device.get_device_queue(decode.family, decode.index) };
        let transfer_queue = unsafe { device.get_device_queue(transfer.family, transfer.index) };
        let compute_queue = unsafe { device.get_device_queue(compute.family, compute.index) };

        let supported_decode_codecs = query_decode_codecs(&entry, &instance, physical_device);

        // The declaration is a promise about what the caller enabled, which
        // cannot be verified. That the hardware could support it can be, so a
        // declaration that is impossible fails here rather than in the driver.
        if declared_unified_image_layouts
            && !supports_unified_image_layouts(&instance, physical_device)
        {
            return Err(PixelForgeError::NoSuitableDevice(
                "unified image layouts were declared, but this device does not support \
                 VK_KHR_unified_image_layouts with both unifiedImageLayouts and \
                 unifiedImageLayoutsVideo"
                    .to_string(),
            ));
        }

        info!(
            "Adopted caller device for decode: decode queue {:?}, transfer queue {:?}, compute queue {:?}",
            decode, transfer, compute
        );

        Ok(VideoContext {
            inner: std::sync::Arc::new(VideoContextInner {
                entry,
                sync2: ash::khr::synchronization2::Device::load(&instance, &device),
                instance,
                physical_device,
                device,
                video_encode_queue_family: None,
                video_encode_timestamp_valid_bits: 0u32,
                video_encode_queue: None,
                video_decode_queue_family: Some(decode.family),
                video_decode_queue: Some(video_decode_queue),
                transfer_queue_family: transfer.family,
                transfer_queue,
                compute_queue_family: compute.family,
                compute_queue,
                memory_properties,
                device_properties,
                supported_encode_codecs: Vec::new(),
                supported_decode_codecs,
                has_push_descriptor: false,
                has_video_encode_rgb_conversion: false,
                has_unified_image_layouts: declared_unified_image_layouts,
                owns_device: false,
                // The caller owns the instance; reporting is theirs to set up.
                debug_messenger: None,
            }),
        })
    }

    fn check_h264_encode_support(
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        _queue_family: u32,
    ) -> bool {
        // Create video queue instance extension.
        let video_queue = ash::khr::video_queue::Instance::load(entry, instance);

        // Create H.264 encode profile info (must stay alive during the call)
        let mut h264_profile = vk::VideoEncodeH264ProfileInfoKHR::default().std_profile_idc(
            ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN,
        );

        // Create video profile info for H.264 encode with typical 8-bit 4:2:0.
        let profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push(&mut h264_profile);

        // Create capabilities structures.
        let mut h264_capabilities = vk::VideoEncodeH264CapabilitiesKHR::default();
        let mut encode_capabilities = vk::VideoEncodeCapabilitiesKHR::default();
        let mut capabilities = vk::VideoCapabilitiesKHR::default()
            .push(&mut h264_capabilities)
            .push(&mut encode_capabilities);

        // Query capabilities.
        let result = unsafe {
            (video_queue.fp().get_physical_device_video_capabilities_khr)(
                physical_device,
                &profile_info,
                &mut capabilities,
            )
        };

        match result {
            vk::Result::SUCCESS => {
                debug!(
                    "H.264 encode supported: max {}x{}, {} DPB slots",
                    capabilities.max_coded_extent.width,
                    capabilities.max_coded_extent.height,
                    capabilities.max_dpb_slots
                );
                true
            }
            vk::Result::ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR => {
                debug!("H.264 encode not supported on this device");
                false
            }
            err => {
                warn!("Failed to query H.264 encode capabilities: {:?}", err);
                false
            }
        }
    }

    fn check_h265_encode_support(
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        _queue_family: u32,
    ) -> bool {
        // Create video queue instance extension.
        let video_queue = ash::khr::video_queue::Instance::load(entry, instance);

        // Create H.265 encode profile info (must stay alive during the call)
        let mut h265_profile = vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(
            ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN,
        );

        // Create video profile info for H.265 encode with typical 8-bit 4:2:0.
        let profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_H265)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push(&mut h265_profile);

        // Create capabilities structures.
        let mut h265_capabilities = vk::VideoEncodeH265CapabilitiesKHR::default();
        let mut encode_capabilities = vk::VideoEncodeCapabilitiesKHR::default();
        let mut capabilities = vk::VideoCapabilitiesKHR::default()
            .push(&mut h265_capabilities)
            .push(&mut encode_capabilities);

        // Query capabilities.
        let result = unsafe {
            (video_queue.fp().get_physical_device_video_capabilities_khr)(
                physical_device,
                &profile_info,
                &mut capabilities,
            )
        };

        match result {
            vk::Result::SUCCESS => {
                debug!(
                    "H.265 encode supported: max {}x{}, {} DPB slots",
                    capabilities.max_coded_extent.width,
                    capabilities.max_coded_extent.height,
                    capabilities.max_dpb_slots
                );
                true
            }
            vk::Result::ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR => {
                debug!("H.265 encode not supported on this device");
                false
            }
            err => {
                warn!("Failed to query H.265 encode capabilities: {:?}", err);
                false
            }
        }
    }

    fn check_av1_encode_support(
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        _queue_family: u32,
    ) -> bool {
        // Create video queue instance extension.
        let video_queue = ash::khr::video_queue::Instance::load(entry, instance);

        // Create AV1 encode profile info (must stay alive during the call)
        let mut av1_profile = vk::VideoEncodeAV1ProfileInfoKHR::default()
            .std_profile(ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN);

        // Create video profile info for AV1 encode with typical 8-bit 4:2:0.
        let profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::ENCODE_AV1)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push(&mut av1_profile);

        // Create capabilities structures.
        let mut av1_capabilities = vk::VideoEncodeAV1CapabilitiesKHR::default();
        let mut encode_capabilities = vk::VideoEncodeCapabilitiesKHR::default();
        let mut capabilities = vk::VideoCapabilitiesKHR::default()
            .push(&mut av1_capabilities)
            .push(&mut encode_capabilities);

        // Query capabilities.
        let result = unsafe {
            (video_queue.fp().get_physical_device_video_capabilities_khr)(
                physical_device,
                &profile_info,
                &mut capabilities,
            )
        };

        match result {
            vk::Result::SUCCESS => {
                debug!(
                    "AV1 encode supported: max {}x{}, {} DPB slots",
                    capabilities.max_coded_extent.width,
                    capabilities.max_coded_extent.height,
                    capabilities.max_dpb_slots
                );
                true
            }
            vk::Result::ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR => {
                debug!("AV1 encode not supported on this device");
                false
            }
            err => {
                warn!("Failed to query AV1 encode capabilities: {:?}", err);
                false
            }
        }
    }

    /// Check if a codec is supported for encoding.
    fn check_h264_decode_support(
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> bool {
        let video_queue = ash::khr::video_queue::Instance::load(entry, instance);

        // H.264 decode profile: High profile, progressive, 8-bit 4:2:0.
        let mut h264_profile = vk::VideoDecodeH264ProfileInfoKHR::default()
            .std_profile_idc(
                ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH,
            )
            .picture_layout(vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE);

        let profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_H264)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push(&mut h264_profile);

        let mut h264_capabilities = vk::VideoDecodeH264CapabilitiesKHR::default();
        let mut decode_capabilities = vk::VideoDecodeCapabilitiesKHR::default();
        let mut capabilities = vk::VideoCapabilitiesKHR::default()
            .push(&mut h264_capabilities)
            .push(&mut decode_capabilities);

        let result = unsafe {
            (video_queue.fp().get_physical_device_video_capabilities_khr)(
                physical_device,
                &profile_info,
                &mut capabilities,
            )
        };

        match result {
            vk::Result::SUCCESS => {
                debug!(
                    "H.264 decode supported: max {}x{}, {} DPB slots",
                    capabilities.max_coded_extent.width,
                    capabilities.max_coded_extent.height,
                    capabilities.max_dpb_slots
                );
                true
            }
            vk::Result::ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR => {
                debug!("H.264 decode not supported on this device");
                false
            }
            _ => {
                warn!("Failed to query H.264 decode capabilities: {:?}", result);
                false
            }
        }
    }

    pub fn supports_encode(&self, codec: Codec) -> bool {
        self.inner.supported_encode_codecs.contains(&codec)
    }

    /// Check if the selected device supports decoding the given codec.
    pub fn supports_decode(&self, codec: Codec) -> bool {
        self.inner.supported_decode_codecs.contains(&codec)
    }

    /// Get the Vulkan entry point.
    pub fn entry(&self) -> &ash::Entry {
        &self.inner.entry
    }

    /// Find a memory type that satisfies the requirements.
    pub fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        (0..self.inner.memory_properties.memory_type_count).find(|&i| {
            (type_filter & (1 << i)) != 0
                && self.inner.memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
        })
    }
}
