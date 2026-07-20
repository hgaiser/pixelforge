//! Vulkan context and initialization for video encoding.
use crate::encoder::Codec;
use crate::error::{PixelForgeError, Result};
use ash::vk;
use ash::vk::TaggedStructure;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use tracing::{debug, info, warn};

/// Builder for creating a VideoContext.
#[must_use]
pub struct VideoContextBuilder {
    app_name: String,
    app_version: (u32, u32, u32),
    enable_validation: bool,
    required_encode_codecs: Vec<Codec>,
    preferred_drm_render_node: Option<std::path::PathBuf>,
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
            preferred_drm_render_node: None,
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

    /// Prefer the physical device backing the given DRM render node
    /// (e.g. `/dev/dri/renderD128`).
    ///
    /// Device selection picks the encode-capable Vulkan device whose DRM
    /// render major/minor (via `VK_EXT_physical_device_drm`) matches this node,
    /// so the encoder runs on the same physical GPU as the compositor that
    /// produced the DMA-BUFs. Without this, importing a buffer allocated on one
    /// GPU into an encoder on another corrupts the image, and identical GPUs
    /// can't be disambiguated by vendor:device id. Falls back to the first
    /// suitable device when unset or unmatched. No-op on non-Linux.
    pub fn drm_render_node(mut self, node: Option<std::path::PathBuf>) -> Self {
        self.preferred_drm_render_node = node;
        self
    }

    /// Build the VideoContext.
    pub fn build(self) -> Result<VideoContext> {
        VideoContext::new(self)
    }
}

/// Inner struct holding the actual Vulkan resources.
struct VideoContextInner {
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    video_encode_queue_family: Option<u32>,
    video_encode_queue: Option<vk::Queue>,
    transfer_queue_family: u32,
    transfer_queue: vk::Queue,
    compute_queue_family: u32,
    compute_queue: vk::Queue,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device_properties: vk::PhysicalDeviceProperties,
    supported_encode_codecs: Vec<Codec>,
    has_descriptor_buffer: bool,
}

impl Drop for VideoContextInner {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
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

    pub(crate) fn video_encode_queue_family(&self) -> Option<u32> {
        self.inner.video_encode_queue_family
    }

    pub(crate) fn video_encode_queue(&self) -> Option<vk::Queue> {
        self.inner.video_encode_queue
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

    /// Returns true if `VK_EXT_descriptor_buffer` is available and enabled.
    pub fn has_descriptor_buffer(&self) -> bool {
        self.inner.has_descriptor_buffer
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

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&instance_extensions);

        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|e| PixelForgeError::InstanceCreation(e.to_string()))?;

        info!("Created Vulkan instance");

        // Find physical device with video support.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| PixelForgeError::NoSuitableDevice(e.to_string()))?;

        // Resolve the preferred DRM render node to (major, minor). Used to pin
        // the encoder to the same physical GPU as the compositor, since importing
        // a DMA-BUF allocated on one GPU into an encoder on another corrupts the
        // image and vendor:device filtering can't disambiguate identical GPUs.
        let preferred_drm = builder.preferred_drm_render_node.as_deref().and_then(|path| {
            match drm_render_major_minor(path) {
                Some(mm) => {
                    info!(
                        "Encoder will prefer the GPU backing {} (drm {}:{})",
                        path.display(),
                        mm.0,
                        mm.1
                    );
                    Some(mm)
                },
                None => {
                    warn!(
                        "Could not resolve DRM major/minor for {}; encoder will pick the first suitable GPU",
                        path.display()
                    );
                    None
                },
            }
        });

        struct DeviceCandidate {
            physical_device: vk::PhysicalDevice,
            device_name: String,
            video_encode_queue_family: Option<u32>,
            transfer_queue_family: u32,
            compute_queue_family: u32,
            supported_encode_codecs: Vec<Codec>,
            has_descriptor_buffer_ext: bool,
            /// DRM render (major, minor) reported by `VK_EXT_physical_device_drm`, if available.
            drm_render: Option<(i64, i64)>,
        }

        let mut candidates: Vec<DeviceCandidate> = Vec::new();

        for physical_device in physical_devices {
            let props = unsafe { instance.get_physical_device_properties(physical_device) };
            let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                .to_string_lossy()
                .to_string();
            debug!("Checking device: {}", device_name);

            // Enumerate device extensions once (used for DRM, descriptor buffer, and codec checks).
            let available_extensions =
                match unsafe { instance.enumerate_device_extension_properties(physical_device) } {
                    Ok(exts) => exts,
                    Err(e) => {
                        warn!(
                            "Failed to enumerate device extension properties for {}: {}. Skipping device.",
                            device_name, e
                        );
                        continue;
                    },
                };
            let has_extension = |name: &std::ffi::CStr| -> bool {
                available_extensions.iter().any(|ext| {
                    let ext_name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
                    ext_name == name
                })
            };

            // Query the DRM render node (major, minor) backing this device so it
            // can be matched against the compositor's render node.
            let drm_render = if has_extension(c"VK_EXT_physical_device_drm") {
                let mut drm_props = vk::PhysicalDeviceDrmPropertiesEXT::default();
                {
                    let mut props2 = vk::PhysicalDeviceProperties2::default().push(&mut drm_props);
                    unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
                }
                if drm_props.has_render != 0 {
                    Some((drm_props.render_major, drm_props.render_minor))
                } else {
                    None
                }
            } else {
                None
            };

            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

            // Find queue families.
            let mut encode_queue = None;
            let mut transfer_q = u32::MAX;
            let mut compute_q = u32::MAX;

            for (idx, qf) in queue_families.iter().enumerate() {
                debug!(
                    "Queue family {}: flags={:?}, count={}",
                    idx, qf.queue_flags, qf.queue_count
                );

                // Check for video encode queue.
                if qf.queue_flags.contains(vk::QueueFlags::VIDEO_ENCODE_KHR) {
                    encode_queue = Some(idx as u32);
                    debug!("Found video encode queue at family {}", idx);
                }

                // Check for transfer queue.
                if qf.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                    transfer_q = idx as u32;
                }

                // Check for compute queue (prefer dedicated compute, otherwise graphics+compute).
                if qf.queue_flags.contains(vk::QueueFlags::COMPUTE) && compute_q == u32::MAX {
                    compute_q = idx as u32;
                    debug!("Found compute queue at family {}", idx);
                }
            }

            // Check codec support for encoding.
            let mut encode_codecs = Vec::new();
            let mut has_descriptor_buffer_ext = false;
            if let Some(eq) = encode_queue {
                // Check if descriptor buffer extension is available.
                has_descriptor_buffer_ext = has_extension(ash::ext::descriptor_buffer::NAME);

                // Only check codec support if the extension exists
                if has_extension(ash::khr::video_encode_h264::NAME)
                    && Self::check_h264_encode_support(&entry, &instance, physical_device, eq)
                {
                    encode_codecs.push(Codec::H264);
                    debug!("Device {} supports H.264 encode", device_name);
                }
                if has_extension(ash::khr::video_encode_h265::NAME)
                    && Self::check_h265_encode_support(&entry, &instance, physical_device, eq)
                {
                    encode_codecs.push(Codec::H265);
                    debug!("Device {} supports H.265 encode", device_name);
                }
                if has_extension(ash::khr::video_encode_av1::NAME)
                    && Self::check_av1_encode_support(&entry, &instance, physical_device, eq)
                {
                    encode_codecs.push(Codec::AV1);
                    debug!("Device {} supports AV1 encode", device_name);
                }
            }

            // Check if all required encode codecs are supported.
            let encode_supported = builder
                .required_encode_codecs
                .iter()
                .all(|codec| encode_codecs.contains(codec));

            // We need encode support and compute support.
            let has_video_support = encode_queue.is_some();
            let has_compute_support = compute_q != u32::MAX;

            if has_video_support && encode_supported && has_compute_support {
                candidates.push(DeviceCandidate {
                    physical_device,
                    device_name,
                    video_encode_queue_family: encode_queue,
                    transfer_queue_family: if transfer_q != u32::MAX {
                        transfer_q
                    } else {
                        encode_queue.unwrap_or(0)
                    },
                    compute_queue_family: compute_q,
                    supported_encode_codecs: encode_codecs,
                    has_descriptor_buffer_ext,
                    drm_render,
                });
            } else {
                warn!(
                    "Device {} skipped: video_support={}, encode_supported={}, compute_support={}",
                    device_name, has_video_support, encode_supported, has_compute_support
                );
                if !has_video_support {
                    warn!("  - No queue with VIDEO_ENCODE_KHR flag found");
                }
                if !encode_supported {
                    warn!(
                        "  - Required codecs not supported: {:?}",
                        builder.required_encode_codecs
                    );
                    warn!("  - Available codecs: {:?}", encode_codecs);
                }
            }
        }

        // Pick the candidate whose DRM render node matches the requested one, so
        // the encoder shares the compositor's GPU. Fall back to the first suitable
        // device (previous behavior) when no node is requested or none matches.
        let chosen_index = match preferred_drm {
            Some(target) => candidates
                .iter()
                .position(|c| c.drm_render == Some(target))
                .or_else(|| {
                    if candidates.is_empty() {
                        None
                    } else {
                        warn!(
                            "No encode-capable Vulkan device matches DRM render node {}:{}; falling back to the first suitable device (cross-GPU corruption possible)",
                            target.0, target.1
                        );
                        Some(0)
                    }
                }),
            None => (!candidates.is_empty()).then_some(0),
        };

        let chosen = chosen_index.map(|i| candidates.swap_remove(i)).ok_or_else(|| {
            PixelForgeError::NoSuitableDevice(
                "No device with required video support found. Ensure your GPU drivers support Vulkan Video extensions (VK_KHR_video_queue, VK_KHR_video_encode_queue, etc.).".to_string(),
            )
        })?;

        let physical_device = chosen.physical_device;
        let video_encode_queue_family = chosen.video_encode_queue_family;
        let transfer_queue_family = chosen.transfer_queue_family;
        let compute_queue_family = chosen.compute_queue_family;
        let supported_encode_codecs = chosen.supported_encode_codecs;
        let has_descriptor_buffer_ext = chosen.has_descriptor_buffer_ext;
        match chosen.drm_render {
            Some((major, minor)) => {
                info!("Selected device: {} (drm {}:{})", chosen.device_name, major, minor)
            },
            None => info!("Selected device: {} (drm render node unknown)", chosen.device_name),
        }

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

        // Enable VK_EXT_descriptor_buffer extension (required for descriptor buffer API).
        if has_descriptor_buffer_ext {
            push_ext(ash::ext::descriptor_buffer::NAME.as_ptr());
        } else {
            warn!("VK_EXT_descriptor_buffer not available on this device");
        }

        // Enable synchronization2 feature.
        let mut sync2_features =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);

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
        let mut ycbcr_features = vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default()
            .sampler_ycbcr_conversion(true);

        // Enable YCbCr 2-plane 444 formats feature (required for YUV444 encoding with NVIDIA).
        let mut ycbcr_2plane_444_features =
            vk::PhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT::default()
                .ycbcr2plane444_formats(true);

        // Add the 2-plane 444 formats extension.
        push_ext(ash::ext::ycbcr_2plane_444_formats::NAME.as_ptr());

        // Enable AV1 video encode feature only if AV1 is supported.
        // Only include AV1 features in the pNext chain when AV1 is actually supported,
        // to avoid chaining unknown feature structs on devices without AV1.
        let mut av1_encode_features =
            vk::PhysicalDeviceVideoEncodeAV1FeaturesKHR::default().video_encode_av1(true);

        // Query descriptor buffer and buffer device address feature support.
        let mut desc_buf_features = vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default();
        let mut buffer_device_address_features =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default();

        if has_descriptor_buffer_ext {
            let mut feat2 = vk::PhysicalDeviceFeatures2::default().push(&mut desc_buf_features);
            unsafe {
                instance.get_physical_device_features2(physical_device, &mut feat2);
            }
            let desc_buf_supported = desc_buf_features.descriptor_buffer != 0
                && desc_buf_features.descriptor_buffer_capture_replay != 0;

            // Query buffer device address support.
            let mut feat2_bda =
                vk::PhysicalDeviceFeatures2::default().push(&mut buffer_device_address_features);
            unsafe {
                instance.get_physical_device_features2(physical_device, &mut feat2_bda);
            }

            if desc_buf_supported && buffer_device_address_features.buffer_device_address != 0 {
                desc_buf_features.descriptor_buffer = 1;
                desc_buf_features.descriptor_buffer_capture_replay = 1;
            } else if desc_buf_supported {
                warn!(
                    "VK_EXT_descriptor_buffer extension present but bufferDeviceAddress not supported; descriptor buffer will not be enabled"
                );
            }
        }

        // Store whether descriptor buffer is available for use by callers.
        let has_descriptor_buffer = has_descriptor_buffer_ext
            && desc_buf_features.descriptor_buffer != 0
            && desc_buf_features.descriptor_buffer_capture_replay != 0;

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

        if supported_encode_codecs.contains(&Codec::AV1) {
            device_create_info = device_create_info.push(&mut av1_encode_features);
        }

        // Attach the chain to device_create_info.
        // When descriptor buffer is available, the chain is:
        //   desc_buf_features -> buffer_device_address_features -> sync2_features -> ...
        // When descriptor buffer is not available, only sync2_features is chained.
        if has_descriptor_buffer_ext {
            device_create_info = device_create_info
                .push(&mut desc_buf_features)
                .push(&mut buffer_device_address_features)
                .push(&mut ycbcr_features)
                .push(&mut ycbcr_2plane_444_features);
        }

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
            .map_err(|e| PixelForgeError::DeviceCreation(e.to_string()))?;

        // Get queues.
        let video_encode_queue =
            video_encode_queue_family.map(|family| unsafe { device.get_device_queue(family, 0) });
        let transfer_queue = unsafe { device.get_device_queue(transfer_queue_family, 0) };
        let compute_queue = unsafe { device.get_device_queue(compute_queue_family, 0) };

        if let Some(family) = video_encode_queue_family {
            info!("Video encode queue family: {}", family);
        }
        info!("Transfer queue family: {}", transfer_queue_family);
        info!("Compute queue family: {}", compute_queue_family);
        info!("Created Vulkan device with video support");

        Ok(Self {
            inner: std::sync::Arc::new(VideoContextInner {
                entry,
                instance,
                physical_device,
                device,
                video_encode_queue_family,
                video_encode_queue,
                transfer_queue_family,
                transfer_queue,
                compute_queue_family,
                compute_queue,
                memory_properties,
                device_properties,
                supported_encode_codecs,
                has_descriptor_buffer,
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
    pub fn supports_encode(&self, codec: Codec) -> bool {
        self.inner.supported_encode_codecs.contains(&codec)
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

/// Resolve a device node path (e.g. `/dev/dri/renderD128`) to its DRM
/// `(major, minor)`, matching `VkPhysicalDeviceDrmPropertiesEXT`'s
/// `render_major`/`render_minor`.
///
/// Uses the glibc `dev_t` encoding (`gnu_dev_major`/`gnu_dev_minor`). Linux only.
#[cfg(target_os = "linux")]
fn drm_render_major_minor(path: &std::path::Path) -> Option<(i64, i64)> {
    use std::os::unix::fs::MetadataExt;
    let rdev = std::fs::metadata(path).ok()?.rdev();
    let major = ((rdev >> 8) & 0xfff) | ((rdev >> 32) & !0xfff);
    let minor = (rdev & 0xff) | ((rdev >> 12) & !0xff);
    Some((major as i64, minor as i64))
}

#[cfg(not(target_os = "linux"))]
fn drm_render_major_minor(_path: &std::path::Path) -> Option<(i64, i64)> {
    None
}
