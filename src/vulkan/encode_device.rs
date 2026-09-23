//! Encoding on a device the caller created.
//!
//! The encode counterpart of [`VideoContextBuilder::build_from_existing_decode`].
//! An application that already renders on its own device, and wants to encode
//! what it renders, can hand that device to pixelforge. The encoder and the
//! colour converter then work on the application's images directly: no
//! external memory, no export and import, and a semaphore instead of a CPU
//! wait where one side hands a frame to the other.

use super::queues::{self, QueueOverrides};
use super::{
    DeviceFeatures, DeviceRequirements, QueueRoles, VideoContext, VideoContextBuilder,
    VideoContextInner, scan_queue_families, supports_internally_synchronized_queues,
    supports_rgb_conversion,
};
use crate::encoder::Codec;
use crate::error::{PixelForgeError, Result};
use ash::vk;
use ash::vk::TaggedStructure;
use tracing::info;

/// The queue families pixelforge selects for encoding on a given device.
struct EncodeQueueFamilies {
    encode: u32,
    encode_timestamp_valid_bits: u32,
    transfer: u32,
    compute: u32,
}

/// What pixelforge will use on `physical_device` to encode `required`.
///
/// When `required` is empty every codec the device encodes is included, so a
/// caller that has not decided yet gets a device that can do any of them.
struct EncodePlan {
    families: EncodeQueueFamilies,
    codecs: Vec<Codec>,
    push_descriptor: bool,
    ycbcr_2plane_444: bool,
    sampler_ycbcr_conversion: bool,
    rgb_conversion: bool,
}

impl EncodePlan {
    fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        required: &[Codec],
    ) -> Result<Self> {
        let scanned =
            scan_queue_families(instance, physical_device, vk::QueueFlags::VIDEO_ENCODE_KHR)?;
        let encode = scanned.video.ok_or_else(|| {
            PixelForgeError::NoSuitableDevice(
                "Physical device has no video encode queue family".to_string(),
            )
        })?;

        let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device) }
            .map_err(|e| PixelForgeError::NoSuitableDevice(e.to_string()))?;
        let has = |name: &std::ffi::CStr| {
            extensions
                .iter()
                .any(|ext| unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) } == name)
        };

        let mut available = Vec::new();
        if has(ash::khr::video_encode_h264::NAME)
            && VideoContext::check_h264_encode_support(entry, instance, physical_device, encode)
        {
            available.push(Codec::H264);
        }
        if has(ash::khr::video_encode_h265::NAME)
            && VideoContext::check_h265_encode_support(entry, instance, physical_device, encode)
        {
            available.push(Codec::H265);
        }
        if has(ash::khr::video_encode_av1::NAME)
            && VideoContext::check_av1_encode_support(entry, instance, physical_device, encode)
        {
            available.push(Codec::AV1);
        }
        let codecs = if required.is_empty() {
            available
        } else {
            for codec in required {
                if !available.contains(codec) {
                    return Err(PixelForgeError::CodecNotSupported(format!(
                        "Physical device does not support encoding {:?}",
                        codec
                    )));
                }
            }
            required.to_vec()
        };
        if codecs.is_empty() {
            return Err(PixelForgeError::NoSuitableDevice(
                "Physical device encodes none of H.264, H.265 or AV1".to_string(),
            ));
        }

        let mut ycbcr = vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default();
        let mut query = vk::PhysicalDeviceFeatures2::default().push(&mut ycbcr);
        unsafe { instance.get_physical_device_features2(physical_device, &mut query) };

        Ok(Self {
            families: EncodeQueueFamilies {
                encode,
                encode_timestamp_valid_bits: scanned.video_timestamp_valid_bits,
                transfer: scanned.transfer,
                compute: scanned.compute,
            },
            codecs,
            push_descriptor: has(ash::khr::push_descriptor::NAME),
            ycbcr_2plane_444: has(ash::ext::ycbcr_2plane_444_formats::NAME),
            sampler_ycbcr_conversion: ycbcr.sampler_ycbcr_conversion != 0,
            rgb_conversion: has(ash::valve::video_encode_rgb_conversion::NAME)
                && supports_rgb_conversion(instance, physical_device),
        })
    }

    fn extensions(&self) -> Vec<&'static std::ffi::CStr> {
        // Synchronization2 and timeline semaphores are core in 1.3 and 1.2, and
        // listed anyway so a device created from an instance asking for less
        // still has them.
        let mut names = vec![
            ash::khr::video_queue::NAME,
            ash::khr::video_encode_queue::NAME,
            ash::khr::synchronization2::NAME,
            ash::khr::timeline_semaphore::NAME,
        ];
        for codec in &self.codecs {
            names.push(match codec {
                Codec::H264 => ash::khr::video_encode_h264::NAME,
                Codec::H265 => ash::khr::video_encode_h265::NAME,
                Codec::AV1 => ash::khr::video_encode_av1::NAME,
            });
        }
        if self.push_descriptor {
            names.push(ash::khr::push_descriptor::NAME);
        }
        if self.ycbcr_2plane_444 {
            names.push(ash::ext::ycbcr_2plane_444_formats::NAME);
        }
        if self.rgb_conversion {
            names.push(ash::valve::video_encode_rgb_conversion::NAME);
        }
        names
    }

    fn features(&self) -> DeviceFeatures {
        DeviceFeatures {
            synchronization2: true,
            timeline_semaphore: true,
            sampler_ycbcr_conversion: self.sampler_ycbcr_conversion,
            ycbcr_2plane_444_formats: self.ycbcr_2plane_444,
            video_encode_av1: self.codecs.contains(&Codec::AV1),
            video_encode_rgb_conversion: self.rgb_conversion,
        }
    }

    fn roles(&self) -> QueueRoles {
        QueueRoles {
            encode: Some(self.families.encode),
            decode: None,
            transfer: self.families.transfer,
            compute: self.families.compute,
        }
    }
}

impl VideoContextBuilder {
    /// What a caller-created device must provide for pixelforge to encode on
    /// it, and to run the colour converter.
    ///
    /// Use this when you already have your own Vulkan device (a renderer's, or
    /// one a game created) and want to encode images that live on it, without
    /// exporting them to a second device. The codecs asked for with
    /// [`Self::require_encode`] must all be supported; with none asked for,
    /// every codec the device encodes is included.
    ///
    /// The colour converter needs `VK_KHR_push_descriptor`, which is included
    /// whenever the device has it. On a device without it, encoding from YUV
    /// input still works and only the converter is unavailable.
    /// `VK_VALVE_video_encode_rgb_conversion` is included the same way, for
    /// [`EncodeConfig::with_rgb_input`](crate::EncodeConfig::with_rgb_input).
    pub fn encode_device_requirements(
        &self,
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<DeviceRequirements> {
        let plan = EncodePlan::new(
            entry,
            instance,
            physical_device,
            &self.required_encode_codecs,
        )?;
        let queues = plan.roles();
        Ok(DeviceRequirements {
            queue_families: queues.unique_families(),
            queues,
            extensions: plan.extensions(),
            features: plan.features(),
            unified_image_layouts: false,
            internally_synchronized_queues: supports_internally_synchronized_queues(
                instance,
                physical_device,
            ),
        })
    }

    /// Adopt a caller-created device for encoding, rather than creating one.
    ///
    /// The device must have been created with the queue families, extensions
    /// and features reported by [`Self::encode_device_requirements`], called
    /// with the same codecs for the same `physical_device`. The instance may
    /// ask for any Vulkan version from 1.1 up.
    ///
    /// The resulting context **borrows** `instance` and `device`: dropping it
    /// frees neither, so the caller must keep both alive for at least as long
    /// as the context and every encoder and converter made from it.
    ///
    /// Pixelforge submits to one queue per role, queue 0 of each family unless
    /// told otherwise. A caller that keeps submitting to its own queues while
    /// pixelforge works on another thread should give it spare ones with
    /// [`Self::with_encode_queue`], [`Self::with_transfer_queue`] and
    /// [`Self::with_compute_queue`].
    pub fn build_from_existing_encode(
        self,
        entry: ash::Entry,
        instance: ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
    ) -> Result<VideoContext> {
        VideoContext::from_existing_encode(
            &self.required_encode_codecs,
            self.queues,
            entry,
            instance,
            physical_device,
            device,
        )
    }
}

impl VideoContext {
    fn from_existing_encode(
        required_encode_codecs: &[Codec],
        queue_overrides: QueueOverrides,
        entry: ash::Entry,
        instance: ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
    ) -> Result<VideoContext> {
        let plan = EncodePlan::new(&entry, &instance, physical_device, required_encode_codecs)?;
        let families = &plan.families;

        let encode = queues::resolve(
            &instance,
            physical_device,
            "encode",
            queue_overrides.encode,
            families.encode,
            vk::QueueFlags::VIDEO_ENCODE_KHR,
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

        // The caller may have picked another encode family than the scan did,
        // so the timestamp support is that family's, not the scanned one's.
        let encode_timestamp_valid_bits = if encode.family == families.encode {
            families.encode_timestamp_valid_bits
        } else {
            let props =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            props[encode.family as usize].timestamp_valid_bits
        };

        let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let video_encode_queue = unsafe { device.get_device_queue(encode.family, encode.index) };
        let transfer_queue = unsafe { device.get_device_queue(transfer.family, transfer.index) };
        let compute_queue = unsafe { device.get_device_queue(compute.family, compute.index) };

        info!(
            "Adopted caller device for encode: {:?}, encode queue {:?}, transfer queue {:?}, compute queue {:?}",
            plan.codecs, encode, transfer, compute
        );

        Ok(VideoContext {
            inner: std::sync::Arc::new(VideoContextInner {
                sync2: ash::khr::synchronization2::Device::load(&instance, &device),
                entry,
                instance,
                physical_device,
                device,
                video_encode_queue_family: Some(encode.family),
                video_encode_timestamp_valid_bits: encode_timestamp_valid_bits,
                video_encode_queue: Some(video_encode_queue),
                video_decode_queue_family: None,
                video_decode_queue: None,
                transfer_queue_family: transfer.family,
                transfer_queue,
                compute_queue_family: compute.family,
                compute_queue,
                memory_properties,
                device_properties,
                supported_encode_codecs: plan.codecs.clone(),
                supported_decode_codecs: Vec::new(),
                has_push_descriptor: plan.push_descriptor,
                has_video_encode_rgb_conversion: plan.rgb_conversion,
                has_unified_image_layouts: false,
                owns_device: false,
                // The caller owns the instance; reporting is theirs to set up.
                debug_messenger: None,
            }),
        })
    }
}
