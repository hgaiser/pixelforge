//! Pixelforge on a device the caller created.
//!
//! An application that already renders on its own device can hand that device
//! to pixelforge, to encode what it renders or to decode into images it
//! samples. The encoder, the decoder and the colour converter then work on the
//! application's images directly: no external memory, no export and import,
//! and a semaphore instead of a CPU wait where one side hands a frame to the
//! other.

use super::ExternalSupport;
use super::queues;
use super::{
    DecodeQueueFamilies, DeviceExtensions, DeviceFeatures, DeviceQueue, DeviceRequirements,
    QueueRoles, VideoContext, VideoContextBuilder, VideoContextInner, decode_extension_names,
    find_decode_queue_families, query_decode_codecs, scan_queue_families,
    supports_internally_synchronized_queues, supports_unified_image_layouts,
};
use crate::encoder::Codec;
use crate::error::{PixelForgeError, Result};
use ash::vk;
use ash::vk::TaggedStructure;
use tracing::info;

/// The queue families pixelforge selects for encoding on a given device.
struct EncodeQueueFamilies {
    /// The encode queue: the caller's choice, or queue 0 of the family the
    /// scan found. Settled before anything else, since codec support and
    /// timestamp support both belong to a family.
    encode: DeviceQueue,
    encode_timestamp_valid_bits: u32,
    transfer: u32,
    compute: u32,
}

/// What pixelforge will use on `physical_device` to encode `required`.
struct EncodePlan {
    families: EncodeQueueFamilies,
    codecs: Vec<Codec>,
    push_descriptor: bool,
    ycbcr_2plane_444: bool,
    sampler_ycbcr_conversion: bool,
    rgb_conversion: bool,
    external: ExternalSupport,
}

impl EncodePlan {
    /// `encode_queue` is the caller's choice of encode queue, if it made one.
    /// Codecs are checked on that queue's family, since that is the one the
    /// encoder will submit to.
    fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        required: &[Codec],
        encode_queue: Option<DeviceQueue>,
    ) -> Result<Self> {
        let scanned =
            scan_queue_families(instance, physical_device, vk::QueueFlags::VIDEO_ENCODE_KHR)?;
        let scanned_encode = scanned.video.ok_or_else(|| {
            PixelForgeError::NoSuitableDevice(
                "Physical device has no video encode queue family".to_string(),
            )
        })?;
        let encode = queues::resolve(
            instance,
            physical_device,
            "encode",
            encode_queue,
            scanned_encode,
            vk::QueueFlags::VIDEO_ENCODE_KHR,
        )?;
        let encode_timestamp_valid_bits = if encode.family == scanned_encode {
            scanned.video_timestamp_valid_bits
        } else {
            let props =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            props[encode.family as usize].timestamp_valid_bits
        };

        let extensions = DeviceExtensions::of(instance, physical_device);
        let has = |name: &std::ffi::CStr| extensions.has(name);

        // The feature bits behind three of the extensions. A driver may offer an
        // extension and still report its feature off, and a caller told to
        // enable a feature the device lacks gets VK_ERROR_FEATURE_NOT_PRESENT
        // from vkCreateDevice. A struct is chained only when its extension is
        // offered, as Vulkan requires.
        let mut ycbcr = vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default();
        let mut ycbcr_444 = vk::PhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT::default();
        let mut av1 = vk::PhysicalDeviceVideoEncodeAV1FeaturesKHR::default();
        let mut rgb = vk::PhysicalDeviceVideoEncodeRgbConversionFeaturesVALVE::default();
        let mut query = vk::PhysicalDeviceFeatures2::default().push(&mut ycbcr);
        if has(ash::ext::ycbcr_2plane_444_formats::NAME) {
            query = query.push(&mut ycbcr_444);
        }
        if has(ash::khr::video_encode_av1::NAME) {
            query = query.push(&mut av1);
        }
        if has(ash::valve::video_encode_rgb_conversion::NAME) {
            query = query.push(&mut rgb);
        }
        unsafe { instance.get_physical_device_features2(physical_device, &mut query) };

        let family = encode.family;
        let mut available = Vec::new();
        if has(ash::khr::video_encode_h264::NAME)
            && VideoContext::check_h264_encode_support(entry, instance, physical_device, family)
        {
            available.push(Codec::H264);
        }
        if has(ash::khr::video_encode_h265::NAME)
            && VideoContext::check_h265_encode_support(entry, instance, physical_device, family)
        {
            available.push(Codec::H265);
        }
        if has(ash::khr::video_encode_av1::NAME)
            && av1.video_encode_av1 != 0
            && VideoContext::check_av1_encode_support(entry, instance, physical_device, family)
        {
            available.push(Codec::AV1);
        }
        for codec in required {
            if !available.contains(codec) {
                return Err(PixelForgeError::NoSuitableDevice(format!(
                    "Queue family {} does not support encoding {:?}",
                    family, codec
                )));
            }
        }
        let codecs = required.to_vec();

        Ok(Self {
            families: EncodeQueueFamilies {
                encode,
                encode_timestamp_valid_bits,
                transfer: scanned.transfer,
                compute: scanned.compute,
            },
            codecs,
            push_descriptor: has(ash::khr::push_descriptor::NAME),
            ycbcr_2plane_444: has(ash::ext::ycbcr_2plane_444_formats::NAME)
                && ycbcr_444.ycbcr2plane444_formats != 0,
            sampler_ycbcr_conversion: ycbcr.sampler_ycbcr_conversion != 0,
            rgb_conversion: has(ash::valve::video_encode_rgb_conversion::NAME)
                && rgb.video_encode_rgb_conversion != 0,
            external: ExternalSupport::detect(has),
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
        names.extend(self.external.extension_names());
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
}

/// What pixelforge will use on an adopted device: the encode role for the
/// codecs asked for with [`VideoContextBuilder::require_encode`], the decode
/// role for those asked for with [`VideoContextBuilder::require_decode`], or
/// both, and a transfer and a compute queue.
struct AdoptPlan {
    encode: Option<EncodePlan>,
    decode: Option<DecodeQueueFamilies>,
    transfer: u32,
    compute: u32,
    decode_codecs: Vec<Codec>,
    /// Whether the device supports unified image layouts, which only the
    /// decoder uses.
    unified_image_layouts: bool,
    internally_synchronized_queues: bool,
}

impl AdoptPlan {
    fn new(
        builder: &VideoContextBuilder,
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let encode_codecs = &builder.required_encode_codecs;
        let decode_codecs = &builder.required_decode_codecs;
        if encode_codecs.is_empty() && decode_codecs.is_empty() {
            return Err(PixelForgeError::InvalidInput(
                "an adopted device needs at least one codec, asked for with require_encode or \
                 require_decode"
                    .to_string(),
            ));
        }
        let encode = (!encode_codecs.is_empty())
            .then(|| {
                EncodePlan::new(
                    entry,
                    instance,
                    physical_device,
                    encode_codecs,
                    builder.queues.encode,
                )
            })
            .transpose()?;
        let decode = (!decode_codecs.is_empty())
            .then(|| find_decode_queue_families(entry, instance, physical_device, decode_codecs))
            .transpose()?;
        // Both scans pick the transfer and compute families the same way.
        let (transfer, compute) = match (&encode, &decode) {
            (Some(e), _) => (e.families.transfer, e.families.compute),
            (None, Some(d)) => (d.transfer, d.compute),
            (None, None) => unreachable!("a codec was asked for"),
        };
        Ok(Self {
            unified_image_layouts: decode.is_some()
                && supports_unified_image_layouts(instance, physical_device),
            internally_synchronized_queues: supports_internally_synchronized_queues(
                instance,
                physical_device,
                &DeviceExtensions::of(instance, physical_device),
            ),
            encode,
            decode,
            transfer,
            compute,
            decode_codecs: decode_codecs.clone(),
        })
    }

    fn roles(&self) -> QueueRoles {
        QueueRoles {
            encode: self.encode.as_ref().map(|e| e.families.encode.family),
            decode: self.decode.as_ref().map(|d| d.decode),
            transfer: self.transfer,
            compute: self.compute,
        }
    }

    fn extensions(&self) -> Vec<&'static std::ffi::CStr> {
        let mut names = self
            .encode
            .as_ref()
            .map(EncodePlan::extensions)
            .unwrap_or_default();
        if self.decode.is_some() {
            names.extend(decode_extension_names(&self.decode_codecs));
        }
        if self.unified_image_layouts {
            names.push(ash::khr::unified_image_layouts::NAME);
        }
        let mut unique = Vec::with_capacity(names.len());
        for name in names {
            if !unique.contains(&name) {
                unique.push(name);
            }
        }
        unique
    }

    fn features(&self) -> DeviceFeatures {
        self.encode
            .as_ref()
            .map(EncodePlan::features)
            .unwrap_or(DeviceFeatures {
                synchronization2: true,
                timeline_semaphore: true,
                ..DeviceFeatures::default()
            })
    }
}

impl VideoContextBuilder {
    /// What a caller-created device must provide for pixelforge to work on
    /// it, for the codecs asked for with [`Self::require_encode`] and
    /// [`Self::require_decode`]. At least one codec must be asked for, and
    /// the device must support every one.
    ///
    /// Use this when you already have your own Vulkan device (a renderer's, or
    /// one a game created) and want to encode images that live on it, or
    /// decode into images it uses, without a second device and the copies
    /// between the two.
    ///
    /// For encoding, the colour converter needs `VK_KHR_push_descriptor`,
    /// which is included whenever the device has it. On a device without it,
    /// encoding from YUV input still works and only the converter is
    /// unavailable. `VK_VALVE_video_encode_rgb_conversion` is included the
    /// same way, for [`Conversion::Encoder`](crate::Conversion::Encoder).
    pub fn device_requirements(
        &self,
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<DeviceRequirements> {
        let plan = AdoptPlan::new(self, entry, instance, physical_device)?;
        let queues = plan.roles();
        Ok(DeviceRequirements {
            queue_families: queues.unique_families(),
            queues,
            extensions: plan.extensions(),
            features: plan.features(),
            unified_image_layouts: plan.unified_image_layouts,
            internally_synchronized_queues: plan.internally_synchronized_queues,
        })
    }

    /// Adopt a caller-created device, rather than creating one.
    ///
    /// The device must have been created with the queue families, extensions
    /// and features reported by [`Self::device_requirements`], called with
    /// the same codecs for the same `physical_device`. The instance may ask
    /// for any Vulkan version from 1.1 up.
    ///
    /// The resulting context **borrows** `instance` and `device`: dropping it
    /// frees neither, so the caller must keep both alive for at least as long
    /// as the context and every encoder, decoder and converter made from it.
    ///
    /// Pixelforge submits to one queue per role, queue 0 of each family unless
    /// told otherwise. A caller that keeps submitting to its own queues while
    /// pixelforge works on another thread should give it spare ones with
    /// [`Self::with_encode_queue`], [`Self::with_decode_queue`],
    /// [`Self::with_transfer_queue`] and [`Self::with_compute_queue`].
    pub fn build_from_existing(
        self,
        entry: ash::Entry,
        instance: ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: ash::Device,
    ) -> Result<VideoContext> {
        let plan = AdoptPlan::new(&self, &entry, &instance, physical_device)?;
        let overrides = self.queues;
        let declared_unified_image_layouts =
            plan.decode.is_some() && self.unified_image_layouts.unwrap_or(false);
        // The declaration is a promise about what the caller enabled, which
        // cannot be verified. That the hardware could support it can be, so a
        // declaration that is impossible fails here rather than in the driver.
        if declared_unified_image_layouts && !plan.unified_image_layouts {
            return Err(PixelForgeError::NoSuitableDevice(
                "unified image layouts were declared, but this device does not support \
                 VK_KHR_unified_image_layouts with both unifiedImageLayouts and \
                 unifiedImageLayoutsVideo"
                    .to_string(),
            ));
        }

        let encode = plan.encode.as_ref().map(|e| e.families.encode);
        let decode = plan
            .decode
            .as_ref()
            .map(|d| {
                queues::resolve(
                    &instance,
                    physical_device,
                    "decode",
                    overrides.decode,
                    d.decode,
                    vk::QueueFlags::VIDEO_DECODE_KHR,
                )
            })
            .transpose()?;
        let transfer = queues::resolve(
            &instance,
            physical_device,
            "transfer",
            overrides.transfer,
            plan.transfer,
            vk::QueueFlags::TRANSFER,
        )?;
        let compute = queues::resolve(
            &instance,
            physical_device,
            "compute",
            overrides.compute,
            plan.compute,
            vk::QueueFlags::COMPUTE,
        )?;

        let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let queue = |q: DeviceQueue| unsafe { device.get_device_queue(q.family, q.index) };
        let video_encode_queue = encode.map(queue);
        let video_decode_queue = decode.map(queue);
        let transfer_queue = queue(transfer);
        let compute_queue = queue(compute);
        let supported_decode_codecs = if plan.decode.is_some() {
            query_decode_codecs(&entry, &instance, physical_device)
        } else {
            Vec::new()
        };

        info!(
            "Adopted caller device: encode {:?} on {:?}, decode {:?} on {:?}, transfer queue {:?}, compute queue {:?}",
            plan.encode.as_ref().map(|e| &e.codecs),
            encode,
            plan.decode.as_ref().map(|_| &plan.decode_codecs),
            decode,
            transfer,
            compute
        );

        let encode_plan = plan.encode.as_ref();
        Ok(VideoContext {
            inner: std::sync::Arc::new(VideoContextInner {
                sync2: ash::khr::synchronization2::Device::load(&instance, &device),
                timeline: ash::khr::timeline_semaphore::Device::load(&instance, &device),
                entry,
                instance,
                physical_device,
                device,
                video_encode_queue_family: encode.map(|q| q.family),
                video_encode_timestamp_valid_bits: encode_plan
                    .map_or(0, |e| e.families.encode_timestamp_valid_bits),
                video_encode_queue,
                video_decode_queue_family: decode.map(|q| q.family),
                video_decode_queue,
                transfer_queue_family: transfer.family,
                transfer_queue,
                compute_queue_family: compute.family,
                compute_queue,
                memory_properties,
                device_properties,
                supported_encode_codecs: encode_plan.map(|e| e.codecs.clone()).unwrap_or_default(),
                supported_decode_codecs,
                has_push_descriptor: encode_plan.is_some_and(|e| e.push_descriptor),
                has_video_encode_rgb_conversion: encode_plan.is_some_and(|e| e.rgb_conversion),
                has_unified_image_layouts: declared_unified_image_layouts,
                external: encode_plan.map(|e| e.external).unwrap_or_default(),
                owns_device: false,
                retired_semaphores: Default::default(),
                // The caller owns the instance; reporting is theirs to set up.
                debug_messenger: None,
            }),
        })
    }
}
