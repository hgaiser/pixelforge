//! Encode on a caller-created device.
//!
//! Mirrors what an application with its own Vulkan device does, such as a
//! renderer that wants to stream what it draws: create the instance and device
//! itself, then hand them to pixelforge via `build_from_existing_encode`, so
//! the encoder works on the application's device instead of a second one.
//!
//! Usage:
//!   encode_adopted [input.yuv] [output.h264]
//!
//! The input is raw 320x240 YUV420, `testdata/test_frames.yuv` by default.

use std::fs::File;
use std::io::Write;

use ash::vk;
use ash::vk::TaggedStructure;
use pixelforge::vulkan::{DeviceQueue, VideoContextBuilder};
use pixelforge::{
    Codec, EncodeBitDepth, EncodeConfig, Encoder, InputImage, PixelFormat, RateControlMode,
};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input_path = args
        .next()
        .unwrap_or_else(|| "testdata/test_frames.yuv".to_string());
    let output_path = args.next().unwrap_or_else(|| "output.h264".to_string());

    // --- The application creates its own instance and device ---
    let entry = unsafe { ash::Entry::load()? };
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"encode-adopted")
        .api_version(vk::API_VERSION_1_3);
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_info, None)? };

    // Ask pixelforge what an encode device needs, and pick a physical device
    // that satisfies it.
    let builder = VideoContextBuilder::new().require_encode(Codec::H264);
    let (physical_device, reqs) = unsafe { instance.enumerate_physical_devices()? }
        .into_iter()
        .find_map(|pd| {
            builder
                .encode_device_requirements(&entry, &instance, pd)
                .ok()
                .map(|r| (pd, r))
        })
        .expect("no physical device can encode H.264");

    // Keep queue 0 of each family for the application, and give pixelforge a
    // second queue wherever the family has one. A `VkQueue` may not be
    // submitted to from two threads at once, so this is what lets the
    // application keep rendering while pixelforge encodes on another thread.
    let family_props =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let priorities = [1.0f32; 2];
    let queue_infos: Vec<vk::DeviceQueueCreateInfo> = reqs
        .queue_families
        .iter()
        .map(|&f| {
            let count = family_props[f as usize].queue_count.min(2) as usize;
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(f)
                .queue_priorities(&priorities[..count])
        })
        .collect();
    let spare = |family: u32| {
        DeviceQueue::new(family, family_props[family as usize].queue_count.min(2) - 1)
    };
    let ext_ptrs: Vec<_> = reqs.extensions.iter().map(|e| e.as_ptr()).collect();

    // `reqs.features` says which feature bits to turn on, and every one it
    // sets has to be: pixelforge takes what the requirements list as enabled,
    // since Vulkan cannot be asked afterwards. An application that already
    // chains `VkPhysicalDeviceVulkan13Features` would set them there; this one
    // has nothing else, so it uses the individual structs.
    let features = reqs.features;
    let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default()
        .synchronization2(features.synchronization2);
    let mut timeline = vk::PhysicalDeviceTimelineSemaphoreFeatures::default()
        .timeline_semaphore(features.timeline_semaphore);
    let mut ycbcr = vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default()
        .sampler_ycbcr_conversion(features.sampler_ycbcr_conversion);
    let mut ycbcr_444 =
        vk::PhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT::default().ycbcr2plane444_formats(true);
    let mut device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&ext_ptrs)
        .push(&mut sync2)
        .push(&mut timeline)
        .push(&mut ycbcr);
    if features.ycbcr_2plane_444_formats {
        device_info = device_info.push(&mut ycbcr_444);
    }
    let mut av1 = vk::PhysicalDeviceVideoEncodeAV1FeaturesKHR::default().video_encode_av1(true);
    if features.video_encode_av1 {
        device_info = device_info.push(&mut av1);
    }
    let mut rgb = vk::PhysicalDeviceVideoEncodeRgbConversionFeaturesVALVE::default()
        .video_encode_rgb_conversion(true);
    if features.video_encode_rgb_conversion {
        device_info = device_info.push(&mut rgb);
    }
    let device = unsafe { instance.create_device(physical_device, &device_info, None)? };

    // --- Hand the application's device to pixelforge ---
    let context = builder
        .with_encode_queue(spare(reqs.queues.encode.unwrap()))
        .with_transfer_queue(spare(reqs.queues.transfer))
        .with_compute_queue(spare(reqs.queues.compute))
        .build_from_existing_encode(
            entry.clone(),
            instance.clone(),
            physical_device,
            device.clone(),
        )?;

    let yuv = std::fs::read(&input_path)?;
    let frame_size = (WIDTH * HEIGHT * 3 / 2) as usize;
    let mut input = InputImage::new(
        context.clone(),
        Codec::H264,
        WIDTH,
        HEIGHT,
        EncodeBitDepth::Eight,
        PixelFormat::Yuv420,
    )?;
    let mut encoder = Encoder::new(
        context,
        EncodeConfig::h264(WIDTH, HEIGHT)
            .with_rate_control(RateControlMode::Cqp)
            .with_quality_level(26)
            .with_frame_rate(30, 1)
            .with_b_frames(0),
    )?;
    let mut output = File::create(&output_path)?;
    let mut count = 0usize;
    for frame in yuv.chunks_exact(frame_size) {
        input.upload_yuv420(frame)?;
        let packet = pollster::block_on(encoder.encode(input.image())?)?;
        output.write_all(&packet.data)?;
        count += 1;
    }
    println!("Encoded {count} frames on the adopted device into {output_path}");

    // The context borrowed the device, so everything holding Vulkan objects
    // has to go before the device does.
    drop(encoder);
    drop(input);
    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    Ok(())
}
