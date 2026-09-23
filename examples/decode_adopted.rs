//! Decode on a caller-created device.
//!
//! Mirrors what an application with its own Vulkan device does: create the
//! instance and device itself, then hand them to pixelforge via
//! `build_from_existing_decode` so decoded images live on the app's device.
//!
//! Usage:
//!   decode_adopted <input.264> [output.yuv]
//!
//! Output is NV12 in display order, comparable to:
//!   ffmpeg -i input.264 -pix_fmt nv12 reference.yuv

use std::fs::File;

use ash::vk;
use ash::vk::TaggedStructure;
#[allow(dead_code)]
mod common;
use common::{Readback, decode_stream, write_nv12};

use pixelforge::decoder::{DecodeConfig, Decoder};
use pixelforge::encoder::Codec;
use pixelforge::vulkan::{DeviceQueue, VideoContextBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input_path = args
        .next()
        .expect("usage: decode_adopted <input.264> [output.yuv]");
    let output_path = args.next();

    // --- The application creates its own instance and device ---
    let entry = unsafe { ash::Entry::load()? };
    let app_name = c"decode-adopted";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .api_version(vk::API_VERSION_1_3);
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_info, None)? };

    // Ask pixelforge what a decode device needs, and pick a physical device that
    // satisfies it — exactly the negotiation an app would do.
    let builder = VideoContextBuilder::new().require_decode(Codec::H264);
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    let (physical_device, reqs) = physical_devices
        .iter()
        .find_map(|&pd| {
            builder
                .decode_device_requirements(&entry, &instance, pd)
                .ok()
                .map(|r| (pd, r))
        })
        .expect("no physical device can decode H.264");

    let name = unsafe {
        std::ffi::CStr::from_ptr(
            instance
                .get_physical_device_properties(physical_device)
                .device_name
                .as_ptr(),
        )
    };
    println!("Device: {}", name.to_string_lossy());
    println!(
        "pixelforge needs queue families {:?} and {} extensions",
        reqs.queue_families,
        reqs.extensions.len()
    );

    // The app merges pixelforge's requirements with its own. A real app keeps
    // queue 0 of each family for itself, so where a family has a second queue
    // this one creates it and hands that to pixelforge instead: a `VkQueue`
    // may not be submitted to from two threads at once, and this is how the
    // app keeps its own queue to itself.
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
        let index = family_props[family as usize].queue_count.min(2) - 1;
        DeviceQueue::new(family, index)
    };
    let ext_ptrs: Vec<*const std::os::raw::c_char> =
        reqs.extensions.iter().map(|e| e.as_ptr()).collect();

    let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
    // The decoder orders its pipelined submissions with timeline semaphores.
    let mut timeline =
        vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
    // Unified image layouts are what let decoded frames be used in place, with
    // no copy and no layout transition. `reqs` says whether this device can,
    // and puts the extension in `reqs.extensions` when it can; both feature
    // bits have to be enabled here, and pixelforge told with
    // `declare_unified_image_layouts` below, because Vulkan cannot be asked
    // afterwards which features a device was created with.
    let mut unified = vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR::default()
        .unified_image_layouts(true)
        .unified_image_layouts_video(true);
    let mut device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&ext_ptrs)
        .push(&mut sync2)
        .push(&mut timeline);
    if reqs.unified_image_layouts {
        device_info = device_info.push(&mut unified);
    }
    let device = unsafe { instance.create_device(physical_device, &device_info, None)? };

    // --- Hand the app's device to pixelforge ---
    let (transfer, compute) = (spare(reqs.queues.transfer), spare(reqs.queues.compute));
    println!("pixelforge gets transfer queue {transfer:?} and compute queue {compute:?}");
    let builder = builder
        .with_transfer_queue(transfer)
        .with_compute_queue(compute);
    let builder = if reqs.unified_image_layouts {
        builder.declare_unified_image_layouts()
    } else {
        println!("device has no unified image layouts; frames will be copied out");
        builder
    };
    let context = builder.build_from_existing_decode(
        entry.clone(),
        instance.clone(),
        physical_device,
        device.clone(),
    )?;

    let stream = std::fs::read(&input_path)?;
    let mut output = output_path.as_ref().map(File::create).transpose()?;
    let mut readback = output
        .is_some()
        .then(|| Readback::new(&context))
        .transpose()?;
    let mut decoder = Decoder::new(context, DecodeConfig::h264().with_byte_stream())?;
    let mut count = 0usize;

    decode_stream(&mut decoder, &stream, |frame| {
        write_nv12(&frame, &mut readback, &mut output)?;
        count += 1;
        Ok(())
    })?;
    println!("Decoded {} frames on the adopted device", count);

    // The context borrowed the device, so everything holding Vulkan objects has
    // to go before the device does. That includes the readback helper, which
    // owns a command pool, a fence and a staging buffer of its own.
    drop(decoder);
    drop(readback);
    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    Ok(())
}
