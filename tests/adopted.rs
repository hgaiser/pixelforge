//! Pixelforge on a device the caller created, from an instance that asked for
//! Vulkan 1.1.
//!
//! An application that shares its device with pixelforge chose that device's
//! API version for its own reasons, and plenty choose less than 1.3. On such a
//! device the core 1.2 and 1.3 entry points are null, so anything pixelforge
//! reaches that way crashes. These tests build exactly that device, hand
//! pixelforge a spare queue where the hardware has one, and check the output
//! against a context pixelforge created itself on the same GPU.

use ash::vk;
use ash::vk::TaggedStructure;
#[allow(dead_code)]
mod common;
use common::source::create_src_image;
use common::{Readback, decode_stream};

use pixelforge::decoder::{DecodeConfig, Decoder};
use pixelforge::encoder::Codec;
use pixelforge::vulkan::{DeviceQueue, DeviceRequirements, VideoContext, VideoContextBuilder};
use pixelforge::{
    ColorConverter, ColorConverterConfig, ColorRange, ColorSpec, EncodeConfig, Encoder,
    InputFormat, OutputFormat, RateControlMode, TimelinePoint,
};

/// An instance and device the test owns, destroyed on drop after everything
/// built on them.
struct AppDevice {
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
}

impl Drop for AppDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn validation_requested() -> bool {
    std::env::var("PIXELFORGE_VALIDATION").is_ok()
}

/// Create a Vulkan 1.1 instance, with validation when asked for.
fn instance_1_1() -> Result<(ash::Entry, ash::Instance), Box<dyn std::error::Error>> {
    let entry = unsafe { ash::Entry::load()? };
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"pixelforge-adopted-test")
        .api_version(vk::API_VERSION_1_1);
    let validation = c"VK_LAYER_KHRONOS_validation";
    let has_validation = unsafe { entry.enumerate_instance_layer_properties()? }
        .iter()
        .any(|l| unsafe { std::ffi::CStr::from_ptr(l.layer_name.as_ptr()) } == validation);
    let layers = if validation_requested() && has_validation {
        vec![validation.as_ptr()]
    } else {
        Vec::new()
    };
    let info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layers);
    let instance = unsafe { entry.create_instance(&info, None)? };
    Ok((entry, instance))
}

/// Create the application's device from `reqs`, keeping queue 0 of every
/// family for itself wherever the family has a second queue.
fn create_app_device(
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    reqs: &DeviceRequirements,
) -> Result<AppDevice, Box<dyn std::error::Error>> {
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
    let ext_ptrs: Vec<_> = reqs.extensions.iter().map(|e| e.as_ptr()).collect();

    // The per-feature structs, since this device is not 1.3: the promoted
    // struct types are the same types under their extension names.
    let features = reqs.features;
    let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default()
        .synchronization2(features.synchronization2);
    let mut timeline = vk::PhysicalDeviceTimelineSemaphoreFeatures::default()
        .timeline_semaphore(features.timeline_semaphore);
    let mut ycbcr = vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default()
        .sampler_ycbcr_conversion(features.sampler_ycbcr_conversion);
    let mut ycbcr_444 =
        vk::PhysicalDeviceYcbcr2Plane444FormatsFeaturesEXT::default().ycbcr2plane444_formats(true);
    let mut av1 = vk::PhysicalDeviceVideoEncodeAV1FeaturesKHR::default().video_encode_av1(true);
    let mut rgb = vk::PhysicalDeviceVideoEncodeRgbConversionFeaturesVALVE::default()
        .video_encode_rgb_conversion(true);
    let mut unified = vk::PhysicalDeviceUnifiedImageLayoutsFeaturesKHR::default()
        .unified_image_layouts(true)
        .unified_image_layouts_video(true);
    let mut info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&ext_ptrs)
        .push(&mut sync2)
        .push(&mut timeline)
        .push(&mut ycbcr);
    if features.ycbcr_2plane_444_formats {
        info = info.push(&mut ycbcr_444);
    }
    if features.video_encode_av1 {
        info = info.push(&mut av1);
    }
    if features.video_encode_rgb_conversion {
        info = info.push(&mut rgb);
    }
    if reqs.unified_image_layouts {
        info = info.push(&mut unified);
    }
    let device = unsafe { instance.create_device(physical_device, &info, None)? };
    Ok(AppDevice {
        entry,
        instance,
        physical_device,
        device,
    })
}

/// The last queue this test created in `family`: a spare where there are two.
fn spare(app: &AppDevice, family: u32) -> DeviceQueue {
    let props = unsafe {
        app.instance
            .get_physical_device_queue_family_properties(app.physical_device)
    };
    DeviceQueue::new(family, props[family as usize].queue_count.min(2) - 1)
}

/// Decode `stream` on `context` and read every frame back.
fn decode_all(
    context: &VideoContext,
    stream: &[u8],
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let mut readback = Readback::new(context)?;
    let mut decoder = Decoder::new(context.clone(), DecodeConfig::h264().with_byte_stream())?;
    let mut frames = Vec::new();
    decode_stream(&mut decoder, stream, |frame| {
        let data = readback.read(&frame)?;
        frames.push([data.y, data.uv].concat());
        Ok(())
    })?;
    drop(decoder);
    drop(readback);
    Ok(frames)
}

/// The GPU a context pixelforge created for itself would pick, found on
/// `instance` by name so an adopted context can be compared against it.
fn same_gpu(instance: &ash::Instance, own: &VideoContext) -> vk::PhysicalDevice {
    let name = own.device_properties().device_name;
    unsafe { instance.enumerate_physical_devices() }
        .expect("physical devices")
        .into_iter()
        .find(|&pd| unsafe { instance.get_physical_device_properties(pd) }.device_name == name)
        .expect("the GPU the own context used")
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn decode_on_a_vulkan_1_1_device() -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();
    let stream = std::fs::read("tests/data/bframes.264")?;

    let own = VideoContextBuilder::new()
        .enable_validation(validation_requested())
        .require_decode(Codec::H264)
        .build()?;
    let expected = decode_all(&own, &stream)?;

    let (entry, instance) = instance_1_1()?;
    let builder = VideoContextBuilder::new().require_decode(Codec::H264);
    let physical_device = same_gpu(&instance, &own);
    drop(own);
    let reqs = builder.decode_device_requirements(&entry, &instance, physical_device)?;
    let app = create_app_device(entry, instance, physical_device, &reqs)?;

    let mut builder = builder
        .with_decode_queue(spare(&app, reqs.queues.decode.unwrap()))
        .with_transfer_queue(spare(&app, reqs.queues.transfer))
        .with_compute_queue(spare(&app, reqs.queues.compute));
    if reqs.unified_image_layouts {
        builder = builder.declare_unified_image_layouts();
    }
    let context = builder.build_from_existing_decode(
        app.entry.clone(),
        app.instance.clone(),
        app.physical_device,
        app.device.clone(),
    )?;
    let actual = decode_all(&context, &stream)?;
    drop(context);

    assert_eq!(actual.len(), expected.len(), "frame count");
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert!(a == e, "frame {i} differs from the own-context decode");
    }
    Ok(())
}

const ENCODE_WIDTH: u32 = 320;
const ENCODE_HEIGHT: u32 = 240;
const ENCODE_FRAMES: u32 = 8;

/// A BGRA frame that moves with `n`, so the encoder has motion to code.
fn moving_frame(n: u32) -> Vec<u8> {
    let mut data = Vec::with_capacity((ENCODE_WIDTH * ENCODE_HEIGHT * 4) as usize);
    for y in 0..ENCODE_HEIGHT {
        for x in 0..ENCODE_WIDTH {
            let (sx, sy) = (x + n * 3, y + n);
            let r = (sx ^ sy) as u8;
            let g = sx.wrapping_mul(2).wrapping_add(sy) as u8;
            let b = (sy * 3) as u8;
            data.extend_from_slice(&[b, g, r, 255]);
        }
    }
    data
}

/// How [`encode_all`] hands each frame from the converter to the encoder.
#[derive(Clone, Copy, PartialEq)]
enum Handoff {
    /// `convert`, which waits on the CPU, then `encode`.
    Cpu,
    /// `convert_async` waiting on a point the caller signals, then
    /// `encode_after` waiting on the conversion: ordered on the GPU, with
    /// nothing waited for on the CPU until the packets are collected.
    Gpu,
}

/// Convert and encode `ENCODE_FRAMES` frames as H.264 on `context`, the way a
/// capture pipeline would: an RGB image goes through the colour converter
/// straight into the encoder's input.
fn encode_all(
    context: &VideoContext,
    handoff: Handoff,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let device = context.device();
    let mut encoder = Encoder::new(
        context.clone(),
        EncodeConfig::h264(ENCODE_WIDTH, ENCODE_HEIGHT)
            .with_rate_control(RateControlMode::Cqp)
            .with_quality_level(22)
            .with_frame_rate(30, 1)
            .with_b_frames(0),
    )?;
    let mut converter = ColorConverter::new(
        context.clone(),
        ColorConverterConfig::new(
            ENCODE_WIDTH,
            ENCODE_HEIGHT,
            InputFormat::BGRA,
            OutputFormat::NV12,
            ColorSpec::Srgb,
            ColorSpec::Srgb,
            ColorRange::Limited,
        ),
    )?;

    // Stands in for the caller's own rendering: each frame's conversion waits
    // for a value the "renderer" signals once the frame is ready. The KHR
    // entry points, since the adopted device is Vulkan 1.1.
    let timeline_fns = ash::khr::timeline_semaphore::Device::load(context.instance(), device);
    let mut type_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let rendered = unsafe {
        device.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push(&mut type_info),
            None,
        )?
    };

    // Every source stays alive until the end: with a GPU handoff nothing says
    // a conversion has finished reading one until its packet is back.
    let mut sources = Vec::new();
    let mut pending = Vec::new();
    for n in 0..ENCODE_FRAMES {
        let src =
            unsafe { create_src_image(context, ENCODE_WIDTH, ENCODE_HEIGHT, &moving_frame(n))? };
        let target = encoder.input_image();
        let future = match handoff {
            Handoff::Cpu => {
                converter.convert(src.image, vk::ImageLayout::GENERAL, target)?;
                encoder.encode(target)?
            }
            Handoff::Gpu => {
                let ready = TimelinePoint::new(rendered, u64::from(n) + 1);
                let converted = converter.convert_async(
                    src.image,
                    vk::ImageLayout::GENERAL,
                    target,
                    &[ready],
                )?;
                // Signalled only after the conversion was submitted, so it
                // really has to wait.
                unsafe {
                    timeline_fns.signal_semaphore(
                        &vk::SemaphoreSignalInfo::default()
                            .semaphore(rendered)
                            .value(ready.value),
                    )?
                };
                encoder.encode_after(target, &[converted])?
            }
        };
        sources.push(src);
        pending.push(future);
    }
    let mut stream = Vec::new();
    for future in pending {
        stream.extend_from_slice(&pollster::block_on(future)?.data);
    }
    drop(converter);
    drop(encoder);
    drop(sources);
    unsafe { device.destroy_semaphore(rendered, None) };
    Ok(stream)
}

/// Luma PSNR between two NV12 frames of the encode size.
fn luma_psnr(a: &[u8], b: &[u8]) -> f64 {
    let n = (ENCODE_WIDTH * ENCODE_HEIGHT) as usize;
    let mse = a[..n]
        .iter()
        .zip(&b[..n])
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum::<f64>()
        / n as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn encode_on_a_vulkan_1_1_device() -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();
    let own = VideoContextBuilder::new()
        .enable_validation(validation_requested())
        .require_encode(Codec::H264)
        .require_decode(Codec::H264)
        .build()?;
    let expected = encode_all(&own, Handoff::Cpu)?;

    let (entry, instance) = instance_1_1()?;
    let builder = VideoContextBuilder::new().require_encode(Codec::H264);
    let physical_device = same_gpu(&instance, &own);
    let reqs = builder.encode_device_requirements(&entry, &instance, physical_device)?;
    assert!(
        reqs.extensions.contains(&ash::khr::push_descriptor::NAME),
        "the converter needs push descriptors, which every driver this runs on has"
    );
    let app = create_app_device(entry, instance, physical_device, &reqs)?;
    let context = builder
        .with_encode_queue(spare(&app, reqs.queues.encode.unwrap()))
        .with_transfer_queue(spare(&app, reqs.queues.transfer))
        .with_compute_queue(spare(&app, reqs.queues.compute))
        .build_from_existing_encode(
            app.entry.clone(),
            app.instance.clone(),
            app.physical_device,
            app.device.clone(),
        )?;
    let actual = encode_all(&context, Handoff::Gpu)?;
    drop(context);

    // Both streams decoded by the same decoder, so any difference is the
    // encode's, or the handoff's: an encode that did not wait for its
    // conversion would pick up the previous frame or a half-written one, far
    // below the floor here. The floor is not equality because not every
    // encoder is reproducible: RADV gives bit-identical streams, while ANV has
    // been seen as low as 47 dB between two runs of the same synchronous path.
    let expected_frames = decode_all(&own, &expected)?;
    let actual_frames = decode_all(&own, &actual)?;
    assert_eq!(actual_frames.len(), ENCODE_FRAMES as usize, "frame count");
    assert_eq!(expected_frames.len(), actual_frames.len(), "frame count");
    let psnrs: Vec<f64> = actual_frames
        .iter()
        .zip(&expected_frames)
        .map(|(a, e)| luma_psnr(a, e))
        .collect();
    println!(
        "adopted vs own encode, luma PSNR per frame: {:?}; streams bit-identical: {}",
        psnrs,
        actual == expected
    );
    for (i, psnr) in psnrs.iter().enumerate() {
        assert!(
            *psnr >= 40.0,
            "frame {i}: {psnr:.2} dB against the own-context encode"
        );
    }
    Ok(())
}
