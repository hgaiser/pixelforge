//! Contexts built one after another on one shared instance.
//!
//! An application that builds a context per stream gives them one
//! [`VideoInstance`], so the Vulkan driver stays loaded. Each context still
//! gets a device of its own, and dropping it destroys the device with all it
//! still holds, such as the semaphores an encoder retires when it is dropped.
//! Without a shared instance, NVIDIA's driver fails `vkCreateInstance` after
//! about 26 reloads.
//!
//! Ignored by default: requires a Vulkan Video device. Run with
//! `cargo test --test shared_instance -- --ignored`.

#[allow(dead_code)]
mod common;

use pixelforge::{
    Codec, EncodeConfig, Encoder, Frame, RateControl, VideoContextBuilder, YuvLayout, YuvUploader,
};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
/// More contexts than NVIDIA's driver survives with an instance each.
const CONTEXTS: usize = 40;
/// Frames encoded per context.
const FRAMES: u32 = 3;

/// The file descriptors the process has open on device nodes.
fn device_fds() -> usize {
    std::fs::read_dir("/proc/self/fd")
        .expect("read /proc/self/fd")
        .filter_map(|entry| std::fs::read_link(entry.ok()?.path()).ok())
        .filter(|target| target.starts_with("/dev/"))
        .count()
}

/// Build a context on `instance`, encode a few frames with it and drop it.
fn encode_once(instance: &pixelforge::VideoInstance) -> Result<(), Box<dyn std::error::Error>> {
    let context = VideoContextBuilder::new()
        .require_encode(Codec::H265)
        .build_on(instance)?;
    let config = EncodeConfig::new(Codec::H265, WIDTH, HEIGHT)
        .with_frame_rate(30, 1)
        .with_rate_control(RateControl::Cbr { bitrate: 500_000 });
    let mut encoder = Encoder::new(context, config)?;
    let mut uploader = YuvUploader::for_encoder(&encoder)?;
    let frame = vec![128u8; (WIDTH * HEIGHT * 3 / 2) as usize];
    for _ in 0..FRAMES {
        let slot = encoder.input_slot()?;
        uploader.upload_to(&slot, YuvLayout::I420, &frame)?;
        let packet = pollster::block_on(encoder.submit(Frame::input(slot))?)?;
        assert!(!packet.data.is_empty(), "an empty packet");
    }
    encoder.flush()?;
    Ok(())
}

/// Contexts one after another on one instance all work, and each frees what
/// its device held: the process does not keep more device files open.
#[test]
#[ignore = "requires a Vulkan Video device"]
fn contexts_one_after_another_share_an_instance() -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();

    let instance = VideoContextBuilder::new()
        .app_name("Shared instance")
        .build_instance()?;
    // The driver may keep files open for the instance once a device existed.
    encode_once(&instance)?;
    let baseline = device_fds();

    for run in 1..CONTEXTS {
        encode_once(&instance).map_err(|e| format!("context {run}: {e}"))?;
        assert_eq!(
            device_fds(),
            baseline,
            "context {run} left device files open"
        );
    }
    Ok(())
}

/// A context keeps its instance alive: the caller may drop its handle first.
#[test]
#[ignore = "requires a Vulkan Video device"]
fn a_context_outlives_the_callers_instance_handle() -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();

    let instance = VideoContextBuilder::new().build_instance()?;
    let context = VideoContextBuilder::new()
        .require_encode(Codec::H265)
        .build_on(&instance)?;
    drop(instance);

    let config = EncodeConfig::new(Codec::H265, WIDTH, HEIGHT).with_frame_rate(30, 1);
    let mut encoder = Encoder::new(context, config)?;
    let mut uploader = YuvUploader::for_encoder(&encoder)?;
    let slot = encoder.input_slot()?;
    uploader.upload_to(
        &slot,
        YuvLayout::I420,
        &vec![128u8; (WIDTH * HEIGHT * 3 / 2) as usize],
    )?;
    let packet = pollster::block_on(encoder.submit(Frame::input(slot))?)?;
    assert!(!packet.data.is_empty(), "an empty packet");
    Ok(())
}
