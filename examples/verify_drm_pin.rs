//! Verify the encoder pins to the GPU backing a given DRM render node.
//!
//! Usage: cargo run --release --example verify_drm_pin -- /dev/dri/renderD129
//!
//! Requests a node, builds the video context, then re-queries the *selected*
//! device's VK_EXT_physical_device_drm render major/minor and asserts it matches
//! the request. This is the disambiguation guarantee for identical GPUs.

use ash::vk;
use ash::vk::TaggedStructure;
use pixelforge::VideoContextBuilder;
use std::ffi::CStr;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};

// glibc dev_t decode, matching VkPhysicalDeviceDrmPropertiesEXT render_major/minor.
fn decode(rdev: u64) -> (i64, i64) {
    let major = ((rdev >> 8) & 0xfff) | ((rdev >> 32) & !0xfff);
    let minor = (rdev & 0xff) | ((rdev >> 12) & !0xff);
    (major as i64, minor as i64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer().with_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            ),
        )
        .init();

    let node = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/dev/dri/renderD128".to_string());
    let path = PathBuf::from(&node);
    let want = decode(std::fs::metadata(&path)?.rdev());
    println!("Requested {} -> drm {}:{}", node, want.0, want.1);

    let context = VideoContextBuilder::new()
        .app_name("verify_drm_pin")
        .drm_render_node(Some(path))
        .build()?;

    let props = context.device_properties();
    let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();

    // Re-query the DRM render node of the *selected* physical device.
    let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
    let mut p2 = vk::PhysicalDeviceProperties2::default().push(&mut drm);
    unsafe {
        context
            .instance()
            .get_physical_device_properties2(context.physical_device(), &mut p2)
    };
    let got = (drm.render_major, drm.render_minor);

    println!(
        "Selected device: {} (vendor=0x{:x} device=0x{:x})",
        name, props.vendor_id, props.device_id
    );
    println!("Selected device drm: {}:{}", got.0, got.1);

    if got == want {
        println!("MATCH: encoder pinned to requested GPU");
        Ok(())
    } else {
        eprintln!(
            "MISMATCH: requested {}:{} but selected {}:{}",
            want.0, want.1, got.0, got.1
        );
        std::process::exit(1);
    }
}
