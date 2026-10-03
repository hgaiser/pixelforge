//! Device selection must fail cleanly, by a typed error rather than a panic or
//! a hang, when no device can decode.
//!
//! Three environments have to be told apart:
//!
//! - a video-capable GPU: the `supports_decode` contract must hold
//! - a driver without video queues (lavapipe, for instance): a typed
//!   [`PixelForgeError::NoSuitableDevice`]
//! - no Vulkan driver at all, which is the case on CI runners: nothing to test,
//!   so the test skips rather than reporting a failure it cannot act on
//!
//! Set `PIXELFORGE_REQUIRE_VULKAN=1` to turn that last case into a failure, on a
//! machine that is supposed to have a driver.

use pixelforge::encoder::Codec;
use pixelforge::error::PixelForgeError;
use pixelforge::vulkan::VideoContextBuilder;

/// Whether the error means the loader found no driver, as opposed to finding one
/// with nothing suitable on it. Instance creation is the only stage that can
/// fail before any device has been looked at.
fn no_vulkan_driver(error: &PixelForgeError) -> bool {
    matches!(error, PixelForgeError::InstanceCreation(_))
}

/// Skip, or fail if the environment claims a driver should be present.
fn skip_without_driver(test: &str, error: &PixelForgeError) {
    assert!(
        std::env::var("PIXELFORGE_REQUIRE_VULKAN").is_err(),
        "{test}: PIXELFORGE_REQUIRE_VULKAN is set but no Vulkan driver is usable: {error}"
    );
    eprintln!("skipping {test}: no Vulkan driver available ({error})");
}

fn build_decode_context() -> Result<pixelforge::vulkan::VideoContext, PixelForgeError> {
    VideoContextBuilder::new()
        .app_name("pixelforge-test")
        .require_decode(Codec::H264)
        .build()
}

#[test]
fn require_decode_fails_cleanly_without_video_support() {
    match build_decode_context() {
        // A real video-capable GPU: the contract must hold.
        Ok(context) => assert!(context.supports_decode(Codec::H264)),
        Err(error) if no_vulkan_driver(&error) => {
            skip_without_driver("require_decode_fails_cleanly_without_video_support", &error);
        }
        // A driver with no video queues: a typed verdict, not a panic.
        Err(PixelForgeError::NoSuitableDevice(_)) => {}
        Err(other) => panic!("expected NoSuitableDevice, got: {other}"),
    }
}

#[test]
fn context_creation_reaches_device_enumeration() {
    // Guards the test above against passing for the wrong reason: with a driver
    // present, the builder must get far enough to judge the devices, so any
    // error has to come from device selection rather than instance creation.
    match build_decode_context() {
        Ok(_) => {}
        Err(error) if no_vulkan_driver(&error) => {
            skip_without_driver("context_creation_reaches_device_enumeration", &error);
        }
        Err(PixelForgeError::NoSuitableDevice(_)) => {}
        Err(other) => panic!("expected to reach device selection, failed earlier with: {other}"),
    }
}

#[test]
fn compute_family_avoids_graphics_when_the_device_allows() {
    use ash::vk::QueueFlags;

    let context = match build_decode_context() {
        Ok(context) => context,
        Err(error) if no_vulkan_driver(&error) => {
            return skip_without_driver(
                "compute_family_avoids_graphics_when_the_device_allows",
                &error,
            );
        }
        Err(PixelForgeError::NoSuitableDevice(_)) => return,
        Err(other) => panic!("expected to reach device selection, failed earlier with: {other}"),
    };
    let families = unsafe {
        context
            .instance()
            .get_physical_device_queue_family_properties(context.physical_device())
    };
    let computes = |flags: QueueFlags| flags.contains(QueueFlags::COMPUTE);
    let graphics = |flags: QueueFlags| flags.contains(QueueFlags::GRAPHICS);

    let chosen = families[context.compute_queue_family() as usize].queue_flags;
    let has_compute_only = families
        .iter()
        .any(|family| computes(family.queue_flags) && !graphics(family.queue_flags));
    assert!(computes(chosen));
    assert_eq!(
        graphics(chosen),
        !has_compute_only,
        "the conversion must not share the graphics family when another one computes"
    );
}
