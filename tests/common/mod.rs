//! The integration tests' fixtures, decode loop and scratch space.
//!
//! Decoded frames are read back with [`YuvDownloader`], from the thread that
//! drives the decoder: both submit to the transfer queue.

pub mod source;

use ash::vk;
use pixelforge::YuvDownloader;
use pixelforge::decoder::{DecodedFrame, Decoder, FramePoll};
use std::fs::File;
use std::io::Write;

/// Show pixelforge's log, validation messages included, on the test's output.
///
/// pixelforge sends the validation layer's findings to `tracing`, and without
/// a subscriber they go nowhere: a test that enabled validation and printed
/// nothing proved nothing. Call before building a context. The level comes
/// from `RUST_LOG`, `warn` when unset, so validation errors always show.
/// Output is captured like any other test output: run with `--nocapture` to
/// see a passing test's.
pub fn init_logging() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        keep_driver_loaded();
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .try_init();
    });
}

/// Keep the Vulkan driver loaded for the life of the test process.
///
/// The loader loads a driver's libraries when an instance is created and
/// unloads them with the last instance, so a binary whose tests each build a
/// context loads the driver once per test. The NVIDIA driver (615.78.08) can
/// be loaded only about 26 times in one process: after that the loader finds
/// no driver. One instance that is never destroyed holds the libraries.
fn keep_driver_loaded() {
    let Ok(entry) = (unsafe { ash::Entry::load() }) else {
        return;
    };
    let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    // The instance is never destroyed, and the entry must not unload the
    // loader under it.
    if unsafe { entry.create_instance(&create_info, None) }.is_ok() {
        std::mem::forget(entry);
    }
}

/// Bytes handed to the decoder per call, standing in for a network read.
pub const CHUNK_SIZE: usize = 64 * 1024;

/// Drive `decoder` over `stream`, calling `on_frame` for every frame it emits.
///
/// Frames are polled after each chunk and again after `finish`, so the
/// in-stream and end-of-stream paths share one callback. `NeedsKeyframe` is not
/// special-cased: polling then is a no-op when nothing is ready.
pub fn decode_stream<F>(
    decoder: &mut Decoder,
    stream: &[u8],
    mut on_frame: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut(DecodedFrame) -> Result<(), Box<dyn std::error::Error>>,
{
    for (i, chunk) in stream.chunks(CHUNK_SIZE).enumerate() {
        decoder.decode(chunk, i as u64)?;
        while let FramePoll::Frame(frame) = decoder.try_next_frame()? {
            on_frame(frame)?;
        }
    }
    // End of stream: decodes the trailing picture, emits what reordering held
    // back, and closes the source.
    decoder.finish()?;
    while let Some(frame) = pollster::block_on(decoder.next_frame())? {
        on_frame(frame)?;
    }
    Ok(())
}

/// Write one decoded frame as NV12 to `output`, when an output is set.
pub fn write_nv12(
    frame: &DecodedFrame,
    readback: &mut Option<YuvDownloader>,
    output: &mut Option<File>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let (Some(file), Some(readback)) = (output.as_mut(), readback.as_mut()) {
        let data = readback.download(frame)?;
        file.write_all(&data.y)?;
        file.write_all(&data.uv)?;
    }
    Ok(())
}

/// A scratch directory for one test, under Cargo's target temp dir so
/// `cargo clean` removes it. Created if missing.
pub fn scratch_dir(name: &str) -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Generate a raw YUV fixture with ffmpeg if it is not already present.
pub fn ensure_test_data(
    width: u32,
    height: u32,
    pix_fmt: &str,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if std::path::Path::new(path).exists() {
        return Ok(());
    }
    println!("Generating {path}...");
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc=duration=1:size={width}x{height}:rate=30"),
            "-pix_fmt",
            pix_fmt,
            "-f",
            "rawvideo",
            "-y",
            path,
        ])
        .output()?;
    if !status.status.success() {
        return Err(format!(
            "failed to generate test data: {}",
            String::from_utf8_lossy(&status.stderr)
        )
        .into());
    }
    Ok(())
}
