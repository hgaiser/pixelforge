//! Decode an H.264, H.265 or AV1 stream to raw YUV.
//!
//! The codec comes from the file extension: `.264`, `.265`/`.hevc`, `.obu`
//! (AV1 OBUs) or `.ivf` (AV1 in IVF).
//!
//! Usage:
//!   cargo run --example decode -- input.264 [output.yuv]
//!
//! Produces the same layout as `ffmpeg -i input.264 -pix_fmt nv12 out.yuv`,
//! so the output can be compared directly:
//!
//!   ffmpeg -i input.264 -pix_fmt nv12 reference.yuv
//!   cmp reference.yuv output.yuv

use std::fs::File;

#[allow(dead_code)]
mod common;
use common::{Readback, decode_input, write_nv12};

use pixelforge::decoder::Decoder;
use pixelforge::vulkan::VideoContextBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args().skip(1);
    let input_path = args.next().unwrap_or_else(|| {
        eprintln!("usage: decode <input.264|.265|.obu|.ivf> [output.yuv]");
        std::process::exit(1);
    });
    let output_path = args.next();
    let (codec, input) = common::open_input(&input_path)?;

    // PIXELFORGE_VALIDATION=1 enables the Vulkan validation layers, which
    // report the exact offending call for any API misuse.
    let validation = std::env::var("PIXELFORGE_VALIDATION").is_ok();
    // Set PIXELFORGE_NO_UNIFIED_LAYOUTS=1 to exercise the copying fallback on
    // hardware that would otherwise take the zero-copy path.
    let mut builder = VideoContextBuilder::new()
        .app_name("pixelforge-decode")
        .require_decode(codec)
        .enable_validation(validation);
    if std::env::var("PIXELFORGE_NO_UNIFIED_LAYOUTS").is_ok() {
        builder = builder.without_unified_image_layouts();
    }
    let context = builder.build()?;

    println!(
        "Device: {}",
        unsafe { std::ffi::CStr::from_ptr(context.device_properties().device_name.as_ptr()) }
            .to_string_lossy()
    );

    // Frames come out in presentation order, ready to show without sorting,
    // and on hardware with unified image layouts without ever being copied: a
    // picture waiting its turn stays pinned in the DPB slot it was decoded
    // into, and so does the one handed to this loop.
    //
    // 64 KB of byte stream holds several coded frames, and every frame a
    // `decode` call emits is outstanding at once. The default budget of 4 is
    // sized for one frame per call, so raise it: past the budget the decoder
    // copies pictures out instead of handing over its own images, which is
    // still correct but gives up the zero-copy path.
    let config = common::decode_config(codec, &input).with_output_depth(8);
    let mut output = output_path.as_ref().map(File::create).transpose()?;
    // Readback lives in the consumer, not in pixelforge: the decoder hands over
    // a GPU image and a renderer would sample it in place. See examples/common.
    // `VideoContext` is an `Arc` handle, so cloning it shares the device.
    let mut readback = output
        .is_some()
        .then(|| Readback::new(&context))
        .transpose()?;
    let mut decoder = Decoder::new(context, config)?;

    let start = std::time::Instant::now();
    let mut frame_count = 0usize;
    let mut last_info: Option<(u32, u32, i32, bool)> = None;
    let mut last_generation: Option<u64> = None;

    // One thread drives both halves: feed a chunk, then take whatever has
    // become ready. `Pending` means the GPU is still working on frames in
    // flight, not that there are none, so they are collected on a later pass.
    // A renderer would instead `split()` the decoder and await `next_frame` on
    // its own thread, which is what the `Decoder::split` docs show.
    // `common::decode_input` owns that loop.
    decode_input(&mut decoder, &input, |frame| {
        write_nv12(&frame, &mut readback, &mut output)?;
        // A rebuilt session means a new set of images, so anything caching
        // per-image state has to notice. Reporting it here is what makes the
        // field visible in an example rather than only in the docs.
        if last_generation.is_some_and(|g| g != frame.generation) {
            println!(
                "  generation {} -> {} at frame {}: the decoder rebuilt its images",
                last_generation.unwrap(),
                frame.generation,
                frame_count
            );
        }
        last_generation = Some(frame.generation);
        last_info = Some((
            frame.width,
            frame.height,
            frame.display_order,
            frame.is_keyframe,
        ));
        frame_count += 1;
        // Dropping the frame here is what hands its DPB slot back. Holding on
        // to frames is what pushes the decoder onto the copying path.
        Ok(())
    })?;
    let decode_time = start.elapsed();

    println!(
        "Decoded {} frames in {:.1?} ({:.1} fps)",
        frame_count,
        decode_time,
        frame_count as f64 / decode_time.as_secs_f64()
    );
    if let Some((w, h, order, key)) = last_info {
        println!(
            "Last frame: {}x{} display_order={} keyframe={}",
            w, h, order, key
        );
    }
    if let Some(path) = output_path {
        println!("Wrote raw frames to {}", path);
    }

    Ok(())
}
