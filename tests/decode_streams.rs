//! Decode third-party streams and compare against ffmpeg's software decoder,
//! which must match byte for byte.
//!
//! The streams are the fixtures in `tests/data`; how each was made is in the
//! parser tests that use them. Each is fed as a byte stream that the decoder
//! frames itself, and takes whichever output path the device offers.
//!
//! `base.264` is left out: on NVIDIA (615.78.08) it differs from ffmpeg from
//! frame 17, the first after its `frame_num` wraps.
//!
//! Ignored by default: requires a Vulkan Video device and ffmpeg. Run with
//! `cargo test -- --ignored`.

use std::fs::File;
use std::path::Path;

#[allow(dead_code)]
mod common;
use common::{Readback, decode_stream, write_nv12};

use pixelforge::decoder::{DecodeConfig, Decoder};
use pixelforge::{Codec, VideoContextBuilder};

/// Decode `name` from `tests/data` and compare it against ffmpeg.
fn check(codec: Codec, name: &str, pix_fmt: &str) -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();
    let input = Path::new("tests/data").join(name);
    let dir = common::scratch_dir("decode_streams");
    let file = name.replace('/', "_");
    let output_path = dir.join(format!("{file}.yuv"));
    let reference_path = dir.join(format!("{file}.ref.yuv"));

    // dav1d is the AV1 reference. `-noautoscale` keeps each frame at its own
    // size, where ffmpeg would otherwise scale a stream that changes size to
    // its first one.
    let mut ffmpeg = std::process::Command::new("ffmpeg");
    ffmpeg.args(["-hide_banner", "-loglevel", "error", "-y"]);
    if codec == Codec::AV1 {
        ffmpeg.args(["-c:v", "libdav1d"]);
    }
    let status = ffmpeg
        .arg("-i")
        .arg(&input)
        .args(["-noautoscale", "-pix_fmt", pix_fmt, "-f", "rawvideo"])
        .arg(&reference_path)
        .status()?;
    assert!(status.success(), "ffmpeg failed to decode {name}");

    let context = VideoContextBuilder::new()
        .app_name("pixelforge-decode-streams")
        .require_decode(codec)
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .build()?;
    let mut readback = Some(Readback::new(&context)?);
    let mut output = Some(File::create(&output_path)?);
    let mut decoder = Decoder::new(context, DecodeConfig::new(codec).with_byte_stream())?;
    let stream = std::fs::read(&input)?;
    decode_stream(&mut decoder, &stream, |frame| {
        write_nv12(&frame, &mut readback, &mut output)
    })?;
    drop(output);

    assert!(
        std::fs::read(&reference_path)? == std::fs::read(&output_path)?,
        "{name}: decoded frames differ from ffmpeg (see {})",
        output_path.display()
    );
    Ok(())
}

macro_rules! stream_tests {
    ($($test:ident: $codec:expr, $name:literal, $pix_fmt:literal;)*) => {
        $(
            #[test]
            #[ignore = "requires a Vulkan Video device and ffmpeg"]
            fn $test() -> Result<(), Box<dyn std::error::Error>> {
                check($codec, $name, $pix_fmt)
            }
        )*
    };
}

stream_tests! {
    h264_bframes: Codec::H264, "bframes.264", "nv12";
    h264_multislice: Codec::H264, "multislice.264", "nv12";
    h264_zerolatency: Codec::H264, "zerolatency.264", "nv12";
    h265_closedgop: Codec::H265, "closedgop.265", "nv12";
    h265_cra_start: Codec::H265, "cra_start.265", "nv12";
    h265_main10: Codec::H265, "main10.265", "p010le";
    h265_nvenc: Codec::H265, "nvenc.265", "nv12";
    h265_opengop: Codec::H265, "opengop.265", "nv12";
    h265_scaling: Codec::H265, "scaling.265", "nv12";
    h265_slices: Codec::H265, "slices.265", "nv12";
    h265_temporal: Codec::H265, "temporal.265", "nv12";
    av1_altref: Codec::AV1, "av1/aom_altref.obu", "nv12";
    av1_tiles: Codec::AV1, "av1/aom_tiles.obu", "nv12";
    av1_tilegroups: Codec::AV1, "av1/aom_tilegroups.obu", "nv12";
    av1_errres: Codec::AV1, "av1/aom_errres.obu", "nv12";
    av1_superres: Codec::AV1, "av1/aom_superres.obu", "nv12";
    av1_resize: Codec::AV1, "av1/aom_resize.obu", "nv12";
    av1_10bit: Codec::AV1, "av1/aom_10bit.obu", "p010le";
    av1_grain: Codec::AV1, "av1/aom_grain.obu", "nv12";
    av1_svt_grain: Codec::AV1, "av1/svt_grain.obu", "nv12";
}
