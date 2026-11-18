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
use common::{decode_stream, write_nv12};

use pixelforge::decoder::{DecodeConfig, Decoder, FramePoll};
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
    let mut readback = Some(pixelforge::YuvDownloader::new(&context)?);
    let mut output = Some(File::create(&output_path)?);
    let mut decoder = Decoder::new(context, DecodeConfig::new(codec).with_byte_stream())?;
    let stream = std::fs::read(&input)?;
    let decoded = decode_stream(&mut decoder, &stream, |frame| {
        write_nv12(&frame, &mut readback, &mut output)
    });
    // Film grain is an optional part of the AV1 decode profile. A device
    // without it reports the profile as not supported when the stream asks
    // for it, which is correct, so there is nothing to compare.
    if FILM_GRAIN.contains(&name)
        && let Err(e) = &decoded
        && matches!(
            e.downcast_ref::<pixelforge::PixelForgeError>(),
            Some(pixelforge::PixelForgeError::NoSuitableDevice(_))
        )
    {
        eprintln!("skipping {name}: {e}");
        return Ok(());
    }
    decoded?;
    drop(output);

    assert!(
        std::fs::read(&reference_path)? == std::fs::read(&output_path)?,
        "{name}: decoded frames differ from ffmpeg (see {})",
        output_path.display()
    );
    Ok(())
}

/// The streams that use film grain, which a device need not support.
const FILM_GRAIN: &[&str] = &["av1/aom_grain.obu", "av1/svt_grain.obu"];

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
    h264_bpyramid: Codec::H264, "bpyramid.264", "nv12";
    h264_bstrict: Codec::H264, "bstrict.264", "nv12";
    h264_novui: Codec::H264, "novui.264", "nv12";
    h264_novui_poc2: Codec::H264, "novui_poc2.264", "nv12";
    h264_multislice: Codec::H264, "multislice.264", "nv12";
    h264_zerolatency: Codec::H264, "zerolatency.264", "nv12";
    h265_closedgop: Codec::H265, "closedgop.265", "nv12";
    h265_bpyramid: Codec::H265, "bpyramid.265", "nv12";
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
    av1_svt_hier: Codec::AV1, "av1/svt_hier.obu", "nv12";
}

/// Split an H.264 byte stream with one slice per picture into access units.
fn h264_access_units(stream: &[u8]) -> Vec<&[u8]> {
    // Offsets of every start code, counting a leading zero byte as part of it.
    let mut nals = Vec::new();
    let mut i = 0;
    while i + 3 <= stream.len() {
        if stream[i..i + 3] == [0, 0, 1] {
            let start = if i > 0 && stream[i - 1] == 0 {
                i - 1
            } else {
                i
            };
            nals.push((start, stream[i + 3] & 0x1f));
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut starts = Vec::new();
    let mut seen_slice = true;
    for (offset, nal_type) in nals {
        let is_slice = matches!(nal_type, 1 | 5);
        // The first unit after a slice opens the next access unit.
        if seen_slice {
            starts.push(offset);
        }
        seen_slice = is_slice;
    }
    let ends = starts.iter().skip(1).copied().chain([stream.len()]);
    starts
        .iter()
        .copied()
        .zip(ends)
        .map(|(start, end)| &stream[start..end])
        .collect()
}

/// A stream that cannot reorder must give each frame back before the next one
/// is fed: the reorder buffer may add no latency to it.
fn check_no_latency(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();
    let context = VideoContextBuilder::new()
        .app_name("pixelforge-decode-latency")
        .require_decode(Codec::H264)
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .build()?;
    let mut decoder = Decoder::new(context, DecodeConfig::new(Codec::H264))?;
    let stream = std::fs::read(Path::new("tests/data").join(name))?;
    let units = h264_access_units(&stream);
    assert_eq!(units.len(), 30, "{name}: access units");

    for (i, unit) in units.iter().enumerate() {
        decoder.decode(unit, i as u64)?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let frame = loop {
            match decoder.try_next_frame()? {
                FramePoll::Frame(frame) => break frame,
                FramePoll::Pending if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                other => panic!("{name}: no frame after access unit {i} ({other:?})"),
            }
        };
        assert_eq!(frame.pts(), i as u64, "{name}: frame order");
    }
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn h264_zerolatency_has_no_delay() -> Result<(), Box<dyn std::error::Error>> {
    check_no_latency("zerolatency.264")
}

/// The same stream without a VUI: POC type 2 tells the decoder that nothing
/// is reordered, where the level alone would allow a delay of several frames.
#[test]
#[ignore = "requires a Vulkan Video device"]
fn h264_novui_poc2_has_no_delay() -> Result<(), Box<dyn std::error::Error>> {
    check_no_latency("novui_poc2.264")
}

/// Compare replay after a discontinuity with a decoder started at the seek
/// point. Priming and replay use different PTS so an old frame cannot pass.
fn check_reset(
    codec: Codec,
    priming: &[u8],
    seek: &[u8],
    expected_count: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();
    let context = VideoContextBuilder::new()
        .app_name("pixelforge-decode-reset")
        .require_decode(codec)
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .build()?;
    let config = DecodeConfig::new(codec).with_byte_stream();
    let mut readback = pixelforge::YuvDownloader::new(&context)?;
    let mut fresh = Decoder::new(context.clone(), config.clone())?;
    let (mut sink, mut source) = Decoder::new(context, config)?.split();

    const BEFORE: u64 = 10;
    const AFTER: u64 = 20;
    sink.decode(priming, BEFORE)?;
    sink.reset()?;
    let FramePoll::Frame(old) = source.try_next_frame()? else {
        panic!("reset must leave submitted frames on the source");
    };
    let generation = old.generation();
    let old_pixels = readback.download(&old)?;
    assert_eq!(old.pts(), BEFORE);
    while let FramePoll::Frame(frame) = source.try_next_frame()? {
        assert_eq!(frame.pts(), BEFORE);
    }

    sink.decode(seek, AFTER)?;
    fresh.decode(seek, AFTER)?;
    sink.finish()?;
    fresh.finish()?;
    let mut count = 0;
    while let Some(frame) = pollster::block_on(source.next_frame())? {
        let expected = pollster::block_on(fresh.next_frame())?.expect("extra frame after reset");
        assert_eq!(frame.pts(), AFTER);
        assert_eq!(frame.display_order(), expected.display_order());
        assert_eq!(frame.generation(), generation);
        assert_eq!(frame.is_keyframe(), expected.is_keyframe());
        let actual = readback.download(&frame)?;
        let expected = readback.download(&expected)?;
        assert!(
            actual.y == expected.y && actual.uv == expected.uv,
            "frame {count}"
        );
        count += 1;
    }
    assert!(pollster::block_on(fresh.next_frame())?.is_none());
    assert!(count > 0);
    if let Some(expected) = expected_count {
        assert_eq!(count, expected);
    }
    let still_held = readback.download(&old)?;
    assert!(old_pixels.y == still_held.y && old_pixels.uv == still_held.uv);
    assert!(matches!(
        sink.reset(),
        Err(pixelforge::PixelForgeError::InvalidInput(_))
    ));
    Ok(())
}

macro_rules! reset_tests {
    ($($test:ident: $codec:expr, $prime:literal, $seek:literal, $count:expr;)*) => {
        $(
            #[test]
            #[ignore = "requires a Vulkan Video device"]
            fn $test() -> Result<(), Box<dyn std::error::Error>> {
                check_reset(
                    $codec,
                    include_bytes!($prime),
                    include_bytes!($seek),
                    $count,
                )
            }
        )*
    };
}

reset_tests! {
    reset_h264_bframes: Codec::H264, "data/bframes.264", "data/bframes.264", None;
    reset_h265_closedgop: Codec::H265, "data/closedgop.265", "data/closedgop.265", None;
    reset_h265_cra: Codec::H265, "data/opengop.265", "data/cra_start.265", Some(6);
    reset_h265_main10: Codec::H265, "data/main10.265", "data/main10.265", None;
    reset_av1_altref: Codec::AV1, "data/av1/aom_altref.obu", "data/av1/aom_altref.obu", None;
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn reset_h264_drops_reorder_and_keeps_parameter_sets() -> Result<(), Box<dyn std::error::Error>> {
    use pixelforge::decoder::DecodeStatus::*;

    common::init_logging();
    let units = h264_access_units(include_bytes!("data/bframes.264"));
    // Skip SPS, PPS and SEI to feed just the IDR slice after the first decode.
    let key = units[0]
        .windows(4)
        .position(|w| w[..3] == [0, 0, 1] && w[3] & 0x1f == 5)
        .map(|i| &units[0][i..])
        .expect("first access unit has an IDR slice");
    for force_copy in [false, true] {
        let mut builder = VideoContextBuilder::new()
            .app_name("pixelforge-decode-reset")
            .require_decode(Codec::H264)
            .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok());
        if force_copy {
            builder = builder.without_unified_image_layouts();
        }
        let context = builder.build()?;
        let mut decoder = Decoder::new(context, DecodeConfig::new(Codec::H264))?;
        decoder.reset()?;
        assert_eq!(decoder.decode(units[0], 0)?, Decoded);
        decoder.reset()?;
        assert!(matches!(decoder.try_next_frame()?, FramePoll::Pending));
        assert_eq!(decoder.decode(units[1], 0)?, NeedsKeyframe);

        // The fixture holds two frames for reordering. Vary the number of
        // decodes before each reset, without waiting for the last submission.
        let mut copied_images = std::collections::HashSet::new();
        for seek in 1..=300u64 {
            let following = seek as usize % 8;
            assert_eq!(decoder.decode(key, seek)?, Decoded);
            for unit in &units[1..=following] {
                assert_eq!(decoder.decode(unit, seek)?, Decoded);
            }
            decoder.reset()?;
            let mut sent = 0;
            while let FramePoll::Frame(frame) = decoder.try_next_frame()? {
                assert_eq!(frame.pts(), seek);
                assert_eq!(frame.generation(), 1);
                if force_copy {
                    copied_images.insert(frame.image());
                }
                sent += 1;
            }
            assert_eq!(sent, (1 + following).saturating_sub(2), "seek {seek}");
        }
        // Eight pictures is the most any iteration can put in the pool.
        assert!(copied_images.len() <= 8, "copy pool grew across resets");
    }
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn flush_delivers_tail_and_allows_reuse() -> Result<(), Box<dyn std::error::Error>> {
    use pixelforge::decoder::DecodeStatus;

    common::init_logging();
    let context = VideoContextBuilder::new()
        .app_name("pixelforge-decode-flush")
        .require_decode(Codec::H264)
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .build()?;
    let config = DecodeConfig::new(Codec::H264).with_byte_stream();
    let mut readback = pixelforge::YuvDownloader::new(&context)?;
    let mut fresh = Decoder::new(context.clone(), config.clone())?;
    let mut decoder = Decoder::new(context, config)?;
    let stream = include_bytes!("data/bframes.264");
    let units = h264_access_units(stream);
    fresh.decode(stream, 0)?;
    fresh.finish()?;
    let mut reference = Vec::new();
    while let Some(frame) = pollster::block_on(fresh.next_frame())? {
        reference.push((frame.display_order(), readback.download(&frame)?));
    }
    assert_eq!(reference.len(), units.len());

    decoder.flush()?;
    assert!(matches!(decoder.try_next_frame()?, FramePoll::Pending));
    for pts in [10, 20] {
        if pts == 20 {
            assert_eq!(
                decoder.decode(&stream[units[0].len()..], 1)?,
                DecodeStatus::NeedsKeyframe
            );
            decoder.flush()?;
            assert!(matches!(decoder.try_next_frame()?, FramePoll::Pending));
        }
        decoder.decode(stream, pts)?;
        decoder.flush()?;
        for (order, expected) in &reference {
            let FramePoll::Frame(frame) = decoder.try_next_frame()? else {
                panic!("flushed frame {order} must be ready when flush returns");
            };
            assert_eq!(frame.pts(), pts);
            assert_eq!(frame.display_order(), *order);
            assert_eq!(frame.generation(), 1);
            let actual = readback.download(&frame)?;
            assert!(actual.y == expected.y && actual.uv == expected.uv);
        }
        assert!(matches!(decoder.try_next_frame()?, FramePoll::Pending));
    }
    decoder.finish()?;
    assert!(matches!(
        decoder.flush(),
        Err(pixelforge::PixelForgeError::InvalidInput(_))
    ));
    assert!(pollster::block_on(decoder.next_frame())?.is_none());
    Ok(())
}
