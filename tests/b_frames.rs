//! B-frames: what the device query says against what `Encoder::new` accepts,
//! and the calls that cut a group of B-frames short.
//!
//! Ignored by default: requires a Vulkan Video device. Run with
//! `cargo test -- --ignored`.

#[allow(dead_code)]
mod common;

use std::collections::VecDeque;

use pixelforge::{
    BFrameReferences, BFrameSupport, BitDepth, ChromaFormat, Codec, EncodeConfig, EncodeFuture,
    EncodedPacket, Encoder, Frame, FrameType, MAX_B_FRAMES, PixelForgeError, RateControl,
    VideoContextBuilder, YuvLayout, YuvUploader,
};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

/// `supports_b_frames` must agree with what `Encoder::new` accepts, for each
/// codec and kind of B-frames, on every device.
#[test]
#[ignore = "requires a Vulkan Video device"]
fn b_frame_query_matches_encoder_creation() -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();
    let context = VideoContextBuilder::new()
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .build()?;
    for codec in [Codec::H264, Codec::H265, Codec::AV1] {
        if !context.supports_encode(codec) {
            continue;
        }
        let support = context.supports_b_frames(codec, ChromaFormat::Yuv420, BitDepth::Eight);
        eprintln!("{codec:?}: {support:?}");
        for (references, needed) in [
            (BFrameReferences::Flat, BFrameSupport::Flat),
            (BFrameReferences::Pyramid, BFrameSupport::Pyramid),
        ] {
            let config = EncodeConfig::new(codec, 320, 240)
                .with_b_frames(3)
                .with_b_frame_references(references);
            let created = Encoder::new(context.clone(), config);
            assert_eq!(
                created.is_ok(),
                support >= needed,
                "{codec:?} {references:?}: query says {support:?}, creation gives {:?}",
                created.err()
            );
        }
    }
    Ok(())
}

/// More than `MAX_B_FRAMES` is an error on any device, before the device is
/// asked.
#[test]
#[ignore = "requires a Vulkan Video device"]
fn too_many_b_frames_is_an_error() -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();
    let context = VideoContextBuilder::new()
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .require_encode(Codec::H264)
        .build()?;
    let config = EncodeConfig::h264(320, 240).with_b_frames(MAX_B_FRAMES + 1);
    assert!(matches!(
        Encoder::new(context, config),
        Err(PixelForgeError::InvalidInput(_))
    ));
    Ok(())
}

/// An I420 frame with a pattern that moves with `index`.
fn pattern(index: usize) -> Vec<u8> {
    let (w, h) = (WIDTH as usize, HEIGHT as usize);
    let mut frame = vec![128u8; w * h * 3 / 2];
    for y in 0..h {
        for x in 0..w {
            frame[y * w + x] = (((x + 3 * index) / 8 + y / 8) * 16 % 256) as u8;
        }
    }
    frame
}

/// Take the packets that are sure to be ready: all but the last `keep`.
fn drain(
    pending: &mut VecDeque<EncodeFuture>,
    keep: usize,
    packets: &mut Vec<EncodedPacket>,
) -> Result<(), Box<dyn std::error::Error>> {
    while pending.len() > keep {
        packets.push(pollster::block_on(pending.pop_front().expect("non-empty"))?);
    }
    Ok(())
}

/// `request_idr`, `repeat`, reference frame invalidation and `flush` in the
/// middle of a group of B-frames: no frame is lost, the IDRs are where they
/// were asked for, and ffmpeg finds no error in the stream.
#[test]
#[ignore = "requires a Vulkan Video device and ffmpeg"]
fn control_calls_close_the_group() -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();
    const B_FRAMES: u32 = 2;
    // With a bitrate too: the rate control is told the GOP has B-frames.
    let cqp = RateControl::Cqp { qp: 26 };
    let cbr = RateControl::Cbr { bitrate: 1_000_000 };
    for (codec, extension, rate_control) in [
        (Codec::H264, "h264", cqp),
        (Codec::H264, "h264", cbr),
        (Codec::H265, "h265", cqp),
        (Codec::H265, "h265", cbr),
        (Codec::AV1, "obu", cqp),
        (Codec::AV1, "obu", cbr),
    ] {
        let context = VideoContextBuilder::new()
            .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
            .build()?;
        if !context.supports_encode(codec)
            || context.supports_b_frames(codec, ChromaFormat::Yuv420, BitDepth::Eight)
                == BFrameSupport::None
        {
            eprintln!("{codec:?}: no B-frames on this device, nothing to check");
            continue;
        }
        let config = EncodeConfig::new(codec, WIDTH, HEIGHT)
            .with_rate_control(rate_control)
            .with_gop_size(0)
            .with_b_frames(B_FRAMES);
        let mut encoder = Encoder::new(context, config)?;
        let mut input = YuvUploader::for_encoder(&encoder)?;

        let keep = 2 + B_FRAMES as usize;
        let mut pending = VecDeque::new();
        let mut packets = Vec::new();
        // The display order of the frames that must be IDRs.
        let mut idrs = vec![0u64];
        let mut submitted = 0u64;
        for i in 0..40 {
            match i {
                // Frame 10 waits as a B-frame: it becomes a P-frame.
                11 => {
                    encoder.request_idr();
                    idrs.push(submitted);
                }
                // Frame 19 waits. The repeated frame is encoded with it.
                20 => {
                    pending.push_back(encoder.repeat()?);
                    submitted += 1;
                }
                // With B-frames a lost frame is answered with an IDR.
                30 => {
                    encoder.invalidate_reference_frames(submitted - 2);
                    idrs.push(submitted);
                }
                // Everything submitted is encoded, and the stream goes on.
                35 => {
                    encoder.flush()?;
                    drain(&mut pending, 0, &mut packets)?;
                }
                _ => {}
            }
            input.upload(YuvLayout::I420, &pattern(i))?;
            pending.push_back(encoder.submit(Frame::image(input.source()))?);
            submitted += 1;
            drain(&mut pending, keep, &mut packets)?;
        }
        encoder.flush()?;
        drain(&mut pending, 0, &mut packets)?;

        let mut pts: Vec<u64> = packets.iter().map(|p| p.pts).collect();
        pts.sort_unstable();
        assert!(pts.iter().copied().eq(0..submitted), "{codec:?}: {pts:?}");
        let mut found: Vec<u64> = packets
            .iter()
            .filter(|p| p.is_key_frame())
            .map(|p| p.pts)
            .collect();
        found.sort_unstable();
        assert_eq!(found, idrs, "{codec:?}: IDRs");
        assert!(
            packets.iter().any(|p| p.frame_type == FrameType::B),
            "{codec:?}: no B-frame"
        );
        for pair in packets.windows(2) {
            assert!(pair[0].dts < pair[1].dts, "{codec:?}: dts must increase");
        }

        let dir = common::scratch_dir(&format!("b-frames-control-{extension}"));
        let path = dir.join(format!("out.{extension}"));
        let data: Vec<u8> = packets
            .iter()
            .flat_map(|p| p.data.iter().copied())
            .collect();
        std::fs::write(&path, data)?;
        let decoded = dir.join("out.yuv");
        let mut ffmpeg = std::process::Command::new("ffmpeg");
        ffmpeg.args(["-hide_banner", "-loglevel", "error", "-y"]);
        ffmpeg.args(["-err_detect", "explode"]);
        if codec == Codec::AV1 {
            ffmpeg.args(["-c:v", "libdav1d"]);
        }
        let status = ffmpeg
            .arg("-i")
            .arg(&path)
            .args(["-pix_fmt", "yuv420p", "-f", "rawvideo"])
            .arg(&decoded)
            .status()?;
        assert!(status.success(), "{codec:?}: ffmpeg found an error");
        let frame_size = (WIDTH * HEIGHT * 3 / 2) as u64;
        assert_eq!(
            std::fs::metadata(&decoded)?.len() / frame_size,
            submitted,
            "{codec:?}: frames ffmpeg decoded"
        );
    }
    Ok(())
}
