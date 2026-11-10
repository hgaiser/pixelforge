//! Encode with pixelforge, then decode the result with pixelforge.
//!
//! The decoder must handle anything this project's encoder can produce. That is
//! not covered by decoding third-party streams: the encoder makes its own
//! choices (notably explicit reference marking), so it needs its own round trip.
//! It is also where B-frames are exercised, for each count of them the
//! device can encode.
//!
//! The input is planar YUV420 (I420). The decoded output is NV12 in display
//! order. The assertions are that no frame is lost across the round trip, and
//! that the output is byte-identical to ffmpeg's software decode of the same
//! stream. With B-frames, also that the timestamps are in order, that ffmpeg
//! sees the frame types the encoder reports and no error in the stream, and
//! that the quality is close to that of the stream without B-frames. The files are
//! left in the scratch dir for inspection.
//!
//! Ignored by default: requires a Vulkan Video device and ffmpeg. Run with
//! `cargo test -- --ignored`.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Write};

#[allow(dead_code)]
mod common;
use common::{Readback, decode_stream, write_nv12};

use pixelforge::decoder::{DecodeConfig, Decoder};
use pixelforge::{
    BFrameReferences, BFrameSupport, BitDepth, ChromaFormat, Codec, EncodeConfig, Encoder,
    FrameType, RateControl, VideoContextBuilder, YuvLayout, YuvUploader,
};
use pixelforge::{Frame, MAX_B_FRAMES};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const GOP: u32 = 30;

#[test]
#[ignore = "requires a Vulkan Video device and ffmpeg"]
fn roundtrip_h264() -> Result<(), Box<dyn std::error::Error>> {
    roundtrip(Codec::H264, "h264", 0, BFrameReferences::Flat, 0).map(|_| ())
}

#[test]
#[ignore = "requires a Vulkan Video device and ffmpeg"]
fn roundtrip_h265() -> Result<(), Box<dyn std::error::Error>> {
    roundtrip(Codec::H265, "h265", 0, BFrameReferences::Flat, 0).map(|_| ())
}

#[test]
#[ignore = "requires a Vulkan Video device and ffmpeg"]
fn roundtrip_av1() -> Result<(), Box<dyn std::error::Error>> {
    roundtrip(Codec::AV1, "obu", 0, BFrameReferences::Flat, 0).map(|_| ())
}

#[test]
#[ignore = "requires a Vulkan Video device and ffmpeg"]
fn roundtrip_h264_top_quality_level() -> Result<(), Box<dyn std::error::Error>> {
    roundtrip_top_quality_level(Codec::H264, "h264")
}

#[test]
#[ignore = "requires a Vulkan Video device and ffmpeg"]
fn roundtrip_h265_top_quality_level() -> Result<(), Box<dyn std::error::Error>> {
    roundtrip_top_quality_level(Codec::H265, "h265")
}

#[test]
#[ignore = "requires a Vulkan Video device and ffmpeg"]
fn roundtrip_av1_top_quality_level() -> Result<(), Box<dyn std::error::Error>> {
    roundtrip_top_quality_level(Codec::AV1, "obu")
}

/// A round trip at the highest quality level the device has, with and
/// without B-frames. A higher level only changes what the driver searches,
/// so the stream must decode as at level 0.
fn roundtrip_top_quality_level(
    codec: Codec,
    extension: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();
    let context = VideoContextBuilder::new().require_encode(codec).build()?;
    let levels = context.encode_quality_levels(codec, ChromaFormat::Yuv420, BitDepth::Eight);
    let b_frames = context.supports_b_frames(codec, ChromaFormat::Yuv420, BitDepth::Eight);
    drop(context);
    assert!(
        levels > 0,
        "{codec:?}: the device reports no quality levels"
    );
    let top = levels - 1;
    eprintln!("{codec:?}: quality levels 0 to {top}");
    roundtrip(codec, extension, 0, BFrameReferences::Flat, top)?;
    if b_frames != BFrameSupport::None {
        roundtrip(codec, extension, 3, BFrameReferences::Flat, top)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan Video device and ffmpeg"]
fn roundtrip_h264_b_frames() -> Result<(), Box<dyn std::error::Error>> {
    roundtrip_b_frames(Codec::H264, "h264")
}

#[test]
#[ignore = "requires a Vulkan Video device and ffmpeg"]
fn roundtrip_h265_b_frames() -> Result<(), Box<dyn std::error::Error>> {
    roundtrip_b_frames(Codec::H265, "h265")
}

#[test]
#[ignore = "requires a Vulkan Video device and ffmpeg"]
fn roundtrip_av1_b_frames() -> Result<(), Box<dyn std::error::Error>> {
    roundtrip_b_frames(Codec::AV1, "obu")
}

/// Round trips with B-frames, against the one without. Nothing to do on a
/// device that encodes none.
fn roundtrip_b_frames(codec: Codec, extension: &str) -> Result<(), Box<dyn std::error::Error>> {
    let support = VideoContextBuilder::new()
        .require_encode(codec)
        .build()?
        .supports_b_frames(codec, ChromaFormat::Yuv420, BitDepth::Eight);
    if support == BFrameSupport::None {
        eprintln!("{codec:?}: no B-frames on this device, nothing to check");
        return Ok(());
    }
    let baseline = roundtrip(codec, extension, 0, BFrameReferences::Flat, 0)?;
    // 30 frames: none of these counts divides the 29 after the IDR, so the
    // last group is a short one, closed by the flush. A pyramid starts at
    // 3 B-frames, has a second level of references at 6, and at 7 a B-frame
    // replaces another as a reference.
    let mut cases = vec![
        (1, BFrameReferences::Flat),
        (2, BFrameReferences::Flat),
        (3, BFrameReferences::Flat),
        (MAX_B_FRAMES, BFrameReferences::Flat),
    ];
    if support == BFrameSupport::Pyramid {
        cases.extend([3, 4, 6, MAX_B_FRAMES].map(|b| (b, BFrameReferences::Pyramid)));
    } else {
        eprintln!("{codec:?}: no B-frame pyramid on this device");
    }
    for (b_frames, references) in cases {
        let psnr = roundtrip(codec, extension, b_frames, references, 0)?;
        // B-frames get a coarser quantizer, one step more for each level
        // of a pyramid, which costs up to about 1.6 dB at a constant QP. A
        // B-frame predicted from the wrong picture costs 5 dB or more.
        assert!(
            psnr > baseline - 2.0,
            "{b_frames} B-frames ({references:?}): luma PSNR {psnr:.2} dB, \
             {baseline:.2} dB without"
        );
    }
    Ok(())
}

/// The luma PSNR of `decoded` (NV12) against `source` (I420), in dB.
fn luma_psnr(source: &[u8], decoded: &[u8]) -> f64 {
    let luma = (WIDTH * HEIGHT) as usize;
    let frame = luma * 3 / 2;
    let mut error = 0u64;
    let mut count = 0u64;
    for (a, b) in source.chunks_exact(frame).zip(decoded.chunks_exact(frame)) {
        for (x, y) in a[..luma].iter().zip(&b[..luma]) {
            let d = i64::from(*x) - i64::from(*y);
            error += (d * d) as u64;
        }
        count += luma as u64;
    }
    10.0 * (255.0f64 * 255.0 * count as f64 / error.max(1) as f64).log10()
}

/// The frame types ffmpeg sees in the stream, in display order, one letter
/// each.
fn ffprobe_frame_types(path: &std::path::Path) -> Result<String, Box<dyn std::error::Error>> {
    let output = std::process::Command::new("ffprobe")
        .args(["-v", "error", "-show_frames", "-show_entries"])
        .args(["frame=pict_type", "-of", "csv=p=0"])
        .arg(path)
        .output()?;
    assert!(output.status.success(), "ffprobe failed");
    Ok(String::from_utf8(output.stdout)?
        .split_whitespace()
        .collect())
}

/// Encode the test frames with `b_frames` B-frames, decode them, and return
/// the luma PSNR against the source.
fn roundtrip(
    codec: Codec,
    extension: &str,
    b_frames: u32,
    references: BFrameReferences,
    quality_level: u32,
) -> Result<f64, Box<dyn std::error::Error>> {
    common::init_logging();

    let input_path = format!("testdata/test_frames_{WIDTH}x{HEIGHT}_yuv420p.yuv");
    common::ensure_test_data(WIDTH, HEIGHT, "yuv420p", &input_path)?;

    let dir = common::scratch_dir(&format!(
        "roundtrip-{extension}-b{b_frames}-{references:?}-q{quality_level}"
    ));
    let bitstream_path = dir.join(format!("out.{extension}"));

    let context = VideoContextBuilder::new()
        .app_name("pixelforge-roundtrip")
        .require_encode(codec)
        .require_decode(codec)
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .build()?;

    // --- Encode ---
    let mut yuv = Vec::new();
    File::open(&input_path)?.read_to_end(&mut yuv)?;
    let frame_size = (WIDTH * HEIGHT * 3 / 2) as usize;
    let frame_count = yuv.len() / frame_size;

    let config = EncodeConfig::new(codec, WIDTH, HEIGHT)
        .with_rate_control(RateControl::Cqp { qp: 26 })
        .with_frame_rate(30, 1)
        .with_gop_size(GOP)
        .with_b_frames(b_frames)
        .with_b_frame_references(references)
        .with_quality_level(quality_level);

    let mut encoder = Encoder::new(context.clone(), config)?;
    let mut input_image = YuvUploader::for_encoder(&encoder)?;

    let mut bitstream = Vec::new();
    // (pts, dts, type) of each packet, in the order they come.
    let mut packets = Vec::new();
    let mut pending: VecDeque<pixelforge::EncodeFuture> = VecDeque::new();
    for i in 0..frame_count {
        input_image.upload(YuvLayout::I420, &yuv[i * frame_size..(i + 1) * frame_size])?;
        pending.push_back(encoder.submit(Frame::image(input_image.source()))?);
        // A B-frame waits for the reference frame after it, so a packet may
        // not come before `b_frames` more frames are submitted.
        while pending.len() > 2 + b_frames as usize {
            let packet = pollster::block_on(pending.pop_front().expect("non-empty"))?;
            packets.push((packet.pts, packet.dts, packet.frame_type));
            bitstream.extend_from_slice(&packet.data);
        }
    }
    encoder.flush()?;
    while let Some(future) = pending.pop_front() {
        let packet = pollster::block_on(future)?;
        packets.push((packet.pts, packet.dts, packet.frame_type));
        bitstream.extend_from_slice(&packet.data);
    }
    check_timestamps(&packets, frame_count, b_frames);
    File::create(&bitstream_path)?.write_all(&bitstream)?;
    println!(
        "Encoded {} frames -> {} bytes (gop={GOP})",
        frame_count,
        bitstream.len()
    );

    // --- Decode it back ---
    // Display order by default, so frames come out ready to write; `flush`
    // drains whatever the reorder buffer still holds at end of stream.
    let yuv_path = dir.join("out.yuv");
    let mut out = Some(File::create(&yuv_path)?);
    let mut readback = Some(Readback::new(&context)?);
    let mut decoder = Decoder::new(context, DecodeConfig::new(codec).with_byte_stream())?;
    let mut decoded_count = 0usize;

    decode_stream(&mut decoder, &bitstream, |frame| {
        write_nv12(&frame, &mut readback, &mut out)?;
        decoded_count += 1;
        Ok(())
    })?;
    println!("Decoded {} frames", decoded_count);

    assert_eq!(
        decoded_count, frame_count,
        "round trip lost frames: encoded {frame_count} but decoded {decoded_count}"
    );
    println!("Wrote decoded frames to {}", yuv_path.display());
    drop(out);

    // --- Compare against ffmpeg's software decoder ---
    let reference_path = dir.join("reference.yuv");
    let mut ffmpeg = std::process::Command::new("ffmpeg");
    ffmpeg.args(["-hide_banner", "-loglevel", "error", "-y"]);
    ffmpeg.args(["-err_detect", "explode"]);
    if codec == Codec::AV1 {
        ffmpeg.args(["-c:v", "libdav1d"]);
    }
    let status = ffmpeg
        .arg("-i")
        .arg(&bitstream_path)
        .args(["-pix_fmt", "nv12", "-f", "rawvideo"])
        .arg(&reference_path)
        .status()?;
    assert!(status.success(), "ffmpeg failed to decode the stream");
    assert!(
        std::fs::read(&reference_path)? == std::fs::read(&yuv_path)?,
        "decoded frames differ from ffmpeg's decode of {}",
        bitstream_path.display()
    );

    // ffmpeg must see the frame types the encoder reports. Only for H.264
    // and H.265: an AV1 frame has no such type in the stream.
    if b_frames > 0 && codec != Codec::AV1 {
        let mut by_pts = packets.clone();
        by_pts.sort_by_key(|&(pts, _, _)| pts);
        let expected: String = by_pts
            .iter()
            .map(|&(_, _, frame_type)| match frame_type {
                FrameType::B => 'B',
                FrameType::P => 'P',
                _ => 'I',
            })
            .collect();
        assert_eq!(ffprobe_frame_types(&bitstream_path)?, expected);
    }

    Ok(luma_psnr(&yuv, &std::fs::read(&yuv_path)?))
}

/// Every frame has one packet, the packets come in decode order, and no
/// packet is to be decoded after it is shown. With B-frames, some of them
/// are B-frames.
fn check_timestamps(packets: &[(u64, i64, FrameType)], frame_count: usize, b_frames: u32) {
    let mut pts: Vec<u64> = packets.iter().map(|&(pts, _, _)| pts).collect();
    pts.sort_unstable();
    assert!(
        pts.iter().copied().eq(0..frame_count as u64),
        "one packet for each frame: {pts:?}"
    );
    for pair in packets.windows(2) {
        assert!(pair[0].1 < pair[1].1, "dts must increase: {pair:?}");
    }
    for &(pts, dts, _) in packets {
        assert!(dts <= pts as i64, "dts {dts} is after pts {pts}");
    }
    let b_count = packets
        .iter()
        .filter(|&&(_, _, frame_type)| frame_type == FrameType::B)
        .count();
    if b_frames == 0 {
        assert_eq!(b_count, 0);
    } else {
        // Each group is `b_frames` long at most, with one reference frame.
        assert!(b_count >= (frame_count - 1) * b_frames as usize / (b_frames as usize + 1) - 1);
    }
}
