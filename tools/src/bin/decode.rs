//! Decode a stream to raw YUV, to compare with a software decoder.
//!
//! Usage:
//!   cargo run -p pixelforge-tools --bin decode -- input.264 [output.yuv]
//!
//! The codec comes from the extension: `.264`/`.h264`, `.265`/`.h265`/`.hevc`,
//! `.obu`/`.av1`, or `.ivf` (AV1 in IVF). The output is NV12, or P010 for a
//! 10-bit stream, so it compares byte for byte with
//! `ffmpeg -i input.264 -pix_fmt nv12 ref.yuv` (`-pix_fmt p010le` for 10
//! bits, and `-c:v libdav1d -noautoscale` for AV1).
//!
//! `PIXELFORGE_VALIDATION=1` turns on the validation layers.
//! `PIXELFORGE_NO_UNIFIED_LAYOUTS=1` forces the path that copies each frame
//! out of the decoder, on a device that would hand over its own images.

use pixelforge::Codec;
use pixelforge::decoder::{DecodeConfig, Decoder, FramePoll};
use pixelforge::{VideoContextBuilder, YuvDownloader};
use std::io::Write;

/// Bytes handed to the decoder per call, standing in for a network read.
const CHUNK_SIZE: usize = 64 * 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let mut args = std::env::args().skip(1);
    let input = args
        .next()
        .ok_or("usage: decode <input.264|.265|.obu|.ivf> [output.yuv]")?;
    let mut output = args.next().map(std::fs::File::create).transpose()?;
    let Input {
        codec,
        pieces,
        config,
    } = open(&input)?;

    let mut builder = VideoContextBuilder::new()
        .require_decode(codec)
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok());
    if std::env::var("PIXELFORGE_NO_UNIFIED_LAYOUTS").is_ok() {
        builder = builder.without_unified_image_layouts();
    }
    let context = builder.build()?;
    let mut downloader = YuvDownloader::new(&context)?;
    // A frame per call can be outstanding at once, and a chunk holds several.
    let mut decoder = Decoder::new(context, config.with_output_depth(8))?;

    let start = std::time::Instant::now();
    let mut count = 0usize;
    let mut generation = None;
    let mut on_frame =
        |frame: pixelforge::decoder::DecodedFrame| -> Result<(), Box<dyn std::error::Error>> {
            if generation.is_some_and(|g| g != frame.generation) {
                println!(
                    "  generation {} -> {} at frame {count}: the decoder rebuilt its images",
                    generation.unwrap(),
                    frame.generation
                );
            }
            generation = Some(frame.generation);
            if let Some(file) = output.as_mut() {
                let yuv = downloader.download(&frame)?;
                file.write_all(&yuv.y)?;
                file.write_all(&yuv.uv)?;
            }
            count += 1;
            Ok(())
        };
    for (i, chunk) in pieces.iter().enumerate() {
        decoder.decode(chunk, i as u64)?;
        while let FramePoll::Frame(frame) = decoder.try_next_frame()? {
            on_frame(frame)?;
        }
    }
    decoder.finish()?;
    while let Some(frame) = pollster::block_on(decoder.next_frame())? {
        on_frame(frame)?;
    }
    let elapsed = start.elapsed();
    println!(
        "Decoded {count} frames in {elapsed:.1?} ({:.1} fps)",
        count as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}

/// A stream read from a file.
struct Input {
    codec: Codec,
    /// The pieces to hand the decoder, one per call.
    pieces: Vec<Vec<u8>>,
    /// The configuration the pieces need.
    config: DecodeConfig,
}

/// Read the stream at `path`.
fn open(path: &str) -> Result<Input, Box<dyn std::error::Error>> {
    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let data = std::fs::read(path)?;
    let codec = match extension.as_str() {
        "264" | "h264" => Codec::H264,
        "265" | "h265" | "hevc" => Codec::H265,
        "obu" | "av1" | "ivf" => Codec::AV1,
        other => return Err(format!("unknown stream extension {other:?}").into()),
    };
    if extension == "ivf" {
        // Whole frames from a container.
        return Ok(Input {
            codec,
            pieces: ivf_frames(&data)?,
            config: DecodeConfig::new(codec),
        });
    }
    // A raw stream, which the decoder frames itself.
    Ok(Input {
        codec,
        pieces: data.chunks(CHUNK_SIZE).map(<[u8]>::to_vec).collect(),
        config: DecodeConfig::new(codec).with_byte_stream(),
    })
}

/// The frames of an IVF file: a 32-byte file header, then each frame behind a
/// 12-byte header holding its size and timestamp.
fn ivf_frames(data: &[u8]) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    if data.len() < 32 || &data[..4] != b"DKIF" {
        return Err("not an IVF file".into());
    }
    let header_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut frames = Vec::new();
    let mut pos = header_len;
    while pos + 12 <= data.len() {
        let size = u32::from_le_bytes(data[pos..pos + 4].try_into()?) as usize;
        let start = pos + 12;
        let frame = data.get(start..start + size).ok_or("truncated IVF frame")?;
        frames.push(frame.to_vec());
        pos = start + size;
    }
    Ok(frames)
}
