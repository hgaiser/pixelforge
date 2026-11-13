//! Decode a stream to a raw YUV file.
//!
//! Decoded frames stay on the GPU. Here [`YuvDownloader`] copies each one to
//! CPU memory to write it to the file. To show frames, sample them where they
//! are instead, as the `sample_frame` example does.
//!
//! Usage:
//!   cargo run --example decode -- input.264 output.yuv
//!
//! The codec comes from the extension: `.264`, `.265` or `.obu` (AV1). The
//! output is NV12, or P010 for a 10-bit stream.

use pixelforge::decoder::{DecodeConfig, DecodedFrame, Decoder, FramePoll};
use pixelforge::{Codec, VideoContextBuilder, YuvDownloader};
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, input, output] = args.as_slice() else {
        return Err("usage: decode <input.264|.265|.obu> <output.yuv>".into());
    };
    let codec = match input.rsplit('.').next() {
        Some("264") => Codec::H264,
        Some("265") => Codec::H265,
        Some("obu") => Codec::AV1,
        _ => return Err(format!("unknown stream extension in {input:?}").into()),
    };
    let stream = std::fs::read(input)?;

    let context = VideoContextBuilder::new().require_decode(codec).build()?;
    // A file can be cut anywhere, so the decoder finds the frames in the byte
    // stream itself.
    let config = DecodeConfig::new(codec).with_byte_stream();
    let mut decoder = Decoder::new(context.clone(), config)?;
    let mut downloader = YuvDownloader::new(&context)?;

    let mut file = File::create(output)?;
    let mut write = |frame: DecodedFrame| -> Result<(), Box<dyn std::error::Error>> {
        let yuv = downloader.download(&frame)?;
        file.write_all(&yuv.y)?;
        file.write_all(&yuv.uv)?;
        // Dropping the frame gives its image back to the decoder.
        Ok(())
    };

    // Feed the stream in pieces, as if from the network, and take each frame
    // the GPU has finished.
    for (index, piece) in stream.chunks(64 * 1024).enumerate() {
        decoder.decode(piece, index as u64)?;
        while let FramePoll::Frame(frame) = decoder.try_next_frame()? {
            write(frame)?;
        }
    }
    // End of stream: decode the last picture and take the frames still held
    // back for display order.
    decoder.finish()?;
    while let Some(frame) = pollster::block_on(decoder.next_frame())? {
        write(frame)?;
    }
    println!("Wrote {output}");
    Ok(())
}
