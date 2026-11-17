//! Decoders made from clones of one `VideoContext`, each driven from its own
//! thread, share its queues. Every submission goes through the context's queue
//! locks, so the threads must not disturb each other: each one decodes the
//! same frames as a single thread alone does.
//!
//! Ignored by default: requires a Vulkan Video device. Run with
//! `cargo test -- --ignored`.

#[allow(dead_code)]
mod common;

use pixelforge::decoder::{DecodeConfig, Decoder};
use pixelforge::{Codec, VideoContext, VideoContextBuilder, YuvDownloader};

const THREADS: usize = 4;
const ROUNDS: usize = 3;

/// Decode `stream` and return every frame's samples, luma then chroma.
fn decode_all(
    context: &VideoContext,
    stream: &[u8],
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let mut downloader = YuvDownloader::new(context)?;
    let mut decoder = Decoder::new(
        context.clone(),
        DecodeConfig::new(Codec::H264).with_byte_stream(),
    )?;
    let mut frames = Vec::new();
    common::decode_stream(&mut decoder, stream, |frame| {
        let yuv = downloader.download(&frame)?;
        let mut bytes = yuv.y;
        bytes.extend_from_slice(&yuv.uv);
        frames.push(bytes);
        Ok(())
    })?;
    Ok(frames)
}

#[test]
#[ignore = "requires a Vulkan Video device"]
fn decoders_on_threads_share_a_context() -> Result<(), Box<dyn std::error::Error>> {
    common::init_logging();
    let stream = std::fs::read("tests/data/bframes.264")?;
    let context = VideoContextBuilder::new()
        .app_name("pixelforge-shared-context")
        .require_decode(Codec::H264)
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .build()?;
    let expected = decode_all(&context, &stream)?;
    assert!(!expected.is_empty());

    for round in 0..ROUNDS {
        let results: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..THREADS)
                .map(|_| scope.spawn(|| decode_all(&context, &stream).map_err(|e| e.to_string())))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (thread, result) in results.into_iter().enumerate() {
            let frames = result.unwrap_or_else(|e| panic!("round {round} thread {thread}: {e}"));
            assert!(
                frames == expected,
                "round {round} thread {thread}: frames differ from a single thread's"
            );
        }
    }
    Ok(())
}
