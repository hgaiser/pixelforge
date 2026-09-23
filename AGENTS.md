# Agent Instructions

## Project Overview
Pixelforge is a Rust library for video encoding and decoding using Vulkan Video.

## Build & Test
```bash
cargo build
cargo test
cargo test -- --ignored        # device + ffmpeg tests, ignored by default
cargo bench --bench encode
cargo run --example encode -- h264
cargo run --example decode -- input.264 output.yuv
```

Decoding on Intel Arc under Mesa needs the video queues enabled explicitly:
`ANV_DEBUG=video-decode,video-encode`.

## README Generation
The `README.md` is generated from the doc comments in `src/lib.rs` using the `README.tpl` template.
To regenerate:
```bash
cargo readme > README.md
```

This is the exact command CI compares against (`diff --brief <(cargo readme)
README.md`). Do not add `--no-indent-headings`: it flattens every section to `#`
and the diff then fails.
Do not edit `README.md` directly; update the doc comments in `src/lib.rs` instead.

To verify the quality of the encoded videos, run:

```bash
cargo run --example encode -- h265 \
    && rm -f decoded.yuv \
    && ffmpeg -hide_banner -loglevel error -y -i output.h265 -pix_fmt yuv420p -f rawvideo decoded.yuv \
    && ffmpeg -hide_banner -loglevel info -s 320x240 -pix_fmt yuv420p -f rawvideo -i testdata/test_frames.yuv -s 320x240 -pix_fmt yuv420p -f rawvideo -i decoded.yuv -lavfi psnr -f null -
```

To verify decoding, decode a stream and compare against ffmpeg's software
decoder, which should be byte-identical:

```bash
cargo run --example decode -- tests/data/bframes.264 out.yuv \
    && ffmpeg -hide_banner -loglevel error -y -i tests/data/bframes.264 -pix_fmt nv12 ref.yuv \
    && cmp ref.yuv out.yuv && echo "decode matches ffmpeg"
```

Decoding has two paths and both need checking. Where the device supports
`VK_KHR_unified_image_layouts`, frames are the decoder's own images, handed over
with no copy. Where it does not, they are copied into private images. Force the
copying path to exercise it on hardware that would otherwise never take it:

```bash
PIXELFORGE_NO_UNIFIED_LAYOUTS=1 cargo run --example decode -- \
    tests/data/bframes.264 out.yuv && cmp ref.yuv out.yuv
```

Which path a run took is in the debug log (`RUST_LOG=debug`), on the
`H.264 decode session:` line as `pinnable=`.

The render path has its own example. `sample_frame` reads decoded frames in a
compute shader through a sampler-YCbCr conversion, which is the path that never
copies anything, and writes RGBA:

```bash
cargo run --example sample_frame -- tests/data/bframes.264 out.rgba
```

Do not judge that one by PSNR. The hardware sampler and ffmpeg reconstruct
subsampled chroma differently, so they agree almost everywhere and disagree
hard on pixels sitting on a colour edge; on the synthetic test patterns that
drags PSNR to ~26 dB while the images are indistinguishable. Compare the share
of pixels that agree instead. For `bframes.264` that has been 87.9% within 2
and 33.2% exact on every device tested so far, and it is deterministic enough
to treat a change in it as a regression.

`sample_planes` is the other half of that story, and the tighter test. It reads
the same frames through per-plane views, with no ycbcr conversion and no sampler
at all, and writes the samples straight through, so its output is plain NV12 and
can be compared exactly:

```bash
cargo run --example sample_planes -- tests/data/bframes.264 out.yuv \
    && cmp ref.yuv out.yuv && echo "plane views match the decoder"
```

Where that example is worth extra attention: on a device *without*
`VK_KHR_unified_image_layouts`, its frames are pool copies, which carry no video
usage, so a mistake in how a plane view is created passes silently. The same
mistake is an immediate validation error on a device that has the extension,
because there the frame really is the decoder's DPB image. So if a device
supporting it is available, run this example there before trusting a green run
elsewhere. `RUST_LOG=debug` reports which case applies, as `pinnable=`.

Nothing in `tests/data` changes resolution mid-stream, and that path has its
own hazards, so build one and check it:

```bash
ffmpeg -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 -c:v libx264 -bf 2 -f h264 a.264
ffmpeg -f lavfi -i testsrc2=size=640x480:rate=30:duration=1 -c:v libx264 -bf 2 -f h264 b.264
cat a.264 b.264 > switch.264
cargo run --example decode -- switch.264 out.yuv
```

Sixty frames, thirty at each size, and one `generation N -> N+1` line at the
changeover. Run `sample_frame` and `sample_planes` over it too: a consumer's own
images are sized from a frame, so a resolution change is where they have to be
rebuilt, and getting that wrong is silent rather than loud. Build the reference per segment, since ffmpeg rescales a
resolution-changing stream to its first size and a whole-stream reference will
not compare.

Pixelforge can also run on a device the caller created, for decode and for
encode. `tests/adopted.rs` builds one from a Vulkan 1.1 instance, as an
application that never asked for 1.3 would, hands pixelforge spare queues, and
checks the output against a context pixelforge created itself. It needs no
ffmpeg. Anything that reaches a core 1.2 or 1.3 entry point panics there, so
run it after touching submission or barrier code.

The RGB input path (`EncodeConfig::with_rgb_input`) needs
`VK_VALVE_video_encode_rgb_conversion`, which today only RADV has.
`tests/rgb_encode.rs` compares it against the colour converter and skips on
other devices, so a green run elsewhere says nothing about it. Under validation it draws two messages that are the layer's,
not pixelforge's: the layer does not model that extension, so it rejects the
RGB profile struct in a query pool's profile chain (the spec requires that
chain to match the session's) and checks the RGB input view against the plain
format features rather than the video profile's.

Make sure there are no Vulkan validation layer errors during execution. Enable
them with `PIXELFORGE_VALIDATION=1`; the layer's messages are routed through
`tracing`, so pair it with `RUST_LOG=warn` (or `debug` for the layer's own
chatter). An integration test needs a subscriber installed for any of it to
show: `common::init_logging()` installs one, and a new test that builds a
context has to call it or install its own, or its validation messages go
nowhere. They are captured like any test output,
so run with `--nocapture` to see a passing test's. Without
`VK_LAYER_KHRONOS_validation` installed, pixelforge logs a warning and carries
on with validation disabled, so absence of errors means nothing if the layer is
missing.

Run the ignored tests once without validation as well. The layer gives every
object a unique handle, while drivers reuse the handles of destroyed objects
freely, so anything that remembers state by handle can pass with validation
on and fail without it. That has happened: the converter once cached an image
view by image handle, and only RADV without validation showed it.

## Code Style
- Follow `rustfmt.toml` formatting rules
- Run `cargo fmt` before committing
- Use clippy for linting
- If a comment is a sentence, it should end with a period
- Avoid long files; split into modules if necessary

## Project Structure
- `src/` - Library source code
- `examples/` - Usage examples for encoding and decoding
- `tests/` - Integration tests; the device + ffmpeg checks are `#[ignore]`d and
  run with `cargo test -- --ignored`
- `benches/` - The encode timing/throughput benchmark (`cargo bench`)
- `testdata/` - Test input files (note: `test_frames.yuv` is a git-LFS pointer;
  generate frames locally if LFS content is unavailable)
- `tests/data/` - Small H.264 streams used by the decoder tests

## Key Dependencies
- Vulkan Video API
- ash (Vulkan bindings for Rust)
