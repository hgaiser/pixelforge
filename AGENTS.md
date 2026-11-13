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
cargo run -p pixelforge-tools --bin encode -- h264 --b-frames 3
cargo run -p pixelforge-tools --bin decode -- input.264 output.yuv
```

The examples (`examples/`) are for users: short, one use each. The tools
(`tools/`, a workspace crate that is not published) are for checking the
library, with every option that takes. The commands below use the tools.

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
cargo run -p pixelforge-tools --bin encode -- h265 \
    && rm -f decoded.yuv \
    && ffmpeg -hide_banner -loglevel error -y -i output.h265 -pix_fmt yuv420p -f rawvideo decoded.yuv \
    && ffmpeg -hide_banner -loglevel info -s 320x240 -pix_fmt yuv420p -f rawvideo -i testdata/test_frames.yuv -s 320x240 -pix_fmt yuv420p -f rawvideo -i decoded.yuv -lavfi psnr -f null -
```

To verify decoding, decode a stream and compare against ffmpeg's software
decoder, which should be byte-identical:

```bash
cargo run -p pixelforge-tools --bin decode -- tests/data/bframes.264 out.yuv \
    && ffmpeg -hide_banner -loglevel error -y -i tests/data/bframes.264 -pix_fmt nv12 ref.yuv \
    && cmp ref.yuv out.yuv && echo "decode matches ffmpeg"
```

The `decode` tool picks the codec from the extension (`.264`, `.265`,
`.obu`, `.ivf`). For AV1 the reference is dav1d, and `-noautoscale` keeps a
stream that changes size from being scaled to its first size (it applies to
any codec; H.264 and H.265 streams in `tests/data` do not change size):

```bash
cargo run -p pixelforge-tools --bin decode -- tests/data/av1/aom_altref.obu out.yuv \
    && ffmpeg -hide_banner -loglevel error -y -c:v libdav1d -i tests/data/av1/aom_altref.obu \
        -noautoscale -pix_fmt nv12 ref.yuv \
    && cmp ref.yuv out.yuv && echo "decode matches dav1d"
```

`tests/decode_streams.rs` does this for every fixture in `tests/data` (use
`p010le` for the 10-bit ones), and `tests/roundtrip.rs` for each codec's own
encoder output. Those tests take whichever path the device has; the
`PIXELFORGE_NO_UNIFIED_LAYOUTS` switch below is read by the `decode` tool
only.

On NVIDIA (615.78.08) the AV1 decoder does not copy the `StdVideoAV1SequenceHeader`
handed to `vkCreateVideoSessionParametersKHR`: it reads it again at every
decode. The decoder keeps it alive for as long as the session parameters. Freed,
nothing fails and only the chroma of inter frames comes out slightly wrong, so
if AV1 inter frames ever differ from dav1d in chroma alone, look for a Std
structure that does not live long enough.

Decoding has two paths and both need checking. Where the device supports
`VK_KHR_unified_image_layouts`, frames are the decoder's own images, handed over
with no copy. Where it does not, they are copied into private images. Force the
copying path to exercise it on hardware that would otherwise never take it:

```bash
PIXELFORGE_NO_UNIFIED_LAYOUTS=1 cargo run -p pixelforge-tools --bin decode -- \
    tests/data/bframes.264 out.yuv && cmp ref.yuv out.yuv
```

Which path a run took is in the debug log (`RUST_LOG=debug`), on the
`<codec> decode session:` line (for example `H.264 decode session:`) as
`pinnable=`.

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
of pixels that agree instead (all three channels). For `bframes.264` that has
been 87.9% within 2 on every device tested so far, and it is deterministic
enough to treat a change in it as a regression.

`tests/sample_planes.rs` is the other half of that story, and the tighter
check. It reads the same frames through per-plane views, with no ycbcr
conversion and no sampler at all, and compares the samples byte for byte with
`YuvDownloader`'s copy of each frame, on both decode paths:

```bash
cargo test --test sample_planes -- --ignored --nocapture
```

On a device *without* `VK_KHR_unified_image_layouts`, its frames are pool
copies, which carry no video usage, so a mistake in how a plane view is created
passes silently. The same mistake is an immediate validation error on a device
that has the extension, because there the frame really is the decoder's DPB
image. So if a device supporting it is available, run it there under
validation before trusting a green run elsewhere.

Nothing in `tests/data` changes resolution mid-stream, and that path has its
own hazards, so build one and check it:

```bash
ffmpeg -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 -c:v libx264 -bf 2 -f h264 a.264
ffmpeg -f lavfi -i testsrc2=size=640x480:rate=30:duration=1 -c:v libx264 -bf 2 -f h264 b.264
cat a.264 b.264 > switch.264
cargo run -p pixelforge-tools --bin decode -- switch.264 out.yuv
```

Sixty frames, thirty at each size, and one `generation N -> N+1` line at the
changeover. Run `sample_frame` over it too: a consumer's own
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

An RGB source (`Source::Rgb`) is converted by the encoder itself
(`Conversion::Encoder`) only with `VK_VALVE_video_encode_rgb_conversion`,
which today only RADV has. The colour converter is crate-private, so its
device tests are in the crate (`src/converter/device_tests.rs`, run with
`cargo test --lib -- --ignored`): they check the shader against the same maths
on the CPU, and the shader path inside the encoder against converting by hand
(byte-identical). `tests/rgb_encode.rs` on RADV compares the encoder's
conversion against the shader's and checks that `supports_rgb_encode_input`
agrees with encoder creation, so a green run elsewhere says little about the
encoder path. Under validation that path draws two messages that are the
layer's, not pixelforge's: the layer does not model that extension, so it
rejects the RGB profile struct in a query pool's profile chain (the spec
requires that chain to match the session's) and checks the RGB input view
against the plain format features rather than the video profile's.

DMA-BUF input (`--features dmabuf`, `tests/dmabuf.rs`) has three ways to read a
buffer: the encode itself, a copy into the input image, and the colour
converter's shader. The tests play the producer by importing the same buffer
into an image of their own. On NVIDIA (615.78.08) the driver reports more than
works, and `quirks_allow` in `src/encoder/external.rs` holds it back: the encode
never reads a DMA-BUF directly (it reads black), and two-plane YUV buffers are
LINEAR only (block-linear ones lose their chroma plane). So on NVIDIA the
direct read and the unaligned-size test are skipped, and only RADV can check
them. `PIXELFORGE_NO_DMABUF_QUIRKS=1` turns the quirks off, which makes two of
those tests fail on that driver.

Reference frame invalidation (`tests/rfi.rs`) recovers from the frames the
encoder keeps in the DPB (the window, `max_reference_frames`), which can be
more than each frame predicts from. A device that allows many references per
frame never exercises that. `PIXELFORGE_MAX_ACTIVE_REFERENCES=1` limits the
references per frame without changing the window, to test it:

```bash
PIXELFORGE_MAX_ACTIVE_REFERENCES=1 cargo test --test rfi -- --ignored --nocapture
```

`RUST_LOG=pixelforge::encoder::codec=debug` reports the negotiated window and
per-frame count, on the `Reference window` line.

`tests/rfi.rs` runs each codec without B-frames and with the B-frames the
device has. It checks each frame's PSNR against the frames before the loss,
not only the average: a decoder that predicts from other pictures than the
encoder did loses a few dB from there on, which the average hides. That is
how the H.264 P-frame after an invalidation was found to predict from the
lost frames in a decoder (it had MMCO operations and no list modification).

B-frames (`EncodeConfig::with_b_frames`, `tests/b_frames.rs`, the `_b_frames`
tests in `tests/roundtrip.rs`) are encoded later than they are submitted. The
planner (`src/encoder/gop.rs`) is pure code and has the unit tests for the
order. What needs a device is the stream each codec writes, and the checks are
ffmpeg's (dav1d's for AV1): decode with `-err_detect explode`, compare with
pixelforge's decoder byte for byte, and compare the frame types `ffprobe`
sees. Run the encode tool with them when touching reference lists:

```bash
cargo run -p pixelforge-tools --bin encode -- h264 --b-frames 7 --pyramid \
    && ffmpeg -hide_banner -loglevel error -err_detect explode -i output.h264 -f null -
```

A wrong reference list does not always make ffmpeg report an error. It can
decode cleanly to the wrong pictures, so also check the PSNR against the source
with the command above: a B-frame that predicts from another picture than the
encoder used drops it by 5 dB or more on those frames. To see what a driver
wrote into a slice header, use
`ffmpeg -i output.h264 -c copy -bsf:v trace_headers -f null -`.

AV1 has no B-frame type. The reference frame after a group is encoded first as
a hidden frame, whose bytes go in front of the next packet, and a
`show_existing_frame` header made on the CPU shows it later
(`src/encoder/av1/references.rs`). So AV1 packets are in display order, and
`ffprobe` reports no B-frames for them.

What the drivers do with B-frames, as far as it was checked:

- NVIDIA (615.78.08) reports pyramid support for H.264 and AV1, and flat only
  for H.265. So the H.265 pyramid does not run in the tests on that driver. It
  did decode correctly there once with the capability check forced off.
- An H.264 pyramid needs `ref_pic_list_modification` on the P-frame after it,
  and drivers differ in how they write the operations they are given. NVIDIA
  writes exactly the entries it gets: the end operation has to be one of them,
  for the list operations, and then for the marking operations too. It ends
  the marking operations by itself only in a P-slice with no list operations.
  This is checked on NVIDIA only, so `implemented` in `src/encoder/bframes.rs`
  offers the H.264 pyramid on that vendor only. Before offering it on another
  driver, check the slice headers there with `trace_headers`.
- NVIDIA's H.264 B-frames use forward prediction only (no skip, no direct, no
  list 1), so they cost more bits than P-frames. ffmpeg's `h264_vulkan` with
  `-bf 2` gives the same on that GPU, so it is the driver's and not
  pixelforge's. H.265 and AV1 B-frames save bits there.
- RADV and ANV were not available when B-frames were written. Nothing about
  B-frames is checked on them.
- NVIDIA (615.78.08) writes its own value for some syntax, whatever
  pixelforge asks for: `sps_temporal_mvp_enabled_flag = 0` (H.265),
  `enable_ref_frame_mvs = 0`, `enable_jnt_comp = 0` and
  `skip_mode_present = 0` (AV1). Those tools would help B-frames most, so
  they are worth trying on another driver, with
  `ffmpeg -c copy -bsf:v trace_headers` to see whether it writes them.

`scripts/compare_b0.sh` builds v0.11.0 in a worktree and compares its output
with the working tree's for each codec, with CQP and CBR. Without B-frames all
streams now differ from v0.11.0, for four reasons, all intended: H.264 has its
deblocking filter on, H.264 CBR has no QP bounds, H.265 has 8x8 as its
smallest coding block, and AV1 uses CDEF and loop restoration. Before those
changes, `h265-*-gop0` was the only difference: there v0.11.0 gave the driver
a picture order count that wrapped at 512, and the count no longer wraps. Run
it after a change to the encode flow, with a base revision that has those
changes (`scripts/compare_b0.sh <revision>`), for an exact comparison.

A test binary that builds many contexts must call `common::init_logging()`
first, also for another reason than logging: the NVIDIA driver can be loaded
only about 26 times in one process, and that call keeps it loaded.

Tests run one at a time (`RUST_TEST_THREADS=1` in `.cargo/config.toml`): the
NVIDIA driver can deadlock when one thread destroys a device while another
creates objects on a different one.

Make sure there are no Vulkan validation layer errors during execution. Enable
them with `PIXELFORGE_VALIDATION=1`; the layer's messages are routed through
`tracing`, so pair it with `RUST_LOG=warn` (or `debug` for the layer's own
chatter). An integration test needs a subscriber installed for any of it to
show: `common::init_logging()` installs one, and a new test that builds a
context has to call it or install its own, or its validation messages go
nowhere. They are captured like any test output, so run with `--nocapture` to
see a passing test's. Without `VK_LAYER_KHRONOS_validation` installed,
pixelforge logs a warning and carries on with validation disabled, so absence
of errors means nothing if the layer is missing.

On NVIDIA (615.78.08) a decode submit under validation crashes inside the
driver (`vkQueueSubmit2` from `DecodePipeline::submit_decode`), so every test
that decodes (`roundtrip`, `adopted`, `submit`'s repeat test, and the
`rgb_encode` tests that compare decoded frames) segfaults there with `PIXELFORGE_VALIDATION=1`. Run those
without validation on that driver, and check their encode half under
validation separately.

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
- `examples/` - Short usage examples, one use each
- `tools/` - Command-line tools to check the library (workspace crate, not
  published)
- `tests/` - Integration tests; the device + ffmpeg checks are `#[ignore]`d and
  run with `cargo test -- --ignored`
- `benches/` - The encode timing/throughput benchmark (`cargo bench`)
- `testdata/` - Test input files (note: `test_frames.yuv` is a git-LFS pointer;
  generate frames locally if LFS content is unavailable)
- `tests/data/` - Small H.264, H.265 and AV1 (`tests/data/av1/`) streams used
  by the decoder tests

## Key Dependencies
- Vulkan Video API
- ash (Vulkan bindings for Rust)
