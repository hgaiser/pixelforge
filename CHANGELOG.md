# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add B-frame encoding for H.264, H.265 and AV1: `EncodeConfig::with_b_frames`, at most `MAX_B_FRAMES`.
- Add B-frame pyramids: `EncodeConfig::with_b_frame_references` and `BFrameReferences`.
- Add `VideoContext::supports_b_frames` and `BFrameSupport`.
- Add `--b-frames` and `--pyramid` to the `encode` example.
- Add `EncodeConfig::with_quality_level` and `VideoContext::encode_quality_levels`. The default stays 0, the fastest level. `Encoder::new` returns an error for a level the device does not have.
- Add `YuvDownloader` and `DownloadedFrame`, to copy decoded frames to CPU memory: the reverse of `YuvUploader`.
- Add the `encode_gpu` example: encode RGB frames that the application draws into its own Vulkan image.
- Add the `tools` workspace crate (not published), with `encode` and `decode` tools that have the options for checking the library.

### Changed

- The examples are short and show one use each. `encode` and `decode` lost their testing options (now in the tools), `query_capabilities` uses pixelforge's own queries, and `sample_planes` is now a test (`tests/sample_planes.rs`) that compares plane views with `YuvDownloader` on both decode paths.

- **Breaking:** `EncodedPacket::dts` is an `i64`. With B-frames it starts below zero.
- **Breaking:** A future from `Encoder::submit` resolves with the next packet in decode order. With B-frames that is not the packet of the submitted frame.
- With B-frames, `Encoder::invalidate_reference_frames` drops every reference encoded from the first lost frame on, and the next frames predict from the references that survive. If none survives, the oldest frame that waits for its group is encoded at once as the IDR.
- With a constant QP, the B-frames of a pyramid get one QP step more for each level (a q-index step of 4 for AV1). With 7 B-frames, streams are 2% to 4% smaller at the same quality.
- The H.264 encoder writes the lowest level from 4.1 that the stream needs, by its size, frame rate, DPB and peak bitrate. It always wrote 4.1, also where the stream needed more. `max_dec_frame_buffering` is one frame smaller (NVIDIA already wrote that value).
- The H.264 and H.265 reference pictures carry their own picture type to the driver, not always P.
- With B-frames, a DMA-BUF input is always copied.
- H.265 streams with more than 256 frames between two IDR frames differ from 0.11.0: the encoder now gets the picture order count without a wrap.
- The AV1 encoder uses CDEF and loop restoration. At the same quality, streams are 3% to 10% smaller, and each frame takes longer to encode (about 0.6 ms at 1080p on an RTX 5080).

### Fixed

- Fix the H.264 decoder's reorder depth for streams without a VUI.
- Fix the H.264 encoder, which turned the deblocking filter off. At the same quality, streams are about 10% smaller.
- Fix the H.264 encoder's CBR and VBR, which held the QP between 18 and 42. On easy content the bitrate stayed far below the target.
- Fix H.264 reference frame invalidation: the P-frame after it predicted from the lost frames in a decoder, because it did not reorder its reference list. The quality dropped by about 8 dB from there on.
- Let the H.265 encoder use 8x8 coding blocks. The minimum was 16x16. At the same quality, streams are 3% to 18% smaller.

## [0.11.0] - 06-11-2026

### Added

- Add H.265 and AV1 decoding: `DecodeConfig::new(codec)`.
- Add `EncodeConfig::new(codec, width, height)` for a codec chosen at runtime.
- Add RGB input: `Source::Rgb`, converted by the encoder where the device supports it. (#46, @DatCaptainHorse)
- Add `Encoder::submit` and `Frame`, to submit frames without CPU waits.
- Add `Encoder::input_slot` and `InputSlot`, to write frames directly into the encoder's input image.
- Add `Encoder::repeat`, to encode the last frame again.
- Add `PixelForgeError::DeviceLost`, reported when the device is lost.
- Add `VideoContext::supports_rgb_encode_input`.

### Changed

- **Breaking:** `Encoder::encode`, `encode_after` and `input_image` are removed. Use `Encoder::submit` instead.
- **Breaking:** `RateControl` replaces `RateControlMode`, `quality_level`, `target_bitrate` and `max_bitrate`.
- **Breaking:** `DecodeConfig::h264()` is replaced by `DecodeConfig::new(codec)`.
- **Breaking:** `ColorConverter` is no longer public.
- **Breaking:** `EncodeConfig::with_b_frames` is removed. No codec encodes B-frames yet.
- **Breaking:** `StreamColor` replaces `ColorDescription` in the encode API.
- **Breaking:** `EncodedPacket::is_key_frame` is a method. IDR frames are `FrameType::Idr`.
- **Breaking:** Several types are renamed, including `InputImage` to `YuvUploader` and `PixelFormat` to `ChromaFormat`.
- **Breaking:** `PixelForgeError` has fewer, typed variants.
- `Encoder::reconfigure(source, color)` changes the source and stream colour together.
- `EncodedPacket::stats` is always present. Only `gpu_time_ns` is optional.

### Fixed

- Fix the reference window being smaller than the references per frame.
- Fix H.264 encoding on profiles the device does not support.
- Fix shaders to SPIR-V 1.3, so Vulkan 1.1 devices accept them.
- Fix packet wall latency, which started after the encode submit.
- Fix RGB chroma siting for drivers that site chroma differently. (#49, @porkloin)

### Removed

- The encoder's internal modules are no longer public.

## [0.10.0] - 19-10-2026

### Added

- Add H.264 decoding support.
- Add queue selection for adopted devices.
- Add encoding on a caller-created device.
- Add `VideoContext::supports_encode_format`.
- Route validation layer messages into `tracing`.

### Changed

- Use push descriptors in the colour converter instead of `VK_EXT_descriptor_buffer`.
- Require the `timelineSemaphore` feature when adopting a device for decode.
- Allow adopted devices to use any Vulkan version from 1.1 up.

### Fixed

- Enable the ycbcr and 4:4:4 features without requiring descriptor buffers.
- Stop the colour converter reusing a stale source view.
- Order the DPB encode-layout barrier before the encode.
- Fix H.264 decoding on NVIDIA hardware.
- Check encode requirements on the queue family the caller chose.
- Stop requesting 4:4:4 and AV1 encode features when the driver has them disabled.
- Keep a converter's timeline semaphore alive until the context is dropped.

### Removed

- Remove host readback and `copy_frame_to_planes`.
- Remove `PixelForgeError::NeedsKeyframe`.
- Remove the free-standing stream splitter.

## [0.9.1] - 01-09-2026

### Fixed

- Reset AV1 coding state on every key frame so CBR/VBR streams with on-demand IDRs stay decodable. (#31, @lutyjj)

## [0.9.0] - 18-08-2026

### Fixed

- Detect when an encoded frame overflows its bitstream buffer and report it instead of reading out of bounds, preventing a segfault on large scene-cut frames.
- Scale the H.264/H.265 bitstream buffer by resolution so a single frame can't overflow the fixed minimum.

## [0.8.1] - 02-08-2026

### Fixed

- Check for `VK_EXT_ycbcr_2plane_444_formats` support before enabling it, so device creation succeeds on devices without the extension. (#29, @DatCaptainHorse)

## [0.8.0] - 02-08-2026

### Added

- Add `ColorDescription::with_full_range` and `ColorDescription::is_hdr`. (#30, @lutyjj)

## [0.7.2] - 2026-07-29

### Fixed

- Acquire the queue family on the first DMA-BUF import. (#25, @schlegp)
- Fix AV1 encoding on AMD RADV — layered DPB, converter layout barrier, GOP clamping. (#27, @schlegp)
- Skip encode timestamp queries on unsupported queues; guard `timestamp_query_pool` destruction. (#28, @schlegp)

## [0.7.1] - 2026-07-21

### Fixed
- `ColorDescription::bt709()` now defaults to `full_range: false` - BT.709 quantization per ITU-R BT.709-6 is limited (studio) range [16–235], not full range. Previously defaulted to `full_range: true`, which caused washed-out SDR blacks when clients decoded as limited range. Fixed in https://github.com/hgaiser/pixelforge/pull/24
- `ColorConverterConfig::new()` now defaults to `full_range: false` to match the BT.709 standard and the `ColorDescription::bt709()` default. (https://github.com/hgaiser/pixelforge/pull/24)

## [0.7.0] - 2026-06-26

### Added
- Encode pipelining with timeline semaphores by @urwrstkn8mare in https://github.com/hgaiser/pixelforge/pull/21
- Async push-based encode readback on a dedicated completion thread by @urwrstkn8mare in https://github.com/hgaiser/pixelforge/pull/21
- Reference frame invalidation (RFI) for H.264/H.265/AV1; AV1 multi-reference prediction by @urwrstkn8mare in https://github.com/hgaiser/pixelforge/pull/21

### Changed
- Unified per-codec encoders into generic `CodecEncoder<C>` with shared `EncoderCommon` by @urwrstkn8mare in https://github.com/hgaiser/pixelforge/pull/21

## [0.6.0] - 2026-06-12

### Added
- CI check for README.md consistency in GitHub Actions workflow.

### Changed
- Cleaner `p_next` chain handling — replaced manual pointer arithmetic with safer `extend`-based construction across H.265 init, session parameters, resources, and Vulkan utilities.

### Fixed
- AV1 reference-frame handling — corrected reference frame list population and cleared clippy warnings.
- AV1 CBR/VBR modes no longer set `min_q_index`/`max_q_index`, which are incompatible with those rate-control modes.
- Missing rate-control push entries in `VkVideoCodingControlInfoKHR` for H.264, H.265, and AV1 encoders.
- AV1 init now uses `extend` for proper `VkVideoEncodeInfoKHR` construction.

## [0.5.0] - 2026-06-09

### Added
- `Bt709LinearToBt2020Pq` color space — converts linear BT.709 (scRGB, FP16) to BT.2020+PQ via gamut mapping + PQ OETF. Used for HDR games that present with `VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT`. `sdr_reference_white_nits` controls the tone-mapping scale (80 nits per IEC 61966-2-2).
- `set_sdr_reference_white_nits()` — dynamically updates the SDR reference white level via push constants without recreating the pipeline.

## [0.4.0] - 2026-06-05

### Added
- `shader/` directory — contains GLSL source (`color_convert.comp`), compile script (`compile.sh`), precompiled SPIR-V (`color_convert.spv`), and documentation (`README.md`).
- Shader development workflow documented in README.md.

### Removed
- `shaderc` dependency — shaders are now precompiled to SPIR-V and embedded at build time via `include_bytes!`. No `glslc` or Vulkan SDK required to build the crate.
- `build.rs` — no longer needed since shaders are precompiled.
- `shader.rs` — SPIR-V constant and `get_spirv_code()` moved to `pipeline.rs`.
