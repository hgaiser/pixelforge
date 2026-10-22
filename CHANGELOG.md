# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add RGB input converted by the encoder itself through
  `VK_VALVE_video_encode_rgb_conversion`: `EncodeConfig::with_rgb_input`.
- Add `VideoContext::supports_rgb_encode_input` and
  `ColorConverterConfig::rgb_encode_input`, which say when a conversion can be
  left to the encoder.
- Add `VideoContext::has_video_encode_rgb_conversion` and
  `DeviceFeatures::video_encode_rgb_conversion`.

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
