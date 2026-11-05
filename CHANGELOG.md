# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add `EncodeConfig::new(codec, width, height)`, for a codec chosen at
  runtime.
- Add RGB input: `EncodeConfig::with_source(Source::Rgb(..))`. The encoder
  converts RGB frames to the YUV its colour description describes, through
  `VK_VALVE_video_encode_rgb_conversion` where the device can and the colour
  converter's shader otherwise (`Encoder::conversion`,
  `EncodeConfig::with_conversion`).
- Add `Encoder::submit` and `Frame`: the one way to hand over a frame,
  whatever the conversion, with its GPU waits.
- Add `SourceImage` and `Handoff`: a caller's image to encode from. It keeps
  track of the frames that read it (`SourceImage::released`), and its drop
  waits for them.
- Add `Encoder::input_slot` and `InputSlot`: the encoder's own input image,
  to write a frame in place and submit with `Frame::input`.
- Add `InputImage::source`.
- Add `Encoder::repeat`: encode the last frame again from the encoder's own
  input image, for a key frame while no new frame arrives. The caller's
  source does not have to stay alive for it.
- Add DMA-BUF input (feature `dmabuf`): `Encoder::input_formats`,
  `Encoder::import`, `Encoder::allocate_input`, `ExternalImage` and
  `Frame::external`. The encoder reads a buffer directly where the device
  can, and copies it otherwise.
- Add sync_file fences (feature `dmabuf`): `Frame::wait_sync_file`,
  `Frame::signal_sync_file` and `EncodeFuture::take_release_sync_file`.
- Add `PixelForgeError::DeviceLost`: a submit or wait that finds the device
  lost reports it as such, so a caller knows to make a new context.
- Add `VideoContext::supports_rgb_encode_input`,
  `VideoContext::has_video_encode_rgb_conversion` and
  `DeviceFeatures::video_encode_rgb_conversion`.

### Changed

- The copy of a frame into the encoder's input image no longer waits on the
  CPU: the encode waits for it on the GPU. `Encoder::submit` returns before
  the source is read.
- **Breaking:** `Encoder::encode`, `Encoder::encode_after` and
  `Encoder::input_image` are removed. Submit a `Frame` with `Encoder::submit`:
  `Frame::image(&source_image)` for an image of the caller's,
  `Frame::input(encoder.input_slot()?)` for the input image written in place.
  An imported image that `UNDEFINED` used to mark is `Handoff::External`.
- **Breaking:** `ColorConverter`, `ColorConverterConfig` and `OutputFormat`
  are no longer public. Describe an RGB source with `Source::Rgb` instead, and
  the encoder runs the converter itself, with the right target and range.
- **Breaking:** `RateControl` replaces `RateControlMode`, `quality_level`,
  `target_bitrate` and `max_bitrate`: `Cqp { qp }`, `Cbr { bitrate }` or
  `Vbr { target, max }`, so each mode carries exactly the values it uses.
  `Encoder::new` refuses a quantizer out of the codec's range, a bitrate of 0,
  and a peak below the target. `RateControlMode::Disabled` was the same as
  `Cqp` and is gone. `DEFAULT_TARGET_BITRATE` and `DEFAULT_MAX_BITRATE` are
  removed; `DEFAULT_AV1_Q_INDEX` is added.
- An RGB conversion the device cannot do is `PixelForgeError::NoSuitableDevice`,
  as other missing device support is, rather than `InvalidInput`.
- **Breaking:** `EncodeConfig::with_b_frames` and the public `b_frame_count`
  field are removed. No codec encodes B-frames yet, and every encoder refused
  a count above 0 at runtime.
- **Breaking:** `InputImage` has two uploads, `upload` to its own image and
  `upload_to` an encoder's `InputSlot`, which take a `YuvLayout`: `I420`,
  `Nv12`, `I420P10` or `I444`. They replace `upload_yuv420`, `upload_nv12`,
  `upload_yuv420_10bit`, `upload_yuv420_to` and `upload_yuv444_to`, which
  covered only some of the combinations. A layout the image cannot hold is
  an error; `upload_nv12` panicked for a 10-bit or 4:4:4 image. The uploads
  wait until the frames that read the image are done with it.
- **Breaking:** `InputImage::new` no longer takes a `Codec`, which it did not
  use. `InputImage::for_encoder` makes one for an encoder's frames.
- **Breaking:** `EncodeConfig::frame_rate` (a `FrameRate`) replaces
  `frame_rate_numerator` and `frame_rate_denominator`. `with_frame_rate` is
  unchanged.
- **Breaking:** the encoder's `BitDepth` is exported as `BitDepth`, not
  `EncodeBitDepth`. `Dimensions` and `FrameRate` are exported from the crate
  root.
- **Breaking:** the `DEFAULT_*` encoder constants are no longer public.
  `EncodeConfig::new` and `RateControl::default_for` apply them.
- **Breaking:** `PixelForgeError` has fewer, typed variants. A failed
  Vulkan call is `Vulkan { call, result }`, with the function's name and its
  `vk::Result`, in place of `DeviceCreation`, `VideoSessionCreation`,
  `SessionParametersCreation`, `MemoryAllocation`, `ResourceCreation`,
  `CommandBuffer`, `Synchronization`, `QueryPool` and `Vulkan(vk::Result)`.
  A codec or memory type the device lacks is `NoSuitableDevice`, which
  replaces `CodecNotSupported`. New are `Driver`, for a driver that reports
  success with no usable result, `Io`, for a failed OS call, and
  `Cancelled`, for a frame whose encoder shut down. `From<vk::Result>` is
  removed.
- Every failed Vulkan call on a lost device is `PixelForgeError::DeviceLost`.
  Some were reported as another variant.
- **Breaking:** `EncodedPacket::is_key_frame` is a method, not a field. An
  IDR's `frame_type` is `FrameType::Idr`; it was `FrameType::I`.
  `FrameType::Unknown`, which no packet had, is removed.
- **Breaking:** `VideoContextBuilder::device_requirements` and
  `build_from_existing` adopt a caller's device for the codecs asked for with
  `require_encode` and `require_decode`, so one adopted context can encode
  and decode. They replace `encode_device_requirements`,
  `decode_device_requirements`, `build_from_existing_encode` and
  `build_from_existing_decode`. At least one codec must be asked for: an
  encode adoption with none no longer takes every codec the device has.
- **Breaking:** `StreamColor` replaces the raw `ColorDescription` in the
  encode API: `EncodeConfig::with_color` replaces `with_color_description`,
  `Encoder::reconfigure` replaces `set_color_description`, and
  `VideoContext::supports_rgb_encode_input` takes a `StreamColor`. `Sdr` and
  `Hdr10` name the two colours an RGB source converts to, with their range;
  `Other` carries any H.273 description, for a YUV source only.
- `Encoder::reconfigure(source, color)` changes the source and the stream
  colour together, so the conversion follows both: the colour converter
  switches in place, and where the encoder converts it is made again. Packet
  timestamps carry on.
- The `dmabuf` feature enables its extensions only where the device has them,
  rather than failing device creation where it does not.
- `EncodedPacket::stats` is always there, and only its `gpu_time_ns` is
  optional. Without encode queue timestamps, a packet still has its wall
  times.

### Fixed

- The source image barrier of the input copy now waits for earlier work on
  the image, rather than starting at `TOP_OF_PIPE`.
- The GPU encode time is correct when the timestamp counter wraps between
  the two timestamps.
- The shaders are SPIR-V 1.3, which a Vulkan 1.1 device must accept. They
  were SPIR-V 1.6, which only a Vulkan 1.3 device must accept.
- An H.264 encoder for a profile the device does not support no longer asks
  for its quality level properties before it fails.
- `wall_latency_ns` starts before the encode submit. It started after it, so
  a submit that blocked in the driver made it shorter than the GPU time.

### Removed

- The encoder's internal modules are no longer public:
  `encoder::{av1, bitwriter, dpb, gop, h264, h265, reorder, resources}` and
  `decoder::h264`. Nothing in them was needed to encode or decode.

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
