# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- H.264 hardware decoding via Vulkan Video: `Decoder` and `DecodeConfig`.
  Stream-driven, so the
  Vulkan session is created from the stream's own parameter sets and a mid-stream
  resolution change is handled transparently. Verified byte-identical to
  `ffmpeg -pix_fmt nv12` on AMD (RADV), NVIDIA and Intel (ANV).
- Asynchronous decoding, split into a sink and a source. `DecodeSink::decode`
  submits without waiting; frames arrive on a `DecodeSource` as the GPU
  finishes with them, in presentation order. `Decoder` holds both halves for
  single-threaded use, and `Decoder::split` separates them so a producer and a
  consumer can run on their own threads. `DecodeSource::next_frame` awaits the
  next frame and `try_next_frame` takes one only if it is ready. Measured
  10-21% higher decode throughput than the previous synchronous decoder.
- A decoded frame's image is now valid for exactly as long as the frame, with
  no exceptions. A session rebuild, which a resolution or parameter set change
  causes, used to destroy the images of frames still in flight; they are now
  reference counted per image, so a rebuild frees only what nothing is holding
  and the last frame to be dropped releases the rest. Before this, a mid-stream
  resolution change crashed a consumer reading a frame decoded just before it.
- `DecodedFrame::generation` says which set of decoder images a frame belongs
  to. Not needed for validity, but a consumer caching per-image state (a
  renderer's views, say) must key on `(generation, image, array_layer)`:
  handles are reused once a generation's images are finally released.
- DPB slot reservations are per session, so a frame outliving a rebuild
  releases into the set it was decoded under and cannot free a slot the current
  session reserved for someone else.
- `DecodeSink::decode` returns a `DecodeStatus` (`Decoded`, `Buffered` or
  `NeedsKeyframe`) rather than reporting a missing keyframe as an error. Data
  that cannot be decoded yet is the normal state of affairs when joining a live
  stream or recovering from loss, so it is no longer something a caller has to
  filter out of their error handling. `PixelForgeError::NeedsKeyframe` is gone.
- `DecodeSink::finish` ends a stream: it decodes whatever framing still holds
  back, emits the frames reordering held back, and closes the source.
- Zero-copy output in presentation order: a decoded picture is never copied on
  its way to the caller. A picture waiting its turn in display order stays
  pinned in the DPB slot it was decoded into, and that pin passes to the
  `DecodedFrame` the caller receives. Devices with too few DPB slots for the
  stream fall back to copying rather than failing.
- `DecodeConfig::with_output_depth` reserves DPB slots so decoded frames can be
  held while decoding continues.
- Decoded picture images are created with `VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT`
  where the driver reports it in `VkVideoFormatPropertiesKHR::imageCreateFlags`,
  paired with `VkImageFormatListCreateInfo` naming the plane formats.
  `DecodedFrame::plane_views` says whether it worked. A consumer can then view
  the luma and chroma planes separately, as `R8_UNORM` and `R8G8_UNORM`, and
  read a decoded frame with no `VkSamplerYcbcrConversion` at all. That matters
  because some shader toolchains cannot express a combined image sampler with an
  immutable ycbcr sampler (naga, and so wgpu), leaving those renderers no
  zero-copy path otherwise. Both RADV and ANV allow it for H.264 4:2:0, and it
  costs no measurable decode throughput on either. Frames from the copying path
  always allow it, since those images carry no video profile.
- `query_capabilities` reports decode picture `imageCreateFlags`, for both the
  usage pixelforge creates pictures with and the reference-only DPB, which are
  not the same answer.
- `examples/sample_planes` reads decoded frames through per-plane views with no
  ycbcr conversion and no sampler, the path a wgpu-style renderer needs, and
  writes the samples through unchanged so its output is byte-identical to the
  decoder's own NV12.
- `examples/sample_frame` shows the render path end to end: a decoded frame
  sampled in a compute shader through a `VkSamplerYcbcrConversion`, with no
  copy and no layout transition, while the decoder is still using that picture
  as a reference.
- Host readback and `copy_frame_to_planes` are gone. A `DecodedFrame` is a GPU
  image the consumer owns until they drop it, and what to do with it is theirs
  to decide; `examples/common` shows one way to read one back.
- Decoded pictures are created with `SAMPLED` usage where the device allows it,
  so a renderer can read a `DecodedFrame` in a shader instead of copying it out
  first. `DecodedFrame::sampleable` reports whether it worked.
- `DecodeConfig::with_consumer_queue_family` names the queue family that will
  read decoded frames, adding it to each picture's sharing set so frames can be
  used from a graphics queue with no ownership transfer.
- `VK_KHR_unified_image_layouts` is enabled where the device supports it, with
  both `unifiedImageLayouts` and `unifiedImageLayoutsVideo`. Decoded pictures
  then live in `VK_IMAGE_LAYOUT_GENERAL` and are never transitioned, which is
  what makes it safe to hand one to a consumer while the decoder is still using
  it as a reference. Without it every frame is copied into a private image
  instead. Measured on RADV: 25% higher decode throughput than the copying
  path, and 12% higher with host readback on top.
- `VideoContextBuilder::declare_unified_image_layouts` lets a caller who adopts
  their own device say they enabled the extension, since Vulkan cannot be asked
  which features a device was created with.
  `DeviceRequirements::unified_image_layouts` reports whether it is worth doing.
- `VideoContextBuilder::without_unified_image_layouts` forces the copying path,
  so it can be exercised on hardware that would otherwise never take it.
- `Framing` says how input is framed, so the decoder can do the framing itself.
  `Framing::FrameAligned` (the default) takes whole coded frames per call, as a
  container or transport delivers them. `Framing::ByteStream`, selected with
  `DecodeConfig::with_byte_stream`, takes a stream that may cut anywhere and
  buffers a partial trailing frame until later bytes complete it. This replaces
  the `Decoder::split` free-standing splitter, which needed the whole stream in
  memory and so could not be used for live decoding.
- Validation layer messages are now routed into `tracing` through a
  `VK_EXT_debug_utils` messenger. Previously the layer was loaded with nowhere to
  report, so enabling validation verified nothing.
- Encoding on a caller-created device: `encode_device_requirements` and
  `build_from_existing_encode`, the counterparts of the decode pair. An
  application that renders on its own device can encode its images there
  directly, with no external memory and no second device.
- `DeviceRequirements::features` names the device features pixelforge needs as
  booleans, so a caller can set them in whichever feature structs it already
  chains.
- `VideoContextBuilder::with_encode_queue`, `with_decode_queue`,
  `with_transfer_queue` and `with_compute_queue` choose exactly which queue an
  adopted context submits to, so the caller can keep its own queues to itself.
  `DeviceRequirements::queues` says which family serves which role, and
  `DeviceRequirements::internally_synchronized_queues` whether the device can
  share a queue safely instead.
- `ColorConverter::convert_async` and `Encoder::encode_after` wait for, and in
  the converter's case signal, a `TimelinePoint`, so a frame can go from the
  caller's rendering through conversion to encode ordered on the GPU, with no
  CPU wait in between.
- `EncodeConfig::with_rgb_input` hands RGB frames to the encoder and lets it
  apply the YUV matrix itself, through `VK_VALVE_video_encode_rgb_conversion`.
  `ColorConverterConfig::rgb_encode_input` says when a conversion can be done
  that way instead of by the shader. Based on the approach in #14.

### Changed

- The colour converter binds its descriptors with push descriptors instead of
  `VK_EXT_descriptor_buffer`, which is deprecated. It now needs only
  `VK_KHR_push_descriptor`, and no longer buffer device addresses or
  descriptor buffer capture-replay. `VideoContext::has_descriptor_buffer` is
  replaced by `VideoContext::has_push_descriptor`.
- Adopted devices may come from an instance asking for any Vulkan version from
  1.1 up. Pixelforge reaches synchronization2 through its extension rather than
  core 1.3, and the requirements list `VK_KHR_timeline_semaphore`.

- Adopting a caller-created device for decode (`build_from_existing_decode`) now
  requires the `timelineSemaphore` feature in addition to `synchronization2`.

### Fixed

- `samplerYcbcrConversion` and the 2-plane 4:4:4 formats feature were only
  enabled on devices that also had descriptor buffers.
- The colour converter reused its source image view whenever the next source
  had the same handle. Drivers reuse the handles of destroyed images, so a
  caller creating a new source per frame got every frame after the first
  converted from a freed image.

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
