# Encode input: design

This document describes how frames get into the encoder. It replaces the
`EncodeConfig::with_rgb_input` / `ColorConverterConfig::rgb_encode_input`
pair, where the caller chose the conversion path and called a different
method for each.

## Goals

1. The caller describes the source one time. Pixelforge selects the path.
2. One encode entry point for all paths, and no second, older way.
3. No CPU wait in the hot path. The GPU orders all work.
4. The caller knows when it can reuse a source image.
5. A DMA-BUF source goes to the encoder with no pixelforge copy, when the
   device can read it directly.
6. Colour changes go through one method, which updates the path that runs.

## The source description

`EncodeConfig` gets a `Source`:

- `Source::Yuv` (the default): frames are already YUV, in the encoder's
  pixel format and bit depth. This is the old behaviour.
- `Source::Rgb(RgbSource { format, color, reference_white_nits })`: frames are
  RGB. `format` is the pixel layout (`InputFormat`) and `color` is what the
  pixels mean (`ColorSpec`).

The stream's colour is `EncodeConfig::color`, a `StreamColor`:

- `Sdr { range }`: BT.709. An RGB source converts to `ColorSpec::Srgb`.
- `Hdr10 { range }`: BT.2020 with PQ. An RGB source converts to
  `ColorSpec::Bt2020Pq`.
- `Other(ColorDescription)`: any other H.273 description, for a YUV source
  whose pixels already are in it. It is an error for an RGB source, because
  no conversion produces it, even when its codes say BT.709.

### Path selection

`Encoder::new` selects one `Conversion`:

| Conversion | When | What runs |
|---|---|---|
| `None` | YUV source | Nothing. |
| `Encoder` | RGB source, `VK_VALVE_video_encode_rgb_conversion`, source space equals target space, 4:2:0 at the input's bit depth, and the driver accepts the matrix, range and format | The encoder converts. The input image is RGB. |
| `Shader` | Any other RGB source the shader can convert | The colour converter writes into the encoder's YUV input image. |

`EncodeConfig::with_conversion(ConversionPreference)` can force `Encoder` or
`Shader`. `Auto` (the default) prefers `Encoder`. A forced path that cannot
work is an error from `Encoder::new`. `Encoder::conversion()` tells which path
was selected.

The colour converter is not public. An RGB source is the only way to use it,
so its target and range always follow the stream's colour.

### Source and colour changes

`Encoder::reconfigure(Source, StreamColor)` is the one way to change the
source or the colour. Both decide the conversion, so they change together:
there is no moment where a new source meets the old colour, a pair the
conversion may not do. A call that changes nothing does nothing, so a caller
whose frames change format or colour space (a compositor) calls it for every
frame.

- `None` kept: the session parameters are made again for a new colour.
- `Shader` kept: the converter switches in place. Its input format, source,
  target and range are all push constants, and its source view is made per
  conversion. The session parameters are made again for a new colour.
- `Encoder`, or a change of conversion: the input format, matrix and range
  are fixed in the video session, so the codec encoder is made again, and the
  path is selected again. Packet timestamps continue from where they were.

A new colour makes the next frame an IDR in all cases.

This removes the bug where the label changed and the hardware conversion did
not.

## Submitting a frame

`Encoder::submit(Frame)` is the one way to encode a frame:

```rust
let slot = encoder.input_slot()?;
// ... write the frame into slot.image() ...
let future = encoder.submit(Frame::input(slot).wait(&points))?;

let future = encoder.submit(Frame::image(&source_image).wait(&points))?;
let future = encoder.submit(Frame::external(&imported).wait_sync_file(fence))?;
```

A `Frame` names its source:

- `Frame::input(InputSlot)`: the encoder's own input image for the current
  slot, written in place. `Encoder::input_slot()` waits for the slot and
  returns the slot. A slot is valid only until the encoder's next submission:
  a later submit of it is an error. With `Shader` there is no slot, because
  the input image belongs to the converter.
- `Frame::image(&SourceImage)`: a `vk::Image` the caller owns. Pixelforge
  copies it (`None`, `Encoder`) or converts it (`Shader`) into the input
  image.
- `Frame::external(&ExternalImage)`: a DMA-BUF that pixelforge imported or
  allocated. See below.

### Source images

A `SourceImage` wraps a caller's `vk::Image` with a `Handoff`, which says how
each frame gets the image and gives it back:

- `Handoff::Device { layout }`: the caller writes it on this device and hands
  it over in `layout`. Pixelforge leaves it in `layout`. `UNDEFINED` and
  `PREINITIALIZED` are refused, because they keep no pixels.
- `Handoff::External`: imported memory, written outside this device. Each
  frame acquires it from the foreign queue family, and releases it back in
  `GENERAL`. The first acquire uses `UNDEFINED`.

The handle keeps the release point of each frame that read the image.
`SourceImage::released()` gives those points. `wait_released()` waits for
them on the CPU, and the drop waits for them too. `InputImage` holds one, so
its uploads wait for the previous frames by themselves.

### No CPU waits

The copy from a caller image into the input image signals a timeline
semaphore, and the encode waits on it on the GPU. The copy no longer waits on
a fence. Each pipeline slot has its own upload command buffer. A slot is used
again only after its encode is complete, and that encode waited for the copy,
so the reuse is safe.

The only CPU wait left is backpressure: when every pipeline slot is in
flight, `submit` waits for the oldest one.

### Release point

The image, not the future, holds the point after which the source can be
written again, or given back to its producer:

| Source | Released by |
|---|---|
| `InputSlot` | the encode |
| `SourceImage`, `None` / `Encoder` | the copy |
| `SourceImage`, `Shader` | the conversion |
| `ExternalImage`, read directly | the encode |
| `ExternalImage`, copied | the copy |

`SourceImage::released()` and `ExternalImage::released()` give it. A slot
needs none: the next `input_slot()` waits for the slot.

The semaphores behind these points are retired to the context when the
encoder is dropped, as the converter already does. A wait on a point that the
caller already submitted stays valid.

## DMA-BUF input (feature `dmabuf`)

### Which buffers the encoder reads directly

`Encoder::input_formats()` returns the DRM format (fourcc) that the encoder
wants, and for each DRM modifier the device knows:

- `zero_copy`: the encoder's path reads it with no pixelforge copy. For
  `None` and `Encoder` that means `VIDEO_ENCODE_SRC` usage with the
  encoder's video profile. For `Shader` it means `SAMPLED` usage, because the
  shader reads the source directly.
- `importable` and `exportable`: whether a DMA-BUF with this modifier can be
  imported, or allocated by pixelforge and exported.

A caller negotiates with its producer: it offers the `zero_copy` modifiers
first, then the others.

### Allocating and exporting (consumer-allocated buffers)

`Encoder::allocate_input(modifiers)` makes an `ExternalImage` with one of the
given modifiers. It prefers one the encoder reads directly; if there is none,
it takes one it can copy from.
`ExternalImage::export()` gives the DMA-BUF (fd, modifier, plane offsets and
strides). This is for `ext-image-copy-capture`, where the client supplies the
buffer and the compositor writes into it.

The protocol requires the buffer to be exactly the capture size. Thus an
exported image has the display size, not the aligned coded size. For H.264
and H.265 the encoder then gives the display size as the source picture
extent. The spec allows this, and the stream headers already crop to the
display size. AV1 already uses the display size.

### Importing (producer-allocated buffers)

`Encoder::import(&DmaBuf)` makes an `ExternalImage` from a producer's buffer.
It is checked one time, at import:

- If the modifier is `zero_copy`, the encoder reads it directly.
- Otherwise, if it can be imported for transfer, `submit` copies it.
- Otherwise the import fails.

Only single-buffer DMA-BUFs are supported: all planes in one buffer, at
different offsets. Disjoint planes are an error.

### Ownership

A DMA-BUF is shared with another process. On each frame, pixelforge acquires
it from `VK_QUEUE_FAMILY_FOREIGN_EXT` (or `EXTERNAL` without
`VK_EXT_queue_family_foreign`) in the submission that reads it, and releases
it back in the same submission. The first acquire uses `UNDEFINED` as the old
layout, as the converter already does for a first-time import, and later ones
use `GENERAL`.

### Lifetime

An `ExternalImage` remembers the release point of its last frame. Its `Drop`
waits for that point before it destroys the image.

## sync_file fences (feature `dmabuf`)

- `Frame::wait_sync_file(fd)`: the frame's first submission waits for the
  fence. Pixelforge imports it into a binary semaphore (temporary import),
  which takes ownership of the fd.
- `Frame::signal_sync_file()`: the submission that releases the source also
  signals a binary semaphore, and `EncodeFuture::take_release_sync_file()`
  gives it as a sync_file. The caller can attach it to the DMA-BUF
  (`DMA_BUF_IOCTL_IMPORT_SYNC_FILE`) or give it to the producer.

Each pipeline slot has one acquire and one release semaphore. They are used
again only after the slot's encode is complete.

### Driver quirks

NVIDIA's driver (615.78.08) reports more DMA-BUF support than works, and
both failures give a wrong picture, not an error. `quirks_allow` in
`src/encoder/external.rs` holds the driver back:

- An image made with a video profile reads a block-linear buffer as black.
  Imported without the profile, the same buffer reads correctly. So the
  encode never reads a DMA-BUF directly on NVIDIA; it is copied.
- A two-plane YUV buffer with a block-linear modifier loses its chroma plane
  between two Vulkan images, with no pixelforge code involved. Only LINEAR is
  offered for YUV on NVIDIA. Single-plane RGB is not affected.

The same driver lays a LINEAR image out by its usage and ignores the plane
layout an importer gives. So an image pixelforge allocates for export also
gets `TRANSFER_DST`, the usage a producer needs to write into it.

`PIXELFORGE_NO_DMABUF_QUIRKS=1` turns the quirks off, to try a newer driver.
The direct read path can only be checked on a device without them (RADV).

## Not in this change

- Selecting the GPU by DRM render node (PR #23). DMA-BUF import and export
  need the encoder on the producer's GPU. Today a caller can make sure of
  that with `build_from_existing_encode`.
- Implicit sync done by pixelforge itself (the DMA-BUF sync_file ioctls). The
  caller does this with the fds above.
- Disjoint multi-plane DMA-BUFs.
