//! Reading a decoded frame back to the CPU.
//!
//! pixelforge hands out a GPU image and stops there: it has no readback path of
//! its own, because a renderer sharing the device does not want one, and a
//! consumer that does want one knows better than the library where the pixels
//! should end up. This is that consumer's side of the deal, shared by the
//! examples that write raw YUV to a file.
//!
//! Two details are worth copying into real code:
//!
//! - A frame in `GENERAL` layout (which is what the zero-copy path hands out)
//!   needs no transition to be copied from. A frame in any other layout is a
//!   private image, so transitioning it is safe.
//! - The copy runs on the transfer queue, which the decoder also uses on the
//!   copying fallback path. Submitting to one `VkQueue` from two threads at
//!   once is undefined, so these examples drive the decoder and the readback
//!   from the same thread.

use ash::vk;
use pixelforge::decoder::{DecodeConfig, DecodedFrame, Decoder, FramePoll};
use pixelforge::encoder::{BitDepth, Codec};
use pixelforge::vulkan::VideoContext;
use std::fs::File;
use std::io::Write;

/// Bytes handed to the decoder per call, standing in for a network read.
pub const CHUNK_SIZE: usize = 64 * 1024;

/// A coded stream read from a file.
pub enum Input {
    /// A raw stream that the decoder frames itself: Annex B for H.264 and
    /// H.265, low-overhead OBUs for AV1.
    ByteStream(Vec<u8>),
    /// Whole coded frames, as a container gives them (IVF).
    Frames(Vec<Vec<u8>>),
}

/// Read `path`, and pick the codec from its extension: `.264`/`.h264`,
/// `.265`/`.h265`/`.hevc`, `.obu`/`.av1` or `.ivf` (AV1 in IVF).
pub fn open_input(path: &str) -> Result<(Codec, Input), Box<dyn std::error::Error>> {
    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let data = std::fs::read(path)?;
    Ok(match extension.as_str() {
        "264" | "h264" => (Codec::H264, Input::ByteStream(data)),
        "265" | "h265" | "hevc" => (Codec::H265, Input::ByteStream(data)),
        "obu" | "av1" => (Codec::AV1, Input::ByteStream(data)),
        "ivf" => (Codec::AV1, Input::Frames(ivf_frames(&data)?)),
        other => return Err(format!("unknown stream extension {other:?}").into()),
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

/// The decode configuration `input` needs: byte-stream framing for a raw
/// stream, frame-aligned for frames from a container.
pub fn decode_config(codec: Codec, input: &Input) -> DecodeConfig {
    match input {
        Input::ByteStream(_) => DecodeConfig::new(codec).with_byte_stream(),
        Input::Frames(_) => DecodeConfig::new(codec),
    }
}

/// Drive `decoder` over `input`, calling `on_frame` for every frame it emits.
///
/// A raw stream is fed in chunks of [`CHUNK_SIZE`], standing in for network
/// reads; frames from a container are fed one per call. Frames are polled
/// after each call and again after `finish`, so the in-stream and
/// end-of-stream paths share one callback. `NeedsKeyframe` is not
/// special-cased: polling then is a no-op when nothing is ready.
pub fn decode_input<F>(
    decoder: &mut Decoder,
    input: &Input,
    mut on_frame: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut(DecodedFrame) -> Result<(), Box<dyn std::error::Error>>,
{
    let chunks: Box<dyn Iterator<Item = &[u8]>> = match input {
        Input::ByteStream(stream) => Box::new(stream.chunks(CHUNK_SIZE)),
        Input::Frames(frames) => Box::new(frames.iter().map(Vec::as_slice)),
    };
    for (i, chunk) in chunks.enumerate() {
        decoder.decode(chunk, i as u64)?;
        while let FramePoll::Frame(frame) = decoder.try_next_frame()? {
            on_frame(frame)?;
        }
    }
    // End of stream: decodes the trailing picture, emits what reordering held
    // back, and closes the source.
    decoder.finish()?;
    while let Some(frame) = pollster::block_on(decoder.next_frame())? {
        on_frame(frame)?;
    }
    Ok(())
}

/// Write one decoded frame as NV12 to `output`, when an output is set.
pub fn write_nv12(
    frame: &DecodedFrame,
    readback: &mut Option<Readback>,
    output: &mut Option<File>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let (Some(file), Some(readback)) = (output.as_mut(), readback.as_mut()) {
        let data = readback.read(frame)?;
        file.write_all(&data.y)?;
        file.write_all(&data.uv)?;
    }
    Ok(())
}

/// Find a memory type satisfying `flags` for `reqs`, and allocate from it.
pub fn allocate(
    device: &ash::Device,
    props: &vk::PhysicalDeviceMemoryProperties,
    reqs: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags,
) -> Result<vk::DeviceMemory, Box<dyn std::error::Error>> {
    let index = (0..props.memory_type_count)
        .find(|&i| {
            reqs.memory_type_bits & (1 << i) != 0
                && props.memory_types[i as usize]
                    .property_flags
                    .contains(flags)
        })
        .ok_or("no suitable memory type")?;
    let memory = unsafe {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(reqs.size)
                .memory_type_index(index),
            None,
        )
    }?;
    Ok(memory)
}

/// One frame's pixels on the host, cropped to the visible region.
pub struct FrameData {
    pub y: Vec<u8>,
    pub uv: Vec<u8>,
}

/// Copies decoded frames into host memory, reusing one staging buffer.
pub struct Readback {
    /// Keeps the device alive: `VideoContext` is an `Arc` handle, and this
    /// struct's objects must outlive nothing and be destroyed before the device
    /// is. Holding a clone makes that ordering automatic rather than something
    /// the caller has to get right.
    _context: VideoContext,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Staging buffer and the size actually allocated for it.
    buffer: Option<(vk::Buffer, vk::DeviceMemory, usize)>,
}

impl Readback {
    pub fn new(context: &VideoContext) -> Result<Self, Box<dyn std::error::Error>> {
        let device = context.device().clone();
        let family = context.transfer_queue_family();
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }?;
        let command_buffer = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }?[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        let memory_properties = unsafe {
            context
                .instance()
                .get_physical_device_memory_properties(context.physical_device())
        };
        Ok(Self {
            _context: context.clone(),
            device,
            queue: context.transfer_queue(),
            pool,
            command_buffer,
            fence,
            memory_properties,
            buffer: None,
        })
    }

    /// Copy `frame` into host memory and return its visible region.
    pub fn read(&mut self, frame: &DecodedFrame) -> Result<FrameData, Box<dyn std::error::Error>> {
        let bytes = match frame.bit_depth {
            BitDepth::Eight => 1usize,
            BitDepth::Ten => 2,
        };
        // NV12/P010 geometry: full-resolution luma, half-resolution interleaved
        // chroma, both at the image's coded width so the copy is contiguous.
        let y_stride = frame.coded_width as usize * bytes;
        let y_size = y_stride * frame.coded_height as usize;
        let uv_stride = y_stride;
        let uv_size = uv_stride * (frame.coded_height as usize / 2);
        self.ensure_capacity(y_size + uv_size)?;
        let (buffer, memory, _) = self.buffer.expect("just ensured");

        let layers = |aspect| vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: 0,
            base_array_layer: frame.array_layer,
            layer_count: 1,
        };
        let regions = [
            vk::BufferImageCopy2::default()
                .buffer_offset(0)
                .buffer_row_length(frame.coded_width)
                .buffer_image_height(frame.coded_height)
                .image_subresource(layers(vk::ImageAspectFlags::PLANE_0))
                .image_extent(vk::Extent3D {
                    width: frame.coded_width,
                    height: frame.coded_height,
                    depth: 1,
                }),
            vk::BufferImageCopy2::default()
                .buffer_offset(y_size as u64)
                .buffer_row_length(frame.coded_width / 2)
                .buffer_image_height(frame.coded_height / 2)
                .image_subresource(layers(vk::ImageAspectFlags::PLANE_1))
                .image_extent(vk::Extent3D {
                    width: frame.coded_width / 2,
                    height: frame.coded_height / 2,
                    depth: 1,
                }),
        ];

        // GENERAL is already a valid source for a transfer read, so the
        // zero-copy path needs no barrier at all. Anything else is a private
        // image this frame owns, so moving it is safe.
        let needs_transition = frame.layout != vk::ImageLayout::GENERAL;
        let planes = vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1;
        let range = vk::ImageSubresourceRange {
            aspect_mask: planes,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: frame.array_layer,
            layer_count: 1,
        };

        unsafe {
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            if needs_transition {
                let to_src = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(frame.layout)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(frame.image)
                    .subresource_range(range)];
                self.device.cmd_pipeline_barrier2(
                    self.command_buffer,
                    &vk::DependencyInfo::default().image_memory_barriers(&to_src),
                );
            }

            self.device.cmd_copy_image_to_buffer2(
                self.command_buffer,
                &vk::CopyImageToBufferInfo2::default()
                    .src_image(frame.image)
                    .src_image_layout(if needs_transition {
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                    } else {
                        frame.layout
                    })
                    .dst_buffer(buffer)
                    .regions(&regions),
            );

            if needs_transition {
                let restore = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                    .dst_access_mask(vk::AccessFlags2::NONE)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(frame.layout)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(frame.image)
                    .subresource_range(range)];
                self.device.cmd_pipeline_barrier2(
                    self.command_buffer,
                    &vk::DependencyInfo::default().image_memory_barriers(&restore),
                );
            }

            self.device.end_command_buffer(self.command_buffer)?;
            let command_buffers = [self.command_buffer];
            self.device.reset_fences(&[self.fence])?;
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&command_buffers)],
                self.fence,
            )?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        // Crop to the visible region on the way out: the image is coded-size,
        // which is padded up to the codec's macroblock granularity.
        let visible_y_stride = frame.width as usize * bytes;
        let visible_uv_stride = visible_y_stride;
        let mut y = Vec::with_capacity(visible_y_stride * frame.height as usize);
        let mut uv = Vec::with_capacity(visible_uv_stride * frame.height as usize / 2);
        unsafe {
            // Map the whole allocation: a sub-range map has to respect
            // nonCoherentAtomSize, and WHOLE_SIZE sidesteps that.
            let ptr =
                self.device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
                    as *const u8;
            for row in 0..frame.height as usize {
                y.extend_from_slice(std::slice::from_raw_parts(
                    ptr.add(row * y_stride),
                    visible_y_stride,
                ));
            }
            for row in 0..frame.height as usize / 2 {
                uv.extend_from_slice(std::slice::from_raw_parts(
                    ptr.add(y_size + row * uv_stride),
                    visible_uv_stride,
                ));
            }
            self.device.unmap_memory(memory);
        }
        Ok(FrameData { y, uv })
    }

    fn ensure_capacity(&mut self, size: usize) -> Result<(), Box<dyn std::error::Error>> {
        if let Some((_, _, capacity)) = self.buffer
            && capacity >= size
        {
            return Ok(());
        }
        unsafe {
            if let Some((buffer, memory, _)) = self.buffer.take() {
                self.device.device_wait_idle()?;
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
            let buffer = self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size as u64)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let reqs = self.device.get_buffer_memory_requirements(buffer);
            let wanted =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let index = (0..self.memory_properties.memory_type_count)
                .find(|&i| {
                    reqs.memory_type_bits & (1 << i) != 0
                        && self.memory_properties.memory_types[i as usize]
                            .property_flags
                            .contains(wanted)
                })
                .ok_or("no host-visible memory for readback")?;
            let memory = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(index),
                None,
            )?;
            self.device.bind_buffer_memory(buffer, memory, 0)?;
            // Record what was allocated, not what was asked for.
            self.buffer = Some((buffer, memory, reqs.size as usize));
        }
        Ok(())
    }
}

impl Drop for Readback {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some((buffer, memory, _)) = self.buffer.take() {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.pool, None);
        }
    }
}
