//! Encode frames that are drawn on the GPU.
//!
//! The application draws RGB frames into a Vulkan image of its own and hands
//! the image to the encoder as a [`SourceImage`]. The encoder converts each
//! frame to YUV on the GPU and encodes it. No frame goes through CPU memory,
//! and the CPU does not wait between drawing and encoding: a timeline
//! semaphore orders the drawing before the encode, and the encoder's
//! [`SourceImage::released`] points order the next drawing after it.
//!
//! The drawing here is a clear to a colour that changes and a square that
//! moves. A renderer records its own commands in the same place.
//!
//! Usage:
//!   cargo run --example encode_gpu -- [h264|h265|av1]

use ash::vk::{self, TaggedStructure as _};
use pixelforge::{
    Codec, ColorSpec, EncodeConfig, EncodedPacket, Encoder, Frame, Handoff, RateControl, RgbFormat,
    RgbSource, Source, SourceImage, TimelinePoint, VideoContext, VideoContextBuilder,
};
use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const FRAMES: u32 = 90;
const SQUARE: u32 = 64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (codec, path) = match std::env::args().nth(1).as_deref() {
        None | Some("h264") => (Codec::H264, "output.h264"),
        Some("h265") => (Codec::H265, "output.h265"),
        Some("av1") => (Codec::AV1, "output.obu"),
        Some(other) => return Err(format!("unknown codec {other:?}").into()),
    };

    let context = VideoContextBuilder::new().require_encode(codec).build()?;
    // The frames are BGRA in sRGB. The encoder converts them to the stream's
    // YUV (BT.709 by default).
    let config = EncodeConfig::new(codec, WIDTH, HEIGHT)
        .with_frame_rate(30, 1)
        .with_rate_control(RateControl::Cbr { bitrate: 2_000_000 })
        .with_source(Source::Rgb(RgbSource::new(
            RgbFormat::BGRA,
            ColorSpec::Srgb,
        )));
    let mut encoder = Encoder::new(context.clone(), config)?;
    println!("Converting with {:?}", encoder.conversion());

    let mut canvas = Canvas::new(&context)?;
    // The canvas leaves the image in GENERAL, and the encoder gives it back
    // in GENERAL.
    let source = SourceImage::new(
        &context,
        canvas.image,
        Handoff::Device {
            layout: vk::ImageLayout::GENERAL,
        },
    )?;

    let mut output = File::create(path)?;
    let mut write = |packet: EncodedPacket| output.write_all(&packet.data);
    let mut pending = VecDeque::new();
    for index in 0..FRAMES {
        // Draw once the encoder no longer reads the image, and encode once
        // the drawing is done. Both waits are on the GPU.
        let drawn = canvas.draw(index, &source.released())?;
        pending.push_back(encoder.submit(Frame::image(&source).wait(&[drawn]))?);
        while pending.len() > 1 {
            write(pollster::block_on(pending.pop_front().unwrap())?)?;
        }
    }
    encoder.flush()?;
    for future in pending {
        write(pollster::block_on(future)?)?;
    }
    println!("Wrote {FRAMES} frames to {path}");

    // The source waits for the frames that read the image, so it goes before
    // the image.
    drop(source);
    drop(canvas);
    Ok(())
}

/// The application's side: a BGRA image, and the commands that draw into it.
struct Canvas {
    context: VideoContext,
    image: vk::Image,
    memory: vk::DeviceMemory,
    /// A white square, copied into the image at a new place each frame.
    square: vk::Buffer,
    square_memory: vk::DeviceMemory,
    pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    /// Signalled when the last drawing is done, so the command buffer can be
    /// recorded again.
    fence: vk::Fence,
    /// Reaches `drawn` when frame `drawn` is drawn.
    timeline: vk::Semaphore,
    drawn: u64,
}

impl Canvas {
    fn new(context: &VideoContext) -> Result<Self, Box<dyn std::error::Error>> {
        let device = context.device();
        // The encoder reads the image with its colour converter (on the
        // compute queue) or with a copy (on the transfer queue), so both
        // families share it. The drawing runs on the compute queue.
        let mut families = vec![
            context.compute_queue_family(),
            context.transfer_queue_family(),
        ];
        families.dedup();
        let sharing = if families.len() > 1 {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };
        let image = unsafe {
            device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(RgbFormat::BGRA.vk_format())
                    .extent(vk::Extent3D {
                        width: WIDTH,
                        height: HEIGHT,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    // Written by the drawing; sampled by the converter, or
                    // copied by the encoder.
                    .usage(
                        vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::SAMPLED
                            | vk::ImageUsageFlags::TRANSFER_SRC,
                    )
                    .sharing_mode(sharing)
                    .queue_family_indices(&families),
                None,
            )
        }?;
        let memory = allocate(
            context,
            unsafe { device.get_image_memory_requirements(image) },
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        unsafe { device.bind_image_memory(image, memory, 0) }?;

        let size = (SQUARE * SQUARE * 4) as u64;
        let square = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                None,
            )
        }?;
        let square_memory = allocate(
            context,
            unsafe { device.get_buffer_memory_requirements(square) },
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            device.bind_buffer_memory(square, square_memory, 0)?;
            let data = device.map_memory(square_memory, 0, size, vk::MemoryMapFlags::empty())?;
            std::ptr::write_bytes(data as *mut u8, 0xFF, size as usize);
            device.unmap_memory(square_memory);
        }

        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(context.compute_queue_family())
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }?;
        let command_buffer = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .command_buffer_count(1),
            )
        }?[0];
        let fence = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        }?;
        let mut timeline_info =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let timeline = unsafe {
            device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push(&mut timeline_info),
                None,
            )
        }?;
        Ok(Self {
            context: context.clone(),
            image,
            memory,
            square,
            square_memory,
            pool,
            command_buffer,
            fence,
            timeline,
            drawn: 0,
        })
    }

    /// Draw frame `index` once the points in `released` are reached, and
    /// return the point where the drawing is done.
    fn draw(
        &mut self,
        index: u32,
        released: &[TimelinePoint],
    ) -> Result<TimelinePoint, Box<dyn std::error::Error>> {
        let device = self.context.device();
        let cb = self.command_buffer;
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        // The first frame starts from nothing. After that the image is in
        // GENERAL, where the encoder gave it back.
        let old_layout = if self.drawn == 0 {
            vk::ImageLayout::UNDEFINED
        } else {
            vk::ImageLayout::GENERAL
        };
        let to_general = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(vk::ImageLayout::GENERAL)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image)
            .subresource_range(range);
        // The clear and the copy both write the image.
        let between = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image)
            .subresource_range(range);
        let t = index as f32 / FRAMES as f32;
        let background = vk::ClearColorValue {
            float32: [0.2 + 0.6 * t, 0.3, 0.8 - 0.6 * t, 1.0],
        };
        let x = (index * 6) % (WIDTH - SQUARE);
        let square_region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D {
                x: x as i32,
                y: ((HEIGHT - SQUARE) / 2) as i32,
                z: 0,
            })
            .image_extent(vk::Extent3D {
                width: SQUARE,
                height: SQUARE,
                depth: 1,
            });

        unsafe {
            // The command buffer is free once the last drawing is done.
            device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            device.reset_fences(&[self.fence])?;
            device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_general],
            );
            device.cmd_clear_color_image(
                cb,
                self.image,
                vk::ImageLayout::GENERAL,
                &background,
                &[range],
            );
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[between],
            );
            device.cmd_copy_buffer_to_image(
                cb,
                self.square,
                self.image,
                vk::ImageLayout::GENERAL,
                &[square_region],
            );
            device.end_command_buffer(cb)?;

            // Wait for the encoder's reads, and signal the next value of the
            // timeline when done.
            self.drawn += 1;
            let wait_semaphores: Vec<vk::Semaphore> =
                released.iter().map(|p| p.semaphore).collect();
            let wait_values: Vec<u64> = released.iter().map(|p| p.value).collect();
            let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; released.len()];
            let signal_values = [self.drawn];
            let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_values)
                .signal_semaphore_values(&signal_values);
            let submit = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(std::slice::from_ref(&cb))
                .signal_semaphores(std::slice::from_ref(&self.timeline))
                .push(&mut timeline_info);
            device.queue_submit(self.context.compute_queue(), &[submit], self.fence)?;
        }
        Ok(TimelinePoint::new(self.timeline, self.drawn))
    }
}

impl Drop for Canvas {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe {
            let _ = device.wait_for_fences(&[self.fence], true, u64::MAX);
            device.destroy_semaphore(self.timeline, None);
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.pool, None);
            device.destroy_buffer(self.square, None);
            device.free_memory(self.square_memory, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Allocate memory for `requirements` with `flags`.
fn allocate(
    context: &VideoContext,
    requirements: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags,
) -> Result<vk::DeviceMemory, Box<dyn std::error::Error>> {
    let memory_type = context
        .find_memory_type(requirements.memory_type_bits, flags)
        .ok_or("no suitable memory type")?;
    let memory = unsafe {
        context.device().allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
            None,
        )
    }?;
    Ok(memory)
}
