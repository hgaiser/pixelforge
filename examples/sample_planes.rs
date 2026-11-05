//! Read decoded frames through per-plane views, with no ycbcr conversion.
//!
//! The companion to `sample_frame`, and the reason
//! [`DecodedFrame::plane_views`] exists. `sample_frame` samples the decoded
//! picture as one multi-planar image, which needs a `VkSamplerYcbcrConversion`
//! bound as a combined image sampler with an immutable sampler. Some shader
//! toolchains cannot express that at all: naga has no combined-image-sampler
//! type, so wgpu has no ycbcr sampler, and a renderer built on either had no
//! way to read a decoded frame without copying it first.
//!
//! This reads the same picture as two ordinary single-plane textures instead,
//! `R8_UNORM` luma and `R8G8_UNORM` chroma, bound as plain `texture2D`
//! descriptors. No conversion object, no immutable sampler, no sampler at all.
//! It works because pixelforge asks for `MUTABLE_FORMAT` on picture images
//! wherever the driver allows it.
//!
//! A real renderer would apply the YUV to RGB matrix in the shader. This writes
//! the samples straight through, which makes the output plain NV12 and lets it
//! be compared byte-for-byte against the decoder:
//!
//!   cargo run --example sample_planes -- tests/data/bframes.264 out.yuv
//!   ffmpeg -i tests/data/bframes.264 -pix_fmt nv12 ref.yuv
//!   cmp ref.yuv out.yuv
//!
//! Unlike `sample_frame`, this comparison is exact. There is no filtering, no
//! chroma reconstruction and no colour matrix in the way.

#[allow(dead_code)]
mod common;
use common::decode_input;

use ash::vk::{self, TaggedStructure as _};
use pixelforge::decoder::{DecodedFrame, Decoder};
use pixelforge::vulkan::{VideoContext, VideoContextBuilder};
use std::fs::File;
use std::io::Write;

const SHADER: &[u8] = include_bytes!("shader/sample_planes.spv");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args().skip(1);
    let input_path = args.next().unwrap_or_else(|| {
        eprintln!("usage: sample_planes <input.264|.265|.obu|.ivf> [output.yuv]");
        std::process::exit(1);
    });
    let output_path = args.next();
    let (codec, input) = common::open_input(&input_path)?;

    let validation = std::env::var("PIXELFORGE_VALIDATION").is_ok();
    let context = VideoContextBuilder::new()
        .app_name("pixelforge-planes")
        .require_decode(codec)
        .enable_validation(validation)
        .build()?;
    println!(
        "Device: {}",
        unsafe { std::ffi::CStr::from_ptr(context.device_properties().device_name.as_ptr()) }
            .to_string_lossy()
    );

    let consumer_family = context.compute_queue_family();
    let config = common::decode_config(codec, &input)
        .with_output_depth(8)
        .with_consumer_queue_family(consumer_family);

    let mut output = output_path.as_ref().map(File::create).transpose()?;
    let mut planes: Option<PlaneReader> = None;
    let mut decoder = Decoder::new(context.clone(), config)?;
    let mut count = 0usize;
    let start = std::time::Instant::now();

    decode_input(&mut decoder, &input, |frame| {
        if !frame.plane_views {
            return Err(
                "this device does not allow per-plane views of decoded pictures; \
                        use the sample_frame example instead"
                    .into(),
            );
        }
        let reader = match &mut planes {
            Some(p) => p,
            none => none.insert(PlaneReader::new(&context, consumer_family, &frame)?),
        };
        let (y, uv) = reader.read(&frame)?;
        if let Some(file) = output.as_mut() {
            file.write_all(&y)?;
            file.write_all(&uv)?;
        }
        count += 1;
        Ok(())
    })?;

    let elapsed = start.elapsed();
    println!(
        "Read {} frames through plane views in {:.1?} ({:.1} fps)",
        count,
        elapsed,
        count as f64 / elapsed.as_secs_f64()
    );
    if let Some(path) = output_path {
        println!("Wrote NV12 frames to {}", path);
    }

    drop(planes);
    drop(decoder);
    Ok(())
}

/// Reads a decoded picture's two planes as ordinary textures.
struct PlaneReader {
    _context: VideoContext,
    device: ash::Device,
    queue: vk::Queue,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shader: vk::ShaderModule,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    /// Destination planes, written by the shader and copied to the host.
    out: [Plane; 2],
    staging: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    /// One pair of plane views per decoded image, since the decoder rotates
    /// through a handful of them.
    views: std::collections::HashMap<(u64, u64, u32), [vk::ImageView; 2]>,
    width: u32,
    height: u32,
}

struct Plane {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

impl PlaneReader {
    fn new(
        context: &VideoContext,
        family: u32,
        frame: &DecodedFrame,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = context.device().clone();
        let memory_properties = unsafe {
            context
                .instance()
                .get_physical_device_memory_properties(context.physical_device())
        };

        // Two sampled images and two storage images. Note what is *not* here:
        // no COMBINED_IMAGE_SAMPLER, no immutable sampler, no ycbcr conversion.
        let bindings = [
            binding(0, vk::DescriptorType::SAMPLED_IMAGE),
            binding(1, vk::DescriptorType::SAMPLED_IMAGE),
            binding(2, vk::DescriptorType::STORAGE_IMAGE),
            binding(3, vk::DescriptorType::STORAGE_IMAGE),
        ];
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }?;
        let set_layouts = [set_layout];
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(8)];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_ranges),
                None,
            )
        }?;
        let code: Vec<u32> = SHADER
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let shader = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
        }?;
        let entry = c"main";
        let pipeline = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(shader)
                            .name(entry),
                    )
                    .layout(pipeline_layout)],
                None,
            )
        }
        .map_err(|(_, e)| e)?[0];

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(2),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(2),
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }?;
        let descriptor_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&set_layouts),
            )
        }?[0];

        let command_pool = unsafe {
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
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }?[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;

        let (w, h) = (frame.width, frame.height);
        let out = [
            make_plane(&device, &memory_properties, w, h, vk::Format::R8_UNORM)?,
            make_plane(
                &device,
                &memory_properties,
                w / 2,
                h / 2,
                vk::Format::R8G8_UNORM,
            )?,
        ];

        let (staging, staging_memory) = make_staging(&device, &memory_properties, w, h)?;

        Ok(Self {
            _context: context.clone(),
            device,
            queue: context.compute_queue(),
            memory_properties,
            set_layout,
            pipeline_layout,
            pipeline,
            shader,
            descriptor_pool,
            descriptor_set,
            command_pool,
            command_buffer,
            fence,
            out,
            staging,
            staging_memory,
            views: std::collections::HashMap::new(),
            width: w,
            height: h,
        })
    }

    /// Run the shader over one frame and return its luma and chroma planes.
    fn read(
        &mut self,
        frame: &DecodedFrame,
    ) -> Result<(Vec<u8>, Vec<u8>), Box<dyn std::error::Error>> {
        self.resize_for(frame)?;
        let [luma, chroma] = self.frame_views(frame)?;
        let device = self.device.clone();

        let src = [
            vk::DescriptorImageInfo::default()
                .image_view(luma)
                .image_layout(vk::ImageLayout::GENERAL),
            vk::DescriptorImageInfo::default()
                .image_view(chroma)
                .image_layout(vk::ImageLayout::GENERAL),
        ];
        let dst = [
            vk::DescriptorImageInfo::default()
                .image_view(self.out[0].view)
                .image_layout(vk::ImageLayout::GENERAL),
            vk::DescriptorImageInfo::default()
                .image_view(self.out[1].view)
                .image_layout(vk::ImageLayout::GENERAL),
        ];
        let writes = [
            write_set(self.descriptor_set, 0, vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&src[0..1]),
            write_set(self.descriptor_set, 1, vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&src[1..2]),
            write_set(self.descriptor_set, 2, vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&dst[0..1]),
            write_set(self.descriptor_set, 3, vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&dst[1..2]),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let push = [frame.width, frame.height];
        let y_size = self.width as u64 * self.height as u64;

        unsafe {
            device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            // The decoded picture stays in whatever layout it arrived in when
            // that is GENERAL, which is what the zero-copy path hands over and
            // is already valid to read. Only a frame from the copying path
            // needs moving, and that image is private to the frame.
            let mut barriers = vec![
                to_general(
                    self.out[0].image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageAspectFlags::COLOR,
                    0,
                ),
                to_general(
                    self.out[1].image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageAspectFlags::COLOR,
                    0,
                ),
            ];
            if frame.layout != vk::ImageLayout::GENERAL {
                barriers.push(to_general(
                    frame.image,
                    frame.layout,
                    vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1,
                    frame.array_layer,
                ));
            }
            device.cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );

            device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                self.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(push.as_ptr() as *const u8, 8),
            );
            device.cmd_dispatch(
                self.command_buffer,
                frame.width.div_ceil(8),
                frame.height.div_ceil(8),
                1,
            );

            let after = [
                storage_to_copy(self.out[0].image),
                storage_to_copy(self.out[1].image),
            ];
            device.cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&after),
            );

            // Both planes land in one buffer, laid out as NV12.
            for (i, plane) in self.out.iter().enumerate() {
                let (w, h, offset) = if i == 0 {
                    (self.width, self.height, 0)
                } else {
                    (self.width / 2, self.height / 2, y_size)
                };
                let region = [vk::BufferImageCopy2::default()
                    .buffer_offset(offset)
                    .buffer_row_length(w)
                    .buffer_image_height(h)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: w,
                        height: h,
                        depth: 1,
                    })];
                device.cmd_copy_image_to_buffer2(
                    self.command_buffer,
                    &vk::CopyImageToBufferInfo2::default()
                        .src_image(plane.image)
                        .src_image_layout(vk::ImageLayout::GENERAL)
                        .dst_buffer(self.staging)
                        .regions(&region),
                );
            }

            device.end_command_buffer(self.command_buffer)?;
            let buffers = [self.command_buffer];
            device.reset_fences(&[self.fence])?;
            device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(&buffers)],
                self.fence,
            )?;
            device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }

        let y_len = y_size as usize;
        let uv_len = y_len / 2;
        let mut y = vec![0u8; y_len];
        let mut uv = vec![0u8; uv_len];
        unsafe {
            let ptr = device.map_memory(
                self.staging_memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )? as *const u8;
            std::ptr::copy_nonoverlapping(ptr, y.as_mut_ptr(), y_len);
            std::ptr::copy_nonoverlapping(ptr.add(y_len), uv.as_mut_ptr(), uv_len);
            device.unmap_memory(self.staging_memory);
        }
        Ok((y, uv))
    }

    /// Rebuild this reader's own images when the stream's size changes.
    ///
    /// A decoder can change resolution mid-stream, so anything sized from a
    /// frame has to be prepared to be resized. Cached views go too: they belong
    /// to images the decoder has moved on from.
    fn resize_for(&mut self, frame: &DecodedFrame) -> Result<(), Box<dyn std::error::Error>> {
        if self.width == frame.width && self.height == frame.height {
            return Ok(());
        }
        unsafe {
            self.device.device_wait_idle()?;
            for view in self.views.values() {
                for v in view {
                    self.device.destroy_image_view(*v, None);
                }
            }
            self.views.clear();
            for plane in &self.out {
                self.device.destroy_image_view(plane.view, None);
                self.device.destroy_image(plane.image, None);
                self.device.free_memory(plane.memory, None);
            }
            self.device.destroy_buffer(self.staging, None);
            self.device.free_memory(self.staging_memory, None);
        }
        self.width = frame.width;
        self.height = frame.height;
        let (w, h) = (self.width, self.height);
        self.out = [
            make_plane(
                &self.device,
                &self.memory_properties,
                w,
                h,
                vk::Format::R8_UNORM,
            )?,
            make_plane(
                &self.device,
                &self.memory_properties,
                w / 2,
                h / 2,
                vk::Format::R8G8_UNORM,
            )?,
        ];
        let (staging, staging_memory) = make_staging(&self.device, &self.memory_properties, w, h)?;
        self.staging = staging;
        self.staging_memory = staging_memory;
        Ok(())
    }

    /// Views of a decoded picture's two planes, created once per image.
    ///
    /// This is the whole trick: an ordinary view, with a plane aspect and that
    /// plane's compatible single-plane format. Legal because the image was
    /// created with `MUTABLE_FORMAT`, which is what
    /// [`DecodedFrame::plane_views`] reports.
    fn frame_views(
        &mut self,
        frame: &DecodedFrame,
    ) -> Result<[vk::ImageView; 2], Box<dyn std::error::Error>> {
        // Keyed on the generation as well as the handle: a rebuilt session
        // destroys its images and drivers reuse handles, so a handle alone can
        // name two different images over a decode's lifetime.
        let key = (
            frame.generation,
            ash::vk::Handle::as_raw(frame.image),
            frame.array_layer,
        );
        if let Some(views) = self.views.get(&key) {
            return Ok(*views);
        }
        let mut made = [vk::ImageView::null(); 2];
        for (i, (aspect, format)) in [
            (vk::ImageAspectFlags::PLANE_0, vk::Format::R8_UNORM),
            (vk::ImageAspectFlags::PLANE_1, vk::Format::R8G8_UNORM),
        ]
        .into_iter()
        .enumerate()
        {
            // A view inherits the image's usage unless told otherwise, and a
            // decoded picture's usage includes VIDEO_DECODE_DST_KHR, which
            // R8_UNORM cannot satisfy: it has no VIDEO_DECODE_OUTPUT format
            // feature. Narrowing the view to what this actually does with it
            // is both correct and honest. Without this the view is invalid,
            // and only on drivers where the frame really is the DPB image, so
            // it passes on hardware that takes the copying path.
            let mut view_usage =
                vk::ImageViewUsageCreateInfo::default().usage(vk::ImageUsageFlags::SAMPLED);
            made[i] = unsafe {
                self.device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .push(&mut view_usage)
                        .image(frame.image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(format)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: aspect,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: frame.array_layer,
                            layer_count: 1,
                        }),
                    None,
                )
            }?;
        }
        self.views.insert(key, made);
        Ok(made)
    }
}

fn binding(index: u32, ty: vk::DescriptorType) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(index)
        .descriptor_type(ty)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

fn write_set(
    set: vk::DescriptorSet,
    binding: u32,
    ty: vk::DescriptorType,
) -> vk::WriteDescriptorSet<'static> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(ty)
}

fn to_general(
    image: vk::Image,
    old: vk::ImageLayout,
    aspect: vk::ImageAspectFlags,
    layer: u32,
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::NONE)
        .src_access_mask(vk::AccessFlags2::NONE)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_access_mask(
            vk::AccessFlags2::SHADER_SAMPLED_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        )
        .old_layout(old)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: layer,
            layer_count: 1,
        })
}

fn storage_to_copy(image: vk::Image) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::COPY)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
}

/// A host-visible buffer big enough for one NV12 frame of `width` x `height`.
fn make_staging(
    device: &ash::Device,
    props: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn std::error::Error>> {
    let size = (width as u64 * height as u64) * 3 / 2;
    let buffer = unsafe {
        device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }?;
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory = common::allocate(
        device,
        props,
        reqs,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;
    Ok((buffer, memory))
}

fn make_plane(
    device: &ash::Device,
    props: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    format: vk::Format,
) -> Result<Plane, Box<dyn std::error::Error>> {
    let image = unsafe {
        device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            None,
        )
    }?;
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let memory = common::allocate(device, props, reqs, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    unsafe { device.bind_image_memory(image, memory, 0) }?;
    let view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            None,
        )
    }?;
    Ok(Plane {
        image,
        memory,
        view,
    })
}

impl Drop for PlaneReader {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for views in self.views.values() {
                for view in views {
                    self.device.destroy_image_view(*view, None);
                }
            }
            for plane in &self.out {
                self.device.destroy_image_view(plane.view, None);
                self.device.destroy_image(plane.image, None);
                self.device.free_memory(plane.memory, None);
            }
            self.device.destroy_buffer(self.staging, None);
            self.device.free_memory(self.staging_memory, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_shader_module(self.shader, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
        let _ = self.memory_properties;
    }
}
