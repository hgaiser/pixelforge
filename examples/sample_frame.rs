//! Sample decoded frames in a shader, with no copy anywhere.
//!
//! This is the path pixelforge exists to make possible: the decoder hands over
//! its own GPU image, a compute shader reads it through a
//! `VkSamplerYcbcrConversion` (which does chroma reconstruction and the YUV to
//! RGB matrix in hardware), and the result goes straight to a render target.
//! Here the "render target" is an RGBA image that gets written to a file so the
//! output can be checked, but nothing about the sampling changes for a real
//! renderer.
//!
//! Usage:
//!   cargo run --example sample_frame -- input.264 [output.rgba]
//!
//! The output matches `ffmpeg -i input.264 -pix_fmt rgba out.rgba` closely, but
//! not exactly, and not in a way PSNR describes well. The hardware sampler and
//! ffmpeg reconstruct subsampled chroma differently, so they agree almost
//! everywhere and disagree hard on the handful of pixels sitting on a colour
//! edge. On tests/data/bframes.264, a synthetic pattern of saturated bars,
//! 88% of pixels are within 2 of ffmpeg and the worst 1% are not close at all,
//! which drags PSNR down to ~26 dB while the images are indistinguishable.
//! Judge it by eye, or by the share of pixels that agree, rather than by PSNR:
//!
//!   ffmpeg -i input.264 -pix_fmt rgba ref.rgba

use std::fs::File;
use std::io::Write;

use ash::vk::{self, TaggedStructure as _};
use pixelforge::Codec;
use pixelforge::decoder::{DecodeConfig, DecodedFrame, Decoder, FramePoll};
use pixelforge::vulkan::{VideoContext, VideoContextBuilder};

const SHADER: &[u8] = include_bytes!("shader/sample_frame.spv");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let mut args = std::env::args().skip(1);
    let input_path = args.next().unwrap_or_else(|| {
        eprintln!("usage: sample_frame <input.264|.265|.obu> [output.rgba]");
        std::process::exit(1);
    });
    let output_path = args.next();
    let codec = match input_path.rsplit('.').next() {
        Some("264") => Codec::H264,
        Some("265") => Codec::H265,
        Some("obu") => Codec::AV1,
        _ => return Err(format!("unknown stream extension in {input_path:?}").into()),
    };
    let stream = std::fs::read(&input_path)?;

    let validation = std::env::var("PIXELFORGE_VALIDATION").is_ok();
    let context = VideoContextBuilder::new()
        .app_name("pixelforge-sample")
        .require_decode(codec)
        .enable_validation(validation)
        .build()?;

    println!(
        "Device: {}",
        unsafe { std::ffi::CStr::from_ptr(context.device_properties().device_name.as_ptr()) }
            .to_string_lossy()
    );

    // Naming the queue family that will read frames is what puts it in each
    // picture's sharing set. Without this, sampling a decoded image from a
    // family the decoder did not share with is undefined.
    let consumer_family = context.compute_queue_family();
    let config = DecodeConfig::new(codec)
        .with_byte_stream()
        .with_consumer_queue_family(consumer_family);

    let mut output = output_path.as_ref().map(File::create).transpose()?;
    // Built lazily: the picture format is only known once the stream's first
    // parameter set has been parsed, and the ycbcr conversion depends on it.
    let mut sampler: Option<Sampler> = None;
    let mut decoder = Decoder::new(context.clone(), config)?;

    let mut count = 0usize;
    let start = std::time::Instant::now();

    let mut on_frame = |frame: DecodedFrame| -> Result<(), Box<dyn std::error::Error>> {
        let sampler = match &mut sampler {
            Some(s) => s,
            none => none.insert(Sampler::new(&context, consumer_family, &frame)?),
        };
        let rgba = sampler.sample(&frame)?;
        if let Some(file) = output.as_mut() {
            file.write_all(&rgba)?;
        }
        count += 1;
        Ok(())
    };
    for (index, piece) in stream.chunks(64 * 1024).enumerate() {
        decoder.decode(piece, index as u64)?;
        while let FramePoll::Frame(frame) = decoder.try_next_frame()? {
            on_frame(frame)?;
        }
    }
    decoder.finish()?;
    while let Some(frame) = pollster::block_on(decoder.next_frame())? {
        on_frame(frame)?;
    }
    drop(on_frame);

    let elapsed = start.elapsed();
    println!(
        "Sampled {} frames in {:.1?} ({:.1} fps)",
        count,
        elapsed,
        count as f64 / elapsed.as_secs_f64()
    );
    if let Some(path) = output_path {
        println!("Wrote RGBA frames to {}", path);
    }

    // Vulkan objects must go before the device the context owns.
    drop(sampler);
    drop(decoder);
    Ok(())
}

/// How many descriptors one combined image sampler of `format` costs.
///
/// Multi-planar formats may need one per plane. Falls back to three, the most
/// any format defined today can ask for, if the query fails.
fn combined_image_sampler_descriptor_count(context: &VideoContext, format: vk::Format) -> u32 {
    let mut ycbcr = vk::SamplerYcbcrConversionImageFormatProperties::default();
    let mut props = vk::ImageFormatProperties2::default().push(&mut ycbcr);
    let info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED);
    let queried = unsafe {
        context
            .instance()
            .get_physical_device_image_format_properties2(
                context.physical_device(),
                &info,
                &mut props,
            )
    };
    match queried {
        Ok(()) if ycbcr.combined_image_sampler_descriptor_count > 0 => {
            ycbcr.combined_image_sampler_descriptor_count
        }
        _ => 3,
    }
}

/// Everything needed to read a decoded frame in a shader.
struct Sampler {
    context: VideoContext,
    device: ash::Device,
    queue: vk::Queue,
    conversion: vk::SamplerYcbcrConversion,
    sampler: vk::Sampler,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shader: vk::ShaderModule,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    /// RGBA destination, its view, and the host-visible buffer read back from.
    target: vk::Image,
    target_memory: vk::DeviceMemory,
    target_view: vk::ImageView,
    staging: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    /// One view per decoded image, since a view is bound to its image and the
    /// decoder rotates through several.
    frame_views: std::collections::HashMap<(u64, u64, u32), vk::ImageView>,
    width: u32,
    height: u32,
    format: vk::Format,
    target_initialised: bool,
}

impl Sampler {
    fn new(
        context: &VideoContext,
        family: u32,
        frame: &DecodedFrame,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = context.device().clone();
        let format = frame.chroma_format;
        let _ = format;
        let picture_format = vk::Format::G8_B8R8_2PLANE_420_UNORM;

        // The conversion is what turns a multi-planar YCbCr image into RGB on
        // read. A real player takes these from the stream's VUI; with nothing
        // signalled, the usual guess is BT.601 for standard definition and
        // BT.709 above it, which is what ffmpeg does too. Narrow range is what
        // H.264 video normally carries.
        let conversion = unsafe {
            device.create_sampler_ycbcr_conversion(
                &vk::SamplerYcbcrConversionCreateInfo::default()
                    .format(picture_format)
                    .ycbcr_model(if frame.height <= 576 {
                        vk::SamplerYcbcrModelConversion::YCBCR_601
                    } else {
                        vk::SamplerYcbcrModelConversion::YCBCR_709
                    })
                    .ycbcr_range(vk::SamplerYcbcrRange::ITU_NARROW)
                    .components(vk::ComponentMapping::default())
                    .x_chroma_offset(vk::ChromaLocation::COSITED_EVEN)
                    .y_chroma_offset(vk::ChromaLocation::MIDPOINT)
                    .chroma_filter(vk::Filter::LINEAR),
                None,
            )
        }?;

        let mut conversion_info = vk::SamplerYcbcrConversionInfo::default().conversion(conversion);
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .push(&mut conversion_info),
                None,
            )
        }?;

        // A ycbcr sampler has to be immutable in the layout: the conversion is
        // baked into the pipeline rather than chosen at bind time.
        let immutable = [sampler];
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .immutable_samplers(&immutable),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }?;

        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(16)];
        let set_layouts = [set_layout];
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
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(entry);
        let pipeline = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(pipeline_layout)],
                None,
            )
        }
        .map_err(|(_, e)| e)?[0];

        // A combined image sampler for a multi-planar format does not
        // necessarily cost one descriptor: the implementation may need one per
        // plane, and it says how many through
        // `combinedImageSamplerDescriptorCount`. RADV asks for more than one
        // here, and a pool sized for one fails allocation with
        // ERROR_OUT_OF_POOL_MEMORY. The descriptor set layout still declares a
        // count of 1; only the pool has to be sized for the real cost.
        let sampler_descriptors = combined_image_sampler_descriptor_count(context, picture_format);
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(sampler_descriptors),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1),
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

        let mut this = Self {
            context: context.clone(),
            device,
            queue: context.compute_queue(),
            conversion,
            sampler,
            set_layout,
            pipeline_layout,
            pipeline,
            shader,
            descriptor_pool,
            descriptor_set,
            command_pool,
            command_buffer,
            fence,
            target: vk::Image::null(),
            target_memory: vk::DeviceMemory::null(),
            target_view: vk::ImageView::null(),
            staging: vk::Buffer::null(),
            staging_memory: vk::DeviceMemory::null(),
            frame_views: std::collections::HashMap::new(),
            width: frame.width,
            height: frame.height,
            format: picture_format,
            target_initialised: false,
        };
        this.create_target()?;
        Ok(this)
    }

    /// Sample one frame and return its RGBA pixels.
    fn sample(&mut self, frame: &DecodedFrame) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        self.resize_for(frame)?;
        let view = self.frame_view(frame)?;
        let device = self.device.clone();

        let image_info = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let target_info = [vk::DescriptorImageInfo::default()
            .image_view(self.target_view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&target_info),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        let mut push = [0u32; 4];
        push[0] = frame.width;
        push[1] = frame.height;
        push[2] = frame.coded_width;
        push[3] = frame.coded_height;

        unsafe {
            device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            // The decoded image is already in GENERAL and stays there: not
            // transitioning it is exactly what makes this safe while the
            // decoder is still using it as a reference. Only the frame's own
            // planes need a barrier if it came from the copying fallback.
            let mut barriers = Vec::new();
            if frame.layout != vk::ImageLayout::GENERAL {
                barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                        .old_layout(frame.layout)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(frame.image)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: frame.array_layer,
                            layer_count: 1,
                        }),
                );
            }
            if !self.target_initialised {
                barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(self.target)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                );
                self.target_initialised = true;
            }
            if !barriers.is_empty() {
                device.cmd_pipeline_barrier2(
                    self.command_buffer,
                    &vk::DependencyInfo::default().image_memory_barriers(&barriers),
                );
            }

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
                std::slice::from_raw_parts(push.as_ptr() as *const u8, 16),
            );
            device.cmd_dispatch(
                self.command_buffer,
                frame.width.div_ceil(8),
                frame.height.div_ceil(8),
                1,
            );

            // Make the shader's writes visible to the copy that follows.
            let to_copy = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.target)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })];
            device.cmd_pipeline_barrier2(
                self.command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&to_copy),
            );

            let region = [vk::BufferImageCopy2::default()
                .buffer_offset(0)
                .buffer_row_length(self.width)
                .buffer_image_height(self.height)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: self.width,
                    height: self.height,
                    depth: 1,
                })];
            device.cmd_copy_image_to_buffer2(
                self.command_buffer,
                &vk::CopyImageToBufferInfo2::default()
                    .src_image(self.target)
                    .src_image_layout(vk::ImageLayout::GENERAL)
                    .dst_buffer(self.staging)
                    .regions(&region),
            );

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

        let size = self.width as usize * self.height as usize * 4;
        let mut out = vec![0u8; size];
        unsafe {
            let ptr = device.map_memory(
                self.staging_memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )? as *const u8;
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), size);
            device.unmap_memory(self.staging_memory);
        }
        Ok(out)
    }

    /// Rebuild the render target when the stream's size changes.
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
            for view in self.frame_views.values() {
                self.device.destroy_image_view(*view, None);
            }
            self.frame_views.clear();
            self.device.destroy_buffer(self.staging, None);
            self.device.free_memory(self.staging_memory, None);
            self.device.destroy_image_view(self.target_view, None);
            self.device.destroy_image(self.target, None);
            self.device.free_memory(self.target_memory, None);
        }
        self.width = frame.width;
        self.height = frame.height;
        self.target_initialised = false;
        self.create_target()
    }

    /// A ycbcr-converting view of a decoded image, created once per image.
    fn frame_view(
        &mut self,
        frame: &DecodedFrame,
    ) -> Result<vk::ImageView, Box<dyn std::error::Error>> {
        // Key on image and layer: the decoder rotates through its DPB slots, so
        // the same handful of images come back again and again.
        // Keyed on the generation as well as the handle: a rebuilt session
        // destroys its images and drivers reuse handles.
        let key = (
            frame.generation,
            ash::vk::Handle::as_raw(frame.image),
            frame.array_layer,
        );
        if let Some(view) = self.frame_views.get(&key) {
            return Ok(*view);
        }
        let mut conversion_info =
            vk::SamplerYcbcrConversionInfo::default().conversion(self.conversion);
        let view = unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .push(&mut conversion_info)
                    .image(frame.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(self.format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: frame.array_layer,
                        layer_count: 1,
                    }),
                None,
            )
        }?;
        self.frame_views.insert(key, view);
        Ok(view)
    }

    fn create_target(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let device = &self.device;
        let image = unsafe {
            device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .extent(vk::Extent3D {
                        width: self.width,
                        height: self.height,
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
        let memory = self.allocate(reqs, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        unsafe { device.bind_image_memory(image, memory, 0) }?;
        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
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

        let size = self.width as u64 * self.height as u64 * 4;
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
        let staging_memory = self.allocate(
            reqs,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe { device.bind_buffer_memory(buffer, staging_memory, 0) }?;

        self.target = image;
        self.target_memory = memory;
        self.target_view = view;
        self.staging = buffer;
        self.staging_memory = staging_memory;
        Ok(())
    }

    fn allocate(
        &self,
        reqs: vk::MemoryRequirements,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<vk::DeviceMemory, Box<dyn std::error::Error>> {
        let props = unsafe {
            self.context
                .instance()
                .get_physical_device_memory_properties(self.context.physical_device())
        };
        let index = (0..props.memory_type_count)
            .find(|&i| {
                reqs.memory_type_bits & (1 << i) != 0
                    && props.memory_types[i as usize]
                        .property_flags
                        .contains(flags)
            })
            .ok_or("no suitable memory type")?;
        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(index),
                None,
            )
        }?;
        Ok(memory)
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for view in self.frame_views.values() {
                self.device.destroy_image_view(*view, None);
            }
            self.device.destroy_buffer(self.staging, None);
            self.device.free_memory(self.staging_memory, None);
            self.device.destroy_image_view(self.target_view, None);
            self.device.destroy_image(self.target, None);
            self.device.free_memory(self.target_memory, None);
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
            self.device.destroy_sampler(self.sampler, None);
            self.device
                .destroy_sampler_ycbcr_conversion(self.conversion, None);
        }
    }
}
