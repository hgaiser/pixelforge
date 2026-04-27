//! Vulkan compute pipeline creation for color conversion.

use super::{ColorConverter, ColorConverterConfig};
use crate::encoder::resources::find_memory_type;
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;
use ash::vk;

/// Create a color converter with all Vulkan resources.
pub fn create_converter(
    context: VideoContext,
    config: ColorConverterConfig,
) -> Result<ColorConverter> {
    let device = context.device();

    // Create descriptor set layout.
    let bindings = [
        // Binding 0: Source image sampler (replaces the old input buffer).
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // Binding 1: Output buffer (YUV)
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(e.to_string()))?;

    // Create pipeline layout with push constants.
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(28); // 7 x u32: width, height, input_format, output_format, color_space, full_range, sdr_white_nits(f32)

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&descriptor_set_layout))
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(e.to_string()))?;

    // Create compute shader module.
    let shader_code = super::shader::get_spirv_code()?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);

    let shader_module = unsafe { device.create_shader_module(&shader_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(e.to_string()))?;

    // Create compute pipeline.
    let entry_point = std::ffi::CString::new("main").unwrap();
    let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&entry_point);

    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage_info)
        .layout(pipeline_layout);

    let pipeline = unsafe {
        device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    }
    .map_err(|(_, e)| PixelForgeError::ResourceCreation(e.to_string()))?[0];

    // Destroy shader module (no longer needed after pipeline creation)
    unsafe { device.destroy_shader_module(shader_module, None) };

    // Create descriptor pool.
    let pool_sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1),
    ];

    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
        .max_sets(1)
        .pool_sizes(&pool_sizes);

    let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(e.to_string()))?;

    // Allocate descriptor set.
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(std::slice::from_ref(&descriptor_set_layout));

    let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info) }
        .map_err(|e| PixelForgeError::ResourceCreation(e.to_string()))?[0];

    // Calculate output buffer size.
    let output_size = config
        .output_format
        .output_size(config.width, config.height);

    // Create a nearest-neighbor sampler for texelFetch (the sampler state doesn't
    // matter for texelFetch, but Vulkan requires a valid one for combined image sampler).
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);

    let sampler = unsafe { device.create_sampler(&sampler_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(format!("sampler creation: {}", e)))?;

    // Create output buffer (device local for compute shader output, transfer source for image copy)
    let (output_buffer, output_memory) = create_buffer(
        device,
        context.memory_properties(),
        output_size as vk::DeviceSize,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    // Write only the output buffer descriptor now; the source image descriptor
    // is written per-frame in convert() when we know the actual source ImageView.
    let output_buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(output_buffer)
        .offset(0)
        .range(output_size as vk::DeviceSize);

    let writes = [vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(std::slice::from_ref(&output_buffer_info))];

    unsafe { device.update_descriptor_sets(&writes, &[]) };

    // Create command pool for compute queue.
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(context.compute_queue_family())
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    // Allocate command buffer.
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?[0];

    // Create fence for synchronization.
    let fence_info = vk::FenceCreateInfo::default();
    let fence = unsafe { device.create_fence(&fence_info, None) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    // Create semaphore for signaling encode queue.
    let semaphore_info = vk::SemaphoreCreateInfo::default();
    let signal_semaphore = unsafe { device.create_semaphore(&semaphore_info, None) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    Ok(ColorConverter {
        context,
        config,
        descriptor_set_layout,
        pipeline_layout,
        pipeline,
        descriptor_pool,
        descriptor_set,
        sampler,
        cached_src_view: None,
        output_buffer,
        output_memory,
        command_pool,
        command_buffer,
        fence,
        signal_semaphore,
        in_flight: false,
    })
}

/// Create a buffer with associated memory.
fn create_buffer(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.create_buffer(&buffer_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(format!("buffer creation: {}", e)))?;

    let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let memory_type_index = find_memory_type(
        memory_properties,
        mem_requirements.memory_type_bits,
        properties,
    )
    .ok_or_else(|| {
        PixelForgeError::MemoryAllocation(format!(
            "No suitable memory type for buffer with properties {:?}",
            properties
        ))
    })?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type_index);

    let memory = unsafe { device.allocate_memory(&alloc_info, None) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

    unsafe { device.bind_buffer_memory(buffer, memory, 0) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

    Ok((buffer, memory))
}
