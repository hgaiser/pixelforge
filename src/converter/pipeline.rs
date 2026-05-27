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
    if !context.has_descriptor_buffer() {
        return Err(PixelForgeError::NoSuitableDevice(
            "VK_EXT_descriptor_buffer with capture-replay is required but not available on this device".to_string(),
        ));
    }

    let device = context.device();
    let instance = context.instance();
    let physical_device = context.physical_device();

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

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::DESCRIPTOR_BUFFER_EXT)
        .bindings(&bindings);

    let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(e.to_string()))?;

    // Create pipeline layout with push constants.
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(40); // 10 x u32, matching the GLSL PushConstants block.

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

    // Calculate output buffer size.
    let output_size = config
        .output_format
        .output_size(config.width, config.height);

    // Create a linear sampler for optional source scaling and aspect-preserving padding.
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);

    let sampler = unsafe { device.create_sampler(&sampler_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(format!("sampler creation: {}", e)))?;

    // Create output buffer (device local for compute shader output, transfer
    // source for image copy). Needs SHADER_DEVICE_ADDRESS so we can query a
    // device address to feed `VkDescriptorAddressInfoEXT` when populating
    // the storage-buffer descriptor at runtime.
    let (output_buffer, output_memory) = create_buffer(
        device,
        context.memory_properties(),
        output_size as vk::DeviceSize,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    // Resolve the output buffer's device address. This goes into the
    // `VkDescriptorAddressInfoEXT` we pass to `vkGetDescriptorEXT` when
    // populating the storage-buffer descriptor each frame.
    let output_buffer_address = unsafe {
        device.get_buffer_device_address(
            &vk::BufferDeviceAddressInfo::default().buffer(output_buffer),
        )
    };

    // Query descriptor buffer properties to determine correct descriptor sizes.
    // The bug in the previous implementation was using the *capture-replay*
    // sizes (e.g. `sampler_capture_replay_descriptor_data_size`) — those are
    // for capture/replay tooling like RenderDoc and have nothing to do with
    // runtime descriptor population. The correct sizes for in-buffer
    // descriptors are the regular `*_descriptor_size` fields.
    let mut db_props = vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
    let mut props = vk::PhysicalDeviceProperties2 {
        p_next: &mut db_props as *mut _ as *mut _,
        ..Default::default()
    };
    unsafe {
        instance.get_physical_device_properties2(physical_device, &mut props);
    }
    let combined_image_sampler_size = db_props.combined_image_sampler_descriptor_size;
    let storage_buffer_size_descriptor = db_props.storage_buffer_descriptor_size;

    // Query descriptor set layout size and binding offsets for correct buffer sizing.
    let ext_device =
        ash::ext::descriptor_buffer::Device::load(context.instance(), context.device());
    let vk_device = context.device().handle();
    let mut layout_size = 0u64;
    unsafe {
        (ext_device.fp().get_descriptor_set_layout_size_ext)(
            vk_device,
            descriptor_set_layout,
            &mut layout_size,
        );
    }
    let binding0_offset = unsafe {
        let mut offset = 0u64;
        (ext_device.fp().get_descriptor_set_layout_binding_offset_ext)(
            vk_device,
            descriptor_set_layout,
            0,
            &mut offset,
        );
        offset
    };
    let binding1_offset = unsafe {
        let mut offset = 0u64;
        (ext_device.fp().get_descriptor_set_layout_binding_offset_ext)(
            vk_device,
            descriptor_set_layout,
            1,
            &mut offset,
        );
        offset
    };

    // Descriptor buffer layout:
    //   Offset 0:    Sampler + image view capture payload (binding 0)
    //   Offset X:    Buffer capture payload (binding 1)
    // The total size is the layout size which accounts for alignment.
    let descriptor_buffer_size: vk::DeviceSize = layout_size as vk::DeviceSize;

    let (descriptor_buffer, descriptor_buffer_memory) =
        crate::encoder::resources::create_buffer_with_device_address(
            device,
            context.memory_properties(),
            descriptor_buffer_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

    // Get the buffer's device address for binding descriptor buffers.
    // cmdBindDescriptorBuffers requires the buffer's device address, not the memory capture address.
    let buf_addr_info = vk::BufferDeviceAddressInfo::default().buffer(descriptor_buffer);
    let descriptor_buffer_address = unsafe { device.get_buffer_device_address(&buf_addr_info) };

    // Persistent map the descriptor buffer (HOST_COHERENT, no flush needed).
    let descriptor_buffer_ptr = unsafe {
        device
            .map_memory(
                descriptor_buffer_memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| {
                PixelForgeError::ResourceCreation(format!("map descriptor buffer: {}", e))
            })?
    };
    let descriptor_buffer_ptr = descriptor_buffer_ptr as *mut u8;

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

    Ok(ColorConverter {
        context,
        config,
        descriptor_set_layout,
        pipeline_layout,
        pipeline,
        sampler,
        cached_src_view: None,
        output_buffer,
        output_memory,
        output_buffer_size: output_size,
        output_buffer_address,
        command_pool,
        command_buffer,
        fence,
        // Descriptor buffer fields.
        descriptor_buffer,
        descriptor_buffer_memory,
        descriptor_buffer_address,
        descriptor_buffer_usage: vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT,
        descriptor_buffer_ptr,
        combined_image_sampler_descriptor_size: combined_image_sampler_size,
        storage_buffer_descriptor_size: storage_buffer_size_descriptor,
        ext_device,
        // Layout info for correct offset computation.
        binding0_offset,
        binding1_offset,
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
