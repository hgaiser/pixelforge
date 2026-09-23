//! A synthetic RGB frame on the GPU, as a colour converter's input, and the
//! small submission helpers it is built with.

use ash::vk;
use pixelforge::VideoContext;

/// The synthetic frame, on the GPU, cleaned up when the test drops it.
pub struct SrcImage {
    /// Keeps the device alive until after the image is destroyed.
    context: VideoContext,
    pub image: vk::Image,
    memory: vk::DeviceMemory,
}

impl Drop for SrcImage {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe {
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

pub unsafe fn one_shot<F: FnOnce(vk::CommandBuffer)>(
    context: &VideoContext,
    record: F,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = context.device();
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(context.transfer_queue_family())
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let pool = unsafe { device.create_command_pool(&pool_info, None) }?;
    let cb_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cb = unsafe { device.allocate_command_buffers(&cb_info) }?[0];
    unsafe {
        device.begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        record(cb);
        device.end_command_buffer(cb)?;
        let cbs = [cb];
        let submit = vk::SubmitInfo::default().command_buffers(&cbs);
        device.queue_submit(context.transfer_queue(), &[submit], vk::Fence::null())?;
        device.queue_wait_idle(context.transfer_queue())?;
        device.destroy_command_pool(pool, None);
    }
    Ok(())
}

pub unsafe fn host_buffer(
    context: &VideoContext,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), Box<dyn std::error::Error>> {
    let device = context.device();
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&info, None) }?;
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_type = context
        .find_memory_type(
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or("no host visible memory")?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&alloc, None) }?;
    unsafe { device.bind_buffer_memory(buffer, memory, 0) }?;
    Ok((buffer, memory))
}

/// Upload `pixels`, tightly packed BGRA8, as a `width` x `height` image left in
/// `GENERAL`.
///
/// `GENERAL` because that is where the colour converter leaves its source after
/// every conversion, so it is the layout the image is in for every conversion
/// after the first, and the one the encoder's own copy expects. Starting it
/// anywhere else means a second conversion told the wrong layout.
pub unsafe fn create_src_image(
    context: &VideoContext,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<SrcImage, Box<dyn std::error::Error>> {
    let device = context.device();
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::B8G8R8A8_UNORM)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        // Sampled by the converter, and copied from by an encoder taking RGB
        // input, which reads its source with a transfer.
        .usage(
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&info, None) }?;
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let mem_type = context
        .find_memory_type(reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .ok_or("no device local memory")?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&alloc, None) }?;
    unsafe { device.bind_image_memory(image, memory, 0) }?;

    let (staging, staging_mem) = unsafe {
        host_buffer(
            context,
            pixels.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?
    };
    unsafe {
        let ptr = device.map_memory(
            staging_mem,
            0,
            pixels.len() as u64,
            vk::MemoryMapFlags::empty(),
        )?;
        std::ptr::copy_nonoverlapping(pixels.as_ptr(), ptr as *mut u8, pixels.len());
        device.unmap_memory(staging_mem);
    }

    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    unsafe {
        one_shot(context, |cb| {
            let to_dst = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_dst],
            );
            let copy = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                });
            device.cmd_copy_buffer_to_image(
                cb,
                staging,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy],
            );
            let to_read = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ);
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_read],
            );
        })?;
        device.destroy_buffer(staging, None);
        device.free_memory(staging_mem, None);
    }
    Ok(SrcImage {
        context: context.clone(),
        image,
        memory,
    })
}
