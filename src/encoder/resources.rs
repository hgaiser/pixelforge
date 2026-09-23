use crate::encoder::{BitDepth, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::sync::TimelinePoint;
use crate::vulkan::VideoContext;
use ash::vk::TaggedStructure;
use ash::vk::{self, Handle};
use std::ptr;

// Shared, direction-agnostic Vulkan Video helpers live in `crate::video`.
// Re-export them so encoder-internal call sites keep working unchanged.
#[cfg(test)]
pub(crate) use crate::video::gcd;
pub(crate) use crate::video::{
    VideoImageParams, align_up, allocate_command_buffers, allocate_session_memory,
    create_bitstream_buffer, create_command_pool, create_dpb_images as create_dpb_images_shared,
    create_fence, create_video_image, find_memory_type, get_video_format, lcm,
    map_bitstream_buffer, query_supported_video_formats,
};

/// Create the encoder's DPB images.
///
/// Thin wrapper over [`crate::video::create_dpb_images`] that supplies the
/// encode-side usage flags; encode DPB images are only ever touched by the
/// video encode queue, so they need no queue sharing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn create_dpb_images(
    context: &VideoContext,
    width: u32,
    height: u32,
    format: vk::Format,
    count: usize,
    profile_info: &vk::VideoProfileInfoKHR,
    use_layered: bool,
) -> Result<(Vec<vk::Image>, Vec<vk::DeviceMemory>, Vec<vk::ImageView>)> {
    let dpb_params = VideoImageParams {
        width,
        height,
        format,
        usage: vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR,
        sharing_families: &[],
        flags: vk::ImageCreateFlags::empty(),
        view_formats: &[],
    };
    create_dpb_images_shared(context, &dpb_params, profile_info, count, use_layered)
}

/// Create an image for video encoding (input or DPB).
///
/// Thin encoder-specific wrapper over [`crate::video::create_video_image`]:
/// picks the encode usage flags and the queue families that may touch an
/// encode input image (encode + transfer + compute).
pub(crate) fn create_image(
    context: &VideoContext,
    width: u32,
    height: u32,
    format: vk::Format,
    is_dpb: bool,
    profile_info: &vk::VideoProfileInfoKHR,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let (usage, families) = if is_dpb {
        (vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR, Vec::new())
    } else {
        let mut families = Vec::new();
        if let Some(encode_family) = context.video_encode_queue_family() {
            families.push(encode_family);
            families.push(context.transfer_queue_family());
            families.push(context.compute_queue_family());
        }
        (
            vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR | vk::ImageUsageFlags::TRANSFER_DST,
            families,
        )
    };
    let dpb_params = VideoImageParams {
        width,
        height,
        format,
        usage,
        sharing_families: &families,
        flags: vk::ImageCreateFlags::empty(),
        view_formats: &[],
    };
    create_video_image(context, &dpb_params, profile_info)
}

/// Minimum bitstream buffer size.
pub(crate) const MIN_BITSTREAM_BUFFER_SIZE: usize = 2 * 1024 * 1024;

pub(crate) fn create_encode_feedback_query_pool(
    context: &VideoContext,
    profile_info: &mut vk::VideoProfileInfoKHR,
) -> Result<vk::QueryPool> {
    let mut encode_feedback_create = vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR::default()
        .encode_feedback_flags(
            vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET
                | vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN,
        );

    let query_pool_create_info = unsafe {
        vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR)
            .query_count(1)
            .extend(profile_info)
            .push(&mut encode_feedback_create)
    };

    unsafe {
        context
            .device()
            .create_query_pool(&query_pool_create_info, None)
    }
    .map_err(|e| PixelForgeError::QueryPool(e.to_string()))
}

pub(crate) fn create_encode_timestamp_query_pool(context: &VideoContext) -> Result<vk::QueryPool> {
    let timestamp_query_pool_create_info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::TIMESTAMP)
        .query_count(2); // start and end

    unsafe {
        context
            .device()
            .create_query_pool(&timestamp_query_pool_create_info, None)
    }
    .map_err(|e| PixelForgeError::QueryPool(e.to_string()))
}

/// Command resources for encoding operations.
pub(crate) struct CommandResources {
    /// Command pool for encode commands.
    pub command_pool: vk::CommandPool,
    /// Command pool for upload/transfer commands (may differ from command_pool when
    /// the encode queue does not support transfer operations).
    pub upload_command_pool: vk::CommandPool,
    /// Command buffer for upload operations.
    pub upload_command_buffer: vk::CommandBuffer,
    /// Fence for upload synchronization.
    pub upload_fence: vk::Fence,
}

/// Create command resources for encoding.
///
/// `encode_queue_family` is the queue family used for video encode commands.
/// `upload_queue_family` is the queue family used for transfer (upload) commands.
/// They may be the same if the encode queue supports transfer operations.
pub(crate) fn create_command_resources(
    context: &VideoContext,
    encode_queue_family: u32,
    upload_queue_family: u32,
) -> Result<CommandResources> {
    let command_pool = create_command_pool(context, encode_queue_family, "encode")?;

    // The upload pool is the encode pool when one family does both.
    let upload_command_pool = if upload_queue_family == encode_queue_family {
        command_pool
    } else {
        create_command_pool(context, upload_queue_family, "encode upload")?
    };

    let upload_command_buffer = allocate_command_buffers(context, upload_command_pool, 1)?[0];

    // Unsignaled: the upload path always submits before it waits. Per-slot
    // encode fences are created by the encode pipeline (see `encoder::pipeline`).
    let upload_fence = create_fence(context, false)?;

    Ok(CommandResources {
        command_pool,
        upload_command_pool,
        upload_command_buffer,
        upload_fence,
    })
}

/// Clear an RGB input image to opaque black and leave it in
/// `VIDEO_ENCODE_SRC_KHR`.
///
/// The YUV path fills each plane from a staging buffer; an RGB image has a
/// single colour plane, so a clear does it.
pub(crate) fn clear_rgb_input_image(
    context: &VideoContext,
    params: &ClearImageParams,
) -> Result<()> {
    let device = context.device();
    let cb = params.command_buffer;
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let err = |e: vk::Result| PixelForgeError::CommandBuffer(e.to_string());
    unsafe {
        device
            .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
            .map_err(err)?;
        device
            .begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(err)?;
        let to_transfer = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(params.image)
            .subresource_range(range)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_transfer],
        );
        device.cmd_clear_color_image(
            cb,
            params.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
            &[range],
        );
        let to_encode = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(params.image)
            .subresource_range(range)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_encode],
        );
        device.end_command_buffer(cb).map_err(err)?;
        let cbs = [cb];
        device.reset_fences(&[params.fence]).map_err(err)?;
        device
            .queue_submit(
                params.queue,
                &[vk::SubmitInfo::default().command_buffers(&cbs)],
                params.fence,
            )
            .map_err(err)?;
        device
            .wait_for_fences(&[params.fence], true, u64::MAX)
            .map_err(err)?;
        device.reset_fences(&[params.fence]).map_err(err)?;
    }
    Ok(())
}

/// Create DPB images for video encoding.
///
/// When `use_layered` is true (required when the driver does not support
/// `VK_VIDEO_CAPABILITY_SEPARATE_REFERENCE_IMAGES_BIT_KHR`), a single
/// `VkImage` with `array_layers = count` is created and one `VkImageView`
/// per layer is returned.  The image and memory vectors will have a single
/// entry while the view vector will have `count` entries.
///
/// When `use_layered` is false the previous behaviour is preserved: one
/// separate image/memory/view per DPB slot.
pub(crate) fn clear_input_image(context: &VideoContext, params: &ClearImageParams) -> Result<()> {
    let device = context.device();
    let bytes_per_component: u32 = match params.bit_depth {
        BitDepth::Eight => 1,
        BitDepth::Ten => 2,
    };

    // Calculate per-plane sizes.
    // For YUV444, align Y plane size to 4 bytes so the UV plane buffer offset
    // meets VkBufferImageCopy::bufferOffset alignment requirements.
    // YUV420/422 dimensions are always even, so alignment is naturally satisfied.
    let plane0_raw = (params.width * params.height * bytes_per_component) as usize;
    let plane0_size = match params.pixel_format {
        PixelFormat::Yuv444 => crate::align4(plane0_raw) as u32,
        _ => plane0_raw as u32,
    };
    let plane1_size = match params.pixel_format {
        // YUV 4:2:0 (e.g., NV12): UV plane is half width, half height, 2 components per pixel.
        PixelFormat::Yuv420 => (params.width / 2) * (params.height / 2) * 2 * bytes_per_component,
        // YUV 4:2:2: UV plane is half width, full height, 2 components per pixel.
        PixelFormat::Yuv422 => (params.width / 2) * params.height * 2 * bytes_per_component,
        // YUV 4:4:4 (e.g., NV24): UV plane is full width, full height, 2 components per pixel.
        PixelFormat::Yuv444 => params.width * params.height * 2 * bytes_per_component,
    };
    let total_size = (plane0_size + plane1_size) as vk::DeviceSize;

    // Create a staging buffer filled with zeros.
    let buffer_create_info = vk::BufferCreateInfo::default()
        .size(total_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let staging_buffer = unsafe { device.create_buffer(&buffer_create_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(format!("staging buffer: {}", e)))?;

    let mem_requirements = unsafe { device.get_buffer_memory_requirements(staging_buffer) };
    let memory_type_index = find_memory_type(
        context.memory_properties(),
        mem_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| {
        PixelForgeError::MemoryAllocation("No suitable memory type for staging buffer".to_string())
    })?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type_index);

    let staging_memory = unsafe { device.allocate_memory(&alloc_info, None) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

    unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

    // Map and zero-fill.
    let data_ptr =
        unsafe { device.map_memory(staging_memory, 0, total_size, vk::MemoryMapFlags::empty()) }
            .map_err(|e| PixelForgeError::MemoryAllocation(format!("map staging buffer: {}", e)))?;
    unsafe { ptr::write_bytes(data_ptr as *mut u8, 0, total_size as usize) };
    unsafe { device.unmap_memory(staging_memory) };

    // Record commands.
    unsafe {
        device.reset_command_buffer(params.command_buffer, vk::CommandBufferResetFlags::empty())
    }
    .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(params.command_buffer, &begin_info) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    // Transition image from UNDEFINED to TRANSFER_DST.
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(params.image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

    unsafe {
        device.cmd_pipeline_barrier(
            params.command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    // Copy from staging buffer to image planes.
    let (uv_width, uv_height) = match params.pixel_format {
        PixelFormat::Yuv420 => (params.width / 2, params.height / 2),
        PixelFormat::Yuv444 => (params.width, params.height),
        _ => (params.width / 2, params.height / 2),
    };

    let copy_regions = [
        vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::PLANE_0,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width: params.width,
                height: params.height,
                depth: 1,
            },
        },
        vk::BufferImageCopy {
            buffer_offset: plane0_size as vk::DeviceSize,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::PLANE_1,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width: uv_width,
                height: uv_height,
                depth: 1,
            },
        },
    ];

    unsafe {
        device.cmd_copy_buffer_to_image(
            params.command_buffer,
            staging_buffer,
            params.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &copy_regions,
        );
    }

    // Transition image to VIDEO_ENCODE_SRC.
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(params.image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::empty());

    unsafe {
        device.cmd_pipeline_barrier(
            params.command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    unsafe { device.end_command_buffer(params.command_buffer) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    // Submit and wait.
    let submit_info =
        vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&params.command_buffer));
    unsafe { device.reset_fences(&[params.fence]) }
        .map_err(|e| PixelForgeError::CommandBuffer(format!("reset fence: {}", e)))?;
    unsafe { device.queue_submit(params.queue, &[submit_info], params.fence) }
        .map_err(|e| PixelForgeError::CommandBuffer(format!("submit clear: {}", e)))?;
    unsafe { device.wait_for_fences(&[params.fence], true, u64::MAX) }
        .map_err(|e| PixelForgeError::CommandBuffer(format!("wait clear: {}", e)))?;
    unsafe { device.reset_fences(&[params.fence]) }
        .map_err(|e| PixelForgeError::CommandBuffer(format!("reset fence after clear: {}", e)))?;

    // Clean up staging buffer.
    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_memory, None);
    }

    Ok(())
}

/// Parameters for uploading an image to the encoder's input image.
pub(crate) struct UploadParams<'a> {
    /// The command buffer to use for the upload.
    pub upload_command_buffer: vk::CommandBuffer,
    /// The fence to use for synchronization.
    pub upload_fence: vk::Fence,
    /// The source image to copy from.
    pub src_image: vk::Image,
    /// The destination image to copy to.
    pub dst_image: vk::Image,
    /// The width of the image.
    pub width: u32,
    /// The height of the image.
    pub height: u32,
    /// The pixel format of the image.
    pub pixel_format: PixelFormat,
    /// The current layout of the input image.
    pub input_image_layout: vk::ImageLayout,
    /// Whether both images are single-plane RGB rather than YUV.
    pub rgb: bool,
    /// The queue to submit transfer operations to.
    pub upload_queue: vk::Queue,
    /// Caller work the copy must wait for.
    pub waits: &'a [TimelinePoint],
}

/// Upload an image to the encoder's input image via GPU-to-GPU copy.
///
/// This function handles:
/// - Resetting and beginning the command buffer
/// - Transitioning source image from GENERAL to TRANSFER_SRC
/// - Transitioning destination image from `input_image_layout` to TRANSFER_DST
/// - Copying Y and UV planes (NV12 format)
/// - Transitioning destination image to VIDEO_ENCODE_SRC
/// - Transitioning source image back to GENERAL
/// - Submitting the command buffer and waiting for completion
///
/// Returns Ok(()) on success, or an error if any Vulkan operation fails.
pub(crate) fn upload_image_to_input(
    context: &crate::vulkan::VideoContext,
    params: &UploadParams<'_>,
) -> Result<()> {
    let device = context.device();

    unsafe {
        device.reset_command_buffer(
            params.upload_command_buffer,
            vk::CommandBufferResetFlags::empty(),
        )
    }
    .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(params.upload_command_buffer, &begin_info) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    // Transition source image from GENERAL to TRANSFER_SRC.
    let src_barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(params.src_image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

    // Transition destination image to TRANSFER_DST.
    let dst_barrier = vk::ImageMemoryBarrier::default()
        .old_layout(params.input_image_layout)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(params.dst_image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

    unsafe {
        device.cmd_pipeline_barrier(
            params.upload_command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[src_barrier, dst_barrier],
        );
    }

    // Copy image to image using per-plane copy regions (NV12 format).
    // Copy Y plane (plane 0).
    let y_copy_region = vk::ImageCopy {
        src_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::PLANE_0,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        dst_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::PLANE_0,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        extent: vk::Extent3D {
            width: params.width,
            height: params.height,
            depth: 1,
        },
    };

    // Copy UV plane (plane 1).
    let (uv_width, uv_height) = match params.pixel_format {
        PixelFormat::Yuv420 => (params.width / 2, params.height / 2),
        PixelFormat::Yuv444 => (params.width, params.height),
        _ => (params.width / 2, params.height / 2),
    };

    let uv_copy_region = vk::ImageCopy {
        src_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::PLANE_1,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        src_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        dst_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::PLANE_1,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        dst_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        extent: vk::Extent3D {
            width: uv_width,
            height: uv_height,
            depth: 1,
        },
    };

    // An RGB input image has one colour plane, copied whole.
    let rgb_copy_region = vk::ImageCopy {
        src_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        dst_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        ..y_copy_region
    };
    let regions: &[vk::ImageCopy] = if params.rgb {
        &[rgb_copy_region]
    } else {
        &[y_copy_region, uv_copy_region]
    };

    unsafe {
        device.cmd_copy_image(
            params.upload_command_buffer,
            params.src_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            params.dst_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            regions,
        );
    }

    // Transition destination image to VIDEO_ENCODE_SRC.
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::VIDEO_ENCODE_SRC_KHR)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(params.dst_image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::empty());

    // Also transition source image back to GENERAL for reuse.
    let src_barrier_back = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(params.src_image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);

    unsafe {
        device.cmd_pipeline_barrier(
            params.upload_command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier, src_barrier_back],
        );
    }

    unsafe { device.end_command_buffer(params.upload_command_buffer) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    let command_buffers =
        [vk::CommandBufferSubmitInfo::default().command_buffer(params.upload_command_buffer)];
    let waits: Vec<vk::SemaphoreSubmitInfo> =
        params.waits.iter().map(TimelinePoint::wait_info).collect();
    let submit_info = vk::SubmitInfo2::default()
        .wait_semaphore_infos(&waits)
        .command_buffer_infos(&command_buffers);

    unsafe {
        context
            .sync2()
            .queue_submit2(params.upload_queue, &[submit_info], params.upload_fence)
    }
    .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    unsafe { device.wait_for_fences(&[params.upload_fence], true, u64::MAX) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    unsafe { device.reset_fences(&[params.upload_fence]) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

    Ok(())
}

/// Record DPB image barriers for encode.
///
/// Transitions the setup DPB slot from UNDEFINED to VIDEO_ENCODE_DPB and
/// adds execution barriers for reference slot images.
///
/// # Safety
///
/// The command buffer must be in recording state.
pub(crate) unsafe fn record_dpb_barriers(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    dpb_images: &[vk::Image],
    use_layered_dpb: bool,
    current_dpb_slot: u8,
    reference_dpb_slots: &[u8],
    setup_slot_active: bool,
) {
    let dpb_image = if use_layered_dpb {
        dpb_images[0]
    } else {
        dpb_images[current_dpb_slot as usize]
    };
    let dpb_base_array_layer = if use_layered_dpb {
        current_dpb_slot as u32
    } else {
        0
    };

    // Use UNDEFINED only on first use of a DPB slot; after that it is already
    // in VIDEO_ENCODE_DPB_KHR and transitioning from UNDEFINED would discard
    // the contents, which is invalid/UB.
    let setup_old_layout = if setup_slot_active {
        vk::ImageLayout::VIDEO_ENCODE_DPB_KHR
    } else {
        vk::ImageLayout::UNDEFINED
    };

    let dpb_barrier = vk::ImageMemoryBarrier::default()
        .old_layout(setup_old_layout)
        .new_layout(vk::ImageLayout::VIDEO_ENCODE_DPB_KHR)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(dpb_image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: dpb_base_array_layer,
            layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::empty())
        // The encode writes the reconstructed picture into this slot, so it has
        // to be in the barrier's second scope, or the layout transition and that
        // write are unordered. The original barrier API has no video access
        // flags; MEMORY_* covers the encode's reads and writes.
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);

    let mut all_barriers = vec![dpb_barrier];

    for &ref_slot in reference_dpb_slots {
        let (ref_image, ref_layer) = if use_layered_dpb {
            (dpb_images[0], ref_slot as u32)
        } else {
            (dpb_images[ref_slot as usize], 0u32)
        };
        all_barriers.push(
            vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::VIDEO_ENCODE_DPB_KHR)
                .new_layout(vk::ImageLayout::VIDEO_ENCODE_DPB_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(ref_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: ref_layer,
                    layer_count: 1,
                })
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ),
        );
    }

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &all_barriers,
        );
    }
}

/// Prepare an encode command buffer for recording.
///
/// Resets the command buffer, begins recording with ONE_TIME_SUBMIT, and resets
/// the query pool. This is the common preamble for all encode operations.
///
/// # Safety
///
/// The command buffer must not be in use by the GPU.
pub(crate) unsafe fn prepare_encode_command_buffer(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    query_pool: vk::QueryPool,
) -> Result<()> {
    unsafe {
        device
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
            .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
    }

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
    }

    unsafe {
        device.cmd_reset_query_pool(command_buffer, query_pool, 0, 1);
    }

    Ok(())
}

/// Record a post-encode DPB synchronization barrier.
///
/// Ensures the DPB image write from the encode operation is visible to subsequent
/// reads (e.g. as a reference frame for the next encode).
///
/// # Safety
///
/// The command buffer must be in recording state.
pub(crate) unsafe fn record_post_encode_dpb_barrier(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    dpb_images: &[vk::Image],
    use_layered_dpb: bool,
    current_dpb_slot: u8,
) {
    let (post_dpb_image, post_dpb_layer) = if use_layered_dpb {
        (dpb_images[0], current_dpb_slot as u32)
    } else {
        (dpb_images[current_dpb_slot as usize], 0)
    };

    let dpb_sync_barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::VIDEO_ENCODE_DPB_KHR)
        .new_layout(vk::ImageLayout::VIDEO_ENCODE_DPB_KHR)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(post_dpb_image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: post_dpb_layer,
            layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ);

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[dpb_sync_barrier],
        );
    }
}

/// Submit an encode command buffer without waiting for completion.
///
/// Timeline semaphore waits/signals are used to keep shared DPB access ordered
/// across pipelined slots while leaving bitstream readback delayed. Uses
/// Synchronization2 (`queue_submit2`) so the timeline wait/signal are scoped to
/// the `VIDEO_ENCODE` stage rather than `ALL_COMMANDS`, avoiding unnecessary
/// over-synchronization.
///
/// # Safety
///
/// The command buffer must have been ended.
pub(crate) unsafe fn submit_encode_only(
    device: &ash::Device,
    sync2: &ash::khr::synchronization2::Device,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    encode_queue: vk::Queue,
    wait_infos: &[vk::SemaphoreSubmitInfo<'_>],
    signal_timeline: Option<(vk::Semaphore, u64)>,
) -> Result<()> {
    let command_buffer_info = vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer);
    let command_buffer_infos = [command_buffer_info];

    let signal_infos: Vec<vk::SemaphoreSubmitInfo> = signal_timeline
        .into_iter()
        .map(|(semaphore, value)| {
            vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore)
                .value(value)
                .stage_mask(vk::PipelineStageFlags2::VIDEO_ENCODE_KHR)
        })
        .collect();

    let submit_info = vk::SubmitInfo2::default()
        .wait_semaphore_infos(wait_infos)
        .command_buffer_infos(&command_buffer_infos)
        .signal_semaphore_infos(&signal_infos);

    unsafe {
        device
            .reset_fences(&[fence])
            .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;
    }

    unsafe {
        sync2
            .queue_submit2(encode_queue, &[submit_info], fence)
            .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
    }

    Ok(())
}

/// Wait for a submitted encode and copy its bitstream data.
///
/// # Safety
///
/// The bitstream buffer pointer must be valid and persistently mapped.
/// Wait for the encode to finish, then append its bitstream directly onto `dst`
/// (which already holds any codec header). Appending in place means the encoded
/// bytes are copied out of the mapped buffer exactly once — there is no
/// intermediate owned `Vec`.
///
/// `bitstream_buffer_size` is the mapped buffer's capacity; if the reported
/// `bytes_written` (+ `offset`) exceeds it, return [`PixelForgeError::BufferOverflow`]
/// instead of reading out of bounds.
pub(crate) unsafe fn wait_and_read_bitstream(
    device: &ash::Device,
    fence: vk::Fence,
    query_pool: vk::QueryPool,
    bitstream_buffer_ptr: *const u8,
    bitstream_buffer_size: usize,
    dst: &mut Vec<u8>,
) -> Result<()> {
    unsafe {
        device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
    }

    // Read query results (offset + bytes_written).
    #[repr(C)]
    struct QueryResult {
        offset: u32,
        bytes_written: u32,
    }

    let mut query_results = [QueryResult {
        offset: 0,
        bytes_written: 0,
    }];

    unsafe {
        device
            .get_query_pool_results(
                query_pool,
                0,
                &mut query_results,
                vk::QueryResultFlags::WAIT,
            )
            .map_err(|e| PixelForgeError::QueryPool(e.to_string()))?;
    }

    let offset = query_results[0].offset as usize;
    let size = query_results[0].bytes_written as usize;

    if size == 0 {
        return Err(PixelForgeError::QueryPool(
            "Encoder produced 0 bytes".to_string(),
        ));
    }

    tracing::debug!("Encoded frame: offset={}, size={}", offset, size);

    // Guard against writing past the destination range: `bytes_written > capacity` is the only overflow signal.
    if offset.saturating_add(size) > bitstream_buffer_size {
        return Err(PixelForgeError::BufferOverflow {
            written: offset + size,
            capacity: bitstream_buffer_size,
        });
    }

    let src = unsafe { std::slice::from_raw_parts(bitstream_buffer_ptr.add(offset), size) };
    dst.extend_from_slice(src);

    Ok(())
}

/// The per-codec Vulkan resources torn down on `Drop`, other than the encode
/// pipeline (which the engine owns and destroys). Bundled so the identical
/// teardown lives in one place; see [`destroy_encoder_resources`].
pub(crate) struct EncoderTeardown<'a> {
    pub command_pool: vk::CommandPool,
    pub upload_command_pool: vk::CommandPool,
    pub upload_fence: vk::Fence,
    pub dpb_images: &'a [vk::Image],
    pub dpb_image_views: &'a [vk::ImageView],
    pub dpb_image_memories: &'a [vk::DeviceMemory],
    pub session: vk::VideoSessionKHR,
    pub session_params: vk::VideoSessionParametersKHR,
    pub session_memory: &'a [vk::DeviceMemory],
}

/// Destroy the common encoder resources (command pools, upload fence, DPB
/// images, and the video session). Identical across codecs.
///
/// # Safety
///
/// All queues that may reference these resources must be idle.
pub(crate) unsafe fn destroy_encoder_resources(
    device: &ash::Device,
    video_queue_fn: &ash::khr::video_queue::Device,
    res: &EncoderTeardown,
) {
    unsafe {
        device.destroy_fence(res.upload_fence, None);
        device.destroy_command_pool(res.command_pool, None);
        if res.upload_command_pool != res.command_pool {
            device.destroy_command_pool(res.upload_command_pool, None);
        }

        for &view in res.dpb_image_views {
            device.destroy_image_view(view, None);
        }
        for &image in res.dpb_images {
            device.destroy_image(image, None);
        }
        for &memory in res.dpb_image_memories {
            device.free_memory(memory, None);
        }

        if res.session_params != vk::VideoSessionParametersKHR::null() {
            (video_queue_fn.fp().destroy_video_session_parameters_khr)(
                device.handle(),
                res.session_params,
                std::ptr::null(),
            );
        }
        (video_queue_fn.fp().destroy_video_session_khr)(
            device.handle(),
            res.session,
            std::ptr::null(),
        );
        for &memory in res.session_memory {
            device.free_memory(memory, None);
        }
    }
}

/// Retrieve driver-encoded video session parameters (H.264 SPS/PPS, H.265
/// VPS/SPS/PPS, or AV1 sequence header) into a byte buffer, retrying on
/// `INCOMPLETE`.
///
/// The codec-specific `*GetInfoKHR` and feedback structs are built by the caller
/// and chained into `get_info`/`feedback`; this owns only the identical query
/// loop. A preallocated buffer is always provided because some drivers misbehave
/// on a size-only query (`pData == NULL`), notably for 4:4:4 profiles.
pub(crate) fn get_encoded_session_params(
    context: &VideoContext,
    video_encode_fn: &ash::khr::video_encode_queue::Device,
    get_info: &vk::VideoEncodeSessionParametersGetInfoKHR,
    feedback: &mut vk::VideoEncodeSessionParametersFeedbackInfoKHR,
) -> Result<Vec<u8>> {
    let mut data = vec![0u8; 4096];
    let mut data_size: usize = data.len();
    let mut attempts = 0;
    loop {
        attempts += 1;
        let result = unsafe {
            (video_encode_fn
                .fp()
                .get_encoded_video_session_parameters_khr)(
                context.device().handle(),
                get_info,
                feedback,
                &mut data_size,
                data.as_mut_ptr() as *mut std::ffi::c_void,
            )
        };

        match result {
            vk::Result::SUCCESS => {
                if data_size == 0 {
                    return Err(PixelForgeError::SessionParametersCreation(
                        "Encoded session parameters size is 0".to_string(),
                    ));
                }
                data.truncate(data_size);
                return Ok(data);
            }
            // Driver indicates the buffer was too small; resize to the reported
            // required size (or grow conservatively if it is not provided).
            vk::Result::INCOMPLETE if attempts < 3 => {
                let new_size = data_size.max(data.len() * 2).max(1);
                data.resize(new_size, 0);
                data_size = data.len();
            }
            err => {
                return Err(PixelForgeError::SessionParametersCreation(format!(
                    "Failed to get encoded session parameters: {err:?}"
                )));
            }
        }
    }
}

/// Resets the given query pool and writes starting timestamp command.
///
/// No-op when `query_pool` is null (i.e. the encode queue family does not
/// support timestamp queries).
pub(crate) fn reset_start_timestamp(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    query_pool: vk::QueryPool,
) {
    if query_pool.is_null() {
        return;
    }
    unsafe {
        device.cmd_reset_query_pool(command_buffer, query_pool, 0, 2);
        device.cmd_write_timestamp(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            query_pool,
            0, // query index 0 = start
        );
    }
}

/// Writes ending timestamp command to the given query pool.
///
/// No-op when `query_pool` is null (i.e. the encode queue family does not
/// support timestamp queries).
pub(crate) fn end_timestamp(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    query_pool: vk::QueryPool,
) {
    if query_pool.is_null() {
        return;
    }
    unsafe {
        device.cmd_write_timestamp(
            command_buffer,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            query_pool,
            1, // query index 1 = end
        );
    }
}

/// Queries the given query pool for recorded timestamps, returning their difference.
///
/// Returns `None` when `query_pool` is null (i.e. the encode queue family does
/// not support timestamp queries).
///
/// # Safety
/// This must only be called if both `reset_start_timestamp` and `end_timestamp`
///  were previously written and executed for the given query pool.
pub(crate) unsafe fn query_timestamp_diff(
    device: &ash::Device,
    query_pool: vk::QueryPool,
    mut timestamps: [u64; 2],
    timestamp_period: f32,
) -> Option<u64> {
    if query_pool.is_null() {
        return None;
    }
    let mut encode_time_ns: Option<u64> = None;
    let result = unsafe {
        device.get_query_pool_results(
            query_pool,
            0,               // first_query
            &mut timestamps, // data slice (length 2)
            vk::QueryResultFlags::WAIT | vk::QueryResultFlags::TYPE_64,
        )
    };
    if result.is_ok() {
        encode_time_ns = Some(((timestamps[1] - timestamps[0]) as f32 * timestamp_period) as u64);
    }
    encode_time_ns
}

pub(crate) struct ClearImageParams {
    pub command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub queue: vk::Queue,
    pub image: vk::Image,
    pub width: u32,
    pub height: u32,
    pub pixel_format: PixelFormat,
    pub bit_depth: BitDepth,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(8, 12), 4);
        assert_eq!(gcd(16, 16), 16);
        assert_eq!(gcd(7, 3), 1);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(5, 0), 5);
    }

    #[test]
    fn test_lcm() {
        assert_eq!(lcm(32, 64), 64);
        assert_eq!(lcm(16, 12), 48);
        assert_eq!(lcm(4, 6), 12);
        assert_eq!(lcm(0, 5), 0);
        assert_eq!(lcm(5, 0), 0);
        assert_eq!(lcm(7, 7), 7);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(130, 64), 192);
        assert_eq!(align_up(128, 64), 128);
        assert_eq!(align_up(1, 64), 64);
        assert_eq!(align_up(0, 64), 0);
        assert_eq!(align_up(100, 1), 100);
        assert_eq!(align_up(100, 0), 100);
        // AMD-realistic case: align 320 to lcm(32, 64) = 64.
        assert_eq!(align_up(320, lcm(32, 64)), 320);
        // AMD-realistic case: align 130 to lcm(32, 16) = 32.
        assert_eq!(align_up(130, lcm(32, 16)), 160);
    }
}
