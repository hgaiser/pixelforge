//! Codec- and direction-agnostic Vulkan Video helpers.
//!
//! Everything here is shared between the encoder and the decoder: format and
//! capability queries, video-profile-tagged image and buffer creation, video
//! session memory binding, and small arithmetic helpers. Direction-specific
//! resource management (encode slot pipelines, decode DPB pools) lives in
//! `encoder::resources` and `decoder` respectively, built on these primitives.

use crate::encoder::{BitDepth, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;
use ash::vk;
use ash::vk::TaggedStructure;
use std::ptr;

/// Compute greatest common divisor of two values.
pub(crate) fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let tmp = a % b;
        a = b;
        b = tmp;
    }
    a
}

/// Compute least common multiple of two values.
pub(crate) fn lcm(a: u32, b: u32) -> u32 {
    if a == 0 || b == 0 {
        0
    } else {
        (a / gcd(a, b)).saturating_mul(b)
    }
}

/// Align a value up to the next multiple of the given alignment.
pub(crate) fn align_up(value: u32, alignment: u32) -> u32 {
    if alignment <= 1 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

/// Parameters describing a video image's dimensions, format, usage and queue sharing.
pub(crate) struct VideoImageParams<'a> {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub sharing_families: &'a [u32],
    /// Extra creation flags, which for a video image may only be ones the
    /// driver reported in [`VideoFormatSupport::create_flags`].
    pub flags: vk::ImageCreateFlags,
    /// Formats views of this image may use, when `flags` includes
    /// `MUTABLE_FORMAT`. Naming them lets a driver keep compression it would
    /// otherwise have to give up, since it no longer has to assume the image
    /// might be reinterpreted as anything at all. Empty chains nothing.
    pub view_formats: &'a [vk::Format],
}

/// Deduplicate `families` and decide CONCURRENT vs EXCLUSIVE sharing mode.
/// Returns an empty family list for EXCLUSIVE, in which case Vulkan ignores it.
pub(crate) fn resolve_sharing_mode(families: &[u32]) -> (Vec<u32>, vk::SharingMode) {
    let mut deduped: Vec<u32> = Vec::new();
    for &family in families {
        if !deduped.contains(&family) {
            deduped.push(family);
        }
    }
    if deduped.len() > 1 {
        (deduped, vk::SharingMode::CONCURRENT)
    } else {
        (Vec::new(), vk::SharingMode::EXCLUSIVE)
    }
}

fn allocate_and_bind_image_memory(
    context: &VideoContext,
    image: vk::Image,
) -> Result<vk::DeviceMemory> {
    let mem_requirements = unsafe { context.device().get_image_memory_requirements(image) };

    let memory_type_index = find_memory_type(
        context.memory_properties(),
        mem_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| {
        PixelForgeError::MemoryAllocation("No suitable memory type for image".to_string())
    })?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type_index);

    let memory = unsafe { context.device().allocate_memory(&alloc_info, None) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

    match unsafe { context.device().bind_image_memory(image, memory, 0) } {
        Ok(()) => Ok(memory),
        Err(e) => {
            unsafe { context.device().free_memory(memory, None) };
            Err(PixelForgeError::MemoryAllocation(e.to_string()))
        }
    }
}

/// Create a video-profile-tagged image with `array_layers`.
fn create_video_image_raw(
    context: &VideoContext,
    params: &VideoImageParams,
    array_layers: u32,
    profile_info: &vk::VideoProfileInfoKHR,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let (queue_families, sharing_mode) = resolve_sharing_mode(params.sharing_families);

    let profiles = [*profile_info];
    let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
    let mut format_list =
        vk::ImageFormatListCreateInfo::default().view_formats(params.view_formats);

    let mut create_info = vk::ImageCreateInfo::default()
        .flags(params.flags)
        .image_type(vk::ImageType::TYPE_2D)
        .format(params.format)
        .extent(vk::Extent3D {
            width: params.width,
            height: params.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(array_layers)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(params.usage)
        .sharing_mode(sharing_mode)
        .queue_family_indices(&queue_families)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push(&mut profile_list);
    if !params.view_formats.is_empty() {
        create_info = create_info.push(&mut format_list);
    }

    let image = unsafe { context.device().create_image(&create_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(format!("image creation: {}", e)))?;

    match allocate_and_bind_image_memory(context, image) {
        Ok(memory) => Ok((image, memory)),
        Err(e) => {
            unsafe { context.device().destroy_image(image, None) };
            Err(e)
        }
    }
}

/// A view over a video picture image, for use as a decode/encode picture
/// resource.
///
/// `image_usage` is the usage the image was created with. The view narrows it
/// to everything except `SAMPLED`, which matters when the image *is*
/// sampleable: a view of a multi-planar YCbCr format that includes `SAMPLED`
/// must carry a `VkSamplerYcbcrConversion`, and a video picture resource must
/// not have one. Declaring the narrower usage says what this view is actually
/// for and keeps both requirements satisfiable at once.
fn create_video_image_view(
    context: &VideoContext,
    image: vk::Image,
    format: vk::Format,
    layer: u32,
    image_usage: vk::ImageUsageFlags,
) -> Result<vk::ImageView> {
    let mut view_usage =
        vk::ImageViewUsageCreateInfo::default().usage(image_usage & !vk::ImageUsageFlags::SAMPLED);
    let view_create_info = vk::ImageViewCreateInfo::default()
        .push(&mut view_usage)
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .components(vk::ComponentMapping::default())
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: layer,
            layer_count: 1,
        });

    unsafe { context.device().create_image_view(&view_create_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(format!("image view creation: {}", e)))
}

/// The single-plane formats a multi-planar picture format's planes can be
/// viewed as, per the format compatibility table in the Vulkan spec.
///
/// Naming these in `VkImageFormatListCreateInfo` alongside `MUTABLE_FORMAT` is
/// what lets a consumer read luma and chroma as two ordinary textures, and what
/// keeps a driver from having to assume the image could become anything.
pub(crate) fn plane_view_formats(picture_format: vk::Format) -> Vec<vk::Format> {
    match picture_format {
        vk::Format::G8_B8R8_2PLANE_420_UNORM | vk::Format::G8_B8R8_2PLANE_444_UNORM => {
            vec![vk::Format::R8_UNORM, vk::Format::R8G8_UNORM]
        }
        vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
        | vk::Format::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16 => vec![
            vk::Format::R10X6_UNORM_PACK16,
            vk::Format::R10X6G10X6_UNORM_2PACK16,
        ],
        _ => Vec::new(),
    }
}

/// What a driver will accept for one picture format under a given profile and
/// usage.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VideoFormatSupport {
    pub format: vk::Format,
    /// Image creation flags this format allows here. `MUTABLE_FORMAT` is the
    /// interesting one: with it, a consumer can view the planes of a
    /// multi-planar picture separately, which is the only way to read one
    /// without a sampler-YCbCr conversion.
    pub create_flags: vk::ImageCreateFlags,
}

pub(crate) fn query_supported_video_formats(
    context: &VideoContext,
    profile_info: &vk::VideoProfileInfoKHR,
    image_usage: vk::ImageUsageFlags,
) -> Result<Vec<VideoFormatSupport>> {
    let video_queue_fn = ash::khr::video_queue::Instance::load(context.entry(), context.instance());

    // Vulkan expects a profile list in the pNext chain.
    let profiles = [*profile_info];
    let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);

    let format_info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
        .image_usage(image_usage)
        .push(&mut profile_list);

    let physical_device = context.physical_device();
    let mut count = 0u32;
    let result = unsafe {
        (video_queue_fn
            .fp()
            .get_physical_device_video_format_properties_khr)(
            physical_device,
            &format_info,
            &mut count,
            ptr::null_mut(),
        )
    };

    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::NoSuitableDevice(format!(
            "Failed to query video format properties for usage {:?}: {:?}",
            image_usage, result
        )));
    }

    if count == 0 {
        return Ok(Vec::new());
    }

    let mut props = vec![vk::VideoFormatPropertiesKHR::default(); count as usize];
    let result = unsafe {
        (video_queue_fn
            .fp()
            .get_physical_device_video_format_properties_khr)(
            physical_device,
            &format_info,
            &mut count,
            props.as_mut_ptr(),
        )
    };

    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::NoSuitableDevice(format!(
            "Failed to enumerate video format properties for usage {:?}: {:?}",
            image_usage, result
        )));
    }

    props.truncate(count as usize);
    Ok(props
        .into_iter()
        .map(|p| VideoFormatSupport {
            format: p.format,
            create_flags: p.image_create_flags,
        })
        .collect())
}

/// Get the Vulkan format for a given pixel format and bit depth.
///
/// Supports YUV420 and YUV444 in 8-bit and 10-bit.
/// For YUV444, uses 2-plane (semi-planar) formats from VK_EXT_ycbcr_2plane_444_formats
/// which are supported by NVIDIA hardware for video encoding.
pub(crate) fn get_video_format(pixel_format: PixelFormat, bit_depth: BitDepth) -> vk::Format {
    match (pixel_format, bit_depth) {
        (PixelFormat::Yuv420, BitDepth::Eight) => vk::Format::G8_B8R8_2PLANE_420_UNORM,
        (PixelFormat::Yuv420, BitDepth::Ten) => {
            vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
        }
        // Use 2-plane semi-planar formats for YUV444 (supported by NVIDIA for video encoding).
        (PixelFormat::Yuv444, BitDepth::Eight) => vk::Format::G8_B8R8_2PLANE_444_UNORM,
        (PixelFormat::Yuv444, BitDepth::Ten) => {
            vk::Format::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16
        }
        // TODO: Add support for YUV422 formats.
        _ => unimplemented!(
            "Unsupported pixel format / bit depth combination: {:?} / {:?}",
            pixel_format,
            bit_depth
        ),
    }
}

pub(crate) fn find_memory_type(
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..memory_props.memory_type_count).find(|&i| {
        (type_filter & (1 << i)) != 0
            && memory_props.memory_types[i as usize]
                .property_flags
                .contains(properties)
    })
}

pub(crate) fn create_bitstream_buffer(
    context: &VideoContext,
    size: usize,
    usage: vk::BufferUsageFlags,
    profile_info: &vk::VideoProfileInfoKHR,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let profiles = [*profile_info];
    let mut profile_list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);

    let create_info = vk::BufferCreateInfo::default()
        .size(size as vk::DeviceSize)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push(&mut profile_list);

    let buffer = unsafe { context.device().create_buffer(&create_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(format!("buffer creation: {}", e)))?;

    let mem_requirements = unsafe { context.device().get_buffer_memory_requirements(buffer) };

    let memory_type_index = find_memory_type(
        context.memory_properties(),
        mem_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| {
        PixelForgeError::MemoryAllocation("No suitable memory type for buffer".to_string())
    })?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type_index);

    let memory = unsafe { context.device().allocate_memory(&alloc_info, None) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

    unsafe { context.device().bind_buffer_memory(buffer, memory, 0) }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

    Ok((buffer, memory))
}

pub(crate) fn create_timeline_semaphore(context: &VideoContext) -> Result<vk::Semaphore> {
    let mut type_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let create_info = vk::SemaphoreCreateInfo::default().push(&mut type_info);

    unsafe { context.device().create_semaphore(&create_info, None) }
        .map_err(|e| PixelForgeError::Synchronization(e.to_string()))
}

/// Create an image for video encoding (input or DPB).
///
/// This creates a VkImage suitable for use with a video encoder.
/// For DPB images, the usage is VIDEO_ENCODE_DPB_KHR.
/// For input images, the usage is VIDEO_ENCODE_SRC_KHR | TRANSFER_DST.
///
/// # Arguments
/// * `context` - The Vulkan video context
/// * `params` - Parameters to use for the video image
/// * `profile_info` - Video profile info for the encoder session
pub(crate) fn create_video_image(
    context: &VideoContext,
    params: &VideoImageParams,
    profile_info: &vk::VideoProfileInfoKHR,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let (image, memory) = create_video_image_raw(context, params, 1, profile_info)?;
    match create_video_image_view(context, image, params.format, 0, params.usage) {
        Ok(view) => Ok((image, memory, view)),
        Err(e) => {
            unsafe {
                context.device().destroy_image(image, None);
                context.device().free_memory(memory, None);
            }
            Err(e)
        }
    }
}

/// Allocate and bind memory for a video session.
///
/// Returns the allocated device memory handles.
pub(crate) fn allocate_session_memory(
    context: &VideoContext,
    session: vk::VideoSessionKHR,
    video_queue_fn: &ash::khr::video_queue::Device,
) -> Result<Vec<vk::DeviceMemory>> {
    // Query memory requirements count.
    let mut memory_requirements_count = 0u32;
    let result = unsafe {
        (video_queue_fn
            .fp()
            .get_video_session_memory_requirements_khr)(
            context.device().handle(),
            session,
            &mut memory_requirements_count,
            ptr::null_mut(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::MemoryAllocation(format!("{:?}", result)));
    }

    // Query actual requirements.
    let mut memory_requirements =
        vec![vk::VideoSessionMemoryRequirementsKHR::default(); memory_requirements_count as usize];
    let result = unsafe {
        (video_queue_fn
            .fp()
            .get_video_session_memory_requirements_khr)(
            context.device().handle(),
            session,
            &mut memory_requirements_count,
            memory_requirements.as_mut_ptr(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::MemoryAllocation(format!("{:?}", result)));
    }

    // Allocate and bind memory for each requirement.
    let mut session_memory = Vec::new();
    let mut bind_infos = Vec::new();

    for req in &memory_requirements {
        let memory_type_index = find_memory_type(
            context.memory_properties(),
            req.memory_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .or_else(|| {
            find_memory_type(
                context.memory_properties(),
                req.memory_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::empty(),
            )
        })
        .ok_or_else(|| {
            PixelForgeError::MemoryAllocation(format!(
                "No suitable memory type for video session (type_bits: 0x{:x})",
                req.memory_requirements.memory_type_bits
            ))
        })?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(req.memory_requirements.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { context.device().allocate_memory(&alloc_info, None) }
            .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

        bind_infos.push(
            vk::BindVideoSessionMemoryInfoKHR::default()
                .memory_bind_index(req.memory_bind_index)
                .memory(memory)
                .memory_offset(0)
                .memory_size(req.memory_requirements.size),
        );

        session_memory.push(memory);
    }

    // Bind all memory to the session.
    let result = unsafe {
        (video_queue_fn.fp().bind_video_session_memory_khr)(
            context.device().handle(),
            session,
            bind_infos.len() as u32,
            bind_infos.as_ptr(),
        )
    };
    if result != vk::Result::SUCCESS {
        return Err(PixelForgeError::MemoryAllocation(format!("{:?}", result)));
    }

    Ok(session_memory)
}

/// Map a bitstream buffer for persistent access.
pub(crate) fn map_bitstream_buffer(
    context: &VideoContext,
    memory: vk::DeviceMemory,
    size: usize,
) -> Result<*mut u8> {
    let ptr = unsafe {
        context.device().map_memory(
            memory,
            0,
            size as vk::DeviceSize,
            vk::MemoryMapFlags::empty(),
        )
    }
    .map_err(|e| {
        PixelForgeError::MemoryAllocation(format!("Failed to map bitstream buffer: {}", e))
    })? as *mut u8;

    Ok(ptr)
}

/// Create the decoded picture buffer images for a video session.
///
/// Two layouts are possible, selected by `use_layered`:
/// - **layered**: a single image with `count` array layers (required when the
///   device does not report `SEPARATE_REFERENCE_IMAGES`), with one view per layer;
/// - **separate**: `count` independent single-layer images, one view each.
///
/// Either way the returned view vector is indexed by DPB slot, so callers need
/// not care which layout was used. `usage` and `sharing_families` differ
/// between encode and decode (decode DPB images may also serve as the decode
/// output and be copied from), so they are supplied by the caller.
pub(crate) fn create_dpb_images(
    context: &VideoContext,
    params: &VideoImageParams,
    profile_info: &vk::VideoProfileInfoKHR,
    count: usize,
    use_layered: bool,
) -> Result<(Vec<vk::Image>, Vec<vk::DeviceMemory>, Vec<vk::ImageView>)> {
    if !use_layered {
        let mut images = Vec::with_capacity(count);
        let mut memories = Vec::with_capacity(count);
        let mut views = Vec::with_capacity(count);
        for _ in 0..count {
            match create_video_image(context, params, profile_info) {
                Ok((image, memory, view)) => {
                    images.push(image);
                    memories.push(memory);
                    views.push(view);
                }
                Err(e) => {
                    unsafe {
                        for &v in &views {
                            context.device().destroy_image_view(v, None);
                        }
                        for &img in &images {
                            context.device().destroy_image(img, None);
                        }
                        for &m in &memories {
                            context.device().free_memory(m, None);
                        }
                    }
                    return Err(e);
                }
            }
        }
        return Ok((images, memories, views));
    }

    let (image, memory) = create_video_image_raw(context, params, count as u32, profile_info)?;

    let mut views = Vec::with_capacity(count);
    for layer in 0..count as u32 {
        match create_video_image_view(context, image, params.format, layer, params.usage) {
            Ok(view) => views.push(view),
            Err(e) => {
                unsafe {
                    for &v in &views {
                        context.device().destroy_image_view(v, None);
                    }
                    context.device().destroy_image(image, None);
                    context.device().free_memory(memory, None);
                }
                return Err(e);
            }
        }
    }

    Ok((vec![image], vec![memory], views))
}

/// A command pool for `family`, with per-buffer reset.
///
/// `label` names the pool in the error message, since a decoder holds several
/// for different queue families.
pub(crate) fn create_command_pool(
    context: &VideoContext,
    family: u32,
    label: &str,
) -> Result<vk::CommandPool> {
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    unsafe { context.device().create_command_pool(&pool_info, None) }
        .map_err(|e| PixelForgeError::ResourceCreation(format!("{} command pool: {}", label, e)))
}

/// Allocate `count` primary command buffers from `pool`.
pub(crate) fn allocate_command_buffers(
    context: &VideoContext,
    pool: vk::CommandPool,
    count: u32,
) -> Result<Vec<vk::CommandBuffer>> {
    let info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(count);
    unsafe { context.device().allocate_command_buffers(&info) }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))
}

/// A fence, created signaled when it will be waited on before its first submit.
pub(crate) fn create_fence(context: &VideoContext, signaled: bool) -> Result<vk::Fence> {
    let mut info = vk::FenceCreateInfo::default();
    if signaled {
        info = info.flags(vk::FenceCreateFlags::SIGNALED);
    }
    unsafe { context.device().create_fence(&info, None) }
        .map_err(|e| PixelForgeError::Synchronization(e.to_string()))
}

/// A chain of submissions ordered by one timeline semaphore.
///
/// Video submissions that share a session and DPB must execute in order even
/// when the CPU runs ahead of the GPU. Each submission waits on the value the
/// previous one signals and signals its own, which this tracks: [`Self::wait`]
/// for what to wait on, [`Self::pending_signal`] for what to signal, and
/// [`Self::commit`] once the submit has actually succeeded.
///
/// Committing separately matters: advancing the chain for a submission that
/// failed to submit would leave every later submission waiting on a value
/// nothing will ever signal.
pub(crate) struct TimelineChain {
    semaphore: vk::Semaphore,
    /// Value the next submission will signal.
    next: u64,
    /// Value the most recent committed submission signals; 0 = none yet.
    last: u64,
}

impl TimelineChain {
    pub(crate) fn new(context: &VideoContext) -> Result<Self> {
        Ok(Self {
            semaphore: create_timeline_semaphore(context)?,
            next: 1,
            last: 0,
        })
    }

    /// What a new submission must wait on to run after the previous one, or
    /// `None` when nothing has been submitted yet.
    pub(crate) fn wait(&self) -> Option<(vk::Semaphore, u64)> {
        (self.last > 0).then_some((self.semaphore, self.last))
    }

    /// The value most recently committed. A timeline wait on 0 is satisfied
    /// immediately, so callers that always wait can use this directly.
    pub(crate) fn last_value(&self) -> u64 {
        self.last
    }

    pub(crate) fn semaphore(&self) -> vk::Semaphore {
        self.semaphore
    }

    /// What a new submission should signal. Not recorded until [`Self::commit`].
    pub(crate) fn pending_signal(&self) -> (vk::Semaphore, u64) {
        (self.semaphore, self.next)
    }

    /// Record the pending signal as committed, after a successful submit.
    pub(crate) fn commit(&mut self) {
        self.last = self.next;
        self.next += 1;
    }

    /// # Safety
    ///
    /// Every submission that waits on or signals this semaphore must have
    /// completed.
    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        unsafe { device.destroy_semaphore(self.semaphore, None) };
    }
}

/// Cross-thread per-slot readiness for a pipelined encode or decode.
///
/// A slot is "busy" from the moment its work is submitted until the completion
/// thread has finished with it (read the bitstream back, or observed the decode
/// fence). The submitting thread waits for a slot to be free before recording
/// over its command buffer and staging memory, which is also what covers the
/// write-after-read hazard on those resources.
pub(crate) struct SlotSync {
    busy: std::sync::Mutex<Vec<bool>>,
    cv: std::sync::Condvar,
}

impl SlotSync {
    pub(crate) fn new(slot_count: usize) -> Self {
        Self {
            busy: std::sync::Mutex::new(vec![false; slot_count]),
            cv: std::sync::Condvar::new(),
        }
    }

    /// Block until slot `index` is free (its previous submission is finished).
    pub(crate) fn wait_free(&self, index: usize) {
        let mut busy = self.busy.lock().unwrap();
        while busy[index] {
            busy = self.cv.wait(busy).unwrap();
        }
    }

    /// Block until every slot is free (nothing in flight).
    pub(crate) fn wait_all_free(&self) {
        let mut busy = self.busy.lock().unwrap();
        while busy.iter().any(|b| *b) {
            busy = self.cv.wait(busy).unwrap();
        }
    }

    /// Mark a slot busy at submit time. No notify: nobody waits to *enter* busy.
    pub(crate) fn set_busy(&self, index: usize) {
        self.busy.lock().unwrap()[index] = true;
    }

    /// Mark a slot free once its submission is finished; wake any waiters.
    pub(crate) fn set_free(&self, index: usize) {
        self.busy.lock().unwrap()[index] = false;
        self.cv.notify_all();
    }
}
