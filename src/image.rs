//! Image utilities for creating and uploading video frames.
//!
//! This module provides [`InputImage`], a helper for uploading YUV data from the CPU
//! to the GPU. The image it creates is a transfer-only staging image (not a Vulkan Video
//! image) and must be copied into an encoder's input image before encoding.
//! Use [`InputImage::upload_yuv420`] to upload to the internal image, then submit
//! [`InputImage::source`] with [`Frame::image`](crate::Frame::image).
//! Alternatively, use [`InputImage::upload_yuv420_to`] / [`InputImage::upload_yuv444_to`]
//! to upload directly into an encoder's [`InputSlot`] (from
//! [`Encoder::input_slot`](crate::encoder::Encoder::input_slot)).

use crate::encoder::{BitDepth, Codec, Handoff, InputSlot, PixelFormat, SourceImage};
use crate::error::{PixelForgeError, Result};
use crate::vulkan::VideoContext;
use ash::vk;

/// A GPU image for staging YUV frame data before encoding.
///
/// This struct owns a transfer-only Vulkan image (no video profile, no
/// `VIDEO_ENCODE_SRC_KHR` usage) with NV12/P010 format (YUV420) or 2-plane
/// semi-planar YUV444 format. It provides methods to upload YUV data from the
/// CPU, which can then be copied into an encoder's input image for encoding.
///
/// The image is **not** directly usable as a Vulkan Video encode source. To
/// encode, either:
/// - Upload to this image and submit [`Self::source`] with
///   [`Frame::image`](crate::Frame::image), which copies it into the encoder's
///   input image, or
/// - Use [`upload_yuv420_to`](Self::upload_yuv420_to) /
///   [`upload_yuv444_to`](Self::upload_yuv444_to) to upload directly into the
///   encoder's input image, through an [`InputSlot`].
pub struct InputImage {
    /// Keeps track of the frames that read `image`.
    source: SourceImage,
    context: VideoContext,
    image: vk::Image,
    /// Current layout of `self.image`.
    image_layout: vk::ImageLayout,
    memory: vk::DeviceMemory,
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    staging_size: usize,
    /// Kept for automatic cleanup via Drop.
    #[allow(dead_code)]
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    width: u32,
    height: u32,
    bit_depth: BitDepth,
    pixel_format: PixelFormat,
}

impl InputImage {
    /// Create a new staging image for uploading YUV frame data.
    ///
    /// Creates a transfer-only image suitable for staging YUV data before
    /// copying it into an encoder's input image. The image has no video
    /// profile and uses `TRANSFER_DST | TRANSFER_SRC` usage flags.
    ///
    /// # Arguments
    /// * `context` - The Vulkan video context
    /// * `codec` - Unused. Kept for API compatibility.
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `bit_depth` - Bit depth for the image (8-bit or 10-bit)
    /// * `pixel_format` - Pixel format (YUV420 or YUV444)
    pub fn new(
        context: VideoContext,
        _codec: Codec,
        width: u32,
        height: u32,
        bit_depth: BitDepth,
        pixel_format: PixelFormat,
    ) -> Result<Self> {
        let device = context.device();

        // Select format based on pixel format and bit depth.
        // Use 2-plane semi-planar formats for both YUV420 and YUV444.
        let format = match (pixel_format, bit_depth) {
            (PixelFormat::Yuv420, BitDepth::Eight) => vk::Format::G8_B8R8_2PLANE_420_UNORM,
            (PixelFormat::Yuv420, BitDepth::Ten) => {
                vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
            }
            (PixelFormat::Yuv444, BitDepth::Eight) => vk::Format::G8_B8R8_2PLANE_444_UNORM,
            (PixelFormat::Yuv444, BitDepth::Ten) => {
                vk::Format::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16
            }
            _ => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "Unsupported pixel format / bit depth combination: {:?} / {:?}",
                    pixel_format, bit_depth
                )));
            }
        };

        // Create the image.
        // This image is used purely for staging (buffer→image copy on the transfer queue),
        // so it only needs TRANSFER_DST | TRANSFER_SRC and no video profile pNext.
        let image_create_info = vk::ImageCreateInfo::default()
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
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.create_image(&image_create_info, None) }
            .map_err(|e| PixelForgeError::ResourceCreation(format!("image creation: {}", e)))?;

        // Get memory requirements and allocate.
        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };

        let memory_type_index = context
            .find_memory_type(
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .ok_or_else(|| {
                PixelForgeError::MemoryAllocation("No suitable memory type for image".to_string())
            })?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { device.allocate_memory(&alloc_info, None) }
            .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

        unsafe { device.bind_image_memory(image, memory, 0) }
            .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

        // Create staging buffer for uploads.
        // Size depends on pixel format, bit depth, and alignment padding between planes.
        let bytes_per_sample_init = match bit_depth {
            BitDepth::Eight => 1usize,
            BitDepth::Ten => 2,
        };
        let luma_pixels = (width * height) as usize;
        let staging_size = match pixel_format {
            PixelFormat::Yuv444 => {
                // Y plane (aligned to 4 bytes) + UV interleaved plane.
                let y_bytes = luma_pixels * bytes_per_sample_init;
                crate::align4(y_bytes) + luma_pixels * 2 * bytes_per_sample_init
            }
            _ => pixel_format.frame_size(width, height) * bytes_per_sample_init,
        };
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(staging_size as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let staging_buffer = unsafe { device.create_buffer(&buffer_create_info, None) }
            .map_err(|e| PixelForgeError::ResourceCreation(format!("buffer creation: {}", e)))?;

        let staging_mem_requirements =
            unsafe { device.get_buffer_memory_requirements(staging_buffer) };

        let staging_memory_type = context
            .find_memory_type(
                staging_mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .ok_or_else(|| {
                PixelForgeError::MemoryAllocation(
                    "No suitable memory type for staging buffer".to_string(),
                )
            })?;

        let staging_alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(staging_mem_requirements.size)
            .memory_type_index(staging_memory_type);

        let staging_memory = unsafe { device.allocate_memory(&staging_alloc_info, None) }
            .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

        unsafe { device.bind_buffer_memory(staging_buffer, staging_memory, 0) }
            .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

        // Create command pool and buffer for transfers.
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(context.transfer_queue_family())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info) }
            .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
        let command_buffer = command_buffers[0];

        // Create fence for synchronization.
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { device.create_fence(&fence_info, None) }
            .map_err(|e| PixelForgeError::ResourceCreation(format!("fence creation: {}", e)))?;

        let source = SourceImage::new(
            &context,
            image,
            Handoff::Device {
                layout: vk::ImageLayout::GENERAL,
            },
        )?;
        let mut input = Self {
            source,
            context,
            image,
            image_layout: vk::ImageLayout::UNDEFINED,
            memory,
            staging_buffer,
            staging_memory,
            staging_size,
            command_pool,
            command_buffer,
            fence,
            width,
            height,
            bit_depth,
            pixel_format,
        };
        // Start from zeroed planes in GENERAL, so the image is a valid source
        // before the first upload.
        unsafe {
            let device = input.context.device();
            let ptr = device
                .map_memory(
                    staging_memory,
                    0,
                    staging_size as vk::DeviceSize,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;
            std::ptr::write_bytes(ptr as *mut u8, 0, staging_size);
            device.unmap_memory(staging_memory);
        }
        input.copy_staging_to_image()?;
        Ok(input)
    }

    /// Upload YUV420 (I420) data to this image.
    ///
    /// The data is expected in I420 format (Y plane, then U plane, then V plane).
    /// This method converts it to NV12 (8-bit) or P010 (10-bit) format and uploads
    /// it to the GPU.
    ///
    /// For 10-bit images, 8-bit input data is expanded to 10-bit by left-shifting
    /// each sample by 6 bits (filling in the 6 lower bits with the high bits).
    pub fn upload_yuv420(&mut self, yuv_data: &[u8]) -> Result<()> {
        let expected_size = (self.width * self.height * 3 / 2) as usize;
        if yuv_data.len() < expected_size {
            return Err(PixelForgeError::InvalidInput(format!(
                "YUV data too small: expected {} bytes, got {}",
                expected_size,
                yuv_data.len()
            )));
        }

        let device = self.context.device();
        let width = self.width as usize;
        let height = self.height as usize;
        let y_size = width * height;

        // Map staging buffer and convert I420 to NV12/P010.
        let data_ptr = unsafe {
            device.map_memory(
                self.staging_memory,
                0,
                self.staging_size as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

        unsafe {
            match self.bit_depth {
                BitDepth::Eight => {
                    // 8-bit NV12 format
                    let dst =
                        std::slice::from_raw_parts_mut(data_ptr as *mut u8, self.staging_size);

                    // Copy Y plane directly.
                    dst[..y_size].copy_from_slice(&yuv_data[..y_size]);

                    // Interleave U and V planes to create NV12 UV plane.
                    let u_plane = &yuv_data[y_size..y_size + y_size / 4];
                    let v_plane = &yuv_data[y_size + y_size / 4..];
                    for i in 0..y_size / 4 {
                        dst[y_size + i * 2] = u_plane[i];
                        dst[y_size + i * 2 + 1] = v_plane[i];
                    }
                }
                BitDepth::Ten => {
                    // 10-bit P010 format: each sample is stored as 16-bit little-endian
                    // with the 10-bit value in the high bits (bits 6-15), and bits 0-5 as padding.
                    //
                    // To convert 8-bit (0-255) to 10-bit (0-1023):
                    // - Multiply by ~4 to scale the range: (val * 1023) / 255 ≈ val * 4
                    // - This is equivalent to: (val << 2) to get 10-bit value
                    // - Then shift left by 6 to place in P010 format: (val << 2) << 6 = val << 8
                    //
                    // For better precision, we can also fill in the 2 LSBs of the 10-bit value
                    // using the 2 MSBs of the original 8-bit value: (val << 8) | (val >> 6 << 6)
                    // Simplified: val << 8 puts the 8-bit value in bits 8-15, which is correct for P010.
                    let dst =
                        std::slice::from_raw_parts_mut(data_ptr as *mut u16, self.staging_size / 2);

                    // Convert and copy Y plane (expand 8-bit to 10-bit P010).
                    for i in 0..y_size {
                        let val = yuv_data[i] as u16;
                        // Shift left by 8 to place 8-bit value in bits 8-15 (P010 format).
                        // This effectively scales 0-255 to 0-65280 with proper bit alignment.
                        dst[i] = val << 8;
                    }

                    // Interleave U and V planes to create P010 UV plane.
                    let u_plane = &yuv_data[y_size..y_size + y_size / 4];
                    let v_plane = &yuv_data[y_size + y_size / 4..];
                    for i in 0..y_size / 4 {
                        let u_val = u_plane[i] as u16;
                        let v_val = v_plane[i] as u16;
                        dst[y_size + i * 2] = u_val << 8;
                        dst[y_size + i * 2 + 1] = v_val << 8;
                    }
                }
            }

            device.unmap_memory(self.staging_memory);
        }

        // Record and submit copy commands.
        self.copy_staging_to_image()?;

        Ok(())
    }

    /// Upload NV12 data directly to this image.
    ///
    /// The data is expected in NV12 format (Y plane, then interleaved UV plane).
    pub fn upload_nv12(&mut self, nv12_data: &[u8]) -> Result<()> {
        let expected_size = (self.width * self.height * 3 / 2) as usize;
        if nv12_data.len() < expected_size {
            return Err(PixelForgeError::InvalidInput(format!(
                "NV12 data too small: expected {} bytes, got {}",
                expected_size,
                nv12_data.len()
            )));
        }

        let device = self.context.device();

        // Map staging buffer and copy directly.
        let data_ptr = unsafe {
            device.map_memory(
                self.staging_memory,
                0,
                self.staging_size as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

        unsafe {
            let dst = std::slice::from_raw_parts_mut(data_ptr as *mut u8, self.staging_size);
            dst.copy_from_slice(&nv12_data[..self.staging_size]);
            device.unmap_memory(self.staging_memory);
        }

        // Record and submit copy commands.
        self.copy_staging_to_image()?;

        Ok(())
    }

    /// Upload native 10-bit YUV420 (I420 10-bit) data to this image.
    ///
    /// The data is expected in yuv420p10le format:
    /// - Y plane: width * height * 2 bytes (16-bit little-endian samples)
    /// - U plane: (width/2) * (height/2) * 2 bytes (16-bit little-endian samples)
    /// - V plane: (width/2) * (height/2) * 2 bytes (16-bit little-endian samples)
    ///
    /// The 10-bit values are stored in the lower 10 bits of each 16-bit sample.
    /// This method converts them to P010 format (10-bit in upper bits) for the encoder.
    ///
    /// This method is only valid for 10-bit InputImage instances.
    pub fn upload_yuv420_10bit(&mut self, yuv_data: &[u8]) -> Result<()> {
        if self.bit_depth != BitDepth::Ten {
            return Err(PixelForgeError::InvalidInput(
                "upload_yuv420_10bit can only be used with 10-bit InputImage".to_string(),
            ));
        }

        // 10-bit YUV420 size: Y (w*h*2) + U (w*h/4*2) + V (w*h/4*2) = w*h*3 bytes
        let expected_size = (self.width * self.height * 3) as usize;
        if yuv_data.len() < expected_size {
            return Err(PixelForgeError::InvalidInput(format!(
                "YUV420 10-bit data too small: expected {} bytes, got {}",
                expected_size,
                yuv_data.len()
            )));
        }

        let device = self.context.device();
        let width = self.width as usize;
        let height = self.height as usize;
        let y_size = width * height; // Number of Y samples

        // Map staging buffer.
        let data_ptr = unsafe {
            device.map_memory(
                self.staging_memory,
                0,
                self.staging_size as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

        unsafe {
            // Source is yuv420p10le (10-bit in lower bits of 16-bit words)
            // Destination is P010 (10-bit in upper bits of 16-bit words)
            let src =
                std::slice::from_raw_parts(yuv_data.as_ptr() as *const u16, expected_size / 2);
            let dst = std::slice::from_raw_parts_mut(data_ptr as *mut u16, self.staging_size / 2);

            // Convert and copy Y plane (shift 10-bit values to upper bits for P010).
            for i in 0..y_size {
                // yuv420p10le has 10-bit value in bits 0-9, P010 needs it in bits 6-15
                dst[i] = src[i] << 6;
            }

            // Convert and interleave U and V planes to create P010 UV plane.
            // Source U plane starts at y_size, V plane at y_size + y_size/4
            let u_offset = y_size;
            let v_offset = y_size + y_size / 4;
            for i in 0..y_size / 4 {
                let u_val = src[u_offset + i];
                let v_val = src[v_offset + i];
                dst[y_size + i * 2] = u_val << 6;
                dst[y_size + i * 2 + 1] = v_val << 6;
            }

            device.unmap_memory(self.staging_memory);
        }

        // Record and submit copy commands.
        self.copy_staging_to_image()?;

        Ok(())
    }

    /// Upload YUV420 (I420) data to an external image.
    ///
    /// Same as `upload_yuv420()` but copies the data into an encoder's input
    /// image instead of this InputImage's own image, so the encode reads it
    /// with no further copy. Submit the slot with
    /// [`Frame::input`](crate::Frame::input) afterwards.
    pub fn upload_yuv420_to(&mut self, slot: &InputSlot, yuv_data: &[u8]) -> Result<()> {
        if self.pixel_format != PixelFormat::Yuv420 {
            return Err(PixelForgeError::InvalidInput(
                "upload_yuv420_to can only be used with YUV420 InputImage".to_string(),
            ));
        }

        let expected_size = (self.width * self.height * 3 / 2) as usize;
        if yuv_data.len() < expected_size {
            return Err(PixelForgeError::InvalidInput(format!(
                "YUV data too small: expected {} bytes, got {}",
                expected_size,
                yuv_data.len()
            )));
        }

        let device = self.context.device();
        let width = self.width as usize;
        let height = self.height as usize;
        let y_size = width * height;

        // Map staging buffer and convert I420 to NV12/P010.
        let data_ptr = unsafe {
            device.map_memory(
                self.staging_memory,
                0,
                self.staging_size as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

        unsafe {
            match self.bit_depth {
                BitDepth::Eight => {
                    let dst =
                        std::slice::from_raw_parts_mut(data_ptr as *mut u8, self.staging_size);
                    dst[..y_size].copy_from_slice(&yuv_data[..y_size]);
                    let u_plane = &yuv_data[y_size..y_size + y_size / 4];
                    let v_plane = &yuv_data[y_size + y_size / 4..];
                    for i in 0..y_size / 4 {
                        dst[y_size + i * 2] = u_plane[i];
                        dst[y_size + i * 2 + 1] = v_plane[i];
                    }
                }
                BitDepth::Ten => {
                    let dst =
                        std::slice::from_raw_parts_mut(data_ptr as *mut u16, self.staging_size / 2);
                    for i in 0..y_size {
                        let val = yuv_data[i] as u16;
                        dst[i] = val << 8;
                    }
                    let u_plane = &yuv_data[y_size..y_size + y_size / 4];
                    let v_plane = &yuv_data[y_size + y_size / 4..];
                    for i in 0..y_size / 4 {
                        let u_val = u_plane[i] as u16;
                        let v_val = v_plane[i] as u16;
                        dst[y_size + i * 2] = u_val << 8;
                        dst[y_size + i * 2 + 1] = v_val << 8;
                    }
                }
            }

            device.unmap_memory(self.staging_memory);
        }

        // Record and submit copy commands to the target image.
        self.copy_staging_to_image_internal(
            slot.image(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
        )?;

        Ok(())
    }

    /// Upload YUV444 (planar) data into an encoder's input image, like
    /// [`Self::upload_yuv420_to`].
    pub fn upload_yuv444_to(&mut self, slot: &InputSlot, yuv_data: &[u8]) -> Result<()> {
        if self.pixel_format != PixelFormat::Yuv444 {
            return Err(PixelForgeError::InvalidInput(
                "upload_yuv444_to can only be used with YUV444 InputImage".to_string(),
            ));
        }

        let plane_size = (self.width * self.height) as usize;
        let expected_size = plane_size * 3; // Y + U + V at full resolution
        if yuv_data.len() < expected_size {
            return Err(PixelForgeError::InvalidInput(format!(
                "YUV444 data too small: expected {} bytes, got {}",
                expected_size,
                yuv_data.len()
            )));
        }

        let device = self.context.device();

        // Map staging buffer.
        let data_ptr = unsafe {
            device.map_memory(
                self.staging_memory,
                0,
                self.staging_size as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(|e| PixelForgeError::MemoryAllocation(e.to_string()))?;

        unsafe {
            match self.bit_depth {
                BitDepth::Eight => {
                    // 8-bit semi-planar YUV444: Y plane, then interleaved UV plane.
                    let dst =
                        std::slice::from_raw_parts_mut(data_ptr as *mut u8, self.staging_size);

                    // Copy Y plane directly.
                    dst[..plane_size].copy_from_slice(&yuv_data[..plane_size]);

                    // Align UV plane offset to 4 bytes for VkBufferImageCopy compliance.
                    let uv_offset = crate::align4(plane_size);

                    // Interleave U and V planes for semi-planar format.
                    let u_plane = &yuv_data[plane_size..plane_size * 2];
                    let v_plane = &yuv_data[plane_size * 2..plane_size * 3];
                    for i in 0..plane_size {
                        dst[uv_offset + i * 2] = u_plane[i];
                        dst[uv_offset + i * 2 + 1] = v_plane[i];
                    }
                }
                BitDepth::Ten => {
                    // 10-bit semi-planar YUV444: Expand 8-bit to 16-bit and interleave UV.
                    let dst =
                        std::slice::from_raw_parts_mut(data_ptr as *mut u16, self.staging_size / 2);

                    // Convert Y plane.
                    for i in 0..plane_size {
                        dst[i] = (yuv_data[i] as u16) << 8;
                    }

                    // Align UV plane offset to 4 bytes for VkBufferImageCopy compliance.
                    let uv_offset_u16 = crate::align4(plane_size * 2) / 2;

                    // Interleave and convert U and V planes.
                    let u_plane = &yuv_data[plane_size..plane_size * 2];
                    let v_plane = &yuv_data[plane_size * 2..plane_size * 3];
                    for i in 0..plane_size {
                        dst[uv_offset_u16 + i * 2] = (u_plane[i] as u16) << 8;
                        dst[uv_offset_u16 + i * 2 + 1] = (v_plane[i] as u16) << 8;
                    }
                }
            }

            device.unmap_memory(self.staging_memory);
        }

        // Record and submit copy commands.
        self.copy_staging_to_image_internal(
            slot.image(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
        )?;

        Ok(())
    }

    fn copy_staging_to_image(&mut self) -> Result<()> {
        // A frame submitted before may still be reading the image.
        self.source.wait_released()?;
        let old_layout = self.image_layout;
        self.copy_staging_to_image_internal(self.image, old_layout, vk::ImageLayout::GENERAL)?;
        self.image_layout = vk::ImageLayout::GENERAL;
        Ok(())
    }

    fn copy_staging_to_image_internal(
        &mut self,
        target_image: vk::Image,
        old_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) -> Result<()> {
        let device = self.context.device();
        let width = self.width;
        let height = self.height;

        // Calculate plane sizes based on bit depth.
        // For 8-bit: 1 byte per sample
        // For 10-bit: 2 bytes per sample (16-bit per sample)
        let bytes_per_sample = match self.bit_depth {
            BitDepth::Eight => 1,
            BitDepth::Ten => 2,
        };
        // For YUV444, align Y plane size to 4 bytes so the UV plane buffer offset
        // meets VkBufferImageCopy::bufferOffset alignment requirements.
        // YUV420 dimensions are always even (required for 4:2:0), so alignment is
        // naturally satisfied and must not pad to stay consistent with upload functions.
        let y_plane_bytes = (width * height) as usize * bytes_per_sample;
        let y_plane_size_bytes = match self.pixel_format {
            PixelFormat::Yuv444 => crate::align4(y_plane_bytes) as vk::DeviceSize,
            _ => y_plane_bytes as vk::DeviceSize,
        };

        unsafe {
            device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
        }
        .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(self.command_buffer, &begin_info) }
            .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

        // Transition image to transfer destination.
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(target_image)
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
                self.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        // Build copy regions based on pixel format.
        let copy_regions = match self.pixel_format {
            PixelFormat::Yuv420 => {
                // NV12/P010: Y plane (PLANE_0) + interleaved UV plane (PLANE_1)
                vec![
                    // Y plane.
                    vk::BufferImageCopy::default()
                        .buffer_offset(0)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_0,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .image_extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        }),
                    // UV plane (interleaved, half resolution).
                    vk::BufferImageCopy::default()
                        .buffer_offset(y_plane_size_bytes)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_1,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .image_extent(vk::Extent3D {
                            width: width / 2,
                            height: height / 2,
                            depth: 1,
                        }),
                ]
            }
            PixelFormat::Yuv444 => {
                // 2-plane semi-planar YUV444: Y (PLANE_0), UV interleaved (PLANE_1) at full resolution.
                vec![
                    // Y plane.
                    vk::BufferImageCopy::default()
                        .buffer_offset(0)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_0,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .image_extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        }),
                    // UV plane (interleaved, full resolution for 444).
                    vk::BufferImageCopy::default()
                        .buffer_offset(y_plane_size_bytes)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::PLANE_1,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                        .image_extent(vk::Extent3D {
                            width,
                            height,
                            depth: 1,
                        }),
                ]
            }
            _ => {
                return Err(PixelForgeError::InvalidInput(format!(
                    "Unsupported pixel format: {:?}",
                    self.pixel_format
                )));
            }
        };

        unsafe {
            device.cmd_copy_buffer_to_image(
                self.command_buffer,
                self.staging_buffer,
                target_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy_regions,
            );
        }

        // Transition image to final layout (ready for its intended use)
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(final_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(target_image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ);

        unsafe {
            device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        unsafe { device.end_command_buffer(self.command_buffer) }
            .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;

        // Submit and wait.
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.command_buffer));

        unsafe { device.queue_submit(self.context.transfer_queue(), &[submit_info], self.fence) }
            .map_err(|e| {
            PixelForgeError::vk(e, |e| {
                PixelForgeError::CommandBuffer(format!("Queue submit failed: {e}"))
            })
        })?;

        unsafe { device.wait_for_fences(&[self.fence], true, u64::MAX) }
            .map_err(|e| PixelForgeError::vk(e, PixelForgeError::Synchronization))?;

        unsafe { device.reset_fences(&[self.fence]) }
            .map_err(|e| PixelForgeError::Synchronization(e.to_string()))?;

        Ok(())
    }

    /// Get the underlying Vulkan image handle.
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// The image as a source to encode from, see
    /// [`Frame::image`](crate::Frame::image). The uploads wait until the
    /// frames that read it are done with it.
    pub fn source(&self) -> &SourceImage {
        &self.source
    }

    /// Get the image dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl Drop for InputImage {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe {
            // Wait for all GPU work to finish before destroying resources.
            let _ = device.device_wait_idle();
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_buffer(self.staging_buffer, None);
            device.free_memory(self.staging_memory, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}
