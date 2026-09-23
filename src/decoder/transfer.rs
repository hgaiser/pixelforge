//! Copying a decoded picture out of the DPB, for the drivers that need it.
//!
//! Only reached when pictures cannot be pinned in place: without unified image
//! layouts a handed-out DPB image could not be read safely, and with a distinct
//! decode output image the next picture overwrites it. Either way the picture
//! is copied into a private image the caller owns outright.
//!
//! The copy runs on the transfer queue rather than the decode queue, because a
//! dedicated video decode queue family need not advertise `TRANSFER_BIT` (it
//! does not on RADV), so a copy recorded there would be invalid.

use ash::vk;

use crate::decoder::common::DecoderCommon;
use crate::decoder::frames::DecodedPicture;
use crate::error::{PixelForgeError, Result};

impl DecoderCommon {
    /// Copy a freshly decoded picture into a pool image, so it survives past the
    /// next decode submission. This is what lets a [`DecodedFrame`] outlive the
    /// DPB slot the picture was decoded into.
    ///
    /// Records only; [`Self::submit_copy`] submits it. The copy runs on the
    /// transfer queue, because a dedicated video decode queue need not support
    /// transfer operations, and waits on the decode's timeline value rather than
    /// a fence. A semaphore wait makes the decode's writes available and
    /// visible, so `NONE` remains a correct source scope, and the DPB image is
    /// created with concurrent sharing between the two families so no ownership
    /// transfer is needed.
    ///
    /// The source's layout is restored afterward so a DPB image stays usable as
    /// a reference. The destination is left in `TRANSFER_DST_OPTIMAL`, which is
    /// what the frame reports as its layout, so a consumer knows to transition
    /// it before reading.
    pub fn record_picture_copy(&self, frame: &DecodedPicture, dst_image: vk::Image) -> Result<()> {
        let command_buffer = self.pipeline.current().transfer_command_buffer;
        let base_layer = frame.array_layer;
        let device = self.context.device().clone();
        let sync2 = self.context.sync2();
        let aspects = vk::ImageAspectFlags::PLANE_0 | vk::ImageAspectFlags::PLANE_1;
        let range = |image: vk::Image, layer: u32| vk::ImageMemoryBarrier2 {
            image,
            ..vk::ImageMemoryBarrier2::default()
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: aspects,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: layer,
                    layer_count: 1,
                })
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        };

        unsafe {
            device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
            device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| PixelForgeError::CommandBuffer(e.to_string()))?;
        }

        let to_transfer = [
            vk::ImageMemoryBarrier2 {
                src_stage_mask: vk::PipelineStageFlags2::NONE,
                src_access_mask: vk::AccessFlags2::NONE,
                dst_stage_mask: vk::PipelineStageFlags2::COPY,
                dst_access_mask: vk::AccessFlags2::TRANSFER_READ,
                old_layout: frame.layout,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                ..range(frame.image, base_layer)
            },
            vk::ImageMemoryBarrier2 {
                src_stage_mask: vk::PipelineStageFlags2::NONE,
                src_access_mask: vk::AccessFlags2::NONE,
                dst_stage_mask: vk::PipelineStageFlags2::COPY,
                dst_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                ..range(dst_image, 0)
            },
        ];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&to_transfer);
        unsafe { sync2.cmd_pipeline_barrier2(command_buffer, &dep) };

        let plane = |aspect: vk::ImageAspectFlags| vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let (cdiv_hor, cdiv_vert) = frame.pixel_format.chroma_div();
        let regions = [
            vk::ImageCopy::default()
                .src_subresource(vk::ImageSubresourceLayers {
                    base_array_layer: base_layer,
                    ..plane(vk::ImageAspectFlags::PLANE_0)
                })
                .dst_subresource(plane(vk::ImageAspectFlags::PLANE_0))
                .extent(vk::Extent3D {
                    width: frame.coded_width,
                    height: frame.coded_height,
                    depth: 1,
                }),
            vk::ImageCopy::default()
                .src_subresource(vk::ImageSubresourceLayers {
                    base_array_layer: base_layer,
                    ..plane(vk::ImageAspectFlags::PLANE_1)
                })
                .dst_subresource(plane(vk::ImageAspectFlags::PLANE_1))
                .extent(vk::Extent3D {
                    width: frame.coded_width / cdiv_hor,
                    height: frame.coded_height / cdiv_vert,
                    depth: 1,
                }),
        ];
        // `vkCmdCopyImage` rather than its `2` form: the copy needs nothing the
        // newer entry point adds, and this one exists on every device, where
        // `vkCmdCopyImage2` needs Vulkan 1.3 or `VK_KHR_copy_commands2`.
        unsafe {
            device.cmd_copy_image(
                command_buffer,
                frame.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            )
        };

        // Restore the source layout so a DPB image stays a valid reference.
        let restore = [vk::ImageMemoryBarrier2 {
            src_stage_mask: vk::PipelineStageFlags2::COPY,
            src_access_mask: vk::AccessFlags2::TRANSFER_READ,
            dst_stage_mask: vk::PipelineStageFlags2::NONE,
            dst_access_mask: vk::AccessFlags2::NONE,
            old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            new_layout: frame.layout,
            ..range(frame.image, base_layer)
        }];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&restore);
        unsafe { sync2.cmd_pipeline_barrier2(command_buffer, &dep) };

        Ok(())
    }

    /// Submit the recorded reorder copy without waiting for it.
    pub fn submit_copy(&mut self) -> Result<()> {
        let device = self.context.device().clone();
        let queue = self.transfer_queue;
        self.pipeline
            .submit_copy(&device, self.context.sync2(), queue)
    }
}
