//! How one frame's source reaches the encoder, inside the crate.
//!
//! [`Encoder::submit`](crate::Encoder::submit) turns a public
//! [`Frame`](crate::Frame) into a [`FrameInput`] for the codec encoder: the
//! slot's own input image, an image to copy into it, or an image the encode
//! reads directly. The submission that reads a source also does its queue
//! family ownership transfers, which [`SourceOwnership`] describes.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use ash::vk;

use crate::error::Result;
use crate::sync::TimelinePoint;
use crate::vulkan::VideoContext;

/// The layout a source is in, and the queue family transfers around the one
/// submission that reads it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SourceOwnership {
    /// Layout the source is in when the reading submission starts.
    pub old_layout: vk::ImageLayout,
    /// Queue family to acquire the source from, if another one owns it
    /// (`EXTERNAL` or `FOREIGN_EXT` for memory shared with another API or
    /// process).
    pub acquire_from: Option<u32>,
    /// Queue family to release the source to once it has been read.
    pub release_to: Option<u32>,
    /// Layout the source is left in once it has been read.
    pub final_layout: vk::ImageLayout,
}

impl SourceOwnership {
    /// The barrier that brings the source into `new_layout` for `family`,
    /// acquiring it first when another family owns it.
    pub(crate) fn acquire_barrier(
        &self,
        image: vk::Image,
        family: u32,
        new_layout: vk::ImageLayout,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
    ) -> vk::ImageMemoryBarrier2<'static> {
        let (src_family, dst_family) = match self.acquire_from {
            Some(from) => (from, family),
            None => (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
        };
        // An acquire has no access of this queue's to make available, and
        // anything before it on this queue is ordered by the semaphore waits.
        let (src_stage, src_access) = if self.acquire_from.is_some() {
            (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE)
        } else {
            (
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            )
        };
        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .old_layout(self.old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(src_family)
            .dst_queue_family_index(dst_family)
            .image(image)
            .subresource_range(color_range())
    }

    /// The barrier that leaves the source in its final layout after it was
    /// read in `layout`, releasing it when another family takes it back.
    pub(crate) fn release_barrier(
        &self,
        image: vk::Image,
        family: u32,
        layout: vk::ImageLayout,
        src_stage: vk::PipelineStageFlags2,
    ) -> vk::ImageMemoryBarrier2<'static> {
        let (src_family, dst_family, dst_stage, dst_access) = match self.release_to {
            // The acquiring side's barrier does the rest.
            Some(to) => (
                family,
                to,
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
            ),
            None => (
                vk::QUEUE_FAMILY_IGNORED,
                vk::QUEUE_FAMILY_IGNORED,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            ),
        };
        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .old_layout(layout)
            .new_layout(self.final_layout)
            .src_queue_family_index(src_family)
            .dst_queue_family_index(dst_family)
            .image(image)
            .subresource_range(color_range())
    }
}

/// The source of one frame, as the codec encoder sees it.
#[derive(Debug, Clone, Copy)]
pub(crate) enum FrameInput {
    /// The current slot's input image, already written.
    InPlace,
    /// Copy `image`, which is not the slot's input image, into it first.
    Copy {
        image: vk::Image,
        ownership: SourceOwnership,
    },
    /// Encode straight from `view`, an image made with the session's profile.
    Direct {
        image: vk::Image,
        view: vk::ImageView,
        /// The image's size, which may be smaller than the aligned coded
        /// extent.
        extent: vk::Extent2D,
        ownership: SourceOwnership,
    },
}

/// Semaphore operations one frame adds to the submissions pixelforge makes.
#[derive(Default)]
pub(crate) struct FrameSync {
    /// What the frame's first submission waits for.
    pub waits: Vec<vk::SemaphoreSubmitInfo<'static>>,
    /// What the submission that is last to read the source signals.
    pub release_signals: Vec<vk::SemaphoreSubmitInfo<'static>>,
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// When the frames that read a source are done with it.
///
/// One point per timeline semaphore, so a source read by two encoders keeps
/// the last read of each.
#[derive(Debug)]
pub(crate) struct Uses {
    read: AtomicBool,
    points: Mutex<Vec<TimelinePoint>>,
}

impl Default for Uses {
    fn default() -> Self {
        Self {
            read: AtomicBool::new(false),
            points: Mutex::new(Vec::new()),
        }
    }
}

impl Uses {
    /// Whether a frame has read the source yet.
    pub(crate) fn was_read(&self) -> bool {
        self.read.load(Ordering::Acquire)
    }

    /// Note a frame that is done with the source at `released`.
    pub(crate) fn record(&self, released: TimelinePoint) {
        self.read.store(true, Ordering::Release);
        let mut points = self.points.lock().unwrap_or_else(|p| p.into_inner());
        match points
            .iter_mut()
            .find(|p| p.semaphore == released.semaphore)
        {
            Some(point) => point.value = point.value.max(released.value),
            None => points.push(released),
        }
    }

    /// The points after which every frame so far is done with the source.
    pub(crate) fn points(&self) -> Vec<TimelinePoint> {
        self.points
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Wait on the CPU until every frame so far is done with the source.
    pub(crate) fn wait(&self, context: &VideoContext) -> Result<()> {
        for point in self.points() {
            context.wait_point(point)?;
        }
        Ok(())
    }
}
