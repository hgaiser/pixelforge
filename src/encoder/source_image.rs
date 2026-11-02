//! A `vk::Image` of the caller's that frames are encoded from.

use ash::vk;

use super::input::{SourceOwnership, Uses};
use crate::error::{PixelForgeError, Result};
use crate::sync::TimelinePoint;
use crate::vulkan::VideoContext;

/// How the caller hands a [`SourceImage`] to pixelforge for each frame, and
/// how pixelforge gives it back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handoff {
    /// The caller writes the image on this device, and hands it over in
    /// `layout`. Pixelforge gives it back in `layout`.
    ///
    /// Pixelforge reads it on its own queues with no queue family transfer,
    /// so make the image with `CONCURRENT` sharing across the caller's queue
    /// families and pixelforge's, or write it on the queue family pixelforge
    /// reads it on.
    Device {
        /// The layout the image is in when a frame is submitted, and is left
        /// in once the frame is done with it. Not `UNDEFINED` or
        /// `PREINITIALIZED`, which do not keep the pixels.
        layout: vk::ImageLayout,
    },
    /// The image's memory is imported, and written outside this device: by
    /// another process, another API or another Vulkan instance.
    ///
    /// Each frame acquires the image from `VK_QUEUE_FAMILY_FOREIGN_EXT` (or
    /// `VK_QUEUE_FAMILY_EXTERNAL` without `VK_EXT_queue_family_foreign`), and
    /// releases it back in `GENERAL`.
    External,
}

/// A `vk::Image` of the caller's that frames are encoded from, see
/// [`Frame::image`](crate::Frame::image).
///
/// It keeps track of the frames that read the image. Write the image again
/// only after [`Self::released`]: wait for those points on the GPU, or call
/// [`Self::wait_released`]. Dropping it waits for them too, so drop it before
/// the image is destroyed.
///
/// The image must be in the format and the size the encoder's
/// [`Source`](crate::Source) says. Pixelforge reads it with a transfer, or
/// with the colour converter's shader for an RGB source, so it needs
/// `TRANSFER_SRC` or `SAMPLED` usage to match.
pub struct SourceImage {
    context: VideoContext,
    image: vk::Image,
    handoff: Handoff,
    uses: Uses,
}

impl std::fmt::Debug for SourceImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceImage")
            .field("image", &self.image)
            .field("handoff", &self.handoff)
            .finish()
    }
}

impl SourceImage {
    /// Encode frames from `image`, handed over as `handoff` says.
    pub fn new(context: &VideoContext, image: vk::Image, handoff: Handoff) -> Result<Self> {
        if let Handoff::Device { layout } = handoff
            && matches!(
                layout,
                vk::ImageLayout::UNDEFINED | vk::ImageLayout::PREINITIALIZED
            )
        {
            return Err(PixelForgeError::InvalidInput(format!(
                "a source image handed over in {layout:?} has no pixels to read"
            )));
        }
        Ok(Self {
            context: context.clone(),
            image,
            handoff,
            uses: Uses::default(),
        })
    }

    /// The image.
    pub fn image(&self) -> vk::Image {
        self.image
    }

    /// How the image is handed over.
    pub fn handoff(&self) -> Handoff {
        self.handoff
    }

    /// The points after which every frame submitted so far is done with the
    /// image, so it can be written again. Empty before the first frame.
    pub fn released(&self) -> Vec<TimelinePoint> {
        self.uses.points()
    }

    /// Wait on the CPU until every frame submitted so far is done with the
    /// image.
    pub fn wait_released(&self) -> Result<()> {
        self.uses.wait(&self.context)
    }

    /// The layout and queue family transfers for the next frame that reads
    /// the image.
    pub(crate) fn ownership(&self) -> SourceOwnership {
        match self.handoff {
            Handoff::Device { layout } => SourceOwnership {
                old_layout: layout,
                acquire_from: None,
                release_to: None,
                final_layout: layout,
            },
            Handoff::External => {
                let foreign = self.context.external().foreign_family();
                SourceOwnership {
                    old_layout: if self.uses.was_read() {
                        vk::ImageLayout::GENERAL
                    } else {
                        vk::ImageLayout::UNDEFINED
                    },
                    acquire_from: Some(foreign),
                    release_to: Some(foreign),
                    final_layout: vk::ImageLayout::GENERAL,
                }
            }
        }
    }

    /// What the next frame has to wait for before it reads the image: the
    /// previous release, for an image that goes back to another owner.
    pub(crate) fn pending_release(&self) -> Vec<TimelinePoint> {
        match self.handoff {
            Handoff::Device { .. } => Vec::new(),
            Handoff::External => self.uses.points(),
        }
    }

    pub(crate) fn mark_used(&self, released: TimelinePoint) {
        self.uses.record(released);
    }
}

impl Drop for SourceImage {
    fn drop(&mut self) {
        let _ = self.uses.wait(&self.context);
    }
}
