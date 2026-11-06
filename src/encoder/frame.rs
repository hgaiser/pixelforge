//! One frame handed to [`Encoder::submit`](crate::Encoder::submit).

use ash::vk;

use super::source_image::SourceImage;
use crate::sync::TimelinePoint;

/// Where a frame's pixels are.
pub(crate) enum FrameSource<'a> {
    /// The encoder's own input image for the current slot.
    Input(InputSlot),
    /// A `vk::Image` of the caller's.
    Image(&'a SourceImage),
    /// A DMA-BUF pixelforge imported or allocated.
    #[cfg(feature = "dmabuf")]
    External(&'a crate::encoder::external::DmaBufImage),
}

/// The encoder's input image for the next frame, from
/// [`Encoder::input_slot`](crate::Encoder::input_slot).
///
/// Write the frame into [`Self::image`], then submit the slot with
/// [`Frame::input`]: the encoder reads it in place, with no copy. The slot is
/// valid until the encoder's next submission; submitting it later is an
/// error.
#[derive(Debug)]
#[must_use]
pub struct InputSlot {
    pub(crate) image: vk::Image,
    pub(crate) encoder: u64,
    pub(crate) ticket: u64,
}

impl InputSlot {
    /// The image to write the frame into. It is in `VIDEO_ENCODE_SRC_KHR`
    /// layout, and must be in that layout when the slot is submitted.
    pub fn image(&self) -> vk::Image {
        self.image
    }
}

/// One frame to encode: its source, and what has to happen on the GPU before
/// it can be read.
///
/// ```no_run
/// # use pixelforge::{Encoder, Frame, Handoff, SourceImage, TimelinePoint, VideoContext};
/// # use ash::vk;
/// # fn f(context: &VideoContext, encoder: &mut Encoder, image: vk::Image, rendered: TimelinePoint) -> Result<(), pixelforge::PixelForgeError> {
/// let source = SourceImage::new(
///     context,
///     image,
///     Handoff::Device { layout: vk::ImageLayout::GENERAL },
/// )?;
/// let packet = encoder.submit(Frame::image(&source).wait(&[rendered]))?;
/// // Render the next frame into `image` only after these points.
/// let reusable = source.released();
/// # Ok(())
/// # }
/// ```
#[must_use]
pub struct Frame<'a> {
    pub(crate) source: FrameSource<'a>,
    pub(crate) waits: &'a [TimelinePoint],
    #[cfg(feature = "dmabuf")]
    pub(crate) wait_sync_file: Option<std::os::fd::OwnedFd>,
    #[cfg(feature = "dmabuf")]
    pub(crate) signal_sync_file: bool,
}

impl<'a> Frame<'a> {
    /// A frame in the encoder's own input image, written in place.
    pub fn input(slot: InputSlot) -> Self {
        Self::new(FrameSource::Input(slot))
    }

    /// A frame in `source`, an image of the caller's.
    ///
    /// Pixelforge copies it into the encoder's input image, or converts it
    /// there for an RGB source. That happens after
    /// [`Encoder::submit`](crate::Encoder::submit) returns, so write the image
    /// again only after [`SourceImage::released`].
    pub fn image(source: &'a SourceImage) -> Self {
        Self::new(FrameSource::Image(source))
    }

    /// A frame in a DMA-BUF that the encoder imported or allocated, see
    /// [`Encoder::import`](crate::Encoder::import) and
    /// [`Encoder::allocate_input`](crate::Encoder::allocate_input).
    ///
    /// It is read directly when the device can, and copied otherwise.
    /// Pixelforge acquires it from the foreign queue family before it reads,
    /// and releases it back after.
    #[cfg(feature = "dmabuf")]
    pub fn external(image: &'a crate::encoder::external::DmaBufImage) -> Self {
        Self::new(FrameSource::External(image))
    }

    fn new(source: FrameSource<'a>) -> Self {
        Self {
            source,
            waits: &[],
            #[cfg(feature = "dmabuf")]
            wait_sync_file: None,
            #[cfg(feature = "dmabuf")]
            signal_sync_file: false,
        }
    }

    /// Read the frame only once every point in `points` is reached, on the
    /// GPU.
    pub fn wait(mut self, points: &'a [TimelinePoint]) -> Self {
        self.waits = points;
        self
    }

    /// Read the frame only once `fence`, a sync_file, signals, on the GPU.
    ///
    /// For a DMA-BUF whose producer gives its write fence as a sync_file,
    /// such as one taken from the buffer with `DMA_BUF_IOCTL_EXPORT_SYNC_FILE`.
    /// Pixelforge takes ownership of the fd.
    #[cfg(feature = "dmabuf")]
    pub fn wait_sync_file(mut self, fence: std::os::fd::OwnedFd) -> Self {
        self.wait_sync_file = Some(fence);
        self
    }

    /// Also give the point after which the source is free as a sync_file,
    /// from [`EncodeFuture::take_release_sync_file`](crate::EncodeFuture::take_release_sync_file).
    ///
    /// For a producer that waits on sync_files, or to attach to the DMA-BUF
    /// with `DMA_BUF_IOCTL_IMPORT_SYNC_FILE` before handing it back.
    #[cfg(feature = "dmabuf")]
    pub fn signal_sync_file(mut self) -> Self {
        self.signal_sync_file = true;
        self
    }
}
