//! Ordering pixelforge's GPU work against the caller's, without the CPU.

use ash::vk;

/// A point on a timeline semaphore: reached once the semaphore's value is at
/// least `value`.
///
/// Returned by [`ColorConverter::convert_async`](crate::ColorConverter::convert_async)
/// for the work it submitted, and accepted by it and by
/// [`Encoder::encode_after`](crate::Encoder::encode_after) as work to wait
/// for. On a device shared with the caller this is how a frame moves from the
/// caller's rendering to the converter to the encoder with every step ordered
/// on the GPU: each submission waits on the point the previous one signals,
/// and nothing waits on the CPU.
///
/// The semaphore must be a timeline semaphore on the same device as the
/// context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelinePoint {
    /// A timeline semaphore.
    pub semaphore: vk::Semaphore,
    /// The value to wait for.
    pub value: u64,
}

impl TimelinePoint {
    /// The point where `semaphore` reaches `value`.
    pub fn new(semaphore: vk::Semaphore, value: u64) -> Self {
        Self { semaphore, value }
    }

    /// A wait on this point that holds back every stage of the submission.
    ///
    /// Pixelforge's submissions start with layout transitions whose stages the
    /// caller cannot know, so a narrower stage mask could let one of them run
    /// before the caller's work is done.
    pub(crate) fn wait_info(&self) -> vk::SemaphoreSubmitInfo<'static> {
        vk::SemaphoreSubmitInfo::default()
            .semaphore(self.semaphore)
            .value(self.value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
    }
}
