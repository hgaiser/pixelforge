//! Decoded Picture Buffer (DPB) management for video encoding.
//!
//! This module tracks the reference pictures of H.264 and H.265 encoding:
//! picture order counts, frame numbers and reference marking.
//!
//! The DPB is codec-agnostic at the interface level, with codec-specific
//! implementations for H.264 and H.265 in separate modules.

mod entry;
mod h264;
mod h265;
mod types;

pub use h264::DpbH264;
pub use h265::DpbH265;
pub use types::*;

/// Maximum DPB size (16 slots for H.264, 15 for H.265, use 16 for both).
pub const MAX_DPB_SIZE: usize = 16;

/// Alias for MAX_DPB_SIZE for backwards compatibility.
pub const MAX_DPB_SLOTS: usize = MAX_DPB_SIZE;

/// Default DPB size for encoding.
pub const DEFAULT_DPB_SIZE: u32 = 4;

/// Picture type for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PictureType {
    /// IDR picture (Instantaneous Decoder Refresh).
    Idr,
    /// I picture (Intra).
    I,
    /// P picture (Predicted, uses L0 references).
    P,
}

/// Common DPB interface for both H.264 and H.265.
///
/// This trait provides a codec-agnostic interface for DPB operations.
pub trait DecodedPictureBufferTrait {
    /// Initialize the DPB for a new sequence.
    fn sequence_start(&mut self, config: DpbConfig);

    /// Start encoding a new picture.
    ///
    /// Returns the DPB slot index allocated for this picture.
    fn picture_start(&mut self, info: PictureStartInfo) -> i8;

    /// End encoding a picture and update the DPB.
    fn picture_end(&mut self, is_reference: bool);

    /// Check if the DPB is full.
    fn is_full(&self) -> bool;

    /// Check if the DPB is empty.
    fn is_empty(&self) -> bool;

    /// Flush all frames from the DPB.
    fn flush(&mut self);

    /// Get the number of short-term reference frames.
    fn num_short_term_refs(&self) -> u32;

    /// Get the number of long-term reference frames.
    fn num_long_term_refs(&self) -> u32;
}

/// Unified DPB that works for both H.264 and H.265.
///
/// This provides a simple interface for encoders while delegating.
/// to codec-specific implementations internally.
pub struct DecodedPictureBuffer {
    /// H.264-specific DPB (used when codec is H.264).
    pub h264: DpbH264,
    /// H.265-specific DPB (used when codec is H.265).
    pub h265: DpbH265,
}

impl DecodedPictureBuffer {
    /// Create a new DPB.
    pub fn new() -> Self {
        Self {
            h264: DpbH264::new(),
            h265: DpbH265::new(),
        }
    }
}

impl Default for DecodedPictureBuffer {
    fn default() -> Self {
        Self::new()
    }
}
