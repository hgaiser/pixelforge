//! GOP (Group of Pictures) structure for video encoding.
//!
//! This module provides GOP structure management for H.264 and H.265 encoders.
//! The GOP structure determines frame types (IDR or P).

/// Frame type in GOP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GopFrameType {
    /// IDR frame.
    Idr,
    /// P frame (predicted).
    P,
}

impl GopFrameType {
    /// Whether this is an IDR frame.
    pub fn is_idr(&self) -> bool {
        matches!(self, Self::Idr)
    }
}

/// Position in the GOP.
#[derive(Debug, Clone)]
pub struct GopPosition {
    /// Frame type.
    pub frame_type: GopFrameType,
    /// POC (Picture Order Count).
    pub pic_order_cnt: i32,
    /// Whether this frame is a reference.
    pub is_reference: bool,
}

/// GOP structure manager.
///
/// Manages the GOP structure for video encoding, determining frame types.
/// and their ordering. This is codec-agnostic and can be used for both
/// H.264 and H.265 encoding.
pub struct GopStructure {
    /// Current frame index.
    frame_index: u64,
    /// Current POC.
    poc: i32,
    /// IDR period.
    idr_period: u32,
    /// Max POC LSB value (log2_max_pic_order_cnt_lsb_minus4 + 4).
    max_poc_lsb: u32,
    /// Flag to force next frame to be an IDR.
    force_idr: bool,
}

impl GopStructure {
    /// An I-P GOP structure, with an IDR every `gop_size` frames.
    pub fn new(gop_size: u32) -> Self {
        Self {
            idr_period: gop_size,
            frame_index: 0,
            poc: 0,
            max_poc_lsb: 16, // log2_max_pic_order_cnt_lsb_minus4=0 -> 2^4 = 16
            force_idr: false,
        }
    }

    /// Set max POC LSB (2^(log2_max_pic_order_cnt_lsb_minus4+4)).
    pub fn set_max_poc_lsb(&mut self, log2_max_pic_order_cnt_lsb_minus4: u8) {
        self.max_poc_lsb = 1 << (log2_max_pic_order_cnt_lsb_minus4 + 4);
    }

    /// Get the next frame position in the GOP.
    pub fn get_next_frame(&mut self) -> GopPosition {
        let is_idr = self.frame_index == 0
            || (self.idr_period > 0 && self.frame_index.is_multiple_of(self.idr_period as u64))
            || self.force_idr;

        // Clear the force IDR flag after checking it.
        self.force_idr = false;

        // If IDR, reset counters.
        if is_idr {
            self.poc = 0;
        }

        let frame_type = if is_idr {
            GopFrameType::Idr
        } else {
            GopFrameType::P
        };

        let position = GopPosition {
            frame_type,
            pic_order_cnt: self.poc,
            is_reference: true,
        };

        // Update counters for next frame.
        self.poc = (self.poc + 2) % (self.max_poc_lsb as i32 * 2);
        self.frame_index += 1;

        position
    }

    /// Request that the next frame be an IDR frame.
    pub fn request_idr(&mut self) {
        self.force_idr = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ip_only_gop() {
        let mut gop = GopStructure::new(30);

        // First frame should be IDR.
        let pos = gop.get_next_frame();
        assert_eq!(pos.frame_type, GopFrameType::Idr);
        assert!(pos.is_reference);

        // Next frames should be P.
        for i in 1..30 {
            let pos = gop.get_next_frame();
            assert_eq!(pos.frame_type, GopFrameType::P, "Frame {i} should be P");
            assert!(pos.is_reference);
        }

        // Frame 30 should be IDR (new GOP)
        let pos = gop.get_next_frame();
        assert_eq!(pos.frame_type, GopFrameType::Idr);
    }
}
