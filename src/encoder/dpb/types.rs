//! DPB types and configuration structures.

use super::DEFAULT_DPB_SIZE;

/// Configuration for DPB initialization.
#[derive(Debug, Clone)]
pub struct DpbConfig {
    /// Maximum DPB size in frames.
    pub dpb_size: u32,
    /// Log2 of max_frame_num (for H.264 frame_num calculation).
    pub log2_max_frame_num_minus4: u8,
    /// Log2 of max_pic_order_cnt_lsb (for POC calculation).
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    /// Number of temporal layers (for temporal SVC support).
    pub num_temporal_layers: u32,
}

impl Default for DpbConfig {
    fn default() -> Self {
        Self {
            dpb_size: DEFAULT_DPB_SIZE,
            log2_max_frame_num_minus4: 0,         // max_frame_num = 16
            log2_max_pic_order_cnt_lsb_minus4: 0, // max_poc_lsb = 16
            num_temporal_layers: 1,
        }
    }
}

impl DpbConfig {
    /// Get the maximum frame_num value.
    pub fn max_frame_num(&self) -> u32 {
        1 << (self.log2_max_frame_num_minus4 + 4)
    }

    /// Get the maximum POC LSB value.
    pub fn max_pic_order_cnt_lsb(&self) -> i32 {
        1 << (self.log2_max_pic_order_cnt_lsb_minus4 + 4)
    }
}

/// Information for starting a new picture.
#[derive(Debug, Clone)]
pub struct PictureStartInfo {
    /// Unique frame ID (internal tracking).
    pub frame_id: u64,
    /// Picture Order Count.
    pub pic_order_cnt: i32,
    /// Frame number (for H.264).
    pub frame_num: u32,
    /// Picture type (IDR, I, P, B).
    pub pic_type: super::PictureType,
    /// Temporal layer ID.
    pub temporal_id: u8,
    /// Timestamp for this picture.
    pub timestamp: u64,
    /// Whether this picture should be output.
    pub pic_output_flag: bool,
    /// Whether this is a reference picture.
    pub is_reference: bool,
    /// For H.264: no_output_of_prior_pics_flag.
    pub no_output_of_prior_pics_flag: bool,
}

impl Default for PictureStartInfo {
    fn default() -> Self {
        Self {
            frame_id: 0,
            pic_order_cnt: 0,
            frame_num: 0,
            pic_type: super::PictureType::I,
            temporal_id: 0,
            timestamp: 0,
            pic_output_flag: true,
            is_reference: true,
            no_output_of_prior_pics_flag: false,
        }
    }
}
