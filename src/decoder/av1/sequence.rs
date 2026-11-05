//! Sequence header OBU (AV1 specification section 5.5).
//!
//! Every syntax element is kept, including those Vulkan does not take, so a
//! change in any of them can be detected by comparing two headers.

use super::bits::BitReader;
use super::consts::{SELECT_INTEGER_MV, SELECT_SCREEN_CONTENT_TOOLS};
use crate::error::{PixelForgeError, Result};

/// `timing_info()` (section 5.5.3).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TimingInfo {
    pub num_units_in_display_tick: u32,
    pub time_scale: u32,
    pub equal_picture_interval: bool,
    /// Zero unless `equal_picture_interval`.
    pub num_ticks_per_picture_minus_1: u32,
}

/// `decoder_model_info()` (section 5.5.4).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DecoderModelInfo {
    pub buffer_delay_length_minus_1: u8,
    pub num_units_in_decoding_tick: u32,
    pub buffer_removal_time_length_minus_1: u8,
    pub frame_presentation_time_length_minus_1: u8,
}

/// One entry of the operating point loop.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OperatingPoint {
    pub operating_point_idc: u16,
    pub seq_level_idx: u8,
    pub seq_tier: u8,
    pub decoder_model_present_for_this_op: bool,
    /// `operating_parameters_info()`, when the decoder model is present.
    pub decoder_buffer_delay: u32,
    pub encoder_buffer_delay: u32,
    pub low_delay_mode_flag: bool,
    pub initial_display_delay_present_for_this_op: bool,
    pub initial_display_delay_minus_1: u8,
}

/// `color_config()` (section 5.5.2), with the derived `BitDepth`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ColorConfig {
    pub high_bitdepth: bool,
    pub twelve_bit: bool,
    /// `BitDepth`: 8, 10 or 12.
    pub bit_depth: u8,
    pub mono_chrome: bool,
    pub color_description_present_flag: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub color_range: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub chroma_sample_position: u8,
    pub separate_uv_delta_q: bool,
}

impl ColorConfig {
    /// `NumPlanes`.
    pub fn num_planes(&self) -> usize {
        if self.mono_chrome { 1 } else { 3 }
    }
}

/// Parsed sequence header OBU.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SequenceHeader {
    pub seq_profile: u8,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub timing_info: Option<TimingInfo>,
    pub decoder_model_info: Option<DecoderModelInfo>,
    pub initial_display_delay_present_flag: bool,
    /// At least one entry; the parser decodes operating point 0.
    pub operating_points: Vec<OperatingPoint>,
    pub frame_width_bits_minus_1: u8,
    pub frame_height_bits_minus_1: u8,
    pub max_frame_width_minus_1: u16,
    pub max_frame_height_minus_1: u16,
    pub frame_id_numbers_present_flag: bool,
    pub delta_frame_id_length_minus_2: u8,
    pub additional_frame_id_length_minus_1: u8,
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    pub seq_choose_screen_content_tools: bool,
    /// 0, 1, or `SELECT_SCREEN_CONTENT_TOOLS` (2).
    pub seq_force_screen_content_tools: u8,
    pub seq_choose_integer_mv: bool,
    /// 0, 1, or `SELECT_INTEGER_MV` (2).
    pub seq_force_integer_mv: u8,
    /// `OrderHintBits`: 0 without order hints, else 1..=8.
    pub order_hint_bits: u8,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub color_config: ColorConfig,
    pub film_grain_params_present: bool,
}

impl SequenceHeader {
    /// `OperatingPointIdc` of operating point 0, the one this parser decodes.
    pub fn operating_point_idc(&self) -> u16 {
        self.operating_points
            .first()
            .map_or(0, |op| op.operating_point_idc)
    }

    /// `order_hint_bits_minus_1` as Vulkan takes it (0 when absent).
    pub fn order_hint_bits_minus_1(&self) -> u8 {
        self.order_hint_bits.saturating_sub(1)
    }

    /// `idLen`, the width of frame ids, when they are present.
    pub fn frame_id_length(&self) -> u32 {
        self.additional_frame_id_length_minus_1 as u32
            + self.delta_frame_id_length_minus_2 as u32
            + 3
    }

    /// `equal_picture_interval`, false without timing info.
    pub fn equal_picture_interval(&self) -> bool {
        self.timing_info
            .as_ref()
            .is_some_and(|t| t.equal_picture_interval)
    }

    pub fn max_frame_width(&self) -> u32 {
        self.max_frame_width_minus_1 as u32 + 1
    }

    pub fn max_frame_height(&self) -> u32 {
        self.max_frame_height_minus_1 as u32 + 1
    }
}

/// Parse a sequence header OBU payload, including its trailing bits.
pub fn parse_sequence_header(payload: &[u8]) -> Result<SequenceHeader> {
    let mut r = BitReader::new(payload);
    let mut seq = SequenceHeader {
        seq_profile: r.f(3)? as u8,
        still_picture: r.flag()?,
        reduced_still_picture_header: r.flag()?,
        ..Default::default()
    };
    if seq.seq_profile > 2 {
        return Err(PixelForgeError::InvalidInput(format!(
            "AV1: reserved seq_profile {}",
            seq.seq_profile
        )));
    }

    if seq.reduced_still_picture_header {
        seq.operating_points.push(OperatingPoint {
            seq_level_idx: r.f(5)? as u8,
            ..Default::default()
        });
    } else {
        if r.flag()? {
            let mut timing = TimingInfo {
                num_units_in_display_tick: r.f(32)?,
                time_scale: r.f(32)?,
                equal_picture_interval: r.flag()?,
                num_ticks_per_picture_minus_1: 0,
            };
            if timing.equal_picture_interval {
                timing.num_ticks_per_picture_minus_1 = r.uvlc()?;
            }
            seq.timing_info = Some(timing);
            if r.flag()? {
                seq.decoder_model_info = Some(DecoderModelInfo {
                    buffer_delay_length_minus_1: r.f(5)? as u8,
                    num_units_in_decoding_tick: r.f(32)?,
                    buffer_removal_time_length_minus_1: r.f(5)? as u8,
                    frame_presentation_time_length_minus_1: r.f(5)? as u8,
                });
            }
        }
        seq.initial_display_delay_present_flag = r.flag()?;
        let count = r.f(5)? + 1;
        for _ in 0..count {
            let mut op = OperatingPoint {
                operating_point_idc: r.f(12)? as u16,
                seq_level_idx: r.f(5)? as u8,
                ..Default::default()
            };
            if op.seq_level_idx > 7 {
                op.seq_tier = r.f(1)? as u8;
            }
            if let Some(model) = &seq.decoder_model_info {
                op.decoder_model_present_for_this_op = r.flag()?;
                if op.decoder_model_present_for_this_op {
                    let n = model.buffer_delay_length_minus_1 as u32 + 1;
                    op.decoder_buffer_delay = r.f(n)?;
                    op.encoder_buffer_delay = r.f(n)?;
                    op.low_delay_mode_flag = r.flag()?;
                }
            }
            if seq.initial_display_delay_present_flag {
                op.initial_display_delay_present_for_this_op = r.flag()?;
                if op.initial_display_delay_present_for_this_op {
                    op.initial_display_delay_minus_1 = r.f(4)? as u8;
                }
            }
            seq.operating_points.push(op);
        }
    }

    seq.frame_width_bits_minus_1 = r.f(4)? as u8;
    seq.frame_height_bits_minus_1 = r.f(4)? as u8;
    seq.max_frame_width_minus_1 = r.f(seq.frame_width_bits_minus_1 as u32 + 1)? as u16;
    seq.max_frame_height_minus_1 = r.f(seq.frame_height_bits_minus_1 as u32 + 1)? as u16;
    if !seq.reduced_still_picture_header {
        seq.frame_id_numbers_present_flag = r.flag()?;
    }
    if seq.frame_id_numbers_present_flag {
        seq.delta_frame_id_length_minus_2 = r.f(4)? as u8;
        seq.additional_frame_id_length_minus_1 = r.f(3)? as u8;
        if seq.frame_id_length() > 16 {
            // The spec caps idLen at 16 bits.
            return Err(PixelForgeError::InvalidInput(
                "AV1: frame id length exceeds 16 bits".to_string(),
            ));
        }
    }
    seq.use_128x128_superblock = r.flag()?;
    seq.enable_filter_intra = r.flag()?;
    seq.enable_intra_edge_filter = r.flag()?;

    if seq.reduced_still_picture_header {
        seq.seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
        seq.seq_force_integer_mv = SELECT_INTEGER_MV;
    } else {
        seq.enable_interintra_compound = r.flag()?;
        seq.enable_masked_compound = r.flag()?;
        seq.enable_warped_motion = r.flag()?;
        seq.enable_dual_filter = r.flag()?;
        seq.enable_order_hint = r.flag()?;
        if seq.enable_order_hint {
            seq.enable_jnt_comp = r.flag()?;
            seq.enable_ref_frame_mvs = r.flag()?;
        }
        seq.seq_choose_screen_content_tools = r.flag()?;
        seq.seq_force_screen_content_tools = if seq.seq_choose_screen_content_tools {
            SELECT_SCREEN_CONTENT_TOOLS
        } else {
            r.f(1)? as u8
        };
        seq.seq_force_integer_mv = if seq.seq_force_screen_content_tools > 0 {
            seq.seq_choose_integer_mv = r.flag()?;
            if seq.seq_choose_integer_mv {
                SELECT_INTEGER_MV
            } else {
                r.f(1)? as u8
            }
        } else {
            SELECT_INTEGER_MV
        };
        if seq.enable_order_hint {
            seq.order_hint_bits = r.f(3)? as u8 + 1;
        }
    }

    seq.enable_superres = r.flag()?;
    seq.enable_cdef = r.flag()?;
    seq.enable_restoration = r.flag()?;
    seq.color_config = parse_color_config(&mut r, seq.seq_profile)?;
    seq.film_grain_params_present = r.flag()?;
    r.trailing_bits(payload.len())?;
    Ok(seq)
}

/// Color primaries, transfer and matrix values the parser branches on.
const CP_BT_709: u8 = 1;
const CP_UNSPECIFIED: u8 = 2;
const TC_UNSPECIFIED: u8 = 2;
const TC_SRGB: u8 = 13;
const MC_IDENTITY: u8 = 0;
const MC_UNSPECIFIED: u8 = 2;

fn parse_color_config(r: &mut BitReader, seq_profile: u8) -> Result<ColorConfig> {
    let mut cc = ColorConfig {
        high_bitdepth: r.flag()?,
        ..Default::default()
    };
    cc.bit_depth = if seq_profile == 2 && cc.high_bitdepth {
        cc.twelve_bit = r.flag()?;
        if cc.twelve_bit { 12 } else { 10 }
    } else if cc.high_bitdepth {
        10
    } else {
        8
    };
    cc.mono_chrome = if seq_profile == 1 { false } else { r.flag()? };
    cc.color_description_present_flag = r.flag()?;
    if cc.color_description_present_flag {
        cc.color_primaries = r.f(8)? as u8;
        cc.transfer_characteristics = r.f(8)? as u8;
        cc.matrix_coefficients = r.f(8)? as u8;
    } else {
        cc.color_primaries = CP_UNSPECIFIED;
        cc.transfer_characteristics = TC_UNSPECIFIED;
        cc.matrix_coefficients = MC_UNSPECIFIED;
    }

    if cc.mono_chrome {
        cc.color_range = r.flag()?;
        cc.subsampling_x = true;
        cc.subsampling_y = true;
        // CSP_UNKNOWN, and no separate_uv_delta_q to read.
        return Ok(cc);
    }
    if cc.color_primaries == CP_BT_709
        && cc.transfer_characteristics == TC_SRGB
        && cc.matrix_coefficients == MC_IDENTITY
    {
        cc.color_range = true;
    } else {
        cc.color_range = r.flag()?;
        match seq_profile {
            0 => {
                cc.subsampling_x = true;
                cc.subsampling_y = true;
            }
            1 => {}
            _ => {
                if cc.bit_depth == 12 {
                    cc.subsampling_x = r.flag()?;
                    if cc.subsampling_x {
                        cc.subsampling_y = r.flag()?;
                    }
                } else {
                    cc.subsampling_x = true;
                }
            }
        }
        if cc.subsampling_x && cc.subsampling_y {
            cc.chroma_sample_position = r.f(2)? as u8;
        }
    }
    cc.separate_uv_delta_q = r.flag()?;
    Ok(cc)
}
