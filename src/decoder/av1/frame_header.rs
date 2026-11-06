//! The uncompressed frame header (AV1 specification section 5.9).
//!
//! The parser keeps coded values where Vulkan's `StdVideoDecodeAV1PictureInfo`
//! takes coded values, and derived ones (`OrderHint`, `OrderHints`,
//! `SkipModeFrame`, `TxMode`, `expectedFrameId`, the frame size after
//! superres) where it takes those. Values a frame inherits from its primary
//! reference frame (loop filter deltas, segmentation features, global motion
//! references, film grain) are resolved here, so a header is self-contained.
//!
//! Header parsing never changes the reference state it is given. The few
//! changes the syntax itself makes to that state (a shown key frame resetting
//! it, error resilient order hints and frame ids invalidating slots) are
//! applied to a copy, which the caller keeps for the reference frame update
//! once the frame is decoded.

use super::bits::BitReader;
use super::consts::*;
use super::obu::ObuHeader;
use super::params::*;
use super::refs::{RefState, mark_ref_frames, relative_dist, set_frame_refs};
use super::sequence::SequenceHeader;
use super::tile::{TileInfo, parse_tile_info};
use crate::error::{PixelForgeError, Result};

/// `frame_type` (section 6.8.2), numbered as Vulkan's `StdVideoAV1FrameType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum FrameType {
    #[default]
    Key = 0,
    Inter = 1,
    IntraOnly = 2,
    Switch = 3,
}

impl FrameType {
    fn from_raw(raw: u32) -> Self {
        match raw {
            0 => FrameType::Key,
            1 => FrameType::Inter,
            2 => FrameType::IntraOnly,
            _ => FrameType::Switch,
        }
    }

    /// `FrameIsIntra`.
    pub fn is_intra(self) -> bool {
        matches!(self, FrameType::Key | FrameType::IntraOnly)
    }
}

/// The frame size syntax (5.9.5 to 5.9.8) and what it derives.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrameSize {
    /// `FrameWidth`: the coded width, after superres downscaling.
    pub frame_width: u32,
    pub frame_height: u32,
    /// `UpscaledWidth`: the width after superres upscaling.
    pub upscaled_width: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub mi_cols: u32,
    pub mi_rows: u32,
    pub use_superres: bool,
    /// The coded `coded_denom` (0 without superres), as Vulkan takes it.
    pub coded_denom: u8,
    /// `SuperresDenom` (8 without superres).
    pub superres_denom: u32,
    pub render_and_frame_size_different: bool,
    /// With `frame_size_with_refs()`, which of the seven references the size
    /// was copied from (`found_ref`).
    pub found_ref: Option<u8>,
}

/// A parsed `uncompressed_header()`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrameHeader {
    pub show_existing_frame: bool,
    pub frame_to_show_map_idx: u8,
    pub display_frame_id: u32,
    pub frame_presentation_time: u32,
    pub frame_type: FrameType,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: bool,
    pub current_frame_id: u32,
    pub frame_size_override_flag: bool,
    /// `OrderHint`.
    pub order_hint: u8,
    pub primary_ref_frame: u8,
    pub buffer_removal_time_present_flag: bool,
    pub refresh_frame_flags: u8,
    /// The coded `ref_order_hint[]` (error resilient frames only).
    pub ref_order_hint: [u8; NUM_REF_FRAMES],
    pub frame_size: FrameSize,
    pub allow_intrabc: bool,
    pub frame_refs_short_signaling: bool,
    pub last_frame_idx: u8,
    pub gold_frame_idx: u8,
    /// The reference slot behind each of LAST_FRAME..=ALTREF_FRAME (coded,
    /// or derived by `set_frame_refs()`). Zero for intra frames.
    pub ref_frame_idx: [u8; REFS_PER_FRAME],
    /// `expectedFrameId[i]` for reference `LAST_FRAME + i`, as section 5.9.2
    /// indexes it (Vulkan's array has an eighth, unused entry).
    pub expected_frame_id: [u32; REFS_PER_FRAME],
    pub allow_high_precision_mv: bool,
    pub is_filter_switchable: bool,
    /// `interpolation_filter`, `SWITCHABLE` (4) when switchable.
    pub interpolation_filter: u8,
    pub is_motion_mode_switchable: bool,
    pub use_ref_frame_mvs: bool,
    /// `OrderHints[refFrame]`, indexed by reference frame name (entry 0,
    /// INTRA_FRAME, is unused).
    pub order_hints: [u8; TOTAL_REFS_PER_FRAME],
    /// `RefFrameSignBias[refFrame]`, indexed by reference frame name.
    pub ref_frame_sign_bias: [bool; TOTAL_REFS_PER_FRAME],
    pub disable_frame_end_update_cdf: bool,
    pub tile_info: TileInfo,
    pub quantization: QuantizationParams,
    pub segmentation: SegmentationParams,
    pub delta: DeltaParams,
    pub coded_lossless: bool,
    pub all_lossless: bool,
    pub lossless_array: [bool; MAX_SEGMENTS],
    /// `SegQMLevel[plane][segment]`, set when `using_qmatrix`.
    pub seg_qm_level: [[u8; MAX_SEGMENTS]; 3],
    pub loop_filter: LoopFilterParams,
    pub cdef: CdefParams,
    pub loop_restoration: LoopRestorationParams,
    /// `TxMode`: 0 ONLY_4X4, 1 TX_MODE_LARGEST, 2 TX_MODE_SELECT.
    pub tx_mode: u8,
    pub reference_select: bool,
    pub skip_mode_present: bool,
    /// `SkipModeFrame[0..2]` as reference frame names; zero when skip mode is
    /// not allowed.
    pub skip_mode_frame: [u8; 2],
    pub allow_warped_motion: bool,
    pub reduced_tx_set: bool,
    pub global_motion: GlobalMotionParams,
    pub film_grain: FilmGrainParams,
    /// Length of `uncompressed_header()` in bits, without the byte
    /// alignment or trailing bits that follow it.
    pub header_bits: usize,
}

impl FrameHeader {
    /// `FrameIsIntra`.
    pub fn frame_is_intra(&self) -> bool {
        self.frame_type.is_intra()
    }

    /// `RefFrameSignBias` as a bitmask indexed by reference frame name.
    pub fn ref_frame_sign_bias_mask(&self) -> u8 {
        self.ref_frame_sign_bias
            .iter()
            .enumerate()
            .fold(0, |mask, (i, &b)| mask | ((b as u8) << i))
    }
}

/// What parsing a frame header produced.
#[derive(Debug)]
pub(crate) enum HeaderOutcome {
    /// The header, and the reference state with the header's own changes
    /// applied, for the update once the frame is decoded.
    Parsed(Box<FrameHeader>, Box<RefState>),
    /// The header needs reference slots that hold no valid frame (bit `i`
    /// for slot `i`), so the frame cannot be decoded; a decoder has to wait
    /// for the next key frame.
    MissingReferences(u8),
}

fn frame_size(
    r: &mut BitReader,
    seq: &SequenceHeader,
    frame_size_override_flag: bool,
) -> Result<FrameSize> {
    let mut size = FrameSize::default();
    if frame_size_override_flag {
        size.frame_width = r.f(seq.frame_width_bits_minus_1 as u32 + 1)? + 1;
        size.frame_height = r.f(seq.frame_height_bits_minus_1 as u32 + 1)? + 1;
    } else {
        size.frame_width = seq.max_frame_width();
        size.frame_height = seq.max_frame_height();
    }
    superres_params(r, seq, &mut size)?;
    Ok(size)
}

/// `superres_params()` followed by `compute_image_size()`.
fn superres_params(r: &mut BitReader, seq: &SequenceHeader, size: &mut FrameSize) -> Result<()> {
    size.use_superres = seq.enable_superres && r.flag()?;
    size.superres_denom = if size.use_superres {
        size.coded_denom = r.f(SUPERRES_DENOM_BITS)? as u8;
        size.coded_denom as u32 + SUPERRES_DENOM_MIN
    } else {
        SUPERRES_NUM
    };
    size.upscaled_width = size.frame_width;
    size.frame_width =
        (size.upscaled_width * SUPERRES_NUM + size.superres_denom / 2) / size.superres_denom;
    size.mi_cols = 2 * ((size.frame_width + 7) >> 3);
    size.mi_rows = 2 * ((size.frame_height + 7) >> 3);
    Ok(())
}

fn render_size(r: &mut BitReader, size: &mut FrameSize) -> Result<()> {
    size.render_and_frame_size_different = r.flag()?;
    if size.render_and_frame_size_different {
        size.render_width = r.f(16)? + 1;
        size.render_height = r.f(16)? + 1;
    } else {
        size.render_width = size.upscaled_width;
        size.render_height = size.frame_height;
    }
    Ok(())
}

/// `frame_size_with_refs()` (5.9.7).
fn frame_size_with_refs(
    r: &mut BitReader,
    seq: &SequenceHeader,
    refs: &RefState,
    ref_frame_idx: &[u8; REFS_PER_FRAME],
    frame_size_override_flag: bool,
) -> Result<FrameSize> {
    for (i, &idx) in ref_frame_idx.iter().enumerate() {
        if r.flag()? {
            let slot = &refs.slots[idx as usize];
            let mut size = FrameSize {
                frame_width: slot.upscaled_width,
                frame_height: slot.frame_height,
                render_width: slot.render_width,
                render_height: slot.render_height,
                found_ref: Some(i as u8),
                ..Default::default()
            };
            superres_params(r, seq, &mut size)?;
            return Ok(size);
        }
    }
    let mut size = frame_size(r, seq, frame_size_override_flag)?;
    render_size(r, &mut size)?;
    Ok(size)
}

/// Fill a `show_existing_frame` header from the slot it shows (the parts of
/// the loading process, 7.21, a decoder needs to output the frame).
fn load_shown_frame(seq: &SequenceHeader, refs: &RefState, h: &mut FrameHeader) {
    let slot = &refs.slots[h.frame_to_show_map_idx as usize];
    h.frame_type = slot.frame_type;
    h.order_hint = slot.order_hint;
    h.order_hints = slot.saved_order_hints;
    h.current_frame_id = slot.frame_id;
    h.frame_size = FrameSize {
        frame_width: slot.frame_width,
        frame_height: slot.frame_height,
        upscaled_width: slot.upscaled_width,
        render_width: slot.render_width,
        render_height: slot.render_height,
        mi_cols: slot.mi_cols,
        mi_rows: slot.mi_rows,
        superres_denom: SUPERRES_NUM,
        ..Default::default()
    };
    if seq.film_grain_params_present {
        h.film_grain = slot.film_grain.clone();
    }
    if h.frame_type == FrameType::Key {
        h.refresh_frame_flags = 0xff;
    }
}

/// Parse `uncompressed_header()` against the current reference state.
///
/// `prev_order_hints` is `OrderHints` as the previous frame header left it:
/// the spec keeps it across frames, and an intra frame stores it unchanged.
pub(crate) fn parse_uncompressed_header(
    r: &mut BitReader,
    seq: &SequenceHeader,
    refs: &RefState,
    obu: &ObuHeader,
    prev_order_hints: [u8; TOTAL_REFS_PER_FRAME],
) -> Result<HeaderOutcome> {
    let start = r.position();
    let mut refs = refs.clone();
    let mut h = FrameHeader {
        order_hints: prev_order_hints,
        ..Default::default()
    };
    let id_len = seq.frame_id_length();
    let decoder_model = seq.decoder_model_info.as_ref();
    let all_frames = 0xffu8;

    if seq.reduced_still_picture_header {
        h.frame_type = FrameType::Key;
        h.show_frame = true;
        h.error_resilient_mode = true;
    } else {
        h.show_existing_frame = r.flag()?;
        if h.show_existing_frame {
            h.frame_to_show_map_idx = r.f(3)? as u8;
            if let Some(model) = decoder_model
                && !seq.equal_picture_interval()
            {
                h.frame_presentation_time =
                    r.f(model.frame_presentation_time_length_minus_1 as u32 + 1)?;
            }
            if seq.frame_id_numbers_present_flag {
                h.display_frame_id = r.f(id_len)?;
            }
            if !refs.slots[h.frame_to_show_map_idx as usize].valid {
                return Ok(HeaderOutcome::MissingReferences(
                    1 << h.frame_to_show_map_idx,
                ));
            }
            load_shown_frame(seq, &refs, &mut h);
            h.header_bits = r.position() - start;
            return Ok(HeaderOutcome::Parsed(Box::new(h), Box::new(refs)));
        }
        h.frame_type = FrameType::from_raw(r.f(2)?);
        h.show_frame = r.flag()?;
        if h.show_frame
            && let Some(model) = decoder_model
            && !seq.equal_picture_interval()
        {
            h.frame_presentation_time =
                r.f(model.frame_presentation_time_length_minus_1 as u32 + 1)?;
        }
        h.showable_frame = if h.show_frame {
            h.frame_type != FrameType::Key
        } else {
            r.flag()?
        };
        h.error_resilient_mode = if h.frame_type == FrameType::Switch
            || (h.frame_type == FrameType::Key && h.show_frame)
        {
            true
        } else {
            r.flag()?
        };
    }
    let frame_is_intra = h.frame_is_intra();

    if h.frame_type == FrameType::Key && h.show_frame {
        for slot in refs.slots.iter_mut() {
            slot.valid = false;
            slot.order_hint = 0;
        }
        for i in 0..REFS_PER_FRAME {
            h.order_hints[LAST_FRAME + i] = 0;
        }
    }
    h.disable_cdf_update = r.flag()?;
    h.allow_screen_content_tools =
        if seq.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
            r.flag()?
        } else {
            seq.seq_force_screen_content_tools != 0
        };
    if h.allow_screen_content_tools {
        h.force_integer_mv = if seq.seq_force_integer_mv == SELECT_INTEGER_MV {
            r.flag()?
        } else {
            seq.seq_force_integer_mv != 0
        };
    }
    if frame_is_intra {
        h.force_integer_mv = true;
    }
    if seq.frame_id_numbers_present_flag {
        h.current_frame_id = r.f(id_len)?;
        mark_ref_frames(seq, &mut refs, h.current_frame_id);
    }
    h.frame_size_override_flag = if h.frame_type == FrameType::Switch {
        true
    } else if seq.reduced_still_picture_header {
        false
    } else {
        r.flag()?
    };
    h.order_hint = r.f(seq.order_hint_bits as u32)? as u8;
    h.primary_ref_frame = if frame_is_intra || h.error_resilient_mode {
        PRIMARY_REF_NONE
    } else {
        r.f(3)? as u8
    };
    if let Some(model) = decoder_model {
        h.buffer_removal_time_present_flag = r.flag()?;
        if h.buffer_removal_time_present_flag {
            // The removal times are only for the decoder model; skip them.
            for op in &seq.operating_points {
                if op.decoder_model_present_for_this_op {
                    let idc = op.operating_point_idc;
                    let in_temporal = (idc >> obu.temporal_id) & 1 != 0;
                    let in_spatial = (idc >> (obu.spatial_id + 8)) & 1 != 0;
                    if idc == 0 || (in_temporal && in_spatial) {
                        r.f(model.buffer_removal_time_length_minus_1 as u32 + 1)?;
                    }
                }
            }
        }
    }
    h.refresh_frame_flags =
        if h.frame_type == FrameType::Switch || (h.frame_type == FrameType::Key && h.show_frame) {
            all_frames
        } else {
            r.f(8)? as u8
        };
    if h.frame_type == FrameType::IntraOnly && h.refresh_frame_flags == all_frames {
        return Err(PixelForgeError::InvalidInput(
            "AV1: intra-only frame refreshes every reference slot".to_string(),
        ));
    }
    if (!frame_is_intra || h.refresh_frame_flags != all_frames)
        && h.error_resilient_mode
        && seq.enable_order_hint
    {
        for i in 0..NUM_REF_FRAMES {
            let hint = r.f(seq.order_hint_bits as u32)? as u8;
            h.ref_order_hint[i] = hint;
            let slot = &mut refs.slots[i];
            if hint != slot.order_hint {
                // The frame the encoder had here is lost. Remember the order
                // hint it had, as libaom does, so later order arithmetic
                // matches the encoder's; the slot is unusable either way.
                slot.valid = false;
                slot.order_hint = hint;
            }
        }
    }

    if frame_is_intra {
        h.frame_size = frame_size(r, seq, h.frame_size_override_flag)?;
        render_size(r, &mut h.frame_size)?;
        if h.allow_screen_content_tools && h.frame_size.upscaled_width == h.frame_size.frame_width {
            h.allow_intrabc = r.flag()?;
        }
    } else {
        if seq.enable_order_hint {
            h.frame_refs_short_signaling = r.flag()?;
            if h.frame_refs_short_signaling {
                h.last_frame_idx = r.f(3)? as u8;
                h.gold_frame_idx = r.f(3)? as u8;
                h.ref_frame_idx =
                    set_frame_refs(seq, &refs, h.order_hint, h.last_frame_idx, h.gold_frame_idx);
            }
        }
        let mut missing = 0u8;
        for i in 0..REFS_PER_FRAME {
            if !h.frame_refs_short_signaling {
                h.ref_frame_idx[i] = r.f(3)? as u8;
            }
            let slot = &refs.slots[h.ref_frame_idx[i] as usize];
            if seq.frame_id_numbers_present_flag {
                let delta = r.f(seq.delta_frame_id_length_minus_2 as u32 + 2)? + 1;
                let expected = (h.current_frame_id + (1 << id_len) - delta) % (1 << id_len);
                h.expected_frame_id[i] = expected;
                if expected != slot.frame_id {
                    missing |= 1 << h.ref_frame_idx[i];
                }
            }
            if !slot.valid {
                missing |= 1 << h.ref_frame_idx[i];
            }
        }
        if missing != 0 {
            return Ok(HeaderOutcome::MissingReferences(missing));
        }
        h.frame_size = if h.frame_size_override_flag && !h.error_resilient_mode {
            frame_size_with_refs(r, seq, &refs, &h.ref_frame_idx, h.frame_size_override_flag)?
        } else {
            let mut size = frame_size(r, seq, h.frame_size_override_flag)?;
            render_size(r, &mut size)?;
            size
        };
        h.allow_high_precision_mv = if h.force_integer_mv { false } else { r.flag()? };
        h.is_filter_switchable = r.flag()?;
        h.interpolation_filter = if h.is_filter_switchable {
            SWITCHABLE
        } else {
            r.f(2)? as u8
        };
        h.is_motion_mode_switchable = r.flag()?;
        h.use_ref_frame_mvs = if h.error_resilient_mode || !seq.enable_ref_frame_mvs {
            false
        } else {
            r.flag()?
        };
        for i in 0..REFS_PER_FRAME {
            let ref_frame = LAST_FRAME + i;
            let hint = refs.slots[h.ref_frame_idx[i] as usize].order_hint;
            h.order_hints[ref_frame] = hint;
            h.ref_frame_sign_bias[ref_frame] =
                seq.enable_order_hint && relative_dist(seq, hint, h.order_hint) > 0;
        }
    }

    h.disable_frame_end_update_cdf = if seq.reduced_still_picture_header || h.disable_cdf_update {
        true
    } else {
        r.flag()?
    };

    // load_previous() or setup_past_independence(): what the frame inherits.
    let (prev_gm_params, prev_lf_ref, prev_lf_mode, prev_segmentation) =
        if h.primary_ref_frame == PRIMARY_REF_NONE {
            (
                [DEFAULT_GM_PARAMS; TOTAL_REFS_PER_FRAME],
                DEFAULT_LOOP_FILTER_REF_DELTAS,
                [0i8; 2],
                SegmentationParams::default(),
            )
        } else {
            let slot = &refs.slots[h.ref_frame_idx[h.primary_ref_frame as usize] as usize];
            (
                slot.gm_params,
                slot.loop_filter_ref_deltas,
                slot.loop_filter_mode_deltas,
                SegmentationParams {
                    feature_enabled: slot.feature_enabled,
                    feature_data: slot.feature_data,
                    ..Default::default()
                },
            )
        };

    h.tile_info = parse_tile_info(
        r,
        seq.use_128x128_superblock,
        h.frame_size.mi_cols,
        h.frame_size.mi_rows,
    )?;
    h.quantization = parse_quantization(r, seq)?;
    h.segmentation = parse_segmentation(r, h.primary_ref_frame, &prev_segmentation)?;
    h.delta = parse_delta_params(r, h.quantization.base_q_idx, h.allow_intrabc)?;

    let q = &h.quantization;
    h.coded_lossless = true;
    for segment_id in 0..MAX_SEGMENTS {
        let qindex = h.segmentation.qindex(q.base_q_idx, segment_id);
        let lossless = qindex == 0
            && q.delta_q_y_dc == 0
            && q.delta_q_u_ac == 0
            && q.delta_q_u_dc == 0
            && q.delta_q_v_ac == 0
            && q.delta_q_v_dc == 0;
        h.lossless_array[segment_id] = lossless;
        if !lossless {
            h.coded_lossless = false;
        }
        if q.using_qmatrix {
            let levels = if lossless {
                [15, 15, 15]
            } else {
                [q.qm_y, q.qm_u, q.qm_v]
            };
            for (plane, level) in levels.into_iter().enumerate() {
                h.seg_qm_level[plane][segment_id] = level;
            }
        }
    }
    h.all_lossless = h.coded_lossless && h.frame_size.frame_width == h.frame_size.upscaled_width;

    h.loop_filter = parse_loop_filter(
        r,
        seq,
        h.coded_lossless,
        h.allow_intrabc,
        prev_lf_ref,
        prev_lf_mode,
    )?;
    h.cdef = parse_cdef(r, seq, h.coded_lossless, h.allow_intrabc)?;
    h.loop_restoration = parse_lr(r, seq, h.all_lossless, h.allow_intrabc)?;
    h.tx_mode = if h.coded_lossless {
        0
    } else if r.flag()? {
        2
    } else {
        1
    };
    h.reference_select = !frame_is_intra && r.flag()?;
    h.skip_mode_frame = skip_mode_frames(seq, &refs, &h);
    h.skip_mode_present = h.skip_mode_frame[0] != 0 && r.flag()?;
    h.allow_warped_motion =
        !(frame_is_intra || h.error_resilient_mode || !seq.enable_warped_motion) && r.flag()?;
    h.reduced_tx_set = r.flag()?;
    h.global_motion = parse_global_motion(
        r,
        frame_is_intra,
        h.allow_high_precision_mv,
        &prev_gm_params,
    )?;

    let ctx = GrainContext {
        show_frame: h.show_frame,
        showable_frame: h.showable_frame,
        is_inter_frame: h.frame_type == FrameType::Inter,
    };
    let load = |idx: u8| {
        let slot = &refs.slots[idx as usize];
        slot.valid.then(|| slot.film_grain.clone())
    };
    h.film_grain = match parse_film_grain(r, seq, &ctx, load)? {
        Ok(grain) => grain,
        Err(slot) => return Ok(HeaderOutcome::MissingReferences(1 << slot)),
    };

    h.header_bits = r.position() - start;
    Ok(HeaderOutcome::Parsed(Box::new(h), Box::new(refs)))
}

/// `skip_mode_params()` (5.9.22) up to `skipModeAllowed`: the two
/// reference frame names skip mode would use, or zeros when it is not
/// allowed.
fn skip_mode_frames(seq: &SequenceHeader, refs: &RefState, h: &FrameHeader) -> [u8; 2] {
    if h.frame_is_intra() || !h.reference_select || !seq.enable_order_hint {
        return [0, 0];
    }
    let hint_of = |i: usize| refs.slots[h.ref_frame_idx[i] as usize].order_hint;
    let dist = |a: u8, b: u8| relative_dist(seq, a, b);
    let mut forward: Option<(usize, u8)> = None;
    let mut backward: Option<(usize, u8)> = None;
    for i in 0..REFS_PER_FRAME {
        let ref_hint = hint_of(i);
        if dist(ref_hint, h.order_hint) < 0 {
            if forward.is_none_or(|(_, f)| dist(ref_hint, f) > 0) {
                forward = Some((i, ref_hint));
            }
        } else if dist(ref_hint, h.order_hint) > 0
            && backward.is_none_or(|(_, b)| dist(ref_hint, b) < 0)
        {
            backward = Some((i, ref_hint));
        }
    }
    let Some((forward_idx, forward_hint)) = forward else {
        return [0, 0];
    };
    let second = if let Some((backward_idx, _)) = backward {
        backward_idx
    } else {
        let mut second_forward: Option<(usize, u8)> = None;
        for i in 0..REFS_PER_FRAME {
            let ref_hint = hint_of(i);
            if dist(ref_hint, forward_hint) < 0
                && second_forward.is_none_or(|(_, s)| dist(ref_hint, s) > 0)
            {
                second_forward = Some((i, ref_hint));
            }
        }
        let Some((second_idx, _)) = second_forward else {
            return [0, 0];
        };
        second_idx
    };
    [
        (LAST_FRAME + forward_idx.min(second)) as u8,
        (LAST_FRAME + forward_idx.max(second)) as u8,
    ]
}
