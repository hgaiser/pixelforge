//! The coding-tool parameter blocks of the uncompressed header (AV1
//! specification sections 5.9.12 to 5.9.30): quantization, segmentation,
//! delta q/lf, loop filter, CDEF, loop restoration, global motion and film
//! grain.
//!
//! Each block stores what Vulkan's matching `StdVideoAV1*` structure takes,
//! in the form the spec's syntax produces. Where Vulkan wants the coded value
//! rather than the derived one (the CDEF secondary strengths) the coded value
//! is kept and the field says so.

use super::bits::BitReader;
use super::consts::*;
use super::sequence::SequenceHeader;
use crate::error::{PixelForgeError, Result};

/// `quantization_params()` (5.9.12).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QuantizationParams {
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_u_dc: i8,
    pub delta_q_u_ac: i8,
    pub delta_q_v_dc: i8,
    pub delta_q_v_ac: i8,
    pub diff_uv_delta: bool,
    pub using_qmatrix: bool,
    pub qm_y: u8,
    pub qm_u: u8,
    pub qm_v: u8,
}

fn read_delta_q(r: &mut BitReader) -> Result<i8> {
    Ok(if r.flag()? { r.su(7)? as i8 } else { 0 })
}

pub(crate) fn parse_quantization(
    r: &mut BitReader,
    seq: &SequenceHeader,
) -> Result<QuantizationParams> {
    let cc = &seq.color_config;
    let mut q = QuantizationParams {
        base_q_idx: r.f(8)? as u8,
        delta_q_y_dc: read_delta_q(r)?,
        ..Default::default()
    };
    if cc.num_planes() > 1 {
        if cc.separate_uv_delta_q {
            q.diff_uv_delta = r.flag()?;
        }
        q.delta_q_u_dc = read_delta_q(r)?;
        q.delta_q_u_ac = read_delta_q(r)?;
        if q.diff_uv_delta {
            q.delta_q_v_dc = read_delta_q(r)?;
            q.delta_q_v_ac = read_delta_q(r)?;
        } else {
            q.delta_q_v_dc = q.delta_q_u_dc;
            q.delta_q_v_ac = q.delta_q_u_ac;
        }
    }
    q.using_qmatrix = r.flag()?;
    if q.using_qmatrix {
        q.qm_y = r.f(4)? as u8;
        q.qm_u = r.f(4)? as u8;
        q.qm_v = if cc.separate_uv_delta_q {
            r.f(4)? as u8
        } else {
            q.qm_u
        };
    }
    Ok(q)
}

/// `segmentation_params()` (5.9.14), with the feature tables in effect for
/// the frame (loaded from the primary reference frame when not updated).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SegmentationParams {
    pub segmentation_enabled: bool,
    pub segmentation_update_map: bool,
    pub segmentation_temporal_update: bool,
    pub segmentation_update_data: bool,
    /// `FeatureEnabled[segment][feature]`.
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    /// `FeatureData[segment][feature]`, already clipped.
    pub feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub seg_id_pre_skip: bool,
    pub last_active_seg_id: u8,
}

impl SegmentationParams {
    /// `FeatureEnabled` as Vulkan takes it: one bitmask per segment, bit `j`
    /// for feature `j`.
    pub fn feature_enabled_masks(&self) -> [u8; MAX_SEGMENTS] {
        self.feature_enabled.map(|features| {
            features
                .iter()
                .enumerate()
                .fold(0u8, |mask, (j, &on)| mask | ((on as u8) << j))
        })
    }

    /// `get_qindex(1, segment_id)` (7.12.2), the qindex ignoring delta q.
    pub fn qindex(&self, base_q_idx: u8, segment_id: usize) -> u8 {
        if self.segmentation_enabled && self.feature_enabled[segment_id][SEG_LVL_ALT_Q] {
            let q = base_q_idx as i32 + self.feature_data[segment_id][SEG_LVL_ALT_Q] as i32;
            q.clamp(0, 255) as u8
        } else {
            base_q_idx
        }
    }
}

const SEGMENTATION_FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];
const SEGMENTATION_FEATURE_SIGNED: [bool; SEG_LVL_MAX] =
    [true, true, true, true, true, false, false, false];
const SEGMENTATION_FEATURE_MAX: [i32; SEG_LVL_MAX] = [
    255,
    MAX_LOOP_FILTER,
    MAX_LOOP_FILTER,
    MAX_LOOP_FILTER,
    MAX_LOOP_FILTER,
    7,
    0,
    0,
];

/// Parse `segmentation_params()`. `previous` holds the feature tables from
/// `load_previous()` (or zeros after `setup_past_independence()`), which
/// stay in effect when the frame does not update them.
pub(crate) fn parse_segmentation(
    r: &mut BitReader,
    primary_ref_frame: u8,
    previous: &SegmentationParams,
) -> Result<SegmentationParams> {
    let mut s = SegmentationParams {
        segmentation_enabled: r.flag()?,
        ..Default::default()
    };
    if s.segmentation_enabled {
        if primary_ref_frame == PRIMARY_REF_NONE {
            s.segmentation_update_map = true;
            s.segmentation_update_data = true;
        } else {
            s.segmentation_update_map = r.flag()?;
            if s.segmentation_update_map {
                s.segmentation_temporal_update = r.flag()?;
            }
            s.segmentation_update_data = r.flag()?;
        }
        if s.segmentation_update_data {
            for i in 0..MAX_SEGMENTS {
                for j in 0..SEG_LVL_MAX {
                    let enabled = r.flag()?;
                    s.feature_enabled[i][j] = enabled;
                    let mut clipped = 0;
                    if enabled {
                        let bits = SEGMENTATION_FEATURE_BITS[j];
                        let limit = SEGMENTATION_FEATURE_MAX[j];
                        clipped = if SEGMENTATION_FEATURE_SIGNED[j] {
                            r.su(1 + bits)?.clamp(-limit, limit)
                        } else {
                            (r.f(bits)? as i32).clamp(0, limit)
                        };
                    }
                    s.feature_data[i][j] = clipped as i16;
                }
            }
        } else {
            s.feature_enabled = previous.feature_enabled;
            s.feature_data = previous.feature_data;
        }
    }
    for i in 0..MAX_SEGMENTS {
        for j in 0..SEG_LVL_MAX {
            if s.feature_enabled[i][j] {
                s.last_active_seg_id = i as u8;
                if j >= SEG_LVL_REF_FRAME {
                    s.seg_id_pre_skip = true;
                }
            }
        }
    }
    Ok(s)
}

/// `delta_q_params()` and `delta_lf_params()` (5.9.17, 5.9.18).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeltaParams {
    pub delta_q_present: bool,
    /// The coded `delta_q_res` (the shift is `1 << delta_q_res`).
    pub delta_q_res: u8,
    pub delta_lf_present: bool,
    pub delta_lf_res: u8,
    pub delta_lf_multi: bool,
}

pub(crate) fn parse_delta_params(
    r: &mut BitReader,
    base_q_idx: u8,
    allow_intrabc: bool,
) -> Result<DeltaParams> {
    let mut d = DeltaParams::default();
    if base_q_idx > 0 {
        d.delta_q_present = r.flag()?;
    }
    if d.delta_q_present {
        d.delta_q_res = r.f(2)? as u8;
        if !allow_intrabc {
            d.delta_lf_present = r.flag()?;
        }
        if d.delta_lf_present {
            d.delta_lf_res = r.f(2)? as u8;
            d.delta_lf_multi = r.flag()?;
        }
    }
    Ok(d)
}

/// `loop_filter_params()` (5.9.11). The deltas are the values in effect for
/// the frame: loaded from the primary reference frame (or defaults), then
/// updated by whatever the frame codes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoopFilterParams {
    pub loop_filter_level: [u8; 4],
    pub loop_filter_sharpness: u8,
    pub loop_filter_delta_enabled: bool,
    pub loop_filter_delta_update: bool,
    /// Bit `i` set when `loop_filter_ref_deltas[i]` was coded (Vulkan form).
    pub update_ref_delta: u8,
    pub loop_filter_ref_deltas: [i8; TOTAL_REFS_PER_FRAME],
    /// Bit `i` set when `loop_filter_mode_deltas[i]` was coded (Vulkan form).
    pub update_mode_delta: u8,
    pub loop_filter_mode_deltas: [i8; 2],
}

pub(crate) fn parse_loop_filter(
    r: &mut BitReader,
    seq: &SequenceHeader,
    coded_lossless: bool,
    allow_intrabc: bool,
    ref_deltas: [i8; TOTAL_REFS_PER_FRAME],
    mode_deltas: [i8; 2],
) -> Result<LoopFilterParams> {
    let mut lf = LoopFilterParams {
        loop_filter_ref_deltas: ref_deltas,
        loop_filter_mode_deltas: mode_deltas,
        ..Default::default()
    };
    if coded_lossless || allow_intrabc {
        lf.loop_filter_ref_deltas = DEFAULT_LOOP_FILTER_REF_DELTAS;
        lf.loop_filter_mode_deltas = [0, 0];
        return Ok(lf);
    }
    lf.loop_filter_level[0] = r.f(6)? as u8;
    lf.loop_filter_level[1] = r.f(6)? as u8;
    if seq.color_config.num_planes() > 1
        && (lf.loop_filter_level[0] != 0 || lf.loop_filter_level[1] != 0)
    {
        lf.loop_filter_level[2] = r.f(6)? as u8;
        lf.loop_filter_level[3] = r.f(6)? as u8;
    }
    lf.loop_filter_sharpness = r.f(3)? as u8;
    lf.loop_filter_delta_enabled = r.flag()?;
    if lf.loop_filter_delta_enabled {
        lf.loop_filter_delta_update = r.flag()?;
        if lf.loop_filter_delta_update {
            for i in 0..TOTAL_REFS_PER_FRAME {
                if r.flag()? {
                    lf.update_ref_delta |= 1 << i;
                    lf.loop_filter_ref_deltas[i] = r.su(7)? as i8;
                }
            }
            for i in 0..2 {
                if r.flag()? {
                    lf.update_mode_delta |= 1 << i;
                    lf.loop_filter_mode_deltas[i] = r.su(7)? as i8;
                }
            }
        }
    }
    Ok(lf)
}

/// `cdef_params()` (5.9.19).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CdefParams {
    pub cdef_damping_minus_3: u8,
    pub cdef_bits: u8,
    pub cdef_y_pri_strength: [u8; 8],
    /// Coded value (0..=3), as Vulkan takes it; the effective strength
    /// maps 3 to 4.
    pub cdef_y_sec_strength: [u8; 8],
    pub cdef_uv_pri_strength: [u8; 8],
    /// Coded value, like `cdef_y_sec_strength`.
    pub cdef_uv_sec_strength: [u8; 8],
}

pub(crate) fn parse_cdef(
    r: &mut BitReader,
    seq: &SequenceHeader,
    coded_lossless: bool,
    allow_intrabc: bool,
) -> Result<CdefParams> {
    let mut c = CdefParams::default();
    if coded_lossless || allow_intrabc || !seq.enable_cdef {
        return Ok(c);
    }
    c.cdef_damping_minus_3 = r.f(2)? as u8;
    c.cdef_bits = r.f(2)? as u8;
    for i in 0..(1usize << c.cdef_bits) {
        c.cdef_y_pri_strength[i] = r.f(4)? as u8;
        c.cdef_y_sec_strength[i] = r.f(2)? as u8;
        if seq.color_config.num_planes() > 1 {
            c.cdef_uv_pri_strength[i] = r.f(4)? as u8;
            c.cdef_uv_sec_strength[i] = r.f(2)? as u8;
        }
    }
    Ok(c)
}

/// `lr_params()` (5.9.20).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoopRestorationParams {
    /// `FrameRestorationType` per plane: 0 none, 1 Wiener, 2 self-guided,
    /// 3 switchable (the same numbering as Vulkan's enum).
    pub frame_restoration_type: [u8; 3],
    /// `LoopRestorationSize` per plane, in samples (32..=256). Vulkan takes
    /// `log2(size) - 5`; see [`Self::size_log2_minus_5`].
    pub loop_restoration_size: [u16; 3],
    pub uses_lr: bool,
    pub uses_chroma_lr: bool,
    /// The derived `lr_unit_shift` (after the extra shift).
    pub lr_unit_shift: u8,
    pub lr_uv_shift: u8,
}

impl LoopRestorationParams {
    pub fn size_log2_minus_5(&self) -> [u16; 3] {
        self.loop_restoration_size
            .map(|s| (s.max(32).trailing_zeros() as u16).saturating_sub(5))
    }
}

/// Maps the coded `lr_type` to `FrameRestorationType`.
const REMAP_LR_TYPE: [u8; 4] = [0, 3, 1, 2];

pub(crate) fn parse_lr(
    r: &mut BitReader,
    seq: &SequenceHeader,
    all_lossless: bool,
    allow_intrabc: bool,
) -> Result<LoopRestorationParams> {
    let mut lr = LoopRestorationParams::default();
    if all_lossless || allow_intrabc || !seq.enable_restoration {
        return Ok(lr);
    }
    for i in 0..seq.color_config.num_planes() {
        let ty = REMAP_LR_TYPE[r.f(2)? as usize];
        lr.frame_restoration_type[i] = ty;
        if ty != 0 {
            lr.uses_lr = true;
            if i > 0 {
                lr.uses_chroma_lr = true;
            }
        }
    }
    if lr.uses_lr {
        let mut shift = r.f(1)? as u8;
        if seq.use_128x128_superblock {
            shift += 1;
        } else if shift != 0 {
            shift += r.f(1)? as u8;
        }
        lr.lr_unit_shift = shift;
        let size = RESTORATION_TILESIZE_MAX >> (2 - shift);
        let cc = &seq.color_config;
        if cc.subsampling_x && cc.subsampling_y && lr.uses_chroma_lr {
            lr.lr_uv_shift = r.f(1)? as u8;
        }
        lr.loop_restoration_size = [size, size >> lr.lr_uv_shift, size >> lr.lr_uv_shift];
    }
    Ok(lr)
}

/// `global_motion_params()` (5.9.24), indexed by reference frame name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalMotionParams {
    pub gm_type: [u8; TOTAL_REFS_PER_FRAME],
    pub gm_params: [[i32; 6]; TOTAL_REFS_PER_FRAME],
}

impl Default for GlobalMotionParams {
    fn default() -> Self {
        Self {
            gm_type: [IDENTITY; TOTAL_REFS_PER_FRAME],
            gm_params: [DEFAULT_GM_PARAMS; TOTAL_REFS_PER_FRAME],
        }
    }
}

/// `decode_subexp(numSyms)` (5.9.27).
fn decode_subexp(r: &mut BitReader, num_syms: i32) -> Result<i32> {
    let mut i = 0;
    let mut mk = 0;
    let k = 3;
    loop {
        let b2 = if i != 0 { k + i - 1 } else { k };
        let a = 1 << b2;
        if num_syms <= mk + 3 * a {
            return Ok(r.ns((num_syms - mk) as u32)? as i32 + mk);
        }
        if r.flag()? {
            i += 1;
            mk += a;
        } else {
            return Ok(r.f(b2 as u32)? as i32 + mk);
        }
    }
}

/// `inverse_recenter(r, v)` (5.9.28).
fn inverse_recenter(r: i32, v: i32) -> i32 {
    if v > 2 * r {
        v
    } else if v & 1 != 0 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

/// `decode_signed_subexp_with_ref(low, high, r)` (5.9.26).
fn decode_signed_subexp_with_ref(
    reader: &mut BitReader,
    low: i32,
    high: i32,
    r: i32,
) -> Result<i32> {
    let mx = high - low;
    let r = r - low;
    let v = decode_subexp(reader, mx)?;
    let x = if (r << 1) <= mx {
        inverse_recenter(r, v)
    } else {
        mx - 1 - inverse_recenter(mx - 1 - r, v)
    };
    Ok(x + low)
}

/// Parse `global_motion_params()` against `prev`, the `PrevGmParams` from
/// the primary reference frame (or the defaults).
pub(crate) fn parse_global_motion(
    r: &mut BitReader,
    frame_is_intra: bool,
    allow_high_precision_mv: bool,
    prev: &[[i32; 6]; TOTAL_REFS_PER_FRAME],
) -> Result<GlobalMotionParams> {
    let mut gm = GlobalMotionParams::default();
    if frame_is_intra {
        return Ok(gm);
    }
    #[allow(clippy::needless_range_loop)] // rf indexes three arrays.
    for rf in LAST_FRAME..=ALTREF_FRAME {
        let ty = if r.flag()? {
            if r.flag()? {
                ROTZOOM
            } else if r.flag()? {
                TRANSLATION
            } else {
                AFFINE
            }
        } else {
            IDENTITY
        };
        gm.gm_type[rf] = ty;
        let mut read = |idx: usize, gm: &mut GlobalMotionParams| -> Result<()> {
            let (abs_bits, prec_bits) = if idx < 2 {
                if ty == TRANSLATION {
                    let hp = !allow_high_precision_mv as u32;
                    (GM_ABS_TRANS_ONLY_BITS - hp, GM_TRANS_ONLY_PREC_BITS - hp)
                } else {
                    (GM_ABS_TRANS_BITS, GM_TRANS_PREC_BITS)
                }
            } else {
                (GM_ABS_ALPHA_BITS, GM_ALPHA_PREC_BITS)
            };
            let prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
            let (round, sub) = if idx % 3 == 2 {
                (1 << WARPEDMODEL_PREC_BITS, 1 << prec_bits)
            } else {
                (0, 0)
            };
            let mx = 1 << abs_bits;
            let reference = (prev[rf][idx] >> prec_diff) - sub;
            let value = decode_signed_subexp_with_ref(r, -mx, mx + 1, reference)?;
            gm.gm_params[rf][idx] = (value << prec_diff) + round;
            Ok(())
        };
        if ty >= ROTZOOM {
            read(2, &mut gm)?;
            read(3, &mut gm)?;
            if ty == AFFINE {
                read(4, &mut gm)?;
                read(5, &mut gm)?;
            } else {
                gm.gm_params[rf][4] = -gm.gm_params[rf][3];
                gm.gm_params[rf][5] = gm.gm_params[rf][2];
            }
        }
        if ty >= TRANSLATION {
            read(0, &mut gm)?;
            read(1, &mut gm)?;
        }
    }
    Ok(gm)
}

/// `film_grain_params()` (5.9.30). After `load_grain_params()` this holds
/// the referenced frame's parameters with this frame's `grain_seed`, which
/// is what Vulkan expects the application to supply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilmGrainParams {
    pub apply_grain: bool,
    pub grain_seed: u16,
    /// As coded on this frame (Vulkan ignores it).
    pub update_grain: bool,
    /// As coded on this frame when `update_grain` is 0 (Vulkan ignores it).
    pub film_grain_params_ref_idx: u8,
    pub num_y_points: u8,
    pub point_y_value: [u8; 14],
    pub point_y_scaling: [u8; 14],
    pub chroma_scaling_from_luma: bool,
    pub num_cb_points: u8,
    pub point_cb_value: [u8; 10],
    pub point_cb_scaling: [u8; 10],
    pub num_cr_points: u8,
    pub point_cr_value: [u8; 10],
    pub point_cr_scaling: [u8; 10],
    pub grain_scaling_minus_8: u8,
    pub ar_coeff_lag: u8,
    pub ar_coeffs_y_plus_128: [u8; 24],
    pub ar_coeffs_cb_plus_128: [u8; 25],
    pub ar_coeffs_cr_plus_128: [u8; 25],
    pub ar_coeff_shift_minus_6: u8,
    pub grain_scale_shift: u8,
    pub cb_mult: u8,
    pub cb_luma_mult: u8,
    pub cb_offset: u16,
    pub cr_mult: u8,
    pub cr_luma_mult: u8,
    pub cr_offset: u16,
    pub overlap_flag: bool,
    pub clip_to_restricted_range: bool,
}

impl Default for FilmGrainParams {
    /// `reset_grain_params()`: everything zero.
    fn default() -> Self {
        Self {
            apply_grain: false,
            grain_seed: 0,
            update_grain: false,
            film_grain_params_ref_idx: 0,
            num_y_points: 0,
            point_y_value: [0; 14],
            point_y_scaling: [0; 14],
            chroma_scaling_from_luma: false,
            num_cb_points: 0,
            point_cb_value: [0; 10],
            point_cb_scaling: [0; 10],
            num_cr_points: 0,
            point_cr_value: [0; 10],
            point_cr_scaling: [0; 10],
            grain_scaling_minus_8: 0,
            ar_coeff_lag: 0,
            ar_coeffs_y_plus_128: [0; 24],
            ar_coeffs_cb_plus_128: [0; 25],
            ar_coeffs_cr_plus_128: [0; 25],
            ar_coeff_shift_minus_6: 0,
            grain_scale_shift: 0,
            cb_mult: 0,
            cb_luma_mult: 0,
            cb_offset: 0,
            cr_mult: 0,
            cr_luma_mult: 0,
            cr_offset: 0,
            overlap_flag: false,
            clip_to_restricted_range: false,
        }
    }
}

fn read_points(r: &mut BitReader, max: u8, values: &mut [u8], scalings: &mut [u8]) -> Result<u8> {
    let n = r.f(4)? as u8;
    if n > max {
        return Err(PixelForgeError::InvalidInput(format!(
            "AV1: {n} film grain scaling points, at most {max} allowed"
        )));
    }
    for i in 0..n as usize {
        values[i] = r.f(8)? as u8;
        scalings[i] = r.f(8)? as u8;
    }
    Ok(n)
}

/// What `film_grain_params()` needs to know about the frame.
pub(crate) struct GrainContext {
    pub show_frame: bool,
    pub showable_frame: bool,
    pub is_inter_frame: bool,
}

/// Parse `film_grain_params()`. `load` performs `load_grain_params(idx)`,
/// returning `None` when slot `idx` holds no valid frame.
pub(crate) fn parse_film_grain(
    r: &mut BitReader,
    seq: &SequenceHeader,
    ctx: &GrainContext,
    load: impl Fn(u8) -> Option<FilmGrainParams>,
) -> Result<std::result::Result<FilmGrainParams, u8>> {
    if !seq.film_grain_params_present || (!ctx.show_frame && !ctx.showable_frame) {
        return Ok(Ok(FilmGrainParams::default()));
    }
    let mut g = FilmGrainParams {
        apply_grain: r.flag()?,
        ..Default::default()
    };
    if !g.apply_grain {
        return Ok(Ok(FilmGrainParams::default()));
    }
    g.grain_seed = r.f(16)? as u16;
    g.update_grain = if ctx.is_inter_frame { r.flag()? } else { true };
    if !g.update_grain {
        let idx = r.f(3)? as u8;
        let Some(mut loaded) = load(idx) else {
            return Ok(Err(idx));
        };
        loaded.grain_seed = g.grain_seed;
        loaded.update_grain = false;
        loaded.film_grain_params_ref_idx = idx;
        return Ok(Ok(loaded));
    }
    g.num_y_points = read_points(r, 14, &mut g.point_y_value, &mut g.point_y_scaling)?;
    let cc = &seq.color_config;
    g.chroma_scaling_from_luma = if cc.mono_chrome { false } else { r.flag()? };
    if !(cc.mono_chrome
        || g.chroma_scaling_from_luma
        || (cc.subsampling_x && cc.subsampling_y && g.num_y_points == 0))
    {
        g.num_cb_points = read_points(r, 10, &mut g.point_cb_value, &mut g.point_cb_scaling)?;
        g.num_cr_points = read_points(r, 10, &mut g.point_cr_value, &mut g.point_cr_scaling)?;
    }
    g.grain_scaling_minus_8 = r.f(2)? as u8;
    g.ar_coeff_lag = r.f(2)? as u8;
    let num_pos_luma = 2 * g.ar_coeff_lag as usize * (g.ar_coeff_lag as usize + 1);
    let num_pos_chroma = if g.num_y_points != 0 {
        for i in 0..num_pos_luma {
            g.ar_coeffs_y_plus_128[i] = r.f(8)? as u8;
        }
        num_pos_luma + 1
    } else {
        num_pos_luma
    };
    if g.chroma_scaling_from_luma || g.num_cb_points != 0 {
        for i in 0..num_pos_chroma {
            g.ar_coeffs_cb_plus_128[i] = r.f(8)? as u8;
        }
    }
    if g.chroma_scaling_from_luma || g.num_cr_points != 0 {
        for i in 0..num_pos_chroma {
            g.ar_coeffs_cr_plus_128[i] = r.f(8)? as u8;
        }
    }
    g.ar_coeff_shift_minus_6 = r.f(2)? as u8;
    g.grain_scale_shift = r.f(2)? as u8;
    if g.num_cb_points != 0 {
        g.cb_mult = r.f(8)? as u8;
        g.cb_luma_mult = r.f(8)? as u8;
        g.cb_offset = r.f(9)? as u16;
    }
    if g.num_cr_points != 0 {
        g.cr_mult = r.f(8)? as u8;
        g.cr_luma_mult = r.f(8)? as u8;
        g.cr_offset = r.f(9)? as u16;
    }
    g.overlap_flag = r.flag()?;
    g.clip_to_restricted_range = r.flag()?;
    Ok(Ok(g))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_recenter_matches_the_spec_table() {
        // v above 2r passes through; odd v steps below r, even v above it.
        assert_eq!(inverse_recenter(2, 5), 5);
        assert_eq!(inverse_recenter(2, 1), 1);
        assert_eq!(inverse_recenter(2, 3), 0);
        assert_eq!(inverse_recenter(2, 2), 3);
        assert_eq!(inverse_recenter(2, 0), 2);
    }

    #[test]
    fn subexp_reads_the_reference_back_for_a_zero_code() {
        // decode_subexp reading 0 (more_bits=0, then f(3)=000) recenters to
        // the reference itself.
        let mut r = BitReader::new(&[0x00]);
        let v = decode_signed_subexp_with_ref(&mut r, -4096, 4097, 17).unwrap();
        assert_eq!(v, 17);
        assert_eq!(r.position(), 4);
    }

    #[test]
    fn loop_restoration_sizes_convert_to_vulkan_form() {
        let lr = LoopRestorationParams {
            loop_restoration_size: [256, 128, 64],
            ..Default::default()
        };
        assert_eq!(lr.size_log2_minus_5(), [3, 2, 1]);
    }

    #[test]
    fn film_grain_reuse_loads_the_reference_with_a_new_seed() {
        let seq = SequenceHeader {
            film_grain_params_present: true,
            ..Default::default()
        };
        let ctx = GrainContext {
            show_frame: true,
            showable_frame: true,
            is_inter_frame: true,
        };
        // apply_grain 1, grain_seed 0xbeef, update_grain 0, ref idx 3.
        let bits = [0b1101_1111, 0b0111_0111, 0b1001_1000];
        let saved = FilmGrainParams {
            apply_grain: true,
            grain_seed: 1,
            update_grain: true,
            num_y_points: 5,
            ..Default::default()
        };
        let mut r = BitReader::new(&bits);
        let loaded = parse_film_grain(&mut r, &seq, &ctx, |idx| {
            assert_eq!(idx, 3);
            Some(saved.clone())
        })
        .unwrap()
        .unwrap();
        assert_eq!(r.position(), 21);
        assert_eq!(loaded.grain_seed, 0xbeef);
        assert_eq!(loaded.num_y_points, 5);
        assert!(!loaded.update_grain);
        assert_eq!(loaded.film_grain_params_ref_idx, 3);

        // An empty slot comes back as the slot to report missing.
        let mut r = BitReader::new(&bits);
        let missing = parse_film_grain(&mut r, &seq, &ctx, |_| None).unwrap();
        assert_eq!(missing, Err(3));
    }

    #[test]
    fn feature_masks_pack_per_segment() {
        let mut s = SegmentationParams::default();
        s.feature_enabled[0][0] = true;
        s.feature_enabled[0][5] = true;
        s.feature_enabled[7][7] = true;
        let masks = s.feature_enabled_masks();
        assert_eq!(masks[0], 0b0010_0001);
        assert_eq!(masks[7], 0b1000_0000);
    }
}
