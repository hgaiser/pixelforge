//! Conversion of parsed headers to the Vulkan Video Std structures for AV1.
//!
//! The Std structures point to each other. Each builder here owns every
//! structure it points to, boxed so the pointers stay valid when the builder
//! moves; the structure it hands out is valid for as long as the builder
//! lives.

use ash::vk::native::*;

use super::consts::{NUM_REF_FRAMES, REFS_PER_FRAME};
use super::frame_header::FrameHeader;
use super::params::{FilmGrainParams, GlobalMotionParams};
use super::refs::RefSlot;
use super::sequence::SequenceHeader;

fn zeroed<T>() -> T {
    // SAFETY: only used for the plain-data Std structures, for which all
    // zero bits are a valid value (null pointers, zero numbers and flags).
    unsafe { std::mem::zeroed() }
}

/// `StdVideoAV1SequenceHeader` and the structures it points to.
pub struct StdSequenceHeader {
    header: StdVideoAV1SequenceHeader,
    _color: Box<StdVideoAV1ColorConfig>,
    _timing: Option<Box<StdVideoAV1TimingInfo>>,
}

// SAFETY: the raw pointers only point into the boxes this value owns.
unsafe impl Send for StdSequenceHeader {}

impl StdSequenceHeader {
    pub fn new(seq: &SequenceHeader) -> Self {
        let cc = &seq.color_config;
        let mut color: Box<StdVideoAV1ColorConfig> = Box::new(zeroed());
        color.flags.set_mono_chrome(cc.mono_chrome as u32);
        color.flags.set_color_range(cc.color_range as u32);
        color
            .flags
            .set_separate_uv_delta_q(cc.separate_uv_delta_q as u32);
        color
            .flags
            .set_color_description_present_flag(cc.color_description_present_flag as u32);
        color.BitDepth = cc.bit_depth;
        color.subsampling_x = cc.subsampling_x as u8;
        color.subsampling_y = cc.subsampling_y as u8;
        color.color_primaries = cc.color_primaries as u32;
        color.transfer_characteristics = cc.transfer_characteristics as u32;
        color.matrix_coefficients = cc.matrix_coefficients as u32;
        color.chroma_sample_position = cc.chroma_sample_position as u32;

        let timing = seq.timing_info.as_ref().map(|t| {
            let mut timing: Box<StdVideoAV1TimingInfo> = Box::new(zeroed());
            timing
                .flags
                .set_equal_picture_interval(t.equal_picture_interval as u32);
            timing.num_units_in_display_tick = t.num_units_in_display_tick;
            timing.time_scale = t.time_scale;
            timing.num_ticks_per_picture_minus_1 = t.num_ticks_per_picture_minus_1;
            timing
        });

        let mut header: StdVideoAV1SequenceHeader = zeroed();
        let f = &mut header.flags;
        f.set_still_picture(seq.still_picture as u32);
        f.set_reduced_still_picture_header(seq.reduced_still_picture_header as u32);
        f.set_use_128x128_superblock(seq.use_128x128_superblock as u32);
        f.set_enable_filter_intra(seq.enable_filter_intra as u32);
        f.set_enable_intra_edge_filter(seq.enable_intra_edge_filter as u32);
        f.set_enable_interintra_compound(seq.enable_interintra_compound as u32);
        f.set_enable_masked_compound(seq.enable_masked_compound as u32);
        f.set_enable_warped_motion(seq.enable_warped_motion as u32);
        f.set_enable_dual_filter(seq.enable_dual_filter as u32);
        f.set_enable_order_hint(seq.enable_order_hint as u32);
        f.set_enable_jnt_comp(seq.enable_jnt_comp as u32);
        f.set_enable_ref_frame_mvs(seq.enable_ref_frame_mvs as u32);
        f.set_frame_id_numbers_present_flag(seq.frame_id_numbers_present_flag as u32);
        f.set_enable_superres(seq.enable_superres as u32);
        f.set_enable_cdef(seq.enable_cdef as u32);
        f.set_enable_restoration(seq.enable_restoration as u32);
        f.set_film_grain_params_present(seq.film_grain_params_present as u32);
        f.set_timing_info_present_flag(seq.timing_info.is_some() as u32);
        f.set_initial_display_delay_present_flag(seq.initial_display_delay_present_flag as u32);
        header.seq_profile = seq.seq_profile as u32;
        header.frame_width_bits_minus_1 = seq.frame_width_bits_minus_1;
        header.frame_height_bits_minus_1 = seq.frame_height_bits_minus_1;
        header.max_frame_width_minus_1 = seq.max_frame_width_minus_1;
        header.max_frame_height_minus_1 = seq.max_frame_height_minus_1;
        header.delta_frame_id_length_minus_2 = seq.delta_frame_id_length_minus_2;
        header.additional_frame_id_length_minus_1 = seq.additional_frame_id_length_minus_1;
        header.order_hint_bits_minus_1 = seq.order_hint_bits_minus_1();
        header.seq_force_integer_mv = seq.seq_force_integer_mv;
        header.seq_force_screen_content_tools = seq.seq_force_screen_content_tools;
        header.pColorConfig = &*color;
        header.pTimingInfo = timing.as_deref().map_or(std::ptr::null(), |t| t);
        Self {
            header,
            _color: color,
            _timing: timing,
        }
    }

    pub fn get(&self) -> &StdVideoAV1SequenceHeader {
        &self.header
    }
}

fn std_global_motion(gm: &GlobalMotionParams) -> StdVideoAV1GlobalMotion {
    StdVideoAV1GlobalMotion {
        GmType: gm.gm_type,
        gm_params: gm.gm_params,
    }
}

fn std_film_grain(g: &FilmGrainParams) -> StdVideoAV1FilmGrain {
    let mut s: StdVideoAV1FilmGrain = zeroed();
    s.flags
        .set_chroma_scaling_from_luma(g.chroma_scaling_from_luma as u32);
    s.flags.set_overlap_flag(g.overlap_flag as u32);
    s.flags
        .set_clip_to_restricted_range(g.clip_to_restricted_range as u32);
    s.flags.set_update_grain(g.update_grain as u32);
    s.grain_scaling_minus_8 = g.grain_scaling_minus_8;
    s.ar_coeff_lag = g.ar_coeff_lag;
    s.ar_coeff_shift_minus_6 = g.ar_coeff_shift_minus_6;
    s.grain_scale_shift = g.grain_scale_shift;
    s.grain_seed = g.grain_seed;
    s.film_grain_params_ref_idx = g.film_grain_params_ref_idx;
    s.num_y_points = g.num_y_points;
    s.point_y_value = g.point_y_value;
    s.point_y_scaling = g.point_y_scaling;
    s.num_cb_points = g.num_cb_points;
    s.point_cb_value = g.point_cb_value;
    s.point_cb_scaling = g.point_cb_scaling;
    s.num_cr_points = g.num_cr_points;
    s.point_cr_value = g.point_cr_value;
    s.point_cr_scaling = g.point_cr_scaling;
    // The Std header declares these int8_t although they hold 0..=255.
    s.ar_coeffs_y_plus_128 = g.ar_coeffs_y_plus_128.map(|v| v as i8);
    s.ar_coeffs_cb_plus_128 = g.ar_coeffs_cb_plus_128.map(|v| v as i8);
    s.ar_coeffs_cr_plus_128 = g.ar_coeffs_cr_plus_128.map(|v| v as i8);
    s.cb_mult = g.cb_mult;
    s.cb_luma_mult = g.cb_luma_mult;
    s.cb_offset = g.cb_offset;
    s.cr_mult = g.cr_mult;
    s.cr_luma_mult = g.cr_luma_mult;
    s.cr_offset = g.cr_offset;
    s
}

/// The arrays a `StdVideoAV1TileInfo` points to.
struct TileArrays {
    mi_col_starts: Box<[u16]>,
    mi_row_starts: Box<[u16]>,
    width_in_sbs_minus_1: Box<[u16]>,
    height_in_sbs_minus_1: Box<[u16]>,
}

/// `StdVideoDecodeAV1PictureInfo` and the structures it points to.
pub struct StdPictureInfo {
    info: StdVideoDecodeAV1PictureInfo,
    _tile_info: Box<StdVideoAV1TileInfo>,
    _tile_arrays: TileArrays,
    _quantization: Box<StdVideoAV1Quantization>,
    _segmentation: Box<StdVideoAV1Segmentation>,
    _loop_filter: Box<StdVideoAV1LoopFilter>,
    _cdef: Box<StdVideoAV1CDEF>,
    _loop_restoration: Box<StdVideoAV1LoopRestoration>,
    _global_motion: Box<StdVideoAV1GlobalMotion>,
    _film_grain: Box<StdVideoAV1FilmGrain>,
}

impl StdPictureInfo {
    /// Build the picture info for a frame (not a `show_existing_frame`,
    /// which is not decoded).
    pub fn new(h: &FrameHeader) -> Self {
        let size = &h.frame_size;
        let ti = &h.tile_info;
        let tile_arrays = TileArrays {
            // Vulkan takes TileCols / TileRows entries, without the final
            // MiCols / MiRows the spec's arrays end with.
            mi_col_starts: ti.mi_col_starts[..ti.tile_cols as usize].into(),
            mi_row_starts: ti.mi_row_starts[..ti.tile_rows as usize].into(),
            width_in_sbs_minus_1: ti.width_in_sbs_minus_1.clone().into(),
            height_in_sbs_minus_1: ti.height_in_sbs_minus_1.clone().into(),
        };
        let mut tile_info: Box<StdVideoAV1TileInfo> = Box::new(zeroed());
        tile_info
            .flags
            .set_uniform_tile_spacing_flag(ti.uniform_tile_spacing_flag as u32);
        tile_info.TileCols = ti.tile_cols;
        tile_info.TileRows = ti.tile_rows;
        tile_info.context_update_tile_id = ti.context_update_tile_id;
        tile_info.tile_size_bytes_minus_1 = ti.tile_size_bytes_minus_1;
        tile_info.pMiColStarts = tile_arrays.mi_col_starts.as_ptr();
        tile_info.pMiRowStarts = tile_arrays.mi_row_starts.as_ptr();
        tile_info.pWidthInSbsMinus1 = tile_arrays.width_in_sbs_minus_1.as_ptr();
        tile_info.pHeightInSbsMinus1 = tile_arrays.height_in_sbs_minus_1.as_ptr();

        let q = &h.quantization;
        let mut quantization: Box<StdVideoAV1Quantization> = Box::new(zeroed());
        quantization.flags.set_using_qmatrix(q.using_qmatrix as u32);
        quantization.flags.set_diff_uv_delta(q.diff_uv_delta as u32);
        quantization.base_q_idx = q.base_q_idx;
        quantization.DeltaQYDc = q.delta_q_y_dc;
        quantization.DeltaQUDc = q.delta_q_u_dc;
        quantization.DeltaQUAc = q.delta_q_u_ac;
        quantization.DeltaQVDc = q.delta_q_v_dc;
        quantization.DeltaQVAc = q.delta_q_v_ac;
        quantization.qm_y = q.qm_y;
        quantization.qm_u = q.qm_u;
        quantization.qm_v = q.qm_v;

        let segmentation = Box::new(StdVideoAV1Segmentation {
            FeatureEnabled: h.segmentation.feature_enabled_masks(),
            FeatureData: h.segmentation.feature_data,
        });

        let lf = &h.loop_filter;
        let mut loop_filter: Box<StdVideoAV1LoopFilter> = Box::new(zeroed());
        loop_filter
            .flags
            .set_loop_filter_delta_enabled(lf.loop_filter_delta_enabled as u32);
        loop_filter
            .flags
            .set_loop_filter_delta_update(lf.loop_filter_delta_update as u32);
        loop_filter.loop_filter_level = lf.loop_filter_level;
        loop_filter.loop_filter_sharpness = lf.loop_filter_sharpness;
        loop_filter.update_ref_delta = lf.update_ref_delta;
        loop_filter.loop_filter_ref_deltas = lf.loop_filter_ref_deltas;
        loop_filter.update_mode_delta = lf.update_mode_delta;
        loop_filter.loop_filter_mode_deltas = lf.loop_filter_mode_deltas;

        let c = &h.cdef;
        let cdef = Box::new(StdVideoAV1CDEF {
            cdef_damping_minus_3: c.cdef_damping_minus_3,
            cdef_bits: c.cdef_bits,
            cdef_y_pri_strength: c.cdef_y_pri_strength,
            cdef_y_sec_strength: c.cdef_y_sec_strength,
            cdef_uv_pri_strength: c.cdef_uv_pri_strength,
            cdef_uv_sec_strength: c.cdef_uv_sec_strength,
        });

        let lr = &h.loop_restoration;
        let loop_restoration = Box::new(StdVideoAV1LoopRestoration {
            FrameRestorationType: lr.frame_restoration_type.map(|t| t as u32),
            LoopRestorationSize: lr.size_log2_minus_5(),
        });
        let global_motion = Box::new(std_global_motion(&h.global_motion));
        let film_grain = Box::new(std_film_grain(&h.film_grain));

        let mut info: StdVideoDecodeAV1PictureInfo = zeroed();
        let f = &mut info.flags;
        f.set_error_resilient_mode(h.error_resilient_mode as u32);
        f.set_disable_cdf_update(h.disable_cdf_update as u32);
        f.set_use_superres(size.use_superres as u32);
        f.set_render_and_frame_size_different(size.render_and_frame_size_different as u32);
        f.set_allow_screen_content_tools(h.allow_screen_content_tools as u32);
        f.set_is_filter_switchable(h.is_filter_switchable as u32);
        f.set_force_integer_mv(h.force_integer_mv as u32);
        f.set_frame_size_override_flag(h.frame_size_override_flag as u32);
        f.set_buffer_removal_time_present_flag(h.buffer_removal_time_present_flag as u32);
        f.set_allow_intrabc(h.allow_intrabc as u32);
        f.set_frame_refs_short_signaling(h.frame_refs_short_signaling as u32);
        f.set_allow_high_precision_mv(h.allow_high_precision_mv as u32);
        f.set_is_motion_mode_switchable(h.is_motion_mode_switchable as u32);
        f.set_use_ref_frame_mvs(h.use_ref_frame_mvs as u32);
        f.set_disable_frame_end_update_cdf(h.disable_frame_end_update_cdf as u32);
        f.set_allow_warped_motion(h.allow_warped_motion as u32);
        f.set_reduced_tx_set(h.reduced_tx_set as u32);
        f.set_reference_select(h.reference_select as u32);
        f.set_skip_mode_present(h.skip_mode_present as u32);
        f.set_delta_q_present(h.delta.delta_q_present as u32);
        f.set_delta_lf_present(h.delta.delta_lf_present as u32);
        f.set_delta_lf_multi(h.delta.delta_lf_multi as u32);
        f.set_segmentation_enabled(h.segmentation.segmentation_enabled as u32);
        f.set_segmentation_update_map(h.segmentation.segmentation_update_map as u32);
        f.set_segmentation_temporal_update(h.segmentation.segmentation_temporal_update as u32);
        f.set_segmentation_update_data(h.segmentation.segmentation_update_data as u32);
        f.set_UsesLr(lr.uses_lr as u32);
        f.set_usesChromaLr(lr.uses_chroma_lr as u32);
        f.set_apply_grain(h.film_grain.apply_grain as u32);
        info.frame_type = h.frame_type as u32;
        info.current_frame_id = h.current_frame_id;
        info.OrderHint = h.order_hint;
        info.primary_ref_frame = h.primary_ref_frame;
        info.refresh_frame_flags = h.refresh_frame_flags;
        info.interpolation_filter = h.interpolation_filter as u32;
        info.TxMode = h.tx_mode as u32;
        info.delta_q_res = h.delta.delta_q_res;
        info.delta_lf_res = h.delta.delta_lf_res;
        info.SkipModeFrame = h.skip_mode_frame;
        info.coded_denom = size.coded_denom;
        info.OrderHints = h.order_hints;
        // Section 5.9.2 indexes expectedFrameId by i, for LAST_FRAME + i.
        info.expectedFrameId[..REFS_PER_FRAME].copy_from_slice(&h.expected_frame_id);
        info.pTileInfo = &*tile_info;
        info.pQuantization = &*quantization;
        info.pSegmentation = &*segmentation;
        info.pLoopFilter = &*loop_filter;
        info.pCDEF = &*cdef;
        info.pLoopRestoration = &*loop_restoration;
        info.pGlobalMotion = &*global_motion;
        info.pFilmGrain = &*film_grain;

        Self {
            info,
            _tile_info: tile_info,
            _tile_arrays: tile_arrays,
            _quantization: quantization,
            _segmentation: segmentation,
            _loop_filter: loop_filter,
            _cdef: cdef,
            _loop_restoration: loop_restoration,
            _global_motion: global_motion,
            _film_grain: film_grain,
        }
    }

    pub fn get(&self) -> &StdVideoDecodeAV1PictureInfo {
        &self.info
    }
}

/// `StdVideoDecodeAV1ReferenceInfo` for a reference slot (or, after the
/// update, for the frame just decoded).
pub fn std_reference_info(slot: &RefSlot) -> StdVideoDecodeAV1ReferenceInfo {
    let mut info: StdVideoDecodeAV1ReferenceInfo = zeroed();
    info.flags
        .set_disable_frame_end_update_cdf(slot.disable_frame_end_update_cdf as u32);
    info.flags
        .set_segmentation_enabled(slot.segmentation_enabled as u32);
    info.frame_type = slot.frame_type as u8;
    info.RefFrameSignBias = slot.ref_frame_sign_bias;
    info.OrderHint = slot.order_hint;
    info.SavedOrderHints = slot.saved_order_hints;
    info
}

/// `VkVideoDecodeAV1PictureInfoKHR::referenceNameSlotIndices`: the DPB slot
/// of each of LAST_FRAME..=ALTREF_FRAME, or -1. Intra frames use none.
pub fn reference_name_slot_indices(
    header: &FrameHeader,
    slots: &[RefSlot; NUM_REF_FRAMES],
) -> [i32; REFS_PER_FRAME] {
    if header.frame_is_intra() {
        return [-1; REFS_PER_FRAME];
    }
    header.ref_frame_idx.map(|idx| {
        slots[idx as usize]
            .dpb_slot
            .map_or(-1, |dpb_slot| dpb_slot as i32)
    })
}
