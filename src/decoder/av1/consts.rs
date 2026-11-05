//! Constants from section 3 of the AV1 specification, as the parser uses them.

pub const NUM_REF_FRAMES: usize = 8;
pub const REFS_PER_FRAME: usize = 7;
pub const TOTAL_REFS_PER_FRAME: usize = 8;
pub const PRIMARY_REF_NONE: u8 = 7;
pub const MAX_SEGMENTS: usize = 8;
pub const SEG_LVL_MAX: usize = 8;
pub const SEG_LVL_ALT_Q: usize = 0;
pub const SEG_LVL_REF_FRAME: usize = 5;
pub const MAX_LOOP_FILTER: i32 = 63;

pub const MAX_TILE_WIDTH: u32 = 4096;
pub const MAX_TILE_AREA: u32 = 4096 * 2304;
pub const MAX_TILE_ROWS: u32 = 64;
pub const MAX_TILE_COLS: u32 = 64;

pub const SUPERRES_NUM: u32 = 8;
pub const SUPERRES_DENOM_MIN: u32 = 9;
pub const SUPERRES_DENOM_BITS: u32 = 3;

pub const SELECT_SCREEN_CONTENT_TOOLS: u8 = 2;
pub const SELECT_INTEGER_MV: u8 = 2;

/// `interpolation_filter` value meaning "chosen per block".
pub const SWITCHABLE: u8 = 4;

pub const RESTORATION_TILESIZE_MAX: u16 = 256;

pub const WARPEDMODEL_PREC_BITS: u32 = 16;
pub const GM_ABS_ALPHA_BITS: u32 = 12;
pub const GM_ALPHA_PREC_BITS: u32 = 15;
pub const GM_ABS_TRANS_ONLY_BITS: u32 = 9;
pub const GM_TRANS_ONLY_PREC_BITS: u32 = 3;
pub const GM_ABS_TRANS_BITS: u32 = 12;
pub const GM_TRANS_PREC_BITS: u32 = 6;

/// Global motion types (`GmType`).
pub const IDENTITY: u8 = 0;
pub const TRANSLATION: u8 = 1;
pub const ROTZOOM: u8 = 2;
pub const AFFINE: u8 = 3;

/// Reference frame names (section 6.10.24), which index `OrderHints`,
/// `RefFrameSignBias`, `gm_params` and the loop filter reference deltas.
pub const LAST_FRAME: usize = 1;
pub const LAST2_FRAME: usize = 2;
pub const LAST3_FRAME: usize = 3;
pub const GOLDEN_FRAME: usize = 4;
pub const BWDREF_FRAME: usize = 5;
pub const ALTREF2_FRAME: usize = 6;
pub const ALTREF_FRAME: usize = 7;

/// `loop_filter_ref_deltas` after `setup_past_independence()`.
pub const DEFAULT_LOOP_FILTER_REF_DELTAS: [i8; TOTAL_REFS_PER_FRAME] = [1, 0, 0, 0, -1, 0, -1, -1];

/// Identity warp parameters: `1 << WARPEDMODEL_PREC_BITS` on the diagonal.
pub const DEFAULT_GM_PARAMS: [i32; 6] = [
    0,
    0,
    1 << WARPEDMODEL_PREC_BITS,
    0,
    0,
    1 << WARPEDMODEL_PREC_BITS,
];
