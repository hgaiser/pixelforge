//! Tile layout and tile group OBUs (AV1 specification sections 5.9.15 and
//! 5.11.1).

use super::bits::BitReader;
use super::consts::{MAX_TILE_AREA, MAX_TILE_COLS, MAX_TILE_ROWS, MAX_TILE_WIDTH};
use crate::error::{PixelForgeError, Result};

/// `tile_info()`, with the derived tile grid.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TileInfo {
    pub uniform_tile_spacing_flag: bool,
    pub tile_cols_log2: u8,
    pub tile_rows_log2: u8,
    pub tile_cols: u8,
    pub tile_rows: u8,
    /// `MiColStarts`, `tile_cols + 1` entries (the last is `MiCols`).
    pub mi_col_starts: Vec<u16>,
    /// `MiRowStarts`, `tile_rows + 1` entries (the last is `MiRows`).
    pub mi_row_starts: Vec<u16>,
    /// `width_in_sbs_minus_1` per tile column. Coded for non-uniform
    /// spacing; derived from the grid for uniform spacing, which is what
    /// Vulkan expects either way.
    pub width_in_sbs_minus_1: Vec<u16>,
    /// `height_in_sbs_minus_1` per tile row, like `width_in_sbs_minus_1`.
    pub height_in_sbs_minus_1: Vec<u16>,
    pub context_update_tile_id: u16,
    /// `TileSizeBytes - 1`; 3 (four-byte sizes) when not coded, which only
    /// happens with a single tile, where no size is ever read.
    pub tile_size_bytes_minus_1: u8,
}

impl TileInfo {
    /// `NumTiles`.
    pub fn num_tiles(&self) -> usize {
        self.tile_cols as usize * self.tile_rows as usize
    }

    /// `TileSizeBytes`.
    pub fn tile_size_bytes(&self) -> u32 {
        self.tile_size_bytes_minus_1 as u32 + 1
    }
}

/// `tile_log2(blkSize, target)`.
fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

/// Superblock sizes per tile from the superblock-unit starts.
fn sizes_minus_1(starts_sb: &[u32]) -> Vec<u16> {
    starts_sb
        .windows(2)
        .map(|w| (w[1] - w[0] - 1) as u16)
        .collect()
}

/// Parse `tile_info()` for a frame of `mi_cols` by `mi_rows` mode-info units.
pub(crate) fn parse_tile_info(
    r: &mut BitReader,
    use_128x128_superblock: bool,
    mi_cols: u32,
    mi_rows: u32,
) -> Result<TileInfo> {
    if mi_cols == 0 || mi_rows == 0 {
        return Err(PixelForgeError::InvalidInput(
            "AV1: frame has zero size".to_string(),
        ));
    }
    let (sb_cols, sb_rows, sb_shift) = if use_128x128_superblock {
        ((mi_cols + 31) >> 5, (mi_rows + 31) >> 5, 5)
    } else {
        ((mi_cols + 15) >> 4, (mi_rows + 15) >> 4, 4)
    };
    let sb_size = sb_shift + 2;
    let max_tile_width_sb = MAX_TILE_WIDTH >> sb_size;
    let mut max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size);
    let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
    let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));

    let mut info = TileInfo {
        uniform_tile_spacing_flag: r.flag()?,
        tile_size_bytes_minus_1: 3,
        ..Default::default()
    };
    // Tile starts in superblocks, ending with the frame size in superblocks.
    let mut col_starts_sb = Vec::new();
    let mut row_starts_sb = Vec::new();
    let (tile_cols_log2, tile_rows_log2);
    if info.uniform_tile_spacing_flag {
        let mut cols_log2 = min_log2_tile_cols;
        while cols_log2 < max_log2_tile_cols {
            if !r.flag()? {
                break;
            }
            cols_log2 += 1;
        }
        let tile_width_sb = (sb_cols + (1 << cols_log2) - 1) >> cols_log2;
        col_starts_sb.extend((0..sb_cols).step_by(tile_width_sb as usize));

        let min_log2_tile_rows = min_log2_tiles.saturating_sub(cols_log2);
        let mut rows_log2 = min_log2_tile_rows;
        while rows_log2 < max_log2_tile_rows {
            if !r.flag()? {
                break;
            }
            rows_log2 += 1;
        }
        let tile_height_sb = (sb_rows + (1 << rows_log2) - 1) >> rows_log2;
        row_starts_sb.extend((0..sb_rows).step_by(tile_height_sb as usize));
        tile_cols_log2 = cols_log2;
        tile_rows_log2 = rows_log2;
    } else {
        let mut widest_tile_sb = 0;
        let mut start_sb = 0;
        while start_sb < sb_cols {
            if col_starts_sb.len() as u32 >= MAX_TILE_COLS {
                return Err(PixelForgeError::InvalidInput(
                    "AV1: more than 64 tile columns".to_string(),
                ));
            }
            col_starts_sb.push(start_sb);
            let max_width = (sb_cols - start_sb).min(max_tile_width_sb);
            let size_sb = r.ns(max_width)? + 1;
            widest_tile_sb = widest_tile_sb.max(size_sb);
            start_sb += size_sb;
        }
        tile_cols_log2 = tile_log2(1, col_starts_sb.len() as u32);

        if min_log2_tiles > 0 {
            max_tile_area_sb = (sb_rows * sb_cols) >> (min_log2_tiles + 1);
        } else {
            max_tile_area_sb = sb_rows * sb_cols;
        }
        let max_tile_height_sb = (max_tile_area_sb / widest_tile_sb).max(1);
        let mut start_sb = 0;
        while start_sb < sb_rows {
            if row_starts_sb.len() as u32 >= MAX_TILE_ROWS {
                return Err(PixelForgeError::InvalidInput(
                    "AV1: more than 64 tile rows".to_string(),
                ));
            }
            row_starts_sb.push(start_sb);
            let max_height = (sb_rows - start_sb).min(max_tile_height_sb);
            start_sb += r.ns(max_height)? + 1;
        }
        tile_rows_log2 = tile_log2(1, row_starts_sb.len() as u32);
    }

    if col_starts_sb.len() as u32 > MAX_TILE_COLS || row_starts_sb.len() as u32 > MAX_TILE_ROWS {
        return Err(PixelForgeError::InvalidInput(
            "AV1: tile grid exceeds 64 columns or rows".to_string(),
        ));
    }
    info.tile_cols = col_starts_sb.len() as u8;
    info.tile_rows = row_starts_sb.len() as u8;
    info.tile_cols_log2 = tile_cols_log2 as u8;
    info.tile_rows_log2 = tile_rows_log2 as u8;
    info.mi_col_starts = col_starts_sb
        .iter()
        .map(|&s| (s << sb_shift) as u16)
        .chain([mi_cols as u16])
        .collect();
    info.mi_row_starts = row_starts_sb
        .iter()
        .map(|&s| (s << sb_shift) as u16)
        .chain([mi_rows as u16])
        .collect();
    col_starts_sb.push(sb_cols);
    row_starts_sb.push(sb_rows);
    info.width_in_sbs_minus_1 = sizes_minus_1(&col_starts_sb);
    info.height_in_sbs_minus_1 = sizes_minus_1(&row_starts_sb);

    if tile_cols_log2 > 0 || tile_rows_log2 > 0 {
        info.context_update_tile_id = r.f(tile_cols_log2 + tile_rows_log2)? as u16;
        if info.context_update_tile_id as usize >= info.num_tiles() {
            return Err(PixelForgeError::InvalidInput(
                "AV1: context_update_tile_id out of range".to_string(),
            ));
        }
        info.tile_size_bytes_minus_1 = r.f(2)? as u8;
    }
    Ok(info)
}

/// Where one tile's data sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileLocation {
    /// Byte offset of the first byte of tile data (after its
    /// `tile_size_minus_1` field, if any), relative to the start of the
    /// buffer the OBU iterator was given.
    pub offset: usize,
    /// Size of the tile data in bytes.
    pub size: usize,
}

/// One parsed tile group: the range of tiles it covers and where each one is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileGroup {
    pub tile_start_and_end_present_flag: bool,
    pub tg_start: u16,
    pub tg_end: u16,
    pub tiles: Vec<TileLocation>,
}

/// Parse `tile_group_obu(sz)` from `data`, which starts where the tile group
/// does: at the payload of a tile group OBU, or after the byte-aligned frame
/// header in a frame OBU. `base_offset` is the offset of `data[0]` in the
/// buffer tile locations are reported against.
pub(crate) fn parse_tile_group(
    data: &[u8],
    base_offset: usize,
    info: &TileInfo,
) -> Result<TileGroup> {
    let num_tiles = info.num_tiles();
    if num_tiles == 0 {
        return Err(PixelForgeError::InvalidInput(
            "AV1: tile group for a frame with no tiles".to_string(),
        ));
    }
    let mut r = BitReader::new(data);
    let mut tg = TileGroup {
        tile_start_and_end_present_flag: false,
        tg_start: 0,
        tg_end: (num_tiles - 1) as u16,
        tiles: Vec::new(),
    };
    if num_tiles > 1 {
        tg.tile_start_and_end_present_flag = r.flag()?;
    }
    if tg.tile_start_and_end_present_flag {
        let tile_bits = info.tile_cols_log2 as u32 + info.tile_rows_log2 as u32;
        tg.tg_start = r.f(tile_bits)? as u16;
        tg.tg_end = r.f(tile_bits)? as u16;
        if tg.tg_start > tg.tg_end || tg.tg_end as usize >= num_tiles {
            return Err(PixelForgeError::InvalidInput(format!(
                "AV1: tile group covers tiles {}..={} of {num_tiles}",
                tg.tg_start, tg.tg_end
            )));
        }
    }
    r.byte_alignment()?;
    let mut pos = r.byte_position();
    let size_bytes = info.tile_size_bytes() as usize;
    for tile_num in tg.tg_start..=tg.tg_end {
        let size = if tile_num == tg.tg_end {
            data.len() - pos
        } else {
            if data.len() - pos < size_bytes {
                return Err(PixelForgeError::InvalidInput(
                    "AV1: tile group ends inside a tile size".to_string(),
                ));
            }
            let mut sr = BitReader::new(&data[pos..pos + size_bytes]);
            let size = sr.le(size_bytes as u32)? as usize + 1;
            pos += size_bytes;
            if size > data.len() - pos {
                return Err(PixelForgeError::InvalidInput(format!(
                    "AV1: tile {tile_num} of {size} bytes overruns its tile group"
                )));
            }
            size
        };
        if size == 0 {
            return Err(PixelForgeError::InvalidInput(format!(
                "AV1: tile {tile_num} is empty"
            )));
        }
        tg.tiles.push(TileLocation {
            offset: base_offset + pos,
            size,
        });
        pos += size;
    }
    Ok(tg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_grid_for_320x240() {
        // 320x240 is 80x60 mode-info units: 5x4 superblocks of 64x64.
        // One increment of each log2: 2x2 tiles of 3 and 2 superblocks.
        // Then context_update_tile_id (2 bits) and tile_size_bytes_minus_1.
        // Bits: 1 1 0 | 1 0 | 11 | 01, then padding.
        let bits = [0b1101_0110, 0b1000_0000];
        let mut r = BitReader::new(&bits);
        let info = parse_tile_info(&mut r, false, 80, 60).unwrap();
        assert_eq!((info.tile_cols, info.tile_rows), (2, 2));
        assert_eq!(info.mi_col_starts, vec![0, 48, 80]);
        assert_eq!(info.mi_row_starts, vec![0, 32, 60]);
        assert_eq!(info.width_in_sbs_minus_1, vec![2, 1]);
        assert_eq!(info.height_in_sbs_minus_1, vec![1, 1]);
        assert_eq!(info.context_update_tile_id, 3);
        assert_eq!(info.tile_size_bytes(), 2);
    }

    #[test]
    fn tile_group_sizes_and_offsets() {
        let info = TileInfo {
            tile_cols: 2,
            tile_rows: 1,
            tile_cols_log2: 1,
            tile_size_bytes_minus_1: 1,
            ..Default::default()
        };
        // No start/end, aligned; tile 0 is 3 bytes (size field 2), tile 1
        // takes the remaining 2 bytes.
        let data = [0x00, 0x02, 0x00, 1, 2, 3, 4, 5];
        let tg = parse_tile_group(&data, 100, &info).unwrap();
        assert_eq!(
            tg.tiles,
            vec![
                TileLocation {
                    offset: 103,
                    size: 3
                },
                TileLocation {
                    offset: 106,
                    size: 2
                }
            ]
        );
        // A tile size that overruns the OBU is an error, not a panic.
        let bad = [0x00, 0x20, 0x00, 1];
        assert!(parse_tile_group(&bad, 0, &info).is_err());
    }
}
