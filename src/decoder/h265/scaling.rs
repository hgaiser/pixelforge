//! `scaling_list_data()` (clause 7.3.4) and the default scaling lists
//! (clause 7.4.5, Tables 7-5 and 7-6).

use super::parser::{Reader, invalid};
use crate::error::Result;

/// Table 7-6, intra (matrixId 0..=2), in coded (up-right diagonal) order.
const DEFAULT_8X8_INTRA: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 17, 16, 17, 18, 17, 18, 18, 17, 18, 21, 19, 20,
    21, 20, 19, 21, 24, 22, 22, 24, 24, 22, 22, 24, 25, 25, 27, 30, 27, 25, 25, 29, 31, 35, 35, 31,
    29, 36, 41, 44, 41, 36, 47, 54, 54, 47, 65, 70, 65, 88, 88, 115,
];

/// Table 7-6, inter (matrixId 3..=5), in coded (up-right diagonal) order.
const DEFAULT_8X8_INTER: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 20, 20, 20,
    20, 20, 20, 20, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 28, 28, 28, 28, 28,
    28, 33, 33, 33, 33, 33, 41, 41, 41, 41, 54, 54, 54, 71, 71, 91,
];

/// A complete set of scaling lists, laid out like `StdVideoH265ScalingLists`.
///
/// Coefficients are in coded order (the up-right diagonal scan the bitstream
/// uses), not raster order; that is the order the Vulkan structure carries
/// them in. The 32x32 lists hold matrixId 0 (intra luma) and 3 (inter luma)
/// at indices 0 and 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ScalingLists {
    pub scaling_list_4x4: [[u8; 16]; 6],
    pub scaling_list_8x8: [[u8; 64]; 6],
    pub scaling_list_16x16: [[u8; 64]; 6],
    pub scaling_list_32x32: [[u8; 64]; 2],
    pub scaling_list_dc_coef_16x16: [u8; 6],
    pub scaling_list_dc_coef_32x32: [u8; 2],
}

impl Default for ScalingLists {
    /// The default lists of Tables 7-5 and 7-6, which apply when
    /// `scaling_list_enabled_flag` is 1 and no list data is coded.
    fn default() -> Self {
        let mut lists = Self {
            scaling_list_4x4: [[16; 16]; 6],
            scaling_list_8x8: [[0; 64]; 6],
            scaling_list_16x16: [[0; 64]; 6],
            scaling_list_32x32: [[0; 64]; 2],
            scaling_list_dc_coef_16x16: [16; 6],
            scaling_list_dc_coef_32x32: [16; 2],
        };
        for size_id in 1..4 {
            for matrix_id in matrix_ids(size_id) {
                lists.set_default(size_id, matrix_id);
            }
        }
        lists
    }
}

/// The matrixId values coded for `size_id`: all six, or only 0 and 3 for 32x32.
fn matrix_ids(size_id: usize) -> impl Iterator<Item = usize> {
    (0..6).step_by(if size_id == 3 { 3 } else { 1 })
}

impl ScalingLists {
    /// The coefficients of list (`size_id`, `matrix_id`).
    pub fn list(&self, size_id: usize, matrix_id: usize) -> &[u8] {
        match size_id {
            0 => &self.scaling_list_4x4[matrix_id],
            1 => &self.scaling_list_8x8[matrix_id],
            2 => &self.scaling_list_16x16[matrix_id],
            _ => &self.scaling_list_32x32[matrix_id / 3],
        }
    }

    fn list_mut(&mut self, size_id: usize, matrix_id: usize) -> &mut [u8] {
        match size_id {
            0 => &mut self.scaling_list_4x4[matrix_id],
            1 => &mut self.scaling_list_8x8[matrix_id],
            2 => &mut self.scaling_list_16x16[matrix_id],
            _ => &mut self.scaling_list_32x32[matrix_id / 3],
        }
    }

    /// The DC coefficient of a 16x16 or 32x32 list (`size_id` 2 or 3).
    pub fn dc(&self, size_id: usize, matrix_id: usize) -> u8 {
        match size_id {
            2 => self.scaling_list_dc_coef_16x16[matrix_id],
            3 => self.scaling_list_dc_coef_32x32[matrix_id / 3],
            _ => 16,
        }
    }

    fn set_dc(&mut self, size_id: usize, matrix_id: usize, value: u8) {
        match size_id {
            2 => self.scaling_list_dc_coef_16x16[matrix_id] = value,
            3 => self.scaling_list_dc_coef_32x32[matrix_id / 3] = value,
            _ => {}
        }
    }

    fn set_default(&mut self, size_id: usize, matrix_id: usize) {
        let list = self.list_mut(size_id, matrix_id);
        if size_id == 0 {
            list.fill(16);
        } else if matrix_id < 3 {
            list.copy_from_slice(&DEFAULT_8X8_INTRA);
        } else {
            list.copy_from_slice(&DEFAULT_8X8_INTER);
        }
        self.set_dc(size_id, matrix_id, 16);
    }
}

/// Parse `scaling_list_data()`.
///
/// For a list predicted from another (`scaling_list_pred_mode_flag` = 0) the
/// reference's coefficients and DC value are copied, or the default list is
/// used when `scaling_list_pred_matrix_id_delta` is 0, so the result is always
/// fully resolved.
pub(crate) fn parse_scaling_list_data(r: &mut Reader) -> Result<ScalingLists> {
    let mut lists = ScalingLists::default();
    for size_id in 0..4 {
        let step = if size_id == 3 { 3 } else { 1 };
        for matrix_id in matrix_ids(size_id) {
            let pred_mode_flag = r.flag()?;
            if !pred_mode_flag {
                let delta = r.ue_max(
                    (matrix_id / step) as u32,
                    "scaling_list_pred_matrix_id_delta",
                )? as usize;
                if delta == 0 {
                    lists.set_default(size_id, matrix_id);
                } else {
                    let ref_matrix_id = matrix_id - delta * step;
                    let reference = lists.list(size_id, ref_matrix_id).to_vec();
                    lists
                        .list_mut(size_id, matrix_id)
                        .copy_from_slice(&reference);
                    let dc = lists.dc(size_id, ref_matrix_id);
                    lists.set_dc(size_id, matrix_id, dc);
                }
                continue;
            }

            let mut next_coef = 8i32;
            if size_id > 1 {
                let dc_minus8 = r.se_range(-7, 247, "scaling_list_dc_coef_minus8")?;
                next_coef = dc_minus8 + 8;
                lists.set_dc(size_id, matrix_id, next_coef as u8);
            }
            let list = lists.list_mut(size_id, matrix_id);
            for coef in list.iter_mut() {
                let delta = r.se_range(-128, 127, "scaling_list_delta_coef")?;
                next_coef = (next_coef + delta + 256) % 256;
                if next_coef == 0 {
                    return Err(invalid("scaling list coefficient is 0"));
                }
                *coef = next_coef as u8;
            }
        }
    }
    Ok(lists)
}
