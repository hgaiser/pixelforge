//! The H.265 level a stream needs (Annex A, Tables A.8 and A.9), and the DPB
//! size that level allows (A.4.2).

use ash::vk::native::{
    StdVideoH265LevelIdc, StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_1_0 as L1_0,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_0 as L2_0,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_1 as L2_1,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_0 as L3_0,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_1 as L3_1,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_0 as L4_0,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_1 as L4_1,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_0 as L5_0,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_1 as L5_1,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_2 as L5_2,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_0 as L6_0,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_1 as L6_1,
    StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_2 as L6_2,
};

/// What a level allows.
struct Limits {
    level: StdVideoH265LevelIdc,
    /// `MaxLumaPs`: luma samples per picture (Table A.8).
    max_luma_ps: u64,
    /// `MaxLumaSr`: luma samples per second (Table A.9).
    max_luma_sr: u64,
}

const fn limits(level: StdVideoH265LevelIdc, max_luma_ps: u64, max_luma_sr: u64) -> Limits {
    Limits {
        level,
        max_luma_ps,
        max_luma_sr,
    }
}

const LEVELS: [Limits; 13] = [
    limits(L1_0, 36_864, 552_960),
    limits(L2_0, 122_880, 3_686_400),
    limits(L2_1, 245_760, 7_372_800),
    limits(L3_0, 552_960, 16_588_800),
    limits(L3_1, 983_040, 33_177_600),
    limits(L4_0, 2_228_224, 66_846_720),
    limits(L4_1, 2_228_224, 133_693_440),
    limits(L5_0, 8_912_896, 267_386_880),
    limits(L5_1, 8_912_896, 534_773_760),
    limits(L5_2, 8_912_896, 1_069_547_520),
    limits(L6_0, 35_651_584, 1_069_547_520),
    limits(L6_1, 35_651_584, 2_139_095_040),
    limits(L6_2, 35_651_584, 4_278_190_080),
];

/// Level 5.1, the lowest level the encoder writes. It was the only level the
/// encoder wrote before, and every stream that fits it keeps it.
pub(super) const LOWEST: StdVideoH265LevelIdc = L5_1;

/// `maxDpbPicBuf` of A.4.2, for every profile but the screen content ones.
const MAX_DPB_PIC_BUF: u32 = 6;

/// What a stream needs from its level.
pub(super) struct Needs {
    /// The coded width in luma samples.
    pub width: u32,
    /// The coded height in luma samples.
    pub height: u32,
    /// Frames per second, rounded up.
    pub frame_rate: u32,
}

/// The lowest level from [`LOWEST`] that allows the picture size and the
/// sample rate of `needs`, and at most `max_level`.
///
/// The DPB does not raise the level: from level 5.1 up, only level 6 allows a
/// larger DPB at the same size, and decoders (Apple VideoToolbox, for one)
/// reject streams of level 6. The encoder holds fewer references instead,
/// see [`max_dpb_size`].
///
/// The bitrate does not raise the level either. Streams for low latency often
/// go over the Main tier bitrate of their level, decoders do not check it,
/// and a higher level is what they do check.
pub(super) fn level_for(needs: &Needs, max_level: StdVideoH265LevelIdc) -> StdVideoH265LevelIdc {
    let luma_ps = u64::from(needs.width) * u64::from(needs.height);
    let fits = |limits: &Limits| {
        // A picture's sides are each at most sqrt(8 * MaxLumaPs).
        let max_side = ((8 * limits.max_luma_ps) as f64).sqrt() as u64;
        luma_ps <= limits.max_luma_ps
            && u64::from(needs.width) <= max_side
            && u64::from(needs.height) <= max_side
            && luma_ps * u64::from(needs.frame_rate) <= limits.max_luma_sr
    };
    let needed = LEVELS
        .iter()
        .filter(|limits| limits.level >= LOWEST)
        .find(|limits| fits(limits))
        .map_or(LEVELS[LEVELS.len() - 1].level, |limits| limits.level);
    if needed > max_level {
        tracing::warn!(
            "H.265: the stream needs level {needed} (as StdVideoH265LevelIdc), and the device \
             encodes up to {max_level}, so the stream says that"
        );
        return max_level;
    }
    needed
}

/// `MaxDpbSize` (A.4.2): the most pictures, the current one included, that the
/// DPB of a `width` x `height` stream of `level` holds. It is the limit of
/// `sps_max_dec_pic_buffering_minus1 + 1`.
pub(super) fn max_dpb_size(width: u32, height: u32, level: StdVideoH265LevelIdc) -> u32 {
    let max_luma_ps = LEVELS
        .iter()
        .find(|limits| limits.level == level)
        .map_or(LEVELS[LEVELS.len() - 1].max_luma_ps, |limits| {
            limits.max_luma_ps
        });
    let luma_ps = u64::from(width) * u64::from(height);
    if luma_ps <= max_luma_ps >> 2 {
        (4 * MAX_DPB_PIC_BUF).min(16)
    } else if luma_ps <= max_luma_ps >> 1 {
        (2 * MAX_DPB_PIC_BUF).min(16)
    } else if luma_ps <= (3 * max_luma_ps) >> 2 {
        (4 * MAX_DPB_PIC_BUF / 3).min(16)
    } else {
        MAX_DPB_PIC_BUF
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn level(width: u32, height: u32, frame_rate: u32) -> StdVideoH265LevelIdc {
        level_for(
            &Needs {
                width,
                height,
                frame_rate,
            },
            L6_2,
        )
    }

    #[test]
    fn small_streams_keep_level_5_1() {
        assert_eq!(level(320, 256, 30), L5_1);
        assert_eq!(level(1920, 1088, 120), L5_1);
        assert_eq!(level(3840, 2176, 60), L5_1);
    }

    #[test]
    fn size_and_rate_need_their_levels() {
        assert_eq!(level(3840, 2176, 120), L5_2);
        assert_eq!(level(7680, 4320, 30), L6_0);
        assert_eq!(level(7680, 4320, 60), L6_1);
    }

    #[test]
    fn the_device_caps_the_level() {
        let needs = Needs {
            width: 3840,
            height: 2176,
            frame_rate: 120,
        };
        assert_eq!(level_for(&needs, L5_1), L5_1);
    }

    #[test]
    fn too_much_for_any_level_is_the_highest() {
        assert_eq!(level(16384, 16384, 30), L6_2);
    }

    #[test]
    fn the_dpb_shrinks_with_the_picture() {
        assert_eq!(max_dpb_size(320, 256, L5_1), 16);
        assert_eq!(max_dpb_size(1920, 1088, L5_1), 16);
        // 2736x1260, aligned to 32.
        assert_eq!(max_dpb_size(2752, 1280, L5_1), 12);
        assert_eq!(max_dpb_size(3840, 2176, L5_1), 6);
        assert_eq!(max_dpb_size(3840, 2176, L5_2), 6);
        assert_eq!(max_dpb_size(3840, 2176, L6_0), 16);
    }
}
