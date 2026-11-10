//! The H.264 level a stream needs (Annex A, Table A-1).

use ash::vk::native::StdVideoH264LevelIdc;

/// What a level allows, from Table A-1.
struct Limits {
    /// `StdVideoH264LevelIdc`, which counts the levels from 1.0 up without
    /// level 1b.
    level: StdVideoH264LevelIdc,
    /// `MaxMBPS`: macroblocks per second.
    max_mbps: u64,
    /// `MaxFS`: macroblocks per frame.
    max_fs: u32,
    /// `MaxDpbMbs`: macroblocks in the DPB.
    max_dpb_mbs: u32,
    /// `MaxBR` in units of 1000 bits per second, before the profile's factor.
    max_br: u64,
}

const fn limits(
    level: StdVideoH264LevelIdc,
    max_mbps: u64,
    max_fs: u32,
    max_dpb_mbs: u32,
    max_br: u64,
) -> Limits {
    Limits {
        level,
        max_mbps,
        max_fs,
        max_dpb_mbs,
        max_br,
    }
}

const LEVELS: [Limits; 19] = [
    limits(0, 1_485, 99, 396, 64),
    limits(1, 3_000, 396, 900, 192),
    limits(2, 6_000, 396, 2_376, 384),
    limits(3, 11_880, 396, 2_376, 768),
    limits(4, 11_880, 396, 2_376, 2_000),
    limits(5, 19_800, 792, 4_752, 4_000),
    limits(6, 20_250, 1_620, 8_100, 4_000),
    limits(7, 40_500, 1_620, 8_100, 10_000),
    limits(8, 108_000, 3_600, 18_000, 14_000),
    limits(9, 216_000, 5_120, 20_480, 20_000),
    limits(10, 245_760, 8_192, 32_768, 20_000),
    limits(11, 245_760, 8_192, 32_768, 50_000),
    limits(12, 522_240, 8_704, 34_816, 50_000),
    limits(13, 589_824, 22_080, 110_400, 135_000),
    limits(14, 983_040, 36_864, 184_320, 240_000),
    limits(15, 2_073_600, 36_864, 184_320, 240_000),
    limits(16, 4_177_920, 139_264, 696_320, 240_000),
    limits(17, 8_355_840, 139_264, 696_320, 480_000),
    limits(18, 16_711_680, 139_264, 696_320, 800_000),
];

/// Level 4.1, the lowest level the encoder writes. It was the only level
/// the encoder wrote before, and every stream that fits it keeps it.
pub(super) const LOWEST: StdVideoH264LevelIdc = 11;

/// What a stream needs from its level.
pub(super) struct Needs {
    pub width_in_mbs: u32,
    pub height_in_mbs: u32,
    /// Frames per second, rounded up.
    pub frame_rate: u32,
    /// `max_dec_frame_buffering`: the frames the DPB holds.
    pub dpb_frames: u32,
    /// The peak bitrate in bits per second, or 0 if not known (constant QP).
    pub max_bitrate: u64,
    /// `cpbBrVclFactor` of the profile (Table A-2): 1250 for High, 4000 for
    /// High 4:4:4 Predictive.
    pub bitrate_factor: u64,
}

/// The lowest level from [`LOWEST`] that allows `needs`, or the highest
/// level there is if none does.
pub(super) fn level_for(needs: &Needs) -> StdVideoH264LevelIdc {
    let frame_mbs = needs.width_in_mbs * needs.height_in_mbs;
    let fits = |limits: &Limits| {
        // A frame's sides are each at most sqrt(8 * MaxFS) macroblocks.
        let max_side = ((8 * limits.max_fs) as f64).sqrt() as u32;
        frame_mbs <= limits.max_fs
            && needs.width_in_mbs <= max_side
            && needs.height_in_mbs <= max_side
            && u64::from(frame_mbs) * u64::from(needs.frame_rate) <= limits.max_mbps
            && needs.dpb_frames <= (limits.max_dpb_mbs / frame_mbs).min(16)
            && needs.max_bitrate <= limits.max_br * needs.bitrate_factor
    };
    LEVELS
        .iter()
        .filter(|limits| limits.level >= LOWEST)
        .find(|limits| fits(limits))
        .map_or(LEVELS[LEVELS.len() - 1].level, |limits| limits.level)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn needs(width: u32, height: u32, frame_rate: u32, dpb_frames: u32) -> Needs {
        Needs {
            width_in_mbs: width.div_ceil(16),
            height_in_mbs: height.div_ceil(16),
            frame_rate,
            dpb_frames,
            max_bitrate: 0,
            bitrate_factor: 1250,
        }
    }

    #[test]
    fn small_streams_keep_level_4_1() {
        assert_eq!(level_for(&needs(320, 240, 30, 4)), 11);
        assert_eq!(level_for(&needs(1920, 1080, 30, 4)), 11);
    }

    #[test]
    fn a_larger_dpb_needs_a_higher_level() {
        // 4 frames of 1080p fill level 4.1's DPB.
        assert_eq!(level_for(&needs(1920, 1080, 30, 5)), 13);
    }

    #[test]
    fn size_and_rate_need_their_levels() {
        assert_eq!(level_for(&needs(1920, 1080, 60, 4)), 12);
        assert_eq!(level_for(&needs(3840, 2160, 30, 4)), 14);
        assert_eq!(level_for(&needs(3840, 2160, 60, 4)), 15);
    }

    #[test]
    fn the_bitrate_needs_its_level() {
        let mut stream = needs(1920, 1080, 30, 4);
        stream.max_bitrate = 80_000_000;
        assert_eq!(level_for(&stream), 13);
    }

    #[test]
    fn too_much_for_any_level_is_the_highest() {
        assert_eq!(level_for(&needs(16384, 16384, 30, 4)), 18);
    }
}
