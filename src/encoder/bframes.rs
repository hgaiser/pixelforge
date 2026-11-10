//! B-frame configuration, and what a device can do with B-frames.

use ash::vk;

use super::{Codec, EncodeConfig};
use crate::error::{PixelForgeError, Result};

/// The most B-frames [`EncodeConfig::with_b_frames`] takes.
///
/// Vulkan reports no limit on consecutive B-frames, only whether a frame can
/// predict from both directions, so this limit is pixelforge's own.
pub const MAX_B_FRAMES: u32 = 7;

/// How the B-frames between two reference frames are referenced.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BFrameReferences {
    /// No B-frame is a reference.
    #[default]
    Flat,
    /// The middle B-frame of each group is a reference for the B-frames on
    /// either side of it, and so on for each half. With fewer than three
    /// B-frames there is no middle one, and this is the same as `Flat`.
    Pyramid,
}

/// What a device can do with B-frames, for one codec and profile. See
/// [`VideoContext::supports_b_frames`](crate::VideoContext::supports_b_frames).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BFrameSupport {
    /// No B-frames.
    None,
    /// B-frames that are not references.
    Flat,
    /// B-frames that are references too.
    Pyramid,
}

/// Whether the B-frames of `config` include references, which is what
/// [`BFrameSupport::Pyramid`] is needed for.
fn needs_pyramid(config: &EncodeConfig) -> bool {
    config.b_frame_references == BFrameReferences::Pyramid && config.b_frames >= 3
}

/// Check the B-frame settings of `config` that need no device.
pub(crate) fn check_config(config: &EncodeConfig) -> Result<()> {
    if config.b_frames > MAX_B_FRAMES {
        return Err(PixelForgeError::InvalidInput(format!(
            "{} B-frames requested, the most is {MAX_B_FRAMES}",
            config.b_frames
        )));
    }
    Ok(())
}

/// Check that a device with `support` can encode the B-frames of `config`.
pub(crate) fn check_support(config: &EncodeConfig, support: BFrameSupport) -> Result<()> {
    if config.b_frames == 0 {
        return Ok(());
    }
    let needed = if needs_pyramid(config) {
        BFrameSupport::Pyramid
    } else {
        BFrameSupport::Flat
    };
    if support < needed {
        return Err(PixelForgeError::NoSuitableDevice(format!(
            "{:?} encode: {} B-frames ({:?}) need {needed:?} B-frame support, \
             and the encoder has {support:?} on this device",
            config.codec, config.b_frames, config.b_frame_references
        )));
    }
    Ok(())
}

/// The B-frames the encoder itself can encode for `codec`, whatever the
/// device can.
fn implemented(codec: Codec) -> BFrameSupport {
    match codec {
        Codec::H264 | Codec::H265 | Codec::AV1 => BFrameSupport::Flat,
    }
}

/// How many references one frame may use at most, as the device reports
/// it, lowered by `PIXELFORGE_MAX_ACTIVE_REFERENCES` as the encoder lowers it.
fn max_active_references(device: u32) -> u32 {
    std::env::var("PIXELFORGE_MAX_ACTIVE_REFERENCES")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .map_or(device, |cap| device.min(cap))
}

/// The B-frame support for `codec` from what the device reports for one
/// profile, capped by what the encoder implements.
fn support(
    codec: Codec,
    device: BFrameSupport,
    max_active_references_device: u32,
) -> BFrameSupport {
    // A B-frame predicts from one reference on each side at least.
    if max_active_references(max_active_references_device) < 2 {
        return BFrameSupport::None;
    }
    let support = device.min(implemented(codec));
    tracing::debug!(
        "{codec:?} B-frames: the device has {device:?}, the encoder {:?}",
        implemented(codec)
    );
    support
}

/// Flat B-frames need one reference in each list. Pyramid needs a B-frame
/// in each list as well.
fn h26x_support(l0: u32, l1: u32, b_in_l0: bool, b_in_l1: bool) -> BFrameSupport {
    if l0 == 0 || l1 == 0 {
        BFrameSupport::None
    } else if b_in_l0 && b_in_l1 {
        BFrameSupport::Pyramid
    } else {
        BFrameSupport::Flat
    }
}

/// The B-frame support of an H.264 encode profile.
pub(crate) fn h264_support(
    caps: &vk::VideoEncodeH264CapabilitiesKHR<'_>,
    max_active_references: u32,
) -> BFrameSupport {
    let device = h26x_support(
        caps.max_b_picture_l0_reference_count,
        caps.max_l1_reference_count,
        caps.flags
            .contains(vk::VideoEncodeH264CapabilityFlagsKHR::B_FRAME_IN_L0_LIST),
        caps.flags
            .contains(vk::VideoEncodeH264CapabilityFlagsKHR::B_FRAME_IN_L1_LIST),
    );
    support(Codec::H264, device, max_active_references)
}

/// The B-frame support of an H.265 encode profile.
pub(crate) fn h265_support(
    caps: &vk::VideoEncodeH265CapabilitiesKHR<'_>,
    max_active_references: u32,
) -> BFrameSupport {
    let device = h26x_support(
        caps.max_b_picture_l0_reference_count,
        caps.max_l1_reference_count,
        caps.flags
            .contains(vk::VideoEncodeH265CapabilityFlagsKHR::B_FRAME_IN_L0_LIST),
        caps.flags
            .contains(vk::VideoEncodeH265CapabilityFlagsKHR::B_FRAME_IN_L1_LIST),
    );
    support(Codec::H265, device, max_active_references)
}

/// The reference names that predict from the past, as bits of a Vulkan
/// reference name mask (bit 0 is `LAST`): `LAST` to `GOLDEN`.
const AV1_PAST_NAMES: u32 = 0b000_1111;
/// The reference names that predict from the future: `BWDREF` to `ALTREF`.
const AV1_FUTURE_NAMES: u32 = 0b111_0000;

/// The B-frame support of an AV1 encode profile.
///
/// A B-frame is a frame with bidirectional compound prediction, from a past
/// reference name and a future one. Any AV1 frame can be a reference, so
/// there is no further condition for pyramid.
pub(crate) fn av1_support(
    caps: &vk::VideoEncodeAV1CapabilitiesKHR<'_>,
    max_active_references: u32,
) -> BFrameSupport {
    let mask = caps.bidirectional_compound_reference_name_mask;
    let device = if caps.max_bidirectional_compound_reference_count >= 2
        && caps.max_bidirectional_compound_group1_reference_count >= 1
        && caps.max_bidirectional_compound_group2_reference_count >= 1
        && mask & AV1_PAST_NAMES != 0
        && mask & AV1_FUTURE_NAMES != 0
    {
        BFrameSupport::Pyramid
    } else {
        BFrameSupport::None
    };
    support(Codec::AV1, device, max_active_references)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h26x_support_needs_both_lists() {
        assert_eq!(h26x_support(0, 1, true, true), BFrameSupport::None);
        assert_eq!(h26x_support(1, 0, true, true), BFrameSupport::None);
        assert_eq!(h26x_support(1, 1, false, false), BFrameSupport::Flat);
        assert_eq!(h26x_support(1, 1, true, false), BFrameSupport::Flat);
        assert_eq!(h26x_support(2, 1, true, true), BFrameSupport::Pyramid);
    }

    #[test]
    fn support_needs_two_active_references() {
        assert_eq!(
            support(Codec::H264, BFrameSupport::Pyramid, 1),
            BFrameSupport::None
        );
    }

    #[test]
    fn check_config_limits_the_count() {
        let config = EncodeConfig::h264(64, 64).with_b_frames(MAX_B_FRAMES);
        assert!(check_config(&config).is_ok());
        let config = EncodeConfig::h264(64, 64).with_b_frames(MAX_B_FRAMES + 1);
        assert!(check_config(&config).is_err());
    }

    #[test]
    fn check_support_asks_pyramid_only_with_a_middle_frame() {
        let flat = EncodeConfig::h264(64, 64).with_b_frames(3);
        assert!(check_support(&flat, BFrameSupport::Flat).is_ok());
        assert!(check_support(&flat, BFrameSupport::None).is_err());

        let pyramid = flat.with_b_frame_references(BFrameReferences::Pyramid);
        assert!(check_support(&pyramid, BFrameSupport::Flat).is_err());
        assert!(check_support(&pyramid, BFrameSupport::Pyramid).is_ok());

        let short = EncodeConfig::h264(64, 64)
            .with_b_frames(2)
            .with_b_frame_references(BFrameReferences::Pyramid);
        assert!(check_support(&short, BFrameSupport::Flat).is_ok());
    }

    #[test]
    fn no_b_frames_need_no_support() {
        let config = EncodeConfig::h264(64, 64);
        assert!(check_support(&config, BFrameSupport::None).is_ok());
    }
}
