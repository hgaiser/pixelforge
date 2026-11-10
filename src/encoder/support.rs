//! Device support queries for encode formats.

use super::bframes::{BFrameSupport, av1_support, h264_support, h265_support};
use super::codec::query_video_capabilities;
use super::profile::{av1_profile, h264_profile_idc, h265_profile_idc};
use super::rgb::rgb_conversion_profile;
use super::{BitDepth, ChromaFormat, Codec};
use crate::vulkan::VideoContext;
use ash::vk;
use ash::vk::TaggedStructure;
use tracing::{debug, warn};

/// What [`VideoContext::query_encode_profile`] learns about a profile, besides
/// the profile itself.
pub(crate) struct ProfileCaps {
    /// What the encoder can do with B-frames, by the rules it checks itself.
    pub b_frames: BFrameSupport,
    /// `VkVideoEncodeCapabilitiesKHR::maxQualityLevels`.
    pub max_quality_levels: u32,
}

/// Why [`VideoContext::query_encode_profile`] has no answer to give.
pub(crate) enum ProfileQueryError {
    /// The encoder has no profile for this combination (H.265 4:2:2).
    NoProfile,
    /// The driver rejected the profile's capability query.
    Query(vk::Result),
}

impl VideoContext {
    /// Check whether the device can encode `codec` in the given chroma format and
    /// bit depth.
    ///
    /// [`supports_encode`](Self::supports_encode) only answers for 8-bit 4:2:0.
    /// Many devices lack other combinations; AMD's VCN, for instance, has no 4:4:4
    /// profiles. This asks the driver for the capabilities of the profile an
    /// [`Encoder`](crate::Encoder) would request, without creating an encoder.
    /// Resolution and rate control are not considered.
    pub fn supports_encode_format(
        &self,
        codec: Codec,
        chroma_format: ChromaFormat,
        bit_depth: BitDepth,
    ) -> bool {
        if !self.supports_encode(codec) {
            return false;
        }
        let result = self.query_encode_profile(codec, chroma_format, bit_depth, None, |_, _| ());
        report(
            &format!("{codec:?} encode {chroma_format:?} {bit_depth:?}"),
            result,
        )
        .is_some()
    }

    /// Check what the device can do with B-frames for `codec` in the given
    /// chroma format and bit depth: what an [`Encoder`](crate::Encoder) with
    /// [`EncodeConfig::with_b_frames`](crate::EncodeConfig::with_b_frames)
    /// accepts. [`BFrameSupport::None`] also where the device cannot encode
    /// the profile at all, and where the encoder has no B-frames for the
    /// codec yet.
    pub fn supports_b_frames(
        &self,
        codec: Codec,
        chroma_format: ChromaFormat,
        bit_depth: BitDepth,
    ) -> BFrameSupport {
        if !self.supports_encode(codec) {
            return BFrameSupport::None;
        }
        let result = self.query_encode_profile(codec, chroma_format, bit_depth, None, |_, caps| {
            caps.b_frames
        });
        report(
            &format!("{codec:?} B-frames {chroma_format:?} {bit_depth:?}"),
            result,
        )
        .unwrap_or(BFrameSupport::None)
    }

    /// The number of quality levels the device has for `codec` in the given
    /// chroma format and bit depth: what
    /// [`EncodeConfig::with_quality_level`](crate::EncodeConfig::with_quality_level)
    /// accepts is less than this. 0 where the device cannot encode the
    /// profile.
    pub fn encode_quality_levels(
        &self,
        codec: Codec,
        chroma_format: ChromaFormat,
        bit_depth: BitDepth,
    ) -> u32 {
        if !self.supports_encode(codec) {
            return 0;
        }
        let result = self.query_encode_profile(codec, chroma_format, bit_depth, None, |_, caps| {
            caps.max_quality_levels
        });
        report(
            &format!("{codec:?} quality levels {chroma_format:?} {bit_depth:?}"),
            result,
        )
        .unwrap_or(0)
    }

    /// Build the profile an encoder would request for `codec`, `chroma_format`
    /// and `bit_depth`, ask the driver for its capabilities, and on success
    /// hand the profile to `then` for any further query about it.
    ///
    /// `then` also gets what else the query says about the profile.
    ///
    /// With `rgb_caps`, the profile is the RGB input one an encoder built with
    /// [`Conversion::Encoder`](crate::Conversion::Encoder)
    /// requests, and the driver fills in what its conversion supports.
    ///
    /// The encoders build their profiles themselves, since theirs also carry
    /// usage hints, but from the same profile numbers in
    /// [`super::profile`], so this answers for the profile they use.
    pub(crate) fn query_encode_profile<R>(
        &self,
        codec: Codec,
        chroma_format: ChromaFormat,
        bit_depth: BitDepth,
        mut rgb_caps: Option<&mut vk::VideoEncodeRgbConversionCapabilitiesVALVE<'static>>,
        then: impl FnOnce(&vk::VideoProfileInfoKHR<'_>, ProfileCaps) -> R,
    ) -> Result<R, ProfileQueryError> {
        let chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR = chroma_format.into();
        let component_bit_depth: vk::VideoComponentBitDepthFlagsKHR = bit_depth.into();
        let mut encode_caps = vk::VideoEncodeCapabilitiesKHR::default();
        let mut rgb_profile = rgb_conversion_profile();

        let base = vk::VideoProfileInfoKHR::default()
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(component_bit_depth)
            .chroma_bit_depth(component_bit_depth);

        // Each arm keeps its profile and capability structs alive across the
        // calls that use them.
        macro_rules! query {
            ($operation:expr, $profile:expr, $codec_caps:expr, $b_frames:expr) => {{
                let mut profile = $profile;
                let mut profile_info = base.video_codec_operation($operation).push(&mut profile);
                if rgb_caps.is_some() {
                    profile_info = profile_info.push(&mut rgb_profile);
                }
                let mut codec_caps = $codec_caps;
                let mut caps = vk::VideoCapabilitiesKHR::default()
                    .push(&mut encode_caps)
                    .push(&mut codec_caps);
                if let Some(rgb_caps) = rgb_caps.as_deref_mut() {
                    caps = caps.push(rgb_caps);
                }
                match query_video_capabilities(self, &profile_info, &mut caps) {
                    vk::Result::SUCCESS => {
                        let max_active = caps.max_active_reference_pictures;
                        let vendor_id = self.device_properties().vendor_id;
                        let caps = ProfileCaps {
                            b_frames: $b_frames(vendor_id, &codec_caps, max_active),
                            max_quality_levels: encode_caps.max_quality_levels,
                        };
                        Ok(then(&profile_info, caps))
                    }
                    err => Err(ProfileQueryError::Query(err)),
                }
            }};
        }

        match codec {
            Codec::H264 => query!(
                vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
                vk::VideoEncodeH264ProfileInfoKHR::default()
                    .std_profile_idc(h264_profile_idc(chroma_format)),
                vk::VideoEncodeH264CapabilitiesKHR::default(),
                h264_support
            ),
            Codec::H265 => {
                let profile_idc = h265_profile_idc(chroma_format, bit_depth)
                    .ok_or(ProfileQueryError::NoProfile)?;
                query!(
                    vk::VideoCodecOperationFlagsKHR::ENCODE_H265,
                    vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(profile_idc),
                    vk::VideoEncodeH265CapabilitiesKHR::default(),
                    h265_support
                )
            }
            Codec::AV1 => query!(
                vk::VideoCodecOperationFlagsKHR::ENCODE_AV1,
                vk::VideoEncodeAV1ProfileInfoKHR::default().std_profile(av1_profile(chroma_format)),
                vk::VideoEncodeAV1CapabilitiesKHR::default(),
                av1_support
            ),
        }
    }
}

/// Log the outcome of a profile query for `what`, and turn it into an
/// `Option`. A profile the driver does not support is ordinary and logged at
/// debug; any other failure is unexpected and logged at warn.
pub(crate) fn report<R>(what: &str, result: Result<R, ProfileQueryError>) -> Option<R> {
    match result {
        Ok(value) => {
            debug!("{what}: supported");
            Some(value)
        }
        Err(ProfileQueryError::NoProfile) => {
            debug!("{what}: no profile");
            None
        }
        Err(ProfileQueryError::Query(
            err @ (vk::Result::ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR
            | vk::Result::ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR
            | vk::Result::ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR),
        )) => {
            debug!("{what}: not supported ({err:?})");
            None
        }
        Err(ProfileQueryError::Query(err)) => {
            warn!("Failed to query {what} capabilities: {err:?}");
            None
        }
    }
}
