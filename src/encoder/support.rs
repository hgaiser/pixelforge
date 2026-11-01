//! Device support queries for encode formats.

use super::codec::query_video_capabilities;
use super::profile::{av1_profile, h264_profile_idc, h265_profile_idc};
use super::rgb::rgb_conversion_profile;
use super::{BitDepth, Codec, PixelFormat};
use crate::vulkan::VideoContext;
use ash::vk;
use ash::vk::TaggedStructure;
use tracing::{debug, warn};

/// Why [`VideoContext::query_encode_profile`] has no answer to give.
pub(crate) enum ProfileQueryError {
    /// The encoder has no profile for this combination (H.265 4:2:2).
    NoProfile,
    /// The driver rejected the profile's capability query.
    Query(vk::Result),
}

impl VideoContext {
    /// Check whether the device can encode `codec` in the given pixel format and
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
        pixel_format: PixelFormat,
        bit_depth: BitDepth,
    ) -> bool {
        if !self.supports_encode(codec) {
            return false;
        }
        let result = self.query_encode_profile(codec, pixel_format, bit_depth, None, |_| ());
        report(
            &format!("{codec:?} encode {pixel_format:?} {bit_depth:?}"),
            result,
        )
        .is_some()
    }

    /// Build the profile an encoder would request for `codec`, `pixel_format`
    /// and `bit_depth`, ask the driver for its capabilities, and on success
    /// hand the profile to `then` for any further query about it.
    ///
    /// With `rgb_caps`, the profile is the RGB input one an encoder built with
    /// [`EncodeConfig::with_rgb_input`](crate::EncodeConfig::with_rgb_input)
    /// requests, and the driver fills in what its conversion supports.
    ///
    /// The encoders build their profiles themselves, since theirs also carry
    /// usage hints, but from the same profile numbers in
    /// [`super::profile`], so this answers for the profile they use.
    pub(crate) fn query_encode_profile<R>(
        &self,
        codec: Codec,
        pixel_format: PixelFormat,
        bit_depth: BitDepth,
        mut rgb_caps: Option<&mut vk::VideoEncodeRgbConversionCapabilitiesVALVE<'static>>,
        then: impl FnOnce(&vk::VideoProfileInfoKHR<'_>) -> R,
    ) -> Result<R, ProfileQueryError> {
        let chroma_subsampling: vk::VideoChromaSubsamplingFlagsKHR = pixel_format.into();
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
            ($operation:expr, $profile:expr, $codec_caps:expr) => {{
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
                    vk::Result::SUCCESS => Ok(then(&profile_info)),
                    err => Err(ProfileQueryError::Query(err)),
                }
            }};
        }

        match codec {
            Codec::H264 => query!(
                vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
                vk::VideoEncodeH264ProfileInfoKHR::default()
                    .std_profile_idc(h264_profile_idc(pixel_format)),
                vk::VideoEncodeH264CapabilitiesKHR::default()
            ),
            Codec::H265 => {
                let profile_idc = h265_profile_idc(pixel_format, bit_depth)
                    .ok_or(ProfileQueryError::NoProfile)?;
                query!(
                    vk::VideoCodecOperationFlagsKHR::ENCODE_H265,
                    vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(profile_idc),
                    vk::VideoEncodeH265CapabilitiesKHR::default()
                )
            }
            Codec::AV1 => query!(
                vk::VideoCodecOperationFlagsKHR::ENCODE_AV1,
                vk::VideoEncodeAV1ProfileInfoKHR::default().std_profile(av1_profile(pixel_format)),
                vk::VideoEncodeAV1CapabilitiesKHR::default()
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
