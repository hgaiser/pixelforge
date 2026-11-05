//! Codec profile selection.
//!
//! The encoders' `create` functions and [`VideoContext::supports_encode_format`](crate::VideoContext::supports_encode_format)
//! both choose their profile here, so the query answers for exactly the profile an
//! encoder would request.

use super::{BitDepth, ChromaFormat, Codec, EncodeConfig};
use crate::error::{PixelForgeError, Result};
use ash::vk;

/// H.264 profile for a pixel format: High 4:4:4 Predictive for 4:4:4, High otherwise.
pub(crate) fn h264_profile_idc(chroma_format: ChromaFormat) -> u32 {
    match chroma_format {
        ChromaFormat::Yuv444 => {
            ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE
        }
        _ => ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH,
    }
}

/// H.265 profile for a pixel format and bit depth, or `None` for combinations the
/// encoder does not support (4:2:2).
pub(crate) fn h265_profile_idc(chroma_format: ChromaFormat, bit_depth: BitDepth) -> Option<u32> {
    match (chroma_format, bit_depth) {
        (ChromaFormat::Yuv420, BitDepth::Eight) => {
            Some(ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN)
        }
        (ChromaFormat::Yuv420, BitDepth::Ten) => {
            Some(ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10)
        }
        (ChromaFormat::Yuv444, _) => Some(
            ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS,
        ),
        _ => None,
    }
}

/// AV1 profile for a pixel format: Main is 4:2:0, High adds 4:4:4.
pub(crate) fn av1_profile(chroma_format: ChromaFormat) -> u32 {
    match chroma_format {
        ChromaFormat::Yuv420 => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN,
        _ => ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_HIGH,
    }
}

/// The video profile an encoder uses, owned so it can be named again after
/// creation.
///
/// Everything that touches the encoder's pictures has to name the session's
/// exact profile: the session itself, its input and DPB images, its query
/// pools, and any image a caller hands over to be read directly. The chain
/// lives in one heap allocation, so the pointers between its structs stay
/// valid for as long as this does.
pub(crate) struct EncodeProfile(Box<ProfileChain>);

struct ProfileChain {
    info: vk::VideoProfileInfoKHR<'static>,
    h264: vk::VideoEncodeH264ProfileInfoKHR<'static>,
    h265: vk::VideoEncodeH265ProfileInfoKHR<'static>,
    av1: vk::VideoEncodeAV1ProfileInfoKHR<'static>,
    usage: vk::VideoEncodeUsageInfoKHR<'static>,
    rgb: vk::VideoEncodeProfileRgbConversionInfoVALVE<'static>,
}

// SAFETY: the raw pointers in the chain only point into the same boxed
// allocation, which moves with this value and is never shared.
unsafe impl Send for EncodeProfile {}

impl EncodeProfile {
    /// The profile `config` asks for, with the RGB conversion struct when the
    /// encoder takes RGB input.
    pub(crate) fn new(config: &EncodeConfig, rgb_input: bool) -> Result<Self> {
        let operation = match config.codec {
            Codec::H264 => vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
            Codec::H265 => vk::VideoCodecOperationFlagsKHR::ENCODE_H265,
            Codec::AV1 => vk::VideoCodecOperationFlagsKHR::ENCODE_AV1,
        };
        let h265_idc = match config.codec {
            Codec::H265 => {
                h265_profile_idc(config.chroma_format, config.bit_depth).ok_or_else(|| {
                    PixelForgeError::InvalidInput(format!(
                        "Unsupported pixel format / bit depth for H.265: {:?} / {:?}",
                        config.chroma_format, config.bit_depth
                    ))
                })?
            }
            _ => 0,
        };
        let bit_depth: vk::VideoComponentBitDepthFlagsKHR = config.bit_depth.into();
        let mut chain = Box::new(ProfileChain {
            info: vk::VideoProfileInfoKHR::default()
                .video_codec_operation(operation)
                .chroma_subsampling(config.chroma_format.into())
                .luma_bit_depth(bit_depth)
                .chroma_bit_depth(bit_depth),
            h264: vk::VideoEncodeH264ProfileInfoKHR::default()
                .std_profile_idc(h264_profile_idc(config.chroma_format)),
            h265: vk::VideoEncodeH265ProfileInfoKHR::default().std_profile_idc(h265_idc),
            av1: vk::VideoEncodeAV1ProfileInfoKHR::default()
                .std_profile(av1_profile(config.chroma_format)),
            usage: vk::VideoEncodeUsageInfoKHR::default()
                .video_usage_hints(config.encode_usage_hint.into())
                .video_content_hints(config.encode_content_hint.into())
                .tuning_mode(config.encoder_tuning_mode.into()),
            rgb: vk::VideoEncodeProfileRgbConversionInfoVALVE::default()
                .perform_encode_rgb_conversion(true),
        });

        // info -> codec profile -> usage [-> rgb]
        let chain_ref = &mut *chain;
        let codec_profile: *mut vk::BaseOutStructure = match config.codec {
            Codec::H264 => (&mut chain_ref.h264 as *mut vk::VideoEncodeH264ProfileInfoKHR).cast(),
            Codec::H265 => (&mut chain_ref.h265 as *mut vk::VideoEncodeH265ProfileInfoKHR).cast(),
            Codec::AV1 => (&mut chain_ref.av1 as *mut vk::VideoEncodeAV1ProfileInfoKHR).cast(),
        };
        let usage: *mut vk::BaseOutStructure =
            (&mut chain_ref.usage as *mut vk::VideoEncodeUsageInfoKHR).cast();
        chain_ref.info.p_next = codec_profile.cast_const().cast();
        unsafe { (*codec_profile).p_next = usage };
        if rgb_input {
            chain_ref.usage.p_next =
                (&chain_ref.rgb as *const vk::VideoEncodeProfileRgbConversionInfoVALVE).cast();
        }
        Ok(Self(chain))
    }

    /// The profile, with its chain.
    pub(crate) fn info(&self) -> &vk::VideoProfileInfoKHR<'_> {
        &self.0.info
    }
}
