//! The video profile a decode session uses.

use ash::vk;

use crate::encoder::{BitDepth, ChromaFormat, Codec};

/// The video profile a decode session uses, owned so it can be named again
/// after creation.
///
/// The session, its DPB images, its output image and every bitstream buffer
/// it reads have to name the session's exact profile. The chain lives in one
/// heap allocation, so the pointer from the base struct to the codec struct
/// stays valid for as long as this does.
pub(crate) struct DecodeProfile(Box<ProfileChain>);

struct ProfileChain {
    info: vk::VideoProfileInfoKHR<'static>,
    h264: vk::VideoDecodeH264ProfileInfoKHR<'static>,
    h265: vk::VideoDecodeH265ProfileInfoKHR<'static>,
    av1: vk::VideoDecodeAV1ProfileInfoKHR<'static>,
}

// SAFETY: the raw pointer in the chain only points into the same boxed
// allocation, which moves with this value and is never shared.
unsafe impl Send for DecodeProfile {}

impl DecodeProfile {
    /// The profile that says whether a device decodes `codec` at all: 8-bit
    /// 4:2:0 in H.264 High, H.265 Main or AV1 Main.
    pub(crate) fn base(codec: Codec) -> Self {
        let (chroma, depth) = (ChromaFormat::Yuv420, BitDepth::Eight);
        match codec {
            Codec::H264 => Self::h264(
                ash::vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH,
                chroma,
                depth,
            ),
            Codec::H265 => Self::h265(
                ash::vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN,
                chroma,
                depth,
            ),
            Codec::AV1 => Self::av1(
                ash::vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN,
                false,
                chroma,
                depth,
            ),
        }
    }

    /// An H.264 decode profile.
    pub(crate) fn h264(
        std_profile_idc: ash::vk::native::StdVideoH264ProfileIdc,
        chroma_format: ChromaFormat,
        bit_depth: BitDepth,
    ) -> Self {
        let mut chain = Self::chain(Codec::H264, chroma_format, bit_depth);
        chain.h264 = vk::VideoDecodeH264ProfileInfoKHR::default()
            .std_profile_idc(std_profile_idc)
            .picture_layout(vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE);
        chain.info.p_next = (&chain.h264 as *const vk::VideoDecodeH264ProfileInfoKHR).cast();
        Self(chain)
    }

    /// An H.265 decode profile.
    pub(crate) fn h265(
        std_profile_idc: ash::vk::native::StdVideoH265ProfileIdc,
        chroma_format: ChromaFormat,
        bit_depth: BitDepth,
    ) -> Self {
        let mut chain = Self::chain(Codec::H265, chroma_format, bit_depth);
        chain.h265 = vk::VideoDecodeH265ProfileInfoKHR::default().std_profile_idc(std_profile_idc);
        chain.info.p_next = (&chain.h265 as *const vk::VideoDecodeH265ProfileInfoKHR).cast();
        Self(chain)
    }

    /// An AV1 decode profile. `film_grain` asks for a session that can apply
    /// film grain to the output.
    pub(crate) fn av1(
        std_profile: ash::vk::native::StdVideoAV1Profile,
        film_grain: bool,
        chroma_format: ChromaFormat,
        bit_depth: BitDepth,
    ) -> Self {
        let mut chain = Self::chain(Codec::AV1, chroma_format, bit_depth);
        chain.av1 = vk::VideoDecodeAV1ProfileInfoKHR::default()
            .std_profile(std_profile)
            .film_grain_support(film_grain);
        chain.info.p_next = (&chain.av1 as *const vk::VideoDecodeAV1ProfileInfoKHR).cast();
        Self(chain)
    }

    fn chain(codec: Codec, chroma_format: ChromaFormat, bit_depth: BitDepth) -> Box<ProfileChain> {
        let operation = match codec {
            Codec::H264 => vk::VideoCodecOperationFlagsKHR::DECODE_H264,
            Codec::H265 => vk::VideoCodecOperationFlagsKHR::DECODE_H265,
            Codec::AV1 => vk::VideoCodecOperationFlagsKHR::DECODE_AV1,
        };
        let depth: vk::VideoComponentBitDepthFlagsKHR = bit_depth.into();
        Box::new(ProfileChain {
            info: vk::VideoProfileInfoKHR::default()
                .video_codec_operation(operation)
                .chroma_subsampling(chroma_format.into())
                .luma_bit_depth(depth)
                .chroma_bit_depth(depth),
            h264: vk::VideoDecodeH264ProfileInfoKHR::default(),
            h265: vk::VideoDecodeH265ProfileInfoKHR::default(),
            av1: vk::VideoDecodeAV1ProfileInfoKHR::default(),
        })
    }

    /// The profile, with its chain.
    pub(crate) fn info(&self) -> &vk::VideoProfileInfoKHR<'_> {
        &self.0.info
    }
}
