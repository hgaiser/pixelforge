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
}

// SAFETY: the raw pointer in the chain only points into the same boxed
// allocation, which moves with this value and is never shared.
unsafe impl Send for DecodeProfile {}

impl DecodeProfile {
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
        })
    }

    /// The profile, with its chain.
    pub(crate) fn info(&self) -> &vk::VideoProfileInfoKHR<'_> {
        &self.0.info
    }
}
