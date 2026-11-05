//! The H.264 structures of a decode command.

use ash::vk;
use ash::vk::native::{StdVideoDecodeH264PictureInfo, StdVideoDecodeH264ReferenceInfo};

use crate::decoder::record::RecordCodec;

/// H.264's part of a [`DecodeRecord`](crate::decoder::record::DecodeRecord).
pub(super) struct H264Record;

impl RecordCodec for H264Record {
    type RefStd = StdVideoDecodeH264ReferenceInfo;
    type DpbSlot = vk::VideoDecodeH264DpbSlotInfoKHR<'static>;
    type Picture = StdVideoDecodeH264PictureInfo;
    type PictureInfo = vk::VideoDecodeH264PictureInfoKHR<'static>;

    fn dpb_slot(std: &'static StdVideoDecodeH264ReferenceInfo) -> Self::DpbSlot {
        vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(std)
    }

    fn picture_info(
        picture: &'static mut StdVideoDecodeH264PictureInfo,
        offsets: &'static [u32],
    ) -> Self::PictureInfo {
        vk::VideoDecodeH264PictureInfoKHR::default()
            .std_picture_info(picture)
            .slice_offsets(offsets)
    }
}
