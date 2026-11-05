//! The H.265 structures of a decode command.

use ash::vk;
use ash::vk::native::{StdVideoDecodeH265PictureInfo, StdVideoDecodeH265ReferenceInfo};

use crate::decoder::record::RecordCodec;

/// H.265's part of a [`DecodeRecord`](crate::decoder::record::DecodeRecord).
pub(super) struct H265Record;

impl RecordCodec for H265Record {
    type RefStd = StdVideoDecodeH265ReferenceInfo;
    type DpbSlot = vk::VideoDecodeH265DpbSlotInfoKHR<'static>;
    type Picture = StdVideoDecodeH265PictureInfo;
    type PictureInfo = vk::VideoDecodeH265PictureInfoKHR<'static>;

    fn dpb_slot(std: &'static StdVideoDecodeH265ReferenceInfo) -> Self::DpbSlot {
        vk::VideoDecodeH265DpbSlotInfoKHR::default().std_reference_info(std)
    }

    fn picture_info(
        picture: &'static mut StdVideoDecodeH265PictureInfo,
        offsets: &'static [u32],
    ) -> Self::PictureInfo {
        vk::VideoDecodeH265PictureInfoKHR::default()
            .std_picture_info(picture)
            .slice_segment_offsets(offsets)
    }
}
