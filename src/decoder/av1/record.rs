//! The AV1 structures of a decode command.

use ash::vk;
use ash::vk::native::StdVideoDecodeAV1ReferenceInfo;

use super::consts::REFS_PER_FRAME;
use super::std_info::StdPictureInfo;
use crate::decoder::record::RecordCodec;

/// AV1's part of a [`DecodeRecord`](crate::decoder::record::DecodeRecord).
pub(super) struct Av1Record;

/// What the AV1 picture info points at, besides the tile offsets the record
/// keeps.
#[derive(Default)]
pub(super) struct Av1Picture {
    /// The Std picture info and its sub-structures. `None` only before the
    /// first frame.
    pub std: Option<StdPictureInfo>,
    /// The DPB slot of each of LAST_FRAME..=ALTREF_FRAME, or -1.
    pub reference_name_slot_indices: [i32; REFS_PER_FRAME],
    /// Where the frame header OBU starts in the bitstream buffer. Always 0:
    /// the buffer holds tile data alone.
    pub frame_header_offset: u32,
    /// The size of each tile, in the order of the record's tile offsets.
    pub tile_sizes: Vec<u32>,
}

impl RecordCodec for Av1Record {
    type RefStd = StdVideoDecodeAV1ReferenceInfo;
    type DpbSlot = vk::VideoDecodeAV1DpbSlotInfoKHR<'static>;
    type Picture = Av1Picture;
    type PictureInfo = vk::VideoDecodeAV1PictureInfoKHR<'static>;

    fn dpb_slot(std: &'static StdVideoDecodeAV1ReferenceInfo) -> Self::DpbSlot {
        vk::VideoDecodeAV1DpbSlotInfoKHR::default().std_reference_info(std)
    }

    fn picture_info(
        picture: &'static mut Av1Picture,
        offsets: &'static [u32],
    ) -> Self::PictureInfo {
        let std = picture
            .std
            .as_ref()
            .expect("the picture info is filled before linking")
            .get();
        vk::VideoDecodeAV1PictureInfoKHR::default()
            .std_picture_info(std)
            .reference_name_slot_indices(picture.reference_name_slot_indices)
            .frame_header_offset(picture.frame_header_offset)
            .tile_offsets(offsets)
            .tile_sizes(&picture.tile_sizes)
    }
}
