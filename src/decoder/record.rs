//! Storage for the structures one decode command is recorded from.
//!
//! NVIDIA's decoder does not copy `VkVideoDecodeInfoKHR` and the structures
//! hanging off it when the command is recorded: it reads them again at
//! `vkQueueSubmit`. They therefore have to stay alive and unmoved until the
//! submission has finished, which a stack frame in the recording function does
//! not guarantee. Left on the stack they are garbage by submit time, which
//! crashed the driver or made it decode B-pictures from stale reference data.
//!
//! A [`DecodeRecord`] owns all of them. There is one per pipeline slot, and it
//! is rebuilt only once that slot's previous submission has completed. The
//! structures every codec shares live here; a [`RecordCodec`] supplies the
//! codec's own.

use ash::vk;
use ash::vk::TaggedStructure;

/// Most references one picture can have: every DPB slot but the one being set
/// up, for the codec with the most slots (H.264 and H.265, 16 references).
const MAX_REFS: usize = 17;

/// The codec-specific structures of a decode command.
pub(crate) trait RecordCodec {
    /// The Std reference info of one DPB slot.
    type RefStd: Copy + 'static;
    /// The codec's `Vk...DpbSlotInfoKHR`, which points at a [`Self::RefStd`].
    type DpbSlot: Copy
        + Default
        + 'static
        + vk::Extends<vk::VideoReferenceSlotInfoKHR<'static>>
        + TaggedStructure<'static>;
    /// What the codec's picture info points at: the Std picture info and
    /// anything it points at in turn. Kept in the record, so it stays in place
    /// until the submission has completed.
    type Picture: 'static;
    /// The codec's `Vk...PictureInfoKHR`.
    type PictureInfo: Default
        + 'static
        + vk::Extends<vk::VideoDecodeInfoKHR<'static>>
        + TaggedStructure<'static>;

    /// A DPB slot info pointing at `std`.
    fn dpb_slot(std: &'static Self::RefStd) -> Self::DpbSlot;

    /// The picture info for `picture`, whose slices or tiles start at
    /// `offsets` in the bitstream buffer.
    ///
    /// May also wire the pointers inside `picture` itself.
    fn picture_info(
        picture: &'static mut Self::Picture,
        offsets: &'static [u32],
    ) -> Self::PictureInfo;
}

/// Everything one decode command points at.
///
/// The Vulkan structures here point at each other and at the plain data beside
/// them, so the whole is a self-referential value. That is sound because it lives in
/// heap storage it is never moved out of once built, and every pointer is rebuilt
/// by [`DecodeRecord::link`] after the last write to what it points at. The
/// `'static` lifetimes on the fields are internal: nothing outside this module
/// can obtain a reference that outlives the record.
///
/// Every collection is a fixed-size array holding `ref_count` live entries, so
/// nothing is ever reallocated and an element's address does not change however
/// the record is filled.
pub(crate) struct DecodeRecord<C: RecordCodec> {
    // Plain data.
    offsets: Vec<u32>,
    ref_count: usize,
    /// DPB slot, image view and coded extent of each reference. A `None`
    /// extent is the picture's own.
    refs: [(u8, vk::ImageView, Option<vk::Extent2D>); MAX_REFS],
    ref_std: [C::RefStd; MAX_REFS],
    setup_std: C::RefStd,
    picture: C::Picture,

    // Picture resources.
    ref_resources: [vk::VideoPictureResourceInfoKHR<'static>; MAX_REFS],
    setup_resource: vk::VideoPictureResourceInfoKHR<'static>,
    output_resource: vk::VideoPictureResourceInfoKHR<'static>,

    // Codec-specific slot info. The begin and decode commands each chain their
    // own copy, since chaining takes the structure mutably.
    begin_dpb: [C::DpbSlot; MAX_REFS],
    decode_dpb: [C::DpbSlot; MAX_REFS],
    setup_dpb: C::DpbSlot,

    // Reference slots. `begin_slots` has room for the slot being set up too.
    begin_slots: [vk::VideoReferenceSlotInfoKHR<'static>; MAX_REFS + 1],
    decode_slots: [vk::VideoReferenceSlotInfoKHR<'static>; MAX_REFS],
    setup_slot: vk::VideoReferenceSlotInfoKHR<'static>,

    // Top level.
    pic_info: C::PictureInfo,
    begin_info: vk::VideoBeginCodingInfoKHR<'static>,
    decode_info: vk::VideoDecodeInfoKHR<'static>,
}

// SAFETY: the raw pointers inside the Vulkan structures only ever point into
// the same record, which is owned by one decoder and used from one thread at a
// time.
unsafe impl<C: RecordCodec> Send for DecodeRecord<C> {}

/// What [`DecodeRecord::link`] needs besides the references already pushed.
pub(crate) struct LinkParams<C: RecordCodec> {
    pub extent: vk::Extent2D,
    pub session: vk::VideoSessionKHR,
    pub session_params: vk::VideoSessionParametersKHR,
    pub bitstream: vk::Buffer,
    pub bitstream_range: u64,
    /// DPB slot, and view of it, that the picture is reconstructed into.
    pub setup_slot: u8,
    pub setup_view: vk::ImageView,
    /// The decode destination when it is not the DPB slot itself.
    pub output_view: Option<vk::ImageView>,
    pub setup_std: C::RefStd,
}

/// Extend a borrow to any lifetime.
///
/// # Safety
/// `value` must stay alive and in place for as long as the result is used.
unsafe fn stable<'a, T: ?Sized>(value: &T) -> &'a T {
    unsafe { &*(value as *const T) }
}

/// As [`stable`], for a mutable borrow.
///
/// # Safety
/// As [`stable`], and nothing else may access `value` while the result is used.
unsafe fn stable_mut<'a, T: ?Sized>(value: &mut T) -> &'a mut T {
    unsafe { &mut *(value as *mut T) }
}

impl<C: RecordCodec> DecodeRecord<C> {
    /// An empty record. `picture` is the codec's picture storage, which
    /// [`Self::link`] overwrites.
    pub(crate) fn new(picture: C::Picture) -> Self {
        // SAFETY: the standard structures are plain C data, for which all-zero
        // is a valid (if meaningless) value.
        let zeroed_ref: C::RefStd = unsafe { std::mem::zeroed() };
        Self {
            offsets: Vec::new(),
            ref_count: 0,
            refs: [(0, vk::ImageView::null(), None); MAX_REFS],
            ref_std: [zeroed_ref; MAX_REFS],
            setup_std: zeroed_ref,
            picture,
            ref_resources: [Default::default(); MAX_REFS],
            setup_resource: Default::default(),
            output_resource: Default::default(),
            begin_dpb: [Default::default(); MAX_REFS],
            decode_dpb: [Default::default(); MAX_REFS],
            setup_dpb: Default::default(),
            begin_slots: [Default::default(); MAX_REFS + 1],
            decode_slots: [Default::default(); MAX_REFS],
            setup_slot: Default::default(),
            pic_info: Default::default(),
            begin_info: Default::default(),
            decode_info: Default::default(),
        }
    }

    /// Start a new picture, discarding the last one.
    ///
    /// Only call once the submission that used this record has completed.
    /// `offsets` are where the picture's slices or tiles start in the
    /// bitstream buffer.
    pub(crate) fn reset(&mut self, offsets: &[u32]) {
        self.offsets.clear();
        self.offsets.extend_from_slice(offsets);
        self.ref_count = 0;
    }

    /// The codec's picture storage, to fill before [`Self::link`].
    pub(crate) fn picture_mut(&mut self) -> &mut C::Picture {
        &mut self.picture
    }

    /// Add a reference picture, in the order the driver is to see it.
    ///
    /// `extent` is the reference's coded extent when it differs from the
    /// picture's own, which only AV1 allows (a frame size change between key
    /// frames).
    pub(crate) fn push_reference(
        &mut self,
        slot: u8,
        view: vk::ImageView,
        extent: Option<vk::Extent2D>,
        std: C::RefStd,
    ) {
        assert!(self.ref_count < MAX_REFS, "too many references");
        self.refs[self.ref_count] = (slot, view, extent);
        self.ref_std[self.ref_count] = std;
        self.ref_count += 1;
    }

    /// Build the structures from what has been pushed, wiring up every pointer.
    ///
    /// Order matters: the plain data and the structures pointed at are written
    /// before the structures that point at them.
    pub(crate) fn link(&mut self, p: &LinkParams<C>) {
        let n = self.ref_count;
        self.setup_std = p.setup_std;

        let resource = |view: vk::ImageView, extent: vk::Extent2D| {
            vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D { x: 0, y: 0 })
                .coded_extent(extent)
                .base_array_layer(0)
                .image_view_binding(view)
        };
        for i in 0..n {
            let (_, view, extent) = self.refs[i];
            self.ref_resources[i] = resource(view, extent.unwrap_or(p.extent));
        }
        self.setup_resource = resource(p.setup_view, p.extent);
        self.output_resource = match p.output_view {
            None => self.setup_resource,
            Some(view) => resource(view, p.extent),
        };

        // SAFETY (for every `stable`/`stable_mut` below): the target is a field
        // of `self`, which is boxed and not written again until the next
        // `reset`/`link`, and fixed arrays never move their elements.
        unsafe {
            for i in 0..n {
                let info = C::dpb_slot(stable(&self.ref_std[i]));
                self.begin_dpb[i] = info;
                self.decode_dpb[i] = info;
            }
            self.setup_dpb = C::dpb_slot(stable(&self.setup_std));

            for i in 0..n {
                let slot = self.refs[i].0 as i32;
                let resource = stable(&self.ref_resources[i]);
                self.begin_slots[i] = vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(slot)
                    .picture_resource(resource)
                    .push(stable_mut(&mut self.begin_dpb[i]));
                self.decode_slots[i] = vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(slot)
                    .picture_resource(resource)
                    .push(stable_mut(&mut self.decode_dpb[i]));
            }
            // The slot being reconstructed is listed to `begin` with slot index
            // -1: it is not active until the decode's setup slot activates it,
            // and a non-negative index here must name an already-active slot
            // (VUID-vkCmdBeginVideoCodingKHR-slotIndex-07239). -1 still binds
            // its picture resource.
            self.begin_slots[n] = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(-1)
                .picture_resource(stable(&self.setup_resource));
            self.setup_slot = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(p.setup_slot as i32)
                .picture_resource(stable(&self.setup_resource))
                .push(stable_mut(&mut self.setup_dpb));

            self.pic_info =
                C::picture_info(stable_mut(&mut self.picture), stable(&self.offsets[..]));

            self.begin_info = vk::VideoBeginCodingInfoKHR::default()
                .video_session(p.session)
                .video_session_parameters(p.session_params)
                .reference_slots(stable(&self.begin_slots[..n + 1]));
            self.decode_info = vk::VideoDecodeInfoKHR::default()
                .src_buffer(p.bitstream)
                .src_buffer_offset(0)
                .src_buffer_range(p.bitstream_range)
                .dst_picture_resource(self.output_resource)
                .setup_reference_slot(stable(&self.setup_slot))
                .reference_slots(stable(&self.decode_slots[..n]))
                .push(stable_mut(&mut self.pic_info));
        }
    }

    /// The `vkCmdBeginVideoCodingKHR` argument. Valid after [`Self::link`].
    pub(crate) fn begin_info(&self) -> &vk::VideoBeginCodingInfoKHR<'_> {
        &self.begin_info
    }

    /// The `vkCmdDecodeVideoKHR` argument. Valid after [`Self::link`].
    pub(crate) fn decode_info(&self) -> &vk::VideoDecodeInfoKHR<'_> {
        &self.decode_info
    }
}
