//! Frame ownership: what a decoded picture is copied into, what keeps it alive,
//! and how display order is restored.
//!
//! The decoder hands out [`DecodedFrame`]s whose storage stays reserved until
//! the caller drops them. Normally that storage is the DPB slot the picture was
//! decoded into, pinned so the codec allocates around it: no copy happens
//! between the decoder and the caller, and reordering into display order works
//! by holding pins rather than by copying pictures out. A pool image (a copy) is
//! the fallback for devices with no slots to spare, and for drivers that decode
//! into a distinct output image the next picture overwrites.

use std::sync::{Arc, Condvar, Mutex};

use ash::vk::{self, TaggedStructure as _};

use crate::decoder::DecodedFrame;
use crate::decoder::common::{DecoderCommon, DpbImage};
use crate::encoder::{BitDepth, ChromaFormat};
use crate::error::{PixelForgeError, Result};
use crate::video::find_memory_type;
use crate::vulkan::VideoContext;

/// A freshly decoded picture, before it is handed to the caller.
///
/// The pixels live in a DPB slot (or the session's decode output image), so this
/// descriptor is only valid until the next decode submission. [`ReorderBuffer`]
/// turns it into a [`DecodedFrame`], which owns its storage.
pub(crate) struct DecodedPicture {
    /// DPB slot the picture was decoded into.
    pub slot: u8,
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub layout: vk::ImageLayout,
    pub array_layer: u32,
    pub chroma_format: ChromaFormat,
    pub bit_depth: BitDepth,
    pub generation: u64,
    pub width: u32,
    pub height: u32,
    pub coded_width: u32,
    pub coded_height: u32,
    pub pts: u64,
    pub display_order: i32,
    pub is_keyframe: bool,
    /// The picture is in its slot's film grain output image, not in the slot.
    pub film_grain: bool,
}

/// An image in the frame pool, by index.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PoolIndex(usize);

/// DPB slots reserved by frames the caller is still holding.
///
/// A pinned slot is one the driver decoded into and the caller is now reading,
/// so the codec must not decode over it. Releases come from [`FramePin::drop`]
/// on whatever thread the frame died on, which is why the decoder waits on the
/// condition variable rather than polling.
///
/// A slot can be pinned more than once: AV1 can show the same picture again
/// (`show_existing_frame`), and each frame that shows it holds its own pin.
/// The slot stays pinned until the last of them is released.
#[derive(Debug, Default)]
pub(crate) struct SlotPins {
    state: Mutex<PinState>,
    released: Condvar,
}

#[derive(Debug, Default)]
struct PinState {
    /// Pins held on each DPB slot.
    counts: [u16; MAX_PINNABLE_SLOTS],
    /// Releases so far, so a waiter can tell that one happened.
    releases: u64,
}

/// The most DPB slots any codec uses: 16 references plus the current picture
/// for H.264 and H.265, with room to spare.
const MAX_PINNABLE_SLOTS: usize = 32;

impl SlotPins {
    pub fn is_pinned(&self, slot: u8) -> bool {
        self.state.lock().unwrap().counts[slot as usize] != 0
    }

    pub fn any_pinned(&self) -> bool {
        self.state.lock().unwrap().counts.iter().any(|&c| c != 0)
    }

    /// How many slots are pinned right now, across everything holding one:
    /// pictures awaiting their turn in display order, frames on their way to
    /// the caller, and frames the caller has not dropped. A slot pinned more
    /// than once counts once, since it holds one slot.
    pub fn count(&self) -> usize {
        let state = self.state.lock().unwrap();
        state.counts.iter().filter(|&&c| c != 0).count()
    }

    pub fn pin(&self, slot: u8) {
        self.state.lock().unwrap().counts[slot as usize] += 1;
    }

    pub fn release(&self, slot: u8) {
        let mut state = self.state.lock().unwrap();
        let count = &mut state.counts[slot as usize];
        debug_assert!(*count > 0, "released slot {slot} was not pinned");
        *count = count.saturating_sub(1);
        state.releases += 1;
        self.released.notify_all();
    }

    /// Block until at least one pin is released. Returns immediately if
    /// nothing is pinned, since then no release can ever come.
    pub fn wait_for_release(&self) {
        let mut state = self.state.lock().unwrap();
        if state.counts.iter().all(|&c| c == 0) {
            return;
        }
        let before = state.releases;
        while state.releases == before {
            state = self.released.wait(state).unwrap();
        }
    }
}

/// Pins released by frames the caller has dropped, waiting to be reclaimed.
///
/// A [`DecodedFrame`] can be dropped on any thread, including while the decoder
/// is submitting the next picture, so releases land here and the decoder folds
/// them in the next time it needs storage.
#[derive(Debug, Default)]
pub(crate) struct ReleaseQueue {
    released: Mutex<Vec<PoolIndex>>,
}

impl ReleaseQueue {
    fn push(&self, index: PoolIndex) {
        self.released.lock().unwrap().push(index);
    }

    fn take(&self) -> Vec<PoolIndex> {
        std::mem::take(&mut *self.released.lock().unwrap())
    }
}

/// Keeps a [`DecodedFrame`]'s storage reserved for as long as the frame lives.
///
/// Dropping the frame drops this, which returns the storage to the decoder.
#[derive(Debug)]
pub(crate) enum FramePin {
    /// A copy in the frame pool. Released lazily: the decoder never waits on
    /// pool images, it just allocates another one.
    Pool {
        index: PoolIndex,
        releases: Arc<ReleaseQueue>,
    },
    /// The DPB slot the picture was decoded into, handed out without a copy.
    /// Releases the slot on drop, and holds the image itself so that a session
    /// rebuilt while this frame is alive cannot take the pixels away.
    DpbSlot {
        slot: u8,
        pins: Arc<SlotPins>,
        #[allow(dead_code)]
        image: Arc<DpbImage>,
    },
}

impl Drop for FramePin {
    fn drop(&mut self) {
        match self {
            FramePin::Pool { index, releases } => releases.push(*index),
            FramePin::DpbSlot { slot, pins, .. } => pins.release(*slot),
        }
    }
}

/// State of one image in the frame pool.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PoolState {
    /// Reusable.
    Free,
    /// Holds a decoded picture awaiting its turn in display order.
    Buffered,
    /// Handed to the caller; reserved until their [`DecodedFrame`] is dropped.
    HandedOut,
}

/// One pooled image, sized to the picture it currently holds.
///
/// No image view: pixelforge only ever writes to a pool image, and a consumer
/// reading one needs a view built for how they intend to read it (a sampled
/// multi-planar view needs a ycbcr conversion, a copy needs no view at all).
struct PoolImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    coded_width: u32,
    coded_height: u32,
    format: vk::Format,
    state: PoolState,
}

/// Images that decoded pictures are copied into so they outlive their DPB slot.
///
/// The fallback path only: most pictures are pinned in place instead. Used when
/// the session has no spare DPB slot to pin, or when the driver decodes into a
/// distinct output image rather than into the DPB.
///
/// The pool grows on demand and shrinks never: an image is reused as soon as the
/// frame holding it is dropped, so steady-state size is "what the caller holds
/// at once, plus the reorder depth". Nothing here blocks, because pool images
/// are ordinary allocations rather than a hardware-bounded resource.
struct FramePool {
    images: Vec<PoolImage>,
    releases: Arc<ReleaseQueue>,
}

impl FramePool {
    fn new() -> Self {
        Self {
            images: Vec::new(),
            releases: Arc::new(ReleaseQueue::default()),
        }
    }

    /// Fold in the images released by dropped frames.
    fn reclaim(&mut self) {
        for PoolIndex(i) in self.releases.take() {
            self.images[i].state = PoolState::Free;
        }
    }

    /// A free image matching `picture`'s geometry, creating or resizing one as
    /// needed. Resolution changes recreate a mismatched image.
    fn acquire(&mut self, common: &DecoderCommon, picture: &DecodedPicture) -> Result<usize> {
        self.reclaim();

        let format = common
            .session
            .as_ref()
            .map(|s| s.picture_format)
            .expect("session active while decoding");

        let matching = self.images.iter().position(|p| {
            p.state == PoolState::Free
                && p.coded_width == picture.coded_width
                && p.coded_height == picture.coded_height
                && p.format == format
        });
        if let Some(i) = matching {
            return Ok(i);
        }

        let (image, memory) = create_pool_image(
            &common.context,
            picture.coded_width,
            picture.coded_height,
            format,
            &common.picture_sharing_families(),
        )?;
        let slot = PoolImage {
            image,
            memory,
            coded_width: picture.coded_width,
            coded_height: picture.coded_height,
            format,
            state: PoolState::Free,
        };

        // Reuse a free-but-mismatched image if one exists, else grow the pool.
        if let Some(i) = self.images.iter().position(|p| p.state == PoolState::Free) {
            self.destroy_image(common, i);
            self.images[i] = slot;
            Ok(i)
        } else {
            self.images.push(slot);
            Ok(self.images.len() - 1)
        }
    }

    /// Mark `index` as handed to the caller and mint its pin.
    fn hand_out(&mut self, index: usize) -> FramePin {
        self.images[index].state = PoolState::HandedOut;
        FramePin::Pool {
            index: PoolIndex(index),
            releases: self.releases.clone(),
        }
    }

    fn destroy_image(&mut self, common: &DecoderCommon, i: usize) {
        let p = &self.images[i];
        if p.image == vk::Image::null() {
            return;
        }
        unsafe {
            common.context.device().device_wait_idle().ok();
            common.context.device().destroy_image(p.image, None);
            common.context.device().free_memory(p.memory, None);
        }
    }

    /// Whether any frame handed to the caller is still alive.
    fn has_live_frames(&mut self) -> bool {
        self.reclaim();
        self.images.iter().any(|p| p.state == PoolState::HandedOut)
    }

    /// Free every pool image. The caller must be done with handed-out frames.
    fn destroy(&mut self, common: &DecoderCommon) {
        for i in 0..self.images.len() {
            self.destroy_image(common, i);
            self.images[i].image = vk::Image::null();
        }
        self.images.clear();
    }
}

/// Where a buffered picture's pixels live until it is emitted.
///
/// Pinning is the normal case: the picture stays in the DPB slot it was decoded
/// into and the codec is told not to reuse that slot, so no copy happens
/// anywhere between the decoder and the caller. Copying is the fallback for the
/// two cases where the pixels cannot survive in place: the device gave the
/// session no spare slots to pin, or the driver decodes into a single distinct
/// output image that the next picture overwrites.
enum Retained {
    /// The DPB slot the picture was decoded into, pinned until the frame dies.
    Slot {
        pin: FramePin,
        image: vk::Image,
        image_view: vk::ImageView,
        layout: vk::ImageLayout,
        array_layer: u32,
    },
    /// A private copy in the frame pool.
    Pool { index: usize },
}

/// A buffered picture, awaiting its turn in display order.
struct ReorderEntry {
    retained: Retained,
    /// Whether this picture's image can be sampled without a copy. Recorded at
    /// retain time, since it depends on which storage the picture ended up in.
    sampleable: bool,
    /// Whether this picture's planes can be viewed separately. Same reason.
    plane_views: bool,
    /// Which set of decoder images this picture's storage belongs to.
    generation: u64,
    display_order: i32,
    pts: u64,
    is_keyframe: bool,
    width: u32,
    height: u32,
    coded_width: u32,
    coded_height: u32,
    chroma_format: ChromaFormat,
    bit_depth: BitDepth,
}

/// Reorders decoded pictures from decode order into display order.
///
/// The hardware produces pictures in decode order, which for streams with
/// B-frames is not display order. A picture therefore has to survive while
/// later pictures are decoded ahead of it. It survives *in place*: the DPB slot
/// holding it is pinned, so the codec allocates around it, and the picture is
/// handed to the caller without ever being copied. The pin then belongs to the
/// caller's [`DecodedFrame`] and is released when they drop it.
///
/// Emission follows the DPB bumping model: a picture is held until at most
/// `reorder_depth` pictures precede it in the buffer, a keyframe drains the
/// previous coded video sequence (display order restarts there), and
/// [`Self::flush`] drains the rest at end of stream. A stream without B-frames
/// has a reorder depth of zero, so each picture is emitted by the same call
/// that decoded it and nothing is buffered at all.
///
/// Slots are finite, so pinning is bounded by what the session reserved. Past
/// that bound a picture is copied into a pool image instead, which costs a copy
/// but keeps decoding correct on devices too small for the stream.
pub(crate) struct ReorderBuffer {
    pool: FramePool,
    buffered: Vec<ReorderEntry>,
}

impl ReorderBuffer {
    pub fn new() -> Self {
        Self {
            pool: FramePool::new(),
            buffered: Vec::new(),
        }
    }

    /// Add a freshly decoded picture and return whatever is now ready to output.
    ///
    /// `reorder_depth` is how many pictures may precede a held one before it
    /// has to be emitted.
    pub fn push(
        &mut self,
        common: &mut DecoderCommon,
        picture: &DecodedPicture,
        reorder_depth: usize,
    ) -> Result<Vec<DecodedFrame>> {
        let mut out = Vec::new();
        // Display order restarts at a keyframe, so the previous sequence must be
        // fully drained before this picture, which belongs to the new one.
        if picture.is_keyframe {
            out.extend(self.drain_all());
        }

        let retained = self.retain(common, picture)?;
        // A pool copy is always sampleable; a pinned DPB image only if the
        // device gave the session a picture format supporting SAMPLED.
        let (sampleable, plane_views) = match retained {
            Retained::Slot { .. } => {
                let session = common.session()?;
                (session.sampleable, session.plane_views)
            }
            // A pool image is an ordinary image: both are unconditional.
            Retained::Pool { .. } => (true, true),
        };
        self.buffered.push(ReorderEntry {
            retained,
            sampleable,
            plane_views,
            generation: picture.generation,
            display_order: picture.display_order,
            pts: picture.pts,
            is_keyframe: picture.is_keyframe,
            width: picture.width,
            height: picture.height,
            coded_width: picture.coded_width,
            coded_height: picture.coded_height,
            chroma_format: picture.chroma_format,
            bit_depth: picture.bit_depth,
        });

        while self.buffered.len() > reorder_depth {
            out.push(self.pop_min_display_order());
        }
        Ok(out)
    }

    /// Make a decoded picture outlive the decode that produced it.
    ///
    /// Pins its DPB slot when the session has room to spare, and copies it out
    /// otherwise. See [`Retained`].
    fn retain(&mut self, common: &mut DecoderCommon, picture: &DecodedPicture) -> Result<Retained> {
        let session = common.session()?;
        //
        // The budget counts every pin that exists, not just this buffer's:
        // frames already emitted hold theirs until the caller drops them, and a
        // single decode call can emit many. Staying within `spare_slots` is what
        // guarantees the codec always has a slot to decode into, so a caller who
        // holds frames gets copies rather than a decoder that cannot proceed.
        if session.pinnable && session.slot_pins.count() < session.spare_slots {
            session.slot_pins.pin(picture.slot);
            let entry = if picture.film_grain {
                session.grain_entry(picture.slot)
            } else {
                session.dpb_entry(picture.slot).0
            };
            return Ok(Retained::Slot {
                pin: FramePin::DpbSlot {
                    slot: picture.slot,
                    pins: session.slot_pins.clone(),
                    // Holding the image is what keeps it alive past a session
                    // rebuild, so the frame stays readable for its whole life.
                    image: entry.clone(),
                },
                image: picture.image,
                image_view: picture.image_view,
                layout: picture.layout,
                array_layer: picture.array_layer,
            });
        }

        let index = self.pool.acquire(common, picture)?;
        common.record_picture_copy(picture, self.pool.images[index].image)?;
        common.submit_copy()?;
        self.pool.images[index].state = PoolState::Buffered;
        Ok(Retained::Pool { index })
    }

    /// Emit every buffered picture in display order. Call at end of stream.
    pub fn flush(&mut self) -> Vec<DecodedFrame> {
        self.drain_all()
    }

    /// Drain the whole buffer in ascending display order.
    fn drain_all(&mut self) -> Vec<DecodedFrame> {
        let mut out = Vec::with_capacity(self.buffered.len());
        while !self.buffered.is_empty() {
            out.push(self.pop_min_display_order());
        }
        out
    }

    /// Remove and return the buffered picture that comes first in display order.
    fn pop_min_display_order(&mut self) -> DecodedFrame {
        let i = self
            .buffered
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.display_order)
            .map(|(i, _)| i)
            .expect("buffer is non-empty");
        let entry = self.buffered.remove(i);

        // Hand the storage over: the pin moves into the frame, so the slot or
        // pool image stays reserved until the caller drops it.
        let (image, image_view, layout, array_layer, pin) = match entry.retained {
            Retained::Slot {
                pin,
                image,
                image_view,
                layout,
                array_layer,
            } => (image, image_view, layout, array_layer, pin),
            Retained::Pool { index } => (
                self.pool.images[index].image,
                // Pool images carry no view (see PoolImage); a caller needing
                // one for GPU work creates it over `image` with the usage it
                // wants.
                vk::ImageView::null(),
                // copy_picture_to_image leaves the pool image in TRANSFER_DST.
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                0,
                self.pool.hand_out(index),
            ),
        };

        DecodedFrame {
            image,
            image_view,
            layout,
            array_layer,
            chroma_format: entry.chroma_format,
            bit_depth: entry.bit_depth,
            width: entry.width,
            height: entry.height,
            coded_width: entry.coded_width,
            coded_height: entry.coded_height,
            pts: entry.pts,
            display_order: entry.display_order,
            is_keyframe: entry.is_keyframe,
            sampleable: entry.sampleable,
            plane_views: entry.plane_views,
            generation: entry.generation,
            pin: Some(pin),
        }
    }

    /// Release everything this buffer holds.
    ///
    /// Drops the buffered pictures first, which releases the DPB slots they
    /// pinned, then frees the pool images. Frames the caller still holds point
    /// at these images, so this warns rather than silently leaving dangling
    /// handles behind.
    pub fn destroy(&mut self, common: &DecoderCommon) {
        self.buffered.clear();
        if self.pool.has_live_frames() {
            tracing::warn!(
                "decoder dropped while decoded frames are still alive; their images \
                 are now invalid. Drop every DecodedFrame before the Decoder."
            );
        }
        self.pool.destroy(common);
    }
}

/// A plain device-local image for the reorder pool: a copy target, readback
/// source and sampling source, with no view and no video profile (so it works
/// on every driver).
fn create_pool_image(
    context: &VideoContext,
    width: u32,
    height: u32,
    format: vk::Format,
    sharing_families: &[u32],
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let (families, sharing_mode) = crate::video::resolve_sharing_mode(sharing_families);
    let create_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        // SAMPLED unconditionally: a pool image carries no video profile, so it
        // is an ordinary image and every driver can sample one. Same for
        // MUTABLE_FORMAT, which needs no driver permission here. That keeps a
        // copied-out frame exactly as usable to a renderer as a pinned one, so
        // a consumer never has to handle "plane views work on some frames".
        .flags(vk::ImageCreateFlags::MUTABLE_FORMAT)
        .usage(
            vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED,
        )
        .sharing_mode(sharing_mode)
        .queue_family_indices(&families)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    // The image's own format has to be in the list too, not just the plane
    // formats: a consumer sampling through a ycbcr conversion views the whole
    // image, and a format list that omits its format makes that view invalid.
    let mut view_formats = vec![format];
    view_formats.extend(crate::video::plane_view_formats(format));
    let mut format_list = vk::ImageFormatListCreateInfo::default().view_formats(&view_formats);
    let create_info = if view_formats.len() < 2 {
        // Nothing to reinterpret this format as, so no list and no promise.
        create_info
    } else {
        create_info.push(&mut format_list)
    };
    let image = unsafe { context.device().create_image(&create_info, None) }
        .map_err(PixelForgeError::vk("vkCreateImage"))?;

    let reqs = unsafe { context.device().get_image_memory_requirements(image) };
    let memory_type_index = find_memory_type(
        context.memory_properties(),
        reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| {
        PixelForgeError::NoSuitableDevice("No device-local memory for reorder pool".to_string())
    })?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(memory_type_index);
    let memory = unsafe { context.device().allocate_memory(&alloc_info, None) }
        .map_err(PixelForgeError::vk("vkAllocateMemory"))?;
    unsafe { context.device().bind_image_memory(image, memory, 0) }
        .map_err(PixelForgeError::vk("vkBindImageMemory"))?;

    Ok((image, memory))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slot_pinned_twice_stays_pinned_until_both_release() {
        let pins = SlotPins::default();
        pins.pin(3);
        pins.pin(3);
        assert_eq!(pins.count(), 1, "one slot is held, however often");
        pins.release(3);
        assert!(pins.is_pinned(3));
        pins.release(3);
        assert!(!pins.is_pinned(3));
        assert!(!pins.any_pinned());
    }
}
