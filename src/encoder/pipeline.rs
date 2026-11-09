//! Asynchronous (push) encode pipelining, shared by all codecs.
//!
//! Each frame owns an [`EncodeSlot`] (its own input image, bitstream buffer,
//! encode command buffer, fence and query pool) from the moment it is submitted
//! until its bitstream is read back. [`EncodePipeline`] hands the slots out
//! oldest-free first, so that the CPU can record and submit frame N+1 while the
//! GPU is still encoding frame N, instead of blocking on a fence after every
//! frame. A frame that is encoded later than it was submitted (a B-frame) keeps
//! its slot, and with it its pixels, until then.
//!
//! Bitstream readback is performed off the calling thread: a single background
//! *completion thread* waits on each submission's fence, copies the bitstream
//! out, and resolves that frame's [`EncodeFuture`] the moment the GPU signals —
//! rather than deferring readback until the slot is reused. This delivers each
//! packet at roughly the GPU encode time instead of one or two `encode()` calls
//! later. Each `submit()` returns its own future (backed by a oneshot channel).
//! The futures resolve in the order they were made: the k-th future gets the
//! k-th packet, whichever frame that packet holds. The senders wait in a queue
//! until an encode takes the oldest one.
//!
//! The DPB images and video session are shared across slots, so encode
//! submissions must still run in DPB order on the GPU. That ordering is enforced
//! with a single timeline semaphore (each submit waits on the previous submit's
//! value and signals its own). Only the calling thread ever touches the queue or
//! the timeline; the completion thread only waits on fences and reads bitstream
//! buffers, so the two never race on the same Vulkan object.
//!
//! Slot reuse is coordinated with a queue of free slots (`FreeSlots`): a
//! slot leaves it when a frame takes it and comes back once the completion
//! thread has finished reading its bitstream. The calling thread waits for a
//! free slot before converting into / recording over it, which also covers the
//! write-after-read hazard on the slot's input image.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::task::{Context, Poll};
use std::thread::JoinHandle;

use ash::vk::{self, Handle};
use futures_channel::oneshot;

use crate::encoder::resources::{
    ClearImageParams, allocate_command_buffers, clear_input_image, create_bitstream_buffer,
    create_encode_feedback_query_pool, create_encode_timestamp_query_pool, create_fence,
    create_image, map_bitstream_buffer, query_timestamp_diff, submit_encode_only,
    wait_and_read_bitstream,
};
use crate::encoder::{BitDepth, ChromaFormat, EncodedPacket, FrameType};
use crate::error::{PixelForgeError, Result};
use crate::sync::TimelinePoint;
use crate::video::TimelineChain;
use crate::vulkan::VideoContext;

/// A handle to one packet of the stream.
///
/// Returned by `Encoder::submit`, one per submitted frame. Awaiting it yields
/// an [`EncodedPacket`] once the GPU finishes encoding and the completion
/// thread reads the bitstream back. The futures resolve in the order `submit`
/// returned them, and the k-th one yields the k-th packet of the stream.
///
/// Dropping the future before it resolves is harmless: the completion thread
/// still reads the slot back (its send simply finds no receiver) and frees it.
pub struct EncodeFuture {
    rx: oneshot::Receiver<Result<EncodedPacket>>,
    released: TimelinePoint,
    /// Added to the packet's timestamps, so they carry on across an encoder
    /// that was made again for a colour change.
    pub(crate) pts_offset: u64,
    #[cfg(feature = "dmabuf")]
    pub(crate) release_sync_file: Option<std::os::fd::OwnedFd>,
}

impl EncodeFuture {
    /// The point after which this frame's source may be written again: the
    /// copy into the encoder's input image, the colour conversion, or the
    /// encode itself for a source read directly.
    pub(crate) fn source_released(&self) -> TimelinePoint {
        self.released
    }

    /// The sync_file asked for with
    /// [`Frame::signal_sync_file`](crate::Frame::signal_sync_file): it
    /// signals once the frame is done with its source. `None` if it was not asked for,
    /// was already taken, or had already signalled when it was made, which
    /// the driver may report that way.
    #[cfg(feature = "dmabuf")]
    pub fn take_release_sync_file(&mut self) -> Option<std::os::fd::OwnedFd> {
        self.release_sync_file.take()
    }

    pub(crate) fn set_released(&mut self, point: TimelinePoint) {
        self.released = point;
    }
}

impl Future for EncodeFuture {
    type Output = Result<EncodedPacket>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let offset = self.pts_offset;
        match Pin::new(&mut self.rx).poll(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(result.map(|mut packet| {
                packet.pts += offset;
                packet.dts += offset as i64;
                packet
            })),
            // The sender was dropped without sending. The completion thread only
            // drops a sender after sending, so the encoder was torn down, or an
            // encode failed before this frame's.
            Poll::Ready(Err(oneshot::Canceled)) => Poll::Ready(Err(PixelForgeError::Cancelled)),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Number of encode submissions allowed to be in flight at once.
///
/// See the module docs and `docs` discussion for why 2 is the sweet spot: it
/// fully overlaps capture/convert/upload of the next frame with the GPU encode
/// of the current one, while keeping the GPU-serialized DPB chain from growing.
pub(crate) const ENCODE_PIPELINE_DEPTH: usize = 2;

/// Per-frame packet info captured at submit time and attached to the bitstream
/// when the completion thread reads the slot back.
pub(crate) struct SlotPacketMetadata {
    pub frame_type: FrameType,
    pub pts: u64,
    pub dts: i64,
    /// Codec header (SPS/PPS, VPS/SPS/PPS, or AV1 sequence header) to prepend.
    /// `Some` only for frames that carry one (e.g. the IDR/key frame).
    pub header: Option<Vec<u8>>,
    pub timestamps: [u64; 2],
    pub now: std::time::Instant,
}

/// A raw pointer wrapper asserting `Send` so the persistently-mapped bitstream
/// pointer can be handed to the completion thread. The memory is only read by
/// that thread, and only after the encode fence has signalled.
struct SendPtr(*const u8);
// SAFETY: the pointed-to bitstream buffer is host-coherent, persistently mapped
// for the lifetime of the slot, and read exclusively by the completion thread
// after the encode fence signals. The calling thread does not touch the buffer
// until the slot is marked free again (after this read completes).
unsafe impl Send for SendPtr {}

/// An encode handed from the calling thread to the completion thread, to
/// read its bitstream back.
struct Readback {
    slot_index: usize,
    fence: vk::Fence,
    query_pool: vk::QueryPool,
    bitstream_ptr: SendPtr,
    /// Capacity of the mapped bitstream buffer, for overflow detection.
    bitstream_buffer_size: usize,
    metadata: SlotPacketMetadata,
    /// Timestamping resources
    timestamp_period: f32,
    timestamp_valid_bits: u32,
    timestamp_query_pool: vk::QueryPool,
    submit_time: std::time::Instant,
}

/// Where an encode's bitstream goes.
pub(crate) enum Delivery {
    /// Into a packet of its own, which resolves the oldest waiting future.
    Packet,
    /// In front of the next packet, in the same temporal unit: an AV1 frame
    /// that is not shown. It resolves no future.
    #[allow(dead_code)]
    Carry,
}

/// What the completion thread does with an encode's bitstream.
enum Deliver {
    /// Resolves a frame's [`EncodeFuture`] once the bitstream is read back.
    Packet(oneshot::Sender<Result<EncodedPacket>>),
    Carry,
}

/// Work for the completion thread, handled in the order it was sent.
enum WorkItem {
    /// An encode on the GPU.
    Encoded(Readback, Deliver),
    /// A packet made on the CPU, with no encode and no slot: an AV1
    /// `show_existing_frame` header.
    Synthetic {
        packet: EncodedPacket,
        result_tx: oneshot::Sender<Result<EncodedPacket>>,
    },
}

/// The slots no frame owns, oldest first.
///
/// The calling thread takes from the front and the completion thread puts
/// back at the end, so the front does not change under the calling thread
/// once there is one.
struct FreeSlots {
    queue: std::sync::Mutex<VecDeque<usize>>,
    cv: std::sync::Condvar,
    slot_count: usize,
}

impl FreeSlots {
    fn new(slot_count: usize) -> Self {
        Self {
            queue: std::sync::Mutex::new((0..slot_count).collect()),
            cv: std::sync::Condvar::new(),
            slot_count,
        }
    }

    /// Wait for a free slot and return the oldest, without taking it.
    fn wait_front(&self) -> usize {
        let queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        let queue = self
            .cv
            .wait_while(queue, |q| q.is_empty())
            .unwrap_or_else(|e| e.into_inner());
        queue[0]
    }

    /// Take the oldest free slot, waiting for one.
    fn take_front(&self) -> usize {
        let queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        let mut queue = self
            .cv
            .wait_while(queue, |q| q.is_empty())
            .unwrap_or_else(|e| e.into_inner());
        queue.pop_front().expect("waited for a free slot")
    }

    fn put_back(&self, slot: usize) {
        let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        queue.push_back(slot);
        self.cv.notify_all();
    }

    /// Wait until every slot is free. Only returns if no frame holds a slot
    /// without an encode in flight.
    fn wait_all(&self) {
        let queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        let _queue = self
            .cv
            .wait_while(queue, |q| q.len() < self.slot_count)
            .unwrap_or_else(|e| e.into_inner());
    }
}

/// All per-frame resources that must be private to a single frame.
pub(crate) struct EncodeSlot {
    pub input_image: vk::Image,
    pub input_image_memory: vk::DeviceMemory,
    pub input_image_view: vk::ImageView,
    /// Tracked layout of `input_image` (to avoid UB when transitioning).
    pub input_image_layout: vk::ImageLayout,

    pub bitstream_buffer: vk::Buffer,
    pub bitstream_buffer_memory: vk::DeviceMemory,
    pub bitstream_buffer_size: usize,
    /// Persistently mapped pointer to the bitstream buffer.
    pub bitstream_buffer_ptr: *mut u8,

    pub encode_command_buffer: vk::CommandBuffer,
    pub encode_fence: vk::Fence,
    pub query_pool: vk::QueryPool,

    pub timestamp_period: f32,
    pub timestamp_valid_bits: u32,
    pub timestamp_query_pool: vk::QueryPool,

    /// Packet metadata recorded before submit, moved to the completion thread
    /// with the work item.
    pub pending_metadata: Option<SlotPacketMetadata>,
}

/// Configuration for building an [`EncodePipeline`].
pub(crate) struct PipelineConfig<'a> {
    pub context: &'a VideoContext,
    pub aligned_width: u32,
    pub aligned_height: u32,
    pub picture_format: vk::Format,
    /// Whether the input image is RGB, for the encoder to convert.
    pub rgb_input: bool,
    pub chroma_format: ChromaFormat,
    pub bit_depth: BitDepth,
    pub bitstream_buffer_size: usize,
    /// Codec profile (with the codec-specific profile chained in) used for the
    /// input images, bitstream buffers and feedback query pools.
    pub profile_info: &'a vk::VideoProfileInfoKHR<'a>,
    pub command_pool: vk::CommandPool,
    /// How many frames can wait for their encode, on top of
    /// [`ENCODE_PIPELINE_DEPTH`]: the B-frames between two reference frames.
    pub held_frames: usize,
    /// Transfer command buffer/fence reused to zero-initialize each input image.
    pub upload_command_buffer: vk::CommandBuffer,
    pub upload_fence: vk::Fence,
}

/// The [`EncodeSlot`]s, plus the timeline semaphore that orders their encode
/// submissions and the completion thread that reads bitstreams back.
pub(crate) struct EncodePipeline {
    slots: Vec<EncodeSlot>,
    /// The slot of the frame being encoded.
    current_slot: usize,
    /// Orders encode submissions that share DPB state.
    timeline: TimelineChain,

    /// The free slots, shared with the completion thread.
    free_slots: Arc<FreeSlots>,
    /// The senders of the futures no packet was assigned to yet, oldest first.
    senders: VecDeque<oneshot::Sender<Result<EncodedPacket>>>,
    /// Sends submitted work to the completion thread. Dropped on shutdown to end
    /// the thread.
    work_tx: Option<Sender<WorkItem>>,
    /// The completion thread handle, joined on shutdown.
    completion_thread: Option<JoinHandle<()>>,
}

impl EncodePipeline {
    /// Allocate the timeline semaphore and the slots, and spawn the
    /// bitstream-readback completion thread.
    pub(crate) fn new(config: &PipelineConfig) -> Result<Self> {
        let context = config.context;
        let device = context.device();

        let slot_count = ENCODE_PIPELINE_DEPTH + config.held_frames;
        let command_buffers =
            allocate_command_buffers(context, config.command_pool, slot_count as u32)?;

        // Timestamp queries are only legal on a queue family with non-zero
        // `timestampValidBits` (VUID-vkCmdWriteTimestamp-timestampValidBits-00829).
        // RADV's dedicated video encode queue reports 0, so recording
        // vkCmdWriteTimestamp there causes device loss. When unsupported we
        // leave the per-slot pools null and the recording/readback helpers treat
        // a null pool as "timestamps disabled".
        let timestamps_supported = context.encode_timestamps_supported();
        let timestamp_period = context.device_properties().limits.timestamp_period;
        let timestamp_valid_bits = context.encode_timestamp_valid_bits();
        if !timestamps_supported {
            tracing::info!(
                "Video encode queue family {:?} reports timestampValidBits=0; \
                 GPU encode timing stats disabled",
                context.video_encode_queue_family()
            );
        }

        let mut slots = Vec::with_capacity(slot_count);
        for &encode_command_buffer in &command_buffers {
            let (input_image, input_image_memory, input_image_view) = create_image(
                context,
                config.aligned_width,
                config.aligned_height,
                config.picture_format,
                false,
                config.profile_info,
            )?;

            let (bitstream_buffer, bitstream_buffer_memory) = create_bitstream_buffer(
                context,
                config.bitstream_buffer_size,
                vk::BufferUsageFlags::VIDEO_ENCODE_DST_KHR,
                config.profile_info,
            )?;
            let bitstream_buffer_ptr = map_bitstream_buffer(
                context,
                bitstream_buffer_memory,
                config.bitstream_buffer_size,
            )?;

            // Zero the padding between the user dimensions and the aligned coded
            // extent so the first frame has no undefined samples.
            clear_input_image(
                context,
                &ClearImageParams {
                    command_buffer: config.upload_command_buffer,
                    fence: config.upload_fence,
                    queue: context.transfer_queue(),
                    image: input_image,
                    width: config.aligned_width,
                    height: config.aligned_height,
                    chroma_format: config.chroma_format,
                    bit_depth: config.bit_depth,
                    rgb: config.rgb_input,
                },
            )?;

            // Created signaled so it is safe to wait on before the first encode;
            // `submit_encode_only` resets it before each submit.
            let encode_fence = create_fence(context, true)?;

            let mut profile = *config.profile_info;
            let query_pool = create_encode_feedback_query_pool(context, &mut profile)?;

            let timestamp_query_pool = if timestamps_supported {
                create_encode_timestamp_query_pool(context)?
            } else {
                vk::QueryPool::null()
            };

            slots.push(EncodeSlot {
                input_image,
                input_image_memory,
                input_image_view,
                input_image_layout: vk::ImageLayout::VIDEO_ENCODE_SRC_KHR,
                bitstream_buffer,
                bitstream_buffer_memory,
                bitstream_buffer_size: config.bitstream_buffer_size,
                bitstream_buffer_ptr,
                encode_command_buffer,
                encode_fence,
                query_pool,
                timestamp_period,
                timestamp_valid_bits,
                timestamp_query_pool,
                pending_metadata: None,
            });
        }

        let free_slots = Arc::new(FreeSlots::new(slots.len()));
        let (work_tx, work_rx) = std::sync::mpsc::channel::<WorkItem>();

        // The completion thread only needs a handle to the device; the Vulkan
        // device handle is internally shared and safe to use from this thread
        // for fence waits, query reads and host-coherent buffer reads.
        let thread_device = device.clone();
        let thread_free = free_slots.clone();
        let completion_thread = std::thread::Builder::new()
            .name("pixelforge-encode-readback".to_string())
            .spawn(move || {
                run_completion_thread(thread_device, work_rx, thread_free);
            })
            .map_err(|source| PixelForgeError::Io {
                what: "spawning the readback thread",
                source,
            })?;

        Ok(Self {
            slots,
            current_slot: 0,
            timeline: TimelineChain::new(context)?,
            free_slots,
            senders: VecDeque::new(),
            work_tx: Some(work_tx),
            completion_thread: Some(completion_thread),
        })
    }

    /// The slot of the frame being encoded, see [`Self::begin_encode`].
    pub(crate) fn current(&self) -> &EncodeSlot {
        &self.slots[self.current_slot]
    }

    /// The slot at `index`.
    pub(crate) fn slot(&self, index: usize) -> &EncodeSlot {
        &self.slots[index]
    }

    pub(crate) fn slot_mut(&mut self, index: usize) -> &mut EncodeSlot {
        &mut self.slots[index]
    }

    pub(crate) fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// The slot the next submitted frame takes, first waiting until there is
    /// a free one. A free slot's last encode was read back, so its input image
    /// is safe to use as a convert/upload target and its command buffers and
    /// semaphores are free.
    pub(crate) fn next_index(&self) -> usize {
        self.free_slots.wait_front()
    }

    /// The input image of the slot the next submitted frame takes, see
    /// [`Self::next_index`].
    pub(crate) fn input_image(&self) -> vk::Image {
        self.slots[self.next_index()].input_image
    }

    /// Give the slot [`Self::next_index`] names to a frame. The frame owns it
    /// until its encode was read back, or until [`Self::release`].
    pub(crate) fn take_slot(&mut self) -> usize {
        self.free_slots.take_front()
    }

    /// Give back the slot of a frame that will not be encoded.
    pub(crate) fn release(&mut self, slot: usize) {
        self.slots[slot].pending_metadata = None;
        self.free_slots.put_back(slot);
    }

    /// Make the future of the next packet. Its sender waits until an encode
    /// takes it.
    pub(crate) fn new_future(&mut self) -> EncodeFuture {
        let (result_tx, result_rx) = oneshot::channel::<Result<EncodedPacket>>();
        self.senders.push_back(result_tx);
        EncodeFuture {
            rx: result_rx,
            // A wait for 0 is satisfied at once: nothing to release yet.
            released: TimelinePoint::new(self.timeline.semaphore(), 0),
            pts_offset: 0,
            #[cfg(feature = "dmabuf")]
            release_sync_file: None,
        }
    }

    /// Drop the newest future's sender: its frame was not accepted.
    pub(crate) fn drop_newest_future(&mut self) {
        self.senders.pop_back();
    }

    /// Wait until every in-flight submission has been read back. Used before
    /// mutating shared session state and at teardown. No frame may wait for
    /// its encode.
    pub(crate) fn wait_all_free(&self) {
        self.free_slots.wait_all();
    }

    /// Encode into `slot` from here on: the slot of the frame whose turn it
    /// is.
    pub(crate) fn begin_encode(&mut self, slot: usize) {
        self.current_slot = slot;
    }

    /// Record the metadata for the packet that the current slot will produce.
    /// Must be called before [`EncodePipeline::submit_current`].
    pub(crate) fn set_pending_metadata(&mut self, metadata: SlotPacketMetadata) {
        self.slots[self.current_slot].pending_metadata = Some(metadata);
    }

    /// Submit the current slot's recorded command buffer without waiting, and
    /// hand the slot to the completion thread for bitstream readback.
    ///
    /// Chains onto the timeline semaphore so the GPU keeps encodes in DPB
    /// order. Returns the point the encode signals.
    pub(crate) fn submit_current(
        &mut self,
        device: &ash::Device,
        sync2: &ash::khr::synchronization2::Device,
        encode_queue: vk::Queue,
        external_waits: &[vk::SemaphoreSubmitInfo<'static>],
        extra_signals: &[vk::SemaphoreSubmitInfo<'static>],
        delivery: Delivery,
    ) -> Result<TimelinePoint> {
        let wait = self.timeline.wait();
        let signal = self.timeline.pending_signal();
        let slot_index = self.current_slot;
        if matches!(delivery, Delivery::Packet) && self.senders.is_empty() {
            return Err(PixelForgeError::InvalidInput(
                "submit_current called with no future waiting for the packet".to_string(),
            ));
        }

        // Capture the Copy handles + metadata, releasing the slot borrow before
        // touching the cross-thread channels.
        let (command_buffer, readback) = {
            let slot = &mut self.slots[slot_index];
            let metadata = slot.pending_metadata.take().ok_or_else(|| {
                PixelForgeError::InvalidInput(
                    "submit_current called without pending packet metadata".to_string(),
                )
            })?;
            (
                slot.encode_command_buffer,
                Readback {
                    slot_index,
                    fence: slot.encode_fence,
                    query_pool: slot.query_pool,
                    bitstream_ptr: SendPtr(slot.bitstream_buffer_ptr as *const u8),
                    bitstream_buffer_size: slot.bitstream_buffer_size,
                    timestamp_period: slot.timestamp_period,
                    timestamp_valid_bits: slot.timestamp_valid_bits,
                    timestamp_query_pool: slot.timestamp_query_pool,
                    // Before the submit, not after: the driver can block in
                    // the submit while the GPU already runs the work.
                    submit_time: std::time::Instant::now(),
                    metadata,
                },
            )
        };

        // The previous encode, which shares the DPB, then whatever the caller
        // asked this frame to wait for.
        let waits: Vec<vk::SemaphoreSubmitInfo> = wait
            .into_iter()
            .map(|(semaphore, value)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(semaphore)
                    .value(value)
                    .stage_mask(vk::PipelineStageFlags2::VIDEO_ENCODE_KHR)
            })
            .chain(external_waits.iter().copied())
            .collect();
        // ALL_COMMANDS rather than the encode stage: the timeline value is
        // also handed out as the point after which a directly read source is
        // free, and its release barrier comes after the encode.
        let signals: Vec<vk::SemaphoreSubmitInfo> = std::iter::once(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(signal.0)
                .value(signal.1)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        )
        .chain(extra_signals.iter().copied())
        .collect();
        unsafe {
            submit_encode_only(
                device,
                sync2,
                command_buffer,
                readback.fence,
                encode_queue,
                &waits,
                &signals,
            )?;
        }

        self.timeline.commit();

        let deliver = match delivery {
            Delivery::Packet => Deliver::Packet(
                self.senders
                    .pop_front()
                    .expect("checked before the submission"),
            ),
            Delivery::Carry => Deliver::Carry,
        };
        if let Some(tx) = &self.work_tx {
            // The receiver only disconnects during shutdown, after the queue is
            // idle; a failed send there is benign.
            let _ = tx.send(WorkItem::Encoded(readback, deliver));
        }

        Ok(TimelinePoint::new(signal.0, signal.1))
    }

    /// Resolve the oldest waiting future with a packet made on the CPU, after
    /// every packet submitted before it.
    #[allow(dead_code)]
    pub(crate) fn submit_synthetic(&mut self, packet: EncodedPacket) -> Result<()> {
        let result_tx = self.senders.pop_front().ok_or_else(|| {
            PixelForgeError::InvalidInput(
                "submit_synthetic called with no future waiting for the packet".to_string(),
            )
        })?;
        if let Some(tx) = &self.work_tx {
            let _ = tx.send(WorkItem::Synthetic { packet, result_tx });
        }
        Ok(())
    }

    /// The timeline the encodes signal. Retired to the context rather than
    /// destroyed, since a caller may still wait on a point it handed out.
    pub(crate) fn timeline_semaphore(&self) -> vk::Semaphore {
        self.timeline.semaphore()
    }

    /// Barrier: wait for every in-flight frame to be read back.
    ///
    /// The completion thread resolves each frame's [`EncodeFuture`] before it
    /// frees the slot, so once every slot is free every outstanding future
    /// has already been resolved. Callers await the futures `submit` returned
    /// to obtain the packets themselves.
    pub(crate) fn flush(&mut self) {
        self.free_slots.wait_all();
    }

    /// Stop the completion thread and wait for it to finish any in-flight
    /// readback. Safe to call more than once.
    fn shutdown(&mut self) {
        // Dropping the sender ends the thread's `for work in rx` loop once it
        // has drained outstanding items.
        self.work_tx.take();
        if let Some(handle) = self.completion_thread.take() {
            let _ = handle.join();
        }
    }

    /// Destroy all slot resources. The timeline semaphore is left for the
    /// caller to retire, see [`Self::timeline_semaphore`].
    ///
    /// # Safety
    ///
    /// All queues that may reference these resources must be idle.
    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        // Join the readback thread before freeing the fences/buffers it reads.
        self.shutdown();

        for slot in &mut self.slots {
            if !slot.bitstream_buffer_ptr.is_null() {
                unsafe {
                    device.unmap_memory(slot.bitstream_buffer_memory);
                }
                slot.bitstream_buffer_ptr = std::ptr::null_mut();
            }
            if !slot.timestamp_query_pool.is_null() {
                unsafe {
                    device.destroy_query_pool(slot.timestamp_query_pool, None);
                }
            }
            unsafe {
                device.destroy_query_pool(slot.query_pool, None);
                device.destroy_fence(slot.encode_fence, None);
                device.destroy_buffer(slot.bitstream_buffer, None);
                device.free_memory(slot.bitstream_buffer_memory, None);
                device.destroy_image_view(slot.input_image_view, None);
                device.destroy_image(slot.input_image, None);
                device.free_memory(slot.input_image_memory, None);
            }
        }
    }
}

/// Completion-thread body: wait on each submission's fence, copy its bitstream
/// out, resolve the oldest waiting future, then free the slot.
fn run_completion_thread(
    device: ash::Device,
    work_rx: Receiver<WorkItem>,
    free_slots: Arc<FreeSlots>,
) {
    // The bitstream of the encodes that go in front of the next packet.
    let mut carried: Option<Result<Vec<u8>>> = None;
    for work in work_rx {
        let (work, deliver) = match work {
            WorkItem::Encoded(work, deliver) => (work, deliver),
            WorkItem::Synthetic { packet, result_tx } => {
                let _ = result_tx.send(Ok(packet));
                continue;
            }
        };
        // Start the packet from any codec header, then read the encoded bitstream
        // straight onto it — a single copy out of the mapped buffer.
        let mut data = work.metadata.header.unwrap_or_default();
        let mut result = Ok(());
        if matches!(deliver, Deliver::Packet(_)) {
            match carried.take() {
                Some(Ok(bytes)) => data.extend_from_slice(&bytes),
                Some(Err(e)) => result = Err(e),
                None => {}
            }
        }
        let read = unsafe {
            wait_and_read_bitstream(
                &device,
                work.fence,
                work.query_pool,
                work.bitstream_ptr.0,
                work.bitstream_buffer_size,
                &mut data,
            )
        };
        let result = result.and(read);
        let bitstream_ready_time = std::time::Instant::now();

        let result_tx = match deliver {
            Deliver::Packet(result_tx) => result_tx,
            Deliver::Carry => {
                carried = Some(match (carried.take(), result) {
                    (Some(Ok(mut bytes)), Ok(())) => {
                        bytes.extend_from_slice(&data);
                        Ok(bytes)
                    }
                    (Some(Err(e)), _) | (_, Err(e)) => Err(e),
                    (None, Ok(())) => Ok(data),
                });
                free_slots.put_back(work.slot_index);
                continue;
            }
        };

        let stats = super::EncodedPacketStats {
            gpu_time_ns: unsafe {
                query_timestamp_diff(
                    &device,
                    work.timestamp_query_pool,
                    work.metadata.timestamps,
                    work.timestamp_period,
                    work.timestamp_valid_bits,
                )
            },
            frame_latency_ns: work.metadata.now.elapsed().as_nanos() as u64,
            wall_latency_ns: bitstream_ready_time
                .duration_since(work.submit_time)
                .as_nanos() as u64,
        };

        let packet = result.map(|()| EncodedPacket {
            data,
            frame_type: work.metadata.frame_type,
            pts: work.metadata.pts,
            dts: work.metadata.dts,
            stats,
        });

        // Resolve the future *before* freeing the slot. This ordering means that
        // once all slots are observed free, every packet has already been
        // delivered, which `flush` relies on for completeness. A dropped
        // receiver (future cancelled) makes the send a no-op, which is fine.
        let _ = result_tx.send(packet);
        free_slots.put_back(work.slot_index);
    }
}
