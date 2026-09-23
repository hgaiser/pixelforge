//! Asynchronous (push) encode pipelining, shared by all codecs.
//!
//! Each in-flight frame owns an [`EncodeSlot`] (its own input image, bitstream
//! buffer, encode command buffer, fence and query pool). [`EncodePipeline`]
//! rotates through the slots so that the CPU can record and submit frame N+1
//! while the GPU is still encoding frame N, instead of blocking on a fence after
//! every frame.
//!
//! Bitstream readback is performed off the calling thread: a single background
//! *completion thread* waits on each submission's fence, copies the bitstream
//! out, and resolves that frame's [`EncodeFuture`] the moment the GPU signals —
//! rather than deferring readback until the slot is reused. This delivers each
//! packet at roughly the GPU encode time instead of one or two `encode()` calls
//! later. Each `encode()` returns its own future (backed by a oneshot channel),
//! so packet delivery is paired with its submission by construction rather than
//! by a shared channel and a separate ordering convention.
//!
//! The DPB images and video session are shared across slots, so encode
//! submissions must still run in DPB order on the GPU. That ordering is enforced
//! with a single timeline semaphore (each submit waits on the previous submit's
//! value and signals its own). Only the calling thread ever touches the queue or
//! the timeline; the completion thread only waits on fences and reads bitstream
//! buffers, so the two never race on the same Vulkan object.
//!
//! Slot reuse is coordinated with a per-slot "busy" flag
//! ([`SlotSync`](crate::video::SlotSync)): a slot
//! is busy from submit until the completion thread has finished reading its
//! bitstream. The calling thread waits for the current slot to be free before
//! converting into / recording over it, which also covers the write-after-read
//! hazard on the shared input image.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::task::{Context, Poll};
use std::thread::JoinHandle;

use ash::vk::{self, Handle};
use futures_channel::oneshot;

use crate::encoder::resources::{
    ClearImageParams, allocate_command_buffers, clear_input_image, clear_rgb_input_image,
    create_bitstream_buffer, create_encode_feedback_query_pool, create_encode_timestamp_query_pool,
    create_fence, create_image, map_bitstream_buffer, query_timestamp_diff, submit_encode_only,
    wait_and_read_bitstream,
};
use crate::encoder::{BitDepth, EncodedPacket, FrameType, PixelFormat};
use crate::error::{PixelForgeError, Result};
use crate::sync::TimelinePoint;
use crate::video::{SlotSync, TimelineChain};
use crate::vulkan::VideoContext;

/// A handle to the packet a single `EncodePipeline::submit_current` will
/// eventually produce.
///
/// Returned by `Encoder::encode`, one per submitted frame. Awaiting it yields
/// that frame's [`EncodedPacket`] once the GPU finishes encoding and the
/// completion thread reads the bitstream back. Futures resolve in submission
/// order (one submission → one readback → one packet) as long as you await them
/// in the order `encode` returned them.
///
/// Dropping the future before it resolves is harmless: the completion thread
/// still reads the slot back (its send simply finds no receiver) and frees it.
pub struct EncodeFuture {
    rx: oneshot::Receiver<Result<EncodedPacket>>,
}

impl Future for EncodeFuture {
    type Output = Result<EncodedPacket>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.rx).poll(cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            // The sender was dropped without sending. The completion thread only
            // drops a sender after sending, so this indicates encoder teardown.
            Poll::Ready(Err(oneshot::Canceled)) => {
                Poll::Ready(Err(PixelForgeError::CommandBuffer(
                    "encode cancelled: encoder shut down before the frame was read back"
                        .to_string(),
                )))
            }
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
    pub is_key_frame: bool,
    pub pts: u64,
    pub dts: u64,
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

/// A submission handed from the calling thread to the completion thread.
struct WorkItem {
    slot_index: usize,
    fence: vk::Fence,
    query_pool: vk::QueryPool,
    bitstream_ptr: SendPtr,
    /// Capacity of the mapped bitstream buffer, for overflow detection.
    bitstream_buffer_size: usize,
    metadata: SlotPacketMetadata,
    /// Timestamping resources
    timestamp_period: f32,
    timestamp_query_pool: vk::QueryPool,
    submit_time: std::time::Instant,
    /// Resolves this frame's [`EncodeFuture`] once the bitstream is read back.
    result_tx: oneshot::Sender<Result<EncodedPacket>>,
}

/// All per-frame resources that must be private to a single in-flight encode.
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
    pub pixel_format: PixelFormat,
    pub bit_depth: BitDepth,
    pub bitstream_buffer_size: usize,
    /// Codec profile (with the codec-specific profile chained in) used for the
    /// input images, bitstream buffers and feedback query pools.
    pub profile_info: &'a vk::VideoProfileInfoKHR<'a>,
    pub command_pool: vk::CommandPool,
    /// Transfer command buffer/fence reused to zero-initialize each input image.
    pub upload_command_buffer: vk::CommandBuffer,
    pub upload_fence: vk::Fence,
}

/// Rotating set of [`EncodeSlot`]s plus the timeline semaphore that orders their
/// encode submissions and the completion thread that reads bitstreams back.
pub(crate) struct EncodePipeline {
    slots: Vec<EncodeSlot>,
    current_slot: usize,
    /// Orders encode submissions that share DPB state.
    timeline: TimelineChain,

    /// Per-slot busy flags shared with the completion thread.
    slot_sync: Arc<SlotSync>,
    /// Sends submitted work to the completion thread. Dropped on shutdown to end
    /// the thread.
    work_tx: Option<Sender<WorkItem>>,
    /// The completion thread handle, joined on shutdown.
    completion_thread: Option<JoinHandle<()>>,
}

impl EncodePipeline {
    /// Allocate the timeline semaphore, `ENCODE_PIPELINE_DEPTH` slots and spawn
    /// the bitstream-readback completion thread.
    pub(crate) fn new(config: &PipelineConfig) -> Result<Self> {
        let context = config.context;
        let device = context.device();

        let command_buffers =
            allocate_command_buffers(context, config.command_pool, ENCODE_PIPELINE_DEPTH as u32)?;

        // Timestamp queries are only legal on a queue family with non-zero
        // `timestampValidBits` (VUID-vkCmdWriteTimestamp-timestampValidBits-00829).
        // RADV's dedicated video encode queue reports 0, so recording
        // vkCmdWriteTimestamp there causes device loss. When unsupported we
        // leave the per-slot pools null and the recording/readback helpers treat
        // a null pool as "timestamps disabled".
        let timestamps_supported = context.encode_timestamps_supported();
        let timestamp_period = context.device_properties().limits.timestamp_period;
        if !timestamps_supported {
            tracing::info!(
                "Video encode queue family {:?} reports timestampValidBits=0; \
                 GPU encode timing stats disabled",
                context.video_encode_queue_family()
            );
        }

        let mut slots = Vec::with_capacity(ENCODE_PIPELINE_DEPTH);
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
            let clear = ClearImageParams {
                command_buffer: config.upload_command_buffer,
                fence: config.upload_fence,
                queue: context.transfer_queue(),
                image: input_image,
                width: config.aligned_width,
                height: config.aligned_height,
                pixel_format: config.pixel_format,
                bit_depth: config.bit_depth,
            };
            if config.rgb_input {
                clear_rgb_input_image(context, &clear)?;
            } else {
                clear_input_image(context, &clear)?;
            }

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
                timestamp_query_pool,
                pending_metadata: None,
            });
        }

        let slot_sync = Arc::new(SlotSync::new(slots.len()));
        let (work_tx, work_rx) = std::sync::mpsc::channel::<WorkItem>();

        // The completion thread only needs a handle to the device; the Vulkan
        // device handle is internally shared and safe to use from this thread
        // for fence waits, query reads and host-coherent buffer reads.
        let thread_device = device.clone();
        let thread_sync = slot_sync.clone();
        let completion_thread = std::thread::Builder::new()
            .name("pixelforge-encode-readback".to_string())
            .spawn(move || {
                run_completion_thread(thread_device, work_rx, thread_sync);
            })
            .map_err(|e| PixelForgeError::CommandBuffer(format!("spawn readback thread: {e}")))?;

        Ok(Self {
            slots,
            current_slot: 0,
            timeline: TimelineChain::new(context)?,
            slot_sync,
            work_tx: Some(work_tx),
            completion_thread: Some(completion_thread),
        })
    }

    /// The slot the next frame will be encoded into.
    pub(crate) fn current(&self) -> &EncodeSlot {
        &self.slots[self.current_slot]
    }

    pub(crate) fn current_mut(&mut self) -> &mut EncodeSlot {
        &mut self.slots[self.current_slot]
    }

    /// Return the current slot's input image, first waiting until the slot is
    /// free so it is safe to use as a convert/upload target (write-after-read on
    /// the shared input image).
    pub(crate) fn input_image(&self) -> vk::Image {
        self.slot_sync.wait_free(self.current_slot);
        self.slots[self.current_slot].input_image
    }

    /// Wait until the current slot is free to record over and submit.
    pub(crate) fn wait_current_free(&self) {
        self.slot_sync.wait_free(self.current_slot);
    }

    /// Wait until every in-flight submission has been read back. Used before
    /// mutating shared session state and at teardown.
    pub(crate) fn wait_all_free(&self) {
        self.slot_sync.wait_all_free();
    }

    /// Record the metadata for the packet that the current slot will produce.
    /// Must be called before [`EncodePipeline::submit_current`].
    pub(crate) fn set_pending_metadata(&mut self, metadata: SlotPacketMetadata) {
        self.slots[self.current_slot].pending_metadata = Some(metadata);
    }

    /// Submit the current slot's recorded command buffer without waiting, and
    /// hand the slot to the completion thread for bitstream readback.
    ///
    /// Chains onto the timeline semaphore so the GPU keeps encodes in DPB order,
    /// and marks the slot busy until the completion thread reads it back. Returns
    /// the [`EncodeFuture`] that resolves with this frame's packet.
    pub(crate) fn submit_current(
        &mut self,
        device: &ash::Device,
        sync2: &ash::khr::synchronization2::Device,
        encode_queue: vk::Queue,
        external_waits: &[TimelinePoint],
    ) -> Result<EncodeFuture> {
        let wait = self.timeline.wait();
        let signal = self.timeline.pending_signal();
        let slot_index = self.current_slot;

        // Capture the Copy handles + metadata, releasing the slot borrow before
        // touching the cross-thread channels.
        let (
            command_buffer,
            fence,
            query_pool,
            bitstream_ptr,
            bitstream_buffer_size,
            timestamp_period,
            timestamp_query_pool,
            metadata,
        ) = {
            let slot = &mut self.slots[slot_index];
            let metadata = slot.pending_metadata.take().ok_or_else(|| {
                PixelForgeError::CommandBuffer(
                    "submit_current called without pending packet metadata".to_string(),
                )
            })?;
            (
                slot.encode_command_buffer,
                slot.encode_fence,
                slot.query_pool,
                slot.bitstream_buffer_ptr as *const u8,
                slot.bitstream_buffer_size,
                slot.timestamp_period,
                slot.timestamp_query_pool,
                metadata,
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
            .chain(external_waits.iter().map(TimelinePoint::wait_info))
            .collect();
        unsafe {
            submit_encode_only(
                device,
                sync2,
                command_buffer,
                fence,
                encode_queue,
                &waits,
                Some(signal),
            )?;
        }
        let submit_time = std::time::Instant::now();

        self.timeline.commit();

        // Mark busy *before* handing the work off, so the completion thread can
        // never clear the flag before it is set.
        self.slot_sync.set_busy(slot_index);

        let (result_tx, result_rx) = oneshot::channel::<Result<EncodedPacket>>();
        let work = WorkItem {
            slot_index,
            fence,
            query_pool,
            bitstream_ptr: SendPtr(bitstream_ptr),
            bitstream_buffer_size,
            timestamp_period,
            timestamp_query_pool,
            submit_time,
            metadata,
            result_tx,
        };
        if let Some(tx) = &self.work_tx {
            // The receiver only disconnects during shutdown, after the queue is
            // idle; a failed send there is benign.
            let _ = tx.send(work);
        }

        Ok(EncodeFuture { rx: result_rx })
    }

    /// Advance to the next slot after a frame has been submitted.
    pub(crate) fn advance(&mut self) {
        self.current_slot = (self.current_slot + 1) % self.slots.len();
    }

    /// Barrier: wait for every in-flight frame to be read back.
    ///
    /// The completion thread resolves each frame's [`EncodeFuture`] before it
    /// marks the slot free, so once every slot is free every outstanding future
    /// has already been resolved. Callers await the futures `encode` returned to
    /// obtain the packets themselves.
    pub(crate) fn flush(&mut self) {
        self.slot_sync.wait_all_free();
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

    /// Destroy all slot resources and the timeline semaphore.
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
        unsafe {
            self.timeline.destroy(device);
        }
    }
}

/// Completion-thread body: wait on each submission's fence, copy its bitstream
/// out, resolve the frame's future, then mark the slot free.
fn run_completion_thread(
    device: ash::Device,
    work_rx: Receiver<WorkItem>,
    slot_sync: Arc<SlotSync>,
) {
    for work in work_rx {
        // Start the packet from any codec header, then read the encoded bitstream
        // straight onto it — a single copy out of the mapped buffer.
        let mut data = work.metadata.header.unwrap_or_default();
        let result = unsafe {
            wait_and_read_bitstream(
                &device,
                work.fence,
                work.query_pool,
                work.bitstream_ptr.0,
                work.bitstream_buffer_size,
                &mut data,
            )
        };
        let bitstream_ready_time = std::time::Instant::now();

        let mut stats: Option<super::EncodedPacketStats> = None;
        if let Some(gpu_encode_ns) = unsafe {
            query_timestamp_diff(
                &device,
                work.timestamp_query_pool,
                work.metadata.timestamps,
                work.timestamp_period,
            )
        } {
            stats = Some(super::EncodedPacketStats {
                gpu_time_ns: gpu_encode_ns,
                frame_latency_ns: work.metadata.now.elapsed().as_nanos() as u64,
                wall_latency_ns: bitstream_ready_time
                    .duration_since(work.submit_time)
                    .as_nanos() as u64,
            });
        }

        let packet = result.map(|()| EncodedPacket {
            data,
            frame_type: work.metadata.frame_type,
            is_key_frame: work.metadata.is_key_frame,
            pts: work.metadata.pts,
            dts: work.metadata.dts,
            stats,
        });

        // Resolve the future *before* freeing the slot. This ordering means that
        // once all slots are observed free, every packet has already been
        // delivered, which `flush` relies on for completeness. A dropped
        // receiver (future cancelled) makes the send a no-op, which is fine.
        let _ = work.result_tx.send(packet);
        slot_sync.set_free(work.slot_index);
    }
}
