//! DMA-BUF input: the encoder reads a buffer straight into the encode, copies
//! it, or converts it with the shader, and orders all of it with sync_files.
//!
//! The tests play the producer themselves: they import the same DMA-BUF into
//! an image of their own and write frames into it, the way a compositor would
//! from another process. Each path is checked against encoding the same
//! frames from an ordinary image, which has to give the same stream.
#![cfg(feature = "dmabuf")]

#[allow(dead_code)]
mod common;
use common::source::{host_buffer, one_shot};
use common::{Readback, decode_stream};

use ash::vk;
use ash::vk::TaggedStructure;
use pixelforge::decoder::{DecodeConfig, Decoder};
use pixelforge::encoder::external::drm_format;
use pixelforge::{
    Codec, ColorSpec, Conversion, ConversionPreference, DmaBuf, DmaBufImage, DmaBufPlane,
    EncodeConfig, Encoder, Frame, Handoff, RateControl, RgbFormat, RgbSource, Source, SourceImage,
    VideoContext, VideoContextBuilder, YuvLayout, YuvUploader,
};
use std::os::fd::{FromRawFd, OwnedFd};

const FRAMES: u32 = 6;

fn context() -> Result<VideoContext, Box<dyn std::error::Error>> {
    common::init_logging();
    Ok(VideoContextBuilder::new()
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .build()?)
}

fn config(codec: Codec, width: u32, height: u32) -> EncodeConfig {
    match codec {
        Codec::H264 => EncodeConfig::h264(width, height),
        Codec::H265 => EncodeConfig::h265(width, height),
        Codec::AV1 => EncodeConfig::av1(width, height),
    }
    .with_rate_control(RateControl::Cqp { qp: 22 })
}

/// An NV12 frame that moves with `n`.
fn nv12_frame(width: u32, height: u32, n: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let mut data = vec![0u8; w * h * 3 / 2];
    for y in 0..h {
        for x in 0..w {
            data[y * w + x] = ((x + y * 2 + n as usize * 5) ^ (x / 8)) as u8;
        }
    }
    for i in 0..w * h / 2 {
        data[w * h + i] = (i / 3 + n as usize * 7) as u8;
    }
    data
}

/// A BGRA frame that moves with `n`.
fn bgra_frame(width: u32, height: u32, n: u32) -> Vec<u8> {
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let (sx, sy) = (x + n * 3, y + n);
            data.extend_from_slice(&[(sy * 3) as u8, (sx * 2 + sy) as u8, (sx ^ sy) as u8, 255]);
        }
    }
    data
}

/// Which modifier a producer image gets.
enum Modifier<'a> {
    OneOf(&'a [u64]),
    Exactly(u64, &'a [vk::SubresourceLayout]),
}

/// The producer's side of a DMA-BUF: its own image over the buffer's memory,
/// which it writes frames into.
struct Producer {
    context: VideoContext,
    image: vk::Image,
    memory: vk::DeviceMemory,
    format: vk::Format,
    width: u32,
    height: u32,
    /// Whether it was written before, and so was released to the foreign
    /// family.
    written: bool,
}

impl Producer {
    /// Make an image over `buffer`'s memory.
    fn import(context: &VideoContext, buffer: &DmaBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let format = pixelforge::encoder::external::vk_format_of(buffer.format).unwrap();
        let layouts: Vec<vk::SubresourceLayout> = buffer
            .planes
            .iter()
            .map(|p| vk::SubresourceLayout {
                offset: p.offset,
                size: 0,
                row_pitch: p.stride,
                array_pitch: 0,
                depth_pitch: 0,
            })
            .collect();
        let image = Self::create_image(
            context,
            format,
            buffer.width,
            buffer.height,
            Modifier::Exactly(buffer.modifier, &layouts),
        )?;
        let device = context.device();
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let fd_fn = ash::khr::external_memory_fd::Device::load(context.instance(), device);
        let mut fd_props = vk::MemoryFdPropertiesKHR::default();
        unsafe {
            fd_fn.get_memory_fd_properties(
                vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                std::os::fd::AsRawFd::as_raw_fd(&buffer.fd),
                &mut fd_props,
            )
        }?;
        let type_index = context
            .find_memory_type(
                requirements.memory_type_bits & fd_props.memory_type_bits,
                vk::MemoryPropertyFlags::empty(),
            )
            .unwrap();
        let fd = std::os::fd::IntoRawFd::into_raw_fd(buffer.fd.try_clone()?);
        // Owned by Vulkan from here on, whatever the import does.
        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(fd);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(type_index)
            .push(&mut import)
            .push(&mut dedicated);
        let memory = unsafe { device.allocate_memory(&info, None) }?;
        unsafe { device.bind_image_memory(image, memory, 0) }?;
        Ok(Self {
            context: context.clone(),
            image,
            memory,
            format,
            width: buffer.width,
            height: buffer.height,
            written: false,
        })
    }

    /// Make a new buffer with `modifier`, the producer allocating it.
    fn allocate(
        context: &VideoContext,
        drm: u32,
        width: u32,
        height: u32,
        modifier: u64,
    ) -> Result<(Self, DmaBuf), Box<dyn std::error::Error>> {
        let format = pixelforge::encoder::external::vk_format_of(drm).unwrap();
        let image =
            Self::create_image(context, format, width, height, Modifier::OneOf(&[modifier]))?;
        let device = context.device();
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let type_index = context
            .find_memory_type(
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::empty(),
            )
            .unwrap();
        let mut export = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(type_index)
            .push(&mut export)
            .push(&mut dedicated);
        let memory = unsafe { device.allocate_memory(&info, None) }?;
        unsafe { device.bind_image_memory(image, memory, 0) }?;
        let fd_fn = ash::khr::external_memory_fd::Device::load(context.instance(), device);
        let fd = unsafe {
            fd_fn.get_memory_fd(
                &vk::MemoryGetFdInfoKHR::default()
                    .memory(memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT),
            )
        }?;
        let planes = (0..if drm == drm_format::NV12 { 2 } else { 1 })
            .map(|plane| {
                let aspect = [
                    vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
                    vk::ImageAspectFlags::MEMORY_PLANE_1_EXT,
                ][plane];
                let layout = unsafe {
                    device.get_image_subresource_layout(
                        image,
                        vk::ImageSubresource {
                            aspect_mask: aspect,
                            mip_level: 0,
                            array_layer: 0,
                        },
                    )
                };
                DmaBufPlane {
                    offset: layout.offset,
                    stride: layout.row_pitch,
                }
            })
            .collect();
        let buffer = DmaBuf {
            fd: unsafe { OwnedFd::from_raw_fd(fd) },
            width,
            height,
            format: drm,
            modifier,
            planes,
        };
        let producer = Self {
            context: context.clone(),
            image,
            memory,
            format,
            width,
            height,
            written: false,
        };
        Ok((producer, buffer))
    }

    fn create_image(
        context: &VideoContext,
        format: vk::Format,
        width: u32,
        height: u32,
        modifier: Modifier<'_>,
    ) -> Result<vk::Image, Box<dyn std::error::Error>> {
        let mut external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let (mut list, mut explicit) = match modifier {
            Modifier::OneOf(modifiers) => (
                Some(
                    vk::ImageDrmFormatModifierListCreateInfoEXT::default()
                        .drm_format_modifiers(modifiers),
                ),
                None,
            ),
            Modifier::Exactly(modifier, layouts) => (
                None,
                Some(
                    vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
                        .drm_format_modifier(modifier)
                        .plane_layouts(layouts),
                ),
            ),
        };
        let mut info = vk::ImageCreateInfo::default()
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
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push(&mut external);
        if let Some(list) = list.as_mut() {
            info = info.push(list);
        }
        if let Some(explicit) = explicit.as_mut() {
            info = info.push(explicit);
        }
        Ok(unsafe { context.device().create_image(&info, None) }?)
    }

    /// Write `data` into the buffer and hand it to the foreign family, as a
    /// producer in another process would leave it.
    fn write(&mut self, data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        let context = self.context.clone();
        let device = context.device();
        let (staging, staging_memory) = unsafe {
            host_buffer(
                &context,
                data.len() as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
            )?
        };
        unsafe {
            let ptr = device.map_memory(
                staging_memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast::<u8>(), data.len());
            device.unmap_memory(staging_memory);
        }
        let family = context.compute_queue_family();
        let color = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let (w, h) = (self.width, self.height);
        let regions: Vec<vk::BufferImageCopy> =
            if self.format == vk::Format::G8_B8R8_2PLANE_420_UNORM {
                vec![
                    plane_copy(0, vk::ImageAspectFlags::PLANE_0, w, h),
                    plane_copy((w * h) as u64, vk::ImageAspectFlags::PLANE_1, w / 2, h / 2),
                ]
            } else {
                vec![plane_copy(0, vk::ImageAspectFlags::COLOR, w, h)]
            };
        let written = self.written;
        unsafe {
            one_shot(&context, |cb| {
                let to_dst = vk::ImageMemoryBarrier::default()
                    .old_layout(if written {
                        vk::ImageLayout::GENERAL
                    } else {
                        vk::ImageLayout::UNDEFINED
                    })
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(if written {
                        vk::QUEUE_FAMILY_FOREIGN_EXT
                    } else {
                        vk::QUEUE_FAMILY_IGNORED
                    })
                    .dst_queue_family_index(if written {
                        family
                    } else {
                        vk::QUEUE_FAMILY_IGNORED
                    })
                    .image(self.image)
                    .subresource_range(color)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_dst],
                );
                device.cmd_copy_buffer_to_image(
                    cb,
                    staging,
                    self.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                );
                let release = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(family)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                    .image(self.image)
                    .subresource_range(color)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[release],
                );
            })?;
            device.destroy_buffer(staging, None);
            device.free_memory(staging_memory, None);
        }
        self.written = true;
        Ok(())
    }
}

impl Drop for Producer {
    fn drop(&mut self) {
        let device = self.context.device();
        unsafe {
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

fn plane_copy(
    offset: u64,
    aspect: vk::ImageAspectFlags,
    width: u32,
    height: u32,
) -> vk::BufferImageCopy {
    vk::BufferImageCopy {
        buffer_offset: offset,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D::default(),
        image_extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
    }
}

/// The stream of encoding `frames` from an ordinary YUV image.
fn encode_reference(
    context: &VideoContext,
    config: EncodeConfig,
    frames: &[Vec<u8>],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut encoder = Encoder::new(context.clone(), config.clone())?;
    let mut input = YuvUploader::for_encoder(&encoder)?;
    let mut stream = Vec::new();
    for frame in frames {
        input.upload(YuvLayout::Nv12, frame)?;
        stream.extend(pollster::block_on(encoder.submit(Frame::image(input.source()))?)?.data);
    }
    Ok(stream)
}

/// The stream of encoding `frames` written by `producer` into `external`.
fn encode_external(
    encoder: &mut Encoder,
    producer: &mut Producer,
    external: &DmaBufImage,
    frames: &[Vec<u8>],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut stream = Vec::new();
    for frame in frames {
        producer.write(frame)?;
        let packet = pollster::block_on(encoder.submit(Frame::external(external))?)?;
        stream.extend(packet.data);
    }
    Ok(stream)
}

/// Every modifier the encoder can allocate.
fn exportable_modifiers(encoder: &Encoder) -> Vec<u64> {
    encoder.dmabuf_format().map_or(Vec::new(), |formats| {
        formats
            .modifiers
            .iter()
            .filter(|m| m.exportable)
            .map(|m| m.modifier)
            .collect()
    })
}

/// A buffer the encoder allocated encodes exactly what an ordinary image with
/// the same pixels does, whether the encode reads it directly or copies it.
/// Then the same buffer, imported back, does too.
#[test]
#[ignore = "requires a Vulkan Video device with DMA-BUF support"]
fn allocated_and_imported_inputs_encode_the_same() -> Result<(), Box<dyn std::error::Error>> {
    let context = context()?;
    // A size every codec's block size divides, so the reference's coded
    // extent is the display size too and the streams can match exactly.
    let (w, h) = (320, 256);
    let frames: Vec<Vec<u8>> = (0..FRAMES).map(|n| nv12_frame(w, h, n)).collect();
    for codec in [Codec::H264, Codec::H265, Codec::AV1] {
        if !context.supports_encode(codec) {
            continue;
        }
        let reference = encode_reference(&context, config(codec, w, h), &frames)?;

        let mut encoder = Encoder::new(context.clone(), config(codec, w, h))?;
        let modifiers = exportable_modifiers(&encoder);
        if modifiers.is_empty() {
            eprintln!("skipping {codec:?}: no modifier the encoder can allocate");
            continue;
        }
        let allocated = encoder.allocate_input(&modifiers)?;
        println!(
            "{codec:?}: allocated modifier {:#x}, zero copy {}",
            allocated.modifier(),
            allocated.is_zero_copy()
        );
        let exported = allocated.export()?;
        let mut producer = Producer::import(&context, &exported)?;
        let stream = encode_external(&mut encoder, &mut producer, &allocated, &frames)?;
        assert!(stream == reference, "{codec:?}: allocated input differs");

        let mut encoder = Encoder::new(context.clone(), config(codec, w, h))?;
        let imported = encoder.import(&exported)?;
        assert_eq!(imported.is_zero_copy(), allocated.is_zero_copy());
        let stream = encode_external(&mut encoder, &mut producer, &imported, &frames)?;
        assert!(stream == reference, "{codec:?}: imported input differs");
    }
    Ok(())
}

/// A buffer with a modifier the encode cannot read is copied instead, and
/// encodes the same.
#[test]
#[ignore = "requires a Vulkan Video device with DMA-BUF support"]
fn copy_only_modifiers_are_copied() -> Result<(), Box<dyn std::error::Error>> {
    let context = context()?;
    let (w, h) = (320, 256);
    let frames: Vec<Vec<u8>> = (0..FRAMES).map(|n| nv12_frame(w, h, n)).collect();
    let reference = encode_reference(&context, config(Codec::H264, w, h), &frames)?;
    let mut encoder = Encoder::new(context.clone(), config(Codec::H264, w, h))?;
    let formats = encoder.dmabuf_format()?;
    let Some(copy_only) = formats
        .modifiers
        .iter()
        .find(|m| m.importable && !m.zero_copy)
    else {
        eprintln!("skipping: every importable modifier is read directly here");
        return Ok(());
    };
    let (mut producer, buffer) =
        Producer::allocate(&context, drm_format::NV12, w, h, copy_only.modifier)?;
    let imported = encoder.import(&buffer)?;
    assert!(!imported.is_zero_copy());
    let stream = encode_external(&mut encoder, &mut producer, &imported, &frames)?;
    assert!(stream == reference, "copied input differs");
    Ok(())
}

/// An RGB buffer goes to the colour converter, which reads it directly, and
/// encodes what converting an ordinary RGB image does.
#[test]
#[ignore = "requires a Vulkan Video device with DMA-BUF support"]
fn rgb_buffers_are_converted_directly() -> Result<(), Box<dyn std::error::Error>> {
    let context = context()?;
    let (w, h) = (320, 256);
    let rgb = Source::Rgb(RgbSource::new(RgbFormat::BGRA, ColorSpec::Srgb));
    let shader = |c: EncodeConfig| {
        c.with_source(rgb)
            .with_conversion(ConversionPreference::Shader)
    };

    let mut reference_encoder = Encoder::new(context.clone(), shader(config(Codec::H264, w, h)))?;
    let mut reference = Vec::new();
    let mut encoder = Encoder::new(context.clone(), shader(config(Codec::H264, w, h)))?;
    assert_eq!(encoder.conversion(), Conversion::Shader);
    let modifiers = exportable_modifiers(&encoder);
    if modifiers.is_empty() {
        eprintln!("skipping: no RGB modifier the shader reads");
        return Ok(());
    }
    let allocated = encoder.allocate_input(&modifiers)?;
    assert!(
        allocated.is_zero_copy(),
        "the shader reads what it can import"
    );
    let mut producer = Producer::import(&context, &allocated.export()?)?;
    let mut stream = Vec::new();
    for n in 0..FRAMES {
        let data = bgra_frame(w, h, n);
        let src = unsafe { common::source::create_src_image(&context, w, h, &data)? };
        let image = SourceImage::new(
            &context,
            src.image,
            Handoff::Device {
                layout: vk::ImageLayout::GENERAL,
            },
        )?;
        reference.extend(pollster::block_on(reference_encoder.submit(Frame::image(&image))?)?.data);
        producer.write(&data)?;
        stream.extend(pollster::block_on(encoder.submit(Frame::external(&allocated))?)?.data);
    }
    assert!(stream == reference, "converted DMA-BUF differs");
    Ok(())
}

/// A source smaller than the aligned coded extent is read directly at the
/// display size, and decodes to the frames it was given.
#[test]
#[ignore = "requires a Vulkan Video device with DMA-BUF support"]
fn unaligned_sizes_are_read_at_the_display_size() -> Result<(), Box<dyn std::error::Error>> {
    let context = VideoContextBuilder::new()
        .enable_validation(std::env::var("PIXELFORGE_VALIDATION").is_ok())
        .require_encode(Codec::H264)
        .require_decode(Codec::H264)
        .build()?;
    // 232 rows is not a whole number of macroblocks.
    let (w, h) = (320, 232);
    let frames: Vec<Vec<u8>> = (0..FRAMES).map(|n| nv12_frame(w, h, n)).collect();
    let mut encoder = Encoder::new(context.clone(), config(Codec::H264, w, h))?;
    let allocated = encoder.allocate_input(&exportable_modifiers(&encoder))?;
    if !allocated.is_zero_copy() {
        eprintln!("skipping: the encoder copies DMA-BUFs on this device");
        return Ok(());
    }
    let mut producer = Producer::import(&context, &allocated.export()?)?;
    let stream = encode_external(&mut encoder, &mut producer, &allocated, &frames)?;

    let mut readback = Readback::new(&context)?;
    let mut decoder = Decoder::new(
        context.clone(),
        DecodeConfig::new(Codec::H264).with_byte_stream(),
    )?;
    let mut decoded = Vec::new();
    decode_stream(&mut decoder, &stream, |frame| {
        decoded.push(readback.read(&frame)?.y);
        Ok(())
    })?;
    assert_eq!(decoded.len(), FRAMES as usize);
    for (n, (got, want)) in decoded.iter().zip(&frames).enumerate() {
        let want = &want[..(w * h) as usize];
        assert_eq!(got.len(), want.len(), "frame {n}: decoded size");
        let mse = got
            .iter()
            .zip(want)
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum::<f64>()
            / want.len() as f64;
        let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
        assert!(psnr > 30.0, "frame {n}: {psnr:.1} dB");
    }
    Ok(())
}

/// The release of one frame, as a sync_file, gates the next one's read.
#[test]
#[ignore = "requires a Vulkan Video device with sync_file support"]
fn sync_files_chain_frames() -> Result<(), Box<dyn std::error::Error>> {
    let context = context()?;
    let (w, h) = (320, 256);
    let mut encoder = Encoder::new(context.clone(), config(Codec::H264, w, h))?;
    let modifiers = exportable_modifiers(&encoder);
    if modifiers.is_empty() {
        eprintln!("skipping: no modifier the encoder can allocate");
        return Ok(());
    }
    let allocated = encoder.allocate_input(&modifiers)?;
    let mut producer = Producer::import(&context, &allocated.export()?)?;
    let mut fence = None;
    let mut released = 0;
    for n in 0..FRAMES {
        producer.write(&nv12_frame(w, h, n))?;
        let mut frame = Frame::external(&allocated).signal_sync_file();
        if let Some(fence) = fence.take() {
            frame = frame.wait_sync_file(fence);
        }
        let mut future = encoder.submit(frame)?;
        fence = future.take_release_sync_file();
        released += usize::from(fence.is_some());
        assert!(!pollster::block_on(future)?.data.is_empty());
    }
    println!("{released} of {FRAMES} releases came as a pending sync_file");
    Ok(())
}
