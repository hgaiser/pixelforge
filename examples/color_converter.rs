//! Example: reproduce a hang in `vkCreateComputePipelines`.
//!
//! [`ColorConverter`] builds its compute pipeline from an embedded SPIR-V
//! module (`shader/color_convert.spv`) using a descriptor-buffer set layout.
//! On some drivers (notably RADV) `vkCreateComputePipelines` can hang while
//! compiling that shader. This example isolates where:
//!
//! - **Variant A**: compute pipeline with a *plain* descriptor set layout (no
//!   `DESCRIPTOR_BUFFER_EXT` flag). If this hangs, the shader/SPIR-V trips the
//!   driver's shader compiler itself.
//! - **Variant B**: same shader with a `DESCRIPTOR_BUFFER_EXT` set layout —
//!   the layout [`ColorConverter`] actually uses. If A passes and this hangs,
//!   the driver's descriptor-buffer pipeline path is the culprit.
//! - **Variant C**: full [`ColorConverter::new`] — reproduces the exact call
//!   moonshine's video pipeline makes on its first frame.
//!
//! Usage:
//! ```text
//! cargo run --example color_converter [-- --variant a|b|c|all]
//! ```
//!
//! `--variant` defaults to `all` (run A, then B, then C). A hang blocks the
//! process; the `--- Variant ...` line printed just before tells you where it
//! got stuck. Useful driver-side switches to bisect further:
//!
//! - `RADV_PERFTEST=llvm` (or your driver's "force LLVM backend" toggle)
//! - `vulkaninfo | grep -i driverName` to record the exact driver version
//! - `RUST_LOG=debug cargo run --example color_converter` to see pixelforge's
//!   internal Vulkan logging

use ash::vk;
use pixelforge::{
    ColorConverter, ColorConverterConfig, InputFormat, OutputFormat, VideoContextBuilder,
};
use std::time::Instant;
use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};

/// Precompiled color-conversion compute shader — the same bytes the library embeds.
const COLOR_CONVERT_SPIRV: &[u8] = include_bytes!("../shader/color_convert.spv");

/// Resolution used for the full-converter variant (matches moonshine's stream).
const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing.
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer().with_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
            ),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let variant = args
        .iter()
        .position(|a| a == "--variant")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("all");

    println!("PixelForge Color Converter Pipeline Test\n");

    println!("Creating Vulkan video context...");
    let context = VideoContextBuilder::new()
        .app_name("Color Converter Pipeline Test")
        .build()?;

    let props = context.device_properties();
    let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }.to_string_lossy();
    let api = props.api_version;
    let drv = props.driver_version;
    println!(
        "GPU: {name}, API {}.{}.{}, driver {}.{}.{}",
        api >> 22,
        (api >> 12) & 0x3FF,
        api & 0xFFF,
        drv >> 22,
        (drv >> 12) & 0x3FF,
        drv & 0xFFF
    );
    println!(
        "VK_EXT_descriptor_buffer: {}",
        if context.has_descriptor_buffer() {
            "yes"
        } else {
            "no"
        }
    );

    // The shader module is identical for both pipeline variants.
    let words: Vec<u32> = COLOR_CONVERT_SPIRV
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&words);
    let shader_module = unsafe { context.device().create_shader_module(&shader_info, None) }
        .map_err(|e| format!("create_shader_module: {e}"))?;

    if variant == "a" || variant == "all" {
        println!("\n--- Variant A: compute pipeline, plain set layout ---");
        match create_pipeline(&context, shader_module, false) {
            Ok(()) => println!("Variant A OK: pipeline created."),
            Err(e) => println!("Variant A FAILED: {e}"),
        }
    }

    if variant == "b" || variant == "all" {
        println!("\n--- Variant B: compute pipeline, descriptor-buffer set layout ---");
        match create_pipeline(&context, shader_module, true) {
            Ok(()) => println!("Variant B OK: pipeline created."),
            Err(e) => println!("Variant B FAILED: {e}"),
        }
    }

    if variant == "c" || variant == "all" {
        println!("\n--- Variant C: full ColorConverter::new() ---");
        let config =
            ColorConverterConfig::new(WIDTH, HEIGHT, InputFormat::RGBA, OutputFormat::NV12);
        match ColorConverter::new(context.clone(), config) {
            Ok(_) => println!("Variant C OK: color converter created."),
            Err(e) => println!("Variant C FAILED: {e}"),
        }
    }

    unsafe { context.device().destroy_shader_module(shader_module, None) };
    println!("\nDone.");
    Ok(())
}

/// Create a compute pipeline for the color-conversion shader, with or without
/// the descriptor-buffer set-layout flag.
fn create_pipeline(
    context: &pixelforge::VideoContext,
    shader_module: vk::ShaderModule,
    descriptor_buffer: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = context.device();

    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];

    let mut layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    if descriptor_buffer {
        layout_info = layout_info.flags(vk::DescriptorSetLayoutCreateFlags::DESCRIPTOR_BUFFER_EXT);
    }
    let set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }
        .map_err(|e| format!("create_descriptor_set_layout: {e}"))?;

    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(28); // 7 x u32 push constants, matching the converter
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&set_layout))
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
        .map_err(|e| format!("create_pipeline_layout: {e}"))?;

    let entry_point = std::ffi::CString::new("main").unwrap();
    let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&entry_point);

    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage_info)
        .layout(pipeline_layout);

    println!("Calling create_compute_pipelines (descriptor_buffer={descriptor_buffer})...");
    let start = Instant::now();
    let pipeline = unsafe {
        device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    }
    .map_err(|(res, _)| format!("create_compute_pipelines: {res:?}"))?[0];
    println!("create_compute_pipelines returned in {:?}", start.elapsed());

    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
    }

    Ok(())
}
