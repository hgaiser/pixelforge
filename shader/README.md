# PixelForge Shaders

Precompiled Vulkan compute shaders for GPU-accelerated color format conversion.

## Shaders

**color_convert.comp** — RGB → YUV compute shader.
Converts BGRx/RGBx/BGRA/RGBA/ABGR2101010/RGBA16F input to NV12/I420/YUV444/P010/YUV444P10 output using BT.709, BT.2020, or sRGB→BT.2020+PQ color space matrices.

## Compilation

Requires `glslc` from the Vulkan SDK.

```bash
./compile.sh
```

This compiles the shaders to SPIR-V 1.3 (Vulkan 1.1, optimized), the newest
version a Vulkan 1.1 device must accept: pixelforge also runs on a device the
caller created, which can be 1.1. The `.spv` files are included in source via
`include_bytes!`, so rerun this after editing a `.comp` and commit the result.

The examples carry one more, outside the library: `examples/shader/sample_frame.comp`
reads a decoded frame through a sampler-YCbCr conversion and writes RGBA, which
is what `examples/sample_frame.rs` uses to show the zero-copy render path.
