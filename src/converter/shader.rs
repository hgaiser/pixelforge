//! Compute shader for color format conversion.
//!
//! This module contains the SPIR-V bytecode for the color conversion compute shader.
//! The shader converts RGB/BGR formats to various YUV formats using BT.709 (SDR)
//! or BT.2020 (HDR) coefficients, selected via push constants.

use crate::error::{PixelForgeError, Result};
use std::sync::OnceLock;

/// Cached compiled SPIR-V bytecode.
static SPIRV_CACHE: OnceLock<Vec<u32>> = OnceLock::new();

/// Get the SPIR-V bytecode for the color conversion shader.
///
/// The shader expects:
/// - Push constants: width, height, input_format, output_format, color_space, full_range (6 × u32)
/// - Binding 0: Input image (sampler2D)
/// - Binding 1: Output buffer (YUV data)
///
/// Workgroup size: 8x8x1.
///
/// The compiled SPIR-V is cached after the first successful compilation.
pub fn get_spirv_code() -> Result<Vec<u32>> {
    if let Some(cached) = SPIRV_CACHE.get() {
        return Ok(cached.clone());
    }
    let spirv = compile_glsl_to_spirv()?;
    Ok(SPIRV_CACHE.get_or_init(|| spirv).clone())
}

/// Compile GLSL to SPIR-V at runtime using shaderc.
fn compile_glsl_to_spirv() -> Result<Vec<u32>> {
    const SHADER_SOURCE: &str = r#"
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
    uint input_format;   // 0=BGRx, 1=RGBx, 2=BGRA, 3=RGBA, 4=ABGR2101010, 5=RGBA16F
    uint output_format;  // 0=NV12, 1=I420, 2=YUV444, 3=P010, 4=YUV444P10
    uint color_space;    // 0=BT.709, 1=BT.2020, 2=sRGB→BT.2020+PQ
    uint full_range;     // 0=limited/studio range, 1=full range
    float sdr_white_nits; // SDR reference white (nits), used for sRGB→BT.2020+PQ
    uint source_width;
    uint source_height;
    uint preserve_aspect;
} params;

// Source image sampled directly — eliminates the image-to-buffer copy.
layout(binding = 0) uniform sampler2D inputImage;

layout(std430, binding = 1) buffer OutputBuffer {
    uint output_data[];
};

// BT.709 conversion coefficients (SDR).
const float BT709_Y_R = 0.2126;
const float BT709_Y_G = 0.7152;
const float BT709_Y_B = 0.0722;
const float BT709_U_R = -0.1146;
const float BT709_U_G = -0.3854;
const float BT709_U_B = 0.5000;
const float BT709_V_R = 0.5000;
const float BT709_V_G = -0.4542;
const float BT709_V_B = -0.0458;

// BT.2020 conversion coefficients (HDR).
const float BT2020_Y_R = 0.2627;
const float BT2020_Y_G = 0.6780;
const float BT2020_Y_B = 0.0593;
const float BT2020_U_R = -0.1396;
const float BT2020_U_G = -0.3604;
const float BT2020_U_B = 0.5000;
const float BT2020_V_R = 0.5000;
const float BT2020_V_G = -0.4598;
const float BT2020_V_B = -0.0402;

// PQ (ST 2084) constants for inverse EOTF.
const float PQ_M1 = 0.1593017578125;
const float PQ_M2 = 78.84375;
const float PQ_C1 = 0.8359375;
const float PQ_C2 = 18.8515625;
const float PQ_C3 = 18.6875;

// Apply PQ inverse EOTF: linear light [0,1] → PQ signal [0,1].
// Input should be normalized to [0,1] where 1.0 = 10,000 nits.
vec3 linear_to_pq(vec3 L) {
    L = max(L, vec3(0.0));
    vec3 Lm1 = pow(L, vec3(PQ_M1));
    vec3 N = pow((PQ_C1 + PQ_C2 * Lm1) / (1.0 + PQ_C3 * Lm1), vec3(PQ_M2));
    return N;
}

// Inverse sRGB EOTF: sRGB gamma-encoded [0,1] → linear light [0,1].
// Uses the standard piecewise sRGB transfer function.
vec3 srgb_to_linear(vec3 srgb) {
    // Clamp to [0,1] to avoid NaN from pow() on negative/out-of-range FP16 inputs.
    srgb = clamp(srgb, 0.0, 1.0);
    // sRGB piecewise: linear below 0.04045, gamma above.
    return mix(
        pow((srgb + 0.055) / 1.055, vec3(2.4)),
        srgb / 12.92,
        lessThanEqual(srgb, vec3(0.04045))
    );
}

// BT.709 to BT.2020 gamut mapping matrix (applied in linear light).
// Derived from the BT.709 and BT.2020 primaries via the CIE XYZ intermediate.
vec3 bt709_to_bt2020(vec3 rgb709) {
    return vec3(
        0.6274 * rgb709.r + 0.3293 * rgb709.g + 0.0433 * rgb709.b,
        0.0691 * rgb709.r + 0.9195 * rgb709.g + 0.0114 * rgb709.b,
        0.0164 * rgb709.r + 0.0880 * rgb709.g + 0.8956 * rgb709.b
    );
}

// Read normalized RGB from source image via the sampler.
// Returns values in [0, 1] range representing the final signal level for YUV conversion.
//
// The compositor (Smithay) does NOT perform color-managed compositing: it blits
// client textures to the GBM output without applying any transfer function.
// Therefore the GBM buffer preserves the source's encoding:
//   - PQ surfaces → PQ-encoded values (even in FP16)
//   - sRGB surfaces → sRGB gamma-encoded values (even in FP16)
//
// color_space 0 (BT.709) and 1 (BT.2020): passthrough — data is already encoded.
// color_space 2 (sRGB→BT.2020+PQ): decode sRGB, convert gamut, apply PQ.
vec3 read_source_rgb(vec2 source_coord) {
    vec2 source_size = vec2(float(max(params.source_width, 1u)), float(max(params.source_height, 1u)));
    vec2 uv = (source_coord + vec2(0.5)) / source_size;
    vec4 rgba = texture(inputImage, uv);
    if (params.color_space == 2u) {
        // sRGB→BT.2020+PQ: decode sRGB gamma → linear BT.709 → BT.2020 gamut → PQ.
        vec3 linear_709 = srgb_to_linear(rgba.rgb);
        vec3 linear_2020 = bt709_to_bt2020(linear_709);
        // SDR reference white (configurable, default 203 nits per ITU-R BT.2408)
        // normalized to PQ's 10000 nit scale.
        return linear_to_pq(linear_2020 * (params.sdr_white_nits / 10000.0));
    }
    // BT.709 or BT.2020 passthrough: values are already properly encoded.
    return rgba.rgb;
}

bool source_coord_for_output(uint out_x, uint out_y, out vec2 source_coord) {
    float source_w = float(max(params.source_width, 1u));
    float source_h = float(max(params.source_height, 1u));
    float output_w = float(max(params.width, 1u));
    float output_h = float(max(params.height, 1u));

    vec2 content_origin = vec2(0.0);
    vec2 content_size = vec2(output_w, output_h);
    if (params.preserve_aspect != 0u) {
        float scale = min(output_w / source_w, output_h / source_h);
        content_size = vec2(source_w * scale, source_h * scale);
        content_origin = (vec2(output_w, output_h) - content_size) * 0.5;
    }

    vec2 output_coord = vec2(float(out_x) + 0.5, float(out_y) + 0.5);
    if (output_coord.x < content_origin.x ||
        output_coord.y < content_origin.y ||
        output_coord.x >= content_origin.x + content_size.x ||
        output_coord.y >= content_origin.y + content_size.y) {
        source_coord = vec2(0.0);
        return false;
    }

    vec2 normalized = (output_coord - content_origin) / content_size;
    source_coord = normalized * vec2(source_w, source_h) - vec2(0.5);
    source_coord = clamp(source_coord, vec2(0.0), vec2(source_w - 1.0, source_h - 1.0));
    return true;
}

vec3 read_rgb(uint out_x, uint out_y) {
    vec2 source_coord;
    if (!source_coord_for_output(out_x, out_y, source_coord)) {
        return vec3(0.0);
    }
    return read_source_rgb(source_coord);
}

// Convert normalized RGB [0,1] to YUV.
// Returns Y in [0, 1], U and V in [0, 1] centered at 0.5.
vec3 rgb_to_yuv(vec3 rgb) {
    float yr, yg, yb, ur, ug, ub, vr, vg, vb;
    if (params.color_space >= 1u) {
        // BT.2020 (also used for sRGB→BT.2020+PQ, since read_rgb already converted).
        yr = BT2020_Y_R; yg = BT2020_Y_G; yb = BT2020_Y_B;
        ur = BT2020_U_R; ug = BT2020_U_G; ub = BT2020_U_B;
        vr = BT2020_V_R; vg = BT2020_V_G; vb = BT2020_V_B;
    } else {
        // BT.709 (default)
        yr = BT709_Y_R; yg = BT709_Y_G; yb = BT709_Y_B;
        ur = BT709_U_R; ug = BT709_U_G; ub = BT709_U_B;
        vr = BT709_V_R; vg = BT709_V_G; vb = BT709_V_B;
    }
    float y = yr * rgb.r + yg * rgb.g + yb * rgb.b;
    float u = 0.5 + ur * rgb.r + ug * rgb.g + ub * rgb.b;
    float v = 0.5 + vr * rgb.r + vg * rgb.g + vb * rgb.b;
    return vec3(clamp(y, 0.0, 1.0), clamp(u, 0.0, 1.0), clamp(v, 0.0, 1.0));
}

// --- 8-bit quantization ---

uint q8_y(float y) {
    if (params.full_range == 0u) return uint(clamp(y * 219.0 + 16.0, 0.0, 255.0));
    return uint(clamp(y * 255.0, 0.0, 255.0));
}

uint q8_c(float c) {
    if (params.full_range == 0u) return uint(clamp((c - 0.5) * 224.0 + 128.0, 0.0, 255.0));
    return uint(clamp(c * 255.0, 0.0, 255.0));
}

// --- 10-bit quantization (P010 layout: value in upper 10 bits of 16-bit word) ---

uint q10_y(float y) {
    uint val;
    if (params.full_range == 0u) val = uint(clamp(y * 876.0 + 64.0, 0.0, 1023.0));
    else val = uint(clamp(y * 1023.0, 0.0, 1023.0));
    return (val << 6u) & 0xFFC0u;
}

uint q10_c(float c) {
    uint val;
    if (params.full_range == 0u) val = uint(clamp((c - 0.5) * 896.0 + 512.0, 0.0, 1023.0));
    else val = uint(clamp(c * 1023.0, 0.0, 1023.0));
    return (val << 6u) & 0xFFC0u;
}

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;

    if (x >= params.width || y >= params.height) return;

    uint pixel_idx = y * params.width + x;
    vec3 rgb = read_rgb(x, y);
    vec3 yuv = rgb_to_yuv(rgb);

    uint pixel_count = params.width * params.height;

    if (params.output_format == 2u) {
        // YUV444 8-bit: 2-plane semi-planar (Y plane + UV interleaved).
        uint y_byte_idx = pixel_idx;
        uint y_word_idx = y_byte_idx / 4u;
        uint y_byte_offset = y_byte_idx % 4u;
        atomicOr(output_data[y_word_idx], q8_y(yuv.x) << (y_byte_offset * 8u));

        // Align UV plane offset to 4 bytes for VkBufferImageCopy compliance.
        uint uv_base = (pixel_count + 3u) & ~3u;
        uint uv_byte_idx = uv_base + pixel_idx * 2u;
        uint uv_word_idx = uv_byte_idx / 4u;
        uint uv_byte_offset = uv_byte_idx % 4u;
        atomicOr(output_data[uv_word_idx], q8_c(yuv.y) << (uv_byte_offset * 8u));

        uint v_byte_idx = uv_byte_idx + 1u;
        uint v_word_idx = v_byte_idx / 4u;
        uint v_byte_offset = v_byte_idx % 4u;
        atomicOr(output_data[v_word_idx], q8_c(yuv.z) << (v_byte_offset * 8u));
    } else if (params.output_format == 4u) {
        // YUV444P10 (10-bit): 2-plane semi-planar format.
        uint y_half_offset = pixel_idx % 2u;
        uint y_packed_idx = pixel_idx / 2u;
        atomicOr(output_data[y_packed_idx], q10_y(yuv.x) << (y_half_offset * 16u));

        // Align UV base to 4 bytes: Y plane is ceil(pixel_count/2) words.
        uint uv_base_words = (pixel_count + 1u) / 2u;
        uint uv_word_idx = uv_base_words + pixel_idx;
        uint uv_packed = q10_c(yuv.y) | (q10_c(yuv.z) << 16u);
        output_data[uv_word_idx] = uv_packed;
    } else if (params.output_format == 3u) {
        // P010 (10-bit NV12): 2-plane semi-planar, 4:2:0 subsampling.
        uint y_half_offset = pixel_idx % 2u;
        uint y_packed_idx = pixel_idx / 2u;
        atomicOr(output_data[y_packed_idx], q10_y(yuv.x) << (y_half_offset * 16u));

        if ((x % 2u == 0u) && (y % 2u == 0u)) {
            uint uv_x = x / 2u;
            uint uv_y = y / 2u;
            uint uv_width = params.width / 2u;
            uint uv_idx = uv_y * uv_width + uv_x;

            vec3 yuv00 = yuv;
            vec3 yuv10 = (x + 1u < params.width) ?
                rgb_to_yuv(read_rgb(x + 1u, y)) : yuv00;
            vec3 yuv01 = (y + 1u < params.height) ?
                rgb_to_yuv(read_rgb(x, y + 1u)) : yuv00;
            vec3 yuv11 = (x + 1u < params.width && y + 1u < params.height) ?
                rgb_to_yuv(read_rgb(x + 1u, y + 1u)) : yuv00;

            float avg_u = (yuv00.y + yuv10.y + yuv01.y + yuv11.y) / 4.0;
            float avg_v = (yuv00.z + yuv10.z + yuv01.z + yuv11.z) / 4.0;

            uint uv_base_words = pixel_count / 2u;
            uint uv_word_idx = uv_base_words + uv_idx;
            uint uv_packed = q10_c(avg_u) | (q10_c(avg_v) << 16u);
            output_data[uv_word_idx] = uv_packed;
        }
    } else {
        // YUV420 8-bit (NV12 or I420): Write Y for every pixel.
        uint y_byte_idx = pixel_idx;
        uint y_word_idx = y_byte_idx / 4u;
        uint y_byte_offset = y_byte_idx % 4u;
        atomicOr(output_data[y_word_idx], q8_y(yuv.x) << (y_byte_offset * 8u));

        if ((x % 2u == 0u) && (y % 2u == 0u)) {
            uint uv_x = x / 2u;
            uint uv_y = y / 2u;
            uint uv_width = params.width / 2u;
            uint uv_idx = uv_y * uv_width + uv_x;

            vec3 yuv00 = yuv;
            vec3 yuv10 = (x + 1u < params.width) ?
                rgb_to_yuv(read_rgb(x + 1u, y)) : yuv00;
            vec3 yuv01 = (y + 1u < params.height) ?
                rgb_to_yuv(read_rgb(x, y + 1u)) : yuv00;
            vec3 yuv11 = (x + 1u < params.width && y + 1u < params.height) ?
                rgb_to_yuv(read_rgb(x + 1u, y + 1u)) : yuv00;

            float avg_u = (yuv00.y + yuv10.y + yuv01.y + yuv11.y) / 4.0;
            float avg_v = (yuv00.z + yuv10.z + yuv01.z + yuv11.z) / 4.0;

            if (params.output_format == 0u) {
                // NV12: Interleaved UV after Y plane.
                uint uv_base_bytes = pixel_count;
                uint uv_byte_idx = uv_base_bytes + uv_idx * 2u;
                uint uv_word_idx = uv_byte_idx / 4u;
                uint uv_byte_offset = uv_byte_idx % 4u;

                if (uv_byte_offset <= 2u) {
                    uint uv_packed = (q8_c(avg_v) << 8u) | q8_c(avg_u);
                    atomicOr(output_data[uv_word_idx], uv_packed << (uv_byte_offset * 8u));
                } else {
                    atomicOr(output_data[uv_word_idx], q8_c(avg_u) << 24u);
                    atomicOr(output_data[uv_word_idx + 1u], q8_c(avg_v));
                }
            } else {
                // I420: Separate U and V planes.
                uint uv_plane_size = pixel_count / 4u;

                uint u_base_bytes = pixel_count;
                uint u_byte_idx = u_base_bytes + uv_idx;
                uint u_word_idx = u_byte_idx / 4u;
                uint u_byte_offset = u_byte_idx % 4u;

                uint v_base_bytes = pixel_count + uv_plane_size;
                uint v_byte_idx = v_base_bytes + uv_idx;
                uint v_word_idx = v_byte_idx / 4u;
                uint v_byte_offset = v_byte_idx % 4u;

                atomicOr(output_data[u_word_idx], q8_c(avg_u) << (u_byte_offset * 8u));
                atomicOr(output_data[v_word_idx], q8_c(avg_v) << (v_byte_offset * 8u));
            }
        }
    }
}
"#;

    let compiler = shaderc::Compiler::new().ok_or_else(|| {
        PixelForgeError::ShaderCompilation("failed to create shaderc compiler".into())
    })?;
    let options = shaderc::CompileOptions::new().ok_or_else(|| {
        PixelForgeError::ShaderCompilation("failed to create compile options".into())
    })?;

    let artifact = compiler
        .compile_into_spirv(
            SHADER_SOURCE,
            shaderc::ShaderKind::Compute,
            "color_convert.comp",
            "main",
            Some(&options),
        )
        .map_err(|e| PixelForgeError::ShaderCompilation(e.to_string()))?;

    Ok(artifact.as_binary().to_vec())
}
