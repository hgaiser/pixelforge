#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPV_PATH="$SCRIPT_DIR/color_convert.spv"

if ! command -v glslc &>/dev/null; then
    echo "Error: glslc not found. Install the Vulkan SDK or glslc package." >&2
    exit 1
fi

echo "Compiling color_convert.comp → color_convert.spv"
glslc -x glsl --target-env=vulkan1.1 -O \
    "$SCRIPT_DIR/color_convert.comp" \
    -o "$SPV_PATH"

FILE_SIZE=$(stat -c%s "$SPV_PATH" 2>/dev/null || stat -f%z "$SPV_PATH" 2>/dev/null)
echo "Written $FILE_SIZE bytes to $SPV_PATH"

# The examples carry shaders of their own.
for name in sample_frame sample_planes; do
    EXAMPLE_SRC="$PROJECT_ROOT/examples/shader/$name.comp"
    EXAMPLE_SPV="$PROJECT_ROOT/examples/shader/$name.spv"
    echo "Compiling $name.comp → $name.spv"
    glslc -x glsl --target-env=vulkan1.1 -O "$EXAMPLE_SRC" -o "$EXAMPLE_SPV"
    FILE_SIZE=$(stat -c%s "$EXAMPLE_SPV" 2>/dev/null || stat -f%z "$EXAMPLE_SPV" 2>/dev/null)
    echo "Written $FILE_SIZE bytes to $EXAMPLE_SPV"
done
