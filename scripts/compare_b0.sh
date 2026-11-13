#!/usr/bin/env bash
# Checks that the encoder output without B-frames did not change.
#
# Builds a base revision (default: the v0.11.0 release commit) in a git
# worktree, encodes the same generated frames with it and with the working
# tree, and compares the streams with cmp. The bytes depend on the device and
# the driver, so both builds run here, on the same device, instead of comparing
# against a stored hash.
#
# Usage: scripts/compare_b0.sh [base revision]
set -euo pipefail

base="${1:-b4e2d1f}"
root="$(git rev-parse --show-toplevel)"
work="$(mktemp -d)"
example="b0_compare"
trap 'rm -f "$root/examples/$example.rs"; git -C "$root" worktree remove --force "$work/base" 2>/dev/null || true; rm -rf "$work"' EXIT

git -C "$root" worktree add --detach "$work/base" "$base" >/dev/null
cp "$root/scripts/$example.rs" "$work/base/examples/$example.rs"
cp "$root/scripts/$example.rs" "$root/examples/$example.rs"

(cd "$work/base" && cargo build --quiet --example "$example" --target-dir "$root/target/b0-base")
(cd "$root" && cargo build --quiet --example "$example")

failed=0
for codec in h264 h265 av1; do
    for rate in cqp cbr; do
        # A GOP shorter than the sequence, and none at all: the second one
        # runs past the wrap of the picture order count.
        for gop in 30 0; do
            name="$codec-$rate-gop$gop"
            (cd "$work/base" && "$root/target/b0-base/debug/examples/$example" \
                "$codec" "$rate" "$gop" 600 "$work/$name.base") || { echo "SKIP $name (base failed)"; continue; }
            (cd "$root" && "$root/target/debug/examples/$example" \
                "$codec" "$rate" "$gop" 600 "$work/$name.new")
            if cmp -s "$work/$name.base" "$work/$name.new"; then
                echo "same $name ($(stat -c %s "$work/$name.new") bytes)"
            else
                echo "DIFFERENT $name"
                failed=1
            fi
        done
    done
done
exit "$failed"
