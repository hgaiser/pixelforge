//! H.264 decoding on top of Vulkan Video.
//!
//! parser does the host-side syntax work; this module owns the Vulkan
//! session, the decoded picture buffer and the per-picture record/submit flow.

pub(crate) mod parser;

/// Parser tests against real x264 streams (host-side, no GPU). In-crate so they
/// reach parser internals directly, compiled only under `cargo test`.
#[cfg(test)]
mod parser_tests;

mod dpb;
mod record;
mod session;

pub(crate) use session::H264Decoder;

/// Byte offsets at which each coded picture begins in an Annex B stream.
///
/// A new picture begins at each VCL NAL whose `first_mb_in_slice` is 0.
/// Parameter sets, SEI and access unit delimiters are attached to the picture
/// that follows them, so the offsets carve the stream up without loss.
///
/// This is what tells a byte-stream feed where it may cut: everything before
/// the last offset is complete, everything from it onward may still be waiting
/// for slices that have not arrived.
pub(crate) fn picture_starts(stream: &[u8]) -> Vec<usize> {
    let mut starts: Vec<usize> = Vec::new();
    let mut pending_start: Option<usize> = None;

    for (offset, nal) in parser::iter_nal_units_with_offsets(stream) {
        if nal.nal_type.is_slice() {
            // first_mb_in_slice is the leading ue(v) of the slice header; it is
            // zero -- encoded as a leading `1` bit -- exactly at a picture start.
            let first_mb_is_zero = nal.payload().first().is_some_and(|b| b & 0x80 != 0);
            if first_mb_is_zero {
                starts.push(pending_start.take().unwrap_or(offset));
            }
            pending_start = None;
        } else if pending_start.is_none() {
            // Parameter sets / SEI / AUD belong to the picture they precede.
            pending_start = Some(offset);
        }
    }
    starts
}

/// How many leading bytes of `buffer` hold pictures known to be complete.
///
/// A coded picture is only known to have ended once the next one starts, so
/// everything from the last picture start onward may still be waiting for
/// slices. Zero means nothing in `buffer` can be decoded yet.
pub(crate) fn complete_prefix(buffer: &[u8]) -> usize {
    picture_starts(buffer).last().copied().unwrap_or(0)
}
