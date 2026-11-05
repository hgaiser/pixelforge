//! H.265 decoding: the host-side syntax parsing and reference bookkeeping.
//!
//! - [`parser`]: NAL units, the shared bit reader, `profile_tier_level()`,
//!   HRD skipping and the VPS.
//! - [`sps`], [`pps`]: the parameter sets, with the scaling lists ([`scaling`])
//!   and short-term reference picture sets ([`rps`]) they contain.
//! - [`slice`]: the slice segment header, up to the reference syntax.
//! - [`dpb`]: picture order count, the reference picture set, random access
//!   and DPB slots.
//!
//! None of this touches Vulkan; the session code fills the `StdVideoH265*`
//! structures from these types, whose field names follow the standard.

pub(crate) mod dpb;
pub(crate) mod parser;
pub(crate) mod pps;
pub(crate) mod rps;
pub(crate) mod scaling;
pub(crate) mod slice;
pub(crate) mod sps;

mod record;
mod session;
mod std_params;

pub(crate) use session::H265Decoder;

/// Parser and DPB tests against real x265 and NVENC streams (host-side, no
/// GPU), compiled only under `cargo test`.
#[cfg(test)]
mod parser_tests;

/// Hand-written slice header syntax the fixtures do not cover.
#[cfg(test)]
mod syntax_tests;

/// Byte offsets at which each coded picture begins in an Annex B stream.
///
/// A new picture begins at each VCL NAL unit with
/// `first_slice_segment_in_pic_flag` set. Parameter sets, SEI, access unit
/// delimiters and end of sequence NAL units are attached to the picture that
/// follows them, so the offsets carve the stream up without loss.
pub(crate) fn picture_starts(stream: &[u8]) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut pending_start: Option<usize> = None;
    for (offset, nal) in parser::iter_nal_units_with_offsets(stream) {
        if nal.nal_type().is_vcl() {
            if slice::starts_picture(&nal) {
                starts.push(pending_start.take().unwrap_or(offset));
            }
            pending_start = None;
        } else if pending_start.is_none() {
            pending_start = Some(offset);
        }
    }
    starts
}

/// How many leading bytes of `buffer` hold pictures known to be complete.
pub(crate) fn complete_prefix(buffer: &[u8]) -> usize {
    picture_starts(buffer).last().copied().unwrap_or(0)
}
