//! Host-side AV1 parsing for Vulkan Video decode (`VK_KHR_video_decode_av1`).
//!
//! Vulkan decodes AV1 tiles in the driver but takes every header value from
//! the application, already parsed: the sequence header, the frame header
//! with everything it inherits from its references, and where each tile is.
//! This module produces exactly that from a low-overhead OBU stream
//! (section 5.2 of the AV1 specification), and keeps the reference state
//! (the eight reference slots) that later headers depend on.
//!
//! - [`obu`]: OBU framing, temporal unit boundaries, operating point
//!   selection, Annex B detection.
//! - [`sequence`]: the sequence header.
//! - [`frame_header`] and [`params`]: the uncompressed frame header.
//! - [`tile`]: the tile grid and tile group OBUs.
//! - [`refs`]: the reference slots and their update.
//! - [`parser`]: [`Av1Parser`], which ties them together frame by frame.
//! - [`std_info`]: conversion to the Vulkan `StdVideoAV1*` structures.
//!
//! Typical use: iterate the OBUs of a temporal unit with
//! [`obu::iter_obus`], pass each to [`Av1Parser::parse_obu`], and for each
//! [`Av1Event::Frame`] decode it, then call [`Av1Parser::commit_frame`]
//! with the DPB slot the picture went to.

pub(crate) mod bits;
pub(crate) mod consts;
pub(crate) mod frame_header;
pub(crate) mod obu;
pub(crate) mod params;
pub(crate) mod parser;
pub(crate) mod refs;
pub(crate) mod sequence;
pub(crate) mod std_info;
pub(crate) mod tile;

mod record;
mod session;

/// Parser tests against real libaom and SVT-AV1 streams (host-side, no GPU).
#[cfg(test)]
mod stream_tests;

pub(crate) use session::Av1Decoder;
