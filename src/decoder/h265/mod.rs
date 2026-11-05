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

/// Parser and DPB tests against real x265 and NVENC streams (host-side, no
/// GPU), compiled only under `cargo test`.
#[cfg(test)]
mod parser_tests;

/// Hand-written slice header syntax the fixtures do not cover.
#[cfg(test)]
mod syntax_tests;
