//! The OBU-level decoding process: which OBUs make up a frame, and when the
//! reference state moves on (AV1 specification sections 7.4, 7.5 and 7.20).

use super::bits::BitReader;
use super::consts::TOTAL_REFS_PER_FRAME;
use super::frame_header::{FrameHeader, FrameType, HeaderOutcome, parse_uncompressed_header};
use super::obu::{Obu, ObuType, in_operating_point};
use super::refs::RefState;
use super::sequence::{SequenceHeader, parse_sequence_header};
use super::tile::{TileGroup, TileLocation, parse_tile_group};
use crate::error::{PixelForgeError, Result};

/// One frame as the OBUs describe it: the parsed header and where its data
/// sits.
///
/// All offsets are relative to the start of the buffer the OBUs were
/// iterated from (see [`Obu::offset`]). Handing the driver that whole buffer
/// makes them usable as they are; handing it a sub-range means subtracting
/// the sub-range's start from each.
#[derive(Debug, Clone)]
pub struct Av1Frame {
    pub header: FrameHeader,
    /// The OBU that carried the header: `Frame`, `FrameHeader` or
    /// `RedundantFrameHeader` (when the original was lost).
    ///
    /// This and the two fields below only check the parse in tests: the
    /// decoder hands the driver tile data alone.
    #[cfg_attr(not(test), allow(dead_code))]
    pub header_obu_type: ObuType,
    /// Offset of the first byte of that OBU (its `obu_header()`).
    #[cfg_attr(not(test), allow(dead_code))]
    pub frame_header_offset: usize,
    /// Bytes of payload the header takes: `uncompressed_header()` plus its
    /// byte alignment in a frame OBU, plus its trailing bits in a frame
    /// header OBU.
    #[cfg_attr(not(test), allow(dead_code))]
    pub frame_header_bytes: usize,
    /// The tile groups in order, which together cover every tile. Empty for
    /// `show_existing_frame`.
    pub tile_groups: Vec<TileGroup>,
}

impl Av1Frame {
    /// Every tile of the frame, in tile order.
    pub fn tiles(&self) -> impl Iterator<Item = TileLocation> + '_ {
        self.tile_groups
            .iter()
            .flat_map(|tg| tg.tiles.iter().copied())
    }
}

/// What one OBU produced.
#[derive(Debug)]
pub enum Av1Event {
    /// A sequence header was parsed. Encoders repeat it at every key frame;
    /// the decoder compares it with the one its session was built from.
    SequenceHeader,
    /// A frame is complete: all of its tile groups have arrived, or it is a
    /// `show_existing_frame`. The caller decodes (or shows) it and then calls
    /// [`Av1Parser::commit_frame`] or [`Av1Parser::discard_frame`].
    Frame(Box<Av1Frame>),
    /// The next frame refers to reference slots that hold nothing (bit `i`
    /// for slot `i`), as after joining a stream mid-way or losing a frame.
    /// It cannot be decoded; its remaining OBUs are skipped, and the
    /// decoder needs a key frame.
    MissingReferences(u8),
}

/// A frame whose header has arrived but not all of its tiles.
#[derive(Debug)]
struct InProgress {
    frame: Av1Frame,
    refs: Box<RefState>,
    next_tile: usize,
}

/// A complete frame waiting for the caller to commit or discard it.
#[derive(Debug)]
struct Pending {
    header: FrameHeader,
    refs: Box<RefState>,
}

/// Stateful AV1 parser: feed it OBUs in order, get frames out.
#[derive(Debug, Default)]
pub struct Av1Parser {
    sequence: Option<SequenceHeader>,
    refs: RefState,
    /// `OrderHints` as the last committed frame left it.
    order_hints: [u8; TOTAL_REFS_PER_FRAME],
    current: Option<InProgress>,
    pending: Option<Pending>,
    /// Set after `MissingReferences`, until the next frame header.
    skipping: bool,
}

impl Av1Parser {
    pub fn new() -> Self {
        Self::default()
    }

    /// The sequence header in effect, once one has been seen.
    pub fn sequence_header(&self) -> Option<&SequenceHeader> {
        self.sequence.as_ref()
    }

    /// The reference slots as of the last committed frame. While a frame is
    /// pending, this is the state its header was parsed against, which is
    /// what its references must be looked up in.
    pub fn refs(&self) -> &RefState {
        &self.refs
    }

    /// Apply the reference frame update process for the pending frame, now
    /// decoded into DPB slot `dpb_slot` (`None` if the caller does not track
    /// it). For `show_existing_frame` of a key frame the shown slot's DPB
    /// slot is used instead of `dpb_slot`.
    pub fn commit_frame(&mut self, dpb_slot: Option<u8>) -> Result<()> {
        let pending = self.pending.take().ok_or_else(|| {
            PixelForgeError::InvalidInput("AV1: no frame waiting to be committed".to_string())
        })?;
        let Some(seq) = &self.sequence else {
            return Err(PixelForgeError::InvalidInput(
                "AV1: frame without a sequence header".to_string(),
            ));
        };
        let header = &pending.header;
        let mut refs = pending.refs;
        refs.update(seq, header, dpb_slot);
        self.refs = *refs;
        if !header.show_existing_frame || header.frame_type == FrameType::Key {
            self.order_hints = header.order_hints;
        }
        Ok(())
    }

    /// Drop the pending frame, as when it failed to decode. The slots it
    /// would have refreshed no longer hold what the stream expects there,
    /// so they become invalid, and frames that refer to them report missing
    /// references until a key frame refreshes them.
    pub fn discard_frame(&mut self) {
        if let Some(pending) = self.pending.take() {
            for (i, slot) in self.refs.slots.iter_mut().enumerate() {
                if (pending.header.refresh_frame_flags >> i) & 1 != 0 {
                    slot.valid = false;
                }
            }
        }
    }

    /// Process one OBU.
    pub fn parse_obu(&mut self, obu: &Obu) -> Result<Option<Av1Event>> {
        match obu.header.obu_type {
            ObuType::TemporalDelimiter => {
                // SeenFrameHeader = 0. A frame still missing tiles here is
                // broken; say so, but leave the parser ready for the next one.
                self.skipping = false;
                if self.current.take().is_some() {
                    return Err(PixelForgeError::InvalidInput(
                        "AV1: temporal unit ended before the frame's last tile group".to_string(),
                    ));
                }
                Ok(None)
            }
            ObuType::SequenceHeader => {
                let seq = parse_sequence_header(obu.payload)?;
                self.sequence = Some(seq);
                Ok(Some(Av1Event::SequenceHeader))
            }
            ObuType::FrameHeader | ObuType::RedundantFrameHeader | ObuType::Frame => {
                if !self.selected(obu)? {
                    return Ok(None);
                }
                self.frame_header(obu)
            }
            ObuType::TileGroup => {
                if !self.selected(obu)? || self.skipping {
                    return Ok(None);
                }
                let Some(current) = self.current.as_mut() else {
                    return Err(PixelForgeError::InvalidInput(
                        "AV1: tile group without a frame header".to_string(),
                    ));
                };
                let tg = parse_tile_group(
                    obu.payload,
                    obu.payload_offset(),
                    &current.frame.header.tile_info,
                )?;
                self.add_tile_group(tg)
            }
            ObuType::Metadata | ObuType::TileList | ObuType::Padding | ObuType::Reserved(_) => {
                Ok(None)
            }
        }
    }

    /// Whether the OBU is in operating point 0; frame data before any
    /// sequence header is an error.
    fn selected(&self, obu: &Obu) -> Result<bool> {
        let seq = self.sequence.as_ref().ok_or_else(|| {
            PixelForgeError::InvalidInput("AV1: frame data before a sequence header".to_string())
        })?;
        Ok(in_operating_point(&obu.header, seq.operating_point_idc()))
    }

    fn frame_header(&mut self, obu: &Obu) -> Result<Option<Av1Event>> {
        let is_frame_obu = obu.header.obu_type == ObuType::Frame;
        if self.current.is_some() {
            if is_frame_obu {
                self.current = None;
                return Err(PixelForgeError::InvalidInput(
                    "AV1: frame OBU before the previous frame's last tile group".to_string(),
                ));
            }
            // frame_header_copy(): a repeat of the header in effect.
            return Ok(None);
        }
        if self.skipping && obu.header.obu_type == ObuType::RedundantFrameHeader {
            return Ok(None);
        }
        if self.pending.is_some() {
            return Err(PixelForgeError::InvalidInput(
                "AV1: new frame before the previous one was committed or discarded".to_string(),
            ));
        }
        let seq = self.sequence.as_ref().expect("checked by selected()");

        let mut r = BitReader::new(obu.payload);
        let outcome =
            parse_uncompressed_header(&mut r, seq, &self.refs, &obu.header, self.order_hints)?;
        let (header, refs) = match outcome {
            HeaderOutcome::Parsed(header, refs) => (header, refs),
            HeaderOutcome::MissingReferences(slots) => {
                self.skipping = true;
                return Ok(Some(Av1Event::MissingReferences(slots)));
            }
        };
        self.skipping = false;

        if is_frame_obu {
            if header.show_existing_frame {
                return Err(PixelForgeError::InvalidInput(
                    "AV1: frame OBU with show_existing_frame".to_string(),
                ));
            }
            r.byte_alignment()?;
        } else {
            r.trailing_bits(obu.payload.len())?;
        }
        let header_bytes = r.byte_position();

        let frame = Av1Frame {
            header: *header,
            header_obu_type: obu.header.obu_type,
            frame_header_offset: obu.offset,
            frame_header_bytes: header_bytes,
            tile_groups: Vec::new(),
        };
        if frame.header.show_existing_frame {
            self.pending = Some(Pending {
                header: frame.header.clone(),
                refs,
            });
            return Ok(Some(Av1Event::Frame(Box::new(frame))));
        }
        self.current = Some(InProgress {
            frame,
            refs,
            next_tile: 0,
        });
        if !is_frame_obu {
            return Ok(None);
        }
        let current = self.current.as_ref().expect("just set");
        let tg = parse_tile_group(
            &obu.payload[header_bytes..],
            obu.payload_offset() + header_bytes,
            &current.frame.header.tile_info,
        );
        match tg {
            Ok(tg) => self.add_tile_group(tg),
            Err(e) => {
                self.current = None;
                Err(e)
            }
        }
    }

    fn add_tile_group(&mut self, tg: TileGroup) -> Result<Option<Av1Event>> {
        let current = self.current.as_mut().expect("caller checked");
        if tg.tg_start as usize != current.next_tile {
            let expected = current.next_tile;
            self.current = None;
            return Err(PixelForgeError::InvalidInput(format!(
                "AV1: tile group starts at tile {}, expected {expected}",
                tg.tg_start
            )));
        }
        current.next_tile = tg.tg_end as usize + 1;
        current.frame.tile_groups.push(tg);
        if current.next_tile < current.frame.header.tile_info.num_tiles() {
            return Ok(None);
        }
        let done = self.current.take().expect("checked above");
        self.pending = Some(Pending {
            header: done.frame.header.clone(),
            refs: done.refs,
        });
        Ok(Some(Av1Event::Frame(Box::new(done.frame))))
    }
}
