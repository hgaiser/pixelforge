//! OBU framing (AV1 specification sections 5.2 and 5.3).
//!
//! The parser takes the low-overhead bitstream format of section 5.2, the
//! format of `.obu` files, of MP4/Matroska samples and of RTP after
//! depacketisation: a plain sequence of OBUs, each carrying its own size.
//! Annex B length-delimited streams wrap every temporal unit, frame unit and
//! OBU in a further length; they are detected where the data makes it
//! possible and rejected with an error that says so.

use super::bits::BitReader;
use crate::error::{PixelForgeError, Result};

/// OBU types (section 6.2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObuType {
    SequenceHeader,
    TemporalDelimiter,
    FrameHeader,
    TileGroup,
    Metadata,
    Frame,
    RedundantFrameHeader,
    TileList,
    Padding,
    /// A reserved type, which decoders must ignore.
    Reserved(u8),
}

impl ObuType {
    fn from_raw(raw: u8) -> Self {
        match raw {
            1 => ObuType::SequenceHeader,
            2 => ObuType::TemporalDelimiter,
            3 => ObuType::FrameHeader,
            4 => ObuType::TileGroup,
            5 => ObuType::Metadata,
            6 => ObuType::Frame,
            7 => ObuType::RedundantFrameHeader,
            8 => ObuType::TileList,
            15 => ObuType::Padding,
            other => ObuType::Reserved(other),
        }
    }
}

/// `obu_header()` (section 5.3.2), plus whether the size field was present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObuHeader {
    pub obu_type: ObuType,
    pub obu_extension_flag: bool,
    pub obu_has_size_field: bool,
    /// Zero when there is no extension header.
    pub temporal_id: u8,
    /// Zero when there is no extension header.
    pub spatial_id: u8,
}

/// One OBU within a buffer.
#[derive(Debug, Clone, Copy)]
pub struct Obu<'a> {
    pub header: ObuHeader,
    /// Byte offset of the first byte of `obu_header()` within the data the
    /// iterator was given.
    pub offset: usize,
    /// Bytes from `offset` to the payload: the header, the optional
    /// extension byte and the optional `obu_size` field.
    pub header_len: usize,
    /// The payload (`obu_size` bytes), without the header.
    pub payload: &'a [u8],
}

impl Obu<'_> {
    /// Byte offset of the payload within the data the iterator was given.
    pub fn payload_offset(&self) -> usize {
        self.offset + self.header_len
    }
}

/// Whether an OBU belongs to the operating point with the given
/// `operating_point_idc` (section 6.2.1, the `drop_obu()` rule).
///
/// OBUs without an extension header, sequence headers and temporal
/// delimiters always belong, as does everything when `idc` is zero.
pub fn in_operating_point(header: &ObuHeader, idc: u16) -> bool {
    if idc == 0
        || !header.obu_extension_flag
        || matches!(
            header.obu_type,
            ObuType::SequenceHeader | ObuType::TemporalDelimiter
        )
    {
        return true;
    }
    let in_temporal = (idc >> header.temporal_id) & 1 != 0;
    let in_spatial = (idc >> (header.spatial_id + 8)) & 1 != 0;
    in_temporal && in_spatial
}

/// Read one OBU from the start of `data`.
///
/// `Ok(None)` means the data ends before the OBU does, which a caller
/// framing a byte stream treats as "wait for more".
fn read_obu(data: &[u8]) -> Result<Option<(ObuHeader, usize, usize)>> {
    let Some(&first) = data.first() else {
        return Ok(None);
    };
    if first & 0x80 != 0 {
        return Err(PixelForgeError::InvalidInput(
            "AV1: obu_forbidden_bit is set".to_string(),
        ));
    }
    let obu_extension_flag = first & 0x04 != 0;
    let obu_has_size_field = first & 0x02 != 0;
    let mut header = ObuHeader {
        obu_type: ObuType::from_raw((first >> 3) & 0x0f),
        obu_extension_flag,
        obu_has_size_field,
        temporal_id: 0,
        spatial_id: 0,
    };
    let mut header_len = 1;
    if obu_extension_flag {
        let Some(&ext) = data.get(1) else {
            return Ok(None);
        };
        header.temporal_id = ext >> 5;
        header.spatial_id = (ext >> 3) & 0x03;
        header_len = 2;
    }
    let payload_len = if obu_has_size_field {
        let mut r = BitReader::new(&data[header_len..]);
        let size = match r.leb128() {
            Ok(size) => size,
            // A leb128 cut short by the end of the data is incomplete, not
            // invalid; one longer than eight bytes is invalid.
            Err(_) if r.bits_left() == 0 => return Ok(None),
            Err(e) => return Err(e),
        };
        header_len += r.byte_position();
        usize::try_from(size).map_err(|_| {
            PixelForgeError::InvalidInput("AV1: obu_size does not fit in memory".to_string())
        })?
    } else {
        // Without a size field the OBU runs to the end of the data, which is
        // how a container may store the last OBU of a sample.
        data.len() - header_len
    };
    if data.len() - header_len < payload_len {
        return Ok(None);
    }
    Ok(Some((header, header_len, payload_len)))
}

/// Whether `data` parses as the start of an Annex B length-delimited stream
/// (`temporal_unit_size`, `frame_unit_size`, `obu_length`, then a temporal
/// delimiter), which low-overhead data never does: there, the first OBU is a
/// temporal delimiter `0x12 0x00`, which reads as a frame unit of size zero.
pub fn looks_like_annex_b(data: &[u8]) -> bool {
    let mut r = BitReader::new(data);
    let (Ok(tu_size), Ok(fu_size)) = (r.leb128(), r.leb128()) else {
        return false;
    };
    let Ok(obu_length) = r.leb128() else {
        return false;
    };
    let obu_start = r.byte_position();
    let Some(&first) = data.get(obu_start) else {
        return false;
    };
    // Sizes must nest: the OBU inside the frame unit inside the temporal
    // unit, each also holding the size field of the one inside it.
    if fu_size == 0 || fu_size >= tu_size || obu_length == 0 || obu_length >= fu_size {
        return false;
    }
    let obu_type = (first >> 3) & 0x0f;
    let has_ext = first & 0x04 != 0;
    // A temporal delimiter has an empty payload, so the whole OBU is its
    // header (plus a zero obu_size, if the encoder wrote one).
    first & 0x80 == 0 && obu_type == 2 && obu_length <= 1 + has_ext as u64 + 1
}

/// Iterator over the OBUs of a low-overhead bitstream.
///
/// Yields an error, then stops, on invalid data or on an OBU that runs past
/// the end of the buffer.
pub struct ObuIter<'a> {
    data: &'a [u8],
    pos: usize,
    done: bool,
}

/// Iterate over the OBUs in `data` (low-overhead format, section 5.2).
pub fn iter_obus(data: &[u8]) -> ObuIter<'_> {
    ObuIter {
        data,
        pos: 0,
        done: false,
    }
}

fn annex_b_error() -> PixelForgeError {
    PixelForgeError::InvalidInput(
        "AV1: input looks like an Annex B length-delimited stream; only the \
         low-overhead OBU format (section 5.2) is supported"
            .to_string(),
    )
}

impl<'a> Iterator for ObuIter<'a> {
    type Item = Result<Obu<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.pos >= self.data.len() {
            return None;
        }
        let rest = &self.data[self.pos..];
        // Annex B data starts with a size, not an OBU header. The checks
        // that would catch it, a forbidden bit or a missing size field, only
        // fire on some first bytes, so look at the structure directly.
        if self.pos == 0 && looks_like_annex_b(rest) {
            self.done = true;
            return Some(Err(annex_b_error()));
        }
        match read_obu(rest) {
            Ok(Some((header, header_len, payload_len))) => {
                let obu = Obu {
                    header,
                    offset: self.pos,
                    header_len,
                    payload: &rest[header_len..header_len + payload_len],
                };
                self.pos += header_len + payload_len;
                Some(Ok(obu))
            }
            Ok(None) => {
                self.done = true;
                Some(Err(PixelForgeError::InvalidInput(format!(
                    "AV1: OBU at byte {} runs past the end of the data",
                    self.pos
                ))))
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

/// Byte offsets at which each temporal unit begins in a low-overhead
/// byte stream: the offset of every temporal delimiter OBU.
///
/// This is what lets a byte-stream feed decide where it may cut: everything
/// before the last offset is complete, and everything from it on may still be
/// waiting for OBUs that have not arrived. An OBU cut short by the end of the
/// data ends the scan without an error for the same reason. Invalid data, or
/// data in Annex B format, is an error.
pub fn temporal_unit_starts(data: &[u8]) -> Result<Vec<usize>> {
    if looks_like_annex_b(data) {
        return Err(annex_b_error());
    }
    let mut starts = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let rest = &data[pos..];
        // Without a size field an OBU would run to the end of whatever has
        // arrived so far, which says nothing about where it really ends.
        if rest[0] & 0x02 == 0 {
            return Err(PixelForgeError::InvalidInput(format!(
                "AV1: OBU at byte {pos} has no obu_size, which a byte stream needs"
            )));
        }
        let Some((header, header_len, payload_len)) = read_obu(rest)? else {
            break;
        };
        if header.obu_type == ObuType::TemporalDelimiter {
            starts.push(pos);
        }
        pos += header_len + payload_len;
    }
    Ok(starts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_headers_and_sizes() {
        // TD, then a padding OBU with an extension header and 2 bytes.
        let data = [0x12, 0x00, 0x7e, 0b0100_1000, 0x02, 0xaa, 0xbb];
        let obus: Vec<_> = iter_obus(&data).collect::<Result<_>>().unwrap();
        assert_eq!(obus.len(), 2);
        assert_eq!(obus[0].header.obu_type, ObuType::TemporalDelimiter);
        assert_eq!(obus[0].payload.len(), 0);
        let pad = &obus[1];
        assert_eq!(pad.header.obu_type, ObuType::Padding);
        assert!(pad.header.obu_extension_flag);
        assert_eq!(pad.header.temporal_id, 2);
        assert_eq!(pad.header.spatial_id, 1);
        assert_eq!(pad.offset, 2);
        assert_eq!(pad.header_len, 3);
        assert_eq!(pad.payload, &[0xaa, 0xbb]);
    }

    #[test]
    fn reports_truncated_and_forbidden() {
        let truncated = [0x12, 0x00, 0x32, 0x05, 0x00];
        let results: Vec<_> = iter_obus(&truncated).collect();
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
        assert_eq!(results.len(), 2);

        let forbidden = [0x92, 0x00];
        assert!(iter_obus(&forbidden).next().unwrap().is_err());
    }

    #[test]
    fn finds_temporal_units_and_tolerates_a_cut_tail() {
        // TD, padding(1), TD, padding cut short.
        let data = [0x12, 0x00, 0x7a, 0x01, 0x00, 0x12, 0x00, 0x7a, 0x04, 0x00];
        assert_eq!(temporal_unit_starts(&data).unwrap(), vec![0, 5]);
    }

    #[test]
    fn detects_annex_b() {
        // temporal_unit_size 3, frame_unit_size 2, obu_length 1, a TD header
        // without a size field, then the next temporal unit.
        let data = [0x03, 0x02, 0x01, 0x10, 0x03];
        assert!(looks_like_annex_b(&data));
        assert!(iter_obus(&data).next().unwrap().is_err());
        assert!(temporal_unit_starts(&data).is_err());
        // A low-overhead stream must not trip the check.
        assert!(!looks_like_annex_b(&[0x12, 0x00, 0x0a, 0x0b]));
    }

    #[test]
    fn drops_obus_outside_the_operating_point() {
        let mut header = ObuHeader {
            obu_type: ObuType::Frame,
            obu_extension_flag: true,
            obu_has_size_field: true,
            temporal_id: 1,
            spatial_id: 0,
        };
        // Temporal layer 0 and spatial layer 0 only.
        let idc = 0x101;
        assert!(!in_operating_point(&header, idc));
        header.temporal_id = 0;
        assert!(in_operating_point(&header, idc));
        header.spatial_id = 1;
        assert!(!in_operating_point(&header, idc));
        header.obu_type = ObuType::SequenceHeader;
        assert!(in_operating_point(&header, idc));
        assert!(in_operating_point(&header, 0));
    }
}
