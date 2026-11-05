//! Annex B byte streams, as H.264 and H.265 both use them: NAL units separated
//! by 3- or 4-byte start codes.
//!
//! Only the splitting lives here. The NAL header differs between the two
//! codecs (one byte in H.264, two in H.265), so each codec parses its own.

/// Annex B start code prefixed to each slice handed to the driver.
///
/// Three bytes, not four. The leading zero byte of a 4-byte start code is legal
/// Annex B, and RADV decodes either form, but Intel's ANV decodes almost nothing
/// when a slice offset points at that extra zero: it reports no error and
/// returns a picture with a handful of macroblocks in it. ffmpeg's Vulkan
/// decoder uses three bytes for the same reason.
pub(crate) const START_CODE: [u8; 3] = [0, 0, 1];

/// Iterate over the NAL units in an Annex B byte stream.
///
/// Yields each NAL's start-code offset within `data` together with the NAL
/// unit itself (header and EBSP payload, without the start code). Trailing
/// zero bytes before the next start code are trimmed, and empty NAL units are
/// skipped.
pub(crate) fn iter_nal_units(data: &[u8]) -> impl Iterator<Item = (usize, &[u8])> {
    NalIterator { data, pos: 0 }
}

struct NalIterator<'a> {
    data: &'a [u8],
    pos: usize,
}

/// Find the next start code at or after `from`. Returns (start_code_pos, payload_pos).
fn find_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut i = from;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1 {
                return Some((i, i + 3));
            }
            if i + 4 <= data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                return Some((i, i + 4));
            }
        }
        i += 1;
    }
    None
}

impl<'a> Iterator for NalIterator<'a> {
    type Item = (usize, &'a [u8]);

    fn next(&mut self) -> Option<(usize, &'a [u8])> {
        loop {
            let (start_code_pos, payload_start) = find_start_code(self.data, self.pos)?;
            let end = match find_start_code(self.data, payload_start) {
                Some((next_start, _)) => next_start,
                None => self.data.len(),
            };
            self.pos = end;

            // Trim trailing zero padding before the next start code.
            let mut trimmed_end = end;
            while trimmed_end > payload_start && self.data[trimmed_end - 1] == 0 {
                trimmed_end -= 1;
            }
            if trimmed_end > payload_start {
                return Some((start_code_pos, &self.data[payload_start..trimmed_end]));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_start_codes_and_trims_zero_padding() {
        let stream = [
            0, 0, 0, 1, 0x67, 0xAA, 0, 0, 1, 0x68, 0xBB, 0, 0, 0, 0, 1, 0x65,
        ];
        let nals: Vec<_> = iter_nal_units(&stream).collect();
        assert_eq!(
            nals,
            vec![
                (0, &[0x67, 0xAA][..]),
                (6, &[0x68, 0xBB][..]),
                (12, &[0x65][..]),
            ]
        );
    }

    #[test]
    fn skips_empty_nal_units() {
        let stream = [0, 0, 1, 0, 0, 1, 0x41];
        let nals: Vec<_> = iter_nal_units(&stream).collect();
        assert_eq!(nals, vec![(3, &[0x41][..])]);
    }
}
