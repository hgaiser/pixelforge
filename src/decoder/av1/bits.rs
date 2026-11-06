//! Bit reader for AV1 syntax (AV1 specification section 4.10 and 8.1).
//!
//! AV1 has no emulation prevention, so unlike the H.26x reader in
//! `crate::decoder::bitreader` this reads the bytes exactly as given. All
//! reads are bounds-checked and fail with `InvalidInput` past the end, so a
//! truncated or corrupt OBU can never panic the parser.

use crate::error::{PixelForgeError, Result};

/// MSB-first bit reader over one OBU payload (or any byte slice).
#[derive(Debug, Clone)]
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Position in bits from the start of `data`.
    pos: usize,
}

fn truncated() -> PixelForgeError {
    PixelForgeError::InvalidInput("AV1: bitstream ended inside a syntax element".to_string())
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Current position in bits (the spec's `get_position()`).
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Current position in whole bytes, rounded up.
    pub fn byte_position(&self) -> usize {
        self.pos.div_ceil(8)
    }

    /// Bits left before the end of the data.
    pub fn bits_left(&self) -> usize {
        (self.data.len() * 8).saturating_sub(self.pos)
    }

    /// One bit as a flag.
    pub fn flag(&mut self) -> Result<bool> {
        Ok(self.f(1)? != 0)
    }

    /// `f(n)`: an unsigned n-bit number, MSB first, for n up to 32.
    pub fn f(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        if n == 0 {
            return Ok(0);
        }
        if self.bits_left() < n as usize {
            return Err(truncated());
        }
        let mut value: u64 = 0;
        for _ in 0..n {
            let byte = self.data[self.pos >> 3];
            let bit = (byte >> (7 - (self.pos & 7))) & 1;
            value = (value << 1) | bit as u64;
            self.pos += 1;
        }
        Ok(value as u32)
    }

    /// `su(n)`: an n-bit two's complement signed number (4.10.6).
    pub fn su(&mut self, n: u32) -> Result<i32> {
        let value = self.f(n)? as i64;
        let sign_mask = 1i64 << (n - 1);
        let value = if value & sign_mask != 0 {
            value - 2 * sign_mask
        } else {
            value
        };
        Ok(value as i32)
    }

    /// `ns(n)`: a non-symmetric unsigned value in `0..n` (4.10.7).
    pub fn ns(&mut self, n: u32) -> Result<u32> {
        if n <= 1 {
            // A single possible value takes no bits.
            return Ok(0);
        }
        let w = 32 - n.leading_zeros(); // FloorLog2(n) + 1
        let m = (1u32 << w) - n;
        let v = self.f(w - 1)?;
        if v < m {
            return Ok(v);
        }
        let extra_bit = self.f(1)?;
        Ok((v << 1) - m + extra_bit)
    }

    /// `le(n)`: an n-byte little-endian unsigned number (4.10.4).
    pub fn le(&mut self, n: u32) -> Result<u32> {
        let mut t: u64 = 0;
        for i in 0..n {
            t += (self.f(8)? as u64) << (i * 8);
        }
        u32::try_from(t)
            .map_err(|_| PixelForgeError::InvalidInput("AV1: le(n) value out of range".to_string()))
    }

    /// `leb128()`: an unsigned little-endian base-128 value (4.10.5).
    pub fn leb128(&mut self) -> Result<u64> {
        let mut value: u64 = 0;
        for i in 0..8 {
            let byte = self.f(8)? as u64;
            value |= (byte & 0x7f) << (i * 7);
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(PixelForgeError::InvalidInput(
            "AV1: leb128 value longer than 8 bytes".to_string(),
        ))
    }

    /// `uvlc()`: a variable-length unsigned value (4.10.3).
    pub fn uvlc(&mut self) -> Result<u32> {
        let mut leading_zeros = 0u32;
        while !self.flag()? {
            leading_zeros += 1;
            if leading_zeros >= 32 {
                // The spec defines this as (1 << 32) - 1; reading on would
                // only consume bits that carry no information.
                return Ok(u32::MAX);
            }
        }
        let value = self.f(leading_zeros)?;
        Ok(value + ((1u64 << leading_zeros) - 1) as u32)
    }

    /// `byte_alignment()`: skip to the next byte boundary. The skipped bits
    /// must be zero (5.3.5).
    pub fn byte_alignment(&mut self) -> Result<()> {
        while self.pos & 7 != 0 {
            if self.flag()? {
                return Err(PixelForgeError::InvalidInput(
                    "AV1: non-zero bit in byte_alignment()".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// `trailing_bits()` (5.3.4): check that the bits from here to the end of
    /// a payload of `payload_bytes` bytes are a one followed by zeros.
    pub fn trailing_bits(&mut self, payload_bytes: usize) -> Result<()> {
        let end = payload_bytes * 8;
        if self.pos >= end || end > self.data.len() * 8 {
            return Err(PixelForgeError::InvalidInput(
                "AV1: OBU payload has no room for its trailing bits".to_string(),
            ));
        }
        if !self.flag()? {
            return Err(PixelForgeError::InvalidInput(
                "AV1: OBU payload overruns its trailing bits".to_string(),
            ));
        }
        while self.pos < end {
            if self.flag()? {
                return Err(PixelForgeError::InvalidInput(
                    "AV1: non-zero trailing_zero_bit".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_fixed_width_fields() {
        let mut r = BitReader::new(&[0b1010_1100, 0xff, 0x01]);
        assert_eq!(r.f(1).unwrap(), 1);
        assert_eq!(r.f(3).unwrap(), 0b010);
        assert_eq!(r.f(4).unwrap(), 0b1100);
        assert_eq!(r.position(), 8);
        assert_eq!(r.f(16).unwrap(), 0xff01);
        assert!(r.f(1).is_err(), "reading past the end must fail");
    }

    #[test]
    fn reads_signed_fields() {
        // su(7) of 0b1111111 is -1, of 0b0111111 is 63, of 0b1000000 is -64.
        let mut r = BitReader::new(&[0b1111_1110, 0b1111_1110, 0b0000_0000]);
        assert_eq!(r.su(7).unwrap(), -1);
        assert_eq!(r.su(7).unwrap(), 63);
        assert_eq!(r.su(7).unwrap(), -64);
    }

    #[test]
    fn reads_leb128() {
        // 0xE5 0x8E 0x26 is the classic 624485 example.
        let mut r = BitReader::new(&[0xe5, 0x8e, 0x26, 0x05]);
        assert_eq!(r.leb128().unwrap(), 624_485);
        assert_eq!(r.leb128().unwrap(), 5);
        assert!(BitReader::new(&[0x80; 9]).leb128().is_err());
        assert!(BitReader::new(&[0x80]).leb128().is_err());
    }

    #[test]
    fn reads_uvlc() {
        // "1" is 0, "010" is 1, "011" is 2, "00100" is 3.
        let mut r = BitReader::new(&[0b1010_0110, 0b0100_0000]);
        assert_eq!(r.uvlc().unwrap(), 0);
        assert_eq!(r.uvlc().unwrap(), 1);
        assert_eq!(r.uvlc().unwrap(), 2);
        assert_eq!(r.uvlc().unwrap(), 3);
    }

    #[test]
    fn reads_ns() {
        // ns(5): w = 3, m = 3. Values 0..3 take 2 bits; 3 and 4 take 3 bits.
        let mut r = BitReader::new(&[0b1011_0111, 0b0000_0000]);
        assert_eq!(r.ns(5).unwrap(), 2); // 10
        assert_eq!(r.ns(5).unwrap(), 3); // 11 0 -> (3 << 1) - 3 + 0
        assert_eq!(r.ns(5).unwrap(), 4); // 11 1 -> (3 << 1) - 3 + 1
        assert_eq!(r.position(), 8);
        assert_eq!(r.ns(1).unwrap(), 0);
        assert_eq!(r.position(), 8, "ns(1) reads nothing");
    }

    #[test]
    fn reads_le() {
        let mut r = BitReader::new(&[0x34, 0x12, 0x00, 0x01]);
        assert_eq!(r.le(2).unwrap(), 0x1234);
        assert_eq!(r.le(2).unwrap(), 0x0100);
    }

    #[test]
    fn checks_alignment_and_trailing_bits() {
        let mut r = BitReader::new(&[0b1000_0000, 0b1100_0000]);
        r.f(1).unwrap();
        r.byte_alignment().unwrap();
        assert_eq!(r.position(), 8);
        r.f(1).unwrap();
        r.trailing_bits(2).unwrap();

        let mut r = BitReader::new(&[0b1010_0000]);
        r.f(1).unwrap();
        assert!(r.byte_alignment().is_err(), "non-zero padding bit");

        let mut r = BitReader::new(&[0b0100_0001]);
        r.f(1).unwrap();
        assert!(r.trailing_bits(1).is_err(), "stray bit after the one");
    }
}
