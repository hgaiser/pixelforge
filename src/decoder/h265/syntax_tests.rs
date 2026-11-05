//! Slice header syntax that no fixture encoder produces: inter RPS prediction,
//! long-term pictures, and emulation prevention inside the short-term set.
//! The bitstreams are written by hand with a small bit writer.

use super::parser::{NalType, NalUnit, Reader, parse_nal_header};
use super::pps::Pps;
use super::rps::parse_st_ref_pic_set;
use super::slice::parse_slice_header;
use super::sps::{LongTermRefPicsSps, Sps};

/// Writes RBSP bits and produces an EBSP payload.
#[derive(Default)]
struct BitWriter {
    bits: Vec<bool>,
}

impl BitWriter {
    fn u(&mut self, n: u32, v: u32) -> &mut Self {
        for i in (0..n).rev() {
            self.bits.push(v >> i & 1 != 0);
        }
        self
    }

    fn flag(&mut self, v: bool) -> &mut Self {
        self.u(1, v as u32)
    }

    fn ue(&mut self, v: u32) -> &mut Self {
        let code = v as u64 + 1;
        let len = 64 - code.leading_zeros();
        self.u(len - 1, 0);
        for i in (0..len).rev() {
            self.bits.push(code >> i & 1 != 0);
        }
        self
    }

    fn len(&self) -> usize {
        self.bits.len()
    }

    /// Append rbsp_trailing_bits() and insert emulation prevention bytes.
    fn finish(&mut self) -> Vec<u8> {
        self.flag(true);
        while !self.bits.len().is_multiple_of(8) {
            self.bits.push(false);
        }
        let rbsp: Vec<u8> = self
            .bits
            .chunks(8)
            .map(|c| c.iter().fold(0u8, |b, bit| b << 1 | *bit as u8))
            .collect();
        let mut out = Vec::new();
        let mut zeros = 0;
        for byte in rbsp {
            if zeros >= 2 && byte <= 3 {
                out.push(3);
                zeros = 0;
            }
            out.push(byte);
            zeros = if byte == 0 { zeros + 1 } else { 0 };
        }
        out
    }
}

fn nal_bytes(nal_type: NalType, payload: &[u8]) -> Vec<u8> {
    let mut data = vec![nal_type.raw() << 1, 1];
    data.extend_from_slice(payload);
    data
}

fn base_sps() -> Sps {
    let mut sps = Sps {
        pic_width_in_luma_samples: 64,
        pic_height_in_luma_samples: 64,
        log2_max_pic_order_cnt_lsb_minus4: 0,
        ..Default::default()
    };
    sps.dec_pic_buf_mgr.max_dec_pic_buffering_minus1 = [5; 7];
    sps
}

/// The leading slice header bits of a P picture up to the short-term set.
fn slice_prefix(w: &mut BitWriter, lsb: u32) {
    w.flag(true) // first_slice_segment_in_pic_flag
        .ue(0) // slice_pic_parameter_set_id
        .ue(1) // slice_type P
        .u(4, lsb)
        .flag(false); // short_term_ref_pic_set_sps_flag
}

/// SPS set 0 explicit, set 1 predicted from it; the slice predicts from set
/// 0 with delta_idx_minus1 = 1. Checks the derived sets, the bit count and
/// NumDeltaPocsOfRefRpsIdx.
#[test]
fn inter_rps_prediction_in_sps_and_slice() {
    let mut sps = base_sps();
    let mut w = BitWriter::default();
    // Set 0: S0 = {-1, -3}, S1 = {+2}, all used.
    w.ue(2)
        .ue(1)
        .ue(0)
        .flag(true)
        .ue(1)
        .flag(true)
        .ue(1)
        .flag(true);
    // Set 1: inter, deltaRps = -1, all four entries used.
    w.flag(true).flag(true).ue(0);
    for _ in 0..4 {
        w.flag(true);
    }
    let payload = w.finish();
    let mut r = Reader::new(&payload);
    for i in 0..2 {
        let set = parse_st_ref_pic_set(&mut r, i, 2, &sps.short_term_ref_pic_sets, 5).unwrap();
        sps.short_term_ref_pic_sets.push(set);
    }
    let set0 = &sps.short_term_ref_pic_sets[0].rps;
    assert_eq!(
        (set0.delta_poc_s0.clone(), set0.delta_poc_s1.clone()),
        (vec![-1, -3], vec![2])
    );
    let set1 = &sps.short_term_ref_pic_sets[1];
    assert!(set1.syntax.inter_ref_pic_set_prediction_flag);
    assert_eq!(set1.rps.delta_poc_s0, vec![-1, -2, -4]);
    assert_eq!(set1.rps.delta_poc_s1, vec![1]);

    let mut w = BitWriter::default();
    slice_prefix(&mut w, 5);
    let start = w.len();
    w.flag(true) // inter_ref_pic_set_prediction_flag
        .ue(1) // delta_idx_minus1: RefRpsIdx = 0
        .flag(false) // delta_rps_sign: deltaRps = +1
        .ue(0);
    // used_by_curr_pic_flag / use_delta_flag for j = 0..=3.
    w.flag(true);
    w.flag(false).flag(true);
    w.flag(false).flag(false);
    w.flag(true);
    let rps_bits = w.len() - start;
    let data = nal_bytes(NalType::TrailR, &w.finish());
    let nal = NalUnit {
        header: parse_nal_header(&data).unwrap(),
        data: &data,
    };
    let h = parse_slice_header(&nal, &sps, &Pps::default()).unwrap();
    assert_eq!(h.num_bits_for_st_ref_pic_set as usize, rps_bits);
    assert_eq!(rps_bits, 12);
    assert_eq!(h.num_delta_pocs_of_ref_rps_idx(&sps), 3);
    let set = h.st_ref_pic_set.as_ref().unwrap();
    assert_eq!(set.syntax.delta_idx_minus1, 1);
    assert_eq!(set.syntax.used_by_curr_pic_flag, 0b1001);
    assert_eq!(set.syntax.use_delta_flag, 0b1011);
    assert_eq!(set.rps.delta_poc_s0, vec![-2]);
    assert_eq!(set.rps.used_s0, vec![false]);
    assert_eq!(set.rps.delta_poc_s1, vec![1]);
    assert_eq!(set.rps.used_s1, vec![true]);
}

/// Long-term entries from the SPS and the slice, with the MSB cycle sum
/// restarting at the first slice-coded entry (equation 7-52).
#[test]
fn long_term_slice_syntax() {
    let mut sps = base_sps();
    sps.long_term_ref_pics_present_flag = true;
    sps.long_term_ref_pics = LongTermRefPicsSps {
        lt_ref_pic_poc_lsb_sps: vec![5, 9, 13],
        used_by_curr_pic_lt_sps_flag: 0b010,
    };
    sps.sps_temporal_mvp_enabled_flag = true;

    let mut w = BitWriter::default();
    slice_prefix(&mut w, 15);
    w.ue(1).ue(0).ue(0).flag(true); // S0 = {-1}
    w.ue(1).ue(2); // num_long_term_sps, num_long_term_pics
    w.u(2, 1).flag(true).ue(2); // lt_idx_sps 1, msb cycle 2
    w.u(4, 7).flag(false).flag(true).ue(1); // slice entry, msb cycle 1
    w.u(4, 3).flag(true).flag(true).ue(4); // slice entry, msb cycle 4
    w.flag(true); // slice_temporal_mvp_enabled_flag
    let data = nal_bytes(NalType::TrailR, &w.finish());
    let nal = NalUnit {
        header: parse_nal_header(&data).unwrap(),
        data: &data,
    };
    let h = parse_slice_header(&nal, &sps, &Pps::default()).unwrap();
    assert_eq!((h.num_long_term_sps, h.num_long_term_pics), (1, 2));
    let lt = &h.long_term_refs;
    assert_eq!(lt[0].lt_idx_sps, Some(1));
    assert_eq!((lt[0].poc_lsb_lt, lt[0].used_by_curr_pic_lt), (9, true));
    assert_eq!((lt[1].poc_lsb_lt, lt[1].used_by_curr_pic_lt), (7, false));
    assert_eq!((lt[2].poc_lsb_lt, lt[2].used_by_curr_pic_lt), (3, true));
    let acc: Vec<u32> = lt
        .iter()
        .map(|e| e.delta_poc_msb_cycle_lt_accumulated)
        .collect();
    assert_eq!(acc, vec![2, 1, 5]);
    assert!(h.slice_temporal_mvp_enabled_flag);
}

/// Long zero runs inside the short-term set force emulation prevention
/// bytes; the bit count must not include them.
#[test]
fn rps_bit_count_excludes_emulation_prevention() {
    let sps = base_sps();
    let mut w = BitWriter::default();
    slice_prefix(&mut w, 0);
    let start = w.len();
    w.ue(2).ue(0);
    w.ue(32767).flag(false).ue(32767).flag(true);
    let rps_bits = w.len() - start;
    let payload = w.finish();
    assert!(
        payload.windows(3).any(|b| b == [0, 0, 3]),
        "the test needs an emulation prevention byte"
    );
    let data = nal_bytes(NalType::TrailR, &payload);
    let nal = NalUnit {
        header: parse_nal_header(&data).unwrap(),
        data: &data,
    };
    let h = parse_slice_header(&nal, &sps, &Pps::default()).unwrap();
    assert_eq!(h.num_bits_for_st_ref_pic_set as usize, rps_bits);
    let set = &h.st_ref_pic_set.as_ref().unwrap().rps;
    assert_eq!(set.delta_poc_s0, vec![-32768, -65536]);
    assert_eq!(set.used_s0, vec![false, true]);
}
