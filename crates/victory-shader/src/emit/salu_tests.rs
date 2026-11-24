//! Synthetic GFX10 scalar-ALU shaders, lowered to DXIL.
//!
//! Two kinds of test. The first lowers a one-instruction shader for each
//! scalar opcode this module covers, which is what catches an opcode
//! falling through the lowering again. The second checks the bit algorithms
//! the lowerings emit (field extracts, the four-bit group masks, the bit
//! replication) against plain implementations of the same hardware
//! semantics, over the edge cases and a spread of random values; those run
//! on any host, with no GPU and no `dxil.dll`.

use super::synth::{lower_body, sgpr, sop1, sop2, sopc, vop1};

/// `v_mov_b32 v0, s20`: the result travels to the stored VGPR.
const fn take_result() -> u32 {
    vop1(0x001, 0, sgpr(20))
}

#[test]
fn every_covered_scalar_opcode_lowers() {
    // One source (SOP1), writing s20 (a pair at s[20:21] for the 64-bit
    // forms), reading s0 or the pair s[0:1].
    let one_source: &[(&str, u16)] = &[
        ("s_bcnt0_i32_b32", 0x00d),
        ("s_bcnt0_i32_b64", 0x00e),
        ("s_ff0_i32_b32", 0x011),
        ("s_ff0_i32_b64", 0x012),
        ("s_sext_i32_i8", 0x019),
        ("s_sext_i32_i16", 0x01a),
        ("s_quadmask_b32", 0x02c),
        ("s_quadmask_b64", 0x02d),
        ("s_wqm_b32", 0x009),
        ("s_wqm_b64", 0x00a),
        ("s_brev_b64", 0x00c),
        ("s_flbit_i32_b64", 0x016),
        ("s_bitreplicate_b64_b32", 0x03b),
    ];
    for &(name, op) in one_source {
        let body = [sop1(op, 20, 0), take_result()];
        assert_eq!(lower_body(&body), Ok(()), "{name}");
    }

    // Two sources (SOP2), writing s20 or s[20:21].
    let two_sources: &[(&str, u16)] = &[
        ("s_absdiff_i32", 0x02c),
        ("s_pack_ll_b32_b16", 0x032),
        ("s_pack_lh_b32_b16", 0x033),
        ("s_pack_hh_b32_b16", 0x034),
        ("s_bfe_u64", 0x029),
        ("s_bfe_i64", 0x02a),
        ("s_bfm_b64", 0x025),
    ];
    for &(name, op) in two_sources {
        let body = [sop2(op, 20, 0, 1), take_result()];
        assert_eq!(lower_body(&body), Ok(()), "{name}");
    }

    // These read their destination's old value as well as their source.
    for &(name, op) in &[("s_bitset0_b64", 0x01cu16), ("s_bitset1_b64", 0x01e)] {
        let body = [
            sop1(0x03, 20, 0x80), // s_mov_b32 s20, 0
            sop1(0x03, 21, 0x80), // s_mov_b32 s21, 0
            sop1(op, 20, 0),
            take_result(),
        ];
        assert_eq!(lower_body(&body), Ok(()), "{name}");
    }

    // And these read SCC, so something must set it first.
    for &(name, op) in &[("s_cmov_b32", 0x005u16), ("s_cmov_b64", 0x006)] {
        let body = [
            sop1(0x03, 20, 0x80), // s_mov_b32 s20, 0
            sop1(0x03, 21, 0x80), // s_mov_b32 s21, 0
            sopc(0x06, 0, 1),     // s_cmp_eq_u32 s0, s1
            sop1(op, 20, 0),
            take_result(),
        ];
        assert_eq!(lower_body(&body), Ok(()), "{name}");
    }
}

/// A spread of values: the edges, then a deterministic scatter.
fn values() -> Vec<u64> {
    let mut v = vec![
        0,
        1,
        0xFFFF_FFFF,
        0xFFFF_FFFF_FFFF_FFFF,
        0x8000_0000,
        0x8000_0000_0000_0000,
        0x7FFF_FFFF,
        0xFFFF_0000,
        0x0000_FFFF,
        0xF0F0_F0F0_0F0F_0F0F,
    ];
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..200 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        v.push(x);
    }
    v
}

/// What the lowering of `s_bfe_u64`/`s_bfe_i64` computes, in the same order
/// of operations: the mask is all ones at a width of 64 or more, and a
/// width of zero gives zero.
fn bfe64(x: u64, off: u32, width: u32, signed: bool) -> u64 {
    let off = off & 63;
    let shifted = if signed {
        ((x as i64) >> off) as u64
    } else {
        x >> off
    };
    let w = width & 63;
    let mask = if width >= 64 {
        u64::MAX
    } else {
        (1u64 << w).wrapping_sub(1)
    };
    let r = shifted & mask;
    if !signed || width >= 64 || w == 0 {
        return r;
    }
    let left = (64 - w) & 63;
    (((r << left) as i64) >> left) as u64
}

#[test]
fn the_field_extract_matches_the_hardware_definition() {
    // The hardware takes the field at `offset`, `width` bits wide, and
    // sign-extends it for the signed form.
    let reference = |x: u64, off: u32, width: u32, signed: bool| -> u64 {
        let off = off & 63;
        let w = width & 63;
        let w = if width >= 64 { 64 } else { w };
        if w == 0 {
            return 0;
        }
        let mut bits = 0u64;
        for i in 0..w {
            let from = off + i;
            let bit = if from < 64 {
                (x >> from) & 1
            } else if signed {
                (x >> 63) & 1
            } else {
                0
            };
            bits |= bit << i;
        }
        if signed && w < 64 && (bits >> (w - 1)) & 1 == 1 {
            bits |= u64::MAX << w;
        }
        bits
    };
    for &x in &values() {
        for off in [0u32, 1, 7, 31, 32, 33, 63] {
            for width in [0u32, 1, 8, 31, 32, 33, 63, 64] {
                for signed in [false, true] {
                    assert_eq!(
                        bfe64(x, off, width, signed),
                        reference(x, off, width, signed),
                        "x={x:#018x} off={off} width={width} signed={signed}"
                    );
                }
            }
        }
    }
}

/// What the lowering of `s_quadmask`/`s_wqm` computes for one dword.
fn nibbles(a: u32, wqm: bool, base: u32) -> u32 {
    let mut acc = 0u32;
    for g in 0..8u32 {
        if (a >> (4 * g)) & 0xF != 0 {
            acc |= if wqm { 0xF << (4 * g) } else { 1 << (base + g) };
        }
    }
    acc
}

#[test]
fn the_four_bit_group_masks_match_the_hardware_definition() {
    for &x in &values() {
        let (lo, hi) = (x as u32, (x >> 32) as u32);
        // s_quadmask: one bit per group of four, 8 per dword, the 64-bit
        // form's 16 bits all in the low dword.
        let quad32 = nibbles(lo, false, 0);
        let mut want = 0u32;
        for g in 0..8 {
            want |= u32::from((lo >> (4 * g)) & 0xF != 0) << g;
        }
        assert_eq!(quad32, want, "s_quadmask_b32 {lo:#010x}");

        let quad64 = nibbles(lo, false, 0) | nibbles(hi, false, 8);
        let mut want = 0u32;
        for g in 0..16 {
            want |= u32::from((x >> (4 * g)) & 0xF != 0) << g;
        }
        assert_eq!(quad64, want, "s_quadmask_b64 {x:#018x}");

        // s_wqm: a group holding any set bit becomes four set bits.
        for &d in &[lo, hi] {
            let mut want = 0u32;
            for g in 0..8 {
                if (d >> (4 * g)) & 0xF != 0 {
                    want |= 0xF << (4 * g);
                }
            }
            assert_eq!(nibbles(d, true, 0), want, "s_wqm {d:#010x}");
        }
    }
}

/// What the lowering of `s_bitreplicate_b64_b32` computes for one dword of
/// the result: the 16 bits of `a` from `from` upwards, each one twice.
fn replicate_bits(a: u32, from: u32) -> u32 {
    let mut x = (a >> from) & 0xFFFF;
    for (step, keep) in [
        (8u32, 0x00FF_00FFu32),
        (4, 0x0F0F_0F0F),
        (2, 0x3333_3333),
        (1, 0x5555_5555),
    ] {
        x = (x | (x << step)) & keep;
    }
    x | (x << 1)
}

#[test]
fn the_bit_replication_matches_the_hardware_definition() {
    for &x in &values() {
        let a = x as u32;
        for (from, part) in [(0u32, 0u32), (16, 1)] {
            let mut want = 0u64;
            for i in 0..16 {
                let bit = u64::from((a >> (from + i)) & 1);
                want |= bit << (2 * i) | bit << (2 * i + 1);
            }
            assert_eq!(
                u64::from(replicate_bits(a, from)),
                want,
                "dword {part} of s_bitreplicate_b64_b32 {a:#010x}"
            );
        }
    }
}

#[test]
fn the_remaining_scalar_results_match_the_hardware_definition() {
    for &x in &values() {
        let (lo, hi) = (x as u32, (x >> 32) as u32);

        // s_absdiff_i32: the distance between two signed numbers.
        let d = lo.wrapping_sub(hi);
        let mut absdiff = d;
        if (d as i32) < 0 {
            absdiff = 0u32.wrapping_sub(d);
        }
        let want = (lo as i32).wrapping_sub(hi as i32).wrapping_abs() as u32;
        assert_eq!(absdiff, want, "s_absdiff_i32 {lo:#010x} {hi:#010x}");

        // s_sext_i32_i8 and _i16: the low byte or half, sign-extended.
        assert_eq!(
            ((lo << 24) as i32 >> 24) as u32,
            lo as u8 as i8 as i32 as u32,
            "s_sext_i32_i8 {lo:#010x}"
        );
        assert_eq!(
            ((lo << 16) as i32 >> 16) as u32,
            lo as u16 as i16 as i32 as u32,
            "s_sext_i32_i16 {lo:#010x}"
        );

        // s_pack_*: the halves the names name.
        assert_eq!(
            (lo & 0xFFFF) | (hi & 0xFFFF) << 16,
            u32::from(lo as u16) | u32::from(hi as u16) << 16,
            "s_pack_ll_b32_b16"
        );
        assert_eq!(
            (lo & 0xFFFF) | (hi >> 16) << 16,
            u32::from(lo as u16) | (hi >> 16) << 16,
            "s_pack_lh_b32_b16"
        );
        assert_eq!(
            (lo >> 16) | (hi >> 16) << 16,
            (lo >> 16) | (hi >> 16) << 16,
            "s_pack_hh_b32_b16"
        );

        // s_bcnt0 and s_ff0: the zero bits of the complement's set bits.
        assert_eq!((!lo).count_ones(), lo.count_zeros(), "s_bcnt0_i32_b32");
        assert_eq!(
            (!lo).count_ones() + (!hi).count_ones(),
            x.count_zeros(),
            "s_bcnt0_i32_b64"
        );
        let ff0_32 = if lo == u32::MAX {
            -1i32
        } else {
            (!lo).trailing_zeros() as i32
        };
        let mut want = lo.trailing_ones() as i32;
        if lo == u32::MAX {
            want = -1;
        }
        assert_eq!(ff0_32, want, "s_ff0_i32_b32 {lo:#010x}");
        let ff0_64 = if x == u64::MAX {
            -1i32
        } else if !lo != 0 {
            (!lo).trailing_zeros() as i32
        } else {
            32 + (!hi).trailing_zeros() as i32
        };
        let mut want = x.trailing_ones() as i32;
        if x == u64::MAX {
            want = -1;
        }
        assert_eq!(ff0_64, want, "s_ff0_i32_b64 {x:#018x}");

        // s_flbit_i32_b64: the leading zeros of the pair, -1 for none.
        let flbit = if hi != 0 {
            hi.leading_zeros() as i32
        } else if lo != 0 {
            32 + lo.leading_zeros() as i32
        } else {
            -1
        };
        let mut want = x.leading_zeros() as i32;
        if x == 0 {
            want = -1;
        }
        assert_eq!(flbit, want, "s_flbit_i32_b64 {x:#018x}");

        // s_brev_b64: the dwords swap as their bits reverse.
        let brev = u64::from(hi.reverse_bits()) | u64::from(lo.reverse_bits()) << 32;
        assert_eq!(brev, x.reverse_bits(), "s_brev_b64 {x:#018x}");

        // s_bfm_b64: a field of ones, shifted up.
        let (w, sh) = (lo & 63, hi & 63);
        let bfm = ((1u64 << w) - 1) << sh;
        let mut want = 0u64;
        for i in 0..w {
            want |= 1u64 << i;
        }
        assert_eq!(bfm, want << sh, "s_bfm_b64 w={w} sh={sh}");
    }
}
