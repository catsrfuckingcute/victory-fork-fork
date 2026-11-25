//! The opcodes added in `salu_tests`, `valu_int_tests` and
//! `valu16_int_tests`, this time validated by `dxil.dll` and run on the
//! host: each shader's one result is compared with the same value worked
//! out in Rust, on the hardware adapter and on WARP.
//!
//! The modules named above check that these opcodes lower and that the
//! arithmetic the lowerings emit is right; these cases check that the DXIL
//! is accepted and that a GPU agrees.

use super::synth::{sgpr, sop1, sop2, sopc, vgpr, vop1, vop2, vop3};
use super::valu_tests::{build_cs, run_cases};

/// `v_mov_b32 v0, s<n>`: the scalar result into the stored VGPR.
const fn take_sgpr(n: u8) -> u32 {
    vop1(0x001, 0, sgpr(n))
}

/// Builds a one-thread shader around `body`, validates it, runs it on both
/// hosts for every input set, and compares each result with `want`.
fn check(name: &str, body: &[u32], want: impl Fn([u32; 4]) -> u32) {
    // Four dwords per case: edges, then a scatter. The last two are often
    // a shift amount and a field width, so they stay small as well as wide.
    const CASES: &[[u32; 4]] = &[
        [0, 0, 0, 0],
        [0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF],
        [1, 0, 0, 1],
        [0x8000_0000, 0x7FFF_FFFF, 31, 32],
        [0x0000_FFFF, 0xFFFF_0000, 16, 16],
        [0x1234_5678, 0x9ABC_DEF0, 8, 12],
        [0xF0F0_F0F0, 0x0F0F_0F0F, 4, 63],
        [0x8080_8080, 0x7F7F_7F7F, 1, 8],
        [0x00FF_00FF, 0xFF00_FF00, 33, 64],
        [0xDEAD_BEEF, 0xCAFE_BABE, 0, 0],
        [0x0000_0080, 0x0000_7FFF, 7, 1],
        [0x8000_0001, 0x0000_0002, 63, 33],
    ];
    let shader = build_cs(body, 1, 0);
    run_cases(&shader, CASES, |backend, i, inputs, got| {
        let [a, b, c, d] = inputs;
        assert_eq!(
            got,
            want(inputs),
            "{name} on {backend}, case {i}: inputs {a:#010x} {b:#010x} {c:#010x} {d:#010x}"
        );
    });
}

fn sext16(x: u32) -> u32 {
    ((x << 16) as i32 >> 16) as u32
}

fn field(x: u32, i: u32, bits: u32) -> u32 {
    (x >> (i * bits)) & ((1u32 << bits) - 1)
}

fn nibbles(a: u32, wqm: bool, base: u32) -> u32 {
    let mut acc = 0u32;
    for g in 0..8u32 {
        if (a >> (4 * g)) & 0xF != 0 {
            acc |= if wqm { 0xF << (4 * g) } else { 1 << (base + g) };
        }
    }
    acc
}

fn replicate(a: u32, from: u32) -> u32 {
    let mut r = 0u32;
    for i in 0..16u32 {
        let bit = (a >> (from + i)) & 1;
        r |= bit << (2 * i) | bit << (2 * i + 1);
    }
    r
}

/// The field `s_bfe_*_64` takes: offset in src1[5:0], width in src1[22:16].
fn bfe64(x: u64, sel: u32, signed: bool) -> u64 {
    let off = sel & 63;
    let width = (sel >> 16) & 0x7F;
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

// ---------------------------------------------------------------- scalar

#[test]
fn scalar_results_on_the_host() {
    check("s_absdiff_i32", &[sop2(0x02c, 20, 0, 1), take_sgpr(20)], |[a, b, ..]| {
        (a as i32).wrapping_sub(b as i32).wrapping_abs() as u32
    });

    check("s_bcnt0_i32_b32", &[sop1(0x00d, 20, 0), take_sgpr(20)], |[a, ..]| {
        a.count_zeros()
    });

    check("s_ff0_i32_b32", &[sop1(0x011, 20, 0), take_sgpr(20)], |[a, ..]| {
        if a == u32::MAX {
            u32::MAX
        } else {
            a.trailing_ones()
        }
    });

    check("s_sext_i32_i8", &[sop1(0x019, 20, 0), take_sgpr(20)], |[a, ..]| {
        a as u8 as i8 as i32 as u32
    });
    check("s_sext_i32_i16", &[sop1(0x01a, 20, 0), take_sgpr(20)], |[a, ..]| {
        a as u16 as i16 as i32 as u32
    });

    check("s_pack_ll_b32_b16", &[sop2(0x032, 20, 0, 1), take_sgpr(20)], |[a, b, ..]| {
        (a & 0xFFFF) | (b & 0xFFFF) << 16
    });
    check("s_pack_lh_b32_b16", &[sop2(0x033, 20, 0, 1), take_sgpr(20)], |[a, b, ..]| {
        (a & 0xFFFF) | (b >> 16) << 16
    });
    check("s_pack_hh_b32_b16", &[sop2(0x034, 20, 0, 1), take_sgpr(20)], |[a, b, ..]| {
        (a >> 16) | (b >> 16) << 16
    });

    check("s_quadmask_b32", &[sop1(0x02c, 20, 0), take_sgpr(20)], |[a, ..]| {
        nibbles(a, false, 0)
    });
    check("s_wqm_b32", &[sop1(0x009, 20, 0), take_sgpr(20)], |[a, ..]| {
        nibbles(a, true, 0)
    });

    // s_cmov_b32 keeps s20's old value (s1) when SCC is clear.
    check(
        "s_cmov_b32",
        &[
            sop1(0x03, 20, 1),
            sopc(0x06, 2, 3),
            sop1(0x005, 20, 0),
            take_sgpr(20),
        ],
        |[a, b, c, d]| if c == d { a } else { b },
    );
}

#[test]
fn scalar_64_bit_results_on_the_host() {
    check("s_bcnt0_i32_b64", &[sop1(0x00e, 20, 0), take_sgpr(20)], |[a, b, ..]| {
        a.count_zeros() + b.count_zeros()
    });

    check("s_ff0_i32_b64", &[sop1(0x012, 20, 0), take_sgpr(20)], |[a, b, ..]| {
        let x = u64::from(a) | u64::from(b) << 32;
        if x == u64::MAX {
            u32::MAX
        } else {
            x.trailing_ones()
        }
    });

    check("s_flbit_i32_b64", &[sop1(0x016, 20, 0), take_sgpr(20)], |[a, b, ..]| {
        let x = u64::from(a) | u64::from(b) << 32;
        if x == 0 { u32::MAX } else { x.leading_zeros() }
    });

    // s_brev_b64: the dwords swap as their bits reverse.
    check("s_brev_b64 lo", &[sop1(0x00c, 20, 0), take_sgpr(20)], |[_, b, ..]| {
        b.reverse_bits()
    });
    check("s_brev_b64 hi", &[sop1(0x00c, 20, 0), take_sgpr(21)], |[a, ..]| {
        a.reverse_bits()
    });

    for (label, dword) in [("lo", 20u8), ("hi", 21)] {
        let shift = u32::from(dword - 20) * 32;

        check(
            &format!("s_bfe_u64 {label}"),
            &[sop2(0x029, 20, 0, 2), take_sgpr(dword)],
            move |[a, b, c, _]| {
                let x = u64::from(a) | u64::from(b) << 32;
                (bfe64(x, c, false) >> shift) as u32
            },
        );
        check(
            &format!("s_bfe_i64 {label}"),
            &[sop2(0x02a, 20, 0, 2), take_sgpr(dword)],
            move |[a, b, c, _]| {
                let x = u64::from(a) | u64::from(b) << 32;
                (bfe64(x, c, true) >> shift) as u32
            },
        );
        check(
            &format!("s_bfm_b64 {label}"),
            &[sop2(0x025, 20, 0, 1), take_sgpr(dword)],
            move |[a, b, ..]| {
                let (w, sh) = (a & 63, b & 63);
                let m = ((1u128 << w) - 1) << sh;
                ((m as u64) >> shift) as u32
            },
        );
        check(
            &format!("s_wqm_b64 {label}"),
            &[sop1(0x00a, 20, 0), take_sgpr(dword)],
            move |[a, b, ..]| nibbles(if shift == 0 { a } else { b }, true, 0),
        );
        check(
            &format!("s_bitreplicate_b64_b32 {label}"),
            &[sop1(0x03b, 20, 0), take_sgpr(dword)],
            move |[a, ..]| replicate(a, if shift == 0 { 0 } else { 16 }),
        );
        // s_bitset0/1_b64 read the destination, so it is set up first.
        for (name, op, set) in [("s_bitset0_b64", 0x01cu16, false), ("s_bitset1_b64", 0x01e, true)] {
            check(
                &format!("{name} {label}"),
                &[
                    sop1(0x03, 20, 0),
                    sop1(0x03, 21, 1),
                    sop1(op, 20, 2),
                    take_sgpr(dword),
                ],
                move |[a, b, c, _]| {
                    let old = u64::from(a) | u64::from(b) << 32;
                    let bit = 1u64 << (c & 63);
                    let r = if set { old | bit } else { old & !bit };
                    (r >> shift) as u32
                },
            );
        }
        // s_cmov_b64 keeps s[20:21]'s old value (s3, s2) when SCC is clear.
        check(
            &format!("s_cmov_b64 {label}"),
            &[
                sop1(0x03, 20, 3),
                sop1(0x03, 21, 2),
                sopc(0x06, 2, 3),
                sop1(0x006, 20, 0),
                take_sgpr(dword),
            ],
            move |[a, b, c, d]| {
                let taken = c == d;
                if shift == 0 {
                    if taken { a } else { d }
                } else if taken {
                    b
                } else {
                    c
                }
            },
        );
    }

    check("s_quadmask_b64 lo", &[sop1(0x02d, 20, 0), take_sgpr(20)], |[a, b, ..]| {
        nibbles(a, false, 0) | nibbles(b, false, 8)
    });
    check("s_quadmask_b64 hi", &[sop1(0x02d, 20, 0), take_sgpr(21)], |_| 0);
}

// ---------------------------------------------------------------- vector

/// The root constants arrive in `v1`, `v2`, `v3` (and `v4`).
fn valu(body: &[u32]) -> Vec<u32> {
    body.to_vec()
}

#[test]
fn integer_vector_results_on_the_host() {
    let three = |op: u16| valu(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));

    check("v_mul_lo_i32", &three(0x16b), |[a, b, ..]| a.wrapping_mul(b));
    check("v_add_nc_i32", &three(0x37f), |[a, b, ..]| a.wrapping_add(b));
    check("v_sub_nc_i32", &three(0x376), |[a, b, ..]| a.wrapping_sub(b));

    let diff = |x: u32, y: u32| (i64::from(x) - i64::from(y)).unsigned_abs() as u32;

    check("v_sad_u32", &three(0x15d), move |[a, b, c, _]| {
        diff(a, b).wrapping_add(c)
    });
    check("v_sad_u8", &three(0x15a), move |[a, b, c, _]| {
        (0..4)
            .map(|i| diff(field(a, i, 8), field(b, i, 8)))
            .sum::<u32>()
            .wrapping_add(c)
    });
    check("v_sad_hi_u8", &three(0x15b), move |[a, b, c, _]| {
        ((0..4)
            .map(|i| diff(field(a, i, 8), field(b, i, 8)))
            .sum::<u32>()
            << 16)
            .wrapping_add(c)
    });
    check("v_sad_u16", &three(0x15c), move |[a, b, c, _]| {
        (0..2)
            .map(|i| diff(field(a, i, 16), field(b, i, 16)))
            .sum::<u32>()
            .wrapping_add(c)
    });
    check("v_msad_u8", &three(0x171), move |[a, b, c, _]| {
        (0..4)
            .map(|i| {
                let y = field(b, i, 8);
                if y == 0 { 0 } else { diff(field(a, i, 8), y) }
            })
            .sum::<u32>()
            .wrapping_add(c)
    });

    check("v_lerp_u8", &three(0x14d), |[a, b, c, _]| {
        let mut r = 0u32;
        for i in 0..4u32 {
            let t = field(a, i, 8) + field(b, i, 8) + (field(c, i, 8) & 1);
            r |= (t >> 1) << (8 * i);
        }
        r
    });

    check("v_cvt_pk_i16_i32", &three(0x36b), |[a, b, ..]| {
        let lo = (a as i32).clamp(-32768, 32767) as i16 as u16;
        let hi = (b as i32).clamp(-32768, 32767) as i16 as u16;
        u32::from(lo) | u32::from(hi) << 16
    });
    check("v_cvt_pk_u16_u32", &three(0x36a), |[a, b, ..]| {
        a.min(0xFFFF) | b.min(0xFFFF) << 16
    });
    check("v_sat_pk_u8_i16", &valu(&[vop1(0x062, 0, vgpr(1))]), |[a, ..]| {
        let b0 = (a as u16 as i16).clamp(0, 255) as u8;
        let b1 = ((a >> 16) as u16 as i16).clamp(0, 255) as u8;
        u32::from(b0) | u32::from(b1) << 8
    });

    check(
        "v_mul_hi_u32_u24",
        &valu(&[vop2(0x00c, 0, vgpr(1), 2)]),
        |[a, b, ..]| ((u64::from(a & 0xFF_FFFF) * u64::from(b & 0xFF_FFFF)) >> 32) as u32,
    );
    check(
        "v_mul_hi_i32_i24",
        &valu(&[vop2(0x00a, 0, vgpr(1), 2)]),
        |[a, b, ..]| {
            let s = |x: u32| i64::from(((x << 8) as i32) >> 8);
            ((s(a).wrapping_mul(s(b)) as u64) >> 32) as u32
        },
    );
}

#[test]
fn sixteen_bit_integer_results_on_the_host() {
    let three = |op: u16| valu(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));

    // The 16-bit forms write the low half of v0 and leave the high half as
    // it was, which is zero here.
    check("v_add_nc_i16", &three(0x30d), |[a, b, ..]| {
        u32::from((a as u16).wrapping_add(b as u16))
    });
    check("v_sub_nc_i16", &three(0x30e), |[a, b, ..]| {
        u32::from((a as u16).wrapping_sub(b as u16))
    });
    check("v_mad_u16", &three(0x340), |[a, b, c, _]| {
        u32::from((a as u16).wrapping_mul(b as u16).wrapping_add(c as u16))
    });
    check("v_mad_i16", &three(0x35e), |[a, b, c, _]| {
        u32::from((a as u16).wrapping_mul(b as u16).wrapping_add(c as u16))
    });

    let halves = |[a, b, c, _]: [u32; 4]| [a as u16, b as u16, c as u16];
    for (name, op, pick) in [
        ("v_min3_u16", 0x353u16, 0usize),
        ("v_med3_u16", 0x359, 1),
        ("v_max3_u16", 0x356, 2),
    ] {
        check(name, &three(op), move |inputs| {
            let mut v = halves(inputs);
            v.sort_unstable();
            u32::from(v[pick])
        });
    }
    for (name, op, pick) in [
        ("v_min3_i16", 0x352u16, 0usize),
        ("v_med3_i16", 0x358, 1),
        ("v_max3_i16", 0x355, 2),
    ] {
        check(name, &three(op), move |inputs| {
            let mut v = halves(inputs).map(|x| x as i16);
            v.sort_unstable();
            u32::from(v[pick] as u16)
        });
    }

    check("v_mad_u32_u16", &three(0x373), |[a, b, c, _]| {
        (u32::from(a as u16) * u32::from(b as u16)).wrapping_add(c)
    });
    check("v_mad_i32_i16", &three(0x375), |[a, b, c, _]| {
        (sext16(a).wrapping_mul(sext16(b))).wrapping_add(c)
    });
}
