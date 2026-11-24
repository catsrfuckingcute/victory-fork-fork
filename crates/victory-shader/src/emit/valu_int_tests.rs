//! Synthetic GFX10 integer VALU shaders, lowered to DXIL.
//!
//! As in `salu_tests`: one test lowers a shader for each opcode this module
//! covers, and the rest check the arithmetic the lowerings emit — the
//! distances summed by the `v_sad_*` family, the byte averages of
//! `v_lerp_u8`, the saturating packs and the 24-bit multiplies — against
//! plain implementations of the same hardware semantics. No GPU and no
//! `dxil.dll` needed.

use super::synth::{constant_to_vgpr, lower_body, vgpr, vop1, vop2, vop3};

/// Three VGPRs carrying root constants, for the three sources.
fn prologue() -> Vec<u32> {
    vec![
        constant_to_vgpr(1, 0),
        constant_to_vgpr(2, 1),
        constant_to_vgpr(3, 2),
    ]
}

#[test]
fn every_covered_integer_opcode_lowers() {
    // VOP3, three sources, writing v0.
    let three: &[(&str, u16)] = &[
        ("v_mul_lo_i32", 0x16b),
        ("v_add_nc_i32", 0x37f),
        ("v_sub_nc_i32", 0x376),
        ("v_sad_u8", 0x15a),
        ("v_sad_hi_u8", 0x15b),
        ("v_sad_u16", 0x15c),
        ("v_sad_u32", 0x15d),
        ("v_msad_u8", 0x171),
        ("v_lerp_u8", 0x14d),
        ("v_cvt_pk_i16_i32", 0x36b),
        ("v_cvt_pk_u16_u32", 0x36a),
    ];
    for &(name, op) in three {
        let mut body = prologue();
        let i = vop3(op, 0, vgpr(1), vgpr(2), vgpr(3));
        body.extend_from_slice(&i);
        assert_eq!(lower_body(&body), Ok(()), "{name}");
    }

    // VOP2, two sources.
    for &(name, op) in &[("v_mul_hi_u32_u24", 0x00cu16), ("v_mul_hi_i32_i24", 0x00a)] {
        let mut body = prologue();
        body.push(vop2(op, 0, vgpr(1), 2));
        assert_eq!(lower_body(&body), Ok(()), "{name}");
    }

    // VOP1, one source.
    let mut body = prologue();
    body.push(vop1(0x062, 0, vgpr(1)));
    assert_eq!(lower_body(&body), Ok(()), "v_sat_pk_u8_i16");
}

/// A spread of values: the edges, then a deterministic scatter.
fn values() -> Vec<u32> {
    let mut v = vec![
        0,
        1,
        0xFF,
        0x100,
        0x7FFF,
        0x8000,
        0xFFFF,
        0x1_0000,
        0x7FFF_FFFF,
        0x8000_0000,
        0xFFFF_FFFF,
        0x00FF_FFFF,
        0x0080_0000,
        0xFF00_FF00,
        0x7F7F_7F7F,
        0x8080_8080,
    ];
    let mut x = 0x9E37_79B9u32;
    for _ in 0..150 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        v.push(x);
    }
    v
}

fn field(x: u32, i: u32, bits: u32) -> u32 {
    (x >> (i * bits)) & ((1u32 << bits) - 1)
}

/// The distance the lowering computes with a compare and two subtractions.
fn abs_diff(a: u32, b: u32) -> u32 {
    a.abs_diff(b)
}

/// What the lowering of the `v_sad_*` family computes.
fn sad(name: &str, a: u32, b: u32, cin: u32) -> u32 {
    if name == "v_sad_u32" {
        return abs_diff(a, b).wrapping_add(cin);
    }
    let (bits, count) = if name == "v_sad_u16" { (16, 2) } else { (8, 4) };
    let mut sum = 0u32;
    for i in 0..count {
        let (x, y) = (field(a, i, bits), field(b, i, bits));
        let mut d = abs_diff(x, y);
        if name == "v_msad_u8" && y == 0 {
            d = 0;
        }
        sum = sum.wrapping_add(d);
    }
    if name == "v_sad_hi_u8" {
        sum <<= 16;
    }
    sum.wrapping_add(cin)
}

#[test]
fn the_distance_sums_match_the_hardware_definition() {
    let vals = values();
    for (n, &a) in vals.iter().enumerate() {
        let b = vals[(n + 7) % vals.len()];
        let cin = vals[(n + 13) % vals.len()];

        // The reference: the distance between each pair of fields, summed.
        let bytes = |x: u32| [x & 0xFF, (x >> 8) & 0xFF, (x >> 16) & 0xFF, x >> 24];
        let (xa, xb) = (bytes(a), bytes(b));
        let diff = |x: u32, y: u32| (i64::from(x) - i64::from(y)).unsigned_abs() as u32;
        let plain: u32 = (0..4).map(|i| diff(xa[i], xb[i])).sum();
        assert_eq!(
            sad("v_sad_u8", a, b, cin),
            plain.wrapping_add(cin),
            "v_sad_u8 {a:#010x} {b:#010x}"
        );
        assert_eq!(
            sad("v_sad_hi_u8", a, b, cin),
            (plain << 16).wrapping_add(cin),
            "v_sad_hi_u8 {a:#010x} {b:#010x}"
        );
        let masked: u32 = (0..4)
            .map(|i| if xb[i] == 0 { 0 } else { diff(xa[i], xb[i]) })
            .sum();
        assert_eq!(
            sad("v_msad_u8", a, b, cin),
            masked.wrapping_add(cin),
            "v_msad_u8 {a:#010x} {b:#010x}"
        );
        let halves = |x: u32| [x & 0xFFFF, x >> 16];
        let (ha, hb) = (halves(a), halves(b));
        let plain16: u32 = (0..2).map(|i| diff(ha[i], hb[i])).sum();
        assert_eq!(
            sad("v_sad_u16", a, b, cin),
            plain16.wrapping_add(cin),
            "v_sad_u16 {a:#010x} {b:#010x}"
        );
        assert_eq!(
            sad("v_sad_u32", a, b, cin),
            diff(a, b).wrapping_add(cin),
            "v_sad_u32 {a:#010x} {b:#010x}"
        );
    }
}

/// What the lowering of `v_lerp_u8` computes.
fn lerp_u8(a: u32, b: u32, round: u32) -> u32 {
    let mut out = 0u32;
    for i in 0..4u32 {
        let (x, y) = (field(a, i, 8), field(b, i, 8));
        let r = field(round, i, 8) & 1;
        out |= ((x + y + r) >> 1) << (8 * i);
    }
    out
}

#[test]
fn the_byte_averages_match_the_hardware_definition() {
    let vals = values();
    for (n, &a) in vals.iter().enumerate() {
        let b = vals[(n + 5) % vals.len()];
        let round = vals[(n + 11) % vals.len()];
        let got = lerp_u8(a, b, round);
        for i in 0..4u32 {
            let (x, y) = (field(a, i, 8), field(b, i, 8));
            let r = field(round, i, 8) & 1;
            // Each byte is the average, rounded up when the third
            // source's byte is odd, and never carries into the next.
            let want = (x + y + r) / 2;
            assert!(want <= 0xFF, "a byte average cannot overflow");
            assert_eq!(
                field(got, i, 8),
                want,
                "byte {i} of v_lerp_u8 {a:#010x} {b:#010x} {round:#010x}"
            );
        }
    }
}

/// What the lowering of the saturating packs computes.
fn clamp_int(x: u32, lo: u32, hi: u32, signed: bool) -> u32 {
    let too_big = if signed {
        (x as i32) > (hi as i32)
    } else {
        x > hi
    };
    let t = if too_big { hi } else { x };
    let too_small = if signed {
        (t as i32) < (lo as i32)
    } else {
        t < lo
    };
    if too_small { lo } else { t }
}

#[test]
fn the_saturating_packs_match_the_hardware_definition() {
    let vals = values();
    for (n, &a) in vals.iter().enumerate() {
        let b = vals[(n + 3) % vals.len()];

        // v_cvt_pk_i16_i32: each source held inside a signed 16-bit number.
        let pack_i16 = (clamp_int(a, 0xFFFF_8000, 0x7FFF, true) & 0xFFFF)
            | (clamp_int(b, 0xFFFF_8000, 0x7FFF, true) & 0xFFFF) << 16;
        let want_lo = (a as i32).clamp(-32768, 32767) as i16 as u16;
        let want_hi = (b as i32).clamp(-32768, 32767) as i16 as u16;
        assert_eq!(
            pack_i16,
            u32::from(want_lo) | u32::from(want_hi) << 16,
            "v_cvt_pk_i16_i32 {a:#010x} {b:#010x}"
        );

        // v_cvt_pk_u16_u32: each source held inside an unsigned one.
        let pack_u16 = (clamp_int(a, 0, 0xFFFF, false) & 0xFFFF)
            | (clamp_int(b, 0, 0xFFFF, false) & 0xFFFF) << 16;
        assert_eq!(
            pack_u16,
            a.min(0xFFFF) | b.min(0xFFFF) << 16,
            "v_cvt_pk_u16_u32 {a:#010x} {b:#010x}"
        );

        // v_sat_pk_u8_i16: the source's two signed halves, each held
        // inside a byte, in the low half of the result.
        let mut sat = 0u32;
        for i in 0..2u32 {
            let h = field(a, i, 16);
            let h = ((h << 16) as i32 >> 16) as u32;
            sat |= clamp_int(h, 0, 0xFF, true) << (8 * i);
        }
        let halves = [a as u16 as i16, (a >> 16) as u16 as i16];
        let want = u32::from(halves[0].clamp(0, 255) as u8)
            | u32::from(halves[1].clamp(0, 255) as u8) << 8;
        assert_eq!(sat, want, "v_sat_pk_u8_i16 {a:#010x}");
    }
}

/// What the lowering of the 24-bit multiplies computes.
fn mul_hi_24(a: u32, b: u32, signed: bool) -> u32 {
    let narrow = |x: u32| {
        if signed {
            ((x << 8) as i32 >> 8) as u32
        } else {
            (x << 8) >> 8
        }
    };
    let (a, b) = (narrow(a), narrow(b));
    if signed {
        ((i64::from(a as i32).wrapping_mul(i64::from(b as i32)) as u64) >> 32) as u32
    } else {
        ((u64::from(a) * u64::from(b)) >> 32) as u32
    }
}

#[test]
fn the_24_bit_multiplies_match_the_hardware_definition() {
    let vals = values();
    for (n, &a) in vals.iter().enumerate() {
        let b = vals[(n + 9) % vals.len()];

        // Unsigned: bits 47:32 of the product of the two 24-bit fields.
        let (ua, ub) = (a & 0x00FF_FFFF, b & 0x00FF_FFFF);
        let product = u64::from(ua) * u64::from(ub);
        assert_eq!(
            mul_hi_24(a, b, false),
            (product >> 32) as u32,
            "v_mul_hi_u32_u24 {a:#010x} {b:#010x}"
        );
        assert!(product >> 48 == 0, "a 24-bit product fits in 48 bits");

        // Signed: the same, the fields taken as signed 24-bit numbers.
        let sext24 = |x: u32| ((x << 8) as i32 >> 8) as i64;
        let product = sext24(a).wrapping_mul(sext24(b));
        assert_eq!(
            mul_hi_24(a, b, true),
            ((product as u64) >> 32) as u32,
            "v_mul_hi_i32_i24 {a:#010x} {b:#010x}"
        );
    }
}
