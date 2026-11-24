//! Synthetic GFX10 16-bit integer VALU shaders, lowered to DXIL.
//!
//! As in `salu_tests` and `valu_int_tests`: one test lowers a shader for
//! each opcode this module covers, and the rest check the arithmetic the
//! lowerings emit against plain implementations of the same hardware
//! semantics. No GPU and no `dxil.dll` needed.

use super::synth::{constant_to_vgpr, lower_body, vgpr, vop3};

fn prologue() -> Vec<u32> {
    vec![
        constant_to_vgpr(1, 0),
        constant_to_vgpr(2, 1),
        constant_to_vgpr(3, 2),
    ]
}

#[test]
fn every_covered_16_bit_opcode_lowers() {
    let ops: &[(&str, u16)] = &[
        ("v_add_nc_i16", 0x30d),
        ("v_sub_nc_i16", 0x30e),
        ("v_mad_u16", 0x340),
        ("v_mad_i16", 0x35e),
        ("v_min3_u16", 0x353),
        ("v_min3_i16", 0x352),
        ("v_max3_u16", 0x356),
        ("v_max3_i16", 0x355),
        ("v_med3_u16", 0x359),
        ("v_med3_i16", 0x358),
        ("v_mad_i32_i16", 0x375),
        ("v_mad_u32_u16", 0x373),
    ];
    for &(name, op) in ops {
        let mut body = prologue();
        body.extend_from_slice(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        assert_eq!(lower_body(&body), Ok(()), "{name}");
    }
}

/// A spread of 16-bit values: the edges, then a deterministic scatter.
fn values() -> Vec<u32> {
    let mut v = vec![0, 1, 2, 0x7F, 0x80, 0xFF, 0x7FFF, 0x8000, 0xFFFE, 0xFFFF];
    let mut x = 0x1234_5678u32;
    for _ in 0..120 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        v.push(x & 0xFFFF);
    }
    v
}

/// The sign extension the lowering applies to a 16-bit source whose
/// opcode name ends in `_i16`.
fn sext16(x: u32) -> u32 {
    ((x << 16) as i32 >> 16) as u32
}

/// What the lowering of the three-way 16-bit forms computes. The operands
/// arrive in the low 16 bits of a dword, sign-extended for the signed
/// forms, and `write_half` keeps the low 16 bits of the result.
fn three(name: &str, a: u32, b: u32, d: u32) -> u32 {
    let signed = name.ends_with("_i16");
    let (a, b, d) = if signed {
        (sext16(a), sext16(b), sext16(d))
    } else {
        (a, b, d)
    };
    let pick = |p: bool, x: u32, y: u32| if p { x } else { y };
    let lt = |x: u32, y: u32| {
        if signed {
            (x as i32) < (y as i32)
        } else {
            x < y
        }
    };
    let gt = |x: u32, y: u32| {
        if signed {
            (x as i32) > (y as i32)
        } else {
            x > y
        }
    };
    let r = if name.starts_with("v_mad") {
        a.wrapping_mul(b).wrapping_add(d)
    } else {
        match &name[2..6] {
            "min3" => {
                let t = pick(lt(a, b), a, b);
                pick(lt(t, d), t, d)
            }
            "max3" => {
                let t = pick(gt(a, b), a, b);
                pick(gt(t, d), t, d)
            }
            _ => {
                let lo = pick(lt(a, b), a, b);
                let hi = pick(gt(a, b), a, b);
                let t = pick(lt(hi, d), hi, d);
                pick(gt(lo, t), lo, t)
            }
        }
    };
    r & 0xFFFF
}

#[test]
fn the_16_bit_three_way_results_match_the_hardware_definition() {
    let vals = values();
    for (n, &a) in vals.iter().enumerate() {
        let b = vals[(n + 7) % vals.len()];
        let d = vals[(n + 17) % vals.len()];

        // Unsigned: the three values compared as they are.
        let mut u = [a, b, d];
        u.sort_unstable();
        assert_eq!(three("v_min3_u16", a, b, d), u[0], "v_min3_u16");
        assert_eq!(three("v_med3_u16", a, b, d), u[1], "v_med3_u16");
        assert_eq!(three("v_max3_u16", a, b, d), u[2], "v_max3_u16");

        // Signed: the same, read as signed 16-bit numbers.
        let mut i = [a as u16 as i16, b as u16 as i16, d as u16 as i16];
        i.sort_unstable();
        let as_u = |x: i16| u32::from(x as u16);
        assert_eq!(three("v_min3_i16", a, b, d), as_u(i[0]), "v_min3_i16");
        assert_eq!(three("v_med3_i16", a, b, d), as_u(i[1]), "v_med3_i16");
        assert_eq!(three("v_max3_i16", a, b, d), as_u(i[2]), "v_max3_i16");

        // The multiply-adds keep the low 16 bits, so the signed and
        // unsigned forms are the same bits.
        let want = (a as u16).wrapping_mul(b as u16).wrapping_add(d as u16);
        assert_eq!(three("v_mad_u16", a, b, d), u32::from(want), "v_mad_u16");
        assert_eq!(three("v_mad_i16", a, b, d), u32::from(want), "v_mad_i16");
    }
}

/// What the lowering of `v_mad_i32_i16`/`v_mad_u32_u16` computes: 16-bit
/// sources, a 32-bit product and addend.
fn mad32(a: u32, b: u32, d: u32, signed: bool) -> u32 {
    let (a, b) = if signed {
        (sext16(a), sext16(b))
    } else {
        (a, b)
    };
    a.wrapping_mul(b).wrapping_add(d)
}

#[test]
fn the_16_bit_multiply_adds_match_the_hardware_definition() {
    let vals = values();
    for (n, &a) in vals.iter().enumerate() {
        let b = vals[(n + 5) % vals.len()];
        let d = vals[(n + 13) % vals.len()] | 0xABCD_0000;

        // Unsigned: a 16x16 product, which cannot overflow 32 bits.
        let product = u32::from(a as u16) * u32::from(b as u16);
        assert_eq!(
            mad32(a, b, d, false),
            product.wrapping_add(d),
            "v_mad_u32_u16 {a:#06x} {b:#06x}"
        );

        // Signed: the sources taken as signed 16-bit numbers.
        let product = i32::from(a as u16 as i16) * i32::from(b as u16 as i16);
        assert_eq!(
            mad32(a, b, d, true),
            (product as u32).wrapping_add(d),
            "v_mad_i32_i16 {a:#06x} {b:#06x}"
        );
    }
}
