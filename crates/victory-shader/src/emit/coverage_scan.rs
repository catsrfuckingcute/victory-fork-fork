//! A coverage scan over the opcode tables: how many opcodes the lowering
//! handles, and which ones it refuses.
//!
//! Lowering a one-instruction shader for every opcode in [`crate::gfx10`]
//! gives a number to compare between changes, and names what is left. It
//! is `#[ignore]`d because it is a report rather than a check:
//!
//! ```text
//! cargo test -p victory-shader --lib coverage_scan -- --ignored --nocapture
//! ```
//!
//! Read the output with the probe in mind: it places one instruction in an
//! otherwise empty shader, so an opcode that needs something set up before
//! it is refused for that reason rather than for want of a lowering. The
//! three to expect are a mask in SCC (`s_cselect`, `s_cmov`, `s_addc_u32`),
//! a lane mask the structure pass has tracked (`v_cndmask_b32`, the
//! carry-in forms, the `saveexec` family) and an accumulator in the
//! destination. The forms carrying a 32-bit literal (`v_fmaak_f32` and
//! the like) are refused for the same reason: the probe does not append
//! one. Those are grouped last and counted apart.

use super::synth::{lower_body, sgpr, sop1, sop2, vgpr, vop1, vop2, vop3, vopc};
use crate::gfx10::{Encoding, Instruction};
use std::collections::BTreeMap;

/// AMD's name for an opcode, if it is in our tables.
fn name_of(encoding: Encoding, opcode: u16) -> Option<&'static str> {
    Instruction {
        offset: 0,
        encoding,
        opcode,
        words: vec![],
    }
    .name()
}

/// The reasons that mean "something was missing before this instruction",
/// not "this opcode has no lowering".
const SETUP: &[&str] = &[
    "SCC on entry is not supported",
    "a carry-in from an untracked mask is not supported",
    "a select on an untracked mask is not supported",
    "a lane mask holding lanes elsewhere read as data is not supported",
    "an interpolated attribute the stage data does not list is not supported",
    "an accumulator is not supported",
];

/// Opcodes that carry a 32-bit literal in the instruction, which the probe
/// does not append, so they are refused for want of that operand.
const NEEDS_LITERAL: &[&str] = &[
    "v_fmaak_f32",
    "v_fmamk_f32",
    "v_fmaak_f16",
    "v_fmamk_f16",
    "v_madak_f32",
    "v_madmk_f32",
    "v_madak_f16",
    "v_madmk_f16",
];

/// Whether a refusal is the probe's doing rather than a missing lowering.
fn is_setup(reason: &str, names: &[&str]) -> bool {
    SETUP.contains(&reason) || names.iter().all(|n| NEEDS_LITERAL.contains(n))
}

#[test]
#[ignore = "a report, not a check"]
fn report_opcode_coverage() {
    let mut ok = 0usize;
    let mut refused: Vec<(&'static str, String)> = Vec::new();

    let mut probe = |encoding: Encoding, opcode: u16, body: Vec<u32>| {
        let Some(name) = name_of(encoding, opcode) else {
            return;
        };
        match lower_body(&body) {
            Ok(()) => ok += 1,
            Err(e) => refused.push((name, e.to_string())),
        }
    };

    // One VALU source, writing v0; the sources hold root constants.
    let valu = |body: &[u32]| {
        let mut v = vec![
            vop1(0x001, 1, sgpr(0)),
            vop1(0x001, 2, sgpr(1)),
            vop1(0x001, 3, sgpr(2)),
        ];
        v.extend_from_slice(body);
        v
    };
    for op in 0..256u16 {
        probe(Encoding::Vop1, op, valu(&[vop1(op, 0, vgpr(1))]));
    }
    for op in 0..64u16 {
        probe(Encoding::Vop2, op, valu(&[vop2(op, 0, vgpr(1), 2)]));
    }
    for op in 0..256u16 {
        // VOPC writes a lane mask, so the shader stores nothing of it.
        probe(Encoding::Vopc, op, valu(&[vopc(op, vgpr(1), 2)]));
    }
    for op in 0..1024u16 {
        let i = vop3(op, 0, vgpr(1), vgpr(2), vgpr(3));
        probe(Encoding::Vop3, op, valu(&[i[0], i[1]]));
    }
    // Scalar: the result moved into v0 so the shader stores it.
    for op in 0..256u16 {
        probe(
            Encoding::Sop1,
            op,
            vec![sop1(op, 20, 1), vop1(0x001, 0, sgpr(20))],
        );
    }
    for op in 0..128u16 {
        probe(
            Encoding::Sop2,
            op,
            vec![sop2(op, 20, 1, 2), vop1(0x001, 0, sgpr(20))],
        );
    }

    let mut by_reason: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (name, reason) in &refused {
        // The reason without the offset the error carries.
        let short = reason.rsplit(": ").next().unwrap_or(reason);
        by_reason.entry(short).or_default().push(name);
    }
    let setup: usize = by_reason
        .iter()
        .filter(|(r, n)| is_setup(r, n))
        .map(|(_, n)| n.len())
        .sum();

    eprintln!(
        "\nlowered: {ok}    refused: {}  (of which {setup} want something set up first)\n",
        refused.len()
    );
    for (reason, names) in by_reason.iter().filter(|(r, n)| !is_setup(r, n)) {
        let mut names = names.clone();
        names.sort_unstable();
        names.dedup();
        eprintln!("{} ({})\n    {}\n", reason, names.len(), names.join(" "));
    }
    eprintln!("--- these want something set up before the instruction ---");
    for (reason, names) in by_reason.iter().filter(|(r, n)| is_setup(r, n)) {
        let mut names = names.clone();
        names.sort_unstable();
        names.dedup();
        eprintln!("{} ({})\n    {}\n", reason, names.len(), names.join(" "));
    }
}
