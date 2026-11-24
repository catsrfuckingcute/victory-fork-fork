//! A synthetic one-thread compute shader, built around a body of GFX10
//! instructions and lowered to DXIL. Shared by the test modules that check
//! one instruction at a time; the lowering needs no GPU and no `dxil.dll`,
//! so those tests run on any host.

use super::lower;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `op sdst, src0`.
pub(super) const fn sop1(op: u16, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | (op as u32 & 0xFF) << 8 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
pub(super) const fn sop2(op: u16, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SOPC `op src0, src1`, which sets SCC.
pub(super) const fn sopc(op: u16, src0: u8, src1: u8) -> u32 {
    0xBF00_0000 | (op as u32 & 0x7F) << 16 | (src1 as u32) << 8 | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0`.
pub(super) const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0`.
const fn buffer_store_dword(vdata: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

pub(super) const fn sgpr(n: u8) -> u16 {
    n as u16
}

/// Lowers a one-thread compute shader that takes 4 dwords of root constants
/// in `s0..s3`, runs `body`, and stores `v0` to a raw UAV.
pub(super) fn lower_body(body: &[u32]) -> Result<(), crate::ShaderError> {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(sop1(0x03, 5, 0x80)); // s_mov_b32 s5, 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8));
    words.push(S_ENDPGM);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .map(|_| ())
}

/// VOP2 `op vdst, src0, vsrc1`.
pub(super) const fn vop2(op: u16, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOPC `op src0, vsrc1`, which writes a lane mask.
pub(super) const fn vopc(op: u16, src0: u16, vsrc1: u8) -> u32 {
    0x7C00_0000 | (op as u32 & 0xFF) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
pub(super) const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// A VGPR as a VALU source.
pub(super) const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// `v_mov_b32 v<n>, s<k>`: a root constant into a VGPR, so the VALU
/// instructions under test work on VGPRs as real shaders do.
pub(super) const fn constant_to_vgpr(n: u8, k: u8) -> u32 {
    vop1(0x001, n, sgpr(k))
}
