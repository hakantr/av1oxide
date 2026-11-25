//! P51 (`F5a` — `F5-REFMVS` `6 / 6` kayıt = `3` yuva + `F5-PAL` `1` yuva) `aarch64` NEON
//! `load_tmvs`, `save_tmvs`, `splat_mv` ve `pal_idx_finish` bit-birebir doğruluk,
//! `SentinelBuffer` ve `OverReadProbe` testleri.

#![forbid(unsafe_code)]

#[cfg(target_arch = "aarch64")]
mod common;

use av1oxide::cpu::{CpuFlags, CpuLevel};
use av1oxide::dsp::{ARM_ISA_CANDIDATES, DspDispatch};

#[cfg(target_arch = "aarch64")]
use av1oxide::decode::refmvs::{Mv, RefBlock};
#[cfg(target_arch = "aarch64")]
use av1oxide::decode::refmvs_tmvs::{INVALID_REF2CUR, MfmvPlan, TemporalBlock};
#[cfg(target_arch = "aarch64")]
use common::sentinel::{OverReadProbe, SentinelBuffer};

#[test]
fn f5a_arm64_refmvs_ve_pal_aday_sozlesmesi() {
    for (family, field) in [
        ("refmvs", "load_tmvs"),
        ("refmvs", "save_tmvs"),
        ("refmvs", "splat_mv"),
        ("pal", "pal_idx_finish"),
    ] {
        assert!(
            ARM_ISA_CANDIDATES.iter().any(|c| {
                c.family == family
                    && c.field == field
                    && c.bit_depth == 0
                    && c.implementation == "neon"
                    && c.requires == CpuFlags::NEON
                    && c.default
            }),
            "eksik ARM64 adayı: {family}/{field}"
        );
    }

    let d_none = DspDispatch::select(CpuLevel::None.allowed());
    assert!(d_none.neon_token().is_none());
    assert!(d_none.dotprod_token().is_none());

    #[cfg(target_arch = "aarch64")]
    {
        let d_neon = DspDispatch::select(CpuFlags::detect());
        for (family, field) in [
            ("refmvs", "load_tmvs"),
            ("refmvs", "save_tmvs"),
            ("refmvs", "splat_mv"),
            ("pal", "pal_idx_finish"),
        ] {
            assert_eq!(
                d_none.independent_slot_implementation(family, field),
                "skaler"
            );
            assert_eq!(
                d_neon.independent_slot_implementation(family, field),
                "neon"
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn f5a_pal_idx_finish_neon_skaler_bit_exact_sentinel_ve_overread() {
    let d_neon = DspDispatch::select(CpuFlags::detect());
    let d_skaler = DspDispatch::select(CpuFlags::NONE);
    let neon_fn = d_neon.pal().pal_idx_finish;
    let skaler_fn = d_skaler.pal().pal_idx_finish;

    for &(bw, bh, w, h) in &[
        (4usize, 4usize, 4usize, 4usize),
        (8, 8, 8, 8),
        (16, 16, 12, 12),
        (32, 32, 20, 28),
        (64, 64, 44, 52),
    ] {
        // P52 (Apple M4 canlı koşusu): dav1d `pal_idx_finish` bütün `bw × bh` bloğu
        // yazar (sağ ve alt dolgu); sözleşme `src ≥ bw·bh`, `dst ≥ (bw/2)·bh`.
        let src_len = bw * bh;
        let dst_len = (bw / 2) * bh;

        let probe = OverReadProbe::new(src_len, 64, dst_len);
        let off = probe.data_offset();
        let res = probe
            .run(|inp, out| {
                let mut src_slice = vec![0u8; src_len];
                for (d, &s) in src_slice.iter_mut().zip(inp[off..off + src_len].iter()) {
                    *d = s & 7;
                }
                let mut sb = SentinelBuffer::with_defaults(dst_len);
                neon_fn(sb.data_mut(), &src_slice, bw, bh, w, h);
                sb.verify().expect("NEON pal_idx_finish sentinel ihlali");
                out.copy_from_slice(sb.data());
            })
            .expect("NEON pal_idx_finish OverReadProbe taşması");

        let mut src_ref = probe.payload();
        for b in &mut src_ref {
            *b &= 7;
        }
        let mut ref_out = vec![0u8; dst_len];
        skaler_fn(&mut ref_out, &src_ref, bw, bh, w, h);
        assert_eq!(
            res, ref_out,
            "pal_idx_finish farkı: bw={bw} bh={bh} w={w} h={h}"
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn f5a_refmvs_splat_save_load_neon_skaler_bit_exact() {
    let d_neon = DspDispatch::select(CpuFlags::detect());
    let d_skaler = DspDispatch::select(CpuFlags::NONE);

    // 1. splat_mv
    let rmv = RefBlock {
        mv: [Mv { y: -240, x: 512 }, Mv { y: 128, x: -320 }],
        ref_pair: [1, 4],
        bs: 9,
        mf: 2,
    };
    for bw4 in [1usize, 2, 4, 8, 16, 32] {
        let iw4 = 40usize;
        let bh4 = 6usize;
        let len = 12 * iw4;
        let sentinel = RefBlock {
            mv: [Mv { y: 7777, x: -7777 }, Mv { y: -5555, x: 5555 }],
            ref_pair: [6, 7],
            bs: 21,
            mf: 3,
        };
        let mut g_neon = vec![sentinel; len];
        let mut g_skaler = vec![sentinel; len];
        (d_neon.refmvs().splat_mv)(&mut g_neon, iw4, rmv, 2, 2, bw4, bh4);
        (d_skaler.refmvs().splat_mv)(&mut g_skaler, iw4, rmv, 2, 2, bw4, bh4);
        assert_eq!(g_neon, g_skaler, "splat_mv farkı (bw4={bw4})");
    }

    // 2. save_tmvs
    let iw4 = 32usize;
    let ih4 = 16usize;
    let mut grid = vec![
        RefBlock {
            mv: [Mv::ZERO; 2],
            ref_pair: [0, 0],
            bs: 12,
            mf: 0,
        };
        iw4 * ih4
    ];
    for (i, cell) in grid.iter_mut().enumerate() {
        cell.mv[0] = Mv {
            y: ((i as i32 * 37) % 1024) as i16 - 512,
            x: ((i as i32 * 53) % 1024) as i16 - 512,
        };
        cell.mv[1] = Mv {
            y: ((i as i32 * 71) % 8192) as i16 - 4096,
            x: ((i as i32 * 29) % 1024) as i16 - 512,
        };
        cell.ref_pair = [((i % 7) + 1) as i8, (((i + 3) % 7) + 1) as i8];
        cell.bs = (i % 22) as u8;
    }
    let ref_sign = [1u8, 0, 1, 1, 0, 1, 0];
    let stride = 16usize;
    let mut rp_neon = vec![
        TemporalBlock {
            mv: Mv { y: 999, x: -999 },
            ref_: 99,
        };
        stride * 8
    ];
    let mut rp_skaler = rp_neon.clone();
    (d_neon.refmvs().save_tmvs)(&mut rp_neon, stride, &grid, iw4, &ref_sign, 16, 8, 0, 0);
    (d_skaler.refmvs().save_tmvs)(&mut rp_skaler, stride, &grid, iw4, &ref_sign, 16, 8, 0, 0);
    assert_eq!(rp_neon, rp_skaler, "save_tmvs farkı");

    // 3. load_tmvs
    let iw8 = 16usize;
    let ih8 = 8usize;
    let plan = MfmvPlan {
        n_mfmvs: 2,
        mfmv_ref: [0, 2, 0],
        mfmv_ref2cur: [4, -2, INVALID_REF2CUR],
        mfmv_ref2ref: [[1, 12, 2, 13, 3, 14, 1]; 3], // P52: `den` 1..=31 (mv_projection sözleşmesi)
    };
    let refs: [Option<&[TemporalBlock]>; 7] = [
        Some(&rp_skaler),
        None,
        Some(&rp_skaler),
        None,
        None,
        None,
        None,
    ];
    let mut proj_neon = vec![
        TemporalBlock {
            mv: Mv { y: 1234, x: -1234 },
            ref_: 77,
        };
        iw8 * ih8
    ];
    let mut proj_skaler = proj_neon.clone();
    (d_neon.refmvs().load_tmvs)(
        &mut proj_neon,
        iw8,
        iw8,
        ih8,
        &plan,
        &refs,
        0,
        0,
        iw8,
        0,
        ih8,
    );
    (d_skaler.refmvs().load_tmvs)(
        &mut proj_skaler,
        iw8,
        iw8,
        ih8,
        &plan,
        &refs,
        0,
        0,
        iw8,
        0,
        ih8,
    );
    assert_eq!(proj_neon, proj_skaler, "load_tmvs farkı");
}
