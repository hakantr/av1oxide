//! `aarch64` NEON `refmvs` (`load_tmvs`, `save_tmvs`, `splat_mv`) çekirdekleri (`F5-REFMVS` `6 / 6` kayıt = `3` yuva; `P51` / `F5a`).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Sembol | Matris Kaydı | Kaynak |
//! |---|---|---|
//! | `dav1d_load_tmvs_neon` | `D00973` (`8` bpc), `D00974` (`16` bpc) | `src/arm/64/refmvs.S:247-445`, `src/arm/refmvs.h:35-46` |
//! | `dav1d_save_tmvs_neon` | `D00975` (`8` bpc), `D00976` (`16` bpc) | `src/arm/64/refmvs.S:114-245`, `src/arm/refmvs.h:36-47` |
//! | `dav1d_splat_mv_neon` | `D00977` (`8` bpc), `D00978` (`16` bpc) | `src/arm/64/refmvs.S:30-112`, `src/arm/refmvs.h:37-48` |
//!
//! ## Güvenlik ve ISA Sınırı (`GUV-02`, `S3ze`)
//!
//! - Bu modül `#![forbid(unsafe_code)]` taşır; tüm bellek yükleme/saklama
//!   işlemleri [`super::mem`] içindeki sabit boyutlu dizi yardımcılarıyla yapılır.
//! - Kopyasız O(1) sözleşme: girişte [`load_tmvs_sozlesmesi`], [`save_tmvs_sozlesmesi`]
//!   ve [`splat_mv_sozlesmesi`] denetlenir; sözleşme dışı girdide ilgili
//!   `*_skaler_devret` yardımcısıyla skaler gövdeye devredilir.
//! - Giriş işaretçileri yalnız [`NeonToken`] ile alınabilir ([`load_tmvs_neon_fn`],
//!   [`save_tmvs_neon_fn`], [`splat_mv_neon_fn`]).
//! - Sıcak döngülerde `get`/`get_mut` + `first_chunk` kullanılarak
//!   `panic_bounds_check` üretimi sıfırlanmıştır.

#![forbid(unsafe_code)]

use core::arch::aarch64::{
    uint8x16_t, vabs_s16, vabsq_s32, vaddq_s32, vdup_n_u64, vdupq_n_s32, vget_lane_u64,
    vget_low_u8, vminq_s32, vmulq_s32, vqtbl1q_u8, vreinterpret_s16_u64, vreinterpret_u16_s16,
    vreinterpret_u64_u16, vshr_n_u16, vshrq_n_s32,
};

use crate::cpu::NeonToken;
use crate::decode::refmvs::{Mv, RefBlock};
use crate::decode::refmvs_tmvs::{INVALID_REF2CUR, MfmvPlan, TemporalBlock};
use crate::dsp::arm::mem;
use crate::dsp::refmvs::{
    LoadTmvsFn, SaveTmvsFn, SplatMvFn, load_tmvs_skaler_devret, load_tmvs_sozlesmesi,
    save_tmvs_skaler_devret, save_tmvs_sozlesmesi, splat_mv_skaler_devret, splat_mv_sozlesmesi,
};

/// `BLOCK_DIMENSIONS[bs][0]` (`4×4` birimi) → `log2(((bw4 + 1) >> 1).max(1))` (`0..=4`) tablosu (`32` öğeye dolgulu).
const BW8_SHIFT_TAB: [u8; 32] = [
    4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// `mv_proj` tablosu (`src/refmvs.c:177-181`).
const MV_PROJ: [i32; 32] = [
    0, 16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638, 1489, 1365, 1260, 1170, 1092,
    1024, 963, 910, 862, 819, 780, 744, 712, 682, 655, 630, 606, 585, 564, 546, 528,
];

/// `refmvs/load_tmvs` ARM64 NEON yuva işaretçisi (`D00973` / `D00974`).
#[must_use]
pub const fn load_tmvs_neon_fn(_token: NeonToken) -> LoadTmvsFn {
    mem::load_tmvs_entry
}

/// `refmvs/save_tmvs` ARM64 NEON yuva işaretçisi (`D00975` / `D00976`).
#[must_use]
pub const fn save_tmvs_neon_fn(_token: NeonToken) -> SaveTmvsFn {
    mem::save_tmvs_entry
}

/// `refmvs/splat_mv` ARM64 NEON yuva işaretçisi (`D00977` / `D00978`).
#[must_use]
pub const fn splat_mv_neon_fn(_token: NeonToken) -> SplatMvFn {
    mem::splat_mv_entry
}

// ============================================================================
// 1. `splat_mv` (`D00977` / `D00978`)
// ============================================================================

#[target_feature(enable = "neon")]
pub(super) fn splat_mv_dispatch_neon(
    grid: &mut [RefBlock],
    iw4: usize,
    rmv: RefBlock,
    bx4: usize,
    by4: usize,
    bw4: usize,
    bh4: usize,
) {
    if !splat_mv_sozlesmesi(grid.len(), iw4, bx4, by4, bw4, bh4) {
        splat_mv_skaler_devret(grid, iw4, rmv, bx4, by4, bw4, bh4);
        return;
    }
    if bw4 == 0 || bh4 == 0 {
        return;
    }
    if let Some(sub) = grid.get_mut(by4 * iw4 + bx4..) {
        splat_mv_neon(sub, iw4, rmv, bw4, bh4);
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn splat_neon_triple(rmv: RefBlock) -> (uint8x16_t, uint8x16_t, uint8x16_t) {
    let d0 = rmv.mv[0].n().to_le_bytes();
    let d1 = rmv.mv[1].n().to_le_bytes();
    let d2 = [rmv.ref_pair[0] as u8, rmv.ref_pair[1] as u8, rmv.bs, rmv.mf];

    let b0 = [
        d0[0], d0[1], d0[2], d0[3], d1[0], d1[1], d1[2], d1[3], d2[0], d2[1], d2[2], d2[3], d0[0],
        d0[1], d0[2], d0[3],
    ];
    let b1 = [
        d1[0], d1[1], d1[2], d1[3], d2[0], d2[1], d2[2], d2[3], d0[0], d0[1], d0[2], d0[3], d1[0],
        d1[1], d1[2], d1[3],
    ];
    let b2 = [
        d2[0], d2[1], d2[2], d2[3], d0[0], d0[1], d0[2], d0[3], d1[0], d1[1], d1[2], d1[3], d2[0],
        d2[1], d2[2], d2[3],
    ];
    (
        mem::load_u8x16(&b0),
        mem::load_u8x16(&b1),
        mem::load_u8x16(&b2),
    )
}

#[target_feature(enable = "neon")]
fn splat_mv_neon(grid: &mut [RefBlock], iw4: usize, rmv: RefBlock, bw4: usize, bh4: usize) {
    let mut off = 0usize;
    match bw4 {
        1 => {
            for _ in 0..bh4 {
                if let Some(cell) = grid.get_mut(off) {
                    *cell = rmv;
                }
                off += iw4;
            }
        }
        2 => {
            for _ in 0..bh4 {
                if let Some(dst2) = grid.get_mut(off..).and_then(|s| s.first_chunk_mut::<2>()) {
                    dst2[0] = rmv;
                    dst2[1] = rmv;
                }
                off += iw4;
            }
        }
        4 => {
            let (m0, m1, m2) = splat_neon_triple(rmv);
            for _ in 0..bh4 {
                if let Some(dst4) = grid.get_mut(off..).and_then(|s| s.first_chunk_mut::<4>()) {
                    mem::store_refblock_x4(dst4, m0, m1, m2);
                }
                off += iw4;
            }
        }
        8 => {
            let (m0, m1, m2) = splat_neon_triple(rmv);
            for _ in 0..bh4 {
                if let Some(dst8) = grid.get_mut(off..).and_then(|s| s.first_chunk_mut::<8>()) {
                    let (chunks, _) = dst8.as_chunks_mut::<4>();
                    for c in chunks {
                        mem::store_refblock_x4(c, m0, m1, m2);
                    }
                }
                off += iw4;
            }
        }
        16 => {
            let (m0, m1, m2) = splat_neon_triple(rmv);
            for _ in 0..bh4 {
                if let Some(dst16) = grid.get_mut(off..).and_then(|s| s.first_chunk_mut::<16>()) {
                    let (chunks, _) = dst16.as_chunks_mut::<4>();
                    for c in chunks {
                        mem::store_refblock_x4(c, m0, m1, m2);
                    }
                }
                off += iw4;
            }
        }
        32 => {
            let (m0, m1, m2) = splat_neon_triple(rmv);
            for _ in 0..bh4 {
                if let Some(dst32) = grid.get_mut(off..).and_then(|s| s.first_chunk_mut::<32>()) {
                    let (chunks, _) = dst32.as_chunks_mut::<4>();
                    for c in chunks {
                        mem::store_refblock_x4(c, m0, m1, m2);
                    }
                }
                off += iw4;
            }
        }
        _ => {
            let (m0, m1, m2) = splat_neon_triple(rmv);
            for _ in 0..bh4 {
                if let Some(row) = grid.get_mut(off..off + bw4) {
                    let (chunks, rem) = row.as_chunks_mut::<4>();
                    for c in chunks {
                        mem::store_refblock_x4(c, m0, m1, m2);
                    }
                    for slot in rem {
                        *slot = rmv;
                    }
                }
                off += iw4;
            }
        }
    }
}

// ============================================================================
// 2. `save_tmvs` (`D00975` / `D00976`)
// ============================================================================

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(super) fn save_tmvs_dispatch_neon(
    rp: &mut [TemporalBlock],
    stride: usize,
    grid: &[RefBlock],
    iw4: usize,
    ref_sign: &[u8; 7],
    col_end8: usize,
    row_end8: usize,
    col_start8: usize,
    row_start8: usize,
) {
    if !save_tmvs_sozlesmesi(
        rp.len(),
        stride,
        grid.len(),
        iw4,
        col_end8,
        row_end8,
        col_start8,
        row_start8,
    ) {
        save_tmvs_skaler_devret(
            rp, stride, grid, iw4, ref_sign, col_end8, row_end8, col_start8, row_start8,
        );
        return;
    }
    save_tmvs_neon(
        rp, stride, grid, iw4, ref_sign, col_end8, row_end8, col_start8, row_start8,
    );
}

#[target_feature(enable = "neon")]
#[inline]
fn select_temporal_candidate_neon(b: RefBlock, sign_lut: &[u8; 8]) -> TemporalBlock {
    // Her iki MV'nin (`mv[0]` ve `mv[1]`) mutlak değerini NEON `vabs_s16` + `vshr_n_u16::<12>` ile denetle.
    let q = (u64::from(b.mv[1].n() as u32) << 32) | u64::from(b.mv[0].n() as u32);
    let v_mvs = vreinterpret_s16_u64(vdup_n_u64(q));
    let v_abs_hi = vshr_n_u16::<12>(vreinterpret_u16_s16(vabs_s16(v_mvs)));
    let flags = vget_lane_u64::<0>(vreinterpret_u64_u16(v_abs_hi));

    let r1 = b.ref_pair[1];
    if r1 > 0 && (r1 as usize) <= 7 && sign_lut[(r1 as usize) & 7] != 0 && (flags >> 32) == 0 {
        return TemporalBlock {
            mv: b.mv[1],
            ref_: r1 as u8,
        };
    }
    let r0 = b.ref_pair[0];
    if r0 > 0
        && (r0 as usize) <= 7
        && sign_lut[(r0 as usize) & 7] != 0
        && (flags & 0xffff_ffff) == 0
    {
        return TemporalBlock {
            mv: b.mv[0],
            ref_: r0 as u8,
        };
    }
    TemporalBlock {
        mv: Mv::ZERO,
        ref_: 0,
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn write_temporal_neon(
    rp: &mut [TemporalBlock],
    start: usize,
    lg: u8,
    cand: TemporalBlock,
    shuf0: uint8x16_t,
    shuf1: uint8x16_t,
    shuf2: uint8x16_t,
) {
    // P52 (Apple M4 canlı koşusu): blok satır/dilim sonunu aşıyorsa skaler gövde
    // gibi yalnız sığan kayıtlar yazılır (`rp.get_mut` sınır denetimli kısmi yazım);
    // önceki sürüm bu durumda hiçbir şey yazmıyordu (rastgele `bs` ızgarasında fark).
    let n = 1usize << lg;
    if start.checked_add(n).is_none_or(|son| son > rp.len()) {
        for k in 0..n {
            if let Some(cell) = rp.get_mut(start + k) {
                *cell = cand;
            }
        }
        return;
    }
    match lg {
        0 => {
            if let Some(cell) = rp.get_mut(start) {
                *cell = cand;
            }
        }
        1 => {
            if let Some(dst2) = rp.get_mut(start..).and_then(|s| s.first_chunk_mut::<2>()) {
                dst2[0] = cand;
                dst2[1] = cand;
            }
        }
        _ => {
            let y_le = cand.mv.y.to_le_bytes();
            let x_le = cand.mv.x.to_le_bytes();
            let b16 = [
                y_le[0], y_le[1], x_le[0], x_le[1], cand.ref_, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ];
            let v = mem::load_u8x16(&b16);
            let v0 = vqtbl1q_u8(v, shuf0);
            let v1 = vqtbl1q_u8(v, shuf1);
            if lg == 2 {
                if let Some(dst4) = rp.get_mut(start..).and_then(|s| s.first_chunk_mut::<4>()) {
                    mem::store_temporal_x4(dst4, v0, vget_low_u8(v1));
                }
            } else if lg == 3 {
                let v2 = vqtbl1q_u8(v, shuf2);
                if let Some(dst8) = rp.get_mut(start..).and_then(|s| s.first_chunk_mut::<8>()) {
                    mem::store_temporal_x8(dst8, v0, v1, v2);
                }
            } else {
                let v2 = vqtbl1q_u8(v, shuf2);
                if let Some(dst16) = rp.get_mut(start..).and_then(|s| s.first_chunk_mut::<16>()) {
                    let (chunks, _) = dst16.as_chunks_mut::<8>();
                    for c in chunks {
                        mem::store_temporal_x8(c, v0, v1, v2);
                    }
                }
            }
        }
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn save_tmvs_neon(
    rp: &mut [TemporalBlock],
    stride: usize,
    grid: &[RefBlock],
    iw4: usize,
    ref_sign: &[u8; 7],
    col_end8: usize,
    row_end8: usize,
    col_start8: usize,
    row_start8: usize,
) {
    let sign_lut: [u8; 8] = [
        0,
        ref_sign[0],
        ref_sign[1],
        ref_sign[2],
        ref_sign[3],
        ref_sign[4],
        ref_sign[5],
        ref_sign[6],
    ];
    let shuf0 = mem::load_u8x16(&[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3]);
    let shuf1 = mem::load_u8x16(&[4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1]);
    let shuf2 = mem::load_u8x16(&[2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]);

    for y in row_start8..row_end8 {
        let row4 = y * 2 + 1;
        let base = row4 * iw4;
        let rp_row = (y - row_start8) * stride;
        let mut x = col_start8;
        while x < col_end8 {
            let Some(&b0) = grid.get(base + x * 2 + 1) else {
                break;
            };
            let lg0 = BW8_SHIFT_TAB[usize::from(b0.bs) & 31];
            let step = 1usize << lg0;
            let cand = select_temporal_candidate_neon(b0, &sign_lut);
            write_temporal_neon(rp, rp_row + x, lg0, cand, shuf0, shuf1, shuf2);
            x += step;
        }
    }
}

// ============================================================================
// 3. `load_tmvs` (`D00973` / `D00974`)
// ============================================================================

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(super) fn load_tmvs_dispatch_neon(
    rp_proj: &mut [TemporalBlock],
    stride: usize,
    iw8: usize,
    ih8: usize,
    plan: &MfmvPlan,
    refs: &[Option<&[TemporalBlock]>; 7],
    tile_row_idx: usize,
    col_start8: usize,
    col_end8: usize,
    row_start8: usize,
    row_end8: usize,
) {
    if !load_tmvs_sozlesmesi(
        rp_proj.len(),
        stride,
        iw8,
        ih8,
        plan,
        refs,
        tile_row_idx,
        col_start8,
        col_end8,
        row_start8,
        row_end8,
    ) {
        load_tmvs_skaler_devret(
            rp_proj,
            stride,
            iw8,
            ih8,
            plan,
            refs,
            tile_row_idx,
            col_start8,
            col_end8,
            row_start8,
            row_end8,
        );
        return;
    }
    load_tmvs_neon(
        rp_proj,
        stride,
        iw8,
        ih8,
        plan,
        refs,
        tile_row_idx,
        col_start8,
        col_end8,
        row_start8,
        row_end8,
    );
}

#[target_feature(enable = "neon")]
#[inline]
fn precomp_frac_neon(ref2cur: i32) -> [i32; 32] {
    let v_r2c = vdupq_n_s32(ref2cur);
    let mut out = [0i32; 32];
    for (g, chunk4) in MV_PROJ.as_chunks::<4>().0.iter().enumerate() {
        let b = [
            chunk4[0].to_le_bytes()[0],
            chunk4[0].to_le_bytes()[1],
            chunk4[0].to_le_bytes()[2],
            chunk4[0].to_le_bytes()[3],
            chunk4[1].to_le_bytes()[0],
            chunk4[1].to_le_bytes()[1],
            chunk4[1].to_le_bytes()[2],
            chunk4[1].to_le_bytes()[3],
            chunk4[2].to_le_bytes()[0],
            chunk4[2].to_le_bytes()[1],
            chunk4[2].to_le_bytes()[2],
            chunk4[2].to_le_bytes()[3],
            chunk4[3].to_le_bytes()[0],
            chunk4[3].to_le_bytes()[1],
            chunk4[3].to_le_bytes()[2],
            chunk4[3].to_le_bytes()[3],
        ];
        let v_in = core::arch::aarch64::vreinterpretq_s32_u8(mem::load_u8x16(&b));
        let v_prod = vmulq_s32(v_in, v_r2c);
        let mut prod_bytes = [0u8; 16];
        mem::store_u8x16(
            &mut prod_bytes,
            core::arch::aarch64::vreinterpretq_u8_s32(v_prod),
        );
        let base = (g * 4) & 28;
        for (i, c4) in prod_bytes.as_chunks::<4>().0.iter().enumerate() {
            out[(base + i) & 31] = i32::from_le_bytes(*c4);
        }
    }
    out
}

#[target_feature(enable = "neon")]
#[inline]
fn project_mv_offset_neon(mv_y: i16, mv_x: i16, frac: i32, ref_sign: i32) -> (i32, i32) {
    let v_mv_bytes = {
        let y_le = i32::from(mv_y).to_le_bytes();
        let x_le = i32::from(mv_x).to_le_bytes();
        [
            y_le[0], y_le[1], y_le[2], y_le[3], x_le[0], x_le[1], x_le[2], x_le[3], 0, 0, 0, 0, 0,
            0, 0, 0,
        ]
    };
    let v_mv = core::arch::aarch64::vreinterpretq_s32_u8(mem::load_u8x16(&v_mv_bytes));
    let xy = vmulq_s32(v_mv, vdupq_n_s32(frac));
    let sign_bias = vshrq_n_s32::<31>(xy);
    let proj = vshrq_n_s32::<14>(vaddq_s32(vaddq_s32(xy, sign_bias), vdupq_n_s32(8192)));
    let mag = vminq_s32(vshrq_n_s32::<6>(vabsq_s32(proj)), vdupq_n_s32(255));

    let mut proj_b = [0u8; 16];
    let mut mag_b = [0u8; 16];
    mem::store_u8x16(&mut proj_b, core::arch::aarch64::vreinterpretq_u8_s32(proj));
    mem::store_u8x16(&mut mag_b, core::arch::aarch64::vreinterpretq_u8_s32(mag));

    let py = i32::from_le_bytes([proj_b[0], proj_b[1], proj_b[2], proj_b[3]]);
    let px = i32::from_le_bytes([proj_b[4], proj_b[5], proj_b[6], proj_b[7]]);
    let my = i32::from_le_bytes([mag_b[0], mag_b[1], mag_b[2], mag_b[3]]);
    let mx = i32::from_le_bytes([mag_b[4], mag_b[5], mag_b[6], mag_b[7]]);

    let off_y = if (py ^ ref_sign) < 0 { -my } else { my };
    let off_x = if (px ^ ref_sign) < 0 { -mx } else { mx };
    (off_y, off_x)
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn load_tmvs_neon(
    rp_proj: &mut [TemporalBlock],
    stride: usize,
    iw8: usize,
    ih8: usize,
    plan: &MfmvPlan,
    refs: &[Option<&[TemporalBlock]>; 7],
    tile_row_idx: usize,
    col_start8: usize,
    col_end8: usize,
    row_start8: usize,
    row_end8: usize,
) {
    let row_end8 = row_end8.min(ih8);
    if row_end8 <= row_start8 {
        return;
    }
    let col_start8i = col_start8.saturating_sub(8);
    let col_end8i = (col_end8 + 8).min(iw8);
    let base = 16 * stride * tile_row_idx;
    let row_off = (row_start8 & 15) * stride;

    for y in row_start8..row_end8 {
        let off = base + row_off + (y - row_start8) * stride;
        if let Some(row_slice) = rp_proj.get_mut(off + col_start8..off + col_end8) {
            for slot in row_slice {
                slot.mv = Mv::INVALID;
            }
        }
    }

    for n in 0..plan.n_mfmvs.min(3) {
        let ref2cur = plan.mfmv_ref2cur[n & 3];
        if ref2cur == INVALID_REF2CUR {
            continue;
        }
        let ref_idx = usize::from(plan.mfmv_ref[n & 3]);
        let Some(Some(r)) = refs.get(ref_idx) else {
            continue;
        };
        let r = *r;
        let fracs = precomp_frac_neon(i32::from(ref2cur));
        let ref_sign = (ref_idx as i32) - 4;
        let r2r_row = &plan.mfmv_ref2ref[n & 3];

        for y in row_start8..row_end8 {
            let y_sb_align = y & !7;
            let y_proj_start = y_sb_align.max(row_start8) as i32;
            let y_proj_end = (y_sb_align + 8).min(row_end8) as i32;
            let Some(r_row) = r.get(y * stride..y * stride + col_end8i) else {
                continue;
            };
            let mut x = col_start8i;
            while x < col_end8i {
                let Some(&rb) = r_row.get(x) else {
                    break;
                };
                let b_ref = rb.ref_;
                if b_ref == 0 || b_ref > 7 {
                    x += 1;
                    continue;
                }
                let Some(&r2r) = r2r_row.get(usize::from(b_ref) - 1) else {
                    x += 1;
                    continue;
                };
                if r2r == 0 {
                    x += 1;
                    continue;
                }
                let b_mv = rb.mv;
                let frac = fracs[usize::from(r2r) & 31];
                let (off_y, off_x) = project_mv_offset_neon(b_mv.y, b_mv.x, frac, ref_sign);
                let pos_y = (y as i32).wrapping_add(off_y);

                let mut run = 1usize;
                while x + run < col_end8i {
                    let Some(&next) = r_row.get(x + run) else {
                        break;
                    };
                    if next.ref_ != b_ref || next.mv != b_mv {
                        break;
                    }
                    run += 1;
                }

                if pos_y >= y_proj_start && pos_y < y_proj_end {
                    let pos = base + ((pos_y as usize) & 15) * stride;
                    let pos_x0 = (x as i32).wrapping_add(off_x);
                    let tmpl = TemporalBlock {
                        mv: b_mv,
                        ref_: r2r,
                    };
                    for k in 0..run {
                        let xx = x + k;
                        let x_sb_align = xx & !7;
                        let px = pos_x0 + (k as i32);
                        let lo = x_sb_align.saturating_sub(8).max(col_start8) as i32;
                        let hi = (x_sb_align + 16).min(col_end8) as i32;
                        if px >= lo
                            && px < hi
                            && let Some(cell) = rp_proj.get_mut(pos + (px as usize))
                        {
                            *cell = tmpl;
                        }
                    }
                }
                x += run;
            }
        }
    }
}
