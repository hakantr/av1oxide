//! SSE2 `splat_mv` çekirdeği (`D03054` + `D03055`; P46 Adım 1).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Sembol | Kaynak |
//! |---|---|
//! | `dav1d_splat_mv_sse2` | `src/x86/refmvs.asm:298-360`, `src/x86/refmvs.h:37,46` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bu modülde bütün işlevler yalnız `#[target_feature(enable = "sse2")]`
//!   taşır; SSSE3+, VEX ve EVEX komutu üretilmez (`0` SSSE3+/VEX/EVEX).
//! - Kopyasız O(1) sözleşme: girişte [`splat_mv_sozlesmesi`] denetlenir;
//!   sözleşme dışı girdide [`splat_mv_skaler_devret`] ile skaler gövdeye
//!   devredilir.
//! - Tek `unsafe` blok [`splat_mv_sse2`] içindeki `#[target_feature]` giriş
//!   çağrısıdır; dosya `audit::UNSAFE_EXEMPTIONS` defterinde kayıtlıdır.

#![allow(unsafe_code)]

use core::arch::x86_64::{__m128i, _mm_setr_epi32, _mm_shuffle_epi32};

use crate::cpu::Sse2Token;
use crate::decode::refmvs::RefBlock;
use crate::dsp::refmvs::{SplatMvFn, splat_mv_skaler_devret, splat_mv_sozlesmesi};
use crate::dsp::x86::mem;

/// `splat_mv` SSE2 yuva işaretçisi (`D03054` / `D03055`).
#[must_use]
pub const fn splat_mv_sse2_fn(_token: Sse2Token) -> SplatMvFn {
    splat_mv_sse2
}

fn splat_mv_sse2(
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
    let sub = &mut grid[by4 * iw4 + bx4..];
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Sse2Token` ile verilir
    // (`splat_mv_sse2_fn`); belirteç SSE2 tespiti + maske izninin kanıtıdır.
    // (2) sınır: `splat_mv_sozlesmesi` `bx4 + bw4 <= iw4` ve
    // `(by4 + bh4 - 1) * iw4 + bx4 + bw4 <= grid.len()` koşullarını O(1)
    // doğruladı. (3) hizalama/takma ad: tüm vektör saklamaları `mem::store_refblock_x4`
    // hizasız sabit boyutlu dizi yardımcısıyla yapılır; `sub` `&mut [RefBlock]`
    // tek erişim yoludur.
    unsafe { splat_mv_sse2_giris(sub, iw4, rmv, bw4, bh4) }
}

#[target_feature(enable = "sse2")]
#[inline]
pub(crate) fn refblock_dwords(rmv: RefBlock) -> [i32; 3] {
    let d0 = rmv.mv[0].n();
    let d1 = rmv.mv[1].n();
    let d2 = i32::from_le_bytes([rmv.ref_pair[0] as u8, rmv.ref_pair[1] as u8, rmv.bs, rmv.mf]);
    [d0, d1, d2]
}

#[target_feature(enable = "sse2")]
#[inline]
pub(crate) fn splat_xmm_triple(rmv: RefBlock) -> (__m128i, __m128i, __m128i) {
    let [d0, d1, d2] = refblock_dwords(rmv);
    let base = _mm_setr_epi32(d0, d1, d2, d0);
    // `q0210 = 0x24`, `q1021 = 0x49`, `q2102 = 0x92` (`src/x86/refmvs.asm:308-310`).
    let m0 = _mm_shuffle_epi32::<0x24>(base);
    let m1 = _mm_shuffle_epi32::<0x49>(base);
    let m2 = _mm_shuffle_epi32::<0x92>(base);
    (m0, m1, m2)
}

#[target_feature(enable = "sse2")]
#[cold]
#[inline(never)]
fn splat_mv_sse2_genel(grid: &mut [RefBlock], iw4: usize, rmv: RefBlock, bw4: usize, bh4: usize) {
    let (m0, m1, m2) = splat_xmm_triple(rmv);
    let mut off = 0usize;
    for _ in 0..bh4 {
        let row = &mut grid[off..off + bw4];
        let (chunks, rem) = row.as_chunks_mut::<4>();
        for c in chunks {
            mem::store_refblock_x4(c, m0, m1, m2);
        }
        for slot in rem {
            *slot = rmv;
        }
        off += iw4;
    }
}

#[target_feature(enable = "sse2")]
fn splat_mv_sse2_giris(grid: &mut [RefBlock], iw4: usize, rmv: RefBlock, bw4: usize, bh4: usize) {
    let mut off = 0usize;
    match bw4 {
        1 => {
            for _ in 0..bh4 {
                grid[off] = rmv;
                off += iw4;
            }
        }
        2 => {
            for _ in 0..bh4 {
                if let Some(dst) = grid[off..].first_chunk_mut::<2>() {
                    dst[0] = rmv;
                    dst[1] = rmv;
                }
                off += iw4;
            }
        }
        4 => {
            let (m0, m1, m2) = splat_xmm_triple(rmv);
            for _ in 0..bh4 {
                if let Some(dst) = grid[off..].first_chunk_mut::<4>() {
                    mem::store_refblock_x4(dst, m0, m1, m2);
                }
                off += iw4;
            }
        }
        8 => {
            let (m0, m1, m2) = splat_xmm_triple(rmv);
            for _ in 0..bh4 {
                if let Some(dst) = grid[off..].first_chunk_mut::<8>() {
                    let (c0, c1) = dst.split_first_chunk_mut::<4>().unwrap();
                    mem::store_refblock_x4(c0, m0, m1, m2);
                    mem::store_refblock_x4(c1.first_chunk_mut::<4>().unwrap(), m0, m1, m2);
                }
                off += iw4;
            }
        }
        16 => {
            let (m0, m1, m2) = splat_xmm_triple(rmv);
            for _ in 0..bh4 {
                if let Some(dst) = grid[off..].first_chunk_mut::<16>() {
                    let (chunks, _) = dst.as_chunks_mut::<4>();
                    for c in chunks {
                        mem::store_refblock_x4(c, m0, m1, m2);
                    }
                }
                off += iw4;
            }
        }
        32 => {
            let (m0, m1, m2) = splat_xmm_triple(rmv);
            for _ in 0..bh4 {
                if let Some(dst) = grid[off..].first_chunk_mut::<32>() {
                    let (chunks, _) = dst.as_chunks_mut::<4>();
                    for c in chunks {
                        mem::store_refblock_x4(c, m0, m1, m2);
                    }
                }
                off += iw4;
            }
        }
        _ => splat_mv_sse2_genel(grid, iw4, rmv, bw4, bh4),
    }
}
