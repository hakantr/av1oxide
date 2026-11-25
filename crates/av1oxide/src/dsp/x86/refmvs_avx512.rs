//! AVX512ICL `save_tmvs` ve `splat_mv` çekirdekleri (`D03046/47`, `D03052/53`; P46 Adım 1).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Sembol | Kaynak |
//! |---|---|
//! | `dav1d_save_tmvs_avx512icl` | `src/x86/refmvs.asm:759-876`, `src/x86/refmvs.h:35,63` |
//! | `dav1d_splat_mv_avx512icl` | `src/x86/refmvs.asm:878-952`, `src/x86/refmvs.h:39,64` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bu modülde bütün işlevler
//!   `#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]` taşır;
//!   gather/scatter komutu üretilmez (`0` gather/scatter, `SLOW_GATHER` engeli
//!   yoktur: `src/x86/refmvs.h:61-65`).
//! - Kopyasız O(1) sözleşme: girişte [`save_tmvs_sozlesmesi`] ve
//!   [`splat_mv_sozlesmesi`] denetlenir; sözleşme dışı girdide ilgili
//!   `*_skaler_devret` yardımcısıyla skaler gövdeye devredilir.
//! - İki `unsafe` blok [`save_tmvs_avx512icl`] ve [`splat_mv_avx512icl`] içindeki
//!   `#[target_feature]` giriş çağrılarıdır; dosya `audit::UNSAFE_EXEMPTIONS`
//!   defterinde kayıtlıdır.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm_cvtsi128_si64, _mm_set_epi64x, _mm_setr_epi8, _mm_shuffle_epi8,
    _mm256_setr_epi32, _mm256_shuffle_epi32, _mm512_abs_epi16, _mm512_and_si512,
    _mm512_broadcast_i32x4, _mm512_castsi128_si512, _mm512_castsi512_si128,
    _mm512_extracti32x4_epi32, _mm512_inserti32x4, _mm512_set1_epi8, _mm512_set1_epi64,
    _mm512_setr_epi32, _mm512_shuffle_epi8, _mm512_shuffle_epi32, _mm512_srli_epi16,
    _mm512_sub_epi32, _mm512_ternarylogic_epi32, _mm512_unpackhi_epi64,
};

use crate::cpu::Avx512IclToken;
use crate::decode::refmvs::{Mv, RefBlock};
use crate::decode::refmvs_tmvs::TemporalBlock;
use crate::dsp::refmvs::{
    SaveTmvsFn, SplatMvFn, save_tmvs_skaler_devret, save_tmvs_sozlesmesi, splat_mv_skaler_devret,
    splat_mv_sozlesmesi,
};
use crate::dsp::x86::mem;
use crate::dsp::x86::refmvs_sse2::{refblock_dwords, splat_xmm_triple};
use crate::dsp::x86::refmvs_ssse3::{bw8_shift_of_bs, refblock_to_m128i, temporal_pack_masks};

/// `save_tmvs` AVX512ICL yuva işaretçisi (`D03046` / `D03047`).
#[must_use]
pub const fn save_tmvs_avx512icl_fn(_token: Avx512IclToken) -> SaveTmvsFn {
    save_tmvs_avx512icl
}

/// `splat_mv` AVX512ICL yuva işaretçisi (`D03052` / `D03053`).
#[must_use]
pub const fn splat_mv_avx512icl_fn(_token: Avx512IclToken) -> SplatMvFn {
    splat_mv_avx512icl
}

#[allow(clippy::too_many_arguments)]
fn save_tmvs_avx512icl(
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`save_tmvs_avx512icl_fn`); belirteç AVX512ICL tespiti + maske izninin kanıtıdır.
    // (2) sınır: `save_tmvs_sozlesmesi` `grid` ve `rp` dilimlerinin sınırlarını
    // O(1) doğruladı. (3) hizalama/takma ad: tüm erişimler sınır denetimli dilim
    // indekslemesiyle ve yazmaç içi AVX-512 işlemleriyle yapılır; `rp` `&mut`,
    // `grid` ve `ref_sign` `&` ayrık referanslardır.
    unsafe {
        save_tmvs_avx512icl_giris(
            rp, stride, grid, iw4, ref_sign, col_end8, row_end8, col_start8, row_start8,
        )
    }
}

fn splat_mv_avx512icl(
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`splat_mv_avx512icl_fn`); belirteç AVX512ICL tespiti + maske izninin kanıtıdır.
    // (2) sınır: `splat_mv_sozlesmesi` `bx4 + bw4 <= iw4` ve
    // `(by4 + bh4 - 1) * iw4 + bx4 + bw4 <= grid.len()` koşullarını O(1)
    // doğruladı. (3) hizalama/takma ad: tüm vektör saklamaları `mem::store_refblock_x16_avx512`,
    // `mem::store_refblock_x8_avx2` ve `mem::store_refblock_x4` hizasız sabit
    // boyutlu dizi yardımcılarıyla yapılır; `sub` `&mut [RefBlock]` tek erişim yoludur.
    unsafe { splat_mv_avx512icl_giris(sub, iw4, rmv, bw4, bh4) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn decode_temporal_slot(v: __m128i) -> TemporalBlock {
    let q = _mm_cvtsi128_si64(v) as u64;
    TemporalBlock {
        mv: Mv {
            y: (q & 0xffff) as u16 as i16,
            x: ((q >> 16) & 0xffff) as u16 as i16,
        },
        ref_: ((q >> 32) & 0xff) as u8,
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn write_temporal_vec_avx512(
    rp: &mut [TemporalBlock],
    start: usize,
    lg: u8,
    v: __m128i,
    shuf0: __m128i,
    shuf1: __m128i,
    shuf2: __m128i,
) {
    if let Some(a) = rp.get_mut(start..).and_then(|s| s.first_chunk_mut::<16>()) {
        match lg {
            0 => a[0] = decode_temporal_slot(v),
            1 => {
                let val = decode_temporal_slot(v);
                a[0] = val;
                a[1] = val;
            }
            2 => {
                let v0 = _mm_shuffle_epi8(v, shuf0);
                let v1 = _mm_shuffle_epi8(v, shuf1);
                mem::store_temporal_x4(a.first_chunk_mut::<4>().unwrap(), v0, v1);
            }
            3 => {
                let v0 = _mm_shuffle_epi8(v, shuf0);
                let v1 = _mm_shuffle_epi8(v, shuf1);
                let v2 = _mm_shuffle_epi8(v, shuf2);
                mem::store_temporal_x8(a.first_chunk_mut::<8>().unwrap(), v0, v1, v2);
            }
            _ => {
                let v0 = _mm_shuffle_epi8(v, shuf0);
                let v1 = _mm_shuffle_epi8(v, shuf1);
                let v2 = _mm_shuffle_epi8(v, shuf2);
                let (c0, c1) = a.split_first_chunk_mut::<8>().unwrap();
                mem::store_temporal_x8(c0, v0, v1, v2);
                mem::store_temporal_x8(c1.first_chunk_mut::<8>().unwrap(), v0, v1, v2);
            }
        }
    } else if let Some(dst) = rp.get_mut(start..) {
        let val = decode_temporal_slot(v);
        let end = (1usize << lg).min(dst.len());
        let mut k = 0;
        while k < end {
            *core::hint::black_box(&mut dst[k]) = val;
            k += 1;
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[allow(clippy::too_many_arguments)]
fn save_tmvs_avx512icl_giris(
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
    // `src/x86/refmvs.asm:771-775`:
    let sign_q = i64::from_le_bytes([
        0,
        ref_sign[0],
        ref_sign[1],
        ref_sign[2],
        ref_sign[3],
        ref_sign[4],
        ref_sign[5],
        ref_sign[6],
    ]);
    let m4 = _mm512_set1_epi64(sign_q);
    let ref_shuf = i64::from_le_bytes([8, 0xff, 0xff, 0xff, 9, 0xff, 0xff, 0xff]);
    let m3 = _mm512_set1_epi64(ref_shuf);
    let cond_shuf = _mm_setr_epi8(3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 3, 3, 3, 3);
    let m5 = _mm512_broadcast_i32x4(cond_shuf);
    let cond0 = i64::from_le_bytes([0x80, 0x81, 0x82, 0x83, 0x89, 0x00, 0x00, 0x00]);
    let cond1 = i64::from_le_bytes([0x84, 0x85, 0x86, 0x87, 0x88, 0x00, 0x00, 0x00]);
    let m6 = _mm512_broadcast_i32x4(_mm_set_epi64x(cond1, cond0));
    let m7 = _mm512_set1_epi8(-128);
    let (shuf0, shuf1, shuf2) = core::hint::black_box(temporal_pack_masks());

    for y in row_start8..row_end8 {
        let row4 = y * 2 + 1;
        let base = row4 * iw4;
        let grid_row = &grid[base..base + col_end8 * 2];
        let rp_row = (y - row_start8) * stride;
        let mut x = col_start8;
        while x < col_end8 {
            // En çok 4 adayı 512-bit ZMM (`4 × 128-bit` lane) içinde birlikte değerlendir
            // (`src/x86/refmvs.asm:795-845`).
            let b0 = grid_row[x * 2 + 1];
            let lg0 = bw8_shift_of_bs(b0.bs);
            let bw0 = 1usize << lg0;
            let mut m0 = _mm512_castsi128_si512(refblock_to_m128i(b0));
            let x1 = x + bw0;
            let mut lg1 = u8::MAX;
            let mut x2 = x1;
            let mut lg2 = u8::MAX;
            let mut x3 = x1;
            let mut lg3 = u8::MAX;

            if x1 < col_end8 {
                let b1 = grid_row[x1 * 2 + 1];
                lg1 = bw8_shift_of_bs(b1.bs);
                m0 = _mm512_inserti32x4::<1>(m0, refblock_to_m128i(b1));
                x2 = x1 + (1usize << lg1);
                if x2 < col_end8 {
                    let b2 = grid_row[x2 * 2 + 1];
                    lg2 = bw8_shift_of_bs(b2.bs);
                    m0 = _mm512_inserti32x4::<2>(m0, refblock_to_m128i(b2));
                    x3 = x2 + (1usize << lg2);
                    if x3 < col_end8 {
                        let b3 = grid_row[x3 * 2 + 1];
                        lg3 = bw8_shift_of_bs(b3.bs);
                        m0 = _mm512_inserti32x4::<3>(m0, refblock_to_m128i(b3));
                    }
                }
            }

            // `src/x86/refmvs.asm:824-833`:
            let mut m1 = _mm512_shuffle_epi8(m0, m3);
            let mut m2 = _mm512_abs_epi16(m0);
            m1 = _mm512_shuffle_epi8(m4, m1);
            m2 = _mm512_srli_epi16::<12>(m2);
            m2 = _mm512_sub_epi32(m2, m1);
            m2 = _mm512_shuffle_epi8(m2, m5);
            m2 = _mm512_and_si512(m2, m6);
            m1 = _mm512_unpackhi_epi64(m2, m2);
            m1 = _mm512_ternarylogic_epi32::<0x56>(m1, m2, m7);
            let m2_out = _mm512_shuffle_epi8(m0, m1);

            write_temporal_vec_avx512(
                rp,
                rp_row + x,
                lg0,
                _mm512_castsi512_si128(m2_out),
                shuf0,
                shuf1,
                shuf2,
            );
            if lg1 != u8::MAX {
                write_temporal_vec_avx512(
                    rp,
                    rp_row + x1,
                    lg1,
                    _mm512_extracti32x4_epi32::<1>(m2_out),
                    shuf0,
                    shuf1,
                    shuf2,
                );
                if lg2 != u8::MAX {
                    write_temporal_vec_avx512(
                        rp,
                        rp_row + x2,
                        lg2,
                        _mm512_extracti32x4_epi32::<2>(m2_out),
                        shuf0,
                        shuf1,
                        shuf2,
                    );
                    if lg3 != u8::MAX {
                        write_temporal_vec_avx512(
                            rp,
                            rp_row + x3,
                            lg3,
                            _mm512_extracti32x4_epi32::<3>(m2_out),
                            shuf0,
                            shuf1,
                            shuf2,
                        );
                        x = x3 + (1usize << lg3);
                    } else {
                        x = x3;
                    }
                } else {
                    x = x2;
                }
            } else {
                x = x1;
            }
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn splat_zmm_triple(rmv: RefBlock) -> (__m512i, __m512i, __m512i) {
    let [d0, d1, d2] = refblock_dwords(rmv);
    // `src/x86/refmvs.asm:54-57, 880-884, 924-925`:
    // `m0` (64 B): `[d0,d1,d2,d0 | d1,d2,d0,d1 | d2,d0,d1,d2 | d0,d1,d2,d0]` (5.33 RefBlock)
    // `m1` = `pshufd m0, q1021 (0x49)`: `[d1,d2,d0,d1 | d2,d0,d1,d2 | d0,d1,d2,d0 | d1,d2,d0,d1]`
    // `m2` = `pshufd m0, q2102 (0x92)`: `[d2,d0,d1,d2 | d0,d1,d2,d0 | d1,d2,d0,d1 | d2,d0,d1,d2]`
    let m0 = _mm512_setr_epi32(
        d0, d1, d2, d0, d1, d2, d0, d1, d2, d0, d1, d2, d0, d1, d2, d0,
    );
    let m1 = _mm512_shuffle_epi32::<0x49>(m0);
    let m2 = _mm512_shuffle_epi32::<0x92>(m0);
    (m0, m1, m2)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn splat_ymm_triple(rmv: RefBlock) -> (__m256i, __m256i, __m256i) {
    let [d0, d1, d2] = refblock_dwords(rmv);
    let m0 = _mm256_setr_epi32(d0, d1, d2, d0, d1, d2, d0, d1);
    let m1 = _mm256_shuffle_epi32::<0x92>(m0);
    let m2 = _mm256_shuffle_epi32::<0x49>(m0);
    (m0, m1, m2)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[cold]
#[inline(never)]
fn splat_mv_avx512icl_genel(
    grid: &mut [RefBlock],
    iw4: usize,
    rmv: RefBlock,
    bw4: usize,
    bh4: usize,
) {
    let (zm0, zm1, zm2) = splat_zmm_triple(rmv);
    let (ym0, ym1, ym2) = splat_ymm_triple(rmv);
    let (xm0, xm1, xm2) = splat_xmm_triple(rmv);
    let mut off = 0usize;
    for _ in 0..bh4 {
        let row = &mut grid[off..off + bw4];
        let (chunks16, rem16) = row.as_chunks_mut::<16>();
        for c in chunks16 {
            mem::store_refblock_x16_avx512(c, zm0, zm1, zm2);
        }
        let (chunks8, rem8) = rem16.as_chunks_mut::<8>();
        for c in chunks8 {
            mem::store_refblock_x8_avx2(c, ym0, ym1, ym2);
        }
        let (chunks4, rem4) = rem8.as_chunks_mut::<4>();
        for c in chunks4 {
            mem::store_refblock_x4(c, xm0, xm1, xm2);
        }
        for slot in rem4 {
            *slot = rmv;
        }
        off += iw4;
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
fn splat_mv_avx512icl_giris(
    grid: &mut [RefBlock],
    iw4: usize,
    rmv: RefBlock,
    bw4: usize,
    bh4: usize,
) {
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
            let (xm0, xm1, xm2) = splat_xmm_triple(rmv);
            for _ in 0..bh4 {
                if let Some(dst) = grid[off..].first_chunk_mut::<4>() {
                    mem::store_refblock_x4(dst, xm0, xm1, xm2);
                }
                off += iw4;
            }
        }
        8 => {
            let (ym0, ym1, ym2) = splat_ymm_triple(rmv);
            for _ in 0..bh4 {
                if let Some(dst) = grid[off..].first_chunk_mut::<8>() {
                    mem::store_refblock_x8_avx2(dst, ym0, ym1, ym2);
                }
                off += iw4;
            }
        }
        16 => {
            let (zm0, zm1, zm2) = splat_zmm_triple(rmv);
            for _ in 0..bh4 {
                if let Some(dst) = grid[off..].first_chunk_mut::<16>() {
                    mem::store_refblock_x16_avx512(dst, zm0, zm1, zm2);
                }
                off += iw4;
            }
        }
        32 => {
            let (zm0, zm1, zm2) = splat_zmm_triple(rmv);
            for _ in 0..bh4 {
                if let Some(dst) = grid[off..].first_chunk_mut::<32>() {
                    let (c0, c1) = dst.split_first_chunk_mut::<16>().unwrap();
                    mem::store_refblock_x16_avx512(c0, zm0, zm1, zm2);
                    mem::store_refblock_x16_avx512(
                        c1.first_chunk_mut::<16>().unwrap(),
                        zm0,
                        zm1,
                        zm2,
                    );
                }
                off += iw4;
            }
        }
        _ => splat_mv_avx512icl_genel(grid, iw4, rmv, bw4, bh4),
    }
}
