//! SSSE3 `save_tmvs` çekirdeği (`D03048` + `D03049`; P46 Adım 1).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Sembol | Kaynak |
//! |---|---|
//! | `dav1d_save_tmvs_ssse3` | `src/x86/refmvs.asm:128-296`, `src/x86/refmvs.h:33,50` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bu modülde bütün işlevler yalnız `#[target_feature(enable = "ssse3")]`
//!   taşır; SSE4.1+, VEX ve EVEX komutu üretilmez (`0` SSE4.1+/VEX/EVEX).
//! - Kopyasız O(1) sözleşme: girişte [`save_tmvs_sozlesmesi`] denetlenir;
//!   sözleşme dışı girdide [`save_tmvs_skaler_devret`] ile skaler gövdeye
//!   devredilir.
//! - Tek `unsafe` blok [`save_tmvs_ssse3`] içindeki `#[target_feature]` giriş
//!   çağrısıdır; dosya `audit::UNSAFE_EXEMPTIONS` defterinde kayıtlıdır.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, _mm_abs_epi16, _mm_and_si128, _mm_cmpgt_epi32, _mm_cvtsi32_si128, _mm_cvtsi64_si128,
    _mm_cvtsi128_si64, _mm_or_si128, _mm_set_epi64x, _mm_set1_epi8, _mm_setr_epi8,
    _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_srli_epi16, _mm_unpackhi_epi64, _mm_unpacklo_epi64,
    _mm_xor_si128,
};

use crate::cpu::Ssse3Token;
use crate::decode::refmvs::{Mv, RefBlock};
use crate::decode::refmvs_tmvs::TemporalBlock;
use crate::dsp::refmvs::{SaveTmvsFn, save_tmvs_skaler_devret, save_tmvs_sozlesmesi};
use crate::dsp::x86::mem;

/// `BLOCK_DIMENSIONS[bs][0]` (`4×4` birimi) → `log2(((bw4 + 1) >> 1).max(1))` (`0..=4`) tablosu.
pub(crate) const BW8_SHIFT_TAB: [u8; 22] = [
    4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
];

#[inline(always)]
pub(crate) fn bw8_shift_of_bs(bs: u8) -> u8 {
    let idx = usize::from(bs);
    if idx < BW8_SHIFT_TAB.len() {
        BW8_SHIFT_TAB[idx]
    } else {
        0
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
pub(crate) fn temporal_pack_masks() -> (__m128i, __m128i, __m128i) {
    (
        _mm_setr_epi8(0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3),
        _mm_setr_epi8(4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1),
        _mm_setr_epi8(2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5),
    )
}

#[target_feature(enable = "ssse3")]
#[inline]
pub(crate) fn write_temporal_vec_ssse3(
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

/// `save_tmvs` SSSE3 yuva işaretçisi (`D03048` / `D03049`).
#[must_use]
pub const fn save_tmvs_ssse3_fn(_token: Ssse3Token) -> SaveTmvsFn {
    save_tmvs_ssse3
}

#[allow(clippy::too_many_arguments)]
fn save_tmvs_ssse3(
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`save_tmvs_ssse3_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır.
    // (2) sınır: `save_tmvs_sozlesmesi` `grid` ve `rp` dilimlerinin sınırlarını
    // O(1) doğruladı. (3) hizalama/takma ad: tüm erişimler sınır denetimli dilim
    // indekslemesiyle ve yazmaç içi SSSE3 işlemleriyle yapılır; `rp` `&mut`,
    // `grid` ve `ref_sign` `&` ayrık referanslardır.
    unsafe {
        save_tmvs_ssse3_giris(
            rp, stride, grid, iw4, ref_sign, col_end8, row_end8, col_start8, row_start8,
        )
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
pub(crate) fn refblock_to_m128i(b: RefBlock) -> __m128i {
    let q0 = (i64::from(b.mv[1].n() as u32) << 32) | i64::from(b.mv[0].n() as u32);
    let d2 = i32::from_le_bytes([b.ref_pair[0] as u8, b.ref_pair[1] as u8, b.bs, b.mf]);
    _mm_unpacklo_epi64(_mm_cvtsi64_si128(q0), _mm_cvtsi32_si128(d2))
}

#[target_feature(enable = "ssse3")]
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

#[target_feature(enable = "ssse3")]
#[allow(clippy::too_many_arguments)]
fn save_tmvs_ssse3_giris(
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
    // `src/x86/refmvs.asm:63-68, 160-169`:
    // `m5` = `[0, ref_sign[0..7], 0..0]` (`psllq m5, 8`).
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
    let m5 = _mm_set_epi64x(0, sign_q);
    let m4 = _mm_setr_epi8(0, -1, -1, -1, 1, -1, -1, -1, 8, -1, -1, -1, 9, -1, -1, -1);
    let cond0 = i64::from_le_bytes([0x80, 0x81, 0x82, 0x83, 0x89, 0x00, 0x00, 0x00]);
    let cond1 = i64::from_le_bytes([0x84, 0x85, 0x86, 0x87, 0x88, 0x00, 0x00, 0x00]);
    let m6 = _mm_set_epi64x(cond0, cond0);
    let m7 = _mm_set_epi64x(cond1, cond1);
    let m8 = _mm_set1_epi8(-128);
    let (shuf0, shuf1, shuf2) = core::hint::black_box(temporal_pack_masks());

    for y in row_start8..row_end8 {
        let row4 = y * 2 + 1;
        let base = row4 * iw4;
        let grid_row = &grid[base..base + col_end8 * 2];
        let rp_row = (y - row_start8) * stride;
        let mut x = col_start8;
        while x < col_end8 {
            let b0 = grid_row[x * 2 + 1];
            let lg0 = bw8_shift_of_bs(b0.bs);
            let bw0 = 1usize << lg0;
            let m0_in = refblock_to_m128i(b0);
            let x1 = x + bw0;
            let (m1_in, lg1) = if x1 < col_end8 {
                let b1 = grid_row[x1 * 2 + 1];
                (refblock_to_m128i(b1), bw8_shift_of_bs(b1.bs))
            } else {
                (m0_in, 0)
            };

            // `src/x86/refmvs.asm:228-244`: iki adayın `ref` ve `|mv| < 4096` koşullarını
            // tek SSSE3 geçişinde hesapla.
            let refs = _mm_unpackhi_epi64(m0_in, m1_in);
            let ref_idx = _mm_shuffle_epi8(refs, m4);
            let mut m3 = _mm_shuffle_epi8(m5, ref_idx);
            let mvs = _mm_unpacklo_epi64(m0_in, m1_in);
            let mv_abs = _mm_srli_epi16::<12>(_mm_abs_epi16(mvs));
            m3 = _mm_cmpgt_epi32(m3, mv_abs);
            let m2 = _mm_shuffle_epi32::<0xB1>(m3);
            let sel = _mm_xor_si128(
                _mm_or_si128(_mm_and_si128(m3, m6), _mm_and_si128(m2, m7)),
                m8,
            );
            let out0_vec = _mm_shuffle_epi8(m0_in, sel);
            write_temporal_vec_ssse3(rp, rp_row + x, lg0, out0_vec, shuf0, shuf1, shuf2);
            if x1 < col_end8 {
                let out1_vec = _mm_shuffle_epi32::<0xEE>(_mm_shuffle_epi8(m1_in, sel));
                write_temporal_vec_ssse3(rp, rp_row + x1, lg1, out1_vec, shuf0, shuf1, shuf2);
                x = x1 + (1usize << lg1);
            } else {
                x = x1;
            }
        }
    }
}
