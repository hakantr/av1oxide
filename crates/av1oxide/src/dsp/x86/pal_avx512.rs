//! AVX512ICL `pal_idx_finish` çekirdeği (`D03038` + `D03039`; P45 Adım 1).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Sembol | Kaynak |
//! |---|---|
//! | `dav1d_pal_idx_finish_avx512icl` | `src/x86/pal.asm:471-640`, `src/x86/pal.h:32,48` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bu modülde bütün işlevler `#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]`
//!   taşır; gather/scatter komutu üretilmez (`0` gather/scatter, `SLOW_GATHER`
//!   engeli yoktur: `src/x86/pal.h:46-48`).
//! - Kopyasız O(1) sözleşme: girişte [`pal_idx_finish_sozlesmesi`] denetlenir;
//!   sözleşme dışı girdide [`pal_idx_finish_skaler_devret`] ile skaler gövdeye
//!   devredilir.
//! - Tek `unsafe` blok [`pal_idx_finish_avx512icl`] içindeki `#[target_feature]`
//!   giriş çağrısıdır; dosya `audit::UNSAFE_EXEMPTIONS` defterinde kayıtlıdır.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, __m512i, _mm_maddubs_epi16, _mm_packus_epi16, _mm_set1_epi8, _mm_set1_epi16,
    _mm_setr_epi8, _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_shufflelo_epi16,
    _mm256_permute4x64_epi64, _mm256_set1_epi8, _mm512_add_epi8, _mm512_broadcast_i32x4,
    _mm512_cvtepi16_epi8, _mm512_maddubs_epi16, _mm512_min_epu8, _mm512_permutex2var_epi8,
    _mm512_permutexvar_epi8, _mm512_set1_epi8, _mm512_set1_epi16, _mm512_shuffle_epi8,
    _mm512_shuffle_i32x4,
};

use crate::cpu::Avx512IclToken;
use crate::dsp::pal::{PalIdxFinishFn, pal_idx_finish_skaler_devret, pal_idx_finish_sozlesmesi};
use crate::dsp::x86::mem;

const PB_0TO63_BYTES: [u8; 64] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
];

const PB_W32_OFFSET_BYTES: [u8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
    32, 32, 32, 32, 32, 32, 32, 32,
];

/// `pal_idx_finish` AVX512ICL yuva işaretçisi (`D03038` / `D03039`).
#[must_use]
pub const fn pal_idx_finish_avx512icl_fn(_token: Avx512IclToken) -> PalIdxFinishFn {
    pal_idx_finish_avx512icl
}

fn pal_idx_finish_avx512icl(dst: &mut [u8], src: &[u8], bw: usize, bh: usize, w: usize, h: usize) {
    if !pal_idx_finish_sozlesmesi(dst.len(), src.len(), bw, bh, w, h) {
        pal_idx_finish_skaler_devret(dst, src, bw, bh, w, h);
        return;
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`pal_idx_finish_avx512icl_fn`); belirteç AVX512ICL tespiti + maske izninin kanıtıdır.
    // (2) sınır: `pal_idx_finish_sozlesmesi` `bw, bh ∈ {4,8,16,32,64}`, `w ∈ 4..=bw (4k)`,
    // `h ∈ 4..=bh (4k)`, `src.len() >= bw*bh`, `dst.len() >= (bw/2)*bh`
    // koşullarını O(1) doğruladı. (3) hizalama/takma ad: tüm yükleme/saklamalar
    // `mem` içindeki hizasız sabit boyutlu dizi yardımcılarıyla yapılır; `dst`
    // `&mut [u8]`, `src` `&[u8]` ayrık referanslardır.
    unsafe { pal_idx_finish_avx512icl_giris(dst, src, bw, bh, w, h) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
fn pb_0to15() -> __m128i {
    _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
fn pb_0to63() -> __m512i {
    mem::load_u8x64(&PB_0TO63_BYTES)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
fn pal_idx_w8_padh() -> __m128i {
    _mm_setr_epi8(0, 1, 2, 3, 3, 3, 3, 3, 8, 9, 10, 11, 11, 11, 11, 11)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
fn pal_idx_finish_avx512icl_giris(
    dst: &mut [u8],
    src: &[u8],
    bw: usize,
    bh: usize,
    w: usize,
    h: usize,
) {
    let src = &src[..bw * bh];
    let dst = &mut dst[..(bw / 2) * bh];
    let xm4 = _mm_set1_epi16(0x1001);
    let m4 = _mm512_set1_epi16(0x1001);
    match bw {
        4 => {
            let mut xm0 = _mm_set1_epi8(0);
            for (s, d) in src[..h * 4]
                .as_chunks::<16>()
                .0
                .iter()
                .zip(dst[..h * 2].as_chunks_mut::<8>().0)
            {
                xm0 = mem::load_u8x16(s);
                xm0 = _mm_maddubs_epi16(xm0, xm4);
                xm0 = _mm_packus_epi16(xm0, xm0);
                mem::store_u8x8(d, xm0);
            }
            if h < bh {
                xm0 = _mm_shufflelo_epi16::<0xFF>(xm0);
                for d in dst[h * 2..bh * 2].as_chunks_mut::<8>().0 {
                    mem::store_u8x8(d, xm0);
                }
            }
        }
        8 => {
            let xm2 = pal_idx_w8_padh();
            let mut xm0 = _mm_set1_epi8(0);
            for ([s0, s1], d) in src[..h * 8]
                .as_chunks::<16>()
                .0
                .as_chunks::<2>()
                .0
                .iter()
                .zip(dst[..h * 4].as_chunks_mut::<16>().0)
            {
                xm0 = mem::load_u8x16(s0);
                let mut xm1 = mem::load_u8x16(s1);
                if w < 8 {
                    xm0 = _mm_shuffle_epi8(xm0, xm2);
                    xm1 = _mm_shuffle_epi8(xm1, xm2);
                }
                xm0 = _mm_maddubs_epi16(xm0, xm4);
                xm1 = _mm_maddubs_epi16(xm1, xm4);
                xm0 = _mm_packus_epi16(xm0, xm1);
                mem::store_u8x16(d, xm0);
            }
            if h < bh {
                xm0 = _mm_shuffle_epi32::<0xFF>(xm0);
                for d in dst[h * 4..bh * 4].as_chunks_mut::<16>().0 {
                    mem::store_u8x16(d, xm0);
                }
            }
        }
        16 => {
            let m2 = _mm512_min_epu8(
                _mm512_broadcast_i32x4(pb_0to15()),
                _mm512_set1_epi8((w - 1) as i8),
            );
            let mut ym0 = _mm256_set1_epi8(0);
            for (s, d) in src[..h * 16]
                .as_chunks::<64>()
                .0
                .iter()
                .zip(dst[..h * 8].as_chunks_mut::<32>().0)
            {
                let mut m0 = mem::load_u8x64(s);
                if w < 16 {
                    m0 = _mm512_shuffle_epi8(m0, m2);
                }
                m0 = _mm512_maddubs_epi16(m0, m4);
                ym0 = _mm512_cvtepi16_epi8(m0);
                mem::store_u8x32(d, ym0);
            }
            if h < bh {
                ym0 = _mm256_permute4x64_epi64::<0xFF>(ym0);
                for d in dst[h * 8..bh * 8].as_chunks_mut::<32>().0 {
                    mem::store_u8x32(d, ym0);
                }
            }
        }
        32 => {
            let mut m2 = pb_0to63();
            let m3 = _mm512_add_epi8(m2, m2);
            if w < 32 {
                let m0 = _mm512_add_epi8(
                    _mm512_set1_epi8((w - 1) as i8),
                    mem::load_u8x64(&PB_W32_OFFSET_BYTES),
                );
                m2 = _mm512_min_epu8(m2, m0);
            }
            let mut m0 = _mm512_set1_epi8(0);
            for ([s0, s1], d) in src[..h * 32]
                .as_chunks::<64>()
                .0
                .as_chunks::<2>()
                .0
                .iter()
                .zip(dst[..h * 16].as_chunks_mut::<64>().0)
            {
                m0 = mem::load_u8x64(s0);
                let mut m1 = mem::load_u8x64(s1);
                if w < 32 {
                    m0 = _mm512_permutexvar_epi8(m2, m0);
                    m1 = _mm512_permutexvar_epi8(m2, m1);
                }
                m0 = _mm512_maddubs_epi16(m0, m4);
                m1 = _mm512_maddubs_epi16(m1, m4);
                m0 = _mm512_permutex2var_epi8(m0, m3, m1);
                mem::store_u8x64(d, m0);
            }
            if h < bh {
                m0 = _mm512_shuffle_i32x4::<0xFF>(m0, m0);
                for d in dst[h * 16..bh * 16].as_chunks_mut::<64>().0 {
                    mem::store_u8x64(d, m0);
                }
            }
        }
        64 => {
            let mut m5 = pb_0to63();
            let m6 = _mm512_add_epi8(m5, m5);
            if w < 64 {
                m5 = _mm512_min_epu8(m5, _mm512_set1_epi8((w - 1) as i8));
            }
            let mut m2 = _mm512_set1_epi8(0);
            for ([s0, s1, s2, s3], [d0, d1]) in src[..h * 64]
                .as_chunks::<64>()
                .0
                .as_chunks::<4>()
                .0
                .iter()
                .zip(dst[..h * 32].as_chunks_mut::<64>().0.as_chunks_mut::<2>().0)
            {
                let mut m0 = mem::load_u8x64(s0);
                let mut m1 = mem::load_u8x64(s1);
                m2 = mem::load_u8x64(s2);
                let mut m3 = mem::load_u8x64(s3);
                if w < 64 {
                    m0 = _mm512_permutexvar_epi8(m5, m0);
                    m1 = _mm512_permutexvar_epi8(m5, m1);
                    m2 = _mm512_permutexvar_epi8(m5, m2);
                    m3 = _mm512_permutexvar_epi8(m5, m3);
                }
                m0 = _mm512_maddubs_epi16(m0, m4);
                m1 = _mm512_maddubs_epi16(m1, m4);
                m2 = _mm512_maddubs_epi16(m2, m4);
                m3 = _mm512_maddubs_epi16(m3, m4);
                m0 = _mm512_permutex2var_epi8(m0, m6, m1);
                m2 = _mm512_permutex2var_epi8(m2, m6, m3);
                mem::store_u8x64(d0, m0);
                mem::store_u8x64(d1, m2);
            }
            if h < bh {
                m2 = _mm512_shuffle_i32x4::<0xEE>(m2, m2);
                for [d0, d1] in dst[h * 32..bh * 32]
                    .as_chunks_mut::<64>()
                    .0
                    .as_chunks_mut::<2>()
                    .0
                {
                    mem::store_u8x64(d0, m2);
                    mem::store_u8x64(d1, m2);
                }
            }
        }
        _ => unreachable!(),
    }
}
