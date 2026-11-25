//! AVX2 `pal_idx_finish` çekirdeği (`D03036` + `D03037`; P45 Adım 1).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Sembol | Kaynak |
//! |---|---|
//! | `dav1d_pal_idx_finish_avx2` | `src/x86/pal.asm:275-470`, `src/x86/pal.h:31,44` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bu modülde bütün işlevler yalnız `#[target_feature(enable = "avx2")]`
//!   taşır; EVEX komutu üretilmez (`0` EVEX).
//! - Kopyasız O(1) sözleşme: girişte [`pal_idx_finish_sozlesmesi`] denetlenir;
//!   sözleşme dışı girdide [`pal_idx_finish_skaler_devret`] ile skaler gövdeye
//!   devredilir.
//! - Tek `unsafe` blok [`pal_idx_finish_avx2`] içindeki `#[target_feature]`
//!   giriş çağrısıdır; dosya `audit::UNSAFE_EXEMPTIONS` defterinde kayıtlıdır.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_maddubs_epi16, _mm_packus_epi16, _mm_set1_epi8, _mm_set1_epi16,
    _mm_setr_epi8, _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_shufflelo_epi16,
    _mm256_broadcastsi128_si256, _mm256_castsi256_si128, _mm256_inserti128_si256,
    _mm256_maddubs_epi16, _mm256_min_epu8, _mm256_packus_epi16, _mm256_permute2x128_si256,
    _mm256_permute4x64_epi64, _mm256_set1_epi8, _mm256_set1_epi16, _mm256_setr_epi8,
    _mm256_shuffle_epi8,
};

use crate::cpu::Avx2Token;
use crate::dsp::pal::{PalIdxFinishFn, pal_idx_finish_skaler_devret, pal_idx_finish_sozlesmesi};
use crate::dsp::x86::mem;

/// `pal_idx_finish` AVX2 yuva işaretçisi (`D03036` / `D03037`).
#[must_use]
pub const fn pal_idx_finish_avx2_fn(_token: Avx2Token) -> PalIdxFinishFn {
    pal_idx_finish_avx2
}

fn pal_idx_finish_avx2(dst: &mut [u8], src: &[u8], bw: usize, bh: usize, w: usize, h: usize) {
    if !pal_idx_finish_sozlesmesi(dst.len(), src.len(), bw, bh, w, h) {
        pal_idx_finish_skaler_devret(dst, src, bw, bh, w, h);
        return;
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`pal_idx_finish_avx2_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır.
    // (2) sınır: `pal_idx_finish_sozlesmesi` `bw, bh ∈ {4,8,16,32,64}`, `w ∈ 4..=bw (4k)`,
    // `h ∈ 4..=bh (4k)`, `src.len() >= bw*bh`, `dst.len() >= (bw/2)*bh`
    // koşullarını O(1) doğruladı. (3) hizalama/takma ad: tüm yükleme/saklamalar
    // `mem` içindeki hizasız sabit boyutlu dizi yardımcılarıyla yapılır; `dst`
    // `&mut [u8]`, `src` `&[u8]` ayrık referanslardır.
    unsafe { pal_idx_finish_avx2_giris(dst, src, bw, bh, w, h) }
}

#[target_feature(enable = "avx2")]
fn pb_0to15() -> __m128i {
    _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
}

#[target_feature(enable = "avx2")]
fn pb_0to31() -> __m256i {
    _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31,
    )
}

#[target_feature(enable = "avx2")]
fn pal_idx_w8_padh() -> __m128i {
    _mm_setr_epi8(0, 1, 2, 3, 3, 3, 3, 3, 8, 9, 10, 11, 11, 11, 11, 11)
}

#[target_feature(enable = "avx2")]
fn pal_idx_finish_avx2_giris(dst: &mut [u8], src: &[u8], bw: usize, bh: usize, w: usize, h: usize) {
    let src = &src[..bw * bh];
    let dst = &mut dst[..(bw / 2) * bh];
    let xm2 = _mm_set1_epi16(0x1001);
    let m2 = _mm256_set1_epi16(0x1001);
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
                xm0 = _mm_maddubs_epi16(xm0, xm2);
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
            let xm3 = pal_idx_w8_padh();
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
                    xm0 = _mm_shuffle_epi8(xm0, xm3);
                    xm1 = _mm_shuffle_epi8(xm1, xm3);
                }
                xm0 = _mm_maddubs_epi16(xm0, xm2);
                xm1 = _mm_maddubs_epi16(xm1, xm2);
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
            let m3 = _mm256_min_epu8(
                _mm256_set1_epi8((w - 1) as i8),
                _mm256_broadcastsi128_si256(pb_0to15()),
            );
            let mut m0 = _mm256_set1_epi8(0);
            for ([s0, s1], d) in src[..h * 16]
                .as_chunks::<32>()
                .0
                .as_chunks::<2>()
                .0
                .iter()
                .zip(dst[..h * 8].as_chunks_mut::<32>().0)
            {
                m0 = mem::load_u8x32(s0);
                let mut m1 = mem::load_u8x32(s1);
                if w < 16 {
                    m0 = _mm256_shuffle_epi8(m0, m3);
                    m1 = _mm256_shuffle_epi8(m1, m3);
                }
                m0 = _mm256_maddubs_epi16(m0, m2);
                m1 = _mm256_maddubs_epi16(m1, m2);
                m0 = _mm256_packus_epi16(m0, m1);
                let out = _mm256_permute4x64_epi64::<0xD8>(m0);
                mem::store_u8x32(d, out);
            }
            if h < bh {
                m0 = _mm256_permute4x64_epi64::<0xFF>(m0);
                for d in dst[h * 8..bh * 8].as_chunks_mut::<32>().0 {
                    mem::store_u8x32(d, m0);
                }
            }
        }
        32 => {
            let m3 = _mm256_min_epu8(_mm256_set1_epi8((w - 1) as i8), pb_0to31());
            let mut m0 = _mm256_set1_epi8(0);
            for ([s0, s1], d) in src[..h * 32]
                .as_chunks::<32>()
                .0
                .as_chunks::<2>()
                .0
                .iter()
                .zip(dst[..h * 16].as_chunks_mut::<32>().0)
            {
                m0 = mem::load_u8x32(s0);
                let mut m1 = mem::load_u8x32(s1);
                if w < 32 {
                    if w <= 16 {
                        m0 = _mm256_inserti128_si256::<1>(m0, _mm256_castsi256_si128(m0));
                        m1 = _mm256_inserti128_si256::<1>(m1, _mm256_castsi256_si128(m1));
                    }
                    m0 = _mm256_shuffle_epi8(m0, m3);
                    m1 = _mm256_shuffle_epi8(m1, m3);
                }
                m0 = _mm256_maddubs_epi16(m0, m2);
                m1 = _mm256_maddubs_epi16(m1, m2);
                m0 = _mm256_packus_epi16(m0, m1);
                let out = _mm256_permute4x64_epi64::<0xD8>(m0);
                mem::store_u8x32(d, out);
            }
            if h < bh {
                m0 = _mm256_permute4x64_epi64::<0xDD>(m0);
                for d in dst[h * 16..bh * 16].as_chunks_mut::<32>().0 {
                    mem::store_u8x32(d, m0);
                }
            }
        }
        64 => {
            let m4 = _mm256_set1_epi8(((w - 1) & 31) as i8);
            let m3 = _mm256_min_epu8(m4, pb_0to31());
            let mut m0 = _mm256_set1_epi8(0);
            for ([s0, s1], d) in src[..h * 64]
                .as_chunks::<32>()
                .0
                .as_chunks::<2>()
                .0
                .iter()
                .zip(dst[..h * 32].as_chunks_mut::<32>().0)
            {
                m0 = mem::load_u8x32(s0);
                let mut m1 = mem::load_u8x32(s1);
                if w < 64 {
                    if w <= 16 {
                        m1 = _mm256_inserti128_si256::<1>(m0, _mm256_castsi256_si128(m0));
                        m0 = _mm256_shuffle_epi8(m1, m3);
                        m1 = _mm256_shuffle_epi8(m1, m4);
                    } else if w <= 32 {
                        m1 = _mm256_permute2x128_si256::<0x11>(m0, m0);
                        m0 = _mm256_shuffle_epi8(m0, m3);
                        m1 = _mm256_shuffle_epi8(m1, m4);
                    } else if w <= 48 {
                        m1 = _mm256_inserti128_si256::<1>(m1, _mm256_castsi256_si128(m1));
                        m1 = _mm256_shuffle_epi8(m1, m3);
                    } else {
                        m1 = _mm256_shuffle_epi8(m1, m3);
                    }
                }
                m0 = _mm256_maddubs_epi16(m0, m2);
                m1 = _mm256_maddubs_epi16(m1, m2);
                m0 = _mm256_packus_epi16(m0, m1);
                m0 = _mm256_permute4x64_epi64::<0xD8>(m0);
                mem::store_u8x32(d, m0);
            }
            if h < bh {
                for d in dst[h * 32..bh * 32].as_chunks_mut::<32>().0 {
                    mem::store_u8x32(d, m0);
                }
            }
        }
        _ => unreachable!(),
    }
}
