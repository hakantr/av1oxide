//! SSSE3 `pal_idx_finish` çekirdeği (`D03040` + `D03041`; P45 Adım 1).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Sembol | Kaynak |
//! |---|---|
//! | `dav1d_pal_idx_finish_ssse3` | `src/x86/pal.asm:62-272`, `src/x86/pal.h:30,39` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bu modülde bütün işlevler yalnız `#[target_feature(enable = "ssse3")]`
//!   taşır; SSE4.1+, VEX ve EVEX komutu üretilmez (`0` SSE4.1+/VEX/EVEX).
//! - Kopyasız O(1) sözleşme: girişte [`pal_idx_finish_sozlesmesi`] denetlenir;
//!   sözleşme dışı girdide [`pal_idx_finish_skaler_devret`] ile skaler gövdeye
//!   devredilir.
//! - Tek `unsafe` blok [`pal_idx_finish_ssse3`] içindeki `#[target_feature]`
//!   giriş çağrısıdır; dosya `audit::UNSAFE_EXEMPTIONS` defterinde kayıtlıdır.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, _mm_maddubs_epi16, _mm_min_epu8, _mm_packus_epi16, _mm_set1_epi8, _mm_set1_epi16,
    _mm_setr_epi8, _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_shufflelo_epi16, _mm_unpackhi_epi64,
};

use crate::cpu::Ssse3Token;
use crate::dsp::pal::{PalIdxFinishFn, pal_idx_finish_skaler_devret, pal_idx_finish_sozlesmesi};
use crate::dsp::x86::mem;

/// `pal_idx_finish` SSSE3 yuva işaretçisi (`D03040` / `D03041`).
#[must_use]
pub const fn pal_idx_finish_ssse3_fn(_token: Ssse3Token) -> PalIdxFinishFn {
    pal_idx_finish_ssse3
}

fn pal_idx_finish_ssse3(dst: &mut [u8], src: &[u8], bw: usize, bh: usize, w: usize, h: usize) {
    if !pal_idx_finish_sozlesmesi(dst.len(), src.len(), bw, bh, w, h) {
        pal_idx_finish_skaler_devret(dst, src, bw, bh, w, h);
        return;
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`pal_idx_finish_ssse3_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır.
    // (2) sınır: `pal_idx_finish_sozlesmesi` `bw, bh ∈ {4,8,16,32,64}`, `w ∈ 4..=bw (4k)`,
    // `h ∈ 4..=bh (4k)`, `src.len() >= bw*bh`, `dst.len() >= (bw/2)*bh`
    // koşullarını O(1) doğruladı. (3) hizalama/takma ad: tüm yükleme/saklamalar
    // `mem` içindeki hizasız sabit boyutlu dizi yardımcılarıyla yapılır; `dst`
    // `&mut [u8]`, `src` `&[u8]` ayrık referanslardır.
    unsafe { pal_idx_finish_ssse3_giris(dst, src, bw, bh, w, h) }
}

#[target_feature(enable = "ssse3")]
fn pb_0to15() -> __m128i {
    _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
}

#[target_feature(enable = "ssse3")]
fn pal_idx_w8_padh() -> __m128i {
    _mm_setr_epi8(0, 1, 2, 3, 3, 3, 3, 3, 8, 9, 10, 11, 11, 11, 11, 11)
}

#[target_feature(enable = "ssse3")]
fn pal_idx_finish_ssse3_giris(
    dst: &mut [u8],
    src: &[u8],
    bw: usize,
    bh: usize,
    w: usize,
    h: usize,
) {
    let src = &src[..bw * bh];
    let dst = &mut dst[..(bw / 2) * bh];
    let m3 = _mm_set1_epi16(0x1001);
    match bw {
        4 => {
            let mut m0 = _mm_set1_epi8(0);
            for (s, d) in src[..h * 4]
                .as_chunks::<16>()
                .0
                .iter()
                .zip(dst[..h * 2].as_chunks_mut::<8>().0)
            {
                m0 = mem::load_u8x16(s);
                m0 = _mm_maddubs_epi16(m0, m3);
                m0 = _mm_packus_epi16(m0, m0);
                mem::store_u8x8(d, m0);
            }
            if h < bh {
                m0 = _mm_shufflelo_epi16::<0xFF>(m0);
                for d in dst[h * 2..bh * 2].as_chunks_mut::<8>().0 {
                    mem::store_u8x8(d, m0);
                }
            }
        }
        8 => {
            let m2 = pal_idx_w8_padh();
            let mut m0 = _mm_set1_epi8(0);
            for ([s0, s1], d) in src[..h * 8]
                .as_chunks::<16>()
                .0
                .as_chunks::<2>()
                .0
                .iter()
                .zip(dst[..h * 4].as_chunks_mut::<16>().0)
            {
                m0 = mem::load_u8x16(s0);
                let mut m1 = mem::load_u8x16(s1);
                if w < 8 {
                    m0 = _mm_shuffle_epi8(m0, m2);
                    m1 = _mm_shuffle_epi8(m1, m2);
                }
                m0 = _mm_maddubs_epi16(m0, m3);
                m1 = _mm_maddubs_epi16(m1, m3);
                m0 = _mm_packus_epi16(m0, m1);
                mem::store_u8x16(d, m0);
            }
            if h < bh {
                m0 = _mm_shuffle_epi32::<0xFF>(m0);
                for d in dst[h * 4..bh * 4].as_chunks_mut::<16>().0 {
                    mem::store_u8x16(d, m0);
                }
            }
        }
        16 => {
            let m5 = _mm_set1_epi8(((w - 1) & 15) as i8);
            let m4 = _mm_min_epu8(pb_0to15(), m5);
            let mut m0 = _mm_set1_epi8(0);
            for ([s0, s1], d) in src[..h * 16]
                .as_chunks::<16>()
                .0
                .as_chunks::<2>()
                .0
                .iter()
                .zip(dst[..h * 8].as_chunks_mut::<16>().0)
            {
                m0 = mem::load_u8x16(s0);
                let mut m1 = mem::load_u8x16(s1);
                if w < 16 {
                    m0 = _mm_shuffle_epi8(m0, m4);
                    m1 = _mm_shuffle_epi8(m1, m4);
                }
                m0 = _mm_maddubs_epi16(m0, m3);
                m1 = _mm_maddubs_epi16(m1, m3);
                m0 = _mm_packus_epi16(m0, m1);
                mem::store_u8x16(d, m0);
            }
            if h < bh {
                m0 = _mm_unpackhi_epi64(m0, m0);
                for d in dst[h * 8..bh * 8].as_chunks_mut::<16>().0 {
                    mem::store_u8x16(d, m0);
                }
            }
        }
        32 => {
            let m5 = _mm_set1_epi8(((w - 1) & 15) as i8);
            let m4 = _mm_min_epu8(pb_0to15(), m5);
            let mut m0 = _mm_set1_epi8(0);
            for ([s0, s1], d) in src[..h * 32]
                .as_chunks::<16>()
                .0
                .as_chunks::<2>()
                .0
                .iter()
                .zip(dst[..h * 16].as_chunks_mut::<16>().0)
            {
                m0 = mem::load_u8x16(s0);
                let mut m1 = mem::load_u8x16(s1);
                if w < 32 {
                    if w <= 16 {
                        m1 = _mm_shuffle_epi8(m0, m5);
                        m0 = _mm_shuffle_epi8(m0, m4);
                    } else {
                        m1 = _mm_shuffle_epi8(m1, m4);
                    }
                }
                m0 = _mm_maddubs_epi16(m0, m3);
                m1 = _mm_maddubs_epi16(m1, m3);
                m0 = _mm_packus_epi16(m0, m1);
                mem::store_u8x16(d, m0);
            }
            if h < bh {
                for d in dst[h * 16..bh * 16].as_chunks_mut::<16>().0 {
                    mem::store_u8x16(d, m0);
                }
            }
        }
        64 => {
            let m5 = _mm_set1_epi8(((w - 1) & 15) as i8);
            let m4 = _mm_min_epu8(pb_0to15(), m5);
            let mut out0 = _mm_set1_epi8(0);
            let mut out1 = _mm_set1_epi8(0);
            for ([s0, s1, s2, s3], [d0, d1]) in src[..h * 64]
                .as_chunks::<16>()
                .0
                .as_chunks::<4>()
                .0
                .iter()
                .zip(dst[..h * 32].as_chunks_mut::<16>().0.as_chunks_mut::<2>().0)
            {
                let m0 = mem::load_u8x16(s0);
                let m1 = mem::load_u8x16(s1);
                if w <= 32 {
                    if w <= 16 {
                        let p1 = _mm_shuffle_epi8(m0, m5);
                        let p0 = _mm_shuffle_epi8(m0, m4);
                        let w0 = _mm_maddubs_epi16(p0, m3);
                        let w1 = _mm_maddubs_epi16(p1, m3);
                        out0 = _mm_packus_epi16(w0, w1);
                        out1 = _mm_packus_epi16(w1, w1);
                    } else {
                        let p1 = _mm_shuffle_epi8(m1, m4);
                        let w0 = _mm_maddubs_epi16(m0, m3);
                        let w2 = _mm_maddubs_epi16(p1, m3);
                        let p1_rep = _mm_shuffle_epi8(p1, m5);
                        let w1 = _mm_maddubs_epi16(p1_rep, m3);
                        out0 = _mm_packus_epi16(w0, w2);
                        out1 = _mm_packus_epi16(w1, w1);
                    }
                } else {
                    let w0 = _mm_maddubs_epi16(m0, m3);
                    let w1 = _mm_maddubs_epi16(m1, m3);
                    out0 = _mm_packus_epi16(w0, w1);
                    let mut m2 = mem::load_u8x16(s2);
                    let mut m3_in = mem::load_u8x16(s3);
                    if w < 64 {
                        if w <= 48 {
                            m3_in = _mm_shuffle_epi8(m2, m5);
                            m2 = _mm_shuffle_epi8(m2, m4);
                        } else {
                            m3_in = _mm_shuffle_epi8(m3_in, m4);
                        }
                    }
                    let w2 = _mm_maddubs_epi16(m2, m3);
                    let w3 = _mm_maddubs_epi16(m3_in, m3);
                    out1 = _mm_packus_epi16(w2, w3);
                }
                mem::store_u8x16(d0, out0);
                mem::store_u8x16(d1, out1);
            }
            if h < bh {
                for [d0, d1] in dst[h * 32..bh * 32]
                    .as_chunks_mut::<16>()
                    .0
                    .as_chunks_mut::<2>()
                    .0
                {
                    mem::store_u8x16(d0, out0);
                    mem::store_u8x16(d1, out1);
                }
            }
        }
        _ => unreachable!(),
    }
}
