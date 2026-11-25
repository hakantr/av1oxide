//! MC bileşik AVX512ICL çekirdekleri (8 bpc) — F4-MC e (P36): `avg`, `w_avg`,
//! `mask`, `w_mask[0..2]`.
//!
//! ## Sözleşme
//!
//! AVX2 gövdesiyle ([`super::mc_bilesik`]) aynı: girişte
//! [`crate::dsp::mc::bilesik_sozlesmesi`] (ve `w_avg` ağırlığı, `w_mask` alt
//! türü, `sign`, maske boyu); sözleşme dışı girdi skalere devreder ve sayılır.
//! Okuma tam `w·h` öğe, yazma `h` satır × `w` bayt (+ maske). Değer tanım
//! kümesi AVX2 gövdesininkiyle aynıdır ([`super::mc_bilesik::PREP_ARALIK`]).
//!
//! ## Yöntem
//!
//! Aritmetik AVX2 gövdesiyle aynıdır (`pmulhrsw`/`pmulhw` formülleri); yazmaç
//! 512 bit: 32 öğe bir yazmaç, 64 öğe bir adım. `packuswb` 128 bitlik şeritler
//! içinde çalıştığından sıralı 64 bayt `vpermq` (`_mm512_permutexvar_epi64`)
//! ile elde edilir. `w_mask` maskesi: `64 − m` baytları aynı paketlemeyle,
//! ikili toplamlar `pmaddubsw`, `i16` → bayt `vpmovwb`. `w = 32`'de bir adım iki
//! satırdır. `w < 32`'de aynı girişten (aynı sözleşme denetiminden sonra) AVX2
//! gövdesi çağrılır: dar blokta 512 bit yazmacın yarısı boş kalırdı.

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mc_bilesik::{self as avx2, AVG, Girdi, W_AVG};
use super::mem::{chunk, chunk_mut, load_i16x32, load_u8x32, store_u8x32, store_u8x64};
use crate::cpu::Avx512IclToken;
use crate::dsp::mc::{AvgFn, MaskFn, WAvgFn, WMaskFn};
use core::arch::x86_64::{
    __m512i, _mm256_add_epi16, _mm256_cvtepi16_epi8, _mm256_set1_epi16, _mm256_srli_epi16,
    _mm512_abs_epi16, _mm512_add_epi16, _mm512_castsi512_si256, _mm512_cvtepi16_epi8,
    _mm512_cvtepu8_epi16, _mm512_extracti64x4_epi64, _mm512_maddubs_epi16, _mm512_mulhi_epi16,
    _mm512_mulhrs_epi16, _mm512_packus_epi16, _mm512_permutexvar_epi64, _mm512_set1_epi8,
    _mm512_set1_epi16, _mm512_setr_epi64, _mm512_setzero_si512, _mm512_slli_epi16,
    _mm512_srli_epi16, _mm512_sub_epi16, _mm512_subs_epu16,
};

/// 32 öğenin `i16` sonucu, öğe ofseti `o`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn hesap32<const OP: u8>(g: &Girdi<'_>, o: usize) -> __m512i {
    let a = load_i16x32(chunk(g.t1, o));
    let b = load_i16x32(chunk(g.t2, o));
    match OP {
        AVG => _mm512_mulhrs_epi16(_mm512_add_epi16(a, b), _mm512_set1_epi16(1024)),
        W_AVG => {
            let r = _mm512_add_epi16(
                a,
                _mm512_mulhi_epi16(_mm512_sub_epi16(a, b), _mm512_set1_epi16(g.k)),
            );
            _mm512_mulhrs_epi16(r, _mm512_set1_epi16(2048))
        }
        _ => {
            let m = _mm512_cvtepu8_epi16(load_u8x32(chunk(g.mask, o)));
            let k = _mm512_slli_epi16::<9>(_mm512_sub_epi16(_mm512_setzero_si512(), m));
            let d = _mm512_slli_epi16::<1>(_mm512_sub_epi16(b, a));
            let r = _mm512_add_epi16(b, _mm512_mulhi_epi16(d, k));
            _mm512_mulhrs_epi16(r, _mm512_set1_epi16(2048))
        }
    }
}

/// İki 32'lik `i16` sonuçtan sıralı 64 bayt (doyurmalı).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn paketle64(r0: __m512i, r1: __m512i) -> __m512i {
    _mm512_permutexvar_epi64(
        _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7),
        _mm512_packus_epi16(r0, r1),
    )
}

/// Öğe ofseti `o`'dan başlayan sıralı 64 pikseli (`W` sütunlu bitişik satırlar)
/// hedefe yazar; `W = 32`'de iki satır.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn sakla64<const W: usize>(dst: &mut [u8], ds: usize, o: usize, p: __m512i) {
    let (y, x) = (o / W, o % W);
    if W == 32 {
        store_u8x32(chunk_mut(dst, y * ds), _mm512_castsi512_si256(p));
        store_u8x32(
            chunk_mut(dst, (y + 1) * ds),
            _mm512_extracti64x4_epi64::<1>(p),
        );
    } else {
        store_u8x64(chunk_mut(dst, y * ds + x), p);
    }
}

/// `avg` / `w_avg` / `mask` gövdesi, `W ≥ 32` sütun.
#[target_feature(enable = "avx512f,avx512bw")]
fn bidir<const OP: u8, const W: usize>(dst: &mut [u8], ds: usize, n: usize, g: &Girdi<'_>) {
    let mut o = 0;
    while o < n {
        let p = paketle64(hesap32::<OP>(g, o), hesap32::<OP>(g, o + 32));
        sakla64::<W>(dst, ds, o, p);
        o += 64;
    }
}

/// Genişliğe göre `bidir` monomorfu (`w ≥ 32`).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline(never)]
pub(super) fn bidir_w<const OP: u8>(dst: &mut [u8], ds: usize, w: usize, h: usize, g: &Girdi<'_>) {
    let n = w * h;
    match w {
        32 => bidir::<OP, 32>(dst, ds, n, g),
        64 => bidir::<OP, 64>(dst, ds, n, g),
        _ => bidir::<OP, 128>(dst, ds, n, g),
    }
}

/// `w_mask` 32 öğesi: (`i16` piksel sonucu, `64 − m`).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn wm32(t1: &[i16], t2: &[i16], o: usize) -> (__m512i, __m512i) {
    let a = load_i16x32(chunk(t1, o));
    let b = load_i16x32(chunk(t2, o));
    let d = _mm512_sub_epi16(b, a);
    let k = _mm512_srli_epi16::<8>(_mm512_subs_epu16(
        _mm512_set1_epi16(6903),
        _mm512_abs_epi16(d),
    ));
    let r = _mm512_add_epi16(a, _mm512_mulhi_epi16(d, _mm512_slli_epi16::<10>(k)));
    (_mm512_mulhrs_epi16(r, _mm512_set1_epi16(2048)), k)
}

/// 64 öğenin `m` baytları (sıralı) ve 32 ikili toplamı (`i16`, sıralı).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn m_bayt_ve_ikili(k0: __m512i, k1: __m512i) -> (__m512i, __m512i) {
    let altmis4 = _mm512_set1_epi16(64);
    let m = paketle64(_mm512_sub_epi16(altmis4, k0), _mm512_sub_epi16(altmis4, k1));
    (m, _mm512_maddubs_epi16(m, _mm512_set1_epi8(1)))
}

/// `w_mask` gövdesi: alt tür `SUB` (0: 4:4:4, 1: 4:2:2, 2: 4:2:0), `W ≥ 32` sütun.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn w_mask_govde<const SUB: usize, const W: usize>(
    dst: &mut [u8],
    ds: usize,
    t1: &[i16],
    t2: &[i16],
    h: usize,
    mask: &mut [u8],
    sign: i16,
) {
    let n = W * h;
    if SUB < 2 {
        // 4:4:4 / 4:2:2: öğe sırasıyla; maske satırları bitişik.
        let ek = _mm512_set1_epi16(1 - sign);
        let mut o = 0;
        while o < n {
            let (r0, k0) = wm32(t1, t2, o);
            let (r1, k1) = wm32(t1, t2, o + 32);
            sakla64::<W>(dst, ds, o, paketle64(r0, r1));
            let (m, ikili) = m_bayt_ve_ikili(k0, k1);
            if SUB == 0 {
                store_u8x64(chunk_mut(mask, o), m);
            } else {
                let v = _mm512_srli_epi16::<1>(_mm512_add_epi16(ikili, ek));
                store_u8x32(chunk_mut(mask, o / 2), _mm512_cvtepi16_epi8(v));
            }
            o += 64;
        }
        return;
    }
    // 4:2:0: satır ikilileri; maske satırı `W/2` bayt.
    if W == 32 {
        // Bir adım = satır `y` (32) + satır `y+1` (32).
        let ek = _mm256_set1_epi16(2 - sign);
        let mut o = 0;
        while o < n {
            let (r0, k0) = wm32(t1, t2, o);
            let (r1, k1) = wm32(t1, t2, o + 32);
            sakla64::<W>(dst, ds, o, paketle64(r0, r1));
            let (_, s) = m_bayt_ve_ikili(k0, k1);
            let t = _mm256_add_epi16(_mm512_castsi512_si256(s), _mm512_extracti64x4_epi64::<1>(s));
            let v = _mm256_srli_epi16::<2>(_mm256_add_epi16(t, ek));
            store_u8x16_256(mask, o / 4, _mm256_cvtepi16_epi8(v));
            o += 64;
        }
        return;
    }
    let ek = _mm512_set1_epi16(2 - sign);
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < W {
            let o = y * W + x;
            let o1 = o + W;
            let (r0, k0) = wm32(t1, t2, o);
            let (r1, k1) = wm32(t1, t2, o + 32);
            let (r2, k2) = wm32(t1, t2, o1);
            let (r3, k3) = wm32(t1, t2, o1 + 32);
            sakla64::<W>(dst, ds, o, paketle64(r0, r1));
            sakla64::<W>(dst, ds, o1, paketle64(r2, r3));
            let (_, s0) = m_bayt_ve_ikili(k0, k1);
            let (_, s1) = m_bayt_ve_ikili(k2, k3);
            let v = _mm512_srli_epi16::<2>(_mm512_add_epi16(_mm512_add_epi16(s0, s1), ek));
            store_u8x32(
                chunk_mut(mask, (y / 2) * (W / 2) + x / 2),
                _mm512_cvtepi16_epi8(v),
            );
            x += 64;
        }
        y += 2;
    }
}

/// 16 baytlık bir `__m128i`'yi maskeye yazar (`vpmovwb` ymm sonucu).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn store_u8x16_256(mask: &mut [u8], o: usize, v: core::arch::x86_64::__m128i) {
    super::mem::store_u8x16(chunk_mut(mask, o), v);
}

/// `w_mask` genişlik monomorfu (`w ≥ 32`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline(never)]
pub(super) fn w_mask_w<const SUB: usize>(
    dst: &mut [u8],
    ds: usize,
    t1: &[i16],
    t2: &[i16],
    w: usize,
    h: usize,
    mask: &mut [u8],
    sign: i16,
) {
    match w {
        32 => w_mask_govde::<SUB, 32>(dst, ds, t1, t2, h, mask, sign),
        64 => w_mask_govde::<SUB, 64>(dst, ds, t1, t2, h, mask, sign),
        _ => w_mask_govde::<SUB, 128>(dst, ds, t1, t2, h, mask, sign),
    }
}

/// `avg` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn avg_fn(_avx512: Avx512IclToken) -> AvgFn {
    avx2::avg_giris_icl
}

/// `w_avg` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn w_avg_fn(_avx512: Avx512IclToken) -> WAvgFn {
    avx2::w_avg_giris_icl
}

/// `mask` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn mask_fn(_avx512: Avx512IclToken) -> MaskFn {
    avx2::mask_giris_icl
}

/// `w_mask[alt]` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn w_mask_fn(_avx512: Avx512IclToken, alt: usize) -> WMaskFn {
    match alt {
        0 => avx2::w_mask_giris_icl::<0>,
        1 => avx2::w_mask_giris_icl::<1>,
        _ => avx2::w_mask_giris_icl::<2>,
    }
}
