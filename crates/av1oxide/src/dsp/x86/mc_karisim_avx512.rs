//! MC karışım AVX512ICL çekirdekleri (8 bpc) — F4-MC e (P36): `blend`,
//! `blend_v`, `blend_h`.
//!
//! ## Sözleşme
//!
//! AVX2 gövdesiyle ([`super::mc_karisim`]) aynı: girişte
//! [`crate::dsp::mc::karisim_sozlesmesi`]; sözleşme dışı girdi skalere
//! devreder ve sayılır. Okuma/yazma yalnız `w` sütunluk satırlar (`blend_h`
//! yalnız ilk `(h·3) >> 2` satır); `tmp` ve `mask` satırları bitişiktir.
//!
//! ## Yöntem
//!
//! `(d·(64−m) + t·m + 32) >> 6`, AVX2 gövdesiyle aynı aritmetik: `punpck{l,h}bw`
//! ikilileri, `pmaddubsw`, `pmulhrsw(·, 512)`, `packuswb` (128 bitlik şerit içi
//! sıra korunur). Yazmaç 512 bit: `w = 32`'de iki hedef satırı bir zmm'e toplanır
//! (`tmp`/`mask` bitişik olduğundan doğrudan 64 bayt); `blend_h` `w ≥ 64`'te satır
//! başına zmm. `w < 32` bloklarda aynı girişten AVX2 gövdeleri çağrılır (P36 hücre
//! ölçümü: `w = 16`'da dört satırı xmm parçalarıyla toplamak AVX2 yolundan yavaştı;
//! AVX2 girişine devretmek çağrı başına sabit maliyet ekliyordu).

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mc_karisim as avx2;
use super::mem::{chunk, chunk_mut, load_u8x32, load_u8x64, store_u8x32, store_u8x64};
use crate::cpu::Avx512IclToken;
use crate::dsp::mc::{BlendDirFn, BlendFn};
use crate::dsp::mc_tables::OBMC_MASKS;
use core::arch::x86_64::{
    __m512i, _mm256_set1_epi8, _mm512_castsi256_si512, _mm512_castsi512_si256,
    _mm512_extracti64x4_epi64, _mm512_inserti64x4, _mm512_maddubs_epi16, _mm512_mulhrs_epi16,
    _mm512_packus_epi16, _mm512_set1_epi8, _mm512_set1_epi16, _mm512_sub_epi8,
    _mm512_unpackhi_epi8, _mm512_unpacklo_epi8, _mm512_zextsi256_si512,
};

/// `(d·(64−m) + t·m + 32) >> 6`, 64 piksel; `m` baytları `mb`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn harmanla(d: __m512i, t: __m512i, mb: __m512i) -> __m512i {
    let ters = _mm512_sub_epi8(_mm512_set1_epi8(64), mb);
    let wlo = _mm512_unpacklo_epi8(ters, mb);
    let whi = _mm512_unpackhi_epi8(ters, mb);
    let k = _mm512_set1_epi16(512);
    let lo = _mm512_mulhrs_epi16(_mm512_maddubs_epi16(_mm512_unpacklo_epi8(d, t), wlo), k);
    let hi = _mm512_mulhrs_epi16(_mm512_maddubs_epi16(_mm512_unpackhi_epi8(d, t), whi), k);
    _mm512_packus_epi16(lo, hi)
}

/// Satır `y` ve `y+1`'in 32 baytı bir zmm'de (`[satır y | satır y+1]`).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn topla(d: &[u8], ds: usize, y: usize) -> __m512i {
    _mm512_inserti64x4::<1>(
        _mm512_castsi256_si512(load_u8x32(chunk(d, y * ds))),
        load_u8x32(chunk(d, (y + 1) * ds)),
    )
}

/// [`topla`]'nın tersi.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn dagit(d: &mut [u8], ds: usize, y: usize, v: __m512i) {
    store_u8x32(chunk_mut(d, y * ds), _mm512_castsi512_si256(v));
    store_u8x32(
        chunk_mut(d, (y + 1) * ds),
        _mm512_extracti64x4_epi64::<1>(v),
    );
}

/// `blend` gövdesi, `w = 32`: satır çiftleri (`h` çift).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline(never)]
pub(super) fn blend_govde(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize, mask: &[u8]) {
    for y in (0..h).step_by(2) {
        let o = y * 32;
        let r = harmanla(
            topla(dst, ds, y),
            load_u8x64(chunk(tmp, o)),
            load_u8x64(chunk(mask, o)),
        );
        dagit(dst, ds, y, r);
    }
}

/// `blend_v` sütun maskesi `OBMC_MASKS[32 + x]`, iki satıra (64 bayt) yayılı.
static BLEND_V_MASKE: [u8; 64] = {
    let mut b = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        b[i] = OBMC_MASKS[32 + i % 32];
        i += 1;
    }
    b
};

/// `blend_v` gövdesi, `w = 32`: satır çiftleri (`h` çift).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline(never)]
pub(super) fn blend_v_govde(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize) {
    let m = load_u8x64(&BLEND_V_MASKE);
    for y in (0..h).step_by(2) {
        let r = harmanla(topla(dst, ds, y), load_u8x64(chunk(tmp, y * 32)), m);
        dagit(dst, ds, y, r);
    }
}

/// `blend_h` gövdesi, `W ≥ 32`: satır maskesi `OBMC_MASKS[h + y]`, ilk
/// `(h·3) >> 2` satır; `W = 32`'de iki satır bir zmm'de.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline(never)]
pub(super) fn blend_h_govde<const W: usize>(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize) {
    let satirlar = (h * 3) >> 2;
    let tablo = &OBMC_MASKS[h..];
    if W >= 64 {
        for (y, &my) in tablo.iter().enumerate().take(satirlar) {
            let m = _mm512_set1_epi8(my as i8);
            for x in (0..W).step_by(64) {
                let d = load_u8x64(chunk(dst, y * ds + x));
                let r = harmanla(d, load_u8x64(chunk(tmp, y * W + x)), m);
                store_u8x64(chunk_mut(dst, y * ds + x), r);
            }
        }
        return;
    }
    // `W = 32`: satır çiftleri; tek kalan son satır AVX2 yolunun 32 baytlık
    // adımıyla aynı aritmetik (zmm'in alt yarısı).
    let mut y = 0;
    while y < satirlar {
        if y + 1 < satirlar {
            let m = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_set1_epi8(tablo[y] as i8)),
                _mm256_set1_epi8(tablo[y + 1] as i8),
            );
            let r = harmanla(topla(dst, ds, y), load_u8x64(chunk(tmp, y * 32)), m);
            dagit(dst, ds, y, r);
            y += 2;
        } else {
            let m = _mm512_set1_epi8(tablo[y] as i8);
            let d = _mm512_zextsi256_si512(load_u8x32(chunk(dst, y * ds)));
            let t = _mm512_zextsi256_si512(load_u8x32(chunk(tmp, y * 32)));
            let r = harmanla(d, t, m);
            store_u8x32(chunk_mut(dst, y * ds), _mm512_castsi512_si256(r));
            y += 1;
        }
    }
}

/// `blend` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn blend_fn(_avx512: Avx512IclToken) -> BlendFn {
    avx2::blend_giris_icl
}

/// `blend_v` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn blend_v_fn(_avx512: Avx512IclToken) -> BlendDirFn {
    avx2::blend_v_giris_icl
}

/// `blend_h` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn blend_h_fn(_avx512: Avx512IclToken) -> BlendDirFn {
    avx2::blend_h_giris_icl
}
