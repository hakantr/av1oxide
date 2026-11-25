//! MC karışım AVX512ICL çekirdekleri (16 bpc, 10/12 bit) — F4-MC e (P36):
//! `blend`, `blend_v`, `blend_h`.
//!
//! ## Sözleşme
//!
//! AVX2 gövdesiyle ([`super::mc16_karisim`]) aynı: girişte
//! [`crate::dsp::mc::karisim16_sozlesmesi`]; sözleşme dışı girdi skalere
//! devreder ve sayılır. Okuma/yazma yalnız `w` piksellik satırlar (`blend_h`
//! yalnız ilk `(h·3) >> 2` satır); `tmp` ve `mask` satırları bitişiktir. Değer
//! tanım kümesi AVX2 gövdesininkiyle aynıdır (pikseller `≤ bitdepth_max`,
//! maske 0…64).
//!
//! ## Yöntem
//!
//! AVX2 gövdesinin aritmetiği 512 bitte: `punpck{l,h}wd` ikilileri, `pmaddwd`,
//! `+ 32`, `psrad 6`, `packusdw` (128 bitlik şerit içi sıra korunur). Bir zmm
//! 32 piksel: `w ≥ 32` satır başına 32'lik parçalar, `w = 16`'da iki hedef satırı
//! bir zmm'e toplanır (`tmp`/`mask` bitişik). `w < 16` bloklarda aynı girişten AVX2
//! gövdeleri çağrılır (P36 hücre ölçümü: `w = 8`'de dört satırı xmm parçalarıyla
//! toplamak AVX2 yolundan yavaştı; AVX2 girişine devretmek çağrı başına sabit
//! maliyet ekliyordu).

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mc16_karisim as avx2;
use super::mem::{chunk, chunk_mut, load_u8x32, load_u8x64, store_u8x32, store_u8x64};
use crate::cpu::Avx512IclToken;
use crate::dsp::mc::{BlendDirFn, BlendFn};
use crate::dsp::mc_tables::OBMC_MASKS;
use core::arch::x86_64::{
    __m512i, _mm256_set1_epi16, _mm512_add_epi32, _mm512_castsi256_si512, _mm512_castsi512_si256,
    _mm512_cvtepu8_epi16, _mm512_extracti64x4_epi64, _mm512_inserti64x4, _mm512_madd_epi16,
    _mm512_packus_epi32, _mm512_set1_epi16, _mm512_set1_epi32, _mm512_srai_epi32, _mm512_sub_epi16,
    _mm512_unpackhi_epi16, _mm512_unpacklo_epi16, _mm512_zextsi256_si512,
};

/// 32 pikselin karışımı `(d·(64−m) + t·m + 32) >> 6`: `d`, `t` sıralı `u16`,
/// `m` sıralı `i16` maske.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn harmanla(d: __m512i, t: __m512i, m: __m512i) -> __m512i {
    let n = _mm512_sub_epi16(_mm512_set1_epi16(64), m);
    let k = _mm512_set1_epi32(32);
    let lo = _mm512_madd_epi16(_mm512_unpacklo_epi16(d, t), _mm512_unpacklo_epi16(n, m));
    let hi = _mm512_madd_epi16(_mm512_unpackhi_epi16(d, t), _mm512_unpackhi_epi16(n, m));
    let lo = _mm512_srai_epi32::<6>(_mm512_add_epi32(lo, k));
    let hi = _mm512_srai_epi32::<6>(_mm512_add_epi32(hi, k));
    _mm512_packus_epi32(lo, hi)
}

/// 32 maske baytı `s[o..o+32]` → sıralı `i16`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn maske32(s: &[u8], o: usize) -> __m512i {
    _mm512_cvtepu8_epi16(load_u8x32(chunk(s, o)))
}

/// Satır `y` ve `y+1`'in 16 pikseli (32 bayt) bir zmm'de.
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

/// `blend` gövdesi, `W` sütun: `W ≥ 32` satır başına 32'lik parçalar, `W = 16`
/// zmm başına iki satır (`h` 2'nin kuvveti ≥ 4: tam çiftler).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline(never)]
pub(super) fn blend_govde<const W: usize>(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    h: usize,
    mask: &[u8],
) {
    if W >= 32 {
        for y in 0..h {
            let (o, od) = (y * W, y * ds);
            let d = load_u8x64(chunk(dst, od));
            let r = harmanla(d, load_u8x64(chunk(tmp, 2 * o)), maske32(mask, o));
            store_u8x64(chunk_mut(dst, od), r);
        }
        return;
    }
    for y in (0..h).step_by(2) {
        let o = y * W;
        let r = harmanla(
            topla(dst, ds, y),
            load_u8x64(chunk(tmp, 2 * o)),
            maske32(mask, o),
        );
        dagit(dst, ds, y, r);
    }
}

/// `blend_v` sütun maskesi `OBMC_MASKS[W + x]`, 32 bayta yayılı (`W` = 16, 32;
/// AVX2 tablosuyla aynı, kendi kopyası).
static BLEND_V_MASKE: [[u8; 32]; 2] = [blend_v_maske(16), blend_v_maske(32)];

const fn blend_v_maske(w: usize) -> [u8; 32] {
    let mut b = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        b[i] = OBMC_MASKS[w + i % w];
        i += 1;
    }
    b
}

/// `blend_v` gövdesi (`W` = 16, 32): sütun maskesi satırlara yayılı; `W = 16`'da
/// satır çiftleri (`h` çift).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline(never)]
pub(super) fn blend_v_govde<const W: usize>(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize) {
    let m = maske32(&BLEND_V_MASKE[usize::from(W == 32)], 0);
    if W == 32 {
        for y in 0..h {
            let od = y * ds;
            let d = load_u8x64(chunk(dst, od));
            let r = harmanla(d, load_u8x64(chunk(tmp, 2 * y * W)), m);
            store_u8x64(chunk_mut(dst, od), r);
        }
        return;
    }
    for y in (0..h).step_by(2) {
        let r = harmanla(topla(dst, ds, y), load_u8x64(chunk(tmp, 2 * y * W)), m);
        dagit(dst, ds, y, r);
    }
}

/// `blend_h` gövdesi (`W ≥ 16`): satır maskesi `OBMC_MASKS[h + y]`, ilk
/// `(h·3) >> 2` satır; `W = 16`'da iki satır bir zmm'de.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline(never)]
pub(super) fn blend_h_govde<const W: usize>(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize) {
    let satirlar = (h * 3) >> 2;
    let tablo = &OBMC_MASKS[h..];
    if W >= 32 {
        for (y, &my) in tablo.iter().enumerate().take(satirlar) {
            let m = _mm512_set1_epi16(i16::from(my));
            for x in (0..W).step_by(32) {
                let od = y * ds + 2 * x;
                let d = load_u8x64(chunk(dst, od));
                let r = harmanla(d, load_u8x64(chunk(tmp, 2 * (y * W + x))), m);
                store_u8x64(chunk_mut(dst, od), r);
            }
        }
        return;
    }
    // `W = 16`: satır çiftleri; tek kalan son satır zmm'in alt yarısında.
    let mut y = 0;
    while y < satirlar {
        if y + 1 < satirlar {
            let m = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_set1_epi16(i16::from(tablo[y]))),
                _mm256_set1_epi16(i16::from(tablo[y + 1])),
            );
            let r = harmanla(topla(dst, ds, y), load_u8x64(chunk(tmp, 2 * y * 16)), m);
            dagit(dst, ds, y, r);
            y += 2;
        } else {
            let m = _mm512_set1_epi16(i16::from(tablo[y]));
            let d = _mm512_zextsi256_si512(load_u8x32(chunk(dst, y * ds)));
            let t = _mm512_zextsi256_si512(load_u8x32(chunk(tmp, 2 * y * 16)));
            store_u8x32(
                chunk_mut(dst, y * ds),
                _mm512_castsi512_si256(harmanla(d, t, m)),
            );
            y += 1;
        }
    }
}

/// 16 bpc `blend` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn blend_fn(_avx512: Avx512IclToken) -> BlendFn {
    avx2::blend16_giris_icl
}

/// 16 bpc `blend_v` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn blend_v_fn(_avx512: Avx512IclToken) -> BlendDirFn {
    avx2::blend_v16_giris_icl
}

/// 16 bpc `blend_h` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn blend_h_fn(_avx512: Avx512IclToken) -> BlendDirFn {
    avx2::blend_h16_giris_icl
}
