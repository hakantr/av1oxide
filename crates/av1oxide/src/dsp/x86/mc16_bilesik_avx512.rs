//! MC bileşik AVX512ICL çekirdekleri (16 bpc, 10/12 bit) — F4-MC e (P36):
//! `avg`, `w_avg`, `mask`, `w_mask[0..2]`.
//!
//! ## Sözleşme
//!
//! AVX2 gövdesiyle ([`super::mc16_bilesik`]) aynı: girişte
//! [`crate::dsp::mc::bilesik16_sozlesmesi`] (bit derinliği 10/12 ve tutarlı
//! `bitdepth_max`; `w_avg` ağırlığı, `w_mask` alt türü, `sign`, maske boyu);
//! sözleşme dışı girdi skalere devreder ve sayılır. Okuma tam `w·h` öğe, yazma
//! `h` satır × `2w` bayt (+ maske). Değer tanım kümesi varsayımı yoktur.
//!
//! ## Yöntem
//!
//! Aritmetik AVX2 gövdesiyle aynıdır: girdiler `punpck{l,h}wd` ile ikililenir,
//! katsayı ikilisiyle `pmaddwd` → `i32`, `+ rnd`, `psrad`, `packusdw` +
//! `pminuw(bitdepth_max)`. Yazmaç 512 bit: bir adım 32 sıralı piksel (64 bayt;
//! `punpck`/`pack` çifti 128 bitlik şerit düzenini korur). `w_mask` maskesi 8
//! bpc gövdesinin yolundan (`packuswb` + `vpermq`, `pmaddubsw`, `vpmovwb`).
//! `avg`/`w_avg`/`mask` `w ≥ 16`'da (bir adım `w = 16`'da iki satır), `w_mask`
//! `w ≥ 32`'de 512 bit; daha dar bloklarda aynı girişten (aynı sözleşme
//! denetiminden sonra) AVX2 gövdesi çağrılır.

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mc16_bilesik::{self as avx2, AVG, MASK, W_AVG};
use super::mem::{
    chunk, chunk_mut, load_i16x32, load_u8x32, store_u8x16, store_u8x32, store_u8x64,
};
use crate::cpu::Avx512IclToken;
use crate::dsp::mc::{AvgFn, MaskFn, WAvgFn, WMaskFn};
use core::arch::x86_64::{
    __m128i, __m512i, _mm_cvtsi32_si128, _mm256_add_epi16, _mm256_cvtepi16_epi8, _mm256_set1_epi16,
    _mm256_srli_epi16, _mm512_abs_epi16, _mm512_add_epi16, _mm512_add_epi32,
    _mm512_castsi512_si256, _mm512_cvtepi16_epi8, _mm512_cvtepu8_epi16, _mm512_extracti64x4_epi64,
    _mm512_madd_epi16, _mm512_maddubs_epi16, _mm512_min_epi16, _mm512_min_epu16,
    _mm512_packus_epi16, _mm512_packus_epi32, _mm512_permutexvar_epi64, _mm512_set1_epi8,
    _mm512_set1_epi16, _mm512_set1_epi32, _mm512_setr_epi64, _mm512_sra_epi32, _mm512_srli_epi16,
    _mm512_sub_epi16, _mm512_subs_epi16, _mm512_unpackhi_epi16, _mm512_unpacklo_epi16,
};

/// `prep` ara değerinden çıkarılan sabit (dav1d `PREP_BIAS`, 16 bpc).
const PREP_BIAS: i32 = 8192;

/// Çağrı sabitleri ve girdiler (AVX2 gövdesindekiyle aynı anlam).
struct Sabit<'a> {
    t1: &'a [i16],
    t2: &'a [i16],
    mask: &'a [u8],
    kat: __m512i,
    rnd: __m512i,
    sh: __m128i,
    max: __m512i,
}

impl<'a> Sabit<'a> {
    /// İşlem `op` için yuvarlama ve kaydırma (`ib` `bitdepth_max`'tan).
    #[target_feature(enable = "avx512f,avx512bw")]
    fn yeni(
        op: u8,
        t1: &'a [i16],
        t2: &'a [i16],
        mask: &'a [u8],
        kat: i32,
        bitdepth_max: u16,
    ) -> Self {
        let ib = if bitdepth_max == 4095 { 2 } else { 4 };
        let (sh, rnd) = match op {
            AVG => (ib + 1, (1 << ib) + 2 * PREP_BIAS),
            W_AVG => (ib + 4, (8 << ib) + 16 * PREP_BIAS),
            _ => (ib + 6, (32 << ib) + 64 * PREP_BIAS),
        };
        Sabit {
            t1,
            t2,
            mask,
            kat: _mm512_set1_epi32(kat),
            rnd: _mm512_set1_epi32(rnd),
            sh: _mm_cvtsi32_si128(sh),
            max: _mm512_set1_epi16(bitdepth_max as i16),
        }
    }
}

/// `i32` ağırlıklı toplamlardan 32 sıralı kırpılmış piksel.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn bitir(lo: __m512i, hi: __m512i, s: &Sabit<'_>) -> __m512i {
    let lo = _mm512_sra_epi32(_mm512_add_epi32(lo, s.rnd), s.sh);
    let hi = _mm512_sra_epi32(_mm512_add_epi32(hi, s.rnd), s.sh);
    _mm512_min_epu16(_mm512_packus_epi32(lo, hi), s.max)
}

/// Öğe ofseti `o`'dan 32 öğenin sonucu.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn hesap32<const OP: u8>(s: &Sabit<'_>, o: usize) -> __m512i {
    let a = load_i16x32(chunk(s.t1, o));
    let b = load_i16x32(chunk(s.t2, o));
    let (klo, khi) = if OP == MASK {
        let m = _mm512_cvtepu8_epi16(load_u8x32(chunk(s.mask, o)));
        let n = _mm512_sub_epi16(_mm512_set1_epi16(64), m);
        (_mm512_unpacklo_epi16(m, n), _mm512_unpackhi_epi16(m, n))
    } else {
        (s.kat, s.kat)
    };
    let lo = _mm512_madd_epi16(_mm512_unpacklo_epi16(a, b), klo);
    let hi = _mm512_madd_epi16(_mm512_unpackhi_epi16(a, b), khi);
    bitir(lo, hi, s)
}

/// Öğe ofseti `o`'dan başlayan 32 sıralı pikseli (`W` sütunlu bitişik
/// satırlar) hedefe yazar; `W = 16`'da iki satır.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn sakla32<const W: usize>(dst: &mut [u8], ds: usize, o: usize, p: __m512i) {
    let (y, x) = (o / W, o % W);
    if W == 16 {
        store_u8x32(chunk_mut(dst, y * ds), _mm512_castsi512_si256(p));
        store_u8x32(
            chunk_mut(dst, (y + 1) * ds),
            _mm512_extracti64x4_epi64::<1>(p),
        );
    } else {
        store_u8x64(chunk_mut(dst, y * ds + 2 * x), p);
    }
}

/// `avg` / `w_avg` / `mask` gövdesi: `W ≥ 16` sütun, `n = W·h` öğe.
#[target_feature(enable = "avx512f,avx512bw")]
fn bidir<const OP: u8, const W: usize>(dst: &mut [u8], ds: usize, n: usize, s: &Sabit<'_>) {
    let mut o = 0;
    while o < n {
        sakla32::<W>(dst, ds, o, hesap32::<OP>(s, o));
        o += 32;
    }
}

/// Giriş: sabitler ve genişlik monomorfu (`w ≥ 16`).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline(never)]
pub(super) fn bidir_giris<const OP: u8>(
    dst: &mut [u8],
    ds: usize,
    t1: &[i16],
    t2: &[i16],
    mask: &[u8],
    w: usize,
    h: usize,
    kat: i32,
    bitdepth_max: u16,
) {
    let s = Sabit::yeni(OP, t1, t2, mask, kat, bitdepth_max);
    let n = w * h;
    match w {
        16 => bidir::<OP, 16>(dst, ds, n, &s),
        32 => bidir::<OP, 32>(dst, ds, n, &s),
        64 => bidir::<OP, 64>(dst, ds, n, &s),
        _ => bidir::<OP, 128>(dst, ds, n, &s),
    }
}

/// `w_mask` 32 öğesi: (sıralı piksel sonucu, `m` `i16`).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn wm32(s: &Sabit<'_>, o: usize) -> (__m512i, __m512i) {
    let a = load_i16x32(chunk(s.t1, o));
    let b = load_i16x32(chunk(s.t2, o));
    let fark = _mm512_abs_epi16(_mm512_subs_epi16(a, b));
    let q = _mm512_srli_epi16::<10>(_mm512_add_epi16(fark, _mm512_set1_epi16(32)));
    let m = _mm512_min_epi16(
        _mm512_add_epi16(q, _mm512_set1_epi16(38)),
        _mm512_set1_epi16(64),
    );
    let n = _mm512_sub_epi16(_mm512_set1_epi16(64), m);
    let lo = _mm512_madd_epi16(_mm512_unpacklo_epi16(a, b), _mm512_unpacklo_epi16(m, n));
    let hi = _mm512_madd_epi16(_mm512_unpackhi_epi16(a, b), _mm512_unpackhi_epi16(m, n));
    (bitir(lo, hi, s), m)
}

/// 64 öğenin `m` baytları (sıralı) ve 32 ikili toplamı (`i16`, sıralı).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn m_bayt_ve_ikili(m0: __m512i, m1: __m512i) -> (__m512i, __m512i) {
    let m = _mm512_permutexvar_epi64(
        _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7),
        _mm512_packus_epi16(m0, m1),
    );
    (m, _mm512_maddubs_epi16(m, _mm512_set1_epi8(1)))
}

/// `w_mask` gövdesi: alt tür `SUB`, `W ≥ 32` sütun.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn w_mask_govde<const SUB: usize, const W: usize>(
    dst: &mut [u8],
    ds: usize,
    h: usize,
    mask: &mut [u8],
    sign: i16,
    s: &Sabit<'_>,
) {
    let n = W * h;
    if SUB < 2 {
        let ek = _mm512_set1_epi16(1 - sign);
        let mut o = 0;
        while o < n {
            let (r0, m0) = wm32(s, o);
            let (r1, m1) = wm32(s, o + 32);
            sakla32::<W>(dst, ds, o, r0);
            sakla32::<W>(dst, ds, o + 32, r1);
            let (m, ikili) = m_bayt_ve_ikili(m0, m1);
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
    if W == 32 {
        // Bir adım = satır `y` (32) + satır `y+1` (32).
        let ek = _mm256_set1_epi16(2 - sign);
        let mut o = 0;
        while o < n {
            let (r0, m0) = wm32(s, o);
            let (r1, m1) = wm32(s, o + 32);
            sakla32::<W>(dst, ds, o, r0);
            sakla32::<W>(dst, ds, o + 32, r1);
            let (_, t) = m_bayt_ve_ikili(m0, m1);
            let t = _mm256_add_epi16(_mm512_castsi512_si256(t), _mm512_extracti64x4_epi64::<1>(t));
            let v = _mm256_srli_epi16::<2>(_mm256_add_epi16(t, ek));
            store_u8x16(chunk_mut(mask, o / 4), _mm256_cvtepi16_epi8(v));
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
            let (r0, m0) = wm32(s, o);
            let (r1, m1) = wm32(s, o + 32);
            let (r2, m2) = wm32(s, o1);
            let (r3, m3) = wm32(s, o1 + 32);
            sakla32::<W>(dst, ds, o, r0);
            sakla32::<W>(dst, ds, o + 32, r1);
            sakla32::<W>(dst, ds, o1, r2);
            sakla32::<W>(dst, ds, o1 + 32, r3);
            let (_, s0) = m_bayt_ve_ikili(m0, m1);
            let (_, s1) = m_bayt_ve_ikili(m2, m3);
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

/// `w_mask` sabitleri ve genişlik monomorfu (`w ≥ 32`).
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
    bitdepth_max: u16,
) {
    let s = Sabit::yeni(MASK, t1, t2, &[], 0, bitdepth_max);
    match w {
        32 => w_mask_govde::<SUB, 32>(dst, ds, h, mask, sign, &s),
        64 => w_mask_govde::<SUB, 64>(dst, ds, h, mask, sign, &s),
        _ => w_mask_govde::<SUB, 128>(dst, ds, h, mask, sign, &s),
    }
}

/// 16 bpc `avg` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn avg_fn(_avx512: Avx512IclToken) -> AvgFn {
    avx2::avg16_giris_icl
}

/// 16 bpc `w_avg` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn w_avg_fn(_avx512: Avx512IclToken) -> WAvgFn {
    avx2::w_avg16_giris_icl
}

/// 16 bpc `mask` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn mask_fn(_avx512: Avx512IclToken) -> MaskFn {
    avx2::mask16_giris_icl
}

/// 16 bpc `w_mask[alt]` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn w_mask_fn(_avx512: Avx512IclToken, alt: usize) -> WMaskFn {
    match alt {
        0 => avx2::w_mask16_giris_icl::<0>,
        1 => avx2::w_mask16_giris_icl::<1>,
        _ => avx2::w_mask16_giris_icl::<2>,
    }
}
