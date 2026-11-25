//! MC AVX512ICL çekirdekleri (8 bpc) — F4-MC e (P36): `put` (`mc[0..9]`) ve
//! `prep` (`mct[0..9]`).
//!
//! ## Sözleşme
//!
//! AVX2 gövdesinin kopyasız sözleşmesi ([`super::mc`]): girişte O(1)
//! [`crate::dsp::mc::simd_sozlesmesi`]; sözleşme dışı girdi skalere devreder ve
//! sayılır. Giriş AVX2 modülündedir ([`super::mc::put_giris_icl`],
//! [`super::mc::prep_giris_icl`]): aynı denetimden sonra `w < 32` blokları AVX2
//! gövdelerine (AVX2 girişinin yol seçimiyle), `w ≥ 32` blokları bu modülün
//! [`genis`] gövdesine verir. Bu modülde `unsafe` yoktur. Yatay pencere **maskeli** yüklenir: maske yalnız okuma
//! bölgesinin baytlarını seçer (maskesiz baytlar okunmaz), dilim denetimi tam
//! pencere boyudur; AVX2 yolunun [`crate::dsp::mc::MC_KUYRUK`] aşımı da yoktur.
//!
//! ## Yöntem
//!
//! AVX2 gövdesinin aritmetiği (tap sayısı, `pmaddubsw`/`pmaddwd`, yuvarlama)
//! 512 bitte, 32 sütunluk şeritlerle:
//!
//! - **Yatay:** şeridin `31+N` baytlık penceresi bir zmm'e maskeli yüklenir;
//!   ikili `k` için `vpermb` (VBMI) her 128 bitlik lane'e 8 çıktının
//!   `(s[x+S+2k], s[x+S+2k+1])` ikililerini dizer, `pmaddubsw`.
//! - **Dikey:** iki satır bir zmm'de (`[satır j | satır j+1]`, 32 bayt),
//!   `punpck*bw` ile iki çıkış satırının ikilileri birden; ikililer kayan
//!   pencerede.
//! - **İki boyutlu:** yatay geçiş `i16` ara satır (zmm, 32 sütun), dikey
//!   `punpck*wd` + `pmaddwd`.
//! - **Saklama:** `put` `packuswb` lane içidir; `vpermq` dizinleri sırayı geri
//!   kurar. `prep` doğrudan 64 baytlık saklama (dikey yolda `vpermt2q`).

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mc as avx2;
use super::mc::{BILIN, Hedef, filtre_satiri};
use super::mem::{
    chunk, chunk_mut, load_u8_kz, load_u8x32, load_u8x64, store_i16x32, store_u8x32, store_u8x64,
};
use crate::cpu::Avx512IclToken;
use crate::dsp::mc::{Filter2d, McPrepFn, McPutFn};
use core::arch::x86_64::{
    __m256i, __m512i, _mm512_add_epi16, _mm512_add_epi32, _mm512_castsi256_si512,
    _mm512_castsi512_si256, _mm512_cvtepu8_epi16, _mm512_extracti64x4_epi64, _mm512_inserti64x4,
    _mm512_madd_epi16, _mm512_maddubs_epi16, _mm512_mulhrs_epi16, _mm512_mullo_epi16,
    _mm512_packs_epi32, _mm512_packus_epi16, _mm512_permutex2var_epi64, _mm512_permutexvar_epi8,
    _mm512_permutexvar_epi64, _mm512_set1_epi16, _mm512_set1_epi32, _mm512_setr_epi64,
    _mm512_setzero_si512, _mm512_slli_epi16, _mm512_srai_epi16, _mm512_srai_epi32,
    _mm512_srli_epi16, _mm512_unpackhi_epi8, _mm512_unpackhi_epi16, _mm512_unpacklo_epi8,
    _mm512_unpacklo_epi16,
};

// ---------------------------------------------------------------------------
// Dış yüzey (belirteçli)
// ---------------------------------------------------------------------------

/// `mc[filtre]` (8 bpc) yuvasının AVX512ICL girişi; yalnız [`Avx512IclToken`] ile.
#[must_use]
pub const fn put_fn(_avx512: Avx512IclToken, filtre: Filter2d) -> McPutFn {
    match filtre {
        Filter2d::Regular => avx2::put_giris_icl::<0, 0>,
        Filter2d::RegularSmooth => avx2::put_giris_icl::<0, 1>,
        Filter2d::RegularSharp => avx2::put_giris_icl::<0, 2>,
        Filter2d::SharpRegular => avx2::put_giris_icl::<2, 0>,
        Filter2d::SharpSmooth => avx2::put_giris_icl::<2, 1>,
        Filter2d::Sharp => avx2::put_giris_icl::<2, 2>,
        Filter2d::SmoothRegular => avx2::put_giris_icl::<1, 0>,
        Filter2d::Smooth => avx2::put_giris_icl::<1, 1>,
        Filter2d::SmoothSharp => avx2::put_giris_icl::<1, 2>,
        Filter2d::Bilinear => avx2::put_giris_icl::<BILIN, BILIN>,
    }
}

/// `mct[filtre]` (8 bpc) yuvasının AVX512ICL girişi; yalnız [`Avx512IclToken`] ile.
#[must_use]
pub const fn prep_fn(_avx512: Avx512IclToken, filtre: Filter2d) -> McPrepFn {
    match filtre {
        Filter2d::Regular => avx2::prep_giris_icl::<0, 0>,
        Filter2d::RegularSmooth => avx2::prep_giris_icl::<0, 1>,
        Filter2d::RegularSharp => avx2::prep_giris_icl::<0, 2>,
        Filter2d::SharpRegular => avx2::prep_giris_icl::<2, 0>,
        Filter2d::SharpSmooth => avx2::prep_giris_icl::<2, 1>,
        Filter2d::Sharp => avx2::prep_giris_icl::<2, 2>,
        Filter2d::SmoothRegular => avx2::prep_giris_icl::<1, 0>,
        Filter2d::Smooth => avx2::prep_giris_icl::<1, 1>,
        Filter2d::SmoothSharp => avx2::prep_giris_icl::<1, 2>,
        Filter2d::Bilinear => avx2::prep_giris_icl::<BILIN, BILIN>,
    }
}

/// Geniş yol (`w ≥ 32`) dağıtımı: [`super::mc::put_giris_icl`] /
/// [`super::mc::prep_giris_icl`] sözleşme denetiminden sonra çağırır.
/// `#[inline(never)]`: ThinLTO bu gövdeyi AVX2 modülünün birimine almaz
/// (alırsa iç çağrılananları dış bağlantıya yükseltilirdi).
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
#[inline(never)]
pub(super) fn genis<const FH: usize, const FV: usize, const PREP: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    match (mx != 0, my != 0) {
        (false, false) => kopya::<PREP>(hd, src, ss, w, h),
        (true, false) => yol_h::<FH, PREP>(hd, src, ss, w, h, mx),
        (false, true) => yol_v::<FV, PREP>(hd, src, ss, w, h, my),
        (true, true) => yol_hv::<FH, FV, PREP>(hd, src, ss, w, h, mx, my),
    }
}

// ---------------------------------------------------------------------------
// Yol seçimi
// ---------------------------------------------------------------------------

/// Tap sayısı (`w`, `h` ≥ 32 dalında boyut > 4): `sharp` → 8; aksi hâlde 6.
/// Dikeyde `h ≤ 4` 4 tap (AVX2 gövdesiyle aynı kural).
const fn tap(t: usize, boyut: usize) -> usize {
    if boyut <= 4 {
        4
    } else if t == 2 {
        8
    } else {
        6
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
fn yol_h<const FH: usize, const PREP: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
) {
    if FH == BILIN {
        return yatay::<2, PREP, true>(hd, src, ss, w, h, &kat_bilin(mx));
    }
    let f = filtre_satiri(FH, w, mx);
    match tap(FH, w) {
        8 => yatay::<8, PREP, false>(hd, src, ss, w, h, &kat_b::<8>(f)),
        _ => yatay::<6, PREP, false>(hd, src, ss, w, h, &kat_b::<6>(f)),
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
fn yol_v<const FV: usize, const PREP: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    my: u8,
) {
    if FV == BILIN {
        return dikey::<2, PREP, true>(hd, src, ss, w, h, &kat_bilin(my));
    }
    let f = filtre_satiri(FV, h, my);
    match tap(FV, h) {
        8 => dikey::<8, PREP, false>(hd, src, ss, w, h, &kat_b::<8>(f)),
        6 => dikey::<6, PREP, false>(hd, src, ss, w, h, &kat_b::<6>(f)),
        _ => dikey::<4, PREP, false>(hd, src, ss, w, h, &kat_b::<4>(f)),
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
fn yol_hv<const FH: usize, const FV: usize, const PREP: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    if FH == BILIN {
        return iki::<2, 2, PREP, true>(hd, src, ss, w, h, &kat_bilin(mx), &kat_bilin_w(my));
    }
    let fh = filtre_satiri(FH, w, mx);
    let fv = filtre_satiri(FV, h, my);
    macro_rules! hv {
        ($nh:literal, $nv:literal) => {
            iki::<$nh, $nv, PREP, false>(hd, src, ss, w, h, &kat_b::<$nh>(fh), &kat_w::<$nv>(fv))
        };
    }
    match (tap(FH, w), tap(FV, h)) {
        (8, 8) => hv!(8, 8),
        (8, 6) => hv!(8, 6),
        (8, _) => hv!(8, 4),
        (_, 8) => hv!(6, 8),
        (_, 6) => hv!(6, 6),
        _ => hv!(6, 4),
    }
}

// ---------------------------------------------------------------------------
// Katsayılar, dizinler, yuvarlama
// ---------------------------------------------------------------------------

/// `pmaddubsw` ikilileri: `k < N/2` için `(f[S+2k], f[S+2k+1])` baytları.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn kat_b<const N: usize>(f: &[i8; 8]) -> [__m512i; 4] {
    let s = (8 - N) / 2;
    let mut c = [_mm512_setzero_si512(); 4];
    for (k, ck) in c.iter_mut().enumerate().take(N / 2) {
        let lo = f[s + 2 * k] as u8;
        let hi = f[s + 2 * k + 1] as u8;
        *ck = _mm512_set1_epi16(i16::from_le_bytes([lo, hi]));
    }
    c
}

/// Bilinear `pmaddubsw` ikilisi `(16 − m, m)`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn kat_bilin(m: u8) -> [__m512i; 4] {
    let mut c = [_mm512_setzero_si512(); 4];
    c[0] = _mm512_set1_epi16(i16::from_le_bytes([16 - m, m]));
    c
}

/// `pmaddwd` ikilileri (iki boyutlu dikey geçiş).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn kat_w<const N: usize>(f: &[i8; 8]) -> [__m512i; 4] {
    let s = (8 - N) / 2;
    let mut c = [_mm512_setzero_si512(); 4];
    for (k, ck) in c.iter_mut().enumerate().take(N / 2) {
        let lo = i32::from(f[s + 2 * k]) & 0xffff;
        let hi = i32::from(f[s + 2 * k + 1]) << 16;
        *ck = _mm512_set1_epi32(hi | lo);
    }
    c
}

/// Bilinear iki boyutlu dikey: [1], [2] `u16` yolu için `16 − m` ve `m`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn kat_bilin_w(m: u8) -> [__m512i; 4] {
    let mut c = [_mm512_setzero_si512(); 4];
    c[1] = _mm512_set1_epi16(16 - i16::from(m));
    c[2] = _mm512_set1_epi16(i16::from(m));
    c
}

/// `vpermb` dizinleri: ikili `k`, lane `L` (çıktı `8L..8L+7`), çıktı `i` →
/// pencere baytları `8L+i+2k`, `8L+i+2k+1`.
static IKILI: [[u8; 64]; 4] = [ikili(0), ikili(1), ikili(2), ikili(3)];

const fn ikili(k: usize) -> [u8; 64] {
    let mut m = [0u8; 64];
    let mut l = 0;
    while l < 4 {
        let mut i = 0;
        while i < 8 {
            m[16 * l + 2 * i] = (8 * l + i + 2 * k) as u8;
            m[16 * l + 2 * i + 1] = (8 * l + i + 2 * k + 1) as u8;
            i += 1;
        }
        l += 1;
    }
    m
}

#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn ikili_dizin() -> [__m512i; 4] {
    core::array::from_fn(|k| load_u8x64(&IKILI[k]))
}

/// Yalnız yatay: `put` `(Σ+34)>>6`, `prep` `(Σ+2)>>2`; bilinear `put`
/// `(Σ+8)>>4`, `prep` `Σ`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn yuv_h<const PREP: bool, const BIL: bool>(v: __m512i) -> __m512i {
    match (PREP, BIL) {
        (false, false) => _mm512_srai_epi16::<6>(_mm512_add_epi16(v, _mm512_set1_epi16(34))),
        (true, false) => _mm512_mulhrs_epi16(v, _mm512_set1_epi16(8192)),
        (false, true) => _mm512_mulhrs_epi16(v, _mm512_set1_epi16(2048)),
        (true, true) => v,
    }
}

/// Yalnız dikey: `put` `(Σ+32)>>6`, `prep` `(Σ+2)>>2`; bilinear `put`
/// `(Σ+8)>>4`, `prep` `Σ`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn yuv_v<const PREP: bool, const BIL: bool>(v: __m512i) -> __m512i {
    match (PREP, BIL) {
        (false, false) => _mm512_mulhrs_epi16(v, _mm512_set1_epi16(512)),
        (true, false) => _mm512_mulhrs_epi16(v, _mm512_set1_epi16(8192)),
        (false, true) => _mm512_mulhrs_epi16(v, _mm512_set1_epi16(2048)),
        (true, true) => v,
    }
}

/// İki boyutlu ara satır: 8-tap `(Σ+2)>>2`, bilinear `Σ`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn yuv_ara<const BIL: bool>(v: __m512i) -> __m512i {
    if BIL {
        v
    } else {
        _mm512_mulhrs_epi16(v, _mm512_set1_epi16(8192))
    }
}

/// İki boyutlu son adım (`i32`): `put` `(Σ+512)>>10`, `prep` `(Σ+32)>>6`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn yuv_son<const PREP: bool>(v: __m512i) -> __m512i {
    if PREP {
        _mm512_srai_epi32::<6>(_mm512_add_epi32(v, _mm512_set1_epi32(32)))
    } else {
        _mm512_srai_epi32::<10>(_mm512_add_epi32(v, _mm512_set1_epi32(512)))
    }
}

/// Bilinear iki boyutlu dikey adım `u16`'da (AVX2 gövdesindeki gerekçe:
/// `t ≤ 65 280`): `put` `(t+128)>>8`, `prep` `(t+8)>>4`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn bilin_dikey<const PREP: bool>(a: __m512i, b: __m512i, cv: &[__m512i; 4]) -> __m512i {
    let t = _mm512_add_epi16(_mm512_mullo_epi16(a, cv[1]), _mm512_mullo_epi16(b, cv[2]));
    if PREP {
        _mm512_srli_epi16::<4>(_mm512_add_epi16(t, _mm512_set1_epi16(8)))
    } else {
        _mm512_srli_epi16::<8>(_mm512_add_epi16(t, _mm512_set1_epi16(128)))
    }
}

// ---------------------------------------------------------------------------
// Saklama
// ---------------------------------------------------------------------------

/// `packuswb(a, b)` lane içi qword sırası `[a0 b0 a1 b1 a2 b2 a3 b3]` → sıralı.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn paket64(a: __m512i, b: __m512i) -> __m512i {
    let sira = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    _mm512_permutexvar_epi64(sira, _mm512_packus_epi16(a, b))
}

/// 32 `i16` → `put` 32 bayt ya da `prep` 32 öğe, satır `y` sütun `x`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn sakla32<const PREP: bool>(hd: &mut Hedef<'_>, w: usize, y: usize, x: usize, v: __m512i) {
    if PREP {
        store_i16x32(chunk_mut(hd.tmp, y * w + x), v);
    } else {
        let b: __m256i = _mm512_castsi512_si256(paket64(v, v));
        store_u8x32(chunk_mut(hd.dst, y * hd.ds + x), b);
    }
}

/// 64 `i16` (`a` sütun `x..x+32`, `b` `x+32..x+64`).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn sakla64<const PREP: bool>(
    hd: &mut Hedef<'_>,
    w: usize,
    y: usize,
    x: usize,
    a: __m512i,
    b: __m512i,
) {
    if PREP {
        store_i16x32(chunk_mut(hd.tmp, y * w + x), a);
        store_i16x32(chunk_mut(hd.tmp, y * w + x + 32), b);
    } else {
        store_u8x64(chunk_mut(hd.dst, y * hd.ds + x), paket64(a, b));
    }
}

/// İki çıkış satırı × 32 (dikey yol): `a0` lane'leri `[y: 0..7, y: 16..23,
/// y+1: 0..7, y+1: 16..23]`, `a1` aynı düzende `8..15`, `24..31`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn sakla32x2<const PREP: bool>(
    hd: &mut Hedef<'_>,
    w: usize,
    y: usize,
    x: usize,
    a0: __m512i,
    a1: __m512i,
) {
    if PREP {
        let s0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
        let s1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
        store_i16x32(
            chunk_mut(hd.tmp, y * w + x),
            _mm512_permutex2var_epi64(a0, s0, a1),
        );
        store_i16x32(
            chunk_mut(hd.tmp, (y + 1) * w + x),
            _mm512_permutex2var_epi64(a0, s1, a1),
        );
    } else {
        // `packuswb` lane içi: lane 0..1 satır `y` sütun 0..31, lane 2..3 satır `y+1`.
        let p = _mm512_packus_epi16(a0, a1);
        let ds = hd.ds;
        store_u8x32(chunk_mut(hd.dst, y * ds + x), _mm512_castsi512_si256(p));
        store_u8x32(
            chunk_mut(hd.dst, (y + 1) * ds + x),
            _mm512_extracti64x4_epi64::<1>(p),
        );
    }
}

// ---------------------------------------------------------------------------
// Yatay toplam
// ---------------------------------------------------------------------------

/// 32 çıktı: `base` = satırın dolgu ofseti + şeridin blok sütunu. Pencere
/// `base+S .. base+S+31+N` (`W = 31+N` bayt) maskeli yüklenir.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
#[inline]
fn h32<const N: usize, const W: usize>(
    src: &[u8],
    base: usize,
    c: &[__m512i; 4],
    ix: &[__m512i; 4],
) -> __m512i {
    const { assert!(W == 31 + N && N.is_multiple_of(2) && N >= 2 && N <= 8) };
    let s = (8 - N) / 2;
    let v = load_u8_kz::<W>(chunk(src, base + s));
    let mut acc = _mm512_setzero_si512();
    for k in 0..N / 2 {
        let p = _mm512_permutexvar_epi8(ix[k], v);
        acc = _mm512_add_epi16(acc, _mm512_maddubs_epi16(p, c[k]));
    }
    acc
}

/// [`h32`]'yi `N`'ye uyan pencere boyuyla çağırır.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
#[inline]
fn h32w<const N: usize>(src: &[u8], base: usize, c: &[__m512i; 4], ix: &[__m512i; 4]) -> __m512i {
    match N {
        8 => h32::<8, 39>(src, base, c, ix),
        6 => h32::<6, 37>(src, base, c, ix),
        4 => h32::<4, 35>(src, base, c, ix),
        _ => h32::<2, 33>(src, base, c, ix),
    }
}

/// Kayan pencere: ilk `n` öğe bir sola kayar, `n-1`. öğe `yeni` olur (AVX2
/// gövdesindeki gerekçe: yeni dizi öğeleri yazmaçta tutar).
#[inline(always)]
fn kaydir<T: Copy, const K: usize>(a: [T; K], n: usize, yeni: T) -> [T; K] {
    core::array::from_fn(|k| {
        if k + 1 < n {
            a[k + 1]
        } else if k + 1 == n {
            yeni
        } else {
            a[k]
        }
    })
}

// ---------------------------------------------------------------------------
// Yollar (`w ≥ 32`)
// ---------------------------------------------------------------------------

/// Kopya: `put` satır başına 32/64 baytlık kopya, `prep` `vpmovzxbw` +
/// `vpsllw 4` (32 piksel).
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
fn kopya<const PREP: bool>(hd: &mut Hedef<'_>, src: &[u8], ss: usize, w: usize, h: usize) {
    let mut so = 3 * ss + 3;
    if PREP {
        for y in 0..h {
            for x in (0..w).step_by(32) {
                let v = _mm512_cvtepu8_epi16(load_u8x32(chunk(src, so + x)));
                store_i16x32(chunk_mut(hd.tmp, y * w + x), _mm512_slli_epi16::<4>(v));
            }
            so += ss;
        }
    } else {
        let ds = hd.ds;
        let mut d = 0;
        if w == 32 {
            for _ in 0..h {
                store_u8x32(chunk_mut(hd.dst, d), load_u8x32(chunk(src, so)));
                so += ss;
                d += ds;
            }
        } else {
            for _ in 0..h {
                for x in (0..w).step_by(64) {
                    store_u8x64(chunk_mut(hd.dst, d + x), load_u8x64(chunk(src, so + x)));
                }
                so += ss;
                d += ds;
            }
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
#[inline]
fn yatay<const N: usize, const PREP: bool, const BIL: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    c: &[__m512i; 4],
) {
    let ix = ikili_dizin();
    if w == 32 {
        for y in 0..h {
            let v = yuv_h::<PREP, BIL>(h32w::<N>(src, (y + 3) * ss, c, &ix));
            sakla32::<PREP>(hd, w, y, 0, v);
        }
        return;
    }
    for y in 0..h {
        let o = (y + 3) * ss;
        for x in (0..w).step_by(64) {
            let a = yuv_h::<PREP, BIL>(h32w::<N>(src, o + x, c, &ix));
            let b = yuv_h::<PREP, BIL>(h32w::<N>(src, o + x + 32, c, &ix));
            sakla64::<PREP>(hd, w, y, x, a, b);
        }
    }
}

/// `[a | b]` (iki 32 baytlık satır) zmm.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn cift(a: __m256i, b: __m256i) -> __m512i {
    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(a), b)
}

/// Yalnız dikey, 32 sütunluk şerit, iki çıkış satırı bir zmm'de: `X_j =
/// [satır j | satır j+1]`; ikili `k` `unpack(X_j, X_{j+1})` (`j = y+S+2k`) iki
/// satırın ikilisini birden kurar; ikililer iki satırlık kayan pencerede.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
#[inline]
fn dikey<const N: usize, const PREP: bool, const BIL: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    c: &[__m512i; 4],
) {
    let s = (8 - N) / 2;
    let n = N / 2;
    for x in (0..w).step_by(32) {
        let yuk = |j: usize| load_u8x32(chunk(src, j * ss + 3 + x));
        let mut lo = [_mm512_setzero_si512(); 4];
        let mut hi = [_mm512_setzero_si512(); 4];
        let mut satir = [_mm512_castsi512_si256(_mm512_setzero_si512()); 9];
        for (j, sj) in satir.iter_mut().enumerate().take(N + 1) {
            *sj = yuk(s + j);
        }
        for k in 0..n {
            let x0 = cift(satir[2 * k], satir[2 * k + 1]);
            let x1 = cift(satir[2 * k + 1], satir[2 * k + 2]);
            lo[k] = _mm512_unpacklo_epi8(x0, x1);
            hi[k] = _mm512_unpackhi_epi8(x0, x1);
        }
        let mut son = satir[N];
        let mut y = 0;
        // Sonraki iki yeni satırın ofseti (satır `y+S+N−1`, `y = 2`'de).
        let mut oa = (s + N + 1) * ss + 3 + x;
        loop {
            let mut a0 = _mm512_setzero_si512();
            let mut a1 = _mm512_setzero_si512();
            for k in 0..n {
                a0 = _mm512_add_epi16(a0, _mm512_maddubs_epi16(lo[k], c[k]));
                a1 = _mm512_add_epi16(a1, _mm512_maddubs_epi16(hi[k], c[k]));
            }
            sakla32x2::<PREP>(hd, w, y, x, yuv_v::<PREP, BIL>(a0), yuv_v::<PREP, BIL>(a1));
            y += 2;
            if y >= h {
                break;
            }
            let a = load_u8x32(chunk(src, oa));
            let b = load_u8x32(chunk(src, oa + ss));
            oa += 2 * ss;
            let x0 = cift(son, a);
            let x1 = cift(a, b);
            lo = kaydir(lo, n, _mm512_unpacklo_epi8(x0, x1));
            hi = kaydir(hi, n, _mm512_unpackhi_epi8(x0, x1));
            son = b;
        }
    }
}

/// İki boyutlu yol, 32 sütunluk şerit: `NH` tap yatay → `i16` ara satır, `NV`
/// tap dikey (`pmaddwd`); ara satırlar kayan pencerede.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
fn iki<const NH: usize, const NV: usize, const PREP: bool, const BIL: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    ch: &[__m512i; 4],
    cv: &[__m512i; 4],
) {
    let sv = (8 - NV) / 2;
    let n = NV / 2;
    let ix = ikili_dizin();
    for x in (0..w).step_by(32) {
        let ara = |j: usize| yuv_ara::<BIL>(h32w::<NH>(src, j * ss + x, ch, &ix));
        let mut m = [_mm512_setzero_si512(); 8];
        for (j, mj) in m.iter_mut().enumerate().take(NV - 1) {
            *mj = ara(sv + j);
        }
        for y in 0..h {
            m[NV - 1] = ara(y + sv + NV - 1);
            let r = if BIL {
                bilin_dikey::<PREP>(m[0], m[1], cv)
            } else {
                let mut lo = _mm512_setzero_si512();
                let mut hi = _mm512_setzero_si512();
                for k in 0..n {
                    let (a, b) = (m[2 * k], m[2 * k + 1]);
                    lo =
                        _mm512_add_epi32(lo, _mm512_madd_epi16(_mm512_unpacklo_epi16(a, b), cv[k]));
                    hi =
                        _mm512_add_epi32(hi, _mm512_madd_epi16(_mm512_unpackhi_epi16(a, b), cv[k]));
                }
                _mm512_packs_epi32(yuv_son::<PREP>(lo), yuv_son::<PREP>(hi))
            };
            sakla32::<PREP>(hd, w, y, x, r);
            m = kaydir(m, NV, m[NV - 1]);
        }
    }
}
