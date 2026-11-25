//! MC 16 bpc (10/12 bit) AVX512ICL `put`/`prep` çekirdekleri — F4-MC e (P36):
//! `mc[0..9]`, `mct[0..9]`.
//!
//! ## Sözleşme
//!
//! AVX2 gövdesinin sözleşmesi ([`super::mc16`]): girişte O(1)
//! [`crate::dsp::mc::simd16_sozlesmesi`]; sözleşme dışı girdi skalere devreder
//! ve sayılır. Aynı giriş, aynı denetimden sonra `w < 32` blokları ve `put`
//! kopyasını AVX2 gövdesine verir. `w ≥ 32` yolunun 32 piksellik (64 bayt) yüklemeleri okuma
//! bölgesinin içinde biter (son şeritte yatay pencere sütun `w+6`'da biter);
//! yazma yalnız `h` satır × `w` piksel.
//!
//! ## Yöntem
//!
//! AVX2 gövdesinin aritmetiği (8-tap ikilileri `pmaddwd`, çağrı başına
//! yuvarlama kaydırmaları, `packusdw` + `pminuw`, `− PREP_BIAS` + `packssdw`)
//! 512 bitte, 32 piksellik şeritlerle. Yatay toplamların `punpck{l,h}dq`
//! düzeni ve `packusdw` 128 bitlik lane içinde sırayı korur (lane `L` çıktı
//! `8L..8L+7`); karıştırma gerekmez. İki boyutlu yol AVX2 gövdesi gibi 8
//! satırlık `i16` ara halka kullanır (`G` sütun).

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mc::{BILIN, filtre_satiri};
use super::mc16::{self as avx2, Hedef};
use super::mem::{
    chunk, chunk_mut, load_i16x32, load_u8x32, load_u8x64, store_i16x32, store_u8x32, store_u8x64,
};
use crate::cpu::Avx512IclToken;
use crate::dsp::mc::{Filter2d, McPrepFn, McPutFn};
use core::arch::x86_64::{
    __m128i, __m512i, _mm_cvtsi32_si128, _mm512_add_epi32, _mm512_madd_epi16, _mm512_min_epu16,
    _mm512_packs_epi32, _mm512_packus_epi32, _mm512_set1_epi16, _mm512_set1_epi32,
    _mm512_setzero_si512, _mm512_sll_epi16, _mm512_sra_epi32, _mm512_sub_epi16, _mm512_sub_epi32,
    _mm512_unpackhi_epi16, _mm512_unpackhi_epi32, _mm512_unpacklo_epi16, _mm512_unpacklo_epi32,
};

/// `prep` ara değerinden çıkarılan sabit (dav1d `PREP_BIAS`, 16 bpc).
const PREP_BIAS: i32 = 8192;

/// Bir geçişin yuvarlaması: `(Σ + r) >> sh` (kaydırma çağrı başına sabit).
#[derive(Clone, Copy)]
struct Yuv {
    r: i32,
    sh: i32,
}

impl Yuv {
    const fn yeni(sh: i32) -> Self {
        Yuv {
            r: (1 << sh) >> 1,
            sh,
        }
    }
}

/// Tap ikilisi katsayıları `(f[S+2i], f[S+2i+1])` → `pmaddwd` sözcükleri.
fn ikililer<const P: usize, const S: usize>(f: &[i8; 8]) -> [i32; 4] {
    let mut k = [0i32; 4];
    for (i, ki) in k.iter_mut().enumerate().take(P) {
        let a = i32::from(f[S + 2 * i]);
        let b = i32::from(f[S + 2 * i + 1]);
        *ki = (a & 0xffff) | (b << 16);
    }
    k
}

/// Bilinear katsayı ikilisi `(16−m, m)`.
const fn bilin_ikili(m: u8) -> [i32; 4] {
    let m = m as i32;
    [((16 - m) & 0xffff) | (m << 16), 0, 0, 0]
}

// ---------------------------------------------------------------------------
// Giriş
// ---------------------------------------------------------------------------

/// Yol seçimi (AVX2 gövdesiyle aynı yuvarlama türetimi).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline(never)]
pub(super) fn govde<const FH: usize, const FV: usize, const PREP: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bitdepth_max: u16,
) {
    let max = i32::from(bitdepth_max);
    let ib = if max == 4095 { 2 } else { 4 };
    let c = Cikti {
        bias: _mm512_set1_epi32(PREP_BIAS),
        max: _mm512_set1_epi16(max as i16),
    };
    let bil = FH == BILIN;
    let sh_ara = if bil { 4 - ib } else { 6 - ib };
    let tam = if bil { 4 } else { 6 };
    match (mx != 0, my != 0) {
        (false, false) => kopya::<PREP>(hd, src, ss, w, h, ib),
        (true, false) => {
            let yuv = if PREP {
                Yuv::yeni(sh_ara)
            } else {
                let ara = Yuv::yeni(sh_ara);
                Yuv {
                    r: ara.r + (((1 << ib) >> 1) << sh_ara),
                    sh: tam,
                }
            };
            if bil {
                yol_h::<PREP, 1, 3>(hd, src, ss, w, h, bilin_ikili(mx), yuv, c);
            } else {
                let f = filtre_satiri(FH, w, mx);
                yol_h::<PREP, 4, 0>(hd, src, ss, w, h, ikililer::<4, 0>(f), yuv, c);
            }
        }
        (false, true) => {
            let yuv = if PREP {
                Yuv::yeni(sh_ara)
            } else {
                Yuv::yeni(tam)
            };
            if bil {
                yol_v::<PREP, 1, 3>(hd, src, ss, w, h, bilin_ikili(my), yuv, c);
            } else {
                let f = filtre_satiri(FV, h, my);
                yol_v::<PREP, 4, 0>(hd, src, ss, w, h, ikililer::<4, 0>(f), yuv, c);
            }
        }
        (true, true) => {
            let ara = Yuv::yeni(sh_ara);
            let son = if PREP {
                Yuv::yeni(tam)
            } else {
                Yuv::yeni(tam + ib)
            };
            let (kh, kv) = if bil {
                (bilin_ikili(mx), bilin_ikili(my))
            } else {
                (
                    ikililer::<4, 0>(filtre_satiri(FH, w, mx)),
                    ikililer::<4, 0>(filtre_satiri(FV, h, my)),
                )
            };
            macro_rules! hv {
                ($p:literal, $s:literal, $g:literal) => {
                    hv::<PREP, $p, $s, $g>(hd, src, ss, w, h, kh, kv, ara, son, c)
                };
            }
            match (bil, w) {
                (true, 32) => hv!(1, 3, 32),
                (true, 64) => hv!(1, 3, 64),
                (true, _) => hv!(1, 3, 128),
                (false, 32) => hv!(4, 0, 32),
                (false, 64) => hv!(4, 0, 64),
                (false, _) => hv!(4, 0, 128),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ortak yardımcılar
// ---------------------------------------------------------------------------

/// Çıktı değerleri: `bias` (`prep`, `i32`), `max` (`put` kırpma, `u16`).
#[derive(Clone, Copy)]
struct Cikti {
    bias: __m512i,
    max: __m512i,
}

/// 32 çıktının yatay tap toplamları (`i32`; lane `L`: `lo` çıktı `8L..8L+3`,
/// `hi` `8L+4..8L+7`); `o` dolgu satırı + sütun `x`'in bayt ofseti.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn yatay32<const P: usize, const S: usize>(
    src: &[u8],
    o: usize,
    k: &[__m512i; 4],
) -> (__m512i, __m512i) {
    let mut e = _mm512_setzero_si512();
    let mut t = _mm512_setzero_si512();
    for (i, ki) in k.iter().enumerate().take(P) {
        let a = load_u8x64(chunk(src, o + 2 * (S + 2 * i)));
        let b = load_u8x64(chunk(src, o + 2 * (S + 2 * i + 1)));
        e = _mm512_add_epi32(e, _mm512_madd_epi16(a, *ki));
        t = _mm512_add_epi32(t, _mm512_madd_epi16(b, *ki));
    }
    (_mm512_unpacklo_epi32(e, t), _mm512_unpackhi_epi32(e, t))
}

/// `(v + r) >> sh`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn yuv512(v: __m512i, r: __m512i, sh: __m128i) -> __m512i {
    _mm512_sra_epi32(_mm512_add_epi32(v, r), sh)
}

/// 32 çıktıyı yazar: `put` `packusdw` + `pminuw(max)`, `prep` `− bias` + `packssdw`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn yaz32<const PREP: bool>(
    hd: &mut Hedef<'_>,
    w: usize,
    y: usize,
    x: usize,
    lo: __m512i,
    hi: __m512i,
    c: Cikti,
) {
    if PREP {
        let v = _mm512_packs_epi32(_mm512_sub_epi32(lo, c.bias), _mm512_sub_epi32(hi, c.bias));
        store_i16x32(chunk_mut(hd.tmp, y * w + x), v);
    } else {
        let v = _mm512_min_epu16(_mm512_packus_epi32(lo, hi), c.max);
        store_u8x64(chunk_mut(hd.dst, y * hd.ds + 2 * x), v);
    }
}

#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn kat512(k: [i32; 4]) -> [__m512i; 4] {
    [
        _mm512_set1_epi32(k[0]),
        _mm512_set1_epi32(k[1]),
        _mm512_set1_epi32(k[2]),
        _mm512_set1_epi32(k[3]),
    ]
}

// ---------------------------------------------------------------------------
// Yollar (`w ≥ 32`)
// ---------------------------------------------------------------------------

/// Kopya: `put` satır başına 32 baytlık parçalar, `prep` `(p << ib) − 8192` (64 bayt).
#[target_feature(enable = "avx512f,avx512bw")]
fn kopya<const PREP: bool>(hd: &mut Hedef<'_>, src: &[u8], ss: usize, w: usize, h: usize, ib: i32) {
    let ibv = _mm_cvtsi32_si128(ib);
    let bias = _mm512_set1_epi16(PREP_BIAS as i16);
    for y in 0..h {
        let s0 = (y + 3) * ss + 6;
        let mut x = 0;
        if PREP {
            let o = y * w;
            while x < w {
                let v = _mm512_sll_epi16(load_u8x64(chunk(src, s0 + 2 * x)), ibv);
                store_i16x32(chunk_mut(hd.tmp, o + x), _mm512_sub_epi16(v, bias));
                x += 32;
            }
        } else {
            // 32 baytlık parçalar (16 piksel): bellek bağlı kopyada 64 baytlık
            // saklamalar satır başına daha çok önbellek satırı sınırı aşıyordu (P36
            // hücre ölçümü: `w` 64/128'de AVX2'den %10 yavaş).
            let d0 = y * hd.ds;
            while x < w {
                store_u8x32(
                    chunk_mut(hd.dst, d0 + 2 * x),
                    load_u8x32(chunk(src, s0 + 2 * x)),
                );
                x += 16;
            }
        }
    }
}

/// Yalnız yatay: dolgu satırı `y + 3`.
#[target_feature(enable = "avx512f,avx512bw")]
fn yol_h<const PREP: bool, const P: usize, const S: usize>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    k: [i32; 4],
    yuv: Yuv,
    c: Cikti,
) {
    let sh = _mm_cvtsi32_si128(yuv.sh);
    let k = kat512(k);
    let r = _mm512_set1_epi32(yuv.r);
    for y in 0..h {
        let o = (y + 3) * ss;
        let mut x = 0;
        while x < w {
            let (lo, hi) = yatay32::<P, S>(src, o + 2 * x, &k);
            yaz32::<PREP>(hd, w, y, x, yuv512(lo, r, sh), yuv512(hi, r, sh), c);
            x += 32;
        }
    }
}

/// Yalnız dikey: dolgu sütunu `x + 3`, satırlar `y + S + 2i`, `+1`.
#[target_feature(enable = "avx512f,avx512bw")]
fn yol_v<const PREP: bool, const P: usize, const S: usize>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    k: [i32; 4],
    yuv: Yuv,
    c: Cikti,
) {
    let sh = _mm_cvtsi32_si128(yuv.sh);
    let k = kat512(k);
    let r = _mm512_set1_epi32(yuv.r);
    for y in 0..h {
        let mut x = 0;
        while x < w {
            let o = (y + S) * ss + 2 * (x + 3);
            let mut lo = r;
            let mut hi = r;
            for (i, ki) in k.iter().enumerate().take(P) {
                let a = load_u8x64(chunk(src, o + 2 * i * ss));
                let b = load_u8x64(chunk(src, o + (2 * i + 1) * ss));
                lo = _mm512_add_epi32(lo, _mm512_madd_epi16(_mm512_unpacklo_epi16(a, b), *ki));
                hi = _mm512_add_epi32(hi, _mm512_madd_epi16(_mm512_unpackhi_epi16(a, b), *ki));
            }
            let (lo, hi) = (_mm512_sra_epi32(lo, sh), _mm512_sra_epi32(hi, sh));
            yaz32::<PREP>(hd, w, y, x, lo, hi, c);
            x += 32;
        }
    }
}

/// İki boyutlu: 8 satırlık ara halka (`G` sütun), dolgu satırı `j` halka
/// yuvası `j & 7`; çıktı satırı `y` ara satırları `y + S + 2i`, `+1`.
#[target_feature(enable = "avx512f,avx512bw")]
fn hv<const PREP: bool, const P: usize, const S: usize, const G: usize>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    kh: [i32; 4],
    kv: [i32; 4],
    ara: Yuv,
    son: Yuv,
    c: Cikti,
) {
    let (kh, kv) = (kat512(kh), kat512(kv));
    let (ra, sa) = (_mm512_set1_epi32(ara.r), _mm_cvtsi32_si128(ara.sh));
    let (rs, ssh) = (_mm512_set1_epi32(son.r), _mm_cvtsi32_si128(son.sh));
    let mut halka = [[0i16; G]; 8];
    let mut j = S;
    for y in 0..h {
        while j < y + S + 2 * P {
            let o = j * ss;
            let satir = &mut halka[j & 7];
            let mut x = 0;
            while x < w {
                let (lo, hi) = yatay32::<P, S>(src, o + 2 * x, &kh);
                let v = _mm512_packs_epi32(yuv512(lo, ra, sa), yuv512(hi, ra, sa));
                store_i16x32(chunk_mut(satir, x), v);
                x += 32;
            }
            j += 1;
        }
        let mut x = 0;
        while x < w {
            let mut lo = rs;
            let mut hi = rs;
            for (i, ki) in kv.iter().enumerate().take(P) {
                let a = load_i16x32(chunk(&halka[(y + S + 2 * i) & 7], x));
                let b = load_i16x32(chunk(&halka[(y + S + 2 * i + 1) & 7], x));
                lo = _mm512_add_epi32(lo, _mm512_madd_epi16(_mm512_unpacklo_epi16(a, b), *ki));
                hi = _mm512_add_epi32(hi, _mm512_madd_epi16(_mm512_unpackhi_epi16(a, b), *ki));
            }
            let (lo, hi) = (_mm512_sra_epi32(lo, ssh), _mm512_sra_epi32(hi, ssh));
            yaz32::<PREP>(hd, w, y, x, lo, hi, c);
            x += 32;
        }
    }
}

// ---------------------------------------------------------------------------
// Yuvalar
// ---------------------------------------------------------------------------

/// 16 bpc `mc[f]` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn put16_fn(_avx512: Avx512IclToken, f: Filter2d) -> McPutFn {
    use Filter2d as F;
    match f {
        F::Regular => avx2::put16_giris_icl::<0, 0>,
        F::RegularSmooth => avx2::put16_giris_icl::<0, 1>,
        F::RegularSharp => avx2::put16_giris_icl::<0, 2>,
        F::SharpRegular => avx2::put16_giris_icl::<2, 0>,
        F::SharpSmooth => avx2::put16_giris_icl::<2, 1>,
        F::Sharp => avx2::put16_giris_icl::<2, 2>,
        F::SmoothRegular => avx2::put16_giris_icl::<1, 0>,
        F::Smooth => avx2::put16_giris_icl::<1, 1>,
        F::SmoothSharp => avx2::put16_giris_icl::<1, 2>,
        F::Bilinear => avx2::put16_giris_icl::<BILIN, BILIN>,
    }
}

/// 16 bpc `mct[f]` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn prep16_fn(_avx512: Avx512IclToken, f: Filter2d) -> McPrepFn {
    use Filter2d as F;
    match f {
        F::Regular => avx2::prep16_giris_icl::<0, 0>,
        F::RegularSmooth => avx2::prep16_giris_icl::<0, 1>,
        F::RegularSharp => avx2::prep16_giris_icl::<0, 2>,
        F::SharpRegular => avx2::prep16_giris_icl::<2, 0>,
        F::SharpSmooth => avx2::prep16_giris_icl::<2, 1>,
        F::Sharp => avx2::prep16_giris_icl::<2, 2>,
        F::SmoothRegular => avx2::prep16_giris_icl::<1, 0>,
        F::Smooth => avx2::prep16_giris_icl::<1, 1>,
        F::SmoothSharp => avx2::prep16_giris_icl::<1, 2>,
        F::Bilinear => avx2::prep16_giris_icl::<BILIN, BILIN>,
    }
}
