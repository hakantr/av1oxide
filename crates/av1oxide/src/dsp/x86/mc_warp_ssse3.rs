//! MC warp SSSE3 çekirdekleri (8 bpc) — F4-MC d (P35): `warp8x8`, `warp8x8t`;
//! SSE4.1 ([`super::mc_warp_sse41`]) ve 16 bpc SSSE3 ([`super::mc16_warp_ssse3`])
//! gövdelerinin ortak yardımcıları.
//!
//! ## Sözleşme
//!
//! AVX2 gövdeleriyle ([`super::mc_warp`]) aynı: girişte O(1) denetim
//! ([`crate::dsp::mc::warp_sozlesmesi`]): kaynak bölgesi `15 × 15` piksel ve
//! `MC_KUYRUK` (satır başına 16 baytlık yükleme bölgeyi en çok 1 bayt aşar),
//! hedef 8 satır × 8 piksel, filtre indeksi köşelerde `[0, 192]`. Sözleşme
//! dışı girdi skalere devreder ve sayılır.
//!
//! ## Yöntem
//!
//! Aritmetik AVX2 gövdesiyle aynıdır; yazmaç 128 bittir.
//!
//! - **Yatay** (15 satır × 8 çıktı): `pshufb` her çıktı `x` için
//!   `(s[x+k], s[x+k+4])` (k = 0…3) baytlarını dizer (bir yazmaçta iki çıktı),
//!   filtreler aynı düzende derleme zamanı tablosundan `u64` olarak;
//!   `pmaddubsw` (ikili toplam en çok 32385, doymaz), `pmaddwd(·, 1)`,
//!   `phaddd` sıralı `[x0..x3]`, `[x4..x7]`; `(Σ + 4) >> 3`, `packssdw`.
//! - **Dikey** (8 satır × 8 çıktı): ara satırlar 8×8 `i16` aktarımıyla
//!   sütunlara çevrilir; çıktı satırı `y` için her sütun `palignr` ile `y`
//!   kadar kaydırılır, `pmaddwd` (filtre `i16`, derleme zamanı tablosu
//!   [`WARP_V16`]; `pmovsxbw` SSE4.1 olduğundan kullanılmaz), `phaddd` ağacı
//!   sıralı çıktı verir (AVX2'deki `vpermd` gerekmez); `warp8x8`
//!   `(Σ + 1024) >> 11` → bayt, `warp8x8t` `(Σ + 64) >> 7`.
//!
//! SSE4.1 ve üstü komut yoktur.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mc_warp::indeks;
use super::mem::{chunk, chunk_mut, load_i16x8, load_u8x16, store_i16x8, store_u8x8};
use crate::cpu::Ssse3Token;
use crate::dsp::mc::{Warp8x8Fn, Warp8x8tFn, skaler_devri_say, warp_sozlesmesi};
use crate::dsp::mc_tables::{MC_WARP_FILTER, N_WARP_FILTERS};
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_alignr_epi8, _mm_hadd_epi32, _mm_madd_epi16, _mm_maddubs_epi16,
    _mm_packs_epi32, _mm_packus_epi16, _mm_set_epi64x, _mm_set1_epi16, _mm_set1_epi32,
    _mm_setr_epi8, _mm_setzero_si128, _mm_shuffle_epi8, _mm_srai_epi32, _mm_unpackhi_epi16,
    _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi16, _mm_unpacklo_epi32,
    _mm_unpacklo_epi64,
};

/// Yatay geçiş filtre tablosu: her satır `(f0, f4, f1, f5, f2, f6, f3, f7)`
/// bayt düzeninde, küçük sonlu `i64` (AVX2 tablosuyla aynı, kendi kopyası).
static WARP_H: [i64; N_WARP_FILTERS] = tablo_h();
/// Dikey geçiş (ve 16 bpc yatay geçiş) filtre tablosu: `(f0, …, f7)` `i16`.
pub(super) static WARP_V16: [[i16; 8]; N_WARP_FILTERS] = tablo_v16();

const fn tablo_h() -> [i64; N_WARP_FILTERS] {
    const SIRA: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
    let mut t = [0i64; N_WARP_FILTERS];
    let mut i = 0;
    while i < N_WARP_FILTERS {
        let mut v = 0u64;
        let mut k = 0;
        while k < 8 {
            v |= ((MC_WARP_FILTER[i][SIRA[k]] as u8) as u64) << (8 * k);
            k += 1;
        }
        t[i] = v as i64;
        i += 1;
    }
    t
}

const fn tablo_v16() -> [[i16; 8]; N_WARP_FILTERS] {
    let mut t = [[0i16; 8]; N_WARP_FILTERS];
    let mut i = 0;
    while i < N_WARP_FILTERS {
        let mut k = 0;
        while k < 8 {
            t[i][k] = MC_WARP_FILTER[i][k] as i16;
            k += 1;
        }
        i += 1;
    }
    t
}

/// Yatay geçiş: 15 ara satır (`i16` × 8).
#[target_feature(enable = "ssse3")]
#[inline]
pub(super) fn yatay(src: &[u8], ss: usize, abcd: &[i16; 4], mx: i32) -> [__m128i; 16] {
    // Çıktı x için (s[x+k], s[x+k+4]) baytları; bir yazmaçta x ve x+1.
    let q = [
        _mm_setr_epi8(0, 4, 1, 5, 2, 6, 3, 7, 1, 5, 2, 6, 3, 7, 4, 8),
        _mm_setr_epi8(2, 6, 3, 7, 4, 8, 5, 9, 3, 7, 4, 8, 5, 9, 6, 10),
        _mm_setr_epi8(4, 8, 5, 9, 6, 10, 7, 11, 5, 9, 6, 10, 7, 11, 8, 12),
        _mm_setr_epi8(6, 10, 7, 11, 8, 12, 9, 13, 7, 11, 8, 12, 9, 13, 10, 14),
    ];
    let bir = _mm_set1_epi16(1);
    let yuv = _mm_set1_epi32(4);
    let (a, b) = (i32::from(abcd[0]), i32::from(abcd[1]));
    let mut ara = [_mm_setzero_si128(); 16];
    let mut t0 = mx;
    for (r, satir) in ara.iter_mut().enumerate().take(15) {
        let s = load_u8x16(chunk(src, r * ss));
        let f = |x: i32| WARP_H[indeks(t0 + x * a)];
        let mut p = [_mm_setzero_si128(); 4];
        for (j, pj) in p.iter_mut().enumerate() {
            let x = 2 * j as i32;
            let fj = _mm_set_epi64x(f(x + 1), f(x));
            *pj = _mm_madd_epi16(_mm_maddubs_epi16(_mm_shuffle_epi8(s, q[j]), fj), bir);
        }
        let lo = _mm_srai_epi32::<3>(_mm_add_epi32(_mm_hadd_epi32(p[0], p[1]), yuv));
        let hi = _mm_srai_epi32::<3>(_mm_add_epi32(_mm_hadd_epi32(p[2], p[3]), yuv));
        *satir = _mm_packs_epi32(lo, hi);
        t0 += b;
    }
    ara
}

/// 8×8 `i16` aktarımı: `r[ilk..ilk+8]` satırları → sütunlar.
#[target_feature(enable = "ssse3")]
#[inline]
pub(super) fn aktar(r: &[__m128i; 16], ilk: usize) -> [__m128i; 8] {
    let a0 = _mm_unpacklo_epi16(r[ilk], r[ilk + 1]);
    let a1 = _mm_unpackhi_epi16(r[ilk], r[ilk + 1]);
    let a2 = _mm_unpacklo_epi16(r[ilk + 2], r[ilk + 3]);
    let a3 = _mm_unpackhi_epi16(r[ilk + 2], r[ilk + 3]);
    let a4 = _mm_unpacklo_epi16(r[ilk + 4], r[ilk + 5]);
    let a5 = _mm_unpackhi_epi16(r[ilk + 4], r[ilk + 5]);
    let a6 = _mm_unpacklo_epi16(r[ilk + 6], r[ilk + 7]);
    let a7 = _mm_unpackhi_epi16(r[ilk + 6], r[ilk + 7]);
    let b0 = _mm_unpacklo_epi32(a0, a2);
    let b1 = _mm_unpackhi_epi32(a0, a2);
    let b2 = _mm_unpacklo_epi32(a1, a3);
    let b3 = _mm_unpackhi_epi32(a1, a3);
    let b4 = _mm_unpacklo_epi32(a4, a6);
    let b5 = _mm_unpackhi_epi32(a4, a6);
    let b6 = _mm_unpacklo_epi32(a5, a7);
    let b7 = _mm_unpackhi_epi32(a5, a7);
    [
        _mm_unpacklo_epi64(b0, b4),
        _mm_unpackhi_epi64(b0, b4),
        _mm_unpacklo_epi64(b1, b5),
        _mm_unpackhi_epi64(b1, b5),
        _mm_unpacklo_epi64(b2, b6),
        _mm_unpackhi_epi64(b2, b6),
        _mm_unpacklo_epi64(b3, b7),
        _mm_unpackhi_epi64(b3, b7),
    ]
}

/// `[lo | hi]` sütununu `y` satır kaydırır (`palignr`).
#[target_feature(enable = "ssse3")]
#[inline]
fn kaydir(hi: __m128i, lo: __m128i, y: usize) -> __m128i {
    match y {
        0 => lo,
        1 => _mm_alignr_epi8::<2>(hi, lo),
        2 => _mm_alignr_epi8::<4>(hi, lo),
        3 => _mm_alignr_epi8::<6>(hi, lo),
        4 => _mm_alignr_epi8::<8>(hi, lo),
        5 => _mm_alignr_epi8::<10>(hi, lo),
        6 => _mm_alignr_epi8::<12>(hi, lo),
        _ => _mm_alignr_epi8::<14>(hi, lo),
    }
}

/// Dikey geçişin çıktı satırı `y`: sütun `x`'in `i16` filtresi `k[x]`;
/// sıralı `i32` toplamlar `([x0..x3], [x4..x7])`.
#[target_feature(enable = "ssse3")]
#[inline]
pub(super) fn dikey(
    ust: &[__m128i; 8],
    alt: &[__m128i; 8],
    y: usize,
    k: &[__m128i; 8],
) -> (__m128i, __m128i) {
    let m: [__m128i; 8] = core::array::from_fn(|x| _mm_madd_epi16(kaydir(alt[x], ust[x], y), k[x]));
    (
        _mm_hadd_epi32(_mm_hadd_epi32(m[0], m[1]), _mm_hadd_epi32(m[2], m[3])),
        _mm_hadd_epi32(_mm_hadd_epi32(m[4], m[5]), _mm_hadd_epi32(m[6], m[7])),
    )
}

/// 8 bpc dikey satır sonu: `T = false` `(Σ + 1024) >> 11` → bayt, `true`
/// `(Σ + 64) >> 7` → `i16`.
#[target_feature(enable = "ssse3")]
#[inline]
pub(super) fn satir_yaz<const T: bool>(
    dst: &mut [u8],
    tmp: &mut [i16],
    o: usize,
    (lo, hi): (__m128i, __m128i),
) {
    let yuv = _mm_set1_epi32(if T { 64 } else { 1024 });
    let (lo, hi) = (_mm_add_epi32(lo, yuv), _mm_add_epi32(hi, yuv));
    if T {
        let p = _mm_packs_epi32(_mm_srai_epi32::<7>(lo), _mm_srai_epi32::<7>(hi));
        store_i16x8(chunk_mut(tmp, o), p);
    } else {
        let p = _mm_packs_epi32(_mm_srai_epi32::<11>(lo), _mm_srai_epi32::<11>(hi));
        store_u8x8(chunk_mut(dst, o), _mm_packus_epi16(p, p));
    }
}

/// Dikey filtre `i16` (tablodan yükleme).
#[target_feature(enable = "ssse3")]
#[inline]
fn tap(i: usize) -> __m128i {
    load_i16x8(&WARP_V16[i])
}

/// Warp gövdesi: `T = false` → `warp8x8` (bayt), `true` → `warp8x8t` (`i16`).
#[target_feature(enable = "ssse3")]
fn warp_govde<const T: bool>(
    dst: &mut [u8],
    tmp: &mut [i16],
    ds: usize,
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
) {
    let ara = yatay(src, ss, abcd, mx);
    let ust = aktar(&ara, 0);
    let alt = aktar(&ara, 8);
    let (c, d) = (i32::from(abcd[2]), i32::from(abcd[3]));
    let mut t0 = my;
    for y in 0..8 {
        let mut k = [_mm_setzero_si128(); 8];
        for (x, kx) in k.iter_mut().enumerate() {
            *kx = tap(indeks(t0 + x as i32 * c));
        }
        satir_yaz::<T>(dst, tmp, y * ds, dikey(&ust, &alt, y, &k));
        t0 += d;
    }
}

macro_rules! skaler {
    ($bd:expr, $f:ident ( $($a:expr),* )) => {
        match $bd {
            BitDepth::Eight => scalar::$f::<8>($($a),*),
            BitDepth::Ten => scalar::$f::<10>($($a),*),
            BitDepth::Twelve => scalar::$f::<12>($($a),*),
        }
    };
}

fn warp8x8_giris(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bd: BitDepth,
    bdmax: u16,
) {
    if bd != BitDepth::Eight || !warp_sozlesmesi(src.len(), ss, abcd, mx, my, dst.len(), ds) {
        skaler_devri_say();
        return skaler!(bd, warp8x8(dst, ds, src, ss, abcd, mx, my, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`warp8x8_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `warp_sozlesmesi` kaynak bölgesini (15 satır, `MC_KUYRUK`), hedefte 8 satır
    // × 8 baytı ve filtre indeksinin `[0, 192]` aralığını kapsar; erişimler
    // denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
    unsafe { warp_govde::<false>(dst, &mut [], ds, src, ss, abcd, mx, my) }
}

fn warp8x8t_giris(
    tmp: &mut [i16],
    ts: usize,
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bd: BitDepth,
    bdmax: u16,
) {
    if bd != BitDepth::Eight || !warp_sozlesmesi(src.len(), ss, abcd, mx, my, tmp.len(), ts) {
        skaler_devri_say();
        return skaler!(bd, warp8x8t(tmp, ts, src, ss, abcd, mx, my, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`warp8x8t_fn`); belirteç SSSE3 tespitinin kanıtıdır. (2) sınır:
    // `warp_sozlesmesi` kaynak bölgesini, `tmp`'de 8 satır × 8 öğeyi ve filtre
    // indeksini kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `tmp`
    // tek `&mut`, `src` `&`.
    unsafe { warp_govde::<true>(&mut [], tmp, ts, src, ss, abcd, mx, my) }
}

/// `warp8x8` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn warp8x8_fn(_ssse3: Ssse3Token) -> Warp8x8Fn {
    warp8x8_giris
}

/// `warp8x8t` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn warp8x8t_fn(_ssse3: Ssse3Token) -> Warp8x8tFn {
    warp8x8t_giris
}
