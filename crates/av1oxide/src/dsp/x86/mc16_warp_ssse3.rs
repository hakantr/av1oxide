//! MC warp SSSE3 çekirdekleri (16 bpc, 10/12 bit) — F4-MC d (P35):
//! `warp8x8`, `warp8x8t`.
//!
//! ## Sözleşme
//!
//! AVX2 gövdeleriyle ([`super::mc16_warp`]) aynı: girişte O(1) denetim
//! ([`crate::dsp::mc::warp16_sozlesmesi`]): bit derinliği 10/12 ve tutarlı
//! `bitdepth_max`, kaynak bölgesi 15 satır × 30 bayt ve `MC_KUYRUK`, hedef 8
//! satır × 8 piksel, filtre indeksi köşelerde `[0, 192]`. Sözleşme dışı girdi
//! skalere devreder ve sayılır. Yatay yüklemeler okuma bölgesinin içinde biter
//! (son yükleme sütun 7…14).
//!
//! ## Yöntem
//!
//! Aritmetik AVX2 gövdesiyle aynıdır; yazmaç 128 bittir.
//!
//! - **Yatay** (15 satır × 8 çıktı): çıktı `x` için `s[x..x+8]` (8 piksel) ve
//!   filtre `f_x` (`i16` tablo, [`super::mc_warp_ssse3::WARP_V16`]) `pmaddwd`;
//!   `phaddd` ağacı sıralı `[x0..x3]`, `[x4..x7]`; `(Σ + r) >> (7−ib)`
//!   (kaydırma yazmaçta), `packssdw` (ara değer `i16`'ya sığar).
//! - **Dikey** (8 satır × 8 çıktı): 8 bpc SSSE3 gövdesinin yolu (aktarım,
//!   `palignr`, `pmaddwd`, `phaddd`). `warp8x8` `(Σ + r) >> (7+ib)`, kırpma
//!   `packssdw` + `pmaxsw(0)` + `pminsw(bitdepth_max)` (`packusdw`/`pminuw`
//!   SSE4.1 yerine; `i32` sonuç `> 32767` iken iki yol da `bitdepth_max`,
//!   `< 0` iken 0 verir); `warp8x8t` `(Σ + 64) >> 7 − 8192` (`PREP_BIAS`),
//!   `packssdw`.
//!
//! SSE4.1 ve üstü komut yoktur.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mc_warp::indeks;
use super::mc_warp_ssse3::{WARP_V16, aktar, dikey};
use super::mem::{chunk, chunk_mut, load_i16x8, load_u8x16, store_i16x8, store_u8x16};
use crate::cpu::Ssse3Token;
use crate::dsp::mc::{Warp8x8Fn, Warp8x8tFn, skaler_devri_say, warp16_sozlesmesi};
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_cvtsi32_si128, _mm_hadd_epi32, _mm_madd_epi16, _mm_max_epi16,
    _mm_min_epi16, _mm_packs_epi32, _mm_set1_epi16, _mm_set1_epi32, _mm_setzero_si128,
    _mm_sra_epi32, _mm_sub_epi32,
};

/// `prep` ara değerinden çıkarılan sabit (dav1d `PREP_BIAS`, 16 bpc).
const PREP_BIAS: i32 = 8192;

/// Yatay geçiş: 15 ara satır (`i16` × 8); `(Σ + yuv) >> sh`.
#[target_feature(enable = "ssse3")]
#[inline]
fn yatay16(
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    yuv: __m128i,
    sh: __m128i,
) -> [__m128i; 16] {
    let (a, b) = (i32::from(abcd[0]), i32::from(abcd[1]));
    let mut ara = [_mm_setzero_si128(); 16];
    let mut t0 = mx;
    for (r, satir) in ara.iter_mut().enumerate().take(15) {
        let o = r * ss;
        let mut m = [_mm_setzero_si128(); 8];
        for (x, mp) in m.iter_mut().enumerate() {
            let p = load_u8x16(chunk(src, o + 2 * x));
            let f = load_i16x8(&WARP_V16[indeks(t0 + x as i32 * a)]);
            *mp = _mm_madd_epi16(p, f);
        }
        let lo = _mm_hadd_epi32(_mm_hadd_epi32(m[0], m[1]), _mm_hadd_epi32(m[2], m[3]));
        let hi = _mm_hadd_epi32(_mm_hadd_epi32(m[4], m[5]), _mm_hadd_epi32(m[6], m[7]));
        let lo = _mm_sra_epi32(_mm_add_epi32(lo, yuv), sh);
        let hi = _mm_sra_epi32(_mm_add_epi32(hi, yuv), sh);
        *satir = _mm_packs_epi32(lo, hi);
        t0 += b;
    }
    ara
}

/// Warp gövdesi: `T = false` → `warp8x8` (`u16` piksel), `true` → `warp8x8t`
/// (`i16`).
#[target_feature(enable = "ssse3")]
fn warp16_govde<const T: bool>(
    dst: &mut [u8],
    tmp: &mut [i16],
    ds: usize,
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bitdepth_max: u16,
) {
    let ib = if bitdepth_max == 4095 { 2 } else { 4 };
    let sh_h = 7 - ib;
    let ara = yatay16(
        src,
        ss,
        abcd,
        mx,
        _mm_set1_epi32(1 << (sh_h - 1)),
        _mm_cvtsi32_si128(sh_h),
    );
    let ust = aktar(&ara, 0);
    let alt = aktar(&ara, 8);
    let (c, d) = (i32::from(abcd[2]), i32::from(abcd[3]));
    let sh_v = if T { 7 } else { 7 + ib };
    let yuv = _mm_set1_epi32(1 << (sh_v - 1));
    let sh_v = _mm_cvtsi32_si128(sh_v);
    let bias = _mm_set1_epi32(PREP_BIAS);
    let max = _mm_set1_epi16(bitdepth_max as i16);
    let mut t0 = my;
    for y in 0..8 {
        let mut k = [_mm_setzero_si128(); 8];
        for (x, kx) in k.iter_mut().enumerate() {
            *kx = load_i16x8(&WARP_V16[indeks(t0 + x as i32 * c)]);
        }
        let (lo, hi) = dikey(&ust, &alt, y, &k);
        let lo = _mm_sra_epi32(_mm_add_epi32(lo, yuv), sh_v);
        let hi = _mm_sra_epi32(_mm_add_epi32(hi, yuv), sh_v);
        if T {
            let p = _mm_packs_epi32(_mm_sub_epi32(lo, bias), _mm_sub_epi32(hi, bias));
            store_i16x8(chunk_mut(tmp, y * ds), p);
        } else {
            let p = _mm_max_epi16(_mm_packs_epi32(lo, hi), _mm_setzero_si128());
            store_u8x16(chunk_mut(dst, y * ds), _mm_min_epi16(p, max));
        }
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

fn warp8x8_16_giris(
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
    if !warp16_sozlesmesi(src.len(), ss, abcd, mx, my, dst.len(), ds, 2, bd, bdmax) {
        skaler_devri_say();
        return skaler!(bd, warp8x8(dst, ds, src, ss, abcd, mx, my, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`warp8x8_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `warp16_sozlesmesi` kaynak bölgesini (15 satır × 30 bayt, `MC_KUYRUK`),
    // hedefte 8 satır × 16 baytı ve filtre indeksinin `[0, 192]` aralığını kapsar;
    // erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
    unsafe { warp16_govde::<false>(dst, &mut [], ds, src, ss, abcd, mx, my, bdmax) }
}

fn warp8x8t_16_giris(
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
    if !warp16_sozlesmesi(src.len(), ss, abcd, mx, my, tmp.len(), ts, 1, bd, bdmax) {
        skaler_devri_say();
        return skaler!(bd, warp8x8t(tmp, ts, src, ss, abcd, mx, my, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`warp8x8t_fn`); belirteç SSSE3 tespitinin kanıtıdır. (2) sınır:
    // `warp16_sozlesmesi` kaynak bölgesini, `tmp`'de 8 satır × 8 öğeyi ve filtre
    // indeksini kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `tmp`
    // tek `&mut`, `src` `&`.
    unsafe { warp16_govde::<true>(&mut [], tmp, ts, src, ss, abcd, mx, my, bdmax) }
}

/// 16 bpc `warp8x8` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn warp8x8_fn(_ssse3: Ssse3Token) -> Warp8x8Fn {
    warp8x8_16_giris
}

/// 16 bpc `warp8x8t` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn warp8x8t_fn(_ssse3: Ssse3Token) -> Warp8x8tFn {
    warp8x8t_16_giris
}
