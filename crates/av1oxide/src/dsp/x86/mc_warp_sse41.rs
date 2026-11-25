//! MC warp SSE4.1 çekirdekleri (8 bpc) — F4-MC d (P35): `warp8x8`,
//! `warp8x8t`.
//!
//! Sözleşme ve aritmetik SSSE3 gövdesiyle ([`super::mc_warp_ssse3`]) aynıdır
//! (yatay geçiş, aktarım ve dikey `pmaddwd`/`phaddd` ağacı oradaki
//! yardımcılardır; SSE4.1 bağlamında derlenir). Fark dikey filtrenin
//! hazırlanışıdır: filtre baytları AVX2 gövdesinin tablosundan
//! ([`super::mc_warp`]) `u64` olarak alınıp `pmovsxbw` ile `i16`'ya genişler
//! (SSSE3'teki `i16` tablosu yerine; tablo yarı boyda). VEX/AVX komutu yoktur.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mc_warp::{WARP_V, indeks};
use super::mc_warp_ssse3::{aktar, dikey, satir_yaz, yatay};
use crate::cpu::Sse41Token;
use crate::dsp::mc::{Warp8x8Fn, Warp8x8tFn, skaler_devri_say, warp_sozlesmesi};
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{__m128i, _mm_cvtepi8_epi16, _mm_cvtsi64_si128, _mm_setzero_si128};

/// Dikey filtre: bayt tablosundan `pmovsxbw`.
#[target_feature(enable = "sse4.1")]
#[inline]
fn tap(i: usize) -> __m128i {
    _mm_cvtepi8_epi16(_mm_cvtsi64_si128(WARP_V[i]))
}

/// Warp gövdesi: `T = false` → `warp8x8` (bayt), `true` → `warp8x8t` (`i16`).
#[target_feature(enable = "sse4.1")]
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Sse41Token` ile verilir
    // (`warp8x8_fn`); belirteç SSE4.1 + SSSE3 tespiti ve maske izninin
    // kanıtıdır. (2) sınır: `warp_sozlesmesi` kaynak bölgesini (15 satır,
    // `MC_KUYRUK`), hedefte 8 satır × 8 baytı ve filtre indeksinin `[0, 192]`
    // aralığını kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `dst`
    // tek `&mut`, `src` `&`.
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Sse41Token` ile verilir
    // (`warp8x8t_fn`); belirteç SSE4.1 + SSSE3 tespitinin kanıtıdır. (2) sınır:
    // `warp_sozlesmesi` kaynak bölgesini, `tmp`'de 8 satır × 8 öğeyi ve filtre
    // indeksini kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `tmp`
    // tek `&mut`, `src` `&`.
    unsafe { warp_govde::<true>(&mut [], tmp, ts, src, ss, abcd, mx, my) }
}

/// `warp8x8` SSE4.1 yuvası (yalnız belirteçle).
#[must_use]
pub fn warp8x8_fn(_sse41: Sse41Token) -> Warp8x8Fn {
    warp8x8_giris
}

/// `warp8x8t` SSE4.1 yuvası (yalnız belirteçle).
#[must_use]
pub fn warp8x8t_fn(_sse41: Sse41Token) -> Warp8x8tFn {
    warp8x8t_giris
}
