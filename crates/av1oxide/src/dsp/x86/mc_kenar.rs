//! MC `emu_edge` AVX2 çekirdeği (8 bpc) — F4-MC b (P33).
//!
//! ## Sözleşme
//!
//! `src` referans düzleminin sol üstü, `(x, y)` bloğun düzlemdeki konumu
//! (negatif olabilir). Okuma yalnız kırpılmış pencere `iclip(y, 0, ih−1) ..
//! +merkez_h` × `iclip(x, 0, iw−1) .. +merkez_w`'dir (skalerle aynı; bu
//! pencerenin dışına okunmaz); yazma yalnız hedefin `bh` satır × `bw`
//! baytıdır. Girişte O(1) denetim ([`crate::dsp::mc::kenar_sozlesmesi`]):
//! boyutlar ≥ 1, `dst_stride ≥ bw`, hedef `(bh−1)·ds + bw` bayt, pencere
//! kaynak diliminin içinde, kenar uzantıları blok içinde (skaler gövdenin
//! `debug_assert`leri). Sözleşme dışı girdi skalere devreder ve sayılır.
//!
//! ## Yöntem (dav1d `mc_avx2.asm` `emu_edge` yaklaşımı: önce merkez, sonra kenarlar)
//!
//! Merkez satırlar 32/16 baytlık parçalarla kopyalanır; `n` baytlık bir
//! kopyanın ya da doldurmanın son parçası `n − 32` (`n − 16`, …) konumundan
//! **örtüşerek** yazılır — okuma ve yazma bölgenin içinde kalır. Sol/sağ
//! uzantı kenar baytının yayınıyla, üst/alt satırlar ilk/son görünür satırın
//! kopyasıyla doldurulur.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mem::{chunk, chunk_mut, load_u8x16, store_u8x4, store_u8x8, store_u8x16, store_u8x32};
use crate::cpu::Avx2Token;
use crate::dsp::mc::{EmuEdgeFn, kenar_sozlesmesi, skaler_devri_say};
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{_mm_set1_epi8, _mm256_set_m128i, _mm256_set1_epi8};

/// Kopya sınıfı: `n` baytı bir döngüyle (32) ya da iki örtüşen parçayla
/// (16/8/4) kapsayan parça boyu; `n < 4` için 1 (bayt bayt).
const fn sinif(n: usize) -> usize {
    if n >= 32 {
        32
    } else if n >= 16 {
        16
    } else if n >= 8 {
        8
    } else if n >= 4 {
        4
    } else {
        1
    }
}

/// `d` ← `s` (eşit uzunluk `n`, `sinif(n) == K`). `K = 32`'de 32 baytlık
/// parçalar, son parça **örtüşerek** `n − 32`'den; `K` 16/8/4'te iki örtüşen
/// parça (`0` ve `n − K`); `K = 1`'de (`n` 1–3) bayt bayt. Okuma ve yazma dilimlerin
/// içinde kalır. Sınıf blok başına bir kez seçilir (satır döngüsünde dal yok).
#[target_feature(enable = "avx2")]
#[inline]
fn kopyala<const K: usize>(d: &mut [u8], s: &[u8]) {
    let n = d.len();
    match K {
        32 => {
            let y32 =
                |k: usize| _mm256_set_m128i(load_u8x16(chunk(s, k + 16)), load_u8x16(chunk(s, k)));
            let mut k = 0;
            while k + 32 < n {
                store_u8x32(chunk_mut(d, k), y32(k));
                k += 32;
            }
            store_u8x32(chunk_mut(d, n - 32), y32(n - 32));
        }
        16 => {
            let (a, b) = (load_u8x16(chunk(s, 0)), load_u8x16(chunk(s, n - 16)));
            store_u8x16(chunk_mut(d, 0), a);
            store_u8x16(chunk_mut(d, n - 16), b);
        }
        8 => {
            let (a, b) = (*chunk::<u8, 8>(s, 0), *chunk::<u8, 8>(s, n - 8));
            *chunk_mut::<u8, 8>(d, 0) = a;
            *chunk_mut::<u8, 8>(d, n - 8) = b;
        }
        4 => {
            let (a, b) = (*chunk::<u8, 4>(s, 0), *chunk::<u8, 4>(s, n - 4));
            *chunk_mut::<u8, 4>(d, 0) = a;
            *chunk_mut::<u8, 4>(d, n - 4) = b;
        }
        _ => {
            // 1–3 bayt: açık koşullu yazma (döngü kütüphane `memcpy`
            // çağrısına dönüşüyordu).
            d[0] = s[0];
            if n > 1 {
                d[1] = s[1];
            }
            if n > 2 {
                d[2] = s[2];
            }
        }
    }
}

/// `d` ← `b` (yayın); `n = d.len() ≥ 1`, son parça örtüşür.
#[target_feature(enable = "avx2")]
#[inline]
fn doldur(d: &mut [u8], b: u8) {
    let n = d.len();
    if n >= 32 {
        let v = _mm256_set1_epi8(b as i8);
        let mut k = 0;
        while k + 32 < n {
            store_u8x32(chunk_mut(d, k), v);
            k += 32;
        }
        store_u8x32(chunk_mut(d, n - 32), v);
    } else if n >= 16 {
        let v = _mm_set1_epi8(b as i8);
        store_u8x16(chunk_mut(d, 0), v);
        store_u8x16(chunk_mut(d, n - 16), v);
    } else if n >= 8 {
        let v = _mm_set1_epi8(b as i8);
        store_u8x8(chunk_mut(d, 0), v);
        store_u8x8(chunk_mut(d, n - 8), v);
    } else if n >= 4 {
        let v = _mm_set1_epi8(b as i8);
        store_u8x4(chunk_mut(d, 0), v);
        store_u8x4(chunk_mut(d, n - 4), v);
    } else {
        d[0] = b;
        if n > 1 {
            d[1] = b;
        }
        if n > 2 {
            d[2] = b;
        }
    }
}

/// Bloğun bölümleri (skaler gövdeyle aynı kırpma).
#[derive(Clone, Copy)]
struct Bolum {
    bw: usize,
    ust: usize,
    mh: usize,
    alt: usize,
    sol: usize,
    sag: usize,
    mw: usize,
    ref_row: usize,
}

/// Kenar dolgusunun yazılacağı uzunluk: `n` tek parçaya (4/8/16/32) genişler,
/// ama merkezin karşı ucunu (`sinir`) aşmaz; taşan baytları merkez kopyası ezer.
const fn genis(n: usize, sinir: usize) -> usize {
    let k = if n <= 4 {
        4
    } else if n <= 8 {
        8
    } else if n <= 16 {
        16
    } else if n <= 32 {
        32
    } else {
        n
    };
    if k < sinir { k } else { sinir }
}

/// Merkez satırlar (`M = sinif(mw)`): önce sol/sağ dolgu, sonra merkez kopyası.
#[target_feature(enable = "avx2")]
fn merkez<const M: usize>(b: Bolum, dst: &mut [u8], ds: usize, src: &[u8], ss: usize) {
    let (k_sol, k_sag) = (genis(b.sol, b.sol + b.mw), genis(b.sag, b.sag + b.mw));
    for r in 0..b.mh {
        let s0 = b.ref_row + r * ss;
        let satir = &mut dst[(b.ust + r) * ds..(b.ust + r) * ds + b.bw];
        let kaynak = &src[s0..s0 + b.mw];
        if b.sol > 0 {
            doldur(&mut satir[..k_sol], kaynak[0]);
        }
        if b.sag > 0 {
            doldur(&mut satir[b.bw - k_sag..], kaynak[b.mw - 1]);
        }
        kopyala::<M>(&mut satir[b.sol..b.sol + b.mw], kaynak);
    }
}

/// Üst/alt satırlar (`B = sinif(bw)`): ilk/son görünür satırın kopyası.
#[target_feature(enable = "avx2")]
fn dikey<const B: usize>(b: Bolum, dst: &mut [u8], ds: usize) {
    let bw = b.bw;
    // Üst: hedef satırlar kaynaktan önce.
    if b.ust > 0 {
        let (once, sonra) = dst.split_at_mut(b.ust * ds);
        let kaynak = &sonra[..bw];
        for r in 0..b.ust {
            kopyala::<B>(&mut once[r * ds..r * ds + bw], kaynak);
        }
    }
    // Alt: hedef satırlar kaynaktan sonra.
    if b.alt > 0 {
        let son = (b.ust + b.mh - 1) * ds;
        let (once, sonra) = dst.split_at_mut(son + ds);
        let kaynak = &once[son..son + bw];
        for r in 0..b.alt {
            kopyala::<B>(&mut sonra[r * ds..r * ds + bw], kaynak);
        }
    }
}

#[target_feature(enable = "avx2")]
fn emu_govde(
    bw: usize,
    bh: usize,
    iw: usize,
    ih: usize,
    x: i32,
    y: i32,
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
) {
    let sol = (-x).clamp(0, bw as i32 - 1) as usize;
    let sag = (x + bw as i32 - iw as i32).clamp(0, bw as i32 - 1) as usize;
    let ust = (-y).clamp(0, bh as i32 - 1) as usize;
    let alt = (y + bh as i32 - ih as i32).clamp(0, bh as i32 - 1) as usize;
    let b = Bolum {
        bw,
        ust,
        mh: bh - ust - alt,
        alt,
        sol,
        sag,
        mw: bw - sol - sag,
        ref_row: y.clamp(0, ih as i32 - 1) as usize * ss + x.clamp(0, iw as i32 - 1) as usize,
    };
    match sinif(b.mw) {
        32 => merkez::<32>(b, dst, ds, src, ss),
        16 => merkez::<16>(b, dst, ds, src, ss),
        8 => merkez::<8>(b, dst, ds, src, ss),
        4 => merkez::<4>(b, dst, ds, src, ss),
        _ => merkez::<1>(b, dst, ds, src, ss),
    }
    if ust + alt > 0 {
        match sinif(bw) {
            32 => dikey::<32>(b, dst, ds),
            16 => dikey::<16>(b, dst, ds),
            8 => dikey::<8>(b, dst, ds),
            4 => dikey::<4>(b, dst, ds),
            _ => dikey::<1>(b, dst, ds),
        }
    }
}

fn emu_edge_giris(
    bw: usize,
    bh: usize,
    iw: usize,
    ih: usize,
    x: i32,
    y: i32,
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    if bd != BitDepth::Eight
        || !kenar_sozlesmesi(bw, bh, iw, ih, x, y, dst.len(), ds, src.len(), ss)
    {
        skaler_devri_say();
        return match bd {
            BitDepth::Eight => {
                scalar::emu_edge::<8>(bw, bh, iw, ih, x, y, dst, ds, src, ss, bd, bdmax)
            }
            BitDepth::Ten => {
                scalar::emu_edge::<10>(bw, bh, iw, ih, x, y, dst, ds, src, ss, bd, bdmax)
            }
            BitDepth::Twelve => {
                scalar::emu_edge::<12>(bw, bh, iw, ih, x, y, dst, ds, src, ss, bd, bdmax)
            }
        };
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`emu_edge_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `kenar_sozlesmesi` kırpılmış okuma penceresini kaynak diliminde ve hedefte
    // `bh` satır × `bw` baytı kapsar; erişimler denetimlidir. (3) hizalama/takma
    // ad: `dst` tek `&mut`, `src` `&`; ödünç kuralları çakışmayı dışlar.
    unsafe { emu_govde(bw, bh, iw, ih, x, y, dst, ds, src, ss) }
}

/// `emu_edge` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn emu_edge_fn(_avx2: Avx2Token) -> EmuEdgeFn {
    emu_edge_giris
}
