//! MC `emu_edge` AVX2 çekirdeği (16 bpc, 10/12 bit) — F4-MC c (P34).
//!
//! ## Sözleşme
//!
//! `src` referans düzleminin sol üstü, `(x, y)` bloğun düzlemdeki konumu
//! (negatif olabilir); pikseller `u16`, satır adımları bayt. Okuma yalnız
//! kırpılmış pencere `iclip(y, 0, ih−1) .. +merkez_h` × `iclip(x, 0, iw−1) ..
//! +merkez_w` pikseldir (skalerle aynı; bu pencerenin dışına okunmaz); yazma
//! yalnız hedefin `bh` satır × `2·bw` baytıdır. Girişte O(1) denetim
//! ([`crate::dsp::mc::kenar16_sozlesmesi`]): bit derinliği 10/12 ve
//! `bitdepth_max` tutarlı, boyutlar ≥ 1, `dst_stride ≥ 2·bw`, hedef
//! `(bh−1)·ds + 2·bw` bayt, pencere kaynak diliminin içinde, kenar uzantıları
//! blok içinde. Sözleşme dışı girdi skalere devreder ve sayılır.
//!
//! ## Yöntem
//!
//! 8 bpc gövdesinin ([`super::mc_kenar`]) yolu bayt biriminde (yardımcılar kendi kopyası): merkez satırlar
//! 32/16 baytlık parçalarla, son parça örtüşerek kopyalanır; üst/alt satırlar
//! ilk/son görünür satırın kopyası. Yalnız sol/sağ uzantı kenar pikselinin
//! `u16` yayınıdır ([`doldur16`]); bütün parça uzunlukları ve örtüşen son
//! parça konumları çift olduğundan desen korunur.
//!
//! `unsafe` yalnız belirteçli girişteki `#[target_feature]` çağrısıdır; dosya
//! `audit::UNSAFE_EXEMPTIONS` defterinde kayıtlıdır. Yükleme/saklama
//! [`super::mem`] yardımcılarıyla yapılır.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mem::{chunk, chunk_mut, load_u8x16, store_u8x4, store_u8x8, store_u8x16, store_u8x32};
use crate::cpu::Avx2Token;
use crate::dsp::mc::{EmuEdgeFn, kenar16_sozlesmesi, skaler_devri_say};
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{_mm_set1_epi16, _mm256_set_m128i, _mm256_set1_epi16};

// 8 bpc `mc_kenar` gövdesinin bayt tabanlı yardımcıları (aynı kod). P34: ortak
// kullanım 8 bpc `emu_govde`'de satır içi alma kararını değiştirip dar hücrelerde
// gerilemeye yol açtı (`dikey` çağrıya dönüştü); 16 bpc gövdesi kendi kopyasını kullanır.

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

/// Bloğun bölümleri (skaler gövdeyle aynı kırpma); yatay alanlar **bayt**.
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

/// `d` ← `v` yayını (`u16`, yerel bayt sırası); `n = d.len()` çift ve ≥ 2,
/// son parça örtüşür (konumu çift).
#[target_feature(enable = "avx2")]
#[inline]
fn doldur16(d: &mut [u8], v: u16) {
    let n = d.len();
    if n >= 32 {
        let y = _mm256_set1_epi16(v as i16);
        let mut k = 0;
        while k + 32 < n {
            store_u8x32(chunk_mut(d, k), y);
            k += 32;
        }
        store_u8x32(chunk_mut(d, n - 32), y);
    } else if n >= 16 {
        let x = _mm_set1_epi16(v as i16);
        store_u8x16(chunk_mut(d, 0), x);
        store_u8x16(chunk_mut(d, n - 16), x);
    } else if n >= 8 {
        let x = _mm_set1_epi16(v as i16);
        store_u8x8(chunk_mut(d, 0), x);
        store_u8x8(chunk_mut(d, n - 8), x);
    } else if n >= 4 {
        let x = _mm_set1_epi16(v as i16);
        store_u8x4(chunk_mut(d, 0), x);
        store_u8x4(chunk_mut(d, n - 4), x);
    } else {
        *chunk_mut::<u8, 2>(d, 0) = v.to_ne_bytes();
    }
}

/// Merkez satırlar (`M = sinif(merkez baytı)`): önce sol/sağ dolgu, sonra
/// merkez kopyası. `b`'nin yatay alanları bayt.
#[target_feature(enable = "avx2")]
fn merkez16<const M: usize>(b: Bolum, dst: &mut [u8], ds: usize, src: &[u8], ss: usize) {
    let (k_sol, k_sag) = (genis(b.sol, b.sol + b.mw), genis(b.sag, b.sag + b.mw));
    for r in 0..b.mh {
        let s0 = b.ref_row + r * ss;
        let satir = &mut dst[(b.ust + r) * ds..(b.ust + r) * ds + b.bw];
        let kaynak = &src[s0..s0 + b.mw];
        if b.sol > 0 {
            doldur16(
                &mut satir[..k_sol],
                u16::from_ne_bytes([kaynak[0], kaynak[1]]),
            );
        }
        if b.sag > 0 {
            let v = u16::from_ne_bytes([kaynak[b.mw - 2], kaynak[b.mw - 1]]);
            doldur16(&mut satir[b.bw - k_sag..], v);
        }
        kopyala::<M>(&mut satir[b.sol..b.sol + b.mw], kaynak);
    }
}

#[target_feature(enable = "avx2")]
fn emu16_govde(
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
    // Yatay alanlar bayt (piksel × 2).
    let b = Bolum {
        bw: 2 * bw,
        ust,
        mh: bh - ust - alt,
        alt,
        sol: 2 * sol,
        sag: 2 * sag,
        mw: 2 * (bw - sol - sag),
        ref_row: y.clamp(0, ih as i32 - 1) as usize * ss + 2 * x.clamp(0, iw as i32 - 1) as usize,
    };
    match sinif(b.mw) {
        32 => merkez16::<32>(b, dst, ds, src, ss),
        16 => merkez16::<16>(b, dst, ds, src, ss),
        8 => merkez16::<8>(b, dst, ds, src, ss),
        4 => merkez16::<4>(b, dst, ds, src, ss),
        _ => merkez16::<1>(b, dst, ds, src, ss),
    }
    if ust + alt > 0 {
        match sinif(b.bw) {
            32 => dikey::<32>(b, dst, ds),
            16 => dikey::<16>(b, dst, ds),
            8 => dikey::<8>(b, dst, ds),
            4 => dikey::<4>(b, dst, ds),
            _ => dikey::<1>(b, dst, ds),
        }
    }
}

fn emu_edge16_giris(
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
    if !kenar16_sozlesmesi(
        bw,
        bh,
        iw,
        ih,
        x,
        y,
        dst.len(),
        ds,
        src.len(),
        ss,
        bd,
        bdmax,
    ) {
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
    // `kenar16_sozlesmesi` kırpılmış okuma penceresini kaynak diliminde ve hedefte
    // `bh` satır × `2·bw` baytı kapsar; erişimler denetimlidir. (3) hizalama/takma
    // ad: `dst` tek `&mut`, `src` `&`; ödünç kuralları çakışmayı dışlar.
    unsafe { emu16_govde(bw, bh, iw, ih, x, y, dst, ds, src, ss) }
}

/// 16 bpc `emu_edge` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn emu_edge_fn(_avx2: Avx2Token) -> EmuEdgeFn {
    emu_edge16_giris
}
