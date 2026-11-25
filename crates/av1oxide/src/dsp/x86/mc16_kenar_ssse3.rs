//! MC `emu_edge` SSSE3 çekirdeği (16 bpc, 10/12 bit) — F4-MC d (P35).
//!
//! ## Sözleşme
//!
//! AVX2 gövdesiyle ([`super::mc16_kenar`]) aynı: okuma yalnız kırpılmış pencere
//! (skalerle aynı), yazma yalnız hedefin `bh` satır × `2·bw` baytı; girişte O(1)
//! denetim ([`crate::dsp::mc::kenar16_sozlesmesi`]). Sözleşme dışı girdi skalere
//! devreder ve sayılır.
//!
//! ## Yöntem
//!
//! AVX2 gövdesinin yolu (önce merkez, sonra kenarlar) 128 bit yazmaçla: merkez
//! satırlar 16 baytlık parçalarla kopyalanır; `n` baytlık bir kopyanın ya da
//! doldurmanın son parçası `n − 16` (`n − 8`, …) konumundan **örtüşerek**
//! yazılır — okuma ve yazma bölgenin içinde kalır. Sol/sağ uzantı kenar
//! pikselinin `u16` yayınıyla (parça uzunlukları ve örtüşen son parça
//! konumları çift), üst/alt satırlar ilk/son görünür satırın kopyasıyla
//! doldurulur. Yalnız SSE2 yükleme/saklama ve yayın komutları; SSE4.1 ve üstü
//! komut yoktur.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mem::{chunk, chunk_mut, load_u8x16, store_u8x4, store_u8x8, store_u8x16};
use crate::cpu::Ssse3Token;
use crate::dsp::mc::{EmuEdgeFn, kenar16_sozlesmesi, skaler_devri_say};
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::_mm_set1_epi16;

// 8 bpc SSSE3 `mc_kenar_ssse3` gövdesinin bayt tabanlı yardımcıları (aynı kod;
// AVX2 gövdelerindeki gibi her bit derinliği kendi kopyasını kullanır).

/// Kopya sınıfı: `n` baytı bir döngüyle (16) ya da iki örtüşen parçayla
/// (8/4) kapsayan parça boyu; `n < 4` için 1 (bayt bayt).
const fn sinif(n: usize) -> usize {
    if n >= 16 {
        16
    } else if n >= 8 {
        8
    } else if n >= 4 {
        4
    } else {
        1
    }
}

/// `d` ← `s` (eşit uzunluk `n`, `sinif(n) == K`). `K = 16`'da 16 baytlık
/// parçalar, son parça **örtüşerek** `n − 16`'dan; `K` 8/4'te iki örtüşen
/// parça (`0` ve `n − K`); `K = 1`'de (`n` 1–3) bayt bayt. Okuma ve yazma dilimlerin
/// içinde kalır. Sınıf blok başına bir kez seçilir (satır döngüsünde dal yok).
#[target_feature(enable = "ssse3")]
#[inline]
fn kopyala<const K: usize>(d: &mut [u8], s: &[u8]) {
    let n = d.len();
    match K {
        16 => {
            let mut k = 0;
            while k + 16 < n {
                store_u8x16(chunk_mut(d, k), load_u8x16(chunk(s, k)));
                k += 16;
            }
            store_u8x16(chunk_mut(d, n - 16), load_u8x16(chunk(s, n - 16)));
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

/// Kenar dolgusunun yazılacağı uzunluk: `n` tek parçaya (4/8/16) genişler,
/// ama merkezin karşı ucunu (`sinir`) aşmaz; taşan baytları merkez kopyası ezer.
const fn genis(n: usize, sinir: usize) -> usize {
    let k = if n <= 4 {
        4
    } else if n <= 8 {
        8
    } else if n <= 16 {
        16
    } else {
        n
    };
    if k < sinir { k } else { sinir }
}

/// Üst/alt satırlar (`B = sinif(bw)`): ilk/son görünür satırın kopyası.
#[target_feature(enable = "ssse3")]
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
#[target_feature(enable = "ssse3")]
#[inline]
fn doldur16(d: &mut [u8], v: u16) {
    let n = d.len();
    if n >= 16 {
        let x = _mm_set1_epi16(v as i16);
        let mut k = 0;
        while k + 16 < n {
            store_u8x16(chunk_mut(d, k), x);
            k += 16;
        }
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
#[target_feature(enable = "ssse3")]
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

#[target_feature(enable = "ssse3")]
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
        16 => merkez16::<16>(b, dst, ds, src, ss),
        8 => merkez16::<8>(b, dst, ds, src, ss),
        4 => merkez16::<4>(b, dst, ds, src, ss),
        _ => merkez16::<1>(b, dst, ds, src, ss),
    }
    if ust + alt > 0 {
        match sinif(b.bw) {
            16 => dikey::<16>(b, dst, ds),
            8 => dikey::<8>(b, dst, ds),
            4 => dikey::<4>(b, dst, ds),
            _ => dikey::<1>(b, dst, ds),
        }
    }
}

// P37: giriş yerel kopya (yuva işlevinin birimine açılır). P36'dan sonra giriş, gövde ve
// `bitstream::frame_header::derive_lossless` aynı kod üretim biriminde kalınca IPSCCP
// birimdeki `i32::clamp` dönüşünü negatif olmayan bilemiyor (`zext nneg` → `sext`) ve
// `emu16_govde` P35'ten ayrışıyordu (`emu_edge` kose/ust w128@10 +%13–18). Giriş birim
// dışındayken gövdenin (ve `mc_kenar::emu_govde`'nin) makine kodu P35 ile aynı.
#[inline(always)]
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`emu_edge_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `kenar16_sozlesmesi` kırpılmış okuma penceresini kaynak diliminde ve hedefte
    // `bh` satır × `2·bw` baytı kapsar; erişimler denetimlidir. (3) hizalama/takma
    // ad: `dst` tek `&mut`, `src` `&`; ödünç kuralları çakışmayı dışlar.
    unsafe { emu16_govde(bw, bh, iw, ih, x, y, dst, ds, src, ss) }
}

/// 16 bpc `emu_edge` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn emu_edge_fn(_ssse3: Ssse3Token) -> EmuEdgeFn {
    emu_edge16_giris
}
