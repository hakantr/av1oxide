//! Superres büyütme SSSE3 çekirdeği (16 bpc, 10/12 bit) — F4-MC d (P35):
//! `resize`.
//!
//! ## Sözleşme
//!
//! AVX2 gövdesiyle ([`super::buyutme16`]) aynı: okuma yalnız her satırın
//! `src_w` pikseli (dokunuşlar `[0, src_w − 1]`'e kırpılır), yazma her satırın
//! `dst_w` pikseli; girişte O(1) denetim
//! ([`crate::dsp::resize::buyutme16_sozlesmesi`]). Sözleşme dışı girdi
//! skalere devreder ve sayılır.
//!
//! ## Yöntem
//!
//! Aritmetik ve yapı AVX2 gövdesiyle aynıdır; yazmaç 128 bittir. Sütun önce
//! (8 çıktı sütunluk grup); sütun başına 8 piksellik (16 bayt) pencere —
//! bütünüyle içerideyse doğrudan, değilse kırpılmış piksellerle — ve `i16`
//! filtre `pmaddwd`; `phaddd` ağacı sıralı `[x0..x3]`, `[x4..x7]`,
//! `(64 − Σ) >> 7`; kırpma `packssdw` + `pmaxsw(0)` + `pminsw(bitdepth_max)`
//! (`packusdw`/`pminuw` SSE4.1 yerine; `i32` sonuç `> 32767` iken iki yol da
//! `bitdepth_max`, `< 0` iken 0 verir). SSE4.1 ve üstü komut yoktur.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mem::{chunk, chunk_mut, load_u8x16, store_u8x16};
use crate::cpu::Ssse3Token;
use crate::dsp::mc::skaler_devri_say;
use crate::dsp::resize::{ResizeBitDepth, ResizeFn, buyutme16_sozlesmesi};
use crate::dsp::resize_tables::RESIZE_FILTER;
use core::arch::x86_64::{
    _mm_hadd_epi32, _mm_madd_epi16, _mm_max_epi16, _mm_min_epi16, _mm_packs_epi32, _mm_set1_epi16,
    _mm_set1_epi32, _mm_setzero_si128, _mm_srai_epi32, _mm_sub_epi32,
};

/// Filtre tablosu: 8 tap `i16` küçük sonlu (16 bayt; AVX2 tablosuyla aynı, kendi kopyası).
static FILTRE16: [[u8; 16]; 64] = {
    let mut t = [[0u8; 16]; 64];
    let mut i = 0;
    while i < 64 {
        let mut k = 0;
        while k < 8 {
            let b = (RESIZE_FILTER[i][k] as i16).to_le_bytes();
            t[i][2 * k] = b[0];
            t[i][2 * k + 1] = b[1];
            k += 1;
        }
        i += 1;
    }
    t
};

#[target_feature(enable = "ssse3")]
fn buyut16_govde(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    dst_w: usize,
    h: usize,
    src_w: usize,
    dx: i32,
    mx0: i32,
    bitdepth_max: u16,
) {
    let son = src_w as i64 - 1;
    let k64 = _mm_set1_epi32(64);
    let max = _mm_set1_epi16(bitdepth_max as i16);
    let mut mx = mx0;
    let mut src_x: i64 = -1;
    let mut x0 = 0;
    while x0 < dst_w {
        let n = (dst_w - x0).min(8);
        let mut bas = [0i64; 8];
        let mut fil = [0usize; 8];
        for k in 0..8 {
            bas[k] = src_x - 3;
            fil[k] = (mx >> 8) as usize;
            mx += dx;
            src_x += i64::from(mx >> 14);
            mx &= 0x3fff;
        }
        // Konumlar artmayan değil (`dx > 0`): ilk ve son pencere içerideyse hepsi içeride.
        let ic = bas[0] >= 0 && bas[7] + 7 <= son;
        let f: [_; 8] = core::array::from_fn(|j| load_u8x16(&FILTRE16[fil[j]]));
        for y in 0..h {
            let satir = y * ss;
            let pencere = |k: usize| -> [u8; 16] {
                let mut b = [0u8; 16];
                for t in 0..8 {
                    let sx = (bas[k] + t as i64).clamp(0, son) as usize;
                    b[2 * t..2 * t + 2].copy_from_slice(chunk::<u8, 2>(src, satir + 2 * sx));
                }
                b
            };
            let mut m = [_mm_setzero_si128(); 8];
            for (j, mj) in m.iter_mut().enumerate() {
                let p = if ic {
                    load_u8x16(chunk(src, satir + 2 * bas[j] as usize))
                } else {
                    load_u8x16(&pencere(j))
                };
                *mj = _mm_madd_epi16(p, f[j]);
            }
            let lo = _mm_hadd_epi32(_mm_hadd_epi32(m[0], m[1]), _mm_hadd_epi32(m[2], m[3]));
            let hi = _mm_hadd_epi32(_mm_hadd_epi32(m[4], m[5]), _mm_hadd_epi32(m[6], m[7]));
            let lo = _mm_srai_epi32::<7>(_mm_sub_epi32(k64, lo));
            let hi = _mm_srai_epi32::<7>(_mm_sub_epi32(k64, hi));
            let v = _mm_max_epi16(_mm_packs_epi32(lo, hi), _mm_setzero_si128());
            let v = _mm_min_epi16(v, max);
            let o = y * ds + 2 * x0;
            if n == 8 {
                store_u8x16(chunk_mut(dst, o), v);
            } else {
                let mut b = [0u8; 16];
                store_u8x16(&mut b, v);
                dst[o..o + 2 * n].copy_from_slice(&b[..2 * n]);
            }
        }
        x0 += 8;
    }
}

fn buyut16_giris(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    dst_w: usize,
    h: usize,
    src_w: usize,
    dx: i32,
    mx0: i32,
    bd: ResizeBitDepth,
) {
    if !buyutme16_sozlesmesi(dst.len(), ds, src.len(), ss, dst_w, h, src_w, dx, mx0, bd) {
        skaler_devri_say();
        return (crate::dsp::resize::scalar_for(bd))(
            dst, ds, src, ss, dst_w, h, src_w, dx, mx0, bd,
        );
    }
    let bitdepth_max = bd.bitdepth_max() as u16;
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`resize_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `buyutme16_sozlesmesi` her satırın `2·src_w` kaynak ve `2·dst_w` hedef
    // baytını dilimlerde kapsar, dokunuşlar `[0, src_w − 1]`'e kırpılır; erişimler
    // denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
    unsafe { buyut16_govde(dst, ds, src, ss, dst_w, h, src_w, dx, mx0, bitdepth_max) }
}

/// 16 bpc `resize` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn resize_fn(_ssse3: Ssse3Token) -> ResizeFn {
    buyut16_giris
}
