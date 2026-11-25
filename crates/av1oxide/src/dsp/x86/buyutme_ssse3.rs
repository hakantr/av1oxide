//! Superres büyütme SSSE3 çekirdeği (8 bpc) — F4-MC d (P35): `resize`.
//!
//! ## Sözleşme
//!
//! AVX2 gövdesiyle ([`super::buyutme`]) aynı: okuma yalnız her satırın
//! `src_w` baytı (dokunuşlar `[0, src_w − 1]`'e kırpılır), yazma her satırın
//! `dst_w` baytı; girişte O(1) denetim
//! ([`crate::dsp::resize::buyutme_sozlesmesi`]). Sözleşme dışı girdi skalere
//! devreder ve sayılır.
//!
//! ## Yöntem
//!
//! Aritmetik ve yapı AVX2 gövdesiyle aynıdır; yazmaç 128 bittir. Sütun önce:
//! 8 çıktı sütunluk grubun kaynak konumları ve filtreleri bir kez hesaplanır,
//! ardından bütün satırlar işlenir. Bir yazmaçta iki sütun: pencereler
//! bütünüyle içerideyse tek 16 baytlık yükleme ve `pshufb` ile iki 8 baytlık
//! pencere (fark en çok 8 bayt; `dx ≤ 2¹⁵`), değilse kırpılmış baytlarla.
//! `pmaddubsw` (ikili toplam en az −32640, doymaz) + `pmaddwd(·, 1)` +
//! `phaddd` sıralı `[x0..x3]`, `[x4..x7]`, `(64 − Σ) >> 7`, `packuswb`
//! kırpması. SSE4.1 ve üstü komut yoktur.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mem::{chunk, chunk_mut, load_u8x16};
use crate::cpu::Ssse3Token;
use crate::dsp::mc::skaler_devri_say;
use crate::dsp::resize::{ResizeBitDepth, ResizeFn, buyutme_sozlesmesi};
use crate::dsp::resize_tables::RESIZE_FILTER;
use core::arch::x86_64::{
    _mm_add_epi8, _mm_cvtsi128_si64, _mm_hadd_epi32, _mm_madd_epi16, _mm_maddubs_epi16,
    _mm_packs_epi32, _mm_packus_epi16, _mm_set_epi64x, _mm_set1_epi16, _mm_set1_epi32,
    _mm_setr_epi8, _mm_setzero_si128, _mm_shuffle_epi8, _mm_srai_epi32, _mm_sub_epi32,
};

/// Filtre tablosu `u64` (küçük sonlu, tap sırası; AVX2 tablosuyla aynı, kendi kopyası).
static FILTRE: [i64; 64] = {
    let mut t = [0i64; 64];
    let mut i = 0;
    while i < 64 {
        let mut v = 0u64;
        let mut k = 0;
        while k < 8 {
            v |= ((RESIZE_FILTER[i][k] as u8) as u64) << (8 * k);
            k += 1;
        }
        t[i] = v as i64;
        i += 1;
    }
    t
};

#[target_feature(enable = "ssse3")]
fn buyut_govde(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    dst_w: usize,
    h: usize,
    src_w: usize,
    dx: i32,
    mx0: i32,
) {
    let son = src_w as i64 - 1;
    let bir = _mm_set1_epi16(1);
    let k64 = _mm_set1_epi32(64);
    let mut mx = mx0;
    let mut src_x: i64 = -1;
    let mut x0 = 0;
    while x0 < dst_w {
        let n = (dst_w - x0).min(8);
        let mut bas = [0i64; 8];
        let mut fil = [0i64; 8];
        for k in 0..8 {
            bas[k] = src_x - 3;
            fil[k] = FILTRE[(mx >> 8) as usize];
            mx += dx;
            src_x += i64::from(mx >> 14);
            mx &= 0x3fff;
        }
        // İç grup: 16 baytlık dört yükleme (`bas[0]`, `bas[2]`, `bas[4]`, `bas[6]`)
        // bölgede kalır ve her yüklemenin iki penceresi (en çok 2 + 8 = 10 bayt
        // aralık; `dx ≤ 2¹⁵`) bu yüklemenin içindedir.
        let ic = bas[0] >= 0 && bas[6] + 15 <= son && bas[7] + 7 <= son && dx <= 1 << 15;
        let f: [_; 4] = core::array::from_fn(|j| _mm_set_epi64x(fil[2 * j + 1], fil[2 * j]));
        // Karıştırma: ikinci pencere = taban + `bas[2j+1] − bas[2j]`.
        let taban = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7);
        let km: [_; 4] = core::array::from_fn(|j| {
            let fark = (bas[2 * j + 1] - bas[2 * j]) * 0x0101_0101_0101_0101;
            _mm_add_epi8(taban, _mm_set_epi64x(fark, 0))
        });
        for y in 0..h {
            let satir = y * ss;
            let mut m = [_mm_setzero_si128(); 4];
            for (j, mj) in m.iter_mut().enumerate() {
                let s = if ic {
                    let l = load_u8x16(chunk(src, satir + bas[2 * j] as usize));
                    _mm_shuffle_epi8(l, km[j])
                } else {
                    let p = |k: usize| -> i64 {
                        let mut b = [0u8; 8];
                        for (t, v) in b.iter_mut().enumerate() {
                            *v = src[satir + (bas[k] + t as i64).clamp(0, son) as usize];
                        }
                        i64::from_le_bytes(b)
                    };
                    _mm_set_epi64x(p(2 * j + 1), p(2 * j))
                };
                *mj = _mm_madd_epi16(_mm_maddubs_epi16(s, f[j]), bir);
            }
            let lo = _mm_srai_epi32::<7>(_mm_sub_epi32(k64, _mm_hadd_epi32(m[0], m[1])));
            let hi = _mm_srai_epi32::<7>(_mm_sub_epi32(k64, _mm_hadd_epi32(m[2], m[3])));
            let v = _mm_packs_epi32(lo, hi);
            let bayt = _mm_cvtsi128_si64(_mm_packus_epi16(v, v)).to_le_bytes();
            let o = y * ds + x0;
            if n == 8 {
                chunk_mut::<u8, 8>(dst, o).copy_from_slice(&bayt);
            } else {
                dst[o..o + n].copy_from_slice(&bayt[..n]);
            }
        }
        x0 += 8;
    }
}

fn buyut_giris(
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
    if bd != ResizeBitDepth::B8
        || !buyutme_sozlesmesi(dst.len(), ds, src.len(), ss, dst_w, h, src_w, dx, mx0)
    {
        skaler_devri_say();
        return (crate::dsp::resize::scalar_for(bd))(
            dst, ds, src, ss, dst_w, h, src_w, dx, mx0, bd,
        );
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`resize_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `buyutme_sozlesmesi` her satırın `src_w` kaynak ve `dst_w` hedef baytını
    // dilimlerde kapsar, dokunuşlar `[0, src_w − 1]`'e kırpılır; erişimler
    // denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
    unsafe { buyut_govde(dst, ds, src, ss, dst_w, h, src_w, dx, mx0) }
}

/// `resize` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn resize_fn(_ssse3: Ssse3Token) -> ResizeFn {
    buyut_giris
}
