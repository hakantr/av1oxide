//! Superres büyütme AVX2 çekirdeği (8 bpc) — F4-MC b (P33): `resize`.
//!
//! ## Sözleşme
//!
//! Çıktı pikseli `x` için filtre `RESIZE_FILTER[mx >> 8]`, kaynak dokunuşları
//! `src_x − 3 .. src_x + 4` (her biri `[0, src_w − 1]`'e kırpılır); her satır
//! `mx = mx0`, `src_x = −1` ile başlar, sonra `mx += dx; src_x += mx >> 14;
//! mx &= 0x3fff`. Okuma yalnız her satırın `src_w` baytıdır (kırpma sayesinde
//! dışarı taşmaz), yazma her satırın `dst_w` baytı. Girişte O(1) denetim
//! ([`crate::dsp::resize::buyutme_sozlesmesi`]): skaler gövdenin koşulları
//! (`h`, `dst_w`, `src_w` ≥ 1, `mx0` 0…0x3fff, `dx` 1…2¹⁶, dilim boyları) ve
//! satır adımları `≥` genişlik. Sözleşme dışı girdi skalere devreder ve
//! sayılır.
//!
//! ## Yöntem (dav1d `resize_8bpc` yaklaşımı; asm kopyalanmadı)
//!
//! Sütun önce: 8 çıktı sütunluk grubun kaynak konumları ve filtreleri bir kez
//! hesaplanır (satırdan bağımsız), ardından bütün satırlar işlenir. Pencere
//! bütünüyle `[0, src_w − 1]` içindeyse sütun başına 8 bayt doğrudan okunur,
//! değilse kırpılmış baytlarla kurulur. `pmaddubsw` (ikili toplam en az
//! −32640, doymaz) + `pmaddwd(·, 1)` + `phaddd`, `(64 − Σ) >> 7`,
//! `packuswb` kırpması.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mem::{chunk, chunk_mut, load_u8x16};
use crate::cpu::Avx2Token;
use crate::dsp::mc::skaler_devri_say;
use crate::dsp::resize::{ResizeBitDepth, ResizeFn, buyutme_sozlesmesi};
use crate::dsp::resize_tables::RESIZE_FILTER;
use core::arch::x86_64::{
    _mm_cvtsi128_si64, _mm_packs_epi32, _mm_packus_epi16, _mm_unpacklo_epi32, _mm256_add_epi8,
    _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_hadd_epi32, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_set_m128i, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi8,
    _mm256_setr_epi64x, _mm256_shuffle_epi8, _mm256_srai_epi32, _mm256_sub_epi32,
};

/// Filtre tablosu `u64` (küçük sonlu, tap sırası).
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

#[target_feature(enable = "avx2")]
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
    let bir = _mm256_set1_epi16(1);
    let k64 = _mm256_set1_epi32(64);
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
        // bölgede kalır ve her lane'in iki penceresi (en çok 2 + 8 = 10 bayt
        // aralık; `dx ≤ 2¹⁵`) bu yüklemenin içindedir.
        let ic = bas[0] >= 0 && bas[6] + 15 <= son && bas[7] + 7 <= son && dx <= 1 << 15;
        let f0 = _mm256_setr_epi64x(fil[0], fil[1], fil[2], fil[3]);
        let f1 = _mm256_setr_epi64x(fil[4], fil[5], fil[6], fil[7]);
        // Lane içi karıştırma: pencere `k` (lane'in ikinci penceresi) = taban + `bas[k] − bas[lane başı]`.
        let taban = _mm256_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, //
            0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
        );
        let fark = |a: usize, b: usize| (bas[b] - bas[a]) * 0x0101_0101_0101_0101;
        let k0 = _mm256_add_epi8(taban, _mm256_setr_epi64x(0, fark(0, 1), 0, fark(2, 3)));
        let k1 = _mm256_add_epi8(taban, _mm256_setr_epi64x(0, fark(4, 5), 0, fark(6, 7)));
        for y in 0..h {
            let satir = y * ss;
            let (s0, s1) = if ic {
                let l = |k: usize| load_u8x16(chunk(src, satir + bas[k] as usize));
                (
                    _mm256_shuffle_epi8(_mm256_set_m128i(l(2), l(0)), k0),
                    _mm256_shuffle_epi8(_mm256_set_m128i(l(6), l(4)), k1),
                )
            } else {
                let p = |k: usize| -> i64 {
                    let mut b = [0u8; 8];
                    for (t, v) in b.iter_mut().enumerate() {
                        *v = src[satir + (bas[k] + t as i64).clamp(0, son) as usize];
                    }
                    i64::from_le_bytes(b)
                };
                (
                    _mm256_setr_epi64x(p(0), p(1), p(2), p(3)),
                    _mm256_setr_epi64x(p(4), p(5), p(6), p(7)),
                )
            };
            let a = _mm256_madd_epi16(_mm256_maddubs_epi16(s0, f0), bir);
            let b = _mm256_madd_epi16(_mm256_maddubs_epi16(s1, f1), bir);
            // [x0, x1, x4, x5 | x2, x3, x6, x7]
            let t = _mm256_srai_epi32::<7>(_mm256_sub_epi32(k64, _mm256_hadd_epi32(a, b)));
            let lo = _mm256_castsi256_si128(t);
            let hi = _mm256_extracti128_si256::<1>(t);
            let v = _mm_unpacklo_epi32(_mm_packs_epi32(lo, lo), _mm_packs_epi32(hi, hi));
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`resize_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `buyutme_sozlesmesi` her satırın `src_w` kaynak ve `dst_w` hedef baytını
    // dilimlerde kapsar, dokunuşlar `[0, src_w − 1]`'e kırpılır; erişimler
    // denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
    unsafe { buyut_govde(dst, ds, src, ss, dst_w, h, src_w, dx, mx0) }
}

/// `resize` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn resize_fn(_avx2: Avx2Token) -> ResizeFn {
    buyut_giris
}

// ---------------------------------------------------------------------------
// P36: AVX512ICL girişleri (adres yalnız [`super::buyutme_avx512`] belirteçli işaretçileriyle)
// ---------------------------------------------------------------------------
//
// Girişler bu modüldedir: dar bloklar bu modülün AVX2 gövdelerini, geniş bloklar
// `buyutme_avx512`'in gövdelerini çağırır. Paylaşılan AVX2 gövdelerinin bütün çağırıcıları
// aynı kod üretim biriminde kalır ve iç bağlantıları korunur (başka birimden
// çağrılan tek kopya dış bağlantıya geçip LLVM'nin yerel işlev iyileştirmelerini
// yitiriyordu: AVX2 hücrelerinde +%10–20, P36 hız bütçesi tanı koşusu).

pub(super) fn buyut_giris_icl(
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`buyutme_avx512::resize_fn`); belirteç AVX512ICL (VBMI, VNNI dahil; AVX2'yi kapsar) tespitinin kanıtıdır.
    // (2) sınır: `buyutme_sozlesmesi` her satırın `src_w` kaynak ve `dst_w`
    // hedef baytını dilimlerde kapsar; maskeli yükleme yalnız satır içindeki
    // aralığı, kenar grubu kırpılmış dokunuşları okur; erişimler denetimlidir.
    // (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
    unsafe {
        if dx > 1 << 15 {
            buyut_govde(dst, ds, src, ss, dst_w, h, src_w, dx, mx0)
        } else {
            super::buyutme_avx512::buyut_govde(dst, ds, src, ss, dst_w, h, src_w, dx, mx0)
        }
    }
}
