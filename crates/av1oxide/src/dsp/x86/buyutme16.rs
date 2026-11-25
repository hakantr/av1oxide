//! Superres büyütme AVX2 çekirdeği (16 bpc, 10/12 bit) — F4-MC c (P34):
//! `resize`.
//!
//! ## Sözleşme
//!
//! Çıktı pikseli `x` için filtre `RESIZE_FILTER[mx >> 8]`, kaynak dokunuşları
//! `src_x − 3 .. src_x + 4` (her biri `[0, src_w − 1]`'e kırpılır); her satır
//! `mx = mx0`, `src_x = −1` ile başlar, sonra `mx += dx; src_x += mx >> 14;
//! mx &= 0x3fff`. Pikseller `u16` küçük sonlu, satır adımları bayt. Okuma
//! yalnız her satırın `src_w` pikselidir (kırpma sayesinde dışarı taşmaz),
//! yazma her satırın `dst_w` pikseli. Girişte O(1) denetim
//! ([`crate::dsp::resize::buyutme16_sozlesmesi`]): bit derinliği 10/12,
//! skaler gövdenin koşulları ve satır adımları `≥ 2·genişlik`. Sözleşme dışı
//! girdi skalere devreder ve sayılır.
//!
//! ## Yöntem
//!
//! Sütun önce (8 bpc gövdesi gibi): 8 çıktı sütunluk grubun kaynak konumları
//! ve filtreleri bir kez hesaplanır, ardından bütün satırlar işlenir. Pencere
//! (sütun başına 8 piksel = 16 bayt) bütünüyle `[0, src_w − 1]` içindeyse
//! doğrudan okunur, değilse kırpılmış piksellerle kurulur. Filtre `i16`
//! (derleme zamanı tablosu [`FILTRE16`]); sütun `x` ve `x + 4` bir ymm'in iki
//! şeridinde `pmaddwd`, `phaddd` ağacı, `(64 − Σ) >> 7`, `packusdw` +
//! `pminuw(bitdepth_max)` kırpması.
//!
//! `unsafe` yalnız belirteçli girişteki `#[target_feature]` çağrısıdır; dosya
//! `audit::UNSAFE_EXEMPTIONS` defterinde kayıtlıdır. Yükleme/saklama
//! [`super::mem`] yardımcılarıyla yapılır.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mem::{chunk, chunk_mut, load_u8x16, store_u8x16};
use crate::cpu::Avx2Token;
use crate::dsp::mc::skaler_devri_say;
use crate::dsp::resize::{ResizeBitDepth, ResizeFn, buyutme16_sozlesmesi};
use crate::dsp::resize_tables::RESIZE_FILTER;
use core::arch::x86_64::{
    _mm_min_epu16, _mm_set1_epi16, _mm_unpacklo_epi64, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_hadd_epi32, _mm256_madd_epi16, _mm256_packus_epi32,
    _mm256_set_m128i, _mm256_set1_epi32, _mm256_setzero_si256, _mm256_srai_epi32, _mm256_sub_epi32,
};

/// Filtre tablosu: 8 tap `i16` küçük sonlu (16 bayt).
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

#[target_feature(enable = "avx2")]
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
    let k64 = _mm256_set1_epi32(64);
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
        let f: [_; 4] = core::array::from_fn(|j| {
            _mm256_set_m128i(
                load_u8x16(&FILTRE16[fil[j + 4]]),
                load_u8x16(&FILTRE16[fil[j]]),
            )
        });
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
            let mut m = [_mm256_setzero_si256(); 4];
            for (j, mj) in m.iter_mut().enumerate() {
                // Şerit 0: sütun `j`, şerit 1: sütun `j + 4`.
                let p = if ic {
                    _mm256_set_m128i(
                        load_u8x16(chunk(src, satir + 2 * bas[j + 4] as usize)),
                        load_u8x16(chunk(src, satir + 2 * bas[j] as usize)),
                    )
                } else {
                    _mm256_set_m128i(load_u8x16(&pencere(j + 4)), load_u8x16(&pencere(j)))
                };
                *mj = _mm256_madd_epi16(p, f[j]);
            }
            // [x0, x1, x2, x3 | x4, x5, x6, x7]
            let s = _mm256_hadd_epi32(_mm256_hadd_epi32(m[0], m[1]), _mm256_hadd_epi32(m[2], m[3]));
            let t = _mm256_srai_epi32::<7>(_mm256_sub_epi32(k64, s));
            let p = _mm256_packus_epi32(t, t);
            let v = _mm_min_epu16(
                _mm_unpacklo_epi64(_mm256_castsi256_si128(p), _mm256_extracti128_si256::<1>(p)),
                max,
            );
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`resize_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `buyutme16_sozlesmesi` her satırın `2·src_w` kaynak ve `2·dst_w` hedef
    // baytını dilimlerde kapsar, dokunuşlar `[0, src_w − 1]`'e kırpılır; erişimler
    // denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
    unsafe { buyut16_govde(dst, ds, src, ss, dst_w, h, src_w, dx, mx0, bitdepth_max) }
}

/// 16 bpc `resize` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn resize_fn(_avx2: Avx2Token) -> ResizeFn {
    buyut16_giris
}

// ---------------------------------------------------------------------------
// P36: AVX512ICL girişleri (adres yalnız [`super::buyutme16_avx512`] belirteçli işaretçileriyle)
// ---------------------------------------------------------------------------
//
// Girişler bu modüldedir: dar bloklar bu modülün AVX2 gövdelerini, geniş bloklar
// `buyutme16_avx512`'in gövdelerini çağırır. Paylaşılan AVX2 gövdelerinin bütün çağırıcıları
// aynı kod üretim biriminde kalır ve iç bağlantıları korunur (başka birimden
// çağrılan tek kopya dış bağlantıya geçip LLVM'nin yerel işlev iyileştirmelerini
// yitiriyordu: AVX2 hücrelerinde +%10–20, P36 hız bütçesi tanı koşusu).

pub(super) fn buyut16_giris_icl(
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`buyutme16_avx512::resize_fn`); belirteç AVX512ICL (VNNI dahil; AVX2'yi kapsar) tespitinin kanıtıdır. (2)
    // sınır: `buyutme16_sozlesmesi` her satırın `2·src_w` kaynak ve `2·dst_w`
    // hedef baytını dilimlerde kapsar; maskeli yüklemeler yalnız satır içindeki
    // aralığı, kenar grubu kırpılmış dokunuşları okur; erişimler denetimlidir.
    // (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
    unsafe {
        if dx > 1 << 15 {
            buyut16_govde(dst, ds, src, ss, dst_w, h, src_w, dx, mx0, bitdepth_max)
        } else {
            super::buyutme16_avx512::buyut16_govde(
                dst,
                ds,
                src,
                ss,
                dst_w,
                h,
                src_w,
                dx,
                mx0,
                bitdepth_max,
            )
        }
    }
}
