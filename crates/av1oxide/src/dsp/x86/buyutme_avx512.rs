//! Superres büyütme AVX512ICL çekirdeği (8 bpc) — F4-MC e (P36): `resize`.
//!
//! ## Sözleşme
//!
//! AVX2 gövdesinin sözleşmesi ([`super::buyutme`],
//! [`crate::dsp::resize::buyutme_sozlesmesi`]); sözleşme dışı girdi skalere
//! devreder ve sayılır. `dx > 2¹⁵` (çıktı başına ikiden çok kaynak pikseli;
//! superres büyütmede `dx < 2¹⁴`) aynı girişten AVX2 gövdesine gider. Okuma
//! yalnız her satırın `src_w` baytı: iç grupta pencere bütünüyle satırın
//! içindedir ve **maskeli** yüklenir (çalışma zamanı uzunluğu), kenar grubunda
//! kırpılmış baytlar yerel diziye kurulur. Son grubun `dst_w`'yi aşan sütunları
//! yazılmaz (maskeli saklama). Yuva `SLOW_GATHER` bayrağı olan
//! işlemcide kurulmaz (dav1d paritesi; gövde toplama kullanmaz).
//!
//! ## Yöntem
//!
//! Sütun önce: 16 çıktı sütunluk grubun kaynak konumları ve filtreleri bir kez
//! hesaplanır. Grubun kaynak aralığı (`dx ≤ 2¹⁵` ile en çok 38 bayt) bir zmm'e
//! yüklenir; iki `vpermb` (VBMI) qword `k` = pencere `k` (8 bayt) dizer;
//! `vpdpbusd` (VNNI, `u8 × i8`, doymasız) qword başına iki dword toplamı,
//! ikisi toplanır, `vpermt2d` 16 toplamı sıraya dizer; `(64 − Σ) >> 7`,
//! `vpmovusdb` (`[0, 255]` kırpması; önce `max(·, 0)`).

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::buyutme as avx2;
use super::mem::{chunk_mut, load_u8_kn, load_u8x64, store_u8_kn, store_u8x16};
use crate::cpu::Avx512IclToken;
use crate::dsp::resize::ResizeFn;
use crate::dsp::resize_tables::RESIZE_FILTER;
use core::arch::x86_64::{
    __m512i, _mm512_add_epi8, _mm512_add_epi32, _mm512_cvtusepi32_epi8, _mm512_dpbusd_epi32,
    _mm512_max_epi32, _mm512_permutex2var_epi32, _mm512_permutexvar_epi8, _mm512_set1_epi32,
    _mm512_setr_epi32, _mm512_setr_epi64, _mm512_setzero_si512, _mm512_srai_epi32,
    _mm512_srli_epi64, _mm512_sub_epi32,
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

/// `vpermb` tabanı: bayt `8k + t` = `t` (pencere içi konum).
static TABAN: [u8; 64] = {
    let mut m = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        m[i] = (i % 8) as u8;
        i += 1;
    }
    m
};

/// Qword `i`'nin her baytına `fark[i]` ekli `vpermb` dizini.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn dizin(taban: __m512i, fark: [i64; 8]) -> __m512i {
    let y = |i: usize| (fark[i] as u64).wrapping_mul(0x0101_0101_0101_0101) as i64;
    _mm512_add_epi8(
        taban,
        _mm512_setr_epi64(y(0), y(1), y(2), y(3), y(4), y(5), y(6), y(7)),
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
pub(super) fn buyut_govde(
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
    let k64 = _mm512_set1_epi32(64);
    let sifir = _mm512_setzero_si512();
    let taban = load_u8x64(&TABAN);
    // `vpermt2d`: iki yazmacın qword alt dword'leri (çift dizinler) sırayla.
    let cift = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    let mut mx = mx0;
    let mut src_x: i64 = -1;
    let mut x0 = 0;
    while x0 < dst_w {
        let n = (dst_w - x0).min(16);
        let mut bas = [0i64; 16];
        let mut fil = [0i64; 16];
        for k in 0..16 {
            bas[k] = src_x - 3;
            fil[k] = FILTRE[(mx >> 8) as usize];
            mx += dx;
            src_x += i64::from(mx >> 14);
            mx &= 0x3fff;
        }
        // `dx ≤ 2¹⁵`: ardışık konum farkı ≤ 2, aralık ≤ 15·2 + 8 = 38 bayt.
        let aralik = (bas[15] + 8 - bas[0]) as usize;
        let ic = bas[0] >= 0 && bas[15] + 7 <= son;
        let fark = |o: usize| -> [i64; 8] { core::array::from_fn(|i| bas[o + i] - bas[0]) };
        let (ix0, ix1) = (dizin(taban, fark(0)), dizin(taban, fark(8)));
        let f = |o: usize| {
            _mm512_setr_epi64(
                fil[o],
                fil[o + 1],
                fil[o + 2],
                fil[o + 3],
                fil[o + 4],
                fil[o + 5],
                fil[o + 6],
                fil[o + 7],
            )
        };
        let (f0, f1) = (f(0), f(8));
        for y in 0..h {
            let satir = y * ss;
            let a = if ic {
                load_u8_kn(&src[satir + bas[0] as usize..], aralik)
            } else {
                let mut b = [0u8; 64];
                for (t, v) in b.iter_mut().enumerate().take(aralik) {
                    *v = src[satir + (bas[0] + t as i64).clamp(0, son) as usize];
                }
                load_u8x64(&b)
            };
            let d0 = _mm512_dpbusd_epi32(sifir, _mm512_permutexvar_epi8(ix0, a), f0);
            let d1 = _mm512_dpbusd_epi32(sifir, _mm512_permutexvar_epi8(ix1, a), f1);
            // Qword alt dword'ü: `Σ_{t<4} + Σ_{t≥4}`.
            let e0 = _mm512_add_epi32(d0, _mm512_srli_epi64::<32>(d0));
            let e1 = _mm512_add_epi32(d1, _mm512_srli_epi64::<32>(d1));
            let s = _mm512_permutex2var_epi32(e0, cift, e1);
            let t = _mm512_srai_epi32::<7>(_mm512_sub_epi32(k64, s));
            let v = _mm512_cvtusepi32_epi8(_mm512_max_epi32(t, sifir));
            let o = y * ds + x0;
            if n == 16 {
                store_u8x16(chunk_mut(dst, o), v);
            } else {
                // Son grup: yalnız `n` bayt (maskeli saklama; satırın ötesi değişmez).
                store_u8_kn(&mut dst[o..o + n], v, n);
            }
        }
        x0 += 16;
    }
}

/// `resize` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn resize_fn(_avx512: Avx512IclToken) -> ResizeFn {
    avx2::buyut_giris_icl
}
