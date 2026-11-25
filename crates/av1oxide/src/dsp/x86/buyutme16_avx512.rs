//! Superres büyütme AVX512ICL çekirdeği (16 bpc, 10/12 bit) — F4-MC e (P36):
//! `resize`.
//!
//! ## Sözleşme
//!
//! AVX2 gövdesinin sözleşmesi ([`super::buyutme16`],
//! [`crate::dsp::resize::buyutme16_sozlesmesi`]); sözleşme dışı girdi skalere
//! devreder ve sayılır. `dx > 2¹⁵` aynı girişten AVX2 gövdesine gider. Okuma
//! yalnız her satırın `src_w` pikseli: iç grupta aralık **maskeli** yüklenir
//! (çalışma zamanı uzunluğu, satırın içinde), kenar grubunda kırpılmış
//! pikseller yerel diziye kurulur. Son grubun `dst_w`'yi aşan sütunları
//! yazılmaz (maskeli saklama). Yuva `SLOW_GATHER` bayrağı olan işlemcide
//! kurulmaz.
//!
//! ## Yöntem
//!
//! Sütun önce, 16 çıktılık gruplar: grubun kaynak aralığı (`dx ≤ 2¹⁵` ile en
//! çok 38 piksel) iki zmm'e yüklenir; tap ikilisi `j` için `vpermt2w` dword `x`
//! = `(s[bas_x + 2j], s[bas_x + 2j + 1])`, filtre ikilisi dword `x` =
//! `(f_x[2j], f_x[2j+1])` (grup başına bir kez); `vpdpwssd` (doymasız) 16
//! çıktının toplamı doğrudan sırayla; `(64 − Σ) >> 7`, `vpmovusdw` +
//! `pminuw(bitdepth_max)`.

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::buyutme16 as avx2;
use super::mem::{chunk, chunk_mut, load_u8_kn, load_u8x64, store_u8x32, store_u8x32_kn};
use crate::cpu::Avx512IclToken;
use crate::dsp::resize::ResizeFn;
use crate::dsp::resize_tables::RESIZE_FILTER;
use core::arch::x86_64::{
    __m512i, _mm256_min_epu16, _mm256_set1_epi16, _mm512_cvtusepi32_epi16, _mm512_dpwssd_epi32,
    _mm512_max_epi32, _mm512_permutex2var_epi16, _mm512_set1_epi32, _mm512_setr_epi32,
    _mm512_setzero_si512, _mm512_srai_epi32, _mm512_sub_epi32,
};

/// Filtre ikilileri: `[j][i]` = filtre `i`'nin `(f[2j], f[2j+1])` `i16`
/// ikilisi (alt sözcük `f[2j]`).
static FILTRE_IKILI: [[i32; 64]; 4] = {
    let mut t = [[0i32; 64]; 4];
    let mut j = 0;
    while j < 4 {
        let mut i = 0;
        while i < 64 {
            let a = RESIZE_FILTER[i][2 * j] as i16 as u16 as u32;
            let b = RESIZE_FILTER[i][2 * j + 1] as i16 as u16 as u32;
            t[j][i] = (a | (b << 16)) as i32;
            i += 1;
        }
        j += 1;
    }
    t
};

/// `vpermt2w` dizinleri (ikili `j`): sözcük `2x + b` = `fark[x] + 2j + b`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn dizin(fark: &[i64; 16], j: usize) -> __m512i {
    let d = |x: usize| {
        let p = fark[x] as i32 + 2 * j as i32;
        p | ((p + 1) << 16)
    };
    _mm512_setr_epi32(
        d(0),
        d(1),
        d(2),
        d(3),
        d(4),
        d(5),
        d(6),
        d(7),
        d(8),
        d(9),
        d(10),
        d(11),
        d(12),
        d(13),
        d(14),
        d(15),
    )
}

/// Filtre ikilisi `j`, 16 çıktı.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn filtre(fil: &[usize; 16], j: usize) -> __m512i {
    let t = &FILTRE_IKILI[j];
    _mm512_setr_epi32(
        t[fil[0]], t[fil[1]], t[fil[2]], t[fil[3]], t[fil[4]], t[fil[5]], t[fil[6]], t[fil[7]],
        t[fil[8]], t[fil[9]], t[fil[10]], t[fil[11]], t[fil[12]], t[fil[13]], t[fil[14]],
        t[fil[15]],
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
#[inline(never)]
pub(super) fn buyut16_govde(
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
    let k64 = _mm512_set1_epi32(64);
    let sifir = _mm512_setzero_si512();
    let max = _mm256_set1_epi16(bitdepth_max as i16);
    let mut mx = mx0;
    let mut src_x: i64 = -1;
    let mut x0 = 0;
    while x0 < dst_w {
        let n = (dst_w - x0).min(16);
        let mut bas = [0i64; 16];
        let mut fil = [0usize; 16];
        for k in 0..16 {
            bas[k] = src_x - 3;
            fil[k] = (mx >> 8) as usize;
            mx += dx;
            src_x += i64::from(mx >> 14);
            mx &= 0x3fff;
        }
        // `dx ≤ 2¹⁵`: aralık ≤ 38 piksel = 76 bayt (iki zmm).
        let aralik = 2 * (bas[15] + 8 - bas[0]) as usize;
        let ic = bas[0] >= 0 && bas[15] + 7 <= son;
        let fark: [i64; 16] = core::array::from_fn(|i| bas[i] - bas[0]);
        let ix: [__m512i; 4] = core::array::from_fn(|j| dizin(&fark, j));
        let g: [__m512i; 4] = core::array::from_fn(|j| filtre(&fil, j));
        for y in 0..h {
            let satir = y * ss;
            let (a, b) = if ic {
                let s = &src[satir + 2 * bas[0] as usize..];
                (
                    load_u8_kn(s, aralik.min(64)),
                    load_u8_kn(s.get(64..).unwrap_or(&[]), aralik.saturating_sub(64)),
                )
            } else {
                let mut t = [0u8; 128];
                for (p, v) in t
                    .as_chunks_mut::<2>()
                    .0
                    .iter_mut()
                    .enumerate()
                    .take(aralik / 2)
                {
                    let sx = (bas[0] + p as i64).clamp(0, son) as usize;
                    v.copy_from_slice(chunk::<u8, 2>(src, satir + 2 * sx));
                }
                (load_u8x64(chunk(&t, 0)), load_u8x64(chunk(&t, 64)))
            };
            let mut acc = sifir;
            for j in 0..4 {
                acc = _mm512_dpwssd_epi32(acc, _mm512_permutex2var_epi16(a, ix[j], b), g[j]);
            }
            let t = _mm512_srai_epi32::<7>(_mm512_sub_epi32(k64, acc));
            let v = _mm256_min_epu16(_mm512_cvtusepi32_epi16(_mm512_max_epi32(t, sifir)), max);
            let o = y * ds + 2 * x0;
            if n == 16 {
                store_u8x32(chunk_mut(dst, o), v);
            } else {
                // Son grup: yalnız `2n` bayt (maskeli saklama; satırın ötesi değişmez).
                store_u8x32_kn(&mut dst[o..o + 2 * n], v, 2 * n);
            }
        }
        x0 += 16;
    }
}

/// 16 bpc `resize` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn resize_fn(_avx512: Avx512IclToken) -> ResizeFn {
    avx2::buyut16_giris_icl
}
