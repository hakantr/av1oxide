//! MC warp AVX2 çekirdekleri (8 bpc) — F4-MC b (P33): `warp8x8`, `warp8x8t`.
//!
//! ## Sözleşme
//!
//! Kaynak dolgu hücresi (blok satırı −3, sütunu −3) `15 × 15` piksellik okuma
//! bölgesinin başıdır. Girişte O(1) denetim ([`crate::dsp::mc::warp_sozlesmesi`]):
//! `src_stride ≥ 15`, kaynak dilimi `14·ss + 15 + MC_KUYRUK` bayt (satır başına
//! 16 baytlık yükleme bölgeyi en çok 1 bayt aşar), hedef 8 satır × 8 piksel
//! (`warp8x8` bayt, `warp8x8t` `i16`, satır adımı ≥ 8) ve filtre indeksinin
//! `64 + ((t + 512) >> 10)` bütün `(satır, sütun)` için `[0, 192]` içinde
//! olması — `t` satır ve sütunda doğrusal olduğundan köşe değerleri yeter.
//! Sözleşme dışı girdi skalere devreder ve sayılır.
//!
//! ## Yöntem
//!
//! - **Yatay** (15 satır × 8 çıktı, her çıktının filtresi ayrı): satırın 16
//!   baytı iki lane'e yayılır, `pshufb` her çıktı `x` için
//!   `(s[x+k], s[x+k+4])` (k = 0…3) baytlarını dizer; filtreler aynı düzende
//!   derleme zamanında hazırlanmış tablodan ([`WARP_H`]) `u64` olarak alınır;
//!   `pmaddubsw` (ikili toplam en çok 255·127 = 32385, doymaz — komşu tap
//!   ikilileri 39780'e ulaşırdı), `pmaddwd(·, 1)`, `phaddd`; `(Σ + 4) >> 3`.
//! - **Dikey** (8 satır × 8 çıktı): ara satırlar 8×8 `i16` aktarımıyla
//!   sütunlara çevrilir; çıktı satırı `y` için sütun ikilileri `palignr` ile `y`
//!   kadar kaydırılır, filtreler `pmovsxbw` ile `i16`, `pmaddwd`, `phaddd`
//!   ağacı, `vpermd`; `warp8x8` `(Σ + 1024) >> 11` → bayt, `warp8x8t`
//!   `(Σ + 64) >> 7` (8 bpc'de `PREP_BIAS` 0).

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mem::{chunk, chunk_mut, load_u8x16, store_i16x8, store_u8x8};
use crate::cpu::Avx2Token;
use crate::dsp::mc::{Warp8x8Fn, Warp8x8tFn, skaler_devri_say, warp_sozlesmesi};
use crate::dsp::mc_tables::{MC_WARP_FILTER, N_WARP_FILTERS};
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, __m256i, _mm_packs_epi32, _mm_packus_epi16, _mm_set_epi64x, _mm_setzero_si128,
    _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi16,
    _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm256_add_epi32, _mm256_alignr_epi8,
    _mm256_broadcastsi128_si256, _mm256_castsi256_si128, _mm256_cvtepi8_epi16,
    _mm256_extracti128_si256, _mm256_hadd_epi32, _mm256_madd_epi16, _mm256_maddubs_epi16,
    _mm256_permutevar8x32_epi32, _mm256_set_m128i, _mm256_set1_epi16, _mm256_set1_epi32,
    _mm256_setr_epi8, _mm256_setr_epi32, _mm256_setr_epi64x, _mm256_shuffle_epi8,
    _mm256_srai_epi32,
};

/// Yatay geçiş filtre tablosu: her satır `(f0, f4, f1, f5, f2, f6, f3, f7)`
/// bayt düzeninde, küçük sonlu `i64`.
static WARP_H: [i64; N_WARP_FILTERS] = tablo(true);
/// Dikey geçiş filtre tablosu: `(f0, …, f7)` sırası, küçük sonlu `i64` (P34 16 bpc
/// gövdesi de kullanır).
pub(super) static WARP_V: [i64; N_WARP_FILTERS] = tablo(false);

const fn tablo(yatay: bool) -> [i64; N_WARP_FILTERS] {
    const SIRA: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
    let mut t = [0i64; N_WARP_FILTERS];
    let mut i = 0;
    while i < N_WARP_FILTERS {
        let mut v = 0u64;
        let mut k = 0;
        while k < 8 {
            let c = MC_WARP_FILTER[i][if yatay { SIRA[k] } else { k }];
            v |= ((c as u8) as u64) << (8 * k);
            k += 1;
        }
        t[i] = v as i64;
        i += 1;
    }
    t
}

/// Warp filtre indeksi `64 + ((t + 512) >> 10)` (sözleşme `[0, 192]`'yi
/// girişte denetler).
#[inline(always)]
pub(super) fn indeks(t: i32) -> usize {
    (64 + ((t + 512) >> 10)) as usize
}

/// Yatay geçiş: 15 ara satır (`i16` × 8, `xmm`).
#[target_feature(enable = "avx2")]
#[inline]
fn yatay(src: &[u8], ss: usize, abcd: &[i16; 4], mx: i32) -> [__m128i; 16] {
    // Çıktı x için (s[x+k], s[x+k+4]) baytları; x0..x3 bir yazmaçta, x4..x7 öbüründe.
    let m0 = _mm256_setr_epi8(
        0, 4, 1, 5, 2, 6, 3, 7, 1, 5, 2, 6, 3, 7, 4, 8, //
        2, 6, 3, 7, 4, 8, 5, 9, 3, 7, 4, 8, 5, 9, 6, 10,
    );
    let m1 = _mm256_setr_epi8(
        4, 8, 5, 9, 6, 10, 7, 11, 5, 9, 6, 10, 7, 11, 8, 12, //
        6, 10, 7, 11, 8, 12, 9, 13, 7, 11, 8, 12, 9, 13, 10, 14,
    );
    let bir = _mm256_set1_epi16(1);
    let yuv = _mm256_set1_epi32(4);
    let (a, b) = (i32::from(abcd[0]), i32::from(abcd[1]));
    let mut ara = [_mm_setzero_si128(); 16];
    let mut t0 = mx;
    for (r, satir) in ara.iter_mut().enumerate().take(15) {
        let s = _mm256_broadcastsi128_si256(load_u8x16(chunk(src, r * ss)));
        let f = |x: i32| WARP_H[indeks(t0 + x * a)];
        let f0 = _mm256_setr_epi64x(f(0), f(1), f(2), f(3));
        let f1 = _mm256_setr_epi64x(f(4), f(5), f(6), f(7));
        let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_shuffle_epi8(s, m0), f0), bir);
        let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_shuffle_epi8(s, m1), f1), bir);
        // [x0, x1, x4, x5 | x2, x3, x6, x7]
        let t = _mm256_srai_epi32::<3>(_mm256_add_epi32(_mm256_hadd_epi32(p0, p1), yuv));
        let lo = _mm256_castsi256_si128(t);
        let hi = _mm256_extracti128_si256::<1>(t);
        // packs: lo → [x0 x1 x4 x5 …], hi → [x2 x3 x6 x7 …]; dword araya alma sırayı kurar.
        *satir = _mm_unpacklo_epi32(_mm_packs_epi32(lo, lo), _mm_packs_epi32(hi, hi));
        t0 += b;
    }
    ara
}

/// 8×8 `i16` aktarımı: `r[0..8]` satırları → sütunlar.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) fn aktar(r: &[__m128i; 16], ilk: usize) -> [__m128i; 8] {
    let a0 = _mm_unpacklo_epi16(r[ilk], r[ilk + 1]);
    let a1 = _mm_unpackhi_epi16(r[ilk], r[ilk + 1]);
    let a2 = _mm_unpacklo_epi16(r[ilk + 2], r[ilk + 3]);
    let a3 = _mm_unpackhi_epi16(r[ilk + 2], r[ilk + 3]);
    let a4 = _mm_unpacklo_epi16(r[ilk + 4], r[ilk + 5]);
    let a5 = _mm_unpackhi_epi16(r[ilk + 4], r[ilk + 5]);
    let a6 = _mm_unpacklo_epi16(r[ilk + 6], r[ilk + 7]);
    let a7 = _mm_unpackhi_epi16(r[ilk + 6], r[ilk + 7]);
    let b0 = _mm_unpacklo_epi32(a0, a2);
    let b1 = _mm_unpackhi_epi32(a0, a2);
    let b2 = _mm_unpacklo_epi32(a1, a3);
    let b3 = _mm_unpackhi_epi32(a1, a3);
    let b4 = _mm_unpacklo_epi32(a4, a6);
    let b5 = _mm_unpackhi_epi32(a4, a6);
    let b6 = _mm_unpacklo_epi32(a5, a7);
    let b7 = _mm_unpackhi_epi32(a5, a7);
    [
        _mm_unpacklo_epi64(b0, b4),
        _mm_unpackhi_epi64(b0, b4),
        _mm_unpacklo_epi64(b1, b5),
        _mm_unpackhi_epi64(b1, b5),
        _mm_unpacklo_epi64(b2, b6),
        _mm_unpackhi_epi64(b2, b6),
        _mm_unpacklo_epi64(b3, b7),
        _mm_unpackhi_epi64(b3, b7),
    ]
}

/// `[lo | hi]` sütun ikilisini `y` satır kaydırır (lane başına `palignr`).
#[target_feature(enable = "avx2")]
#[inline]
pub(super) fn kaydir(hi: __m256i, lo: __m256i, y: usize) -> __m256i {
    match y {
        0 => lo,
        1 => _mm256_alignr_epi8::<2>(hi, lo),
        2 => _mm256_alignr_epi8::<4>(hi, lo),
        3 => _mm256_alignr_epi8::<6>(hi, lo),
        4 => _mm256_alignr_epi8::<8>(hi, lo),
        5 => _mm256_alignr_epi8::<10>(hi, lo),
        6 => _mm256_alignr_epi8::<12>(hi, lo),
        _ => _mm256_alignr_epi8::<14>(hi, lo),
    }
}

/// Warp gövdesi: `T = false` → `warp8x8` (bayt), `true` → `warp8x8t` (`i16`).
#[target_feature(enable = "avx2")]
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
    // Sütun ikilileri (x, x+1): üst 8 satır ve alt 8 satır (son satır 0).
    let lo: [__m256i; 4] = core::array::from_fn(|p| _mm256_set_m128i(ust[2 * p + 1], ust[2 * p]));
    let hi: [__m256i; 4] = core::array::from_fn(|p| _mm256_set_m128i(alt[2 * p + 1], alt[2 * p]));
    let sira = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let (c, d) = (i32::from(abcd[2]), i32::from(abcd[3]));
    let yuv = _mm256_set1_epi32(if T { 64 } else { 1024 });
    let mut t0 = my;
    for y in 0..8 {
        let f = |x: i32| WARP_V[indeks(t0 + x * c)];
        let mut m = [_mm256_set1_epi32(0); 4];
        for (p, mp) in m.iter_mut().enumerate() {
            let x = 2 * p as i32;
            let k = _mm256_cvtepi8_epi16(_mm_set_epi64x(f(x + 1), f(x)));
            *mp = _mm256_madd_epi16(kaydir(hi[p], lo[p], y), k);
        }
        // [x0, x2, x4, x6 | x1, x3, x5, x7] → x sırası.
        let s = _mm256_hadd_epi32(_mm256_hadd_epi32(m[0], m[1]), _mm256_hadd_epi32(m[2], m[3]));
        let s = _mm256_add_epi32(_mm256_permutevar8x32_epi32(s, sira), yuv);
        let v = if T {
            _mm256_srai_epi32::<7>(s)
        } else {
            _mm256_srai_epi32::<11>(s)
        };
        let p16 = _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        if T {
            store_i16x8(chunk_mut(tmp, y * ds), p16);
        } else {
            store_u8x8(chunk_mut(dst, y * ds), _mm_packus_epi16(p16, p16));
        }
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`warp8x8_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `warp_sozlesmesi` kaynak bölgesini (15 satır, `MC_KUYRUK`), hedefte 8 satır
    // × 8 baytı ve filtre indeksinin `[0, 192]` aralığını kapsar; erişimler
    // denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`warp8x8t_fn`); belirteç AVX2 tespitinin kanıtıdır. (2) sınır:
    // `warp_sozlesmesi` kaynak bölgesini, `tmp`'de 8 satır × 8 öğeyi ve filtre
    // indeksini kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `tmp`
    // tek `&mut`, `src` `&`.
    unsafe { warp_govde::<true>(&mut [], tmp, ts, src, ss, abcd, mx, my) }
}

/// `warp8x8` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn warp8x8_fn(_avx2: Avx2Token) -> Warp8x8Fn {
    warp8x8_giris
}

/// `warp8x8t` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn warp8x8t_fn(_avx2: Avx2Token) -> Warp8x8tFn {
    warp8x8t_giris
}
