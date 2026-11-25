//! CDEF AVX-512 (AVX512ICL) `fb` çekirdekleri — 8 ve 16 bpc, F4-CDEF c (P30).
//!
//! Yöntem AVX2 ve SSE gövdeleriyle aynıdır (dolgu tamponu yok, satırlar
//! dilimlerden, tap komşusu satır yazmacından `pshufb`, eksik kenar = merkez
//! piksel); 512 bit yazmaç dört 128 bit şerit taşır:
//!
//! - `W = 8`: şerit `k` = satır `y + k` (yineleme başına **dört** satır; AVX2'de
//!   iki);
//! - `W = 4`: şerit `k` = satırlar `y + 2k`, `y + 2k + 1` (yineleme başına
//!   **sekiz** satır; 4×4 blokta üst iki şerit boştur ve yazılmaz).
//!
//! AVX-512'nin maske yazmaçları kullanılır: işaret uygulaması (`psignw`
//! AVX-512'de yok) `vpmovw2m` + maskeli `vpsubw`, yuvarlamadaki `sum < 0`
//! düzeltmesi maskeli `vpsubw`, eksik komşu seçimi `vpmovw2m` + maskeli
//! `vpblendmw`, çıktı satırları maskeli saklama (`vmovdqu8`/`vmovdqu16` `{k}`;
//! [`super::mem::store_u8_k`], [`super::mem::store_u16_k`]) ile yazılır —
//! yalnız bloğun `w` sütunu açılır.
//!
//! dav1d'de AVX512ICL CDEF `dir` gövdesi yoktur (`src/x86/cdef.h`); yuvaya
//! atama yalnız ölçümle (iki CCD'de AVX2'den hızlıysa ve çözme hattında
//! gerileme yoksa).
//!
//! ## Yetki ve `unsafe`
//!
//! Gövdeler güvenli `#[target_feature(enable = "avx512f,avx512bw,avx512vl")]`
//! işlevleridir; yuva işaretçileri yalnız [`Avx512IclToken`] ile verilir (dav1d
//! AVX512ICL bayrağı bu üç özelliği ve fazlasını içerir). `unsafe` yalnız iki
//! girişteki tek çağrıdır; dosya `audit::UNSAFE_EXEMPTIONS` defterindedir.
//!
//! ```compile_fail,E0061
//! // Belirteçsiz yuva işaretçisi alınamaz.
//! let _ = av1oxide::dsp::x86::cdef_avx512::filter_avx512_fn();
//! ```

#![allow(clippy::needless_range_loop)]
#![allow(unsafe_code)]

use super::cdef::{MAX_ROWS, N16, TAPS16, Tap, ilog2};
use super::cdef_sse::{
    SEL4, SEL8, SEL16_4A, SEL16_4B, SEL16_8A, SEL16_8B, VSEL4, VSEL8, filter_ok, filter_skaler,
};
use super::mem::{chunk, chunk_mut, load_u8x16, store_u8_k, store_u16_k};
use crate::cpu::Avx512IclToken;
use crate::dsp::cdef::{CdefEdgeFlags, CdefFilterFn, CdefFilterParams};
use core::arch::x86_64::{
    __m128i, __m512i, _mm_cvtsi32_si128, _mm_cvtsi64_si128, _mm_insert_epi16, _mm_insert_epi32,
    _mm_setzero_si128, _mm_slli_si128, _mm_srli_si128, _mm_unpacklo_epi64, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm512_abs_epi16, _mm512_add_epi16, _mm512_broadcast_i32x4,
    _mm512_castsi128_si512, _mm512_cvtepi16_epi8, _mm512_extracti32x4_epi32, _mm512_inserti32x4,
    _mm512_mask_blend_epi16, _mm512_mask_sub_epi16, _mm512_max_epi16, _mm512_min_epi16,
    _mm512_min_epu16, _mm512_movepi16_mask, _mm512_mulhrs_epi16, _mm512_mullo_epi16,
    _mm512_or_si512, _mm512_set1_epi16, _mm512_setzero_si512, _mm512_shuffle_epi8,
    _mm512_slli_epi16, _mm512_srai_epi16, _mm512_srl_epi16, _mm512_sub_epi16, _mm512_subs_epu16,
};

/// `s[off..off+N]` küçük sonlu tamsayısı (denetimli dizi referansıyla).
macro_rules! le {
    ($t:ty, $n:literal, $s:expr, $off:expr) => {
        <$t>::from_le_bytes(*chunk::<u8, $n>($s, $off))
    };
}

/// 8 bpc `fb` (AVX512ICL): yalnız [`Avx512IclToken`] ile.
#[must_use]
pub const fn filter_avx512_fn(_t: Avx512IclToken) -> CdefFilterFn {
    filter_avx512_entry
}

/// 16 bpc `fb` (AVX512ICL): yalnız [`Avx512IclToken`] ile.
#[must_use]
pub const fn filter16_avx512_fn(_t: Avx512IclToken) -> CdefFilterFn {
    filter16_avx512_entry
}

/// 8 bpc `fb` (AVX512ICL) girişi.
fn filter_avx512_entry(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    if !filter_ok(stride, dst, left, top, bottom, &params, false) {
        return filter_skaler(dst, stride, left, top, bottom, params);
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`filter_avx512_fn`); belirteç AVX512ICL (F + BW + VL ve fazlası)
    // tespiti + maske izninin kanıtıdır. (2) sınır: `filter_ok` gövdenin
    // okuduğu her baytı kapsar (dst satırları sütun `0..w+2`, bayrağı olan
    // kenarda top/bottom `row*stride + w + 4`, left `2h`); maskeli saklama
    // satırın `w + 2` baytlık dizisine yazar. (3) hizalama/takma ad: `dst` tek
    // `&mut`, kaynaklar `&`.
    unsafe { filter8(dst, stride, left, top, bottom, params) }
}

/// 16 bpc `fb` (AVX512ICL) girişi.
fn filter16_avx512_entry(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    if !filter_ok(stride, dst, left, top, bottom, &params, true) {
        return filter_skaler(dst, stride, left, top, bottom, params);
    }
    // SAFETY: (1) ISA: adres yalnız `Avx512IclToken` ile verilir
    // (`filter16_avx512_fn`). (2) sınır: `filter_ok` (`bpp = 2`) gövdenin
    // okuduğu her baytı kapsar; maskeli saklama satırın `2(w + 2)` baytlık
    // dizisine yazar. (3) hizalama/takma ad: `dst` tek `&mut`, kaynaklar `&`.
    unsafe { filter16(dst, stride, left, top, bottom, params) }
}

/// İki 128 bit şeritten 512 bit yazmaç (üst iki şerit 0).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn z2(a: __m128i, b: __m128i) -> __m512i {
    _mm512_inserti32x4::<1>(_mm512_castsi128_si512(a), b)
}

/// Dört 128 bit şeritten 512 bit yazmaç.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn z4(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> __m512i {
    let v = _mm512_inserti32x4::<1>(_mm512_castsi128_si512(a), b);
    _mm512_inserti32x4::<3>(_mm512_inserti32x4::<2>(v, c), d)
}

/// 128 bit tablo satırı dört şeride yayılmış.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn yay(m: &[u8; 16]) -> __m512i {
    _mm512_broadcast_i32x4(load_u8x16(m))
}

/// `d < 0` olan lane'lerde `−c`, diğerlerinde `c` (`psignw` eşi; `c = 0` iken
/// fark etmez).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn isaret(c: __m512i, d: __m512i) -> __m512i {
    _mm512_mask_sub_epi16(c, _mm512_movepi16_mask(d), _mm512_setzero_si512(), c)
}

// ---------------------------------------------------------------------------
// 8 bpc
// ---------------------------------------------------------------------------

/// 8 bpc kenar satırı: sütun `−2 .. w+2` (bayt 0 = sütun −2).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn edge_row<const W: usize>(s: &[u8], o: usize) -> __m128i {
    let v = _mm_cvtsi64_si128(le!(i64, 8, s, o));
    if W == 8 {
        _mm_insert_epi32::<2>(v, le!(i32, 4, s, o + 8))
    } else {
        v
    }
}

/// 8 bpc blok satırı: sütun −2, −1 `left`'ten (`l`), `0 .. w+2` `dst`'den.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn block_row<const W: usize>(dst: &[u8], o: usize, l: i32) -> __m128i {
    if W == 8 {
        let v = _mm_slli_si128::<2>(_mm_cvtsi64_si128(le!(i64, 8, dst, o)));
        let v = _mm_insert_epi16::<0>(v, l);
        _mm_insert_epi16::<5>(v, i32::from(le!(u16, 2, dst, o + 8)))
    } else {
        let v = _mm_slli_si128::<2>(_mm_cvtsi32_si128(le!(i32, 4, dst, o)));
        let v = _mm_insert_epi16::<0>(v, l);
        _mm_insert_epi16::<3>(v, i32::from(le!(u16, 2, dst, o + 4)))
    }
}

/// 8 bpc CDEF filtresi (AVX-512) — dav1d `cdef_filter_block_c` ile bit-exact.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn filter8(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    let pri = i32::from(params.pri_strength);
    let sec = i32::from(params.sec_strength);
    if pri == 0 && sec == 0 {
        return;
    }
    let all = params.edges.bits() == 0b1111;
    match (params.size.w(), all, pri != 0 && sec != 0) {
        (8, true, true) => filter8_w::<8, false, true>(dst, stride, left, top, bottom, params),
        (8, true, false) => filter8_w::<8, false, false>(dst, stride, left, top, bottom, params),
        (8, false, true) => filter8_w::<8, true, true>(dst, stride, left, top, bottom, params),
        (8, false, false) => filter8_w::<8, true, false>(dst, stride, left, top, bottom, params),
        (_, true, true) => filter8_w::<4, false, true>(dst, stride, left, top, bottom, params),
        (_, true, false) => filter8_w::<4, false, false>(dst, stride, left, top, bottom, params),
        (_, false, true) => filter8_w::<4, true, true>(dst, stride, left, top, bottom, params),
        (_, false, false) => filter8_w::<4, true, false>(dst, stride, left, top, bottom, params),
    }
}

/// `W = 8`: şerit `k` = satır `i + k`; `W = 4`: şerit `k` = satırlar
/// `i + 2k`, `i + 2k + 1` (`[r ‖ r']`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn pack8<const W: usize>(rs: &[__m128i; MAX_ROWS], i: usize, h: usize) -> __m512i {
    if W == 8 {
        z4(rs[i], rs[i + 1], rs[i + 2], rs[i + 3])
    } else {
        let p = |j: usize| _mm_unpacklo_epi64(rs[j], rs[j + 1]);
        if h == 4 {
            z2(p(i), p(i + 2))
        } else {
            z4(p(i), p(i + 2), p(i + 4), p(i + 6))
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn filter8_w<const W: usize, const EDGES: bool, const MINMAX: bool>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    let h = params.size.h();
    let n_rows = h + 4;
    let e = params.edges;
    let has_top = !EDGES || e.contains(CdefEdgeFlags::HAVE_TOP);
    let has_bottom = !EDGES || e.contains(CdefEdgeFlags::HAVE_BOTTOM);
    let has_left = !EDGES || e.contains(CdefEdgeFlags::HAVE_LEFT);

    // ---- 1) kaynak satırları (rows[i] = mantıksal satır i − 2; 4×4'te
    // rows[8..12] sıfır: yalnız yazılmayan şeritleri besler) --------------
    let z128 = _mm_setzero_si128();
    let mut rows = [z128; MAX_ROWS];
    if has_top {
        rows[0] = edge_row::<W>(top, 0);
        rows[1] = edge_row::<W>(top, stride);
    }
    for r in 0..h {
        let l = if has_left {
            i32::from(le!(u16, 2, left, 2 * r))
        } else {
            0
        };
        rows[r + 2] = block_row::<W>(dst, r * stride, l);
    }
    if has_bottom {
        rows[h + 2] = edge_row::<W>(bottom, 0);
        rows[h + 3] = edge_row::<W>(bottom, stride);
    }
    // Grup kaynakları: W = 8 → i = 0 .. n_rows − 3; W = 4 → i = 0 ..= 4.
    let z = _mm512_setzero_si512();
    let n_src = if W == 8 { n_rows - 3 } else { 5 };
    let mut srcs = [z; MAX_ROWS];
    for (i, v) in srcs.iter_mut().enumerate().take(n_src) {
        *v = pack8::<W>(&rows, i, h);
    }
    // Geçerlilik (yalnız kenar yolunda): bayt 0xFF = komşu yok.
    let mut vsrcs = [z; MAX_ROWS];
    if EDGES {
        let mut col = [0u8; 16];
        if !e.contains(CdefEdgeFlags::HAVE_LEFT) {
            col[0] = 0xff;
            col[1] = 0xff;
        }
        if !e.contains(CdefEdgeFlags::HAVE_RIGHT) {
            for c in col.iter_mut().skip(W + 2) {
                *c = 0xff;
            }
        }
        let colv = load_u8x16(&col);
        let allv = load_u8x16(&[0xff; 16]);
        let mut vrows = [z128; MAX_ROWS];
        for (i, v) in vrows.iter_mut().enumerate().take(n_rows) {
            let missing = (i < 2 && !has_top) || (i >= h + 2 && !has_bottom);
            *v = if missing { allv } else { colv };
        }
        for (i, v) in vsrcs.iter_mut().enumerate().take(n_src) {
            *v = pack8::<W>(&vrows, i, h);
        }
    }

    // ---- 2) tap listesi (dav1d `src/cdef_tmpl.c:121-155`) -----------------
    let (sel, vsel) = if W == 8 {
        (&SEL8, &VSEL8)
    } else {
        (&SEL4, &VSEL4)
    };
    let selz: [__m512i; 5] = core::array::from_fn(|i| yay(&sel[i]));
    let vselz: [__m512i; 5] = if EDGES {
        core::array::from_fn(|i| yay(&vsel[i]))
    } else {
        [z; 5]
    };
    let pri = i32::from(params.pri_strength);
    let sec = i32::from(params.sec_strength);
    let (pt, st) = TAPS16[usize::from(params.dir) & 7].split_at(4);
    let k0 = 4 - (pri & 1);
    let pk = [
        _mm512_set1_epi16(k0 as i16),
        _mm512_set1_epi16(((k0 & 3) | 2) as i16),
    ];
    let pri_thr = _mm512_set1_epi16(pri as i16);
    let sec_thr = _mm512_set1_epi16(sec as i16);
    let pri_cnt = _mm_cvtsi32_si128(if pri != 0 {
        (i32::from(params.damping) - ilog2(pri)).clamp(0, 31)
    } else {
        0
    });
    let sec_cnt = _mm_cvtsi32_si128(if sec != 0 {
        (i32::from(params.damping) - ilog2(sec)).clamp(0, 31)
    } else {
        0
    });
    let r2048 = _mm512_set1_epi16(2048);

    // ---- 3) satır grupları (W = 8: 4 satır; W = 4: 8 satır) ---------------
    let group = if W == 8 { 4 } else { 8 };
    let mut y = 0usize;
    while y < h {
        let px = _mm512_shuffle_epi8(srcs[y + 2], selz[2]);
        let mut minv = px;
        let mut maxv = px;
        let mut cons = |t: &Tap, thr: __m512i, cnt: __m128i| -> __m512i {
            let i = y + t.row;
            let bi = t.bi.min(4);
            let mut p = _mm512_shuffle_epi8(srcs[i], selz[bi]);
            if EDGES {
                let inv = _mm512_shuffle_epi8(vsrcs[i], vselz[bi]);
                p = _mm512_mask_blend_epi16(_mm512_movepi16_mask(inv), p, px);
            }
            let d = _mm512_sub_epi16(p, px);
            let ad = _mm512_abs_epi16(d);
            let c = _mm512_min_epu16(ad, _mm512_subs_epu16(thr, _mm512_srl_epi16(ad, cnt)));
            if MINMAX {
                minv = _mm512_min_epi16(p, minv);
                maxv = _mm512_max_epi16(p, maxv);
            }
            isaret(c, d)
        };
        let mut sum = z;
        if pri != 0 {
            for j in 0..2usize {
                let c = _mm512_add_epi16(
                    cons(&pt[2 * j], pri_thr, pri_cnt),
                    cons(&pt[2 * j + 1], pri_thr, pri_cnt),
                );
                sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(c, pk[j]));
            }
        }
        if sec != 0 {
            for j in 0..2usize {
                let c = _mm512_add_epi16(
                    _mm512_add_epi16(
                        cons(&st[4 * j], sec_thr, sec_cnt),
                        cons(&st[4 * j + 1], sec_thr, sec_cnt),
                    ),
                    _mm512_add_epi16(
                        cons(&st[4 * j + 2], sec_thr, sec_cnt),
                        cons(&st[4 * j + 3], sec_thr, sec_cnt),
                    ),
                );
                let c = if j == 0 { _mm512_slli_epi16::<1>(c) } else { c };
                sum = _mm512_add_epi16(sum, c);
            }
        }
        // `(sum - (sum < 0) + 8) >> 4` (dav1d `src/cdef_tmpl.c:165,184,210`)
        let r = _mm512_add_epi16(sum, _mm512_srai_epi16::<15>(sum));
        let mut r = _mm512_add_epi16(_mm512_mulhrs_epi16(r, r2048), px);
        if MINMAX {
            r = _mm512_min_epi16(_mm512_max_epi16(r, minv), maxv);
        }
        // i16 → u8 (alt bayt, C `pixel` dönüşümü: mod 256; `vpmovwb`).
        let b = _mm512_cvtepi16_epi8(r);
        let (lo, hi) = (_mm256_castsi256_si128(b), _mm256_extracti128_si256::<1>(b));
        if W == 8 {
            // satır y+k = bayt 8k .. 8k+8; maske yalnız w = 8 sütunu açar.
            for (k, v) in [lo, _mm_srli_si128::<8>(lo), hi, _mm_srli_si128::<8>(hi)]
                .into_iter()
                .enumerate()
            {
                store_u8_k::<10>(chunk_mut(dst, (y + k) * stride), v, 0x00ff);
            }
        } else {
            // satır y+j = bayt 4j .. 4j+4; 4×4'te yalnız ilk dört satır.
            let satirlar = [
                lo,
                _mm_srli_si128::<4>(lo),
                _mm_srli_si128::<8>(lo),
                _mm_srli_si128::<12>(lo),
                hi,
                _mm_srli_si128::<4>(hi),
                _mm_srli_si128::<8>(hi),
                _mm_srli_si128::<12>(hi),
            ];
            for (j, v) in satirlar.into_iter().enumerate().take(h - y) {
                store_u8_k::<6>(chunk_mut(dst, (y + j) * stride), v, 0x000f);
            }
        }
        y += group;
    }
}

// ---------------------------------------------------------------------------
// 16 bpc
// ---------------------------------------------------------------------------

/// 16 bpc kenar satırı: `(lo, hi)`, `lo` = sütun −2..5; `W = 8` iken `hi` =
/// sütun 6..9, `W = 4` iken 0.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn edge_row16<const W: usize>(s: &[u8], o: usize) -> (__m128i, __m128i) {
    let lo = load_u8x16(chunk(s, o));
    if W == 8 {
        (lo, _mm_cvtsi64_si128(le!(i64, 8, s, o + 16)))
    } else {
        (lo, _mm_setzero_si128())
    }
}

/// 16 bpc blok satırı: sütun −2, −1 `left`'ten (`l`), `0 .. w+2` `dst`'den.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn block_row16<const W: usize>(dst: &[u8], o: usize, l: i32) -> (__m128i, __m128i) {
    if W == 8 {
        let v = _mm_slli_si128::<4>(load_u8x16(chunk(dst, o)));
        (
            _mm_insert_epi32::<0>(v, l),
            _mm_cvtsi64_si128(le!(i64, 8, dst, o + 12)),
        )
    } else {
        let v = _mm_slli_si128::<4>(_mm_cvtsi64_si128(le!(i64, 8, dst, o)));
        let v = _mm_insert_epi32::<0>(v, l);
        (
            _mm_insert_epi32::<3>(v, le!(i32, 4, dst, o + 8)),
            _mm_setzero_si128(),
        )
    }
}

/// 16 bpc CDEF filtresi (AVX-512) — dav1d `cdef_filter_block_c` ile bit-exact.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn filter16(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    let pri = i32::from(params.pri_strength);
    let sec = i32::from(params.sec_strength);
    if pri == 0 && sec == 0 {
        return;
    }
    let all = params.edges.bits() == 0b1111;
    match (params.size.w(), all, pri != 0 && sec != 0) {
        (8, true, true) => filter16_w::<8, false, true>(dst, stride, left, top, bottom, params),
        (8, true, false) => filter16_w::<8, false, false>(dst, stride, left, top, bottom, params),
        (8, false, true) => filter16_w::<8, true, true>(dst, stride, left, top, bottom, params),
        (8, false, false) => filter16_w::<8, true, false>(dst, stride, left, top, bottom, params),
        (_, true, true) => filter16_w::<4, false, true>(dst, stride, left, top, bottom, params),
        (_, true, false) => filter16_w::<4, false, false>(dst, stride, left, top, bottom, params),
        (_, false, true) => filter16_w::<4, true, true>(dst, stride, left, top, bottom, params),
        (_, false, false) => filter16_w::<4, true, false>(dst, stride, left, top, bottom, params),
    }
}

/// 16 bpc grup kaynakları: `W = 8` → (`lo` dörtlüsü, `hi` dörtlüsü);
/// `W = 4` → (`r(i)`, `r(i+2)`, `r(i+4)`, `r(i+6)`) ve bir satır kaymışı.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn pack16<const W: usize>(
    lo: &[__m128i; N16],
    hi: &[__m128i; N16],
    i: usize,
    h: usize,
) -> (__m512i, __m512i) {
    let m = N16 - 1;
    if W == 8 {
        (
            z4(lo[i & m], lo[(i + 1) & m], lo[(i + 2) & m], lo[(i + 3) & m]),
            z4(hi[i & m], hi[(i + 1) & m], hi[(i + 2) & m], hi[(i + 3) & m]),
        )
    } else if h == 4 {
        (
            z2(lo[i & m], lo[(i + 2) & m]),
            z2(lo[(i + 1) & m], lo[(i + 3) & m]),
        )
    } else {
        (
            z4(lo[i & m], lo[(i + 2) & m], lo[(i + 4) & m], lo[(i + 6) & m]),
            z4(
                lo[(i + 1) & m],
                lo[(i + 3) & m],
                lo[(i + 5) & m],
                lo[(i + 7) & m],
            ),
        )
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
fn filter16_w<const W: usize, const EDGES: bool, const MINMAX: bool>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    let h = params.size.h();
    let n_rows = h + 4;
    let e = params.edges;
    let has_top = !EDGES || e.contains(CdefEdgeFlags::HAVE_TOP);
    let has_bottom = !EDGES || e.contains(CdefEdgeFlags::HAVE_BOTTOM);
    let has_left = !EDGES || e.contains(CdefEdgeFlags::HAVE_LEFT);
    let m = N16 - 1;

    // ---- 1) kaynak satırları ----------------------------------------------
    let z128 = _mm_setzero_si128();
    let mut lo = [z128; N16];
    let mut hi = [z128; N16];
    if has_top {
        (lo[0], hi[0]) = edge_row16::<W>(top, 0);
        (lo[1], hi[1]) = edge_row16::<W>(top, stride);
    }
    for r in 0..h {
        let l = if has_left {
            le!(i32, 4, left, 4 * r)
        } else {
            0
        };
        (lo[(r + 2) & m], hi[(r + 2) & m]) = block_row16::<W>(dst, r * stride, l);
    }
    if has_bottom {
        (lo[(h + 2) & m], hi[(h + 2) & m]) = edge_row16::<W>(bottom, 0);
        (lo[(h + 3) & m], hi[(h + 3) & m]) = edge_row16::<W>(bottom, stride);
    }
    let z = _mm512_setzero_si512();
    let n_src = if W == 8 { n_rows - 3 } else { 5 };
    let mut srcs_a = [z; N16];
    let mut srcs_b = [z; N16];
    for i in 0..n_src {
        (srcs_a[i & m], srcs_b[i & m]) = pack16::<W>(&lo, &hi, i, h);
    }
    let mut vsrcs_a = [z; N16];
    let mut vsrcs_b = [z; N16];
    if EDGES {
        let mut col_lo = [0u8; 16];
        let mut col_hi = [0u8; 16];
        if !e.contains(CdefEdgeFlags::HAVE_LEFT) {
            col_lo[..4].fill(0xff);
        }
        if !e.contains(CdefEdgeFlags::HAVE_RIGHT) {
            if W == 8 {
                col_hi[4..8].fill(0xff);
            } else {
                col_lo[12..16].fill(0xff);
            }
        }
        let (cl, ch) = (load_u8x16(&col_lo), load_u8x16(&col_hi));
        let allv = load_u8x16(&[0xff; 16]);
        let mut vlo = [z128; N16];
        let mut vhi = [z128; N16];
        for i in 0..n_rows {
            let missing = (i < 2 && !has_top) || (i >= h + 2 && !has_bottom);
            (vlo[i & m], vhi[i & m]) = if missing { (allv, allv) } else { (cl, ch) };
        }
        for i in 0..n_src {
            (vsrcs_a[i & m], vsrcs_b[i & m]) = pack16::<W>(&vlo, &vhi, i, h);
        }
    }

    // ---- 2) tap listesi (dav1d `src/cdef_tmpl.c:121-155`) -----------------
    let (sel_a, sel_b) = if W == 8 {
        (&SEL16_8A, &SEL16_8B)
    } else {
        (&SEL16_4A, &SEL16_4B)
    };
    let sel_az: [__m512i; 5] = core::array::from_fn(|i| yay(&sel_a[i]));
    let sel_bz: [__m512i; 5] = core::array::from_fn(|i| yay(&sel_b[i]));
    let bd_min8 = params.bit_depth.min_8() as i32;
    let pri = i32::from(params.pri_strength);
    let sec = i32::from(params.sec_strength);
    let (pt, st) = TAPS16[usize::from(params.dir) & 7].split_at(4);
    let k0 = 4 - ((pri >> bd_min8) & 1);
    let pk = [
        _mm512_set1_epi16(k0 as i16),
        _mm512_set1_epi16(((k0 & 3) | 2) as i16),
    ];
    let pri_thr = _mm512_set1_epi16(pri as i16);
    let sec_thr = _mm512_set1_epi16(sec as i16);
    let pri_cnt = _mm_cvtsi32_si128(if pri != 0 {
        (i32::from(params.damping) - ilog2(pri)).clamp(0, 31)
    } else {
        0
    });
    let sec_cnt = _mm_cvtsi32_si128(if sec != 0 {
        (i32::from(params.damping) - ilog2(sec)).clamp(0, 31)
    } else {
        0
    });
    // `(x + 8) >> 4` = `pmulhrsw(x, 2048)`
    let r2048 = _mm512_set1_epi16(2048);

    // ---- 3) satır grupları (W = 8: 4 satır; W = 4: 8 satır) ---------------
    let group = if W == 8 { 4 } else { 8 };
    let mut y = 0usize;
    while y < h {
        let px = _mm512_or_si512(
            _mm512_shuffle_epi8(srcs_a[(y + 2) & m], sel_az[2]),
            _mm512_shuffle_epi8(srcs_b[(y + 2) & m], sel_bz[2]),
        );
        let mut minv = px;
        let mut maxv = px;
        let mut cons = |t: &Tap, thr: __m512i, cnt: __m128i| -> __m512i {
            let i = (y + t.row) & m;
            let bi = t.bi.min(4);
            let mut p = _mm512_or_si512(
                _mm512_shuffle_epi8(srcs_a[i], sel_az[bi]),
                _mm512_shuffle_epi8(srcs_b[i], sel_bz[bi]),
            );
            if EDGES {
                let inv = _mm512_or_si512(
                    _mm512_shuffle_epi8(vsrcs_a[i], sel_az[bi]),
                    _mm512_shuffle_epi8(vsrcs_b[i], sel_bz[bi]),
                );
                p = _mm512_mask_blend_epi16(_mm512_movepi16_mask(inv), p, px);
            }
            let d = _mm512_sub_epi16(p, px);
            let ad = _mm512_abs_epi16(d);
            let c = _mm512_min_epu16(ad, _mm512_subs_epu16(thr, _mm512_srl_epi16(ad, cnt)));
            if MINMAX {
                minv = _mm512_min_epi16(p, minv);
                maxv = _mm512_max_epi16(p, maxv);
            }
            isaret(c, d)
        };
        let mut sum = z;
        if pri != 0 {
            for j in 0..2usize {
                let c = _mm512_add_epi16(
                    cons(&pt[2 * j], pri_thr, pri_cnt),
                    cons(&pt[2 * j + 1], pri_thr, pri_cnt),
                );
                sum = _mm512_add_epi16(sum, _mm512_mullo_epi16(c, pk[j]));
            }
        }
        if sec != 0 {
            for j in 0..2usize {
                let c = _mm512_add_epi16(
                    _mm512_add_epi16(
                        cons(&st[4 * j], sec_thr, sec_cnt),
                        cons(&st[4 * j + 1], sec_thr, sec_cnt),
                    ),
                    _mm512_add_epi16(
                        cons(&st[4 * j + 2], sec_thr, sec_cnt),
                        cons(&st[4 * j + 3], sec_thr, sec_cnt),
                    ),
                );
                let c = if j == 0 { _mm512_slli_epi16::<1>(c) } else { c };
                sum = _mm512_add_epi16(sum, c);
            }
        }
        let r = _mm512_add_epi16(sum, _mm512_srai_epi16::<15>(sum));
        let mut r = _mm512_add_epi16(_mm512_mulhrs_epi16(r, r2048), px);
        if MINMAX {
            r = _mm512_min_epi16(_mm512_max_epi16(r, minv), maxv);
        }
        let seritler = [
            _mm512_extracti32x4_epi32::<0>(r),
            _mm512_extracti32x4_epi32::<1>(r),
            _mm512_extracti32x4_epi32::<2>(r),
            _mm512_extracti32x4_epi32::<3>(r),
        ];
        if W == 8 {
            // şerit k = satır y+k; maske yalnız w = 8 lane'i açar.
            for (k, v) in seritler.into_iter().enumerate() {
                store_u16_k::<20>(chunk_mut(dst, (y + k) * stride), v, 0xff);
            }
        } else {
            // şerit k = satırlar y+2k (lane 0..3), y+2k+1 (lane 4..7).
            for (k, v) in seritler.into_iter().enumerate().take((h - y) / 2) {
                store_u16_k::<12>(chunk_mut(dst, (y + 2 * k) * stride), v, 0x0f);
                store_u16_k::<12>(
                    chunk_mut(dst, (y + 2 * k + 1) * stride),
                    _mm_srli_si128::<8>(v),
                    0x0f,
                );
            }
        }
        y += group;
    }
}
