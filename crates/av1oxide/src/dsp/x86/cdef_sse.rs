//! CDEF SSSE3 ve SSE4.1 çekirdekleri (128 bit) — F4-CDEF c (P30).
//!
//! Yöntem AVX2 gövdeleriyle ([`super::cdef`]) aynıdır: dolgu tamponu yok,
//! kaynak satırları dilimlerden doğrudan okunur, bir tapın komşuları satır
//! yazmacından tek `pshufb` ile alınır (8 bpc'de seçme **ve** `u8 → i16`
//! genişletme), eksik kenar komşusu = merkez piksel (maske), dört kenarlı yol
//! maskesiz. Fark yazmaç genişliğidir: 128 bit yazmaçta `W = 8` **bir** satır,
//! `W = 4` **iki** satır taşır (AVX2'de iki / dört). Tap seçme maskeleri AVX2
//! tablolarının alt yarısıdır (her 128 bit yarı aynı).
//!
//! ## SSSE3 ↔ SSE4.1
//!
//! Gövdeler bir kez yazılır (`dir_govdesi!`, `filter8_govdesi!`,
//! `filter16_govdesi!`) ve iki modülde, her biri kendi `#[target_feature]`'ı
//! ve düzeye özgü yardımcılarıyla açılır; makine kodu düzey başına ayrıdır:
//!
//! | İşlem | SSSE3 (`ssse3`) | SSE4.1 (`sse41`) |
//! |---|---|---|
//! | maskeli seçim | `pand` + `pandn` + `por` | aynı (`pblendvb` 8×8'de yavaş; aşağıda) |
//! | `u8 → i16` | `punpcklbw` (sıfırla) | `pmovzxbw` |
//! | 32 bit ekleme | `movd` + `pslldq` + `por` | `pinsrd` |
//! | 32 bit çarpım (`dir` maliyeti) | `pmuludq` ×2 + karıştırma | `pmulld` |
//! | en büyük maliyet | `pcmpgtd` + seçim | `pmaxud` |
//! | `constrain` alt sınırı | `pminsw` | `pminuw` |
//!
//! `pminsw` ile `pminuw` burada aynı sonucu verir: iki işlenen de
//! `[0, 2^15)` içindedir (`|d| ≤ 4095`, eşik ≤ 240). İşaretli 32 bit en büyük
//! seçimi de işaretsizle aynıdır: her yön maliyeti `< 2^31`'dir (en büyük
//! kısmi toplam `|Σ| ≤ 1024`; köşegen maliyeti `≤ 28 · 27 525 120 + 105 ·
//! 2^20 ≈ 881·10^6`, alt yönler ve satır/sütun da `≤ 881·10^6`).
//!
//! dav1d SSE4.1'de 16 bpc `fb` gövdesi yoktur (`src/x86/cdef.h`: 16 bpc SSE4.1
//! yalnız `dir`); 16 bpc `fb` SSE4.1 düzeyinde SSSE3 gövdesiyle kalır.
//!
//! ## Yetki ve `unsafe` (ADR-0003 §2.3-2.4)
//!
//! Gövdeler güvenli `#[target_feature]` işlevleridir; yuva işaretçileri yalnız
//! [`Ssse3Token`] / [`Sse41Token`] ile verilir. `unsafe` yalnız yedi girişteki
//! tek çağrıdır; dosya `audit::UNSAFE_EXEMPTIONS` defterindedir.
//!
//! ```compile_fail,E0061
//! // Belirteçsiz yuva işaretçisi alınamaz.
//! let _ = av1oxide::dsp::x86::cdef_sse::filter_ssse3_fn();
//! ```

#![allow(clippy::needless_range_loop)]
#![allow(unsafe_code)]

use super::cdef::{DIR_AB, MAX_ROWS, N16, TAPS16, Tap, ilog2, sel_table, sel16_table};
use super::mem::{chunk, chunk_mut, load_u8x16, store_u8x4, store_u8x8, store_u8x16};
use crate::cpu::{Sse41Token, Ssse3Token};
use crate::dsp::cdef::{
    CdefBitDepth, CdefDirFn, CdefDirResult, CdefDsp, CdefEdgeFlags, CdefFilterFn, CdefFilterParams,
    DIR_CENTER_WEIGHT, DIV_TABLE,
};
use core::arch::x86_64::{
    __m128i, _mm_abs_epi16, _mm_add_epi16, _mm_add_epi32, _mm_and_si128, _mm_andnot_si128,
    _mm_bslli_si128, _mm_bsrli_si128, _mm_castsi128_ps, _mm_cmpeq_epi32, _mm_cmpgt_epi16,
    _mm_cmpgt_epi32, _mm_cvtepu8_epi16, _mm_cvtsi32_si128, _mm_cvtsi64_si128, _mm_hadd_epi16,
    _mm_hadd_epi32, _mm_insert_epi16, _mm_insert_epi32, _mm_madd_epi16, _mm_max_epi16,
    _mm_max_epu32, _mm_min_epi16, _mm_min_epu16, _mm_movemask_ps, _mm_mul_epu32, _mm_mulhrs_epi16,
    _mm_mullo_epi16, _mm_mullo_epi32, _mm_or_si128, _mm_set1_epi16, _mm_set1_epi32, _mm_setr_epi8,
    _mm_setr_epi32, _mm_setzero_si128, _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_sign_epi16,
    _mm_slli_epi16, _mm_slli_si128, _mm_srai_epi16, _mm_srl_epi16, _mm_srli_epi16, _mm_srli_epi64,
    _mm_srli_si128, _mm_sub_epi16, _mm_subs_epu16, _mm_unpackhi_epi16, _mm_unpacklo_epi8,
    _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64,
};

/// `s[off..off+N]` küçük sonlu tamsayısı (denetimli dizi referansıyla).
macro_rules! le {
    ($t:ty, $n:literal, $s:expr, $off:expr) => {
        <$t>::from_le_bytes(*chunk::<u8, $n>($s, $off))
    };
}

/// 256 bit seçme tablosunun alt 128 bit yarısı (bayt dizisi; iki yarı aynı).
const fn alt_yari<const N: usize>(t: [[i16; 16]; N]) -> [[u8; 16]; N] {
    let mut o = [[0u8; 16]; N];
    let mut i = 0;
    while i < N {
        let mut x = 0;
        while x < 8 {
            let v = t[i][x] as u16;
            o[i][2 * x] = v as u8;
            o[i][2 * x + 1] = (v >> 8) as u8;
            x += 1;
        }
        i += 1;
    }
    o
}

/// 8 bpc seçme maskeleri ([`sel_table`]); `V*` geçerlilik maskeleri.
pub(super) const SEL8: [[u8; 16]; 5] = alt_yari(sel_table(8, false));
pub(super) const VSEL8: [[u8; 16]; 5] = alt_yari(sel_table(8, true));
pub(super) const SEL4: [[u8; 16]; 5] = alt_yari(sel_table(4, false));
pub(super) const VSEL4: [[u8; 16]; 5] = alt_yari(sel_table(4, true));
/// 16 bpc seçme maskeleri ([`sel16_table`]): `A` = birinci yazmaç, `B` = ikinci.
pub(super) const SEL16_8A: [[u8; 16]; 8] = alt_yari(sel16_table(8, 0));
pub(super) const SEL16_8B: [[u8; 16]; 8] = alt_yari(sel16_table(8, 1));
pub(super) const SEL16_4A: [[u8; 16]; 8] = alt_yari(sel16_table(4, 0));
pub(super) const SEL16_4B: [[u8; 16]; 8] = alt_yari(sel16_table(4, 1));

// ---------------------------------------------------------------------------
// Belirteçli yuva işaretçileri
// ---------------------------------------------------------------------------

/// 8 bpc `dir` (SSSE3): yalnız [`Ssse3Token`] ile.
#[must_use]
pub const fn find_dir_ssse3_fn(_t: Ssse3Token) -> CdefDirFn {
    find_dir_ssse3_entry
}

/// 16 bpc `dir` (SSSE3): yalnız [`Ssse3Token`] ile.
#[must_use]
pub const fn find_dir16_ssse3_fn(_t: Ssse3Token) -> CdefDirFn {
    find_dir16_ssse3_entry
}

/// 8 bpc `fb` (SSSE3): yalnız [`Ssse3Token`] ile.
#[must_use]
pub const fn filter_ssse3_fn(_t: Ssse3Token) -> CdefFilterFn {
    filter_ssse3_entry
}

/// 16 bpc `fb` (SSSE3): yalnız [`Ssse3Token`] ile.
#[must_use]
pub const fn filter16_ssse3_fn(_t: Ssse3Token) -> CdefFilterFn {
    filter16_ssse3_entry
}

/// 8 bpc `dir` (SSE4.1): yalnız [`Sse41Token`] ile.
#[must_use]
pub const fn find_dir_sse41_fn(_t: Sse41Token) -> CdefDirFn {
    find_dir_sse41_entry
}

/// 16 bpc `dir` (SSE4.1): yalnız [`Sse41Token`] ile.
#[must_use]
pub const fn find_dir16_sse41_fn(_t: Sse41Token) -> CdefDirFn {
    find_dir16_sse41_entry
}

/// 8 bpc `fb` (SSE4.1): yalnız [`Sse41Token`] ile.
#[must_use]
pub const fn filter_sse41_fn(_t: Sse41Token) -> CdefFilterFn {
    filter_sse41_entry
}

// ---------------------------------------------------------------------------
// Sözleşme denetimleri (AVX2 girişleriyle aynı okuma sınırları)
// ---------------------------------------------------------------------------

/// `dir` sözleşmesi: bit derinliği sınıfı ve `img` boyu (8 bpc: en büyük okuma
/// `img[7*stride .. 7*stride + 8]`, 16 bpc: `.. + 16`).
fn dir_ok(img: &[u8], stride: usize, bit_depth: CdefBitDepth, hbd: bool) -> bool {
    (bit_depth != CdefBitDepth::B8) == hbd && img.len() >= 7 * stride + if hbd { 16 } else { 8 }
}

/// `fb` sözleşmesi: yalnız **okuma** sınırları (`bpp` = 1 ya da 2). Eksik
/// kenarın dilimi okunmaz; varsa tam satır aralığı gerekir.
pub(super) fn filter_ok(
    stride: usize,
    dst: &[u8],
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    p: &CdefFilterParams,
    hbd: bool,
) -> bool {
    let (w, h, e) = (p.size.w(), p.size.h(), p.edges);
    let bpp = if hbd { 2 } else { 1 };
    let damping_max = 6 + p.bit_depth.min_8() as u8;
    (p.bit_depth != CdefBitDepth::B8) == hbd
        && p.damping <= damping_max
        && p.dir < 8
        && stride >= bpp * (w + 4)
        && dst.len() >= (h - 1) * stride + bpp * (w + 2)
        && (!e.contains(CdefEdgeFlags::HAVE_LEFT) || left.len() >= h * 2 * bpp)
        && (!e.contains(CdefEdgeFlags::HAVE_TOP) || top.len() >= stride + bpp * (w + 4))
        && (!e.contains(CdefEdgeFlags::HAVE_BOTTOM) || bottom.len() >= stride + bpp * (w + 4))
}

/// Sözleşme dışı `fb` girdisi: o bit derinliğinin skaler monomorfu.
pub(super) fn filter_skaler(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    let fb = CdefDsp::scalar_for(params.bit_depth).fb[params.size.index()];
    fb(dst, stride, left, top, bottom, params);
}

// ---------------------------------------------------------------------------
// Girişler (belirteçli işaretçilerin hedefi; tek `unsafe` çağrı)
// ---------------------------------------------------------------------------

/// 8 bpc `dir` (SSSE3) girişi.
fn find_dir_ssse3_entry(img: &[u8], stride: usize, bit_depth: CdefBitDepth) -> CdefDirResult {
    if !dir_ok(img, stride, bit_depth, false) {
        return (CdefDsp::scalar_for(bit_depth).dir)(img, stride, bit_depth);
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`find_dir_ssse3_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır.
    // (2) sınır: `dir_ok` en büyük okumayı (`img[7*stride .. 7*stride + 8]`)
    // kapsar; gövdenin yüklemeleri ayrıca denetimli dizi referanslarıdır.
    // (3) hizalama/takma ad: gövde `img`'yi yalnız okur, yerel dizilere yazar.
    unsafe { ssse3::find_dir(img, stride) }
}

/// 16 bpc `dir` (SSSE3) girişi.
fn find_dir16_ssse3_entry(img: &[u8], stride: usize, bit_depth: CdefBitDepth) -> CdefDirResult {
    if !dir_ok(img, stride, bit_depth, true) {
        return (CdefDsp::scalar_for(bit_depth).dir)(img, stride, bit_depth);
    }
    // SAFETY: (1) ISA: adres yalnız `Ssse3Token` ile verilir
    // (`find_dir16_ssse3_fn`). (2) sınır: `dir_ok` en büyük okumayı
    // (`img[7*stride .. 7*stride + 16]`) kapsar. (3) hizalama/takma ad: gövde
    // `img`'yi yalnız okur.
    unsafe { ssse3::find_dir16(img, stride, bit_depth.min_8() as i32) }
}

/// 8 bpc `dir` (SSE4.1) girişi.
fn find_dir_sse41_entry(img: &[u8], stride: usize, bit_depth: CdefBitDepth) -> CdefDirResult {
    if !dir_ok(img, stride, bit_depth, false) {
        return (CdefDsp::scalar_for(bit_depth).dir)(img, stride, bit_depth);
    }
    // SAFETY: (1) ISA: adres yalnız `Sse41Token` ile verilir
    // (`find_dir_sse41_fn`); belirteç SSE4.1 + SSSE3 tespiti ve maske iznidir.
    // (2) sınır: `dir_ok` en büyük okumayı kapsar. (3) hizalama/takma ad:
    // gövde `img`'yi yalnız okur.
    unsafe { sse41::find_dir(img, stride) }
}

/// 16 bpc `dir` (SSE4.1) girişi.
fn find_dir16_sse41_entry(img: &[u8], stride: usize, bit_depth: CdefBitDepth) -> CdefDirResult {
    if !dir_ok(img, stride, bit_depth, true) {
        return (CdefDsp::scalar_for(bit_depth).dir)(img, stride, bit_depth);
    }
    // SAFETY: (1) ISA: adres yalnız `Sse41Token` ile verilir
    // (`find_dir16_sse41_fn`). (2) sınır: `dir_ok` en büyük okumayı kapsar.
    // (3) hizalama/takma ad: gövde `img`'yi yalnız okur.
    unsafe { sse41::find_dir16(img, stride, bit_depth.min_8() as i32) }
}

/// 8 bpc `fb` (SSSE3) girişi.
fn filter_ssse3_entry(
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
    // SAFETY: (1) ISA: adres yalnız `Ssse3Token` ile verilir
    // (`filter_ssse3_fn`). (2) sınır: `filter_ok` gövdenin okuduğu her baytı
    // kapsar (dst satırları sütun `0..w+2`, bayrağı olan kenarda top/bottom
    // `row*stride + w + 4`, left `2h`); yüklemeler denetimli dizi
    // referanslarıdır. (3) hizalama/takma ad: `dst` tek `&mut`, kaynaklar `&`.
    unsafe { ssse3::filter(dst, stride, left, top, bottom, params) }
}

/// 16 bpc `fb` (SSSE3) girişi.
fn filter16_ssse3_entry(
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
    // SAFETY: (1) ISA: adres yalnız `Ssse3Token` ile verilir
    // (`filter16_ssse3_fn`). (2) sınır: `filter_ok` (`bpp = 2`) gövdenin
    // okuduğu her baytı kapsar; yüklemeler denetimli dizi referanslarıdır.
    // (3) hizalama/takma ad: `dst` tek `&mut`, kaynaklar `&`.
    unsafe { ssse3::filter16(dst, stride, left, top, bottom, params) }
}

/// 8 bpc `fb` (SSE4.1) girişi.
fn filter_sse41_entry(
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
    // SAFETY: (1) ISA: adres yalnız `Sse41Token` ile verilir
    // (`filter_sse41_fn`). (2) sınır: `filter_ok` gövdenin okuduğu her baytı
    // kapsar. (3) hizalama/takma ad: `dst` tek `&mut`, kaynaklar `&`.
    unsafe { sse41::filter(dst, stride, left, top, bottom, params) }
}

// ---------------------------------------------------------------------------
// Gövde makroları (düzey modüllerinde açılır)
// ---------------------------------------------------------------------------

/// `dir` (8 ve 16 bpc): satırlar `i16` lane'lerde `px − 128`
/// (16 bpc: `(px >> bitdepth_min_8) − 128`), maliyet dav1d
/// `src/cdef_tmpl.c:247-303` ile bit-exact. Kısmi toplamlar 16 lane'lik
/// (alt `l` = indeks 0..7, üst `h` = 8..15) yazmaç çiftlerinde bayt
/// kaydırmalarıyla birikir.
macro_rules! dir_govdesi {
    ($tf:literal) => {
        /// 8 satırdan yön sonucu.
        #[target_feature(enable = $tf)]
        fn dir_maliyet(r: [__m128i; 8]) -> CdefDirResult {
            let z = _mm_setzero_si128();
            // lane x → 7 − x
            let rev = _mm_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
            let (mut d0l, mut d0h, mut d1l, mut d1h) = (z, z, z, z);
            let (mut a0l, mut a0h, mut a1l, mut a1h) = (z, z, z, z);
            macro_rules! ekle {
                ($l:ident, $h:ident, $v:expr, $b:literal, $rr:literal) => {{
                    let v = $v;
                    $l = _mm_add_epi16($l, _mm_bslli_si128::<$b>(v));
                    $h = _mm_add_epi16($h, _mm_bsrli_si128::<$rr>(v));
                }};
            }
            macro_rules! satir {
                ($y:literal, $b:literal, $rr:literal) => {{
                    let x = r[$y];
                    let xr = _mm_shuffle_epi8(x, rev);
                    // diag[0][y + x], diag[1][7 + y − x] (ters satır)
                    ekle!(d0l, d0h, x, $b, $rr);
                    ekle!(d1l, d1h, xr, $b, $rr);
                    // alt[0][y + (x >> 1)], alt[1][3 + y − (x >> 1)]
                    ekle!(a0l, a0h, _mm_hadd_epi16(x, z), $b, $rr);
                    ekle!(a1l, a1h, _mm_hadd_epi16(xr, z), $b, $rr);
                }};
            }
            satir!(0, 0, 16);
            satir!(1, 2, 14);
            satir!(2, 4, 12);
            satir!(3, 6, 10);
            satir!(4, 8, 8);
            satir!(5, 10, 6);
            satir!(6, 12, 4);
            satir!(7, 14, 2);
            // alt[2][3 − (y >> 1) + x], alt[3][(y >> 1) + x]: s_m = r_2m + r_2m+1
            let s0 = _mm_add_epi16(r[0], r[1]);
            let s1 = _mm_add_epi16(r[2], r[3]);
            let s2 = _mm_add_epi16(r[4], r[5]);
            let s3 = _mm_add_epi16(r[6], r[7]);
            let (mut a2l, mut a2h, mut a3l, mut a3h) = (z, z, z, z);
            ekle!(a2l, a2h, s0, 6, 10);
            ekle!(a2l, a2h, s1, 4, 12);
            ekle!(a2l, a2h, s2, 2, 14);
            ekle!(a2l, a2h, s3, 0, 16);
            ekle!(a3l, a3h, s0, 0, 16);
            ekle!(a3l, a3h, s1, 2, 14);
            ekle!(a3l, a3h, s2, 4, 12);
            ekle!(a3l, a3h, s3, 6, 10);
            // hv: satır toplamları (hadd ağacı) ve sütun toplamları
            let hv0 = _mm_hadd_epi16(
                _mm_hadd_epi16(_mm_hadd_epi16(r[0], r[1]), _mm_hadd_epi16(r[2], r[3])),
                _mm_hadd_epi16(_mm_hadd_epi16(r[4], r[5]), _mm_hadd_epi16(r[6], r[7])),
            );
            let hv1 = _mm_add_epi16(_mm_add_epi16(s0, s1), _mm_add_epi16(s2, s3));

            // ---- maliyetler (dav1d `src/cdef_tmpl.c:263-292`) ---------------
            // köşegen: (d[n]² + d[14−n]²)·div[n] (n < 7), d[7]²·105 — üst
            // yazmacın lane n ← lane 6 − n, son lane 0. alt: (a[n]² +
            // a[10−n]²)·div[2n+1] (n < 3), a[n]²·105 — üst lane n ← 2 − n.
            let rev7 = _mm_setr_epi8(12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, -128, -128);
            let rev3 = _mm_setr_epi8(
                4, 5, 2, 3, 0, 1, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
            );
            let d = |i: usize| DIV_TABLE[i] as i32;
            let c = DIR_CENTER_WEIGHT as i32;
            let w_d_lo = _mm_setr_epi32(d(0), d(1), d(2), d(3));
            let w_d_hi = _mm_setr_epi32(d(4), d(5), d(6), c);
            let w_a_lo = _mm_setr_epi32(d(1), d(3), d(5), c);
            let w_c = _mm_set1_epi32(c);
            let terim = |lo: __m128i, hi: __m128i, wl: __m128i, wh: __m128i| {
                let a = _mm_unpacklo_epi16(lo, hi);
                let b = _mm_unpackhi_epi16(lo, hi);
                _mm_add_epi32(
                    mul32(_mm_madd_epi16(a, a), wl),
                    mul32(_mm_madd_epi16(b, b), wh),
                )
            };
            let c0 = terim(d0l, _mm_shuffle_epi8(d0h, rev7), w_d_lo, w_d_hi);
            let c4 = terim(d1l, _mm_shuffle_epi8(d1h, rev7), w_d_lo, w_d_hi);
            let c1 = terim(a0l, _mm_shuffle_epi8(a0h, rev3), w_a_lo, w_c);
            let c3 = terim(a1l, _mm_shuffle_epi8(a1h, rev3), w_a_lo, w_c);
            let c5 = terim(a2l, _mm_shuffle_epi8(a2h, rev3), w_a_lo, w_c);
            let c7 = terim(a3l, _mm_shuffle_epi8(a3h, rev3), w_a_lo, w_c);
            let c2 = mul32(_mm_madd_epi16(hv0, hv0), w_c);
            let c6 = mul32(_mm_madd_epi16(hv1, hv1), w_c);
            let cost_lo = _mm_hadd_epi32(_mm_hadd_epi32(c0, c1), _mm_hadd_epi32(c2, c3));
            let cost_hi = _mm_hadd_epi32(_mm_hadd_epi32(c4, c5), _mm_hadd_epi32(c6, c7));

            // En iyi yön dalsız (dav1d `src/cdef_tmpl.c:294-301`, katı `>`): en
            // büyük maliyet, ona eşit **ilk** lane.
            let m = max32(cost_lo, cost_hi);
            let m = max32(m, _mm_shuffle_epi32::<0x4e>(m));
            let m = max32(m, _mm_shuffle_epi32::<0xb1>(m));
            let maske = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(cost_lo, m)))
                | (_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(cost_hi, m))) << 4);
            let best = (maske as u32).trailing_zeros() as usize & 7;
            let mut b = [0u8; 32];
            store_u8x16(chunk_mut(&mut b, 0), cost_lo);
            store_u8x16(chunk_mut(&mut b, 16), cost_hi);
            let cost = |i: usize| u32::from_le_bytes(*chunk::<u8, 4>(&b, 4 * i));
            CdefDirResult {
                dir: best as u8,
                var: cost(best).wrapping_sub(cost(best ^ 4)) >> 10,
            }
        }

        /// 8 bpc yön taraması — dav1d `cdef_find_dir_c` ile bit-exact.
        #[target_feature(enable = $tf)]
        pub(super) fn find_dir(img: &[u8], stride: usize) -> CdefDirResult {
            let k = _mm_set1_epi16(128);
            let satir = |y: usize| {
                _mm_sub_epi16(u8_i16(_mm_cvtsi64_si128(le!(i64, 8, img, y * stride))), k)
            };
            dir_maliyet([
                satir(0),
                satir(1),
                satir(2),
                satir(3),
                satir(4),
                satir(5),
                satir(6),
                satir(7),
            ])
        }

        /// 16 bpc yön taraması: `(px >> shift) − 128`, sonra 8 bpc maliyeti.
        #[target_feature(enable = $tf)]
        pub(super) fn find_dir16(img: &[u8], stride: usize, shift: i32) -> CdefDirResult {
            let sh = _mm_cvtsi32_si128(shift);
            let k = _mm_set1_epi16(128);
            let satir =
                |y: usize| _mm_sub_epi16(_mm_srl_epi16(load_u8x16(chunk(img, y * stride)), sh), k);
            dir_maliyet([
                satir(0),
                satir(1),
                satir(2),
                satir(3),
                satir(4),
                satir(5),
                satir(6),
                satir(7),
            ])
        }
    };
}

/// 8 bpc `fb`: `W = 8` yazmaç = bir satır; `W = 4` yazmaç = iki satır
/// (lane 0..3 satır `y`, 4..7 satır `y+1`).
macro_rules! filter8_govdesi {
    ($tf:literal) => {
        /// Kenar satırı (`top`/`bottom`, bayt 0 = sütun −2): sütun `−2 .. w+2`.
        #[target_feature(enable = $tf)]
        #[inline]
        fn edge_row<const W: usize>(s: &[u8], o: usize) -> __m128i {
            let v = _mm_cvtsi64_si128(le!(i64, 8, s, o));
            if W == 8 {
                ek32_2(v, le!(i32, 4, s, o + 8))
            } else {
                v
            }
        }

        /// Blok satırı: sütun −2, −1 `left`'ten (`l`), `0 .. w+2` `dst`'den.
        #[target_feature(enable = $tf)]
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

        /// 8 bpc CDEF filtresi — dav1d `cdef_filter_block_c` ile bit-exact.
        #[target_feature(enable = $tf)]
        pub(super) fn filter(
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
                return; // sözleşme dışı; skaler yolla aynı karar
            }
            let all = params.edges.bits() == 0b1111;
            match (params.size.w(), all, pri != 0 && sec != 0) {
                (8, true, true) => {
                    filter_w::<8, false, true>(dst, stride, left, top, bottom, params)
                }
                (8, true, false) => {
                    filter_w::<8, false, false>(dst, stride, left, top, bottom, params)
                }
                (8, false, true) => {
                    filter_w::<8, true, true>(dst, stride, left, top, bottom, params)
                }
                (8, false, false) => {
                    filter_w::<8, true, false>(dst, stride, left, top, bottom, params)
                }
                (_, true, true) => {
                    filter_w::<4, false, true>(dst, stride, left, top, bottom, params)
                }
                (_, true, false) => {
                    filter_w::<4, false, false>(dst, stride, left, top, bottom, params)
                }
                (_, false, true) => {
                    filter_w::<4, true, true>(dst, stride, left, top, bottom, params)
                }
                (_, false, false) => {
                    filter_w::<4, true, false>(dst, stride, left, top, bottom, params)
                }
            }
        }

        #[target_feature(enable = $tf)]
        fn filter_w<const W: usize, const EDGES: bool, const MINMAX: bool>(
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

            // ---- 1) kaynak satırları (rows[i] = mantıksal satır i − 2) ------
            let z = _mm_setzero_si128();
            let mut rows = [z; MAX_ROWS];
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
            // W = 8 → srcs[i] = rows[i]; W = 4 → srcs[i] = [rows[i] ‖ rows[i+1]]
            let n_src = if W == 8 { n_rows } else { n_rows - 1 };
            let pack = |rs: &[__m128i; MAX_ROWS], i: usize| -> __m128i {
                if W == 8 {
                    rs[i]
                } else {
                    _mm_unpacklo_epi64(rs[i], rs[i + 1])
                }
            };
            let mut srcs = [z; MAX_ROWS];
            for (i, v) in srcs.iter_mut().enumerate().take(n_src) {
                *v = pack(&rows, i);
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
                let mut vrows = [z; MAX_ROWS];
                for (i, v) in vrows.iter_mut().enumerate().take(n_rows) {
                    let missing = (i < 2 && !has_top) || (i >= h + 2 && !has_bottom);
                    *v = if missing { allv } else { colv };
                }
                for (i, v) in vsrcs.iter_mut().enumerate().take(n_src) {
                    *v = pack(&vrows, i);
                }
            }

            // ---- 2) tap listesi (dav1d `src/cdef_tmpl.c:121-155`) -----------
            let (sel, vsel) = if W == 8 {
                (&SEL8, &VSEL8)
            } else {
                (&SEL4, &VSEL4)
            };
            let pri = i32::from(params.pri_strength);
            let sec = i32::from(params.sec_strength);
            let dir = usize::from(params.dir);
            let mut pt = [Tap::default(); 4];
            let mut st = [Tap::default(); 8];
            let (mut np, mut ns) = (0usize, 0usize);
            let tap = |a: i32, b: i32, k: i32| Tap {
                row: (a + 2) as usize,
                bi: (b + 2) as usize,
                k: k as i16,
            };
            if pri != 0 {
                let mut k = 4 - (params.pri_strength & 1) as i32;
                for j in 0..2usize {
                    let (a, b) = DIR_AB[dir + 2][j];
                    pt[np] = tap(a, b, k);
                    pt[np + 1] = tap(-a, -b, k);
                    np += 2;
                    k = (k & 3) | 2;
                }
            }
            if sec != 0 {
                for j in 0..2usize {
                    let k = 2 - j as i32;
                    let (a2, b2) = DIR_AB[dir + 4][j];
                    let (a3, b3) = DIR_AB[dir][j];
                    st[ns] = tap(a2, b2, k);
                    st[ns + 1] = tap(-a2, -b2, k);
                    st[ns + 2] = tap(a3, b3, k);
                    st[ns + 3] = tap(-a3, -b3, k);
                    ns += 4;
                }
            }
            let pri_thr = _mm_set1_epi16(pri as i16);
            let sec_thr = _mm_set1_epi16(sec as i16);
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
            let center = load_u8x16(&sel[2]);
            let pick = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
            let eight = _mm_set1_epi16(8);

            // ---- 3) satır grupları (W = 8: 1 satır; W = 4: 2 satır) ---------
            let group = if W == 8 { 1 } else { 2 };
            let mut y = 0usize;
            while y < h {
                let px = _mm_shuffle_epi8(srcs[y + 2], center);
                let mut sum = z;
                let mut minv = px;
                let mut maxv = px;
                let mut step = |t: &Tap, thr: __m128i, cnt: __m128i| {
                    let i = y + t.row;
                    let mut p = _mm_shuffle_epi8(srcs[i], load_u8x16(&sel[t.bi]));
                    if EDGES {
                        let inv = _mm_shuffle_epi8(vsrcs[i], load_u8x16(&vsel[t.bi]));
                        p = blend(p, px, inv);
                    }
                    let d = _mm_sub_epi16(p, px);
                    let ad = _mm_abs_epi16(d);
                    let c = min16(ad, _mm_subs_epu16(thr, _mm_srl_epi16(ad, cnt)));
                    let cs = _mm_sign_epi16(c, d);
                    sum = _mm_add_epi16(sum, _mm_mullo_epi16(cs, _mm_set1_epi16(t.k)));
                    if MINMAX {
                        minv = _mm_min_epi16(p, minv);
                        maxv = _mm_max_epi16(p, maxv);
                    }
                };
                for t in &pt[..np] {
                    step(t, pri_thr, pri_cnt);
                }
                for t in &st[..ns] {
                    step(t, sec_thr, sec_cnt);
                }
                // `(sum - (sum < 0) + 8) >> 4` (dav1d `src/cdef_tmpl.c:165,184,210`)
                let neg = _mm_cmpgt_epi16(z, sum);
                let r = _mm_sub_epi16(sum, _mm_srli_epi16::<15>(neg));
                let r = _mm_srai_epi16::<4>(_mm_add_epi16(r, eight));
                let mut r = _mm_add_epi16(r, px);
                if MINMAX {
                    r = _mm_min_epi16(_mm_max_epi16(r, minv), maxv);
                }
                // i16 → u8: alt bayt (C `pixel` dönüşümü, mod 256).
                let packed = _mm_shuffle_epi8(r, pick);
                if W == 8 {
                    store_u8x8(chunk_mut(dst, y * stride), packed);
                } else {
                    store_u8x4(chunk_mut(dst, y * stride), packed);
                    store_u8x4(
                        chunk_mut(dst, (y + 1) * stride),
                        _mm_srli_si128::<4>(packed),
                    );
                }
                y += group;
            }
        }
    };
}

/// 16 bpc `fb`: `W = 8` yazmaç çifti = bir satır (`lo` sütun −2..5, `hi`
/// 6..9); `W = 4` = iki satır (birinci yazmaç satır `y`, ikinci `y+1`).
macro_rules! filter16_govdesi {
    ($tf:literal) => {
        /// 16 bpc kenar satırı: `(lo, hi)`; `W = 4` iken `hi` = 0.
        #[target_feature(enable = $tf)]
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
        #[target_feature(enable = $tf)]
        #[inline]
        fn block_row16<const W: usize>(dst: &[u8], o: usize, l: i32) -> (__m128i, __m128i) {
            if W == 8 {
                let v = _mm_slli_si128::<4>(load_u8x16(chunk(dst, o)));
                (
                    _mm_or_si128(v, _mm_cvtsi32_si128(l)),
                    _mm_cvtsi64_si128(le!(i64, 8, dst, o + 12)),
                )
            } else {
                let v = _mm_slli_si128::<4>(_mm_cvtsi64_si128(le!(i64, 8, dst, o)));
                let v = _mm_or_si128(v, _mm_cvtsi32_si128(l));
                (ek32_3(v, le!(i32, 4, dst, o + 8)), _mm_setzero_si128())
            }
        }

        /// 16 bpc CDEF filtresi — dav1d `cdef_filter_block_c` ile bit-exact.
        #[target_feature(enable = $tf)]
        pub(super) fn filter16(
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
                (8, true, true) => {
                    filter16_w::<8, false, true>(dst, stride, left, top, bottom, params)
                }
                (8, true, false) => {
                    filter16_w::<8, false, false>(dst, stride, left, top, bottom, params)
                }
                (8, false, true) => {
                    filter16_w::<8, true, true>(dst, stride, left, top, bottom, params)
                }
                (8, false, false) => {
                    filter16_w::<8, true, false>(dst, stride, left, top, bottom, params)
                }
                (_, true, true) => {
                    filter16_w::<4, false, true>(dst, stride, left, top, bottom, params)
                }
                (_, true, false) => {
                    filter16_w::<4, false, false>(dst, stride, left, top, bottom, params)
                }
                (_, false, true) => {
                    filter16_w::<4, true, true>(dst, stride, left, top, bottom, params)
                }
                (_, false, false) => {
                    filter16_w::<4, true, false>(dst, stride, left, top, bottom, params)
                }
            }
        }

        #[target_feature(enable = $tf)]
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

            // ---- 1) kaynak satırları ----------------------------------------
            let z = _mm_setzero_si128();
            let mut lo = [z; N16];
            let mut hi = [z; N16];
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
            // W = 8: (a, b) = (lo[i], hi[i]); W = 4: (a, b) = (r(i), r(i+1)).
            let n_src = if W == 8 { n_rows } else { n_rows - 1 };
            let pack = |l: &[__m128i; N16], hh: &[__m128i; N16], i: usize| -> (__m128i, __m128i) {
                if W == 8 {
                    (l[i & m], hh[i & m])
                } else {
                    (l[i & m], l[(i + 1) & m])
                }
            };
            let mut srcs_a = [z; N16];
            let mut srcs_b = [z; N16];
            for i in 0..n_src {
                (srcs_a[i & m], srcs_b[i & m]) = pack(&lo, &hi, i);
            }
            // Geçerlilik (yalnız kenar yolunda): piksel 0xFFFF = komşu yok.
            let mut vsrcs_a = [z; N16];
            let mut vsrcs_b = [z; N16];
            if EDGES {
                let mut col_lo = [0u8; 16];
                let mut col_hi = [0u8; 16];
                if !e.contains(CdefEdgeFlags::HAVE_LEFT) {
                    col_lo[..4].fill(0xff);
                }
                if !e.contains(CdefEdgeFlags::HAVE_RIGHT) {
                    // sütun w, w+1: W = 8 → hi piksel 2, 3; W = 4 → lo piksel 6, 7.
                    if W == 8 {
                        col_hi[4..8].fill(0xff);
                    } else {
                        col_lo[12..16].fill(0xff);
                    }
                }
                let (cl, ch) = (load_u8x16(&col_lo), load_u8x16(&col_hi));
                let allv = load_u8x16(&[0xff; 16]);
                let mut vlo = [z; N16];
                let mut vhi = [z; N16];
                for i in 0..n_rows {
                    let missing = (i < 2 && !has_top) || (i >= h + 2 && !has_bottom);
                    (vlo[i & m], vhi[i & m]) = if missing { (allv, allv) } else { (cl, ch) };
                }
                for i in 0..n_src {
                    (vsrcs_a[i & m], vsrcs_b[i & m]) = pack(&vlo, &vhi, i);
                }
            }

            // ---- 2) tap listesi (dav1d `src/cdef_tmpl.c:121-155`) -----------
            let (sel_a, sel_b) = if W == 8 {
                (&SEL16_8A, &SEL16_8B)
            } else {
                (&SEL16_4A, &SEL16_4B)
            };
            let bd_min8 = params.bit_depth.min_8() as i32;
            let pri = i32::from(params.pri_strength);
            let sec = i32::from(params.sec_strength);
            let (pt, st) = TAPS16[usize::from(params.dir) & 7].split_at(4);
            let k0 = 4 - ((pri >> bd_min8) & 1);
            let pk = [
                _mm_set1_epi16(k0 as i16),
                _mm_set1_epi16(((k0 & 3) | 2) as i16),
            ];
            let pri_thr = _mm_set1_epi16(pri as i16);
            let sec_thr = _mm_set1_epi16(sec as i16);
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
            let (center_a, center_b) = (load_u8x16(&sel_a[2]), load_u8x16(&sel_b[2]));
            // `(x + 8) >> 4` = `pmulhrsw(x, 2048)`
            let r2048 = _mm_set1_epi16(2048);

            // ---- 3) satır grupları (W = 8: 1 satır; W = 4: 2 satır) ---------
            let group = if W == 8 { 1 } else { 2 };
            let mut y = 0usize;
            while y < h {
                let px = _mm_or_si128(
                    _mm_shuffle_epi8(srcs_a[(y + 2) & m], center_a),
                    _mm_shuffle_epi8(srcs_b[(y + 2) & m], center_b),
                );
                let mut minv = px;
                let mut maxv = px;
                let mut cons = |t: &Tap, thr: __m128i, cnt: __m128i| -> __m128i {
                    let i = (y + t.row) & m;
                    let bi = t.bi & 7;
                    let (ma, mb) = (load_u8x16(&sel_a[bi]), load_u8x16(&sel_b[bi]));
                    let mut p = _mm_or_si128(
                        _mm_shuffle_epi8(srcs_a[i], ma),
                        _mm_shuffle_epi8(srcs_b[i], mb),
                    );
                    if EDGES {
                        let inv = _mm_or_si128(
                            _mm_shuffle_epi8(vsrcs_a[i], ma),
                            _mm_shuffle_epi8(vsrcs_b[i], mb),
                        );
                        p = blend(p, px, inv);
                    }
                    let d = _mm_sub_epi16(p, px);
                    let ad = _mm_abs_epi16(d);
                    let c = min16(ad, _mm_subs_epu16(thr, _mm_srl_epi16(ad, cnt)));
                    if MINMAX {
                        minv = _mm_min_epi16(p, minv);
                        maxv = _mm_max_epi16(p, maxv);
                    }
                    _mm_sign_epi16(c, d)
                };
                let mut sum = z;
                if pri != 0 {
                    for j in 0..2usize {
                        let c = _mm_add_epi16(
                            cons(&pt[2 * j], pri_thr, pri_cnt),
                            cons(&pt[2 * j + 1], pri_thr, pri_cnt),
                        );
                        sum = _mm_add_epi16(sum, _mm_mullo_epi16(c, pk[j]));
                    }
                }
                if sec != 0 {
                    for j in 0..2usize {
                        let c = _mm_add_epi16(
                            _mm_add_epi16(
                                cons(&st[4 * j], sec_thr, sec_cnt),
                                cons(&st[4 * j + 1], sec_thr, sec_cnt),
                            ),
                            _mm_add_epi16(
                                cons(&st[4 * j + 2], sec_thr, sec_cnt),
                                cons(&st[4 * j + 3], sec_thr, sec_cnt),
                            ),
                        );
                        // sec_tap: k = 0 → 2, k = 1 → 1
                        let c = if j == 0 { _mm_slli_epi16::<1>(c) } else { c };
                        sum = _mm_add_epi16(sum, c);
                    }
                }
                // `sum + (sum >> 15)` sonra `pmulhrsw(·, 2048)` = `(· + 8) >> 4`.
                let r = _mm_add_epi16(sum, _mm_srai_epi16::<15>(sum));
                let mut r = _mm_add_epi16(_mm_mulhrs_epi16(r, r2048), px);
                if MINMAX {
                    r = _mm_min_epi16(_mm_max_epi16(r, minv), maxv);
                }
                if W == 8 {
                    store_u8x16(chunk_mut(dst, y * stride), r);
                } else {
                    store_u8x8(chunk_mut(dst, y * stride), r);
                    store_u8x8(chunk_mut(dst, (y + 1) * stride), _mm_srli_si128::<8>(r));
                }
                y += group;
            }
        }
    };
}

// ---------------------------------------------------------------------------
// SSSE3
// ---------------------------------------------------------------------------

/// SSSE3 gövdeleri (8 bpc `dir` + `fb`, 16 bpc `dir` + `fb`).
mod ssse3 {
    use super::*;

    /// `m ? b : a` (bayt maskesi; `pand` + `pandn` + `por`).
    #[target_feature(enable = "ssse3")]
    #[inline]
    fn blend(a: __m128i, b: __m128i, m: __m128i) -> __m128i {
        _mm_or_si128(_mm_and_si128(m, b), _mm_andnot_si128(m, a))
    }

    /// İşlenenler `[0, 2^15)` içinde: işaretli en küçük (`pminsw`).
    #[target_feature(enable = "ssse3")]
    #[inline]
    fn min16(a: __m128i, b: __m128i) -> __m128i {
        _mm_min_epi16(a, b)
    }

    /// Alt 8 bayt `u8 → i16` (`punpcklbw` sıfırla).
    #[target_feature(enable = "ssse3")]
    #[inline]
    fn u8_i16(v: __m128i) -> __m128i {
        _mm_unpacklo_epi8(v, _mm_setzero_si128())
    }

    /// 32 bit lane 2'ye ekleme (lane 2 sıfırken; `movd` + `pslldq` + `por`).
    #[target_feature(enable = "ssse3")]
    #[inline]
    fn ek32_2(v: __m128i, x: i32) -> __m128i {
        _mm_or_si128(v, _mm_slli_si128::<8>(_mm_cvtsi32_si128(x)))
    }

    /// 32 bit lane 3'e ekleme (lane 3 sıfırken).
    #[target_feature(enable = "ssse3")]
    #[inline]
    fn ek32_3(v: __m128i, x: i32) -> __m128i {
        _mm_or_si128(v, _mm_slli_si128::<12>(_mm_cvtsi32_si128(x)))
    }

    /// 32 bit çarpımın alt yarısı (`pmuludq` çift + tek lane).
    #[target_feature(enable = "ssse3")]
    #[inline]
    fn mul32(a: __m128i, b: __m128i) -> __m128i {
        let cift = _mm_mul_epu32(a, b);
        let tek = _mm_mul_epu32(_mm_srli_epi64::<32>(a), _mm_srli_epi64::<32>(b));
        _mm_unpacklo_epi32(
            _mm_shuffle_epi32::<0b00_00_10_00>(cift),
            _mm_shuffle_epi32::<0b00_00_10_00>(tek),
        )
    }

    /// 32 bit en büyük (işaretli; maliyetler `< 2^31`).
    #[target_feature(enable = "ssse3")]
    #[inline]
    fn max32(a: __m128i, b: __m128i) -> __m128i {
        blend(b, a, _mm_cmpgt_epi32(a, b))
    }

    dir_govdesi!("ssse3");
    filter8_govdesi!("ssse3");
    filter16_govdesi!("ssse3");
}

// ---------------------------------------------------------------------------
// SSE4.1
// ---------------------------------------------------------------------------

/// SSE4.1 gövdeleri (8 bpc `dir` + `fb`, 16 bpc `dir`).
mod sse41 {
    use super::*;

    /// `m ? b : a` (`pand` + `pandn` + `por`). `pblendvb` (SSE4.1) denendi:
    /// eski kodlamada maske örtük `xmm0` işlenenidir; 8×8 gövdesinde yazmaç
    /// baskısıyla çekirdek ölçümü 83,6 → 113,9 ns (+%36; 4×8 ve 4×4 değişmedi),
    /// geri alındı (P30 ölçümü, `docs/baseline/f4-cdef-kapisi.md`).
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn blend(a: __m128i, b: __m128i, m: __m128i) -> __m128i {
        _mm_or_si128(_mm_and_si128(m, b), _mm_andnot_si128(m, a))
    }

    /// İşaretsiz en küçük (`pminuw`).
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn min16(a: __m128i, b: __m128i) -> __m128i {
        _mm_min_epu16(a, b)
    }

    /// Alt 8 bayt `u8 → i16` (`pmovzxbw`).
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn u8_i16(v: __m128i) -> __m128i {
        _mm_cvtepu8_epi16(v)
    }

    /// 32 bit lane 2'ye ekleme (`pinsrd`).
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn ek32_2(v: __m128i, x: i32) -> __m128i {
        _mm_insert_epi32::<2>(v, x)
    }

    /// 32 bit çarpımın alt yarısı (`pmulld`).
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn mul32(a: __m128i, b: __m128i) -> __m128i {
        _mm_mullo_epi32(a, b)
    }

    /// 32 bit işaretsiz en büyük (`pmaxud`).
    #[target_feature(enable = "sse4.1")]
    #[inline]
    fn max32(a: __m128i, b: __m128i) -> __m128i {
        _mm_max_epu32(a, b)
    }

    dir_govdesi!("sse4.1");
    filter8_govdesi!("sse4.1");
}
