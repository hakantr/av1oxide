//! Kayıpsız `TX_4X4 × WHT_WHT` x86_64 SSE2 ve AVX2 çekirdekleri — F4-ITX a (P43 Adım 1).
//!
//! Bu modül dav1d'nin `src/x86/itx_sse.asm` (`D02583`), `src/x86/itx16_sse.asm` (`D02584`),
//! `src/x86/itx_avx2.asm` (`D02581`) ve `src/x86/itx16_avx2.asm` (`D02582`) dosyalarındaki
//! 4 kayıpsız Walsh-Hadamard (`WHT_WHT`) 4×4 ters dönüşüm sembolünü (`8`, `10` ve `12` bpc =
//! 6 yuva adayı) **saf Rust intrinsics** ile gerçekler; dış `.asm`/`.S`, `asm!` ve
//! `global_asm!` yoktur (CEK-02).
//!
//! ## Sözleşme ve ISA sınırları
//!
//! - `WHT_WHT` `eob` değerine bakmaz; her zaman `coeff[..16]` okunur, `>> 2` ölçeklenir,
//!   iki 1B WHT4 geçişi uygulanır, **`coeff[..16]` sıfırlanır** (`coeff[16..1024]` dokunulmaz!)
//!   ve `dst`'nin yalnızca `4` satır × `4 · bpp` baytı okunup `[0, (1 << BPC) − 1]` aralığına
//!   kırpılarak geri yazılır (`0` taşma).
//! - Girişte O(1) sözleşme denetimi ([`itx8_sozlesmesi`], [`itx16_sozlesmesi`]) ve
//!   `(tx, txtp) == (TxSize::Tx4x4, TxType::WhtWht)` ile `bit_depth` uyumu aranır;
//!   sözleşme dışı girdide paniklemeden [`itx_skaler_devret`] üzerinden sayaç artırılıp
//!   skaler monomorfa devredilir.
//! - `itx_sse2_*` gövdeleri yalnız `#[target_feature(enable = "sse2")]` ve [`Sse2Token`]
//!   ile derlenir; SSSE3+ ya da VEX yönergesi üretmez.
//! - `itx_avx2_*` gövdeleri yalnız `#[target_feature(enable = "avx2")]` ve [`Avx2Token`]
//!   ile derlenir; EVEX (`zmm` / `{k*}`) yönergesi üretmez.
//!
//! ## Sembol eşlemesi (dav1d @ `99f3354`)
//!
//! | ID | Bit derinliği | ISA | dav1d sembolü |
//! |---|---|---|---|
//! | D02583 | 8 bpc | SSE2 | `dav1d_inv_txfm_add_wht_wht_4x4_8bpc_sse2` |
//! | D02584 | 10, 12 bpc | SSE2 | `dav1d_inv_txfm_add_wht_wht_4x4_16bpc_sse2` |
//! | D02581 | 8 bpc | AVX2 | `dav1d_inv_txfm_add_wht_wht_4x4_8bpc_avx2` |
//! | D02582 | 10, 12 bpc | AVX2 | `dav1d_inv_txfm_add_wht_wht_4x4_16bpc_avx2` |

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_add_epi32, _mm_adds_epi16, _mm_blend_epi32,
    _mm_cvtepu8_epi16, _mm_cvtsi32_si128, _mm_cvtsi64_si128, _mm_max_epi16, _mm_min_epi16,
    _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi16, _mm_setr_epi32, _mm_setzero_si128,
    _mm_srai_epi16, _mm_srai_epi32, _mm_srli_si128, _mm_sub_epi16, _mm_sub_epi32,
    _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi8,
    _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm256_add_epi32,
    _mm256_blend_epi32, _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_packs_epi32,
    _mm256_permute4x64_epi64, _mm256_set_m128i, _mm256_setzero_si256, _mm256_srai_epi32,
    _mm256_sub_epi32, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi32,
    _mm256_unpacklo_epi64,
};

use super::mem::{chunk, chunk_mut, store_i32x8, store_u8x4, store_u8x8};
use crate::cpu::{Avx2Token, Sse2Token};
use crate::dsp::itx::{
    ItxFn, TxSize, TxType, itx_skaler_devret, itx8_sozlesmesi, itx16_sozlesmesi,
};
use crate::frame::BitDepth;

/// 4 × `i32` dilimini `__m128i`'ye yükler (`movdqu`).
#[target_feature(enable = "sse2")]
#[inline]
fn yukle_i32x4(c: &[i32; 4]) -> __m128i {
    _mm_setr_epi32(c[0], c[1], c[2], c[3])
}

/// 8 bpc paketlenmiş 1B WHT4 geçişi (`itx_sse.asm:411-423` `IWHT4_1D_PACKED`).
///
/// Giriş: `m0 = [in0 (4×i16), in1 (4×i16)]`, `m1 = [in2 (4×i16), in3 (4×i16)]`.
/// Çıkış: `(m0, m1, m2)`:
/// - `m0` üst 64 bit: `out0`
/// - `m1` alt 64 bit: `out2`, üst 64 bit: `out1`
/// - `m2` alt 64 bit: `out3`
#[target_feature(enable = "sse2")]
#[inline]
fn iwht4_1d_packed_8(m0: __m128i, m1: __m128i) -> (__m128i, __m128i, __m128i) {
    let m3 = _mm_unpackhi_epi64(m0, m1);
    let m0 = _mm_unpacklo_epi64(m0, m1);
    let m2 = _mm_sub_epi16(m0, m3);
    let m0 = _mm_add_epi16(m0, m3);
    let m2 = _mm_unpackhi_epi64(m2, m2);
    let m0 = _mm_unpacklo_epi64(m0, m0);
    let m1 = _mm_srai_epi16::<1>(_mm_sub_epi16(m0, m2));
    let m1 = _mm_sub_epi16(m1, m3);
    let m0 = _mm_sub_epi16(m0, m1);
    let m2 = _mm_add_epi16(m2, m1);
    (m0, m1, m2)
}

/// 16 bpc 1B WHT4 geçişi (`itx16_sse.asm:190-202` `IWHT4_1D`).
///
/// Giriş: `(in0, in1, in2, in3)` (her biri `4 × i32`).
/// Çıkış: `(out0, out1, out2, out3)`.
#[target_feature(enable = "sse2")]
#[inline]
fn iwht4_1d_16_sse2(
    in0: __m128i,
    in1: __m128i,
    in2: __m128i,
    in3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let m0 = _mm_add_epi32(in0, in1);
    let m4 = _mm_sub_epi32(in2, in3);
    let m5 = _mm_srai_epi32::<1>(_mm_sub_epi32(m0, m4));
    let out2 = _mm_sub_epi32(m5, in1);
    let out1 = _mm_sub_epi32(m5, in3);
    let out0 = _mm_sub_epi32(m0, out1);
    let out3 = _mm_add_epi32(m4, out2);
    (out0, out1, out2, out3)
}

/// 16 bpc AVX2 256-bit paketlenmiş 1B WHT4 geçişi (`itx16_avx2.asm:161-173`).
#[target_feature(enable = "avx2")]
#[inline]
fn iwht4_1d_16_avx2(m0: __m256i, m1: __m256i) -> (__m256i, __m256i, __m256i) {
    let m2 = _mm256_sub_epi32(m0, m1);
    let m0 = _mm256_add_epi32(m0, m1);
    let m2 = _mm256_permute4x64_epi64::<0xFA>(m2);
    let m0 = _mm256_permute4x64_epi64::<0x50>(m0);
    let m1 = _mm256_permute4x64_epi64::<0xD8>(m1);
    let m3 = _mm256_srai_epi32::<1>(_mm256_sub_epi32(m0, m2));
    let m3 = _mm256_sub_epi32(m3, m1);
    let m0 = _mm256_sub_epi32(m0, m3);
    let m2 = _mm256_add_epi32(m2, m3);
    (m0, m2, m3)
}

/// `D02583`: `dav1d_inv_txfm_add_wht_wht_4x4_8bpc_sse2` çekirdek gövdesi.
#[target_feature(enable = "sse2")]
#[inline(never)]
pub(crate) fn itx_sse2_wht_wht_4x4_8bpc(dst: &mut [u8], dst_stride: usize, coeff: &mut [i32; 16]) {
    let c0 = yukle_i32x4(chunk(coeff, 0));
    let c1 = yukle_i32x4(chunk(coeff, 4));
    let c2 = yukle_i32x4(chunk(coeff, 8));
    let c3 = yukle_i32x4(chunk(coeff, 12));
    coeff.fill(0);

    let m0 = _mm_srai_epi16::<2>(_mm_packs_epi32(c0, c1));
    let m1 = _mm_srai_epi16::<2>(_mm_packs_epi32(c2, c3));
    let (m0, m1, m2) = iwht4_1d_packed_8(m0, m1);

    let t0 = _mm_unpackhi_epi16(m0, m1);
    let m3 = _mm_unpacklo_epi16(m1, m2);
    let m1 = _mm_unpackhi_epi32(t0, m3);
    let m0 = _mm_unpacklo_epi32(t0, m3);
    let (m0, m1, m2) = iwht4_1d_packed_8(m0, m1);

    // Alt 64 bit: out0 (`m0`'ın üst 64 biti), üst 64 bit: out3 (`m2`'nin alt 64 biti).
    let m0 = _mm_unpacklo_epi64(_mm_srli_si128::<8>(m0), m2);

    let r0 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, 0)));
    let r1 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, dst_stride)));
    let r2 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, 2 * dst_stride)));
    let r3 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, 3 * dst_stride)));

    let z = _mm_setzero_si128();
    let d03 = _mm_unpacklo_epi8(_mm_unpacklo_epi32(r0, r3), z);
    let d21 = _mm_unpacklo_epi8(_mm_unpacklo_epi32(r2, r1), z);
    let s03 = _mm_add_epi16(d03, m0);
    let s21 = _mm_add_epi16(d21, m1);
    let p = _mm_packus_epi16(s03, s21);

    store_u8x4(chunk_mut(dst, 0), p);
    store_u8x4(chunk_mut(dst, 3 * dst_stride), _mm_srli_si128::<4>(p));
    store_u8x4(chunk_mut(dst, 2 * dst_stride), _mm_srli_si128::<8>(p));
    store_u8x4(chunk_mut(dst, dst_stride), _mm_srli_si128::<12>(p));
}

/// `D02584`: `dav1d_inv_txfm_add_wht_wht_4x4_16bpc_sse2` çekirdek gövdesi.
#[target_feature(enable = "sse2")]
#[inline(never)]
pub(crate) fn itx_sse2_wht_wht_4x4_16bpc(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32; 16],
    bdmax: i16,
) {
    let m0 = _mm_srai_epi32::<2>(yukle_i32x4(chunk(coeff, 0)));
    let m1 = _mm_srai_epi32::<2>(yukle_i32x4(chunk(coeff, 4)));
    let m2 = _mm_srai_epi32::<2>(yukle_i32x4(chunk(coeff, 8)));
    let m3 = _mm_srai_epi32::<2>(yukle_i32x4(chunk(coeff, 12)));
    coeff.fill(0);

    let (m0, m5, m2, m4) = iwht4_1d_16_sse2(m0, m1, m2, m3);
    let t1 = _mm_unpacklo_epi32(m0, m5);
    let t3 = _mm_unpackhi_epi32(m0, m5);
    let t5 = _mm_unpacklo_epi32(m2, m4);
    let t2 = _mm_unpackhi_epi32(m2, m4);
    let m0 = _mm_unpacklo_epi64(t1, t5);
    let m1 = _mm_unpackhi_epi64(t1, t5);
    let m2 = _mm_unpacklo_epi64(t3, t2);
    let m3 = _mm_unpackhi_epi64(t3, t2);

    let (m0, m5, m2, m4) = iwht4_1d_16_sse2(m0, m1, m2, m3);
    // `m0`: alt 64 bit = out0 (satır 0), üst 64 bit = out3 (satır 3).
    // `m2`: alt 64 bit = out2 (satır 2), üst 64 bit = out1 (satır 1).
    let m0 = _mm_packs_epi32(m0, m4);
    let m2 = _mm_packs_epi32(m2, m5);

    let r0 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, 0)));
    let r1 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, dst_stride)));
    let r2 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, 2 * dst_stride)));
    let r3 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, 3 * dst_stride)));

    let d03 = _mm_unpacklo_epi64(r0, r3);
    let d21 = _mm_unpacklo_epi64(r2, r1);
    let z = _mm_setzero_si128();
    let max_v = _mm_set1_epi16(bdmax);

    let s03 = _mm_min_epi16(_mm_max_epi16(_mm_adds_epi16(m0, d03), z), max_v);
    let s21 = _mm_min_epi16(_mm_max_epi16(_mm_adds_epi16(m2, d21), z), max_v);

    store_u8x8(chunk_mut(dst, 0), s03);
    store_u8x8(chunk_mut(dst, 3 * dst_stride), _mm_srli_si128::<8>(s03));
    store_u8x8(chunk_mut(dst, 2 * dst_stride), s21);
    store_u8x8(chunk_mut(dst, dst_stride), _mm_srli_si128::<8>(s21));
}

/// `D02581`: `dav1d_inv_txfm_add_wht_wht_4x4_8bpc_avx2` çekirdek gövdesi.
#[target_feature(enable = "avx2")]
#[inline(never)]
pub(crate) fn itx_avx2_wht_wht_4x4_8bpc(dst: &mut [u8], dst_stride: usize, coeff: &mut [i32; 16]) {
    let c0 = yukle_i32x4(chunk(coeff, 0));
    let c1 = yukle_i32x4(chunk(coeff, 4));
    let c2 = yukle_i32x4(chunk(coeff, 8));
    let c3 = yukle_i32x4(chunk(coeff, 12));
    let z256 = _mm256_setzero_si256();
    store_i32x8(chunk_mut(coeff, 0), z256);
    store_i32x8(chunk_mut(coeff, 8), z256);

    let m0 = _mm_srai_epi16::<2>(_mm_packs_epi32(c0, c1));
    let m1 = _mm_srai_epi16::<2>(_mm_packs_epi32(c2, c3));
    let (m0, m1, m2) = iwht4_1d_packed_8(m0, m1);

    let t0 = _mm_unpackhi_epi16(m0, m1);
    let m3 = _mm_unpacklo_epi16(m1, m2);
    let m1 = _mm_unpackhi_epi32(t0, m3);
    let m0 = _mm_unpacklo_epi32(t0, m3);
    let (m0, m1, m2) = iwht4_1d_packed_8(m0, m1);

    // `vpblendd m0, m2, 0x03`: alt 64 bit = out3 (`m2`), üst 64 bit = out0 (`m0`).
    let m0 = _mm_blend_epi32::<0x03>(m0, m2);

    let r0 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, 0)));
    let r1 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, dst_stride)));
    let r2 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, 2 * dst_stride)));
    let r3 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, 3 * dst_stride)));

    let d30 = _mm_cvtepu8_epi16(_mm_unpacklo_epi32(r3, r0));
    let d21 = _mm_cvtepu8_epi16(_mm_unpacklo_epi32(r2, r1));
    let s30 = _mm_add_epi16(m0, d30);
    let s21 = _mm_add_epi16(m1, d21);
    let p = _mm_packus_epi16(s30, s21);

    store_u8x4(chunk_mut(dst, 3 * dst_stride), p);
    store_u8x4(chunk_mut(dst, 0), _mm_srli_si128::<4>(p));
    store_u8x4(chunk_mut(dst, 2 * dst_stride), _mm_srli_si128::<8>(p));
    store_u8x4(chunk_mut(dst, dst_stride), _mm_srli_si128::<12>(p));
}

/// `D02582`: `dav1d_inv_txfm_add_wht_wht_4x4_16bpc_avx2` çekirdek gövdesi.
#[target_feature(enable = "avx2")]
#[inline(never)]
pub(crate) fn itx_avx2_wht_wht_4x4_16bpc(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32; 16],
    bdmax: i16,
) {
    let m0 = _mm256_set_m128i(yukle_i32x4(chunk(coeff, 8)), yukle_i32x4(chunk(coeff, 0)));
    let m1 = _mm256_set_m128i(yukle_i32x4(chunk(coeff, 12)), yukle_i32x4(chunk(coeff, 4)));
    let z256 = _mm256_setzero_si256();
    store_i32x8(chunk_mut(coeff, 0), z256);
    store_i32x8(chunk_mut(coeff, 8), z256);

    let m0 = _mm256_srai_epi32::<2>(m0);
    let m1 = _mm256_srai_epi32::<2>(m1);
    let (m0, m2, m3) = iwht4_1d_16_avx2(m0, m1);

    let t0 = _mm256_unpackhi_epi32(m0, m3);
    let t3 = _mm256_unpacklo_epi32(m3, m2);
    let m1 = _mm256_unpackhi_epi64(t0, t3);
    let m0 = _mm256_unpacklo_epi64(t0, t3);
    let (m0, m2, m3) = iwht4_1d_16_avx2(m0, m1);

    let m0 = _mm256_blend_epi32::<0x33>(m0, m2);
    let m0 = _mm256_packs_epi32(m0, m3);
    let xm0 = _mm256_castsi256_si128(m0);
    let xm2 = _mm256_extracti128_si256::<1>(m0);
    // `xm1`: alt 64 bit = out2 (satır 2), üst 64 bit = out1 (satır 1).
    // `xm0`: alt 64 bit = out3 (satır 3), üst 64 bit = out0 (satır 0).
    let xm1 = _mm_unpackhi_epi32(xm0, xm2);
    let xm0 = _mm_unpacklo_epi32(xm0, xm2);

    let r0 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, 0)));
    let r1 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, dst_stride)));
    let r2 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, 2 * dst_stride)));
    let r3 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, 3 * dst_stride)));

    let d30 = _mm_unpacklo_epi64(r3, r0);
    let d21 = _mm_unpacklo_epi64(r2, r1);
    let z = _mm_setzero_si128();
    let max_v = _mm_set1_epi16(bdmax);

    let s30 = _mm_min_epi16(_mm_max_epi16(_mm_adds_epi16(xm0, d30), z), max_v);
    let s21 = _mm_min_epi16(_mm_max_epi16(_mm_adds_epi16(xm1, d21), z), max_v);

    store_u8x8(chunk_mut(dst, 0), _mm_srli_si128::<8>(s30));
    store_u8x8(chunk_mut(dst, dst_stride), _mm_srli_si128::<8>(s21));
    store_u8x8(chunk_mut(dst, 2 * dst_stride), s21);
    store_u8x8(chunk_mut(dst, 3 * dst_stride), s30);
}

/// 8 bpc SSE2 `TX_4X4 × WHT_WHT` giriş sarmalayıcısı (`D02583`).
fn wht8_sse2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
    bit_depth: BitDepth,
) {
    if bit_depth != BitDepth::Eight
        || tx != TxSize::Tx4x4
        || txtp != TxType::WhtWht
        || !itx8_sozlesmesi(dst.len(), dst_stride, coeff.len(), eob, tx, txtp)
    {
        itx_skaler_devret::<8>(dst, dst_stride, coeff, eob, tx, txtp);
        return;
    }
    let coeff16 = chunk_mut::<i32, 16>(coeff, 0);
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Sse2Token` ile verilir (`wht8_sse2_fn`);
    // belirteç SSE2 tespiti + maske izninin kanıtıdır. (2) sınır: `itx8_sozlesmesi`
    // `dst`'nin `3·dst_stride + 4` baytı ve `coeff`'un `1024` öğeyi kapsadığını garanti eder;
    // bütün erişimler sabit boyutlu `chunk`/`chunk_mut` dizileri üzerindendir. (3) hizalama/takma ad:
    // `dst` ve `coeff16` ayrık `&mut` referanslardır, hizasız erişim kullanılır.
    unsafe {
        itx_sse2_wht_wht_4x4_8bpc(dst, dst_stride, coeff16);
    }
}

/// 16 bpc (`10`/`12` bit) SSE2 `TX_4X4 × WHT_WHT` giriş sarmalayıcısı (`D02584`).
fn wht16_sse2_giris<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
    bit_depth: BitDepth,
) {
    if bit_depth.bits() != BPC
        || tx != TxSize::Tx4x4
        || txtp != TxType::WhtWht
        || !itx16_sozlesmesi(dst.len(), dst_stride, coeff.len(), eob, tx, txtp)
    {
        itx_skaler_devret::<BPC>(dst, dst_stride, coeff, eob, tx, txtp);
        return;
    }
    let coeff16 = chunk_mut::<i32, 16>(coeff, 0);
    let bdmax = ((1i32 << BPC) - 1) as i16;
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Sse2Token` ile verilir (`wht16_sse2_fn`);
    // belirteç SSE2 tespiti + maske izninin kanıtıdır. (2) sınır: `itx16_sozlesmesi`
    // `dst`'nin `3·dst_stride + 8` baytı ve `coeff`'un `1024` öğeyi kapsadığını garanti eder;
    // bütün erişimler sabit boyutlu `chunk`/`chunk_mut` dizileri üzerindendir. (3) hizalama/takma ad:
    // `dst` ve `coeff16` ayrık `&mut` referanslardır, hizasız erişim kullanılır.
    unsafe {
        itx_sse2_wht_wht_4x4_16bpc(dst, dst_stride, coeff16, bdmax);
    }
}

/// 8 bpc AVX2 `TX_4X4 × WHT_WHT` giriş sarmalayıcısı (`D02581`).
fn wht8_avx2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
    bit_depth: BitDepth,
) {
    if bit_depth != BitDepth::Eight
        || tx != TxSize::Tx4x4
        || txtp != TxType::WhtWht
        || !itx8_sozlesmesi(dst.len(), dst_stride, coeff.len(), eob, tx, txtp)
    {
        itx_skaler_devret::<8>(dst, dst_stride, coeff, eob, tx, txtp);
        return;
    }
    let coeff16 = chunk_mut::<i32, 16>(coeff, 0);
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir (`wht8_avx2_fn`);
    // belirteç AVX2 tespiti + maske izninin kanıtıdır. (2) sınır: `itx8_sozlesmesi`
    // `dst`'nin `3·dst_stride + 4` baytı ve `coeff`'un `1024` öğeyi kapsadığını garanti eder;
    // bütün erişimler sabit boyutlu `chunk`/`chunk_mut` dizileri üzerindendir. (3) hizalama/takma ad:
    // `dst` ve `coeff16` ayrık `&mut` referanslardır, hizasız erişim kullanılır.
    unsafe {
        itx_avx2_wht_wht_4x4_8bpc(dst, dst_stride, coeff16);
    }
}

/// 16 bpc (`10`/`12` bit) AVX2 `TX_4X4 × WHT_WHT` giriş sarmalayıcısı (`D02582`).
fn wht16_avx2_giris<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
    bit_depth: BitDepth,
) {
    if bit_depth.bits() != BPC
        || tx != TxSize::Tx4x4
        || txtp != TxType::WhtWht
        || !itx16_sozlesmesi(dst.len(), dst_stride, coeff.len(), eob, tx, txtp)
    {
        itx_skaler_devret::<BPC>(dst, dst_stride, coeff, eob, tx, txtp);
        return;
    }
    let coeff16 = chunk_mut::<i32, 16>(coeff, 0);
    let bdmax = ((1i32 << BPC) - 1) as i16;
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir (`wht16_avx2_fn`);
    // belirteç AVX2 tespiti + maske izninin kanıtıdır. (2) sınır: `itx16_sozlesmesi`
    // `dst`'nin `3·dst_stride + 8` baytı ve `coeff`'un `1024` öğeyi kapsadığını garanti eder;
    // bütün erişimler sabit boyutlu `chunk`/`chunk_mut` dizileri üzerindendir. (3) hizalama/takma ad:
    // `dst` ve `coeff16` ayrık `&mut` referanslardır, hizasız erişim kullanılır.
    unsafe {
        itx_avx2_wht_wht_4x4_16bpc(dst, dst_stride, coeff16, bdmax);
    }
}

/// `itxfm_add[TX_4X4][WHT_WHT]` 8 bpc SSE2 yuvası (yalnız [`Sse2Token`] ile; `D02583`).
#[must_use]
pub const fn wht8_sse2_fn(_sse2: Sse2Token) -> ItxFn {
    wht8_sse2_giris
}

/// `itxfm_add[TX_4X4][WHT_WHT]` 10/12 bpc SSE2 yuvası (yalnız [`Sse2Token`] ile; `D02584`).
#[must_use]
pub const fn wht16_sse2_fn(_sse2: Sse2Token, bit_depth: u8) -> ItxFn {
    if bit_depth == 10 {
        wht16_sse2_giris::<10>
    } else {
        wht16_sse2_giris::<12>
    }
}

/// `itxfm_add[TX_4X4][WHT_WHT]` 8 bpc AVX2 yuvası (yalnız [`Avx2Token`] ile; `D02581`).
#[must_use]
pub const fn wht8_avx2_fn(_avx2: Avx2Token) -> ItxFn {
    wht8_avx2_giris
}

/// `itxfm_add[TX_4X4][WHT_WHT]` 10/12 bpc AVX2 yuvası (yalnız [`Avx2Token`] ile; `D02582`).
#[must_use]
pub const fn wht16_avx2_fn(_avx2: Avx2Token, bit_depth: u8) -> ItxFn {
    if bit_depth == 10 {
        wht16_avx2_giris::<10>
    } else {
        wht16_avx2_giris::<12>
    }
}
