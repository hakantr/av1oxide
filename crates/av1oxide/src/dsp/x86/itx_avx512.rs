//! 8 bpc kayıplı / genel ters dönüşüm (`w, h ≤ 16`) x86_64 AVX512ICL çekirdekleri — F4-ITX a (P43 Adım 3).
//!
//! Bu modül dav1d'nin `src/x86/itx_avx512.asm` ve `src/x86/itx.h:221-244` dosyalarındaki
//! `w, h ≤ 16` boyutlu **140** adet WHT-dışı 8 bpc AVX512ICL ters dönüşüm kaydını saf kararlı
//! Rust `core::arch::x86_64` intrinsics ile gerçekler (`.asm`, `.S`, `asm!`, `global_asm!`
//! ve dış C bağımlılığı yoktur; CEK-02):
//!
//! - `TX_4X4`, `RTX_4X8`, `RTX_8X4`, `TX_8X8`, `RTX_4X16`, `RTX_16X4`, `RTX_8X16`,
//!   `RTX_16X8`: 16 WHT-dışı `TxType` türünün tamamı (`8 × 16 = 128` kayıt),
//! - `TX_16X16`: desteklenen 12 `TxType` türü (`DctDct`, `AdstDct`, `DctAdst`, `AdstAdst`,
//!   `FlipadstDct`, `DctFlipadst`, `FlipadstFlipadst`, `AdstFlipadst`, `FlipadstAdst`,
//!   `Idtx`, `VDct`, `HDct` = `12` kayıt).
//!
//! ## Sözleşme, P34 izolasyonu, Zen 4 (`SLOW_GATHER`) kuralı ve ISA sınırı
//!
//! - **P34 modül izolasyonu:** Bu modül `itx_buyuk_avx512`, `itx`, `itx_buyuk`, `itx_ssse3`
//!   ya da `itx_wht` ile hiçbir `#[target_feature]` yardımcısı paylaşmaz; bütün 1B dönüşüm,
//!   ZMM/VL transpoz ve piksel ekleme yardımcıları modül içinde kendi kopyalarıyla tanımlıdır.
//! - **Zen 4 kuralı (`0` gather/scatter):** `dav1d` `src/x86/itx.h:221-244` içinde 8 bpc
//!   AVX512ICL ITX yuvalarında `SLOW_GATHER` bayrağını denetlemez ve Zen 4 (`0x3f`) üzerinde
//!   155 yuvanın tamamını canlı tabloya kurar. Bu modülde `vpgather*` / `vpscatter*`
//!   yönergesi kesinlikle üretilmez (`0` adet); tüm hedef satır erişimleri güvenli dilim
//!   indekslemesiyle (`chunk` / `chunk_mut`) yapılır.
//! - **512-bit Split-ZMM + VNNI kelebekler:** 16 şeritli geçişler (`4×16`, `16×4`, `8×16`,
//!   `16×8`, `16×16`) `z16` Split-ZMM (`__m512i`, her 128-bit şeritte 4 etkin `i16` = toplam
//!   16 şerit) düzeninde tek yarı-şerit (`$HALF_LANE = true`) + `_mm512_dpwssd_epi32`
//!   (AVX-512 VNNI) ile işlenir; 4 ve 8 şeritli geçişler AVX512VL + VNNI (`_mm_dpwssd_epi32`,
//!   `_mm256_dpwssd_epi32`) kullanır.
//! - **Katsayı sıfırlama sözleşmesi:**
//!   - `txtp == TxType::DctDct && eob == 0` (DC-only hızlı yolu): **yalnız `coeff[0] = 0`**
//!     sıfırlanır; `coeff[1..1024]` öğelerine dokunulmaz.
//!   - Genel yol (`eob ≥ 1` veya `txtp != TxType::DctDct`): `eob` alt-blok hızlı yoluna
//!     girilse bile `coeff[..w * h]` tamamen sıfırlanır; `coeff[w * h..1024]` kuyruğuna
//!     dokunulmaz.
//! - **Sınır güvenliği ve tek `unsafe` bloğu:** Her satırda tam `w` bayt (`4`, `8` veya `16`)
//!   okunur/yazılır; sözleşme dışı girdide paniklemeden [`itx_skaler_devret`] ile skaler
//!   monomorfa devredilir. Modülde tek `unsafe` bloğu [`itx8_avx512_giris`] içindedir.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm_add_epi16, _mm_adds_epi16, _mm_adds_epu8, _mm_avg_epu16,
    _mm_cvtepu8_epi16, _mm_cvtsi32_si128, _mm_cvtsi64_si128, _mm_dpwssd_epi32, _mm_mulhrs_epi16,
    _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi8, _mm_set1_epi16, _mm_set1_epi32,
    _mm_setr_epi32, _mm_setzero_si128, _mm_srai_epi16, _mm_srai_epi32, _mm_srli_si128,
    _mm_subs_epi16, _mm_subs_epu8, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64,
    _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm_xor_si128, _mm256_add_epi16,
    _mm256_adds_epi16, _mm256_avg_epu16, _mm256_castsi256_si128, _mm256_cvtepu8_epi16,
    _mm256_dpwssd_epi32, _mm256_extracti128_si256, _mm256_mulhrs_epi16, _mm256_packs_epi32,
    _mm256_packus_epi16, _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set_m128i,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi32, _mm256_setzero_si256,
    _mm256_srai_epi16, _mm256_srai_epi32, _mm256_srli_si256, _mm256_subs_epi16,
    _mm256_unpackhi_epi16, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi16,
    _mm256_unpacklo_epi32, _mm256_unpacklo_epi64, _mm256_xor_si256, _mm512_add_epi16,
    _mm512_adds_epi16, _mm512_avg_epu16, _mm512_bsrli_epi128, _mm512_castsi256_si512,
    _mm512_castsi512_si256, _mm512_cvtepu8_epi16, _mm512_dpwssd_epi32, _mm512_extracti64x4_epi64,
    _mm512_inserti64x4, _mm512_mulhrs_epi16, _mm512_packs_epi32, _mm512_permutexvar_epi64,
    _mm512_set1_epi16, _mm512_set1_epi32, _mm512_setr_epi32, _mm512_setr_epi64,
    _mm512_setzero_si512, _mm512_shuffle_i32x4, _mm512_srai_epi16, _mm512_srai_epi32,
    _mm512_subs_epi16, _mm512_unpackhi_epi16, _mm512_unpackhi_epi32, _mm512_unpacklo_epi16,
    _mm512_unpacklo_epi32, _mm512_unpacklo_epi64, _mm512_xor_si512,
};

use super::mem::{chunk, chunk_mut, load_u8x16, store_i32x8, store_u8x4, store_u8x8, store_u8x16};
use crate::cpu::Avx512IclToken;
use crate::dsp::itx::{ItxFn, TxSize, TxType, is_supported, itx_skaler_devret, itx8_sozlesmesi};
use crate::dsp::scalar::itx_1d::Tx1dType;
use crate::frame::BitDepth;

/// İki `i16` katsayısını `_mm*_dpwssd_epi32` için tek `i32` sabitine paketler (`[lo, hi]`).
#[inline(always)]
const fn pair(lo: i16, hi: i16) -> i32 {
    (lo as u16 as i32) | ((hi as u16 as i32) << 16)
}

/// 4 × `i32` dilimini `__m128i`'ye yükler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_i32x4(c: &[i32; 4]) -> __m128i {
    _mm_setr_epi32(c[0], c[1], c[2], c[3])
}

/// 8 × `i32` dilimini `__m256i`'ye yükler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_i32x8(c: &[i32; 8]) -> __m256i {
    _mm256_setr_epi32(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7])
}

/// 16 × `i32` dilimini `__m512i`'ye yükler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_i32x16(c: &[i32; 16]) -> __m512i {
    _mm512_setr_epi32(
        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12], c[13],
        c[14], c[15],
    )
}

/// 4 × `i32` dilimini doyurmalı `i16`'ya çevirip alt 4 şeride (`[c0..c3, 0, 0, 0, 0]`) yükler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_i16x4(c: &[i32; 4]) -> __m128i {
    _mm_packs_epi32(yukle_i32x4(c), _mm_setzero_si128())
}

/// 8 × `i32` dilimini Split-YMM düzeninde (`[c0..c3, 0..0 || c4..c7, 0..0]`) doyurmalı `i16` olarak yükler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_i16x8(c: &[i32; 8]) -> __m256i {
    _mm256_packs_epi32(yukle_i32x8(c), _mm256_setzero_si256())
}

/// 16 × `i32` dilimini Split-ZMM düzeninde (`[c0..3, 0..0 || c4..7, 0..0 || c8..11, 0..0 || c12..15, 0..0]`)
/// doyurmalı `i16` olarak yükler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_z16(c: &[i32; 16]) -> __m512i {
    _mm512_packs_epi32(yukle_i32x16(c), _mm512_setzero_si512())
}

/// `coeff[..N]` bölgesini (`N` 8'in katı) 256-bit sıfır saklamalarıyla temizler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn sifirla_katsayi<const N: usize>(coeff: &mut [i32; 1024]) {
    const { assert!(N.is_multiple_of(8) && N <= 1024) };
    let z = _mm256_setzero_si256();
    let mut i = 0;
    while i < N {
        store_i32x8(chunk_mut(coeff, i), z);
        i += 8;
    }
}

/// Şerit-paralel AVX512ICL + VNNI (`vpdpwssd`) 1B ters dönüşüm çekirdekleri:
/// - `x4` (`__m128i`, alt 4 şerit etkin, `$HALF_LANE = true`),
/// - `x8` (`__m256i` Split-YMM: her 128-bit yarıda alt 4 şerit etkin = 8 şerit, `$HALF_LANE = true`),
/// - `z16` (`__m512i` Split-ZMM: her 128-bit çeyrekte alt 4 şerit etkin = 16 şerit, `$HALF_LANE = true`).
macro_rules! itx_1d_avx512_impl {
    (
        $mod_ad:ident,
        $V:ty,
        $HALF_LANE:literal,
        $set1_16:ident,
        $set1_32:ident,
        $zero:ident,
        $unpacklo16:ident,
        $unpackhi16:ident,
        $dpwssd:ident,
        $srai32:ident,
        $packs32:ident,
        $adds16:ident,
        $subs16:ident,
        $mulhrs16:ident,
        $srai16:ident,
        $xor:ident,
        $avg_epu16:ident
    ) => {
        #[allow(dead_code)]
        mod $mod_ad {
            use super::*;

            /// İşaretli yuvarlamalı ortalama: `(a + b + 1) >> 1` (`[-32768, 32767]` tam aralıkta taşmasız).
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn savg16(a: $V, b: $V) -> $V {
                let bias = $set1_16(-32768);
                $xor($avg_epu16($xor(a, bias), $xor(b, bias)), bias)
            }

            /// `(out1, out2) = (((a*c1 - b*c2 + 2048) >> 12), ((a*c2 + b*c1 + 2048) >> 12))` (`vpdpwssd`).
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn mulsub(a: $V, b: $V, c1: i16, c2: i16) -> ($V, $V) {
                let lo = $unpacklo16(b, a);
                let k1 = $set1_32(pair(c2.wrapping_neg(), c1));
                let k2 = $set1_32(pair(c1, c2));
                let rnd = $set1_32(2048);
                let r1_lo = $srai32::<12>($dpwssd(rnd, lo, k1));
                let r2_lo = $srai32::<12>($dpwssd(rnd, lo, k2));
                if $HALF_LANE {
                    ($packs32(r1_lo, r1_lo), $packs32(r2_lo, r2_lo))
                } else {
                    let hi = $unpackhi16(b, a);
                    let r1_hi = $srai32::<12>($dpwssd(rnd, hi, k1));
                    let r2_hi = $srai32::<12>($dpwssd(rnd, hi, k2));
                    ($packs32(r1_lo, r1_hi), $packs32(r2_lo, r2_hi))
                }
            }

            /// `b == 0` olduğunda `(a*c1 + 2048) >> 12` ve `(a*c2 + 2048) >> 12` (`vpmulhrsw(a, c * 8)`).
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn mul_single(a: $V, c1: i16, c2: i16) -> ($V, $V) {
                (
                    $mulhrs16(a, $set1_16(c1 * 8)),
                    $mulhrs16(a, $set1_16(c2 * 8)),
                )
            }

            /// `inv_dct4_1d_c` (`src/itx_1d.c:66-96`).
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn idct4(c: &mut [$V; 4]) {
                let (t1, t0) = mulsub(c[0], c[2], 2896, 2896);
                let (t2, t3) = mulsub(c[1], c[3], 1567, 3784);
                c[0] = $adds16(t0, t3);
                c[1] = $adds16(t1, t2);
                c[2] = $subs16(t1, t2);
                c[3] = $subs16(t0, t3);
            }

            /// `c[4..8] == 0` olduğunda `idct8`.
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn idct8_half(c: &mut [$V; 8]) {
                let z = $zero();
                let (t1, t0) = mulsub(c[0], z, 2896, 2896);
                let (t2, t3) = mul_single(c[2], 1567, 3784);
                let e0 = $adds16(t0, t3);
                let e1 = $adds16(t1, t2);
                let e2 = $subs16(t1, t2);
                let e3 = $subs16(t0, t3);

                let (t4a, t7a) = mul_single(c[1], 799, 4017);
                let (t5a, t6a) = mulsub(z, c[3], 3406, 2276);

                let t4 = $adds16(t4a, t5a);
                let t5_in = $subs16(t4a, t5a);
                let t7 = $adds16(t7a, t6a);
                let t6_in = $subs16(t7a, t6a);

                let (t5, t6) = mulsub(t6_in, t5_in, 2896, 2896);

                c[0] = $adds16(e0, t7);
                c[1] = $adds16(e1, t6);
                c[2] = $adds16(e2, t5);
                c[3] = $adds16(e3, t4);
                c[4] = $subs16(e3, t4);
                c[5] = $subs16(e2, t5);
                c[6] = $subs16(e1, t6);
                c[7] = $subs16(e0, t7);
            }

            /// `inv_dct8_1d_c` (`src/itx_1d.c:99-149`).
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn idct8(c: &mut [$V; 8]) {
                let mut even = [c[0], c[2], c[4], c[6]];
                idct4(&mut even);
                let [t0, t1, t2, t3] = even;

                let (t4a, t7a) = mulsub(c[1], c[7], 799, 4017);
                let (t5a, t6a) = mulsub(c[5], c[3], 3406, 2276);

                let t4 = $adds16(t4a, t5a);
                let t5a = $subs16(t4a, t5a);
                let t7 = $adds16(t7a, t6a);
                let t6a = $subs16(t7a, t6a);

                let (t5, t6) = mulsub(t6a, t5a, 2896, 2896);

                c[0] = $adds16(t0, t7);
                c[1] = $adds16(t1, t6);
                c[2] = $adds16(t2, t5);
                c[3] = $adds16(t3, t4);
                c[4] = $subs16(t3, t4);
                c[5] = $subs16(t2, t5);
                c[6] = $subs16(t1, t6);
                c[7] = $subs16(t0, t7);
            }

            /// `c[8..16] == 0` olduğunda `idct16`.
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn idct16_half(c: &mut [$V; 16]) {
                let z = $zero();
                let mut even = [c[0], c[2], c[4], c[6], z, z, z, z];
                idct8_half(&mut even);
                let [t0, t1, t2, t3, t4, t5, t6, t7] = even;

                let (t8a, t15a) = mul_single(c[1], 401, 4076);
                let (t9a, t14a) = mulsub(z, c[7], 3166, 2598);
                let (t10a, t13a) = mul_single(c[5], 1931, 3612);
                let (t11a, t12a) = mulsub(z, c[3], 3920, 1189);

                let t8 = $adds16(t8a, t9a);
                let t9 = $subs16(t8a, t9a);
                let t10 = $subs16(t11a, t10a);
                let t11 = $adds16(t11a, t10a);
                let t12 = $adds16(t12a, t13a);
                let t13 = $subs16(t12a, t13a);
                let t14 = $subs16(t15a, t14a);
                let t15 = $adds16(t15a, t14a);

                let (t9a, t14a) = mulsub(t14, t9, 1567, 3784);
                let (t10a, t13a) = mulsub(t13, t10, -3784, 1567);

                let t8a = $adds16(t8, t11);
                let t9 = $adds16(t9a, t10a);
                let t10 = $subs16(t9a, t10a);
                let t11a = $subs16(t8, t11);
                let t12a = $subs16(t15, t12);
                let t13 = $subs16(t14a, t13a);
                let t14 = $adds16(t14a, t13a);
                let t15a = $adds16(t15, t12);

                let (t10a, t13a) = mulsub(t13, t10, 2896, 2896);
                let (t11, t12) = mulsub(t12a, t11a, 2896, 2896);

                c[0] = $adds16(t0, t15a);
                c[1] = $adds16(t1, t14);
                c[2] = $adds16(t2, t13a);
                c[3] = $adds16(t3, t12);
                c[4] = $adds16(t4, t11);
                c[5] = $adds16(t5, t10a);
                c[6] = $adds16(t6, t9);
                c[7] = $adds16(t7, t8a);
                c[8] = $subs16(t7, t8a);
                c[9] = $subs16(t6, t9);
                c[10] = $subs16(t5, t10a);
                c[11] = $subs16(t4, t11);
                c[12] = $subs16(t3, t12);
                c[13] = $subs16(t2, t13a);
                c[14] = $subs16(t1, t14);
                c[15] = $subs16(t0, t15a);
            }

            /// `inv_dct16_1d_c` (`src/itx_1d.c:152-244`).
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn idct16(c: &mut [$V; 16]) {
                let mut even = [c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]];
                idct8(&mut even);
                let [t0, t1, t2, t3, t4, t5, t6, t7] = even;

                let (t8a, t15a) = mulsub(c[1], c[15], 401, 4076);
                let (t9a, t14a) = mulsub(c[9], c[7], 3166, 2598);
                let (t10a, t13a) = mulsub(c[5], c[11], 1931, 3612);
                let (t11a, t12a) = mulsub(c[13], c[3], 3920, 1189);

                let t8 = $adds16(t8a, t9a);
                let t9 = $subs16(t8a, t9a);
                let t10 = $subs16(t11a, t10a);
                let t11 = $adds16(t11a, t10a);
                let t12 = $adds16(t12a, t13a);
                let t13 = $subs16(t12a, t13a);
                let t14 = $subs16(t15a, t14a);
                let t15 = $adds16(t15a, t14a);

                let (t9a, t14a) = mulsub(t14, t9, 1567, 3784);
                let (t10a, t13a) = mulsub(t13, t10, -3784, 1567);

                let t8a = $adds16(t8, t11);
                let t9 = $adds16(t9a, t10a);
                let t10 = $subs16(t9a, t10a);
                let t11a = $subs16(t8, t11);
                let t12a = $subs16(t15, t12);
                let t13 = $subs16(t14a, t13a);
                let t14 = $adds16(t14a, t13a);
                let t15a = $adds16(t15, t12);

                let (t10a, t13a) = mulsub(t13, t10, 2896, 2896);
                let (t11, t12) = mulsub(t12a, t11a, 2896, 2896);

                c[0] = $adds16(t0, t15a);
                c[1] = $adds16(t1, t14);
                c[2] = $adds16(t2, t13a);
                c[3] = $adds16(t3, t12);
                c[4] = $adds16(t4, t11);
                c[5] = $adds16(t5, t10a);
                c[6] = $adds16(t6, t9);
                c[7] = $adds16(t7, t8a);
                c[8] = $subs16(t7, t8a);
                c[9] = $subs16(t6, t9);
                c[10] = $subs16(t5, t10a);
                c[11] = $subs16(t4, t11);
                c[12] = $subs16(t3, t12);
                c[13] = $subs16(t2, t13a);
                c[14] = $subs16(t1, t14);
                c[15] = $subs16(t0, t15a);
            }

            /// `inv_adst4_1d_c` / `inv_flipadst4_1d_c` (`src/itx_1d.c:784-802`).
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn iadst4(c: &mut [$V; 4], flip: bool) {
                let p01_lo = $unpacklo16(c[0], c[1]);
                let p23_lo = $unpacklo16(c[2], c[3]);
                let p01_hi = if $HALF_LANE {
                    $zero()
                } else {
                    $unpackhi16(c[0], c[1])
                };
                let p23_hi = if $HALF_LANE {
                    $zero()
                } else {
                    $unpackhi16(c[2], c[3])
                };
                let rnd = $set1_32(2048);

                let dot = |k0: i16, k1: i16, k2: i16, k3: i16| -> $V {
                    let a = $set1_32(pair(k0, k1));
                    let b = $set1_32(pair(k2, k3));
                    let lo = $srai32::<12>($dpwssd($dpwssd(rnd, p01_lo, a), p23_lo, b));
                    if $HALF_LANE {
                        $packs32(lo, lo)
                    } else {
                        let hi = $srai32::<12>($dpwssd($dpwssd(rnd, p01_hi, a), p23_hi, b));
                        $packs32(lo, hi)
                    }
                };

                let out0 = dot(1321, 3344, 3803, 2482);
                let out1 = dot(2482, 3344, -1321, -3803);
                let out2 = dot(3344, 0, -3344, 3344);
                let out3 = dot(3803, -3344, 2482, -1321);

                *c = if flip {
                    [out3, out2, out1, out0]
                } else {
                    [out0, out1, out2, out3]
                };
            }

            /// `inv_adst8_1d_c` / `inv_flipadst8_1d_c` (`src/itx_1d.c:805-851`).
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn iadst8(c: &mut [$V; 8]) {
                let (t1a, t0a) = mulsub(c[7], c[0], 401, 4076);
                let (t3a, t2a) = mulsub(c[5], c[2], 1931, 3612);
                let (t5a, t4a) = mulsub(c[3], c[4], 3166, 2598);
                let (t7a, t6a) = mulsub(c[1], c[6], 3920, 1189);

                let t0 = $adds16(t0a, t4a);
                let t4 = $subs16(t0a, t4a);
                let t1 = $adds16(t1a, t5a);
                let t5 = $subs16(t1a, t5a);
                let t2_0 = $adds16(t2a, t6a);
                let t6_0 = $subs16(t2a, t6a);
                let t3_0 = $adds16(t3a, t7a);
                let t7_0 = $subs16(t3a, t7a);

                let (t5a, t4a) = mulsub(t4, t5, 1567, 3784);
                let (t6a, t7a) = mulsub(t7_0, t6_0, 3784, 1567);

                let u0 = $adds16(t0, t2_0);
                let u7 = $adds16(t1, t3_0);
                let t2 = $subs16(t0, t2_0);
                let t3 = $subs16(t1, t3_0);
                let u1 = $adds16(t4a, t6a);
                let u6 = $adds16(t5a, t7a);
                let t6 = $subs16(t4a, t6a);
                let t7 = $subs16(t5a, t7a);

                let (u4, u3) = mulsub(t2, t3, 2896, 2896);
                let (u5, u2) = mulsub(t6, t7, 2896, 2896);

                *c = [u0, u1, u2, u3, u4, u5, u6, u7];
            }

            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            fn adst_finish<const N: usize>(c: &mut [$V; N], shift_mul: i16, flip: bool) {
                if shift_mul == 0 {
                    let z = $zero();
                    let mut i = 1;
                    while i < N {
                        c[i] = $subs16(z, c[i]);
                        i += 2;
                    }
                    if flip {
                        let mut lo = 0;
                        let mut hi = N - 1;
                        while lo < hi {
                            c.swap(lo, hi);
                            lo += 1;
                            hi -= 1;
                        }
                    }
                } else {
                    let p = $set1_16(shift_mul);
                    let n = $set1_16(-shift_mul);
                    let mut i = 0;
                    if flip {
                        while i < N / 2 {
                            let j = N - 1 - i;
                            let ci = $mulhrs16(c[i], p);
                            let ci1 = $mulhrs16(c[i + 1], n);
                            let cj1 = $mulhrs16(c[j - 1], p);
                            let cj = $mulhrs16(c[j], n);
                            c[i] = cj;
                            c[i + 1] = cj1;
                            c[j - 1] = ci1;
                            c[j] = ci;
                            i += 2;
                        }
                    } else {
                        while i < N {
                            c[i] = $mulhrs16(c[i], p);
                            c[i + 1] = $mulhrs16(c[i + 1], n);
                            i += 2;
                        }
                    }
                }
            }

            /// `inv_adst16_1d_c` / `inv_flipadst16_1d_c` (`src/itx_1d.c:854-962`).
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn iadst16(c: &mut [$V; 16]) {
                let (t1, t0) = mulsub(c[15], c[0], 201, 4091);
                let (t3, t2) = mulsub(c[13], c[2], 995, 3973);
                let (t5, t4) = mulsub(c[11], c[4], 1751, 3703);
                let (t7, t6) = mulsub(c[9], c[6], 2440, 3290);
                let (t9, t8) = mulsub(c[7], c[8], 3035, 2751);
                let (t11, t10) = mulsub(c[5], c[10], 3513, 2106);
                let (t13, t12) = mulsub(c[3], c[12], 3857, 1380);
                let (t15, t14) = mulsub(c[1], c[14], 4052, 601);

                let t0a = $adds16(t0, t8);
                let t8a = $subs16(t0, t8);
                let t1a = $adds16(t1, t9);
                let t9a = $subs16(t1, t9);
                let t2a = $adds16(t2, t10);
                let t10a = $subs16(t2, t10);
                let t3a = $adds16(t3, t11);
                let t11a = $subs16(t3, t11);
                let t4a = $adds16(t4, t12);
                let t12a = $subs16(t4, t12);
                let t5a = $adds16(t5, t13);
                let t13a = $subs16(t5, t13);
                let t6a = $adds16(t6, t14);
                let t14a = $subs16(t6, t14);
                let t7a = $adds16(t7, t15);
                let t15a = $subs16(t7, t15);

                let (t9, t8) = mulsub(t8a, t9a, 799, 4017);
                let (t11, t10) = mulsub(t10a, t11a, 3406, 2276);
                let (t12, t13) = mulsub(t13a, t12a, 4017, 799);
                let (t14, t15) = mulsub(t15a, t14a, 2276, 3406);

                let t0 = $adds16(t0a, t4a);
                let t4 = $subs16(t0a, t4a);
                let t1 = $adds16(t1a, t5a);
                let t5 = $subs16(t1a, t5a);
                let t2 = $adds16(t2a, t6a);
                let t6 = $subs16(t2a, t6a);
                let t3 = $adds16(t3a, t7a);
                let t7 = $subs16(t3a, t7a);
                let t8a = $adds16(t8, t12);
                let t12a = $subs16(t8, t12);
                let t9a = $adds16(t9, t13);
                let t13a = $subs16(t9, t13);
                let t10a = $adds16(t10, t14);
                let t14a = $subs16(t10, t14);
                let t11a = $adds16(t11, t15);
                let t15a = $subs16(t11, t15);

                let (t5a, t4a) = mulsub(t4, t5, 1567, 3784);
                let (t6a, t7a) = mulsub(t7, t6, 3784, 1567);
                let (t13, t12) = mulsub(t12a, t13a, 1567, 3784);
                let (t14, t15) = mulsub(t15a, t14a, 3784, 1567);

                let u0 = $adds16(t0, t2);
                let u15 = $adds16(t1, t3);
                let t2a = $subs16(t0, t2);
                let t3a = $subs16(t1, t3);

                let u3 = $adds16(t4a, t6a);
                let u12 = $adds16(t5a, t7a);
                let t6 = $subs16(t4a, t6a);
                let t7 = $subs16(t5a, t7a);

                let u1 = $adds16(t8a, t10a);
                let u14 = $adds16(t9a, t11a);
                let t10 = $subs16(t8a, t10a);
                let t11 = $subs16(t9a, t11a);

                let u2 = $adds16(t12, t14);
                let u13 = $adds16(t13, t15);
                let t14a = $subs16(t12, t14);
                let t15a = $subs16(t13, t15);

                let (u8, u7) = mulsub(t2a, t3a, 2896, 2896);
                let (u11, u4) = mulsub(t6, t7, 2896, 2896);
                let (u9, u6) = mulsub(t10, t11, 2896, 2896);
                let (u10, u5) = mulsub(t14a, t15a, 2896, 2896);

                *c = [
                    u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15,
                ];
            }

            /// 1. geçiş: 4 noktalı yatay dönüşüm + `shift` (`0` veya `1`) yuvarlaması.
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn pass1_4<const KIND: u8, const SHIFT: u32>(c: &mut [$V; 4]) {
                match KIND {
                    0 => {
                        idct4(c);
                        round_shift::<4, SHIFT>(c);
                    }
                    1 => {
                        iadst4(c, false);
                        round_shift::<4, SHIFT>(c);
                    }
                    3 => {
                        iadst4(c, true);
                        round_shift::<4, SHIFT>(c);
                    }
                    _ => {
                        let k = $set1_16(13576); // 1697 * 8
                        for v in c.iter_mut() {
                            let t = $mulhrs16(*v, k);
                            *v = if SHIFT == 0 {
                                $adds16(*v, t)
                            } else {
                                savg16(*v, t)
                            };
                        }
                    }
                }
            }

            /// 1. geçiş: 8 noktalı yatay dönüşüm + `shift` (`0` veya `1`) yuvarlaması.
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn pass1_8<const KIND: u8, const SHIFT: u32>(c: &mut [$V; 8]) {
                let sm = if SHIFT == 0 {
                    0
                } else if SHIFT == 1 {
                    16384
                } else {
                    8192
                };
                match KIND {
                    0 => {
                        idct8(c);
                        round_shift::<8, SHIFT>(c);
                    }
                    1 => {
                        iadst8(c);
                        adst_finish(c, sm, false);
                    }
                    3 => {
                        iadst8(c);
                        adst_finish(c, sm, true);
                    }
                    _ => {
                        if SHIFT == 0 {
                            for v in c.iter_mut() {
                                *v = $adds16(*v, *v);
                            }
                        }
                    }
                }
            }

            /// 1. geçiş: 16 noktalı yatay dönüşüm + `shift` (`1` veya `2`) yuvarlaması.
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn pass1_16_eob<const KIND: u8, const SHIFT: u32>(
                c: &mut [$V; 16],
                half: bool,
            ) {
                let sm = if SHIFT == 1 { 16384 } else { 8192 };
                match KIND {
                    0 => {
                        if half {
                            idct16_half(c);
                        } else {
                            idct16(c);
                        }
                        round_shift::<16, SHIFT>(c);
                    }
                    1 => {
                        iadst16(c);
                        adst_finish(c, sm, false);
                    }
                    3 => {
                        iadst16(c);
                        adst_finish(c, sm, true);
                    }
                    _ => {
                        let k = $set1_16(27152); // 1697 * 16
                        let r1 = $set1_16(16384);
                        for v in c.iter_mut() {
                            let t = $mulhrs16(*v, k);
                            *v = if SHIFT == 1 {
                                $adds16(*v, $mulhrs16(t, r1))
                            } else {
                                $avg_epu16(*v, $srai16::<1>(t))
                            };
                        }
                    }
                }
            }

            /// 2. geçiş: 4 noktalı dikey dönüşüm + son `(c + 8) >> 4` yuvarlaması.
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn pass2_4<const KIND: u8>(c: &mut [$V; 4]) {
                let r4 = $set1_16(2048);
                match KIND {
                    0 => {
                        idct4(c);
                        for v in c.iter_mut() {
                            *v = $mulhrs16(*v, r4);
                        }
                    }
                    1 => {
                        iadst4(c, false);
                        for v in c.iter_mut() {
                            *v = $mulhrs16(*v, r4);
                        }
                    }
                    3 => {
                        iadst4(c, true);
                        for v in c.iter_mut() {
                            *v = $mulhrs16(*v, r4);
                        }
                    }
                    _ => {
                        let k = $set1_16(13576);
                        for v in c.iter_mut() {
                            *v = $mulhrs16($adds16(*v, $mulhrs16(*v, k)), r4);
                        }
                    }
                }
            }

            /// 2. geçiş: 8 noktalı dikey dönüşüm + son `(c + 8) >> 4` yuvarlaması.
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn pass2_8<const KIND: u8>(c: &mut [$V; 8]) {
                let r4 = $set1_16(2048);
                match KIND {
                    0 => {
                        idct8(c);
                        for v in c.iter_mut() {
                            *v = $mulhrs16(*v, r4);
                        }
                    }
                    1 => {
                        iadst8(c);
                        adst_finish(c, 2048, false);
                    }
                    3 => {
                        iadst8(c);
                        adst_finish(c, 2048, true);
                    }
                    _ => {
                        let r3 = $set1_16(4096);
                        for v in c.iter_mut() {
                            *v = $mulhrs16(*v, r3);
                        }
                    }
                }
            }

            /// 2. geçiş: 16 noktalı dikey dönüşüm + son `(c + 8) >> 4` yuvarlaması.
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn pass2_16_eob<const KIND: u8>(c: &mut [$V; 16], half: bool) {
                let r4 = $set1_16(2048);
                match KIND {
                    0 => {
                        if half {
                            idct16_half(c);
                        } else {
                            idct16(c);
                        }
                        for v in c.iter_mut() {
                            *v = $mulhrs16(*v, r4);
                        }
                    }
                    1 => {
                        iadst16(c);
                        adst_finish(c, 2048, false);
                    }
                    3 => {
                        iadst16(c);
                        adst_finish(c, 2048, true);
                    }
                    _ => {
                        let k = $set1_16(27152);
                        for v in c.iter_mut() {
                            let t = $mulhrs16(*v, k);
                            let s = $adds16($adds16(*v, *v), t);
                            *v = $mulhrs16(s, r4);
                        }
                    }
                }
            }

            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn round_shift<const N: usize, const SHIFT: u32>(c: &mut [$V; N]) {
                if SHIFT == 1 {
                    let m = $set1_16(16384);
                    for v in c.iter_mut() {
                        *v = $mulhrs16(*v, m);
                    }
                } else if SHIFT == 2 {
                    let m = $set1_16(8192);
                    for v in c.iter_mut() {
                        *v = $mulhrs16(*v, m);
                    }
                }
            }
        }
    };
}

itx_1d_avx512_impl!(
    x4,
    __m128i,
    true,
    _mm_set1_epi16,
    _mm_set1_epi32,
    _mm_setzero_si128,
    _mm_unpacklo_epi16,
    _mm_unpackhi_epi16,
    _mm_dpwssd_epi32,
    _mm_srai_epi32,
    _mm_packs_epi32,
    _mm_adds_epi16,
    _mm_subs_epi16,
    _mm_mulhrs_epi16,
    _mm_srai_epi16,
    _mm_xor_si128,
    _mm_avg_epu16
);

itx_1d_avx512_impl!(
    x8,
    __m256i,
    true,
    _mm256_set1_epi16,
    _mm256_set1_epi32,
    _mm256_setzero_si256,
    _mm256_unpacklo_epi16,
    _mm256_unpackhi_epi16,
    _mm256_dpwssd_epi32,
    _mm256_srai_epi32,
    _mm256_packs_epi32,
    _mm256_adds_epi16,
    _mm256_subs_epi16,
    _mm256_mulhrs_epi16,
    _mm256_srai_epi16,
    _mm256_xor_si256,
    _mm256_avg_epu16
);

itx_1d_avx512_impl!(
    y16,
    __m256i,
    false,
    _mm256_set1_epi16,
    _mm256_set1_epi32,
    _mm256_setzero_si256,
    _mm256_unpacklo_epi16,
    _mm256_unpackhi_epi16,
    _mm256_dpwssd_epi32,
    _mm256_srai_epi32,
    _mm256_packs_epi32,
    _mm256_adds_epi16,
    _mm256_subs_epi16,
    _mm256_mulhrs_epi16,
    _mm256_srai_epi16,
    _mm256_xor_si256,
    _mm256_avg_epu16
);

itx_1d_avx512_impl!(
    z16,
    __m512i,
    true,
    _mm512_set1_epi16,
    _mm512_set1_epi32,
    _mm512_setzero_si512,
    _mm512_unpacklo_epi16,
    _mm512_unpackhi_epi16,
    _mm512_dpwssd_epi32,
    _mm512_srai_epi32,
    _mm512_packs_epi32,
    _mm512_adds_epi16,
    _mm512_subs_epi16,
    _mm512_mulhrs_epi16,
    _mm512_srai_epi16,
    _mm512_xor_si512,
    _mm512_avg_epu16
);

// ---------------------------------------------------------------------------
// Matris transpoz yardımcıları (`i16`, XMM / Split-YMM / Split-ZMM)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn transpose_4x4_x4(r: &mut [__m128i; 4]) {
    let t0 = _mm_unpacklo_epi16(r[0], r[1]);
    let t1 = _mm_unpacklo_epi16(r[2], r[3]);
    let s0 = _mm_unpacklo_epi32(t0, t1);
    let s1 = _mm_unpackhi_epi32(t0, t1);
    r[0] = s0;
    r[1] = _mm_srli_si128::<8>(s0);
    r[2] = s1;
    r[3] = _mm_srli_si128::<8>(s1);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn transpose_4x4_in_ymm(r0: __m256i, r1: __m256i, r2: __m256i, r3: __m256i) -> [__m256i; 4] {
    let a0 = _mm256_unpacklo_epi16(r0, r1);
    let a1 = _mm256_unpacklo_epi16(r2, r3);
    let b0 = _mm256_unpacklo_epi32(a0, a1);
    let b1 = _mm256_unpackhi_epi32(a0, a1);
    [
        b0,
        _mm256_srli_si256::<8>(b0),
        b1,
        _mm256_srli_si256::<8>(b1),
    ]
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn transpose_8x8_split_ymm(r: &mut [__m256i; 8]) {
    let t = transpose_4x4_in_ymm(r[0], r[1], r[2], r[3]);
    let b = transpose_4x4_in_ymm(r[4], r[5], r[6], r[7]);
    for k in 0..4 {
        r[k] = _mm256_permute2x128_si256::<0x20>(t[k], b[k]);
        r[k + 4] = _mm256_permute2x128_si256::<0x31>(t[k], b[k]);
    }
}

/// Her 128-bit ZMM şeridi içindeki alt `4×4` `i16` alt-matrisini paralel transpozlar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn transpose_4x4_in_zmm(r0: __m512i, r1: __m512i, r2: __m512i, r3: __m512i) -> [__m512i; 4] {
    let a0 = _mm512_unpacklo_epi16(r0, r1);
    let a1 = _mm512_unpacklo_epi16(r2, r3);
    let b0 = _mm512_unpacklo_epi32(a0, a1);
    let b1 = _mm512_unpackhi_epi32(a0, a1);
    [
        b0,
        _mm512_bsrli_epi128::<8>(b0),
        b1,
        _mm512_bsrli_epi128::<8>(b1),
    ]
}

/// `16 × 16` Split-ZMM (`z16`) yazmaçlarını (`m0..m15`) yığın dizisi (`alloca`) açmadan
/// doğrudan `zmm0..zmm31` içinde transpozlar.
macro_rules! transpose_16x16_z16_v {
    ($m0:ident, $m1:ident, $m2:ident, $m3:ident, $m4:ident, $m5:ident, $m6:ident, $m7:ident,
     $m8:ident, $m9:ident, $m10:ident, $m11:ident, $m12:ident, $m13:ident, $m14:ident, $m15:ident) => {{
        macro_rules! tr4 {
            ($r0:expr, $r1:expr, $r2:expr, $r3:expr) => {{
                let a0 = _mm512_unpacklo_epi16($r0, $r1);
                let a1 = _mm512_unpacklo_epi16($r2, $r3);
                let b0 = _mm512_unpacklo_epi32(a0, a1);
                let b1 = _mm512_unpackhi_epi32(a0, a1);
                (
                    b0,
                    _mm512_bsrli_epi128::<8>(b0),
                    b1,
                    _mm512_bsrli_epi128::<8>(b1),
                )
            }};
        }
        let (t0_0, t0_1, t0_2, t0_3) = tr4!($m0, $m1, $m2, $m3);
        let (t1_0, t1_1, t1_2, t1_3) = tr4!($m4, $m5, $m6, $m7);
        let (t2_0, t2_1, t2_2, t2_3) = tr4!($m8, $m9, $m10, $m11);
        let (t3_0, t3_1, t3_2, t3_3) = tr4!($m12, $m13, $m14, $m15);
        macro_rules! mix_k {
            ($a:expr, $b:expr, $c:expr, $d:expr, $o0:ident, $o4:ident, $o8:ident, $o12:ident) => {{
                let u01_lo = _mm512_shuffle_i32x4::<0x44>($a, $b);
                let u23_lo = _mm512_shuffle_i32x4::<0x44>($c, $d);
                let u01_hi = _mm512_shuffle_i32x4::<0xEE>($a, $b);
                let u23_hi = _mm512_shuffle_i32x4::<0xEE>($c, $d);
                $o0 = _mm512_shuffle_i32x4::<0x88>(u01_lo, u23_lo);
                $o4 = _mm512_shuffle_i32x4::<0xDD>(u01_lo, u23_lo);
                $o8 = _mm512_shuffle_i32x4::<0x88>(u01_hi, u23_hi);
                $o12 = _mm512_shuffle_i32x4::<0xDD>(u01_hi, u23_hi);
            }};
        }
        mix_k!(t0_0, t1_0, t2_0, t3_0, $m0, $m4, $m8, $m12);
        mix_k!(t0_1, t1_1, t2_1, t3_1, $m1, $m5, $m9, $m13);
        mix_k!(t0_2, t1_2, t2_2, t3_2, $m2, $m6, $m10, $m14);
        mix_k!(t0_3, t1_3, t2_3, t3_3, $m3, $m7, $m11, $m15);
    }};
}

/// `16` sütun × `4` satır `x4` dizisini `4` satır × `16` sütun Split-ZMM (`z16`) dizisine çevirir.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn transpose_16x4_x4_to_z16(cols: &[__m128i; 16]) -> [__m512i; 4] {
    let pack4 = |a: __m128i, b: __m128i, c: __m128i, d: __m128i| -> __m512i {
        _mm512_inserti64x4::<1>(
            _mm512_castsi256_si512(_mm256_set_m128i(b, a)),
            _mm256_set_m128i(d, c),
        )
    };
    transpose_4x4_in_zmm(
        pack4(cols[0], cols[4], cols[8], cols[12]),
        pack4(cols[1], cols[5], cols[9], cols[13]),
        pack4(cols[2], cols[6], cols[10], cols[14]),
        pack4(cols[3], cols[7], cols[11], cols[15]),
    )
}

/// `8` sütun × `16` satır Split-ZMM (`z16`) dizisini `16` satır × `8` sütun Split-YMM (`x8`) dizisine çevirir.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn transpose_8x16_z16_to_x8(cols: &[__m512i; 8]) -> [__m256i; 16] {
    let t0 = transpose_4x4_in_zmm(cols[0], cols[1], cols[2], cols[3]);
    let t1 = transpose_4x4_in_zmm(cols[4], cols[5], cols[6], cols[7]);
    let mut rows = [_mm256_setzero_si256(); 16];
    for k in 0..4 {
        let t0_lo = _mm512_castsi512_si256(t0[k]);
        let t1_lo = _mm512_castsi512_si256(t1[k]);
        let t0_hi = _mm512_extracti64x4_epi64::<1>(t0[k]);
        let t1_hi = _mm512_extracti64x4_epi64::<1>(t1[k]);
        rows[k] = _mm256_permute2x128_si256::<0x20>(t0_lo, t1_lo);
        rows[k + 4] = _mm256_permute2x128_si256::<0x31>(t0_lo, t1_lo);
        rows[k + 8] = _mm256_permute2x128_si256::<0x20>(t0_hi, t1_hi);
        rows[k + 12] = _mm256_permute2x128_si256::<0x31>(t0_hi, t1_hi);
    }
    rows
}

/// `16` sütun × `8` satır Split-YMM (`x8`) dizisini `8` satır × `16` sütun Split-ZMM (`z16`) dizisine çevirir.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn transpose_16x8_x8_to_z16(cols: &[__m256i; 16]) -> [__m512i; 8] {
    let t0 = transpose_4x4_in_ymm(cols[0], cols[1], cols[2], cols[3]);
    let t1 = transpose_4x4_in_ymm(cols[4], cols[5], cols[6], cols[7]);
    let t2 = transpose_4x4_in_ymm(cols[8], cols[9], cols[10], cols[11]);
    let t3 = transpose_4x4_in_ymm(cols[12], cols[13], cols[14], cols[15]);
    let mut rows = [_mm512_setzero_si512(); 8];
    for k in 0..4 {
        let z01 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(t0[k]), t1[k]);
        let z23 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(t2[k]), t3[k]);
        rows[k] = _mm512_shuffle_i32x4::<0x88>(z01, z23);
        rows[k + 4] = _mm512_shuffle_i32x4::<0xDD>(z01, z23);
    }
    rows
}

// ---------------------------------------------------------------------------
// Sınır-güvenli piksel ekleme ve saklama yardımcıları (`0` gather/scatter)
// ---------------------------------------------------------------------------

/// `w = 4`: her satırda tam 4 bayt okur, `del` alt 4 `i16` şeridini ekler, `[0, 255]` kırpıp 4 bayt yazar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_w4<const H: usize>(dst: &mut [u8], stride: usize, rows: &[__m128i; H]) {
    const { assert!(H.is_multiple_of(4)) };
    let mut y = 0;
    while y < H {
        let b0 = y * stride;
        let b1 = b0 + stride;
        let b2 = b1 + stride;
        let b3 = b2 + stride;
        let d0 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, b0)));
        let d1 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, b1)));
        let d2 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, b2)));
        let d3 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, b3)));
        let px01 = _mm_cvtepu8_epi16(_mm_unpacklo_epi32(d0, d1));
        let px23 = _mm_cvtepu8_epi16(_mm_unpacklo_epi32(d2, d3));
        let del01 = _mm_unpacklo_epi64(rows[y], rows[y + 1]);
        let del23 = _mm_unpacklo_epi64(rows[y + 2], rows[y + 3]);
        let p = _mm_packus_epi16(_mm_add_epi16(px01, del01), _mm_add_epi16(px23, del23));
        store_u8x4(chunk_mut(dst, b0), p);
        store_u8x4(chunk_mut(dst, b1), _mm_srli_si128::<4>(p));
        store_u8x4(chunk_mut(dst, b2), _mm_srli_si128::<8>(p));
        store_u8x4(chunk_mut(dst, b3), _mm_srli_si128::<12>(p));
        y += 4;
    }
}

/// `w = 8` (bitişik `__m128i`): her satırda tam 8 bayt okur, `8 × i16` ekler, `[0, 255]` kırpıp 8 bayt yazar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_w8<const H: usize>(dst: &mut [u8], stride: usize, rows: &[__m128i; H]) {
    const { assert!(H.is_multiple_of(2)) };
    let mut y = 0;
    while y < H {
        let d0 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, y * stride)));
        let d1 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, (y + 1) * stride)));
        let s0 = _mm_add_epi16(_mm_cvtepu8_epi16(d0), rows[y]);
        let s1 = _mm_add_epi16(_mm_cvtepu8_epi16(d1), rows[y + 1]);
        let p = _mm_packus_epi16(s0, s1);
        store_u8x8(chunk_mut(dst, y * stride), p);
        store_u8x8(chunk_mut(dst, (y + 1) * stride), _mm_srli_si128::<8>(p));
        y += 2;
    }
}

/// `w = 8` (Split-YMM `__m256i`: `[x0..3, _ || x4..7, _]`): her satırda tam 8 bayt okur, ekler ve yazar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_w8_split<const H: usize>(dst: &mut [u8], stride: usize, rows: &[__m256i; H]) {
    const { assert!(H.is_multiple_of(2)) };
    let mut y = 0;
    while y < H {
        let d0 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, y * stride)));
        let d1 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, (y + 1) * stride)));
        let px01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(d0, d1));
        let del01 = _mm256_permute4x64_epi64::<0xD8>(_mm256_unpacklo_epi64(rows[y], rows[y + 1]));
        let s = _mm256_add_epi16(px01, del01);
        let p = _mm_packus_epi16(_mm256_castsi256_si128(s), _mm256_extracti128_si256::<1>(s));
        store_u8x8(chunk_mut(dst, y * stride), p);
        store_u8x8(chunk_mut(dst, (y + 1) * stride), _mm_srli_si128::<8>(p));
        y += 2;
    }
}

/// İki Split-ZMM (`z16`) satırını (`r0, r1`) değerle (`zmm0, zmm1`) alıp `dst` üzerine yazar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_w16_z16_cift(dst: &mut [u8], stride: usize, y: usize, r0: __m512i, r1: __m512i) {
    let perm = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let d0 = load_u8x16(chunk(dst, y * stride));
    let d1 = load_u8x16(chunk(dst, (y + 1) * stride));
    let px01 = _mm512_cvtepu8_epi16(_mm256_set_m128i(d1, d0));
    let del01 = _mm512_permutexvar_epi64(perm, _mm512_unpacklo_epi64(r0, r1));
    let s = _mm512_add_epi16(px01, del01);
    let p = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(
        _mm512_castsi512_si256(s),
        _mm512_extracti64x4_epi64::<1>(s),
    ));
    store_u8x16(chunk_mut(dst, y * stride), _mm256_castsi256_si128(p));
    store_u8x16(
        chunk_mut(dst, (y + 1) * stride),
        _mm256_extracti128_si256::<1>(p),
    );
}

/// `w = 16` (`H ∈ {4, 8}`, Split-ZMM `z16`): ikişer satır birleştirip yazar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_w16_split_zmm<const H: usize>(dst: &mut [u8], stride: usize, rows: &[__m512i; H]) {
    const { assert!(H.is_multiple_of(2)) };
    let mut y = 0;
    while y < H {
        ekle_yaz_w16_z16_cift(dst, stride, y, rows[y], rows[y + 1]);
        y += 2;
    }
}

/// DC-only hızlı yolu (`txtp == DCT_DCT && eob == 0`, `src/itx_tmpl.c:58-70`).
///
/// Yalnız `coeff[0] = 0` yapar; `coeff[1..1024]` öğelerine dokunmaz.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn dc_only<const W: usize, const H: usize, const SHIFT: u32, const RECT2: bool>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32; 1024],
) {
    let mut dc = i64::from(coeff[0]);
    coeff[0] = 0;
    if RECT2 {
        dc = (dc * 181 + 128) >> 8;
    }
    dc = (dc * 181 + 128) >> 8;
    let rnd = (1i64 << SHIFT) >> 1;
    dc = (dc + rnd) >> SHIFT;
    dc = (dc * 181 + 128 + 2048) >> 12;
    let va = _mm_set1_epi8(dc.clamp(0, 255) as u8 as i8);
    let vs = _mm_set1_epi8((-dc).clamp(0, 255) as u8 as i8);

    if W == 4 {
        let mut y = 0;
        while y < H {
            let d0 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, y * stride)));
            let d1 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, (y + 1) * stride)));
            let p = _mm_subs_epu8(_mm_adds_epu8(_mm_unpacklo_epi32(d0, d1), va), vs);
            store_u8x4(chunk_mut(dst, y * stride), p);
            store_u8x4(chunk_mut(dst, (y + 1) * stride), _mm_srli_si128::<4>(p));
            y += 2;
        }
    } else if W == 8 {
        let mut y = 0;
        while y < H {
            let d0 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, y * stride)));
            let d1 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, (y + 1) * stride)));
            let p = _mm_subs_epu8(_mm_adds_epu8(_mm_unpacklo_epi64(d0, d1), va), vs);
            store_u8x8(chunk_mut(dst, y * stride), p);
            store_u8x8(chunk_mut(dst, (y + 1) * stride), _mm_srli_si128::<8>(p));
            y += 2;
        }
    } else {
        for y in 0..H {
            let off = y * stride;
            let d = load_u8x16(chunk(dst, off));
            store_u8x16(chunk_mut(dst, off), _mm_subs_epu8(_mm_adds_epu8(d, va), vs));
        }
    }
}

// ---------------------------------------------------------------------------
// Boyut bazında 2B AVX512ICL çekirdek gövdeleri (`w, h ≤ 16`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn cekirdek_4x4<const H1D: u8, const V1D: u8>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32; 1024],
    _eob: usize,
) {
    if H1D == 2 && V1D == 2 {
        let k = _mm256_set1_epi16(13576);
        let r4 = _mm256_set1_epi16(2048);
        let v = _mm256_packs_epi32(yukle_i32x8(chunk(coeff, 0)), yukle_i32x8(chunk(coeff, 8)));
        sifirla_katsayi::<16>(coeff);
        let v1 = _mm256_adds_epi16(v, _mm256_mulhrs_epi16(v, k));
        let v2 = _mm256_mulhrs_epi16(_mm256_adds_epi16(v1, _mm256_mulhrs_epi16(v1, k)), r4);
        let lo = _mm256_castsi256_si128(v2);
        let hi = _mm256_extracti128_si256::<1>(v2);
        let a0 = _mm_unpacklo_epi16(lo, hi);
        let a1 = _mm_unpackhi_epi16(lo, hi);
        let r01 = _mm_unpacklo_epi32(a0, a1);
        let r23 = _mm_unpackhi_epi32(a0, a1);
        let rows = [r01, _mm_srli_si128::<8>(r01), r23, _mm_srli_si128::<8>(r23)];
        ekle_yaz_w4(dst, stride, &rows);
        return;
    }
    let mut c = [
        yukle_i16x4(chunk(coeff, 0)),
        yukle_i16x4(chunk(coeff, 4)),
        yukle_i16x4(chunk(coeff, 8)),
        yukle_i16x4(chunk(coeff, 12)),
    ];
    sifirla_katsayi::<16>(coeff);
    x4::pass1_4::<H1D, 0>(&mut c);
    transpose_4x4_x4(&mut c);
    x4::pass2_4::<V1D>(&mut c);
    ekle_yaz_w4(dst, stride, &c);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_w4_y8_packed(dst: &mut [u8], stride: usize, y01_45: __m256i, y23_67: __m256i) {
    let b0 = 0;
    let b1 = stride;
    let b2 = 2 * stride;
    let b3 = 3 * stride;
    let b4 = 4 * stride;
    let b5 = 5 * stride;
    let b6 = 6 * stride;
    let b7 = 7 * stride;
    let d0 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, b0)));
    let d1 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, b1)));
    let d2 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, b2)));
    let d3 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, b3)));
    let d4 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, b4)));
    let d5 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, b5)));
    let d6 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, b6)));
    let d7 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, b7)));
    let px01_45 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(
        _mm_unpacklo_epi32(d0, d1),
        _mm_unpacklo_epi32(d4, d5),
    ));
    let px23_67 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(
        _mm_unpacklo_epi32(d2, d3),
        _mm_unpacklo_epi32(d6, d7),
    ));
    let p = _mm256_packus_epi16(
        _mm256_add_epi16(px01_45, y01_45),
        _mm256_add_epi16(px23_67, y23_67),
    );
    let p03 = _mm256_castsi256_si128(p);
    let p47 = _mm256_extracti128_si256::<1>(p);
    store_u8x4(chunk_mut(dst, b0), p03);
    store_u8x4(chunk_mut(dst, b1), _mm_srli_si128::<4>(p03));
    store_u8x4(chunk_mut(dst, b2), _mm_srli_si128::<8>(p03));
    store_u8x4(chunk_mut(dst, b3), _mm_srli_si128::<12>(p03));
    store_u8x4(chunk_mut(dst, b4), p47);
    store_u8x4(chunk_mut(dst, b5), _mm_srli_si128::<4>(p47));
    store_u8x4(chunk_mut(dst, b6), _mm_srli_si128::<8>(p47));
    store_u8x4(chunk_mut(dst, b7), _mm_srli_si128::<12>(p47));
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn cekirdek_4x8<const H1D: u8, const V1D: u8>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32; 1024],
    _eob: usize,
) {
    if H1D == 2 && V1D == 0 {
        let r2 = _mm256_set1_epi16(23168);
        let k = _mm256_set1_epi16(13576);
        let c01 = _mm256_mulhrs_epi16(
            _mm256_packs_epi32(yukle_i32x8(chunk(coeff, 0)), yukle_i32x8(chunk(coeff, 8))),
            r2,
        );
        let c23 = _mm256_mulhrs_epi16(
            _mm256_packs_epi32(yukle_i32x8(chunk(coeff, 16)), yukle_i32x8(chunk(coeff, 24))),
            r2,
        );
        sifirla_katsayi::<32>(coeff);
        let s01 = _mm256_adds_epi16(c01, _mm256_mulhrs_epi16(c01, k));
        let s23 = _mm256_adds_epi16(c23, _mm256_mulhrs_epi16(c23, k));
        // `s01` = `[col0[0..3], col1[0..3] || col0[4..7], col1[4..7]]`
        // `s23` = `[col2[0..3], col3[0..3] || col2[4..7], col3[4..7]]`
        let a0 = _mm256_unpacklo_epi16(s01, s23); // `[r0(c0,c2), r1(c0,c2), r2(c0,c2), r3(c0,c2) || r4..r7(c0,c2)]`
        let a1 = _mm256_unpackhi_epi16(s01, s23); // `[r0(c1,c3), r1(c1,c3), r2(c1,c3), r3(c1,c3) || r4..r7(c1,c3)]`
        let u0 = _mm256_unpacklo_epi16(a0, a1); // `[r0(4), r1(4) || r4(4), r5(4)]`
        let u1 = _mm256_unpackhi_epi16(a0, a1); // `[r2(4), r3(4) || r6(4), r7(4)]`
        let r01 = _mm256_castsi256_si128(u0);
        let r45 = _mm256_extracti128_si256::<1>(u0);
        let r23 = _mm256_castsi256_si128(u1);
        let r67 = _mm256_extracti128_si256::<1>(u1);

        let c04 = _mm_unpacklo_epi64(r01, r45); // `[r0(4), r4(4)]`
        let c26 = _mm_unpacklo_epi64(r23, r67); // `[r2(4), r6(4)]`
        let c15 = _mm_unpackhi_epi64(r01, r45); // `[r1(4), r5(4)]`
        let c37 = _mm_unpackhi_epi64(r23, r67); // `[r3(4), r7(4)]`

        let rnd = _mm_set1_epi32(2048);
        // Even: r0, r4 in c04; r2, r6 in c26
        // We need `idct4` on `(r0, r2, r4, r6)`:
        // `mulsub(r0, r4, 2896, 2896)` and `mulsub(r2, r6, 1567, 3784)`!
        let p04 = _mm_unpacklo_epi16(_mm_srli_si128::<8>(c04), c04); // `(r4, r0)` pairs
        let t1_e = _mm_srai_epi32::<12>(_mm_dpwssd_epi32(
            rnd,
            p04,
            _mm_set1_epi32(pair(-2896, 2896)),
        ));
        let t0_e =
            _mm_srai_epi32::<12>(_mm_dpwssd_epi32(rnd, p04, _mm_set1_epi32(pair(2896, 2896))));
        let t01_e = _mm_packs_epi32(t0_e, t1_e); // `[t0(4), t1(4)]`

        let p26 = _mm_unpacklo_epi16(_mm_srli_si128::<8>(c26), c26); // `(r6, r2)` pairs
        let t2_e = _mm_srai_epi32::<12>(_mm_dpwssd_epi32(
            rnd,
            p26,
            _mm_set1_epi32(pair(-3784, 1567)),
        ));
        let t3_e =
            _mm_srai_epi32::<12>(_mm_dpwssd_epi32(rnd, p26, _mm_set1_epi32(pair(1567, 3784))));
        let t32_e = _mm_packs_epi32(t3_e, t2_e); // `[t3(4), t2(4)]`

        let e01 = _mm_adds_epi16(t01_e, t32_e); // `[e0 = t0+t3, e1 = t1+t2]`
        let e32 = _mm_subs_epi16(t01_e, t32_e); // `[e3 = t0-t3, e2 = t1-t2]`

        // Odd: `mulsub(r1, r7, 799, 4017)` and `mulsub(r5, r3, 3406, 2276)`
        let p17 = _mm_unpacklo_epi16(_mm_srli_si128::<8>(c37), c15); // `(r7, r1)` pairs
        let t4a =
            _mm_srai_epi32::<12>(_mm_dpwssd_epi32(rnd, p17, _mm_set1_epi32(pair(-4017, 799))));
        let t7a = _mm_srai_epi32::<12>(_mm_dpwssd_epi32(rnd, p17, _mm_set1_epi32(pair(799, 4017))));
        let t74a = _mm_packs_epi32(t7a, t4a); // `[t7a(4), t4a(4)]`

        let p53 = _mm_unpacklo_epi16(c37, _mm_srli_si128::<8>(c15)); // `(r3, r5)` pairs
        let t5a = _mm_srai_epi32::<12>(_mm_dpwssd_epi32(
            rnd,
            p53,
            _mm_set1_epi32(pair(-2276, 3406)),
        ));
        let t6a =
            _mm_srai_epi32::<12>(_mm_dpwssd_epi32(rnd, p53, _mm_set1_epi32(pair(3406, 2276))));
        let t65a = _mm_packs_epi32(t6a, t5a); // `[t6a(4), t5a(4)]`

        let o74 = _mm_adds_epi16(t74a, t65a); // `[t7 = t7a+t6a, t4 = t4a+t5a]`
        let d65 = _mm_subs_epi16(t74a, t65a); // `[t6a' = t7a-t6a, t5a' = t4a-t5a]`

        let p65 = _mm_unpacklo_epi16(_mm_srli_si128::<8>(d65), d65); // `(t5a', t6a')` pairs
        let t5 = _mm_srai_epi32::<12>(_mm_dpwssd_epi32(
            rnd,
            p65,
            _mm_set1_epi32(pair(-2896, 2896)),
        ));
        let t6 = _mm_srai_epi32::<12>(_mm_dpwssd_epi32(rnd, p65, _mm_set1_epi32(pair(2896, 2896))));
        let o65 = _mm_packs_epi32(t6, t5); // `[t6(4), t5(4)]`
        let o56 = _mm_packs_epi32(t5, t6); // `[t5(4), t6(4)]`

        let o76 = _mm_unpacklo_epi64(o74, o65); // `[t7(4), t6(4)]`
        let o54 = _mm_unpacklo_epi64(o56, _mm_srli_si128::<8>(o74)); // `[t5(4), t4(4)]`
        let o45 = _mm_unpacklo_epi64(_mm_srli_si128::<8>(o74), o56); // `[t4(4), t5(4)]`
        let o67 = _mm_unpacklo_epi64(o65, o74); // `[t6(4), t7(4)]`
        let e23 = _mm_unpacklo_epi64(_mm_srli_si128::<8>(e32), e32); // `[e2(4), e3(4)]`
        let e10 = _mm_unpacklo_epi64(_mm_srli_si128::<8>(e01), e01); // `[e1(4), e0(4)]`

        let r4 = _mm256_set1_epi16(2048);
        let y01_45 = _mm256_mulhrs_epi16(
            _mm256_set_m128i(_mm_subs_epi16(e32, o45), _mm_adds_epi16(e01, o76)),
            r4,
        );
        let y23_67 = _mm256_mulhrs_epi16(
            _mm256_set_m128i(_mm_subs_epi16(e10, o67), _mm_adds_epi16(e23, o54)),
            r4,
        );
        ekle_yaz_w4_y8_packed(dst, stride, y01_45, y23_67);
        return;
    }
    if H1D == 2 && V1D != 2 {
        let r2 = _mm256_set1_epi16(23168);
        let k = _mm256_set1_epi16(13576);
        // `yukle_i32x8` in-lane `_mm256_packs_epi32`:
        // `c02` alt 128-bit = `[col0[0..3], col2[0..3]]`, üst 128-bit = `[col0[4..7], col2[4..7]]`
        // `c13` alt 128-bit = `[col1[0..3], col3[0..3]]`, üst 128-bit = `[col1[4..7], col3[4..7]]`
        let c02 = _mm256_mulhrs_epi16(
            _mm256_packs_epi32(yukle_i32x8(chunk(coeff, 0)), yukle_i32x8(chunk(coeff, 16))),
            r2,
        );
        let c13 = _mm256_mulhrs_epi16(
            _mm256_packs_epi32(yukle_i32x8(chunk(coeff, 8)), yukle_i32x8(chunk(coeff, 24))),
            r2,
        );
        sifirla_katsayi::<32>(coeff);
        let s02 = _mm256_adds_epi16(c02, _mm256_mulhrs_epi16(c02, k));
        let s13 = _mm256_adds_epi16(c13, _mm256_mulhrs_epi16(c13, k));
        let a0 = _mm256_unpacklo_epi16(s02, s13); // `[row0_c01, row1_c01, row2_c01, row3_c01 || row4_c01..row7_c01]`
        let a1 = _mm256_unpackhi_epi16(s02, s13); // `[row0_c23, row1_c23, row2_c23, row3_c23 || row4_c23..row7_c23]`
        let r01_45 = _mm256_unpacklo_epi32(a0, a1); // `[row0, row1 || row4, row5]`
        let r23_67 = _mm256_unpackhi_epi32(a0, a1); // `[row2, row3 || row6, row7]`
        let r01 = _mm256_castsi256_si128(r01_45);
        let r45 = _mm256_extracti128_si256::<1>(r01_45);
        let r23 = _mm256_castsi256_si128(r23_67);
        let r67 = _mm256_extracti128_si256::<1>(r23_67);
        let mut rows = [
            r01,
            _mm_srli_si128::<8>(r01),
            r23,
            _mm_srli_si128::<8>(r23),
            r45,
            _mm_srli_si128::<8>(r45),
            r67,
            _mm_srli_si128::<8>(r67),
        ];
        x4::pass2_8::<V1D>(&mut rows);
        ekle_yaz_w4(dst, stride, &rows);
        return;
    }
    let r2 = _mm256_set1_epi16(23168);
    let mut cols = [
        _mm256_mulhrs_epi16(yukle_i16x8(chunk(coeff, 0)), r2),
        _mm256_mulhrs_epi16(yukle_i16x8(chunk(coeff, 8)), r2),
        _mm256_mulhrs_epi16(yukle_i16x8(chunk(coeff, 16)), r2),
        _mm256_mulhrs_epi16(yukle_i16x8(chunk(coeff, 24)), r2),
    ];
    sifirla_katsayi::<32>(coeff);
    x8::pass1_4::<H1D, 0>(&mut cols);
    if V1D == 2 {
        let r3 = _mm256_set1_epi16(4096);
        for v in &mut cols {
            *v = _mm256_mulhrs_epi16(*v, r3);
        }
        let t = transpose_4x4_in_ymm(cols[0], cols[1], cols[2], cols[3]);
        let rows = [
            _mm256_castsi256_si128(t[0]),
            _mm256_castsi256_si128(t[1]),
            _mm256_castsi256_si128(t[2]),
            _mm256_castsi256_si128(t[3]),
            _mm256_extracti128_si256::<1>(t[0]),
            _mm256_extracti128_si256::<1>(t[1]),
            _mm256_extracti128_si256::<1>(t[2]),
            _mm256_extracti128_si256::<1>(t[3]),
        ];
        ekle_yaz_w4(dst, stride, &rows);
        return;
    }
    let t = transpose_4x4_in_ymm(cols[0], cols[1], cols[2], cols[3]);
    let mut rows = [
        _mm256_castsi256_si128(t[0]),
        _mm256_castsi256_si128(t[1]),
        _mm256_castsi256_si128(t[2]),
        _mm256_castsi256_si128(t[3]),
        _mm256_extracti128_si256::<1>(t[0]),
        _mm256_extracti128_si256::<1>(t[1]),
        _mm256_extracti128_si256::<1>(t[2]),
        _mm256_extracti128_si256::<1>(t[3]),
    ];
    x4::pass2_8::<V1D>(&mut rows);
    ekle_yaz_w4(dst, stride, &rows);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn cekirdek_8x4<const H1D: u8, const V1D: u8>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32; 1024],
    _eob: usize,
) {
    if H1D == 2 {
        let (r2, z) = (_mm256_set1_epi16(23168), _mm256_setzero_si256());
        let p0123 = _mm256_mulhrs_epi16(
            _mm256_packs_epi32(yukle_i32x8(chunk(coeff, 0)), yukle_i32x8(chunk(coeff, 8))),
            r2,
        );
        let p4567 = _mm256_mulhrs_epi16(
            _mm256_packs_epi32(yukle_i32x8(chunk(coeff, 16)), yukle_i32x8(chunk(coeff, 24))),
            r2,
        );
        sifirla_katsayi::<32>(coeff);
        let s0 = _mm256_adds_epi16(p0123, p0123);
        let s4 = _mm256_adds_epi16(p4567, p4567);
        let c04 = _mm256_permute2x128_si256::<0x20>(s0, s4);
        let c15 = _mm256_permute2x128_si256::<0x31>(s0, s4);
        let a0 = _mm256_unpacklo_epi16(c04, c15);
        let a1 = _mm256_unpackhi_epi16(c04, c15);
        let (u0, u1) = (_mm256_unpacklo_epi32(a0, a1), _mm256_unpackhi_epi32(a0, a1));
        let mut rows = [
            _mm256_unpacklo_epi64(u0, z),
            _mm256_unpackhi_epi64(u0, z),
            _mm256_unpacklo_epi64(u1, z),
            _mm256_unpackhi_epi64(u1, z),
        ];
        x8::pass2_4::<V1D>(&mut rows);
        ekle_yaz_w8_split(dst, stride, &rows);
        return;
    }
    let r2 = _mm_set1_epi16(23168);
    let mut cols = [
        _mm_mulhrs_epi16(yukle_i16x4(chunk(coeff, 0)), r2),
        _mm_mulhrs_epi16(yukle_i16x4(chunk(coeff, 4)), r2),
        _mm_mulhrs_epi16(yukle_i16x4(chunk(coeff, 8)), r2),
        _mm_mulhrs_epi16(yukle_i16x4(chunk(coeff, 12)), r2),
        _mm_mulhrs_epi16(yukle_i16x4(chunk(coeff, 16)), r2),
        _mm_mulhrs_epi16(yukle_i16x4(chunk(coeff, 20)), r2),
        _mm_mulhrs_epi16(yukle_i16x4(chunk(coeff, 24)), r2),
        _mm_mulhrs_epi16(yukle_i16x4(chunk(coeff, 28)), r2),
    ];
    sifirla_katsayi::<32>(coeff);
    x4::pass1_8::<H1D, 0>(&mut cols);
    let mut rows = transpose_4x4_in_ymm(
        _mm256_set_m128i(cols[4], cols[0]),
        _mm256_set_m128i(cols[5], cols[1]),
        _mm256_set_m128i(cols[6], cols[2]),
        _mm256_set_m128i(cols[7], cols[3]),
    );
    x8::pass2_4::<V1D>(&mut rows);
    ekle_yaz_w8_split(dst, stride, &rows);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn cekirdek_8x8<const H1D: u8, const V1D: u8>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32; 1024],
    _eob: usize,
) {
    if H1D == 2 && V1D == 2 {
        let r3 = _mm256_set1_epi16(4096);
        let p0 = _mm256_mulhrs_epi16(
            _mm256_packs_epi32(yukle_i32x8(chunk(coeff, 0)), yukle_i32x8(chunk(coeff, 32))),
            r3,
        );
        let p1 = _mm256_mulhrs_epi16(
            _mm256_packs_epi32(yukle_i32x8(chunk(coeff, 8)), yukle_i32x8(chunk(coeff, 40))),
            r3,
        );
        let p2 = _mm256_mulhrs_epi16(
            _mm256_packs_epi32(yukle_i32x8(chunk(coeff, 16)), yukle_i32x8(chunk(coeff, 48))),
            r3,
        );
        let p3 = _mm256_mulhrs_epi16(
            _mm256_packs_epi32(yukle_i32x8(chunk(coeff, 24)), yukle_i32x8(chunk(coeff, 56))),
            r3,
        );
        sifirla_katsayi::<64>(coeff);
        let a0 = _mm256_unpacklo_epi16(p0, p1);
        let a1 = _mm256_unpackhi_epi16(p0, p1);
        let a2 = _mm256_unpacklo_epi16(p2, p3);
        let a3 = _mm256_unpackhi_epi16(p2, p3);
        let b0 = _mm256_unpacklo_epi32(a0, a2);
        let b1 = _mm256_unpackhi_epi32(a0, a2);
        let b2 = _mm256_unpacklo_epi32(a1, a3);
        let b3 = _mm256_unpackhi_epi32(a1, a3);
        let r04 = _mm256_unpacklo_epi64(b0, b2);
        let r15 = _mm256_unpackhi_epi64(b0, b2);
        let r26 = _mm256_unpacklo_epi64(b1, b3);
        let r37 = _mm256_unpackhi_epi64(b1, b3);
        let rows = [
            _mm256_castsi256_si128(r04),
            _mm256_castsi256_si128(r15),
            _mm256_castsi256_si128(r26),
            _mm256_castsi256_si128(r37),
            _mm256_extracti128_si256::<1>(r04),
            _mm256_extracti128_si256::<1>(r15),
            _mm256_extracti128_si256::<1>(r26),
            _mm256_extracti128_si256::<1>(r37),
        ];
        ekle_yaz_w8(dst, stride, &rows);
        return;
    }
    let mut m = [
        yukle_i16x8(chunk(coeff, 0)),
        yukle_i16x8(chunk(coeff, 8)),
        yukle_i16x8(chunk(coeff, 16)),
        yukle_i16x8(chunk(coeff, 24)),
        yukle_i16x8(chunk(coeff, 32)),
        yukle_i16x8(chunk(coeff, 40)),
        yukle_i16x8(chunk(coeff, 48)),
        yukle_i16x8(chunk(coeff, 56)),
    ];
    sifirla_katsayi::<64>(coeff);
    x8::pass1_8::<H1D, 1>(&mut m);
    transpose_8x8_split_ymm(&mut m);
    x8::pass2_8::<V1D>(&mut m);
    ekle_yaz_w8_split(dst, stride, &m);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn cekirdek_4x16<const H1D: u8, const V1D: u8>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32; 1024],
    eob: usize,
) {
    let half = H1D == 0 && V1D == 0 && eob <= 28;
    if half {
        let mut cols = [
            yukle_i16x8(chunk(coeff, 0)),
            yukle_i16x8(chunk(coeff, 16)),
            yukle_i16x8(chunk(coeff, 32)),
            yukle_i16x8(chunk(coeff, 48)),
        ];
        sifirla_katsayi::<64>(coeff);
        x8::pass1_4::<0, 1>(&mut cols);
        let t = transpose_4x4_in_ymm(cols[0], cols[1], cols[2], cols[3]);
        let z = _mm_setzero_si128();
        let mut rows = [
            _mm256_castsi256_si128(t[0]),
            _mm256_castsi256_si128(t[1]),
            _mm256_castsi256_si128(t[2]),
            _mm256_castsi256_si128(t[3]),
            _mm256_extracti128_si256::<1>(t[0]),
            _mm256_extracti128_si256::<1>(t[1]),
            _mm256_extracti128_si256::<1>(t[2]),
            _mm256_extracti128_si256::<1>(t[3]),
            z,
            z,
            z,
            z,
            z,
            z,
            z,
            z,
        ];
        x4::pass2_16_eob::<0>(&mut rows, true);
        ekle_yaz_w4(dst, stride, &rows);
        return;
    }

    let mut cols = [
        _mm256_packs_epi32(yukle_i32x8(chunk(coeff, 0)), yukle_i32x8(chunk(coeff, 8))),
        _mm256_packs_epi32(yukle_i32x8(chunk(coeff, 16)), yukle_i32x8(chunk(coeff, 24))),
        _mm256_packs_epi32(yukle_i32x8(chunk(coeff, 32)), yukle_i32x8(chunk(coeff, 40))),
        _mm256_packs_epi32(yukle_i32x8(chunk(coeff, 48)), yukle_i32x8(chunk(coeff, 56))),
    ];
    sifirla_katsayi::<64>(coeff);
    y16::pass1_4::<H1D, 1>(&mut cols);
    if V1D == 2 {
        let k = _mm256_set1_epi16(27152);
        let r4 = _mm256_set1_epi16(2048);
        for v in &mut cols {
            let t = _mm256_mulhrs_epi16(*v, k);
            let s = _mm256_adds_epi16(_mm256_adds_epi16(*v, *v), t);
            *v = _mm256_mulhrs_epi16(s, r4);
        }
    }
    let a0 = _mm256_unpacklo_epi16(cols[0], cols[1]);
    let a1 = _mm256_unpackhi_epi16(cols[0], cols[1]);
    let a2 = _mm256_unpacklo_epi16(cols[2], cols[3]);
    let a3 = _mm256_unpackhi_epi16(cols[2], cols[3]);
    let r01_45 = _mm256_unpacklo_epi32(a0, a2);
    let r23_67 = _mm256_unpackhi_epi32(a0, a2);
    let r89_cd = _mm256_unpacklo_epi32(a1, a3);
    let rab_ef = _mm256_unpackhi_epi32(a1, a3);
    let lo01 = _mm256_castsi256_si128(r01_45);
    let hi45 = _mm256_extracti128_si256::<1>(r01_45);
    let lo23 = _mm256_castsi256_si128(r23_67);
    let hi67 = _mm256_extracti128_si256::<1>(r23_67);
    let lo89 = _mm256_castsi256_si128(r89_cd);
    let hicd = _mm256_extracti128_si256::<1>(r89_cd);
    let loab = _mm256_castsi256_si128(rab_ef);
    let hief = _mm256_extracti128_si256::<1>(rab_ef);
    let mut rows = [
        lo01,
        _mm_srli_si128::<8>(lo01),
        lo23,
        _mm_srli_si128::<8>(lo23),
        hi45,
        _mm_srli_si128::<8>(hi45),
        hi67,
        _mm_srli_si128::<8>(hi67),
        lo89,
        _mm_srli_si128::<8>(lo89),
        loab,
        _mm_srli_si128::<8>(loab),
        hicd,
        _mm_srli_si128::<8>(hicd),
        hief,
        _mm_srli_si128::<8>(hief),
    ];
    if V1D != 2 {
        x4::pass2_16_eob::<V1D>(&mut rows, false);
    }
    ekle_yaz_w4(dst, stride, &rows);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn cekirdek_16x4<const H1D: u8, const V1D: u8>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32; 1024],
    eob: usize,
) {
    if H1D == 2 {
        let mut cols = [_mm_setzero_si128(); 16];
        for (x, col) in cols.iter_mut().enumerate() {
            *col = yukle_i16x4(chunk(coeff, x * 4));
        }
        sifirla_katsayi::<64>(coeff);
        let mut rows = transpose_16x4_x4_to_z16(&cols);
        let k16 = _mm512_set1_epi16(27152);
        let r1 = _mm512_set1_epi16(16384);
        for v in &mut rows {
            *v = _mm512_adds_epi16(*v, _mm512_mulhrs_epi16(_mm512_mulhrs_epi16(*v, k16), r1));
        }
        z16::pass2_4::<V1D>(&mut rows);
        ekle_yaz_w16_split_zmm(dst, stride, &rows);
        return;
    }

    let half = H1D == 0 && V1D == 0 && eob <= 28;
    let mut cols = [_mm_setzero_si128(); 16];
    let n_load = if half { 8 } else { 16 };
    for (x, col) in cols.iter_mut().enumerate().take(n_load) {
        *col = yukle_i16x4(chunk(coeff, x * 4));
    }
    sifirla_katsayi::<64>(coeff);
    x4::pass1_16_eob::<H1D, 1>(&mut cols, half);

    let mut rows = transpose_16x4_x4_to_z16(&cols);
    z16::pass2_4::<V1D>(&mut rows);
    ekle_yaz_w16_split_zmm(dst, stride, &rows);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn cekirdek_8x16<const H1D: u8, const V1D: u8>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32; 1024],
    eob: usize,
) {
    let half = H1D == 0 && V1D == 0 && eob <= 42;
    if half {
        let r2 = _mm256_set1_epi16(23168);
        let mut cols = [_mm256_setzero_si256(); 8];
        for (x, col) in cols.iter_mut().enumerate() {
            *col = _mm256_mulhrs_epi16(yukle_i16x8(chunk(coeff, x * 16)), r2);
        }
        sifirla_katsayi::<128>(coeff);
        x8::pass1_8::<0, 1>(&mut cols);
        transpose_8x8_split_ymm(&mut cols);
        let mut rows = [_mm256_setzero_si256(); 16];
        rows[..8].copy_from_slice(&cols);
        x8::pass2_16_eob::<0>(&mut rows, true);
        ekle_yaz_w8_split(dst, stride, &rows);
        return;
    }

    let r2 = _mm512_set1_epi16(23168);
    let mut cols = [_mm512_setzero_si512(); 8];
    for (x, col) in cols.iter_mut().enumerate() {
        *col = _mm512_mulhrs_epi16(yukle_z16(chunk(coeff, x * 16)), r2);
    }
    sifirla_katsayi::<128>(coeff);
    z16::pass1_8::<H1D, 1>(&mut cols);
    if V1D == 2 {
        let k = _mm512_set1_epi16(27152);
        let r4 = _mm512_set1_epi16(2048);
        for v in &mut cols {
            let t = _mm512_mulhrs_epi16(*v, k);
            let s = _mm512_adds_epi16(_mm512_adds_epi16(*v, *v), t);
            *v = _mm512_mulhrs_epi16(s, r4);
        }
        let rows = transpose_8x16_z16_to_x8(&cols);
        ekle_yaz_w8_split(dst, stride, &rows);
        return;
    }
    let mut rows = transpose_8x16_z16_to_x8(&cols);
    x8::pass2_16_eob::<V1D>(&mut rows, false);
    ekle_yaz_w8_split(dst, stride, &rows);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn cekirdek_16x8<const H1D: u8, const V1D: u8>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32; 1024],
    eob: usize,
) {
    if H1D == 2 {
        let mut cols = [_mm256_setzero_si256(); 16];
        for (x, col) in cols.iter_mut().enumerate() {
            *col = yukle_i16x8(chunk(coeff, x * 8));
        }
        sifirla_katsayi::<128>(coeff);
        let mut rows = transpose_16x8_x8_to_z16(&cols);
        let r2 = _mm512_set1_epi16(23168);
        let k16 = _mm512_set1_epi16(27152);
        let r1 = _mm512_set1_epi16(16384);
        for v in &mut rows {
            let v0 = _mm512_mulhrs_epi16(*v, r2);
            let t = _mm512_mulhrs_epi16(v0, k16);
            *v = _mm512_adds_epi16(v0, _mm512_mulhrs_epi16(t, r1));
        }
        z16::pass2_8::<V1D>(&mut rows);
        ekle_yaz_w16_split_zmm(dst, stride, &rows);
        return;
    }

    let half = H1D == 0 && V1D == 0 && eob <= 42;
    let r2 = _mm256_set1_epi16(23168);
    let mut cols = [_mm256_setzero_si256(); 16];
    let n_load = if half { 8 } else { 16 };
    for (x, col) in cols.iter_mut().enumerate().take(n_load) {
        *col = _mm256_mulhrs_epi16(yukle_i16x8(chunk(coeff, x * 8)), r2);
    }
    sifirla_katsayi::<128>(coeff);
    x8::pass1_16_eob::<H1D, 1>(&mut cols, half);

    let mut rows = transpose_16x8_x8_to_z16(&cols);
    z16::pass2_8::<V1D>(&mut rows);
    ekle_yaz_w16_split_zmm(dst, stride, &rows);
}

/// `cekirdek_16x16` için 16 adet ZMM yazmacı (`m0..m15`) üzerinde yığınsız (`0` `alloca`) 1B geçişler.
macro_rules! z16_16_gecis {
    ($kind:expr, $pass:literal, $m0:ident, $m1:ident, $m2:ident, $m3:ident, $m4:ident, $m5:ident, $m6:ident, $m7:ident,
     $m8:ident, $m9:ident, $m10:ident, $m11:ident, $m12:ident, $m13:ident, $m14:ident, $m15:ident) => {{
        let sm_val: i16 = if $pass == 1 { 8192 } else { 2048 };
        match $kind {
            0 => {
                // `idct16` (`even` 8 + `odd` 8) doğrudan ZMM yazmaçlarında:
                let (e_t1, e_t0) = z16::mulsub($m0, $m8, 2896, 2896);
                let (e_t2, e_t3) = z16::mulsub($m4, $m12, 1567, 3784);
                let ee0 = _mm512_adds_epi16(e_t0, e_t3);
                let ee1 = _mm512_adds_epi16(e_t1, e_t2);
                let ee2 = _mm512_subs_epi16(e_t1, e_t2);
                let ee3 = _mm512_subs_epi16(e_t0, e_t3);
                let (eo4a, eo7a) = z16::mulsub($m2, $m14, 799, 4017);
                let (eo5a, eo6a) = z16::mulsub($m10, $m6, 3406, 2276);
                let eo4 = _mm512_adds_epi16(eo4a, eo5a);
                let eo5_in = _mm512_subs_epi16(eo4a, eo5a);
                let eo7 = _mm512_adds_epi16(eo7a, eo6a);
                let eo6_in = _mm512_subs_epi16(eo7a, eo6a);
                let (eo5, eo6) = z16::mulsub(eo6_in, eo5_in, 2896, 2896);
                let t0 = _mm512_adds_epi16(ee0, eo7);
                let t1 = _mm512_adds_epi16(ee1, eo6);
                let t2 = _mm512_adds_epi16(ee2, eo5);
                let t3 = _mm512_adds_epi16(ee3, eo4);
                let t4 = _mm512_subs_epi16(ee3, eo4);
                let t5 = _mm512_subs_epi16(ee2, eo5);
                let t6 = _mm512_subs_epi16(ee1, eo6);
                let t7 = _mm512_subs_epi16(ee0, eo7);

                let (t8a, t15a) = z16::mulsub($m1, $m15, 401, 4076);
                let (t9a, t14a) = z16::mulsub($m9, $m7, 3166, 2598);
                let (t10a, t13a) = z16::mulsub($m5, $m11, 1931, 3612);
                let (t11a, t12a) = z16::mulsub($m13, $m3, 3920, 1189);

                let t8 = _mm512_adds_epi16(t8a, t9a);
                let t9 = _mm512_subs_epi16(t8a, t9a);
                let t10 = _mm512_subs_epi16(t11a, t10a);
                let t11 = _mm512_adds_epi16(t11a, t10a);
                let t12 = _mm512_adds_epi16(t12a, t13a);
                let t13 = _mm512_subs_epi16(t12a, t13a);
                let t14 = _mm512_subs_epi16(t15a, t14a);
                let t15 = _mm512_adds_epi16(t15a, t14a);

                let (t9a, t14a) = z16::mulsub(t14, t9, 1567, 3784);
                let (t10a, t13a) = z16::mulsub(t13, t10, -3784, 1567);

                let t8a = _mm512_adds_epi16(t8, t11);
                let t9 = _mm512_adds_epi16(t9a, t10a);
                let t10 = _mm512_subs_epi16(t9a, t10a);
                let t11a = _mm512_subs_epi16(t8, t11);
                let t12a = _mm512_subs_epi16(t15, t12);
                let t13 = _mm512_subs_epi16(t14a, t13a);
                let t14 = _mm512_adds_epi16(t14a, t13a);
                let t15a = _mm512_adds_epi16(t15, t12);

                let (t10a, t13a) = z16::mulsub(t13, t10, 2896, 2896);
                let (t11, t12) = z16::mulsub(t12a, t11a, 2896, 2896);

                let sm = _mm512_set1_epi16(sm_val);
                $m0 = _mm512_mulhrs_epi16(_mm512_adds_epi16(t0, t15a), sm);
                $m1 = _mm512_mulhrs_epi16(_mm512_adds_epi16(t1, t14), sm);
                $m2 = _mm512_mulhrs_epi16(_mm512_adds_epi16(t2, t13a), sm);
                $m3 = _mm512_mulhrs_epi16(_mm512_adds_epi16(t3, t12), sm);
                $m4 = _mm512_mulhrs_epi16(_mm512_adds_epi16(t4, t11), sm);
                $m5 = _mm512_mulhrs_epi16(_mm512_adds_epi16(t5, t10a), sm);
                $m6 = _mm512_mulhrs_epi16(_mm512_adds_epi16(t6, t9), sm);
                $m7 = _mm512_mulhrs_epi16(_mm512_adds_epi16(t7, t8a), sm);
                $m8 = _mm512_mulhrs_epi16(_mm512_subs_epi16(t7, t8a), sm);
                $m9 = _mm512_mulhrs_epi16(_mm512_subs_epi16(t6, t9), sm);
                $m10 = _mm512_mulhrs_epi16(_mm512_subs_epi16(t5, t10a), sm);
                $m11 = _mm512_mulhrs_epi16(_mm512_subs_epi16(t4, t11), sm);
                $m12 = _mm512_mulhrs_epi16(_mm512_subs_epi16(t3, t12), sm);
                $m13 = _mm512_mulhrs_epi16(_mm512_subs_epi16(t2, t13a), sm);
                $m14 = _mm512_mulhrs_epi16(_mm512_subs_epi16(t1, t14), sm);
                $m15 = _mm512_mulhrs_epi16(_mm512_subs_epi16(t0, t15a), sm);
            }
            1 | 3 => {
                let (t1, t0) = z16::mulsub($m15, $m0, 201, 4091);
                let (t3, t2) = z16::mulsub($m13, $m2, 995, 3973);
                let (t5, t4) = z16::mulsub($m11, $m4, 1751, 3703);
                let (t7, t6) = z16::mulsub($m9, $m6, 2440, 3290);
                let (t9, t8) = z16::mulsub($m7, $m8, 3035, 2751);
                let (t11, t10) = z16::mulsub($m5, $m10, 3513, 2106);
                let (t13, t12) = z16::mulsub($m3, $m12, 3857, 1380);
                let (t15, t14) = z16::mulsub($m1, $m14, 4052, 601);

                let t0a = _mm512_adds_epi16(t0, t8);
                let t8a = _mm512_subs_epi16(t0, t8);
                let t1a = _mm512_adds_epi16(t1, t9);
                let t9a = _mm512_subs_epi16(t1, t9);
                let t2a = _mm512_adds_epi16(t2, t10);
                let t10a = _mm512_subs_epi16(t2, t10);
                let t3a = _mm512_adds_epi16(t3, t11);
                let t11a = _mm512_subs_epi16(t3, t11);
                let t4a = _mm512_adds_epi16(t4, t12);
                let t12a = _mm512_subs_epi16(t4, t12);
                let t5a = _mm512_adds_epi16(t5, t13);
                let t13a = _mm512_subs_epi16(t5, t13);
                let t6a = _mm512_adds_epi16(t6, t14);
                let t14a = _mm512_subs_epi16(t6, t14);
                let t7a = _mm512_adds_epi16(t7, t15);
                let t15a = _mm512_subs_epi16(t7, t15);

                let (t9, t8) = z16::mulsub(t8a, t9a, 799, 4017);
                let (t11, t10) = z16::mulsub(t10a, t11a, 3406, 2276);
                let (t12, t13) = z16::mulsub(t13a, t12a, 4017, 799);
                let (t14, t15) = z16::mulsub(t15a, t14a, 2276, 3406);

                let t0 = _mm512_adds_epi16(t0a, t4a);
                let t4 = _mm512_subs_epi16(t0a, t4a);
                let t1 = _mm512_adds_epi16(t1a, t5a);
                let t5 = _mm512_subs_epi16(t1a, t5a);
                let t2 = _mm512_adds_epi16(t2a, t6a);
                let t6 = _mm512_subs_epi16(t2a, t6a);
                let t3 = _mm512_adds_epi16(t3a, t7a);
                let t7 = _mm512_subs_epi16(t3a, t7a);
                let t8a = _mm512_adds_epi16(t8, t12);
                let t12a = _mm512_subs_epi16(t8, t12);
                let t9a = _mm512_adds_epi16(t9, t13);
                let t13a = _mm512_subs_epi16(t9, t13);
                let t10a = _mm512_adds_epi16(t10, t14);
                let t14a = _mm512_subs_epi16(t10, t14);
                let t11a = _mm512_adds_epi16(t11, t15);
                let t15a = _mm512_subs_epi16(t11, t15);

                let (t5a, t4a) = z16::mulsub(t4, t5, 1567, 3784);
                let (t6a, t7a) = z16::mulsub(t7, t6, 3784, 1567);
                let (t13, t12) = z16::mulsub(t12a, t13a, 1567, 3784);
                let (t14, t15) = z16::mulsub(t15a, t14a, 3784, 1567);

                let u0 = _mm512_adds_epi16(t0, t2);
                let u15 = _mm512_adds_epi16(t1, t3);
                let t2a = _mm512_subs_epi16(t0, t2);
                let t3a = _mm512_subs_epi16(t1, t3);
                let u3 = _mm512_adds_epi16(t4a, t6a);
                let u12 = _mm512_adds_epi16(t5a, t7a);
                let t6 = _mm512_subs_epi16(t4a, t6a);
                let t7 = _mm512_subs_epi16(t5a, t7a);
                let u1 = _mm512_adds_epi16(t8a, t10a);
                let u14 = _mm512_adds_epi16(t9a, t11a);
                let t10 = _mm512_subs_epi16(t8a, t10a);
                let t11 = _mm512_subs_epi16(t9a, t11a);
                let u2 = _mm512_adds_epi16(t12, t14);
                let u13 = _mm512_adds_epi16(t13, t15);
                let t14a = _mm512_subs_epi16(t12, t14);
                let t15a = _mm512_subs_epi16(t13, t15);

                let (u8, u7) = z16::mulsub(t2a, t3a, 2896, 2896);
                let (u11, u4) = z16::mulsub(t6, t7, 2896, 2896);
                let (u9, u6) = z16::mulsub(t10, t11, 2896, 2896);
                let (u10, u5) = z16::mulsub(t14a, t15a, 2896, 2896);

                let p = _mm512_set1_epi16(sm_val);
                let n = _mm512_set1_epi16(-sm_val);
                let r0 = _mm512_mulhrs_epi16(u0, p);
                let r1 = _mm512_mulhrs_epi16(u1, n);
                let r2 = _mm512_mulhrs_epi16(u2, p);
                let r3 = _mm512_mulhrs_epi16(u3, n);
                let r4 = _mm512_mulhrs_epi16(u4, p);
                let r5 = _mm512_mulhrs_epi16(u5, n);
                let r6 = _mm512_mulhrs_epi16(u6, p);
                let r7 = _mm512_mulhrs_epi16(u7, n);
                let r8 = _mm512_mulhrs_epi16(u8, p);
                let r9 = _mm512_mulhrs_epi16(u9, n);
                let r10 = _mm512_mulhrs_epi16(u10, p);
                let r11 = _mm512_mulhrs_epi16(u11, n);
                let r12 = _mm512_mulhrs_epi16(u12, p);
                let r13 = _mm512_mulhrs_epi16(u13, n);
                let r14 = _mm512_mulhrs_epi16(u14, p);
                let r15 = _mm512_mulhrs_epi16(u15, n);
                if $kind == 3 {
                    $m0 = r15;
                    $m1 = r14;
                    $m2 = r13;
                    $m3 = r12;
                    $m4 = r11;
                    $m5 = r10;
                    $m6 = r9;
                    $m7 = r8;
                    $m8 = r7;
                    $m9 = r6;
                    $m10 = r5;
                    $m11 = r4;
                    $m12 = r3;
                    $m13 = r2;
                    $m14 = r1;
                    $m15 = r0;
                } else {
                    $m0 = r0;
                    $m1 = r1;
                    $m2 = r2;
                    $m3 = r3;
                    $m4 = r4;
                    $m5 = r5;
                    $m6 = r6;
                    $m7 = r7;
                    $m8 = r8;
                    $m9 = r9;
                    $m10 = r10;
                    $m11 = r11;
                    $m12 = r12;
                    $m13 = r13;
                    $m14 = r14;
                    $m15 = r15;
                }
            }
            _ => {
                let k = _mm512_set1_epi16(27152);
                let r4 = _mm512_set1_epi16(2048);
                macro_rules! id16_one {
                    ($v:ident) => {{
                        let t = _mm512_mulhrs_epi16($v, k);
                        $v = if $pass == 1 {
                            _mm512_avg_epu16($v, _mm512_srai_epi16::<1>(t))
                        } else {
                            let s = _mm512_adds_epi16(_mm512_adds_epi16($v, $v), t);
                            _mm512_mulhrs_epi16(s, r4)
                        };
                    }};
                }
                id16_one!($m0);
                id16_one!($m1);
                id16_one!($m2);
                id16_one!($m3);
                id16_one!($m4);
                id16_one!($m5);
                id16_one!($m6);
                id16_one!($m7);
                id16_one!($m8);
                id16_one!($m9);
                id16_one!($m10);
                id16_one!($m11);
                id16_one!($m12);
                id16_one!($m13);
                id16_one!($m14);
                id16_one!($m15);
            }
        }
    }};
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn cekirdek_16x16_dct_qtr(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    let mut m0 = yukle_z16(chunk(coeff, 0));
    let mut m1 = yukle_z16(chunk(coeff, 16));
    let mut m2 = yukle_z16(chunk(coeff, 32));
    let mut m3 = yukle_z16(chunk(coeff, 48));
    let mut m4 = yukle_z16(chunk(coeff, 64));
    let mut m5 = yukle_z16(chunk(coeff, 80));
    let mut m6 = yukle_z16(chunk(coeff, 96));
    let mut m7 = yukle_z16(chunk(coeff, 112));
    let z = _mm512_setzero_si512();
    let (mut m8, mut m9, mut m10, mut m11, mut m12, mut m13, mut m14, mut m15) =
        (z, z, z, z, z, z, z, z);
    z16_16_gecis!(
        0, 1, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15
    );
    transpose_16x16_z16_v!(
        m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15
    );
    z16_16_gecis!(
        0, 2, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15
    );
    ekle_yaz_w16_z16_cift(dst, stride, 0, m0, m1);
    ekle_yaz_w16_z16_cift(dst, stride, 2, m2, m3);
    ekle_yaz_w16_z16_cift(dst, stride, 4, m4, m5);
    ekle_yaz_w16_z16_cift(dst, stride, 6, m6, m7);
    ekle_yaz_w16_z16_cift(dst, stride, 8, m8, m9);
    ekle_yaz_w16_z16_cift(dst, stride, 10, m10, m11);
    ekle_yaz_w16_z16_cift(dst, stride, 12, m12, m13);
    ekle_yaz_w16_z16_cift(dst, stride, 14, m14, m15);
    sifirla_katsayi::<256>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn cekirdek_16x16_gecis1<const H1D: u8>(coeff: &mut [i32; 1024]) {
    let mut m0 = yukle_z16(chunk(coeff, 0));
    let mut m1 = yukle_z16(chunk(coeff, 16));
    let mut m2 = yukle_z16(chunk(coeff, 32));
    let mut m3 = yukle_z16(chunk(coeff, 48));
    let mut m4 = yukle_z16(chunk(coeff, 64));
    let mut m5 = yukle_z16(chunk(coeff, 80));
    let mut m6 = yukle_z16(chunk(coeff, 96));
    let mut m7 = yukle_z16(chunk(coeff, 112));
    let mut m8 = yukle_z16(chunk(coeff, 128));
    let mut m9 = yukle_z16(chunk(coeff, 144));
    let mut m10 = yukle_z16(chunk(coeff, 160));
    let mut m11 = yukle_z16(chunk(coeff, 176));
    let mut m12 = yukle_z16(chunk(coeff, 192));
    let mut m13 = yukle_z16(chunk(coeff, 208));
    let mut m14 = yukle_z16(chunk(coeff, 224));
    let mut m15 = yukle_z16(chunk(coeff, 240));
    z16_16_gecis!(
        H1D, 1, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15
    );
    transpose_16x16_z16_v!(
        m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15
    );
    macro_rules! st_z16 {
        ($off:literal, $v:ident) => {{
            store_i32x8(chunk_mut(coeff, $off), _mm512_castsi512_si256($v));
            store_i32x8(
                chunk_mut(coeff, $off + 8),
                _mm512_extracti64x4_epi64::<1>($v),
            );
        }};
    }
    st_z16!(0, m0);
    st_z16!(16, m1);
    st_z16!(32, m2);
    st_z16!(48, m3);
    st_z16!(64, m4);
    st_z16!(80, m5);
    st_z16!(96, m6);
    st_z16!(112, m7);
    st_z16!(128, m8);
    st_z16!(144, m9);
    st_z16!(160, m10);
    st_z16!(176, m11);
    st_z16!(192, m12);
    st_z16!(208, m13);
    st_z16!(224, m14);
    st_z16!(240, m15);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn cekirdek_16x16_gecis2<const V1D: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    let mut m0 = yukle_i32x16(chunk(coeff, 0));
    let mut m1 = yukle_i32x16(chunk(coeff, 16));
    let mut m2 = yukle_i32x16(chunk(coeff, 32));
    let mut m3 = yukle_i32x16(chunk(coeff, 48));
    let mut m4 = yukle_i32x16(chunk(coeff, 64));
    let mut m5 = yukle_i32x16(chunk(coeff, 80));
    let mut m6 = yukle_i32x16(chunk(coeff, 96));
    let mut m7 = yukle_i32x16(chunk(coeff, 112));
    let mut m8 = yukle_i32x16(chunk(coeff, 128));
    let mut m9 = yukle_i32x16(chunk(coeff, 144));
    let mut m10 = yukle_i32x16(chunk(coeff, 160));
    let mut m11 = yukle_i32x16(chunk(coeff, 176));
    let mut m12 = yukle_i32x16(chunk(coeff, 192));
    let mut m13 = yukle_i32x16(chunk(coeff, 208));
    let mut m14 = yukle_i32x16(chunk(coeff, 224));
    let mut m15 = yukle_i32x16(chunk(coeff, 240));
    z16_16_gecis!(
        V1D, 2, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15
    );
    ekle_yaz_w16_z16_cift(dst, stride, 0, m0, m1);
    ekle_yaz_w16_z16_cift(dst, stride, 2, m2, m3);
    ekle_yaz_w16_z16_cift(dst, stride, 4, m4, m5);
    ekle_yaz_w16_z16_cift(dst, stride, 6, m6, m7);
    ekle_yaz_w16_z16_cift(dst, stride, 8, m8, m9);
    ekle_yaz_w16_z16_cift(dst, stride, 10, m10, m11);
    ekle_yaz_w16_z16_cift(dst, stride, 12, m12, m13);
    ekle_yaz_w16_z16_cift(dst, stride, 14, m14, m15);
    sifirla_katsayi::<256>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn cekirdek_16x16<const H1D: u8, const V1D: u8>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32; 1024],
    eob: usize,
) {
    if H1D == 2 && V1D == 2 {
        let k = _mm512_set1_epi16(27152);
        let r4 = _mm512_set1_epi16(2048);
        macro_rules! yukle_idtx {
            ($off:expr) => {{
                let v0 = yukle_z16(chunk(coeff, $off));
                let t1 = _mm512_mulhrs_epi16(v0, k);
                let v1 = _mm512_avg_epu16(v0, _mm512_srai_epi16::<1>(t1));
                let t2 = _mm512_mulhrs_epi16(v1, k);
                let s2 = _mm512_adds_epi16(_mm512_adds_epi16(v1, v1), t2);
                _mm512_mulhrs_epi16(s2, r4)
            }};
        }
        let mut m0 = yukle_idtx!(0);
        let mut m1 = yukle_idtx!(16);
        let mut m2 = yukle_idtx!(32);
        let mut m3 = yukle_idtx!(48);
        let mut m4 = yukle_idtx!(64);
        let mut m5 = yukle_idtx!(80);
        let mut m6 = yukle_idtx!(96);
        let mut m7 = yukle_idtx!(112);
        let mut m8 = yukle_idtx!(128);
        let mut m9 = yukle_idtx!(144);
        let mut m10 = yukle_idtx!(160);
        let mut m11 = yukle_idtx!(176);
        let mut m12 = yukle_idtx!(192);
        let mut m13 = yukle_idtx!(208);
        let mut m14 = yukle_idtx!(224);
        let mut m15 = yukle_idtx!(240);
        transpose_16x16_z16_v!(
            m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15
        );
        ekle_yaz_w16_z16_cift(dst, stride, 0, m0, m1);
        ekle_yaz_w16_z16_cift(dst, stride, 2, m2, m3);
        ekle_yaz_w16_z16_cift(dst, stride, 4, m4, m5);
        ekle_yaz_w16_z16_cift(dst, stride, 6, m6, m7);
        ekle_yaz_w16_z16_cift(dst, stride, 8, m8, m9);
        ekle_yaz_w16_z16_cift(dst, stride, 10, m10, m11);
        ekle_yaz_w16_z16_cift(dst, stride, 12, m12, m13);
        ekle_yaz_w16_z16_cift(dst, stride, 14, m14, m15);
        sifirla_katsayi::<256>(coeff);
        return;
    }

    if H1D == 0 && V1D == 0 && eob <= 35 {
        cekirdek_16x16_dct_qtr(dst, stride, coeff);
        return;
    }

    cekirdek_16x16_gecis1::<H1D>(coeff);
    cekirdek_16x16_gecis2::<V1D>(dst, stride, coeff);
}

#[inline(always)]
const fn tx1d_kod(t: Tx1dType) -> u8 {
    t as u8
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn itx8_avx512_cekirdek<const TX: u8, const TXTP: u8>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32; 1024],
    eob: usize,
) {
    let tx = const { TxSize::from_index(TX as usize).unwrap() };
    let txtp = const { TxType::from_index(TXTP as usize).unwrap() };
    let w = const { TxSize::from_index(TX as usize).unwrap().width() };
    let h = const { TxSize::from_index(TX as usize).unwrap().height() };
    let shift = const { TxSize::from_index(TX as usize).unwrap().shift() };
    let rect2 = const {
        let t = TxSize::from_index(TX as usize).unwrap();
        t.width() * 2 == t.height() || t.height() * 2 == t.width()
    };

    if matches!(txtp, TxType::DctDct) && eob == 0 {
        match (w, h, shift, rect2) {
            (4, 4, 0, false) => dc_only::<4, 4, 0, false>(dst, dst_stride, coeff),
            (4, 8, 0, true) => dc_only::<4, 8, 0, true>(dst, dst_stride, coeff),
            (8, 4, 0, true) => dc_only::<8, 4, 0, true>(dst, dst_stride, coeff),
            (8, 8, 1, false) => dc_only::<8, 8, 1, false>(dst, dst_stride, coeff),
            (4, 16, 1, false) => dc_only::<4, 16, 1, false>(dst, dst_stride, coeff),
            (16, 4, 1, false) => dc_only::<16, 4, 1, false>(dst, dst_stride, coeff),
            (8, 16, 1, true) => dc_only::<8, 16, 1, true>(dst, dst_stride, coeff),
            (16, 8, 1, true) => dc_only::<16, 8, 1, true>(dst, dst_stride, coeff),
            (16, 16, 2, false) => dc_only::<16, 16, 2, false>(dst, dst_stride, coeff),
            _ => unreachable!(),
        }
        return;
    }

    let h1d = const { tx1d_kod(TxType::from_index(TXTP as usize).unwrap().tx1d().0) };
    let v1d = const { tx1d_kod(TxType::from_index(TXTP as usize).unwrap().tx1d().1) };

    macro_rules! cagir {
        ($fn_ad:ident) => {
            match (h1d, v1d) {
                (0, 0) => $fn_ad::<0, 0>(dst, dst_stride, coeff, eob),
                (0, 1) => $fn_ad::<0, 1>(dst, dst_stride, coeff, eob),
                (0, 2) => $fn_ad::<0, 2>(dst, dst_stride, coeff, eob),
                (0, 3) => $fn_ad::<0, 3>(dst, dst_stride, coeff, eob),
                (1, 0) => $fn_ad::<1, 0>(dst, dst_stride, coeff, eob),
                (1, 1) => $fn_ad::<1, 1>(dst, dst_stride, coeff, eob),
                (1, 2) => $fn_ad::<1, 2>(dst, dst_stride, coeff, eob),
                (1, 3) => $fn_ad::<1, 3>(dst, dst_stride, coeff, eob),
                (2, 0) => $fn_ad::<2, 0>(dst, dst_stride, coeff, eob),
                (2, 1) => $fn_ad::<2, 1>(dst, dst_stride, coeff, eob),
                (2, 2) => $fn_ad::<2, 2>(dst, dst_stride, coeff, eob),
                (2, 3) => $fn_ad::<2, 3>(dst, dst_stride, coeff, eob),
                (3, 0) => $fn_ad::<3, 0>(dst, dst_stride, coeff, eob),
                (3, 1) => $fn_ad::<3, 1>(dst, dst_stride, coeff, eob),
                (3, 2) => $fn_ad::<3, 2>(dst, dst_stride, coeff, eob),
                (3, 3) => $fn_ad::<3, 3>(dst, dst_stride, coeff, eob),
                _ => unreachable!(),
            }
        };
    }

    match tx {
        TxSize::Tx4x4 => cagir!(cekirdek_4x4),
        TxSize::Rtx4x8 => cagir!(cekirdek_4x8),
        TxSize::Rtx8x4 => cagir!(cekirdek_8x4),
        TxSize::Tx8x8 => cagir!(cekirdek_8x8),
        TxSize::Rtx4x16 => cagir!(cekirdek_4x16),
        TxSize::Rtx16x4 => cagir!(cekirdek_16x4),
        TxSize::Rtx8x16 => cagir!(cekirdek_8x16),
        TxSize::Rtx16x8 => cagir!(cekirdek_16x8),
        TxSize::Tx16x16 => cagir!(cekirdek_16x16),
        _ => unreachable!(),
    }
}

/// 8 bpc AVX512ICL `w, h ≤ 16` genel `ItxFn` giriş sarmalayıcısı (monomorfize `(TX, TXTP)`).
fn itx8_avx512_giris<const TX: u8, const TXTP: u8>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
    bit_depth: BitDepth,
) {
    if bit_depth != BitDepth::Eight
        || tx as u8 != TX
        || txtp as u8 != TXTP
        || matches!(txtp, TxType::WhtWht)
        || !is_supported(tx, txtp)
        || !itx8_sozlesmesi(dst.len(), dst_stride, coeff.len(), eob, tx, txtp)
    {
        itx_skaler_devret::<8>(dst, dst_stride, coeff, eob, tx, txtp);
        return;
    }
    let coeff1024 = chunk_mut::<i32, 1024>(coeff, 0);
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`itx8_avx512_fn`); belirteç AVX512ICL (F/BW/VL/VBMI/VNNI) tespiti + maske izninin
    // kanıtıdır. (2) sınır: `itx8_sozlesmesi` `dst`'nin `(h − 1)·dst_stride + w` baytı ve
    // `coeff`'un `1024` öğeyi kapsadığını garanti eder; çekirdek içindeki bütün erişimler
    // sabit boyutlu `chunk`/`chunk_mut` dizileri üzerindendir. (3) hizalama/takma ad: `dst`
    // ve `coeff1024` ayrık `&mut` referanslardır, yalnız hizasız erişim kullanılır.
    unsafe {
        itx8_avx512_cekirdek::<TX, TXTP>(dst, dst_stride, coeff1024, eob);
    }
}

/// 140 adet `w, h ≤ 16` WHT-dışı 8 bpc AVX512ICL `ItxFn` tablosu (yalnız [`Avx512IclToken`] ile).
#[must_use]
pub fn itx8_avx512_fn(_token: Avx512IclToken, tx: TxSize, txtp: TxType) -> Option<ItxFn> {
    if matches!(txtp, TxType::WhtWht) || !is_supported(tx, txtp) {
        return None;
    }
    macro_rules! turler16 {
        ($tx_c:literal) => {
            match txtp {
                TxType::DctDct => Some(itx8_avx512_giris::<$tx_c, 0>),
                TxType::AdstDct => Some(itx8_avx512_giris::<$tx_c, 1>),
                TxType::DctAdst => Some(itx8_avx512_giris::<$tx_c, 2>),
                TxType::AdstAdst => Some(itx8_avx512_giris::<$tx_c, 3>),
                TxType::FlipadstDct => Some(itx8_avx512_giris::<$tx_c, 4>),
                TxType::DctFlipadst => Some(itx8_avx512_giris::<$tx_c, 5>),
                TxType::FlipadstFlipadst => Some(itx8_avx512_giris::<$tx_c, 6>),
                TxType::AdstFlipadst => Some(itx8_avx512_giris::<$tx_c, 7>),
                TxType::FlipadstAdst => Some(itx8_avx512_giris::<$tx_c, 8>),
                TxType::Idtx => Some(itx8_avx512_giris::<$tx_c, 9>),
                TxType::VDct => Some(itx8_avx512_giris::<$tx_c, 10>),
                TxType::HDct => Some(itx8_avx512_giris::<$tx_c, 11>),
                TxType::VAdst => Some(itx8_avx512_giris::<$tx_c, 12>),
                TxType::HAdst => Some(itx8_avx512_giris::<$tx_c, 13>),
                TxType::VFlipadst => Some(itx8_avx512_giris::<$tx_c, 14>),
                TxType::HFlipadst => Some(itx8_avx512_giris::<$tx_c, 15>),
                TxType::WhtWht => None,
            }
        };
    }
    match tx {
        TxSize::Tx4x4 => turler16!(0),
        TxSize::Tx8x8 => turler16!(1),
        TxSize::Tx16x16 => turler16!(2),
        TxSize::Rtx4x8 => turler16!(5),
        TxSize::Rtx8x4 => turler16!(6),
        TxSize::Rtx8x16 => turler16!(7),
        TxSize::Rtx16x8 => turler16!(8),
        TxSize::Rtx4x16 => turler16!(13),
        TxSize::Rtx16x4 => turler16!(14),
        _ => None,
    }
}
