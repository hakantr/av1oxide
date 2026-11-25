//! 8 bpc büyük blok ters dönüşüm (`max(w, h) ∈ {32, 64}`) x86_64 AVX2 çekirdekleri — F4-ITX a (P43 Adım 2).
//!
//! Bu modül dav1d'nin `src/x86/itx_avx2.asm` ve `src/x86/itx.h:214-226` dosyalarındaki
//! `max(w, h) ∈ {32, 64}` boyutlu **15** adet 8 bpc AVX2 ters dönüşüm kaydını saf kararlı
//! Rust `core::arch::x86_64` intrinsics ile gerçekler (`.asm`, `.S`, `asm!`, `global_asm!`
//! ve dış C bağımlılığı yoktur; CEK-02):
//!
//! - `RTX_8X32`, `RTX_32X8`, `RTX_16X32`, `RTX_32X16`, `TX_32X32`: `DctDct` ve `Idtx` (`5 × 2 = 10`),
//! - `RTX_16X64`, `RTX_64X16`, `RTX_32X64`, `RTX_64X32`, `TX_64X64`: `DctDct` (`5 × 1 = 5`).
//!
//! ## Sözleşme, P34 izolasyonu, `eob` hızlı yolları ve ISA sınırı
//!
//! - **P34 modül izolasyonu:** Bu modül `itx` ya da `itx_wht` ile hiçbir
//!   `#[target_feature]` yardımcısı paylaşmaz; tüm 1B DCT/IDTX ve transpoz yardımcıları
//!   modül içinde kendi kopyalarıyla tanımlıdır.
//! - **`eob` çeyrek / alt-blok hızlı yolları (`itx_avx2.asm` eşikleriyle birebir):**
//!   - `DctDct && eob == 0`: DC-only hızlı yolu (**yalnız `coeff[0] = 0`** sıfırlanır;
//!     `coeff[1..1024]` öğelerine dokunulmaz).
//!   - `8×32` ve `32×8` (`DctDct` ve `Idtx`): `eob ≤ 106` alt yarı sıfır hızlı yolu.
//!   - `16×32 DctDct`, `16×64 DctDct`, `32×16 DctDct`, `64×16 DctDct`: `eob ≤ 150` yarı blok hızlı yolu.
//!   - `16×32 Idtx`: `eob ≤ 43`, `≤ 150`, `≤ 278` şerit hızlı yolları; `32×16 Idtx`: `eob ≤ 35`, `≤ 150`.
//!   - `32×32` (`DctDct` ve `Idtx`), `32×64`, `64×32`, `64×64`: `eob < 136` sol-üst `16×16` hızlı yolu.
//! - **Genel yol katsayı sıfırlama sözleşmesi:** `eob ≥ 1` veya `txtp != DctDct` olduğunda
//!   alt-blok hızlı yolu seçilse bile `coeff[..sw * sh]` (`sw = min(w, 32)`, `sh = min(h, 32)`)
//!   tamamen sıfırlanır; `coeff[sw * sh..1024]` kuyruğuna dokunulmaz.
//! - **Sınır güvenliği ve ISA sınırı:** Her satırda tam `w` bayt (`8`, `16`, `32` veya `64`)
//!   okunur/yazılır; tekil `dst_stride` ve `SentinelBuffer` altında taşma `0`'dır. Yalnız
//!   `#[target_feature(enable = "avx2")]` kullanılır (`0` EVEX yönergesi); tek `unsafe`
//!   bloğu [`itx8_buyuk_avx2_giris`] içindedir.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_adds_epu8, _mm_cvtepu8_epi16, _mm_cvtsi64_si128,
    _mm_packus_epi16, _mm_set1_epi8, _mm_setzero_si128, _mm_srli_si128, _mm_subs_epu8,
    _mm_unpacklo_epi64, _mm256_add_epi16, _mm256_add_epi32, _mm256_adds_epi16, _mm256_adds_epu8,
    _mm256_castsi256_si128, _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_madd_epi16,
    _mm256_mulhrs_epi16, _mm256_packs_epi32, _mm256_packus_epi16, _mm256_permute2x128_si256,
    _mm256_permute4x64_epi64, _mm256_set1_epi8, _mm256_set1_epi16, _mm256_set1_epi32,
    _mm256_setr_epi32, _mm256_setzero_si256, _mm256_srai_epi32, _mm256_srli_si256,
    _mm256_subs_epi16, _mm256_subs_epu8, _mm256_unpackhi_epi16, _mm256_unpackhi_epi32,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi16, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
};

use super::mem::{
    chunk, chunk_mut, load_u8x16, load_u8x32, store_i32x8, store_u8x8, store_u8x16, store_u8x32,
};
use crate::cpu::Avx2Token;
use crate::dsp::itx::{ItxFn, TxSize, TxType, itx_skaler_devret, itx8_sozlesmesi};
use crate::frame::BitDepth;

/// İki `i16` katsayısını `_mm*_madd_epi16` için tek `i32` sabitine paketler (`[lo, hi]`).
#[inline(always)]
const fn pair(lo: i16, hi: i16) -> i32 {
    (lo as u16 as i32) | ((hi as u16 as i32) << 16)
}

#[target_feature(enable = "avx2")]
#[inline]
fn yukle_i32x8(c: &[i32; 8]) -> __m256i {
    _mm256_setr_epi32(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7])
}

/// 8 × `i32` dilimini Split-YMM düzeninde (`[c0..c3, 0..0 || c4..c7, 0..0]`) doyurmalı `i16` olarak yükler.
#[target_feature(enable = "avx2")]
#[inline]
fn yukle_i16x8(c: &[i32; 8]) -> __m256i {
    _mm256_packs_epi32(yukle_i32x8(c), _mm256_setzero_si256())
}

/// İki ayrı `8 × i32` sütun dilimini `[c0 || c1]` düzeninde `16 × i16` (`__m256i`) olarak yükler.
#[target_feature(enable = "avx2")]
#[inline]
fn yukle_2col_8(c0: &[i32; 8], c1: &[i32; 8]) -> __m256i {
    _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(yukle_i32x8(c0), yukle_i32x8(c1)))
}

#[target_feature(enable = "avx2")]
#[inline]
fn yukle_i16x16(c: &[i32; 16]) -> __m256i {
    yukle_2col_8(chunk(c, 0), chunk(c, 8))
}

#[target_feature(enable = "avx2")]
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

/// Şerit-paralel 1B büyük DCT çekirdekleri (`x8` = Split-YMM `__m256i`, `x16` = tam `__m256i`).
macro_rules! buyuk_1d_impl {
    (
        $mod_ad:ident,
        $V:ty,
        $HALF_LANE:literal,
        $set1_16:ident,
        $set1_32:ident,
        $zero:ident,
        $unpacklo16:ident,
        $unpackhi16:ident,
        $madd16:ident,
        $add32:ident,
        $srai32:ident,
        $packs32:ident,
        $adds16:ident,
        $subs16:ident,
        $mulhrs16:ident
        $(, $enable_64:ident)?
    ) => {
        mod $mod_ad {
            use super::*;

            /// `(v * c + 2048) >> 12 == pmulhrsw(v, c * 8)` (`|c| <= 4095`).
            #[target_feature(enable = "avx2")]
            #[inline]
            fn mul12(v: $V, c: i16) -> $V {
                $mulhrs16(v, $set1_16(c * 8))
            }

            /// `(((a*c1 - b*c2 + 2048) >> 12), ((a*c2 + b*c1 + 2048) >> 12))`.
            #[target_feature(enable = "avx2")]
            #[inline]
            fn mulsub(a: $V, b: $V, c1: i16, c2: i16) -> ($V, $V) {
                let lo = $unpacklo16(b, a);
                let k1 = $set1_32(pair(c2.wrapping_neg(), c1));
                let k2 = $set1_32(pair(c1, c2));
                let rnd = $set1_32(2048);
                let r1_lo = $srai32::<12>($add32($madd16(lo, k1), rnd));
                let r2_lo = $srai32::<12>($add32($madd16(lo, k2), rnd));
                if $HALF_LANE {
                    ($packs32(r1_lo, r1_lo), $packs32(r2_lo, r2_lo))
                } else {
                    let hi = $unpackhi16(b, a);
                    let r1_hi = $srai32::<12>($add32($madd16(hi, k1), rnd));
                    let r2_hi = $srai32::<12>($add32($madd16(hi, k2), rnd));
                    ($packs32(r1_lo, r1_hi), $packs32(r2_lo, r2_hi))
                }
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn idct4(c: &mut [$V; 4]) {
                let (t1, t0) = mulsub(c[0], c[2], 2896, 2896);
                let (t2, t3) = mulsub(c[1], c[3], 1567, 3784);
                c[0] = $adds16(t0, t3);
                c[1] = $adds16(t1, t2);
                c[2] = $subs16(t1, t2);
                c[3] = $subs16(t0, t3);
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            fn idct4_half(inp: &[$V; 2]) -> [$V; 4] {
                let t0 = $mulhrs16(inp[0], $set1_16(23168));
                let t1 = t0;
                let t2 = mul12(inp[1], 1567);
                let t3 = mul12(inp[1], 3784);
                [
                    $adds16(t0, t3),
                    $adds16(t1, t2),
                    $subs16(t1, t2),
                    $subs16(t0, t3),
                ]
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            fn idct8_finish(even: &[$V; 4], t4a: $V, t5a: $V, t6a: $V, t7a: $V) -> [$V; 8] {
                let [t0, t1, t2, t3] = *even;
                let t4 = $adds16(t4a, t5a);
                let t5a = $subs16(t4a, t5a);
                let t7 = $adds16(t7a, t6a);
                let t6a = $subs16(t7a, t6a);
                let (t5, t6) = mulsub(t6a, t5a, 2896, 2896);
                [
                    $adds16(t0, t7),
                    $adds16(t1, t6),
                    $adds16(t2, t5),
                    $adds16(t3, t4),
                    $subs16(t3, t4),
                    $subs16(t2, t5),
                    $subs16(t1, t6),
                    $subs16(t0, t7),
                ]
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn idct8(c: &mut [$V; 8]) {
                let mut even = [c[0], c[2], c[4], c[6]];
                idct4(&mut even);
                let (t4a, t7a) = mulsub(c[1], c[7], 799, 4017);
                let (t5a, t6a) = mulsub(c[5], c[3], 3406, 2276);
                *c = idct8_finish(&even, t4a, t5a, t6a, t7a);
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            fn idct8_half(inp: &[$V; 4]) -> [$V; 8] {
                let even = idct4_half(&[inp[0], inp[2]]);
                let t4a = mul12(inp[1], 799);
                let t5a = mul12(inp[3], -2276);
                let t6a = mul12(inp[3], 3406);
                let t7a = mul12(inp[1], 4017);
                idct8_finish(&even, t4a, t5a, t6a, t7a)
            }

            #[allow(clippy::too_many_arguments)]
            #[target_feature(enable = "avx2")]
            #[inline]
            fn idct16_finish(
                even: &[$V; 8],
                t8a: $V,
                t9a: $V,
                t10a: $V,
                t11a: $V,
                t12a: $V,
                t13a: $V,
                t14a: $V,
                t15a: $V,
            ) -> [$V; 16] {
                let [t0, t1, t2, t3, t4, t5, t6, t7] = *even;
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

                [
                    $adds16(t0, t15a),
                    $adds16(t1, t14),
                    $adds16(t2, t13a),
                    $adds16(t3, t12),
                    $adds16(t4, t11),
                    $adds16(t5, t10a),
                    $adds16(t6, t9),
                    $adds16(t7, t8a),
                    $subs16(t7, t8a),
                    $subs16(t6, t9),
                    $subs16(t5, t10a),
                    $subs16(t4, t11),
                    $subs16(t3, t12),
                    $subs16(t2, t13a),
                    $subs16(t1, t14),
                    $subs16(t0, t15a),
                ]
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn idct16(c: &mut [$V; 16]) {
                let mut even = [c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]];
                idct8(&mut even);
                let (t8a, t15a) = mulsub(c[1], c[15], 401, 4076);
                let (t9a, t14a) = mulsub(c[9], c[7], 3166, 2598);
                let (t10a, t13a) = mulsub(c[5], c[11], 1931, 3612);
                let (t11a, t12a) = mulsub(c[13], c[3], 3920, 1189);
                *c = idct16_finish(&even, t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a);
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn idct16_half(inp: &[$V; 8]) -> [$V; 16] {
                let even = idct8_half(&[inp[0], inp[2], inp[4], inp[6]]);
                let t8a = mul12(inp[1], 401);
                let t9a = mul12(inp[7], -2598);
                let t10a = mul12(inp[5], 1931);
                let t11a = mul12(inp[3], -1189);
                let t12a = mul12(inp[3], 3920);
                let t13a = mul12(inp[5], 3612);
                let t14a = mul12(inp[7], 3166);
                let t15a = mul12(inp[1], 4076);
                idct16_finish(&even, t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a)
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            fn idct16_qtr(inp: &[$V; 4]) -> [$V; 16] {
                let z = $zero();
                let even = idct8_half(&[inp[0], inp[2], z, z]);
                let t8a = mul12(inp[1], 401);
                let t11a = mul12(inp[3], -1189);
                let t12a = mul12(inp[3], 3920);
                let t15a = mul12(inp[1], 4076);
                idct16_finish(&even, t8a, z, z, t11a, t12a, z, z, t15a)
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            fn idct32_finish(even: &[$V; 16], odd_a: &[$V; 16]) -> [$V; 32] {
                let [
                    t16a,
                    t17a,
                    t18a,
                    t19a,
                    t20a,
                    t21a,
                    t22a,
                    t23a,
                    t24a,
                    t25a,
                    t26a,
                    t27a,
                    t28a,
                    t29a,
                    t30a,
                    t31a,
                ] = *odd_a;

                let t16 = $adds16(t16a, t17a);
                let t17 = $subs16(t16a, t17a);
                let t18 = $subs16(t19a, t18a);
                let t19 = $adds16(t19a, t18a);
                let t20 = $adds16(t20a, t21a);
                let t21 = $subs16(t20a, t21a);
                let t22 = $subs16(t23a, t22a);
                let t23 = $adds16(t23a, t22a);
                let t24 = $adds16(t24a, t25a);
                let t25 = $subs16(t24a, t25a);
                let t26 = $subs16(t27a, t26a);
                let t27 = $adds16(t27a, t26a);
                let t28 = $adds16(t28a, t29a);
                let t29 = $subs16(t28a, t29a);
                let t30 = $subs16(t31a, t30a);
                let t31 = $adds16(t31a, t30a);

                let (t17a, t30a) = mulsub(t30, t17, 799, 4017);
                let (t18a, t29a) = mulsub(t29, t18, -4017, 799);
                let (t21a, t26a) = mulsub(t26, t21, 3406, 2276);
                let (t22a, t25a) = mulsub(t25, t22, -2276, 3406);

                let t16a = $adds16(t16, t19);
                let t17 = $adds16(t17a, t18a);
                let t18 = $subs16(t17a, t18a);
                let t19a = $subs16(t16, t19);
                let t20a = $subs16(t23, t20);
                let t21 = $subs16(t22a, t21a);
                let t22 = $adds16(t22a, t21a);
                let t23a = $adds16(t23, t20);
                let t24a = $adds16(t24, t27);
                let t25 = $adds16(t25a, t26a);
                let t26 = $subs16(t25a, t26a);
                let t27a = $subs16(t24, t27);
                let t28a = $subs16(t31, t28);
                let t29 = $subs16(t30a, t29a);
                let t30 = $adds16(t30a, t29a);
                let t31a = $adds16(t31, t28);

                let (t18a, t29a) = mulsub(t29, t18, 1567, 3784);
                let (t19, t28) = mulsub(t28a, t19a, 1567, 3784);
                let (t20, t27) = mulsub(t27a, t20a, -3784, 1567);
                let (t21a, t26a) = mulsub(t26, t21, -3784, 1567);

                let t16 = $adds16(t16a, t23a);
                let t17a = $adds16(t17, t22);
                let t18 = $adds16(t18a, t21a);
                let t19a = $adds16(t19, t20);
                let t20a = $subs16(t19, t20);
                let t21 = $subs16(t18a, t21a);
                let t22a = $subs16(t17, t22);
                let t23 = $subs16(t16a, t23a);
                let t24 = $subs16(t31a, t24a);
                let t25a = $subs16(t30, t25);
                let t26 = $subs16(t29a, t26a);
                let t27a = $subs16(t28, t27);
                let t28a = $adds16(t28, t27);
                let t29 = $adds16(t29a, t26a);
                let t30a = $adds16(t30, t25);
                let t31 = $adds16(t31a, t24a);

                let (t20, t27) = mulsub(t27a, t20a, 2896, 2896);
                let (t21a, t26a) = mulsub(t26, t21, 2896, 2896);
                let (t22, t25) = mulsub(t25a, t22a, 2896, 2896);
                let (t23a, t24a) = mulsub(t24, t23, 2896, 2896);

                let odd = [
                    t31, t30a, t29, t28a, t27, t26a, t25, t24a, t23a, t22, t21a, t20, t19a, t18,
                    t17a, t16,
                ];
                let mut out = [$zero(); 32];
                for k in 0..16 {
                    out[k] = $adds16(even[k], odd[k]);
                    out[31 - k] = $subs16(even[k], odd[k]);
                }
                out
            }

            /// `inv_dct32_1d_c` (`src/itx_1d.c:247-434`).
            #[target_feature(enable = "avx2")]
            pub(super) fn idct32(c: &mut [$V; 32]) {
                let mut even = [
                    c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14], c[16], c[18], c[20], c[22],
                    c[24], c[26], c[28], c[30],
                ];
                idct16(&mut even);

                let (t16a, t31a) = mulsub(c[1], c[31], 201, 4091);
                let (t17a, t30a) = mulsub(c[17], c[15], 3035, 2751);
                let (t18a, t29a) = mulsub(c[9], c[23], 1751, 3703);
                let (t19a, t28a) = mulsub(c[25], c[7], 3857, 1380);
                let (t20a, t27a) = mulsub(c[5], c[27], 995, 3973);
                let (t21a, t26a) = mulsub(c[21], c[11], 3513, 2106);
                let (t22a, t25a) = mulsub(c[13], c[19], 2440, 3290);
                let (t23a, t24a) = mulsub(c[29], c[3], 4052, 601);

                let odd_a = [
                    t16a, t17a, t18a, t19a, t20a, t21a, t22a, t23a, t24a, t25a, t26a, t27a, t28a,
                    t29a, t30a, t31a,
                ];
                *c = idct32_finish(&even, &odd_a);
            }

            /// Üst yarısı sıfır olan 32 noktalı ters DCT (`c[16..32] == 0` / `tx64 = true`).
            #[target_feature(enable = "avx2")]
            pub(super) fn idct32_half(inp: &[$V; 16]) -> [$V; 32] {
                let even = idct16_half(&[
                    inp[0], inp[2], inp[4], inp[6], inp[8], inp[10], inp[12], inp[14],
                ]);
                let odd_a = [
                    mul12(inp[1], 201),
                    mul12(inp[15], -2751),
                    mul12(inp[9], 1751),
                    mul12(inp[7], -1380),
                    mul12(inp[5], 995),
                    mul12(inp[11], -2106),
                    mul12(inp[13], 2440),
                    mul12(inp[3], -601),
                    mul12(inp[3], 4052),
                    mul12(inp[13], 3290),
                    mul12(inp[11], 3513),
                    mul12(inp[5], 3973),
                    mul12(inp[7], 3857),
                    mul12(inp[9], 3703),
                    mul12(inp[15], 3035),
                    mul12(inp[1], 4091),
                ];
                idct32_finish(&even, &odd_a)
            }

            /// İlk çeyrek (`inp[0..8]`) dışındaki katsayıları sıfır olan 32 noktalı ters DCT.
            #[target_feature(enable = "avx2")]
            pub(super) fn idct32_qtr(inp: &[$V; 8]) -> [$V; 32] {
                let z = $zero();
                let even = idct16_qtr(&[inp[0], inp[2], inp[4], inp[6]]);
                let odd_a = [
                    mul12(inp[1], 201),
                    z,
                    z,
                    mul12(inp[7], -1380),
                    mul12(inp[5], 995),
                    z,
                    z,
                    mul12(inp[3], -601),
                    mul12(inp[3], 4052),
                    z,
                    z,
                    mul12(inp[5], 3973),
                    mul12(inp[7], 3857),
                    z,
                    z,
                    mul12(inp[1], 4091),
                ];
                idct32_finish(&even, &odd_a)
            }

            $(
            const _: () = {
                let _ = stringify!($enable_64);
            };

            #[target_feature(enable = "avx2")]
            fn idct64_finish(even: &[$V; 32], odd_a: &[$V; 32]) -> [$V; 64] {
                let mut t32 = $adds16(odd_a[0], odd_a[1]);
                let mut t33 = $subs16(odd_a[0], odd_a[1]);
                let mut t34 = $subs16(odd_a[3], odd_a[2]);
                let t35 = $adds16(odd_a[3], odd_a[2]);
                let t36 = $adds16(odd_a[4], odd_a[5]);
                let mut t37 = $subs16(odd_a[4], odd_a[5]);
                let mut t38 = $subs16(odd_a[7], odd_a[6]);
                let t39 = $adds16(odd_a[7], odd_a[6]);
                let t40 = $adds16(odd_a[8], odd_a[9]);
                let mut t41 = $subs16(odd_a[8], odd_a[9]);
                let mut t42 = $subs16(odd_a[11], odd_a[10]);
                let t43 = $adds16(odd_a[11], odd_a[10]);
                let t44 = $adds16(odd_a[12], odd_a[13]);
                let mut t45 = $subs16(odd_a[12], odd_a[13]);
                let t46 = $subs16(odd_a[15], odd_a[14]);
                let t47 = $adds16(odd_a[15], odd_a[14]);
                let t48 = $adds16(odd_a[16], odd_a[17]);
                let mut t49 = $subs16(odd_a[16], odd_a[17]);
                let mut t50 = $subs16(odd_a[19], odd_a[18]);
                let t51 = $adds16(odd_a[19], odd_a[18]);
                let t52 = $adds16(odd_a[20], odd_a[21]);
                let mut t53 = $subs16(odd_a[20], odd_a[21]);
                let mut t54 = $subs16(odd_a[23], odd_a[22]);
                let t55 = $adds16(odd_a[23], odd_a[22]);
                let t56 = $adds16(odd_a[24], odd_a[25]);
                let mut t57 = $subs16(odd_a[24], odd_a[25]);
                let mut t58 = $subs16(odd_a[27], odd_a[26]);
                let t59 = $adds16(odd_a[27], odd_a[26]);
                let t60 = $adds16(odd_a[28], odd_a[29]);
                let mut t61 = $subs16(odd_a[28], odd_a[29]);
                let mut t62 = $subs16(odd_a[31], odd_a[30]);
                let t63 = $adds16(odd_a[31], odd_a[30]);

                let (t33a, t62a) = mulsub(t62, t33, 401, 4076);
                let (t34a, t61a) = mulsub(t61, t34, -4076, 401);
                let (t37a, t58a) = mulsub(t58, t37, 3166, 2598);
                let (t38a, t57a) = mulsub(t57, t38, -2598, 3166);
                let (t41a, t54a) = mulsub(t54, t41, 1931, 3612);
                let (t42a, t53a) = mulsub(t53, t42, -3612, 1931);
                let (t45a, t50a) = mulsub(t50, t45, 3920, 1189);
                let (t46a, t49a) = mulsub(t49, t46, -1189, 3920);

                let t32a = $adds16(t32, t35);
                t33 = $adds16(t33a, t34a);
                t34 = $subs16(t33a, t34a);
                let t35a = $subs16(t32, t35);
                let t36a = $subs16(t39, t36);
                t37 = $subs16(t38a, t37a);
                t38 = $adds16(t38a, t37a);
                let t39a = $adds16(t39, t36);
                let t40a = $adds16(t40, t43);
                t41 = $adds16(t41a, t42a);
                t42 = $subs16(t41a, t42a);
                let t43a = $subs16(t40, t43);
                let t44a = $subs16(t47, t44);
                t45 = $subs16(t46a, t45a);
                let t46 = $adds16(t46a, t45a);
                let t47a = $adds16(t47, t44);
                let t48a = $adds16(t48, t51);
                t49 = $adds16(t49a, t50a);
                t50 = $subs16(t49a, t50a);
                let t51a = $subs16(t48, t51);
                let t52a = $subs16(t55, t52);
                t53 = $subs16(t54a, t53a);
                t54 = $adds16(t54a, t53a);
                let t55a = $adds16(t55, t52);
                let t56a = $adds16(t56, t59);
                t57 = $adds16(t57a, t58a);
                t58 = $subs16(t57a, t58a);
                let t59a = $subs16(t56, t59);
                let t60a = $subs16(t63, t60);
                t61 = $subs16(t62a, t61a);
                t62 = $adds16(t62a, t61a);
                let t63a = $adds16(t63, t60);

                let (t34a, t61a) = mulsub(t61, t34, 799, 4017);
                let (t35, t60) = mulsub(t60a, t35a, 799, 4017);
                let (t36, t59) = mulsub(t59a, t36a, -4017, 799);
                let (t37a, t58a) = mulsub(t58, t37, -4017, 799);
                let (t42a, t53a) = mulsub(t53, t42, 3406, 2276);
                let (t43, t52) = mulsub(t52a, t43a, 3406, 2276);
                let (t44, t51) = mulsub(t51a, t44a, -2276, 3406);
                let (t45a, t50a) = mulsub(t50, t45, -2276, 3406);

                t32 = $adds16(t32a, t39a);
                let t33a = $adds16(t33, t38);
                t34 = $adds16(t34a, t37a);
                let t35a = $adds16(t35, t36);
                let t36a = $subs16(t35, t36);
                t37 = $subs16(t34a, t37a);
                let t38a = $subs16(t33, t38);
                let t39 = $subs16(t32a, t39a);
                let t40 = $subs16(t47a, t40a);
                let t41a = $subs16(t46, t41);
                t42 = $subs16(t45a, t42a);
                let t43a = $subs16(t44, t43);
                let t44a = $adds16(t44, t43);
                t45 = $adds16(t45a, t42a);
                let t46a = $adds16(t46, t41);
                let t47 = $adds16(t47a, t40a);
                let t48 = $adds16(t48a, t55a);
                let t49a = $adds16(t49, t54);
                t50 = $adds16(t50a, t53a);
                let t51a = $adds16(t51, t52);
                let t52a = $subs16(t51, t52);
                t53 = $subs16(t50a, t53a);
                let t54a = $subs16(t49, t54);
                let t55 = $subs16(t48a, t55a);
                let t56 = $subs16(t63a, t56a);
                let t57a = $subs16(t62, t57);
                t58 = $subs16(t61a, t58a);
                let t59a = $subs16(t60, t59);
                let t60a = $adds16(t60, t59);
                t61 = $adds16(t61a, t58a);
                let t62a = $adds16(t62, t57);
                let t63 = $adds16(t63a, t56a);

                let (t36, t59) = mulsub(t59a, t36a, 1567, 3784);
                let (t37a, t58a) = mulsub(t58, t37, 1567, 3784);
                let (t38, t57) = mulsub(t57a, t38a, 1567, 3784);
                let (t39a, t56a) = mulsub(t56, t39, 1567, 3784);
                let (t40a, t55a) = mulsub(t55, t40, -3784, 1567);
                let (t41, t54) = mulsub(t54a, t41a, -3784, 1567);
                let (t42a, t53a) = mulsub(t53, t42, -3784, 1567);
                let (t43, t52) = mulsub(t52a, t43a, -3784, 1567);

                let t32a = $adds16(t32, t47);
                t33 = $adds16(t33a, t46a);
                let t34a = $adds16(t34, t45);
                let t35 = $adds16(t35a, t44a);
                let t36a = $adds16(t36, t43);
                t37 = $adds16(t37a, t42a);
                let t38a = $adds16(t38, t41);
                let t39 = $adds16(t39a, t40a);
                let t40 = $subs16(t39a, t40a);
                let t41a = $subs16(t38, t41);
                t42 = $subs16(t37a, t42a);
                let t43a = $subs16(t36, t43);
                let t44 = $subs16(t35a, t44a);
                let t45a = $subs16(t34, t45);
                let t46 = $subs16(t33a, t46a);
                let t47a = $subs16(t32, t47);
                let t48a = $subs16(t63, t48);
                t49 = $subs16(t62a, t49a);
                let t50a = $subs16(t61, t50);
                let t51 = $subs16(t60a, t51a);
                let t52a = $subs16(t59, t52);
                t53 = $subs16(t58a, t53a);
                let t54a = $subs16(t57, t54);
                let t55 = $subs16(t56a, t55a);
                let t56 = $adds16(t56a, t55a);
                let t57a = $adds16(t57, t54);
                t58 = $adds16(t58a, t53a);
                let t59a = $adds16(t59, t52);
                let t60 = $adds16(t60a, t51a);
                let t61a = $adds16(t61, t50);
                t62 = $adds16(t62a, t49a);
                let t63a = $adds16(t63, t48);

                let (t40a, t55a) = mulsub(t55, t40, 2896, 2896);
                let (t41, t54) = mulsub(t54a, t41a, 2896, 2896);
                let (t42a, t53a) = mulsub(t53, t42, 2896, 2896);
                let (t43, t52) = mulsub(t52a, t43a, 2896, 2896);
                let (t44a, t51a) = mulsub(t51, t44, 2896, 2896);
                let (t45, t50) = mulsub(t50a, t45a, 2896, 2896);
                let (t46a, t49a) = mulsub(t49, t46, 2896, 2896);
                let (t47, t48) = mulsub(t48a, t47a, 2896, 2896);

                let odd = [
                    t63a, t62, t61a, t60, t59a, t58, t57a, t56, t55a, t54, t53a, t52, t51a, t50,
                    t49a, t48, t47, t46a, t45, t44a, t43, t42a, t41, t40a, t39, t38a, t37, t36a,
                    t35, t34a, t33, t32a,
                ];
                let mut out = [$zero(); 64];
                for k in 0..32 {
                    out[k] = $adds16(even[k], odd[k]);
                    out[63 - k] = $subs16(even[k], odd[k]);
                }
                out
            }

            /// `inv_dct64_1d_c` (`src/itx_1d.c:436-781`, yalnız ilk 32 giriş okunur).
            #[target_feature(enable = "avx2")]
            pub(super) fn idct64(inp: &[$V; 32]) -> [$V; 64] {
                let even = idct32_half(&[
                    inp[0], inp[2], inp[4], inp[6], inp[8], inp[10], inp[12], inp[14], inp[16],
                    inp[18], inp[20], inp[22], inp[24], inp[26], inp[28], inp[30],
                ]);
                let odd_a = [
                    mul12(inp[1], 101),
                    mul12(inp[31], -2824),
                    mul12(inp[17], 1660),
                    mul12(inp[15], -1474),
                    mul12(inp[9], 897),
                    mul12(inp[23], -2191),
                    mul12(inp[25], 2359),
                    mul12(inp[7], -700),
                    mul12(inp[5], 501),
                    mul12(inp[27], -2520),
                    mul12(inp[21], 2019),
                    mul12(inp[11], -1092),
                    mul12(inp[13], 1285),
                    mul12(inp[19], -1842),
                    mul12(inp[29], 2675),
                    mul12(inp[3], -301),
                    mul12(inp[3], 4085),
                    mul12(inp[29], 3102),
                    mul12(inp[19], 3659),
                    mul12(inp[13], 3889),
                    mul12(inp[11], 3948),
                    mul12(inp[21], 3564),
                    mul12(inp[27], 3229),
                    mul12(inp[5], 4065),
                    mul12(inp[7], 4036),
                    mul12(inp[25], 3349),
                    mul12(inp[23], 3461),
                    mul12(inp[9], 3996),
                    mul12(inp[15], 3822),
                    mul12(inp[17], 3745),
                    mul12(inp[31], 2967),
                    mul12(inp[1], 4095),
                ];
                idct64_finish(&even, &odd_a)
            }

            /// İlk çeyrek (`inp[0..16]`) dışındaki katsayıları sıfır olan 64 noktalı ters DCT.
            #[target_feature(enable = "avx2")]
            pub(super) fn idct64_qtr(inp: &[$V; 16]) -> [$V; 64] {
                let z = $zero();
                let even = idct32_half(&[
                    inp[0], inp[2], inp[4], inp[6], inp[8], inp[10], inp[12], inp[14], z, z, z, z,
                    z, z, z, z,
                ]);
                let odd_a = [
                    mul12(inp[1], 101),
                    z,
                    z,
                    mul12(inp[15], -1474),
                    mul12(inp[9], 897),
                    z,
                    z,
                    mul12(inp[7], -700),
                    mul12(inp[5], 501),
                    z,
                    z,
                    mul12(inp[11], -1092),
                    mul12(inp[13], 1285),
                    z,
                    z,
                    mul12(inp[3], -301),
                    mul12(inp[3], 4085),
                    z,
                    z,
                    mul12(inp[13], 3889),
                    mul12(inp[11], 3948),
                    z,
                    z,
                    mul12(inp[5], 4065),
                    mul12(inp[7], 4036),
                    z,
                    z,
                    mul12(inp[9], 3996),
                    mul12(inp[15], 3822),
                    z,
                    z,
                    mul12(inp[1], 4095),
                ];
                idct64_finish(&even, &odd_a)
            }

            /// İlk sekizde birlik (`inp[0..8]`) dışındaki katsayıları sıfır olan 64 noktalı ters DCT.
            #[target_feature(enable = "avx2")]
            pub(super) fn idct64_eighth(inp: &[$V; 8]) -> [$V; 64] {
                let z = $zero();
                let even = idct32_qtr(&[inp[0], inp[2], inp[4], inp[6], z, z, z, z]);
                let odd_a = [
                    mul12(inp[1], 101),
                    z,
                    z,
                    z,
                    z,
                    z,
                    z,
                    mul12(inp[7], -700),
                    mul12(inp[5], 501),
                    z,
                    z,
                    z,
                    z,
                    z,
                    z,
                    mul12(inp[3], -301),
                    mul12(inp[3], 4085),
                    z,
                    z,
                    z,
                    z,
                    z,
                    z,
                    mul12(inp[5], 4065),
                    mul12(inp[7], 4036),
                    z,
                    z,
                    z,
                    z,
                    z,
                    z,
                    mul12(inp[1], 4095),
                ];
                idct64_finish(&even, &odd_a)
            }
            )?

            #[target_feature(enable = "avx2")]
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

            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn round_pass2<const N: usize>(c: &mut [$V; N]) {
                let r4 = $set1_16(2048);
                for v in c.iter_mut() {
                    *v = $mulhrs16(*v, r4);
                }
            }
        }
    };
}

buyuk_1d_impl!(
    x8,
    __m256i,
    true,
    _mm256_set1_epi16,
    _mm256_set1_epi32,
    _mm256_setzero_si256,
    _mm256_unpacklo_epi16,
    _mm256_unpackhi_epi16,
    _mm256_madd_epi16,
    _mm256_add_epi32,
    _mm256_srai_epi32,
    _mm256_packs_epi32,
    _mm256_adds_epi16,
    _mm256_subs_epi16,
    _mm256_mulhrs_epi16
);

buyuk_1d_impl!(
    x16,
    __m256i,
    false,
    _mm256_set1_epi16,
    _mm256_set1_epi32,
    _mm256_setzero_si256,
    _mm256_unpacklo_epi16,
    _mm256_unpackhi_epi16,
    _mm256_madd_epi16,
    _mm256_add_epi32,
    _mm256_srai_epi32,
    _mm256_packs_epi32,
    _mm256_adds_epi16,
    _mm256_subs_epi16,
    _mm256_mulhrs_epi16,
    with_idct64
);

// ---------------------------------------------------------------------------
// Matris transpoz yardımcıları (`i16`, P34 modül içi kopyalar)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
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

#[target_feature(enable = "avx2")]
#[inline]
fn transpose_8x8_split_ymm(r: &mut [__m256i; 8]) {
    let t = transpose_4x4_in_ymm(r[0], r[1], r[2], r[3]);
    let b = transpose_4x4_in_ymm(r[4], r[5], r[6], r[7]);
    for k in 0..4 {
        r[k] = _mm256_permute2x128_si256::<0x20>(t[k], b[k]);
        r[k + 4] = _mm256_permute2x128_si256::<0x31>(t[k], b[k]);
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn transpose_8x8_in_ymm(r: &mut [__m256i; 8]) {
    let t0 = _mm256_unpacklo_epi16(r[0], r[1]);
    let t1 = _mm256_unpackhi_epi16(r[0], r[1]);
    let t2 = _mm256_unpacklo_epi16(r[2], r[3]);
    let t3 = _mm256_unpackhi_epi16(r[2], r[3]);
    let t4 = _mm256_unpacklo_epi16(r[4], r[5]);
    let t5 = _mm256_unpackhi_epi16(r[4], r[5]);
    let t6 = _mm256_unpacklo_epi16(r[6], r[7]);
    let t7 = _mm256_unpackhi_epi16(r[6], r[7]);

    let s0 = _mm256_unpacklo_epi32(t0, t2);
    let s1 = _mm256_unpackhi_epi32(t0, t2);
    let s2 = _mm256_unpacklo_epi32(t1, t3);
    let s3 = _mm256_unpackhi_epi32(t1, t3);
    let s4 = _mm256_unpacklo_epi32(t4, t6);
    let s5 = _mm256_unpackhi_epi32(t4, t6);
    let s6 = _mm256_unpacklo_epi32(t5, t7);
    let s7 = _mm256_unpackhi_epi32(t5, t7);

    r[0] = _mm256_unpacklo_epi64(s0, s4);
    r[1] = _mm256_unpackhi_epi64(s0, s4);
    r[2] = _mm256_unpacklo_epi64(s1, s5);
    r[3] = _mm256_unpackhi_epi64(s1, s5);
    r[4] = _mm256_unpacklo_epi64(s2, s6);
    r[5] = _mm256_unpackhi_epi64(s2, s6);
    r[6] = _mm256_unpacklo_epi64(s3, s7);
    r[7] = _mm256_unpackhi_epi64(s3, s7);
}

#[target_feature(enable = "avx2")]
#[inline]
fn transpose_16x16_ymm(r: &mut [__m256i; 16]) {
    let mut lo = [r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]];
    let mut hi = [r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15]];
    transpose_8x8_in_ymm(&mut lo);
    transpose_8x8_in_ymm(&mut hi);
    for k in 0..8 {
        r[k] = _mm256_permute2x128_si256::<0x20>(lo[k], hi[k]);
        r[k + 8] = _mm256_permute2x128_si256::<0x31>(lo[k], hi[k]);
    }
}

/// `8` sütun × `16` satırlık `YMM` dizisini `16` satır × `8` sütunluk Split-YMM dizisine çevirir.
#[target_feature(enable = "avx2")]
#[inline]
fn transpose_8x16_to_split_ymm(mut cols: [__m256i; 8]) -> [__m256i; 16] {
    transpose_8x8_in_ymm(&mut cols);
    let mut rows = [_mm256_setzero_si256(); 16];
    for k in 0..8 {
        rows[k] = _mm256_permute4x64_epi64::<0x50>(cols[k]);
        rows[k + 8] = _mm256_permute4x64_epi64::<0xFA>(cols[k]);
    }
    rows
}

/// `16` sütun × `8` satırlık Split-YMM dizisini `8` satır × `16` sütunluk `YMM` dizisine çevirir.
#[target_feature(enable = "avx2")]
#[inline]
fn transpose_16x8_split_to_ymm(cols: &[__m256i; 16]) -> [__m256i; 8] {
    let mut rows = [_mm256_setzero_si256(); 8];
    for (k, rk) in rows.iter_mut().enumerate() {
        let lo = _mm256_permute4x64_epi64::<0xD8>(cols[k]);
        let hi = _mm256_permute4x64_epi64::<0xD8>(cols[k + 8]);
        *rk = _mm256_permute2x128_si256::<0x20>(lo, hi);
    }
    transpose_8x8_in_ymm(&mut rows);
    rows
}

// ---------------------------------------------------------------------------
// Sınır-güvenli piksel ekleme ve saklama yardımcıları (`w ∈ {8, 16, 32, 64}`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
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

#[target_feature(enable = "avx2")]
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

#[target_feature(enable = "avx2")]
#[inline]
fn ekle_yaz_w16_off<const H: usize>(
    dst: &mut [u8],
    stride: usize,
    y_off: usize,
    x_off: usize,
    rows: &[__m256i; H],
) {
    const { assert!(H.is_multiple_of(2)) };
    let mut y = 0;
    while y < H {
        let r0 = (y_off + y) * stride + x_off;
        let r1 = (y_off + y + 1) * stride + x_off;
        let px0 = _mm256_cvtepu8_epi16(load_u8x16(chunk(dst, r0)));
        let px1 = _mm256_cvtepu8_epi16(load_u8x16(chunk(dst, r1)));
        let s0 = _mm256_add_epi16(px0, rows[y]);
        let s1 = _mm256_add_epi16(px1, rows[y + 1]);
        let p = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(s0, s1));
        store_u8x16(chunk_mut(dst, r0), _mm256_castsi256_si128(p));
        store_u8x16(chunk_mut(dst, r1), _mm256_extracti128_si256::<1>(p));
        y += 2;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn ekle_yaz_w32<const H: usize>(
    dst: &mut [u8],
    stride: usize,
    lo: &[__m256i; H],
    hi: &[__m256i; H],
) {
    for y in 0..H {
        let off = y * stride;
        let px = load_u8x32(chunk(dst, off));
        let px_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(px));
        let px_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(px));
        let s0 = _mm256_add_epi16(px_lo, lo[y]);
        let s1 = _mm256_add_epi16(px_hi, hi[y]);
        let p = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(s0, s1));
        store_u8x32(chunk_mut(dst, off), p);
    }
}

#[target_feature(enable = "avx2")]
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
    let add8 = dc.clamp(0, 255) as u8 as i8;
    let sub8 = (-dc).clamp(0, 255) as u8 as i8;

    if W == 8 {
        let va = _mm_set1_epi8(add8);
        let vs = _mm_set1_epi8(sub8);
        let mut y = 0;
        while y < H {
            let d0 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, y * stride)));
            let d1 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, (y + 1) * stride)));
            let p = _mm_subs_epu8(_mm_adds_epu8(_mm_unpacklo_epi64(d0, d1), va), vs);
            store_u8x8(chunk_mut(dst, y * stride), p);
            store_u8x8(chunk_mut(dst, (y + 1) * stride), _mm_srli_si128::<8>(p));
            y += 2;
        }
    } else if W == 16 {
        let va = _mm_set1_epi8(add8);
        let vs = _mm_set1_epi8(sub8);
        for y in 0..H {
            let off = y * stride;
            let d = load_u8x16(chunk(dst, off));
            store_u8x16(chunk_mut(dst, off), _mm_subs_epu8(_mm_adds_epu8(d, va), vs));
        }
    } else if W == 32 {
        let va = _mm256_set1_epi8(add8);
        let vs = _mm256_set1_epi8(sub8);
        for y in 0..H {
            let off = y * stride;
            let d = load_u8x32(chunk(dst, off));
            store_u8x32(
                chunk_mut(dst, off),
                _mm256_subs_epu8(_mm256_adds_epu8(d, va), vs),
            );
        }
    } else {
        let va = _mm256_set1_epi8(add8);
        let vs = _mm256_set1_epi8(sub8);
        for y in 0..H {
            let off = y * stride;
            let d0 = load_u8x32(chunk(dst, off));
            let d1 = load_u8x32(chunk(dst, off + 32));
            store_u8x32(
                chunk_mut(dst, off),
                _mm256_subs_epu8(_mm256_adds_epu8(d0, va), vs),
            );
            store_u8x32(
                chunk_mut(dst, off + 32),
                _mm256_subs_epu8(_mm256_adds_epu8(d1, va), vs),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Boyut × tür bazında büyük 2B AVX2 çekirdek gövdeleri (`15` kayıt)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
fn dct_dct_8x32(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    if eob <= 42 {
        let mut cols8 = [_mm256_setzero_si256(); 8];
        for (x, col) in cols8.iter_mut().enumerate() {
            *col = yukle_i16x8(chunk(coeff, x * 32));
        }
        sifirla_katsayi::<256>(coeff);
        x8::idct8(&mut cols8);
        x8::round_shift::<8, 2>(&mut cols8);
        transpose_8x8_split_ymm(&mut cols8);
        let mut out = x8::idct32_qtr(&cols8);
        x8::round_pass2(&mut out);
        ekle_yaz_w8_split(dst, stride, &out);
        return;
    }

    let mut top_cols = [_mm256_setzero_si256(); 8];
    for (x, col) in top_cols.iter_mut().enumerate() {
        *col = yukle_i16x16(chunk(coeff, x * 32));
    }
    if eob <= 106 {
        sifirla_katsayi::<256>(coeff);
        x16::idct8(&mut top_cols);
        x16::round_shift::<8, 2>(&mut top_cols);
        let rows16 = transpose_8x16_to_split_ymm(top_cols);
        let mut out = x8::idct32_half(&rows16);
        x8::round_pass2(&mut out);
        ekle_yaz_w8_split(dst, stride, &out);
    } else {
        let mut bot_cols = [_mm256_setzero_si256(); 8];
        for (x, col) in bot_cols.iter_mut().enumerate() {
            *col = yukle_i16x16(chunk(coeff, x * 32 + 16));
        }
        sifirla_katsayi::<256>(coeff);
        x16::idct8(&mut top_cols);
        x16::round_shift::<8, 2>(&mut top_cols);
        x16::idct8(&mut bot_cols);
        x16::round_shift::<8, 2>(&mut bot_cols);
        let top16 = transpose_8x16_to_split_ymm(top_cols);
        let bot16 = transpose_8x16_to_split_ymm(bot_cols);
        let mut rows = [_mm256_setzero_si256(); 32];
        rows[..16].copy_from_slice(&top16);
        rows[16..].copy_from_slice(&bot16);
        x8::idct32(&mut rows);
        x8::round_pass2(&mut rows);
        ekle_yaz_w8_split(dst, stride, &rows);
    }
}

#[target_feature(enable = "avx2")]
fn idtx_8x32(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    let r1 = _mm256_set1_epi16(16384);
    let r2 = _mm256_set1_epi16(8192);
    let mut top = [_mm256_setzero_si256(); 8];
    for (x, tx) in top.iter_mut().enumerate() {
        let v = yukle_i16x16(chunk(coeff, x * 32));
        *tx = _mm256_mulhrs_epi16(_mm256_mulhrs_epi16(v, r1), r2);
    }
    if eob <= 106 {
        sifirla_katsayi::<256>(coeff);
        transpose_8x8_in_ymm(&mut top);
        let mut rows = [_mm_setzero_si128(); 16];
        for k in 0..8 {
            rows[k] = _mm256_castsi256_si128(top[k]);
            rows[k + 8] = _mm256_extracti128_si256::<1>(top[k]);
        }
        ekle_yaz_w8(dst, stride, &rows);
    } else {
        let mut bot = [_mm256_setzero_si256(); 8];
        for (x, bx) in bot.iter_mut().enumerate() {
            let v = yukle_i16x16(chunk(coeff, x * 32 + 16));
            *bx = _mm256_mulhrs_epi16(_mm256_mulhrs_epi16(v, r1), r2);
        }
        sifirla_katsayi::<256>(coeff);
        transpose_8x8_in_ymm(&mut top);
        let mut rows_top = [_mm_setzero_si128(); 16];
        for k in 0..8 {
            rows_top[k] = _mm256_castsi256_si128(top[k]);
            rows_top[k + 8] = _mm256_extracti128_si256::<1>(top[k]);
        }
        ekle_yaz_w8(dst, stride, &rows_top);

        transpose_8x8_in_ymm(&mut bot);
        let mut rows_bot = [_mm_setzero_si128(); 16];
        for k in 0..8 {
            rows_bot[k] = _mm256_castsi256_si128(bot[k]);
            rows_bot[k + 8] = _mm256_extracti128_si256::<1>(bot[k]);
        }
        ekle_yaz_w8(&mut dst[16 * stride..], stride, &rows_bot);
    }
}

#[target_feature(enable = "avx2")]
fn dct_dct_32x8(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    let mut cols = if eob <= 42 {
        let mut qtr = [_mm256_setzero_si256(); 8];
        for (x, qx) in qtr.iter_mut().enumerate() {
            *qx = yukle_i16x8(chunk(coeff, x * 8));
        }
        sifirla_katsayi::<256>(coeff);
        x8::idct32_qtr(&qtr)
    } else if eob <= 106 {
        let mut left = [_mm256_setzero_si256(); 16];
        for (x, lx) in left.iter_mut().enumerate() {
            *lx = yukle_i16x8(chunk(coeff, x * 8));
        }
        sifirla_katsayi::<256>(coeff);
        x8::idct32_half(&left)
    } else {
        let mut all = [_mm256_setzero_si256(); 32];
        for (x, ax) in all.iter_mut().enumerate() {
            *ax = yukle_i16x8(chunk(coeff, x * 8));
        }
        sifirla_katsayi::<256>(coeff);
        x8::idct32(&mut all);
        all
    };
    x8::round_shift::<32, 2>(&mut cols);
    let mut lo = transpose_16x8_split_to_ymm(chunk(&cols, 0));
    let mut hi = transpose_16x8_split_to_ymm(chunk(&cols, 16));
    x16::idct8(&mut lo);
    x16::idct8(&mut hi);
    x16::round_pass2(&mut lo);
    x16::round_pass2(&mut hi);
    ekle_yaz_w32(dst, stride, &lo, &hi);
}

#[target_feature(enable = "avx2")]
fn idtx_32x8(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    let r3 = _mm256_set1_epi16(4096); // `(2*v + 8) >> 4 == (v + 4) >> 3`
    let mut lo = [_mm256_setzero_si256(); 8];
    for (k, lk) in lo.iter_mut().enumerate() {
        let v = yukle_2col_8(chunk(coeff, k * 8), chunk(coeff, (k + 8) * 8));
        *lk = _mm256_mulhrs_epi16(v, r3);
    }
    transpose_8x8_in_ymm(&mut lo);
    if eob <= 106 {
        sifirla_katsayi::<256>(coeff);
        ekle_yaz_w16_off(dst, stride, 0, 0, &lo);
    } else {
        let mut hi = [_mm256_setzero_si256(); 8];
        for (k, hk) in hi.iter_mut().enumerate() {
            let v = yukle_2col_8(chunk(coeff, (k + 16) * 8), chunk(coeff, (k + 24) * 8));
            *hk = _mm256_mulhrs_epi16(v, r3);
        }
        sifirla_katsayi::<256>(coeff);
        transpose_8x8_in_ymm(&mut hi);
        ekle_yaz_w32(dst, stride, &lo, &hi);
    }
}

#[target_feature(enable = "avx2")]
fn dct_dct_16x32(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    let r2 = _mm256_set1_epi16(23168);
    let mut top = [_mm256_setzero_si256(); 16];
    for (x, tx) in top.iter_mut().enumerate() {
        *tx = _mm256_mulhrs_epi16(yukle_i16x16(chunk(coeff, x * 32)), r2);
    }
    if eob <= 150 {
        sifirla_katsayi::<512>(coeff);
        x16::idct16(&mut top);
        x16::round_shift::<16, 1>(&mut top);
        transpose_16x16_ymm(&mut top);
        let mut out = x16::idct32_half(&top);
        x16::round_pass2(&mut out);
        ekle_yaz_w16_off(dst, stride, 0, 0, &out);
    } else {
        let mut rows = [_mm256_setzero_si256(); 32];
        rows[..16].copy_from_slice(&top);
        for (x, bx) in rows[16..].iter_mut().enumerate() {
            *bx = _mm256_mulhrs_epi16(yukle_i16x16(chunk(coeff, x * 32 + 16)), r2);
        }
        sifirla_katsayi::<512>(coeff);
        x16::idct16(chunk_mut(&mut rows, 0));
        x16::round_shift::<16, 1>(chunk_mut(&mut rows, 0));
        transpose_16x16_ymm(chunk_mut(&mut rows, 0));

        x16::idct16(chunk_mut(&mut rows, 16));
        x16::round_shift::<16, 1>(chunk_mut(&mut rows, 16));
        transpose_16x16_ymm(chunk_mut(&mut rows, 16));

        x16::idct32(&mut rows);
        x16::round_pass2(&mut rows);
        ekle_yaz_w16_off(dst, stride, 0, 0, &rows);
    }
}

#[target_feature(enable = "avx2")]
fn idtx_16x32(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    let strips = if eob <= 43 {
        1
    } else if eob <= 150 {
        2
    } else if eob <= 278 {
        3
    } else {
        4
    };
    let r_rect = _mm256_set1_epi16(23168);
    let k16 = _mm256_set1_epi16(27152);
    let r1 = _mm256_set1_epi16(16384);
    let r2 = _mm256_set1_epi16(8192);

    for s in 0..strips {
        let y_off = s * 8;
        let mut m = [_mm256_setzero_si256(); 8];
        for (k, mk) in m.iter_mut().enumerate() {
            let v0 = _mm256_mulhrs_epi16(
                yukle_2col_8(
                    chunk(coeff, k * 32 + y_off),
                    chunk(coeff, (k + 8) * 32 + y_off),
                ),
                r_rect,
            );
            let t = _mm256_mulhrs_epi16(v0, k16);
            let v1 = _mm256_adds_epi16(v0, _mm256_mulhrs_epi16(t, r1));
            *mk = _mm256_mulhrs_epi16(v1, r2);
        }
        transpose_8x8_in_ymm(&mut m);
        ekle_yaz_w16_off(dst, stride, y_off, 0, &m);
    }
    sifirla_katsayi::<512>(coeff);
}

#[target_feature(enable = "avx2")]
fn dct_dct_32x16(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    let r2 = _mm256_set1_epi16(23168);
    let mut cols = if eob <= 150 {
        let mut left = [_mm256_setzero_si256(); 16];
        for (x, lx) in left.iter_mut().enumerate() {
            *lx = _mm256_mulhrs_epi16(yukle_i16x16(chunk(coeff, x * 16)), r2);
        }
        sifirla_katsayi::<512>(coeff);
        x16::idct32_half(&left)
    } else {
        let mut all = [_mm256_setzero_si256(); 32];
        for (x, ax) in all.iter_mut().enumerate() {
            *ax = _mm256_mulhrs_epi16(yukle_i16x16(chunk(coeff, x * 16)), r2);
        }
        sifirla_katsayi::<512>(coeff);
        x16::idct32(&mut all);
        all
    };
    x16::round_shift::<32, 1>(&mut cols);
    transpose_16x16_ymm(chunk_mut(&mut cols, 0));
    transpose_16x16_ymm(chunk_mut(&mut cols, 16));
    x16::idct16(chunk_mut(&mut cols, 0));
    x16::idct16(chunk_mut(&mut cols, 16));
    x16::round_pass2(&mut cols);
    ekle_yaz_w32::<16>(dst, stride, chunk(&cols, 0), chunk(&cols, 16));
}

#[target_feature(enable = "avx2")]
fn idtx_32x16(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    let r_rect = _mm256_set1_epi16(23168);
    let k16 = _mm256_set1_epi16(27152);
    let r4 = _mm256_set1_epi16(2048);

    let mut tile16x8 = |x_base: usize, y_off: usize| {
        let mut m = [_mm256_setzero_si256(); 8];
        for (k, mk) in m.iter_mut().enumerate() {
            let v0 = _mm256_mulhrs_epi16(
                yukle_2col_8(
                    chunk(coeff, (x_base + k) * 16 + y_off),
                    chunk(coeff, (x_base + k + 8) * 16 + y_off),
                ),
                r_rect,
            );
            let v1 = _mm256_adds_epi16(v0, v0);
            let t = _mm256_mulhrs_epi16(v1, k16);
            let s = _mm256_adds_epi16(_mm256_adds_epi16(v1, v1), t);
            *mk = _mm256_mulhrs_epi16(s, r4);
        }
        transpose_8x8_in_ymm(&mut m);
        ekle_yaz_w16_off(dst, stride, y_off, x_base, &m);
    };

    if eob <= 35 {
        tile16x8(0, 0);
    } else if eob <= 150 {
        tile16x8(0, 0);
        tile16x8(0, 8);
    } else {
        for &(y_off, x_base) in &[(0usize, 0usize), (0, 16), (8, 0), (8, 16)] {
            tile16x8(x_base, y_off);
        }
    }
    sifirla_katsayi::<512>(coeff);
}

#[target_feature(enable = "avx2")]
fn dct_dct_32x32(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    if eob < 136 {
        let mut tl = [_mm256_setzero_si256(); 16];
        for (x, tlx) in tl.iter_mut().enumerate() {
            *tlx = yukle_i16x16(chunk(coeff, x * 32));
        }
        sifirla_katsayi::<1024>(coeff);
        let mut p1 = x16::idct32_half(&tl);
        x16::round_shift::<32, 2>(&mut p1);
        transpose_16x16_ymm(chunk_mut(&mut p1, 0));
        transpose_16x16_ymm(chunk_mut(&mut p1, 16));

        let mut lo_out = x16::idct32_half(chunk(&p1, 0));
        let mut hi_out = x16::idct32_half(chunk(&p1, 16));
        x16::round_pass2(&mut lo_out);
        x16::round_pass2(&mut hi_out);
        ekle_yaz_w32(dst, stride, &lo_out, &hi_out);
    } else {
        let mut top = [_mm256_setzero_si256(); 32];
        let mut bot = [_mm256_setzero_si256(); 32];
        for (x, (tx, bx)) in top.iter_mut().zip(bot.iter_mut()).enumerate() {
            *tx = yukle_i16x16(chunk(coeff, x * 32));
            *bx = yukle_i16x16(chunk(coeff, x * 32 + 16));
        }
        sifirla_katsayi::<1024>(coeff);
        x16::idct32(&mut top);
        x16::round_shift::<32, 2>(&mut top);
        x16::idct32(&mut bot);
        x16::round_shift::<32, 2>(&mut bot);

        transpose_16x16_ymm(chunk_mut(&mut top, 0));
        transpose_16x16_ymm(chunk_mut(&mut top, 16));
        transpose_16x16_ymm(chunk_mut(&mut bot, 0));
        transpose_16x16_ymm(chunk_mut(&mut bot, 16));
        top[16..32].swap_with_slice(&mut bot[0..16]);

        x16::idct32(&mut top);
        x16::idct32(&mut bot);
        x16::round_pass2(&mut top);
        x16::round_pass2(&mut bot);
        ekle_yaz_w32(dst, stride, &top, &bot);
    }
}

#[target_feature(enable = "avx2")]
fn idtx_32x32(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    let r2 = _mm256_set1_epi16(8192);
    let mut tile16x8 = |x_base: usize, y_off: usize| {
        let mut m = [_mm256_setzero_si256(); 8];
        for (k, mk) in m.iter_mut().enumerate() {
            let v = yukle_2col_8(
                chunk(coeff, (x_base + k) * 32 + y_off),
                chunk(coeff, (x_base + k + 8) * 32 + y_off),
            );
            *mk = _mm256_mulhrs_epi16(v, r2);
        }
        transpose_8x8_in_ymm(&mut m);
        ekle_yaz_w16_off(dst, stride, y_off, x_base, &m);
    };

    if eob < 136 {
        tile16x8(0, 0);
        tile16x8(0, 8);
    } else {
        for &(y_off, x_base) in &[
            (0usize, 0usize),
            (8, 0),
            (0, 16),
            (8, 16),
            (16, 0),
            (24, 0),
            (16, 16),
            (24, 16),
        ] {
            tile16x8(x_base, y_off);
        }
    }
    sifirla_katsayi::<1024>(coeff);
}

#[target_feature(enable = "avx2")]
fn dct_dct_16x64(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    let mut top = [_mm256_setzero_si256(); 16];
    for (x, tx) in top.iter_mut().enumerate() {
        *tx = yukle_i16x16(chunk(coeff, x * 32));
    }
    if eob <= 150 {
        sifirla_katsayi::<512>(coeff);
        x16::idct16(&mut top);
        x16::round_shift::<16, 2>(&mut top);
        transpose_16x16_ymm(&mut top);
        let mut out = x16::idct64_qtr(&top);
        x16::round_pass2(&mut out);
        ekle_yaz_w16_off(dst, stride, 0, 0, &out);
    } else {
        let mut rows = [_mm256_setzero_si256(); 32];
        rows[..16].copy_from_slice(&top);
        for (x, bx) in rows[16..].iter_mut().enumerate() {
            *bx = yukle_i16x16(chunk(coeff, x * 32 + 16));
        }
        sifirla_katsayi::<512>(coeff);
        x16::idct16(chunk_mut(&mut rows, 0));
        x16::round_shift::<16, 2>(chunk_mut(&mut rows, 0));
        transpose_16x16_ymm(chunk_mut(&mut rows, 0));

        x16::idct16(chunk_mut(&mut rows, 16));
        x16::round_shift::<16, 2>(chunk_mut(&mut rows, 16));
        transpose_16x16_ymm(chunk_mut(&mut rows, 16));

        let mut out = x16::idct64(&rows);
        x16::round_pass2(&mut out);
        ekle_yaz_w16_off(dst, stride, 0, 0, &out);
    }
}

#[target_feature(enable = "avx2")]
fn dct_dct_64x16(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    if eob <= 35 {
        let mut qtr = [_mm256_setzero_si256(); 8];
        for (x, qx) in qtr.iter_mut().enumerate() {
            *qx = yukle_i16x16(chunk(coeff, x * 16));
        }
        sifirla_katsayi::<512>(coeff);
        let mut cols = x16::idct64_eighth(&qtr);
        x16::round_shift::<64, 2>(&mut cols);
        for s in 0..4 {
            let m = chunk_mut::<__m256i, 16>(&mut cols, s * 16);
            transpose_16x16_ymm(m);
            let mut out = x16::idct16_half(chunk(m, 0));
            x16::round_pass2(&mut out);
            ekle_yaz_w16_off(dst, stride, 0, s * 16, &out);
        }
        return;
    }

    let mut cols = if eob <= 150 {
        let mut left = [_mm256_setzero_si256(); 16];
        for (x, lx) in left.iter_mut().enumerate() {
            *lx = yukle_i16x16(chunk(coeff, x * 16));
        }
        sifirla_katsayi::<512>(coeff);
        x16::idct64_qtr(&left)
    } else {
        let mut all = [_mm256_setzero_si256(); 32];
        for (x, ax) in all.iter_mut().enumerate() {
            *ax = yukle_i16x16(chunk(coeff, x * 16));
        }
        sifirla_katsayi::<512>(coeff);
        x16::idct64(&all)
    };
    x16::round_shift::<64, 2>(&mut cols);

    for s in 0..4 {
        let m = chunk_mut::<__m256i, 16>(&mut cols, s * 16);
        transpose_16x16_ymm(m);
        x16::idct16(m);
        x16::round_pass2(m);
        ekle_yaz_w16_off(dst, stride, 0, s * 16, m);
    }
}

#[target_feature(enable = "avx2")]
fn dct_dct_32x64(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    let r2 = _mm256_set1_epi16(23168);
    if eob < 136 {
        let mut tl = [_mm256_setzero_si256(); 16];
        for (x, tlx) in tl.iter_mut().enumerate() {
            *tlx = _mm256_mulhrs_epi16(yukle_i16x16(chunk(coeff, x * 32)), r2);
        }
        sifirla_katsayi::<1024>(coeff);
        let mut p1 = x16::idct32_half(&tl);
        x16::round_shift::<32, 1>(&mut p1);
        transpose_16x16_ymm(chunk_mut(&mut p1, 0));
        transpose_16x16_ymm(chunk_mut(&mut p1, 16));

        let mut lo_out = x16::idct64_qtr(chunk(&p1, 0));
        let mut hi_out = x16::idct64_qtr(chunk(&p1, 16));
        x16::round_pass2(&mut lo_out);
        x16::round_pass2(&mut hi_out);
        ekle_yaz_w32(dst, stride, &lo_out, &hi_out);
    } else {
        let mut top = [_mm256_setzero_si256(); 32];
        let mut bot = [_mm256_setzero_si256(); 32];
        for (x, (tx, bx)) in top.iter_mut().zip(bot.iter_mut()).enumerate() {
            *tx = _mm256_mulhrs_epi16(yukle_i16x16(chunk(coeff, x * 32)), r2);
            *bx = _mm256_mulhrs_epi16(yukle_i16x16(chunk(coeff, x * 32 + 16)), r2);
        }
        sifirla_katsayi::<1024>(coeff);
        x16::idct32(&mut top);
        x16::round_shift::<32, 1>(&mut top);
        x16::idct32(&mut bot);
        x16::round_shift::<32, 1>(&mut bot);

        transpose_16x16_ymm(chunk_mut(&mut top, 0));
        transpose_16x16_ymm(chunk_mut(&mut top, 16));
        transpose_16x16_ymm(chunk_mut(&mut bot, 0));
        transpose_16x16_ymm(chunk_mut(&mut bot, 16));
        top[16..32].swap_with_slice(&mut bot[0..16]);

        let mut lo_out = x16::idct64(&top);
        let mut hi_out = x16::idct64(&bot);
        x16::round_pass2(&mut lo_out);
        x16::round_pass2(&mut hi_out);
        ekle_yaz_w32(dst, stride, &lo_out, &hi_out);
    }
}

#[target_feature(enable = "avx2")]
fn dct_dct_64x32(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    let r2 = _mm256_set1_epi16(23168);
    if eob < 136 {
        let mut tl = [_mm256_setzero_si256(); 16];
        for (x, tlx) in tl.iter_mut().enumerate() {
            *tlx = _mm256_mulhrs_epi16(yukle_i16x16(chunk(coeff, x * 32)), r2);
        }
        sifirla_katsayi::<1024>(coeff);
        let mut p1 = x16::idct64_qtr(&tl);
        x16::round_shift::<64, 1>(&mut p1);

        for s in 0..4 {
            let m = chunk_mut::<__m256i, 16>(&mut p1, s * 16);
            transpose_16x16_ymm(m);
            let mut out = x16::idct32_half(m);
            x16::round_pass2(&mut out);
            ekle_yaz_w16_off(dst, stride, 0, s * 16, &out);
        }
    } else {
        let mut top_in = [_mm256_setzero_si256(); 32];
        let mut bot_in = [_mm256_setzero_si256(); 32];
        for (x, (tx, bx)) in top_in.iter_mut().zip(bot_in.iter_mut()).enumerate() {
            *tx = _mm256_mulhrs_epi16(yukle_i16x16(chunk(coeff, x * 32)), r2);
            *bx = _mm256_mulhrs_epi16(yukle_i16x16(chunk(coeff, x * 32 + 16)), r2);
        }
        sifirla_katsayi::<1024>(coeff);
        let mut p1_top = x16::idct64(&top_in);
        let mut p1_bot = x16::idct64(&bot_in);
        x16::round_shift::<64, 1>(&mut p1_top);
        x16::round_shift::<64, 1>(&mut p1_bot);

        for s in 0..4 {
            let qt = chunk_mut::<__m256i, 16>(&mut p1_top, s * 16);
            let qb = chunk_mut::<__m256i, 16>(&mut p1_bot, s * 16);
            transpose_16x16_ymm(qt);
            transpose_16x16_ymm(qb);

            let mut col = [_mm256_setzero_si256(); 32];
            col[..16].copy_from_slice(qt);
            col[16..].copy_from_slice(qb);
            x16::idct32(&mut col);
            x16::round_pass2(&mut col);
            ekle_yaz_w16_off(dst, stride, 0, s * 16, &col);
        }
    }
}

#[target_feature(enable = "avx2")]
fn dct_dct_64x64(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    if eob < 136 {
        let mut tl = [_mm256_setzero_si256(); 16];
        for (x, tlx) in tl.iter_mut().enumerate() {
            *tlx = yukle_i16x16(chunk(coeff, x * 32));
        }
        sifirla_katsayi::<1024>(coeff);
        let mut p1 = x16::idct64_qtr(&tl);
        x16::round_shift::<64, 2>(&mut p1);

        for s in 0..4 {
            let m = chunk_mut::<__m256i, 16>(&mut p1, s * 16);
            transpose_16x16_ymm(m);
            let mut out = x16::idct64_qtr(m);
            x16::round_pass2(&mut out);
            ekle_yaz_w16_off(dst, stride, 0, s * 16, &out);
        }
    } else {
        let mut top_in = [_mm256_setzero_si256(); 32];
        let mut bot_in = [_mm256_setzero_si256(); 32];
        for (x, (tx, bx)) in top_in.iter_mut().zip(bot_in.iter_mut()).enumerate() {
            *tx = yukle_i16x16(chunk(coeff, x * 32));
            *bx = yukle_i16x16(chunk(coeff, x * 32 + 16));
        }
        sifirla_katsayi::<1024>(coeff);
        let mut p1_top = x16::idct64(&top_in);
        let mut p1_bot = x16::idct64(&bot_in);
        x16::round_shift::<64, 2>(&mut p1_top);
        x16::round_shift::<64, 2>(&mut p1_bot);

        for s in 0..4 {
            let qt = chunk_mut::<__m256i, 16>(&mut p1_top, s * 16);
            let qb = chunk_mut::<__m256i, 16>(&mut p1_bot, s * 16);
            transpose_16x16_ymm(qt);
            transpose_16x16_ymm(qb);

            let mut col = [_mm256_setzero_si256(); 32];
            col[..16].copy_from_slice(qt);
            col[16..].copy_from_slice(qb);
            let mut out = x16::idct64(&col);
            x16::round_pass2(&mut out);
            ekle_yaz_w16_off(dst, stride, 0, s * 16, &out);
        }
    }
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn itx8_buyuk_avx2_cekirdek<const TX: u8, const TXTP: u8>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32; 1024],
    eob: usize,
) {
    let tx = const { TxSize::from_index(TX as usize).unwrap() };
    let txtp = const { TxType::from_index(TXTP as usize).unwrap() };

    if matches!(txtp, TxType::DctDct) && eob == 0 {
        match tx {
            TxSize::Rtx8x32 => dc_only::<8, 32, 2, false>(dst, dst_stride, coeff),
            TxSize::Rtx32x8 => dc_only::<32, 8, 2, false>(dst, dst_stride, coeff),
            TxSize::Rtx16x32 => dc_only::<16, 32, 1, true>(dst, dst_stride, coeff),
            TxSize::Rtx32x16 => dc_only::<32, 16, 1, true>(dst, dst_stride, coeff),
            TxSize::Tx32x32 => dc_only::<32, 32, 2, false>(dst, dst_stride, coeff),
            TxSize::Rtx16x64 => dc_only::<16, 64, 2, false>(dst, dst_stride, coeff),
            TxSize::Rtx64x16 => dc_only::<64, 16, 2, false>(dst, dst_stride, coeff),
            TxSize::Rtx32x64 => dc_only::<32, 64, 1, true>(dst, dst_stride, coeff),
            TxSize::Rtx64x32 => dc_only::<64, 32, 1, true>(dst, dst_stride, coeff),
            TxSize::Tx64x64 => dc_only::<64, 64, 2, false>(dst, dst_stride, coeff),
            _ => unreachable!(),
        }
        return;
    }

    match (tx, txtp) {
        (TxSize::Rtx8x32, TxType::DctDct) => dct_dct_8x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx8x32, TxType::Idtx) => idtx_8x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x8, TxType::DctDct) => dct_dct_32x8(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x8, TxType::Idtx) => idtx_32x8(dst, dst_stride, coeff, eob),
        (TxSize::Rtx16x32, TxType::DctDct) => dct_dct_16x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx16x32, TxType::Idtx) => idtx_16x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x16, TxType::DctDct) => dct_dct_32x16(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x16, TxType::Idtx) => idtx_32x16(dst, dst_stride, coeff, eob),
        (TxSize::Tx32x32, TxType::DctDct) => dct_dct_32x32(dst, dst_stride, coeff, eob),
        (TxSize::Tx32x32, TxType::Idtx) => idtx_32x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx16x64, TxType::DctDct) => dct_dct_16x64(dst, dst_stride, coeff, eob),
        (TxSize::Rtx64x16, TxType::DctDct) => dct_dct_64x16(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x64, TxType::DctDct) => dct_dct_32x64(dst, dst_stride, coeff, eob),
        (TxSize::Rtx64x32, TxType::DctDct) => dct_dct_64x32(dst, dst_stride, coeff, eob),
        (TxSize::Tx64x64, TxType::DctDct) => dct_dct_64x64(dst, dst_stride, coeff, eob),
        _ => unreachable!(),
    }
}

/// 8 bpc AVX2 büyük (`max(w, h) ∈ {32, 64}`) `ItxFn` giriş sarmalayıcısı (monomorfize `(TX, TXTP)`).
fn itx8_buyuk_avx2_giris<const TX: u8, const TXTP: u8>(
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
        || !itx8_sozlesmesi(dst.len(), dst_stride, coeff.len(), eob, tx, txtp)
    {
        itx_skaler_devret::<8>(dst, dst_stride, coeff, eob, tx, txtp);
        return;
    }
    let coeff1024 = chunk_mut::<i32, 1024>(coeff, 0);
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`itx8_buyuk_avx2_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır.
    // (2) sınır: `itx8_sozlesmesi` `dst`'nin `(h − 1)·dst_stride + w` baytı ve
    // `coeff`'un `1024` öğeyi kapsadığını garanti eder; çekirdek içindeki bütün
    // erişimler sabit boyutlu `chunk`/`chunk_mut` dizileri üzerindendir.
    // (3) hizalama/takma ad: `dst` ve `coeff1024` ayrık `&mut` referanslardır,
    // yalnız hizasız erişim kullanılır.
    unsafe {
        itx8_buyuk_avx2_cekirdek::<TX, TXTP>(dst, dst_stride, coeff1024, eob);
    }
}

/// 15 adet `max(w, h) ∈ {32, 64}` 8 bpc AVX2 `ItxFn` tablosu (yalnız [`Avx2Token`] ile).
#[must_use]
pub(crate) fn itx8_buyuk_avx2_fn(_token: Avx2Token, tx: TxSize, txtp: TxType) -> Option<ItxFn> {
    match (tx, txtp) {
        (TxSize::Tx32x32, TxType::DctDct) => Some(itx8_buyuk_avx2_giris::<3, 0>),
        (TxSize::Tx32x32, TxType::Idtx) => Some(itx8_buyuk_avx2_giris::<3, 9>),
        (TxSize::Tx64x64, TxType::DctDct) => Some(itx8_buyuk_avx2_giris::<4, 0>),
        (TxSize::Rtx16x32, TxType::DctDct) => Some(itx8_buyuk_avx2_giris::<9, 0>),
        (TxSize::Rtx16x32, TxType::Idtx) => Some(itx8_buyuk_avx2_giris::<9, 9>),
        (TxSize::Rtx32x16, TxType::DctDct) => Some(itx8_buyuk_avx2_giris::<10, 0>),
        (TxSize::Rtx32x16, TxType::Idtx) => Some(itx8_buyuk_avx2_giris::<10, 9>),
        (TxSize::Rtx32x64, TxType::DctDct) => Some(itx8_buyuk_avx2_giris::<11, 0>),
        (TxSize::Rtx64x32, TxType::DctDct) => Some(itx8_buyuk_avx2_giris::<12, 0>),
        (TxSize::Rtx8x32, TxType::DctDct) => Some(itx8_buyuk_avx2_giris::<15, 0>),
        (TxSize::Rtx8x32, TxType::Idtx) => Some(itx8_buyuk_avx2_giris::<15, 9>),
        (TxSize::Rtx32x8, TxType::DctDct) => Some(itx8_buyuk_avx2_giris::<16, 0>),
        (TxSize::Rtx32x8, TxType::Idtx) => Some(itx8_buyuk_avx2_giris::<16, 9>),
        (TxSize::Rtx16x64, TxType::DctDct) => Some(itx8_buyuk_avx2_giris::<17, 0>),
        (TxSize::Rtx64x16, TxType::DctDct) => Some(itx8_buyuk_avx2_giris::<18, 0>),
        _ => None,
    }
}
