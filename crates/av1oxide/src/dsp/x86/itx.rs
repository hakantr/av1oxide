//! 8 bpc kayıplı / genel ters dönüşüm (`w, h ≤ 16`) x86_64 AVX2 çekirdekleri — F4-ITX a (P43 Adım 2).
//!
//! Bu modül dav1d'nin `src/x86/itx_avx2.asm` ve `src/x86/itx.h:208-218` dosyalarındaki
//! `w, h ≤ 16` boyutlu **140** adet WHT-dışı 8 bpc AVX2 ters dönüşüm kaydını saf kararlı
//! Rust `core::arch::x86_64` intrinsics ile gerçekler (`.asm`, `.S`, `asm!`, `global_asm!`
//! ve dış C bağımlılığı yoktur; CEK-02):
//!
//! - `TX_4X4`, `RTX_4X8`, `RTX_8X4`, `TX_8X8`, `RTX_4X16`, `RTX_16X4`, `RTX_8X16`,
//!   `RTX_16X8`: 16 WHT-dışı `TxType` türünün tamamı (`8 × 16 = 128` kayıt),
//! - `TX_16X16`: desteklenen 12 `TxType` türü (`DctDct`, `AdstDct`, `DctAdst`, `AdstAdst`,
//!   `FlipadstDct`, `DctFlipadst`, `FlipadstFlipadst`, `AdstFlipadst`, `FlipadstAdst`,
//!   `Idtx`, `VDct`, `HDct` = `12` kayıt).
//!
//! ## Sözleşme, P34 izolasyonu ve ISA sınırı
//!
//! - **P34 modül izolasyonu:** Bu modül `itx_buyuk` ya da `itx_wht` ile hiçbir
//!   `#[target_feature]` yardımcısı paylaşmaz; bütün 1B dönüşüm ve transpoz yardımcıları
//!   modül içinde kendi kopyalarıyla tanımlıdır.
//! - **Katsayı sıfırlama sözleşmesi:**
//!   - `txtp == TxType::DctDct && eob == 0` (DC-only hızlı yolu): **yalnız `coeff[0] = 0`**
//!     sıfırlanır; `coeff[1..1024]` öğelerine dokunulmaz.
//!   - Genel yol (`eob ≥ 1` veya `txtp != TxType::DctDct`): `coeff[..w * h]` sıfırlanır;
//!     `coeff[w * h..1024]` kuyruğuna dokunulmaz.
//! - **Sınır güvenliği:** `dst` üzerinde her satırda tam `w` bayt (`4`, `8` veya `16` bayt)
//!   okunur ve yazılır; tekil `dst_stride` (`w + 1`, `w + 7`, `w + 13`) ve `SentinelBuffer`
//!   altında taşma `0`'dır. Sözleşme dışı girdide paniklemeden [`itx_skaler_devret`] ile
//!   sayaç artırılıp skaler monomorfa devredilir.
//! - **ISA sınırı:** Yalnız `#[target_feature(enable = "avx2")]` kullanılır; EVEX (`zmm` /
//!   `{k*}`) yönergesi üretilmez. Modülde tek `unsafe` bloğu belirteçli giriş
//!   sarmalayıcısındadır ([`itx8_avx2_giris`]).

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_add_epi32, _mm_adds_epi16, _mm_adds_epu8, _mm_avg_epu16,
    _mm_cvtepu8_epi16, _mm_cvtsi32_si128, _mm_cvtsi64_si128, _mm_madd_epi16, _mm_mulhrs_epi16,
    _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi8, _mm_set1_epi16, _mm_set1_epi32,
    _mm_setr_epi32, _mm_setzero_si128, _mm_srai_epi16, _mm_srai_epi32, _mm_srli_si128,
    _mm_subs_epi16, _mm_subs_epu8, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpacklo_epi16,
    _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm_xor_si128, _mm256_add_epi16, _mm256_add_epi32,
    _mm256_adds_epi16, _mm256_avg_epu16, _mm256_castsi256_si128, _mm256_cvtepu8_epi16,
    _mm256_extracti128_si256, _mm256_madd_epi16, _mm256_mulhrs_epi16, _mm256_packs_epi32,
    _mm256_packus_epi16, _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set_m128i,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi32, _mm256_setzero_si256,
    _mm256_srai_epi16, _mm256_srai_epi32, _mm256_srli_si256, _mm256_subs_epi16,
    _mm256_unpackhi_epi16, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi16,
    _mm256_unpacklo_epi32, _mm256_unpacklo_epi64, _mm256_xor_si256,
};

use super::mem::{chunk, chunk_mut, load_u8x16, store_i32x8, store_u8x4, store_u8x8, store_u8x16};
use crate::cpu::Avx2Token;
use crate::dsp::itx::{ItxFn, TxSize, TxType, is_supported, itx_skaler_devret, itx8_sozlesmesi};
use crate::dsp::scalar::itx_1d::Tx1dType;
use crate::frame::BitDepth;

/// İki `i16` katsayısını `_mm*_madd_epi16` için tek `i32` sabitine paketler (`[lo, hi]`).
#[inline(always)]
const fn pair(lo: i16, hi: i16) -> i32 {
    (lo as u16 as i32) | ((hi as u16 as i32) << 16)
}

/// 4 × `i32` dilimini `__m128i`'ye yükler.
#[target_feature(enable = "avx2")]
#[inline]
fn yukle_i32x4(c: &[i32; 4]) -> __m128i {
    _mm_setr_epi32(c[0], c[1], c[2], c[3])
}

/// 8 × `i32` dilimini `__m256i`'ye yükler.
#[target_feature(enable = "avx2")]
#[inline]
fn yukle_i32x8(c: &[i32; 8]) -> __m256i {
    _mm256_setr_epi32(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7])
}

/// 4 × `i32` dilimini doyurmalı `i16`'ya çevirip alt 4 şeride (`[c0..c3, 0, 0, 0, 0]`) yükler.
#[target_feature(enable = "avx2")]
#[inline]
fn yukle_i16x4(c: &[i32; 4]) -> __m128i {
    _mm_packs_epi32(yukle_i32x4(c), _mm_setzero_si128())
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

/// 16 × `i32` dilimini doğal sırada doyurmalı `16 × i16` (`__m256i`) olarak yükler.
#[target_feature(enable = "avx2")]
#[inline]
fn yukle_i16x16(c: &[i32; 16]) -> __m256i {
    yukle_2col_8(chunk(c, 0), chunk(c, 8))
}

/// `coeff[..N]` bölgesini (`N` 8'in katı) 256-bit sıfır saklamalarıyla temizler.
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

/// Şerit-paralel 1B ters dönüşüm çekirdekleri:
/// - `x4` (`__m128i`, alt 4 şerit etkin, `$HALF_LANE = true`),
/// - `x8` (`__m256i` Split-YMM: her 128-bit yarıda alt 4 şerit etkin = 8 şerit, `$HALF_LANE = true`),
/// - `x16` (`__m256i`, tam 16 şerit etkin, `$HALF_LANE = false`).
macro_rules! itx_1d_impl {
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
        $mulhrs16:ident,
        $srai16:ident,
        $xor:ident,
        $avg_epu16:ident
    ) => {
        mod $mod_ad {
            use super::*;

            /// İşaretli yuvarlamalı ortalama: `(a + b + 1) >> 1` (`[-32768, 32767]` tam aralıkta taşmasız).
            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn savg16(a: $V, b: $V) -> $V {
                let bias = $set1_16(-32768);
                $xor($avg_epu16($xor(a, bias), $xor(b, bias)), bias)
            }

            /// `(out1, out2) = (((a*c1 - b*c2 + 2048) >> 12), ((a*c2 + b*c1 + 2048) >> 12))`.
            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn mulsub(a: $V, b: $V, c1: i16, c2: i16) -> ($V, $V) {
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

            /// `b == 0` olduğunda `(a*c1 + 2048) >> 12` ve `(a*c2 + 2048) >> 12` (`pmulhrsw(a, c * 8)`).
            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn mul_single(a: $V, c1: i16, c2: i16) -> ($V, $V) {
                (
                    $mulhrs16(a, $set1_16(c1 * 8)),
                    $mulhrs16(a, $set1_16(c2 * 8)),
                )
            }

            /// `inv_dct4_1d_c` (`src/itx_1d.c:66-96`).
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

            /// `c[4..8] == 0` olduğunda `idct8`.
            #[target_feature(enable = "avx2")]
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
            #[target_feature(enable = "avx2")]
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
            #[target_feature(enable = "avx2")]
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
            #[target_feature(enable = "avx2")]
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
            #[target_feature(enable = "avx2")]
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
                    let lo =
                        $srai32::<12>($add32($add32($madd16(p01_lo, a), $madd16(p23_lo, b)), rnd));
                    if $HALF_LANE {
                        $packs32(lo, lo)
                    } else {
                        let hi = $srai32::<12>($add32(
                            $add32($madd16(p01_hi, a), $madd16(p23_hi, b)),
                            rnd,
                        ));
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
            #[target_feature(enable = "avx2")]
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

            #[target_feature(enable = "avx2")]
            #[inline]
            fn adst_finish<const N: usize>(c: &mut [$V; N], shift_mul: i16, flip: bool) {
                if shift_mul == 0 {
                    let z = $zero();
                    let mut i = 1;
                    while i < N {
                        c[i] = $subs16(z, c[i]);
                        i += 2;
                    }
                } else {
                    let p = $set1_16(shift_mul);
                    let n = $set1_16(-shift_mul);
                    let mut i = 0;
                    while i < N {
                        c[i] = $mulhrs16(c[i], p);
                        c[i + 1] = $mulhrs16(c[i + 1], n);
                        i += 2;
                    }
                }
                if flip {
                    c.reverse();
                }
            }

            /// `inv_adst16_1d_c` / `inv_flipadst16_1d_c` (`src/itx_1d.c:854-962`).
            #[target_feature(enable = "avx2")]
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
            #[target_feature(enable = "avx2")]
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
                        // Identity4: `val + ((val * 1697 + 2048) >> 12)` ile `>> SHIFT` kaynaşık.
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
            #[target_feature(enable = "avx2")]
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
                        // Identity8: `(2 * val + rnd) >> SHIFT`.
                        // SHIFT == 0 -> `2 * val`; SHIFT == 1 -> `(2 * val + 1) >> 1 = val` (no-op).
                        if SHIFT == 0 {
                            for v in c.iter_mut() {
                                *v = $adds16(*v, *v);
                            }
                        }
                    }
                }
            }

            /// 1. geçiş: 16 noktalı yatay dönüşüm + `shift` (`1` veya `2`) yuvarlaması.
            #[target_feature(enable = "avx2")]
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
                        // Identity16: `(2 * val + ((val * 1697 + 1024) >> 11) + rnd) >> SHIFT`.
                        let k = $set1_16(27152); // 1697 * 16
                        let r1 = $set1_16(16384);
                        for v in c.iter_mut() {
                            let t = $mulhrs16(*v, k);
                            *v = if SHIFT == 1 {
                                $adds16(*v, $mulhrs16(t, r1))
                            } else {
                                // SHIFT == 2: `*v` ve `t >> 1` aynı işarettedir (`v = -1` için `t = -1`).
                                $avg_epu16(*v, $srai16::<1>(t))
                            };
                        }
                    }
                }
            }

            /// 2. geçiş: 4 noktalı dikey dönüşüm + son `(c + 8) >> 4` yuvarlaması.
            #[target_feature(enable = "avx2")]
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
            #[target_feature(enable = "avx2")]
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
                        // `(2 * val + 8) >> 4 == (val + 4) >> 3`.
                        let r3 = $set1_16(4096);
                        for v in c.iter_mut() {
                            *v = $mulhrs16(*v, r3);
                        }
                    }
                }
            }

            /// 2. geçiş: 16 noktalı dikey dönüşüm + son `(c + 8) >> 4` yuvarlaması.
            #[target_feature(enable = "avx2")]
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
        }
    };
}

itx_1d_impl!(
    x4,
    __m128i,
    true,
    _mm_set1_epi16,
    _mm_set1_epi32,
    _mm_setzero_si128,
    _mm_unpacklo_epi16,
    _mm_unpackhi_epi16,
    _mm_madd_epi16,
    _mm_add_epi32,
    _mm_srai_epi32,
    _mm_packs_epi32,
    _mm_adds_epi16,
    _mm_subs_epi16,
    _mm_mulhrs_epi16,
    _mm_srai_epi16,
    _mm_xor_si128,
    _mm_avg_epu16
);

itx_1d_impl!(
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
    _mm256_mulhrs_epi16,
    _mm256_srai_epi16,
    _mm256_xor_si256,
    _mm256_avg_epu16
);

itx_1d_impl!(
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
    _mm256_srai_epi16,
    _mm256_xor_si256,
    _mm256_avg_epu16
);

// ---------------------------------------------------------------------------
// Matris transpoz yardımcıları (`i16`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
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
fn transpose_4x16_to_x4(cols: &[__m256i; 4]) -> [__m128i; 16] {
    let a0 = _mm256_unpacklo_epi16(cols[0], cols[1]);
    let a1 = _mm256_unpackhi_epi16(cols[0], cols[1]);
    let a2 = _mm256_unpacklo_epi16(cols[2], cols[3]);
    let a3 = _mm256_unpackhi_epi16(cols[2], cols[3]);
    let r01 = _mm256_unpacklo_epi32(a0, a2);
    let r23 = _mm256_unpackhi_epi32(a0, a2);
    let r45 = _mm256_unpacklo_epi32(a1, a3);
    let r67 = _mm256_unpackhi_epi32(a1, a3);

    let lo01 = _mm256_castsi256_si128(r01);
    let hi01 = _mm256_extracti128_si256::<1>(r01);
    let lo23 = _mm256_castsi256_si128(r23);
    let hi23 = _mm256_extracti128_si256::<1>(r23);
    let lo45 = _mm256_castsi256_si128(r45);
    let hi45 = _mm256_extracti128_si256::<1>(r45);
    let lo67 = _mm256_castsi256_si128(r67);
    let hi67 = _mm256_extracti128_si256::<1>(r67);

    [
        lo01,
        _mm_srli_si128::<8>(lo01),
        lo23,
        _mm_srli_si128::<8>(lo23),
        lo45,
        _mm_srli_si128::<8>(lo45),
        lo67,
        _mm_srli_si128::<8>(lo67),
        hi01,
        _mm_srli_si128::<8>(hi01),
        hi23,
        _mm_srli_si128::<8>(hi23),
        hi45,
        _mm_srli_si128::<8>(hi45),
        hi67,
        _mm_srli_si128::<8>(hi67),
    ]
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

// ---------------------------------------------------------------------------
// Sınır-güvenli piksel ekleme ve saklama yardımcıları
// ---------------------------------------------------------------------------

/// `w = 4`: her satırda tam 4 bayt okur, `del` alt 4 `i16` şeridini ekler, `[0, 255]` kırpıp 4 bayt yazar.
#[target_feature(enable = "avx2")]
#[inline]
fn ekle_yaz_w4<const H: usize>(dst: &mut [u8], stride: usize, rows: &[__m128i; H]) {
    const { assert!(H.is_multiple_of(2)) };
    let mut y = 0;
    while y < H {
        let d0 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, y * stride)));
        let d1 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, (y + 1) * stride)));
        let px01 = _mm_cvtepu8_epi16(_mm_unpacklo_epi32(d0, d1));
        let del01 = _mm_unpacklo_epi64(rows[y], rows[y + 1]);
        let s = _mm_add_epi16(px01, del01);
        let p = _mm_packus_epi16(s, s);
        store_u8x4(chunk_mut(dst, y * stride), p);
        store_u8x4(chunk_mut(dst, (y + 1) * stride), _mm_srli_si128::<4>(p));
        y += 2;
    }
}

/// `w = 8` (bitişik `__m128i`): her satırda tam 8 bayt okur, `8 × i16` ekler, `[0, 255]` kırpıp 8 bayt yazar.
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

/// `w = 8` (Split-YMM `__m256i`: `[x0..3, _ || x4..7, _]`): her satırda tam 8 bayt okur, ekler ve yazar.
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

/// `w = 16`: her satırda tam 16 bayt okur, `16 × i16` ekler, `[0, 255]` kırpıp 16 bayt yazar.
#[target_feature(enable = "avx2")]
#[inline]
fn ekle_yaz_w16<const H: usize>(dst: &mut [u8], stride: usize, rows: &[__m256i; H]) {
    const { assert!(H.is_multiple_of(2)) };
    let mut y = 0;
    while y < H {
        let px0 = _mm256_cvtepu8_epi16(load_u8x16(chunk(dst, y * stride)));
        let px1 = _mm256_cvtepu8_epi16(load_u8x16(chunk(dst, (y + 1) * stride)));
        let s0 = _mm256_add_epi16(px0, rows[y]);
        let s1 = _mm256_add_epi16(px1, rows[y + 1]);
        let p = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(s0, s1));
        store_u8x16(chunk_mut(dst, y * stride), _mm256_castsi256_si128(p));
        store_u8x16(
            chunk_mut(dst, (y + 1) * stride),
            _mm256_extracti128_si256::<1>(p),
        );
        y += 2;
    }
}

/// DC-only hızlı yolu (`txtp == DCT_DCT && eob == 0`, `src/itx_tmpl.c:58-70`).
///
/// Yalnız `coeff[0] = 0` yapar; `coeff[1..1024]` öğelerine dokunmaz.
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
// Boyut bazında 2B AVX2 çekirdek gövdeleri (`w, h ≤ 16`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
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
        let lo = _mm256_castsi256_si128(v2); // [col0, col2]
        let hi = _mm256_extracti128_si256::<1>(v2); // [col1, col3]
        let a0 = _mm_unpacklo_epi16(lo, hi); // col0, col1
        let a1 = _mm_unpackhi_epi16(lo, hi); // col2, col3
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

#[target_feature(enable = "avx2")]
fn cekirdek_4x8<const H1D: u8, const V1D: u8>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32; 1024],
    _eob: usize,
) {
    let r2 = _mm256_set1_epi16(23168); // 181 * 128 (rect2)
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

#[target_feature(enable = "avx2")]
#[inline]
fn cekirdek_8x4<const H1D: u8, const V1D: u8>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32; 1024],
    _eob: usize,
) {
    if H1D == 2 {
        let r2 = _mm256_set1_epi16(23168);
        let p0145 = _mm256_packs_epi32(yukle_i32x8(chunk(coeff, 0)), yukle_i32x8(chunk(coeff, 16)));
        let p2367 = _mm256_packs_epi32(yukle_i32x8(chunk(coeff, 8)), yukle_i32x8(chunk(coeff, 24)));
        sifirla_katsayi::<32>(coeff);
        let s0145 = _mm256_mulhrs_epi16(p0145, r2);
        let s0145 = _mm256_adds_epi16(s0145, s0145);
        let s2367 = _mm256_mulhrs_epi16(p2367, r2);
        let s2367 = _mm256_adds_epi16(s2367, s2367);
        let mut c04 = _mm256_permute4x64_epi64::<0x50>(s0145);
        let mut c15 = _mm256_permute4x64_epi64::<0xFA>(s0145);
        let mut c26 = _mm256_permute4x64_epi64::<0x50>(s2367);
        let mut c37 = _mm256_permute4x64_epi64::<0xFA>(s2367);
        if V1D == 2 {
            let k = _mm256_set1_epi16(13576);
            let r4 = _mm256_set1_epi16(2048);
            for v in [&mut c04, &mut c15, &mut c26, &mut c37] {
                *v = _mm256_mulhrs_epi16(_mm256_adds_epi16(*v, _mm256_mulhrs_epi16(*v, k)), r4);
            }
            let rows = transpose_4x4_in_ymm(c04, c15, c26, c37);
            ekle_yaz_w8_split(dst, stride, &rows);
            return;
        }
        let mut rows = transpose_4x4_in_ymm(c04, c15, c26, c37);
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

#[target_feature(enable = "avx2")]
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

#[target_feature(enable = "avx2")]
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
        yukle_i16x16(chunk(coeff, 0)),
        yukle_i16x16(chunk(coeff, 16)),
        yukle_i16x16(chunk(coeff, 32)),
        yukle_i16x16(chunk(coeff, 48)),
    ];
    sifirla_katsayi::<64>(coeff);
    x16::pass1_4::<H1D, 1>(&mut cols);
    if V1D == 2 {
        let k = _mm256_set1_epi16(27152);
        let r4 = _mm256_set1_epi16(2048);
        for v in &mut cols {
            let t = _mm256_mulhrs_epi16(*v, k);
            let s = _mm256_adds_epi16(_mm256_adds_epi16(*v, *v), t);
            *v = _mm256_mulhrs_epi16(s, r4);
        }
        let rows = transpose_4x16_to_x4(&cols);
        ekle_yaz_w4(dst, stride, &rows);
        return;
    }
    let mut rows = transpose_4x16_to_x4(&cols);
    x4::pass2_16_eob::<V1D>(&mut rows, false);
    ekle_yaz_w4(dst, stride, &rows);
}

#[target_feature(enable = "avx2")]
fn cekirdek_16x4<const H1D: u8, const V1D: u8>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32; 1024],
    eob: usize,
) {
    if H1D == 2 {
        let k16 = _mm256_set1_epi16(27152);
        let r1 = _mm256_set1_epi16(16384);
        let k4 = _mm256_set1_epi16(13576);
        let r4 = _mm256_set1_epi16(2048);
        let load_scale = |c: &[i32; 16]| -> __m256i {
            let v0 = _mm256_packs_epi32(yukle_i32x8(chunk(c, 0)), yukle_i32x8(chunk(c, 8)));
            let v1 = _mm256_adds_epi16(v0, _mm256_mulhrs_epi16(_mm256_mulhrs_epi16(v0, k16), r1));
            if V1D == 2 {
                _mm256_mulhrs_epi16(_mm256_adds_epi16(v1, _mm256_mulhrs_epi16(v1, k4)), r4)
            } else {
                v1
            }
        };
        // `p0`: lane 0 = [col0, col2], lane 1 = [col1, col3]
        let p0 = load_scale(chunk(coeff, 0));
        let p1 = load_scale(chunk(coeff, 16));
        let p2 = load_scale(chunk(coeff, 32));
        let p3 = load_scale(chunk(coeff, 48));
        sifirla_katsayi::<64>(coeff);

        let a0 = _mm256_unpacklo_epi16(p0, p1);
        let a1 = _mm256_unpackhi_epi16(p0, p1);
        let a2 = _mm256_unpacklo_epi16(p2, p3);
        let a3 = _mm256_unpackhi_epi16(p2, p3);
        // `a0`: lane 0 = col0,4 (y=0..4), lane 1 = col1,5 (y=0..4)
        let b0 = _mm256_unpacklo_epi16(
            _mm256_permute2x128_si256::<0x20>(a0, a2),
            _mm256_permute2x128_si256::<0x31>(a0, a2),
        );
        let b1 = _mm256_unpackhi_epi16(
            _mm256_permute2x128_si256::<0x20>(a0, a2),
            _mm256_permute2x128_si256::<0x31>(a0, a2),
        );
        let b2 = _mm256_unpacklo_epi16(
            _mm256_permute2x128_si256::<0x20>(a1, a3),
            _mm256_permute2x128_si256::<0x31>(a1, a3),
        );
        let b3 = _mm256_unpackhi_epi16(
            _mm256_permute2x128_si256::<0x20>(a1, a3),
            _mm256_permute2x128_si256::<0x31>(a1, a3),
        );
        let mut rows = [
            _mm256_unpacklo_epi32(b0, b2),
            _mm256_unpackhi_epi32(b0, b2),
            _mm256_unpacklo_epi32(b1, b3),
            _mm256_unpackhi_epi32(b1, b3),
        ];
        if V1D != 2 {
            x16::pass2_4::<V1D>(&mut rows);
        }
        ekle_yaz_w16(dst, stride, &rows);
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

    let t0 = transpose_4x4_in_ymm(
        _mm256_set_m128i(cols[8], cols[0]),
        _mm256_set_m128i(cols[9], cols[1]),
        _mm256_set_m128i(cols[10], cols[2]),
        _mm256_set_m128i(cols[11], cols[3]),
    );
    let t1 = transpose_4x4_in_ymm(
        _mm256_set_m128i(cols[12], cols[4]),
        _mm256_set_m128i(cols[13], cols[5]),
        _mm256_set_m128i(cols[14], cols[6]),
        _mm256_set_m128i(cols[15], cols[7]),
    );
    let mut rows = [
        _mm256_unpacklo_epi64(t0[0], t1[0]),
        _mm256_unpacklo_epi64(t0[1], t1[1]),
        _mm256_unpacklo_epi64(t0[2], t1[2]),
        _mm256_unpacklo_epi64(t0[3], t1[3]),
    ];
    x16::pass2_4::<V1D>(&mut rows);
    ekle_yaz_w16(dst, stride, &rows);
}

#[target_feature(enable = "avx2")]
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

    let r2 = _mm256_set1_epi16(23168);
    let mut cols = [_mm256_setzero_si256(); 8];
    for (x, col) in cols.iter_mut().enumerate() {
        *col = _mm256_mulhrs_epi16(yukle_i16x16(chunk(coeff, x * 16)), r2);
    }
    sifirla_katsayi::<128>(coeff);
    x16::pass1_8::<H1D, 1>(&mut cols);
    if V1D == 2 {
        let k = _mm256_set1_epi16(27152);
        let r4 = _mm256_set1_epi16(2048);
        for v in &mut cols {
            let t = _mm256_mulhrs_epi16(*v, k);
            let s = _mm256_adds_epi16(_mm256_adds_epi16(*v, *v), t);
            *v = _mm256_mulhrs_epi16(s, r4);
        }
        transpose_8x8_in_ymm(&mut cols);
        let mut rows = [_mm_setzero_si128(); 16];
        for k in 0..8 {
            rows[k] = _mm256_castsi256_si128(cols[k]);
            rows[k + 8] = _mm256_extracti128_si256::<1>(cols[k]);
        }
        ekle_yaz_w8(dst, stride, &rows);
        return;
    }
    transpose_8x8_in_ymm(&mut cols);

    let mut rows = [_mm256_setzero_si256(); 16];
    for k in 0..8 {
        rows[k] = _mm256_permute4x64_epi64::<0x50>(cols[k]);
        rows[k + 8] = _mm256_permute4x64_epi64::<0xFA>(cols[k]);
    }
    x8::pass2_16_eob::<V1D>(&mut rows, false);
    ekle_yaz_w8_split(dst, stride, &rows);
}

#[target_feature(enable = "avx2")]
fn cekirdek_16x8<const H1D: u8, const V1D: u8>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32; 1024],
    eob: usize,
) {
    if H1D == 2 {
        let r2 = _mm256_set1_epi16(23168);
        let k16 = _mm256_set1_epi16(27152);
        let r1 = _mm256_set1_epi16(16384);
        let r3 = _mm256_set1_epi16(4096);
        let mut rows = [_mm256_setzero_si256(); 8];
        for (k, rk) in rows.iter_mut().enumerate() {
            let v0 = _mm256_mulhrs_epi16(
                yukle_2col_8(chunk(coeff, k * 8), chunk(coeff, (k + 8) * 8)),
                r2,
            );
            let t = _mm256_mulhrs_epi16(v0, k16);
            let v1 = _mm256_adds_epi16(v0, _mm256_mulhrs_epi16(t, r1));
            *rk = if V1D == 2 {
                _mm256_mulhrs_epi16(v1, r3)
            } else {
                v1
            };
        }
        sifirla_katsayi::<128>(coeff);
        transpose_8x8_in_ymm(&mut rows);
        if V1D != 2 {
            x16::pass2_8::<V1D>(&mut rows);
        }
        ekle_yaz_w16(dst, stride, &rows);
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

    let mut rows = [_mm256_setzero_si256(); 8];
    for (k, rk) in rows.iter_mut().enumerate() {
        let lo = _mm256_permute4x64_epi64::<0xD8>(cols[k]);
        let hi = _mm256_permute4x64_epi64::<0xD8>(cols[k + 8]);
        *rk = _mm256_permute2x128_si256::<0x20>(lo, hi);
    }
    transpose_8x8_in_ymm(&mut rows);
    x16::pass2_8::<V1D>(&mut rows);
    ekle_yaz_w16(dst, stride, &rows);
}

#[target_feature(enable = "avx2")]
fn cekirdek_16x16<const H1D: u8, const V1D: u8>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32; 1024],
    eob: usize,
) {
    if H1D == 2 && V1D == 2 {
        let k = _mm256_set1_epi16(27152);
        let r4 = _mm256_set1_epi16(2048);
        for s in 0..2 {
            let y_off = s * 8;
            let mut m = [_mm256_setzero_si256(); 8];
            for (k_idx, mk) in m.iter_mut().enumerate() {
                let v0 = yukle_2col_8(
                    chunk(coeff, k_idx * 16 + y_off),
                    chunk(coeff, (k_idx + 8) * 16 + y_off),
                );
                let t1 = _mm256_mulhrs_epi16(v0, k);
                let v1 = _mm256_avg_epu16(v0, _mm256_srai_epi16::<1>(t1));
                let t2 = _mm256_mulhrs_epi16(v1, k);
                let s2 = _mm256_adds_epi16(_mm256_adds_epi16(v1, v1), t2);
                *mk = _mm256_mulhrs_epi16(s2, r4);
            }
            transpose_8x8_in_ymm(&mut m);
            ekle_yaz_w16(&mut dst[y_off * stride..], stride, &m);
        }
        sifirla_katsayi::<256>(coeff);
        return;
    }

    let half = H1D == 0 && V1D == 0 && eob <= 35;
    if half {
        let mut cols = [_mm256_setzero_si256(); 16];
        for (x, col) in cols.iter_mut().enumerate().take(8) {
            *col = yukle_i16x8(chunk(coeff, x * 16));
        }
        sifirla_katsayi::<256>(coeff);
        x8::pass1_16_eob::<0, 2>(&mut cols, true);
        let mut rows = [_mm256_setzero_si256(); 16];
        for (k, rk) in rows.iter_mut().enumerate().take(8) {
            let lo = _mm256_permute4x64_epi64::<0xD8>(cols[k]);
            let hi = _mm256_permute4x64_epi64::<0xD8>(cols[k + 8]);
            *rk = _mm256_permute2x128_si256::<0x20>(lo, hi);
        }
        let mut top8 = [
            rows[0], rows[1], rows[2], rows[3], rows[4], rows[5], rows[6], rows[7],
        ];
        transpose_8x8_in_ymm(&mut top8);
        rows[..8].copy_from_slice(&top8);
        x16::pass2_16_eob::<0>(&mut rows, true);
        ekle_yaz_w16(dst, stride, &rows);
        return;
    }

    let mut m = [_mm256_setzero_si256(); 16];
    for (x, mx) in m.iter_mut().enumerate() {
        *mx = yukle_i16x16(chunk(coeff, x * 16));
    }
    sifirla_katsayi::<256>(coeff);
    x16::pass1_16_eob::<H1D, 2>(&mut m, false);
    transpose_16x16_ymm(&mut m);
    x16::pass2_16_eob::<V1D>(&mut m, false);
    ekle_yaz_w16(dst, stride, &m);
}

#[inline(always)]
const fn tx1d_kod(t: Tx1dType) -> u8 {
    t as u8
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn itx8_avx2_cekirdek<const TX: u8, const TXTP: u8>(
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

/// 8 bpc AVX2 `w, h ≤ 16` genel `ItxFn` giriş sarmalayıcısı (monomorfize `(TX, TXTP)`).
fn itx8_avx2_giris<const TX: u8, const TXTP: u8>(
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir (`itx8_avx2_fn`);
    // belirteç AVX2 tespiti + maske izninin kanıtıdır. (2) sınır: `itx8_sozlesmesi`
    // `dst`'nin `(h − 1)·dst_stride + w` baytı ve `coeff`'un `1024` öğeyi kapsadığını
    // garanti eder; çekirdek içindeki bütün erişimler sabit boyutlu `chunk`/`chunk_mut`
    // dizileri üzerindendir. (3) hizalama/takma ad: `dst` ve `coeff1024` ayrık `&mut`
    // referanslardır, yalnız hizasız erişim kullanılır.
    unsafe {
        itx8_avx2_cekirdek::<TX, TXTP>(dst, dst_stride, coeff1024, eob);
    }
}

/// 140 adet `w, h ≤ 16` WHT-dışı 8 bpc AVX2 `ItxFn` tablosu (yalnız [`Avx2Token`] ile).
#[must_use]
pub(crate) fn itx8_avx2_fn(_token: Avx2Token, tx: TxSize, txtp: TxType) -> Option<ItxFn> {
    if matches!(txtp, TxType::WhtWht) || !is_supported(tx, txtp) {
        return None;
    }
    macro_rules! turler16 {
        ($tx_c:literal) => {
            match txtp {
                TxType::DctDct => Some(itx8_avx2_giris::<$tx_c, 0>),
                TxType::AdstDct => Some(itx8_avx2_giris::<$tx_c, 1>),
                TxType::DctAdst => Some(itx8_avx2_giris::<$tx_c, 2>),
                TxType::AdstAdst => Some(itx8_avx2_giris::<$tx_c, 3>),
                TxType::FlipadstDct => Some(itx8_avx2_giris::<$tx_c, 4>),
                TxType::DctFlipadst => Some(itx8_avx2_giris::<$tx_c, 5>),
                TxType::FlipadstFlipadst => Some(itx8_avx2_giris::<$tx_c, 6>),
                TxType::AdstFlipadst => Some(itx8_avx2_giris::<$tx_c, 7>),
                TxType::FlipadstAdst => Some(itx8_avx2_giris::<$tx_c, 8>),
                TxType::Idtx => Some(itx8_avx2_giris::<$tx_c, 9>),
                TxType::VDct => Some(itx8_avx2_giris::<$tx_c, 10>),
                TxType::HDct => Some(itx8_avx2_giris::<$tx_c, 11>),
                TxType::VAdst => Some(itx8_avx2_giris::<$tx_c, 12>),
                TxType::HAdst => Some(itx8_avx2_giris::<$tx_c, 13>),
                TxType::VFlipadst => Some(itx8_avx2_giris::<$tx_c, 14>),
                TxType::HFlipadst => Some(itx8_avx2_giris::<$tx_c, 15>),
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
