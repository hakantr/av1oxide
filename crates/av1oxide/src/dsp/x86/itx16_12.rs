//! 12 bpc WHT-dışı ters dönüşüm (`w, h ≤ 16`, 140 kayıt) x86_64 AVX2 çekirdekleri — P44 Adım 1.
//!
//! Bu modül dav1d'nin `src/x86/itx.h:249-258` ve `src/x86/itx16_avx2.asm` dosyalarındaki
//! `w, h ≤ 16` boyutlu **140** adet **yalnız 12 bpc** (`BitDepth::Twelve`) AVX2 `itxfm_add`
//! kaydını **saf Rust intrinsics** ile gerçekler; dış `.asm`/`.S`, `asm!` ve `global_asm!`
//! yoktur (CEK-02):
//!
//! - `TX_4X4`, `RTX_4X8`, `RTX_8X4`, `TX_8X8`, `RTX_4X16`, `RTX_16X4`, `RTX_8X16`,
//!   `RTX_16X8`: 16 WHT-dışı `TxType` türünün tamamı (`8 × 16 = 128`),
//! - `TX_16X16`: desteklenen 12 `TxType` (`DCT_DCT`, `ADST_DCT`, `DCT_ADST`,
//!   `ADST_ADST`, `FLIPADST_DCT`, `DCT_FLIPADST`, `FLIPADST_FLIPADST`,
//!   `ADST_FLIPADST`, `FLIPADST_ADST`, `IDTX`, `V_DCT`, `H_DCT`).
//!
//! ## Sözleşme, iki geçişli 32-bit 12 bpc aritmetik ve ISA saflığı (P34 / P44)
//!
//! - Girişte O(1) sözleşme denetimi ([`itx16_sozlesmesi`]) ile `(tx, txtp, bit_depth == Twelve)`
//!   uyumu aranır; sözleşme dışı çağrıda (`10 bpc`, `8 bpc`, tek bayt `dst_stride` vb.)
//!   paniklemeden [`itx_skaler_devret`] üzerinden sayaç artırılır ve ilgili bit derinliğinin
//!   skaler monomorfuna devredilir.
//! - 12 bpc'de `row_clip = [-524288, +524287]` (`clip_20b`) ve `col_clip = [-131072, +131071]`
//!   (`clip_18b`) sınırlarının ikisi de `i16` aralığını aştığı için hem Pass 1 hem Pass 2
//!   `i32` şeritlerinde (`4 × i32` `__m128i` ve `8 × i32` `__m256i`) çalışır. `clip_20b`
//!   altında `a · c0 + b · c1 + 2048` ara toplamlarının `i32` taşmasını önlemek için çift/tek
//!   şeritler `_mm256_mul_epi32` / `_mm_mul_epi32` (`vpmuldq`, `32×32 → 64-bit`) ve
//!   `_mm256_add_epi64` / `_mm_add_epi64` ile 64-bit `i64` duyarlığında hesaplanır.
//! - Bütün çekirdek ve yardımcılar yalnız `#[target_feature(enable = "avx2")]` taşır;
//!   başka `dsp/x86/*` modülleriyle `#[target_feature]` yardımcı paylaşmaz (P34) ve
//!   EVEX (`%zmm`, `{%k`) yönergesi üretmez.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_add_epi32, _mm_add_epi64, _mm_and_si128, _mm_blend_epi32,
    _mm_blendv_epi8, _mm_cmpgt_epi64, _mm_cvtsi64_si128, _mm_max_epi16, _mm_max_epi32,
    _mm_min_epi16, _mm_min_epi32, _mm_mul_epi32, _mm_mullo_epi32, _mm_packs_epi32, _mm_set1_epi16,
    _mm_set1_epi32, _mm_set1_epi64x, _mm_setr_epi32, _mm_setzero_si128, _mm_slli_epi64,
    _mm_srai_epi32, _mm_srli_epi64, _mm_srli_si128, _mm_sub_epi32, _mm_sub_epi64,
    _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm_xor_si128,
    _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi64, _mm256_and_si256, _mm256_blend_epi32,
    _mm256_blendv_epi8, _mm256_castsi256_si128, _mm256_cmpgt_epi64, _mm256_extracti128_si256,
    _mm256_max_epi16, _mm256_max_epi32, _mm256_min_epi16, _mm256_min_epi32, _mm256_mul_epi32,
    _mm256_mullo_epi32, _mm256_permute2x128_si256, _mm256_set_m128i, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_set1_epi64x, _mm256_setr_epi32, _mm256_setzero_si256,
    _mm256_slli_epi64, _mm256_srai_epi32, _mm256_srli_epi64, _mm256_sub_epi32, _mm256_sub_epi64,
    _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
    _mm256_xor_si256,
};

use super::mem::{
    chunk, chunk_mut, load_u8x16, load_u8x32, store_i32x8, store_u8x8, store_u8x16, store_u8x32,
};
use crate::cpu::Avx2Token;
use crate::dsp::itx::{ItxFn, TxSize, TxType, itx_skaler_devret, itx16_sozlesmesi};
use crate::dsp::scalar::itx_1d::Tx1dType;
use crate::frame::BitDepth;

// ---------------------------------------------------------------------------
// Katsayı yükleme ve sıfırlama yardımcıları (`i32`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
fn yukle_i32x4(coeff: &[i32], off: usize) -> __m128i {
    let c = chunk::<i32, 4>(coeff, off);
    _mm_setr_epi32(c[0], c[1], c[2], c[3])
}

#[target_feature(enable = "avx2")]
#[inline]
fn yukle_i32x8(coeff: &[i32], off: usize) -> __m256i {
    let c = chunk::<i32, 8>(coeff, off);
    _mm256_setr_epi32(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7])
}

#[target_feature(enable = "avx2")]
#[inline]
fn sifirla_katsayi<const N: usize>(coeff: &mut [i32]) {
    const { assert!(N.is_multiple_of(8) && N <= 1024) };
    let z = _mm256_setzero_si256();
    let mut i = 0;
    while i < N {
        store_i32x8(chunk_mut(coeff, i), z);
        i += 8;
    }
}

// ---------------------------------------------------------------------------
// Hedef `u16` düzlem piksel ekleme + `[0, 4095]` doyurmalı yazma yardımcıları
// ---------------------------------------------------------------------------

/// `4 × 2` bölgeye (satır başına `8` bayt) iki satırlık `4 × i32` Pass 2 çıktısını
/// `(c + 8) >> 4`, `[-4096, +4095]` ve `[0, 4095]` kırpmalarıyla ekler ve yazar.
#[target_feature(enable = "avx2")]
#[inline]
fn ekle_yaz_4x2(dst: &mut [u8], stride: usize, y_off: usize, r0: __m128i, r1: __m128i) {
    let sekiz = _mm_set1_epi32(8);
    let s0 = _mm_srai_epi32::<4>(_mm_add_epi32(r0, sekiz));
    let s1 = _mm_srai_epi32::<4>(_mm_add_epi32(r1, sekiz));
    let res16 = _mm_min_epi16(
        _mm_max_epi16(_mm_packs_epi32(s0, s1), _mm_set1_epi16(-4096)),
        _mm_set1_epi16(4095),
    );
    let b0 = y_off * stride;
    let b1 = b0 + stride;
    let d0 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, b0)));
    let d1 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, b1)));
    let dst16 = _mm_unpacklo_epi64(d0, d1);
    let out = _mm_min_epi16(
        _mm_max_epi16(_mm_add_epi16(dst16, res16), _mm_setzero_si128()),
        _mm_set1_epi16(4095),
    );
    store_u8x8(chunk_mut(dst, b0), out);
    store_u8x8(chunk_mut(dst, b1), _mm_srli_si128::<8>(out));
}

/// Tek bir `8` piksellik (`16` bayt) satıra `8 × i32` Pass 2 çıktısını ekler ve yazar.
#[target_feature(enable = "avx2")]
#[inline]
fn ekle_yaz_8x1(dst: &mut [u8], off: usize, r: __m256i) {
    let s = _mm256_srai_epi32::<4>(_mm256_add_epi32(r, _mm256_set1_epi32(8)));
    let s_lo = _mm256_castsi256_si128(s);
    let s_hi = _mm256_extracti128_si256::<1>(s);
    let res16 = _mm_min_epi16(
        _mm_max_epi16(_mm_packs_epi32(s_lo, s_hi), _mm_set1_epi16(-4096)),
        _mm_set1_epi16(4095),
    );
    let dst16 = load_u8x16(chunk::<u8, 16>(dst, off));
    let out = _mm_min_epi16(
        _mm_max_epi16(_mm_add_epi16(dst16, res16), _mm_setzero_si128()),
        _mm_set1_epi16(4095),
    );
    store_u8x16(chunk_mut(dst, off), out);
}

/// `DCT_DCT` + `eob == 0` DC-only hızlı yolu (`coeff[0] = 0`; `coeff[1..]` korunur).
#[target_feature(enable = "avx2")]
#[inline]
fn dc_yalniz<const W: usize, const H: usize, const SHIFT: u32, const RECT2: bool>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32],
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
    let dc16 = dc.clamp(-4096, 4095) as i16;

    match W {
        4 => {
            let dc_v = _mm_set1_epi16(dc16);
            let sifir = _mm_setzero_si128();
            let tavan = _mm_set1_epi16(4095);
            let mut y = 0;
            while y < H {
                let b0 = y * stride;
                let b1 = b0 + stride;
                let d0 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, b0)));
                let d1 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, b1)));
                let dst16 = _mm_unpacklo_epi64(d0, d1);
                let out = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(dst16, dc_v), sifir), tavan);
                store_u8x8(chunk_mut(dst, b0), out);
                store_u8x8(chunk_mut(dst, b1), _mm_srli_si128::<8>(out));
                y += 2;
            }
        }
        8 => {
            let dc_v = _mm_set1_epi16(dc16);
            let sifir = _mm_setzero_si128();
            let tavan = _mm_set1_epi16(4095);
            let mut y = 0;
            while y < H {
                let off = y * stride;
                let d = load_u8x16(chunk::<u8, 16>(dst, off));
                let out = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(d, dc_v), sifir), tavan);
                store_u8x16(chunk_mut(dst, off), out);
                y += 1;
            }
        }
        16 => {
            let dc_v = _mm256_set1_epi16(dc16);
            let sifir = _mm256_setzero_si256();
            let tavan = _mm256_set1_epi16(4095);
            let mut y = 0;
            while y < H {
                let off = y * stride;
                let d = load_u8x32(chunk::<u8, 32>(dst, off));
                let out =
                    _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(d, dc_v), sifir), tavan);
                store_u8x32(chunk_mut(dst, off), out);
                y += 1;
            }
        }
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// 32-bit (`i32`) matris devrikleme (transpose) yardımcıları
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
fn devrik_4x4_i32(r: [__m128i; 4]) -> [__m128i; 4] {
    let t0 = _mm_unpacklo_epi32(r[0], r[1]);
    let t1 = _mm_unpackhi_epi32(r[0], r[1]);
    let t2 = _mm_unpacklo_epi32(r[2], r[3]);
    let t3 = _mm_unpackhi_epi32(r[2], r[3]);
    [
        _mm_unpacklo_epi64(t0, t2),
        _mm_unpackhi_epi64(t0, t2),
        _mm_unpacklo_epi64(t1, t3),
        _mm_unpackhi_epi64(t1, t3),
    ]
}

#[target_feature(enable = "avx2")]
#[inline]
fn devrik_4x8_i32(c: [__m256i; 4]) -> [__m128i; 8] {
    let lo = devrik_4x4_i32([
        _mm256_castsi256_si128(c[0]),
        _mm256_castsi256_si128(c[1]),
        _mm256_castsi256_si128(c[2]),
        _mm256_castsi256_si128(c[3]),
    ]);
    let hi = devrik_4x4_i32([
        _mm256_extracti128_si256::<1>(c[0]),
        _mm256_extracti128_si256::<1>(c[1]),
        _mm256_extracti128_si256::<1>(c[2]),
        _mm256_extracti128_si256::<1>(c[3]),
    ]);
    [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]]
}

#[target_feature(enable = "avx2")]
#[inline]
fn devrik_8x4_i32(c: [__m128i; 8]) -> [__m256i; 4] {
    let lo = devrik_4x4_i32([c[0], c[1], c[2], c[3]]);
    let hi = devrik_4x4_i32([c[4], c[5], c[6], c[7]]);
    [
        _mm256_set_m128i(hi[0], lo[0]),
        _mm256_set_m128i(hi[1], lo[1]),
        _mm256_set_m128i(hi[2], lo[2]),
        _mm256_set_m128i(hi[3], lo[3]),
    ]
}

#[target_feature(enable = "avx2")]
#[inline]
fn devrik_8x8_i32(r: [__m256i; 8]) -> [__m256i; 8] {
    let t0 = _mm256_unpacklo_epi32(r[0], r[1]);
    let t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    let t2 = _mm256_unpacklo_epi32(r[2], r[3]);
    let t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    let t4 = _mm256_unpacklo_epi32(r[4], r[5]);
    let t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    let t6 = _mm256_unpacklo_epi32(r[6], r[7]);
    let t7 = _mm256_unpackhi_epi32(r[6], r[7]);

    let u0 = _mm256_unpacklo_epi64(t0, t2);
    let u1 = _mm256_unpackhi_epi64(t0, t2);
    let u2 = _mm256_unpacklo_epi64(t1, t3);
    let u3 = _mm256_unpackhi_epi64(t1, t3);
    let u4 = _mm256_unpacklo_epi64(t4, t6);
    let u5 = _mm256_unpackhi_epi64(t4, t6);
    let u6 = _mm256_unpacklo_epi64(t5, t7);
    let u7 = _mm256_unpackhi_epi64(t5, t7);

    [
        _mm256_permute2x128_si256::<0x20>(u0, u4),
        _mm256_permute2x128_si256::<0x20>(u1, u5),
        _mm256_permute2x128_si256::<0x20>(u2, u6),
        _mm256_permute2x128_si256::<0x20>(u3, u7),
        _mm256_permute2x128_si256::<0x31>(u0, u4),
        _mm256_permute2x128_si256::<0x31>(u1, u5),
        _mm256_permute2x128_si256::<0x31>(u2, u6),
        _mm256_permute2x128_si256::<0x31>(u3, u7),
    ]
}

// ---------------------------------------------------------------------------
// Şerit-paralel 32-bit (`i32`, 64-bit ara çarpımlı) 1B dönüşüm çekirdekleri
// ---------------------------------------------------------------------------

macro_rules! itx12_1d_impl {
    (
        $mod_ad:ident,
        $V:ty,
        $BLEND_MASK:literal,
        $zero:ident,
        $set1_32:ident,
        $set1_64:ident,
        $add32:ident,
        $sub32:ident,
        $max32:ident,
        $min32:ident,
        $srai32:ident,
        $mullo32:ident,
        $and:ident,
        $xor:ident,
        $mul32:ident,
        $add64:ident,
        $sub64:ident,
        $srli64:ident,
        $slli64:ident,
        $cmpgt64:ident,
        $blendv8:ident,
        $blend32:ident
    ) => {
        mod $mod_ad {
            use super::*;

            #[derive(Clone, Copy)]
            pub(super) struct I64 {
                pub(super) ev: $V,
                pub(super) od: $V,
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            fn srai64<const SHIFT: i32>(x: $V) -> $V {
                $blend32::<$BLEND_MASK>($srli64::<SHIFT>(x), $srai32::<SHIFT>(x))
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            fn clamp64(v: $V, min_v: $V, max_v: $V) -> $V {
                let v = $blendv8(v, min_v, $cmpgt64(min_v, v));
                $blendv8(v, max_v, $cmpgt64(v, max_v))
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn kirp<const MIN: i32, const MAX: i32>(v: $V) -> $V {
                $min32($max32(v, $set1_32(MIN)), $set1_32(MAX))
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn kirp_64<const MIN: i32, const MAX: i32>(x: I64) -> $V {
                if MIN >= -131_072 {
                    return kirp::<MIN, MAX>(x.ev);
                }
                let min_v = $set1_64(i64::from(MIN));
                let max_v = $set1_64(i64::from(MAX));
                let ev = clamp64(x.ev, min_v, max_v);
                let od = clamp64(x.od, min_v, max_v);
                $blend32::<$BLEND_MASK>(ev, $slli64::<32>(od))
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn add_sub_kirp_64<const MIN: i32, const MAX: i32>(
                u: I64,
                v: I64,
            ) -> ($V, $V) {
                if MIN >= -131_072 {
                    return (
                        kirp::<MIN, MAX>($add32(u.ev, v.ev)),
                        kirp::<MIN, MAX>($sub32(u.ev, v.ev)),
                    );
                }
                let sum = I64 {
                    ev: $add64(u.ev, v.ev),
                    od: $add64(u.od, v.od),
                };
                let diff = I64 {
                    ev: $sub64(u.ev, v.ev),
                    od: $sub64(u.od, v.od),
                };
                (kirp_64::<MIN, MAX>(sum), kirp_64::<MIN, MAX>(diff))
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn sub_add_kirp_64<const MIN: i32, const MAX: i32>(
                u: I64,
                v: I64,
            ) -> ($V, $V) {
                if MIN >= -131_072 {
                    return (
                        kirp::<MIN, MAX>($sub32(u.ev, v.ev)),
                        kirp::<MIN, MAX>($add32(u.ev, v.ev)),
                    );
                }
                let diff = I64 {
                    ev: $sub64(u.ev, v.ev),
                    od: $sub64(u.od, v.od),
                };
                let sum = I64 {
                    ev: $add64(u.ev, v.ev),
                    od: $add64(u.od, v.od),
                };
                (kirp_64::<MIN, MAX>(diff), kirp_64::<MIN, MAX>(sum))
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn mul1_64<const SHIFT: i32, const MIN: i32>(
                a: $V,
                c0: i32,
                rnd: i64,
            ) -> I64 {
                if MIN >= -131_072 {
                    let ev =
                        $srai32::<SHIFT>($add32($mullo32(a, $set1_32(c0)), $set1_32(rnd as i32)));
                    return I64 { ev, od: $zero() };
                }
                let a_od = $srli64::<32>(a);
                let kc0 = $set1_32(c0);
                let rnd_v = $set1_64(rnd);
                let ev = srai64::<SHIFT>($add64($mul32(a, kc0), rnd_v));
                let od = srai64::<SHIFT>($add64($mul32(a_od, kc0), rnd_v));
                I64 { ev, od }
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn mul2_64<const SHIFT: i32, const MIN: i32>(
                a: $V,
                b: $V,
                c0: i32,
                c1: i32,
                d0: i32,
                d1: i32,
                rnd: i64,
            ) -> (I64, I64) {
                if MIN >= -131_072 {
                    let kc0 = $set1_32(c0);
                    let kc1 = $set1_32(c1);
                    let kd0 = $set1_32(d0);
                    let kd1 = $set1_32(d1);
                    let rnd_v = $set1_32(rnd as i32);
                    let u =
                        $srai32::<SHIFT>($add32($add32($mullo32(a, kc0), $mullo32(b, kc1)), rnd_v));
                    let v =
                        $srai32::<SHIFT>($add32($add32($mullo32(a, kd0), $mullo32(b, kd1)), rnd_v));
                    return (I64 { ev: u, od: $zero() }, I64 { ev: v, od: $zero() });
                }
                let a_od = $srli64::<32>(a);
                let b_od = $srli64::<32>(b);
                let kc0 = $set1_32(c0);
                let kc1 = $set1_32(c1);
                let kd0 = $set1_32(d0);
                let kd1 = $set1_32(d1);
                let rnd_v = $set1_64(rnd);

                let u_ev = srai64::<SHIFT>($add64($add64($mul32(a, kc0), $mul32(b, kc1)), rnd_v));
                let u_od =
                    srai64::<SHIFT>($add64($add64($mul32(a_od, kc0), $mul32(b_od, kc1)), rnd_v));
                let v_ev = srai64::<SHIFT>($add64($add64($mul32(a, kd0), $mul32(b, kd1)), rnd_v));
                let v_od =
                    srai64::<SHIFT>($add64($add64($mul32(a_od, kd0), $mul32(b_od, kd1)), rnd_v));
                (I64 { ev: u_ev, od: u_od }, I64 { ev: v_ev, od: v_od })
            }

            /// `(v · 181 + 128) >> 8` (64-bit çarpımlı; ham `coeff` `i32::MIN..=i32::MAX` aralığında taşmasız).
            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn olcekle_181(v: $V) -> $V {
                let v_od = $srli64::<32>(v);
                let k181 = $set1_32(181);
                let rnd = $set1_64(128);
                let ev = $srli64::<8>($add64($mul32(v, k181), rnd));
                let od = $slli64::<24>($add64($mul32(v_od, k181), rnd));
                $blend32::<$BLEND_MASK>(ev, od)
            }

            /// Kırpılmış `[-524288, +524287]` kelebek toplam/farkları (`|v| <= 1_048_576`) için
            /// 32-bit taşmasız `(v · 181 + 128) >> 8`.
            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn olcekle_181_ic(v: $V) -> $V {
                $srai32::<8>($add32($mullo32(v, $set1_32(181)), $set1_32(128)))
            }

            /// `[-524288, +524287]` (`MIN == -524288`) veya `[-131072, +131071]` (`MIN == -131072`)
            /// aralığına kırpılmış girdiler için 32-bit `vpmulld` tabanlı ikili döndürme (`>> 12`).
            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn mul2<const MIN: i32>(
                a: $V,
                b: $V,
                c0: i32,
                c1: i32,
                d0: i32,
                d1: i32,
            ) -> ($V, $V) {
                let kc0 = $set1_32(c0);
                let kc1 = $set1_32(c1);
                let kd0 = $set1_32(d0);
                let kd1 = $set1_32(d1);
                let rnd = $set1_32(2048);

                let u0 = $add32($mullo32(a, kc0), rnd);
                let u1 = $mullo32(b, kc1);
                let v0 = $add32($mullo32(a, kd0), rnd);
                let v1 = $mullo32(b, kd1);

                if MIN >= -131_072 {
                    ($srai32::<12>($add32(u0, u1)), $srai32::<12>($add32(v0, v1)))
                } else {
                    let u_avg = $add32($and(u0, u1), $srai32::<1>($xor(u0, u1)));
                    let v_avg = $add32($and(v0, v1), $srai32::<1>($xor(v0, v1)));
                    ($srai32::<11>(u_avg), $srai32::<11>(v_avg))
                }
            }

            /// Pass 1 çıkışı üzerinde `(v + rnd) >> SHIFT` ve `col_clip = [-131072, +131071]`.
            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn kaydir_kirp_18b<const SHIFT: u32>(v: $V) -> $V {
                let s = match SHIFT {
                    0 => v,
                    1 => $srai32::<1>($add32(v, $set1_32(1))),
                    2 => $srai32::<2>($add32(v, $set1_32(2))),
                    _ => unreachable!(),
                };
                kirp::<-131_072, 131_071>(s)
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn kaydir_dizi_18b<const N: usize, const SHIFT: u32>(
                mut c: [$V; N],
            ) -> [$V; N] {
                let mut i = 0;
                while i < N {
                    c[i] = kaydir_kirp_18b::<SHIFT>(c[i]);
                    i += 1;
                }
                c
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn idct4<const MIN: i32, const MAX: i32>(c: [$V; 4]) -> [$V; 4] {
                let (t0, t1) = mul2_64::<8, MIN>(c[0], c[2], 181, 181, 181, -181, 128);
                let (t2, t3) = mul2_64::<12, MIN>(c[1], c[3], 1567, -3784, 3784, 1567, 2048);
                let (o0, o3) = add_sub_kirp_64::<MIN, MAX>(t0, t3);
                let (o1, o2) = add_sub_kirp_64::<MIN, MAX>(t1, t2);
                [o0, o1, o2, o3]
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn idct4_yarim<const MIN: i32, const MAX: i32>(c0: $V, c1: $V) -> [$V; 4] {
                let t0 = mul1_64::<8, MIN>(c0, 181, 128);
                let (t2, t3) = mul2_64::<12, MIN>(c1, $zero(), 1567, 0, 3784, 0, 2048);
                let (o0, o3) = add_sub_kirp_64::<MIN, MAX>(t0, t3);
                let (o1, o2) = add_sub_kirp_64::<MIN, MAX>(t0, t2);
                [o0, o1, o2, o3]
            }

            #[target_feature(enable = "avx2")]
            #[inline(never)]
            pub(super) fn iadst4<const MIN: i32>(c: [$V; 4], flip: bool) -> [$V; 4] {
                if MIN >= -131_072 {
                    let k1321 = $set1_32(1321);
                    let k2482 = $set1_32(2482);
                    let k3344 = $set1_32(3344);
                    let k3803 = $set1_32(3803);
                    let rnd12 = $set1_32(2048);
                    let m1 = $mullo32(c[1], k3344);
                    let o0 = $srai32::<12>($add32(
                        $add32($mullo32(c[0], k1321), $mullo32(c[2], k3803)),
                        $add32(m1, $add32($mullo32(c[3], k2482), rnd12)),
                    ));
                    let o1 = $srai32::<12>($add32(
                        $sub32($mullo32(c[0], k2482), $mullo32(c[2], k1321)),
                        $add32($sub32(m1, $mullo32(c[3], k3803)), rnd12),
                    ));
                    let o2 = $srai32::<8>($add32(
                        $mullo32($add32($sub32(c[0], c[2]), c[3]), $set1_32(209)),
                        $set1_32(128),
                    ));
                    let o3 = $srai32::<12>($add32(
                        $sub32(
                            $add32($mullo32(c[0], k3803), $mullo32(c[2], k2482)),
                            $add32(m1, $mullo32(c[3], k1321)),
                        ),
                        rnd12,
                    ));
                    return if flip {
                        [o3, o2, o1, o0]
                    } else {
                        [o0, o1, o2, o3]
                    };
                }

                let c0_od = $srli64::<32>(c[0]);
                let c1_od = $srli64::<32>(c[1]);
                let c2_od = $srli64::<32>(c[2]);
                let c3_od = $srli64::<32>(c[3]);

                let k1321 = $set1_32(1321);
                let k2482 = $set1_32(2482);
                let k3344 = $set1_32(3344);
                let k3803 = $set1_32(3803);
                let k209 = $set1_32(209);
                let rnd12 = $set1_64(2048);
                let rnd8 = $set1_64(128);

                let m1_ev = $mul32(c[1], k3344);
                let m1_od = $mul32(c1_od, k3344);

                let s0_ev = $add64(
                    $add64($mul32(c[0], k1321), $mul32(c[2], k3803)),
                    $add64(m1_ev, $add64($mul32(c[3], k2482), rnd12)),
                );
                let s0_od = $add64(
                    $add64($mul32(c0_od, k1321), $mul32(c2_od, k3803)),
                    $add64(m1_od, $add64($mul32(c3_od, k2482), rnd12)),
                );
                let o0 = kirp_64::<-16_777_216, 16_777_215>(I64 {
                    ev: srai64::<12>(s0_ev),
                    od: srai64::<12>(s0_od),
                });

                let s1_ev = $add64(
                    $sub64($mul32(c[0], k2482), $mul32(c[2], k1321)),
                    $add64($sub64(m1_ev, $mul32(c[3], k3803)), rnd12),
                );
                let s1_od = $add64(
                    $sub64($mul32(c0_od, k2482), $mul32(c2_od, k1321)),
                    $add64($sub64(m1_od, $mul32(c3_od, k3803)), rnd12),
                );
                let o1 = kirp_64::<-16_777_216, 16_777_215>(I64 {
                    ev: srai64::<12>(s1_ev),
                    od: srai64::<12>(s1_od),
                });

                let s2_ev = $add64(
                    $add64(
                        $sub64($mul32(c[0], k209), $mul32(c[2], k209)),
                        $mul32(c[3], k209),
                    ),
                    rnd8,
                );
                let s2_od = $add64(
                    $add64(
                        $sub64($mul32(c0_od, k209), $mul32(c2_od, k209)),
                        $mul32(c3_od, k209),
                    ),
                    rnd8,
                );
                let o2 = kirp_64::<-16_777_216, 16_777_215>(I64 {
                    ev: srai64::<8>(s2_ev),
                    od: srai64::<8>(s2_od),
                });

                let s3_ev = $add64(
                    $sub64(
                        $add64($mul32(c[0], k3803), $mul32(c[2], k2482)),
                        $add64(m1_ev, $mul32(c[3], k1321)),
                    ),
                    rnd12,
                );
                let s3_od = $add64(
                    $sub64(
                        $add64($mul32(c0_od, k3803), $mul32(c2_od, k2482)),
                        $add64(m1_od, $mul32(c3_od, k1321)),
                    ),
                    rnd12,
                );
                let o3 = kirp_64::<-16_777_216, 16_777_215>(I64 {
                    ev: srai64::<12>(s3_ev),
                    od: srai64::<12>(s3_od),
                });

                if flip {
                    [o3, o2, o1, o0]
                } else {
                    [o0, o1, o2, o3]
                }
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn gecis_4<const MIN: i32, const MAX: i32>(
                c: [$V; 4],
                kind: Tx1dType,
            ) -> [$V; 4] {
                match kind {
                    Tx1dType::Identity => {
                        if MIN >= -131_072 {
                            let k5793 = $set1_32(5793);
                            let rnd = $set1_32(2048);
                            [
                                $srai32::<12>($add32($mullo32(c[0], k5793), rnd)),
                                $srai32::<12>($add32($mullo32(c[1], k5793), rnd)),
                                $srai32::<12>($add32($mullo32(c[2], k5793), rnd)),
                                $srai32::<12>($add32($mullo32(c[3], k5793), rnd)),
                            ]
                        } else {
                            [
                                kirp_64::<-16_777_216, 16_777_215>(mul1_64::<12, MIN>(
                                    c[0], 5793, 2048,
                                )),
                                kirp_64::<-16_777_216, 16_777_215>(mul1_64::<12, MIN>(
                                    c[1], 5793, 2048,
                                )),
                                kirp_64::<-16_777_216, 16_777_215>(mul1_64::<12, MIN>(
                                    c[2], 5793, 2048,
                                )),
                                kirp_64::<-16_777_216, 16_777_215>(mul1_64::<12, MIN>(
                                    c[3], 5793, 2048,
                                )),
                            ]
                        }
                    }
                    Tx1dType::Dct => idct4::<MIN, MAX>(c),
                    Tx1dType::Adst | Tx1dType::FlipAdst => {
                        iadst4::<MIN>(c, matches!(kind, Tx1dType::FlipAdst))
                    }
                }
            }

            #[target_feature(enable = "avx2")]
            #[inline(never)]
            pub(super) fn idct8<const MIN: i32, const MAX: i32>(c: [$V; 8]) -> [$V; 8] {
                let [t0, t1, t2, t3] = idct4::<MIN, MAX>([c[0], c[2], c[4], c[6]]);
                let (t4a, t7a) = mul2_64::<12, MIN>(c[1], c[7], 799, -4017, 4017, 799, 2048);
                let (t5a, t6a) = mul2_64::<11, MIN>(c[5], c[3], 1703, -1138, 1138, 1703, 1024);
                let (t4, t5a) = add_sub_kirp_64::<MIN, MAX>(t4a, t5a);
                let (t7, t6a) = add_sub_kirp_64::<MIN, MAX>(t7a, t6a);
                let t5 = olcekle_181_ic($sub32(t6a, t5a));
                let t6 = olcekle_181_ic($add32(t6a, t5a));
                [
                    kirp::<MIN, MAX>($add32(t0, t7)),
                    kirp::<MIN, MAX>($add32(t1, t6)),
                    kirp::<MIN, MAX>($add32(t2, t5)),
                    kirp::<MIN, MAX>($add32(t3, t4)),
                    kirp::<MIN, MAX>($sub32(t3, t4)),
                    kirp::<MIN, MAX>($sub32(t2, t5)),
                    kirp::<MIN, MAX>($sub32(t1, t6)),
                    kirp::<MIN, MAX>($sub32(t0, t7)),
                ]
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn idct8_yarim<const MIN: i32, const MAX: i32>(c: [$V; 4]) -> [$V; 8] {
                let [t0, t1, t2, t3] = idct4_yarim::<MIN, MAX>(c[0], c[2]);
                let (t4a, t7a) = mul2_64::<12, MIN>(c[1], $zero(), 799, 0, 4017, 0, 2048);
                let (t5a, t6a) = mul2_64::<12, MIN>(c[3], $zero(), -2276, 0, 3406, 0, 2048);
                let (t4, t5a) = add_sub_kirp_64::<MIN, MAX>(t4a, t5a);
                let (t7, t6a) = add_sub_kirp_64::<MIN, MAX>(t7a, t6a);
                let t5 = olcekle_181_ic($sub32(t6a, t5a));
                let t6 = olcekle_181_ic($add32(t6a, t5a));
                [
                    kirp::<MIN, MAX>($add32(t0, t7)),
                    kirp::<MIN, MAX>($add32(t1, t6)),
                    kirp::<MIN, MAX>($add32(t2, t5)),
                    kirp::<MIN, MAX>($add32(t3, t4)),
                    kirp::<MIN, MAX>($sub32(t3, t4)),
                    kirp::<MIN, MAX>($sub32(t2, t5)),
                    kirp::<MIN, MAX>($sub32(t1, t6)),
                    kirp::<MIN, MAX>($sub32(t0, t7)),
                ]
            }

            #[target_feature(enable = "avx2")]
            #[inline(never)]
            pub(super) fn iadst8<const MIN: i32, const MAX: i32>(
                c: [$V; 8],
                flip: bool,
            ) -> [$V; 8] {
                let (t0a, t1a) = mul2_64::<12, MIN>(c[7], c[0], 4076, 401, 401, -4076, 2048);
                let (t2a, t3a) = mul2_64::<12, MIN>(c[5], c[2], 3612, 1931, 1931, -3612, 2048);
                let (t4a, t5a) = mul2_64::<11, MIN>(c[3], c[4], 1299, 1583, 1583, -1299, 1024);
                let (t6a, t7a) = mul2_64::<12, MIN>(c[1], c[6], 1189, 3920, 3920, -1189, 2048);

                let (t0, t4) = add_sub_kirp_64::<MIN, MAX>(t0a, t4a);
                let (t1, t5) = add_sub_kirp_64::<MIN, MAX>(t1a, t5a);
                let (t2, t6) = add_sub_kirp_64::<MIN, MAX>(t2a, t6a);
                let (t3, t7) = add_sub_kirp_64::<MIN, MAX>(t3a, t7a);

                let (t4a, t5a) = mul2::<MIN>(t4, t5, 3784, 1567, 1567, -3784);
                let (t6a, t7a) = mul2::<MIN>(t7, t6, 3784, -1567, 1567, 3784);

                let z = $zero();
                let o0 = kirp::<MIN, MAX>($add32(t0, t2));
                let o7 = $sub32(z, kirp::<MIN, MAX>($add32(t1, t3)));
                let t2b = kirp::<MIN, MAX>($sub32(t0, t2));
                let t3b = kirp::<MIN, MAX>($sub32(t1, t3));
                let o1 = $sub32(z, kirp::<MIN, MAX>($add32(t4a, t6a)));
                let o6 = kirp::<MIN, MAX>($add32(t5a, t7a));
                let t6b = kirp::<MIN, MAX>($sub32(t4a, t6a));
                let t7b = kirp::<MIN, MAX>($sub32(t5a, t7a));

                let o3 = $sub32(z, olcekle_181_ic($add32(t2b, t3b)));
                let o4 = olcekle_181_ic($sub32(t2b, t3b));
                let o2 = olcekle_181_ic($add32(t6b, t7b));
                let o5 = $sub32(z, olcekle_181_ic($sub32(t6b, t7b)));

                let mut o = [o0, o1, o2, o3, o4, o5, o6, o7];
                if flip {
                    o.reverse();
                }
                o
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn gecis_8<const MIN: i32, const MAX: i32>(
                c: [$V; 8],
                kind: Tx1dType,
            ) -> [$V; 8] {
                match kind {
                    Tx1dType::Identity => {
                        let mut o = c;
                        let mut i = 0;
                        while i < 8 {
                            let x = if MIN >= -131_072 {
                                c[i]
                            } else {
                                kirp::<-16_777_216, 16_777_215>(c[i])
                            };
                            o[i] = $add32(x, x);
                            i += 1;
                        }
                        o
                    }
                    Tx1dType::Dct => idct8::<MIN, MAX>(c),
                    Tx1dType::Adst | Tx1dType::FlipAdst => {
                        iadst8::<MIN, MAX>(c, matches!(kind, Tx1dType::FlipAdst))
                    }
                }
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            fn idct16_ortak_son<const MIN: i32, const MAX: i32>(
                c: &mut [$V; 16],
                e: [$V; 8],
                [t8, t9, t10, t11, t12, t13, t14, t15]: [$V; 8],
            ) {
                let (t9a, t14a) = mul2::<MIN>(t14, t9, 1567, -3784, 3784, 1567);
                let (t10a, t13a) = mul2::<MIN>(t13, t10, -3784, -1567, 1567, -3784);

                let t8a = kirp::<MIN, MAX>($add32(t8, t11));
                let t9 = kirp::<MIN, MAX>($add32(t9a, t10a));
                let t10 = kirp::<MIN, MAX>($sub32(t9a, t10a));
                let t11a = kirp::<MIN, MAX>($sub32(t8, t11));
                let t12a = kirp::<MIN, MAX>($sub32(t15, t12));
                let t13 = kirp::<MIN, MAX>($sub32(t14a, t13a));
                let t14 = kirp::<MIN, MAX>($add32(t14a, t13a));
                let t15a = kirp::<MIN, MAX>($add32(t15, t12));

                let t10a = olcekle_181_ic($sub32(t13, t10));
                let t13a = olcekle_181_ic($add32(t13, t10));
                let t11 = olcekle_181_ic($sub32(t12a, t11a));
                let t12 = olcekle_181_ic($add32(t12a, t11a));

                c[0] = kirp::<MIN, MAX>($add32(e[0], t15a));
                c[1] = kirp::<MIN, MAX>($add32(e[1], t14));
                c[2] = kirp::<MIN, MAX>($add32(e[2], t13a));
                c[3] = kirp::<MIN, MAX>($add32(e[3], t12));
                c[4] = kirp::<MIN, MAX>($add32(e[4], t11));
                c[5] = kirp::<MIN, MAX>($add32(e[5], t10a));
                c[6] = kirp::<MIN, MAX>($add32(e[6], t9));
                c[7] = kirp::<MIN, MAX>($add32(e[7], t8a));
                c[8] = kirp::<MIN, MAX>($sub32(e[7], t8a));
                c[9] = kirp::<MIN, MAX>($sub32(e[6], t9));
                c[10] = kirp::<MIN, MAX>($sub32(e[5], t10a));
                c[11] = kirp::<MIN, MAX>($sub32(e[4], t11));
                c[12] = kirp::<MIN, MAX>($sub32(e[3], t12));
                c[13] = kirp::<MIN, MAX>($sub32(e[2], t13a));
                c[14] = kirp::<MIN, MAX>($sub32(e[1], t14));
                c[15] = kirp::<MIN, MAX>($sub32(e[0], t15a));
            }

            #[target_feature(enable = "avx2")]
            #[inline(never)]
            pub(super) fn idct16<const MIN: i32, const MAX: i32>(c: &mut [$V; 16]) {
                let e = idct8::<MIN, MAX>([c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]]);
                let (t8a, t15a) = mul2_64::<12, MIN>(c[1], c[15], 401, -4076, 4076, 401, 2048);
                let (t9a, t14a) = mul2_64::<11, MIN>(c[9], c[7], 1583, -1299, 1299, 1583, 1024);
                let (t8, t9) = add_sub_kirp_64::<MIN, MAX>(t8a, t9a);
                let (t14, t15) = sub_add_kirp_64::<MIN, MAX>(t15a, t14a);

                let (t10a, t13a) = mul2_64::<12, MIN>(c[5], c[11], 1931, -3612, 3612, 1931, 2048);
                let (t11a, t12a) = mul2_64::<12, MIN>(c[13], c[3], 3920, -1189, 1189, 3920, 2048);
                let (t10, t11) = sub_add_kirp_64::<MIN, MAX>(t11a, t10a);
                let (t12, t13) = add_sub_kirp_64::<MIN, MAX>(t12a, t13a);
                idct16_ortak_son::<MIN, MAX>(c, e, [t8, t9, t10, t11, t12, t13, t14, t15]);
            }

            #[target_feature(enable = "avx2")]
            #[inline(never)]
            pub(super) fn idct16_yarim<const MIN: i32, const MAX: i32>(c: &mut [$V; 16]) {
                let e = idct8_yarim::<MIN, MAX>([c[0], c[2], c[4], c[6]]);
                let z = $zero();
                let (t8a, t15a) = mul2_64::<12, MIN>(c[1], z, 401, 0, 4076, 0, 2048);
                let (t9a, t14a) = mul2_64::<12, MIN>(c[7], z, -2598, 0, 3166, 0, 2048);
                let (t8, t9) = add_sub_kirp_64::<MIN, MAX>(t8a, t9a);
                let (t14, t15) = sub_add_kirp_64::<MIN, MAX>(t15a, t14a);

                let (t10a, t13a) = mul2_64::<12, MIN>(c[5], z, 1931, 0, 3612, 0, 2048);
                let (t11a, t12a) = mul2_64::<12, MIN>(c[3], z, -1189, 0, 3920, 0, 2048);
                let (t10, t11) = sub_add_kirp_64::<MIN, MAX>(t11a, t10a);
                let (t12, t13) = add_sub_kirp_64::<MIN, MAX>(t12a, t13a);
                idct16_ortak_son::<MIN, MAX>(c, e, [t8, t9, t10, t11, t12, t13, t14, t15]);
            }

            #[target_feature(enable = "avx2")]
            #[inline(never)]
            pub(super) fn iadst16<const MIN: i32, const MAX: i32>(c: &mut [$V; 16], flip: bool) {
                let (t0, t1) = mul2_64::<12, MIN>(c[15], c[0], 4091, 201, 201, -4091, 2048);
                let (t8, t9) = mul2_64::<12, MIN>(c[7], c[8], 2751, 3035, 3035, -2751, 2048);
                let (t0a, t8a) = add_sub_kirp_64::<MIN, MAX>(t0, t8);
                let (t1a, t9a) = add_sub_kirp_64::<MIN, MAX>(t1, t9);
                let (t8, t9) = mul2::<MIN>(t8a, t9a, 4017, 799, 799, -4017);

                let (t4, t5) = mul2_64::<12, MIN>(c[11], c[4], 3703, 1751, 1751, -3703, 2048);
                let (t12, t13) = mul2_64::<12, MIN>(c[3], c[12], 1380, 3857, 3857, -1380, 2048);
                let (t4a, t12a) = add_sub_kirp_64::<MIN, MAX>(t4, t12);
                let (t5a, t13a) = add_sub_kirp_64::<MIN, MAX>(t5, t13);
                let (t12, t13) = mul2::<MIN>(t13a, t12a, 4017, -799, 799, 4017);

                let t0 = kirp::<MIN, MAX>($add32(t0a, t4a));
                let t1 = kirp::<MIN, MAX>($add32(t1a, t5a));
                let t4 = kirp::<MIN, MAX>($sub32(t0a, t4a));
                let t5 = kirp::<MIN, MAX>($sub32(t1a, t5a));
                let t8a = kirp::<MIN, MAX>($add32(t8, t12));
                let t9a = kirp::<MIN, MAX>($add32(t9, t13));
                let t12a = kirp::<MIN, MAX>($sub32(t8, t12));
                let t13a = kirp::<MIN, MAX>($sub32(t9, t13));

                let (t2, t3) = mul2_64::<12, MIN>(c[13], c[2], 3973, 995, 995, -3973, 2048);
                let (t10, t11) = mul2_64::<12, MIN>(c[5], c[10], 2106, 3513, 3513, -2106, 2048);
                let (t2a, t10a) = add_sub_kirp_64::<MIN, MAX>(t2, t10);
                let (t3a, t11a) = add_sub_kirp_64::<MIN, MAX>(t3, t11);
                let (t10, t11) = mul2::<MIN>(t10a, t11a, 2276, 3406, 3406, -2276);

                let (t6, t7) = mul2_64::<11, MIN>(c[9], c[6], 1645, 1220, 1220, -1645, 1024);
                let (t14, t15) = mul2_64::<12, MIN>(c[1], c[14], 601, 4052, 4052, -601, 2048);
                let (t6a, t14a) = add_sub_kirp_64::<MIN, MAX>(t6, t14);
                let (t7a, t15a) = add_sub_kirp_64::<MIN, MAX>(t7, t15);
                let (t14, t15) = mul2::<MIN>(t15a, t14a, 2276, -3406, 3406, 2276);

                let t2 = kirp::<MIN, MAX>($add32(t2a, t6a));
                let t3 = kirp::<MIN, MAX>($add32(t3a, t7a));
                let t6 = kirp::<MIN, MAX>($sub32(t2a, t6a));
                let t7 = kirp::<MIN, MAX>($sub32(t3a, t7a));
                let t10a = kirp::<MIN, MAX>($add32(t10, t14));
                let t11a = kirp::<MIN, MAX>($add32(t11, t15));
                let t14a = kirp::<MIN, MAX>($sub32(t10, t14));
                let t15a = kirp::<MIN, MAX>($sub32(t11, t15));

                let (t4a, t5a) = mul2::<MIN>(t4, t5, 3784, 1567, 1567, -3784);
                let (t6a, t7a) = mul2::<MIN>(t7, t6, 3784, -1567, 1567, 3784);
                let (t12, t13) = mul2::<MIN>(t12a, t13a, 3784, 1567, 1567, -3784);
                let (t14, t15) = mul2::<MIN>(t15a, t14a, 3784, -1567, 1567, 3784);

                let z = $zero();
                let o0 = kirp::<MIN, MAX>($add32(t0, t2));
                let o15 = $sub32(z, kirp::<MIN, MAX>($add32(t1, t3)));
                let t2a = kirp::<MIN, MAX>($sub32(t0, t2));
                let t3a = kirp::<MIN, MAX>($sub32(t1, t3));
                let o3 = $sub32(z, kirp::<MIN, MAX>($add32(t4a, t6a)));
                let o12 = kirp::<MIN, MAX>($add32(t5a, t7a));
                let t6 = kirp::<MIN, MAX>($sub32(t4a, t6a));
                let t7 = kirp::<MIN, MAX>($sub32(t5a, t7a));
                let o1 = $sub32(z, kirp::<MIN, MAX>($add32(t8a, t10a)));
                let o14 = kirp::<MIN, MAX>($add32(t9a, t11a));
                let t10 = kirp::<MIN, MAX>($sub32(t8a, t10a));
                let t11 = kirp::<MIN, MAX>($sub32(t9a, t11a));
                let o2 = kirp::<MIN, MAX>($add32(t12, t14));
                let o13 = $sub32(z, kirp::<MIN, MAX>($add32(t13, t15)));
                let t14a = kirp::<MIN, MAX>($sub32(t12, t14));
                let t15a = kirp::<MIN, MAX>($sub32(t13, t15));

                let o7 = $sub32(z, olcekle_181_ic($add32(t2a, t3a)));
                let o8 = olcekle_181_ic($sub32(t2a, t3a));
                let o4 = olcekle_181_ic($add32(t6, t7));
                let o11 = $sub32(z, olcekle_181_ic($sub32(t6, t7)));
                let o6 = olcekle_181_ic($add32(t10, t11));
                let o9 = $sub32(z, olcekle_181_ic($sub32(t10, t11)));
                let o5 = $sub32(z, olcekle_181_ic($add32(t14a, t15a)));
                let o10 = olcekle_181_ic($sub32(t14a, t15a));

                if flip {
                    c[0] = o15;
                    c[1] = o14;
                    c[2] = o13;
                    c[3] = o12;
                    c[4] = o11;
                    c[5] = o10;
                    c[6] = o9;
                    c[7] = o8;
                    c[8] = o7;
                    c[9] = o6;
                    c[10] = o5;
                    c[11] = o4;
                    c[12] = o3;
                    c[13] = o2;
                    c[14] = o1;
                    c[15] = o0;
                } else {
                    c[0] = o0;
                    c[1] = o1;
                    c[2] = o2;
                    c[3] = o3;
                    c[4] = o4;
                    c[5] = o5;
                    c[6] = o6;
                    c[7] = o7;
                    c[8] = o8;
                    c[9] = o9;
                    c[10] = o10;
                    c[11] = o11;
                    c[12] = o12;
                    c[13] = o13;
                    c[14] = o14;
                    c[15] = o15;
                }
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            pub(super) fn gecis_16<const MIN: i32, const MAX: i32>(
                c: &mut [$V; 16],
                kind: Tx1dType,
                yarim_sifir: bool,
            ) {
                match kind {
                    Tx1dType::Identity => {
                        if MIN >= -131_072 {
                            let k5793 = $set1_32(5793);
                            let rnd = $set1_32(1024);
                            let mut i = 0;
                            while i < 16 {
                                c[i] = $srai32::<11>($add32($mullo32(c[i], k5793), rnd));
                                i += 1;
                            }
                        } else {
                            let mut i = 0;
                            while i < 16 {
                                c[i] = kirp_64::<-16_777_216, 16_777_215>(mul1_64::<11, MIN>(
                                    c[i], 5793, 1024,
                                ));
                                i += 1;
                            }
                        }
                    }
                    Tx1dType::Dct => {
                        if yarim_sifir {
                            idct16_yarim::<MIN, MAX>(c);
                        } else {
                            idct16::<MIN, MAX>(c);
                        }
                    }
                    Tx1dType::Adst | Tx1dType::FlipAdst => {
                        iadst16::<MIN, MAX>(c, matches!(kind, Tx1dType::FlipAdst));
                    }
                }
            }
        }
    };
}

itx12_1d_impl!(
    x4,
    __m128i,
    0x0A,
    _mm_setzero_si128,
    _mm_set1_epi32,
    _mm_set1_epi64x,
    _mm_add_epi32,
    _mm_sub_epi32,
    _mm_max_epi32,
    _mm_min_epi32,
    _mm_srai_epi32,
    _mm_mullo_epi32,
    _mm_and_si128,
    _mm_xor_si128,
    _mm_mul_epi32,
    _mm_add_epi64,
    _mm_sub_epi64,
    _mm_srli_epi64,
    _mm_slli_epi64,
    _mm_cmpgt_epi64,
    _mm_blendv_epi8,
    _mm_blend_epi32
);

itx12_1d_impl!(
    x8,
    __m256i,
    0xAA,
    _mm256_setzero_si256,
    _mm256_set1_epi32,
    _mm256_set1_epi64x,
    _mm256_add_epi32,
    _mm256_sub_epi32,
    _mm256_max_epi32,
    _mm256_min_epi32,
    _mm256_srai_epi32,
    _mm256_mullo_epi32,
    _mm256_and_si256,
    _mm256_xor_si256,
    _mm256_mul_epi32,
    _mm256_add_epi64,
    _mm256_sub_epi64,
    _mm256_srli_epi64,
    _mm256_slli_epi64,
    _mm256_cmpgt_epi64,
    _mm256_blendv_epi8,
    _mm256_blend_epi32
);

// ---------------------------------------------------------------------------
// 2B blok gövdeleri (`w, h ≤ 16`, 12 bpc)
// ---------------------------------------------------------------------------

const ROW_MIN: i32 = -524_288;
const ROW_MAX: i32 = 524_287;
const COL_MIN: i32 = -131_072;
const COL_MAX: i32 = 131_071;

#[target_feature(enable = "avx2")]
#[inline]
fn blok_4x4<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<4, 4, 0, false>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let giris = [
        yukle_i32x4(coeff, 0),
        yukle_i32x4(coeff, 4),
        yukle_i32x4(coeff, 8),
        yukle_i32x4(coeff, 12),
    ];
    sifirla_katsayi::<16>(coeff);
    let p1 = x4::kaydir_dizi_18b::<4, 0>(x4::gecis_4::<ROW_MIN, ROW_MAX>(giris, yatay));
    let t = devrik_4x4_i32(p1);
    let cikis = x4::gecis_4::<COL_MIN, COL_MAX>(t, dikey);
    ekle_yaz_4x2(dst, stride, 0, cikis[0], cikis[1]);
    ekle_yaz_4x2(dst, stride, 2, cikis[2], cikis[3]);
}

#[target_feature(enable = "avx2")]
#[inline]
fn blok_4x8<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<4, 8, 0, true>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let giris = [
        x8::olcekle_181(yukle_i32x8(coeff, 0)),
        x8::olcekle_181(yukle_i32x8(coeff, 8)),
        x8::olcekle_181(yukle_i32x8(coeff, 16)),
        x8::olcekle_181(yukle_i32x8(coeff, 24)),
    ];
    sifirla_katsayi::<32>(coeff);
    let p1 = x8::kaydir_dizi_18b::<4, 0>(x8::gecis_4::<ROW_MIN, ROW_MAX>(giris, yatay));
    let t = devrik_4x8_i32(p1);
    let cikis = x4::gecis_8::<COL_MIN, COL_MAX>(t, dikey);
    ekle_yaz_4x2(dst, stride, 0, cikis[0], cikis[1]);
    ekle_yaz_4x2(dst, stride, 2, cikis[2], cikis[3]);
    ekle_yaz_4x2(dst, stride, 4, cikis[4], cikis[5]);
    ekle_yaz_4x2(dst, stride, 6, cikis[6], cikis[7]);
}

#[target_feature(enable = "avx2")]
#[inline]
fn blok_8x4<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<8, 4, 0, true>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let giris = [
        x4::olcekle_181(yukle_i32x4(coeff, 0)),
        x4::olcekle_181(yukle_i32x4(coeff, 4)),
        x4::olcekle_181(yukle_i32x4(coeff, 8)),
        x4::olcekle_181(yukle_i32x4(coeff, 12)),
        x4::olcekle_181(yukle_i32x4(coeff, 16)),
        x4::olcekle_181(yukle_i32x4(coeff, 20)),
        x4::olcekle_181(yukle_i32x4(coeff, 24)),
        x4::olcekle_181(yukle_i32x4(coeff, 28)),
    ];
    sifirla_katsayi::<32>(coeff);
    let p1 = x4::kaydir_dizi_18b::<8, 0>(x4::gecis_8::<ROW_MIN, ROW_MAX>(giris, yatay));
    let t = devrik_8x4_i32(p1);
    let cikis = x8::gecis_4::<COL_MIN, COL_MAX>(t, dikey);
    ekle_yaz_8x1(dst, 0, cikis[0]);
    ekle_yaz_8x1(dst, stride, cikis[1]);
    ekle_yaz_8x1(dst, stride * 2, cikis[2]);
    ekle_yaz_8x1(dst, stride * 3, cikis[3]);
}

#[target_feature(enable = "avx2")]
#[inline]
fn blok_8x8<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<8, 8, 1, false>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let giris = [
        yukle_i32x8(coeff, 0),
        yukle_i32x8(coeff, 8),
        yukle_i32x8(coeff, 16),
        yukle_i32x8(coeff, 24),
        yukle_i32x8(coeff, 32),
        yukle_i32x8(coeff, 40),
        yukle_i32x8(coeff, 48),
        yukle_i32x8(coeff, 56),
    ];
    sifirla_katsayi::<64>(coeff);
    let p1 = x8::kaydir_dizi_18b::<8, 1>(x8::gecis_8::<ROW_MIN, ROW_MAX>(giris, yatay));
    let t = devrik_8x8_i32(p1);
    let cikis = x8::gecis_8::<COL_MIN, COL_MAX>(t, dikey);
    let mut y = 0;
    while y < 8 {
        ekle_yaz_8x1(dst, y * stride, cikis[y]);
        y += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn blok_4x16_gecis1(coeff: &mut [i32], yatay: Tx1dType, yarim: bool) -> [__m128i; 16] {
    let g_ust = [
        yukle_i32x8(coeff, 0),
        yukle_i32x8(coeff, 16),
        yukle_i32x8(coeff, 32),
        yukle_i32x8(coeff, 48),
    ];
    let t_ust = devrik_4x8_i32(x8::kaydir_dizi_18b::<4, 1>(
        x8::gecis_4::<ROW_MIN, ROW_MAX>(g_ust, yatay),
    ));
    let mut t = [_mm_setzero_si128(); 16];
    let mut i = 0;
    while i < 8 {
        t[i] = t_ust[i];
        i += 1;
    }
    if !yarim {
        let g_alt = [
            yukle_i32x8(coeff, 8),
            yukle_i32x8(coeff, 24),
            yukle_i32x8(coeff, 40),
            yukle_i32x8(coeff, 56),
        ];
        let t_alt = devrik_4x8_i32(x8::kaydir_dizi_18b::<4, 1>(
            x8::gecis_4::<ROW_MIN, ROW_MAX>(g_alt, yatay),
        ));
        let mut j = 0;
        while j < 8 {
            t[8 + j] = t_alt[j];
            j += 1;
        }
    }
    sifirla_katsayi::<64>(coeff);
    t
}

#[target_feature(enable = "avx2")]
#[inline]
fn blok_4x16<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<4, 16, 1, false>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let yarim = matches!(txtp, TxType::DctDct) && eob <= 28;
    let mut t = blok_4x16_gecis1(coeff, yatay, yarim);
    x4::gecis_16::<COL_MIN, COL_MAX>(&mut t, dikey, yarim);
    let mut y = 0;
    while y < 16 {
        ekle_yaz_4x2(dst, stride, y, t[y], t[y + 1]);
        y += 2;
    }
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn blok_16x4_gecis1(coeff: &mut [i32], yatay: Tx1dType, yarim: bool) {
    let mut giris = [_mm_setzero_si128(); 16];
    let sinir = if yarim { 8 } else { 16 };
    let mut x = 0;
    while x < sinir {
        giris[x] = yukle_i32x4(coeff, x * 4);
        x += 1;
    }
    x4::gecis_16::<ROW_MIN, ROW_MAX>(&mut giris, yatay, yarim);
    let t_sol = devrik_8x4_i32([
        x4::kaydir_kirp_18b::<1>(giris[0]),
        x4::kaydir_kirp_18b::<1>(giris[1]),
        x4::kaydir_kirp_18b::<1>(giris[2]),
        x4::kaydir_kirp_18b::<1>(giris[3]),
        x4::kaydir_kirp_18b::<1>(giris[4]),
        x4::kaydir_kirp_18b::<1>(giris[5]),
        x4::kaydir_kirp_18b::<1>(giris[6]),
        x4::kaydir_kirp_18b::<1>(giris[7]),
    ]);
    let mut y = 0;
    while y < 4 {
        store_i32x8(chunk_mut(coeff, y * 8), t_sol[y]);
        y += 1;
    }
    let t_sag = devrik_8x4_i32([
        x4::kaydir_kirp_18b::<1>(giris[8]),
        x4::kaydir_kirp_18b::<1>(giris[9]),
        x4::kaydir_kirp_18b::<1>(giris[10]),
        x4::kaydir_kirp_18b::<1>(giris[11]),
        x4::kaydir_kirp_18b::<1>(giris[12]),
        x4::kaydir_kirp_18b::<1>(giris[13]),
        x4::kaydir_kirp_18b::<1>(giris[14]),
        x4::kaydir_kirp_18b::<1>(giris[15]),
    ]);
    let mut y = 0;
    while y < 4 {
        store_i32x8(chunk_mut(coeff, 32 + y * 8), t_sag[y]);
        y += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn blok_16x4<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<16, 4, 1, false>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let yarim = matches!(txtp, TxType::DctDct) && eob <= 28;
    blok_16x4_gecis1(coeff, yatay, yarim);
    let mut x_grup = 0;
    while x_grup < 2 {
        let taban = x_grup * 32;
        let x_bayt = x_grup * 16;
        let t = [
            yukle_i32x8(coeff, taban),
            yukle_i32x8(coeff, taban + 8),
            yukle_i32x8(coeff, taban + 16),
            yukle_i32x8(coeff, taban + 24),
        ];
        let z = _mm256_setzero_si256();
        store_i32x8(chunk_mut(coeff, taban), z);
        store_i32x8(chunk_mut(coeff, taban + 8), z);
        store_i32x8(chunk_mut(coeff, taban + 16), z);
        store_i32x8(chunk_mut(coeff, taban + 24), z);
        let c = x8::gecis_4::<COL_MIN, COL_MAX>(t, dikey);
        let mut y = 0;
        while y < 4 {
            ekle_yaz_8x1(dst, y * stride + x_bayt, c[y]);
            y += 1;
        }
        x_grup += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn blok_8x16_gecis1(coeff: &mut [i32], yatay: Tx1dType, yarim: bool) {
    let yb_sinir = if yarim { 1 } else { 2 };
    let mut yb = 0;
    while yb < yb_sinir {
        let y0 = yb * 8;
        let g = [
            x8::olcekle_181(yukle_i32x8(coeff, y0)),
            x8::olcekle_181(yukle_i32x8(coeff, y0 + 16)),
            x8::olcekle_181(yukle_i32x8(coeff, y0 + 32)),
            x8::olcekle_181(yukle_i32x8(coeff, y0 + 48)),
            x8::olcekle_181(yukle_i32x8(coeff, y0 + 64)),
            x8::olcekle_181(yukle_i32x8(coeff, y0 + 80)),
            x8::olcekle_181(yukle_i32x8(coeff, y0 + 96)),
            x8::olcekle_181(yukle_i32x8(coeff, y0 + 112)),
        ];
        let t = devrik_8x8_i32(x8::kaydir_dizi_18b::<8, 1>(
            x8::gecis_8::<ROW_MIN, ROW_MAX>(g, yatay),
        ));
        let mut dy = 0;
        while dy < 8 {
            store_i32x8(chunk_mut(coeff, y0 + dy * 16), t[dy]);
            dy += 1;
        }
        yb += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn blok_8x16<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<8, 16, 1, true>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let yarim = matches!(txtp, TxType::DctDct) && eob <= 42;
    blok_8x16_gecis1(coeff, yatay, yarim);
    let mut t = [_mm256_setzero_si256(); 16];
    let mut y = 0;
    while y < 8 {
        t[y] = yukle_i32x8(coeff, y * 16);
        if !yarim {
            t[8 + y] = yukle_i32x8(coeff, 8 + y * 16);
        }
        y += 1;
    }
    sifirla_katsayi::<128>(coeff);
    x8::gecis_16::<COL_MIN, COL_MAX>(&mut t, dikey, yarim);
    let mut y = 0;
    while y < 16 {
        ekle_yaz_8x1(dst, y * stride, t[y]);
        y += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn blok_16x8_gecis1(coeff: &mut [i32], yatay: Tx1dType, yarim: bool) {
    let mut g = [_mm256_setzero_si256(); 16];
    let sinir = if yarim { 8 } else { 16 };
    let mut x = 0;
    while x < sinir {
        g[x] = x8::olcekle_181(yukle_i32x8(coeff, x * 8));
        x += 1;
    }
    x8::gecis_16::<ROW_MIN, ROW_MAX>(&mut g, yatay, yarim);
    let mut x = 0;
    while x < 16 {
        store_i32x8(chunk_mut(coeff, x * 8), x8::kaydir_kirp_18b::<1>(g[x]));
        x += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn blok_16x8_gecis2_yari(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32],
    x_grup: usize,
    dikey: Tx1dType,
) {
    let taban = x_grup * 64;
    let x_bayt = x_grup * 16;
    let p = [
        yukle_i32x8(coeff, taban),
        yukle_i32x8(coeff, taban + 8),
        yukle_i32x8(coeff, taban + 16),
        yukle_i32x8(coeff, taban + 24),
        yukle_i32x8(coeff, taban + 32),
        yukle_i32x8(coeff, taban + 40),
        yukle_i32x8(coeff, taban + 48),
        yukle_i32x8(coeff, taban + 56),
    ];
    let z = _mm256_setzero_si256();
    let mut i = 0;
    while i < 64 {
        store_i32x8(chunk_mut(coeff, taban + i), z);
        i += 8;
    }
    let c = x8::gecis_8::<COL_MIN, COL_MAX>(devrik_8x8_i32(p), dikey);
    let mut y = 0;
    while y < 8 {
        ekle_yaz_8x1(dst, y * stride + x_bayt, c[y]);
        y += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn blok_16x8<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<16, 8, 1, true>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let yarim = matches!(txtp, TxType::DctDct) && eob <= 42;
    blok_16x8_gecis1(coeff, yatay, yarim);
    blok_16x8_gecis2_yari(dst, stride, coeff, 0, dikey);
    blok_16x8_gecis2_yari(dst, stride, coeff, 1, dikey);
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn blok_16x16_gecis1_yari(coeff: &mut [i32], y0: usize, yatay: Tx1dType, yarim: bool) {
    let mut g = [_mm256_setzero_si256(); 16];
    let sinir = if yarim { 8 } else { 16 };
    let mut x = 0;
    while x < sinir {
        g[x] = yukle_i32x8(coeff, y0 + x * 16);
        x += 1;
    }
    x8::gecis_16::<ROW_MIN, ROW_MAX>(&mut g, yatay, yarim);
    let tl = devrik_8x8_i32([
        x8::kaydir_kirp_18b::<2>(g[0]),
        x8::kaydir_kirp_18b::<2>(g[1]),
        x8::kaydir_kirp_18b::<2>(g[2]),
        x8::kaydir_kirp_18b::<2>(g[3]),
        x8::kaydir_kirp_18b::<2>(g[4]),
        x8::kaydir_kirp_18b::<2>(g[5]),
        x8::kaydir_kirp_18b::<2>(g[6]),
        x8::kaydir_kirp_18b::<2>(g[7]),
    ]);
    let mut x = 0;
    while x < 8 {
        store_i32x8(chunk_mut(coeff, y0 + x * 16), tl[x]);
        x += 1;
    }
    let tr = devrik_8x8_i32([
        x8::kaydir_kirp_18b::<2>(g[8]),
        x8::kaydir_kirp_18b::<2>(g[9]),
        x8::kaydir_kirp_18b::<2>(g[10]),
        x8::kaydir_kirp_18b::<2>(g[11]),
        x8::kaydir_kirp_18b::<2>(g[12]),
        x8::kaydir_kirp_18b::<2>(g[13]),
        x8::kaydir_kirp_18b::<2>(g[14]),
        x8::kaydir_kirp_18b::<2>(g[15]),
    ]);
    let mut x = 0;
    while x < 8 {
        store_i32x8(chunk_mut(coeff, y0 + (8 + x) * 16), tr[x]);
        x += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn blok_16x16_gecis2_yari(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32],
    x_grup: usize,
    dikey: Tx1dType,
    yarim: bool,
) {
    let taban = x_grup * 128;
    let x_bayt = x_grup * 16;
    let mut sutun = [_mm256_setzero_si256(); 16];
    let mut y = 0;
    while y < 8 {
        sutun[y] = yukle_i32x8(coeff, taban + y * 16);
        if !yarim {
            sutun[8 + y] = yukle_i32x8(coeff, taban + 8 + y * 16);
        }
        y += 1;
    }
    let z = _mm256_setzero_si256();
    let mut i = 0;
    while i < 128 {
        store_i32x8(chunk_mut(coeff, taban + i), z);
        i += 8;
    }
    x8::gecis_16::<COL_MIN, COL_MAX>(&mut sutun, dikey, yarim);
    let mut y = 0;
    while y < 16 {
        ekle_yaz_8x1(dst, y * stride + x_bayt, sutun[y]);
        y += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn blok_16x16<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<16, 16, 2, false>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let yarim = matches!(txtp, TxType::DctDct) && eob <= 35;
    blok_16x16_gecis1_yari(coeff, 0, yatay, yarim);
    if !yarim {
        blok_16x16_gecis1_yari(coeff, 8, yatay, false);
    }
    blok_16x16_gecis2_yari(dst, stride, coeff, 0, dikey, yarim);
    blok_16x16_gecis2_yari(dst, stride, coeff, 1, dikey, yarim);
}

/// `(TX, TXTP)` çiftine özelleştirilmiş 12 bpc AVX2 ters dönüşüm çekirdek gövdesi.
#[target_feature(enable = "avx2")]
#[inline(never)]
fn itx12_avx2_cekirdek<const TX: u8, const TXTP: u8>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
) {
    match TxSize::from_index(TX as usize).expect("geçerli tx") {
        TxSize::Tx4x4 => blok_4x4::<TXTP>(dst, dst_stride, coeff, eob),
        TxSize::Rtx4x8 => blok_4x8::<TXTP>(dst, dst_stride, coeff, eob),
        TxSize::Rtx8x4 => blok_8x4::<TXTP>(dst, dst_stride, coeff, eob),
        TxSize::Tx8x8 => blok_8x8::<TXTP>(dst, dst_stride, coeff, eob),
        TxSize::Rtx4x16 => blok_4x16::<TXTP>(dst, dst_stride, coeff, eob),
        TxSize::Rtx16x4 => blok_16x4::<TXTP>(dst, dst_stride, coeff, eob),
        TxSize::Rtx8x16 => blok_8x16::<TXTP>(dst, dst_stride, coeff, eob),
        TxSize::Rtx16x8 => blok_16x8::<TXTP>(dst, dst_stride, coeff, eob),
        TxSize::Tx16x16 => blok_16x16::<TXTP>(dst, dst_stride, coeff, eob),
        _ => unreachable!(),
    }
}

/// `(TX, TXTP)` çiftinin sözleşme denetimli güvenli 12 bpc AVX2 yuva girişi.
fn itx12_avx2_giris<const TX: u8, const TXTP: u8>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
    bit_depth: BitDepth,
) {
    if bit_depth != BitDepth::Twelve
        || tx as u8 != TX
        || txtp as u8 != TXTP
        || !itx16_sozlesmesi(dst.len(), dst_stride, coeff.len(), eob, tx, txtp)
    {
        match bit_depth {
            BitDepth::Eight => itx_skaler_devret::<8>(dst, dst_stride, coeff, eob, tx, txtp),
            BitDepth::Ten => itx_skaler_devret::<10>(dst, dst_stride, coeff, eob, tx, txtp),
            BitDepth::Twelve => itx_skaler_devret::<12>(dst, dst_stride, coeff, eob, tx, txtp),
        }
        return;
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir (`itx12_avx2_fn` /
    // `itx12_avx2_lookup`); belirteç AVX2 tespiti ve maske izninin kanıtıdır. (2) sınır:
    // `itx16_sozlesmesi` `dst_stride`'ın çift ve `≥ w·2` olduğunu, `dst`'nin `(h − 1)·dst_stride + w·2`
    // baytı ve `coeff`'un `1024` öğeyi kapsadığını garanti eder; bütün bellek erişimleri sabit
    // boyutlu `chunk`/`chunk_mut` dizileri üzerindendir. (3) hizalama/takma ad: `dst` ve `coeff`
    // ayrık `&mut` referanslardır, hizasız erişim kullanılır.
    unsafe {
        itx12_avx2_cekirdek::<TX, TXTP>(dst, dst_stride, coeff, eob);
    }
}

macro_rules! esle_16_tur {
    ($tx:ident, $txtp:expr) => {
        match $txtp {
            TxType::DctDct => {
                Some(itx12_avx2_giris::<{ TxSize::$tx as u8 }, { TxType::DctDct as u8 }>)
            }
            TxType::AdstDct => {
                Some(itx12_avx2_giris::<{ TxSize::$tx as u8 }, { TxType::AdstDct as u8 }>)
            }
            TxType::DctAdst => {
                Some(itx12_avx2_giris::<{ TxSize::$tx as u8 }, { TxType::DctAdst as u8 }>)
            }
            TxType::AdstAdst => {
                Some(itx12_avx2_giris::<{ TxSize::$tx as u8 }, { TxType::AdstAdst as u8 }>)
            }
            TxType::FlipadstDct => {
                Some(itx12_avx2_giris::<{ TxSize::$tx as u8 }, { TxType::FlipadstDct as u8 }>)
            }
            TxType::DctFlipadst => {
                Some(itx12_avx2_giris::<{ TxSize::$tx as u8 }, { TxType::DctFlipadst as u8 }>)
            }
            TxType::FlipadstFlipadst => {
                Some(itx12_avx2_giris::<{ TxSize::$tx as u8 }, { TxType::FlipadstFlipadst as u8 }>)
            }
            TxType::AdstFlipadst => {
                Some(itx12_avx2_giris::<{ TxSize::$tx as u8 }, { TxType::AdstFlipadst as u8 }>)
            }
            TxType::FlipadstAdst => {
                Some(itx12_avx2_giris::<{ TxSize::$tx as u8 }, { TxType::FlipadstAdst as u8 }>)
            }
            TxType::Idtx => Some(itx12_avx2_giris::<{ TxSize::$tx as u8 }, { TxType::Idtx as u8 }>),
            TxType::VDct => Some(itx12_avx2_giris::<{ TxSize::$tx as u8 }, { TxType::VDct as u8 }>),
            TxType::HDct => Some(itx12_avx2_giris::<{ TxSize::$tx as u8 }, { TxType::HDct as u8 }>),
            TxType::VAdst => {
                Some(itx12_avx2_giris::<{ TxSize::$tx as u8 }, { TxType::VAdst as u8 }>)
            }
            TxType::HAdst => {
                Some(itx12_avx2_giris::<{ TxSize::$tx as u8 }, { TxType::HAdst as u8 }>)
            }
            TxType::VFlipadst => {
                Some(itx12_avx2_giris::<{ TxSize::$tx as u8 }, { TxType::VFlipadst as u8 }>)
            }
            TxType::HFlipadst => {
                Some(itx12_avx2_giris::<{ TxSize::$tx as u8 }, { TxType::HFlipadst as u8 }>)
            }
            TxType::WhtWht => None,
        }
    };
}

/// `w, h ≤ 16` boyutlu 140 adet 12 bpc WHT-dışı AVX2 `itxfm_add[tx][txtp]` yuvası
/// (yalnız [`Avx2Token`] ile).
#[must_use]
pub const fn itx12_avx2_fn(_token: Avx2Token, tx: TxSize, txtp: TxType) -> Option<ItxFn> {
    match tx {
        TxSize::Tx4x4 => esle_16_tur!(Tx4x4, txtp),
        TxSize::Rtx4x8 => esle_16_tur!(Rtx4x8, txtp),
        TxSize::Rtx8x4 => esle_16_tur!(Rtx8x4, txtp),
        TxSize::Tx8x8 => esle_16_tur!(Tx8x8, txtp),
        TxSize::Rtx4x16 => esle_16_tur!(Rtx4x16, txtp),
        TxSize::Rtx16x4 => esle_16_tur!(Rtx16x4, txtp),
        TxSize::Rtx8x16 => esle_16_tur!(Rtx8x16, txtp),
        TxSize::Rtx16x8 => esle_16_tur!(Rtx16x8, txtp),
        TxSize::Tx16x16 => match txtp {
            TxType::DctDct => {
                Some(itx12_avx2_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::DctDct as u8 }>)
            }
            TxType::AdstDct => {
                Some(itx12_avx2_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::AdstDct as u8 }>)
            }
            TxType::DctAdst => {
                Some(itx12_avx2_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::DctAdst as u8 }>)
            }
            TxType::AdstAdst => {
                Some(itx12_avx2_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::AdstAdst as u8 }>)
            }
            TxType::FlipadstDct => {
                Some(itx12_avx2_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::FlipadstDct as u8 }>)
            }
            TxType::DctFlipadst => {
                Some(itx12_avx2_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::DctFlipadst as u8 }>)
            }
            TxType::FlipadstFlipadst => Some(
                itx12_avx2_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::FlipadstFlipadst as u8 }>,
            ),
            TxType::AdstFlipadst => {
                Some(itx12_avx2_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::AdstFlipadst as u8 }>)
            }
            TxType::FlipadstAdst => {
                Some(itx12_avx2_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::FlipadstAdst as u8 }>)
            }
            TxType::Idtx => {
                Some(itx12_avx2_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::Idtx as u8 }>)
            }
            TxType::VDct => {
                Some(itx12_avx2_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::VDct as u8 }>)
            }
            TxType::HDct => {
                Some(itx12_avx2_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::HDct as u8 }>)
            }
            _ => None,
        },
        _ => None,
    }
}

/// [`itx12_avx2_fn`] eşdeğer arama girişi.
#[must_use]
pub const fn itx12_avx2_lookup(token: Avx2Token, tx: TxSize, txtp: TxType) -> Option<ItxFn> {
    itx12_avx2_fn(token, tx, txtp)
}
