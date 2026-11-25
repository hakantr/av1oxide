//! 10 bpc WHT-dışı ters dönüşüm (`w, h ≤ 16`, 140 kayıt) x86_64 AVX2 çekirdekleri — P44 Adım 1.
//!
//! Bu modül dav1d'nin `src/x86/itx.h:228-247` ve `src/x86/itx16_avx2.asm` dosyalarındaki
//! `w, h ≤ 16` boyutlu **140** adet **yalnız 10 bpc** (`BitDepth::Ten`) AVX2 `itxfm_add`
//! kaydını **saf Rust intrinsics** ile gerçekler; dış `.asm`/`.S`, `asm!` ve `global_asm!`
//! yoktur (CEK-02). `12 bpc` (`BitDepth::Twelve`) bu modülde asla SIMD çalıştırmaz:
//!
//! - `TX_4X4`, `RTX_4X8`, `RTX_8X4`, `TX_8X8`, `RTX_4X16`, `RTX_16X4`, `RTX_8X16`,
//!   `RTX_16X8`: 16 WHT-dışı `TxType` türünün tamamı (`8 × 16 = 128`),
//! - `TX_16X16`: desteklenen 12 `TxType` (`DCT_DCT`, `ADST_DCT`, `DCT_ADST`,
//!   `ADST_ADST`, `FLIPADST_DCT`, `DCT_FLIPADST`, `FLIPADST_FLIPADST`,
//!   `ADST_FLIPADST`, `FLIPADST_ADST`, `IDTX`, `V_DCT`, `H_DCT`).
//!
//! ## Sözleşme, iki geçişli 10 bpc AVX2 aritmetik ve ISA saflığı (P34 / P44)
//!
//! - Girişte O(1) sözleşme denetimi ([`itx16_sozlesmesi`]) ile `(tx, txtp, bit_depth == Ten)`
//!   uyumu aranır; sözleşme dışı çağrıda (`12 bpc`, `8 bpc`, tek bayt `dst_stride` vb.)
//!   paniklemeden [`itx_skaler_devret`] üzerinden sayaç artırılır ve ilgili bit derinliğinin
//!   skaler monomorfuna devredilir.
//! - Pass 1 32-bit `i32` şeritlerde (`h == 4` için `4 × i32` `__m128i`, `h ≥ 8` için `8 × i32`
//!   `__m256i`) `[-131072, +131071]` (`18` bit `row_clip`) ile çalışır; Pass 1 sonunda
//!   `(v + rnd) >> shift` uygulanıp `_mm*_packs_epi32` ile `[-32768, +32767]` (`col_clip`)
//!   aralığına doyurularak `i16`'ya indirilir.
//! - Pass 2 `i16` kelebek aritmetiğinde (`w == 4` için `4 × i16`, `w == 8` için `8 × i16`
//!   `__m128i`, `w == 16` için `16 × i16` `__m256i`) `iidtx4`/`iidtx16`/`iadst4` 32-bit ara
//!   toplam korumalarıyla tamamlanıp `[0, 1023]` aralığına kırpılarak `u16` hedef düzleme eklenir.
//! - Bütün çekirdek ve yardımcılar yalnız `#[target_feature(enable = "avx2")]` taşır;
//!   başka `dsp/x86/*` modülleriyle `#[target_feature]` yardımcı paylaşmaz (P34) ve
//!   EVEX (`%zmm`, `{%k`) yönergesi üretmez. Modülde tek `unsafe` bloğu [`itx10_avx2_giris`]
//!   içindedir.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi32, _mm_adds_epi16, _mm_cvtsi64_si128, _mm_madd_epi16,
    _mm_max_epi16, _mm_max_epi32, _mm_min_epi16, _mm_min_epi32, _mm_mulhrs_epi16, _mm_mullo_epi32,
    _mm_packs_epi32, _mm_set1_epi16, _mm_set1_epi32, _mm_setr_epi32, _mm_setzero_si128,
    _mm_srai_epi32, _mm_srli_si128, _mm_sub_epi32, _mm_subs_epi16, _mm_unpackhi_epi16,
    _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi16, _mm_unpacklo_epi32,
    _mm_unpacklo_epi64, _mm256_add_epi32, _mm256_adds_epi16, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_madd_epi16, _mm256_max_epi16, _mm256_max_epi32,
    _mm256_min_epi16, _mm256_min_epi32, _mm256_mulhrs_epi16, _mm256_mullo_epi32,
    _mm256_packs_epi32, _mm256_permute2x128_si256, _mm256_set_m128i, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_setr_epi32, _mm256_setzero_si256, _mm256_srai_epi32,
    _mm256_sub_epi32, _mm256_subs_epi16, _mm256_unpackhi_epi16, _mm256_unpackhi_epi32,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi16, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
};

use super::mem::{
    chunk, chunk_mut, load_u8x16, load_u8x32, store_i32x8, store_u8x8, store_u8x16, store_u8x32,
};
use crate::cpu::Avx2Token;
use crate::dsp::itx::{ItxFn, TxSize, TxType, is_supported, itx_skaler_devret, itx16_sozlesmesi};
use crate::dsp::scalar::itx_1d::Tx1dType;
use crate::frame::BitDepth;

/// İki `i16` çarpanını `_mm*_madd_epi16` için `i32` şeritlerine paketler (`[c0, c1, ...]`).
#[inline(always)]
const fn cift_katsayi(c0: i16, c1: i16) -> i32 {
    (c0 as u16 as i32) | ((c1 as u16 as i32) << 16)
}

/// `coeff[..N]` bölgesini (`N` 8'in katı) 256-bit sıfır saklamalarıyla temizler.
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
// Pass 1 (`i32` 4 şerit `__m128i` ve 8 şerit `__m256i`) temel yardımcıları
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
fn kirp_18b_x4(v: __m128i) -> __m128i {
    _mm_min_epi32(
        _mm_max_epi32(v, _mm_set1_epi32(-131_072)),
        _mm_set1_epi32(131_071),
    )
}

#[target_feature(enable = "avx2")]
#[inline]
fn kirp_18b_x8(v: __m256i) -> __m256i {
    _mm256_min_epi32(
        _mm256_max_epi32(v, _mm256_set1_epi32(-131_072)),
        _mm256_set1_epi32(131_071),
    )
}

#[target_feature(enable = "avx2")]
#[inline]
fn olcekle_181_i32_x4(v: __m128i) -> __m128i {
    _mm_srai_epi32::<8>(_mm_add_epi32(
        _mm_mullo_epi32(v, _mm_set1_epi32(181)),
        _mm_set1_epi32(128),
    ))
}

#[target_feature(enable = "avx2")]
#[inline]
fn olcekle_181_i32_x8(v: __m256i) -> __m256i {
    _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_mullo_epi32(v, _mm256_set1_epi32(181)),
        _mm256_set1_epi32(128),
    ))
}

#[target_feature(enable = "avx2")]
#[inline]
fn mul1_i32_x4(a: __m128i, k: i32) -> __m128i {
    _mm_srai_epi32::<12>(_mm_add_epi32(
        _mm_mullo_epi32(a, _mm_set1_epi32(k)),
        _mm_set1_epi32(2048),
    ))
}

#[target_feature(enable = "avx2")]
#[inline]
fn mul1_i32_x8(a: __m256i, k: i32) -> __m256i {
    _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_mullo_epi32(a, _mm256_set1_epi32(k)),
        _mm256_set1_epi32(2048),
    ))
}

#[target_feature(enable = "avx2")]
#[inline]
fn mul2_i32_x4(a: __m128i, b: __m128i, c0: i32, c1: i32, d0: i32, d1: i32) -> (__m128i, __m128i) {
    let rnd = _mm_set1_epi32(2048);
    let u = _mm_srai_epi32::<12>(_mm_add_epi32(
        _mm_add_epi32(
            _mm_mullo_epi32(a, _mm_set1_epi32(c0)),
            _mm_mullo_epi32(b, _mm_set1_epi32(c1)),
        ),
        rnd,
    ));
    let v = _mm_srai_epi32::<12>(_mm_add_epi32(
        _mm_add_epi32(
            _mm_mullo_epi32(a, _mm_set1_epi32(d0)),
            _mm_mullo_epi32(b, _mm_set1_epi32(d1)),
        ),
        rnd,
    ));
    (u, v)
}

#[target_feature(enable = "avx2")]
#[inline]
fn mul2_i32_x8(a: __m256i, b: __m256i, c0: i32, c1: i32, d0: i32, d1: i32) -> (__m256i, __m256i) {
    let rnd = _mm256_set1_epi32(2048);
    let u = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_mullo_epi32(a, _mm256_set1_epi32(c0)),
            _mm256_mullo_epi32(b, _mm256_set1_epi32(c1)),
        ),
        rnd,
    ));
    let v = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_mullo_epi32(a, _mm256_set1_epi32(d0)),
            _mm256_mullo_epi32(b, _mm256_set1_epi32(d1)),
        ),
        rnd,
    ));
    (u, v)
}

#[target_feature(enable = "avx2")]
#[inline]
fn kaydir_i32_x4<const SHIFT: u32>(v: __m128i) -> __m128i {
    match SHIFT {
        0 => v,
        1 => _mm_srai_epi32::<1>(_mm_add_epi32(v, _mm_set1_epi32(1))),
        2 => _mm_srai_epi32::<2>(_mm_add_epi32(v, _mm_set1_epi32(2))),
        _ => unreachable!(),
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn kaydir_i32_x8<const SHIFT: u32>(v: __m256i) -> __m256i {
    match SHIFT {
        0 => v,
        1 => _mm256_srai_epi32::<1>(_mm256_add_epi32(v, _mm256_set1_epi32(1))),
        2 => _mm256_srai_epi32::<2>(_mm256_add_epi32(v, _mm256_set1_epi32(2))),
        _ => unreachable!(),
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn paketle_x4<const N: usize, const SHIFT: u32>(c: [__m128i; N]) -> [__m128i; N] {
    let mut out = [_mm_setzero_si128(); N];
    let mut i = 0;
    while i < N {
        let s = kaydir_i32_x4::<SHIFT>(c[i]);
        out[i] = _mm_packs_epi32(s, s);
        i += 1;
    }
    out
}

/// `8 × i32` (`__m256i`) sütun dizisini kaydırıp `8 × i16` (`__m128i`) sütun dizisine paketler (`col_clip`).
#[target_feature(enable = "avx2")]
#[inline]
fn paketle_y8_to_x8<const N: usize, const SHIFT: u32>(c: [__m256i; N]) -> [__m128i; N] {
    let mut out = [_mm_setzero_si128(); N];
    let mut i = 0;
    while i < N {
        let s = kaydir_i32_x8::<SHIFT>(c[i]);
        out[i] = _mm_packs_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256::<1>(s));
        i += 1;
    }
    out
}

/// `16` adet `8 × i32` (`__m256i`) sütununu kaydırıp `[col_k (8×i16) | col_{k+8} (8×i16)]`
/// düzeninde `8` adet `16 × i16` (`__m256i`) vektörüne paketler (`col_clip`).
#[target_feature(enable = "avx2")]
#[inline]
fn paketle_2sutun_x8<const SHIFT: u32>(c: &[__m256i; 16]) -> [__m256i; 8] {
    let mut out = [_mm256_setzero_si256(); 8];
    let mut k = 0;
    while k < 8 {
        let s0 = kaydir_i32_x8::<SHIFT>(c[k]);
        let s1 = kaydir_i32_x8::<SHIFT>(c[k + 8]);
        let lo = _mm256_permute2x128_si256::<0x20>(s0, s1);
        let hi = _mm256_permute2x128_si256::<0x31>(s0, s1);
        out[k] = _mm256_packs_epi32(lo, hi);
        k += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// Pass 2 (`i16` 4/8/16 şerit) temel yardımcıları
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
fn mul2_x4(a: __m128i, b: __m128i, c0: i16, c1: i16, d0: i16, d1: i16) -> (__m128i, __m128i) {
    let lo = _mm_unpacklo_epi16(a, b);
    let rnd = _mm_set1_epi32(2048);
    let kc = _mm_set1_epi32(cift_katsayi(c0, c1));
    let kd = _mm_set1_epi32(cift_katsayi(d0, d1));
    let u = _mm_srai_epi32::<12>(_mm_add_epi32(_mm_madd_epi16(lo, kc), rnd));
    let v = _mm_srai_epi32::<12>(_mm_add_epi32(_mm_madd_epi16(lo, kd), rnd));
    (_mm_packs_epi32(u, u), _mm_packs_epi32(v, v))
}

#[target_feature(enable = "avx2")]
#[inline]
fn mul2_x8(a: __m128i, b: __m128i, c0: i16, c1: i16, d0: i16, d1: i16) -> (__m128i, __m128i) {
    let lo = _mm_unpacklo_epi16(a, b);
    let hi = _mm_unpackhi_epi16(a, b);
    let rnd = _mm_set1_epi32(2048);
    let kc = _mm_set1_epi32(cift_katsayi(c0, c1));
    let kd = _mm_set1_epi32(cift_katsayi(d0, d1));
    let u_lo = _mm_srai_epi32::<12>(_mm_add_epi32(_mm_madd_epi16(lo, kc), rnd));
    let u_hi = _mm_srai_epi32::<12>(_mm_add_epi32(_mm_madd_epi16(hi, kc), rnd));
    let v_lo = _mm_srai_epi32::<12>(_mm_add_epi32(_mm_madd_epi16(lo, kd), rnd));
    let v_hi = _mm_srai_epi32::<12>(_mm_add_epi32(_mm_madd_epi16(hi, kd), rnd));
    (_mm_packs_epi32(u_lo, u_hi), _mm_packs_epi32(v_lo, v_hi))
}

#[target_feature(enable = "avx2")]
#[inline]
fn mul2_x16(a: __m256i, b: __m256i, c0: i16, c1: i16, d0: i16, d1: i16) -> (__m256i, __m256i) {
    let lo = _mm256_unpacklo_epi16(a, b);
    let hi = _mm256_unpackhi_epi16(a, b);
    let rnd = _mm256_set1_epi32(2048);
    let kc = _mm256_set1_epi32(cift_katsayi(c0, c1));
    let kd = _mm256_set1_epi32(cift_katsayi(d0, d1));
    let u_lo = _mm256_srai_epi32::<12>(_mm256_add_epi32(_mm256_madd_epi16(lo, kc), rnd));
    let u_hi = _mm256_srai_epi32::<12>(_mm256_add_epi32(_mm256_madd_epi16(hi, kc), rnd));
    let v_lo = _mm256_srai_epi32::<12>(_mm256_add_epi32(_mm256_madd_epi16(lo, kd), rnd));
    let v_hi = _mm256_srai_epi32::<12>(_mm256_add_epi32(_mm256_madd_epi16(hi, kd), rnd));
    (
        _mm256_packs_epi32(u_lo, u_hi),
        _mm256_packs_epi32(v_lo, v_hi),
    )
}

// ---------------------------------------------------------------------------
// Hedef `u16` düzlem piksel ekleme + `[0, 1023]` doyurmalı yazma yardımcıları
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
fn ekle_yaz_4x2(dst: &mut [u8], stride: usize, y_off: usize, r0: __m128i, r1: __m128i) {
    let b0 = y_off * stride;
    let b1 = b0 + stride;
    let d0 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, b0)));
    let d1 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, b1)));
    let d01 = _mm_unpacklo_epi64(d0, d1);
    let r01 = _mm_unpacklo_epi64(r0, r1);
    let s = _mm_min_epi16(
        _mm_max_epi16(_mm_adds_epi16(d01, r01), _mm_setzero_si128()),
        _mm_set1_epi16(1023),
    );
    store_u8x8(chunk_mut(dst, b0), s);
    store_u8x8(chunk_mut(dst, b1), _mm_srli_si128::<8>(s));
}

#[target_feature(enable = "avx2")]
#[inline]
fn ekle_yaz_8x1(dst: &mut [u8], off: usize, r: __m128i) {
    let d = load_u8x16(chunk::<u8, 16>(dst, off));
    let s = _mm_min_epi16(
        _mm_max_epi16(_mm_adds_epi16(d, r), _mm_setzero_si128()),
        _mm_set1_epi16(1023),
    );
    store_u8x16(chunk_mut(dst, off), s);
}

#[target_feature(enable = "avx2")]
#[inline]
fn ekle_yaz_16x1(dst: &mut [u8], off: usize, r: __m256i) {
    let d = load_u8x32(chunk::<u8, 32>(dst, off));
    let s = _mm256_min_epi16(
        _mm256_max_epi16(_mm256_adds_epi16(d, r), _mm256_setzero_si256()),
        _mm256_set1_epi16(1023),
    );
    store_u8x32(chunk_mut(dst, off), s);
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
    let dc16 = dc.clamp(-16_384, 16_384) as i16;

    match W {
        4 => {
            let dc_v = _mm_set1_epi16(dc16);
            let mut y = 0;
            while y < H {
                ekle_yaz_4x2(dst, stride, y, dc_v, dc_v);
                y += 2;
            }
        }
        8 => {
            let dc_v = _mm_set1_epi16(dc16);
            let mut y = 0;
            while y < H {
                ekle_yaz_8x1(dst, y * stride, dc_v);
                y += 1;
            }
        }
        16 => {
            let dc_v = _mm256_set1_epi16(dc16);
            let mut y = 0;
            while y < H {
                ekle_yaz_16x1(dst, y * stride, dc_v);
                y += 1;
            }
        }
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Matris devrikleme (transpose) yardımcıları (`i16`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
fn devrik_4x4(r: [__m128i; 4]) -> [__m128i; 4] {
    let a0 = _mm_unpacklo_epi16(r[0], r[1]);
    let a1 = _mm_unpacklo_epi16(r[2], r[3]);
    let b0 = _mm_unpacklo_epi32(a0, a1);
    let b1 = _mm_unpackhi_epi32(a0, a1);
    [b0, _mm_srli_si128::<8>(b0), b1, _mm_srli_si128::<8>(b1)]
}

#[target_feature(enable = "avx2")]
#[inline]
fn devrik_4x8(c: [__m128i; 4]) -> [__m128i; 8] {
    let a0 = _mm_unpacklo_epi16(c[0], c[1]);
    let a1 = _mm_unpackhi_epi16(c[0], c[1]);
    let a2 = _mm_unpacklo_epi16(c[2], c[3]);
    let a3 = _mm_unpackhi_epi16(c[2], c[3]);
    let r01 = _mm_unpacklo_epi32(a0, a2);
    let r23 = _mm_unpackhi_epi32(a0, a2);
    let r45 = _mm_unpacklo_epi32(a1, a3);
    let r67 = _mm_unpackhi_epi32(a1, a3);
    [
        r01,
        _mm_srli_si128::<8>(r01),
        r23,
        _mm_srli_si128::<8>(r23),
        r45,
        _mm_srli_si128::<8>(r45),
        r67,
        _mm_srli_si128::<8>(r67),
    ]
}

#[target_feature(enable = "avx2")]
#[inline]
fn devrik_8x4(c: [__m128i; 8]) -> [__m128i; 4] {
    let a0 = _mm_unpacklo_epi16(c[0], c[1]);
    let a1 = _mm_unpacklo_epi16(c[2], c[3]);
    let a2 = _mm_unpacklo_epi16(c[4], c[5]);
    let a3 = _mm_unpacklo_epi16(c[6], c[7]);
    let b0 = _mm_unpacklo_epi32(a0, a1);
    let b1 = _mm_unpackhi_epi32(a0, a1);
    let b2 = _mm_unpacklo_epi32(a2, a3);
    let b3 = _mm_unpackhi_epi32(a2, a3);
    [
        _mm_unpacklo_epi64(b0, b2),
        _mm_unpackhi_epi64(b0, b2),
        _mm_unpacklo_epi64(b1, b3),
        _mm_unpackhi_epi64(b1, b3),
    ]
}

#[target_feature(enable = "avx2")]
#[inline]
fn devrik_8x8(r: [__m128i; 8]) -> [__m128i; 8] {
    let a0 = _mm_unpacklo_epi16(r[0], r[1]);
    let a1 = _mm_unpackhi_epi16(r[0], r[1]);
    let a2 = _mm_unpacklo_epi16(r[2], r[3]);
    let a3 = _mm_unpackhi_epi16(r[2], r[3]);
    let a4 = _mm_unpacklo_epi16(r[4], r[5]);
    let a5 = _mm_unpackhi_epi16(r[4], r[5]);
    let a6 = _mm_unpacklo_epi16(r[6], r[7]);
    let a7 = _mm_unpackhi_epi16(r[6], r[7]);

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

/// İki adet `8 × 8` `i16` bloğunu (`[sol_8 | sag_8]`) `__m256i` şerit-içi (in-lane) devrikler.
#[target_feature(enable = "avx2")]
#[inline]
fn devrik_8x8_x16(r: &mut [__m256i; 8]) {
    let a0 = _mm256_unpacklo_epi16(r[0], r[1]);
    let a1 = _mm256_unpackhi_epi16(r[0], r[1]);
    let a2 = _mm256_unpacklo_epi16(r[2], r[3]);
    let a3 = _mm256_unpackhi_epi16(r[2], r[3]);
    let a4 = _mm256_unpacklo_epi16(r[4], r[5]);
    let a5 = _mm256_unpackhi_epi16(r[4], r[5]);
    let a6 = _mm256_unpacklo_epi16(r[6], r[7]);
    let a7 = _mm256_unpackhi_epi16(r[6], r[7]);

    let b0 = _mm256_unpacklo_epi32(a0, a2);
    let b1 = _mm256_unpackhi_epi32(a0, a2);
    let b2 = _mm256_unpacklo_epi32(a1, a3);
    let b3 = _mm256_unpackhi_epi32(a1, a3);
    let b4 = _mm256_unpacklo_epi32(a4, a6);
    let b5 = _mm256_unpackhi_epi32(a4, a6);
    let b6 = _mm256_unpacklo_epi32(a5, a7);
    let b7 = _mm256_unpackhi_epi32(a5, a7);

    r[0] = _mm256_unpacklo_epi64(b0, b4);
    r[1] = _mm256_unpackhi_epi64(b0, b4);
    r[2] = _mm256_unpacklo_epi64(b1, b5);
    r[3] = _mm256_unpackhi_epi64(b1, b5);
    r[4] = _mm256_unpacklo_epi64(b2, b6);
    r[5] = _mm256_unpackhi_epi64(b2, b6);
    r[6] = _mm256_unpacklo_epi64(b3, b7);
    r[7] = _mm256_unpackhi_epi64(b3, b7);
}

// ---------------------------------------------------------------------------
// Pass 1: Yatay 1B dönüşümler (`i32` 4 şerit `__m128i` ve 8 şerit `__m256i`)
// ---------------------------------------------------------------------------

macro_rules! impl_gecis1_i32 {
    (
        $V:ty,
        $zero:ident,
        $set1:ident,
        $add:ident,
        $sub:ident,
        $mullo:ident,
        $srai:ident,
        $kirp:ident,
        $olcekle_181:ident,
        $mul1:ident,
        $mul2:ident,
        $idct4:ident,
        $idct4_yarim:ident,
        $iadst4:ident,
        $gecis1_4:ident,
        $idct8:ident,
        $idct8_yarim:ident,
        $iadst8:ident,
        $gecis1_8:ident,
        $idct16_ortak_son:ident,
        $idct16:ident,
        $idct16_yarim:ident,
        $iadst16:ident,
        $gecis1_16:ident
    ) => {
        #[target_feature(enable = "avx2")]
        #[inline]
        fn $idct4(c: [$V; 4]) -> [$V; 4] {
            let t0 = $olcekle_181($add(c[0], c[2]));
            let t1 = $olcekle_181($sub(c[0], c[2]));
            let (t2, t3) = $mul2(c[1], c[3], 1567, -3784, 3784, 1567);
            [
                $kirp($add(t0, t3)),
                $kirp($add(t1, t2)),
                $kirp($sub(t1, t2)),
                $kirp($sub(t0, t3)),
            ]
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $idct4_yarim(c0: $V, c1: $V) -> [$V; 4] {
            let t0 = $olcekle_181(c0);
            let t2 = $mul1(c1, 1567);
            let t3 = $mul1(c1, 3784);
            [
                $kirp($add(t0, t3)),
                $kirp($add(t0, t2)),
                $kirp($sub(t0, t2)),
                $kirp($sub(t0, t3)),
            ]
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $iadst4(c: [$V; 4], flip: bool) -> [$V; 4] {
            let rnd = $set1(2048);
            let k1321 = $set1(1321);
            let k2482 = $set1(2482);
            let k3344 = $set1(3344);
            let k3803 = $set1(3803);

            let m1 = $mullo(c[1], k3344);

            let s0 = $add(
                $add($mullo(c[0], k1321), $mullo(c[2], k3803)),
                $add(m1, $mullo(c[3], k2482)),
            );
            let o0 = $srai::<12>($add(s0, rnd));

            let s1 = $add(
                $sub($mullo(c[0], k2482), $mullo(c[2], k1321)),
                $sub(m1, $mullo(c[3], k3803)),
            );
            let o1 = $srai::<12>($add(s1, rnd));

            let d023 = $add($sub(c[0], c[2]), c[3]);
            let o2 = $srai::<12>($add($mullo(d023, k3344), rnd));

            let s3 = $sub(
                $add($mullo(c[0], k3803), $mullo(c[2], k2482)),
                $add(m1, $mullo(c[3], k1321)),
            );
            let o3 = $srai::<12>($add(s3, rnd));

            if flip {
                [o3, o2, o1, o0]
            } else {
                [o0, o1, o2, o3]
            }
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $gecis1_4(c: [$V; 4], kind: Tx1dType) -> [$V; 4] {
            match kind {
                Tx1dType::Identity => [
                    $mul1(c[0], 5793),
                    $mul1(c[1], 5793),
                    $mul1(c[2], 5793),
                    $mul1(c[3], 5793),
                ],
                Tx1dType::Dct => $idct4(c),
                Tx1dType::Adst | Tx1dType::FlipAdst => {
                    $iadst4(c, matches!(kind, Tx1dType::FlipAdst))
                }
            }
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $idct8(c: [$V; 8]) -> [$V; 8] {
            let [t0, t1, t2, t3] = $idct4([c[0], c[2], c[4], c[6]]);
            let (t4a, t7a) = $mul2(c[1], c[7], 799, -4017, 4017, 799);
            let (t5a, t6a) = $mul2(c[5], c[3], 3406, -2276, 2276, 3406);
            let t4 = $kirp($add(t4a, t5a));
            let t5a = $kirp($sub(t4a, t5a));
            let t7 = $kirp($add(t7a, t6a));
            let t6a = $kirp($sub(t7a, t6a));
            let t5 = $olcekle_181($sub(t6a, t5a));
            let t6 = $olcekle_181($add(t6a, t5a));
            [
                $kirp($add(t0, t7)),
                $kirp($add(t1, t6)),
                $kirp($add(t2, t5)),
                $kirp($add(t3, t4)),
                $kirp($sub(t3, t4)),
                $kirp($sub(t2, t5)),
                $kirp($sub(t1, t6)),
                $kirp($sub(t0, t7)),
            ]
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $idct8_yarim(c: [$V; 4]) -> [$V; 8] {
            let [t0, t1, t2, t3] = $idct4_yarim(c[0], c[2]);
            let t4a = $mul1(c[1], 799);
            let t7a = $mul1(c[1], 4017);
            let t5a = $mul1(c[3], -2276);
            let t6a = $mul1(c[3], 3406);
            let t4 = $kirp($add(t4a, t5a));
            let t5a = $kirp($sub(t4a, t5a));
            let t7 = $kirp($add(t7a, t6a));
            let t6a = $kirp($sub(t7a, t6a));
            let t5 = $olcekle_181($sub(t6a, t5a));
            let t6 = $olcekle_181($add(t6a, t5a));
            [
                $kirp($add(t0, t7)),
                $kirp($add(t1, t6)),
                $kirp($add(t2, t5)),
                $kirp($add(t3, t4)),
                $kirp($sub(t3, t4)),
                $kirp($sub(t2, t5)),
                $kirp($sub(t1, t6)),
                $kirp($sub(t0, t7)),
            ]
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $iadst8(c: [$V; 8], flip: bool) -> [$V; 8] {
            let (t0a, t1a) = $mul2(c[7], c[0], 4076, 401, 401, -4076);
            let (t2a, t3a) = $mul2(c[5], c[2], 3612, 1931, 1931, -3612);
            let (t4a, t5a) = $mul2(c[3], c[4], 2598, 3166, 3166, -2598);
            let (t6a, t7a) = $mul2(c[1], c[6], 1189, 3920, 3920, -1189);

            let t0 = $kirp($add(t0a, t4a));
            let t1 = $kirp($add(t1a, t5a));
            let t2 = $kirp($add(t2a, t6a));
            let t3 = $kirp($add(t3a, t7a));
            let t4 = $kirp($sub(t0a, t4a));
            let t5 = $kirp($sub(t1a, t5a));
            let t6 = $kirp($sub(t2a, t6a));
            let t7 = $kirp($sub(t3a, t7a));

            let (t4a, t5a) = $mul2(t4, t5, 3784, 1567, 1567, -3784);
            let (t6a, t7a) = $mul2(t7, t6, 3784, -1567, 1567, 3784);

            let z = $zero();
            let o0 = $kirp($add(t0, t2));
            let o7 = $sub(z, $kirp($add(t1, t3)));
            let t2b = $kirp($sub(t0, t2));
            let t3b = $kirp($sub(t1, t3));
            let o1 = $sub(z, $kirp($add(t4a, t6a)));
            let o6 = $kirp($add(t5a, t7a));
            let t6b = $kirp($sub(t4a, t6a));
            let t7b = $kirp($sub(t5a, t7a));

            let o3 = $sub(z, $olcekle_181($add(t2b, t3b)));
            let o4 = $olcekle_181($sub(t2b, t3b));
            let o2 = $olcekle_181($add(t6b, t7b));
            let o5 = $sub(z, $olcekle_181($sub(t6b, t7b)));

            let mut o = [o0, o1, o2, o3, o4, o5, o6, o7];
            if flip {
                o.reverse();
            }
            o
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $gecis1_8(c: [$V; 8], kind: Tx1dType) -> [$V; 8] {
            match kind {
                Tx1dType::Identity => {
                    let mut o = c;
                    let mut i = 0;
                    while i < 8 {
                        o[i] = $add(c[i], c[i]);
                        i += 1;
                    }
                    o
                }
                Tx1dType::Dct => $idct8(c),
                Tx1dType::Adst | Tx1dType::FlipAdst => {
                    $iadst8(c, matches!(kind, Tx1dType::FlipAdst))
                }
            }
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $idct16_ortak_son(
            c: &mut [$V; 16],
            e: [$V; 8],
            [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a]: [$V; 8],
        ) {
            let t8 = $kirp($add(t8a, t9a));
            let t9 = $kirp($sub(t8a, t9a));
            let t10 = $kirp($sub(t11a, t10a));
            let t11 = $kirp($add(t11a, t10a));
            let t12 = $kirp($add(t12a, t13a));
            let t13 = $kirp($sub(t12a, t13a));
            let t14 = $kirp($sub(t15a, t14a));
            let t15 = $kirp($add(t15a, t14a));

            let (t9a, t14a) = $mul2(t14, t9, 1567, -3784, 3784, 1567);
            let (t10a, t13a) = $mul2(t13, t10, -3784, -1567, 1567, -3784);

            let t8a = $kirp($add(t8, t11));
            let t9 = $kirp($add(t9a, t10a));
            let t10 = $kirp($sub(t9a, t10a));
            let t11a = $kirp($sub(t8, t11));
            let t12a = $kirp($sub(t15, t12));
            let t13 = $kirp($sub(t14a, t13a));
            let t14 = $kirp($add(t14a, t13a));
            let t15a = $kirp($add(t15, t12));

            let t10a = $olcekle_181($sub(t13, t10));
            let t13a = $olcekle_181($add(t13, t10));
            let t11 = $olcekle_181($sub(t12a, t11a));
            let t12 = $olcekle_181($add(t12a, t11a));

            c[0] = $kirp($add(e[0], t15a));
            c[1] = $kirp($add(e[1], t14));
            c[2] = $kirp($add(e[2], t13a));
            c[3] = $kirp($add(e[3], t12));
            c[4] = $kirp($add(e[4], t11));
            c[5] = $kirp($add(e[5], t10a));
            c[6] = $kirp($add(e[6], t9));
            c[7] = $kirp($add(e[7], t8a));
            c[8] = $kirp($sub(e[7], t8a));
            c[9] = $kirp($sub(e[6], t9));
            c[10] = $kirp($sub(e[5], t10a));
            c[11] = $kirp($sub(e[4], t11));
            c[12] = $kirp($sub(e[3], t12));
            c[13] = $kirp($sub(e[2], t13a));
            c[14] = $kirp($sub(e[1], t14));
            c[15] = $kirp($sub(e[0], t15a));
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $idct16(c: &mut [$V; 16]) {
            let e = $idct8([c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]]);
            let (t8a, t15a) = $mul2(c[1], c[15], 401, -4076, 4076, 401);
            let (t9a, t14a) = $mul2(c[9], c[7], 3166, -2598, 2598, 3166);
            let (t10a, t13a) = $mul2(c[5], c[11], 1931, -3612, 3612, 1931);
            let (t11a, t12a) = $mul2(c[13], c[3], 3920, -1189, 1189, 3920);
            $idct16_ortak_son(c, e, [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a]);
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $idct16_yarim(c: &mut [$V; 16]) {
            let e = $idct8_yarim([c[0], c[2], c[4], c[6]]);
            let t8a = $mul1(c[1], 401);
            let t15a = $mul1(c[1], 4076);
            let t9a = $mul1(c[7], -2598);
            let t14a = $mul1(c[7], 3166);
            let t10a = $mul1(c[5], 1931);
            let t13a = $mul1(c[5], 3612);
            let t11a = $mul1(c[3], -1189);
            let t12a = $mul1(c[3], 3920);
            $idct16_ortak_son(c, e, [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a]);
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $iadst16(c: &mut [$V; 16], flip: bool) {
            let (t0, t1) = $mul2(c[15], c[0], 4091, 201, 201, -4091);
            let (t2, t3) = $mul2(c[13], c[2], 3973, 995, 995, -3973);
            let (t4, t5) = $mul2(c[11], c[4], 3703, 1751, 1751, -3703);
            let (t6, t7) = $mul2(c[9], c[6], 3290, 2440, 2440, -3290);
            let (t8, t9) = $mul2(c[7], c[8], 2751, 3035, 3035, -2751);
            let (t10, t11) = $mul2(c[5], c[10], 2106, 3513, 3513, -2106);
            let (t12, t13) = $mul2(c[3], c[12], 1380, 3857, 3857, -1380);
            let (t14, t15) = $mul2(c[1], c[14], 601, 4052, 4052, -601);

            let t0a = $kirp($add(t0, t8));
            let t1a = $kirp($add(t1, t9));
            let t2a = $kirp($add(t2, t10));
            let t3a = $kirp($add(t3, t11));
            let t4a = $kirp($add(t4, t12));
            let t5a = $kirp($add(t5, t13));
            let t6a = $kirp($add(t6, t14));
            let t7a = $kirp($add(t7, t15));
            let t8a = $kirp($sub(t0, t8));
            let t9a = $kirp($sub(t1, t9));
            let t10a = $kirp($sub(t2, t10));
            let t11a = $kirp($sub(t3, t11));
            let t12a = $kirp($sub(t4, t12));
            let t13a = $kirp($sub(t5, t13));
            let t14a = $kirp($sub(t6, t14));
            let t15a = $kirp($sub(t7, t15));

            let (t8, t9) = $mul2(t8a, t9a, 4017, 799, 799, -4017);
            let (t10, t11) = $mul2(t10a, t11a, 2276, 3406, 3406, -2276);
            let (t12, t13) = $mul2(t13a, t12a, 4017, -799, 799, 4017);
            let (t14, t15) = $mul2(t15a, t14a, 2276, -3406, 3406, 2276);

            let t0 = $kirp($add(t0a, t4a));
            let t1 = $kirp($add(t1a, t5a));
            let t2 = $kirp($add(t2a, t6a));
            let t3 = $kirp($add(t3a, t7a));
            let t4 = $kirp($sub(t0a, t4a));
            let t5 = $kirp($sub(t1a, t5a));
            let t6 = $kirp($sub(t2a, t6a));
            let t7 = $kirp($sub(t3a, t7a));
            let t8a = $kirp($add(t8, t12));
            let t9a = $kirp($add(t9, t13));
            let t10a = $kirp($add(t10, t14));
            let t11a = $kirp($add(t11, t15));
            let t12a = $kirp($sub(t8, t12));
            let t13a = $kirp($sub(t9, t13));
            let t14a = $kirp($sub(t10, t14));
            let t15a = $kirp($sub(t11, t15));

            let (t4a, t5a) = $mul2(t4, t5, 3784, 1567, 1567, -3784);
            let (t6a, t7a) = $mul2(t7, t6, 3784, -1567, 1567, 3784);
            let (t12, t13) = $mul2(t12a, t13a, 3784, 1567, 1567, -3784);
            let (t14, t15) = $mul2(t15a, t14a, 3784, -1567, 1567, 3784);

            let z = $zero();
            let o0 = $kirp($add(t0, t2));
            let o15 = $sub(z, $kirp($add(t1, t3)));
            let t2a = $kirp($sub(t0, t2));
            let t3a = $kirp($sub(t1, t3));
            let o3 = $sub(z, $kirp($add(t4a, t6a)));
            let o12 = $kirp($add(t5a, t7a));
            let t6 = $kirp($sub(t4a, t6a));
            let t7 = $kirp($sub(t5a, t7a));
            let o1 = $sub(z, $kirp($add(t8a, t10a)));
            let o14 = $kirp($add(t9a, t11a));
            let t10 = $kirp($sub(t8a, t10a));
            let t11 = $kirp($sub(t9a, t11a));
            let o2 = $kirp($add(t12, t14));
            let o13 = $sub(z, $kirp($add(t13, t15)));
            let t14a = $kirp($sub(t12, t14));
            let t15a = $kirp($sub(t13, t15));

            let o7 = $sub(z, $olcekle_181($add(t2a, t3a)));
            let o8 = $olcekle_181($sub(t2a, t3a));
            let o4 = $olcekle_181($add(t6, t7));
            let o11 = $sub(z, $olcekle_181($sub(t6, t7)));
            let o6 = $olcekle_181($add(t10, t11));
            let o9 = $sub(z, $olcekle_181($sub(t10, t11)));
            let o5 = $sub(z, $olcekle_181($add(t14a, t15a)));
            let o10 = $olcekle_181($sub(t14a, t15a));

            *c = [
                o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15,
            ];
            if flip {
                c.reverse();
            }
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $gecis1_16(c: &mut [$V; 16], kind: Tx1dType, yarim_sifir: bool) {
            match kind {
                Tx1dType::Identity => {
                    let k = $set1(5793);
                    let rnd = $set1(1024);
                    let mut i = 0;
                    while i < 16 {
                        c[i] = $srai::<11>($add($mullo(c[i], k), rnd));
                        i += 1;
                    }
                }
                Tx1dType::Dct => {
                    if yarim_sifir {
                        $idct16_yarim(c);
                    } else {
                        $idct16(c);
                    }
                }
                Tx1dType::Adst | Tx1dType::FlipAdst => {
                    $iadst16(c, matches!(kind, Tx1dType::FlipAdst));
                }
            }
        }
    };
}

impl_gecis1_i32!(
    __m128i,
    _mm_setzero_si128,
    _mm_set1_epi32,
    _mm_add_epi32,
    _mm_sub_epi32,
    _mm_mullo_epi32,
    _mm_srai_epi32,
    kirp_18b_x4,
    olcekle_181_i32_x4,
    mul1_i32_x4,
    mul2_i32_x4,
    idct4_i32_x4,
    idct4_yarim_i32_x4,
    iadst4_i32_x4,
    gecis1_4_x4,
    idct8_i32_x4,
    idct8_yarim_i32_x4,
    iadst8_i32_x4,
    gecis1_8_x4,
    idct16_ortak_son_i32_x4,
    idct16_i32_x4,
    idct16_yarim_i32_x4,
    iadst16_i32_x4,
    gecis1_16_x4
);

impl_gecis1_i32!(
    __m256i,
    _mm256_setzero_si256,
    _mm256_set1_epi32,
    _mm256_add_epi32,
    _mm256_sub_epi32,
    _mm256_mullo_epi32,
    _mm256_srai_epi32,
    kirp_18b_x8,
    olcekle_181_i32_x8,
    mul1_i32_x8,
    mul2_i32_x8,
    idct4_i32_x8,
    idct4_yarim_i32_x8,
    iadst4_i32_x8,
    gecis1_4_x8,
    idct8_i32_x8,
    idct8_yarim_i32_x8,
    iadst8_i32_x8,
    gecis1_8_x8,
    idct16_ortak_son_i32_x8,
    idct16_i32_x8,
    idct16_yarim_i32_x8,
    iadst16_i32_x8,
    gecis1_16_x8
);

// ---------------------------------------------------------------------------
// Pass 2: Dikey 1B dönüşümler (`i16` 4, 8 ve 16 şerit)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
fn iadst4_gecis2_x4(c: [__m128i; 4], flip: bool) -> [__m128i; 4] {
    let p02 = _mm_unpacklo_epi16(c[0], c[2]);
    let p13 = _mm_unpacklo_epi16(c[1], c[3]);
    let rnd = _mm_set1_epi32(34_816);

    let y0 = _mm_srai_epi32::<16>(_mm_add_epi32(
        _mm_add_epi32(
            _mm_madd_epi16(p02, _mm_set1_epi32(cift_katsayi(1321, 3803))),
            _mm_madd_epi16(p13, _mm_set1_epi32(cift_katsayi(3344, 2482))),
        ),
        rnd,
    ));
    let y1 = _mm_srai_epi32::<16>(_mm_add_epi32(
        _mm_add_epi32(
            _mm_madd_epi16(p02, _mm_set1_epi32(cift_katsayi(2482, -1321))),
            _mm_madd_epi16(p13, _mm_set1_epi32(cift_katsayi(3344, -3803))),
        ),
        rnd,
    ));
    let y2 = _mm_srai_epi32::<16>(_mm_add_epi32(
        _mm_add_epi32(
            _mm_madd_epi16(p02, _mm_set1_epi32(cift_katsayi(3344, -3344))),
            _mm_madd_epi16(p13, _mm_set1_epi32(cift_katsayi(0, 3344))),
        ),
        rnd,
    ));
    let y3 = _mm_srai_epi32::<16>(_mm_add_epi32(
        _mm_add_epi32(
            _mm_madd_epi16(p02, _mm_set1_epi32(cift_katsayi(3803, 2482))),
            _mm_madd_epi16(p13, _mm_set1_epi32(cift_katsayi(-3344, -1321))),
        ),
        rnd,
    ));

    let o0 = _mm_packs_epi32(y0, y0);
    let o1 = _mm_packs_epi32(y1, y1);
    let o2 = _mm_packs_epi32(y2, y2);
    let o3 = _mm_packs_epi32(y3, y3);
    if flip {
        [o3, o2, o1, o0]
    } else {
        [o0, o1, o2, o3]
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn iidtx4_gecis2_x4(c: [__m128i; 4]) -> [__m128i; 4] {
    let k = _mm_set1_epi16(1697 * 8);
    let bir = _mm_set1_epi32(cift_katsayi(1, 1));
    let sekiz = _mm_set1_epi32(8);
    let mut out = [_mm_setzero_si128(); 4];
    let mut i = 0;
    while i < 4 {
        let t = _mm_mulhrs_epi16(c[i], k);
        let lo = _mm_unpacklo_epi16(c[i], t);
        let s = _mm_srai_epi32::<4>(_mm_add_epi32(_mm_madd_epi16(lo, bir), sekiz));
        out[i] = _mm_packs_epi32(s, s);
        i += 1;
    }
    out
}

#[target_feature(enable = "avx2")]
#[inline]
fn iidtx16_gecis2_x4(c: &mut [__m128i; 16]) {
    let k = _mm_set1_epi16(1697 * 16);
    let iki_bir = _mm_set1_epi32(cift_katsayi(2, 1));
    let sekiz = _mm_set1_epi32(8);
    let mut i = 0;
    while i < 16 {
        let t = _mm_mulhrs_epi16(c[i], k);
        let lo = _mm_unpacklo_epi16(c[i], t);
        let s = _mm_srai_epi32::<4>(_mm_add_epi32(_mm_madd_epi16(lo, iki_bir), sekiz));
        c[i] = _mm_packs_epi32(s, s);
        i += 1;
    }
}

macro_rules! impl_gecis2_tam_serit {
    (
        $V:ty,
        $zero:ident,
        $set1_16:ident,
        $set1_32:ident,
        $adds16:ident,
        $subs16:ident,
        $mulhrs16:ident,
        $unpacklo16:ident,
        $unpackhi16:ident,
        $madd16:ident,
        $add32:ident,
        $srai32:ident,
        $packs32:ident,
        $iadst4_g2:ident,
        $iidtx4_g2:ident,
        $iidtx16_g2:ident
    ) => {
        #[target_feature(enable = "avx2")]
        #[inline]
        fn $iadst4_g2(c: [$V; 4], flip: bool) -> [$V; 4] {
            let p02_lo = $unpacklo16(c[0], c[2]);
            let p02_hi = $unpackhi16(c[0], c[2]);
            let p13_lo = $unpacklo16(c[1], c[3]);
            let p13_hi = $unpackhi16(c[1], c[3]);
            let rnd = $set1_32(34_816);

            let dot = |k02: i32, k13: i32| -> $V {
                let v02 = $set1_32(k02);
                let v13 = $set1_32(k13);
                let y_lo = $srai32::<16>($add32(
                    $add32($madd16(p02_lo, v02), $madd16(p13_lo, v13)),
                    rnd,
                ));
                let y_hi = $srai32::<16>($add32(
                    $add32($madd16(p02_hi, v02), $madd16(p13_hi, v13)),
                    rnd,
                ));
                $packs32(y_lo, y_hi)
            };

            let o0 = dot(cift_katsayi(1321, 3803), cift_katsayi(3344, 2482));
            let o1 = dot(cift_katsayi(2482, -1321), cift_katsayi(3344, -3803));
            let o2 = dot(cift_katsayi(3344, -3344), cift_katsayi(0, 3344));
            let o3 = dot(cift_katsayi(3803, 2482), cift_katsayi(-3344, -1321));
            if flip {
                [o3, o2, o1, o0]
            } else {
                [o0, o1, o2, o3]
            }
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $iidtx4_g2(c: [$V; 4]) -> [$V; 4] {
            let k = $set1_16(1697 * 8);
            let bir = $set1_32(cift_katsayi(1, 1));
            let sekiz = $set1_32(8);
            let mut out = [$zero(); 4];
            let mut i = 0;
            while i < 4 {
                let t = $mulhrs16(c[i], k);
                let lo = $unpacklo16(c[i], t);
                let hi = $unpackhi16(c[i], t);
                let s_lo = $srai32::<4>($add32($madd16(lo, bir), sekiz));
                let s_hi = $srai32::<4>($add32($madd16(hi, bir), sekiz));
                out[i] = $packs32(s_lo, s_hi);
                i += 1;
            }
            out
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $iidtx16_g2(c: &mut [$V; 16]) {
            let k = $set1_16(1697 * 16);
            let iki_bir = $set1_32(cift_katsayi(2, 1));
            let sekiz = $set1_32(8);
            let mut i = 0;
            while i < 16 {
                let t = $mulhrs16(c[i], k);
                let lo = $unpacklo16(c[i], t);
                let hi = $unpackhi16(c[i], t);
                let s_lo = $srai32::<4>($add32($madd16(lo, iki_bir), sekiz));
                let s_hi = $srai32::<4>($add32($madd16(hi, iki_bir), sekiz));
                c[i] = $packs32(s_lo, s_hi);
                i += 1;
            }
        }
    };
}

impl_gecis2_tam_serit!(
    __m128i,
    _mm_setzero_si128,
    _mm_set1_epi16,
    _mm_set1_epi32,
    _mm_adds_epi16,
    _mm_subs_epi16,
    _mm_mulhrs_epi16,
    _mm_unpacklo_epi16,
    _mm_unpackhi_epi16,
    _mm_madd_epi16,
    _mm_add_epi32,
    _mm_srai_epi32,
    _mm_packs_epi32,
    iadst4_gecis2_x8,
    iidtx4_gecis2_x8,
    iidtx16_gecis2_x8
);

impl_gecis2_tam_serit!(
    __m256i,
    _mm256_setzero_si256,
    _mm256_set1_epi16,
    _mm256_set1_epi32,
    _mm256_adds_epi16,
    _mm256_subs_epi16,
    _mm256_mulhrs_epi16,
    _mm256_unpacklo_epi16,
    _mm256_unpackhi_epi16,
    _mm256_madd_epi16,
    _mm256_add_epi32,
    _mm256_srai_epi32,
    _mm256_packs_epi32,
    iadst4_gecis2_x16,
    iidtx4_gecis2_x16,
    iidtx16_gecis2_x16
);

macro_rules! impl_gecis2_ortak {
    (
        $V:ty,
        $set1_16:ident,
        $adds16:ident,
        $subs16:ident,
        $mulhrs16:ident,
        $mul2:ident,
        $iadst4_g2:ident,
        $iidtx4_g2:ident,
        $iidtx16_g2:ident,
        $idct4:ident,
        $idct4_yarim:ident,
        $gecis2_4:ident,
        $idct8:ident,
        $idct8_yarim:ident,
        $iadst8_ham:ident,
        $adst8_son:ident,
        $gecis2_8:ident,
        $idct16_ortak_son:ident,
        $idct16:ident,
        $idct16_yarim:ident,
        $iadst16_ham:ident,
        $adst16_son:ident,
        $gecis2_16:ident
    ) => {
        #[target_feature(enable = "avx2")]
        #[inline]
        fn $idct4(c: [$V; 4]) -> [$V; 4] {
            let (t0, t1) = $mul2(c[0], c[2], 2896, 2896, 2896, -2896);
            let (t2, t3) = $mul2(c[1], c[3], 1567, -3784, 3784, 1567);
            [
                $adds16(t0, t3),
                $adds16(t1, t2),
                $subs16(t1, t2),
                $subs16(t0, t3),
            ]
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $idct4_yarim(c0: $V, c1: $V) -> [$V; 4] {
            let t0 = $mulhrs16(c0, $set1_16(2896 * 8));
            let t2 = $mulhrs16(c1, $set1_16(1567 * 8));
            let t3 = $mulhrs16(c1, $set1_16(3784 * 8));
            [
                $adds16(t0, t3),
                $adds16(t0, t2),
                $subs16(t0, t2),
                $subs16(t0, t3),
            ]
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $gecis2_4(c: [$V; 4], kind: Tx1dType) -> [$V; 4] {
            match kind {
                Tx1dType::Identity => $iidtx4_g2(c),
                Tx1dType::Dct => {
                    let o = $idct4(c);
                    let m = $set1_16(2048);
                    [
                        $mulhrs16(o[0], m),
                        $mulhrs16(o[1], m),
                        $mulhrs16(o[2], m),
                        $mulhrs16(o[3], m),
                    ]
                }
                Tx1dType::Adst | Tx1dType::FlipAdst => {
                    $iadst4_g2(c, matches!(kind, Tx1dType::FlipAdst))
                }
            }
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $idct8(c: [$V; 8]) -> [$V; 8] {
            let [t0, t1, t2, t3] = $idct4([c[0], c[2], c[4], c[6]]);
            let (t4a, t7a) = $mul2(c[1], c[7], 799, -4017, 4017, 799);
            let (t5a, t6a) = $mul2(c[5], c[3], 3406, -2276, 2276, 3406);
            let t4 = $adds16(t4a, t5a);
            let t5a = $subs16(t4a, t5a);
            let t7 = $adds16(t7a, t6a);
            let t6a = $subs16(t7a, t6a);
            let (t5, t6) = $mul2(t6a, t5a, 2896, -2896, 2896, 2896);
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
        fn $idct8_yarim(c: [$V; 4]) -> [$V; 8] {
            let [t0, t1, t2, t3] = $idct4_yarim(c[0], c[2]);
            let t4a = $mulhrs16(c[1], $set1_16(799 * 8));
            let t7a = $mulhrs16(c[1], $set1_16(4017 * 8));
            let t5a = $mulhrs16(c[3], $set1_16(-2276 * 8));
            let t6a = $mulhrs16(c[3], $set1_16(3406 * 8));
            let t4 = $adds16(t4a, t5a);
            let t5a = $subs16(t4a, t5a);
            let t7 = $adds16(t7a, t6a);
            let t6a = $subs16(t7a, t6a);
            let (t5, t6) = $mul2(t6a, t5a, 2896, -2896, 2896, 2896);
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
        fn $iadst8_ham(c: [$V; 8]) -> [$V; 8] {
            let (t0a, t1a) = $mul2(c[7], c[0], 4076, 401, 401, -4076);
            let (t2a, t3a) = $mul2(c[5], c[2], 3612, 1931, 1931, -3612);
            let (t4a, t5a) = $mul2(c[3], c[4], 2598, 3166, 3166, -2598);
            let (t6a, t7a) = $mul2(c[1], c[6], 1189, 3920, 3920, -1189);

            let t0 = $adds16(t0a, t4a);
            let t1 = $adds16(t1a, t5a);
            let t2 = $adds16(t2a, t6a);
            let t3 = $adds16(t3a, t7a);
            let t4 = $subs16(t0a, t4a);
            let t5 = $subs16(t1a, t5a);
            let t6 = $subs16(t2a, t6a);
            let t7 = $subs16(t3a, t7a);

            let (t4a, t5a) = $mul2(t4, t5, 3784, 1567, 1567, -3784);
            let (t6a, t7a) = $mul2(t7, t6, 3784, -1567, 1567, 3784);

            let u0 = $adds16(t0, t2);
            let u7 = $adds16(t1, t3);
            let t2b = $subs16(t0, t2);
            let t3b = $subs16(t1, t3);
            let u1 = $adds16(t4a, t6a);
            let u6 = $adds16(t5a, t7a);
            let t6b = $subs16(t4a, t6a);
            let t7b = $subs16(t5a, t7a);

            let (u3, u4) = $mul2(t2b, t3b, 2896, 2896, 2896, -2896);
            let (u2, u5) = $mul2(t6b, t7b, 2896, 2896, 2896, -2896);
            [u0, u1, u2, u3, u4, u5, u6, u7]
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $adst8_son(u: [$V; 8], flip: bool) -> [$V; 8] {
            let p = $set1_16(2048);
            let n = $set1_16(-2048);
            let mut o = [
                $mulhrs16(u[0], p),
                $mulhrs16(u[1], n),
                $mulhrs16(u[2], p),
                $mulhrs16(u[3], n),
                $mulhrs16(u[4], p),
                $mulhrs16(u[5], n),
                $mulhrs16(u[6], p),
                $mulhrs16(u[7], n),
            ];
            if flip {
                o.reverse();
            }
            o
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $gecis2_8(c: [$V; 8], kind: Tx1dType) -> [$V; 8] {
            match kind {
                Tx1dType::Identity => {
                    let m = $set1_16(4096);
                    let mut o = c;
                    let mut i = 0;
                    while i < 8 {
                        o[i] = $mulhrs16(c[i], m);
                        i += 1;
                    }
                    o
                }
                Tx1dType::Dct => {
                    let mut o = $idct8(c);
                    let m = $set1_16(2048);
                    let mut i = 0;
                    while i < 8 {
                        o[i] = $mulhrs16(o[i], m);
                        i += 1;
                    }
                    o
                }
                Tx1dType::Adst | Tx1dType::FlipAdst => {
                    $adst8_son($iadst8_ham(c), matches!(kind, Tx1dType::FlipAdst))
                }
            }
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $idct16_ortak_son(
            c: &mut [$V; 16],
            e: [$V; 8],
            [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a]: [$V; 8],
        ) {
            let t8 = $adds16(t8a, t9a);
            let t9 = $subs16(t8a, t9a);
            let t10 = $subs16(t11a, t10a);
            let t11 = $adds16(t11a, t10a);
            let t12 = $adds16(t12a, t13a);
            let t13 = $subs16(t12a, t13a);
            let t14 = $subs16(t15a, t14a);
            let t15 = $adds16(t15a, t14a);

            let (t9a, t14a) = $mul2(t14, t9, 1567, -3784, 3784, 1567);
            let (t10a, t13a) = $mul2(t13, t10, -3784, -1567, 1567, -3784);

            let t8a = $adds16(t8, t11);
            let t9 = $adds16(t9a, t10a);
            let t10 = $subs16(t9a, t10a);
            let t11a = $subs16(t8, t11);
            let t12a = $subs16(t15, t12);
            let t13 = $subs16(t14a, t13a);
            let t14 = $adds16(t14a, t13a);
            let t15a = $adds16(t15, t12);

            let (t10a, t13a) = $mul2(t13, t10, 2896, -2896, 2896, 2896);
            let (t11, t12) = $mul2(t12a, t11a, 2896, -2896, 2896, 2896);

            c[0] = $adds16(e[0], t15a);
            c[1] = $adds16(e[1], t14);
            c[2] = $adds16(e[2], t13a);
            c[3] = $adds16(e[3], t12);
            c[4] = $adds16(e[4], t11);
            c[5] = $adds16(e[5], t10a);
            c[6] = $adds16(e[6], t9);
            c[7] = $adds16(e[7], t8a);
            c[8] = $subs16(e[7], t8a);
            c[9] = $subs16(e[6], t9);
            c[10] = $subs16(e[5], t10a);
            c[11] = $subs16(e[4], t11);
            c[12] = $subs16(e[3], t12);
            c[13] = $subs16(e[2], t13a);
            c[14] = $subs16(e[1], t14);
            c[15] = $subs16(e[0], t15a);
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $idct16(c: &mut [$V; 16]) {
            let e = $idct8([c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]]);
            let (t8a, t15a) = $mul2(c[1], c[15], 401, -4076, 4076, 401);
            let (t9a, t14a) = $mul2(c[9], c[7], 3166, -2598, 2598, 3166);
            let (t10a, t13a) = $mul2(c[5], c[11], 1931, -3612, 3612, 1931);
            let (t11a, t12a) = $mul2(c[13], c[3], 3920, -1189, 1189, 3920);
            $idct16_ortak_son(c, e, [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a]);
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $idct16_yarim(c: &mut [$V; 16]) {
            let e = $idct8_yarim([c[0], c[2], c[4], c[6]]);
            let t8a = $mulhrs16(c[1], $set1_16(401 * 8));
            let t15a = $mulhrs16(c[1], $set1_16(4076 * 8));
            let t9a = $mulhrs16(c[7], $set1_16(-2598 * 8));
            let t14a = $mulhrs16(c[7], $set1_16(3166 * 8));
            let t10a = $mulhrs16(c[5], $set1_16(1931 * 8));
            let t13a = $mulhrs16(c[5], $set1_16(3612 * 8));
            let t11a = $mulhrs16(c[3], $set1_16(-1189 * 8));
            let t12a = $mulhrs16(c[3], $set1_16(3920 * 8));
            $idct16_ortak_son(c, e, [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a]);
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $iadst16_ham(c: &mut [$V; 16]) {
            let (t0, t1) = $mul2(c[15], c[0], 4091, 201, 201, -4091);
            let (t2, t3) = $mul2(c[13], c[2], 3973, 995, 995, -3973);
            let (t4, t5) = $mul2(c[11], c[4], 3703, 1751, 1751, -3703);
            let (t6, t7) = $mul2(c[9], c[6], 3290, 2440, 2440, -3290);
            let (t8, t9) = $mul2(c[7], c[8], 2751, 3035, 3035, -2751);
            let (t10, t11) = $mul2(c[5], c[10], 2106, 3513, 3513, -2106);
            let (t12, t13) = $mul2(c[3], c[12], 1380, 3857, 3857, -1380);
            let (t14, t15) = $mul2(c[1], c[14], 601, 4052, 4052, -601);

            let t0a = $adds16(t0, t8);
            let t1a = $adds16(t1, t9);
            let t2a = $adds16(t2, t10);
            let t3a = $adds16(t3, t11);
            let t4a = $adds16(t4, t12);
            let t5a = $adds16(t5, t13);
            let t6a = $adds16(t6, t14);
            let t7a = $adds16(t7, t15);
            let t8a = $subs16(t0, t8);
            let t9a = $subs16(t1, t9);
            let t10a = $subs16(t2, t10);
            let t11a = $subs16(t3, t11);
            let t12a = $subs16(t4, t12);
            let t13a = $subs16(t5, t13);
            let t14a = $subs16(t6, t14);
            let t15a = $subs16(t7, t15);

            let (t8, t9) = $mul2(t8a, t9a, 4017, 799, 799, -4017);
            let (t10, t11) = $mul2(t10a, t11a, 2276, 3406, 3406, -2276);
            let (t12, t13) = $mul2(t13a, t12a, 4017, -799, 799, 4017);
            let (t14, t15) = $mul2(t15a, t14a, 2276, -3406, 3406, 2276);

            let t0 = $adds16(t0a, t4a);
            let t1 = $adds16(t1a, t5a);
            let t2 = $adds16(t2a, t6a);
            let t3 = $adds16(t3a, t7a);
            let t4 = $subs16(t0a, t4a);
            let t5 = $subs16(t1a, t5a);
            let t6 = $subs16(t2a, t6a);
            let t7 = $subs16(t3a, t7a);
            let t8a = $adds16(t8, t12);
            let t9a = $adds16(t9, t13);
            let t10a = $adds16(t10, t14);
            let t11a = $adds16(t11, t15);
            let t12a = $subs16(t8, t12);
            let t13a = $subs16(t9, t13);
            let t14a = $subs16(t10, t14);
            let t15a = $subs16(t11, t15);

            let (t4a, t5a) = $mul2(t4, t5, 3784, 1567, 1567, -3784);
            let (t6a, t7a) = $mul2(t7, t6, 3784, -1567, 1567, 3784);
            let (t12, t13) = $mul2(t12a, t13a, 3784, 1567, 1567, -3784);
            let (t14, t15) = $mul2(t15a, t14a, 3784, -1567, 1567, 3784);

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

            let (u7, u8) = $mul2(t2a, t3a, 2896, 2896, 2896, -2896);
            let (u4, u11) = $mul2(t6, t7, 2896, 2896, 2896, -2896);
            let (u6, u9) = $mul2(t10, t11, 2896, 2896, 2896, -2896);
            let (u5, u10) = $mul2(t14a, t15a, 2896, 2896, 2896, -2896);

            *c = [
                u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15,
            ];
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $adst16_son(c: &mut [$V; 16], flip: bool) {
            let p = $set1_16(2048);
            let n = $set1_16(-2048);
            let mut i = 0;
            while i < 16 {
                c[i] = $mulhrs16(c[i], if i & 1 == 0 { p } else { n });
                i += 1;
            }
            if flip {
                c.reverse();
            }
        }

        #[target_feature(enable = "avx2")]
        #[inline]
        fn $gecis2_16(c: &mut [$V; 16], kind: Tx1dType, yarim_sifir: bool) {
            match kind {
                Tx1dType::Identity => $iidtx16_g2(c),
                Tx1dType::Dct => {
                    if yarim_sifir {
                        $idct16_yarim(c);
                    } else {
                        $idct16(c);
                    }
                    let m = $set1_16(2048);
                    let mut i = 0;
                    while i < 16 {
                        c[i] = $mulhrs16(c[i], m);
                        i += 1;
                    }
                }
                Tx1dType::Adst | Tx1dType::FlipAdst => {
                    $iadst16_ham(c);
                    $adst16_son(c, matches!(kind, Tx1dType::FlipAdst));
                }
            }
        }
    };
}

impl_gecis2_ortak!(
    __m128i,
    _mm_set1_epi16,
    _mm_adds_epi16,
    _mm_subs_epi16,
    _mm_mulhrs_epi16,
    mul2_x4,
    iadst4_gecis2_x4,
    iidtx4_gecis2_x4,
    iidtx16_gecis2_x4,
    idct4_x4,
    idct4_yarim_x4,
    gecis2_4_x4,
    idct8_x4,
    idct8_yarim_x4,
    iadst8_ham_x4,
    adst8_son_x4,
    gecis2_8_x4,
    idct16_ortak_son_x4,
    idct16_x4,
    idct16_yarim_x4,
    iadst16_ham_x4,
    adst16_son_x4,
    gecis2_16_x4
);

impl_gecis2_ortak!(
    __m128i,
    _mm_set1_epi16,
    _mm_adds_epi16,
    _mm_subs_epi16,
    _mm_mulhrs_epi16,
    mul2_x8,
    iadst4_gecis2_x8,
    iidtx4_gecis2_x8,
    iidtx16_gecis2_x8,
    idct4_x8,
    idct4_yarim_x8,
    gecis2_4_x8,
    idct8_x8,
    idct8_yarim_x8,
    iadst8_ham_x8,
    adst8_son_x8,
    gecis2_8_x8,
    idct16_ortak_son_x8,
    idct16_x8,
    idct16_yarim_x8,
    iadst16_ham_x8,
    adst16_son_x8,
    gecis2_16_x8
);

impl_gecis2_ortak!(
    __m256i,
    _mm256_set1_epi16,
    _mm256_adds_epi16,
    _mm256_subs_epi16,
    _mm256_mulhrs_epi16,
    mul2_x16,
    iadst4_gecis2_x16,
    iidtx4_gecis2_x16,
    iidtx16_gecis2_x16,
    idct4_x16,
    idct4_yarim_x16,
    gecis2_4_x16,
    idct8_x16,
    idct8_yarim_x16,
    iadst8_ham_x16,
    adst8_son_x16,
    gecis2_8_x16,
    idct16_ortak_son_x16,
    idct16_x16,
    idct16_yarim_x16,
    iadst16_ham_x16,
    adst16_son_x16,
    gecis2_16_x16
);

// ---------------------------------------------------------------------------
// 2B blok gövdeleri (`w, h ≤ 16`, 10 bpc AVX2)
// ---------------------------------------------------------------------------

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
    let p1 = paketle_x4::<4, 0>(gecis1_4_x4(giris, yatay));
    let t = devrik_4x4(p1);
    let cikis = gecis2_4_x4(t, dikey);
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
        olcekle_181_i32_x8(yukle_i32x8(coeff, 0)),
        olcekle_181_i32_x8(yukle_i32x8(coeff, 8)),
        olcekle_181_i32_x8(yukle_i32x8(coeff, 16)),
        olcekle_181_i32_x8(yukle_i32x8(coeff, 24)),
    ];
    sifirla_katsayi::<32>(coeff);
    let p1 = paketle_y8_to_x8::<4, 0>(gecis1_4_x8(giris, yatay));
    let t = devrik_4x8(p1);
    let cikis = gecis2_8_x4(t, dikey);
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
        olcekle_181_i32_x4(yukle_i32x4(coeff, 0)),
        olcekle_181_i32_x4(yukle_i32x4(coeff, 4)),
        olcekle_181_i32_x4(yukle_i32x4(coeff, 8)),
        olcekle_181_i32_x4(yukle_i32x4(coeff, 12)),
        olcekle_181_i32_x4(yukle_i32x4(coeff, 16)),
        olcekle_181_i32_x4(yukle_i32x4(coeff, 20)),
        olcekle_181_i32_x4(yukle_i32x4(coeff, 24)),
        olcekle_181_i32_x4(yukle_i32x4(coeff, 28)),
    ];
    sifirla_katsayi::<32>(coeff);
    let p1 = paketle_x4::<8, 0>(gecis1_8_x4(giris, yatay));
    let t = devrik_8x4(p1);
    let cikis = gecis2_4_x8(t, dikey);
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
    let p1 = paketle_y8_to_x8::<8, 1>(gecis1_8_x8(giris, yatay));
    let t = devrik_8x8(p1);
    let cikis = gecis2_8_x8(t, dikey);
    let mut y = 0;
    while y < 8 {
        ekle_yaz_8x1(dst, y * stride, cikis[y]);
        y += 1;
    }
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
    let g_ust = [
        yukle_i32x8(coeff, 0),
        yukle_i32x8(coeff, 16),
        yukle_i32x8(coeff, 32),
        yukle_i32x8(coeff, 48),
    ];
    let t_ust = devrik_4x8(paketle_y8_to_x8::<4, 1>(gecis1_4_x8(g_ust, yatay)));
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
        let t_alt = devrik_4x8(paketle_y8_to_x8::<4, 1>(gecis1_4_x8(g_alt, yatay)));
        let mut j = 0;
        while j < 8 {
            t[8 + j] = t_alt[j];
            j += 1;
        }
    }
    sifirla_katsayi::<64>(coeff);
    gecis2_16_x4(&mut t, dikey, yarim);
    let mut y = 0;
    while y < 16 {
        ekle_yaz_4x2(dst, stride, y, t[y], t[y + 1]);
        y += 2;
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
    let mut giris = [_mm_setzero_si128(); 16];
    let sinir = if yarim { 8 } else { 16 };
    let mut x = 0;
    while x < sinir {
        giris[x] = yukle_i32x4(coeff, x * 4);
        x += 1;
    }
    sifirla_katsayi::<64>(coeff);
    gecis1_16_x4(&mut giris, yatay, yarim);
    let p1 = paketle_x4::<16, 1>(giris);
    let t_sol = devrik_8x4([p1[0], p1[1], p1[2], p1[3], p1[4], p1[5], p1[6], p1[7]]);
    let t_sag = devrik_8x4([p1[8], p1[9], p1[10], p1[11], p1[12], p1[13], p1[14], p1[15]]);
    let t = [
        _mm256_set_m128i(t_sag[0], t_sol[0]),
        _mm256_set_m128i(t_sag[1], t_sol[1]),
        _mm256_set_m128i(t_sag[2], t_sol[2]),
        _mm256_set_m128i(t_sag[3], t_sol[3]),
    ];
    let cikis = gecis2_4_x16(t, dikey);
    let mut y = 0;
    while y < 4 {
        ekle_yaz_16x1(dst, y * stride, cikis[y]);
        y += 1;
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
    let mut g_ust = [_mm256_setzero_si256(); 8];
    let mut x = 0;
    while x < 8 {
        g_ust[x] = olcekle_181_i32_x8(yukle_i32x8(coeff, x * 16));
        x += 1;
    }
    let t_ust = devrik_8x8(paketle_y8_to_x8::<8, 1>(gecis1_8_x8(g_ust, yatay)));
    let mut t = [_mm_setzero_si128(); 16];
    let mut i = 0;
    while i < 8 {
        t[i] = t_ust[i];
        i += 1;
    }
    if !yarim {
        let mut g_alt = [_mm256_setzero_si256(); 8];
        let mut x2 = 0;
        while x2 < 8 {
            g_alt[x2] = olcekle_181_i32_x8(yukle_i32x8(coeff, x2 * 16 + 8));
            x2 += 1;
        }
        let t_alt = devrik_8x8(paketle_y8_to_x8::<8, 1>(gecis1_8_x8(g_alt, yatay)));
        let mut j = 0;
        while j < 8 {
            t[8 + j] = t_alt[j];
            j += 1;
        }
    }
    sifirla_katsayi::<128>(coeff);
    gecis2_16_x8(&mut t, dikey, yarim);
    let mut y = 0;
    while y < 16 {
        ekle_yaz_8x1(dst, y * stride, t[y]);
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
    if matches!(yatay, Tx1dType::Identity) {
        let k5793 = _mm256_set1_epi32(5793);
        let r1024 = _mm256_set1_epi32(1024);
        let mut t = [_mm256_setzero_si256(); 8];
        let mut k = 0;
        while k < 8 {
            let v0 = olcekle_181_i32_x8(yukle_i32x8(coeff, k * 8));
            let v1 = olcekle_181_i32_x8(yukle_i32x8(coeff, (k + 8) * 8));
            let s0 = kaydir_i32_x8::<1>(_mm256_srai_epi32::<11>(_mm256_add_epi32(
                _mm256_mullo_epi32(v0, k5793),
                r1024,
            )));
            let s1 = kaydir_i32_x8::<1>(_mm256_srai_epi32::<11>(_mm256_add_epi32(
                _mm256_mullo_epi32(v1, k5793),
                r1024,
            )));
            let lo = _mm256_permute2x128_si256::<0x20>(s0, s1);
            let hi = _mm256_permute2x128_si256::<0x31>(s0, s1);
            t[k] = _mm256_packs_epi32(lo, hi);
            k += 1;
        }
        sifirla_katsayi::<128>(coeff);
        devrik_8x8_x16(&mut t);
        let cikis = gecis2_8_x16(t, dikey);
        let mut y = 0;
        while y < 8 {
            ekle_yaz_16x1(dst, y * stride, cikis[y]);
            y += 1;
        }
        return;
    }
    let yarim = matches!(txtp, TxType::DctDct) && eob <= 42;
    let mut g = [_mm256_setzero_si256(); 16];
    let sinir = if yarim { 8 } else { 16 };
    let mut x = 0;
    while x < sinir {
        g[x] = olcekle_181_i32_x8(yukle_i32x8(coeff, x * 8));
        x += 1;
    }
    sifirla_katsayi::<128>(coeff);
    gecis1_16_x8(&mut g, yatay, yarim);
    let mut t = paketle_2sutun_x8::<1>(&g);
    devrik_8x8_x16(&mut t);
    let cikis = gecis2_8_x16(t, dikey);
    let mut y = 0;
    while y < 8 {
        ekle_yaz_16x1(dst, y * stride, cikis[y]);
        y += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn gecis1_16x16_serit(
    coeff: &[i32],
    y_off: usize,
    yatay: Tx1dType,
    yarim: bool,
    out: &mut [__m256i; 8],
) {
    let mut g = [_mm256_setzero_si256(); 16];
    let x_sinir = if yarim { 8 } else { 16 };
    let mut x = 0;
    while x < x_sinir {
        g[x] = yukle_i32x8(coeff, x * 16 + y_off);
        x += 1;
    }
    gecis1_16_x8(&mut g, yatay, yarim);
    *out = paketle_2sutun_x8::<2>(&g);
    devrik_8x8_x16(out);
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

    let mut sutun = [_mm256_setzero_si256(); 16];
    gecis1_16x16_serit(coeff, 0, yatay, yarim, chunk_mut(&mut sutun, 0));
    if !yarim {
        gecis1_16x16_serit(coeff, 8, yatay, false, chunk_mut(&mut sutun, 8));
    }

    sifirla_katsayi::<256>(coeff);
    gecis2_16_x16(&mut sutun, dikey, yarim);
    let mut y = 0;
    while y < 16 {
        ekle_yaz_16x1(dst, y * stride, sutun[y]);
        y += 1;
    }
}

/// `(TX, TXTP)` çiftine özelleştirilmiş 10 bpc AVX2 ters dönüşüm çekirdek gövdesi.
#[target_feature(enable = "avx2")]
#[inline(never)]
fn itx10_avx2_cekirdek<const TX: u8, const TXTP: u8>(
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

/// `(TX, TXTP)` çiftinin sözleşme denetimli güvenli 10 bpc AVX2 yuva girişi.
fn itx10_avx2_giris<const TX: u8, const TXTP: u8>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
    bit_depth: BitDepth,
) {
    if bit_depth != BitDepth::Ten
        || tx as u8 != TX
        || txtp as u8 != TXTP
        || matches!(txtp, TxType::WhtWht)
        || !is_supported(tx, txtp)
        || !itx16_sozlesmesi(dst.len(), dst_stride, coeff.len(), eob, tx, txtp)
    {
        match bit_depth {
            BitDepth::Eight => itx_skaler_devret::<8>(dst, dst_stride, coeff, eob, tx, txtp),
            BitDepth::Ten => itx_skaler_devret::<10>(dst, dst_stride, coeff, eob, tx, txtp),
            BitDepth::Twelve => itx_skaler_devret::<12>(dst, dst_stride, coeff, eob, tx, txtp),
        }
        return;
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir (`itx10_avx2_fn`);
    // belirteç AVX2 tespiti ve maske izninin kanıtıdır. (2) sınır: `itx16_sozlesmesi`
    // `dst_stride`'ın çift ve `≥ w·2` olduğunu, `dst`'nin `(h − 1)·dst_stride + w·2` baytı ve
    // `coeff`'un `1024` öğeyi kapsadığını garanti eder; bütün bellek erişimleri sabit boyutlu
    // `chunk`/`chunk_mut` dizileri üzerindendir. (3) hizalama/takma ad: `dst` ve `coeff` ayrık
    // `&mut` referanslardır, hizasız erişim kullanılır.
    unsafe {
        itx10_avx2_cekirdek::<TX, TXTP>(dst, dst_stride, coeff, eob);
    }
}

/// `(TX, TXTP)` çiftinin [`Avx2Token`] ile kapılı 10 bpc AVX2 `ItxFn` kurucusu.
#[must_use]
pub const fn itx10_avx2_fn<const TX: u8, const TXTP: u8>(_token: Avx2Token) -> ItxFn {
    itx10_avx2_giris::<TX, TXTP>
}

macro_rules! esle_16_tur {
    ($tx:ident, $txtp:expr) => {
        match $txtp {
            TxType::DctDct => {
                Some(itx10_avx2_fn::<{ TxSize::$tx as u8 }, { TxType::DctDct as u8 }>)
            }
            TxType::AdstDct => {
                Some(itx10_avx2_fn::<{ TxSize::$tx as u8 }, { TxType::AdstDct as u8 }>)
            }
            TxType::DctAdst => {
                Some(itx10_avx2_fn::<{ TxSize::$tx as u8 }, { TxType::DctAdst as u8 }>)
            }
            TxType::AdstAdst => {
                Some(itx10_avx2_fn::<{ TxSize::$tx as u8 }, { TxType::AdstAdst as u8 }>)
            }
            TxType::FlipadstDct => {
                Some(itx10_avx2_fn::<{ TxSize::$tx as u8 }, { TxType::FlipadstDct as u8 }>)
            }
            TxType::DctFlipadst => {
                Some(itx10_avx2_fn::<{ TxSize::$tx as u8 }, { TxType::DctFlipadst as u8 }>)
            }
            TxType::FlipadstFlipadst => {
                Some(itx10_avx2_fn::<{ TxSize::$tx as u8 }, { TxType::FlipadstFlipadst as u8 }>)
            }
            TxType::AdstFlipadst => {
                Some(itx10_avx2_fn::<{ TxSize::$tx as u8 }, { TxType::AdstFlipadst as u8 }>)
            }
            TxType::FlipadstAdst => {
                Some(itx10_avx2_fn::<{ TxSize::$tx as u8 }, { TxType::FlipadstAdst as u8 }>)
            }
            TxType::Idtx => Some(itx10_avx2_fn::<{ TxSize::$tx as u8 }, { TxType::Idtx as u8 }>),
            TxType::VDct => Some(itx10_avx2_fn::<{ TxSize::$tx as u8 }, { TxType::VDct as u8 }>),
            TxType::HDct => Some(itx10_avx2_fn::<{ TxSize::$tx as u8 }, { TxType::HDct as u8 }>),
            TxType::VAdst => Some(itx10_avx2_fn::<{ TxSize::$tx as u8 }, { TxType::VAdst as u8 }>),
            TxType::HAdst => Some(itx10_avx2_fn::<{ TxSize::$tx as u8 }, { TxType::HAdst as u8 }>),
            TxType::VFlipadst => {
                Some(itx10_avx2_fn::<{ TxSize::$tx as u8 }, { TxType::VFlipadst as u8 }>)
            }
            TxType::HFlipadst => {
                Some(itx10_avx2_fn::<{ TxSize::$tx as u8 }, { TxType::HFlipadst as u8 }>)
            }
            TxType::WhtWht => None,
        }
    };
}

/// `w, h ≤ 16` boyutlu 140 adet 10 bpc WHT-dışı AVX2 `itxfm_add[tx][txtp]` yuvası
/// için çalışma zamanı tablo araması.
#[must_use]
pub const fn itx10_avx2_lookup(tx: TxSize, txtp: TxType) -> Option<fn(Avx2Token) -> ItxFn> {
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
                Some(itx10_avx2_fn::<{ TxSize::Tx16x16 as u8 }, { TxType::DctDct as u8 }>)
            }
            TxType::AdstDct => {
                Some(itx10_avx2_fn::<{ TxSize::Tx16x16 as u8 }, { TxType::AdstDct as u8 }>)
            }
            TxType::DctAdst => {
                Some(itx10_avx2_fn::<{ TxSize::Tx16x16 as u8 }, { TxType::DctAdst as u8 }>)
            }
            TxType::AdstAdst => {
                Some(itx10_avx2_fn::<{ TxSize::Tx16x16 as u8 }, { TxType::AdstAdst as u8 }>)
            }
            TxType::FlipadstDct => {
                Some(itx10_avx2_fn::<{ TxSize::Tx16x16 as u8 }, { TxType::FlipadstDct as u8 }>)
            }
            TxType::DctFlipadst => {
                Some(itx10_avx2_fn::<{ TxSize::Tx16x16 as u8 }, { TxType::DctFlipadst as u8 }>)
            }
            TxType::FlipadstFlipadst => {
                Some(itx10_avx2_fn::<{ TxSize::Tx16x16 as u8 }, { TxType::FlipadstFlipadst as u8 }>)
            }
            TxType::AdstFlipadst => {
                Some(itx10_avx2_fn::<{ TxSize::Tx16x16 as u8 }, { TxType::AdstFlipadst as u8 }>)
            }
            TxType::FlipadstAdst => {
                Some(itx10_avx2_fn::<{ TxSize::Tx16x16 as u8 }, { TxType::FlipadstAdst as u8 }>)
            }
            TxType::Idtx => {
                Some(itx10_avx2_fn::<{ TxSize::Tx16x16 as u8 }, { TxType::Idtx as u8 }>)
            }
            TxType::VDct => {
                Some(itx10_avx2_fn::<{ TxSize::Tx16x16 as u8 }, { TxType::VDct as u8 }>)
            }
            TxType::HDct => {
                Some(itx10_avx2_fn::<{ TxSize::Tx16x16 as u8 }, { TxType::HDct as u8 }>)
            }
            _ => None,
        },
        _ => None,
    }
}
