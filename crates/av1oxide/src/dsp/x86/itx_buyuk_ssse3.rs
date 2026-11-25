//! Büyük boyutlu (`max(w, h) ∈ {32, 64}`, 15 kayıt) 8 bpc ters dönüşüm x86_64 SSSE3 çekirdekleri — P43 Adım 2.
//!
//! Bu modül dav1d'nin `src/x86/itx.h` ve `src/x86/itx_sse.asm` dosyalarındaki
//! `max(w, h) ∈ {32, 64}` boyutlu **15** adet 8 bpc SSSE3 `itxfm_add` kaydını
//! **saf Rust intrinsics** ile gerçekler; dış `.asm`/`.S`, `asm!` ve `global_asm!`
//! yoktur (CEK-02):
//!
//! - `RTX_8X32`, `RTX_32X8`, `RTX_16X32`, `RTX_32X16`, `TX_32X32`: `DCT_DCT` ve `IDTX` (`5 × 2 = 10`),
//! - `RTX_16X64`, `RTX_64X16`, `RTX_32X64`, `RTX_64X32`, `TX_64X64`: `DCT_DCT` (`5 × 1 = 5`).
//!
//! ## Sözleşme, `eob` hızlı yolları ve ISA saflığı (P34 / P43)
//!
//! - Girişte O(1) sözleşme denetimi ([`itx8_sozlesmesi`]) ile `(tx, txtp, bit_depth)`
//!   uyumu aranır; sözleşme dışı çağrıda paniklemeden [`itx_skaler_devret`] üzerinden
//!   sayaç artırılır ve skaler monomorfa devredilir.
//! - Katsayı sıfırlama sözleşmesi:
//!   - `txtp == TxType::DctDct && eob == 0` (DC-only): **yalnız `coeff[0] = 0`**
//!     sıfırlanır; `coeff[1..1024]` dokunulmadan kalır.
//!   - Genel yol (`eob ≥ 1` ya da `txtp != TxType::DctDct`): `eob` alt-kadran hızlı
//!     yolu seçilse bile `coeff[..sw·sh]` (`sw = min(w, 32), sh = min(h, 32)`)
//!     bölgesinin tamamı sıfırlanır; `coeff[sw·sh..1024]` kuyruğuna dokunulmaz.
//! - `dav1d` tarama tablolarıyla (`src/scan.c`, `src/x86/itx_sse.asm`) uyumlu
//!   alt-blok `eob` eşikleri (`eob ≤ 35`, `≤ 42`, `≤ 43`, `≤ 106`, `≤ 135`, `≤ 150`,
//!   `≤ 278`) sıfır alt-kadran okumalarını ve sıfır 1B kelebek kollarını atlar.
//! - Bütün çekirdek ve yardımcılar yalnız `#[target_feature(enable = "ssse3")]`
//!   taşır; `itx_ssse3.rs` ve `itx_wht.rs` ile hiçbir `#[target_feature]` yardımcı
//!   paylaşmaz (P34 modül yalıtımı) ve SSE4.1+ ya da VEX/EVEX yönergesi üretmez.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_adds_epi16, _mm_cvtsi64_si128, _mm_madd_epi16, _mm_mulhrs_epi16,
    _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi16, _mm_set1_epi32, _mm_setr_epi32,
    _mm_setzero_si128, _mm_srai_epi32, _mm_srli_si128, _mm_subs_epi16, _mm_unpackhi_epi8,
    _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi8,
    _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64,
};

use super::mem::{chunk, chunk_mut, load_u8x16, store_u8x8, store_u8x16};
use crate::cpu::Ssse3Token;
use crate::dsp::itx::{ItxFn, TxSize, TxType, itx_skaler_devret, itx8_sozlesmesi};
use crate::frame::BitDepth;

#[inline(always)]
const fn cift_katsayi(c0: i16, c1: i16) -> i32 {
    (c0 as u16 as i32) | ((c1 as u16 as i32) << 16)
}

#[target_feature(enable = "ssse3")]
#[inline]
fn yukle_katsayi8(coeff: &[i32], off: usize) -> __m128i {
    let c = chunk::<i32, 8>(coeff, off);
    let lo = _mm_setr_epi32(c[0], c[1], c[2], c[3]);
    let hi = _mm_setr_epi32(c[4], c[5], c[6], c[7]);
    _mm_packs_epi32(lo, hi)
}

#[target_feature(enable = "ssse3")]
#[inline]
fn yukle_katsayi8_opt<const RECT2: bool>(coeff: &[i32], off: usize) -> __m128i {
    let v = yukle_katsayi8(coeff, off);
    if RECT2 {
        _mm_mulhrs_epi16(v, _mm_set1_epi16(2896 * 8))
    } else {
        v
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn mul2(a: __m128i, b: __m128i, c0: i16, c1: i16, d0: i16, d1: i16) -> (__m128i, __m128i) {
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

#[target_feature(enable = "ssse3")]
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

// ---------------------------------------------------------------------------
// Hedef düzlem piksel ekleme + doyurmalı [0, 255] yazma yardımcıları
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
#[inline]
fn ekle_yaz_8x2_konum(
    dst: &mut [u8],
    stride: usize,
    x_off: usize,
    y_off: usize,
    r0: __m128i,
    r1: __m128i,
) {
    let b0 = y_off * stride + x_off;
    let b1 = b0 + stride;
    let d0 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, b0)));
    let d1 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, b1)));
    let z = _mm_setzero_si128();
    let s0 = _mm_adds_epi16(_mm_unpacklo_epi8(d0, z), r0);
    let s1 = _mm_adds_epi16(_mm_unpacklo_epi8(d1, z), r1);
    let p = _mm_packus_epi16(s0, s1);
    store_u8x8(chunk_mut(dst, b0), p);
    store_u8x8(chunk_mut(dst, b1), _mm_srli_si128::<8>(p));
}

#[target_feature(enable = "ssse3")]
#[inline]
fn ekle_yaz_16x1(dst: &mut [u8], off: usize, r_lo: __m128i, r_hi: __m128i) {
    let d = load_u8x16(chunk::<u8, 16>(dst, off));
    let z = _mm_setzero_si128();
    let s_lo = _mm_adds_epi16(_mm_unpacklo_epi8(d, z), r_lo);
    let s_hi = _mm_adds_epi16(_mm_unpackhi_epi8(d, z), r_hi);
    store_u8x16(chunk_mut(dst, off), _mm_packus_epi16(s_lo, s_hi));
}

#[target_feature(enable = "ssse3")]
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
    let dc_v = _mm_set1_epi16(dc.clamp(-512, 512) as i16);

    if W == 8 {
        let mut y = 0;
        while y < H {
            ekle_yaz_8x2_konum(dst, stride, 0, y, dc_v, dc_v);
            y += 2;
        }
    } else {
        let mut y = 0;
        while y < H {
            let taban = y * stride;
            let mut x = 0;
            while x < W {
                ekle_yaz_16x1(dst, taban + x, dc_v, dc_v);
                x += 16;
            }
            y += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// 1B DCT çekirdekleri (4, 8, 16, 32, 64 nokta; tam / yarım / çeyrek hızlı yollar)
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
#[inline]
fn idct4_tam(c: [__m128i; 4]) -> [__m128i; 4] {
    let (t0, t1) = mul2(c[0], c[2], 2896, 2896, 2896, -2896);
    let (t2, t3) = mul2(c[1], c[3], 1567, -3784, 3784, 1567);
    [
        _mm_adds_epi16(t0, t3),
        _mm_adds_epi16(t1, t2),
        _mm_subs_epi16(t1, t2),
        _mm_subs_epi16(t0, t3),
    ]
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idct4_yarim(c0: __m128i, c1: __m128i) -> [__m128i; 4] {
    let t0 = _mm_mulhrs_epi16(c0, _mm_set1_epi16(2896 * 8));
    let t2 = _mm_mulhrs_epi16(c1, _mm_set1_epi16(1567 * 8));
    let t3 = _mm_mulhrs_epi16(c1, _mm_set1_epi16(3784 * 8));
    [
        _mm_adds_epi16(t0, t3),
        _mm_adds_epi16(t0, t2),
        _mm_subs_epi16(t0, t2),
        _mm_subs_epi16(t0, t3),
    ]
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idct8_tam(c: [__m128i; 8]) -> [__m128i; 8] {
    let [t0, t1, t2, t3] = idct4_tam([c[0], c[2], c[4], c[6]]);
    let (t4a, t7a) = mul2(c[1], c[7], 799, -4017, 4017, 799);
    let (t5a, t6a) = mul2(c[5], c[3], 3406, -2276, 2276, 3406);
    let t4 = _mm_adds_epi16(t4a, t5a);
    let t5a = _mm_subs_epi16(t4a, t5a);
    let t7 = _mm_adds_epi16(t7a, t6a);
    let t6a = _mm_subs_epi16(t7a, t6a);
    let (t5, t6) = mul2(t6a, t5a, 2896, -2896, 2896, 2896);
    [
        _mm_adds_epi16(t0, t7),
        _mm_adds_epi16(t1, t6),
        _mm_adds_epi16(t2, t5),
        _mm_adds_epi16(t3, t4),
        _mm_subs_epi16(t3, t4),
        _mm_subs_epi16(t2, t5),
        _mm_subs_epi16(t1, t6),
        _mm_subs_epi16(t0, t7),
    ]
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idct8_yarim(c: [__m128i; 4]) -> [__m128i; 8] {
    let [t0, t1, t2, t3] = idct4_yarim(c[0], c[2]);
    let t4a = _mm_mulhrs_epi16(c[1], _mm_set1_epi16(799 * 8));
    let t7a = _mm_mulhrs_epi16(c[1], _mm_set1_epi16(4017 * 8));
    let t5a = _mm_mulhrs_epi16(c[3], _mm_set1_epi16(-2276 * 8));
    let t6a = _mm_mulhrs_epi16(c[3], _mm_set1_epi16(3406 * 8));
    let t4 = _mm_adds_epi16(t4a, t5a);
    let t5a = _mm_subs_epi16(t4a, t5a);
    let t7 = _mm_adds_epi16(t7a, t6a);
    let t6a = _mm_subs_epi16(t7a, t6a);
    let (t5, t6) = mul2(t6a, t5a, 2896, -2896, 2896, 2896);
    [
        _mm_adds_epi16(t0, t7),
        _mm_adds_epi16(t1, t6),
        _mm_adds_epi16(t2, t5),
        _mm_adds_epi16(t3, t4),
        _mm_subs_epi16(t3, t4),
        _mm_subs_epi16(t2, t5),
        _mm_subs_epi16(t1, t6),
        _mm_subs_epi16(t0, t7),
    ]
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idct16_ortak_son(
    e: [__m128i; 8],
    [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a]: [__m128i; 8],
) -> [__m128i; 16] {
    let t8 = _mm_adds_epi16(t8a, t9a);
    let t9 = _mm_subs_epi16(t8a, t9a);
    let t10 = _mm_subs_epi16(t11a, t10a);
    let t11 = _mm_adds_epi16(t11a, t10a);
    let t12 = _mm_adds_epi16(t12a, t13a);
    let t13 = _mm_subs_epi16(t12a, t13a);
    let t14 = _mm_subs_epi16(t15a, t14a);
    let t15 = _mm_adds_epi16(t15a, t14a);

    let (t9a, t14a) = mul2(t14, t9, 1567, -3784, 3784, 1567);
    let (t10a, t13a) = mul2(t13, t10, -3784, -1567, 1567, -3784);

    let t8a = _mm_adds_epi16(t8, t11);
    let t9 = _mm_adds_epi16(t9a, t10a);
    let t10 = _mm_subs_epi16(t9a, t10a);
    let t11a = _mm_subs_epi16(t8, t11);
    let t12a = _mm_subs_epi16(t15, t12);
    let t13 = _mm_subs_epi16(t14a, t13a);
    let t14 = _mm_adds_epi16(t14a, t13a);
    let t15a = _mm_adds_epi16(t15, t12);

    let (t10a, t13a) = mul2(t13, t10, 2896, -2896, 2896, 2896);
    let (t11, t12) = mul2(t12a, t11a, 2896, -2896, 2896, 2896);

    [
        _mm_adds_epi16(e[0], t15a),
        _mm_adds_epi16(e[1], t14),
        _mm_adds_epi16(e[2], t13a),
        _mm_adds_epi16(e[3], t12),
        _mm_adds_epi16(e[4], t11),
        _mm_adds_epi16(e[5], t10a),
        _mm_adds_epi16(e[6], t9),
        _mm_adds_epi16(e[7], t8a),
        _mm_subs_epi16(e[7], t8a),
        _mm_subs_epi16(e[6], t9),
        _mm_subs_epi16(e[5], t10a),
        _mm_subs_epi16(e[4], t11),
        _mm_subs_epi16(e[3], t12),
        _mm_subs_epi16(e[2], t13a),
        _mm_subs_epi16(e[1], t14),
        _mm_subs_epi16(e[0], t15a),
    ]
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idct16_tam(c: [__m128i; 16]) -> [__m128i; 16] {
    let e = idct8_tam([c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]]);
    let (t8a, t15a) = mul2(c[1], c[15], 401, -4076, 4076, 401);
    let (t9a, t14a) = mul2(c[9], c[7], 3166, -2598, 2598, 3166);
    let (t10a, t13a) = mul2(c[5], c[11], 1931, -3612, 3612, 1931);
    let (t11a, t12a) = mul2(c[13], c[3], 3920, -1189, 1189, 3920);
    idct16_ortak_son(e, [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a])
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idct16_yarim(c: [__m128i; 8]) -> [__m128i; 16] {
    let e = idct8_yarim([c[0], c[2], c[4], c[6]]);
    let t8a = _mm_mulhrs_epi16(c[1], _mm_set1_epi16(401 * 8));
    let t15a = _mm_mulhrs_epi16(c[1], _mm_set1_epi16(4076 * 8));
    let t9a = _mm_mulhrs_epi16(c[7], _mm_set1_epi16(-2598 * 8));
    let t14a = _mm_mulhrs_epi16(c[7], _mm_set1_epi16(3166 * 8));
    let t10a = _mm_mulhrs_epi16(c[5], _mm_set1_epi16(1931 * 8));
    let t13a = _mm_mulhrs_epi16(c[5], _mm_set1_epi16(3612 * 8));
    let t11a = _mm_mulhrs_epi16(c[3], _mm_set1_epi16(-1189 * 8));
    let t12a = _mm_mulhrs_epi16(c[3], _mm_set1_epi16(3920 * 8));
    idct16_ortak_son(e, [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a])
}

#[target_feature(enable = "ssse3")]
#[allow(clippy::too_many_arguments)]
#[inline]
fn idct32_ortak_son(
    e: [__m128i; 16],
    t16a: __m128i,
    t17a: __m128i,
    t18a: __m128i,
    t19a: __m128i,
    t20a: __m128i,
    t21a: __m128i,
    t22a: __m128i,
    t23a: __m128i,
    t24a: __m128i,
    t25a: __m128i,
    t26a: __m128i,
    t27a: __m128i,
    t28a: __m128i,
    t29a: __m128i,
    t30a: __m128i,
    t31a: __m128i,
) -> [__m128i; 32] {
    let t16 = _mm_adds_epi16(t16a, t17a);
    let t17 = _mm_subs_epi16(t16a, t17a);
    let t18 = _mm_subs_epi16(t19a, t18a);
    let t19 = _mm_adds_epi16(t19a, t18a);
    let t20 = _mm_adds_epi16(t20a, t21a);
    let t21 = _mm_subs_epi16(t20a, t21a);
    let t22 = _mm_subs_epi16(t23a, t22a);
    let t23 = _mm_adds_epi16(t23a, t22a);
    let t24 = _mm_adds_epi16(t24a, t25a);
    let t25 = _mm_subs_epi16(t24a, t25a);
    let t26 = _mm_subs_epi16(t27a, t26a);
    let t27 = _mm_adds_epi16(t27a, t26a);
    let t28 = _mm_adds_epi16(t28a, t29a);
    let t29 = _mm_subs_epi16(t28a, t29a);
    let t30 = _mm_subs_epi16(t31a, t30a);
    let t31 = _mm_adds_epi16(t31a, t30a);

    let (t17a, t30a) = mul2(t30, t17, 799, -4017, 4017, 799);
    let (t18a, t29a) = mul2(t29, t18, -4017, -799, 799, -4017);
    let (t21a, t26a) = mul2(t26, t21, 3406, -2276, 2276, 3406);
    let (t22a, t25a) = mul2(t25, t22, -2276, -3406, 3406, -2276);

    let t16a = _mm_adds_epi16(t16, t19);
    let t17 = _mm_adds_epi16(t17a, t18a);
    let t18 = _mm_subs_epi16(t17a, t18a);
    let t19a = _mm_subs_epi16(t16, t19);
    let t20a = _mm_subs_epi16(t23, t20);
    let t21 = _mm_subs_epi16(t22a, t21a);
    let t22 = _mm_adds_epi16(t22a, t21a);
    let t23a = _mm_adds_epi16(t23, t20);
    let t24a = _mm_adds_epi16(t24, t27);
    let t25 = _mm_adds_epi16(t25a, t26a);
    let t26 = _mm_subs_epi16(t25a, t26a);
    let t27a = _mm_subs_epi16(t24, t27);
    let t28a = _mm_subs_epi16(t31, t28);
    let t29 = _mm_subs_epi16(t30a, t29a);
    let t30 = _mm_adds_epi16(t30a, t29a);
    let t31a = _mm_adds_epi16(t31, t28);

    let (t18a, t29a) = mul2(t29, t18, 1567, -3784, 3784, 1567);
    let (t19, t28) = mul2(t28a, t19a, 1567, -3784, 3784, 1567);
    let (t20, t27) = mul2(t27a, t20a, -3784, -1567, 1567, -3784);
    let (t21a, t26a) = mul2(t26, t21, -3784, -1567, 1567, -3784);

    let t16 = _mm_adds_epi16(t16a, t23a);
    let t17a = _mm_adds_epi16(t17, t22);
    let t18 = _mm_adds_epi16(t18a, t21a);
    let t19a = _mm_adds_epi16(t19, t20);
    let t20a = _mm_subs_epi16(t19, t20);
    let t21 = _mm_subs_epi16(t18a, t21a);
    let t22a = _mm_subs_epi16(t17, t22);
    let t23 = _mm_subs_epi16(t16a, t23a);
    let t24 = _mm_subs_epi16(t31a, t24a);
    let t25a = _mm_subs_epi16(t30, t25);
    let t26 = _mm_subs_epi16(t29a, t26a);
    let t27a = _mm_subs_epi16(t28, t27);
    let t28a = _mm_adds_epi16(t28, t27);
    let t29 = _mm_adds_epi16(t29a, t26a);
    let t30a = _mm_adds_epi16(t30, t25);
    let t31 = _mm_adds_epi16(t31a, t24a);

    let (t20, t27) = mul2(t27a, t20a, 2896, -2896, 2896, 2896);
    let (t21a, t26a) = mul2(t26, t21, 2896, -2896, 2896, 2896);
    let (t22, t25) = mul2(t25a, t22a, 2896, -2896, 2896, 2896);
    let (t23a, t24a) = mul2(t24, t23, 2896, -2896, 2896, 2896);

    let odd = [
        t16, t17a, t18, t19a, t20, t21a, t22, t23a, t24a, t25, t26a, t27, t28a, t29, t30a, t31,
    ];
    let mut out = [_mm_setzero_si128(); 32];
    let mut i = 0;
    while i < 16 {
        out[i] = _mm_adds_epi16(e[i], odd[15 - i]);
        out[31 - i] = _mm_subs_epi16(e[i], odd[15 - i]);
        i += 1;
    }
    out
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idct32_tam(c: &[__m128i; 32]) -> [__m128i; 32] {
    let e = idct16_tam([
        c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14], c[16], c[18], c[20], c[22], c[24],
        c[26], c[28], c[30],
    ]);
    let (t16a, t31a) = mul2(c[1], c[31], 201, -4091, 4091, 201);
    let (t17a, t30a) = mul2(c[17], c[15], 3035, -2751, 2751, 3035);
    let (t18a, t29a) = mul2(c[9], c[23], 1751, -3703, 3703, 1751);
    let (t19a, t28a) = mul2(c[25], c[7], 3857, -1380, 1380, 3857);
    let (t20a, t27a) = mul2(c[5], c[27], 995, -3973, 3973, 995);
    let (t21a, t26a) = mul2(c[21], c[11], 3513, -2106, 2106, 3513);
    let (t22a, t25a) = mul2(c[13], c[19], 2440, -3290, 3290, 2440);
    let (t23a, t24a) = mul2(c[29], c[3], 4052, -601, 601, 4052);
    idct32_ortak_son(
        e, t16a, t17a, t18a, t19a, t20a, t21a, t22a, t23a, t24a, t25a, t26a, t27a, t28a, t29a,
        t30a, t31a,
    )
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idct32_yarim(c: &[__m128i; 16]) -> [__m128i; 32] {
    let e = idct16_yarim([c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]]);
    let t16a = _mm_mulhrs_epi16(c[1], _mm_set1_epi16(201 * 8));
    let t31a = _mm_mulhrs_epi16(c[1], _mm_set1_epi16(4091 * 8));
    let t17a = _mm_mulhrs_epi16(c[15], _mm_set1_epi16(-2751 * 8));
    let t30a = _mm_mulhrs_epi16(c[15], _mm_set1_epi16(3035 * 8));
    let t18a = _mm_mulhrs_epi16(c[9], _mm_set1_epi16(1751 * 8));
    let t29a = _mm_mulhrs_epi16(c[9], _mm_set1_epi16(3703 * 8));
    let t19a = _mm_mulhrs_epi16(c[7], _mm_set1_epi16(-1380 * 8));
    let t28a = _mm_mulhrs_epi16(c[7], _mm_set1_epi16(3857 * 8));
    let t20a = _mm_mulhrs_epi16(c[5], _mm_set1_epi16(995 * 8));
    let t27a = _mm_mulhrs_epi16(c[5], _mm_set1_epi16(3973 * 8));
    let t21a = _mm_mulhrs_epi16(c[11], _mm_set1_epi16(-2106 * 8));
    let t26a = _mm_mulhrs_epi16(c[11], _mm_set1_epi16(3513 * 8));
    let t22a = _mm_mulhrs_epi16(c[13], _mm_set1_epi16(2440 * 8));
    let t25a = _mm_mulhrs_epi16(c[13], _mm_set1_epi16(3290 * 8));
    let t23a = _mm_mulhrs_epi16(c[3], _mm_set1_epi16(-601 * 8));
    let t24a = _mm_mulhrs_epi16(c[3], _mm_set1_epi16(4052 * 8));
    idct32_ortak_son(
        e, t16a, t17a, t18a, t19a, t20a, t21a, t22a, t23a, t24a, t25a, t26a, t27a, t28a, t29a,
        t30a, t31a,
    )
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idct32_ceyrek(c: &[__m128i; 8]) -> [__m128i; 32] {
    let z = _mm_setzero_si128();
    let e = idct16_yarim([c[0], c[2], c[4], c[6], z, z, z, z]);
    let t16a = _mm_mulhrs_epi16(c[1], _mm_set1_epi16(201 * 8));
    let t31a = _mm_mulhrs_epi16(c[1], _mm_set1_epi16(4091 * 8));
    let t19a = _mm_mulhrs_epi16(c[7], _mm_set1_epi16(-1380 * 8));
    let t28a = _mm_mulhrs_epi16(c[7], _mm_set1_epi16(3857 * 8));
    let t20a = _mm_mulhrs_epi16(c[5], _mm_set1_epi16(995 * 8));
    let t27a = _mm_mulhrs_epi16(c[5], _mm_set1_epi16(3973 * 8));
    let t23a = _mm_mulhrs_epi16(c[3], _mm_set1_epi16(-601 * 8));
    let t24a = _mm_mulhrs_epi16(c[3], _mm_set1_epi16(4052 * 8));
    idct32_ortak_son(
        e, t16a, z, z, t19a, t20a, z, z, t23a, t24a, z, z, t27a, t28a, z, z, t31a,
    )
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idct64_ortak_son(e: [__m128i; 32], ta: [__m128i; 32]) -> [__m128i; 64] {
    let t32 = _mm_adds_epi16(ta[0], ta[1]);
    let t33 = _mm_subs_epi16(ta[0], ta[1]);
    let t34 = _mm_subs_epi16(ta[3], ta[2]);
    let t35 = _mm_adds_epi16(ta[3], ta[2]);
    let t36 = _mm_adds_epi16(ta[4], ta[5]);
    let t37 = _mm_subs_epi16(ta[4], ta[5]);
    let t38 = _mm_subs_epi16(ta[7], ta[6]);
    let t39 = _mm_adds_epi16(ta[7], ta[6]);
    let t40 = _mm_adds_epi16(ta[8], ta[9]);
    let t41 = _mm_subs_epi16(ta[8], ta[9]);
    let t42 = _mm_subs_epi16(ta[11], ta[10]);
    let t43 = _mm_adds_epi16(ta[11], ta[10]);
    let t44 = _mm_adds_epi16(ta[12], ta[13]);
    let t45 = _mm_subs_epi16(ta[12], ta[13]);
    let t46 = _mm_subs_epi16(ta[15], ta[14]);
    let t47 = _mm_adds_epi16(ta[15], ta[14]);
    let t48 = _mm_adds_epi16(ta[16], ta[17]);
    let t49 = _mm_subs_epi16(ta[16], ta[17]);
    let t50 = _mm_subs_epi16(ta[19], ta[18]);
    let t51 = _mm_adds_epi16(ta[19], ta[18]);
    let t52 = _mm_adds_epi16(ta[20], ta[21]);
    let t53 = _mm_subs_epi16(ta[20], ta[21]);
    let t54 = _mm_subs_epi16(ta[23], ta[22]);
    let t55 = _mm_adds_epi16(ta[23], ta[22]);
    let t56 = _mm_adds_epi16(ta[24], ta[25]);
    let t57 = _mm_subs_epi16(ta[24], ta[25]);
    let t58 = _mm_subs_epi16(ta[27], ta[26]);
    let t59 = _mm_adds_epi16(ta[27], ta[26]);
    let t60 = _mm_adds_epi16(ta[28], ta[29]);
    let t61 = _mm_subs_epi16(ta[28], ta[29]);
    let t62 = _mm_subs_epi16(ta[31], ta[30]);
    let t63 = _mm_adds_epi16(ta[31], ta[30]);

    let (t33a, t62a) = mul2(t33, t62, -4076, 401, 401, 4076);
    let (t34a, t61a) = mul2(t34, t61, -401, -4076, -4076, 401);
    let (t37a, t58a) = mul2(t37, t58, -2598, 3166, 3166, 2598);
    let (t38a, t57a) = mul2(t38, t57, -3166, -2598, -2598, 3166);
    let (t41a, t54a) = mul2(t41, t54, -3612, 1931, 1931, 3612);
    let (t42a, t53a) = mul2(t42, t53, -1931, -3612, -3612, 1931);
    let (t45a, t50a) = mul2(t45, t50, -1189, 3920, 3920, 1189);
    let (t46a, t49a) = mul2(t46, t49, -3920, -1189, -1189, 3920);

    let t32a = _mm_adds_epi16(t32, t35);
    let t33 = _mm_adds_epi16(t33a, t34a);
    let t34 = _mm_subs_epi16(t33a, t34a);
    let t35a = _mm_subs_epi16(t32, t35);
    let t36a = _mm_subs_epi16(t39, t36);
    let t37 = _mm_subs_epi16(t38a, t37a);
    let t38 = _mm_adds_epi16(t38a, t37a);
    let t39a = _mm_adds_epi16(t39, t36);
    let t40a = _mm_adds_epi16(t40, t43);
    let t41 = _mm_adds_epi16(t41a, t42a);
    let t42 = _mm_subs_epi16(t41a, t42a);
    let t43a = _mm_subs_epi16(t40, t43);
    let t44a = _mm_subs_epi16(t47, t44);
    let t45 = _mm_subs_epi16(t46a, t45a);
    let t46 = _mm_adds_epi16(t46a, t45a);
    let t47a = _mm_adds_epi16(t47, t44);
    let t48a = _mm_adds_epi16(t48, t51);
    let t49 = _mm_adds_epi16(t49a, t50a);
    let t50 = _mm_subs_epi16(t49a, t50a);
    let t51a = _mm_subs_epi16(t48, t51);
    let t52a = _mm_subs_epi16(t55, t52);
    let t53 = _mm_subs_epi16(t54a, t53a);
    let t54 = _mm_adds_epi16(t54a, t53a);
    let t55a = _mm_adds_epi16(t55, t52);
    let t56a = _mm_adds_epi16(t56, t59);
    let t57 = _mm_adds_epi16(t57a, t58a);
    let t58 = _mm_subs_epi16(t57a, t58a);
    let t59a = _mm_subs_epi16(t56, t59);
    let t60a = _mm_subs_epi16(t63, t60);
    let t61 = _mm_subs_epi16(t62a, t61a);
    let t62 = _mm_adds_epi16(t62a, t61a);
    let t63a = _mm_adds_epi16(t63, t60);

    let (t34a, t61a) = mul2(t34, t61, -4017, 799, 799, 4017);
    let (t35, t60) = mul2(t35a, t60a, -4017, 799, 799, 4017);
    let (t36, t59) = mul2(t36a, t59a, -799, -4017, -4017, 799);
    let (t37a, t58a) = mul2(t37, t58, -799, -4017, -4017, 799);
    let (t42a, t53a) = mul2(t42, t53, -2276, 3406, 3406, 2276);
    let (t43, t52) = mul2(t43a, t52a, -2276, 3406, 3406, 2276);
    let (t44, t51) = mul2(t44a, t51a, -3406, -2276, -2276, 3406);
    let (t45a, t50a) = mul2(t45, t50, -3406, -2276, -2276, 3406);

    let t32 = _mm_adds_epi16(t32a, t39a);
    let t33a = _mm_adds_epi16(t33, t38);
    let t34 = _mm_adds_epi16(t34a, t37a);
    let t35a = _mm_adds_epi16(t35, t36);
    let t36a = _mm_subs_epi16(t35, t36);
    let t37 = _mm_subs_epi16(t34a, t37a);
    let t38a = _mm_subs_epi16(t33, t38);
    let t39 = _mm_subs_epi16(t32a, t39a);
    let t40 = _mm_subs_epi16(t47a, t40a);
    let t41a = _mm_subs_epi16(t46, t41);
    let t42 = _mm_subs_epi16(t45a, t42a);
    let t43a = _mm_subs_epi16(t44, t43);
    let t44a = _mm_adds_epi16(t44, t43);
    let t45 = _mm_adds_epi16(t45a, t42a);
    let t46a = _mm_adds_epi16(t46, t41);
    let t47 = _mm_adds_epi16(t47a, t40a);
    let t48 = _mm_adds_epi16(t48a, t55a);
    let t49a = _mm_adds_epi16(t49, t54);
    let t50 = _mm_adds_epi16(t50a, t53a);
    let t51a = _mm_adds_epi16(t51, t52);
    let t52a = _mm_subs_epi16(t51, t52);
    let t53 = _mm_subs_epi16(t50a, t53a);
    let t54a = _mm_subs_epi16(t49, t54);
    let t55 = _mm_subs_epi16(t48a, t55a);
    let t56 = _mm_subs_epi16(t63a, t56a);
    let t57a = _mm_subs_epi16(t62, t57);
    let t58 = _mm_subs_epi16(t61a, t58a);
    let t59a = _mm_subs_epi16(t60, t59);
    let t60a = _mm_adds_epi16(t60, t59);
    let t61 = _mm_adds_epi16(t61a, t58a);
    let t62a = _mm_adds_epi16(t62, t57);
    let t63 = _mm_adds_epi16(t63a, t56a);

    let (t36, t59) = mul2(t36a, t59a, -3784, 1567, 1567, 3784);
    let (t37a, t58a) = mul2(t37, t58, -3784, 1567, 1567, 3784);
    let (t38, t57) = mul2(t38a, t57a, -3784, 1567, 1567, 3784);
    let (t39a, t56a) = mul2(t39, t56, -3784, 1567, 1567, 3784);
    let (t40a, t55a) = mul2(t40, t55, -1567, -3784, -3784, 1567);
    let (t41, t54) = mul2(t41a, t54a, -1567, -3784, -3784, 1567);
    let (t42a, t53a) = mul2(t42, t53, -1567, -3784, -3784, 1567);
    let (t43, t52) = mul2(t43a, t52a, -1567, -3784, -3784, 1567);

    let t32a = _mm_adds_epi16(t32, t47);
    let t33 = _mm_adds_epi16(t33a, t46a);
    let t34a = _mm_adds_epi16(t34, t45);
    let t35 = _mm_adds_epi16(t35a, t44a);
    let t36a = _mm_adds_epi16(t36, t43);
    let t37 = _mm_adds_epi16(t37a, t42a);
    let t38a = _mm_adds_epi16(t38, t41);
    let t39 = _mm_adds_epi16(t39a, t40a);
    let t40 = _mm_subs_epi16(t39a, t40a);
    let t41a = _mm_subs_epi16(t38, t41);
    let t42 = _mm_subs_epi16(t37a, t42a);
    let t43a = _mm_subs_epi16(t36, t43);
    let t44 = _mm_subs_epi16(t35a, t44a);
    let t45a = _mm_subs_epi16(t34, t45);
    let t46 = _mm_subs_epi16(t33a, t46a);
    let t47a = _mm_subs_epi16(t32, t47);
    let t48a = _mm_subs_epi16(t63, t48);
    let t49 = _mm_subs_epi16(t62a, t49a);
    let t50a = _mm_subs_epi16(t61, t50);
    let t51 = _mm_subs_epi16(t60a, t51a);
    let t52a = _mm_subs_epi16(t59, t52);
    let t53 = _mm_subs_epi16(t58a, t53a);
    let t54a = _mm_subs_epi16(t57, t54);
    let t55 = _mm_subs_epi16(t56a, t55a);
    let t56 = _mm_adds_epi16(t56a, t55a);
    let t57a = _mm_adds_epi16(t57, t54);
    let t58 = _mm_adds_epi16(t58a, t53a);
    let t59a = _mm_adds_epi16(t59, t52);
    let t60 = _mm_adds_epi16(t60a, t51a);
    let t61a = _mm_adds_epi16(t61, t50);
    let t62 = _mm_adds_epi16(t62a, t49a);
    let t63a = _mm_adds_epi16(t63, t48);

    let (t40a, t55a) = mul2(t55, t40, 2896, -2896, 2896, 2896);
    let (t41, t54) = mul2(t54a, t41a, 2896, -2896, 2896, 2896);
    let (t42a, t53a) = mul2(t53, t42, 2896, -2896, 2896, 2896);
    let (t43, t52) = mul2(t52a, t43a, 2896, -2896, 2896, 2896);
    let (t44a, t51a) = mul2(t51, t44, 2896, -2896, 2896, 2896);
    let (t45, t50) = mul2(t50a, t45a, 2896, -2896, 2896, 2896);
    let (t46a, t49a) = mul2(t49, t46, 2896, -2896, 2896, 2896);
    let (t47, t48) = mul2(t48a, t47a, 2896, -2896, 2896, 2896);

    let odd = [
        t32a, t33, t34a, t35, t36a, t37, t38a, t39, t40a, t41, t42a, t43, t44a, t45, t46a, t47,
        t48, t49a, t50, t51a, t52, t53a, t54, t55a, t56, t57a, t58, t59a, t60, t61a, t62, t63a,
    ];
    let mut out = [_mm_setzero_si128(); 64];
    let mut i = 0;
    while i < 32 {
        out[i] = _mm_adds_epi16(e[i], odd[31 - i]);
        out[63 - i] = _mm_subs_epi16(e[i], odd[31 - i]);
        i += 1;
    }
    out
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idct64_tam(c: &[__m128i; 32]) -> [__m128i; 64] {
    let e = idct32_yarim(&[
        c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14], c[16], c[18], c[20], c[22], c[24],
        c[26], c[28], c[30],
    ]);
    let ta = [
        _mm_mulhrs_epi16(c[1], _mm_set1_epi16(101 * 8)),
        _mm_mulhrs_epi16(c[31], _mm_set1_epi16(-2824 * 8)),
        _mm_mulhrs_epi16(c[17], _mm_set1_epi16(1660 * 8)),
        _mm_mulhrs_epi16(c[15], _mm_set1_epi16(-1474 * 8)),
        _mm_mulhrs_epi16(c[9], _mm_set1_epi16(897 * 8)),
        _mm_mulhrs_epi16(c[23], _mm_set1_epi16(-2191 * 8)),
        _mm_mulhrs_epi16(c[25], _mm_set1_epi16(2359 * 8)),
        _mm_mulhrs_epi16(c[7], _mm_set1_epi16(-700 * 8)),
        _mm_mulhrs_epi16(c[5], _mm_set1_epi16(501 * 8)),
        _mm_mulhrs_epi16(c[27], _mm_set1_epi16(-2520 * 8)),
        _mm_mulhrs_epi16(c[21], _mm_set1_epi16(2019 * 8)),
        _mm_mulhrs_epi16(c[11], _mm_set1_epi16(-1092 * 8)),
        _mm_mulhrs_epi16(c[13], _mm_set1_epi16(1285 * 8)),
        _mm_mulhrs_epi16(c[19], _mm_set1_epi16(-1842 * 8)),
        _mm_mulhrs_epi16(c[29], _mm_set1_epi16(2675 * 8)),
        _mm_mulhrs_epi16(c[3], _mm_set1_epi16(-301 * 8)),
        _mm_mulhrs_epi16(c[3], _mm_set1_epi16(4085 * 8)),
        _mm_mulhrs_epi16(c[29], _mm_set1_epi16(3102 * 8)),
        _mm_mulhrs_epi16(c[19], _mm_set1_epi16(3659 * 8)),
        _mm_mulhrs_epi16(c[13], _mm_set1_epi16(3889 * 8)),
        _mm_mulhrs_epi16(c[11], _mm_set1_epi16(3948 * 8)),
        _mm_mulhrs_epi16(c[21], _mm_set1_epi16(3564 * 8)),
        _mm_mulhrs_epi16(c[27], _mm_set1_epi16(3229 * 8)),
        _mm_mulhrs_epi16(c[5], _mm_set1_epi16(4065 * 8)),
        _mm_mulhrs_epi16(c[7], _mm_set1_epi16(4036 * 8)),
        _mm_mulhrs_epi16(c[25], _mm_set1_epi16(3349 * 8)),
        _mm_mulhrs_epi16(c[23], _mm_set1_epi16(3461 * 8)),
        _mm_mulhrs_epi16(c[9], _mm_set1_epi16(3996 * 8)),
        _mm_mulhrs_epi16(c[15], _mm_set1_epi16(3822 * 8)),
        _mm_mulhrs_epi16(c[17], _mm_set1_epi16(3745 * 8)),
        _mm_mulhrs_epi16(c[31], _mm_set1_epi16(2967 * 8)),
        _mm_mulhrs_epi16(c[1], _mm_set1_epi16(4095 * 8)),
    ];
    idct64_ortak_son(e, ta)
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idct64_yarim(c: &[__m128i; 16]) -> [__m128i; 64] {
    let e = idct32_ceyrek(&[c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]]);
    let z = _mm_setzero_si128();
    let ta = [
        _mm_mulhrs_epi16(c[1], _mm_set1_epi16(101 * 8)),
        z,
        z,
        _mm_mulhrs_epi16(c[15], _mm_set1_epi16(-1474 * 8)),
        _mm_mulhrs_epi16(c[9], _mm_set1_epi16(897 * 8)),
        z,
        z,
        _mm_mulhrs_epi16(c[7], _mm_set1_epi16(-700 * 8)),
        _mm_mulhrs_epi16(c[5], _mm_set1_epi16(501 * 8)),
        z,
        z,
        _mm_mulhrs_epi16(c[11], _mm_set1_epi16(-1092 * 8)),
        _mm_mulhrs_epi16(c[13], _mm_set1_epi16(1285 * 8)),
        z,
        z,
        _mm_mulhrs_epi16(c[3], _mm_set1_epi16(-301 * 8)),
        _mm_mulhrs_epi16(c[3], _mm_set1_epi16(4085 * 8)),
        z,
        z,
        _mm_mulhrs_epi16(c[13], _mm_set1_epi16(3889 * 8)),
        _mm_mulhrs_epi16(c[11], _mm_set1_epi16(3948 * 8)),
        z,
        z,
        _mm_mulhrs_epi16(c[5], _mm_set1_epi16(4065 * 8)),
        _mm_mulhrs_epi16(c[7], _mm_set1_epi16(4036 * 8)),
        z,
        z,
        _mm_mulhrs_epi16(c[9], _mm_set1_epi16(3996 * 8)),
        _mm_mulhrs_epi16(c[15], _mm_set1_epi16(3822 * 8)),
        z,
        z,
        _mm_mulhrs_epi16(c[1], _mm_set1_epi16(4095 * 8)),
    ];
    idct64_ortak_son(e, ta)
}

#[target_feature(enable = "ssse3")]
#[inline]
fn kaydir_dizi<const N: usize>(mut a: [__m128i; N], mul: i16) -> [__m128i; N] {
    let m = _mm_set1_epi16(mul);
    let mut i = 0;
    while i < N {
        a[i] = _mm_mulhrs_epi16(a[i], m);
        i += 1;
    }
    a
}

// ---------------------------------------------------------------------------
// `IDTX` (`Identity × Identity`) blok gövdeleri (5 kayıt)
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
#[inline]
fn idtx_8x32(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let y_blok = if eob <= 42 {
        1
    } else if eob <= 106 {
        2
    } else {
        4
    };
    let m_p1 = _mm_set1_epi16(16384);
    let m_p2 = _mm_set1_epi16(8192);
    let mut yb = 0;
    while yb < y_blok {
        let y0 = yb * 8;
        let mut giris = [_mm_setzero_si128(); 8];
        let mut x = 0;
        while x < 8 {
            giris[x] = _mm_mulhrs_epi16(yukle_katsayi8(coeff, y0 + x * 32), m_p1);
            x += 1;
        }
        let t = devrik_8x8(giris);
        let mut dy = 0;
        while dy < 8 {
            let r0 = _mm_mulhrs_epi16(t[dy], m_p2);
            let r1 = _mm_mulhrs_epi16(t[dy + 1], m_p2);
            ekle_yaz_8x2_konum(dst, stride, 0, y0 + dy, r0, r1);
            dy += 2;
        }
        yb += 1;
    }
    chunk_mut::<i32, 256>(coeff, 0).fill(0);
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idtx_32x8(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let x_blok = if eob <= 42 {
        1
    } else if eob <= 106 {
        2
    } else {
        4
    };
    let m_p2 = _mm_set1_epi16(4096);
    let mut xb = 0;
    while xb < x_blok {
        let x0 = xb * 8;
        let mut giris = [_mm_setzero_si128(); 8];
        let mut dx = 0;
        while dx < 8 {
            giris[dx] = yukle_katsayi8(coeff, (x0 + dx) * 8);
            dx += 1;
        }
        let t = devrik_8x8(giris);
        let mut y = 0;
        while y < 8 {
            let r0 = _mm_mulhrs_epi16(t[y], m_p2);
            let r1 = _mm_mulhrs_epi16(t[y + 1], m_p2);
            ekle_yaz_8x2_konum(dst, stride, x0, y, r0, r1);
            y += 2;
        }
        xb += 1;
    }
    chunk_mut::<i32, 256>(coeff, 0).fill(0);
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idtx_16x32(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let y_blok = if eob <= 43 {
        1
    } else if eob <= 150 {
        2
    } else if eob <= 278 {
        3
    } else {
        4
    };
    let x_blok = if eob <= 35 { 1 } else { 2 };
    let k16 = _mm_set1_epi16(1697 * 16);
    let m_yarim = _mm_set1_epi16(16384);
    let m_p2 = _mm_set1_epi16(8192);

    let mut xb = 0;
    while xb < x_blok {
        let x0 = xb * 8;
        let mut yb = 0;
        while yb < y_blok {
            let y0 = yb * 8;
            let mut giris = [_mm_setzero_si128(); 8];
            let mut dx = 0;
            while dx < 8 {
                let v = yukle_katsayi8_opt::<true>(coeff, y0 + (x0 + dx) * 32);
                let t16 = _mm_mulhrs_epi16(_mm_mulhrs_epi16(v, k16), m_yarim);
                giris[dx] = _mm_adds_epi16(v, t16);
                dx += 1;
            }
            let t = devrik_8x8(giris);
            let mut dy = 0;
            while dy < 8 {
                let r0 = _mm_mulhrs_epi16(t[dy], m_p2);
                let r1 = _mm_mulhrs_epi16(t[dy + 1], m_p2);
                ekle_yaz_8x2_konum(dst, stride, x0, y0 + dy, r0, r1);
                dy += 2;
            }
            yb += 1;
        }
        xb += 1;
    }
    chunk_mut::<i32, 512>(coeff, 0).fill(0);
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idtx_32x16(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let x_blok = if eob <= 43 {
        1
    } else if eob <= 150 {
        2
    } else if eob <= 278 {
        3
    } else {
        4
    };
    let y_blok = if eob <= 35 { 1 } else { 2 };
    let k16 = _mm_set1_epi16(1697 * 16);
    let iki_bir = _mm_set1_epi32(cift_katsayi(2, 1));
    let sekiz = _mm_set1_epi32(8);

    let mut yb = 0;
    while yb < y_blok {
        let y0 = yb * 8;
        let mut xb = 0;
        while xb < x_blok {
            let x0 = xb * 8;
            let mut giris = [_mm_setzero_si128(); 8];
            let mut dx = 0;
            while dx < 8 {
                let v = yukle_katsayi8_opt::<true>(coeff, y0 + (x0 + dx) * 16);
                giris[dx] = _mm_adds_epi16(v, v);
                dx += 1;
            }
            let t = devrik_8x8(giris);
            let mut dy = 0;
            while dy < 8 {
                let r0 = {
                    let u = _mm_mulhrs_epi16(t[dy], k16);
                    let lo = _mm_unpacklo_epi16(t[dy], u);
                    let hi = _mm_unpackhi_epi16(t[dy], u);
                    let s_lo =
                        _mm_srai_epi32::<4>(_mm_add_epi32(_mm_madd_epi16(lo, iki_bir), sekiz));
                    let s_hi =
                        _mm_srai_epi32::<4>(_mm_add_epi32(_mm_madd_epi16(hi, iki_bir), sekiz));
                    _mm_packs_epi32(s_lo, s_hi)
                };
                let r1 = {
                    let u = _mm_mulhrs_epi16(t[dy + 1], k16);
                    let lo = _mm_unpacklo_epi16(t[dy + 1], u);
                    let hi = _mm_unpackhi_epi16(t[dy + 1], u);
                    let s_lo =
                        _mm_srai_epi32::<4>(_mm_add_epi32(_mm_madd_epi16(lo, iki_bir), sekiz));
                    let s_hi =
                        _mm_srai_epi32::<4>(_mm_add_epi32(_mm_madd_epi16(hi, iki_bir), sekiz));
                    _mm_packs_epi32(s_lo, s_hi)
                };
                ekle_yaz_8x2_konum(dst, stride, x0, y0 + dy, r0, r1);
                dy += 2;
            }
            xb += 1;
        }
        yb += 1;
    }
    chunk_mut::<i32, 512>(coeff, 0).fill(0);
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idtx_32x32(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let blok = if eob <= 35 {
        1
    } else if eob <= 135 {
        2
    } else {
        4
    };
    let m_p2 = _mm_set1_epi16(8192);
    let mut yb = 0;
    while yb < blok {
        let y0 = yb * 8;
        let mut xb = 0;
        while xb < blok {
            let x0 = xb * 8;
            let mut giris = [_mm_setzero_si128(); 8];
            let mut dx = 0;
            while dx < 8 {
                giris[dx] = yukle_katsayi8(coeff, y0 + (x0 + dx) * 32);
                dx += 1;
            }
            let t = devrik_8x8(giris);
            let mut dy = 0;
            while dy < 8 {
                let r0 = _mm_mulhrs_epi16(t[dy], m_p2);
                let r1 = _mm_mulhrs_epi16(t[dy + 1], m_p2);
                ekle_yaz_8x2_konum(dst, stride, x0, y0 + dy, r0, r1);
                dy += 2;
            }
            xb += 1;
        }
        yb += 1;
    }
    chunk_mut::<i32, 1024>(coeff, 0).fill(0);
}

// ---------------------------------------------------------------------------
// `DCT_DCT` büyük blok gövdeleri (10 kayıt)
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
#[inline]
fn dct_8x32(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<8, 32, 2, false>(dst, stride, coeff);
        return;
    }
    let y_blok = if eob <= 42 {
        1
    } else if eob <= 106 {
        2
    } else {
        4
    };
    let mut sutun = [_mm_setzero_si128(); 32];
    let mut yb = 0;
    while yb < y_blok {
        let y0 = yb * 8;
        let mut giris = [_mm_setzero_si128(); 8];
        let mut x = 0;
        while x < 8 {
            giris[x] = yukle_katsayi8(coeff, y0 + x * 32);
            x += 1;
        }
        let p1 = devrik_8x8(kaydir_dizi(idct8_tam(giris), 8192));
        let mut dy = 0;
        while dy < 8 {
            sutun[y0 + dy] = p1[dy];
            dy += 1;
        }
        yb += 1;
    }
    chunk_mut::<i32, 256>(coeff, 0).fill(0);

    let cikis = if y_blok == 1 {
        let s8: [__m128i; 8] = core::array::from_fn(|i| sutun[i]);
        kaydir_dizi(idct32_ceyrek(&s8), 2048)
    } else if y_blok == 2 {
        let s16: [__m128i; 16] = core::array::from_fn(|i| sutun[i]);
        kaydir_dizi(idct32_yarim(&s16), 2048)
    } else {
        kaydir_dizi(idct32_tam(&sutun), 2048)
    };

    let mut y = 0;
    while y < 32 {
        ekle_yaz_8x2_konum(dst, stride, 0, y, cikis[y], cikis[y + 1]);
        y += 2;
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn dct_32x8(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<32, 8, 2, false>(dst, stride, coeff);
        return;
    }
    let p1 = if eob <= 42 {
        let g8: [__m128i; 8] = core::array::from_fn(|x| yukle_katsayi8(coeff, x * 8));
        kaydir_dizi(idct32_ceyrek(&g8), 8192)
    } else if eob <= 106 {
        let g16: [__m128i; 16] = core::array::from_fn(|x| yukle_katsayi8(coeff, x * 8));
        kaydir_dizi(idct32_yarim(&g16), 8192)
    } else {
        let g32: [__m128i; 32] = core::array::from_fn(|x| yukle_katsayi8(coeff, x * 8));
        kaydir_dizi(idct32_tam(&g32), 8192)
    };
    chunk_mut::<i32, 256>(coeff, 0).fill(0);

    let mut c_blok = [[_mm_setzero_si128(); 8]; 4];
    let mut xb = 0;
    while xb < 4 {
        let x0 = xb * 8;
        let t = devrik_8x8([
            p1[x0],
            p1[x0 + 1],
            p1[x0 + 2],
            p1[x0 + 3],
            p1[x0 + 4],
            p1[x0 + 5],
            p1[x0 + 6],
            p1[x0 + 7],
        ]);
        c_blok[xb] = kaydir_dizi(idct8_tam(t), 2048);
        xb += 1;
    }

    let mut y = 0;
    while y < 8 {
        let taban = y * stride;
        ekle_yaz_16x1(dst, taban, c_blok[0][y], c_blok[1][y]);
        ekle_yaz_16x1(dst, taban + 16, c_blok[2][y], c_blok[3][y]);
        y += 1;
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn dct_16x32(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<16, 32, 1, true>(dst, stride, coeff);
        return;
    }
    let ceyrek = eob <= 35;
    let yarim = eob <= 150;
    let y_blok = if ceyrek {
        1
    } else if yarim {
        2
    } else {
        4
    };

    let mut sutun = [[_mm_setzero_si128(); 32]; 2];
    let mut yb = 0;
    while yb < y_blok {
        let y0 = yb * 8;
        let p1 = if ceyrek {
            let g8: [__m128i; 8] =
                core::array::from_fn(|x| yukle_katsayi8_opt::<true>(coeff, y0 + x * 32));
            kaydir_dizi(idct16_yarim(g8), 16384)
        } else {
            let g16: [__m128i; 16] =
                core::array::from_fn(|x| yukle_katsayi8_opt::<true>(coeff, y0 + x * 32));
            kaydir_dizi(idct16_tam(g16), 16384)
        };
        let tl = devrik_8x8([p1[0], p1[1], p1[2], p1[3], p1[4], p1[5], p1[6], p1[7]]);
        let tr = devrik_8x8([p1[8], p1[9], p1[10], p1[11], p1[12], p1[13], p1[14], p1[15]]);
        let mut dy = 0;
        while dy < 8 {
            sutun[0][y0 + dy] = tl[dy];
            sutun[1][y0 + dy] = tr[dy];
            dy += 1;
        }
        yb += 1;
    }
    chunk_mut::<i32, 512>(coeff, 0).fill(0);

    let mut c_blok = [[_mm_setzero_si128(); 32]; 2];
    let mut xb = 0;
    while xb < 2 {
        c_blok[xb] = if ceyrek {
            let s8: [__m128i; 8] = core::array::from_fn(|i| sutun[xb][i]);
            kaydir_dizi(idct32_ceyrek(&s8), 2048)
        } else if yarim {
            let s16: [__m128i; 16] = core::array::from_fn(|i| sutun[xb][i]);
            kaydir_dizi(idct32_yarim(&s16), 2048)
        } else {
            kaydir_dizi(idct32_tam(&sutun[xb]), 2048)
        };
        xb += 1;
    }

    let mut y = 0;
    while y < 32 {
        ekle_yaz_16x1(dst, y * stride, c_blok[0][y], c_blok[1][y]);
        y += 1;
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn dct_32x16(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<32, 16, 1, true>(dst, stride, coeff);
        return;
    }
    let ceyrek = eob <= 35;
    let yarim = eob <= 150;
    let y_blok = if ceyrek { 1 } else { 2 };

    let mut sutun = [[_mm_setzero_si128(); 16]; 4];
    let mut yb = 0;
    while yb < y_blok {
        let y0 = yb * 8;
        let p1 = if ceyrek {
            let g8: [__m128i; 8] =
                core::array::from_fn(|x| yukle_katsayi8_opt::<true>(coeff, y0 + x * 16));
            kaydir_dizi(idct32_ceyrek(&g8), 16384)
        } else if yarim {
            let g16: [__m128i; 16] =
                core::array::from_fn(|x| yukle_katsayi8_opt::<true>(coeff, y0 + x * 16));
            kaydir_dizi(idct32_yarim(&g16), 16384)
        } else {
            let g32: [__m128i; 32] =
                core::array::from_fn(|x| yukle_katsayi8_opt::<true>(coeff, y0 + x * 16));
            kaydir_dizi(idct32_tam(&g32), 16384)
        };
        let mut xb = 0;
        while xb < 4 {
            let x0 = xb * 8;
            let t = devrik_8x8([
                p1[x0],
                p1[x0 + 1],
                p1[x0 + 2],
                p1[x0 + 3],
                p1[x0 + 4],
                p1[x0 + 5],
                p1[x0 + 6],
                p1[x0 + 7],
            ]);
            let mut dy = 0;
            while dy < 8 {
                sutun[xb][y0 + dy] = t[dy];
                dy += 1;
            }
            xb += 1;
        }
        yb += 1;
    }
    chunk_mut::<i32, 512>(coeff, 0).fill(0);

    let mut c_blok = [[_mm_setzero_si128(); 16]; 4];
    let mut xb = 0;
    while xb < 4 {
        c_blok[xb] = if ceyrek {
            let s8: [__m128i; 8] = core::array::from_fn(|i| sutun[xb][i]);
            kaydir_dizi(idct16_yarim(s8), 2048)
        } else {
            kaydir_dizi(idct16_tam(sutun[xb]), 2048)
        };
        xb += 1;
    }

    let mut y = 0;
    while y < 16 {
        let taban = y * stride;
        ekle_yaz_16x1(dst, taban, c_blok[0][y], c_blok[1][y]);
        ekle_yaz_16x1(dst, taban + 16, c_blok[2][y], c_blok[3][y]);
        y += 1;
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn dct_32x32(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<32, 32, 2, false>(dst, stride, coeff);
        return;
    }
    let ceyrek = eob <= 35;
    let yarim = eob <= 135;
    let y_blok = if ceyrek {
        1
    } else if yarim {
        2
    } else {
        4
    };

    let mut sutun = [[_mm_setzero_si128(); 32]; 4];
    let mut yb = 0;
    while yb < y_blok {
        let y0 = yb * 8;
        let p1 = if ceyrek {
            let g8: [__m128i; 8] = core::array::from_fn(|x| yukle_katsayi8(coeff, y0 + x * 32));
            kaydir_dizi(idct32_ceyrek(&g8), 8192)
        } else if yarim {
            let g16: [__m128i; 16] = core::array::from_fn(|x| yukle_katsayi8(coeff, y0 + x * 32));
            kaydir_dizi(idct32_yarim(&g16), 8192)
        } else {
            let g32: [__m128i; 32] = core::array::from_fn(|x| yukle_katsayi8(coeff, y0 + x * 32));
            kaydir_dizi(idct32_tam(&g32), 8192)
        };
        let mut xb = 0;
        while xb < 4 {
            let x0 = xb * 8;
            let t = devrik_8x8([
                p1[x0],
                p1[x0 + 1],
                p1[x0 + 2],
                p1[x0 + 3],
                p1[x0 + 4],
                p1[x0 + 5],
                p1[x0 + 6],
                p1[x0 + 7],
            ]);
            let mut dy = 0;
            while dy < 8 {
                sutun[xb][y0 + dy] = t[dy];
                dy += 1;
            }
            xb += 1;
        }
        yb += 1;
    }
    chunk_mut::<i32, 1024>(coeff, 0).fill(0);

    let mut c_sol = [[_mm_setzero_si128(); 32]; 2];
    let mut g = 0;
    while g < 2 {
        let xb = g * 2;
        let mut k = 0;
        while k < 2 {
            c_sol[k] = if ceyrek {
                let s8: [__m128i; 8] = core::array::from_fn(|i| sutun[xb + k][i]);
                kaydir_dizi(idct32_ceyrek(&s8), 2048)
            } else if yarim {
                let s16: [__m128i; 16] = core::array::from_fn(|i| sutun[xb + k][i]);
                kaydir_dizi(idct32_yarim(&s16), 2048)
            } else {
                kaydir_dizi(idct32_tam(&sutun[xb + k]), 2048)
            };
            k += 1;
        }
        let x_off = xb * 8;
        let mut y = 0;
        while y < 32 {
            ekle_yaz_16x1(dst, y * stride + x_off, c_sol[0][y], c_sol[1][y]);
            y += 1;
        }
        g += 1;
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn dct_16x64(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<16, 64, 2, false>(dst, stride, coeff);
        return;
    }
    let ceyrek = eob <= 35;
    let yarim = eob <= 150;
    let y_blok = if ceyrek {
        1
    } else if yarim {
        2
    } else {
        4
    };

    let mut sutun = [[_mm_setzero_si128(); 32]; 2];
    let mut yb = 0;
    while yb < y_blok {
        let y0 = yb * 8;
        let p1 = if ceyrek {
            let g8: [__m128i; 8] = core::array::from_fn(|x| yukle_katsayi8(coeff, y0 + x * 32));
            kaydir_dizi(idct16_yarim(g8), 8192)
        } else {
            let g16: [__m128i; 16] = core::array::from_fn(|x| yukle_katsayi8(coeff, y0 + x * 32));
            kaydir_dizi(idct16_tam(g16), 8192)
        };
        let tl = devrik_8x8([p1[0], p1[1], p1[2], p1[3], p1[4], p1[5], p1[6], p1[7]]);
        let tr = devrik_8x8([p1[8], p1[9], p1[10], p1[11], p1[12], p1[13], p1[14], p1[15]]);
        let mut dy = 0;
        while dy < 8 {
            sutun[0][y0 + dy] = tl[dy];
            sutun[1][y0 + dy] = tr[dy];
            dy += 1;
        }
        yb += 1;
    }
    chunk_mut::<i32, 512>(coeff, 0).fill(0);

    let c0 = if yarim {
        let s16: [__m128i; 16] = core::array::from_fn(|i| sutun[0][i]);
        kaydir_dizi(idct64_yarim(&s16), 2048)
    } else {
        kaydir_dizi(idct64_tam(&sutun[0]), 2048)
    };
    let c1 = if yarim {
        let s16: [__m128i; 16] = core::array::from_fn(|i| sutun[1][i]);
        kaydir_dizi(idct64_yarim(&s16), 2048)
    } else {
        kaydir_dizi(idct64_tam(&sutun[1]), 2048)
    };

    let mut y = 0;
    while y < 64 {
        ekle_yaz_16x1(dst, y * stride, c0[y], c1[y]);
        y += 1;
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn dct_64x16(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<64, 16, 2, false>(dst, stride, coeff);
        return;
    }
    let ceyrek = eob <= 35;
    let yarim = eob <= 150;
    let y_blok = if ceyrek { 1 } else { 2 };

    let mut sutun = [[_mm_setzero_si128(); 16]; 8];
    let mut yb = 0;
    while yb < y_blok {
        let y0 = yb * 8;
        let p1 = if yarim {
            let g16: [__m128i; 16] = core::array::from_fn(|x| yukle_katsayi8(coeff, y0 + x * 16));
            kaydir_dizi(idct64_yarim(&g16), 8192)
        } else {
            let g32: [__m128i; 32] = core::array::from_fn(|x| yukle_katsayi8(coeff, y0 + x * 16));
            kaydir_dizi(idct64_tam(&g32), 8192)
        };
        let mut xb = 0;
        while xb < 8 {
            let x0 = xb * 8;
            let t = devrik_8x8([
                p1[x0],
                p1[x0 + 1],
                p1[x0 + 2],
                p1[x0 + 3],
                p1[x0 + 4],
                p1[x0 + 5],
                p1[x0 + 6],
                p1[x0 + 7],
            ]);
            let mut dy = 0;
            while dy < 8 {
                sutun[xb][y0 + dy] = t[dy];
                dy += 1;
            }
            xb += 1;
        }
        yb += 1;
    }
    chunk_mut::<i32, 512>(coeff, 0).fill(0);

    let mut g = 0;
    while g < 4 {
        let xb = g * 2;
        let c0 = if ceyrek {
            let s8: [__m128i; 8] = core::array::from_fn(|i| sutun[xb][i]);
            kaydir_dizi(idct16_yarim(s8), 2048)
        } else {
            kaydir_dizi(idct16_tam(sutun[xb]), 2048)
        };
        let c1 = if ceyrek {
            let s8: [__m128i; 8] = core::array::from_fn(|i| sutun[xb + 1][i]);
            kaydir_dizi(idct16_yarim(s8), 2048)
        } else {
            kaydir_dizi(idct16_tam(sutun[xb + 1]), 2048)
        };
        let x_off = xb * 8;
        let mut y = 0;
        while y < 16 {
            ekle_yaz_16x1(dst, y * stride + x_off, c0[y], c1[y]);
            y += 1;
        }
        g += 1;
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn dct_32x64(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<32, 64, 1, true>(dst, stride, coeff);
        return;
    }
    let ceyrek = eob <= 35;
    let yarim = eob <= 135;
    let y_blok = if ceyrek {
        1
    } else if yarim {
        2
    } else {
        4
    };

    let mut sutun = [[_mm_setzero_si128(); 32]; 4];
    let mut yb = 0;
    while yb < y_blok {
        let y0 = yb * 8;
        let p1 = if ceyrek {
            let g8: [__m128i; 8] =
                core::array::from_fn(|x| yukle_katsayi8_opt::<true>(coeff, y0 + x * 32));
            kaydir_dizi(idct32_ceyrek(&g8), 16384)
        } else if yarim {
            let g16: [__m128i; 16] =
                core::array::from_fn(|x| yukle_katsayi8_opt::<true>(coeff, y0 + x * 32));
            kaydir_dizi(idct32_yarim(&g16), 16384)
        } else {
            let g32: [__m128i; 32] =
                core::array::from_fn(|x| yukle_katsayi8_opt::<true>(coeff, y0 + x * 32));
            kaydir_dizi(idct32_tam(&g32), 16384)
        };
        let mut xb = 0;
        while xb < 4 {
            let x0 = xb * 8;
            let t = devrik_8x8([
                p1[x0],
                p1[x0 + 1],
                p1[x0 + 2],
                p1[x0 + 3],
                p1[x0 + 4],
                p1[x0 + 5],
                p1[x0 + 6],
                p1[x0 + 7],
            ]);
            let mut dy = 0;
            while dy < 8 {
                sutun[xb][y0 + dy] = t[dy];
                dy += 1;
            }
            xb += 1;
        }
        yb += 1;
    }
    chunk_mut::<i32, 1024>(coeff, 0).fill(0);

    let mut g = 0;
    while g < 2 {
        let xb = g * 2;
        let c0 = if yarim {
            let s16: [__m128i; 16] = core::array::from_fn(|i| sutun[xb][i]);
            kaydir_dizi(idct64_yarim(&s16), 2048)
        } else {
            kaydir_dizi(idct64_tam(&sutun[xb]), 2048)
        };
        let c1 = if yarim {
            let s16: [__m128i; 16] = core::array::from_fn(|i| sutun[xb + 1][i]);
            kaydir_dizi(idct64_yarim(&s16), 2048)
        } else {
            kaydir_dizi(idct64_tam(&sutun[xb + 1]), 2048)
        };
        let x_off = xb * 8;
        let mut y = 0;
        while y < 64 {
            ekle_yaz_16x1(dst, y * stride + x_off, c0[y], c1[y]);
            y += 1;
        }
        g += 1;
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn dct_64x32(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<64, 32, 1, true>(dst, stride, coeff);
        return;
    }
    let ceyrek = eob <= 35;
    let yarim = eob <= 135;
    let y_blok = if ceyrek {
        1
    } else if yarim {
        2
    } else {
        4
    };

    let mut sutun = [[_mm_setzero_si128(); 32]; 8];
    let mut yb = 0;
    while yb < y_blok {
        let y0 = yb * 8;
        let p1 = if yarim {
            let g16: [__m128i; 16] =
                core::array::from_fn(|x| yukle_katsayi8_opt::<true>(coeff, y0 + x * 32));
            kaydir_dizi(idct64_yarim(&g16), 16384)
        } else {
            let g32: [__m128i; 32] =
                core::array::from_fn(|x| yukle_katsayi8_opt::<true>(coeff, y0 + x * 32));
            kaydir_dizi(idct64_tam(&g32), 16384)
        };
        let mut xb = 0;
        while xb < 8 {
            let x0 = xb * 8;
            let t = devrik_8x8([
                p1[x0],
                p1[x0 + 1],
                p1[x0 + 2],
                p1[x0 + 3],
                p1[x0 + 4],
                p1[x0 + 5],
                p1[x0 + 6],
                p1[x0 + 7],
            ]);
            let mut dy = 0;
            while dy < 8 {
                sutun[xb][y0 + dy] = t[dy];
                dy += 1;
            }
            xb += 1;
        }
        yb += 1;
    }
    chunk_mut::<i32, 1024>(coeff, 0).fill(0);

    let mut g = 0;
    while g < 4 {
        let xb = g * 2;
        let c0 = if ceyrek {
            let s8: [__m128i; 8] = core::array::from_fn(|i| sutun[xb][i]);
            kaydir_dizi(idct32_ceyrek(&s8), 2048)
        } else if yarim {
            let s16: [__m128i; 16] = core::array::from_fn(|i| sutun[xb][i]);
            kaydir_dizi(idct32_yarim(&s16), 2048)
        } else {
            kaydir_dizi(idct32_tam(&sutun[xb]), 2048)
        };
        let c1 = if ceyrek {
            let s8: [__m128i; 8] = core::array::from_fn(|i| sutun[xb + 1][i]);
            kaydir_dizi(idct32_ceyrek(&s8), 2048)
        } else if yarim {
            let s16: [__m128i; 16] = core::array::from_fn(|i| sutun[xb + 1][i]);
            kaydir_dizi(idct32_yarim(&s16), 2048)
        } else {
            kaydir_dizi(idct32_tam(&sutun[xb + 1]), 2048)
        };
        let x_off = xb * 8;
        let mut y = 0;
        while y < 32 {
            ekle_yaz_16x1(dst, y * stride + x_off, c0[y], c1[y]);
            y += 1;
        }
        g += 1;
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn dct_64x64(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<64, 64, 2, false>(dst, stride, coeff);
        return;
    }
    let ceyrek = eob <= 35;
    let yarim = eob <= 135;
    let y_blok = if ceyrek {
        1
    } else if yarim {
        2
    } else {
        4
    };

    let mut sutun = [[_mm_setzero_si128(); 32]; 8];
    let mut yb = 0;
    while yb < y_blok {
        let y0 = yb * 8;
        let p1 = if yarim {
            let g16: [__m128i; 16] = core::array::from_fn(|x| yukle_katsayi8(coeff, y0 + x * 32));
            kaydir_dizi(idct64_yarim(&g16), 8192)
        } else {
            let g32: [__m128i; 32] = core::array::from_fn(|x| yukle_katsayi8(coeff, y0 + x * 32));
            kaydir_dizi(idct64_tam(&g32), 8192)
        };
        let mut xb = 0;
        while xb < 8 {
            let x0 = xb * 8;
            let t = devrik_8x8([
                p1[x0],
                p1[x0 + 1],
                p1[x0 + 2],
                p1[x0 + 3],
                p1[x0 + 4],
                p1[x0 + 5],
                p1[x0 + 6],
                p1[x0 + 7],
            ]);
            let mut dy = 0;
            while dy < 8 {
                sutun[xb][y0 + dy] = t[dy];
                dy += 1;
            }
            xb += 1;
        }
        yb += 1;
    }
    chunk_mut::<i32, 1024>(coeff, 0).fill(0);

    let mut g = 0;
    while g < 4 {
        let xb = g * 2;
        let c0 = if yarim {
            let s16: [__m128i; 16] = core::array::from_fn(|i| sutun[xb][i]);
            kaydir_dizi(idct64_yarim(&s16), 2048)
        } else {
            kaydir_dizi(idct64_tam(&sutun[xb]), 2048)
        };
        let c1 = if yarim {
            let s16: [__m128i; 16] = core::array::from_fn(|i| sutun[xb + 1][i]);
            kaydir_dizi(idct64_yarim(&s16), 2048)
        } else {
            kaydir_dizi(idct64_tam(&sutun[xb + 1]), 2048)
        };
        let x_off = xb * 8;
        let mut y = 0;
        while y < 64 {
            ekle_yaz_16x1(dst, y * stride + x_off, c0[y], c1[y]);
            y += 1;
        }
        g += 1;
    }
}

/// `(TX, TXTP)` çiftine özelleştirilmiş büyük (`max(w, h) ∈ {32, 64}`) 8 bpc SSSE3 çekirdek gövdesi.
#[target_feature(enable = "ssse3")]
#[inline(never)]
fn itx8_buyuk_ssse3_cekirdek<const TX: u8, const TXTP: u8>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
) {
    let tx = TxSize::from_index(TX as usize).expect("geçerli tx");
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    match (tx, txtp) {
        (TxSize::Rtx8x32, TxType::DctDct) => dct_8x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx8x32, TxType::Idtx) => idtx_8x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x8, TxType::DctDct) => dct_32x8(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x8, TxType::Idtx) => idtx_32x8(dst, dst_stride, coeff, eob),
        (TxSize::Rtx16x32, TxType::DctDct) => dct_16x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx16x32, TxType::Idtx) => idtx_16x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x16, TxType::DctDct) => dct_32x16(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x16, TxType::Idtx) => idtx_32x16(dst, dst_stride, coeff, eob),
        (TxSize::Tx32x32, TxType::DctDct) => dct_32x32(dst, dst_stride, coeff, eob),
        (TxSize::Tx32x32, TxType::Idtx) => idtx_32x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx16x64, TxType::DctDct) => dct_16x64(dst, dst_stride, coeff, eob),
        (TxSize::Rtx64x16, TxType::DctDct) => dct_64x16(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x64, TxType::DctDct) => dct_32x64(dst, dst_stride, coeff, eob),
        (TxSize::Rtx64x32, TxType::DctDct) => dct_64x32(dst, dst_stride, coeff, eob),
        (TxSize::Tx64x64, TxType::DctDct) => dct_64x64(dst, dst_stride, coeff, eob),
        _ => unreachable!(),
    }
}

/// `(TX, TXTP)` çiftinin sözleşme denetimli güvenli büyük 8 bpc SSSE3 yuva girişi.
fn itx8_buyuk_ssse3_giris<const TX: u8, const TXTP: u8>(
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`itx8_buyuk_ssse3_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır.
    // (2) sınır: `itx8_sozlesmesi` `dst`'nin `(h − 1)·dst_stride + w` baytı ve
    // `coeff`'un `1024` öğeyi kapsadığını garanti eder; bütün bellek erişimleri
    // sabit boyutlu `chunk`/`chunk_mut` dizileri üzerindendir. (3) hizalama/takma ad:
    // `dst` ve `coeff` ayrık `&mut` referanslardır, hizasız erişim kullanılır.
    unsafe {
        itx8_buyuk_ssse3_cekirdek::<TX, TXTP>(dst, dst_stride, coeff, eob);
    }
}

/// `max(w, h) ∈ {32, 64}` boyutlu 15 adet 8 bpc SSSE3 `itxfm_add[tx][txtp]` yuvası
/// (yalnız [`Ssse3Token`] ile).
#[must_use]
pub(crate) const fn itx8_buyuk_ssse3_fn(
    _token: Ssse3Token,
    tx: TxSize,
    txtp: TxType,
) -> Option<ItxFn> {
    match (tx, txtp) {
        (TxSize::Rtx8x32, TxType::DctDct) => {
            Some(itx8_buyuk_ssse3_giris::<{ TxSize::Rtx8x32 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx8x32, TxType::Idtx) => {
            Some(itx8_buyuk_ssse3_giris::<{ TxSize::Rtx8x32 as u8 }, { TxType::Idtx as u8 }>)
        }
        (TxSize::Rtx32x8, TxType::DctDct) => {
            Some(itx8_buyuk_ssse3_giris::<{ TxSize::Rtx32x8 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx32x8, TxType::Idtx) => {
            Some(itx8_buyuk_ssse3_giris::<{ TxSize::Rtx32x8 as u8 }, { TxType::Idtx as u8 }>)
        }
        (TxSize::Rtx16x32, TxType::DctDct) => {
            Some(itx8_buyuk_ssse3_giris::<{ TxSize::Rtx16x32 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx16x32, TxType::Idtx) => {
            Some(itx8_buyuk_ssse3_giris::<{ TxSize::Rtx16x32 as u8 }, { TxType::Idtx as u8 }>)
        }
        (TxSize::Rtx32x16, TxType::DctDct) => {
            Some(itx8_buyuk_ssse3_giris::<{ TxSize::Rtx32x16 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx32x16, TxType::Idtx) => {
            Some(itx8_buyuk_ssse3_giris::<{ TxSize::Rtx32x16 as u8 }, { TxType::Idtx as u8 }>)
        }
        (TxSize::Tx32x32, TxType::DctDct) => {
            Some(itx8_buyuk_ssse3_giris::<{ TxSize::Tx32x32 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Tx32x32, TxType::Idtx) => {
            Some(itx8_buyuk_ssse3_giris::<{ TxSize::Tx32x32 as u8 }, { TxType::Idtx as u8 }>)
        }
        (TxSize::Rtx16x64, TxType::DctDct) => {
            Some(itx8_buyuk_ssse3_giris::<{ TxSize::Rtx16x64 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx64x16, TxType::DctDct) => {
            Some(itx8_buyuk_ssse3_giris::<{ TxSize::Rtx64x16 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx32x64, TxType::DctDct) => {
            Some(itx8_buyuk_ssse3_giris::<{ TxSize::Rtx32x64 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx64x32, TxType::DctDct) => {
            Some(itx8_buyuk_ssse3_giris::<{ TxSize::Rtx64x32 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Tx64x64, TxType::DctDct) => {
            Some(itx8_buyuk_ssse3_giris::<{ TxSize::Tx64x64 as u8 }, { TxType::DctDct as u8 }>)
        }
        _ => None,
    }
}
