//! 8 bpc WHT-dışı ters dönüşüm (`w, h ≤ 16`, 140 kayıt) x86_64 SSSE3 çekirdekleri — P43 Adım 2.
//!
//! Bu modül dav1d'nin `src/x86/itx.h` ve `src/x86/itx_sse.asm` dosyalarındaki
//! `w, h ≤ 16` boyutlu **140** adet 8 bpc SSSE3 `itxfm_add` kaydını **saf Rust
//! intrinsics** ile gerçekler; dış `.asm`/`.S`, `asm!` ve `global_asm!` yoktur (CEK-02):
//!
//! - `TX_4X4`, `RTX_4X8`, `RTX_8X4`, `TX_8X8`, `RTX_4X16`, `RTX_16X4`, `RTX_8X16`,
//!   `RTX_16X8`: 16 WHT-dışı `TxType` türünün tamamı (`8 × 16 = 128`),
//! - `TX_16X16`: desteklenen 12 `TxType` (`DCT_DCT`, `ADST_DCT`, `DCT_ADST`,
//!   `ADST_ADST`, `FLIPADST_DCT`, `DCT_FLIPADST`, `FLIPADST_FLIPADST`,
//!   `ADST_FLIPADST`, `FLIPADST_ADST`, `IDTX`, `V_DCT`, `H_DCT`).
//!
//! ## Sözleşme ve ISA saflığı (P34 / P43)
//!
//! - Girişte O(1) sözleşme denetimi ([`itx8_sozlesmesi`]) ile `(tx, txtp, bit_depth)`
//!   uyumu aranır; sözleşme dışı çağrıda paniklemeden [`itx_skaler_devret`]
//!   üzerinden sayaç artırılır ve skaler monomorfa devredilir.
//! - Katsayı sıfırlama sözleşmesi:
//!   - `txtp == TxType::DctDct && eob == 0` (DC-only): **yalnız `coeff[0] = 0`**
//!     sıfırlanır; `coeff[1..1024]` dokunulmadan kalır.
//!   - Genel yol (`eob ≥ 1` ya da `txtp != TxType::DctDct`): `coeff[..w·h]`
//!     tamamen sıfırlanır; `coeff[w·h..1024]` kuyruğuna dokunulmaz.
//! - Bütün çekirdek ve yardımcılar yalnız `#[target_feature(enable = "ssse3")]`
//!   taşır; başka `dsp/x86/*` çekirdek modülleriyle `#[target_feature]` yardımcı
//!   paylaşmaz (P34 modül yalıtımı) ve SSE4.1+ ya da VEX/EVEX yönergesi üretmez.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_adds_epi16, _mm_avg_epu16, _mm_cvtsi32_si128, _mm_cvtsi64_si128,
    _mm_madd_epi16, _mm_mulhrs_epi16, _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi16,
    _mm_set1_epi32, _mm_setr_epi32, _mm_setzero_si128, _mm_srai_epi16, _mm_srai_epi32,
    _mm_srli_si128, _mm_subs_epi16, _mm_unpackhi_epi8, _mm_unpackhi_epi16, _mm_unpackhi_epi32,
    _mm_unpackhi_epi64, _mm_unpacklo_epi8, _mm_unpacklo_epi16, _mm_unpacklo_epi32,
    _mm_unpacklo_epi64, _mm_xor_si128,
};

use super::mem::{chunk, chunk_mut, load_u8x16, store_u8x4, store_u8x8, store_u8x16};
use crate::cpu::Ssse3Token;
use crate::dsp::itx::{ItxFn, TxSize, TxType, itx_skaler_devret, itx8_sozlesmesi};
use crate::dsp::scalar::itx_1d::Tx1dType;
use crate::frame::BitDepth;

/// İki `i16` çarpanını `_mm_madd_epi16` için `i32` şeritlerine paketler (`[c0, c1, ...]`).
#[inline(always)]
const fn cift_katsayi(c0: i16, c1: i16) -> i32 {
    (c0 as u16 as i32) | ((c1 as u16 as i32) << 16)
}

/// 4 × `i32` katsayıyı alt 4 `i16` şeridine doyurmalı paketleyerek yükler.
#[target_feature(enable = "ssse3")]
#[inline]
fn yukle_katsayi4(coeff: &[i32], off: usize) -> __m128i {
    let c = chunk::<i32, 4>(coeff, off);
    let v = _mm_setr_epi32(c[0], c[1], c[2], c[3]);
    _mm_packs_epi32(v, v)
}

/// 8 × `i32` katsayıyı 8 `i16` şeridine doyurmalı paketleyerek yükler.
#[target_feature(enable = "ssse3")]
#[inline]
fn yukle_katsayi8(coeff: &[i32], off: usize) -> __m128i {
    let c = chunk::<i32, 8>(coeff, off);
    let lo = _mm_setr_epi32(c[0], c[1], c[2], c[3]);
    let hi = _mm_setr_epi32(c[4], c[5], c[6], c[7]);
    _mm_packs_epi32(lo, hi)
}

/// `rect2` ön ölçeklemesi: `(c · 181 + 128) >> 8 == pmulhrsw(c, 2896·8)`.
#[target_feature(enable = "ssse3")]
#[inline]
fn olcekle_rect2(v: __m128i) -> __m128i {
    _mm_mulhrs_epi16(v, _mm_set1_epi16(2896 * 8))
}

/// İşaretli `i16` yuvarlamalı ortalama: `(a + b + 1) >> 1` (bütün `i16` aralığında taşmasız).
#[target_feature(enable = "ssse3")]
#[inline]
fn savg_epi16(a: __m128i, b: __m128i) -> __m128i {
    let bias = _mm_set1_epi16(-32768);
    _mm_xor_si128(
        _mm_avg_epu16(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias)),
        bias,
    )
}

/// 4 etkin `i16` şeridi için ikili döndürme:
/// `((a·c0 + b·c1 + 2048) >> 12, (a·d0 + b·d1 + 2048) >> 12)`.
#[target_feature(enable = "ssse3")]
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

/// 8 `i16` şeridi için ikili döndürme:
/// `((a·c0 + b·c1 + 2048) >> 12, (a·d0 + b·d1 + 2048) >> 12)`.
#[target_feature(enable = "ssse3")]
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

// ---------------------------------------------------------------------------
// Hedef düzlem piksel ekleme + doyurmalı [0, 255] yazma yardımcıları
// ---------------------------------------------------------------------------

/// `4 × 4` bölgeye 4 şeritli `i16` artıkları ekler ve yazar (`0` taşma).
#[target_feature(enable = "ssse3")]
#[inline]
fn ekle_yaz_4x4(
    dst: &mut [u8],
    stride: usize,
    y_off: usize,
    r0: __m128i,
    r1: __m128i,
    r2: __m128i,
    r3: __m128i,
) {
    let b0 = y_off * stride;
    let b1 = b0 + stride;
    let b2 = b1 + stride;
    let b3 = b2 + stride;

    let d0 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, b0)));
    let d1 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, b1)));
    let d2 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, b2)));
    let d3 = _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(dst, b3)));

    let z = _mm_setzero_si128();
    let p01 = _mm_unpacklo_epi8(_mm_unpacklo_epi32(d0, d1), z);
    let p23 = _mm_unpacklo_epi8(_mm_unpacklo_epi32(d2, d3), z);
    let m01 = _mm_unpacklo_epi64(r0, r1);
    let m23 = _mm_unpacklo_epi64(r2, r3);
    let s = _mm_packus_epi16(_mm_adds_epi16(p01, m01), _mm_adds_epi16(p23, m23));

    store_u8x4(chunk_mut(dst, b0), s);
    store_u8x4(chunk_mut(dst, b1), _mm_srli_si128::<4>(s));
    store_u8x4(chunk_mut(dst, b2), _mm_srli_si128::<8>(s));
    store_u8x4(chunk_mut(dst, b3), _mm_srli_si128::<12>(s));
}

/// `8 × 2` bölgeye 8 şeritli iki satır `i16` artığı ekler ve yazar (`0` taşma).
#[target_feature(enable = "ssse3")]
#[inline]
fn ekle_yaz_8x2(dst: &mut [u8], stride: usize, y_off: usize, r0: __m128i, r1: __m128i) {
    let b0 = y_off * stride;
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

/// Tek bir `16` piksellik satıra sol/sağ 8 şeritli `i16` artığı ekler ve yazar (`0` taşma).
#[target_feature(enable = "ssse3")]
#[inline]
fn ekle_yaz_16x1(dst: &mut [u8], off: usize, r_lo: __m128i, r_hi: __m128i) {
    let d = load_u8x16(chunk::<u8, 16>(dst, off));
    let z = _mm_setzero_si128();
    let s_lo = _mm_adds_epi16(_mm_unpacklo_epi8(d, z), r_lo);
    let s_hi = _mm_adds_epi16(_mm_unpackhi_epi8(d, z), r_hi);
    store_u8x16(chunk_mut(dst, off), _mm_packus_epi16(s_lo, s_hi));
}

/// `DCT_DCT` + `eob == 0` DC-only hızlı yolu (`coeff[0] = 0`; `coeff[1..]` korunur).
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

    match W {
        4 => {
            let mut y = 0;
            while y < H {
                ekle_yaz_4x4(dst, stride, y, dc_v, dc_v, dc_v, dc_v);
                y += 4;
            }
        }
        8 => {
            let mut y = 0;
            while y < H {
                ekle_yaz_8x2(dst, stride, y, dc_v, dc_v);
                y += 2;
            }
        }
        16 => {
            let mut y = 0;
            while y < H {
                ekle_yaz_16x1(dst, y * stride, dc_v, dc_v);
                y += 1;
            }
        }
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Matris devrikleme (transpose) yardımcıları
// ---------------------------------------------------------------------------

/// `4 × 4` devrikleme (alt 4 `i16` şeridi etkin).
#[target_feature(enable = "ssse3")]
#[inline]
fn devrik_4x4(r: [__m128i; 4]) -> [__m128i; 4] {
    let a0 = _mm_unpacklo_epi16(r[0], r[1]);
    let a1 = _mm_unpacklo_epi16(r[2], r[3]);
    let b0 = _mm_unpacklo_epi32(a0, a1);
    let b1 = _mm_unpackhi_epi32(a0, a1);
    [b0, _mm_srli_si128::<8>(b0), b1, _mm_srli_si128::<8>(b1)]
}

/// `4 sütun (her biri 8 satır) → 8 satır (her biri alt 4 şerit)` devrikleme.
#[target_feature(enable = "ssse3")]
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

/// `8 sütun (her biri alt 4 satır) → 4 satır (her biri 8 şerit)` devrikleme.
#[target_feature(enable = "ssse3")]
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

/// `8 × 8` `i16` devrikleme.
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
// 1B dönüşümler — 4 nokta (4 şerit ve 8 şerit)
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
#[inline]
fn idct4_x4(c: [__m128i; 4]) -> [__m128i; 4] {
    let (t0, t1) = mul2_x4(c[0], c[2], 2896, 2896, 2896, -2896);
    let (t2, t3) = mul2_x4(c[1], c[3], 1567, -3784, 3784, 1567);
    [
        _mm_adds_epi16(t0, t3),
        _mm_adds_epi16(t1, t2),
        _mm_subs_epi16(t1, t2),
        _mm_subs_epi16(t0, t3),
    ]
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idct4_x8(c: [__m128i; 4]) -> [__m128i; 4] {
    let (t0, t1) = mul2_x8(c[0], c[2], 2896, 2896, 2896, -2896);
    let (t2, t3) = mul2_x8(c[1], c[3], 1567, -3784, 3784, 1567);
    [
        _mm_adds_epi16(t0, t3),
        _mm_adds_epi16(t1, t2),
        _mm_subs_epi16(t1, t2),
        _mm_subs_epi16(t0, t3),
    ]
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idct4_yarim_x4(c0: __m128i, c1: __m128i) -> [__m128i; 4] {
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
fn idct4_yarim_x8(c0: __m128i, c1: __m128i) -> [__m128i; 4] {
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
fn iadst4_x4(c: [__m128i; 4], flip: bool) -> [__m128i; 4] {
    let p02 = _mm_unpacklo_epi16(c[0], c[2]);
    let p13 = _mm_unpacklo_epi16(c[1], c[3]);
    let rnd = _mm_set1_epi32(2048);

    let y0 = _mm_srai_epi32::<12>(_mm_add_epi32(
        _mm_add_epi32(
            _mm_madd_epi16(p02, _mm_set1_epi32(cift_katsayi(1321, 3803))),
            _mm_madd_epi16(p13, _mm_set1_epi32(cift_katsayi(3344, 2482))),
        ),
        rnd,
    ));
    let y1 = _mm_srai_epi32::<12>(_mm_add_epi32(
        _mm_add_epi32(
            _mm_madd_epi16(p02, _mm_set1_epi32(cift_katsayi(2482, -1321))),
            _mm_madd_epi16(p13, _mm_set1_epi32(cift_katsayi(3344, -3803))),
        ),
        rnd,
    ));
    let y2 = _mm_srai_epi32::<12>(_mm_add_epi32(
        _mm_add_epi32(
            _mm_madd_epi16(p02, _mm_set1_epi32(cift_katsayi(3344, -3344))),
            _mm_madd_epi16(p13, _mm_set1_epi32(cift_katsayi(0, 3344))),
        ),
        rnd,
    ));
    let y3 = _mm_srai_epi32::<12>(_mm_add_epi32(
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

#[target_feature(enable = "ssse3")]
#[inline]
fn iadst4_x8(c: [__m128i; 4], flip: bool) -> [__m128i; 4] {
    let p02_lo = _mm_unpacklo_epi16(c[0], c[2]);
    let p02_hi = _mm_unpackhi_epi16(c[0], c[2]);
    let p13_lo = _mm_unpacklo_epi16(c[1], c[3]);
    let p13_hi = _mm_unpackhi_epi16(c[1], c[3]);
    let rnd = _mm_set1_epi32(2048);

    let k0_02 = _mm_set1_epi32(cift_katsayi(1321, 3803));
    let k0_13 = _mm_set1_epi32(cift_katsayi(3344, 2482));
    let y0_lo = _mm_srai_epi32::<12>(_mm_add_epi32(
        _mm_add_epi32(_mm_madd_epi16(p02_lo, k0_02), _mm_madd_epi16(p13_lo, k0_13)),
        rnd,
    ));
    let y0_hi = _mm_srai_epi32::<12>(_mm_add_epi32(
        _mm_add_epi32(_mm_madd_epi16(p02_hi, k0_02), _mm_madd_epi16(p13_hi, k0_13)),
        rnd,
    ));

    let k1_02 = _mm_set1_epi32(cift_katsayi(2482, -1321));
    let k1_13 = _mm_set1_epi32(cift_katsayi(3344, -3803));
    let y1_lo = _mm_srai_epi32::<12>(_mm_add_epi32(
        _mm_add_epi32(_mm_madd_epi16(p02_lo, k1_02), _mm_madd_epi16(p13_lo, k1_13)),
        rnd,
    ));
    let y1_hi = _mm_srai_epi32::<12>(_mm_add_epi32(
        _mm_add_epi32(_mm_madd_epi16(p02_hi, k1_02), _mm_madd_epi16(p13_hi, k1_13)),
        rnd,
    ));

    let k2_02 = _mm_set1_epi32(cift_katsayi(3344, -3344));
    let k2_13 = _mm_set1_epi32(cift_katsayi(0, 3344));
    let y2_lo = _mm_srai_epi32::<12>(_mm_add_epi32(
        _mm_add_epi32(_mm_madd_epi16(p02_lo, k2_02), _mm_madd_epi16(p13_lo, k2_13)),
        rnd,
    ));
    let y2_hi = _mm_srai_epi32::<12>(_mm_add_epi32(
        _mm_add_epi32(_mm_madd_epi16(p02_hi, k2_02), _mm_madd_epi16(p13_hi, k2_13)),
        rnd,
    ));

    let k3_02 = _mm_set1_epi32(cift_katsayi(3803, 2482));
    let k3_13 = _mm_set1_epi32(cift_katsayi(-3344, -1321));
    let y3_lo = _mm_srai_epi32::<12>(_mm_add_epi32(
        _mm_add_epi32(_mm_madd_epi16(p02_lo, k3_02), _mm_madd_epi16(p13_lo, k3_13)),
        rnd,
    ));
    let y3_hi = _mm_srai_epi32::<12>(_mm_add_epi32(
        _mm_add_epi32(_mm_madd_epi16(p02_hi, k3_02), _mm_madd_epi16(p13_hi, k3_13)),
        rnd,
    ));

    let o0 = _mm_packs_epi32(y0_lo, y0_hi);
    let o1 = _mm_packs_epi32(y1_lo, y1_hi);
    let o2 = _mm_packs_epi32(y2_lo, y2_hi);
    let o3 = _mm_packs_epi32(y3_lo, y3_hi);
    if flip {
        [o3, o2, o1, o0]
    } else {
        [o0, o1, o2, o3]
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn iidtx4_gecis1(c: [__m128i; 4], shift: u32) -> [__m128i; 4] {
    let k = _mm_set1_epi16(1697 * 8);
    let mut out = [_mm_setzero_si128(); 4];
    let mut i = 0;
    while i < 4 {
        let t = _mm_mulhrs_epi16(c[i], k);
        out[i] = if shift == 0 {
            _mm_adds_epi16(c[i], t)
        } else {
            savg_epi16(c[i], t)
        };
        i += 1;
    }
    out
}

#[target_feature(enable = "ssse3")]
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

#[target_feature(enable = "ssse3")]
#[inline]
fn iidtx4_gecis2_x8(c: [__m128i; 4]) -> [__m128i; 4] {
    let k = _mm_set1_epi16(1697 * 8);
    let bir = _mm_set1_epi32(cift_katsayi(1, 1));
    let sekiz = _mm_set1_epi32(8);
    let mut out = [_mm_setzero_si128(); 4];
    let mut i = 0;
    while i < 4 {
        let t = _mm_mulhrs_epi16(c[i], k);
        let lo = _mm_unpacklo_epi16(c[i], t);
        let hi = _mm_unpackhi_epi16(c[i], t);
        let s_lo = _mm_srai_epi32::<4>(_mm_add_epi32(_mm_madd_epi16(lo, bir), sekiz));
        let s_hi = _mm_srai_epi32::<4>(_mm_add_epi32(_mm_madd_epi16(hi, bir), sekiz));
        out[i] = _mm_packs_epi32(s_lo, s_hi);
        i += 1;
    }
    out
}

#[target_feature(enable = "ssse3")]
#[inline]
fn gecis1_4_x4(c: [__m128i; 4], kind: Tx1dType, shift: u32) -> [__m128i; 4] {
    match kind {
        Tx1dType::Identity => iidtx4_gecis1(c, shift),
        Tx1dType::Dct => {
            let o = idct4_x4(c);
            if shift == 0 {
                o
            } else {
                let m = _mm_set1_epi16(16384);
                [
                    _mm_mulhrs_epi16(o[0], m),
                    _mm_mulhrs_epi16(o[1], m),
                    _mm_mulhrs_epi16(o[2], m),
                    _mm_mulhrs_epi16(o[3], m),
                ]
            }
        }
        Tx1dType::Adst | Tx1dType::FlipAdst => {
            let o = iadst4_x4(c, matches!(kind, Tx1dType::FlipAdst));
            if shift == 0 {
                o
            } else {
                let m = _mm_set1_epi16(16384);
                [
                    _mm_mulhrs_epi16(o[0], m),
                    _mm_mulhrs_epi16(o[1], m),
                    _mm_mulhrs_epi16(o[2], m),
                    _mm_mulhrs_epi16(o[3], m),
                ]
            }
        }
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn gecis1_4_x8(c: [__m128i; 4], kind: Tx1dType, shift: u32) -> [__m128i; 4] {
    match kind {
        Tx1dType::Identity => iidtx4_gecis1(c, shift),
        Tx1dType::Dct => {
            let o = idct4_x8(c);
            if shift == 0 {
                o
            } else {
                let m = _mm_set1_epi16(16384);
                [
                    _mm_mulhrs_epi16(o[0], m),
                    _mm_mulhrs_epi16(o[1], m),
                    _mm_mulhrs_epi16(o[2], m),
                    _mm_mulhrs_epi16(o[3], m),
                ]
            }
        }
        Tx1dType::Adst | Tx1dType::FlipAdst => {
            let o = iadst4_x8(c, matches!(kind, Tx1dType::FlipAdst));
            if shift == 0 {
                o
            } else {
                let m = _mm_set1_epi16(16384);
                [
                    _mm_mulhrs_epi16(o[0], m),
                    _mm_mulhrs_epi16(o[1], m),
                    _mm_mulhrs_epi16(o[2], m),
                    _mm_mulhrs_epi16(o[3], m),
                ]
            }
        }
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn gecis2_4_x4(c: [__m128i; 4], kind: Tx1dType) -> [__m128i; 4] {
    match kind {
        Tx1dType::Identity => iidtx4_gecis2_x4(c),
        Tx1dType::Dct => {
            let o = idct4_x4(c);
            let m = _mm_set1_epi16(2048);
            [
                _mm_mulhrs_epi16(o[0], m),
                _mm_mulhrs_epi16(o[1], m),
                _mm_mulhrs_epi16(o[2], m),
                _mm_mulhrs_epi16(o[3], m),
            ]
        }
        Tx1dType::Adst | Tx1dType::FlipAdst => {
            let o = iadst4_x4(c, matches!(kind, Tx1dType::FlipAdst));
            let m = _mm_set1_epi16(2048);
            [
                _mm_mulhrs_epi16(o[0], m),
                _mm_mulhrs_epi16(o[1], m),
                _mm_mulhrs_epi16(o[2], m),
                _mm_mulhrs_epi16(o[3], m),
            ]
        }
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn gecis2_4_x8(c: [__m128i; 4], kind: Tx1dType) -> [__m128i; 4] {
    match kind {
        Tx1dType::Identity => iidtx4_gecis2_x8(c),
        Tx1dType::Dct => {
            let o = idct4_x8(c);
            let m = _mm_set1_epi16(2048);
            [
                _mm_mulhrs_epi16(o[0], m),
                _mm_mulhrs_epi16(o[1], m),
                _mm_mulhrs_epi16(o[2], m),
                _mm_mulhrs_epi16(o[3], m),
            ]
        }
        Tx1dType::Adst | Tx1dType::FlipAdst => {
            let o = iadst4_x8(c, matches!(kind, Tx1dType::FlipAdst));
            let m = _mm_set1_epi16(2048);
            [
                _mm_mulhrs_epi16(o[0], m),
                _mm_mulhrs_epi16(o[1], m),
                _mm_mulhrs_epi16(o[2], m),
                _mm_mulhrs_epi16(o[3], m),
            ]
        }
    }
}

// ---------------------------------------------------------------------------
// 1B dönüşümler — 8 nokta (4 şerit ve 8 şerit)
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
#[inline]
fn idct8_x4(c: [__m128i; 8]) -> [__m128i; 8] {
    let [t0, t1, t2, t3] = idct4_x4([c[0], c[2], c[4], c[6]]);
    let (t4a, t7a) = mul2_x4(c[1], c[7], 799, -4017, 4017, 799);
    let (t5a, t6a) = mul2_x4(c[5], c[3], 3406, -2276, 2276, 3406);
    let t4 = _mm_adds_epi16(t4a, t5a);
    let t5a = _mm_subs_epi16(t4a, t5a);
    let t7 = _mm_adds_epi16(t7a, t6a);
    let t6a = _mm_subs_epi16(t7a, t6a);
    let (t5, t6) = mul2_x4(t6a, t5a, 2896, -2896, 2896, 2896);
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
fn idct8_x8(c: [__m128i; 8]) -> [__m128i; 8] {
    let [t0, t1, t2, t3] = idct4_x8([c[0], c[2], c[4], c[6]]);
    let (t4a, t7a) = mul2_x8(c[1], c[7], 799, -4017, 4017, 799);
    let (t5a, t6a) = mul2_x8(c[5], c[3], 3406, -2276, 2276, 3406);
    let t4 = _mm_adds_epi16(t4a, t5a);
    let t5a = _mm_subs_epi16(t4a, t5a);
    let t7 = _mm_adds_epi16(t7a, t6a);
    let t6a = _mm_subs_epi16(t7a, t6a);
    let (t5, t6) = mul2_x8(t6a, t5a, 2896, -2896, 2896, 2896);
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
fn idct8_yarim_x4(c: [__m128i; 4]) -> [__m128i; 8] {
    let [t0, t1, t2, t3] = idct4_yarim_x4(c[0], c[2]);
    let t4a = _mm_mulhrs_epi16(c[1], _mm_set1_epi16(799 * 8));
    let t7a = _mm_mulhrs_epi16(c[1], _mm_set1_epi16(4017 * 8));
    let t5a = _mm_mulhrs_epi16(c[3], _mm_set1_epi16(-2276 * 8));
    let t6a = _mm_mulhrs_epi16(c[3], _mm_set1_epi16(3406 * 8));
    let t4 = _mm_adds_epi16(t4a, t5a);
    let t5a = _mm_subs_epi16(t4a, t5a);
    let t7 = _mm_adds_epi16(t7a, t6a);
    let t6a = _mm_subs_epi16(t7a, t6a);
    let (t5, t6) = mul2_x4(t6a, t5a, 2896, -2896, 2896, 2896);
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
fn idct8_yarim_x8(c: [__m128i; 4]) -> [__m128i; 8] {
    let [t0, t1, t2, t3] = idct4_yarim_x8(c[0], c[2]);
    let t4a = _mm_mulhrs_epi16(c[1], _mm_set1_epi16(799 * 8));
    let t7a = _mm_mulhrs_epi16(c[1], _mm_set1_epi16(4017 * 8));
    let t5a = _mm_mulhrs_epi16(c[3], _mm_set1_epi16(-2276 * 8));
    let t6a = _mm_mulhrs_epi16(c[3], _mm_set1_epi16(3406 * 8));
    let t4 = _mm_adds_epi16(t4a, t5a);
    let t5a = _mm_subs_epi16(t4a, t5a);
    let t7 = _mm_adds_epi16(t7a, t6a);
    let t6a = _mm_subs_epi16(t7a, t6a);
    let (t5, t6) = mul2_x8(t6a, t5a, 2896, -2896, 2896, 2896);
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

/// Ham `iadst8` (çift dizinler `+out`, tek dizinler `-out`).
#[target_feature(enable = "ssse3")]
#[inline]
fn iadst8_ham_x4(c: [__m128i; 8]) -> [__m128i; 8] {
    let (t0a, t1a) = mul2_x4(c[7], c[0], 4076, 401, 401, -4076);
    let (t2a, t3a) = mul2_x4(c[5], c[2], 3612, 1931, 1931, -3612);
    let (t4a, t5a) = mul2_x4(c[3], c[4], 2598, 3166, 3166, -2598);
    let (t6a, t7a) = mul2_x4(c[1], c[6], 1189, 3920, 3920, -1189);

    let t0 = _mm_adds_epi16(t0a, t4a);
    let t1 = _mm_adds_epi16(t1a, t5a);
    let t2 = _mm_adds_epi16(t2a, t6a);
    let t3 = _mm_adds_epi16(t3a, t7a);
    let t4 = _mm_subs_epi16(t0a, t4a);
    let t5 = _mm_subs_epi16(t1a, t5a);
    let t6 = _mm_subs_epi16(t2a, t6a);
    let t7 = _mm_subs_epi16(t3a, t7a);

    let (t4a, t5a) = mul2_x4(t4, t5, 3784, 1567, 1567, -3784);
    let (t6a, t7a) = mul2_x4(t7, t6, 3784, -1567, 1567, 3784);

    let u0 = _mm_adds_epi16(t0, t2);
    let u7 = _mm_adds_epi16(t1, t3);
    let t2b = _mm_subs_epi16(t0, t2);
    let t3b = _mm_subs_epi16(t1, t3);
    let u1 = _mm_adds_epi16(t4a, t6a);
    let u6 = _mm_adds_epi16(t5a, t7a);
    let t6b = _mm_subs_epi16(t4a, t6a);
    let t7b = _mm_subs_epi16(t5a, t7a);

    let (u3, u4) = mul2_x4(t2b, t3b, 2896, 2896, 2896, -2896);
    let (u2, u5) = mul2_x4(t6b, t7b, 2896, 2896, 2896, -2896);
    [u0, u1, u2, u3, u4, u5, u6, u7]
}

/// Ham `iadst8` (8 şerit; çift dizinler `+out`, tek dizinler `-out`).
#[target_feature(enable = "ssse3")]
#[inline]
fn iadst8_ham_x8(c: [__m128i; 8]) -> [__m128i; 8] {
    let (t0a, t1a) = mul2_x8(c[7], c[0], 4076, 401, 401, -4076);
    let (t2a, t3a) = mul2_x8(c[5], c[2], 3612, 1931, 1931, -3612);
    let (t4a, t5a) = mul2_x8(c[3], c[4], 2598, 3166, 3166, -2598);
    let (t6a, t7a) = mul2_x8(c[1], c[6], 1189, 3920, 3920, -1189);

    let t0 = _mm_adds_epi16(t0a, t4a);
    let t1 = _mm_adds_epi16(t1a, t5a);
    let t2 = _mm_adds_epi16(t2a, t6a);
    let t3 = _mm_adds_epi16(t3a, t7a);
    let t4 = _mm_subs_epi16(t0a, t4a);
    let t5 = _mm_subs_epi16(t1a, t5a);
    let t6 = _mm_subs_epi16(t2a, t6a);
    let t7 = _mm_subs_epi16(t3a, t7a);

    let (t4a, t5a) = mul2_x8(t4, t5, 3784, 1567, 1567, -3784);
    let (t6a, t7a) = mul2_x8(t7, t6, 3784, -1567, 1567, 3784);

    let u0 = _mm_adds_epi16(t0, t2);
    let u7 = _mm_adds_epi16(t1, t3);
    let t2b = _mm_subs_epi16(t0, t2);
    let t3b = _mm_subs_epi16(t1, t3);
    let u1 = _mm_adds_epi16(t4a, t6a);
    let u6 = _mm_adds_epi16(t5a, t7a);
    let t6b = _mm_subs_epi16(t4a, t6a);
    let t7b = _mm_subs_epi16(t5a, t7a);

    let (u3, u4) = mul2_x8(t2b, t3b, 2896, 2896, 2896, -2896);
    let (u2, u5) = mul2_x8(t6b, t7b, 2896, 2896, 2896, -2896);
    [u0, u1, u2, u3, u4, u5, u6, u7]
}

#[target_feature(enable = "ssse3")]
#[inline]
fn adst8_sonlandir(u: [__m128i; 8], shift_mul: i16, flip: bool) -> [__m128i; 8] {
    let mut o = if shift_mul == 0 {
        let z = _mm_setzero_si128();
        [
            u[0],
            _mm_subs_epi16(z, u[1]),
            u[2],
            _mm_subs_epi16(z, u[3]),
            u[4],
            _mm_subs_epi16(z, u[5]),
            u[6],
            _mm_subs_epi16(z, u[7]),
        ]
    } else {
        let p = _mm_set1_epi16(shift_mul);
        let n = _mm_set1_epi16(-shift_mul);
        [
            _mm_mulhrs_epi16(u[0], p),
            _mm_mulhrs_epi16(u[1], n),
            _mm_mulhrs_epi16(u[2], p),
            _mm_mulhrs_epi16(u[3], n),
            _mm_mulhrs_epi16(u[4], p),
            _mm_mulhrs_epi16(u[5], n),
            _mm_mulhrs_epi16(u[6], p),
            _mm_mulhrs_epi16(u[7], n),
        ]
    };
    if flip {
        o.reverse();
    }
    o
}

#[target_feature(enable = "ssse3")]
#[inline]
fn gecis1_8_x4(c: [__m128i; 8], kind: Tx1dType, shift: u32) -> [__m128i; 8] {
    let shift_mul: i16 = match shift {
        0 => 0,
        1 => 16384,
        _ => 8192,
    };
    match kind {
        Tx1dType::Identity => {
            if shift == 0 {
                let mut o = c;
                let mut i = 0;
                while i < 8 {
                    o[i] = _mm_adds_epi16(c[i], c[i]);
                    i += 1;
                }
                o
            } else {
                c
            }
        }
        Tx1dType::Dct => {
            let mut o = idct8_x4(c);
            if shift_mul != 0 {
                let m = _mm_set1_epi16(shift_mul);
                let mut i = 0;
                while i < 8 {
                    o[i] = _mm_mulhrs_epi16(o[i], m);
                    i += 1;
                }
            }
            o
        }
        Tx1dType::Adst | Tx1dType::FlipAdst => adst8_sonlandir(
            iadst8_ham_x4(c),
            shift_mul,
            matches!(kind, Tx1dType::FlipAdst),
        ),
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn gecis1_8_x8(c: [__m128i; 8], kind: Tx1dType, shift: u32) -> [__m128i; 8] {
    let shift_mul: i16 = match shift {
        0 => 0,
        1 => 16384,
        _ => 8192,
    };
    match kind {
        Tx1dType::Identity => {
            if shift == 0 {
                let mut o = c;
                let mut i = 0;
                while i < 8 {
                    o[i] = _mm_adds_epi16(c[i], c[i]);
                    i += 1;
                }
                o
            } else {
                c
            }
        }
        Tx1dType::Dct => {
            let mut o = idct8_x8(c);
            if shift_mul != 0 {
                let m = _mm_set1_epi16(shift_mul);
                let mut i = 0;
                while i < 8 {
                    o[i] = _mm_mulhrs_epi16(o[i], m);
                    i += 1;
                }
            }
            o
        }
        Tx1dType::Adst | Tx1dType::FlipAdst => adst8_sonlandir(
            iadst8_ham_x8(c),
            shift_mul,
            matches!(kind, Tx1dType::FlipAdst),
        ),
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn gecis2_8_x4(c: [__m128i; 8], kind: Tx1dType) -> [__m128i; 8] {
    match kind {
        Tx1dType::Identity => {
            let m = _mm_set1_epi16(4096);
            let mut o = c;
            let mut i = 0;
            while i < 8 {
                o[i] = _mm_mulhrs_epi16(c[i], m);
                i += 1;
            }
            o
        }
        Tx1dType::Dct => {
            let mut o = idct8_x4(c);
            let m = _mm_set1_epi16(2048);
            let mut i = 0;
            while i < 8 {
                o[i] = _mm_mulhrs_epi16(o[i], m);
                i += 1;
            }
            o
        }
        Tx1dType::Adst | Tx1dType::FlipAdst => {
            adst8_sonlandir(iadst8_ham_x4(c), 2048, matches!(kind, Tx1dType::FlipAdst))
        }
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn gecis2_8_x8(c: [__m128i; 8], kind: Tx1dType) -> [__m128i; 8] {
    match kind {
        Tx1dType::Identity => {
            let m = _mm_set1_epi16(4096);
            let mut o = c;
            let mut i = 0;
            while i < 8 {
                o[i] = _mm_mulhrs_epi16(c[i], m);
                i += 1;
            }
            o
        }
        Tx1dType::Dct => {
            let mut o = idct8_x8(c);
            let m = _mm_set1_epi16(2048);
            let mut i = 0;
            while i < 8 {
                o[i] = _mm_mulhrs_epi16(o[i], m);
                i += 1;
            }
            o
        }
        Tx1dType::Adst | Tx1dType::FlipAdst => {
            adst8_sonlandir(iadst8_ham_x8(c), 2048, matches!(kind, Tx1dType::FlipAdst))
        }
    }
}

// ---------------------------------------------------------------------------
// 1B dönüşümler — 16 nokta (4 şerit ve 8 şerit)
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
#[inline]
fn idct16_ortak_son_x4(
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

    let (t9a, t14a) = mul2_x4(t14, t9, 1567, -3784, 3784, 1567);
    let (t10a, t13a) = mul2_x4(t13, t10, -3784, -1567, 1567, -3784);

    let t8a = _mm_adds_epi16(t8, t11);
    let t9 = _mm_adds_epi16(t9a, t10a);
    let t10 = _mm_subs_epi16(t9a, t10a);
    let t11a = _mm_subs_epi16(t8, t11);
    let t12a = _mm_subs_epi16(t15, t12);
    let t13 = _mm_subs_epi16(t14a, t13a);
    let t14 = _mm_adds_epi16(t14a, t13a);
    let t15a = _mm_adds_epi16(t15, t12);

    let (t10a, t13a) = mul2_x4(t13, t10, 2896, -2896, 2896, 2896);
    let (t11, t12) = mul2_x4(t12a, t11a, 2896, -2896, 2896, 2896);

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
fn idct16_ortak_son_x8(
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

    let (t9a, t14a) = mul2_x8(t14, t9, 1567, -3784, 3784, 1567);
    let (t10a, t13a) = mul2_x8(t13, t10, -3784, -1567, 1567, -3784);

    let t8a = _mm_adds_epi16(t8, t11);
    let t9 = _mm_adds_epi16(t9a, t10a);
    let t10 = _mm_subs_epi16(t9a, t10a);
    let t11a = _mm_subs_epi16(t8, t11);
    let t12a = _mm_subs_epi16(t15, t12);
    let t13 = _mm_subs_epi16(t14a, t13a);
    let t14 = _mm_adds_epi16(t14a, t13a);
    let t15a = _mm_adds_epi16(t15, t12);

    let (t10a, t13a) = mul2_x8(t13, t10, 2896, -2896, 2896, 2896);
    let (t11, t12) = mul2_x8(t12a, t11a, 2896, -2896, 2896, 2896);

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
fn idct16_x4(c: [__m128i; 16]) -> [__m128i; 16] {
    let e = idct8_x4([c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]]);
    let (t8a, t15a) = mul2_x4(c[1], c[15], 401, -4076, 4076, 401);
    let (t9a, t14a) = mul2_x4(c[9], c[7], 3166, -2598, 2598, 3166);
    let (t10a, t13a) = mul2_x4(c[5], c[11], 1931, -3612, 3612, 1931);
    let (t11a, t12a) = mul2_x4(c[13], c[3], 3920, -1189, 1189, 3920);
    idct16_ortak_son_x4(e, [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a])
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idct16_yarim_x4(c: [__m128i; 8]) -> [__m128i; 16] {
    let e = idct8_yarim_x4([c[0], c[2], c[4], c[6]]);
    let t8a = _mm_mulhrs_epi16(c[1], _mm_set1_epi16(401 * 8));
    let t15a = _mm_mulhrs_epi16(c[1], _mm_set1_epi16(4076 * 8));
    let t9a = _mm_mulhrs_epi16(c[7], _mm_set1_epi16(-2598 * 8));
    let t14a = _mm_mulhrs_epi16(c[7], _mm_set1_epi16(3166 * 8));
    let t10a = _mm_mulhrs_epi16(c[5], _mm_set1_epi16(1931 * 8));
    let t13a = _mm_mulhrs_epi16(c[5], _mm_set1_epi16(3612 * 8));
    let t11a = _mm_mulhrs_epi16(c[3], _mm_set1_epi16(-1189 * 8));
    let t12a = _mm_mulhrs_epi16(c[3], _mm_set1_epi16(3920 * 8));
    idct16_ortak_son_x4(e, [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a])
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idct16_x8(c: [__m128i; 16]) -> [__m128i; 16] {
    let e = idct8_x8([c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]]);
    let (t8a, t15a) = mul2_x8(c[1], c[15], 401, -4076, 4076, 401);
    let (t9a, t14a) = mul2_x8(c[9], c[7], 3166, -2598, 2598, 3166);
    let (t10a, t13a) = mul2_x8(c[5], c[11], 1931, -3612, 3612, 1931);
    let (t11a, t12a) = mul2_x8(c[13], c[3], 3920, -1189, 1189, 3920);
    idct16_ortak_son_x8(e, [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a])
}

#[target_feature(enable = "ssse3")]
#[inline]
fn idct16_yarim_x8(c: [__m128i; 8]) -> [__m128i; 16] {
    let e = idct8_yarim_x8([c[0], c[2], c[4], c[6]]);
    let t8a = _mm_mulhrs_epi16(c[1], _mm_set1_epi16(401 * 8));
    let t15a = _mm_mulhrs_epi16(c[1], _mm_set1_epi16(4076 * 8));
    let t9a = _mm_mulhrs_epi16(c[7], _mm_set1_epi16(-2598 * 8));
    let t14a = _mm_mulhrs_epi16(c[7], _mm_set1_epi16(3166 * 8));
    let t10a = _mm_mulhrs_epi16(c[5], _mm_set1_epi16(1931 * 8));
    let t13a = _mm_mulhrs_epi16(c[5], _mm_set1_epi16(3612 * 8));
    let t11a = _mm_mulhrs_epi16(c[3], _mm_set1_epi16(-1189 * 8));
    let t12a = _mm_mulhrs_epi16(c[3], _mm_set1_epi16(3920 * 8));
    idct16_ortak_son_x8(e, [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a])
}

/// Ham `iadst16` (4 şerit; çift dizinler `+out`, tek dizinler `-out`).
#[target_feature(enable = "ssse3")]
#[inline]
fn iadst16_ham_x4(c: [__m128i; 16]) -> [__m128i; 16] {
    let (t0, t1) = mul2_x4(c[15], c[0], 4091, 201, 201, -4091);
    let (t2, t3) = mul2_x4(c[13], c[2], 3973, 995, 995, -3973);
    let (t4, t5) = mul2_x4(c[11], c[4], 3703, 1751, 1751, -3703);
    let (t6, t7) = mul2_x4(c[9], c[6], 3290, 2440, 2440, -3290);
    let (t8, t9) = mul2_x4(c[7], c[8], 2751, 3035, 3035, -2751);
    let (t10, t11) = mul2_x4(c[5], c[10], 2106, 3513, 3513, -2106);
    let (t12, t13) = mul2_x4(c[3], c[12], 1380, 3857, 3857, -1380);
    let (t14, t15) = mul2_x4(c[1], c[14], 601, 4052, 4052, -601);

    let t0a = _mm_adds_epi16(t0, t8);
    let t1a = _mm_adds_epi16(t1, t9);
    let t2a = _mm_adds_epi16(t2, t10);
    let t3a = _mm_adds_epi16(t3, t11);
    let t4a = _mm_adds_epi16(t4, t12);
    let t5a = _mm_adds_epi16(t5, t13);
    let t6a = _mm_adds_epi16(t6, t14);
    let t7a = _mm_adds_epi16(t7, t15);
    let t8a = _mm_subs_epi16(t0, t8);
    let t9a = _mm_subs_epi16(t1, t9);
    let t10a = _mm_subs_epi16(t2, t10);
    let t11a = _mm_subs_epi16(t3, t11);
    let t12a = _mm_subs_epi16(t4, t12);
    let t13a = _mm_subs_epi16(t5, t13);
    let t14a = _mm_subs_epi16(t6, t14);
    let t15a = _mm_subs_epi16(t7, t15);

    let (t8, t9) = mul2_x4(t8a, t9a, 4017, 799, 799, -4017);
    let (t10, t11) = mul2_x4(t10a, t11a, 2276, 3406, 3406, -2276);
    let (t12, t13) = mul2_x4(t13a, t12a, 4017, -799, 799, 4017);
    let (t14, t15) = mul2_x4(t15a, t14a, 2276, -3406, 3406, 2276);

    let t0 = _mm_adds_epi16(t0a, t4a);
    let t1 = _mm_adds_epi16(t1a, t5a);
    let t2 = _mm_adds_epi16(t2a, t6a);
    let t3 = _mm_adds_epi16(t3a, t7a);
    let t4 = _mm_subs_epi16(t0a, t4a);
    let t5 = _mm_subs_epi16(t1a, t5a);
    let t6 = _mm_subs_epi16(t2a, t6a);
    let t7 = _mm_subs_epi16(t3a, t7a);
    let t8a = _mm_adds_epi16(t8, t12);
    let t9a = _mm_adds_epi16(t9, t13);
    let t10a = _mm_adds_epi16(t10, t14);
    let t11a = _mm_adds_epi16(t11, t15);
    let t12a = _mm_subs_epi16(t8, t12);
    let t13a = _mm_subs_epi16(t9, t13);
    let t14a = _mm_subs_epi16(t10, t14);
    let t15a = _mm_subs_epi16(t11, t15);

    let (t4a, t5a) = mul2_x4(t4, t5, 3784, 1567, 1567, -3784);
    let (t6a, t7a) = mul2_x4(t7, t6, 3784, -1567, 1567, 3784);
    let (t12, t13) = mul2_x4(t12a, t13a, 3784, 1567, 1567, -3784);
    let (t14, t15) = mul2_x4(t15a, t14a, 3784, -1567, 1567, 3784);

    let u0 = _mm_adds_epi16(t0, t2);
    let u15 = _mm_adds_epi16(t1, t3);
    let t2a = _mm_subs_epi16(t0, t2);
    let t3a = _mm_subs_epi16(t1, t3);
    let u3 = _mm_adds_epi16(t4a, t6a);
    let u12 = _mm_adds_epi16(t5a, t7a);
    let t6 = _mm_subs_epi16(t4a, t6a);
    let t7 = _mm_subs_epi16(t5a, t7a);
    let u1 = _mm_adds_epi16(t8a, t10a);
    let u14 = _mm_adds_epi16(t9a, t11a);
    let t10 = _mm_subs_epi16(t8a, t10a);
    let t11 = _mm_subs_epi16(t9a, t11a);
    let u2 = _mm_adds_epi16(t12, t14);
    let u13 = _mm_adds_epi16(t13, t15);
    let t14a = _mm_subs_epi16(t12, t14);
    let t15a = _mm_subs_epi16(t13, t15);

    let (u7, u8) = mul2_x4(t2a, t3a, 2896, 2896, 2896, -2896);
    let (u4, u11) = mul2_x4(t6, t7, 2896, 2896, 2896, -2896);
    let (u6, u9) = mul2_x4(t10, t11, 2896, 2896, 2896, -2896);
    let (u5, u10) = mul2_x4(t14a, t15a, 2896, 2896, 2896, -2896);

    [
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15,
    ]
}

/// Ham `iadst16` (8 şerit; çift dizinler `+out`, tek dizinler `-out`).
#[target_feature(enable = "ssse3")]
#[inline]
fn iadst16_ham_x8(c: [__m128i; 16]) -> [__m128i; 16] {
    let (t0, t1) = mul2_x8(c[15], c[0], 4091, 201, 201, -4091);
    let (t2, t3) = mul2_x8(c[13], c[2], 3973, 995, 995, -3973);
    let (t4, t5) = mul2_x8(c[11], c[4], 3703, 1751, 1751, -3703);
    let (t6, t7) = mul2_x8(c[9], c[6], 3290, 2440, 2440, -3290);
    let (t8, t9) = mul2_x8(c[7], c[8], 2751, 3035, 3035, -2751);
    let (t10, t11) = mul2_x8(c[5], c[10], 2106, 3513, 3513, -2106);
    let (t12, t13) = mul2_x8(c[3], c[12], 1380, 3857, 3857, -1380);
    let (t14, t15) = mul2_x8(c[1], c[14], 601, 4052, 4052, -601);

    let t0a = _mm_adds_epi16(t0, t8);
    let t1a = _mm_adds_epi16(t1, t9);
    let t2a = _mm_adds_epi16(t2, t10);
    let t3a = _mm_adds_epi16(t3, t11);
    let t4a = _mm_adds_epi16(t4, t12);
    let t5a = _mm_adds_epi16(t5, t13);
    let t6a = _mm_adds_epi16(t6, t14);
    let t7a = _mm_adds_epi16(t7, t15);
    let t8a = _mm_subs_epi16(t0, t8);
    let t9a = _mm_subs_epi16(t1, t9);
    let t10a = _mm_subs_epi16(t2, t10);
    let t11a = _mm_subs_epi16(t3, t11);
    let t12a = _mm_subs_epi16(t4, t12);
    let t13a = _mm_subs_epi16(t5, t13);
    let t14a = _mm_subs_epi16(t6, t14);
    let t15a = _mm_subs_epi16(t7, t15);

    let (t8, t9) = mul2_x8(t8a, t9a, 4017, 799, 799, -4017);
    let (t10, t11) = mul2_x8(t10a, t11a, 2276, 3406, 3406, -2276);
    let (t12, t13) = mul2_x8(t13a, t12a, 4017, -799, 799, 4017);
    let (t14, t15) = mul2_x8(t15a, t14a, 2276, -3406, 3406, 2276);

    let t0 = _mm_adds_epi16(t0a, t4a);
    let t1 = _mm_adds_epi16(t1a, t5a);
    let t2 = _mm_adds_epi16(t2a, t6a);
    let t3 = _mm_adds_epi16(t3a, t7a);
    let t4 = _mm_subs_epi16(t0a, t4a);
    let t5 = _mm_subs_epi16(t1a, t5a);
    let t6 = _mm_subs_epi16(t2a, t6a);
    let t7 = _mm_subs_epi16(t3a, t7a);
    let t8a = _mm_adds_epi16(t8, t12);
    let t9a = _mm_adds_epi16(t9, t13);
    let t10a = _mm_adds_epi16(t10, t14);
    let t11a = _mm_adds_epi16(t11, t15);
    let t12a = _mm_subs_epi16(t8, t12);
    let t13a = _mm_subs_epi16(t9, t13);
    let t14a = _mm_subs_epi16(t10, t14);
    let t15a = _mm_subs_epi16(t11, t15);

    let (t4a, t5a) = mul2_x8(t4, t5, 3784, 1567, 1567, -3784);
    let (t6a, t7a) = mul2_x8(t7, t6, 3784, -1567, 1567, 3784);
    let (t12, t13) = mul2_x8(t12a, t13a, 3784, 1567, 1567, -3784);
    let (t14, t15) = mul2_x8(t15a, t14a, 3784, -1567, 1567, 3784);

    let u0 = _mm_adds_epi16(t0, t2);
    let u15 = _mm_adds_epi16(t1, t3);
    let t2a = _mm_subs_epi16(t0, t2);
    let t3a = _mm_subs_epi16(t1, t3);
    let u3 = _mm_adds_epi16(t4a, t6a);
    let u12 = _mm_adds_epi16(t5a, t7a);
    let t6 = _mm_subs_epi16(t4a, t6a);
    let t7 = _mm_subs_epi16(t5a, t7a);
    let u1 = _mm_adds_epi16(t8a, t10a);
    let u14 = _mm_adds_epi16(t9a, t11a);
    let t10 = _mm_subs_epi16(t8a, t10a);
    let t11 = _mm_subs_epi16(t9a, t11a);
    let u2 = _mm_adds_epi16(t12, t14);
    let u13 = _mm_adds_epi16(t13, t15);
    let t14a = _mm_subs_epi16(t12, t14);
    let t15a = _mm_subs_epi16(t13, t15);

    let (u7, u8) = mul2_x8(t2a, t3a, 2896, 2896, 2896, -2896);
    let (u4, u11) = mul2_x8(t6, t7, 2896, 2896, 2896, -2896);
    let (u6, u9) = mul2_x8(t10, t11, 2896, 2896, 2896, -2896);
    let (u5, u10) = mul2_x8(t14a, t15a, 2896, 2896, 2896, -2896);

    [
        u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15,
    ]
}

#[target_feature(enable = "ssse3")]
#[inline]
fn adst16_sonlandir(u: [__m128i; 16], shift_mul: i16, flip: bool) -> [__m128i; 16] {
    let p = _mm_set1_epi16(shift_mul);
    let n = _mm_set1_epi16(-shift_mul);
    let mut o = [_mm_setzero_si128(); 16];
    let mut i = 0;
    while i < 16 {
        o[i] = _mm_mulhrs_epi16(u[i], if i & 1 == 0 { p } else { n });
        i += 1;
    }
    if flip {
        o.reverse();
    }
    o
}

#[target_feature(enable = "ssse3")]
#[inline]
fn iidtx16_gecis1(c: [__m128i; 16], shift: u32) -> [__m128i; 16] {
    let k = _mm_set1_epi16(1697 * 16);
    let yarim = _mm_set1_epi16(16384);
    let mut o = [_mm_setzero_si128(); 16];
    let mut i = 0;
    while i < 16 {
        let t = _mm_mulhrs_epi16(c[i], k);
        o[i] = if shift == 1 {
            _mm_adds_epi16(c[i], _mm_mulhrs_epi16(t, yarim))
        } else {
            savg_epi16(c[i], _mm_srai_epi16::<1>(t))
        };
        i += 1;
    }
    o
}

#[target_feature(enable = "ssse3")]
#[inline]
fn iidtx16_gecis2_x4(c: [__m128i; 16]) -> [__m128i; 16] {
    let k = _mm_set1_epi16(1697 * 16);
    let r4 = _mm_set1_epi16(2048);
    let mut o = [_mm_setzero_si128(); 16];
    let mut i = 0;
    while i < 16 {
        let t = _mm_mulhrs_epi16(c[i], k);
        let s = _mm_adds_epi16(_mm_adds_epi16(c[i], c[i]), t);
        o[i] = _mm_mulhrs_epi16(s, r4);
        i += 1;
    }
    o
}

#[target_feature(enable = "ssse3")]
#[inline]
fn iidtx16_gecis2_x8(c: [__m128i; 16]) -> [__m128i; 16] {
    let k = _mm_set1_epi16(1697 * 16);
    let r4 = _mm_set1_epi16(2048);
    let mut o = [_mm_setzero_si128(); 16];
    let mut i = 0;
    while i < 16 {
        let t = _mm_mulhrs_epi16(c[i], k);
        let s = _mm_adds_epi16(_mm_adds_epi16(c[i], c[i]), t);
        o[i] = _mm_mulhrs_epi16(s, r4);
        i += 1;
    }
    o
}

#[target_feature(enable = "ssse3")]
#[inline]
fn gecis1_16_x4(c: [__m128i; 16], kind: Tx1dType, shift: u32, yarim_sifir: bool) -> [__m128i; 16] {
    let shift_mul: i16 = if shift == 1 { 16384 } else { 8192 };
    match kind {
        Tx1dType::Identity => iidtx16_gecis1(c, shift),
        Tx1dType::Dct => {
            let mut o = if yarim_sifir {
                idct16_yarim_x4([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
            } else {
                idct16_x4(c)
            };
            let m = _mm_set1_epi16(shift_mul);
            let mut i = 0;
            while i < 16 {
                o[i] = _mm_mulhrs_epi16(o[i], m);
                i += 1;
            }
            o
        }
        Tx1dType::Adst | Tx1dType::FlipAdst => adst16_sonlandir(
            iadst16_ham_x4(c),
            shift_mul,
            matches!(kind, Tx1dType::FlipAdst),
        ),
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn gecis1_16_x8(c: [__m128i; 16], kind: Tx1dType, shift: u32, yarim_sifir: bool) -> [__m128i; 16] {
    let shift_mul: i16 = if shift == 1 { 16384 } else { 8192 };
    match kind {
        Tx1dType::Identity => iidtx16_gecis1(c, shift),
        Tx1dType::Dct => {
            let mut o = if yarim_sifir {
                idct16_yarim_x8([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
            } else {
                idct16_x8(c)
            };
            let m = _mm_set1_epi16(shift_mul);
            let mut i = 0;
            while i < 16 {
                o[i] = _mm_mulhrs_epi16(o[i], m);
                i += 1;
            }
            o
        }
        Tx1dType::Adst | Tx1dType::FlipAdst => adst16_sonlandir(
            iadst16_ham_x8(c),
            shift_mul,
            matches!(kind, Tx1dType::FlipAdst),
        ),
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn gecis2_16_x4(c: [__m128i; 16], kind: Tx1dType, yarim_sifir: bool) -> [__m128i; 16] {
    match kind {
        Tx1dType::Identity => iidtx16_gecis2_x4(c),
        Tx1dType::Dct => {
            let mut o = if yarim_sifir {
                idct16_yarim_x4([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
            } else {
                idct16_x4(c)
            };
            let m = _mm_set1_epi16(2048);
            let mut i = 0;
            while i < 16 {
                o[i] = _mm_mulhrs_epi16(o[i], m);
                i += 1;
            }
            o
        }
        Tx1dType::Adst | Tx1dType::FlipAdst => {
            adst16_sonlandir(iadst16_ham_x4(c), 2048, matches!(kind, Tx1dType::FlipAdst))
        }
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn gecis2_16_x8(c: [__m128i; 16], kind: Tx1dType, yarim_sifir: bool) -> [__m128i; 16] {
    match kind {
        Tx1dType::Identity => iidtx16_gecis2_x8(c),
        Tx1dType::Dct => {
            let mut o = if yarim_sifir {
                idct16_yarim_x8([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
            } else {
                idct16_x8(c)
            };
            let m = _mm_set1_epi16(2048);
            let mut i = 0;
            while i < 16 {
                o[i] = _mm_mulhrs_epi16(o[i], m);
                i += 1;
            }
            o
        }
        Tx1dType::Adst | Tx1dType::FlipAdst => {
            adst16_sonlandir(iadst16_ham_x8(c), 2048, matches!(kind, Tx1dType::FlipAdst))
        }
    }
}

// ---------------------------------------------------------------------------
// 2B blok gövdeleri (`w, h ≤ 16`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
#[inline]
fn blok_4x4<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<4, 4, 0, false>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let giris = [
        yukle_katsayi4(coeff, 0),
        yukle_katsayi4(coeff, 4),
        yukle_katsayi4(coeff, 8),
        yukle_katsayi4(coeff, 12),
    ];
    chunk_mut::<i32, 16>(coeff, 0).fill(0);
    let p1 = gecis1_4_x4(giris, yatay, 0);
    let t = devrik_4x4(p1);
    let cikis = gecis2_4_x4(t, dikey);
    ekle_yaz_4x4(dst, stride, 0, cikis[0], cikis[1], cikis[2], cikis[3]);
}

#[target_feature(enable = "ssse3")]
#[inline]
fn blok_4x8<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<4, 8, 0, true>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let giris = [
        olcekle_rect2(yukle_katsayi8(coeff, 0)),
        olcekle_rect2(yukle_katsayi8(coeff, 8)),
        olcekle_rect2(yukle_katsayi8(coeff, 16)),
        olcekle_rect2(yukle_katsayi8(coeff, 24)),
    ];
    chunk_mut::<i32, 32>(coeff, 0).fill(0);
    let p1 = gecis1_4_x8(giris, yatay, 0);
    let t = devrik_4x8(p1);
    let cikis = gecis2_8_x4(t, dikey);
    ekle_yaz_4x4(dst, stride, 0, cikis[0], cikis[1], cikis[2], cikis[3]);
    ekle_yaz_4x4(dst, stride, 4, cikis[4], cikis[5], cikis[6], cikis[7]);
}

#[target_feature(enable = "ssse3")]
#[inline]
fn blok_8x4<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<8, 4, 0, true>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    if matches!(yatay, Tx1dType::Identity) {
        let mut c01 = olcekle_rect2(yukle_katsayi8(coeff, 0));
        let mut c23 = olcekle_rect2(yukle_katsayi8(coeff, 8));
        let mut c45 = olcekle_rect2(yukle_katsayi8(coeff, 16));
        let mut c67 = olcekle_rect2(yukle_katsayi8(coeff, 24));
        chunk_mut::<i32, 32>(coeff, 0).fill(0);
        c01 = _mm_adds_epi16(c01, c01);
        c23 = _mm_adds_epi16(c23, c23);
        c45 = _mm_adds_epi16(c45, c45);
        c67 = _mm_adds_epi16(c67, c67);
        if matches!(dikey, Tx1dType::Identity) {
            let k = _mm_set1_epi16(13576);
            let m = _mm_set1_epi16(2048);
            c01 = _mm_mulhrs_epi16(_mm_adds_epi16(c01, _mm_mulhrs_epi16(c01, k)), m);
            c23 = _mm_mulhrs_epi16(_mm_adds_epi16(c23, _mm_mulhrs_epi16(c23, k)), m);
            c45 = _mm_mulhrs_epi16(_mm_adds_epi16(c45, _mm_mulhrs_epi16(c45, k)), m);
            c67 = _mm_mulhrs_epi16(_mm_adds_epi16(c67, _mm_mulhrs_epi16(c67, k)), m);
        }
        let a0 = _mm_unpacklo_epi16(c01, c23);
        let a1 = _mm_unpackhi_epi16(c01, c23);
        let b0 = _mm_unpacklo_epi16(a0, a1);
        let b1 = _mm_unpackhi_epi16(a0, a1);
        let a2 = _mm_unpacklo_epi16(c45, c67);
        let a3 = _mm_unpackhi_epi16(c45, c67);
        let d0 = _mm_unpacklo_epi16(a2, a3);
        let d1 = _mm_unpackhi_epi16(a2, a3);
        let t = [
            _mm_unpacklo_epi64(b0, d0),
            _mm_unpackhi_epi64(b0, d0),
            _mm_unpacklo_epi64(b1, d1),
            _mm_unpackhi_epi64(b1, d1),
        ];
        let cikis = if matches!(dikey, Tx1dType::Identity) {
            t
        } else {
            gecis2_4_x8(t, dikey)
        };
        ekle_yaz_8x2(dst, stride, 0, cikis[0], cikis[1]);
        ekle_yaz_8x2(dst, stride, 2, cikis[2], cikis[3]);
        return;
    }
    let giris = [
        olcekle_rect2(yukle_katsayi4(coeff, 0)),
        olcekle_rect2(yukle_katsayi4(coeff, 4)),
        olcekle_rect2(yukle_katsayi4(coeff, 8)),
        olcekle_rect2(yukle_katsayi4(coeff, 12)),
        olcekle_rect2(yukle_katsayi4(coeff, 16)),
        olcekle_rect2(yukle_katsayi4(coeff, 20)),
        olcekle_rect2(yukle_katsayi4(coeff, 24)),
        olcekle_rect2(yukle_katsayi4(coeff, 28)),
    ];
    chunk_mut::<i32, 32>(coeff, 0).fill(0);
    let p1 = gecis1_8_x4(giris, yatay, 0);
    let t = devrik_8x4(p1);
    let cikis = gecis2_4_x8(t, dikey);
    ekle_yaz_8x2(dst, stride, 0, cikis[0], cikis[1]);
    ekle_yaz_8x2(dst, stride, 2, cikis[2], cikis[3]);
}

#[target_feature(enable = "ssse3")]
#[inline]
fn blok_8x8<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<8, 8, 1, false>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let giris = [
        yukle_katsayi8(coeff, 0),
        yukle_katsayi8(coeff, 8),
        yukle_katsayi8(coeff, 16),
        yukle_katsayi8(coeff, 24),
        yukle_katsayi8(coeff, 32),
        yukle_katsayi8(coeff, 40),
        yukle_katsayi8(coeff, 48),
        yukle_katsayi8(coeff, 56),
    ];
    chunk_mut::<i32, 64>(coeff, 0).fill(0);
    let p1 = gecis1_8_x8(giris, yatay, 1);
    let t = devrik_8x8(p1);
    let cikis = gecis2_8_x8(t, dikey);
    ekle_yaz_8x2(dst, stride, 0, cikis[0], cikis[1]);
    ekle_yaz_8x2(dst, stride, 2, cikis[2], cikis[3]);
    ekle_yaz_8x2(dst, stride, 4, cikis[4], cikis[5]);
    ekle_yaz_8x2(dst, stride, 6, cikis[6], cikis[7]);
}

#[target_feature(enable = "ssse3")]
#[inline]
fn blok_4x16<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<4, 16, 1, false>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let yarim = matches!(txtp, TxType::DctDct) && eob <= 28;
    let ust = [
        yukle_katsayi8(coeff, 0),
        yukle_katsayi8(coeff, 16),
        yukle_katsayi8(coeff, 32),
        yukle_katsayi8(coeff, 48),
    ];
    let t_ust = devrik_4x8(gecis1_4_x8(ust, yatay, 1));
    let mut t = [_mm_setzero_si128(); 16];
    let mut i = 0;
    while i < 8 {
        t[i] = t_ust[i];
        i += 1;
    }
    if !yarim {
        let alt = [
            yukle_katsayi8(coeff, 8),
            yukle_katsayi8(coeff, 24),
            yukle_katsayi8(coeff, 40),
            yukle_katsayi8(coeff, 56),
        ];
        let t_alt = devrik_4x8(gecis1_4_x8(alt, yatay, 1));
        let mut j = 0;
        while j < 8 {
            t[8 + j] = t_alt[j];
            j += 1;
        }
    }
    chunk_mut::<i32, 64>(coeff, 0).fill(0);
    let cikis = gecis2_16_x4(t, dikey, yarim);
    ekle_yaz_4x4(dst, stride, 0, cikis[0], cikis[1], cikis[2], cikis[3]);
    ekle_yaz_4x4(dst, stride, 4, cikis[4], cikis[5], cikis[6], cikis[7]);
    ekle_yaz_4x4(dst, stride, 8, cikis[8], cikis[9], cikis[10], cikis[11]);
    ekle_yaz_4x4(dst, stride, 12, cikis[12], cikis[13], cikis[14], cikis[15]);
}

#[target_feature(enable = "ssse3")]
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
        giris[x] = yukle_katsayi4(coeff, x * 4);
        x += 1;
    }
    chunk_mut::<i32, 64>(coeff, 0).fill(0);
    let p1 = gecis1_16_x4(giris, yatay, 1, yarim);
    let t_sol = devrik_8x4([p1[0], p1[1], p1[2], p1[3], p1[4], p1[5], p1[6], p1[7]]);
    let t_sag = devrik_8x4([p1[8], p1[9], p1[10], p1[11], p1[12], p1[13], p1[14], p1[15]]);
    let c_sol = gecis2_4_x8(t_sol, dikey);
    let c_sag = gecis2_4_x8(t_sag, dikey);
    let mut y = 0;
    while y < 4 {
        ekle_yaz_16x1(dst, y * stride, c_sol[y], c_sag[y]);
        y += 1;
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn blok_8x16<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<8, 16, 1, true>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let yarim = matches!(txtp, TxType::DctDct) && eob <= 42;
    let mut ust = [_mm_setzero_si128(); 8];
    let mut x = 0;
    while x < 8 {
        ust[x] = olcekle_rect2(yukle_katsayi8(coeff, x * 16));
        x += 1;
    }
    let t_ust = devrik_8x8(gecis1_8_x8(ust, yatay, 1));
    let mut t = [_mm_setzero_si128(); 16];
    let mut i = 0;
    while i < 8 {
        t[i] = t_ust[i];
        i += 1;
    }
    if !yarim {
        let mut alt = [_mm_setzero_si128(); 8];
        let mut x2 = 0;
        while x2 < 8 {
            alt[x2] = olcekle_rect2(yukle_katsayi8(coeff, x2 * 16 + 8));
            x2 += 1;
        }
        let t_alt = devrik_8x8(gecis1_8_x8(alt, yatay, 1));
        let mut j = 0;
        while j < 8 {
            t[8 + j] = t_alt[j];
            j += 1;
        }
    }
    chunk_mut::<i32, 128>(coeff, 0).fill(0);
    let cikis = gecis2_16_x8(t, dikey, yarim);
    let mut y = 0;
    while y < 16 {
        ekle_yaz_8x2(dst, stride, y, cikis[y], cikis[y + 1]);
        y += 2;
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn blok_16x8<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<16, 8, 1, true>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let yarim = matches!(txtp, TxType::DctDct) && eob <= 42;
    let mut giris = [_mm_setzero_si128(); 16];
    let sinir = if yarim { 8 } else { 16 };
    let mut x = 0;
    while x < sinir {
        giris[x] = olcekle_rect2(yukle_katsayi8(coeff, x * 8));
        x += 1;
    }
    chunk_mut::<i32, 128>(coeff, 0).fill(0);
    let p1 = gecis1_16_x8(giris, yatay, 1, yarim);
    let t_sol = devrik_8x8([p1[0], p1[1], p1[2], p1[3], p1[4], p1[5], p1[6], p1[7]]);
    let t_sag = devrik_8x8([p1[8], p1[9], p1[10], p1[11], p1[12], p1[13], p1[14], p1[15]]);
    let c_sol = gecis2_8_x8(t_sol, dikey);
    let c_sag = gecis2_8_x8(t_sag, dikey);
    let mut y = 0;
    while y < 8 {
        ekle_yaz_16x1(dst, y * stride, c_sol[y], c_sag[y]);
        y += 1;
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn blok_16x16<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<16, 16, 2, false>(dst, stride, coeff);
        return;
    }
    if matches!(txtp, TxType::Idtx) {
        let k = _mm_set1_epi16(1697 * 16);
        let r4 = _mm_set1_epi16(2048);
        let olcekle = |v: __m128i| -> __m128i {
            let t1 = _mm_mulhrs_epi16(v, k);
            let p1 = savg_epi16(v, _mm_srai_epi16::<1>(t1));
            let t2 = _mm_mulhrs_epi16(p1, k);
            let s2 = _mm_adds_epi16(_mm_adds_epi16(p1, p1), t2);
            _mm_mulhrs_epi16(s2, r4)
        };
        let mut y_off = 0;
        while y_off < 16 {
            let mut sol = [_mm_setzero_si128(); 8];
            let mut sag = [_mm_setzero_si128(); 8];
            let mut x = 0;
            while x < 8 {
                sol[x] = olcekle(yukle_katsayi8(coeff, x * 16 + y_off));
                sag[x] = olcekle(yukle_katsayi8(coeff, (x + 8) * 16 + y_off));
                x += 1;
            }
            let c_sol = devrik_8x8(sol);
            let c_sag = devrik_8x8(sag);
            let mut dy = 0;
            while dy < 8 {
                ekle_yaz_16x1(dst, (y_off + dy) * stride, c_sol[dy], c_sag[dy]);
                dy += 1;
            }
            y_off += 8;
        }
        chunk_mut::<i32, 256>(coeff, 0).fill(0);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let yarim = matches!(txtp, TxType::DctDct) && eob <= 35;
    let x_sinir = if yarim { 8 } else { 16 };

    let mut ust_in = [_mm_setzero_si128(); 16];
    let mut x = 0;
    while x < x_sinir {
        ust_in[x] = yukle_katsayi8(coeff, x * 16);
        x += 1;
    }
    let p1_ust = gecis1_16_x8(ust_in, yatay, 2, yarim);
    let tl = devrik_8x8([
        p1_ust[0], p1_ust[1], p1_ust[2], p1_ust[3], p1_ust[4], p1_ust[5], p1_ust[6], p1_ust[7],
    ]);
    let tr = devrik_8x8([
        p1_ust[8], p1_ust[9], p1_ust[10], p1_ust[11], p1_ust[12], p1_ust[13], p1_ust[14],
        p1_ust[15],
    ]);

    let mut sutun_sol = [_mm_setzero_si128(); 16];
    let mut sutun_sag = [_mm_setzero_si128(); 16];
    let mut i = 0;
    while i < 8 {
        sutun_sol[i] = tl[i];
        sutun_sag[i] = tr[i];
        i += 1;
    }

    if !yarim {
        let mut alt_in = [_mm_setzero_si128(); 16];
        let mut x2 = 0;
        while x2 < 16 {
            alt_in[x2] = yukle_katsayi8(coeff, x2 * 16 + 8);
            x2 += 1;
        }
        let p1_alt = gecis1_16_x8(alt_in, yatay, 2, false);
        let bl = devrik_8x8([
            p1_alt[0], p1_alt[1], p1_alt[2], p1_alt[3], p1_alt[4], p1_alt[5], p1_alt[6], p1_alt[7],
        ]);
        let br = devrik_8x8([
            p1_alt[8], p1_alt[9], p1_alt[10], p1_alt[11], p1_alt[12], p1_alt[13], p1_alt[14],
            p1_alt[15],
        ]);
        let mut j = 0;
        while j < 8 {
            sutun_sol[8 + j] = bl[j];
            sutun_sag[8 + j] = br[j];
            j += 1;
        }
    }

    chunk_mut::<i32, 256>(coeff, 0).fill(0);
    let c_sol = gecis2_16_x8(sutun_sol, dikey, yarim);
    let c_sag = gecis2_16_x8(sutun_sag, dikey, yarim);
    let mut y = 0;
    while y < 16 {
        ekle_yaz_16x1(dst, y * stride, c_sol[y], c_sag[y]);
        y += 1;
    }
}

/// `(TX, TXTP)` çiftine özelleştirilmiş 8 bpc SSSE3 ters dönüşüm çekirdek gövdesi.
#[target_feature(enable = "ssse3")]
#[inline(never)]
fn itx8_ssse3_cekirdek<const TX: u8, const TXTP: u8>(
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

/// `(TX, TXTP)` çiftinin sözleşme denetimli güvenli 8 bpc SSSE3 yuva girişi.
fn itx8_ssse3_giris<const TX: u8, const TXTP: u8>(
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir (`itx8_ssse3_fn`);
    // belirteç SSSE3 tespiti + maske izninin kanıtıdır. (2) sınır: `itx8_sozlesmesi`
    // `dst`'nin `(h − 1)·dst_stride + w` baytı ve `coeff`'un `1024` öğeyi kapsadığını
    // garanti eder; bütün bellek erişimleri sabit boyutlu `chunk`/`chunk_mut` dizileri
    // üzerindendir. (3) hizalama/takma ad: `dst` ve `coeff` ayrık `&mut` referanslardır,
    // hizasız erişim kullanılır.
    unsafe {
        itx8_ssse3_cekirdek::<TX, TXTP>(dst, dst_stride, coeff, eob);
    }
}

macro_rules! esle_16_tur {
    ($tx:ident, $txtp:expr) => {
        match $txtp {
            TxType::DctDct => {
                Some(itx8_ssse3_giris::<{ TxSize::$tx as u8 }, { TxType::DctDct as u8 }>)
            }
            TxType::AdstDct => {
                Some(itx8_ssse3_giris::<{ TxSize::$tx as u8 }, { TxType::AdstDct as u8 }>)
            }
            TxType::DctAdst => {
                Some(itx8_ssse3_giris::<{ TxSize::$tx as u8 }, { TxType::DctAdst as u8 }>)
            }
            TxType::AdstAdst => {
                Some(itx8_ssse3_giris::<{ TxSize::$tx as u8 }, { TxType::AdstAdst as u8 }>)
            }
            TxType::FlipadstDct => {
                Some(itx8_ssse3_giris::<{ TxSize::$tx as u8 }, { TxType::FlipadstDct as u8 }>)
            }
            TxType::DctFlipadst => {
                Some(itx8_ssse3_giris::<{ TxSize::$tx as u8 }, { TxType::DctFlipadst as u8 }>)
            }
            TxType::FlipadstFlipadst => {
                Some(itx8_ssse3_giris::<{ TxSize::$tx as u8 }, { TxType::FlipadstFlipadst as u8 }>)
            }
            TxType::AdstFlipadst => {
                Some(itx8_ssse3_giris::<{ TxSize::$tx as u8 }, { TxType::AdstFlipadst as u8 }>)
            }
            TxType::FlipadstAdst => {
                Some(itx8_ssse3_giris::<{ TxSize::$tx as u8 }, { TxType::FlipadstAdst as u8 }>)
            }
            TxType::Idtx => Some(itx8_ssse3_giris::<{ TxSize::$tx as u8 }, { TxType::Idtx as u8 }>),
            TxType::VDct => Some(itx8_ssse3_giris::<{ TxSize::$tx as u8 }, { TxType::VDct as u8 }>),
            TxType::HDct => Some(itx8_ssse3_giris::<{ TxSize::$tx as u8 }, { TxType::HDct as u8 }>),
            TxType::VAdst => {
                Some(itx8_ssse3_giris::<{ TxSize::$tx as u8 }, { TxType::VAdst as u8 }>)
            }
            TxType::HAdst => {
                Some(itx8_ssse3_giris::<{ TxSize::$tx as u8 }, { TxType::HAdst as u8 }>)
            }
            TxType::VFlipadst => {
                Some(itx8_ssse3_giris::<{ TxSize::$tx as u8 }, { TxType::VFlipadst as u8 }>)
            }
            TxType::HFlipadst => {
                Some(itx8_ssse3_giris::<{ TxSize::$tx as u8 }, { TxType::HFlipadst as u8 }>)
            }
            TxType::WhtWht => None,
        }
    };
}

/// `w, h ≤ 16` boyutlu 140 adet 8 bpc WHT-dışı SSSE3 `itxfm_add[tx][txtp]` yuvası
/// (yalnız [`Ssse3Token`] ile).
#[must_use]
pub(crate) const fn itx8_ssse3_fn(_token: Ssse3Token, tx: TxSize, txtp: TxType) -> Option<ItxFn> {
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
                Some(itx8_ssse3_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::DctDct as u8 }>)
            }
            TxType::AdstDct => {
                Some(itx8_ssse3_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::AdstDct as u8 }>)
            }
            TxType::DctAdst => {
                Some(itx8_ssse3_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::DctAdst as u8 }>)
            }
            TxType::AdstAdst => {
                Some(itx8_ssse3_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::AdstAdst as u8 }>)
            }
            TxType::FlipadstDct => {
                Some(itx8_ssse3_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::FlipadstDct as u8 }>)
            }
            TxType::DctFlipadst => {
                Some(itx8_ssse3_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::DctFlipadst as u8 }>)
            }
            TxType::FlipadstFlipadst => Some(
                itx8_ssse3_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::FlipadstFlipadst as u8 }>,
            ),
            TxType::AdstFlipadst => {
                Some(itx8_ssse3_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::AdstFlipadst as u8 }>)
            }
            TxType::FlipadstAdst => {
                Some(itx8_ssse3_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::FlipadstAdst as u8 }>)
            }
            TxType::Idtx => {
                Some(itx8_ssse3_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::Idtx as u8 }>)
            }
            TxType::VDct => {
                Some(itx8_ssse3_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::VDct as u8 }>)
            }
            TxType::HDct => {
                Some(itx8_ssse3_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::HDct as u8 }>)
            }
            _ => None,
        },
        _ => None,
    }
}
