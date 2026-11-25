//! 10 bpc WHT-dışı ters dönüşüm (`w, h ≤ 16`, 140 kayıt) x86_64 SSE4.1 çekirdekleri — P43 Adım 3.
//!
//! Bu modül dav1d'nin `src/x86/itx.h:178-199` ve `src/x86/itx16_sse.asm` dosyalarındaki
//! `w, h ≤ 16` boyutlu **140** adet **yalnız 10 bpc** (`BitDepth::Ten`) SSE4.1 `itxfm_add`
//! kaydını **saf Rust intrinsics** ile gerçekler; dış `.asm`/`.S`, `asm!` ve `global_asm!`
//! yoktur (CEK-02). `12 bpc` (`BitDepth::Twelve`) dav1d'de de SSE4.1'de skalerde kalır:
//!
//! - `TX_4X4`, `RTX_4X8`, `RTX_8X4`, `TX_8X8`, `RTX_4X16`, `RTX_16X4`, `RTX_8X16`,
//!   `RTX_16X8`: 16 WHT-dışı `TxType` türünün tamamı (`8 × 16 = 128`),
//! - `TX_16X16`: desteklenen 12 `TxType` (`DCT_DCT`, `ADST_DCT`, `DCT_ADST`,
//!   `ADST_ADST`, `FLIPADST_DCT`, `DCT_FLIPADST`, `FLIPADST_FLIPADST`,
//!   `ADST_FLIPADST`, `FLIPADST_ADST`, `IDTX`, `V_DCT`, `H_DCT`).
//!
//! ## Sözleşme, iki geçişli 10 bpc aritmetik ve ISA saflığı (P34 / P43)
//!
//! - Girişte O(1) sözleşme denetimi ([`itx16_sozlesmesi`]) ile `(tx, txtp, bit_depth == Ten)`
//!   uyumu aranır; sözleşme dışı çağrıda (`12 bpc`, `8 bpc`, tek bayt `dst_stride` vb.)
//!   paniklemeden [`itx_skaler_devret`] üzerinden sayaç artırılır ve ilgili bit derinliğinin
//!   skaler monomorfuna devredilir.
//! - 10 bpc katsayıları `[-131072, +131071]` (18 bit işaretli) aralığında olduğundan Pass 1
//!   öncesi `i16`'ya daraltılmaz; `4 × i32` şeritlerinde (`_mm_mullo_epi32` + `_mm_max_epi32` /
//!   `_mm_min_epi32` `[-131072, +131071]` `row_clip`) çalışır. Pass 1 sonunda `(v + rnd) >> shift`
//!   uygulanıp `_mm_packs_epi32` ile `[-32768, +32767]` (`col_clip`) aralığına doyurularak
//!   `i16`'ya indirilir; Pass 2 `i16` kelebek aritmetiğinde tamamlanıp `[0, 1023]` aralığına
//!   kırpılarak `u16` hedef düzleme eklenir.
//! - Bütün çekirdek ve yardımcılar yalnız `#[target_feature(enable = "sse4.1")]` taşır;
//!   başka `dsp/x86/*` modülleriyle `#[target_feature]` yardımcı paylaşmaz (P34) ve
//!   SSE4.2+ ya da VEX/EVEX yönergesi üretmez.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_adds_epi16, _mm_cvtsi64_si128, _mm_madd_epi16, _mm_max_epi16,
    _mm_max_epi32, _mm_min_epi16, _mm_min_epi32, _mm_mulhrs_epi16, _mm_mullo_epi32,
    _mm_packs_epi32, _mm_set1_epi16, _mm_set1_epi32, _mm_setr_epi32, _mm_setzero_si128,
    _mm_srai_epi32, _mm_srli_si128, _mm_sub_epi32, _mm_subs_epi16, _mm_unpackhi_epi16,
    _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi16, _mm_unpacklo_epi32,
    _mm_unpacklo_epi64,
};

use super::mem::{chunk, chunk_mut, load_u8x16, store_u8x8, store_u8x16};
use crate::cpu::Sse41Token;
use crate::dsp::itx::{ItxFn, TxSize, TxType, itx_skaler_devret, itx16_sozlesmesi};
use crate::dsp::scalar::itx_1d::Tx1dType;
use crate::frame::BitDepth;

/// İki `i16` çarpanını `_mm_madd_epi16` için `i32` şeritlerine paketler (`[c0, c1, ...]`).
#[inline(always)]
const fn cift_katsayi(c0: i16, c1: i16) -> i32 {
    (c0 as u16 as i32) | ((c1 as u16 as i32) << 16)
}

// ---------------------------------------------------------------------------
// Pass 1 (`i32` 4 şerit) temel yardımcıları
// ---------------------------------------------------------------------------

/// `4 × i32` katsayıyı `i16`'ya daraltmadan `i32` şeritlerine yükler.
#[target_feature(enable = "sse4.1")]
#[inline]
fn yukle_i32x4(coeff: &[i32], off: usize) -> __m128i {
    let c = chunk::<i32, 4>(coeff, off);
    _mm_setr_epi32(c[0], c[1], c[2], c[3])
}

/// 10 bpc `row_clip`: `[-131072, +131071]` (`18` bit işaretli).
#[target_feature(enable = "sse4.1")]
#[inline]
fn kirp_18b(v: __m128i) -> __m128i {
    _mm_min_epi32(
        _mm_max_epi32(v, _mm_set1_epi32(-131_072)),
        _mm_set1_epi32(131_071),
    )
}

/// `(v · 181 + 128) >> 8` (`rect2` ön ölçeklemesi ve 1B `×181` adımları).
#[target_feature(enable = "sse4.1")]
#[inline]
fn olcekle_181_i32(v: __m128i) -> __m128i {
    _mm_srai_epi32::<8>(_mm_add_epi32(
        _mm_mullo_epi32(v, _mm_set1_epi32(181)),
        _mm_set1_epi32(128),
    ))
}

/// Tekli `i32` ölçekleme: `(a · k + 2048) >> 12`.
#[target_feature(enable = "sse4.1")]
#[inline]
fn mul1_i32(a: __m128i, k: i32) -> __m128i {
    _mm_srai_epi32::<12>(_mm_add_epi32(
        _mm_mullo_epi32(a, _mm_set1_epi32(k)),
        _mm_set1_epi32(2048),
    ))
}

/// İkili `i32` döndürme: `((a·c0 + b·c1 + 2048) >> 12, (a·d0 + b·d1 + 2048) >> 12)`.
#[target_feature(enable = "sse4.1")]
#[inline]
fn mul2_i32(a: __m128i, b: __m128i, c0: i32, c1: i32, d0: i32, d1: i32) -> (__m128i, __m128i) {
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

/// Pass 1 çıkışı üzerinde `(v + rnd) >> SHIFT` yuvarlamalı sağ kaydırması.
#[target_feature(enable = "sse4.1")]
#[inline]
fn kaydir_i32<const SHIFT: u32>(v: __m128i) -> __m128i {
    match SHIFT {
        0 => v,
        1 => _mm_srai_epi32::<1>(_mm_add_epi32(v, _mm_set1_epi32(1))),
        2 => _mm_srai_epi32::<2>(_mm_add_epi32(v, _mm_set1_epi32(2))),
        _ => unreachable!(),
    }
}

/// `h == 4` için `N` sütunluk `i32` Pass 1 çıktısını kaydırıp alt 4 `i16` şeridine paketler (`col_clip`).
#[target_feature(enable = "sse4.1")]
#[inline]
fn paketle_x4<const N: usize, const SHIFT: u32>(c: [__m128i; N]) -> [__m128i; N] {
    let mut out = [_mm_setzero_si128(); N];
    let mut i = 0;
    while i < N {
        let s = kaydir_i32::<SHIFT>(c[i]);
        out[i] = _mm_packs_epi32(s, s);
        i += 1;
    }
    out
}

/// `h >= 8` için iki `4 × i32` yarı-şeridi kaydırıp tek `8 × i16` vektörüne paketler (`col_clip`).
#[target_feature(enable = "sse4.1")]
#[inline]
fn paketle_x8<const N: usize, const SHIFT: u32>(
    lo: [__m128i; N],
    hi: [__m128i; N],
) -> [__m128i; N] {
    let mut out = [_mm_setzero_si128(); N];
    let mut i = 0;
    while i < N {
        out[i] = _mm_packs_epi32(kaydir_i32::<SHIFT>(lo[i]), kaydir_i32::<SHIFT>(hi[i]));
        i += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// Pass 2 (`i16` 4/8 şerit) temel yardımcıları
// ---------------------------------------------------------------------------

/// 4 etkin `i16` şeridi için ikili döndürme:
/// `((a·c0 + b·c1 + 2048) >> 12, (a·d0 + b·d1 + 2048) >> 12)`.
#[target_feature(enable = "sse4.1")]
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
#[target_feature(enable = "sse4.1")]
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
// Hedef `u16` düzlem piksel ekleme + `[0, 1023]` doyurmalı yazma yardımcıları
// ---------------------------------------------------------------------------

/// `4 × 2` bölgeye (satır başına `8` bayt) alt 4 şeritli iki satır `i16` artığı ekler ve yazar.
#[target_feature(enable = "sse4.1")]
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

/// Tek bir `8` piksellik (`16` bayt) satıra 8 şeritli `i16` artığı ekler ve yazar.
#[target_feature(enable = "sse4.1")]
#[inline]
fn ekle_yaz_8x1(dst: &mut [u8], off: usize, r: __m128i) {
    let d = load_u8x16(chunk::<u8, 16>(dst, off));
    let s = _mm_min_epi16(
        _mm_max_epi16(_mm_adds_epi16(d, r), _mm_setzero_si128()),
        _mm_set1_epi16(1023),
    );
    store_u8x16(chunk_mut(dst, off), s);
}

/// Tek bir `16` piksellik (`32` bayt) satıra sol/sağ 8 şeritli `i16` artığı ekler ve yazar.
#[target_feature(enable = "sse4.1")]
#[inline]
fn ekle_yaz_16x1(dst: &mut [u8], off: usize, r_lo: __m128i, r_hi: __m128i) {
    ekle_yaz_8x1(dst, off, r_lo);
    ekle_yaz_8x1(dst, off + 16, r_hi);
}

/// `DCT_DCT` + `eob == 0` DC-only hızlı yolu (`coeff[0] = 0`; `coeff[1..]` korunur).
#[target_feature(enable = "sse4.1")]
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
    let dc_v = _mm_set1_epi16(dc.clamp(-16_384, 16_384) as i16);

    match W {
        4 => {
            let mut y = 0;
            while y < H {
                ekle_yaz_4x2(dst, stride, y, dc_v, dc_v);
                y += 2;
            }
        }
        8 => {
            let mut y = 0;
            while y < H {
                ekle_yaz_8x1(dst, y * stride, dc_v);
                y += 1;
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
// Matris devrikleme (transpose) yardımcıları (`i16`)
// ---------------------------------------------------------------------------

/// `4 × 4` devrikleme (alt 4 `i16` şeridi etkin).
#[target_feature(enable = "sse4.1")]
#[inline]
fn devrik_4x4(r: [__m128i; 4]) -> [__m128i; 4] {
    let a0 = _mm_unpacklo_epi16(r[0], r[1]);
    let a1 = _mm_unpacklo_epi16(r[2], r[3]);
    let b0 = _mm_unpacklo_epi32(a0, a1);
    let b1 = _mm_unpackhi_epi32(a0, a1);
    [b0, _mm_srli_si128::<8>(b0), b1, _mm_srli_si128::<8>(b1)]
}

/// `4 sütun (her biri 8 satır) → 8 satır (her biri alt 4 şerit)` devrikleme.
#[target_feature(enable = "sse4.1")]
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
#[target_feature(enable = "sse4.1")]
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
#[target_feature(enable = "sse4.1")]
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
// Pass 1: Yatay 1B dönüşümler (`i32` 4 şerit; 4, 8 ve 16 nokta)
// ---------------------------------------------------------------------------

#[target_feature(enable = "sse4.1")]
#[inline]
fn idct4_i32(c: [__m128i; 4]) -> [__m128i; 4] {
    let t0 = olcekle_181_i32(_mm_add_epi32(c[0], c[2]));
    let t1 = olcekle_181_i32(_mm_sub_epi32(c[0], c[2]));
    let (t2, t3) = mul2_i32(c[1], c[3], 1567, -3784, 3784, 1567);
    [
        kirp_18b(_mm_add_epi32(t0, t3)),
        kirp_18b(_mm_add_epi32(t1, t2)),
        kirp_18b(_mm_sub_epi32(t1, t2)),
        kirp_18b(_mm_sub_epi32(t0, t3)),
    ]
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn idct4_yarim_i32(c0: __m128i, c1: __m128i) -> [__m128i; 4] {
    let t0 = olcekle_181_i32(c0);
    let t2 = mul1_i32(c1, 1567);
    let t3 = mul1_i32(c1, 3784);
    [
        kirp_18b(_mm_add_epi32(t0, t3)),
        kirp_18b(_mm_add_epi32(t0, t2)),
        kirp_18b(_mm_sub_epi32(t0, t2)),
        kirp_18b(_mm_sub_epi32(t0, t3)),
    ]
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn iadst4_i32(c: [__m128i; 4], flip: bool) -> [__m128i; 4] {
    let rnd = _mm_set1_epi32(2048);
    let k1321 = _mm_set1_epi32(1321);
    let k2482 = _mm_set1_epi32(2482);
    let k3344 = _mm_set1_epi32(3344);
    let k3803 = _mm_set1_epi32(3803);

    let m1 = _mm_mullo_epi32(c[1], k3344);

    let s0 = _mm_add_epi32(
        _mm_add_epi32(_mm_mullo_epi32(c[0], k1321), _mm_mullo_epi32(c[2], k3803)),
        _mm_add_epi32(m1, _mm_mullo_epi32(c[3], k2482)),
    );
    let o0 = _mm_srai_epi32::<12>(_mm_add_epi32(s0, rnd));

    let s1 = _mm_add_epi32(
        _mm_sub_epi32(_mm_mullo_epi32(c[0], k2482), _mm_mullo_epi32(c[2], k1321)),
        _mm_sub_epi32(m1, _mm_mullo_epi32(c[3], k3803)),
    );
    let o1 = _mm_srai_epi32::<12>(_mm_add_epi32(s1, rnd));

    let d023 = _mm_add_epi32(_mm_sub_epi32(c[0], c[2]), c[3]);
    let o2 = _mm_srai_epi32::<12>(_mm_add_epi32(_mm_mullo_epi32(d023, k3344), rnd));

    let s3 = _mm_sub_epi32(
        _mm_add_epi32(_mm_mullo_epi32(c[0], k3803), _mm_mullo_epi32(c[2], k2482)),
        _mm_add_epi32(m1, _mm_mullo_epi32(c[3], k1321)),
    );
    let o3 = _mm_srai_epi32::<12>(_mm_add_epi32(s3, rnd));

    if flip {
        [o3, o2, o1, o0]
    } else {
        [o0, o1, o2, o3]
    }
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn gecis1_4_i32(c: [__m128i; 4], kind: Tx1dType) -> [__m128i; 4] {
    match kind {
        Tx1dType::Identity => [
            mul1_i32(c[0], 5793),
            mul1_i32(c[1], 5793),
            mul1_i32(c[2], 5793),
            mul1_i32(c[3], 5793),
        ],
        Tx1dType::Dct => idct4_i32(c),
        Tx1dType::Adst | Tx1dType::FlipAdst => iadst4_i32(c, matches!(kind, Tx1dType::FlipAdst)),
    }
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn idct8_i32(c: [__m128i; 8]) -> [__m128i; 8] {
    let [t0, t1, t2, t3] = idct4_i32([c[0], c[2], c[4], c[6]]);
    let (t4a, t7a) = mul2_i32(c[1], c[7], 799, -4017, 4017, 799);
    let (t5a, t6a) = mul2_i32(c[5], c[3], 3406, -2276, 2276, 3406);
    let t4 = kirp_18b(_mm_add_epi32(t4a, t5a));
    let t5a = kirp_18b(_mm_sub_epi32(t4a, t5a));
    let t7 = kirp_18b(_mm_add_epi32(t7a, t6a));
    let t6a = kirp_18b(_mm_sub_epi32(t7a, t6a));
    let t5 = olcekle_181_i32(_mm_sub_epi32(t6a, t5a));
    let t6 = olcekle_181_i32(_mm_add_epi32(t6a, t5a));
    [
        kirp_18b(_mm_add_epi32(t0, t7)),
        kirp_18b(_mm_add_epi32(t1, t6)),
        kirp_18b(_mm_add_epi32(t2, t5)),
        kirp_18b(_mm_add_epi32(t3, t4)),
        kirp_18b(_mm_sub_epi32(t3, t4)),
        kirp_18b(_mm_sub_epi32(t2, t5)),
        kirp_18b(_mm_sub_epi32(t1, t6)),
        kirp_18b(_mm_sub_epi32(t0, t7)),
    ]
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn idct8_yarim_i32(c: [__m128i; 4]) -> [__m128i; 8] {
    let [t0, t1, t2, t3] = idct4_yarim_i32(c[0], c[2]);
    let t4a = mul1_i32(c[1], 799);
    let t7a = mul1_i32(c[1], 4017);
    let t5a = mul1_i32(c[3], -2276);
    let t6a = mul1_i32(c[3], 3406);
    let t4 = kirp_18b(_mm_add_epi32(t4a, t5a));
    let t5a = kirp_18b(_mm_sub_epi32(t4a, t5a));
    let t7 = kirp_18b(_mm_add_epi32(t7a, t6a));
    let t6a = kirp_18b(_mm_sub_epi32(t7a, t6a));
    let t5 = olcekle_181_i32(_mm_sub_epi32(t6a, t5a));
    let t6 = olcekle_181_i32(_mm_add_epi32(t6a, t5a));
    [
        kirp_18b(_mm_add_epi32(t0, t7)),
        kirp_18b(_mm_add_epi32(t1, t6)),
        kirp_18b(_mm_add_epi32(t2, t5)),
        kirp_18b(_mm_add_epi32(t3, t4)),
        kirp_18b(_mm_sub_epi32(t3, t4)),
        kirp_18b(_mm_sub_epi32(t2, t5)),
        kirp_18b(_mm_sub_epi32(t1, t6)),
        kirp_18b(_mm_sub_epi32(t0, t7)),
    ]
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn iadst8_i32(c: [__m128i; 8], flip: bool) -> [__m128i; 8] {
    let (t0a, t1a) = mul2_i32(c[7], c[0], 4076, 401, 401, -4076);
    let (t2a, t3a) = mul2_i32(c[5], c[2], 3612, 1931, 1931, -3612);
    let (t4a, t5a) = mul2_i32(c[3], c[4], 2598, 3166, 3166, -2598);
    let (t6a, t7a) = mul2_i32(c[1], c[6], 1189, 3920, 3920, -1189);

    let t0 = kirp_18b(_mm_add_epi32(t0a, t4a));
    let t1 = kirp_18b(_mm_add_epi32(t1a, t5a));
    let t2 = kirp_18b(_mm_add_epi32(t2a, t6a));
    let t3 = kirp_18b(_mm_add_epi32(t3a, t7a));
    let t4 = kirp_18b(_mm_sub_epi32(t0a, t4a));
    let t5 = kirp_18b(_mm_sub_epi32(t1a, t5a));
    let t6 = kirp_18b(_mm_sub_epi32(t2a, t6a));
    let t7 = kirp_18b(_mm_sub_epi32(t3a, t7a));

    let (t4a, t5a) = mul2_i32(t4, t5, 3784, 1567, 1567, -3784);
    let (t6a, t7a) = mul2_i32(t7, t6, 3784, -1567, 1567, 3784);

    let z = _mm_setzero_si128();
    let o0 = kirp_18b(_mm_add_epi32(t0, t2));
    let o7 = _mm_sub_epi32(z, kirp_18b(_mm_add_epi32(t1, t3)));
    let t2b = kirp_18b(_mm_sub_epi32(t0, t2));
    let t3b = kirp_18b(_mm_sub_epi32(t1, t3));
    let o1 = _mm_sub_epi32(z, kirp_18b(_mm_add_epi32(t4a, t6a)));
    let o6 = kirp_18b(_mm_add_epi32(t5a, t7a));
    let t6b = kirp_18b(_mm_sub_epi32(t4a, t6a));
    let t7b = kirp_18b(_mm_sub_epi32(t5a, t7a));

    let o3 = _mm_sub_epi32(z, olcekle_181_i32(_mm_add_epi32(t2b, t3b)));
    let o4 = olcekle_181_i32(_mm_sub_epi32(t2b, t3b));
    let o2 = olcekle_181_i32(_mm_add_epi32(t6b, t7b));
    let o5 = _mm_sub_epi32(z, olcekle_181_i32(_mm_sub_epi32(t6b, t7b)));

    let mut o = [o0, o1, o2, o3, o4, o5, o6, o7];
    if flip {
        o.reverse();
    }
    o
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn gecis1_8_i32(c: [__m128i; 8], kind: Tx1dType) -> [__m128i; 8] {
    match kind {
        Tx1dType::Identity => {
            let mut o = c;
            let mut i = 0;
            while i < 8 {
                o[i] = _mm_add_epi32(c[i], c[i]);
                i += 1;
            }
            o
        }
        Tx1dType::Dct => idct8_i32(c),
        Tx1dType::Adst | Tx1dType::FlipAdst => iadst8_i32(c, matches!(kind, Tx1dType::FlipAdst)),
    }
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn idct16_ortak_son_i32(
    e: [__m128i; 8],
    [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a]: [__m128i; 8],
) -> [__m128i; 16] {
    let t8 = kirp_18b(_mm_add_epi32(t8a, t9a));
    let t9 = kirp_18b(_mm_sub_epi32(t8a, t9a));
    let t10 = kirp_18b(_mm_sub_epi32(t11a, t10a));
    let t11 = kirp_18b(_mm_add_epi32(t11a, t10a));
    let t12 = kirp_18b(_mm_add_epi32(t12a, t13a));
    let t13 = kirp_18b(_mm_sub_epi32(t12a, t13a));
    let t14 = kirp_18b(_mm_sub_epi32(t15a, t14a));
    let t15 = kirp_18b(_mm_add_epi32(t15a, t14a));

    let (t9a, t14a) = mul2_i32(t14, t9, 1567, -3784, 3784, 1567);
    let (t10a, t13a) = mul2_i32(t13, t10, -3784, -1567, 1567, -3784);

    let t8a = kirp_18b(_mm_add_epi32(t8, t11));
    let t9 = kirp_18b(_mm_add_epi32(t9a, t10a));
    let t10 = kirp_18b(_mm_sub_epi32(t9a, t10a));
    let t11a = kirp_18b(_mm_sub_epi32(t8, t11));
    let t12a = kirp_18b(_mm_sub_epi32(t15, t12));
    let t13 = kirp_18b(_mm_sub_epi32(t14a, t13a));
    let t14 = kirp_18b(_mm_add_epi32(t14a, t13a));
    let t15a = kirp_18b(_mm_add_epi32(t15, t12));

    let t10a = olcekle_181_i32(_mm_sub_epi32(t13, t10));
    let t13a = olcekle_181_i32(_mm_add_epi32(t13, t10));
    let t11 = olcekle_181_i32(_mm_sub_epi32(t12a, t11a));
    let t12 = olcekle_181_i32(_mm_add_epi32(t12a, t11a));

    [
        kirp_18b(_mm_add_epi32(e[0], t15a)),
        kirp_18b(_mm_add_epi32(e[1], t14)),
        kirp_18b(_mm_add_epi32(e[2], t13a)),
        kirp_18b(_mm_add_epi32(e[3], t12)),
        kirp_18b(_mm_add_epi32(e[4], t11)),
        kirp_18b(_mm_add_epi32(e[5], t10a)),
        kirp_18b(_mm_add_epi32(e[6], t9)),
        kirp_18b(_mm_add_epi32(e[7], t8a)),
        kirp_18b(_mm_sub_epi32(e[7], t8a)),
        kirp_18b(_mm_sub_epi32(e[6], t9)),
        kirp_18b(_mm_sub_epi32(e[5], t10a)),
        kirp_18b(_mm_sub_epi32(e[4], t11)),
        kirp_18b(_mm_sub_epi32(e[3], t12)),
        kirp_18b(_mm_sub_epi32(e[2], t13a)),
        kirp_18b(_mm_sub_epi32(e[1], t14)),
        kirp_18b(_mm_sub_epi32(e[0], t15a)),
    ]
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn idct16_i32(c: [__m128i; 16]) -> [__m128i; 16] {
    let e = idct8_i32([c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]]);
    let (t8a, t15a) = mul2_i32(c[1], c[15], 401, -4076, 4076, 401);
    let (t9a, t14a) = mul2_i32(c[9], c[7], 3166, -2598, 2598, 3166);
    let (t10a, t13a) = mul2_i32(c[5], c[11], 1931, -3612, 3612, 1931);
    let (t11a, t12a) = mul2_i32(c[13], c[3], 3920, -1189, 1189, 3920);
    idct16_ortak_son_i32(e, [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a])
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn idct16_yarim_i32(c: [__m128i; 8]) -> [__m128i; 16] {
    let e = idct8_yarim_i32([c[0], c[2], c[4], c[6]]);
    let t8a = mul1_i32(c[1], 401);
    let t15a = mul1_i32(c[1], 4076);
    let t9a = mul1_i32(c[7], -2598);
    let t14a = mul1_i32(c[7], 3166);
    let t10a = mul1_i32(c[5], 1931);
    let t13a = mul1_i32(c[5], 3612);
    let t11a = mul1_i32(c[3], -1189);
    let t12a = mul1_i32(c[3], 3920);
    idct16_ortak_son_i32(e, [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a])
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn iadst16_i32(c: [__m128i; 16], flip: bool) -> [__m128i; 16] {
    let (t0, t1) = mul2_i32(c[15], c[0], 4091, 201, 201, -4091);
    let (t2, t3) = mul2_i32(c[13], c[2], 3973, 995, 995, -3973);
    let (t4, t5) = mul2_i32(c[11], c[4], 3703, 1751, 1751, -3703);
    let (t6, t7) = mul2_i32(c[9], c[6], 3290, 2440, 2440, -3290);
    let (t8, t9) = mul2_i32(c[7], c[8], 2751, 3035, 3035, -2751);
    let (t10, t11) = mul2_i32(c[5], c[10], 2106, 3513, 3513, -2106);
    let (t12, t13) = mul2_i32(c[3], c[12], 1380, 3857, 3857, -1380);
    let (t14, t15) = mul2_i32(c[1], c[14], 601, 4052, 4052, -601);

    let t0a = kirp_18b(_mm_add_epi32(t0, t8));
    let t1a = kirp_18b(_mm_add_epi32(t1, t9));
    let t2a = kirp_18b(_mm_add_epi32(t2, t10));
    let t3a = kirp_18b(_mm_add_epi32(t3, t11));
    let t4a = kirp_18b(_mm_add_epi32(t4, t12));
    let t5a = kirp_18b(_mm_add_epi32(t5, t13));
    let t6a = kirp_18b(_mm_add_epi32(t6, t14));
    let t7a = kirp_18b(_mm_add_epi32(t7, t15));
    let t8a = kirp_18b(_mm_sub_epi32(t0, t8));
    let t9a = kirp_18b(_mm_sub_epi32(t1, t9));
    let t10a = kirp_18b(_mm_sub_epi32(t2, t10));
    let t11a = kirp_18b(_mm_sub_epi32(t3, t11));
    let t12a = kirp_18b(_mm_sub_epi32(t4, t12));
    let t13a = kirp_18b(_mm_sub_epi32(t5, t13));
    let t14a = kirp_18b(_mm_sub_epi32(t6, t14));
    let t15a = kirp_18b(_mm_sub_epi32(t7, t15));

    let (t8, t9) = mul2_i32(t8a, t9a, 4017, 799, 799, -4017);
    let (t10, t11) = mul2_i32(t10a, t11a, 2276, 3406, 3406, -2276);
    let (t12, t13) = mul2_i32(t13a, t12a, 4017, -799, 799, 4017);
    let (t14, t15) = mul2_i32(t15a, t14a, 2276, -3406, 3406, 2276);

    let t0 = kirp_18b(_mm_add_epi32(t0a, t4a));
    let t1 = kirp_18b(_mm_add_epi32(t1a, t5a));
    let t2 = kirp_18b(_mm_add_epi32(t2a, t6a));
    let t3 = kirp_18b(_mm_add_epi32(t3a, t7a));
    let t4 = kirp_18b(_mm_sub_epi32(t0a, t4a));
    let t5 = kirp_18b(_mm_sub_epi32(t1a, t5a));
    let t6 = kirp_18b(_mm_sub_epi32(t2a, t6a));
    let t7 = kirp_18b(_mm_sub_epi32(t3a, t7a));
    let t8a = kirp_18b(_mm_add_epi32(t8, t12));
    let t9a = kirp_18b(_mm_add_epi32(t9, t13));
    let t10a = kirp_18b(_mm_add_epi32(t10, t14));
    let t11a = kirp_18b(_mm_add_epi32(t11, t15));
    let t12a = kirp_18b(_mm_sub_epi32(t8, t12));
    let t13a = kirp_18b(_mm_sub_epi32(t9, t13));
    let t14a = kirp_18b(_mm_sub_epi32(t10, t14));
    let t15a = kirp_18b(_mm_sub_epi32(t11, t15));

    let (t4a, t5a) = mul2_i32(t4, t5, 3784, 1567, 1567, -3784);
    let (t6a, t7a) = mul2_i32(t7, t6, 3784, -1567, 1567, 3784);
    let (t12, t13) = mul2_i32(t12a, t13a, 3784, 1567, 1567, -3784);
    let (t14, t15) = mul2_i32(t15a, t14a, 3784, -1567, 1567, 3784);

    let z = _mm_setzero_si128();
    let o0 = kirp_18b(_mm_add_epi32(t0, t2));
    let o15 = _mm_sub_epi32(z, kirp_18b(_mm_add_epi32(t1, t3)));
    let t2a = kirp_18b(_mm_sub_epi32(t0, t2));
    let t3a = kirp_18b(_mm_sub_epi32(t1, t3));
    let o3 = _mm_sub_epi32(z, kirp_18b(_mm_add_epi32(t4a, t6a)));
    let o12 = kirp_18b(_mm_add_epi32(t5a, t7a));
    let t6 = kirp_18b(_mm_sub_epi32(t4a, t6a));
    let t7 = kirp_18b(_mm_sub_epi32(t5a, t7a));
    let o1 = _mm_sub_epi32(z, kirp_18b(_mm_add_epi32(t8a, t10a)));
    let o14 = kirp_18b(_mm_add_epi32(t9a, t11a));
    let t10 = kirp_18b(_mm_sub_epi32(t8a, t10a));
    let t11 = kirp_18b(_mm_sub_epi32(t9a, t11a));
    let o2 = kirp_18b(_mm_add_epi32(t12, t14));
    let o13 = _mm_sub_epi32(z, kirp_18b(_mm_add_epi32(t13, t15)));
    let t14a = kirp_18b(_mm_sub_epi32(t12, t14));
    let t15a = kirp_18b(_mm_sub_epi32(t13, t15));

    let o7 = _mm_sub_epi32(z, olcekle_181_i32(_mm_add_epi32(t2a, t3a)));
    let o8 = olcekle_181_i32(_mm_sub_epi32(t2a, t3a));
    let o4 = olcekle_181_i32(_mm_add_epi32(t6, t7));
    let o11 = _mm_sub_epi32(z, olcekle_181_i32(_mm_sub_epi32(t6, t7)));
    let o6 = olcekle_181_i32(_mm_add_epi32(t10, t11));
    let o9 = _mm_sub_epi32(z, olcekle_181_i32(_mm_sub_epi32(t10, t11)));
    let o5 = _mm_sub_epi32(z, olcekle_181_i32(_mm_add_epi32(t14a, t15a)));
    let o10 = olcekle_181_i32(_mm_sub_epi32(t14a, t15a));

    let mut o = [
        o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15,
    ];
    if flip {
        o.reverse();
    }
    o
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn gecis1_16_i32(c: [__m128i; 16], kind: Tx1dType, yarim_sifir: bool) -> [__m128i; 16] {
    match kind {
        Tx1dType::Identity => {
            let k = _mm_set1_epi32(5793);
            let rnd = _mm_set1_epi32(1024);
            let mut o = [_mm_setzero_si128(); 16];
            let mut i = 0;
            while i < 16 {
                o[i] = _mm_srai_epi32::<11>(_mm_add_epi32(_mm_mullo_epi32(c[i], k), rnd));
                i += 1;
            }
            o
        }
        Tx1dType::Dct => {
            if yarim_sifir {
                idct16_yarim_i32([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
            } else {
                idct16_i32(c)
            }
        }
        Tx1dType::Adst | Tx1dType::FlipAdst => iadst16_i32(c, matches!(kind, Tx1dType::FlipAdst)),
    }
}

// ---------------------------------------------------------------------------
// Pass 2: Dikey 1B dönüşümler (`i16` 4/8 şerit; 4, 8 ve 16 nokta)
// ---------------------------------------------------------------------------

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
#[inline]
fn iadst4_gecis2_x8(c: [__m128i; 4], flip: bool) -> [__m128i; 4] {
    let p02_lo = _mm_unpacklo_epi16(c[0], c[2]);
    let p02_hi = _mm_unpackhi_epi16(c[0], c[2]);
    let p13_lo = _mm_unpacklo_epi16(c[1], c[3]);
    let p13_hi = _mm_unpackhi_epi16(c[1], c[3]);
    let rnd = _mm_set1_epi32(34_816);

    let k0_02 = _mm_set1_epi32(cift_katsayi(1321, 3803));
    let k0_13 = _mm_set1_epi32(cift_katsayi(3344, 2482));
    let y0_lo = _mm_srai_epi32::<16>(_mm_add_epi32(
        _mm_add_epi32(_mm_madd_epi16(p02_lo, k0_02), _mm_madd_epi16(p13_lo, k0_13)),
        rnd,
    ));
    let y0_hi = _mm_srai_epi32::<16>(_mm_add_epi32(
        _mm_add_epi32(_mm_madd_epi16(p02_hi, k0_02), _mm_madd_epi16(p13_hi, k0_13)),
        rnd,
    ));

    let k1_02 = _mm_set1_epi32(cift_katsayi(2482, -1321));
    let k1_13 = _mm_set1_epi32(cift_katsayi(3344, -3803));
    let y1_lo = _mm_srai_epi32::<16>(_mm_add_epi32(
        _mm_add_epi32(_mm_madd_epi16(p02_lo, k1_02), _mm_madd_epi16(p13_lo, k1_13)),
        rnd,
    ));
    let y1_hi = _mm_srai_epi32::<16>(_mm_add_epi32(
        _mm_add_epi32(_mm_madd_epi16(p02_hi, k1_02), _mm_madd_epi16(p13_hi, k1_13)),
        rnd,
    ));

    let k2_02 = _mm_set1_epi32(cift_katsayi(3344, -3344));
    let k2_13 = _mm_set1_epi32(cift_katsayi(0, 3344));
    let y2_lo = _mm_srai_epi32::<16>(_mm_add_epi32(
        _mm_add_epi32(_mm_madd_epi16(p02_lo, k2_02), _mm_madd_epi16(p13_lo, k2_13)),
        rnd,
    ));
    let y2_hi = _mm_srai_epi32::<16>(_mm_add_epi32(
        _mm_add_epi32(_mm_madd_epi16(p02_hi, k2_02), _mm_madd_epi16(p13_hi, k2_13)),
        rnd,
    ));

    let k3_02 = _mm_set1_epi32(cift_katsayi(3803, 2482));
    let k3_13 = _mm_set1_epi32(cift_katsayi(-3344, -1321));
    let y3_lo = _mm_srai_epi32::<16>(_mm_add_epi32(
        _mm_add_epi32(_mm_madd_epi16(p02_lo, k3_02), _mm_madd_epi16(p13_lo, k3_13)),
        rnd,
    ));
    let y3_hi = _mm_srai_epi32::<16>(_mm_add_epi32(
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

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
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
            iadst4_gecis2_x4(c, matches!(kind, Tx1dType::FlipAdst))
        }
    }
}

#[target_feature(enable = "sse4.1")]
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
            iadst4_gecis2_x8(c, matches!(kind, Tx1dType::FlipAdst))
        }
    }
}

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
#[inline]
fn adst8_sonlandir_gecis2(u: [__m128i; 8], flip: bool) -> [__m128i; 8] {
    let p = _mm_set1_epi16(2048);
    let n = _mm_set1_epi16(-2048);
    let mut o = [
        _mm_mulhrs_epi16(u[0], p),
        _mm_mulhrs_epi16(u[1], n),
        _mm_mulhrs_epi16(u[2], p),
        _mm_mulhrs_epi16(u[3], n),
        _mm_mulhrs_epi16(u[4], p),
        _mm_mulhrs_epi16(u[5], n),
        _mm_mulhrs_epi16(u[6], p),
        _mm_mulhrs_epi16(u[7], n),
    ];
    if flip {
        o.reverse();
    }
    o
}

#[target_feature(enable = "sse4.1")]
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
            adst8_sonlandir_gecis2(iadst8_ham_x4(c), matches!(kind, Tx1dType::FlipAdst))
        }
    }
}

#[target_feature(enable = "sse4.1")]
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
            adst8_sonlandir_gecis2(iadst8_ham_x8(c), matches!(kind, Tx1dType::FlipAdst))
        }
    }
}

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
#[inline]
fn idct16_x4(c: [__m128i; 16]) -> [__m128i; 16] {
    let e = idct8_x4([c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]]);
    let (t8a, t15a) = mul2_x4(c[1], c[15], 401, -4076, 4076, 401);
    let (t9a, t14a) = mul2_x4(c[9], c[7], 3166, -2598, 2598, 3166);
    let (t10a, t13a) = mul2_x4(c[5], c[11], 1931, -3612, 3612, 1931);
    let (t11a, t12a) = mul2_x4(c[13], c[3], 3920, -1189, 1189, 3920);
    idct16_ortak_son_x4(e, [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a])
}

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
#[inline]
fn idct16_x8(c: [__m128i; 16]) -> [__m128i; 16] {
    let e = idct8_x8([c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]]);
    let (t8a, t15a) = mul2_x8(c[1], c[15], 401, -4076, 4076, 401);
    let (t9a, t14a) = mul2_x8(c[9], c[7], 3166, -2598, 2598, 3166);
    let (t10a, t13a) = mul2_x8(c[5], c[11], 1931, -3612, 3612, 1931);
    let (t11a, t12a) = mul2_x8(c[13], c[3], 3920, -1189, 1189, 3920);
    idct16_ortak_son_x8(e, [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a])
}

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
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

#[target_feature(enable = "sse4.1")]
#[inline]
fn adst16_sonlandir_gecis2(u: [__m128i; 16], flip: bool) -> [__m128i; 16] {
    let p = _mm_set1_epi16(2048);
    let n = _mm_set1_epi16(-2048);
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

#[target_feature(enable = "sse4.1")]
#[inline]
fn iidtx16_gecis2_x4(c: [__m128i; 16]) -> [__m128i; 16] {
    let k = _mm_set1_epi16(1697 * 16);
    let iki_bir = _mm_set1_epi32(cift_katsayi(2, 1));
    let sekiz = _mm_set1_epi32(8);
    let mut o = [_mm_setzero_si128(); 16];
    let mut i = 0;
    while i < 16 {
        let t = _mm_mulhrs_epi16(c[i], k);
        let lo = _mm_unpacklo_epi16(c[i], t);
        let s = _mm_srai_epi32::<4>(_mm_add_epi32(_mm_madd_epi16(lo, iki_bir), sekiz));
        o[i] = _mm_packs_epi32(s, s);
        i += 1;
    }
    o
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn iidtx16_gecis2_x8(c: [__m128i; 16]) -> [__m128i; 16] {
    let k = _mm_set1_epi16(1697 * 16);
    let iki_bir = _mm_set1_epi32(cift_katsayi(2, 1));
    let sekiz = _mm_set1_epi32(8);
    let mut o = [_mm_setzero_si128(); 16];
    let mut i = 0;
    while i < 16 {
        let t = _mm_mulhrs_epi16(c[i], k);
        let lo = _mm_unpacklo_epi16(c[i], t);
        let hi = _mm_unpackhi_epi16(c[i], t);
        let s_lo = _mm_srai_epi32::<4>(_mm_add_epi32(_mm_madd_epi16(lo, iki_bir), sekiz));
        let s_hi = _mm_srai_epi32::<4>(_mm_add_epi32(_mm_madd_epi16(hi, iki_bir), sekiz));
        o[i] = _mm_packs_epi32(s_lo, s_hi);
        i += 1;
    }
    o
}

#[target_feature(enable = "sse4.1")]
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
            adst16_sonlandir_gecis2(iadst16_ham_x4(c), matches!(kind, Tx1dType::FlipAdst))
        }
    }
}

#[target_feature(enable = "sse4.1")]
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
            adst16_sonlandir_gecis2(iadst16_ham_x8(c), matches!(kind, Tx1dType::FlipAdst))
        }
    }
}

// ---------------------------------------------------------------------------
// 2B blok gövdeleri (`w, h ≤ 16`, 10 bpc)
// ---------------------------------------------------------------------------

#[target_feature(enable = "sse4.1")]
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
    chunk_mut::<i32, 16>(coeff, 0).fill(0);
    let p1 = paketle_x4::<4, 0>(gecis1_4_i32(giris, yatay));
    let t = devrik_4x4(p1);
    let cikis = gecis2_4_x4(t, dikey);
    ekle_yaz_4x2(dst, stride, 0, cikis[0], cikis[1]);
    ekle_yaz_4x2(dst, stride, 2, cikis[2], cikis[3]);
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn blok_4x8<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<4, 8, 0, true>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let ust = [
        olcekle_181_i32(yukle_i32x4(coeff, 0)),
        olcekle_181_i32(yukle_i32x4(coeff, 8)),
        olcekle_181_i32(yukle_i32x4(coeff, 16)),
        olcekle_181_i32(yukle_i32x4(coeff, 24)),
    ];
    let alt = [
        olcekle_181_i32(yukle_i32x4(coeff, 4)),
        olcekle_181_i32(yukle_i32x4(coeff, 12)),
        olcekle_181_i32(yukle_i32x4(coeff, 20)),
        olcekle_181_i32(yukle_i32x4(coeff, 28)),
    ];
    chunk_mut::<i32, 32>(coeff, 0).fill(0);
    let p1 = paketle_x8::<4, 0>(gecis1_4_i32(ust, yatay), gecis1_4_i32(alt, yatay));
    let t = devrik_4x8(p1);
    let cikis = gecis2_8_x4(t, dikey);
    ekle_yaz_4x2(dst, stride, 0, cikis[0], cikis[1]);
    ekle_yaz_4x2(dst, stride, 2, cikis[2], cikis[3]);
    ekle_yaz_4x2(dst, stride, 4, cikis[4], cikis[5]);
    ekle_yaz_4x2(dst, stride, 6, cikis[6], cikis[7]);
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn blok_8x4<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<8, 4, 0, true>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let giris = [
        olcekle_181_i32(yukle_i32x4(coeff, 0)),
        olcekle_181_i32(yukle_i32x4(coeff, 4)),
        olcekle_181_i32(yukle_i32x4(coeff, 8)),
        olcekle_181_i32(yukle_i32x4(coeff, 12)),
        olcekle_181_i32(yukle_i32x4(coeff, 16)),
        olcekle_181_i32(yukle_i32x4(coeff, 20)),
        olcekle_181_i32(yukle_i32x4(coeff, 24)),
        olcekle_181_i32(yukle_i32x4(coeff, 28)),
    ];
    chunk_mut::<i32, 32>(coeff, 0).fill(0);
    let p1 = paketle_x4::<8, 0>(gecis1_8_i32(giris, yatay));
    let t = devrik_8x4(p1);
    let cikis = gecis2_4_x8(t, dikey);
    ekle_yaz_8x1(dst, 0, cikis[0]);
    ekle_yaz_8x1(dst, stride, cikis[1]);
    ekle_yaz_8x1(dst, stride * 2, cikis[2]);
    ekle_yaz_8x1(dst, stride * 3, cikis[3]);
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn blok_8x8<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<8, 8, 1, false>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let ust = [
        yukle_i32x4(coeff, 0),
        yukle_i32x4(coeff, 8),
        yukle_i32x4(coeff, 16),
        yukle_i32x4(coeff, 24),
        yukle_i32x4(coeff, 32),
        yukle_i32x4(coeff, 40),
        yukle_i32x4(coeff, 48),
        yukle_i32x4(coeff, 56),
    ];
    let alt = [
        yukle_i32x4(coeff, 4),
        yukle_i32x4(coeff, 12),
        yukle_i32x4(coeff, 20),
        yukle_i32x4(coeff, 28),
        yukle_i32x4(coeff, 36),
        yukle_i32x4(coeff, 44),
        yukle_i32x4(coeff, 52),
        yukle_i32x4(coeff, 60),
    ];
    chunk_mut::<i32, 64>(coeff, 0).fill(0);
    let p1 = paketle_x8::<8, 1>(gecis1_8_i32(ust, yatay), gecis1_8_i32(alt, yatay));
    let t = devrik_8x8(p1);
    let cikis = gecis2_8_x8(t, dikey);
    let mut y = 0;
    while y < 8 {
        ekle_yaz_8x1(dst, y * stride, cikis[y]);
        y += 1;
    }
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn blok_4x16<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<4, 16, 1, false>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let yarim = matches!(txtp, TxType::DctDct) && eob <= 28;
    let g0 = [
        yukle_i32x4(coeff, 0),
        yukle_i32x4(coeff, 16),
        yukle_i32x4(coeff, 32),
        yukle_i32x4(coeff, 48),
    ];
    let g1 = [
        yukle_i32x4(coeff, 4),
        yukle_i32x4(coeff, 20),
        yukle_i32x4(coeff, 36),
        yukle_i32x4(coeff, 52),
    ];
    let t_ust = devrik_4x8(paketle_x8::<4, 1>(
        gecis1_4_i32(g0, yatay),
        gecis1_4_i32(g1, yatay),
    ));
    let mut t = [_mm_setzero_si128(); 16];
    let mut i = 0;
    while i < 8 {
        t[i] = t_ust[i];
        i += 1;
    }
    if !yarim {
        let g2 = [
            yukle_i32x4(coeff, 8),
            yukle_i32x4(coeff, 24),
            yukle_i32x4(coeff, 40),
            yukle_i32x4(coeff, 56),
        ];
        let g3 = [
            yukle_i32x4(coeff, 12),
            yukle_i32x4(coeff, 28),
            yukle_i32x4(coeff, 44),
            yukle_i32x4(coeff, 60),
        ];
        let t_alt = devrik_4x8(paketle_x8::<4, 1>(
            gecis1_4_i32(g2, yatay),
            gecis1_4_i32(g3, yatay),
        ));
        let mut j = 0;
        while j < 8 {
            t[8 + j] = t_alt[j];
            j += 1;
        }
    }
    chunk_mut::<i32, 64>(coeff, 0).fill(0);
    let cikis = gecis2_16_x4(t, dikey, yarim);
    let mut y = 0;
    while y < 16 {
        ekle_yaz_4x2(dst, stride, y, cikis[y], cikis[y + 1]);
        y += 2;
    }
}

#[target_feature(enable = "sse4.1")]
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
    chunk_mut::<i32, 64>(coeff, 0).fill(0);
    let p1 = paketle_x4::<16, 1>(gecis1_16_i32(giris, yatay, yarim));
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

#[target_feature(enable = "sse4.1")]
#[inline]
fn blok_8x16<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<8, 16, 1, true>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let yarim = matches!(txtp, TxType::DctDct) && eob <= 42;
    let mut g0 = [_mm_setzero_si128(); 8];
    let mut g1 = [_mm_setzero_si128(); 8];
    let mut x = 0;
    while x < 8 {
        g0[x] = olcekle_181_i32(yukle_i32x4(coeff, x * 16));
        g1[x] = olcekle_181_i32(yukle_i32x4(coeff, x * 16 + 4));
        x += 1;
    }
    let t_ust = devrik_8x8(paketle_x8::<8, 1>(
        gecis1_8_i32(g0, yatay),
        gecis1_8_i32(g1, yatay),
    ));
    let mut t = [_mm_setzero_si128(); 16];
    let mut i = 0;
    while i < 8 {
        t[i] = t_ust[i];
        i += 1;
    }
    if !yarim {
        let mut g2 = [_mm_setzero_si128(); 8];
        let mut g3 = [_mm_setzero_si128(); 8];
        let mut x2 = 0;
        while x2 < 8 {
            g2[x2] = olcekle_181_i32(yukle_i32x4(coeff, x2 * 16 + 8));
            g3[x2] = olcekle_181_i32(yukle_i32x4(coeff, x2 * 16 + 12));
            x2 += 1;
        }
        let t_alt = devrik_8x8(paketle_x8::<8, 1>(
            gecis1_8_i32(g2, yatay),
            gecis1_8_i32(g3, yatay),
        ));
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
        ekle_yaz_8x1(dst, y * stride, cikis[y]);
        y += 1;
    }
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn blok_16x8<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<16, 8, 1, true>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let yarim = matches!(txtp, TxType::DctDct) && eob <= 42;
    let mut g0 = [_mm_setzero_si128(); 16];
    let mut g1 = [_mm_setzero_si128(); 16];
    let sinir = if yarim { 8 } else { 16 };
    let mut x = 0;
    while x < sinir {
        g0[x] = olcekle_181_i32(yukle_i32x4(coeff, x * 8));
        g1[x] = olcekle_181_i32(yukle_i32x4(coeff, x * 8 + 4));
        x += 1;
    }
    chunk_mut::<i32, 128>(coeff, 0).fill(0);
    let p1 = paketle_x8::<16, 1>(
        gecis1_16_i32(g0, yatay, yarim),
        gecis1_16_i32(g1, yatay, yarim),
    );
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

#[target_feature(enable = "sse4.1")]
#[inline]
fn blok_16x16<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<16, 16, 2, false>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let yarim = matches!(txtp, TxType::DctDct) && eob <= 35;
    let x_sinir = if yarim { 8 } else { 16 };

    let mut g0 = [_mm_setzero_si128(); 16];
    let mut g1 = [_mm_setzero_si128(); 16];
    let mut x = 0;
    while x < x_sinir {
        g0[x] = yukle_i32x4(coeff, x * 16);
        g1[x] = yukle_i32x4(coeff, x * 16 + 4);
        x += 1;
    }
    let p1_ust = paketle_x8::<16, 2>(
        gecis1_16_i32(g0, yatay, yarim),
        gecis1_16_i32(g1, yatay, yarim),
    );
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
        let mut g2 = [_mm_setzero_si128(); 16];
        let mut g3 = [_mm_setzero_si128(); 16];
        let mut x2 = 0;
        while x2 < 16 {
            g2[x2] = yukle_i32x4(coeff, x2 * 16 + 8);
            g3[x2] = yukle_i32x4(coeff, x2 * 16 + 12);
            x2 += 1;
        }
        let p1_alt = paketle_x8::<16, 2>(
            gecis1_16_i32(g2, yatay, false),
            gecis1_16_i32(g3, yatay, false),
        );
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

/// `(TX, TXTP)` çiftine özelleştirilmiş 10 bpc SSE4.1 ters dönüşüm çekirdek gövdesi.
#[target_feature(enable = "sse4.1")]
#[inline(never)]
fn itx10_sse4_cekirdek<const TX: u8, const TXTP: u8>(
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

/// `(TX, TXTP)` çiftinin sözleşme denetimli güvenli 10 bpc SSE4.1 yuva girişi.
fn itx10_sse4_giris<const TX: u8, const TXTP: u8>(
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
        || !itx16_sozlesmesi(dst.len(), dst_stride, coeff.len(), eob, tx, txtp)
    {
        match bit_depth {
            BitDepth::Eight => itx_skaler_devret::<8>(dst, dst_stride, coeff, eob, tx, txtp),
            BitDepth::Ten => itx_skaler_devret::<10>(dst, dst_stride, coeff, eob, tx, txtp),
            BitDepth::Twelve => itx_skaler_devret::<12>(dst, dst_stride, coeff, eob, tx, txtp),
        }
        return;
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Sse41Token` ile verilir (`itx10_sse4_fn`);
    // belirteç SSE4.1 + SSSE3 tespiti ve maske izninin kanıtıdır. (2) sınır: `itx16_sozlesmesi`
    // `dst_stride`'ın çift ve `≥ w·2` olduğunu, `dst`'nin `(h − 1)·dst_stride + w·2` baytı ve
    // `coeff`'un `1024` öğeyi kapsadığını garanti eder; bütün bellek erişimleri sabit boyutlu
    // `chunk`/`chunk_mut` dizileri üzerindendir. (3) hizalama/takma ad: `dst` ve `coeff` ayrık
    // `&mut` referanslardır, hizasız erişim kullanılır.
    unsafe {
        itx10_sse4_cekirdek::<TX, TXTP>(dst, dst_stride, coeff, eob);
    }
}

macro_rules! esle_16_tur {
    ($tx:ident, $txtp:expr) => {
        match $txtp {
            TxType::DctDct => {
                Some(itx10_sse4_giris::<{ TxSize::$tx as u8 }, { TxType::DctDct as u8 }>)
            }
            TxType::AdstDct => {
                Some(itx10_sse4_giris::<{ TxSize::$tx as u8 }, { TxType::AdstDct as u8 }>)
            }
            TxType::DctAdst => {
                Some(itx10_sse4_giris::<{ TxSize::$tx as u8 }, { TxType::DctAdst as u8 }>)
            }
            TxType::AdstAdst => {
                Some(itx10_sse4_giris::<{ TxSize::$tx as u8 }, { TxType::AdstAdst as u8 }>)
            }
            TxType::FlipadstDct => {
                Some(itx10_sse4_giris::<{ TxSize::$tx as u8 }, { TxType::FlipadstDct as u8 }>)
            }
            TxType::DctFlipadst => {
                Some(itx10_sse4_giris::<{ TxSize::$tx as u8 }, { TxType::DctFlipadst as u8 }>)
            }
            TxType::FlipadstFlipadst => {
                Some(itx10_sse4_giris::<{ TxSize::$tx as u8 }, { TxType::FlipadstFlipadst as u8 }>)
            }
            TxType::AdstFlipadst => {
                Some(itx10_sse4_giris::<{ TxSize::$tx as u8 }, { TxType::AdstFlipadst as u8 }>)
            }
            TxType::FlipadstAdst => {
                Some(itx10_sse4_giris::<{ TxSize::$tx as u8 }, { TxType::FlipadstAdst as u8 }>)
            }
            TxType::Idtx => Some(itx10_sse4_giris::<{ TxSize::$tx as u8 }, { TxType::Idtx as u8 }>),
            TxType::VDct => Some(itx10_sse4_giris::<{ TxSize::$tx as u8 }, { TxType::VDct as u8 }>),
            TxType::HDct => Some(itx10_sse4_giris::<{ TxSize::$tx as u8 }, { TxType::HDct as u8 }>),
            TxType::VAdst => {
                Some(itx10_sse4_giris::<{ TxSize::$tx as u8 }, { TxType::VAdst as u8 }>)
            }
            TxType::HAdst => {
                Some(itx10_sse4_giris::<{ TxSize::$tx as u8 }, { TxType::HAdst as u8 }>)
            }
            TxType::VFlipadst => {
                Some(itx10_sse4_giris::<{ TxSize::$tx as u8 }, { TxType::VFlipadst as u8 }>)
            }
            TxType::HFlipadst => {
                Some(itx10_sse4_giris::<{ TxSize::$tx as u8 }, { TxType::HFlipadst as u8 }>)
            }
            TxType::WhtWht => None,
        }
    };
}

/// `w, h ≤ 16` boyutlu 140 adet 10 bpc WHT-dışı SSE4.1 `itxfm_add[tx][txtp]` yuvası
/// (yalnız [`Sse41Token`] ile).
#[must_use]
pub const fn itx10_sse4_fn(_token: Sse41Token, tx: TxSize, txtp: TxType) -> Option<ItxFn> {
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
                Some(itx10_sse4_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::DctDct as u8 }>)
            }
            TxType::AdstDct => {
                Some(itx10_sse4_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::AdstDct as u8 }>)
            }
            TxType::DctAdst => {
                Some(itx10_sse4_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::DctAdst as u8 }>)
            }
            TxType::AdstAdst => {
                Some(itx10_sse4_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::AdstAdst as u8 }>)
            }
            TxType::FlipadstDct => {
                Some(itx10_sse4_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::FlipadstDct as u8 }>)
            }
            TxType::DctFlipadst => {
                Some(itx10_sse4_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::DctFlipadst as u8 }>)
            }
            TxType::FlipadstFlipadst => Some(
                itx10_sse4_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::FlipadstFlipadst as u8 }>,
            ),
            TxType::AdstFlipadst => {
                Some(itx10_sse4_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::AdstFlipadst as u8 }>)
            }
            TxType::FlipadstAdst => {
                Some(itx10_sse4_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::FlipadstAdst as u8 }>)
            }
            TxType::Idtx => {
                Some(itx10_sse4_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::Idtx as u8 }>)
            }
            TxType::VDct => {
                Some(itx10_sse4_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::VDct as u8 }>)
            }
            TxType::HDct => {
                Some(itx10_sse4_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::HDct as u8 }>)
            }
            _ => None,
        },
        _ => None,
    }
}
