//! 10 bpc WHT-dışı küçük blok (`8 ≤ w, h ≤ 16`, 60 kayıt) x86_64 AVX512ICL ters dönüşüm çekirdekleri — P44 Adım 2.
//!
//! Bu modül dav1d'nin `src/x86/itx.h:289-295` ve `src/x86/itx16_avx512.asm` dosyalarındaki
//! `8 ≤ w, h ≤ 16` boyutlu **60** adet **yalnız 10 bpc** (`BitDepth::Ten`) AVX512ICL `itxfm_add`
//! kaydını **saf kararlı Rust intrinsics** ile gerçekler; dış `.asm`/`.S`, `asm!`, `global_asm!`
//! ve C bağımlılığı yoktur (CEK-02). `w == 4 || h == 4` boyutları (`Tx4x4`, `Rtx4x8`, `Rtx8x4`,
//! `Rtx4x16`, `Rtx16x4`) dav1d paritesiyle AVX2'de (`itx16_10`) kalır; `12 bpc` bu modülde asla
//! SIMD çalıştırmaz:
//!
//! - `TX_8X8`, `RTX_8X16`, `RTX_16X8`: 16 WHT-dışı `TxType` türünün tamamı (`3 × 16 = 48` kayıt),
//! - `TX_16X16`: desteklenen 12 `TxType` (`DCT_DCT`, `ADST_DCT`, `DCT_ADST`, `ADST_ADST`,
//!   `FLIPADST_DCT`, `DCT_FLIPADST`, `FLIPADST_FLIPADST`, `ADST_FLIPADST`, `FLIPADST_ADST`,
//!   `IDTX`, `V_DCT`, `H_DCT` = `12` kayıt).
//!
//! ## Sözleşme, P34 izolasyonu, Zen 4 (`0` gather/scatter) ve VNNI + ZMM aritmetiği
//!
//! - **P34 modül izolasyonu:** Bütün SIMD işlevleri
//!   `#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]` taşır ve
//!   başka `dsp/x86/*` çekirdek modülleriyle `#[target_feature]` yardımcısı paylaşmaz.
//! - **Zen 4 kuralı (`0` gather/scatter):** `vpgather*` / `vpscatter*` yönergesi kesinlikle
//!   üretilmez (`0` adet); tüm hedef satır erişimleri güvenli dilim indekslemesiyle
//!   (`chunk` / `chunk_mut`) yapılır.
//! - **Pass 1 (`i32`, ZMM / YMM):** `8×16` ve `16×16` bloklarında `16` satır tek seferde
//!   `16 × i32` `__m512i` (`_mm512_mullo_epi32`, `_mm512_max_epi32`, `_mm512_min_epi32`)
//!   şeritlerinde `[-131072, +131071]` (`clip_18b`) ile işlenir ve `_mm512_packs_epi32` /
//!   `_mm256_packs_epi32` ile `[-32768, +32767]` (`col_clip`) `i16` düzenine daraltılır.
//! - **Pass 2 (`i16`, AVX-512 VNNI):** Kelebek çarpımlarında ve `iadst4_gecis2`, `iidtx4_gecis2`,
//!   `iidtx16_gecis2` dönüşümlerinde `_mm_dpwssd_epi32` / `_mm256_dpwssd_epi32` (`vpdpwssd`)
//!   kullanılır; çıktılar `[0, 1023]` aralığına kırpılarak `u16` hedef düzleme eklenir.
//! - **Tek `unsafe` bloğu:** Modülde yalnızca [`itx10_avx512_giris`] içinde 1 adet `unsafe`
//!   bloğu bulunur.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm_adds_epi16, _mm_dpwssd_epi32, _mm_max_epi16, _mm_min_epi16,
    _mm_mulhrs_epi16, _mm_packs_epi32, _mm_set1_epi16, _mm_set1_epi32, _mm_setzero_si128,
    _mm_srai_epi32, _mm_subs_epi16, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64,
    _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm256_add_epi32,
    _mm256_adds_epi16, _mm256_castsi256_si128, _mm256_dpwssd_epi32, _mm256_extracti128_si256,
    _mm256_max_epi16, _mm256_max_epi32, _mm256_min_epi16, _mm256_min_epi32, _mm256_mulhrs_epi16,
    _mm256_mullo_epi32, _mm256_packs_epi32, _mm256_permute2x128_si256, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_setr_epi32, _mm256_setzero_si256, _mm256_srai_epi32,
    _mm256_sub_epi32, _mm256_subs_epi16, _mm256_unpackhi_epi16, _mm256_unpackhi_epi32,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi16, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64,
    _mm512_add_epi32, _mm512_castsi512_si256, _mm512_extracti64x4_epi64, _mm512_max_epi32,
    _mm512_min_epi32, _mm512_mullo_epi32, _mm512_packs_epi32, _mm512_set1_epi32, _mm512_setr_epi32,
    _mm512_setzero_si512, _mm512_shuffle_i32x4, _mm512_srai_epi32, _mm512_sub_epi32,
    _mm512_unpackhi_epi16, _mm512_unpackhi_epi32, _mm512_unpackhi_epi64, _mm512_unpacklo_epi16,
    _mm512_unpacklo_epi32, _mm512_unpacklo_epi64,
};

use super::mem::{chunk, chunk_mut, load_u8x16, load_u8x32, store_i32x8, store_u8x16, store_u8x32};
use crate::cpu::Avx512IclToken;
use crate::dsp::itx::{ItxFn, TxSize, TxType, is_supported, itx_skaler_devret, itx16_sozlesmesi};
use crate::dsp::scalar::itx_1d::Tx1dType;
use crate::frame::BitDepth;

/// İki `i16` çarpanını `_mm*_dpwssd_epi32` için `i32` şeritlerine paketler (`[c0, c1, ...]`).
#[inline(always)]
const fn cift_katsayi(c0: i16, c1: i16) -> i32 {
    (c0 as u16 as i32) | ((c1 as u16 as i32) << 16)
}

/// `coeff[..N]` bölgesini (`N` 8'in katı) 256-bit sıfır saklamalarıyla temizler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
// Pass 1 (`i32` 8 şerit `__m256i` ve 16 şerit `__m512i`) temel yardımcıları
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_i32x8(coeff: &[i32], off: usize) -> __m256i {
    let c = chunk::<i32, 8>(coeff, off);
    _mm256_setr_epi32(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7])
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_i32x16(coeff: &[i32], off: usize) -> __m512i {
    let c = chunk::<i32, 16>(coeff, off);
    _mm512_setr_epi32(
        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12], c[13],
        c[14], c[15],
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn kirp_18b_x8(v: __m256i) -> __m256i {
    _mm256_min_epi32(
        _mm256_max_epi32(v, _mm256_set1_epi32(-131_072)),
        _mm256_set1_epi32(131_071),
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn kirp_18b_z16(v: __m512i) -> __m512i {
    _mm512_min_epi32(
        _mm512_max_epi32(v, _mm512_set1_epi32(-131_072)),
        _mm512_set1_epi32(131_071),
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn olcekle_181_i32_x8(v: __m256i) -> __m256i {
    _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_mullo_epi32(v, _mm256_set1_epi32(181)),
        _mm256_set1_epi32(128),
    ))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn olcekle_181_i32_z16(v: __m512i) -> __m512i {
    _mm512_srai_epi32::<8>(_mm512_add_epi32(
        _mm512_mullo_epi32(v, _mm512_set1_epi32(181)),
        _mm512_set1_epi32(128),
    ))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn mul1_i32_x8(a: __m256i, k: i32) -> __m256i {
    _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_mullo_epi32(a, _mm256_set1_epi32(k)),
        _mm256_set1_epi32(2048),
    ))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn mul1_i32_z16(a: __m512i, k: i32) -> __m512i {
    _mm512_srai_epi32::<12>(_mm512_add_epi32(
        _mm512_mullo_epi32(a, _mm512_set1_epi32(k)),
        _mm512_set1_epi32(2048),
    ))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn mul2_i32_z16(a: __m512i, b: __m512i, c0: i32, c1: i32, d0: i32, d1: i32) -> (__m512i, __m512i) {
    let rnd = _mm512_set1_epi32(2048);
    let u = _mm512_srai_epi32::<12>(_mm512_add_epi32(
        _mm512_add_epi32(
            _mm512_mullo_epi32(a, _mm512_set1_epi32(c0)),
            _mm512_mullo_epi32(b, _mm512_set1_epi32(c1)),
        ),
        rnd,
    ));
    let v = _mm512_srai_epi32::<12>(_mm512_add_epi32(
        _mm512_add_epi32(
            _mm512_mullo_epi32(a, _mm512_set1_epi32(d0)),
            _mm512_mullo_epi32(b, _mm512_set1_epi32(d1)),
        ),
        rnd,
    ));
    (u, v)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn kaydir_i32_x8<const SHIFT: u32>(v: __m256i) -> __m256i {
    match SHIFT {
        0 => v,
        1 => _mm256_srai_epi32::<1>(_mm256_add_epi32(v, _mm256_set1_epi32(1))),
        2 => _mm256_srai_epi32::<2>(_mm256_add_epi32(v, _mm256_set1_epi32(2))),
        _ => unreachable!(),
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn kaydir_i32_z16<const SHIFT: u32>(v: __m512i) -> __m512i {
    match SHIFT {
        0 => v,
        1 => _mm512_srai_epi32::<1>(_mm512_add_epi32(v, _mm512_set1_epi32(1))),
        2 => _mm512_srai_epi32::<2>(_mm512_add_epi32(v, _mm512_set1_epi32(2))),
        _ => unreachable!(),
    }
}

/// `8 × i32` (`__m256i`) sütun dizisini kaydırıp `8 × i16` (`__m128i`) sütun dizisine paketler (`col_clip`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
// Pass 2 (`i16` 8 ve 16 şerit, AVX-512 VNNI `vpdpwssd`) temel yardımcıları
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn mul2_x8(a: __m128i, b: __m128i, c0: i16, c1: i16, d0: i16, d1: i16) -> (__m128i, __m128i) {
    let lo = _mm_unpacklo_epi16(a, b);
    let hi = _mm_unpackhi_epi16(a, b);
    let rnd = _mm_set1_epi32(2048);
    let kc = _mm_set1_epi32(cift_katsayi(c0, c1));
    let kd = _mm_set1_epi32(cift_katsayi(d0, d1));
    let u_lo = _mm_srai_epi32::<12>(_mm_dpwssd_epi32(rnd, lo, kc));
    let u_hi = _mm_srai_epi32::<12>(_mm_dpwssd_epi32(rnd, hi, kc));
    let v_lo = _mm_srai_epi32::<12>(_mm_dpwssd_epi32(rnd, lo, kd));
    let v_hi = _mm_srai_epi32::<12>(_mm_dpwssd_epi32(rnd, hi, kd));
    (_mm_packs_epi32(u_lo, u_hi), _mm_packs_epi32(v_lo, v_hi))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn mul2_x16(a: __m256i, b: __m256i, c0: i16, c1: i16, d0: i16, d1: i16) -> (__m256i, __m256i) {
    let lo = _mm256_unpacklo_epi16(a, b);
    let hi = _mm256_unpackhi_epi16(a, b);
    let rnd = _mm256_set1_epi32(2048);
    let kc = _mm256_set1_epi32(cift_katsayi(c0, c1));
    let kd = _mm256_set1_epi32(cift_katsayi(d0, d1));
    let u_lo = _mm256_srai_epi32::<12>(_mm256_dpwssd_epi32(rnd, lo, kc));
    let u_hi = _mm256_srai_epi32::<12>(_mm256_dpwssd_epi32(rnd, hi, kc));
    let v_lo = _mm256_srai_epi32::<12>(_mm256_dpwssd_epi32(rnd, lo, kd));
    let v_hi = _mm256_srai_epi32::<12>(_mm256_dpwssd_epi32(rnd, hi, kd));
    (
        _mm256_packs_epi32(u_lo, u_hi),
        _mm256_packs_epi32(v_lo, v_hi),
    )
}

// ---------------------------------------------------------------------------
// Hedef `u16` düzlem piksel ekleme + `[0, 1023]` doyurmalı yazma yardımcıları
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_8x1(dst: &mut [u8], off: usize, r: __m128i) {
    let d = load_u8x16(chunk::<u8, 16>(dst, off));
    let s = _mm_min_epi16(
        _mm_max_epi16(_mm_adds_epi16(d, r), _mm_setzero_si128()),
        _mm_set1_epi16(1023),
    );
    store_u8x16(chunk_mut(dst, off), s);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

/// Dört adet `8 × 8` `i16` bloğunu `__m512i` 128-bit şerit-içi (in-lane) aynı anda devrikler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn devrik_8x8_z32(r: &mut [__m512i; 8]) {
    let a0 = _mm512_unpacklo_epi16(r[0], r[1]);
    let a1 = _mm512_unpackhi_epi16(r[0], r[1]);
    let a2 = _mm512_unpacklo_epi16(r[2], r[3]);
    let a3 = _mm512_unpackhi_epi16(r[2], r[3]);
    let a4 = _mm512_unpacklo_epi16(r[4], r[5]);
    let a5 = _mm512_unpackhi_epi16(r[4], r[5]);
    let a6 = _mm512_unpacklo_epi16(r[6], r[7]);
    let a7 = _mm512_unpackhi_epi16(r[6], r[7]);

    let b0 = _mm512_unpacklo_epi32(a0, a2);
    let b1 = _mm512_unpackhi_epi32(a0, a2);
    let b2 = _mm512_unpacklo_epi32(a1, a3);
    let b3 = _mm512_unpackhi_epi32(a1, a3);
    let b4 = _mm512_unpacklo_epi32(a4, a6);
    let b5 = _mm512_unpackhi_epi32(a4, a6);
    let b6 = _mm512_unpacklo_epi32(a5, a7);
    let b7 = _mm512_unpackhi_epi32(a5, a7);

    r[0] = _mm512_unpacklo_epi64(b0, b4);
    r[1] = _mm512_unpackhi_epi64(b0, b4);
    r[2] = _mm512_unpacklo_epi64(b1, b5);
    r[3] = _mm512_unpackhi_epi64(b1, b5);
    r[4] = _mm512_unpacklo_epi64(b2, b6);
    r[5] = _mm512_unpackhi_epi64(b2, b6);
    r[6] = _mm512_unpacklo_epi64(b3, b7);
    r[7] = _mm512_unpackhi_epi64(b3, b7);
}

/// `8×16` bloğunun Pass 1 512-bit (`[__m512i; 8]`, `16 × i32`) çıktısını kaydırıp `i16`'ya
/// paketler ve üst/alt `8×8` bloklarını tek `devrik_8x8_x16` çağrısıyla `[__m128i; 16]`
/// satırlarına devrikler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn paketle_devrik_8x16_z16(c: [__m512i; 8]) -> [__m128i; 16] {
    macro_rules! paket_sutun {
        ($v:expr) => {{
            let s = kaydir_i32_z16::<1>($v);
            let s_ust = _mm512_castsi512_si256(s);
            let s_alt = _mm512_extracti64x4_epi64::<1>(s);
            let lo = _mm256_permute2x128_si256::<0x20>(s_ust, s_alt);
            let hi = _mm256_permute2x128_si256::<0x31>(s_ust, s_alt);
            _mm256_packs_epi32(lo, hi)
        }};
    }
    let mut p = [
        paket_sutun!(c[0]),
        paket_sutun!(c[1]),
        paket_sutun!(c[2]),
        paket_sutun!(c[3]),
        paket_sutun!(c[4]),
        paket_sutun!(c[5]),
        paket_sutun!(c[6]),
        paket_sutun!(c[7]),
    ];
    devrik_8x8_x16(&mut p);
    [
        _mm256_castsi256_si128(p[0]),
        _mm256_castsi256_si128(p[1]),
        _mm256_castsi256_si128(p[2]),
        _mm256_castsi256_si128(p[3]),
        _mm256_castsi256_si128(p[4]),
        _mm256_castsi256_si128(p[5]),
        _mm256_castsi256_si128(p[6]),
        _mm256_castsi256_si128(p[7]),
        _mm256_extracti128_si256::<1>(p[0]),
        _mm256_extracti128_si256::<1>(p[1]),
        _mm256_extracti128_si256::<1>(p[2]),
        _mm256_extracti128_si256::<1>(p[3]),
        _mm256_extracti128_si256::<1>(p[4]),
        _mm256_extracti128_si256::<1>(p[5]),
        _mm256_extracti128_si256::<1>(p[6]),
        _mm256_extracti128_si256::<1>(p[7]),
    ]
}

// ---------------------------------------------------------------------------
// Pass 1: Yatay 1B dönüşümler (`i32` 8 şerit `__m256i` ve 16 şerit `__m512i`)
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
        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

            if flip {
                [o7, o6, o5, o4, o3, o2, o1, o0]
            } else {
                [o0, o1, o2, o3, o4, o5, o6, o7]
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
        #[inline]
        fn $gecis1_8(c: [$V; 8], kind: Tx1dType) -> [$V; 8] {
            match kind {
                Tx1dType::Identity => [
                    $add(c[0], c[0]),
                    $add(c[1], c[1]),
                    $add(c[2], c[2]),
                    $add(c[3], c[3]),
                    $add(c[4], c[4]),
                    $add(c[5], c[5]),
                    $add(c[6], c[6]),
                    $add(c[7], c[7]),
                ],
                Tx1dType::Dct => $idct8(c),
                Tx1dType::Adst | Tx1dType::FlipAdst => {
                    $iadst8(c, matches!(kind, Tx1dType::FlipAdst))
                }
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
        #[inline]
        fn $idct16(c: &mut [$V; 16]) {
            let e = $idct8([c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]]);
            let (t8a, t15a) = $mul2(c[1], c[15], 401, -4076, 4076, 401);
            let (t9a, t14a) = $mul2(c[9], c[7], 3166, -2598, 2598, 3166);
            let (t10a, t13a) = $mul2(c[5], c[11], 1931, -3612, 3612, 1931);
            let (t11a, t12a) = $mul2(c[13], c[3], 3920, -1189, 1189, 3920);
            $idct16_ortak_son(c, e, [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a]);
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

            *c = if flip {
                [
                    o15, o14, o13, o12, o11, o10, o9, o8, o7, o6, o5, o4, o3, o2, o1, o0,
                ]
            } else {
                [
                    o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15,
                ]
            };
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
        #[inline]
        fn $gecis1_16(c: &mut [$V; 16], kind: Tx1dType, yarim_sifir: bool) {
            match kind {
                Tx1dType::Identity => {
                    let k = $set1(5793);
                    let rnd = $set1(1024);
                    *c = [
                        $srai::<11>($add($mullo(c[0], k), rnd)),
                        $srai::<11>($add($mullo(c[1], k), rnd)),
                        $srai::<11>($add($mullo(c[2], k), rnd)),
                        $srai::<11>($add($mullo(c[3], k), rnd)),
                        $srai::<11>($add($mullo(c[4], k), rnd)),
                        $srai::<11>($add($mullo(c[5], k), rnd)),
                        $srai::<11>($add($mullo(c[6], k), rnd)),
                        $srai::<11>($add($mullo(c[7], k), rnd)),
                        $srai::<11>($add($mullo(c[8], k), rnd)),
                        $srai::<11>($add($mullo(c[9], k), rnd)),
                        $srai::<11>($add($mullo(c[10], k), rnd)),
                        $srai::<11>($add($mullo(c[11], k), rnd)),
                        $srai::<11>($add($mullo(c[12], k), rnd)),
                        $srai::<11>($add($mullo(c[13], k), rnd)),
                        $srai::<11>($add($mullo(c[14], k), rnd)),
                        $srai::<11>($add($mullo(c[15], k), rnd)),
                    ];
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

impl_gecis1_i32!(
    __m512i,
    _mm512_setzero_si512,
    _mm512_set1_epi32,
    _mm512_add_epi32,
    _mm512_sub_epi32,
    _mm512_mullo_epi32,
    _mm512_srai_epi32,
    kirp_18b_z16,
    olcekle_181_i32_z16,
    mul1_i32_z16,
    mul2_i32_z16,
    idct4_i32_z16,
    idct4_yarim_i32_z16,
    idct8_i32_z16,
    idct8_yarim_i32_z16,
    iadst8_i32_z16,
    gecis1_8_z16,
    idct16_ortak_son_i32_z16,
    idct16_i32_z16,
    idct16_yarim_i32_z16,
    iadst16_i32_z16,
    gecis1_16_z16
);

// ---------------------------------------------------------------------------
// Pass 2: Dikey 1B dönüşümler (`i16` 8 ve 16 şerit, AVX-512 VNNI `vpdpwssd`)
// ---------------------------------------------------------------------------

macro_rules! impl_gecis2_tam_serit {
    (
        $V:ty,
        $set1_16:ident,
        $set1_32:ident,
        $mulhrs16:ident,
        $unpacklo16:ident,
        $unpackhi16:ident,
        $dpwssd:ident,
        $srai32:ident,
        $packs32:ident,
        $iadst4_g2:ident,
        $iidtx4_g2:ident,
        $iidtx16_g2:ident
    ) => {
        #[allow(dead_code)]
        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
                let y_lo = $srai32::<16>($dpwssd($dpwssd(rnd, p02_lo, v02), p13_lo, v13));
                let y_hi = $srai32::<16>($dpwssd($dpwssd(rnd, p02_hi, v02), p13_hi, v13));
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

        #[allow(dead_code)]
        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
        #[inline]
        fn $iidtx4_g2(c: [$V; 4]) -> [$V; 4] {
            let k = $set1_16(1697 * 8);
            let bir = $set1_32(cift_katsayi(1, 1));
            let sekiz = $set1_32(8);
            macro_rules! id4 {
                ($x:expr) => {{
                    let t = $mulhrs16($x, k);
                    let lo = $unpacklo16($x, t);
                    let hi = $unpackhi16($x, t);
                    let s_lo = $srai32::<4>($dpwssd(sekiz, lo, bir));
                    let s_hi = $srai32::<4>($dpwssd(sekiz, hi, bir));
                    $packs32(s_lo, s_hi)
                }};
            }
            [id4!(c[0]), id4!(c[1]), id4!(c[2]), id4!(c[3])]
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
        #[inline]
        fn $iidtx16_g2(c: &mut [$V; 16]) {
            let k = $set1_16(1697 * 16);
            let iki_bir = $set1_32(cift_katsayi(2, 1));
            let sekiz = $set1_32(8);
            macro_rules! id16_bir {
                ($x:expr) => {{
                    let v = $x;
                    let t = $mulhrs16(v, k);
                    let lo = $unpacklo16(v, t);
                    let hi = $unpackhi16(v, t);
                    let s_lo = $srai32::<4>($dpwssd(sekiz, lo, iki_bir));
                    let s_hi = $srai32::<4>($dpwssd(sekiz, hi, iki_bir));
                    $packs32(s_lo, s_hi)
                }};
            }
            *c = [
                id16_bir!(c[0]),
                id16_bir!(c[1]),
                id16_bir!(c[2]),
                id16_bir!(c[3]),
                id16_bir!(c[4]),
                id16_bir!(c[5]),
                id16_bir!(c[6]),
                id16_bir!(c[7]),
                id16_bir!(c[8]),
                id16_bir!(c[9]),
                id16_bir!(c[10]),
                id16_bir!(c[11]),
                id16_bir!(c[12]),
                id16_bir!(c[13]),
                id16_bir!(c[14]),
                id16_bir!(c[15]),
            ];
        }
    };
}

impl_gecis2_tam_serit!(
    __m128i,
    _mm_set1_epi16,
    _mm_set1_epi32,
    _mm_mulhrs_epi16,
    _mm_unpacklo_epi16,
    _mm_unpackhi_epi16,
    _mm_dpwssd_epi32,
    _mm_srai_epi32,
    _mm_packs_epi32,
    iadst4_gecis2_x8,
    iidtx4_gecis2_x8,
    iidtx16_gecis2_x8
);

impl_gecis2_tam_serit!(
    __m256i,
    _mm256_set1_epi16,
    _mm256_set1_epi32,
    _mm256_mulhrs_epi16,
    _mm256_unpacklo_epi16,
    _mm256_unpackhi_epi16,
    _mm256_dpwssd_epi32,
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
        $iidtx16_g2:ident,
        $idct4:ident,
        $idct4_yarim:ident,
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
        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
        #[inline]
        fn $adst8_son(u: [$V; 8], flip: bool) -> [$V; 8] {
            let p = $set1_16(2048);
            let n = $set1_16(-2048);
            let o0 = $mulhrs16(u[0], p);
            let o1 = $mulhrs16(u[1], n);
            let o2 = $mulhrs16(u[2], p);
            let o3 = $mulhrs16(u[3], n);
            let o4 = $mulhrs16(u[4], p);
            let o5 = $mulhrs16(u[5], n);
            let o6 = $mulhrs16(u[6], p);
            let o7 = $mulhrs16(u[7], n);
            if flip {
                [o7, o6, o5, o4, o3, o2, o1, o0]
            } else {
                [o0, o1, o2, o3, o4, o5, o6, o7]
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
        #[inline]
        fn $gecis2_8(c: [$V; 8], kind: Tx1dType) -> [$V; 8] {
            match kind {
                Tx1dType::Identity => {
                    let m = $set1_16(4096);
                    [
                        $mulhrs16(c[0], m),
                        $mulhrs16(c[1], m),
                        $mulhrs16(c[2], m),
                        $mulhrs16(c[3], m),
                        $mulhrs16(c[4], m),
                        $mulhrs16(c[5], m),
                        $mulhrs16(c[6], m),
                        $mulhrs16(c[7], m),
                    ]
                }
                Tx1dType::Dct => {
                    let o = $idct8(c);
                    let m = $set1_16(2048);
                    [
                        $mulhrs16(o[0], m),
                        $mulhrs16(o[1], m),
                        $mulhrs16(o[2], m),
                        $mulhrs16(o[3], m),
                        $mulhrs16(o[4], m),
                        $mulhrs16(o[5], m),
                        $mulhrs16(o[6], m),
                        $mulhrs16(o[7], m),
                    ]
                }
                Tx1dType::Adst | Tx1dType::FlipAdst => {
                    $adst8_son($iadst8_ham(c), matches!(kind, Tx1dType::FlipAdst))
                }
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
        #[inline]
        fn $idct16(c: &mut [$V; 16]) {
            let e = $idct8([c[0], c[2], c[4], c[6], c[8], c[10], c[12], c[14]]);
            let (t8a, t15a) = $mul2(c[1], c[15], 401, -4076, 4076, 401);
            let (t9a, t14a) = $mul2(c[9], c[7], 3166, -2598, 2598, 3166);
            let (t10a, t13a) = $mul2(c[5], c[11], 1931, -3612, 3612, 1931);
            let (t11a, t12a) = $mul2(c[13], c[3], 3920, -1189, 1189, 3920);
            $idct16_ortak_son(c, e, [t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a]);
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
        #[inline]
        fn $adst16_son(c: &mut [$V; 16], flip: bool) {
            let p = $set1_16(2048);
            let n = $set1_16(-2048);
            let o0 = $mulhrs16(c[0], p);
            let o1 = $mulhrs16(c[1], n);
            let o2 = $mulhrs16(c[2], p);
            let o3 = $mulhrs16(c[3], n);
            let o4 = $mulhrs16(c[4], p);
            let o5 = $mulhrs16(c[5], n);
            let o6 = $mulhrs16(c[6], p);
            let o7 = $mulhrs16(c[7], n);
            let o8 = $mulhrs16(c[8], p);
            let o9 = $mulhrs16(c[9], n);
            let o10 = $mulhrs16(c[10], p);
            let o11 = $mulhrs16(c[11], n);
            let o12 = $mulhrs16(c[12], p);
            let o13 = $mulhrs16(c[13], n);
            let o14 = $mulhrs16(c[14], p);
            let o15 = $mulhrs16(c[15], n);
            *c = if flip {
                [
                    o15, o14, o13, o12, o11, o10, o9, o8, o7, o6, o5, o4, o3, o2, o1, o0,
                ]
            } else {
                [
                    o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15,
                ]
            };
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
        #[inline(never)]
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
                    *c = [
                        $mulhrs16(c[0], m),
                        $mulhrs16(c[1], m),
                        $mulhrs16(c[2], m),
                        $mulhrs16(c[3], m),
                        $mulhrs16(c[4], m),
                        $mulhrs16(c[5], m),
                        $mulhrs16(c[6], m),
                        $mulhrs16(c[7], m),
                        $mulhrs16(c[8], m),
                        $mulhrs16(c[9], m),
                        $mulhrs16(c[10], m),
                        $mulhrs16(c[11], m),
                        $mulhrs16(c[12], m),
                        $mulhrs16(c[13], m),
                        $mulhrs16(c[14], m),
                        $mulhrs16(c[15], m),
                    ];
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
    mul2_x8,
    iidtx16_gecis2_x8,
    idct4_x8,
    idct4_yarim_x8,
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
    iidtx16_gecis2_x16,
    idct4_x16,
    idct4_yarim_x16,
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
// 2B blok gövdeleri (`8 ≤ w, h ≤ 16`, 10 bpc AVX512ICL)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn blok_8x16<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<8, 16, 1, true>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
    let yarim = matches!(txtp, TxType::DctDct) && eob <= 42;
    let mut t = if yarim {
        let g_ust = [
            olcekle_181_i32_x8(yukle_i32x8(coeff, 0)),
            olcekle_181_i32_x8(yukle_i32x8(coeff, 16)),
            olcekle_181_i32_x8(yukle_i32x8(coeff, 32)),
            olcekle_181_i32_x8(yukle_i32x8(coeff, 48)),
            olcekle_181_i32_x8(yukle_i32x8(coeff, 64)),
            olcekle_181_i32_x8(yukle_i32x8(coeff, 80)),
            olcekle_181_i32_x8(yukle_i32x8(coeff, 96)),
            olcekle_181_i32_x8(yukle_i32x8(coeff, 112)),
        ];
        let t_ust = devrik_8x8(paketle_y8_to_x8::<8, 1>(gecis1_8_x8(g_ust, yatay)));
        let z = _mm_setzero_si128();
        [
            t_ust[0], t_ust[1], t_ust[2], t_ust[3], t_ust[4], t_ust[5], t_ust[6], t_ust[7], z, z,
            z, z, z, z, z, z,
        ]
    } else {
        let g = [
            olcekle_181_i32_z16(yukle_i32x16(coeff, 0)),
            olcekle_181_i32_z16(yukle_i32x16(coeff, 16)),
            olcekle_181_i32_z16(yukle_i32x16(coeff, 32)),
            olcekle_181_i32_z16(yukle_i32x16(coeff, 48)),
            olcekle_181_i32_z16(yukle_i32x16(coeff, 64)),
            olcekle_181_i32_z16(yukle_i32x16(coeff, 80)),
            olcekle_181_i32_z16(yukle_i32x16(coeff, 96)),
            olcekle_181_i32_z16(yukle_i32x16(coeff, 112)),
        ];
        paketle_devrik_8x16_z16(gecis1_8_z16(g, yatay))
    };
    sifirla_katsayi::<128>(coeff);
    gecis2_16_x8(&mut t, dikey, yarim);
    let mut y = 0;
    while y < 16 {
        ekle_yaz_8x1(dst, y * stride, t[y]);
        y += 1;
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn blok_16x8<const TXTP: u8>(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    if matches!(txtp, TxType::DctDct) && eob == 0 {
        dc_yalniz::<16, 8, 1, true>(dst, stride, coeff);
        return;
    }
    let (yatay, dikey) = txtp.tx1d();
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn gecis1_16x16_yarim_x8(coeff: &[i32], yatay: Tx1dType, out: &mut [__m256i; 8]) {
    let z = _mm256_setzero_si256();
    let mut g = [
        yukle_i32x8(coeff, 0),
        yukle_i32x8(coeff, 16),
        yukle_i32x8(coeff, 32),
        yukle_i32x8(coeff, 48),
        yukle_i32x8(coeff, 64),
        yukle_i32x8(coeff, 80),
        yukle_i32x8(coeff, 96),
        yukle_i32x8(coeff, 112),
        z,
        z,
        z,
        z,
        z,
        z,
        z,
        z,
    ];
    gecis1_16_x8(&mut g, yatay, true);
    *out = paketle_2sutun_x8::<2>(&g);
    devrik_8x8_x16(out);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn gecis1_16x16_tam_z16(coeff: &[i32], yatay: Tx1dType, sutun: &mut [__m256i; 16]) {
    let mut g = [
        yukle_i32x16(coeff, 0),
        yukle_i32x16(coeff, 16),
        yukle_i32x16(coeff, 32),
        yukle_i32x16(coeff, 48),
        yukle_i32x16(coeff, 64),
        yukle_i32x16(coeff, 80),
        yukle_i32x16(coeff, 96),
        yukle_i32x16(coeff, 112),
        yukle_i32x16(coeff, 128),
        yukle_i32x16(coeff, 144),
        yukle_i32x16(coeff, 160),
        yukle_i32x16(coeff, 176),
        yukle_i32x16(coeff, 192),
        yukle_i32x16(coeff, 208),
        yukle_i32x16(coeff, 224),
        yukle_i32x16(coeff, 240),
    ];
    gecis1_16_z16(&mut g, yatay, false);

    macro_rules! paket_cift {
        ($k:literal, $k8:literal) => {{
            let s0 = kaydir_i32_z16::<2>(g[$k]);
            let s1 = kaydir_i32_z16::<2>(g[$k8]);
            let lo = _mm512_shuffle_i32x4::<0x88>(s0, s1);
            let hi = _mm512_shuffle_i32x4::<0xDD>(s0, s1);
            _mm512_packs_epi32(lo, hi)
        }};
    }
    let mut pk = [
        paket_cift!(0, 8),
        paket_cift!(1, 9),
        paket_cift!(2, 10),
        paket_cift!(3, 11),
        paket_cift!(4, 12),
        paket_cift!(5, 13),
        paket_cift!(6, 14),
        paket_cift!(7, 15),
    ];
    devrik_8x8_z32(&mut pk);

    macro_rules! cikar_satir {
        ($($r:literal),*) => {
            $({
                let v = pk[$r];
                sutun[$r] = _mm512_castsi512_si256(_mm512_shuffle_i32x4::<0x88>(v, v));
                sutun[$r + 8] = _mm512_castsi512_si256(_mm512_shuffle_i32x4::<0xDD>(v, v));
            })*
        };
    }
    cikar_satir!(0, 1, 2, 3, 4, 5, 6, 7);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
    if yarim {
        gecis1_16x16_yarim_x8(coeff, yatay, chunk_mut(&mut sutun, 0));
    } else {
        gecis1_16x16_tam_z16(coeff, yatay, &mut sutun);
    }

    sifirla_katsayi::<256>(coeff);
    gecis2_16_x16(&mut sutun, dikey, yarim);
    let mut y = 0;
    while y < 16 {
        ekle_yaz_16x1(dst, y * stride, sutun[y]);
        y += 1;
    }
}

/// `(TX, TXTP)` çiftine özelleştirilmiş 10 bpc AVX512ICL ters dönüşüm çekirdek gövdesi.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn itx10_avx512_cekirdek<const TX: u8, const TXTP: u8>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
) {
    match TxSize::from_index(TX as usize).expect("geçerli tx") {
        TxSize::Tx8x8 => blok_8x8::<TXTP>(dst, dst_stride, coeff, eob),
        TxSize::Rtx8x16 => blok_8x16::<TXTP>(dst, dst_stride, coeff, eob),
        TxSize::Rtx16x8 => blok_16x8::<TXTP>(dst, dst_stride, coeff, eob),
        TxSize::Tx16x16 => blok_16x16::<TXTP>(dst, dst_stride, coeff, eob),
        _ => unreachable!(),
    }
}

/// `(TX, TXTP)` çiftinin sözleşme denetimli güvenli 10 bpc AVX512ICL yuva girişi.
fn itx10_avx512_giris<const TX: u8, const TXTP: u8>(
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir (`itx10_avx512_fn`);
    // belirteç AVX512ICL (F/BW/VL/VBMI/VNNI) tespiti ve maske izninin kanıtıdır. (2) sınır:
    // `itx16_sozlesmesi` `dst_stride`'ın çift ve `≥ w·2` olduğunu, `dst`'nin
    // `(h − 1)·dst_stride + w·2` baytı ve `coeff`'un `1024` öğeyi kapsadığını garanti eder; bütün
    // bellek erişimleri sabit boyutlu `chunk`/`chunk_mut` dizileri üzerindendir. (3) hizalama/takma ad:
    // `dst` ve `coeff` ayrık `&mut` referanslardır, yalnız hizasız erişim kullanılır.
    unsafe {
        itx10_avx512_cekirdek::<TX, TXTP>(dst, dst_stride, coeff, eob);
    }
}

macro_rules! esle_16_tur {
    ($tx:ident, $txtp:expr) => {
        match $txtp {
            TxType::DctDct => {
                Some(itx10_avx512_giris::<{ TxSize::$tx as u8 }, { TxType::DctDct as u8 }>)
            }
            TxType::AdstDct => {
                Some(itx10_avx512_giris::<{ TxSize::$tx as u8 }, { TxType::AdstDct as u8 }>)
            }
            TxType::DctAdst => {
                Some(itx10_avx512_giris::<{ TxSize::$tx as u8 }, { TxType::DctAdst as u8 }>)
            }
            TxType::AdstAdst => {
                Some(itx10_avx512_giris::<{ TxSize::$tx as u8 }, { TxType::AdstAdst as u8 }>)
            }
            TxType::FlipadstDct => {
                Some(itx10_avx512_giris::<{ TxSize::$tx as u8 }, { TxType::FlipadstDct as u8 }>)
            }
            TxType::DctFlipadst => {
                Some(itx10_avx512_giris::<{ TxSize::$tx as u8 }, { TxType::DctFlipadst as u8 }>)
            }
            TxType::FlipadstFlipadst => Some(
                itx10_avx512_giris::<{ TxSize::$tx as u8 }, { TxType::FlipadstFlipadst as u8 }>,
            ),
            TxType::AdstFlipadst => {
                Some(itx10_avx512_giris::<{ TxSize::$tx as u8 }, { TxType::AdstFlipadst as u8 }>)
            }
            TxType::FlipadstAdst => {
                Some(itx10_avx512_giris::<{ TxSize::$tx as u8 }, { TxType::FlipadstAdst as u8 }>)
            }
            TxType::Idtx => {
                Some(itx10_avx512_giris::<{ TxSize::$tx as u8 }, { TxType::Idtx as u8 }>)
            }
            TxType::VDct => {
                Some(itx10_avx512_giris::<{ TxSize::$tx as u8 }, { TxType::VDct as u8 }>)
            }
            TxType::HDct => {
                Some(itx10_avx512_giris::<{ TxSize::$tx as u8 }, { TxType::HDct as u8 }>)
            }
            TxType::VAdst => {
                Some(itx10_avx512_giris::<{ TxSize::$tx as u8 }, { TxType::VAdst as u8 }>)
            }
            TxType::HAdst => {
                Some(itx10_avx512_giris::<{ TxSize::$tx as u8 }, { TxType::HAdst as u8 }>)
            }
            TxType::VFlipadst => {
                Some(itx10_avx512_giris::<{ TxSize::$tx as u8 }, { TxType::VFlipadst as u8 }>)
            }
            TxType::HFlipadst => {
                Some(itx10_avx512_giris::<{ TxSize::$tx as u8 }, { TxType::HFlipadst as u8 }>)
            }
            TxType::WhtWht => None,
        }
    };
}

/// `8 ≤ w, h ≤ 16` boyutlu 60 adet 10 bpc WHT-dışı AVX512ICL `itxfm_add[tx][txtp]` yuvası
/// için [`Avx512IclToken`] ile kapılı `ItxFn` kurucusu.
#[must_use]
pub const fn itx10_avx512_fn(_token: Avx512IclToken, tx: TxSize, txtp: TxType) -> Option<ItxFn> {
    match tx {
        TxSize::Tx8x8 => esle_16_tur!(Tx8x8, txtp),
        TxSize::Rtx8x16 => esle_16_tur!(Rtx8x16, txtp),
        TxSize::Rtx16x8 => esle_16_tur!(Rtx16x8, txtp),
        TxSize::Tx16x16 => match txtp {
            TxType::DctDct => {
                Some(itx10_avx512_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::DctDct as u8 }>)
            }
            TxType::AdstDct => {
                Some(itx10_avx512_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::AdstDct as u8 }>)
            }
            TxType::DctAdst => {
                Some(itx10_avx512_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::DctAdst as u8 }>)
            }
            TxType::AdstAdst => {
                Some(itx10_avx512_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::AdstAdst as u8 }>)
            }
            TxType::FlipadstDct => {
                Some(itx10_avx512_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::FlipadstDct as u8 }>)
            }
            TxType::DctFlipadst => {
                Some(itx10_avx512_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::DctFlipadst as u8 }>)
            }
            TxType::FlipadstFlipadst => Some(
                itx10_avx512_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::FlipadstFlipadst as u8 }>,
            ),
            TxType::AdstFlipadst => Some(
                itx10_avx512_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::AdstFlipadst as u8 }>,
            ),
            TxType::FlipadstAdst => Some(
                itx10_avx512_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::FlipadstAdst as u8 }>,
            ),
            TxType::Idtx => {
                Some(itx10_avx512_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::Idtx as u8 }>)
            }
            TxType::VDct => {
                Some(itx10_avx512_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::VDct as u8 }>)
            }
            TxType::HDct => {
                Some(itx10_avx512_giris::<{ TxSize::Tx16x16 as u8 }, { TxType::HDct as u8 }>)
            }
            _ => None,
        },
        _ => None,
    }
}

/// `8 ≤ w, h ≤ 16` boyutlu 60 adet 10 bpc WHT-dışı AVX512ICL `itxfm_add[tx][txtp]` yuvası
/// için çalışma zamanı tablo araması.
#[must_use]
pub const fn itx10_avx512_lookup(token: Avx512IclToken, tx: TxSize, txtp: TxType) -> Option<ItxFn> {
    itx10_avx512_fn(token, tx, txtp)
}
