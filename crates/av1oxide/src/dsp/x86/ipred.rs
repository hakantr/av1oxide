//! 8 bpc intra tahmin (`intra_pred[14]`, `cfl_ac[3]`, `cfl_pred[4]`, `pal_pred`;
//! 22 kayıt) x86_64 AVX2 çekirdekleri — P45 Adım 2 (Paket A).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Kayıtlar | Yuva | Kaynak |
//! |---|---|---|
//! | `D01618`, `D01623`, `D01628`, `D01634`, `D01639`, `D01644` | `intra_pred[DC/VERT/HOR/LEFT_DC/TOP_DC/DC_128]` | `src/x86/ipred_avx2.asm:341-724`, `src/x86/ipred.h:103-108` |
//! | `D01650`, `D01656`, `D01662` | `intra_pred[Z1/Z2/Z3]` | `src/x86/ipred_avx2.asm:725-1840`, `src/x86/ipred.h:109-111` |
//! | `D01668`, `D01673`, `D01678` | `intra_pred[SMOOTH/SMOOTH_V/SMOOTH_H]` | `src/x86/ipred_avx2.asm:2218-2810`, `src/x86/ipred.h:112-114` |
//! | `D01684` | `intra_pred[PAETH]` | `src/x86/ipred_avx2.asm:1841-2217`, `src/x86/ipred.h:115` |
//! | `D01690` | `intra_pred[FILTER]` | `src/x86/ipred_avx2.asm:3684-4110`, `src/x86/ipred.h:116` |
//! | `D01590`, `D01594`, `D01598` | `cfl_ac[420/422/444]` | `src/x86/ipred_avx2.asm:3138-3608`, `src/x86/ipred.h:118-120` |
//! | `D01602`, `D01606`, `D01610`, `D01614` | `cfl_pred[DC/LEFT_DC/TOP_DC/DC_128]` | `src/x86/ipred_avx2.asm:2811-3137`, `src/x86/ipred.h:122-125` |
//! | `D01696` | `pal_pred` | `src/x86/ipred_avx2.asm:3609-3683`, `src/x86/ipred.h:127` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bu modülde tüm SIMD işlevleri yalnız `#[target_feature(enable = "avx2")]`
//!   taşır; EVEX yönergesi üretilmez (`0` EVEX).
//! - Kopyasız O(1) sözleşme: girişte [`angular_sozlesmesi::<8>`],
//!   [`cfl_ac_sozlesmesi::<8>`], [`cfl_pred_sozlesmesi::<8>`],
//!   [`pal_pred_sozlesmesi::<8>`] denetlenir; sözleşme dışı girdide ilgili
//!   `*_skaler_devret::<8>` ile skaler gövdeye devredilir.
//! - `OverReadProbe` güvenliği: `edge.bytes` üzerinde yalnız izinli
//!   `[topleft - back .. topleft + 1 + front]` aralığı okunur; `cfl_ac` içinde
//!   yalnız görünür luma bölgesi okunur; `dst` üzerinde her satırda yalnız
//!   `dst[y * dst_stride .. y * dst_stride + w]` baytlarına yazılır.

#![allow(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_abs_epi16, _mm_add_epi16, _mm_add_epi32, _mm_and_si128, _mm_andnot_si128,
    _mm_avg_epu16, _mm_cmpgt_epi16, _mm_cvtsi32_si128, _mm_cvtsi64_si128, _mm_cvtsi128_si32,
    _mm_maddubs_epi16, _mm_mulhrs_epi16, _mm_mullo_epi16, _mm_or_si128, _mm_packus_epi16,
    _mm_sad_epu8, _mm_set1_epi8, _mm_set1_epi16, _mm_setr_epi8, _mm_setzero_si128,
    _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_sign_epi16, _mm_slli_epi16, _mm_srai_epi16,
    _mm_srli_epi16, _mm_sub_epi16, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64,
    _mm_unpacklo_epi8, _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm256_abs_epi16, _mm256_add_epi16,
    _mm256_add_epi32, _mm256_and_si256, _mm256_andnot_si256, _mm256_avg_epu16,
    _mm256_castsi256_si128, _mm256_cmpgt_epi16, _mm256_cvtepu8_epi16, _mm256_extracti128_si256,
    _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_mulhrs_epi16, _mm256_mullo_epi16,
    _mm256_or_si256, _mm256_packus_epi16, _mm256_permute4x64_epi64, _mm256_sad_epu8,
    _mm256_set1_epi8, _mm256_set1_epi16, _mm256_set1_epi64x, _mm256_setr_epi8,
    _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_sign_epi16, _mm256_slli_epi16,
    _mm256_srli_epi16, _mm256_sub_epi16, _mm256_unpackhi_epi8, _mm256_unpacklo_epi8,
};

use crate::cpu::Avx2Token;
use crate::dsp::ipred::{
    AngularPredFn, CflAcFn, CflPredFn, CflPredMode, Edge, IntraPredMode, PalPredFn,
    angular_skaler_devret, angular_sozlesmesi, cfl_ac_skaler_devret, cfl_ac_sozlesmesi,
    cfl_pred_skaler_devret, cfl_pred_sozlesmesi, pal_pred_skaler_devret, pal_pred_sozlesmesi,
};
use crate::dsp::ipred_tables::{DR_INTRA_DERIVATIVE, FILTER_INTRA_TAPS, SM_WEIGHTS};
use crate::dsp::x86::mem;
use crate::frame::BitDepth;

const TMP_LEN: usize = 257;
const TMP_BASE: isize = 128;

// ---------------------------------------------------------------------------
// Kurucular (`Avx2Token` kapılı, çalışma zamanında `Token::new` çağrılmaz)
// ---------------------------------------------------------------------------

/// 8 bpc `intra_pred[mode]` AVX2 yuva işaretçisi (`D01618`..`D01690`).
#[must_use]
pub const fn intra_pred_fn(_token: Avx2Token, mode: IntraPredMode) -> AngularPredFn {
    match mode {
        IntraPredMode::Dc => ipred_dc_avx2,
        IntraPredMode::Vert => ipred_vert_avx2,
        IntraPredMode::Hor => ipred_hor_avx2,
        IntraPredMode::LeftDc => ipred_left_dc_avx2,
        IntraPredMode::TopDc => ipred_top_dc_avx2,
        IntraPredMode::Dc128 => ipred_dc_128_avx2,
        IntraPredMode::Z1 => ipred_z1_avx2,
        IntraPredMode::Z2 => ipred_z2_avx2,
        IntraPredMode::Z3 => ipred_z3_avx2,
        IntraPredMode::Smooth => ipred_smooth_avx2,
        IntraPredMode::SmoothV => ipred_smooth_v_avx2,
        IntraPredMode::SmoothH => ipred_smooth_h_avx2,
        IntraPredMode::Paeth => ipred_paeth_avx2,
        IntraPredMode::Filter => ipred_filter_avx2,
    }
}

/// 8 bpc `cfl_ac[layout]` AVX2 yuva işaretçisi (`D01590`, `D01594`, `D01598`).
#[must_use]
pub const fn cfl_ac_fn(_token: Avx2Token, layout: usize) -> Option<CflAcFn> {
    match layout {
        0 => Some(cfl_ac_420_avx2),
        1 => Some(cfl_ac_422_avx2),
        2 => Some(cfl_ac_444_avx2),
        _ => None,
    }
}

/// 8 bpc `cfl_pred[mode]` AVX2 yuva işaretçisi (`D01602`, `D01606`, `D01610`, `D01614`).
#[must_use]
pub const fn cfl_pred_fn(_token: Avx2Token, mode: CflPredMode) -> CflPredFn {
    match mode {
        CflPredMode::Dc => cfl_pred_dc_avx2,
        CflPredMode::LeftDc => cfl_pred_left_dc_avx2,
        CflPredMode::TopDc => cfl_pred_top_dc_avx2,
        CflPredMode::Dc128 => cfl_pred_dc_128_avx2,
    }
}

/// 8 bpc `pal_pred` AVX2 yuva işaretçisi (`D01696`).
#[must_use]
pub const fn pal_pred_fn(_token: Avx2Token) -> PalPredFn {
    pal_pred_avx2
}

// ---------------------------------------------------------------------------
// Ortak yardımcılar (`#[target_feature(enable = "avx2")]`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
fn sum_bytes_avx2(src: &[u8], n: usize) -> u32 {
    let zero128 = _mm_setzero_si128();
    let zero256 = _mm256_setzero_si256();
    match n {
        4 => {
            let v = u32::from_le_bytes(*mem::chunk::<u8, 4>(src, 0));
            _mm_cvtsi128_si32(_mm_sad_epu8(_mm_cvtsi32_si128(v as i32), zero128)) as u32
        }
        8 => {
            let v = u64::from_le_bytes(*mem::chunk::<u8, 8>(src, 0));
            _mm_cvtsi128_si32(_mm_sad_epu8(_mm_cvtsi64_si128(v as i64), zero128)) as u32
        }
        16 => {
            let s = _mm_sad_epu8(mem::load_u8x16(mem::chunk::<u8, 16>(src, 0)), zero128);
            let s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, zero128));
            _mm_cvtsi128_si32(s) as u32
        }
        32 => {
            let s = _mm256_sad_epu8(mem::load_u8x32(mem::chunk::<u8, 32>(src, 0)), zero256);
            let lo = _mm256_castsi256_si128(s);
            let hi = _mm256_extracti128_si256::<1>(s);
            let sum = _mm_add_epi32(lo, hi);
            let sum = _mm_add_epi32(sum, _mm_unpackhi_epi64(sum, zero128));
            _mm_cvtsi128_si32(sum) as u32
        }
        64 => {
            let s0 = _mm256_sad_epu8(mem::load_u8x32(mem::chunk::<u8, 32>(src, 0)), zero256);
            let s1 = _mm256_sad_epu8(mem::load_u8x32(mem::chunk::<u8, 32>(src, 32)), zero256);
            let s = _mm256_add_epi32(s0, s1);
            let lo = _mm256_castsi256_si128(s);
            let hi = _mm256_extracti128_si256::<1>(s);
            let sum = _mm_add_epi32(lo, hi);
            let sum = _mm_add_epi32(sum, _mm_unpackhi_epi64(sum, zero128));
            _mm_cvtsi128_si32(sum) as u32
        }
        _ => unreachable!(),
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn dc_gen_top_avx2(edge: Edge<'_>, width: usize) -> u32 {
    let top = &edge.bytes[edge.topleft + 1..edge.topleft + 1 + width];
    (sum_bytes_avx2(top, width) + (width as u32 >> 1)) >> width.trailing_zeros()
}

#[target_feature(enable = "avx2")]
#[inline]
fn dc_gen_left_avx2(edge: Edge<'_>, height: usize) -> u32 {
    let left = &edge.bytes[edge.topleft - height..edge.topleft];
    (sum_bytes_avx2(left, height) + (height as u32 >> 1)) >> height.trailing_zeros()
}

#[target_feature(enable = "avx2")]
#[inline]
fn dc_gen_avx2(edge: Edge<'_>, width: usize, height: usize) -> u32 {
    let top = &edge.bytes[edge.topleft + 1..edge.topleft + 1 + width];
    let left = &edge.bytes[edge.topleft - height..edge.topleft];
    let mut dc =
        sum_bytes_avx2(top, width) + sum_bytes_avx2(left, height) + ((width + height) as u32 >> 1);
    dc >>= (width + height).trailing_zeros();
    if width != height {
        let mul = if width > height * 2 || height > width * 2 {
            0x3334u64
        } else {
            0x5556u64
        };
        dc = ((u64::from(dc) * mul) >> 16) as u32;
    }
    dc
}

#[target_feature(enable = "avx2")]
#[inline]
fn splat_dc_avx2(dst: &mut [u8], stride: usize, width: usize, height: usize, dc: i8) {
    match width {
        4 => {
            let v = _mm_set1_epi8(dc);
            for y in 0..height {
                mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, y * stride), v);
            }
        }
        8 => {
            let v = _mm_set1_epi8(dc);
            for y in 0..height {
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, y * stride), v);
            }
        }
        16 => {
            let v = _mm_set1_epi8(dc);
            for y in 0..height {
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, y * stride), v);
            }
        }
        32 => {
            let v = _mm256_set1_epi8(dc);
            for y in 0..height {
                mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, y * stride), v);
            }
        }
        64 => {
            let v = _mm256_set1_epi8(dc);
            for y in 0..height {
                let row = y * stride;
                mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row), v);
                mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row + 32), v);
            }
        }
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// 1. DC / VERT / HOR / LEFT_DC / TOP_DC / DC_128 (`D01618`..`D01644`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
fn ipred_dc_avx2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let dc = dc_gen_avx2(edge, width, height) as i8;
    splat_dc_avx2(dst, dst_stride, width, height, dc);
}

#[target_feature(enable = "avx2")]
fn ipred_top_dc_avx2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let dc = dc_gen_top_avx2(edge, width) as i8;
    splat_dc_avx2(dst, dst_stride, width, height, dc);
}

#[target_feature(enable = "avx2")]
fn ipred_left_dc_avx2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let dc = dc_gen_left_avx2(edge, height) as i8;
    splat_dc_avx2(dst, dst_stride, width, height, dc);
}

#[target_feature(enable = "avx2")]
fn ipred_dc_128_avx2_giris(dst: &mut [u8], dst_stride: usize, width: usize, height: usize) {
    splat_dc_avx2(dst, dst_stride, width, height, -128i8);
}

#[target_feature(enable = "avx2")]
fn ipred_vert_avx2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let top = &edge.bytes[edge.topleft + 1..edge.topleft + 1 + width];
    match width {
        4 => {
            let v = _mm_cvtsi32_si128(u32::from_le_bytes(*mem::chunk::<u8, 4>(top, 0)) as i32);
            for y in 0..height {
                mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, y * dst_stride), v);
            }
        }
        8 => {
            let v = _mm_cvtsi64_si128(u64::from_le_bytes(*mem::chunk::<u8, 8>(top, 0)) as i64);
            for y in 0..height {
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, y * dst_stride), v);
            }
        }
        16 => {
            let v = mem::load_u8x16(mem::chunk::<u8, 16>(top, 0));
            for y in 0..height {
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, y * dst_stride), v);
            }
        }
        32 => {
            let v = mem::load_u8x32(mem::chunk::<u8, 32>(top, 0));
            for y in 0..height {
                mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, y * dst_stride), v);
            }
        }
        64 => {
            let v0 = mem::load_u8x32(mem::chunk::<u8, 32>(top, 0));
            let v1 = mem::load_u8x32(mem::chunk::<u8, 32>(top, 32));
            for y in 0..height {
                let row = y * dst_stride;
                mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row), v0);
                mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row + 32), v1);
            }
        }
        _ => unreachable!(),
    }
}

#[target_feature(enable = "avx2")]
fn ipred_hor_avx2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let left = &edge.bytes[edge.topleft - height..edge.topleft];
    let shuf3 = _mm256_set1_epi8(3);
    let shuf2 = _mm256_set1_epi8(2);
    let shuf1 = _mm256_set1_epi8(1);
    let shuf0 = _mm256_set1_epi8(0);
    let mut y = 0;
    while y < height {
        let chunk4 = u32::from_le_bytes(*mem::chunk::<u8, 4>(left, height - 4 - y));
        let base = _mm256_set1_epi64x(i64::from(chunk4));
        let r0 = _mm256_shuffle_epi8(base, shuf3);
        let r1 = _mm256_shuffle_epi8(base, shuf2);
        let r2 = _mm256_shuffle_epi8(base, shuf1);
        let r3 = _mm256_shuffle_epi8(base, shuf0);
        for (dy, rv) in [(0, r0), (1, r1), (2, r2), (3, r3)] {
            let row = (y + dy) * dst_stride;
            let rv128 = _mm256_castsi256_si128(rv);
            match width {
                4 => mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, row), rv128),
                8 => mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, row), rv128),
                16 => mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row), rv128),
                32 => mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row), rv),
                64 => {
                    mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row), rv);
                    mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row + 32), rv);
                }
                _ => unreachable!(),
            }
        }
        y += 4;
    }
}

// ---------------------------------------------------------------------------
// 2. PAETH (`D01684`) — 128/256-bit AVX2
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
fn paeth_row8_avx2(
    t16: __m128i,
    ldiff16: __m128i,
    l16: __m128i,
    tl16: __m128i,
    tdiff16: __m128i,
) -> __m128i {
    let t_minus_tl = _mm_sub_epi16(t16, tl16);
    let l_minus_tl = _mm_sub_epi16(l16, tl16);
    let tldiff16 = _mm_abs_epi16(_mm_add_epi16(l_minus_tl, t_minus_tl));
    let gt_lt = _mm_cmpgt_epi16(ldiff16, tdiff16);
    let gt_ltl = _mm_cmpgt_epi16(ldiff16, tldiff16);
    let not_l_mask = _mm_or_si128(gt_lt, gt_ltl);
    let not_t_mask = _mm_cmpgt_epi16(tdiff16, tldiff16);
    let t_or_tl = _mm_or_si128(
        _mm_and_si128(not_t_mask, tl16),
        _mm_andnot_si128(not_t_mask, t16),
    );
    _mm_or_si128(
        _mm_and_si128(not_l_mask, t_or_tl),
        _mm_andnot_si128(not_l_mask, l16),
    )
}

#[target_feature(enable = "avx2")]
#[inline]
fn paeth_row16_avx2(
    t16: __m256i,
    ldiff16: __m256i,
    l16: __m256i,
    tl16: __m256i,
    tdiff16: __m256i,
) -> __m256i {
    let t_minus_tl = _mm256_sub_epi16(t16, tl16);
    let l_minus_tl = _mm256_sub_epi16(l16, tl16);
    let tldiff16 = _mm256_abs_epi16(_mm256_add_epi16(l_minus_tl, t_minus_tl));
    let gt_lt = _mm256_cmpgt_epi16(ldiff16, tdiff16);
    let gt_ltl = _mm256_cmpgt_epi16(ldiff16, tldiff16);
    let not_l_mask = _mm256_or_si256(gt_lt, gt_ltl);
    let not_t_mask = _mm256_cmpgt_epi16(tdiff16, tldiff16);
    let t_or_tl = _mm256_or_si256(
        _mm256_and_si256(not_t_mask, tl16),
        _mm256_andnot_si256(not_t_mask, t16),
    );
    _mm256_or_si256(
        _mm256_and_si256(not_l_mask, t_or_tl),
        _mm256_andnot_si256(not_l_mask, l16),
    )
}

#[target_feature(enable = "avx2")]
fn ipred_paeth_avx2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let tl = i16::from(edge.bytes[edge.topleft]);
    let top_slice = &edge.bytes[edge.topleft + 1..edge.topleft + 1 + width];
    let left_slice = &edge.bytes[edge.topleft - height..edge.topleft];

    if width <= 8 {
        let zero = _mm_setzero_si128();
        let tl16 = _mm_set1_epi16(tl);
        let mut top_buf = [0u8; 8];
        top_buf[..width].copy_from_slice(top_slice);
        let raw8 = u64::from_le_bytes(top_buf);
        let t16 = _mm_unpacklo_epi8(_mm_cvtsi64_si128(raw8 as i64), zero);
        let ldiff16 = _mm_abs_epi16(_mm_sub_epi16(t16, tl16));
        for y in 0..height {
            let l = i16::from(left_slice[height - 1 - y]);
            let l16 = _mm_set1_epi16(l);
            let tdiff16 = _mm_set1_epi16((l - tl).abs());
            let p0 = paeth_row8_avx2(t16, ldiff16, l16, tl16, tdiff16);
            let packed = _mm_packus_epi16(p0, p0);
            let row = y * dst_stride;
            if width == 4 {
                mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, row), packed);
            } else {
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, row), packed);
            }
        }
        return;
    }

    // width ∈ {16, 32, 64}: 256-bit (16 × i16) şeritlerle işle
    let tl256 = _mm256_set1_epi16(tl);
    let mut top256 = [_mm256_setzero_si256(); 4];
    let mut ldiff256 = [_mm256_setzero_si256(); 4];
    let chunks16 = width / 16;
    for c in 0..chunks16 {
        let mut raw16 = [0i16; 16];
        for i in 0..16 {
            raw16[i] = i16::from(top_slice[c * 16 + i]);
        }
        let t16 = mem::load_i16x16(&raw16);
        top256[c] = t16;
        ldiff256[c] = _mm256_abs_epi16(_mm256_sub_epi16(t16, tl256));
    }

    for y in 0..height {
        let l = i16::from(left_slice[height - 1 - y]);
        let l256 = _mm256_set1_epi16(l);
        let tdiff256 = _mm256_set1_epi16((l - tl).abs());
        let row = y * dst_stride;
        match width {
            16 => {
                let p0 = paeth_row16_avx2(top256[0], ldiff256[0], l256, tl256, tdiff256);
                let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(p0, p0));
                mem::store_u8x16(
                    mem::chunk_mut::<u8, 16>(dst, row),
                    _mm256_castsi256_si128(packed),
                );
            }
            32 => {
                let p0 = paeth_row16_avx2(top256[0], ldiff256[0], l256, tl256, tdiff256);
                let p1 = paeth_row16_avx2(top256[1], ldiff256[1], l256, tl256, tdiff256);
                let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(p0, p1));
                mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row), packed);
            }
            64 => {
                for g in 0..2 {
                    let p0 =
                        paeth_row16_avx2(top256[2 * g], ldiff256[2 * g], l256, tl256, tdiff256);
                    let p1 = paeth_row16_avx2(
                        top256[2 * g + 1],
                        ldiff256[2 * g + 1],
                        l256,
                        tl256,
                        tdiff256,
                    );
                    let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(p0, p1));
                    mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row + g * 32), packed);
                }
            }
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------
// 3. SMOOTH / SMOOTH_V / SMOOTH_H (`D01668`, `D01673`, `D01678`) — 128/256-bit AVX2
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
fn smooth_term8_avx2(
    a16: __m128i,
    b16: __m128i,
    w16: __m128i,
    inv_w16: __m128i,
    bias16: __m128i,
) -> __m128i {
    let p0 = _mm_mullo_epi16(a16, w16);
    let p1 = _mm_mullo_epi16(b16, inv_w16);
    _mm_add_epi16(_mm_add_epi16(p0, p1), bias16)
}

#[target_feature(enable = "avx2")]
#[inline]
fn smooth_term16_avx2(
    a16: __m256i,
    b16: __m256i,
    w16: __m256i,
    inv_w16: __m256i,
    bias16: __m256i,
) -> __m256i {
    let p0 = _mm256_mullo_epi16(a16, w16);
    let p1 = _mm256_mullo_epi16(b16, inv_w16);
    _mm256_add_epi16(_mm256_add_epi16(p0, p1), bias16)
}

#[target_feature(enable = "avx2")]
fn ipred_smooth_v_avx2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let weights_ver = &SM_WEIGHTS[height..2 * height];
    let bottom = i16::from(edge.bytes[edge.topleft - height]);
    let top_slice = &edge.bytes[edge.topleft + 1..edge.topleft + 1 + width];

    if width <= 8 {
        let zero = _mm_setzero_si128();
        let bottom16 = _mm_set1_epi16(bottom);
        let bias128 = _mm_set1_epi16(128);
        let mut top_buf = [0u8; 8];
        top_buf[..width].copy_from_slice(top_slice);
        let top16 = _mm_unpacklo_epi8(_mm_cvtsi64_si128(u64::from_le_bytes(top_buf) as i64), zero);
        for y in 0..height {
            let wv = i16::from(weights_ver[y]);
            let wv16 = _mm_set1_epi16(wv);
            let inv_wv16 = _mm_set1_epi16(256 - wv);
            let s0 =
                _mm_srli_epi16::<8>(smooth_term8_avx2(top16, bottom16, wv16, inv_wv16, bias128));
            let packed = _mm_packus_epi16(s0, s0);
            let row = y * dst_stride;
            if width == 4 {
                mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, row), packed);
            } else {
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, row), packed);
            }
        }
        return;
    }

    let bottom256 = _mm256_set1_epi16(bottom);
    let bias128 = _mm256_set1_epi16(128);
    let mut top256 = [_mm256_setzero_si256(); 4];
    let chunks16 = width / 16;
    for c in 0..chunks16 {
        top256[c] = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(top_slice, c * 16)));
    }

    for y in 0..height {
        let wv = i16::from(weights_ver[y]);
        let wv256 = _mm256_set1_epi16(wv);
        let inv_wv256 = _mm256_set1_epi16(256 - wv);
        let row = y * dst_stride;
        match width {
            16 => {
                let s0 = _mm256_srli_epi16::<8>(smooth_term16_avx2(
                    top256[0], bottom256, wv256, inv_wv256, bias128,
                ));
                let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(s0, s0));
                mem::store_u8x16(
                    mem::chunk_mut::<u8, 16>(dst, row),
                    _mm256_castsi256_si128(packed),
                );
            }
            32 => {
                let s0 = _mm256_srli_epi16::<8>(smooth_term16_avx2(
                    top256[0], bottom256, wv256, inv_wv256, bias128,
                ));
                let s1 = _mm256_srli_epi16::<8>(smooth_term16_avx2(
                    top256[1], bottom256, wv256, inv_wv256, bias128,
                ));
                let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(s0, s1));
                mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row), packed);
            }
            64 => {
                for g in 0..2 {
                    let s0 = _mm256_srli_epi16::<8>(smooth_term16_avx2(
                        top256[2 * g],
                        bottom256,
                        wv256,
                        inv_wv256,
                        bias128,
                    ));
                    let s1 = _mm256_srli_epi16::<8>(smooth_term16_avx2(
                        top256[2 * g + 1],
                        bottom256,
                        wv256,
                        inv_wv256,
                        bias128,
                    ));
                    let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(s0, s1));
                    mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row + g * 32), packed);
                }
            }
            _ => unreachable!(),
        }
    }
}

#[target_feature(enable = "avx2")]
fn ipred_smooth_h_avx2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let weights_hor = &SM_WEIGHTS[width..2 * width];
    let right = i16::from(edge.bytes[edge.topleft + width]);
    let left_slice = &edge.bytes[edge.topleft - height..edge.topleft];

    if width <= 8 {
        let zero = _mm_setzero_si128();
        let right16 = _mm_set1_epi16(right);
        let bias128 = _mm_set1_epi16(128);
        let c256 = _mm_set1_epi16(256);
        let mut wh_buf = [0u8; 8];
        wh_buf[..width].copy_from_slice(weights_hor);
        let wh16 = _mm_unpacklo_epi8(_mm_cvtsi64_si128(u64::from_le_bytes(wh_buf) as i64), zero);
        let inv_wh16 = _mm_sub_epi16(c256, wh16);
        for y in 0..height {
            let l16 = _mm_set1_epi16(i16::from(left_slice[height - 1 - y]));
            let s0 = _mm_srli_epi16::<8>(smooth_term8_avx2(l16, right16, wh16, inv_wh16, bias128));
            let packed = _mm_packus_epi16(s0, s0);
            let row = y * dst_stride;
            if width == 4 {
                mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, row), packed);
            } else {
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, row), packed);
            }
        }
        return;
    }

    let right256 = _mm256_set1_epi16(right);
    let bias128 = _mm256_set1_epi16(128);
    let c256 = _mm256_set1_epi16(256);
    match width {
        16 => {
            let w0 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(weights_hor, 0)));
            let r0 = _mm256_add_epi16(
                _mm256_mullo_epi16(right256, _mm256_sub_epi16(c256, w0)),
                bias128,
            );
            for y in 0..height {
                let l256 = _mm256_set1_epi16(i16::from(left_slice[height - 1 - y]));
                let s0 = _mm256_srli_epi16::<8>(_mm256_add_epi16(_mm256_mullo_epi16(l256, w0), r0));
                let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(s0, s0));
                mem::store_u8x16(
                    mem::chunk_mut::<u8, 16>(dst, y * dst_stride),
                    _mm256_castsi256_si128(packed),
                );
            }
        }
        32 => {
            let w0 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(weights_hor, 0)));
            let w1 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(weights_hor, 16)));
            let r0 = _mm256_add_epi16(
                _mm256_mullo_epi16(right256, _mm256_sub_epi16(c256, w0)),
                bias128,
            );
            let r1 = _mm256_add_epi16(
                _mm256_mullo_epi16(right256, _mm256_sub_epi16(c256, w1)),
                bias128,
            );
            for y in 0..height {
                let l256 = _mm256_set1_epi16(i16::from(left_slice[height - 1 - y]));
                let s0 = _mm256_srli_epi16::<8>(_mm256_add_epi16(_mm256_mullo_epi16(l256, w0), r0));
                let s1 = _mm256_srli_epi16::<8>(_mm256_add_epi16(_mm256_mullo_epi16(l256, w1), r1));
                let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(s0, s1));
                mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, y * dst_stride), packed);
            }
        }
        64 => {
            let w0 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(weights_hor, 0)));
            let w1 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(weights_hor, 16)));
            let w2 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(weights_hor, 32)));
            let w3 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(weights_hor, 48)));
            let r0 = _mm256_add_epi16(
                _mm256_mullo_epi16(right256, _mm256_sub_epi16(c256, w0)),
                bias128,
            );
            let r1 = _mm256_add_epi16(
                _mm256_mullo_epi16(right256, _mm256_sub_epi16(c256, w1)),
                bias128,
            );
            let r2 = _mm256_add_epi16(
                _mm256_mullo_epi16(right256, _mm256_sub_epi16(c256, w2)),
                bias128,
            );
            let r3 = _mm256_add_epi16(
                _mm256_mullo_epi16(right256, _mm256_sub_epi16(c256, w3)),
                bias128,
            );
            for y in 0..height {
                let l256 = _mm256_set1_epi16(i16::from(left_slice[height - 1 - y]));
                let row = y * dst_stride;
                let s0 = _mm256_srli_epi16::<8>(_mm256_add_epi16(_mm256_mullo_epi16(l256, w0), r0));
                let s1 = _mm256_srli_epi16::<8>(_mm256_add_epi16(_mm256_mullo_epi16(l256, w1), r1));
                let packed0 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(s0, s1));
                mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row), packed0);
                let s2 = _mm256_srli_epi16::<8>(_mm256_add_epi16(_mm256_mullo_epi16(l256, w2), r2));
                let s3 = _mm256_srli_epi16::<8>(_mm256_add_epi16(_mm256_mullo_epi16(l256, w3), r3));
                let packed1 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(s2, s3));
                mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row + 32), packed1);
            }
        }
        _ => unreachable!(),
    }
}

#[target_feature(enable = "avx2")]
fn ipred_smooth_avx2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let weights_hor = &SM_WEIGHTS[width..2 * width];
    let weights_ver = &SM_WEIGHTS[height..2 * height];
    let right = i16::from(edge.bytes[edge.topleft + width]);
    let bottom = i16::from(edge.bytes[edge.topleft - height]);
    let top_slice = &edge.bytes[edge.topleft + 1..edge.topleft + 1 + width];
    let left_slice = &edge.bytes[edge.topleft - height..edge.topleft];

    if width <= 8 {
        let zero = _mm_setzero_si128();
        let right16 = _mm_set1_epi16(right);
        let bottom16 = _mm_set1_epi16(bottom);
        let bias255 = _mm_set1_epi16(255);
        let c256 = _mm_set1_epi16(256);
        let mut top_buf = [0u8; 8];
        top_buf[..width].copy_from_slice(top_slice);
        let top16 = _mm_unpacklo_epi8(_mm_cvtsi64_si128(u64::from_le_bytes(top_buf) as i64), zero);
        let mut wh_buf = [0u8; 8];
        wh_buf[..width].copy_from_slice(weights_hor);
        let wh16 = _mm_unpacklo_epi8(_mm_cvtsi64_si128(u64::from_le_bytes(wh_buf) as i64), zero);
        let inv_wh16 = _mm_sub_epi16(c256, wh16);
        for y in 0..height {
            let wv = i16::from(weights_ver[y]);
            let wv16 = _mm_set1_epi16(wv);
            let inv_wv16 = _mm_set1_epi16(256 - wv);
            let l16 = _mm_set1_epi16(i16::from(left_slice[height - 1 - y]));
            let v = smooth_term8_avx2(top16, bottom16, wv16, inv_wv16, bias255);
            let h = smooth_term8_avx2(l16, right16, wh16, inv_wh16, zero);
            let s0 = _mm_srli_epi16::<8>(_mm_avg_epu16(v, h));
            let packed = _mm_packus_epi16(s0, s0);
            let row = y * dst_stride;
            if width == 4 {
                mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, row), packed);
            } else {
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, row), packed);
            }
        }
        return;
    }

    let zero256 = _mm256_setzero_si256();
    let right256 = _mm256_set1_epi16(right);
    let bottom256 = _mm256_set1_epi16(bottom);
    let bias255 = _mm256_set1_epi16(255);
    let c256 = _mm256_set1_epi16(256);

    let mut top256 = [_mm256_setzero_si256(); 4];
    let mut wh256 = [_mm256_setzero_si256(); 4];
    let mut inv_wh256 = [_mm256_setzero_si256(); 4];
    let chunks16 = width / 16;
    for c in 0..chunks16 {
        top256[c] = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(top_slice, c * 16)));
        let w = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(weights_hor, c * 16)));
        wh256[c] = w;
        inv_wh256[c] = _mm256_sub_epi16(c256, w);
    }

    for y in 0..height {
        let wv = i16::from(weights_ver[y]);
        let wv256 = _mm256_set1_epi16(wv);
        let inv_wv256 = _mm256_set1_epi16(256 - wv);
        let l256 = _mm256_set1_epi16(i16::from(left_slice[height - 1 - y]));
        let row = y * dst_stride;
        match width {
            16 => {
                let v = smooth_term16_avx2(top256[0], bottom256, wv256, inv_wv256, bias255);
                let h = smooth_term16_avx2(l256, right256, wh256[0], inv_wh256[0], zero256);
                let s0 = _mm256_srli_epi16::<8>(_mm256_avg_epu16(v, h));
                let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(s0, s0));
                mem::store_u8x16(
                    mem::chunk_mut::<u8, 16>(dst, row),
                    _mm256_castsi256_si128(packed),
                );
            }
            32 => {
                let v0 = smooth_term16_avx2(top256[0], bottom256, wv256, inv_wv256, bias255);
                let h0 = smooth_term16_avx2(l256, right256, wh256[0], inv_wh256[0], zero256);
                let s0 = _mm256_srli_epi16::<8>(_mm256_avg_epu16(v0, h0));
                let v1 = smooth_term16_avx2(top256[1], bottom256, wv256, inv_wv256, bias255);
                let h1 = smooth_term16_avx2(l256, right256, wh256[1], inv_wh256[1], zero256);
                let s1 = _mm256_srli_epi16::<8>(_mm256_avg_epu16(v1, h1));
                let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(s0, s1));
                mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row), packed);
            }
            64 => {
                for g in 0..2 {
                    let v0 =
                        smooth_term16_avx2(top256[2 * g], bottom256, wv256, inv_wv256, bias255);
                    let h0 =
                        smooth_term16_avx2(l256, right256, wh256[2 * g], inv_wh256[2 * g], zero256);
                    let s0 = _mm256_srli_epi16::<8>(_mm256_avg_epu16(v0, h0));
                    let v1 =
                        smooth_term16_avx2(top256[2 * g + 1], bottom256, wv256, inv_wv256, bias255);
                    let h1 = smooth_term16_avx2(
                        l256,
                        right256,
                        wh256[2 * g + 1],
                        inv_wh256[2 * g + 1],
                        zero256,
                    );
                    let s1 = _mm256_srli_epi16::<8>(_mm256_avg_epu16(v1, h1));
                    let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(s0, s1));
                    mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row + g * 32), packed);
                }
            }
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------
// 4. FILTER_PRED (`D01690`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
fn ipred_filter_avx2_giris(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
) {
    let filter_index = angle & 511;
    let taps_i8 = &FILTER_INTRA_TAPS[filter_index];
    let mut taps_bytes = [0u8; 64];
    for i in 0..64 {
        taps_bytes[i] = taps_i8[i] as u8;
    }
    let t0 = mem::load_u8x16(mem::chunk::<u8, 16>(&taps_bytes, 0));
    let t1 = mem::load_u8x16(mem::chunk::<u8, 16>(&taps_bytes, 16));
    let t2 = mem::load_u8x16(mem::chunk::<u8, 16>(&taps_bytes, 32));
    let t3 = mem::load_u8x16(mem::chunk::<u8, 16>(&taps_bytes, 48));
    let round8 = _mm_set1_epi16(8);

    let mut y = 0;
    while y < height {
        let top_from_edge = y == 0;
        let edge_tl = edge.topleft - y;
        let mut x = 0;
        while x < width {
            let group = x / 4;
            let (p0, p1, p2, p3, p4) = if top_from_edge {
                if group == 0 {
                    (
                        edge.bytes[edge_tl],
                        edge.bytes[edge.topleft + 1],
                        edge.bytes[edge.topleft + 2],
                        edge.bytes[edge.topleft + 3],
                        edge.bytes[edge.topleft + 4],
                    )
                } else {
                    let col = edge.topleft + 4 * group;
                    (
                        edge.bytes[col],
                        edge.bytes[col + 1],
                        edge.bytes[col + 2],
                        edge.bytes[col + 3],
                        edge.bytes[col + 4],
                    )
                }
            } else {
                let top_row = (y - 1) * stride;
                let col = 4 * group;
                let p0 = if group == 0 {
                    edge.bytes[edge_tl]
                } else {
                    dst[top_row + col - 1]
                };
                (
                    p0,
                    dst[top_row + col],
                    dst[top_row + col + 1],
                    dst[top_row + col + 2],
                    dst[top_row + col + 3],
                )
            };
            let (p5, p6) = if group == 0 {
                (edge.bytes[edge_tl - 1], edge.bytes[edge_tl - 2])
            } else {
                (dst[y * stride + x - 1], dst[(y + 1) * stride + x - 1])
            };

            let v01 = _mm_set1_epi16(u16::from_le_bytes([p0, p1]) as i16);
            let v23 = _mm_set1_epi16(u16::from_le_bytes([p2, p3]) as i16);
            let v45 = _mm_set1_epi16(u16::from_le_bytes([p4, p5]) as i16);
            let v60 = _mm_set1_epi16(u16::from_le_bytes([p6, 0]) as i16);

            let acc0 = _mm_add_epi16(_mm_maddubs_epi16(v01, t0), _mm_maddubs_epi16(v23, t1));
            let acc1 = _mm_add_epi16(_mm_maddubs_epi16(v45, t2), _mm_maddubs_epi16(v60, t3));
            let sum = _mm_srai_epi16::<4>(_mm_add_epi16(_mm_add_epi16(acc0, acc1), round8));
            let packed = _mm_packus_epi16(sum, sum);
            let row1 = _mm_shuffle_epi32::<0x55>(packed);

            mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, y * stride + x), packed);
            mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, (y + 1) * stride + x), row1);
            x += 4;
        }
        y += 2;
    }
}

// ---------------------------------------------------------------------------
// 5. Z1 / Z2 / Z3 (`D01650`, `D01656`, `D01662`) — 128/256-bit AVX2
// ---------------------------------------------------------------------------

#[inline(always)]
fn clip_isize(value: isize, low: isize, high: isize) -> isize {
    value.max(low).min(high)
}

#[inline(always)]
fn read_edge8(edge: Edge<'_>, relative: isize) -> i32 {
    i32::from(edge.bytes[(edge.topleft as isize + relative) as usize])
}

#[inline(always)]
fn get_filter_strength(wh: usize, angle: usize, is_sm: bool) -> usize {
    if is_sm {
        if wh <= 8 {
            if angle >= 64 {
                return 2;
            }
            if angle >= 40 {
                return 1;
            }
        } else if wh <= 16 {
            if angle >= 48 {
                return 2;
            }
            if angle >= 20 {
                return 1;
            }
        } else if wh <= 24 {
            if angle >= 4 {
                return 3;
            }
        } else {
            return 3;
        }
    } else if wh <= 8 {
        if angle >= 56 {
            return 1;
        }
    } else if wh <= 16 {
        if angle >= 40 {
            return 1;
        }
    } else if wh <= 24 {
        if angle >= 32 {
            return 3;
        }
        if angle >= 16 {
            return 2;
        }
        if angle >= 8 {
            return 1;
        }
    } else if wh <= 32 {
        if angle >= 32 {
            return 3;
        }
        if angle >= 4 {
            return 2;
        }
        return 1;
    } else {
        return 3;
    }
    0
}

#[inline(always)]
fn get_upsample(wh: usize, angle: usize, is_sm: bool) -> bool {
    angle < 40 && wh <= (16 >> usize::from(is_sm))
}

fn filter_edge8(
    out: &mut [u8],
    sz: usize,
    lim_from: isize,
    lim_to: isize,
    edge: Edge<'_>,
    in_base: isize,
    from: isize,
    to: isize,
    strength: usize,
) {
    const KERNEL: [[i32; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
    let copy_end = if lim_from < 0 {
        0
    } else {
        sz.min(lim_from as usize)
    };
    let mut i = 0;
    while i < copy_end {
        out[i] = read_edge8(edge, in_base + clip_isize(i as isize, from, to - 1)) as u8;
        i += 1;
    }
    let filter_end = if lim_to < 0 {
        0
    } else {
        sz.min(lim_to as usize)
    };
    while i < filter_end {
        let mut sum = 0;
        for (j, weight) in KERNEL[strength - 1].iter().enumerate() {
            let position = clip_isize(i as isize - 2 + j as isize, from, to - 1);
            sum += read_edge8(edge, in_base + position) * weight;
        }
        out[i] = ((sum + 8) >> 4) as u8;
        i += 1;
    }
    while i < sz {
        out[i] = read_edge8(edge, in_base + clip_isize(i as isize, from, to - 1)) as u8;
        i += 1;
    }
}

fn upsample_edge8(
    out: &mut [u8],
    hsz: usize,
    edge: Edge<'_>,
    in_base: isize,
    from: isize,
    to: isize,
) {
    const KERNEL: [i32; 4] = [-1, 9, 9, -1];
    let mut i = 0;
    while i < hsz - 1 {
        out[i * 2] = read_edge8(edge, in_base + clip_isize(i as isize, from, to - 1)) as u8;
        let mut sum = 0;
        for (j, weight) in KERNEL.iter().enumerate() {
            let position = clip_isize(i as isize + j as isize - 1, from, to - 1);
            sum += read_edge8(edge, in_base + position) * weight;
        }
        out[i * 2 + 1] = ((sum + 8) >> 4).clamp(0, 255) as u8;
        i += 1;
    }
    out[(hsz - 1) * 2] =
        read_edge8(edge, in_base + clip_isize(hsz as isize - 1, from, to - 1)) as u8;
}

#[target_feature(enable = "avx2")]
#[inline]
fn transpose_store_4x4_u8(dst: &mut [u8], stride: usize, y: usize, x: usize, u0: __m128i) {
    mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, y * stride + x), u0);
    mem::store_u8x4(
        mem::chunk_mut::<u8, 4>(dst, (y + 1) * stride + x),
        _mm_shuffle_epi32::<0x55>(u0),
    );
    mem::store_u8x4(
        mem::chunk_mut::<u8, 4>(dst, (y + 2) * stride + x),
        _mm_shuffle_epi32::<0xAA>(u0),
    );
    mem::store_u8x4(
        mem::chunk_mut::<u8, 4>(dst, (y + 3) * stride + x),
        _mm_shuffle_epi32::<0xFF>(u0),
    );
}

#[target_feature(enable = "avx2")]
#[inline]
fn transpose_store_8x8_u8(dst: &mut [u8], stride: usize, y: usize, x: usize, c: &[__m128i; 8]) {
    let t0 = _mm_unpacklo_epi8(c[0], c[1]);
    let t1 = _mm_unpacklo_epi8(c[2], c[3]);
    let t2 = _mm_unpacklo_epi8(c[4], c[5]);
    let t3 = _mm_unpacklo_epi8(c[6], c[7]);
    let u0 = _mm_unpacklo_epi16(t0, t1);
    let u1 = _mm_unpackhi_epi16(t0, t1);
    let u2 = _mm_unpacklo_epi16(t2, t3);
    let u3 = _mm_unpackhi_epi16(t2, t3);
    let v0 = _mm_unpacklo_epi32(u0, u2);
    let v1 = _mm_unpackhi_epi32(u0, u2);
    let v2 = _mm_unpacklo_epi32(u1, u3);
    let v3 = _mm_unpackhi_epi32(u1, u3);
    mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, y * stride + x), v0);
    mem::store_u8x8(
        mem::chunk_mut::<u8, 8>(dst, (y + 1) * stride + x),
        _mm_unpackhi_epi64(v0, v0),
    );
    mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, (y + 2) * stride + x), v1);
    mem::store_u8x8(
        mem::chunk_mut::<u8, 8>(dst, (y + 3) * stride + x),
        _mm_unpackhi_epi64(v1, v1),
    );
    mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, (y + 4) * stride + x), v2);
    mem::store_u8x8(
        mem::chunk_mut::<u8, 8>(dst, (y + 5) * stride + x),
        _mm_unpackhi_epi64(v2, v2),
    );
    mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, (y + 6) * stride + x), v3);
    mem::store_u8x8(
        mem::chunk_mut::<u8, 8>(dst, (y + 7) * stride + x),
        _mm_unpackhi_epi64(v3, v3),
    );
}

#[target_feature(enable = "avx2")]
fn ipred_z1_avx2_giris(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
) {
    let is_sm = (angle >> 9) & 1 == 1;
    let enable = (angle >> 10) != 0;
    let base_angle = angle & 511;
    let mut dx = i32::from(DR_INTRA_DERIVATIVE[base_angle >> 1]);
    let mut buf = [0u8; 256];
    let upsample = enable && get_upsample(width + height, 90 - base_angle, is_sm);
    let (base_inc, max_base_x) = if upsample {
        upsample_edge8(
            &mut buf,
            width + height,
            edge,
            1,
            -1,
            (width + width.min(height)) as isize,
        );
        dx <<= 1;
        (2usize, 2 * (width + height) - 2)
    } else {
        let strength = if enable {
            get_filter_strength(width + height, 90 - base_angle, is_sm)
        } else {
            0
        };
        if strength > 0 {
            filter_edge8(
                &mut buf,
                width + height,
                0,
                (width + height) as isize,
                edge,
                1,
                -1,
                (width + width.min(height)) as isize,
                strength,
            );
            (1usize, width + height - 1)
        } else {
            let n = width + width.min(height);
            buf[..n].copy_from_slice(&edge.bytes[edge.topleft + 1..edge.topleft + 1 + n]);
            (1usize, n - 1)
        }
    };
    let fill = buf[max_base_x];
    buf[max_base_x + 1..max_base_x + 1 + 64].fill(fill);

    let shuf128 = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);
    let shuf256 = _mm256_setr_epi8(
        0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7,
        7, 8,
    );
    let mul512_128 = _mm_set1_epi16(512);
    let mul512_256 = _mm256_set1_epi16(512);

    let mut xpos = dx;
    for y in 0..height {
        let frac = (xpos & 0x3E) as i8;
        let w16 = u16::from_le_bytes([(64 - frac) as u8, frac as u8]) as i16;
        let wpair128 = _mm_set1_epi16(w16);
        let wpair256 = _mm256_set1_epi16(w16);
        let base = ((xpos >> 6) as usize).min(max_base_x);
        let row = y * stride;
        if base_inc == 1 {
            match width {
                4 => {
                    let s0 = mem::load_u8x16(mem::chunk::<u8, 16>(&buf, base));
                    let pairs = _mm_shuffle_epi8(s0, shuf128);
                    let prod = _mm_mulhrs_epi16(_mm_maddubs_epi16(pairs, wpair128), mul512_128);
                    mem::store_u8x4(
                        mem::chunk_mut::<u8, 4>(dst, row),
                        _mm_packus_epi16(prod, prod),
                    );
                }
                8 => {
                    let s0 = mem::load_u8x16(mem::chunk::<u8, 16>(&buf, base));
                    let pairs = _mm_shuffle_epi8(s0, shuf128);
                    let prod = _mm_mulhrs_epi16(_mm_maddubs_epi16(pairs, wpair128), mul512_128);
                    mem::store_u8x8(
                        mem::chunk_mut::<u8, 8>(dst, row),
                        _mm_packus_epi16(prod, prod),
                    );
                }
                16 => {
                    let s0 = _mm256_permute4x64_epi64::<0x94>(mem::load_u8x32(
                        mem::chunk::<u8, 32>(&buf, base),
                    ));
                    let pairs = _mm256_shuffle_epi8(s0, shuf256);
                    let prod =
                        _mm256_mulhrs_epi16(_mm256_maddubs_epi16(pairs, wpair256), mul512_256);
                    let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(prod, prod));
                    mem::store_u8x16(
                        mem::chunk_mut::<u8, 16>(dst, row),
                        _mm256_castsi256_si128(packed),
                    );
                }
                32 => {
                    let s0 = _mm256_permute4x64_epi64::<0x94>(mem::load_u8x32(
                        mem::chunk::<u8, 32>(&buf, base),
                    ));
                    let s1 = _mm256_permute4x64_epi64::<0x94>(mem::load_u8x32(
                        mem::chunk::<u8, 32>(&buf, base + 16),
                    ));
                    let p0 = _mm256_mulhrs_epi16(
                        _mm256_maddubs_epi16(_mm256_shuffle_epi8(s0, shuf256), wpair256),
                        mul512_256,
                    );
                    let p1 = _mm256_mulhrs_epi16(
                        _mm256_maddubs_epi16(_mm256_shuffle_epi8(s1, shuf256), wpair256),
                        mul512_256,
                    );
                    let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(p0, p1));
                    mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row), packed);
                }
                64 => {
                    for g in 0..2 {
                        let s0 =
                            _mm256_permute4x64_epi64::<0x94>(mem::load_u8x32(
                                mem::chunk::<u8, 32>(&buf, base + g * 32),
                            ));
                        let s1 =
                            _mm256_permute4x64_epi64::<0x94>(mem::load_u8x32(
                                mem::chunk::<u8, 32>(&buf, base + g * 32 + 16),
                            ));
                        let p0 = _mm256_mulhrs_epi16(
                            _mm256_maddubs_epi16(_mm256_shuffle_epi8(s0, shuf256), wpair256),
                            mul512_256,
                        );
                        let p1 = _mm256_mulhrs_epi16(
                            _mm256_maddubs_epi16(_mm256_shuffle_epi8(s1, shuf256), wpair256),
                            mul512_256,
                        );
                        let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(p0, p1));
                        mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row + g * 32), packed);
                    }
                }
                _ => unreachable!(),
            }
        } else {
            let p0 = mem::load_u8x16(mem::chunk::<u8, 16>(&buf, base));
            let r0 = _mm_mulhrs_epi16(_mm_maddubs_epi16(p0, wpair128), mul512_128);
            if width == 4 {
                mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, row), _mm_packus_epi16(r0, r0));
            } else {
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, row), _mm_packus_epi16(r0, r0));
            }
        }
        xpos += dx;
    }
}

#[target_feature(enable = "avx2")]
fn ipred_z2_avx2_giris(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
    max_width: usize,
    max_height: usize,
) {
    let is_sm = (angle >> 9) & 1 == 1;
    let enable = (angle >> 10) != 0;
    let base_angle = angle & 511;
    let mut dy = i32::from(DR_INTRA_DERIVATIVE[(base_angle - 90) >> 1]);
    let mut dx = i32::from(DR_INTRA_DERIVATIVE[(180 - base_angle) >> 1]);
    let upsample_left = enable && get_upsample(width + height, 180 - base_angle, is_sm);
    let upsample_above = enable && get_upsample(width + height, base_angle - 90, is_sm);
    let mut buf = [0u8; TMP_LEN + 64];
    let base = TMP_BASE;

    if upsample_above {
        upsample_edge8(
            &mut buf[TMP_BASE as usize..],
            width + 1,
            edge,
            0,
            0,
            (width + 1) as isize,
        );
        dx <<= 1;
    } else {
        let strength = if enable {
            get_filter_strength(width + height, base_angle - 90, is_sm)
        } else {
            0
        };
        if strength > 0 {
            filter_edge8(
                &mut buf[(TMP_BASE + 1) as usize..],
                width,
                0,
                max_width as isize,
                edge,
                1,
                -1,
                width as isize,
                strength,
            );
        } else {
            buf[(base + 1) as usize..(base + 1) as usize + width]
                .copy_from_slice(&edge.bytes[edge.topleft + 1..edge.topleft + 1 + width]);
        }
    }

    if upsample_left {
        upsample_edge8(
            &mut buf[(TMP_BASE - 2 * height as isize) as usize..],
            height + 1,
            edge,
            -(height as isize),
            0,
            (height + 1) as isize,
        );
        dy <<= 1;
    } else {
        let strength = if enable {
            get_filter_strength(width + height, 180 - base_angle, is_sm)
        } else {
            0
        };
        if strength > 0 {
            filter_edge8(
                &mut buf[(TMP_BASE - height as isize) as usize..],
                height,
                height as isize - max_height as isize,
                height as isize,
                edge,
                -(height as isize),
                0,
                (height + 1) as isize,
                strength,
            );
        } else {
            buf[(base - height as isize) as usize..base as usize]
                .copy_from_slice(&edge.bytes[edge.topleft - height..edge.topleft]);
        }
    }
    buf[base as usize] = edge.bytes[edge.topleft];

    let shuf_inc1 = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);
    let mul512 = _mm_set1_epi16(512);
    let base_inc_x = 1 + isize::from(upsample_above);
    let left = base - (1 + isize::from(upsample_left));
    let mut xpos = ((1 + isize::from(upsample_above)) << 6) - dx as isize;

    if width == 4 {
        for y in 0..height {
            let mut base_x = xpos >> 6;
            let frac_x = (xpos & 0x3E) as i32;
            let mut ypos = ((y as isize) << (6 + u32::from(upsample_left))) - dy as isize;
            let row = y * stride;
            let mut p_bytes = [0u8; 16];
            let mut w_bytes = [0u8; 16];
            for k in 0..4 {
                if base_x >= 0 {
                    let t_idx = (base + base_x) as usize;
                    p_bytes[2 * k] = buf[t_idx];
                    p_bytes[2 * k + 1] = buf[t_idx + 1];
                    w_bytes[2 * k] = (64 - frac_x) as u8;
                    w_bytes[2 * k + 1] = frac_x as u8;
                } else {
                    let by = ypos >> 6;
                    let fy = (ypos & 0x3E) as u8;
                    let l_idx = (left - by) as usize;
                    p_bytes[2 * k] = buf[l_idx];
                    p_bytes[2 * k + 1] = buf[l_idx - 1];
                    w_bytes[2 * k] = 64 - fy;
                    w_bytes[2 * k + 1] = fy;
                }
                base_x += base_inc_x;
                ypos -= dy as isize;
            }
            let pv = mem::load_u8x16(&p_bytes);
            let wv = mem::load_u8x16(&w_bytes);
            let prod = _mm_mulhrs_epi16(_mm_maddubs_epi16(pv, wv), mul512);
            mem::store_u8x4(
                mem::chunk_mut::<u8, 4>(dst, row),
                _mm_packus_epi16(prod, prod),
            );
            xpos -= dx as isize;
        }
        return;
    }

    let shuf256 = _mm256_setr_epi8(
        0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7,
        7, 8,
    );
    let mul512_256 = _mm256_set1_epi16(512);
    let base_inc_y = 1usize << u32::from(upsample_left);

    let mut left_idx0 = [0u8; 64];
    let mut left_w_bytes = [0u8; 128];
    for col in 0..width {
        let yp0 = -((col as isize + 1) * dy as isize);
        let by0 = yp0 >> 6;
        let fy = (yp0 & 0x3E) as u8;
        left_idx0[col] = (left - by0) as u8;
        left_w_bytes[2 * col] = 64 - fy;
        left_w_bytes[2 * col + 1] = fy;
    }

    for y in 0..height {
        let mut base_x = xpos >> 6;
        let frac_x = (xpos & 0x3E) as u8;
        let w16_x = u16::from_le_bytes([64 - frac_x, frac_x]) as i16;
        let wpair_x = _mm_set1_epi16(w16_x);
        let wpair_x256 = _mm256_set1_epi16(w16_x);
        let y_off = y * base_inc_y;
        let row = y * stride;

        let mut x = 0;
        while x + 8 <= width {
            if base_x >= 0 {
                if base_inc_x == 1 && x + 16 <= width {
                    let idx = (base + base_x) as usize;
                    let s0 = _mm256_permute4x64_epi64::<0x94>(mem::load_u8x32(
                        mem::chunk::<u8, 32>(&buf, idx),
                    ));
                    let pairs = _mm256_shuffle_epi8(s0, shuf256);
                    let prod =
                        _mm256_mulhrs_epi16(_mm256_maddubs_epi16(pairs, wpair_x256), mul512_256);
                    let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(prod, prod));
                    mem::store_u8x16(
                        mem::chunk_mut::<u8, 16>(dst, row + x),
                        _mm256_castsi256_si128(packed),
                    );
                    base_x += 16;
                    x += 16;
                } else {
                    let idx = (base + base_x) as usize;
                    let pairs = if base_inc_x == 1 {
                        _mm_shuffle_epi8(
                            mem::load_u8x16(mem::chunk::<u8, 16>(&buf, idx)),
                            shuf_inc1,
                        )
                    } else {
                        mem::load_u8x16(mem::chunk::<u8, 16>(&buf, idx))
                    };
                    let prod = _mm_mulhrs_epi16(_mm_maddubs_epi16(pairs, wpair_x), mul512);
                    mem::store_u8x8(
                        mem::chunk_mut::<u8, 8>(dst, row + x),
                        _mm_packus_epi16(prod, prod),
                    );
                    base_x += 8 * base_inc_x;
                    x += 8;
                }
            } else if base_x + 7 * base_inc_x < 0 {
                let mut p_bytes = [0u8; 16];
                for k in 0..8 {
                    let l_idx = usize::from(left_idx0[x + k]) - y_off;
                    p_bytes[2 * k] = buf[l_idx];
                    p_bytes[2 * k + 1] = buf[l_idx - 1];
                }
                let pv = mem::load_u8x16(&p_bytes);
                let wv = mem::load_u8x16(mem::chunk::<u8, 16>(&left_w_bytes, 2 * x));
                let prod = _mm_mulhrs_epi16(_mm_maddubs_epi16(pv, wv), mul512);
                mem::store_u8x8(
                    mem::chunk_mut::<u8, 8>(dst, row + x),
                    _mm_packus_epi16(prod, prod),
                );
                base_x += 8 * base_inc_x;
                x += 8;
            } else {
                let mut p_bytes = [0u8; 16];
                let mut w_bytes = [0u8; 16];
                for k in 0..8 {
                    if base_x >= 0 {
                        let t_idx = (base + base_x) as usize;
                        p_bytes[2 * k] = buf[t_idx];
                        p_bytes[2 * k + 1] = buf[t_idx + 1];
                        w_bytes[2 * k] = 64 - frac_x;
                        w_bytes[2 * k + 1] = frac_x;
                    } else {
                        let l_idx = usize::from(left_idx0[x + k]) - y_off;
                        p_bytes[2 * k] = buf[l_idx];
                        p_bytes[2 * k + 1] = buf[l_idx - 1];
                        w_bytes[2 * k] = left_w_bytes[2 * (x + k)];
                        w_bytes[2 * k + 1] = left_w_bytes[2 * (x + k) + 1];
                    }
                    base_x += base_inc_x;
                }
                let pv = mem::load_u8x16(&p_bytes);
                let wv = mem::load_u8x16(&w_bytes);
                let prod = _mm_mulhrs_epi16(_mm_maddubs_epi16(pv, wv), mul512);
                mem::store_u8x8(
                    mem::chunk_mut::<u8, 8>(dst, row + x),
                    _mm_packus_epi16(prod, prod),
                );
                x += 8;
            }
        }
        xpos -= dx as isize;
    }
}

#[target_feature(enable = "avx2")]
fn ipred_z3_avx2_giris(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
) {
    let is_sm = (angle >> 9) & 1 == 1;
    let enable = (angle >> 10) != 0;
    let base_angle = angle & 511;
    let mut dy = i32::from(DR_INTRA_DERIVATIVE[(270 - base_angle) >> 1]);
    let upsample = enable && get_upsample(width + height, base_angle - 180, is_sm);
    let strength = if enable && !upsample {
        get_filter_strength(width + height, base_angle - 180, is_sm)
    } else {
        0
    };
    let direct = !upsample && strength == 0;

    let mut rev_buf = [0u8; 256];
    let (base_inc, max_base_y) = if upsample {
        let mut left_out = [0u8; 40];
        upsample_edge8(
            &mut left_out,
            width + height,
            edge,
            -((width + height) as isize),
            width.saturating_sub(height) as isize,
            (width + height + 1) as isize,
        );
        let lb = 2 * (width + height) - 2;
        for b in 0..=lb {
            rev_buf[b] = left_out[lb - b];
        }
        dy <<= 1;
        (2usize, lb)
    } else if !direct {
        let mut left_out = [0u8; 140];
        filter_edge8(
            &mut left_out,
            width + height,
            0,
            (width + height) as isize,
            edge,
            -((width + height) as isize),
            width.saturating_sub(height) as isize,
            (width + height + 1) as isize,
            strength,
        );
        let lb = width + height - 1;
        for b in 0..=lb {
            rev_buf[b] = left_out[lb - b];
        }
        (1usize, lb)
    } else {
        let mb = height + width.min(height) - 1;
        for b in 0..=mb {
            rev_buf[b] = edge.bytes[edge.topleft - 1 - b];
        }
        (1usize, mb)
    };
    let fill = rev_buf[max_base_y];
    let pad_len = height * base_inc + 16;
    rev_buf[max_base_y + 1..max_base_y + 1 + pad_len].fill(fill);

    let shuf_inc1 = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);
    let shuf256 = _mm256_setr_epi8(
        0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7,
        7, 8,
    );
    let mul512 = _mm_set1_epi16(512);
    let mul512_256 = _mm256_set1_epi16(512);
    let step_dy = dy as usize;

    if width == 4 {
        let mut bases = [0usize; 4];
        let mut wpairs = [_mm_setzero_si128(); 4];
        for k in 0..4 {
            let yp = (k + 1) * step_dy;
            let frac = (yp & 0x3E) as u8;
            bases[k] = (yp >> 6).min(max_base_y);
            wpairs[k] = _mm_set1_epi16(u16::from_le_bytes([64 - frac, frac]) as i16);
        }
        if base_inc == 1 {
            if height == 4 {
                let mut c = [_mm_setzero_si128(); 4];
                for k in 0..4 {
                    let p0 = _mm_shuffle_epi8(
                        mem::load_u8x16(mem::chunk::<u8, 16>(&rev_buf, bases[k])),
                        shuf_inc1,
                    );
                    let r0 = _mm_mulhrs_epi16(_mm_maddubs_epi16(p0, wpairs[k]), mul512);
                    c[k] = _mm_packus_epi16(r0, r0);
                }
                let t0 = _mm_unpacklo_epi8(c[0], c[1]);
                let t1 = _mm_unpacklo_epi8(c[2], c[3]);
                let u0 = _mm_unpacklo_epi16(t0, t1);
                transpose_store_4x4_u8(dst, stride, 0, 0, u0);
            } else {
                for y in (0..height).step_by(8) {
                    let mut c = [_mm_setzero_si128(); 4];
                    for k in 0..4 {
                        let p0 = _mm_shuffle_epi8(
                            mem::load_u8x16(mem::chunk::<u8, 16>(&rev_buf, bases[k] + y)),
                            shuf_inc1,
                        );
                        let r0 = _mm_mulhrs_epi16(_mm_maddubs_epi16(p0, wpairs[k]), mul512);
                        c[k] = _mm_packus_epi16(r0, r0);
                    }
                    let t0 = _mm_unpacklo_epi8(c[0], c[1]);
                    let t1 = _mm_unpacklo_epi8(c[2], c[3]);
                    transpose_store_4x4_u8(dst, stride, y, 0, _mm_unpacklo_epi16(t0, t1));
                    transpose_store_4x4_u8(dst, stride, y + 4, 0, _mm_unpackhi_epi16(t0, t1));
                }
            }
        } else {
            let mut c = [_mm_setzero_si128(); 4];
            for k in 0..4 {
                let p0 = mem::load_u8x16(mem::chunk::<u8, 16>(&rev_buf, bases[k]));
                let r0 = _mm_mulhrs_epi16(_mm_maddubs_epi16(p0, wpairs[k]), mul512);
                c[k] = _mm_packus_epi16(r0, r0);
            }
            let t0 = _mm_unpacklo_epi8(c[0], c[1]);
            let t1 = _mm_unpacklo_epi8(c[2], c[3]);
            transpose_store_4x4_u8(dst, stride, 0, 0, _mm_unpacklo_epi16(t0, t1));
            if height == 8 {
                transpose_store_4x4_u8(dst, stride, 4, 0, _mm_unpackhi_epi16(t0, t1));
            }
        }
        return;
    }

    for x in (0..width).step_by(8) {
        let mut bases = [0usize; 8];
        let mut wpairs128 = [_mm_setzero_si128(); 8];
        let mut wpairs256 = [_mm256_setzero_si256(); 8];
        for k in 0..8 {
            let yp = (x + k + 1) * step_dy;
            let frac = (yp & 0x3E) as u8;
            let w16 = u16::from_le_bytes([64 - frac, frac]) as i16;
            bases[k] = (yp >> 6).min(max_base_y);
            wpairs128[k] = _mm_set1_epi16(w16);
            wpairs256[k] = _mm256_set1_epi16(w16);
        }
        if base_inc == 1 {
            if height == 4 {
                let mut c = [_mm_setzero_si128(); 8];
                for k in 0..8 {
                    let p0 = _mm_shuffle_epi8(
                        mem::load_u8x16(mem::chunk::<u8, 16>(&rev_buf, bases[k])),
                        shuf_inc1,
                    );
                    let r0 = _mm_mulhrs_epi16(_mm_maddubs_epi16(p0, wpairs128[k]), mul512);
                    c[k] = _mm_packus_epi16(r0, r0);
                }
                let t0 = _mm_unpacklo_epi8(c[0], c[1]);
                let t1 = _mm_unpacklo_epi8(c[2], c[3]);
                let t2 = _mm_unpacklo_epi8(c[4], c[5]);
                let t3 = _mm_unpacklo_epi8(c[6], c[7]);
                let u0 = _mm_unpacklo_epi16(t0, t1);
                let u2 = _mm_unpacklo_epi16(t2, t3);
                let v0 = _mm_unpacklo_epi32(u0, u2);
                let v1 = _mm_unpackhi_epi32(u0, u2);
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, x), v0);
                mem::store_u8x8(
                    mem::chunk_mut::<u8, 8>(dst, stride + x),
                    _mm_unpackhi_epi64(v0, v0),
                );
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, 2 * stride + x), v1);
                mem::store_u8x8(
                    mem::chunk_mut::<u8, 8>(dst, 3 * stride + x),
                    _mm_unpackhi_epi64(v1, v1),
                );
            } else if height == 8 {
                let mut c = [_mm_setzero_si128(); 8];
                for k in 0..8 {
                    let p0 = _mm_shuffle_epi8(
                        mem::load_u8x16(mem::chunk::<u8, 16>(&rev_buf, bases[k])),
                        shuf_inc1,
                    );
                    let r0 = _mm_mulhrs_epi16(_mm_maddubs_epi16(p0, wpairs128[k]), mul512);
                    c[k] = _mm_packus_epi16(r0, r0);
                }
                transpose_store_8x8_u8(dst, stride, 0, x, &c);
            } else {
                for y in (0..height).step_by(16) {
                    let mut c_lo = [_mm_setzero_si128(); 8];
                    let mut c_hi = [_mm_setzero_si128(); 8];
                    for k in 0..8 {
                        let s0 =
                            _mm256_permute4x64_epi64::<0x94>(mem::load_u8x32(
                                mem::chunk::<u8, 32>(&rev_buf, bases[k] + y),
                            ));
                        let p0 = _mm256_shuffle_epi8(s0, shuf256);
                        let r0 =
                            _mm256_mulhrs_epi16(_mm256_maddubs_epi16(p0, wpairs256[k]), mul512_256);
                        let packed = _mm256_packus_epi16(r0, r0);
                        c_lo[k] = _mm256_castsi256_si128(packed);
                        c_hi[k] = _mm256_extracti128_si256::<1>(packed);
                    }
                    transpose_store_8x8_u8(dst, stride, y, x, &c_lo);
                    transpose_store_8x8_u8(dst, stride, y + 8, x, &c_hi);
                }
            }
        } else {
            let mut c = [_mm_setzero_si128(); 8];
            for k in 0..8 {
                let p0 = mem::load_u8x16(mem::chunk::<u8, 16>(&rev_buf, bases[k]));
                let r0 = _mm_mulhrs_epi16(_mm_maddubs_epi16(p0, wpairs128[k]), mul512);
                c[k] = _mm_packus_epi16(r0, r0);
            }
            if height == 4 {
                let t0 = _mm_unpacklo_epi8(c[0], c[1]);
                let t1 = _mm_unpacklo_epi8(c[2], c[3]);
                let t2 = _mm_unpacklo_epi8(c[4], c[5]);
                let t3 = _mm_unpacklo_epi8(c[6], c[7]);
                let u0 = _mm_unpacklo_epi16(t0, t1);
                let u2 = _mm_unpacklo_epi16(t2, t3);
                let v0 = _mm_unpacklo_epi32(u0, u2);
                let v1 = _mm_unpackhi_epi32(u0, u2);
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, x), v0);
                mem::store_u8x8(
                    mem::chunk_mut::<u8, 8>(dst, stride + x),
                    _mm_unpackhi_epi64(v0, v0),
                );
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, 2 * stride + x), v1);
                mem::store_u8x8(
                    mem::chunk_mut::<u8, 8>(dst, 3 * stride + x),
                    _mm_unpackhi_epi64(v1, v1),
                );
            } else {
                transpose_store_8x8_u8(dst, stride, 0, x, &c);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 6. CFL_AC (`D01590`, `D01594`, `D01598`) — 128/256-bit AVX2
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
fn cfl_ac_dc_subtract_avx2(ac: &mut [i16], cw: usize, ch: usize) {
    let total = cw * ch;
    let log2sz = cw.trailing_zeros() + ch.trailing_zeros();
    let ones256 = _mm256_set1_epi16(1);
    let mut sum_v = _mm256_setzero_si256();
    let mut i = 0;
    while i < total {
        let v = mem::load_i16x16(mem::chunk::<i16, 16>(ac, i));
        sum_v = _mm256_add_epi32(sum_v, _mm256_madd_epi16(v, ones256));
        i += 16;
    }
    let lo = _mm256_castsi256_si128(sum_v);
    let hi = _mm256_extracti128_si256::<1>(sum_v);
    let mut sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_add_epi32(sum128, _mm_unpackhi_epi64(sum128, _mm_setzero_si128()));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32::<1>(sum128));
    let mut sum = _mm_cvtsi128_si32(sum128) + ((1i32 << log2sz) >> 1);
    sum >>= log2sz;

    let avg_v = _mm256_set1_epi16(sum as i16);
    let mut j = 0;
    while j < total {
        let v = mem::load_i16x16(mem::chunk::<i16, 16>(ac, j));
        mem::store_i16x16(mem::chunk_mut::<i16, 16>(ac, j), _mm256_sub_epi16(v, avg_v));
        j += 16;
    }
}

#[target_feature(enable = "avx2")]
fn cfl_ac_core_avx2<const SS_HOR: usize, const SS_VER: usize>(
    ac: &mut [i16],
    y_src: &[u8],
    y_stride: usize,
    w_pad: usize,
    h_pad: usize,
    cw: usize,
    ch: usize,
) {
    let visible_w = cw - 4 * w_pad;
    let visible_h = ch - 4 * h_pad;
    let zero128 = _mm_setzero_si128();
    let madd_w128 = if SS_VER == 1 {
        _mm_set1_epi8(2)
    } else {
        _mm_set1_epi8(4)
    };
    let madd_w256 = if SS_VER == 1 {
        _mm256_set1_epi8(2)
    } else {
        _mm256_set1_epi8(4)
    };

    for row in 0..visible_h {
        let luma_row = row << SS_VER;
        let r0_off = luma_row * y_stride;
        let r1_off = (luma_row + 1) * y_stride;
        let ac_row = row * cw;

        let mut x = 0;
        while x + 16 <= visible_w {
            if SS_HOR == 1 {
                let col = x * 2;
                let v0 = mem::load_u8x32(mem::chunk::<u8, 32>(y_src, r0_off + col));
                let s0 = _mm256_maddubs_epi16(v0, madd_w256);
                let res = if SS_VER == 1 {
                    let v1 = mem::load_u8x32(mem::chunk::<u8, 32>(y_src, r1_off + col));
                    _mm256_add_epi16(s0, _mm256_maddubs_epi16(v1, madd_w256))
                } else {
                    s0
                };
                mem::store_i16x16(mem::chunk_mut::<i16, 16>(ac, ac_row + x), res);
            } else {
                let v16 =
                    _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(y_src, r0_off + x)));
                let res = _mm256_slli_epi16::<3>(v16);
                mem::store_i16x16(mem::chunk_mut::<i16, 16>(ac, ac_row + x), res);
            }
            x += 16;
        }
        while x + 8 <= visible_w {
            if SS_HOR == 1 {
                let col = x * 2;
                let v0 = mem::load_u8x16(mem::chunk::<u8, 16>(y_src, r0_off + col));
                let s0 = _mm_maddubs_epi16(v0, madd_w128);
                let res = if SS_VER == 1 {
                    let v1 = mem::load_u8x16(mem::chunk::<u8, 16>(y_src, r1_off + col));
                    _mm_add_epi16(s0, _mm_maddubs_epi16(v1, madd_w128))
                } else {
                    s0
                };
                mem::store_i16x8(mem::chunk_mut::<i16, 8>(ac, ac_row + x), res);
            } else {
                let raw8 = u64::from_le_bytes(*mem::chunk::<u8, 8>(y_src, r0_off + x));
                let v16 = _mm_unpacklo_epi8(_mm_cvtsi64_si128(raw8 as i64), zero128);
                let res = _mm_slli_epi16::<3>(v16);
                mem::store_i16x8(mem::chunk_mut::<i16, 8>(ac, ac_row + x), res);
            }
            x += 8;
        }
        if x + 4 <= visible_w {
            if SS_HOR == 1 {
                let col = x * 2;
                let raw0 = u64::from_le_bytes(*mem::chunk::<u8, 8>(y_src, r0_off + col));
                let s0 = _mm_maddubs_epi16(_mm_cvtsi64_si128(raw0 as i64), madd_w128);
                let res = if SS_VER == 1 {
                    let raw1 = u64::from_le_bytes(*mem::chunk::<u8, 8>(y_src, r1_off + col));
                    _mm_add_epi16(
                        s0,
                        _mm_maddubs_epi16(_mm_cvtsi64_si128(raw1 as i64), madd_w128),
                    )
                } else {
                    s0
                };
                let mut tmp = [0i16; 8];
                mem::store_i16x8(&mut tmp, res);
                ac[ac_row + x..ac_row + x + 4].copy_from_slice(&tmp[..4]);
            } else {
                let raw4 = u32::from_le_bytes(*mem::chunk::<u8, 4>(y_src, r0_off + x));
                let v16 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(raw4 as i32), zero128);
                let res = _mm_slli_epi16::<3>(v16);
                let mut tmp = [0i16; 8];
                mem::store_i16x8(&mut tmp, res);
                ac[ac_row + x..ac_row + x + 4].copy_from_slice(&tmp[..4]);
            }
            x += 4;
        }
        if x < cw {
            let fill = ac[ac_row + x - 1];
            while x < cw {
                ac[ac_row + x] = fill;
                x += 1;
            }
        }
    }

    for row in visible_h..ch {
        let src_row = (visible_h - 1) * cw;
        let dst_row = row * cw;
        for col in 0..cw {
            ac[dst_row + col] = ac[src_row + col];
        }
    }

    cfl_ac_dc_subtract_avx2(ac, cw, ch);
}

// ---------------------------------------------------------------------------
// 7. CFL_PRED (`D01602`, `D01606`, `D01610`, `D01614`) — 128/256-bit AVX2
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
fn cfl_pred_vec8_avx2(
    v_ac: __m128i,
    v_alpha: __m128i,
    v_abs_alpha_shl9: __m128i,
    v_dc: __m128i,
) -> __m128i {
    let sign = _mm_sign_epi16(v_ac, v_alpha);
    let abs_ac = _mm_abs_epi16(v_ac);
    let scaled = _mm_mulhrs_epi16(abs_ac, v_abs_alpha_shl9);
    let signed = _mm_sign_epi16(scaled, sign);
    _mm_add_epi16(signed, v_dc)
}

#[target_feature(enable = "avx2")]
#[inline]
fn cfl_pred_vec16_avx2(
    v_ac: __m256i,
    v_alpha: __m256i,
    v_abs_alpha_shl9: __m256i,
    v_dc: __m256i,
) -> __m256i {
    let sign = _mm256_sign_epi16(v_ac, v_alpha);
    let abs_ac = _mm256_abs_epi16(v_ac);
    let scaled = _mm256_mulhrs_epi16(abs_ac, v_abs_alpha_shl9);
    let signed = _mm256_sign_epi16(scaled, sign);
    _mm256_add_epi16(signed, v_dc)
}

#[target_feature(enable = "avx2")]
fn cfl_pred_core_avx2(
    mode: CflPredMode,
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    ac: &[i16],
    alpha: i32,
) {
    let dc = match mode {
        CflPredMode::Dc => dc_gen_avx2(edge, width, height) as i16,
        CflPredMode::LeftDc => dc_gen_left_avx2(edge, height) as i16,
        CflPredMode::TopDc => dc_gen_top_avx2(edge, width) as i16,
        CflPredMode::Dc128 => 128,
    };
    if alpha == 0 {
        splat_dc_avx2(dst, dst_stride, width, height, dc as i8);
        return;
    }

    let v_dc128 = _mm_set1_epi16(dc);
    let v_alpha128 = _mm_set1_epi16(alpha as i16);
    let v_abs128 = _mm_slli_epi16::<9>(_mm_abs_epi16(v_alpha128));

    let v_dc256 = _mm256_set1_epi16(dc);
    let v_alpha256 = _mm256_set1_epi16(alpha as i16);
    let v_abs256 = _mm256_slli_epi16::<9>(_mm256_abs_epi16(v_alpha256));

    match width {
        4 => {
            let mut y = 0;
            while y < height {
                let v_ac = mem::load_i16x8(mem::chunk::<i16, 8>(ac, y * 4));
                let res = cfl_pred_vec8_avx2(v_ac, v_alpha128, v_abs128, v_dc128);
                let packed = _mm_packus_epi16(res, res);
                let row1 = _mm_shuffle_epi32::<0x55>(packed);
                mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, y * dst_stride), packed);
                mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, (y + 1) * dst_stride), row1);
                y += 2;
            }
        }
        8 => {
            let mut y = 0;
            while y < height {
                let v_ac = mem::load_i16x16(mem::chunk::<i16, 16>(ac, y * 8));
                let res = cfl_pred_vec16_avx2(v_ac, v_alpha256, v_abs256, v_dc256);
                let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(res, res));
                let p128 = _mm256_castsi256_si128(packed);
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, y * dst_stride), p128);
                mem::store_u8x8(
                    mem::chunk_mut::<u8, 8>(dst, (y + 1) * dst_stride),
                    _mm_unpackhi_epi64(p128, p128),
                );
                y += 2;
            }
        }
        16 => {
            for y in 0..height {
                let v_ac = mem::load_i16x16(mem::chunk::<i16, 16>(ac, y * 16));
                let res = cfl_pred_vec16_avx2(v_ac, v_alpha256, v_abs256, v_dc256);
                let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(res, res));
                mem::store_u8x16(
                    mem::chunk_mut::<u8, 16>(dst, y * dst_stride),
                    _mm256_castsi256_si128(packed),
                );
            }
        }
        32 => {
            for y in 0..height {
                let ac_off = y * 32;
                let r0 = cfl_pred_vec16_avx2(
                    mem::load_i16x16(mem::chunk::<i16, 16>(ac, ac_off)),
                    v_alpha256,
                    v_abs256,
                    v_dc256,
                );
                let r1 = cfl_pred_vec16_avx2(
                    mem::load_i16x16(mem::chunk::<i16, 16>(ac, ac_off + 16)),
                    v_alpha256,
                    v_abs256,
                    v_dc256,
                );
                let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(r0, r1));
                mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, y * dst_stride), packed);
            }
        }
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// 8. PAL_PRED (`D01696`) — 128/256-bit `vpshufb`
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
fn pal_pred_avx2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
) {
    let pal64 = u64::from_le_bytes(*mem::chunk::<u8, 8>(pal, 0));
    let v_pal128 = _mm_cvtsi64_si128(pal64 as i64);
    let v_pal256 = _mm256_set1_epi64x(pal64 as i64);
    let mask_lo128 = _mm_set1_epi8(0x07);
    let mask_lo256 = _mm256_set1_epi8(0x07);

    match w {
        4 => {
            let mut y = 0;
            while y < h {
                let raw8 = u64::from_le_bytes(*mem::chunk::<u8, 8>(idx, y * 2));
                let v_idx = _mm_cvtsi64_si128(raw8 as i64);
                let lo = _mm_and_si128(v_idx, mask_lo128);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(v_idx), mask_lo128);
                let even = _mm_shuffle_epi8(v_pal128, lo);
                let odd = _mm_shuffle_epi8(v_pal128, hi);
                let out16 = _mm_unpacklo_epi8(even, odd);
                mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, y * dst_stride), out16);
                mem::store_u8x4(
                    mem::chunk_mut::<u8, 4>(dst, (y + 1) * dst_stride),
                    _mm_shuffle_epi32::<0x55>(out16),
                );
                mem::store_u8x4(
                    mem::chunk_mut::<u8, 4>(dst, (y + 2) * dst_stride),
                    _mm_shuffle_epi32::<0xAA>(out16),
                );
                mem::store_u8x4(
                    mem::chunk_mut::<u8, 4>(dst, (y + 3) * dst_stride),
                    _mm_shuffle_epi32::<0xFF>(out16),
                );
                y += 4;
            }
        }
        8 => {
            let mut y = 0;
            while y < h {
                let raw8 = u64::from_le_bytes(*mem::chunk::<u8, 8>(idx, y * 4));
                let v_idx = _mm_cvtsi64_si128(raw8 as i64);
                let lo = _mm_and_si128(v_idx, mask_lo128);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(v_idx), mask_lo128);
                let even = _mm_shuffle_epi8(v_pal128, lo);
                let odd = _mm_shuffle_epi8(v_pal128, hi);
                let out16 = _mm_unpacklo_epi8(even, odd);
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, y * dst_stride), out16);
                mem::store_u8x8(
                    mem::chunk_mut::<u8, 8>(dst, (y + 1) * dst_stride),
                    _mm_unpackhi_epi64(out16, out16),
                );
                y += 2;
            }
        }
        16 => {
            let mut y = 0;
            while y < h {
                // 2 satır = 16 idx baytı = 32 piksel
                let v_idx128 = mem::load_u8x16(mem::chunk::<u8, 16>(idx, y * 8));
                let raw0 = u64::from_le_bytes(*mem::chunk::<u8, 8>(idx, y * 8));
                let raw1 = u64::from_le_bytes(*mem::chunk::<u8, 8>(idx, y * 8 + 8));
                let _ = v_idx128;
                for (dy, raw) in [(0, raw0), (1, raw1)] {
                    let v_idx = _mm_cvtsi64_si128(raw as i64);
                    let lo = _mm_and_si128(v_idx, mask_lo128);
                    let hi = _mm_and_si128(_mm_srli_epi16::<4>(v_idx), mask_lo128);
                    let even = _mm_shuffle_epi8(v_pal128, lo);
                    let odd = _mm_shuffle_epi8(v_pal128, hi);
                    mem::store_u8x16(
                        mem::chunk_mut::<u8, 16>(dst, (y + dy) * dst_stride),
                        _mm_unpacklo_epi8(even, odd),
                    );
                }
                y += 2;
            }
        }
        32 => {
            let mut y = 0;
            while y < h {
                // 2 satır = 32 idx baytı = 64 piksel (256-bit `vpshufb`)
                let v_idx = mem::load_u8x32(mem::chunk::<u8, 32>(idx, y * 16));
                let lo = _mm256_and_si256(v_idx, mask_lo256);
                let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v_idx), mask_lo256);
                let even = _mm256_permute4x64_epi64::<0xD8>(_mm256_shuffle_epi8(v_pal256, lo));
                let odd = _mm256_permute4x64_epi64::<0xD8>(_mm256_shuffle_epi8(v_pal256, hi));
                let row0 = _mm256_unpacklo_epi8(even, odd);
                let row1 = _mm256_unpackhi_epi8(even, odd);
                mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, y * dst_stride), row0);
                mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, (y + 1) * dst_stride), row1);
                y += 2;
            }
        }
        64 => {
            for y in 0..h {
                let v_idx = mem::load_u8x32(mem::chunk::<u8, 32>(idx, y * 32));
                let lo = _mm256_and_si256(v_idx, mask_lo256);
                let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v_idx), mask_lo256);
                let even = _mm256_permute4x64_epi64::<0xD8>(_mm256_shuffle_epi8(v_pal256, lo));
                let odd = _mm256_permute4x64_epi64::<0xD8>(_mm256_shuffle_epi8(v_pal256, hi));
                let p0 = _mm256_unpacklo_epi8(even, odd);
                let p1 = _mm256_unpackhi_epi8(even, odd);
                let row = y * dst_stride;
                mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row), p0);
                mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row + 32), p1);
            }
        }
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Güvenli giriş sarmalayıcıları (O(1) sözleşme + tek `unsafe` blok)
// ---------------------------------------------------------------------------

macro_rules! avx2_angular_entry {
    ($fn_name:ident, $mode:expr, $call:expr) => {
        fn $fn_name(
            dst: &mut [u8],
            dst_stride: usize,
            edge: Edge<'_>,
            width: usize,
            height: usize,
            angle: usize,
            max_width: usize,
            max_height: usize,
            bit_depth: BitDepth,
            bitdepth_max: u16,
        ) {
            if !angular_sozlesmesi::<8>(
                $mode,
                dst.len(),
                dst_stride,
                edge,
                width,
                height,
                angle,
                bit_depth,
                bitdepth_max,
            ) {
                angular_skaler_devret::<8>(
                    $mode,
                    dst,
                    dst_stride,
                    edge,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bit_depth,
                    bitdepth_max,
                );
                return;
            }
            let call_fn = $call;
            // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
            // (`intra_pred_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır.
            // (2) sınır: `angular_sozlesmesi::<8>` blok boyutunu, `edge` okuma aralığını
            // ve `dst` yazma bölgesini O(1) doğruladı. (3) hizalama/takma ad: tüm SIMD
            // yükleme/saklamalar `mem` içindeki hizasız sabit boyutlu dizi yardımcılarıyla
            // yapılır; `dst` `&mut [u8]` ve `edge.bytes` `&[u8]` ayrık referanslardır.
            unsafe {
                call_fn(
                    dst, dst_stride, edge, width, height, angle, max_width, max_height,
                )
            }
        }
    };
}

#[target_feature(enable = "avx2")]
fn call_dc(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_dc_avx2_giris(dst, s, e, w, h);
}
#[target_feature(enable = "avx2")]
fn call_vert(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_vert_avx2_giris(dst, s, e, w, h);
}
#[target_feature(enable = "avx2")]
fn call_hor(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_hor_avx2_giris(dst, s, e, w, h);
}
#[target_feature(enable = "avx2")]
fn call_left_dc(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_left_dc_avx2_giris(dst, s, e, w, h);
}
#[target_feature(enable = "avx2")]
fn call_top_dc(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_top_dc_avx2_giris(dst, s, e, w, h);
}
#[target_feature(enable = "avx2")]
fn call_dc_128(
    dst: &mut [u8],
    s: usize,
    _e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_dc_128_avx2_giris(dst, s, w, h);
}
#[target_feature(enable = "avx2")]
fn call_z1(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_z1_avx2_giris(dst, s, e, w, h, a);
}
#[target_feature(enable = "avx2")]
fn call_z2(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    a: usize,
    mw: usize,
    mh: usize,
) {
    ipred_z2_avx2_giris(dst, s, e, w, h, a, mw, mh);
}
#[target_feature(enable = "avx2")]
fn call_z3(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_z3_avx2_giris(dst, s, e, w, h, a);
}
#[target_feature(enable = "avx2")]
fn call_smooth(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_smooth_avx2_giris(dst, s, e, w, h);
}
#[target_feature(enable = "avx2")]
fn call_smooth_v(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_smooth_v_avx2_giris(dst, s, e, w, h);
}
#[target_feature(enable = "avx2")]
fn call_smooth_h(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_smooth_h_avx2_giris(dst, s, e, w, h);
}
#[target_feature(enable = "avx2")]
fn call_paeth(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_paeth_avx2_giris(dst, s, e, w, h);
}
#[target_feature(enable = "avx2")]
fn call_filter(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_filter_avx2_giris(dst, s, e, w, h, a);
}

avx2_angular_entry!(ipred_dc_avx2, IntraPredMode::Dc, call_dc);
avx2_angular_entry!(ipred_vert_avx2, IntraPredMode::Vert, call_vert);
avx2_angular_entry!(ipred_hor_avx2, IntraPredMode::Hor, call_hor);
avx2_angular_entry!(ipred_left_dc_avx2, IntraPredMode::LeftDc, call_left_dc);
avx2_angular_entry!(ipred_top_dc_avx2, IntraPredMode::TopDc, call_top_dc);
avx2_angular_entry!(ipred_dc_128_avx2, IntraPredMode::Dc128, call_dc_128);
avx2_angular_entry!(ipred_z1_avx2, IntraPredMode::Z1, call_z1);
avx2_angular_entry!(ipred_z2_avx2, IntraPredMode::Z2, call_z2);
avx2_angular_entry!(ipred_z3_avx2, IntraPredMode::Z3, call_z3);
avx2_angular_entry!(ipred_smooth_avx2, IntraPredMode::Smooth, call_smooth);
avx2_angular_entry!(ipred_smooth_v_avx2, IntraPredMode::SmoothV, call_smooth_v);
avx2_angular_entry!(ipred_smooth_h_avx2, IntraPredMode::SmoothH, call_smooth_h);
avx2_angular_entry!(ipred_paeth_avx2, IntraPredMode::Paeth, call_paeth);
avx2_angular_entry!(ipred_filter_avx2, IntraPredMode::Filter, call_filter);

macro_rules! avx2_cfl_ac_entry {
    ($fn_name:ident, $layout:expr, $ss_hor:expr, $ss_ver:expr) => {
        fn $fn_name(
            ac: &mut [i16],
            y: &[u8],
            y_stride: usize,
            w_pad: usize,
            h_pad: usize,
            cw: usize,
            ch: usize,
            bit_depth: BitDepth,
        ) {
            if !cfl_ac_sozlesmesi::<8>(
                $layout,
                ac.len(),
                y.len(),
                y_stride,
                w_pad,
                h_pad,
                cw,
                ch,
                bit_depth,
            ) {
                cfl_ac_skaler_devret::<8>(
                    $layout, ac, y, y_stride, w_pad, h_pad, cw, ch, bit_depth,
                );
                return;
            }
            // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
            // (`cfl_ac_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır.
            // (2) sınır: `cfl_ac_sozlesmesi::<8>` `cw, ch`, `w_pad, h_pad`, `ac.len()`
            // ve `y` luma dilim sınırlarını O(1) doğruladı. (3) hizalama/takma ad: tüm
            // yükleme/saklamalar `mem` içindeki hizasız sabit boyutlu dizi yardımcılarıyla
            // yapılır; `ac` `&mut [i16]` ve `y` `&[u8]` ayrık referanslardır.
            unsafe {
                cfl_ac_core_avx2::<$ss_hor, $ss_ver>(ac, y, y_stride, w_pad, h_pad, cw, ch);
            }
        }
    };
}

avx2_cfl_ac_entry!(cfl_ac_420_avx2, 0, 1, 1);
avx2_cfl_ac_entry!(cfl_ac_422_avx2, 1, 1, 0);
avx2_cfl_ac_entry!(cfl_ac_444_avx2, 2, 0, 0);

macro_rules! avx2_cfl_pred_entry {
    ($fn_name:ident, $mode:expr) => {
        fn $fn_name(
            dst: &mut [u8],
            dst_stride: usize,
            edge: Edge<'_>,
            width: usize,
            height: usize,
            ac: &[i16],
            alpha: i32,
            bit_depth: BitDepth,
            bitdepth_max: u16,
        ) {
            if !cfl_pred_sozlesmesi::<8>(
                $mode,
                dst.len(),
                dst_stride,
                edge,
                width,
                height,
                ac.len(),
                alpha,
                bit_depth,
                bitdepth_max,
            ) {
                cfl_pred_skaler_devret::<8>(
                    $mode,
                    dst,
                    dst_stride,
                    edge,
                    width,
                    height,
                    ac,
                    alpha,
                    bit_depth,
                    bitdepth_max,
                );
                return;
            }
            // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
            // (`cfl_pred_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır.
            // (2) sınır: `cfl_pred_sozlesmesi::<8>` `width, height ∈ 4..=32`,
            // `edge`, `ac` ve `dst` sınırlarını O(1) doğruladı. (3) hizalama/takma ad:
            // tüm yükleme/saklamalar `mem` yardımcılarıyla hizasız yapılır; `dst`,
            // `edge.bytes` ve `ac` ayrık referanslardır.
            unsafe {
                cfl_pred_core_avx2($mode, dst, dst_stride, edge, width, height, ac, alpha);
            }
        }
    };
}

avx2_cfl_pred_entry!(cfl_pred_dc_avx2, CflPredMode::Dc);
avx2_cfl_pred_entry!(cfl_pred_left_dc_avx2, CflPredMode::LeftDc);
avx2_cfl_pred_entry!(cfl_pred_top_dc_avx2, CflPredMode::TopDc);
avx2_cfl_pred_entry!(cfl_pred_dc_128_avx2, CflPredMode::Dc128);

fn pal_pred_avx2(
    dst: &mut [u8],
    dst_stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
    bit_depth: BitDepth,
) {
    if !pal_pred_sozlesmesi::<8>(dst.len(), dst_stride, pal.len(), idx.len(), w, h, bit_depth) {
        pal_pred_skaler_devret::<8>(dst, dst_stride, pal, idx, w, h, bit_depth);
        return;
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`pal_pred_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır.
    // (2) sınır: `pal_pred_sozlesmesi::<8>` `w, h ∈ 4..=64`, `pal.len() >= 8`,
    // `idx.len() >= (w*h)/2` ve `dst` sınırlarını O(1) doğruladı. (3) hizalama/takma ad:
    // tüm yükleme/saklamalar `mem` yardımcılarıyla hizasız yapılır; `dst`, `pal`
    // ve `idx` ayrık referanslardır.
    unsafe { pal_pred_avx2_giris(dst, dst_stride, pal, idx, w, h) }
}
