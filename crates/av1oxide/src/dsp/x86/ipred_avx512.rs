//! 8 bpc intra tahmin (`intra_pred[14]`, `pal_pred`; 15 kayıt) x86_64 AVX512ICL
//! çekirdekleri — P45 Adım 2 (Paket A).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! `src/x86/ipred.h:133-151` kapsam eşitliği: 8-bpc'de `avx512icl` yalnız
//! `intra_pred[14]` + `pal_pred` (`1`) = **15 kayıt** tanımlar (`cfl_ac[3]` ve
//! `cfl_pred[4]` 8-bpc'de `avx512icl` tanımlamaz; `avx2`'de kalır).
//!
//! | Kayıtlar | Yuva | Kaynak |
//! |---|---|---|
//! | `D01620`, `D01625`, `D01630`, `D01636`, `D01641`, `D01646` | `intra_pred[DC/VERT/HOR/LEFT_DC/TOP_DC/DC_128]` | `src/x86/ipred_avx512.asm:123-411`, `src/x86/ipred.h:134-139` |
//! | `D01652`, `D01658`, `D01664` | `intra_pred[Z1/Z2/Z3]` | `src/x86/ipred_avx512.asm:412-1046`, `src/x86/ipred.h:140-142` |
//! | `D01670`, `D01675`, `D01680` | `intra_pred[SMOOTH/SMOOTH_V/SMOOTH_H]` | `src/x86/ipred_avx512.asm:1244-1692`, `src/x86/ipred.h:143-145` |
//! | `D01686` | `intra_pred[PAETH]` | `src/x86/ipred_avx512.asm:1047-1243`, `src/x86/ipred.h:146` |
//! | `D01692` | `intra_pred[FILTER]` | `src/x86/ipred_avx512.asm:1770-2177`, `src/x86/ipred.h:147` |
//! | `D01698` | `pal_pred` | `src/x86/ipred_avx512.asm:1693-1769`, `src/x86/ipred.h:149` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bu modülde tüm SIMD işlevleri
//!   `#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]`
//!   taşır; gather/scatter yönergesi üretilmez (`0` gather/scatter, `SLOW_GATHER`
//!   engeli yoktur).
//! - Kopyasız O(1) sözleşme: girişte [`angular_sozlesmesi::<8>`] ve
//!   [`pal_pred_sozlesmesi::<8>`] denetlenir; sözleşme dışı girdide ilgili
//!   `*_skaler_devret::<8>` ile skaler gövdeye devredilir.

#![allow(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use core::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm_abs_epi16, _mm_add_epi16, _mm_add_epi32, _mm_and_si128,
    _mm_andnot_si128, _mm_avg_epu16, _mm_cmpgt_epi16, _mm_cvtsi32_si128, _mm_cvtsi64_si128,
    _mm_cvtsi128_si32, _mm_maddubs_epi16, _mm_mulhrs_epi16, _mm_mullo_epi16, _mm_or_si128,
    _mm_packus_epi16, _mm_sad_epu8, _mm_set1_epi8, _mm_set1_epi16, _mm_setr_epi8,
    _mm_setzero_si128, _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_srai_epi16, _mm_srli_epi16,
    _mm_sub_epi16, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi8,
    _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm256_abs_epi16, _mm256_add_epi16, _mm256_add_epi32,
    _mm256_and_si256, _mm256_andnot_si256, _mm256_avg_epu16, _mm256_castsi256_si128,
    _mm256_cmpgt_epi16, _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_maddubs_epi16,
    _mm256_mulhrs_epi16, _mm256_mullo_epi16, _mm256_or_si256, _mm256_packus_epi16,
    _mm256_permute4x64_epi64, _mm256_sad_epu8, _mm256_set1_epi8, _mm256_set1_epi16,
    _mm256_set1_epi64x, _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8,
    _mm256_srli_epi16, _mm256_sub_epi16, _mm256_unpackhi_epi8, _mm256_unpacklo_epi8,
    _mm512_abs_epi16, _mm512_add_epi16, _mm512_avg_epu16, _mm512_cmpgt_epi16_mask,
    _mm512_cvtepu8_epi16, _mm512_cvtusepi16_epi8, _mm512_mask_blend_epi16, _mm512_mullo_epi16,
    _mm512_set1_epi8, _mm512_set1_epi16, _mm512_setzero_si512, _mm512_srli_epi16, _mm512_sub_epi16,
};

use crate::cpu::Avx512IclToken;
use crate::dsp::ipred::{
    AngularPredFn, Edge, IntraPredMode, PalPredFn, angular_skaler_devret, angular_sozlesmesi,
    pal_pred_skaler_devret, pal_pred_sozlesmesi,
};
use crate::dsp::ipred_tables::{DR_INTRA_DERIVATIVE, FILTER_INTRA_TAPS, SM_WEIGHTS};
use crate::dsp::x86::mem;
use crate::frame::BitDepth;

const TMP_LEN: usize = 257;
const TMP_BASE: isize = 128;

// ---------------------------------------------------------------------------
// Kurucular (`Avx512IclToken` kapılı, çalışma zamanında `Token::new` çağrılmaz)
// ---------------------------------------------------------------------------

/// 8 bpc `intra_pred[mode]` AVX512ICL yuva işaretçisi (`D01620`..`D01692`).
#[must_use]
pub const fn intra_pred_fn(_token: Avx512IclToken, mode: IntraPredMode) -> AngularPredFn {
    match mode {
        IntraPredMode::Dc => ipred_dc_avx512,
        IntraPredMode::Vert => ipred_vert_avx512,
        IntraPredMode::Hor => ipred_hor_avx512,
        IntraPredMode::LeftDc => ipred_left_dc_avx512,
        IntraPredMode::TopDc => ipred_top_dc_avx512,
        IntraPredMode::Dc128 => ipred_dc_128_avx512,
        IntraPredMode::Z1 => ipred_z1_avx512,
        IntraPredMode::Z2 => ipred_z2_avx512,
        IntraPredMode::Z3 => ipred_z3_avx512,
        IntraPredMode::Smooth => ipred_smooth_avx512,
        IntraPredMode::SmoothV => ipred_smooth_v_avx512,
        IntraPredMode::SmoothH => ipred_smooth_h_avx512,
        IntraPredMode::Paeth => ipred_paeth_avx512,
        IntraPredMode::Filter => ipred_filter_avx512,
    }
}

/// 8 bpc `pal_pred` AVX512ICL yuva işaretçisi (`D01698`).
#[must_use]
pub const fn pal_pred_fn(_token: Avx512IclToken) -> PalPredFn {
    pal_pred_avx512
}

// ---------------------------------------------------------------------------
// Ortak yardımcılar
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn sum_bytes_avx512(src: &[u8], n: usize) -> u32 {
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
            let s256 = _mm256_add_epi32(s0, s1);
            let lo = _mm256_castsi256_si128(s256);
            let hi = _mm256_extracti128_si256::<1>(s256);
            let sum = _mm_add_epi32(lo, hi);
            let sum = _mm_add_epi32(sum, _mm_unpackhi_epi64(sum, zero128));
            _mm_cvtsi128_si32(sum) as u32
        }
        _ => unreachable!(),
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn dc_gen_top_avx512(edge: Edge<'_>, width: usize) -> u32 {
    let top = &edge.bytes[edge.topleft + 1..edge.topleft + 1 + width];
    (sum_bytes_avx512(top, width) + (width as u32 >> 1)) >> width.trailing_zeros()
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn dc_gen_left_avx512(edge: Edge<'_>, height: usize) -> u32 {
    let left = &edge.bytes[edge.topleft - height..edge.topleft];
    (sum_bytes_avx512(left, height) + (height as u32 >> 1)) >> height.trailing_zeros()
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn dc_gen_avx512(edge: Edge<'_>, width: usize, height: usize) -> u32 {
    let top = &edge.bytes[edge.topleft + 1..edge.topleft + 1 + width];
    let left = &edge.bytes[edge.topleft - height..edge.topleft];
    let mut dc = sum_bytes_avx512(top, width)
        + sum_bytes_avx512(left, height)
        + ((width + height) as u32 >> 1);
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn splat_dc_avx512(dst: &mut [u8], stride: usize, width: usize, height: usize, dc: i8) {
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
            let v = _mm512_set1_epi8(dc);
            for y in 0..height {
                mem::store_u8x64(mem::chunk_mut::<u8, 64>(dst, y * stride), v);
            }
        }
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// 1. DC / VERT / HOR / LEFT_DC / TOP_DC / DC_128 (`D01620`..`D01646`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_dc_avx512_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let dc = dc_gen_avx512(edge, width, height) as i8;
    splat_dc_avx512(dst, dst_stride, width, height, dc);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_top_dc_avx512_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let dc = dc_gen_top_avx512(edge, width) as i8;
    splat_dc_avx512(dst, dst_stride, width, height, dc);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_left_dc_avx512_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let dc = dc_gen_left_avx512(edge, height) as i8;
    splat_dc_avx512(dst, dst_stride, width, height, dc);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_dc_128_avx512_giris(dst: &mut [u8], dst_stride: usize, width: usize, height: usize) {
    splat_dc_avx512(dst, dst_stride, width, height, -128i8);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_vert_avx512_giris(
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
            let v = mem::load_u8x64(mem::chunk::<u8, 64>(top, 0));
            for y in 0..height {
                mem::store_u8x64(mem::chunk_mut::<u8, 64>(dst, y * dst_stride), v);
            }
        }
        _ => unreachable!(),
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_hor_avx512_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let left = &edge.bytes[edge.topleft - height..edge.topleft];
    for y in 0..height {
        let b = left[height - 1 - y] as i8;
        let row = y * dst_stride;
        match width {
            4 => mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, row), _mm_set1_epi8(b)),
            8 => mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, row), _mm_set1_epi8(b)),
            16 => mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row), _mm_set1_epi8(b)),
            32 => mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row), _mm256_set1_epi8(b)),
            64 => mem::store_u8x64(mem::chunk_mut::<u8, 64>(dst, row), _mm512_set1_epi8(b)),
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------
// 2. PAETH (`D01686`) — 128/256/512-bit AVX512ICL
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn paeth_row8_avx512(
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn paeth_row16_avx512(
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn paeth_row32_avx512(
    t16: __m512i,
    ldiff16: __m512i,
    l16: __m512i,
    tl16: __m512i,
    tdiff16: __m512i,
) -> __m256i {
    let t_minus_tl = _mm512_sub_epi16(t16, tl16);
    let l_minus_tl = _mm512_sub_epi16(l16, tl16);
    let tldiff16 = _mm512_abs_epi16(_mm512_add_epi16(l_minus_tl, t_minus_tl));
    let not_l_mask =
        _mm512_cmpgt_epi16_mask(ldiff16, tdiff16) | _mm512_cmpgt_epi16_mask(ldiff16, tldiff16);
    let not_t_mask = _mm512_cmpgt_epi16_mask(tdiff16, tldiff16);
    let t_or_tl = _mm512_mask_blend_epi16(not_t_mask, t16, tl16);
    let res = _mm512_mask_blend_epi16(not_l_mask, l16, t_or_tl);
    _mm512_cvtusepi16_epi8(res)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_paeth_avx512_giris(
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
        let t16 = _mm_unpacklo_epi8(_mm_cvtsi64_si128(u64::from_le_bytes(top_buf) as i64), zero);
        let ldiff16 = _mm_abs_epi16(_mm_sub_epi16(t16, tl16));
        for y in 0..height {
            let l = i16::from(left_slice[height - 1 - y]);
            let l16 = _mm_set1_epi16(l);
            let tdiff16 = _mm_set1_epi16((l - tl).abs());
            let p0 = paeth_row8_avx512(t16, ldiff16, l16, tl16, tdiff16);
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

    if width == 16 {
        let tl256 = _mm256_set1_epi16(tl);
        let mut raw16 = [0i16; 16];
        for i in 0..16 {
            raw16[i] = i16::from(top_slice[i]);
        }
        let top256 = mem::load_i16x16(&raw16);
        let ldiff256 = _mm256_abs_epi16(_mm256_sub_epi16(top256, tl256));
        for y in 0..height {
            let l = i16::from(left_slice[height - 1 - y]);
            let l256 = _mm256_set1_epi16(l);
            let tdiff256 = _mm256_set1_epi16((l - tl).abs());
            let p0 = paeth_row16_avx512(top256, ldiff256, l256, tl256, tdiff256);
            let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(p0, p0));
            mem::store_u8x16(
                mem::chunk_mut::<u8, 16>(dst, y * dst_stride),
                _mm256_castsi256_si128(packed),
            );
        }
        return;
    }

    // width ∈ {32, 64}: 512-bit (32 × i16) şeritlerle işle
    let tl512 = _mm512_set1_epi16(tl);
    let mut top512 = [_mm512_setzero_si512(); 2];
    let mut ldiff512 = [_mm512_setzero_si512(); 2];
    let chunks32 = width / 32;
    for c in 0..chunks32 {
        let mut raw32 = [0i16; 32];
        for i in 0..32 {
            raw32[i] = i16::from(top_slice[c * 32 + i]);
        }
        let t16 = mem::load_i16x32(&raw32);
        top512[c] = t16;
        ldiff512[c] = _mm512_abs_epi16(_mm512_sub_epi16(t16, tl512));
    }

    for y in 0..height {
        let l = i16::from(left_slice[height - 1 - y]);
        let l512 = _mm512_set1_epi16(l);
        let tdiff512 = _mm512_set1_epi16((l - tl).abs());
        let row = y * dst_stride;
        let p0 = paeth_row32_avx512(top512[0], ldiff512[0], l512, tl512, tdiff512);
        mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row), p0);
        if width == 64 {
            let p1 = paeth_row32_avx512(top512[1], ldiff512[1], l512, tl512, tdiff512);
            mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row + 32), p1);
        }
    }
}

// ---------------------------------------------------------------------------
// 3. SMOOTH / SMOOTH_V / SMOOTH_H (`D01670`, `D01675`, `D01680`) — 128/256/512-bit
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn smooth_term8_avx512(
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn smooth_term16_avx512(
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn smooth_term32_avx512(
    a16: __m512i,
    b16: __m512i,
    w16: __m512i,
    inv_w16: __m512i,
    bias16: __m512i,
) -> __m512i {
    let p0 = _mm512_mullo_epi16(a16, w16);
    let p1 = _mm512_mullo_epi16(b16, inv_w16);
    _mm512_add_epi16(_mm512_add_epi16(p0, p1), bias16)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_smooth_v_avx512_giris(
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
            let s0 = _mm_srli_epi16::<8>(smooth_term8_avx512(
                top16, bottom16, wv16, inv_wv16, bias128,
            ));
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

    if width == 16 {
        let bottom256 = _mm256_set1_epi16(bottom);
        let bias128 = _mm256_set1_epi16(128);
        let top256 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(top_slice, 0)));
        for y in 0..height {
            let wv = i16::from(weights_ver[y]);
            let wv256 = _mm256_set1_epi16(wv);
            let inv_wv256 = _mm256_set1_epi16(256 - wv);
            let s0 = _mm256_srli_epi16::<8>(smooth_term16_avx512(
                top256, bottom256, wv256, inv_wv256, bias128,
            ));
            let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(s0, s0));
            mem::store_u8x16(
                mem::chunk_mut::<u8, 16>(dst, y * dst_stride),
                _mm256_castsi256_si128(packed),
            );
        }
        return;
    }

    let bottom512 = _mm512_set1_epi16(bottom);
    let bias128 = _mm512_set1_epi16(128);
    let mut top512 = [_mm512_setzero_si512(); 2];
    let chunks32 = width / 32;
    for c in 0..chunks32 {
        top512[c] = _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(top_slice, c * 32)));
    }

    for y in 0..height {
        let wv = i16::from(weights_ver[y]);
        let wv512 = _mm512_set1_epi16(wv);
        let inv_wv512 = _mm512_set1_epi16(256 - wv);
        let row = y * dst_stride;
        let s0 = _mm512_srli_epi16::<8>(smooth_term32_avx512(
            top512[0], bottom512, wv512, inv_wv512, bias128,
        ));
        mem::store_u8x32(
            mem::chunk_mut::<u8, 32>(dst, row),
            _mm512_cvtusepi16_epi8(s0),
        );
        if width == 64 {
            let s1 = _mm512_srli_epi16::<8>(smooth_term32_avx512(
                top512[1], bottom512, wv512, inv_wv512, bias128,
            ));
            mem::store_u8x32(
                mem::chunk_mut::<u8, 32>(dst, row + 32),
                _mm512_cvtusepi16_epi8(s1),
            );
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_smooth_h_avx512_giris(
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
            let s0 =
                _mm_srli_epi16::<8>(smooth_term8_avx512(l16, right16, wh16, inv_wh16, bias128));
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

    if width == 16 {
        let right256 = _mm256_set1_epi16(right);
        let bias128 = _mm256_set1_epi16(128);
        let c256 = _mm256_set1_epi16(256);
        let wh256 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(weights_hor, 0)));
        let inv_wh256 = _mm256_sub_epi16(c256, wh256);
        for y in 0..height {
            let l256 = _mm256_set1_epi16(i16::from(left_slice[height - 1 - y]));
            let s0 = _mm256_srli_epi16::<8>(smooth_term16_avx512(
                l256, right256, wh256, inv_wh256, bias128,
            ));
            let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(s0, s0));
            mem::store_u8x16(
                mem::chunk_mut::<u8, 16>(dst, y * dst_stride),
                _mm256_castsi256_si128(packed),
            );
        }
        return;
    }

    let right512 = _mm512_set1_epi16(right);
    let bias128 = _mm512_set1_epi16(128);
    let c256 = _mm512_set1_epi16(256);
    let mut wh512 = [_mm512_setzero_si512(); 2];
    let mut inv_wh512 = [_mm512_setzero_si512(); 2];
    let chunks32 = width / 32;
    for c in 0..chunks32 {
        let w = _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(weights_hor, c * 32)));
        wh512[c] = w;
        inv_wh512[c] = _mm512_sub_epi16(c256, w);
    }

    for y in 0..height {
        let l512 = _mm512_set1_epi16(i16::from(left_slice[height - 1 - y]));
        let row = y * dst_stride;
        let s0 = _mm512_srli_epi16::<8>(smooth_term32_avx512(
            l512,
            right512,
            wh512[0],
            inv_wh512[0],
            bias128,
        ));
        mem::store_u8x32(
            mem::chunk_mut::<u8, 32>(dst, row),
            _mm512_cvtusepi16_epi8(s0),
        );
        if width == 64 {
            let s1 = _mm512_srli_epi16::<8>(smooth_term32_avx512(
                l512,
                right512,
                wh512[1],
                inv_wh512[1],
                bias128,
            ));
            mem::store_u8x32(
                mem::chunk_mut::<u8, 32>(dst, row + 32),
                _mm512_cvtusepi16_epi8(s1),
            );
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_smooth_avx512_giris(
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
            let v = smooth_term8_avx512(top16, bottom16, wv16, inv_wv16, bias255);
            let h = smooth_term8_avx512(l16, right16, wh16, inv_wh16, zero);
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

    if width == 16 {
        let zero256 = _mm256_setzero_si256();
        let right256 = _mm256_set1_epi16(right);
        let bottom256 = _mm256_set1_epi16(bottom);
        let bias255 = _mm256_set1_epi16(255);
        let c256 = _mm256_set1_epi16(256);
        let top256 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(top_slice, 0)));
        let wh256 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(weights_hor, 0)));
        let inv_wh256 = _mm256_sub_epi16(c256, wh256);
        for y in 0..height {
            let wv = i16::from(weights_ver[y]);
            let wv256 = _mm256_set1_epi16(wv);
            let inv_wv256 = _mm256_set1_epi16(256 - wv);
            let l256 = _mm256_set1_epi16(i16::from(left_slice[height - 1 - y]));
            let v = smooth_term16_avx512(top256, bottom256, wv256, inv_wv256, bias255);
            let h = smooth_term16_avx512(l256, right256, wh256, inv_wh256, zero256);
            let s0 = _mm256_srli_epi16::<8>(_mm256_avg_epu16(v, h));
            let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(s0, s0));
            mem::store_u8x16(
                mem::chunk_mut::<u8, 16>(dst, y * dst_stride),
                _mm256_castsi256_si128(packed),
            );
        }
        return;
    }

    let zero512 = _mm512_setzero_si512();
    let right512 = _mm512_set1_epi16(right);
    let bottom512 = _mm512_set1_epi16(bottom);
    let bias255 = _mm512_set1_epi16(255);
    let c256 = _mm512_set1_epi16(256);

    let mut top512 = [_mm512_setzero_si512(); 2];
    let mut wh512 = [_mm512_setzero_si512(); 2];
    let mut inv_wh512 = [_mm512_setzero_si512(); 2];
    let chunks32 = width / 32;
    for c in 0..chunks32 {
        top512[c] = _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(top_slice, c * 32)));
        let w = _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(weights_hor, c * 32)));
        wh512[c] = w;
        inv_wh512[c] = _mm512_sub_epi16(c256, w);
    }

    for y in 0..height {
        let wv = i16::from(weights_ver[y]);
        let wv512 = _mm512_set1_epi16(wv);
        let inv_wv512 = _mm512_set1_epi16(256 - wv);
        let l512 = _mm512_set1_epi16(i16::from(left_slice[height - 1 - y]));
        let row = y * dst_stride;

        let v0 = smooth_term32_avx512(top512[0], bottom512, wv512, inv_wv512, bias255);
        let h0 = smooth_term32_avx512(l512, right512, wh512[0], inv_wh512[0], zero512);
        let s0 = _mm512_srli_epi16::<8>(_mm512_avg_epu16(v0, h0));
        mem::store_u8x32(
            mem::chunk_mut::<u8, 32>(dst, row),
            _mm512_cvtusepi16_epi8(s0),
        );

        if width == 64 {
            let v1 = smooth_term32_avx512(top512[1], bottom512, wv512, inv_wv512, bias255);
            let h1 = smooth_term32_avx512(l512, right512, wh512[1], inv_wh512[1], zero512);
            let s1 = _mm512_srli_epi16::<8>(_mm512_avg_epu16(v1, h1));
            mem::store_u8x32(
                mem::chunk_mut::<u8, 32>(dst, row + 32),
                _mm512_cvtusepi16_epi8(s1),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 4. FILTER_PRED (`D01692`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_filter_avx512_giris(
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
// 5. Z1 / Z2 / Z3 (`D01652`, `D01658`, `D01664`)
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_z1_avx512_giris(
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
                    let b0 = base;
                    let b1 = (base + 16).min(max_base_x);
                    let s0 = _mm256_permute4x64_epi64::<0x94>(mem::load_u8x32(
                        mem::chunk::<u8, 32>(&buf, b0),
                    ));
                    let s1 = _mm256_permute4x64_epi64::<0x94>(mem::load_u8x32(
                        mem::chunk::<u8, 32>(&buf, b1),
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
                        let b0 = (base + g * 32).min(max_base_x);
                        let b1 = (base + g * 32 + 16).min(max_base_x);
                        let s0 =
                            _mm256_permute4x64_epi64::<0x94>(mem::load_u8x32(
                                mem::chunk::<u8, 32>(&buf, b0),
                            ));
                        let s1 =
                            _mm256_permute4x64_epi64::<0x94>(mem::load_u8x32(
                                mem::chunk::<u8, 32>(&buf, b1),
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_z2_avx512_giris(
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
    let y_step = 1usize << u32::from(upsample_left);

    // Sol kenar için y'den bağımsız ağırlıkları ve taban indekslerini önceden hesapla.
    let mut left_idx0 = [0u8; 64];
    let mut left_w_bytes = [0u8; 128];
    let mut ypos0 = -(dy as isize);
    for col in 0..width {
        let by = ypos0 >> 6;
        let fy = (ypos0 & 0x3E) as u8;
        left_idx0[col] = (left - by) as u8;
        left_w_bytes[2 * col] = 64 - fy;
        left_w_bytes[2 * col + 1] = fy;
        ypos0 -= dy as isize;
    }

    for y in 0..height {
        let mut base_x = xpos >> 6;
        let frac_x = (xpos & 0x3E) as i32;
        let w16_x = u16::from_le_bytes([(64 - frac_x) as u8, frac_x as u8]) as i16;
        let wpair_x = _mm_set1_epi16(w16_x);
        let wpair_x256 = _mm256_set1_epi16(w16_x);
        let y_off = y * y_step;
        let row = y * stride;

        let mut x = 0;
        while x + 16 <= width && base_x >= 0 && base_inc_x == 1 {
            let idx = (base + base_x) as usize;
            let s0 =
                _mm256_permute4x64_epi64::<0x94>(mem::load_u8x32(mem::chunk::<u8, 32>(&buf, idx)));
            let pairs = _mm256_shuffle_epi8(s0, shuf256);
            let prod = _mm256_mulhrs_epi16(_mm256_maddubs_epi16(pairs, wpair_x256), mul512_256);
            let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(prod, prod));
            mem::store_u8x16(
                mem::chunk_mut::<u8, 16>(dst, row + x),
                _mm256_castsi256_si128(packed),
            );
            base_x += 16;
            x += 16;
        }
        while x + 8 <= width {
            if base_x >= 0 {
                let idx = (base + base_x) as usize;
                let pairs = if base_inc_x == 1 {
                    _mm_shuffle_epi8(mem::load_u8x16(mem::chunk::<u8, 16>(&buf, idx)), shuf_inc1)
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
                        w_bytes[2 * k] = (64 - frac_x) as u8;
                        w_bytes[2 * k + 1] = frac_x as u8;
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn transpose_store_4x4_u8(
    cols: [__m128i; 4],
    dst: &mut [u8],
    stride: usize,
    row_off: usize,
    col_off: usize,
) {
    let t01 = _mm_unpacklo_epi8(cols[0], cols[1]);
    let t23 = _mm_unpacklo_epi8(cols[2], cols[3]);
    let r01 = _mm_unpacklo_epi16(t01, t23);
    let r23 = _mm_unpackhi_epi16(t01, t23);
    let base = row_off * stride + col_off;
    mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, base), r01);
    mem::store_u8x4(
        mem::chunk_mut::<u8, 4>(dst, base + stride),
        _mm_shuffle_epi32::<0x55>(r01),
    );
    mem::store_u8x4(
        mem::chunk_mut::<u8, 4>(dst, base + 2 * stride),
        _mm_shuffle_epi32::<0xAA>(r01),
    );
    mem::store_u8x4(
        mem::chunk_mut::<u8, 4>(dst, base + 3 * stride),
        _mm_shuffle_epi32::<0xFF>(r01),
    );
    let _ = r23;
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn transpose_store_8x8_u8(
    cols: [__m128i; 8],
    dst: &mut [u8],
    stride: usize,
    row_off: usize,
    col_off: usize,
) {
    let a01 = _mm_unpacklo_epi8(cols[0], cols[1]);
    let a23 = _mm_unpacklo_epi8(cols[2], cols[3]);
    let a45 = _mm_unpacklo_epi8(cols[4], cols[5]);
    let a67 = _mm_unpacklo_epi8(cols[6], cols[7]);

    let b0 = _mm_unpacklo_epi16(a01, a23);
    let b1 = _mm_unpackhi_epi16(a01, a23);
    let b2 = _mm_unpacklo_epi16(a45, a67);
    let b3 = _mm_unpackhi_epi16(a45, a67);

    let r01 = _mm_unpacklo_epi32(b0, b2);
    let r23 = _mm_unpackhi_epi32(b0, b2);
    let r45 = _mm_unpacklo_epi32(b1, b3);
    let r67 = _mm_unpackhi_epi32(b1, b3);

    let zero = _mm_setzero_si128();
    let base = row_off * stride + col_off;
    mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, base), r01);
    mem::store_u8x8(
        mem::chunk_mut::<u8, 8>(dst, base + stride),
        _mm_unpackhi_epi64(r01, zero),
    );
    mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, base + 2 * stride), r23);
    mem::store_u8x8(
        mem::chunk_mut::<u8, 8>(dst, base + 3 * stride),
        _mm_unpackhi_epi64(r23, zero),
    );
    mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, base + 4 * stride), r45);
    mem::store_u8x8(
        mem::chunk_mut::<u8, 8>(dst, base + 5 * stride),
        _mm_unpackhi_epi64(r45, zero),
    );
    mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, base + 6 * stride), r67);
    mem::store_u8x8(
        mem::chunk_mut::<u8, 8>(dst, base + 7 * stride),
        _mm_unpackhi_epi64(r67, zero),
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_z3_avx512_giris(
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
    let mut left_out = [0u8; TMP_LEN];
    let upsample = enable && get_upsample(width + height, base_angle - 180, is_sm);
    let strength = if enable && !upsample {
        get_filter_strength(width + height, base_angle - 180, is_sm)
    } else {
        0
    };
    let direct = !upsample && strength == 0;

    let mut rev_buf = [0u8; 256];
    let (base_inc, max_base_y) = if upsample {
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
    let pad_len = (height * base_inc).max(32);
    rev_buf[max_base_y + 1..max_base_y + 1 + pad_len].fill(fill);

    let shuf_inc1 = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);
    let shuf256 = _mm256_setr_epi8(
        0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7,
        7, 8,
    );
    let mul512 = _mm_set1_epi16(512);
    let mul512_256 = _mm256_set1_epi16(512);
    let dy_u = dy as usize;

    if width == 4 {
        let mut wpair = [_mm_setzero_si128(); 4];
        let mut wpair256 = [_mm256_setzero_si256(); 4];
        let mut bases = [0usize; 4];
        let mut ypos = dy_u;
        for k in 0..4 {
            let frac = (ypos & 0x3E) as i8;
            let w16 = u16::from_le_bytes([(64 - frac) as u8, frac as u8]) as i16;
            wpair[k] = _mm_set1_epi16(w16);
            wpair256[k] = _mm256_set1_epi16(w16);
            bases[k] = ypos >> 6;
            ypos += dy_u;
        }
        if height == 4 {
            let mut cols = [_mm_setzero_si128(); 4];
            for k in 0..4 {
                let b = bases[k].min(max_base_y);
                let p = if base_inc == 1 {
                    _mm_shuffle_epi8(
                        mem::load_u8x16(mem::chunk::<u8, 16>(&rev_buf, b)),
                        shuf_inc1,
                    )
                } else {
                    mem::load_u8x16(mem::chunk::<u8, 16>(&rev_buf, b))
                };
                let r = _mm_mulhrs_epi16(_mm_maddubs_epi16(p, wpair[k]), mul512);
                cols[k] = _mm_packus_epi16(r, r);
            }
            transpose_store_4x4_u8(cols, dst, stride, 0, 0);
        } else if height == 8 {
            let mut cols = [_mm_setzero_si128(); 4];
            for k in 0..4 {
                let b = bases[k].min(max_base_y);
                let p = if base_inc == 1 {
                    _mm_shuffle_epi8(
                        mem::load_u8x16(mem::chunk::<u8, 16>(&rev_buf, b)),
                        shuf_inc1,
                    )
                } else {
                    mem::load_u8x16(mem::chunk::<u8, 16>(&rev_buf, b))
                };
                let r = _mm_mulhrs_epi16(_mm_maddubs_epi16(p, wpair[k]), mul512);
                cols[k] = _mm_packus_epi16(r, r);
            }
            let t01 = _mm_unpacklo_epi8(cols[0], cols[1]);
            let t23 = _mm_unpacklo_epi8(cols[2], cols[3]);
            let r03 = _mm_unpacklo_epi16(t01, t23);
            let r47 = _mm_unpackhi_epi16(t01, t23);
            mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, 0), r03);
            mem::store_u8x4(
                mem::chunk_mut::<u8, 4>(dst, stride),
                _mm_shuffle_epi32::<0x55>(r03),
            );
            mem::store_u8x4(
                mem::chunk_mut::<u8, 4>(dst, 2 * stride),
                _mm_shuffle_epi32::<0xAA>(r03),
            );
            mem::store_u8x4(
                mem::chunk_mut::<u8, 4>(dst, 3 * stride),
                _mm_shuffle_epi32::<0xFF>(r03),
            );
            mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, 4 * stride), r47);
            mem::store_u8x4(
                mem::chunk_mut::<u8, 4>(dst, 5 * stride),
                _mm_shuffle_epi32::<0x55>(r47),
            );
            mem::store_u8x4(
                mem::chunk_mut::<u8, 4>(dst, 6 * stride),
                _mm_shuffle_epi32::<0xAA>(r47),
            );
            mem::store_u8x4(
                mem::chunk_mut::<u8, 4>(dst, 7 * stride),
                _mm_shuffle_epi32::<0xFF>(r47),
            );
        } else {
            let mut y = 0;
            while y < height {
                let mut cols = [_mm_setzero_si128(); 4];
                for k in 0..4 {
                    let b = (bases[k] + y).min(max_base_y);
                    let s0 = _mm256_permute4x64_epi64::<0x94>(mem::load_u8x32(
                        mem::chunk::<u8, 32>(&rev_buf, b),
                    ));
                    let p = _mm256_shuffle_epi8(s0, shuf256);
                    let r = _mm256_mulhrs_epi16(_mm256_maddubs_epi16(p, wpair256[k]), mul512_256);
                    let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(r, r));
                    cols[k] = _mm256_castsi256_si128(packed);
                }
                let t01_lo = _mm_unpacklo_epi8(cols[0], cols[1]);
                let t23_lo = _mm_unpacklo_epi8(cols[2], cols[3]);
                let r03 = _mm_unpacklo_epi16(t01_lo, t23_lo);
                let r47 = _mm_unpackhi_epi16(t01_lo, t23_lo);
                let zero = _mm_setzero_si128();
                let c0_hi = _mm_unpackhi_epi64(cols[0], zero);
                let c1_hi = _mm_unpackhi_epi64(cols[1], zero);
                let c2_hi = _mm_unpackhi_epi64(cols[2], zero);
                let c3_hi = _mm_unpackhi_epi64(cols[3], zero);
                let t01_hi = _mm_unpacklo_epi8(c0_hi, c1_hi);
                let t23_hi = _mm_unpacklo_epi8(c2_hi, c3_hi);
                let r8b = _mm_unpacklo_epi16(t01_hi, t23_hi);
                let rcf = _mm_unpackhi_epi16(t01_hi, t23_hi);

                for (off, reg) in [(0, r03), (4, r47), (8, r8b), (12, rcf)] {
                    let base_row = (y + off) * stride;
                    mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, base_row), reg);
                    mem::store_u8x4(
                        mem::chunk_mut::<u8, 4>(dst, base_row + stride),
                        _mm_shuffle_epi32::<0x55>(reg),
                    );
                    mem::store_u8x4(
                        mem::chunk_mut::<u8, 4>(dst, base_row + 2 * stride),
                        _mm_shuffle_epi32::<0xAA>(reg),
                    );
                    mem::store_u8x4(
                        mem::chunk_mut::<u8, 4>(dst, base_row + 3 * stride),
                        _mm_shuffle_epi32::<0xFF>(reg),
                    );
                }
                y += 16;
            }
        }
        return;
    }

    let mut x = 0;
    let mut ypos_x = dy_u;
    while x < width {
        let mut wpair = [_mm_setzero_si128(); 8];
        let mut wpair256 = [_mm256_setzero_si256(); 8];
        let mut bases = [0usize; 8];
        for k in 0..8 {
            let frac = (ypos_x & 0x3E) as i8;
            let w16 = u16::from_le_bytes([(64 - frac) as u8, frac as u8]) as i16;
            wpair[k] = _mm_set1_epi16(w16);
            wpair256[k] = _mm256_set1_epi16(w16);
            bases[k] = ypos_x >> 6;
            ypos_x += dy_u;
        }

        if height == 4 {
            let mut cols = [_mm_setzero_si128(); 8];
            for k in 0..8 {
                let b = bases[k].min(max_base_y);
                let p = if base_inc == 1 {
                    _mm_shuffle_epi8(
                        mem::load_u8x16(mem::chunk::<u8, 16>(&rev_buf, b)),
                        shuf_inc1,
                    )
                } else {
                    mem::load_u8x16(mem::chunk::<u8, 16>(&rev_buf, b))
                };
                let r = _mm_mulhrs_epi16(_mm_maddubs_epi16(p, wpair[k]), mul512);
                cols[k] = _mm_packus_epi16(r, r);
            }
            let a01 = _mm_unpacklo_epi8(cols[0], cols[1]);
            let a23 = _mm_unpacklo_epi8(cols[2], cols[3]);
            let a45 = _mm_unpacklo_epi8(cols[4], cols[5]);
            let a67 = _mm_unpacklo_epi8(cols[6], cols[7]);
            let b0 = _mm_unpacklo_epi16(a01, a23);
            let b2 = _mm_unpacklo_epi16(a45, a67);
            let r01 = _mm_unpacklo_epi32(b0, b2);
            let r23 = _mm_unpackhi_epi32(b0, b2);
            let zero = _mm_setzero_si128();
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, x), r01);
            mem::store_u8x8(
                mem::chunk_mut::<u8, 8>(dst, stride + x),
                _mm_unpackhi_epi64(r01, zero),
            );
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, 2 * stride + x), r23);
            mem::store_u8x8(
                mem::chunk_mut::<u8, 8>(dst, 3 * stride + x),
                _mm_unpackhi_epi64(r23, zero),
            );
        } else if height == 8 || base_inc != 1 {
            let mut cols = [_mm_setzero_si128(); 8];
            for k in 0..8 {
                let b = bases[k].min(max_base_y);
                let p = if base_inc == 1 {
                    _mm_shuffle_epi8(
                        mem::load_u8x16(mem::chunk::<u8, 16>(&rev_buf, b)),
                        shuf_inc1,
                    )
                } else {
                    mem::load_u8x16(mem::chunk::<u8, 16>(&rev_buf, b))
                };
                let r = _mm_mulhrs_epi16(_mm_maddubs_epi16(p, wpair[k]), mul512);
                cols[k] = _mm_packus_epi16(r, r);
            }
            transpose_store_8x8_u8(cols, dst, stride, 0, x);
        } else {
            let zero = _mm_setzero_si128();
            let mut y = 0;
            while y < height {
                let mut cols_lo = [_mm_setzero_si128(); 8];
                let mut cols_hi = [_mm_setzero_si128(); 8];
                for k in 0..8 {
                    let b = (bases[k] + y).min(max_base_y);
                    let s0 = _mm256_permute4x64_epi64::<0x94>(mem::load_u8x32(
                        mem::chunk::<u8, 32>(&rev_buf, b),
                    ));
                    let p = _mm256_shuffle_epi8(s0, shuf256);
                    let r = _mm256_mulhrs_epi16(_mm256_maddubs_epi16(p, wpair256[k]), mul512_256);
                    let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(r, r));
                    let v128 = _mm256_castsi256_si128(packed);
                    cols_lo[k] = v128;
                    cols_hi[k] = _mm_unpackhi_epi64(v128, zero);
                }
                transpose_store_8x8_u8(cols_lo, dst, stride, y, x);
                transpose_store_8x8_u8(cols_hi, dst, stride, y + 8, x);
                y += 16;
            }
        }
        x += 8;
    }
}

// ---------------------------------------------------------------------------
// 6. PAL_PRED (`D01698`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn pal_pred_avx512_giris(
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
            for y in 0..h {
                let raw8 = u64::from_le_bytes(*mem::chunk::<u8, 8>(idx, y * 8));
                let v_idx = _mm_cvtsi64_si128(raw8 as i64);
                let lo = _mm_and_si128(v_idx, mask_lo128);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(v_idx), mask_lo128);
                let even = _mm_shuffle_epi8(v_pal128, lo);
                let odd = _mm_shuffle_epi8(v_pal128, hi);
                mem::store_u8x16(
                    mem::chunk_mut::<u8, 16>(dst, y * dst_stride),
                    _mm_unpacklo_epi8(even, odd),
                );
            }
        }
        32 => {
            let mut y = 0;
            while y < h {
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

macro_rules! avx512_angular_entry {
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
            // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
            // (`intra_pred_fn`); belirteç AVX512ICL tespiti + maske izninin kanıtıdır.
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
    ipred_dc_avx512_giris(dst, s, e, w, h);
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
    ipred_vert_avx512_giris(dst, s, e, w, h);
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
    ipred_hor_avx512_giris(dst, s, e, w, h);
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
    ipred_left_dc_avx512_giris(dst, s, e, w, h);
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
    ipred_top_dc_avx512_giris(dst, s, e, w, h);
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
    ipred_dc_128_avx512_giris(dst, s, w, h);
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
    ipred_z1_avx512_giris(dst, s, e, w, h, a);
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
    ipred_z2_avx512_giris(dst, s, e, w, h, a, mw, mh);
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
    ipred_z3_avx512_giris(dst, s, e, w, h, a);
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
    ipred_smooth_avx512_giris(dst, s, e, w, h);
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
    ipred_smooth_v_avx512_giris(dst, s, e, w, h);
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
    ipred_smooth_h_avx512_giris(dst, s, e, w, h);
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
    ipred_paeth_avx512_giris(dst, s, e, w, h);
}
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
    ipred_filter_avx512_giris(dst, s, e, w, h, a);
}

avx512_angular_entry!(ipred_dc_avx512, IntraPredMode::Dc, call_dc);
avx512_angular_entry!(ipred_vert_avx512, IntraPredMode::Vert, call_vert);
avx512_angular_entry!(ipred_hor_avx512, IntraPredMode::Hor, call_hor);
avx512_angular_entry!(ipred_left_dc_avx512, IntraPredMode::LeftDc, call_left_dc);
avx512_angular_entry!(ipred_top_dc_avx512, IntraPredMode::TopDc, call_top_dc);
avx512_angular_entry!(ipred_dc_128_avx512, IntraPredMode::Dc128, call_dc_128);
avx512_angular_entry!(ipred_z1_avx512, IntraPredMode::Z1, call_z1);
avx512_angular_entry!(ipred_z2_avx512, IntraPredMode::Z2, call_z2);
avx512_angular_entry!(ipred_z3_avx512, IntraPredMode::Z3, call_z3);
avx512_angular_entry!(ipred_smooth_avx512, IntraPredMode::Smooth, call_smooth);
avx512_angular_entry!(ipred_smooth_v_avx512, IntraPredMode::SmoothV, call_smooth_v);
avx512_angular_entry!(ipred_smooth_h_avx512, IntraPredMode::SmoothH, call_smooth_h);
avx512_angular_entry!(ipred_paeth_avx512, IntraPredMode::Paeth, call_paeth);
avx512_angular_entry!(ipred_filter_avx512, IntraPredMode::Filter, call_filter);

fn pal_pred_avx512(
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`pal_pred_fn`); belirteç AVX512ICL tespiti + maske izninin kanıtıdır.
    // (2) sınır: `pal_pred_sozlesmesi::<8>` `w, h ∈ 4..=64`, `pal.len() >= 8`,
    // `idx.len() >= (w*h)/2` ve `dst` sınırlarını O(1) doğruladı. (3) hizalama/takma ad:
    // tüm yükleme/saklamalar `mem` yardımcılarıyla hizasız yapılır; `dst`, `pal`
    // ve `idx` ayrık referanslardır.
    unsafe { pal_pred_avx512_giris(dst, dst_stride, pal, idx, w, h) }
}
