//! AVX-512 (Ice Lake) 16-bpc (`10/12` bit ortak) intra tahmin çekirdekleri (`9` kayıt / `18` yuva; P45 Adım 2).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! `src/x86/ipred.h:142-151` kapsam eşitliği gereği 16-bpc AVX-512ICL yalnızca
//! `8` `intra_pred` kipi (`Paeth`, `Smooth`, `SmoothH`, `SmoothV`, `Z1`, `Z2`,
//! `Z3`, `Filter`) ve `pal_pred` (`1`) için tanımlanır (`DC`, `VERT`, `HOR`,
//! `LEFT_DC`, `TOP_DC`, `DC_128`, `cfl_ac[3]` ve `cfl_pred[4]` 16-bpc'de
//! `avx2`'de kalır):
//!
//! | Yuva | Sembol | Kaynak |
//! |---|---|---|
//! | `intra_pred[PAETH_PRED]` (`D01647`) | `dav1d_ipred_paeth_16bpc_avx512icl` | `src/x86/ipred16_avx512.asm:117-227` |
//! | `intra_pred[SMOOTH_V_PRED]` (`D01665`) | `dav1d_ipred_smooth_v_16bpc_avx512icl` | `src/x86/ipred16_avx512.asm:229-313` |
//! | `intra_pred[SMOOTH_H_PRED]` (`D01653`) | `dav1d_ipred_smooth_h_16bpc_avx512icl` | `src/x86/ipred16_avx512.asm:315-419` |
//! | `intra_pred[SMOOTH_PRED]` (`D01659`) | `dav1d_ipred_smooth_16bpc_avx512icl` | `src/x86/ipred16_avx512.asm:421-570` |
//! | `intra_pred[Z1_PRED]` (`D01681`) | `dav1d_ipred_z1_16bpc_avx512icl` | `src/x86/ipred16_avx512.asm:572-1215` |
//! | `intra_pred[Z2_PRED]` (`D01687`) | `dav1d_ipred_z2_16bpc_avx512icl` | `src/x86/ipred16_avx512.asm:1217-1895` |
//! | `intra_pred[Z3_PRED]` (`D01693`) | `dav1d_ipred_z3_16bpc_avx512icl` | `src/x86/ipred16_avx512.asm:1897-2418` |
//! | `intra_pred[FILTER_PRED]` (`D01631`) | `dav1d_ipred_filter_16bpc_avx512icl` | `src/x86/ipred16_avx512.asm:2521-2658` |
//! | `pal_pred` (`D01699`) | `dav1d_pal_pred_16bpc_avx512icl` | `src/x86/ipred16_avx512.asm:2420-2511` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bütün SIMD işlevleri `#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]`
//!   taşır; `vpgather*`/`vpscatter*` üretilmez (`0` gather/scatter, `SLOW_GATHER`
//!   engeli yoktur).
//! - Kopyasız O(1) sözleşme: girişlerde `ipred16_ssse3` içindeki ortak sözleşme
//!   denetleyicileri kullanılır; sözleşme dışı girdide skaler gövdeye devredilir.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use core::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm_abs_epi16, _mm_add_epi16, _mm_blendv_epi8, _mm_bsrli_si128,
    _mm_cmpeq_epi16, _mm_cmpgt_epi16, _mm_cvtsi64_si128, _mm_dpwssd_epi32, _mm_max_epi16,
    _mm_min_epi16, _mm_mulhrs_epi16, _mm_packs_epi32, _mm_set1_epi16, _mm_set1_epi32,
    _mm_setr_epi8, _mm_setr_epi16, _mm_setzero_si128, _mm_shuffle_epi8, _mm_srai_epi32,
    _mm_sub_epi16, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi16,
    _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm256_abs_epi16, _mm256_add_epi16, _mm256_blendv_epi8,
    _mm256_broadcastsi128_si256, _mm256_castsi256_si128, _mm256_cmpeq_epi16, _mm256_cmpgt_epi16,
    _mm256_cvtepi8_epi16, _mm256_cvtepu8_epi32, _mm256_dpwssd_epi32, _mm256_extracti128_si256,
    _mm256_min_epi16, _mm256_mulhrs_epi16, _mm256_multishift_epi64_epi8, _mm256_packs_epi32,
    _mm256_permutexvar_epi16, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_epi64x,
    _mm256_setr_epi16, _mm256_shuffle_epi32, _mm256_srai_epi32, _mm256_srli_epi16,
    _mm256_sub_epi16, _mm256_unpackhi_epi16, _mm256_unpacklo_epi16, _mm512_abs_epi16,
    _mm512_add_epi16, _mm512_broadcast_i32x4, _mm512_castsi512_si128, _mm512_castsi512_si256,
    _mm512_cmpeq_epi16_mask, _mm512_cmpgt_epi16_mask, _mm512_cvtepu8_epi32, _mm512_dpwssd_epi32,
    _mm512_extracti32x4_epi32, _mm512_mask_blend_epi16, _mm512_mulhrs_epi16,
    _mm512_multishift_epi64_epi8, _mm512_packs_epi32, _mm512_permutexvar_epi16, _mm512_set1_epi16,
    _mm512_set1_epi32, _mm512_set1_epi64, _mm512_setr_epi32, _mm512_shuffle_i32x4,
    _mm512_srai_epi32, _mm512_srli_epi16, _mm512_sub_epi16, _mm512_unpackhi_epi16,
    _mm512_unpacklo_epi16,
};

use crate::cpu::Avx512IclToken;
use crate::dsp::ipred::{AngularPredFn, Edge, IntraPredMode, PalPredFn};
use crate::dsp::ipred_tables::{FILTER_INTRA_TAPS, SM_WEIGHTS};
use crate::dsp::x86::ipred16_ssse3::{
    angular_kontrol_16, pal_pred_kontrol_16, read_px_16, z1_hazirla_16, z2_hazirla_16,
    z3_hazirla_16,
};
use crate::dsp::x86::mem;
use crate::frame::BitDepth;

const TMP_BASE: isize = 128;

/// `intra_pred[mode]` 16-bpc AVX-512ICL yuva kurucusu (yalnız `8` kip:
/// `Paeth`, `Smooth`, `SmoothH`, `SmoothV`, `Z1`, `Z2`, `Z3`, `Filter`).
#[must_use]
pub const fn intra_pred_fn(_token: Avx512IclToken, mode: IntraPredMode) -> Option<AngularPredFn> {
    match mode {
        IntraPredMode::Paeth => Some(ipred_paeth_16bpc_avx512),
        IntraPredMode::Smooth => Some(ipred_smooth_16bpc_avx512),
        IntraPredMode::SmoothH => Some(ipred_smooth_h_16bpc_avx512),
        IntraPredMode::SmoothV => Some(ipred_smooth_v_16bpc_avx512),
        IntraPredMode::Z1 => Some(ipred_z1_16bpc_avx512),
        IntraPredMode::Z2 => Some(ipred_z2_16bpc_avx512),
        IntraPredMode::Z3 => Some(ipred_z3_16bpc_avx512),
        IntraPredMode::Filter => Some(ipred_filter_16bpc_avx512),
        IntraPredMode::Dc
        | IntraPredMode::Vert
        | IntraPredMode::Hor
        | IntraPredMode::LeftDc
        | IntraPredMode::TopDc
        | IntraPredMode::Dc128 => None,
    }
}

/// `pal_pred` 16-bpc AVX-512ICL yuva kurucusu.
#[must_use]
pub const fn pal_pred_fn(_token: Avx512IclToken) -> PalPredFn {
    pal_pred_16bpc_avx512
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn load_u8x8(src: &[u8; 8]) -> __m128i {
    _mm_cvtsi64_si128(i64::from_le_bytes(*src))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn splat_dc_avx512(dst: &mut [u8], dst_stride: usize, width: usize, height: usize, dc: i16) {
    if width == 4 {
        let v = _mm_set1_epi16(dc);
        for y in 0..height {
            mem::store_u8x8(mem::chunk_mut(dst, y * dst_stride), v);
        }
    } else if width == 8 {
        let v = _mm_set1_epi16(dc);
        for y in 0..height {
            mem::store_u8x16(mem::chunk_mut(dst, y * dst_stride), v);
        }
    } else if width == 16 {
        let v = _mm256_set1_epi16(dc);
        for y in 0..height {
            mem::store_u8x32(mem::chunk_mut(dst, y * dst_stride), v);
        }
    } else {
        let v = _mm512_set1_epi16(dc);
        let row_bytes = width * 2;
        for y in 0..height {
            let row = &mut dst[y * dst_stride..y * dst_stride + row_bytes];
            for d in row.as_chunks_mut::<64>().0 {
                mem::store_u8x64(d, v);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PAETH
// ---------------------------------------------------------------------------

fn ipred_paeth_16bpc_avx512(
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
    if !angular_kontrol_16(
        IntraPredMode::Paeth,
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
    ) {
        return;
    }
    // SAFETY: (1) ISA: işlev yalnız `Avx512IclToken` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { ipred_paeth_16bpc_avx512_giris(dst, dst_stride, edge, width, height) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn paeth_vec_128_avx512(
    top: __m128i,
    left: __m128i,
    tl: __m128i,
    signed_ldiff: __m128i,
    ldiff: __m128i,
) -> __m128i {
    let base = _mm_add_epi16(signed_ldiff, left);
    let tldiff = _mm_abs_epi16(_mm_sub_epi16(tl, base));
    let tdiff = _mm_abs_epi16(_mm_sub_epi16(base, top));
    let min_t_tl = _mm_min_epi16(tldiff, tdiff);
    let mask_t = _mm_cmpeq_epi16(tdiff, min_t_tl);
    let t_or_tl = _mm_blendv_epi8(tl, top, mask_t);
    let mask_not_l = _mm_cmpgt_epi16(ldiff, min_t_tl);
    _mm_blendv_epi8(left, t_or_tl, mask_not_l)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn paeth_vec_256_avx512(
    top: __m256i,
    left: __m256i,
    tl: __m256i,
    signed_ldiff: __m256i,
    ldiff: __m256i,
) -> __m256i {
    let base = _mm256_add_epi16(signed_ldiff, left);
    let tldiff = _mm256_abs_epi16(_mm256_sub_epi16(tl, base));
    let tdiff = _mm256_abs_epi16(_mm256_sub_epi16(base, top));
    let min_t_tl = _mm256_min_epi16(tldiff, tdiff);
    let mask_t = _mm256_cmpeq_epi16(tdiff, min_t_tl);
    let t_or_tl = _mm256_blendv_epi8(tl, top, mask_t);
    let mask_not_l = _mm256_cmpgt_epi16(ldiff, min_t_tl);
    _mm256_blendv_epi8(left, t_or_tl, mask_not_l)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn paeth_vec_512_avx512(
    top: __m512i,
    left: __m512i,
    tl: __m512i,
    signed_ldiff: __m512i,
    ldiff: __m512i,
) -> __m512i {
    let base = _mm512_add_epi16(signed_ldiff, left);
    let tldiff = _mm512_abs_epi16(_mm512_sub_epi16(tl, base));
    let tdiff = _mm512_abs_epi16(_mm512_sub_epi16(base, top));
    let mask_tl_lt_t = _mm512_cmpgt_epi16_mask(tdiff, tldiff);
    let min_t_tl = _mm512_mask_blend_epi16(mask_tl_lt_t, tdiff, tldiff);
    let mask_t = _mm512_cmpeq_epi16_mask(tdiff, min_t_tl);
    let t_or_tl = _mm512_mask_blend_epi16(mask_t, tl, top);
    let mask_not_l = _mm512_cmpgt_epi16_mask(ldiff, min_t_tl);
    _mm512_mask_blend_epi16(mask_not_l, left, t_or_tl)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_paeth_16bpc_avx512_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let tl_idx = edge.topleft;
    let tl_px = read_px_16(edge.bytes, tl_idx);
    if width == 4 {
        let tl_v = _mm_set1_epi16(tl_px);
        let top = load_u8x8(mem::chunk::<u8, 8>(edge.bytes, tl_idx + 2));
        let signed_ldiff = _mm_sub_epi16(top, tl_v);
        let ldiff = _mm_abs_epi16(signed_ldiff);
        for y in 0..height {
            let left_v = _mm_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
            let res = paeth_vec_128_avx512(top, left_v, tl_v, signed_ldiff, ldiff);
            mem::store_u8x8(mem::chunk_mut(dst, y * dst_stride), res);
        }
    } else if width == 8 {
        let tl_v = _mm_set1_epi16(tl_px);
        let top = mem::load_u8x16(mem::chunk::<u8, 16>(edge.bytes, tl_idx + 2));
        let signed_ldiff = _mm_sub_epi16(top, tl_v);
        let ldiff = _mm_abs_epi16(signed_ldiff);
        for y in 0..height {
            let left_v = _mm_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
            let res = paeth_vec_128_avx512(top, left_v, tl_v, signed_ldiff, ldiff);
            mem::store_u8x16(mem::chunk_mut(dst, y * dst_stride), res);
        }
    } else if width == 16 {
        let tl_v = _mm256_set1_epi16(tl_px);
        let top = mem::load_u8x32(mem::chunk::<u8, 32>(edge.bytes, tl_idx + 2));
        let signed_ldiff = _mm256_sub_epi16(top, tl_v);
        let ldiff = _mm256_abs_epi16(signed_ldiff);
        for y in 0..height {
            let left_v = _mm256_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
            let res = paeth_vec_256_avx512(top, left_v, tl_v, signed_ldiff, ldiff);
            mem::store_u8x32(mem::chunk_mut(dst, y * dst_stride), res);
        }
    } else {
        let tl_v = _mm512_set1_epi16(tl_px);
        for x in (0..width).step_by(32) {
            let top = mem::load_u8x64(mem::chunk::<u8, 64>(edge.bytes, tl_idx + 2 + x * 2));
            let signed_ldiff = _mm512_sub_epi16(top, tl_v);
            let ldiff = _mm512_abs_epi16(signed_ldiff);
            for y in 0..height {
                let left_v = _mm512_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
                let res = paeth_vec_512_avx512(top, left_v, tl_v, signed_ldiff, ldiff);
                mem::store_u8x64(mem::chunk_mut(dst, y * dst_stride + x * 2), res);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SMOOTH_V / SMOOTH_H / SMOOTH
// ---------------------------------------------------------------------------

fn ipred_smooth_v_16bpc_avx512(
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
    if !angular_kontrol_16(
        IntraPredMode::SmoothV,
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
    ) {
        return;
    }
    // SAFETY: (1) ISA: işlev yalnız `Avx512IclToken` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { ipred_smooth_v_16bpc_avx512_giris(dst, dst_stride, edge, width, height) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_smooth_v_16bpc_avx512_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let tl_idx = edge.topleft;
    let bottom_px = read_px_16(edge.bytes, tl_idx - height * 2);
    let weights_ver = &SM_WEIGHTS[height..height + height];
    if width == 4 {
        let bottom = _mm_set1_epi16(bottom_px);
        let top = load_u8x8(mem::chunk::<u8, 8>(edge.bytes, tl_idx + 2));
        let diff = _mm_sub_epi16(top, bottom);
        for y in 0..height {
            let wv = _mm_set1_epi16(i16::from(weights_ver[y]) << 7);
            let res = _mm_add_epi16(_mm_mulhrs_epi16(diff, wv), bottom);
            mem::store_u8x8(mem::chunk_mut(dst, y * dst_stride), res);
        }
    } else if width == 8 {
        let bottom = _mm_set1_epi16(bottom_px);
        let top = mem::load_u8x16(mem::chunk::<u8, 16>(edge.bytes, tl_idx + 2));
        let diff = _mm_sub_epi16(top, bottom);
        for y in 0..height {
            let wv = _mm_set1_epi16(i16::from(weights_ver[y]) << 7);
            let res = _mm_add_epi16(_mm_mulhrs_epi16(diff, wv), bottom);
            mem::store_u8x16(mem::chunk_mut(dst, y * dst_stride), res);
        }
    } else if width == 16 {
        let bottom = _mm256_set1_epi16(bottom_px);
        let top = mem::load_u8x32(mem::chunk::<u8, 32>(edge.bytes, tl_idx + 2));
        let diff = _mm256_sub_epi16(top, bottom);
        for y in 0..height {
            let wv = _mm256_set1_epi16(i16::from(weights_ver[y]) << 7);
            let res = _mm256_add_epi16(_mm256_mulhrs_epi16(diff, wv), bottom);
            mem::store_u8x32(mem::chunk_mut(dst, y * dst_stride), res);
        }
    } else {
        let bottom = _mm512_set1_epi16(bottom_px);
        for x in (0..width).step_by(32) {
            let top = mem::load_u8x64(mem::chunk::<u8, 64>(edge.bytes, tl_idx + 2 + x * 2));
            let diff = _mm512_sub_epi16(top, bottom);
            for y in 0..height {
                let wv = _mm512_set1_epi16(i16::from(weights_ver[y]) << 7);
                let res = _mm512_add_epi16(_mm512_mulhrs_epi16(diff, wv), bottom);
                mem::store_u8x64(mem::chunk_mut(dst, y * dst_stride + x * 2), res);
            }
        }
    }
}

fn ipred_smooth_h_16bpc_avx512(
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
    if !angular_kontrol_16(
        IntraPredMode::SmoothH,
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
    ) {
        return;
    }
    // SAFETY: (1) ISA: işlev yalnız `Avx512IclToken` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { ipred_smooth_h_16bpc_avx512_giris(dst, dst_stride, edge, width, height) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn sm_weights_1d_8_avx512(w_slice: &[u8], offset: usize) -> __m128i {
    _mm_setr_epi16(
        i16::from(w_slice[offset]) << 7,
        i16::from(w_slice[offset + 1]) << 7,
        i16::from(w_slice[offset + 2]) << 7,
        i16::from(w_slice[offset + 3]) << 7,
        i16::from(w_slice[offset + 4]) << 7,
        i16::from(w_slice[offset + 5]) << 7,
        i16::from(w_slice[offset + 6]) << 7,
        i16::from(w_slice[offset + 7]) << 7,
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn sm_weights_1d_16_avx512(w_slice: &[u8], offset: usize) -> __m256i {
    _mm256_setr_epi16(
        i16::from(w_slice[offset]) << 7,
        i16::from(w_slice[offset + 1]) << 7,
        i16::from(w_slice[offset + 2]) << 7,
        i16::from(w_slice[offset + 3]) << 7,
        i16::from(w_slice[offset + 4]) << 7,
        i16::from(w_slice[offset + 5]) << 7,
        i16::from(w_slice[offset + 6]) << 7,
        i16::from(w_slice[offset + 7]) << 7,
        i16::from(w_slice[offset + 8]) << 7,
        i16::from(w_slice[offset + 9]) << 7,
        i16::from(w_slice[offset + 10]) << 7,
        i16::from(w_slice[offset + 11]) << 7,
        i16::from(w_slice[offset + 12]) << 7,
        i16::from(w_slice[offset + 13]) << 7,
        i16::from(w_slice[offset + 14]) << 7,
        i16::from(w_slice[offset + 15]) << 7,
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn sm_weights_1d_32_avx512(w_slice: &[u8], offset: usize) -> __m512i {
    #[inline(always)]
    fn pack2(w: &[u8], i: usize) -> i32 {
        (((i32::from(w[i + 1]) << 7) & 0xFFFF) << 16) | ((i32::from(w[i]) << 7) & 0xFFFF)
    }
    _mm512_setr_epi32(
        pack2(w_slice, offset),
        pack2(w_slice, offset + 2),
        pack2(w_slice, offset + 4),
        pack2(w_slice, offset + 6),
        pack2(w_slice, offset + 8),
        pack2(w_slice, offset + 10),
        pack2(w_slice, offset + 12),
        pack2(w_slice, offset + 14),
        pack2(w_slice, offset + 16),
        pack2(w_slice, offset + 18),
        pack2(w_slice, offset + 20),
        pack2(w_slice, offset + 22),
        pack2(w_slice, offset + 24),
        pack2(w_slice, offset + 26),
        pack2(w_slice, offset + 28),
        pack2(w_slice, offset + 30),
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_smooth_h_16bpc_avx512_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let tl_idx = edge.topleft;
    let right_px = read_px_16(edge.bytes, tl_idx + width * 2);
    let weights_hor = &SM_WEIGHTS[width..width + width];
    if width == 4 {
        let right = _mm_set1_epi16(right_px);
        let wh = _mm_setr_epi16(
            i16::from(weights_hor[0]) << 7,
            i16::from(weights_hor[1]) << 7,
            i16::from(weights_hor[2]) << 7,
            i16::from(weights_hor[3]) << 7,
            0,
            0,
            0,
            0,
        );
        for y in 0..height {
            let left = _mm_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
            let diff = _mm_sub_epi16(left, right);
            let res = _mm_add_epi16(_mm_mulhrs_epi16(diff, wh), right);
            mem::store_u8x8(mem::chunk_mut(dst, y * dst_stride), res);
        }
    } else if width == 8 {
        let right = _mm_set1_epi16(right_px);
        let wh = sm_weights_1d_8_avx512(weights_hor, 0);
        for y in 0..height {
            let left = _mm_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
            let diff = _mm_sub_epi16(left, right);
            let res = _mm_add_epi16(_mm_mulhrs_epi16(diff, wh), right);
            mem::store_u8x16(mem::chunk_mut(dst, y * dst_stride), res);
        }
    } else if width == 16 {
        let right = _mm256_set1_epi16(right_px);
        let wh = sm_weights_1d_16_avx512(weights_hor, 0);
        for y in 0..height {
            let left = _mm256_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
            let diff = _mm256_sub_epi16(left, right);
            let res = _mm256_add_epi16(_mm256_mulhrs_epi16(diff, wh), right);
            mem::store_u8x32(mem::chunk_mut(dst, y * dst_stride), res);
        }
    } else {
        let right = _mm512_set1_epi16(right_px);
        for x in (0..width).step_by(32) {
            let wh = sm_weights_1d_32_avx512(weights_hor, x);
            for y in 0..height {
                let left = _mm512_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
                let diff = _mm512_sub_epi16(left, right);
                let res = _mm512_add_epi16(_mm512_mulhrs_epi16(diff, wh), right);
                mem::store_u8x64(mem::chunk_mut(dst, y * dst_stride + x * 2), res);
            }
        }
    }
}

fn ipred_smooth_16bpc_avx512(
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
    if !angular_kontrol_16(
        IntraPredMode::Smooth,
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
    ) {
        return;
    }
    // SAFETY: (1) ISA: işlev yalnız `Avx512IclToken` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { ipred_smooth_16bpc_avx512_giris(dst, dst_stride, edge, width, height) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn sm_weights_2d_4_avx512(w_slice: &[u8], offset: usize) -> __m128i {
    let w0 = i16::from(w_slice[offset]);
    let w1 = i16::from(w_slice[offset + 1]);
    let w2 = i16::from(w_slice[offset + 2]);
    let w3 = i16::from(w_slice[offset + 3]);
    _mm_setr_epi16(w0, 256 - w0, w1, 256 - w1, w2, 256 - w2, w3, 256 - w3)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_smooth_16bpc_avx512_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let tl_idx = edge.topleft;
    let right_px = read_px_16(edge.bytes, tl_idx + width * 2);
    let bottom_px = read_px_16(edge.bytes, tl_idx - height * 2);
    let weights_hor = &SM_WEIGHTS[width..width + width];
    let weights_ver = &SM_WEIGHTS[height..height + height];

    if width == 4 {
        let right_v = _mm_set1_epi16(right_px);
        let bottom_v = _mm_set1_epi16(bottom_px);
        let rnd = _mm_set1_epi32(256);
        let top = load_u8x8(mem::chunk::<u8, 8>(edge.bytes, tl_idx + 2));
        let top_bottom = _mm_unpacklo_epi16(top, bottom_v);
        let wh = sm_weights_2d_4_avx512(weights_hor, 0);
        for y in 0..height {
            let wv = i16::from(weights_ver[y]);
            let wv_pair = _mm_set1_epi32((i32::from(256 - wv) << 16) | (i32::from(wv) & 0xFFFF));
            let left_v = _mm_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
            let left_right = _mm_unpacklo_epi16(left_v, right_v);
            let sum = _mm_dpwssd_epi32(_mm_dpwssd_epi32(rnd, top_bottom, wv_pair), left_right, wh);
            let out32 = _mm_srai_epi32::<9>(sum);
            let out16 = _mm_packs_epi32(out32, out32);
            mem::store_u8x8(mem::chunk_mut(dst, y * dst_stride), out16);
        }
    } else if width == 8 {
        let right_v = _mm_set1_epi16(right_px);
        let bottom_v = _mm_set1_epi16(bottom_px);
        let rnd = _mm_set1_epi32(256);
        let top = mem::load_u8x16(mem::chunk::<u8, 16>(edge.bytes, tl_idx + 2));
        let tb_lo = _mm_unpacklo_epi16(top, bottom_v);
        let tb_hi = _mm_unpackhi_epi16(top, bottom_v);
        let wh_lo = sm_weights_2d_4_avx512(weights_hor, 0);
        let wh_hi = sm_weights_2d_4_avx512(weights_hor, 4);
        for y in 0..height {
            let wv = i16::from(weights_ver[y]);
            let wv_pair = _mm_set1_epi32((i32::from(256 - wv) << 16) | (i32::from(wv) & 0xFFFF));
            let left_v = _mm_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
            let left_right = _mm_unpacklo_epi16(left_v, right_v);
            let sum_lo = _mm_dpwssd_epi32(_mm_dpwssd_epi32(rnd, tb_lo, wv_pair), left_right, wh_lo);
            let sum_hi = _mm_dpwssd_epi32(_mm_dpwssd_epi32(rnd, tb_hi, wv_pair), left_right, wh_hi);
            let out_lo = _mm_srai_epi32::<9>(sum_lo);
            let out_hi = _mm_srai_epi32::<9>(sum_hi);
            let out16 = _mm_packs_epi32(out_lo, out_hi);
            mem::store_u8x16(mem::chunk_mut(dst, y * dst_stride), out16);
        }
    } else if width == 16 {
        let right_v = _mm256_set1_epi16(right_px);
        let bottom_v = _mm256_set1_epi16(bottom_px);
        let rnd = _mm256_set1_epi32(256);
        let c256 = _mm256_set1_epi16(256);
        let top = mem::load_u8x32(mem::chunk::<u8, 32>(edge.bytes, tl_idx + 2));
        let tb_lo = _mm256_unpacklo_epi16(top, bottom_v);
        let tb_hi = _mm256_unpackhi_epi16(top, bottom_v);
        let wh_raw = _mm256_srli_epi16::<7>(sm_weights_1d_16_avx512(weights_hor, 0));
        let inv_wh = _mm256_sub_epi16(c256, wh_raw);
        let wh_lo = _mm256_unpacklo_epi16(wh_raw, inv_wh);
        let wh_hi = _mm256_unpackhi_epi16(wh_raw, inv_wh);
        for y in 0..height {
            let wv = i16::from(weights_ver[y]);
            let wv_pair = _mm256_set1_epi32((i32::from(256 - wv) << 16) | (i32::from(wv) & 0xFFFF));
            let left_v = _mm256_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
            let left_right = _mm256_unpacklo_epi16(left_v, right_v);
            let sum_lo =
                _mm256_dpwssd_epi32(_mm256_dpwssd_epi32(rnd, tb_lo, wv_pair), left_right, wh_lo);
            let sum_hi =
                _mm256_dpwssd_epi32(_mm256_dpwssd_epi32(rnd, tb_hi, wv_pair), left_right, wh_hi);
            let out_lo = _mm256_srai_epi32::<9>(sum_lo);
            let out_hi = _mm256_srai_epi32::<9>(sum_hi);
            let out16 = _mm256_packs_epi32(out_lo, out_hi);
            mem::store_u8x32(mem::chunk_mut(dst, y * dst_stride), out16);
        }
    } else {
        let right_v = _mm512_set1_epi16(right_px);
        let bottom_v = _mm512_set1_epi16(bottom_px);
        let rnd = _mm512_set1_epi32(256);
        let c256 = _mm512_set1_epi16(256);
        for x in (0..width).step_by(32) {
            let top = mem::load_u8x64(mem::chunk::<u8, 64>(edge.bytes, tl_idx + 2 + x * 2));
            let tb_lo = _mm512_unpacklo_epi16(top, bottom_v);
            let tb_hi = _mm512_unpackhi_epi16(top, bottom_v);
            let wh_raw = _mm512_srli_epi16::<7>(sm_weights_1d_32_avx512(weights_hor, x));
            let inv_wh = _mm512_sub_epi16(c256, wh_raw);
            let wh_lo = _mm512_unpacklo_epi16(wh_raw, inv_wh);
            let wh_hi = _mm512_unpackhi_epi16(wh_raw, inv_wh);
            for y in 0..height {
                let wv = i16::from(weights_ver[y]);
                let wv_pair =
                    _mm512_set1_epi32((i32::from(256 - wv) << 16) | (i32::from(wv) & 0xFFFF));
                let left_v = _mm512_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
                let left_right = _mm512_unpacklo_epi16(left_v, right_v);
                let sum_lo = _mm512_dpwssd_epi32(
                    _mm512_dpwssd_epi32(rnd, tb_lo, wv_pair),
                    left_right,
                    wh_lo,
                );
                let sum_hi = _mm512_dpwssd_epi32(
                    _mm512_dpwssd_epi32(rnd, tb_hi, wv_pair),
                    left_right,
                    wh_hi,
                );
                let out_lo = _mm512_srai_epi32::<9>(sum_lo);
                let out_hi = _mm512_srai_epi32::<9>(sum_hi);
                let out16 = _mm512_packs_epi32(out_lo, out_hi);
                mem::store_u8x64(mem::chunk_mut(dst, y * dst_stride + x * 2), out16);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Z1 / Z2 / Z3
// ---------------------------------------------------------------------------

fn ipred_z1_16bpc_avx512(
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
    if !angular_kontrol_16(
        IntraPredMode::Z1,
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
    ) {
        return;
    }
    // SAFETY: (1) ISA: işlev yalnız `Avx512IclToken` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe {
        ipred_z1_16bpc_avx512_giris(dst, dst_stride, edge, width, height, angle, bitdepth_max)
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_z1_16bpc_avx512_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
    bitdepth_max: u16,
) {
    let mut top_buf = [0i16; 320];
    let (dx, base_inc, max_base_x) =
        z1_hazirla_16(&mut top_buf, edge, width, height, angle, bitdepth_max);
    let fill = top_buf[max_base_x];
    let mut xpos = dx;

    if base_inc == 1 {
        for y in 0..height {
            let base = (xpos >> 6) as usize;
            if base >= max_base_x {
                splat_dc_avx512(
                    &mut dst[y * dst_stride..],
                    dst_stride,
                    width,
                    height - y,
                    fill,
                );
                break;
            }
            if width == 4 {
                let vf = _mm_set1_epi16(((xpos & 0x3E) as i16) << 9);
                let v0 = mem::load_i16x8(mem::chunk(&top_buf, base));
                let v1 = mem::load_i16x8(mem::chunk(&top_buf, base + 1));
                let out = _mm_add_epi16(v0, _mm_mulhrs_epi16(_mm_sub_epi16(v1, v0), vf));
                mem::store_u8x8(mem::chunk_mut(dst, y * dst_stride), out);
            } else if width == 8 {
                let vf = _mm_set1_epi16(((xpos & 0x3E) as i16) << 9);
                let v0 = mem::load_i16x8(mem::chunk(&top_buf, base));
                let v1 = mem::load_i16x8(mem::chunk(&top_buf, base + 1));
                let out = _mm_add_epi16(v0, _mm_mulhrs_epi16(_mm_sub_epi16(v1, v0), vf));
                mem::store_u8x16(mem::chunk_mut(dst, y * dst_stride), out);
            } else if width == 16 {
                let vf = _mm256_set1_epi16(((xpos & 0x3E) as i16) << 9);
                let v0 = mem::load_i16x16(mem::chunk(&top_buf, base));
                let v1 = mem::load_i16x16(mem::chunk(&top_buf, base + 1));
                let out = _mm256_add_epi16(v0, _mm256_mulhrs_epi16(_mm256_sub_epi16(v1, v0), vf));
                mem::store_u8x32(mem::chunk_mut(dst, y * dst_stride), out);
            } else {
                let vf = _mm512_set1_epi16(((xpos & 0x3E) as i16) << 9);
                for x in (0..width).step_by(32) {
                    let v0 = mem::load_i16x32(mem::chunk(&top_buf, base + x));
                    let v1 = mem::load_i16x32(mem::chunk(&top_buf, base + x + 1));
                    let out =
                        _mm512_add_epi16(v0, _mm512_mulhrs_epi16(_mm512_sub_epi16(v1, v0), vf));
                    mem::store_u8x64(mem::chunk_mut(dst, y * dst_stride + x * 2), out);
                }
            }
            xpos += dx;
        }
    } else {
        let shuf = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
        for y in 0..height {
            let base = (xpos >> 6) as usize;
            if base >= max_base_x {
                splat_dc_avx512(
                    &mut dst[y * dst_stride..],
                    dst_stride,
                    width,
                    height - y,
                    fill,
                );
                break;
            }
            let vf = _mm_set1_epi16(((xpos & 0x3E) as i16) << 9);
            if width == 4 {
                let raw = _mm_shuffle_epi8(mem::load_i16x8(mem::chunk(&top_buf, base)), shuf);
                let v0 = raw;
                let v1 = _mm_bsrli_si128::<8>(raw);
                let out = _mm_add_epi16(v0, _mm_mulhrs_epi16(_mm_sub_epi16(v1, v0), vf));
                mem::store_u8x8(mem::chunk_mut(dst, y * dst_stride), out);
            } else {
                for x in (0..width).step_by(8) {
                    let raw0 =
                        _mm_shuffle_epi8(mem::load_i16x8(mem::chunk(&top_buf, base + x * 2)), shuf);
                    let raw1 = _mm_shuffle_epi8(
                        mem::load_i16x8(mem::chunk(&top_buf, base + x * 2 + 8)),
                        shuf,
                    );
                    let v0 = _mm_unpacklo_epi64(raw0, raw1);
                    let v1 = _mm_unpackhi_epi64(raw0, raw1);
                    let out = _mm_add_epi16(v0, _mm_mulhrs_epi16(_mm_sub_epi16(v1, v0), vf));
                    mem::store_u8x16(mem::chunk_mut(dst, y * dst_stride + x * 2), out);
                }
            }
            xpos += dx;
        }
    }
}

fn ipred_z2_16bpc_avx512(
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
    if !angular_kontrol_16(
        IntraPredMode::Z2,
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
    ) {
        return;
    }
    // SAFETY: (1) ISA: işlev yalnız `Avx512IclToken` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe {
        ipred_z2_16bpc_avx512_giris(
            dst,
            dst_stride,
            edge,
            width,
            height,
            angle,
            max_width,
            max_height,
            bitdepth_max,
        )
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_z2_16bpc_avx512_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
    max_width: usize,
    max_height: usize,
    bitdepth_max: u16,
) {
    let z = z2_hazirla_16(
        edge,
        width,
        height,
        angle,
        max_width,
        max_height,
        bitdepth_max,
    );
    let base = TMP_BASE;
    let base_inc_x = 1 + isize::from(z.upsample_above);
    let y_step = 1usize << u32::from(z.upsample_left);
    let left = base - (1 + isize::from(z.upsample_left));
    let mut xpos = ((1 + isize::from(z.upsample_above)) << 6) - z.dx;
    let shuf = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);

    let mut left_idx0 = [0usize; 64];
    let mut left_af = [0i16; 72];
    let mut yp0 = -z.dy;
    for col in 0..width {
        left_idx0[col] = (left - (yp0 >> 6)) as usize;
        left_af[col] = ((yp0 & 0x3E) as i16) << 9;
        yp0 -= z.dy;
    }

    for y in 0..height {
        let base_x0 = xpos >> 6;
        let fx = ((xpos & 0x3E) as i16) << 9;
        let vf_x = _mm_set1_epi16(fx);
        let vf_x256 = _mm256_set1_epi16(fx);
        let y_off = y * y_step;
        let row = y * dst_stride;

        let mut x = 0;
        while x + 16 <= width && base_x0 + (x as isize) * base_inc_x >= 0 && base_inc_x == 1 {
            let idx = (base + base_x0 + x as isize) as usize;
            let v0 = mem::load_i16x16(mem::chunk(&z.buffer, idx));
            let v1 = mem::load_i16x16(mem::chunk(&z.buffer, idx + 1));
            let out = _mm256_add_epi16(v0, _mm256_mulhrs_epi16(_mm256_sub_epi16(v1, v0), vf_x256));
            mem::store_u8x32(mem::chunk_mut(dst, row + x * 2), out);
            x += 16;
        }
        while x + 8 <= width && base_x0 + (x as isize) * base_inc_x >= 0 && base_inc_x == 1 {
            let idx = (base + base_x0 + x as isize) as usize;
            let v0 = mem::load_i16x8(mem::chunk(&z.buffer, idx));
            let v1 = mem::load_i16x8(mem::chunk(&z.buffer, idx + 1));
            let out = _mm_add_epi16(v0, _mm_mulhrs_epi16(_mm_sub_epi16(v1, v0), vf_x));
            mem::store_u8x16(mem::chunk_mut(dst, row + x * 2), out);
            x += 8;
        }
        while x < width {
            let bx0 = base_x0 + (x as isize) * base_inc_x;
            let bx3 = bx0 + 3 * base_inc_x;
            let out = if bx0 >= 0 {
                let idx = (base + bx0) as usize;
                let (v0, v1) = if base_inc_x == 1 {
                    (
                        mem::load_i16x8(mem::chunk(&z.buffer, idx)),
                        mem::load_i16x8(mem::chunk(&z.buffer, idx + 1)),
                    )
                } else {
                    let raw = _mm_shuffle_epi8(mem::load_i16x8(mem::chunk(&z.buffer, idx)), shuf);
                    (raw, _mm_bsrli_si128::<8>(raw))
                };
                _mm_add_epi16(v0, _mm_mulhrs_epi16(_mm_sub_epi16(v1, v0), vf_x))
            } else if bx3 < 0 {
                let l0 = left_idx0[x] - y_off;
                let l1 = left_idx0[x + 1] - y_off;
                let l2 = left_idx0[x + 2] - y_off;
                let l3 = left_idx0[x + 3] - y_off;
                let v0 = _mm_setr_epi16(
                    z.buffer[l0],
                    z.buffer[l1],
                    z.buffer[l2],
                    z.buffer[l3],
                    0,
                    0,
                    0,
                    0,
                );
                let v1 = _mm_setr_epi16(
                    z.buffer[l0 - 1],
                    z.buffer[l1 - 1],
                    z.buffer[l2 - 1],
                    z.buffer[l3 - 1],
                    0,
                    0,
                    0,
                    0,
                );
                let vf = mem::load_i16x8(mem::chunk(&left_af, x));
                _mm_add_epi16(v0, _mm_mulhrs_epi16(_mm_sub_epi16(v1, v0), vf))
            } else {
                let mut a0 = [0i16; 8];
                let mut a1 = [0i16; 8];
                let mut af = [0i16; 8];
                for k in 0..4 {
                    let bx = bx0 + (k as isize) * base_inc_x;
                    if bx >= 0 {
                        let ti = (base + bx) as usize;
                        a0[k] = z.buffer[ti];
                        a1[k] = z.buffer[ti + 1];
                        af[k] = fx;
                    } else {
                        let li = left_idx0[x + k] - y_off;
                        a0[k] = z.buffer[li];
                        a1[k] = z.buffer[li - 1];
                        af[k] = left_af[x + k];
                    }
                }
                let v0 = mem::load_i16x8(&a0);
                let v1 = mem::load_i16x8(&a1);
                let vf = mem::load_i16x8(&af);
                _mm_add_epi16(v0, _mm_mulhrs_epi16(_mm_sub_epi16(v1, v0), vf))
            };
            mem::store_u8x8(mem::chunk_mut(dst, row + x * 2), out);
            x += 4;
        }
        xpos -= z.dx;
    }
}

fn ipred_z3_16bpc_avx512(
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
    if !angular_kontrol_16(
        IntraPredMode::Z3,
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
    ) {
        return;
    }
    // SAFETY: (1) ISA: işlev yalnız `Avx512IclToken` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe {
        ipred_z3_16bpc_avx512_giris(dst, dst_stride, edge, width, height, angle, bitdepth_max)
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn z3_col4_avx512(col_buf: &[i16; 320], base: usize, base_inc: usize, vf: __m128i) -> __m128i {
    if base_inc == 1 {
        let v0 = mem::load_i16x8(mem::chunk(col_buf, base));
        let v1 = mem::load_i16x8(mem::chunk(col_buf, base + 1));
        _mm_add_epi16(v0, _mm_mulhrs_epi16(_mm_sub_epi16(v1, v0), vf))
    } else {
        let shuf = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
        let raw = _mm_shuffle_epi8(mem::load_i16x8(mem::chunk(col_buf, base)), shuf);
        let v0 = raw;
        let v1 = _mm_bsrli_si128::<8>(raw);
        _mm_add_epi16(v0, _mm_mulhrs_epi16(_mm_sub_epi16(v1, v0), vf))
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn z3_col16_avx512(col_buf: &[i16; 320], base: usize, vf: __m256i) -> __m256i {
    let v0 = mem::load_i16x16(mem::chunk(col_buf, base));
    let v1 = mem::load_i16x16(mem::chunk(col_buf, base + 1));
    _mm256_add_epi16(v0, _mm256_mulhrs_epi16(_mm256_sub_epi16(v1, v0), vf))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn transpose_store_8x4_i16_avx512(
    c0: __m128i,
    c1: __m128i,
    c2: __m128i,
    c3: __m128i,
    dst: &mut [u8],
    dst_stride: usize,
    y: usize,
    x_bytes: usize,
) {
    let t0_lo = _mm_unpacklo_epi16(c0, c1);
    let t1_lo = _mm_unpacklo_epi16(c2, c3);
    let r01 = _mm_unpacklo_epi32(t0_lo, t1_lo);
    let r23 = _mm_unpackhi_epi32(t0_lo, t1_lo);

    let t0_hi = _mm_unpackhi_epi16(c0, c1);
    let t1_hi = _mm_unpackhi_epi16(c2, c3);
    let r45 = _mm_unpacklo_epi32(t0_hi, t1_hi);
    let r67 = _mm_unpackhi_epi32(t0_hi, t1_hi);

    mem::store_u8x8(mem::chunk_mut(dst, y * dst_stride + x_bytes), r01);
    mem::store_u8x8(
        mem::chunk_mut(dst, (y + 1) * dst_stride + x_bytes),
        _mm_bsrli_si128::<8>(r01),
    );
    mem::store_u8x8(mem::chunk_mut(dst, (y + 2) * dst_stride + x_bytes), r23);
    mem::store_u8x8(
        mem::chunk_mut(dst, (y + 3) * dst_stride + x_bytes),
        _mm_bsrli_si128::<8>(r23),
    );
    mem::store_u8x8(mem::chunk_mut(dst, (y + 4) * dst_stride + x_bytes), r45);
    mem::store_u8x8(
        mem::chunk_mut(dst, (y + 5) * dst_stride + x_bytes),
        _mm_bsrli_si128::<8>(r45),
    );
    mem::store_u8x8(mem::chunk_mut(dst, (y + 6) * dst_stride + x_bytes), r67);
    mem::store_u8x8(
        mem::chunk_mut(dst, (y + 7) * dst_stride + x_bytes),
        _mm_bsrli_si128::<8>(r67),
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_z3_16bpc_avx512_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
    bitdepth_max: u16,
) {
    let mut col_buf = [0i16; 320];
    let (dy, base_inc, max_base_y) =
        z3_hazirla_16(&mut col_buf, edge, width, height, angle, bitdepth_max);

    for x in (0..width).step_by(4) {
        let yp0 = (x as i32 + 1) * dy;
        let yp1 = (x as i32 + 2) * dy;
        let yp2 = (x as i32 + 3) * dy;
        let yp3 = (x as i32 + 4) * dy;
        let b0 = ((yp0 >> 6) as usize).min(max_base_y);
        let b1 = ((yp1 >> 6) as usize).min(max_base_y);
        let b2 = ((yp2 >> 6) as usize).min(max_base_y);
        let b3 = ((yp3 >> 6) as usize).min(max_base_y);
        let f0 = ((yp0 & 0x3E) as i16) << 9;
        let f1 = ((yp1 & 0x3E) as i16) << 9;
        let f2 = ((yp2 & 0x3E) as i16) << 9;
        let f3 = ((yp3 & 0x3E) as i16) << 9;

        if base_inc == 1 && height >= 16 {
            let vf0 = _mm256_set1_epi16(f0);
            let vf1 = _mm256_set1_epi16(f1);
            let vf2 = _mm256_set1_epi16(f2);
            let vf3 = _mm256_set1_epi16(f3);
            for y in (0..height).step_by(16) {
                let c0 = z3_col16_avx512(&col_buf, (b0 + y).min(max_base_y), vf0);
                let c1 = z3_col16_avx512(&col_buf, (b1 + y).min(max_base_y), vf1);
                let c2 = z3_col16_avx512(&col_buf, (b2 + y).min(max_base_y), vf2);
                let c3 = z3_col16_avx512(&col_buf, (b3 + y).min(max_base_y), vf3);
                transpose_store_8x4_i16_avx512(
                    _mm256_castsi256_si128(c0),
                    _mm256_castsi256_si128(c1),
                    _mm256_castsi256_si128(c2),
                    _mm256_castsi256_si128(c3),
                    dst,
                    dst_stride,
                    y,
                    x * 2,
                );
                transpose_store_8x4_i16_avx512(
                    _mm256_extracti128_si256::<1>(c0),
                    _mm256_extracti128_si256::<1>(c1),
                    _mm256_extracti128_si256::<1>(c2),
                    _mm256_extracti128_si256::<1>(c3),
                    dst,
                    dst_stride,
                    y + 8,
                    x * 2,
                );
            }
        } else if base_inc == 1 && height == 8 {
            let vf0 = _mm_set1_epi16(f0);
            let vf1 = _mm_set1_epi16(f1);
            let vf2 = _mm_set1_epi16(f2);
            let vf3 = _mm_set1_epi16(f3);
            let c0 = z3_col4_avx512(&col_buf, b0, 1, vf0);
            let c1 = z3_col4_avx512(&col_buf, b1, 1, vf1);
            let c2 = z3_col4_avx512(&col_buf, b2, 1, vf2);
            let c3 = z3_col4_avx512(&col_buf, b3, 1, vf3);
            transpose_store_8x4_i16_avx512(c0, c1, c2, c3, dst, dst_stride, 0, x * 2);
        } else {
            let vf0 = _mm_set1_epi16(f0);
            let vf1 = _mm_set1_epi16(f1);
            let vf2 = _mm_set1_epi16(f2);
            let vf3 = _mm_set1_epi16(f3);
            for y in (0..height).step_by(4) {
                let step = y * base_inc;
                let c0 = z3_col4_avx512(&col_buf, (b0 + step).min(max_base_y), base_inc, vf0);
                let c1 = z3_col4_avx512(&col_buf, (b1 + step).min(max_base_y), base_inc, vf1);
                let c2 = z3_col4_avx512(&col_buf, (b2 + step).min(max_base_y), base_inc, vf2);
                let c3 = z3_col4_avx512(&col_buf, (b3 + step).min(max_base_y), base_inc, vf3);

                let t0 = _mm_unpacklo_epi16(c0, c1);
                let t1 = _mm_unpacklo_epi16(c2, c3);
                let r01 = _mm_unpacklo_epi32(t0, t1);
                let r23 = _mm_unpackhi_epi32(t0, t1);

                mem::store_u8x8(mem::chunk_mut(dst, y * dst_stride + x * 2), r01);
                mem::store_u8x8(
                    mem::chunk_mut(dst, (y + 1) * dst_stride + x * 2),
                    _mm_bsrli_si128::<8>(r01),
                );
                mem::store_u8x8(mem::chunk_mut(dst, (y + 2) * dst_stride + x * 2), r23);
                mem::store_u8x8(
                    mem::chunk_mut(dst, (y + 3) * dst_stride + x * 2),
                    _mm_bsrli_si128::<8>(r23),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FILTER_PRED
// ---------------------------------------------------------------------------

fn ipred_filter_16bpc_avx512(
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
    if !angular_kontrol_16(
        IntraPredMode::Filter,
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
    ) {
        return;
    }
    // SAFETY: (1) ISA: işlev yalnız `Avx512IclToken` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe {
        ipred_filter_16bpc_avx512_giris(
            dst,
            dst_stride,
            edge,
            width,
            height,
            angle & 511,
            bitdepth_max,
        )
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn ipred_filter_16bpc_avx512_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    filter_idx: usize,
    bitdepth_max: u16,
) {
    let taps_raw = &FILTER_INTRA_TAPS[filter_idx];
    let mut taps_u8 = [[0u8; 16]; 4];
    for g in 0..4 {
        for i in 0..16 {
            taps_u8[g][i] = taps_raw[g * 16 + i] as u8;
        }
    }
    let t01 = _mm256_cvtepi8_epi16(mem::load_u8x16(&taps_u8[0]));
    let t23 = _mm256_cvtepi8_epi16(mem::load_u8x16(&taps_u8[1]));
    let t45 = _mm256_cvtepi8_epi16(mem::load_u8x16(&taps_u8[2]));
    let t67 = _mm256_cvtepi8_epi16(mem::load_u8x16(&taps_u8[3]));

    let zero = _mm_setzero_si128();
    let max_v = _mm_set1_epi16(bitdepth_max as i16);
    let rnd = _mm256_set1_epi32(8);
    let tl_idx = edge.topleft;

    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(4) {
            let group = x / 4;
            let (p0, p1, p2, p3, p4) = if y == 0 {
                if group == 0 {
                    (
                        read_px_16(edge.bytes, tl_idx),
                        read_px_16(edge.bytes, tl_idx + 2),
                        read_px_16(edge.bytes, tl_idx + 4),
                        read_px_16(edge.bytes, tl_idx + 6),
                        read_px_16(edge.bytes, tl_idx + 8),
                    )
                } else {
                    let col = x * 2;
                    (
                        read_px_16(edge.bytes, tl_idx + col),
                        read_px_16(edge.bytes, tl_idx + col + 2),
                        read_px_16(edge.bytes, tl_idx + col + 4),
                        read_px_16(edge.bytes, tl_idx + col + 6),
                        read_px_16(edge.bytes, tl_idx + col + 8),
                    )
                }
            } else {
                let top_row = (y - 1) * dst_stride;
                let p0 = if group == 0 {
                    read_px_16(edge.bytes, tl_idx - y * 2)
                } else {
                    read_px_16(dst, top_row + (x - 1) * 2)
                };
                (
                    p0,
                    read_px_16(dst, top_row + x * 2),
                    read_px_16(dst, top_row + (x + 1) * 2),
                    read_px_16(dst, top_row + (x + 2) * 2),
                    read_px_16(dst, top_row + (x + 3) * 2),
                )
            };
            let (p5, p6) = if group == 0 {
                (
                    read_px_16(edge.bytes, tl_idx - (y + 1) * 2),
                    read_px_16(edge.bytes, tl_idx - (y + 2) * 2),
                )
            } else {
                (
                    read_px_16(dst, y * dst_stride + (x - 1) * 2),
                    read_px_16(dst, (y + 1) * dst_stride + (x - 1) * 2),
                )
            };

            let p128 = _mm_setr_epi16(p0, p1, p2, p3, p4, p5, p6, 0);
            let p256 = _mm256_broadcastsi128_si256(p128);
            let p01 = _mm256_shuffle_epi32::<0x00>(p256);
            let p23 = _mm256_shuffle_epi32::<0x55>(p256);
            let p45 = _mm256_shuffle_epi32::<0xAA>(p256);
            let p67 = _mm256_shuffle_epi32::<0xFF>(p256);

            let acc = _mm256_dpwssd_epi32(
                _mm256_dpwssd_epi32(
                    _mm256_dpwssd_epi32(_mm256_dpwssd_epi32(rnd, t01, p01), t23, p23),
                    t45,
                    p45,
                ),
                t67,
                p67,
            );
            let shifted = _mm256_srai_epi32::<4>(acc);
            let lo = _mm256_castsi256_si128(shifted);
            let hi = _mm256_extracti128_si256::<1>(shifted);
            let packed = _mm_packs_epi32(lo, hi);
            let clipped = _mm_min_epi16(_mm_max_epi16(packed, zero), max_v);

            mem::store_u8x8(mem::chunk_mut(dst, y * dst_stride + x * 2), clipped);
            mem::store_u8x8(
                mem::chunk_mut(dst, (y + 1) * dst_stride + x * 2),
                _mm_bsrli_si128::<8>(clipped),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// PAL_PRED
// ---------------------------------------------------------------------------

fn pal_pred_16bpc_avx512(
    dst: &mut [u8],
    dst_stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
    bit_depth: BitDepth,
) {
    if !pal_pred_kontrol_16(dst, dst_stride, pal, idx, w, h, bit_depth) {
        return;
    }
    // SAFETY: (1) ISA: işlev yalnız `Avx512IclToken` ile kurulur. (2) sınır:
    // `pal_pred_kontrol_16` tüm tampon sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { pal_pred_16bpc_avx512_giris(dst, dst_stride, pal, idx, w, h) }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn pal_pred_16bpc_avx512_giris(
    dst: &mut [u8],
    dst_stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
) {
    let pal128 = mem::load_u8x16(mem::chunk::<u8, 16>(pal, 0));
    let pal256 = _mm256_broadcastsi128_si256(pal128);
    let pal512 = _mm512_broadcast_i32x4(pal128);
    let unpack256 = _mm256_set1_epi64x(0x0024_0020_0004_0000);
    let unpack512 = _mm512_set1_epi64(0x0024_0020_0004_0000);

    if w == 4 {
        for y in (0..h).step_by(4) {
            let idx8 = load_u8x8(mem::chunk::<u8, 8>(idx, y * 2));
            let ext = _mm256_cvtepu8_epi32(idx8);
            let nib = _mm256_multishift_epi64_epi8(unpack256, ext);
            let px = _mm256_permutexvar_epi16(nib, pal256);
            let lo = _mm256_castsi256_si128(px);
            let hi = _mm256_extracti128_si256::<1>(px);
            mem::store_u8x8(mem::chunk_mut(dst, y * dst_stride), lo);
            mem::store_u8x8(
                mem::chunk_mut(dst, (y + 1) * dst_stride),
                _mm_bsrli_si128::<8>(lo),
            );
            mem::store_u8x8(mem::chunk_mut(dst, (y + 2) * dst_stride), hi);
            mem::store_u8x8(
                mem::chunk_mut(dst, (y + 3) * dst_stride),
                _mm_bsrli_si128::<8>(hi),
            );
        }
    } else if w == 8 {
        for y in (0..h).step_by(4) {
            let idx16 = mem::load_u8x16(mem::chunk::<u8, 16>(idx, y * 4));
            let ext = _mm512_cvtepu8_epi32(idx16);
            let nib = _mm512_multishift_epi64_epi8(unpack512, ext);
            let px = _mm512_permutexvar_epi16(nib, pal512);
            mem::store_u8x16(
                mem::chunk_mut(dst, y * dst_stride),
                _mm512_castsi512_si128(px),
            );
            mem::store_u8x16(
                mem::chunk_mut(dst, (y + 1) * dst_stride),
                _mm512_extracti32x4_epi32::<1>(px),
            );
            mem::store_u8x16(
                mem::chunk_mut(dst, (y + 2) * dst_stride),
                _mm512_extracti32x4_epi32::<2>(px),
            );
            mem::store_u8x16(
                mem::chunk_mut(dst, (y + 3) * dst_stride),
                _mm512_extracti32x4_epi32::<3>(px),
            );
        }
    } else if w == 16 {
        for y in (0..h).step_by(2) {
            let idx16 = mem::load_u8x16(mem::chunk::<u8, 16>(idx, y * 8));
            let ext = _mm512_cvtepu8_epi32(idx16);
            let nib = _mm512_multishift_epi64_epi8(unpack512, ext);
            let px = _mm512_permutexvar_epi16(nib, pal512);
            mem::store_u8x32(
                mem::chunk_mut(dst, y * dst_stride),
                _mm512_castsi512_si256(px),
            );
            mem::store_u8x32(
                mem::chunk_mut(dst, (y + 1) * dst_stride),
                _mm512_castsi512_si256(_mm512_shuffle_i32x4::<0xEE>(px, px)),
            );
        }
    } else {
        let idx_row = w / 2;
        for y in 0..h {
            for x in (0..w).step_by(32) {
                let idx16 = mem::load_u8x16(mem::chunk::<u8, 16>(idx, y * idx_row + x / 2));
                let ext = _mm512_cvtepu8_epi32(idx16);
                let nib = _mm512_multishift_epi64_epi8(unpack512, ext);
                let px = _mm512_permutexvar_epi16(nib, pal512);
                mem::store_u8x64(mem::chunk_mut(dst, y * dst_stride + x * 2), px);
            }
        }
    }
}
