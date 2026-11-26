//! AVX2 16-bpc (`10/12` bit ortak) intra tahmin çekirdekleri (`22` kayıt / `44` yuva; P45 Adım 2).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Yuva | Sembol | Kaynak |
//! |---|---|---|
//! | `intra_pred[DC_PRED]` (`D01629`) | `dav1d_ipred_dc_16bpc_avx2` | `src/x86/ipred16_avx2.asm:202-463` |
//! | `intra_pred[VERT_PRED]` (`D01679`) | `dav1d_ipred_v_16bpc_avx2` | `src/x86/ipred16_avx2.asm:479-520` |
//! | `intra_pred[HOR_PRED]` (`D01640`) | `dav1d_ipred_h_16bpc_avx2` | `src/x86/ipred16_avx2.asm:522-641` |
//! | `intra_pred[LEFT_DC_PRED]` (`D01645`) | `dav1d_ipred_dc_left_16bpc_avx2` | `src/x86/ipred16_avx2.asm:150-200` |
//! | `intra_pred[TOP_DC_PRED]` (`D01674`) | `dav1d_ipred_dc_top_16bpc_avx2` | `src/x86/ipred16_avx2.asm:110-148` |
//! | `intra_pred[DC_128_PRED]` (`D01624`) | `dav1d_ipred_dc_128_16bpc_avx2` | `src/x86/ipred16_avx2.asm:464-477` |
//! | `intra_pred[Z1_PRED]` (`D01685`) | `dav1d_ipred_z1_16bpc_avx2` | `src/x86/ipred16_avx2.asm:1118-1748` |
//! | `intra_pred[Z2_PRED]` (`D01691`) | `dav1d_ipred_z2_16bpc_avx2` | `src/x86/ipred16_avx2.asm:1750-2580` |
//! | `intra_pred[Z3_PRED]` (`D01697`) | `dav1d_ipred_z3_16bpc_avx2` | `src/x86/ipred16_avx2.asm:2582-3224` |
//! | `intra_pred[SMOOTH_PRED]` (`D01663`) | `dav1d_ipred_smooth_16bpc_avx2` | `src/x86/ipred16_avx2.asm:971-1116` |
//! | `intra_pred[SMOOTH_V_PRED]` (`D01669`) | `dav1d_ipred_smooth_v_16bpc_avx2` | `src/x86/ipred16_avx2.asm:774-869` |
//! | `intra_pred[SMOOTH_H_PRED]` (`D01657`) | `dav1d_ipred_smooth_h_16bpc_avx2` | `src/x86/ipred16_avx2.asm:871-969` |
//! | `intra_pred[PAETH_PRED]` (`D01651`) | `dav1d_ipred_paeth_16bpc_avx2` | `src/x86/ipred16_avx2.asm:643-772` |
//! | `intra_pred[FILTER_PRED]` (`D01635`) | `dav1d_ipred_filter_16bpc_avx2` | `src/x86/ipred16_avx2.asm:3226-3411` |
//! | `cfl_ac[0..2]` (`D01591`/`D01595`/`D01599`) | `dav1d_ipred_cfl_ac_{420,422,444}_16bpc_avx2` | `src/x86/ipred16_avx2.asm:3912-4475` |
//! | `cfl_pred[0..3]` (`D01607`/`D01611`/`D01615`/`D01603`) | `dav1d_ipred_cfl{,_left,_top,_128}_16bpc_avx2` | `src/x86/ipred16_avx2.asm:3413-3910` |
//! | `pal_pred` (`D01699` avx512 / `D01619` avx2) | `dav1d_pal_pred_16bpc_avx2` | `src/x86/ipred16_avx2.asm:4477-4626` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bütün SIMD işlevleri yalnız `#[target_feature(enable = "avx2")]` taşır;
//!   EVEX komutu üretilmez (`0` EVEX).
//! - Kopyasız O(1) sözleşme: girişlerde `ipred16_ssse3` içindeki ortak sözleşme
//!   denetleyicileri kullanılır; sözleşme dışı girdide skaler gövdeye devredilir.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_abs_epi16, _mm_add_epi16, _mm_add_epi32, _mm_and_si128, _mm_blendv_epi8,
    _mm_bsrli_si128, _mm_cmpeq_epi16, _mm_cmpgt_epi16, _mm_cvtsi64_si128, _mm_cvtsi128_si32,
    _mm_hadd_epi32, _mm_madd_epi16, _mm_max_epi16, _mm_min_epi16, _mm_mulhrs_epi16,
    _mm_packs_epi32, _mm_set1_epi8, _mm_set1_epi16, _mm_set1_epi32, _mm_setr_epi8, _mm_setr_epi16,
    _mm_setzero_si128, _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_sign_epi16, _mm_slli_epi16,
    _mm_srai_epi32, _mm_srli_epi16, _mm_sub_epi16, _mm_unpackhi_epi8, _mm_unpackhi_epi16,
    _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi8, _mm_unpacklo_epi16,
    _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm256_abs_epi16, _mm256_add_epi16, _mm256_add_epi32,
    _mm256_blendv_epi8, _mm256_broadcastsi128_si256, _mm256_castsi256_si128, _mm256_cmpeq_epi16,
    _mm256_cmpgt_epi16, _mm256_cvtepi8_epi16, _mm256_extracti128_si256, _mm256_madd_epi16,
    _mm256_max_epi16, _mm256_min_epi16, _mm256_mulhrs_epi16, _mm256_packs_epi32,
    _mm256_permute4x64_epi64, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi16,
    _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_shuffle_epi32, _mm256_sign_epi16,
    _mm256_slli_epi16, _mm256_srai_epi32, _mm256_srli_epi16, _mm256_sub_epi16,
    _mm256_unpackhi_epi16, _mm256_unpacklo_epi8, _mm256_unpacklo_epi16,
};

use crate::cpu::Avx2Token;
use crate::dsp::ipred::{
    AngularPredFn, CflAcFn, CflPredFn, CflPredMode, Edge, IntraPredMode, PalPredFn,
};
use crate::dsp::ipred_tables::{FILTER_INTRA_TAPS, SM_WEIGHTS};
use crate::dsp::x86::ipred16_ssse3::{
    angular_kontrol_16, cfl_ac_kontrol_16, cfl_pred_kontrol_16, pal_pred_kontrol_16, read_px_16,
    z1_hazirla_16, z2_hazirla_16, z3_hazirla_16,
};
use crate::dsp::x86::mem;
use crate::frame::BitDepth;

const TMP_BASE: isize = 128;

/// `intra_pred[mode]` 16-bpc AVX2 yuva kurucusu (`14` kip).
#[must_use]
pub const fn intra_pred_fn(_token: Avx2Token, mode: IntraPredMode) -> AngularPredFn {
    match mode {
        IntraPredMode::Dc => ipred_dc_16bpc_avx2,
        IntraPredMode::Vert => ipred_v_16bpc_avx2,
        IntraPredMode::Hor => ipred_h_16bpc_avx2,
        IntraPredMode::LeftDc => ipred_dc_left_16bpc_avx2,
        IntraPredMode::TopDc => ipred_dc_top_16bpc_avx2,
        IntraPredMode::Dc128 => ipred_dc_128_16bpc_avx2,
        IntraPredMode::Z1 => ipred_z1_16bpc_avx2,
        IntraPredMode::Z2 => ipred_z2_16bpc_avx2,
        IntraPredMode::Z3 => ipred_z3_16bpc_avx2,
        IntraPredMode::Smooth => ipred_smooth_16bpc_avx2,
        IntraPredMode::SmoothV => ipred_smooth_v_16bpc_avx2,
        IntraPredMode::SmoothH => ipred_smooth_h_16bpc_avx2,
        IntraPredMode::Paeth => ipred_paeth_16bpc_avx2,
        IntraPredMode::Filter => ipred_filter_16bpc_avx2,
    }
}

/// `cfl_ac[layout]` 16-bpc AVX2 yuva kurucusu (`0 => 420`, `1 => 422`, `2 => 444`).
#[must_use]
pub const fn cfl_ac_fn(_token: Avx2Token, layout: usize) -> Option<CflAcFn> {
    match layout {
        0 => Some(cfl_ac_420_16bpc_avx2),
        1 => Some(cfl_ac_422_16bpc_avx2),
        2 => Some(cfl_ac_444_16bpc_avx2),
        _ => None,
    }
}

/// `cfl_pred[mode]` 16-bpc AVX2 yuva kurucusu (`4` kip).
#[must_use]
pub const fn cfl_pred_fn(_token: Avx2Token, mode: CflPredMode) -> CflPredFn {
    match mode {
        CflPredMode::Dc => cfl_pred_dc_16bpc_avx2,
        CflPredMode::LeftDc => cfl_pred_left_dc_16bpc_avx2,
        CflPredMode::TopDc => cfl_pred_top_dc_16bpc_avx2,
        CflPredMode::Dc128 => cfl_pred_dc_128_16bpc_avx2,
    }
}

/// `pal_pred` 16-bpc AVX2 yuva kurucusu.
#[must_use]
pub const fn pal_pred_fn(_token: Avx2Token) -> PalPredFn {
    pal_pred_16bpc_avx2
}

#[target_feature(enable = "avx2")]
#[inline]
fn load_u8x8(src: &[u8; 8]) -> __m128i {
    _mm_cvtsi64_si128(i64::from_le_bytes(*src))
}

#[target_feature(enable = "avx2")]
#[inline]
fn sum_u16_slice_avx2(bytes: &[u8], count_px: usize) -> u32 {
    let ones128 = _mm_set1_epi16(1);
    if count_px == 4 {
        let v = load_u8x8(mem::chunk::<u8, 8>(bytes, 0));
        let s = _mm_madd_epi16(v, ones128);
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0x01>(s));
        return _mm_cvtsi128_si32(s) as u32;
    }
    if count_px == 8 {
        let v = mem::load_u8x16(mem::chunk::<u8, 16>(bytes, 0));
        let s = _mm_madd_epi16(v, ones128);
        let h = _mm_hadd_epi32(s, s);
        let h = _mm_hadd_epi32(h, h);
        return _mm_cvtsi128_si32(h) as u32;
    }
    let ones256 = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_si256();
    for c in bytes[..count_px * 2].as_chunks::<32>().0 {
        let v = mem::load_u8x32(c);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(v, ones256));
    }
    let sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256::<1>(acc),
    );
    let h = _mm_hadd_epi32(sum128, sum128);
    let h = _mm_hadd_epi32(h, h);
    _mm_cvtsi128_si32(h) as u32
}

#[target_feature(enable = "avx2")]
#[inline]
fn dc_gen_top_avx2(edge: Edge<'_>, width: usize) -> u32 {
    let top = &edge.bytes[edge.topleft + 2..edge.topleft + 2 + width * 2];
    let sum = sum_u16_slice_avx2(top, width) + (width >> 1) as u32;
    sum >> width.trailing_zeros()
}

#[target_feature(enable = "avx2")]
#[inline]
fn dc_gen_left_avx2(edge: Edge<'_>, height: usize) -> u32 {
    let left = &edge.bytes[edge.topleft - height * 2..edge.topleft];
    let sum = sum_u16_slice_avx2(left, height) + (height >> 1) as u32;
    sum >> height.trailing_zeros()
}

#[target_feature(enable = "avx2")]
#[inline]
fn dc_gen_avx2(edge: Edge<'_>, width: usize, height: usize) -> u32 {
    let top = &edge.bytes[edge.topleft + 2..edge.topleft + 2 + width * 2];
    let left = &edge.bytes[edge.topleft - height * 2..edge.topleft];
    let mut dc = sum_u16_slice_avx2(top, width)
        + sum_u16_slice_avx2(left, height)
        + ((width + height) >> 1) as u32;
    dc >>= (width + height).trailing_zeros();
    if width != height {
        let mult = if width > height * 2 || height > width * 2 {
            0x6667u64
        } else {
            0xAAABu64
        };
        dc = ((u64::from(dc) * mult) >> 17) as u32;
    }
    dc
}

#[target_feature(enable = "avx2")]
#[inline]
fn splat_dc_avx2(dst: &mut [u8], dst_stride: usize, width: usize, height: usize, dc: i16) {
    let row_bytes = width * 2;
    let dst = &mut dst[..(height - 1) * dst_stride + row_bytes];
    let mut off = 0usize;
    if width == 4 {
        let pat = (u64::from(dc as u16) * 0x0001_0001_0001_0001).to_le_bytes();
        for _ in 0..height {
            dst[off..off + 8].copy_from_slice(&pat);
            off += dst_stride;
        }
    } else if width == 8 {
        let v = _mm_set1_epi16(dc);
        for _ in 0..height {
            mem::store_u8x16(mem::chunk_mut(dst, off), v);
            off += dst_stride;
        }
    } else {
        let v = _mm256_set1_epi16(dc);
        for _ in 0..height {
            let row = &mut dst[off..off + row_bytes];
            for d in row.as_chunks_mut::<32>().0 {
                mem::store_u8x32(d, v);
            }
            off += dst_stride;
        }
    }
}

// ---------------------------------------------------------------------------
// DC / TopDC / LeftDC / DC128 / V / H
// ---------------------------------------------------------------------------

fn ipred_dc_16bpc_avx2(
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
        IntraPredMode::Dc,
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
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { ipred_dc_16bpc_avx2_giris(dst, dst_stride, edge, width, height) }
}

#[target_feature(enable = "avx2")]
fn ipred_dc_16bpc_avx2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let dc = dc_gen_avx2(edge, width, height) as i16;
    splat_dc_avx2(dst, dst_stride, width, height, dc);
}

fn ipred_dc_top_16bpc_avx2(
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
        IntraPredMode::TopDc,
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
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { ipred_dc_top_16bpc_avx2_giris(dst, dst_stride, edge, width, height) }
}

#[target_feature(enable = "avx2")]
fn ipred_dc_top_16bpc_avx2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let dc = dc_gen_top_avx2(edge, width) as i16;
    splat_dc_avx2(dst, dst_stride, width, height, dc);
}

fn ipred_dc_left_16bpc_avx2(
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
        IntraPredMode::LeftDc,
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
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { ipred_dc_left_16bpc_avx2_giris(dst, dst_stride, edge, width, height) }
}

#[target_feature(enable = "avx2")]
fn ipred_dc_left_16bpc_avx2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let dc = dc_gen_left_avx2(edge, height) as i16;
    splat_dc_avx2(dst, dst_stride, width, height, dc);
}

fn ipred_dc_128_16bpc_avx2(
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
        IntraPredMode::Dc128,
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
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { ipred_dc_128_16bpc_avx2_giris(dst, dst_stride, width, height, bitdepth_max) }
}

#[target_feature(enable = "avx2")]
fn ipred_dc_128_16bpc_avx2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    width: usize,
    height: usize,
    bitdepth_max: u16,
) {
    let dc = ((bitdepth_max + 1) >> 1) as i16;
    splat_dc_avx2(dst, dst_stride, width, height, dc);
}

fn ipred_v_16bpc_avx2(
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
        IntraPredMode::Vert,
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
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { ipred_v_16bpc_avx2_giris(dst, dst_stride, edge, width, height) }
}

#[target_feature(enable = "avx2")]
fn ipred_v_16bpc_avx2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let top = &edge.bytes[edge.topleft + 2..edge.topleft + 2 + width * 2];
    if width == 4 {
        let v = load_u8x8(mem::chunk::<u8, 8>(top, 0));
        for y in 0..height {
            mem::store_u8x8(mem::chunk_mut(dst, y * dst_stride), v);
        }
    } else if width == 8 {
        let v = mem::load_u8x16(mem::chunk::<u8, 16>(top, 0));
        for y in 0..height {
            mem::store_u8x16(mem::chunk_mut(dst, y * dst_stride), v);
        }
    } else {
        let row_bytes = width * 2;
        for y in 0..height {
            let row = &mut dst[y * dst_stride..y * dst_stride + row_bytes];
            for (s, d) in top
                .as_chunks::<32>()
                .0
                .iter()
                .zip(row.as_chunks_mut::<32>().0)
            {
                mem::store_u8x32(d, mem::load_u8x32(s));
            }
        }
    }
}

fn ipred_h_16bpc_avx2(
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
        IntraPredMode::Hor,
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
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { ipred_h_16bpc_avx2_giris(dst, dst_stride, edge, width, height) }
}

#[target_feature(enable = "avx2")]
fn ipred_h_16bpc_avx2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let tl = edge.topleft;
    let row_bytes = width * 2;
    for y in (0..height).step_by(4) {
        let l4 = load_u8x8(mem::chunk::<u8, 8>(edge.bytes, tl - (y + 4) * 2));
        let l4 = _mm_unpacklo_epi16(l4, l4);
        if width == 4 {
            mem::store_u8x8(
                mem::chunk_mut(dst, y * dst_stride),
                _mm_shuffle_epi32::<0xFF>(l4),
            );
            mem::store_u8x8(
                mem::chunk_mut(dst, (y + 1) * dst_stride),
                _mm_shuffle_epi32::<0xAA>(l4),
            );
            mem::store_u8x8(
                mem::chunk_mut(dst, (y + 2) * dst_stride),
                _mm_shuffle_epi32::<0x55>(l4),
            );
            mem::store_u8x8(
                mem::chunk_mut(dst, (y + 3) * dst_stride),
                _mm_shuffle_epi32::<0x00>(l4),
            );
        } else if width == 8 {
            mem::store_u8x16(
                mem::chunk_mut(dst, y * dst_stride),
                _mm_shuffle_epi32::<0xFF>(l4),
            );
            mem::store_u8x16(
                mem::chunk_mut(dst, (y + 1) * dst_stride),
                _mm_shuffle_epi32::<0xAA>(l4),
            );
            mem::store_u8x16(
                mem::chunk_mut(dst, (y + 2) * dst_stride),
                _mm_shuffle_epi32::<0x55>(l4),
            );
            mem::store_u8x16(
                mem::chunk_mut(dst, (y + 3) * dst_stride),
                _mm_shuffle_epi32::<0x00>(l4),
            );
        } else {
            let l4_256 = _mm256_broadcastsi128_si256(l4);
            let r0 = _mm256_shuffle_epi32::<0xFF>(l4_256);
            let r1 = _mm256_shuffle_epi32::<0xAA>(l4_256);
            let r2 = _mm256_shuffle_epi32::<0x55>(l4_256);
            let r3 = _mm256_shuffle_epi32::<0x00>(l4_256);
            for (dy, rv) in [(0, r0), (1, r1), (2, r2), (3, r3)] {
                let row = &mut dst[(y + dy) * dst_stride..(y + dy) * dst_stride + row_bytes];
                for d in row.as_chunks_mut::<32>().0 {
                    mem::store_u8x32(d, rv);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PAETH
// ---------------------------------------------------------------------------

fn ipred_paeth_16bpc_avx2(
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
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { ipred_paeth_16bpc_avx2_giris(dst, dst_stride, edge, width, height) }
}

#[target_feature(enable = "avx2")]
#[inline]
fn paeth_vec_128_avx2(
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

#[target_feature(enable = "avx2")]
#[inline]
fn paeth_vec_256_avx2(
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

#[target_feature(enable = "avx2")]
fn ipred_paeth_16bpc_avx2_giris(
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
            let res = paeth_vec_128_avx2(top, left_v, tl_v, signed_ldiff, ldiff);
            mem::store_u8x8(mem::chunk_mut(dst, y * dst_stride), res);
        }
    } else if width == 8 {
        let tl_v = _mm_set1_epi16(tl_px);
        let top = mem::load_u8x16(mem::chunk::<u8, 16>(edge.bytes, tl_idx + 2));
        let signed_ldiff = _mm_sub_epi16(top, tl_v);
        let ldiff = _mm_abs_epi16(signed_ldiff);
        for y in 0..height {
            let left_v = _mm_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
            let res = paeth_vec_128_avx2(top, left_v, tl_v, signed_ldiff, ldiff);
            mem::store_u8x16(mem::chunk_mut(dst, y * dst_stride), res);
        }
    } else {
        let tl_v = _mm256_set1_epi16(tl_px);
        for x in (0..width).step_by(16) {
            let top = mem::load_u8x32(mem::chunk::<u8, 32>(edge.bytes, tl_idx + 2 + x * 2));
            let signed_ldiff = _mm256_sub_epi16(top, tl_v);
            let ldiff = _mm256_abs_epi16(signed_ldiff);
            for y in 0..height {
                let left_v = _mm256_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
                let res = paeth_vec_256_avx2(top, left_v, tl_v, signed_ldiff, ldiff);
                mem::store_u8x32(mem::chunk_mut(dst, y * dst_stride + x * 2), res);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SMOOTH_V / SMOOTH_H / SMOOTH
// ---------------------------------------------------------------------------

fn ipred_smooth_v_16bpc_avx2(
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
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { ipred_smooth_v_16bpc_avx2_giris(dst, dst_stride, edge, width, height) }
}

#[target_feature(enable = "avx2")]
fn ipred_smooth_v_16bpc_avx2_giris(
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
    } else {
        let bottom = _mm256_set1_epi16(bottom_px);
        for x in (0..width).step_by(16) {
            let top = mem::load_u8x32(mem::chunk::<u8, 32>(edge.bytes, tl_idx + 2 + x * 2));
            let diff = _mm256_sub_epi16(top, bottom);
            for y in 0..height {
                let wv = _mm256_set1_epi16(i16::from(weights_ver[y]) << 7);
                let res = _mm256_add_epi16(_mm256_mulhrs_epi16(diff, wv), bottom);
                mem::store_u8x32(mem::chunk_mut(dst, y * dst_stride + x * 2), res);
            }
        }
    }
}

fn ipred_smooth_h_16bpc_avx2(
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
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { ipred_smooth_h_16bpc_avx2_giris(dst, dst_stride, edge, width, height) }
}

#[target_feature(enable = "avx2")]
#[inline]
fn sm_weights_1d_8_avx2(w_slice: &[u8], offset: usize) -> __m128i {
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

#[target_feature(enable = "avx2")]
#[inline]
fn sm_weights_1d_16_avx2(w_slice: &[u8], offset: usize) -> __m256i {
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

#[target_feature(enable = "avx2")]
fn ipred_smooth_h_16bpc_avx2_giris(
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
        let wh = sm_weights_1d_8_avx2(weights_hor, 0);
        for y in 0..height {
            let left = _mm_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
            let diff = _mm_sub_epi16(left, right);
            let res = _mm_add_epi16(_mm_mulhrs_epi16(diff, wh), right);
            mem::store_u8x16(mem::chunk_mut(dst, y * dst_stride), res);
        }
    } else {
        let right = _mm256_set1_epi16(right_px);
        for x in (0..width).step_by(16) {
            let wh = sm_weights_1d_16_avx2(weights_hor, x);
            for y in 0..height {
                let left = _mm256_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
                let diff = _mm256_sub_epi16(left, right);
                let res = _mm256_add_epi16(_mm256_mulhrs_epi16(diff, wh), right);
                mem::store_u8x32(mem::chunk_mut(dst, y * dst_stride + x * 2), res);
            }
        }
    }
}

fn ipred_smooth_16bpc_avx2(
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
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { ipred_smooth_16bpc_avx2_giris(dst, dst_stride, edge, width, height) }
}

#[target_feature(enable = "avx2")]
#[inline]
fn sm_weights_2d_4_avx2(w_slice: &[u8], offset: usize) -> __m128i {
    let w0 = i16::from(w_slice[offset]);
    let w1 = i16::from(w_slice[offset + 1]);
    let w2 = i16::from(w_slice[offset + 2]);
    let w3 = i16::from(w_slice[offset + 3]);
    _mm_setr_epi16(w0, 256 - w0, w1, 256 - w1, w2, 256 - w2, w3, 256 - w3)
}

#[target_feature(enable = "avx2")]
fn ipred_smooth_16bpc_avx2_giris(
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
        let wh = sm_weights_2d_4_avx2(weights_hor, 0);
        for y in 0..height {
            let wv = i16::from(weights_ver[y]);
            let wv_pair = _mm_set1_epi32((i32::from(256 - wv) << 16) | (i32::from(wv) & 0xFFFF));
            let left_v = _mm_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
            let left_right = _mm_unpacklo_epi16(left_v, right_v);
            let sum = _mm_add_epi32(
                _mm_madd_epi16(top_bottom, wv_pair),
                _mm_madd_epi16(left_right, wh),
            );
            let out32 = _mm_srai_epi32::<9>(_mm_add_epi32(sum, rnd));
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
        let wh_lo = sm_weights_2d_4_avx2(weights_hor, 0);
        let wh_hi = sm_weights_2d_4_avx2(weights_hor, 4);
        for y in 0..height {
            let wv = i16::from(weights_ver[y]);
            let wv_pair = _mm_set1_epi32((i32::from(256 - wv) << 16) | (i32::from(wv) & 0xFFFF));
            let left_v = _mm_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
            let left_right = _mm_unpacklo_epi16(left_v, right_v);
            let sum_lo = _mm_add_epi32(
                _mm_madd_epi16(tb_lo, wv_pair),
                _mm_madd_epi16(left_right, wh_lo),
            );
            let sum_hi = _mm_add_epi32(
                _mm_madd_epi16(tb_hi, wv_pair),
                _mm_madd_epi16(left_right, wh_hi),
            );
            let out_lo = _mm_srai_epi32::<9>(_mm_add_epi32(sum_lo, rnd));
            let out_hi = _mm_srai_epi32::<9>(_mm_add_epi32(sum_hi, rnd));
            let out16 = _mm_packs_epi32(out_lo, out_hi);
            mem::store_u8x16(mem::chunk_mut(dst, y * dst_stride), out16);
        }
    } else {
        let right_v = _mm256_set1_epi16(right_px);
        let bottom_v = _mm256_set1_epi16(bottom_px);
        let rnd = _mm256_set1_epi32(256);
        let c256 = _mm256_set1_epi16(256);
        for x in (0..width).step_by(16) {
            let top = mem::load_u8x32(mem::chunk::<u8, 32>(edge.bytes, tl_idx + 2 + x * 2));
            let tb_lo = _mm256_unpacklo_epi16(top, bottom_v);
            let tb_hi = _mm256_unpackhi_epi16(top, bottom_v);
            let wh_raw = _mm256_srli_epi16::<7>(sm_weights_1d_16_avx2(weights_hor, x));
            let inv_wh = _mm256_sub_epi16(c256, wh_raw);
            let wh_lo = _mm256_unpacklo_epi16(wh_raw, inv_wh);
            let wh_hi = _mm256_unpackhi_epi16(wh_raw, inv_wh);
            for y in 0..height {
                let wv = i16::from(weights_ver[y]);
                let wv_pair =
                    _mm256_set1_epi32((i32::from(256 - wv) << 16) | (i32::from(wv) & 0xFFFF));
                let left_v = _mm256_set1_epi16(read_px_16(edge.bytes, tl_idx - (y + 1) * 2));
                let left_right = _mm256_unpacklo_epi16(left_v, right_v);
                let sum_lo = _mm256_add_epi32(
                    _mm256_madd_epi16(tb_lo, wv_pair),
                    _mm256_madd_epi16(left_right, wh_lo),
                );
                let sum_hi = _mm256_add_epi32(
                    _mm256_madd_epi16(tb_hi, wv_pair),
                    _mm256_madd_epi16(left_right, wh_hi),
                );
                let out_lo = _mm256_srai_epi32::<9>(_mm256_add_epi32(sum_lo, rnd));
                let out_hi = _mm256_srai_epi32::<9>(_mm256_add_epi32(sum_hi, rnd));
                let out16 = _mm256_packs_epi32(out_lo, out_hi);
                mem::store_u8x32(mem::chunk_mut(dst, y * dst_stride + x * 2), out16);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Z1 / Z2 / Z3
// ---------------------------------------------------------------------------

fn ipred_z1_16bpc_avx2(
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
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { ipred_z1_16bpc_avx2_giris(dst, dst_stride, edge, width, height, angle, bitdepth_max) }
}

#[target_feature(enable = "avx2")]
fn ipred_z1_16bpc_avx2_giris(
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
                splat_dc_avx2(
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
            } else {
                let vf = _mm256_set1_epi16(((xpos & 0x3E) as i16) << 9);
                for x in (0..width).step_by(16) {
                    let v0 = mem::load_i16x16(mem::chunk(&top_buf, base + x));
                    let v1 = mem::load_i16x16(mem::chunk(&top_buf, base + x + 1));
                    let out =
                        _mm256_add_epi16(v0, _mm256_mulhrs_epi16(_mm256_sub_epi16(v1, v0), vf));
                    mem::store_u8x32(mem::chunk_mut(dst, y * dst_stride + x * 2), out);
                }
            }
            xpos += dx;
        }
    } else {
        let shuf = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
        for y in 0..height {
            let base = (xpos >> 6) as usize;
            if base >= max_base_x {
                splat_dc_avx2(
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

fn ipred_z2_16bpc_avx2(
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
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe {
        ipred_z2_16bpc_avx2_giris(
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

#[target_feature(enable = "avx2")]
fn ipred_z2_16bpc_avx2_giris(
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

fn ipred_z3_16bpc_avx2(
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
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { ipred_z3_16bpc_avx2_giris(dst, dst_stride, edge, width, height, angle, bitdepth_max) }
}

#[target_feature(enable = "avx2")]
#[inline]
fn z3_col4_avx2(col_buf: &[i16; 320], base: usize, base_inc: usize, vf: __m128i) -> __m128i {
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

#[target_feature(enable = "avx2")]
#[inline]
fn z3_col16_avx2(col_buf: &[i16; 320], base: usize, vf: __m256i) -> __m256i {
    let v0 = mem::load_i16x16(mem::chunk(col_buf, base));
    let v1 = mem::load_i16x16(mem::chunk(col_buf, base + 1));
    _mm256_add_epi16(v0, _mm256_mulhrs_epi16(_mm256_sub_epi16(v1, v0), vf))
}

#[target_feature(enable = "avx2")]
#[inline]
fn transpose_store_8x4_i16(
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

#[target_feature(enable = "avx2")]
fn ipred_z3_16bpc_avx2_giris(
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
                let c0 = z3_col16_avx2(&col_buf, (b0 + y).min(max_base_y), vf0);
                let c1 = z3_col16_avx2(&col_buf, (b1 + y).min(max_base_y), vf1);
                let c2 = z3_col16_avx2(&col_buf, (b2 + y).min(max_base_y), vf2);
                let c3 = z3_col16_avx2(&col_buf, (b3 + y).min(max_base_y), vf3);
                transpose_store_8x4_i16(
                    _mm256_castsi256_si128(c0),
                    _mm256_castsi256_si128(c1),
                    _mm256_castsi256_si128(c2),
                    _mm256_castsi256_si128(c3),
                    dst,
                    dst_stride,
                    y,
                    x * 2,
                );
                transpose_store_8x4_i16(
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
            let c0 = z3_col4_avx2(&col_buf, b0, 1, vf0);
            let c1 = z3_col4_avx2(&col_buf, b1, 1, vf1);
            let c2 = z3_col4_avx2(&col_buf, b2, 1, vf2);
            let c3 = z3_col4_avx2(&col_buf, b3, 1, vf3);
            transpose_store_8x4_i16(c0, c1, c2, c3, dst, dst_stride, 0, x * 2);
        } else {
            let vf0 = _mm_set1_epi16(f0);
            let vf1 = _mm_set1_epi16(f1);
            let vf2 = _mm_set1_epi16(f2);
            let vf3 = _mm_set1_epi16(f3);
            for y in (0..height).step_by(4) {
                let step = y * base_inc;
                let c0 = z3_col4_avx2(&col_buf, (b0 + step).min(max_base_y), base_inc, vf0);
                let c1 = z3_col4_avx2(&col_buf, (b1 + step).min(max_base_y), base_inc, vf1);
                let c2 = z3_col4_avx2(&col_buf, (b2 + step).min(max_base_y), base_inc, vf2);
                let c3 = z3_col4_avx2(&col_buf, (b3 + step).min(max_base_y), base_inc, vf3);

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

fn ipred_filter_16bpc_avx2(
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
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `angular_kontrol_16` tüm kenar ve hedef sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe {
        ipred_filter_16bpc_avx2_giris(
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

#[target_feature(enable = "avx2")]
fn ipred_filter_16bpc_avx2_giris(
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

            let acc = _mm256_add_epi32(
                _mm256_add_epi32(_mm256_madd_epi16(t01, p01), _mm256_madd_epi16(t23, p23)),
                _mm256_add_epi32(
                    _mm256_add_epi32(_mm256_madd_epi16(t45, p45), _mm256_madd_epi16(t67, p67)),
                    rnd,
                ),
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
// CfL AC (4:2:0, 4:2:2, 4:4:4)
// ---------------------------------------------------------------------------

fn cfl_ac_420_16bpc_avx2(
    ac: &mut [i16],
    y: &[u8],
    y_stride: usize,
    w_pad: usize,
    h_pad: usize,
    cw: usize,
    ch: usize,
    bit_depth: BitDepth,
) {
    if !cfl_ac_kontrol_16(0, ac, y, y_stride, w_pad, h_pad, cw, ch, bit_depth) {
        return;
    }
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `cfl_ac_kontrol_16` tüm tampon sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { cfl_ac_16bpc_avx2_giris::<true, true>(ac, y, y_stride, w_pad, h_pad, cw, ch) }
}

fn cfl_ac_422_16bpc_avx2(
    ac: &mut [i16],
    y: &[u8],
    y_stride: usize,
    w_pad: usize,
    h_pad: usize,
    cw: usize,
    ch: usize,
    bit_depth: BitDepth,
) {
    if !cfl_ac_kontrol_16(1, ac, y, y_stride, w_pad, h_pad, cw, ch, bit_depth) {
        return;
    }
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `cfl_ac_kontrol_16` tüm tampon sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { cfl_ac_16bpc_avx2_giris::<true, false>(ac, y, y_stride, w_pad, h_pad, cw, ch) }
}

fn cfl_ac_444_16bpc_avx2(
    ac: &mut [i16],
    y: &[u8],
    y_stride: usize,
    w_pad: usize,
    h_pad: usize,
    cw: usize,
    ch: usize,
    bit_depth: BitDepth,
) {
    if !cfl_ac_kontrol_16(2, ac, y, y_stride, w_pad, h_pad, cw, ch, bit_depth) {
        return;
    }
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `cfl_ac_kontrol_16` tüm tampon sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { cfl_ac_16bpc_avx2_giris::<false, false>(ac, y, y_stride, w_pad, h_pad, cw, ch) }
}

#[target_feature(enable = "avx2")]
fn cfl_ac_16bpc_avx2_giris<const SS_HOR: bool, const SS_VER: bool>(
    ac: &mut [i16],
    y: &[u8],
    y_stride: usize,
    w_pad: usize,
    h_pad: usize,
    cw: usize,
    ch: usize,
) {
    let visible_w = cw - 4 * w_pad;
    let visible_h = ch - 4 * h_pad;

    if SS_HOR {
        let weight128 = _mm_set1_epi16(if SS_VER { 2 } else { 4 });
        let weight256 = _mm256_set1_epi16(if SS_VER { 2 } else { 4 });
        for row in 0..visible_h {
            let luma_row = row << usize::from(SS_VER);
            let r0 = luma_row * y_stride;
            let r1 = (luma_row + 1) * y_stride;
            let mut x = 0;
            while x + 8 <= visible_w {
                let col_bytes = x * 4;
                let v0 = mem::load_u8x32(mem::chunk::<u8, 32>(y, r0 + col_bytes));
                let mut sum32 = _mm256_madd_epi16(v0, weight256);
                if SS_VER {
                    let v1 = mem::load_u8x32(mem::chunk::<u8, 32>(y, r1 + col_bytes));
                    sum32 = _mm256_add_epi32(sum32, _mm256_madd_epi16(v1, weight256));
                }
                let packed = _mm256_packs_epi32(sum32, sum32);
                let perm = _mm256_permute4x64_epi64::<0x88>(packed);
                mem::store_i16x8(
                    mem::chunk_mut(ac, row * cw + x),
                    _mm256_castsi256_si128(perm),
                );
                x += 8;
            }
            if x < visible_w {
                let col_bytes = x * 4;
                let v0 = mem::load_u8x16(mem::chunk::<u8, 16>(y, r0 + col_bytes));
                let mut sum32 = _mm_madd_epi16(v0, weight128);
                if SS_VER {
                    let v1 = mem::load_u8x16(mem::chunk::<u8, 16>(y, r1 + col_bytes));
                    sum32 = _mm_add_epi32(sum32, _mm_madd_epi16(v1, weight128));
                }
                let packed = _mm_packs_epi32(sum32, sum32);
                let mut tmp = [0i16; 8];
                mem::store_i16x8(&mut tmp, packed);
                ac[row * cw + x..row * cw + x + 4].copy_from_slice(&tmp[..4]);
            }
            if visible_w < cw {
                let last = ac[row * cw + visible_w - 1];
                for col in visible_w..cw {
                    ac[row * cw + col] = last;
                }
            }
        }
    } else {
        for row in 0..visible_h {
            let r0 = row * y_stride;
            if visible_w == 4 {
                let v0 = load_u8x8(mem::chunk::<u8, 8>(y, r0));
                let scaled = _mm_slli_epi16::<3>(v0);
                let mut tmp = [0i16; 8];
                mem::store_i16x8(&mut tmp, scaled);
                ac[row * cw..row * cw + 4].copy_from_slice(&tmp[..4]);
            } else {
                let mut x = 0;
                while x + 16 <= visible_w {
                    let v0 = mem::load_u8x32(mem::chunk::<u8, 32>(y, r0 + x * 2));
                    let scaled = _mm256_slli_epi16::<3>(v0);
                    mem::store_i16x16(mem::chunk_mut(ac, row * cw + x), scaled);
                    x += 16;
                }
                if x < visible_w {
                    let v0 = mem::load_u8x16(mem::chunk::<u8, 16>(y, r0 + x * 2));
                    let scaled = _mm_slli_epi16::<3>(v0);
                    mem::store_i16x8(mem::chunk_mut(ac, row * cw + x), scaled);
                }
            }
            if visible_w < cw {
                let last = ac[row * cw + visible_w - 1];
                for col in visible_w..cw {
                    ac[row * cw + col] = last;
                }
            }
        }
    }

    for row in visible_h..ch {
        for x in (0..cw).step_by(4) {
            for k in 0..4 {
                ac[row * cw + x + k] = ac[(visible_h - 1) * cw + x + k];
            }
        }
    }

    let total = cw * ch;
    let ones128 = _mm_set1_epi16(1);
    let mut sum128 = _mm_setzero_si128();
    for c in ac[..total].as_chunks::<8>().0 {
        let v = mem::load_i16x8(c);
        sum128 = _mm_add_epi32(sum128, _mm_madd_epi16(v, ones128));
    }
    let h = _mm_hadd_epi32(sum128, sum128);
    let h = _mm_hadd_epi32(h, h);
    let log2sz = cw.trailing_zeros() + ch.trailing_zeros();
    let dc_avg = ((_mm_cvtsi128_si32(h) + ((1i32 << log2sz) >> 1)) >> log2sz) as i16;
    let dc_v128 = _mm_set1_epi16(dc_avg);
    for c in ac[..total].as_chunks_mut::<8>().0 {
        let v = mem::load_i16x8(c);
        mem::store_i16x8(c, _mm_sub_epi16(v, dc_v128));
    }
}

// ---------------------------------------------------------------------------
// CfL PRED (DC, LEFT_DC, TOP_DC, DC_128)
// ---------------------------------------------------------------------------

fn cfl_pred_dc_16bpc_avx2(
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
    if !cfl_pred_kontrol_16(
        CflPredMode::Dc,
        dst,
        dst_stride,
        edge,
        width,
        height,
        ac,
        alpha,
        bit_depth,
        bitdepth_max,
    ) {
        return;
    }
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `cfl_pred_kontrol_16` tüm tampon sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe {
        cfl_pred_16bpc_avx2_giris(
            CflPredMode::Dc,
            dst,
            dst_stride,
            edge,
            width,
            height,
            ac,
            alpha,
            bitdepth_max,
        )
    }
}

fn cfl_pred_left_dc_16bpc_avx2(
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
    if !cfl_pred_kontrol_16(
        CflPredMode::LeftDc,
        dst,
        dst_stride,
        edge,
        width,
        height,
        ac,
        alpha,
        bit_depth,
        bitdepth_max,
    ) {
        return;
    }
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `cfl_pred_kontrol_16` tüm tampon sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe {
        cfl_pred_16bpc_avx2_giris(
            CflPredMode::LeftDc,
            dst,
            dst_stride,
            edge,
            width,
            height,
            ac,
            alpha,
            bitdepth_max,
        )
    }
}

fn cfl_pred_top_dc_16bpc_avx2(
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
    if !cfl_pred_kontrol_16(
        CflPredMode::TopDc,
        dst,
        dst_stride,
        edge,
        width,
        height,
        ac,
        alpha,
        bit_depth,
        bitdepth_max,
    ) {
        return;
    }
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `cfl_pred_kontrol_16` tüm tampon sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe {
        cfl_pred_16bpc_avx2_giris(
            CflPredMode::TopDc,
            dst,
            dst_stride,
            edge,
            width,
            height,
            ac,
            alpha,
            bitdepth_max,
        )
    }
}

fn cfl_pred_dc_128_16bpc_avx2(
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
    if !cfl_pred_kontrol_16(
        CflPredMode::Dc128,
        dst,
        dst_stride,
        edge,
        width,
        height,
        ac,
        alpha,
        bit_depth,
        bitdepth_max,
    ) {
        return;
    }
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `cfl_pred_kontrol_16` tüm tampon sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe {
        cfl_pred_16bpc_avx2_giris(
            CflPredMode::Dc128,
            dst,
            dst_stride,
            edge,
            width,
            height,
            ac,
            alpha,
            bitdepth_max,
        )
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn cfl_pred_8_avx2(
    ac_v: __m128i,
    dc_v: __m128i,
    alpha_sign: __m128i,
    alpha_mag: __m128i,
    zero: __m128i,
    max_v: __m128i,
) -> __m128i {
    let ac_clamped = _mm_max_epi16(ac_v, _mm_set1_epi16(-32767));
    let sign = _mm_sign_epi16(ac_clamped, alpha_sign);
    let abs_ac = _mm_abs_epi16(ac_clamped);
    let scaled = _mm_mulhrs_epi16(abs_ac, alpha_mag);
    let signed = _mm_sign_epi16(scaled, sign);
    let pred = _mm_add_epi16(dc_v, signed);
    _mm_min_epi16(_mm_max_epi16(pred, zero), max_v)
}

#[target_feature(enable = "avx2")]
#[inline]
fn cfl_pred_16_avx2(
    ac_v: __m256i,
    dc_v: __m256i,
    alpha_sign: __m256i,
    alpha_mag: __m256i,
    zero: __m256i,
    max_v: __m256i,
) -> __m256i {
    let ac_clamped = _mm256_max_epi16(ac_v, _mm256_set1_epi16(-32767));
    let sign = _mm256_sign_epi16(ac_clamped, alpha_sign);
    let abs_ac = _mm256_abs_epi16(ac_clamped);
    let scaled = _mm256_mulhrs_epi16(abs_ac, alpha_mag);
    let signed = _mm256_sign_epi16(scaled, sign);
    let pred = _mm256_add_epi16(dc_v, signed);
    _mm256_min_epi16(_mm256_max_epi16(pred, zero), max_v)
}

#[target_feature(enable = "avx2")]
fn cfl_pred_16bpc_avx2_giris(
    mode: CflPredMode,
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    ac: &[i16],
    alpha: i32,
    bitdepth_max: u16,
) {
    let dc = match mode {
        CflPredMode::Dc => dc_gen_avx2(edge, width, height) as i16,
        CflPredMode::LeftDc => dc_gen_left_avx2(edge, height) as i16,
        CflPredMode::TopDc => dc_gen_top_avx2(edge, width) as i16,
        CflPredMode::Dc128 => ((bitdepth_max + 1) >> 1) as i16,
    };

    if width == 4 {
        let dc_v = _mm_set1_epi16(dc);
        let alpha_sign = _mm_set1_epi16(alpha as i16);
        let alpha_mag = _mm_slli_epi16::<9>(_mm_abs_epi16(alpha_sign));
        let zero = _mm_setzero_si128();
        let max_v = _mm_set1_epi16(bitdepth_max as i16);
        for y in (0..height).step_by(2) {
            let ac_v = mem::load_i16x8(mem::chunk(ac, y * 4));
            let res = cfl_pred_8_avx2(ac_v, dc_v, alpha_sign, alpha_mag, zero, max_v);
            mem::store_u8x8(mem::chunk_mut(dst, y * dst_stride), res);
            mem::store_u8x8(
                mem::chunk_mut(dst, (y + 1) * dst_stride),
                _mm_bsrli_si128::<8>(res),
            );
        }
    } else if width == 8 {
        let dc_v = _mm_set1_epi16(dc);
        let alpha_sign = _mm_set1_epi16(alpha as i16);
        let alpha_mag = _mm_slli_epi16::<9>(_mm_abs_epi16(alpha_sign));
        let zero = _mm_setzero_si128();
        let max_v = _mm_set1_epi16(bitdepth_max as i16);
        for y in 0..height {
            let ac_v = mem::load_i16x8(mem::chunk(ac, y * 8));
            let res = cfl_pred_8_avx2(ac_v, dc_v, alpha_sign, alpha_mag, zero, max_v);
            mem::store_u8x16(mem::chunk_mut(dst, y * dst_stride), res);
        }
    } else {
        let dc_v = _mm256_set1_epi16(dc);
        let alpha_sign = _mm256_set1_epi16(alpha as i16);
        let alpha_mag = _mm256_slli_epi16::<9>(_mm256_abs_epi16(alpha_sign));
        let zero = _mm256_setzero_si256();
        let max_v = _mm256_set1_epi16(bitdepth_max as i16);
        for y in 0..height {
            for x in (0..width).step_by(16) {
                let ac_v = mem::load_i16x16(mem::chunk(ac, y * width + x));
                let res = cfl_pred_16_avx2(ac_v, dc_v, alpha_sign, alpha_mag, zero, max_v);
                mem::store_u8x32(mem::chunk_mut(dst, y * dst_stride + x * 2), res);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PAL_PRED
// ---------------------------------------------------------------------------

fn pal_pred_16bpc_avx2(
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
    // SAFETY: (1) ISA: işlev yalnız `Avx2Token` ile kurulur. (2) sınır:
    // `pal_pred_kontrol_16` tüm tampon sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: hizasız erişimler `mem` yardımcılarıyladır.
    unsafe { pal_pred_16bpc_avx2_giris(dst, dst_stride, pal, idx, w, h) }
}

#[target_feature(enable = "avx2")]
fn pal_pred_16bpc_avx2_giris(
    dst: &mut [u8],
    dst_stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
) {
    let pal_v128 = mem::load_u8x16(mem::chunk::<u8, 16>(pal, 0));
    let shuf_lo128 = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14);
    let shuf_hi128 = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15);
    let pal_lo128 = _mm_shuffle_epi8(pal_v128, shuf_lo128);
    let pal_hi128 = _mm_shuffle_epi8(pal_v128, shuf_hi128);
    let mask7_128 = _mm_set1_epi8(0x07);

    if w == 4 {
        for y in (0..h).step_by(4) {
            let idx_v = load_u8x8(mem::chunk::<u8, 8>(idx, y * 2));
            let lo_nib = _mm_and_si128(idx_v, mask7_128);
            let hi_nib = _mm_and_si128(_mm_srli_epi16::<4>(idx_v), mask7_128);
            let unpacked = _mm_unpacklo_epi8(lo_nib, hi_nib);
            let b_lo = _mm_shuffle_epi8(pal_lo128, unpacked);
            let b_hi = _mm_shuffle_epi8(pal_hi128, unpacked);
            let px01 = _mm_unpacklo_epi8(b_lo, b_hi);
            let px23 = _mm_unpackhi_epi8(b_lo, b_hi);
            mem::store_u8x8(mem::chunk_mut(dst, y * dst_stride), px01);
            mem::store_u8x8(
                mem::chunk_mut(dst, (y + 1) * dst_stride),
                _mm_bsrli_si128::<8>(px01),
            );
            mem::store_u8x8(mem::chunk_mut(dst, (y + 2) * dst_stride), px23);
            mem::store_u8x8(
                mem::chunk_mut(dst, (y + 3) * dst_stride),
                _mm_bsrli_si128::<8>(px23),
            );
        }
    } else if w == 8 {
        for y in (0..h).step_by(2) {
            let idx_v = load_u8x8(mem::chunk::<u8, 8>(idx, y * 4));
            let lo_nib = _mm_and_si128(idx_v, mask7_128);
            let hi_nib = _mm_and_si128(_mm_srli_epi16::<4>(idx_v), mask7_128);
            let unpacked = _mm_unpacklo_epi8(lo_nib, hi_nib);
            let b_lo = _mm_shuffle_epi8(pal_lo128, unpacked);
            let b_hi = _mm_shuffle_epi8(pal_hi128, unpacked);
            let px0 = _mm_unpacklo_epi8(b_lo, b_hi);
            let px1 = _mm_unpackhi_epi8(b_lo, b_hi);
            mem::store_u8x16(mem::chunk_mut(dst, y * dst_stride), px0);
            mem::store_u8x16(mem::chunk_mut(dst, (y + 1) * dst_stride), px1);
        }
    } else {
        let pal_lo256 = _mm256_broadcastsi128_si256(pal_lo128);
        let pal_hi256 = _mm256_broadcastsi128_si256(pal_hi128);
        let idx_row = w / 2;
        for y in 0..h {
            for x in (0..w).step_by(16) {
                let idx_v = load_u8x8(mem::chunk::<u8, 8>(idx, y * idx_row + x / 2));
                let lo_nib = _mm_and_si128(idx_v, mask7_128);
                let hi_nib = _mm_and_si128(_mm_srli_epi16::<4>(idx_v), mask7_128);
                let unpacked128 = _mm_unpacklo_epi8(lo_nib, hi_nib);
                let unpacked256 =
                    _mm256_permute4x64_epi64::<0x50>(_mm256_broadcastsi128_si256(unpacked128));
                let b_lo = _mm256_shuffle_epi8(pal_lo256, unpacked256);
                let b_hi = _mm256_shuffle_epi8(pal_hi256, unpacked256);
                let px = _mm256_unpacklo_epi8(b_lo, b_hi);
                mem::store_u8x32(mem::chunk_mut(dst, y * dst_stride + x * 2), px);
            }
        }
    }
}
