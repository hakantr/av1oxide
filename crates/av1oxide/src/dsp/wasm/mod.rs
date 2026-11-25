//! WebAssembly 128-bit sabit genişlikli SIMD (`simd128`, `core::arch::wasm32`)
//! DSP çekirdekleri (`F6` / `P50` — `WP-10`, `DSP-07`).
//!
//! ## Mimari ve Güvenlik Sözleşmesi (`GUV-08`, `GUV-10`, `DSP-07`)
//!
//! - `%100 #![forbid(unsafe_code)]`: `core::arch::wasm32` içindeki değer-tabanlı
//!   vektör intrinsikleri (`v128`, `i16x8_*`, `u16x8_*`, `i32x4_*`, `u32x4_*`,
//!   `u8x16_*`, `i8x16_shuffle`, `v128_bitselect`, `v128_and`, `v128_or`)
//!   kararlı Rust'ta (`rustc 1.100.0`) güvenli işlevlerdir.
//! - Sıfır yeni `unsafe` blok (`GUV-10` bütçesi değişmez).
//! - Yalnız `#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]`
//!   koşulunda derlenir; skaler WASM artefaktına (`not(target_feature = "simd128")`)
//!   hiçbir `v128` / `simd128` komutu sızmaz (`wasm-objdump -d` ile doğrulanır).
//!
//! ## Kapsanan `DspDispatch` Yuvaları (`31` yuva)
//!
//! - **MC (`18` yuva, `8 / 10 / 12` bpc):** `avg`, `w_avg`, `mask`, `blend`,
//!   `blend_v`, `blend_h`
//! - **CDEF (`12` yuva, `8 / 10 / 12` bpc):** `dir`, `fb[0]`, `fb[1]`, `fb[2]`
//! - **PAL (`1` yuva, `0` bpc):** `pal/pal_idx_finish`

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
#![forbid(unsafe_code)]

use core::arch::wasm32::{
    i8x16_shuffle, i16x8, i16x8_add, i16x8_extract_lane, i16x8_max, i16x8_min, i16x8_narrow_i32x4,
    i16x8_splat, i16x8_sub, i32x4_abs, i32x4_add, i32x4_dot_i16x8, i32x4_extend_high_i16x8,
    i32x4_extend_low_i16x8, i32x4_extract_lane, i32x4_lt, i32x4_max, i32x4_min, i32x4_mul,
    i32x4_neg, i32x4_shr, i32x4_splat, i32x4_sub, i64x2, i64x2_extract_lane, u8x16_narrow_i16x8,
    u8x16_splat, u16x8_extend_low_u8x16, u16x8_shl, u16x8_shr, u32x4_min, v128, v128_and,
    v128_bitselect, v128_or,
};

use crate::cpu::CpuFlags;
use crate::dsp::cdef::{
    CDEF_DIRECTIONS, CdefBitDepth, CdefDirResult, CdefEdgeFlags, CdefFilterParams,
    DIR_CENTER_WEIGHT, DIV_TABLE, TMP_BUF_LEN, TMP_FILL, clamped_shift, px_at, px_put, tmp_index,
    ulog2,
};
use crate::dsp::mc_tables::OBMC_MASKS;
use crate::dsp::{IsaCandidate, IsaEntry};
use crate::frame::BitDepth;

const WASM_SIMD128_KARAR: &str = "varsayılan (P50 / F6 WP-10, wasm32 simd128): %100 #![forbid(unsafe_code)] kararlı core::arch::wasm32 v128 çekirdeği; derleme zamanı +simd128 artefaktında etkin, skaler WASM artefaktına 0 sızıntı (rust_method: core::arch)";

const fn wasm_aday(family: &'static str, field: &'static str, bit_depth: u8) -> IsaCandidate {
    IsaCandidate {
        family,
        field,
        bit_depth,
        implementation: "simd128",
        requires: CpuFlags::WASM_SIMD128,
        default: true,
        decision: WASM_SIMD128_KARAR,
    }
}

/// `wasm32` (`target_feature = "simd128"`) `DspDispatch` ISA aday tablosu (`31` yuva).
pub const WASM_ISA_CANDIDATES: &[IsaCandidate] = &[
    // MC (6 işlev × 3 bit derinliği = 18 yuva)
    wasm_aday("mc", "avg", 8),
    wasm_aday("mc", "w_avg", 8),
    wasm_aday("mc", "mask", 8),
    wasm_aday("mc", "blend", 8),
    wasm_aday("mc", "blend_v", 8),
    wasm_aday("mc", "blend_h", 8),
    wasm_aday("mc", "avg", 10),
    wasm_aday("mc", "w_avg", 10),
    wasm_aday("mc", "mask", 10),
    wasm_aday("mc", "blend", 10),
    wasm_aday("mc", "blend_v", 10),
    wasm_aday("mc", "blend_h", 10),
    wasm_aday("mc", "avg", 12),
    wasm_aday("mc", "w_avg", 12),
    wasm_aday("mc", "mask", 12),
    wasm_aday("mc", "blend", 12),
    wasm_aday("mc", "blend_v", 12),
    wasm_aday("mc", "blend_h", 12),
    // CDEF (4 işlev × 3 bit derinliği = 12 yuva)
    wasm_aday("cdef", "dir", 8),
    wasm_aday("cdef", "fb[0]", 8),
    wasm_aday("cdef", "fb[1]", 8),
    wasm_aday("cdef", "fb[2]", 8),
    wasm_aday("cdef", "dir", 10),
    wasm_aday("cdef", "fb[0]", 10),
    wasm_aday("cdef", "fb[1]", 10),
    wasm_aday("cdef", "fb[2]", 10),
    wasm_aday("cdef", "dir", 12),
    wasm_aday("cdef", "fb[0]", 12),
    wasm_aday("cdef", "fb[1]", 12),
    wasm_aday("cdef", "fb[2]", 12),
    // PAL (1 yuva, bit derinliğinden bağımsız)
    wasm_aday("pal", "pal_idx_finish", 0),
];

/// `DspDispatch::candidate_entry` için `wasm32` `simd128` yuva eşlemesi.
#[must_use]
pub(super) fn candidate_entry(candidate: &IsaCandidate) -> Option<IsaEntry> {
    if candidate.implementation != "simd128" {
        return None;
    }
    match (candidate.family, candidate.field, candidate.bit_depth) {
        ("mc", "avg", 8) => Some(IsaEntry::Avg(avg::<8>)),
        ("mc", "avg", 10) => Some(IsaEntry::Avg(avg::<10>)),
        ("mc", "avg", 12) => Some(IsaEntry::Avg(avg::<12>)),
        ("mc", "w_avg", 8) => Some(IsaEntry::WAvg(w_avg::<8>)),
        ("mc", "w_avg", 10) => Some(IsaEntry::WAvg(w_avg::<10>)),
        ("mc", "w_avg", 12) => Some(IsaEntry::WAvg(w_avg::<12>)),
        ("mc", "mask", 8) => Some(IsaEntry::Mask(mask::<8>)),
        ("mc", "mask", 10) => Some(IsaEntry::Mask(mask::<10>)),
        ("mc", "mask", 12) => Some(IsaEntry::Mask(mask::<12>)),
        ("mc", "blend", 8) => Some(IsaEntry::Blend(blend::<8>)),
        ("mc", "blend", 10) => Some(IsaEntry::Blend(blend::<10>)),
        ("mc", "blend", 12) => Some(IsaEntry::Blend(blend::<12>)),
        ("mc", "blend_v", 8) => Some(IsaEntry::BlendV(blend_v::<8>)),
        ("mc", "blend_v", 10) => Some(IsaEntry::BlendV(blend_v::<10>)),
        ("mc", "blend_v", 12) => Some(IsaEntry::BlendV(blend_v::<12>)),
        ("mc", "blend_h", 8) => Some(IsaEntry::BlendH(blend_h::<8>)),
        ("mc", "blend_h", 10) => Some(IsaEntry::BlendH(blend_h::<10>)),
        ("mc", "blend_h", 12) => Some(IsaEntry::BlendH(blend_h::<12>)),
        ("cdef", "dir", 8) => Some(IsaEntry::CdefDir(cdef_find_dir::<8>)),
        ("cdef", "dir", 10) => Some(IsaEntry::CdefDir(cdef_find_dir::<10>)),
        ("cdef", "dir", 12) => Some(IsaEntry::CdefDir(cdef_find_dir::<12>)),
        ("cdef", "fb[0]", 8) => Some(IsaEntry::CdefFb(0, cdef_filter::<8>)),
        ("cdef", "fb[1]", 8) => Some(IsaEntry::CdefFb(1, cdef_filter::<8>)),
        ("cdef", "fb[2]", 8) => Some(IsaEntry::CdefFb(2, cdef_filter::<8>)),
        ("cdef", "fb[0]", 10) => Some(IsaEntry::CdefFb(0, cdef_filter::<10>)),
        ("cdef", "fb[1]", 10) => Some(IsaEntry::CdefFb(1, cdef_filter::<10>)),
        ("cdef", "fb[2]", 10) => Some(IsaEntry::CdefFb(2, cdef_filter::<10>)),
        ("cdef", "fb[0]", 12) => Some(IsaEntry::CdefFb(0, cdef_filter::<12>)),
        ("cdef", "fb[1]", 12) => Some(IsaEntry::CdefFb(1, cdef_filter::<12>)),
        ("cdef", "fb[2]", 12) => Some(IsaEntry::CdefFb(2, cdef_filter::<12>)),
        ("pal", "pal_idx_finish", 0) => Some(IsaEntry::PalIdxFinish(pal_idx_finish)),
        _ => None,
    }
}

// ============================================================================
// Güvenli `v128` yükleme / saklama yardımcıları (`#![forbid(unsafe_code)]`)
// ============================================================================

#[inline]
#[target_feature(enable = "simd128")]
fn load_v128_u8(src: &[u8], off: usize) -> v128 {
    let b0: [u8; 8] = src[off..off + 8].try_into().unwrap();
    let b1: [u8; 8] = src[off + 8..off + 16].try_into().unwrap();
    i64x2(i64::from_le_bytes(b0), i64::from_le_bytes(b1))
}

#[inline]
#[target_feature(enable = "simd128")]
fn load_low64_u8(src: &[u8], off: usize) -> v128 {
    let b0: [u8; 8] = src[off..off + 8].try_into().unwrap();
    i64x2(i64::from_le_bytes(b0), 0)
}

#[inline]
#[target_feature(enable = "simd128")]
fn load_low32_u8(src: &[u8], off: usize) -> v128 {
    let b0: [u8; 4] = src[off..off + 4].try_into().unwrap();
    i64x2(i64::from(u32::from_le_bytes(b0)), 0)
}

#[inline]
#[target_feature(enable = "simd128")]
fn load_i16x8(src: &[i16], off: usize) -> v128 {
    let s = &src[off..off + 8];
    i16x8(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7])
}

#[inline]
#[target_feature(enable = "simd128")]
fn load_i16x4(src: &[i16], off: usize) -> v128 {
    let s = &src[off..off + 4];
    i16x8(s[0], s[1], s[2], s[3], 0, 0, 0, 0)
}

#[inline]
#[target_feature(enable = "simd128")]
fn store_v128_u8(dst: &mut [u8], off: usize, v: v128) {
    dst[off..off + 8].copy_from_slice(&i64x2_extract_lane::<0>(v).to_le_bytes());
    dst[off + 8..off + 16].copy_from_slice(&i64x2_extract_lane::<1>(v).to_le_bytes());
}

#[inline]
#[target_feature(enable = "simd128")]
fn store_low64_u8(dst: &mut [u8], off: usize, v: v128) {
    dst[off..off + 8].copy_from_slice(&i64x2_extract_lane::<0>(v).to_le_bytes());
}

#[inline]
#[target_feature(enable = "simd128")]
fn store_low32_u8(dst: &mut [u8], off: usize, v: v128) {
    dst[off..off + 4].copy_from_slice(&i32x4_extract_lane::<0>(v).to_le_bytes());
}

#[inline(always)]
const fn bpp_of(bits: u32) -> usize {
    if bits == 8 { 1 } else { 2 }
}

#[inline(always)]
const fn intermediate_bits_of(bits: u32) -> u32 {
    if bits == 8 { 4 } else { 14 - bits }
}

#[inline(always)]
const fn prep_bias_of(bits: u32) -> i32 {
    if bits == 8 { 0 } else { 8192 }
}

#[inline]
#[target_feature(enable = "simd128")]
fn load_pixels_8<const BPC: u32>(buf: &[u8], off: usize) -> v128 {
    if BPC == 8 {
        u16x8_extend_low_u8x16(load_low64_u8(buf, off))
    } else {
        load_v128_u8(buf, off)
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn load_pixels_4<const BPC: u32>(buf: &[u8], off: usize) -> v128 {
    if BPC == 8 {
        u16x8_extend_low_u8x16(load_low32_u8(buf, off))
    } else {
        load_low64_u8(buf, off)
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn pack_and_store_8<const BPC: u32>(dst: &mut [u8], off: usize, lo: v128, hi: v128, v_max: v128) {
    let p16 = i16x8_min(i16x8_max(i16x8_narrow_i32x4(lo, hi), i16x8_splat(0)), v_max);
    if BPC == 8 {
        let p8 = u8x16_narrow_i16x8(p16, p16);
        store_low64_u8(dst, off, p8);
    } else {
        store_v128_u8(dst, off, p16);
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn pack_and_store_4<const BPC: u32>(dst: &mut [u8], off: usize, lo: v128, v_max: v128) {
    let p16 = i16x8_min(i16x8_max(i16x8_narrow_i32x4(lo, lo), i16x8_splat(0)), v_max);
    if BPC == 8 {
        let p8 = u8x16_narrow_i16x8(p16, p16);
        store_low32_u8(dst, off, p8);
    } else {
        store_low64_u8(dst, off, p16);
    }
}

#[inline(always)]
fn load_scalar<const BPC: u32>(buf: &[u8], off: usize) -> i32 {
    if BPC == 8 {
        i32::from(buf[off])
    } else {
        i32::from(u16::from_le_bytes([buf[off], buf[off + 1]]))
    }
}

#[inline(always)]
fn store_scalar<const BPC: u32>(buf: &mut [u8], off: usize, value: i32) {
    if BPC == 8 {
        buf[off] = value as u8;
    } else {
        buf[off..off + 2].copy_from_slice(&(value as u16).to_le_bytes());
    }
}

// ============================================================================
// 1. MC `avg`, `w_avg`, `mask`, `blend`, `blend_v`, `blend_h` (18 yuva)
// ============================================================================

#[target_feature(enable = "simd128")]
fn avg_simd128<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
) {
    let ib = intermediate_bits_of(BPC);
    let shift = ib + 1;
    let rnd = (1i32 << ib) + prep_bias_of(BPC) * 2;
    let max_pixel = (1i32 << BPC) - 1;
    let bpp = bpp_of(BPC);
    let v_rnd = i32x4_splat(rnd);
    let v_max = i16x8_splat(max_pixel as i16);

    for y in 0..h {
        let drow = y * dst_stride;
        let trow = y * w;
        let mut x = 0usize;
        while x + 8 <= w {
            let a = load_i16x8(tmp1, trow + x);
            let b = load_i16x8(tmp2, trow + x);
            let lo = i32x4_shr(
                i32x4_add(
                    i32x4_add(i32x4_extend_low_i16x8(a), i32x4_extend_low_i16x8(b)),
                    v_rnd,
                ),
                shift,
            );
            let hi = i32x4_shr(
                i32x4_add(
                    i32x4_add(i32x4_extend_high_i16x8(a), i32x4_extend_high_i16x8(b)),
                    v_rnd,
                ),
                shift,
            );
            pack_and_store_8::<BPC>(dst, drow + x * bpp, lo, hi, v_max);
            x += 8;
        }
        while x + 4 <= w {
            let a = load_i16x4(tmp1, trow + x);
            let b = load_i16x4(tmp2, trow + x);
            let lo = i32x4_shr(
                i32x4_add(
                    i32x4_add(i32x4_extend_low_i16x8(a), i32x4_extend_low_i16x8(b)),
                    v_rnd,
                ),
                shift,
            );
            pack_and_store_4::<BPC>(dst, drow + x * bpp, lo, v_max);
            x += 4;
        }
        while x < w {
            let a = i32::from(tmp1[trow + x]);
            let b = i32::from(tmp2[trow + x]);
            let val = ((a + b + rnd) >> shift).clamp(0, max_pixel);
            store_scalar::<BPC>(dst, drow + x * bpp, val);
            x += 1;
        }
    }
}

/// `mc/avg` WASM `simd128` yuva girişi.
pub fn avg<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    debug_assert_eq!(bit_depth.bits(), BPC);
    debug_assert_eq!(u32::from(bitdepth_max), (1u32 << BPC) - 1);
    avg_simd128::<BPC>(dst, dst_stride, tmp1, tmp2, w, h);
}

#[target_feature(enable = "simd128")]
fn w_avg_simd128<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
) {
    let ib = intermediate_bits_of(BPC);
    let shift = ib + 4;
    let rnd = (8i32 << ib) + prep_bias_of(BPC) * 16;
    let max_pixel = (1i32 << BPC) - 1;
    let bpp = bpp_of(BPC);
    let v_w1 = i32x4_splat(weight);
    let v_w2 = i32x4_splat(16 - weight);
    let v_rnd = i32x4_splat(rnd);
    let v_max = i16x8_splat(max_pixel as i16);

    for y in 0..h {
        let drow = y * dst_stride;
        let trow = y * w;
        let mut x = 0usize;
        while x + 8 <= w {
            let a = load_i16x8(tmp1, trow + x);
            let b = load_i16x8(tmp2, trow + x);
            let lo = i32x4_shr(
                i32x4_add(
                    i32x4_add(
                        i32x4_mul(i32x4_extend_low_i16x8(a), v_w1),
                        i32x4_mul(i32x4_extend_low_i16x8(b), v_w2),
                    ),
                    v_rnd,
                ),
                shift,
            );
            let hi = i32x4_shr(
                i32x4_add(
                    i32x4_add(
                        i32x4_mul(i32x4_extend_high_i16x8(a), v_w1),
                        i32x4_mul(i32x4_extend_high_i16x8(b), v_w2),
                    ),
                    v_rnd,
                ),
                shift,
            );
            pack_and_store_8::<BPC>(dst, drow + x * bpp, lo, hi, v_max);
            x += 8;
        }
        while x + 4 <= w {
            let a = load_i16x4(tmp1, trow + x);
            let b = load_i16x4(tmp2, trow + x);
            let lo = i32x4_shr(
                i32x4_add(
                    i32x4_add(
                        i32x4_mul(i32x4_extend_low_i16x8(a), v_w1),
                        i32x4_mul(i32x4_extend_low_i16x8(b), v_w2),
                    ),
                    v_rnd,
                ),
                shift,
            );
            pack_and_store_4::<BPC>(dst, drow + x * bpp, lo, v_max);
            x += 4;
        }
        while x < w {
            let a = i32::from(tmp1[trow + x]);
            let b = i32::from(tmp2[trow + x]);
            let val = ((a * weight + b * (16 - weight) + rnd) >> shift).clamp(0, max_pixel);
            store_scalar::<BPC>(dst, drow + x * bpp, val);
            x += 1;
        }
    }
}

/// `mc/w_avg` WASM `simd128` yuva girişi.
pub fn w_avg<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    debug_assert_eq!(bit_depth.bits(), BPC);
    debug_assert_eq!(u32::from(bitdepth_max), (1u32 << BPC) - 1);
    w_avg_simd128::<BPC>(dst, dst_stride, tmp1, tmp2, w, h, weight);
}

#[target_feature(enable = "simd128")]
fn mask_simd128<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask_buf: &[u8],
) {
    let ib = intermediate_bits_of(BPC);
    let shift = ib + 6;
    let rnd = (32i32 << ib) + prep_bias_of(BPC) * 64;
    let max_pixel = (1i32 << BPC) - 1;
    let bpp = bpp_of(BPC);
    let v_rnd = i32x4_splat(rnd);
    let v_64 = i16x8_splat(64);
    let v_max = i16x8_splat(max_pixel as i16);

    for y in 0..h {
        let drow = y * dst_stride;
        let trow = y * w;
        let mut x = 0usize;
        while x + 8 <= w {
            let a = load_i16x8(tmp1, trow + x);
            let b = load_i16x8(tmp2, trow + x);
            let m16 = u16x8_extend_low_u8x16(load_low64_u8(mask_buf, trow + x));
            let inv_m16 = i16x8_sub(v_64, m16);
            let lo = i32x4_shr(
                i32x4_add(
                    i32x4_add(
                        i32x4_mul(i32x4_extend_low_i16x8(a), i32x4_extend_low_i16x8(m16)),
                        i32x4_mul(i32x4_extend_low_i16x8(b), i32x4_extend_low_i16x8(inv_m16)),
                    ),
                    v_rnd,
                ),
                shift,
            );
            let hi = i32x4_shr(
                i32x4_add(
                    i32x4_add(
                        i32x4_mul(i32x4_extend_high_i16x8(a), i32x4_extend_high_i16x8(m16)),
                        i32x4_mul(i32x4_extend_high_i16x8(b), i32x4_extend_high_i16x8(inv_m16)),
                    ),
                    v_rnd,
                ),
                shift,
            );
            pack_and_store_8::<BPC>(dst, drow + x * bpp, lo, hi, v_max);
            x += 8;
        }
        while x + 4 <= w {
            let a = load_i16x4(tmp1, trow + x);
            let b = load_i16x4(tmp2, trow + x);
            let m16 = u16x8_extend_low_u8x16(load_low32_u8(mask_buf, trow + x));
            let inv_m16 = i16x8_sub(v_64, m16);
            let lo = i32x4_shr(
                i32x4_add(
                    i32x4_add(
                        i32x4_mul(i32x4_extend_low_i16x8(a), i32x4_extend_low_i16x8(m16)),
                        i32x4_mul(i32x4_extend_low_i16x8(b), i32x4_extend_low_i16x8(inv_m16)),
                    ),
                    v_rnd,
                ),
                shift,
            );
            pack_and_store_4::<BPC>(dst, drow + x * bpp, lo, v_max);
            x += 4;
        }
        while x < w {
            let m = i32::from(mask_buf[trow + x]);
            let a = i32::from(tmp1[trow + x]);
            let b = i32::from(tmp2[trow + x]);
            let val = ((a * m + b * (64 - m) + rnd) >> shift).clamp(0, max_pixel);
            store_scalar::<BPC>(dst, drow + x * bpp, val);
            x += 1;
        }
    }
}

/// `mc/mask` WASM `simd128` yuva girişi.
pub fn mask<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask_buf: &[u8],
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    debug_assert_eq!(bit_depth.bits(), BPC);
    debug_assert_eq!(u32::from(bitdepth_max), (1u32 << BPC) - 1);
    mask_simd128::<BPC>(dst, dst_stride, tmp1, tmp2, w, h, mask_buf);
}

#[inline]
#[target_feature(enable = "simd128")]
fn blend_8_pixels<const BPC: u32>(
    dst: &mut [u8],
    d_off: usize,
    tmp: &[u8],
    t_off: usize,
    m16: v128,
    v_max: v128,
) {
    let a = load_pixels_8::<BPC>(dst, d_off);
    let b = load_pixels_8::<BPC>(tmp, t_off);
    let inv_m16 = i16x8_sub(i16x8_splat(64), m16);
    let v_rnd = i32x4_splat(32);
    let lo = i32x4_shr(
        i32x4_add(
            i32x4_add(
                i32x4_mul(i32x4_extend_low_i16x8(a), i32x4_extend_low_i16x8(inv_m16)),
                i32x4_mul(i32x4_extend_low_i16x8(b), i32x4_extend_low_i16x8(m16)),
            ),
            v_rnd,
        ),
        6,
    );
    let hi = i32x4_shr(
        i32x4_add(
            i32x4_add(
                i32x4_mul(i32x4_extend_high_i16x8(a), i32x4_extend_high_i16x8(inv_m16)),
                i32x4_mul(i32x4_extend_high_i16x8(b), i32x4_extend_high_i16x8(m16)),
            ),
            v_rnd,
        ),
        6,
    );
    pack_and_store_8::<BPC>(dst, d_off, lo, hi, v_max);
}

#[inline]
#[target_feature(enable = "simd128")]
fn blend_4_pixels<const BPC: u32>(
    dst: &mut [u8],
    d_off: usize,
    tmp: &[u8],
    t_off: usize,
    m16: v128,
    v_max: v128,
) {
    let a = load_pixels_4::<BPC>(dst, d_off);
    let b = load_pixels_4::<BPC>(tmp, t_off);
    let inv_m16 = i16x8_sub(i16x8_splat(64), m16);
    let v_rnd = i32x4_splat(32);
    let lo = i32x4_shr(
        i32x4_add(
            i32x4_add(
                i32x4_mul(i32x4_extend_low_i16x8(a), i32x4_extend_low_i16x8(inv_m16)),
                i32x4_mul(i32x4_extend_low_i16x8(b), i32x4_extend_low_i16x8(m16)),
            ),
            v_rnd,
        ),
        6,
    );
    pack_and_store_4::<BPC>(dst, d_off, lo, v_max);
}

#[target_feature(enable = "simd128")]
fn blend_simd128<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    mask_buf: &[u8],
) {
    let bpp = bpp_of(BPC);
    let max_pixel = (1i32 << BPC) - 1;
    let v_max = i16x8_splat(max_pixel as i16);
    for y in 0..h {
        let drow = y * dst_stride;
        let trow = y * w;
        let mut x = 0usize;
        while x + 8 <= w {
            let m16 = u16x8_extend_low_u8x16(load_low64_u8(mask_buf, trow + x));
            blend_8_pixels::<BPC>(dst, drow + x * bpp, tmp, (trow + x) * bpp, m16, v_max);
            x += 8;
        }
        while x + 4 <= w {
            let m16 = u16x8_extend_low_u8x16(load_low32_u8(mask_buf, trow + x));
            blend_4_pixels::<BPC>(dst, drow + x * bpp, tmp, (trow + x) * bpp, m16, v_max);
            x += 4;
        }
        while x < w {
            let a = load_scalar::<BPC>(dst, drow + x * bpp);
            let b = load_scalar::<BPC>(tmp, (trow + x) * bpp);
            let m = i32::from(mask_buf[trow + x]);
            store_scalar::<BPC>(dst, drow + x * bpp, (a * (64 - m) + b * m + 32) >> 6);
            x += 1;
        }
    }
}

/// `mc/blend` WASM `simd128` yuva girişi.
pub fn blend<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    mask_buf: &[u8],
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    debug_assert_eq!(bit_depth.bits(), BPC);
    debug_assert_eq!(u32::from(bitdepth_max), (1u32 << BPC) - 1);
    blend_simd128::<BPC>(dst, dst_stride, tmp, w, h, mask_buf);
}

#[target_feature(enable = "simd128")]
fn blend_v_simd128<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
) {
    let bpp = bpp_of(BPC);
    let max_pixel = (1i32 << BPC) - 1;
    let v_max = i16x8_splat(max_pixel as i16);
    let mask_row = &OBMC_MASKS[w..];
    let cols = (w * 3) >> 2;
    for y in 0..h {
        let drow = y * dst_stride;
        let trow = y * w;
        let mut x = 0usize;
        while x + 8 <= cols {
            let s = &mask_row[x..x + 8];
            let m16 = i16x8(
                i16::from(s[0]),
                i16::from(s[1]),
                i16::from(s[2]),
                i16::from(s[3]),
                i16::from(s[4]),
                i16::from(s[5]),
                i16::from(s[6]),
                i16::from(s[7]),
            );
            blend_8_pixels::<BPC>(dst, drow + x * bpp, tmp, (trow + x) * bpp, m16, v_max);
            x += 8;
        }
        while x + 4 <= cols {
            let s = &mask_row[x..x + 4];
            let m16 = i16x8(
                i16::from(s[0]),
                i16::from(s[1]),
                i16::from(s[2]),
                i16::from(s[3]),
                0,
                0,
                0,
                0,
            );
            blend_4_pixels::<BPC>(dst, drow + x * bpp, tmp, (trow + x) * bpp, m16, v_max);
            x += 4;
        }
        while x < cols {
            let a = load_scalar::<BPC>(dst, drow + x * bpp);
            let b = load_scalar::<BPC>(tmp, (trow + x) * bpp);
            let m = i32::from(mask_row[x]);
            store_scalar::<BPC>(dst, drow + x * bpp, (a * (64 - m) + b * m + 32) >> 6);
            x += 1;
        }
    }
}

/// `mc/blend_v` WASM `simd128` yuva girişi.
pub fn blend_v<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    debug_assert_eq!(bit_depth.bits(), BPC);
    debug_assert_eq!(u32::from(bitdepth_max), (1u32 << BPC) - 1);
    blend_v_simd128::<BPC>(dst, dst_stride, tmp, w, h);
}

#[target_feature(enable = "simd128")]
fn blend_h_simd128<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
) {
    let bpp = bpp_of(BPC);
    let max_pixel = (1i32 << BPC) - 1;
    let v_max = i16x8_splat(max_pixel as i16);
    let mask_col = &OBMC_MASKS[h..];
    let rows = (h * 3) >> 2;
    for y in 0..rows {
        let m = i32::from(mask_col[y]);
        let m16 = i16x8_splat(m as i16);
        let drow = y * dst_stride;
        let trow = y * w;
        let mut x = 0usize;
        while x + 8 <= w {
            blend_8_pixels::<BPC>(dst, drow + x * bpp, tmp, (trow + x) * bpp, m16, v_max);
            x += 8;
        }
        while x + 4 <= w {
            blend_4_pixels::<BPC>(dst, drow + x * bpp, tmp, (trow + x) * bpp, m16, v_max);
            x += 4;
        }
        while x < w {
            let a = load_scalar::<BPC>(dst, drow + x * bpp);
            let b = load_scalar::<BPC>(tmp, (trow + x) * bpp);
            store_scalar::<BPC>(dst, drow + x * bpp, (a * (64 - m) + b * m + 32) >> 6);
            x += 1;
        }
    }
}

/// `mc/blend_h` WASM `simd128` yuva girişi.
pub fn blend_h<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    debug_assert_eq!(bit_depth.bits(), BPC);
    debug_assert_eq!(u32::from(bitdepth_max), (1u32 << BPC) - 1);
    blend_h_simd128::<BPC>(dst, dst_stride, tmp, w, h);
}

// ============================================================================
// 2. CDEF `dir` ve `fb[0..2]` (12 yuva)
// ============================================================================

#[inline]
#[target_feature(enable = "simd128")]
fn hsum_i32x4(v: v128) -> i32 {
    i32x4_extract_lane::<0>(v)
        .wrapping_add(i32x4_extract_lane::<1>(v))
        .wrapping_add(i32x4_extract_lane::<2>(v))
        .wrapping_add(i32x4_extract_lane::<3>(v))
}

#[target_feature(enable = "simd128")]
fn cdef_find_dir_simd128<const BPC: u32>(img: &[u8], stride: usize) -> CdefDirResult {
    let bpp = bpp_of(BPC);
    let bd_min_8 = BPC - 8;
    let v_128 = i16x8_splat(128);

    let mut partial_sum_hv0 = [0i16; 8];
    let mut v_hv1 = i16x8_splat(0);
    let mut partial_sum_diag = [[0i32; 15]; 2];
    let mut partial_sum_alt = [[0i32; 11]; 4];

    for y in 0..8usize {
        let raw = load_pixels_8::<BPC>(img, y * stride);
        let row_vec = i16x8_sub(u16x8_shr(raw, bd_min_8), v_128);
        v_hv1 = i16x8_add(v_hv1, row_vec);

        let r = [
            i32::from(i16x8_extract_lane::<0>(row_vec)),
            i32::from(i16x8_extract_lane::<1>(row_vec)),
            i32::from(i16x8_extract_lane::<2>(row_vec)),
            i32::from(i16x8_extract_lane::<3>(row_vec)),
            i32::from(i16x8_extract_lane::<4>(row_vec)),
            i32::from(i16x8_extract_lane::<5>(row_vec)),
            i32::from(i16x8_extract_lane::<6>(row_vec)),
            i32::from(i16x8_extract_lane::<7>(row_vec)),
        ];
        let mut row_sum = 0i32;
        for x in 0..8usize {
            let px = r[x];
            row_sum += px;
            partial_sum_diag[0][y + x] += px;
            partial_sum_alt[0][y + (x >> 1)] += px;
            partial_sum_alt[1][3 + y - (x >> 1)] += px;
            partial_sum_diag[1][7 + y - x] += px;
            partial_sum_alt[2][3 - (y >> 1) + x] += px;
            partial_sum_alt[3][(y >> 1) + x] += px;
        }
        partial_sum_hv0[y] = row_sum as i16;
    }

    let v_hv0 = load_i16x8(&partial_sum_hv0, 0);
    let mut cost = [0u32; 8];
    cost[2] = (hsum_i32x4(i32x4_dot_i16x8(v_hv0, v_hv0)) as u32).wrapping_mul(DIR_CENTER_WEIGHT);
    cost[6] = (hsum_i32x4(i32x4_dot_i16x8(v_hv1, v_hv1)) as u32).wrapping_mul(DIR_CENTER_WEIGHT);

    for n in 0..7usize {
        let d = DIV_TABLE[n];
        let a0 = partial_sum_diag[0][n];
        let b0 = partial_sum_diag[0][14 - n];
        cost[0] = cost[0].wrapping_add(
            (a0.wrapping_mul(a0).wrapping_add(b0.wrapping_mul(b0)) as u32).wrapping_mul(d),
        );
        let a1 = partial_sum_diag[1][n];
        let b1 = partial_sum_diag[1][14 - n];
        cost[4] = cost[4].wrapping_add(
            (a1.wrapping_mul(a1).wrapping_add(b1.wrapping_mul(b1)) as u32).wrapping_mul(d),
        );
    }
    let c0 = partial_sum_diag[0][7];
    let c1 = partial_sum_diag[1][7];
    cost[0] = cost[0].wrapping_add((c0.wrapping_mul(c0) as u32).wrapping_mul(DIR_CENTER_WEIGHT));
    cost[4] = cost[4].wrapping_add((c1.wrapping_mul(c1) as u32).wrapping_mul(DIR_CENTER_WEIGHT));

    for n in 0..4usize {
        let mut acc = 0u32;
        for m in 0..5usize {
            let v = partial_sum_alt[n][3 + m];
            acc = acc.wrapping_add(v.wrapping_mul(v) as u32);
        }
        acc = acc.wrapping_mul(DIR_CENTER_WEIGHT);
        for m in 0..3usize {
            let d = DIV_TABLE[2 * m + 1];
            let a = partial_sum_alt[n][m];
            let b = partial_sum_alt[n][10 - m];
            acc = acc.wrapping_add(
                (a.wrapping_mul(a).wrapping_add(b.wrapping_mul(b)) as u32).wrapping_mul(d),
            );
        }
        cost[n * 2 + 1] = acc;
    }

    let mut best_dir = 0usize;
    let mut best_cost = cost[0];
    for n in 1..8usize {
        if cost[n] > best_cost {
            best_cost = cost[n];
            best_dir = n;
        }
    }

    let _ = bpp;
    CdefDirResult {
        dir: best_dir as u8,
        var: best_cost.wrapping_sub(cost[best_dir ^ 4]) >> 10,
    }
}

/// `cdef/dir` WASM `simd128` yuva girişi.
pub fn cdef_find_dir<const BPC: u32>(
    img: &[u8],
    stride: usize,
    bit_depth: CdefBitDepth,
) -> CdefDirResult {
    debug_assert_eq!(bit_depth.bits(), BPC);
    cdef_find_dir_simd128::<BPC>(img, stride)
}

#[inline]
#[target_feature(enable = "simd128")]
fn constrain_i32x4(diff: v128, v_thresh: v128, shift: u32) -> v128 {
    let adiff = i32x4_abs(diff);
    let shifted = i32x4_shr(adiff, shift);
    let limit = i32x4_max(i32x4_sub(v_thresh, shifted), i32x4_splat(0));
    let mag = i32x4_min(adiff, limit);
    let neg_mask = i32x4_lt(diff, i32x4_splat(0));
    v128_bitselect(i32x4_neg(mag), mag, neg_mask)
}

#[inline]
#[target_feature(enable = "simd128")]
fn round_shift_i32x4(sum: v128) -> v128 {
    // `(sum - (sum < 0) + 8) >> 4`; `i32x4_lt(sum, 0)` negatif şeritlerde `-1` üretir.
    let neg = i32x4_lt(sum, i32x4_splat(0));
    i32x4_shr(i32x4_add(i32x4_add(sum, neg), i32x4_splat(8)), 4)
}

#[inline]
#[target_feature(enable = "simd128")]
fn load_tmp_i32x4(tmp: &[i16; TMP_BUF_LEN], base_idx: usize) -> v128 {
    i32x4_extend_low_i16x8(load_i16x4(tmp, base_idx))
}

#[target_feature(enable = "simd128")]
fn cdef_filter_simd128<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    let bpp = bpp_of(BPC);
    let w = params.size.w();
    let h = params.size.h();

    let has_left = params.edges.contains(CdefEdgeFlags::HAVE_LEFT);
    let has_right = params.edges.contains(CdefEdgeFlags::HAVE_RIGHT);
    let has_top = params.edges.contains(CdefEdgeFlags::HAVE_TOP);
    let has_bottom = params.edges.contains(CdefEdgeFlags::HAVE_BOTTOM);

    let x_start: i32 = if has_left { -2 } else { 0 };
    let x_end: i32 = if has_right { w as i32 + 2 } else { w as i32 };

    let mut tmp = [TMP_FILL; TMP_BUF_LEN];
    let top_rows = if has_top { 2usize } else { 0 };
    let bottom_rows = if has_bottom { 2usize } else { 0 };

    for row in 0..top_rows {
        let y = -2 + row as i32;
        for x in x_start..x_end {
            let v = px_at::<BPC>(top, row * stride + (x + 2) as usize * bpp);
            tmp[tmp_index(x, y)] = v as i16;
        }
    }
    if has_left {
        for y in 0..h {
            for x in -2i32..0 {
                let v = px_at::<BPC>(left, y * 2 * bpp + (x + 2) as usize * bpp);
                tmp[tmp_index(x, y as i32)] = v as i16;
            }
        }
    }
    for y in 0..h {
        for x in 0..x_end {
            let v = px_at::<BPC>(dst, y * stride + x as usize * bpp);
            tmp[tmp_index(x, y as i32)] = v as i16;
        }
    }
    for row in 0..bottom_rows {
        let y = h as i32 + row as i32;
        for x in x_start..x_end {
            let v = px_at::<BPC>(bottom, row * stride + (x + 2) as usize * bpp);
            tmp[tmp_index(x, y)] = v as i16;
        }
    }

    let dir = usize::from(params.dir & 7);
    let pri_strength = i32::from(params.pri_strength);
    let sec_strength = i32::from(params.sec_strength);
    if pri_strength == 0 && sec_strength == 0 {
        return;
    }

    let v_pri_thresh = i32x4_splat(pri_strength);
    let v_sec_thresh = i32x4_splat(sec_strength);
    let bd_min_8 = BPC - 8;
    let pri_tap0 = 4 - (((params.pri_strength as u32) >> bd_min_8) & 1) as i32;
    let pri_tap1 = (pri_tap0 & 3) | 2;
    let pri_taps = [i32x4_splat(pri_tap0), i32x4_splat(pri_tap1)];
    let sec_taps = [i32x4_splat(2), i32x4_splat(1)];
    let pri_shift = if pri_strength != 0 {
        clamped_shift(params.damping as i32 - ulog2(params.pri_strength as u32) as i32)
    } else {
        0
    };
    let sec_shift = if sec_strength != 0 {
        clamped_shift(params.damping as i32 - ulog2(params.sec_strength as u32) as i32)
    } else {
        0
    };

    for y in 0..h {
        let mut x = 0usize;
        while x + 4 <= w {
            let base = tmp_index(x as i32, y as i32);
            let px = load_tmp_i32x4(&tmp, base);
            let mut sum = i32x4_splat(0);
            let mut min_v = px;
            let mut max_v = px;

            if pri_strength != 0 {
                for k in 0..2usize {
                    let off1 = isize::from(CDEF_DIRECTIONS[dir + 2][k]);
                    let p0 = load_tmp_i32x4(&tmp, (base as isize + off1) as usize);
                    let p1 = load_tmp_i32x4(&tmp, (base as isize - off1) as usize);
                    let c0 = constrain_i32x4(i32x4_sub(p0, px), v_pri_thresh, pri_shift);
                    let c1 = constrain_i32x4(i32x4_sub(p1, px), v_pri_thresh, pri_shift);
                    sum = i32x4_add(sum, i32x4_mul(pri_taps[k], i32x4_add(c0, c1)));
                    if sec_strength != 0 {
                        min_v = u32x4_min(min_v, p0);
                        max_v = i32x4_max(max_v, p0);
                        min_v = u32x4_min(min_v, p1);
                        max_v = i32x4_max(max_v, p1);
                    }
                }
            }

            if sec_strength != 0 {
                for k in 0..2usize {
                    let off2 = isize::from(CDEF_DIRECTIONS[dir + 4][k]);
                    let off3 = isize::from(CDEF_DIRECTIONS[dir][k]);
                    let s0 = load_tmp_i32x4(&tmp, (base as isize + off2) as usize);
                    let s1 = load_tmp_i32x4(&tmp, (base as isize - off2) as usize);
                    let s2 = load_tmp_i32x4(&tmp, (base as isize + off3) as usize);
                    let s3 = load_tmp_i32x4(&tmp, (base as isize - off3) as usize);
                    let cs0 = constrain_i32x4(i32x4_sub(s0, px), v_sec_thresh, sec_shift);
                    let cs1 = constrain_i32x4(i32x4_sub(s1, px), v_sec_thresh, sec_shift);
                    let cs2 = constrain_i32x4(i32x4_sub(s2, px), v_sec_thresh, sec_shift);
                    let cs3 = constrain_i32x4(i32x4_sub(s3, px), v_sec_thresh, sec_shift);
                    let cs_tot = i32x4_add(i32x4_add(cs0, cs1), i32x4_add(cs2, cs3));
                    sum = i32x4_add(sum, i32x4_mul(sec_taps[k], cs_tot));
                    if pri_strength != 0 {
                        min_v = u32x4_min(u32x4_min(min_v, s0), u32x4_min(s1, u32x4_min(s2, s3)));
                        max_v = i32x4_max(i32x4_max(max_v, s0), i32x4_max(s1, i32x4_max(s2, s3)));
                    }
                }
            }

            let mut out = i32x4_add(px, round_shift_i32x4(sum));
            if pri_strength != 0 && sec_strength != 0 {
                out = i32x4_min(i32x4_max(out, min_v), max_v);
            }
            let drow = y * stride;
            px_put::<BPC>(dst, drow + x * bpp, i32x4_extract_lane::<0>(out) as u16);
            px_put::<BPC>(
                dst,
                drow + (x + 1) * bpp,
                i32x4_extract_lane::<1>(out) as u16,
            );
            px_put::<BPC>(
                dst,
                drow + (x + 2) * bpp,
                i32x4_extract_lane::<2>(out) as u16,
            );
            px_put::<BPC>(
                dst,
                drow + (x + 3) * bpp,
                i32x4_extract_lane::<3>(out) as u16,
            );
            x += 4;
        }
    }
}

/// `cdef/fb[0..2]` WASM `simd128` yuva girişi.
pub fn cdef_filter<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    debug_assert_eq!(params.bit_depth.bits(), BPC);
    cdef_filter_simd128::<BPC>(dst, stride, left, top, bottom, params);
}

// ============================================================================
// 3. PAL `pal_idx_finish` (1 yuva)
// ============================================================================

#[inline]
#[target_feature(enable = "simd128")]
fn pack_pal_16(a: v128) -> v128 {
    let even = i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14>(a, a);
    let odd = i8x16_shuffle::<1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15>(a, a);
    let odd_shl4 = v128_and(u16x8_shl(odd, 4), u8x16_splat(0xF0));
    v128_or(even, odd_shl4)
}

#[target_feature(enable = "simd128")]
fn pal_idx_finish_simd128(dst: &mut [u8], src: &[u8], bw: usize, bh: usize, w: usize, h: usize) {
    assert!(
        (4..=64).contains(&bw) && bw.is_power_of_two(),
        "pal_idx_finish bw 4..=64 ve 2'nin kuvveti olmalı: {bw}"
    );
    assert!(
        (4..=64).contains(&bh) && bh.is_power_of_two(),
        "pal_idx_finish bh 4..=64 ve 2'nin kuvveti olmalı: {bh}"
    );
    assert!(
        (4..=w).contains(&w) && w <= bw && w.is_multiple_of(4),
        "pal_idx_finish w 4..=bw ve 4'ün katı olmalı: {w}"
    );
    assert!(
        (4..=h).contains(&h) && h <= bh && h.is_multiple_of(4),
        "pal_idx_finish h 4..=bh ve 4'ün katı olmalı: {h}"
    );
    assert!(
        src.len() >= bw * bh,
        "pal_idx_finish src dilimi kısa: {} < {}",
        src.len(),
        bw * bh
    );
    let dst_width = w / 2;
    let dst_bw = bw / 2;
    assert!(
        dst.len() >= dst_bw * bh,
        "pal_idx_finish dst dilimi kısa: {} < {}",
        dst.len(),
        dst_bw * bh
    );

    for y in 0..h {
        let src_row = y * bw;
        let dst_row = y * dst_bw;
        let mut sx = 0usize;
        while sx + 16 <= w {
            let a = load_v128_u8(src, src_row + sx);
            let packed = pack_pal_16(a);
            store_low64_u8(dst, dst_row + sx / 2, packed);
            sx += 16;
        }
        while sx + 8 <= w {
            let a = load_low64_u8(src, src_row + sx);
            let packed = pack_pal_16(a);
            store_low32_u8(dst, dst_row + sx / 2, packed);
            sx += 8;
        }
        while sx < w {
            dst[dst_row + sx / 2] = src[src_row + sx] | (src[src_row + sx + 1] << 4);
            sx += 2;
        }
        if dst_width < dst_bw {
            let fill = src[src_row + w - 1].wrapping_mul(0x11);
            dst[dst_row + dst_width..dst_row + dst_bw].fill(fill);
        }
    }

    if h < bh {
        let last_row = (h - 1) * dst_bw;
        for y in h..bh {
            let dst_row = y * dst_bw;
            for x in 0..dst_bw {
                dst[dst_row + x] = dst[last_row + x];
            }
        }
    }
}

/// `pal/pal_idx_finish` WASM `simd128` yuva girişi.
pub fn pal_idx_finish(dst: &mut [u8], src: &[u8], bw: usize, bh: usize, w: usize, h: usize) {
    pal_idx_finish_simd128(dst, src, bw, bh, w, h);
}
