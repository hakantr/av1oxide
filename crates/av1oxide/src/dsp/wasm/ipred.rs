//! WebAssembly `simd128` intra tahmin (`ipred`, `cfl_ac`, `cfl_pred`, `pal_pred`) çekirdekleri.
//!
//! Kapsam (`66` yuva @ `8 / 10 / 12` bpc):
//! - `intra_pred[0..13]`: `DC`, `VERT`, `HOR`, `LEFT_DC`, `TOP_DC`, `DC_128`,
//!   `Z1`, `Z2`, `Z3`, `SMOOTH`, `SMOOTH_V`, `SMOOTH_H`, `PAETH`, `FILTER`
//! - `cfl_ac[0..2]`: `4:2:0`, `4:2:2`, `4:4:4`
//! - `cfl_pred[0..3]`: `DC`, `LEFT_DC`, `TOP_DC`, `DC_128`
//! - `pal_pred`

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use core::arch::wasm32::{
    i8x16_shuffle, i16x8, i16x8_abs, i16x8_add, i16x8_le, i16x8_narrow_i32x4, i16x8_shl,
    i16x8_splat, i16x8_sub, i32x4_abs, i32x4_add, i32x4_extadd_pairwise_i16x8,
    i32x4_extmul_high_i16x8, i32x4_extmul_low_i16x8, i32x4_lt, i32x4_max, i32x4_min, i32x4_neg,
    i32x4_shr, i32x4_splat, i32x4_sub, i64x2_extract_lane, u8x16_add, u8x16_shl, u8x16_shr,
    u8x16_splat, u8x16_swizzle, u16x8_add, u16x8_extadd_pairwise_u8x16, u16x8_extend_low_u8x16,
    u16x8_mul, u16x8_shl, u16x8_shr, u16x8_shuffle, u16x8_splat, u16x8_sub, u32x4_add,
    u32x4_extadd_pairwise_u16x8, u32x4_extend_high_u16x8, u32x4_extend_low_u16x8,
    u32x4_extmul_high_u16x8, u32x4_extmul_low_u16x8, u32x4_mul, u32x4_shr, u32x4_splat, v128,
    v128_and, v128_any_true, v128_bitselect,
};

use super::{
    bpp_of, hsum_i32x4, load_i16x8, load_low32_u8, load_low64_u8, load_pixels_4, load_pixels_8,
    load_scalar, load_u16x8, load_v128_u8, pack_and_store_4, pack_and_store_8, store_i16x8,
    store_low64_u8, store_pixels_4_i16, store_pixels_8_i16, store_scalar, store_v128_u8,
};
use crate::dsp::ipred::{
    AngularPredFn, CflAcFn, CflPredFn, CflPredMode, Edge, IntraPredMode, angular_skaler_devret,
    angular_sozlesmesi, cfl_ac_skaler_devret, cfl_ac_sozlesmesi, cfl_pred_skaler_devret,
    cfl_pred_sozlesmesi, pal_pred_skaler_devret, pal_pred_sozlesmesi,
};
use crate::dsp::ipred_tables::{DR_INTRA_DERIVATIVE, FILTER_INTRA_TAPS, SM_WEIGHTS};
use crate::frame::BitDepth;

const TMP_LEN: usize = 320;
const TMP_BASE: isize = 128;

const FILTER_TAPS_SIMD: [[[i16; 8]; 7]; 5] = {
    let mut out = [[[0i16; 8]; 7]; 5];
    let mut fi = 0usize;
    while fi < 5 {
        let taps = &FILTER_INTRA_TAPS[fi];
        let mut idx = 0usize;
        while idx < 8 {
            let slot = idx * 2;
            out[fi][0][idx] = taps[slot] as i16;
            out[fi][1][idx] = taps[slot + 1] as i16;
            out[fi][2][idx] = taps[slot + 16] as i16;
            out[fi][3][idx] = taps[slot + 17] as i16;
            out[fi][4][idx] = taps[slot + 32] as i16;
            out[fi][5][idx] = taps[slot + 33] as i16;
            out[fi][6][idx] = taps[slot + 48] as i16;
            idx += 1;
        }
        fi += 1;
    }
    out
};

/// `intra_pred[mode]` `wasm32 +simd128` giriş fonksiyonu.
#[must_use]
pub fn intra_pred_fn<const BPC: u32>(mode: IntraPredMode) -> AngularPredFn {
    match mode {
        IntraPredMode::Dc => ipred_angular::<BPC, 0>,
        IntraPredMode::Vert => ipred_angular::<BPC, 1>,
        IntraPredMode::Hor => ipred_angular::<BPC, 2>,
        IntraPredMode::LeftDc => ipred_angular::<BPC, 3>,
        IntraPredMode::TopDc => ipred_angular::<BPC, 4>,
        IntraPredMode::Dc128 => ipred_angular::<BPC, 5>,
        IntraPredMode::Z1 => ipred_angular::<BPC, 6>,
        IntraPredMode::Z2 => ipred_angular::<BPC, 7>,
        IntraPredMode::Z3 => ipred_angular::<BPC, 8>,
        IntraPredMode::Smooth => ipred_angular::<BPC, 9>,
        IntraPredMode::SmoothV => ipred_angular::<BPC, 10>,
        IntraPredMode::SmoothH => ipred_angular::<BPC, 11>,
        IntraPredMode::Paeth => ipred_angular::<BPC, 12>,
        IntraPredMode::Filter => ipred_angular::<BPC, 13>,
    }
}

/// `cfl_ac[layout]` `wasm32 +simd128` giriş fonksiyonu.
#[must_use]
pub fn cfl_ac_fn<const BPC: u32>(layout: usize) -> Option<CflAcFn> {
    match layout {
        0 => Some(ipred_cfl_ac::<BPC, 0>),
        1 => Some(ipred_cfl_ac::<BPC, 1>),
        2 => Some(ipred_cfl_ac::<BPC, 2>),
        _ => None,
    }
}

/// `cfl_pred[mode]` `wasm32 +simd128` giriş fonksiyonu.
#[must_use]
pub fn cfl_pred_fn<const BPC: u32>(mode: CflPredMode) -> CflPredFn {
    match mode {
        CflPredMode::Dc => ipred_cfl_pred::<BPC, 0>,
        CflPredMode::LeftDc => ipred_cfl_pred::<BPC, 1>,
        CflPredMode::TopDc => ipred_cfl_pred::<BPC, 2>,
        CflPredMode::Dc128 => ipred_cfl_pred::<BPC, 3>,
    }
}

#[inline(always)]
const fn bd_max_of<const BPC: u32>() -> i32 {
    (1i32 << BPC) - 1
}

#[inline(always)]
fn read_edge<const BPC: u32>(edge: Edge<'_>, rel: isize) -> i32 {
    let off = (edge.topleft as isize + rel * bpp_of(BPC) as isize) as usize;
    load_scalar::<BPC>(edge.bytes, off)
}

#[inline(always)]
fn mode_from_const<const MODE: usize>() -> IntraPredMode {
    match MODE {
        0 => IntraPredMode::Dc,
        1 => IntraPredMode::Vert,
        2 => IntraPredMode::Hor,
        3 => IntraPredMode::LeftDc,
        4 => IntraPredMode::TopDc,
        5 => IntraPredMode::Dc128,
        6 => IntraPredMode::Z1,
        7 => IntraPredMode::Z2,
        8 => IntraPredMode::Z3,
        9 => IntraPredMode::Smooth,
        10 => IntraPredMode::SmoothV,
        11 => IntraPredMode::SmoothH,
        12 => IntraPredMode::Paeth,
        _ => IntraPredMode::Filter,
    }
}

#[inline(always)]
fn cfl_mode_from_const<const MODE: usize>() -> CflPredMode {
    match MODE {
        0 => CflPredMode::Dc,
        1 => CflPredMode::LeftDc,
        2 => CflPredMode::TopDc,
        _ => CflPredMode::Dc128,
    }
}

#[target_feature(enable = "simd128")]
pub fn ipred_angular<const BPC: u32, const MODE: usize>(
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
    let mode = mode_from_const::<MODE>();
    if !angular_sozlesmesi::<BPC>(
        mode,
        dst.len(),
        dst_stride,
        edge,
        width,
        height,
        angle,
        bit_depth,
        bitdepth_max,
    ) {
        angular_skaler_devret::<BPC>(
            mode,
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
    match MODE {
        0 => splat_dc_wasm::<BPC>(
            dst,
            dst_stride,
            width,
            height,
            dc_gen_wasm::<BPC>(edge, width, height) as u16,
        ),
        1 => vert_wasm::<BPC>(dst, dst_stride, edge, width, height),
        2 => hor_wasm::<BPC>(dst, dst_stride, edge, width, height),
        3 => splat_dc_wasm::<BPC>(
            dst,
            dst_stride,
            width,
            height,
            dc_gen_left_wasm::<BPC>(edge, height) as u16,
        ),
        4 => splat_dc_wasm::<BPC>(
            dst,
            dst_stride,
            width,
            height,
            dc_gen_top_wasm::<BPC>(edge, width) as u16,
        ),
        5 => splat_dc_wasm::<BPC>(dst, dst_stride, width, height, 1u16 << (BPC - 1)),
        6 => z1_wasm::<BPC>(dst, dst_stride, edge, width, height, angle),
        7 => z2_wasm::<BPC>(
            dst, dst_stride, edge, width, height, angle, max_width, max_height,
        ),
        8 => z3_wasm::<BPC>(dst, dst_stride, edge, width, height, angle),
        9 => smooth_wasm::<BPC>(dst, dst_stride, edge, width, height),
        10 => smooth_v_wasm::<BPC>(dst, dst_stride, edge, width, height),
        11 => smooth_h_wasm::<BPC>(dst, dst_stride, edge, width, height),
        12 => paeth_wasm::<BPC>(dst, dst_stride, edge, width, height),
        _ => filter_pred_wasm::<BPC>(dst, dst_stride, edge, width, height, angle),
    }
}

#[target_feature(enable = "simd128")]
pub fn ipred_cfl_ac<const BPC: u32, const LAYOUT: usize>(
    ac: &mut [i16],
    y: &[u8],
    y_stride: usize,
    w_pad: usize,
    h_pad: usize,
    cw: usize,
    ch: usize,
    bit_depth: BitDepth,
) {
    if !cfl_ac_sozlesmesi::<BPC>(
        LAYOUT,
        ac.len(),
        y.len(),
        y_stride,
        w_pad,
        h_pad,
        cw,
        ch,
        bit_depth,
    ) {
        cfl_ac_skaler_devret::<BPC>(LAYOUT, ac, y, y_stride, w_pad, h_pad, cw, ch, bit_depth);
        return;
    }
    cfl_ac_wasm::<BPC, LAYOUT>(ac, y, y_stride, w_pad, h_pad, cw, ch);
}

#[target_feature(enable = "simd128")]
pub fn ipred_cfl_pred<const BPC: u32, const MODE: usize>(
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
    let mode = cfl_mode_from_const::<MODE>();
    if !cfl_pred_sozlesmesi::<BPC>(
        mode,
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
        cfl_pred_skaler_devret::<BPC>(
            mode,
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
    cfl_pred_wasm::<BPC, MODE>(dst, dst_stride, edge, width, height, ac, alpha);
}

#[target_feature(enable = "simd128")]
fn pal_idx_gecerli_wasm(idx: &[u8], n: usize) -> bool {
    let m88 = u8x16_splat(0x88);
    let mut i = 0usize;
    while i + 16 <= n {
        if v128_any_true(v128_and(load_v128_u8(idx, i), m88)) {
            return false;
        }
        i += 16;
    }
    let mut bad = 0u8;
    while i < n {
        bad |= idx[i] & 0x88;
        i += 1;
    }
    bad == 0
}

#[target_feature(enable = "simd128")]
pub fn pal_pred<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
    bit_depth: BitDepth,
) {
    if !pal_pred_sozlesmesi::<BPC>(dst.len(), dst_stride, pal.len(), idx.len(), w, h, bit_depth)
        || !pal_idx_gecerli_wasm(idx, (w * h) / 2)
    {
        pal_pred_skaler_devret::<BPC>(dst, dst_stride, pal, idx, w, h, bit_depth);
        return;
    }
    pal_pred_wasm::<BPC>(dst, dst_stride, pal, idx, w, h);
}

// ============================================================================
// DC / VERT / HOR
// ============================================================================

#[inline]
#[target_feature(enable = "simd128")]
fn sum_contiguous<const BPC: u32>(buf: &[u8], byte_off: usize, n_px: usize) -> u32 {
    if BPC == 8 {
        match n_px {
            4 => {
                u32::from(buf[byte_off])
                    + u32::from(buf[byte_off + 1])
                    + u32::from(buf[byte_off + 2])
                    + u32::from(buf[byte_off + 3])
            }
            8 => {
                let u16v = u16x8_extend_low_u8x16(load_low64_u8(buf, byte_off));
                hsum_i32x4(u32x4_extadd_pairwise_u16x8(u16v)) as u32
            }
            _ => {
                let mut acc = i32x4_splat(0);
                let mut x = 0usize;
                while x < n_px {
                    let v = load_v128_u8(buf, byte_off + x);
                    acc = i32x4_add(
                        acc,
                        u32x4_extadd_pairwise_u16x8(u16x8_extadd_pairwise_u8x16(v)),
                    );
                    x += 16;
                }
                hsum_i32x4(acc) as u32
            }
        }
    } else if n_px == 4 {
        let v = load_pixels_4::<BPC>(buf, byte_off);
        hsum_i32x4(u32x4_extend_low_u16x8(v)) as u32
    } else {
        let mut acc = i32x4_splat(0);
        let mut x = 0usize;
        while x < n_px {
            let v = load_v128_u8(buf, byte_off + x * 2);
            acc = i32x4_add(acc, u32x4_extadd_pairwise_u16x8(v));
            x += 8;
        }
        hsum_i32x4(acc) as u32
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn dc_gen_top_wasm<const BPC: u32>(edge: Edge<'_>, w: usize) -> u32 {
    let sum = sum_contiguous::<BPC>(edge.bytes, edge.topleft + bpp_of(BPC), w);
    (sum + (w as u32 >> 1)) >> w.trailing_zeros()
}

#[inline]
#[target_feature(enable = "simd128")]
fn dc_gen_left_wasm<const BPC: u32>(edge: Edge<'_>, h: usize) -> u32 {
    let bpp = bpp_of(BPC);
    let sum = sum_contiguous::<BPC>(edge.bytes, edge.topleft - h * bpp, h);
    (sum + (h as u32 >> 1)) >> h.trailing_zeros()
}

#[inline]
#[target_feature(enable = "simd128")]
fn dc_gen_wasm<const BPC: u32>(edge: Edge<'_>, w: usize, h: usize) -> u32 {
    let bpp = bpp_of(BPC);
    let mut dc = ((w + h) >> 1) as u32
        + sum_contiguous::<BPC>(edge.bytes, edge.topleft + bpp, w)
        + sum_contiguous::<BPC>(edge.bytes, edge.topleft - h * bpp, h);
    dc >>= (w + h).trailing_zeros();
    if w != h {
        let (mul, sh) = if BPC > 8 {
            if w > h * 2 || h > w * 2 {
                (0x6667u64, 17)
            } else {
                (0xAAABu64, 17)
            }
        } else if w > h * 2 || h > w * 2 {
            (0x3334u64, 16)
        } else {
            (0x5556u64, 16)
        };
        dc = ((u64::from(dc) * mul) >> sh) as u32;
    }
    dc
}

#[inline]
#[target_feature(enable = "simd128")]
fn splat_row<const BPC: u32>(dst: &mut [u8], row_off: usize, w: usize, val: u16) {
    let w_bytes = w * bpp_of(BPC);
    let v16 = if BPC == 8 {
        u8x16_splat(val as u8)
    } else {
        u16x8_splat(val)
    };
    match w_bytes {
        4 => dst[row_off..row_off + 4].copy_from_slice(&[val as u8; 4]),
        8 => store_low64_u8(dst, row_off, v16),
        16 => store_v128_u8(dst, row_off, v16),
        _ => {
            let mut x = 0usize;
            while x < w_bytes {
                store_v128_u8(dst, row_off + x, v16);
                x += 16;
            }
        }
    }
}

#[target_feature(enable = "simd128")]
fn splat_dc_wasm<const BPC: u32>(dst: &mut [u8], stride: usize, w: usize, h: usize, dc: u16) {
    let w_bytes = w * bpp_of(BPC);
    let v16 = if BPC == 8 {
        u8x16_splat(dc as u8)
    } else {
        u16x8_splat(dc)
    };
    match w_bytes {
        4 => {
            let b4 = [dc as u8; 4];
            for y in 0..h {
                dst[y * stride..y * stride + 4].copy_from_slice(&b4);
            }
        }
        8 => {
            for y in 0..h {
                store_low64_u8(dst, y * stride, v16);
            }
        }
        16 => {
            for y in 0..h {
                store_v128_u8(dst, y * stride, v16);
            }
        }
        32 => {
            for y in 0..h {
                let off = y * stride;
                store_v128_u8(dst, off, v16);
                store_v128_u8(dst, off + 16, v16);
            }
        }
        _ => {
            for y in 0..h {
                let row_off = y * stride;
                let mut x = 0usize;
                while x < w_bytes {
                    store_v128_u8(dst, row_off + x, v16);
                    x += 16;
                }
            }
        }
    }
}

#[target_feature(enable = "simd128")]
fn vert_wasm<const BPC: u32>(dst: &mut [u8], stride: usize, edge: Edge<'_>, w: usize, h: usize) {
    let bpp = bpp_of(BPC);
    let top_off = edge.topleft + bpp;
    let w_bytes = w * bpp;
    match w_bytes {
        4 => {
            let mut t = [0u8; 4];
            t.copy_from_slice(&edge.bytes[top_off..top_off + 4]);
            for y in 0..h {
                dst[y * stride..y * stride + 4].copy_from_slice(&t);
            }
        }
        8 => {
            let v0 = load_low64_u8(edge.bytes, top_off);
            for y in 0..h {
                store_low64_u8(dst, y * stride, v0);
            }
        }
        16 => {
            let v0 = load_v128_u8(edge.bytes, top_off);
            for y in 0..h {
                store_v128_u8(dst, y * stride, v0);
            }
        }
        32 => {
            let v0 = load_v128_u8(edge.bytes, top_off);
            let v1 = load_v128_u8(edge.bytes, top_off + 16);
            for y in 0..h {
                let off = y * stride;
                store_v128_u8(dst, off, v0);
                store_v128_u8(dst, off + 16, v1);
            }
        }
        _ => {
            for y in 0..h {
                let row_off = y * stride;
                let mut x = 0usize;
                while x < w_bytes {
                    let v = load_v128_u8(edge.bytes, top_off + x);
                    store_v128_u8(dst, row_off + x, v);
                    x += 16;
                }
            }
        }
    }
}

#[target_feature(enable = "simd128")]
fn hor_wasm<const BPC: u32>(dst: &mut [u8], stride: usize, edge: Edge<'_>, w: usize, h: usize) {
    for y in 0..h {
        let val = read_edge::<BPC>(edge, -(y as isize + 1)) as u16;
        splat_row::<BPC>(dst, y * stride, w, val);
    }
}

// ============================================================================
// PAETH / SMOOTH
// ============================================================================

#[inline]
#[target_feature(enable = "simd128")]
fn paeth8_step(lv: v128, top_v: v128, tl: v128) -> v128 {
    let base = i16x8_sub(i16x8_add(lv, top_v), tl);
    let ld = i16x8_abs(i16x8_sub(lv, base));
    let td = i16x8_abs(i16x8_sub(top_v, base));
    let tld = i16x8_abs(i16x8_sub(tl, base));
    let m_l = v128_and(i16x8_le(ld, td), i16x8_le(ld, tld));
    let m_t = i16x8_le(td, tld);
    v128_bitselect(lv, v128_bitselect(top_v, tl, m_t), m_l)
}

#[target_feature(enable = "simd128")]
fn paeth_wasm<const BPC: u32>(dst: &mut [u8], stride: usize, edge: Edge<'_>, w: usize, h: usize) {
    let bpp = bpp_of(BPC);
    let tl = i16x8_splat(read_edge::<BPC>(edge, 0) as i16);
    let top_base = edge.topleft + bpp;

    if w == 4 {
        let top_v = load_pixels_4::<BPC>(edge.bytes, top_base);
        for y in 0..h {
            let lv = i16x8_splat(read_edge::<BPC>(edge, -(y as isize + 1)) as i16);
            store_pixels_4_i16::<BPC>(dst, y * stride, paeth8_step(lv, top_v, tl));
        }
        return;
    }

    for y in 0..h {
        let lv = i16x8_splat(read_edge::<BPC>(edge, -(y as isize + 1)) as i16);
        let row_off = y * stride;
        let mut x = 0usize;
        while x < w {
            let top_v = load_pixels_8::<BPC>(edge.bytes, top_base + x * bpp);
            store_pixels_8_i16::<BPC>(dst, row_off + x * bpp, paeth8_step(lv, top_v, tl));
            x += 8;
        }
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn smooth_step8(
    top_v: v128,
    wh_v: v128,
    r_lo: v128,
    r_hi: v128,
    bot_term: v128,
    wy_v: v128,
    left_v: v128,
) -> v128 {
    let r256 = u32x4_splat(256);
    let mut acc_lo = u32x4_add(
        u32x4_add(r_lo, bot_term),
        u32x4_extmul_low_u16x8(top_v, wy_v),
    );
    acc_lo = u32x4_add(acc_lo, u32x4_extmul_low_u16x8(wh_v, left_v));
    let mut acc_hi = u32x4_add(
        u32x4_add(r_hi, bot_term),
        u32x4_extmul_high_u16x8(top_v, wy_v),
    );
    acc_hi = u32x4_add(acc_hi, u32x4_extmul_high_u16x8(wh_v, left_v));
    let q_lo = u32x4_shr(u32x4_add(acc_lo, r256), 9);
    let q_hi = u32x4_shr(u32x4_add(acc_hi, r256), 9);
    i16x8_narrow_i32x4(q_lo, q_hi)
}

#[target_feature(enable = "simd128")]
fn smooth_wasm<const BPC: u32>(dst: &mut [u8], stride: usize, edge: Edge<'_>, w: usize, h: usize) {
    let bpp = bpp_of(BPC);
    let wh = &SM_WEIGHTS[w..w + w];
    let wv = &SM_WEIGHTS[h..h + h];
    let right = read_edge::<BPC>(edge, w as isize) as u16;
    let bottom = read_edge::<BPC>(edge, -(h as isize)) as u32;
    let top_base = edge.topleft + bpp;
    let c256 = u16x8_splat(256);

    if w == 4 {
        let top_v = load_pixels_4::<BPC>(edge.bytes, top_base);
        let wh_v = u16x8_extend_low_u8x16(load_low32_u8(wh, 0));
        let inv_wh_v = u16x8_sub(c256, wh_v);
        let right_term = u32x4_extmul_low_u16x8(inv_wh_v, u16x8_splat(right));
        let r256 = u32x4_splat(256);

        for y in 0..h {
            let wy = u32::from(wv[y]);
            let left = read_edge::<BPC>(edge, -(y as isize + 1)) as u16;
            let bot_term = u32x4_splat(bottom * (256 - wy));
            let mut acc = u32x4_add(right_term, bot_term);
            acc = u32x4_add(acc, u32x4_extmul_low_u16x8(top_v, u16x8_splat(wy as u16)));
            acc = u32x4_add(acc, u32x4_extmul_low_u16x8(wh_v, u16x8_splat(left)));
            let q = u32x4_shr(u32x4_add(acc, r256), 9);
            store_pixels_4_i16::<BPC>(dst, y * stride, i16x8_narrow_i32x4(q, q));
        }
        return;
    }

    let right_v = u16x8_splat(right);
    for y in 0..h {
        let wy = u32::from(wv[y]);
        let wy_v = u16x8_splat(wy as u16);
        let bot_term = u32x4_splat(bottom * (256 - wy));
        let left_v = u16x8_splat(read_edge::<BPC>(edge, -(y as isize + 1)) as u16);
        let row_off = y * stride;
        let mut x = 0usize;
        while x < w {
            let top_v = load_pixels_8::<BPC>(edge.bytes, top_base + x * bpp);
            let wh_v = u16x8_extend_low_u8x16(load_low64_u8(wh, x));
            let inv_wh_v = u16x8_sub(c256, wh_v);
            let r_lo = u32x4_extmul_low_u16x8(inv_wh_v, right_v);
            let r_hi = u32x4_extmul_high_u16x8(inv_wh_v, right_v);
            let out = smooth_step8(top_v, wh_v, r_lo, r_hi, bot_term, wy_v, left_v);
            store_pixels_8_i16::<BPC>(dst, row_off + x * bpp, out);
            x += 8;
        }
    }
}

#[target_feature(enable = "simd128")]
fn smooth_v_wasm<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    w: usize,
    h: usize,
) {
    let bpp = bpp_of(BPC);
    let wv = &SM_WEIGHTS[h..h + h];
    let bottom = read_edge::<BPC>(edge, -(h as isize)) as u32;
    let top_base = edge.topleft + bpp;
    let r128 = u32x4_splat(128);

    if w == 4 {
        let top_v = load_pixels_4::<BPC>(edge.bytes, top_base);
        for y in 0..h {
            let wy = u32::from(wv[y]);
            let bot_term = u32x4_splat(bottom * (256 - wy));
            let acc = u32x4_add(
                bot_term,
                u32x4_extmul_low_u16x8(top_v, u16x8_splat(wy as u16)),
            );
            let q = u32x4_shr(u32x4_add(acc, r128), 8);
            store_pixels_4_i16::<BPC>(dst, y * stride, i16x8_narrow_i32x4(q, q));
        }
        return;
    }

    for y in 0..h {
        let wy = u32::from(wv[y]);
        let wy_v = u16x8_splat(wy as u16);
        let bot_term = u32x4_splat(bottom * (256 - wy));
        let row_off = y * stride;
        let mut x = 0usize;
        while x < w {
            let top_v = load_pixels_8::<BPC>(edge.bytes, top_base + x * bpp);
            let acc_lo = u32x4_add(bot_term, u32x4_extmul_low_u16x8(top_v, wy_v));
            let acc_hi = u32x4_add(bot_term, u32x4_extmul_high_u16x8(top_v, wy_v));
            let q_lo = u32x4_shr(u32x4_add(acc_lo, r128), 8);
            let q_hi = u32x4_shr(u32x4_add(acc_hi, r128), 8);
            store_pixels_8_i16::<BPC>(dst, row_off + x * bpp, i16x8_narrow_i32x4(q_lo, q_hi));
            x += 8;
        }
    }
}

#[target_feature(enable = "simd128")]
fn smooth_h_wasm<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    w: usize,
    h: usize,
) {
    let bpp = bpp_of(BPC);
    let wh = &SM_WEIGHTS[w..w + w];
    let right = read_edge::<BPC>(edge, w as isize) as u16;
    let c256 = u16x8_splat(256);
    let r128 = u32x4_splat(128);

    if w == 4 {
        let wh_v = u16x8_extend_low_u8x16(load_low32_u8(wh, 0));
        let inv_wh_v = u16x8_sub(c256, wh_v);
        let right_term = u32x4_extmul_low_u16x8(inv_wh_v, u16x8_splat(right));
        for y in 0..h {
            let left = read_edge::<BPC>(edge, -(y as isize + 1)) as u16;
            let acc = u32x4_add(right_term, u32x4_extmul_low_u16x8(wh_v, u16x8_splat(left)));
            let q = u32x4_shr(u32x4_add(acc, r128), 8);
            store_pixels_4_i16::<BPC>(dst, y * stride, i16x8_narrow_i32x4(q, q));
        }
        return;
    }

    let right_v = u16x8_splat(right);
    for y in 0..h {
        let left_v = u16x8_splat(read_edge::<BPC>(edge, -(y as isize + 1)) as u16);
        let row_off = y * stride;
        let mut x = 0usize;
        while x < w {
            let wh_v = u16x8_extend_low_u8x16(load_low64_u8(wh, x));
            let inv_wh_v = u16x8_sub(c256, wh_v);
            let r_lo = u32x4_extmul_low_u16x8(inv_wh_v, right_v);
            let r_hi = u32x4_extmul_high_u16x8(inv_wh_v, right_v);
            let acc_lo = u32x4_add(r_lo, u32x4_extmul_low_u16x8(wh_v, left_v));
            let acc_hi = u32x4_add(r_hi, u32x4_extmul_high_u16x8(wh_v, left_v));
            let q_lo = u32x4_shr(u32x4_add(acc_lo, r128), 8);
            let q_hi = u32x4_shr(u32x4_add(acc_hi, r128), 8);
            store_pixels_8_i16::<BPC>(dst, row_off + x * bpp, i16x8_narrow_i32x4(q_lo, q_hi));
            x += 8;
        }
    }
}

// ============================================================================
// Yönlü modlar (Z1 / Z2 / Z3) ve FILTER_PRED
// ============================================================================

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

#[target_feature(enable = "simd128")]
fn filter_edge_i16<const BPC: u32>(
    out: &mut [i16],
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
    let bpp = bpp_of(BPC);
    let copy_end = if lim_from < 0 {
        0
    } else {
        sz.min(lim_from as usize)
    };
    let mut i = 0usize;
    while i < copy_end {
        let pos = (i as isize).max(from).min(to - 1);
        out[i] = read_edge::<BPC>(edge, in_base + pos) as i16;
        i += 1;
    }
    let filter_end = if lim_to < 0 {
        0
    } else {
        sz.min(lim_to as usize)
    };

    let safe_start = (from + 2).max(0) as usize;
    let safe_limit = if to >= 10 { (to - 9) as usize } else { 0 };
    while i < filter_end && i < safe_start {
        let mut sum = 0i32;
        for (j, &w) in KERNEL[strength - 1].iter().enumerate() {
            let pos = (i as isize - 2 + j as isize).max(from).min(to - 1);
            sum += read_edge::<BPC>(edge, in_base + pos) * w;
        }
        out[i] = ((sum + 8) >> 4) as i16;
        i += 1;
    }

    let c2 = u16x8_splat(2);
    let c4 = u16x8_splat(4);
    let c5 = u16x8_splat(5);
    let c6 = u16x8_splat(6);
    let c8 = u16x8_splat(8);

    while i + 8 <= filter_end && i < safe_limit {
        let base_px = in_base + i as isize;
        let byte_0 = (edge.topleft as isize + base_px * bpp as isize) as usize;
        let p_m1 = load_pixels_8::<BPC>(edge.bytes, byte_0 - bpp);
        let p_0 = load_pixels_8::<BPC>(edge.bytes, byte_0);
        let p_p1 = load_pixels_8::<BPC>(edge.bytes, byte_0 + bpp);
        let res = match strength {
            1 => {
                let s = u16x8_add(u16x8_add(p_m1, p_p1), u16x8_shl(p_0, 1));
                u16x8_shr(u16x8_add(s, c2), 2)
            }
            2 => {
                let s = u16x8_add(u16x8_mul(u16x8_add(p_m1, p_p1), c5), u16x8_mul(p_0, c6));
                u16x8_shr(u16x8_add(s, c8), 4)
            }
            _ => {
                let p_m2 = load_pixels_8::<BPC>(edge.bytes, byte_0 - 2 * bpp);
                let p_p2 = load_pixels_8::<BPC>(edge.bytes, byte_0 + 2 * bpp);
                let outer = u16x8_add(p_m2, p_p2);
                let inner = u16x8_shl(u16x8_add(u16x8_add(p_m1, p_0), p_p1), 1);
                u16x8_shr(u16x8_add(u16x8_add(outer, inner), c4), 3)
            }
        };
        store_i16x8(out, i, res);
        i += 8;
    }

    while i < filter_end {
        let mut sum = 0i32;
        for (j, &w) in KERNEL[strength - 1].iter().enumerate() {
            let pos = (i as isize - 2 + j as isize).max(from).min(to - 1);
            sum += read_edge::<BPC>(edge, in_base + pos) * w;
        }
        out[i] = ((sum + 8) >> 4) as i16;
        i += 1;
    }
    while i < sz {
        let pos = (i as isize).max(from).min(to - 1);
        out[i] = read_edge::<BPC>(edge, in_base + pos) as i16;
        i += 1;
    }
}

#[target_feature(enable = "simd128")]
fn upsample_edge_i16<const BPC: u32>(
    out: &mut [i16],
    hsz: usize,
    edge: Edge<'_>,
    in_base: isize,
    from: isize,
    to: isize,
) {
    const KERNEL: [i32; 4] = [-1, 9, 9, -1];
    let bpp = bpp_of(BPC);
    let max = bd_max_of::<BPC>();
    let zero32 = i32x4_splat(0);
    let max32 = i32x4_splat(max);
    let r8 = i32x4_splat(8);
    let c9 = u32x4_splat(9);

    let safe_start = (from + 1).max(0) as usize;
    let safe_limit = if to >= 10 { (to - 9) as usize } else { 0 };
    let mut i = 0usize;

    while i < hsz - 1 && i < safe_start {
        let p0 = (i as isize).max(from).min(to - 1);
        out[2 * i] = read_edge::<BPC>(edge, in_base + p0) as i16;
        let mut sum = 0i32;
        for (j, &w) in KERNEL.iter().enumerate() {
            let pos = (i as isize + j as isize - 1).max(from).min(to - 1);
            sum += read_edge::<BPC>(edge, in_base + pos) * w;
        }
        out[2 * i + 1] = ((sum + 8) >> 4).clamp(0, max) as i16;
        i += 1;
    }

    while i + 8 < hsz && i < safe_limit {
        let base_px = in_base + i as isize;
        let byte_0 = (edge.topleft as isize + base_px * bpp as isize) as usize;
        let v_m1 = load_pixels_8::<BPC>(edge.bytes, byte_0 - bpp);
        let v_0 = load_pixels_8::<BPC>(edge.bytes, byte_0);
        let v_1 = load_pixels_8::<BPC>(edge.bytes, byte_0 + bpp);
        let v_2 = load_pixels_8::<BPC>(edge.bytes, byte_0 + 2 * bpp);

        let mid_lo = u32x4_add(u32x4_extend_low_u16x8(v_0), u32x4_extend_low_u16x8(v_1));
        let mid_hi = u32x4_add(u32x4_extend_high_u16x8(v_0), u32x4_extend_high_u16x8(v_1));
        let out_lo = u32x4_add(u32x4_extend_low_u16x8(v_m1), u32x4_extend_low_u16x8(v_2));
        let out_hi = u32x4_add(u32x4_extend_high_u16x8(v_m1), u32x4_extend_high_u16x8(v_2));

        let s_lo = i32x4_sub(u32x4_mul(mid_lo, c9), out_lo);
        let s_hi = i32x4_sub(u32x4_mul(mid_hi, c9), out_hi);
        let q_lo = i32x4_min(i32x4_max(i32x4_shr(i32x4_add(s_lo, r8), 4), zero32), max32);
        let q_hi = i32x4_min(i32x4_max(i32x4_shr(i32x4_add(s_hi, r8), 4), zero32), max32);
        let odd = i16x8_narrow_i32x4(q_lo, q_hi);
        let lo_zip = u16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(v_0, odd);
        let hi_zip = u16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(v_0, odd);
        store_i16x8(out, 2 * i, lo_zip);
        store_i16x8(out, 2 * i + 8, hi_zip);
        i += 8;
    }

    while i < hsz - 1 {
        let p0 = (i as isize).max(from).min(to - 1);
        out[2 * i] = read_edge::<BPC>(edge, in_base + p0) as i16;
        let mut sum = 0i32;
        for (j, &w) in KERNEL.iter().enumerate() {
            let pos = (i as isize + j as isize - 1).max(from).min(to - 1);
            sum += read_edge::<BPC>(edge, in_base + pos) * w;
        }
        out[2 * i + 1] = ((sum + 8) >> 4).clamp(0, max) as i16;
        i += 1;
    }
    let plast = (hsz as isize - 1).max(from).min(to - 1);
    out[2 * (hsz - 1)] = read_edge::<BPC>(edge, in_base + plast) as i16;
}

#[inline]
#[target_feature(enable = "simd128")]
fn interp8_u16(b0: v128, b1: v128, w0v: v128, w1v: v128) -> v128 {
    let r32 = u32x4_splat(32);
    let lo = u32x4_add(
        u32x4_extmul_low_u16x8(b0, w0v),
        u32x4_extmul_low_u16x8(b1, w1v),
    );
    let hi = u32x4_add(
        u32x4_extmul_high_u16x8(b0, w0v),
        u32x4_extmul_high_u16x8(b1, w1v),
    );
    let q_lo = u32x4_shr(u32x4_add(lo, r32), 6);
    let q_hi = u32x4_shr(u32x4_add(hi, r32), 6);
    i16x8_narrow_i32x4(q_lo, q_hi)
}

#[inline]
#[target_feature(enable = "simd128")]
fn copy_edge_contiguous_i16<const BPC: u32>(
    out: &mut [i16],
    edge: Edge<'_>,
    byte_start: usize,
    len: usize,
) {
    let bpp = bpp_of(BPC);
    let mut i = 0usize;
    while i + 8 <= len {
        let v = load_pixels_8::<BPC>(edge.bytes, byte_start + i * bpp);
        store_i16x8(out, i, v);
        i += 8;
    }
    while i < len {
        out[i] = load_scalar::<BPC>(edge.bytes, byte_start + i * bpp) as i16;
        i += 1;
    }
}

#[target_feature(enable = "simd128")]
fn z1_wasm<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    w: usize,
    h: usize,
    angle_in: usize,
) {
    let bpp = bpp_of(BPC);
    let angle = angle_in & 511;
    let is_sm = ((angle_in >> 9) & 1) == 1;
    let enable = (angle_in >> 10) != 0;
    let mut dx = i32::from(DR_INTRA_DERIVATIVE[angle >> 1]);
    let mut top_out = [0i16; 288];
    let upsample = enable && get_upsample(w + h, 90 - angle, is_sm);

    let (base_inc, max_base_x) = if upsample {
        upsample_edge_i16::<BPC>(&mut top_out, w + h, edge, 1, -1, (w + w.min(h)) as isize);
        dx <<= 1;
        (2usize, 2 * (w + h) - 2)
    } else {
        let strength = if enable {
            get_filter_strength(w + h, 90 - angle, is_sm)
        } else {
            0
        };
        if strength > 0 {
            filter_edge_i16::<BPC>(
                &mut top_out,
                w + h,
                0,
                (w + h) as isize,
                edge,
                1,
                -1,
                (w + w.min(h)) as isize,
                strength,
            );
            (1usize, w + h - 1)
        } else {
            let len = w + w.min(h);
            copy_edge_contiguous_i16::<BPC>(&mut top_out, edge, edge.topleft + bpp, len);
            (1usize, len - 1)
        }
    };

    let fill = top_out[max_base_x];
    let tail_end = (max_base_x + 18).min(top_out.len());
    for cell in &mut top_out[max_base_x + 1..tail_end] {
        *cell = fill;
    }

    let mut xpos = dx;
    for y in 0..h {
        let frac = (xpos & 0x3E) as u16;
        let w0v = u16x8_splat(64 - frac);
        let w1v = u16x8_splat(frac);
        let base = ((xpos >> 6) as usize).min(max_base_x + 1);
        let row_off = y * stride;

        if base >= max_base_x {
            splat_row::<BPC>(dst, row_off, w, fill as u16);
        } else if base_inc == 1 {
            if w == 4 {
                let b0 = load_i16x8(&top_out, base);
                let b1 = load_i16x8(&top_out, base + 1);
                store_pixels_4_i16::<BPC>(dst, row_off, interp8_u16(b0, b1, w0v, w1v));
            } else {
                let mut x = 0usize;
                while x < w {
                    let idx = (base + x).min(max_base_x + 1);
                    let b0 = load_i16x8(&top_out, idx);
                    let b1 = load_i16x8(&top_out, idx + 1);
                    store_pixels_8_i16::<BPC>(
                        dst,
                        row_off + x * bpp,
                        interp8_u16(b0, b1, w0v, w1v),
                    );
                    x += 8;
                }
            }
        } else if w == 4 {
            let raw = load_i16x8(&top_out, base);
            let b0 = u16x8_shuffle::<0, 2, 4, 6, 0, 2, 4, 6>(raw, raw);
            let b1 = u16x8_shuffle::<1, 3, 5, 7, 1, 3, 5, 7>(raw, raw);
            store_pixels_4_i16::<BPC>(dst, row_off, interp8_u16(b0, b1, w0v, w1v));
        } else {
            let mut x = 0usize;
            while x < w {
                let idx = (base + 2 * x).min(max_base_x + 1);
                let r0 = load_i16x8(&top_out, idx);
                let r1 = load_i16x8(&top_out, idx + 8);
                let b0 = u16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(r0, r1);
                let b1 = u16x8_shuffle::<1, 3, 5, 7, 9, 11, 13, 15>(r0, r1);
                store_pixels_8_i16::<BPC>(dst, row_off + x * bpp, interp8_u16(b0, b1, w0v, w1v));
                x += 8;
            }
        }
        xpos += dx;
    }
}

#[target_feature(enable = "simd128")]
fn z2_wasm<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    w: usize,
    h: usize,
    angle_in: usize,
    max_width: usize,
    max_height: usize,
) {
    let bpp = bpp_of(BPC);
    let angle = angle_in & 511;
    let is_sm = ((angle_in >> 9) & 1) == 1;
    let enable = (angle_in >> 10) != 0;
    let mut dy = i32::from(DR_INTRA_DERIVATIVE[(angle - 90) >> 1]);
    let mut dx = i32::from(DR_INTRA_DERIVATIVE[(180 - angle) >> 1]);
    let upsample_left = enable && get_upsample(w + h, 180 - angle, is_sm);
    let upsample_above = enable && get_upsample(w + h, angle - 90, is_sm);
    let mut buffer = [0i16; TMP_LEN];
    let base = TMP_BASE;

    if upsample_above {
        upsample_edge_i16::<BPC>(
            &mut buffer[base as usize..],
            w + 1,
            edge,
            0,
            0,
            (w + 1) as isize,
        );
        dx <<= 1;
    } else {
        let strength = if enable {
            get_filter_strength(w + h, angle - 90, is_sm)
        } else {
            0
        };
        if strength > 0 {
            filter_edge_i16::<BPC>(
                &mut buffer[(base + 1) as usize..],
                w,
                0,
                max_width as isize,
                edge,
                1,
                -1,
                w as isize,
                strength,
            );
        } else {
            copy_edge_contiguous_i16::<BPC>(
                &mut buffer[(base + 1) as usize..],
                edge,
                edge.topleft + bpp,
                w,
            );
        }
    }

    if upsample_left {
        upsample_edge_i16::<BPC>(
            &mut buffer[(base - 2 * h as isize) as usize..],
            h + 1,
            edge,
            -(h as isize),
            0,
            (h + 1) as isize,
        );
        dy <<= 1;
    } else {
        let strength = if enable {
            get_filter_strength(w + h, 180 - angle, is_sm)
        } else {
            0
        };
        if strength > 0 {
            filter_edge_i16::<BPC>(
                &mut buffer[(base - h as isize) as usize..],
                h,
                h as isize - max_height as isize,
                h as isize,
                edge,
                -(h as isize),
                0,
                (h + 1) as isize,
                strength,
            );
        } else {
            copy_edge_contiguous_i16::<BPC>(
                &mut buffer[(base - h as isize) as usize..],
                edge,
                edge.topleft - h * bpp,
                h,
            );
        }
    }
    buffer[base as usize] = read_edge::<BPC>(edge, 0) as i16;

    let base_inc_x = 1 + isize::from(upsample_above);
    let left = base - (1 + isize::from(upsample_left));
    let mut xpos = ((1 + isize::from(upsample_above)) << 6) - dx as isize;

    let mut y_step_shift = [0isize; 64];
    let mut y_w0 = [0u16; 64];
    let mut y_w1 = [0u16; 64];
    for k in 0..w {
        let yk = -dy - (k as i32) * dy;
        y_step_shift[k] = (yk >> 6) as isize;
        let fy = (yk & 0x3E) as u16;
        y_w1[k] = fy;
        y_w0[k] = 64 - fy;
    }

    for y in 0..h {
        let mut base_x = xpos >> 6;
        let frac_x = (xpos & 0x3E) as i32;
        let row_left = left - ((y as isize) << u32::from(upsample_left));
        let row_off = y * stride;
        let mut x = 0usize;

        while x + 8 <= w && base_x + 7 * base_inc_x < 0 {
            let i0 = (row_left - y_step_shift[x]) as usize;
            let i1 = (row_left - y_step_shift[x + 1]) as usize;
            let i2 = (row_left - y_step_shift[x + 2]) as usize;
            let i3 = (row_left - y_step_shift[x + 3]) as usize;
            let i4 = (row_left - y_step_shift[x + 4]) as usize;
            let i5 = (row_left - y_step_shift[x + 5]) as usize;
            let i6 = (row_left - y_step_shift[x + 6]) as usize;
            let i7 = (row_left - y_step_shift[x + 7]) as usize;
            let b0 = i16x8(
                buffer[i0], buffer[i1], buffer[i2], buffer[i3], buffer[i4], buffer[i5], buffer[i6],
                buffer[i7],
            );
            let b1 = i16x8(
                buffer[i0 - 1],
                buffer[i1 - 1],
                buffer[i2 - 1],
                buffer[i3 - 1],
                buffer[i4 - 1],
                buffer[i5 - 1],
                buffer[i6 - 1],
                buffer[i7 - 1],
            );
            let w0v = load_u16x8(&y_w0, x);
            let w1v = load_u16x8(&y_w1, x);
            store_pixels_8_i16::<BPC>(dst, row_off + x * bpp, interp8_u16(b0, b1, w0v, w1v));
            base_x += 8 * base_inc_x;
            x += 8;
        }

        while x < w && base_x < 0 {
            let idx = (row_left - y_step_shift[x]) as usize;
            let val = i32::from(buffer[idx]) * i32::from(y_w0[x])
                + i32::from(buffer[idx - 1]) * i32::from(y_w1[x]);
            store_scalar::<BPC>(dst, row_off + x * bpp, (val + 32) >> 6);
            base_x += base_inc_x;
            x += 1;
        }

        let w0v = u16x8_splat((64 - frac_x) as u16);
        let w1v = u16x8_splat(frac_x as u16);
        if base_inc_x == 1 {
            while x + 8 <= w {
                let idx = (base + base_x) as usize;
                let b0 = load_i16x8(&buffer, idx);
                let b1 = load_i16x8(&buffer, idx + 1);
                store_pixels_8_i16::<BPC>(dst, row_off + x * bpp, interp8_u16(b0, b1, w0v, w1v));
                base_x += 8;
                x += 8;
            }
        } else {
            while x + 8 <= w {
                let idx = (base + base_x) as usize;
                let r0 = load_i16x8(&buffer, idx);
                let r1 = load_i16x8(&buffer, idx + 8);
                let b0 = u16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(r0, r1);
                let b1 = u16x8_shuffle::<1, 3, 5, 7, 9, 11, 13, 15>(r0, r1);
                store_pixels_8_i16::<BPC>(dst, row_off + x * bpp, interp8_u16(b0, b1, w0v, w1v));
                base_x += 16;
                x += 8;
            }
        }

        while x < w {
            let idx = (base + base_x) as usize;
            let val = i32::from(buffer[idx]) * (64 - frac_x) + i32::from(buffer[idx + 1]) * frac_x;
            store_scalar::<BPC>(dst, row_off + x * bpp, (val + 32) >> 6);
            base_x += base_inc_x;
            x += 1;
        }
        xpos -= dx as isize;
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn transpose_store_4x4_low_u16<const BPC: u32>(
    dst: &mut [u8],
    base_off: usize,
    stride: usize,
    c0: v128,
    c1: v128,
    c2: v128,
    c3: v128,
) {
    let t0 = u16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(c0, c1);
    let t1 = u16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(c2, c3);
    let r0 = u16x8_shuffle::<0, 1, 8, 9, 0, 1, 8, 9>(t0, t1);
    let r1 = u16x8_shuffle::<2, 3, 10, 11, 2, 3, 10, 11>(t0, t1);
    let r2 = u16x8_shuffle::<4, 5, 12, 13, 4, 5, 12, 13>(t0, t1);
    let r3 = u16x8_shuffle::<6, 7, 14, 15, 6, 7, 14, 15>(t0, t1);
    store_pixels_4_i16::<BPC>(dst, base_off, r0);
    store_pixels_4_i16::<BPC>(dst, base_off + stride, r1);
    store_pixels_4_i16::<BPC>(dst, base_off + 2 * stride, r2);
    store_pixels_4_i16::<BPC>(dst, base_off + 3 * stride, r3);
}

#[inline]
#[target_feature(enable = "simd128")]
fn transpose_store_4x4_high_u16<const BPC: u32>(
    dst: &mut [u8],
    base_off: usize,
    stride: usize,
    c0: v128,
    c1: v128,
    c2: v128,
    c3: v128,
) {
    let t0 = u16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(c0, c1);
    let t1 = u16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(c2, c3);
    let r0 = u16x8_shuffle::<0, 1, 8, 9, 0, 1, 8, 9>(t0, t1);
    let r1 = u16x8_shuffle::<2, 3, 10, 11, 2, 3, 10, 11>(t0, t1);
    let r2 = u16x8_shuffle::<4, 5, 12, 13, 4, 5, 12, 13>(t0, t1);
    let r3 = u16x8_shuffle::<6, 7, 14, 15, 6, 7, 14, 15>(t0, t1);
    store_pixels_4_i16::<BPC>(dst, base_off, r0);
    store_pixels_4_i16::<BPC>(dst, base_off + stride, r1);
    store_pixels_4_i16::<BPC>(dst, base_off + 2 * stride, r2);
    store_pixels_4_i16::<BPC>(dst, base_off + 3 * stride, r3);
}

#[inline]
#[target_feature(enable = "simd128")]
fn z3_col8(
    rev_left: &[i16; 288],
    ypos: i32,
    y: usize,
    base_inc: usize,
    max_base_y: usize,
    fill_v: v128,
) -> v128 {
    let frac = (ypos & 0x3E) as u16;
    let w0v = u16x8_splat(64 - frac);
    let w1v = u16x8_splat(frac);
    let base = ((ypos >> 6) as usize).min(max_base_y + 1);
    if base >= max_base_y {
        fill_v
    } else if base_inc == 1 {
        let idx = (base + y).min(max_base_y + 1);
        let b0 = load_i16x8(rev_left, idx);
        let b1 = load_i16x8(rev_left, idx + 1);
        interp8_u16(b0, b1, w0v, w1v)
    } else {
        let idx = (base + 2 * y).min(max_base_y + 1);
        let r0 = load_i16x8(rev_left, idx);
        let r1 = load_i16x8(rev_left, idx + 8);
        let b0 = u16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(r0, r1);
        let b1 = u16x8_shuffle::<1, 3, 5, 7, 9, 11, 13, 15>(r0, r1);
        interp8_u16(b0, b1, w0v, w1v)
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn z3_col4(
    rev_left: &[i16; 288],
    ypos: i32,
    base_inc: usize,
    max_base_y: usize,
    fill_v: v128,
) -> v128 {
    let frac = (ypos & 0x3E) as u16;
    let w0v = u16x8_splat(64 - frac);
    let w1v = u16x8_splat(frac);
    let base = ((ypos >> 6) as usize).min(max_base_y + 1);
    if base >= max_base_y {
        fill_v
    } else if base_inc == 1 {
        let b0 = load_i16x8(rev_left, base);
        let b1 = load_i16x8(rev_left, base + 1);
        interp8_u16(b0, b1, w0v, w1v)
    } else {
        let raw = load_i16x8(rev_left, base);
        let b0 = u16x8_shuffle::<0, 2, 4, 6, 0, 2, 4, 6>(raw, raw);
        let b1 = u16x8_shuffle::<1, 3, 5, 7, 1, 3, 5, 7>(raw, raw);
        interp8_u16(b0, b1, w0v, w1v)
    }
}

#[target_feature(enable = "simd128")]
fn z3_wasm<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    w: usize,
    h: usize,
    angle_in: usize,
) {
    let bpp = bpp_of(BPC);
    let angle = angle_in & 511;
    let is_sm = ((angle_in >> 9) & 1) == 1;
    let enable = (angle_in >> 10) != 0;
    let mut dy = i32::from(DR_INTRA_DERIVATIVE[(270 - angle) >> 1]);
    let upsample = enable && get_upsample(w + h, angle - 180, is_sm);
    let strength = if enable && !upsample {
        get_filter_strength(w + h, angle - 180, is_sm)
    } else {
        0
    };
    let direct = !upsample && strength == 0;

    let mut rev_left = [0i16; 288];
    let (base_inc, max_base_y) = if upsample {
        let mut left_raw = [0i16; 288];
        upsample_edge_i16::<BPC>(
            &mut left_raw,
            w + h,
            edge,
            -((w + h) as isize),
            w.saturating_sub(h) as isize,
            (w + h + 1) as isize,
        );
        let m = 2 * (w + h) - 2;
        for i in 0..=m {
            rev_left[i] = left_raw[m - i];
        }
        dy <<= 1;
        (2usize, m)
    } else if !direct {
        let mut left_raw = [0i16; 144];
        filter_edge_i16::<BPC>(
            &mut left_raw,
            w + h,
            0,
            (w + h) as isize,
            edge,
            -((w + h) as isize),
            w.saturating_sub(h) as isize,
            (w + h + 1) as isize,
            strength,
        );
        let m = w + h - 1;
        for i in 0..=m {
            rev_left[i] = left_raw[m - i];
        }
        (1usize, m)
    } else {
        let m = h + w.min(h) - 1;
        for i in 0..=m {
            rev_left[i] = read_edge::<BPC>(edge, -(1 + i as isize)) as i16;
        }
        (1usize, m)
    };

    let fill = rev_left[max_base_y];
    let tail_end = (max_base_y + 18).min(rev_left.len());
    for cell in &mut rev_left[max_base_y + 1..tail_end] {
        *cell = fill;
    }

    let fill8 = u16x8_splat(fill as u16);
    let mut x = 0usize;
    if h == 4 {
        while x < w {
            let c0 = z3_col4(
                &rev_left,
                ((x + 1) as i32) * dy,
                base_inc,
                max_base_y,
                fill8,
            );
            let c1 = z3_col4(
                &rev_left,
                ((x + 2) as i32) * dy,
                base_inc,
                max_base_y,
                fill8,
            );
            let c2 = z3_col4(
                &rev_left,
                ((x + 3) as i32) * dy,
                base_inc,
                max_base_y,
                fill8,
            );
            let c3 = z3_col4(
                &rev_left,
                ((x + 4) as i32) * dy,
                base_inc,
                max_base_y,
                fill8,
            );
            transpose_store_4x4_low_u16::<BPC>(dst, x * bpp, stride, c0, c1, c2, c3);
            x += 4;
        }
    } else {
        while x < w {
            let yp0 = ((x + 1) as i32) * dy;
            let yp1 = ((x + 2) as i32) * dy;
            let yp2 = ((x + 3) as i32) * dy;
            let yp3 = ((x + 4) as i32) * dy;
            let mut y = 0usize;
            while y < h {
                let c0 = z3_col8(&rev_left, yp0, y, base_inc, max_base_y, fill8);
                let c1 = z3_col8(&rev_left, yp1, y, base_inc, max_base_y, fill8);
                let c2 = z3_col8(&rev_left, yp2, y, base_inc, max_base_y, fill8);
                let c3 = z3_col8(&rev_left, yp3, y, base_inc, max_base_y, fill8);
                let off0 = y * stride + x * bpp;
                transpose_store_4x4_low_u16::<BPC>(dst, off0, stride, c0, c1, c2, c3);
                transpose_store_4x4_high_u16::<BPC>(dst, off0 + 4 * stride, stride, c0, c1, c2, c3);
                y += 8;
            }
            x += 4;
        }
    }
}

#[target_feature(enable = "simd128")]
fn filter_pred_wasm<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    w: usize,
    h: usize,
    angle_in: usize,
) {
    let bpp = bpp_of(BPC);
    let fi = angle_in & 511;
    let t = &FILTER_TAPS_SIMD[fi];
    let t0 = load_i16x8(&t[0], 0);
    let t1 = load_i16x8(&t[1], 0);
    let t2 = load_i16x8(&t[2], 0);
    let t3 = load_i16x8(&t[3], 0);
    let t4 = load_i16x8(&t[4], 0);
    let t5 = load_i16x8(&t[5], 0);
    let t6 = load_i16x8(&t[6], 0);

    let r8 = i32x4_splat(8);
    let v_max = i16x8_splat(bd_max_of::<BPC>() as i16);

    let mut y = 0usize;
    while y < h {
        let mut x = 0usize;
        while x < w {
            let group = x / 4;
            let edge_tl = -(y as isize);
            let (p0, p1, p2, p3, p4) = if y == 0 {
                if group == 0 {
                    (
                        read_edge::<BPC>(edge, edge_tl),
                        read_edge::<BPC>(edge, 1),
                        read_edge::<BPC>(edge, 2),
                        read_edge::<BPC>(edge, 3),
                        read_edge::<BPC>(edge, 4),
                    )
                } else {
                    let col = 4 * group as isize;
                    (
                        read_edge::<BPC>(edge, col),
                        read_edge::<BPC>(edge, col + 1),
                        read_edge::<BPC>(edge, col + 2),
                        read_edge::<BPC>(edge, col + 3),
                        read_edge::<BPC>(edge, col + 4),
                    )
                }
            } else {
                let top_row = (y - 1) * stride;
                let col = 4 * group;
                let p0 = if group == 0 {
                    read_edge::<BPC>(edge, edge_tl)
                } else {
                    load_scalar::<BPC>(dst, top_row + (col - 1) * bpp)
                };
                (
                    p0,
                    load_scalar::<BPC>(dst, top_row + col * bpp),
                    load_scalar::<BPC>(dst, top_row + (col + 1) * bpp),
                    load_scalar::<BPC>(dst, top_row + (col + 2) * bpp),
                    load_scalar::<BPC>(dst, top_row + (col + 3) * bpp),
                )
            };
            let (p5, p6) = if group == 0 {
                (
                    read_edge::<BPC>(edge, edge_tl - 1),
                    read_edge::<BPC>(edge, edge_tl - 2),
                )
            } else {
                (
                    load_scalar::<BPC>(dst, y * stride + (x - 1) * bpp),
                    load_scalar::<BPC>(dst, (y + 1) * stride + (x - 1) * bpp),
                )
            };

            let v0 = i16x8_splat(p0 as i16);
            let v1 = i16x8_splat(p1 as i16);
            let v2 = i16x8_splat(p2 as i16);
            let v3 = i16x8_splat(p3 as i16);
            let v4 = i16x8_splat(p4 as i16);
            let v5 = i16x8_splat(p5 as i16);
            let v6 = i16x8_splat(p6 as i16);

            let mut acc_lo = i32x4_add(r8, i32x4_extmul_low_i16x8(t0, v0));
            acc_lo = i32x4_add(acc_lo, i32x4_extmul_low_i16x8(t1, v1));
            acc_lo = i32x4_add(acc_lo, i32x4_extmul_low_i16x8(t2, v2));
            acc_lo = i32x4_add(acc_lo, i32x4_extmul_low_i16x8(t3, v3));
            acc_lo = i32x4_add(acc_lo, i32x4_extmul_low_i16x8(t4, v4));
            acc_lo = i32x4_add(acc_lo, i32x4_extmul_low_i16x8(t5, v5));
            acc_lo = i32x4_add(acc_lo, i32x4_extmul_low_i16x8(t6, v6));

            let mut acc_hi = i32x4_add(r8, i32x4_extmul_high_i16x8(t0, v0));
            acc_hi = i32x4_add(acc_hi, i32x4_extmul_high_i16x8(t1, v1));
            acc_hi = i32x4_add(acc_hi, i32x4_extmul_high_i16x8(t2, v2));
            acc_hi = i32x4_add(acc_hi, i32x4_extmul_high_i16x8(t3, v3));
            acc_hi = i32x4_add(acc_hi, i32x4_extmul_high_i16x8(t4, v4));
            acc_hi = i32x4_add(acc_hi, i32x4_extmul_high_i16x8(t5, v5));
            acc_hi = i32x4_add(acc_hi, i32x4_extmul_high_i16x8(t6, v6));

            pack_and_store_4::<BPC>(dst, y * stride + x * bpp, i32x4_shr(acc_lo, 4), v_max);
            pack_and_store_4::<BPC>(dst, (y + 1) * stride + x * bpp, i32x4_shr(acc_hi, 4), v_max);
            x += 4;
        }
        y += 2;
    }
}

// ============================================================================
// CfL AC / CfL PRED / PAL_PRED
// ============================================================================

#[inline]
#[target_feature(enable = "simd128")]
fn pair_sum_16_to_8<const BPC: u32>(y: &[u8], off: usize) -> v128 {
    if BPC == 8 {
        u16x8_extadd_pairwise_u8x16(load_v128_u8(y, off))
    } else {
        let a = load_v128_u8(y, off);
        let b = load_v128_u8(y, off + 16);
        i16x8_narrow_i32x4(
            u32x4_extadd_pairwise_u16x8(a),
            u32x4_extadd_pairwise_u16x8(b),
        )
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn cfl_ac_row8<const BPC: u32, const LAYOUT: usize>(
    y: &[u8],
    y0_off: usize,
    y1_off: usize,
    x: usize,
) -> v128 {
    let bpp = bpp_of(BPC);
    if LAYOUT == 0 {
        let r0 = pair_sum_16_to_8::<BPC>(y, y0_off + 2 * x * bpp);
        let r1 = pair_sum_16_to_8::<BPC>(y, y1_off + 2 * x * bpp);
        i16x8_shl(u16x8_add(r0, r1), 1)
    } else if LAYOUT == 1 {
        let r0 = pair_sum_16_to_8::<BPC>(y, y0_off + 2 * x * bpp);
        i16x8_shl(r0, 2)
    } else {
        let r0 = load_pixels_8::<BPC>(y, y0_off + x * bpp);
        i16x8_shl(r0, 3)
    }
}

#[target_feature(enable = "simd128")]
fn cfl_ac_wasm<const BPC: u32, const LAYOUT: usize>(
    ac: &mut [i16],
    y: &[u8],
    y_stride: usize,
    w_pad: usize,
    h_pad: usize,
    cw: usize,
    ch: usize,
) {
    let ss_hor = usize::from(LAYOUT != 2);
    let ss_ver = usize::from(LAYOUT == 0);
    let bpp = bpp_of(BPC);
    let visible_w = cw - 4 * w_pad;
    let visible_h = ch - 4 * h_pad;
    let n = cw * ch;
    let log2sz = cw.trailing_zeros() + ch.trailing_zeros();

    if w_pad == 0 && h_pad == 0 && cw >= 8 {
        let mut acc_s32 = i32x4_splat(0);
        for row in 0..ch {
            let luma_row = row << ss_ver;
            let y0_off = luma_row * y_stride;
            let y1_off = (luma_row + 1) * y_stride;
            let mut x = 0usize;
            while x < cw {
                let sum_s16 = cfl_ac_row8::<BPC, LAYOUT>(y, y0_off, y1_off, x);
                acc_s32 = i32x4_add(acc_s32, i32x4_extadd_pairwise_i16x8(sum_s16));
                store_i16x8(ac, row * cw + x, sum_s16);
                x += 8;
            }
        }
        let sum = ((1i32 << log2sz) >> 1) + hsum_i32x4(acc_s32);
        let avg = i16x8_splat((sum >> log2sz) as i16);
        let mut i = 0usize;
        while i < n {
            let v = i16x8_sub(load_i16x8(ac, i), avg);
            store_i16x8(ac, i, v);
            i += 8;
        }
        return;
    }

    for row in 0..visible_h {
        let luma_row = row << ss_ver;
        let y0_off = luma_row * y_stride;
        let y1_off = (luma_row + 1) * y_stride;
        let mut x = 0usize;

        while x + 8 <= visible_w {
            let sum_s16 = cfl_ac_row8::<BPC, LAYOUT>(y, y0_off, y1_off, x);
            store_i16x8(ac, row * cw + x, sum_s16);
            x += 8;
        }

        while x < visible_w {
            let col = (x << ss_hor) * bpp;
            let mut sum = load_scalar::<BPC>(y, y0_off + col);
            if ss_hor != 0 {
                sum += load_scalar::<BPC>(y, y0_off + col + bpp);
            }
            if ss_ver != 0 {
                sum += load_scalar::<BPC>(y, y1_off + col);
                if ss_hor != 0 {
                    sum += load_scalar::<BPC>(y, y1_off + col + bpp);
                }
            }
            ac[row * cw + x] = (sum << (1 + (1 - ss_ver) + (1 - ss_hor))) as i16;
            x += 1;
        }
        let pad_val = ac[row * cw + visible_w - 1];
        while x < cw {
            ac[row * cw + x] = pad_val;
            x += 1;
        }
    }

    for row in visible_h..ch {
        for x in 0..cw {
            ac[row * cw + x] = ac[(row - 1) * cw + x];
        }
    }

    let mut acc_s32 = i32x4_splat(0);
    let mut i = 0usize;
    while i < n {
        acc_s32 = i32x4_add(acc_s32, i32x4_extadd_pairwise_i16x8(load_i16x8(ac, i)));
        i += 8;
    }
    let sum = ((1i32 << log2sz) >> 1) + hsum_i32x4(acc_s32);
    let avg = i16x8_splat((sum >> log2sz) as i16);
    let mut i = 0usize;
    while i < n {
        let v = i16x8_sub(load_i16x8(ac, i), avg);
        store_i16x8(ac, i, v);
        i += 8;
    }
}

#[target_feature(enable = "simd128")]
fn cfl_pred_wasm<const BPC: u32, const MODE: usize>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    w: usize,
    h: usize,
    ac: &[i16],
    alpha: i32,
) {
    let dc = match MODE {
        0 => dc_gen_wasm::<BPC>(edge, w, h) as i32,
        1 => dc_gen_left_wasm::<BPC>(edge, h) as i32,
        2 => dc_gen_top_wasm::<BPC>(edge, w) as i32,
        _ => 1i32 << (BPC - 1),
    };
    if alpha == 0 {
        splat_dc_wasm::<BPC>(dst, stride, w, h, dc as u16);
        return;
    }

    let bpp = bpp_of(BPC);
    let dc_v = i32x4_splat(dc);
    let alpha_v = i16x8_splat(alpha as i16);
    let zero = i32x4_splat(0);
    let r32 = i32x4_splat(32);
    let v_max = i16x8_splat(bd_max_of::<BPC>() as i16);

    if w == 4 {
        let mut y = 0usize;
        while y + 2 <= h {
            let ac_v = load_i16x8(ac, y * 4);
            let d_lo = i32x4_extmul_low_i16x8(ac_v, alpha_v);
            let d_hi = i32x4_extmul_high_i16x8(ac_v, alpha_v);
            let q_lo = i32x4_shr(i32x4_add(i32x4_abs(d_lo), r32), 6);
            let q_hi = i32x4_shr(i32x4_add(i32x4_abs(d_hi), r32), 6);
            let s_lo = v128_bitselect(i32x4_neg(q_lo), q_lo, i32x4_lt(d_lo, zero));
            let s_hi = v128_bitselect(i32x4_neg(q_hi), q_hi, i32x4_lt(d_hi, zero));
            pack_and_store_4::<BPC>(dst, y * stride, i32x4_add(dc_v, s_lo), v_max);
            pack_and_store_4::<BPC>(dst, (y + 1) * stride, i32x4_add(dc_v, s_hi), v_max);
            y += 2;
        }
        return;
    }

    for y in 0..h {
        let row_off = y * stride;
        let mut x = 0usize;
        while x < w {
            let ac_v = load_i16x8(ac, y * w + x);
            let d_lo = i32x4_extmul_low_i16x8(ac_v, alpha_v);
            let d_hi = i32x4_extmul_high_i16x8(ac_v, alpha_v);
            let q_lo = i32x4_shr(i32x4_add(i32x4_abs(d_lo), r32), 6);
            let q_hi = i32x4_shr(i32x4_add(i32x4_abs(d_hi), r32), 6);
            let s_lo = v128_bitselect(i32x4_neg(q_lo), q_lo, i32x4_lt(d_lo, zero));
            let s_hi = v128_bitselect(i32x4_neg(q_hi), q_hi, i32x4_lt(d_hi, zero));
            pack_and_store_8::<BPC>(
                dst,
                row_off + x * bpp,
                i32x4_add(dc_v, s_lo),
                i32x4_add(dc_v, s_hi),
                v_max,
            );
            x += 8;
        }
    }
}

#[target_feature(enable = "simd128")]
fn pal_pred_wasm<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
) {
    let m07 = u8x16_splat(0x07);
    let half_w = w / 2;

    if BPC == 8 {
        let pal_v = load_low64_u8(pal, 0);
        match w {
            4 => {
                let mut y = 0usize;
                while y + 4 <= h {
                    let raw = load_low64_u8(idx, y * 2);
                    let lo = v128_and(raw, m07);
                    let hi = v128_and(u8x16_shr(raw, 4), m07);
                    let nibs =
                        i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                            lo, hi,
                        );
                    let out = u8x16_swizzle(pal_v, nibs);
                    let lo64 = i64x2_extract_lane::<0>(out) as u64;
                    let hi64 = i64x2_extract_lane::<1>(out) as u64;
                    let off = y * stride;
                    dst[off..off + 4].copy_from_slice(&(lo64 as u32).to_le_bytes());
                    dst[off + stride..off + stride + 4]
                        .copy_from_slice(&((lo64 >> 32) as u32).to_le_bytes());
                    dst[off + 2 * stride..off + 2 * stride + 4]
                        .copy_from_slice(&(hi64 as u32).to_le_bytes());
                    dst[off + 3 * stride..off + 3 * stride + 4]
                        .copy_from_slice(&((hi64 >> 32) as u32).to_le_bytes());
                    y += 4;
                }
            }
            8 => {
                let mut y = 0usize;
                while y + 2 <= h {
                    let raw = load_low64_u8(idx, y * 4);
                    let lo = v128_and(raw, m07);
                    let hi = v128_and(u8x16_shr(raw, 4), m07);
                    let nibs =
                        i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                            lo, hi,
                        );
                    let out = u8x16_swizzle(pal_v, nibs);
                    store_low64_u8(dst, y * stride, out);
                    let hi_half =
                        i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0>(
                            out, out,
                        );
                    store_low64_u8(dst, (y + 1) * stride, hi_half);
                    y += 2;
                }
            }
            _ => {
                for y in 0..h {
                    let row_off = y * stride;
                    let idx_off = y * half_w;
                    let mut x = 0usize;
                    while x < w {
                        let raw = load_low64_u8(idx, idx_off + x / 2);
                        let lo = v128_and(raw, m07);
                        let hi = v128_and(u8x16_shr(raw, 4), m07);
                        let nibs = i8x16_shuffle::<
                            0,
                            16,
                            1,
                            17,
                            2,
                            18,
                            3,
                            19,
                            4,
                            20,
                            5,
                            21,
                            6,
                            22,
                            7,
                            23,
                        >(lo, hi);
                        let out = u8x16_swizzle(pal_v, nibs);
                        store_v128_u8(dst, row_off + x, out);
                        x += 16;
                    }
                }
            }
        }
    } else {
        let pal_v = load_v128_u8(pal, 0);
        let ones = u8x16_splat(1);
        if w == 4 {
            let mut y = 0usize;
            while y + 2 <= h {
                let raw4 = load_low32_u8(idx, y * 2);
                let lo = v128_and(raw4, m07);
                let hi = v128_and(u8x16_shr(raw4, 4), m07);
                let nibs8 =
                    i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 0, 0, 0, 0, 0, 0, 0, 0>(lo, hi);
                let even = u8x16_shl(nibs8, 1);
                let odd = u8x16_add(even, ones);
                let b_idx = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                    even, odd,
                );
                let out = u8x16_swizzle(pal_v, b_idx);
                store_low64_u8(dst, y * stride, out);
                let hi_half =
                    i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0>(out, out);
                store_low64_u8(dst, (y + 1) * stride, hi_half);
                y += 2;
            }
        } else {
            for y in 0..h {
                let row_off = y * stride;
                let idx_off = y * half_w;
                let mut x = 0usize;
                while x < w {
                    let raw4 = load_low32_u8(idx, idx_off + x / 2);
                    let lo = v128_and(raw4, m07);
                    let hi = v128_and(u8x16_shr(raw4, 4), m07);
                    let nibs8 =
                        i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 0, 0, 0, 0, 0, 0, 0, 0>(lo, hi);
                    let even = u8x16_shl(nibs8, 1);
                    let odd = u8x16_add(even, ones);
                    let b_idx =
                        i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                            even, odd,
                        );
                    let out = u8x16_swizzle(pal_v, b_idx);
                    store_v128_u8(dst, row_off + 2 * x, out);
                    x += 8;
                }
            }
        }
    }
}
