//! WebAssembly `simd128` film grain uygulama çekirdekleri (`12` yuva, `8 / 10 / 12` bpc).
//!
//! Kapsam: `fgy_32x32xn` (`3` yuva) ve `fguv_32x32xn[0..2]` (`9` yuva).
//! `%100 #![forbid(unsafe_code)]`.

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use core::arch::wasm32::{
    i8x16_shuffle, i16x8, i16x8_extract_lane, i16x8_max, i16x8_min, i16x8_narrow_i32x4,
    i16x8_splat, i32x4_add, i32x4_extend_high_i16x8, i32x4_extend_low_i16x8,
    i32x4_extmul_high_i16x8, i32x4_extmul_low_i16x8, i32x4_max, i32x4_min, i32x4_shr, i32x4_splat,
    u8x16_avgr, u16x8_avgr, u16x8_extend_low_u8x16, v128,
};

use super::{
    bpp_of, load_i16x8, load_pixels_8, load_scalar, load_v128_u8, store_pixels_8_i16, store_scalar,
};
use crate::bitstream::frame_header::FilmGrainData;
use crate::dsp::fg::{
    FG_BLOCK_SIZE, FgBitDepth, FgDsp, FguvFn, GrainLut, fg_skaler_devri_say, fguv_sozlesmesi,
    fguv16_sozlesmesi, fgy_sozlesmesi, fgy16_sozlesmesi,
};
use crate::dsp::scalar::fg as scalar;

const W_OV: [[[i32; 2]; 2]; 2] = [[[27, 17], [17, 27]], [[23, 22], [0, 0]]];

#[must_use]
pub const fn fguv_fn<const BPC: u32>(layout: usize) -> Option<FguvFn> {
    match layout {
        0 => Some(fguv_32x32xn::<BPC, 1, 1>),
        1 => Some(fguv_32x32xn::<BPC, 1, 0>),
        2 => Some(fguv_32x32xn::<BPC, 0, 0>),
        _ => None,
    }
}

#[inline(always)]
const fn ofset(randval: i32, sx: usize, sy: usize) -> (usize, usize) {
    (
        3 + (2 >> sx) * (3 + (randval >> 4)) as usize,
        3 + (2 >> sy) * (3 + (randval & 0xF)) as usize,
    )
}

#[inline]
#[target_feature(enable = "simd128")]
fn scale_lookup8<const SCALE_N: usize>(idx_v: v128, scaling: &[u8; SCALE_N]) -> v128 {
    let m = SCALE_N - 1;
    let s0 = i16::from(scaling[(i16x8_extract_lane::<0>(idx_v) as u16 as usize) & m]);
    let s1 = i16::from(scaling[(i16x8_extract_lane::<1>(idx_v) as u16 as usize) & m]);
    let s2 = i16::from(scaling[(i16x8_extract_lane::<2>(idx_v) as u16 as usize) & m]);
    let s3 = i16::from(scaling[(i16x8_extract_lane::<3>(idx_v) as u16 as usize) & m]);
    let s4 = i16::from(scaling[(i16x8_extract_lane::<4>(idx_v) as u16 as usize) & m]);
    let s5 = i16::from(scaling[(i16x8_extract_lane::<5>(idx_v) as u16 as usize) & m]);
    let s6 = i16::from(scaling[(i16x8_extract_lane::<6>(idx_v) as u16 as usize) & m]);
    let s7 = i16::from(scaling[(i16x8_extract_lane::<7>(idx_v) as u16 as usize) & m]);
    i16x8(s0, s1, s2, s3, s4, s5, s6, s7)
}

#[inline]
#[target_feature(enable = "simd128")]
fn blend_grain8(g_old: v128, g_new: v128, w0: i16, w1: i16, gmin_v: v128, gmax_v: v128) -> v128 {
    let vw0 = i16x8_splat(w0);
    let vw1 = i16x8_splat(w1);
    let v_rnd = i32x4_splat(16);
    let lo = i32x4_shr(
        i32x4_add(
            i32x4_add(
                i32x4_extmul_low_i16x8(g_old, vw0),
                i32x4_extmul_low_i16x8(g_new, vw1),
            ),
            v_rnd,
        ),
        5,
    );
    let hi = i32x4_shr(
        i32x4_add(
            i32x4_add(
                i32x4_extmul_high_i16x8(g_old, vw0),
                i32x4_extmul_high_i16x8(g_new, vw1),
            ),
            v_rnd,
        ),
        5,
    );
    i16x8_min(i16x8_max(i16x8_narrow_i32x4(lo, hi), gmin_v), gmax_v)
}

#[inline]
#[target_feature(enable = "simd128")]
fn apply_noise8(
    s_v: v128,
    sc_v: v128,
    g_v: v128,
    shift: u32,
    v_rnd: v128,
    min_v: v128,
    max_v: v128,
) -> v128 {
    let n_lo = i32x4_shr(i32x4_add(i32x4_extmul_low_i16x8(sc_v, g_v), v_rnd), shift);
    let n_hi = i32x4_shr(i32x4_add(i32x4_extmul_high_i16x8(sc_v, g_v), v_rnd), shift);
    let r_lo = i32x4_min(
        i32x4_max(i32x4_add(i32x4_extend_low_i16x8(s_v), n_lo), min_v),
        max_v,
    );
    let r_hi = i32x4_min(
        i32x4_max(i32x4_add(i32x4_extend_high_i16x8(s_v), n_hi), min_v),
        max_v,
    );
    i16x8_narrow_i32x4(r_lo, r_hi)
}

#[target_feature(enable = "simd128")]
fn fgy_simd128<const BPC: u32, const SCALE_N: usize>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8; SCALE_N],
    grain_lut: &GrainLut,
    bh: usize,
    row_num: usize,
) {
    let bpp = bpp_of(BPC);
    let max = (1i32 << BPC) - 1;
    let mask = SCALE_N - 1;
    let overlap = data.overlap_flag;
    let rows = 1 + usize::from(overlap && row_num > 0);
    let min_8 = BPC - 8;
    let grain_ctr = 128i32 << min_8;
    let (grain_min, grain_max) = (-grain_ctr, grain_ctr - 1);
    let gmin_v = i16x8_splat(grain_min as i16);
    let gmax_v = i16x8_splat(grain_max as i16);
    let (min_value, max_value) = if data.clip_to_restricted_range {
        (16 << min_8, 235 << min_8)
    } else {
        (0, max)
    };
    let min_v = i32x4_splat(min_value);
    let max_v = i32x4_splat(max_value);
    let shift = data.scaling_shift as u32;
    let v_rnd = i32x4_splat((1i32 << shift) >> 1);
    let mut seed = scalar::row_seeds(data, row_num, rows);
    let mut offsets = [[0i32; 2]; 2];

    let add_scalar = |dst: &mut [u8], x: usize, y: usize, bx: usize, grain: i32| {
        let off = y * stride + (bx + x) * bpp;
        let s = load_scalar::<BPC>(src, off);
        let noise = scalar::round2(i32::from(scaling[(s as usize) & mask]) * grain, shift);
        store_scalar::<BPC>(dst, off, (s + noise).clamp(min_value, max_value));
    };

    let mut bx = 0usize;
    while bx < pw {
        let bw = FG_BLOCK_SIZE.min(pw - bx);
        if overlap && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = scalar::get_random_number(8, &mut seed[i]);
        }
        let ystart = if overlap && row_num != 0 {
            bh.min(2)
        } else {
            0
        };
        let xstart = if overlap && bx != 0 { bw.min(2) } else { 0 };
        let (offx0, offy0) = ofset(offsets[0][0], 0, 0);

        for y in ystart..bh {
            let lut_row = &grain_lut[offy0 + y];
            let mut x = xstart;
            while x + 8 <= bw {
                let byte_off = y * stride + (bx + x) * bpp;
                let s_v = load_pixels_8::<BPC>(src, byte_off);
                let sc_v = scale_lookup8::<SCALE_N>(s_v, scaling);
                let g_v = load_i16x8(lut_row, offx0 + x);
                let out = apply_noise8(s_v, sc_v, g_v, shift, v_rnd, min_v, max_v);
                store_pixels_8_i16::<BPC>(dst, byte_off, out);
                x += 8;
            }
            while x < bw {
                let grain = i32::from(lut_row[offx0 + x]);
                add_scalar(dst, x, y, bx, grain);
                x += 1;
            }
            for x in 0..xstart {
                let grain = scalar::sample_lut(grain_lut, &offsets, 0, 0, 0, 0, x, y);
                let old = scalar::sample_lut(grain_lut, &offsets, 0, 0, 1, 0, x, y);
                let grain = scalar::round2(old * W_OV[0][x][0] + grain * W_OV[0][x][1], 5)
                    .clamp(grain_min, grain_max);
                add_scalar(dst, x, y, bx, grain);
            }
        }

        if ystart > 0 {
            let (offx_top, offy_top) = ofset(offsets[0][1], 0, 0);
            for y in 0..ystart {
                let lut_new = &grain_lut[offy0 + y];
                let lut_old = &grain_lut[offy_top + FG_BLOCK_SIZE + y];
                let w0 = W_OV[0][y][0] as i16;
                let w1 = W_OV[0][y][1] as i16;
                let mut x = xstart;
                while x + 8 <= bw {
                    let byte_off = y * stride + (bx + x) * bpp;
                    let s_v = load_pixels_8::<BPC>(src, byte_off);
                    let sc_v = scale_lookup8::<SCALE_N>(s_v, scaling);
                    let g_new = load_i16x8(lut_new, offx0 + x);
                    let g_old = load_i16x8(lut_old, offx_top + x);
                    let g_v = blend_grain8(g_old, g_new, w0, w1, gmin_v, gmax_v);
                    let out = apply_noise8(s_v, sc_v, g_v, shift, v_rnd, min_v, max_v);
                    store_pixels_8_i16::<BPC>(dst, byte_off, out);
                    x += 8;
                }
                while x < bw {
                    let grain = scalar::sample_lut(grain_lut, &offsets, 0, 0, 0, 0, x, y);
                    let old = scalar::sample_lut(grain_lut, &offsets, 0, 0, 0, 1, x, y);
                    let grain = scalar::round2(old * W_OV[0][y][0] + grain * W_OV[0][y][1], 5)
                        .clamp(grain_min, grain_max);
                    add_scalar(dst, x, y, bx, grain);
                    x += 1;
                }
                for x in 0..xstart {
                    let top = scalar::sample_lut(grain_lut, &offsets, 0, 0, 0, 1, x, y);
                    let old = scalar::sample_lut(grain_lut, &offsets, 0, 0, 1, 1, x, y);
                    let top = scalar::round2(old * W_OV[0][x][0] + top * W_OV[0][x][1], 5)
                        .clamp(grain_min, grain_max);
                    let grain = scalar::sample_lut(grain_lut, &offsets, 0, 0, 0, 0, x, y);
                    let old = scalar::sample_lut(grain_lut, &offsets, 0, 0, 1, 0, x, y);
                    let grain = scalar::round2(old * W_OV[0][x][0] + grain * W_OV[0][x][1], 5)
                        .clamp(grain_min, grain_max);
                    let grain = scalar::round2(top * W_OV[0][y][0] + grain * W_OV[0][y][1], 5)
                        .clamp(grain_min, grain_max);
                    add_scalar(dst, x, y, bx, grain);
                }
            }
        }
        bx += FG_BLOCK_SIZE;
    }
}

pub fn fgy_32x32xn<const BPC: u32>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8],
    grain_lut: &GrainLut,
    bh: usize,
    row_num: usize,
    bd: FgBitDepth,
) {
    let ok = bd.bits() == BPC
        && if BPC == 8 {
            fgy_sozlesmesi(dst.len(), src.len(), stride, pw, bh, data)
        } else {
            fgy16_sozlesmesi(dst.len(), src.len(), stride, pw, bh, data, bd)
        };
    if ok {
        match BPC {
            8 => {
                if let Some(sc) = scaling
                    .get(..256)
                    .and_then(|s| <&[u8; 256]>::try_from(s).ok())
                {
                    return fgy_simd128::<8, 256>(
                        dst, src, stride, data, pw, sc, grain_lut, bh, row_num,
                    );
                }
            }
            10 => {
                if let Some(sc) = scaling
                    .get(..1024)
                    .and_then(|s| <&[u8; 1024]>::try_from(s).ok())
                {
                    return fgy_simd128::<10, 1024>(
                        dst, src, stride, data, pw, sc, grain_lut, bh, row_num,
                    );
                }
            }
            _ => {
                if let Some(sc) = scaling
                    .get(..4096)
                    .and_then(|s| <&[u8; 4096]>::try_from(s).ok())
                {
                    return fgy_simd128::<12, 4096>(
                        dst, src, stride, data, pw, sc, grain_lut, bh, row_num,
                    );
                }
            }
        }
    }
    fg_skaler_devri_say();
    (FgDsp::scalar::<BPC>().fgy_32x32xn)(
        dst, src, stride, data, pw, scaling, grain_lut, bh, row_num, bd,
    );
}

#[inline]
#[target_feature(enable = "simd128")]
fn load_luma_avg8<const BPC: u32, const SX: usize>(luma: &[u8], byte_off: usize) -> v128 {
    if SX == 0 {
        load_pixels_8::<BPC>(luma, byte_off)
    } else if BPC == 8 {
        let raw = load_v128_u8(luma, byte_off);
        let evens = i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14>(raw, raw);
        let odds = i8x16_shuffle::<1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15>(raw, raw);
        u16x8_extend_low_u8x16(u8x16_avgr(evens, odds))
    } else {
        let v0 = load_v128_u8(luma, byte_off);
        let v1 = load_v128_u8(luma, byte_off + 16);
        let evens =
            i8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29>(v0, v1);
        let odds =
            i8x16_shuffle::<2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31>(v0, v1);
        u16x8_avgr(evens, odds)
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn uv_scale_vec8(
    avg_v: v128,
    s_v: v128,
    csfl: bool,
    lm_v: v128,
    um_v: v128,
    off_v: v128,
    scale_max_v: v128,
) -> v128 {
    if csfl {
        avg_v
    } else {
        let zero = i32x4_splat(0);
        let c_lo = i32x4_add(
            i32x4_extmul_low_i16x8(avg_v, lm_v),
            i32x4_extmul_low_i16x8(s_v, um_v),
        );
        let c_hi = i32x4_add(
            i32x4_extmul_high_i16x8(avg_v, lm_v),
            i32x4_extmul_high_i16x8(s_v, um_v),
        );
        let v_lo = i32x4_min(
            i32x4_max(i32x4_add(i32x4_shr(c_lo, 6), off_v), zero),
            scale_max_v,
        );
        let v_hi = i32x4_min(
            i32x4_max(i32x4_add(i32x4_shr(c_hi, 6), off_v), zero),
            scale_max_v,
        );
        i16x8_narrow_i32x4(v_lo, v_hi)
    }
}

#[target_feature(enable = "simd128")]
fn fguv_simd128<const BPC: u32, const SX: usize, const SY: usize, const SCALE_N: usize>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8; SCALE_N],
    grain_lut: &GrainLut,
    bh: usize,
    row_num: usize,
    luma: &[u8],
    luma_stride: usize,
    uv: usize,
    is_id: bool,
) {
    let bpp = bpp_of(BPC);
    let max = (1i32 << BPC) - 1;
    let mask = SCALE_N - 1;
    let overlap = data.overlap_flag;
    let rows = 1 + usize::from(overlap && row_num > 0);
    let min_8 = BPC - 8;
    let grain_ctr = 128i32 << min_8;
    let (grain_min, grain_max) = (-grain_ctr, grain_ctr - 1);
    let gmin_v = i16x8_splat(grain_min as i16);
    let gmax_v = i16x8_splat(grain_max as i16);
    let (min_value, max_value) = if data.clip_to_restricted_range {
        (16 << min_8, (if is_id { 235 } else { 240 }) << min_8)
    } else {
        (0, max)
    };
    let min_v = i32x4_splat(min_value);
    let max_v = i32x4_splat(max_value);
    let shift = data.scaling_shift as u32;
    let v_rnd = i32x4_splat((1i32 << shift) >> 1);
    let mut seed = scalar::row_seeds(data, row_num, rows);
    let mut offsets = [[0i32; 2]; 2];
    let csfl = data.chroma_scaling_from_luma;
    let uv_luma_mult = data.uv_luma_mult[uv];
    let uv_mult = data.uv_mult[uv];
    let uv_offset = data.uv_offset[uv] * (1 << min_8);
    let lm_v = i16x8_splat(uv_luma_mult as i16);
    let um_v = i16x8_splat(uv_mult as i16);
    let off_v = i32x4_splat(uv_offset);
    let scale_max_v = i32x4_splat(max);

    let add_scalar = |dst: &mut [u8], x: usize, y: usize, bx: usize, grain: i32| {
        let lx = (bx + x) << SX;
        let ly = y << SY;
        let l_off = ly * luma_stride + lx * bpp;
        let mut avg = load_scalar::<BPC>(luma, l_off);
        if SX != 0 {
            avg = (avg + load_scalar::<BPC>(luma, l_off + bpp) + 1) >> 1;
        }
        let off = y * stride + (bx + x) * bpp;
        let s = load_scalar::<BPC>(src, off);
        let val = if csfl {
            avg
        } else {
            (((avg * uv_luma_mult + s * uv_mult) >> 6) + uv_offset).clamp(0, max)
        };
        let noise = scalar::round2(i32::from(scaling[(val as usize) & mask]) * grain, shift);
        store_scalar::<BPC>(dst, off, (s + noise).clamp(min_value, max_value));
    };

    let bstep = FG_BLOCK_SIZE >> SX;
    let bstep_y = FG_BLOCK_SIZE >> SY;
    let mut bx = 0usize;
    while bx < pw {
        let bw = bstep.min(pw - bx);
        if overlap && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = scalar::get_random_number(8, &mut seed[i]);
        }
        let ystart = if overlap && row_num != 0 {
            bh.min(2 >> SY)
        } else {
            0
        };
        let xstart = if overlap && bx != 0 {
            bw.min(2 >> SX)
        } else {
            0
        };
        let (offx0, offy0) = ofset(offsets[0][0], SX, SY);

        for y in ystart..bh {
            let lut_row = &grain_lut[offy0 + y];
            let mut x = xstart;
            while x + 8 <= bw {
                let byte_off = y * stride + (bx + x) * bpp;
                let luma_off = (y << SY) * luma_stride + ((bx + x) << SX) * bpp;
                let s_v = load_pixels_8::<BPC>(src, byte_off);
                let avg_v = load_luma_avg8::<BPC, SX>(luma, luma_off);
                let idx_v = uv_scale_vec8(avg_v, s_v, csfl, lm_v, um_v, off_v, scale_max_v);
                let sc_v = scale_lookup8::<SCALE_N>(idx_v, scaling);
                let g_v = load_i16x8(lut_row, offx0 + x);
                let out = apply_noise8(s_v, sc_v, g_v, shift, v_rnd, min_v, max_v);
                store_pixels_8_i16::<BPC>(dst, byte_off, out);
                x += 8;
            }
            while x < bw {
                let grain = i32::from(lut_row[offx0 + x]);
                add_scalar(dst, x, y, bx, grain);
                x += 1;
            }
            for x in 0..xstart {
                let grain = scalar::sample_lut(grain_lut, &offsets, SX, SY, 0, 0, x, y);
                let old = scalar::sample_lut(grain_lut, &offsets, SX, SY, 1, 0, x, y);
                let grain = scalar::round2(old * W_OV[SX][x][0] + grain * W_OV[SX][x][1], 5)
                    .clamp(grain_min, grain_max);
                add_scalar(dst, x, y, bx, grain);
            }
        }

        if ystart > 0 {
            let (offx_top, offy_top) = ofset(offsets[0][1], SX, SY);
            for y in 0..ystart {
                let lut_new = &grain_lut[offy0 + y];
                let lut_old = &grain_lut[offy_top + bstep_y + y];
                let w0 = W_OV[SY][y][0] as i16;
                let w1 = W_OV[SY][y][1] as i16;
                let mut x = xstart;
                while x + 8 <= bw {
                    let byte_off = y * stride + (bx + x) * bpp;
                    let luma_off = (y << SY) * luma_stride + ((bx + x) << SX) * bpp;
                    let s_v = load_pixels_8::<BPC>(src, byte_off);
                    let avg_v = load_luma_avg8::<BPC, SX>(luma, luma_off);
                    let idx_v = uv_scale_vec8(avg_v, s_v, csfl, lm_v, um_v, off_v, scale_max_v);
                    let sc_v = scale_lookup8::<SCALE_N>(idx_v, scaling);
                    let g_new = load_i16x8(lut_new, offx0 + x);
                    let g_old = load_i16x8(lut_old, offx_top + x);
                    let g_v = blend_grain8(g_old, g_new, w0, w1, gmin_v, gmax_v);
                    let out = apply_noise8(s_v, sc_v, g_v, shift, v_rnd, min_v, max_v);
                    store_pixels_8_i16::<BPC>(dst, byte_off, out);
                    x += 8;
                }
                while x < bw {
                    let grain = scalar::sample_lut(grain_lut, &offsets, SX, SY, 0, 0, x, y);
                    let old = scalar::sample_lut(grain_lut, &offsets, SX, SY, 0, 1, x, y);
                    let grain = scalar::round2(old * W_OV[SY][y][0] + grain * W_OV[SY][y][1], 5)
                        .clamp(grain_min, grain_max);
                    add_scalar(dst, x, y, bx, grain);
                    x += 1;
                }
                for x in 0..xstart {
                    let top = scalar::sample_lut(grain_lut, &offsets, SX, SY, 0, 1, x, y);
                    let old = scalar::sample_lut(grain_lut, &offsets, SX, SY, 1, 1, x, y);
                    let top = scalar::round2(old * W_OV[SX][x][0] + top * W_OV[SX][x][1], 5)
                        .clamp(grain_min, grain_max);
                    let grain = scalar::sample_lut(grain_lut, &offsets, SX, SY, 0, 0, x, y);
                    let old = scalar::sample_lut(grain_lut, &offsets, SX, SY, 1, 0, x, y);
                    let grain = scalar::round2(old * W_OV[SX][x][0] + grain * W_OV[SX][x][1], 5)
                        .clamp(grain_min, grain_max);
                    let grain = scalar::round2(top * W_OV[SY][y][0] + grain * W_OV[SY][y][1], 5)
                        .clamp(grain_min, grain_max);
                    add_scalar(dst, x, y, bx, grain);
                }
            }
        }
        bx += bstep;
    }
}

pub fn fguv_32x32xn<const BPC: u32, const SX: usize, const SY: usize>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8],
    grain_lut: &GrainLut,
    bh: usize,
    row_num: usize,
    luma: &[u8],
    luma_stride: usize,
    uv: usize,
    is_id: bool,
    bd: FgBitDepth,
) {
    let ok = bd.bits() == BPC
        && if BPC == 8 {
            fguv_sozlesmesi(
                dst.len(),
                src.len(),
                stride,
                pw,
                bh,
                luma.len(),
                luma_stride,
                uv,
                SX,
                SY,
                data,
            )
        } else {
            fguv16_sozlesmesi(
                dst.len(),
                src.len(),
                stride,
                pw,
                bh,
                luma.len(),
                luma_stride,
                uv,
                SX,
                SY,
                data,
                bd,
            )
        };
    if ok {
        match BPC {
            8 => {
                if let Some(sc) = scaling
                    .get(..256)
                    .and_then(|s| <&[u8; 256]>::try_from(s).ok())
                {
                    return fguv_simd128::<8, SX, SY, 256>(
                        dst,
                        src,
                        stride,
                        data,
                        pw,
                        sc,
                        grain_lut,
                        bh,
                        row_num,
                        luma,
                        luma_stride,
                        uv,
                        is_id,
                    );
                }
            }
            10 => {
                if let Some(sc) = scaling
                    .get(..1024)
                    .and_then(|s| <&[u8; 1024]>::try_from(s).ok())
                {
                    return fguv_simd128::<10, SX, SY, 1024>(
                        dst,
                        src,
                        stride,
                        data,
                        pw,
                        sc,
                        grain_lut,
                        bh,
                        row_num,
                        luma,
                        luma_stride,
                        uv,
                        is_id,
                    );
                }
            }
            _ => {
                if let Some(sc) = scaling
                    .get(..4096)
                    .and_then(|s| <&[u8; 4096]>::try_from(s).ok())
                {
                    return fguv_simd128::<12, SX, SY, 4096>(
                        dst,
                        src,
                        stride,
                        data,
                        pw,
                        sc,
                        grain_lut,
                        bh,
                        row_num,
                        luma,
                        luma_stride,
                        uv,
                        is_id,
                    );
                }
            }
        }
    }
    fg_skaler_devri_say();
    let layout = match (SX, SY) {
        (1, 1) => 0,
        (1, 0) => 1,
        _ => 2,
    };
    (FgDsp::scalar::<BPC>().fguv_32x32xn[layout])(
        dst,
        src,
        stride,
        data,
        pw,
        scaling,
        grain_lut,
        bh,
        row_num,
        luma,
        luma_stride,
        uv,
        is_id,
        bd,
    );
}
