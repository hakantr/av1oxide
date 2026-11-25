//! `aarch64` NEON film grain çekirdekleri (`P54` / `F5d` — `F5-FG`).
//!
//! Referans: `dav1d` `src/arm/filmgrain.h` + `src/arm/64/{filmgrain.S,filmgrain16.S}`
//! (`99f3354`). `generate_grain_y`, `generate_grain_uv[0..2]`, `fgy_32x32xn` ve
//! `fguv_32x32xn[0..2]` (`8 / 10 / 12` bpc = `16` matris kaydı, `24` yuva)
//! `%100 #![forbid(unsafe_code)]` kararlı `core::arch::aarch64` NEON intrinsikleriyle
//! gerçeklenmiştir.

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use core::arch::aarch64::{
    int16x8_t, uint16x8_t, vaddq_s32, vaddvq_s32, vcombine_u16, vcreate_u16, vdupq_n_s16,
    vdupq_n_s32, vget_high_s16, vget_high_u8, vget_high_u16, vget_lane_u64, vget_low_s16,
    vget_low_u8, vget_low_u16, vmaxq_s16, vmaxq_s32, vminq_s16, vminq_s32, vmlal_high_s16,
    vmlal_s16, vmovl_s16, vmovl_u8, vmovl_u16, vmovn_high_u32, vmovn_u16, vmovn_u32,
    vmull_high_s16, vmull_s16, vreinterpret_u64_u16, vreinterpretq_s16_u16, vreinterpretq_s32_u32,
    vreinterpretq_u8_u16, vreinterpretq_u16_u8, vreinterpretq_u32_s32, vrhadd_u8, vrhaddq_u16,
    vrshlq_s32, vshrq_n_s32, vuzp1_u8, vuzp1q_u16, vuzp2_u8, vuzp2q_u16,
};

use super::giris::store_i16x8;
use super::mem::{chunk, chunk_mut, load_i16x8, load_u8x8, load_u8x16, store_u8x8, store_u8x16};
use crate::bitstream::frame_header::FilmGrainData;
use crate::cpu::NeonToken;
use crate::dsp::fg::{
    FG_BLOCK_SIZE, FguvFn, FgyFn, GRAIN_HEIGHT, GRAIN_WIDTH, GenerateGrainUvFn, GenerateGrainYFn,
    GrainLut, SUB_GRAIN_HEIGHT, SUB_GRAIN_WIDTH,
};
use crate::dsp::fg_tables::GAUSSIAN_SEQUENCE;
use crate::dsp::scalar::fg as scalar;

const W_OV: [[[i32; 2]; 2]; 2] = [[[27, 17], [17, 27]], [[23, 22], [0, 0]]];

/// `generate_grain_y` `NeonToken`-kapılı giriş fonksiyonu.
#[must_use]
pub fn generate_grain_y_fn<const BPC: u32>(_neon: NeonToken) -> GenerateGrainYFn {
    super::giris::fg_grain_y::<BPC>
}

/// `generate_grain_uv[layout]` `NeonToken`-kapılı giriş fonksiyonu.
#[must_use]
pub fn generate_grain_uv_fn<const BPC: u32>(
    _neon: NeonToken,
    layout: usize,
) -> Option<GenerateGrainUvFn> {
    match layout {
        0 => Some(super::giris::fg_grain_uv::<BPC, 0>),
        1 => Some(super::giris::fg_grain_uv::<BPC, 1>),
        2 => Some(super::giris::fg_grain_uv::<BPC, 2>),
        _ => None,
    }
}

/// `fgy_32x32xn` `NeonToken`-kapılı giriş fonksiyonu.
#[must_use]
pub fn fgy_fn<const BPC: u32>(_neon: NeonToken) -> FgyFn {
    super::giris::fg_fgy::<BPC>
}

/// `fguv_32x32xn[layout]` `NeonToken`-kapılı giriş fonksiyonu.
#[must_use]
pub fn fguv_fn<const BPC: u32>(_neon: NeonToken, layout: usize) -> Option<FguvFn> {
    match layout {
        0 => Some(super::giris::fg_fguv::<BPC, 0>),
        1 => Some(super::giris::fg_fguv::<BPC, 1>),
        2 => Some(super::giris::fg_fguv::<BPC, 2>),
        _ => None,
    }
}

#[inline(always)]
const fn min_8_of<const BPC: u32>() -> u32 {
    BPC - 8
}

#[inline(always)]
const fn bd_max_of<const BPC: u32>() -> i32 {
    (1i32 << BPC) - 1
}

#[inline(always)]
const fn ofset(randval: i32, sx: usize, sy: usize) -> (usize, usize) {
    (
        3 + (2 >> sx) * (3 + (randval >> 4)) as usize,
        3 + (2 >> sy) * (3 + (randval & 0xF)) as usize,
    )
}

macro_rules! dot7_3rows {
    ($r0:expr, $r1:expr, $r2:expr, $x_start:expr, $cv0:expr, $cv1:expr, $cv2:expr) => {{
        let xs = $x_start;
        let b0 = load_i16x8(chunk::<i16, 8>($r0, xs));
        let b1 = load_i16x8(chunk::<i16, 8>($r1, xs));
        let b2 = load_i16x8(chunk::<i16, 8>($r2, xs));
        let acc0 = vmlal_high_s16(vmull_s16(vget_low_s16(b0), vget_low_s16($cv0)), b0, $cv0);
        let acc1 = vmlal_high_s16(vmull_s16(vget_low_s16(b1), vget_low_s16($cv1)), b1, $cv1);
        let acc2 = vmlal_high_s16(vmull_s16(vget_low_s16(b2), vget_low_s16($cv2)), b2, $cv2);
        vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), acc2))
    }};
}

#[target_feature(enable = "neon")]
pub(super) fn generate_grain_y_neon<const BPC: u32>(buf: &mut GrainLut, data: &FilmGrainData) {
    let min_8 = min_8_of::<BPC>();
    let mut seed = data.seed;
    let shift = 4 - min_8 + data.grain_scale_shift as u32;
    let neg_shift = vdupq_n_s32(-(shift as i32));
    let grain_ctr = 128i32 << min_8;
    let (grain_min, grain_max) = (-grain_ctr, grain_ctr - 1);

    for row in buf.iter_mut() {
        let mut x = 0usize;
        while x + 8 <= GRAIN_WIDTH {
            let mut g = [0i16; 8];
            for k in 0..8 {
                let idx = scalar::get_random_number(11, &mut seed) as usize;
                g[k] = GAUSSIAN_SEQUENCE[idx];
            }
            let gv = load_i16x8(&g);
            let lo = vrshlq_s32(vmovl_s16(vget_low_s16(gv)), neg_shift);
            let hi = vrshlq_s32(vmovl_s16(vget_high_s16(gv)), neg_shift);
            let out_u16 = vmovn_high_u32(
                vmovn_u32(vreinterpretq_u32_s32(lo)),
                vreinterpretq_u32_s32(hi),
            );
            store_i16x8(chunk_mut::<i16, 8>(row, x), vreinterpretq_s16_u16(out_u16));
            x += 8;
        }
        while x < GRAIN_WIDTH {
            let idx = scalar::get_random_number(11, &mut seed) as usize;
            row[x] = scalar::round2(i32::from(GAUSSIAN_SEQUENCE[idx]), shift) as i16;
            x += 1;
        }
    }

    let ar_lag = data.ar_coeff_lag as usize;
    if ar_lag == 0 {
        return;
    }
    let ar_pad = 3usize;
    let ar_shift = data.ar_coeff_shift as u32;

    if ar_lag == 3 {
        let c = &data.ar_coeffs_y;
        let cv0 = load_i16x8(&[
            i16::from(c[0]),
            i16::from(c[1]),
            i16::from(c[2]),
            i16::from(c[3]),
            i16::from(c[4]),
            i16::from(c[5]),
            i16::from(c[6]),
            0,
        ]);
        let cv1 = load_i16x8(&[
            i16::from(c[7]),
            i16::from(c[8]),
            i16::from(c[9]),
            i16::from(c[10]),
            i16::from(c[11]),
            i16::from(c[12]),
            i16::from(c[13]),
            0,
        ]);
        let cv2 = load_i16x8(&[
            i16::from(c[14]),
            i16::from(c[15]),
            i16::from(c[16]),
            i16::from(c[17]),
            i16::from(c[18]),
            i16::from(c[19]),
            i16::from(c[20]),
            0,
        ]);
        let c21 = i32::from(c[21]);
        let c22 = i32::from(c[22]);
        let c23 = i32::from(c[23]);

        for y in ar_pad..GRAIN_HEIGHT {
            for x in ar_pad..GRAIN_WIDTH - ar_pad {
                let mut sum = if x + 5 <= GRAIN_WIDTH {
                    dot7_3rows!(&buf[y - 3], &buf[y - 2], &buf[y - 1], x - 3, cv0, cv1, cv2)
                } else {
                    let mut s = 0i32;
                    for dx in 0..7 {
                        s += i32::from(c[dx]) * i32::from(buf[y - 3][x - 3 + dx])
                            + i32::from(c[7 + dx]) * i32::from(buf[y - 2][x - 3 + dx])
                            + i32::from(c[14 + dx]) * i32::from(buf[y - 1][x - 3 + dx]);
                    }
                    s
                };
                sum += c21 * i32::from(buf[y][x - 3])
                    + c22 * i32::from(buf[y][x - 2])
                    + c23 * i32::from(buf[y][x - 1]);
                let grain = i32::from(buf[y][x]) + scalar::round2(sum, ar_shift);
                buf[y][x] = grain.clamp(grain_min, grain_max) as i16;
            }
        }
        return;
    }

    for y in ar_pad..GRAIN_HEIGHT {
        for x in ar_pad..GRAIN_WIDTH - ar_pad {
            let mut ci = 0usize;
            let mut sum = 0i32;
            'outer: for yy in y - ar_lag..=y {
                for xx in x - ar_lag..=x + ar_lag {
                    if yy == y && xx == x {
                        break 'outer;
                    }
                    sum += i32::from(data.ar_coeffs_y[ci]) * i32::from(buf[yy][xx]);
                    ci += 1;
                }
            }
            let grain = i32::from(buf[y][x]) + scalar::round2(sum, ar_shift);
            buf[y][x] = grain.clamp(grain_min, grain_max) as i16;
        }
    }
}

#[target_feature(enable = "neon")]
pub(super) fn generate_grain_uv_neon<const BPC: u32, const SX: usize, const SY: usize>(
    buf: &mut GrainLut,
    buf_y: &GrainLut,
    data: &FilmGrainData,
    uv: usize,
) {
    let min_8 = min_8_of::<BPC>();
    let mut seed = data.seed ^ if uv != 0 { 0x49d8 } else { 0xb524 };
    let shift = 4 - min_8 + data.grain_scale_shift as u32;
    let neg_shift = vdupq_n_s32(-(shift as i32));
    let grain_ctr = 128i32 << min_8;
    let (grain_min, grain_max) = (-grain_ctr, grain_ctr - 1);

    let chroma_w = if SX != 0 {
        SUB_GRAIN_WIDTH
    } else {
        GRAIN_WIDTH
    };
    let chroma_h = if SY != 0 {
        SUB_GRAIN_HEIGHT
    } else {
        GRAIN_HEIGHT
    };

    for row in buf.iter_mut().take(chroma_h) {
        let mut x = 0usize;
        while x + 8 <= chroma_w {
            let mut g = [0i16; 8];
            for k in 0..8 {
                let idx = scalar::get_random_number(11, &mut seed) as usize;
                g[k] = GAUSSIAN_SEQUENCE[idx];
            }
            let gv = load_i16x8(&g);
            let lo = vrshlq_s32(vmovl_s16(vget_low_s16(gv)), neg_shift);
            let hi = vrshlq_s32(vmovl_s16(vget_high_s16(gv)), neg_shift);
            let out_u16 = vmovn_high_u32(
                vmovn_u32(vreinterpretq_u32_s32(lo)),
                vreinterpretq_u32_s32(hi),
            );
            store_i16x8(
                chunk_mut::<i16, 8>(row, x),
                core::arch::aarch64::vreinterpretq_s16_u16(out_u16),
            );
            x += 8;
        }
        while x < chroma_w {
            let idx = scalar::get_random_number(11, &mut seed) as usize;
            row[x] = scalar::round2(i32::from(GAUSSIAN_SEQUENCE[idx]), shift) as i16;
            x += 1;
        }
    }

    let ar_pad = 3usize;
    let ar_lag = data.ar_coeff_lag as usize;
    let ar_shift = data.ar_coeff_shift as u32;
    let coeffs = &data.ar_coeffs_uv[uv];

    if ar_lag == 3 {
        let cv0 = load_i16x8(&[
            i16::from(coeffs[0]),
            i16::from(coeffs[1]),
            i16::from(coeffs[2]),
            i16::from(coeffs[3]),
            i16::from(coeffs[4]),
            i16::from(coeffs[5]),
            i16::from(coeffs[6]),
            0,
        ]);
        let cv1 = load_i16x8(&[
            i16::from(coeffs[7]),
            i16::from(coeffs[8]),
            i16::from(coeffs[9]),
            i16::from(coeffs[10]),
            i16::from(coeffs[11]),
            i16::from(coeffs[12]),
            i16::from(coeffs[13]),
            0,
        ]);
        let cv2 = load_i16x8(&[
            i16::from(coeffs[14]),
            i16::from(coeffs[15]),
            i16::from(coeffs[16]),
            i16::from(coeffs[17]),
            i16::from(coeffs[18]),
            i16::from(coeffs[19]),
            i16::from(coeffs[20]),
            0,
        ]);
        let c21 = i32::from(coeffs[21]);
        let c22 = i32::from(coeffs[22]);
        let c23 = i32::from(coeffs[23]);
        let c24 = i32::from(coeffs[24]);

        for y in ar_pad..chroma_h {
            for x in ar_pad..chroma_w - ar_pad {
                let mut sum = if x + 5 <= GRAIN_WIDTH {
                    dot7_3rows!(&buf[y - 3], &buf[y - 2], &buf[y - 1], x - 3, cv0, cv1, cv2)
                } else {
                    let mut s = 0i32;
                    for dx in 0..7 {
                        s += i32::from(coeffs[dx]) * i32::from(buf[y - 3][x - 3 + dx])
                            + i32::from(coeffs[7 + dx]) * i32::from(buf[y - 2][x - 3 + dx])
                            + i32::from(coeffs[14 + dx]) * i32::from(buf[y - 1][x - 3 + dx]);
                    }
                    s
                };
                sum += c21 * i32::from(buf[y][x - 3])
                    + c22 * i32::from(buf[y][x - 2])
                    + c23 * i32::from(buf[y][x - 1]);
                if data.num_y_points != 0 {
                    let luma_x = ((x - ar_pad) << SX) + ar_pad;
                    let luma_y = ((y - ar_pad) << SY) + ar_pad;
                    let mut luma = 0i32;
                    for i in 0..=SY {
                        for j in 0..=SX {
                            luma += i32::from(buf_y[luma_y + i][luma_x + j]);
                        }
                    }
                    let luma = scalar::round2(luma, (SX + SY) as u32);
                    sum += luma * c24;
                }
                let grain = i32::from(buf[y][x]) + scalar::round2(sum, ar_shift);
                buf[y][x] = grain.clamp(grain_min, grain_max) as i16;
            }
        }
        return;
    }

    for y in ar_pad..chroma_h {
        for x in ar_pad..chroma_w - ar_pad {
            let mut ci = 0usize;
            let mut sum = 0i32;
            'outer: for yy in y - ar_lag..=y {
                for xx in x - ar_lag..=x + ar_lag {
                    if yy == y && xx == x {
                        if data.num_y_points != 0 {
                            let luma_x = ((x - ar_pad) << SX) + ar_pad;
                            let luma_y = ((y - ar_pad) << SY) + ar_pad;
                            let mut luma = 0i32;
                            for i in 0..=SY {
                                for j in 0..=SX {
                                    luma += i32::from(buf_y[luma_y + i][luma_x + j]);
                                }
                            }
                            let luma = scalar::round2(luma, (SX + SY) as u32);
                            sum += luma * i32::from(coeffs[ci]);
                        }
                        break 'outer;
                    }
                    sum += i32::from(coeffs[ci]) * i32::from(buf[yy][xx]);
                    ci += 1;
                }
            }
            let grain = i32::from(buf[y][x]) + scalar::round2(sum, ar_shift);
            buf[y][x] = grain.clamp(grain_min, grain_max) as i16;
        }
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn load_px8<const BPC: u32>(buf: &[u8], byte_off: usize) -> uint16x8_t {
    if BPC == 8 {
        vmovl_u8(load_u8x8(chunk::<u8, 8>(buf, byte_off)))
    } else {
        vreinterpretq_u16_u8(load_u8x16(chunk::<u8, 16>(buf, byte_off)))
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn store_px8<const BPC: u32>(buf: &mut [u8], byte_off: usize, v: uint16x8_t) {
    if BPC == 8 {
        store_u8x8(chunk_mut::<u8, 8>(buf, byte_off), vmovn_u16(v));
    } else {
        store_u8x16(chunk_mut::<u8, 16>(buf, byte_off), vreinterpretq_u8_u16(v));
    }
}

macro_rules! scale_lookup8 {
    ($idx_u16:expr, $scaling:expr, $scale_n:expr) => {{
        let idx_v = $idx_u16;
        let mask_v = ($scale_n) - 1;
        let lo64 = vget_lane_u64::<0>(vreinterpret_u64_u16(vget_low_u16(idx_v)));
        let hi64 = vget_lane_u64::<0>(vreinterpret_u64_u16(vget_high_u16(idx_v)));
        let s0 = u64::from($scaling[(lo64 as u16 as usize) & mask_v]);
        let s1 = u64::from($scaling[((lo64 >> 16) as u16 as usize) & mask_v]);
        let s2 = u64::from($scaling[((lo64 >> 32) as u16 as usize) & mask_v]);
        let s3 = u64::from($scaling[((lo64 >> 48) as u16 as usize) & mask_v]);
        let s4 = u64::from($scaling[(hi64 as u16 as usize) & mask_v]);
        let s5 = u64::from($scaling[((hi64 >> 16) as u16 as usize) & mask_v]);
        let s6 = u64::from($scaling[((hi64 >> 32) as u16 as usize) & mask_v]);
        let s7 = u64::from($scaling[((hi64 >> 48) as u16 as usize) & mask_v]);
        let lo_pack = s0 | (s1 << 16) | (s2 << 32) | (s3 << 48);
        let hi_pack = s4 | (s5 << 16) | (s6 << 32) | (s7 << 48);
        vreinterpretq_s16_u16(vcombine_u16(vcreate_u16(lo_pack), vcreate_u16(hi_pack)))
    }};
}

macro_rules! apply_noise8 {
    ($s_u16:expr, $scale_i16:expr, $grain_i16:expr, $neg_shift:expr, $min_v:expr, $max_v:expr) => {{
        let sv = $s_u16;
        let scv = $scale_i16;
        let grv = $grain_i16;
        let n_lo = vrshlq_s32(vmull_s16(vget_low_s16(scv), vget_low_s16(grv)), $neg_shift);
        let n_hi = vrshlq_s32(vmull_high_s16(scv, grv), $neg_shift);
        let s_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(sv)));
        let s_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(sv)));
        let o_lo = vminq_s32(vmaxq_s32(vaddq_s32(s_lo, n_lo), $min_v), $max_v);
        let o_hi = vminq_s32(vmaxq_s32(vaddq_s32(s_hi, n_hi), $min_v), $max_v);
        vmovn_high_u32(
            vmovn_u32(vreinterpretq_u32_s32(o_lo)),
            vreinterpretq_u32_s32(o_hi),
        )
    }};
}

#[target_feature(enable = "neon")]
#[inline]
fn blend_grain8(
    old_g: int16x8_t,
    new_g: int16x8_t,
    w0: i16,
    w1: i16,
    gmin: int16x8_t,
    gmax: int16x8_t,
) -> int16x8_t {
    let w0v = vdupq_n_s16(w0);
    let w1v = vdupq_n_s16(w1);
    let neg5 = vdupq_n_s32(-5);
    let lo = vrshlq_s32(
        vmlal_s16(
            vmull_s16(vget_low_s16(old_g), vget_low_s16(w0v)),
            vget_low_s16(new_g),
            vget_low_s16(w1v),
        ),
        neg5,
    );
    let hi = vrshlq_s32(vmlal_high_s16(vmull_high_s16(old_g, w0v), new_g, w1v), neg5);
    let p_u16 = vmovn_high_u32(
        vmovn_u32(vreinterpretq_u32_s32(lo)),
        vreinterpretq_u32_s32(hi),
    );
    let p_s16 = vreinterpretq_s16_u16(p_u16);
    vminq_s16(vmaxq_s16(p_s16, gmin), gmax)
}

#[inline(always)]
fn load_px_scalar<const BPC: u32>(buf: &[u8], px_idx: usize) -> i32 {
    if BPC == 8 {
        i32::from(buf[px_idx])
    } else {
        i32::from(u16::from_le_bytes([buf[2 * px_idx], buf[2 * px_idx + 1]]))
    }
}

#[inline(always)]
fn store_px_scalar<const BPC: u32>(buf: &mut [u8], px_idx: usize, val: i32) {
    if BPC == 8 {
        buf[px_idx] = val as u8;
    } else {
        buf[2 * px_idx..2 * px_idx + 2].copy_from_slice(&(val as u16).to_le_bytes());
    }
}

#[target_feature(enable = "neon")]
pub(super) fn fgy_neon<const BPC: u32, const SCALE_N: usize>(
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
    let bpp = if BPC == 8 { 1 } else { 2 };
    let max = bd_max_of::<BPC>();
    let mask = SCALE_N - 1;
    let overlap = data.overlap_flag;
    let rows = 1 + usize::from(overlap && row_num > 0);
    let min_8 = min_8_of::<BPC>();
    let grain_ctr = 128i32 << min_8;
    let (grain_min, grain_max) = (-grain_ctr, grain_ctr - 1);
    let gmin_v = vdupq_n_s16(grain_min as i16);
    let gmax_v = vdupq_n_s16(grain_max as i16);
    let (min_value, max_value) = if data.clip_to_restricted_range {
        (16 << min_8, 235 << min_8)
    } else {
        (0, max)
    };
    let min_v = vdupq_n_s32(min_value);
    let max_v = vdupq_n_s32(max_value);
    let shift = data.scaling_shift as u32;
    let neg_shift = vdupq_n_s32(-(shift as i32));
    let mut seed = scalar::row_seeds(data, row_num, rows);
    let mut offsets = [[0i32; 2]; 2];
    let stride_px = stride / bpp;

    let add_scalar = |dst: &mut [u8], x: usize, y: usize, bx: usize, grain: i32| {
        let idx = y * stride_px + bx + x;
        let s = load_px_scalar::<BPC>(src, idx);
        let noise = scalar::round2(i32::from(scaling[(s as usize) & mask]) * grain, shift);
        store_px_scalar::<BPC>(dst, idx, (s + noise).clamp(min_value, max_value));
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
                let s_v = load_px8::<BPC>(src, byte_off);
                let sc_v = scale_lookup8!(s_v, scaling, SCALE_N);
                let g_v = load_i16x8(chunk::<i16, 8>(lut_row, offx0 + x));
                let out = apply_noise8!(s_v, sc_v, g_v, neg_shift, min_v, max_v);
                store_px8::<BPC>(dst, byte_off, out);
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
                    let s_v = load_px8::<BPC>(src, byte_off);
                    let sc_v = scale_lookup8!(s_v, scaling, SCALE_N);
                    let g_new = load_i16x8(chunk::<i16, 8>(lut_new, offx0 + x));
                    let g_old = load_i16x8(chunk::<i16, 8>(lut_old, offx_top + x));
                    let g_v = blend_grain8(g_old, g_new, w0, w1, gmin_v, gmax_v);
                    let out = apply_noise8!(s_v, sc_v, g_v, neg_shift, min_v, max_v);
                    store_px8::<BPC>(dst, byte_off, out);
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

#[target_feature(enable = "neon")]
#[inline]
fn load_luma_avg8<const BPC: u32, const SX: usize>(luma: &[u8], byte_off: usize) -> uint16x8_t {
    if SX == 0 {
        load_px8::<BPC>(luma, byte_off)
    } else if BPC == 8 {
        let raw = load_u8x16(chunk::<u8, 16>(luma, byte_off));
        let lo = vget_low_u8(raw);
        let hi = vget_high_u8(raw);
        let evens = vuzp1_u8(lo, hi);
        let odds = vuzp2_u8(lo, hi);
        vmovl_u8(vrhadd_u8(evens, odds))
    } else {
        let v0 = vreinterpretq_u16_u8(load_u8x16(chunk::<u8, 16>(luma, byte_off)));
        let v1 = vreinterpretq_u16_u8(load_u8x16(chunk::<u8, 16>(luma, byte_off + 16)));
        let evens = vuzp1q_u16(v0, v1);
        let odds = vuzp2q_u16(v0, v1);
        vrhaddq_u16(evens, odds)
    }
}

macro_rules! uv_scale_vec8 {
    ($avg_u16:expr, $s_u16:expr, $csfl:expr, $lm_v:expr, $um_v:expr, $off_v:expr, $scale_max_v:expr) => {{
        let av = $avg_u16;
        if $csfl {
            av
        } else {
            let zero = vdupq_n_s32(0);
            let avg_s = vreinterpretq_s16_u16(av);
            let s_s = vreinterpretq_s16_u16($s_u16);
            let c_lo = vmlal_s16(
                vmull_s16(vget_low_s16(avg_s), vget_low_s16($lm_v)),
                vget_low_s16(s_s),
                vget_low_s16($um_v),
            );
            let c_hi = vmlal_high_s16(vmull_high_s16(avg_s, $lm_v), s_s, $um_v);
            let v_lo = vminq_s32(
                vmaxq_s32(vaddq_s32(vshrq_n_s32::<6>(c_lo), $off_v), zero),
                $scale_max_v,
            );
            let v_hi = vminq_s32(
                vmaxq_s32(vaddq_s32(vshrq_n_s32::<6>(c_hi), $off_v), zero),
                $scale_max_v,
            );
            vcombine_u16(
                vmovn_u32(vreinterpretq_u32_s32(v_lo)),
                vmovn_u32(vreinterpretq_u32_s32(v_hi)),
            )
        }
    }};
}

#[target_feature(enable = "neon")]
pub(super) fn fguv_neon<const BPC: u32, const SX: usize, const SY: usize, const SCALE_N: usize>(
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
    let bpp = if BPC == 8 { 1 } else { 2 };
    let max = bd_max_of::<BPC>();
    let mask = SCALE_N - 1;
    let overlap = data.overlap_flag;
    let rows = 1 + usize::from(overlap && row_num > 0);
    let min_8 = min_8_of::<BPC>();
    let grain_ctr = 128i32 << min_8;
    let (grain_min, grain_max) = (-grain_ctr, grain_ctr - 1);
    let gmin_v = vdupq_n_s16(grain_min as i16);
    let gmax_v = vdupq_n_s16(grain_max as i16);
    let (min_value, max_value) = if data.clip_to_restricted_range {
        (16 << min_8, (if is_id { 235 } else { 240 }) << min_8)
    } else {
        (0, max)
    };
    let min_v = vdupq_n_s32(min_value);
    let max_v = vdupq_n_s32(max_value);
    let shift = data.scaling_shift as u32;
    let neg_shift = vdupq_n_s32(-(shift as i32));
    let mut seed = scalar::row_seeds(data, row_num, rows);
    let mut offsets = [[0i32; 2]; 2];
    let csfl = data.chroma_scaling_from_luma;
    let uv_luma_mult = data.uv_luma_mult[uv];
    let uv_mult = data.uv_mult[uv];
    let uv_offset = data.uv_offset[uv] * (1 << min_8);
    let lm_v = vdupq_n_s16(uv_luma_mult as i16);
    let um_v = vdupq_n_s16(uv_mult as i16);
    let off_v = vdupq_n_s32(uv_offset);
    let scale_max_v = vdupq_n_s32(max);
    let stride_px = stride / bpp;
    let luma_stride_px = luma_stride / bpp;

    let add_scalar = |dst: &mut [u8], x: usize, y: usize, bx: usize, grain: i32| {
        let lx = (bx + x) << SX;
        let ly = y << SY;
        let lrow = ly * luma_stride_px;
        let mut avg = load_px_scalar::<BPC>(luma, lrow + lx);
        if SX != 0 {
            avg = (avg + load_px_scalar::<BPC>(luma, lrow + lx + 1) + 1) >> 1;
        }
        let idx = y * stride_px + bx + x;
        let s = load_px_scalar::<BPC>(src, idx);
        let val = if csfl {
            avg
        } else {
            (((avg * uv_luma_mult + s * uv_mult) >> 6) + uv_offset).clamp(0, max)
        };
        let noise = scalar::round2(i32::from(scaling[(val as usize) & mask]) * grain, shift);
        store_px_scalar::<BPC>(dst, idx, (s + noise).clamp(min_value, max_value));
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
                let s_v = load_px8::<BPC>(src, byte_off);
                let avg_v = load_luma_avg8::<BPC, SX>(luma, luma_off);
                let idx_v = uv_scale_vec8!(avg_v, s_v, csfl, lm_v, um_v, off_v, scale_max_v);
                let sc_v = scale_lookup8!(idx_v, scaling, SCALE_N);
                let g_v = load_i16x8(chunk::<i16, 8>(lut_row, offx0 + x));
                let out = apply_noise8!(s_v, sc_v, g_v, neg_shift, min_v, max_v);
                store_px8::<BPC>(dst, byte_off, out);
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
                    let s_v = load_px8::<BPC>(src, byte_off);
                    let avg_v = load_luma_avg8::<BPC, SX>(luma, luma_off);
                    let idx_v = uv_scale_vec8!(avg_v, s_v, csfl, lm_v, um_v, off_v, scale_max_v);
                    let sc_v = scale_lookup8!(idx_v, scaling, SCALE_N);
                    let g_new = load_i16x8(chunk::<i16, 8>(lut_new, offx0 + x));
                    let g_old = load_i16x8(chunk::<i16, 8>(lut_old, offx_top + x));
                    let g_v = blend_grain8(g_old, g_new, w0, w1, gmin_v, gmax_v);
                    let out = apply_noise8!(s_v, sc_v, g_v, neg_shift, min_v, max_v);
                    store_px8::<BPC>(dst, byte_off, out);
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
