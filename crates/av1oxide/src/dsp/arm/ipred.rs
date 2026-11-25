//! `aarch64` NEON intra tahmin (`ipred`, `cfl_ac`, `cfl_pred`, `pal_pred`) çekirdekleri
//! (`P54` / `F5d` — `F5-IPRED`, `P58-M4` sıcak döngü iyileştirmesi).
//!
//! Referans: `dav1d` `src/arm/ipred.h` + `src/arm/64/{ipred.S,ipred16.S}` (`99f3354`).
//! Kapsam (`44` `dispatch-matrix.csv` kaydı = `66` yuva @ `8 / 10 / 12` bpc):
//! - `intra_pred[0..13]`: `DC`, `VERT`, `HOR`, `LEFT_DC`, `TOP_DC`, `DC_128`,
//!   `Z1`, `Z2`, `Z3`, `SMOOTH`, `SMOOTH_V`, `SMOOTH_H`, `PAETH`, `FILTER`
//! - `cfl_ac[0..2]`: `4:2:0`, `4:2:2`, `4:4:4`
//! - `cfl_pred[0..3]`: `DC`, `LEFT_DC`, `TOP_DC`, `DC_128`
//! - `pal_pred`

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use core::arch::aarch64::{
    int16x8_t, int32x4_t, uint8x16_t, uint16x4_t, uint16x8_t, uint32x4_t, vabdq_s16, vabsq_s32,
    vaddlvq_u8, vaddlvq_u16, vaddq_s16, vaddq_s32, vaddq_u8, vaddq_u16, vaddq_u32, vaddvq_s32,
    vandq_u8, vandq_u16, vbslq_s16, vbslq_s32, vcleq_s16, vcltq_s32, vcombine_u8, vcombine_u16,
    vdup_n_u8, vdup_n_u16, vdup_n_u32, vdupq_n_s16, vdupq_n_s32, vdupq_n_u8, vdupq_n_u16,
    vdupq_n_u32, vget_high_u8, vget_high_u16, vget_lane_u32, vget_lane_u64, vget_low_s16,
    vget_low_u8, vget_low_u16, vmaxq_s32, vmaxvq_u8, vminq_s32, vmlal_high_s16, vmlal_high_u16,
    vmlal_s16, vmlal_u16, vmlaq_n_u16, vmovl_high_u16, vmovl_u8, vmovl_u16, vmovn_high_u32,
    vmovn_u16, vmovn_u32, vmull_high_s16, vmull_high_u16, vmull_s16, vmull_u16, vmulq_n_u16,
    vmulq_n_u32, vnegq_s32, vpadalq_s16, vpaddlq_u8, vpaddlq_u16, vqtbl1q_u8, vreinterpret_u8_u32,
    vreinterpret_u16_u8, vreinterpret_u16_u32, vreinterpret_u32_u8, vreinterpret_u32_u16,
    vreinterpret_u64_u8, vreinterpret_u64_u16, vreinterpretq_s16_u16, vreinterpretq_s32_u32,
    vreinterpretq_u8_u16, vreinterpretq_u8_u32, vreinterpretq_u16_s16, vreinterpretq_u16_u8,
    vreinterpretq_u32_s32, vrshrn_high_n_u32, vrshrn_n_u32, vrshrq_n_s32, vrshrq_n_u16,
    vshlq_n_s16, vshlq_n_u8, vshlq_n_u16, vshrq_n_s32, vshrq_n_u8, vsubq_s16, vsubq_s32, vsubq_u16,
    vtrn1_u16, vtrn1_u32, vtrn2_u16, vtrn2_u32, vuzp1q_s16, vuzp2q_s16, vzip1q_s16, vzip1q_u8,
    vzip2q_s16,
};

use super::giris::store_i16x8;
use super::mem::{
    chunk, chunk_mut, load_i16x8, load_u8x8, load_u8x16, load_u16x4, load_u16x8, store_u8x8,
    store_u8x16,
};
use crate::cpu::NeonToken;
use crate::dsp::ipred::{
    AngularPredFn, CflAcFn, CflPredFn, CflPredMode, Edge, IntraPredMode, PalPredFn,
};
use crate::dsp::ipred_tables::{DR_INTRA_DERIVATIVE, FILTER_INTRA_TAPS, SM_WEIGHTS};

const TMP_LEN: usize = 320;
const TMP_BASE: isize = 128;

/// `FILTER_INTRA_TAPS` tablosunun 8 çıktı (`yy=0..2 × xx=0..4`) × 7 girdi (`p0..p6`)
/// için NEON dostu `i16` sütun düzeni (derleme zamanında hesaplanır).
const FILTER_TAPS_NEON: [[[i16; 8]; 7]; 5] = {
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

/// `intra_pred[mode]` `NeonToken`-kapılı giriş fonksiyonu.
#[must_use]
pub fn intra_pred_fn<const BPC: u32>(_neon: NeonToken, mode: IntraPredMode) -> AngularPredFn {
    match mode {
        IntraPredMode::Dc => super::giris::ipred_angular::<BPC, 0>,
        IntraPredMode::Vert => super::giris::ipred_angular::<BPC, 1>,
        IntraPredMode::Hor => super::giris::ipred_angular::<BPC, 2>,
        IntraPredMode::LeftDc => super::giris::ipred_angular::<BPC, 3>,
        IntraPredMode::TopDc => super::giris::ipred_angular::<BPC, 4>,
        IntraPredMode::Dc128 => super::giris::ipred_angular::<BPC, 5>,
        IntraPredMode::Z1 => super::giris::ipred_angular::<BPC, 6>,
        IntraPredMode::Z2 => super::giris::ipred_angular::<BPC, 7>,
        IntraPredMode::Z3 => super::giris::ipred_angular::<BPC, 8>,
        IntraPredMode::Smooth => super::giris::ipred_angular::<BPC, 9>,
        IntraPredMode::SmoothV => super::giris::ipred_angular::<BPC, 10>,
        IntraPredMode::SmoothH => super::giris::ipred_angular::<BPC, 11>,
        IntraPredMode::Paeth => super::giris::ipred_angular::<BPC, 12>,
        IntraPredMode::Filter => super::giris::ipred_angular::<BPC, 13>,
    }
}

/// `cfl_ac[layout]` `NeonToken`-kapılı giriş fonksiyonu.
#[must_use]
pub fn cfl_ac_fn<const BPC: u32>(_neon: NeonToken, layout: usize) -> Option<CflAcFn> {
    match layout {
        0 => Some(super::giris::ipred_cfl_ac::<BPC, 0>),
        1 => Some(super::giris::ipred_cfl_ac::<BPC, 1>),
        2 => Some(super::giris::ipred_cfl_ac::<BPC, 2>),
        _ => None,
    }
}

/// `cfl_pred[mode]` `NeonToken`-kapılı giriş fonksiyonu.
#[must_use]
pub fn cfl_pred_fn<const BPC: u32>(_neon: NeonToken, mode: CflPredMode) -> CflPredFn {
    match mode {
        CflPredMode::Dc => super::giris::ipred_cfl_pred::<BPC, 0>,
        CflPredMode::LeftDc => super::giris::ipred_cfl_pred::<BPC, 1>,
        CflPredMode::TopDc => super::giris::ipred_cfl_pred::<BPC, 2>,
        CflPredMode::Dc128 => super::giris::ipred_cfl_pred::<BPC, 3>,
    }
}

/// `pal_pred` `NeonToken`-kapılı giriş fonksiyonu.
#[must_use]
pub fn pal_pred_fn<const BPC: u32>(_neon: NeonToken) -> PalPredFn {
    super::giris::ipred_pal_pred::<BPC>
}

#[inline(always)]
const fn bpp_of<const BPC: u32>() -> usize {
    if BPC == 8 { 1 } else { 2 }
}

#[inline(always)]
const fn bd_max_of<const BPC: u32>() -> i32 {
    (1i32 << BPC) - 1
}

#[inline(always)]
fn read_edge<const BPC: u32>(edge: Edge<'_>, rel: isize) -> i32 {
    let off = (edge.topleft as isize + rel * bpp_of::<BPC>() as isize) as usize;
    if BPC == 8 {
        i32::from(edge.bytes[off])
    } else {
        i32::from(u16::from_le_bytes([edge.bytes[off], edge.bytes[off + 1]]))
    }
}

#[inline(always)]
fn read_px<const BPC: u32>(buf: &[u8], off: usize) -> i32 {
    if BPC == 8 {
        i32::from(buf[off])
    } else {
        i32::from(u16::from_le_bytes([buf[off], buf[off + 1]]))
    }
}

#[inline(always)]
fn write_px<const BPC: u32>(dst: &mut [u8], off: usize, val: i32) {
    if BPC == 8 {
        dst[off] = val as u8;
    } else {
        dst[off..off + 2].copy_from_slice(&(val as u16).to_le_bytes());
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn load_px8_u16<const BPC: u32>(buf: &[u8], byte_off: usize) -> uint16x8_t {
    if BPC == 8 {
        vmovl_u8(load_u8x8(chunk::<u8, 8>(buf, byte_off)))
    } else {
        vreinterpretq_u16_u8(load_u8x16(chunk::<u8, 16>(buf, byte_off)))
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn load_px4_u16<const BPC: u32>(buf: &[u8], byte_off: usize) -> uint16x4_t {
    if BPC == 8 {
        let u = u32::from_le_bytes(*chunk::<u8, 4>(buf, byte_off));
        vget_low_u16(vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(u))))
    } else {
        vreinterpret_u16_u8(load_u8x8(chunk::<u8, 8>(buf, byte_off)))
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn store_px4_u16<const BPC: u32>(dst: &mut [u8], byte_off: usize, v: uint16x4_t) {
    if BPC == 8 {
        let u8v = vmovn_u16(vcombine_u16(v, v));
        let w = vget_lane_u32::<0>(vreinterpret_u32_u8(u8v));
        *chunk_mut::<u8, 4>(dst, byte_off) = w.to_ne_bytes();
    } else {
        let w = vget_lane_u64::<0>(vreinterpret_u64_u16(v));
        *chunk_mut::<u8, 8>(dst, byte_off) = w.to_ne_bytes();
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn store_px8_u16<const BPC: u32>(dst: &mut [u8], byte_off: usize, v: uint16x8_t) {
    if BPC == 8 {
        store_u8x8(chunk_mut::<u8, 8>(dst, byte_off), vmovn_u16(v));
    } else {
        store_u8x16(chunk_mut::<u8, 16>(dst, byte_off), vreinterpretq_u8_u16(v));
    }
}

// ============================================================================
// DC / VERT / HOR
// ============================================================================

#[target_feature(enable = "neon")]
#[inline]
fn sum_contiguous<const BPC: u32>(buf: &[u8], byte_off: usize, n_px: usize) -> u32 {
    if BPC == 8 {
        match n_px {
            4 => {
                let b = chunk::<u8, 4>(buf, byte_off);
                u32::from(b[0]) + u32::from(b[1]) + u32::from(b[2]) + u32::from(b[3])
            }
            8 => {
                let v = vcombine_u8(load_u8x8(chunk::<u8, 8>(buf, byte_off)), vdup_n_u8(0));
                u32::from(vaddlvq_u8(v))
            }
            16 => u32::from(vaddlvq_u8(load_u8x16(chunk::<u8, 16>(buf, byte_off)))),
            32 => {
                let v0 = load_u8x16(chunk::<u8, 16>(buf, byte_off));
                let v1 = load_u8x16(chunk::<u8, 16>(buf, byte_off + 16));
                u32::from(vaddlvq_u8(v0)) + u32::from(vaddlvq_u8(v1))
            }
            _ => {
                let mut s = 0u32;
                let mut x = 0usize;
                while x < n_px {
                    s += u32::from(vaddlvq_u8(load_u8x16(chunk::<u8, 16>(buf, byte_off + x))));
                    x += 16;
                }
                s
            }
        }
    } else {
        match n_px {
            4 => {
                let v = vcombine_u16(load_px4_u16::<BPC>(buf, byte_off), vdup_n_u16(0));
                vaddlvq_u16(v)
            }
            8 => vaddlvq_u16(load_px8_u16::<BPC>(buf, byte_off)),
            16 => {
                let v0 = load_px8_u16::<BPC>(buf, byte_off);
                let v1 = load_px8_u16::<BPC>(buf, byte_off + 16);
                vaddlvq_u16(v0) + vaddlvq_u16(v1)
            }
            _ => {
                let mut s = 0u32;
                let mut x = 0usize;
                while x < n_px {
                    let v = load_px8_u16::<BPC>(buf, byte_off + x * 2);
                    s += vaddlvq_u16(v);
                    x += 8;
                }
                s
            }
        }
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn dc_gen_top_neon<const BPC: u32>(edge: Edge<'_>, w: usize) -> u32 {
    let sum = sum_contiguous::<BPC>(edge.bytes, edge.topleft + bpp_of::<BPC>(), w);
    (sum + (w as u32 >> 1)) >> w.trailing_zeros()
}

#[target_feature(enable = "neon")]
#[inline]
fn dc_gen_left_neon<const BPC: u32>(edge: Edge<'_>, h: usize) -> u32 {
    let bpp = bpp_of::<BPC>();
    let sum = sum_contiguous::<BPC>(edge.bytes, edge.topleft - h * bpp, h);
    (sum + (h as u32 >> 1)) >> h.trailing_zeros()
}

#[target_feature(enable = "neon")]
#[inline]
fn dc_gen_neon<const BPC: u32>(edge: Edge<'_>, w: usize, h: usize) -> u32 {
    let bpp = bpp_of::<BPC>();
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

#[target_feature(enable = "neon")]
#[inline]
fn splat_row<const BPC: u32>(dst: &mut [u8], row_off: usize, w: usize, val: u16) {
    if BPC == 8 {
        let b = val as u8;
        match w {
            4 => {
                *chunk_mut::<u8, 4>(dst, row_off) = [b; 4];
            }
            8 => store_u8x8(chunk_mut::<u8, 8>(dst, row_off), vdup_n_u8(b)),
            16 => store_u8x16(chunk_mut::<u8, 16>(dst, row_off), vdupq_n_u8(b)),
            32 => {
                let v = vdupq_n_u8(b);
                store_u8x16(chunk_mut::<u8, 16>(dst, row_off), v);
                store_u8x16(chunk_mut::<u8, 16>(dst, row_off + 16), v);
            }
            _ => {
                let v = vdupq_n_u8(b);
                let mut x = 0usize;
                while x < w {
                    store_u8x16(chunk_mut::<u8, 16>(dst, row_off + x), v);
                    x += 16;
                }
            }
        }
    } else {
        match w {
            4 => store_px4_u16::<BPC>(dst, row_off, vdup_n_u16(val)),
            8 => store_px8_u16::<BPC>(dst, row_off, vdupq_n_u16(val)),
            16 => {
                let v = vreinterpretq_u8_u16(vdupq_n_u16(val));
                store_u8x16(chunk_mut::<u8, 16>(dst, row_off), v);
                store_u8x16(chunk_mut::<u8, 16>(dst, row_off + 16), v);
            }
            _ => {
                let v = vreinterpretq_u8_u16(vdupq_n_u16(val));
                let mut x = 0usize;
                while x < w * 2 {
                    store_u8x16(chunk_mut::<u8, 16>(dst, row_off + x), v);
                    x += 16;
                }
            }
        }
    }
}

#[target_feature(enable = "neon")]
fn splat_dc_neon<const BPC: u32>(dst: &mut [u8], stride: usize, w: usize, h: usize, dc: u16) {
    let bpp = bpp_of::<BPC>();
    let w_bytes = w * bpp;
    let v16 = if BPC == 8 {
        vdupq_n_u8(dc as u8)
    } else {
        vreinterpretq_u8_u16(vdupq_n_u16(dc))
    };
    match w_bytes {
        4 => {
            let b4 = [dc as u8; 4];
            for y in 0..h {
                *chunk_mut::<u8, 4>(dst, y * stride) = b4;
            }
        }
        8 => {
            let v8 = vget_low_u8(v16);
            for y in 0..h {
                store_u8x8(chunk_mut::<u8, 8>(dst, y * stride), v8);
            }
        }
        16 => {
            for y in 0..h {
                store_u8x16(chunk_mut::<u8, 16>(dst, y * stride), v16);
            }
        }
        32 => {
            for y in 0..h {
                let off = y * stride;
                store_u8x16(chunk_mut::<u8, 16>(dst, off), v16);
                store_u8x16(chunk_mut::<u8, 16>(dst, off + 16), v16);
            }
        }
        64 => {
            for y in 0..h {
                let off = y * stride;
                store_u8x16(chunk_mut::<u8, 16>(dst, off), v16);
                store_u8x16(chunk_mut::<u8, 16>(dst, off + 16), v16);
                store_u8x16(chunk_mut::<u8, 16>(dst, off + 32), v16);
                store_u8x16(chunk_mut::<u8, 16>(dst, off + 48), v16);
            }
        }
        _ => {
            for y in 0..h {
                let row_off = y * stride;
                let mut x = 0usize;
                while x < w_bytes {
                    store_u8x16(chunk_mut::<u8, 16>(dst, row_off + x), v16);
                    x += 16;
                }
            }
        }
    }
}

#[target_feature(enable = "neon")]
fn vert_neon<const BPC: u32>(dst: &mut [u8], stride: usize, edge: Edge<'_>, w: usize, h: usize) {
    let bpp = bpp_of::<BPC>();
    let top_off = edge.topleft + bpp;
    let w_bytes = w * bpp;
    match w_bytes {
        4 => {
            let t = *chunk::<u8, 4>(edge.bytes, top_off);
            for y in 0..h {
                *chunk_mut::<u8, 4>(dst, y * stride) = t;
            }
        }
        8 => {
            let v0 = load_u8x8(chunk::<u8, 8>(edge.bytes, top_off));
            for y in 0..h {
                store_u8x8(chunk_mut::<u8, 8>(dst, y * stride), v0);
            }
        }
        16 => {
            let v0 = load_u8x16(chunk::<u8, 16>(edge.bytes, top_off));
            for y in 0..h {
                store_u8x16(chunk_mut::<u8, 16>(dst, y * stride), v0);
            }
        }
        32 => {
            let v0 = load_u8x16(chunk::<u8, 16>(edge.bytes, top_off));
            let v1 = load_u8x16(chunk::<u8, 16>(edge.bytes, top_off + 16));
            for y in 0..h {
                let off = y * stride;
                store_u8x16(chunk_mut::<u8, 16>(dst, off), v0);
                store_u8x16(chunk_mut::<u8, 16>(dst, off + 16), v1);
            }
        }
        64 => {
            let v0 = load_u8x16(chunk::<u8, 16>(edge.bytes, top_off));
            let v1 = load_u8x16(chunk::<u8, 16>(edge.bytes, top_off + 16));
            let v2 = load_u8x16(chunk::<u8, 16>(edge.bytes, top_off + 32));
            let v3 = load_u8x16(chunk::<u8, 16>(edge.bytes, top_off + 48));
            for y in 0..h {
                let off = y * stride;
                store_u8x16(chunk_mut::<u8, 16>(dst, off), v0);
                store_u8x16(chunk_mut::<u8, 16>(dst, off + 16), v1);
                store_u8x16(chunk_mut::<u8, 16>(dst, off + 32), v2);
                store_u8x16(chunk_mut::<u8, 16>(dst, off + 48), v3);
            }
        }
        128 => {
            let v0 = load_u8x16(chunk::<u8, 16>(edge.bytes, top_off));
            let v1 = load_u8x16(chunk::<u8, 16>(edge.bytes, top_off + 16));
            let v2 = load_u8x16(chunk::<u8, 16>(edge.bytes, top_off + 32));
            let v3 = load_u8x16(chunk::<u8, 16>(edge.bytes, top_off + 48));
            let v4 = load_u8x16(chunk::<u8, 16>(edge.bytes, top_off + 64));
            let v5 = load_u8x16(chunk::<u8, 16>(edge.bytes, top_off + 80));
            let v6 = load_u8x16(chunk::<u8, 16>(edge.bytes, top_off + 96));
            let v7 = load_u8x16(chunk::<u8, 16>(edge.bytes, top_off + 112));
            for y in 0..h {
                let off = y * stride;
                store_u8x16(chunk_mut::<u8, 16>(dst, off), v0);
                store_u8x16(chunk_mut::<u8, 16>(dst, off + 16), v1);
                store_u8x16(chunk_mut::<u8, 16>(dst, off + 32), v2);
                store_u8x16(chunk_mut::<u8, 16>(dst, off + 48), v3);
                store_u8x16(chunk_mut::<u8, 16>(dst, off + 64), v4);
                store_u8x16(chunk_mut::<u8, 16>(dst, off + 80), v5);
                store_u8x16(chunk_mut::<u8, 16>(dst, off + 96), v6);
                store_u8x16(chunk_mut::<u8, 16>(dst, off + 112), v7);
            }
        }
        _ => {
            for y in 0..h {
                let row_off = y * stride;
                let mut x = 0usize;
                while x < w_bytes {
                    let v = load_u8x16(chunk::<u8, 16>(edge.bytes, top_off + x));
                    store_u8x16(chunk_mut::<u8, 16>(dst, row_off + x), v);
                    x += 16;
                }
            }
        }
    }
}

#[target_feature(enable = "neon")]
fn hor_neon<const BPC: u32>(dst: &mut [u8], stride: usize, edge: Edge<'_>, w: usize, h: usize) {
    let tl = edge.topleft;
    if BPC == 8 {
        match w {
            4 => {
                let mut y = 0usize;
                while y + 4 <= h {
                    let [b3, b2, b1, b0] = *chunk::<u8, 4>(edge.bytes, tl - 4 - y);
                    let off = y * stride;
                    *chunk_mut::<u8, 4>(dst, off) = [b0; 4];
                    *chunk_mut::<u8, 4>(dst, off + stride) = [b1; 4];
                    *chunk_mut::<u8, 4>(dst, off + 2 * stride) = [b2; 4];
                    *chunk_mut::<u8, 4>(dst, off + 3 * stride) = [b3; 4];
                    y += 4;
                }
                while y < h {
                    let b = edge.bytes[tl - 1 - y];
                    *chunk_mut::<u8, 4>(dst, y * stride) = [b; 4];
                    y += 1;
                }
            }
            8 => {
                let mut y = 0usize;
                while y + 4 <= h {
                    let [b3, b2, b1, b0] = *chunk::<u8, 4>(edge.bytes, tl - 4 - y);
                    let off = y * stride;
                    store_u8x8(chunk_mut::<u8, 8>(dst, off), vdup_n_u8(b0));
                    store_u8x8(chunk_mut::<u8, 8>(dst, off + stride), vdup_n_u8(b1));
                    store_u8x8(chunk_mut::<u8, 8>(dst, off + 2 * stride), vdup_n_u8(b2));
                    store_u8x8(chunk_mut::<u8, 8>(dst, off + 3 * stride), vdup_n_u8(b3));
                    y += 4;
                }
                while y < h {
                    let b = edge.bytes[tl - 1 - y];
                    store_u8x8(chunk_mut::<u8, 8>(dst, y * stride), vdup_n_u8(b));
                    y += 1;
                }
            }
            16 => {
                let mut y = 0usize;
                while y + 4 <= h {
                    let [b3, b2, b1, b0] = *chunk::<u8, 4>(edge.bytes, tl - 4 - y);
                    let off = y * stride;
                    store_u8x16(chunk_mut::<u8, 16>(dst, off), vdupq_n_u8(b0));
                    store_u8x16(chunk_mut::<u8, 16>(dst, off + stride), vdupq_n_u8(b1));
                    store_u8x16(chunk_mut::<u8, 16>(dst, off + 2 * stride), vdupq_n_u8(b2));
                    store_u8x16(chunk_mut::<u8, 16>(dst, off + 3 * stride), vdupq_n_u8(b3));
                    y += 4;
                }
                while y < h {
                    let b = edge.bytes[tl - 1 - y];
                    store_u8x16(chunk_mut::<u8, 16>(dst, y * stride), vdupq_n_u8(b));
                    y += 1;
                }
            }
            32 => {
                for y in 0..h {
                    let v = vdupq_n_u8(edge.bytes[tl - 1 - y]);
                    let off = y * stride;
                    store_u8x16(chunk_mut::<u8, 16>(dst, off), v);
                    store_u8x16(chunk_mut::<u8, 16>(dst, off + 16), v);
                }
            }
            _ => {
                for y in 0..h {
                    let v = vdupq_n_u8(edge.bytes[tl - 1 - y]);
                    let off = y * stride;
                    let mut x = 0usize;
                    while x < w {
                        store_u8x16(chunk_mut::<u8, 16>(dst, off + x), v);
                        x += 16;
                    }
                }
            }
        }
    } else {
        match w {
            4 => {
                for y in 0..h {
                    let v = u16::from_le_bytes(*chunk::<u8, 2>(edge.bytes, tl - 2 * (y + 1)));
                    store_px4_u16::<BPC>(dst, y * stride, vdup_n_u16(v));
                }
            }
            8 => {
                for y in 0..h {
                    let v = u16::from_le_bytes(*chunk::<u8, 2>(edge.bytes, tl - 2 * (y + 1)));
                    store_u8x16(
                        chunk_mut::<u8, 16>(dst, y * stride),
                        vreinterpretq_u8_u16(vdupq_n_u16(v)),
                    );
                }
            }
            16 => {
                for y in 0..h {
                    let v = vreinterpretq_u8_u16(vdupq_n_u16(u16::from_le_bytes(*chunk::<u8, 2>(
                        edge.bytes,
                        tl - 2 * (y + 1),
                    ))));
                    let off = y * stride;
                    store_u8x16(chunk_mut::<u8, 16>(dst, off), v);
                    store_u8x16(chunk_mut::<u8, 16>(dst, off + 16), v);
                }
            }
            32 => {
                for y in 0..h {
                    let v = vreinterpretq_u8_u16(vdupq_n_u16(u16::from_le_bytes(*chunk::<u8, 2>(
                        edge.bytes,
                        tl - 2 * (y + 1),
                    ))));
                    let off = y * stride;
                    store_u8x16(chunk_mut::<u8, 16>(dst, off), v);
                    store_u8x16(chunk_mut::<u8, 16>(dst, off + 16), v);
                    store_u8x16(chunk_mut::<u8, 16>(dst, off + 32), v);
                    store_u8x16(chunk_mut::<u8, 16>(dst, off + 48), v);
                }
            }
            _ => {
                let w_bytes = w * 2;
                for y in 0..h {
                    let v = vreinterpretq_u8_u16(vdupq_n_u16(u16::from_le_bytes(*chunk::<u8, 2>(
                        edge.bytes,
                        tl - 2 * (y + 1),
                    ))));
                    let off = y * stride;
                    let mut x = 0usize;
                    while x < w_bytes {
                        store_u8x16(chunk_mut::<u8, 16>(dst, off + x), v);
                        x += 16;
                    }
                }
            }
        }
    }
}

// ============================================================================
// PAETH / SMOOTH
// ============================================================================

#[target_feature(enable = "neon")]
#[inline]
fn paeth8_step(lv: int16x8_t, top_v: int16x8_t, tl: int16x8_t) -> uint16x8_t {
    let base = vsubq_s16(vaddq_s16(lv, top_v), tl);
    let ld = vabdq_s16(lv, base);
    let td = vabdq_s16(top_v, base);
    let tld = vabdq_s16(tl, base);
    let m_l = vandq_u16(vcleq_s16(ld, td), vcleq_s16(ld, tld));
    let m_t = vcleq_s16(td, tld);
    vreinterpretq_u16_s16(vbslq_s16(m_l, lv, vbslq_s16(m_t, top_v, tl)))
}

#[target_feature(enable = "neon")]
fn paeth_neon<const BPC: u32>(dst: &mut [u8], stride: usize, edge: Edge<'_>, w: usize, h: usize) {
    let bpp = bpp_of::<BPC>();
    let tl = vdupq_n_s16(read_edge::<BPC>(edge, 0) as i16);
    let top_base = edge.topleft + bpp;

    match w {
        4 => {
            let top_v = vreinterpretq_s16_u16(vcombine_u16(
                load_px4_u16::<BPC>(edge.bytes, top_base),
                vdup_n_u16(0),
            ));
            for y in 0..h {
                let lv = vdupq_n_s16(read_edge::<BPC>(edge, -(y as isize + 1)) as i16);
                let out = paeth8_step(lv, top_v, tl);
                store_px4_u16::<BPC>(dst, y * stride, vget_low_u16(out));
            }
        }
        8 => {
            let t0 = vreinterpretq_s16_u16(load_px8_u16::<BPC>(edge.bytes, top_base));
            for y in 0..h {
                let lv = vdupq_n_s16(read_edge::<BPC>(edge, -(y as isize + 1)) as i16);
                store_px8_u16::<BPC>(dst, y * stride, paeth8_step(lv, t0, tl));
            }
        }
        16 => {
            let t0 = vreinterpretq_s16_u16(load_px8_u16::<BPC>(edge.bytes, top_base));
            let t1 = vreinterpretq_s16_u16(load_px8_u16::<BPC>(edge.bytes, top_base + 8 * bpp));
            for y in 0..h {
                let lv = vdupq_n_s16(read_edge::<BPC>(edge, -(y as isize + 1)) as i16);
                let off = y * stride;
                store_px8_u16::<BPC>(dst, off, paeth8_step(lv, t0, tl));
                store_px8_u16::<BPC>(dst, off + 8 * bpp, paeth8_step(lv, t1, tl));
            }
        }
        32 => {
            let t0 = vreinterpretq_s16_u16(load_px8_u16::<BPC>(edge.bytes, top_base));
            let t1 = vreinterpretq_s16_u16(load_px8_u16::<BPC>(edge.bytes, top_base + 8 * bpp));
            let t2 = vreinterpretq_s16_u16(load_px8_u16::<BPC>(edge.bytes, top_base + 16 * bpp));
            let t3 = vreinterpretq_s16_u16(load_px8_u16::<BPC>(edge.bytes, top_base + 24 * bpp));
            for y in 0..h {
                let lv = vdupq_n_s16(read_edge::<BPC>(edge, -(y as isize + 1)) as i16);
                let off = y * stride;
                store_px8_u16::<BPC>(dst, off, paeth8_step(lv, t0, tl));
                store_px8_u16::<BPC>(dst, off + 8 * bpp, paeth8_step(lv, t1, tl));
                store_px8_u16::<BPC>(dst, off + 16 * bpp, paeth8_step(lv, t2, tl));
                store_px8_u16::<BPC>(dst, off + 24 * bpp, paeth8_step(lv, t3, tl));
            }
        }
        _ => {
            for y in 0..h {
                let lv = vdupq_n_s16(read_edge::<BPC>(edge, -(y as isize + 1)) as i16);
                let row_off = y * stride;
                let mut x = 0usize;
                while x < w {
                    let top_v =
                        vreinterpretq_s16_u16(load_px8_u16::<BPC>(edge.bytes, top_base + x * bpp));
                    store_px8_u16::<BPC>(dst, row_off + x * bpp, paeth8_step(lv, top_v, tl));
                    x += 8;
                }
            }
        }
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn smooth_prep8<const BPC: u32>(
    edge_bytes: &[u8],
    top_off: usize,
    wh: &[u8],
    x: usize,
    right_v: uint16x8_t,
    c256: uint16x8_t,
) -> (uint16x8_t, uint16x8_t, uint32x4_t, uint32x4_t) {
    let top_v = load_px8_u16::<BPC>(edge_bytes, top_off);
    let wh_v = vmovl_u8(load_u8x8(chunk::<u8, 8>(wh, x)));
    let inv_wh_v = vsubq_u16(c256, wh_v);
    let r_lo = vmull_u16(vget_low_u16(inv_wh_v), vget_low_u16(right_v));
    let r_hi = vmull_high_u16(inv_wh_v, right_v);
    (top_v, wh_v, r_lo, r_hi)
}

#[target_feature(enable = "neon")]
#[inline]
fn smooth_step8(
    top_v: uint16x8_t,
    wh_v: uint16x8_t,
    r_lo: uint32x4_t,
    r_hi: uint32x4_t,
    bot_term: uint32x4_t,
    wy_v: uint16x8_t,
    left_v: uint16x8_t,
) -> uint16x8_t {
    let mut acc_lo = vmlal_u16(
        vaddq_u32(r_lo, bot_term),
        vget_low_u16(top_v),
        vget_low_u16(wy_v),
    );
    acc_lo = vmlal_u16(acc_lo, vget_low_u16(wh_v), vget_low_u16(left_v));
    let mut acc_hi = vmlal_high_u16(vaddq_u32(r_hi, bot_term), top_v, wy_v);
    acc_hi = vmlal_high_u16(acc_hi, wh_v, left_v);
    vrshrn_high_n_u32::<9>(vrshrn_n_u32::<9>(acc_lo), acc_hi)
}

#[target_feature(enable = "neon")]
fn smooth_neon<const BPC: u32>(dst: &mut [u8], stride: usize, edge: Edge<'_>, w: usize, h: usize) {
    let bpp = bpp_of::<BPC>();
    let wh = &SM_WEIGHTS[w..w + w];
    let wv = &SM_WEIGHTS[h..h + h];
    let right = read_edge::<BPC>(edge, w as isize) as u16;
    let bottom = read_edge::<BPC>(edge, -(h as isize)) as u16;
    let top_base = edge.topleft + bpp;
    let c256 = vdupq_n_u16(256);

    if w == 4 {
        let top_v = load_px4_u16::<BPC>(edge.bytes, top_base);
        let wh4 = *chunk::<u8, 4>(wh, 0);
        let wh_arr = [
            u16::from(wh4[0]),
            u16::from(wh4[1]),
            u16::from(wh4[2]),
            u16::from(wh4[3]),
        ];
        let wh_v = load_u16x4(&wh_arr);
        let inv_wh_v = load_u16x4(&[
            256 - wh_arr[0],
            256 - wh_arr[1],
            256 - wh_arr[2],
            256 - wh_arr[3],
        ]);
        let right_term = vmull_u16(inv_wh_v, vdup_n_u16(right));
        let bottom4 = vdup_n_u16(bottom);

        for y in 0..h {
            let wy = u16::from(wv[y]);
            let left = read_edge::<BPC>(edge, -(y as isize + 1)) as u16;
            let mut acc = vmlal_u16(right_term, top_v, vdup_n_u16(wy));
            acc = vmlal_u16(acc, bottom4, vdup_n_u16(256 - wy));
            acc = vmlal_u16(acc, wh_v, vdup_n_u16(left));
            store_px4_u16::<BPC>(dst, y * stride, vrshrn_n_u32::<9>(acc));
        }
        return;
    }

    let right_v = vdupq_n_u16(right);
    let bottom4 = vdup_n_u16(bottom);

    match w {
        8 => {
            let (t0, wh0, rl0, rh0) =
                smooth_prep8::<BPC>(edge.bytes, top_base, wh, 0, right_v, c256);
            for y in 0..h {
                let wy = u16::from(wv[y]);
                let wy_v = vdupq_n_u16(wy);
                let bot_term = vmull_u16(bottom4, vdup_n_u16(256 - wy));
                let left_v = vdupq_n_u16(read_edge::<BPC>(edge, -(y as isize + 1)) as u16);
                let out = smooth_step8(t0, wh0, rl0, rh0, bot_term, wy_v, left_v);
                store_px8_u16::<BPC>(dst, y * stride, out);
            }
        }
        16 => {
            let (t0, wh0, rl0, rh0) =
                smooth_prep8::<BPC>(edge.bytes, top_base, wh, 0, right_v, c256);
            let (t1, wh1, rl1, rh1) =
                smooth_prep8::<BPC>(edge.bytes, top_base + 8 * bpp, wh, 8, right_v, c256);
            for y in 0..h {
                let wy = u16::from(wv[y]);
                let wy_v = vdupq_n_u16(wy);
                let bot_term = vmull_u16(bottom4, vdup_n_u16(256 - wy));
                let left_v = vdupq_n_u16(read_edge::<BPC>(edge, -(y as isize + 1)) as u16);
                let off = y * stride;
                store_px8_u16::<BPC>(
                    dst,
                    off,
                    smooth_step8(t0, wh0, rl0, rh0, bot_term, wy_v, left_v),
                );
                store_px8_u16::<BPC>(
                    dst,
                    off + 8 * bpp,
                    smooth_step8(t1, wh1, rl1, rh1, bot_term, wy_v, left_v),
                );
            }
        }
        32 => {
            let (t0, wh0, rl0, rh0) =
                smooth_prep8::<BPC>(edge.bytes, top_base, wh, 0, right_v, c256);
            let (t1, wh1, rl1, rh1) =
                smooth_prep8::<BPC>(edge.bytes, top_base + 8 * bpp, wh, 8, right_v, c256);
            let (t2, wh2, rl2, rh2) =
                smooth_prep8::<BPC>(edge.bytes, top_base + 16 * bpp, wh, 16, right_v, c256);
            let (t3, wh3, rl3, rh3) =
                smooth_prep8::<BPC>(edge.bytes, top_base + 24 * bpp, wh, 24, right_v, c256);
            for y in 0..h {
                let wy = u16::from(wv[y]);
                let wy_v = vdupq_n_u16(wy);
                let bot_term = vmull_u16(bottom4, vdup_n_u16(256 - wy));
                let left_v = vdupq_n_u16(read_edge::<BPC>(edge, -(y as isize + 1)) as u16);
                let off = y * stride;
                store_px8_u16::<BPC>(
                    dst,
                    off,
                    smooth_step8(t0, wh0, rl0, rh0, bot_term, wy_v, left_v),
                );
                store_px8_u16::<BPC>(
                    dst,
                    off + 8 * bpp,
                    smooth_step8(t1, wh1, rl1, rh1, bot_term, wy_v, left_v),
                );
                store_px8_u16::<BPC>(
                    dst,
                    off + 16 * bpp,
                    smooth_step8(t2, wh2, rl2, rh2, bot_term, wy_v, left_v),
                );
                store_px8_u16::<BPC>(
                    dst,
                    off + 24 * bpp,
                    smooth_step8(t3, wh3, rl3, rh3, bot_term, wy_v, left_v),
                );
            }
        }
        _ => {
            for y in 0..h {
                let wy = u16::from(wv[y]);
                let wy_v = vdupq_n_u16(wy);
                let bot_term = vmull_u16(bottom4, vdup_n_u16(256 - wy));
                let left_v = vdupq_n_u16(read_edge::<BPC>(edge, -(y as isize + 1)) as u16);
                let row_off = y * stride;

                let mut x = 0usize;
                while x < w {
                    let (t_v, wh_v, rl, rh) =
                        smooth_prep8::<BPC>(edge.bytes, top_base + x * bpp, wh, x, right_v, c256);
                    let out = smooth_step8(t_v, wh_v, rl, rh, bot_term, wy_v, left_v);
                    store_px8_u16::<BPC>(dst, row_off + x * bpp, out);
                    x += 8;
                }
            }
        }
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn smooth_v_step8(top_v: uint16x8_t, bot_term: uint32x4_t, wy_v: uint16x8_t) -> uint16x8_t {
    let acc_lo = vmlal_u16(bot_term, vget_low_u16(top_v), vget_low_u16(wy_v));
    let acc_hi = vmlal_high_u16(bot_term, top_v, wy_v);
    vrshrn_high_n_u32::<8>(vrshrn_n_u32::<8>(acc_lo), acc_hi)
}

#[target_feature(enable = "neon")]
fn smooth_v_neon<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    w: usize,
    h: usize,
) {
    let bpp = bpp_of::<BPC>();
    let wv = &SM_WEIGHTS[h..h + h];
    let bottom4 = vdup_n_u16(read_edge::<BPC>(edge, -(h as isize)) as u16);
    let top_base = edge.topleft + bpp;

    match w {
        4 => {
            let top_v = load_px4_u16::<BPC>(edge.bytes, top_base);
            for y in 0..h {
                let wy = u16::from(wv[y]);
                let acc = vmlal_u16(
                    vmull_u16(bottom4, vdup_n_u16(256 - wy)),
                    top_v,
                    vdup_n_u16(wy),
                );
                store_px4_u16::<BPC>(dst, y * stride, vrshrn_n_u32::<8>(acc));
            }
        }
        8 => {
            let t0 = load_px8_u16::<BPC>(edge.bytes, top_base);
            for y in 0..h {
                let wy = u16::from(wv[y]);
                let wy_v = vdupq_n_u16(wy);
                let bot_term = vmull_u16(bottom4, vdup_n_u16(256 - wy));
                store_px8_u16::<BPC>(dst, y * stride, smooth_v_step8(t0, bot_term, wy_v));
            }
        }
        16 => {
            let t0 = load_px8_u16::<BPC>(edge.bytes, top_base);
            let t1 = load_px8_u16::<BPC>(edge.bytes, top_base + 8 * bpp);
            for y in 0..h {
                let wy = u16::from(wv[y]);
                let wy_v = vdupq_n_u16(wy);
                let bot_term = vmull_u16(bottom4, vdup_n_u16(256 - wy));
                let off = y * stride;
                store_px8_u16::<BPC>(dst, off, smooth_v_step8(t0, bot_term, wy_v));
                store_px8_u16::<BPC>(dst, off + 8 * bpp, smooth_v_step8(t1, bot_term, wy_v));
            }
        }
        32 => {
            let t0 = load_px8_u16::<BPC>(edge.bytes, top_base);
            let t1 = load_px8_u16::<BPC>(edge.bytes, top_base + 8 * bpp);
            let t2 = load_px8_u16::<BPC>(edge.bytes, top_base + 16 * bpp);
            let t3 = load_px8_u16::<BPC>(edge.bytes, top_base + 24 * bpp);
            for y in 0..h {
                let wy = u16::from(wv[y]);
                let wy_v = vdupq_n_u16(wy);
                let bot_term = vmull_u16(bottom4, vdup_n_u16(256 - wy));
                let off = y * stride;
                store_px8_u16::<BPC>(dst, off, smooth_v_step8(t0, bot_term, wy_v));
                store_px8_u16::<BPC>(dst, off + 8 * bpp, smooth_v_step8(t1, bot_term, wy_v));
                store_px8_u16::<BPC>(dst, off + 16 * bpp, smooth_v_step8(t2, bot_term, wy_v));
                store_px8_u16::<BPC>(dst, off + 24 * bpp, smooth_v_step8(t3, bot_term, wy_v));
            }
        }
        _ => {
            for y in 0..h {
                let wy = u16::from(wv[y]);
                let wy_v = vdupq_n_u16(wy);
                let bot_term = vmull_u16(bottom4, vdup_n_u16(256 - wy));
                let row_off = y * stride;
                let mut x = 0usize;
                while x < w {
                    let top_v = load_px8_u16::<BPC>(edge.bytes, top_base + x * bpp);
                    store_px8_u16::<BPC>(
                        dst,
                        row_off + x * bpp,
                        smooth_v_step8(top_v, bot_term, wy_v),
                    );
                    x += 8;
                }
            }
        }
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn smooth_h_prep8(
    wh: &[u8],
    x: usize,
    right_v: uint16x8_t,
    c256: uint16x8_t,
) -> (uint16x8_t, uint32x4_t, uint32x4_t) {
    let wh_v = vmovl_u8(load_u8x8(chunk::<u8, 8>(wh, x)));
    let inv_wh_v = vsubq_u16(c256, wh_v);
    let r_lo = vmull_u16(vget_low_u16(inv_wh_v), vget_low_u16(right_v));
    let r_hi = vmull_high_u16(inv_wh_v, right_v);
    (wh_v, r_lo, r_hi)
}

#[target_feature(enable = "neon")]
#[inline]
fn smooth_h_step8(
    wh_v: uint16x8_t,
    r_lo: uint32x4_t,
    r_hi: uint32x4_t,
    left_v: uint16x8_t,
) -> uint16x8_t {
    let acc_lo = vmlal_u16(r_lo, vget_low_u16(wh_v), vget_low_u16(left_v));
    let acc_hi = vmlal_high_u16(r_hi, wh_v, left_v);
    vrshrn_high_n_u32::<8>(vrshrn_n_u32::<8>(acc_lo), acc_hi)
}

#[target_feature(enable = "neon")]
fn smooth_h_neon<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    w: usize,
    h: usize,
) {
    let bpp = bpp_of::<BPC>();
    let wh = &SM_WEIGHTS[w..w + w];
    let right = read_edge::<BPC>(edge, w as isize) as u16;
    let c256 = vdupq_n_u16(256);

    if w == 4 {
        let wh4 = *chunk::<u8, 4>(wh, 0);
        let wh_arr = [
            u16::from(wh4[0]),
            u16::from(wh4[1]),
            u16::from(wh4[2]),
            u16::from(wh4[3]),
        ];
        let wh_v = load_u16x4(&wh_arr);
        let inv_wh_v = load_u16x4(&[
            256 - wh_arr[0],
            256 - wh_arr[1],
            256 - wh_arr[2],
            256 - wh_arr[3],
        ]);
        let right_term = vmull_u16(inv_wh_v, vdup_n_u16(right));
        for y in 0..h {
            let left = read_edge::<BPC>(edge, -(y as isize + 1)) as u16;
            let acc = vmlal_u16(right_term, wh_v, vdup_n_u16(left));
            store_px4_u16::<BPC>(dst, y * stride, vrshrn_n_u32::<8>(acc));
        }
        return;
    }

    let right_v = vdupq_n_u16(right);
    match w {
        8 => {
            let (wh0, rl0, rh0) = smooth_h_prep8(wh, 0, right_v, c256);
            for y in 0..h {
                let left_v = vdupq_n_u16(read_edge::<BPC>(edge, -(y as isize + 1)) as u16);
                store_px8_u16::<BPC>(dst, y * stride, smooth_h_step8(wh0, rl0, rh0, left_v));
            }
        }
        16 => {
            let (wh0, rl0, rh0) = smooth_h_prep8(wh, 0, right_v, c256);
            let (wh1, rl1, rh1) = smooth_h_prep8(wh, 8, right_v, c256);
            for y in 0..h {
                let left_v = vdupq_n_u16(read_edge::<BPC>(edge, -(y as isize + 1)) as u16);
                let off = y * stride;
                store_px8_u16::<BPC>(dst, off, smooth_h_step8(wh0, rl0, rh0, left_v));
                store_px8_u16::<BPC>(dst, off + 8 * bpp, smooth_h_step8(wh1, rl1, rh1, left_v));
            }
        }
        32 => {
            let (wh0, rl0, rh0) = smooth_h_prep8(wh, 0, right_v, c256);
            let (wh1, rl1, rh1) = smooth_h_prep8(wh, 8, right_v, c256);
            let (wh2, rl2, rh2) = smooth_h_prep8(wh, 16, right_v, c256);
            let (wh3, rl3, rh3) = smooth_h_prep8(wh, 24, right_v, c256);
            for y in 0..h {
                let left_v = vdupq_n_u16(read_edge::<BPC>(edge, -(y as isize + 1)) as u16);
                let off = y * stride;
                store_px8_u16::<BPC>(dst, off, smooth_h_step8(wh0, rl0, rh0, left_v));
                store_px8_u16::<BPC>(dst, off + 8 * bpp, smooth_h_step8(wh1, rl1, rh1, left_v));
                store_px8_u16::<BPC>(dst, off + 16 * bpp, smooth_h_step8(wh2, rl2, rh2, left_v));
                store_px8_u16::<BPC>(dst, off + 24 * bpp, smooth_h_step8(wh3, rl3, rh3, left_v));
            }
        }
        _ => {
            for y in 0..h {
                let left_v = vdupq_n_u16(read_edge::<BPC>(edge, -(y as isize + 1)) as u16);
                let row_off = y * stride;
                let mut x = 0usize;
                while x < w {
                    let (wh_v, rl, rh) = smooth_h_prep8(wh, x, right_v, c256);
                    store_px8_u16::<BPC>(
                        dst,
                        row_off + x * bpp,
                        smooth_h_step8(wh_v, rl, rh, left_v),
                    );
                    x += 8;
                }
            }
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

#[target_feature(enable = "neon")]
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
    let bpp = bpp_of::<BPC>();
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

    while i + 8 <= filter_end && i < safe_limit {
        let base_px = in_base + i as isize;
        let byte_0 = (edge.topleft as isize + base_px * bpp as isize) as usize;
        let p_m1 = load_px8_u16::<BPC>(edge.bytes, byte_0 - bpp);
        let p_0 = load_px8_u16::<BPC>(edge.bytes, byte_0);
        let p_p1 = load_px8_u16::<BPC>(edge.bytes, byte_0 + bpp);
        let res_u16 = match strength {
            1 => {
                let s = vaddq_u16(vaddq_u16(p_m1, p_p1), vshlq_n_u16::<1>(p_0));
                vrshrq_n_u16::<2>(s)
            }
            2 => {
                let s = vmlaq_n_u16(vmulq_n_u16(vaddq_u16(p_m1, p_p1), 5), p_0, 6);
                vrshrq_n_u16::<4>(s)
            }
            _ => {
                let p_m2 = load_px8_u16::<BPC>(edge.bytes, byte_0 - 2 * bpp);
                let p_p2 = load_px8_u16::<BPC>(edge.bytes, byte_0 + 2 * bpp);
                let outer = vaddq_u16(p_m2, p_p2);
                let inner = vshlq_n_u16::<1>(vaddq_u16(vaddq_u16(p_m1, p_0), p_p1));
                vrshrq_n_u16::<3>(vaddq_u16(outer, inner))
            }
        };
        store_i16x8(chunk_mut::<i16, 8>(out, i), vreinterpretq_s16_u16(res_u16));
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

#[target_feature(enable = "neon")]
fn upsample_edge_i16<const BPC: u32>(
    out: &mut [i16],
    hsz: usize,
    edge: Edge<'_>,
    in_base: isize,
    from: isize,
    to: isize,
) {
    const KERNEL: [i32; 4] = [-1, 9, 9, -1];
    let bpp = bpp_of::<BPC>();
    let max = bd_max_of::<BPC>();
    let zero32 = vdupq_n_s32(0);
    let max32 = vdupq_n_s32(max);

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
        let v_m1 = load_px8_u16::<BPC>(edge.bytes, byte_0 - bpp);
        let v_0 = load_px8_u16::<BPC>(edge.bytes, byte_0);
        let v_1 = load_px8_u16::<BPC>(edge.bytes, byte_0 + bpp);
        let v_2 = load_px8_u16::<BPC>(edge.bytes, byte_0 + 2 * bpp);

        let mid_lo = vaddq_u32(vmovl_u16(vget_low_u16(v_0)), vmovl_u16(vget_low_u16(v_1)));
        let mid_hi = vaddq_u32(vmovl_high_u16(v_0), vmovl_high_u16(v_1));
        let out_lo = vaddq_u32(vmovl_u16(vget_low_u16(v_m1)), vmovl_u16(vget_low_u16(v_2)));
        let out_hi = vaddq_u32(vmovl_high_u16(v_m1), vmovl_high_u16(v_2));

        let s_lo = vsubq_s32(
            vreinterpretq_s32_u32(vmulq_n_u32(mid_lo, 9)),
            vreinterpretq_s32_u32(out_lo),
        );
        let s_hi = vsubq_s32(
            vreinterpretq_s32_u32(vmulq_n_u32(mid_hi, 9)),
            vreinterpretq_s32_u32(out_hi),
        );

        let q_lo = vminq_s32(vmaxq_s32(vrshrq_n_s32::<4>(s_lo), zero32), max32);
        let q_hi = vminq_s32(vmaxq_s32(vrshrq_n_s32::<4>(s_hi), zero32), max32);
        let odd = vreinterpretq_s16_u16(vmovn_high_u32(
            vmovn_u32(vreinterpretq_u32_s32(q_lo)),
            vreinterpretq_u32_s32(q_hi),
        ));
        let e = vreinterpretq_s16_u16(v_0);
        store_i16x8(chunk_mut::<i16, 8>(out, 2 * i), vzip1q_s16(e, odd));
        store_i16x8(chunk_mut::<i16, 8>(out, 2 * i + 8), vzip2q_s16(e, odd));
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

#[target_feature(enable = "neon")]
#[inline]
fn interp8_s16(b0: int16x8_t, b1: int16x8_t, w0: i16, w1: i16) -> uint16x8_t {
    let u0 = vreinterpretq_u16_s16(b0);
    let u1 = vreinterpretq_u16_s16(b1);
    let w0v = vdupq_n_u16(w0 as u16);
    let w1v = vdupq_n_u16(w1 as u16);
    let lo = vmlal_u16(
        vmull_u16(vget_low_u16(u0), vget_low_u16(w0v)),
        vget_low_u16(u1),
        vget_low_u16(w1v),
    );
    let hi = vmlal_high_u16(vmull_high_u16(u0, w0v), u1, w1v);
    vrshrn_high_n_u32::<6>(vrshrn_n_u32::<6>(lo), hi)
}

#[target_feature(enable = "neon")]
#[inline]
fn copy_edge_contiguous_i16<const BPC: u32>(
    out: &mut [i16],
    edge: Edge<'_>,
    byte_start: usize,
    len: usize,
) {
    let bpp = bpp_of::<BPC>();
    let mut i = 0usize;
    while i + 8 <= len {
        let v = vreinterpretq_s16_u16(load_px8_u16::<BPC>(edge.bytes, byte_start + i * bpp));
        store_i16x8(chunk_mut::<i16, 8>(out, i), v);
        i += 8;
    }
    while i < len {
        out[i] = read_px::<BPC>(edge.bytes, byte_start + i * bpp) as i16;
        i += 1;
    }
}

#[target_feature(enable = "neon")]
fn z1_neon<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    w: usize,
    h: usize,
    angle_in: usize,
) {
    let bpp = bpp_of::<BPC>();
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
        let frac = (xpos & 0x3E) as i16;
        let w0 = 64 - frac;
        let base = ((xpos >> 6) as usize).min(max_base_x + 1);
        let row_off = y * stride;

        if base >= max_base_x {
            splat_row::<BPC>(dst, row_off, w, fill as u16);
        } else if base_inc == 1 {
            if w == 4 {
                let b0 = load_i16x8(chunk::<i16, 8>(&top_out, base));
                let b1 = load_i16x8(chunk::<i16, 8>(&top_out, base + 1));
                let out = interp8_s16(b0, b1, w0, frac);
                store_px4_u16::<BPC>(dst, row_off, vget_low_u16(out));
            } else {
                let mut x = 0usize;
                while x < w {
                    let idx = (base + x).min(max_base_x + 1);
                    let b0 = load_i16x8(chunk::<i16, 8>(&top_out, idx));
                    let b1 = load_i16x8(chunk::<i16, 8>(&top_out, idx + 1));
                    let out = interp8_s16(b0, b1, w0, frac);
                    store_px8_u16::<BPC>(dst, row_off + x * bpp, out);
                    x += 8;
                }
            }
        } else if w == 4 {
            let raw = load_i16x8(chunk::<i16, 8>(&top_out, base));
            let b0 = vuzp1q_s16(raw, raw);
            let b1 = vuzp2q_s16(raw, raw);
            let out = interp8_s16(b0, b1, w0, frac);
            store_px4_u16::<BPC>(dst, row_off, vget_low_u16(out));
        } else {
            let mut x = 0usize;
            while x < w {
                let idx = (base + 2 * x).min(max_base_x + 1);
                let r0 = load_i16x8(chunk::<i16, 8>(&top_out, idx));
                let r1 = load_i16x8(chunk::<i16, 8>(&top_out, idx + 8));
                let b0 = vuzp1q_s16(r0, r1);
                let b1 = vuzp2q_s16(r0, r1);
                let out = interp8_s16(b0, b1, w0, frac);
                store_px8_u16::<BPC>(dst, row_off + x * bpp, out);
                x += 8;
            }
        }
        xpos += dx;
    }
}

#[target_feature(enable = "neon")]
fn z2_neon<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    w: usize,
    h: usize,
    angle_in: usize,
    max_width: usize,
    max_height: usize,
) {
    let bpp = bpp_of::<BPC>();
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

    // Sol kenar (`base_x < 0`) için y'den bağımsız kesir ağırlıkları ve satır-içi kaymaları hazırla.
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
            let s = chunk::<isize, 8>(&y_step_shift, x);
            let i0 = (row_left - s[0]) as usize;
            let i1 = (row_left - s[1]) as usize;
            let i2 = (row_left - s[2]) as usize;
            let i3 = (row_left - s[3]) as usize;
            let i4 = (row_left - s[4]) as usize;
            let i5 = (row_left - s[5]) as usize;
            let i6 = (row_left - s[6]) as usize;
            let i7 = (row_left - s[7]) as usize;
            let b0 = load_u16x8(&[
                buffer[i0] as u16,
                buffer[i1] as u16,
                buffer[i2] as u16,
                buffer[i3] as u16,
                buffer[i4] as u16,
                buffer[i5] as u16,
                buffer[i6] as u16,
                buffer[i7] as u16,
            ]);
            let b1 = load_u16x8(&[
                buffer[i0 - 1] as u16,
                buffer[i1 - 1] as u16,
                buffer[i2 - 1] as u16,
                buffer[i3 - 1] as u16,
                buffer[i4 - 1] as u16,
                buffer[i5 - 1] as u16,
                buffer[i6 - 1] as u16,
                buffer[i7 - 1] as u16,
            ]);
            let w0v = load_u16x8(chunk::<u16, 8>(&y_w0, x));
            let w1v = load_u16x8(chunk::<u16, 8>(&y_w1, x));
            let lo = vmlal_u16(
                vmull_u16(vget_low_u16(b0), vget_low_u16(w0v)),
                vget_low_u16(b1),
                vget_low_u16(w1v),
            );
            let hi = vmlal_high_u16(vmull_high_u16(b0, w0v), b1, w1v);
            let out = vrshrn_high_n_u32::<6>(vrshrn_n_u32::<6>(lo), hi);
            store_px8_u16::<BPC>(dst, row_off + x * bpp, out);
            base_x += 8 * base_inc_x;
            x += 8;
        }

        if x + 4 <= w && base_x + 3 * base_inc_x < 0 {
            let s = chunk::<isize, 4>(&y_step_shift, x);
            let i0 = (row_left - s[0]) as usize;
            let i1 = (row_left - s[1]) as usize;
            let i2 = (row_left - s[2]) as usize;
            let i3 = (row_left - s[3]) as usize;
            let b0 = load_u16x4(&[
                buffer[i0] as u16,
                buffer[i1] as u16,
                buffer[i2] as u16,
                buffer[i3] as u16,
            ]);
            let b1 = load_u16x4(&[
                buffer[i0 - 1] as u16,
                buffer[i1 - 1] as u16,
                buffer[i2 - 1] as u16,
                buffer[i3 - 1] as u16,
            ]);
            let w0v = load_u16x4(chunk::<u16, 4>(&y_w0, x));
            let w1v = load_u16x4(chunk::<u16, 4>(&y_w1, x));
            let acc = vmlal_u16(vmull_u16(b0, w0v), b1, w1v);
            store_px4_u16::<BPC>(dst, row_off + x * bpp, vrshrn_n_u32::<6>(acc));
            base_x += 4 * base_inc_x;
            x += 4;
        }

        while x < w && base_x < 0 {
            let idx = (row_left - y_step_shift[x]) as usize;
            let val = i32::from(buffer[idx]) * i32::from(y_w0[x])
                + i32::from(buffer[idx - 1]) * i32::from(y_w1[x]);
            write_px::<BPC>(dst, row_off + x * bpp, (val + 32) >> 6);
            base_x += base_inc_x;
            x += 1;
        }

        let w0 = (64 - frac_x) as i16;
        let w1 = frac_x as i16;
        if base_inc_x == 1 {
            while x + 8 <= w {
                let idx = (base + base_x) as usize;
                let b0 = load_i16x8(chunk::<i16, 8>(&buffer, idx));
                let b1 = load_i16x8(chunk::<i16, 8>(&buffer, idx + 1));
                let out = interp8_s16(b0, b1, w0, w1);
                store_px8_u16::<BPC>(dst, row_off + x * bpp, out);
                base_x += 8;
                x += 8;
            }
        } else {
            while x + 8 <= w {
                let idx = (base + base_x) as usize;
                let r0 = load_i16x8(chunk::<i16, 8>(&buffer, idx));
                let r1 = load_i16x8(chunk::<i16, 8>(&buffer, idx + 8));
                let b0 = vuzp1q_s16(r0, r1);
                let b1 = vuzp2q_s16(r0, r1);
                let out = interp8_s16(b0, b1, w0, w1);
                store_px8_u16::<BPC>(dst, row_off + x * bpp, out);
                base_x += 16;
                x += 8;
            }
        }

        while x < w {
            let idx = (base + base_x) as usize;
            let val = i32::from(buffer[idx]) * (64 - frac_x) + i32::from(buffer[idx + 1]) * frac_x;
            write_px::<BPC>(dst, row_off + x * bpp, (val + 32) >> 6);
            base_x += base_inc_x;
            x += 1;
        }
        xpos -= dx as isize;
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn transpose_store_4x4_u16<const BPC: u32>(
    dst: &mut [u8],
    base_off: usize,
    stride: usize,
    c0: uint16x4_t,
    c1: uint16x4_t,
    c2: uint16x4_t,
    c3: uint16x4_t,
) {
    let t0 = vtrn1_u16(c0, c1);
    let t1 = vtrn2_u16(c0, c1);
    let t2 = vtrn1_u16(c2, c3);
    let t3 = vtrn2_u16(c2, c3);
    let r0 = vreinterpret_u16_u32(vtrn1_u32(
        vreinterpret_u32_u16(t0),
        vreinterpret_u32_u16(t2),
    ));
    let r2 = vreinterpret_u16_u32(vtrn2_u32(
        vreinterpret_u32_u16(t0),
        vreinterpret_u32_u16(t2),
    ));
    let r1 = vreinterpret_u16_u32(vtrn1_u32(
        vreinterpret_u32_u16(t1),
        vreinterpret_u32_u16(t3),
    ));
    let r3 = vreinterpret_u16_u32(vtrn2_u32(
        vreinterpret_u32_u16(t1),
        vreinterpret_u32_u16(t3),
    ));
    store_px4_u16::<BPC>(dst, base_off, r0);
    store_px4_u16::<BPC>(dst, base_off + stride, r1);
    store_px4_u16::<BPC>(dst, base_off + 2 * stride, r2);
    store_px4_u16::<BPC>(dst, base_off + 3 * stride, r3);
}

#[target_feature(enable = "neon")]
#[inline]
fn z3_col8(
    rev_left: &[i16; 288],
    ypos: i32,
    y: usize,
    base_inc: usize,
    max_base_y: usize,
    fill_v: uint16x8_t,
) -> uint16x8_t {
    let frac = (ypos & 0x3E) as i16;
    let w0 = 64 - frac;
    let base = ((ypos >> 6) as usize).min(max_base_y + 1);
    if base >= max_base_y {
        fill_v
    } else if base_inc == 1 {
        let idx = (base + y).min(max_base_y + 1);
        let b0 = load_i16x8(chunk::<i16, 8>(rev_left, idx));
        let b1 = load_i16x8(chunk::<i16, 8>(rev_left, idx + 1));
        interp8_s16(b0, b1, w0, frac)
    } else {
        let idx = (base + 2 * y).min(max_base_y + 1);
        let r0 = load_i16x8(chunk::<i16, 8>(rev_left, idx));
        let r1 = load_i16x8(chunk::<i16, 8>(rev_left, idx + 8));
        interp8_s16(vuzp1q_s16(r0, r1), vuzp2q_s16(r0, r1), w0, frac)
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn z3_col4(
    rev_left: &[i16; 288],
    ypos: i32,
    base_inc: usize,
    max_base_y: usize,
    fill_v: uint16x4_t,
) -> uint16x4_t {
    let frac = (ypos & 0x3E) as i16;
    let w0 = 64 - frac;
    let base = ((ypos >> 6) as usize).min(max_base_y + 1);
    if base >= max_base_y {
        fill_v
    } else if base_inc == 1 {
        let b0 = load_i16x8(chunk::<i16, 8>(rev_left, base));
        let b1 = load_i16x8(chunk::<i16, 8>(rev_left, base + 1));
        vget_low_u16(interp8_s16(b0, b1, w0, frac))
    } else {
        let raw = load_i16x8(chunk::<i16, 8>(rev_left, base));
        vget_low_u16(interp8_s16(
            vuzp1q_s16(raw, raw),
            vuzp2q_s16(raw, raw),
            w0,
            frac,
        ))
    }
}

#[target_feature(enable = "neon")]
fn z3_neon<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    w: usize,
    h: usize,
    angle_in: usize,
) {
    let bpp = bpp_of::<BPC>();
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

    let fill8 = vdupq_n_u16(fill as u16);
    let fill4 = vdup_n_u16(fill as u16);
    let mut x = 0usize;
    if h == 4 {
        while x < w {
            let c0 = z3_col4(
                &rev_left,
                ((x + 1) as i32) * dy,
                base_inc,
                max_base_y,
                fill4,
            );
            let c1 = z3_col4(
                &rev_left,
                ((x + 2) as i32) * dy,
                base_inc,
                max_base_y,
                fill4,
            );
            let c2 = z3_col4(
                &rev_left,
                ((x + 3) as i32) * dy,
                base_inc,
                max_base_y,
                fill4,
            );
            let c3 = z3_col4(
                &rev_left,
                ((x + 4) as i32) * dy,
                base_inc,
                max_base_y,
                fill4,
            );
            transpose_store_4x4_u16::<BPC>(dst, x * bpp, stride, c0, c1, c2, c3);
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
                transpose_store_4x4_u16::<BPC>(
                    dst,
                    off0,
                    stride,
                    vget_low_u16(c0),
                    vget_low_u16(c1),
                    vget_low_u16(c2),
                    vget_low_u16(c3),
                );
                transpose_store_4x4_u16::<BPC>(
                    dst,
                    off0 + 4 * stride,
                    stride,
                    vget_high_u16(c0),
                    vget_high_u16(c1),
                    vget_high_u16(c2),
                    vget_high_u16(c3),
                );
                y += 8;
            }
            x += 4;
        }
    }
}

#[target_feature(enable = "neon")]
fn filter_pred_neon<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    w: usize,
    h: usize,
    angle_in: usize,
) {
    let bpp = bpp_of::<BPC>();
    let fi = angle_in & 511;
    let t = &FILTER_TAPS_NEON[fi];
    let t0 = load_i16x8(&t[0]);
    let t1 = load_i16x8(&t[1]);
    let t2 = load_i16x8(&t[2]);
    let t3 = load_i16x8(&t[3]);
    let t4 = load_i16x8(&t[4]);
    let t5 = load_i16x8(&t[5]);
    let t6 = load_i16x8(&t[6]);

    let r8 = vdupq_n_s32(8);
    let zero = vdupq_n_s32(0);
    let max_v = vdupq_n_s32(bd_max_of::<BPC>());

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
                    read_px::<BPC>(dst, top_row + (col - 1) * bpp)
                };
                (
                    p0,
                    read_px::<BPC>(dst, top_row + col * bpp),
                    read_px::<BPC>(dst, top_row + (col + 1) * bpp),
                    read_px::<BPC>(dst, top_row + (col + 2) * bpp),
                    read_px::<BPC>(dst, top_row + (col + 3) * bpp),
                )
            };
            let (p5, p6) = if group == 0 {
                (
                    read_edge::<BPC>(edge, edge_tl - 1),
                    read_edge::<BPC>(edge, edge_tl - 2),
                )
            } else {
                (
                    read_px::<BPC>(dst, y * stride + (x - 1) * bpp),
                    read_px::<BPC>(dst, (y + 1) * stride + (x - 1) * bpp),
                )
            };

            let v0 = vdupq_n_s16(p0 as i16);
            let v1 = vdupq_n_s16(p1 as i16);
            let v2 = vdupq_n_s16(p2 as i16);
            let v3 = vdupq_n_s16(p3 as i16);
            let v4 = vdupq_n_s16(p4 as i16);
            let v5 = vdupq_n_s16(p5 as i16);
            let v6 = vdupq_n_s16(p6 as i16);

            let mut acc_lo = vmlal_s16(r8, vget_low_s16(t0), vget_low_s16(v0));
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(t1), vget_low_s16(v1));
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(t2), vget_low_s16(v2));
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(t3), vget_low_s16(v3));
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(t4), vget_low_s16(v4));
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(t5), vget_low_s16(v5));
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(t6), vget_low_s16(v6));

            let mut acc_hi = vmlal_high_s16(r8, t0, v0);
            acc_hi = vmlal_high_s16(acc_hi, t1, v1);
            acc_hi = vmlal_high_s16(acc_hi, t2, v2);
            acc_hi = vmlal_high_s16(acc_hi, t3, v3);
            acc_hi = vmlal_high_s16(acc_hi, t4, v4);
            acc_hi = vmlal_high_s16(acc_hi, t5, v5);
            acc_hi = vmlal_high_s16(acc_hi, t6, v6);

            let o_lo = vminq_s32(vmaxq_s32(vshrq_n_s32::<4>(acc_lo), zero), max_v);
            let o_hi = vminq_s32(vmaxq_s32(vshrq_n_s32::<4>(acc_hi), zero), max_v);
            store_px4_u16::<BPC>(
                dst,
                y * stride + x * bpp,
                vmovn_u32(vreinterpretq_u32_s32(o_lo)),
            );
            store_px4_u16::<BPC>(
                dst,
                (y + 1) * stride + x * bpp,
                vmovn_u32(vreinterpretq_u32_s32(o_hi)),
            );
            x += 4;
        }
        y += 2;
    }
}

#[target_feature(enable = "neon")]
pub(super) fn angular_neon<const BPC: u32, const MODE: usize>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    w: usize,
    h: usize,
    angle: usize,
    max_width: usize,
    max_height: usize,
) {
    match MODE {
        0 => splat_dc_neon::<BPC>(dst, stride, w, h, dc_gen_neon::<BPC>(edge, w, h) as u16),
        1 => vert_neon::<BPC>(dst, stride, edge, w, h),
        2 => hor_neon::<BPC>(dst, stride, edge, w, h),
        3 => splat_dc_neon::<BPC>(dst, stride, w, h, dc_gen_left_neon::<BPC>(edge, h) as u16),
        4 => splat_dc_neon::<BPC>(dst, stride, w, h, dc_gen_top_neon::<BPC>(edge, w) as u16),
        5 => splat_dc_neon::<BPC>(dst, stride, w, h, 1u16 << (BPC - 1)),
        6 => z1_neon::<BPC>(dst, stride, edge, w, h, angle),
        7 => z2_neon::<BPC>(dst, stride, edge, w, h, angle, max_width, max_height),
        8 => z3_neon::<BPC>(dst, stride, edge, w, h, angle),
        9 => smooth_neon::<BPC>(dst, stride, edge, w, h),
        10 => smooth_v_neon::<BPC>(dst, stride, edge, w, h),
        11 => smooth_h_neon::<BPC>(dst, stride, edge, w, h),
        12 => paeth_neon::<BPC>(dst, stride, edge, w, h),
        _ => filter_pred_neon::<BPC>(dst, stride, edge, w, h, angle),
    }
}

// ============================================================================
// CfL AC / CfL PRED / PAL_PRED
// ============================================================================

#[target_feature(enable = "neon")]
#[inline]
fn cfl_ac_row8<const BPC: u32, const LAYOUT: usize>(
    y: &[u8],
    y0_off: usize,
    y1_off: usize,
    x: usize,
) -> int16x8_t {
    let bpp = bpp_of::<BPC>();
    if LAYOUT == 0 {
        let r0 = if BPC == 8 {
            vpaddlq_u8(load_u8x16(chunk::<u8, 16>(y, y0_off + 2 * x)))
        } else {
            let a = load_px8_u16::<BPC>(y, y0_off + 4 * x);
            let b = load_px8_u16::<BPC>(y, y0_off + 4 * x + 16);
            vcombine_u16(vmovn_u32(vpaddlq_u16(a)), vmovn_u32(vpaddlq_u16(b)))
        };
        let r1 = if BPC == 8 {
            vpaddlq_u8(load_u8x16(chunk::<u8, 16>(y, y1_off + 2 * x)))
        } else {
            let a = load_px8_u16::<BPC>(y, y1_off + 4 * x);
            let b = load_px8_u16::<BPC>(y, y1_off + 4 * x + 16);
            vcombine_u16(vmovn_u32(vpaddlq_u16(a)), vmovn_u32(vpaddlq_u16(b)))
        };
        vshlq_n_s16::<1>(vreinterpretq_s16_u16(vaddq_u16(r0, r1)))
    } else if LAYOUT == 1 {
        let r0 = if BPC == 8 {
            vpaddlq_u8(load_u8x16(chunk::<u8, 16>(y, y0_off + 2 * x)))
        } else {
            let a = load_px8_u16::<BPC>(y, y0_off + 4 * x);
            let b = load_px8_u16::<BPC>(y, y0_off + 4 * x + 16);
            vcombine_u16(vmovn_u32(vpaddlq_u16(a)), vmovn_u32(vpaddlq_u16(b)))
        };
        vshlq_n_s16::<2>(vreinterpretq_s16_u16(r0))
    } else {
        let r0 = load_px8_u16::<BPC>(y, y0_off + x * bpp);
        vshlq_n_s16::<3>(vreinterpretq_s16_u16(r0))
    }
}

#[target_feature(enable = "neon")]
pub(super) fn cfl_ac_neon<const BPC: u32, const LAYOUT: usize>(
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
    let bpp = bpp_of::<BPC>();
    let visible_w = cw - 4 * w_pad;
    let visible_h = ch - 4 * h_pad;
    let n = cw * ch;
    let log2sz = cw.trailing_zeros() + ch.trailing_zeros();

    // Hızlı yol: dolgu yok ve genişlik 8'in katıysa tek geçişte hem üret hem toplamı biriktir.
    if w_pad == 0 && h_pad == 0 && cw >= 8 {
        let mut acc_s32: int32x4_t = vdupq_n_s32(0);
        for row in 0..ch {
            let luma_row = row << ss_ver;
            let y0_off = luma_row * y_stride;
            let y1_off = (luma_row + 1) * y_stride;
            let mut x = 0usize;
            while x < cw {
                let sum_s16 = cfl_ac_row8::<BPC, LAYOUT>(y, y0_off, y1_off, x);
                acc_s32 = vpadalq_s16(acc_s32, sum_s16);
                store_i16x8(chunk_mut::<i16, 8>(ac, row * cw + x), sum_s16);
                x += 8;
            }
        }
        let sum = ((1i32 << log2sz) >> 1) + vaddvq_s32(acc_s32);
        let avg = vdupq_n_s16((sum >> log2sz) as i16);
        let mut i = 0usize;
        while i < n {
            let v = vsubq_s16(load_i16x8(chunk::<i16, 8>(ac, i)), avg);
            store_i16x8(chunk_mut::<i16, 8>(ac, i), v);
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
            store_i16x8(chunk_mut::<i16, 8>(ac, row * cw + x), sum_s16);
            x += 8;
        }

        while x < visible_w {
            let col = (x << ss_hor) * bpp;
            let mut sum = read_px::<BPC>(y, y0_off + col);
            if ss_hor != 0 {
                sum += read_px::<BPC>(y, y0_off + col + bpp);
            }
            if ss_ver != 0 {
                sum += read_px::<BPC>(y, y1_off + col);
                if ss_hor != 0 {
                    sum += read_px::<BPC>(y, y1_off + col + bpp);
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

    let mut acc_s32: int32x4_t = vdupq_n_s32(0);
    let mut i = 0usize;
    while i < n {
        acc_s32 = vpadalq_s16(acc_s32, load_i16x8(chunk::<i16, 8>(ac, i)));
        i += 8;
    }
    let sum = ((1i32 << log2sz) >> 1) + vaddvq_s32(acc_s32);
    let avg = vdupq_n_s16((sum >> log2sz) as i16);
    let mut i = 0usize;
    while i < n {
        let v = vsubq_s16(load_i16x8(chunk::<i16, 8>(ac, i)), avg);
        store_i16x8(chunk_mut::<i16, 8>(ac, i), v);
        i += 8;
    }
}

#[target_feature(enable = "neon")]
pub(super) fn cfl_pred_neon<const BPC: u32, const MODE: usize>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    w: usize,
    h: usize,
    ac: &[i16],
    alpha: i32,
) {
    let dc = match MODE {
        0 => dc_gen_neon::<BPC>(edge, w, h) as i32,
        1 => dc_gen_left_neon::<BPC>(edge, h) as i32,
        2 => dc_gen_top_neon::<BPC>(edge, w) as i32,
        _ => 1i32 << (BPC - 1),
    };
    if alpha == 0 {
        splat_dc_neon::<BPC>(dst, stride, w, h, dc as u16);
        return;
    }

    let bpp = bpp_of::<BPC>();
    let dc_v = vdupq_n_s32(dc);
    let alpha_v = vdupq_n_s16(alpha as i16);
    let zero = vdupq_n_s32(0);
    let max_v = vdupq_n_s32(bd_max_of::<BPC>());

    if w == 4 {
        let mut y = 0usize;
        while y + 2 <= h {
            let ac_v = load_i16x8(chunk::<i16, 8>(ac, y * 4));
            let d_lo = vmull_s16(vget_low_s16(ac_v), vget_low_s16(alpha_v));
            let d_hi = vmull_high_s16(ac_v, alpha_v);
            let q_lo = vrshrq_n_s32::<6>(vabsq_s32(d_lo));
            let q_hi = vrshrq_n_s32::<6>(vabsq_s32(d_hi));
            let s_lo = vbslq_s32(vcltq_s32(d_lo, zero), vnegq_s32(q_lo), q_lo);
            let s_hi = vbslq_s32(vcltq_s32(d_hi, zero), vnegq_s32(q_hi), q_hi);
            let v_lo = vminq_s32(vmaxq_s32(vaddq_s32(dc_v, s_lo), zero), max_v);
            let v_hi = vminq_s32(vmaxq_s32(vaddq_s32(dc_v, s_hi), zero), max_v);
            store_px4_u16::<BPC>(dst, y * stride, vmovn_u32(vreinterpretq_u32_s32(v_lo)));
            store_px4_u16::<BPC>(
                dst,
                (y + 1) * stride,
                vmovn_u32(vreinterpretq_u32_s32(v_hi)),
            );
            y += 2;
        }
        return;
    }

    for y in 0..h {
        let row_off = y * stride;
        let mut x = 0usize;
        while x < w {
            let ac_v = load_i16x8(chunk::<i16, 8>(ac, y * w + x));
            let d_lo = vmull_s16(vget_low_s16(ac_v), vget_low_s16(alpha_v));
            let d_hi = vmull_high_s16(ac_v, alpha_v);

            let q_lo = vrshrq_n_s32::<6>(vabsq_s32(d_lo));
            let q_hi = vrshrq_n_s32::<6>(vabsq_s32(d_hi));

            let s_lo = vbslq_s32(vcltq_s32(d_lo, zero), vnegq_s32(q_lo), q_lo);
            let s_hi = vbslq_s32(vcltq_s32(d_hi, zero), vnegq_s32(q_hi), q_hi);

            let v_lo = vminq_s32(vmaxq_s32(vaddq_s32(dc_v, s_lo), zero), max_v);
            let v_hi = vminq_s32(vmaxq_s32(vaddq_s32(dc_v, s_hi), zero), max_v);
            let out = vmovn_high_u32(
                vmovn_u32(vreinterpretq_u32_s32(v_lo)),
                vreinterpretq_u32_s32(v_hi),
            );
            store_px8_u16::<BPC>(dst, row_off + x * bpp, out);
            x += 8;
        }
    }
}

/// `idx[0..n]` içinde geçersiz nibble (`b & 0x88 != 0`) olup olmadığını NEON ile denetler.
#[target_feature(enable = "neon")]
pub(super) fn pal_idx_gecerli_neon(idx: &[u8], n: usize) -> bool {
    let m88 = vdupq_n_u8(0x88);
    let mut i = 0usize;
    let mut bad = 0u8;
    while i + 16 <= n {
        let v = load_u8x16(chunk::<u8, 16>(idx, i));
        bad |= vmaxvq_u8(vandq_u8(v, m88));
        i += 16;
    }
    while i < n {
        bad |= idx[i] & 0x88;
        i += 1;
    }
    bad == 0
}

#[target_feature(enable = "neon")]
pub(super) fn pal_pred_neon<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
) {
    let m07 = vdupq_n_u8(0x07);
    let half_w = w / 2;

    if BPC == 8 {
        let pal_v: uint8x16_t = vcombine_u8(load_u8x8(chunk::<u8, 8>(pal, 0)), vdup_n_u8(0));
        match w {
            4 => {
                let mut y = 0usize;
                while y + 4 <= h {
                    let b8 = load_u8x8(chunk::<u8, 8>(idx, y * 2));
                    let raw = vcombine_u8(b8, vdup_n_u8(0));
                    let lo = vandq_u8(raw, m07);
                    let hi = vandq_u8(vshrq_n_u8::<4>(raw), m07);
                    let nibs = vzip1q_u8(lo, hi);
                    let out = vqtbl1q_u8(pal_v, nibs);
                    let lo64 = vget_lane_u64::<0>(vreinterpret_u64_u8(vget_low_u8(out)));
                    let hi64 = vget_lane_u64::<0>(vreinterpret_u64_u8(vget_high_u8(out)));
                    let off = y * stride;
                    *chunk_mut::<u8, 4>(dst, off) = (lo64 as u32).to_le_bytes();
                    *chunk_mut::<u8, 4>(dst, off + stride) = ((lo64 >> 32) as u32).to_le_bytes();
                    *chunk_mut::<u8, 4>(dst, off + 2 * stride) = (hi64 as u32).to_le_bytes();
                    *chunk_mut::<u8, 4>(dst, off + 3 * stride) =
                        ((hi64 >> 32) as u32).to_le_bytes();
                    y += 4;
                }
            }
            8 => {
                let mut y = 0usize;
                while y + 2 <= h {
                    let b8 = load_u8x8(chunk::<u8, 8>(idx, y * 4));
                    let raw = vcombine_u8(b8, vdup_n_u8(0));
                    let lo = vandq_u8(raw, m07);
                    let hi = vandq_u8(vshrq_n_u8::<4>(raw), m07);
                    let nibs = vzip1q_u8(lo, hi);
                    let out = vqtbl1q_u8(pal_v, nibs);
                    store_u8x8(chunk_mut::<u8, 8>(dst, y * stride), vget_low_u8(out));
                    store_u8x8(chunk_mut::<u8, 8>(dst, (y + 1) * stride), vget_high_u8(out));
                    y += 2;
                }
            }
            16 => {
                for y in 0..h {
                    let b8 = load_u8x8(chunk::<u8, 8>(idx, y * 8));
                    let raw = vcombine_u8(b8, vdup_n_u8(0));
                    let lo = vandq_u8(raw, m07);
                    let hi = vandq_u8(vshrq_n_u8::<4>(raw), m07);
                    let nibs = vzip1q_u8(lo, hi);
                    let out = vqtbl1q_u8(pal_v, nibs);
                    store_u8x16(chunk_mut::<u8, 16>(dst, y * stride), out);
                }
            }
            _ => {
                for y in 0..h {
                    let row_off = y * stride;
                    let idx_off = y * half_w;
                    let mut x = 0usize;
                    while x < w {
                        let b8 = load_u8x8(chunk::<u8, 8>(idx, idx_off + x / 2));
                        let raw = vcombine_u8(b8, vdup_n_u8(0));
                        let lo = vandq_u8(raw, m07);
                        let hi = vandq_u8(vshrq_n_u8::<4>(raw), m07);
                        let nibs = vzip1q_u8(lo, hi);
                        let out = vqtbl1q_u8(pal_v, nibs);
                        store_u8x16(chunk_mut::<u8, 16>(dst, row_off + x), out);
                        x += 16;
                    }
                }
            }
        }
    } else {
        let pal_v: uint8x16_t = load_u8x16(chunk::<u8, 16>(pal, 0));
        let ones = vdupq_n_u8(1);
        if w == 4 {
            let mut y = 0usize;
            while y + 2 <= h {
                let u = u32::from_le_bytes(*chunk::<u8, 4>(idx, y * 2));
                let raw4 = vreinterpretq_u8_u32(vdupq_n_u32(u));
                let lo = vandq_u8(raw4, m07);
                let hi = vandq_u8(vshrq_n_u8::<4>(raw4), m07);
                let nibs8 = vzip1q_u8(lo, hi);
                let even = vshlq_n_u8::<1>(nibs8);
                let odd = vaddq_u8(even, ones);
                let b_idx = vzip1q_u8(even, odd);
                let out = vqtbl1q_u8(pal_v, b_idx);
                store_u8x8(chunk_mut::<u8, 8>(dst, y * stride), vget_low_u8(out));
                store_u8x8(chunk_mut::<u8, 8>(dst, (y + 1) * stride), vget_high_u8(out));
                y += 2;
            }
        } else {
            for y in 0..h {
                let row_off = y * stride;
                let idx_off = y * half_w;
                let mut x = 0usize;
                while x < w {
                    let u = u32::from_le_bytes(*chunk::<u8, 4>(idx, idx_off + x / 2));
                    let raw4 = vreinterpretq_u8_u32(vdupq_n_u32(u));
                    let lo = vandq_u8(raw4, m07);
                    let hi = vandq_u8(vshrq_n_u8::<4>(raw4), m07);
                    let nibs8 = vzip1q_u8(lo, hi);
                    let even = vshlq_n_u8::<1>(nibs8);
                    let odd = vaddq_u8(even, ones);
                    let b_idx = vzip1q_u8(even, odd);
                    let out = vqtbl1q_u8(pal_v, b_idx);
                    store_u8x16(chunk_mut::<u8, 16>(dst, row_off + 2 * x), out);
                    x += 8;
                }
            }
        }
    }
}
