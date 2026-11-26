//! `aarch64` NEON SuperRes yatay büyütme çekirdeği (`resize/resize`, 8 / 10 / 12 bpc).
//!
//! `%100 #![forbid(unsafe_code)]`; tek giriş kapısı `super::giris::mc_resize` üzerinden
//! [`NeonToken`] ile verilir. Aritmetik [`crate::dsp::scalar::resize`] ile bit-aynıdır:
//! her satırda `mx = mx0`, `src_x = -1`; 8 dokunuş `RESIZE_FILTER[mx >> 8]`, kaynak
//! `[0, src_w - 1]` aralığına kırpılıp `(-sum + 64) >> 7` sonucu `[0, (1 << BPC) - 1]`
//! aralığına kırpılır.

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use core::arch::aarch64::{
    int16x8_t, int32x4_t, vcombine_u16, vdupq_n_s32, vget_lane_u32, vget_lane_u64, vget_low_s16,
    vmaxq_s32, vminq_s32, vmlal_high_s16, vmovl_u8, vmovn_u16, vmovn_u32, vmull_s16, vpaddq_s32,
    vreinterpret_u32_u8, vreinterpret_u64_u16, vreinterpretq_s16_u8, vreinterpretq_s16_u16,
    vreinterpretq_u32_s32, vshrq_n_s32, vsubq_s32,
};

use super::mem::{chunk, chunk_mut, load_i16x8, load_u8x8, load_u8x16};
use crate::cpu::NeonToken;
use crate::dsp::resize::ResizeFn;
use crate::dsp::resize_tables::RESIZE_FILTER;

/// `RESIZE_FILTER` katsayıları `i16` × 8 düzeninde (derleme zamanı tablosu).
static RESIZE_FILT_I16: [[i16; 8]; 64] = filtre_i16();

const fn filtre_i16() -> [[i16; 8]; 64] {
    let mut out = [[0i16; 8]; 64];
    let mut i = 0;
    while i < 64 {
        let mut k = 0;
        while k < 8 {
            out[i][k] = RESIZE_FILTER[i][k] as i16;
            k += 1;
        }
        i += 1;
    }
    out
}

/// `resize/resize` (`8 / 10 / 12` bpc) için [`NeonToken`]-kapılı yuva kurucusu.
#[must_use]
pub const fn resize_fn<const BPC: u32>(_neon: NeonToken) -> ResizeFn {
    super::giris::mc_resize::<BPC>
}

/// 8 kaynak pikselini (`bas .. bas + 8`, `[0, son]` kırpmalı) `int16x8_t` olarak yükler.
#[target_feature(enable = "neon")]
#[inline]
fn yukle_8px<const BPC: u32>(s: &[u8], bas: i64, son: i64) -> int16x8_t {
    if bas >= 0 && bas + 7 <= son {
        let o = bas as usize;
        if BPC == 8 {
            vreinterpretq_s16_u16(vmovl_u8(load_u8x8(chunk::<u8, 8>(s, o))))
        } else {
            vreinterpretq_s16_u8(load_u8x16(chunk::<u8, 16>(s, 2 * o)))
        }
    } else {
        let mut p = [0i16; 8];
        let mut k = 0usize;
        while k < 8 {
            let sx = (bas + k as i64).clamp(0, son) as usize;
            p[k] = if BPC == 8 {
                i16::from(s[sx])
            } else {
                u16::from_le_bytes([s[2 * sx], s[2 * sx + 1]]) as i16
            };
            k += 1;
        }
        load_i16x8(&p)
    }
}

/// Tek çıktı sütununun 8 dokunuşlu kısmi çarpım toplamı (`4 × i32`).
#[target_feature(enable = "neon")]
#[inline]
fn dokunus8<const BPC: u32>(s: &[u8], bas: i64, son: i64, fil: usize) -> int32x4_t {
    let p = yukle_8px::<BPC>(s, bas, son);
    let f = load_i16x8(&RESIZE_FILT_I16[fil]);
    vmlal_high_s16(vmull_s16(vget_low_s16(p), vget_low_s16(f)), p, f)
}

/// 8 / 10 / 12 bpc NEON yatay SuperRes büyütme gövdesi.
///
/// Sözleşme (`buyutme_sozlesmesi` / `buyutme16_sozlesmesi`) çağıran köprüde denetlenmiştir.
#[target_feature(enable = "neon")]
pub(super) fn resize_neon<const BPC: u32>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    dst_w: usize,
    h: usize,
    src_w: usize,
    dx: i32,
    mx0: i32,
) {
    let max_px = (1i32 << BPC) - 1;
    let son = src_w as i64 - 1;
    let v64 = vdupq_n_s32(64);
    let v0 = vdupq_n_s32(0);
    let vmax = vdupq_n_s32(max_px);

    let mut y = 0usize;
    while y < h {
        let s = &src[y * ss..];
        let d = &mut dst[y * ds..];
        let mut mx = mx0;
        let mut src_x: i64 = -1;
        let mut x = 0usize;
        while x < dst_w {
            let n = (dst_w - x).min(4);
            let mut bas = [0i64; 4];
            let mut fil = [0usize; 4];
            let mut k = 0usize;
            while k < n {
                bas[k] = src_x - 3;
                fil[k] = (mx >> 8) as usize;
                mx += dx;
                src_x += i64::from(mx >> 14);
                mx &= 0x3fff;
                k += 1;
            }
            let a0 = dokunus8::<BPC>(s, bas[0], son, fil[0]);
            let a1 = if n > 1 {
                dokunus8::<BPC>(s, bas[1], son, fil[1])
            } else {
                v0
            };
            let a2 = if n > 2 {
                dokunus8::<BPC>(s, bas[2], son, fil[2])
            } else {
                v0
            };
            let a3 = if n > 3 {
                dokunus8::<BPC>(s, bas[3], son, fil[3])
            } else {
                v0
            };
            let sums = vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3));
            let res = vminq_s32(vmaxq_s32(vshrq_n_s32::<7>(vsubq_s32(v64, sums)), v0), vmax);
            let u16_4 = vmovn_u32(vreinterpretq_u32_s32(res));
            if BPC == 8 {
                let b4 =
                    vget_lane_u32::<0>(vreinterpret_u32_u8(vmovn_u16(vcombine_u16(u16_4, u16_4))))
                        .to_le_bytes();
                if n == 4 {
                    *chunk_mut::<u8, 4>(d, x) = b4;
                } else {
                    d[x..x + n].copy_from_slice(&b4[..n]);
                }
            } else {
                let b8 = vget_lane_u64::<0>(vreinterpret_u64_u16(u16_4)).to_le_bytes();
                if n == 4 {
                    *chunk_mut::<u8, 8>(d, 2 * x) = b8;
                } else {
                    d[2 * x..2 * (x + n)].copy_from_slice(&b8[..2 * n]);
                }
            }
            x += n;
        }
        y += 1;
    }
}
