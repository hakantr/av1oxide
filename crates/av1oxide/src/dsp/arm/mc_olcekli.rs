//! `aarch64` NEON ölçekli hareket dengelemesi (`mc_scaled[0..=9]`, `mct_scaled[0..=9]`;
//! 8 / 10 / 12 bpc — `60` yuva).
//!
//! `%100 #![forbid(unsafe_code)]`; giriş köprüleri `super::giris::mc_put_scaled` ve
//! `super::giris::mc_prep_scaled` üzerindendir ([`NeonToken`] kapılı).
//!
//! ## Aritmetik paritesi (`scalar/mc.rs`, `x86/mc_olcekli.rs`, `x86/mc16_olcekli.rs`)
//!
//! - `ib = if BPC == 12 { 2 } else { 4 }`, `prep_bias = if BPC == 8 { 0 } else { 8192 }`.
//! - **8-tap (`FI < 9`):**
//!   - Yatay: `s[ofs[x]..ofs[x]+8] · f_h`, `(sum + (1 << (6 - ib) >> 1)) >> (6 - ib)` → `i16`.
//!   - Dikey (8 satırlık halka): `sum(halka[r] * f_v[r])`;
//!     - `put`: `(sum + (1 << (6 + ib) >> 1)) >> (6 + ib)`, `[0, (1 << BPC) - 1]` kırpma.
//!     - `prep`: `((sum + 32) >> 6) - prep_bias`, `i16` doyurma.
//! - **Bilinear (`FI == 9`):**
//!   - Yatay: `s[ofs[x] + 3] * (16 - m) + s[ofs[x] + 4] * m`; `BPC == 12` için `(sum + 2) >> 2`,
//!     `8 / 10` bpc için kaydırmasız `i16`.
//!   - Dikey (2 satırlık halka): `r0 * (16 - m) + r1 * m`;
//!     - `put`: `(sum + (1 << (4 + ib) >> 1)) >> (4 + ib)`, `[0, (1 << BPC) - 1]` kırpma.
//!     - `prep`: `((sum + 8) >> 4) - prep_bias`, `i16` doyurma.

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use core::arch::aarch64::{
    int16x8_t, int32x4_t, vaddq_s32, vdupq_n_s32, vget_high_s16, vget_lane_u32, vget_lane_u64,
    vget_low_s16, vmaxq_s32, vminq_s32, vmlal_high_s16, vmlal_n_s16, vmovl_u8, vmovn_high_u32,
    vmovn_u16, vmovn_u32, vmull_high_s16, vmull_s16, vpaddq_s32, vqmovn_high_s32, vqmovn_s32,
    vreinterpret_u32_s16, vreinterpret_u32_u8, vreinterpret_u64_s16, vreinterpret_u64_u16,
    vreinterpretq_s16_u8, vreinterpretq_s16_u16, vreinterpretq_u32_s32, vshrq_n_s32, vsubq_s32,
};

use super::giris::store_i16x8;
use super::mem::{
    chunk, chunk_mut, load_i16x8, load_u8x8, load_u8x16, store_u8x8, store_u16x4, store_u16x8,
};
use crate::cpu::NeonToken;
use crate::dsp::mc::{Filter2d, McPrepScaledFn, McPutScaledFn};
use crate::dsp::mc_tables::MC_SUBPEL_FILTERS;

/// Bilinear yuva dizini (`FILTER_2D_BILINEAR`).
pub(super) const BILIN: usize = 9;

/// Alt piksel filtreleri `i16` × 8 düzeninde: `[tablo][p]`, `p = 0` → birim `[0,0,0,64,0,0,0,0]`.
static FILTRE_I16: [[[i16; 8]; 16]; 6] = filtre_tablosu();

const fn filtre_tablosu() -> [[[i16; 8]; 16]; 6] {
    let mut t = [[[0i16; 8]; 16]; 6];
    let mut k = 0;
    while k < 6 {
        t[k][0][3] = 64;
        let mut p = 1;
        while p < 16 {
            let mut i = 0;
            while i < 8 {
                t[k][p][i] = MC_SUBPEL_FILTERS[k][p - 1][i] as i16;
                i += 1;
            }
            p += 1;
        }
        k += 1;
    }
    t
}

/// `mc_scaled[fi]` (`8 / 10 / 12` bpc) için [`NeonToken`]-kapılı kurucu.
#[must_use]
pub const fn put_scaled_fn<const BPC: u32>(_neon: NeonToken, fi: usize) -> McPutScaledFn {
    match fi {
        0 => super::giris::mc_put_scaled::<BPC, 0>,
        1 => super::giris::mc_put_scaled::<BPC, 1>,
        2 => super::giris::mc_put_scaled::<BPC, 2>,
        3 => super::giris::mc_put_scaled::<BPC, 3>,
        4 => super::giris::mc_put_scaled::<BPC, 4>,
        5 => super::giris::mc_put_scaled::<BPC, 5>,
        6 => super::giris::mc_put_scaled::<BPC, 6>,
        7 => super::giris::mc_put_scaled::<BPC, 7>,
        8 => super::giris::mc_put_scaled::<BPC, 8>,
        _ => super::giris::mc_put_scaled::<BPC, 9>,
    }
}

/// `mct_scaled[fi]` (`8 / 10 / 12` bpc) için [`NeonToken`]-kapılı kurucu.
#[must_use]
pub const fn prep_scaled_fn<const BPC: u32>(_neon: NeonToken, fi: usize) -> McPrepScaledFn {
    match fi {
        0 => super::giris::mc_prep_scaled::<BPC, 0>,
        1 => super::giris::mc_prep_scaled::<BPC, 1>,
        2 => super::giris::mc_prep_scaled::<BPC, 2>,
        3 => super::giris::mc_prep_scaled::<BPC, 3>,
        4 => super::giris::mc_prep_scaled::<BPC, 4>,
        5 => super::giris::mc_prep_scaled::<BPC, 5>,
        6 => super::giris::mc_prep_scaled::<BPC, 6>,
        7 => super::giris::mc_prep_scaled::<BPC, 7>,
        8 => super::giris::mc_prep_scaled::<BPC, 8>,
        _ => super::giris::mc_prep_scaled::<BPC, 9>,
    }
}

#[inline]
const fn tablolar(f: Filter2d, w: usize, h: usize) -> (usize, usize) {
    let (kh, kv) = (f.horizontal_table(), f.vertical_table());
    (
        if w > 4 { kh } else { 3 + (kh & 1) },
        if h > 4 { kv } else { 3 + (kv & 1) },
    )
}

#[target_feature(enable = "neon")]
#[inline]
fn yukle_8_kaynak<const BPC: u32>(src: &[u8], taban: usize, ofs_px: usize) -> int16x8_t {
    if BPC == 8 {
        vreinterpretq_s16_u16(vmovl_u8(load_u8x8(chunk::<u8, 8>(src, taban + ofs_px))))
    } else {
        vreinterpretq_s16_u8(load_u8x16(chunk::<u8, 16>(src, taban + 2 * ofs_px)))
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn yatay8_satir<const BPC: u32>(
    ara: &mut [i16; 136],
    src: &[u8],
    taban: usize,
    ofs: &[usize; 136],
    fil: &[u8; 136],
    th: usize,
    gw: usize,
) {
    let yuv = vdupq_n_s32(if BPC == 12 { 8 } else { 2 });
    let mut g = 0usize;
    while g < gw {
        let mut s_v = [yuv; 2];
        let mut half = 0usize;
        while half < 2 {
            let b = g + half * 4;
            let p0 = yukle_8_kaynak::<BPC>(src, taban, ofs[b]);
            let f0 = load_i16x8(&FILTRE_I16[th][usize::from(fil[b])]);
            let a0 = vmlal_high_s16(vmull_s16(vget_low_s16(p0), vget_low_s16(f0)), p0, f0);

            let p1 = yukle_8_kaynak::<BPC>(src, taban, ofs[b + 1]);
            let f1 = load_i16x8(&FILTRE_I16[th][usize::from(fil[b + 1])]);
            let a1 = vmlal_high_s16(vmull_s16(vget_low_s16(p1), vget_low_s16(f1)), p1, f1);

            let p2 = yukle_8_kaynak::<BPC>(src, taban, ofs[b + 2]);
            let f2 = load_i16x8(&FILTRE_I16[th][usize::from(fil[b + 2])]);
            let a2 = vmlal_high_s16(vmull_s16(vget_low_s16(p2), vget_low_s16(f2)), p2, f2);

            let p3 = yukle_8_kaynak::<BPC>(src, taban, ofs[b + 3]);
            let f3 = load_i16x8(&FILTRE_I16[th][usize::from(fil[b + 3])]);
            let a3 = vmlal_high_s16(vmull_s16(vget_low_s16(p3), vget_low_s16(f3)), p3, f3);

            let sum = vaddq_s32(vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3)), yuv);
            s_v[half] = if BPC == 12 {
                vshrq_n_s32::<4>(sum)
            } else {
                vshrq_n_s32::<2>(sum)
            };
            half += 1;
        }
        let packed = vqmovn_high_s32(vqmovn_s32(s_v[0]), s_v[1]);
        store_i16x8(chunk_mut::<i16, 8>(ara, g), packed);
        g += 8;
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn sakla_put<const BPC: u32>(
    dst: &mut [u8],
    ds: usize,
    y: usize,
    x: usize,
    w: usize,
    lo: int32x4_t,
    hi: int32x4_t,
) {
    let v0 = vdupq_n_s32(0);
    let vmax = vdupq_n_s32((1i32 << BPC) - 1);
    let c_lo = vminq_s32(vmaxq_s32(lo, v0), vmax);
    let c_hi = vminq_s32(vmaxq_s32(hi, v0), vmax);
    let u16_8 = vmovn_high_u32(
        vmovn_u32(vreinterpretq_u32_s32(c_lo)),
        vreinterpretq_u32_s32(c_hi),
    );
    let n = (w - x).min(8);
    if BPC == 8 {
        let u8_8 = vmovn_u16(u16_8);
        let o = y * ds + x;
        match n {
            8 => store_u8x8(chunk_mut::<u8, 8>(dst, o), u8_8),
            4 => {
                *chunk_mut::<u8, 4>(dst, o) =
                    vget_lane_u32::<0>(vreinterpret_u32_u8(u8_8)).to_le_bytes();
            }
            _ => {
                let b = vget_lane_u32::<0>(vreinterpret_u32_u8(u8_8)).to_le_bytes();
                *chunk_mut::<u8, 2>(dst, o) = [b[0], b[1]];
            }
        }
    } else {
        let o = y * ds + 2 * x;
        match n {
            8 => {
                let mut tmp = [0u16; 8];
                store_u16x8(&mut tmp, u16_8);
                let mut k = 0;
                while k < 8 {
                    let b = tmp[k].to_le_bytes();
                    dst[o + 2 * k] = b[0];
                    dst[o + 2 * k + 1] = b[1];
                    k += 1;
                }
            }
            4 => {
                let mut tmp = [0u16; 4];
                store_u16x4(&mut tmp, vmovn_u32(vreinterpretq_u32_s32(c_lo)));
                let mut k = 0;
                while k < 4 {
                    let b = tmp[k].to_le_bytes();
                    dst[o + 2 * k] = b[0];
                    dst[o + 2 * k + 1] = b[1];
                    k += 1;
                }
            }
            _ => {
                let b = vget_lane_u64::<0>(vreinterpret_u64_u16(vmovn_u32(vreinterpretq_u32_s32(
                    c_lo,
                ))))
                .to_le_bytes();
                *chunk_mut::<u8, 4>(dst, o) = [b[0], b[1], b[2], b[3]];
            }
        }
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn sakla_prep<const BPC: u32>(
    tmp: &mut [i16],
    y: usize,
    x: usize,
    w: usize,
    lo: int32x4_t,
    hi: int32x4_t,
) {
    let (lo, hi) = if BPC == 8 {
        (lo, hi)
    } else {
        let bias = vdupq_n_s32(8192);
        (vsubq_s32(lo, bias), vsubq_s32(hi, bias))
    };
    let v = vqmovn_high_s32(vqmovn_s32(lo), hi);
    let o = y * w + x;
    match (w - x).min(8) {
        8 => store_i16x8(chunk_mut::<i16, 8>(tmp, o), v),
        4 => {
            let b = vget_lane_u64::<0>(vreinterpret_u64_s16(vget_low_s16(v))).to_le_bytes();
            let mut k = 0;
            while k < 4 {
                tmp[o + k] = i16::from_le_bytes([b[2 * k], b[2 * k + 1]]);
                k += 1;
            }
        }
        _ => {
            let b = vget_lane_u32::<0>(vreinterpret_u32_s16(vget_low_s16(v))).to_le_bytes();
            tmp[o] = i16::from_le_bytes([b[0], b[1]]);
            tmp[o + 1] = i16::from_le_bytes([b[2], b[3]]);
        }
    }
}

/// 8-tap ölçekli NEON çekirdeği (`PREP = false` → `put`, `PREP = true` → `prep`).
#[target_feature(enable = "neon")]
fn olcekli8_neon<const BPC: u32, const PREP: bool>(
    dst: &mut [u8],
    ds: usize,
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
    f: Filter2d,
) {
    let (th, tv) = tablolar(f, w, h);
    let gw = w.max(8);
    let mut ofs = [0usize; 136];
    let mut fil = [0u8; 136];
    let mut x = 0usize;
    while x < gw {
        if x < w {
            let t = mx + x as i32 * dx;
            ofs[x] = (t >> 10) as usize;
            fil[x] = ((t & 1023) >> 6) as u8;
        } else {
            ofs[x] = ofs[w - 1];
            fil[x] = 0;
        }
        x += 1;
    }

    let mut halka = [[0i16; 136]; 8];
    let mut bas = 0usize;
    let mut in_y: i32 = -8;
    let mut my = my;
    let yuv_v = vdupq_n_s32(if PREP {
        32
    } else if BPC == 12 {
        128
    } else {
        512
    });

    let mut y = 0usize;
    while y < h {
        let hedef_y = my >> 10;
        while in_y < hedef_y {
            yatay8_satir::<BPC>(
                &mut halka[bas],
                src,
                (in_y + 8) as usize * ss,
                &ofs,
                &fil,
                th,
                gw,
            );
            bas = (bas + 1) & 7;
            in_y += 1;
        }
        let fv = &FILTRE_I16[tv][((my & 1023) >> 6) as usize];
        let mut x = 0usize;
        while x < w {
            let r0 = load_i16x8(chunk::<i16, 8>(&halka[bas], x));
            let mut lo = vmlal_n_s16(yuv_v, vget_low_s16(r0), fv[0]);
            let mut hi = vmlal_n_s16(yuv_v, vget_high_s16(r0), fv[0]);
            let mut j = 1usize;
            while j < 8 {
                let rj = load_i16x8(chunk::<i16, 8>(&halka[(bas + j) & 7], x));
                lo = vmlal_n_s16(lo, vget_low_s16(rj), fv[j]);
                hi = vmlal_n_s16(hi, vget_high_s16(rj), fv[j]);
                j += 1;
            }
            if PREP {
                sakla_prep::<BPC>(tmp, y, x, w, vshrq_n_s32::<6>(lo), vshrq_n_s32::<6>(hi));
            } else if BPC == 12 {
                sakla_put::<BPC>(dst, ds, y, x, w, vshrq_n_s32::<8>(lo), vshrq_n_s32::<8>(hi));
            } else {
                sakla_put::<BPC>(
                    dst,
                    ds,
                    y,
                    x,
                    w,
                    vshrq_n_s32::<10>(lo),
                    vshrq_n_s32::<10>(hi),
                );
            }
            x += 8;
        }
        my += dy;
        y += 1;
    }
}

/// Bilinear ölçekli NEON çekirdeği (`PREP = false` → `put`, `PREP = true` → `prep`).
#[target_feature(enable = "neon")]
fn olcekli2_neon<const BPC: u32, const PREP: bool>(
    dst: &mut [u8],
    ds: usize,
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
) {
    let gw = w.max(8);
    let mut ofs = [0usize; 136];
    let mut agr_m = [0i16; 136];
    let mut x = 0usize;
    while x < gw {
        if x < w {
            let t = mx + x as i32 * dx;
            ofs[x] = (t >> 10) as usize;
            agr_m[x] = ((t & 1023) >> 6) as i16;
        } else {
            ofs[x] = ofs[w - 1];
            agr_m[x] = 0;
        }
        x += 1;
    }

    let mut halka = [[0i16; 136]; 2];
    let mut in_y: i32 = -2;
    let mut my = my;
    let yuv_h = vdupq_n_s32(2);
    let yuv_v = vdupq_n_s32(if PREP {
        8
    } else if BPC == 12 {
        32
    } else {
        128
    });

    let mut y = 0usize;
    while y < h {
        let src_y = my >> 10;
        while in_y < src_y {
            let slot = (in_y & 1) as usize;
            let taban = (in_y + 5) as usize * ss;
            let ara = &mut halka[slot];
            let mut g = 0usize;
            while g < gw {
                let mut s0_a = [0i16; 8];
                let mut s1_a = [0i16; 8];
                let mut k = 0usize;
                while k < 8 {
                    let o = ofs[g + k] + 3;
                    if BPC == 8 {
                        s0_a[k] = i16::from(src[taban + o]);
                        s1_a[k] = i16::from(src[taban + o + 1]);
                    } else {
                        let p = taban + 2 * o;
                        s0_a[k] = u16::from_le_bytes([src[p], src[p + 1]]) as i16;
                        s1_a[k] = u16::from_le_bytes([src[p + 2], src[p + 3]]) as i16;
                    }
                    k += 1;
                }
                let p0 = load_i16x8(&s0_a);
                let p1 = load_i16x8(&s1_a);
                let wm = load_i16x8(chunk::<i16, 8>(&agr_m, g));
                let w16 = core::arch::aarch64::vsubq_s16(core::arch::aarch64::vdupq_n_s16(16), wm);
                let mut lo = vmull_s16(vget_low_s16(p0), vget_low_s16(w16));
                lo = core::arch::aarch64::vmlal_s16(lo, vget_low_s16(p1), vget_low_s16(wm));
                let mut hi = vmull_high_s16(p0, w16);
                hi = vmlal_high_s16(hi, p1, wm);
                if BPC == 12 {
                    lo = vshrq_n_s32::<2>(vaddq_s32(lo, yuv_h));
                    hi = vshrq_n_s32::<2>(vaddq_s32(hi, yuv_h));
                }
                store_i16x8(
                    chunk_mut::<i16, 8>(ara, g),
                    vqmovn_high_s32(vqmovn_s32(lo), hi),
                );
                g += 8;
            }
            in_y += 1;
        }
        let m = ((my & 1023) >> 6) as i16;
        let m_inv = 16 - m;
        let (ra, rb) = (
            &halka[(src_y & 1) as usize],
            &halka[((src_y + 1) & 1) as usize],
        );
        let mut x = 0usize;
        while x < w {
            let a = load_i16x8(chunk::<i16, 8>(ra, x));
            let b = load_i16x8(chunk::<i16, 8>(rb, x));
            let mut lo = vmlal_n_s16(yuv_v, vget_low_s16(a), m_inv);
            lo = vmlal_n_s16(lo, vget_low_s16(b), m);
            let mut hi = vmlal_n_s16(yuv_v, vget_high_s16(a), m_inv);
            hi = vmlal_n_s16(hi, vget_high_s16(b), m);
            if PREP {
                sakla_prep::<BPC>(tmp, y, x, w, vshrq_n_s32::<4>(lo), vshrq_n_s32::<4>(hi));
            } else if BPC == 12 {
                sakla_put::<BPC>(dst, ds, y, x, w, vshrq_n_s32::<6>(lo), vshrq_n_s32::<6>(hi));
            } else {
                sakla_put::<BPC>(dst, ds, y, x, w, vshrq_n_s32::<8>(lo), vshrq_n_s32::<8>(hi));
            }
            x += 8;
        }
        my += dy;
        y += 1;
    }
}

/// `mc_scaled[FI]` NEON gövdesi (`put`).
#[target_feature(enable = "neon")]
pub(super) fn put_scaled_neon<const BPC: u32, const FI: usize>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
) {
    if FI == BILIN {
        olcekli2_neon::<BPC, false>(dst, ds, &mut [], src, ss, w, h, mx, my, dx, dy);
    } else {
        let f = Filter2d::from_index(FI).unwrap_or(Filter2d::Regular);
        olcekli8_neon::<BPC, false>(dst, ds, &mut [], src, ss, w, h, mx, my, dx, dy, f);
    }
}

/// `mct_scaled[FI]` NEON gövdesi (`prep`).
#[target_feature(enable = "neon")]
pub(super) fn prep_scaled_neon<const BPC: u32, const FI: usize>(
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
) {
    if FI == BILIN {
        olcekli2_neon::<BPC, true>(&mut [], 0, tmp, src, ss, w, h, mx, my, dx, dy);
    } else {
        let f = Filter2d::from_index(FI).unwrap_or(Filter2d::Regular);
        olcekli8_neon::<BPC, true>(&mut [], 0, tmp, src, ss, w, h, mx, my, dx, dy, f);
    }
}
