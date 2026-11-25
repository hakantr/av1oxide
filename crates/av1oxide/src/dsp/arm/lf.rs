//! `aarch64` NEON deblocking loop filter çekirdekleri (`P54` / `F5d` — `F5-LF`).
//!
//! Referans: `dav1d` `src/arm/loopfilter.h` + `src/arm/64/{loopfilter.S,loopfilter16.S}`
//! (`99f3354`). `loop_filter_sb[2][2]` (`8 / 10 / 12` bpc = `8` matris kaydı, `12` yuva)
//! `%100 #![forbid(unsafe_code)]` kararlı `core::arch::aarch64` NEON intrinsikleriyle
//! gerçeklenmiştir:
//! - Yatay kenarlar (`dir == 1`, dikey süzme `lpf_v_sb_{y,uv}`): 4 piksellik kenar
//!   dilimi doğrudan 128-bit `int32x4_t` vektörleriyle yüklenir ve paralel süzülür.
//! - Dikey kenarlar (`dir == 0`, yatay süzme `lpf_h_sb_{y,uv}`): yalnız `wd` boyunun
//!   izin verdiği sütunlar 4 satırdan okunup NEON `vtrn1q_s32`/`vtrn2q_s32`/`vzip*`
//!   matris devriğiyle 4-şeritli `int32x4_t` vektörlerine çevrilir, aynı NEON süzgeç
//!   çekirdeğinden geçirilip ters devrikle yalnız değişen sütunlara geri yazılır.

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use core::arch::aarch64::{
    int32x4_t, uint32x4_t, vabdq_s32, vaddq_s32, vandq_s32, vandq_u32, vbicq_u32, vbslq_s32,
    vcgtq_s32, vcleq_s32, vcombine_u16, vcreate_u16, vdup_n_u32, vdupq_n_s32, vget_lane_u32,
    vget_lane_u64, vget_low_u16, vmaxq_s32, vmaxvq_u32, vminq_s32, vmovl_u8, vmovl_u16, vmovn_u16,
    vmovn_u32, vmulq_n_s32, vorrq_u32, vreinterpret_u8_u32, vreinterpret_u32_u8,
    vreinterpret_u64_u16, vreinterpretq_s32_s64, vreinterpretq_s32_u32, vreinterpretq_s64_s32,
    vreinterpretq_u32_s32, vshlq_n_s32, vshrq_n_s32, vsubq_s32, vtrn1q_s32, vtrn1q_s64, vtrn2q_s32,
    vtrn2q_s64,
};

use super::mem::{chunk, chunk_mut};
use crate::cpu::NeonToken;
use crate::dsp::lf::{LimLut, LoopFilterSbFn};

/// `loop_filter_sb[PLANE][DIR]` için `NeonToken`-kapılı giriş fonksiyonu döndürür.
#[must_use]
pub fn loop_filter_sb_fn<const BPC: u32>(
    _neon: NeonToken,
    plane: usize,
    dir: usize,
) -> Option<LoopFilterSbFn> {
    match (plane, dir) {
        (0, 0) => Some(super::giris::lf_sb::<BPC, 0, 0>),
        (0, 1) => Some(super::giris::lf_sb::<BPC, 0, 1>),
        (1, 0) => Some(super::giris::lf_sb::<BPC, 1, 0>),
        (1, 1) => Some(super::giris::lf_sb::<BPC, 1, 1>),
        _ => None,
    }
}

#[inline(always)]
fn level_or(level: &[[u8; 4]], at: usize, fallback: usize, idx: usize) -> u32 {
    let l = level[at][idx];
    u32::from(if l != 0 { l } else { level[fallback][idx] })
}

#[target_feature(enable = "neon")]
#[inline]
fn load_row4<const BPC: u32>(data: &[u8], off: usize) -> int32x4_t {
    if BPC == 8 {
        let w = u32::from_ne_bytes(*chunk::<u8, 4>(data, off));
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vreinterpret_u8_u32(
            vdup_n_u32(w),
        )))))
    } else {
        let w = u64::from_ne_bytes(*chunk::<u8, 8>(data, off));
        vreinterpretq_s32_u32(vmovl_u16(vcreate_u16(w)))
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn store_row4<const BPC: u32>(data: &mut [u8], off: usize, v: int32x4_t) {
    let u16v = vmovn_u32(vreinterpretq_u32_s32(v));
    if BPC == 8 {
        let u8v = vmovn_u16(vcombine_u16(u16v, u16v));
        let w = vget_lane_u32::<0>(vreinterpret_u32_u8(u8v));
        *chunk_mut::<u8, 4>(data, off) = w.to_ne_bytes();
    } else {
        let w = vget_lane_u64::<0>(vreinterpret_u64_u16(u16v));
        *chunk_mut::<u8, 8>(data, off) = w.to_ne_bytes();
    }
}

/// 4×4 `int32x4_t` matris devriği (kendi tersidir).
#[target_feature(enable = "neon")]
#[inline]
fn transpose_4x4(
    r0: int32x4_t,
    r1: int32x4_t,
    r2: int32x4_t,
    r3: int32x4_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let t0 = vtrn1q_s32(r0, r1);
    let t1 = vtrn2q_s32(r0, r1);
    let t2 = vtrn1q_s32(r2, r3);
    let t3 = vtrn2q_s32(r2, r3);
    let c0 = vreinterpretq_s32_s64(vtrn1q_s64(
        vreinterpretq_s64_s32(t0),
        vreinterpretq_s64_s32(t2),
    ));
    let c1 = vreinterpretq_s32_s64(vtrn1q_s64(
        vreinterpretq_s64_s32(t1),
        vreinterpretq_s64_s32(t3),
    ));
    let c2 = vreinterpretq_s32_s64(vtrn2q_s64(
        vreinterpretq_s64_s32(t0),
        vreinterpretq_s64_s32(t2),
    ));
    let c3 = vreinterpretq_s32_s64(vtrn2q_s64(
        vreinterpretq_s64_s32(t1),
        vreinterpretq_s64_s32(t3),
    ));
    (c0, c1, c2, c3)
}

#[target_feature(enable = "neon")]
#[inline]
fn clamp_px<const BPC: u32>(v: int32x4_t) -> int32x4_t {
    let zero = vdupq_n_s32(0);
    let max_v = vdupq_n_s32((1i32 << BPC) - 1);
    vminq_s32(vmaxq_s32(v, zero), max_v)
}

#[target_feature(enable = "neon")]
#[inline]
fn narrow_filter_4<const BPC: u32>(
    p1: int32x4_t,
    p0: int32x4_t,
    q0: int32x4_t,
    q1: int32x4_t,
    abs_p1p0: int32x4_t,
    abs_q1q0: int32x4_t,
    vh: int32x4_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let bd_min_8 = BPC - 8;
    let clip_lo = vdupq_n_s32(-128 << bd_min_8);
    let clip_hi = vdupq_n_s32((128 << bd_min_8) - 1);
    let iclip = |x: int32x4_t| vminq_s32(vmaxq_s32(x, clip_lo), clip_hi);

    let hev = vorrq_u32(vcgtq_s32(abs_p1p0, vh), vcgtq_s32(abs_q1q0, vh));
    let p1_sub_q1 = iclip(vsubq_s32(p1, q1));
    let hev_term = vandq_s32(p1_sub_q1, vreinterpretq_s32_u32(hev));
    let fv = iclip(vaddq_s32(vmulq_n_s32(vsubq_s32(q0, p0), 3), hev_term));

    let f1 = vshrq_n_s32::<3>(vminq_s32(vaddq_s32(fv, vdupq_n_s32(4)), clip_hi));
    let f2 = vshrq_n_s32::<3>(vminq_s32(vaddq_s32(fv, vdupq_n_s32(3)), clip_hi));

    let np0 = clamp_px::<BPC>(vaddq_s32(p0, f2));
    let nq0 = clamp_px::<BPC>(vsubq_s32(q0, f1));

    let fv2 = vshrq_n_s32::<1>(vaddq_s32(f1, vdupq_n_s32(1)));
    let np1 = vbslq_s32(hev, p1, clamp_px::<BPC>(vaddq_s32(p1, fv2)));
    let nq1 = vbslq_s32(hev, q1, clamp_px::<BPC>(vsubq_s32(q1, fv2)));
    (np1, np0, nq0, nq1)
}

#[target_feature(enable = "neon")]
#[inline]
fn filter_8tap_4<const BPC: u32>(
    p3: int32x4_t,
    p2: int32x4_t,
    p1: int32x4_t,
    p0: int32x4_t,
    q0: int32x4_t,
    q1: int32x4_t,
    q2: int32x4_t,
    q3: int32x4_t,
) -> (
    int32x4_t,
    int32x4_t,
    int32x4_t,
    int32x4_t,
    int32x4_t,
    int32x4_t,
) {
    let c4 = vdupq_n_s32(4);
    let r = |x: int32x4_t| clamp_px::<BPC>(vshrq_n_s32::<3>(vaddq_s32(x, c4)));

    // sum_base = p2 + p1 + p0 + q0
    let p1p0q0 = vaddq_s32(vaddq_s32(p1, p0), q0);
    let o_p3 = r(vaddq_s32(
        vaddq_s32(vmulq_n_s32(p3, 3), vshlq_n_s32::<1>(p2)),
        p1p0q0,
    ));
    let o_p2 = r(vaddq_s32(
        vaddq_s32(vshlq_n_s32::<1>(p3), p2),
        vaddq_s32(vaddq_s32(p1, p1p0q0), q1),
    ));
    let o_p1 = r(vaddq_s32(
        vaddq_s32(p3, p2),
        vaddq_s32(vaddq_s32(p0, p1p0q0), vaddq_s32(q1, q2)),
    ));
    let q1q2q3 = vaddq_s32(vaddq_s32(q1, q2), q3);
    let o_q0 = r(vaddq_s32(vaddq_s32(p2, p1p0q0), vaddq_s32(q0, q1q2q3)));
    let o_q1 = r(vaddq_s32(vaddq_s32(p1p0q0, q1), vaddq_s32(q1q2q3, q3)));
    let o_q2 = r(vaddq_s32(
        vaddq_s32(vaddq_s32(p0, q0), q1),
        vaddq_s32(vshlq_n_s32::<1>(q2), vmulq_n_s32(q3, 3)),
    ));
    (o_p3, o_p2, o_p1, o_q0, o_q1, o_q2)
}

#[target_feature(enable = "neon")]
#[inline]
fn filter_6tap_4<const BPC: u32>(
    p2: int32x4_t,
    p1: int32x4_t,
    p0: int32x4_t,
    q0: int32x4_t,
    q1: int32x4_t,
    q2: int32x4_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let c4 = vdupq_n_s32(4);
    let r = |x: int32x4_t| clamp_px::<BPC>(vshrq_n_s32::<3>(vaddq_s32(x, c4)));
    let p0_2 = vshlq_n_s32::<1>(p0);
    let q0_2 = vshlq_n_s32::<1>(q0);
    let p1_2 = vshlq_n_s32::<1>(p1);
    let q1_2 = vshlq_n_s32::<1>(q1);
    let o_p2 = r(vaddq_s32(
        vaddq_s32(vmulq_n_s32(p2, 3), p1_2),
        vaddq_s32(p0_2, q0),
    ));
    let o_p1 = r(vaddq_s32(
        vaddq_s32(p2, p1_2),
        vaddq_s32(vaddq_s32(p0_2, q0_2), q1),
    ));
    let o_q0 = r(vaddq_s32(
        vaddq_s32(p1, p0_2),
        vaddq_s32(vaddq_s32(q0_2, q1_2), q2),
    ));
    let o_q1 = r(vaddq_s32(
        vaddq_s32(p0, q0_2),
        vaddq_s32(q1_2, vmulq_n_s32(q2, 3)),
    ));
    (o_p2, o_p1, o_q0, o_q1)
}

#[target_feature(enable = "neon")]
#[inline]
fn filter_16tap_4<const BPC: u32>(
    p6: int32x4_t,
    p5: int32x4_t,
    p4: int32x4_t,
    p3: int32x4_t,
    p2: int32x4_t,
    p1: int32x4_t,
    p0: int32x4_t,
    q0: int32x4_t,
    q1: int32x4_t,
    q2: int32x4_t,
    q3: int32x4_t,
    q4: int32x4_t,
    q5: int32x4_t,
    q6: int32x4_t,
) -> [int32x4_t; 12] {
    let c8 = vdupq_n_s32(8);
    let r = |x: int32x4_t| clamp_px::<BPC>(vshrq_n_s32::<4>(vaddq_s32(x, c8)));
    let s2 = |a: int32x4_t, b: int32x4_t| vaddq_s32(a, b);
    let s3 = |a: int32x4_t, b: int32x4_t, c: int32x4_t| vaddq_s32(vaddq_s32(a, b), c);
    let m2 = |a: int32x4_t| vshlq_n_s32::<1>(a);

    let o0 = r(s3(
        s3(vmulq_n_s32(p6, 7), m2(p5), m2(p4)),
        s3(p3, p2, p1),
        s2(p0, q0),
    ));
    let o1 = r(s3(
        s3(vmulq_n_s32(p6, 5), m2(p5), m2(p4)),
        s3(m2(p3), p2, p1),
        s3(p0, q0, q1),
    ));
    let o2 = r(s3(
        s3(vmulq_n_s32(p6, 4), p5, m2(p4)),
        s3(m2(p3), m2(p2), p1),
        s2(s2(p0, q0), s2(q1, q2)),
    ));
    let o3 = r(s3(
        s3(vmulq_n_s32(p6, 3), p5, p4),
        s3(m2(p3), m2(p2), m2(p1)),
        s3(s2(p0, q0), s2(q1, q2), q3),
    ));
    let o4 = r(s3(
        s3(m2(p6), p5, s2(p4, p3)),
        s3(m2(p2), m2(p1), m2(p0)),
        s3(s2(q0, q1), s2(q2, q3), q4),
    ));
    let o5 = r(s3(
        s3(s2(p6, p5), s2(p4, p3), p2),
        s3(m2(p1), m2(p0), m2(q0)),
        s3(s2(q1, q2), s2(q3, q4), q5),
    ));
    let o6 = r(s3(
        s3(s2(p5, p4), s2(p3, p2), p1),
        s3(m2(p0), m2(q0), m2(q1)),
        s3(s2(q2, q3), s2(q4, q5), q6),
    ));
    let o7 = r(s3(
        s3(s2(p4, p3), s2(p2, p1), p0),
        s3(m2(q0), m2(q1), m2(q2)),
        s3(q3, s2(q4, q5), m2(q6)),
    ));
    let o8 = r(s3(
        s3(s2(p3, p2), s2(p1, p0), q0),
        s3(m2(q1), m2(q2), m2(q3)),
        s3(q4, q5, vmulq_n_s32(q6, 3)),
    ));
    let o9 = r(s3(
        s2(s2(p2, p1), s2(p0, q0)),
        s3(q1, m2(q2), m2(q3)),
        s3(m2(q4), q5, vmulq_n_s32(q6, 4)),
    ));
    let o10 = r(s3(
        s3(p1, p0, q0),
        s3(q1, q2, m2(q3)),
        s3(m2(q4), m2(q5), vmulq_n_s32(q6, 5)),
    ));
    let o11 = r(s3(
        s2(p0, q0),
        s3(q1, q2, q3),
        s3(m2(q4), m2(q5), vmulq_n_s32(q6, 7)),
    ));
    [o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11]
}

/// Yatay kenar (`dir == 1`, dikey süzme: `stride_b = stride`, 4 sütun bitişik).
#[target_feature(enable = "neon")]
#[inline]
fn filter_v_edge<const BPC: u32>(
    data: &mut [u8],
    off: usize,
    stride: usize,
    wd: usize,
    l: u32,
    lut: &LimLut,
) {
    let bd_min_8 = BPC - 8;
    let vf = vdupq_n_s32(1i32 << bd_min_8);
    let ve = vdupq_n_s32(lut.e[l as usize] << bd_min_8);
    let vi = vdupq_n_s32(lut.i[l as usize] << bd_min_8);
    let vh = vdupq_n_s32(((l >> 4) as i32) << bd_min_8);

    let p1 = load_row4::<BPC>(data, off - 2 * stride);
    let p0 = load_row4::<BPC>(data, off - stride);
    let q0 = load_row4::<BPC>(data, off);
    let q1 = load_row4::<BPC>(data, off + stride);

    let abs_p1p0 = vabdq_s32(p1, p0);
    let abs_q1q0 = vabdq_s32(q1, q0);
    let abs_p0q0 = vabdq_s32(p0, q0);
    let abs_p1q1 = vabdq_s32(p1, q1);

    let edge_term = vaddq_s32(vshlq_n_s32::<1>(abs_p0q0), vshrq_n_s32::<1>(abs_p1q1));
    let mut fm: uint32x4_t = vandq_u32(
        vandq_u32(vcleq_s32(abs_p1p0, vi), vcleq_s32(abs_q1q0, vi)),
        vcleq_s32(edge_term, ve),
    );

    if wd == 4 {
        if vmaxvq_u32(fm) == 0 {
            return;
        }
        let (np1, np0, nq0, nq1) = narrow_filter_4::<BPC>(p1, p0, q0, q1, abs_p1p0, abs_q1q0, vh);
        store_row4::<BPC>(data, off - 2 * stride, vbslq_s32(fm, np1, p1));
        store_row4::<BPC>(data, off - stride, vbslq_s32(fm, np0, p0));
        store_row4::<BPC>(data, off, vbslq_s32(fm, nq0, q0));
        store_row4::<BPC>(data, off + stride, vbslq_s32(fm, nq1, q1));
        return;
    }

    let p2 = load_row4::<BPC>(data, off - 3 * stride);
    let q2 = load_row4::<BPC>(data, off + 2 * stride);
    fm = vandq_u32(
        fm,
        vandq_u32(
            vcleq_s32(vabdq_s32(p2, p1), vi),
            vcleq_s32(vabdq_s32(q2, q1), vi),
        ),
    );

    if wd == 6 {
        if vmaxvq_u32(fm) == 0 {
            return;
        }
        let flat8in = vandq_u32(
            vandq_u32(vcleq_s32(vabdq_s32(p2, p0), vf), vcleq_s32(abs_p1p0, vf)),
            vandq_u32(vcleq_s32(abs_q1q0, vf), vcleq_s32(vabdq_s32(q2, q0), vf)),
        );
        let (np1, np0, nq0, nq1) = narrow_filter_4::<BPC>(p1, p0, q0, q1, abs_p1p0, abs_q1q0, vh);
        let (fp2, fp1, fq0, fq1) = filter_6tap_4::<BPC>(p2, p1, p0, q0, q1, q2);
        let m6 = vandq_u32(fm, flat8in);
        store_row4::<BPC>(
            data,
            off - 2 * stride,
            vbslq_s32(m6, fp2, vbslq_s32(fm, np1, p1)),
        );
        store_row4::<BPC>(
            data,
            off - stride,
            vbslq_s32(m6, fp1, vbslq_s32(fm, np0, p0)),
        );
        store_row4::<BPC>(data, off, vbslq_s32(m6, fq0, vbslq_s32(fm, nq0, q0)));
        store_row4::<BPC>(
            data,
            off + stride,
            vbslq_s32(m6, fq1, vbslq_s32(fm, nq1, q1)),
        );
        return;
    }

    let p3 = load_row4::<BPC>(data, off - 4 * stride);
    let q3 = load_row4::<BPC>(data, off + 3 * stride);
    fm = vandq_u32(
        fm,
        vandq_u32(
            vcleq_s32(vabdq_s32(p3, p2), vi),
            vcleq_s32(vabdq_s32(q3, q2), vi),
        ),
    );
    if vmaxvq_u32(fm) == 0 {
        return;
    }

    let flat8in = vandq_u32(
        vandq_u32(
            vandq_u32(vcleq_s32(vabdq_s32(p2, p0), vf), vcleq_s32(abs_p1p0, vf)),
            vandq_u32(vcleq_s32(abs_q1q0, vf), vcleq_s32(vabdq_s32(q2, q0), vf)),
        ),
        vandq_u32(
            vcleq_s32(vabdq_s32(p3, p0), vf),
            vcleq_s32(vabdq_s32(q3, q0), vf),
        ),
    );

    let (np1, np0, nq0, nq1) = narrow_filter_4::<BPC>(p1, p0, q0, q1, abs_p1p0, abs_q1q0, vh);
    let (fp3, fp2, fp1, fq0, fq1, fq2) = filter_8tap_4::<BPC>(p3, p2, p1, p0, q0, q1, q2, q3);

    if wd == 8 {
        let m8 = vandq_u32(fm, flat8in);
        store_row4::<BPC>(data, off - 3 * stride, vbslq_s32(m8, fp3, p2));
        store_row4::<BPC>(
            data,
            off - 2 * stride,
            vbslq_s32(m8, fp2, vbslq_s32(fm, np1, p1)),
        );
        store_row4::<BPC>(
            data,
            off - stride,
            vbslq_s32(m8, fp1, vbslq_s32(fm, np0, p0)),
        );
        store_row4::<BPC>(data, off, vbslq_s32(m8, fq0, vbslq_s32(fm, nq0, q0)));
        store_row4::<BPC>(
            data,
            off + stride,
            vbslq_s32(m8, fq1, vbslq_s32(fm, nq1, q1)),
        );
        store_row4::<BPC>(data, off + 2 * stride, vbslq_s32(m8, fq2, q2));
        return;
    }

    // wd >= 16
    let p6 = load_row4::<BPC>(data, off - 7 * stride);
    let p5 = load_row4::<BPC>(data, off - 6 * stride);
    let p4 = load_row4::<BPC>(data, off - 5 * stride);
    let q4 = load_row4::<BPC>(data, off + 4 * stride);
    let q5 = load_row4::<BPC>(data, off + 5 * stride);
    let q6 = load_row4::<BPC>(data, off + 6 * stride);

    let flat8out = vandq_u32(
        vandq_u32(
            vandq_u32(
                vcleq_s32(vabdq_s32(p6, p0), vf),
                vcleq_s32(vabdq_s32(p5, p0), vf),
            ),
            vcleq_s32(vabdq_s32(p4, p0), vf),
        ),
        vandq_u32(
            vandq_u32(
                vcleq_s32(vabdq_s32(q4, q0), vf),
                vcleq_s32(vabdq_s32(q5, q0), vf),
            ),
            vcleq_s32(vabdq_s32(q6, q0), vf),
        ),
    );
    let m16 = vandq_u32(fm, vandq_u32(flat8in, flat8out));
    let m8 = vandq_u32(fm, vbicq_u32(flat8in, flat8out));

    if vmaxvq_u32(m16) == 0 {
        store_row4::<BPC>(data, off - 3 * stride, vbslq_s32(m8, fp3, p2));
        store_row4::<BPC>(
            data,
            off - 2 * stride,
            vbslq_s32(m8, fp2, vbslq_s32(fm, np1, p1)),
        );
        store_row4::<BPC>(
            data,
            off - stride,
            vbslq_s32(m8, fp1, vbslq_s32(fm, np0, p0)),
        );
        store_row4::<BPC>(data, off, vbslq_s32(m8, fq0, vbslq_s32(fm, nq0, q0)));
        store_row4::<BPC>(
            data,
            off + stride,
            vbslq_s32(m8, fq1, vbslq_s32(fm, nq1, q1)),
        );
        store_row4::<BPC>(data, off + 2 * stride, vbslq_s32(m8, fq2, q2));
        return;
    }

    let w16 = filter_16tap_4::<BPC>(p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6);
    store_row4::<BPC>(data, off - 6 * stride, vbslq_s32(m16, w16[0], p5));
    store_row4::<BPC>(data, off - 5 * stride, vbslq_s32(m16, w16[1], p4));
    store_row4::<BPC>(data, off - 4 * stride, vbslq_s32(m16, w16[2], p3));
    store_row4::<BPC>(
        data,
        off - 3 * stride,
        vbslq_s32(m16, w16[3], vbslq_s32(m8, fp3, p2)),
    );
    store_row4::<BPC>(
        data,
        off - 2 * stride,
        vbslq_s32(m16, w16[4], vbslq_s32(m8, fp2, vbslq_s32(fm, np1, p1))),
    );
    store_row4::<BPC>(
        data,
        off - stride,
        vbslq_s32(m16, w16[5], vbslq_s32(m8, fp1, vbslq_s32(fm, np0, p0))),
    );
    store_row4::<BPC>(
        data,
        off,
        vbslq_s32(m16, w16[6], vbslq_s32(m8, fq0, vbslq_s32(fm, nq0, q0))),
    );
    store_row4::<BPC>(
        data,
        off + stride,
        vbslq_s32(m16, w16[7], vbslq_s32(m8, fq1, vbslq_s32(fm, nq1, q1))),
    );
    store_row4::<BPC>(
        data,
        off + 2 * stride,
        vbslq_s32(m16, w16[8], vbslq_s32(m8, fq2, q2)),
    );
    store_row4::<BPC>(data, off + 3 * stride, vbslq_s32(m16, w16[9], q3));
    store_row4::<BPC>(data, off + 4 * stride, vbslq_s32(m16, w16[10], q4));
    store_row4::<BPC>(data, off + 5 * stride, vbslq_s32(m16, w16[11], q5));
}

#[target_feature(enable = "neon")]
#[inline]
fn load_cols4_transposed<const BPC: u32>(
    data: &[u8],
    off: usize,
    stride: usize,
    col_rel: isize,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let bpp = if BPC == 8 { 1isize } else { 2isize };
    let base = (off as isize + col_rel * bpp) as usize;
    let r0 = load_row4::<BPC>(data, base);
    let r1 = load_row4::<BPC>(data, base + stride);
    let r2 = load_row4::<BPC>(data, base + 2 * stride);
    let r3 = load_row4::<BPC>(data, base + 3 * stride);
    transpose_4x4(r0, r1, r2, r3)
}

#[target_feature(enable = "neon")]
#[inline]
fn store_cols4_transposed<const BPC: u32>(
    data: &mut [u8],
    off: usize,
    stride: usize,
    col_rel: isize,
    c0: int32x4_t,
    c1: int32x4_t,
    c2: int32x4_t,
    c3: int32x4_t,
) {
    let bpp = if BPC == 8 { 1isize } else { 2isize };
    let base = (off as isize + col_rel * bpp) as usize;
    let (r0, r1, r2, r3) = transpose_4x4(c0, c1, c2, c3);
    store_row4::<BPC>(data, base, r0);
    store_row4::<BPC>(data, base + stride, r1);
    store_row4::<BPC>(data, base + 2 * stride, r2);
    store_row4::<BPC>(data, base + 3 * stride, r3);
}

#[target_feature(enable = "neon")]
#[inline]
fn load_single_col<const BPC: u32>(
    data: &[u8],
    off: usize,
    stride: usize,
    col_rel: isize,
) -> int32x4_t {
    let bpp = if BPC == 8 { 1isize } else { 2isize };
    let base = (off as isize + col_rel * bpp) as usize;
    let mut pack = 0u64;
    for r in 0..4 {
        let o = base + r * stride;
        let px = if BPC == 8 {
            u64::from(*chunk::<u8, 1>(data, o).first().unwrap_or(&0))
        } else {
            u64::from(u16::from_le_bytes(*chunk::<u8, 2>(data, o)))
        };
        pack |= px << (r * 16);
    }
    vreinterpretq_s32_u32(vmovl_u16(vcreate_u16(pack)))
}

#[target_feature(enable = "neon")]
#[inline]
fn store_single_col<const BPC: u32>(
    data: &mut [u8],
    off: usize,
    stride: usize,
    col_rel: isize,
    v: int32x4_t,
) {
    let bpp = if BPC == 8 { 1isize } else { 2isize };
    let base = (off as isize + col_rel * bpp) as usize;
    let u16v = vmovn_u32(vreinterpretq_u32_s32(v));
    let pack = vget_lane_u64::<0>(vreinterpret_u64_u16(u16v));
    for r in 0..4 {
        let px = (pack >> (r * 16)) as u16;
        let o = base + r * stride;
        if BPC == 8 {
            *chunk_mut::<u8, 1>(data, o) = [px as u8];
        } else {
            *chunk_mut::<u8, 2>(data, o) = px.to_le_bytes();
        }
    }
}

/// Dikey kenar (`dir == 0`, yatay süzme: 4 satır paralel, yalnız `wd`'nin izin verdiği pikseller okunur).
#[target_feature(enable = "neon")]
#[inline]
fn filter_h_edge<const BPC: u32>(
    data: &mut [u8],
    off: usize,
    stride: usize,
    wd: usize,
    l: u32,
    lut: &LimLut,
) {
    let bd_min_8 = BPC - 8;
    let vf = vdupq_n_s32(1i32 << bd_min_8);
    let ve = vdupq_n_s32(lut.e[l as usize] << bd_min_8);
    let vi = vdupq_n_s32(lut.i[l as usize] << bd_min_8);
    let vh = vdupq_n_s32(((l >> 4) as i32) << bd_min_8);

    let (p1, p0, q0, q1) = load_cols4_transposed::<BPC>(data, off, stride, -2);

    let abs_p1p0 = vabdq_s32(p1, p0);
    let abs_q1q0 = vabdq_s32(q1, q0);
    let abs_p0q0 = vabdq_s32(p0, q0);
    let abs_p1q1 = vabdq_s32(p1, q1);

    let edge_term = vaddq_s32(vshlq_n_s32::<1>(abs_p0q0), vshrq_n_s32::<1>(abs_p1q1));
    let mut fm: uint32x4_t = vandq_u32(
        vandq_u32(vcleq_s32(abs_p1p0, vi), vcleq_s32(abs_q1q0, vi)),
        vcleq_s32(edge_term, ve),
    );

    if wd == 4 {
        if vmaxvq_u32(fm) == 0 {
            return;
        }
        let (np1, np0, nq0, nq1) = narrow_filter_4::<BPC>(p1, p0, q0, q1, abs_p1p0, abs_q1q0, vh);
        store_cols4_transposed::<BPC>(
            data,
            off,
            stride,
            -2,
            vbslq_s32(fm, np1, p1),
            vbslq_s32(fm, np0, p0),
            vbslq_s32(fm, nq0, q0),
            vbslq_s32(fm, nq1, q1),
        );
        return;
    }

    let p2 = load_single_col::<BPC>(data, off, stride, -3);
    let q2 = load_single_col::<BPC>(data, off, stride, 2);
    fm = vandq_u32(
        fm,
        vandq_u32(
            vcleq_s32(vabdq_s32(p2, p1), vi),
            vcleq_s32(vabdq_s32(q2, q1), vi),
        ),
    );

    if wd == 6 {
        if vmaxvq_u32(fm) == 0 {
            return;
        }
        let flat8in = vandq_u32(
            vandq_u32(vcleq_s32(vabdq_s32(p2, p0), vf), vcleq_s32(abs_p1p0, vf)),
            vandq_u32(vcleq_s32(abs_q1q0, vf), vcleq_s32(vabdq_s32(q2, q0), vf)),
        );
        let (np1, np0, nq0, nq1) = narrow_filter_4::<BPC>(p1, p0, q0, q1, abs_p1p0, abs_q1q0, vh);
        let (fp2, fp1, fq0, fq1) = filter_6tap_4::<BPC>(p2, p1, p0, q0, q1, q2);
        let m6 = vandq_u32(fm, flat8in);
        store_cols4_transposed::<BPC>(
            data,
            off,
            stride,
            -2,
            vbslq_s32(m6, fp2, vbslq_s32(fm, np1, p1)),
            vbslq_s32(m6, fp1, vbslq_s32(fm, np0, p0)),
            vbslq_s32(m6, fq0, vbslq_s32(fm, nq0, q0)),
            vbslq_s32(m6, fq1, vbslq_s32(fm, nq1, q1)),
        );
        return;
    }

    let p3 = load_single_col::<BPC>(data, off, stride, -4);
    let q3 = load_single_col::<BPC>(data, off, stride, 3);
    fm = vandq_u32(
        fm,
        vandq_u32(
            vcleq_s32(vabdq_s32(p3, p2), vi),
            vcleq_s32(vabdq_s32(q3, q2), vi),
        ),
    );
    if vmaxvq_u32(fm) == 0 {
        return;
    }

    let flat8in = vandq_u32(
        vandq_u32(
            vandq_u32(vcleq_s32(vabdq_s32(p2, p0), vf), vcleq_s32(abs_p1p0, vf)),
            vandq_u32(vcleq_s32(abs_q1q0, vf), vcleq_s32(vabdq_s32(q2, q0), vf)),
        ),
        vandq_u32(
            vcleq_s32(vabdq_s32(p3, p0), vf),
            vcleq_s32(vabdq_s32(q3, q0), vf),
        ),
    );

    let (np1, np0, nq0, nq1) = narrow_filter_4::<BPC>(p1, p0, q0, q1, abs_p1p0, abs_q1q0, vh);
    let (fp3, fp2, fp1, fq0, fq1, fq2) = filter_8tap_4::<BPC>(p3, p2, p1, p0, q0, q1, q2, q3);

    if wd == 8 {
        let m8 = vandq_u32(fm, flat8in);
        store_single_col::<BPC>(data, off, stride, -3, vbslq_s32(m8, fp3, p2));
        store_cols4_transposed::<BPC>(
            data,
            off,
            stride,
            -2,
            vbslq_s32(m8, fp2, vbslq_s32(fm, np1, p1)),
            vbslq_s32(m8, fp1, vbslq_s32(fm, np0, p0)),
            vbslq_s32(m8, fq0, vbslq_s32(fm, nq0, q0)),
            vbslq_s32(m8, fq1, vbslq_s32(fm, nq1, q1)),
        );
        store_single_col::<BPC>(data, off, stride, 2, vbslq_s32(m8, fq2, q2));
        return;
    }

    // wd >= 16: yalnız fm geçerli olduğunda dış p6..p4 ve q4..q6 okunur.
    let p6 = load_single_col::<BPC>(data, off, stride, -7);
    let p5 = load_single_col::<BPC>(data, off, stride, -6);
    let p4 = load_single_col::<BPC>(data, off, stride, -5);
    let q4 = load_single_col::<BPC>(data, off, stride, 4);
    let q5 = load_single_col::<BPC>(data, off, stride, 5);
    let q6 = load_single_col::<BPC>(data, off, stride, 6);

    let flat8out = vandq_u32(
        vandq_u32(
            vandq_u32(
                vcleq_s32(vabdq_s32(p6, p0), vf),
                vcleq_s32(vabdq_s32(p5, p0), vf),
            ),
            vcleq_s32(vabdq_s32(p4, p0), vf),
        ),
        vandq_u32(
            vandq_u32(
                vcleq_s32(vabdq_s32(q4, q0), vf),
                vcleq_s32(vabdq_s32(q5, q0), vf),
            ),
            vcleq_s32(vabdq_s32(q6, q0), vf),
        ),
    );
    let m16 = vandq_u32(fm, vandq_u32(flat8in, flat8out));
    let m8 = vandq_u32(fm, vbicq_u32(flat8in, flat8out));

    if vmaxvq_u32(m16) == 0 {
        store_single_col::<BPC>(data, off, stride, -3, vbslq_s32(m8, fp3, p2));
        store_cols4_transposed::<BPC>(
            data,
            off,
            stride,
            -2,
            vbslq_s32(m8, fp2, vbslq_s32(fm, np1, p1)),
            vbslq_s32(m8, fp1, vbslq_s32(fm, np0, p0)),
            vbslq_s32(m8, fq0, vbslq_s32(fm, nq0, q0)),
            vbslq_s32(m8, fq1, vbslq_s32(fm, nq1, q1)),
        );
        store_single_col::<BPC>(data, off, stride, 2, vbslq_s32(m8, fq2, q2));
        return;
    }

    let w16 = filter_16tap_4::<BPC>(p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6);
    store_cols4_transposed::<BPC>(
        data,
        off,
        stride,
        -6,
        vbslq_s32(m16, w16[0], p5),
        vbslq_s32(m16, w16[1], p4),
        vbslq_s32(m16, w16[2], p3),
        vbslq_s32(m16, w16[3], vbslq_s32(m8, fp3, p2)),
    );
    store_cols4_transposed::<BPC>(
        data,
        off,
        stride,
        -2,
        vbslq_s32(m16, w16[4], vbslq_s32(m8, fp2, vbslq_s32(fm, np1, p1))),
        vbslq_s32(m16, w16[5], vbslq_s32(m8, fp1, vbslq_s32(fm, np0, p0))),
        vbslq_s32(m16, w16[6], vbslq_s32(m8, fq0, vbslq_s32(fm, nq0, q0))),
        vbslq_s32(m16, w16[7], vbslq_s32(m8, fq1, vbslq_s32(fm, nq1, q1))),
    );
    store_cols4_transposed::<BPC>(
        data,
        off,
        stride,
        2,
        vbslq_s32(m16, w16[8], vbslq_s32(m8, fq2, q2)),
        vbslq_s32(m16, w16[9], q3),
        vbslq_s32(m16, w16[10], q4),
        vbslq_s32(m16, w16[11], q5),
    );
}

#[target_feature(enable = "neon")]
pub(super) fn lpf_sb_neon<const BPC: u32, const PLANE: usize, const DIR: usize>(
    dst: &mut [u8],
    stride: usize,
    mask: &[u8],
    mask_stride: usize,
    level: &[[u8; 4]],
    lvl_stride: usize,
    lvl_idx: usize,
    lut: &LimLut,
    w4: usize,
    y0: usize,
    count: usize,
) {
    let bpp = if BPC == 8 { 1 } else { 2 };
    if DIR == 0 {
        for gx in 1..w4 {
            for gy in y0..y0 + count {
                let v = mask[gy * mask_stride + gx];
                if v == 0 {
                    continue;
                }
                let l = level_or(
                    level,
                    gy * lvl_stride + gx,
                    gy * lvl_stride + gx - 1,
                    lvl_idx,
                );
                if l == 0 {
                    continue;
                }
                let off = gy * 4 * stride + gx * 4 * bpp;
                let wd = if PLANE == 0 {
                    4 << (v - 1)
                } else {
                    4 + 2 * usize::from(v - 1)
                };
                filter_h_edge::<BPC>(dst, off, stride, wd, l, lut);
            }
        }
    } else {
        for gy in y0..y0 + count {
            if gy == 0 {
                continue;
            }
            for gx in 0..w4 {
                let v = mask[gy * mask_stride + gx];
                if v == 0 {
                    continue;
                }
                let l = level_or(
                    level,
                    gy * lvl_stride + gx,
                    (gy - 1) * lvl_stride + gx,
                    lvl_idx,
                );
                if l == 0 {
                    continue;
                }
                let off = gy * 4 * stride + gx * 4 * bpp;
                let wd = if PLANE == 0 {
                    4 << (v - 1)
                } else {
                    4 + 2 * usize::from(v - 1)
                };
                filter_v_edge::<BPC>(dst, off, stride, wd, l, lut);
            }
        }
    }
}
