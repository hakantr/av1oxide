//! WebAssembly `simd128` deblocking (loop filter) çekirdekleri (`12` yuva, `8 / 10 / 12` bpc).
//!
//! Kapsam: `loop_filter_sb[0..1][0..1]` (`luma`/`chroma` × `h`/`v`).
//! `%100 #![forbid(unsafe_code)]`.

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use core::arch::wasm32::{
    i16x8_narrow_i32x4, i32x4_abs, i32x4_add, i32x4_gt, i32x4_le, i32x4_max, i32x4_min, i32x4_mul,
    i32x4_shl, i32x4_shr, i32x4_splat, i32x4_sub, u8x16_narrow_i16x8, u16x8_extend_low_u8x16,
    u32x4_extend_low_u16x8, v128, v128_and, v128_andnot, v128_any_true, v128_bitselect, v128_or,
};

use super::{
    bpp_of, load_low32_u8, load_low64_u8, store_low32_u8, store_low64_u8, transpose_4x4_i32,
};
use crate::dsp::lf::{LimLut, LoopFilterSbFn, lpf_sb_skaler_devret, lpf_sb_sozlesmesi};

#[must_use]
pub const fn loop_filter_sb_fn<const BPC: u32>(plane: usize, dir: usize) -> Option<LoopFilterSbFn> {
    match (plane, dir) {
        (0, 0) => Some(lpf_sb::<BPC, 0, 0>),
        (0, 1) => Some(lpf_sb::<BPC, 0, 1>),
        (1, 0) => Some(lpf_sb::<BPC, 1, 0>),
        (1, 1) => Some(lpf_sb::<BPC, 1, 1>),
        _ => None,
    }
}

#[inline(always)]
fn level_or(level: &[[u8; 4]], at: usize, fallback: usize, idx: usize) -> u32 {
    let l = level[at][idx];
    u32::from(if l != 0 { l } else { level[fallback][idx] })
}

#[inline]
#[target_feature(enable = "simd128")]
fn load_row4<const BPC: u32>(data: &[u8], off: usize) -> v128 {
    if BPC == 8 {
        u32x4_extend_low_u16x8(u16x8_extend_low_u8x16(load_low32_u8(data, off)))
    } else {
        u32x4_extend_low_u16x8(load_low64_u8(data, off))
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn store_row4<const BPC: u32>(data: &mut [u8], off: usize, v: v128) {
    let p16 = i16x8_narrow_i32x4(v, v);
    if BPC == 8 {
        store_low32_u8(data, off, u8x16_narrow_i16x8(p16, p16));
    } else {
        store_low64_u8(data, off, p16);
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn clamp_px<const BPC: u32>(v: v128) -> v128 {
    i32x4_min(i32x4_max(v, i32x4_splat(0)), i32x4_splat((1i32 << BPC) - 1))
}

#[inline]
#[target_feature(enable = "simd128")]
fn abd(a: v128, b: v128) -> v128 {
    i32x4_abs(i32x4_sub(a, b))
}

#[inline]
#[target_feature(enable = "simd128")]
fn narrow_filter_4<const BPC: u32>(
    p1: v128,
    p0: v128,
    q0: v128,
    q1: v128,
    abs_p1p0: v128,
    abs_q1q0: v128,
    vh: v128,
) -> (v128, v128, v128, v128) {
    let bd_min_8 = BPC - 8;
    let clip_lo = i32x4_splat(-128 << bd_min_8);
    let clip_hi = i32x4_splat((128 << bd_min_8) - 1);
    let iclip = |x: v128| i32x4_min(i32x4_max(x, clip_lo), clip_hi);

    let hev = v128_or(i32x4_gt(abs_p1p0, vh), i32x4_gt(abs_q1q0, vh));
    let p1_sub_q1 = iclip(i32x4_sub(p1, q1));
    let hev_term = v128_and(p1_sub_q1, hev);
    let fv = iclip(i32x4_add(
        i32x4_mul(i32x4_sub(q0, p0), i32x4_splat(3)),
        hev_term,
    ));

    let f1 = i32x4_shr(i32x4_min(i32x4_add(fv, i32x4_splat(4)), clip_hi), 3);
    let f2 = i32x4_shr(i32x4_min(i32x4_add(fv, i32x4_splat(3)), clip_hi), 3);

    let np0 = clamp_px::<BPC>(i32x4_add(p0, f2));
    let nq0 = clamp_px::<BPC>(i32x4_sub(q0, f1));

    let fv2 = i32x4_shr(i32x4_add(f1, i32x4_splat(1)), 1);
    let np1 = v128_bitselect(p1, clamp_px::<BPC>(i32x4_add(p1, fv2)), hev);
    let nq1 = v128_bitselect(q1, clamp_px::<BPC>(i32x4_sub(q1, fv2)), hev);
    (np1, np0, nq0, nq1)
}

#[inline]
#[target_feature(enable = "simd128")]
fn filter_6tap_4<const BPC: u32>(
    p2: v128,
    p1: v128,
    p0: v128,
    q0: v128,
    q1: v128,
    q2: v128,
) -> (v128, v128, v128, v128) {
    let c4 = i32x4_splat(4);
    let c3 = i32x4_splat(3);
    let r = |x: v128| clamp_px::<BPC>(i32x4_shr(i32x4_add(x, c4), 3));
    let p0_2 = i32x4_shl(p0, 1);
    let q0_2 = i32x4_shl(q0, 1);
    let p1_2 = i32x4_shl(p1, 1);
    let q1_2 = i32x4_shl(q1, 1);
    let o_p2 = r(i32x4_add(
        i32x4_add(i32x4_mul(p2, c3), p1_2),
        i32x4_add(p0_2, q0),
    ));
    let o_p1 = r(i32x4_add(
        i32x4_add(p2, p1_2),
        i32x4_add(i32x4_add(p0_2, q0_2), q1),
    ));
    let o_q0 = r(i32x4_add(
        i32x4_add(p1, p0_2),
        i32x4_add(i32x4_add(q0_2, q1_2), q2),
    ));
    let o_q1 = r(i32x4_add(
        i32x4_add(p0, q0_2),
        i32x4_add(q1_2, i32x4_mul(q2, c3)),
    ));
    (o_p2, o_p1, o_q0, o_q1)
}

#[inline]
#[target_feature(enable = "simd128")]
fn filter_8tap_4<const BPC: u32>(
    p3: v128,
    p2: v128,
    p1: v128,
    p0: v128,
    q0: v128,
    q1: v128,
    q2: v128,
    q3: v128,
) -> (v128, v128, v128, v128, v128, v128) {
    let c4 = i32x4_splat(4);
    let c3 = i32x4_splat(3);
    let r = |x: v128| clamp_px::<BPC>(i32x4_shr(i32x4_add(x, c4), 3));
    let p1p0q0 = i32x4_add(i32x4_add(p1, p0), q0);
    let o_p3 = r(i32x4_add(
        i32x4_add(i32x4_mul(p3, c3), i32x4_shl(p2, 1)),
        p1p0q0,
    ));
    let o_p2 = r(i32x4_add(
        i32x4_add(i32x4_shl(p3, 1), p2),
        i32x4_add(i32x4_add(p1, p1p0q0), q1),
    ));
    let o_p1 = r(i32x4_add(
        i32x4_add(p3, p2),
        i32x4_add(i32x4_add(p0, p1p0q0), i32x4_add(q1, q2)),
    ));
    let q1q2q3 = i32x4_add(i32x4_add(q1, q2), q3);
    let o_q0 = r(i32x4_add(i32x4_add(p2, p1p0q0), i32x4_add(q0, q1q2q3)));
    let o_q1 = r(i32x4_add(i32x4_add(p1p0q0, q1), i32x4_add(q1q2q3, q3)));
    let o_q2 = r(i32x4_add(
        i32x4_add(i32x4_add(p0, q0), q1),
        i32x4_add(i32x4_shl(q2, 1), i32x4_mul(q3, c3)),
    ));
    (o_p3, o_p2, o_p1, o_q0, o_q1, o_q2)
}

#[inline]
#[target_feature(enable = "simd128")]
fn filter_16tap_4<const BPC: u32>(
    p6: v128,
    p5: v128,
    p4: v128,
    p3: v128,
    p2: v128,
    p1: v128,
    p0: v128,
    q0: v128,
    q1: v128,
    q2: v128,
    q3: v128,
    q4: v128,
    q5: v128,
    q6: v128,
) -> [v128; 12] {
    let c8 = i32x4_splat(8);
    let r = |x: v128| clamp_px::<BPC>(i32x4_shr(i32x4_add(x, c8), 4));
    let s2 = |a: v128, b: v128| i32x4_add(a, b);
    let s3 = |a: v128, b: v128, c: v128| i32x4_add(i32x4_add(a, b), c);
    let m2 = |a: v128| i32x4_shl(a, 1);

    let o0 = r(s3(
        s3(i32x4_mul(p6, i32x4_splat(7)), m2(p5), m2(p4)),
        s3(p3, p2, p1),
        s2(p0, q0),
    ));
    let o1 = r(s3(
        s3(i32x4_mul(p6, i32x4_splat(5)), m2(p5), m2(p4)),
        s3(m2(p3), p2, p1),
        s3(p0, q0, q1),
    ));
    let o2 = r(s3(
        s3(i32x4_mul(p6, i32x4_splat(4)), p5, m2(p4)),
        s3(m2(p3), m2(p2), p1),
        s2(s2(p0, q0), s2(q1, q2)),
    ));
    let o3 = r(s3(
        s3(i32x4_mul(p6, i32x4_splat(3)), p5, p4),
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
        s3(q4, q5, i32x4_mul(q6, i32x4_splat(3))),
    ));
    let o9 = r(s3(
        s2(s2(p2, p1), s2(p0, q0)),
        s3(q1, m2(q2), m2(q3)),
        s3(m2(q4), q5, i32x4_mul(q6, i32x4_splat(4))),
    ));
    let o10 = r(s3(
        s3(p1, p0, q0),
        s3(q1, q2, m2(q3)),
        s3(m2(q4), m2(q5), i32x4_mul(q6, i32x4_splat(5))),
    ));
    let o11 = r(s3(
        s2(p0, q0),
        s3(q1, q2, q3),
        s3(m2(q4), m2(q5), i32x4_mul(q6, i32x4_splat(7))),
    ));
    [o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11]
}

#[inline]
#[target_feature(enable = "simd128")]
fn filter_v_edge<const BPC: u32>(
    data: &mut [u8],
    off: usize,
    stride: usize,
    wd: usize,
    l: u32,
    lut: &LimLut,
) {
    let bd_min_8 = BPC - 8;
    let vf = i32x4_splat(1i32 << bd_min_8);
    let ve = i32x4_splat(lut.e[l as usize] << bd_min_8);
    let vi = i32x4_splat(lut.i[l as usize] << bd_min_8);
    let vh = i32x4_splat(((l >> 4) as i32) << bd_min_8);

    let p1 = load_row4::<BPC>(data, off - 2 * stride);
    let p0 = load_row4::<BPC>(data, off - stride);
    let q0 = load_row4::<BPC>(data, off);
    let q1 = load_row4::<BPC>(data, off + stride);

    let abs_p1p0 = abd(p1, p0);
    let abs_q1q0 = abd(q1, q0);
    let edge_term = i32x4_add(i32x4_shl(abd(p0, q0), 1), i32x4_shr(abd(p1, q1), 1));
    let mut fm = v128_and(
        v128_and(i32x4_le(abs_p1p0, vi), i32x4_le(abs_q1q0, vi)),
        i32x4_le(edge_term, ve),
    );

    if wd == 4 {
        if !v128_any_true(fm) {
            return;
        }
        let (np1, np0, nq0, nq1) = narrow_filter_4::<BPC>(p1, p0, q0, q1, abs_p1p0, abs_q1q0, vh);
        store_row4::<BPC>(data, off - 2 * stride, v128_bitselect(np1, p1, fm));
        store_row4::<BPC>(data, off - stride, v128_bitselect(np0, p0, fm));
        store_row4::<BPC>(data, off, v128_bitselect(nq0, q0, fm));
        store_row4::<BPC>(data, off + stride, v128_bitselect(nq1, q1, fm));
        return;
    }

    let p2 = load_row4::<BPC>(data, off - 3 * stride);
    let q2 = load_row4::<BPC>(data, off + 2 * stride);
    fm = v128_and(
        fm,
        v128_and(i32x4_le(abd(p2, p1), vi), i32x4_le(abd(q2, q1), vi)),
    );

    if wd == 6 {
        if !v128_any_true(fm) {
            return;
        }
        let flat8in = v128_and(
            v128_and(i32x4_le(abd(p2, p0), vf), i32x4_le(abs_p1p0, vf)),
            v128_and(i32x4_le(abs_q1q0, vf), i32x4_le(abd(q2, q0), vf)),
        );
        let (np1, np0, nq0, nq1) = narrow_filter_4::<BPC>(p1, p0, q0, q1, abs_p1p0, abs_q1q0, vh);
        let (fp2, fp1, fq0, fq1) = filter_6tap_4::<BPC>(p2, p1, p0, q0, q1, q2);
        let m6 = v128_and(fm, flat8in);
        store_row4::<BPC>(
            data,
            off - 2 * stride,
            v128_bitselect(fp2, v128_bitselect(np1, p1, fm), m6),
        );
        store_row4::<BPC>(
            data,
            off - stride,
            v128_bitselect(fp1, v128_bitselect(np0, p0, fm), m6),
        );
        store_row4::<BPC>(
            data,
            off,
            v128_bitselect(fq0, v128_bitselect(nq0, q0, fm), m6),
        );
        store_row4::<BPC>(
            data,
            off + stride,
            v128_bitselect(fq1, v128_bitselect(nq1, q1, fm), m6),
        );
        return;
    }

    let p3 = load_row4::<BPC>(data, off - 4 * stride);
    let q3 = load_row4::<BPC>(data, off + 3 * stride);
    fm = v128_and(
        fm,
        v128_and(i32x4_le(abd(p3, p2), vi), i32x4_le(abd(q3, q2), vi)),
    );
    if !v128_any_true(fm) {
        return;
    }

    let flat8in = v128_and(
        v128_and(
            v128_and(i32x4_le(abd(p2, p0), vf), i32x4_le(abs_p1p0, vf)),
            v128_and(i32x4_le(abs_q1q0, vf), i32x4_le(abd(q2, q0), vf)),
        ),
        v128_and(i32x4_le(abd(p3, p0), vf), i32x4_le(abd(q3, q0), vf)),
    );

    let (np1, np0, nq0, nq1) = narrow_filter_4::<BPC>(p1, p0, q0, q1, abs_p1p0, abs_q1q0, vh);
    let (fp3, fp2, fp1, fq0, fq1, fq2) = filter_8tap_4::<BPC>(p3, p2, p1, p0, q0, q1, q2, q3);

    if wd == 8 {
        let m8 = v128_and(fm, flat8in);
        store_row4::<BPC>(data, off - 3 * stride, v128_bitselect(fp3, p2, m8));
        store_row4::<BPC>(
            data,
            off - 2 * stride,
            v128_bitselect(fp2, v128_bitselect(np1, p1, fm), m8),
        );
        store_row4::<BPC>(
            data,
            off - stride,
            v128_bitselect(fp1, v128_bitselect(np0, p0, fm), m8),
        );
        store_row4::<BPC>(
            data,
            off,
            v128_bitselect(fq0, v128_bitselect(nq0, q0, fm), m8),
        );
        store_row4::<BPC>(
            data,
            off + stride,
            v128_bitselect(fq1, v128_bitselect(nq1, q1, fm), m8),
        );
        store_row4::<BPC>(data, off + 2 * stride, v128_bitselect(fq2, q2, m8));
        return;
    }

    let p6 = load_row4::<BPC>(data, off - 7 * stride);
    let p5 = load_row4::<BPC>(data, off - 6 * stride);
    let p4 = load_row4::<BPC>(data, off - 5 * stride);
    let q4 = load_row4::<BPC>(data, off + 4 * stride);
    let q5 = load_row4::<BPC>(data, off + 5 * stride);
    let q6 = load_row4::<BPC>(data, off + 6 * stride);

    let flat8out = v128_and(
        v128_and(
            v128_and(i32x4_le(abd(p6, p0), vf), i32x4_le(abd(p5, p0), vf)),
            i32x4_le(abd(p4, p0), vf),
        ),
        v128_and(
            v128_and(i32x4_le(abd(q4, q0), vf), i32x4_le(abd(q5, q0), vf)),
            i32x4_le(abd(q6, q0), vf),
        ),
    );
    let m16 = v128_and(fm, v128_and(flat8in, flat8out));
    let m8 = v128_and(fm, v128_andnot(flat8in, flat8out));

    if !v128_any_true(m16) {
        store_row4::<BPC>(data, off - 3 * stride, v128_bitselect(fp3, p2, m8));
        store_row4::<BPC>(
            data,
            off - 2 * stride,
            v128_bitselect(fp2, v128_bitselect(np1, p1, fm), m8),
        );
        store_row4::<BPC>(
            data,
            off - stride,
            v128_bitselect(fp1, v128_bitselect(np0, p0, fm), m8),
        );
        store_row4::<BPC>(
            data,
            off,
            v128_bitselect(fq0, v128_bitselect(nq0, q0, fm), m8),
        );
        store_row4::<BPC>(
            data,
            off + stride,
            v128_bitselect(fq1, v128_bitselect(nq1, q1, fm), m8),
        );
        store_row4::<BPC>(data, off + 2 * stride, v128_bitselect(fq2, q2, m8));
        return;
    }

    let o16 = filter_16tap_4::<BPC>(p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6);
    store_row4::<BPC>(data, off - 6 * stride, v128_bitselect(o16[0], p5, m16));
    store_row4::<BPC>(data, off - 5 * stride, v128_bitselect(o16[1], p4, m16));
    store_row4::<BPC>(data, off - 4 * stride, v128_bitselect(o16[2], p3, m16));
    store_row4::<BPC>(
        data,
        off - 3 * stride,
        v128_bitselect(o16[3], v128_bitselect(fp3, p2, m8), m16),
    );
    store_row4::<BPC>(
        data,
        off - 2 * stride,
        v128_bitselect(
            o16[4],
            v128_bitselect(fp2, v128_bitselect(np1, p1, fm), m8),
            m16,
        ),
    );
    store_row4::<BPC>(
        data,
        off - stride,
        v128_bitselect(
            o16[5],
            v128_bitselect(fp1, v128_bitselect(np0, p0, fm), m8),
            m16,
        ),
    );
    store_row4::<BPC>(
        data,
        off,
        v128_bitselect(
            o16[6],
            v128_bitselect(fq0, v128_bitselect(nq0, q0, fm), m8),
            m16,
        ),
    );
    store_row4::<BPC>(
        data,
        off + stride,
        v128_bitselect(
            o16[7],
            v128_bitselect(fq1, v128_bitselect(nq1, q1, fm), m8),
            m16,
        ),
    );
    store_row4::<BPC>(
        data,
        off + 2 * stride,
        v128_bitselect(o16[8], v128_bitselect(fq2, q2, m8), m16),
    );
    store_row4::<BPC>(data, off + 3 * stride, v128_bitselect(o16[9], q3, m16));
    store_row4::<BPC>(data, off + 4 * stride, v128_bitselect(o16[10], q4, m16));
    store_row4::<BPC>(data, off + 5 * stride, v128_bitselect(o16[11], q5, m16));
}

#[inline]
#[target_feature(enable = "simd128")]
fn load_h_4x4<const BPC: u32>(data: &[u8], off: usize, stride: usize) -> [v128; 4] {
    let r0 = load_row4::<BPC>(data, off);
    let r1 = load_row4::<BPC>(data, off + stride);
    let r2 = load_row4::<BPC>(data, off + 2 * stride);
    let r3 = load_row4::<BPC>(data, off + 3 * stride);
    transpose_4x4_i32(r0, r1, r2, r3)
}

#[inline]
#[target_feature(enable = "simd128")]
fn store_h_4x4<const BPC: u32>(data: &mut [u8], off: usize, stride: usize, cols: [v128; 4]) {
    let [r0, r1, r2, r3] = transpose_4x4_i32(cols[0], cols[1], cols[2], cols[3]);
    store_row4::<BPC>(data, off, r0);
    store_row4::<BPC>(data, off + stride, r1);
    store_row4::<BPC>(data, off + 2 * stride, r2);
    store_row4::<BPC>(data, off + 3 * stride, r3);
}

#[inline]
#[target_feature(enable = "simd128")]
fn filter_h_edge<const BPC: u32>(
    data: &mut [u8],
    off: usize,
    stride: usize,
    wd: usize,
    l: u32,
    lut: &LimLut,
) {
    let bpp = bpp_of(BPC);
    let bd_min_8 = BPC - 8;
    let vf = i32x4_splat(1i32 << bd_min_8);
    let ve = i32x4_splat(lut.e[l as usize] << bd_min_8);
    let vi = i32x4_splat(lut.i[l as usize] << bd_min_8);
    let vh = i32x4_splat(((l >> 4) as i32) << bd_min_8);

    if wd == 4 {
        let [p1, p0, q0, q1] = load_h_4x4::<BPC>(data, off - 2 * bpp, stride);
        let abs_p1p0 = abd(p1, p0);
        let abs_q1q0 = abd(q1, q0);
        let edge_term = i32x4_add(i32x4_shl(abd(p0, q0), 1), i32x4_shr(abd(p1, q1), 1));
        let fm = v128_and(
            v128_and(i32x4_le(abs_p1p0, vi), i32x4_le(abs_q1q0, vi)),
            i32x4_le(edge_term, ve),
        );
        if !v128_any_true(fm) {
            return;
        }
        let (np1, np0, nq0, nq1) = narrow_filter_4::<BPC>(p1, p0, q0, q1, abs_p1p0, abs_q1q0, vh);
        store_h_4x4::<BPC>(
            data,
            off - 2 * bpp,
            stride,
            [
                v128_bitselect(np1, p1, fm),
                v128_bitselect(np0, p0, fm),
                v128_bitselect(nq0, q0, fm),
                v128_bitselect(nq1, q1, fm),
            ],
        );
        return;
    }

    if wd == 6 {
        let [p2, p1, p0, q0] = load_h_4x4::<BPC>(data, off - 3 * bpp, stride);
        let [_, _, q1, q2] = load_h_4x4::<BPC>(data, off - bpp, stride);
        let abs_p1p0 = abd(p1, p0);
        let abs_q1q0 = abd(q1, q0);
        let edge_term = i32x4_add(i32x4_shl(abd(p0, q0), 1), i32x4_shr(abd(p1, q1), 1));
        let fm = v128_and(
            v128_and(
                v128_and(i32x4_le(abs_p1p0, vi), i32x4_le(abs_q1q0, vi)),
                i32x4_le(edge_term, ve),
            ),
            v128_and(i32x4_le(abd(p2, p1), vi), i32x4_le(abd(q2, q1), vi)),
        );
        if !v128_any_true(fm) {
            return;
        }
        let flat8in = v128_and(
            v128_and(i32x4_le(abd(p2, p0), vf), i32x4_le(abs_p1p0, vf)),
            v128_and(i32x4_le(abs_q1q0, vf), i32x4_le(abd(q2, q0), vf)),
        );
        let (np1, np0, nq0, nq1) = narrow_filter_4::<BPC>(p1, p0, q0, q1, abs_p1p0, abs_q1q0, vh);
        let (fp2, fp1, fq0, fq1) = filter_6tap_4::<BPC>(p2, p1, p0, q0, q1, q2);
        let m6 = v128_and(fm, flat8in);
        store_h_4x4::<BPC>(
            data,
            off - 2 * bpp,
            stride,
            [
                v128_bitselect(fp2, v128_bitselect(np1, p1, fm), m6),
                v128_bitselect(fp1, v128_bitselect(np0, p0, fm), m6),
                v128_bitselect(fq0, v128_bitselect(nq0, q0, fm), m6),
                v128_bitselect(fq1, v128_bitselect(nq1, q1, fm), m6),
            ],
        );
        return;
    }

    let [p3, p2, p1, p0] = load_h_4x4::<BPC>(data, off - 4 * bpp, stride);
    let [q0, q1, q2, q3] = load_h_4x4::<BPC>(data, off, stride);

    let abs_p1p0 = abd(p1, p0);
    let abs_q1q0 = abd(q1, q0);
    let edge_term = i32x4_add(i32x4_shl(abd(p0, q0), 1), i32x4_shr(abd(p1, q1), 1));
    let fm = v128_and(
        v128_and(
            v128_and(i32x4_le(abs_p1p0, vi), i32x4_le(abs_q1q0, vi)),
            i32x4_le(edge_term, ve),
        ),
        v128_and(
            v128_and(i32x4_le(abd(p2, p1), vi), i32x4_le(abd(q2, q1), vi)),
            v128_and(i32x4_le(abd(p3, p2), vi), i32x4_le(abd(q3, q2), vi)),
        ),
    );
    if !v128_any_true(fm) {
        return;
    }

    let flat8in = v128_and(
        v128_and(
            v128_and(i32x4_le(abd(p2, p0), vf), i32x4_le(abs_p1p0, vf)),
            v128_and(i32x4_le(abs_q1q0, vf), i32x4_le(abd(q2, q0), vf)),
        ),
        v128_and(i32x4_le(abd(p3, p0), vf), i32x4_le(abd(q3, q0), vf)),
    );
    let (np1, np0, nq0, nq1) = narrow_filter_4::<BPC>(p1, p0, q0, q1, abs_p1p0, abs_q1q0, vh);
    let (fp3, fp2, fp1, fq0, fq1, fq2) = filter_8tap_4::<BPC>(p3, p2, p1, p0, q0, q1, q2, q3);

    if wd == 8 {
        let m8 = v128_and(fm, flat8in);
        let op2 = v128_bitselect(fp3, p2, m8);
        let op1 = v128_bitselect(fp2, v128_bitselect(np1, p1, fm), m8);
        let op0 = v128_bitselect(fp1, v128_bitselect(np0, p0, fm), m8);
        let oq0 = v128_bitselect(fq0, v128_bitselect(nq0, q0, fm), m8);
        let oq1 = v128_bitselect(fq1, v128_bitselect(nq1, q1, fm), m8);
        let oq2 = v128_bitselect(fq2, q2, m8);
        store_h_4x4::<BPC>(data, off - 4 * bpp, stride, [p3, op2, op1, op0]);
        store_h_4x4::<BPC>(data, off, stride, [oq0, oq1, oq2, q3]);
        return;
    }

    let [p6, p5, p4, _] = load_h_4x4::<BPC>(data, off - 7 * bpp, stride);
    let [_, q4, q5, q6] = load_h_4x4::<BPC>(data, off + 3 * bpp, stride);

    let flat8out = v128_and(
        v128_and(
            v128_and(i32x4_le(abd(p6, p0), vf), i32x4_le(abd(p5, p0), vf)),
            i32x4_le(abd(p4, p0), vf),
        ),
        v128_and(
            v128_and(i32x4_le(abd(q4, q0), vf), i32x4_le(abd(q5, q0), vf)),
            i32x4_le(abd(q6, q0), vf),
        ),
    );
    let m16 = v128_and(fm, v128_and(flat8in, flat8out));
    let m8 = v128_and(fm, v128_andnot(flat8in, flat8out));

    if !v128_any_true(m16) {
        let op2 = v128_bitselect(fp3, p2, m8);
        let op1 = v128_bitselect(fp2, v128_bitselect(np1, p1, fm), m8);
        let op0 = v128_bitselect(fp1, v128_bitselect(np0, p0, fm), m8);
        let oq0 = v128_bitselect(fq0, v128_bitselect(nq0, q0, fm), m8);
        let oq1 = v128_bitselect(fq1, v128_bitselect(nq1, q1, fm), m8);
        let oq2 = v128_bitselect(fq2, q2, m8);
        store_h_4x4::<BPC>(data, off - 4 * bpp, stride, [p3, op2, op1, op0]);
        store_h_4x4::<BPC>(data, off, stride, [oq0, oq1, oq2, q3]);
        return;
    }

    let o16 = filter_16tap_4::<BPC>(p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6);
    let op5 = v128_bitselect(o16[0], p5, m16);
    let op4 = v128_bitselect(o16[1], p4, m16);
    let op3 = v128_bitselect(o16[2], p3, m16);
    let op2 = v128_bitselect(o16[3], v128_bitselect(fp3, p2, m8), m16);
    let op1 = v128_bitselect(
        o16[4],
        v128_bitselect(fp2, v128_bitselect(np1, p1, fm), m8),
        m16,
    );
    let op0 = v128_bitselect(
        o16[5],
        v128_bitselect(fp1, v128_bitselect(np0, p0, fm), m8),
        m16,
    );
    let oq0 = v128_bitselect(
        o16[6],
        v128_bitselect(fq0, v128_bitselect(nq0, q0, fm), m8),
        m16,
    );
    let oq1 = v128_bitselect(
        o16[7],
        v128_bitselect(fq1, v128_bitselect(nq1, q1, fm), m8),
        m16,
    );
    let oq2 = v128_bitselect(o16[8], v128_bitselect(fq2, q2, m8), m16);
    let oq3 = v128_bitselect(o16[9], q3, m16);
    let oq4 = v128_bitselect(o16[10], q4, m16);
    let oq5 = v128_bitselect(o16[11], q5, m16);

    store_h_4x4::<BPC>(data, off - 6 * bpp, stride, [op5, op4, op3, op2]);
    store_h_4x4::<BPC>(data, off - 2 * bpp, stride, [op1, op0, oq0, oq1]);
    store_h_4x4::<BPC>(data, off + 2 * bpp, stride, [oq2, oq3, oq4, oq5]);
}

#[target_feature(enable = "simd128")]
fn lpf_sb_simd128<const BPC: u32, const PLANE: usize, const DIR: usize>(
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
    let bpp = bpp_of(BPC);
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

pub fn lpf_sb<const BPC: u32, const PLANE: usize, const DIR: usize>(
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
    if !lpf_sb_sozlesmesi::<BPC>(
        PLANE,
        DIR,
        dst.len(),
        stride,
        mask.len(),
        mask_stride,
        level.len(),
        lvl_stride,
        lvl_idx,
        w4,
        y0,
        count,
    ) {
        return lpf_sb_skaler_devret::<BPC>(
            PLANE,
            DIR,
            dst,
            stride,
            mask,
            mask_stride,
            level,
            lvl_stride,
            lvl_idx,
            lut,
            w4,
            y0,
            count,
        );
    }
    lpf_sb_simd128::<BPC, PLANE, DIR>(
        dst,
        stride,
        mask,
        mask_stride,
        level,
        lvl_stride,
        lvl_idx,
        lut,
        w4,
        y0,
        count,
    );
}
