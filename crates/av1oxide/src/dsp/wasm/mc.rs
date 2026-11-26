//! WebAssembly `simd128` hareket telafisi (MC) çekirdekleri (`96` yuva, `8 / 10 / 12` bpc).
//!
//! Kapsam: `mc[0..9]`, `mct[0..9]` (8-tap + bilinear), `avg`, `w_avg`, `mask`,
//! `w_mask[0..2]`, `blend`, `blend_v`, `blend_h`, `warp8x8`, `warp8x8t`, `emu_edge`.
//! `%100 #![forbid(unsafe_code)]`.

#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use core::arch::wasm32::{
    i8x16_shuffle, i16x8, i16x8_add, i16x8_extract_lane, i16x8_max, i16x8_min, i16x8_mul,
    i16x8_narrow_i32x4, i16x8_shl, i16x8_shr, i16x8_splat, i16x8_sub, i32x4_abs, i32x4_add,
    i32x4_dot_i16x8, i32x4_extend_high_i16x8, i32x4_extend_low_i16x8, i32x4_extmul_high_i16x8,
    i32x4_extmul_low_i16x8, i32x4_min, i32x4_mul, i32x4_shl, i32x4_shr, i32x4_splat, i32x4_sub,
    i64x2_extract_lane, u8x16_narrow_i16x8, u8x16_splat, u16x8_extend_high_u8x16,
    u16x8_extend_low_u8x16, u16x8_min, u16x8_shr, u16x8_splat, u32x4_extadd_pairwise_u16x8,
    u32x4_extmul_high_u16x8, u32x4_extmul_low_u16x8, u32x4_shr, v128,
};

use super::{
    bpp_of, hsum_i32x4, intermediate_bits_of, load_i16x4, load_i16x8, load_low32_u8, load_low64_u8,
    load_pixels_4, load_pixels_8, load_scalar, load_v128_u8, pack_and_store_4, pack_and_store_8,
    prep_bias_of, store_i16x8, store_low32_u8, store_low64_u8, store_scalar, store_v128_u8,
};
use crate::dsp::mc::{
    Filter2d, McDsp, McPrepFn, McPutFn, bilesik_sozlesmesi, bilesik16_sozlesmesi, kenar_sozlesmesi,
    kenar16_sozlesmesi, simd_sozlesmesi, simd16_sozlesmesi, skaler_devri_say, warp_sozlesmesi,
    warp16_sozlesmesi,
};
use crate::dsp::mc_tables::{MC_SUBPEL_FILTERS, MC_WARP_FILTER, OBMC_MASKS};
use crate::frame::BitDepth;

const BILIN: usize = 9;

#[inline(always)]
const fn filtre(fi: usize) -> Filter2d {
    match Filter2d::from_index(fi) {
        Some(f) => f,
        None => Filter2d::Bilinear,
    }
}

#[inline(always)]
fn bd_uyumlu<const BPC: u32>(bd: BitDepth, bdmax: u16) -> bool {
    bd.bits() == BPC && u32::from(bdmax) == (1u32 << BPC) - 1
}

#[inline]
fn katsayi(tablo: usize, boyut: usize, m: u8) -> [i16; 8] {
    let t = if boyut > 4 { tablo } else { 3 + (tablo & 1) };
    MC_SUBPEL_FILTERS[t][usize::from(m) - 1].map(i16::from)
}

#[inline]
#[target_feature(enable = "simd128")]
fn ext_i16<const K: usize>(a: v128, b: v128) -> v128 {
    match K {
        0 => a,
        1 => i8x16_shuffle::<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17>(a, b),
        2 => i8x16_shuffle::<4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19>(a, b),
        3 => i8x16_shuffle::<6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21>(a, b),
        4 => i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23>(a, b),
        5 => i8x16_shuffle::<10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25>(a, b),
        6 => i8x16_shuffle::<12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27>(a, b),
        7 => i8x16_shuffle::<14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29>(a, b),
        _ => b,
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn yaz_u8(dst: &mut [u8], off: usize, v: v128, n: usize) {
    if n >= 8 {
        store_low64_u8(dst, off, v);
    } else if n == 4 {
        store_low32_u8(dst, off, v);
    } else if n == 2 {
        let w = i64x2_extract_lane::<0>(v) as u16;
        dst[off..off + 2].copy_from_slice(&w.to_le_bytes());
    } else {
        let bytes = i64x2_extract_lane::<0>(v).to_le_bytes();
        dst[off..off + n].copy_from_slice(&bytes[..n]);
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn yaz_u16(dst: &mut [u8], off: usize, v: v128, n: usize) {
    if n >= 8 {
        store_v128_u8(dst, off, v);
    } else if n == 4 {
        store_low64_u8(dst, off, v);
    } else if n == 2 {
        store_low32_u8(dst, off, v);
    } else {
        let mut t = [0u8; 16];
        store_v128_u8(&mut t, 0, v);
        dst[off..off + 2 * n].copy_from_slice(&t[..2 * n]);
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn yaz_i16(tmp: &mut [i16], off: usize, v: v128, n: usize) {
    if n >= 8 {
        store_i16x8(tmp, off, v);
    } else if n == 4 {
        let d: &mut [i16; 4] = (&mut tmp[off..off + 4]).try_into().unwrap();
        d[0] = i16x8_extract_lane::<0>(v);
        d[1] = i16x8_extract_lane::<1>(v);
        d[2] = i16x8_extract_lane::<2>(v);
        d[3] = i16x8_extract_lane::<3>(v);
    } else if n == 2 {
        tmp[off] = i16x8_extract_lane::<0>(v);
        tmp[off + 1] = i16x8_extract_lane::<1>(v);
    } else {
        let mut t = [0i16; 8];
        store_i16x8(&mut t, 0, v);
        tmp[off..off + n].copy_from_slice(&t[..n]);
    }
}

#[inline]
fn kopya(dst: &mut [u8], ds: usize, src: &[u8], ss: usize, w: usize, h: usize, bpp: usize) {
    let n = w * bpp;
    for y in 0..h {
        let s = (y + 3) * ss + 3 * bpp;
        let d = y * ds;
        dst[d..d + n].copy_from_slice(&src[s..s + n]);
    }
}

// ============================================================================
// 1. 8-tap ve bilinear `mc[0..9]` / `mct[0..9]` (60 yuva)
// ============================================================================

#[inline]
#[target_feature(enable = "simd128")]
fn yatay8(src: &[u8], off: usize, f: &[v128; 8]) -> v128 {
    let v = load_v128_u8(src, off);
    let a = u16x8_extend_low_u8x16(v);
    let b = u16x8_extend_high_u8x16(v);
    let mut t = i16x8_mul(a, f[0]);
    t = i16x8_add(t, i16x8_mul(ext_i16::<1>(a, b), f[1]));
    t = i16x8_add(t, i16x8_mul(ext_i16::<2>(a, b), f[2]));
    t = i16x8_add(t, i16x8_mul(ext_i16::<3>(a, b), f[3]));
    t = i16x8_add(t, i16x8_mul(ext_i16::<4>(a, b), f[4]));
    t = i16x8_add(t, i16x8_mul(ext_i16::<5>(a, b), f[5]));
    t = i16x8_add(t, i16x8_mul(ext_i16::<6>(a, b), f[6]));
    i16x8_add(t, i16x8_mul(ext_i16::<7>(a, b), f[7]))
}

#[inline]
#[target_feature(enable = "simd128")]
fn yatay16(src: &[u8], off: usize, f: &[v128; 8]) -> (v128, v128) {
    let a = load_v128_u8(src, off);
    let b = load_v128_u8(src, off + 16);
    let mut lo = i32x4_extmul_low_i16x8(a, f[0]);
    let mut hi = i32x4_extmul_high_i16x8(a, f[0]);
    macro_rules! tap {
        ($k:literal) => {{
            let e = ext_i16::<$k>(a, b);
            lo = i32x4_add(lo, i32x4_extmul_low_i16x8(e, f[$k]));
            hi = i32x4_add(hi, i32x4_extmul_high_i16x8(e, f[$k]));
        }};
    }
    tap!(1);
    tap!(2);
    tap!(3);
    tap!(4);
    tap!(5);
    tap!(6);
    tap!(7);
    (lo, hi)
}

#[inline]
#[target_feature(enable = "simd128")]
fn dikey_i16(p: &[v128; 8], f: &[v128; 8]) -> v128 {
    let mut t = i16x8_mul(p[0], f[0]);
    for k in 1..8 {
        t = i16x8_add(t, i16x8_mul(p[k], f[k]));
    }
    t
}

#[inline]
#[target_feature(enable = "simd128")]
fn dikey_i32(p: &[v128; 8], f: &[v128; 8]) -> (v128, v128) {
    let mut lo = i32x4_extmul_low_i16x8(p[0], f[0]);
    let mut hi = i32x4_extmul_high_i16x8(p[0], f[0]);
    for k in 1..8 {
        lo = i32x4_add(lo, i32x4_extmul_low_i16x8(p[k], f[k]));
        hi = i32x4_add(hi, i32x4_extmul_high_i16x8(p[k], f[k]));
    }
    (lo, hi)
}

#[inline]
#[target_feature(enable = "simd128")]
fn splat_taps(f: &[i16; 8]) -> [v128; 8] {
    [
        i16x8_splat(f[0]),
        i16x8_splat(f[1]),
        i16x8_splat(f[2]),
        i16x8_splat(f[3]),
        i16x8_splat(f[4]),
        i16x8_splat(f[5]),
        i16x8_splat(f[6]),
        i16x8_splat(f[7]),
    ]
}

#[inline]
#[target_feature(enable = "simd128")]
fn pack_i32_to_i16(lo: v128, hi: v128) -> v128 {
    i16x8_narrow_i32x4(lo, hi)
}

#[inline]
#[target_feature(enable = "simd128")]
fn clip_u8_from_i16(v: v128) -> v128 {
    u8x16_narrow_i16x8(v, v)
}

#[inline]
#[target_feature(enable = "simd128")]
fn clip_u16_from_i32(lo: v128, hi: v128, v_max: v128) -> v128 {
    i16x8_min(i16x8_max(i16x8_narrow_i32x4(lo, hi), i16x8_splat(0)), v_max)
}

#[target_feature(enable = "simd128")]
fn sekiz_tap8<const PREP: bool>(
    dst: &mut [u8],
    ds: usize,
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    filt: Filter2d,
) {
    if mx == 0 {
        if my == 0 {
            if PREP {
                for y in 0..h {
                    let so = (y + 3) * ss + 3;
                    let mut x = 0;
                    while x < w {
                        let p = i16x8_shl(u16x8_extend_low_u8x16(load_low64_u8(src, so + x)), 4);
                        yaz_i16(tmp, y * w + x, p, w - x);
                        x += 8;
                    }
                }
            } else {
                kopya(dst, ds, src, ss, w, h, 1);
            }
            return;
        }
        let fv = splat_taps(&katsayi(filt.vertical_table(), h, my));
        let v_rnd2 = i16x8_splat(2);
        let v_rnd32 = i16x8_splat(32);
        let mut x = 0;
        while x < w {
            let mut p = [i16x8_splat(0); 8];
            for k in 0..7 {
                p[k] = u16x8_extend_low_u8x16(load_low64_u8(src, k * ss + x + 3));
            }
            for y in 0..h {
                p[7] = u16x8_extend_low_u8x16(load_low64_u8(src, (y + 7) * ss + x + 3));
                let t = dikey_i16(&p, &fv);
                if PREP {
                    yaz_i16(tmp, y * w + x, i16x8_shr(i16x8_add(t, v_rnd2), 2), w - x);
                } else {
                    let r = clip_u8_from_i16(i16x8_shr(i16x8_add(t, v_rnd32), 6));
                    yaz_u8(dst, y * ds + x, r, w - x);
                }
                p.copy_within(1.., 0);
            }
            x += 8;
        }
        return;
    }

    let fh = splat_taps(&katsayi(filt.horizontal_table(), w, mx));
    let v_rnd2 = i16x8_splat(2);
    if my == 0 {
        let v_rnd34 = i16x8_splat(34);
        for y in 0..h {
            let so = (y + 3) * ss;
            let mut x = 0;
            while x < w {
                let t = yatay8(src, so + x, &fh);
                if PREP {
                    yaz_i16(tmp, y * w + x, i16x8_shr(i16x8_add(t, v_rnd2), 2), w - x);
                } else {
                    let r = clip_u8_from_i16(i16x8_shr(i16x8_add(t, v_rnd34), 6));
                    yaz_u8(dst, y * ds + x, r, w - x);
                }
                x += 8;
            }
        }
        return;
    }

    let fv = splat_taps(&katsayi(filt.vertical_table(), h, my));
    let v_rnd32 = i32x4_splat(32);
    let v_rnd512 = i32x4_splat(512);
    let mut x = 0;
    while x < w {
        let mut p = [i16x8_splat(0); 8];
        for k in 0..7 {
            p[k] = i16x8_shr(i16x8_add(yatay8(src, k * ss + x, &fh), v_rnd2), 2);
        }
        for y in 0..h {
            p[7] = i16x8_shr(i16x8_add(yatay8(src, (y + 7) * ss + x, &fh), v_rnd2), 2);
            let (lo, hi) = dikey_i32(&p, &fv);
            if PREP {
                let lo = i32x4_shr(i32x4_add(lo, v_rnd32), 6);
                let hi = i32x4_shr(i32x4_add(hi, v_rnd32), 6);
                yaz_i16(tmp, y * w + x, pack_i32_to_i16(lo, hi), w - x);
            } else {
                let lo = i32x4_shr(i32x4_add(lo, v_rnd512), 10);
                let hi = i32x4_shr(i32x4_add(hi, v_rnd512), 10);
                let p16 = i16x8_narrow_i32x4(lo, hi);
                yaz_u8(dst, y * ds + x, clip_u8_from_i16(p16), w - x);
            }
            p.copy_within(1.., 0);
        }
        x += 8;
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn bilin_yatay8(src: &[u8], off: usize, va: v128, vb: v128) -> v128 {
    let v = load_v128_u8(src, off);
    let a = u16x8_extend_low_u8x16(v);
    let b = u16x8_extend_low_u8x16(i8x16_shuffle::<
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    >(v, v));
    i16x8_add(i16x8_mul(a, va), i16x8_mul(b, vb))
}

#[target_feature(enable = "simd128")]
fn bilin8<const PREP: bool>(
    dst: &mut [u8],
    ds: usize,
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    match (mx != 0, my != 0) {
        (false, false) => {
            sekiz_tap8::<PREP>(dst, ds, tmp, src, ss, w, h, 0, 0, Filter2d::Regular);
        }
        (true, false) => {
            let va = i16x8_splat(16 - i16::from(mx));
            let vb = i16x8_splat(i16::from(mx));
            let v_rnd8 = i16x8_splat(8);
            for y in 0..h {
                let so = (y + 3) * ss + 3;
                let mut x = 0;
                while x < w {
                    let v = bilin_yatay8(src, so + x, va, vb);
                    if PREP {
                        yaz_i16(tmp, y * w + x, v, w - x);
                    } else {
                        yaz_u8(
                            dst,
                            y * ds + x,
                            clip_u8_from_i16(u16x8_shr(i16x8_add(v, v_rnd8), 4)),
                            w - x,
                        );
                    }
                    x += 8;
                }
            }
        }
        (false, true) => {
            let va = i16x8_splat(16 - i16::from(my));
            let vb = i16x8_splat(i16::from(my));
            let v_rnd8 = i16x8_splat(8);
            for y in 0..h {
                let so = (y + 3) * ss + 3;
                let mut x = 0;
                while x < w {
                    let a = u16x8_extend_low_u8x16(load_low64_u8(src, so + x));
                    let b = u16x8_extend_low_u8x16(load_low64_u8(src, so + ss + x));
                    let v = i16x8_add(i16x8_mul(a, va), i16x8_mul(b, vb));
                    if PREP {
                        yaz_i16(tmp, y * w + x, v, w - x);
                    } else {
                        yaz_u8(
                            dst,
                            y * ds + x,
                            clip_u8_from_i16(u16x8_shr(i16x8_add(v, v_rnd8), 4)),
                            w - x,
                        );
                    }
                    x += 8;
                }
            }
        }
        (true, true) => {
            let h_a = i16x8_splat(16 - i16::from(mx));
            let h_b = i16x8_splat(i16::from(mx));
            let v_a = i16x8_splat(16 - i16::from(my));
            let v_b = i16x8_splat(i16::from(my));
            let v_rnd8 = i16x8_splat(8);
            let v_rnd128 = i16x8_splat(128);
            let mut x = 0;
            while x < w {
                let mut m0 = bilin_yatay8(src, 3 * ss + 3 + x, h_a, h_b);
                for y in 0..h {
                    let m1 = bilin_yatay8(src, (y + 4) * ss + 3 + x, h_a, h_b);
                    let v = i16x8_add(i16x8_mul(m0, v_a), i16x8_mul(m1, v_b));
                    if PREP {
                        yaz_i16(tmp, y * w + x, u16x8_shr(i16x8_add(v, v_rnd8), 4), w - x);
                    } else {
                        yaz_u8(
                            dst,
                            y * ds + x,
                            clip_u8_from_i16(u16x8_shr(i16x8_add(v, v_rnd128), 8)),
                            w - x,
                        );
                    }
                    m0 = m1;
                }
                x += 8;
            }
        }
    }
}

#[target_feature(enable = "simd128")]
fn sekiz_tap16<const BPC: u32, const PREP: bool>(
    dst: &mut [u8],
    ds: usize,
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    filt: Filter2d,
) {
    let sh = BPC - 8;
    let ib = 14 - BPC;
    let rnd_mid = i32x4_splat((1i32 << sh) >> 1);
    let v_max = i16x8_splat(((1u32 << BPC) - 1) as i16);
    let bias = i32x4_splat(8192);
    match (mx != 0, my != 0) {
        (false, false) => {
            if PREP {
                let b16 = i16x8_splat(8192);
                for y in 0..h {
                    let so = (y + 3) * ss + 6;
                    let mut x = 0;
                    while x < w {
                        let p = load_v128_u8(src, so + 2 * x);
                        yaz_i16(tmp, y * w + x, i16x8_sub(i16x8_shl(p, ib), b16), w - x);
                        x += 8;
                    }
                }
            } else {
                kopya(dst, ds, src, ss, w, h, 2);
            }
        }
        (true, false) => {
            let fh = splat_taps(&katsayi(filt.horizontal_table(), w, mx));
            let c = i32x4_splat(32 + ((1i32 << sh) >> 1));
            for y in 0..h {
                let so = (y + 3) * ss;
                let mut x = 0;
                while x < w {
                    let (lo, hi) = yatay16(src, so + 2 * x, &fh);
                    if PREP {
                        let lo = i32x4_sub(i32x4_shr(i32x4_add(lo, rnd_mid), sh), bias);
                        let hi = i32x4_sub(i32x4_shr(i32x4_add(hi, rnd_mid), sh), bias);
                        yaz_i16(tmp, y * w + x, pack_i32_to_i16(lo, hi), w - x);
                    } else {
                        let lo = i32x4_shr(i32x4_add(lo, c), 6);
                        let hi = i32x4_shr(i32x4_add(hi, c), 6);
                        yaz_u16(dst, y * ds + 2 * x, clip_u16_from_i32(lo, hi, v_max), w - x);
                    }
                    x += 8;
                }
            }
        }
        (false, true) => {
            let fv = splat_taps(&katsayi(filt.vertical_table(), h, my));
            let v_rnd32 = i32x4_splat(32);
            let mut x = 0;
            while x < w {
                let mut p = [i16x8_splat(0); 8];
                for k in 0..7 {
                    p[k] = load_v128_u8(src, k * ss + 2 * (x + 3));
                }
                for y in 0..h {
                    p[7] = load_v128_u8(src, (y + 7) * ss + 2 * (x + 3));
                    let (lo, hi) = dikey_i32(&p, &fv);
                    if PREP {
                        let lo = i32x4_sub(i32x4_shr(i32x4_add(lo, rnd_mid), sh), bias);
                        let hi = i32x4_sub(i32x4_shr(i32x4_add(hi, rnd_mid), sh), bias);
                        yaz_i16(tmp, y * w + x, pack_i32_to_i16(lo, hi), w - x);
                    } else {
                        let lo = i32x4_shr(i32x4_add(lo, v_rnd32), 6);
                        let hi = i32x4_shr(i32x4_add(hi, v_rnd32), 6);
                        yaz_u16(dst, y * ds + 2 * x, clip_u16_from_i32(lo, hi, v_max), w - x);
                    }
                    p.copy_within(1.., 0);
                }
                x += 8;
            }
        }
        (true, true) => {
            let fh = splat_taps(&katsayi(filt.horizontal_table(), w, mx));
            let fv = splat_taps(&katsayi(filt.vertical_table(), h, my));
            let sv = 6 + ib;
            let rnd_v = i32x4_splat((1i32 << sv) >> 1);
            let v_rnd32 = i32x4_splat(32);
            let mut x = 0;
            while x < w {
                let mut p = [i16x8_splat(0); 8];
                for k in 0..7 {
                    let (lo, hi) = yatay16(src, k * ss + 2 * x, &fh);
                    p[k] = pack_i32_to_i16(
                        i32x4_shr(i32x4_add(lo, rnd_mid), sh),
                        i32x4_shr(i32x4_add(hi, rnd_mid), sh),
                    );
                }
                for y in 0..h {
                    let (lo, hi) = yatay16(src, (y + 7) * ss + 2 * x, &fh);
                    p[7] = pack_i32_to_i16(
                        i32x4_shr(i32x4_add(lo, rnd_mid), sh),
                        i32x4_shr(i32x4_add(hi, rnd_mid), sh),
                    );
                    let (lo, hi) = dikey_i32(&p, &fv);
                    if PREP {
                        let lo = i32x4_sub(i32x4_shr(i32x4_add(lo, v_rnd32), 6), bias);
                        let hi = i32x4_sub(i32x4_shr(i32x4_add(hi, v_rnd32), 6), bias);
                        yaz_i16(tmp, y * w + x, pack_i32_to_i16(lo, hi), w - x);
                    } else {
                        let lo = i32x4_shr(i32x4_add(lo, rnd_v), sv);
                        let hi = i32x4_shr(i32x4_add(hi, rnd_v), sv);
                        yaz_u16(dst, y * ds + 2 * x, clip_u16_from_i32(lo, hi, v_max), w - x);
                    }
                    p.copy_within(1.., 0);
                }
                x += 8;
            }
        }
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn bilin_yatay16(src: &[u8], off: usize, va: v128, vb: v128) -> v128 {
    let a = load_v128_u8(src, off);
    let b = ext_i16::<1>(a, load_v128_u8(src, off + 16));
    i16x8_add(i16x8_mul(a, va), i16x8_mul(b, vb))
}

#[target_feature(enable = "simd128")]
fn bilin16<const BPC: u32, const PREP: bool>(
    dst: &mut [u8],
    ds: usize,
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    let ib = 14 - BPC;
    let hsh = 4 - ib;
    let h_rnd = u16x8_splat((1u16 << hsh) >> 1);
    let v_max = u16x8_splat(((1u32 << BPC) - 1) as u16);
    let b16 = i16x8_splat(8192);
    match (mx != 0, my != 0) {
        (false, false) => {
            sekiz_tap16::<BPC, PREP>(dst, ds, tmp, src, ss, w, h, 0, 0, Filter2d::Regular);
        }
        (true, false) => {
            let va = i16x8_splat(16 - i16::from(mx));
            let vb = i16x8_splat(i16::from(mx));
            let ib_rnd = u16x8_splat((1u16 << ib) >> 1);
            for y in 0..h {
                let so = (y + 3) * ss + 6;
                let mut x = 0;
                while x < w {
                    let raw = bilin_yatay16(src, so + 2 * x, va, vb);
                    let px = if hsh > 0 {
                        u16x8_shr(i16x8_add(raw, h_rnd), hsh)
                    } else {
                        raw
                    };
                    if PREP {
                        yaz_i16(tmp, y * w + x, i16x8_sub(px, b16), w - x);
                    } else {
                        let out = u16x8_min(u16x8_shr(i16x8_add(px, ib_rnd), ib), v_max);
                        yaz_u16(dst, y * ds + 2 * x, out, w - x);
                    }
                    x += 8;
                }
            }
        }
        (false, true) => {
            let va = i16x8_splat(16 - i16::from(my));
            let vb = i16x8_splat(i16::from(my));
            let rnd8 = u16x8_splat(8);
            for y in 0..h {
                let so = (y + 3) * ss + 6;
                let mut x = 0;
                while x < w {
                    let p0 = load_v128_u8(src, so + 2 * x);
                    let p1 = load_v128_u8(src, so + ss + 2 * x);
                    let v = i16x8_add(i16x8_mul(p0, va), i16x8_mul(p1, vb));
                    if PREP {
                        let t = if hsh > 0 {
                            u16x8_shr(i16x8_add(v, h_rnd), hsh)
                        } else {
                            v
                        };
                        yaz_i16(tmp, y * w + x, i16x8_sub(t, b16), w - x);
                    } else {
                        let out = u16x8_min(u16x8_shr(i16x8_add(v, rnd8), 4), v_max);
                        yaz_u16(dst, y * ds + 2 * x, out, w - x);
                    }
                    x += 8;
                }
            }
        }
        (true, true) => {
            let h_a = i16x8_splat(16 - i16::from(mx));
            let h_b = i16x8_splat(i16::from(mx));
            let v_a = u16x8_splat(16 - u16::from(my));
            let v_b = u16x8_splat(u16::from(my));
            let sv = 4 + ib;
            let rnd_v = i32x4_splat((1i32 << sv) >> 1);
            let rnd8 = i32x4_splat(8);
            let bias = i32x4_splat(8192);
            let v_max_s = i16x8_splat(((1u32 << BPC) - 1) as i16);
            let mut x = 0;
            while x < w {
                let r0 = bilin_yatay16(src, 3 * ss + 6 + 2 * x, h_a, h_b);
                let mut m0 = if hsh > 0 {
                    u16x8_shr(i16x8_add(r0, h_rnd), hsh)
                } else {
                    r0
                };
                for y in 0..h {
                    let r1 = bilin_yatay16(src, (y + 4) * ss + 6 + 2 * x, h_a, h_b);
                    let m1 = if hsh > 0 {
                        u16x8_shr(i16x8_add(r1, h_rnd), hsh)
                    } else {
                        r1
                    };
                    let lo = i32x4_add(
                        u32x4_extmul_low_u16x8(m0, v_a),
                        u32x4_extmul_low_u16x8(m1, v_b),
                    );
                    let hi = i32x4_add(
                        u32x4_extmul_high_u16x8(m0, v_a),
                        u32x4_extmul_high_u16x8(m1, v_b),
                    );
                    if PREP {
                        let lo = i32x4_sub(u32x4_shr(i32x4_add(lo, rnd8), 4), bias);
                        let hi = i32x4_sub(u32x4_shr(i32x4_add(hi, rnd8), 4), bias);
                        yaz_i16(tmp, y * w + x, pack_i32_to_i16(lo, hi), w - x);
                    } else {
                        let lo = u32x4_shr(i32x4_add(lo, rnd_v), sv);
                        let hi = u32x4_shr(i32x4_add(hi, rnd_v), sv);
                        yaz_u16(
                            dst,
                            y * ds + 2 * x,
                            clip_u16_from_i32(lo, hi, v_max_s),
                            w - x,
                        );
                    }
                    m0 = m1;
                }
                x += 8;
            }
        }
    }
}

#[inline]
fn put_kabul<const BPC: u32>(
    dst_len: usize,
    ds: usize,
    src_len: usize,
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bd: BitDepth,
    bdmax: u16,
) -> bool {
    let bpp = bpp_of(BPC);
    bd_uyumlu::<BPC>(bd, bdmax)
        && h >= 1
        && ds >= w * bpp
        && if BPC == 8 {
            simd_sozlesmesi(src_len, ss, w, h, mx, my, dst_len, (h - 1) * ds + w)
        } else {
            simd16_sozlesmesi(
                src_len,
                ss,
                w,
                h,
                mx,
                my,
                bd,
                bdmax,
                dst_len,
                (h - 1) * ds + 2 * w,
            )
        }
}

#[inline]
fn prep_kabul<const BPC: u32>(
    tmp_len: usize,
    src_len: usize,
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bd: BitDepth,
    bdmax: u16,
) -> bool {
    bd_uyumlu::<BPC>(bd, bdmax)
        && if BPC == 8 {
            simd_sozlesmesi(src_len, ss, w, h, mx, my, tmp_len, w * h)
        } else {
            simd16_sozlesmesi(src_len, ss, w, h, mx, my, bd, bdmax, tmp_len, w * h)
        }
}

#[target_feature(enable = "simd128")]
fn put_simd128<const BPC: u32, const FI: usize>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    let mut bos: [i16; 0] = [];
    if BPC == 8 {
        if FI == BILIN {
            bilin8::<false>(dst, ds, &mut bos, src, ss, w, h, mx, my);
        } else {
            sekiz_tap8::<false>(dst, ds, &mut bos, src, ss, w, h, mx, my, filtre(FI));
        }
    } else if FI == BILIN {
        bilin16::<BPC, false>(dst, ds, &mut bos, src, ss, w, h, mx, my);
    } else {
        sekiz_tap16::<BPC, false>(dst, ds, &mut bos, src, ss, w, h, mx, my, filtre(FI));
    }
}

#[target_feature(enable = "simd128")]
fn prep_simd128<const BPC: u32, const FI: usize>(
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    let mut bos: [u8; 0] = [];
    if BPC == 8 {
        if FI == BILIN {
            bilin8::<true>(&mut bos, 0, tmp, src, ss, w, h, mx, my);
        } else {
            sekiz_tap8::<true>(&mut bos, 0, tmp, src, ss, w, h, mx, my, filtre(FI));
        }
    } else if FI == BILIN {
        bilin16::<BPC, true>(&mut bos, 0, tmp, src, ss, w, h, mx, my);
    } else {
        sekiz_tap16::<BPC, true>(&mut bos, 0, tmp, src, ss, w, h, mx, my, filtre(FI));
    }
}

pub fn mc_put<const BPC: u32, const FI: usize>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bd: BitDepth,
    bdmax: u16,
) {
    if !put_kabul::<BPC>(dst.len(), ds, src.len(), ss, w, h, mx, my, bd, bdmax) {
        skaler_devri_say();
        return (McDsp::scalar_for(bd).mc[FI])(dst, ds, src, ss, w, h, mx, my, bd, bdmax);
    }
    put_simd128::<BPC, FI>(dst, ds, src, ss, w, h, mx, my);
}

pub fn mc_prep<const BPC: u32, const FI: usize>(
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bd: BitDepth,
    bdmax: u16,
) {
    if !prep_kabul::<BPC>(tmp.len(), src.len(), ss, w, h, mx, my, bd, bdmax) {
        skaler_devri_say();
        return (McDsp::scalar_for(bd).mct[FI])(tmp, src, ss, w, h, mx, my, bd, bdmax);
    }
    prep_simd128::<BPC, FI>(tmp, src, ss, w, h, mx, my);
}

macro_rules! fi_tablosu {
    ($fi:expr, $f:ident, $bpc:ident) => {
        match $fi {
            0 => $f::<$bpc, 0>,
            1 => $f::<$bpc, 1>,
            2 => $f::<$bpc, 2>,
            3 => $f::<$bpc, 3>,
            4 => $f::<$bpc, 4>,
            5 => $f::<$bpc, 5>,
            6 => $f::<$bpc, 6>,
            7 => $f::<$bpc, 7>,
            8 => $f::<$bpc, 8>,
            _ => $f::<$bpc, 9>,
        }
    };
}

#[must_use]
pub const fn put_fn<const BPC: u32>(fi: usize) -> McPutFn {
    fi_tablosu!(fi, mc_put, BPC)
}

#[must_use]
pub const fn prep_fn<const BPC: u32>(fi: usize) -> McPrepFn {
    fi_tablosu!(fi, mc_prep, BPC)
}

// ============================================================================
// 2. Bileşik: `avg`, `w_avg`, `mask`, `w_mask[0..2]` (18 yuva)
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
    let v_w1 = i16x8_splat(weight as i16);
    let v_w2 = i16x8_splat((16 - weight) as i16);
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
                        i32x4_extmul_low_i16x8(a, v_w1),
                        i32x4_extmul_low_i16x8(b, v_w2),
                    ),
                    v_rnd,
                ),
                shift,
            );
            let hi = i32x4_shr(
                i32x4_add(
                    i32x4_add(
                        i32x4_extmul_high_i16x8(a, v_w1),
                        i32x4_extmul_high_i16x8(b, v_w2),
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
                        i32x4_extmul_low_i16x8(a, v_w1),
                        i32x4_extmul_low_i16x8(b, v_w2),
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
                        i32x4_extmul_low_i16x8(a, m16),
                        i32x4_extmul_low_i16x8(b, inv_m16),
                    ),
                    v_rnd,
                ),
                shift,
            );
            let hi = i32x4_shr(
                i32x4_add(
                    i32x4_add(
                        i32x4_extmul_high_i16x8(a, m16),
                        i32x4_extmul_high_i16x8(b, inv_m16),
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
                        i32x4_extmul_low_i16x8(a, m16),
                        i32x4_extmul_low_i16x8(b, inv_m16),
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

#[target_feature(enable = "simd128")]
fn w_mask_simd128<const BPC: u32, const SUB: usize>(
    dst: &mut [u8],
    ds: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask_buf: &mut [u8],
    sign: i32,
) {
    let ib = intermediate_bits_of(BPC);
    let shift = ib + 6;
    let rnd = (32i32 << ib) + prep_bias_of(BPC) * 64;
    let mask_sh = BPC + ib - 4;
    let mask_rnd = i32x4_splat(1i32 << (mask_sh - 5));
    let v_rnd = i32x4_splat(rnd);
    let c38 = i32x4_splat(38);
    let c64 = i32x4_splat(64);
    let v_max = i16x8_splat(((1i32 << BPC) - 1) as i16);
    let bpp = bpp_of(BPC);
    let ss_hor = SUB != 0;
    let ss_ver = SUB == 2;
    let mask_row = w >> usize::from(ss_hor);
    let mut mask_off = 0usize;

    for y in 0..h {
        let rows_tek = (h - y) & 1 != 0;
        let mut x = 0usize;
        while x < w {
            let i = y * w + x;
            let t1 = load_i16x8(tmp1, i);
            let t2 = load_i16x8(tmp2, i);
            let t2_lo = i32x4_extend_low_i16x8(t2);
            let t2_hi = i32x4_extend_high_i16x8(t2);
            let dlo = i32x4_sub(i32x4_extend_low_i16x8(t1), t2_lo);
            let dhi = i32x4_sub(i32x4_extend_high_i16x8(t1), t2_hi);

            let mlo = i32x4_min(
                i32x4_add(i32x4_shr(i32x4_add(i32x4_abs(dlo), mask_rnd), mask_sh), c38),
                c64,
            );
            let mhi = i32x4_min(
                i32x4_add(i32x4_shr(i32x4_add(i32x4_abs(dhi), mask_rnd), mask_sh), c38),
                c64,
            );

            let vlo = i32x4_shr(
                i32x4_add(i32x4_add(i32x4_mul(dlo, mlo), i32x4_shl(t2_lo, 6)), v_rnd),
                shift,
            );
            let vhi = i32x4_shr(
                i32x4_add(i32x4_add(i32x4_mul(dhi, mhi), i32x4_shl(t2_hi, 6)), v_rnd),
                shift,
            );
            pack_and_store_8::<BPC>(dst, y * ds + x * bpp, vlo, vhi, v_max);

            let m16 = i16x8_narrow_i32x4(mlo, mhi);
            if !ss_hor {
                store_low64_u8(mask_buf, mask_off + x, u8x16_narrow_i16x8(m16, m16));
            } else {
                // Pairwise sum of 8 u16 values -> 4 u32 values (each <= 128)
                let pair = u32x4_extadd_pairwise_u16x8(m16);
                let idx = mask_off + (x >> 1);
                let val32 = if !ss_ver {
                    i32x4_shr(i32x4_add(pair, i32x4_splat(1 - sign)), 1)
                } else if rows_tek {
                    let prev = i32x4_extend_low_i16x8(u16x8_extend_low_u8x16(load_low32_u8(
                        mask_buf, idx,
                    )));
                    i32x4_shr(i32x4_add(i32x4_add(pair, prev), i32x4_splat(2 - sign)), 2)
                } else {
                    pair
                };
                let v16 = i16x8_narrow_i32x4(val32, val32);
                store_low32_u8(mask_buf, idx, u8x16_narrow_i16x8(v16, v16));
            }
            x += 8;
        }
        if !ss_ver || rows_tek {
            mask_off += mask_row;
        }
    }
}

pub fn w_mask<const BPC: u32, const SUB: usize>(
    dst: &mut [u8],
    ds: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask_buf: &mut [u8],
    sign: i32,
    ss_hor: bool,
    ss_ver: bool,
    bd: BitDepth,
    bdmax: u16,
) {
    let ok = (ss_hor, ss_ver) == (SUB != 0, SUB == 2)
        && (0..=1).contains(&sign)
        && mask_buf.len() >= (w >> usize::from(ss_hor)) * (h >> usize::from(ss_ver))
        && bd_uyumlu::<BPC>(bd, bdmax)
        && if BPC == 8 {
            bilesik_sozlesmesi(w, h, 8, tmp1.len(), tmp2.len(), w * h, dst.len(), ds)
        } else {
            bilesik16_sozlesmesi(w, h, 8, tmp1.len(), tmp2.len(), dst.len(), ds, bd, bdmax)
        };
    if !ok {
        skaler_devri_say();
        return (McDsp::scalar_for(bd).w_mask[SUB])(
            dst, ds, tmp1, tmp2, w, h, mask_buf, sign, ss_hor, ss_ver, bd, bdmax,
        );
    }
    w_mask_simd128::<BPC, SUB>(dst, ds, tmp1, tmp2, w, h, mask_buf, sign);
}

// ============================================================================
// 3. Karışım: `blend`, `blend_v`, `blend_h` (9 yuva)
// ============================================================================

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
                i32x4_extmul_low_i16x8(a, inv_m16),
                i32x4_extmul_low_i16x8(b, m16),
            ),
            v_rnd,
        ),
        6,
    );
    let hi = i32x4_shr(
        i32x4_add(
            i32x4_add(
                i32x4_extmul_high_i16x8(a, inv_m16),
                i32x4_extmul_high_i16x8(b, m16),
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
                i32x4_extmul_low_i16x8(a, inv_m16),
                i32x4_extmul_low_i16x8(b, m16),
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
            let m16 = u16x8_extend_low_u8x16(load_low64_u8(mask_row, x));
            blend_8_pixels::<BPC>(dst, drow + x * bpp, tmp, (trow + x) * bpp, m16, v_max);
            x += 8;
        }
        while x + 4 <= cols {
            let m16 = u16x8_extend_low_u8x16(load_low32_u8(mask_row, x));
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
// 4. `warp8x8` ve `warp8x8t` (6 yuva)
// ============================================================================

#[inline]
#[target_feature(enable = "simd128")]
fn load_warp_taps(idx: usize) -> v128 {
    let f = &MC_WARP_FILTER[idx];
    i16x8(
        i16::from(f[0]),
        i16::from(f[1]),
        i16::from(f[2]),
        i16::from(f[3]),
        i16::from(f[4]),
        i16::from(f[5]),
        i16::from(f[6]),
        i16::from(f[7]),
    )
}

#[target_feature(enable = "simd128")]
fn warp_horiz<const BPC: u32>(
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mut mx: i32,
) -> [[i16; 15]; 8] {
    let ib = intermediate_bits_of(BPC);
    let sh = 7 - ib;
    let rnd = (1i32 << sh) >> 1;
    let bpp = bpp_of(BPC);
    // Transposed layout: mid_t[x][y] so vertical 8-tap is contiguous.
    let mut mid_t = [[0i16; 15]; 8];
    for y in 0..15 {
        let row = y * ss;
        let mut tmx = mx;
        for x in 0..8 {
            let idx = (64 + ((tmx + 512) >> 10)) as usize;
            let px = load_pixels_8::<BPC>(src, row + x * bpp);
            let fv = load_warp_taps(idx);
            let sum = hsum_i32x4(i32x4_dot_i16x8(px, fv));
            mid_t[x][y] = (sum.wrapping_add(rnd) >> sh) as i16;
            tmx += i32::from(abcd[0]);
        }
        mx += i32::from(abcd[1]);
    }
    mid_t
}

#[target_feature(enable = "simd128")]
fn warp8x8_simd128<const BPC: u32>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    mut my: i32,
) {
    let mid_t = warp_horiz::<BPC>(src, ss, abcd, mx);
    let ib = intermediate_bits_of(BPC);
    let sh = 7 + ib;
    let rnd = (1i32 << sh) >> 1;
    let max_pixel = (1i32 << BPC) - 1;
    let bpp = bpp_of(BPC);

    for y in 0..8 {
        let drow = y * ds;
        let mut tmy = my;
        for x in 0..8 {
            let idx = (64 + ((tmy + 512) >> 10)) as usize;
            let col = load_i16x8(&mid_t[x], y);
            let fv = load_warp_taps(idx);
            let sum = hsum_i32x4(i32x4_dot_i16x8(col, fv));
            let val = (sum.wrapping_add(rnd) >> sh).clamp(0, max_pixel);
            store_scalar::<BPC>(dst, drow + x * bpp, val);
            tmy += i32::from(abcd[2]);
        }
        my += i32::from(abcd[3]);
    }
}

pub fn warp8x8<const BPC: u32>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bd: BitDepth,
    bdmax: u16,
) {
    let ok = bd_uyumlu::<BPC>(bd, bdmax)
        && if BPC == 8 {
            warp_sozlesmesi(src.len(), ss, abcd, mx, my, dst.len(), ds)
        } else {
            warp16_sozlesmesi(src.len(), ss, abcd, mx, my, dst.len(), ds, 2, bd, bdmax)
        };
    if !ok {
        skaler_devri_say();
        return (McDsp::scalar_for(bd).warp8x8)(dst, ds, src, ss, abcd, mx, my, bd, bdmax);
    }
    warp8x8_simd128::<BPC>(dst, ds, src, ss, abcd, mx, my);
}

#[target_feature(enable = "simd128")]
fn warp8x8t_simd128<const BPC: u32>(
    tmp: &mut [i16],
    ts: usize,
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    mut my: i32,
) {
    let mid_t = warp_horiz::<BPC>(src, ss, abcd, mx);
    let bias = prep_bias_of(BPC);
    for y in 0..8 {
        let mut tmy = my;
        for x in 0..8 {
            let idx = (64 + ((tmy + 512) >> 10)) as usize;
            let col = load_i16x8(&mid_t[x], y);
            let fv = load_warp_taps(idx);
            let sum = hsum_i32x4(i32x4_dot_i16x8(col, fv));
            tmp[y * ts + x] = ((sum.wrapping_add(64) >> 7).wrapping_sub(bias)) as i16;
            tmy += i32::from(abcd[2]);
        }
        my += i32::from(abcd[3]);
    }
}

pub fn warp8x8t<const BPC: u32>(
    tmp: &mut [i16],
    ts: usize,
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bd: BitDepth,
    bdmax: u16,
) {
    let ok = bd_uyumlu::<BPC>(bd, bdmax)
        && if BPC == 8 {
            warp_sozlesmesi(src.len(), ss, abcd, mx, my, tmp.len(), ts)
        } else {
            warp16_sozlesmesi(src.len(), ss, abcd, mx, my, tmp.len(), ts, 1, bd, bdmax)
        };
    if !ok {
        skaler_devri_say();
        return (McDsp::scalar_for(bd).warp8x8t)(tmp, ts, src, ss, abcd, mx, my, bd, bdmax);
    }
    warp8x8t_simd128::<BPC>(tmp, ts, src, ss, abcd, mx, my);
}

// ============================================================================
// 5. `emu_edge` (3 yuva)
// ============================================================================

#[inline]
#[target_feature(enable = "simd128")]
fn fill_px<const BPC: u32>(dst: &mut [u8], off: usize, count: usize, val: i32) {
    let bpp = bpp_of(BPC);
    let total = count * bpp;
    let v = if BPC == 8 {
        u8x16_splat(val as u8)
    } else {
        u16x8_splat(val as u16)
    };
    let mut i = 0usize;
    while i + 16 <= total {
        store_v128_u8(dst, off + i, v);
        i += 16;
    }
    while i + 8 <= total {
        store_low64_u8(dst, off + i, v);
        i += 8;
    }
    while i < total {
        store_scalar::<BPC>(dst, off + i, val);
        i += bpp;
    }
}

#[target_feature(enable = "simd128")]
fn emu_edge_simd128<const BPC: u32>(
    bw: usize,
    bh: usize,
    iw: usize,
    ih: usize,
    x: i32,
    y: i32,
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
) {
    let bpp = bpp_of(BPC);
    let ref_row =
        y.clamp(0, ih as i32 - 1) as usize * ss + x.clamp(0, iw as i32 - 1) as usize * bpp;
    let left_ext = (-x).clamp(0, bw as i32 - 1) as usize;
    let right_ext = (x + bw as i32 - iw as i32).clamp(0, bw as i32 - 1) as usize;
    let top_ext = (-y).clamp(0, bh as i32 - 1) as usize;
    let bottom_ext = (y + bh as i32 - ih as i32).clamp(0, bh as i32 - 1) as usize;

    let center_w = bw - left_ext - right_ext;
    let center_h = bh - top_ext - bottom_ext;
    let center_bytes = center_w * bpp;

    for row in 0..center_h {
        let s = ref_row + row * ss;
        let d = (top_ext + row) * ds;
        let d_center = d + left_ext * bpp;
        dst[d_center..d_center + center_bytes].copy_from_slice(&src[s..s + center_bytes]);
        if left_ext > 0 {
            let edge_left = load_scalar::<BPC>(dst, d_center);
            fill_px::<BPC>(dst, d, left_ext, edge_left);
        }
        if right_ext > 0 {
            let edge_right = load_scalar::<BPC>(dst, d_center + (center_w - 1) * bpp);
            fill_px::<BPC>(dst, d_center + center_bytes, right_ext, edge_right);
        }
    }

    let row_bytes = bw * bpp;
    if top_ext > 0 {
        let src_start = top_ext * ds;
        for row in 0..top_ext {
            let dst_start = row * ds;
            dst.copy_within(src_start..src_start + row_bytes, dst_start);
        }
    }
    if bottom_ext > 0 {
        let last = (top_ext + center_h - 1) * ds;
        for row in 0..bottom_ext {
            let dst_start = (top_ext + center_h + row) * ds;
            dst.copy_within(last..last + row_bytes, dst_start);
        }
    }
}

pub fn emu_edge<const BPC: u32>(
    bw: usize,
    bh: usize,
    iw: usize,
    ih: usize,
    x: i32,
    y: i32,
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    let ok = bd_uyumlu::<BPC>(bd, bdmax)
        && if BPC == 8 {
            kenar_sozlesmesi(bw, bh, iw, ih, x, y, dst.len(), ds, src.len(), ss)
        } else {
            kenar16_sozlesmesi(
                bw,
                bh,
                iw,
                ih,
                x,
                y,
                dst.len(),
                ds,
                src.len(),
                ss,
                bd,
                bdmax,
            )
        };
    if !ok {
        skaler_devri_say();
        return (McDsp::scalar_for(bd).emu_edge)(bw, bh, iw, ih, x, y, dst, ds, src, ss, bd, bdmax);
    }
    emu_edge_simd128::<BPC>(bw, bh, iw, ih, x, y, dst, ds, src, ss);
}
