//! WebAssembly `simd128` ters dönüşüm + piksel ekleme (`468` yuva @ `8 / 10 / 12` bpc).
//!
//! Kapsam: desteklenen tüm `156` `itxfm_add[tx][txtp]` hücresi (`WHT_WHT` ve
//! `eob == 0` DC-only hızlı yolu dâhil) × 3 bit derinliği (`8 / 10 / 12` bpc).
//! 1B kelebek çekirdekleri `crate::dsp::arm::itx_1d` (`dav1d` `src/itx_1d.c` portu)
//! `i32x4` aritmetiğinin birebir `v128` karşılığıdır.

#![forbid(unsafe_code)]
#![allow(
    clippy::too_many_arguments,
    clippy::identity_op,
    clippy::erasing_op,
    clippy::neg_multiply,
    clippy::too_many_lines,
    clippy::similar_names
)]

use core::arch::wasm32::{
    i16x8_splat, i32x4, i32x4_add, i32x4_le, i32x4_max, i32x4_min, i32x4_mul, i32x4_neg, i32x4_shr,
    i32x4_splat, i32x4_sub, u32x4_extend_low_u16x8, v128, v128_and,
};

use super::{bpp_of, load_i32x4, load_pixels_4, pack_and_store_4, transpose_4x4_i32};
use crate::dsp::itx::{TxSize, TxType, itx_skaler_devret, itx8_sozlesmesi, itx16_sozlesmesi};
use crate::dsp::itx_tables::LNZC_TABLES;
use crate::dsp::scalar::itx_1d::Tx1dType;
use crate::frame::BitDepth;

#[inline]
#[target_feature(enable = "simd128")]
fn vmul_n(a: v128, n: i32) -> v128 {
    i32x4_mul(a, i32x4_splat(n))
}

#[inline]
#[target_feature(enable = "simd128")]
fn vshr<const N: u32>(a: v128) -> v128 {
    i32x4_shr(a, N)
}

#[inline]
#[target_feature(enable = "simd128")]
fn vrshr<const N: u32>(a: v128) -> v128 {
    i32x4_shr(i32x4_add(a, i32x4_splat(1i32 << (N - 1))), N)
}

/// `itxfm_add[tx][txtp]` `wasm32 +simd128` giriş fonksiyonu.
#[target_feature(enable = "simd128")]
pub fn itx_add<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
    bit_depth: BitDepth,
) {
    let ok = bit_depth.bits() == BPC
        && if BPC == 8 {
            itx8_sozlesmesi(dst.len(), dst_stride, coeff.len(), eob, tx, txtp)
        } else {
            itx16_sozlesmesi(dst.len(), dst_stride, coeff.len(), eob, tx, txtp)
        };
    if !ok {
        itx_skaler_devret::<BPC>(dst, dst_stride, coeff, eob, tx, txtp);
        return;
    }
    itx_wasm::<BPC>(dst, dst_stride, coeff, eob, tx, txtp);
}

#[inline]
#[target_feature(enable = "simd128")]
fn uygula_1d(lsz: u8, tur: Tx1dType, c: &mut [v128], min: v128, max: v128) {
    match (lsz, tur) {
        (0, Tx1dType::Dct) => inv_dct4_1d_c(c, 1, min, max),
        (0, Tx1dType::Adst) => inv_adst4_1d_c(c, 1, min, max),
        (0, Tx1dType::FlipAdst) => inv_flipadst4_1d_c(c, 1, min, max),
        (0, Tx1dType::Identity) => inv_identity4_1d_c(c, 1, min, max),
        (1, Tx1dType::Dct) => inv_dct8_1d_c(c, 1, min, max),
        (1, Tx1dType::Adst) => inv_adst8_1d_c(c, 1, min, max),
        (1, Tx1dType::FlipAdst) => inv_flipadst8_1d_c(c, 1, min, max),
        (1, Tx1dType::Identity) => inv_identity8_1d_c(c, 1, min, max),
        (2, Tx1dType::Dct) => inv_dct16_1d_c(c, 1, min, max),
        (2, Tx1dType::Adst) => inv_adst16_1d_c(c, 1, min, max),
        (2, Tx1dType::FlipAdst) => inv_flipadst16_1d_c(c, 1, min, max),
        (2, Tx1dType::Identity) => inv_identity16_1d_c(c, 1, min, max),
        (3, Tx1dType::Dct) => inv_dct32_1d_c(c, 1, min, max),
        (3, Tx1dType::Identity) => inv_identity32_1d_c(c, 1, min, max),
        (4, Tx1dType::Dct) => inv_dct64_1d_c(c, 1, min, max),
        _ => unreachable!("desteklenmeyen 1B hücre"),
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn yaz4<const BPC: u32>(dst: &mut [u8], o: usize, d: v128, v_max: v128) {
    let cur = u32x4_extend_low_u16x8(load_pixels_4::<BPC>(dst, o));
    pack_and_store_4::<BPC>(dst, o, i32x4_add(cur, d), v_max);
}

#[target_feature(enable = "simd128")]
fn itx_wasm<const BPC: u32>(
    dst: &mut [u8],
    ds: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
) {
    let v_max = i16x8_splat(((1 << BPC) - 1) as i16);
    let bpp = bpp_of(BPC);
    if txtp == TxType::WhtWht {
        wht::<BPC>(dst, ds, coeff, v_max);
        return;
    }
    let w = tx.width();
    let h = tx.height();
    let shift = tx.shift();
    let rect2 = w * 2 == h || h * 2 == w;
    if txtp == TxType::DctDct && eob == 0 {
        let rnd = (1i64 << shift) >> 1;
        let mut dc = i64::from(coeff[0]);
        coeff[0] = 0;
        if rect2 {
            dc = (dc * 181 + 128) >> 8;
        }
        dc = (dc * 181 + 128) >> 8;
        dc = (dc + rnd) >> shift;
        dc = (dc * 181 + 128 + 2048) >> 12;
        let d = i32x4_splat(dc as i32);
        for y in 0..h {
            let mut x = 0usize;
            while x < w {
                yaz4::<BPC>(dst, y * ds + x * bpp, d, v_max);
                x += 4;
            }
        }
        return;
    }
    macro_rules! boyut {
        ($($t:ident => ($w:literal, $h:literal)),* $(,)?) => {
            match tx {
                $(TxSize::$t => genel::<BPC, $w, $h, { $w / 4 }>(dst, ds, coeff, eob, tx, txtp, rect2, shift, v_max),)*
            }
        };
    }
    boyut!(
        Tx4x4 => (4, 4), Tx8x8 => (8, 8), Tx16x16 => (16, 16), Tx32x32 => (32, 32),
        Tx64x64 => (64, 64), Rtx4x8 => (4, 8), Rtx8x4 => (8, 4), Rtx8x16 => (8, 16),
        Rtx16x8 => (16, 8), Rtx16x32 => (16, 32), Rtx32x16 => (32, 16), Rtx32x64 => (32, 64),
        Rtx64x32 => (64, 32), Rtx4x16 => (4, 16), Rtx16x4 => (16, 4), Rtx8x32 => (8, 32),
        Rtx32x8 => (32, 8), Rtx16x64 => (16, 64), Rtx64x16 => (64, 16),
    );
}

#[target_feature(enable = "simd128")]
fn wht<const BPC: u32>(dst: &mut [u8], ds: usize, coeff: &mut [i32], v_max: v128) {
    let mut v = [i32x4_splat(0); 4];
    for (x, vx) in v.iter_mut().enumerate() {
        *vx = vshr::<2>(load_i32x4(coeff, x * 4));
    }
    dav1d_inv_wht4_1d_c(&mut v, 1);
    coeff[..16].fill(0);
    let mut u = transpose_4x4_i32(v[0], v[1], v[2], v[3]);
    dav1d_inv_wht4_1d_c(&mut u, 1);
    for (y, uy) in u.iter().enumerate() {
        yaz4::<BPC>(dst, y * ds, *uy, v_max);
    }
}

#[target_feature(enable = "simd128")]
fn genel<const BPC: u32, const W: usize, const H: usize, const WG: usize>(
    dst: &mut [u8],
    ds: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
    rect2: bool,
    shift: u32,
    v_max: v128,
) {
    let bpp = bpp_of(BPC);
    let (rmin, cmin) = if BPC == 8 {
        (-32768i32, -32768i32)
    } else {
        (-((1i32 << BPC) * 128), -((1i32 << BPC) * 32))
    };
    let (rmin_v, rmax_v) = (i32x4_splat(rmin), i32x4_splat(-rmin - 1));
    let (cmin_v, cmax_v) = (i32x4_splat(cmin), i32x4_splat(-cmin - 1));
    let info = tx.info();
    let sh = H.min(32);
    let sw = W.min(32);
    let (yatay, dikey) = txtp.tx1d();
    let lnzc = if dikey == Tx1dType::Identity && yatay != Tx1dType::Identity {
        (sh - 1).min(eob)
    } else if yatay == Tx1dType::Identity && dikey != Tx1dType::Identity {
        eob >> (info.lw as usize + 2)
    } else {
        usize::from(LNZC_TABLES[tx.index()][eob])
    };
    let rnd_v = i32x4_splat((1i32 << shift) >> 1);
    let seritler = i32x4(0, 1, 2, 3);
    let zero = i32x4_splat(0);

    let mut sutun = [[zero; H]; WG];
    let mut satir = [zero; W];
    let mut y0 = 0usize;
    while y0 <= lnzc {
        let gecerli = i32x4_le(seritler, i32x4_splat((lnzc - y0) as i32));
        for (x, r) in satir.iter_mut().enumerate() {
            if x < sw {
                let c = v128_and(load_i32x4(coeff, x * sh + y0), gecerli);
                *r = if rect2 { vrshr::<8>(vmul_n(c, 181)) } else { c };
            } else {
                *r = zero;
            }
        }
        uygula_1d(info.lw, yatay, &mut satir, rmin_v, rmax_v);
        if shift == 0 {
            for r in &mut satir {
                *r = i32x4_min(i32x4_max(*r, cmin_v), cmax_v);
            }
        } else {
            for r in &mut satir {
                let shifted = i32x4_shr(i32x4_add(*r, rnd_v), shift);
                *r = i32x4_min(i32x4_max(shifted, cmin_v), cmax_v);
            }
        }
        for (xg, s) in sutun.iter_mut().enumerate() {
            let t = transpose_4x4_i32(
                satir[4 * xg],
                satir[4 * xg + 1],
                satir[4 * xg + 2],
                satir[4 * xg + 3],
            );
            s[y0..y0 + 4].copy_from_slice(&t);
        }
        y0 += 4;
    }
    coeff[..sw * sh].fill(0);
    for (xg, s) in sutun.iter_mut().enumerate() {
        uygula_1d(info.lh, dikey, s, cmin_v, cmax_v);
        for (y, v) in s.iter().enumerate() {
            yaz4::<BPC>(dst, y * ds + 4 * xg * bpp, vrshr::<4>(*v), v_max);
        }
    }
}

// ============================================================================
// 1B ters dönüşüm kelebek çekirdekleri (`v128` `i32x4` şeritleri)
// ============================================================================

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_dct4_1d_internal(c: &mut [v128], stride: usize, min: v128, max: v128, tx64: bool) {
    let in0: v128 = c[0];
    let in1: v128 = c[stride];

    let t0: v128;
    let t1: v128;
    let t2: v128;
    let t3: v128;
    if tx64 {
        t0 = vrshr::<8>(vmul_n(in0, 181));
        t1 = vrshr::<8>(vmul_n(in0, 181));
        t2 = vrshr::<12>(vmul_n(in1, 1567));
        t3 = vrshr::<12>(vmul_n(in1, 3784));
    } else {
        let in2: v128 = c[2 * stride];
        let in3: v128 = c[3 * stride];

        t0 = vrshr::<8>(vmul_n(i32x4_add(in0, in2), 181));
        t1 = vrshr::<8>(vmul_n(i32x4_sub(in0, in2), 181));
        t2 = i32x4_sub(
            vrshr::<12>(i32x4_sub(vmul_n(in1, 1567), vmul_n(in3, -312))),
            in3,
        );
        t3 = i32x4_add(
            vrshr::<12>(i32x4_add(vmul_n(in1, -312), vmul_n(in3, 1567))),
            in1,
        );
    }

    c[0] = i32x4_min(i32x4_max(i32x4_add(t0, t3), min), max);
    c[stride] = i32x4_min(i32x4_max(i32x4_add(t1, t2), min), max);
    c[2 * stride] = i32x4_min(i32x4_max(i32x4_sub(t1, t2), min), max);
    c[3 * stride] = i32x4_min(i32x4_max(i32x4_sub(t0, t3), min), max);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_dct8_1d_internal(c: &mut [v128], stride: usize, min: v128, max: v128, tx64: bool) {
    inv_dct4_1d_internal(c, stride << 1, min, max, tx64);

    let in1: v128 = c[stride];
    let in3: v128 = c[3 * stride];

    let t4a: v128;
    let mut t5a: v128;
    let mut t6a: v128;
    let t7a: v128;
    if tx64 {
        t4a = vrshr::<12>(vmul_n(in1, 799));
        t5a = vrshr::<12>(vmul_n(in3, -2276));
        t6a = vrshr::<12>(vmul_n(in3, 3406));
        t7a = vrshr::<12>(vmul_n(in1, 4017));
    } else {
        let in5: v128 = c[5 * stride];
        let in7: v128 = c[7 * stride];

        t4a = i32x4_sub(
            vrshr::<12>(i32x4_sub(vmul_n(in1, 799), vmul_n(in7, -79))),
            in7,
        );
        t5a = vrshr::<11>(i32x4_sub(vmul_n(in5, 1703), vmul_n(in3, 1138)));
        t6a = vrshr::<11>(i32x4_add(vmul_n(in5, 1138), vmul_n(in3, 1703)));
        t7a = i32x4_add(
            vrshr::<12>(i32x4_add(vmul_n(in1, -79), vmul_n(in7, 799))),
            in1,
        );
    }

    let t4: v128 = i32x4_min(i32x4_max(i32x4_add(t4a, t5a), min), max);
    t5a = i32x4_min(i32x4_max(i32x4_sub(t4a, t5a), min), max);
    let t7: v128 = i32x4_min(i32x4_max(i32x4_add(t7a, t6a), min), max);
    t6a = i32x4_min(i32x4_max(i32x4_sub(t7a, t6a), min), max);

    let t5: v128 = vrshr::<8>(vmul_n(i32x4_sub(t6a, t5a), 181));
    let t6: v128 = vrshr::<8>(vmul_n(i32x4_add(t6a, t5a), 181));

    let t0: v128 = c[0];
    let t1: v128 = c[2 * stride];
    let t2: v128 = c[4 * stride];
    let t3: v128 = c[6 * stride];

    c[0] = i32x4_min(i32x4_max(i32x4_add(t0, t7), min), max);
    c[stride] = i32x4_min(i32x4_max(i32x4_add(t1, t6), min), max);
    c[2 * stride] = i32x4_min(i32x4_max(i32x4_add(t2, t5), min), max);
    c[3 * stride] = i32x4_min(i32x4_max(i32x4_add(t3, t4), min), max);
    c[4 * stride] = i32x4_min(i32x4_max(i32x4_sub(t3, t4), min), max);
    c[5 * stride] = i32x4_min(i32x4_max(i32x4_sub(t2, t5), min), max);
    c[6 * stride] = i32x4_min(i32x4_max(i32x4_sub(t1, t6), min), max);
    c[7 * stride] = i32x4_min(i32x4_max(i32x4_sub(t0, t7), min), max);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_dct16_1d_internal(c: &mut [v128], stride: usize, min: v128, max: v128, tx64: bool) {
    inv_dct8_1d_internal(c, stride << 1, min, max, tx64);

    let in1: v128 = c[stride];
    let in3: v128 = c[3 * stride];
    let in5: v128 = c[5 * stride];
    let in7: v128 = c[7 * stride];

    let mut t8a: v128;
    let mut t9a: v128;
    let mut t10a: v128;
    let mut t11a: v128;
    let mut t12a: v128;
    let mut t13a: v128;
    let mut t14a: v128;
    let mut t15a: v128;
    if tx64 {
        t8a = vrshr::<12>(vmul_n(in1, 401));
        t9a = vrshr::<12>(vmul_n(in7, -2598));
        t10a = vrshr::<12>(vmul_n(in5, 1931));
        t11a = vrshr::<12>(vmul_n(in3, -1189));
        t12a = vrshr::<12>(vmul_n(in3, 3920));
        t13a = vrshr::<12>(vmul_n(in5, 3612));
        t14a = vrshr::<12>(vmul_n(in7, 3166));
        t15a = vrshr::<12>(vmul_n(in1, 4076));
    } else {
        let in9: v128 = c[9 * stride];
        let in11: v128 = c[11 * stride];
        let in13: v128 = c[13 * stride];
        let in15: v128 = c[15 * stride];

        t8a = i32x4_sub(
            vrshr::<12>(i32x4_sub(vmul_n(in1, 401), vmul_n(in15, -20))),
            in15,
        );
        t9a = vrshr::<11>(i32x4_sub(vmul_n(in9, 1583), vmul_n(in7, 1299)));
        t10a = i32x4_sub(
            vrshr::<12>(i32x4_sub(vmul_n(in5, 1931), vmul_n(in11, -484))),
            in11,
        );
        t11a = i32x4_add(
            vrshr::<12>(i32x4_sub(vmul_n(in13, -176), vmul_n(in3, 1189))),
            in13,
        );
        t12a = i32x4_add(
            vrshr::<12>(i32x4_add(vmul_n(in13, 1189), vmul_n(in3, -176))),
            in3,
        );
        t13a = i32x4_add(
            vrshr::<12>(i32x4_add(vmul_n(in5, -484), vmul_n(in11, 1931))),
            in5,
        );
        t14a = vrshr::<11>(i32x4_add(vmul_n(in9, 1299), vmul_n(in7, 1583)));
        t15a = i32x4_add(
            vrshr::<12>(i32x4_add(vmul_n(in1, -20), vmul_n(in15, 401))),
            in1,
        );
    }

    let t8: v128 = i32x4_min(i32x4_max(i32x4_add(t8a, t9a), min), max);
    let mut t9: v128 = i32x4_min(i32x4_max(i32x4_sub(t8a, t9a), min), max);
    let mut t10: v128 = i32x4_min(i32x4_max(i32x4_sub(t11a, t10a), min), max);
    let mut t11: v128 = i32x4_min(i32x4_max(i32x4_add(t11a, t10a), min), max);
    let mut t12: v128 = i32x4_min(i32x4_max(i32x4_add(t12a, t13a), min), max);
    let mut t13: v128 = i32x4_min(i32x4_max(i32x4_sub(t12a, t13a), min), max);
    let mut t14: v128 = i32x4_min(i32x4_max(i32x4_sub(t15a, t14a), min), max);
    let t15: v128 = i32x4_min(i32x4_max(i32x4_add(t15a, t14a), min), max);

    t9a = i32x4_sub(
        vrshr::<12>(i32x4_sub(vmul_n(t14, 1567), vmul_n(t9, -312))),
        t9,
    );
    t14a = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t14, -312), vmul_n(t9, 1567))),
        t14,
    );
    t10a = i32x4_sub(
        vrshr::<12>(i32x4_neg(i32x4_add(vmul_n(t13, -312), vmul_n(t10, 1567)))),
        t13,
    );
    t13a = i32x4_sub(
        vrshr::<12>(i32x4_sub(vmul_n(t13, 1567), vmul_n(t10, -312))),
        t10,
    );

    t8a = i32x4_min(i32x4_max(i32x4_add(t8, t11), min), max);
    t9 = i32x4_min(i32x4_max(i32x4_add(t9a, t10a), min), max);
    t10 = i32x4_min(i32x4_max(i32x4_sub(t9a, t10a), min), max);
    t11a = i32x4_min(i32x4_max(i32x4_sub(t8, t11), min), max);
    t12a = i32x4_min(i32x4_max(i32x4_sub(t15, t12), min), max);
    t13 = i32x4_min(i32x4_max(i32x4_sub(t14a, t13a), min), max);
    t14 = i32x4_min(i32x4_max(i32x4_add(t14a, t13a), min), max);
    t15a = i32x4_min(i32x4_max(i32x4_add(t15, t12), min), max);

    t10a = vrshr::<8>(vmul_n(i32x4_sub(t13, t10), 181));
    t13a = vrshr::<8>(vmul_n(i32x4_add(t13, t10), 181));
    t11 = vrshr::<8>(vmul_n(i32x4_sub(t12a, t11a), 181));
    t12 = vrshr::<8>(vmul_n(i32x4_add(t12a, t11a), 181));

    let t0: v128 = c[0];
    let t1: v128 = c[2 * stride];
    let t2: v128 = c[4 * stride];
    let t3: v128 = c[6 * stride];
    let t4: v128 = c[8 * stride];
    let t5: v128 = c[10 * stride];
    let t6: v128 = c[12 * stride];
    let t7: v128 = c[14 * stride];

    c[0] = i32x4_min(i32x4_max(i32x4_add(t0, t15a), min), max);
    c[stride] = i32x4_min(i32x4_max(i32x4_add(t1, t14), min), max);
    c[2 * stride] = i32x4_min(i32x4_max(i32x4_add(t2, t13a), min), max);
    c[3 * stride] = i32x4_min(i32x4_max(i32x4_add(t3, t12), min), max);
    c[4 * stride] = i32x4_min(i32x4_max(i32x4_add(t4, t11), min), max);
    c[5 * stride] = i32x4_min(i32x4_max(i32x4_add(t5, t10a), min), max);
    c[6 * stride] = i32x4_min(i32x4_max(i32x4_add(t6, t9), min), max);
    c[7 * stride] = i32x4_min(i32x4_max(i32x4_add(t7, t8a), min), max);
    c[8 * stride] = i32x4_min(i32x4_max(i32x4_sub(t7, t8a), min), max);
    c[9 * stride] = i32x4_min(i32x4_max(i32x4_sub(t6, t9), min), max);
    c[10 * stride] = i32x4_min(i32x4_max(i32x4_sub(t5, t10a), min), max);
    c[11 * stride] = i32x4_min(i32x4_max(i32x4_sub(t4, t11), min), max);
    c[12 * stride] = i32x4_min(i32x4_max(i32x4_sub(t3, t12), min), max);
    c[13 * stride] = i32x4_min(i32x4_max(i32x4_sub(t2, t13a), min), max);
    c[14 * stride] = i32x4_min(i32x4_max(i32x4_sub(t1, t14), min), max);
    c[15 * stride] = i32x4_min(i32x4_max(i32x4_sub(t0, t15a), min), max);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_dct32_1d_internal(c: &mut [v128], stride: usize, min: v128, max: v128, tx64: bool) {
    inv_dct16_1d_internal(c, stride << 1, min, max, tx64);

    let in1: v128 = c[stride];
    let in3: v128 = c[3 * stride];
    let in5: v128 = c[5 * stride];
    let in7: v128 = c[7 * stride];
    let in9: v128 = c[9 * stride];
    let in11: v128 = c[11 * stride];
    let in13: v128 = c[13 * stride];
    let in15: v128 = c[15 * stride];

    let mut t16a: v128;
    let mut t17a: v128;
    let mut t18a: v128;
    let mut t19a: v128;
    let mut t20a: v128;
    let mut t21a: v128;
    let mut t22a: v128;
    let mut t23a: v128;
    let mut t24a: v128;
    let mut t25a: v128;
    let mut t26a: v128;
    let mut t27a: v128;
    let mut t28a: v128;
    let mut t29a: v128;
    let mut t30a: v128;
    let mut t31a: v128;
    if tx64 {
        t16a = vrshr::<12>(vmul_n(in1, 201));
        t17a = vrshr::<12>(vmul_n(in15, -2751));
        t18a = vrshr::<12>(vmul_n(in9, 1751));
        t19a = vrshr::<12>(vmul_n(in7, -1380));
        t20a = vrshr::<12>(vmul_n(in5, 995));
        t21a = vrshr::<12>(vmul_n(in11, -2106));
        t22a = vrshr::<12>(vmul_n(in13, 2440));
        t23a = vrshr::<12>(vmul_n(in3, -601));
        t24a = vrshr::<12>(vmul_n(in3, 4052));
        t25a = vrshr::<12>(vmul_n(in13, 3290));
        t26a = vrshr::<12>(vmul_n(in11, 3513));
        t27a = vrshr::<12>(vmul_n(in5, 3973));
        t28a = vrshr::<12>(vmul_n(in7, 3857));
        t29a = vrshr::<12>(vmul_n(in9, 3703));
        t30a = vrshr::<12>(vmul_n(in15, 3035));
        t31a = vrshr::<12>(vmul_n(in1, 4091));
    } else {
        let in17: v128 = c[17 * stride];
        let in19: v128 = c[19 * stride];
        let in21: v128 = c[21 * stride];
        let in23: v128 = c[23 * stride];
        let in25: v128 = c[25 * stride];
        let in27: v128 = c[27 * stride];
        let in29: v128 = c[29 * stride];
        let in31: v128 = c[31 * stride];

        t16a = i32x4_sub(
            vrshr::<12>(i32x4_sub(vmul_n(in1, 201), vmul_n(in31, -5))),
            in31,
        );
        t17a = i32x4_add(
            vrshr::<12>(i32x4_sub(vmul_n(in17, -1061), vmul_n(in15, 2751))),
            in17,
        );
        t18a = i32x4_sub(
            vrshr::<12>(i32x4_sub(vmul_n(in9, 1751), vmul_n(in23, -393))),
            in23,
        );
        t19a = i32x4_add(
            vrshr::<12>(i32x4_sub(vmul_n(in25, -239), vmul_n(in7, 1380))),
            in25,
        );
        t20a = i32x4_sub(
            vrshr::<12>(i32x4_sub(vmul_n(in5, 995), vmul_n(in27, -123))),
            in27,
        );
        t21a = i32x4_add(
            vrshr::<12>(i32x4_sub(vmul_n(in21, -583), vmul_n(in11, 2106))),
            in21,
        );
        t22a = vrshr::<11>(i32x4_sub(vmul_n(in13, 1220), vmul_n(in19, 1645)));
        t23a = i32x4_add(
            vrshr::<12>(i32x4_sub(vmul_n(in29, -44), vmul_n(in3, 601))),
            in29,
        );
        t24a = i32x4_add(
            vrshr::<12>(i32x4_add(vmul_n(in29, 601), vmul_n(in3, -44))),
            in3,
        );
        t25a = vrshr::<11>(i32x4_add(vmul_n(in13, 1645), vmul_n(in19, 1220)));
        t26a = i32x4_add(
            vrshr::<12>(i32x4_add(vmul_n(in21, 2106), vmul_n(in11, -583))),
            in11,
        );
        t27a = i32x4_add(
            vrshr::<12>(i32x4_add(vmul_n(in5, -123), vmul_n(in27, 995))),
            in5,
        );
        t28a = i32x4_add(
            vrshr::<12>(i32x4_add(vmul_n(in25, 1380), vmul_n(in7, -239))),
            in7,
        );
        t29a = i32x4_add(
            vrshr::<12>(i32x4_add(vmul_n(in9, -393), vmul_n(in23, 1751))),
            in9,
        );
        t30a = i32x4_add(
            vrshr::<12>(i32x4_add(vmul_n(in17, 2751), vmul_n(in15, -1061))),
            in15,
        );
        t31a = i32x4_add(
            vrshr::<12>(i32x4_add(vmul_n(in1, -5), vmul_n(in31, 201))),
            in1,
        );
    }

    let mut t16: v128 = i32x4_min(i32x4_max(i32x4_add(t16a, t17a), min), max);
    let mut t17: v128 = i32x4_min(i32x4_max(i32x4_sub(t16a, t17a), min), max);
    let mut t18: v128 = i32x4_min(i32x4_max(i32x4_sub(t19a, t18a), min), max);
    let mut t19: v128 = i32x4_min(i32x4_max(i32x4_add(t19a, t18a), min), max);
    let mut t20: v128 = i32x4_min(i32x4_max(i32x4_add(t20a, t21a), min), max);
    let mut t21: v128 = i32x4_min(i32x4_max(i32x4_sub(t20a, t21a), min), max);
    let mut t22: v128 = i32x4_min(i32x4_max(i32x4_sub(t23a, t22a), min), max);
    let mut t23: v128 = i32x4_min(i32x4_max(i32x4_add(t23a, t22a), min), max);
    let mut t24: v128 = i32x4_min(i32x4_max(i32x4_add(t24a, t25a), min), max);
    let mut t25: v128 = i32x4_min(i32x4_max(i32x4_sub(t24a, t25a), min), max);
    let mut t26: v128 = i32x4_min(i32x4_max(i32x4_sub(t27a, t26a), min), max);
    let mut t27: v128 = i32x4_min(i32x4_max(i32x4_add(t27a, t26a), min), max);
    let mut t28: v128 = i32x4_min(i32x4_max(i32x4_add(t28a, t29a), min), max);
    let mut t29: v128 = i32x4_min(i32x4_max(i32x4_sub(t28a, t29a), min), max);
    let mut t30: v128 = i32x4_min(i32x4_max(i32x4_sub(t31a, t30a), min), max);
    let mut t31: v128 = i32x4_min(i32x4_max(i32x4_add(t31a, t30a), min), max);

    t17a = i32x4_sub(
        vrshr::<12>(i32x4_sub(vmul_n(t30, 799), vmul_n(t17, -79))),
        t17,
    );
    t30a = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t30, -79), vmul_n(t17, 799))),
        t30,
    );
    t18a = i32x4_sub(
        vrshr::<12>(i32x4_neg(i32x4_add(vmul_n(t29, -79), vmul_n(t18, 799)))),
        t29,
    );
    t29a = i32x4_sub(
        vrshr::<12>(i32x4_sub(vmul_n(t29, 799), vmul_n(t18, -79))),
        t18,
    );
    t21a = vrshr::<11>(i32x4_sub(vmul_n(t26, 1703), vmul_n(t21, 1138)));
    t26a = vrshr::<11>(i32x4_add(vmul_n(t26, 1138), vmul_n(t21, 1703)));
    t22a = vrshr::<11>(i32x4_neg(i32x4_add(vmul_n(t25, 1138), vmul_n(t22, 1703))));
    t25a = vrshr::<11>(i32x4_sub(vmul_n(t25, 1703), vmul_n(t22, 1138)));

    t16a = i32x4_min(i32x4_max(i32x4_add(t16, t19), min), max);
    t17 = i32x4_min(i32x4_max(i32x4_add(t17a, t18a), min), max);
    t18 = i32x4_min(i32x4_max(i32x4_sub(t17a, t18a), min), max);
    t19a = i32x4_min(i32x4_max(i32x4_sub(t16, t19), min), max);
    t20a = i32x4_min(i32x4_max(i32x4_sub(t23, t20), min), max);
    t21 = i32x4_min(i32x4_max(i32x4_sub(t22a, t21a), min), max);
    t22 = i32x4_min(i32x4_max(i32x4_add(t22a, t21a), min), max);
    t23a = i32x4_min(i32x4_max(i32x4_add(t23, t20), min), max);
    t24a = i32x4_min(i32x4_max(i32x4_add(t24, t27), min), max);
    t25 = i32x4_min(i32x4_max(i32x4_add(t25a, t26a), min), max);
    t26 = i32x4_min(i32x4_max(i32x4_sub(t25a, t26a), min), max);
    t27a = i32x4_min(i32x4_max(i32x4_sub(t24, t27), min), max);
    t28a = i32x4_min(i32x4_max(i32x4_sub(t31, t28), min), max);
    t29 = i32x4_min(i32x4_max(i32x4_sub(t30a, t29a), min), max);
    t30 = i32x4_min(i32x4_max(i32x4_add(t30a, t29a), min), max);
    t31a = i32x4_min(i32x4_max(i32x4_add(t31, t28), min), max);

    t18a = i32x4_sub(
        vrshr::<12>(i32x4_sub(vmul_n(t29, 1567), vmul_n(t18, -312))),
        t18,
    );
    t29a = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t29, -312), vmul_n(t18, 1567))),
        t29,
    );
    t19 = i32x4_sub(
        vrshr::<12>(i32x4_sub(vmul_n(t28a, 1567), vmul_n(t19a, -312))),
        t19a,
    );
    t28 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t28a, -312), vmul_n(t19a, 1567))),
        t28a,
    );
    t20 = i32x4_sub(
        vrshr::<12>(i32x4_neg(i32x4_add(vmul_n(t27a, -312), vmul_n(t20a, 1567)))),
        t27a,
    );
    t27 = i32x4_sub(
        vrshr::<12>(i32x4_sub(vmul_n(t27a, 1567), vmul_n(t20a, -312))),
        t20a,
    );
    t21a = i32x4_sub(
        vrshr::<12>(i32x4_neg(i32x4_add(vmul_n(t26, -312), vmul_n(t21, 1567)))),
        t26,
    );
    t26a = i32x4_sub(
        vrshr::<12>(i32x4_sub(vmul_n(t26, 1567), vmul_n(t21, -312))),
        t21,
    );

    t16 = i32x4_min(i32x4_max(i32x4_add(t16a, t23a), min), max);
    t17a = i32x4_min(i32x4_max(i32x4_add(t17, t22), min), max);
    t18 = i32x4_min(i32x4_max(i32x4_add(t18a, t21a), min), max);
    t19a = i32x4_min(i32x4_max(i32x4_add(t19, t20), min), max);
    t20a = i32x4_min(i32x4_max(i32x4_sub(t19, t20), min), max);
    t21 = i32x4_min(i32x4_max(i32x4_sub(t18a, t21a), min), max);
    t22a = i32x4_min(i32x4_max(i32x4_sub(t17, t22), min), max);
    t23 = i32x4_min(i32x4_max(i32x4_sub(t16a, t23a), min), max);
    t24 = i32x4_min(i32x4_max(i32x4_sub(t31a, t24a), min), max);
    t25a = i32x4_min(i32x4_max(i32x4_sub(t30, t25), min), max);
    t26 = i32x4_min(i32x4_max(i32x4_sub(t29a, t26a), min), max);
    t27a = i32x4_min(i32x4_max(i32x4_sub(t28, t27), min), max);
    t28a = i32x4_min(i32x4_max(i32x4_add(t28, t27), min), max);
    t29 = i32x4_min(i32x4_max(i32x4_add(t29a, t26a), min), max);
    t30a = i32x4_min(i32x4_max(i32x4_add(t30, t25), min), max);
    t31 = i32x4_min(i32x4_max(i32x4_add(t31a, t24a), min), max);

    t20 = vrshr::<8>(vmul_n(i32x4_sub(t27a, t20a), 181));
    t27 = vrshr::<8>(vmul_n(i32x4_add(t27a, t20a), 181));
    t21a = vrshr::<8>(vmul_n(i32x4_sub(t26, t21), 181));
    t26a = vrshr::<8>(vmul_n(i32x4_add(t26, t21), 181));
    t22 = vrshr::<8>(vmul_n(i32x4_sub(t25a, t22a), 181));
    t25 = vrshr::<8>(vmul_n(i32x4_add(t25a, t22a), 181));
    t23a = vrshr::<8>(vmul_n(i32x4_sub(t24, t23), 181));
    t24a = vrshr::<8>(vmul_n(i32x4_add(t24, t23), 181));

    let t0: v128 = c[0];
    let t1: v128 = c[2 * stride];
    let t2: v128 = c[4 * stride];
    let t3: v128 = c[6 * stride];
    let t4: v128 = c[8 * stride];
    let t5: v128 = c[10 * stride];
    let t6: v128 = c[12 * stride];
    let t7: v128 = c[14 * stride];
    let t8: v128 = c[16 * stride];
    let t9: v128 = c[18 * stride];
    let t10: v128 = c[20 * stride];
    let t11: v128 = c[22 * stride];
    let t12: v128 = c[24 * stride];
    let t13: v128 = c[26 * stride];
    let t14: v128 = c[28 * stride];
    let t15: v128 = c[30 * stride];

    c[0] = i32x4_min(i32x4_max(i32x4_add(t0, t31), min), max);
    c[stride] = i32x4_min(i32x4_max(i32x4_add(t1, t30a), min), max);
    c[2 * stride] = i32x4_min(i32x4_max(i32x4_add(t2, t29), min), max);
    c[3 * stride] = i32x4_min(i32x4_max(i32x4_add(t3, t28a), min), max);
    c[4 * stride] = i32x4_min(i32x4_max(i32x4_add(t4, t27), min), max);
    c[5 * stride] = i32x4_min(i32x4_max(i32x4_add(t5, t26a), min), max);
    c[6 * stride] = i32x4_min(i32x4_max(i32x4_add(t6, t25), min), max);
    c[7 * stride] = i32x4_min(i32x4_max(i32x4_add(t7, t24a), min), max);
    c[8 * stride] = i32x4_min(i32x4_max(i32x4_add(t8, t23a), min), max);
    c[9 * stride] = i32x4_min(i32x4_max(i32x4_add(t9, t22), min), max);
    c[10 * stride] = i32x4_min(i32x4_max(i32x4_add(t10, t21a), min), max);
    c[11 * stride] = i32x4_min(i32x4_max(i32x4_add(t11, t20), min), max);
    c[12 * stride] = i32x4_min(i32x4_max(i32x4_add(t12, t19a), min), max);
    c[13 * stride] = i32x4_min(i32x4_max(i32x4_add(t13, t18), min), max);
    c[14 * stride] = i32x4_min(i32x4_max(i32x4_add(t14, t17a), min), max);
    c[15 * stride] = i32x4_min(i32x4_max(i32x4_add(t15, t16), min), max);
    c[16 * stride] = i32x4_min(i32x4_max(i32x4_sub(t15, t16), min), max);
    c[17 * stride] = i32x4_min(i32x4_max(i32x4_sub(t14, t17a), min), max);
    c[18 * stride] = i32x4_min(i32x4_max(i32x4_sub(t13, t18), min), max);
    c[19 * stride] = i32x4_min(i32x4_max(i32x4_sub(t12, t19a), min), max);
    c[20 * stride] = i32x4_min(i32x4_max(i32x4_sub(t11, t20), min), max);
    c[21 * stride] = i32x4_min(i32x4_max(i32x4_sub(t10, t21a), min), max);
    c[22 * stride] = i32x4_min(i32x4_max(i32x4_sub(t9, t22), min), max);
    c[23 * stride] = i32x4_min(i32x4_max(i32x4_sub(t8, t23a), min), max);
    c[24 * stride] = i32x4_min(i32x4_max(i32x4_sub(t7, t24a), min), max);
    c[25 * stride] = i32x4_min(i32x4_max(i32x4_sub(t6, t25), min), max);
    c[26 * stride] = i32x4_min(i32x4_max(i32x4_sub(t5, t26a), min), max);
    c[27 * stride] = i32x4_min(i32x4_max(i32x4_sub(t4, t27), min), max);
    c[28 * stride] = i32x4_min(i32x4_max(i32x4_sub(t3, t28a), min), max);
    c[29 * stride] = i32x4_min(i32x4_max(i32x4_sub(t2, t29), min), max);
    c[30 * stride] = i32x4_min(i32x4_max(i32x4_sub(t1, t30a), min), max);
    c[31 * stride] = i32x4_min(i32x4_max(i32x4_sub(t0, t31), min), max);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_dct64_1d_internal(c: &mut [v128], stride: usize, min: v128, max: v128) {
    inv_dct32_1d_internal(c, stride << 1, min, max, true);

    let in1: v128 = c[stride];
    let in3: v128 = c[3 * stride];
    let in5: v128 = c[5 * stride];
    let in7: v128 = c[7 * stride];
    let in9: v128 = c[9 * stride];
    let in11: v128 = c[11 * stride];
    let in13: v128 = c[13 * stride];
    let in15: v128 = c[15 * stride];
    let in17: v128 = c[17 * stride];
    let in19: v128 = c[19 * stride];
    let in21: v128 = c[21 * stride];
    let in23: v128 = c[23 * stride];
    let in25: v128 = c[25 * stride];
    let in27: v128 = c[27 * stride];
    let in29: v128 = c[29 * stride];
    let in31: v128 = c[31 * stride];

    let mut t32a: v128 = vrshr::<12>(vmul_n(in1, 101));
    let mut t33a: v128 = vrshr::<12>(vmul_n(in31, -2824));
    let mut t34a: v128 = vrshr::<12>(vmul_n(in17, 1660));
    let mut t35a: v128 = vrshr::<12>(vmul_n(in15, -1474));
    let mut t36a: v128 = vrshr::<12>(vmul_n(in9, 897));
    let mut t37a: v128 = vrshr::<12>(vmul_n(in23, -2191));
    let mut t38a: v128 = vrshr::<12>(vmul_n(in25, 2359));
    let mut t39a: v128 = vrshr::<12>(vmul_n(in7, -700));
    let mut t40a: v128 = vrshr::<12>(vmul_n(in5, 501));
    let mut t41a: v128 = vrshr::<12>(vmul_n(in27, -2520));
    let mut t42a: v128 = vrshr::<12>(vmul_n(in21, 2019));
    let mut t43a: v128 = vrshr::<12>(vmul_n(in11, -1092));
    let mut t44a: v128 = vrshr::<12>(vmul_n(in13, 1285));
    let mut t45a: v128 = vrshr::<12>(vmul_n(in19, -1842));
    let mut t46a: v128 = vrshr::<12>(vmul_n(in29, 2675));
    let mut t47a: v128 = vrshr::<12>(vmul_n(in3, -301));
    let mut t48a: v128 = vrshr::<12>(vmul_n(in3, 4085));
    let mut t49a: v128 = vrshr::<12>(vmul_n(in29, 3102));
    let mut t50a: v128 = vrshr::<12>(vmul_n(in19, 3659));
    let mut t51a: v128 = vrshr::<12>(vmul_n(in13, 3889));
    let mut t52a: v128 = vrshr::<12>(vmul_n(in11, 3948));
    let mut t53a: v128 = vrshr::<12>(vmul_n(in21, 3564));
    let mut t54a: v128 = vrshr::<12>(vmul_n(in27, 3229));
    let mut t55a: v128 = vrshr::<12>(vmul_n(in5, 4065));
    let mut t56a: v128 = vrshr::<12>(vmul_n(in7, 4036));
    let mut t57a: v128 = vrshr::<12>(vmul_n(in25, 3349));
    let mut t58a: v128 = vrshr::<12>(vmul_n(in23, 3461));
    let mut t59a: v128 = vrshr::<12>(vmul_n(in9, 3996));
    let mut t60a: v128 = vrshr::<12>(vmul_n(in15, 3822));
    let mut t61a: v128 = vrshr::<12>(vmul_n(in17, 3745));
    let mut t62a: v128 = vrshr::<12>(vmul_n(in31, 2967));
    let mut t63a: v128 = vrshr::<12>(vmul_n(in1, 4095));

    let mut t32: v128 = i32x4_min(i32x4_max(i32x4_add(t32a, t33a), min), max);
    let mut t33: v128 = i32x4_min(i32x4_max(i32x4_sub(t32a, t33a), min), max);
    let mut t34: v128 = i32x4_min(i32x4_max(i32x4_sub(t35a, t34a), min), max);
    let mut t35: v128 = i32x4_min(i32x4_max(i32x4_add(t35a, t34a), min), max);
    let mut t36: v128 = i32x4_min(i32x4_max(i32x4_add(t36a, t37a), min), max);
    let mut t37: v128 = i32x4_min(i32x4_max(i32x4_sub(t36a, t37a), min), max);
    let mut t38: v128 = i32x4_min(i32x4_max(i32x4_sub(t39a, t38a), min), max);
    let mut t39: v128 = i32x4_min(i32x4_max(i32x4_add(t39a, t38a), min), max);
    let mut t40: v128 = i32x4_min(i32x4_max(i32x4_add(t40a, t41a), min), max);
    let mut t41: v128 = i32x4_min(i32x4_max(i32x4_sub(t40a, t41a), min), max);
    let mut t42: v128 = i32x4_min(i32x4_max(i32x4_sub(t43a, t42a), min), max);
    let mut t43: v128 = i32x4_min(i32x4_max(i32x4_add(t43a, t42a), min), max);
    let mut t44: v128 = i32x4_min(i32x4_max(i32x4_add(t44a, t45a), min), max);
    let mut t45: v128 = i32x4_min(i32x4_max(i32x4_sub(t44a, t45a), min), max);
    let mut t46: v128 = i32x4_min(i32x4_max(i32x4_sub(t47a, t46a), min), max);
    let mut t47: v128 = i32x4_min(i32x4_max(i32x4_add(t47a, t46a), min), max);
    let mut t48: v128 = i32x4_min(i32x4_max(i32x4_add(t48a, t49a), min), max);
    let mut t49: v128 = i32x4_min(i32x4_max(i32x4_sub(t48a, t49a), min), max);
    let mut t50: v128 = i32x4_min(i32x4_max(i32x4_sub(t51a, t50a), min), max);
    let mut t51: v128 = i32x4_min(i32x4_max(i32x4_add(t51a, t50a), min), max);
    let mut t52: v128 = i32x4_min(i32x4_max(i32x4_add(t52a, t53a), min), max);
    let mut t53: v128 = i32x4_min(i32x4_max(i32x4_sub(t52a, t53a), min), max);
    let mut t54: v128 = i32x4_min(i32x4_max(i32x4_sub(t55a, t54a), min), max);
    let mut t55: v128 = i32x4_min(i32x4_max(i32x4_add(t55a, t54a), min), max);
    let mut t56: v128 = i32x4_min(i32x4_max(i32x4_add(t56a, t57a), min), max);
    let mut t57: v128 = i32x4_min(i32x4_max(i32x4_sub(t56a, t57a), min), max);
    let mut t58: v128 = i32x4_min(i32x4_max(i32x4_sub(t59a, t58a), min), max);
    let mut t59: v128 = i32x4_min(i32x4_max(i32x4_add(t59a, t58a), min), max);
    let mut t60: v128 = i32x4_min(i32x4_max(i32x4_add(t60a, t61a), min), max);
    let mut t61: v128 = i32x4_min(i32x4_max(i32x4_sub(t60a, t61a), min), max);
    let mut t62: v128 = i32x4_min(i32x4_max(i32x4_sub(t63a, t62a), min), max);
    let mut t63: v128 = i32x4_min(i32x4_max(i32x4_add(t63a, t62a), min), max);

    t33a = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t33, 20), vmul_n(t62, 401))),
        t33,
    );
    t34a = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t34, -401), vmul_n(t61, 20))),
        t61,
    );
    t37a = vrshr::<11>(i32x4_add(vmul_n(t37, -1299), vmul_n(t58, 1583)));
    t38a = vrshr::<11>(i32x4_add(vmul_n(t38, -1583), vmul_n(t57, -1299)));
    t41a = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t41, 484), vmul_n(t54, 1931))),
        t41,
    );
    t42a = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t42, -1931), vmul_n(t53, 484))),
        t53,
    );
    t45a = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t45, -1189), vmul_n(t50, -176))),
        t50,
    );
    t46a = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t46, 176), vmul_n(t49, -1189))),
        t46,
    );
    t49a = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t46, -1189), vmul_n(t49, -176))),
        t49,
    );
    t50a = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t45, -176), vmul_n(t50, 1189))),
        t45,
    );
    t53a = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t42, 484), vmul_n(t53, 1931))),
        t42,
    );
    t54a = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t41, 1931), vmul_n(t54, -484))),
        t54,
    );
    t57a = vrshr::<11>(i32x4_add(vmul_n(t38, -1299), vmul_n(t57, 1583)));
    t58a = vrshr::<11>(i32x4_add(vmul_n(t37, 1583), vmul_n(t58, 1299)));
    t61a = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t34, 20), vmul_n(t61, 401))),
        t34,
    );
    t62a = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t33, 401), vmul_n(t62, -20))),
        t62,
    );

    t32a = i32x4_min(i32x4_max(i32x4_add(t32, t35), min), max);
    t33 = i32x4_min(i32x4_max(i32x4_add(t33a, t34a), min), max);
    t34 = i32x4_min(i32x4_max(i32x4_sub(t33a, t34a), min), max);
    t35a = i32x4_min(i32x4_max(i32x4_sub(t32, t35), min), max);
    t36a = i32x4_min(i32x4_max(i32x4_sub(t39, t36), min), max);
    t37 = i32x4_min(i32x4_max(i32x4_sub(t38a, t37a), min), max);
    t38 = i32x4_min(i32x4_max(i32x4_add(t38a, t37a), min), max);
    t39a = i32x4_min(i32x4_max(i32x4_add(t39, t36), min), max);
    t40a = i32x4_min(i32x4_max(i32x4_add(t40, t43), min), max);
    t41 = i32x4_min(i32x4_max(i32x4_add(t41a, t42a), min), max);
    t42 = i32x4_min(i32x4_max(i32x4_sub(t41a, t42a), min), max);
    t43a = i32x4_min(i32x4_max(i32x4_sub(t40, t43), min), max);
    t44a = i32x4_min(i32x4_max(i32x4_sub(t47, t44), min), max);
    t45 = i32x4_min(i32x4_max(i32x4_sub(t46a, t45a), min), max);
    t46 = i32x4_min(i32x4_max(i32x4_add(t46a, t45a), min), max);
    t47a = i32x4_min(i32x4_max(i32x4_add(t47, t44), min), max);
    t48a = i32x4_min(i32x4_max(i32x4_add(t48, t51), min), max);
    t49 = i32x4_min(i32x4_max(i32x4_add(t49a, t50a), min), max);
    t50 = i32x4_min(i32x4_max(i32x4_sub(t49a, t50a), min), max);
    t51a = i32x4_min(i32x4_max(i32x4_sub(t48, t51), min), max);
    t52a = i32x4_min(i32x4_max(i32x4_sub(t55, t52), min), max);
    t53 = i32x4_min(i32x4_max(i32x4_sub(t54a, t53a), min), max);
    t54 = i32x4_min(i32x4_max(i32x4_add(t54a, t53a), min), max);
    t55a = i32x4_min(i32x4_max(i32x4_add(t55, t52), min), max);
    t56a = i32x4_min(i32x4_max(i32x4_add(t56, t59), min), max);
    t57 = i32x4_min(i32x4_max(i32x4_add(t57a, t58a), min), max);
    t58 = i32x4_min(i32x4_max(i32x4_sub(t57a, t58a), min), max);
    t59a = i32x4_min(i32x4_max(i32x4_sub(t56, t59), min), max);
    t60a = i32x4_min(i32x4_max(i32x4_sub(t63, t60), min), max);
    t61 = i32x4_min(i32x4_max(i32x4_sub(t62a, t61a), min), max);
    t62 = i32x4_min(i32x4_max(i32x4_add(t62a, t61a), min), max);
    t63a = i32x4_min(i32x4_max(i32x4_add(t63, t60), min), max);

    t34a = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t34, 79), vmul_n(t61, 799))),
        t34,
    );
    t35 = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t35a, 79), vmul_n(t60a, 799))),
        t35a,
    );
    t36 = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t36a, -799), vmul_n(t59a, 79))),
        t59a,
    );
    t37a = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t37, -799), vmul_n(t58, 79))),
        t58,
    );
    t42a = vrshr::<11>(i32x4_add(vmul_n(t42, -1138), vmul_n(t53, 1703)));
    t43 = vrshr::<11>(i32x4_add(vmul_n(t43a, -1138), vmul_n(t52a, 1703)));
    t44 = vrshr::<11>(i32x4_add(vmul_n(t44a, -1703), vmul_n(t51a, -1138)));
    t45a = vrshr::<11>(i32x4_add(vmul_n(t45, -1703), vmul_n(t50, -1138)));
    t50a = vrshr::<11>(i32x4_add(vmul_n(t45, -1138), vmul_n(t50, 1703)));
    t51 = vrshr::<11>(i32x4_add(vmul_n(t44a, -1138), vmul_n(t51a, 1703)));
    t52 = vrshr::<11>(i32x4_add(vmul_n(t43a, 1703), vmul_n(t52a, 1138)));
    t53a = vrshr::<11>(i32x4_add(vmul_n(t42, 1703), vmul_n(t53, 1138)));
    t58a = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t37, 79), vmul_n(t58, 799))),
        t37,
    );
    t59 = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t36a, 79), vmul_n(t59a, 799))),
        t36a,
    );
    t60 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t35a, 799), vmul_n(t60a, -79))),
        t60a,
    );
    t61a = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t34, 799), vmul_n(t61, -79))),
        t61,
    );

    t32 = i32x4_min(i32x4_max(i32x4_add(t32a, t39a), min), max);
    t33a = i32x4_min(i32x4_max(i32x4_add(t33, t38), min), max);
    t34 = i32x4_min(i32x4_max(i32x4_add(t34a, t37a), min), max);
    t35a = i32x4_min(i32x4_max(i32x4_add(t35, t36), min), max);
    t36a = i32x4_min(i32x4_max(i32x4_sub(t35, t36), min), max);
    t37 = i32x4_min(i32x4_max(i32x4_sub(t34a, t37a), min), max);
    t38a = i32x4_min(i32x4_max(i32x4_sub(t33, t38), min), max);
    t39 = i32x4_min(i32x4_max(i32x4_sub(t32a, t39a), min), max);
    t40 = i32x4_min(i32x4_max(i32x4_sub(t47a, t40a), min), max);
    t41a = i32x4_min(i32x4_max(i32x4_sub(t46, t41), min), max);
    t42 = i32x4_min(i32x4_max(i32x4_sub(t45a, t42a), min), max);
    t43a = i32x4_min(i32x4_max(i32x4_sub(t44, t43), min), max);
    t44a = i32x4_min(i32x4_max(i32x4_add(t44, t43), min), max);
    t45 = i32x4_min(i32x4_max(i32x4_add(t45a, t42a), min), max);
    t46a = i32x4_min(i32x4_max(i32x4_add(t46, t41), min), max);
    t47 = i32x4_min(i32x4_max(i32x4_add(t47a, t40a), min), max);
    t48 = i32x4_min(i32x4_max(i32x4_add(t48a, t55a), min), max);
    t49a = i32x4_min(i32x4_max(i32x4_add(t49, t54), min), max);
    t50 = i32x4_min(i32x4_max(i32x4_add(t50a, t53a), min), max);
    t51a = i32x4_min(i32x4_max(i32x4_add(t51, t52), min), max);
    t52a = i32x4_min(i32x4_max(i32x4_sub(t51, t52), min), max);
    t53 = i32x4_min(i32x4_max(i32x4_sub(t50a, t53a), min), max);
    t54a = i32x4_min(i32x4_max(i32x4_sub(t49, t54), min), max);
    t55 = i32x4_min(i32x4_max(i32x4_sub(t48a, t55a), min), max);
    t56 = i32x4_min(i32x4_max(i32x4_sub(t63a, t56a), min), max);
    t57a = i32x4_min(i32x4_max(i32x4_sub(t62, t57), min), max);
    t58 = i32x4_min(i32x4_max(i32x4_sub(t61a, t58a), min), max);
    t59a = i32x4_min(i32x4_max(i32x4_sub(t60, t59), min), max);
    t60a = i32x4_min(i32x4_max(i32x4_add(t60, t59), min), max);
    t61 = i32x4_min(i32x4_max(i32x4_add(t61a, t58a), min), max);
    t62a = i32x4_min(i32x4_max(i32x4_add(t62, t57), min), max);
    t63 = i32x4_min(i32x4_max(i32x4_add(t63a, t56a), min), max);

    t36 = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t36a, 312), vmul_n(t59a, 1567))),
        t36a,
    );
    t37a = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t37, 312), vmul_n(t58, 1567))),
        t37,
    );
    t38 = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t38a, 312), vmul_n(t57a, 1567))),
        t38a,
    );
    t39a = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t39, 312), vmul_n(t56, 1567))),
        t39,
    );
    t40a = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t40, -1567), vmul_n(t55, 312))),
        t55,
    );
    t41 = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t41a, -1567), vmul_n(t54a, 312))),
        t54a,
    );
    t42a = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t42, -1567), vmul_n(t53, 312))),
        t53,
    );
    t43 = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t43a, -1567), vmul_n(t52a, 312))),
        t52a,
    );
    t52 = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t43a, 312), vmul_n(t52a, 1567))),
        t43a,
    );
    t53a = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t42, 312), vmul_n(t53, 1567))),
        t42,
    );
    t54 = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t41a, 312), vmul_n(t54a, 1567))),
        t41a,
    );
    t55a = i32x4_sub(
        vrshr::<12>(i32x4_add(vmul_n(t40, 312), vmul_n(t55, 1567))),
        t40,
    );
    t56a = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t39, 1567), vmul_n(t56, -312))),
        t56,
    );
    t57 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t38a, 1567), vmul_n(t57a, -312))),
        t57a,
    );
    t58a = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t37, 1567), vmul_n(t58, -312))),
        t58,
    );
    t59 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t36a, 1567), vmul_n(t59a, -312))),
        t59a,
    );

    t32a = i32x4_min(i32x4_max(i32x4_add(t32, t47), min), max);
    t33 = i32x4_min(i32x4_max(i32x4_add(t33a, t46a), min), max);
    t34a = i32x4_min(i32x4_max(i32x4_add(t34, t45), min), max);
    t35 = i32x4_min(i32x4_max(i32x4_add(t35a, t44a), min), max);
    t36a = i32x4_min(i32x4_max(i32x4_add(t36, t43), min), max);
    t37 = i32x4_min(i32x4_max(i32x4_add(t37a, t42a), min), max);
    t38a = i32x4_min(i32x4_max(i32x4_add(t38, t41), min), max);
    t39 = i32x4_min(i32x4_max(i32x4_add(t39a, t40a), min), max);
    t40 = i32x4_min(i32x4_max(i32x4_sub(t39a, t40a), min), max);
    t41a = i32x4_min(i32x4_max(i32x4_sub(t38, t41), min), max);
    t42 = i32x4_min(i32x4_max(i32x4_sub(t37a, t42a), min), max);
    t43a = i32x4_min(i32x4_max(i32x4_sub(t36, t43), min), max);
    t44 = i32x4_min(i32x4_max(i32x4_sub(t35a, t44a), min), max);
    t45a = i32x4_min(i32x4_max(i32x4_sub(t34, t45), min), max);
    t46 = i32x4_min(i32x4_max(i32x4_sub(t33a, t46a), min), max);
    t47a = i32x4_min(i32x4_max(i32x4_sub(t32, t47), min), max);
    t48a = i32x4_min(i32x4_max(i32x4_sub(t63, t48), min), max);
    t49 = i32x4_min(i32x4_max(i32x4_sub(t62a, t49a), min), max);
    t50a = i32x4_min(i32x4_max(i32x4_sub(t61, t50), min), max);
    t51 = i32x4_min(i32x4_max(i32x4_sub(t60a, t51a), min), max);
    t52a = i32x4_min(i32x4_max(i32x4_sub(t59, t52), min), max);
    t53 = i32x4_min(i32x4_max(i32x4_sub(t58a, t53a), min), max);
    t54a = i32x4_min(i32x4_max(i32x4_sub(t57, t54), min), max);
    t55 = i32x4_min(i32x4_max(i32x4_sub(t56a, t55a), min), max);
    t56 = i32x4_min(i32x4_max(i32x4_add(t56a, t55a), min), max);
    t57a = i32x4_min(i32x4_max(i32x4_add(t57, t54), min), max);
    t58 = i32x4_min(i32x4_max(i32x4_add(t58a, t53a), min), max);
    t59a = i32x4_min(i32x4_max(i32x4_add(t59, t52), min), max);
    t60 = i32x4_min(i32x4_max(i32x4_add(t60a, t51a), min), max);
    t61a = i32x4_min(i32x4_max(i32x4_add(t61, t50), min), max);
    t62 = i32x4_min(i32x4_max(i32x4_add(t62a, t49a), min), max);
    t63a = i32x4_min(i32x4_max(i32x4_add(t63, t48), min), max);

    t40a = vrshr::<8>(vmul_n(i32x4_sub(t55, t40), 181));
    t41 = vrshr::<8>(vmul_n(i32x4_sub(t54a, t41a), 181));
    t42a = vrshr::<8>(vmul_n(i32x4_sub(t53, t42), 181));
    t43 = vrshr::<8>(vmul_n(i32x4_sub(t52a, t43a), 181));
    t44a = vrshr::<8>(vmul_n(i32x4_sub(t51, t44), 181));
    t45 = vrshr::<8>(vmul_n(i32x4_sub(t50a, t45a), 181));
    t46a = vrshr::<8>(vmul_n(i32x4_sub(t49, t46), 181));
    t47 = vrshr::<8>(vmul_n(i32x4_sub(t48a, t47a), 181));
    t48 = vrshr::<8>(vmul_n(i32x4_add(t47a, t48a), 181));
    t49a = vrshr::<8>(vmul_n(i32x4_add(t46, t49), 181));
    t50 = vrshr::<8>(vmul_n(i32x4_add(t45a, t50a), 181));
    t51a = vrshr::<8>(vmul_n(i32x4_add(t44, t51), 181));
    t52 = vrshr::<8>(vmul_n(i32x4_add(t43a, t52a), 181));
    t53a = vrshr::<8>(vmul_n(i32x4_add(t42, t53), 181));
    t54 = vrshr::<8>(vmul_n(i32x4_add(t41a, t54a), 181));
    t55a = vrshr::<8>(vmul_n(i32x4_add(t40, t55), 181));

    let t0: v128 = c[0];
    let t1: v128 = c[2 * stride];
    let t2: v128 = c[4 * stride];
    let t3: v128 = c[6 * stride];
    let t4: v128 = c[8 * stride];
    let t5: v128 = c[10 * stride];
    let t6: v128 = c[12 * stride];
    let t7: v128 = c[14 * stride];
    let t8: v128 = c[16 * stride];
    let t9: v128 = c[18 * stride];
    let t10: v128 = c[20 * stride];
    let t11: v128 = c[22 * stride];
    let t12: v128 = c[24 * stride];
    let t13: v128 = c[26 * stride];
    let t14: v128 = c[28 * stride];
    let t15: v128 = c[30 * stride];
    let t16: v128 = c[32 * stride];
    let t17: v128 = c[34 * stride];
    let t18: v128 = c[36 * stride];
    let t19: v128 = c[38 * stride];
    let t20: v128 = c[40 * stride];
    let t21: v128 = c[42 * stride];
    let t22: v128 = c[44 * stride];
    let t23: v128 = c[46 * stride];
    let t24: v128 = c[48 * stride];
    let t25: v128 = c[50 * stride];
    let t26: v128 = c[52 * stride];
    let t27: v128 = c[54 * stride];
    let t28: v128 = c[56 * stride];
    let t29: v128 = c[58 * stride];
    let t30: v128 = c[60 * stride];
    let t31: v128 = c[62 * stride];

    c[0] = i32x4_min(i32x4_max(i32x4_add(t0, t63a), min), max);
    c[stride] = i32x4_min(i32x4_max(i32x4_add(t1, t62), min), max);
    c[2 * stride] = i32x4_min(i32x4_max(i32x4_add(t2, t61a), min), max);
    c[3 * stride] = i32x4_min(i32x4_max(i32x4_add(t3, t60), min), max);
    c[4 * stride] = i32x4_min(i32x4_max(i32x4_add(t4, t59a), min), max);
    c[5 * stride] = i32x4_min(i32x4_max(i32x4_add(t5, t58), min), max);
    c[6 * stride] = i32x4_min(i32x4_max(i32x4_add(t6, t57a), min), max);
    c[7 * stride] = i32x4_min(i32x4_max(i32x4_add(t7, t56), min), max);
    c[8 * stride] = i32x4_min(i32x4_max(i32x4_add(t8, t55a), min), max);
    c[9 * stride] = i32x4_min(i32x4_max(i32x4_add(t9, t54), min), max);
    c[10 * stride] = i32x4_min(i32x4_max(i32x4_add(t10, t53a), min), max);
    c[11 * stride] = i32x4_min(i32x4_max(i32x4_add(t11, t52), min), max);
    c[12 * stride] = i32x4_min(i32x4_max(i32x4_add(t12, t51a), min), max);
    c[13 * stride] = i32x4_min(i32x4_max(i32x4_add(t13, t50), min), max);
    c[14 * stride] = i32x4_min(i32x4_max(i32x4_add(t14, t49a), min), max);
    c[15 * stride] = i32x4_min(i32x4_max(i32x4_add(t15, t48), min), max);
    c[16 * stride] = i32x4_min(i32x4_max(i32x4_add(t16, t47), min), max);
    c[17 * stride] = i32x4_min(i32x4_max(i32x4_add(t17, t46a), min), max);
    c[18 * stride] = i32x4_min(i32x4_max(i32x4_add(t18, t45), min), max);
    c[19 * stride] = i32x4_min(i32x4_max(i32x4_add(t19, t44a), min), max);
    c[20 * stride] = i32x4_min(i32x4_max(i32x4_add(t20, t43), min), max);
    c[21 * stride] = i32x4_min(i32x4_max(i32x4_add(t21, t42a), min), max);
    c[22 * stride] = i32x4_min(i32x4_max(i32x4_add(t22, t41), min), max);
    c[23 * stride] = i32x4_min(i32x4_max(i32x4_add(t23, t40a), min), max);
    c[24 * stride] = i32x4_min(i32x4_max(i32x4_add(t24, t39), min), max);
    c[25 * stride] = i32x4_min(i32x4_max(i32x4_add(t25, t38a), min), max);
    c[26 * stride] = i32x4_min(i32x4_max(i32x4_add(t26, t37), min), max);
    c[27 * stride] = i32x4_min(i32x4_max(i32x4_add(t27, t36a), min), max);
    c[28 * stride] = i32x4_min(i32x4_max(i32x4_add(t28, t35), min), max);
    c[29 * stride] = i32x4_min(i32x4_max(i32x4_add(t29, t34a), min), max);
    c[30 * stride] = i32x4_min(i32x4_max(i32x4_add(t30, t33), min), max);
    c[31 * stride] = i32x4_min(i32x4_max(i32x4_add(t31, t32a), min), max);
    c[32 * stride] = i32x4_min(i32x4_max(i32x4_sub(t31, t32a), min), max);
    c[33 * stride] = i32x4_min(i32x4_max(i32x4_sub(t30, t33), min), max);
    c[34 * stride] = i32x4_min(i32x4_max(i32x4_sub(t29, t34a), min), max);
    c[35 * stride] = i32x4_min(i32x4_max(i32x4_sub(t28, t35), min), max);
    c[36 * stride] = i32x4_min(i32x4_max(i32x4_sub(t27, t36a), min), max);
    c[37 * stride] = i32x4_min(i32x4_max(i32x4_sub(t26, t37), min), max);
    c[38 * stride] = i32x4_min(i32x4_max(i32x4_sub(t25, t38a), min), max);
    c[39 * stride] = i32x4_min(i32x4_max(i32x4_sub(t24, t39), min), max);
    c[40 * stride] = i32x4_min(i32x4_max(i32x4_sub(t23, t40a), min), max);
    c[41 * stride] = i32x4_min(i32x4_max(i32x4_sub(t22, t41), min), max);
    c[42 * stride] = i32x4_min(i32x4_max(i32x4_sub(t21, t42a), min), max);
    c[43 * stride] = i32x4_min(i32x4_max(i32x4_sub(t20, t43), min), max);
    c[44 * stride] = i32x4_min(i32x4_max(i32x4_sub(t19, t44a), min), max);
    c[45 * stride] = i32x4_min(i32x4_max(i32x4_sub(t18, t45), min), max);
    c[46 * stride] = i32x4_min(i32x4_max(i32x4_sub(t17, t46a), min), max);
    c[47 * stride] = i32x4_min(i32x4_max(i32x4_sub(t16, t47), min), max);
    c[48 * stride] = i32x4_min(i32x4_max(i32x4_sub(t15, t48), min), max);
    c[49 * stride] = i32x4_min(i32x4_max(i32x4_sub(t14, t49a), min), max);
    c[50 * stride] = i32x4_min(i32x4_max(i32x4_sub(t13, t50), min), max);
    c[51 * stride] = i32x4_min(i32x4_max(i32x4_sub(t12, t51a), min), max);
    c[52 * stride] = i32x4_min(i32x4_max(i32x4_sub(t11, t52), min), max);
    c[53 * stride] = i32x4_min(i32x4_max(i32x4_sub(t10, t53a), min), max);
    c[54 * stride] = i32x4_min(i32x4_max(i32x4_sub(t9, t54), min), max);
    c[55 * stride] = i32x4_min(i32x4_max(i32x4_sub(t8, t55a), min), max);
    c[56 * stride] = i32x4_min(i32x4_max(i32x4_sub(t7, t56), min), max);
    c[57 * stride] = i32x4_min(i32x4_max(i32x4_sub(t6, t57a), min), max);
    c[58 * stride] = i32x4_min(i32x4_max(i32x4_sub(t5, t58), min), max);
    c[59 * stride] = i32x4_min(i32x4_max(i32x4_sub(t4, t59a), min), max);
    c[60 * stride] = i32x4_min(i32x4_max(i32x4_sub(t3, t60), min), max);
    c[61 * stride] = i32x4_min(i32x4_max(i32x4_sub(t2, t61a), min), max);
    c[62 * stride] = i32x4_min(i32x4_max(i32x4_sub(t1, t62), min), max);
    c[63 * stride] = i32x4_min(i32x4_max(i32x4_sub(t0, t63a), min), max);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_adst4_1d_internal(c: &mut [v128], stride: usize, _min: v128, _max: v128, flip: bool) {
    let out_idx = |k: usize| {
        if flip {
            (4 - 1 - k) * stride
        } else {
            k * stride
        }
    };

    let in0: v128 = c[0];
    let in1: v128 = c[stride];
    let in2: v128 = c[2 * stride];
    let in3: v128 = c[3 * stride];

    c[out_idx(0)] = i32x4_add(
        i32x4_add(
            i32x4_add(
                vrshr::<12>(i32x4_add(
                    i32x4_add(
                        i32x4_add(vmul_n(in0, 1321), vmul_n(in2, -293)),
                        vmul_n(in3, -1614),
                    ),
                    vmul_n(in1, -752),
                )),
                in2,
            ),
            in3,
        ),
        in1,
    );
    c[out_idx(1)] = i32x4_add(
        i32x4_sub(
            i32x4_add(
                vrshr::<12>(i32x4_add(
                    i32x4_sub(
                        i32x4_sub(vmul_n(in0, -1614), vmul_n(in2, 1321)),
                        vmul_n(in3, -293),
                    ),
                    vmul_n(in1, -752),
                )),
                in0,
            ),
            in3,
        ),
        in1,
    );
    c[out_idx(2)] = vrshr::<8>(vmul_n(i32x4_add(i32x4_sub(in0, in2), in3), 209));
    c[out_idx(3)] = i32x4_sub(
        i32x4_add(
            i32x4_add(
                vrshr::<12>(i32x4_sub(
                    i32x4_sub(
                        i32x4_add(vmul_n(in0, -293), vmul_n(in2, -1614)),
                        vmul_n(in3, 1321),
                    ),
                    vmul_n(in1, -752),
                )),
                in0,
            ),
            in2,
        ),
        in1,
    );
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_adst8_1d_internal(c: &mut [v128], stride: usize, min: v128, max: v128, flip: bool) {
    let out_idx = |k: usize| {
        if flip {
            (8 - 1 - k) * stride
        } else {
            k * stride
        }
    };

    let in0: v128 = c[0];
    let in1: v128 = c[stride];
    let in2: v128 = c[2 * stride];
    let in3: v128 = c[3 * stride];
    let in4: v128 = c[4 * stride];
    let in5: v128 = c[5 * stride];
    let in6: v128 = c[6 * stride];
    let in7: v128 = c[7 * stride];

    let t0a: v128 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(in7, -20), vmul_n(in0, 401))),
        in7,
    );
    let t1a: v128 = i32x4_sub(
        vrshr::<12>(i32x4_sub(vmul_n(in7, 401), vmul_n(in0, -20))),
        in0,
    );
    let t2a: v128 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(in5, -484), vmul_n(in2, 1931))),
        in5,
    );
    let t3a: v128 = i32x4_sub(
        vrshr::<12>(i32x4_sub(vmul_n(in5, 1931), vmul_n(in2, -484))),
        in2,
    );
    let mut t4a: v128 = vrshr::<11>(i32x4_add(vmul_n(in3, 1299), vmul_n(in4, 1583)));
    let mut t5a: v128 = vrshr::<11>(i32x4_sub(vmul_n(in3, 1583), vmul_n(in4, 1299)));
    let mut t6a: v128 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(in1, 1189), vmul_n(in6, -176))),
        in6,
    );
    let mut t7a: v128 = i32x4_add(
        vrshr::<12>(i32x4_sub(vmul_n(in1, -176), vmul_n(in6, 1189))),
        in1,
    );

    let t0: v128 = i32x4_min(i32x4_max(i32x4_add(t0a, t4a), min), max);
    let t1: v128 = i32x4_min(i32x4_max(i32x4_add(t1a, t5a), min), max);
    let mut t2: v128 = i32x4_min(i32x4_max(i32x4_add(t2a, t6a), min), max);
    let mut t3: v128 = i32x4_min(i32x4_max(i32x4_add(t3a, t7a), min), max);
    let t4: v128 = i32x4_min(i32x4_max(i32x4_sub(t0a, t4a), min), max);
    let t5: v128 = i32x4_min(i32x4_max(i32x4_sub(t1a, t5a), min), max);
    let mut t6: v128 = i32x4_min(i32x4_max(i32x4_sub(t2a, t6a), min), max);
    let mut t7: v128 = i32x4_min(i32x4_max(i32x4_sub(t3a, t7a), min), max);

    t4a = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t4, -312), vmul_n(t5, 1567))),
        t4,
    );
    t5a = i32x4_sub(
        vrshr::<12>(i32x4_sub(vmul_n(t4, 1567), vmul_n(t5, -312))),
        t5,
    );
    t6a = i32x4_add(
        vrshr::<12>(i32x4_sub(vmul_n(t7, -312), vmul_n(t6, 1567))),
        t7,
    );
    t7a = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t7, 1567), vmul_n(t6, -312))),
        t6,
    );

    c[out_idx(0)] = i32x4_min(i32x4_max(i32x4_add(t0, t2), min), max);
    c[out_idx(7)] = i32x4_neg(i32x4_min(i32x4_max(i32x4_add(t1, t3), min), max));
    t2 = i32x4_min(i32x4_max(i32x4_sub(t0, t2), min), max);
    t3 = i32x4_min(i32x4_max(i32x4_sub(t1, t3), min), max);
    c[out_idx(1)] = i32x4_neg(i32x4_min(i32x4_max(i32x4_add(t4a, t6a), min), max));
    c[out_idx(6)] = i32x4_min(i32x4_max(i32x4_add(t5a, t7a), min), max);
    t6 = i32x4_min(i32x4_max(i32x4_sub(t4a, t6a), min), max);
    t7 = i32x4_min(i32x4_max(i32x4_sub(t5a, t7a), min), max);

    c[out_idx(3)] = i32x4_neg(vrshr::<8>(vmul_n(i32x4_add(t2, t3), 181)));
    c[out_idx(4)] = vrshr::<8>(vmul_n(i32x4_sub(t2, t3), 181));
    c[out_idx(2)] = vrshr::<8>(vmul_n(i32x4_add(t6, t7), 181));
    c[out_idx(5)] = i32x4_neg(vrshr::<8>(vmul_n(i32x4_sub(t6, t7), 181)));
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_adst16_1d_internal(c: &mut [v128], stride: usize, min: v128, max: v128, flip: bool) {
    let out_idx = |k: usize| {
        if flip {
            (16 - 1 - k) * stride
        } else {
            k * stride
        }
    };

    let in0: v128 = c[0];
    let in1: v128 = c[stride];
    let in2: v128 = c[2 * stride];
    let in3: v128 = c[3 * stride];
    let in4: v128 = c[4 * stride];
    let in5: v128 = c[5 * stride];
    let in6: v128 = c[6 * stride];
    let in7: v128 = c[7 * stride];
    let in8: v128 = c[8 * stride];
    let in9: v128 = c[9 * stride];
    let in10: v128 = c[10 * stride];
    let in11: v128 = c[11 * stride];
    let in12: v128 = c[12 * stride];
    let in13: v128 = c[13 * stride];
    let in14: v128 = c[14 * stride];
    let in15: v128 = c[15 * stride];

    let mut t0: v128 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(in15, -5), vmul_n(in0, 201))),
        in15,
    );
    let mut t1: v128 = i32x4_sub(
        vrshr::<12>(i32x4_sub(vmul_n(in15, 201), vmul_n(in0, -5))),
        in0,
    );
    let mut t2: v128 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(in13, -123), vmul_n(in2, 995))),
        in13,
    );
    let mut t3: v128 = i32x4_sub(
        vrshr::<12>(i32x4_sub(vmul_n(in13, 995), vmul_n(in2, -123))),
        in2,
    );
    let mut t4: v128 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(in11, -393), vmul_n(in4, 1751))),
        in11,
    );
    let mut t5: v128 = i32x4_sub(
        vrshr::<12>(i32x4_sub(vmul_n(in11, 1751), vmul_n(in4, -393))),
        in4,
    );
    let mut t6: v128 = vrshr::<11>(i32x4_add(vmul_n(in9, 1645), vmul_n(in6, 1220)));
    let mut t7: v128 = vrshr::<11>(i32x4_sub(vmul_n(in9, 1220), vmul_n(in6, 1645)));
    let mut t8: v128 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(in7, 2751), vmul_n(in8, -1061))),
        in8,
    );
    let mut t9: v128 = i32x4_add(
        vrshr::<12>(i32x4_sub(vmul_n(in7, -1061), vmul_n(in8, 2751))),
        in7,
    );
    let mut t10: v128 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(in5, 2106), vmul_n(in10, -583))),
        in10,
    );
    let mut t11: v128 = i32x4_add(
        vrshr::<12>(i32x4_sub(vmul_n(in5, -583), vmul_n(in10, 2106))),
        in5,
    );
    let mut t12: v128 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(in3, 1380), vmul_n(in12, -239))),
        in12,
    );
    let mut t13: v128 = i32x4_add(
        vrshr::<12>(i32x4_sub(vmul_n(in3, -239), vmul_n(in12, 1380))),
        in3,
    );
    let mut t14: v128 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(in1, 601), vmul_n(in14, -44))),
        in14,
    );
    let mut t15: v128 = i32x4_add(
        vrshr::<12>(i32x4_sub(vmul_n(in1, -44), vmul_n(in14, 601))),
        in1,
    );

    let t0a: v128 = i32x4_min(i32x4_max(i32x4_add(t0, t8), min), max);
    let t1a: v128 = i32x4_min(i32x4_max(i32x4_add(t1, t9), min), max);
    let mut t2a: v128 = i32x4_min(i32x4_max(i32x4_add(t2, t10), min), max);
    let mut t3a: v128 = i32x4_min(i32x4_max(i32x4_add(t3, t11), min), max);
    let mut t4a: v128 = i32x4_min(i32x4_max(i32x4_add(t4, t12), min), max);
    let mut t5a: v128 = i32x4_min(i32x4_max(i32x4_add(t5, t13), min), max);
    let mut t6a: v128 = i32x4_min(i32x4_max(i32x4_add(t6, t14), min), max);
    let mut t7a: v128 = i32x4_min(i32x4_max(i32x4_add(t7, t15), min), max);
    let mut t8a: v128 = i32x4_min(i32x4_max(i32x4_sub(t0, t8), min), max);
    let mut t9a: v128 = i32x4_min(i32x4_max(i32x4_sub(t1, t9), min), max);
    let mut t10a: v128 = i32x4_min(i32x4_max(i32x4_sub(t2, t10), min), max);
    let mut t11a: v128 = i32x4_min(i32x4_max(i32x4_sub(t3, t11), min), max);
    let mut t12a: v128 = i32x4_min(i32x4_max(i32x4_sub(t4, t12), min), max);
    let mut t13a: v128 = i32x4_min(i32x4_max(i32x4_sub(t5, t13), min), max);
    let mut t14a: v128 = i32x4_min(i32x4_max(i32x4_sub(t6, t14), min), max);
    let mut t15a: v128 = i32x4_min(i32x4_max(i32x4_sub(t7, t15), min), max);

    t8 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t8a, -79), vmul_n(t9a, 799))),
        t8a,
    );
    t9 = i32x4_sub(
        vrshr::<12>(i32x4_sub(vmul_n(t8a, 799), vmul_n(t9a, -79))),
        t9a,
    );
    t10 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t10a, 2276), vmul_n(t11a, -690))),
        t11a,
    );
    t11 = i32x4_add(
        vrshr::<12>(i32x4_sub(vmul_n(t10a, -690), vmul_n(t11a, 2276))),
        t10a,
    );
    t12 = i32x4_add(
        vrshr::<12>(i32x4_sub(vmul_n(t13a, -79), vmul_n(t12a, 799))),
        t13a,
    );
    t13 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t13a, 799), vmul_n(t12a, -79))),
        t12a,
    );
    t14 = i32x4_sub(
        vrshr::<12>(i32x4_sub(vmul_n(t15a, 2276), vmul_n(t14a, -690))),
        t14a,
    );
    t15 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t15a, -690), vmul_n(t14a, 2276))),
        t15a,
    );

    t0 = i32x4_min(i32x4_max(i32x4_add(t0a, t4a), min), max);
    t1 = i32x4_min(i32x4_max(i32x4_add(t1a, t5a), min), max);
    t2 = i32x4_min(i32x4_max(i32x4_add(t2a, t6a), min), max);
    t3 = i32x4_min(i32x4_max(i32x4_add(t3a, t7a), min), max);
    t4 = i32x4_min(i32x4_max(i32x4_sub(t0a, t4a), min), max);
    t5 = i32x4_min(i32x4_max(i32x4_sub(t1a, t5a), min), max);
    t6 = i32x4_min(i32x4_max(i32x4_sub(t2a, t6a), min), max);
    t7 = i32x4_min(i32x4_max(i32x4_sub(t3a, t7a), min), max);
    t8a = i32x4_min(i32x4_max(i32x4_add(t8, t12), min), max);
    t9a = i32x4_min(i32x4_max(i32x4_add(t9, t13), min), max);
    t10a = i32x4_min(i32x4_max(i32x4_add(t10, t14), min), max);
    t11a = i32x4_min(i32x4_max(i32x4_add(t11, t15), min), max);
    t12a = i32x4_min(i32x4_max(i32x4_sub(t8, t12), min), max);
    t13a = i32x4_min(i32x4_max(i32x4_sub(t9, t13), min), max);
    t14a = i32x4_min(i32x4_max(i32x4_sub(t10, t14), min), max);
    t15a = i32x4_min(i32x4_max(i32x4_sub(t11, t15), min), max);

    t4a = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t4, -312), vmul_n(t5, 1567))),
        t4,
    );
    t5a = i32x4_sub(
        vrshr::<12>(i32x4_sub(vmul_n(t4, 1567), vmul_n(t5, -312))),
        t5,
    );
    t6a = i32x4_add(
        vrshr::<12>(i32x4_sub(vmul_n(t7, -312), vmul_n(t6, 1567))),
        t7,
    );
    t7a = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t7, 1567), vmul_n(t6, -312))),
        t6,
    );
    t12 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t12a, -312), vmul_n(t13a, 1567))),
        t12a,
    );
    t13 = i32x4_sub(
        vrshr::<12>(i32x4_sub(vmul_n(t12a, 1567), vmul_n(t13a, -312))),
        t13a,
    );
    t14 = i32x4_add(
        vrshr::<12>(i32x4_sub(vmul_n(t15a, -312), vmul_n(t14a, 1567))),
        t15a,
    );
    t15 = i32x4_add(
        vrshr::<12>(i32x4_add(vmul_n(t15a, 1567), vmul_n(t14a, -312))),
        t14a,
    );

    c[out_idx(0)] = i32x4_min(i32x4_max(i32x4_add(t0, t2), min), max);
    c[out_idx(15)] = i32x4_neg(i32x4_min(i32x4_max(i32x4_add(t1, t3), min), max));
    t2a = i32x4_min(i32x4_max(i32x4_sub(t0, t2), min), max);
    t3a = i32x4_min(i32x4_max(i32x4_sub(t1, t3), min), max);
    c[out_idx(3)] = i32x4_neg(i32x4_min(i32x4_max(i32x4_add(t4a, t6a), min), max));
    c[out_idx(12)] = i32x4_min(i32x4_max(i32x4_add(t5a, t7a), min), max);
    t6 = i32x4_min(i32x4_max(i32x4_sub(t4a, t6a), min), max);
    t7 = i32x4_min(i32x4_max(i32x4_sub(t5a, t7a), min), max);
    c[out_idx(1)] = i32x4_neg(i32x4_min(i32x4_max(i32x4_add(t8a, t10a), min), max));
    c[out_idx(14)] = i32x4_min(i32x4_max(i32x4_add(t9a, t11a), min), max);
    t10 = i32x4_min(i32x4_max(i32x4_sub(t8a, t10a), min), max);
    t11 = i32x4_min(i32x4_max(i32x4_sub(t9a, t11a), min), max);
    c[out_idx(2)] = i32x4_min(i32x4_max(i32x4_add(t12, t14), min), max);
    c[out_idx(13)] = i32x4_neg(i32x4_min(i32x4_max(i32x4_add(t13, t15), min), max));
    t14a = i32x4_min(i32x4_max(i32x4_sub(t12, t14), min), max);
    t15a = i32x4_min(i32x4_max(i32x4_sub(t13, t15), min), max);

    c[out_idx(7)] = i32x4_neg(vrshr::<8>(vmul_n(i32x4_add(t2a, t3a), 181)));
    c[out_idx(8)] = vrshr::<8>(vmul_n(i32x4_sub(t2a, t3a), 181));
    c[out_idx(4)] = vrshr::<8>(vmul_n(i32x4_add(t6, t7), 181));
    c[out_idx(11)] = i32x4_neg(vrshr::<8>(vmul_n(i32x4_sub(t6, t7), 181)));
    c[out_idx(6)] = vrshr::<8>(vmul_n(i32x4_add(t10, t11), 181));
    c[out_idx(9)] = i32x4_neg(vrshr::<8>(vmul_n(i32x4_sub(t10, t11), 181)));
    c[out_idx(5)] = i32x4_neg(vrshr::<8>(vmul_n(i32x4_add(t14a, t15a), 181)));
    c[out_idx(10)] = vrshr::<8>(vmul_n(i32x4_sub(t14a, t15a), 181));
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_identity4_1d(c: &mut [v128], stride: usize, _min: v128, _max: v128) {
    for i in 0..4 {
        let val: v128 = c[stride * i];
        c[stride * i] = i32x4_add(val, vrshr::<12>(vmul_n(val, 1697)));
    }
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_identity8_1d(c: &mut [v128], stride: usize, _min: v128, _max: v128) {
    for i in 0..8 {
        c[stride * i] = vmul_n(c[stride * i], 2);
    }
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_identity16_1d(c: &mut [v128], stride: usize, _min: v128, _max: v128) {
    for i in 0..16 {
        let val: v128 = c[stride * i];
        c[stride * i] = i32x4_add(vmul_n(val, 2), vrshr::<11>(vmul_n(val, 1697)));
    }
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_identity32_1d(c: &mut [v128], stride: usize, _min: v128, _max: v128) {
    for i in 0..32 {
        c[stride * i] = vmul_n(c[stride * i], 4);
    }
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn dav1d_inv_wht4_1d_internal(c: &mut [v128], stride: usize) {
    let in0: v128 = c[0];
    let in1: v128 = c[stride];
    let in2: v128 = c[2 * stride];
    let in3: v128 = c[3 * stride];

    let t0: v128 = i32x4_add(in0, in1);
    let t2: v128 = i32x4_sub(in2, in3);
    let t4: v128 = vshr::<1>(i32x4_sub(t0, t2));
    let t3: v128 = i32x4_sub(t4, in3);
    let t1: v128 = i32x4_sub(t4, in1);

    c[0] = i32x4_sub(t0, t3);
    c[stride] = t3;
    c[2 * stride] = t1;
    c[3 * stride] = i32x4_add(t2, t1);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_dct4_1d_c(c: &mut [v128], stride: usize, min: v128, max: v128) {
    inv_dct4_1d_internal(c, stride, min, max, false);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_dct8_1d_c(c: &mut [v128], stride: usize, min: v128, max: v128) {
    inv_dct8_1d_internal(c, stride, min, max, false);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_dct16_1d_c(c: &mut [v128], stride: usize, min: v128, max: v128) {
    inv_dct16_1d_internal(c, stride, min, max, false);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_dct32_1d_c(c: &mut [v128], stride: usize, min: v128, max: v128) {
    inv_dct32_1d_internal(c, stride, min, max, false);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_dct64_1d_c(c: &mut [v128], stride: usize, min: v128, max: v128) {
    inv_dct64_1d_internal(c, stride, min, max);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_adst4_1d_c(c: &mut [v128], stride: usize, min: v128, max: v128) {
    inv_adst4_1d_internal(c, stride, min, max, false);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_flipadst4_1d_c(c: &mut [v128], stride: usize, min: v128, max: v128) {
    inv_adst4_1d_internal(c, stride, min, max, true);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_adst8_1d_c(c: &mut [v128], stride: usize, min: v128, max: v128) {
    inv_adst8_1d_internal(c, stride, min, max, false);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_flipadst8_1d_c(c: &mut [v128], stride: usize, min: v128, max: v128) {
    inv_adst8_1d_internal(c, stride, min, max, true);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_adst16_1d_c(c: &mut [v128], stride: usize, min: v128, max: v128) {
    inv_adst16_1d_internal(c, stride, min, max, false);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_flipadst16_1d_c(c: &mut [v128], stride: usize, min: v128, max: v128) {
    inv_adst16_1d_internal(c, stride, min, max, true);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_identity4_1d_c(c: &mut [v128], stride: usize, min: v128, max: v128) {
    inv_identity4_1d(c, stride, min, max);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_identity8_1d_c(c: &mut [v128], stride: usize, min: v128, max: v128) {
    inv_identity8_1d(c, stride, min, max);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_identity16_1d_c(c: &mut [v128], stride: usize, min: v128, max: v128) {
    inv_identity16_1d(c, stride, min, max);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn inv_identity32_1d_c(c: &mut [v128], stride: usize, min: v128, max: v128) {
    inv_identity32_1d(c, stride, min, max);
}

#[target_feature(enable = "simd128")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
fn dav1d_inv_wht4_1d_c(c: &mut [v128], stride: usize) {
    dav1d_inv_wht4_1d_internal(c, stride);
}
