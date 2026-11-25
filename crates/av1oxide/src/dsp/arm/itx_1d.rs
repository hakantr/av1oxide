//! ÜRETİLMİŞ DOSYA — elle düzenlemeyin. Üreteç: `scripts/uretec/itx1d_neon.py`
//! (kaynak: `crates/av1oxide/src/dsp/scalar/itx_1d.rs`, dav1d `src/itx_1d.c` portu).
//!
//! `aarch64` NEON 1B ters dönüşümler (`F5c` / `P53`): her `int32x4_t` şeridi bağımsız
//! bir 1B dönüşümdür; ifadeler skalerin birebir çevirisidir (çeviri tablosu ve
//! bit-exact gerekçesi üreteç belgesinde). `c` dizisi vektör dizisidir, `stride`
//! vektör adımıdır; `min`/`max` kırpma sınırları vektördür.

#![forbid(unsafe_code)]
#![allow(clippy::identity_op, clippy::erasing_op, clippy::neg_multiply)]

use core::arch::aarch64::{
    int32x4_t, vaddq_s32, vmaxq_s32, vminq_s32, vmulq_n_s32, vnegq_s32, vrshrq_n_s32, vshrq_n_s32,
    vsubq_s32,
};

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_dct4_1d_internal(
    c: &mut [int32x4_t],
    stride: usize,
    min: int32x4_t,
    max: int32x4_t,
    tx64: bool,
) {
    let in0: int32x4_t = c[0];
    let in1: int32x4_t = c[stride];

    let t0: int32x4_t;
    let t1: int32x4_t;
    let t2: int32x4_t;
    let t3: int32x4_t;
    if tx64 {
        t0 = vrshrq_n_s32::<8>(vmulq_n_s32(in0, 181));
        t1 = vrshrq_n_s32::<8>(vmulq_n_s32(in0, 181));
        t2 = vrshrq_n_s32::<12>(vmulq_n_s32(in1, 1567));
        t3 = vrshrq_n_s32::<12>(vmulq_n_s32(in1, 3784));
    } else {
        let in2: int32x4_t = c[2 * stride];
        let in3: int32x4_t = c[3 * stride];

        t0 = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(in0, in2), 181));
        t1 = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(in0, in2), 181));
        t2 = vsubq_s32(
            vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in1, 1567), vmulq_n_s32(in3, -312))),
            in3,
        );
        t3 = vaddq_s32(
            vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in1, -312), vmulq_n_s32(in3, 1567))),
            in1,
        );
    }

    c[0] = vminq_s32(vmaxq_s32(vaddq_s32(t0, t3), min), max);
    c[stride] = vminq_s32(vmaxq_s32(vaddq_s32(t1, t2), min), max);
    c[2 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t1, t2), min), max);
    c[3 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t0, t3), min), max);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_dct8_1d_internal(
    c: &mut [int32x4_t],
    stride: usize,
    min: int32x4_t,
    max: int32x4_t,
    tx64: bool,
) {
    inv_dct4_1d_internal(c, stride << 1, min, max, tx64);

    let in1: int32x4_t = c[stride];
    let in3: int32x4_t = c[3 * stride];

    let t4a: int32x4_t;
    let mut t5a: int32x4_t;
    let mut t6a: int32x4_t;
    let t7a: int32x4_t;
    if tx64 {
        t4a = vrshrq_n_s32::<12>(vmulq_n_s32(in1, 799));
        t5a = vrshrq_n_s32::<12>(vmulq_n_s32(in3, -2276));
        t6a = vrshrq_n_s32::<12>(vmulq_n_s32(in3, 3406));
        t7a = vrshrq_n_s32::<12>(vmulq_n_s32(in1, 4017));
    } else {
        let in5: int32x4_t = c[5 * stride];
        let in7: int32x4_t = c[7 * stride];

        t4a = vsubq_s32(
            vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in1, 799), vmulq_n_s32(in7, -79))),
            in7,
        );
        t5a = vrshrq_n_s32::<11>(vsubq_s32(vmulq_n_s32(in5, 1703), vmulq_n_s32(in3, 1138)));
        t6a = vrshrq_n_s32::<11>(vaddq_s32(vmulq_n_s32(in5, 1138), vmulq_n_s32(in3, 1703)));
        t7a = vaddq_s32(
            vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in1, -79), vmulq_n_s32(in7, 799))),
            in1,
        );
    }

    let t4: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t4a, t5a), min), max);
    t5a = vminq_s32(vmaxq_s32(vsubq_s32(t4a, t5a), min), max);
    let t7: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t7a, t6a), min), max);
    t6a = vminq_s32(vmaxq_s32(vsubq_s32(t7a, t6a), min), max);

    let t5: int32x4_t = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t6a, t5a), 181));
    let t6: int32x4_t = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t6a, t5a), 181));

    let t0: int32x4_t = c[0];
    let t1: int32x4_t = c[2 * stride];
    let t2: int32x4_t = c[4 * stride];
    let t3: int32x4_t = c[6 * stride];

    c[0] = vminq_s32(vmaxq_s32(vaddq_s32(t0, t7), min), max);
    c[stride] = vminq_s32(vmaxq_s32(vaddq_s32(t1, t6), min), max);
    c[2 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t2, t5), min), max);
    c[3 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t3, t4), min), max);
    c[4 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t3, t4), min), max);
    c[5 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t2, t5), min), max);
    c[6 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t1, t6), min), max);
    c[7 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t0, t7), min), max);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_dct16_1d_internal(
    c: &mut [int32x4_t],
    stride: usize,
    min: int32x4_t,
    max: int32x4_t,
    tx64: bool,
) {
    inv_dct8_1d_internal(c, stride << 1, min, max, tx64);

    let in1: int32x4_t = c[stride];
    let in3: int32x4_t = c[3 * stride];
    let in5: int32x4_t = c[5 * stride];
    let in7: int32x4_t = c[7 * stride];

    let mut t8a: int32x4_t;
    let mut t9a: int32x4_t;
    let mut t10a: int32x4_t;
    let mut t11a: int32x4_t;
    let mut t12a: int32x4_t;
    let mut t13a: int32x4_t;
    let mut t14a: int32x4_t;
    let mut t15a: int32x4_t;
    if tx64 {
        t8a = vrshrq_n_s32::<12>(vmulq_n_s32(in1, 401));
        t9a = vrshrq_n_s32::<12>(vmulq_n_s32(in7, -2598));
        t10a = vrshrq_n_s32::<12>(vmulq_n_s32(in5, 1931));
        t11a = vrshrq_n_s32::<12>(vmulq_n_s32(in3, -1189));
        t12a = vrshrq_n_s32::<12>(vmulq_n_s32(in3, 3920));
        t13a = vrshrq_n_s32::<12>(vmulq_n_s32(in5, 3612));
        t14a = vrshrq_n_s32::<12>(vmulq_n_s32(in7, 3166));
        t15a = vrshrq_n_s32::<12>(vmulq_n_s32(in1, 4076));
    } else {
        let in9: int32x4_t = c[9 * stride];
        let in11: int32x4_t = c[11 * stride];
        let in13: int32x4_t = c[13 * stride];
        let in15: int32x4_t = c[15 * stride];

        t8a = vsubq_s32(
            vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in1, 401), vmulq_n_s32(in15, -20))),
            in15,
        );
        t9a = vrshrq_n_s32::<11>(vsubq_s32(vmulq_n_s32(in9, 1583), vmulq_n_s32(in7, 1299)));
        t10a = vsubq_s32(
            vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in5, 1931), vmulq_n_s32(in11, -484))),
            in11,
        );
        t11a = vaddq_s32(
            vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in13, -176), vmulq_n_s32(in3, 1189))),
            in13,
        );
        t12a = vaddq_s32(
            vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in13, 1189), vmulq_n_s32(in3, -176))),
            in3,
        );
        t13a = vaddq_s32(
            vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in5, -484), vmulq_n_s32(in11, 1931))),
            in5,
        );
        t14a = vrshrq_n_s32::<11>(vaddq_s32(vmulq_n_s32(in9, 1299), vmulq_n_s32(in7, 1583)));
        t15a = vaddq_s32(
            vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in1, -20), vmulq_n_s32(in15, 401))),
            in1,
        );
    }

    let t8: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t8a, t9a), min), max);
    let mut t9: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t8a, t9a), min), max);
    let mut t10: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t11a, t10a), min), max);
    let mut t11: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t11a, t10a), min), max);
    let mut t12: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t12a, t13a), min), max);
    let mut t13: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t12a, t13a), min), max);
    let mut t14: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t15a, t14a), min), max);
    let t15: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t15a, t14a), min), max);

    t9a = vsubq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(t14, 1567), vmulq_n_s32(t9, -312))),
        t9,
    );
    t14a = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t14, -312), vmulq_n_s32(t9, 1567))),
        t14,
    );
    t10a = vsubq_s32(
        vrshrq_n_s32::<12>(vnegq_s32(vaddq_s32(
            vmulq_n_s32(t13, -312),
            vmulq_n_s32(t10, 1567),
        ))),
        t13,
    );
    t13a = vsubq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(t13, 1567), vmulq_n_s32(t10, -312))),
        t10,
    );

    t8a = vminq_s32(vmaxq_s32(vaddq_s32(t8, t11), min), max);
    t9 = vminq_s32(vmaxq_s32(vaddq_s32(t9a, t10a), min), max);
    t10 = vminq_s32(vmaxq_s32(vsubq_s32(t9a, t10a), min), max);
    t11a = vminq_s32(vmaxq_s32(vsubq_s32(t8, t11), min), max);
    t12a = vminq_s32(vmaxq_s32(vsubq_s32(t15, t12), min), max);
    t13 = vminq_s32(vmaxq_s32(vsubq_s32(t14a, t13a), min), max);
    t14 = vminq_s32(vmaxq_s32(vaddq_s32(t14a, t13a), min), max);
    t15a = vminq_s32(vmaxq_s32(vaddq_s32(t15, t12), min), max);

    t10a = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t13, t10), 181));
    t13a = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t13, t10), 181));
    t11 = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t12a, t11a), 181));
    t12 = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t12a, t11a), 181));

    let t0: int32x4_t = c[0];
    let t1: int32x4_t = c[2 * stride];
    let t2: int32x4_t = c[4 * stride];
    let t3: int32x4_t = c[6 * stride];
    let t4: int32x4_t = c[8 * stride];
    let t5: int32x4_t = c[10 * stride];
    let t6: int32x4_t = c[12 * stride];
    let t7: int32x4_t = c[14 * stride];

    c[0] = vminq_s32(vmaxq_s32(vaddq_s32(t0, t15a), min), max);
    c[stride] = vminq_s32(vmaxq_s32(vaddq_s32(t1, t14), min), max);
    c[2 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t2, t13a), min), max);
    c[3 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t3, t12), min), max);
    c[4 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t4, t11), min), max);
    c[5 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t5, t10a), min), max);
    c[6 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t6, t9), min), max);
    c[7 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t7, t8a), min), max);
    c[8 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t7, t8a), min), max);
    c[9 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t6, t9), min), max);
    c[10 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t5, t10a), min), max);
    c[11 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t4, t11), min), max);
    c[12 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t3, t12), min), max);
    c[13 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t2, t13a), min), max);
    c[14 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t1, t14), min), max);
    c[15 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t0, t15a), min), max);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_dct32_1d_internal(
    c: &mut [int32x4_t],
    stride: usize,
    min: int32x4_t,
    max: int32x4_t,
    tx64: bool,
) {
    inv_dct16_1d_internal(c, stride << 1, min, max, tx64);

    let in1: int32x4_t = c[stride];
    let in3: int32x4_t = c[3 * stride];
    let in5: int32x4_t = c[5 * stride];
    let in7: int32x4_t = c[7 * stride];
    let in9: int32x4_t = c[9 * stride];
    let in11: int32x4_t = c[11 * stride];
    let in13: int32x4_t = c[13 * stride];
    let in15: int32x4_t = c[15 * stride];

    let mut t16a: int32x4_t;
    let mut t17a: int32x4_t;
    let mut t18a: int32x4_t;
    let mut t19a: int32x4_t;
    let mut t20a: int32x4_t;
    let mut t21a: int32x4_t;
    let mut t22a: int32x4_t;
    let mut t23a: int32x4_t;
    let mut t24a: int32x4_t;
    let mut t25a: int32x4_t;
    let mut t26a: int32x4_t;
    let mut t27a: int32x4_t;
    let mut t28a: int32x4_t;
    let mut t29a: int32x4_t;
    let mut t30a: int32x4_t;
    let mut t31a: int32x4_t;
    if tx64 {
        t16a = vrshrq_n_s32::<12>(vmulq_n_s32(in1, 201));
        t17a = vrshrq_n_s32::<12>(vmulq_n_s32(in15, -2751));
        t18a = vrshrq_n_s32::<12>(vmulq_n_s32(in9, 1751));
        t19a = vrshrq_n_s32::<12>(vmulq_n_s32(in7, -1380));
        t20a = vrshrq_n_s32::<12>(vmulq_n_s32(in5, 995));
        t21a = vrshrq_n_s32::<12>(vmulq_n_s32(in11, -2106));
        t22a = vrshrq_n_s32::<12>(vmulq_n_s32(in13, 2440));
        t23a = vrshrq_n_s32::<12>(vmulq_n_s32(in3, -601));
        t24a = vrshrq_n_s32::<12>(vmulq_n_s32(in3, 4052));
        t25a = vrshrq_n_s32::<12>(vmulq_n_s32(in13, 3290));
        t26a = vrshrq_n_s32::<12>(vmulq_n_s32(in11, 3513));
        t27a = vrshrq_n_s32::<12>(vmulq_n_s32(in5, 3973));
        t28a = vrshrq_n_s32::<12>(vmulq_n_s32(in7, 3857));
        t29a = vrshrq_n_s32::<12>(vmulq_n_s32(in9, 3703));
        t30a = vrshrq_n_s32::<12>(vmulq_n_s32(in15, 3035));
        t31a = vrshrq_n_s32::<12>(vmulq_n_s32(in1, 4091));
    } else {
        let in17: int32x4_t = c[17 * stride];
        let in19: int32x4_t = c[19 * stride];
        let in21: int32x4_t = c[21 * stride];
        let in23: int32x4_t = c[23 * stride];
        let in25: int32x4_t = c[25 * stride];
        let in27: int32x4_t = c[27 * stride];
        let in29: int32x4_t = c[29 * stride];
        let in31: int32x4_t = c[31 * stride];

        t16a = vsubq_s32(
            vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in1, 201), vmulq_n_s32(in31, -5))),
            in31,
        );
        t17a = vaddq_s32(
            vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in17, -1061), vmulq_n_s32(in15, 2751))),
            in17,
        );
        t18a = vsubq_s32(
            vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in9, 1751), vmulq_n_s32(in23, -393))),
            in23,
        );
        t19a = vaddq_s32(
            vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in25, -239), vmulq_n_s32(in7, 1380))),
            in25,
        );
        t20a = vsubq_s32(
            vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in5, 995), vmulq_n_s32(in27, -123))),
            in27,
        );
        t21a = vaddq_s32(
            vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in21, -583), vmulq_n_s32(in11, 2106))),
            in21,
        );
        t22a = vrshrq_n_s32::<11>(vsubq_s32(vmulq_n_s32(in13, 1220), vmulq_n_s32(in19, 1645)));
        t23a = vaddq_s32(
            vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in29, -44), vmulq_n_s32(in3, 601))),
            in29,
        );
        t24a = vaddq_s32(
            vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in29, 601), vmulq_n_s32(in3, -44))),
            in3,
        );
        t25a = vrshrq_n_s32::<11>(vaddq_s32(vmulq_n_s32(in13, 1645), vmulq_n_s32(in19, 1220)));
        t26a = vaddq_s32(
            vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in21, 2106), vmulq_n_s32(in11, -583))),
            in11,
        );
        t27a = vaddq_s32(
            vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in5, -123), vmulq_n_s32(in27, 995))),
            in5,
        );
        t28a = vaddq_s32(
            vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in25, 1380), vmulq_n_s32(in7, -239))),
            in7,
        );
        t29a = vaddq_s32(
            vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in9, -393), vmulq_n_s32(in23, 1751))),
            in9,
        );
        t30a = vaddq_s32(
            vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in17, 2751), vmulq_n_s32(in15, -1061))),
            in15,
        );
        t31a = vaddq_s32(
            vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in1, -5), vmulq_n_s32(in31, 201))),
            in1,
        );
    }

    let mut t16: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t16a, t17a), min), max);
    let mut t17: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t16a, t17a), min), max);
    let mut t18: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t19a, t18a), min), max);
    let mut t19: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t19a, t18a), min), max);
    let mut t20: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t20a, t21a), min), max);
    let mut t21: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t20a, t21a), min), max);
    let mut t22: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t23a, t22a), min), max);
    let mut t23: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t23a, t22a), min), max);
    let mut t24: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t24a, t25a), min), max);
    let mut t25: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t24a, t25a), min), max);
    let mut t26: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t27a, t26a), min), max);
    let mut t27: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t27a, t26a), min), max);
    let mut t28: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t28a, t29a), min), max);
    let mut t29: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t28a, t29a), min), max);
    let mut t30: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t31a, t30a), min), max);
    let mut t31: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t31a, t30a), min), max);

    t17a = vsubq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(t30, 799), vmulq_n_s32(t17, -79))),
        t17,
    );
    t30a = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t30, -79), vmulq_n_s32(t17, 799))),
        t30,
    );
    t18a = vsubq_s32(
        vrshrq_n_s32::<12>(vnegq_s32(vaddq_s32(
            vmulq_n_s32(t29, -79),
            vmulq_n_s32(t18, 799),
        ))),
        t29,
    );
    t29a = vsubq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(t29, 799), vmulq_n_s32(t18, -79))),
        t18,
    );
    t21a = vrshrq_n_s32::<11>(vsubq_s32(vmulq_n_s32(t26, 1703), vmulq_n_s32(t21, 1138)));
    t26a = vrshrq_n_s32::<11>(vaddq_s32(vmulq_n_s32(t26, 1138), vmulq_n_s32(t21, 1703)));
    t22a = vrshrq_n_s32::<11>(vnegq_s32(vaddq_s32(
        vmulq_n_s32(t25, 1138),
        vmulq_n_s32(t22, 1703),
    )));
    t25a = vrshrq_n_s32::<11>(vsubq_s32(vmulq_n_s32(t25, 1703), vmulq_n_s32(t22, 1138)));

    t16a = vminq_s32(vmaxq_s32(vaddq_s32(t16, t19), min), max);
    t17 = vminq_s32(vmaxq_s32(vaddq_s32(t17a, t18a), min), max);
    t18 = vminq_s32(vmaxq_s32(vsubq_s32(t17a, t18a), min), max);
    t19a = vminq_s32(vmaxq_s32(vsubq_s32(t16, t19), min), max);
    t20a = vminq_s32(vmaxq_s32(vsubq_s32(t23, t20), min), max);
    t21 = vminq_s32(vmaxq_s32(vsubq_s32(t22a, t21a), min), max);
    t22 = vminq_s32(vmaxq_s32(vaddq_s32(t22a, t21a), min), max);
    t23a = vminq_s32(vmaxq_s32(vaddq_s32(t23, t20), min), max);
    t24a = vminq_s32(vmaxq_s32(vaddq_s32(t24, t27), min), max);
    t25 = vminq_s32(vmaxq_s32(vaddq_s32(t25a, t26a), min), max);
    t26 = vminq_s32(vmaxq_s32(vsubq_s32(t25a, t26a), min), max);
    t27a = vminq_s32(vmaxq_s32(vsubq_s32(t24, t27), min), max);
    t28a = vminq_s32(vmaxq_s32(vsubq_s32(t31, t28), min), max);
    t29 = vminq_s32(vmaxq_s32(vsubq_s32(t30a, t29a), min), max);
    t30 = vminq_s32(vmaxq_s32(vaddq_s32(t30a, t29a), min), max);
    t31a = vminq_s32(vmaxq_s32(vaddq_s32(t31, t28), min), max);

    t18a = vsubq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(t29, 1567), vmulq_n_s32(t18, -312))),
        t18,
    );
    t29a = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t29, -312), vmulq_n_s32(t18, 1567))),
        t29,
    );
    t19 = vsubq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(t28a, 1567), vmulq_n_s32(t19a, -312))),
        t19a,
    );
    t28 = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t28a, -312), vmulq_n_s32(t19a, 1567))),
        t28a,
    );
    t20 = vsubq_s32(
        vrshrq_n_s32::<12>(vnegq_s32(vaddq_s32(
            vmulq_n_s32(t27a, -312),
            vmulq_n_s32(t20a, 1567),
        ))),
        t27a,
    );
    t27 = vsubq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(t27a, 1567), vmulq_n_s32(t20a, -312))),
        t20a,
    );
    t21a = vsubq_s32(
        vrshrq_n_s32::<12>(vnegq_s32(vaddq_s32(
            vmulq_n_s32(t26, -312),
            vmulq_n_s32(t21, 1567),
        ))),
        t26,
    );
    t26a = vsubq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(t26, 1567), vmulq_n_s32(t21, -312))),
        t21,
    );

    t16 = vminq_s32(vmaxq_s32(vaddq_s32(t16a, t23a), min), max);
    t17a = vminq_s32(vmaxq_s32(vaddq_s32(t17, t22), min), max);
    t18 = vminq_s32(vmaxq_s32(vaddq_s32(t18a, t21a), min), max);
    t19a = vminq_s32(vmaxq_s32(vaddq_s32(t19, t20), min), max);
    t20a = vminq_s32(vmaxq_s32(vsubq_s32(t19, t20), min), max);
    t21 = vminq_s32(vmaxq_s32(vsubq_s32(t18a, t21a), min), max);
    t22a = vminq_s32(vmaxq_s32(vsubq_s32(t17, t22), min), max);
    t23 = vminq_s32(vmaxq_s32(vsubq_s32(t16a, t23a), min), max);
    t24 = vminq_s32(vmaxq_s32(vsubq_s32(t31a, t24a), min), max);
    t25a = vminq_s32(vmaxq_s32(vsubq_s32(t30, t25), min), max);
    t26 = vminq_s32(vmaxq_s32(vsubq_s32(t29a, t26a), min), max);
    t27a = vminq_s32(vmaxq_s32(vsubq_s32(t28, t27), min), max);
    t28a = vminq_s32(vmaxq_s32(vaddq_s32(t28, t27), min), max);
    t29 = vminq_s32(vmaxq_s32(vaddq_s32(t29a, t26a), min), max);
    t30a = vminq_s32(vmaxq_s32(vaddq_s32(t30, t25), min), max);
    t31 = vminq_s32(vmaxq_s32(vaddq_s32(t31a, t24a), min), max);

    t20 = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t27a, t20a), 181));
    t27 = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t27a, t20a), 181));
    t21a = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t26, t21), 181));
    t26a = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t26, t21), 181));
    t22 = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t25a, t22a), 181));
    t25 = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t25a, t22a), 181));
    t23a = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t24, t23), 181));
    t24a = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t24, t23), 181));

    let t0: int32x4_t = c[0];
    let t1: int32x4_t = c[2 * stride];
    let t2: int32x4_t = c[4 * stride];
    let t3: int32x4_t = c[6 * stride];
    let t4: int32x4_t = c[8 * stride];
    let t5: int32x4_t = c[10 * stride];
    let t6: int32x4_t = c[12 * stride];
    let t7: int32x4_t = c[14 * stride];
    let t8: int32x4_t = c[16 * stride];
    let t9: int32x4_t = c[18 * stride];
    let t10: int32x4_t = c[20 * stride];
    let t11: int32x4_t = c[22 * stride];
    let t12: int32x4_t = c[24 * stride];
    let t13: int32x4_t = c[26 * stride];
    let t14: int32x4_t = c[28 * stride];
    let t15: int32x4_t = c[30 * stride];

    c[0] = vminq_s32(vmaxq_s32(vaddq_s32(t0, t31), min), max);
    c[stride] = vminq_s32(vmaxq_s32(vaddq_s32(t1, t30a), min), max);
    c[2 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t2, t29), min), max);
    c[3 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t3, t28a), min), max);
    c[4 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t4, t27), min), max);
    c[5 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t5, t26a), min), max);
    c[6 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t6, t25), min), max);
    c[7 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t7, t24a), min), max);
    c[8 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t8, t23a), min), max);
    c[9 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t9, t22), min), max);
    c[10 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t10, t21a), min), max);
    c[11 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t11, t20), min), max);
    c[12 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t12, t19a), min), max);
    c[13 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t13, t18), min), max);
    c[14 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t14, t17a), min), max);
    c[15 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t15, t16), min), max);
    c[16 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t15, t16), min), max);
    c[17 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t14, t17a), min), max);
    c[18 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t13, t18), min), max);
    c[19 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t12, t19a), min), max);
    c[20 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t11, t20), min), max);
    c[21 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t10, t21a), min), max);
    c[22 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t9, t22), min), max);
    c[23 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t8, t23a), min), max);
    c[24 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t7, t24a), min), max);
    c[25 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t6, t25), min), max);
    c[26 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t5, t26a), min), max);
    c[27 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t4, t27), min), max);
    c[28 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t3, t28a), min), max);
    c[29 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t2, t29), min), max);
    c[30 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t1, t30a), min), max);
    c[31 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t0, t31), min), max);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_dct64_1d_internal(
    c: &mut [int32x4_t],
    stride: usize,
    min: int32x4_t,
    max: int32x4_t,
) {
    inv_dct32_1d_internal(c, stride << 1, min, max, true);

    let in1: int32x4_t = c[stride];
    let in3: int32x4_t = c[3 * stride];
    let in5: int32x4_t = c[5 * stride];
    let in7: int32x4_t = c[7 * stride];
    let in9: int32x4_t = c[9 * stride];
    let in11: int32x4_t = c[11 * stride];
    let in13: int32x4_t = c[13 * stride];
    let in15: int32x4_t = c[15 * stride];
    let in17: int32x4_t = c[17 * stride];
    let in19: int32x4_t = c[19 * stride];
    let in21: int32x4_t = c[21 * stride];
    let in23: int32x4_t = c[23 * stride];
    let in25: int32x4_t = c[25 * stride];
    let in27: int32x4_t = c[27 * stride];
    let in29: int32x4_t = c[29 * stride];
    let in31: int32x4_t = c[31 * stride];

    let mut t32a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in1, 101));
    let mut t33a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in31, -2824));
    let mut t34a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in17, 1660));
    let mut t35a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in15, -1474));
    let mut t36a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in9, 897));
    let mut t37a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in23, -2191));
    let mut t38a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in25, 2359));
    let mut t39a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in7, -700));
    let mut t40a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in5, 501));
    let mut t41a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in27, -2520));
    let mut t42a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in21, 2019));
    let mut t43a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in11, -1092));
    let mut t44a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in13, 1285));
    let mut t45a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in19, -1842));
    let mut t46a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in29, 2675));
    let mut t47a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in3, -301));
    let mut t48a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in3, 4085));
    let mut t49a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in29, 3102));
    let mut t50a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in19, 3659));
    let mut t51a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in13, 3889));
    let mut t52a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in11, 3948));
    let mut t53a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in21, 3564));
    let mut t54a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in27, 3229));
    let mut t55a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in5, 4065));
    let mut t56a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in7, 4036));
    let mut t57a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in25, 3349));
    let mut t58a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in23, 3461));
    let mut t59a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in9, 3996));
    let mut t60a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in15, 3822));
    let mut t61a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in17, 3745));
    let mut t62a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in31, 2967));
    let mut t63a: int32x4_t = vrshrq_n_s32::<12>(vmulq_n_s32(in1, 4095));

    let mut t32: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t32a, t33a), min), max);
    let mut t33: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t32a, t33a), min), max);
    let mut t34: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t35a, t34a), min), max);
    let mut t35: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t35a, t34a), min), max);
    let mut t36: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t36a, t37a), min), max);
    let mut t37: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t36a, t37a), min), max);
    let mut t38: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t39a, t38a), min), max);
    let mut t39: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t39a, t38a), min), max);
    let mut t40: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t40a, t41a), min), max);
    let mut t41: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t40a, t41a), min), max);
    let mut t42: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t43a, t42a), min), max);
    let mut t43: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t43a, t42a), min), max);
    let mut t44: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t44a, t45a), min), max);
    let mut t45: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t44a, t45a), min), max);
    let mut t46: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t47a, t46a), min), max);
    let mut t47: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t47a, t46a), min), max);
    let mut t48: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t48a, t49a), min), max);
    let mut t49: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t48a, t49a), min), max);
    let mut t50: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t51a, t50a), min), max);
    let mut t51: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t51a, t50a), min), max);
    let mut t52: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t52a, t53a), min), max);
    let mut t53: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t52a, t53a), min), max);
    let mut t54: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t55a, t54a), min), max);
    let mut t55: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t55a, t54a), min), max);
    let mut t56: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t56a, t57a), min), max);
    let mut t57: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t56a, t57a), min), max);
    let mut t58: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t59a, t58a), min), max);
    let mut t59: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t59a, t58a), min), max);
    let mut t60: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t60a, t61a), min), max);
    let mut t61: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t60a, t61a), min), max);
    let mut t62: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t63a, t62a), min), max);
    let mut t63: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t63a, t62a), min), max);

    t33a = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t33, 20), vmulq_n_s32(t62, 401))),
        t33,
    );
    t34a = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t34, -401), vmulq_n_s32(t61, 20))),
        t61,
    );
    t37a = vrshrq_n_s32::<11>(vaddq_s32(vmulq_n_s32(t37, -1299), vmulq_n_s32(t58, 1583)));
    t38a = vrshrq_n_s32::<11>(vaddq_s32(vmulq_n_s32(t38, -1583), vmulq_n_s32(t57, -1299)));
    t41a = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t41, 484), vmulq_n_s32(t54, 1931))),
        t41,
    );
    t42a = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t42, -1931), vmulq_n_s32(t53, 484))),
        t53,
    );
    t45a = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t45, -1189), vmulq_n_s32(t50, -176))),
        t50,
    );
    t46a = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t46, 176), vmulq_n_s32(t49, -1189))),
        t46,
    );
    t49a = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t46, -1189), vmulq_n_s32(t49, -176))),
        t49,
    );
    t50a = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t45, -176), vmulq_n_s32(t50, 1189))),
        t45,
    );
    t53a = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t42, 484), vmulq_n_s32(t53, 1931))),
        t42,
    );
    t54a = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t41, 1931), vmulq_n_s32(t54, -484))),
        t54,
    );
    t57a = vrshrq_n_s32::<11>(vaddq_s32(vmulq_n_s32(t38, -1299), vmulq_n_s32(t57, 1583)));
    t58a = vrshrq_n_s32::<11>(vaddq_s32(vmulq_n_s32(t37, 1583), vmulq_n_s32(t58, 1299)));
    t61a = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t34, 20), vmulq_n_s32(t61, 401))),
        t34,
    );
    t62a = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t33, 401), vmulq_n_s32(t62, -20))),
        t62,
    );

    t32a = vminq_s32(vmaxq_s32(vaddq_s32(t32, t35), min), max);
    t33 = vminq_s32(vmaxq_s32(vaddq_s32(t33a, t34a), min), max);
    t34 = vminq_s32(vmaxq_s32(vsubq_s32(t33a, t34a), min), max);
    t35a = vminq_s32(vmaxq_s32(vsubq_s32(t32, t35), min), max);
    t36a = vminq_s32(vmaxq_s32(vsubq_s32(t39, t36), min), max);
    t37 = vminq_s32(vmaxq_s32(vsubq_s32(t38a, t37a), min), max);
    t38 = vminq_s32(vmaxq_s32(vaddq_s32(t38a, t37a), min), max);
    t39a = vminq_s32(vmaxq_s32(vaddq_s32(t39, t36), min), max);
    t40a = vminq_s32(vmaxq_s32(vaddq_s32(t40, t43), min), max);
    t41 = vminq_s32(vmaxq_s32(vaddq_s32(t41a, t42a), min), max);
    t42 = vminq_s32(vmaxq_s32(vsubq_s32(t41a, t42a), min), max);
    t43a = vminq_s32(vmaxq_s32(vsubq_s32(t40, t43), min), max);
    t44a = vminq_s32(vmaxq_s32(vsubq_s32(t47, t44), min), max);
    t45 = vminq_s32(vmaxq_s32(vsubq_s32(t46a, t45a), min), max);
    t46 = vminq_s32(vmaxq_s32(vaddq_s32(t46a, t45a), min), max);
    t47a = vminq_s32(vmaxq_s32(vaddq_s32(t47, t44), min), max);
    t48a = vminq_s32(vmaxq_s32(vaddq_s32(t48, t51), min), max);
    t49 = vminq_s32(vmaxq_s32(vaddq_s32(t49a, t50a), min), max);
    t50 = vminq_s32(vmaxq_s32(vsubq_s32(t49a, t50a), min), max);
    t51a = vminq_s32(vmaxq_s32(vsubq_s32(t48, t51), min), max);
    t52a = vminq_s32(vmaxq_s32(vsubq_s32(t55, t52), min), max);
    t53 = vminq_s32(vmaxq_s32(vsubq_s32(t54a, t53a), min), max);
    t54 = vminq_s32(vmaxq_s32(vaddq_s32(t54a, t53a), min), max);
    t55a = vminq_s32(vmaxq_s32(vaddq_s32(t55, t52), min), max);
    t56a = vminq_s32(vmaxq_s32(vaddq_s32(t56, t59), min), max);
    t57 = vminq_s32(vmaxq_s32(vaddq_s32(t57a, t58a), min), max);
    t58 = vminq_s32(vmaxq_s32(vsubq_s32(t57a, t58a), min), max);
    t59a = vminq_s32(vmaxq_s32(vsubq_s32(t56, t59), min), max);
    t60a = vminq_s32(vmaxq_s32(vsubq_s32(t63, t60), min), max);
    t61 = vminq_s32(vmaxq_s32(vsubq_s32(t62a, t61a), min), max);
    t62 = vminq_s32(vmaxq_s32(vaddq_s32(t62a, t61a), min), max);
    t63a = vminq_s32(vmaxq_s32(vaddq_s32(t63, t60), min), max);

    t34a = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t34, 79), vmulq_n_s32(t61, 799))),
        t34,
    );
    t35 = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t35a, 79), vmulq_n_s32(t60a, 799))),
        t35a,
    );
    t36 = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t36a, -799), vmulq_n_s32(t59a, 79))),
        t59a,
    );
    t37a = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t37, -799), vmulq_n_s32(t58, 79))),
        t58,
    );
    t42a = vrshrq_n_s32::<11>(vaddq_s32(vmulq_n_s32(t42, -1138), vmulq_n_s32(t53, 1703)));
    t43 = vrshrq_n_s32::<11>(vaddq_s32(vmulq_n_s32(t43a, -1138), vmulq_n_s32(t52a, 1703)));
    t44 = vrshrq_n_s32::<11>(vaddq_s32(
        vmulq_n_s32(t44a, -1703),
        vmulq_n_s32(t51a, -1138),
    ));
    t45a = vrshrq_n_s32::<11>(vaddq_s32(vmulq_n_s32(t45, -1703), vmulq_n_s32(t50, -1138)));
    t50a = vrshrq_n_s32::<11>(vaddq_s32(vmulq_n_s32(t45, -1138), vmulq_n_s32(t50, 1703)));
    t51 = vrshrq_n_s32::<11>(vaddq_s32(vmulq_n_s32(t44a, -1138), vmulq_n_s32(t51a, 1703)));
    t52 = vrshrq_n_s32::<11>(vaddq_s32(vmulq_n_s32(t43a, 1703), vmulq_n_s32(t52a, 1138)));
    t53a = vrshrq_n_s32::<11>(vaddq_s32(vmulq_n_s32(t42, 1703), vmulq_n_s32(t53, 1138)));
    t58a = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t37, 79), vmulq_n_s32(t58, 799))),
        t37,
    );
    t59 = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t36a, 79), vmulq_n_s32(t59a, 799))),
        t36a,
    );
    t60 = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t35a, 799), vmulq_n_s32(t60a, -79))),
        t60a,
    );
    t61a = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t34, 799), vmulq_n_s32(t61, -79))),
        t61,
    );

    t32 = vminq_s32(vmaxq_s32(vaddq_s32(t32a, t39a), min), max);
    t33a = vminq_s32(vmaxq_s32(vaddq_s32(t33, t38), min), max);
    t34 = vminq_s32(vmaxq_s32(vaddq_s32(t34a, t37a), min), max);
    t35a = vminq_s32(vmaxq_s32(vaddq_s32(t35, t36), min), max);
    t36a = vminq_s32(vmaxq_s32(vsubq_s32(t35, t36), min), max);
    t37 = vminq_s32(vmaxq_s32(vsubq_s32(t34a, t37a), min), max);
    t38a = vminq_s32(vmaxq_s32(vsubq_s32(t33, t38), min), max);
    t39 = vminq_s32(vmaxq_s32(vsubq_s32(t32a, t39a), min), max);
    t40 = vminq_s32(vmaxq_s32(vsubq_s32(t47a, t40a), min), max);
    t41a = vminq_s32(vmaxq_s32(vsubq_s32(t46, t41), min), max);
    t42 = vminq_s32(vmaxq_s32(vsubq_s32(t45a, t42a), min), max);
    t43a = vminq_s32(vmaxq_s32(vsubq_s32(t44, t43), min), max);
    t44a = vminq_s32(vmaxq_s32(vaddq_s32(t44, t43), min), max);
    t45 = vminq_s32(vmaxq_s32(vaddq_s32(t45a, t42a), min), max);
    t46a = vminq_s32(vmaxq_s32(vaddq_s32(t46, t41), min), max);
    t47 = vminq_s32(vmaxq_s32(vaddq_s32(t47a, t40a), min), max);
    t48 = vminq_s32(vmaxq_s32(vaddq_s32(t48a, t55a), min), max);
    t49a = vminq_s32(vmaxq_s32(vaddq_s32(t49, t54), min), max);
    t50 = vminq_s32(vmaxq_s32(vaddq_s32(t50a, t53a), min), max);
    t51a = vminq_s32(vmaxq_s32(vaddq_s32(t51, t52), min), max);
    t52a = vminq_s32(vmaxq_s32(vsubq_s32(t51, t52), min), max);
    t53 = vminq_s32(vmaxq_s32(vsubq_s32(t50a, t53a), min), max);
    t54a = vminq_s32(vmaxq_s32(vsubq_s32(t49, t54), min), max);
    t55 = vminq_s32(vmaxq_s32(vsubq_s32(t48a, t55a), min), max);
    t56 = vminq_s32(vmaxq_s32(vsubq_s32(t63a, t56a), min), max);
    t57a = vminq_s32(vmaxq_s32(vsubq_s32(t62, t57), min), max);
    t58 = vminq_s32(vmaxq_s32(vsubq_s32(t61a, t58a), min), max);
    t59a = vminq_s32(vmaxq_s32(vsubq_s32(t60, t59), min), max);
    t60a = vminq_s32(vmaxq_s32(vaddq_s32(t60, t59), min), max);
    t61 = vminq_s32(vmaxq_s32(vaddq_s32(t61a, t58a), min), max);
    t62a = vminq_s32(vmaxq_s32(vaddq_s32(t62, t57), min), max);
    t63 = vminq_s32(vmaxq_s32(vaddq_s32(t63a, t56a), min), max);

    t36 = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t36a, 312), vmulq_n_s32(t59a, 1567))),
        t36a,
    );
    t37a = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t37, 312), vmulq_n_s32(t58, 1567))),
        t37,
    );
    t38 = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t38a, 312), vmulq_n_s32(t57a, 1567))),
        t38a,
    );
    t39a = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t39, 312), vmulq_n_s32(t56, 1567))),
        t39,
    );
    t40a = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t40, -1567), vmulq_n_s32(t55, 312))),
        t55,
    );
    t41 = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t41a, -1567), vmulq_n_s32(t54a, 312))),
        t54a,
    );
    t42a = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t42, -1567), vmulq_n_s32(t53, 312))),
        t53,
    );
    t43 = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t43a, -1567), vmulq_n_s32(t52a, 312))),
        t52a,
    );
    t52 = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t43a, 312), vmulq_n_s32(t52a, 1567))),
        t43a,
    );
    t53a = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t42, 312), vmulq_n_s32(t53, 1567))),
        t42,
    );
    t54 = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t41a, 312), vmulq_n_s32(t54a, 1567))),
        t41a,
    );
    t55a = vsubq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t40, 312), vmulq_n_s32(t55, 1567))),
        t40,
    );
    t56a = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t39, 1567), vmulq_n_s32(t56, -312))),
        t56,
    );
    t57 = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t38a, 1567), vmulq_n_s32(t57a, -312))),
        t57a,
    );
    t58a = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t37, 1567), vmulq_n_s32(t58, -312))),
        t58,
    );
    t59 = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t36a, 1567), vmulq_n_s32(t59a, -312))),
        t59a,
    );

    t32a = vminq_s32(vmaxq_s32(vaddq_s32(t32, t47), min), max);
    t33 = vminq_s32(vmaxq_s32(vaddq_s32(t33a, t46a), min), max);
    t34a = vminq_s32(vmaxq_s32(vaddq_s32(t34, t45), min), max);
    t35 = vminq_s32(vmaxq_s32(vaddq_s32(t35a, t44a), min), max);
    t36a = vminq_s32(vmaxq_s32(vaddq_s32(t36, t43), min), max);
    t37 = vminq_s32(vmaxq_s32(vaddq_s32(t37a, t42a), min), max);
    t38a = vminq_s32(vmaxq_s32(vaddq_s32(t38, t41), min), max);
    t39 = vminq_s32(vmaxq_s32(vaddq_s32(t39a, t40a), min), max);
    t40 = vminq_s32(vmaxq_s32(vsubq_s32(t39a, t40a), min), max);
    t41a = vminq_s32(vmaxq_s32(vsubq_s32(t38, t41), min), max);
    t42 = vminq_s32(vmaxq_s32(vsubq_s32(t37a, t42a), min), max);
    t43a = vminq_s32(vmaxq_s32(vsubq_s32(t36, t43), min), max);
    t44 = vminq_s32(vmaxq_s32(vsubq_s32(t35a, t44a), min), max);
    t45a = vminq_s32(vmaxq_s32(vsubq_s32(t34, t45), min), max);
    t46 = vminq_s32(vmaxq_s32(vsubq_s32(t33a, t46a), min), max);
    t47a = vminq_s32(vmaxq_s32(vsubq_s32(t32, t47), min), max);
    t48a = vminq_s32(vmaxq_s32(vsubq_s32(t63, t48), min), max);
    t49 = vminq_s32(vmaxq_s32(vsubq_s32(t62a, t49a), min), max);
    t50a = vminq_s32(vmaxq_s32(vsubq_s32(t61, t50), min), max);
    t51 = vminq_s32(vmaxq_s32(vsubq_s32(t60a, t51a), min), max);
    t52a = vminq_s32(vmaxq_s32(vsubq_s32(t59, t52), min), max);
    t53 = vminq_s32(vmaxq_s32(vsubq_s32(t58a, t53a), min), max);
    t54a = vminq_s32(vmaxq_s32(vsubq_s32(t57, t54), min), max);
    t55 = vminq_s32(vmaxq_s32(vsubq_s32(t56a, t55a), min), max);
    t56 = vminq_s32(vmaxq_s32(vaddq_s32(t56a, t55a), min), max);
    t57a = vminq_s32(vmaxq_s32(vaddq_s32(t57, t54), min), max);
    t58 = vminq_s32(vmaxq_s32(vaddq_s32(t58a, t53a), min), max);
    t59a = vminq_s32(vmaxq_s32(vaddq_s32(t59, t52), min), max);
    t60 = vminq_s32(vmaxq_s32(vaddq_s32(t60a, t51a), min), max);
    t61a = vminq_s32(vmaxq_s32(vaddq_s32(t61, t50), min), max);
    t62 = vminq_s32(vmaxq_s32(vaddq_s32(t62a, t49a), min), max);
    t63a = vminq_s32(vmaxq_s32(vaddq_s32(t63, t48), min), max);

    t40a = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t55, t40), 181));
    t41 = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t54a, t41a), 181));
    t42a = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t53, t42), 181));
    t43 = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t52a, t43a), 181));
    t44a = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t51, t44), 181));
    t45 = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t50a, t45a), 181));
    t46a = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t49, t46), 181));
    t47 = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t48a, t47a), 181));
    t48 = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t47a, t48a), 181));
    t49a = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t46, t49), 181));
    t50 = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t45a, t50a), 181));
    t51a = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t44, t51), 181));
    t52 = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t43a, t52a), 181));
    t53a = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t42, t53), 181));
    t54 = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t41a, t54a), 181));
    t55a = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t40, t55), 181));

    let t0: int32x4_t = c[0];
    let t1: int32x4_t = c[2 * stride];
    let t2: int32x4_t = c[4 * stride];
    let t3: int32x4_t = c[6 * stride];
    let t4: int32x4_t = c[8 * stride];
    let t5: int32x4_t = c[10 * stride];
    let t6: int32x4_t = c[12 * stride];
    let t7: int32x4_t = c[14 * stride];
    let t8: int32x4_t = c[16 * stride];
    let t9: int32x4_t = c[18 * stride];
    let t10: int32x4_t = c[20 * stride];
    let t11: int32x4_t = c[22 * stride];
    let t12: int32x4_t = c[24 * stride];
    let t13: int32x4_t = c[26 * stride];
    let t14: int32x4_t = c[28 * stride];
    let t15: int32x4_t = c[30 * stride];
    let t16: int32x4_t = c[32 * stride];
    let t17: int32x4_t = c[34 * stride];
    let t18: int32x4_t = c[36 * stride];
    let t19: int32x4_t = c[38 * stride];
    let t20: int32x4_t = c[40 * stride];
    let t21: int32x4_t = c[42 * stride];
    let t22: int32x4_t = c[44 * stride];
    let t23: int32x4_t = c[46 * stride];
    let t24: int32x4_t = c[48 * stride];
    let t25: int32x4_t = c[50 * stride];
    let t26: int32x4_t = c[52 * stride];
    let t27: int32x4_t = c[54 * stride];
    let t28: int32x4_t = c[56 * stride];
    let t29: int32x4_t = c[58 * stride];
    let t30: int32x4_t = c[60 * stride];
    let t31: int32x4_t = c[62 * stride];

    c[0] = vminq_s32(vmaxq_s32(vaddq_s32(t0, t63a), min), max);
    c[stride] = vminq_s32(vmaxq_s32(vaddq_s32(t1, t62), min), max);
    c[2 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t2, t61a), min), max);
    c[3 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t3, t60), min), max);
    c[4 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t4, t59a), min), max);
    c[5 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t5, t58), min), max);
    c[6 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t6, t57a), min), max);
    c[7 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t7, t56), min), max);
    c[8 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t8, t55a), min), max);
    c[9 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t9, t54), min), max);
    c[10 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t10, t53a), min), max);
    c[11 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t11, t52), min), max);
    c[12 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t12, t51a), min), max);
    c[13 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t13, t50), min), max);
    c[14 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t14, t49a), min), max);
    c[15 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t15, t48), min), max);
    c[16 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t16, t47), min), max);
    c[17 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t17, t46a), min), max);
    c[18 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t18, t45), min), max);
    c[19 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t19, t44a), min), max);
    c[20 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t20, t43), min), max);
    c[21 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t21, t42a), min), max);
    c[22 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t22, t41), min), max);
    c[23 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t23, t40a), min), max);
    c[24 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t24, t39), min), max);
    c[25 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t25, t38a), min), max);
    c[26 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t26, t37), min), max);
    c[27 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t27, t36a), min), max);
    c[28 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t28, t35), min), max);
    c[29 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t29, t34a), min), max);
    c[30 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t30, t33), min), max);
    c[31 * stride] = vminq_s32(vmaxq_s32(vaddq_s32(t31, t32a), min), max);
    c[32 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t31, t32a), min), max);
    c[33 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t30, t33), min), max);
    c[34 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t29, t34a), min), max);
    c[35 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t28, t35), min), max);
    c[36 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t27, t36a), min), max);
    c[37 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t26, t37), min), max);
    c[38 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t25, t38a), min), max);
    c[39 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t24, t39), min), max);
    c[40 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t23, t40a), min), max);
    c[41 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t22, t41), min), max);
    c[42 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t21, t42a), min), max);
    c[43 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t20, t43), min), max);
    c[44 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t19, t44a), min), max);
    c[45 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t18, t45), min), max);
    c[46 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t17, t46a), min), max);
    c[47 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t16, t47), min), max);
    c[48 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t15, t48), min), max);
    c[49 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t14, t49a), min), max);
    c[50 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t13, t50), min), max);
    c[51 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t12, t51a), min), max);
    c[52 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t11, t52), min), max);
    c[53 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t10, t53a), min), max);
    c[54 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t9, t54), min), max);
    c[55 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t8, t55a), min), max);
    c[56 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t7, t56), min), max);
    c[57 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t6, t57a), min), max);
    c[58 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t5, t58), min), max);
    c[59 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t4, t59a), min), max);
    c[60 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t3, t60), min), max);
    c[61 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t2, t61a), min), max);
    c[62 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t1, t62), min), max);
    c[63 * stride] = vminq_s32(vmaxq_s32(vsubq_s32(t0, t63a), min), max);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_adst4_1d_internal(
    c: &mut [int32x4_t],
    stride: usize,
    _min: int32x4_t,
    _max: int32x4_t,
    flip: bool,
) {
    let out_idx = |k: usize| {
        if flip {
            (4 - 1 - k) * stride
        } else {
            k * stride
        }
    };

    let in0: int32x4_t = c[0];
    let in1: int32x4_t = c[stride];
    let in2: int32x4_t = c[2 * stride];
    let in3: int32x4_t = c[3 * stride];

    c[out_idx(0)] = vaddq_s32(
        vaddq_s32(
            vaddq_s32(
                vrshrq_n_s32::<12>(vaddq_s32(
                    vaddq_s32(
                        vaddq_s32(vmulq_n_s32(in0, 1321), vmulq_n_s32(in2, -293)),
                        vmulq_n_s32(in3, -1614),
                    ),
                    vmulq_n_s32(in1, -752),
                )),
                in2,
            ),
            in3,
        ),
        in1,
    );
    c[out_idx(1)] = vaddq_s32(
        vsubq_s32(
            vaddq_s32(
                vrshrq_n_s32::<12>(vaddq_s32(
                    vsubq_s32(
                        vsubq_s32(vmulq_n_s32(in0, -1614), vmulq_n_s32(in2, 1321)),
                        vmulq_n_s32(in3, -293),
                    ),
                    vmulq_n_s32(in1, -752),
                )),
                in0,
            ),
            in3,
        ),
        in1,
    );
    c[out_idx(2)] = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(vsubq_s32(in0, in2), in3), 209));
    c[out_idx(3)] = vsubq_s32(
        vaddq_s32(
            vaddq_s32(
                vrshrq_n_s32::<12>(vsubq_s32(
                    vsubq_s32(
                        vaddq_s32(vmulq_n_s32(in0, -293), vmulq_n_s32(in2, -1614)),
                        vmulq_n_s32(in3, 1321),
                    ),
                    vmulq_n_s32(in1, -752),
                )),
                in0,
            ),
            in2,
        ),
        in1,
    );
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_adst8_1d_internal(
    c: &mut [int32x4_t],
    stride: usize,
    min: int32x4_t,
    max: int32x4_t,
    flip: bool,
) {
    let out_idx = |k: usize| {
        if flip {
            (8 - 1 - k) * stride
        } else {
            k * stride
        }
    };

    let in0: int32x4_t = c[0];
    let in1: int32x4_t = c[stride];
    let in2: int32x4_t = c[2 * stride];
    let in3: int32x4_t = c[3 * stride];
    let in4: int32x4_t = c[4 * stride];
    let in5: int32x4_t = c[5 * stride];
    let in6: int32x4_t = c[6 * stride];
    let in7: int32x4_t = c[7 * stride];

    let t0a: int32x4_t = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in7, -20), vmulq_n_s32(in0, 401))),
        in7,
    );
    let t1a: int32x4_t = vsubq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in7, 401), vmulq_n_s32(in0, -20))),
        in0,
    );
    let t2a: int32x4_t = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in5, -484), vmulq_n_s32(in2, 1931))),
        in5,
    );
    let t3a: int32x4_t = vsubq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in5, 1931), vmulq_n_s32(in2, -484))),
        in2,
    );
    let mut t4a: int32x4_t =
        vrshrq_n_s32::<11>(vaddq_s32(vmulq_n_s32(in3, 1299), vmulq_n_s32(in4, 1583)));
    let mut t5a: int32x4_t =
        vrshrq_n_s32::<11>(vsubq_s32(vmulq_n_s32(in3, 1583), vmulq_n_s32(in4, 1299)));
    let mut t6a: int32x4_t = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in1, 1189), vmulq_n_s32(in6, -176))),
        in6,
    );
    let mut t7a: int32x4_t = vaddq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in1, -176), vmulq_n_s32(in6, 1189))),
        in1,
    );

    let t0: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t0a, t4a), min), max);
    let t1: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t1a, t5a), min), max);
    let mut t2: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t2a, t6a), min), max);
    let mut t3: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t3a, t7a), min), max);
    let t4: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t0a, t4a), min), max);
    let t5: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t1a, t5a), min), max);
    let mut t6: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t2a, t6a), min), max);
    let mut t7: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t3a, t7a), min), max);

    t4a = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t4, -312), vmulq_n_s32(t5, 1567))),
        t4,
    );
    t5a = vsubq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(t4, 1567), vmulq_n_s32(t5, -312))),
        t5,
    );
    t6a = vaddq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(t7, -312), vmulq_n_s32(t6, 1567))),
        t7,
    );
    t7a = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t7, 1567), vmulq_n_s32(t6, -312))),
        t6,
    );

    c[out_idx(0)] = vminq_s32(vmaxq_s32(vaddq_s32(t0, t2), min), max);
    c[out_idx(7)] = vnegq_s32(vminq_s32(vmaxq_s32(vaddq_s32(t1, t3), min), max));
    t2 = vminq_s32(vmaxq_s32(vsubq_s32(t0, t2), min), max);
    t3 = vminq_s32(vmaxq_s32(vsubq_s32(t1, t3), min), max);
    c[out_idx(1)] = vnegq_s32(vminq_s32(vmaxq_s32(vaddq_s32(t4a, t6a), min), max));
    c[out_idx(6)] = vminq_s32(vmaxq_s32(vaddq_s32(t5a, t7a), min), max);
    t6 = vminq_s32(vmaxq_s32(vsubq_s32(t4a, t6a), min), max);
    t7 = vminq_s32(vmaxq_s32(vsubq_s32(t5a, t7a), min), max);

    c[out_idx(3)] = vnegq_s32(vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t2, t3), 181)));
    c[out_idx(4)] = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t2, t3), 181));
    c[out_idx(2)] = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t6, t7), 181));
    c[out_idx(5)] = vnegq_s32(vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t6, t7), 181)));
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_adst16_1d_internal(
    c: &mut [int32x4_t],
    stride: usize,
    min: int32x4_t,
    max: int32x4_t,
    flip: bool,
) {
    let out_idx = |k: usize| {
        if flip {
            (16 - 1 - k) * stride
        } else {
            k * stride
        }
    };

    let in0: int32x4_t = c[0];
    let in1: int32x4_t = c[stride];
    let in2: int32x4_t = c[2 * stride];
    let in3: int32x4_t = c[3 * stride];
    let in4: int32x4_t = c[4 * stride];
    let in5: int32x4_t = c[5 * stride];
    let in6: int32x4_t = c[6 * stride];
    let in7: int32x4_t = c[7 * stride];
    let in8: int32x4_t = c[8 * stride];
    let in9: int32x4_t = c[9 * stride];
    let in10: int32x4_t = c[10 * stride];
    let in11: int32x4_t = c[11 * stride];
    let in12: int32x4_t = c[12 * stride];
    let in13: int32x4_t = c[13 * stride];
    let in14: int32x4_t = c[14 * stride];
    let in15: int32x4_t = c[15 * stride];

    let mut t0: int32x4_t = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in15, -5), vmulq_n_s32(in0, 201))),
        in15,
    );
    let mut t1: int32x4_t = vsubq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in15, 201), vmulq_n_s32(in0, -5))),
        in0,
    );
    let mut t2: int32x4_t = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in13, -123), vmulq_n_s32(in2, 995))),
        in13,
    );
    let mut t3: int32x4_t = vsubq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in13, 995), vmulq_n_s32(in2, -123))),
        in2,
    );
    let mut t4: int32x4_t = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in11, -393), vmulq_n_s32(in4, 1751))),
        in11,
    );
    let mut t5: int32x4_t = vsubq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in11, 1751), vmulq_n_s32(in4, -393))),
        in4,
    );
    let mut t6: int32x4_t =
        vrshrq_n_s32::<11>(vaddq_s32(vmulq_n_s32(in9, 1645), vmulq_n_s32(in6, 1220)));
    let mut t7: int32x4_t =
        vrshrq_n_s32::<11>(vsubq_s32(vmulq_n_s32(in9, 1220), vmulq_n_s32(in6, 1645)));
    let mut t8: int32x4_t = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in7, 2751), vmulq_n_s32(in8, -1061))),
        in8,
    );
    let mut t9: int32x4_t = vaddq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in7, -1061), vmulq_n_s32(in8, 2751))),
        in7,
    );
    let mut t10: int32x4_t = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in5, 2106), vmulq_n_s32(in10, -583))),
        in10,
    );
    let mut t11: int32x4_t = vaddq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in5, -583), vmulq_n_s32(in10, 2106))),
        in5,
    );
    let mut t12: int32x4_t = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in3, 1380), vmulq_n_s32(in12, -239))),
        in12,
    );
    let mut t13: int32x4_t = vaddq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in3, -239), vmulq_n_s32(in12, 1380))),
        in3,
    );
    let mut t14: int32x4_t = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(in1, 601), vmulq_n_s32(in14, -44))),
        in14,
    );
    let mut t15: int32x4_t = vaddq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(in1, -44), vmulq_n_s32(in14, 601))),
        in1,
    );

    let t0a: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t0, t8), min), max);
    let t1a: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t1, t9), min), max);
    let mut t2a: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t2, t10), min), max);
    let mut t3a: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t3, t11), min), max);
    let mut t4a: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t4, t12), min), max);
    let mut t5a: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t5, t13), min), max);
    let mut t6a: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t6, t14), min), max);
    let mut t7a: int32x4_t = vminq_s32(vmaxq_s32(vaddq_s32(t7, t15), min), max);
    let mut t8a: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t0, t8), min), max);
    let mut t9a: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t1, t9), min), max);
    let mut t10a: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t2, t10), min), max);
    let mut t11a: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t3, t11), min), max);
    let mut t12a: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t4, t12), min), max);
    let mut t13a: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t5, t13), min), max);
    let mut t14a: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t6, t14), min), max);
    let mut t15a: int32x4_t = vminq_s32(vmaxq_s32(vsubq_s32(t7, t15), min), max);

    t8 = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t8a, -79), vmulq_n_s32(t9a, 799))),
        t8a,
    );
    t9 = vsubq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(t8a, 799), vmulq_n_s32(t9a, -79))),
        t9a,
    );
    t10 = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t10a, 2276), vmulq_n_s32(t11a, -690))),
        t11a,
    );
    t11 = vaddq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(t10a, -690), vmulq_n_s32(t11a, 2276))),
        t10a,
    );
    t12 = vaddq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(t13a, -79), vmulq_n_s32(t12a, 799))),
        t13a,
    );
    t13 = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t13a, 799), vmulq_n_s32(t12a, -79))),
        t12a,
    );
    t14 = vsubq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(t15a, 2276), vmulq_n_s32(t14a, -690))),
        t14a,
    );
    t15 = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t15a, -690), vmulq_n_s32(t14a, 2276))),
        t15a,
    );

    t0 = vminq_s32(vmaxq_s32(vaddq_s32(t0a, t4a), min), max);
    t1 = vminq_s32(vmaxq_s32(vaddq_s32(t1a, t5a), min), max);
    t2 = vminq_s32(vmaxq_s32(vaddq_s32(t2a, t6a), min), max);
    t3 = vminq_s32(vmaxq_s32(vaddq_s32(t3a, t7a), min), max);
    t4 = vminq_s32(vmaxq_s32(vsubq_s32(t0a, t4a), min), max);
    t5 = vminq_s32(vmaxq_s32(vsubq_s32(t1a, t5a), min), max);
    t6 = vminq_s32(vmaxq_s32(vsubq_s32(t2a, t6a), min), max);
    t7 = vminq_s32(vmaxq_s32(vsubq_s32(t3a, t7a), min), max);
    t8a = vminq_s32(vmaxq_s32(vaddq_s32(t8, t12), min), max);
    t9a = vminq_s32(vmaxq_s32(vaddq_s32(t9, t13), min), max);
    t10a = vminq_s32(vmaxq_s32(vaddq_s32(t10, t14), min), max);
    t11a = vminq_s32(vmaxq_s32(vaddq_s32(t11, t15), min), max);
    t12a = vminq_s32(vmaxq_s32(vsubq_s32(t8, t12), min), max);
    t13a = vminq_s32(vmaxq_s32(vsubq_s32(t9, t13), min), max);
    t14a = vminq_s32(vmaxq_s32(vsubq_s32(t10, t14), min), max);
    t15a = vminq_s32(vmaxq_s32(vsubq_s32(t11, t15), min), max);

    t4a = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t4, -312), vmulq_n_s32(t5, 1567))),
        t4,
    );
    t5a = vsubq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(t4, 1567), vmulq_n_s32(t5, -312))),
        t5,
    );
    t6a = vaddq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(t7, -312), vmulq_n_s32(t6, 1567))),
        t7,
    );
    t7a = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t7, 1567), vmulq_n_s32(t6, -312))),
        t6,
    );
    t12 = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t12a, -312), vmulq_n_s32(t13a, 1567))),
        t12a,
    );
    t13 = vsubq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(t12a, 1567), vmulq_n_s32(t13a, -312))),
        t13a,
    );
    t14 = vaddq_s32(
        vrshrq_n_s32::<12>(vsubq_s32(vmulq_n_s32(t15a, -312), vmulq_n_s32(t14a, 1567))),
        t15a,
    );
    t15 = vaddq_s32(
        vrshrq_n_s32::<12>(vaddq_s32(vmulq_n_s32(t15a, 1567), vmulq_n_s32(t14a, -312))),
        t14a,
    );

    c[out_idx(0)] = vminq_s32(vmaxq_s32(vaddq_s32(t0, t2), min), max);
    c[out_idx(15)] = vnegq_s32(vminq_s32(vmaxq_s32(vaddq_s32(t1, t3), min), max));
    t2a = vminq_s32(vmaxq_s32(vsubq_s32(t0, t2), min), max);
    t3a = vminq_s32(vmaxq_s32(vsubq_s32(t1, t3), min), max);
    c[out_idx(3)] = vnegq_s32(vminq_s32(vmaxq_s32(vaddq_s32(t4a, t6a), min), max));
    c[out_idx(12)] = vminq_s32(vmaxq_s32(vaddq_s32(t5a, t7a), min), max);
    t6 = vminq_s32(vmaxq_s32(vsubq_s32(t4a, t6a), min), max);
    t7 = vminq_s32(vmaxq_s32(vsubq_s32(t5a, t7a), min), max);
    c[out_idx(1)] = vnegq_s32(vminq_s32(vmaxq_s32(vaddq_s32(t8a, t10a), min), max));
    c[out_idx(14)] = vminq_s32(vmaxq_s32(vaddq_s32(t9a, t11a), min), max);
    t10 = vminq_s32(vmaxq_s32(vsubq_s32(t8a, t10a), min), max);
    t11 = vminq_s32(vmaxq_s32(vsubq_s32(t9a, t11a), min), max);
    c[out_idx(2)] = vminq_s32(vmaxq_s32(vaddq_s32(t12, t14), min), max);
    c[out_idx(13)] = vnegq_s32(vminq_s32(vmaxq_s32(vaddq_s32(t13, t15), min), max));
    t14a = vminq_s32(vmaxq_s32(vsubq_s32(t12, t14), min), max);
    t15a = vminq_s32(vmaxq_s32(vsubq_s32(t13, t15), min), max);

    c[out_idx(7)] = vnegq_s32(vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t2a, t3a), 181)));
    c[out_idx(8)] = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t2a, t3a), 181));
    c[out_idx(4)] = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t6, t7), 181));
    c[out_idx(11)] = vnegq_s32(vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t6, t7), 181)));
    c[out_idx(6)] = vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t10, t11), 181));
    c[out_idx(9)] = vnegq_s32(vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t10, t11), 181)));
    c[out_idx(5)] = vnegq_s32(vrshrq_n_s32::<8>(vmulq_n_s32(vaddq_s32(t14a, t15a), 181)));
    c[out_idx(10)] = vrshrq_n_s32::<8>(vmulq_n_s32(vsubq_s32(t14a, t15a), 181));
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_identity4_1d(
    c: &mut [int32x4_t],
    stride: usize,
    _min: int32x4_t,
    _max: int32x4_t,
) {
    for i in 0..4 {
        let val: int32x4_t = c[stride * i];
        c[stride * i] = vaddq_s32(val, vrshrq_n_s32::<12>(vmulq_n_s32(val, 1697)));
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_identity8_1d(
    c: &mut [int32x4_t],
    stride: usize,
    _min: int32x4_t,
    _max: int32x4_t,
) {
    for i in 0..8 {
        c[stride * i] = vmulq_n_s32(c[stride * i], 2);
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_identity16_1d(
    c: &mut [int32x4_t],
    stride: usize,
    _min: int32x4_t,
    _max: int32x4_t,
) {
    for i in 0..16 {
        let val: int32x4_t = c[stride * i];
        c[stride * i] = vaddq_s32(
            vmulq_n_s32(val, 2),
            vrshrq_n_s32::<11>(vmulq_n_s32(val, 1697)),
        );
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_identity32_1d(
    c: &mut [int32x4_t],
    stride: usize,
    _min: int32x4_t,
    _max: int32x4_t,
) {
    for i in 0..32 {
        c[stride * i] = vmulq_n_s32(c[stride * i], 4);
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn dav1d_inv_wht4_1d_internal(c: &mut [int32x4_t], stride: usize) {
    let in0: int32x4_t = c[0];
    let in1: int32x4_t = c[stride];
    let in2: int32x4_t = c[2 * stride];
    let in3: int32x4_t = c[3 * stride];

    let t0: int32x4_t = vaddq_s32(in0, in1);
    let t2: int32x4_t = vsubq_s32(in2, in3);
    let t4: int32x4_t = vshrq_n_s32::<1>(vsubq_s32(t0, t2));
    let t3: int32x4_t = vsubq_s32(t4, in3);
    let t1: int32x4_t = vsubq_s32(t4, in1);

    c[0] = vsubq_s32(t0, t3);
    c[stride] = t3;
    c[2 * stride] = t1;
    c[3 * stride] = vaddq_s32(t2, t1);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_dct4_1d_c(c: &mut [int32x4_t], stride: usize, min: int32x4_t, max: int32x4_t) {
    inv_dct4_1d_internal(c, stride, min, max, false);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_dct8_1d_c(c: &mut [int32x4_t], stride: usize, min: int32x4_t, max: int32x4_t) {
    inv_dct8_1d_internal(c, stride, min, max, false);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_dct16_1d_c(c: &mut [int32x4_t], stride: usize, min: int32x4_t, max: int32x4_t) {
    inv_dct16_1d_internal(c, stride, min, max, false);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_dct32_1d_c(c: &mut [int32x4_t], stride: usize, min: int32x4_t, max: int32x4_t) {
    inv_dct32_1d_internal(c, stride, min, max, false);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_dct64_1d_c(c: &mut [int32x4_t], stride: usize, min: int32x4_t, max: int32x4_t) {
    inv_dct64_1d_internal(c, stride, min, max);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_adst4_1d_c(c: &mut [int32x4_t], stride: usize, min: int32x4_t, max: int32x4_t) {
    inv_adst4_1d_internal(c, stride, min, max, false);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_flipadst4_1d_c(
    c: &mut [int32x4_t],
    stride: usize,
    min: int32x4_t,
    max: int32x4_t,
) {
    inv_adst4_1d_internal(c, stride, min, max, true);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_adst8_1d_c(c: &mut [int32x4_t], stride: usize, min: int32x4_t, max: int32x4_t) {
    inv_adst8_1d_internal(c, stride, min, max, false);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_flipadst8_1d_c(
    c: &mut [int32x4_t],
    stride: usize,
    min: int32x4_t,
    max: int32x4_t,
) {
    inv_adst8_1d_internal(c, stride, min, max, true);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_adst16_1d_c(c: &mut [int32x4_t], stride: usize, min: int32x4_t, max: int32x4_t) {
    inv_adst16_1d_internal(c, stride, min, max, false);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_flipadst16_1d_c(
    c: &mut [int32x4_t],
    stride: usize,
    min: int32x4_t,
    max: int32x4_t,
) {
    inv_adst16_1d_internal(c, stride, min, max, true);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_identity4_1d_c(
    c: &mut [int32x4_t],
    stride: usize,
    min: int32x4_t,
    max: int32x4_t,
) {
    inv_identity4_1d(c, stride, min, max);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_identity8_1d_c(
    c: &mut [int32x4_t],
    stride: usize,
    min: int32x4_t,
    max: int32x4_t,
) {
    inv_identity8_1d(c, stride, min, max);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_identity16_1d_c(
    c: &mut [int32x4_t],
    stride: usize,
    min: int32x4_t,
    max: int32x4_t,
) {
    inv_identity16_1d(c, stride, min, max);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn inv_identity32_1d_c(
    c: &mut [int32x4_t],
    stride: usize,
    min: int32x4_t,
    max: int32x4_t,
) {
    inv_identity32_1d(c, stride, min, max);
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_lines, clippy::similar_names, unused_mut)]
pub(super) fn dav1d_inv_wht4_1d_c(c: &mut [int32x4_t], stride: usize) {
    dav1d_inv_wht4_1d_internal(c, stride);
}
