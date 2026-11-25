//! 8 bpc loop filter (deblocking) x86_64 SSSE3 çekirdekleri (`D02707`, `D02713`,
//! `D02719`, `D02725`; P46 Adım 2 / Paket A).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Kayıt | Yuva | Sembol | Kaynak |
//! |---|---|---|---|
//! | `D02707` | `loop_filter_sb[0][0]` | `dav1d_lpf_h_sb_y_8bpc_ssse3` | `src/x86/loopfilter_sse.asm:2078-2180`, `src/x86/loopfilter.h:46` |
//! | `D02713` | `loop_filter_sb[0][1]` | `dav1d_lpf_v_sb_y_8bpc_ssse3` | `src/x86/loopfilter_sse.asm:1980-2076`, `src/x86/loopfilter.h:47` |
//! | `D02719` | `loop_filter_sb[1][0]` | `dav1d_lpf_h_sb_uv_8bpc_ssse3` | `src/x86/loopfilter_sse.asm:2264-2340`, `src/x86/loopfilter.h:48` |
//! | `D02725` | `loop_filter_sb[1][1]` | `dav1d_lpf_v_sb_uv_8bpc_ssse3` | `src/x86/loopfilter_sse.asm:2182-2262`, `src/x86/loopfilter.h:49` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bu modülde tüm SIMD işlevleri yalnız `#[target_feature(enable = "ssse3")]`
//!   taşır; SSE4.1+, VEX ve EVEX yönergesi üretilmez (`0` SSE4.1+/VEX/EVEX).
//! - Kopyasız O(1) sözleşme: girişte [`lpf_sb_sozlesmesi::<8>`] denetlenir;
//!   sözleşme dışı girdide [`lpf_sb_skaler_devret::<8>`] ile skaler gövdeye
//!   devredilir.
//! - Sınır-güvenli (`SentinelBuffer` / `OverReadProbe`): yalnız etkin
//!   (`v != 0 && l != 0`) kenarların kendi filtre genişliği (`wd ∈ {4, 6, 8, 16}`)
//!   kapsamındaki komşu pikselleri okunur ve yalnız değiştirilen genişlik
//!   (`wd ∈ {4, 6}` → 4 piksel, `wd == 8` → 6 piksel, `wd == 16` → 12 piksel)
//!   geri yazılır.

#![allow(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use core::arch::x86_64::{
    __m128i, _mm_abs_epi16, _mm_add_epi16, _mm_and_si128, _mm_andnot_si128, _mm_cmpeq_epi16,
    _mm_cmpgt_epi16, _mm_max_epi16, _mm_min_epi16, _mm_movemask_epi8, _mm_mullo_epi16,
    _mm_or_si128, _mm_packus_epi16, _mm_set1_epi16, _mm_setzero_si128, _mm_slli_epi16,
    _mm_srai_epi16, _mm_srli_epi16, _mm_sub_epi16, _mm_unpacklo_epi8,
};

use crate::cpu::Ssse3Token;
use crate::dsp::lf::{LimLut, LoopFilterSbFn, lpf_sb_skaler_devret, lpf_sb_sozlesmesi};
use crate::dsp::x86::mem;

/// 8 bpc `loop_filter_sb[plane][dir]` SSSE3 kurucusu (`D02707`, `D02713`,
/// `D02719`, `D02725`).
#[must_use]
pub const fn loop_filter_sb_fn(
    _token: Ssse3Token,
    plane: usize,
    dir: usize,
) -> Option<LoopFilterSbFn> {
    match (plane, dir) {
        (0, 0) => Some(lpf_h_sb_y_8bpc_ssse3),
        (0, 1) => Some(lpf_v_sb_y_8bpc_ssse3),
        (1, 0) => Some(lpf_h_sb_uv_8bpc_ssse3),
        (1, 1) => Some(lpf_v_sb_uv_8bpc_ssse3),
        _ => None,
    }
}

fn lpf_h_sb_y_8bpc_ssse3(
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
    lpf_sb_8bpc_ssse3::<0, 0>(
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

fn lpf_v_sb_y_8bpc_ssse3(
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
    lpf_sb_8bpc_ssse3::<0, 1>(
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

fn lpf_h_sb_uv_8bpc_ssse3(
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
    lpf_sb_8bpc_ssse3::<1, 0>(
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

fn lpf_v_sb_uv_8bpc_ssse3(
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
    lpf_sb_8bpc_ssse3::<1, 1>(
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

#[inline(always)]
fn lpf_sb_8bpc_ssse3<const PLANE: usize, const DIR: usize>(
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
    if !lpf_sb_sozlesmesi::<8>(
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
        lpf_sb_skaler_devret::<8>(
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
        return;
    }
    if count == 0 || w4 == 0 || (DIR == 0 && w4 <= 1) || (DIR == 1 && y0 == 0 && count == 1) {
        return;
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`loop_filter_sb_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır.
    // (2) sınır: `lpf_sb_sozlesmesi::<8>` `dst`, `mask` ve `level` dilimlerinin
    // sınırlarını O(1) doğruladı; çekirdek yalnız etkin kenarların kendi `wd`
    // penceresini sınır denetimli dizi yardımcılarıyla okur ve yazar.
    // (3) hizalama/takma ad: tüm vektör erişimleri `mem` içindeki hizasız sabit
    // boyutlu dizi sarmalayıcılarıyla yapılır; `dst` `&mut` tek yazma yoludur.
    unsafe {
        if DIR == 0 {
            lpf_h_sb_ssse3_giris::<PLANE>(
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
        } else {
            lpf_v_sb_ssse3_giris::<PLANE>(
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
    }
}

#[inline(always)]
fn level_or(level: &[[u8; 4]], at: usize, fallback: usize, idx: usize) -> usize {
    let l = level[at][idx];
    usize::from(if l != 0 { l } else { level[fallback][idx] })
}

#[inline(always)]
const fn edge_wd<const PLANE: usize>(v: u8) -> usize {
    if PLANE == 0 {
        4 << (v - 1)
    } else {
        4 + 2 * ((v - 1) as usize)
    }
}

#[derive(Clone, Copy, Default)]
struct EdgeSlot {
    off: usize,
    wd: usize,
    e: i16,
    i: i16,
    h: i16,
}

#[target_feature(enable = "ssse3")]
#[inline]
fn le_epi16_ssse3(a: __m128i, b: __m128i) -> __m128i {
    _mm_andnot_si128(_mm_cmpgt_epi16(a, b), _mm_set1_epi16(-1))
}

#[target_feature(enable = "ssse3")]
#[inline]
fn blend_ssse3(mask: __m128i, a: __m128i, b: __m128i) -> __m128i {
    _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b))
}

#[target_feature(enable = "ssse3")]
#[inline]
fn load_tap_ssse3(row: &[u8; 16]) -> __m128i {
    _mm_unpacklo_epi8(mem::load_u8x16(row), _mm_setzero_si128())
}

#[target_feature(enable = "ssse3")]
#[inline]
fn store_tap_ssse3(row: &mut [u8; 16], v: __m128i) {
    mem::store_u8x16(row, _mm_packus_epi16(v, v));
}

/// 8 şerit (`2 × 4×4` kenar) paralel 8 bpc loop filter SSSE3 çekirdeği.
#[target_feature(enable = "ssse3")]
#[inline]
fn filter_batch_ssse3<const PLANE: usize>(
    taps: &mut [[u8; 16]; 14],
    e_arr: &[i16; 8],
    i_arr: &[i16; 8],
    h_arr: &[i16; 8],
    wd_arr: &[i16; 8],
    max_wd: usize,
) {
    let e_v = mem::load_i16x8(e_arr);
    let i_v = mem::load_i16x8(i_arr);
    let h_v = mem::load_i16x8(h_arr);
    let wd_v = mem::load_i16x8(wd_arr);

    let p1 = load_tap_ssse3(&taps[5]);
    let p0 = load_tap_ssse3(&taps[6]);
    let q0 = load_tap_ssse3(&taps[7]);
    let q1 = load_tap_ssse3(&taps[8]);

    let abs_p1_p0 = _mm_abs_epi16(_mm_sub_epi16(p1, p0));
    let abs_q1_q0 = _mm_abs_epi16(_mm_sub_epi16(q1, q0));
    let abs_p0_q0 = _mm_abs_epi16(_mm_sub_epi16(p0, q0));
    let abs_p1_q1 = _mm_abs_epi16(_mm_sub_epi16(p1, q1));

    let term_e = _mm_add_epi16(
        _mm_slli_epi16::<1>(abs_p0_q0),
        _mm_srli_epi16::<1>(abs_p1_q1),
    );
    let mut fm = _mm_and_si128(
        _mm_cmpgt_epi16(wd_v, _mm_setzero_si128()),
        _mm_and_si128(
            le_epi16_ssse3(abs_p1_p0, i_v),
            _mm_and_si128(le_epi16_ssse3(abs_q1_q0, i_v), le_epi16_ssse3(term_e, e_v)),
        ),
    );

    let (p2, q2) = if max_wd >= 6 {
        let p2 = load_tap_ssse3(&taps[4]);
        let q2 = load_tap_ssse3(&taps[9]);
        let c6 = _mm_and_si128(
            le_epi16_ssse3(_mm_abs_epi16(_mm_sub_epi16(p2, p1)), i_v),
            le_epi16_ssse3(_mm_abs_epi16(_mm_sub_epi16(q2, q1)), i_v),
        );
        let gt4 = _mm_cmpgt_epi16(wd_v, _mm_set1_epi16(4));
        fm = _mm_and_si128(fm, blend_ssse3(gt4, c6, _mm_set1_epi16(-1)));
        (p2, q2)
    } else {
        (_mm_setzero_si128(), _mm_setzero_si128())
    };

    let (p3, q3) = if PLANE == 0 && max_wd >= 8 {
        let p3 = load_tap_ssse3(&taps[3]);
        let q3 = load_tap_ssse3(&taps[10]);
        let c8 = _mm_and_si128(
            le_epi16_ssse3(_mm_abs_epi16(_mm_sub_epi16(p3, p2)), i_v),
            le_epi16_ssse3(_mm_abs_epi16(_mm_sub_epi16(q3, q2)), i_v),
        );
        let gt6 = _mm_cmpgt_epi16(wd_v, _mm_set1_epi16(6));
        fm = _mm_and_si128(fm, blend_ssse3(gt6, c8, _mm_set1_epi16(-1)));
        (p3, q3)
    } else {
        (_mm_setzero_si128(), _mm_setzero_si128())
    };

    if _mm_movemask_epi8(fm) == 0 {
        return;
    }

    let one = _mm_set1_epi16(1);
    let flat6 = if max_wd >= 6 {
        _mm_and_si128(
            _mm_and_si128(
                le_epi16_ssse3(abs_p1_p0, one),
                le_epi16_ssse3(abs_q1_q0, one),
            ),
            _mm_and_si128(
                le_epi16_ssse3(_mm_abs_epi16(_mm_sub_epi16(p2, p0)), one),
                le_epi16_ssse3(_mm_abs_epi16(_mm_sub_epi16(q2, q0)), one),
            ),
        )
    } else {
        _mm_setzero_si128()
    };

    let flat8 = if PLANE == 0 && max_wd >= 8 {
        _mm_and_si128(
            flat6,
            _mm_and_si128(
                le_epi16_ssse3(_mm_abs_epi16(_mm_sub_epi16(p3, p0)), one),
                le_epi16_ssse3(_mm_abs_epi16(_mm_sub_epi16(q3, q0)), one),
            ),
        )
    } else {
        _mm_setzero_si128()
    };

    let (p6, p5, p4, q4, q5, q6, m16) = if PLANE == 0 && max_wd >= 16 {
        let p6 = load_tap_ssse3(&taps[0]);
        let p5 = load_tap_ssse3(&taps[1]);
        let p4 = load_tap_ssse3(&taps[2]);
        let q4 = load_tap_ssse3(&taps[11]);
        let q5 = load_tap_ssse3(&taps[12]);
        let q6 = load_tap_ssse3(&taps[13]);
        let flat8out = _mm_and_si128(
            _mm_and_si128(
                le_epi16_ssse3(_mm_abs_epi16(_mm_sub_epi16(p6, p0)), one),
                le_epi16_ssse3(_mm_abs_epi16(_mm_sub_epi16(p5, p0)), one),
            ),
            _mm_and_si128(
                _mm_and_si128(
                    le_epi16_ssse3(_mm_abs_epi16(_mm_sub_epi16(p4, p0)), one),
                    le_epi16_ssse3(_mm_abs_epi16(_mm_sub_epi16(q4, q0)), one),
                ),
                _mm_and_si128(
                    le_epi16_ssse3(_mm_abs_epi16(_mm_sub_epi16(q5, q0)), one),
                    le_epi16_ssse3(_mm_abs_epi16(_mm_sub_epi16(q6, q0)), one),
                ),
            ),
        );
        let is16 = _mm_cmpeq_epi16(wd_v, _mm_set1_epi16(16));
        let m16 = _mm_and_si128(fm, _mm_and_si128(is16, _mm_and_si128(flat8, flat8out)));
        (p6, p5, p4, q4, q5, q6, m16)
    } else {
        let z = _mm_setzero_si128();
        (z, z, z, z, z, z, z)
    };

    let m8 = if PLANE == 0 && max_wd >= 8 {
        let ge8 = _mm_cmpgt_epi16(wd_v, _mm_set1_epi16(6));
        _mm_andnot_si128(m16, _mm_and_si128(fm, _mm_and_si128(ge8, flat8)))
    } else {
        _mm_setzero_si128()
    };

    let m6 = if PLANE == 1 && max_wd >= 6 {
        let is6 = _mm_cmpeq_epi16(wd_v, _mm_set1_epi16(6));
        _mm_and_si128(fm, _mm_and_si128(is6, flat6))
    } else {
        _mm_setzero_si128()
    };

    let m_wide = _mm_or_si128(m16, _mm_or_si128(m8, m6));
    let m4 = _mm_andnot_si128(m_wide, fm);

    let mut out_p5 = p5;
    let mut out_p4 = p4;
    let mut out_p3 = p3;
    let mut out_p2 = p2;
    let mut out_p1 = p1;
    let mut out_p0 = p0;
    let mut out_q0 = q0;
    let mut out_q1 = q1;
    let mut out_q2 = q2;
    let mut out_q3 = q3;
    let mut out_q4 = q4;
    let mut out_q5 = q5;

    // 4-tap dalı
    if _mm_movemask_epi8(m4) != 0 {
        let hev = _mm_or_si128(
            _mm_cmpgt_epi16(abs_p1_p0, h_v),
            _mm_cmpgt_epi16(abs_q1_q0, h_v),
        );
        let lo = _mm_set1_epi16(-128);
        let hi = _mm_set1_epi16(127);
        let clip = |v: __m128i| _mm_max_epi16(_mm_min_epi16(v, hi), lo);

        let fv_hev = _mm_and_si128(hev, clip(_mm_sub_epi16(p1, q1)));
        let q0_m_p0_3 = _mm_mullo_epi16(_mm_sub_epi16(q0, p0), _mm_set1_epi16(3));
        let fv = clip(_mm_add_epi16(q0_m_p0_3, fv_hev));
        let f1 = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(fv, _mm_set1_epi16(4)), hi));
        let f2 = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(fv, _mm_set1_epi16(3)), hi));
        let fv2 = _mm_andnot_si128(hev, _mm_srai_epi16::<1>(_mm_add_epi16(f1, one)));

        out_p1 = blend_ssse3(m4, _mm_add_epi16(p1, fv2), out_p1);
        out_p0 = blend_ssse3(m4, _mm_add_epi16(p0, f2), out_p0);
        out_q0 = blend_ssse3(m4, _mm_sub_epi16(q0, f1), out_q0);
        out_q1 = blend_ssse3(m4, _mm_sub_epi16(q1, fv2), out_q1);
    }

    // 6-tap dalı (yalnız kroma `PLANE == 1`)
    if PLANE == 1 && max_wd >= 6 && _mm_movemask_epi8(m6) != 0 {
        let four = _mm_set1_epi16(4);
        let p2_2 = _mm_slli_epi16::<1>(p2);
        let p1_2 = _mm_slli_epi16::<1>(p1);
        let p0_2 = _mm_slli_epi16::<1>(p0);
        let q0_2 = _mm_slli_epi16::<1>(q0);
        let q1_2 = _mm_slli_epi16::<1>(q1);
        let q2_2 = _mm_slli_epi16::<1>(q2);

        let mid = _mm_add_epi16(p0_2, four);
        let mid_q = _mm_add_epi16(q0_2, four);
        // p1' = (p2*3 + p1*2 + p0*2 + q0 + 4) >> 3
        let r_p1 = _mm_srli_epi16::<3>(_mm_add_epi16(
            _mm_add_epi16(_mm_add_epi16(p2_2, p2), p1_2),
            _mm_add_epi16(mid, q0),
        ));
        // p0' = (p2 + p1*2 + p0*2 + q0*2 + q1 + 4) >> 3
        let r_p0 = _mm_srli_epi16::<3>(_mm_add_epi16(
            _mm_add_epi16(p2, p1_2),
            _mm_add_epi16(_mm_add_epi16(mid, q0_2), q1),
        ));
        // q0' = (p1 + p0*2 + q0*2 + q1*2 + q2 + 4) >> 3
        let r_q0 = _mm_srli_epi16::<3>(_mm_add_epi16(
            _mm_add_epi16(p1, p0_2),
            _mm_add_epi16(_mm_add_epi16(mid_q, q1_2), q2),
        ));
        // q1' = (p0 + q0*2 + q1*2 + q2*3 + 4) >> 3
        let r_q1 = _mm_srli_epi16::<3>(_mm_add_epi16(
            _mm_add_epi16(p0, mid_q),
            _mm_add_epi16(q1_2, _mm_add_epi16(q2_2, q2)),
        ));

        out_p1 = blend_ssse3(m6, r_p1, out_p1);
        out_p0 = blend_ssse3(m6, r_p0, out_p0);
        out_q0 = blend_ssse3(m6, r_q0, out_q0);
        out_q1 = blend_ssse3(m6, r_q1, out_q1);
    }

    // 8-tap dalı (yalnız luma `PLANE == 0`)
    if PLANE == 0 && max_wd >= 8 && _mm_movemask_epi8(m8) != 0 {
        let four = _mm_set1_epi16(4);
        let p3_2 = _mm_slli_epi16::<1>(p3);
        let p2_2 = _mm_slli_epi16::<1>(p2);
        let p1_2 = _mm_slli_epi16::<1>(p1);
        let p0_2 = _mm_slli_epi16::<1>(p0);
        let q0_2 = _mm_slli_epi16::<1>(q0);
        let q1_2 = _mm_slli_epi16::<1>(q1);
        let q2_2 = _mm_slli_epi16::<1>(q2);
        let q3_2 = _mm_slli_epi16::<1>(q3);

        let p1_p0_q0_4 = _mm_add_epi16(_mm_add_epi16(p1, p0), _mm_add_epi16(q0, four));
        let p0_q0_q1_4 = _mm_add_epi16(_mm_add_epi16(p0, q0), _mm_add_epi16(q1, four));

        let r_p2 = _mm_srli_epi16::<3>(_mm_add_epi16(
            _mm_add_epi16(p3_2, p3),
            _mm_add_epi16(p2_2, p1_p0_q0_4),
        ));
        let r_p1 = _mm_srli_epi16::<3>(_mm_add_epi16(
            _mm_add_epi16(p3_2, p2),
            _mm_add_epi16(p1_2, p0_q0_q1_4),
        ));
        let r_p0 = _mm_srli_epi16::<3>(_mm_add_epi16(
            _mm_add_epi16(p3, p2),
            _mm_add_epi16(
                _mm_add_epi16(p1, p0_2),
                _mm_add_epi16(_mm_add_epi16(q0, q1), _mm_add_epi16(q2, four)),
            ),
        ));
        let r_q0 = _mm_srli_epi16::<3>(_mm_add_epi16(
            _mm_add_epi16(p2, p1),
            _mm_add_epi16(
                _mm_add_epi16(p0, q0_2),
                _mm_add_epi16(_mm_add_epi16(q1, q2), _mm_add_epi16(q3, four)),
            ),
        ));
        let r_q1 = _mm_srli_epi16::<3>(_mm_add_epi16(
            _mm_add_epi16(p1_p0_q0_4, q1_2),
            _mm_add_epi16(q2, q3_2),
        ));
        let r_q2 = _mm_srli_epi16::<3>(_mm_add_epi16(
            _mm_add_epi16(p0_q0_q1_4, q2_2),
            _mm_add_epi16(q3_2, q3),
        ));

        out_p2 = blend_ssse3(m8, r_p2, out_p2);
        out_p1 = blend_ssse3(m8, r_p1, out_p1);
        out_p0 = blend_ssse3(m8, r_p0, out_p0);
        out_q0 = blend_ssse3(m8, r_q0, out_q0);
        out_q1 = blend_ssse3(m8, r_q1, out_q1);
        out_q2 = blend_ssse3(m8, r_q2, out_q2);
    }

    // 16-tap dalı (yalnız luma `PLANE == 0`)
    if PLANE == 0 && max_wd >= 16 && _mm_movemask_epi8(m16) != 0 {
        let eight = _mm_set1_epi16(8);
        // s5 = p6*7 + p5*2 + p4*2 + p3 + p2 + p1 + p0 + q0 + 8
        let p6_2 = _mm_slli_epi16::<1>(p6);
        let p6_4 = _mm_slli_epi16::<2>(p6);
        let p6_7 = _mm_sub_epi16(_mm_slli_epi16::<3>(p6), p6);
        let p5_2 = _mm_slli_epi16::<1>(p5);
        let p4_2 = _mm_slli_epi16::<1>(p4);
        let q4_2 = _mm_slli_epi16::<1>(q4);
        let q5_2 = _mm_slli_epi16::<1>(q5);
        let q6_2 = _mm_slli_epi16::<1>(q6);
        let q6_4 = _mm_slli_epi16::<2>(q6);
        let q6_7 = _mm_sub_epi16(_mm_slli_epi16::<3>(q6), q6);

        let sum_p3_q0 = _mm_add_epi16(
            _mm_add_epi16(p3, p2),
            _mm_add_epi16(p1, _mm_add_epi16(p0, q0)),
        );
        let sum_p0_q3 = _mm_add_epi16(
            _mm_add_epi16(p0, q0),
            _mm_add_epi16(q1, _mm_add_epi16(q2, q3)),
        );

        let s_p5 = _mm_add_epi16(
            _mm_add_epi16(p6_7, _mm_add_epi16(p5_2, p4_2)),
            _mm_add_epi16(sum_p3_q0, eight),
        );
        // Ardışık farklarla p4..q5 toplamları:
        // s_p4 = s_p5 - 2*p6 + p3 + q1
        let s_p4 = _mm_add_epi16(_mm_sub_epi16(s_p5, p6_2), _mm_add_epi16(p3, q1));
        // s_p3 = s_p4 - p6 - p5 + p2 + q2
        let s_p3 = _mm_add_epi16(
            _mm_sub_epi16(s_p4, _mm_add_epi16(p6, p5)),
            _mm_add_epi16(p2, q2),
        );
        // s_p2 = s_p3 - p6 - p4 + p1 + q3
        let s_p2 = _mm_add_epi16(
            _mm_sub_epi16(s_p3, _mm_add_epi16(p6, p4)),
            _mm_add_epi16(p1, q3),
        );
        // s_p1 = s_p2 - p6 - p3 + p0 + q4
        let s_p1 = _mm_add_epi16(
            _mm_sub_epi16(s_p2, _mm_add_epi16(p6, p3)),
            _mm_add_epi16(p0, q4),
        );
        // s_p0 = s_p1 - p6 - p2 + q0 + q5
        let s_p0 = _mm_add_epi16(
            _mm_sub_epi16(s_p1, _mm_add_epi16(p6, p2)),
            _mm_add_epi16(q0, q5),
        );
        // s_q0 = s_p0 - p6 - p1 + q1 + q6
        let s_q0 = _mm_add_epi16(
            _mm_sub_epi16(s_p0, _mm_add_epi16(p6, p1)),
            _mm_add_epi16(q1, q6),
        );
        // s_q1 = s_q0 - p5 - p0 + q2 + q6
        let s_q1 = _mm_add_epi16(
            _mm_sub_epi16(s_q0, _mm_add_epi16(p5, p0)),
            _mm_add_epi16(q2, q6),
        );
        // s_q2 = s_q1 - p4 - q0 + q3 + q6
        let s_q2 = _mm_add_epi16(
            _mm_sub_epi16(s_q1, _mm_add_epi16(p4, q0)),
            _mm_add_epi16(q3, q6),
        );
        // s_q3 = s_q2 - p3 - q1 + q4 + q6
        let s_q3 = _mm_add_epi16(
            _mm_sub_epi16(s_q2, _mm_add_epi16(p3, q1)),
            _mm_add_epi16(q4, q6),
        );
        // s_q4 = s_q3 - p2 - q2 + q5 + q6
        let s_q4 = _mm_add_epi16(
            _mm_sub_epi16(s_q3, _mm_add_epi16(p2, q2)),
            _mm_add_epi16(q5, q6),
        );
        // s_q5 = p0 + q0 + q1 + q2 + q3 + q4*2 + q5*2 + q6*7 + 8
        let s_q5 = _mm_add_epi16(
            _mm_add_epi16(q6_7, _mm_add_epi16(q5_2, q4_2)),
            _mm_add_epi16(sum_p0_q3, eight),
        );
        let _ = (p6_4, q6_2, q6_4);

        out_p5 = blend_ssse3(m16, _mm_srli_epi16::<4>(s_p5), out_p5);
        out_p4 = blend_ssse3(m16, _mm_srli_epi16::<4>(s_p4), out_p4);
        out_p3 = blend_ssse3(m16, _mm_srli_epi16::<4>(s_p3), out_p3);
        out_p2 = blend_ssse3(m16, _mm_srli_epi16::<4>(s_p2), out_p2);
        out_p1 = blend_ssse3(m16, _mm_srli_epi16::<4>(s_p1), out_p1);
        out_p0 = blend_ssse3(m16, _mm_srli_epi16::<4>(s_p0), out_p0);
        out_q0 = blend_ssse3(m16, _mm_srli_epi16::<4>(s_q0), out_q0);
        out_q1 = blend_ssse3(m16, _mm_srli_epi16::<4>(s_q1), out_q1);
        out_q2 = blend_ssse3(m16, _mm_srli_epi16::<4>(s_q2), out_q2);
        out_q3 = blend_ssse3(m16, _mm_srli_epi16::<4>(s_q3), out_q3);
        out_q4 = blend_ssse3(m16, _mm_srli_epi16::<4>(s_q4), out_q4);
        out_q5 = blend_ssse3(m16, _mm_srli_epi16::<4>(s_q5), out_q5);
    }

    store_tap_ssse3(&mut taps[5], out_p1);
    store_tap_ssse3(&mut taps[6], out_p0);
    store_tap_ssse3(&mut taps[7], out_q0);
    store_tap_ssse3(&mut taps[8], out_q1);
    if PLANE == 0 && max_wd >= 8 {
        store_tap_ssse3(&mut taps[4], out_p2);
        store_tap_ssse3(&mut taps[9], out_q2);
    }
    if PLANE == 0 && max_wd >= 16 {
        store_tap_ssse3(&mut taps[1], out_p5);
        store_tap_ssse3(&mut taps[2], out_p4);
        store_tap_ssse3(&mut taps[3], out_p3);
        store_tap_ssse3(&mut taps[10], out_q3);
        store_tap_ssse3(&mut taps[11], out_q4);
        store_tap_ssse3(&mut taps[12], out_q5);
    }
}

#[target_feature(enable = "ssse3")]
fn flush_h_batch_ssse3<const PLANE: usize>(
    dst: &mut [u8],
    stride: usize,
    batch: &[EdgeSlot; 2],
    n: usize,
    max_wd: usize,
) {
    let mut taps = [[0u8; 16]; 14];
    let mut e_arr = [0i16; 8];
    let mut i_arr = [0i16; 8];
    let mut h_arr = [0i16; 8];
    let mut wd_arr = [0i16; 8];

    for b in 0..n {
        let slot = batch[b];
        let base_lane = b * 4;
        for k in 0..4 {
            let lane = base_lane + k;
            e_arr[lane] = slot.e;
            i_arr[lane] = slot.i;
            h_arr[lane] = slot.h;
            wd_arr[lane] = slot.wd as i16;
            let row_off = slot.off + k * stride;
            match slot.wd {
                4 => {
                    let s = mem::chunk::<u8, 4>(dst, row_off - 2);
                    for t in 0..4 {
                        taps[5 + t][lane] = s[t];
                    }
                }
                6 => {
                    let s = mem::chunk::<u8, 6>(dst, row_off - 3);
                    for t in 0..6 {
                        taps[4 + t][lane] = s[t];
                    }
                }
                8 => {
                    let s = mem::chunk::<u8, 8>(dst, row_off - 4);
                    for t in 0..8 {
                        taps[3 + t][lane] = s[t];
                    }
                }
                _ => {
                    let s = mem::chunk::<u8, 14>(dst, row_off - 7);
                    for t in 0..14 {
                        taps[t][lane] = s[t];
                    }
                }
            }
        }
    }

    filter_batch_ssse3::<PLANE>(&mut taps, &e_arr, &i_arr, &h_arr, &wd_arr, max_wd);

    for b in 0..n {
        let slot = batch[b];
        let base_lane = b * 4;
        for k in 0..4 {
            let lane = base_lane + k;
            let row_off = slot.off + k * stride;
            match slot.wd {
                4 | 6 => {
                    let d = mem::chunk_mut::<u8, 4>(dst, row_off - 2);
                    for t in 0..4 {
                        d[t] = taps[5 + t][lane];
                    }
                }
                8 => {
                    let d = mem::chunk_mut::<u8, 6>(dst, row_off - 3);
                    for t in 0..6 {
                        d[t] = taps[4 + t][lane];
                    }
                }
                _ => {
                    let d = mem::chunk_mut::<u8, 12>(dst, row_off - 6);
                    for t in 0..12 {
                        d[t] = taps[1 + t][lane];
                    }
                }
            }
        }
    }
}

#[target_feature(enable = "ssse3")]
fn flush_v_batch_ssse3<const PLANE: usize>(
    dst: &mut [u8],
    stride: usize,
    batch: &[EdgeSlot; 2],
    n: usize,
    max_wd: usize,
) {
    let mut taps = [[0u8; 16]; 14];
    let mut e_arr = [0i16; 8];
    let mut i_arr = [0i16; 8];
    let mut h_arr = [0i16; 8];
    let mut wd_arr = [0i16; 8];

    for b in 0..n {
        let slot = batch[b];
        let base_lane = b * 4;
        for k in 0..4 {
            let lane = base_lane + k;
            e_arr[lane] = slot.e;
            i_arr[lane] = slot.i;
            h_arr[lane] = slot.h;
            wd_arr[lane] = slot.wd as i16;
        }
        let (t_lo, t_hi) = match slot.wd {
            4 => (5usize, 9usize),
            6 => (4, 10),
            8 => (3, 11),
            _ => (0, 14),
        };
        for t in t_lo..t_hi {
            let row_off = (slot.off + t * stride) - 7 * stride;
            let s = mem::chunk::<u8, 4>(dst, row_off);
            taps[t][base_lane..base_lane + 4].copy_from_slice(s);
        }
    }

    filter_batch_ssse3::<PLANE>(&mut taps, &e_arr, &i_arr, &h_arr, &wd_arr, max_wd);

    for b in 0..n {
        let slot = batch[b];
        let base_lane = b * 4;
        let (t_lo, t_hi) = match slot.wd {
            4 | 6 => (5usize, 9usize),
            8 => (4, 10),
            _ => (1, 13),
        };
        for t in t_lo..t_hi {
            let row_off = (slot.off + t * stride) - 7 * stride;
            let d = mem::chunk_mut::<u8, 4>(dst, row_off);
            d.copy_from_slice(&taps[t][base_lane..base_lane + 4]);
        }
    }
}

#[target_feature(enable = "ssse3")]
fn lpf_h_sb_ssse3_giris<const PLANE: usize>(
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
    for gx in 1..w4 {
        let mut batch = [EdgeSlot::default(); 2];
        let mut n = 0usize;
        let mut max_wd = 0usize;
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
            let wd = edge_wd::<PLANE>(v);
            let off = gy * 4 * stride + gx * 4;
            batch[n] = EdgeSlot {
                off,
                wd,
                e: lut.e[l] as i16,
                i: lut.i[l] as i16,
                h: (l >> 4) as i16,
            };
            if wd > max_wd {
                max_wd = wd;
            }
            n += 1;
            if n == 2 {
                flush_h_batch_ssse3::<PLANE>(dst, stride, &batch, 2, max_wd);
                n = 0;
                max_wd = 0;
            }
        }
        if n > 0 {
            flush_h_batch_ssse3::<PLANE>(dst, stride, &batch, n, max_wd);
        }
    }
}

#[target_feature(enable = "ssse3")]
fn lpf_v_sb_ssse3_giris<const PLANE: usize>(
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
    for gy in y0..y0 + count {
        if gy == 0 {
            continue;
        }
        let mut batch = [EdgeSlot::default(); 2];
        let mut n = 0usize;
        let mut max_wd = 0usize;
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
            let wd = edge_wd::<PLANE>(v);
            let off = gy * 4 * stride + gx * 4;
            batch[n] = EdgeSlot {
                off,
                wd,
                e: lut.e[l] as i16,
                i: lut.i[l] as i16,
                h: (l >> 4) as i16,
            };
            if wd > max_wd {
                max_wd = wd;
            }
            n += 1;
            if n == 2 {
                flush_v_batch_ssse3::<PLANE>(dst, stride, &batch, 2, max_wd);
                n = 0;
                max_wd = 0;
            }
        }
        if n > 0 {
            flush_v_batch_ssse3::<PLANE>(dst, stride, &batch, n, max_wd);
        }
    }
}
