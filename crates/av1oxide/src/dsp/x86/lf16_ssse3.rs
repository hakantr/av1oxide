//! SSSE3 16-bpc (`10/12` bit) loop filter (deblocking) çekirdekleri (`4` kayıt / `8` yuva; P46 Adım 2).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Yuva | Sembol | Kaynak |
//! |---|---|---|
//! | `loop_filter_sb[0][0]` (`D02708`) | `dav1d_lpf_h_sb_y_16bpc_ssse3` | `src/x86/loopfilter16_sse.asm:1509-1606`, `src/x86/loopfilter.h:46` |
//! | `loop_filter_sb[0][1]` (`D02714`) | `dav1d_lpf_v_sb_y_16bpc_ssse3` | `src/x86/loopfilter16_sse.asm:1390-1499`, `src/x86/loopfilter.h:47` |
//! | `loop_filter_sb[1][0]` (`D02720`) | `dav1d_lpf_h_sb_uv_16bpc_ssse3` | `src/x86/loopfilter16_sse.asm:1702-1794`, `src/x86/loopfilter.h:48` |
//! | `loop_filter_sb[1][1]` (`D02726`) | `dav1d_lpf_v_sb_uv_16bpc_ssse3` | `src/x86/loopfilter16_sse.asm:1609-1699`, `src/x86/loopfilter.h:49` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bütün SIMD işlevleri yalnız `#[target_feature(enable = "ssse3")]` taşır;
//!   SSE4.1+, VEX ve EVEX komutu üretilmez (`0` SSE4.1+/VEX/EVEX).
//! - Kopyasız O(1) sözleşme: girişlerde [`lpf_sb_sozlesmesi`] denetlenir;
//!   sözleşme dışı girdide [`lpf_sb_skaler_devret`] ile skaler gövdeye devredilir.
//! - Sınır-güvenli erişim: yalnız etkin kenarların (`mask != 0 && level != 0`)
//!   gerçekten ihtiyaç duyduğu komşu pikseller (`wd == 4` için 4 piksel,
//!   `wd == 6` için 6 piksel, `wd == 8` için 8 piksel, `wd == 16` için 14 piksel)
//!   okunur ve yazılır.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use core::arch::x86_64::{
    __m128i, _mm_abs_epi16, _mm_add_epi16, _mm_and_si128, _mm_andnot_si128, _mm_cmpgt_epi16,
    _mm_cvtsi32_si128, _mm_cvtsi64_si128, _mm_max_epi16, _mm_min_epi16, _mm_or_si128,
    _mm_set1_epi16, _mm_setzero_si128, _mm_srai_epi16, _mm_srli_epi16, _mm_sub_epi16,
    _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64, _mm_unpacklo_epi16,
    _mm_unpacklo_epi32, _mm_unpacklo_epi64,
};

use crate::cpu::Ssse3Token;
use crate::dsp::lf::{LimLut, LoopFilterSbFn, lpf_sb_skaler_devret, lpf_sb_sozlesmesi};
use crate::dsp::x86::mem;

/// `loop_filter_sb[plane][dir]` 16-bpc (`BPC ∈ {10, 12}`) SSSE3 yuva kurucusu.
#[must_use]
pub const fn loop_filter_sb_fn<const BPC: u32>(
    _token: Ssse3Token,
    plane: usize,
    dir: usize,
) -> Option<LoopFilterSbFn> {
    const { assert!(BPC == 10 || BPC == 12) };
    match (plane, dir) {
        (0, 0) => Some(lpf_h_sb_y_16bpc_ssse3::<BPC>),
        (0, 1) => Some(lpf_v_sb_y_16bpc_ssse3::<BPC>),
        (1, 0) => Some(lpf_h_sb_uv_16bpc_ssse3::<BPC>),
        (1, 1) => Some(lpf_v_sb_uv_16bpc_ssse3::<BPC>),
        _ => None,
    }
}

#[inline(always)]
fn level_or(level: &[[u8; 4]], at: usize, fallback: usize, idx: usize) -> usize {
    let l = level[at][idx];
    usize::from(if l != 0 { l } else { level[fallback][idx] })
}

#[target_feature(enable = "ssse3")]
#[inline]
fn load_u8x8(src: &[u8; 8]) -> __m128i {
    _mm_cvtsi64_si128(i64::from_le_bytes(*src))
}

#[target_feature(enable = "ssse3")]
#[inline]
fn load_u8x4(src: &[u8; 4]) -> __m128i {
    _mm_cvtsi32_si128(i32::from_le_bytes(*src))
}

#[target_feature(enable = "ssse3")]
#[inline]
fn blend_ssse3(if_false: __m128i, if_true: __m128i, mask: __m128i) -> __m128i {
    _mm_or_si128(
        _mm_and_si128(if_true, mask),
        _mm_andnot_si128(mask, if_false),
    )
}

#[target_feature(enable = "ssse3")]
#[inline]
fn transpose_4x4_epi16(
    r0: __m128i,
    r1: __m128i,
    r2: __m128i,
    r3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let t0 = _mm_unpacklo_epi16(r0, r1);
    let t1 = _mm_unpacklo_epi16(r2, r3);
    let c01 = _mm_unpacklo_epi32(t0, t1);
    let c23 = _mm_unpackhi_epi32(t0, t1);
    let c0 = c01;
    let c1 = _mm_unpackhi_epi64(c01, c01);
    let c2 = c23;
    let c3 = _mm_unpackhi_epi64(c23, c23);
    (c0, c1, c2, c3)
}

#[target_feature(enable = "ssse3")]
#[inline]
fn transpose_4x8_to_8x4_epi16(r0: __m128i, r1: __m128i, r2: __m128i, r3: __m128i) -> [__m128i; 8] {
    let t0 = _mm_unpacklo_epi16(r0, r1);
    let t1 = _mm_unpacklo_epi16(r2, r3);
    let t2 = _mm_unpackhi_epi16(r0, r1);
    let t3 = _mm_unpackhi_epi16(r2, r3);
    let q01 = _mm_unpacklo_epi32(t0, t1);
    let q23 = _mm_unpackhi_epi32(t0, t1);
    let q45 = _mm_unpacklo_epi32(t2, t3);
    let q67 = _mm_unpackhi_epi32(t2, t3);
    [
        q01,
        _mm_unpackhi_epi64(q01, q01),
        q23,
        _mm_unpackhi_epi64(q23, q23),
        q45,
        _mm_unpackhi_epi64(q45, q45),
        q67,
        _mm_unpackhi_epi64(q67, q67),
    ]
}

/// 4 şeritli (`alt 4 × i16`) 4-tap dar filtre + `[0, max_px]` kırpma.
#[target_feature(enable = "ssse3")]
#[inline]
fn filter_narrow4_ssse3<const BPC: u32>(
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    fm: __m128i,
    hev: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let shift = BPC - 8;
    let max_px = _mm_set1_epi16(((1i32 << BPC) - 1) as i16);
    let zero = _mm_setzero_si128();
    let clip_min = _mm_set1_epi16((-128i32 << shift) as i16);
    let clip_max = _mm_set1_epi16(((128i32 << shift) - 1) as i16);
    let c4 = _mm_set1_epi16(4);
    let c3 = _mm_set1_epi16(3);
    let c1 = _mm_set1_epi16(1);

    let p1_m_q1 = _mm_sub_epi16(p1, q1);
    let fv_hev = _mm_and_si128(
        _mm_min_epi16(_mm_max_epi16(p1_m_q1, clip_min), clip_max),
        hev,
    );
    let q0_m_p0 = _mm_sub_epi16(q0, p0);
    let q0_m_p0_x3 = _mm_add_epi16(_mm_add_epi16(q0_m_p0, q0_m_p0), q0_m_p0);
    let fv = _mm_min_epi16(
        _mm_max_epi16(_mm_add_epi16(q0_m_p0_x3, fv_hev), clip_min),
        clip_max,
    );
    let f1 = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(fv, c4), clip_max));
    let f2 = _mm_srai_epi16::<3>(_mm_min_epi16(_mm_add_epi16(fv, c3), clip_max));

    let np0 = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(p0, f2), zero), max_px);
    let nq0 = _mm_min_epi16(_mm_max_epi16(_mm_sub_epi16(q0, f1), zero), max_px);

    let f_outer = _mm_andnot_si128(hev, _mm_srai_epi16::<1>(_mm_add_epi16(f1, c1)));
    let np1 = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(p1, f_outer), zero), max_px);
    let nq1 = _mm_min_epi16(_mm_max_epi16(_mm_sub_epi16(q1, f_outer), zero), max_px);

    (
        blend_ssse3(p1, np1, fm),
        blend_ssse3(p0, np0, fm),
        blend_ssse3(q0, nq0, fm),
        blend_ssse3(q1, nq1, fm),
    )
}

#[target_feature(enable = "ssse3")]
#[inline]
fn filter_wd4_core_ssse3<const BPC: u32>(
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    e: i16,
    i: i16,
    h: i16,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let ve = _mm_set1_epi16(e);
    let vi = _mm_set1_epi16(i);
    let vh = _mm_set1_epi16(h);

    let d_p1p0 = _mm_abs_epi16(_mm_sub_epi16(p1, p0));
    let d_q1q0 = _mm_abs_epi16(_mm_sub_epi16(q1, q0));
    let d_p0q0 = _mm_abs_epi16(_mm_sub_epi16(p0, q0));
    let d_p1q1 = _mm_abs_epi16(_mm_sub_epi16(p1, q1));

    let max_pq10 = _mm_max_epi16(d_p1p0, d_q1q0);
    let hev = _mm_cmpgt_epi16(max_pq10, vh);
    let edge_val = _mm_add_epi16(_mm_add_epi16(d_p0q0, d_p0q0), _mm_srli_epi16::<1>(d_p1q1));
    let not_fm = _mm_or_si128(_mm_cmpgt_epi16(max_pq10, vi), _mm_cmpgt_epi16(edge_val, ve));
    let fm = _mm_andnot_si128(not_fm, _mm_set1_epi16(-1));

    filter_narrow4_ssse3::<BPC>(p1, p0, q0, q1, fm, hev)
}

#[target_feature(enable = "ssse3")]
#[inline]
fn filter_wd6_core_ssse3<const BPC: u32>(
    p2: __m128i,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    q2: __m128i,
    e: i16,
    i: i16,
    h: i16,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let shift = BPC - 8;
    let vf = _mm_set1_epi16((1i32 << shift) as i16);
    let ve = _mm_set1_epi16(e);
    let vi = _mm_set1_epi16(i);
    let vh = _mm_set1_epi16(h);
    let c4 = _mm_set1_epi16(4);

    let d_p1p0 = _mm_abs_epi16(_mm_sub_epi16(p1, p0));
    let d_q1q0 = _mm_abs_epi16(_mm_sub_epi16(q1, q0));
    let d_p2p1 = _mm_abs_epi16(_mm_sub_epi16(p2, p1));
    let d_q2q1 = _mm_abs_epi16(_mm_sub_epi16(q2, q1));
    let d_p0q0 = _mm_abs_epi16(_mm_sub_epi16(p0, q0));
    let d_p1q1 = _mm_abs_epi16(_mm_sub_epi16(p1, q1));

    let max_pq10 = _mm_max_epi16(d_p1p0, d_q1q0);
    let hev = _mm_cmpgt_epi16(max_pq10, vh);
    let max_step = _mm_max_epi16(max_pq10, _mm_max_epi16(d_p2p1, d_q2q1));
    let edge_val = _mm_add_epi16(_mm_add_epi16(d_p0q0, d_p0q0), _mm_srli_epi16::<1>(d_p1q1));
    let not_fm = _mm_or_si128(_mm_cmpgt_epi16(max_step, vi), _mm_cmpgt_epi16(edge_val, ve));
    let fm = _mm_andnot_si128(not_fm, _mm_set1_epi16(-1));

    let d_p2p0 = _mm_abs_epi16(_mm_sub_epi16(p2, p0));
    let d_q2q0 = _mm_abs_epi16(_mm_sub_epi16(q2, q0));
    let max_flat = _mm_max_epi16(max_pq10, _mm_max_epi16(d_p2p0, d_q2q0));
    let flat8in = _mm_andnot_si128(_mm_cmpgt_epi16(max_flat, vf), fm);
    let narrow_mask = _mm_andnot_si128(flat8in, fm);

    let (np1, np0, nq0, nq1) = filter_narrow4_ssse3::<BPC>(p1, p0, q0, q1, narrow_mask, hev);

    // 6-tap flat:
    // out_p1 = (p2*3 + p1*2 + p0*2 + q0 + 4) >> 3
    // out_p0 = (p2 + p1*2 + p0*2 + q0*2 + q1 + 4) >> 3
    // out_q0 = (p1 + p0*2 + q0*2 + q1*2 + q2 + 4) >> 3
    // out_q1 = (p0 + q0*2 + q1*2 + q2*3 + 4) >> 3
    let p1p0_x2 = _mm_add_epi16(_mm_add_epi16(p1, p0), _mm_add_epi16(p1, p0));
    let q0q1_x2 = _mm_add_epi16(_mm_add_epi16(q0, q1), _mm_add_epi16(q0, q1));
    let p2_x3 = _mm_add_epi16(_mm_add_epi16(p2, p2), p2);
    let q2_x3 = _mm_add_epi16(_mm_add_epi16(q2, q2), q2);

    let fp1 = _mm_srli_epi16::<3>(_mm_add_epi16(
        _mm_add_epi16(p2_x3, p1p0_x2),
        _mm_add_epi16(q0, c4),
    ));
    let fp0 = _mm_srli_epi16::<3>(_mm_add_epi16(
        _mm_add_epi16(p2, p1p0_x2),
        _mm_add_epi16(_mm_add_epi16(q0, q0), _mm_add_epi16(q1, c4)),
    ));
    let fq0 = _mm_srli_epi16::<3>(_mm_add_epi16(
        _mm_add_epi16(p1, q0q1_x2),
        _mm_add_epi16(_mm_add_epi16(p0, p0), _mm_add_epi16(q2, c4)),
    ));
    let fq1 = _mm_srli_epi16::<3>(_mm_add_epi16(
        _mm_add_epi16(p0, q0q1_x2),
        _mm_add_epi16(q2_x3, c4),
    ));

    (
        blend_ssse3(np1, fp1, flat8in),
        blend_ssse3(np0, fp0, flat8in),
        blend_ssse3(nq0, fq0, flat8in),
        blend_ssse3(nq1, fq1, flat8in),
    )
}

#[target_feature(enable = "ssse3")]
#[inline]
fn filter_wd8_core_ssse3<const BPC: u32>(
    p3: __m128i,
    p2: __m128i,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    q2: __m128i,
    q3: __m128i,
    e: i16,
    i: i16,
    h: i16,
) -> (__m128i, __m128i, __m128i, __m128i, __m128i, __m128i) {
    let shift = BPC - 8;
    let vf = _mm_set1_epi16((1i32 << shift) as i16);
    let ve = _mm_set1_epi16(e);
    let vi = _mm_set1_epi16(i);
    let vh = _mm_set1_epi16(h);
    let c4 = _mm_set1_epi16(4);

    let d_p1p0 = _mm_abs_epi16(_mm_sub_epi16(p1, p0));
    let d_q1q0 = _mm_abs_epi16(_mm_sub_epi16(q1, q0));
    let d_p2p1 = _mm_abs_epi16(_mm_sub_epi16(p2, p1));
    let d_q2q1 = _mm_abs_epi16(_mm_sub_epi16(q2, q1));
    let d_p3p2 = _mm_abs_epi16(_mm_sub_epi16(p3, p2));
    let d_q3q2 = _mm_abs_epi16(_mm_sub_epi16(q3, q2));
    let d_p0q0 = _mm_abs_epi16(_mm_sub_epi16(p0, q0));
    let d_p1q1 = _mm_abs_epi16(_mm_sub_epi16(p1, q1));

    let max_pq10 = _mm_max_epi16(d_p1p0, d_q1q0);
    let hev = _mm_cmpgt_epi16(max_pq10, vh);
    let max_step = _mm_max_epi16(
        max_pq10,
        _mm_max_epi16(_mm_max_epi16(d_p2p1, d_q2q1), _mm_max_epi16(d_p3p2, d_q3q2)),
    );
    let edge_val = _mm_add_epi16(_mm_add_epi16(d_p0q0, d_p0q0), _mm_srli_epi16::<1>(d_p1q1));
    let not_fm = _mm_or_si128(_mm_cmpgt_epi16(max_step, vi), _mm_cmpgt_epi16(edge_val, ve));
    let fm = _mm_andnot_si128(not_fm, _mm_set1_epi16(-1));

    let d_p2p0 = _mm_abs_epi16(_mm_sub_epi16(p2, p0));
    let d_q2q0 = _mm_abs_epi16(_mm_sub_epi16(q2, q0));
    let d_p3p0 = _mm_abs_epi16(_mm_sub_epi16(p3, p0));
    let d_q3q0 = _mm_abs_epi16(_mm_sub_epi16(q3, q0));
    let max_flat = _mm_max_epi16(
        max_pq10,
        _mm_max_epi16(_mm_max_epi16(d_p2p0, d_q2q0), _mm_max_epi16(d_p3p0, d_q3q0)),
    );
    let flat8in = _mm_andnot_si128(_mm_cmpgt_epi16(max_flat, vf), fm);
    let narrow_mask = _mm_andnot_si128(flat8in, fm);

    let (np1, np0, nq0, nq1) = filter_narrow4_ssse3::<BPC>(p1, p0, q0, q1, narrow_mask, hev);

    // 8-tap flat:
    let sum_mid = _mm_add_epi16(_mm_add_epi16(p2, p1), _mm_add_epi16(p0, q0));
    let p3_x2 = _mm_add_epi16(p3, p3);
    let q3_x2 = _mm_add_epi16(q3, q3);

    let fp2 = _mm_srli_epi16::<3>(_mm_add_epi16(
        _mm_add_epi16(p3_x2, p3),
        _mm_add_epi16(_mm_add_epi16(p2, sum_mid), c4),
    ));
    let fp1 = _mm_srli_epi16::<3>(_mm_add_epi16(
        _mm_add_epi16(p3_x2, p1),
        _mm_add_epi16(_mm_add_epi16(sum_mid, q1), c4),
    ));
    let fp0 = _mm_srli_epi16::<3>(_mm_add_epi16(
        _mm_add_epi16(p3, p0),
        _mm_add_epi16(_mm_add_epi16(sum_mid, _mm_add_epi16(q1, q2)), c4),
    ));
    let fq0 = _mm_srli_epi16::<3>(_mm_add_epi16(
        _mm_add_epi16(sum_mid, q0),
        _mm_add_epi16(_mm_add_epi16(q1, _mm_add_epi16(q2, q3)), c4),
    ));
    let fq1 = _mm_srli_epi16::<3>(_mm_add_epi16(
        _mm_add_epi16(_mm_add_epi16(p1, p0), _mm_add_epi16(q0, q2)),
        _mm_add_epi16(_mm_add_epi16(q1, q1), _mm_add_epi16(q3_x2, c4)),
    ));
    let fq2 = _mm_srli_epi16::<3>(_mm_add_epi16(
        _mm_add_epi16(_mm_add_epi16(p0, q0), _mm_add_epi16(q1, c4)),
        _mm_add_epi16(_mm_add_epi16(q2, q2), _mm_add_epi16(q3_x2, q3)),
    ));

    (
        blend_ssse3(p2, fp2, flat8in),
        blend_ssse3(np1, fp1, flat8in),
        blend_ssse3(np0, fp0, flat8in),
        blend_ssse3(nq0, fq0, flat8in),
        blend_ssse3(nq1, fq1, flat8in),
        blend_ssse3(q2, fq2, flat8in),
    )
}

#[target_feature(enable = "ssse3")]
#[inline]
fn filter_wd16_core_ssse3<const BPC: u32>(
    p: &[__m128i; 14],
    e: i16,
    i: i16,
    h: i16,
) -> [__m128i; 12] {
    let [p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6] = *p;
    let shift = BPC - 8;
    let vf = _mm_set1_epi16((1i32 << shift) as i16);
    let c8 = _mm_set1_epi16(8);

    let (w8_p2, w8_p1, w8_p0, w8_q0, w8_q1, w8_q2) =
        filter_wd8_core_ssse3::<BPC>(p3, p2, p1, p0, q0, q1, q2, q3, e, i, h);

    // fm ve flat8in maskelerini yeniden hesapla + flat8out:
    let ve = _mm_set1_epi16(e);
    let vi = _mm_set1_epi16(i);
    let d_p1p0 = _mm_abs_epi16(_mm_sub_epi16(p1, p0));
    let d_q1q0 = _mm_abs_epi16(_mm_sub_epi16(q1, q0));
    let d_p2p1 = _mm_abs_epi16(_mm_sub_epi16(p2, p1));
    let d_q2q1 = _mm_abs_epi16(_mm_sub_epi16(q2, q1));
    let d_p3p2 = _mm_abs_epi16(_mm_sub_epi16(p3, p2));
    let d_q3q2 = _mm_abs_epi16(_mm_sub_epi16(q3, q2));
    let d_p0q0 = _mm_abs_epi16(_mm_sub_epi16(p0, q0));
    let d_p1q1 = _mm_abs_epi16(_mm_sub_epi16(p1, q1));

    let max_pq10 = _mm_max_epi16(d_p1p0, d_q1q0);
    let max_step = _mm_max_epi16(
        max_pq10,
        _mm_max_epi16(_mm_max_epi16(d_p2p1, d_q2q1), _mm_max_epi16(d_p3p2, d_q3q2)),
    );
    let edge_val = _mm_add_epi16(_mm_add_epi16(d_p0q0, d_p0q0), _mm_srli_epi16::<1>(d_p1q1));
    let not_fm = _mm_or_si128(_mm_cmpgt_epi16(max_step, vi), _mm_cmpgt_epi16(edge_val, ve));
    let fm = _mm_andnot_si128(not_fm, _mm_set1_epi16(-1));

    let d_p2p0 = _mm_abs_epi16(_mm_sub_epi16(p2, p0));
    let d_q2q0 = _mm_abs_epi16(_mm_sub_epi16(q2, q0));
    let d_p3p0 = _mm_abs_epi16(_mm_sub_epi16(p3, p0));
    let d_q3q0 = _mm_abs_epi16(_mm_sub_epi16(q3, q0));
    let d_p4p0 = _mm_abs_epi16(_mm_sub_epi16(p4, p0));
    let d_q4q0 = _mm_abs_epi16(_mm_sub_epi16(q4, q0));
    let d_p5p0 = _mm_abs_epi16(_mm_sub_epi16(p5, p0));
    let d_q5q0 = _mm_abs_epi16(_mm_sub_epi16(q5, q0));
    let d_p6p0 = _mm_abs_epi16(_mm_sub_epi16(p6, p0));
    let d_q6q0 = _mm_abs_epi16(_mm_sub_epi16(q6, q0));

    let max_flat16 = _mm_max_epi16(
        _mm_max_epi16(
            max_pq10,
            _mm_max_epi16(_mm_max_epi16(d_p2p0, d_q2q0), _mm_max_epi16(d_p3p0, d_q3q0)),
        ),
        _mm_max_epi16(
            _mm_max_epi16(d_p4p0, d_q4q0),
            _mm_max_epi16(_mm_max_epi16(d_p5p0, d_q5q0), _mm_max_epi16(d_p6p0, d_q6q0)),
        ),
    );
    let flat16 = _mm_andnot_si128(_mm_cmpgt_epi16(max_flat16, vf), fm);

    // 16-tap toplamları (u16 mod 65536'da 65528 <= 65535 taşmasız, `_mm_srli_epi16::<4>` işaretsiz):
    // S0 = p6*7 + p5*2 + p4*2 + p3 + p2 + p1 + p0 + q0 + 8
    let p6_x2 = _mm_add_epi16(p6, p6);
    let p6_x4 = _mm_add_epi16(p6_x2, p6_x2);
    let p6_x7 = _mm_sub_epi16(_mm_add_epi16(p6_x4, p6_x4), p6);
    let p54_x2 = _mm_add_epi16(_mm_add_epi16(p5, p4), _mm_add_epi16(p5, p4));
    let mut s = _mm_add_epi16(
        _mm_add_epi16(_mm_add_epi16(p6_x7, p54_x2), _mm_add_epi16(p3, p2)),
        _mm_add_epi16(_mm_add_epi16(p1, p0), _mm_add_epi16(q0, c8)),
    );
    let f16_p5 = _mm_srli_epi16::<4>(s);

    // p4: + p3 + q1 - p6*2
    s = _mm_sub_epi16(_mm_add_epi16(s, _mm_add_epi16(p3, q1)), p6_x2);
    let f16_p4 = _mm_srli_epi16::<4>(s);

    // p3: + p2 + q2 - p6 - p5
    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(p2, q2)),
        _mm_add_epi16(p6, p5),
    );
    let f16_p3 = _mm_srli_epi16::<4>(s);

    // p2: + p1 + q3 - p6 - p4
    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(p1, q3)),
        _mm_add_epi16(p6, p4),
    );
    let f16_p2 = _mm_srli_epi16::<4>(s);

    // p1: + p0 + q4 - p6 - p3
    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(p0, q4)),
        _mm_add_epi16(p6, p3),
    );
    let f16_p1 = _mm_srli_epi16::<4>(s);

    // p0: + q0 + q5 - p6 - p2
    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(q0, q5)),
        _mm_add_epi16(p6, p2),
    );
    let f16_p0 = _mm_srli_epi16::<4>(s);

    // q0: + q1 + q6 - p6 - p1
    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(q1, q6)),
        _mm_add_epi16(p6, p1),
    );
    let f16_q0 = _mm_srli_epi16::<4>(s);

    // q1: + q2 + q6 - p5 - p0
    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(q2, q6)),
        _mm_add_epi16(p5, p0),
    );
    let f16_q1 = _mm_srli_epi16::<4>(s);

    // q2: + q3 + q6 - p4 - q0
    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(q3, q6)),
        _mm_add_epi16(p4, q0),
    );
    let f16_q2 = _mm_srli_epi16::<4>(s);

    // q3: + q4 + q6 - p3 - q1
    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(q4, q6)),
        _mm_add_epi16(p3, q1),
    );
    let f16_q3 = _mm_srli_epi16::<4>(s);

    // q4: + q5 + q6 - p2 - q2
    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(q5, q6)),
        _mm_add_epi16(p2, q2),
    );
    let f16_q4 = _mm_srli_epi16::<4>(s);

    // q5: + q6*2 - p1 - q3
    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(q6, q6)),
        _mm_add_epi16(p1, q3),
    );
    let f16_q5 = _mm_srli_epi16::<4>(s);

    [
        blend_ssse3(p5, f16_p5, flat16),
        blend_ssse3(p4, f16_p4, flat16),
        blend_ssse3(p3, f16_p3, flat16),
        blend_ssse3(w8_p2, f16_p2, flat16),
        blend_ssse3(w8_p1, f16_p1, flat16),
        blend_ssse3(w8_p0, f16_p0, flat16),
        blend_ssse3(w8_q0, f16_q0, flat16),
        blend_ssse3(w8_q1, f16_q1, flat16),
        blend_ssse3(w8_q2, f16_q2, flat16),
        blend_ssse3(q3, f16_q3, flat16),
        blend_ssse3(q4, f16_q4, flat16),
        blend_ssse3(q5, f16_q5, flat16),
    ]
}

#[target_feature(enable = "ssse3")]
#[inline]
fn edge_h_ssse3<const BPC: u32>(
    dst: &mut [u8],
    off: usize,
    stride: usize,
    wd: usize,
    e: i16,
    i: i16,
    h: i16,
) {
    match wd {
        4 => {
            let r0 = load_u8x8(mem::chunk::<u8, 8>(dst, off - 4));
            let r1 = load_u8x8(mem::chunk::<u8, 8>(dst, off + stride - 4));
            let r2 = load_u8x8(mem::chunk::<u8, 8>(dst, off + 2 * stride - 4));
            let r3 = load_u8x8(mem::chunk::<u8, 8>(dst, off + 3 * stride - 4));
            let (p1, p0, q0, q1) = transpose_4x4_epi16(r0, r1, r2, r3);
            let (op1, op0, oq0, oq1) = filter_wd4_core_ssse3::<BPC>(p1, p0, q0, q1, e, i, h);
            let (w0, w1, w2, w3) = transpose_4x4_epi16(op1, op0, oq0, oq1);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off - 4), w0);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + stride - 4), w1);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + 2 * stride - 4), w2);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + 3 * stride - 4), w3);
        }
        6 => {
            // Yalnız [p2..q2] (6 piksel = 12 bayt) oku: 8 bayt + 4 bayt
            let load6 = |d: &[u8], row_off: usize| -> __m128i {
                let lo = load_u8x8(mem::chunk::<u8, 8>(d, row_off));
                let hi = load_u8x4(mem::chunk::<u8, 4>(d, row_off + 8));
                _mm_unpacklo_epi64(lo, hi)
            };
            let r0 = load6(dst, off - 6);
            let r1 = load6(dst, off + stride - 6);
            let r2 = load6(dst, off + 2 * stride - 6);
            let r3 = load6(dst, off + 3 * stride - 6);
            let cols = transpose_4x8_to_8x4_epi16(r0, r1, r2, r3);
            let (op1, op0, oq0, oq1) = filter_wd6_core_ssse3::<BPC>(
                cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], e, i, h,
            );
            let (w0, w1, w2, w3) = transpose_4x4_epi16(op1, op0, oq0, oq1);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off - 4), w0);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + stride - 4), w1);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + 2 * stride - 4), w2);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + 3 * stride - 4), w3);
        }
        8 => {
            let r0 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off - 8));
            let r1 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off + stride - 8));
            let r2 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off + 2 * stride - 8));
            let r3 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off + 3 * stride - 8));
            let cols = transpose_4x8_to_8x4_epi16(r0, r1, r2, r3);
            let (op2, op1, op0, oq0, oq1, oq2) = filter_wd8_core_ssse3::<BPC>(
                cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], e, i, h,
            );
            // Yalnız [p2..q2] (6 piksel = 12 bayt) yaz: [p2, p1, p0, q0] (8 bayt) + [q1, q2] (4 bayt)
            let (w0_lo, w1_lo, w2_lo, w3_lo) = transpose_4x4_epi16(op2, op1, op0, oq0);
            let z = _mm_setzero_si128();
            let (w0_hi, w1_hi, w2_hi, w3_hi) = transpose_4x4_epi16(oq1, oq2, z, z);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off - 6), w0_lo);
            mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, off + 2), w0_hi);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + stride - 6), w1_lo);
            mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, off + stride + 2), w1_hi);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + 2 * stride - 6), w2_lo);
            mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, off + 2 * stride + 2), w2_hi);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + 3 * stride - 6), w3_lo);
            mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, off + 3 * stride + 2), w3_hi);
        }
        _ => {
            // wd == 16: yalnız [p6..q6] (14 piksel = 28 bayt) oku:
            // sol 8 piksel [p6..q0] (off - 14 .. off + 2) + sağ 6 piksel [q1..q6] (off + 2 .. off + 14)
            let load6 = |d: &[u8], row_off: usize| -> __m128i {
                let lo = load_u8x8(mem::chunk::<u8, 8>(d, row_off));
                let hi = load_u8x4(mem::chunk::<u8, 4>(d, row_off + 8));
                _mm_unpacklo_epi64(lo, hi)
            };
            let r0_l = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off - 14));
            let r1_l = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off + stride - 14));
            let r2_l = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off + 2 * stride - 14));
            let r3_l = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off + 3 * stride - 14));
            let cols_l = transpose_4x8_to_8x4_epi16(r0_l, r1_l, r2_l, r3_l);

            let r0_r = load6(dst, off + 2);
            let r1_r = load6(dst, off + stride + 2);
            let r2_r = load6(dst, off + 2 * stride + 2);
            let r3_r = load6(dst, off + 3 * stride + 2);
            let cols_r = transpose_4x8_to_8x4_epi16(r0_r, r1_r, r2_r, r3_r);

            let p = [
                cols_l[0], cols_l[1], cols_l[2], cols_l[3], cols_l[4], cols_l[5], cols_l[6],
                cols_l[7], cols_r[0], cols_r[1], cols_r[2], cols_r[3], cols_r[4], cols_r[5],
            ];
            let out = filter_wd16_core_ssse3::<BPC>(&p, e, i, h);
            // Yalnız [p5..q5] (12 piksel = 24 bayt, off - 12 .. off + 12) yaz:
            let (w0_a, w1_a, w2_a, w3_a) = transpose_4x4_epi16(out[0], out[1], out[2], out[3]);
            let (w0_b, w1_b, w2_b, w3_b) = transpose_4x4_epi16(out[4], out[5], out[6], out[7]);
            let (w0_c, w1_c, w2_c, w3_c) = transpose_4x4_epi16(out[8], out[9], out[10], out[11]);

            for (row_idx, (wa, wb, wc)) in [
                (0usize, (w0_a, w0_b, w0_c)),
                (1, (w1_a, w1_b, w1_c)),
                (2, (w2_a, w2_b, w2_c)),
                (3, (w3_a, w3_b, w3_c)),
            ] {
                let roff = off + row_idx * stride;
                mem::store_u8x16(
                    mem::chunk_mut::<u8, 16>(dst, roff - 12),
                    _mm_unpacklo_epi64(wa, wb),
                );
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, roff + 4), wc);
            }
        }
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn edge_v_ssse3<const BPC: u32>(
    dst: &mut [u8],
    off: usize,
    stride: usize,
    wd: usize,
    e: i16,
    i: i16,
    h: i16,
) {
    match wd {
        4 => {
            let p1 = load_u8x8(mem::chunk::<u8, 8>(dst, off - 2 * stride));
            let p0 = load_u8x8(mem::chunk::<u8, 8>(dst, off - stride));
            let q0 = load_u8x8(mem::chunk::<u8, 8>(dst, off));
            let q1 = load_u8x8(mem::chunk::<u8, 8>(dst, off + stride));
            let (op1, op0, oq0, oq1) = filter_wd4_core_ssse3::<BPC>(p1, p0, q0, q1, e, i, h);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off - 2 * stride), op1);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off - stride), op0);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off), oq0);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + stride), oq1);
        }
        6 => {
            let p2 = load_u8x8(mem::chunk::<u8, 8>(dst, off - 3 * stride));
            let p1 = load_u8x8(mem::chunk::<u8, 8>(dst, off - 2 * stride));
            let p0 = load_u8x8(mem::chunk::<u8, 8>(dst, off - stride));
            let q0 = load_u8x8(mem::chunk::<u8, 8>(dst, off));
            let q1 = load_u8x8(mem::chunk::<u8, 8>(dst, off + stride));
            let q2 = load_u8x8(mem::chunk::<u8, 8>(dst, off + 2 * stride));
            let (op1, op0, oq0, oq1) =
                filter_wd6_core_ssse3::<BPC>(p2, p1, p0, q0, q1, q2, e, i, h);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off - 2 * stride), op1);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off - stride), op0);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off), oq0);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + stride), oq1);
        }
        8 => {
            let p3 = load_u8x8(mem::chunk::<u8, 8>(dst, off - 4 * stride));
            let p2 = load_u8x8(mem::chunk::<u8, 8>(dst, off - 3 * stride));
            let p1 = load_u8x8(mem::chunk::<u8, 8>(dst, off - 2 * stride));
            let p0 = load_u8x8(mem::chunk::<u8, 8>(dst, off - stride));
            let q0 = load_u8x8(mem::chunk::<u8, 8>(dst, off));
            let q1 = load_u8x8(mem::chunk::<u8, 8>(dst, off + stride));
            let q2 = load_u8x8(mem::chunk::<u8, 8>(dst, off + 2 * stride));
            let q3 = load_u8x8(mem::chunk::<u8, 8>(dst, off + 3 * stride));
            let (op2, op1, op0, oq0, oq1, oq2) =
                filter_wd8_core_ssse3::<BPC>(p3, p2, p1, p0, q0, q1, q2, q3, e, i, h);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off - 3 * stride), op2);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off - 2 * stride), op1);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off - stride), op0);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off), oq0);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + stride), oq1);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + 2 * stride), oq2);
        }
        _ => {
            let p = [
                load_u8x8(mem::chunk::<u8, 8>(dst, off - 7 * stride)),
                load_u8x8(mem::chunk::<u8, 8>(dst, off - 6 * stride)),
                load_u8x8(mem::chunk::<u8, 8>(dst, off - 5 * stride)),
                load_u8x8(mem::chunk::<u8, 8>(dst, off - 4 * stride)),
                load_u8x8(mem::chunk::<u8, 8>(dst, off - 3 * stride)),
                load_u8x8(mem::chunk::<u8, 8>(dst, off - 2 * stride)),
                load_u8x8(mem::chunk::<u8, 8>(dst, off - stride)),
                load_u8x8(mem::chunk::<u8, 8>(dst, off)),
                load_u8x8(mem::chunk::<u8, 8>(dst, off + stride)),
                load_u8x8(mem::chunk::<u8, 8>(dst, off + 2 * stride)),
                load_u8x8(mem::chunk::<u8, 8>(dst, off + 3 * stride)),
                load_u8x8(mem::chunk::<u8, 8>(dst, off + 4 * stride)),
                load_u8x8(mem::chunk::<u8, 8>(dst, off + 5 * stride)),
                load_u8x8(mem::chunk::<u8, 8>(dst, off + 6 * stride)),
            ];
            let out = filter_wd16_core_ssse3::<BPC>(&p, e, i, h);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off - 6 * stride), out[0]);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off - 5 * stride), out[1]);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off - 4 * stride), out[2]);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off - 3 * stride), out[3]);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off - 2 * stride), out[4]);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off - stride), out[5]);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off), out[6]);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + stride), out[7]);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + 2 * stride), out[8]);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + 3 * stride), out[9]);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + 4 * stride), out[10]);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + 5 * stride), out[11]);
        }
    }
}

#[target_feature(enable = "ssse3")]
fn lpf_h_sb_16bpc_ssse3_giris<const BPC: u32, const PLANE: usize>(
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
    let shift = BPC - 8;
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
            if l == 0 || l >= 64 {
                continue;
            }
            let wd = if PLANE == 0 {
                if v > 3 {
                    continue;
                }
                4usize << (v - 1)
            } else {
                if v > 2 {
                    continue;
                }
                4 + 2 * usize::from(v - 1)
            };
            if gx == 1 && wd > 8 {
                continue;
            }
            let off = gy * 4 * stride + gx * 8;
            let e = (lut.e[l] << shift) as i16;
            let i = (lut.i[l] << shift) as i16;
            let h = (((l >> 4) as i32) << shift) as i16;
            edge_h_ssse3::<BPC>(dst, off, stride, wd, e, i, h);
        }
    }
}

#[target_feature(enable = "ssse3")]
fn lpf_v_sb_16bpc_ssse3_giris<const BPC: u32, const PLANE: usize>(
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
    let shift = BPC - 8;
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
            if l == 0 || l >= 64 {
                continue;
            }
            let wd = if PLANE == 0 {
                if v > 3 {
                    continue;
                }
                4usize << (v - 1)
            } else {
                if v > 2 {
                    continue;
                }
                4 + 2 * usize::from(v - 1)
            };
            if gy == 1 && wd > 8 {
                continue;
            }
            let off = gy * 4 * stride + gx * 8;
            let e = (lut.e[l] << shift) as i16;
            let i = (lut.i[l] << shift) as i16;
            let h = (((l >> 4) as i32) << shift) as i16;
            edge_v_ssse3::<BPC>(dst, off, stride, wd, e, i, h);
        }
    }
}

fn lpf_h_sb_y_16bpc_ssse3<const BPC: u32>(
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
        0,
        0,
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
        lpf_sb_skaler_devret::<BPC>(
            0,
            0,
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
    // SAFETY: (1) ISA: bu işlev yalnız `Ssse3Token` ile kurulur (`loop_filter_sb_fn`).
    // (2) sınır: `lpf_sb_sozlesmesi` `dst`, `mask` ve `level` dilim sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: tüm bellek erişimleri `mem::chunk` ve `mem::chunk_mut` üzerinden hizasızdır; `&mut` tek erişim yoludur.
    unsafe {
        lpf_h_sb_16bpc_ssse3_giris::<BPC, 0>(
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
        )
    }
}

fn lpf_v_sb_y_16bpc_ssse3<const BPC: u32>(
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
        0,
        1,
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
        lpf_sb_skaler_devret::<BPC>(
            0,
            1,
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
    // SAFETY: (1) ISA: bu işlev yalnız `Ssse3Token` ile kurulur (`loop_filter_sb_fn`).
    // (2) sınır: `lpf_sb_sozlesmesi` `dst`, `mask` ve `level` dilim sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: tüm bellek erişimleri `mem::chunk` ve `mem::chunk_mut` üzerinden hizasızdır; `&mut` tek erişim yoludur.
    unsafe {
        lpf_v_sb_16bpc_ssse3_giris::<BPC, 0>(
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
        )
    }
}

fn lpf_h_sb_uv_16bpc_ssse3<const BPC: u32>(
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
        1,
        0,
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
        lpf_sb_skaler_devret::<BPC>(
            1,
            0,
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
    // SAFETY: (1) ISA: bu işlev yalnız `Ssse3Token` ile kurulur (`loop_filter_sb_fn`).
    // (2) sınır: `lpf_sb_sozlesmesi` `dst`, `mask` ve `level` dilim sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: tüm bellek erişimleri `mem::chunk` ve `mem::chunk_mut` üzerinden hizasızdır; `&mut` tek erişim yoludur.
    unsafe {
        lpf_h_sb_16bpc_ssse3_giris::<BPC, 1>(
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
        )
    }
}

fn lpf_v_sb_uv_16bpc_ssse3<const BPC: u32>(
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
        1,
        1,
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
        lpf_sb_skaler_devret::<BPC>(
            1,
            1,
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
    // SAFETY: (1) ISA: bu işlev yalnız `Ssse3Token` ile kurulur (`loop_filter_sb_fn`).
    // (2) sınır: `lpf_sb_sozlesmesi` `dst`, `mask` ve `level` dilim sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: tüm bellek erişimleri `mem::chunk` ve `mem::chunk_mut` üzerinden hizasızdır; `&mut` tek erişim yoludur.
    unsafe {
        lpf_v_sb_16bpc_ssse3_giris::<BPC, 1>(
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
        )
    }
}
