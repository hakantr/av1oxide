//! AVX-512 (Ice Lake) 16-bpc (`10/12` bit) loop filter (deblocking) çekirdekleri (`4` kayıt / `8` yuva; P46 Adım 2).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Yuva | Sembol | Kaynak |
//! |---|---|---|
//! | `loop_filter_sb[0][0]` (`D02706`) | `dav1d_lpf_h_sb_y_16bpc_avx512icl` | `src/x86/loopfilter16_avx512.asm:792-840`, `src/x86/loopfilter.h:65` |
//! | `loop_filter_sb[0][1]` (`D02712`) | `dav1d_lpf_v_sb_y_16bpc_avx512icl` | `src/x86/loopfilter16_avx512.asm:745-790`, `src/x86/loopfilter.h:61` |
//! | `loop_filter_sb[1][0]` (`D02718`) | `dav1d_lpf_h_sb_uv_16bpc_avx512icl` | `src/x86/loopfilter16_avx512.asm:876-910`, `src/x86/loopfilter.h:66` |
//! | `loop_filter_sb[1][1]` (`D02724`) | `dav1d_lpf_v_sb_uv_16bpc_avx512icl` | `src/x86/loopfilter16_avx512.asm:842-874`, `src/x86/loopfilter.h:62` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bütün SIMD işlevleri `#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]`
//!   taşır; `vpgather*`/`vpscatter*` üretilmez (`0` gather/scatter).
//! - Kopyasız O(1) sözleşme: girişlerde [`lpf_sb_sozlesmesi`] denetlenir;
//!   sözleşme dışı girdide [`lpf_sb_skaler_devret`] ile skaler gövdeye devredilir.
//! - Sınır-güvenli erişim: yalnız etkin kenarların (`mask != 0 && level != 0`)
//!   gerçekten ihtiyaç duyduğu komşu pikseller okunur ve yazılır.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_abs_epi16, _mm_add_epi16, _mm_and_si128, _mm_andnot_si128,
    _mm_blendv_epi8, _mm_cmpgt_epi16, _mm_cvtsi32_si128, _mm_cvtsi64_si128, _mm_max_epi16,
    _mm_min_epi16, _mm_or_si128, _mm_set1_epi16, _mm_setr_epi16, _mm_setzero_si128, _mm_srai_epi16,
    _mm_srli_epi16, _mm_sub_epi16, _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpackhi_epi64,
    _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm256_abs_epi16, _mm256_add_epi16,
    _mm256_and_si256, _mm256_andnot_si256, _mm256_blendv_epi8, _mm256_castsi256_si128,
    _mm256_cmpgt_epi16, _mm256_extracti128_si256, _mm256_inserti128_si256, _mm256_max_epi16,
    _mm256_min_epi16, _mm256_or_si256, _mm256_set1_epi16, _mm256_setr_epi16, _mm256_setr_m128i,
    _mm256_setzero_si256, _mm256_srai_epi16, _mm256_srli_epi16, _mm256_sub_epi16,
    _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi16, _mm256_unpacklo_epi32,
};

use crate::cpu::Avx512IclToken;
use crate::dsp::lf::{LimLut, LoopFilterSbFn, lpf_sb_skaler_devret, lpf_sb_sozlesmesi};
use crate::dsp::x86::mem;

/// `loop_filter_sb[plane][dir]` 16-bpc (`BPC ∈ {10, 12}`) AVX512ICL yuva kurucusu.
#[must_use]
pub const fn loop_filter_sb_fn<const BPC: u32>(
    _token: Avx512IclToken,
    plane: usize,
    dir: usize,
) -> Option<LoopFilterSbFn> {
    const { assert!(BPC == 10 || BPC == 12) };
    match (plane, dir) {
        (0, 0) => Some(lpf_h_sb_y_16bpc_avx512::<BPC>),
        (0, 1) => Some(lpf_v_sb_y_16bpc_avx512::<BPC>),
        (1, 0) => Some(lpf_h_sb_uv_16bpc_avx512::<BPC>),
        (1, 1) => Some(lpf_v_sb_uv_16bpc_avx512::<BPC>),
        _ => None,
    }
}

#[inline(always)]
fn level_or(level: &[[u8; 4]], at: usize, fallback: usize, idx: usize) -> usize {
    let l = level[at][idx];
    usize::from(if l != 0 { l } else { level[fallback][idx] })
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn load_u8x8(src: &[u8; 8]) -> __m128i {
    _mm_cvtsi64_si128(i64::from_le_bytes(*src))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn load_u8x4(src: &[u8; 4]) -> __m128i {
    _mm_cvtsi32_si128(i32::from_le_bytes(*src))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
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
    (
        c01,
        _mm_unpackhi_epi64(c01, c01),
        c23,
        _mm_unpackhi_epi64(c23, c23),
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn filter_narrow4_128_avx512<const BPC: u32>(
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
        _mm_blendv_epi8(p1, np1, fm),
        _mm_blendv_epi8(p0, np0, fm),
        _mm_blendv_epi8(q0, nq0, fm),
        _mm_blendv_epi8(q1, nq1, fm),
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn filter_wd4_128_avx512<const BPC: u32>(
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    ve: __m128i,
    vi: __m128i,
    vh: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let d_p1p0 = _mm_abs_epi16(_mm_sub_epi16(p1, p0));
    let d_q1q0 = _mm_abs_epi16(_mm_sub_epi16(q1, q0));
    let d_p0q0 = _mm_abs_epi16(_mm_sub_epi16(p0, q0));
    let d_p1q1 = _mm_abs_epi16(_mm_sub_epi16(p1, q1));

    let max_pq10 = _mm_max_epi16(d_p1p0, d_q1q0);
    let hev = _mm_cmpgt_epi16(max_pq10, vh);
    let edge_val = _mm_add_epi16(_mm_add_epi16(d_p0q0, d_p0q0), _mm_srli_epi16::<1>(d_p1q1));
    let not_fm = _mm_or_si128(_mm_cmpgt_epi16(max_pq10, vi), _mm_cmpgt_epi16(edge_val, ve));
    let fm = _mm_andnot_si128(not_fm, _mm_set1_epi16(-1));

    filter_narrow4_128_avx512::<BPC>(p1, p0, q0, q1, fm, hev)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn filter_wd6_128_avx512<const BPC: u32>(
    p2: __m128i,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    q2: __m128i,
    ve: __m128i,
    vi: __m128i,
    vh: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let shift = BPC - 8;
    let vf = _mm_set1_epi16((1i32 << shift) as i16);
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

    let (np1, np0, nq0, nq1) = filter_narrow4_128_avx512::<BPC>(p1, p0, q0, q1, narrow_mask, hev);

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
        _mm_blendv_epi8(np1, fp1, flat8in),
        _mm_blendv_epi8(np0, fp0, flat8in),
        _mm_blendv_epi8(nq0, fq0, flat8in),
        _mm_blendv_epi8(nq1, fq1, flat8in),
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn filter_wd8_128_avx512<const BPC: u32>(
    p3: __m128i,
    p2: __m128i,
    p1: __m128i,
    p0: __m128i,
    q0: __m128i,
    q1: __m128i,
    q2: __m128i,
    q3: __m128i,
    ve: __m128i,
    vi: __m128i,
    vh: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i, __m128i, __m128i) {
    let shift = BPC - 8;
    let vf = _mm_set1_epi16((1i32 << shift) as i16);
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

    let (np1, np0, nq0, nq1) = filter_narrow4_128_avx512::<BPC>(p1, p0, q0, q1, narrow_mask, hev);

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
        _mm_blendv_epi8(p2, fp2, flat8in),
        _mm_blendv_epi8(np1, fp1, flat8in),
        _mm_blendv_epi8(np0, fp0, flat8in),
        _mm_blendv_epi8(nq0, fq0, flat8in),
        _mm_blendv_epi8(nq1, fq1, flat8in),
        _mm_blendv_epi8(q2, fq2, flat8in),
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn filter_wd16_128_avx512<const BPC: u32>(
    p: &[__m128i; 14],
    ve: __m128i,
    vi: __m128i,
    vh: __m128i,
) -> [__m128i; 12] {
    let [p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6] = *p;
    let shift = BPC - 8;
    let vf = _mm_set1_epi16((1i32 << shift) as i16);
    let c8 = _mm_set1_epi16(8);

    let (w8_p2, w8_p1, w8_p0, w8_q0, w8_q1, w8_q2) =
        filter_wd8_128_avx512::<BPC>(p3, p2, p1, p0, q0, q1, q2, q3, ve, vi, vh);

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

    let p6_x2 = _mm_add_epi16(p6, p6);
    let p6_x4 = _mm_add_epi16(p6_x2, p6_x2);
    let p6_x7 = _mm_sub_epi16(_mm_add_epi16(p6_x4, p6_x4), p6);
    let p54_x2 = _mm_add_epi16(_mm_add_epi16(p5, p4), _mm_add_epi16(p5, p4));
    let mut s = _mm_add_epi16(
        _mm_add_epi16(_mm_add_epi16(p6_x7, p54_x2), _mm_add_epi16(p3, p2)),
        _mm_add_epi16(_mm_add_epi16(p1, p0), _mm_add_epi16(q0, c8)),
    );
    let f16_p5 = _mm_srli_epi16::<4>(s);

    s = _mm_sub_epi16(_mm_add_epi16(s, _mm_add_epi16(p3, q1)), p6_x2);
    let f16_p4 = _mm_srli_epi16::<4>(s);

    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(p2, q2)),
        _mm_add_epi16(p6, p5),
    );
    let f16_p3 = _mm_srli_epi16::<4>(s);

    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(p1, q3)),
        _mm_add_epi16(p6, p4),
    );
    let f16_p2 = _mm_srli_epi16::<4>(s);

    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(p0, q4)),
        _mm_add_epi16(p6, p3),
    );
    let f16_p1 = _mm_srli_epi16::<4>(s);

    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(q0, q5)),
        _mm_add_epi16(p6, p2),
    );
    let f16_p0 = _mm_srli_epi16::<4>(s);

    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(q1, q6)),
        _mm_add_epi16(p6, p1),
    );
    let f16_q0 = _mm_srli_epi16::<4>(s);

    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(q2, q6)),
        _mm_add_epi16(p5, p0),
    );
    let f16_q1 = _mm_srli_epi16::<4>(s);

    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(q3, q6)),
        _mm_add_epi16(p4, q0),
    );
    let f16_q2 = _mm_srli_epi16::<4>(s);

    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(q4, q6)),
        _mm_add_epi16(p3, q1),
    );
    let f16_q3 = _mm_srli_epi16::<4>(s);

    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(q5, q6)),
        _mm_add_epi16(p2, q2),
    );
    let f16_q4 = _mm_srli_epi16::<4>(s);

    s = _mm_sub_epi16(
        _mm_add_epi16(s, _mm_add_epi16(q6, q6)),
        _mm_add_epi16(p1, q3),
    );
    let f16_q5 = _mm_srli_epi16::<4>(s);

    [
        _mm_blendv_epi8(p5, f16_p5, flat16),
        _mm_blendv_epi8(p4, f16_p4, flat16),
        _mm_blendv_epi8(p3, f16_p3, flat16),
        _mm_blendv_epi8(w8_p2, f16_p2, flat16),
        _mm_blendv_epi8(w8_p1, f16_p1, flat16),
        _mm_blendv_epi8(w8_p0, f16_p0, flat16),
        _mm_blendv_epi8(w8_q0, f16_q0, flat16),
        _mm_blendv_epi8(w8_q1, f16_q1, flat16),
        _mm_blendv_epi8(w8_q2, f16_q2, flat16),
        _mm_blendv_epi8(q3, f16_q3, flat16),
        _mm_blendv_epi8(q4, f16_q4, flat16),
        _mm_blendv_epi8(q5, f16_q5, flat16),
    ]
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn filter_narrow4_256_avx512<const BPC: u32>(
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    fm: __m256i,
    hev: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let shift = BPC - 8;
    let max_px = _mm256_set1_epi16(((1i32 << BPC) - 1) as i16);
    let zero = _mm256_setzero_si256();
    let clip_min = _mm256_set1_epi16((-128i32 << shift) as i16);
    let clip_max = _mm256_set1_epi16(((128i32 << shift) - 1) as i16);
    let c4 = _mm256_set1_epi16(4);
    let c3 = _mm256_set1_epi16(3);
    let c1 = _mm256_set1_epi16(1);

    let p1_m_q1 = _mm256_sub_epi16(p1, q1);
    let fv_hev = _mm256_and_si256(
        _mm256_min_epi16(_mm256_max_epi16(p1_m_q1, clip_min), clip_max),
        hev,
    );
    let q0_m_p0 = _mm256_sub_epi16(q0, p0);
    let q0_m_p0_x3 = _mm256_add_epi16(_mm256_add_epi16(q0_m_p0, q0_m_p0), q0_m_p0);
    let fv = _mm256_min_epi16(
        _mm256_max_epi16(_mm256_add_epi16(q0_m_p0_x3, fv_hev), clip_min),
        clip_max,
    );
    let f1 = _mm256_srai_epi16::<3>(_mm256_min_epi16(_mm256_add_epi16(fv, c4), clip_max));
    let f2 = _mm256_srai_epi16::<3>(_mm256_min_epi16(_mm256_add_epi16(fv, c3), clip_max));

    let np0 = _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(p0, f2), zero), max_px);
    let nq0 = _mm256_min_epi16(_mm256_max_epi16(_mm256_sub_epi16(q0, f1), zero), max_px);

    let f_outer = _mm256_andnot_si256(hev, _mm256_srai_epi16::<1>(_mm256_add_epi16(f1, c1)));
    let np1 = _mm256_min_epi16(
        _mm256_max_epi16(_mm256_add_epi16(p1, f_outer), zero),
        max_px,
    );
    let nq1 = _mm256_min_epi16(
        _mm256_max_epi16(_mm256_sub_epi16(q1, f_outer), zero),
        max_px,
    );

    (
        _mm256_blendv_epi8(p1, np1, fm),
        _mm256_blendv_epi8(p0, np0, fm),
        _mm256_blendv_epi8(q0, nq0, fm),
        _mm256_blendv_epi8(q1, nq1, fm),
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn filter_wd4_256_avx512<const BPC: u32>(
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    ve: __m256i,
    vi: __m256i,
    vh: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let d_p1p0 = _mm256_abs_epi16(_mm256_sub_epi16(p1, p0));
    let d_q1q0 = _mm256_abs_epi16(_mm256_sub_epi16(q1, q0));
    let d_p0q0 = _mm256_abs_epi16(_mm256_sub_epi16(p0, q0));
    let d_p1q1 = _mm256_abs_epi16(_mm256_sub_epi16(p1, q1));

    let max_pq10 = _mm256_max_epi16(d_p1p0, d_q1q0);
    let hev = _mm256_cmpgt_epi16(max_pq10, vh);
    let edge_val = _mm256_add_epi16(
        _mm256_add_epi16(d_p0q0, d_p0q0),
        _mm256_srli_epi16::<1>(d_p1q1),
    );
    let not_fm = _mm256_or_si256(
        _mm256_cmpgt_epi16(max_pq10, vi),
        _mm256_cmpgt_epi16(edge_val, ve),
    );
    let fm = _mm256_andnot_si256(not_fm, _mm256_set1_epi16(-1));

    filter_narrow4_256_avx512::<BPC>(p1, p0, q0, q1, fm, hev)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn edge_h_avx512<const BPC: u32>(
    dst: &mut [u8],
    off: usize,
    stride: usize,
    wd: usize,
    e: i16,
    i: i16,
    h: i16,
) {
    let ve = _mm_set1_epi16(e);
    let vi = _mm_set1_epi16(i);
    let vh = _mm_set1_epi16(h);
    match wd {
        4 => {
            let r0 = load_u8x8(mem::chunk::<u8, 8>(dst, off - 4));
            let r1 = load_u8x8(mem::chunk::<u8, 8>(dst, off + stride - 4));
            let r2 = load_u8x8(mem::chunk::<u8, 8>(dst, off + 2 * stride - 4));
            let r3 = load_u8x8(mem::chunk::<u8, 8>(dst, off + 3 * stride - 4));
            let (p1, p0, q0, q1) = transpose_4x4_epi16(r0, r1, r2, r3);
            let (op1, op0, oq0, oq1) = filter_wd4_128_avx512::<BPC>(p1, p0, q0, q1, ve, vi, vh);
            let (w0, w1, w2, w3) = transpose_4x4_epi16(op1, op0, oq0, oq1);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off - 4), w0);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + stride - 4), w1);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + 2 * stride - 4), w2);
            mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, off + 3 * stride - 4), w3);
        }
        6 => {
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
            let (op1, op0, oq0, oq1) = filter_wd6_128_avx512::<BPC>(
                cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], ve, vi, vh,
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
            let (op2, op1, op0, oq0, oq1, oq2) = filter_wd8_128_avx512::<BPC>(
                cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], cols[6], cols[7], ve, vi, vh,
            );
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
            let out = filter_wd16_128_avx512::<BPC>(&p, ve, vi, vh);
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn edge_h_pair_avx512<const BPC: u32>(
    dst: &mut [u8],
    off0: usize,
    off1: usize,
    stride: usize,
    wd: usize,
    e0: i16,
    i0: i16,
    h0: i16,
    e1: i16,
    i1: i16,
    h1: i16,
) {
    if wd == 4 {
        let r0_0 = load_u8x8(mem::chunk::<u8, 8>(dst, off0 - 4));
        let r1_0 = load_u8x8(mem::chunk::<u8, 8>(dst, off0 + stride - 4));
        let r2_0 = load_u8x8(mem::chunk::<u8, 8>(dst, off0 + 2 * stride - 4));
        let r3_0 = load_u8x8(mem::chunk::<u8, 8>(dst, off0 + 3 * stride - 4));

        let r0_1 = load_u8x8(mem::chunk::<u8, 8>(dst, off1 - 4));
        let r1_1 = load_u8x8(mem::chunk::<u8, 8>(dst, off1 + stride - 4));
        let r2_1 = load_u8x8(mem::chunk::<u8, 8>(dst, off1 + 2 * stride - 4));
        let r3_1 = load_u8x8(mem::chunk::<u8, 8>(dst, off1 + 3 * stride - 4));

        let r0 = _mm256_inserti128_si256::<1>(_mm256_setr_m128i(r0_0, r0_0), r0_1);
        let r1 = _mm256_inserti128_si256::<1>(_mm256_setr_m128i(r1_0, r1_0), r1_1);
        let r2 = _mm256_inserti128_si256::<1>(_mm256_setr_m128i(r2_0, r2_0), r2_1);
        let r3 = _mm256_inserti128_si256::<1>(_mm256_setr_m128i(r3_0, r3_0), r3_1);

        let t0 = _mm256_unpacklo_epi16(r0, r1);
        let t1 = _mm256_unpacklo_epi16(r2, r3);
        let c01 = _mm256_unpacklo_epi32(t0, t1);
        let c23 = _mm256_unpackhi_epi32(t0, t1);
        let p1 = c01;
        let p0 = _mm256_unpackhi_epi64(c01, c01);
        let q0 = c23;
        let q1 = _mm256_unpackhi_epi64(c23, c23);

        let ve = _mm256_setr_m128i(_mm_set1_epi16(e0), _mm_set1_epi16(e1));
        let vi = _mm256_setr_m128i(_mm_set1_epi16(i0), _mm_set1_epi16(i1));
        let vh = _mm256_setr_m128i(_mm_set1_epi16(h0), _mm_set1_epi16(h1));

        let (op1, op0, oq0, oq1) = filter_wd4_256_avx512::<BPC>(p1, p0, q0, q1, ve, vi, vh);

        let u0 = _mm256_unpacklo_epi16(op1, op0);
        let u1 = _mm256_unpacklo_epi16(oq0, oq1);
        let w01 = _mm256_unpacklo_epi32(u0, u1);
        let w23 = _mm256_unpackhi_epi32(u0, u1);
        let w0 = w01;
        let w1 = _mm256_unpackhi_epi64(w01, w01);
        let w2 = w23;
        let w3 = _mm256_unpackhi_epi64(w23, w23);

        mem::store_u8x8(
            mem::chunk_mut::<u8, 8>(dst, off0 - 4),
            _mm256_castsi256_si128(w0),
        );
        mem::store_u8x8(
            mem::chunk_mut::<u8, 8>(dst, off0 + stride - 4),
            _mm256_castsi256_si128(w1),
        );
        mem::store_u8x8(
            mem::chunk_mut::<u8, 8>(dst, off0 + 2 * stride - 4),
            _mm256_castsi256_si128(w2),
        );
        mem::store_u8x8(
            mem::chunk_mut::<u8, 8>(dst, off0 + 3 * stride - 4),
            _mm256_castsi256_si128(w3),
        );

        mem::store_u8x8(
            mem::chunk_mut::<u8, 8>(dst, off1 - 4),
            _mm256_extracti128_si256::<1>(w0),
        );
        mem::store_u8x8(
            mem::chunk_mut::<u8, 8>(dst, off1 + stride - 4),
            _mm256_extracti128_si256::<1>(w1),
        );
        mem::store_u8x8(
            mem::chunk_mut::<u8, 8>(dst, off1 + 2 * stride - 4),
            _mm256_extracti128_si256::<1>(w2),
        );
        mem::store_u8x8(
            mem::chunk_mut::<u8, 8>(dst, off1 + 3 * stride - 4),
            _mm256_extracti128_si256::<1>(w3),
        );
    } else {
        edge_h_avx512::<BPC>(dst, off0, stride, wd, e0, i0, h0);
        edge_h_avx512::<BPC>(dst, off1, stride, wd, e1, i1, h1);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn edge_v_avx512<const BPC: u32>(
    dst: &mut [u8],
    off: usize,
    stride: usize,
    wd: usize,
    e: i16,
    i: i16,
    h: i16,
) {
    let ve = _mm_set1_epi16(e);
    let vi = _mm_set1_epi16(i);
    let vh = _mm_set1_epi16(h);
    match wd {
        4 => {
            let p1 = load_u8x8(mem::chunk::<u8, 8>(dst, off - 2 * stride));
            let p0 = load_u8x8(mem::chunk::<u8, 8>(dst, off - stride));
            let q0 = load_u8x8(mem::chunk::<u8, 8>(dst, off));
            let q1 = load_u8x8(mem::chunk::<u8, 8>(dst, off + stride));
            let (op1, op0, oq0, oq1) = filter_wd4_128_avx512::<BPC>(p1, p0, q0, q1, ve, vi, vh);
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
                filter_wd6_128_avx512::<BPC>(p2, p1, p0, q0, q1, q2, ve, vi, vh);
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
                filter_wd8_128_avx512::<BPC>(p3, p2, p1, p0, q0, q1, q2, q3, ve, vi, vh);
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
            let out = filter_wd16_128_avx512::<BPC>(&p, ve, vi, vh);
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

/// 4 ardışık yatay kenar bloğunu (`gx..gx+4`, 16 piksel = 32 bayt) `wd == 4` ile 256-bit YMM'de birlikte işler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn edge_v_quad_wd4_avx512<const BPC: u32>(
    dst: &mut [u8],
    off: usize,
    stride: usize,
    e: [i16; 4],
    i: [i16; 4],
    h: [i16; 4],
) {
    let ve = _mm256_setr_epi16(
        e[0], e[0], e[0], e[0], e[1], e[1], e[1], e[1], e[2], e[2], e[2], e[2], e[3], e[3], e[3],
        e[3],
    );
    let vi = _mm256_setr_epi16(
        i[0], i[0], i[0], i[0], i[1], i[1], i[1], i[1], i[2], i[2], i[2], i[2], i[3], i[3], i[3],
        i[3],
    );
    let vh = _mm256_setr_epi16(
        h[0], h[0], h[0], h[0], h[1], h[1], h[1], h[1], h[2], h[2], h[2], h[2], h[3], h[3], h[3],
        h[3],
    );
    let p1 = mem::load_u8x32(mem::chunk::<u8, 32>(dst, off - 2 * stride));
    let p0 = mem::load_u8x32(mem::chunk::<u8, 32>(dst, off - stride));
    let q0 = mem::load_u8x32(mem::chunk::<u8, 32>(dst, off));
    let q1 = mem::load_u8x32(mem::chunk::<u8, 32>(dst, off + stride));
    let (op1, op0, oq0, oq1) = filter_wd4_256_avx512::<BPC>(p1, p0, q0, q1, ve, vi, vh);
    mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, off - 2 * stride), op1);
    mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, off - stride), op0);
    mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, off), oq0);
    mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, off + stride), oq1);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn edge_v_pair_avx512<const BPC: u32>(
    dst: &mut [u8],
    off: usize,
    stride: usize,
    wd: usize,
    e0: i16,
    i0: i16,
    h0: i16,
    e1: i16,
    i1: i16,
    h1: i16,
) {
    let ve = _mm_setr_epi16(e0, e0, e0, e0, e1, e1, e1, e1);
    let vi = _mm_setr_epi16(i0, i0, i0, i0, i1, i1, i1, i1);
    let vh = _mm_setr_epi16(h0, h0, h0, h0, h1, h1, h1, h1);
    match wd {
        4 => {
            let p1 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off - 2 * stride));
            let p0 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off - stride));
            let q0 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off));
            let q1 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off + stride));
            let (op1, op0, oq0, oq1) = filter_wd4_128_avx512::<BPC>(p1, p0, q0, q1, ve, vi, vh);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off - 2 * stride), op1);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off - stride), op0);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off), oq0);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off + stride), oq1);
        }
        6 => {
            let p2 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off - 3 * stride));
            let p1 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off - 2 * stride));
            let p0 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off - stride));
            let q0 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off));
            let q1 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off + stride));
            let q2 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off + 2 * stride));
            let (op1, op0, oq0, oq1) =
                filter_wd6_128_avx512::<BPC>(p2, p1, p0, q0, q1, q2, ve, vi, vh);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off - 2 * stride), op1);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off - stride), op0);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off), oq0);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off + stride), oq1);
        }
        8 => {
            let p3 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off - 4 * stride));
            let p2 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off - 3 * stride));
            let p1 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off - 2 * stride));
            let p0 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off - stride));
            let q0 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off));
            let q1 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off + stride));
            let q2 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off + 2 * stride));
            let q3 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, off + 3 * stride));
            let (op2, op1, op0, oq0, oq1, oq2) =
                filter_wd8_128_avx512::<BPC>(p3, p2, p1, p0, q0, q1, q2, q3, ve, vi, vh);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off - 3 * stride), op2);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off - 2 * stride), op1);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off - stride), op0);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off), oq0);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off + stride), oq1);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off + 2 * stride), oq2);
        }
        _ => {
            let p = [
                mem::load_u8x16(mem::chunk::<u8, 16>(dst, off - 7 * stride)),
                mem::load_u8x16(mem::chunk::<u8, 16>(dst, off - 6 * stride)),
                mem::load_u8x16(mem::chunk::<u8, 16>(dst, off - 5 * stride)),
                mem::load_u8x16(mem::chunk::<u8, 16>(dst, off - 4 * stride)),
                mem::load_u8x16(mem::chunk::<u8, 16>(dst, off - 3 * stride)),
                mem::load_u8x16(mem::chunk::<u8, 16>(dst, off - 2 * stride)),
                mem::load_u8x16(mem::chunk::<u8, 16>(dst, off - stride)),
                mem::load_u8x16(mem::chunk::<u8, 16>(dst, off)),
                mem::load_u8x16(mem::chunk::<u8, 16>(dst, off + stride)),
                mem::load_u8x16(mem::chunk::<u8, 16>(dst, off + 2 * stride)),
                mem::load_u8x16(mem::chunk::<u8, 16>(dst, off + 3 * stride)),
                mem::load_u8x16(mem::chunk::<u8, 16>(dst, off + 4 * stride)),
                mem::load_u8x16(mem::chunk::<u8, 16>(dst, off + 5 * stride)),
                mem::load_u8x16(mem::chunk::<u8, 16>(dst, off + 6 * stride)),
            ];
            let out = filter_wd16_128_avx512::<BPC>(&p, ve, vi, vh);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off - 6 * stride), out[0]);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off - 5 * stride), out[1]);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off - 4 * stride), out[2]);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off - 3 * stride), out[3]);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off - 2 * stride), out[4]);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off - stride), out[5]);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off), out[6]);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off + stride), out[7]);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off + 2 * stride), out[8]);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off + 3 * stride), out[9]);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off + 4 * stride), out[10]);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, off + 5 * stride), out[11]);
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
fn lpf_h_sb_16bpc_avx512_giris<const BPC: u32, const PLANE: usize>(
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
    let y_end = y0 + count;
    for gx in 1..w4 {
        let mut gy = y0;
        while gy < y_end {
            let v0 = mask[gy * mask_stride + gx];
            if v0 == 0 {
                gy += 1;
                continue;
            }
            let l0 = level_or(
                level,
                gy * lvl_stride + gx,
                gy * lvl_stride + gx - 1,
                lvl_idx,
            );
            if l0 == 0 || l0 >= 64 {
                gy += 1;
                continue;
            }
            let wd0 = if PLANE == 0 {
                if v0 > 3 {
                    gy += 1;
                    continue;
                }
                4usize << (v0 - 1)
            } else {
                if v0 > 2 {
                    gy += 1;
                    continue;
                }
                4 + 2 * usize::from(v0 - 1)
            };
            if gx == 1 && wd0 > 8 {
                gy += 1;
                continue;
            }
            let off0 = gy * 4 * stride + gx * 8;
            let e0 = (lut.e[l0] << shift) as i16;
            let i0 = (lut.i[l0] << shift) as i16;
            let h0 = (((l0 >> 4) as i32) << shift) as i16;

            if wd0 == 4 && gy + 1 < y_end && mask[(gy + 1) * mask_stride + gx] == v0 {
                let l1 = level_or(
                    level,
                    (gy + 1) * lvl_stride + gx,
                    (gy + 1) * lvl_stride + gx - 1,
                    lvl_idx,
                );
                if l1 > 0 && l1 < 64 {
                    let off1 = (gy + 1) * 4 * stride + gx * 8;
                    let e1 = (lut.e[l1] << shift) as i16;
                    let i1 = (lut.i[l1] << shift) as i16;
                    let h1 = (((l1 >> 4) as i32) << shift) as i16;
                    edge_h_pair_avx512::<BPC>(dst, off0, off1, stride, 4, e0, i0, h0, e1, i1, h1);
                    gy += 2;
                    continue;
                }
            }

            edge_h_avx512::<BPC>(dst, off0, stride, wd0, e0, i0, h0);
            gy += 1;
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
fn lpf_v_sb_16bpc_avx512_giris<const BPC: u32, const PLANE: usize>(
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
        let mut gx = 0usize;
        while gx < w4 {
            let v0 = mask[gy * mask_stride + gx];
            if v0 == 0 {
                gx += 1;
                continue;
            }
            let l0 = level_or(
                level,
                gy * lvl_stride + gx,
                (gy - 1) * lvl_stride + gx,
                lvl_idx,
            );
            if l0 == 0 || l0 >= 64 {
                gx += 1;
                continue;
            }
            let wd0 = if PLANE == 0 {
                if v0 > 3 {
                    gx += 1;
                    continue;
                }
                4usize << (v0 - 1)
            } else {
                if v0 > 2 {
                    gx += 1;
                    continue;
                }
                4 + 2 * usize::from(v0 - 1)
            };
            if gy == 1 && wd0 > 8 {
                gx += 1;
                continue;
            }
            let off0 = gy * 4 * stride + gx * 8;
            let e0 = (lut.e[l0] << shift) as i16;
            let i0 = (lut.i[l0] << shift) as i16;
            let h0 = (((l0 >> 4) as i32) << shift) as i16;

            if wd0 == 4
                && gx + 3 < w4
                && mask[gy * mask_stride + gx + 1] == v0
                && mask[gy * mask_stride + gx + 2] == v0
                && mask[gy * mask_stride + gx + 3] == v0
            {
                let l1 = level_or(
                    level,
                    gy * lvl_stride + gx + 1,
                    (gy - 1) * lvl_stride + gx + 1,
                    lvl_idx,
                );
                let l2 = level_or(
                    level,
                    gy * lvl_stride + gx + 2,
                    (gy - 1) * lvl_stride + gx + 2,
                    lvl_idx,
                );
                let l3 = level_or(
                    level,
                    gy * lvl_stride + gx + 3,
                    (gy - 1) * lvl_stride + gx + 3,
                    lvl_idx,
                );
                if l1 > 0 && l1 < 64 && l2 > 0 && l2 < 64 && l3 > 0 && l3 < 64 {
                    let e = [
                        e0,
                        (lut.e[l1] << shift) as i16,
                        (lut.e[l2] << shift) as i16,
                        (lut.e[l3] << shift) as i16,
                    ];
                    let i = [
                        i0,
                        (lut.i[l1] << shift) as i16,
                        (lut.i[l2] << shift) as i16,
                        (lut.i[l3] << shift) as i16,
                    ];
                    let h = [
                        h0,
                        (((l1 >> 4) as i32) << shift) as i16,
                        (((l2 >> 4) as i32) << shift) as i16,
                        (((l3 >> 4) as i32) << shift) as i16,
                    ];
                    edge_v_quad_wd4_avx512::<BPC>(dst, off0, stride, e, i, h);
                    gx += 4;
                    continue;
                }
            }

            if gx + 1 < w4 && mask[gy * mask_stride + gx + 1] == v0 {
                let l1 = level_or(
                    level,
                    gy * lvl_stride + gx + 1,
                    (gy - 1) * lvl_stride + gx + 1,
                    lvl_idx,
                );
                if l1 > 0 && l1 < 64 {
                    let e1 = (lut.e[l1] << shift) as i16;
                    let i1 = (lut.i[l1] << shift) as i16;
                    let h1 = (((l1 >> 4) as i32) << shift) as i16;
                    edge_v_pair_avx512::<BPC>(dst, off0, stride, wd0, e0, i0, h0, e1, i1, h1);
                    gx += 2;
                    continue;
                }
            }

            edge_v_avx512::<BPC>(dst, off0, stride, wd0, e0, i0, h0);
            gx += 1;
        }
    }
}

fn lpf_h_sb_y_16bpc_avx512<const BPC: u32>(
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
    // SAFETY: (1) ISA: bu işlev yalnız `Avx512IclToken` ile kurulur (`loop_filter_sb_fn`).
    // (2) sınır: `lpf_sb_sozlesmesi` `dst`, `mask` ve `level` dilim sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: tüm bellek erişimleri `mem::chunk` ve `mem::chunk_mut` üzerinden hizasızdır; `&mut` tek erişim yoludur.
    unsafe {
        lpf_h_sb_16bpc_avx512_giris::<BPC, 0>(
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

fn lpf_v_sb_y_16bpc_avx512<const BPC: u32>(
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
    // SAFETY: (1) ISA: bu işlev yalnız `Avx512IclToken` ile kurulur (`loop_filter_sb_fn`).
    // (2) sınır: `lpf_sb_sozlesmesi` `dst`, `mask` ve `level` dilim sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: tüm bellek erişimleri `mem::chunk` ve `mem::chunk_mut` üzerinden hizasızdır; `&mut` tek erişim yoludur.
    unsafe {
        lpf_v_sb_16bpc_avx512_giris::<BPC, 0>(
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

fn lpf_h_sb_uv_16bpc_avx512<const BPC: u32>(
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
    // SAFETY: (1) ISA: bu işlev yalnız `Avx512IclToken` ile kurulur (`loop_filter_sb_fn`).
    // (2) sınır: `lpf_sb_sozlesmesi` `dst`, `mask` ve `level` dilim sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: tüm bellek erişimleri `mem::chunk` ve `mem::chunk_mut` üzerinden hizasızdır; `&mut` tek erişim yoludur.
    unsafe {
        lpf_h_sb_16bpc_avx512_giris::<BPC, 1>(
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

fn lpf_v_sb_uv_16bpc_avx512<const BPC: u32>(
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
    // SAFETY: (1) ISA: bu işlev yalnız `Avx512IclToken` ile kurulur (`loop_filter_sb_fn`).
    // (2) sınır: `lpf_sb_sozlesmesi` `dst`, `mask` ve `level` dilim sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: tüm bellek erişimleri `mem::chunk` ve `mem::chunk_mut` üzerinden hizasızdır; `&mut` tek erişim yoludur.
    unsafe {
        lpf_v_sb_16bpc_avx512_giris::<BPC, 1>(
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
