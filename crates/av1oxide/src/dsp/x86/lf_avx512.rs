//! 8 bpc loop filter (deblocking) x86_64 AVX512ICL çekirdekleri (`D02705`,
//! `D02711`, `D02717`, `D02723`; P46 Adım 2 / Paket A).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Kayıt | Yuva | Sembol | Kaynak |
//! |---|---|---|---|
//! | `D02705` | `loop_filter_sb[0][0]` | `dav1d_lpf_h_sb_y_8bpc_avx512icl` | `src/x86/loopfilter_avx512.asm:1391-1440`, `src/x86/loopfilter.h:65` |
//! | `D02711` | `loop_filter_sb[0][1]` | `dav1d_lpf_v_sb_y_8bpc_avx512icl` | `src/x86/loopfilter_avx512.asm:1339-1389`, `src/x86/loopfilter.h:61` |
//! | `D02717` | `loop_filter_sb[1][0]` | `dav1d_lpf_h_sb_uv_8bpc_avx512icl` | `src/x86/loopfilter_avx512.asm:1482-1522`, `src/x86/loopfilter.h:66` |
//! | `D02723` | `loop_filter_sb[1][1]` | `dav1d_lpf_v_sb_uv_8bpc_avx512icl` | `src/x86/loopfilter_avx512.asm:1442-1480`, `src/x86/loopfilter.h:62` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bu modülde tüm SIMD işlevleri
//!   `#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]` taşır;
//!   `vpgather*` / `vpscatter*` komutu üretilmez (`0` gather/scatter).
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
    __m128i, __m256i, _mm_cvtsi64_si128, _mm_cvtsi128_si32, _mm_extract_epi32, _mm_setr_epi8,
    _mm_setr_epi32, _mm_setzero_si128, _mm_shuffle_epi8, _mm_unpackhi_epi32, _mm_unpackhi_epi64,
    _mm_unpacklo_epi32, _mm_unpacklo_epi64, _mm256_abs_epi16, _mm256_add_epi16,
    _mm256_cmpeq_epi16_mask, _mm256_cmpgt_epi16_mask, _mm256_cvtepu8_epi16, _mm256_cvtusepi16_epi8,
    _mm256_mask_mov_epi16, _mm256_maskz_mov_epi16, _mm256_max_epi16, _mm256_min_epi16,
    _mm256_mullo_epi16, _mm256_set1_epi16, _mm256_set1_epi64x, _mm256_setr_epi8,
    _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_slli_epi16, _mm256_srai_epi16,
    _mm256_srli_epi16, _mm256_sub_epi16,
};

use crate::cpu::Avx512IclToken;
use crate::dsp::lf::{LimLut, LoopFilterSbFn, lpf_sb_skaler_devret, lpf_sb_sozlesmesi};
use crate::dsp::x86::mem;

/// 8 bpc `loop_filter_sb[plane][dir]` AVX512ICL kurucusu (`D02705`, `D02711`,
/// `D02717`, `D02723`).
#[must_use]
pub const fn loop_filter_sb_fn(
    _token: Avx512IclToken,
    plane: usize,
    dir: usize,
) -> Option<LoopFilterSbFn> {
    match (plane, dir) {
        (0, 0) => Some(lpf_h_sb_y_8bpc_avx512icl),
        (0, 1) => Some(lpf_v_sb_y_8bpc_avx512icl),
        (1, 0) => Some(lpf_h_sb_uv_8bpc_avx512icl),
        (1, 1) => Some(lpf_v_sb_uv_8bpc_avx512icl),
        _ => None,
    }
}

fn lpf_h_sb_y_8bpc_avx512icl(
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
    lpf_sb_8bpc_avx512::<0, 0>(
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

fn lpf_v_sb_y_8bpc_avx512icl(
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
    lpf_sb_8bpc_avx512::<0, 1>(
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

fn lpf_h_sb_uv_8bpc_avx512icl(
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
    lpf_sb_8bpc_avx512::<1, 0>(
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

fn lpf_v_sb_uv_8bpc_avx512icl(
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
    lpf_sb_8bpc_avx512::<1, 1>(
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
fn lpf_sb_8bpc_avx512<const PLANE: usize, const DIR: usize>(
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`loop_filter_sb_fn`); belirteç AVX512ICL tespiti + maske izninin kanıtıdır.
    // (2) sınır: `lpf_sb_sozlesmesi::<8>` `dst`, `mask` ve `level` dilimlerinin
    // sınırlarını O(1) doğruladı; çekirdek yalnız etkin kenarların kendi `wd`
    // penceresini sınır denetimli dizi yardımcılarıyla okur ve yazar.
    // (3) hizalama/takma ad: tüm vektör erişimleri `mem` içindeki hizasız sabit
    // boyutlu dizi sarmalayıcılarıyla yapılır; `dst` `&mut` tek yazma yoludur.
    unsafe {
        if DIR == 0 {
            lpf_h_sb_avx512_giris::<PLANE>(
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
            lpf_v_sb_avx512_giris::<PLANE>(
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
struct Batch4 {
    off: [usize; 4],
    wd: [u8; 4],
    e_packed: u64,
    i_packed: u64,
    h_packed: u64,
    wd_packed: u64,
}

impl Batch4 {
    #[inline(always)]
    fn reset(&mut self) {
        self.e_packed = 0;
        self.i_packed = 0;
        self.h_packed = 0;
        self.wd_packed = 0;
    }

    #[inline(always)]
    fn push(&mut self, n: usize, off: usize, wd: usize, e: i32, i: i32, h: usize) {
        self.off[n] = off;
        self.wd[n] = wd as u8;
        let sh = n * 16;
        self.e_packed |= ((e as u16) as u64) << sh;
        self.i_packed |= ((i as u16) as u64) << sh;
        self.h_packed |= ((h as u16) as u64) << sh;
        self.wd_packed |= (wd as u64) << sh;
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn broadcast_packed4_avx512(packed: u64) -> __m256i {
    let shuf = _mm256_setr_epi8(
        0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7,
        6, 7,
    );
    _mm256_shuffle_epi8(_mm256_set1_epi64x(packed as i64), shuf)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn le_mask_avx512(a: __m256i, b: __m256i) -> u16 {
    !_mm256_cmpgt_epi16_mask(a, b)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn pack_tap_u8_avx512(v: __m256i) -> __m128i {
    _mm256_cvtusepi16_epi8(_mm256_max_epi16(v, _mm256_setzero_si256()))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn prepare_batch_avx512(batch: &Batch4, n: usize) -> (__m256i, __m256i, __m256i, __m256i, usize) {
    let mut min_wd = batch.wd[0] as usize;
    for b in 1..n {
        let w = batch.wd[b] as usize;
        if w < min_wd {
            min_wd = w;
        }
    }
    let e_v = broadcast_packed4_avx512(batch.e_packed);
    let i_v = broadcast_packed4_avx512(batch.i_packed);
    let h_v = broadcast_packed4_avx512(batch.h_packed);
    let wd_v = broadcast_packed4_avx512(batch.wd_packed);
    (e_v, i_v, h_v, wd_v, min_wd)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn filter_wd4_only_avx512(
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    e_v: __m256i,
    i_v: __m256i,
    h_v: __m256i,
    wd_v: __m256i,
) -> Option<(__m256i, __m256i, __m256i, __m256i)> {
    let abs_p1_p0 = _mm256_abs_epi16(_mm256_sub_epi16(p1, p0));
    let abs_q1_q0 = _mm256_abs_epi16(_mm256_sub_epi16(q1, q0));
    let abs_p0_q0 = _mm256_abs_epi16(_mm256_sub_epi16(p0, q0));
    let abs_p1_q1 = _mm256_abs_epi16(_mm256_sub_epi16(p1, q1));
    let term_e = _mm256_add_epi16(
        _mm256_slli_epi16::<1>(abs_p0_q0),
        _mm256_srli_epi16::<1>(abs_p1_q1),
    );
    let fm = _mm256_cmpgt_epi16_mask(wd_v, _mm256_setzero_si256())
        & le_mask_avx512(abs_p1_p0, i_v)
        & le_mask_avx512(abs_q1_q0, i_v)
        & le_mask_avx512(term_e, e_v);
    if fm == 0 {
        return None;
    }
    let one = _mm256_set1_epi16(1);
    let hev = _mm256_cmpgt_epi16_mask(abs_p1_p0, h_v) | _mm256_cmpgt_epi16_mask(abs_q1_q0, h_v);
    let lo = _mm256_set1_epi16(-128);
    let hi = _mm256_set1_epi16(127);
    let clip = |v: __m256i| _mm256_max_epi16(_mm256_min_epi16(v, hi), lo);
    let fv_hev = _mm256_maskz_mov_epi16(hev, clip(_mm256_sub_epi16(p1, q1)));
    let q0_m_p0_3 = _mm256_mullo_epi16(_mm256_sub_epi16(q0, p0), _mm256_set1_epi16(3));
    let fv = clip(_mm256_add_epi16(q0_m_p0_3, fv_hev));
    let f1 = _mm256_srai_epi16::<3>(_mm256_min_epi16(
        _mm256_add_epi16(fv, _mm256_set1_epi16(4)),
        hi,
    ));
    let f2 = _mm256_srai_epi16::<3>(_mm256_min_epi16(
        _mm256_add_epi16(fv, _mm256_set1_epi16(3)),
        hi,
    ));
    let fv2 = _mm256_maskz_mov_epi16(!hev, _mm256_srai_epi16::<1>(_mm256_add_epi16(f1, one)));
    Some((
        _mm256_mask_mov_epi16(p1, fm, _mm256_add_epi16(p1, fv2)),
        _mm256_mask_mov_epi16(p0, fm, _mm256_add_epi16(p0, f2)),
        _mm256_mask_mov_epi16(q0, fm, _mm256_sub_epi16(q0, f1)),
        _mm256_mask_mov_epi16(q1, fm, _mm256_sub_epi16(q1, fv2)),
    ))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn filter_step1_avx512<const PLANE: usize>(
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    p2: __m256i,
    q2: __m256i,
    p3: __m256i,
    q3: __m256i,
    e_v: __m256i,
    i_v: __m256i,
    wd_v: __m256i,
    max_wd: usize,
) -> (u16, u16, u16, __m256i, __m256i) {
    let abs_p1_p0 = _mm256_abs_epi16(_mm256_sub_epi16(p1, p0));
    let abs_q1_q0 = _mm256_abs_epi16(_mm256_sub_epi16(q1, q0));
    let abs_p0_q0 = _mm256_abs_epi16(_mm256_sub_epi16(p0, q0));
    let abs_p1_q1 = _mm256_abs_epi16(_mm256_sub_epi16(p1, q1));

    let term_e = _mm256_add_epi16(
        _mm256_slli_epi16::<1>(abs_p0_q0),
        _mm256_srli_epi16::<1>(abs_p1_q1),
    );
    let mut fm = _mm256_cmpgt_epi16_mask(wd_v, _mm256_setzero_si256())
        & le_mask_avx512(abs_p1_p0, i_v)
        & le_mask_avx512(abs_q1_q0, i_v)
        & le_mask_avx512(term_e, e_v);

    if max_wd >= 6 {
        let c6 = le_mask_avx512(_mm256_abs_epi16(_mm256_sub_epi16(p2, p1)), i_v)
            & le_mask_avx512(_mm256_abs_epi16(_mm256_sub_epi16(q2, q1)), i_v);
        let gt4 = _mm256_cmpgt_epi16_mask(wd_v, _mm256_set1_epi16(4));
        fm &= !gt4 | c6;
    }

    if PLANE == 0 && max_wd >= 8 {
        let c8 = le_mask_avx512(_mm256_abs_epi16(_mm256_sub_epi16(p3, p2)), i_v)
            & le_mask_avx512(_mm256_abs_epi16(_mm256_sub_epi16(q3, q2)), i_v);
        let gt6 = _mm256_cmpgt_epi16_mask(wd_v, _mm256_set1_epi16(6));
        fm &= !gt6 | c8;
    }

    if fm == 0 {
        return (0, 0, 0, abs_p1_p0, abs_q1_q0);
    }

    let one = _mm256_set1_epi16(1);
    let flat6 = if max_wd >= 6 {
        le_mask_avx512(abs_p1_p0, one)
            & le_mask_avx512(abs_q1_q0, one)
            & le_mask_avx512(_mm256_abs_epi16(_mm256_sub_epi16(p2, p0)), one)
            & le_mask_avx512(_mm256_abs_epi16(_mm256_sub_epi16(q2, q0)), one)
    } else {
        0
    };

    let flat8 = if PLANE == 0 && max_wd >= 8 {
        flat6
            & le_mask_avx512(_mm256_abs_epi16(_mm256_sub_epi16(p3, p0)), one)
            & le_mask_avx512(_mm256_abs_epi16(_mm256_sub_epi16(q3, q0)), one)
    } else {
        0
    };

    (fm, flat6, flat8, abs_p1_p0, abs_q1_q0)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn filter_step2_avx512<const PLANE: usize>(
    p6: __m256i,
    p5: __m256i,
    p4: __m256i,
    p3: __m256i,
    p2: __m256i,
    p1: __m256i,
    p0: __m256i,
    q0: __m256i,
    q1: __m256i,
    q2: __m256i,
    q3: __m256i,
    q4: __m256i,
    q5: __m256i,
    q6: __m256i,
    fm: u16,
    flat6: u16,
    flat8: u16,
    cand16: u16,
    abs_p1_p0: __m256i,
    abs_q1_q0: __m256i,
    h_v: __m256i,
    wd_v: __m256i,
    max_wd: usize,
) -> ([__m256i; 12], bool, bool) {
    let one = _mm256_set1_epi16(1);
    let m16 = if PLANE == 0 && max_wd >= 16 && cand16 != 0 {
        let flat8out = le_mask_avx512(_mm256_abs_epi16(_mm256_sub_epi16(p6, p0)), one)
            & le_mask_avx512(_mm256_abs_epi16(_mm256_sub_epi16(p5, p0)), one)
            & le_mask_avx512(_mm256_abs_epi16(_mm256_sub_epi16(p4, p0)), one)
            & le_mask_avx512(_mm256_abs_epi16(_mm256_sub_epi16(q4, q0)), one)
            & le_mask_avx512(_mm256_abs_epi16(_mm256_sub_epi16(q5, q0)), one)
            & le_mask_avx512(_mm256_abs_epi16(_mm256_sub_epi16(q6, q0)), one);
        cand16 & flat8out
    } else {
        0
    };

    let m8 = if PLANE == 0 && max_wd >= 8 {
        let ge8 = _mm256_cmpgt_epi16_mask(wd_v, _mm256_set1_epi16(6));
        fm & !m16 & ge8 & flat8
    } else {
        0
    };

    let m6 = if PLANE == 1 && max_wd >= 6 {
        let is6 = _mm256_cmpeq_epi16_mask(wd_v, _mm256_set1_epi16(6));
        fm & is6 & flat6
    } else {
        0
    };

    let m4 = fm & !(m16 | m8 | m6);

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

    if m4 != 0 {
        let hev = _mm256_cmpgt_epi16_mask(abs_p1_p0, h_v) | _mm256_cmpgt_epi16_mask(abs_q1_q0, h_v);
        let lo = _mm256_set1_epi16(-128);
        let hi = _mm256_set1_epi16(127);
        let clip = |v: __m256i| _mm256_max_epi16(_mm256_min_epi16(v, hi), lo);

        let fv_hev = _mm256_maskz_mov_epi16(hev, clip(_mm256_sub_epi16(p1, q1)));
        let q0_m_p0_3 = _mm256_mullo_epi16(_mm256_sub_epi16(q0, p0), _mm256_set1_epi16(3));
        let fv = clip(_mm256_add_epi16(q0_m_p0_3, fv_hev));
        let f1 = _mm256_srai_epi16::<3>(_mm256_min_epi16(
            _mm256_add_epi16(fv, _mm256_set1_epi16(4)),
            hi,
        ));
        let f2 = _mm256_srai_epi16::<3>(_mm256_min_epi16(
            _mm256_add_epi16(fv, _mm256_set1_epi16(3)),
            hi,
        ));
        let fv2 = _mm256_maskz_mov_epi16(!hev, _mm256_srai_epi16::<1>(_mm256_add_epi16(f1, one)));

        out_p1 = _mm256_mask_mov_epi16(out_p1, m4, _mm256_add_epi16(p1, fv2));
        out_p0 = _mm256_mask_mov_epi16(out_p0, m4, _mm256_add_epi16(p0, f2));
        out_q0 = _mm256_mask_mov_epi16(out_q0, m4, _mm256_sub_epi16(q0, f1));
        out_q1 = _mm256_mask_mov_epi16(out_q1, m4, _mm256_sub_epi16(q1, fv2));
    }

    if PLANE == 1 && max_wd >= 6 && m6 != 0 {
        let four = _mm256_set1_epi16(4);
        let p2_2 = _mm256_slli_epi16::<1>(p2);
        let p1_2 = _mm256_slli_epi16::<1>(p1);
        let p0_2 = _mm256_slli_epi16::<1>(p0);
        let q0_2 = _mm256_slli_epi16::<1>(q0);
        let q1_2 = _mm256_slli_epi16::<1>(q1);
        let q2_2 = _mm256_slli_epi16::<1>(q2);

        let mid = _mm256_add_epi16(p0_2, four);
        let mid_q = _mm256_add_epi16(q0_2, four);
        let r_p1 = _mm256_srli_epi16::<3>(_mm256_add_epi16(
            _mm256_add_epi16(_mm256_add_epi16(p2_2, p2), p1_2),
            _mm256_add_epi16(mid, q0),
        ));
        let r_p0 = _mm256_srli_epi16::<3>(_mm256_add_epi16(
            _mm256_add_epi16(p2, p1_2),
            _mm256_add_epi16(_mm256_add_epi16(mid, q0_2), q1),
        ));
        let r_q0 = _mm256_srli_epi16::<3>(_mm256_add_epi16(
            _mm256_add_epi16(p1, p0_2),
            _mm256_add_epi16(_mm256_add_epi16(mid_q, q1_2), q2),
        ));
        let r_q1 = _mm256_srli_epi16::<3>(_mm256_add_epi16(
            _mm256_add_epi16(p0, mid_q),
            _mm256_add_epi16(q1_2, _mm256_add_epi16(q2_2, q2)),
        ));

        out_p1 = _mm256_mask_mov_epi16(out_p1, m6, r_p1);
        out_p0 = _mm256_mask_mov_epi16(out_p0, m6, r_p0);
        out_q0 = _mm256_mask_mov_epi16(out_q0, m6, r_q0);
        out_q1 = _mm256_mask_mov_epi16(out_q1, m6, r_q1);
    }

    let has_m8 = PLANE == 0 && max_wd >= 8 && m8 != 0;
    if has_m8 {
        let four = _mm256_set1_epi16(4);
        let p3_2 = _mm256_slli_epi16::<1>(p3);
        let p2_2 = _mm256_slli_epi16::<1>(p2);
        let p1_2 = _mm256_slli_epi16::<1>(p1);
        let p0_2 = _mm256_slli_epi16::<1>(p0);
        let q0_2 = _mm256_slli_epi16::<1>(q0);
        let q1_2 = _mm256_slli_epi16::<1>(q1);
        let q2_2 = _mm256_slli_epi16::<1>(q2);
        let q3_2 = _mm256_slli_epi16::<1>(q3);

        let p1_p0_q0_4 = _mm256_add_epi16(_mm256_add_epi16(p1, p0), _mm256_add_epi16(q0, four));
        let p0_q0_q1_4 = _mm256_add_epi16(_mm256_add_epi16(p0, q0), _mm256_add_epi16(q1, four));

        let r_p2 = _mm256_srli_epi16::<3>(_mm256_add_epi16(
            _mm256_add_epi16(p3_2, p3),
            _mm256_add_epi16(p2_2, p1_p0_q0_4),
        ));
        let r_p1 = _mm256_srli_epi16::<3>(_mm256_add_epi16(
            _mm256_add_epi16(p3_2, p2),
            _mm256_add_epi16(p1_2, p0_q0_q1_4),
        ));
        let r_p0 = _mm256_srli_epi16::<3>(_mm256_add_epi16(
            _mm256_add_epi16(p3, p2),
            _mm256_add_epi16(
                _mm256_add_epi16(p1, p0_2),
                _mm256_add_epi16(_mm256_add_epi16(q0, q1), _mm256_add_epi16(q2, four)),
            ),
        ));
        let r_q0 = _mm256_srli_epi16::<3>(_mm256_add_epi16(
            _mm256_add_epi16(p2, p1),
            _mm256_add_epi16(
                _mm256_add_epi16(p0, q0_2),
                _mm256_add_epi16(_mm256_add_epi16(q1, q2), _mm256_add_epi16(q3, four)),
            ),
        ));
        let r_q1 = _mm256_srli_epi16::<3>(_mm256_add_epi16(
            _mm256_add_epi16(p1_p0_q0_4, q1_2),
            _mm256_add_epi16(q2, q3_2),
        ));
        let r_q2 = _mm256_srli_epi16::<3>(_mm256_add_epi16(
            _mm256_add_epi16(p0_q0_q1_4, q2_2),
            _mm256_add_epi16(q3_2, q3),
        ));

        out_p2 = _mm256_mask_mov_epi16(out_p2, m8, r_p2);
        out_p1 = _mm256_mask_mov_epi16(out_p1, m8, r_p1);
        out_p0 = _mm256_mask_mov_epi16(out_p0, m8, r_p0);
        out_q0 = _mm256_mask_mov_epi16(out_q0, m8, r_q0);
        out_q1 = _mm256_mask_mov_epi16(out_q1, m8, r_q1);
        out_q2 = _mm256_mask_mov_epi16(out_q2, m8, r_q2);
    }

    let has_m16 = PLANE == 0 && max_wd >= 16 && m16 != 0;
    if has_m16 {
        let eight = _mm256_set1_epi16(8);
        let p6_2 = _mm256_slli_epi16::<1>(p6);
        let p6_7 = _mm256_sub_epi16(_mm256_slli_epi16::<3>(p6), p6);
        let p5_2 = _mm256_slli_epi16::<1>(p5);
        let p4_2 = _mm256_slli_epi16::<1>(p4);
        let q4_2 = _mm256_slli_epi16::<1>(q4);
        let q5_2 = _mm256_slli_epi16::<1>(q5);
        let q6_7 = _mm256_sub_epi16(_mm256_slli_epi16::<3>(q6), q6);

        let sum_p3_q0 = _mm256_add_epi16(
            _mm256_add_epi16(p3, p2),
            _mm256_add_epi16(p1, _mm256_add_epi16(p0, q0)),
        );
        let sum_p0_q3 = _mm256_add_epi16(
            _mm256_add_epi16(p0, q0),
            _mm256_add_epi16(q1, _mm256_add_epi16(q2, q3)),
        );

        let s_p5 = _mm256_add_epi16(
            _mm256_add_epi16(p6_7, _mm256_add_epi16(p5_2, p4_2)),
            _mm256_add_epi16(sum_p3_q0, eight),
        );
        let s_p4 = _mm256_add_epi16(_mm256_sub_epi16(s_p5, p6_2), _mm256_add_epi16(p3, q1));
        let s_p3 = _mm256_add_epi16(
            _mm256_sub_epi16(s_p4, _mm256_add_epi16(p6, p5)),
            _mm256_add_epi16(p2, q2),
        );
        let s_p2 = _mm256_add_epi16(
            _mm256_sub_epi16(s_p3, _mm256_add_epi16(p6, p4)),
            _mm256_add_epi16(p1, q3),
        );
        let s_p1 = _mm256_add_epi16(
            _mm256_sub_epi16(s_p2, _mm256_add_epi16(p6, p3)),
            _mm256_add_epi16(p0, q4),
        );
        let s_p0 = _mm256_add_epi16(
            _mm256_sub_epi16(s_p1, _mm256_add_epi16(p6, p2)),
            _mm256_add_epi16(q0, q5),
        );
        let s_q0 = _mm256_add_epi16(
            _mm256_sub_epi16(s_p0, _mm256_add_epi16(p6, p1)),
            _mm256_add_epi16(q1, q6),
        );
        let s_q1 = _mm256_add_epi16(
            _mm256_sub_epi16(s_q0, _mm256_add_epi16(p5, p0)),
            _mm256_add_epi16(q2, q6),
        );
        let s_q2 = _mm256_add_epi16(
            _mm256_sub_epi16(s_q1, _mm256_add_epi16(p4, q0)),
            _mm256_add_epi16(q3, q6),
        );
        let s_q3 = _mm256_add_epi16(
            _mm256_sub_epi16(s_q2, _mm256_add_epi16(p3, q1)),
            _mm256_add_epi16(q4, q6),
        );
        let s_q4 = _mm256_add_epi16(
            _mm256_sub_epi16(s_q3, _mm256_add_epi16(p2, q2)),
            _mm256_add_epi16(q5, q6),
        );
        let s_q5 = _mm256_add_epi16(
            _mm256_add_epi16(q6_7, _mm256_add_epi16(q5_2, q4_2)),
            _mm256_add_epi16(sum_p0_q3, eight),
        );

        out_p5 = _mm256_mask_mov_epi16(out_p5, m16, _mm256_srli_epi16::<4>(s_p5));
        out_p4 = _mm256_mask_mov_epi16(out_p4, m16, _mm256_srli_epi16::<4>(s_p4));
        out_p3 = _mm256_mask_mov_epi16(out_p3, m16, _mm256_srli_epi16::<4>(s_p3));
        out_p2 = _mm256_mask_mov_epi16(out_p2, m16, _mm256_srli_epi16::<4>(s_p2));
        out_p1 = _mm256_mask_mov_epi16(out_p1, m16, _mm256_srli_epi16::<4>(s_p1));
        out_p0 = _mm256_mask_mov_epi16(out_p0, m16, _mm256_srli_epi16::<4>(s_p0));
        out_q0 = _mm256_mask_mov_epi16(out_q0, m16, _mm256_srli_epi16::<4>(s_q0));
        out_q1 = _mm256_mask_mov_epi16(out_q1, m16, _mm256_srli_epi16::<4>(s_q1));
        out_q2 = _mm256_mask_mov_epi16(out_q2, m16, _mm256_srli_epi16::<4>(s_q2));
        out_q3 = _mm256_mask_mov_epi16(out_q3, m16, _mm256_srli_epi16::<4>(s_q3));
        out_q4 = _mm256_mask_mov_epi16(out_q4, m16, _mm256_srli_epi16::<4>(s_q4));
        out_q5 = _mm256_mask_mov_epi16(out_q5, m16, _mm256_srli_epi16::<4>(s_q5));
    }

    (
        [
            out_p5, out_p4, out_p3, out_p2, out_p1, out_p0, out_q0, out_q1, out_q2, out_q3, out_q4,
            out_q5,
        ],
        has_m8,
        has_m16,
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn load_h_4x4_avx512(dst: &[u8], off: usize, stride: usize) -> __m128i {
    let r0 = u32::from_le_bytes(*mem::chunk::<u8, 4>(dst, off)) as i32;
    let r1 = u32::from_le_bytes(*mem::chunk::<u8, 4>(dst, off + stride)) as i32;
    let r2 = u32::from_le_bytes(*mem::chunk::<u8, 4>(dst, off + 2 * stride)) as i32;
    let r3 = u32::from_le_bytes(*mem::chunk::<u8, 4>(dst, off + 3 * stride)) as i32;
    _mm_setr_epi32(r0, r1, r2, r3)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn load_h_2x2_avx512(dst: &[u8], off_a: usize, off_b: usize, stride: usize) -> __m128i {
    let mut rows = [0i32; 4];
    for k in 0..4 {
        let d = k * stride;
        let a = u16::from_le_bytes(*mem::chunk::<u8, 2>(dst, off_a + d)) as u32;
        let b = u16::from_le_bytes(*mem::chunk::<u8, 2>(dst, off_b + d)) as u32;
        rows[k] = (a | (b << 16)) as i32;
    }
    _mm_setr_epi32(rows[0], rows[1], rows[2], rows[3])
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn transpose_in_h_avx512(
    m0: __m128i,
    m1: __m128i,
    m2: __m128i,
    m3: __m128i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let shuf = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    let t0 = _mm_shuffle_epi8(m0, shuf);
    let t1 = _mm_shuffle_epi8(m1, shuf);
    let t2 = _mm_shuffle_epi8(m2, shuf);
    let t3 = _mm_shuffle_epi8(m3, shuf);
    let u01_lo = _mm_unpacklo_epi32(t0, t1);
    let u01_hi = _mm_unpackhi_epi32(t0, t1);
    let u23_lo = _mm_unpacklo_epi32(t2, t3);
    let u23_hi = _mm_unpackhi_epi32(t2, t3);
    (
        _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(u01_lo, u23_lo)),
        _mm256_cvtepu8_epi16(_mm_unpackhi_epi64(u01_lo, u23_lo)),
        _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(u01_hi, u23_hi)),
        _mm256_cvtepu8_epi16(_mm_unpackhi_epi64(u01_hi, u23_hi)),
    )
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn transpose_out_h_avx512(c0: __m256i, c1: __m256i, c2: __m256i, c3: __m256i) -> [__m128i; 4] {
    let shuf = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    let b0 = pack_tap_u8_avx512(c0);
    let b1 = pack_tap_u8_avx512(c1);
    let b2 = pack_tap_u8_avx512(c2);
    let b3 = pack_tap_u8_avx512(c3);
    let v_lo = _mm_unpacklo_epi32(b0, b1);
    let v_hi = _mm_unpackhi_epi32(b0, b1);
    let w_lo = _mm_unpacklo_epi32(b2, b3);
    let w_hi = _mm_unpackhi_epi32(b2, b3);
    [
        _mm_shuffle_epi8(_mm_unpacklo_epi64(v_lo, w_lo), shuf),
        _mm_shuffle_epi8(_mm_unpackhi_epi64(v_lo, w_lo), shuf),
        _mm_shuffle_epi8(_mm_unpacklo_epi64(v_hi, w_hi), shuf),
        _mm_shuffle_epi8(_mm_unpackhi_epi64(v_hi, w_hi), shuf),
    ]
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn store_h_4x4_avx512(dst: &mut [u8], off: usize, stride: usize, r: __m128i) {
    *mem::chunk_mut::<u8, 4>(dst, off) = (_mm_cvtsi128_si32(r) as u32).to_le_bytes();
    *mem::chunk_mut::<u8, 4>(dst, off + stride) = (_mm_extract_epi32::<1>(r) as u32).to_le_bytes();
    *mem::chunk_mut::<u8, 4>(dst, off + 2 * stride) =
        (_mm_extract_epi32::<2>(r) as u32).to_le_bytes();
    *mem::chunk_mut::<u8, 4>(dst, off + 3 * stride) =
        (_mm_extract_epi32::<3>(r) as u32).to_le_bytes();
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn flush_h_batch_avx512<const PLANE: usize>(
    dst: &mut [u8],
    stride: usize,
    batch: &Batch4,
    n: usize,
    max_wd: usize,
) {
    if max_wd == 4 {
        let e_v = broadcast_packed4_avx512(batch.e_packed);
        let i_v = broadcast_packed4_avx512(batch.i_packed);
        let h_v = broadcast_packed4_avx512(batch.h_packed);
        let wd_v = broadcast_packed4_avx512(batch.wd_packed);
        let z128 = _mm_setzero_si128();
        let m0 = load_h_4x4_avx512(dst, batch.off[0] - 2, stride);
        let m1 = if n > 1 {
            load_h_4x4_avx512(dst, batch.off[1] - 2, stride)
        } else {
            z128
        };
        let m2 = if n > 2 {
            load_h_4x4_avx512(dst, batch.off[2] - 2, stride)
        } else {
            z128
        };
        let m3 = if n > 3 {
            load_h_4x4_avx512(dst, batch.off[3] - 2, stride)
        } else {
            z128
        };
        let (p1, p0, q0, q1) = transpose_in_h_avx512(m0, m1, m2, m3);
        if let Some((op1, op0, oq0, oq1)) =
            filter_wd4_only_avx512(p1, p0, q0, q1, e_v, i_v, h_v, wd_v)
        {
            let mid = transpose_out_h_avx512(op1, op0, oq0, oq1);
            for b in 0..n {
                store_h_4x4_avx512(dst, batch.off[b] - 2, stride, mid[b]);
            }
        }
        return;
    }
    flush_h_batch_wide_avx512::<PLANE>(dst, stride, batch, n, max_wd);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline(never)]
fn flush_h_batch_wide_avx512<const PLANE: usize>(
    dst: &mut [u8],
    stride: usize,
    batch: &Batch4,
    n: usize,
    max_wd: usize,
) {
    let (e_v, i_v, h_v, wd_v, _min_wd) = prepare_batch_avx512(batch, n);
    let mut m_in = [_mm_setzero_si128(); 4];
    for b in 0..n {
        m_in[b] = load_h_4x4_avx512(dst, batch.off[b] - 2, stride);
    }
    let (p1, p0, q0, q1) = transpose_in_h_avx512(m_in[0], m_in[1], m_in[2], m_in[3]);

    let z = _mm256_setzero_si256();
    let (p3, p2, q2, q3) = if PLANE == 0 && max_wd >= 8 {
        let mut m8 = [_mm_setzero_si128(); 4];
        for b in 0..n {
            if batch.wd[b] >= 8 {
                let o = batch.off[b];
                m8[b] = load_h_2x2_avx512(dst, o - 4, o + 2, stride);
            }
        }
        transpose_in_h_avx512(m8[0], m8[1], m8[2], m8[3])
    } else if PLANE == 1 && max_wd >= 6 {
        let mut m6 = [_mm_setzero_si128(); 4];
        for b in 0..n {
            if batch.wd[b] >= 6 {
                let o = batch.off[b];
                m6[b] = load_h_2x2_avx512(dst, o - 3, o + 1, stride);
            }
        }
        let (p2_v, _, _, q2_v) = transpose_in_h_avx512(m6[0], m6[1], m6[2], m6[3]);
        (z, p2_v, q2_v, z)
    } else {
        (z, z, z, z)
    };

    let (fm, flat6, flat8, abs_p1_p0, abs_q1_q0) =
        filter_step1_avx512::<PLANE>(p1, p0, q0, q1, p2, q2, p3, q3, e_v, i_v, wd_v, max_wd);
    if fm == 0 {
        return;
    }

    let cand16 = if PLANE == 0 && max_wd >= 16 {
        fm & _mm256_cmpeq_epi16_mask(wd_v, _mm256_set1_epi16(16)) & flat8
    } else {
        0
    };

    let (p6, p5, p4, q4, q5, q6) = if PLANE == 0 && max_wd >= 16 && cand16 != 0 {
        let mut m16_lo = [_mm_setzero_si128(); 4];
        let mut m16_hi = [_mm_setzero_si128(); 4];
        for b in 0..n {
            if batch.wd[b] >= 16 {
                let o = batch.off[b];
                m16_lo[b] = load_h_4x4_avx512(dst, o - 7, stride);
                m16_hi[b] = load_h_4x4_avx512(dst, o + 3, stride);
            }
        }
        let (p6_v, p5_v, p4_v, _) =
            transpose_in_h_avx512(m16_lo[0], m16_lo[1], m16_lo[2], m16_lo[3]);
        let (_, q4_v, q5_v, q6_v) =
            transpose_in_h_avx512(m16_hi[0], m16_hi[1], m16_hi[2], m16_hi[3]);
        (p6_v, p5_v, p4_v, q4_v, q5_v, q6_v)
    } else {
        (z, z, z, z, z, z)
    };

    let (out, has_m8, has_m16) = filter_step2_avx512::<PLANE>(
        p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, fm, flat6, flat8, cand16,
        abs_p1_p0, abs_q1_q0, h_v, wd_v, max_wd,
    );

    let mid = transpose_out_h_avx512(out[4], out[5], out[6], out[7]);
    for b in 0..n {
        store_h_4x4_avx512(dst, batch.off[b] - 2, stride, mid[b]);
    }
    if has_m8 || has_m16 {
        let lo8 = transpose_out_h_avx512(out[3], out[4], out[5], out[6]);
        let hi8 = transpose_out_h_avx512(out[5], out[6], out[7], out[8]);
        for b in 0..n {
            if batch.wd[b] >= 8 {
                store_h_4x4_avx512(dst, batch.off[b] - 3, stride, lo8[b]);
                store_h_4x4_avx512(dst, batch.off[b] - 1, stride, hi8[b]);
            }
        }
    }
    if has_m16 {
        let lo16 = transpose_out_h_avx512(out[0], out[1], out[2], out[3]);
        let hi16 = transpose_out_h_avx512(out[8], out[9], out[10], out[11]);
        for b in 0..n {
            if batch.wd[b] == 16 {
                store_h_4x4_avx512(dst, batch.off[b] - 6, stride, lo16[b]);
                store_h_4x4_avx512(dst, batch.off[b] + 2, stride, hi16[b]);
            }
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn load_v_tap_avx512(
    dst: &[u8],
    stride: usize,
    tap_idx: usize,
    need_wd: usize,
    batch: &Batch4,
    n: usize,
    contig4: bool,
    contig2: bool,
    min_wd: usize,
) -> __m256i {
    let add_stride = tap_idx * stride;
    let sub_stride = 7 * stride;
    if contig4 && min_wd >= need_wd {
        let base = (batch.off[0] + add_stride) - sub_stride;
        return _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(dst, base)));
    }
    if contig2 && min_wd >= need_wd {
        let base = (batch.off[0] + add_stride) - sub_stride;
        return _mm256_cvtepu8_epi16(_mm_cvtsi64_si128(i64::from_le_bytes(*mem::chunk::<u8, 8>(
            dst, base,
        ))));
    }
    let need_u8 = need_wd as u8;
    let mut row = [0u8; 16];
    for b in 0..n {
        if batch.wd[b] >= need_u8 {
            let s = mem::chunk::<u8, 4>(dst, (batch.off[b] + add_stride) - sub_stride);
            row[b * 4..b * 4 + 4].copy_from_slice(s);
        }
    }
    _mm256_cvtepu8_epi16(mem::load_u8x16(&row))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn store_v_tap_avx512(
    dst: &mut [u8],
    stride: usize,
    tap_idx: usize,
    need_wd: usize,
    batch: &Batch4,
    n: usize,
    contig4: bool,
    contig2: bool,
    min_wd: usize,
    v: __m256i,
) {
    let packed = pack_tap_u8_avx512(v);
    let add_stride = tap_idx * stride;
    let sub_stride = 7 * stride;
    if contig4 && min_wd >= need_wd {
        let base = (batch.off[0] + add_stride) - sub_stride;
        mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, base), packed);
        return;
    }
    if contig2 && min_wd >= need_wd {
        let base = (batch.off[0] + add_stride) - sub_stride;
        mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, base), packed);
        return;
    }
    let need_u8 = need_wd as u8;
    let mut row = [0u8; 16];
    mem::store_u8x16(&mut row, packed);
    for b in 0..n {
        if batch.wd[b] >= need_u8 {
            let d = mem::chunk_mut::<u8, 4>(dst, (batch.off[b] + add_stride) - sub_stride);
            d.copy_from_slice(&row[b * 4..b * 4 + 4]);
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn flush_v_batch_avx512<const PLANE: usize>(
    dst: &mut [u8],
    stride: usize,
    batch: &Batch4,
    n: usize,
    max_wd: usize,
) {
    let contig4 = n == 4
        && batch.off[1] == batch.off[0] + 4
        && batch.off[2] == batch.off[0] + 8
        && batch.off[3] == batch.off[0] + 12;
    let contig2 = n == 2 && batch.off[1] == batch.off[0] + 4;
    if max_wd == 4 {
        let e_v = broadcast_packed4_avx512(batch.e_packed);
        let i_v = broadcast_packed4_avx512(batch.i_packed);
        let h_v = broadcast_packed4_avx512(batch.h_packed);
        let wd_v = broadcast_packed4_avx512(batch.wd_packed);
        let p1 = load_v_tap_avx512(dst, stride, 5, 4, batch, n, contig4, contig2, 4);
        let p0 = load_v_tap_avx512(dst, stride, 6, 4, batch, n, contig4, contig2, 4);
        let q0 = load_v_tap_avx512(dst, stride, 7, 4, batch, n, contig4, contig2, 4);
        let q1 = load_v_tap_avx512(dst, stride, 8, 4, batch, n, contig4, contig2, 4);
        if let Some((op1, op0, oq0, oq1)) =
            filter_wd4_only_avx512(p1, p0, q0, q1, e_v, i_v, h_v, wd_v)
        {
            store_v_tap_avx512(dst, stride, 5, 4, batch, n, contig4, contig2, 4, op1);
            store_v_tap_avx512(dst, stride, 6, 4, batch, n, contig4, contig2, 4, op0);
            store_v_tap_avx512(dst, stride, 7, 4, batch, n, contig4, contig2, 4, oq0);
            store_v_tap_avx512(dst, stride, 8, 4, batch, n, contig4, contig2, 4, oq1);
        }
        return;
    }
    flush_v_batch_wide_avx512::<PLANE>(dst, stride, batch, n, max_wd, contig4, contig2);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline(never)]
fn flush_v_batch_wide_avx512<const PLANE: usize>(
    dst: &mut [u8],
    stride: usize,
    batch: &Batch4,
    n: usize,
    max_wd: usize,
    contig4: bool,
    contig2: bool,
) {
    let (e_v, i_v, h_v, wd_v, min_wd) = prepare_batch_avx512(batch, n);

    let p1 = load_v_tap_avx512(dst, stride, 5, 4, batch, n, contig4, contig2, min_wd);
    let p0 = load_v_tap_avx512(dst, stride, 6, 4, batch, n, contig4, contig2, min_wd);
    let q0 = load_v_tap_avx512(dst, stride, 7, 4, batch, n, contig4, contig2, min_wd);
    let q1 = load_v_tap_avx512(dst, stride, 8, 4, batch, n, contig4, contig2, min_wd);

    let z = _mm256_setzero_si256();
    let (p2, q2) = if max_wd >= 6 {
        let need = if PLANE == 0 { 8 } else { 6 };
        (
            load_v_tap_avx512(dst, stride, 4, need, batch, n, contig4, contig2, min_wd),
            load_v_tap_avx512(dst, stride, 9, need, batch, n, contig4, contig2, min_wd),
        )
    } else {
        (z, z)
    };

    let (p3, q3) = if PLANE == 0 && max_wd >= 8 {
        (
            load_v_tap_avx512(dst, stride, 3, 8, batch, n, contig4, contig2, min_wd),
            load_v_tap_avx512(dst, stride, 10, 8, batch, n, contig4, contig2, min_wd),
        )
    } else {
        (z, z)
    };

    let (fm, flat6, flat8, abs_p1_p0, abs_q1_q0) =
        filter_step1_avx512::<PLANE>(p1, p0, q0, q1, p2, q2, p3, q3, e_v, i_v, wd_v, max_wd);
    if fm == 0 {
        return;
    }

    let cand16 = if PLANE == 0 && max_wd >= 16 {
        fm & _mm256_cmpeq_epi16_mask(wd_v, _mm256_set1_epi16(16)) & flat8
    } else {
        0
    };

    let (p6, p5, p4, q4, q5, q6) = if PLANE == 0 && max_wd >= 16 && cand16 != 0 {
        (
            load_v_tap_avx512(dst, stride, 0, 16, batch, n, contig4, contig2, min_wd),
            load_v_tap_avx512(dst, stride, 1, 16, batch, n, contig4, contig2, min_wd),
            load_v_tap_avx512(dst, stride, 2, 16, batch, n, contig4, contig2, min_wd),
            load_v_tap_avx512(dst, stride, 11, 16, batch, n, contig4, contig2, min_wd),
            load_v_tap_avx512(dst, stride, 12, 16, batch, n, contig4, contig2, min_wd),
            load_v_tap_avx512(dst, stride, 13, 16, batch, n, contig4, contig2, min_wd),
        )
    } else {
        (z, z, z, z, z, z)
    };

    let (out, has_m8, has_m16) = filter_step2_avx512::<PLANE>(
        p6, p5, p4, p3, p2, p1, p0, q0, q1, q2, q3, q4, q5, q6, fm, flat6, flat8, cand16,
        abs_p1_p0, abs_q1_q0, h_v, wd_v, max_wd,
    );

    store_v_tap_avx512(
        dst, stride, 5, 4, batch, n, contig4, contig2, min_wd, out[4],
    );
    store_v_tap_avx512(
        dst, stride, 6, 4, batch, n, contig4, contig2, min_wd, out[5],
    );
    store_v_tap_avx512(
        dst, stride, 7, 4, batch, n, contig4, contig2, min_wd, out[6],
    );
    store_v_tap_avx512(
        dst, stride, 8, 4, batch, n, contig4, contig2, min_wd, out[7],
    );
    if has_m8 || has_m16 {
        store_v_tap_avx512(
            dst, stride, 4, 8, batch, n, contig4, contig2, min_wd, out[3],
        );
        store_v_tap_avx512(
            dst, stride, 9, 8, batch, n, contig4, contig2, min_wd, out[8],
        );
    }
    if has_m16 {
        store_v_tap_avx512(
            dst, stride, 1, 16, batch, n, contig4, contig2, min_wd, out[0],
        );
        store_v_tap_avx512(
            dst, stride, 2, 16, batch, n, contig4, contig2, min_wd, out[1],
        );
        store_v_tap_avx512(
            dst, stride, 3, 16, batch, n, contig4, contig2, min_wd, out[2],
        );
        store_v_tap_avx512(
            dst, stride, 10, 16, batch, n, contig4, contig2, min_wd, out[9],
        );
        store_v_tap_avx512(
            dst, stride, 11, 16, batch, n, contig4, contig2, min_wd, out[10],
        );
        store_v_tap_avx512(
            dst, stride, 12, 16, batch, n, contig4, contig2, min_wd, out[11],
        );
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
fn lpf_h_sb_avx512_giris<const PLANE: usize>(
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
    let mut batch = Batch4::default();
    let mut n = 0usize;
    let mut max_wd = 0usize;
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
            let wd = edge_wd::<PLANE>(v);
            if count == 1 && n > 0 && (max_wd > 4 || wd > 4) {
                flush_h_batch_avx512::<PLANE>(dst, stride, &batch, n, max_wd);
                batch.reset();
                n = 0;
                max_wd = 0;
            }
            let off = gy * 4 * stride + gx * 4;
            batch.push(n, off, wd, lut.e[l], lut.i[l], l >> 4);
            if wd > max_wd {
                max_wd = wd;
            }
            n += 1;
            if n == 4 {
                flush_h_batch_avx512::<PLANE>(dst, stride, &batch, 4, max_wd);
                batch.reset();
                n = 0;
                max_wd = 0;
            }
        }
        if count > 1 && n > 0 {
            flush_h_batch_avx512::<PLANE>(dst, stride, &batch, n, max_wd);
            batch.reset();
            n = 0;
            max_wd = 0;
        }
    }
    if n > 0 {
        flush_h_batch_avx512::<PLANE>(dst, stride, &batch, n, max_wd);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
fn lpf_v_sb_avx512_giris<const PLANE: usize>(
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
    let mut batch = Batch4::default();
    for gy in y0..y0 + count {
        if gy == 0 {
            continue;
        }
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
            batch.push(n, off, wd, lut.e[l], lut.i[l], l >> 4);
            if wd > max_wd {
                max_wd = wd;
            }
            n += 1;
            if n == 4 {
                flush_v_batch_avx512::<PLANE>(dst, stride, &batch, 4, max_wd);
                batch.reset();
                n = 0;
                max_wd = 0;
            }
        }
        if n > 0 {
            flush_v_batch_avx512::<PLANE>(dst, stride, &batch, n, max_wd);
            batch.reset();
        }
    }
}
