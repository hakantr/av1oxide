//! 8 bpc loop restoration (Wiener / SGRPROJ) x86_64 SSSE3 çekirdekleri (`D02750`,
//! `D02756`, `D02731`, `D02737`, `D02743`; P46 Adım 3 / Paket A).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Kayıt | Yuva | Sembol | Kaynak |
//! |---|---|---|---|
//! | `D02750` | `wiener[0]` | `dav1d_wiener_filter7_8bpc_ssse3` | `src/x86/looprestoration_sse.asm:257-579`, `src/x86/looprestoration.h:54` |
//! | `D02756` | `wiener[1]` | `dav1d_wiener_filter5_8bpc_ssse3` | `src/x86/looprestoration_sse.asm:257-579`, `src/x86/looprestoration.h:55` |
//! | `D02731` | `sgr[0]` | `dav1d_sgr_filter_5x5_8bpc_ssse3` | `src/x86/looprestoration_sse.asm:581-1565`, `src/x86/looprestoration.h:56-67` |
//! | `D02737` | `sgr[1]` | `dav1d_sgr_filter_3x3_8bpc_ssse3` | `src/x86/looprestoration_sse.asm:581-1565`, `src/x86/looprestoration.h:56-67` |
//! | `D02743` | `sgr[2]` | `dav1d_sgr_filter_mix_8bpc_ssse3` | `src/x86/looprestoration_sse.asm:581-1565`, `src/x86/looprestoration.h:56-67` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bu modülde tüm SIMD işlevleri yalnız `#[target_feature(enable = "ssse3")]`
//!   taşır; SSE4.1+, VEX ve EVEX yönergesi üretilmez (`0` SSE4.1+/VEX/EVEX).
//! - Kopyasız O(1) sözleşme: girişte [`wiener_sozlesmesi::<8>`] veya
//!   [`sgr_sozlesmesi::<8>`] ve `Ssse3Token::new(CpuFlags::detect())`
//!   denetlenir; sözleşme dışı girdide ilgili skaler devir yardımcısıyla
//!   skaler gövdeye devredilir.
//! - Sınır-güvenli (`SentinelBuffer` / `OverReadProbe`): her kaynak satırı
//!   yalnız sözleşmenin izin verdiği pikseller okunarak yığın üzerindeki
//!   dolgulu `[u8; 420]` tamponuna hazırlanır; çıktı da yalnız `[0, w)`
//!   sütunlarına yazılır.

#![allow(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_add_epi32, _mm_madd_epi16, _mm_packs_epi32, _mm_packus_epi16,
    _mm_set1_epi16, _mm_set1_epi32, _mm_setzero_si128, _mm_slli_epi32, _mm_srai_epi32,
    _mm_sub_epi32, _mm_unpackhi_epi8, _mm_unpackhi_epi16, _mm_unpacklo_epi8, _mm_unpacklo_epi16,
};

use crate::cpu::{CpuFlags, Ssse3Token};
use crate::dsp::lr::{
    LrBitDepth, LrEdgeFlags, LrFilterFn, LrParams, sgr_skaler_devret, sgr_sozlesmesi,
    wiener_skaler_devret, wiener_sozlesmesi,
};
use crate::dsp::lr_tables::SGR_X_BY_X;
use crate::dsp::x86::lr_sse2::{
    HOR_STRIDE, LrCtx, PAD_ROW_LEN, RowRef, pack32_to_14b_sse2, pack32_to_u8x16_sse2, pad_row_8bpc,
};
use crate::dsp::x86::mem;

/// SGR iç tampon adımı (`384 + 16 = 400`, 16'nın katı).
pub(crate) const SGR_BUF_STRIDE: usize = 400;
/// SGR `finish_filter` çıkış adımı (`384`, 16'nın katı).
pub(crate) const SGR_OUT_STRIDE: usize = 384;

/// `N` satırlık SGR çift tamponu (`sq` = kareler toplamı / `A`, `s` = toplam / `B`).
pub(crate) struct SgrRows<const N: usize> {
    pub(crate) sq: [[i32; SGR_BUF_STRIDE]; N],
    pub(crate) s: [[i32; SGR_BUF_STRIDE]; N],
}

impl<const N: usize> SgrRows<N> {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self {
            sq: [[0; SGR_BUF_STRIDE]; N],
            s: [[0; SGR_BUF_STRIDE]; N],
        }
    }
}

/// 8 bpc `wiener[tap_idx]` SSSE3 kurucusu (`D02750`, `D02756`).
#[must_use]
pub const fn wiener_filter_fn(_token: Ssse3Token, tap_idx: usize) -> Option<LrFilterFn> {
    match tap_idx {
        0 => Some(wiener_filter7_8bpc_ssse3),
        1 => Some(wiener_filter5_8bpc_ssse3),
        _ => None,
    }
}

/// 8 bpc `sgr[sgr_idx]` SSSE3 kurucusu (`D02731`, `D02737`, `D02743`).
#[must_use]
pub const fn sgr_filter_fn(_token: Ssse3Token, sgr_idx: usize) -> Option<LrFilterFn> {
    match sgr_idx {
        0 => Some(sgr_filter_5x5_8bpc_ssse3),
        1 => Some(sgr_filter_3x3_8bpc_ssse3),
        2 => Some(sgr_filter_mix_8bpc_ssse3),
        _ => None,
    }
}

/// 8 bpc 7-dokunuşlu Wiener filtresi — SSSE3 (`D02750`, `wiener[0]`).
pub fn wiener_filter7_8bpc_ssse3(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    lpf: &[u8],
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    params: &LrParams,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
) {
    wiener_8bpc_ssse3::<false>(
        0, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
    );
}

/// 8 bpc 5-dokunuşlu Wiener filtresi — SSSE3 (`D02756`, `wiener[1]`).
pub fn wiener_filter5_8bpc_ssse3(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    lpf: &[u8],
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    params: &LrParams,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
) {
    wiener_8bpc_ssse3::<true>(
        1, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
    );
}

/// 8 bpc SGR 5×5 filtresi — SSSE3 (`D02731`, `sgr[0]`).
pub fn sgr_filter_5x5_8bpc_ssse3(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    lpf: &[u8],
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    params: &LrParams,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
) {
    sgr_8bpc_ssse3::<0>(
        dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
    );
}

/// 8 bpc SGR 3×3 filtresi — SSSE3 (`D02737`, `sgr[1]`).
pub fn sgr_filter_3x3_8bpc_ssse3(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    lpf: &[u8],
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    params: &LrParams,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
) {
    sgr_8bpc_ssse3::<1>(
        dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
    );
}

/// 8 bpc SGR karma (5×5 + 3×3) filtresi — SSSE3 (`D02743`, `sgr[2]`).
pub fn sgr_filter_mix_8bpc_ssse3(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    lpf: &[u8],
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    params: &LrParams,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
) {
    sgr_8bpc_ssse3::<2>(
        dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
    );
}

#[inline(always)]
fn wiener_8bpc_ssse3<const FIVE_TAP: bool>(
    tap_idx: usize,
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    lpf: &[u8],
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    params: &LrParams,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
) {
    if !wiener_sozlesmesi::<8>(
        tap_idx,
        dst.len(),
        stride,
        left.len(),
        lpf.len(),
        lpf_stride,
        lpf_x,
        w,
        h,
        params,
        edges,
        bd,
    ) || Ssse3Token::new(CpuFlags::detect()).is_none()
    {
        wiener_skaler_devret::<8>(
            tap_idx, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
        );
        return;
    }
    let LrParams::Wiener { filter } = params else {
        return;
    };
    let c = LrCtx {
        left,
        lpf,
        stride,
        lpf_stride,
        lpf_x,
        w,
        edges,
    };
    // SAFETY: (1) ISA: yukarıda `Ssse3Token::new(CpuFlags::detect())` doğrulandı;
    // SSSE3 çalışma zamanında mevcuttur. (2) sınır: `wiener_sozlesmesi::<8>` `dst`,
    // `left` ve `lpf` dilimlerinin sınırlarını O(1) doğruladı; `pad_row_8bpc` yalnız
    // izinli pikselleri yığın tamponuna kopyalar ve çıktı yalnız `[0, w)` aralığına
    // yazılır. (3) hizalama/takma ad: tüm vektör yükleme ve saklamaları `mem`
    // içindeki hizasız sabit boyutlu dizi yardımcılarıyla yapılır.
    unsafe {
        wiener_ssse3_giris::<FIVE_TAP>(dst, &c, h, filter);
    }
}

#[inline(always)]
fn sgr_8bpc_ssse3<const SGR_IDX: usize>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    lpf: &[u8],
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    params: &LrParams,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
) {
    if !sgr_sozlesmesi::<8>(
        SGR_IDX,
        dst.len(),
        stride,
        left.len(),
        lpf.len(),
        lpf_stride,
        lpf_x,
        w,
        h,
        params,
        edges,
        bd,
    ) || Ssse3Token::new(CpuFlags::detect()).is_none()
    {
        sgr_skaler_devret::<8>(
            SGR_IDX, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
        );
        return;
    }
    let LrParams::Sgr { s0, s1, w0, w1 } = *params else {
        return;
    };
    let c = LrCtx {
        left,
        lpf,
        stride,
        lpf_stride,
        lpf_x,
        w,
        edges,
    };
    // SAFETY: (1) ISA: yukarıda `Ssse3Token::new(CpuFlags::detect())` doğrulandı;
    // SSSE3 çalışma zamanında mevcuttur. (2) sınır: `sgr_sozlesmesi::<8>` `dst`,
    // `left` ve `lpf` dilimlerinin sınırlarını O(1) doğruladı; `pad_row_8bpc` yalnız
    // izinli pikselleri yığın tamponuna kopyalar ve çıktı yalnız `[0, w)` aralığına
    // yazılır. (3) hizalama/takma ad: tüm vektör yükleme ve saklamaları `mem`
    // içindeki hizasız sabit boyutlu dizi yardımcılarıyla yapılır.
    unsafe {
        match SGR_IDX {
            0 => sgr_5x5_ssse3_giris(dst, &c, h, s0, w0),
            1 => sgr_3x3_ssse3_giris(dst, &c, h, s1, w1),
            _ => sgr_mix_ssse3_giris(dst, &c, h, s0, s1, w0, w1),
        }
    }
}

// ---------------------------------------------------------------------------
// Wiener SSSE3 (`_mm_maddubs_epi16` + `_mm_madd_epi16`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
#[inline]
fn wiener_filter_h_ssse3<const FIVE_TAP: bool>(
    out: &mut [i16; HOR_STRIDE],
    c: &LrCtx<'_>,
    dst: &[u8],
    row: RowRef,
    fh: &[i16; 8],
) {
    let mut pad = [0u8; PAD_ROW_LEN];
    pad_row_8bpc::<3>(&mut pad, c, dst, row);

    let z = _mm_setzero_si128();
    let c06 = _mm_set1_epi32((i32::from(fh[0]) & 0xffff) | ((i32::from(fh[6]) & 0xffff) << 16));
    let c15 = _mm_set1_epi32((i32::from(fh[1]) & 0xffff) | ((i32::from(fh[5]) & 0xffff) << 16));
    let c24 = _mm_set1_epi32((i32::from(fh[2]) & 0xffff) | ((i32::from(fh[4]) & 0xffff) << 16));
    let c3 = _mm_set1_epi32(((i32::from(fh[3]) + 128) & 0xffff) | (16388 << 16));
    let one16 = _mm_set1_epi16(1);

    let mut x = 0usize;
    while x < c.w {
        let p0 = mem::load_u8x16(mem::chunk::<u8, 16>(&pad, x));
        let p1 = mem::load_u8x16(mem::chunk::<u8, 16>(&pad, x + 1));
        let p2 = mem::load_u8x16(mem::chunk::<u8, 16>(&pad, x + 2));
        let p3 = mem::load_u8x16(mem::chunk::<u8, 16>(&pad, x + 3));
        let p4 = mem::load_u8x16(mem::chunk::<u8, 16>(&pad, x + 4));
        let p5 = mem::load_u8x16(mem::chunk::<u8, 16>(&pad, x + 5));
        let p6 = mem::load_u8x16(mem::chunk::<u8, 16>(&pad, x + 6));

        let p1_lo = _mm_unpacklo_epi8(p1, z);
        let p1_hi = _mm_unpackhi_epi8(p1, z);
        let p2_lo = _mm_unpacklo_epi8(p2, z);
        let p2_hi = _mm_unpackhi_epi8(p2, z);
        let p3_lo = _mm_unpacklo_epi8(p3, z);
        let p3_hi = _mm_unpackhi_epi8(p3, z);
        let p4_lo = _mm_unpacklo_epi8(p4, z);
        let p4_hi = _mm_unpackhi_epi8(p4, z);
        let p5_lo = _mm_unpacklo_epi8(p5, z);
        let p5_hi = _mm_unpackhi_epi8(p5, z);

        let mut s0 = _mm_add_epi32(
            _mm_add_epi32(
                _mm_madd_epi16(_mm_unpacklo_epi16(p1_lo, p5_lo), c15),
                _mm_madd_epi16(_mm_unpacklo_epi16(p2_lo, p4_lo), c24),
            ),
            _mm_madd_epi16(_mm_unpacklo_epi16(p3_lo, one16), c3),
        );
        let mut s1 = _mm_add_epi32(
            _mm_add_epi32(
                _mm_madd_epi16(_mm_unpackhi_epi16(p1_lo, p5_lo), c15),
                _mm_madd_epi16(_mm_unpackhi_epi16(p2_lo, p4_lo), c24),
            ),
            _mm_madd_epi16(_mm_unpackhi_epi16(p3_lo, one16), c3),
        );
        let mut s2 = _mm_add_epi32(
            _mm_add_epi32(
                _mm_madd_epi16(_mm_unpacklo_epi16(p1_hi, p5_hi), c15),
                _mm_madd_epi16(_mm_unpacklo_epi16(p2_hi, p4_hi), c24),
            ),
            _mm_madd_epi16(_mm_unpacklo_epi16(p3_hi, one16), c3),
        );
        let mut s3 = _mm_add_epi32(
            _mm_add_epi32(
                _mm_madd_epi16(_mm_unpackhi_epi16(p1_hi, p5_hi), c15),
                _mm_madd_epi16(_mm_unpackhi_epi16(p2_hi, p4_hi), c24),
            ),
            _mm_madd_epi16(_mm_unpackhi_epi16(p3_hi, one16), c3),
        );

        if !FIVE_TAP {
            let p0_lo = _mm_unpacklo_epi8(p0, z);
            let p0_hi = _mm_unpackhi_epi8(p0, z);
            let p6_lo = _mm_unpacklo_epi8(p6, z);
            let p6_hi = _mm_unpackhi_epi8(p6, z);
            s0 = _mm_add_epi32(s0, _mm_madd_epi16(_mm_unpacklo_epi16(p0_lo, p6_lo), c06));
            s1 = _mm_add_epi32(s1, _mm_madd_epi16(_mm_unpackhi_epi16(p0_lo, p6_lo), c06));
            s2 = _mm_add_epi32(s2, _mm_madd_epi16(_mm_unpacklo_epi16(p0_hi, p6_hi), c06));
            s3 = _mm_add_epi32(s3, _mm_madd_epi16(_mm_unpackhi_epi16(p0_hi, p6_hi), c06));
        }

        s0 = _mm_srai_epi32::<3>(s0);
        s1 = _mm_srai_epi32::<3>(s1);
        s2 = _mm_srai_epi32::<3>(s2);
        s3 = _mm_srai_epi32::<3>(s3);

        let r_lo = pack32_to_14b_sse2(s0, s1);
        let r_hi = pack32_to_14b_sse2(s2, s3);

        mem::store_i16x8(mem::chunk_mut::<i16, 8>(out, x), r_lo);
        mem::store_i16x8(mem::chunk_mut::<i16, 8>(out, x + 8), r_hi);
        x += 16;
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn wiener_filter_v_row_ssse3<const FIVE_TAP: bool>(
    dst: &mut [u8],
    c: &LrCtx<'_>,
    p_row: usize,
    r0: &[i16; HOR_STRIDE],
    r1: &[i16; HOR_STRIDE],
    r2: &[i16; HOR_STRIDE],
    r3: &[i16; HOR_STRIDE],
    r4: &[i16; HOR_STRIDE],
    r5: &[i16; HOR_STRIDE],
    r6: &[i16; HOR_STRIDE],
    fv: &[i16; 8],
) {
    let z = _mm_setzero_si128();
    let c06 = _mm_set1_epi32((i32::from(fv[0]) & 0xffff) | ((i32::from(fv[6]) & 0xffff) << 16));
    let c15 = _mm_set1_epi32((i32::from(fv[1]) & 0xffff) | ((i32::from(fv[5]) & 0xffff) << 16));
    let c24 = _mm_set1_epi32((i32::from(fv[2]) & 0xffff) | ((i32::from(fv[4]) & 0xffff) << 16));
    let c3 = _mm_set1_epi32(i32::from(fv[3]) & 0xffff);
    let bias = _mm_set1_epi32(-261120);

    let mut out_row = [0u8; HOR_STRIDE];
    let mut x = 0usize;
    while x < c.w {
        let sum_8 = |off: usize| -> (__m128i, __m128i) {
            let v1 = mem::load_i16x8(mem::chunk::<i16, 8>(r1, off));
            let v2 = mem::load_i16x8(mem::chunk::<i16, 8>(r2, off));
            let v3 = mem::load_i16x8(mem::chunk::<i16, 8>(r3, off));
            let v4 = mem::load_i16x8(mem::chunk::<i16, 8>(r4, off));
            let v5 = mem::load_i16x8(mem::chunk::<i16, 8>(r5, off));

            let mut s_lo = _mm_add_epi32(
                _mm_add_epi32(bias, _mm_madd_epi16(_mm_unpacklo_epi16(v1, v5), c15)),
                _mm_add_epi32(
                    _mm_madd_epi16(_mm_unpacklo_epi16(v2, v4), c24),
                    _mm_madd_epi16(_mm_unpacklo_epi16(v3, z), c3),
                ),
            );
            let mut s_hi = _mm_add_epi32(
                _mm_add_epi32(bias, _mm_madd_epi16(_mm_unpackhi_epi16(v1, v5), c15)),
                _mm_add_epi32(
                    _mm_madd_epi16(_mm_unpackhi_epi16(v2, v4), c24),
                    _mm_madd_epi16(_mm_unpackhi_epi16(v3, z), c3),
                ),
            );

            if !FIVE_TAP {
                let v0 = mem::load_i16x8(mem::chunk::<i16, 8>(r0, off));
                let v6 = mem::load_i16x8(mem::chunk::<i16, 8>(r6, off));
                s_lo = _mm_add_epi32(s_lo, _mm_madd_epi16(_mm_unpacklo_epi16(v0, v6), c06));
                s_hi = _mm_add_epi32(s_hi, _mm_madd_epi16(_mm_unpackhi_epi16(v0, v6), c06));
            }

            (_mm_srai_epi32::<11>(s_lo), _mm_srai_epi32::<11>(s_hi))
        };

        let (s0, s1) = sum_8(x);
        let (s2, s3) = sum_8(x + 8);
        let bytes = pack32_to_u8x16_sse2(s0, s1, s2, s3);
        mem::store_u8x16(mem::chunk_mut::<u8, 16>(&mut out_row, x), bytes);
        x += 16;
    }

    let p = p_row * c.stride;
    dst[p..p + c.w].copy_from_slice(&out_row[..c.w]);
}

struct WienerSsse3State {
    hor: [[i16; HOR_STRIDE]; 6],
    tmp: [i16; HOR_STRIDE],
    ptrs: [usize; 7],
}

impl WienerSsse3State {
    #[target_feature(enable = "ssse3")]
    #[inline]
    fn filter_h<const FIVE_TAP: bool>(
        &mut self,
        c: &LrCtx<'_>,
        dst: &[u8],
        row: RowRef,
        r: usize,
        fh: &[i16; 8],
    ) {
        wiener_filter_h_ssse3::<FIVE_TAP>(&mut self.hor[r], c, dst, row, fh);
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    fn filter_v<const FIVE_TAP: bool>(
        &mut self,
        dst: &mut [u8],
        c: &LrCtx<'_>,
        p_row: usize,
        fv: &[i16; 8],
    ) {
        wiener_filter_v_row_ssse3::<FIVE_TAP>(
            dst,
            c,
            p_row,
            &self.hor[self.ptrs[0]],
            &self.hor[self.ptrs[1]],
            &self.hor[self.ptrs[2]],
            &self.hor[self.ptrs[3]],
            &self.hor[self.ptrs[4]],
            &self.hor[self.ptrs[5]],
            &self.hor[self.ptrs[5]],
            fv,
        );
        self.ptrs.copy_within(1..6, 0);
    }

    #[target_feature(enable = "ssse3")]
    #[inline]
    fn filter_hv<const FIVE_TAP: bool>(
        &mut self,
        dst: &mut [u8],
        c: &LrCtx<'_>,
        p_row: usize,
        row: RowRef,
        filter: &[[i16; 8]; 2],
    ) {
        wiener_filter_h_ssse3::<FIVE_TAP>(&mut self.tmp, c, dst, row, &filter[0]);
        wiener_filter_v_row_ssse3::<FIVE_TAP>(
            dst,
            c,
            p_row,
            &self.hor[self.ptrs[0]],
            &self.hor[self.ptrs[1]],
            &self.hor[self.ptrs[2]],
            &self.hor[self.ptrs[3]],
            &self.hor[self.ptrs[4]],
            &self.hor[self.ptrs[5]],
            &self.tmp,
            &filter[1],
        );
        let dst_row = self.ptrs[6];
        let w = c.w;
        self.hor[dst_row][..w].copy_from_slice(&self.tmp[..w]);
        self.ptrs.copy_within(1..7, 0);
        self.ptrs[6] = self.ptrs[0];
    }
}

#[target_feature(enable = "ssse3")]
fn wiener_ssse3_giris<const FIVE_TAP: bool>(
    dst: &mut [u8],
    c: &LrCtx<'_>,
    h: usize,
    filter: &[[i16; 8]; 2],
) {
    let fh = &filter[0];
    let fv = &filter[1];
    let mut st = WienerSsse3State {
        hor: [[0; HOR_STRIDE]; 6],
        tmp: [0; HOR_STRIDE],
        ptrs: [0; 7],
    };
    let mut h = h;
    let mut src_row = 0usize;
    let mut p_row = 0usize;

    let tail: usize = 'akis: {
        if c.edges.contains(LrEdgeFlags::HAVE_TOP) {
            st.ptrs[..6].copy_from_slice(&[0, 0, 1, 2, 2, 2]);
            st.filter_h::<FIVE_TAP>(c, dst, RowRef::Lpf(0), 0, fh);
            st.filter_h::<FIVE_TAP>(c, dst, RowRef::Lpf(1), 1, fh);
            st.filter_h::<FIVE_TAP>(c, dst, RowRef::Main(src_row), 2, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 1;
            }
            st.ptrs[4] = 3;
            st.ptrs[5] = 3;
            st.filter_h::<FIVE_TAP>(c, dst, RowRef::Main(src_row), 3, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 2;
            }
            st.ptrs[5] = 4;
            st.filter_h::<FIVE_TAP>(c, dst, RowRef::Main(src_row), 4, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
        } else {
            st.ptrs[..6].copy_from_slice(&[0; 6]);
            st.filter_h::<FIVE_TAP>(c, dst, RowRef::Main(src_row), 0, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 1;
            }
            st.ptrs[4] = 1;
            st.ptrs[5] = 1;
            st.filter_h::<FIVE_TAP>(c, dst, RowRef::Main(src_row), 1, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 2;
            }
            st.ptrs[5] = 2;
            st.filter_h::<FIVE_TAP>(c, dst, RowRef::Main(src_row), 2, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
            st.ptrs[6] = 3;
            st.filter_hv::<FIVE_TAP>(dst, c, p_row, RowRef::Main(src_row), filter);
            src_row += 1;
            p_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
            st.ptrs[6] = 4;
            st.filter_hv::<FIVE_TAP>(dst, c, p_row, RowRef::Main(src_row), filter);
            src_row += 1;
            p_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
        }

        st.ptrs[6] = st.ptrs[5] + 1;
        loop {
            st.filter_hv::<FIVE_TAP>(dst, c, p_row, RowRef::Main(src_row), filter);
            src_row += 1;
            p_row += 1;
            h -= 1;
            if h == 0 {
                break;
            }
        }

        if !c.edges.contains(LrEdgeFlags::HAVE_BOTTOM) {
            break 'akis 3;
        }
        st.filter_hv::<FIVE_TAP>(dst, c, p_row, RowRef::Lpf(6), filter);
        p_row += 1;
        st.filter_hv::<FIVE_TAP>(dst, c, p_row, RowRef::Lpf(7), filter);
        p_row += 1;
        1
    };

    for _ in 0..tail {
        st.filter_v::<FIVE_TAP>(dst, c, p_row, fv);
        p_row += 1;
    }
}

// ---------------------------------------------------------------------------
// SGR SSSE3 (`box3_row_h`, `box5_row_h`, `box3_vert`, `box5_vert`, `finish`, `weighted`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
#[inline]
fn box3_row_h_ssse3(
    sumsq: &mut [i32; SGR_BUF_STRIDE],
    sum: &mut [i32; SGR_BUF_STRIDE],
    pad: &[u8; PAD_ROW_LEN],
    w: usize,
) {
    let z = _mm_setzero_si128();
    let mut j = 0usize;
    while j < w + 2 {
        let a8 = mem::load_u8x16(mem::chunk::<u8, 16>(pad, j + 1));
        let b8 = mem::load_u8x16(mem::chunk::<u8, 16>(pad, j + 2));
        let c8 = mem::load_u8x16(mem::chunk::<u8, 16>(pad, j + 3));

        let mut proc8 = |a: __m128i, b: __m128i, c: __m128i, off: usize| {
            let s16 = _mm_add_epi16(_mm_add_epi16(a, b), c);
            mem::store_i32x4(
                mem::chunk_mut::<i32, 4>(sum, off),
                _mm_unpacklo_epi16(s16, z),
            );
            mem::store_i32x4(
                mem::chunk_mut::<i32, 4>(sum, off + 4),
                _mm_unpackhi_epi16(s16, z),
            );

            let ab_lo = _mm_unpacklo_epi16(a, b);
            let ab_hi = _mm_unpackhi_epi16(a, b);
            let c0_lo = _mm_unpacklo_epi16(c, z);
            let c0_hi = _mm_unpackhi_epi16(c, z);
            let sq_lo = _mm_add_epi32(_mm_madd_epi16(ab_lo, ab_lo), _mm_madd_epi16(c0_lo, c0_lo));
            let sq_hi = _mm_add_epi32(_mm_madd_epi16(ab_hi, ab_hi), _mm_madd_epi16(c0_hi, c0_hi));
            mem::store_i32x4(mem::chunk_mut::<i32, 4>(sumsq, off), sq_lo);
            mem::store_i32x4(mem::chunk_mut::<i32, 4>(sumsq, off + 4), sq_hi);
        };

        proc8(
            _mm_unpacklo_epi8(a8, z),
            _mm_unpacklo_epi8(b8, z),
            _mm_unpacklo_epi8(c8, z),
            j,
        );
        proc8(
            _mm_unpackhi_epi8(a8, z),
            _mm_unpackhi_epi8(b8, z),
            _mm_unpackhi_epi8(c8, z),
            j + 8,
        );
        j += 16;
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn box5_row_h_ssse3(
    sumsq: &mut [i32; SGR_BUF_STRIDE],
    sum: &mut [i32; SGR_BUF_STRIDE],
    pad: &[u8; PAD_ROW_LEN],
    w: usize,
) {
    let z = _mm_setzero_si128();
    let mut j = 0usize;
    while j < w + 2 {
        let a8 = mem::load_u8x16(mem::chunk::<u8, 16>(pad, j));
        let b8 = mem::load_u8x16(mem::chunk::<u8, 16>(pad, j + 1));
        let c8 = mem::load_u8x16(mem::chunk::<u8, 16>(pad, j + 2));
        let d8 = mem::load_u8x16(mem::chunk::<u8, 16>(pad, j + 3));
        let e8 = mem::load_u8x16(mem::chunk::<u8, 16>(pad, j + 4));

        let mut proc8 = |a: __m128i, b: __m128i, c: __m128i, d: __m128i, e: __m128i, off: usize| {
            let s16 = _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(a, b), _mm_add_epi16(c, d)), e);
            mem::store_i32x4(
                mem::chunk_mut::<i32, 4>(sum, off),
                _mm_unpacklo_epi16(s16, z),
            );
            mem::store_i32x4(
                mem::chunk_mut::<i32, 4>(sum, off + 4),
                _mm_unpackhi_epi16(s16, z),
            );

            let ab_lo = _mm_unpacklo_epi16(a, b);
            let ab_hi = _mm_unpackhi_epi16(a, b);
            let cd_lo = _mm_unpacklo_epi16(c, d);
            let cd_hi = _mm_unpackhi_epi16(c, d);
            let e0_lo = _mm_unpacklo_epi16(e, z);
            let e0_hi = _mm_unpackhi_epi16(e, z);
            let sq_lo = _mm_add_epi32(
                _mm_add_epi32(_mm_madd_epi16(ab_lo, ab_lo), _mm_madd_epi16(cd_lo, cd_lo)),
                _mm_madd_epi16(e0_lo, e0_lo),
            );
            let sq_hi = _mm_add_epi32(
                _mm_add_epi32(_mm_madd_epi16(ab_hi, ab_hi), _mm_madd_epi16(cd_hi, cd_hi)),
                _mm_madd_epi16(e0_hi, e0_hi),
            );
            mem::store_i32x4(mem::chunk_mut::<i32, 4>(sumsq, off), sq_lo);
            mem::store_i32x4(mem::chunk_mut::<i32, 4>(sumsq, off + 4), sq_hi);
        };

        proc8(
            _mm_unpacklo_epi8(a8, z),
            _mm_unpacklo_epi8(b8, z),
            _mm_unpacklo_epi8(c8, z),
            _mm_unpacklo_epi8(d8, z),
            _mm_unpacklo_epi8(e8, z),
            j,
        );
        proc8(
            _mm_unpackhi_epi8(a8, z),
            _mm_unpackhi_epi8(b8, z),
            _mm_unpackhi_epi8(c8, z),
            _mm_unpackhi_epi8(d8, z),
            _mm_unpackhi_epi8(e8, z),
            j + 8,
        );
        j += 16;
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
pub(crate) fn calc_row_ab_8bpc(
    aa: &mut [i32; SGR_BUF_STRIDE],
    bb: &mut [i32; SGR_BUF_STRIDE],
    w: usize,
    s: u32,
    n: i32,
    one_by_x: u32,
) {
    for i in 0..w + 2 {
        let a = aa[i];
        let b = bb[i];
        let p = (a * n - b * b).max(0) as u32;
        let z = p.wrapping_mul(s).wrapping_add(1 << 19) >> 20;
        let x = u32::from(SGR_X_BY_X[z.min(255) as usize]);
        aa[i] = (x
            .wrapping_mul(b as u32)
            .wrapping_mul(one_by_x)
            .wrapping_add(1 << 11)
            >> 12) as i32;
        bb[i] = x as i32;
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn box3_vert_ssse3<const K: usize>(
    sums: &SgrRows<3>,
    sp: &mut [usize; 3],
    ab: &mut SgrRows<K>,
    out: usize,
    w: usize,
    s: u32,
) {
    let (oa, ob) = (&mut ab.sq[out], &mut ab.s[out]);
    let (q0, q1, q2) = (&sums.sq[sp[0]], &sums.sq[sp[1]], &sums.sq[sp[2]]);
    let (s0, s1, s2) = (&sums.s[sp[0]], &sums.s[sp[1]], &sums.s[sp[2]]);
    let mut x = 0usize;
    while x < w + 2 {
        let vq = _mm_add_epi32(
            _mm_add_epi32(
                mem::load_i32x4(mem::chunk::<i32, 4>(q0, x)),
                mem::load_i32x4(mem::chunk::<i32, 4>(q1, x)),
            ),
            mem::load_i32x4(mem::chunk::<i32, 4>(q2, x)),
        );
        let vs = _mm_add_epi32(
            _mm_add_epi32(
                mem::load_i32x4(mem::chunk::<i32, 4>(s0, x)),
                mem::load_i32x4(mem::chunk::<i32, 4>(s1, x)),
            ),
            mem::load_i32x4(mem::chunk::<i32, 4>(s2, x)),
        );
        mem::store_i32x4(mem::chunk_mut::<i32, 4>(oa, x), vq);
        mem::store_i32x4(mem::chunk_mut::<i32, 4>(ob, x), vs);
        x += 4;
    }
    calc_row_ab_8bpc(oa, ob, w, s, 9, 455);
    sp.rotate_left(1);
}

#[target_feature(enable = "ssse3")]
#[inline]
fn box5_vert_ssse3(
    sums: &SgrRows<5>,
    sp: &mut [usize; 5],
    ab: &mut SgrRows<2>,
    out: usize,
    w: usize,
    s: u32,
) {
    let (oa, ob) = (&mut ab.sq[out], &mut ab.s[out]);
    let q = [
        &sums.sq[sp[0]],
        &sums.sq[sp[1]],
        &sums.sq[sp[2]],
        &sums.sq[sp[3]],
        &sums.sq[sp[4]],
    ];
    let t = [
        &sums.s[sp[0]],
        &sums.s[sp[1]],
        &sums.s[sp[2]],
        &sums.s[sp[3]],
        &sums.s[sp[4]],
    ];
    let mut x = 0usize;
    while x < w + 2 {
        let vq = _mm_add_epi32(
            _mm_add_epi32(
                _mm_add_epi32(
                    mem::load_i32x4(mem::chunk::<i32, 4>(q[0], x)),
                    mem::load_i32x4(mem::chunk::<i32, 4>(q[1], x)),
                ),
                _mm_add_epi32(
                    mem::load_i32x4(mem::chunk::<i32, 4>(q[2], x)),
                    mem::load_i32x4(mem::chunk::<i32, 4>(q[3], x)),
                ),
            ),
            mem::load_i32x4(mem::chunk::<i32, 4>(q[4], x)),
        );
        let vs = _mm_add_epi32(
            _mm_add_epi32(
                _mm_add_epi32(
                    mem::load_i32x4(mem::chunk::<i32, 4>(t[0], x)),
                    mem::load_i32x4(mem::chunk::<i32, 4>(t[1], x)),
                ),
                _mm_add_epi32(
                    mem::load_i32x4(mem::chunk::<i32, 4>(t[2], x)),
                    mem::load_i32x4(mem::chunk::<i32, 4>(t[3], x)),
                ),
            ),
            mem::load_i32x4(mem::chunk::<i32, 4>(t[4], x)),
        );
        mem::store_i32x4(mem::chunk_mut::<i32, 4>(oa, x), vq);
        mem::store_i32x4(mem::chunk_mut::<i32, 4>(ob, x), vs);
        x += 4;
    }
    calc_row_ab_8bpc(oa, ob, w, s, 25, 164);
    sp.rotate_left(2);
}

#[target_feature(enable = "ssse3")]
#[inline]
fn mul3_epi32(v: __m128i) -> __m128i {
    _mm_add_epi32(_mm_slli_epi32::<1>(v), v)
}

#[target_feature(enable = "ssse3")]
#[inline]
fn mul5_epi32(v: __m128i) -> __m128i {
    _mm_add_epi32(_mm_slli_epi32::<2>(v), v)
}

#[target_feature(enable = "ssse3")]
#[inline]
fn mul6_epi32(v: __m128i) -> __m128i {
    _mm_add_epi32(_mm_slli_epi32::<2>(v), _mm_slli_epi32::<1>(v))
}

#[target_feature(enable = "ssse3")]
#[inline]
fn finish_filter_row1_ssse3<const K: usize>(
    tmp: &mut [i32],
    dst: &[u8],
    row_off: usize,
    ab: &SgrRows<K>,
    ap: [usize; 3],
    w: usize,
) {
    let (b0, b1, b2) = (&ab.s[ap[0]], &ab.s[ap[1]], &ab.s[ap[2]]);
    let (a0, a1, a2) = (&ab.sq[ap[0]], &ab.sq[ap[1]], &ab.sq[ap[2]]);
    let z = _mm_setzero_si128();
    let rnd = _mm_set1_epi32(1 << 8);
    let mut src_row = [0u8; HOR_STRIDE];
    src_row[..w].copy_from_slice(&dst[row_off..row_off + w]);

    let mut i = 0usize;
    while i < w {
        let s8 = mem::load_u8x16(mem::chunk::<u8, 16>(&src_row, i));
        let s16_lo = _mm_unpacklo_epi8(s8, z);
        let s16_hi = _mm_unpackhi_epi8(s8, z);
        let s32 = [
            _mm_unpacklo_epi16(s16_lo, z),
            _mm_unpackhi_epi16(s16_lo, z),
            _mm_unpacklo_epi16(s16_hi, z),
            _mm_unpackhi_epi16(s16_hi, z),
        ];

        for (q, &src_v) in s32.iter().enumerate() {
            let col = i + q * 4;
            let j = col + 1;
            let b_cross = _mm_add_epi32(
                _mm_add_epi32(
                    mem::load_i32x4(mem::chunk::<i32, 4>(b1, j)),
                    mem::load_i32x4(mem::chunk::<i32, 4>(b1, j - 1)),
                ),
                _mm_add_epi32(
                    mem::load_i32x4(mem::chunk::<i32, 4>(b1, j + 1)),
                    _mm_add_epi32(
                        mem::load_i32x4(mem::chunk::<i32, 4>(b0, j)),
                        mem::load_i32x4(mem::chunk::<i32, 4>(b2, j)),
                    ),
                ),
            );
            let b_diag = _mm_add_epi32(
                _mm_add_epi32(
                    mem::load_i32x4(mem::chunk::<i32, 4>(b0, j - 1)),
                    mem::load_i32x4(mem::chunk::<i32, 4>(b2, j - 1)),
                ),
                _mm_add_epi32(
                    mem::load_i32x4(mem::chunk::<i32, 4>(b0, j + 1)),
                    mem::load_i32x4(mem::chunk::<i32, 4>(b2, j + 1)),
                ),
            );
            // `a` in `[0, 32 * 255 = 8160]` fits in positive signed `i16`, `src_v` in `[0, 255]`,
            // so `_mm_madd_epi16(a, src_v)` computes exact `32-bit` product `a * src_v` in SSE2!
            let a_v = _mm_add_epi32(_mm_slli_epi32::<2>(b_cross), mul3_epi32(b_diag));

            let a_cross = _mm_add_epi32(
                _mm_add_epi32(
                    mem::load_i32x4(mem::chunk::<i32, 4>(a1, j)),
                    mem::load_i32x4(mem::chunk::<i32, 4>(a1, j - 1)),
                ),
                _mm_add_epi32(
                    mem::load_i32x4(mem::chunk::<i32, 4>(a1, j + 1)),
                    _mm_add_epi32(
                        mem::load_i32x4(mem::chunk::<i32, 4>(a0, j)),
                        mem::load_i32x4(mem::chunk::<i32, 4>(a2, j)),
                    ),
                ),
            );
            let a_diag = _mm_add_epi32(
                _mm_add_epi32(
                    mem::load_i32x4(mem::chunk::<i32, 4>(a0, j - 1)),
                    mem::load_i32x4(mem::chunk::<i32, 4>(a2, j - 1)),
                ),
                _mm_add_epi32(
                    mem::load_i32x4(mem::chunk::<i32, 4>(a0, j + 1)),
                    mem::load_i32x4(mem::chunk::<i32, 4>(a2, j + 1)),
                ),
            );
            let b_v = _mm_add_epi32(_mm_slli_epi32::<2>(a_cross), mul3_epi32(a_diag));

            let prod = _mm_madd_epi16(a_v, src_v);
            let res = _mm_srai_epi32::<9>(_mm_add_epi32(_mm_sub_epi32(b_v, prod), rnd));
            mem::store_i32x4(mem::chunk_mut::<i32, 4>(tmp, col), res);
        }
        i += 16;
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn finish_filter2_ssse3(
    tmp: &mut [i32; 2 * SGR_OUT_STRIDE],
    dst: &[u8],
    r0_off: usize,
    r1_off: usize,
    ab: &SgrRows<2>,
    ap: [usize; 2],
    w: usize,
    h: usize,
) {
    let (b0, b1) = (&ab.s[ap[0]], &ab.s[ap[1]]);
    let (a0, a1) = (&ab.sq[ap[0]], &ab.sq[ap[1]]);
    let z = _mm_setzero_si128();
    let rnd0 = _mm_set1_epi32(1 << 8);
    let mut src0_row = [0u8; HOR_STRIDE];
    src0_row[..w].copy_from_slice(&dst[r0_off..r0_off + w]);

    let mut i = 0usize;
    while i < w {
        let s8 = mem::load_u8x16(mem::chunk::<u8, 16>(&src0_row, i));
        let s16_lo = _mm_unpacklo_epi8(s8, z);
        let s16_hi = _mm_unpackhi_epi8(s8, z);
        let s32 = [
            _mm_unpacklo_epi16(s16_lo, z),
            _mm_unpackhi_epi16(s16_lo, z),
            _mm_unpacklo_epi16(s16_hi, z),
            _mm_unpackhi_epi16(s16_hi, z),
        ];

        for (q, &src_v) in s32.iter().enumerate() {
            let col = i + q * 4;
            let j = col + 1;
            let b_mid = _mm_add_epi32(
                mem::load_i32x4(mem::chunk::<i32, 4>(b0, j)),
                mem::load_i32x4(mem::chunk::<i32, 4>(b1, j)),
            );
            let b_side = _mm_add_epi32(
                _mm_add_epi32(
                    mem::load_i32x4(mem::chunk::<i32, 4>(b0, j - 1)),
                    mem::load_i32x4(mem::chunk::<i32, 4>(b1, j - 1)),
                ),
                _mm_add_epi32(
                    mem::load_i32x4(mem::chunk::<i32, 4>(b0, j + 1)),
                    mem::load_i32x4(mem::chunk::<i32, 4>(b1, j + 1)),
                ),
            );
            let a_v = _mm_add_epi32(mul6_epi32(b_mid), mul5_epi32(b_side));

            let a_mid = _mm_add_epi32(
                mem::load_i32x4(mem::chunk::<i32, 4>(a0, j)),
                mem::load_i32x4(mem::chunk::<i32, 4>(a1, j)),
            );
            let a_side = _mm_add_epi32(
                _mm_add_epi32(
                    mem::load_i32x4(mem::chunk::<i32, 4>(a0, j - 1)),
                    mem::load_i32x4(mem::chunk::<i32, 4>(a1, j - 1)),
                ),
                _mm_add_epi32(
                    mem::load_i32x4(mem::chunk::<i32, 4>(a0, j + 1)),
                    mem::load_i32x4(mem::chunk::<i32, 4>(a1, j + 1)),
                ),
            );
            let b_v = _mm_add_epi32(mul6_epi32(a_mid), mul5_epi32(a_side));

            let prod = _mm_madd_epi16(a_v, src_v);
            let res = _mm_srai_epi32::<9>(_mm_add_epi32(_mm_sub_epi32(b_v, prod), rnd0));
            mem::store_i32x4(mem::chunk_mut::<i32, 4>(tmp, col), res);
        }
        i += 16;
    }

    if h <= 1 {
        return;
    }

    let rnd1 = _mm_set1_epi32(1 << 7);
    let mut src1_row = [0u8; HOR_STRIDE];
    src1_row[..w].copy_from_slice(&dst[r1_off..r1_off + w]);

    let mut i = 0usize;
    while i < w {
        let s8 = mem::load_u8x16(mem::chunk::<u8, 16>(&src1_row, i));
        let s16_lo = _mm_unpacklo_epi8(s8, z);
        let s16_hi = _mm_unpackhi_epi8(s8, z);
        let s32 = [
            _mm_unpacklo_epi16(s16_lo, z),
            _mm_unpackhi_epi16(s16_lo, z),
            _mm_unpacklo_epi16(s16_hi, z),
            _mm_unpackhi_epi16(s16_hi, z),
        ];

        for (q, &src_v) in s32.iter().enumerate() {
            let col = i + q * 4;
            let j = col + 1;
            let b_mid = mem::load_i32x4(mem::chunk::<i32, 4>(b1, j));
            let b_side = _mm_add_epi32(
                mem::load_i32x4(mem::chunk::<i32, 4>(b1, j - 1)),
                mem::load_i32x4(mem::chunk::<i32, 4>(b1, j + 1)),
            );
            let a_v = _mm_add_epi32(mul6_epi32(b_mid), mul5_epi32(b_side));

            let a_mid = mem::load_i32x4(mem::chunk::<i32, 4>(a1, j));
            let a_side = _mm_add_epi32(
                mem::load_i32x4(mem::chunk::<i32, 4>(a1, j - 1)),
                mem::load_i32x4(mem::chunk::<i32, 4>(a1, j + 1)),
            );
            let b_v = _mm_add_epi32(mul6_epi32(a_mid), mul5_epi32(a_side));

            let prod = _mm_madd_epi16(a_v, src_v);
            let res = _mm_srai_epi32::<8>(_mm_add_epi32(_mm_sub_epi32(b_v, prod), rnd1));
            mem::store_i32x4(mem::chunk_mut::<i32, 4>(tmp, SGR_OUT_STRIDE + col), res);
        }
        i += 16;
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn weighted_row1_ssse3(dst: &mut [u8], off: usize, t1: &[i32], w: usize, w1: i32) {
    let z = _mm_setzero_si128();
    let w1_v = _mm_set1_epi32(w1 & 0xffff);
    let rnd = _mm_set1_epi32(1 << 10);
    let mut io_row = [0u8; HOR_STRIDE];
    io_row[..w].copy_from_slice(&dst[off..off + w]);

    let mut i = 0usize;
    while i < w {
        let d8 = mem::load_u8x16(mem::chunk::<u8, 16>(&io_row, i));
        let d16_lo = _mm_unpacklo_epi8(d8, z);
        let d16_hi = _mm_unpackhi_epi8(d8, z);

        let t_0 = _mm_packs_epi32(
            mem::load_i32x4(mem::chunk::<i32, 4>(t1, i)),
            mem::load_i32x4(mem::chunk::<i32, 4>(t1, i + 4)),
        );
        let t_1 = _mm_packs_epi32(
            mem::load_i32x4(mem::chunk::<i32, 4>(t1, i + 8)),
            mem::load_i32x4(mem::chunk::<i32, 4>(t1, i + 12)),
        );

        let v0 = _mm_srai_epi32::<11>(_mm_add_epi32(
            _mm_madd_epi16(_mm_unpacklo_epi16(t_0, z), w1_v),
            rnd,
        ));
        let v1 = _mm_srai_epi32::<11>(_mm_add_epi32(
            _mm_madd_epi16(_mm_unpackhi_epi16(t_0, z), w1_v),
            rnd,
        ));
        let v2 = _mm_srai_epi32::<11>(_mm_add_epi32(
            _mm_madd_epi16(_mm_unpacklo_epi16(t_1, z), w1_v),
            rnd,
        ));
        let v3 = _mm_srai_epi32::<11>(_mm_add_epi32(
            _mm_madd_epi16(_mm_unpackhi_epi16(t_1, z), w1_v),
            rnd,
        ));

        let out_lo = _mm_add_epi16(d16_lo, _mm_packs_epi32(v0, v1));
        let out_hi = _mm_add_epi16(d16_hi, _mm_packs_epi32(v2, v3));
        mem::store_u8x16(
            mem::chunk_mut::<u8, 16>(&mut io_row, i),
            _mm_packus_epi16(out_lo, out_hi),
        );
        i += 16;
    }

    dst[off..off + w].copy_from_slice(&io_row[..w]);
}

#[target_feature(enable = "ssse3")]
#[inline]
fn weighted2_ssse3(
    dst: &mut [u8],
    off: usize,
    stride: usize,
    t1: &[i32; 2 * SGR_OUT_STRIDE],
    t2: &[i32; 2 * SGR_OUT_STRIDE],
    w: usize,
    h: usize,
    w0: i32,
    w1: i32,
) {
    let z = _mm_setzero_si128();
    let w01_v = _mm_set1_epi32((w0 & 0xffff) | (w1 << 16));
    let rnd = _mm_set1_epi32(1 << 10);

    for j in 0..h {
        let row = off + j * stride;
        let t = j * SGR_OUT_STRIDE;
        let mut io_row = [0u8; HOR_STRIDE];
        io_row[..w].copy_from_slice(&dst[row..row + w]);

        let mut i = 0usize;
        while i < w {
            let d8 = mem::load_u8x16(mem::chunk::<u8, 16>(&io_row, i));
            let d16_lo = _mm_unpacklo_epi8(d8, z);
            let d16_hi = _mm_unpackhi_epi8(d8, z);

            let t1_0 = _mm_packs_epi32(
                mem::load_i32x4(mem::chunk::<i32, 4>(t1, t + i)),
                mem::load_i32x4(mem::chunk::<i32, 4>(t1, t + i + 4)),
            );
            let t1_1 = _mm_packs_epi32(
                mem::load_i32x4(mem::chunk::<i32, 4>(t1, t + i + 8)),
                mem::load_i32x4(mem::chunk::<i32, 4>(t1, t + i + 12)),
            );
            let t2_0 = _mm_packs_epi32(
                mem::load_i32x4(mem::chunk::<i32, 4>(t2, t + i)),
                mem::load_i32x4(mem::chunk::<i32, 4>(t2, t + i + 4)),
            );
            let t2_1 = _mm_packs_epi32(
                mem::load_i32x4(mem::chunk::<i32, 4>(t2, t + i + 8)),
                mem::load_i32x4(mem::chunk::<i32, 4>(t2, t + i + 12)),
            );

            let v0 = _mm_srai_epi32::<11>(_mm_add_epi32(
                _mm_madd_epi16(_mm_unpacklo_epi16(t1_0, t2_0), w01_v),
                rnd,
            ));
            let v1 = _mm_srai_epi32::<11>(_mm_add_epi32(
                _mm_madd_epi16(_mm_unpackhi_epi16(t1_0, t2_0), w01_v),
                rnd,
            ));
            let v2 = _mm_srai_epi32::<11>(_mm_add_epi32(
                _mm_madd_epi16(_mm_unpacklo_epi16(t1_1, t2_1), w01_v),
                rnd,
            ));
            let v3 = _mm_srai_epi32::<11>(_mm_add_epi32(
                _mm_madd_epi16(_mm_unpackhi_epi16(t1_1, t2_1), w01_v),
                rnd,
            ));

            let out_lo = _mm_add_epi16(d16_lo, _mm_packs_epi32(v0, v1));
            let out_hi = _mm_add_epi16(d16_hi, _mm_packs_epi32(v2, v3));
            mem::store_u8x16(
                mem::chunk_mut::<u8, 16>(&mut io_row, i),
                _mm_packus_epi16(out_lo, out_hi),
            );
            i += 16;
        }

        dst[row..row + w].copy_from_slice(&io_row[..w]);
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn finish1_ssse3(
    dst: &mut [u8],
    c: &LrCtx<'_>,
    dst_row: &mut usize,
    ab: &SgrRows<3>,
    ap: &mut [usize; 3],
    tmp: &mut [i32; 2 * SGR_OUT_STRIDE],
    w1: i32,
) {
    let r = *dst_row;
    finish_filter_row1_ssse3::<3>(tmp, dst, r * c.stride, ab, *ap, c.w);
    weighted_row1_ssse3(dst, r * c.stride, &tmp[..], c.w, w1);
    *dst_row += 1;
    ap.rotate_left(1);
}

#[target_feature(enable = "ssse3")]
#[inline]
fn finish2_ssse3(
    dst: &mut [u8],
    c: &LrCtx<'_>,
    dst_row: &mut usize,
    ab: &SgrRows<2>,
    ap: &mut [usize; 2],
    tmp: &mut [i32; 2 * SGR_OUT_STRIDE],
    h: usize,
    w0: i32,
) {
    let r = *dst_row;
    let r1 = if h > 1 { r + 1 } else { r };
    finish_filter2_ssse3(tmp, dst, r * c.stride, r1 * c.stride, ab, *ap, c.w, h);
    weighted_row1_ssse3(dst, r * c.stride, &tmp[..], c.w, w0);
    *dst_row += 1;
    if h > 1 {
        weighted_row1_ssse3(dst, (r + 1) * c.stride, &tmp[SGR_OUT_STRIDE..], c.w, w0);
        *dst_row += 1;
    }
    ap.rotate_left(1);
}

#[target_feature(enable = "ssse3")]
#[inline]
fn finish_mix_ssse3(
    dst: &mut [u8],
    c: &LrCtx<'_>,
    dst_row: &mut usize,
    ab5: &SgrRows<2>,
    ap5: &mut [usize; 2],
    ab3: &SgrRows<4>,
    ap3: &mut [usize; 4],
    tmp5: &mut [i32; 2 * SGR_OUT_STRIDE],
    tmp3: &mut [i32; 2 * SGR_OUT_STRIDE],
    h: usize,
    w0: i32,
    w1: i32,
) {
    let r = *dst_row;
    let r1 = if h > 1 { r + 1 } else { r };
    finish_filter2_ssse3(tmp5, dst, r * c.stride, r1 * c.stride, ab5, *ap5, c.w, h);
    finish_filter_row1_ssse3::<4>(
        &mut tmp3[..],
        dst,
        r * c.stride,
        ab3,
        [ap3[0], ap3[1], ap3[2]],
        c.w,
    );
    if h > 1 {
        finish_filter_row1_ssse3::<4>(
            &mut tmp3[SGR_OUT_STRIDE..],
            dst,
            (r + 1) * c.stride,
            ab3,
            [ap3[1], ap3[2], ap3[3]],
            c.w,
        );
    }
    weighted2_ssse3(dst, r * c.stride, c.stride, tmp5, tmp3, c.w, h, w0, w1);
    *dst_row += h;
    ap5.rotate_left(1);
    ap3.rotate_left(1);
}

#[target_feature(enable = "ssse3")]
fn sgr_3x3_ssse3_giris(dst: &mut [u8], c: &LrCtx<'_>, h: usize, s1: u32, w1: i32) {
    enum Kuyruk {
        Bitti,
        Vert2,
        Vert1,
    }
    let w = c.w;
    let mut sums = SgrRows::<3>::new();
    let mut ab = SgrRows::<3>::new();
    let mut sp: [usize; 3];
    let mut ap = [0usize, 1, 2];
    let mut tmp = [0i32; 2 * SGR_OUT_STRIDE];
    let mut pad = [0u8; PAD_ROW_LEN];
    let mut h = h;
    let mut sr = 0usize;
    let mut dr = 0usize;

    macro_rules! row_h {
        ($row:expr, $r:expr) => {{
            pad_row_8bpc::<2>(&mut pad, c, dst, $row);
            box3_row_h_ssse3(&mut sums.sq[$r], &mut sums.s[$r], &pad, w);
        }};
    }

    let kuyruk = 'akis: {
        if c.edges.contains(LrEdgeFlags::HAVE_TOP) {
            sp = [0, 1, 2];
            row_h!(RowRef::Lpf(0), 0);
            row_h!(RowRef::Lpf(1), 1);
            row_h!(RowRef::Main(sr), sp[2]);
            box3_vert_ssse3::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            row_h!(RowRef::Main(sr), sp[2]);
            box3_vert_ssse3::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
        } else {
            sp = [0, 0, 0];
            row_h!(RowRef::Main(sr), 0);
            sr += 1;
            box3_vert_ssse3::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp[2] = 1;
            row_h!(RowRef::Main(sr), sp[2]);
            box3_vert_ssse3::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp[2] = 2;
        }

        loop {
            row_h!(RowRef::Main(sr), sp[2]);
            box3_vert_ssse3::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            finish1_ssse3(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
            h -= 1;
            if h == 0 {
                break;
            }
        }

        if !c.edges.contains(LrEdgeFlags::HAVE_BOTTOM) {
            break 'akis Kuyruk::Vert2;
        }
        row_h!(RowRef::Lpf(6), sp[2]);
        box3_vert_ssse3::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
        finish1_ssse3(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        row_h!(RowRef::Lpf(7), sp[2]);
        box3_vert_ssse3::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
        finish1_ssse3(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        Kuyruk::Bitti
    };

    match kuyruk {
        Kuyruk::Bitti => return,
        Kuyruk::Vert2 => {
            sp[2] = sp[1];
            box3_vert_ssse3::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            finish1_ssse3(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        }
        Kuyruk::Vert1 => {
            sp[2] = sp[1];
            box3_vert_ssse3::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            ap.rotate_left(1);
        }
    }
    sp[2] = sp[1];
    box3_vert_ssse3::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
    finish1_ssse3(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
}

#[target_feature(enable = "ssse3")]
fn sgr_5x5_ssse3_giris(dst: &mut [u8], c: &LrCtx<'_>, h: usize, s0: u32, w0: i32) {
    enum Kuyruk {
        Output2,
        Vert2,
        Odd,
        Vert1,
    }
    let w = c.w;
    let mut sums = SgrRows::<5>::new();
    let mut ab = SgrRows::<2>::new();
    let mut sp: [usize; 5];
    let mut ap = [0usize, 1];
    let mut tmp = [0i32; 2 * SGR_OUT_STRIDE];
    let mut pad = [0u8; PAD_ROW_LEN];
    let mut h = h;
    let mut sr = 0usize;
    let mut dr = 0usize;

    macro_rules! row_h {
        ($row:expr, $r:expr) => {{
            pad_row_8bpc::<3>(&mut pad, c, dst, $row);
            box5_row_h_ssse3(&mut sums.sq[$r], &mut sums.s[$r], &pad, w);
        }};
    }

    let kuyruk = 'akis: {
        if c.edges.contains(LrEdgeFlags::HAVE_TOP) {
            sp = [0, 0, 1, 2, 3];
            row_h!(RowRef::Lpf(0), 0);
            row_h!(RowRef::Lpf(1), 1);
            row_h!(RowRef::Main(sr), 2);
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            row_h!(RowRef::Main(sr), 3);
            sr += 1;
            box5_vert_ssse3(&sums, &mut sp, &mut ab, ap[1], w, s0);
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp[3] = 4;
        } else {
            sp = [0; 5];
            row_h!(RowRef::Main(sr), 0);
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp[4] = 1;
            row_h!(RowRef::Main(sr), 1);
            sr += 1;
            box5_vert_ssse3(&sums, &mut sp, &mut ab, ap[1], w, s0);
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp[3] = 2;
            sp[4] = 3;
            row_h!(RowRef::Main(sr), 2);
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            row_h!(RowRef::Main(sr), 3);
            sr += 1;
            box5_vert_ssse3(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2_ssse3(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp[3] = 4;
        }

        loop {
            row_h!(RowRef::Main(sr), sp[3]);
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            row_h!(RowRef::Main(sr), sp[4]);
            sr += 1;
            box5_vert_ssse3(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2_ssse3(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
            h -= 1;
            if h == 0 {
                break;
            }
        }

        if !c.edges.contains(LrEdgeFlags::HAVE_BOTTOM) {
            break 'akis Kuyruk::Vert2;
        }
        row_h!(RowRef::Lpf(6), sp[3]);
        row_h!(RowRef::Lpf(7), sp[4]);
        Kuyruk::Output2
    };

    let output_1 = match kuyruk {
        Kuyruk::Output2 => false,
        Kuyruk::Vert2 => {
            sp[3] = sp[2];
            sp[4] = sp[2];
            false
        }
        Kuyruk::Odd => {
            sp[4] = sp[3];
            box5_vert_ssse3(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2_ssse3(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
            true
        }
        Kuyruk::Vert1 => {
            sp[4] = sp[3];
            box5_vert_ssse3(&sums, &mut sp, &mut ab, ap[1], w, s0);
            ap.rotate_left(1);
            true
        }
    };
    if output_1 {
        sp[3] = sp[2];
        sp[4] = sp[2];
        box5_vert_ssse3(&sums, &mut sp, &mut ab, ap[1], w, s0);
        finish2_ssse3(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 1, w0);
    } else {
        box5_vert_ssse3(&sums, &mut sp, &mut ab, ap[1], w, s0);
        finish2_ssse3(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
    }
}

#[target_feature(enable = "ssse3")]
fn sgr_mix_ssse3_giris(
    dst: &mut [u8],
    c: &LrCtx<'_>,
    h: usize,
    s0: u32,
    s1: u32,
    w0: i32,
    w1: i32,
) {
    enum Kuyruk {
        Output2,
        Vert2,
        Odd,
        Vert1,
    }
    let w = c.w;
    let mut sums5 = SgrRows::<5>::new();
    let mut sums3 = SgrRows::<3>::new();
    let mut ab5 = SgrRows::<2>::new();
    let mut ab3 = SgrRows::<4>::new();
    let mut sp5: [usize; 5];
    let mut sp3: [usize; 3];
    let mut ap5 = [0usize, 1];
    let mut ap3 = [0usize, 1, 2, 3];
    let mut tmp5 = [0i32; 2 * SGR_OUT_STRIDE];
    let mut tmp3 = [0i32; 2 * SGR_OUT_STRIDE];
    let mut pad = [0u8; PAD_ROW_LEN];
    let mut h = h;
    let mut sr = 0usize;
    let mut dr = 0usize;

    macro_rules! row_h35 {
        ($row:expr, $r3:expr, $r5:expr) => {{
            pad_row_8bpc::<3>(&mut pad, c, dst, $row);
            box3_row_h_ssse3(&mut sums3.sq[$r3], &mut sums3.s[$r3], &pad, w);
            box5_row_h_ssse3(&mut sums5.sq[$r5], &mut sums5.s[$r5], &pad, w);
        }};
    }

    let kuyruk = 'akis: {
        if c.edges.contains(LrEdgeFlags::HAVE_TOP) {
            sp5 = [0, 0, 1, 2, 3];
            sp3 = [0, 1, 2];
            row_h35!(RowRef::Lpf(0), 0, 0);
            row_h35!(RowRef::Lpf(1), 1, 1);
            row_h35!(RowRef::Main(sr), 2, 2);
            sr += 1;
            box3_vert_ssse3::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            row_h35!(RowRef::Main(sr), sp3[2], 3);
            sr += 1;
            box5_vert_ssse3(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            ap5.rotate_left(1);
            box3_vert_ssse3::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp5[3] = 4;
        } else {
            sp5 = [0; 5];
            sp3 = [0; 3];
            row_h35!(RowRef::Main(sr), 0, 0);
            sr += 1;
            box3_vert_ssse3::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp5[4] = 1;
            sp3[2] = 1;
            row_h35!(RowRef::Main(sr), 1, 1);
            sr += 1;
            box5_vert_ssse3(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            ap5.rotate_left(1);
            box3_vert_ssse3::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp5[3] = 2;
            sp5[4] = 3;
            sp3[2] = 2;
            row_h35!(RowRef::Main(sr), 2, 2);
            sr += 1;
            box3_vert_ssse3::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            row_h35!(RowRef::Main(sr), sp3[2], 3);
            sr += 1;
            box5_vert_ssse3(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert_ssse3::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix_ssse3(
                dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 2, w0, w1,
            );
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp5[3] = 4;
        }

        loop {
            row_h35!(RowRef::Main(sr), sp3[2], sp5[3]);
            sr += 1;
            box3_vert_ssse3::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            row_h35!(RowRef::Main(sr), sp3[2], sp5[4]);
            sr += 1;
            box5_vert_ssse3(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert_ssse3::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix_ssse3(
                dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 2, w0, w1,
            );
            h -= 1;
            if h == 0 {
                break;
            }
        }

        if !c.edges.contains(LrEdgeFlags::HAVE_BOTTOM) {
            break 'akis Kuyruk::Vert2;
        }
        row_h35!(RowRef::Lpf(6), sp3[2], sp5[3]);
        box3_vert_ssse3::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
        ap3.rotate_left(1);
        row_h35!(RowRef::Lpf(7), sp3[2], sp5[4]);
        Kuyruk::Output2
    };

    let output_1 = match kuyruk {
        Kuyruk::Output2 => false,
        Kuyruk::Vert2 => {
            sp5[3] = sp5[2];
            sp5[4] = sp5[2];
            sp3[2] = sp3[1];
            box3_vert_ssse3::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            sp3[2] = sp3[1];
            false
        }
        Kuyruk::Odd => {
            sp5[4] = sp5[3];
            sp3[2] = sp3[1];
            box5_vert_ssse3(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert_ssse3::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix_ssse3(
                dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 2, w0, w1,
            );
            true
        }
        Kuyruk::Vert1 => {
            sp5[4] = sp5[3];
            sp3[2] = sp3[1];
            box5_vert_ssse3(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            ap5.rotate_left(1);
            box3_vert_ssse3::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            true
        }
    };
    if output_1 {
        sp5[3] = sp5[2];
        sp5[4] = sp5[2];
        sp3[2] = sp3[1];
        box5_vert_ssse3(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
        box3_vert_ssse3::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
        ap3.rotate_left(1);
        finish_mix_ssse3(
            dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 1, w0, w1,
        );
    } else {
        box5_vert_ssse3(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
        box3_vert_ssse3::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
        finish_mix_ssse3(
            dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 2, w0, w1,
        );
    }
}
