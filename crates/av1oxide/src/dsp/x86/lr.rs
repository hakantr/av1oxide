//! 8 bpc loop restoration (Wiener / SGRPROJ) x86_64 AVX2 çekirdekleri (`D02745`,
//! `D02752`, `D02727`, `D02733`, `D02739`; P46 Adım 3 / Paket A).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Kayıt | Yuva | Sembol | Kaynak |
//! |---|---|---|---|
//! | `D02745` | `wiener[0]` | `dav1d_wiener_filter7_8bpc_avx2` | `src/x86/looprestoration_avx2.asm:252-558`, `src/x86/looprestoration.h:73` |
//! | `D02752` | `wiener[1]` | `dav1d_wiener_filter5_8bpc_avx2` | `src/x86/looprestoration_avx2.asm:252-558`, `src/x86/looprestoration.h:74` |
//! | `D02727` | `sgr[0]` | `dav1d_sgr_filter_5x5_8bpc_avx2` | `src/x86/looprestoration_avx2.asm:560-1565`, `src/x86/looprestoration.h:75-77` |
//! | `D02733` | `sgr[1]` | `dav1d_sgr_filter_3x3_8bpc_avx2` | `src/x86/looprestoration_avx2.asm:560-1565`, `src/x86/looprestoration.h:75-77` |
//! | `D02739` | `sgr[2]` | `dav1d_sgr_filter_mix_8bpc_avx2` | `src/x86/looprestoration_avx2.asm:560-1565`, `src/x86/looprestoration.h:75-77` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bu modülde tüm SIMD işlevleri yalnız `#[target_feature(enable = "avx2")]`
//!   taşır; EVEX yönergesi üretilmez (`0` EVEX).
//! - Kopyasız O(1) sözleşme: girişte [`wiener_sozlesmesi::<8>`] veya
//!   [`sgr_sozlesmesi::<8>`] ve `Avx2Token::new(CpuFlags::detect())`
//!   denetlenir; sözleşme dışı girdide ilgili skaler devir yardımcısıyla
//!   skaler gövdeye devredilir.
//! - Sınır-güvenli (`SentinelBuffer` / `OverReadProbe`): her kaynak satırı
//!   yalnız sözleşmenin izin verdiği pikseller okunarak yığın üzerindeki
//!   dolgulu `[u8; 420]` tamponuna hazırlanır; çıktı da yalnız `[0, w)`
//!   sütunlarına yazılır.

#![allow(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_packus_epi16, _mm256_add_epi16, _mm256_add_epi32, _mm256_castsi256_si128,
    _mm256_cvtepu8_epi16, _mm256_cvtepu8_epi32, _mm256_extracti128_si256, _mm256_madd_epi16,
    _mm256_max_epi32, _mm256_min_epi32, _mm256_mullo_epi32, _mm256_packs_epi32,
    _mm256_permute4x64_epi64, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setzero_si256,
    _mm256_srai_epi32, _mm256_sub_epi32, _mm256_unpackhi_epi16, _mm256_unpacklo_epi16,
};

use crate::cpu::{Avx2Token, CpuFlags};
use crate::dsp::lr::{
    LrBitDepth, LrEdgeFlags, LrFilterFn, LrParams, sgr_skaler_devret, sgr_sozlesmesi,
    wiener_skaler_devret, wiener_sozlesmesi,
};
use crate::dsp::x86::lr_sse2::{HOR_STRIDE, LrCtx, PAD_ROW_LEN, RowRef, pad_row_8bpc};
use crate::dsp::x86::lr_ssse3::{SGR_BUF_STRIDE, SGR_OUT_STRIDE, SgrRows, calc_row_ab_8bpc};
use crate::dsp::x86::mem;

/// 8 bpc `wiener[tap_idx]` AVX2 kurucusu (`D02745`, `D02752`).
#[must_use]
pub const fn wiener_filter_fn(_token: Avx2Token, tap_idx: usize) -> Option<LrFilterFn> {
    match tap_idx {
        0 => Some(wiener_filter7_8bpc_avx2),
        1 => Some(wiener_filter5_8bpc_avx2),
        _ => None,
    }
}

/// 8 bpc `sgr[sgr_idx]` AVX2 kurucusu (`D02727`, `D02733`, `D02739`).
#[must_use]
pub const fn sgr_filter_fn(_token: Avx2Token, sgr_idx: usize) -> Option<LrFilterFn> {
    match sgr_idx {
        0 => Some(sgr_filter_5x5_8bpc_avx2),
        1 => Some(sgr_filter_3x3_8bpc_avx2),
        2 => Some(sgr_filter_mix_8bpc_avx2),
        _ => None,
    }
}

/// 8 bpc 7-dokunuşlu Wiener filtresi — AVX2 (`D02745`, `wiener[0]`).
pub fn wiener_filter7_8bpc_avx2(
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
    wiener_8bpc_avx2::<false>(
        0, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
    );
}

/// 8 bpc 5-dokunuşlu Wiener filtresi — AVX2 (`D02752`, `wiener[1]`).
pub fn wiener_filter5_8bpc_avx2(
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
    wiener_8bpc_avx2::<true>(
        1, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
    );
}

/// 8 bpc SGR 5×5 filtresi — AVX2 (`D02727`, `sgr[0]`).
pub fn sgr_filter_5x5_8bpc_avx2(
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
    sgr_8bpc_avx2::<0>(
        dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
    );
}

/// 8 bpc SGR 3×3 filtresi — AVX2 (`D02733`, `sgr[1]`).
pub fn sgr_filter_3x3_8bpc_avx2(
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
    sgr_8bpc_avx2::<1>(
        dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
    );
}

/// 8 bpc SGR karma (5×5 + 3×3) filtresi — AVX2 (`D02739`, `sgr[2]`).
pub fn sgr_filter_mix_8bpc_avx2(
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
    sgr_8bpc_avx2::<2>(
        dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
    );
}

#[inline(always)]
fn wiener_8bpc_avx2<const FIVE_TAP: bool>(
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
    ) || Avx2Token::new(CpuFlags::detect()).is_none()
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
    // SAFETY: (1) ISA: yukarıda `Avx2Token::new(CpuFlags::detect())` doğrulandı;
    // AVX2 çalışma zamanında mevcuttur. (2) sınır: `wiener_sozlesmesi::<8>` `dst`,
    // `left` ve `lpf` dilimlerinin sınırlarını O(1) doğruladı; `pad_row_8bpc` yalnız
    // izinli pikselleri yığın tamponuna kopyalar ve çıktı yalnız `[0, w)` aralığına
    // yazılır. (3) hizalama/takma ad: tüm vektör yükleme ve saklamaları `mem`
    // içindeki hizasız sabit boyutlu dizi yardımcılarıyla yapılır.
    unsafe {
        wiener_avx2_giris::<FIVE_TAP>(dst, &c, h, filter);
    }
}

#[inline(always)]
fn sgr_8bpc_avx2<const SGR_IDX: usize>(
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
    ) || Avx2Token::new(CpuFlags::detect()).is_none()
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
    // SAFETY: (1) ISA: yukarıda `Avx2Token::new(CpuFlags::detect())` doğrulandı;
    // AVX2 çalışma zamanında mevcuttur. (2) sınır: `sgr_sozlesmesi::<8>` `dst`,
    // `left` ve `lpf` dilimlerinin sınırlarını O(1) doğruladı; `pad_row_8bpc` yalnız
    // izinli pikselleri yığın tamponuna kopyalar ve çıktı yalnız `[0, w)` aralığına
    // yazılır. (3) hizalama/takma ad: tüm vektör yükleme ve saklamaları `mem`
    // içindeki hizasız sabit boyutlu dizi yardımcılarıyla yapılır.
    unsafe {
        match SGR_IDX {
            0 => sgr_5x5_avx2_giris(dst, &c, h, s0, w0),
            1 => sgr_3x3_avx2_giris(dst, &c, h, s1, w1),
            _ => sgr_mix_avx2_giris(dst, &c, h, s0, s1, w0, w1),
        }
    }
}

// ---------------------------------------------------------------------------
// Wiener AVX2 (256-bit, 16 piksel / adım)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
fn wiener_filter_h_avx2<const FIVE_TAP: bool>(
    out: &mut [i16; HOR_STRIDE],
    c: &LrCtx<'_>,
    dst: &[u8],
    row: RowRef,
    fh: &[i16; 8],
) {
    let mut pad = [0u8; PAD_ROW_LEN];
    pad_row_8bpc::<3>(&mut pad, c, dst, row);

    let z = _mm256_setzero_si256();
    let c06 = _mm256_set1_epi32((i32::from(fh[0]) & 0xffff) | ((i32::from(fh[6]) & 0xffff) << 16));
    let c15 = _mm256_set1_epi32((i32::from(fh[1]) & 0xffff) | ((i32::from(fh[5]) & 0xffff) << 16));
    let c24 = _mm256_set1_epi32((i32::from(fh[2]) & 0xffff) | ((i32::from(fh[4]) & 0xffff) << 16));
    let c3 = _mm256_set1_epi32(((i32::from(fh[3]) + 128) & 0xffff) | (16388 << 16));
    let one16 = _mm256_set1_epi16(1);
    let max_h = _mm256_set1_epi32(8191);

    let mut x = 0usize;
    while x < c.w {
        let p1 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(&pad, x + 1)));
        let p2 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(&pad, x + 2)));
        let p3 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(&pad, x + 3)));
        let p4 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(&pad, x + 4)));
        let p5 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(&pad, x + 5)));

        let mut s_lo = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_madd_epi16(_mm256_unpacklo_epi16(p1, p5), c15),
                _mm256_madd_epi16(_mm256_unpacklo_epi16(p2, p4), c24),
            ),
            _mm256_madd_epi16(_mm256_unpacklo_epi16(p3, one16), c3),
        );
        let mut s_hi = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_madd_epi16(_mm256_unpackhi_epi16(p1, p5), c15),
                _mm256_madd_epi16(_mm256_unpackhi_epi16(p2, p4), c24),
            ),
            _mm256_madd_epi16(_mm256_unpackhi_epi16(p3, one16), c3),
        );

        if !FIVE_TAP {
            let p0 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(&pad, x)));
            let p6 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(&pad, x + 6)));
            s_lo = _mm256_add_epi32(s_lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(p0, p6), c06));
            s_hi = _mm256_add_epi32(s_hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(p0, p6), c06));
        }

        s_lo = _mm256_min_epi32(_mm256_max_epi32(_mm256_srai_epi32::<3>(s_lo), z), max_h);
        s_hi = _mm256_min_epi32(_mm256_max_epi32(_mm256_srai_epi32::<3>(s_hi), z), max_h);

        let r = _mm256_packs_epi32(s_lo, s_hi);
        mem::store_i16x16(mem::chunk_mut::<i16, 16>(out, x), r);
        x += 16;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn wiener_filter_v_row_avx2<const FIVE_TAP: bool>(
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
    let z = _mm256_setzero_si256();
    let c06 = _mm256_set1_epi32((i32::from(fv[0]) & 0xffff) | ((i32::from(fv[6]) & 0xffff) << 16));
    let c15 = _mm256_set1_epi32((i32::from(fv[1]) & 0xffff) | ((i32::from(fv[5]) & 0xffff) << 16));
    let c24 = _mm256_set1_epi32((i32::from(fv[2]) & 0xffff) | ((i32::from(fv[4]) & 0xffff) << 16));
    let c3 = _mm256_set1_epi32(i32::from(fv[3]) & 0xffff);
    let bias = _mm256_set1_epi32(-261120);

    let mut out_row = [0u8; HOR_STRIDE];
    let mut x = 0usize;
    while x < c.w {
        let v1 = mem::load_i16x16(mem::chunk::<i16, 16>(r1, x));
        let v2 = mem::load_i16x16(mem::chunk::<i16, 16>(r2, x));
        let v3 = mem::load_i16x16(mem::chunk::<i16, 16>(r3, x));
        let v4 = mem::load_i16x16(mem::chunk::<i16, 16>(r4, x));
        let v5 = mem::load_i16x16(mem::chunk::<i16, 16>(r5, x));

        let mut s_lo = _mm256_add_epi32(
            _mm256_add_epi32(bias, _mm256_madd_epi16(_mm256_unpacklo_epi16(v1, v5), c15)),
            _mm256_add_epi32(
                _mm256_madd_epi16(_mm256_unpacklo_epi16(v2, v4), c24),
                _mm256_madd_epi16(_mm256_unpacklo_epi16(v3, z), c3),
            ),
        );
        let mut s_hi = _mm256_add_epi32(
            _mm256_add_epi32(bias, _mm256_madd_epi16(_mm256_unpackhi_epi16(v1, v5), c15)),
            _mm256_add_epi32(
                _mm256_madd_epi16(_mm256_unpackhi_epi16(v2, v4), c24),
                _mm256_madd_epi16(_mm256_unpackhi_epi16(v3, z), c3),
            ),
        );

        if !FIVE_TAP {
            let v0 = mem::load_i16x16(mem::chunk::<i16, 16>(r0, x));
            let v6 = mem::load_i16x16(mem::chunk::<i16, 16>(r6, x));
            s_lo = _mm256_add_epi32(s_lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(v0, v6), c06));
            s_hi = _mm256_add_epi32(s_hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(v0, v6), c06));
        }

        let max_v = _mm256_set1_epi32(255);
        s_lo = _mm256_min_epi32(_mm256_max_epi32(_mm256_srai_epi32::<11>(s_lo), z), max_v);
        s_hi = _mm256_min_epi32(_mm256_max_epi32(_mm256_srai_epi32::<11>(s_hi), z), max_v);
        let packed16 = _mm256_packs_epi32(s_lo, s_hi);
        let bytes = _mm_packus_epi16(
            _mm256_castsi256_si128(packed16),
            _mm256_extracti128_si256::<1>(packed16),
        );
        mem::store_u8x16(mem::chunk_mut::<u8, 16>(&mut out_row, x), bytes);
        x += 16;
    }

    let p = p_row * c.stride;
    dst[p..p + c.w].copy_from_slice(&out_row[..c.w]);
}

struct WienerAvx2State {
    hor: [[i16; HOR_STRIDE]; 6],
    tmp: [i16; HOR_STRIDE],
    ptrs: [usize; 7],
}

impl WienerAvx2State {
    #[target_feature(enable = "avx2")]
    #[inline]
    fn filter_h<const FIVE_TAP: bool>(
        &mut self,
        c: &LrCtx<'_>,
        dst: &[u8],
        row: RowRef,
        r: usize,
        fh: &[i16; 8],
    ) {
        wiener_filter_h_avx2::<FIVE_TAP>(&mut self.hor[r], c, dst, row, fh);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn filter_v<const FIVE_TAP: bool>(
        &mut self,
        dst: &mut [u8],
        c: &LrCtx<'_>,
        p_row: usize,
        fv: &[i16; 8],
    ) {
        wiener_filter_v_row_avx2::<FIVE_TAP>(
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

    #[target_feature(enable = "avx2")]
    #[inline]
    fn filter_hv<const FIVE_TAP: bool>(
        &mut self,
        dst: &mut [u8],
        c: &LrCtx<'_>,
        p_row: usize,
        row: RowRef,
        filter: &[[i16; 8]; 2],
    ) {
        wiener_filter_h_avx2::<FIVE_TAP>(&mut self.tmp, c, dst, row, &filter[0]);
        wiener_filter_v_row_avx2::<FIVE_TAP>(
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
        let n = (c.w + 15) & !15;
        self.hor[dst_row][..n].copy_from_slice(&self.tmp[..n]);
        self.ptrs.copy_within(1..7, 0);
        self.ptrs[6] = self.ptrs[0];
    }
}

#[target_feature(enable = "avx2")]
fn wiener_avx2_giris<const FIVE_TAP: bool>(
    dst: &mut [u8],
    c: &LrCtx<'_>,
    h: usize,
    filter: &[[i16; 8]; 2],
) {
    let fh = &filter[0];
    let fv = &filter[1];
    let mut st = WienerAvx2State {
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
// SGR AVX2 (256-bit, 8 i32 / 16 u8)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
fn box3_row_h_avx2(
    sumsq: &mut [i32; SGR_BUF_STRIDE],
    sum: &mut [i32; SGR_BUF_STRIDE],
    pad: &[u8; PAD_ROW_LEN],
    w: usize,
) {
    let z = _mm256_setzero_si256();
    let mut j = 0usize;
    while j < w + 2 {
        let a = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(pad, j + 1)));
        let b = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(pad, j + 2)));
        let c = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(pad, j + 3)));

        let s16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_add_epi16(_mm256_add_epi16(a, b), c));
        mem::store_i32x8(
            mem::chunk_mut::<i32, 8>(sum, j),
            _mm256_unpacklo_epi16(s16, z),
        );
        mem::store_i32x8(
            mem::chunk_mut::<i32, 8>(sum, j + 8),
            _mm256_unpackhi_epi16(s16, z),
        );

        let a_p = _mm256_permute4x64_epi64::<0xD8>(a);
        let b_p = _mm256_permute4x64_epi64::<0xD8>(b);
        let c_p = _mm256_permute4x64_epi64::<0xD8>(c);
        let ab_lo = _mm256_unpacklo_epi16(a_p, b_p);
        let ab_hi = _mm256_unpackhi_epi16(a_p, b_p);
        let c0_lo = _mm256_unpacklo_epi16(c_p, z);
        let c0_hi = _mm256_unpackhi_epi16(c_p, z);
        let sq_lo = _mm256_add_epi32(
            _mm256_madd_epi16(ab_lo, ab_lo),
            _mm256_madd_epi16(c0_lo, c0_lo),
        );
        let sq_hi = _mm256_add_epi32(
            _mm256_madd_epi16(ab_hi, ab_hi),
            _mm256_madd_epi16(c0_hi, c0_hi),
        );
        mem::store_i32x8(mem::chunk_mut::<i32, 8>(sumsq, j), sq_lo);
        mem::store_i32x8(mem::chunk_mut::<i32, 8>(sumsq, j + 8), sq_hi);
        j += 16;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn box5_row_h_avx2(
    sumsq: &mut [i32; SGR_BUF_STRIDE],
    sum: &mut [i32; SGR_BUF_STRIDE],
    pad: &[u8; PAD_ROW_LEN],
    w: usize,
) {
    let z = _mm256_setzero_si256();
    let mut j = 0usize;
    while j < w + 2 {
        let a = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(pad, j)));
        let b = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(pad, j + 1)));
        let c = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(pad, j + 2)));
        let d = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(pad, j + 3)));
        let e = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(pad, j + 4)));

        let s16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_add_epi16(
            _mm256_add_epi16(_mm256_add_epi16(a, b), _mm256_add_epi16(c, d)),
            e,
        ));
        mem::store_i32x8(
            mem::chunk_mut::<i32, 8>(sum, j),
            _mm256_unpacklo_epi16(s16, z),
        );
        mem::store_i32x8(
            mem::chunk_mut::<i32, 8>(sum, j + 8),
            _mm256_unpackhi_epi16(s16, z),
        );

        let a_p = _mm256_permute4x64_epi64::<0xD8>(a);
        let b_p = _mm256_permute4x64_epi64::<0xD8>(b);
        let c_p = _mm256_permute4x64_epi64::<0xD8>(c);
        let d_p = _mm256_permute4x64_epi64::<0xD8>(d);
        let e_p = _mm256_permute4x64_epi64::<0xD8>(e);
        let ab_lo = _mm256_unpacklo_epi16(a_p, b_p);
        let ab_hi = _mm256_unpackhi_epi16(a_p, b_p);
        let cd_lo = _mm256_unpacklo_epi16(c_p, d_p);
        let cd_hi = _mm256_unpackhi_epi16(c_p, d_p);
        let e0_lo = _mm256_unpacklo_epi16(e_p, z);
        let e0_hi = _mm256_unpackhi_epi16(e_p, z);
        let sq_lo = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_madd_epi16(ab_lo, ab_lo),
                _mm256_madd_epi16(cd_lo, cd_lo),
            ),
            _mm256_madd_epi16(e0_lo, e0_lo),
        );
        let sq_hi = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_madd_epi16(ab_hi, ab_hi),
                _mm256_madd_epi16(cd_hi, cd_hi),
            ),
            _mm256_madd_epi16(e0_hi, e0_hi),
        );
        mem::store_i32x8(mem::chunk_mut::<i32, 8>(sumsq, j), sq_lo);
        mem::store_i32x8(mem::chunk_mut::<i32, 8>(sumsq, j + 8), sq_hi);
        j += 16;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn box3_vert_avx2<const K: usize>(
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
        let vq = _mm256_add_epi32(
            _mm256_add_epi32(
                mem::load_i32x8(mem::chunk::<i32, 8>(q0, x)),
                mem::load_i32x8(mem::chunk::<i32, 8>(q1, x)),
            ),
            mem::load_i32x8(mem::chunk::<i32, 8>(q2, x)),
        );
        let vs = _mm256_add_epi32(
            _mm256_add_epi32(
                mem::load_i32x8(mem::chunk::<i32, 8>(s0, x)),
                mem::load_i32x8(mem::chunk::<i32, 8>(s1, x)),
            ),
            mem::load_i32x8(mem::chunk::<i32, 8>(s2, x)),
        );
        mem::store_i32x8(mem::chunk_mut::<i32, 8>(oa, x), vq);
        mem::store_i32x8(mem::chunk_mut::<i32, 8>(ob, x), vs);
        x += 8;
    }
    calc_row_ab_8bpc(oa, ob, w, s, 9, 455);
    sp.rotate_left(1);
}

#[target_feature(enable = "avx2")]
#[inline]
fn box5_vert_avx2(
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
        let vq = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    mem::load_i32x8(mem::chunk::<i32, 8>(q[0], x)),
                    mem::load_i32x8(mem::chunk::<i32, 8>(q[1], x)),
                ),
                _mm256_add_epi32(
                    mem::load_i32x8(mem::chunk::<i32, 8>(q[2], x)),
                    mem::load_i32x8(mem::chunk::<i32, 8>(q[3], x)),
                ),
            ),
            mem::load_i32x8(mem::chunk::<i32, 8>(q[4], x)),
        );
        let vs = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    mem::load_i32x8(mem::chunk::<i32, 8>(t[0], x)),
                    mem::load_i32x8(mem::chunk::<i32, 8>(t[1], x)),
                ),
                _mm256_add_epi32(
                    mem::load_i32x8(mem::chunk::<i32, 8>(t[2], x)),
                    mem::load_i32x8(mem::chunk::<i32, 8>(t[3], x)),
                ),
            ),
            mem::load_i32x8(mem::chunk::<i32, 8>(t[4], x)),
        );
        mem::store_i32x8(mem::chunk_mut::<i32, 8>(oa, x), vq);
        mem::store_i32x8(mem::chunk_mut::<i32, 8>(ob, x), vs);
        x += 8;
    }
    calc_row_ab_8bpc(oa, ob, w, s, 25, 164);
    sp.rotate_left(2);
}

#[target_feature(enable = "avx2")]
#[inline]
fn finish_filter_row1_avx2<const K: usize>(
    tmp: &mut [i32],
    dst: &[u8],
    row_off: usize,
    ab: &SgrRows<K>,
    ap: [usize; 3],
    w: usize,
) {
    let (b0, b1, b2) = (&ab.s[ap[0]], &ab.s[ap[1]], &ab.s[ap[2]]);
    let (a0, a1, a2) = (&ab.sq[ap[0]], &ab.sq[ap[1]], &ab.sq[ap[2]]);
    let c4 = _mm256_set1_epi32(4);
    let c3 = _mm256_set1_epi32(3);
    let rnd = _mm256_set1_epi32(1 << 8);
    let mut src_row = [0u8; HOR_STRIDE];
    src_row[..w].copy_from_slice(&dst[row_off..row_off + w]);

    let mut i = 0usize;
    while i < w {
        let s16 = mem::load_u8x16(mem::chunk::<u8, 16>(&src_row, i));
        let s_lo = _mm256_cvtepu8_epi32(s16);
        // Shift upper 8 bytes down via a second load at `i + 8`.
        let s_hi = _mm256_cvtepu8_epi32(mem::load_u8x16(mem::chunk::<u8, 16>(&src_row, i + 8)));

        for (off, src_v) in [(i, s_lo), (i + 8, s_hi)] {
            let j = off + 1;
            let b_cross = _mm256_add_epi32(
                _mm256_add_epi32(
                    mem::load_i32x8(mem::chunk::<i32, 8>(b1, j)),
                    mem::load_i32x8(mem::chunk::<i32, 8>(b1, j - 1)),
                ),
                _mm256_add_epi32(
                    mem::load_i32x8(mem::chunk::<i32, 8>(b1, j + 1)),
                    _mm256_add_epi32(
                        mem::load_i32x8(mem::chunk::<i32, 8>(b0, j)),
                        mem::load_i32x8(mem::chunk::<i32, 8>(b2, j)),
                    ),
                ),
            );
            let b_diag = _mm256_add_epi32(
                _mm256_add_epi32(
                    mem::load_i32x8(mem::chunk::<i32, 8>(b0, j - 1)),
                    mem::load_i32x8(mem::chunk::<i32, 8>(b2, j - 1)),
                ),
                _mm256_add_epi32(
                    mem::load_i32x8(mem::chunk::<i32, 8>(b0, j + 1)),
                    mem::load_i32x8(mem::chunk::<i32, 8>(b2, j + 1)),
                ),
            );
            let a_v = _mm256_add_epi32(
                _mm256_mullo_epi32(b_cross, c4),
                _mm256_mullo_epi32(b_diag, c3),
            );

            let a_cross = _mm256_add_epi32(
                _mm256_add_epi32(
                    mem::load_i32x8(mem::chunk::<i32, 8>(a1, j)),
                    mem::load_i32x8(mem::chunk::<i32, 8>(a1, j - 1)),
                ),
                _mm256_add_epi32(
                    mem::load_i32x8(mem::chunk::<i32, 8>(a1, j + 1)),
                    _mm256_add_epi32(
                        mem::load_i32x8(mem::chunk::<i32, 8>(a0, j)),
                        mem::load_i32x8(mem::chunk::<i32, 8>(a2, j)),
                    ),
                ),
            );
            let a_diag = _mm256_add_epi32(
                _mm256_add_epi32(
                    mem::load_i32x8(mem::chunk::<i32, 8>(a0, j - 1)),
                    mem::load_i32x8(mem::chunk::<i32, 8>(a2, j - 1)),
                ),
                _mm256_add_epi32(
                    mem::load_i32x8(mem::chunk::<i32, 8>(a0, j + 1)),
                    mem::load_i32x8(mem::chunk::<i32, 8>(a2, j + 1)),
                ),
            );
            let b_v = _mm256_add_epi32(
                _mm256_mullo_epi32(a_cross, c4),
                _mm256_mullo_epi32(a_diag, c3),
            );

            let prod = _mm256_mullo_epi32(a_v, src_v);
            let res = _mm256_srai_epi32::<9>(_mm256_add_epi32(_mm256_sub_epi32(b_v, prod), rnd));
            mem::store_i32x8(mem::chunk_mut::<i32, 8>(tmp, off), res);
        }
        i += 16;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn finish_filter2_avx2(
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
    let c6 = _mm256_set1_epi32(6);
    let c5 = _mm256_set1_epi32(5);
    let rnd0 = _mm256_set1_epi32(1 << 8);
    let mut src0_row = [0u8; HOR_STRIDE];
    src0_row[..w].copy_from_slice(&dst[r0_off..r0_off + w]);

    let mut i = 0usize;
    while i < w {
        let s_lo = _mm256_cvtepu8_epi32(mem::load_u8x16(mem::chunk::<u8, 16>(&src0_row, i)));
        let s_hi = _mm256_cvtepu8_epi32(mem::load_u8x16(mem::chunk::<u8, 16>(&src0_row, i + 8)));

        for (off, src_v) in [(i, s_lo), (i + 8, s_hi)] {
            let j = off + 1;
            let b_mid = _mm256_add_epi32(
                mem::load_i32x8(mem::chunk::<i32, 8>(b0, j)),
                mem::load_i32x8(mem::chunk::<i32, 8>(b1, j)),
            );
            let b_side = _mm256_add_epi32(
                _mm256_add_epi32(
                    mem::load_i32x8(mem::chunk::<i32, 8>(b0, j - 1)),
                    mem::load_i32x8(mem::chunk::<i32, 8>(b1, j - 1)),
                ),
                _mm256_add_epi32(
                    mem::load_i32x8(mem::chunk::<i32, 8>(b0, j + 1)),
                    mem::load_i32x8(mem::chunk::<i32, 8>(b1, j + 1)),
                ),
            );
            let a_v = _mm256_add_epi32(
                _mm256_mullo_epi32(b_mid, c6),
                _mm256_mullo_epi32(b_side, c5),
            );

            let a_mid = _mm256_add_epi32(
                mem::load_i32x8(mem::chunk::<i32, 8>(a0, j)),
                mem::load_i32x8(mem::chunk::<i32, 8>(a1, j)),
            );
            let a_side = _mm256_add_epi32(
                _mm256_add_epi32(
                    mem::load_i32x8(mem::chunk::<i32, 8>(a0, j - 1)),
                    mem::load_i32x8(mem::chunk::<i32, 8>(a1, j - 1)),
                ),
                _mm256_add_epi32(
                    mem::load_i32x8(mem::chunk::<i32, 8>(a0, j + 1)),
                    mem::load_i32x8(mem::chunk::<i32, 8>(a1, j + 1)),
                ),
            );
            let b_v = _mm256_add_epi32(
                _mm256_mullo_epi32(a_mid, c6),
                _mm256_mullo_epi32(a_side, c5),
            );

            let prod = _mm256_mullo_epi32(a_v, src_v);
            let res = _mm256_srai_epi32::<9>(_mm256_add_epi32(_mm256_sub_epi32(b_v, prod), rnd0));
            mem::store_i32x8(mem::chunk_mut::<i32, 8>(tmp, off), res);
        }
        i += 16;
    }

    if h <= 1 {
        return;
    }

    let rnd1 = _mm256_set1_epi32(1 << 7);
    let mut src1_row = [0u8; HOR_STRIDE];
    src1_row[..w].copy_from_slice(&dst[r1_off..r1_off + w]);

    let mut i = 0usize;
    while i < w {
        let s_lo = _mm256_cvtepu8_epi32(mem::load_u8x16(mem::chunk::<u8, 16>(&src1_row, i)));
        let s_hi = _mm256_cvtepu8_epi32(mem::load_u8x16(mem::chunk::<u8, 16>(&src1_row, i + 8)));

        for (off, src_v) in [(i, s_lo), (i + 8, s_hi)] {
            let j = off + 1;
            let b_mid = mem::load_i32x8(mem::chunk::<i32, 8>(b1, j));
            let b_side = _mm256_add_epi32(
                mem::load_i32x8(mem::chunk::<i32, 8>(b1, j - 1)),
                mem::load_i32x8(mem::chunk::<i32, 8>(b1, j + 1)),
            );
            let a_v = _mm256_add_epi32(
                _mm256_mullo_epi32(b_mid, c6),
                _mm256_mullo_epi32(b_side, c5),
            );

            let a_mid = mem::load_i32x8(mem::chunk::<i32, 8>(a1, j));
            let a_side = _mm256_add_epi32(
                mem::load_i32x8(mem::chunk::<i32, 8>(a1, j - 1)),
                mem::load_i32x8(mem::chunk::<i32, 8>(a1, j + 1)),
            );
            let b_v = _mm256_add_epi32(
                _mm256_mullo_epi32(a_mid, c6),
                _mm256_mullo_epi32(a_side, c5),
            );

            let prod = _mm256_mullo_epi32(a_v, src_v);
            let res = _mm256_srai_epi32::<8>(_mm256_add_epi32(_mm256_sub_epi32(b_v, prod), rnd1));
            mem::store_i32x8(mem::chunk_mut::<i32, 8>(tmp, SGR_OUT_STRIDE + off), res);
        }
        i += 16;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn pack_i32x16_to_u8x16(d16: __m256i, v_lo: __m256i, v_hi: __m256i) -> __m128i {
    let v16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(v_lo, v_hi));
    let sum16 = _mm256_add_epi16(d16, v16);
    _mm_packus_epi16(
        _mm256_castsi256_si128(sum16),
        _mm256_extracti128_si256::<1>(sum16),
    )
}

#[target_feature(enable = "avx2")]
#[inline]
fn weighted_row1_avx2(dst: &mut [u8], off: usize, t1: &[i32], w: usize, w1: i32) {
    let w1_v = _mm256_set1_epi32(w1);
    let rnd = _mm256_set1_epi32(1 << 10);
    let mut io_row = [0u8; HOR_STRIDE];
    io_row[..w].copy_from_slice(&dst[off..off + w]);

    let mut i = 0usize;
    while i < w {
        let d16 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(&io_row, i)));
        let v_lo = _mm256_srai_epi32::<11>(_mm256_add_epi32(
            _mm256_mullo_epi32(mem::load_i32x8(mem::chunk::<i32, 8>(t1, i)), w1_v),
            rnd,
        ));
        let v_hi = _mm256_srai_epi32::<11>(_mm256_add_epi32(
            _mm256_mullo_epi32(mem::load_i32x8(mem::chunk::<i32, 8>(t1, i + 8)), w1_v),
            rnd,
        ));
        let out = pack_i32x16_to_u8x16(d16, v_lo, v_hi);
        mem::store_u8x16(mem::chunk_mut::<u8, 16>(&mut io_row, i), out);
        i += 16;
    }

    dst[off..off + w].copy_from_slice(&io_row[..w]);
}

#[target_feature(enable = "avx2")]
#[inline]
fn weighted2_avx2(
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
    let w0_v = _mm256_set1_epi32(w0);
    let w1_v = _mm256_set1_epi32(w1);
    let rnd = _mm256_set1_epi32(1 << 10);

    for j in 0..h {
        let row = off + j * stride;
        let t = j * SGR_OUT_STRIDE;
        let mut io_row = [0u8; HOR_STRIDE];
        io_row[..w].copy_from_slice(&dst[row..row + w]);

        let mut i = 0usize;
        while i < w {
            let d16 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(&io_row, i)));
            let v_lo = _mm256_srai_epi32::<11>(_mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(mem::load_i32x8(mem::chunk::<i32, 8>(t1, t + i)), w0_v),
                    _mm256_mullo_epi32(mem::load_i32x8(mem::chunk::<i32, 8>(t2, t + i)), w1_v),
                ),
                rnd,
            ));
            let v_hi = _mm256_srai_epi32::<11>(_mm256_add_epi32(
                _mm256_add_epi32(
                    _mm256_mullo_epi32(mem::load_i32x8(mem::chunk::<i32, 8>(t1, t + i + 8)), w0_v),
                    _mm256_mullo_epi32(mem::load_i32x8(mem::chunk::<i32, 8>(t2, t + i + 8)), w1_v),
                ),
                rnd,
            ));
            let out = pack_i32x16_to_u8x16(d16, v_lo, v_hi);
            mem::store_u8x16(mem::chunk_mut::<u8, 16>(&mut io_row, i), out);
            i += 16;
        }

        dst[row..row + w].copy_from_slice(&io_row[..w]);
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn finish1_avx2(
    dst: &mut [u8],
    c: &LrCtx<'_>,
    dst_row: &mut usize,
    ab: &SgrRows<3>,
    ap: &mut [usize; 3],
    tmp: &mut [i32; 2 * SGR_OUT_STRIDE],
    w1: i32,
) {
    let r = *dst_row;
    finish_filter_row1_avx2::<3>(tmp, dst, r * c.stride, ab, *ap, c.w);
    weighted_row1_avx2(dst, r * c.stride, &tmp[..], c.w, w1);
    *dst_row += 1;
    ap.rotate_left(1);
}

#[target_feature(enable = "avx2")]
#[inline]
fn finish2_avx2(
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
    finish_filter2_avx2(tmp, dst, r * c.stride, r1 * c.stride, ab, *ap, c.w, h);
    weighted_row1_avx2(dst, r * c.stride, &tmp[..], c.w, w0);
    *dst_row += 1;
    if h > 1 {
        weighted_row1_avx2(dst, (r + 1) * c.stride, &tmp[SGR_OUT_STRIDE..], c.w, w0);
        *dst_row += 1;
    }
    ap.rotate_left(1);
}

#[target_feature(enable = "avx2")]
#[inline]
fn finish_mix_avx2(
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
    finish_filter2_avx2(tmp5, dst, r * c.stride, r1 * c.stride, ab5, *ap5, c.w, h);
    finish_filter_row1_avx2::<4>(
        &mut tmp3[..],
        dst,
        r * c.stride,
        ab3,
        [ap3[0], ap3[1], ap3[2]],
        c.w,
    );
    if h > 1 {
        finish_filter_row1_avx2::<4>(
            &mut tmp3[SGR_OUT_STRIDE..],
            dst,
            (r + 1) * c.stride,
            ab3,
            [ap3[1], ap3[2], ap3[3]],
            c.w,
        );
    }
    weighted2_avx2(dst, r * c.stride, c.stride, tmp5, tmp3, c.w, h, w0, w1);
    *dst_row += h;
    ap5.rotate_left(1);
    ap3.rotate_left(1);
}

#[target_feature(enable = "avx2")]
fn sgr_3x3_avx2_giris(dst: &mut [u8], c: &LrCtx<'_>, h: usize, s1: u32, w1: i32) {
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
            box3_row_h_avx2(&mut sums.sq[$r], &mut sums.s[$r], &pad, w);
        }};
    }

    let kuyruk = 'akis: {
        if c.edges.contains(LrEdgeFlags::HAVE_TOP) {
            sp = [0, 1, 2];
            row_h!(RowRef::Lpf(0), 0);
            row_h!(RowRef::Lpf(1), 1);
            row_h!(RowRef::Main(sr), sp[2]);
            box3_vert_avx2::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            row_h!(RowRef::Main(sr), sp[2]);
            box3_vert_avx2::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
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
            box3_vert_avx2::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp[2] = 1;
            row_h!(RowRef::Main(sr), sp[2]);
            box3_vert_avx2::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
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
            box3_vert_avx2::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            finish1_avx2(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
            h -= 1;
            if h == 0 {
                break;
            }
        }

        if !c.edges.contains(LrEdgeFlags::HAVE_BOTTOM) {
            break 'akis Kuyruk::Vert2;
        }
        row_h!(RowRef::Lpf(6), sp[2]);
        box3_vert_avx2::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
        finish1_avx2(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        row_h!(RowRef::Lpf(7), sp[2]);
        box3_vert_avx2::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
        finish1_avx2(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        Kuyruk::Bitti
    };

    match kuyruk {
        Kuyruk::Bitti => return,
        Kuyruk::Vert2 => {
            sp[2] = sp[1];
            box3_vert_avx2::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            finish1_avx2(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        }
        Kuyruk::Vert1 => {
            sp[2] = sp[1];
            box3_vert_avx2::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            ap.rotate_left(1);
        }
    }
    sp[2] = sp[1];
    box3_vert_avx2::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
    finish1_avx2(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
}

#[target_feature(enable = "avx2")]
fn sgr_5x5_avx2_giris(dst: &mut [u8], c: &LrCtx<'_>, h: usize, s0: u32, w0: i32) {
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
            box5_row_h_avx2(&mut sums.sq[$r], &mut sums.s[$r], &pad, w);
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
            box5_vert_avx2(&sums, &mut sp, &mut ab, ap[1], w, s0);
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
            box5_vert_avx2(&sums, &mut sp, &mut ab, ap[1], w, s0);
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
            box5_vert_avx2(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2_avx2(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
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
            box5_vert_avx2(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2_avx2(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
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
            box5_vert_avx2(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2_avx2(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
            true
        }
        Kuyruk::Vert1 => {
            sp[4] = sp[3];
            box5_vert_avx2(&sums, &mut sp, &mut ab, ap[1], w, s0);
            ap.rotate_left(1);
            true
        }
    };
    if output_1 {
        sp[3] = sp[2];
        sp[4] = sp[2];
        box5_vert_avx2(&sums, &mut sp, &mut ab, ap[1], w, s0);
        finish2_avx2(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 1, w0);
    } else {
        box5_vert_avx2(&sums, &mut sp, &mut ab, ap[1], w, s0);
        finish2_avx2(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
    }
}

#[target_feature(enable = "avx2")]
fn sgr_mix_avx2_giris(dst: &mut [u8], c: &LrCtx<'_>, h: usize, s0: u32, s1: u32, w0: i32, w1: i32) {
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
            box3_row_h_avx2(&mut sums3.sq[$r3], &mut sums3.s[$r3], &pad, w);
            box5_row_h_avx2(&mut sums5.sq[$r5], &mut sums5.s[$r5], &pad, w);
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
            box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            row_h35!(RowRef::Main(sr), sp3[2], 3);
            sr += 1;
            box5_vert_avx2(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            ap5.rotate_left(1);
            box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
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
            box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp5[4] = 1;
            sp3[2] = 1;
            row_h35!(RowRef::Main(sr), 1, 1);
            sr += 1;
            box5_vert_avx2(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            ap5.rotate_left(1);
            box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
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
            box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            row_h35!(RowRef::Main(sr), sp3[2], 3);
            sr += 1;
            box5_vert_avx2(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix_avx2(
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
            box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            row_h35!(RowRef::Main(sr), sp3[2], sp5[4]);
            sr += 1;
            box5_vert_avx2(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix_avx2(
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
        box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
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
            box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            sp3[2] = sp3[1];
            false
        }
        Kuyruk::Odd => {
            sp5[4] = sp5[3];
            sp3[2] = sp3[1];
            box5_vert_avx2(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix_avx2(
                dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 2, w0, w1,
            );
            true
        }
        Kuyruk::Vert1 => {
            sp5[4] = sp5[3];
            sp3[2] = sp3[1];
            box5_vert_avx2(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            ap5.rotate_left(1);
            box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            true
        }
    };
    if output_1 {
        sp5[3] = sp5[2];
        sp5[4] = sp5[2];
        sp3[2] = sp3[1];
        box5_vert_avx2(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
        box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
        ap3.rotate_left(1);
        finish_mix_avx2(
            dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 1, w0, w1,
        );
    } else {
        box5_vert_avx2(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
        box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
        finish_mix_avx2(
            dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 2, w0, w1,
        );
    }
}
