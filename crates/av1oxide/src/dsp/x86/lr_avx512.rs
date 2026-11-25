//! 8 bpc loop restoration (Wiener / SGRPROJ) x86_64 AVX-512ICL çekirdekleri
//! (`D02747` + `wiener[1]@8` takma adı, `D02729`, `D02735`, `D02741`; P46 Adım 3 / Paket A).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Kayıt | Yuva | Sembol | Kaynak |
//! |---|---|---|---|
//! | `D02747` | `wiener[0]`, `wiener[1]` | `dav1d_wiener_filter7_8bpc_avx512icl` | `src/x86/looprestoration_avx512.asm:212-502`, `src/x86/looprestoration.h:82-83` |
//! | `D02729` | `sgr[0]` | `dav1d_sgr_filter_5x5_8bpc_avx512icl` | `src/x86/looprestoration_avx512.asm:504-1219`, `src/x86/looprestoration.h:84-86` |
//! | `D02735` | `sgr[1]` | `dav1d_sgr_filter_3x3_8bpc_avx512icl` | `src/x86/looprestoration_avx512.asm:504-1219`, `src/x86/looprestoration.h:84-86` |
//! | `D02741` | `sgr[2]` | `dav1d_sgr_filter_mix_8bpc_avx512icl` | `src/x86/looprestoration_avx512.asm:504-1219`, `src/x86/looprestoration.h:84-86` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bu modülde tüm SIMD işlevleri
//!   `#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]`
//!   taşır; `vpgather*` / `vpscatter*` yönergesi üretilmez (`0` gather/scatter,
//!   `SLOW_GATHER` engeli yoktur).
//! - Kopyasız O(1) sözleşme: girişte [`wiener_sozlesmesi::<8>`] veya
//!   [`sgr_sozlesmesi::<8>`] ve `Avx512IclToken::new(CpuFlags::detect())`
//!   denetlenir; sözleşme dışı girdide ilgili skaler devir yardımcısıyla
//!   skaler gövdeye devredilir.
//! - Sınır-güvenli (`SentinelBuffer` / `OverReadProbe`): her kaynak satırı
//!   yalnız sözleşmenin izin verdiği pikseller okunarak yığın üzerindeki
//!   dolgulu `[u8; 420]` tamponuna hazırlanır; çıktı da yalnız `[0, w)`
//!   sütunlarına yazılır.

#![allow(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use core::arch::x86_64::{
    __m256i, __m512i, _mm256_add_epi16, _mm256_cvtepu8_epi16, _mm512_add_epi32,
    _mm512_cvtepu8_epi16, _mm512_cvtepu8_epi32, _mm512_cvtsepi32_epi16, _mm512_cvtusepi16_epi8,
    _mm512_madd_epi16, _mm512_max_epi32, _mm512_min_epi32, _mm512_mullo_epi32, _mm512_packs_epi32,
    _mm512_permutexvar_epi64, _mm512_set1_epi16, _mm512_set1_epi32, _mm512_setr_epi64,
    _mm512_setzero_si512, _mm512_srai_epi32, _mm512_sub_epi32, _mm512_unpackhi_epi16,
    _mm512_unpacklo_epi16,
};

use crate::cpu::{Avx512IclToken, CpuFlags};
use crate::dsp::lr::{
    LrBitDepth, LrEdgeFlags, LrFilterFn, LrParams, sgr_skaler_devret, sgr_sozlesmesi,
    wiener_skaler_devret, wiener_sozlesmesi,
};
use crate::dsp::x86::lr_sse2::{HOR_STRIDE, LrCtx, PAD_ROW_LEN, RowRef, pad_row_8bpc};
use crate::dsp::x86::lr_ssse3::{SGR_BUF_STRIDE, SGR_OUT_STRIDE, SgrRows, calc_row_ab_8bpc};
use crate::dsp::x86::mem;

/// 8 bpc `wiener[0]` ve `wiener[1]` AVX512ICL kurucusu (`D02747`;
/// `src/x86/looprestoration.h:82-83`: iki yuva da `wiener_filter7_8bpc_avx512icl`'e bağlanır).
#[must_use]
pub const fn wiener_filter_fn(_token: Avx512IclToken, tap_idx: usize) -> Option<LrFilterFn> {
    match tap_idx {
        0 | 1 => Some(wiener_filter7_8bpc_avx512icl),
        _ => None,
    }
}

/// 8 bpc `sgr[sgr_idx]` AVX512ICL kurucusu (`D02729`, `D02735`, `D02741`).
#[must_use]
pub const fn sgr_filter_fn(_token: Avx512IclToken, sgr_idx: usize) -> Option<LrFilterFn> {
    match sgr_idx {
        0 => Some(sgr_filter_5x5_8bpc_avx512icl),
        1 => Some(sgr_filter_3x3_8bpc_avx512icl),
        2 => Some(sgr_filter_mix_8bpc_avx512icl),
        _ => None,
    }
}

/// 8 bpc 7-dokunuşlu (ve 5-dokunuşlu) Wiener filtresi — AVX512ICL (`D02747`, `wiener[0..2]`).
pub fn wiener_filter7_8bpc_avx512icl(
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
        0,
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
    ) || Avx512IclToken::new(CpuFlags::detect()).is_none()
    {
        wiener_skaler_devret::<8>(
            0, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
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
    // SAFETY: (1) ISA: yukarıda `Avx512IclToken::new(CpuFlags::detect())` doğrulandı;
    // AVX512ICL çalışma zamanında mevcuttur. (2) sınır: `wiener_sozlesmesi::<8>` `dst`,
    // `left` ve `lpf` dilimlerinin sınırlarını O(1) doğruladı; `pad_row_8bpc` yalnız
    // izinli pikselleri yığın tamponuna kopyalar ve çıktı yalnız `[0, w)` aralığına
    // yazılır. (3) hizalama/takma ad: tüm vektör yükleme ve saklamaları `mem`
    // içindeki hizasız sabit boyutlu dizi yardımcılarıyla yapılır.
    unsafe {
        wiener_avx512_giris(dst, &c, h, filter);
    }
}

/// 8 bpc SGR 5×5 filtresi — AVX512ICL (`D02729`, `sgr[0]`).
pub fn sgr_filter_5x5_8bpc_avx512icl(
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
    sgr_8bpc_avx512::<0>(
        dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
    );
}

/// 8 bpc SGR 3×3 filtresi — AVX512ICL (`D02735`, `sgr[1]`).
pub fn sgr_filter_3x3_8bpc_avx512icl(
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
    sgr_8bpc_avx512::<1>(
        dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
    );
}

/// 8 bpc SGR karma (5×5 + 3×3) filtresi — AVX512ICL (`D02741`, `sgr[2]`).
pub fn sgr_filter_mix_8bpc_avx512icl(
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
    sgr_8bpc_avx512::<2>(
        dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
    );
}

#[inline(always)]
fn sgr_8bpc_avx512<const SGR_IDX: usize>(
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
    ) || Avx512IclToken::new(CpuFlags::detect()).is_none()
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
    // SAFETY: (1) ISA: yukarıda `Avx512IclToken::new(CpuFlags::detect())` doğrulandı;
    // AVX512ICL çalışma zamanında mevcuttur. (2) sınır: `sgr_sozlesmesi::<8>` `dst`,
    // `left` ve `lpf` dilimlerinin sınırlarını O(1) doğruladı; `pad_row_8bpc` yalnız
    // izinli pikselleri yığın tamponuna kopyalar ve çıktı yalnız `[0, w)` aralığına
    // yazılır. (3) hizalama/takma ad: tüm vektör yükleme ve saklamaları `mem`
    // içindeki hizasız sabit boyutlu dizi yardımcılarıyla yapılır.
    unsafe {
        match SGR_IDX {
            0 => sgr_5x5_avx512_giris(dst, &c, h, s0, w0),
            1 => sgr_3x3_avx512_giris(dst, &c, h, s1, w1),
            _ => sgr_mix_avx512_giris(dst, &c, h, s0, s1, w0, w1),
        }
    }
}

// ---------------------------------------------------------------------------
// Wiener AVX512ICL (512-bit, 32 piksel / adım)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn perm_interleave_idx() -> __m512i {
    _mm512_setr_epi64(0, 4, 1, 5, 2, 6, 3, 7)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn wiener_filter_h_avx512(
    out: &mut [i16; HOR_STRIDE],
    c: &LrCtx<'_>,
    dst: &[u8],
    row: RowRef,
    fh: &[i16; 8],
) {
    let mut pad = [0u8; PAD_ROW_LEN];
    pad_row_8bpc::<3>(&mut pad, c, dst, row);

    let z = _mm512_setzero_si512();
    let c06 = _mm512_set1_epi32((i32::from(fh[0]) & 0xffff) | ((i32::from(fh[6]) & 0xffff) << 16));
    let c15 = _mm512_set1_epi32((i32::from(fh[1]) & 0xffff) | ((i32::from(fh[5]) & 0xffff) << 16));
    let c24 = _mm512_set1_epi32((i32::from(fh[2]) & 0xffff) | ((i32::from(fh[4]) & 0xffff) << 16));
    let c3 = _mm512_set1_epi32(((i32::from(fh[3]) + 128) & 0xffff) | (16388 << 16));
    let one16 = _mm512_set1_epi16(1);
    let max_h = _mm512_set1_epi32(8191);

    let mut x = 0usize;
    while x < c.w {
        let p0 = _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(&pad, x)));
        let p1 = _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(&pad, x + 1)));
        let p2 = _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(&pad, x + 2)));
        let p3 = _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(&pad, x + 3)));
        let p4 = _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(&pad, x + 4)));
        let p5 = _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(&pad, x + 5)));
        let p6 = _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(&pad, x + 6)));

        let s_lo = _mm512_srai_epi32::<3>(_mm512_add_epi32(
            _mm512_add_epi32(
                _mm512_madd_epi16(_mm512_unpacklo_epi16(p0, p6), c06),
                _mm512_madd_epi16(_mm512_unpacklo_epi16(p1, p5), c15),
            ),
            _mm512_add_epi32(
                _mm512_madd_epi16(_mm512_unpacklo_epi16(p2, p4), c24),
                _mm512_madd_epi16(_mm512_unpacklo_epi16(p3, one16), c3),
            ),
        ));
        let s_hi = _mm512_srai_epi32::<3>(_mm512_add_epi32(
            _mm512_add_epi32(
                _mm512_madd_epi16(_mm512_unpackhi_epi16(p0, p6), c06),
                _mm512_madd_epi16(_mm512_unpackhi_epi16(p1, p5), c15),
            ),
            _mm512_add_epi32(
                _mm512_madd_epi16(_mm512_unpackhi_epi16(p2, p4), c24),
                _mm512_madd_epi16(_mm512_unpackhi_epi16(p3, one16), c3),
            ),
        ));

        let s_lo = _mm512_min_epi32(_mm512_max_epi32(s_lo, z), max_h);
        let s_hi = _mm512_min_epi32(_mm512_max_epi32(s_hi, z), max_h);
        let r = _mm512_packs_epi32(s_lo, s_hi);
        mem::store_i16x32(mem::chunk_mut::<i16, 32>(out, x), r);
        x += 32;
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn wiener_filter_v_row_avx512(
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
    let z = _mm512_setzero_si512();
    let c06 = _mm512_set1_epi32((i32::from(fv[0]) & 0xffff) | ((i32::from(fv[6]) & 0xffff) << 16));
    let c15 = _mm512_set1_epi32((i32::from(fv[1]) & 0xffff) | ((i32::from(fv[5]) & 0xffff) << 16));
    let c24 = _mm512_set1_epi32((i32::from(fv[2]) & 0xffff) | ((i32::from(fv[4]) & 0xffff) << 16));
    let c3 = _mm512_set1_epi32(i32::from(fv[3]) & 0xffff);
    let bias = _mm512_set1_epi32(-261120);

    let mut out_row = [0u8; HOR_STRIDE];
    let mut x = 0usize;
    while x < c.w {
        let v0 = mem::load_i16x32(mem::chunk::<i16, 32>(r0, x));
        let v1 = mem::load_i16x32(mem::chunk::<i16, 32>(r1, x));
        let v2 = mem::load_i16x32(mem::chunk::<i16, 32>(r2, x));
        let v3 = mem::load_i16x32(mem::chunk::<i16, 32>(r3, x));
        let v4 = mem::load_i16x32(mem::chunk::<i16, 32>(r4, x));
        let v5 = mem::load_i16x32(mem::chunk::<i16, 32>(r5, x));
        let v6 = mem::load_i16x32(mem::chunk::<i16, 32>(r6, x));

        let s_lo = _mm512_srai_epi32::<11>(_mm512_add_epi32(
            _mm512_add_epi32(
                bias,
                _mm512_add_epi32(
                    _mm512_madd_epi16(_mm512_unpacklo_epi16(v0, v6), c06),
                    _mm512_madd_epi16(_mm512_unpacklo_epi16(v1, v5), c15),
                ),
            ),
            _mm512_add_epi32(
                _mm512_madd_epi16(_mm512_unpacklo_epi16(v2, v4), c24),
                _mm512_madd_epi16(_mm512_unpacklo_epi16(v3, z), c3),
            ),
        ));
        let s_hi = _mm512_srai_epi32::<11>(_mm512_add_epi32(
            _mm512_add_epi32(
                bias,
                _mm512_add_epi32(
                    _mm512_madd_epi16(_mm512_unpackhi_epi16(v0, v6), c06),
                    _mm512_madd_epi16(_mm512_unpackhi_epi16(v1, v5), c15),
                ),
            ),
            _mm512_add_epi32(
                _mm512_madd_epi16(_mm512_unpackhi_epi16(v2, v4), c24),
                _mm512_madd_epi16(_mm512_unpackhi_epi16(v3, z), c3),
            ),
        ));

        let max_v = _mm512_set1_epi32(255);
        let s_lo = _mm512_min_epi32(_mm512_max_epi32(s_lo, z), max_v);
        let s_hi = _mm512_min_epi32(_mm512_max_epi32(s_hi, z), max_v);
        let packed16 = _mm512_packs_epi32(s_lo, s_hi);
        let bytes = _mm512_cvtusepi16_epi8(packed16);
        mem::store_u8x32(mem::chunk_mut::<u8, 32>(&mut out_row, x), bytes);
        x += 32;
    }

    let p = p_row * c.stride;
    dst[p..p + c.w].copy_from_slice(&out_row[..c.w]);
}

struct WienerAvx512State {
    hor: [[i16; HOR_STRIDE]; 6],
    tmp: [i16; HOR_STRIDE],
    ptrs: [usize; 7],
}

impl WienerAvx512State {
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    fn filter_h(&mut self, c: &LrCtx<'_>, dst: &[u8], row: RowRef, r: usize, fh: &[i16; 8]) {
        wiener_filter_h_avx512(&mut self.hor[r], c, dst, row, fh);
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    fn filter_v(&mut self, dst: &mut [u8], c: &LrCtx<'_>, p_row: usize, fv: &[i16; 8]) {
        wiener_filter_v_row_avx512(
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

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    fn filter_hv(
        &mut self,
        dst: &mut [u8],
        c: &LrCtx<'_>,
        p_row: usize,
        row: RowRef,
        filter: &[[i16; 8]; 2],
    ) {
        wiener_filter_h_avx512(&mut self.tmp, c, dst, row, &filter[0]);
        wiener_filter_v_row_avx512(
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
        let n = (c.w + 31) & !31;
        self.hor[dst_row][..n].copy_from_slice(&self.tmp[..n]);
        self.ptrs.copy_within(1..7, 0);
        self.ptrs[6] = self.ptrs[0];
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn wiener_avx512_giris(dst: &mut [u8], c: &LrCtx<'_>, h: usize, filter: &[[i16; 8]; 2]) {
    let fh = &filter[0];
    let fv = &filter[1];
    let mut st = WienerAvx512State {
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
            st.filter_h(c, dst, RowRef::Lpf(0), 0, fh);
            st.filter_h(c, dst, RowRef::Lpf(1), 1, fh);
            st.filter_h(c, dst, RowRef::Main(src_row), 2, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 1;
            }
            st.ptrs[4] = 3;
            st.ptrs[5] = 3;
            st.filter_h(c, dst, RowRef::Main(src_row), 3, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 2;
            }
            st.ptrs[5] = 4;
            st.filter_h(c, dst, RowRef::Main(src_row), 4, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
        } else {
            st.ptrs[..6].copy_from_slice(&[0; 6]);
            st.filter_h(c, dst, RowRef::Main(src_row), 0, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 1;
            }
            st.ptrs[4] = 1;
            st.ptrs[5] = 1;
            st.filter_h(c, dst, RowRef::Main(src_row), 1, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 2;
            }
            st.ptrs[5] = 2;
            st.filter_h(c, dst, RowRef::Main(src_row), 2, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
            st.ptrs[6] = 3;
            st.filter_hv(dst, c, p_row, RowRef::Main(src_row), filter);
            src_row += 1;
            p_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
            st.ptrs[6] = 4;
            st.filter_hv(dst, c, p_row, RowRef::Main(src_row), filter);
            src_row += 1;
            p_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
        }

        st.ptrs[6] = st.ptrs[5] + 1;
        loop {
            st.filter_hv(dst, c, p_row, RowRef::Main(src_row), filter);
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
        st.filter_hv(dst, c, p_row, RowRef::Lpf(6), filter);
        p_row += 1;
        st.filter_hv(dst, c, p_row, RowRef::Lpf(7), filter);
        p_row += 1;
        1
    };

    for _ in 0..tail {
        st.filter_v(dst, c, p_row, fv);
        p_row += 1;
    }
}

// ---------------------------------------------------------------------------
// SGR AVX512ICL (512-bit, 16 i32 / 16..32 u8)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn box3_row_h_avx512(
    sumsq: &mut [i32; SGR_BUF_STRIDE],
    sum: &mut [i32; SGR_BUF_STRIDE],
    pad: &[u8; PAD_ROW_LEN],
    w: usize,
) {
    let z = _mm512_setzero_si512();
    let perm = perm_interleave_idx();
    let mut j = 0usize;
    while j < w + 2 {
        let a = _mm512_permutexvar_epi64(
            perm,
            _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(pad, j + 1))),
        );
        let b = _mm512_permutexvar_epi64(
            perm,
            _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(pad, j + 2))),
        );
        let c = _mm512_permutexvar_epi64(
            perm,
            _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(pad, j + 3))),
        );

        let a_lo = _mm512_unpacklo_epi16(a, z);
        let a_hi = _mm512_unpackhi_epi16(a, z);
        let b_lo = _mm512_unpacklo_epi16(b, z);
        let b_hi = _mm512_unpackhi_epi16(b, z);
        let c_lo = _mm512_unpacklo_epi16(c, z);
        let c_hi = _mm512_unpackhi_epi16(c, z);

        let s_lo = _mm512_add_epi32(_mm512_add_epi32(a_lo, b_lo), c_lo);
        let s_hi = _mm512_add_epi32(_mm512_add_epi32(a_hi, b_hi), c_hi);

        let ab_lo = _mm512_unpacklo_epi16(a, b);
        let ab_hi = _mm512_unpackhi_epi16(a, b);
        let sq_lo = _mm512_add_epi32(
            _mm512_madd_epi16(ab_lo, ab_lo),
            _mm512_madd_epi16(c_lo, c_lo),
        );
        let sq_hi = _mm512_add_epi32(
            _mm512_madd_epi16(ab_hi, ab_hi),
            _mm512_madd_epi16(c_hi, c_hi),
        );

        mem::store_i32x16(mem::chunk_mut::<i32, 16>(sum, j), s_lo);
        mem::store_i32x16(mem::chunk_mut::<i32, 16>(sumsq, j), sq_lo);
        if j + 16 < SGR_BUF_STRIDE {
            mem::store_i32x16(mem::chunk_mut::<i32, 16>(sum, j + 16), s_hi);
            mem::store_i32x16(mem::chunk_mut::<i32, 16>(sumsq, j + 16), sq_hi);
        }
        j += 32;
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn box5_row_h_avx512(
    sumsq: &mut [i32; SGR_BUF_STRIDE],
    sum: &mut [i32; SGR_BUF_STRIDE],
    pad: &[u8; PAD_ROW_LEN],
    w: usize,
) {
    let z = _mm512_setzero_si512();
    let perm = perm_interleave_idx();
    let mut j = 0usize;
    while j < w + 2 {
        let a = _mm512_permutexvar_epi64(
            perm,
            _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(pad, j))),
        );
        let b = _mm512_permutexvar_epi64(
            perm,
            _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(pad, j + 1))),
        );
        let c = _mm512_permutexvar_epi64(
            perm,
            _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(pad, j + 2))),
        );
        let d = _mm512_permutexvar_epi64(
            perm,
            _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(pad, j + 3))),
        );
        let e = _mm512_permutexvar_epi64(
            perm,
            _mm512_cvtepu8_epi16(mem::load_u8x32(mem::chunk::<u8, 32>(pad, j + 4))),
        );

        let a_lo = _mm512_unpacklo_epi16(a, z);
        let a_hi = _mm512_unpackhi_epi16(a, z);
        let b_lo = _mm512_unpacklo_epi16(b, z);
        let b_hi = _mm512_unpackhi_epi16(b, z);
        let c_lo = _mm512_unpacklo_epi16(c, z);
        let c_hi = _mm512_unpackhi_epi16(c, z);
        let d_lo = _mm512_unpacklo_epi16(d, z);
        let d_hi = _mm512_unpackhi_epi16(d, z);
        let e_lo = _mm512_unpacklo_epi16(e, z);
        let e_hi = _mm512_unpackhi_epi16(e, z);

        let s_lo = _mm512_add_epi32(
            _mm512_add_epi32(_mm512_add_epi32(a_lo, b_lo), _mm512_add_epi32(c_lo, d_lo)),
            e_lo,
        );
        let s_hi = _mm512_add_epi32(
            _mm512_add_epi32(_mm512_add_epi32(a_hi, b_hi), _mm512_add_epi32(c_hi, d_hi)),
            e_hi,
        );

        let ab_lo = _mm512_unpacklo_epi16(a, b);
        let ab_hi = _mm512_unpackhi_epi16(a, b);
        let cd_lo = _mm512_unpacklo_epi16(c, d);
        let cd_hi = _mm512_unpackhi_epi16(c, d);
        let sq_lo = _mm512_add_epi32(
            _mm512_add_epi32(
                _mm512_madd_epi16(ab_lo, ab_lo),
                _mm512_madd_epi16(cd_lo, cd_lo),
            ),
            _mm512_madd_epi16(e_lo, e_lo),
        );
        let sq_hi = _mm512_add_epi32(
            _mm512_add_epi32(
                _mm512_madd_epi16(ab_hi, ab_hi),
                _mm512_madd_epi16(cd_hi, cd_hi),
            ),
            _mm512_madd_epi16(e_hi, e_hi),
        );

        mem::store_i32x16(mem::chunk_mut::<i32, 16>(sum, j), s_lo);
        mem::store_i32x16(mem::chunk_mut::<i32, 16>(sumsq, j), sq_lo);
        if j + 16 < SGR_BUF_STRIDE {
            mem::store_i32x16(mem::chunk_mut::<i32, 16>(sum, j + 16), s_hi);
            mem::store_i32x16(mem::chunk_mut::<i32, 16>(sumsq, j + 16), sq_hi);
        }
        j += 32;
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn box3_vert_avx512<const K: usize>(
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
        let vq = _mm512_add_epi32(
            _mm512_add_epi32(
                mem::load_i32x16(mem::chunk::<i32, 16>(q0, x)),
                mem::load_i32x16(mem::chunk::<i32, 16>(q1, x)),
            ),
            mem::load_i32x16(mem::chunk::<i32, 16>(q2, x)),
        );
        let vs = _mm512_add_epi32(
            _mm512_add_epi32(
                mem::load_i32x16(mem::chunk::<i32, 16>(s0, x)),
                mem::load_i32x16(mem::chunk::<i32, 16>(s1, x)),
            ),
            mem::load_i32x16(mem::chunk::<i32, 16>(s2, x)),
        );
        mem::store_i32x16(mem::chunk_mut::<i32, 16>(oa, x), vq);
        mem::store_i32x16(mem::chunk_mut::<i32, 16>(ob, x), vs);
        x += 16;
    }
    calc_row_ab_8bpc(oa, ob, w, s, 9, 455);
    sp.rotate_left(1);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn box5_vert_avx512(
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
        let vq = _mm512_add_epi32(
            _mm512_add_epi32(
                _mm512_add_epi32(
                    mem::load_i32x16(mem::chunk::<i32, 16>(q[0], x)),
                    mem::load_i32x16(mem::chunk::<i32, 16>(q[1], x)),
                ),
                _mm512_add_epi32(
                    mem::load_i32x16(mem::chunk::<i32, 16>(q[2], x)),
                    mem::load_i32x16(mem::chunk::<i32, 16>(q[3], x)),
                ),
            ),
            mem::load_i32x16(mem::chunk::<i32, 16>(q[4], x)),
        );
        let vs = _mm512_add_epi32(
            _mm512_add_epi32(
                _mm512_add_epi32(
                    mem::load_i32x16(mem::chunk::<i32, 16>(t[0], x)),
                    mem::load_i32x16(mem::chunk::<i32, 16>(t[1], x)),
                ),
                _mm512_add_epi32(
                    mem::load_i32x16(mem::chunk::<i32, 16>(t[2], x)),
                    mem::load_i32x16(mem::chunk::<i32, 16>(t[3], x)),
                ),
            ),
            mem::load_i32x16(mem::chunk::<i32, 16>(t[4], x)),
        );
        mem::store_i32x16(mem::chunk_mut::<i32, 16>(oa, x), vq);
        mem::store_i32x16(mem::chunk_mut::<i32, 16>(ob, x), vs);
        x += 16;
    }
    calc_row_ab_8bpc(oa, ob, w, s, 25, 164);
    sp.rotate_left(2);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn finish_filter_row1_avx512<const K: usize>(
    tmp: &mut [i32],
    dst: &[u8],
    row_off: usize,
    ab: &SgrRows<K>,
    ap: [usize; 3],
    w: usize,
) {
    let (b0, b1, b2) = (&ab.s[ap[0]], &ab.s[ap[1]], &ab.s[ap[2]]);
    let (a0, a1, a2) = (&ab.sq[ap[0]], &ab.sq[ap[1]], &ab.sq[ap[2]]);
    let c4 = _mm512_set1_epi32(4);
    let c3 = _mm512_set1_epi32(3);
    let rnd = _mm512_set1_epi32(1 << 8);
    let mut src_row = [0u8; HOR_STRIDE];
    src_row[..w].copy_from_slice(&dst[row_off..row_off + w]);

    let mut i = 0usize;
    while i < w {
        let src_v = _mm512_cvtepu8_epi32(mem::load_u8x16(mem::chunk::<u8, 16>(&src_row, i)));
        let j = i + 1;
        let b_cross = _mm512_add_epi32(
            _mm512_add_epi32(
                mem::load_i32x16(mem::chunk::<i32, 16>(b1, j)),
                mem::load_i32x16(mem::chunk::<i32, 16>(b1, j - 1)),
            ),
            _mm512_add_epi32(
                mem::load_i32x16(mem::chunk::<i32, 16>(b1, j + 1)),
                _mm512_add_epi32(
                    mem::load_i32x16(mem::chunk::<i32, 16>(b0, j)),
                    mem::load_i32x16(mem::chunk::<i32, 16>(b2, j)),
                ),
            ),
        );
        let b_diag = _mm512_add_epi32(
            _mm512_add_epi32(
                mem::load_i32x16(mem::chunk::<i32, 16>(b0, j - 1)),
                mem::load_i32x16(mem::chunk::<i32, 16>(b2, j - 1)),
            ),
            _mm512_add_epi32(
                mem::load_i32x16(mem::chunk::<i32, 16>(b0, j + 1)),
                mem::load_i32x16(mem::chunk::<i32, 16>(b2, j + 1)),
            ),
        );
        let a_v = _mm512_add_epi32(
            _mm512_mullo_epi32(b_cross, c4),
            _mm512_mullo_epi32(b_diag, c3),
        );

        let a_cross = _mm512_add_epi32(
            _mm512_add_epi32(
                mem::load_i32x16(mem::chunk::<i32, 16>(a1, j)),
                mem::load_i32x16(mem::chunk::<i32, 16>(a1, j - 1)),
            ),
            _mm512_add_epi32(
                mem::load_i32x16(mem::chunk::<i32, 16>(a1, j + 1)),
                _mm512_add_epi32(
                    mem::load_i32x16(mem::chunk::<i32, 16>(a0, j)),
                    mem::load_i32x16(mem::chunk::<i32, 16>(a2, j)),
                ),
            ),
        );
        let a_diag = _mm512_add_epi32(
            _mm512_add_epi32(
                mem::load_i32x16(mem::chunk::<i32, 16>(a0, j - 1)),
                mem::load_i32x16(mem::chunk::<i32, 16>(a2, j - 1)),
            ),
            _mm512_add_epi32(
                mem::load_i32x16(mem::chunk::<i32, 16>(a0, j + 1)),
                mem::load_i32x16(mem::chunk::<i32, 16>(a2, j + 1)),
            ),
        );
        let b_v = _mm512_add_epi32(
            _mm512_mullo_epi32(a_cross, c4),
            _mm512_mullo_epi32(a_diag, c3),
        );

        let prod = _mm512_mullo_epi32(a_v, src_v);
        let res = _mm512_srai_epi32::<9>(_mm512_add_epi32(_mm512_sub_epi32(b_v, prod), rnd));
        mem::store_i32x16(mem::chunk_mut::<i32, 16>(tmp, i), res);
        i += 16;
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn finish_filter2_avx512(
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
    let c6 = _mm512_set1_epi32(6);
    let c5 = _mm512_set1_epi32(5);
    let rnd0 = _mm512_set1_epi32(1 << 8);
    let mut src0_row = [0u8; HOR_STRIDE];
    src0_row[..w].copy_from_slice(&dst[r0_off..r0_off + w]);

    let mut i = 0usize;
    while i < w {
        let src_v = _mm512_cvtepu8_epi32(mem::load_u8x16(mem::chunk::<u8, 16>(&src0_row, i)));
        let j = i + 1;
        let b_mid = _mm512_add_epi32(
            mem::load_i32x16(mem::chunk::<i32, 16>(b0, j)),
            mem::load_i32x16(mem::chunk::<i32, 16>(b1, j)),
        );
        let b_side = _mm512_add_epi32(
            _mm512_add_epi32(
                mem::load_i32x16(mem::chunk::<i32, 16>(b0, j - 1)),
                mem::load_i32x16(mem::chunk::<i32, 16>(b1, j - 1)),
            ),
            _mm512_add_epi32(
                mem::load_i32x16(mem::chunk::<i32, 16>(b0, j + 1)),
                mem::load_i32x16(mem::chunk::<i32, 16>(b1, j + 1)),
            ),
        );
        let a_v = _mm512_add_epi32(
            _mm512_mullo_epi32(b_mid, c6),
            _mm512_mullo_epi32(b_side, c5),
        );

        let a_mid = _mm512_add_epi32(
            mem::load_i32x16(mem::chunk::<i32, 16>(a0, j)),
            mem::load_i32x16(mem::chunk::<i32, 16>(a1, j)),
        );
        let a_side = _mm512_add_epi32(
            _mm512_add_epi32(
                mem::load_i32x16(mem::chunk::<i32, 16>(a0, j - 1)),
                mem::load_i32x16(mem::chunk::<i32, 16>(a1, j - 1)),
            ),
            _mm512_add_epi32(
                mem::load_i32x16(mem::chunk::<i32, 16>(a0, j + 1)),
                mem::load_i32x16(mem::chunk::<i32, 16>(a1, j + 1)),
            ),
        );
        let b_v = _mm512_add_epi32(
            _mm512_mullo_epi32(a_mid, c6),
            _mm512_mullo_epi32(a_side, c5),
        );

        let prod = _mm512_mullo_epi32(a_v, src_v);
        let res = _mm512_srai_epi32::<9>(_mm512_add_epi32(_mm512_sub_epi32(b_v, prod), rnd0));
        mem::store_i32x16(mem::chunk_mut::<i32, 16>(tmp, i), res);
        i += 16;
    }

    if h <= 1 {
        return;
    }

    let rnd1 = _mm512_set1_epi32(1 << 7);
    let mut src1_row = [0u8; HOR_STRIDE];
    src1_row[..w].copy_from_slice(&dst[r1_off..r1_off + w]);

    let mut i = 0usize;
    while i < w {
        let src_v = _mm512_cvtepu8_epi32(mem::load_u8x16(mem::chunk::<u8, 16>(&src1_row, i)));
        let j = i + 1;
        let b_mid = mem::load_i32x16(mem::chunk::<i32, 16>(b1, j));
        let b_side = _mm512_add_epi32(
            mem::load_i32x16(mem::chunk::<i32, 16>(b1, j - 1)),
            mem::load_i32x16(mem::chunk::<i32, 16>(b1, j + 1)),
        );
        let a_v = _mm512_add_epi32(
            _mm512_mullo_epi32(b_mid, c6),
            _mm512_mullo_epi32(b_side, c5),
        );

        let a_mid = mem::load_i32x16(mem::chunk::<i32, 16>(a1, j));
        let a_side = _mm512_add_epi32(
            mem::load_i32x16(mem::chunk::<i32, 16>(a1, j - 1)),
            mem::load_i32x16(mem::chunk::<i32, 16>(a1, j + 1)),
        );
        let b_v = _mm512_add_epi32(
            _mm512_mullo_epi32(a_mid, c6),
            _mm512_mullo_epi32(a_side, c5),
        );

        let prod = _mm512_mullo_epi32(a_v, src_v);
        let res = _mm512_srai_epi32::<8>(_mm512_add_epi32(_mm512_sub_epi32(b_v, prod), rnd1));
        mem::store_i32x16(mem::chunk_mut::<i32, 16>(tmp, SGR_OUT_STRIDE + i), res);
        i += 16;
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn pack_i32x16_to_u8x16_avx512(d16: __m256i, v: __m512i) -> [u8; 16] {
    let v16 = _mm512_cvtsepi32_epi16(v);
    let sum16 = _mm256_add_epi16(d16, v16);
    let mut tmp16 = [0i16; 16];
    mem::store_i16x16(&mut tmp16, sum16);
    let mut out = [0u8; 16];
    for k in 0..16 {
        out[k] = tmp16[k].clamp(0, 255) as u8;
    }
    out
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn weighted_row1_avx512(dst: &mut [u8], off: usize, t1: &[i32], w: usize, w1: i32) {
    let w1_v = _mm512_set1_epi32(w1);
    let rnd = _mm512_set1_epi32(1 << 10);
    let mut io_row = [0u8; HOR_STRIDE];
    io_row[..w].copy_from_slice(&dst[off..off + w]);

    let mut i = 0usize;
    while i < w {
        let d16 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(&io_row, i)));
        let v = _mm512_srai_epi32::<11>(_mm512_add_epi32(
            _mm512_mullo_epi32(mem::load_i32x16(mem::chunk::<i32, 16>(t1, i)), w1_v),
            rnd,
        ));
        *mem::chunk_mut::<u8, 16>(&mut io_row, i) = pack_i32x16_to_u8x16_avx512(d16, v);
        i += 16;
    }

    dst[off..off + w].copy_from_slice(&io_row[..w]);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn weighted2_avx512(
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
    let w0_v = _mm512_set1_epi32(w0);
    let w1_v = _mm512_set1_epi32(w1);
    let rnd = _mm512_set1_epi32(1 << 10);

    for j in 0..h {
        let row = off + j * stride;
        let t = j * SGR_OUT_STRIDE;
        let mut io_row = [0u8; HOR_STRIDE];
        io_row[..w].copy_from_slice(&dst[row..row + w]);

        let mut i = 0usize;
        while i < w {
            let d16 = _mm256_cvtepu8_epi16(mem::load_u8x16(mem::chunk::<u8, 16>(&io_row, i)));
            let v = _mm512_srai_epi32::<11>(_mm512_add_epi32(
                _mm512_add_epi32(
                    _mm512_mullo_epi32(mem::load_i32x16(mem::chunk::<i32, 16>(t1, t + i)), w0_v),
                    _mm512_mullo_epi32(mem::load_i32x16(mem::chunk::<i32, 16>(t2, t + i)), w1_v),
                ),
                rnd,
            ));
            *mem::chunk_mut::<u8, 16>(&mut io_row, i) = pack_i32x16_to_u8x16_avx512(d16, v);
            i += 16;
        }

        dst[row..row + w].copy_from_slice(&io_row[..w]);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn finish1_avx512(
    dst: &mut [u8],
    c: &LrCtx<'_>,
    dst_row: &mut usize,
    ab: &SgrRows<3>,
    ap: &mut [usize; 3],
    tmp: &mut [i32; 2 * SGR_OUT_STRIDE],
    w1: i32,
) {
    let r = *dst_row;
    finish_filter_row1_avx512::<3>(tmp, dst, r * c.stride, ab, *ap, c.w);
    weighted_row1_avx512(dst, r * c.stride, &tmp[..], c.w, w1);
    *dst_row += 1;
    ap.rotate_left(1);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn finish2_avx512(
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
    finish_filter2_avx512(tmp, dst, r * c.stride, r1 * c.stride, ab, *ap, c.w, h);
    weighted_row1_avx512(dst, r * c.stride, &tmp[..], c.w, w0);
    *dst_row += 1;
    if h > 1 {
        weighted_row1_avx512(dst, (r + 1) * c.stride, &tmp[SGR_OUT_STRIDE..], c.w, w0);
        *dst_row += 1;
    }
    ap.rotate_left(1);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn finish_mix_avx512(
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
    finish_filter2_avx512(tmp5, dst, r * c.stride, r1 * c.stride, ab5, *ap5, c.w, h);
    finish_filter_row1_avx512::<4>(
        &mut tmp3[..],
        dst,
        r * c.stride,
        ab3,
        [ap3[0], ap3[1], ap3[2]],
        c.w,
    );
    if h > 1 {
        finish_filter_row1_avx512::<4>(
            &mut tmp3[SGR_OUT_STRIDE..],
            dst,
            (r + 1) * c.stride,
            ab3,
            [ap3[1], ap3[2], ap3[3]],
            c.w,
        );
    }
    weighted2_avx512(dst, r * c.stride, c.stride, tmp5, tmp3, c.w, h, w0, w1);
    *dst_row += h;
    ap5.rotate_left(1);
    ap3.rotate_left(1);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn sgr_3x3_avx512_giris(dst: &mut [u8], c: &LrCtx<'_>, h: usize, s1: u32, w1: i32) {
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
            box3_row_h_avx512(&mut sums.sq[$r], &mut sums.s[$r], &pad, w);
        }};
    }

    let kuyruk = 'akis: {
        if c.edges.contains(LrEdgeFlags::HAVE_TOP) {
            sp = [0, 1, 2];
            row_h!(RowRef::Lpf(0), 0);
            row_h!(RowRef::Lpf(1), 1);
            row_h!(RowRef::Main(sr), sp[2]);
            box3_vert_avx512::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            row_h!(RowRef::Main(sr), sp[2]);
            box3_vert_avx512::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
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
            box3_vert_avx512::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp[2] = 1;
            row_h!(RowRef::Main(sr), sp[2]);
            box3_vert_avx512::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
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
            box3_vert_avx512::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            finish1_avx512(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
            h -= 1;
            if h == 0 {
                break;
            }
        }

        if !c.edges.contains(LrEdgeFlags::HAVE_BOTTOM) {
            break 'akis Kuyruk::Vert2;
        }
        row_h!(RowRef::Lpf(6), sp[2]);
        box3_vert_avx512::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
        finish1_avx512(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        row_h!(RowRef::Lpf(7), sp[2]);
        box3_vert_avx512::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
        finish1_avx512(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        Kuyruk::Bitti
    };

    match kuyruk {
        Kuyruk::Bitti => return,
        Kuyruk::Vert2 => {
            sp[2] = sp[1];
            box3_vert_avx512::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            finish1_avx512(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        }
        Kuyruk::Vert1 => {
            sp[2] = sp[1];
            box3_vert_avx512::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            ap.rotate_left(1);
        }
    }
    sp[2] = sp[1];
    box3_vert_avx512::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
    finish1_avx512(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn sgr_5x5_avx512_giris(dst: &mut [u8], c: &LrCtx<'_>, h: usize, s0: u32, w0: i32) {
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
            box5_row_h_avx512(&mut sums.sq[$r], &mut sums.s[$r], &pad, w);
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
            box5_vert_avx512(&sums, &mut sp, &mut ab, ap[1], w, s0);
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
            box5_vert_avx512(&sums, &mut sp, &mut ab, ap[1], w, s0);
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
            box5_vert_avx512(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2_avx512(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
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
            box5_vert_avx512(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2_avx512(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
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
            box5_vert_avx512(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2_avx512(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
            true
        }
        Kuyruk::Vert1 => {
            sp[4] = sp[3];
            box5_vert_avx512(&sums, &mut sp, &mut ab, ap[1], w, s0);
            ap.rotate_left(1);
            true
        }
    };
    if output_1 {
        sp[3] = sp[2];
        sp[4] = sp[2];
        box5_vert_avx512(&sums, &mut sp, &mut ab, ap[1], w, s0);
        finish2_avx512(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 1, w0);
    } else {
        box5_vert_avx512(&sums, &mut sp, &mut ab, ap[1], w, s0);
        finish2_avx512(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn sgr_mix_avx512_giris(
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
            box3_row_h_avx512(&mut sums3.sq[$r3], &mut sums3.s[$r3], &pad, w);
            box5_row_h_avx512(&mut sums5.sq[$r5], &mut sums5.s[$r5], &pad, w);
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
            box3_vert_avx512::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            row_h35!(RowRef::Main(sr), sp3[2], 3);
            sr += 1;
            box5_vert_avx512(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            ap5.rotate_left(1);
            box3_vert_avx512::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
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
            box3_vert_avx512::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp5[4] = 1;
            sp3[2] = 1;
            row_h35!(RowRef::Main(sr), 1, 1);
            sr += 1;
            box5_vert_avx512(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            ap5.rotate_left(1);
            box3_vert_avx512::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
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
            box3_vert_avx512::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            row_h35!(RowRef::Main(sr), sp3[2], 3);
            sr += 1;
            box5_vert_avx512(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert_avx512::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix_avx512(
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
            box3_vert_avx512::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            row_h35!(RowRef::Main(sr), sp3[2], sp5[4]);
            sr += 1;
            box5_vert_avx512(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert_avx512::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix_avx512(
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
        box3_vert_avx512::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
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
            box3_vert_avx512::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            sp3[2] = sp3[1];
            false
        }
        Kuyruk::Odd => {
            sp5[4] = sp5[3];
            sp3[2] = sp3[1];
            box5_vert_avx512(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert_avx512::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix_avx512(
                dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 2, w0, w1,
            );
            true
        }
        Kuyruk::Vert1 => {
            sp5[4] = sp5[3];
            sp3[2] = sp3[1];
            box5_vert_avx512(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            ap5.rotate_left(1);
            box3_vert_avx512::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            true
        }
    };
    if output_1 {
        sp5[3] = sp5[2];
        sp5[4] = sp5[2];
        sp3[2] = sp3[1];
        box5_vert_avx512(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
        box3_vert_avx512::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
        ap3.rotate_left(1);
        finish_mix_avx512(
            dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 1, w0, w1,
        );
    } else {
        box5_vert_avx512(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
        box3_vert_avx512::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
        finish_mix_avx512(
            dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 2, w0, w1,
        );
    }
}
