//! 8 bpc loop restoration (Wiener) x86_64 SSE2 çekirdekleri (`D02749`, `D02755`;
//! P46 Adım 3 / Paket A).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Kayıt | Yuva | Sembol | Kaynak |
//! |---|---|---|---|
//! | `D02749` | `wiener[0]` | `dav1d_wiener_filter7_8bpc_sse2` | `src/x86/looprestoration_sse.asm:257-579`, `src/x86/looprestoration.h:48` |
//! | `D02755` | `wiener[1]` | `dav1d_wiener_filter5_8bpc_sse2` | `src/x86/looprestoration_sse.asm:257-579`, `src/x86/looprestoration.h:49` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bu modülde tüm SIMD işlevleri yalnız `#[target_feature(enable = "sse2")]`
//!   taşır; SSE3+, VEX ve EVEX yönergesi üretilmez (`0` SSE3+/VEX/EVEX).
//! - Kopyasız O(1) sözleşme: girişte [`wiener_sozlesmesi::<8>`] ve
//!   `Sse2Token::new(CpuFlags::detect())` denetlenir; sözleşme dışı girdide
//!   [`wiener_skaler_devret::<8>`] ile skaler gövdeye devredilir.
//! - Sınır-güvenli (`SentinelBuffer` / `OverReadProbe`): her kaynak satırı
//!   yalnız sözleşmenin izin verdiği pikseller okunarak yığın üzerindeki
//!   dolgulu `[u8; 420]` tamponuna hazırlanır; çıktı da yalnız `[0, w)`
//!   sütunlarına yazılır.

#![allow(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_and_si128, _mm_andnot_si128, _mm_cmpgt_epi32, _mm_madd_epi16,
    _mm_max_epi16, _mm_min_epi16, _mm_or_si128, _mm_packs_epi32, _mm_packus_epi16, _mm_set1_epi16,
    _mm_set1_epi32, _mm_setzero_si128, _mm_srai_epi32, _mm_unpackhi_epi8, _mm_unpackhi_epi16,
    _mm_unpacklo_epi8, _mm_unpacklo_epi16,
};

use crate::cpu::{CpuFlags, Sse2Token};
use crate::dsp::lr::{
    LrBitDepth, LrEdgeFlags, LrFilterFn, LrParams, wiener_skaler_devret, wiener_sozlesmesi,
};
use crate::dsp::x86::mem;

/// `REST_UNIT_STRIDE` (`384 + 16 = 400`, 16'nın katı).
pub(crate) const HOR_STRIDE: usize = 400;
/// Yığın üzerindeki dolgulu kaynak satır boyu (`3 + 384 + 33 = 420`).
pub(crate) const PAD_ROW_LEN: usize = 420;

/// Kaynak satır referansı (`dst` ya da `lpf`).
#[derive(Clone, Copy)]
pub(crate) enum RowRef {
    Main(usize),
    Lpf(usize),
}

/// Değişmeyen çağrı bağlamı.
pub(crate) struct LrCtx<'l> {
    pub(crate) left: &'l [u8],
    pub(crate) lpf: &'l [u8],
    pub(crate) stride: usize,
    pub(crate) lpf_stride: usize,
    pub(crate) lpf_x: usize,
    pub(crate) w: usize,
    pub(crate) edges: LrEdgeFlags,
}

/// Kaynak satırı yalnız sözleşmenin izin verdiği pikselleri okuyarak `[u8; PAD_ROW_LEN]`
/// tamponuna kopyalar (`pad[3 + x]` = `src[x]`, `x ∈ -reach .. w + reach`).
/// Sağ kuyrukta SIMD geniş yüklemeleri için `32` bayt daha son pikselle doldurulur.
#[inline(always)]
pub(crate) fn pad_row_8bpc<const REACH: usize>(
    pad: &mut [u8; PAD_ROW_LEN],
    c: &LrCtx<'_>,
    dst: &[u8],
    row: RowRef,
) {
    let w = c.w;
    let have_left = c.edges.contains(LrEdgeFlags::HAVE_LEFT);
    let have_right = c.edges.contains(LrEdgeFlags::HAVE_RIGHT);
    let rreach = if have_right { REACH } else { 0 };

    match row {
        RowRef::Main(r) => {
            let base = r * c.stride;
            let n = w + rreach;
            pad[3..3 + n].copy_from_slice(&dst[base..base + n]);
            if have_left {
                let lbase = r * 4;
                for k in 1..=REACH {
                    pad[3 - k] = c.left[lbase + 4 - k];
                }
            } else {
                let p0 = pad[3];
                for k in 1..=REACH {
                    pad[3 - k] = p0;
                }
            }
        }
        RowRef::Lpf(k) => {
            let base = k * c.lpf_stride + c.lpf_x;
            let lreach = if have_left { REACH } else { 0 };
            let n = lreach + w + rreach;
            pad[3 - lreach..3 + w + rreach]
                .copy_from_slice(&c.lpf[base - lreach..base - lreach + n]);
            if !have_left {
                let p0 = pad[3];
                for j in 1..=REACH {
                    pad[3 - j] = p0;
                }
            }
        }
    }

    let p_last = pad[3 + w - 1];
    if !have_right {
        for j in 0..REACH {
            pad[3 + w + j] = p_last;
        }
    }
    let tail_start = 3 + w + REACH;
    for slot in &mut pad[tail_start..(tail_start + 32).min(PAD_ROW_LEN)] {
        *slot = p_last;
    }
}

/// 8 bpc `wiener[tap_idx]` SSE2 kurucusu (`D02749`, `D02755`).
#[must_use]
pub const fn wiener_filter_fn(_token: Sse2Token, tap_idx: usize) -> Option<LrFilterFn> {
    match tap_idx {
        0 => Some(wiener_filter7_8bpc_sse2),
        1 => Some(wiener_filter5_8bpc_sse2),
        _ => None,
    }
}

/// 8 bpc 7-dokunuşlu Wiener filtresi — SSE2 (`D02749`, `wiener[0]`).
pub fn wiener_filter7_8bpc_sse2(
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
    wiener_8bpc_sse2::<false>(
        0, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
    );
}

/// 8 bpc 5-dokunuşlu Wiener filtresi — SSE2 (`D02755`, `wiener[1]`).
pub fn wiener_filter5_8bpc_sse2(
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
    wiener_8bpc_sse2::<true>(
        1, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
    );
}

#[inline(always)]
fn wiener_8bpc_sse2<const FIVE_TAP: bool>(
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
    ) || Sse2Token::new(CpuFlags::detect()).is_none()
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
    // SAFETY: (1) ISA: yukarıda `Sse2Token::new(CpuFlags::detect())` doğrulandı;
    // SSE2 çalışma zamanında mevcuttur. (2) sınır: `wiener_sozlesmesi::<8>` `dst`,
    // `left` ve `lpf` dilimlerinin sınırlarını O(1) doğruladı; `pad_row_8bpc` yalnız
    // izinli pikselleri yığın tamponuna kopyalar ve çıktı yalnız `[0, w)` aralığına
    // yazılır. (3) hizalama/takma ad: tüm vektör yükleme ve saklamaları `mem`
    // içindeki hizasız sabit boyutlu dizi yardımcılarıyla yapılır.
    unsafe {
        wiener_sse2_giris::<FIVE_TAP>(dst, &c, h, filter);
    }
}

#[target_feature(enable = "sse2")]
#[inline]
pub(crate) fn pack32_to_14b_sse2(s0: __m128i, s1: __m128i) -> __m128i {
    let z = _mm_setzero_si128();
    let max_h = _mm_set1_epi16(8191);
    _mm_min_epi16(_mm_max_epi16(_mm_packs_epi32(s0, s1), z), max_h)
}

#[target_feature(enable = "sse2")]
#[inline]
fn wiener_filter_h_sse2<const FIVE_TAP: bool>(
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

#[target_feature(enable = "sse2")]
#[inline]
pub(crate) fn pack32_to_u8x16_sse2(s0: __m128i, s1: __m128i, s2: __m128i, s3: __m128i) -> __m128i {
    let z = _mm_setzero_si128();
    let ceil = _mm_set1_epi32(255);
    let clamp = |v: __m128i| -> __m128i {
        let m_gt = _mm_cmpgt_epi32(v, ceil);
        let c = _mm_or_si128(_mm_andnot_si128(m_gt, v), _mm_and_si128(m_gt, ceil));
        let m_pos = _mm_cmpgt_epi32(c, z);
        _mm_and_si128(m_pos, c)
    };
    let w0 = _mm_packs_epi32(clamp(s0), clamp(s1));
    let w1 = _mm_packs_epi32(clamp(s2), clamp(s3));
    _mm_packus_epi16(w0, w1)
}

#[target_feature(enable = "sse2")]
#[inline]
fn wiener_filter_v_row_sse2<const FIVE_TAP: bool>(
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

struct WienerSse2State {
    hor: [[i16; HOR_STRIDE]; 6],
    tmp: [i16; HOR_STRIDE],
    ptrs: [usize; 7],
}

impl WienerSse2State {
    #[target_feature(enable = "sse2")]
    #[inline]
    fn filter_h<const FIVE_TAP: bool>(
        &mut self,
        c: &LrCtx<'_>,
        dst: &[u8],
        row: RowRef,
        r: usize,
        fh: &[i16; 8],
    ) {
        wiener_filter_h_sse2::<FIVE_TAP>(&mut self.hor[r], c, dst, row, fh);
    }

    #[target_feature(enable = "sse2")]
    #[inline]
    fn filter_v<const FIVE_TAP: bool>(
        &mut self,
        dst: &mut [u8],
        c: &LrCtx<'_>,
        p_row: usize,
        fv: &[i16; 8],
    ) {
        wiener_filter_v_row_sse2::<FIVE_TAP>(
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

    #[target_feature(enable = "sse2")]
    #[inline]
    fn filter_hv<const FIVE_TAP: bool>(
        &mut self,
        dst: &mut [u8],
        c: &LrCtx<'_>,
        p_row: usize,
        row: RowRef,
        filter: &[[i16; 8]; 2],
    ) {
        wiener_filter_h_sse2::<FIVE_TAP>(&mut self.tmp, c, dst, row, &filter[0]);
        wiener_filter_v_row_sse2::<FIVE_TAP>(
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

#[target_feature(enable = "sse2")]
fn wiener_sse2_giris<const FIVE_TAP: bool>(
    dst: &mut [u8],
    c: &LrCtx<'_>,
    h: usize,
    filter: &[[i16; 8]; 2],
) {
    let fh = &filter[0];
    let fv = &filter[1];
    let mut st = WienerSse2State {
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
