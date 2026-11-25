//! AVX2 16-bpc (`10/12` bit) Loop Restoration (Wiener 7/5 dokunuş ve 10-bpc SGRPROJ 5×5 / 3×3 / karma)
//! çekirdekleri (`5` matris kaydı / `7` yuva adayı; P46 Adım 3 / Paket B).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Matris kaydı | Yuva | Bit derinliği | Sembol | Kaynak |
//! |---|---|---|---|---|
//! | `D02746` | `wiener[0]` | `10` + `12` | `dav1d_wiener_filter7_16bpc_avx2` | `src/x86/looprestoration16.asm:72-358`, `src/x86/looprestoration.h:46` |
//! | `D02753` | `wiener[1]` | `10` + `12` | `dav1d_wiener_filter5_16bpc_avx2` | `src/x86/looprestoration16.asm:72-358`, `src/x86/looprestoration.h:47` |
//! | `D02728` | `sgr[0]` | yalnız `10` | `dav1d_sgr_filter_5x5_16bpc_avx2` | `src/x86/looprestoration16.asm:360-990`, `src/x86/looprestoration.h:49` |
//! | `D02734` | `sgr[1]` | yalnız `10` | `dav1d_sgr_filter_3x3_16bpc_avx2` | `src/x86/looprestoration16.asm:360-990`, `src/x86/looprestoration.h:50` |
//! | `D02740` | `sgr[2]` | yalnız `10` | `dav1d_sgr_filter_mix_16bpc_avx2` | `src/x86/looprestoration16.asm:360-990`, `src/x86/looprestoration.h:51` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bütün SIMD işlevleri yalnız `#[target_feature(enable = "avx2")]` taşır;
//!   EVEX (`zmm` / `k0..k7`) yönergesi üretilmez (`0` EVEX).
//! - O(1) sözleşme: girişlerde [`wiener_sozlesmesi`] / [`sgr_sozlesmesi`] ve
//!   [`Avx2Token`] denetlenir; sözleşme dışı girdide [`wiener_skaler_devret`] /
//!   [`sgr_skaler_devret`] ile skaler gövdeye devredilir.
//! - Sınır-güvenli erişim (`0` over-read, `0` over-write): her kaynak satır yalnız
//!   sözleşmenin izin verdiği pikseller okunarak yığın üzerindeki dolgulu
//!   `[i16; 408]` tamponuna hazırlanır; çıktı yalnız `[0, w * 2)` bayt aralığına yazılır.

#![allow(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use core::arch::x86_64::{
    __m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_madd_epi16, _mm256_max_epi16,
    _mm256_max_epi32, _mm256_min_epi16, _mm256_mullo_epi32, _mm256_packs_epi32, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_setzero_si256, _mm256_srai_epi32, _mm256_sub_epi32,
    _mm256_unpackhi_epi16, _mm256_unpacklo_epi16,
};

use crate::cpu::{Avx2Token, CpuFlags};
use crate::dsp::lr::{
    LrBitDepth, LrEdgeFlags, LrFilterFn, LrParams, sgr_skaler_devret, sgr_sozlesmesi,
    wiener_skaler_devret, wiener_sozlesmesi,
};
use crate::dsp::lr_tables::SGR_X_BY_X;
use crate::dsp::x86::mem;

/// dav1d `REST_UNIT_STRIDE` (`390`) + 16'lı AVX2 adımı hizalama payı (`400`, 16'nın katı).
const REST_UNIT_STRIDE: usize = 400;
/// Dolgulu kaynak satır kapasitesi (`3 + 384 + 16 = 403`, 8'in katına yuvarlanmış `408`).
const PAD_CAP: usize = 408;
/// dav1d `BUF_STRIDE` (`384 + 16 = 400`, 16'nın katı).
const BUF_STRIDE: usize = 400;
/// dav1d `FILTER_OUT_STRIDE` (`384`, 16'nın katı).
const FILTER_OUT_STRIDE: usize = 384;

/// `wiener[0]` (7-tap, 10/12-bpc) AVX2 yuva kurucusu (`D02746`).
#[must_use]
pub const fn wiener_filter7_16bpc_avx2_fn(_token: Avx2Token) -> LrFilterFn {
    wiener_filter7_16bpc_avx2
}

/// `wiener[1]` (5-tap, 10/12-bpc) AVX2 yuva kurucusu (`D02753`).
#[must_use]
pub const fn wiener_filter5_16bpc_avx2_fn(_token: Avx2Token) -> LrFilterFn {
    wiener_filter5_16bpc_avx2
}

/// `sgr[0]` (5×5, yalnız 10-bpc) AVX2 yuva kurucusu (`D02728`).
#[must_use]
pub const fn sgr_filter_5x5_16bpc_avx2_fn(_token: Avx2Token) -> LrFilterFn {
    sgr_filter_5x5_16bpc_avx2
}

/// `sgr[1]` (3×3, yalnız 10-bpc) AVX2 yuva kurucusu (`D02734`).
#[must_use]
pub const fn sgr_filter_3x3_16bpc_avx2_fn(_token: Avx2Token) -> LrFilterFn {
    sgr_filter_3x3_16bpc_avx2
}

/// `sgr[2]` (mix, yalnız 10-bpc) AVX2 yuva kurucusu (`D02740`).
#[must_use]
pub const fn sgr_filter_mix_16bpc_avx2_fn(_token: Avx2Token) -> LrFilterFn {
    sgr_filter_mix_16bpc_avx2
}

/// `wiener[tap_idx]` (10/12-bpc) AVX2 yuva kurucusu (`D02746`, `D02753`).
#[must_use]
pub const fn wiener_16bpc_avx2_fn(_token: Avx2Token, tap_idx: usize) -> Option<LrFilterFn> {
    match tap_idx {
        0 => Some(wiener_filter7_16bpc_avx2),
        1 => Some(wiener_filter5_16bpc_avx2),
        _ => None,
    }
}

/// `sgr[sgr_idx]` (yalnız 10-bpc) AVX2 yuva kurucusu (`D02728`, `D02734`, `D02740`).
#[must_use]
pub const fn sgr_10bpc_avx2_fn(_token: Avx2Token, sgr_idx: usize) -> Option<LrFilterFn> {
    match sgr_idx {
        0 => Some(sgr_filter_5x5_16bpc_avx2),
        1 => Some(sgr_filter_3x3_16bpc_avx2),
        2 => Some(sgr_filter_mix_16bpc_avx2),
        _ => None,
    }
}

/// `dav1d_wiener_filter7_16bpc_avx2` (`D02746`, 10-bpc ve 12-bpc).
pub fn wiener_filter7_16bpc_avx2(
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
    wiener_dispatch_avx2::<0>(
        dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
    );
}

/// `dav1d_wiener_filter5_16bpc_avx2` (`D02753`, 10-bpc ve 12-bpc).
pub fn wiener_filter5_16bpc_avx2(
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
    wiener_dispatch_avx2::<1>(
        dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
    );
}

#[inline(always)]
fn wiener_dispatch_avx2<const TAP_IDX: usize>(
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
    match bd {
        LrBitDepth::B10 => {
            if !wiener_sozlesmesi::<10>(
                TAP_IDX,
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
            ) || Avx2Token::new(CpuFlags::AVX2).is_none()
            {
                wiener_skaler_devret::<10>(
                    TAP_IDX, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
                );
                return;
            }
            let LrParams::Wiener { filter } = params else {
                return;
            };
            let c = Ctx {
                left,
                lpf,
                stride,
                lpf_stride,
                lpf_x,
                w,
                edges,
            };
            // SAFETY: (1) ISA: `Avx2Token::new(CpuFlags::AVX2)` bu makinede AVX2 tespitini doğruladı.
            // (2) sınır: `wiener_sozlesmesi::<10>` `dst`, `left` ve `lpf` dilim sınırlarını O(1) doğruladı.
            // (3) hizalama/takma ad: tüm SIMD yükleme/saklamalar yığın üzerindeki sabit boyutlu dizilerde `mem` yardımcılarıyla hizasız yapılır; `dst` tek `&mut` referanstır.
            unsafe { wiener_16bpc_avx2_giris::<10, TAP_IDX>(dst, &c, h, filter) }
        }
        LrBitDepth::B12 => {
            if !wiener_sozlesmesi::<12>(
                TAP_IDX,
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
            ) || Avx2Token::new(CpuFlags::AVX2).is_none()
            {
                wiener_skaler_devret::<12>(
                    TAP_IDX, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
                );
                return;
            }
            let LrParams::Wiener { filter } = params else {
                return;
            };
            let c = Ctx {
                left,
                lpf,
                stride,
                lpf_stride,
                lpf_x,
                w,
                edges,
            };
            // SAFETY: (1) ISA: `Avx2Token::new(CpuFlags::AVX2)` bu makinede AVX2 tespitini doğruladı.
            // (2) sınır: `wiener_sozlesmesi::<12>` `dst`, `left` ve `lpf` dilim sınırlarını O(1) doğruladı.
            // (3) hizalama/takma ad: tüm SIMD yükleme/saklamalar yığın üzerindeki sabit boyutlu dizilerde `mem` yardımcılarıyla hizasız yapılır; `dst` tek `&mut` referanstır.
            unsafe { wiener_16bpc_avx2_giris::<12, TAP_IDX>(dst, &c, h, filter) }
        }
        LrBitDepth::B8 => {
            wiener_skaler_devret::<10>(
                TAP_IDX, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
            );
        }
    }
}

/// `dav1d_sgr_filter_5x5_16bpc_avx2` (`D02728`, yalnız 10-bpc).
pub fn sgr_filter_5x5_16bpc_avx2(
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
    if !sgr_sozlesmesi::<10>(
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
    ) || Avx2Token::new(CpuFlags::AVX2).is_none()
    {
        if bd == LrBitDepth::B12 {
            sgr_skaler_devret::<12>(
                0, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
            );
        } else {
            sgr_skaler_devret::<10>(
                0, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
            );
        }
        return;
    }
    let LrParams::Sgr { s0, w0, .. } = *params else {
        return;
    };
    let c = Ctx {
        left,
        lpf,
        stride,
        lpf_stride,
        lpf_x,
        w,
        edges,
    };
    // SAFETY: (1) ISA: `Avx2Token::new(CpuFlags::AVX2)` bu makinede AVX2 tespitini doğruladı.
    // (2) sınır: `sgr_sozlesmesi::<10>` `dst`, `left` ve `lpf` dilim sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: tüm SIMD yükleme/saklamalar yığın üzerindeki sabit boyutlu dizilerde `mem` yardımcılarıyla hizasız yapılır; `dst` tek `&mut` referanstır.
    unsafe { sgr_5x5_10bpc_avx2_giris(dst, &c, h, s0, w0) }
}

/// `dav1d_sgr_filter_3x3_16bpc_avx2` (`D02734`, yalnız 10-bpc).
pub fn sgr_filter_3x3_16bpc_avx2(
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
    if !sgr_sozlesmesi::<10>(
        1,
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
    ) || Avx2Token::new(CpuFlags::AVX2).is_none()
    {
        if bd == LrBitDepth::B12 {
            sgr_skaler_devret::<12>(
                1, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
            );
        } else {
            sgr_skaler_devret::<10>(
                1, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
            );
        }
        return;
    }
    let LrParams::Sgr { s1, w1, .. } = *params else {
        return;
    };
    let c = Ctx {
        left,
        lpf,
        stride,
        lpf_stride,
        lpf_x,
        w,
        edges,
    };
    // SAFETY: (1) ISA: `Avx2Token::new(CpuFlags::AVX2)` bu makinede AVX2 tespitini doğruladı.
    // (2) sınır: `sgr_sozlesmesi::<10>` `dst`, `left` ve `lpf` dilim sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: tüm SIMD yükleme/saklamalar yığın üzerindeki sabit boyutlu dizilerde `mem` yardımcılarıyla hizasız yapılır; `dst` tek `&mut` referanstır.
    unsafe { sgr_3x3_10bpc_avx2_giris(dst, &c, h, s1, w1) }
}

/// `dav1d_sgr_filter_mix_16bpc_avx2` (`D02740`, yalnız 10-bpc).
pub fn sgr_filter_mix_16bpc_avx2(
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
    if !sgr_sozlesmesi::<10>(
        2,
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
    ) || Avx2Token::new(CpuFlags::AVX2).is_none()
    {
        if bd == LrBitDepth::B12 {
            sgr_skaler_devret::<12>(
                2, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
            );
        } else {
            sgr_skaler_devret::<10>(
                2, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
            );
        }
        return;
    }
    let LrParams::Sgr { s0, s1, w0, w1 } = *params else {
        return;
    };
    let c = Ctx {
        left,
        lpf,
        stride,
        lpf_stride,
        lpf_x,
        w,
        edges,
    };
    // SAFETY: (1) ISA: `Avx2Token::new(CpuFlags::AVX2)` bu makinede AVX2 tespitini doğruladı.
    // (2) sınır: `sgr_sozlesmesi::<10>` `dst`, `left` ve `lpf` dilim sınırlarını O(1) doğruladı.
    // (3) hizalama/takma ad: tüm SIMD yükleme/saklamalar yığın üzerindeki sabit boyutlu dizilerde `mem` yardımcılarıyla hizasız yapılır; `dst` tek `&mut` referanstır.
    unsafe { sgr_mix_10bpc_avx2_giris(dst, &c, h, s0, s1, w0, w1) }
}

// ---------------------------------------------------------------------------
// Sınır-güvenli satır hazırlama ve çıktı yardımcıları
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum RowRef {
    Main(usize),
    Lpf(usize),
}

struct Ctx<'a> {
    left: &'a [u8],
    lpf: &'a [u8],
    stride: usize,
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    edges: LrEdgeFlags,
}

#[inline(always)]
fn read_u16_le(buf: &[u8], off: usize) -> i16 {
    u16::from_le_bytes([buf[off], buf[off + 1]]) as i16
}

#[inline(always)]
fn write_u16_le(buf: &mut [u8], off: usize, v: i16) {
    buf[off..off + 2].copy_from_slice(&(v as u16).to_le_bytes());
}

impl Ctx<'_> {
    #[inline(always)]
    fn stage_row(&self, dst: &[u8], row: RowRef, reach: usize, pad: &mut [i16; PAD_CAP]) {
        let w = self.w;
        let have_left = self.edges.contains(LrEdgeFlags::HAVE_LEFT);
        let have_right = self.edges.contains(LrEdgeFlags::HAVE_RIGHT);
        let rreach = if have_right { reach } else { 0 };

        match row {
            RowRef::Main(r) => {
                let base = r * self.stride;
                for x in 0..w + rreach {
                    pad[3 + x] = read_u16_le(dst, base + x * 2);
                }
                if have_left {
                    let lbase = r * 8;
                    for k in (4 - reach)..4 {
                        pad[k - 1] = read_u16_le(self.left, lbase + k * 2);
                    }
                } else {
                    let p0 = pad[3];
                    for k in 0..reach {
                        pad[3 - 1 - k] = p0;
                    }
                }
            }
            RowRef::Lpf(k) => {
                let base = k * self.lpf_stride + self.lpf_x * 2;
                let lreach = if have_left { reach } else { 0 };
                for x in 0..lreach {
                    let off = base - (lreach - x) * 2;
                    pad[3 - lreach + x] = read_u16_le(self.lpf, off);
                }
                for x in 0..w + rreach {
                    pad[3 + x] = read_u16_le(self.lpf, base + x * 2);
                }
                if !have_left {
                    let p0 = pad[3];
                    for k in 0..reach {
                        pad[3 - 1 - k] = p0;
                    }
                }
            }
        }
        if !have_right {
            let plast = pad[3 + w - 1];
            for x in 0..reach {
                pad[3 + w + x] = plast;
            }
        }
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn load_i32x8(src: &[i32; 8]) -> __m256i {
    let mut b = [0u8; 32];
    for (i, &val) in src.iter().enumerate() {
        b[4 * i..4 * i + 4].copy_from_slice(&val.to_le_bytes());
    }
    mem::load_u8x32(&b)
}

#[target_feature(enable = "avx2")]
#[inline]
fn store_row_u16_avx2(
    dst: &mut [u8],
    row_off: usize,
    w: usize,
    out_row: &[i16; FILTER_OUT_STRIDE],
) {
    let mut x = 0usize;
    while x + 16 <= w {
        let v = mem::load_i16x16(mem::chunk::<i16, 16>(out_row, x));
        mem::store_u8x32(mem::chunk_mut::<u8, 32>(dst, row_off + x * 2), v);
        x += 16;
    }
    while x + 8 <= w {
        let v = mem::load_i16x8(mem::chunk::<i16, 8>(out_row, x));
        mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row_off + x * 2), v);
        x += 8;
    }
    while x < w {
        write_u16_le(dst, row_off + x * 2, out_row[x]);
        x += 1;
    }
}

// ---------------------------------------------------------------------------
// Wiener 16-bpc (10-bpc ve 12-bpc) AVX2 çekirdekleri (16 piksel/adım)
// ---------------------------------------------------------------------------

#[inline(always)]
fn pack_f_pair(a: i16, b: i16) -> i32 {
    i32::from(a as u16) | (i32::from(b as u16) << 16)
}

#[target_feature(enable = "avx2")]
#[inline]
fn wiener_filter_h_avx2<const BPC: u32, const TAP_IDX: usize>(
    out: &mut [i16; REST_UNIT_STRIDE],
    pad: &[i16; PAD_CAP],
    fh: &[i16; 8],
    w: usize,
) {
    let f01 = _mm256_set1_epi32(pack_f_pair(fh[0], fh[1]));
    let f1z = _mm256_set1_epi32(pack_f_pair(0, fh[1]));
    let f23 = _mm256_set1_epi32(pack_f_pair(fh[2], fh[3]));
    let f45 = _mm256_set1_epi32(pack_f_pair(fh[4], fh[5]));
    let f6z = _mm256_set1_epi32(pack_f_pair(fh[6], 0));
    let _ = if TAP_IDX == 0 { f1z } else { (f01, f6z).0 };
    let base_const = if BPC == 12 {
        _mm256_set1_epi32((1 << 18) + 16)
    } else {
        _mm256_set1_epi32((1 << 16) + 4)
    };
    let zero = _mm256_setzero_si256();

    let mut x = 0usize;
    while x < w {
        let p0 = mem::load_i16x16(mem::chunk::<i16, 16>(pad, x));
        let p1 = mem::load_i16x16(mem::chunk::<i16, 16>(pad, x + 1));
        let p2 = mem::load_i16x16(mem::chunk::<i16, 16>(pad, x + 2));
        let p3 = mem::load_i16x16(mem::chunk::<i16, 16>(pad, x + 3));
        let p4 = mem::load_i16x16(mem::chunk::<i16, 16>(pad, x + 4));
        let p5 = mem::load_i16x16(mem::chunk::<i16, 16>(pad, x + 5));
        let p6 = mem::load_i16x16(mem::chunk::<i16, 16>(pad, x + 6));

        let p23_lo = _mm256_unpacklo_epi16(p2, p3);
        let p23_hi = _mm256_unpackhi_epi16(p2, p3);
        let p45_lo = _mm256_unpacklo_epi16(p4, p5);
        let p45_hi = _mm256_unpackhi_epi16(p4, p5);

        let mut acc_lo = _mm256_add_epi32(
            base_const,
            _mm256_add_epi32(
                _mm256_madd_epi16(p23_lo, f23),
                _mm256_madd_epi16(p45_lo, f45),
            ),
        );
        let mut acc_hi = _mm256_add_epi32(
            base_const,
            _mm256_add_epi32(
                _mm256_madd_epi16(p23_hi, f23),
                _mm256_madd_epi16(p45_hi, f45),
            ),
        );

        if TAP_IDX == 0 {
            let p01_lo = _mm256_unpacklo_epi16(p0, p1);
            let p01_hi = _mm256_unpackhi_epi16(p0, p1);
            let p6z_lo = _mm256_unpacklo_epi16(p6, zero);
            let p6z_hi = _mm256_unpackhi_epi16(p6, zero);
            acc_lo = _mm256_add_epi32(
                acc_lo,
                _mm256_add_epi32(
                    _mm256_madd_epi16(p01_lo, f01),
                    _mm256_madd_epi16(p6z_lo, f6z),
                ),
            );
            acc_hi = _mm256_add_epi32(
                acc_hi,
                _mm256_add_epi32(
                    _mm256_madd_epi16(p01_hi, f01),
                    _mm256_madd_epi16(p6z_hi, f6z),
                ),
            );
        } else {
            let p01_lo = _mm256_unpacklo_epi16(zero, p1);
            let p01_hi = _mm256_unpackhi_epi16(zero, p1);
            acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(p01_lo, f1z));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(p01_hi, f1z));
        }

        // Not: _mm256_unpacklo/hi_epi16 ve _mm256_packs_epi32 her ikisi de 128-bit şerit
        // içinde çalıştığı için, unpacklo (0..3, 8..11) ve unpackhi (4..7, 12..15) üzerine
        // doğrudan _mm256_packs_epi32(s_lo, s_hi) uygulamak tam doğal sırayı (0..15) geri verir!
        let (s_lo, s_hi) = if BPC == 12 {
            (
                _mm256_srai_epi32::<5>(acc_lo),
                _mm256_srai_epi32::<5>(acc_hi),
            )
        } else {
            (
                _mm256_srai_epi32::<3>(acc_lo),
                _mm256_srai_epi32::<3>(acc_hi),
            )
        };
        let packed = _mm256_max_epi16(_mm256_packs_epi32(s_lo, s_hi), zero);

        mem::store_i16x16(mem::chunk_mut::<i16, 16>(out, x), packed);
        x += 16;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn wiener_filter_v_rows_avx2<const BPC: u32, const TAP_IDX: usize>(
    dst: &mut [u8],
    row_off: usize,
    w: usize,
    r0: &[i16; REST_UNIT_STRIDE],
    r1: &[i16; REST_UNIT_STRIDE],
    r2: &[i16; REST_UNIT_STRIDE],
    r3: &[i16; REST_UNIT_STRIDE],
    r4: &[i16; REST_UNIT_STRIDE],
    r5: &[i16; REST_UNIT_STRIDE],
    r6: &[i16; REST_UNIT_STRIDE],
    fv: &[i16; 8],
) {
    let f01 = _mm256_set1_epi32(pack_f_pair(fv[0], fv[1]));
    let f1z = _mm256_set1_epi32(pack_f_pair(0, fv[1]));
    let f23 = _mm256_set1_epi32(pack_f_pair(fv[2], fv[3]));
    let f45 = _mm256_set1_epi32(pack_f_pair(fv[4], fv[5]));
    let f6z = _mm256_set1_epi32(pack_f_pair(fv[6], 0));
    let _ = if TAP_IDX == 0 { f1z } else { (f01, f6z).0 };

    let round_bits_v = if BPC == 12 { 9 } else { 11 };
    let rounding_off_v = 1i32 << (round_bits_v - 1);
    let round_offset = 1i32 << (BPC as i32 + round_bits_v - 1);
    let const_off = -round_offset + rounding_off_v;
    let v_const = _mm256_set1_epi32(const_off);
    let zero = _mm256_setzero_si256();
    let max_px = _mm256_set1_epi16(((1i32 << BPC) - 1) as i16);

    let mut out_row = [0i16; FILTER_OUT_STRIDE];
    let mut x = 0usize;
    while x < w {
        let h1 = mem::load_i16x16(mem::chunk::<i16, 16>(r1, x));
        let h2 = mem::load_i16x16(mem::chunk::<i16, 16>(r2, x));
        let h3 = mem::load_i16x16(mem::chunk::<i16, 16>(r3, x));
        let h4 = mem::load_i16x16(mem::chunk::<i16, 16>(r4, x));
        let h5 = mem::load_i16x16(mem::chunk::<i16, 16>(r5, x));

        let h23_lo = _mm256_unpacklo_epi16(h2, h3);
        let h23_hi = _mm256_unpackhi_epi16(h2, h3);
        let h45_lo = _mm256_unpacklo_epi16(h4, h5);
        let h45_hi = _mm256_unpackhi_epi16(h4, h5);

        let mut acc_lo = _mm256_add_epi32(
            v_const,
            _mm256_add_epi32(
                _mm256_madd_epi16(h23_lo, f23),
                _mm256_madd_epi16(h45_lo, f45),
            ),
        );
        let mut acc_hi = _mm256_add_epi32(
            v_const,
            _mm256_add_epi32(
                _mm256_madd_epi16(h23_hi, f23),
                _mm256_madd_epi16(h45_hi, f45),
            ),
        );

        if TAP_IDX == 0 {
            let h0 = mem::load_i16x16(mem::chunk::<i16, 16>(r0, x));
            let h6 = mem::load_i16x16(mem::chunk::<i16, 16>(r6, x));
            let h01_lo = _mm256_unpacklo_epi16(h0, h1);
            let h01_hi = _mm256_unpackhi_epi16(h0, h1);
            let h6z_lo = _mm256_unpacklo_epi16(h6, zero);
            let h6z_hi = _mm256_unpackhi_epi16(h6, zero);
            acc_lo = _mm256_add_epi32(
                acc_lo,
                _mm256_add_epi32(
                    _mm256_madd_epi16(h01_lo, f01),
                    _mm256_madd_epi16(h6z_lo, f6z),
                ),
            );
            acc_hi = _mm256_add_epi32(
                acc_hi,
                _mm256_add_epi32(
                    _mm256_madd_epi16(h01_hi, f01),
                    _mm256_madd_epi16(h6z_hi, f6z),
                ),
            );
        } else {
            let h01_lo = _mm256_unpacklo_epi16(zero, h1);
            let h01_hi = _mm256_unpackhi_epi16(zero, h1);
            acc_lo = _mm256_add_epi32(acc_lo, _mm256_madd_epi16(h01_lo, f1z));
            acc_hi = _mm256_add_epi32(acc_hi, _mm256_madd_epi16(h01_hi, f1z));
        }

        let (s_lo, s_hi) = if BPC == 12 {
            (
                _mm256_srai_epi32::<9>(acc_lo),
                _mm256_srai_epi32::<9>(acc_hi),
            )
        } else {
            (
                _mm256_srai_epi32::<11>(acc_lo),
                _mm256_srai_epi32::<11>(acc_hi),
            )
        };
        let p = _mm256_min_epi16(
            _mm256_max_epi16(_mm256_packs_epi32(s_lo, s_hi), zero),
            max_px,
        );
        mem::store_i16x16(mem::chunk_mut::<i16, 16>(&mut out_row, x), p);
        x += 16;
    }

    store_row_u16_avx2(dst, row_off, w, &out_row);
}

struct WienerStateAvx2 {
    hor: [[i16; REST_UNIT_STRIDE]; 6],
    tmp: [i16; REST_UNIT_STRIDE],
    pad: [i16; PAD_CAP],
    ptrs: [usize; 7],
}

impl WienerStateAvx2 {
    #[target_feature(enable = "avx2")]
    fn filter_h<const BPC: u32, const TAP_IDX: usize>(
        &mut self,
        c: &Ctx<'_>,
        dst: &[u8],
        row: RowRef,
        r: usize,
        fh: &[i16; 8],
    ) {
        c.stage_row(dst, row, 3, &mut self.pad);
        wiener_filter_h_avx2::<BPC, TAP_IDX>(&mut self.hor[r], &self.pad, fh, c.w);
    }

    #[target_feature(enable = "avx2")]
    fn filter_v<const BPC: u32, const TAP_IDX: usize>(
        &mut self,
        dst: &mut [u8],
        c: &Ctx<'_>,
        p_row: usize,
        fv: &[i16; 8],
    ) {
        let [p0, p1, p2, p3, p4, p5, _] = self.ptrs;
        wiener_filter_v_rows_avx2::<BPC, TAP_IDX>(
            dst,
            p_row * c.stride,
            c.w,
            &self.hor[p0],
            &self.hor[p1],
            &self.hor[p2],
            &self.hor[p3],
            &self.hor[p4],
            &self.hor[p5],
            &self.hor[p5],
            fv,
        );
        self.ptrs.copy_within(1..6, 0);
    }

    #[target_feature(enable = "avx2")]
    fn filter_hv<const BPC: u32, const TAP_IDX: usize>(
        &mut self,
        dst: &mut [u8],
        c: &Ctx<'_>,
        p_row: usize,
        row: RowRef,
        filter: &[[i16; 8]; 2],
    ) {
        c.stage_row(dst, row, 3, &mut self.pad);
        wiener_filter_h_avx2::<BPC, TAP_IDX>(&mut self.tmp, &self.pad, &filter[0], c.w);
        let [p0, p1, p2, p3, p4, p5, p6] = self.ptrs;
        wiener_filter_v_rows_avx2::<BPC, TAP_IDX>(
            dst,
            p_row * c.stride,
            c.w,
            &self.hor[p0],
            &self.hor[p1],
            &self.hor[p2],
            &self.hor[p3],
            &self.hor[p4],
            &self.hor[p5],
            &self.tmp,
            &filter[1],
        );
        self.hor[p6][..c.w].copy_from_slice(&self.tmp[..c.w]);
        self.ptrs.copy_within(1..7, 0);
        self.ptrs[6] = self.ptrs[0];
    }
}

#[target_feature(enable = "avx2")]
fn wiener_16bpc_avx2_giris<const BPC: u32, const TAP_IDX: usize>(
    dst: &mut [u8],
    c: &Ctx<'_>,
    h: usize,
    filter: &[[i16; 8]; 2],
) {
    let fh = &filter[0];
    let fv = &filter[1];
    let mut st = WienerStateAvx2 {
        hor: [[0; REST_UNIT_STRIDE]; 6],
        tmp: [0; REST_UNIT_STRIDE],
        pad: [0; PAD_CAP],
        ptrs: [0; 7],
    };
    let mut h = h;
    let mut src_row = 0usize;
    let mut p_row = 0usize;

    let tail: usize = 'akis: {
        if c.edges.contains(LrEdgeFlags::HAVE_TOP) {
            st.ptrs[..6].copy_from_slice(&[0, 0, 1, 2, 2, 2]);
            st.filter_h::<BPC, TAP_IDX>(c, dst, RowRef::Lpf(0), 0, fh);
            st.filter_h::<BPC, TAP_IDX>(c, dst, RowRef::Lpf(1), 1, fh);
            st.filter_h::<BPC, TAP_IDX>(c, dst, RowRef::Main(src_row), 2, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 1;
            }
            st.ptrs[4] = 3;
            st.ptrs[5] = 3;
            st.filter_h::<BPC, TAP_IDX>(c, dst, RowRef::Main(src_row), 3, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 2;
            }
            st.ptrs[5] = 4;
            st.filter_h::<BPC, TAP_IDX>(c, dst, RowRef::Main(src_row), 4, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
        } else {
            st.ptrs[..6].copy_from_slice(&[0; 6]);
            st.filter_h::<BPC, TAP_IDX>(c, dst, RowRef::Main(src_row), 0, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 1;
            }
            st.ptrs[4] = 1;
            st.ptrs[5] = 1;
            st.filter_h::<BPC, TAP_IDX>(c, dst, RowRef::Main(src_row), 1, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 2;
            }
            st.ptrs[5] = 2;
            st.filter_h::<BPC, TAP_IDX>(c, dst, RowRef::Main(src_row), 2, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
            st.ptrs[6] = 3;
            st.filter_hv::<BPC, TAP_IDX>(dst, c, p_row, RowRef::Main(src_row), filter);
            src_row += 1;
            p_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
            st.ptrs[6] = 4;
            st.filter_hv::<BPC, TAP_IDX>(dst, c, p_row, RowRef::Main(src_row), filter);
            src_row += 1;
            p_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
        }

        st.ptrs[6] = st.ptrs[5] + 1;
        loop {
            st.filter_hv::<BPC, TAP_IDX>(dst, c, p_row, RowRef::Main(src_row), filter);
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
        st.filter_hv::<BPC, TAP_IDX>(dst, c, p_row, RowRef::Lpf(6), filter);
        p_row += 1;
        st.filter_hv::<BPC, TAP_IDX>(dst, c, p_row, RowRef::Lpf(7), filter);
        p_row += 1;
        1
    };

    for _ in 0..tail {
        st.filter_v::<BPC, TAP_IDX>(dst, c, p_row, fv);
        p_row += 1;
    }
}

// ---------------------------------------------------------------------------
// SGR 10-bpc AVX2 çekirdekleri (0 EVEX yönergesi!)
// ---------------------------------------------------------------------------

struct Rows<const N: usize> {
    sq: [[i32; BUF_STRIDE]; N],
    s: [[i32; BUF_STRIDE]; N],
}

impl<const N: usize> Rows<N> {
    fn new() -> Self {
        Self {
            sq: [[0; BUF_STRIDE]; N],
            s: [[0; BUF_STRIDE]; N],
        }
    }
}

/// `sgr_box3_row_h` — AVX2 ile yatay 3'lü kutu toplamı ve kare toplamı.
#[target_feature(enable = "avx2")]
#[inline]
fn box3_row_h_avx2(
    sumsq: &mut [i32; BUF_STRIDE],
    sum: &mut [i32; BUF_STRIDE],
    pad: &[i16; PAD_CAP],
    w: usize,
) {
    let one = _mm256_set1_epi16(1);
    let zero = _mm256_setzero_si256();
    let cols = w + 2;
    let mut j = 0usize;
    while j < cols {
        let a128 = mem::load_i16x8(mem::chunk::<i16, 8>(pad, 1 + j));
        let b128 = mem::load_i16x8(mem::chunk::<i16, 8>(pad, 2 + j));
        let c128 = mem::load_i16x8(mem::chunk::<i16, 8>(pad, 3 + j));

        // 8 piksel -> 256-bit (8 × i32):
        let a = core::arch::x86_64::_mm256_cvtepi16_epi32(a128);
        let b = core::arch::x86_64::_mm256_cvtepi16_epi32(b128);
        let c = core::arch::x86_64::_mm256_cvtepi16_epi32(c128);

        let s = _mm256_add_epi32(_mm256_add_epi32(a, b), c);
        let sq = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_madd_epi16(a, a), _mm256_madd_epi16(b, b)),
            _mm256_madd_epi16(c, c),
        );
        let _ = (one, zero);

        mem::store_i32x8(mem::chunk_mut::<i32, 8>(sum, j), s);
        mem::store_i32x8(mem::chunk_mut::<i32, 8>(sumsq, j), sq);
        j += 8;
    }
}

/// `sgr_box5_row_h` — AVX2 ile yatay 5'li kutu toplamı ve kare toplamı.
#[target_feature(enable = "avx2")]
#[inline]
fn box5_row_h_avx2(
    sumsq: &mut [i32; BUF_STRIDE],
    sum: &mut [i32; BUF_STRIDE],
    pad: &[i16; PAD_CAP],
    w: usize,
) {
    let cols = w + 2;
    let mut j = 0usize;
    while j < cols {
        let a = core::arch::x86_64::_mm256_cvtepi16_epi32(mem::load_i16x8(mem::chunk::<i16, 8>(
            pad, j,
        )));
        let b = core::arch::x86_64::_mm256_cvtepi16_epi32(mem::load_i16x8(mem::chunk::<i16, 8>(
            pad,
            1 + j,
        )));
        let c = core::arch::x86_64::_mm256_cvtepi16_epi32(mem::load_i16x8(mem::chunk::<i16, 8>(
            pad,
            2 + j,
        )));
        let d = core::arch::x86_64::_mm256_cvtepi16_epi32(mem::load_i16x8(mem::chunk::<i16, 8>(
            pad,
            3 + j,
        )));
        let e = core::arch::x86_64::_mm256_cvtepi16_epi32(mem::load_i16x8(mem::chunk::<i16, 8>(
            pad,
            4 + j,
        )));

        let s = _mm256_add_epi32(
            _mm256_add_epi32(a, b),
            _mm256_add_epi32(_mm256_add_epi32(c, d), e),
        );
        let sq = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_madd_epi16(a, a), _mm256_madd_epi16(b, b)),
            _mm256_add_epi32(
                _mm256_add_epi32(_mm256_madd_epi16(c, c), _mm256_madd_epi16(d, d)),
                _mm256_madd_epi16(e, e),
            ),
        );

        mem::store_i32x8(mem::chunk_mut::<i32, 8>(sum, j), s);
        mem::store_i32x8(mem::chunk_mut::<i32, 8>(sumsq, j), sq);
        j += 8;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn calc_row_ab_10bpc_avx2<const IS_5X5: bool>(
    aa: &mut [i32; BUF_STRIDE],
    bb: &mut [i32; BUF_STRIDE],
    w: usize,
    s: u32,
) {
    let c8 = _mm256_set1_epi32(8);
    let c2 = _mm256_set1_epi32(2);
    let vn = _mm256_set1_epi32(if IS_5X5 { 25 } else { 9 });
    let zero = _mm256_setzero_si256();
    let one_by_x: u32 = if IS_5X5 { 164 } else { 455 };
    let cols = w + 2;

    let mut i = 0usize;
    while i < cols {
        let a_raw = load_i32x8(mem::chunk::<i32, 8>(aa, i));
        let b_raw = load_i32x8(mem::chunk::<i32, 8>(bb, i));

        let a = _mm256_srai_epi32::<4>(_mm256_add_epi32(a_raw, c8));
        let b = _mm256_srai_epi32::<2>(_mm256_add_epi32(b_raw, c2));
        let p_vec = _mm256_max_epi32(
            _mm256_sub_epi32(_mm256_mullo_epi32(a, vn), _mm256_mullo_epi32(b, b)),
            zero,
        );

        let mut p_arr = [0i32; 8];
        mem::store_i32x8(&mut p_arr, p_vec);

        for k in 0..8 {
            let p = p_arr[k] as u32;
            let z = p.wrapping_mul(s).wrapping_add(1 << 19) >> 20;
            let x = u32::from(SGR_X_BY_X[z.min(255) as usize]);
            let bb_val = bb[i + k] as u32;
            aa[i + k] = (x
                .wrapping_mul(bb_val)
                .wrapping_mul(one_by_x)
                .wrapping_add(1 << 11)
                >> 12) as i32;
            bb[i + k] = x as i32;
        }
        i += 8;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn box3_vert_avx2<const K: usize>(
    sums: &Rows<3>,
    sp: &mut [usize; 3],
    ab: &mut Rows<K>,
    out: usize,
    w: usize,
    s: u32,
) {
    let (oa, ob) = (&mut ab.sq[out], &mut ab.s[out]);
    let (q0, q1, q2) = (&sums.sq[sp[0]], &sums.sq[sp[1]], &sums.sq[sp[2]]);
    let (s0, s1, s2) = (&sums.s[sp[0]], &sums.s[sp[1]], &sums.s[sp[2]]);
    let cols = w + 2;
    let mut x = 0usize;
    while x < cols {
        let vq = _mm256_add_epi32(
            _mm256_add_epi32(
                load_i32x8(mem::chunk::<i32, 8>(q0, x)),
                load_i32x8(mem::chunk::<i32, 8>(q1, x)),
            ),
            load_i32x8(mem::chunk::<i32, 8>(q2, x)),
        );
        let vs = _mm256_add_epi32(
            _mm256_add_epi32(
                load_i32x8(mem::chunk::<i32, 8>(s0, x)),
                load_i32x8(mem::chunk::<i32, 8>(s1, x)),
            ),
            load_i32x8(mem::chunk::<i32, 8>(s2, x)),
        );
        mem::store_i32x8(mem::chunk_mut::<i32, 8>(oa, x), vq);
        mem::store_i32x8(mem::chunk_mut::<i32, 8>(ob, x), vs);
        x += 8;
    }
    calc_row_ab_10bpc_avx2::<false>(oa, ob, w, s);
    sp.rotate_left(1);
}

#[target_feature(enable = "avx2")]
#[inline]
fn box5_vert_avx2(
    sums: &Rows<5>,
    sp: &mut [usize; 5],
    ab: &mut Rows<2>,
    out: usize,
    w: usize,
    s: u32,
) {
    let (oa, ob) = (&mut ab.sq[out], &mut ab.s[out]);
    let q0 = &sums.sq[sp[0]];
    let q1 = &sums.sq[sp[1]];
    let q2 = &sums.sq[sp[2]];
    let q3 = &sums.sq[sp[3]];
    let q4 = &sums.sq[sp[4]];
    let s0 = &sums.s[sp[0]];
    let s1 = &sums.s[sp[1]];
    let s2 = &sums.s[sp[2]];
    let s3 = &sums.s[sp[3]];
    let s4 = &sums.s[sp[4]];
    let cols = w + 2;
    let mut x = 0usize;
    while x < cols {
        let vq = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    load_i32x8(mem::chunk::<i32, 8>(q0, x)),
                    load_i32x8(mem::chunk::<i32, 8>(q1, x)),
                ),
                _mm256_add_epi32(
                    load_i32x8(mem::chunk::<i32, 8>(q2, x)),
                    load_i32x8(mem::chunk::<i32, 8>(q3, x)),
                ),
            ),
            load_i32x8(mem::chunk::<i32, 8>(q4, x)),
        );
        let vs = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_add_epi32(
                    load_i32x8(mem::chunk::<i32, 8>(s0, x)),
                    load_i32x8(mem::chunk::<i32, 8>(s1, x)),
                ),
                _mm256_add_epi32(
                    load_i32x8(mem::chunk::<i32, 8>(s2, x)),
                    load_i32x8(mem::chunk::<i32, 8>(s3, x)),
                ),
            ),
            load_i32x8(mem::chunk::<i32, 8>(s4, x)),
        );
        mem::store_i32x8(mem::chunk_mut::<i32, 8>(oa, x), vq);
        mem::store_i32x8(mem::chunk_mut::<i32, 8>(ob, x), vs);
        x += 8;
    }
    calc_row_ab_10bpc_avx2::<true>(oa, ob, w, s);
    sp.rotate_left(2);
}

#[target_feature(enable = "avx2")]
#[inline]
fn finish_filter_row1_avx2<const K: usize>(
    tmp: &mut [i32; FILTER_OUT_STRIDE],
    dst: &[u8],
    row_off: usize,
    ab: &Rows<K>,
    ap: [usize; 3],
    w: usize,
) {
    let (b0, b1, b2) = (&ab.s[ap[0]], &ab.s[ap[1]], &ab.s[ap[2]]);
    let (a0, a1, a2) = (&ab.sq[ap[0]], &ab.sq[ap[1]], &ab.sq[ap[2]]);
    let c4 = _mm256_set1_epi32(4);
    let c3 = _mm256_set1_epi32(3);
    let c256 = _mm256_set1_epi32(1 << 8);

    let mut i = 0usize;
    while i + 8 <= w {
        let j = i + 1;
        let b_cross = _mm256_add_epi32(
            _mm256_add_epi32(
                load_i32x8(mem::chunk::<i32, 8>(b1, j)),
                _mm256_add_epi32(
                    load_i32x8(mem::chunk::<i32, 8>(b1, j - 1)),
                    load_i32x8(mem::chunk::<i32, 8>(b1, j + 1)),
                ),
            ),
            _mm256_add_epi32(
                load_i32x8(mem::chunk::<i32, 8>(b0, j)),
                load_i32x8(mem::chunk::<i32, 8>(b2, j)),
            ),
        );
        let b_diag = _mm256_add_epi32(
            _mm256_add_epi32(
                load_i32x8(mem::chunk::<i32, 8>(b0, j - 1)),
                load_i32x8(mem::chunk::<i32, 8>(b2, j - 1)),
            ),
            _mm256_add_epi32(
                load_i32x8(mem::chunk::<i32, 8>(b0, j + 1)),
                load_i32x8(mem::chunk::<i32, 8>(b2, j + 1)),
            ),
        );
        let a_coef = _mm256_add_epi32(
            _mm256_mullo_epi32(b_cross, c4),
            _mm256_mullo_epi32(b_diag, c3),
        );

        let a_cross = _mm256_add_epi32(
            _mm256_add_epi32(
                load_i32x8(mem::chunk::<i32, 8>(a1, j)),
                _mm256_add_epi32(
                    load_i32x8(mem::chunk::<i32, 8>(a1, j - 1)),
                    load_i32x8(mem::chunk::<i32, 8>(a1, j + 1)),
                ),
            ),
            _mm256_add_epi32(
                load_i32x8(mem::chunk::<i32, 8>(a0, j)),
                load_i32x8(mem::chunk::<i32, 8>(a2, j)),
            ),
        );
        let a_diag = _mm256_add_epi32(
            _mm256_add_epi32(
                load_i32x8(mem::chunk::<i32, 8>(a0, j - 1)),
                load_i32x8(mem::chunk::<i32, 8>(a2, j - 1)),
            ),
            _mm256_add_epi32(
                load_i32x8(mem::chunk::<i32, 8>(a0, j + 1)),
                load_i32x8(mem::chunk::<i32, 8>(a2, j + 1)),
            ),
        );
        let b_acc = _mm256_add_epi32(
            _mm256_mullo_epi32(a_cross, c4),
            _mm256_mullo_epi32(a_diag, c3),
        );

        let src128 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, row_off + i * 2));
        let v_src = core::arch::x86_64::_mm256_cvtepu16_epi32(src128);
        let prod = _mm256_mullo_epi32(a_coef, v_src);
        let res = _mm256_srai_epi32::<9>(_mm256_add_epi32(_mm256_sub_epi32(b_acc, prod), c256));
        mem::store_i32x8(mem::chunk_mut::<i32, 8>(tmp, i), res);
        i += 8;
    }
    while i < w {
        let j = i + 1;
        let a = (b1[j] + b1[j - 1] + b1[j + 1] + b0[j] + b2[j]) * 4
            + (b0[j - 1] + b2[j - 1] + b0[j + 1] + b2[j + 1]) * 3;
        let b = (a1[j] + a1[j - 1] + a1[j + 1] + a0[j] + a2[j]) * 4
            + (a0[j - 1] + a2[j - 1] + a0[j + 1] + a2[j + 1]) * 3;
        let s = i32::from(read_u16_le(dst, row_off + i * 2));
        tmp[i] = (b - a * s + (1 << 8)) >> 9;
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn finish_filter2_avx2(
    tmp0: &mut [i32; FILTER_OUT_STRIDE],
    tmp1: &mut [i32; FILTER_OUT_STRIDE],
    dst: &[u8],
    row0_off: usize,
    row1_off: usize,
    ab: &Rows<2>,
    ap: [usize; 2],
    w: usize,
    h: usize,
) {
    let (b0, b1) = (&ab.s[ap[0]], &ab.s[ap[1]]);
    let (a0, a1) = (&ab.sq[ap[0]], &ab.sq[ap[1]]);
    let c6 = _mm256_set1_epi32(6);
    let c5 = _mm256_set1_epi32(5);
    let c256 = _mm256_set1_epi32(1 << 8);
    let c128 = _mm256_set1_epi32(1 << 7);

    let mut i = 0usize;
    while i + 8 <= w {
        let j = i + 1;
        let b_mid = _mm256_add_epi32(
            load_i32x8(mem::chunk::<i32, 8>(b0, j)),
            load_i32x8(mem::chunk::<i32, 8>(b1, j)),
        );
        let b_side = _mm256_add_epi32(
            _mm256_add_epi32(
                load_i32x8(mem::chunk::<i32, 8>(b0, j - 1)),
                load_i32x8(mem::chunk::<i32, 8>(b1, j - 1)),
            ),
            _mm256_add_epi32(
                load_i32x8(mem::chunk::<i32, 8>(b0, j + 1)),
                load_i32x8(mem::chunk::<i32, 8>(b1, j + 1)),
            ),
        );
        let a_coef = _mm256_add_epi32(
            _mm256_mullo_epi32(b_mid, c6),
            _mm256_mullo_epi32(b_side, c5),
        );

        let a_mid = _mm256_add_epi32(
            load_i32x8(mem::chunk::<i32, 8>(a0, j)),
            load_i32x8(mem::chunk::<i32, 8>(a1, j)),
        );
        let a_side = _mm256_add_epi32(
            _mm256_add_epi32(
                load_i32x8(mem::chunk::<i32, 8>(a0, j - 1)),
                load_i32x8(mem::chunk::<i32, 8>(a1, j - 1)),
            ),
            _mm256_add_epi32(
                load_i32x8(mem::chunk::<i32, 8>(a0, j + 1)),
                load_i32x8(mem::chunk::<i32, 8>(a1, j + 1)),
            ),
        );
        let b_acc = _mm256_add_epi32(
            _mm256_mullo_epi32(a_mid, c6),
            _mm256_mullo_epi32(a_side, c5),
        );

        let src128 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, row0_off + i * 2));
        let v_src = core::arch::x86_64::_mm256_cvtepu16_epi32(src128);
        let prod = _mm256_mullo_epi32(a_coef, v_src);
        let res = _mm256_srai_epi32::<9>(_mm256_add_epi32(_mm256_sub_epi32(b_acc, prod), c256));
        mem::store_i32x8(mem::chunk_mut::<i32, 8>(tmp0, i), res);
        i += 8;
    }
    while i < w {
        let j = i + 1;
        let a = (b0[j] + b1[j]) * 6 + (b0[j - 1] + b1[j - 1] + b0[j + 1] + b1[j + 1]) * 5;
        let b = (a0[j] + a1[j]) * 6 + (a0[j - 1] + a1[j - 1] + a0[j + 1] + a1[j + 1]) * 5;
        let s = i32::from(read_u16_le(dst, row0_off + i * 2));
        tmp0[i] = (b - a * s + (1 << 8)) >> 9;
        i += 1;
    }

    if h <= 1 {
        return;
    }

    let mut i = 0usize;
    while i + 8 <= w {
        let j = i + 1;
        let b_mid = load_i32x8(mem::chunk::<i32, 8>(b1, j));
        let b_side = _mm256_add_epi32(
            load_i32x8(mem::chunk::<i32, 8>(b1, j - 1)),
            load_i32x8(mem::chunk::<i32, 8>(b1, j + 1)),
        );
        let a_coef = _mm256_add_epi32(
            _mm256_mullo_epi32(b_mid, c6),
            _mm256_mullo_epi32(b_side, c5),
        );

        let a_mid = load_i32x8(mem::chunk::<i32, 8>(a1, j));
        let a_side = _mm256_add_epi32(
            load_i32x8(mem::chunk::<i32, 8>(a1, j - 1)),
            load_i32x8(mem::chunk::<i32, 8>(a1, j + 1)),
        );
        let b_acc = _mm256_add_epi32(
            _mm256_mullo_epi32(a_mid, c6),
            _mm256_mullo_epi32(a_side, c5),
        );

        let src128 = mem::load_u8x16(mem::chunk::<u8, 16>(dst, row1_off + i * 2));
        let v_src = core::arch::x86_64::_mm256_cvtepu16_epi32(src128);
        let prod = _mm256_mullo_epi32(a_coef, v_src);
        let res = _mm256_srai_epi32::<8>(_mm256_add_epi32(_mm256_sub_epi32(b_acc, prod), c128));
        mem::store_i32x8(mem::chunk_mut::<i32, 8>(tmp1, i), res);
        i += 8;
    }
    while i < w {
        let j = i + 1;
        let a = b1[j] * 6 + (b1[j - 1] + b1[j + 1]) * 5;
        let b = a1[j] * 6 + (a1[j - 1] + a1[j + 1]) * 5;
        let s = i32::from(read_u16_le(dst, row1_off + i * 2));
        tmp1[i] = (b - a * s + (1 << 7)) >> 8;
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn weighted_row1_10bpc_avx2(
    dst: &mut [u8],
    row_off: usize,
    t1: &[i32; FILTER_OUT_STRIDE],
    w: usize,
    w1: i32,
) {
    let vw1 = _mm256_set1_epi32(w1);
    let c1024 = _mm256_set1_epi32(1 << 10);
    let zero = _mm256_setzero_si256();
    let max_px = _mm256_set1_epi16(1023);
    let mut out_row = [0i16; FILTER_OUT_STRIDE];

    let mut i = 0usize;
    while i + 16 <= w {
        let t_0 = load_i32x8(mem::chunk::<i32, 8>(t1, i));
        let t_1 = load_i32x8(mem::chunk::<i32, 8>(t1, i + 8));
        let v_0 = _mm256_srai_epi32::<11>(_mm256_add_epi32(_mm256_mullo_epi32(t_0, vw1), c1024));
        let v_1 = _mm256_srai_epi32::<11>(_mm256_add_epi32(_mm256_mullo_epi32(t_1, vw1), c1024));
        // _mm256_packs_epi32(v_0, v_1) 128-bit şeritleri [0..3, 8..11, 4..7, 12..15] olarak paketler;
        // _mm256_permute4x64_epi64::<0xD8> ile [0..15] doğal sıraya döner!
        let delta =
            core::arch::x86_64::_mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(v_0, v_1));
        let d = mem::load_u8x32(mem::chunk::<u8, 32>(dst, row_off + i * 2));
        let res = _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(d, delta), zero), max_px);
        mem::store_i16x16(mem::chunk_mut::<i16, 16>(&mut out_row, i), res);
        i += 16;
    }
    while i < w {
        let d = i32::from(read_u16_le(dst, row_off + i * 2));
        let v = d + ((w1 * t1[i] + (1 << 10)) >> 11);
        out_row[i] = v.clamp(0, 1023) as i16;
        i += 1;
    }
    store_row_u16_avx2(dst, row_off, w, &out_row);
}

#[target_feature(enable = "avx2")]
#[inline]
fn weighted2_row_10bpc_avx2(
    dst: &mut [u8],
    row_off: usize,
    t1: &[i32; FILTER_OUT_STRIDE],
    t2: &[i32; FILTER_OUT_STRIDE],
    w: usize,
    w0: i32,
    w1: i32,
) {
    let vw0 = _mm256_set1_epi32(w0);
    let vw1 = _mm256_set1_epi32(w1);
    let c1024 = _mm256_set1_epi32(1 << 10);
    let zero = _mm256_setzero_si256();
    let max_px = _mm256_set1_epi16(1023);
    let mut out_row = [0i16; FILTER_OUT_STRIDE];

    let mut i = 0usize;
    while i + 16 <= w {
        let t1_0 = load_i32x8(mem::chunk::<i32, 8>(t1, i));
        let t1_1 = load_i32x8(mem::chunk::<i32, 8>(t1, i + 8));
        let t2_0 = load_i32x8(mem::chunk::<i32, 8>(t2, i));
        let t2_1 = load_i32x8(mem::chunk::<i32, 8>(t2, i + 8));

        let v_0 = _mm256_srai_epi32::<11>(_mm256_add_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(t1_0, vw0), _mm256_mullo_epi32(t2_0, vw1)),
            c1024,
        ));
        let v_1 = _mm256_srai_epi32::<11>(_mm256_add_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(t1_1, vw0), _mm256_mullo_epi32(t2_1, vw1)),
            c1024,
        ));
        let delta =
            core::arch::x86_64::_mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(v_0, v_1));
        let d = mem::load_u8x32(mem::chunk::<u8, 32>(dst, row_off + i * 2));
        let res = _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(d, delta), zero), max_px);
        mem::store_i16x16(mem::chunk_mut::<i16, 16>(&mut out_row, i), res);
        i += 16;
    }
    while i < w {
        let d = i32::from(read_u16_le(dst, row_off + i * 2));
        let v = d + ((w0 * t1[i] + w1 * t2[i] + (1 << 10)) >> 11);
        out_row[i] = v.clamp(0, 1023) as i16;
        i += 1;
    }
    store_row_u16_avx2(dst, row_off, w, &out_row);
}

#[target_feature(enable = "avx2")]
#[inline]
fn finish1_avx2(
    dst: &mut [u8],
    c: &Ctx<'_>,
    dst_row: &mut usize,
    ab: &Rows<3>,
    ap: &mut [usize; 3],
    tmp: &mut [i32; FILTER_OUT_STRIDE],
    w1: i32,
) {
    let r = *dst_row;
    finish_filter_row1_avx2::<3>(tmp, dst, r * c.stride, ab, *ap, c.w);
    weighted_row1_10bpc_avx2(dst, r * c.stride, tmp, c.w, w1);
    *dst_row += 1;
    ap.rotate_left(1);
}

#[target_feature(enable = "avx2")]
#[inline]
fn finish2_avx2(
    dst: &mut [u8],
    c: &Ctx<'_>,
    dst_row: &mut usize,
    ab: &Rows<2>,
    ap: &mut [usize; 2],
    tmp0: &mut [i32; FILTER_OUT_STRIDE],
    tmp1: &mut [i32; FILTER_OUT_STRIDE],
    h: usize,
    w0: i32,
) {
    let r = *dst_row;
    let r1 = if h > 1 { r + 1 } else { r };
    finish_filter2_avx2(
        tmp0,
        tmp1,
        dst,
        r * c.stride,
        r1 * c.stride,
        ab,
        *ap,
        c.w,
        h,
    );
    weighted_row1_10bpc_avx2(dst, r * c.stride, tmp0, c.w, w0);
    *dst_row += 1;
    if h > 1 {
        weighted_row1_10bpc_avx2(dst, (r + 1) * c.stride, tmp1, c.w, w0);
        *dst_row += 1;
    }
    ap.rotate_left(1);
}

#[target_feature(enable = "avx2")]
#[inline]
fn finish_mix_avx2(
    dst: &mut [u8],
    c: &Ctx<'_>,
    dst_row: &mut usize,
    ab5: &Rows<2>,
    ap5: &mut [usize; 2],
    ab3: &Rows<4>,
    ap3: &mut [usize; 4],
    tmp5_0: &mut [i32; FILTER_OUT_STRIDE],
    tmp5_1: &mut [i32; FILTER_OUT_STRIDE],
    tmp3_0: &mut [i32; FILTER_OUT_STRIDE],
    tmp3_1: &mut [i32; FILTER_OUT_STRIDE],
    h: usize,
    w0: i32,
    w1: i32,
) {
    let r = *dst_row;
    let r1 = if h > 1 { r + 1 } else { r };
    finish_filter2_avx2(
        tmp5_0,
        tmp5_1,
        dst,
        r * c.stride,
        r1 * c.stride,
        ab5,
        *ap5,
        c.w,
        h,
    );
    finish_filter_row1_avx2::<4>(
        tmp3_0,
        dst,
        r * c.stride,
        ab3,
        [ap3[0], ap3[1], ap3[2]],
        c.w,
    );
    weighted2_row_10bpc_avx2(dst, r * c.stride, tmp5_0, tmp3_0, c.w, w0, w1);
    if h > 1 {
        finish_filter_row1_avx2::<4>(
            tmp3_1,
            dst,
            (r + 1) * c.stride,
            ab3,
            [ap3[1], ap3[2], ap3[3]],
            c.w,
        );
        weighted2_row_10bpc_avx2(dst, (r + 1) * c.stride, tmp5_1, tmp3_1, c.w, w0, w1);
    }
    *dst_row += h;
    ap5.rotate_left(1);
    ap3.rotate_left(1);
}

#[target_feature(enable = "avx2")]
fn sgr_3x3_10bpc_avx2_giris(dst: &mut [u8], c: &Ctx<'_>, h: usize, s1: u32, w1: i32) {
    enum Kuyruk {
        Bitti,
        Vert2,
        Vert1,
    }
    let w = c.w;
    let mut sums = Rows::<3>::new();
    let mut ab = Rows::<3>::new();
    let mut pad = [0i16; PAD_CAP];
    let mut sp: [usize; 3];
    let mut ap = [0usize, 1, 2];
    let mut tmp = [0i32; FILTER_OUT_STRIDE];
    let mut h = h;
    let mut sr = 0usize;
    let mut dr = 0usize;

    let mut row_h = |sums: &mut Rows<3>, dst: &[u8], row: RowRef, r: usize| {
        c.stage_row(dst, row, 2, &mut pad);
        box3_row_h_avx2(&mut sums.sq[r], &mut sums.s[r], &pad, w);
    };

    let kuyruk = 'akis: {
        if c.edges.contains(LrEdgeFlags::HAVE_TOP) {
            sp = [0, 1, 2];
            row_h(&mut sums, dst, RowRef::Lpf(0), 0);
            row_h(&mut sums, dst, RowRef::Lpf(1), 1);
            row_h(&mut sums, dst, RowRef::Main(sr), sp[2]);
            box3_vert_avx2::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            row_h(&mut sums, dst, RowRef::Main(sr), sp[2]);
            box3_vert_avx2::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
        } else {
            sp = [0, 0, 0];
            row_h(&mut sums, dst, RowRef::Main(sr), 0);
            sr += 1;
            box3_vert_avx2::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp[2] = 1;
            row_h(&mut sums, dst, RowRef::Main(sr), sp[2]);
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
            row_h(&mut sums, dst, RowRef::Main(sr), sp[2]);
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
        row_h(&mut sums, dst, RowRef::Lpf(6), sp[2]);
        box3_vert_avx2::<3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
        finish1_avx2(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        row_h(&mut sums, dst, RowRef::Lpf(7), sp[2]);
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
fn sgr_5x5_10bpc_avx2_giris(dst: &mut [u8], c: &Ctx<'_>, h: usize, s0: u32, w0: i32) {
    enum Kuyruk {
        Output2,
        Vert2,
        Odd,
        Vert1,
    }
    let w = c.w;
    let mut sums = Rows::<5>::new();
    let mut ab = Rows::<2>::new();
    let mut pad = [0i16; PAD_CAP];
    let mut sp: [usize; 5];
    let mut ap = [0usize, 1];
    let mut tmp0 = [0i32; FILTER_OUT_STRIDE];
    let mut tmp1 = [0i32; FILTER_OUT_STRIDE];
    let mut h = h;
    let mut sr = 0usize;
    let mut dr = 0usize;

    let mut row_h = |sums: &mut Rows<5>, dst: &[u8], row: RowRef, r: usize| {
        c.stage_row(dst, row, 3, &mut pad);
        box5_row_h_avx2(&mut sums.sq[r], &mut sums.s[r], &pad, w);
    };

    let kuyruk = 'akis: {
        if c.edges.contains(LrEdgeFlags::HAVE_TOP) {
            sp = [0, 0, 1, 2, 3];
            row_h(&mut sums, dst, RowRef::Lpf(0), 0);
            row_h(&mut sums, dst, RowRef::Lpf(1), 1);
            row_h(&mut sums, dst, RowRef::Main(sr), 2);
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            row_h(&mut sums, dst, RowRef::Main(sr), 3);
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
            row_h(&mut sums, dst, RowRef::Main(sr), 0);
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp[4] = 1;
            row_h(&mut sums, dst, RowRef::Main(sr), 1);
            sr += 1;
            box5_vert_avx2(&sums, &mut sp, &mut ab, ap[1], w, s0);
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp[3] = 2;
            sp[4] = 3;
            row_h(&mut sums, dst, RowRef::Main(sr), 2);
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            row_h(&mut sums, dst, RowRef::Main(sr), 3);
            sr += 1;
            box5_vert_avx2(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2_avx2(dst, c, &mut dr, &ab, &mut ap, &mut tmp0, &mut tmp1, 2, w0);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp[3] = 4;
        }

        loop {
            row_h(&mut sums, dst, RowRef::Main(sr), sp[3]);
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            row_h(&mut sums, dst, RowRef::Main(sr), sp[4]);
            sr += 1;
            box5_vert_avx2(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2_avx2(dst, c, &mut dr, &ab, &mut ap, &mut tmp0, &mut tmp1, 2, w0);
            h -= 1;
            if h == 0 {
                break;
            }
        }

        if !c.edges.contains(LrEdgeFlags::HAVE_BOTTOM) {
            break 'akis Kuyruk::Vert2;
        }
        row_h(&mut sums, dst, RowRef::Lpf(6), sp[3]);
        row_h(&mut sums, dst, RowRef::Lpf(7), sp[4]);
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
            finish2_avx2(dst, c, &mut dr, &ab, &mut ap, &mut tmp0, &mut tmp1, 2, w0);
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
        finish2_avx2(dst, c, &mut dr, &ab, &mut ap, &mut tmp0, &mut tmp1, 1, w0);
    } else {
        box5_vert_avx2(&sums, &mut sp, &mut ab, ap[1], w, s0);
        finish2_avx2(dst, c, &mut dr, &ab, &mut ap, &mut tmp0, &mut tmp1, 2, w0);
    }
}

#[target_feature(enable = "avx2")]
fn sgr_mix_10bpc_avx2_giris(
    dst: &mut [u8],
    c: &Ctx<'_>,
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
    let mut sums5 = Rows::<5>::new();
    let mut sums3 = Rows::<3>::new();
    let mut ab5 = Rows::<2>::new();
    let mut ab3 = Rows::<4>::new();
    let mut pad = [0i16; PAD_CAP];
    let mut sp5: [usize; 5];
    let mut sp3: [usize; 3];
    let mut ap5 = [0usize, 1];
    let mut ap3 = [0usize, 1, 2, 3];
    let mut tmp5_0 = [0i32; FILTER_OUT_STRIDE];
    let mut tmp5_1 = [0i32; FILTER_OUT_STRIDE];
    let mut tmp3_0 = [0i32; FILTER_OUT_STRIDE];
    let mut tmp3_1 = [0i32; FILTER_OUT_STRIDE];
    let mut h = h;
    let mut sr = 0usize;
    let mut dr = 0usize;

    let mut row_h35 =
        |s3: &mut Rows<3>, s5: &mut Rows<5>, dst: &[u8], row: RowRef, r3: usize, r5: usize| {
            c.stage_row(dst, row, 3, &mut pad);
            box3_row_h_avx2(&mut s3.sq[r3], &mut s3.s[r3], &pad, w);
            box5_row_h_avx2(&mut s5.sq[r5], &mut s5.s[r5], &pad, w);
        };

    let kuyruk = 'akis: {
        if c.edges.contains(LrEdgeFlags::HAVE_TOP) {
            sp5 = [0, 0, 1, 2, 3];
            sp3 = [0, 1, 2];
            row_h35(&mut sums3, &mut sums5, dst, RowRef::Lpf(0), 0, 0);
            row_h35(&mut sums3, &mut sums5, dst, RowRef::Lpf(1), 1, 1);
            row_h35(&mut sums3, &mut sums5, dst, RowRef::Main(sr), 2, 2);
            sr += 1;
            box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            row_h35(&mut sums3, &mut sums5, dst, RowRef::Main(sr), sp3[2], 3);
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
            row_h35(&mut sums3, &mut sums5, dst, RowRef::Main(sr), 0, 0);
            sr += 1;
            box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp5[4] = 1;
            sp3[2] = 1;
            row_h35(&mut sums3, &mut sums5, dst, RowRef::Main(sr), 1, 1);
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
            row_h35(&mut sums3, &mut sums5, dst, RowRef::Main(sr), 2, 2);
            sr += 1;
            box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            row_h35(&mut sums3, &mut sums5, dst, RowRef::Main(sr), sp3[2], 3);
            sr += 1;
            box5_vert_avx2(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix_avx2(
                dst,
                c,
                &mut dr,
                &ab5,
                &mut ap5,
                &ab3,
                &mut ap3,
                &mut tmp5_0,
                &mut tmp5_1,
                &mut tmp3_0,
                &mut tmp3_1,
                2,
                w0,
                w1,
            );
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp5[3] = 4;
        }

        loop {
            row_h35(
                &mut sums3,
                &mut sums5,
                dst,
                RowRef::Main(sr),
                sp3[2],
                sp5[3],
            );
            sr += 1;
            box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            row_h35(
                &mut sums3,
                &mut sums5,
                dst,
                RowRef::Main(sr),
                sp3[2],
                sp5[4],
            );
            sr += 1;
            box5_vert_avx2(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix_avx2(
                dst,
                c,
                &mut dr,
                &ab5,
                &mut ap5,
                &ab3,
                &mut ap3,
                &mut tmp5_0,
                &mut tmp5_1,
                &mut tmp3_0,
                &mut tmp3_1,
                2,
                w0,
                w1,
            );
            h -= 1;
            if h == 0 {
                break;
            }
        }

        if !c.edges.contains(LrEdgeFlags::HAVE_BOTTOM) {
            break 'akis Kuyruk::Vert2;
        }
        row_h35(&mut sums3, &mut sums5, dst, RowRef::Lpf(6), sp3[2], sp5[3]);
        box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
        ap3.rotate_left(1);
        row_h35(&mut sums3, &mut sums5, dst, RowRef::Lpf(7), sp3[2], sp5[4]);
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
                dst,
                c,
                &mut dr,
                &ab5,
                &mut ap5,
                &ab3,
                &mut ap3,
                &mut tmp5_0,
                &mut tmp5_1,
                &mut tmp3_0,
                &mut tmp3_1,
                2,
                w0,
                w1,
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
            dst,
            c,
            &mut dr,
            &ab5,
            &mut ap5,
            &ab3,
            &mut ap3,
            &mut tmp5_0,
            &mut tmp5_1,
            &mut tmp3_0,
            &mut tmp3_1,
            1,
            w0,
            w1,
        );
    } else {
        box5_vert_avx2(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
        box3_vert_avx2::<4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
        finish_mix_avx2(
            dst,
            c,
            &mut dr,
            &ab5,
            &mut ap5,
            &ab3,
            &mut ap3,
            &mut tmp5_0,
            &mut tmp5_1,
            &mut tmp3_0,
            &mut tmp3_1,
            2,
            w0,
            w1,
        );
    }
}
