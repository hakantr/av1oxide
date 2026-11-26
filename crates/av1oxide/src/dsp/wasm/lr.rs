//! WebAssembly `simd128` loop restoration (Wiener / SGRPROJ) çekirdekleri.
//!
//! Kapsam (`15` yuva @ `8 / 10 / 12` bpc):
//! - `wiener[0..1]` (`7-tap` ve `5-tap`) @ `8 / 10 / 12` bpc (`6` yuva)
//! - `sgr[0..2]` (`5×5`, `3×3`, `mix`) @ `8 / 10 / 12` bpc (`9` yuva)

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use core::arch::wasm32::{
    i16x8_narrow_i32x4, i16x8_splat, i32x4, i32x4_add, i32x4_extmul_high_i16x8,
    i32x4_extmul_low_i16x8, i32x4_extract_lane, i32x4_max, i32x4_min, i32x4_mul, i32x4_shr,
    i32x4_splat, i32x4_sub, u16x8_add, u16x8_shuffle, u32x4_add, u32x4_extend_high_u16x8,
    u32x4_extend_low_u16x8, u32x4_extmul_high_u16x8, u32x4_extmul_low_u16x8, u32x4_min, u32x4_shr,
    u32x4_splat, v128,
};

use super::{
    bpp_of, load_i32x4, load_pixels_8, load_scalar, load_u16x8, pack_and_store_8, store_i32x4,
    store_scalar, store_u16x8,
};
use crate::dsp::lr::{
    LrBitDepth, LrEdgeFlags, LrFilterFn, LrParams, sgr_skaler_devret, sgr_sozlesmesi,
    wiener_skaler_devret, wiener_sozlesmesi,
};
use crate::dsp::lr_tables::SGR_X_BY_X;
use crate::dsp::scalar::lr::LR_MAX_W;

const REST_UNIT_STRIDE: usize = 390;
const BUF_STRIDE: usize = 384 + 16;
const FILTER_OUT_STRIDE: usize = 384;

/// `wiener[tap]` (`8 / 10 / 12` bpc) giriş fonksiyonu.
#[must_use]
pub fn wiener_fn<const BPC: u32>(tap: usize) -> Option<LrFilterFn> {
    match tap {
        0 => Some(lr_wiener::<BPC, 0>),
        1 => Some(lr_wiener::<BPC, 1>),
        _ => None,
    }
}

/// `sgr[idx]` (`8 / 10 / 12` bpc) giriş fonksiyonu.
#[must_use]
pub fn sgr_fn<const BPC: u32>(idx: usize) -> Option<LrFilterFn> {
    match idx {
        0 => Some(lr_sgr::<BPC, 0>),
        1 => Some(lr_sgr::<BPC, 1>),
        2 => Some(lr_sgr::<BPC, 2>),
        _ => None,
    }
}

#[target_feature(enable = "simd128")]
pub fn lr_wiener<const BPC: u32, const TAP: usize>(
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
    if !wiener_sozlesmesi::<BPC>(
        TAP,
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
    ) {
        wiener_skaler_devret::<BPC>(
            TAP, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
        );
        return;
    }
    let LrParams::Wiener { filter } = params else {
        return;
    };
    if TAP == 1 {
        wiener_wasm::<BPC, true>(
            dst, stride, left, lpf, lpf_stride, lpf_x, w, h, filter, edges,
        );
    } else {
        wiener_wasm::<BPC, false>(
            dst, stride, left, lpf, lpf_stride, lpf_x, w, h, filter, edges,
        );
    }
}

#[target_feature(enable = "simd128")]
pub fn lr_sgr<const BPC: u32, const SGR_IDX: usize>(
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
    if !sgr_sozlesmesi::<BPC>(
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
    ) {
        sgr_skaler_devret::<BPC>(
            SGR_IDX, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
        );
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
    match SGR_IDX {
        0 => sgr_5x5_impl::<BPC>(dst, &c, h, s0, w0),
        1 => sgr_3x3_impl::<BPC>(dst, &c, h, s1, w1),
        _ => sgr_mix_impl::<BPC>(dst, &c, h, s0, s1, w0, w1),
    }
}

#[inline(always)]
fn iclip_pixel<const BPC: u32>(v: i32) -> i32 {
    v.clamp(0, (1 << BPC) - 1)
}

#[inline(always)]
fn coef<const BPC: u32>(v: i32) -> i32 {
    if BPC == 8 { i32::from(v as i16) } else { v }
}

#[derive(Clone, Copy)]
struct Src<'a> {
    buf: &'a [u8],
    base: usize,
}

impl Src<'_> {
    #[inline(always)]
    fn at<const BPC: u32>(self, x: isize) -> i32 {
        let off = (self.base as isize + x * bpp_of(BPC) as isize) as usize;
        load_scalar::<BPC>(self.buf, off)
    }
}

#[derive(Clone, Copy)]
enum RowRef {
    Main(usize),
    Lpf(usize),
}

struct Ctx<'l> {
    left: &'l [u8],
    lpf: &'l [u8],
    stride: usize,
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    edges: LrEdgeFlags,
}

impl<'l> Ctx<'l> {
    #[inline(always)]
    fn row<'a, const BPC: u32>(&'a self, dst: &'a [u8], row: RowRef) -> (Option<Src<'a>>, Src<'a>) {
        let bpp = bpp_of(BPC);
        match row {
            RowRef::Main(r) => {
                let left = if self.edges.contains(LrEdgeFlags::HAVE_LEFT) {
                    Some(Src {
                        buf: self.left,
                        base: r * 4 * bpp,
                    })
                } else {
                    None
                };
                (
                    left,
                    Src {
                        buf: dst,
                        base: r * self.stride,
                    },
                )
            }
            RowRef::Lpf(k) => (
                None,
                Src {
                    buf: self.lpf,
                    base: k * self.lpf_stride + self.lpf_x * bpp,
                },
            ),
        }
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn unpack_row_padded<const BPC: u32, const REACH: usize>(
    line: &mut [u16],
    left: Option<Src<'_>>,
    src: Src<'_>,
    w: usize,
    edges: LrEdgeFlags,
) {
    let have_left = edges.contains(LrEdgeFlags::HAVE_LEFT);
    let have_right = edges.contains(LrEdgeFlags::HAVE_RIGHT);
    let bpp = bpp_of(BPC);

    for k in 1..=REACH {
        let idx = -(k as isize);
        let v = if !have_left {
            src.at::<BPC>(0)
        } else if let Some(l) = left {
            l.at::<BPC>(4 + idx)
        } else {
            src.at::<BPC>(idx)
        };
        line[REACH - k] = v as u16;
    }

    let total_src = w + if have_right { REACH } else { 0 };
    let mut x = 0usize;
    while x + 8 <= total_src {
        let v = load_pixels_8::<BPC>(src.buf, src.base + x * bpp);
        store_u16x8(line, REACH + x, v);
        x += 8;
    }
    while x < total_src {
        line[REACH + x] = src.at::<BPC>(x as isize) as u16;
        x += 1;
    }

    if !have_right {
        let last = line[REACH + w - 1];
        for k in 0..REACH {
            line[REACH + w + k] = last;
        }
    }
}

// ============================================================================
// Wiener (7-tap ve 5-tap, 8/10/12 bpc)
// ============================================================================

#[target_feature(enable = "simd128")]
fn wiener_filter_h_wasm<const BPC: u32, const FIVE: bool>(
    out: &mut [u16],
    left: Option<Src<'_>>,
    src: Src<'_>,
    fh: &[i16; 8],
    w: usize,
    edges: LrEdgeFlags,
) {
    let mut line = [0u16; LR_MAX_W + 16];
    unpack_row_padded::<BPC, 3>(&mut line, left, src, w, edges);

    let bitdepth = BPC as i32;
    let round_bits_h: u32 = if BPC == 12 { 5 } else { 3 };
    let rounding_off_h = 1 << (round_bits_h - 1);
    let clip_max = (1 << (bitdepth + 8 - round_bits_h as i32)) - 1;
    let init_sum = (1 << (bitdepth + 6)) + rounding_off_h;

    let f0 = i16x8_splat(fh[0]);
    let f1 = i16x8_splat(fh[1]);
    let f2 = i16x8_splat(fh[2]);
    let f3_val = i32::from(fh[3]) + if BPC == 8 { 128 } else { 0 };
    let f3 = i16x8_splat(f3_val as i16);
    let f4 = i16x8_splat(fh[4]);
    let f5 = i16x8_splat(fh[5]);
    let f6 = i16x8_splat(fh[6]);

    let init_v = i32x4_splat(init_sum);
    let zero = i32x4_splat(0);
    let max_v = i32x4_splat(clip_max);

    let mut x = 0usize;
    while x < w {
        let p0 = load_u16x8(&line, x);
        let p1 = load_u16x8(&line, x + 8);

        let w1 = u16x8_shuffle::<1, 2, 3, 4, 5, 6, 7, 8>(p0, p1);
        let w2 = u16x8_shuffle::<2, 3, 4, 5, 6, 7, 8, 9>(p0, p1);
        let w3 = u16x8_shuffle::<3, 4, 5, 6, 7, 8, 9, 10>(p0, p1);
        let w4 = u16x8_shuffle::<4, 5, 6, 7, 8, 9, 10, 11>(p0, p1);
        let w5 = u16x8_shuffle::<5, 6, 7, 8, 9, 10, 11, 12>(p0, p1);

        let mut acc_lo = i32x4_add(init_v, i32x4_extmul_low_i16x8(w1, f1));
        acc_lo = i32x4_add(acc_lo, i32x4_extmul_low_i16x8(w2, f2));
        acc_lo = i32x4_add(acc_lo, i32x4_extmul_low_i16x8(w3, f3));
        acc_lo = i32x4_add(acc_lo, i32x4_extmul_low_i16x8(w4, f4));
        acc_lo = i32x4_add(acc_lo, i32x4_extmul_low_i16x8(w5, f5));

        let mut acc_hi = i32x4_add(init_v, i32x4_extmul_high_i16x8(w1, f1));
        acc_hi = i32x4_add(acc_hi, i32x4_extmul_high_i16x8(w2, f2));
        acc_hi = i32x4_add(acc_hi, i32x4_extmul_high_i16x8(w3, f3));
        acc_hi = i32x4_add(acc_hi, i32x4_extmul_high_i16x8(w4, f4));
        acc_hi = i32x4_add(acc_hi, i32x4_extmul_high_i16x8(w5, f5));

        if !FIVE {
            let w6 = u16x8_shuffle::<6, 7, 8, 9, 10, 11, 12, 13>(p0, p1);
            acc_lo = i32x4_add(acc_lo, i32x4_extmul_low_i16x8(p0, f0));
            acc_lo = i32x4_add(acc_lo, i32x4_extmul_low_i16x8(w6, f6));
            acc_hi = i32x4_add(acc_hi, i32x4_extmul_high_i16x8(p0, f0));
            acc_hi = i32x4_add(acc_hi, i32x4_extmul_high_i16x8(w6, f6));
        }

        let sh_lo = i32x4_min(i32x4_max(i32x4_shr(acc_lo, round_bits_h), zero), max_v);
        let sh_hi = i32x4_min(i32x4_max(i32x4_shr(acc_hi, round_bits_h), zero), max_v);
        // `clip_max <= 32767` (8-bit: 8191, 10-bit: 32767, 12-bit: 32767), böylece i16 daraltma birebir.
        store_u16x8(out, x, i16x8_narrow_i32x4(sh_lo, sh_hi));
        x += 8;
    }
}

const fn wiener_v_init(bpc: u32) -> i32 {
    let round_bits_v = if bpc == 12 { 9 } else { 11 };
    let rounding_off_v = 1 << (round_bits_v - 1);
    let round_offset = 1 << (bpc as i32 + (round_bits_v - 1));
    -round_offset + rounding_off_v
}

#[target_feature(enable = "simd128")]
fn wiener_vert_pass_wasm<const BPC: u32, const FIVE: bool>(
    dst: &mut [u8],
    dst_off: usize,
    r0: &[u16],
    r1: &[u16],
    r2: &[u16],
    r3: &[u16],
    r4: &[u16],
    r5: &[u16],
    r6: &[u16],
    fv: &[i16; 8],
    w: usize,
) {
    let bpp = bpp_of(BPC);
    let round_bits_v: u32 = if BPC == 12 { 9 } else { 11 };
    let init_v = i32x4_splat(wiener_v_init(BPC));
    let v_max = i16x8_splat(((1 << BPC) - 1) as i16);

    let f0 = i16x8_splat(fv[0]);
    let f1 = i16x8_splat(fv[1]);
    let f2 = i16x8_splat(fv[2]);
    let f3 = i16x8_splat(fv[3]);
    let f4 = i16x8_splat(fv[4]);
    let f5 = i16x8_splat(fv[5]);
    let f6 = i16x8_splat(fv[6]);

    let mut x = 0usize;
    while x + 8 <= w {
        let v1 = load_u16x8(r1, x);
        let v2 = load_u16x8(r2, x);
        let v3 = load_u16x8(r3, x);
        let v4 = load_u16x8(r4, x);
        let v5 = load_u16x8(r5, x);

        let mut acc_lo = i32x4_add(init_v, i32x4_extmul_low_i16x8(v1, f1));
        acc_lo = i32x4_add(acc_lo, i32x4_extmul_low_i16x8(v2, f2));
        acc_lo = i32x4_add(acc_lo, i32x4_extmul_low_i16x8(v3, f3));
        acc_lo = i32x4_add(acc_lo, i32x4_extmul_low_i16x8(v4, f4));
        acc_lo = i32x4_add(acc_lo, i32x4_extmul_low_i16x8(v5, f5));

        let mut acc_hi = i32x4_add(init_v, i32x4_extmul_high_i16x8(v1, f1));
        acc_hi = i32x4_add(acc_hi, i32x4_extmul_high_i16x8(v2, f2));
        acc_hi = i32x4_add(acc_hi, i32x4_extmul_high_i16x8(v3, f3));
        acc_hi = i32x4_add(acc_hi, i32x4_extmul_high_i16x8(v4, f4));
        acc_hi = i32x4_add(acc_hi, i32x4_extmul_high_i16x8(v5, f5));

        if !FIVE {
            let v0 = load_u16x8(r0, x);
            let v6 = load_u16x8(r6, x);
            acc_lo = i32x4_add(acc_lo, i32x4_extmul_low_i16x8(v0, f0));
            acc_lo = i32x4_add(acc_lo, i32x4_extmul_low_i16x8(v6, f6));
            acc_hi = i32x4_add(acc_hi, i32x4_extmul_high_i16x8(v0, f0));
            acc_hi = i32x4_add(acc_hi, i32x4_extmul_high_i16x8(v6, f6));
        }

        pack_and_store_8::<BPC>(
            dst,
            dst_off + x * bpp,
            i32x4_shr(acc_lo, round_bits_v),
            i32x4_shr(acc_hi, round_bits_v),
            v_max,
        );
        x += 8;
    }

    let init_s = wiener_v_init(BPC);
    while x < w {
        let mut sum = init_s
            + i32::from(r1[x]) * i32::from(fv[1])
            + i32::from(r2[x]) * i32::from(fv[2])
            + i32::from(r3[x]) * i32::from(fv[3])
            + i32::from(r4[x]) * i32::from(fv[4])
            + i32::from(r5[x]) * i32::from(fv[5]);
        if !FIVE {
            sum += i32::from(r0[x]) * i32::from(fv[0]) + i32::from(r6[x]) * i32::from(fv[6]);
        }
        store_scalar::<BPC>(
            dst,
            dst_off + x * bpp,
            iclip_pixel::<BPC>(sum >> round_bits_v),
        );
        x += 1;
    }
}

struct WienerState {
    hor: [[u16; REST_UNIT_STRIDE + 8]; 6],
    tmp: [u16; REST_UNIT_STRIDE + 8],
    ptrs: [usize; 7],
}

impl WienerState {
    #[target_feature(enable = "simd128")]
    fn filter_h<const BPC: u32, const FIVE: bool>(
        &mut self,
        c: &Ctx<'_>,
        dst: &[u8],
        row: RowRef,
        r: usize,
        fh: &[i16; 8],
    ) {
        let (left, src) = c.row::<BPC>(dst, row);
        wiener_filter_h_wasm::<BPC, FIVE>(&mut self.hor[r], left, src, fh, c.w, c.edges);
    }

    #[target_feature(enable = "simd128")]
    fn filter_v<const BPC: u32, const FIVE: bool>(
        &mut self,
        dst: &mut [u8],
        c: &Ctx<'_>,
        p_row: usize,
        fv: &[i16; 8],
    ) {
        wiener_vert_pass_wasm::<BPC, FIVE>(
            dst,
            p_row * c.stride,
            &self.hor[self.ptrs[0]],
            &self.hor[self.ptrs[1]],
            &self.hor[self.ptrs[2]],
            &self.hor[self.ptrs[3]],
            &self.hor[self.ptrs[4]],
            &self.hor[self.ptrs[5]],
            &self.hor[self.ptrs[5]],
            fv,
            c.w,
        );
        self.ptrs.copy_within(1..6, 0);
    }

    #[target_feature(enable = "simd128")]
    fn filter_hv<const BPC: u32, const FIVE: bool>(
        &mut self,
        dst: &mut [u8],
        c: &Ctx<'_>,
        p_row: usize,
        row: RowRef,
        filter: &[[i16; 8]; 2],
    ) {
        {
            let (left, src) = c.row::<BPC>(dst, row);
            wiener_filter_h_wasm::<BPC, FIVE>(&mut self.tmp, left, src, &filter[0], c.w, c.edges);
        }
        wiener_vert_pass_wasm::<BPC, FIVE>(
            dst,
            p_row * c.stride,
            &self.hor[self.ptrs[0]],
            &self.hor[self.ptrs[1]],
            &self.hor[self.ptrs[2]],
            &self.hor[self.ptrs[3]],
            &self.hor[self.ptrs[4]],
            &self.hor[self.ptrs[5]],
            &self.tmp,
            &filter[1],
            c.w,
        );
        let dst_row = self.ptrs[6];
        let w = c.w;
        self.hor[dst_row][..w].copy_from_slice(&self.tmp[..w]);
        self.ptrs.copy_within(1..7, 0);
        self.ptrs[6] = self.ptrs[0];
    }
}

#[target_feature(enable = "simd128")]
fn wiener_wasm<const BPC: u32, const FIVE: bool>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    lpf: &[u8],
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    filter: &[[i16; 8]; 2],
    edges: LrEdgeFlags,
) {
    let c = Ctx {
        left,
        lpf,
        stride,
        lpf_stride,
        lpf_x,
        w,
        edges,
    };
    let fh = &filter[0];
    let fv = &filter[1];
    let mut st = WienerState {
        hor: [[0; REST_UNIT_STRIDE + 8]; 6],
        tmp: [0; REST_UNIT_STRIDE + 8],
        ptrs: [0; 7],
    };
    let mut h = h;
    let mut src_row = 0usize;
    let mut p_row = 0usize;

    let tail: usize = 'akis: {
        if c.edges.contains(LrEdgeFlags::HAVE_TOP) {
            st.ptrs[..6].copy_from_slice(&[0, 0, 1, 2, 2, 2]);
            st.filter_h::<BPC, FIVE>(&c, dst, RowRef::Lpf(0), 0, fh);
            st.filter_h::<BPC, FIVE>(&c, dst, RowRef::Lpf(1), 1, fh);
            st.filter_h::<BPC, FIVE>(&c, dst, RowRef::Main(src_row), 2, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 1;
            }
            st.ptrs[4] = 3;
            st.ptrs[5] = 3;
            st.filter_h::<BPC, FIVE>(&c, dst, RowRef::Main(src_row), 3, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 2;
            }
            st.ptrs[5] = 4;
            st.filter_h::<BPC, FIVE>(&c, dst, RowRef::Main(src_row), 4, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
        } else {
            st.ptrs[..6].copy_from_slice(&[0; 6]);
            st.filter_h::<BPC, FIVE>(&c, dst, RowRef::Main(src_row), 0, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 1;
            }
            st.ptrs[4] = 1;
            st.ptrs[5] = 1;
            st.filter_h::<BPC, FIVE>(&c, dst, RowRef::Main(src_row), 1, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 2;
            }
            st.ptrs[5] = 2;
            st.filter_h::<BPC, FIVE>(&c, dst, RowRef::Main(src_row), 2, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
            st.ptrs[6] = 3;
            st.filter_hv::<BPC, FIVE>(dst, &c, p_row, RowRef::Main(src_row), filter);
            src_row += 1;
            p_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
            st.ptrs[6] = 4;
            st.filter_hv::<BPC, FIVE>(dst, &c, p_row, RowRef::Main(src_row), filter);
            src_row += 1;
            p_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
        }

        st.ptrs[6] = st.ptrs[5] + 1;
        loop {
            st.filter_hv::<BPC, FIVE>(dst, &c, p_row, RowRef::Main(src_row), filter);
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
        st.filter_hv::<BPC, FIVE>(dst, &c, p_row, RowRef::Lpf(6), filter);
        p_row += 1;
        st.filter_hv::<BPC, FIVE>(dst, &c, p_row, RowRef::Lpf(7), filter);
        p_row += 1;
        1
    };

    for _ in 0..tail {
        st.filter_v::<BPC, FIVE>(dst, &c, p_row, fv);
        p_row += 1;
    }
}

// ============================================================================
// SGR (5×5, 3×3, mix — 8, 10 ve 12 bpc)
// ============================================================================

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

#[target_feature(enable = "simd128")]
fn box3_row_h_wasm<const BPC: u32>(
    sumsq: &mut [i32; BUF_STRIDE],
    sum: &mut [i32; BUF_STRIDE],
    left: Option<Src<'_>>,
    src: Src<'_>,
    w: usize,
    edges: LrEdgeFlags,
) {
    let mut small = [0u16; 80];
    let mut large;
    let line: &mut [u16] = if w <= 64 {
        &mut small
    } else {
        large = [0u16; LR_MAX_W + 16];
        &mut large
    };
    unpack_row_padded::<BPC, 2>(line, left, src, w, edges);

    let n = w + 2;
    let mut j = 0usize;
    while j < n {
        let p0 = load_u16x8(line, j);
        let p1 = load_u16x8(line, j + 8);
        let a = p0;
        let b = u16x8_shuffle::<1, 2, 3, 4, 5, 6, 7, 8>(p0, p1);
        let c = u16x8_shuffle::<2, 3, 4, 5, 6, 7, 8, 9>(p0, p1);

        let s16 = u16x8_add(u16x8_add(a, b), c);
        store_i32x4(sum, j, u32x4_extend_low_u16x8(s16));
        store_i32x4(sum, j + 4, u32x4_extend_high_u16x8(s16));

        let sq_lo = u32x4_add(
            u32x4_add(u32x4_extmul_low_u16x8(a, a), u32x4_extmul_low_u16x8(b, b)),
            u32x4_extmul_low_u16x8(c, c),
        );
        let sq_hi = u32x4_add(
            u32x4_add(u32x4_extmul_high_u16x8(a, a), u32x4_extmul_high_u16x8(b, b)),
            u32x4_extmul_high_u16x8(c, c),
        );
        store_i32x4(sumsq, j, sq_lo);
        store_i32x4(sumsq, j + 4, sq_hi);
        j += 8;
    }
}

#[target_feature(enable = "simd128")]
fn box5_row_h_wasm<const BPC: u32>(
    sumsq: &mut [i32; BUF_STRIDE],
    sum: &mut [i32; BUF_STRIDE],
    left: Option<Src<'_>>,
    src: Src<'_>,
    w: usize,
    edges: LrEdgeFlags,
) {
    let mut small = [0u16; 80];
    let mut large;
    let line: &mut [u16] = if w <= 64 {
        &mut small
    } else {
        large = [0u16; LR_MAX_W + 16];
        &mut large
    };
    unpack_row_padded::<BPC, 3>(line, left, src, w, edges);

    let n = w + 2;
    let mut j = 0usize;
    while j < n {
        let p0 = load_u16x8(line, j);
        let p1 = load_u16x8(line, j + 8);
        let a = p0;
        let b = u16x8_shuffle::<1, 2, 3, 4, 5, 6, 7, 8>(p0, p1);
        let c = u16x8_shuffle::<2, 3, 4, 5, 6, 7, 8, 9>(p0, p1);
        let d = u16x8_shuffle::<3, 4, 5, 6, 7, 8, 9, 10>(p0, p1);
        let e = u16x8_shuffle::<4, 5, 6, 7, 8, 9, 10, 11>(p0, p1);

        // 12-bit'te en fazla 5 * 4095 = 20475 < 65535 (u16 taşmaz).
        let s16 = u16x8_add(u16x8_add(u16x8_add(a, b), u16x8_add(c, d)), e);
        store_i32x4(sum, j, u32x4_extend_low_u16x8(s16));
        store_i32x4(sum, j + 4, u32x4_extend_high_u16x8(s16));

        let sq_lo = u32x4_add(
            u32x4_add(
                u32x4_add(u32x4_extmul_low_u16x8(a, a), u32x4_extmul_low_u16x8(b, b)),
                u32x4_add(u32x4_extmul_low_u16x8(c, c), u32x4_extmul_low_u16x8(d, d)),
            ),
            u32x4_extmul_low_u16x8(e, e),
        );
        let sq_hi = u32x4_add(
            u32x4_add(
                u32x4_add(u32x4_extmul_high_u16x8(a, a), u32x4_extmul_high_u16x8(b, b)),
                u32x4_add(u32x4_extmul_high_u16x8(c, c), u32x4_extmul_high_u16x8(d, d)),
            ),
            u32x4_extmul_high_u16x8(e, e),
        );
        store_i32x4(sumsq, j, sq_lo);
        store_i32x4(sumsq, j + 4, sq_hi);
        j += 8;
    }
}

#[inline]
#[target_feature(enable = "simd128")]
fn calc_ab4_vec<const BPC: u32>(
    av_raw: v128,
    bv_raw: v128,
    s: u32,
    n: i32,
    one_by_x: u32,
) -> (v128, v128) {
    let zero = i32x4_splat(0);
    let z_round = u32x4_splat(1 << 19);
    let z_max = u32x4_splat(255);
    let aa_round = u32x4_splat(1 << 11);

    let (a, b) = if BPC == 10 {
        (
            i32x4_shr(i32x4_add(av_raw, i32x4_splat(8)), 4),
            i32x4_shr(i32x4_add(bv_raw, i32x4_splat(2)), 2),
        )
    } else if BPC == 12 {
        (
            i32x4_shr(i32x4_add(av_raw, i32x4_splat(128)), 8),
            i32x4_shr(i32x4_add(bv_raw, i32x4_splat(8)), 4),
        )
    } else {
        (av_raw, bv_raw)
    };

    let p = i32x4_max(
        i32x4_sub(i32x4_mul(a, i32x4_splat(n)), i32x4_mul(b, b)),
        zero,
    );
    let z = u32x4_min(
        u32x4_shr(u32x4_add(i32x4_mul(p, u32x4_splat(s)), z_round), 20),
        z_max,
    );

    let x_v = i32x4(
        i32::from(SGR_X_BY_X[i32x4_extract_lane::<0>(z) as usize]),
        i32::from(SGR_X_BY_X[i32x4_extract_lane::<1>(z) as usize]),
        i32::from(SGR_X_BY_X[i32x4_extract_lane::<2>(z) as usize]),
        i32::from(SGR_X_BY_X[i32x4_extract_lane::<3>(z) as usize]),
    );
    let aa_out = u32x4_shr(
        u32x4_add(
            i32x4_mul(i32x4_mul(x_v, bv_raw), u32x4_splat(one_by_x)),
            aa_round,
        ),
        12,
    );
    (aa_out, x_v)
}

#[target_feature(enable = "simd128")]
fn box3_vert_wasm<const BPC: u32, const K: usize>(
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
    let len = w + 2;
    let mut x = 0usize;
    while x < len {
        let vq = i32x4_add(
            i32x4_add(load_i32x4(q0, x), load_i32x4(q1, x)),
            load_i32x4(q2, x),
        );
        let vs = i32x4_add(
            i32x4_add(load_i32x4(s0, x), load_i32x4(s1, x)),
            load_i32x4(s2, x),
        );
        let (aa_out, x_v) = calc_ab4_vec::<BPC>(vq, vs, s, 9, 455);
        store_i32x4(oa, x, aa_out);
        store_i32x4(ob, x, x_v);
        x += 4;
    }
    sp.rotate_left(1);
}

#[target_feature(enable = "simd128")]
fn box5_vert_wasm<const BPC: u32>(
    sums: &Rows<5>,
    sp: &mut [usize; 5],
    ab: &mut Rows<2>,
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
    let len = w + 2;
    let mut x = 0usize;
    while x < len {
        let vq = i32x4_add(
            i32x4_add(
                i32x4_add(load_i32x4(q[0], x), load_i32x4(q[1], x)),
                i32x4_add(load_i32x4(q[2], x), load_i32x4(q[3], x)),
            ),
            load_i32x4(q[4], x),
        );
        let vs = i32x4_add(
            i32x4_add(
                i32x4_add(load_i32x4(t[0], x), load_i32x4(t[1], x)),
                i32x4_add(load_i32x4(t[2], x), load_i32x4(t[3], x)),
            ),
            load_i32x4(t[4], x),
        );
        let (aa_out, x_v) = calc_ab4_vec::<BPC>(vq, vs, s, 25, 164);
        store_i32x4(oa, x, aa_out);
        store_i32x4(ob, x, x_v);
        x += 4;
    }
    sp.rotate_left(2);
}

#[target_feature(enable = "simd128")]
fn finish_filter_row1_wasm<const BPC: u32, const K: usize>(
    tmp: &mut [i32],
    src: Src<'_>,
    ab: &Rows<K>,
    ap: [usize; 3],
    w: usize,
) {
    let (b0, b1, b2) = (&ab.s[ap[0]], &ab.s[ap[1]], &ab.s[ap[2]]);
    let (a0, a1, a2) = (&ab.sq[ap[0]], &ab.sq[ap[1]], &ab.sq[ap[2]]);
    let r256 = i32x4_splat(1 << 8);
    let c3 = i32x4_splat(3);
    let c4 = i32x4_splat(4);
    let bpp = bpp_of(BPC);

    let mut i = 0usize;
    while i + 8 <= w {
        let s8 = load_pixels_8::<BPC>(src.buf, src.base + i * bpp);
        let s_pair = [u32x4_extend_low_u16x8(s8), u32x4_extend_high_u16x8(s8)];
        for (half, s_v) in s_pair.into_iter().enumerate() {
            let k = i + half * 4;
            let j = k + 1;
            let b_cross = i32x4_add(
                i32x4_add(load_i32x4(b1, j), load_i32x4(b1, j - 1)),
                i32x4_add(
                    load_i32x4(b1, j + 1),
                    i32x4_add(load_i32x4(b0, j), load_i32x4(b2, j)),
                ),
            );
            let b_diag = i32x4_add(
                i32x4_add(load_i32x4(b0, j - 1), load_i32x4(b2, j - 1)),
                i32x4_add(load_i32x4(b0, j + 1), load_i32x4(b2, j + 1)),
            );
            let a_sum = i32x4_add(i32x4_mul(b_cross, c4), i32x4_mul(b_diag, c3));

            let a_cross = i32x4_add(
                i32x4_add(load_i32x4(a1, j), load_i32x4(a1, j - 1)),
                i32x4_add(
                    load_i32x4(a1, j + 1),
                    i32x4_add(load_i32x4(a0, j), load_i32x4(a2, j)),
                ),
            );
            let a_diag = i32x4_add(
                i32x4_add(load_i32x4(a0, j - 1), load_i32x4(a2, j - 1)),
                i32x4_add(load_i32x4(a0, j + 1), load_i32x4(a2, j + 1)),
            );
            let b_sum = i32x4_add(i32x4_mul(a_cross, c4), i32x4_mul(a_diag, c3));
            let out = i32x4_shr(i32x4_add(i32x4_sub(b_sum, i32x4_mul(a_sum, s_v)), r256), 9);
            store_i32x4(tmp, k, out);
        }
        i += 8;
    }
    while i < w {
        let j = i + 1;
        let a = (b1[j] + b1[j - 1] + b1[j + 1] + b0[j] + b2[j]) * 4
            + (b0[j - 1] + b2[j - 1] + b0[j + 1] + b2[j + 1]) * 3;
        let b = (a1[j] + a1[j - 1] + a1[j + 1] + a0[j] + a2[j]) * 4
            + (a0[j - 1] + a2[j - 1] + a0[j + 1] + a2[j + 1]) * 3;
        tmp[i] = coef::<BPC>((b - a * src.at::<BPC>(i as isize) + (1 << 8)) >> 9);
        i += 1;
    }
}

#[target_feature(enable = "simd128")]
fn finish_filter2_wasm<const BPC: u32>(
    tmp: &mut [i32; 2 * FILTER_OUT_STRIDE],
    src0: Src<'_>,
    src1: Src<'_>,
    ab: &Rows<2>,
    ap: [usize; 2],
    w: usize,
    h: usize,
) {
    let (b0, b1) = (&ab.s[ap[0]], &ab.s[ap[1]]);
    let (a0, a1) = (&ab.sq[ap[0]], &ab.sq[ap[1]]);
    let r256 = i32x4_splat(1 << 8);
    let r128 = i32x4_splat(1 << 7);
    let c5 = i32x4_splat(5);
    let c6 = i32x4_splat(6);
    let bpp = bpp_of(BPC);

    let mut i = 0usize;
    while i + 8 <= w {
        let s8 = load_pixels_8::<BPC>(src0.buf, src0.base + i * bpp);
        let s_pair = [u32x4_extend_low_u16x8(s8), u32x4_extend_high_u16x8(s8)];
        for (half, s0_v) in s_pair.into_iter().enumerate() {
            let k = i + half * 4;
            let j = k + 1;
            let b_mid = i32x4_add(load_i32x4(b0, j), load_i32x4(b1, j));
            let b_side = i32x4_add(
                i32x4_add(load_i32x4(b0, j - 1), load_i32x4(b1, j - 1)),
                i32x4_add(load_i32x4(b0, j + 1), load_i32x4(b1, j + 1)),
            );
            let a_sum = i32x4_add(i32x4_mul(b_mid, c6), i32x4_mul(b_side, c5));

            let a_mid = i32x4_add(load_i32x4(a0, j), load_i32x4(a1, j));
            let a_side = i32x4_add(
                i32x4_add(load_i32x4(a0, j - 1), load_i32x4(a1, j - 1)),
                i32x4_add(load_i32x4(a0, j + 1), load_i32x4(a1, j + 1)),
            );
            let b_sum = i32x4_add(i32x4_mul(a_mid, c6), i32x4_mul(a_side, c5));
            let out0 = i32x4_shr(i32x4_add(i32x4_sub(b_sum, i32x4_mul(a_sum, s0_v)), r256), 9);
            store_i32x4(tmp, k, out0);
        }
        i += 8;
    }
    while i < w {
        let j = i + 1;
        let a = (b0[j] + b1[j]) * 6 + (b0[j - 1] + b1[j - 1] + b0[j + 1] + b1[j + 1]) * 5;
        let b = (a0[j] + a1[j]) * 6 + (a0[j - 1] + a1[j - 1] + a0[j + 1] + a1[j + 1]) * 5;
        tmp[i] = coef::<BPC>((b - a * src0.at::<BPC>(i as isize) + (1 << 8)) >> 9);
        i += 1;
    }

    if h <= 1 {
        return;
    }

    let mut i = 0usize;
    while i + 8 <= w {
        let s8 = load_pixels_8::<BPC>(src1.buf, src1.base + i * bpp);
        let s_pair = [u32x4_extend_low_u16x8(s8), u32x4_extend_high_u16x8(s8)];
        for (half, s1_v) in s_pair.into_iter().enumerate() {
            let k = i + half * 4;
            let j = k + 1;
            let a_sum = i32x4_add(
                i32x4_mul(load_i32x4(b1, j), c6),
                i32x4_mul(i32x4_add(load_i32x4(b1, j - 1), load_i32x4(b1, j + 1)), c5),
            );
            let b_sum = i32x4_add(
                i32x4_mul(load_i32x4(a1, j), c6),
                i32x4_mul(i32x4_add(load_i32x4(a1, j - 1), load_i32x4(a1, j + 1)), c5),
            );
            let out1 = i32x4_shr(i32x4_add(i32x4_sub(b_sum, i32x4_mul(a_sum, s1_v)), r128), 8);
            store_i32x4(tmp, FILTER_OUT_STRIDE + k, out1);
        }
        i += 8;
    }
    while i < w {
        let j = i + 1;
        let a = b1[j] * 6 + (b1[j - 1] + b1[j + 1]) * 5;
        let b = a1[j] * 6 + (a1[j - 1] + a1[j + 1]) * 5;
        tmp[FILTER_OUT_STRIDE + i] =
            coef::<BPC>((b - a * src1.at::<BPC>(i as isize) + (1 << 7)) >> 8);
        i += 1;
    }
}

#[target_feature(enable = "simd128")]
fn weighted_row1_wasm<const BPC: u32>(dst: &mut [u8], off: usize, t1: &[i32], w: usize, w1: i32) {
    let bpp = bpp_of(BPC);
    let r1024 = i32x4_splat(1 << 10);
    let w1_v = i32x4_splat(w1);
    let v_max = i16x8_splat(((1 << BPC) - 1) as i16);

    let mut i = 0usize;
    while i + 8 <= w {
        let d_u16 = load_pixels_8::<BPC>(dst, off + i * bpp);
        let d_lo = u32x4_extend_low_u16x8(d_u16);
        let d_hi = u32x4_extend_high_u16x8(d_u16);
        let t_lo = i32x4_mul(load_i32x4(t1, i), w1_v);
        let t_hi = i32x4_mul(load_i32x4(t1, i + 4), w1_v);
        let o_lo = i32x4_add(d_lo, i32x4_shr(i32x4_add(t_lo, r1024), 11));
        let o_hi = i32x4_add(d_hi, i32x4_shr(i32x4_add(t_hi, r1024), 11));
        pack_and_store_8::<BPC>(dst, off + i * bpp, o_lo, o_hi, v_max);
        i += 8;
    }
    while i < w {
        let v = w1 * t1[i];
        let o = off + i * bpp;
        let d = load_scalar::<BPC>(dst, o);
        store_scalar::<BPC>(dst, o, iclip_pixel::<BPC>(d + ((v + (1 << 10)) >> 11)));
        i += 1;
    }
}

#[target_feature(enable = "simd128")]
fn weighted2_wasm<const BPC: u32>(
    dst: &mut [u8],
    off: usize,
    stride: usize,
    t1: &[i32; 2 * FILTER_OUT_STRIDE],
    t2: &[i32; 2 * FILTER_OUT_STRIDE],
    w: usize,
    h: usize,
    w0: i32,
    w1: i32,
) {
    let bpp = bpp_of(BPC);
    let r1024 = i32x4_splat(1 << 10);
    let w0_v = i32x4_splat(w0);
    let w1_v = i32x4_splat(w1);
    let v_max = i16x8_splat(((1 << BPC) - 1) as i16);

    for j in 0..h {
        let row = off + j * stride;
        let t = j * FILTER_OUT_STRIDE;
        let mut i = 0usize;
        while i + 8 <= w {
            let d_u16 = load_pixels_8::<BPC>(dst, row + i * bpp);
            let d_lo = u32x4_extend_low_u16x8(d_u16);
            let d_hi = u32x4_extend_high_u16x8(d_u16);
            let v_lo = i32x4_add(
                i32x4_mul(load_i32x4(t1, t + i), w0_v),
                i32x4_mul(load_i32x4(t2, t + i), w1_v),
            );
            let v_hi = i32x4_add(
                i32x4_mul(load_i32x4(t1, t + i + 4), w0_v),
                i32x4_mul(load_i32x4(t2, t + i + 4), w1_v),
            );
            let o_lo = i32x4_add(d_lo, i32x4_shr(i32x4_add(v_lo, r1024), 11));
            let o_hi = i32x4_add(d_hi, i32x4_shr(i32x4_add(v_hi, r1024), 11));
            pack_and_store_8::<BPC>(dst, row + i * bpp, o_lo, o_hi, v_max);
            i += 8;
        }
        while i < w {
            let v = w0 * t1[t + i] + w1 * t2[t + i];
            let o = row + i * bpp;
            let d = load_scalar::<BPC>(dst, o);
            store_scalar::<BPC>(dst, o, iclip_pixel::<BPC>(d + ((v + (1 << 10)) >> 11)));
            i += 1;
        }
    }
}

#[inline(always)]
fn dst_src<'a>(dst: &'a [u8], c: &Ctx<'_>, r: usize) -> Src<'a> {
    Src {
        buf: dst,
        base: r * c.stride,
    }
}

#[target_feature(enable = "simd128")]
fn finish1_wasm<const BPC: u32>(
    dst: &mut [u8],
    c: &Ctx<'_>,
    dst_row: &mut usize,
    ab: &Rows<3>,
    ap: &mut [usize; 3],
    tmp: &mut [i32; 2 * FILTER_OUT_STRIDE],
    w1: i32,
) {
    finish_filter_row1_wasm::<BPC, 3>(tmp, dst_src(dst, c, *dst_row), ab, *ap, c.w);
    weighted_row1_wasm::<BPC>(dst, *dst_row * c.stride, &tmp[..], c.w, w1);
    *dst_row += 1;
    ap.rotate_left(1);
}

#[target_feature(enable = "simd128")]
fn finish2_wasm<const BPC: u32>(
    dst: &mut [u8],
    c: &Ctx<'_>,
    dst_row: &mut usize,
    ab: &Rows<2>,
    ap: &mut [usize; 2],
    tmp: &mut [i32; 2 * FILTER_OUT_STRIDE],
    h: usize,
    w0: i32,
) {
    let r = *dst_row;
    let r1 = if h > 1 { r + 1 } else { r };
    finish_filter2_wasm::<BPC>(
        tmp,
        dst_src(dst, c, r),
        dst_src(dst, c, r1),
        ab,
        *ap,
        c.w,
        h,
    );
    weighted_row1_wasm::<BPC>(dst, r * c.stride, &tmp[..], c.w, w0);
    *dst_row += 1;
    if h > 1 {
        weighted_row1_wasm::<BPC>(dst, (r + 1) * c.stride, &tmp[FILTER_OUT_STRIDE..], c.w, w0);
        *dst_row += 1;
    }
    ap.rotate_left(1);
}

#[target_feature(enable = "simd128")]
fn finish_mix_wasm<const BPC: u32>(
    dst: &mut [u8],
    c: &Ctx<'_>,
    dst_row: &mut usize,
    ab5: &Rows<2>,
    ap5: &mut [usize; 2],
    ab3: &Rows<4>,
    ap3: &mut [usize; 4],
    tmp5: &mut [i32; 2 * FILTER_OUT_STRIDE],
    tmp3: &mut [i32; 2 * FILTER_OUT_STRIDE],
    h: usize,
    w0: i32,
    w1: i32,
) {
    let r = *dst_row;
    let r1 = if h > 1 { r + 1 } else { r };
    finish_filter2_wasm::<BPC>(
        tmp5,
        dst_src(dst, c, r),
        dst_src(dst, c, r1),
        ab5,
        *ap5,
        c.w,
        h,
    );
    finish_filter_row1_wasm::<BPC, 4>(
        &mut tmp3[..],
        dst_src(dst, c, r),
        ab3,
        [ap3[0], ap3[1], ap3[2]],
        c.w,
    );
    if h > 1 {
        finish_filter_row1_wasm::<BPC, 4>(
            &mut tmp3[FILTER_OUT_STRIDE..],
            dst_src(dst, c, r + 1),
            ab3,
            [ap3[1], ap3[2], ap3[3]],
            c.w,
        );
    }
    weighted2_wasm::<BPC>(dst, r * c.stride, c.stride, tmp5, tmp3, c.w, h, w0, w1);
    *dst_row += h;
    ap5.rotate_left(1);
    ap3.rotate_left(1);
}

#[target_feature(enable = "simd128")]
fn sgr_3x3_impl<const BPC: u32>(dst: &mut [u8], c: &Ctx<'_>, h: usize, s1: u32, w1: i32) {
    enum Kuyruk {
        Bitti,
        Vert2,
        Vert1,
    }
    let w = c.w;
    let mut sums = Rows::<3>::new();
    let mut ab = Rows::<3>::new();
    let mut sp: [usize; 3];
    let mut ap = [0usize, 1, 2];
    let mut tmp = [0i32; 2 * FILTER_OUT_STRIDE];
    let mut h = h;
    let mut sr = 0usize;
    let mut dr = 0usize;

    let kuyruk = 'akis: {
        if c.edges.contains(LrEdgeFlags::HAVE_TOP) {
            sp = [0, 1, 2];
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Lpf(0));
                box3_row_h_wasm::<BPC>(&mut sums.sq[0], &mut sums.s[0], l, s, w, c.edges);
                let (l, s) = c.row::<BPC>(dst, RowRef::Lpf(1));
                box3_row_h_wasm::<BPC>(&mut sums.sq[1], &mut sums.s[1], l, s, w, c.edges);
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_wasm::<BPC>(&mut sums.sq[sp[2]], &mut sums.s[sp[2]], l, s, w, c.edges);
            }
            box3_vert_wasm::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_wasm::<BPC>(&mut sums.sq[sp[2]], &mut sums.s[sp[2]], l, s, w, c.edges);
            }
            box3_vert_wasm::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
        } else {
            sp = [0, 0, 0];
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_wasm::<BPC>(&mut sums.sq[0], &mut sums.s[0], l, s, w, c.edges);
            }
            sr += 1;
            box3_vert_wasm::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp[2] = 1;
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_wasm::<BPC>(&mut sums.sq[sp[2]], &mut sums.s[sp[2]], l, s, w, c.edges);
            }
            box3_vert_wasm::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp[2] = 2;
        }

        loop {
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_wasm::<BPC>(&mut sums.sq[sp[2]], &mut sums.s[sp[2]], l, s, w, c.edges);
            }
            box3_vert_wasm::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            finish1_wasm::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
            h -= 1;
            if h == 0 {
                break;
            }
        }

        if !c.edges.contains(LrEdgeFlags::HAVE_BOTTOM) {
            break 'akis Kuyruk::Vert2;
        }
        {
            let (l, s) = c.row::<BPC>(dst, RowRef::Lpf(6));
            box3_row_h_wasm::<BPC>(&mut sums.sq[sp[2]], &mut sums.s[sp[2]], l, s, w, c.edges);
        }
        box3_vert_wasm::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
        finish1_wasm::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        {
            let (l, s) = c.row::<BPC>(dst, RowRef::Lpf(7));
            box3_row_h_wasm::<BPC>(&mut sums.sq[sp[2]], &mut sums.s[sp[2]], l, s, w, c.edges);
        }
        box3_vert_wasm::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
        finish1_wasm::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        Kuyruk::Bitti
    };

    match kuyruk {
        Kuyruk::Bitti => return,
        Kuyruk::Vert2 => {
            sp[2] = sp[1];
            box3_vert_wasm::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            finish1_wasm::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        }
        Kuyruk::Vert1 => {
            sp[2] = sp[1];
            box3_vert_wasm::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            ap.rotate_left(1);
        }
    }
    sp[2] = sp[1];
    box3_vert_wasm::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
    finish1_wasm::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
}

#[target_feature(enable = "simd128")]
fn sgr_5x5_impl<const BPC: u32>(dst: &mut [u8], c: &Ctx<'_>, h: usize, s0: u32, w0: i32) {
    enum Kuyruk {
        Output2,
        Vert2,
        Odd,
        Vert1,
    }
    let w = c.w;
    let mut sums = Rows::<5>::new();
    let mut ab = Rows::<2>::new();
    let mut sp: [usize; 5];
    let mut ap = [0usize, 1];
    let mut tmp = [0i32; 2 * FILTER_OUT_STRIDE];
    let mut h = h;
    let mut sr = 0usize;
    let mut dr = 0usize;

    let kuyruk = 'akis: {
        if c.edges.contains(LrEdgeFlags::HAVE_TOP) {
            sp = [0, 0, 1, 2, 3];
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Lpf(0));
                box5_row_h_wasm::<BPC>(&mut sums.sq[0], &mut sums.s[0], l, s, w, c.edges);
                let (l, s) = c.row::<BPC>(dst, RowRef::Lpf(1));
                box5_row_h_wasm::<BPC>(&mut sums.sq[1], &mut sums.s[1], l, s, w, c.edges);
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box5_row_h_wasm::<BPC>(&mut sums.sq[2], &mut sums.s[2], l, s, w, c.edges);
            }
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box5_row_h_wasm::<BPC>(&mut sums.sq[3], &mut sums.s[3], l, s, w, c.edges);
            }
            sr += 1;
            box5_vert_wasm::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp[3] = 4;
        } else {
            sp = [0; 5];
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box5_row_h_wasm::<BPC>(&mut sums.sq[0], &mut sums.s[0], l, s, w, c.edges);
            }
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp[4] = 1;
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box5_row_h_wasm::<BPC>(&mut sums.sq[1], &mut sums.s[1], l, s, w, c.edges);
            }
            sr += 1;
            box5_vert_wasm::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp[3] = 2;
            sp[4] = 3;
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box5_row_h_wasm::<BPC>(&mut sums.sq[2], &mut sums.s[2], l, s, w, c.edges);
            }
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box5_row_h_wasm::<BPC>(&mut sums.sq[3], &mut sums.s[3], l, s, w, c.edges);
            }
            sr += 1;
            box5_vert_wasm::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2_wasm::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp[3] = 4;
        }

        loop {
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box5_row_h_wasm::<BPC>(&mut sums.sq[sp[3]], &mut sums.s[sp[3]], l, s, w, c.edges);
            }
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box5_row_h_wasm::<BPC>(&mut sums.sq[sp[4]], &mut sums.s[sp[4]], l, s, w, c.edges);
            }
            sr += 1;
            box5_vert_wasm::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2_wasm::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
            h -= 1;
            if h == 0 {
                break;
            }
        }

        if !c.edges.contains(LrEdgeFlags::HAVE_BOTTOM) {
            break 'akis Kuyruk::Vert2;
        }
        {
            let (l, s) = c.row::<BPC>(dst, RowRef::Lpf(6));
            box5_row_h_wasm::<BPC>(&mut sums.sq[sp[3]], &mut sums.s[sp[3]], l, s, w, c.edges);
            let (l, s) = c.row::<BPC>(dst, RowRef::Lpf(7));
            box5_row_h_wasm::<BPC>(&mut sums.sq[sp[4]], &mut sums.s[sp[4]], l, s, w, c.edges);
        }
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
            box5_vert_wasm::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2_wasm::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
            true
        }
        Kuyruk::Vert1 => {
            sp[4] = sp[3];
            box5_vert_wasm::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
            ap.rotate_left(1);
            true
        }
    };
    if output_1 {
        sp[3] = sp[2];
        sp[4] = sp[2];
        box5_vert_wasm::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
        finish2_wasm::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 1, w0);
    } else {
        box5_vert_wasm::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
        finish2_wasm::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
    }
}

#[target_feature(enable = "simd128")]
fn sgr_mix_impl<const BPC: u32>(
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
    let mut sp5: [usize; 5];
    let mut sp3: [usize; 3];
    let mut ap5 = [0usize, 1];
    let mut ap3 = [0usize, 1, 2, 3];
    let mut tmp5 = [0i32; 2 * FILTER_OUT_STRIDE];
    let mut tmp3 = [0i32; 2 * FILTER_OUT_STRIDE];
    let mut h = h;
    let mut sr = 0usize;
    let mut dr = 0usize;

    let kuyruk = 'akis: {
        if c.edges.contains(LrEdgeFlags::HAVE_TOP) {
            sp5 = [0, 0, 1, 2, 3];
            sp3 = [0, 1, 2];
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Lpf(0));
                box3_row_h_wasm::<BPC>(&mut sums3.sq[0], &mut sums3.s[0], l, s, w, c.edges);
                box5_row_h_wasm::<BPC>(&mut sums5.sq[0], &mut sums5.s[0], l, s, w, c.edges);
                let (l, s) = c.row::<BPC>(dst, RowRef::Lpf(1));
                box3_row_h_wasm::<BPC>(&mut sums3.sq[1], &mut sums3.s[1], l, s, w, c.edges);
                box5_row_h_wasm::<BPC>(&mut sums5.sq[1], &mut sums5.s[1], l, s, w, c.edges);
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_wasm::<BPC>(&mut sums3.sq[2], &mut sums3.s[2], l, s, w, c.edges);
                box5_row_h_wasm::<BPC>(&mut sums5.sq[2], &mut sums5.s[2], l, s, w, c.edges);
            }
            sr += 1;
            box3_vert_wasm::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_wasm::<BPC>(
                    &mut sums3.sq[sp3[2]],
                    &mut sums3.s[sp3[2]],
                    l,
                    s,
                    w,
                    c.edges,
                );
                box5_row_h_wasm::<BPC>(&mut sums5.sq[3], &mut sums5.s[3], l, s, w, c.edges);
            }
            sr += 1;
            box5_vert_wasm::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            ap5.rotate_left(1);
            box3_vert_wasm::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp5[3] = 4;
        } else {
            sp5 = [0; 5];
            sp3 = [0; 3];
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_wasm::<BPC>(&mut sums3.sq[0], &mut sums3.s[0], l, s, w, c.edges);
                box5_row_h_wasm::<BPC>(&mut sums5.sq[0], &mut sums5.s[0], l, s, w, c.edges);
            }
            sr += 1;
            box3_vert_wasm::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp5[4] = 1;
            sp3[2] = 1;
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_wasm::<BPC>(&mut sums3.sq[1], &mut sums3.s[1], l, s, w, c.edges);
                box5_row_h_wasm::<BPC>(&mut sums5.sq[1], &mut sums5.s[1], l, s, w, c.edges);
            }
            sr += 1;
            box5_vert_wasm::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            ap5.rotate_left(1);
            box3_vert_wasm::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp5[3] = 2;
            sp5[4] = 3;
            sp3[2] = 2;
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_wasm::<BPC>(&mut sums3.sq[2], &mut sums3.s[2], l, s, w, c.edges);
                box5_row_h_wasm::<BPC>(&mut sums5.sq[2], &mut sums5.s[2], l, s, w, c.edges);
            }
            sr += 1;
            box3_vert_wasm::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_wasm::<BPC>(
                    &mut sums3.sq[sp3[2]],
                    &mut sums3.s[sp3[2]],
                    l,
                    s,
                    w,
                    c.edges,
                );
                box5_row_h_wasm::<BPC>(&mut sums5.sq[3], &mut sums5.s[3], l, s, w, c.edges);
            }
            sr += 1;
            box5_vert_wasm::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert_wasm::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix_wasm::<BPC>(
                dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 2, w0, w1,
            );
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp5[3] = 4;
        }

        loop {
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_wasm::<BPC>(
                    &mut sums3.sq[sp3[2]],
                    &mut sums3.s[sp3[2]],
                    l,
                    s,
                    w,
                    c.edges,
                );
                box5_row_h_wasm::<BPC>(
                    &mut sums5.sq[sp5[3]],
                    &mut sums5.s[sp5[3]],
                    l,
                    s,
                    w,
                    c.edges,
                );
            }
            sr += 1;
            box3_vert_wasm::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_wasm::<BPC>(
                    &mut sums3.sq[sp3[2]],
                    &mut sums3.s[sp3[2]],
                    l,
                    s,
                    w,
                    c.edges,
                );
                box5_row_h_wasm::<BPC>(
                    &mut sums5.sq[sp5[4]],
                    &mut sums5.s[sp5[4]],
                    l,
                    s,
                    w,
                    c.edges,
                );
            }
            sr += 1;
            box5_vert_wasm::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert_wasm::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix_wasm::<BPC>(
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
        {
            let (l, s) = c.row::<BPC>(dst, RowRef::Lpf(6));
            box3_row_h_wasm::<BPC>(
                &mut sums3.sq[sp3[2]],
                &mut sums3.s[sp3[2]],
                l,
                s,
                w,
                c.edges,
            );
            box5_row_h_wasm::<BPC>(
                &mut sums5.sq[sp5[3]],
                &mut sums5.s[sp5[3]],
                l,
                s,
                w,
                c.edges,
            );
        }
        box3_vert_wasm::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
        ap3.rotate_left(1);
        {
            let (l, s) = c.row::<BPC>(dst, RowRef::Lpf(7));
            box3_row_h_wasm::<BPC>(
                &mut sums3.sq[sp3[2]],
                &mut sums3.s[sp3[2]],
                l,
                s,
                w,
                c.edges,
            );
            box5_row_h_wasm::<BPC>(
                &mut sums5.sq[sp5[4]],
                &mut sums5.s[sp5[4]],
                l,
                s,
                w,
                c.edges,
            );
        }
        Kuyruk::Output2
    };

    let output_1 = match kuyruk {
        Kuyruk::Output2 => false,
        Kuyruk::Vert2 => {
            sp5[3] = sp5[2];
            sp5[4] = sp5[2];
            sp3[2] = sp3[1];
            box3_vert_wasm::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            sp3[2] = sp3[1];
            false
        }
        Kuyruk::Odd => {
            sp5[4] = sp5[3];
            sp3[2] = sp3[1];
            box5_vert_wasm::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert_wasm::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix_wasm::<BPC>(
                dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 2, w0, w1,
            );
            true
        }
        Kuyruk::Vert1 => {
            sp5[4] = sp5[3];
            sp3[2] = sp3[1];
            box5_vert_wasm::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            ap5.rotate_left(1);
            box3_vert_wasm::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            true
        }
    };
    if output_1 {
        sp5[3] = sp5[2];
        sp5[4] = sp5[2];
        sp3[2] = sp3[1];
        box5_vert_wasm::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
        box3_vert_wasm::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
        ap3.rotate_left(1);
        finish_mix_wasm::<BPC>(
            dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 1, w0, w1,
        );
    } else {
        box5_vert_wasm::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
        box3_vert_wasm::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
        finish_mix_wasm::<BPC>(
            dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 2, w0, w1,
        );
    }
}
