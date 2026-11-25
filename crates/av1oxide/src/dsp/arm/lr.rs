//! `aarch64` NEON loop restoration (Wiener / SGRPROJ) çekirdekleri (`P54` / `F5d` — `F5-LR`).
//!
//! Referans: `dav1d` `src/arm/looprestoration.h` +
//! `src/arm/64/{looprestoration.S,looprestoration16.S,looprestoration_common.S,looprestoration_tmpl.S}`
//! (`99f3354`).
//! Kapsam (`10` `dispatch-matrix.csv` kaydı = `12` yuva):
//! - `wiener[0..1]` (`7-tap` ve `5-tap`) @ `8 / 10 / 12` bpc (`6` yuva)
//! - `sgr[0..2]` (`5×5`, `3×3`, `mix`) @ `8 / 10` bpc (`6` yuva; `dav1d` ARM64 12-bpc SGR tanımlamaz)

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use core::arch::aarch64::{
    int32x4_t, uint16x8_t, vaddq_s32, vaddq_u16, vaddq_u32, vdupq_n_s16, vdupq_n_s32, vdupq_n_u32,
    vextq_s16, vextq_u16, vget_high_u16, vget_lane_u64, vget_low_s16, vget_low_u16, vmaxq_s32,
    vminq_s32, vminq_u32, vmlal_high_s16, vmlal_high_u16, vmlal_s16, vmlal_u16, vmovl_u8,
    vmovl_u16, vmovn_high_u32, vmovn_u16, vmovn_u32, vmull_high_u16, vmull_u16, vmulq_n_s32,
    vmulq_n_u32, vmulq_s32, vmulq_u32, vreinterpret_u64_u16, vreinterpretq_s16_u16,
    vreinterpretq_s32_u32, vreinterpretq_u8_u16, vreinterpretq_u16_u8, vreinterpretq_u32_s32,
    vshrq_n_s32, vshrq_n_u32, vsubq_s32,
};

use super::giris::{load_i32x4, store_i32x4};
use super::mem::{
    chunk, chunk_mut, load_u8x8, load_u8x16, load_u16x8, store_u8x8, store_u8x16, store_u16x8,
};
use crate::cpu::NeonToken;
use crate::dsp::lr::{LrEdgeFlags, LrFilterFn, LrParams};
use crate::dsp::lr_tables::SGR_X_BY_X;
use crate::dsp::scalar::lr::LR_MAX_W;

const REST_UNIT_STRIDE: usize = 390;
const BUF_STRIDE: usize = 384 + 16;
const FILTER_OUT_STRIDE: usize = 384;

/// `wiener[tap]` (`8 / 10 / 12` bpc) için `NeonToken`-kapılı giriş fonksiyonu.
#[must_use]
pub fn wiener_fn<const BPC: u32>(_neon: NeonToken, tap: usize) -> Option<LrFilterFn> {
    match tap {
        0 => Some(super::giris::lr_wiener::<BPC, 0>),
        1 => Some(super::giris::lr_wiener::<BPC, 1>),
        _ => None,
    }
}

/// `sgr[idx]` (`8 / 10` bpc) için `NeonToken`-kapılı giriş fonksiyonu.
#[must_use]
pub fn sgr_fn<const BPC: u32>(_neon: NeonToken, idx: usize) -> Option<LrFilterFn> {
    if BPC == 12 {
        return None;
    }
    match idx {
        0 => Some(super::giris::lr_sgr::<BPC, 0>),
        1 => Some(super::giris::lr_sgr::<BPC, 1>),
        2 => Some(super::giris::lr_sgr::<BPC, 2>),
        _ => None,
    }
}

#[inline(always)]
const fn bpp_of(bpc: u32) -> usize {
    if bpc == 8 { 1 } else { 2 }
}

#[inline(always)]
fn px<const BPC: u32>(buf: &[u8], off: usize) -> i32 {
    if BPC == 8 {
        i32::from(buf[off])
    } else {
        i32::from(u16::from_le_bytes([buf[off], buf[off + 1]]))
    }
}

#[inline(always)]
fn put<const BPC: u32>(buf: &mut [u8], off: usize, v: i32) {
    if BPC == 8 {
        buf[off] = v as u8;
    } else {
        buf[off..off + 2].copy_from_slice(&(v as u16).to_le_bytes());
    }
}

#[inline(always)]
fn iclip(v: i32, min: i32, max: i32) -> i32 {
    v.max(min).min(max)
}

#[inline(always)]
fn iclip_pixel<const BPC: u32>(v: i32) -> i32 {
    iclip(v, 0, (1 << BPC) - 1)
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
        let off = self.base as isize + x * bpp_of(BPC) as isize;
        px::<BPC>(self.buf, off as usize)
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

#[target_feature(enable = "neon")]
#[inline]
fn load_px8_u16<const BPC: u32>(buf: &[u8], byte_off: usize) -> uint16x8_t {
    if BPC == 8 {
        vmovl_u8(load_u8x8(chunk::<u8, 8>(buf, byte_off)))
    } else {
        vreinterpretq_u16_u8(load_u8x16(chunk::<u8, 16>(buf, byte_off)))
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn store_px8_u16<const BPC: u32>(buf: &mut [u8], byte_off: usize, v: uint16x8_t) {
    if BPC == 8 {
        store_u8x8(chunk_mut::<u8, 8>(buf, byte_off), vmovn_u16(v));
    } else {
        store_u8x16(chunk_mut::<u8, 16>(buf, byte_off), vreinterpretq_u8_u16(v));
    }
}

/// Satırı sol/sağ kenar kuralları uygulanmış dilime açar:
/// `line[REACH + x] = p(x)` (`x = -REACH .. w + REACH`).
#[target_feature(enable = "neon")]
#[inline]
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
        let v = load_px8_u16::<BPC>(src.buf, src.base + x * bpp);
        store_u16x8(chunk_mut::<u16, 8>(line, REACH + x), v);
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

#[target_feature(enable = "neon")]
fn wiener_filter_h_neon<const BPC: u32, const FIVE: bool>(
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
    let round_bits_h = if BPC == 12 { 5 } else { 3 };
    let rounding_off_h = 1 << (round_bits_h - 1);
    let clip_max = (1 << (bitdepth + 8 - round_bits_h)) - 1;
    let init_sum = (1 << (bitdepth + 6)) + rounding_off_h;

    let f0 = vdupq_n_s16(fh[0]);
    let f1 = vdupq_n_s16(fh[1]);
    let f2 = vdupq_n_s16(fh[2]);
    let f3_val = i32::from(fh[3]) + if BPC == 8 { 128 } else { 0 };
    let f3 = vdupq_n_s16(f3_val as i16);
    let f4 = vdupq_n_s16(fh[4]);
    let f5 = vdupq_n_s16(fh[5]);
    let f6 = vdupq_n_s16(fh[6]);

    let init_v = vdupq_n_s32(init_sum);
    let zero = vdupq_n_s32(0);
    let max_v = vdupq_n_s32(clip_max);

    let mut x = 0usize;
    while x < w {
        let p0 = vreinterpretq_s16_u16(load_u16x8(chunk::<u16, 8>(&line, x)));
        let p1 = vreinterpretq_s16_u16(load_u16x8(chunk::<u16, 8>(&line, x + 8)));

        let w1 = vextq_s16::<1>(p0, p1);
        let w2 = vextq_s16::<2>(p0, p1);
        let w3 = vextq_s16::<3>(p0, p1);
        let w4 = vextq_s16::<4>(p0, p1);
        let w5 = vextq_s16::<5>(p0, p1);

        let mut acc_lo = vmlal_s16(init_v, vget_low_s16(w1), vget_low_s16(f1));
        acc_lo = vmlal_s16(acc_lo, vget_low_s16(w2), vget_low_s16(f2));
        acc_lo = vmlal_s16(acc_lo, vget_low_s16(w3), vget_low_s16(f3));
        acc_lo = vmlal_s16(acc_lo, vget_low_s16(w4), vget_low_s16(f4));
        acc_lo = vmlal_s16(acc_lo, vget_low_s16(w5), vget_low_s16(f5));

        let mut acc_hi = vmlal_high_s16(init_v, w1, f1);
        acc_hi = vmlal_high_s16(acc_hi, w2, f2);
        acc_hi = vmlal_high_s16(acc_hi, w3, f3);
        acc_hi = vmlal_high_s16(acc_hi, w4, f4);
        acc_hi = vmlal_high_s16(acc_hi, w5, f5);

        if !FIVE {
            let w0 = p0;
            let w6 = vextq_s16::<6>(p0, p1);
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(w0), vget_low_s16(f0));
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(w6), vget_low_s16(f6));
            acc_hi = vmlal_high_s16(acc_hi, w0, f0);
            acc_hi = vmlal_high_s16(acc_hi, w6, f6);
        }

        let (sh_lo, sh_hi) = if BPC == 12 {
            (vshrq_n_s32::<5>(acc_lo), vshrq_n_s32::<5>(acc_hi))
        } else {
            (vshrq_n_s32::<3>(acc_lo), vshrq_n_s32::<3>(acc_hi))
        };
        let cl_lo = vminq_s32(vmaxq_s32(sh_lo, zero), max_v);
        let cl_hi = vminq_s32(vmaxq_s32(sh_hi, zero), max_v);
        let packed = vmovn_high_u32(
            vmovn_u32(vreinterpretq_u32_s32(cl_lo)),
            vreinterpretq_u32_s32(cl_hi),
        );
        store_u16x8(chunk_mut::<u16, 8>(out, x), packed);
        x += 8;
    }
}

const fn wiener_v_init(bpc: u32) -> i32 {
    let round_bits_v = if bpc == 12 { 9 } else { 11 };
    let rounding_off_v = 1 << (round_bits_v - 1);
    let round_offset = 1 << (bpc as i32 + (round_bits_v - 1));
    -round_offset + rounding_off_v
}

#[target_feature(enable = "neon")]
fn wiener_vert_pass_neon<const BPC: u32, const FIVE: bool>(
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
    let init_v = vdupq_n_s32(wiener_v_init(BPC));
    let zero = vdupq_n_s32(0);
    let max_v = vdupq_n_s32((1 << BPC) - 1);

    let f0 = vdupq_n_s16(fv[0]);
    let f1 = vdupq_n_s16(fv[1]);
    let f2 = vdupq_n_s16(fv[2]);
    let f3 = vdupq_n_s16(fv[3]);
    let f4 = vdupq_n_s16(fv[4]);
    let f5 = vdupq_n_s16(fv[5]);
    let f6 = vdupq_n_s16(fv[6]);

    let mut x = 0usize;
    while x + 8 <= w {
        let v1 = vreinterpretq_s16_u16(load_u16x8(chunk::<u16, 8>(r1, x)));
        let v2 = vreinterpretq_s16_u16(load_u16x8(chunk::<u16, 8>(r2, x)));
        let v3 = vreinterpretq_s16_u16(load_u16x8(chunk::<u16, 8>(r3, x)));
        let v4 = vreinterpretq_s16_u16(load_u16x8(chunk::<u16, 8>(r4, x)));
        let v5 = vreinterpretq_s16_u16(load_u16x8(chunk::<u16, 8>(r5, x)));

        let mut acc_lo = vmlal_s16(init_v, vget_low_s16(v1), vget_low_s16(f1));
        acc_lo = vmlal_s16(acc_lo, vget_low_s16(v2), vget_low_s16(f2));
        acc_lo = vmlal_s16(acc_lo, vget_low_s16(v3), vget_low_s16(f3));
        acc_lo = vmlal_s16(acc_lo, vget_low_s16(v4), vget_low_s16(f4));
        acc_lo = vmlal_s16(acc_lo, vget_low_s16(v5), vget_low_s16(f5));

        let mut acc_hi = vmlal_high_s16(init_v, v1, f1);
        acc_hi = vmlal_high_s16(acc_hi, v2, f2);
        acc_hi = vmlal_high_s16(acc_hi, v3, f3);
        acc_hi = vmlal_high_s16(acc_hi, v4, f4);
        acc_hi = vmlal_high_s16(acc_hi, v5, f5);

        if !FIVE {
            let v0 = vreinterpretq_s16_u16(load_u16x8(chunk::<u16, 8>(r0, x)));
            let v6 = vreinterpretq_s16_u16(load_u16x8(chunk::<u16, 8>(r6, x)));
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(v0), vget_low_s16(f0));
            acc_lo = vmlal_s16(acc_lo, vget_low_s16(v6), vget_low_s16(f6));
            acc_hi = vmlal_high_s16(acc_hi, v0, f0);
            acc_hi = vmlal_high_s16(acc_hi, v6, f6);
        }

        let (sh_lo, sh_hi) = if BPC == 12 {
            (vshrq_n_s32::<9>(acc_lo), vshrq_n_s32::<9>(acc_hi))
        } else {
            (vshrq_n_s32::<11>(acc_lo), vshrq_n_s32::<11>(acc_hi))
        };
        let cl_lo = vminq_s32(vmaxq_s32(sh_lo, zero), max_v);
        let cl_hi = vminq_s32(vmaxq_s32(sh_hi, zero), max_v);
        let out_v = vmovn_high_u32(
            vmovn_u32(vreinterpretq_u32_s32(cl_lo)),
            vreinterpretq_u32_s32(cl_hi),
        );
        store_px8_u16::<BPC>(dst, dst_off + x * bpp, out_v);
        x += 8;
    }

    let round_bits_v = if BPC == 12 { 9 } else { 11 };
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
        put::<BPC>(
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
    #[target_feature(enable = "neon")]
    fn filter_h<const BPC: u32, const FIVE: bool>(
        &mut self,
        c: &Ctx<'_>,
        dst: &[u8],
        row: RowRef,
        r: usize,
        fh: &[i16; 8],
    ) {
        let (left, src) = c.row::<BPC>(dst, row);
        wiener_filter_h_neon::<BPC, FIVE>(&mut self.hor[r], left, src, fh, c.w, c.edges);
    }

    #[target_feature(enable = "neon")]
    fn filter_v<const BPC: u32, const FIVE: bool>(
        &mut self,
        dst: &mut [u8],
        c: &Ctx<'_>,
        p_row: usize,
        fv: &[i16; 8],
    ) {
        wiener_vert_pass_neon::<BPC, FIVE>(
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

    #[target_feature(enable = "neon")]
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
            wiener_filter_h_neon::<BPC, FIVE>(&mut self.tmp, left, src, &filter[0], c.w, c.edges);
        }
        wiener_vert_pass_neon::<BPC, FIVE>(
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

#[target_feature(enable = "neon")]
pub(super) fn wiener_neon<const BPC: u32, const FIVE: bool>(
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
// SGR (5×5, 3×3, mix — 8 ve 10 bpc)
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

#[target_feature(enable = "neon")]
fn box3_row_h_neon<const BPC: u32>(
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
        let p0 = load_u16x8(chunk::<u16, 8>(line, j));
        let p1 = load_u16x8(chunk::<u16, 8>(line, j + 8));
        let a = p0;
        let b = vextq_u16::<1>(p0, p1);
        let c = vextq_u16::<2>(p0, p1);

        let s16 = vaddq_u16(vaddq_u16(a, b), c);
        let s_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s16)));
        let s_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s16)));
        store_i32x4(chunk_mut::<i32, 4>(sum, j), s_lo);
        store_i32x4(chunk_mut::<i32, 4>(sum, j + 4), s_hi);

        let mut sq_lo = vmull_u16(vget_low_u16(a), vget_low_u16(a));
        sq_lo = vmlal_u16(sq_lo, vget_low_u16(b), vget_low_u16(b));
        sq_lo = vmlal_u16(sq_lo, vget_low_u16(c), vget_low_u16(c));

        let mut sq_hi = vmull_high_u16(a, a);
        sq_hi = vmlal_high_u16(sq_hi, b, b);
        sq_hi = vmlal_high_u16(sq_hi, c, c);

        store_i32x4(chunk_mut::<i32, 4>(sumsq, j), vreinterpretq_s32_u32(sq_lo));
        store_i32x4(
            chunk_mut::<i32, 4>(sumsq, j + 4),
            vreinterpretq_s32_u32(sq_hi),
        );
        j += 8;
    }
}

#[target_feature(enable = "neon")]
fn box5_row_h_neon<const BPC: u32>(
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
        let p0 = load_u16x8(chunk::<u16, 8>(line, j));
        let p1 = load_u16x8(chunk::<u16, 8>(line, j + 8));
        let a = p0;
        let b = vextq_u16::<1>(p0, p1);
        let c = vextq_u16::<2>(p0, p1);
        let d = vextq_u16::<3>(p0, p1);
        let e = vextq_u16::<4>(p0, p1);

        let s16 = vaddq_u16(vaddq_u16(vaddq_u16(a, b), vaddq_u16(c, d)), e);
        let s_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s16)));
        let s_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s16)));
        store_i32x4(chunk_mut::<i32, 4>(sum, j), s_lo);
        store_i32x4(chunk_mut::<i32, 4>(sum, j + 4), s_hi);

        let mut sq_lo = vmull_u16(vget_low_u16(a), vget_low_u16(a));
        sq_lo = vmlal_u16(sq_lo, vget_low_u16(b), vget_low_u16(b));
        sq_lo = vmlal_u16(sq_lo, vget_low_u16(c), vget_low_u16(c));
        sq_lo = vmlal_u16(sq_lo, vget_low_u16(d), vget_low_u16(d));
        sq_lo = vmlal_u16(sq_lo, vget_low_u16(e), vget_low_u16(e));

        let mut sq_hi = vmull_high_u16(a, a);
        sq_hi = vmlal_high_u16(sq_hi, b, b);
        sq_hi = vmlal_high_u16(sq_hi, c, c);
        sq_hi = vmlal_high_u16(sq_hi, d, d);
        sq_hi = vmlal_high_u16(sq_hi, e, e);

        store_i32x4(chunk_mut::<i32, 4>(sumsq, j), vreinterpretq_s32_u32(sq_lo));
        store_i32x4(
            chunk_mut::<i32, 4>(sumsq, j + 4),
            vreinterpretq_s32_u32(sq_hi),
        );
        j += 8;
    }
}

#[target_feature(enable = "neon")]
#[inline]
fn calc_ab4_vec<const BPC: u32>(
    av_raw: int32x4_t,
    bv_raw: int32x4_t,
    s: u32,
    n: i32,
    one_by_x: u32,
) -> (int32x4_t, int32x4_t) {
    let zero = vdupq_n_s32(0);
    let z_round = vdupq_n_u32(1 << 19);
    let z_max = vdupq_n_u32(255);
    let aa_round = vdupq_n_u32(1 << 11);

    let (a, b) = if BPC == 10 {
        (
            vshrq_n_s32::<4>(vaddq_s32(av_raw, vdupq_n_s32(8))),
            vshrq_n_s32::<2>(vaddq_s32(bv_raw, vdupq_n_s32(2))),
        )
    } else {
        (av_raw, bv_raw)
    };

    let p = vmaxq_s32(vsubq_s32(vmulq_n_s32(a, n), vmulq_s32(b, b)), zero);
    let z = vminq_u32(
        vshrq_n_u32::<20>(vaddq_u32(vmulq_n_u32(vreinterpretq_u32_s32(p), s), z_round)),
        z_max,
    );

    let z64 = vget_lane_u64::<0>(vreinterpret_u64_u16(vmovn_u32(z)));
    let x_arr = [
        i32::from(SGR_X_BY_X[(z64 & 0xff) as usize]),
        i32::from(SGR_X_BY_X[((z64 >> 16) & 0xff) as usize]),
        i32::from(SGR_X_BY_X[((z64 >> 32) & 0xff) as usize]),
        i32::from(SGR_X_BY_X[((z64 >> 48) & 0xff) as usize]),
    ];
    let x_v = load_i32x4(&x_arr);
    let x_u = vreinterpretq_u32_s32(x_v);
    let bb_u = vreinterpretq_u32_s32(bv_raw);
    let aa_out = vreinterpretq_s32_u32(vshrq_n_u32::<12>(vaddq_u32(
        vmulq_n_u32(vmulq_u32(x_u, bb_u), one_by_x),
        aa_round,
    )));
    (aa_out, x_v)
}

#[target_feature(enable = "neon")]
fn box3_vert_neon<const BPC: u32, const K: usize>(
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
        let vq = vaddq_s32(
            vaddq_s32(
                load_i32x4(chunk::<i32, 4>(q0, x)),
                load_i32x4(chunk::<i32, 4>(q1, x)),
            ),
            load_i32x4(chunk::<i32, 4>(q2, x)),
        );
        let vs = vaddq_s32(
            vaddq_s32(
                load_i32x4(chunk::<i32, 4>(s0, x)),
                load_i32x4(chunk::<i32, 4>(s1, x)),
            ),
            load_i32x4(chunk::<i32, 4>(s2, x)),
        );
        let (aa_out, x_v) = calc_ab4_vec::<BPC>(vq, vs, s, 9, 455);
        store_i32x4(chunk_mut::<i32, 4>(oa, x), aa_out);
        store_i32x4(chunk_mut::<i32, 4>(ob, x), x_v);
        x += 4;
    }
    sp.rotate_left(1);
}

#[target_feature(enable = "neon")]
fn box5_vert_neon<const BPC: u32>(
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
        let vq = vaddq_s32(
            vaddq_s32(
                vaddq_s32(
                    load_i32x4(chunk::<i32, 4>(q[0], x)),
                    load_i32x4(chunk::<i32, 4>(q[1], x)),
                ),
                vaddq_s32(
                    load_i32x4(chunk::<i32, 4>(q[2], x)),
                    load_i32x4(chunk::<i32, 4>(q[3], x)),
                ),
            ),
            load_i32x4(chunk::<i32, 4>(q[4], x)),
        );
        let vs = vaddq_s32(
            vaddq_s32(
                vaddq_s32(
                    load_i32x4(chunk::<i32, 4>(t[0], x)),
                    load_i32x4(chunk::<i32, 4>(t[1], x)),
                ),
                vaddq_s32(
                    load_i32x4(chunk::<i32, 4>(t[2], x)),
                    load_i32x4(chunk::<i32, 4>(t[3], x)),
                ),
            ),
            load_i32x4(chunk::<i32, 4>(t[4], x)),
        );
        let (aa_out, x_v) = calc_ab4_vec::<BPC>(vq, vs, s, 25, 164);
        store_i32x4(chunk_mut::<i32, 4>(oa, x), aa_out);
        store_i32x4(chunk_mut::<i32, 4>(ob, x), x_v);
        x += 4;
    }
    sp.rotate_left(2);
}

#[target_feature(enable = "neon")]
fn finish_filter_row1_neon<const BPC: u32, const K: usize>(
    tmp: &mut [i32],
    src: Src<'_>,
    ab: &Rows<K>,
    ap: [usize; 3],
    w: usize,
) {
    let (b0, b1, b2) = (&ab.s[ap[0]], &ab.s[ap[1]], &ab.s[ap[2]]);
    let (a0, a1, a2) = (&ab.sq[ap[0]], &ab.sq[ap[1]], &ab.sq[ap[2]]);
    let r256 = vdupq_n_s32(1 << 8);
    let bpp = bpp_of(BPC);

    let mut i = 0usize;
    while i + 8 <= w {
        let s8 = load_px8_u16::<BPC>(src.buf, src.base + i * bpp);
        let s_pair = [
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s8))),
            vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s8))),
        ];
        for (half, s_v) in s_pair.into_iter().enumerate() {
            let k = i + half * 4;
            let j = k + 1;
            let b_cross = vaddq_s32(
                vaddq_s32(
                    load_i32x4(chunk::<i32, 4>(b1, j)),
                    load_i32x4(chunk::<i32, 4>(b1, j - 1)),
                ),
                vaddq_s32(
                    load_i32x4(chunk::<i32, 4>(b1, j + 1)),
                    vaddq_s32(
                        load_i32x4(chunk::<i32, 4>(b0, j)),
                        load_i32x4(chunk::<i32, 4>(b2, j)),
                    ),
                ),
            );
            let b_diag = vaddq_s32(
                vaddq_s32(
                    load_i32x4(chunk::<i32, 4>(b0, j - 1)),
                    load_i32x4(chunk::<i32, 4>(b2, j - 1)),
                ),
                vaddq_s32(
                    load_i32x4(chunk::<i32, 4>(b0, j + 1)),
                    load_i32x4(chunk::<i32, 4>(b2, j + 1)),
                ),
            );
            let a_sum = vaddq_s32(vmulq_n_s32(b_cross, 4), vmulq_n_s32(b_diag, 3));

            let a_cross = vaddq_s32(
                vaddq_s32(
                    load_i32x4(chunk::<i32, 4>(a1, j)),
                    load_i32x4(chunk::<i32, 4>(a1, j - 1)),
                ),
                vaddq_s32(
                    load_i32x4(chunk::<i32, 4>(a1, j + 1)),
                    vaddq_s32(
                        load_i32x4(chunk::<i32, 4>(a0, j)),
                        load_i32x4(chunk::<i32, 4>(a2, j)),
                    ),
                ),
            );
            let a_diag = vaddq_s32(
                vaddq_s32(
                    load_i32x4(chunk::<i32, 4>(a0, j - 1)),
                    load_i32x4(chunk::<i32, 4>(a2, j - 1)),
                ),
                vaddq_s32(
                    load_i32x4(chunk::<i32, 4>(a0, j + 1)),
                    load_i32x4(chunk::<i32, 4>(a2, j + 1)),
                ),
            );
            let b_sum = vaddq_s32(vmulq_n_s32(a_cross, 4), vmulq_n_s32(a_diag, 3));
            let out = vshrq_n_s32::<9>(vaddq_s32(vsubq_s32(b_sum, vmulq_s32(a_sum, s_v)), r256));
            store_i32x4(chunk_mut::<i32, 4>(tmp, k), out);
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

#[target_feature(enable = "neon")]
fn finish_filter2_neon<const BPC: u32>(
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
    let r256 = vdupq_n_s32(1 << 8);
    let r128 = vdupq_n_s32(1 << 7);
    let bpp = bpp_of(BPC);

    let mut i = 0usize;
    while i + 8 <= w {
        let s8 = load_px8_u16::<BPC>(src0.buf, src0.base + i * bpp);
        let s_pair = [
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s8))),
            vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s8))),
        ];
        for (half, s0_v) in s_pair.into_iter().enumerate() {
            let k = i + half * 4;
            let j = k + 1;
            let b_mid = vaddq_s32(
                load_i32x4(chunk::<i32, 4>(b0, j)),
                load_i32x4(chunk::<i32, 4>(b1, j)),
            );
            let b_side = vaddq_s32(
                vaddq_s32(
                    load_i32x4(chunk::<i32, 4>(b0, j - 1)),
                    load_i32x4(chunk::<i32, 4>(b1, j - 1)),
                ),
                vaddq_s32(
                    load_i32x4(chunk::<i32, 4>(b0, j + 1)),
                    load_i32x4(chunk::<i32, 4>(b1, j + 1)),
                ),
            );
            let a_sum = vaddq_s32(vmulq_n_s32(b_mid, 6), vmulq_n_s32(b_side, 5));

            let a_mid = vaddq_s32(
                load_i32x4(chunk::<i32, 4>(a0, j)),
                load_i32x4(chunk::<i32, 4>(a1, j)),
            );
            let a_side = vaddq_s32(
                vaddq_s32(
                    load_i32x4(chunk::<i32, 4>(a0, j - 1)),
                    load_i32x4(chunk::<i32, 4>(a1, j - 1)),
                ),
                vaddq_s32(
                    load_i32x4(chunk::<i32, 4>(a0, j + 1)),
                    load_i32x4(chunk::<i32, 4>(a1, j + 1)),
                ),
            );
            let b_sum = vaddq_s32(vmulq_n_s32(a_mid, 6), vmulq_n_s32(a_side, 5));
            let out0 = vshrq_n_s32::<9>(vaddq_s32(vsubq_s32(b_sum, vmulq_s32(a_sum, s0_v)), r256));
            store_i32x4(chunk_mut::<i32, 4>(tmp, k), out0);
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
        let s8 = load_px8_u16::<BPC>(src1.buf, src1.base + i * bpp);
        let s_pair = [
            vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(s8))),
            vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(s8))),
        ];
        for (half, s1_v) in s_pair.into_iter().enumerate() {
            let k = i + half * 4;
            let j = k + 1;
            let a_sum = vaddq_s32(
                vmulq_n_s32(load_i32x4(chunk::<i32, 4>(b1, j)), 6),
                vmulq_n_s32(
                    vaddq_s32(
                        load_i32x4(chunk::<i32, 4>(b1, j - 1)),
                        load_i32x4(chunk::<i32, 4>(b1, j + 1)),
                    ),
                    5,
                ),
            );
            let b_sum = vaddq_s32(
                vmulq_n_s32(load_i32x4(chunk::<i32, 4>(a1, j)), 6),
                vmulq_n_s32(
                    vaddq_s32(
                        load_i32x4(chunk::<i32, 4>(a1, j - 1)),
                        load_i32x4(chunk::<i32, 4>(a1, j + 1)),
                    ),
                    5,
                ),
            );
            let out1 = vshrq_n_s32::<8>(vaddq_s32(vsubq_s32(b_sum, vmulq_s32(a_sum, s1_v)), r128));
            store_i32x4(chunk_mut::<i32, 4>(tmp, FILTER_OUT_STRIDE + k), out1);
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

#[target_feature(enable = "neon")]
fn weighted_row1_neon<const BPC: u32>(dst: &mut [u8], off: usize, t1: &[i32], w: usize, w1: i32) {
    let bpp = bpp_of(BPC);
    let r1024 = vdupq_n_s32(1 << 10);
    let zero = vdupq_n_s32(0);
    let max_v = vdupq_n_s32((1 << BPC) - 1);

    let mut i = 0usize;
    while i + 8 <= w {
        let d_u16 = load_px8_u16::<BPC>(dst, off + i * bpp);
        let d_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(d_u16)));
        let d_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(d_u16)));
        let t_lo = vmulq_n_s32(load_i32x4(chunk::<i32, 4>(t1, i)), w1);
        let t_hi = vmulq_n_s32(load_i32x4(chunk::<i32, 4>(t1, i + 4)), w1);
        let o_lo = vminq_s32(
            vmaxq_s32(
                vaddq_s32(d_lo, vshrq_n_s32::<11>(vaddq_s32(t_lo, r1024))),
                zero,
            ),
            max_v,
        );
        let o_hi = vminq_s32(
            vmaxq_s32(
                vaddq_s32(d_hi, vshrq_n_s32::<11>(vaddq_s32(t_hi, r1024))),
                zero,
            ),
            max_v,
        );
        let out = vmovn_high_u32(
            vmovn_u32(vreinterpretq_u32_s32(o_lo)),
            vreinterpretq_u32_s32(o_hi),
        );
        store_px8_u16::<BPC>(dst, off + i * bpp, out);
        i += 8;
    }
    while i < w {
        let v = w1 * t1[i];
        let o = off + i * bpp;
        let d = px::<BPC>(dst, o);
        put::<BPC>(dst, o, iclip_pixel::<BPC>(d + ((v + (1 << 10)) >> 11)));
        i += 1;
    }
}

#[target_feature(enable = "neon")]
fn weighted2_neon<const BPC: u32>(
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
    let r1024 = vdupq_n_s32(1 << 10);
    let zero = vdupq_n_s32(0);
    let max_v = vdupq_n_s32((1 << BPC) - 1);

    for j in 0..h {
        let row = off + j * stride;
        let t = j * FILTER_OUT_STRIDE;
        let mut i = 0usize;
        while i + 8 <= w {
            let d_u16 = load_px8_u16::<BPC>(dst, row + i * bpp);
            let d_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(d_u16)));
            let d_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(d_u16)));
            let v_lo = vaddq_s32(
                vmulq_n_s32(load_i32x4(chunk::<i32, 4>(t1, t + i)), w0),
                vmulq_n_s32(load_i32x4(chunk::<i32, 4>(t2, t + i)), w1),
            );
            let v_hi = vaddq_s32(
                vmulq_n_s32(load_i32x4(chunk::<i32, 4>(t1, t + i + 4)), w0),
                vmulq_n_s32(load_i32x4(chunk::<i32, 4>(t2, t + i + 4)), w1),
            );
            let o_lo = vminq_s32(
                vmaxq_s32(
                    vaddq_s32(d_lo, vshrq_n_s32::<11>(vaddq_s32(v_lo, r1024))),
                    zero,
                ),
                max_v,
            );
            let o_hi = vminq_s32(
                vmaxq_s32(
                    vaddq_s32(d_hi, vshrq_n_s32::<11>(vaddq_s32(v_hi, r1024))),
                    zero,
                ),
                max_v,
            );
            let out = vmovn_high_u32(
                vmovn_u32(vreinterpretq_u32_s32(o_lo)),
                vreinterpretq_u32_s32(o_hi),
            );
            store_px8_u16::<BPC>(dst, row + i * bpp, out);
            i += 8;
        }
        while i < w {
            let v = w0 * t1[t + i] + w1 * t2[t + i];
            let o = row + i * bpp;
            let d = px::<BPC>(dst, o);
            put::<BPC>(dst, o, iclip_pixel::<BPC>(d + ((v + (1 << 10)) >> 11)));
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

#[target_feature(enable = "neon")]
fn finish1_neon<const BPC: u32>(
    dst: &mut [u8],
    c: &Ctx<'_>,
    dst_row: &mut usize,
    ab: &Rows<3>,
    ap: &mut [usize; 3],
    tmp: &mut [i32; 2 * FILTER_OUT_STRIDE],
    w1: i32,
) {
    finish_filter_row1_neon::<BPC, 3>(tmp, dst_src(dst, c, *dst_row), ab, *ap, c.w);
    weighted_row1_neon::<BPC>(dst, *dst_row * c.stride, &tmp[..], c.w, w1);
    *dst_row += 1;
    ap.rotate_left(1);
}

#[target_feature(enable = "neon")]
fn finish2_neon<const BPC: u32>(
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
    finish_filter2_neon::<BPC>(
        tmp,
        dst_src(dst, c, r),
        dst_src(dst, c, r1),
        ab,
        *ap,
        c.w,
        h,
    );
    weighted_row1_neon::<BPC>(dst, r * c.stride, &tmp[..], c.w, w0);
    *dst_row += 1;
    if h > 1 {
        weighted_row1_neon::<BPC>(dst, (r + 1) * c.stride, &tmp[FILTER_OUT_STRIDE..], c.w, w0);
        *dst_row += 1;
    }
    ap.rotate_left(1);
}

#[target_feature(enable = "neon")]
fn finish_mix_neon<const BPC: u32>(
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
    finish_filter2_neon::<BPC>(
        tmp5,
        dst_src(dst, c, r),
        dst_src(dst, c, r1),
        ab5,
        *ap5,
        c.w,
        h,
    );
    finish_filter_row1_neon::<BPC, 4>(
        &mut tmp3[..],
        dst_src(dst, c, r),
        ab3,
        [ap3[0], ap3[1], ap3[2]],
        c.w,
    );
    if h > 1 {
        finish_filter_row1_neon::<BPC, 4>(
            &mut tmp3[FILTER_OUT_STRIDE..],
            dst_src(dst, c, r + 1),
            ab3,
            [ap3[1], ap3[2], ap3[3]],
            c.w,
        );
    }
    weighted2_neon::<BPC>(dst, r * c.stride, c.stride, tmp5, tmp3, c.w, h, w0, w1);
    *dst_row += h;
    ap5.rotate_left(1);
    ap3.rotate_left(1);
}

#[target_feature(enable = "neon")]
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
                box3_row_h_neon::<BPC>(&mut sums.sq[0], &mut sums.s[0], l, s, w, c.edges);
                let (l, s) = c.row::<BPC>(dst, RowRef::Lpf(1));
                box3_row_h_neon::<BPC>(&mut sums.sq[1], &mut sums.s[1], l, s, w, c.edges);
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_neon::<BPC>(&mut sums.sq[sp[2]], &mut sums.s[sp[2]], l, s, w, c.edges);
            }
            box3_vert_neon::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_neon::<BPC>(&mut sums.sq[sp[2]], &mut sums.s[sp[2]], l, s, w, c.edges);
            }
            box3_vert_neon::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
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
                box3_row_h_neon::<BPC>(&mut sums.sq[0], &mut sums.s[0], l, s, w, c.edges);
            }
            sr += 1;
            box3_vert_neon::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp[2] = 1;
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_neon::<BPC>(&mut sums.sq[sp[2]], &mut sums.s[sp[2]], l, s, w, c.edges);
            }
            box3_vert_neon::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
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
                box3_row_h_neon::<BPC>(&mut sums.sq[sp[2]], &mut sums.s[sp[2]], l, s, w, c.edges);
            }
            box3_vert_neon::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            finish1_neon::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
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
            box3_row_h_neon::<BPC>(&mut sums.sq[sp[2]], &mut sums.s[sp[2]], l, s, w, c.edges);
        }
        box3_vert_neon::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
        finish1_neon::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        {
            let (l, s) = c.row::<BPC>(dst, RowRef::Lpf(7));
            box3_row_h_neon::<BPC>(&mut sums.sq[sp[2]], &mut sums.s[sp[2]], l, s, w, c.edges);
        }
        box3_vert_neon::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
        finish1_neon::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        Kuyruk::Bitti
    };

    match kuyruk {
        Kuyruk::Bitti => return,
        Kuyruk::Vert2 => {
            sp[2] = sp[1];
            box3_vert_neon::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            finish1_neon::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        }
        Kuyruk::Vert1 => {
            sp[2] = sp[1];
            box3_vert_neon::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            ap.rotate_left(1);
        }
    }
    sp[2] = sp[1];
    box3_vert_neon::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
    finish1_neon::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
}

#[target_feature(enable = "neon")]
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
                box5_row_h_neon::<BPC>(&mut sums.sq[0], &mut sums.s[0], l, s, w, c.edges);
                let (l, s) = c.row::<BPC>(dst, RowRef::Lpf(1));
                box5_row_h_neon::<BPC>(&mut sums.sq[1], &mut sums.s[1], l, s, w, c.edges);
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box5_row_h_neon::<BPC>(&mut sums.sq[2], &mut sums.s[2], l, s, w, c.edges);
            }
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box5_row_h_neon::<BPC>(&mut sums.sq[3], &mut sums.s[3], l, s, w, c.edges);
            }
            sr += 1;
            box5_vert_neon::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
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
                box5_row_h_neon::<BPC>(&mut sums.sq[0], &mut sums.s[0], l, s, w, c.edges);
            }
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp[4] = 1;
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box5_row_h_neon::<BPC>(&mut sums.sq[1], &mut sums.s[1], l, s, w, c.edges);
            }
            sr += 1;
            box5_vert_neon::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp[3] = 2;
            sp[4] = 3;
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box5_row_h_neon::<BPC>(&mut sums.sq[2], &mut sums.s[2], l, s, w, c.edges);
            }
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box5_row_h_neon::<BPC>(&mut sums.sq[3], &mut sums.s[3], l, s, w, c.edges);
            }
            sr += 1;
            box5_vert_neon::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2_neon::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp[3] = 4;
        }

        loop {
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box5_row_h_neon::<BPC>(&mut sums.sq[sp[3]], &mut sums.s[sp[3]], l, s, w, c.edges);
            }
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box5_row_h_neon::<BPC>(&mut sums.sq[sp[4]], &mut sums.s[sp[4]], l, s, w, c.edges);
            }
            sr += 1;
            box5_vert_neon::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2_neon::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
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
            box5_row_h_neon::<BPC>(&mut sums.sq[sp[3]], &mut sums.s[sp[3]], l, s, w, c.edges);
            let (l, s) = c.row::<BPC>(dst, RowRef::Lpf(7));
            box5_row_h_neon::<BPC>(&mut sums.sq[sp[4]], &mut sums.s[sp[4]], l, s, w, c.edges);
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
            box5_vert_neon::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2_neon::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
            true
        }
        Kuyruk::Vert1 => {
            sp[4] = sp[3];
            box5_vert_neon::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
            ap.rotate_left(1);
            true
        }
    };
    if output_1 {
        sp[3] = sp[2];
        sp[4] = sp[2];
        box5_vert_neon::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
        finish2_neon::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 1, w0);
    } else {
        box5_vert_neon::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
        finish2_neon::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
    }
}

#[target_feature(enable = "neon")]
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
                box3_row_h_neon::<BPC>(&mut sums3.sq[0], &mut sums3.s[0], l, s, w, c.edges);
                box5_row_h_neon::<BPC>(&mut sums5.sq[0], &mut sums5.s[0], l, s, w, c.edges);
                let (l, s) = c.row::<BPC>(dst, RowRef::Lpf(1));
                box3_row_h_neon::<BPC>(&mut sums3.sq[1], &mut sums3.s[1], l, s, w, c.edges);
                box5_row_h_neon::<BPC>(&mut sums5.sq[1], &mut sums5.s[1], l, s, w, c.edges);
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_neon::<BPC>(&mut sums3.sq[2], &mut sums3.s[2], l, s, w, c.edges);
                box5_row_h_neon::<BPC>(&mut sums5.sq[2], &mut sums5.s[2], l, s, w, c.edges);
            }
            sr += 1;
            box3_vert_neon::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_neon::<BPC>(
                    &mut sums3.sq[sp3[2]],
                    &mut sums3.s[sp3[2]],
                    l,
                    s,
                    w,
                    c.edges,
                );
                box5_row_h_neon::<BPC>(&mut sums5.sq[3], &mut sums5.s[3], l, s, w, c.edges);
            }
            sr += 1;
            box5_vert_neon::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            ap5.rotate_left(1);
            box3_vert_neon::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
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
                box3_row_h_neon::<BPC>(&mut sums3.sq[0], &mut sums3.s[0], l, s, w, c.edges);
                box5_row_h_neon::<BPC>(&mut sums5.sq[0], &mut sums5.s[0], l, s, w, c.edges);
            }
            sr += 1;
            box3_vert_neon::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp5[4] = 1;
            sp3[2] = 1;
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_neon::<BPC>(&mut sums3.sq[1], &mut sums3.s[1], l, s, w, c.edges);
                box5_row_h_neon::<BPC>(&mut sums5.sq[1], &mut sums5.s[1], l, s, w, c.edges);
            }
            sr += 1;
            box5_vert_neon::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            ap5.rotate_left(1);
            box3_vert_neon::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
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
                box3_row_h_neon::<BPC>(&mut sums3.sq[2], &mut sums3.s[2], l, s, w, c.edges);
                box5_row_h_neon::<BPC>(&mut sums5.sq[2], &mut sums5.s[2], l, s, w, c.edges);
            }
            sr += 1;
            box3_vert_neon::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_neon::<BPC>(
                    &mut sums3.sq[sp3[2]],
                    &mut sums3.s[sp3[2]],
                    l,
                    s,
                    w,
                    c.edges,
                );
                box5_row_h_neon::<BPC>(&mut sums5.sq[3], &mut sums5.s[3], l, s, w, c.edges);
            }
            sr += 1;
            box5_vert_neon::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert_neon::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix_neon::<BPC>(
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
                box3_row_h_neon::<BPC>(
                    &mut sums3.sq[sp3[2]],
                    &mut sums3.s[sp3[2]],
                    l,
                    s,
                    w,
                    c.edges,
                );
                box5_row_h_neon::<BPC>(
                    &mut sums5.sq[sp5[3]],
                    &mut sums5.s[sp5[3]],
                    l,
                    s,
                    w,
                    c.edges,
                );
            }
            sr += 1;
            box3_vert_neon::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            {
                let (l, s) = c.row::<BPC>(dst, RowRef::Main(sr));
                box3_row_h_neon::<BPC>(
                    &mut sums3.sq[sp3[2]],
                    &mut sums3.s[sp3[2]],
                    l,
                    s,
                    w,
                    c.edges,
                );
                box5_row_h_neon::<BPC>(
                    &mut sums5.sq[sp5[4]],
                    &mut sums5.s[sp5[4]],
                    l,
                    s,
                    w,
                    c.edges,
                );
            }
            sr += 1;
            box5_vert_neon::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert_neon::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix_neon::<BPC>(
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
            box3_row_h_neon::<BPC>(
                &mut sums3.sq[sp3[2]],
                &mut sums3.s[sp3[2]],
                l,
                s,
                w,
                c.edges,
            );
            box5_row_h_neon::<BPC>(
                &mut sums5.sq[sp5[3]],
                &mut sums5.s[sp5[3]],
                l,
                s,
                w,
                c.edges,
            );
        }
        box3_vert_neon::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
        ap3.rotate_left(1);
        {
            let (l, s) = c.row::<BPC>(dst, RowRef::Lpf(7));
            box3_row_h_neon::<BPC>(
                &mut sums3.sq[sp3[2]],
                &mut sums3.s[sp3[2]],
                l,
                s,
                w,
                c.edges,
            );
            box5_row_h_neon::<BPC>(
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
            box3_vert_neon::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            sp3[2] = sp3[1];
            false
        }
        Kuyruk::Odd => {
            sp5[4] = sp5[3];
            sp3[2] = sp3[1];
            box5_vert_neon::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert_neon::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix_neon::<BPC>(
                dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 2, w0, w1,
            );
            true
        }
        Kuyruk::Vert1 => {
            sp5[4] = sp5[3];
            sp3[2] = sp3[1];
            box5_vert_neon::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            ap5.rotate_left(1);
            box3_vert_neon::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            true
        }
    };
    if output_1 {
        sp5[3] = sp5[2];
        sp5[4] = sp5[2];
        sp3[2] = sp3[1];
        box5_vert_neon::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
        box3_vert_neon::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
        ap3.rotate_left(1);
        finish_mix_neon::<BPC>(
            dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 1, w0, w1,
        );
    } else {
        box5_vert_neon::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
        box3_vert_neon::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
        finish_mix_neon::<BPC>(
            dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 2, w0, w1,
        );
    }
}

#[target_feature(enable = "neon")]
pub(super) fn sgr_neon<const BPC: u32, const SGR_IDX: usize>(
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
) {
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
