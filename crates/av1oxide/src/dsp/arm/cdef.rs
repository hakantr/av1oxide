//! `aarch64` NEON 8-bpc CDEF çekirdekleri (`F5-CDEF` `4 / 8` kayıt = `4` yuva; `P51` / `F5a`, `P58-M4` vektörizasyonu).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Sembol | Matris Kaydı | Kaynak |
//! |---|---|---|
//! | `dav1d_cdef_find_dir_8bpc_neon` | `D00440` | `src/arm/64/cdef_tmpl.S:269-518`, `src/arm/cdef_init_tmpl.c:55` |
//! | `dav1d_cdef_filter8_8bpc_neon` | `D00442` (`fb[0]`, `8×8`) | `src/arm/64/cdef.S:35-320`, `src/arm/64/cdef_tmpl.S:30-267` |
//! | `dav1d_cdef_filter4_8bpc_neon` | `D00444` (`fb[1]`, `4×8`), `D00446` (`fb[2]`, `4×4`) | `src/arm/64/cdef.S:35-320`, `src/arm/64/cdef_tmpl.S:30-267` |
//!
//! ## Güvenlik ve ISA Sınırı (`GUV-02`, `S3ze`)
//!
//! - Bu modül `#![forbid(unsafe_code)]` taşır; tüm bellek yükleme/saklama
//!   işlemleri [`super::mem`] içindeki sabit boyutlu dizi yardımcılarıyla yapılır.
//! - Giriş işaretçileri yalnız [`NeonToken`] ile alınabilir ([`find_dir_neon_fn`],
//!   [`filter_8x8_neon_fn`], [`filter_4x8_neon_fn`], [`filter_4x4_neon_fn`]).
//! - Sıcak döngülerde sabit `256` (`8×8`, `16` adımlı) ve `128` (`4×H`, `8` adımlı,
//!   2 satır birleşik) yığın tamponları ile sabit boyutlu dilim pencereleri
//!   (`first_chunk`) kullanılarak `panic_bounds_check` üretimi sıfırlanmıştır.

#![forbid(unsafe_code)]

use core::arch::aarch64::{
    int16x8_t, uint16x8_t, uint32x4_t, vaddq_s16, vaddq_s32, vaddq_u32, vaddvq_s32, vaddvq_u32,
    vcombine_s16, vdup_n_s16, vdup_n_u8, vdupq_n_s16, vdupq_n_u16, vextq_s16, vget_high_s16,
    vget_lane_u64, vget_low_s16, vget_low_u16, vmlal_high_s16, vmlal_s16, vmovl_high_u16, vmovl_u8,
    vmovl_u16, vmovn_u16, vmull_high_s16, vmull_s16, vmulq_u32, vpaddq_s16, vreinterpret_u64_u8,
    vreinterpretq_s16_u16, vreinterpretq_u16_s16, vreinterpretq_u32_s32, vrev64_s16, vrev64q_s16,
    vsubl_u8,
};

use super::giris::store_i16x8;
use crate::cpu::NeonToken;
use crate::dsp::arm::mem;
use crate::dsp::cdef::{
    CdefBitDepth, CdefBlockSize, CdefDirFn, CdefDirResult, CdefEdgeFlags, CdefFilterFn,
    CdefFilterParams, DIR_CENTER_WEIGHT, clamped_shift, ulog2,
};

/// Eksik kenar dolgu sabiti (`i16::MIN` = `-32768`, `u16` görünümünde `0x8000 = 32768`;
/// `dav1d` `CDEF_VERY_LARGE`):
/// - İşaretsiz uzayda `32768 > 4095` olduğundan `vminq_u16` dolguyu yok sayar ve
///   `damping <= 13` için `|32768 - px| >> shift > threshold` olduğundan `clip == 0` üretir.
/// - İşaretli uzayda `-32768 < 0 <= px` olduğundan `vmaxq_s16` maskesiz olarak dolguyu yok sayar.
pub(crate) const PAD_FILL_16: i16 = i16::MIN;

/// `W = 8` için 16-adımlı dolgu tamponu (`16 × 16 = 256` `i16`) ve `(2, 2)` başlangıç ofseti.
pub(crate) const PAD_STRIDE: usize = 16;
pub(crate) const PAD_BASE: isize = (2 * PAD_STRIDE + 2) as isize;

/// `W = 4` için 8-adımlı dolgu tamponu (`16 × 8 = 128` `i16`) ve `(2, 2)` başlangıç ofseti.
pub(crate) const PAD_STRIDE_4: usize = 8;
pub(crate) const PAD_BASE_4: isize = (2 * PAD_STRIDE_4 + 2) as isize;

/// `PAD_STRIDE = 16` (`W = 8`) için `dav1d_cdef_directions[12][2]` (`y * 16 + x`) tablosu.
#[allow(clippy::identity_op, clippy::neg_multiply, clippy::erasing_op)]
pub(crate) const CDEF_DIRECTIONS_16: [[isize; 2]; 12] = [
    [1 * 16 + 0, 2 * 16 + 0],   // 6
    [1 * 16 + 0, 2 * 16 - 1],   // 7
    [-1 * 16 + 1, -2 * 16 + 2], // 0
    [0 * 16 + 1, -1 * 16 + 2],  // 1
    [0 * 16 + 1, 0 * 16 + 2],   // 2
    [0 * 16 + 1, 1 * 16 + 2],   // 3
    [1 * 16 + 1, 2 * 16 + 2],   // 4
    [1 * 16 + 0, 2 * 16 + 1],   // 5
    [1 * 16 + 0, 2 * 16 + 0],   // 6
    [1 * 16 + 0, 2 * 16 - 1],   // 7
    [-1 * 16 + 1, -2 * 16 + 2], // 0
    [0 * 16 + 1, -1 * 16 + 2],  // 1
];

/// `PAD_STRIDE_4 = 8` (`W = 4`) için `dav1d_cdef_directions[12][2]` (`y * 8 + x`) tablosu.
#[allow(clippy::identity_op, clippy::neg_multiply, clippy::erasing_op)]
pub(crate) const CDEF_DIRECTIONS_8: [[isize; 2]; 12] = [
    [1 * 8 + 0, 2 * 8 + 0],   // 6
    [1 * 8 + 0, 2 * 8 - 1],   // 7
    [-1 * 8 + 1, -2 * 8 + 2], // 0
    [0 * 8 + 1, -1 * 8 + 2],  // 1
    [0 * 8 + 1, 0 * 8 + 2],   // 2
    [0 * 8 + 1, 1 * 8 + 2],   // 3
    [1 * 8 + 1, 2 * 8 + 2],   // 4
    [1 * 8 + 0, 2 * 8 + 1],   // 5
    [1 * 8 + 0, 2 * 8 + 0],   // 6
    [1 * 8 + 0, 2 * 8 - 1],   // 7
    [-1 * 8 + 1, -2 * 8 + 2], // 0
    [0 * 8 + 1, -1 * 8 + 2],  // 1
];

const DIV8_U16: [u16; 8] = [840, 420, 280, 210, 168, 140, 120, 105];
const ALT_FACT0_U16: [u16; 8] = [420, 210, 140, 105, 105, 105, 105, 105];
const ALT_FACT1_U16: [u16; 4] = [140, 210, 420, 0];

/// 8-bpc `cdef/dir` ARM64 NEON yuva işaretçisi (`D00440`).
#[must_use]
pub const fn find_dir_neon_fn(_token: NeonToken) -> CdefDirFn {
    mem::find_dir_8bpc_entry
}

/// 8-bpc `cdef/fb[0]` (`8×8`) ARM64 NEON yuva işaretçisi (`D00442`).
#[must_use]
pub const fn filter_8x8_neon_fn(_token: NeonToken) -> CdefFilterFn {
    mem::filter_8bpc_entry::<8, 8>
}

/// 8-bpc `cdef/fb[1]` (`4×8`) ARM64 NEON yuva işaretçisi (`D00444`).
#[must_use]
pub const fn filter_4x8_neon_fn(_token: NeonToken) -> CdefFilterFn {
    mem::filter_8bpc_entry::<4, 8>
}

/// 8-bpc `cdef/fb[2]` (`4×4`) ARM64 NEON yuva işaretçisi (`D00446`).
#[must_use]
pub const fn filter_4x4_neon_fn(_token: NeonToken) -> CdefFilterFn {
    mem::filter_8bpc_entry::<4, 4>
}

#[target_feature(enable = "neon")]
pub(super) fn find_dir_8bpc_neon(
    img: &[u8],
    stride: usize,
    bit_depth: CdefBitDepth,
) -> CdefDirResult {
    debug_assert_eq!(bit_depth, CdefBitDepth::B8);
    if stride == 0 || img.len() < 7usize.saturating_mul(stride).saturating_add(8) {
        return crate::dsp::scalar::cdef::find_dir::<8>(img, stride, bit_depth);
    }
    cdef_find_dir_8bpc_neon(img, stride)
}

#[target_feature(enable = "neon")]
pub(super) fn filter_8bpc_dispatch_neon<const W: usize, const H: usize>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    debug_assert_eq!(params.bit_depth, CdefBitDepth::B8);
    if params.damping > 13 {
        crate::dsp::scalar::cdef::filter::<8>(dst, stride, left, top, bottom, params);
        return;
    }
    let w = params.size.w();
    let h = params.size.h();
    if (w, h) != (W, H) {
        match params.size {
            CdefBlockSize::W8H8 => {
                cdef_filter_8bpc_neon::<8, 8>(dst, stride, left, top, bottom, params)
            }
            CdefBlockSize::W4H8 => {
                cdef_filter_8bpc_neon::<4, 8>(dst, stride, left, top, bottom, params)
            }
            CdefBlockSize::W4H4 => {
                cdef_filter_8bpc_neon::<4, 4>(dst, stride, left, top, bottom, params)
            }
        }
        return;
    }
    cdef_filter_8bpc_neon::<W, H>(dst, stride, left, top, bottom, params);
}

#[inline]
#[target_feature(enable = "neon")]
fn rev8_s16(v: int16x8_t) -> int16x8_t {
    let r = vrev64q_s16(v);
    vextq_s16::<4>(r, r)
}

#[inline]
#[target_feature(enable = "neon")]
fn add_shifted_16<const EXT: i32>(
    lo: int16x8_t,
    hi: int16x8_t,
    v: int16x8_t,
    zero: int16x8_t,
) -> (int16x8_t, int16x8_t) {
    (
        vaddq_s16(lo, vextq_s16::<EXT>(zero, v)),
        vaddq_s16(hi, vextq_s16::<EXT>(v, zero)),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn cost_hv_neon(v: int16x8_t) -> u32 {
    let sq = vaddq_s32(
        vmull_s16(vget_low_s16(v), vget_low_s16(v)),
        vmull_high_s16(v, v),
    );
    (vaddvq_s32(sq) as u32).wrapping_mul(DIR_CENTER_WEIGHT)
}

#[inline]
#[target_feature(enable = "neon")]
fn cost_diag_neon(lo: int16x8_t, hi: int16x8_t, div_lo: uint32x4_t, div_hi: uint32x4_t) -> u32 {
    let r = vrev64q_s16(hi);
    let rev_hi = vextq_s16::<5>(r, r);
    let sq_lo = vmlal_s16(
        vmull_s16(vget_low_s16(lo), vget_low_s16(lo)),
        vget_low_s16(rev_hi),
        vget_low_s16(rev_hi),
    );
    let sq_hi = vmlal_high_s16(vmull_high_s16(lo, lo), rev_hi, rev_hi);
    let w_lo = vmulq_u32(vreinterpretq_u32_s32(sq_lo), div_lo);
    let w_hi = vmulq_u32(vreinterpretq_u32_s32(sq_hi), div_hi);
    vaddvq_u32(vaddq_u32(w_lo, w_hi))
}

#[inline]
#[target_feature(enable = "neon")]
fn cost_alt_neon(
    lo: int16x8_t,
    hi: int16x8_t,
    f0: uint32x4_t,
    f1: uint32x4_t,
    f2: uint32x4_t,
) -> u32 {
    let s0 = vmull_s16(vget_low_s16(lo), vget_low_s16(lo));
    let s1 = vmull_high_s16(lo, lo);
    let s2 = vmull_s16(vget_low_s16(hi), vget_low_s16(hi));
    let w0 = vmulq_u32(vreinterpretq_u32_s32(s0), f0);
    let w1 = vmulq_u32(vreinterpretq_u32_s32(s1), f1);
    let w2 = vmulq_u32(vreinterpretq_u32_s32(s2), f2);
    vaddvq_u32(vaddq_u32(vaddq_u32(w0, w1), w2))
}

/// 8×8 merkezlenmiş (`px ∈ [-128, 127]`) NEON satırlarından 8 yön maliyetini ve en iyi yönü hesaplar.
#[target_feature(enable = "neon")]
pub(crate) fn compute_dir_from_rows_neon(rows: &[int16x8_t; 8]) -> CdefDirResult {
    let zero = vdupq_n_s16(0);
    let zero4 = vdup_n_s16(0);
    let [r0, r1, r2, r3, r4, r5, r6, r7] = *rows;

    // Yatay toplam ağacı (`v_hv0`) + `(x >> 1)` çift toplamları (`p01..p67`).
    let p01 = vpaddq_s16(r0, r1);
    let p23 = vpaddq_s16(r2, r3);
    let p45 = vpaddq_s16(r4, r5);
    let p67 = vpaddq_s16(r6, r7);
    let v_hv0 = vpaddq_s16(vpaddq_s16(p01, p23), vpaddq_s16(p45, p67));

    // Çift satır toplamları (`s01..s67`) ve dikey toplam (`v_hv1`).
    let s01 = vaddq_s16(r0, r1);
    let s23 = vaddq_s16(r2, r3);
    let s45 = vaddq_s16(r4, r5);
    let s67 = vaddq_s16(r6, r7);
    let v_hv1 = vaddq_s16(vaddq_s16(s01, s23), vaddq_s16(s45, s67));

    // alt[2] (`3 - (y >> 1) + x`) ve alt[3] (`(y >> 1) + x`).
    let (mut a2_lo, mut a2_hi) = (s67, zero);
    (a2_lo, a2_hi) = add_shifted_16::<7>(a2_lo, a2_hi, s45, zero);
    (a2_lo, a2_hi) = add_shifted_16::<6>(a2_lo, a2_hi, s23, zero);
    (a2_lo, a2_hi) = add_shifted_16::<5>(a2_lo, a2_hi, s01, zero);

    let (mut a3_lo, mut a3_hi) = (s01, zero);
    (a3_lo, a3_hi) = add_shifted_16::<7>(a3_lo, a3_hi, s23, zero);
    (a3_lo, a3_hi) = add_shifted_16::<6>(a3_lo, a3_hi, s45, zero);
    (a3_lo, a3_hi) = add_shifted_16::<5>(a3_lo, a3_hi, s67, zero);

    // diag[0] (`y + x`) ve diag[1] (`7 + y - x`).
    let (mut d0_lo, mut d0_hi) = (r0, zero);
    (d0_lo, d0_hi) = add_shifted_16::<7>(d0_lo, d0_hi, r1, zero);
    (d0_lo, d0_hi) = add_shifted_16::<6>(d0_lo, d0_hi, r2, zero);
    (d0_lo, d0_hi) = add_shifted_16::<5>(d0_lo, d0_hi, r3, zero);
    (d0_lo, d0_hi) = add_shifted_16::<4>(d0_lo, d0_hi, r4, zero);
    (d0_lo, d0_hi) = add_shifted_16::<3>(d0_lo, d0_hi, r5, zero);
    (d0_lo, d0_hi) = add_shifted_16::<2>(d0_lo, d0_hi, r6, zero);
    (d0_lo, d0_hi) = add_shifted_16::<1>(d0_lo, d0_hi, r7, zero);

    let (mut d1_lo, mut d1_hi) = (rev8_s16(r0), zero);
    (d1_lo, d1_hi) = add_shifted_16::<7>(d1_lo, d1_hi, rev8_s16(r1), zero);
    (d1_lo, d1_hi) = add_shifted_16::<6>(d1_lo, d1_hi, rev8_s16(r2), zero);
    (d1_lo, d1_hi) = add_shifted_16::<5>(d1_lo, d1_hi, rev8_s16(r3), zero);
    (d1_lo, d1_hi) = add_shifted_16::<4>(d1_lo, d1_hi, rev8_s16(r4), zero);
    (d1_lo, d1_hi) = add_shifted_16::<3>(d1_lo, d1_hi, rev8_s16(r5), zero);
    (d1_lo, d1_hi) = add_shifted_16::<2>(d1_lo, d1_hi, rev8_s16(r6), zero);
    (d1_lo, d1_hi) = add_shifted_16::<1>(d1_lo, d1_hi, rev8_s16(r7), zero);

    // alt[0] (`y + (x >> 1)`) ve alt[1] (`3 + y - (x >> 1)`).
    let h0 = vget_low_s16(p01);
    let h1 = vget_high_s16(p01);
    let h2 = vget_low_s16(p23);
    let h3 = vget_high_s16(p23);
    let h4 = vget_low_s16(p45);
    let h5 = vget_high_s16(p45);
    let h6 = vget_low_s16(p67);
    let h7 = vget_high_s16(p67);

    let (mut a0_lo, mut a0_hi) = (vcombine_s16(h0, zero4), zero);
    (a0_lo, a0_hi) = add_shifted_16::<7>(a0_lo, a0_hi, vcombine_s16(h1, zero4), zero);
    (a0_lo, a0_hi) = add_shifted_16::<6>(a0_lo, a0_hi, vcombine_s16(h2, zero4), zero);
    (a0_lo, a0_hi) = add_shifted_16::<5>(a0_lo, a0_hi, vcombine_s16(h3, zero4), zero);
    (a0_lo, a0_hi) = add_shifted_16::<4>(a0_lo, a0_hi, vcombine_s16(h4, zero4), zero);
    (a0_lo, a0_hi) = add_shifted_16::<3>(a0_lo, a0_hi, vcombine_s16(h5, zero4), zero);
    (a0_lo, a0_hi) = add_shifted_16::<2>(a0_lo, a0_hi, vcombine_s16(h6, zero4), zero);
    (a0_lo, a0_hi) = add_shifted_16::<1>(a0_lo, a0_hi, vcombine_s16(h7, zero4), zero);

    let (mut a1_lo, mut a1_hi) = (vcombine_s16(vrev64_s16(h0), zero4), zero);
    (a1_lo, a1_hi) = add_shifted_16::<7>(a1_lo, a1_hi, vcombine_s16(vrev64_s16(h1), zero4), zero);
    (a1_lo, a1_hi) = add_shifted_16::<6>(a1_lo, a1_hi, vcombine_s16(vrev64_s16(h2), zero4), zero);
    (a1_lo, a1_hi) = add_shifted_16::<5>(a1_lo, a1_hi, vcombine_s16(vrev64_s16(h3), zero4), zero);
    (a1_lo, a1_hi) = add_shifted_16::<4>(a1_lo, a1_hi, vcombine_s16(vrev64_s16(h4), zero4), zero);
    (a1_lo, a1_hi) = add_shifted_16::<3>(a1_lo, a1_hi, vcombine_s16(vrev64_s16(h5), zero4), zero);
    (a1_lo, a1_hi) = add_shifted_16::<2>(a1_lo, a1_hi, vcombine_s16(vrev64_s16(h6), zero4), zero);
    (a1_lo, a1_hi) = add_shifted_16::<1>(a1_lo, a1_hi, vcombine_s16(vrev64_s16(h7), zero4), zero);

    let div8 = mem::load_u16x8(&DIV8_U16);
    let div_lo = vmovl_u16(vget_low_u16(div8));
    let div_hi = vmovl_high_u16(div8);

    let af0_8 = mem::load_u16x8(&ALT_FACT0_U16);
    let af0 = vmovl_u16(vget_low_u16(af0_8));
    let af1 = vmovl_high_u16(af0_8);
    let af2 = vmovl_u16(mem::load_u16x4(&ALT_FACT1_U16));

    let c0 = cost_diag_neon(d0_lo, d0_hi, div_lo, div_hi);
    let c1 = cost_alt_neon(a0_lo, a0_hi, af0, af1, af2);
    let c2 = cost_hv_neon(v_hv0);
    let c3 = cost_alt_neon(a1_lo, a1_hi, af0, af1, af2);
    let c4 = cost_diag_neon(d1_lo, d1_hi, div_lo, div_hi);
    let c5 = cost_alt_neon(a2_lo, a2_hi, af0, af1, af2);
    let c6 = cost_hv_neon(v_hv1);
    let c7 = cost_alt_neon(a3_lo, a3_hi, af0, af1, af2);
    let cost = [c0, c1, c2, c3, c4, c5, c6, c7];

    let mut best_dir = 0usize;
    let mut best_cost = c0;
    for (n, &c) in cost.iter().enumerate().skip(1) {
        if c > best_cost {
            best_cost = c;
            best_dir = n;
        }
    }

    CdefDirResult {
        dir: best_dir as u8,
        var: best_cost.wrapping_sub(cost[(best_dir ^ 4) & 7]) >> 10,
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_dir_row_8bpc(img: &[u8], off: usize, v_128: core::arch::aarch64::uint8x8_t) -> int16x8_t {
    if let Some(src8) = img.get(off..).and_then(|s| s.first_chunk::<8>()) {
        vreinterpretq_s16_u16(vsubl_u8(mem::load_u8x8(src8), v_128))
    } else {
        vdupq_n_s16(0)
    }
}

#[target_feature(enable = "neon")]
fn cdef_find_dir_8bpc_neon(img: &[u8], stride: usize) -> CdefDirResult {
    let v_128 = vdup_n_u8(128);
    let rows = [
        load_dir_row_8bpc(img, 0, v_128),
        load_dir_row_8bpc(img, stride, v_128),
        load_dir_row_8bpc(img, 2 * stride, v_128),
        load_dir_row_8bpc(img, 3 * stride, v_128),
        load_dir_row_8bpc(img, 4 * stride, v_128),
        load_dir_row_8bpc(img, 5 * stride, v_128),
        load_dir_row_8bpc(img, 6 * stride, v_128),
        load_dir_row_8bpc(img, 7 * stride, v_128),
    ];
    compute_dir_from_rows_neon(&rows)
}

macro_rules! cdef_constrain_i16x8 {
    ($p:expr, $px:expr, $v_thresh:expr, $neg_shift:expr) => {{
        let p_v = $p;
        let px_v = $px;
        let adiff = core::arch::aarch64::vabdq_u16(
            core::arch::aarch64::vreinterpretq_u16_s16(p_v),
            core::arch::aarch64::vreinterpretq_u16_s16(px_v),
        );
        let shifted = core::arch::aarch64::vshlq_u16(adiff, $neg_shift);
        let clip = core::arch::aarch64::vreinterpretq_s16_u16(core::arch::aarch64::vqsubq_u16(
            $v_thresh, shifted,
        ));
        let diff = core::arch::aarch64::vqsubq_s16(p_v, px_v);
        core::arch::aarch64::vmaxq_s16(
            core::arch::aarch64::vminq_s16(diff, clip),
            core::arch::aarch64::vnegq_s16(clip),
        )
    }};
}

macro_rules! cdef_load_tmp_w8 {
    ($tmp:expr, $idx_expr:expr) => {{
        let idx = (($idx_expr) as usize) & 255;
        if let Some(c8) = $tmp.get(idx..).and_then(|s| s.first_chunk::<8>()) {
            $crate::dsp::arm::mem::load_i16x8(c8)
        } else {
            core::arch::aarch64::vdupq_n_s16($crate::dsp::arm::cdef::PAD_FILL_16)
        }
    }};
}

macro_rules! cdef_load_tmp_w4_2rows {
    ($tmp:expr, $idx_expr:expr) => {{
        let idx0 = (($idx_expr) as usize) & 127;
        let idx1 = (idx0 + $crate::dsp::arm::cdef::PAD_STRIDE_4) & 127;
        let lo = if let Some(c4) = $tmp.get(idx0..).and_then(|s| s.first_chunk::<4>()) {
            $crate::dsp::arm::mem::load_i16x4(c4)
        } else {
            core::arch::aarch64::vdup_n_s16($crate::dsp::arm::cdef::PAD_FILL_16)
        };
        let hi = if let Some(c4) = $tmp.get(idx1..).and_then(|s| s.first_chunk::<4>()) {
            $crate::dsp::arm::mem::load_i16x4(c4)
        } else {
            core::arch::aarch64::vdup_n_s16($crate::dsp::arm::cdef::PAD_FILL_16)
        };
        core::arch::aarch64::vcombine_s16(lo, hi)
    }};
}

macro_rules! cdef_filter_core_8lanes {
    (
        $load_mac:ident,
        $pri:expr,
        $sec:expr,
        $tmp:expr,
        $base:expr,
        $d_pri:expr,
        $d_sec0:expr,
        $d_sec1:expr,
        $v_pri_thresh:expr,
        $v_sec_thresh:expr,
        $pri_neg_shift:expr,
        $sec_neg_shift:expr,
        $pri_tap0:expr,
        $pri_tap1:expr
    ) => {{
        use core::arch::aarch64::{
            vaddq_s16, vcltzq_s16, vdupq_n_s16, vmaxq_s16, vminq_s16, vminq_u16, vmulq_n_s16,
            vreinterpretq_s16_u16, vreinterpretq_u16_s16, vrshrq_n_s16,
        };
        let px = $load_mac!($tmp, $base);
        let mut sum = vdupq_n_s16(0);
        let mut min_u = vreinterpretq_u16_s16(px);
        let mut max_s = px;

        if $pri {
            let p0_0 = $load_mac!($tmp, $base + $d_pri[0]);
            let p1_0 = $load_mac!($tmp, $base - $d_pri[0]);
            let c0_0 = cdef_constrain_i16x8!(p0_0, px, $v_pri_thresh, $pri_neg_shift);
            let c1_0 = cdef_constrain_i16x8!(p1_0, px, $v_pri_thresh, $pri_neg_shift);
            sum = vaddq_s16(sum, vmulq_n_s16(vaddq_s16(c0_0, c1_0), $pri_tap0));

            let p0_1 = $load_mac!($tmp, $base + $d_pri[1]);
            let p1_1 = $load_mac!($tmp, $base - $d_pri[1]);
            let c0_1 = cdef_constrain_i16x8!(p0_1, px, $v_pri_thresh, $pri_neg_shift);
            let c1_1 = cdef_constrain_i16x8!(p1_1, px, $v_pri_thresh, $pri_neg_shift);
            sum = vaddq_s16(sum, vmulq_n_s16(vaddq_s16(c0_1, c1_1), $pri_tap1));

            if $sec {
                min_u = vminq_u16(
                    min_u,
                    vminq_u16(
                        vminq_u16(vreinterpretq_u16_s16(p0_0), vreinterpretq_u16_s16(p1_0)),
                        vminq_u16(vreinterpretq_u16_s16(p0_1), vreinterpretq_u16_s16(p1_1)),
                    ),
                );
                max_s = vmaxq_s16(
                    max_s,
                    vmaxq_s16(vmaxq_s16(p0_0, p1_0), vmaxq_s16(p0_1, p1_1)),
                );
            }
        }

        if $sec {
            for k in 0..2usize {
                let off2 = $d_sec0[k & 1];
                let off3 = $d_sec1[k & 1];
                let s0 = $load_mac!($tmp, $base + off2);
                let s1 = $load_mac!($tmp, $base - off2);
                let s2 = $load_mac!($tmp, $base + off3);
                let s3 = $load_mac!($tmp, $base - off3);
                let cs0 = cdef_constrain_i16x8!(s0, px, $v_sec_thresh, $sec_neg_shift);
                let cs1 = cdef_constrain_i16x8!(s1, px, $v_sec_thresh, $sec_neg_shift);
                let cs2 = cdef_constrain_i16x8!(s2, px, $v_sec_thresh, $sec_neg_shift);
                let cs3 = cdef_constrain_i16x8!(s3, px, $v_sec_thresh, $sec_neg_shift);
                let cs_tot = vaddq_s16(vaddq_s16(cs0, cs1), vaddq_s16(cs2, cs3));
                if k == 0 {
                    sum = vaddq_s16(sum, vaddq_s16(cs_tot, cs_tot));
                } else {
                    sum = vaddq_s16(sum, cs_tot);
                }
                if $pri {
                    let u01 = vminq_u16(vreinterpretq_u16_s16(s0), vreinterpretq_u16_s16(s1));
                    let u23 = vminq_u16(vreinterpretq_u16_s16(s2), vreinterpretq_u16_s16(s3));
                    min_u = vminq_u16(min_u, vminq_u16(u01, u23));
                    max_s = vmaxq_s16(max_s, vmaxq_s16(vmaxq_s16(s0, s1), vmaxq_s16(s2, s3)));
                }
            }
        }

        let neg = vreinterpretq_s16_u16(vcltzq_s16(sum));
        let mut out = vaddq_s16(px, vrshrq_n_s16::<4>(vaddq_s16(sum, neg)));
        if $pri && $sec {
            out = vminq_s16(vmaxq_s16(out, vreinterpretq_s16_u16(min_u)), max_s);
        }
        out
    }};
}

pub(crate) use {
    cdef_constrain_i16x8, cdef_filter_core_8lanes, cdef_load_tmp_w4_2rows, cdef_load_tmp_w8,
};

#[inline]
#[target_feature(enable = "neon")]
fn copy_row8_8bpc(tmp: &mut [i16; 256], t_idx: usize, src: &[u8], off: usize) {
    if let Some(s8) = src.get(off..).and_then(|s| s.first_chunk::<8>())
        && let Some(d8) = tmp.get_mut(t_idx..).and_then(|d| d.first_chunk_mut::<8>())
    {
        store_i16x8(d8, vreinterpretq_s16_u16(vmovl_u8(mem::load_u8x8(s8))));
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn copy_row4_8bpc(tmp: &mut [i16; 128], t_idx: usize, src: &[u8], off: usize) {
    if let Some(s4) = src.get(off..).and_then(|s| s.first_chunk::<4>())
        && let Some(d4) = tmp.get_mut(t_idx..).and_then(|d| d.first_chunk_mut::<4>())
    {
        d4[0] = i16::from(s4[0]);
        d4[1] = i16::from(s4[1]);
        d4[2] = i16::from(s4[2]);
        d4[3] = i16::from(s4[3]);
    }
}

#[target_feature(enable = "neon")]
fn pad_w8_8bpc_into_tmp<const H: usize>(
    tmp: &mut [i16; 256],
    dst: &[u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    edges: CdefEdgeFlags,
) {
    if edges == CdefEdgeFlags::ALL {
        for row in 0..2usize {
            let t_base = (PAD_BASE + (-2isize + row as isize) * PAD_STRIDE as isize) as usize;
            let row_base = row * stride;
            if let Some(s12) = top.get(row_base..).and_then(|s| s.first_chunk::<12>())
                && let Some((s8, s_rest)) = s12.split_first_chunk::<8>()
                && let Some(s4) = s_rest.first_chunk::<4>()
                && let Some(d12) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<12>())
                && let Some((d8, d_rest)) = d12.split_first_chunk_mut::<8>()
                && let Some(d4) = d_rest.first_chunk_mut::<4>()
            {
                store_i16x8(d8, vreinterpretq_s16_u16(vmovl_u8(mem::load_u8x8(s8))));
                d4[0] = i16::from(s4[0]);
                d4[1] = i16::from(s4[1]);
                d4[2] = i16::from(s4[2]);
                d4[3] = i16::from(s4[3]);
            }
        }
        for y in 0..H {
            let t_base = PAD_BASE as usize + y * PAD_STRIDE;
            let l_base = y * 2;
            if let Some(s2) = left.get(l_base..).and_then(|s| s.first_chunk::<2>())
                && let Some(d2) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                d2[0] = i16::from(s2[0]);
                d2[1] = i16::from(s2[1]);
            }
            let d_base = y * stride;
            if let Some(s10) = dst.get(d_base..).and_then(|s| s.first_chunk::<10>())
                && let Some((s8, s_rest)) = s10.split_first_chunk::<8>()
                && let Some(s2) = s_rest.first_chunk::<2>()
                && let Some(d10) = tmp
                    .get_mut(t_base..)
                    .and_then(|d| d.first_chunk_mut::<10>())
                && let Some((d8, d_rest)) = d10.split_first_chunk_mut::<8>()
                && let Some(d2) = d_rest.first_chunk_mut::<2>()
            {
                store_i16x8(d8, vreinterpretq_s16_u16(vmovl_u8(mem::load_u8x8(s8))));
                d2[0] = i16::from(s2[0]);
                d2[1] = i16::from(s2[1]);
            }
        }
        for row in 0..2usize {
            let t_base = PAD_BASE as usize + (H + row) * PAD_STRIDE;
            let row_base = row * stride;
            if let Some(s12) = bottom.get(row_base..).and_then(|s| s.first_chunk::<12>())
                && let Some((s8, s_rest)) = s12.split_first_chunk::<8>()
                && let Some(s4) = s_rest.first_chunk::<4>()
                && let Some(d12) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<12>())
                && let Some((d8, d_rest)) = d12.split_first_chunk_mut::<8>()
                && let Some(d4) = d_rest.first_chunk_mut::<4>()
            {
                store_i16x8(d8, vreinterpretq_s16_u16(vmovl_u8(mem::load_u8x8(s8))));
                d4[0] = i16::from(s4[0]);
                d4[1] = i16::from(s4[1]);
                d4[2] = i16::from(s4[2]);
                d4[3] = i16::from(s4[3]);
            }
        }
        return;
    }

    let has_left = edges.contains(CdefEdgeFlags::HAVE_LEFT);
    let has_right = edges.contains(CdefEdgeFlags::HAVE_RIGHT);
    let has_top = edges.contains(CdefEdgeFlags::HAVE_TOP);
    let has_bottom = edges.contains(CdefEdgeFlags::HAVE_BOTTOM);

    if has_top {
        for row in 0..2usize {
            let t_base = (PAD_BASE + (-2isize + row as isize) * PAD_STRIDE as isize) as usize;
            let row_base = row * stride;
            if has_left
                && let Some(s2) = top.get(row_base..).and_then(|s| s.first_chunk::<2>())
                && let Some(d2) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                d2[0] = i16::from(s2[0]);
                d2[1] = i16::from(s2[1]);
            }
            copy_row8_8bpc(tmp, t_base, top, row_base + 2);
            if has_right
                && let Some(s2) = top.get(row_base + 10..).and_then(|s| s.first_chunk::<2>())
                && let Some(d2) = tmp
                    .get_mut(t_base + 8..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                d2[0] = i16::from(s2[0]);
                d2[1] = i16::from(s2[1]);
            }
        }
    }
    if has_left {
        for y in 0..H {
            let t_base = PAD_BASE as usize + y * PAD_STRIDE;
            let l_base = y * 2;
            if let Some(s2) = left.get(l_base..).and_then(|s| s.first_chunk::<2>())
                && let Some(d2) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                d2[0] = i16::from(s2[0]);
                d2[1] = i16::from(s2[1]);
            }
        }
    }
    for y in 0..H {
        let t_base = PAD_BASE as usize + y * PAD_STRIDE;
        let d_base = y * stride;
        copy_row8_8bpc(tmp, t_base, dst, d_base);
        if has_right
            && let Some(s2) = dst.get(d_base + 8..).and_then(|s| s.first_chunk::<2>())
            && let Some(d2) = tmp
                .get_mut(t_base + 8..)
                .and_then(|d| d.first_chunk_mut::<2>())
        {
            d2[0] = i16::from(s2[0]);
            d2[1] = i16::from(s2[1]);
        }
    }
    if has_bottom {
        for row in 0..2usize {
            let t_base = PAD_BASE as usize + (H + row) * PAD_STRIDE;
            let row_base = row * stride;
            if has_left
                && let Some(s2) = bottom.get(row_base..).and_then(|s| s.first_chunk::<2>())
                && let Some(d2) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                d2[0] = i16::from(s2[0]);
                d2[1] = i16::from(s2[1]);
            }
            copy_row8_8bpc(tmp, t_base, bottom, row_base + 2);
            if has_right
                && let Some(s2) = bottom
                    .get(row_base + 10..)
                    .and_then(|s| s.first_chunk::<2>())
                && let Some(d2) = tmp
                    .get_mut(t_base + 8..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                d2[0] = i16::from(s2[0]);
                d2[1] = i16::from(s2[1]);
            }
        }
    }
}

#[target_feature(enable = "neon")]
fn pad_w4_8bpc_into_tmp<const H: usize>(
    tmp: &mut [i16; 128],
    dst: &[u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    edges: CdefEdgeFlags,
) {
    if edges == CdefEdgeFlags::ALL {
        for row in 0..2usize {
            let t_base = (PAD_BASE_4 + (-2isize + row as isize) * PAD_STRIDE_4 as isize) as usize;
            let row_base = row * stride;
            if let Some(s8) = top.get(row_base..).and_then(|s| s.first_chunk::<8>())
                && let Some(d8) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<8>())
            {
                store_i16x8(d8, vreinterpretq_s16_u16(vmovl_u8(mem::load_u8x8(s8))));
            }
        }
        for y in 0..H {
            let t_base = PAD_BASE_4 as usize + y * PAD_STRIDE_4;
            let l_base = y * 2;
            if let Some(s2) = left.get(l_base..).and_then(|s| s.first_chunk::<2>())
                && let Some(d2) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                d2[0] = i16::from(s2[0]);
                d2[1] = i16::from(s2[1]);
            }
            let d_base = y * stride;
            if let Some(s6) = dst.get(d_base..).and_then(|s| s.first_chunk::<6>())
                && let Some(d6) = tmp.get_mut(t_base..).and_then(|d| d.first_chunk_mut::<6>())
            {
                d6[0] = i16::from(s6[0]);
                d6[1] = i16::from(s6[1]);
                d6[2] = i16::from(s6[2]);
                d6[3] = i16::from(s6[3]);
                d6[4] = i16::from(s6[4]);
                d6[5] = i16::from(s6[5]);
            }
        }
        for row in 0..2usize {
            let t_base = PAD_BASE_4 as usize + (H + row) * PAD_STRIDE_4;
            let row_base = row * stride;
            if let Some(s8) = bottom.get(row_base..).and_then(|s| s.first_chunk::<8>())
                && let Some(d8) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<8>())
            {
                store_i16x8(d8, vreinterpretq_s16_u16(vmovl_u8(mem::load_u8x8(s8))));
            }
        }
        return;
    }

    let has_left = edges.contains(CdefEdgeFlags::HAVE_LEFT);
    let has_right = edges.contains(CdefEdgeFlags::HAVE_RIGHT);
    let has_top = edges.contains(CdefEdgeFlags::HAVE_TOP);
    let has_bottom = edges.contains(CdefEdgeFlags::HAVE_BOTTOM);

    if has_top {
        for row in 0..2usize {
            let t_base = (PAD_BASE_4 + (-2isize + row as isize) * PAD_STRIDE_4 as isize) as usize;
            let row_base = row * stride;
            if has_left
                && let Some(s2) = top.get(row_base..).and_then(|s| s.first_chunk::<2>())
                && let Some(d2) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                d2[0] = i16::from(s2[0]);
                d2[1] = i16::from(s2[1]);
            }
            copy_row4_8bpc(tmp, t_base, top, row_base + 2);
            if has_right
                && let Some(s2) = top.get(row_base + 6..).and_then(|s| s.first_chunk::<2>())
                && let Some(d2) = tmp
                    .get_mut(t_base + 4..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                d2[0] = i16::from(s2[0]);
                d2[1] = i16::from(s2[1]);
            }
        }
    }
    if has_left {
        for y in 0..H {
            let t_base = PAD_BASE_4 as usize + y * PAD_STRIDE_4;
            let l_base = y * 2;
            if let Some(s2) = left.get(l_base..).and_then(|s| s.first_chunk::<2>())
                && let Some(d2) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                d2[0] = i16::from(s2[0]);
                d2[1] = i16::from(s2[1]);
            }
        }
    }
    for y in 0..H {
        let t_base = PAD_BASE_4 as usize + y * PAD_STRIDE_4;
        let d_base = y * stride;
        copy_row4_8bpc(tmp, t_base, dst, d_base);
        if has_right
            && let Some(s2) = dst.get(d_base + 4..).and_then(|s| s.first_chunk::<2>())
            && let Some(d2) = tmp
                .get_mut(t_base + 4..)
                .and_then(|d| d.first_chunk_mut::<2>())
        {
            d2[0] = i16::from(s2[0]);
            d2[1] = i16::from(s2[1]);
        }
    }
    if has_bottom {
        for row in 0..2usize {
            let t_base = PAD_BASE_4 as usize + (H + row) * PAD_STRIDE_4;
            let row_base = row * stride;
            if has_left
                && let Some(s2) = bottom.get(row_base..).and_then(|s| s.first_chunk::<2>())
                && let Some(d2) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                d2[0] = i16::from(s2[0]);
                d2[1] = i16::from(s2[1]);
            }
            copy_row4_8bpc(tmp, t_base, bottom, row_base + 2);
            if has_right
                && let Some(s2) = bottom
                    .get(row_base + 6..)
                    .and_then(|s| s.first_chunk::<2>())
                && let Some(d2) = tmp
                    .get_mut(t_base + 4..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                d2[0] = i16::from(s2[0]);
                d2[1] = i16::from(s2[1]);
            }
        }
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn run_w8_8bpc_loop<const H: usize, const PRI: bool, const SEC: bool>(
    dst: &mut [u8],
    stride: usize,
    tmp: &[i16; 256],
    dir: usize,
    v_pri_thresh: uint16x8_t,
    v_sec_thresh: uint16x8_t,
    pri_neg_shift: int16x8_t,
    sec_neg_shift: int16x8_t,
    pri_tap0: i16,
    pri_tap1: i16,
) {
    let d_pri = &CDEF_DIRECTIONS_16[(dir & 7) + 2];
    let d_sec0 = &CDEF_DIRECTIONS_16[(dir & 7) + 4];
    let d_sec1 = &CDEF_DIRECTIONS_16[dir & 7];
    for y in 0..H {
        let base_y = PAD_BASE + (y as isize) * (PAD_STRIDE as isize);
        let out8 = cdef_filter_core_8lanes!(
            cdef_load_tmp_w8,
            PRI,
            SEC,
            tmp,
            base_y,
            d_pri,
            d_sec0,
            d_sec1,
            v_pri_thresh,
            v_sec_thresh,
            pri_neg_shift,
            sec_neg_shift,
            pri_tap0,
            pri_tap1
        );
        let u8_vec = vmovn_u16(vreinterpretq_u16_s16(out8));
        if let Some(dst8) = dst
            .get_mut(y * stride..)
            .and_then(|s| s.first_chunk_mut::<8>())
        {
            mem::store_u8x8(dst8, u8_vec);
        }
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn run_w4_8bpc_loop<const H: usize, const PRI: bool, const SEC: bool>(
    dst: &mut [u8],
    stride: usize,
    tmp: &[i16; 128],
    dir: usize,
    v_pri_thresh: uint16x8_t,
    v_sec_thresh: uint16x8_t,
    pri_neg_shift: int16x8_t,
    sec_neg_shift: int16x8_t,
    pri_tap0: i16,
    pri_tap1: i16,
) {
    let d_pri = &CDEF_DIRECTIONS_8[(dir & 7) + 2];
    let d_sec0 = &CDEF_DIRECTIONS_8[(dir & 7) + 4];
    let d_sec1 = &CDEF_DIRECTIONS_8[dir & 7];
    let mut y = 0usize;
    while y < H {
        let base_y = PAD_BASE_4 + (y as isize) * (PAD_STRIDE_4 as isize);
        let out8 = cdef_filter_core_8lanes!(
            cdef_load_tmp_w4_2rows,
            PRI,
            SEC,
            tmp,
            base_y,
            d_pri,
            d_sec0,
            d_sec1,
            v_pri_thresh,
            v_sec_thresh,
            pri_neg_shift,
            sec_neg_shift,
            pri_tap0,
            pri_tap1
        );
        let u8_vec = vmovn_u16(vreinterpretq_u16_s16(out8));
        let packed64 = vget_lane_u64::<0>(vreinterpret_u64_u8(u8_vec));
        if let Some(dst4) = dst
            .get_mut(y * stride..)
            .and_then(|s| s.first_chunk_mut::<4>())
        {
            *dst4 = (packed64 as u32).to_le_bytes();
        }
        if let Some(dst4) = dst
            .get_mut((y + 1) * stride..)
            .and_then(|s| s.first_chunk_mut::<4>())
        {
            *dst4 = ((packed64 >> 32) as u32).to_le_bytes();
        }
        y += 2;
    }
}

#[target_feature(enable = "neon")]
fn cdef_filter_8bpc_neon<const W: usize, const H: usize>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    let pri_strength = u16::from(params.pri_strength);
    let sec_strength = u16::from(params.sec_strength);
    if pri_strength == 0 && sec_strength == 0 {
        return;
    }

    let dir = usize::from(params.dir & 7);
    let v_pri_thresh = vdupq_n_u16(pri_strength);
    let v_sec_thresh = vdupq_n_u16(sec_strength);
    let pri_tap0 = (4 - ((i32::from(params.pri_strength)) & 1)) as i16;
    let pri_tap1 = (pri_tap0 & 3) | 2;

    let pri_shift = if pri_strength != 0 {
        clamped_shift(i32::from(params.damping) - ulog2(u32::from(params.pri_strength)) as i32)
    } else {
        0
    };
    let sec_shift = if sec_strength != 0 {
        clamped_shift(i32::from(params.damping) - ulog2(u32::from(params.sec_strength)) as i32)
    } else {
        0
    };
    let pri_neg_shift = vdupq_n_s16(-(pri_shift as i16));
    let sec_neg_shift = vdupq_n_s16(-(sec_shift as i16));

    if W == 8 {
        let mut tmp = if params.edges == CdefEdgeFlags::ALL {
            [0i16; 256]
        } else {
            [PAD_FILL_16; 256]
        };
        pad_w8_8bpc_into_tmp::<H>(&mut tmp, dst, stride, left, top, bottom, params.edges);
        match (pri_strength != 0, sec_strength != 0) {
            (true, true) => run_w8_8bpc_loop::<H, true, true>(
                dst,
                stride,
                &tmp,
                dir,
                v_pri_thresh,
                v_sec_thresh,
                pri_neg_shift,
                sec_neg_shift,
                pri_tap0,
                pri_tap1,
            ),
            (true, false) => run_w8_8bpc_loop::<H, true, false>(
                dst,
                stride,
                &tmp,
                dir,
                v_pri_thresh,
                v_sec_thresh,
                pri_neg_shift,
                sec_neg_shift,
                pri_tap0,
                pri_tap1,
            ),
            _ => run_w8_8bpc_loop::<H, false, true>(
                dst,
                stride,
                &tmp,
                dir,
                v_pri_thresh,
                v_sec_thresh,
                pri_neg_shift,
                sec_neg_shift,
                pri_tap0,
                pri_tap1,
            ),
        }
    } else {
        let mut tmp = if params.edges == CdefEdgeFlags::ALL {
            [0i16; 128]
        } else {
            [PAD_FILL_16; 128]
        };
        pad_w4_8bpc_into_tmp::<H>(&mut tmp, dst, stride, left, top, bottom, params.edges);
        match (pri_strength != 0, sec_strength != 0) {
            (true, true) => run_w4_8bpc_loop::<H, true, true>(
                dst,
                stride,
                &tmp,
                dir,
                v_pri_thresh,
                v_sec_thresh,
                pri_neg_shift,
                sec_neg_shift,
                pri_tap0,
                pri_tap1,
            ),
            (true, false) => run_w4_8bpc_loop::<H, true, false>(
                dst,
                stride,
                &tmp,
                dir,
                v_pri_thresh,
                v_sec_thresh,
                pri_neg_shift,
                sec_neg_shift,
                pri_tap0,
                pri_tap1,
            ),
            _ => run_w4_8bpc_loop::<H, false, true>(
                dst,
                stride,
                &tmp,
                dir,
                v_pri_thresh,
                v_sec_thresh,
                pri_neg_shift,
                sec_neg_shift,
                pri_tap0,
                pri_tap1,
            ),
        }
    }
}
