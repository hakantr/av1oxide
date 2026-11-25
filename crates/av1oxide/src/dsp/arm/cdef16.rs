//! `aarch64` NEON 16-bpc (`10` ve `12` bit) CDEF çekirdekleri (`F5-CDEF` `4 / 8` kayıt = `8` yuva; `P51` / `F5a`, `P58-M4` vektörizasyonu).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Sembol | Matris Kaydı | Kaynak |
//! |---|---|---|
//! | `dav1d_cdef_find_dir_16bpc_neon` | `D00441` (`10` & `12` bit) | `src/arm/64/cdef_tmpl.S:269-518`, `src/arm/cdef_init_tmpl.c:55` |
//! | `dav1d_cdef_filter8_16bpc_neon` | `D00443` (`fb[0]`, `8×8`, `10` & `12` bit) | `src/arm/64/cdef16.S:35-315`, `src/arm/64/cdef_tmpl.S:30-267` |
//! | `dav1d_cdef_filter4_16bpc_neon` | `D00445` (`fb[1]`, `4×8`), `D00447` (`fb[2]`, `4×4`, `10` & `12` bit) | `src/arm/64/cdef16.S:35-315`, `src/arm/64/cdef_tmpl.S:30-267` |
//!
//! ## Güvenlik ve ISA Sınırı (`GUV-02`, `S3ze`)
//!
//! - Bu modül `#![forbid(unsafe_code)]` taşır; tüm bellek yükleme/saklama
//!   işlemleri [`super::mem`] içindeki sabit boyutlu dizi yardımcılarıyla yapılır.
//! - Giriş işaretçileri yalnız [`NeonToken`] ile alınabilir ([`find_dir_neon_fn`],
//!   [`filter_8x8_neon_fn`], [`filter_4x8_neon_fn`], [`filter_4x4_neon_fn`]).

#![forbid(unsafe_code)]

use core::arch::aarch64::{
    int16x8_t, uint16x8_t, vdupq_n_s16, vdupq_n_u16, vget_high_s16, vget_lane_u64, vget_low_s16,
    vreinterpret_u16_s16, vreinterpret_u64_u16, vreinterpretq_s16_u8, vreinterpretq_s16_u16,
    vreinterpretq_u8_s16, vreinterpretq_u16_u8, vshlq_u16, vsubq_s16,
};

use super::giris::store_i16x8;
use crate::cpu::NeonToken;
use crate::dsp::arm::cdef::{
    CDEF_DIRECTIONS_8, CDEF_DIRECTIONS_16, PAD_BASE, PAD_BASE_4, PAD_FILL_16, PAD_STRIDE,
    PAD_STRIDE_4, cdef_constrain_i16x8, cdef_filter_core_8lanes, cdef_load_tmp_w4_2rows,
    cdef_load_tmp_w8, compute_dir_from_rows_neon,
};
use crate::dsp::arm::mem;
use crate::dsp::cdef::{
    CdefBitDepth, CdefBlockSize, CdefDirFn, CdefDirResult, CdefEdgeFlags, CdefFilterFn,
    CdefFilterParams, clamped_shift, ulog2,
};

/// 16-bpc (`BPC ∈ {10, 12}`) `cdef/dir` ARM64 NEON yuva işaretçisi (`D00441`).
#[must_use]
pub const fn find_dir_neon_fn<const BPC: u32>(_token: NeonToken) -> CdefDirFn {
    mem::find_dir_16bpc_entry::<BPC>
}

/// 16-bpc (`BPC ∈ {10, 12}`) `cdef/fb[0]` (`8×8`) ARM64 NEON yuva işaretçisi (`D00443`).
#[must_use]
pub const fn filter_8x8_neon_fn<const BPC: u32>(_token: NeonToken) -> CdefFilterFn {
    mem::filter_16bpc_entry::<BPC, 8, 8>
}

/// 16-bpc (`BPC ∈ {10, 12}`) `cdef/fb[1]` (`4×8`) ARM64 NEON yuva işaretçisi (`D00445`).
#[must_use]
pub const fn filter_4x8_neon_fn<const BPC: u32>(_token: NeonToken) -> CdefFilterFn {
    mem::filter_16bpc_entry::<BPC, 4, 8>
}

/// 16-bpc (`BPC ∈ {10, 12}`) `cdef/fb[2]` (`4×4`) ARM64 NEON yuva işaretçisi (`D00447`).
#[must_use]
pub const fn filter_4x4_neon_fn<const BPC: u32>(_token: NeonToken) -> CdefFilterFn {
    mem::filter_16bpc_entry::<BPC, 4, 4>
}

#[target_feature(enable = "neon")]
pub(super) fn find_dir_16bpc_neon<const BPC: u32>(
    img: &[u8],
    stride: usize,
    bit_depth: CdefBitDepth,
) -> CdefDirResult {
    debug_assert_eq!(bit_depth.bits(), BPC);
    if stride == 0 || img.len() < 7usize.saturating_mul(stride).saturating_add(16) {
        return crate::dsp::scalar::cdef::find_dir::<BPC>(img, stride, bit_depth);
    }
    cdef_find_dir_16bpc_neon(img, stride, bit_depth)
}

#[target_feature(enable = "neon")]
pub(super) fn filter_16bpc_dispatch_neon<const BPC: u32, const W: usize, const H: usize>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    debug_assert_eq!(params.bit_depth.bits(), BPC);
    if params.damping > 13 {
        crate::dsp::scalar::cdef::filter::<BPC>(dst, stride, left, top, bottom, params);
        return;
    }
    let w = params.size.w();
    let h = params.size.h();
    if (w, h) != (W, H) {
        match params.size {
            CdefBlockSize::W8H8 => {
                cdef_filter_16bpc_neon::<BPC, 8, 8>(dst, stride, left, top, bottom, params)
            }
            CdefBlockSize::W4H8 => {
                cdef_filter_16bpc_neon::<BPC, 4, 8>(dst, stride, left, top, bottom, params)
            }
            CdefBlockSize::W4H4 => {
                cdef_filter_16bpc_neon::<BPC, 4, 4>(dst, stride, left, top, bottom, params)
            }
        }
        return;
    }
    cdef_filter_16bpc_neon::<BPC, W, H>(dst, stride, left, top, bottom, params);
}

#[inline]
#[target_feature(enable = "neon")]
fn load_dir_row_16bpc(img: &[u8], off: usize, v_shift: int16x8_t, v_128: int16x8_t) -> int16x8_t {
    if let Some(src16) = img.get(off..).and_then(|s| s.first_chunk::<16>()) {
        let raw_u16 = vreinterpretq_u16_u8(mem::load_u8x16(src16));
        let shifted = vreinterpretq_s16_u16(vshlq_u16(raw_u16, v_shift));
        vsubq_s16(shifted, v_128)
    } else {
        vdupq_n_s16(0)
    }
}

#[target_feature(enable = "neon")]
fn cdef_find_dir_16bpc_neon(img: &[u8], stride: usize, bit_depth: CdefBitDepth) -> CdefDirResult {
    let shift = -(bit_depth.min_8() as i16);
    let v_shift = vdupq_n_s16(shift);
    let v_128 = vdupq_n_s16(128);
    let rows = [
        load_dir_row_16bpc(img, 0, v_shift, v_128),
        load_dir_row_16bpc(img, stride, v_shift, v_128),
        load_dir_row_16bpc(img, 2 * stride, v_shift, v_128),
        load_dir_row_16bpc(img, 3 * stride, v_shift, v_128),
        load_dir_row_16bpc(img, 4 * stride, v_shift, v_128),
        load_dir_row_16bpc(img, 5 * stride, v_shift, v_128),
        load_dir_row_16bpc(img, 6 * stride, v_shift, v_128),
        load_dir_row_16bpc(img, 7 * stride, v_shift, v_128),
    ];
    compute_dir_from_rows_neon(&rows)
}

#[inline]
#[target_feature(enable = "neon")]
fn copy_row8_16bpc(tmp: &mut [i16; 256], t_idx: usize, src: &[u8], byte_off: usize) {
    if let Some(s16) = src.get(byte_off..).and_then(|s| s.first_chunk::<16>())
        && let Some(d8) = tmp.get_mut(t_idx..).and_then(|d| d.first_chunk_mut::<8>())
    {
        store_i16x8(d8, vreinterpretq_s16_u8(mem::load_u8x16(s16)));
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn copy_row4_16bpc(tmp: &mut [i16; 128], t_idx: usize, src: &[u8], byte_off: usize) {
    if let Some(s8) = src.get(byte_off..).and_then(|s| s.first_chunk::<8>())
        && let Some(d4) = tmp.get_mut(t_idx..).and_then(|d| d.first_chunk_mut::<4>())
    {
        let w = u64::from_le_bytes(*s8);
        d4[0] = w as i16;
        d4[1] = (w >> 16) as i16;
        d4[2] = (w >> 32) as i16;
        d4[3] = (w >> 48) as i16;
    }
}

#[target_feature(enable = "neon")]
fn pad_w8_16bpc_into_tmp<const H: usize>(
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
            if let Some(s24) = top.get(row_base..).and_then(|s| s.first_chunk::<24>())
                && let Some((s16, s_rest)) = s24.split_first_chunk::<16>()
                && let Some(s8) = s_rest.first_chunk::<8>()
                && let Some(d12) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<12>())
                && let Some((d8, d_rest)) = d12.split_first_chunk_mut::<8>()
                && let Some(d4) = d_rest.first_chunk_mut::<4>()
            {
                store_i16x8(d8, vreinterpretq_s16_u8(mem::load_u8x16(s16)));
                let w = u64::from_le_bytes(*s8);
                d4[0] = w as i16;
                d4[1] = (w >> 16) as i16;
                d4[2] = (w >> 32) as i16;
                d4[3] = (w >> 48) as i16;
            }
        }
        for y in 0..H {
            let t_base = PAD_BASE as usize + y * PAD_STRIDE;
            let l_base = y * 4;
            if let Some(s4) = left.get(l_base..).and_then(|s| s.first_chunk::<4>())
                && let Some(d2) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                let w = u32::from_le_bytes(*s4);
                d2[0] = w as i16;
                d2[1] = (w >> 16) as i16;
            }
            let d_base = y * stride;
            if let Some(s20) = dst.get(d_base..).and_then(|s| s.first_chunk::<20>())
                && let Some((s16, s_rest)) = s20.split_first_chunk::<16>()
                && let Some(s4) = s_rest.first_chunk::<4>()
                && let Some(d10) = tmp
                    .get_mut(t_base..)
                    .and_then(|d| d.first_chunk_mut::<10>())
                && let Some((d8, d_rest)) = d10.split_first_chunk_mut::<8>()
                && let Some(d2) = d_rest.first_chunk_mut::<2>()
            {
                store_i16x8(d8, vreinterpretq_s16_u8(mem::load_u8x16(s16)));
                let w = u32::from_le_bytes(*s4);
                d2[0] = w as i16;
                d2[1] = (w >> 16) as i16;
            }
        }
        for row in 0..2usize {
            let t_base = PAD_BASE as usize + (H + row) * PAD_STRIDE;
            let row_base = row * stride;
            if let Some(s24) = bottom.get(row_base..).and_then(|s| s.first_chunk::<24>())
                && let Some((s16, s_rest)) = s24.split_first_chunk::<16>()
                && let Some(s8) = s_rest.first_chunk::<8>()
                && let Some(d12) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<12>())
                && let Some((d8, d_rest)) = d12.split_first_chunk_mut::<8>()
                && let Some(d4) = d_rest.first_chunk_mut::<4>()
            {
                store_i16x8(d8, vreinterpretq_s16_u8(mem::load_u8x16(s16)));
                let w = u64::from_le_bytes(*s8);
                d4[0] = w as i16;
                d4[1] = (w >> 16) as i16;
                d4[2] = (w >> 32) as i16;
                d4[3] = (w >> 48) as i16;
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
                && let Some(s4) = top.get(row_base..).and_then(|s| s.first_chunk::<4>())
                && let Some(d2) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                let w = u32::from_le_bytes(*s4);
                d2[0] = w as i16;
                d2[1] = (w >> 16) as i16;
            }
            copy_row8_16bpc(tmp, t_base, top, row_base + 4);
            if has_right
                && let Some(s4) = top.get(row_base + 20..).and_then(|s| s.first_chunk::<4>())
                && let Some(d2) = tmp
                    .get_mut(t_base + 8..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                let w = u32::from_le_bytes(*s4);
                d2[0] = w as i16;
                d2[1] = (w >> 16) as i16;
            }
        }
    }
    if has_left {
        for y in 0..H {
            let t_base = PAD_BASE as usize + y * PAD_STRIDE;
            let l_base = y * 4;
            if let Some(s4) = left.get(l_base..).and_then(|s| s.first_chunk::<4>())
                && let Some(d2) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                let w = u32::from_le_bytes(*s4);
                d2[0] = w as i16;
                d2[1] = (w >> 16) as i16;
            }
        }
    }
    for y in 0..H {
        let t_base = PAD_BASE as usize + y * PAD_STRIDE;
        let d_base = y * stride;
        copy_row8_16bpc(tmp, t_base, dst, d_base);
        if has_right
            && let Some(s4) = dst.get(d_base + 16..).and_then(|s| s.first_chunk::<4>())
            && let Some(d2) = tmp
                .get_mut(t_base + 8..)
                .and_then(|d| d.first_chunk_mut::<2>())
        {
            let w = u32::from_le_bytes(*s4);
            d2[0] = w as i16;
            d2[1] = (w >> 16) as i16;
        }
    }
    if has_bottom {
        for row in 0..2usize {
            let t_base = PAD_BASE as usize + (H + row) * PAD_STRIDE;
            let row_base = row * stride;
            if has_left
                && let Some(s4) = bottom.get(row_base..).and_then(|s| s.first_chunk::<4>())
                && let Some(d2) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                let w = u32::from_le_bytes(*s4);
                d2[0] = w as i16;
                d2[1] = (w >> 16) as i16;
            }
            copy_row8_16bpc(tmp, t_base, bottom, row_base + 4);
            if has_right
                && let Some(s4) = bottom
                    .get(row_base + 20..)
                    .and_then(|s| s.first_chunk::<4>())
                && let Some(d2) = tmp
                    .get_mut(t_base + 8..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                let w = u32::from_le_bytes(*s4);
                d2[0] = w as i16;
                d2[1] = (w >> 16) as i16;
            }
        }
    }
}

#[target_feature(enable = "neon")]
fn pad_w4_16bpc_into_tmp<const H: usize>(
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
            if let Some(s16) = top.get(row_base..).and_then(|s| s.first_chunk::<16>())
                && let Some(d8) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<8>())
            {
                store_i16x8(d8, vreinterpretq_s16_u8(mem::load_u8x16(s16)));
            }
        }
        for y in 0..H {
            let t_base = PAD_BASE_4 as usize + y * PAD_STRIDE_4;
            let l_base = y * 4;
            if let Some(s4) = left.get(l_base..).and_then(|s| s.first_chunk::<4>())
                && let Some(d2) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                let w = u32::from_le_bytes(*s4);
                d2[0] = w as i16;
                d2[1] = (w >> 16) as i16;
            }
            let d_base = y * stride;
            if let Some(s12) = dst.get(d_base..).and_then(|s| s.first_chunk::<12>())
                && let Some((s8, s_rest)) = s12.split_first_chunk::<8>()
                && let Some(s4) = s_rest.first_chunk::<4>()
                && let Some(d6) = tmp.get_mut(t_base..).and_then(|d| d.first_chunk_mut::<6>())
            {
                let w0 = u64::from_le_bytes(*s8);
                let w1 = u32::from_le_bytes(*s4);
                d6[0] = w0 as i16;
                d6[1] = (w0 >> 16) as i16;
                d6[2] = (w0 >> 32) as i16;
                d6[3] = (w0 >> 48) as i16;
                d6[4] = w1 as i16;
                d6[5] = (w1 >> 16) as i16;
            }
        }
        for row in 0..2usize {
            let t_base = PAD_BASE_4 as usize + (H + row) * PAD_STRIDE_4;
            let row_base = row * stride;
            if let Some(s16) = bottom.get(row_base..).and_then(|s| s.first_chunk::<16>())
                && let Some(d8) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<8>())
            {
                store_i16x8(d8, vreinterpretq_s16_u8(mem::load_u8x16(s16)));
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
                && let Some(s4) = top.get(row_base..).and_then(|s| s.first_chunk::<4>())
                && let Some(d2) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                let w = u32::from_le_bytes(*s4);
                d2[0] = w as i16;
                d2[1] = (w >> 16) as i16;
            }
            copy_row4_16bpc(tmp, t_base, top, row_base + 4);
            if has_right
                && let Some(s4) = top.get(row_base + 12..).and_then(|s| s.first_chunk::<4>())
                && let Some(d2) = tmp
                    .get_mut(t_base + 4..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                let w = u32::from_le_bytes(*s4);
                d2[0] = w as i16;
                d2[1] = (w >> 16) as i16;
            }
        }
    }
    if has_left {
        for y in 0..H {
            let t_base = PAD_BASE_4 as usize + y * PAD_STRIDE_4;
            let l_base = y * 4;
            if let Some(s4) = left.get(l_base..).and_then(|s| s.first_chunk::<4>())
                && let Some(d2) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                let w = u32::from_le_bytes(*s4);
                d2[0] = w as i16;
                d2[1] = (w >> 16) as i16;
            }
        }
    }
    for y in 0..H {
        let t_base = PAD_BASE_4 as usize + y * PAD_STRIDE_4;
        let d_base = y * stride;
        copy_row4_16bpc(tmp, t_base, dst, d_base);
        if has_right
            && let Some(s4) = dst.get(d_base + 8..).and_then(|s| s.first_chunk::<4>())
            && let Some(d2) = tmp
                .get_mut(t_base + 4..)
                .and_then(|d| d.first_chunk_mut::<2>())
        {
            let w = u32::from_le_bytes(*s4);
            d2[0] = w as i16;
            d2[1] = (w >> 16) as i16;
        }
    }
    if has_bottom {
        for row in 0..2usize {
            let t_base = PAD_BASE_4 as usize + (H + row) * PAD_STRIDE_4;
            let row_base = row * stride;
            if has_left
                && let Some(s4) = bottom.get(row_base..).and_then(|s| s.first_chunk::<4>())
                && let Some(d2) = tmp
                    .get_mut(t_base - 2..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                let w = u32::from_le_bytes(*s4);
                d2[0] = w as i16;
                d2[1] = (w >> 16) as i16;
            }
            copy_row4_16bpc(tmp, t_base, bottom, row_base + 4);
            if has_right
                && let Some(s4) = bottom
                    .get(row_base + 12..)
                    .and_then(|s| s.first_chunk::<4>())
                && let Some(d2) = tmp
                    .get_mut(t_base + 4..)
                    .and_then(|d| d.first_chunk_mut::<2>())
            {
                let w = u32::from_le_bytes(*s4);
                d2[0] = w as i16;
                d2[1] = (w >> 16) as i16;
            }
        }
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn run_w8_16bpc_loop<const H: usize, const PRI: bool, const SEC: bool>(
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
        if let Some(dst16) = dst
            .get_mut(y * stride..)
            .and_then(|s| s.first_chunk_mut::<16>())
        {
            mem::store_u8x16(dst16, vreinterpretq_u8_s16(out8));
        }
    }
}

#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn run_w4_16bpc_loop<const H: usize, const PRI: bool, const SEC: bool>(
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
        let lo64 = vget_lane_u64::<0>(vreinterpret_u64_u16(vreinterpret_u16_s16(vget_low_s16(
            out8,
        ))));
        let hi64 = vget_lane_u64::<0>(vreinterpret_u64_u16(vreinterpret_u16_s16(vget_high_s16(
            out8,
        ))));
        if let Some(dst8) = dst
            .get_mut(y * stride..)
            .and_then(|s| s.first_chunk_mut::<8>())
        {
            *dst8 = lo64.to_le_bytes();
        }
        if let Some(dst8) = dst
            .get_mut((y + 1) * stride..)
            .and_then(|s| s.first_chunk_mut::<8>())
        {
            *dst8 = hi64.to_le_bytes();
        }
        y += 2;
    }
}

#[target_feature(enable = "neon")]
fn cdef_filter_16bpc_neon<const BPC: u32, const W: usize, const H: usize>(
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
    let bd_min_8 = params.bit_depth.min_8();
    let v_pri_thresh = vdupq_n_u16(pri_strength);
    let v_sec_thresh = vdupq_n_u16(sec_strength);
    let pri_tap0 = (4 - (((u32::from(params.pri_strength)) >> bd_min_8) & 1) as i32) as i16;
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
        pad_w8_16bpc_into_tmp::<H>(&mut tmp, dst, stride, left, top, bottom, params.edges);
        match (pri_strength != 0, sec_strength != 0) {
            (true, true) => run_w8_16bpc_loop::<H, true, true>(
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
            (true, false) => run_w8_16bpc_loop::<H, true, false>(
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
            _ => run_w8_16bpc_loop::<H, false, true>(
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
        pad_w4_16bpc_into_tmp::<H>(&mut tmp, dst, stride, left, top, bottom, params.edges);
        match (pri_strength != 0, sec_strength != 0) {
            (true, true) => run_w4_16bpc_loop::<H, true, true>(
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
            (true, false) => run_w4_16bpc_loop::<H, true, false>(
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
            _ => run_w4_16bpc_loop::<H, false, true>(
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
