//! `aarch64` NEON `pal_idx_finish` çekirdeği (`F5-PAL` `1` yuva; `P51` / `F5a`).
//!
//! ## Güvenlik ve ISA Sınırı (`GUV-02`, `S3ze`)
//!
//! - Bu modül `#![forbid(unsafe_code)]` taşır; tüm bellek yükleme/saklama
//!   işlemleri [`super::mem`] içindeki sabit boyutlu dizi yardımcılarıyla yapılır.
//! - Kopyasız O(1) sözleşme: girişte [`pal_idx_finish_sozlesmesi`] denetlenir;
//!   sözleşme dışı girdide [`pal_idx_finish_skaler_devret`] ile skaler gövdeye
//!   devredilir.
//! - Giriş işaretçisi yalnız [`NeonToken`] ile alınabilir ([`pal_idx_finish_neon_fn`]).
//! - Sıcak döngülerde `get`/`get_mut` + `first_chunk` kullanılarak
//!   `panic_bounds_check` üretimi sıfırlanmıştır.

#![forbid(unsafe_code)]

use core::arch::aarch64::{
    vdup_n_u32, vget_lane_u16, vget_lane_u32, vget_low_u8, vorr_u8, vreinterpret_u8_u32,
    vreinterpret_u16_u8, vreinterpret_u32_u8, vshl_n_u8, vuzp1_u8, vuzp1q_u8, vuzp2_u8, vuzp2q_u8,
};

use crate::cpu::NeonToken;
use crate::dsp::arm::mem;
use crate::dsp::pal::{PalIdxFinishFn, pal_idx_finish_skaler_devret, pal_idx_finish_sozlesmesi};

/// `pal/pal_idx_finish` ARM64 NEON yuva işaretçisi.
#[must_use]
pub const fn pal_idx_finish_neon_fn(_token: NeonToken) -> PalIdxFinishFn {
    mem::pal_idx_finish_entry
}

#[target_feature(enable = "neon")]
pub(super) fn pal_idx_finish_dispatch_neon(
    dst: &mut [u8],
    src: &[u8],
    bw: usize,
    bh: usize,
    w: usize,
    h: usize,
) {
    if !pal_idx_finish_sozlesmesi(dst.len(), src.len(), bw, bh, w, h) {
        pal_idx_finish_skaler_devret(dst, src, bw, bh, w, h);
        return;
    }
    pal_idx_finish_neon(dst, src, bw, bh, w, h);
}

#[target_feature(enable = "neon")]
fn pal_idx_finish_neon(dst: &mut [u8], src: &[u8], bw: usize, bh: usize, w: usize, h: usize) {
    let dst_width = w / 2;
    let dst_bw = bw / 2;

    for y in 0..h {
        let src_row = y * bw;
        let dst_row = y * dst_bw;
        let mut sx = 0usize;
        while sx + 16 <= w {
            if let (Some(src16), Some(dst8)) = (
                src.get(src_row + sx..).and_then(|s| s.first_chunk::<16>()),
                dst.get_mut(dst_row + sx / 2..)
                    .and_then(|d| d.first_chunk_mut::<8>()),
            ) {
                let a = mem::load_u8x16(src16);
                let even = vget_low_u8(vuzp1q_u8(a, a));
                let odd = vget_low_u8(vuzp2q_u8(a, a));
                let packed = vorr_u8(even, vshl_n_u8::<4>(odd));
                mem::store_u8x8(dst8, packed);
            }
            sx += 16;
        }
        while sx + 8 <= w {
            if let (Some(src8), Some(dst4)) = (
                src.get(src_row + sx..).and_then(|s| s.first_chunk::<8>()),
                dst.get_mut(dst_row + sx / 2..)
                    .and_then(|d| d.first_chunk_mut::<4>()),
            ) {
                let a = mem::load_u8x8(src8);
                let even = vuzp1_u8(a, a);
                let odd = vuzp2_u8(a, a);
                let packed = vorr_u8(even, vshl_n_u8::<4>(odd));
                let p32 = vget_lane_u32::<0>(vreinterpret_u32_u8(packed));
                dst4.copy_from_slice(&p32.to_le_bytes());
            }
            sx += 8;
        }
        while sx + 4 <= w {
            if let (Some(src4), Some(dst2)) = (
                src.get(src_row + sx..).and_then(|s| s.first_chunk::<4>()),
                dst.get_mut(dst_row + sx / 2..)
                    .and_then(|d| d.first_chunk_mut::<2>()),
            ) {
                let a = vreinterpret_u8_u32(vdup_n_u32(u32::from_le_bytes(*src4)));
                let even = vuzp1_u8(a, a);
                let odd = vuzp2_u8(a, a);
                let packed = vorr_u8(even, vshl_n_u8::<4>(odd));
                let p16 = vget_lane_u16::<0>(vreinterpret_u16_u8(packed));
                dst2.copy_from_slice(&p16.to_le_bytes());
            }
            sx += 4;
        }
        if dst_width < dst_bw
            && let Some(&last_px) = src.get(src_row + w.saturating_sub(1))
            && let Some(pad_slice) = dst.get_mut(dst_row + dst_width..dst_row + dst_bw)
        {
            pad_slice.fill(last_px.wrapping_mul(0x11));
        }
    }

    if h < bh {
        let last_row = (h - 1) * dst_bw;
        for y in h..bh {
            let dst_row = y * dst_bw;
            for x in 0..dst_bw {
                if let Some(&v) = dst.get(last_row + x)
                    && let Some(cell) = dst.get_mut(dst_row + x)
                {
                    *cell = v;
                }
            }
        }
    }
}
