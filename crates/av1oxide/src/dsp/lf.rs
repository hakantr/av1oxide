//! Deblocking (loop filter) çekirdek sözleşmesi — dav1d
//! `Dav1dLoopFilterDSPContext` (`src/loopfilter.h:47-58`).
//!
//! Yuva: `loop_filter_sb[2][2]` — `[0]` luma, `[1]` kroma; `[..][0]` dikey
//! kenarlar (yatay süzme, `h`), `[..][1]` yatay kenarlar (dikey süzme, `v`).
//! Gövdeler [`crate::dsp::scalar::lf`] içindedir; her bit derinliğinin kendi
//! tablosu vardır (F3c).
//!
//! ## İmza (F3c)
//!
//! dav1d girişi SB başına bit maskesi alır (`vmask[3]`); bu port kare geneli
//! **bayt** maskesi tutar (filtre indeksi + 1, 0 = kenar yok; bkz.
//! [`crate::decode::loopfilter`] modül belgesi). Giriş bir düzlemin bir yöndeki
//! kenarlarını `y0..y0+count` 4×4 satırı ve `w4` 4×4 sütunu için işler:
//!
//! - `dst`/`stride`: düzlem dilimi ve bayt satır adımı;
//! - `mask`/`mask_stride`: düzlemin 4×4 ızgarasında kenar maskesi;
//! - `level`/`lvl_stride`/`lvl_idx`: seviye ızgarası (`[y_dikey, y_yatay, u, v]`)
//!   ve kullanılacak bileşen;
//! - `lut`: `Av1FilterLUT` (`e`/`i` sınırları; `h = seviye >> 4`).
//!
//! Sözleşme dışı girdide (kısa dilim) güvenli indeksleme panikler.

#![forbid(unsafe_code)]

use crate::dsp::scalar::lf as scalar;

/// `Av1FilterLUT` (`src/lf_mask.h:36-40`).
#[derive(Clone, Copy, Debug)]
pub struct LimLut {
    /// `i[level]`.
    pub i: [i32; 64],
    /// `e[level]`.
    pub e: [i32; 64],
    /// `sharp[0] = (sharp + 3) >> 2`, `sharp[1] = if sharp != 0 { 9 - sharp } else { 0xff }`.
    pub sharp: [u8; 2],
}

impl Default for LimLut {
    fn default() -> Self {
        Self {
            i: [0; 64],
            e: [0; 64],
            sharp: [0, 0xff],
        }
    }
}

/// `dav1d_calc_eih` (`src/lf_mask.c:382-399`).
#[must_use]
pub fn calc_eih(sharp: u8) -> LimLut {
    let s = i32::from(sharp);
    let mut lut = LimLut {
        i: [0; 64],
        e: [0; 64],
        sharp: [
            ((sharp + 3) >> 2),
            if sharp != 0 { 9 - sharp } else { 0xff },
        ],
    };
    let mut level = 0usize;
    while level < 64 {
        let mut limit = level as i32;
        if s > 0 {
            limit >>= (s + 3) >> 2;
            if limit > 9 - s {
                limit = 9 - s;
            }
        }
        if limit < 1 {
            limit = 1;
        }
        lut.i[level] = limit;
        lut.e[level] = 2 * (level as i32 + 2) + limit;
        level += 1;
    }
    lut
}

/// `loop_filter_sb` giriş imzası (modül belgesi).
pub type LoopFilterSbFn = fn(
    dst: &mut [u8],
    stride: usize,
    mask: &[u8],
    mask_stride: usize,
    level: &[[u8; 4]],
    lvl_stride: usize,
    lvl_idx: usize,
    lut: &LimLut,
    w4: usize,
    y0: usize,
    count: usize,
);

/// Loop filter çekirdek ailesi — dav1d `Dav1dLoopFilterDSPContext`.
#[derive(Clone, Copy)]
pub struct LoopFilterDsp {
    /// `loop_filter_sb[düzlem: 0 = luma, 1 = kroma][yön: 0 = h, 1 = v]`.
    pub loop_filter_sb: [[LoopFilterSbFn; 2]; 2],
}

impl LoopFilterDsp {
    /// `BPC` bit derinliğinin skaler tablosu (`loopfilter_tmpl.c:257-266`).
    #[must_use]
    pub const fn scalar<const BPC: u32>() -> Self {
        Self {
            loop_filter_sb: [
                [scalar::lpf_h_sb_y::<BPC>, scalar::lpf_v_sb_y::<BPC>],
                [scalar::lpf_h_sb_uv::<BPC>, scalar::lpf_v_sb_uv::<BPC>],
            ],
        }
    }

    /// Çalışma zamanı bit derinliğinin skaler tablosu (kurulum ve testler).
    #[must_use]
    pub const fn scalar_for(bit_depth: crate::frame::BitDepth) -> Self {
        match bit_depth {
            crate::frame::BitDepth::Eight => Self::scalar::<8>(),
            crate::frame::BitDepth::Ten => Self::scalar::<10>(),
            crate::frame::BitDepth::Twelve => Self::scalar::<12>(),
        }
    }
}

impl std::fmt::Debug for LoopFilterDsp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LoopFilterDsp { loop_filter_sb: [[fn; 2]; 2] }")
    }
}

/// P46: SIMD `loop_filter_sb` girişlerinin sözleşme dışı girdide skaler gövdeye
/// devrettiği çağrı sayısı (süreç başından beri). Çözme hattında 0 olmalıdır.
static LF_SKALER_DEVIR: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// [`LF_SKALER_DEVIR`] değeri.
#[must_use]
pub fn lf_skaler_devir_sayisi() -> u64 {
    LF_SKALER_DEVIR.load(core::sync::atomic::Ordering::Relaxed)
}

/// Skaler devri sayar (yalnız SIMD girişlerinin soğuk yolu).
#[cold]
pub fn lf_skaler_devri_say() {
    LF_SKALER_DEVIR.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
}

/// P46: `loop_filter_sb[plane][dir]` SIMD girişinin O(1) sözleşmesi:
/// - `plane < 2`, `dir < 2`, `lvl_idx < 4`
/// - `BPC > 8` ise `stride % 2 == 0`
/// - `count == 0` veya `w4 == 0` (ya da `dir == 0 && w4 <= 1` veya `dir == 1 && y0 == 0 && count == 1`)
///   ise etkin kenar yoktur (geçerli).
/// - Etkin kenar varsa:
///   - `mask_stride >= w4` ve `lvl_stride >= w4`
///   - `mask_len >= (y0 + count - 1) * mask_stride + w4`
///   - `level_len >= (y0 + count - 1) * lvl_stride + w4`
///   - `dir == 0` (dikey kenarlar `gx = 1..w4`, yatay süzme):
///     her 4 satırlık blokta `gx * 4` sütunundan geriye en çok 7 piksel (`wd = 16`,
///     yani `gx >= 2` veya `gx == 1` iken en çok 4 piksel), ileriye en çok 6 piksel
///     okunur; `stride >= w4 * 4 * bpp` ve `dst_len >= ((y0 + count) * 4 - 1) * stride + w4 * 4 * bpp`
///   - `dir == 1` (yatay kenarlar `gy = max(y0, 1)..y0 + count`, dikey süzme):
///     `gy * 4` satırından geriye en çok 7 satır (`wd = 16`, `gy >= 2` veya `gy == 1` iken 4 satır),
///     ileriye en çok 6 satır okunur; `stride >= w4 * 4 * bpp` ve
///     `dst_len >= ((y0 + count - 1) * 4 + 3) * stride + w4 * 4 * bpp`
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn lpf_sb_sozlesmesi<const BPC: u32>(
    plane: usize,
    dir: usize,
    dst_len: usize,
    stride: usize,
    mask_len: usize,
    mask_stride: usize,
    level_len: usize,
    lvl_stride: usize,
    lvl_idx: usize,
    w4: usize,
    y0: usize,
    count: usize,
) -> bool {
    let bpp = if BPC == 8 { 1 } else { 2 };
    if plane >= 2 || dir >= 2 || lvl_idx >= 4 || (BPC > 8 && !stride.is_multiple_of(2)) {
        return false;
    }
    if count == 0 || w4 == 0 || (dir == 0 && w4 <= 1) || (dir == 1 && y0 == 0 && count == 1) {
        return true;
    }
    let Some(y_end) = y0.checked_add(count) else {
        return false;
    };
    if mask_stride < w4 || lvl_stride < w4 {
        return false;
    }
    let Some(mask_need) = (y_end - 1)
        .checked_mul(mask_stride)
        .and_then(|v| v.checked_add(w4))
    else {
        return false;
    };
    let Some(lvl_need) = (y_end - 1)
        .checked_mul(lvl_stride)
        .and_then(|v| v.checked_add(w4))
    else {
        return false;
    };
    if mask_len < mask_need || level_len < lvl_need {
        return false;
    }
    let Some(row_bytes) = w4.checked_mul(4 * bpp) else {
        return false;
    };
    if stride < row_bytes {
        return false;
    }
    let Some(last_px_row) = y_end.checked_mul(4).and_then(|r| r.checked_sub(1)) else {
        return false;
    };
    let Some(dst_need) = last_px_row
        .checked_mul(stride)
        .and_then(|v| v.checked_add(row_bytes))
    else {
        return false;
    };
    dst_len >= dst_need
}

/// P46: Sözleşme dışı `loop_filter_sb` girdisinde skaler devir sayacını artırır;
/// skaler gövdenin önkoşulları sağlanıyorsa ilgili skaler gövdeyi çağırır,
/// değilse paniklemeden döner.
#[cold]
#[allow(clippy::too_many_arguments)]
pub fn lpf_sb_skaler_devret<const BPC: u32>(
    plane: usize,
    dir: usize,
    dst: &mut [u8],
    stride: usize,
    mask: &[u8],
    mask_stride: usize,
    level: &[[u8; 4]],
    lvl_stride: usize,
    lvl_idx: usize,
    lut: &LimLut,
    w4: usize,
    y0: usize,
    count: usize,
) {
    lf_skaler_devri_say();
    if lpf_sb_sozlesmesi::<BPC>(
        plane,
        dir,
        dst.len(),
        stride,
        mask.len(),
        mask_stride,
        level.len(),
        lvl_stride,
        lvl_idx,
        w4,
        y0,
        count,
    ) {
        let f = LoopFilterDsp::scalar::<BPC>().loop_filter_sb[plane][dir];
        f(
            dst,
            stride,
            mask,
            mask_stride,
            level,
            lvl_stride,
            lvl_idx,
            lut,
            w4,
            y0,
            count,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p46_lf_sozlesmesi_ve_paniksiz_skaler_devir() {
        assert!(lpf_sb_sozlesmesi::<8>(
            0,
            0,
            64 * 64,
            64,
            16 * 16,
            16,
            16 * 16,
            16,
            0,
            16,
            0,
            16
        ));
        // Kısa dst / kısa mask / kısa level / tek stride (16bpc) reddedilir.
        assert!(!lpf_sb_sozlesmesi::<8>(
            0,
            0,
            63 * 64,
            64,
            16 * 16,
            16,
            16 * 16,
            16,
            0,
            16,
            0,
            16
        ));
        assert!(!lpf_sb_sozlesmesi::<8>(
            0,
            0,
            64 * 64,
            64,
            16 * 15,
            16,
            16 * 16,
            16,
            0,
            16,
            0,
            16
        ));
        assert!(!lpf_sb_sozlesmesi::<10>(
            0,
            0,
            128 * 64,
            129,
            16 * 16,
            16,
            16 * 16,
            16,
            0,
            16,
            0,
            16
        ));

        let once = lf_skaler_devir_sayisi();
        let mut kisa_dst = [0u8; 16];
        let mask = [0u8; 16];
        let level = [[0u8; 4]; 16];
        let lut = LimLut::default();
        lpf_sb_skaler_devret::<8>(
            0,
            0,
            &mut kisa_dst,
            16,
            &mask,
            4,
            &level,
            4,
            0,
            &lut,
            4,
            0,
            4,
        );
        assert!(lf_skaler_devir_sayisi() > once);
    }
}
