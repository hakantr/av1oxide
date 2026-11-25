//! Refmvs DSP sözleşmesi — dav1d `Dav1dRefmvsDSPContext` (`src/refmvs.h:123-127`).
//!
//! Üç yuva: `save_tmvs`, `load_tmvs`, `splat_mv`. dav1d'de tablo **bit
//! derinliğinden bağımsızdır** (`dav1d_refmvs_dsp_init`, `src/refmvs.c:953`;
//! x86 sembolleri bit derinliği soneki taşımaz); bu port da tek tablo tutar
//! ([`crate::dsp::DspDispatch::refmvs`]). Gövdeler
//! [`crate::dsp::scalar::refmvs`] içindedir. Veri türleri
//! [`crate::decode::refmvs`] ve [`crate::decode::refmvs_tmvs`] içindedir.

#![forbid(unsafe_code)]

use crate::decode::refmvs::RefBlock;
use crate::decode::refmvs_tmvs::{MfmvPlan, TemporalBlock};
use crate::dsp::scalar::refmvs as scalar;

/// dav1d `save_tmvs_fn` (`src/refmvs.h:113-117`).
pub type SaveTmvsFn = fn(
    rp: &mut [TemporalBlock],
    stride: usize,
    grid: &[RefBlock],
    iw4: usize,
    ref_sign: &[u8; 7],
    col_end8: usize,
    row_end8: usize,
    col_start8: usize,
    row_start8: usize,
);

/// dav1d `load_tmvs_fn` (`src/refmvs.h:107-111`).
pub type LoadTmvsFn = fn(
    rp_proj: &mut [TemporalBlock],
    stride: usize,
    iw8: usize,
    ih8: usize,
    plan: &MfmvPlan,
    refs: &[Option<&[TemporalBlock]>; 7],
    tile_row_idx: usize,
    col_start8: usize,
    col_end8: usize,
    row_start8: usize,
    row_end8: usize,
);

/// dav1d `splat_mv_fn` (`src/refmvs.h:119-121`).
pub type SplatMvFn = fn(
    grid: &mut [RefBlock],
    iw4: usize,
    rmv: RefBlock,
    bx4: usize,
    by4: usize,
    bw4: usize,
    bh4: usize,
);

/// Refmvs çekirdek ailesi.
#[derive(Clone, Copy)]
pub struct RefmvsDsp {
    /// `load_tmvs`.
    pub load_tmvs: LoadTmvsFn,
    /// `save_tmvs`.
    pub save_tmvs: SaveTmvsFn,
    /// `splat_mv`.
    pub splat_mv: SplatMvFn,
}

impl RefmvsDsp {
    /// Skaler tablo (`dav1d_refmvs_dsp_init`, asm yok).
    pub const SCALAR: Self = Self {
        load_tmvs: scalar::load_tmvs,
        save_tmvs: scalar::save_tmvs,
        splat_mv: scalar::splat_mv,
    };
}

impl std::fmt::Debug for RefmvsDsp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RefmvsDsp { load_tmvs: fn, save_tmvs: fn, splat_mv: fn }")
    }
}

/// P46: SIMD `refmvs` girişlerinin sözleşme dışı girdide skaler gövdeye
/// devrettiği çağrı sayısı (süreç başından beri). Çözme hattında 0 olmalıdır.
static REFMVS_SKALER_DEVIR: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// [`REFMVS_SKALER_DEVIR`] değeri.
#[must_use]
pub fn refmvs_skaler_devir_sayisi() -> u64 {
    REFMVS_SKALER_DEVIR.load(core::sync::atomic::Ordering::Relaxed)
}

/// Skaler devri sayar (yalnız SIMD girişlerinin soğuk yolu).
#[cold]
pub fn refmvs_skaler_devri_say() {
    REFMVS_SKALER_DEVIR.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
}

/// P46: `save_tmvs` SIMD girişinin O(1) sözleşmesi:
/// - `col_start8 <= col_end8 <= stride` ve `stride > 0`
/// - `row_start8 <= row_end8`
/// - `iw4 >= col_end8 * 2`
/// - `row_start8 == row_end8` ise boş çağrı (geçerli); aksi hâlde:
///   - `rp_len >= (row_end8 - row_start8 - 1) * stride + col_end8`
///   - `col_start8 == col_end8` ise `grid` okunmaz; `col_start8 < col_end8` ise
///     `grid_len >= ((row_end8 - 1) * 2 + 1) * iw4 + ((col_end8 - 1) * 2 + 1) + 1`
#[must_use]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn save_tmvs_sozlesmesi(
    rp_len: usize,
    stride: usize,
    grid_len: usize,
    iw4: usize,
    col_end8: usize,
    row_end8: usize,
    col_start8: usize,
    row_start8: usize,
) -> bool {
    if stride == 0
        || iw4 == 0
        || col_start8 > col_end8
        || col_end8 > stride
        || row_start8 > row_end8
    {
        return false;
    }
    let rows = row_end8 - row_start8;
    if rows == 0 || col_start8 == col_end8 {
        return true;
    }
    let Some(rp_last_row) = (rows - 1).checked_mul(stride) else {
        return false;
    };
    let Some(rp_need) = rp_last_row.checked_add(col_end8) else {
        return false;
    };
    if rp_len < rp_need {
        return false;
    }
    let Some(last_row4) = (row_end8 - 1).checked_mul(2).and_then(|v| v.checked_add(1)) else {
        return false;
    };
    let Some(last_col4) = (col_end8 - 1).checked_mul(2).and_then(|v| v.checked_add(1)) else {
        return false;
    };
    let Some(grid_need) = last_row4
        .checked_mul(iw4)
        .and_then(|v| v.checked_add(last_col4))
        .and_then(|v| v.checked_add(1))
    else {
        return false;
    };
    grid_len >= grid_need
}

/// P46: `load_tmvs` SIMD girişinin O(1) sözleşmesi:
/// - `stride > 0`, `iw8 <= stride`, `col_start8 <= col_end8 <= stride`
/// - `row_start8 <= row_end8` ve `row_end8 - row_start8 <= 16`
/// - `plan.n_mfmvs <= 3` ve her aktif `n` için `plan.mfmv_ref[n] < 7`,
///   `plan.mfmv_ref2cur[n] == INVALID_REF2CUR` veya `(-31..=31)` içinde,
///   ve her `m` için `plan.mfmv_ref2ref[n][m] <= 31`
/// - `row_end8.min(ih8) > row_start8` ve `col_start8 < col_end8` ise:
///   hedef pencere `16 * stride * tile_row_idx + ((row_start8 & 15) + rows - 1) * stride + col_end8 <= rp_proj_len`
///   ve `y_proj_end` döngüsünde `(py & 15) * stride + col_end8 <= rp_proj_len - base`
/// - Her aktif referans dilimi `refs[r]` varsa ve `col_start8i < col_end8i` ise
///   `r.len() >= (row_end8.min(ih8) - 1) * stride + col_end8i`
#[must_use]
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn load_tmvs_sozlesmesi(
    rp_proj_len: usize,
    stride: usize,
    iw8: usize,
    ih8: usize,
    plan: &MfmvPlan,
    refs: &[Option<&[TemporalBlock]>; 7],
    tile_row_idx: usize,
    col_start8: usize,
    col_end8: usize,
    row_start8: usize,
    row_end8: usize,
) -> bool {
    if stride == 0
        || iw8 > stride
        || col_start8 > col_end8
        || col_end8 > stride
        || row_start8 > row_end8
        || row_end8 - row_start8 > 16
        || plan.n_mfmvs > 3
    {
        return false;
    }
    let Some(base) = 16usize
        .checked_mul(stride)
        .and_then(|s16| s16.checked_mul(tile_row_idx))
    else {
        return false;
    };
    let eff_row_end8 = row_end8.min(ih8);
    if eff_row_end8 <= row_start8 {
        return true;
    }
    let rows = eff_row_end8 - row_start8;
    // Bölge temizliği `(row_start8 & 15) + (rows - 1)` satırına kadar, projeksiyon ise
    // `(py & 15)` satırına yazar. `row_start8 & 15 == 0` olduğunda ikisi de `< rows`
    // satırındadır; genel durumda maksimum satır indeksi `max((row_start8 & 15) + rows - 1, 15)`
    // veya `row_start8 & 15 == 0` iken `rows - 1`.
    let max_win_row = if (row_start8 & 15) == 0 {
        rows - 1
    } else {
        ((row_start8 & 15) + rows - 1).max(15)
    };
    if col_end8 > 0 {
        let Some(proj_need) = max_win_row
            .checked_mul(stride)
            .and_then(|v| v.checked_add(col_end8))
            .and_then(|v| base.checked_add(v))
        else {
            return false;
        };
        if rp_proj_len < proj_need {
            return false;
        }
    }
    let col_start8i = col_start8.saturating_sub(8);
    let col_end8i = (col_end8.saturating_add(8)).min(iw8);
    for n in 0..plan.n_mfmvs {
        let r_idx = usize::from(plan.mfmv_ref[n]);
        if r_idx >= 7 {
            return false;
        }
        let r2c = plan.mfmv_ref2cur[n];
        if r2c == crate::decode::refmvs_tmvs::INVALID_REF2CUR {
            continue;
        }
        if !(-31..=31).contains(&r2c) {
            return false;
        }
        let mut i = 0;
        while i < 7 {
            if plan.mfmv_ref2ref[n][i] > 31 {
                return false;
            }
            i += 1;
        }
        if let Some(r) = refs[r_idx]
            && col_start8i < col_end8i
        {
            let Some(ref_need) = (eff_row_end8 - 1)
                .checked_mul(stride)
                .and_then(|v| v.checked_add(col_end8i))
            else {
                return false;
            };
            if r.len() < ref_need {
                return false;
            }
        }
    }
    true
}

/// P46: `splat_mv` SIMD girişinin O(1) sözleşmesi:
/// - `bw4 == 0 || bh4 == 0` ise boş çağrı (geçerli)
/// - `bx4 + bw4 <= iw4`
/// - `(by4 + bh4 - 1) * iw4 + bx4 + bw4 <= grid_len`
#[must_use]
#[inline]
pub fn splat_mv_sozlesmesi(
    grid_len: usize,
    iw4: usize,
    bx4: usize,
    by4: usize,
    bw4: usize,
    bh4: usize,
) -> bool {
    if bw4 == 0 || bh4 == 0 {
        return true;
    }
    let Some(x_end) = bx4.checked_add(bw4) else {
        return false;
    };
    if x_end > iw4 {
        return false;
    }
    let Some(last_y) = by4.checked_add(bh4 - 1) else {
        return false;
    };
    let Some(need) = last_y.checked_mul(iw4).and_then(|v| v.checked_add(x_end)) else {
        return false;
    };
    grid_len >= need
}

/// P46: Sözleşme dışı `save_tmvs` girdisinde skaler devir sayacını artırır ve
/// panik-güvenli skaler gövdeye devreder.
#[cold]
#[allow(clippy::too_many_arguments)]
pub fn save_tmvs_skaler_devret(
    rp: &mut [TemporalBlock],
    stride: usize,
    grid: &[RefBlock],
    iw4: usize,
    ref_sign: &[u8; 7],
    col_end8: usize,
    row_end8: usize,
    col_start8: usize,
    row_start8: usize,
) {
    refmvs_skaler_devri_say();
    if save_tmvs_sozlesmesi(
        rp.len(),
        stride,
        grid.len(),
        iw4,
        col_end8,
        row_end8,
        col_start8,
        row_start8,
    ) {
        scalar::save_tmvs(
            rp, stride, grid, iw4, ref_sign, col_end8, row_end8, col_start8, row_start8,
        );
    }
}

/// P46: Sözleşme dışı `load_tmvs` girdisinde skaler devir sayacını artırır ve
/// önkoşul sağlanıyorsa skaler gövdeye devreder.
#[cold]
#[allow(clippy::too_many_arguments)]
pub fn load_tmvs_skaler_devret(
    rp_proj: &mut [TemporalBlock],
    stride: usize,
    iw8: usize,
    ih8: usize,
    plan: &MfmvPlan,
    refs: &[Option<&[TemporalBlock]>; 7],
    tile_row_idx: usize,
    col_start8: usize,
    col_end8: usize,
    row_start8: usize,
    row_end8: usize,
) {
    refmvs_skaler_devri_say();
    if load_tmvs_sozlesmesi(
        rp_proj.len(),
        stride,
        iw8,
        ih8,
        plan,
        refs,
        tile_row_idx,
        col_start8,
        col_end8,
        row_start8,
        row_end8,
    ) {
        scalar::load_tmvs(
            rp_proj,
            stride,
            iw8,
            ih8,
            plan,
            refs,
            tile_row_idx,
            col_start8,
            col_end8,
            row_start8,
            row_end8,
        );
    }
}

/// P46: Sözleşme dışı `splat_mv` girdisinde skaler devir sayacını artırır ve
/// önkoşul sağlanıyorsa skaler gövdeye devreder.
#[cold]
pub fn splat_mv_skaler_devret(
    grid: &mut [RefBlock],
    iw4: usize,
    rmv: RefBlock,
    bx4: usize,
    by4: usize,
    bw4: usize,
    bh4: usize,
) {
    refmvs_skaler_devri_say();
    if splat_mv_sozlesmesi(grid.len(), iw4, bx4, by4, bw4, bh4) {
        scalar::splat_mv(grid, iw4, rmv, bx4, by4, bw4, bh4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p46_refmvs_sozlesmesi_ve_paniksiz_skaler_devir() {
        assert!(splat_mv_sozlesmesi(64, 8, 0, 0, 4, 4));
        assert!(splat_mv_sozlesmesi(64, 8, 4, 4, 4, 4));
        assert!(!splat_mv_sozlesmesi(63, 8, 4, 4, 4, 4));
        assert!(!splat_mv_sozlesmesi(64, 8, 6, 0, 4, 4));

        assert!(save_tmvs_sozlesmesi(64, 8, 16 * 16, 16, 8, 8, 0, 0));
        assert!(!save_tmvs_sozlesmesi(63, 8, 16 * 16, 16, 8, 8, 0, 0));
        assert!(!save_tmvs_sozlesmesi(64, 8, 16 * 15, 16, 8, 8, 0, 0));
        assert!(!save_tmvs_sozlesmesi(64, 8, 16 * 16, 0, 8, 8, 0, 0));

        let plan = MfmvPlan {
            n_mfmvs: 1,
            mfmv_ref: [0, 0, 0],
            mfmv_ref2cur: [4, 0, 0],
            mfmv_ref2ref: [[2; 7]; 3],
        };
        let ref_buf = vec![TemporalBlock::default(); 8 * 8];
        let refs: [Option<&[TemporalBlock]>; 7] =
            [Some(&ref_buf), None, None, None, None, None, None];
        assert!(load_tmvs_sozlesmesi(
            16 * 8,
            8,
            8,
            8,
            &plan,
            &refs,
            0,
            0,
            8,
            0,
            8
        ));
        assert!(!load_tmvs_sozlesmesi(
            7 * 8 + 7,
            8,
            8,
            8,
            &plan,
            &refs,
            0,
            0,
            8,
            0,
            8
        ));
        let kisa_ref = vec![TemporalBlock::default(); 7 * 8 + 7];
        let kisa_refs: [Option<&[TemporalBlock]>; 7] =
            [Some(&kisa_ref), None, None, None, None, None, None];
        assert!(!load_tmvs_sozlesmesi(
            16 * 8,
            8,
            8,
            8,
            &plan,
            &kisa_refs,
            0,
            0,
            8,
            0,
            8
        ));

        let once = refmvs_skaler_devir_sayisi();
        let mut kisa_grid = [RefBlock::default(); 4];
        splat_mv_skaler_devret(&mut kisa_grid, 8, RefBlock::default(), 0, 0, 4, 4);
        assert!(refmvs_skaler_devir_sayisi() > once);
    }
}
