//! Refmvs DSP skaler gövdeleri — dav1d `src/refmvs.c` (`save_tmvs_c`,
//! `load_tmvs_c`, `splat_mv_c`).
//!
//! Sözleşme: [`crate::dsp::refmvs`]. Gövdeler F3c'de
//! `crate::decode::refmvs_tmvs`'ten değiştirilmeden taşındı; veri türleri ve
//! [`mv_projection`] orada kalır. dav1d'de bu tablo bit derinliğinden
//! bağımsızdır (`dav1d_refmvs_dsp_init`, `src/refmvs.c:953`).

#![forbid(unsafe_code)]

use crate::decode::refmvs::{Mv, RefBlock};
use crate::decode::refmvs_tmvs::{INVALID_REF2CUR, MfmvPlan, TemporalBlock, mv_projection};
use crate::tables::decode as t;

/// `save_tmvs_c` (`refmvs.c:792-830`).
///
/// `grid`, `iw4` sütunlu 4×4 ızgarasıdır (kare geneli, mutlak koordinat).
/// `rp`, `stride` sütunlu 8×8 hedefidir; `(row_start8..row_end8) ×
/// (col_start8..col_end8)` bölgesi yazılır.
#[allow(clippy::too_many_arguments)]
pub fn save_tmvs(
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
    for y in row_start8..row_end8 {
        // `refmvs.c:795` + `dav1d_refmvs_save_tmvs` (`refmvs.h:157-172`):
        // `rr`, `row_start8*2 + 1` mutlak 4×4 satırından başlar; `(y & 15)*2`
        // indeksi mutlak satır `2y + 1`'e karşılık gelir (halka tampon
        // 16 satırlık penceredir ve `row_start8` 16'nın katıdır).
        let row4 = y * 2 + 1;
        let base = row4 * iw4;
        let rp_row = (y - row_start8) * stride;
        let mut x = col_start8;
        while x < col_end8 {
            let col4 = x * 2 + 1;
            let idx = base + col4;
            let cand = if idx < grid.len() {
                grid[idx]
            } else {
                RefBlock::default()
            };
            let bw8 = (usize::from(t::BLOCK_DIMENSIONS[usize::from(cand.bs)][0]) + 1) >> 1;
            let bw8 = bw8.max(1);
            // `refmvs.c:803-812`: önce `ref[1]` (varsa), sonra `ref[0]`.
            let pick = [1usize, 0].into_iter().find(|&n| {
                let r = cand.ref_pair[n];
                r > 0
                    && ref_sign[usize::from(r as u8) - 1] != 0
                    && (i32::from(cand.mv[n].y).abs() | i32::from(cand.mv[n].x).abs()) < 4096
            });
            // dav1d `refmvs.c:806-828`: `bw8` kayıt **tamamen** yazılır;
            // satır sonunu aşan yazma düz bellek indeksinde bir sonraki
            // satıra taşar (C ile aynı). Blok hizalaması nedeniyle
            // `round_up(col_end8, bw8) <= stride` olduğundan taşma
            // gerçekleşmez; yine de panik yok (sınır denetimli).
            match pick {
                Some(n_ref) => {
                    let tmpl = TemporalBlock {
                        mv: cand.mv[n_ref],
                        ref_: cand.ref_pair[n_ref] as u8,
                    };
                    for k in 0..bw8 {
                        if let Some(slot) = rp.get_mut(rp_row + x + k) {
                            *slot = tmpl;
                        }
                    }
                }
                None => {
                    for k in 0..bw8 {
                        if let Some(slot) = rp.get_mut(rp_row + x + k) {
                            *slot = TemporalBlock::default();
                        }
                    }
                }
            }
            x += bw8;
        }
    }
}

/// `load_tmvs_c` (`refmvs.c:719-790`).
///
/// `refs[i]`, `i` indeksli referansın kaydedilmiş 8×8 MV ızgarasıdır
/// (dav1d `rf->rp_ref[i]`; boyut uyuşmazsa `None`).
#[allow(clippy::too_many_arguments)]
pub fn load_tmvs(
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
    debug_assert!(row_start8 <= row_end8);
    debug_assert!(row_end8 - row_start8 <= 16);
    let row_end8 = row_end8.min(ih8);
    let col_start8i = col_start8.saturating_sub(8);
    let col_end8i = (col_end8 + 8).min(iw8);
    let base = 16 * stride * tile_row_idx;
    let row_off = (row_start8 & 15) * stride;

    // `refmvs.c:735-741`: bölgeyi geçersiz kıl.
    for y in row_start8..row_end8 {
        let off = base + row_off + (y - row_start8) * stride;
        for x in col_start8..col_end8 {
            if off + x < rp_proj.len() {
                rp_proj[off + x].mv = Mv::INVALID;
            }
        }
    }

    for n in 0..plan.n_mfmvs.min(3) {
        let ref2cur = plan.mfmv_ref2cur[n];
        if ref2cur == INVALID_REF2CUR {
            continue;
        }
        let Some(r) = refs[usize::from(plan.mfmv_ref[n])] else {
            continue;
        };
        let ref_sign = i32::from(plan.mfmv_ref[n]) - 4; // `ref - 4`
        let ref2cur = i32::from(ref2cur);
        for y in row_start8..row_end8 {
            let y_sb_align = y & !7;
            let y_proj_start = y_sb_align.max(row_start8);
            let y_proj_end = (y_sb_align + 8).min(row_end8);
            let r_row = y * stride;
            let mut x = col_start8i;
            while x < col_end8i {
                let rb = if r_row + x < r.len() {
                    r[r_row + x]
                } else {
                    TemporalBlock::default()
                };
                let b_ref = rb.ref_;
                if b_ref == 0 {
                    x += 1;
                    continue;
                }
                let r2r = plan.mfmv_ref2ref[n][usize::from(b_ref) - 1];
                if r2r == 0 {
                    x += 1;
                    continue;
                }
                let b_mv = rb.mv;
                let offset = mv_projection(b_mv, ref2cur, i32::from(r2r));
                let pos_y = y as i32
                    + apply_sign(i32::from(offset.y).abs() >> 6, offset.y ^ (ref_sign as i16));
                // Koşu: aynı `(ref, mv)` kayıtları birlikte işlenir
                // (`refmvs.c:766-786`) — `pos_x` koşu boyunca artar.
                let mut run = 1usize;
                while x + run < col_end8i {
                    let nx = r_row + x + run;
                    if nx >= r.len() || r[nx].ref_ != b_ref || r[nx].mv != b_mv {
                        break;
                    }
                    run += 1;
                }
                let pos_x0 = x as i32
                    + apply_sign(i32::from(offset.x).abs() >> 6, offset.x ^ (ref_sign as i16));
                let py = pos_y as usize;
                if py >= y_proj_start && py < y_proj_end {
                    let pos = base + (py & 15) * stride;
                    for k in 0..run {
                        let xx = x + k;
                        let x_sb_align = xx & !7;
                        let px = pos_x0 + k as i32;
                        let lo = x_sb_align.saturating_sub(8).max(col_start8) as i32;
                        let hi = ((x_sb_align + 16).min(col_end8)) as i32;
                        if px >= lo && px < hi {
                            let o = pos + px as usize;
                            if o < rp_proj.len() {
                                rp_proj[o] = TemporalBlock {
                                    mv: b_mv,
                                    ref_: r2r,
                                };
                            }
                        }
                    }
                }
                x += run;
            }
        }
    }
}

/// `apply_sign` (`common/intops.h`).
#[inline]
fn apply_sign(v: i32, s: i16) -> i32 {
    if s < 0 { -v } else { v }
}

/// `splat_mv_c` (`refmvs.c:933-941`) — bloğun kaydını ızgaraya yazar.
///
/// `iw4` ızgara genişliği (4×4 birimi); `(bx4, by4)` mutlak koordinattır.
pub fn splat_mv(
    grid: &mut [RefBlock],
    iw4: usize,
    rmv: RefBlock,
    bx4: usize,
    by4: usize,
    bw4: usize,
    bh4: usize,
) {
    for y in by4..by4 + bh4 {
        let row = y * iw4;
        for x in bx4..bx4 + bw4 {
            if row + x < grid.len() {
                grid[row + x] = rmv;
            }
        }
    }
}
