//! Deblocking (loop filter) — S2c.1.
//!
//! dav1d `src/loopfilter_tmpl.c`, `src/lf_mask.c` ve `src/lf_apply_tmpl.c`
//! C yolunun birebir karşılığı. Kare çözüldükten sonra **SB satırı sırasıyla**
//! önce dikey (sütun) sonra yatay (satır) kenarlar süzülür; renk düzeni
//! 4:0:0/4:2:0/4:2:2/4:4:4 ve 8/10/12 bit desteklenir.
//!
//! ## Kaynak eşlemesi (dav1d @ `99f3354`, salt-okunur)
//!
//! | Rust | dav1d |
//! |---|---|
//! | `LfFrame::calc_eih` | `lf_mask.c:382` (`dav1d_calc_eih`) |
//! | [`calc_lf_values`] | `lf_mask.c:401-470` |
//! | [`LfFrame::create_mask_intra`] | `lf_mask.c:274` |
//! | [`LfFrame::create_mask_inter`] | `lf_mask.c:327` |
//! | [`LfFrame::apply`] | `lf_apply_tmpl.c:317-520` |
//! | `dsp::scalar::lf::loop_filter` (F3c: yuva gövdesi, [`crate::dsp::lf`]) | `loopfilter_tmpl.c:37-163` |
//!
//! Maske bitleri dav1d'de 32×32 4×4'lük SB başına üç filtre boyu bankasına
//! yazılır; burada **kare geneli** tek bayt (filtre indeksi + 1, 0 = yok)
//! olarak tutulur. Bit bankası → indeks dönüşümü `min(2, …)` olduğundan ve
//! tile sınırı düzeltmesi bitleri temizleyip tek banka yazdığından bu
//! gösterim birebir eşdeğerdir.

#![forbid(unsafe_code)]
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_div_ceil,
    clippy::manual_memcpy
)]

use crate::bitstream::frame_header::{FrameHeader, MAX_SEGMENTS};
use crate::tables::decode as t;
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::sync::OnceLock;

pub(crate) use crate::dsp::lf::LimLut;
use crate::dsp::lf::LoopFilterDsp;

/// `calc_lf_value` çıktısı: `[düzlem/yön 0=y-dikey,1=y-yatay,2=u,3=v][ref][mode]`.
pub(crate) type LfLvl = [[[u8; 2]; 8]; 4];

/// Segment başına LF seviye tablosu (`ts->lflvl[8]`).
pub(crate) type LfLvlAll = [LfLvl; MAX_SEGMENTS];

/// `dav1d_calc_eih` (`lf_mask.c:382-399`).
pub(crate) use crate::dsp::lf::calc_eih;

/// `calc_lf_value` (`lf_mask.c:401-423`).
fn calc_lf_value(
    out: &mut [[u8; 2]; 8],
    base_lvl: i32,
    lf_delta: i32,
    seg_delta: i32,
    mr: Option<&crate::bitstream::frame_header::LoopfilterModeRefDeltas>,
) {
    let base = (base_lvl + lf_delta).clamp(0, 63) + seg_delta;
    let base = base.clamp(0, 63);
    match mr {
        None => {
            for e in out.iter_mut() {
                *e = [base as u8; 2];
            }
        }
        Some(d) => {
            let sh = i32::from(base >= 32);
            let v = base + i32::from(d.ref_delta[0]) * (1 << sh);
            out[0] = [v.clamp(0, 63) as u8; 2];
            for r in 1..8 {
                for m in 0..2 {
                    let delta = i32::from(d.mode_delta[m]) + i32::from(d.ref_delta[r]);
                    out[r][m] = (base + delta * (1 << sh)).clamp(0, 63) as u8;
                }
            }
        }
    }
}

/// `calc_lf_value_chroma` (`lf_mask.c:424-433`).
fn calc_lf_value_chroma(
    out: &mut [[u8; 2]; 8],
    base_lvl: i32,
    lf_delta: i32,
    seg_delta: i32,
    mr: Option<&crate::bitstream::frame_header::LoopfilterModeRefDeltas>,
) {
    if base_lvl == 0 {
        for e in out.iter_mut() {
            *e = [0; 2];
        }
    } else {
        calc_lf_value(out, base_lvl, lf_delta, seg_delta, mr);
    }
}

/// `dav1d_calc_lf_values` (`lf_mask.c:435-472`).
pub(crate) fn calc_lf_values(hdr: &FrameHeader, lf_delta: [i8; 4]) -> LfLvlAll {
    let mut all = [[[[0u8; 2]; 8]; 4]; MAX_SEGMENTS];
    if hdr.loopfilter.level_y[0] == 0 && hdr.loopfilter.level_y[1] == 0 {
        return all;
    }
    let n_seg = if hdr.segmentation.enabled {
        MAX_SEGMENTS
    } else {
        1
    };
    #[cfg(feature = "std")]
    {
        static LF_CALC: OnceLock<bool> = OnceLock::new();
        if *LF_CALC.get_or_init(|| std::env::var_os("AV1OXIDE_LF_CALC").is_some()) {
            crate::trace_eprintln!(
                "CALC en={} nseg={} lv={:?} mr={} u={} v={}",
                hdr.segmentation.enabled,
                n_seg,
                hdr.segmentation.seg_data.d[0],
                hdr.loopfilter.mode_ref_delta_enabled,
                hdr.loopfilter.level_u,
                hdr.loopfilter.level_v
            );
        }
    }
    let mr = if hdr.loopfilter.mode_ref_delta_enabled {
        Some(&hdr.loopfilter.mode_ref_deltas)
    } else {
        None
    };
    let multi = hdr.delta.lf_multi;
    let d = |i: usize| i32::from(lf_delta[usize::from(multi) * i]);
    for s in 0..n_seg {
        let mut out = [[[0u8; 2]; 8]; 4];
        let segd = if hdr.segmentation.enabled {
            Some(&hdr.segmentation.seg_data.d[s])
        } else {
            None
        };
        calc_lf_value(
            &mut out[0],
            i32::from(hdr.loopfilter.level_y[0]),
            d(0),
            segd.map_or(0, |s| i32::from(s.delta_lf_y_v)),
            mr,
        );
        calc_lf_value(
            &mut out[1],
            i32::from(hdr.loopfilter.level_y[1]),
            if multi { d(1) } else { d(0) },
            segd.map_or(0, |s| i32::from(s.delta_lf_y_h)),
            mr,
        );
        calc_lf_value_chroma(
            &mut out[2],
            i32::from(hdr.loopfilter.level_u),
            if multi { d(2) } else { d(0) },
            segd.map_or(0, |s| i32::from(s.delta_lf_u)),
            mr,
        );
        calc_lf_value_chroma(
            &mut out[3],
            i32::from(hdr.loopfilter.level_v),
            if multi { d(3) } else { d(0) },
            segd.map_or(0, |s| i32::from(s.delta_lf_v)),
            mr,
        );
        // Segment seçimi yalnız `segmentation.enabled` iken döner; döngü
        // gövdesi `out`u yerinde günceller. `n_seg` 8 ise segment verisi
        // `segd`den gelir; 1 ise yalnız 0. indeks geçerlidir.
        all[s] = out;
    }
    all
}

/// `decomp_tx` için 4×4 tx haritası (`txsz, step` iki yön).
#[derive(Clone, Copy, Default)]
struct TxMap {
    tx0: [[u8; 32]; 32],
    step0: [[u8; 32]; 32],
    tx1: [[u8; 32]; 32],
    step1: [[u8; 32]; 32],
}

/// `decomp_tx` (`lf_mask.c:39-77`).
fn decomp_tx(
    map: &mut TxMap,
    from: usize,
    depth: usize,
    y0: usize,
    x0: usize,
    y_off: usize,
    x_off: usize,
    tx_masks: &[u16; 2],
) {
    let td = &t::TXFM_DIMENSIONS[from];
    let is_split = from != 0 && depth <= 1 && ((tx_masks[depth] >> (y_off * 4 + x_off)) & 1) != 0;
    if is_split {
        let sub = usize::from(td[6]);
        let htw4 = usize::from(td[0]) >> 1;
        let hth4 = usize::from(td[1]) >> 1;
        decomp_tx(map, sub, depth + 1, y0, x0, y_off * 2, x_off * 2, tx_masks);
        if td[0] >= td[1] {
            decomp_tx(
                map,
                sub,
                depth + 1,
                y0,
                x0 + htw4,
                y_off * 2,
                x_off * 2 + 1,
                tx_masks,
            );
        }
        if td[1] >= td[0] {
            decomp_tx(
                map,
                sub,
                depth + 1,
                y0 + hth4,
                x0,
                y_off * 2 + 1,
                x_off * 2,
                tx_masks,
            );
            if td[0] >= td[1] {
                decomp_tx(
                    map,
                    sub,
                    depth + 1,
                    y0 + hth4,
                    x0 + htw4,
                    y_off * 2 + 1,
                    x_off * 2 + 1,
                    tx_masks,
                );
            }
        }
    } else {
        let lw = td[2].min(2);
        let lh = td[3].min(2);
        let n = 1usize << td[2];
        for y in 0..usize::from(td[1]) {
            for x in 0..n {
                map.tx0[y0 + y][x0 + x] = lw;
                map.tx1[y0 + y][x0 + x] = lh;
            }
            map.step0[y0 + y][x0] = td[0];
        }
        for x in 0..n {
            map.step1[y0][x0 + x] = td[1];
        }
    }
}

/// Deblocking kare durumu: seviye önbelleği, kenar maskeleri, tile bağlamları.
pub(crate) struct LfFrame {
    pub bw: usize,
    pub bh: usize,
    pub cw: usize,
    pub ss_hor: usize,
    pub ss_ver: usize,
    pub x_base: usize,
    pub y_base: usize,
    pub stride_y: usize,
    pub stride_uv: usize,
    pub sb_step: usize,
    pub sb128w: usize,
    pub sbh: usize,
    pub align_h: usize,
    pub n_cols: usize,
    pub n_rows: usize,
    /// Karo-yerel daraltılmış örnek mi (`fork_for_tile`)?
    pub tile_local: bool,
    /// `level[y][x] = [y_dikey, y_yatay, u, v]` (luma 4×4 ızgarası).
    pub level: Vec<[u8; 4]>,
    /// Sütun (dikey) kenar maskeleri; değer = filtre indeksi + 1, 0 = yok.
    pub mask_y_col: Vec<u8>,
    /// Satır (yatay) kenar maskeleri.
    pub mask_y_row: Vec<u8>,
    /// Kroma sütun (dikey) kenar maskeleri.
    pub mask_uv_col: Vec<u8>,
    /// Kroma satır (yatay) kenar maskeleri.
    pub mask_uv_row: Vec<u8>,
    /// `tx_lpf_right_edge[0]`, tile sütunu başına `align_h` bayt.
    pub tx_right_y: Vec<u8>,
    /// `tx_lpf_right_edge[1]`, tile sütunu başına `align_h >> ss_ver` bayt.
    pub tx_right_uv: Vec<u8>,
    /// Tile satırı başı için önceki tile satırının `above.tx_lpf_y` kopyası.
    pub row_lpf_y: Vec<[u8; 32]>,
    /// Tile satırı başı için `above.tx_lpf_uv` kopyası.
    pub row_lpf_uv: Vec<[u8; 32]>,
    pub lim: LimLut,
    /// Tanılama: kare `frame_offset`ı.
    trace_off: u32,
    /// Tanılama: loop filter seviyeleri.
    trace_lv: [u8; 4],
}

impl LfFrame {
    /// Kare için maske/seviye alanlarını ayırır (`dav1d_decode_frame_init`).
    #[cfg(test)]
    pub fn new(
        hdr: &FrameHeader,
        bw: usize,
        bh: usize,
        ss_hor: usize,
        ss_ver: usize,
        sb128: bool,
        butce: &crate::frame::butce::Butce,
    ) -> crate::error::Result<Self> {
        Self::new_with_reuse(hdr, bw, bh, ss_hor, ss_ver, sb128, None, butce)
    }

    /// Havuzdan geri dönüştürülmüş `old` varsa kapasiteleri eşleştiğinde yeniden
    /// kullanarak `LfFrame` kurar (`F8b-1`).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_reuse(
        hdr: &FrameHeader,
        bw: usize,
        bh: usize,
        ss_hor: usize,
        ss_ver: usize,
        sb128: bool,
        old: Option<Self>,
        butce: &crate::frame::butce::Butce,
    ) -> crate::error::Result<Self> {
        let cw = (bw + ss_hor) >> ss_hor;
        let ch = (bh + ss_ver) >> ss_ver;
        let sb_step = if sb128 { 32 } else { 16 };
        let align_h = (bh + 31) & !31;
        let n_cols = usize::from(hdr.tiling.cols);
        let n_rows = usize::from(hdr.tiling.rows);
        let sb128w = (bw + 31) >> 5;
        let (old_level, old_myc, old_myr, old_muc, old_mur, old_try, old_truv, old_rly, old_rluv) =
            match old {
                Some(o) => (
                    o.level,
                    o.mask_y_col,
                    o.mask_y_row,
                    o.mask_uv_col,
                    o.mask_uv_row,
                    o.tx_right_y,
                    o.tx_right_uv,
                    o.row_lpf_y,
                    o.row_lpf_uv,
                ),
                None => (
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ),
            };
        Ok(Self {
            bw,
            bh,
            cw,
            ss_hor,
            ss_ver,
            x_base: 0,
            y_base: 0,
            stride_y: bw,
            stride_uv: cw,
            sb_step,
            sb128w,
            sbh: bh.div_ceil(sb_step),
            align_h,
            n_cols,
            n_rows,
            tile_local: false,
            level: butce.reuse_vec(old_level, bw * bh, [0u8; 4])?,
            mask_y_col: butce.reuse_vec(old_myc, bw * bh, 0u8)?,
            mask_y_row: butce.reuse_vec(old_myr, bw * bh, 0u8)?,
            mask_uv_col: butce.reuse_vec(old_muc, cw * ch, 0u8)?,
            mask_uv_row: butce.reuse_vec(old_mur, cw * ch, 0u8)?,
            tx_right_y: butce.reuse_vec(old_try, n_cols * align_h, 0u8)?,
            tx_right_uv: butce.reuse_vec(old_truv, n_cols * (align_h >> ss_ver), 0u8)?,
            row_lpf_y: butce.reuse_vec(old_rly, n_rows * sb128w, [2u8; 32])?,
            row_lpf_uv: butce.reuse_vec(old_rluv, n_rows * sb128w, [1u8; 32])?,
            lim: calc_eih(hdr.loopfilter.sharpness),
            trace_off: u32::from(hdr.frame_offset),
            trace_lv: [
                hdr.loopfilter.level_y[0],
                hdr.loopfilter.level_y[1],
                hdr.loopfilter.level_u,
                hdr.loopfilter.level_v,
            ],
        })
    }

    /// Paralel işçi başlatması için tahsissiz (0 bayt) üst-veri şablonu.
    #[must_use]
    pub(crate) fn meta_template(&self) -> Self {
        Self {
            bw: self.bw,
            bh: self.bh,
            cw: self.cw,
            ss_hor: self.ss_hor,
            ss_ver: self.ss_ver,
            x_base: self.x_base,
            y_base: self.y_base,
            stride_y: self.stride_y,
            stride_uv: self.stride_uv,
            sb_step: self.sb_step,
            sb128w: self.sb128w,
            sbh: self.sbh,
            align_h: self.align_h,
            n_cols: self.n_cols,
            n_rows: self.n_rows,
            tile_local: false,
            level: Vec::new(),
            mask_y_col: Vec::new(),
            mask_y_row: Vec::new(),
            mask_uv_col: Vec::new(),
            mask_uv_row: Vec::new(),
            tx_right_y: Vec::new(),
            tx_right_uv: Vec::new(),
            row_lpf_y: Vec::new(),
            row_lpf_uv: Vec::new(),
            lim: self.lim,
            trace_off: self.trace_off,
            trace_lv: self.trace_lv,
        }
    }

    /// Paralel tile çözümü için yalnız tile dikdörtgeni (`[row_start..row_end) × [col_start..col_end)`)
    /// boyutunda geçici `LfFrame` oluşturur (`F8b-2`).
    ///
    /// Kare düzeyindeki `LfFrame` zaten `Butce`'de sayılmıştır; tile yerel
    /// maske ve seviye tabloları tile tamamlanınca `merge_tile_from` ile
    /// kare `LfFrame` üzerine kopyalanıp düşürülür.
    pub fn fork_for_tile(
        &self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> crate::error::Result<Self> {
        use crate::frame::butce::dolu_vec;
        let x_base = col_start.min(self.bw) & !1;
        let y_base = row_start.min(self.bh) & !1;
        let stride_y = col_end.min(self.bw).saturating_sub(x_base).max(32);
        let rows_y = ((row_end.min(self.bh).saturating_sub(y_base).max(32) + 31) & !31).max(32);
        let stride_uv = (stride_y + self.ss_hor) >> self.ss_hor;
        let rows_uv = ((rows_y + self.ss_ver) >> self.ss_ver).max(16);
        Ok(Self {
            bw: self.bw,
            bh: self.bh,
            cw: self.cw,
            ss_hor: self.ss_hor,
            ss_ver: self.ss_ver,
            x_base,
            y_base,
            stride_y,
            stride_uv,
            sb_step: self.sb_step,
            sb128w: self.sb128w,
            sbh: self.sbh,
            align_h: self.align_h,
            n_cols: self.n_cols,
            n_rows: self.n_rows,
            tile_local: true,
            level: dolu_vec(stride_y * rows_y, [0u8; 4])?,
            mask_y_col: dolu_vec(stride_y * rows_y, 0u8)?,
            mask_y_row: dolu_vec(stride_y * rows_y, 0u8)?,
            mask_uv_col: dolu_vec(stride_uv * rows_uv, 0u8)?,
            mask_uv_row: dolu_vec(stride_uv * rows_uv, 0u8)?,
            tx_right_y: dolu_vec(rows_y, 0u8)?,
            tx_right_uv: dolu_vec(rows_uv, 0u8)?,
            row_lpf_y: dolu_vec(self.sb128w, [2u8; 32])?,
            row_lpf_uv: dolu_vec(self.sb128w, [1u8; 32])?,
            lim: self.lim,
            trace_off: self.trace_off,
            trace_lv: self.trace_lv,
        })
    }

    /// Tile `(tile_row, tile_col)` yerel `other` maske ve seviye alanlarını
    /// (`[row_start..row_end) × [col_start..col_end)`) kare `LfFrame` içine birleştirir.
    #[allow(clippy::too_many_arguments)]
    pub fn merge_tile_from(
        &mut self,
        other: &Self,
        tile_row: usize,
        tile_col: usize,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) {
        let y0 = row_start.min(self.bh);
        let y1 = row_end.min(self.bh);
        let x0 = col_start.min(self.bw);
        let x1 = col_end.min(self.bw);
        if x0 < x1 {
            let w = x1 - x0;
            for gy in y0..y1 {
                let row_off = gy * self.bw;
                let src_off = (gy - other.y_base) * other.stride_y + (x0 - other.x_base);
                for dx in 0..w {
                    self.level[row_off + x0 + dx][0] = other.level[src_off + dx][0];
                    self.level[row_off + x0 + dx][1] = other.level[src_off + dx][1];
                }
                self.mask_y_col[row_off + x0..row_off + x1]
                    .copy_from_slice(&other.mask_y_col[src_off..src_off + w]);
                self.mask_y_row[row_off + x0..row_off + x1]
                    .copy_from_slice(&other.mask_y_row[src_off..src_off + w]);
            }
        }
        let ch = (self.bh + self.ss_ver) >> self.ss_ver;
        let cy0 = (y0 >> self.ss_ver).min(ch);
        let cy1 = ((y1 + self.ss_ver) >> self.ss_ver).min(ch);
        let cx0 = (x0 >> self.ss_hor).min(self.cw);
        let cx1 = ((x1 + self.ss_hor) >> self.ss_hor).min(self.cw);
        if cx0 < cx1 {
            let cw_len = cx1 - cx0;
            let other_cy_base = other.y_base >> self.ss_ver;
            let other_cx_base = other.x_base >> self.ss_hor;
            for cgy in cy0..cy1 {
                let lvl_off = cgy * self.bw;
                let src_lvl_off = (cgy - other_cy_base) * other.stride_y + (cx0 - other_cx_base);
                for dx in 0..cw_len {
                    self.level[lvl_off + cx0 + dx][2] = other.level[src_lvl_off + dx][2];
                    self.level[lvl_off + cx0 + dx][3] = other.level[src_lvl_off + dx][3];
                }
                let uv_off = cgy * self.cw;
                let src_uv_off = (cgy - other_cy_base) * other.stride_uv + (cx0 - other_cx_base);
                self.mask_uv_col[uv_off + cx0..uv_off + cx1]
                    .copy_from_slice(&other.mask_uv_col[src_uv_off..src_uv_off + cw_len]);
                self.mask_uv_row[uv_off + cx0..uv_off + cx1]
                    .copy_from_slice(&other.mask_uv_row[src_uv_off..src_uv_off + cw_len]);
            }
        }
        // `tx_right_y` ve `tx_right_uv`: bu `tile_col` ve `[y0..y1)` satırları.
        if tile_col < self.n_cols && y0 < y1 {
            let base_y = tile_col * self.align_h;
            let ry1 = y1.min(self.align_h);
            if y0 < ry1 {
                let len_y = ry1 - y0;
                let src_off_y = if other.tile_local {
                    y0 - other.y_base
                } else {
                    base_y + y0
                };
                self.tx_right_y[base_y + y0..base_y + ry1]
                    .copy_from_slice(&other.tx_right_y[src_off_y..src_off_y + len_y]);
            }
            let align_uv = self.align_h >> self.ss_ver;
            let base_uv = tile_col * align_uv;
            let rcy0 = (y0 >> self.ss_ver).min(align_uv);
            let rcy1 = ((y1 + self.ss_ver) >> self.ss_ver).min(align_uv);
            if rcy0 < rcy1 {
                let len_uv = rcy1 - rcy0;
                let src_off_uv = if other.tile_local {
                    rcy0 - (other.y_base >> self.ss_ver)
                } else {
                    base_uv + rcy0
                };
                self.tx_right_uv[base_uv + rcy0..base_uv + rcy1]
                    .copy_from_slice(&other.tx_right_uv[src_off_uv..src_off_uv + len_uv]);
            }
        }
        // `row_lpf_y` ve `row_lpf_uv`: bu tile'ın bittiği alt sınır (`tile_row` yuvası,
        // `tile_row + 1` satırı tarafından `(tr - 1) * sb128w` ile okunur).
        if tile_row + 1 < self.n_rows && x0 < x1 {
            let row_idx = tile_row;
            let sb_x0 = x0 >> 5;
            let sb_x1 = x1.div_ceil(32).min(self.sb128w);
            for sbx in sb_x0..sb_x1 {
                let lx0 = x0.saturating_sub(sbx * 32).min(32);
                let lx1 = x1.saturating_sub(sbx * 32).min(32);
                let dst_slot = row_idx * self.sb128w + sbx;
                let src_slot = if other.tile_local { sbx } else { dst_slot };
                if lx0 < lx1 {
                    self.row_lpf_y[dst_slot][lx0..lx1]
                        .copy_from_slice(&other.row_lpf_y[src_slot][lx0..lx1]);
                }
                let uv_x0 = (x0 >> self.ss_hor)
                    .saturating_sub(sbx * (32 >> self.ss_hor))
                    .min(32);
                let uv_x1 = ((x1 + self.ss_hor) >> self.ss_hor)
                    .saturating_sub(sbx * (32 >> self.ss_hor))
                    .min(32);
                if uv_x0 < uv_x1 {
                    self.row_lpf_uv[dst_slot][uv_x0..uv_x1]
                        .copy_from_slice(&other.row_lpf_uv[src_slot][uv_x0..uv_x1]);
                }
            }
        }
    }

    /// `fill_levelcache_block` (`lf_mask.c:264-279`).
    fn fill_level(&mut self, x: usize, y: usize, w: usize, h: usize, luma: bool, v0: u8, v1: u8) {
        let (xb, yb) = if luma {
            (self.x_base, self.y_base)
        } else {
            (self.x_base >> self.ss_hor, self.y_base >> self.ss_ver)
        };
        let lx = x - xb;
        let ly = y - yb;
        let stride = self.stride_y;
        if luma {
            for yy in 0..h {
                let base = (ly + yy) * stride + lx;
                for e in &mut self.level[base..base + w] {
                    e[0] = v0;
                    e[1] = v1;
                }
            }
        } else {
            for yy in 0..h {
                let base = (ly + yy) * stride + lx;
                for e in &mut self.level[base..base + w] {
                    e[2] = v0;
                    e[3] = v1;
                }
            }
        }
    }

    /// `mask_edges_intra` (`lf_mask.c:147-199`).
    fn mask_edges_intra(
        &mut self,
        gx: usize,
        gy: usize,
        w4: usize,
        h4: usize,
        tx: usize,
        a: &mut [u8],
        l: &mut [u8],
    ) {
        let lgx = gx - self.x_base;
        let lgy = gy - self.y_base;
        let stride = self.stride_y;
        let td = &t::TXFM_DIMENSIONS[tx];
        let twl4 = td[2];
        let thl4 = td[3];
        let twl4c = twl4.min(2);
        let thl4c = thl4.min(2);
        for y in 0..h4 {
            let idx = usize::from(twl4c).min(usize::from(l[y]));
            self.mask_y_col[(lgy + y) * stride + lgx] = idx as u8 + 1;
        }
        for x in 0..w4 {
            let idx = usize::from(thl4c).min(usize::from(a[x]));
            self.mask_y_row[lgy * stride + lgx + x] = idx as u8 + 1;
        }
        let hstep = usize::from(td[0]);
        let mut x = hstep;
        while x < w4 {
            for y in 0..h4 {
                self.mask_y_col[(lgy + y) * stride + lgx + x] = twl4c + 1;
            }
            x += hstep;
        }
        let vstep = usize::from(td[1]);
        let mut y = vstep;
        while y < h4 {
            for x in 0..w4 {
                self.mask_y_row[(lgy + y) * stride + lgx + x] = thl4c + 1;
            }
            y += vstep;
        }
        for y in 0..h4 {
            l[y] = twl4c;
        }
        for x in 0..w4 {
            a[x] = thl4c;
        }
    }

    /// `mask_edges_inter` (`lf_mask.c:79-145`).
    fn mask_edges_inter(
        &mut self,
        gx: usize,
        gy: usize,
        w4: usize,
        h4: usize,
        skip: bool,
        max_tx: usize,
        tx_masks: &[u16; 2],
        a: &mut [u8],
        l: &mut [u8],
    ) {
        let lgx = gx - self.x_base;
        let lgy = gy - self.y_base;
        let stride = self.stride_y;
        let td = &t::TXFM_DIMENSIONS[max_tx];
        if (tx_masks[0] | tx_masks[1]) == 0 {
            let twl4c = td[2].min(2);
            let thl4c = td[3].min(2);
            for y in 0..h4 {
                self.mask_y_col[(lgy + y) * stride + lgx] = twl4c.min(l[y]) + 1;
            }
            for x in 0..w4 {
                self.mask_y_row[lgy * stride + lgx + x] = thl4c.min(a[x]) + 1;
            }
            if !skip {
                let hstep = usize::from(td[0]);
                let mut x = hstep;
                while x < w4 {
                    for y in 0..h4 {
                        self.mask_y_col[(lgy + y) * stride + lgx + x] = twl4c + 1;
                    }
                    x += hstep;
                }
                let vstep = usize::from(td[1]);
                let mut y = vstep;
                while y < h4 {
                    self.mask_y_row[(lgy + y) * stride + lgx..(lgy + y) * stride + lgx + w4]
                        .fill(thl4c + 1);
                    y += vstep;
                }
            }
            l[..h4].fill(twl4c);
            a[..w4].fill(thl4c);
            return;
        }
        let mut map = TxMap::default();
        let mut y_off = 0;
        let mut y = 0;
        while y < h4 {
            let mut x_off = 0;
            let mut x = 0;
            while x < w4 {
                decomp_tx(&mut map, max_tx, 0, y, x, y_off, x_off, tx_masks);
                x += usize::from(td[0]);
                x_off += 1;
            }
            y += usize::from(td[1]);
            y_off += 1;
        }
        for y in 0..h4 {
            let idx = usize::from(map.tx0[y][0]).min(usize::from(l[y]));
            self.mask_y_col[(lgy + y) * stride + lgx] = idx as u8 + 1;
        }
        for x in 0..w4 {
            let idx = usize::from(map.tx1[0][x]).min(usize::from(a[x]));
            self.mask_y_row[lgy * stride + lgx + x] = idx as u8 + 1;
        }
        if !skip {
            for y in 0..h4 {
                let mut ltx = usize::from(map.tx0[y][0]);
                let mut step = usize::from(map.step0[y][0]);
                let mut x = step;
                while x < w4 && step > 0 {
                    let rtx = usize::from(map.tx0[y][x]);
                    let idx = rtx.min(ltx);
                    self.mask_y_col[(lgy + y) * stride + lgx + x] = idx as u8 + 1;
                    ltx = rtx;
                    step = usize::from(map.step0[y][x]);
                    x += step;
                }
            }
            for x in 0..w4 {
                let mut ttx = usize::from(map.tx1[0][x]);
                let mut step = usize::from(map.step1[0][x]);
                let mut y = step;
                while y < h4 && step > 0 {
                    let btx = usize::from(map.tx1[y][x]);
                    let idx = ttx.min(btx);
                    self.mask_y_row[(lgy + y) * stride + lgx + x] = idx as u8 + 1;
                    ttx = btx;
                    step = usize::from(map.step1[y][x]);
                    y += step;
                }
            }
        }
        for y in 0..h4 {
            l[y] = map.tx0[y][w4 - 1];
        }
        for x in 0..w4 {
            a[x] = map.tx1[h4 - 1][x];
        }
    }

    /// `mask_edges_chroma` (`lf_mask.c:200-262`).
    #[allow(clippy::too_many_arguments)]
    fn mask_edges_chroma(
        &mut self,
        gx: usize,
        gy: usize,
        w4: usize,
        h4: usize,
        skip_inter: bool,
        tx: usize,
        a: &mut [u8],
        l: &mut [u8],
        ss_hor: usize,
        ss_ver: usize,
    ) {
        let cgx = (gx >> ss_hor) - (self.x_base >> ss_hor);
        let cgy = (gy >> ss_ver) - (self.y_base >> ss_ver);
        let stride_uv = self.stride_uv;
        let td = &t::TXFM_DIMENSIONS[tx];
        let twl4 = td[2];
        let thl4 = td[3];
        let twl4c = u8::from(twl4 != 0);
        let thl4c = u8::from(thl4 != 0);
        for y in 0..h4 {
            let idx = twl4c.min(l[y]);
            self.mask_uv_col[(cgy + y) * stride_uv + cgx] = idx + 1;
        }
        for x in 0..w4 {
            let idx = thl4c.min(a[x]);
            self.mask_uv_row[cgy * stride_uv + cgx + x] = idx + 1;
        }
        if !skip_inter {
            let hstep = usize::from(td[0]);
            let mut x = hstep;
            while x < w4 {
                for y in 0..h4 {
                    self.mask_uv_col[(cgy + y) * stride_uv + cgx + x] = twl4c + 1;
                }
                x += hstep;
            }
            let vstep = usize::from(td[1]);
            let mut y = vstep;
            while y < h4 {
                for x in 0..w4 {
                    self.mask_uv_row[(cgy + y) * stride_uv + cgx + x] = thl4c + 1;
                }
                y += vstep;
            }
        }
        for y in 0..h4 {
            l[y] = twl4c;
        }
        for x in 0..w4 {
            a[x] = thl4c;
        }
    }

    /// `dav1d_create_lf_mask_intra` (`lf_mask.c:274-325`).
    ///
    /// `lv` seçili segmentin/yönün dört seviyesidir:
    /// `[y_dikey, y_yatay, u, v]`.
    #[allow(clippy::too_many_arguments)]
    pub fn create_mask_intra(
        &mut self,
        bx: usize,
        by: usize,
        iw: usize,
        ih: usize,
        bs: u8,
        ytx: usize,
        uvtx: usize,
        lv: [u8; 4],
        ay: &mut [u8],
        ly: &mut [u8],
        auv: Option<(&mut [u8], &mut [u8])>,
    ) {
        let b_dim = &t::BLOCK_DIMENSIONS[usize::from(bs)];
        let bw4 = usize::from(b_dim[0]).min(iw - bx);
        let bh4 = usize::from(b_dim[1]).min(ih - by);
        // C yalnız luma dalını `if (bw4 && bh4)` ile korur; kroma dalı
        // `auv` varsa **bw4/bh4 sıfır olsa da** çalışır (`lf_mask.c:299-315`).
        if bw4 > 0 && bh4 > 0 {
            self.fill_level(bx, by, bw4, bh4, true, lv[0], lv[1]);
            self.mask_edges_intra(bx, by, bw4, bh4, ytx, ay, ly);
        }
        let Some((auv, luv)) = auv else { return };
        let cbw4 = ((iw + self.ss_hor) >> self.ss_hor) - (bx >> self.ss_hor);
        let cbw4 = cbw4.min((usize::from(b_dim[0]) + self.ss_hor) >> self.ss_hor);
        let cbh4 = ((ih + self.ss_ver) >> self.ss_ver) - (by >> self.ss_ver);
        let cbh4 = cbh4.min((usize::from(b_dim[1]) + self.ss_ver) >> self.ss_ver);
        if cbw4 == 0 || cbh4 == 0 {
            return;
        }
        self.fill_level(
            bx >> self.ss_hor,
            by >> self.ss_ver,
            cbw4,
            cbh4,
            false,
            lv[2],
            lv[3],
        );
        self.mask_edges_chroma(
            bx,
            by,
            cbw4,
            cbh4,
            false,
            uvtx,
            auv,
            luv,
            self.ss_hor,
            self.ss_ver,
        );
    }

    /// `dav1d_create_lf_mask_inter` (`lf_mask.c:327-380`).
    #[allow(clippy::too_many_arguments)]
    pub fn create_mask_inter(
        &mut self,
        bx: usize,
        by: usize,
        iw: usize,
        ih: usize,
        bs: u8,
        skip: bool,
        max_ytx: usize,
        tx_masks: &[u16; 2],
        uvtx: usize,
        lv: [u8; 4],
        ay: &mut [u8],
        ly: &mut [u8],
        auv: Option<(&mut [u8], &mut [u8])>,
    ) {
        let b_dim = &t::BLOCK_DIMENSIONS[usize::from(bs)];
        let bw4 = usize::from(b_dim[0]).min(iw - bx);
        let bh4 = usize::from(b_dim[1]).min(ih - by);
        if bw4 > 0 && bh4 > 0 {
            self.fill_level(bx, by, bw4, bh4, true, lv[0], lv[1]);
            self.mask_edges_inter(bx, by, bw4, bh4, skip, max_ytx, tx_masks, ay, ly);
        }
        let Some((auv, luv)) = auv else { return };
        let cbw4 = ((iw + self.ss_hor) >> self.ss_hor) - (bx >> self.ss_hor);
        let cbw4 = cbw4.min((usize::from(b_dim[0]) + self.ss_hor) >> self.ss_hor);
        let cbh4 = ((ih + self.ss_ver) >> self.ss_ver) - (by >> self.ss_ver);
        let cbh4 = cbh4.min((usize::from(b_dim[1]) + self.ss_ver) >> self.ss_ver);
        if cbw4 == 0 || cbh4 == 0 {
            return;
        }
        self.fill_level(
            bx >> self.ss_hor,
            by >> self.ss_ver,
            cbw4,
            cbh4,
            false,
            lv[2],
            lv[3],
        );
        self.mask_edges_chroma(
            bx,
            by,
            cbw4,
            cbh4,
            skip,
            uvtx,
            auv,
            luv,
            self.ss_hor,
            self.ss_ver,
        );
    }
}

/// Tanılama: FNV-1a 64 (varsayılan kapalı).
fn fnv_bytes(mut h: u64, b: &[u8]) -> u64 {
    for &v in b {
        h ^= u64::from(v);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

#[inline(always)]
fn trace_on() -> bool {
    #[cfg(feature = "std")]
    {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("AV1OXIDE_LF_TRACE").is_some())
    }
    #[cfg(not(feature = "std"))]
    {
        false
    }
}

impl LfFrame {
    fn trace_plane(
        &self,
        pic: &crate::decode::intra::PicBuf,
        plane: usize,
        y_lo: usize,
        y_hi: usize,
    ) -> u64 {
        if plane >= 1 && !pic.has_chroma {
            return 0;
        }
        let w = if plane == 0 { pic.width } else { pic.cw };
        let h = if plane == 0 { pic.height } else { pic.ch };
        let (mut lo, mut hi) = (y_lo, y_hi);
        if plane >= 1 {
            lo = (lo + self.ss_ver) >> self.ss_ver;
            hi = (hi + self.ss_ver) >> self.ss_ver;
        }
        if hi > h {
            hi = h;
        }
        let stride = pic.stride[if plane == 0 { 0 } else { 1 }];
        let row = w * pic.bpp;
        let mut acc = 0xcbf2_9ce4_8422_2325u64;
        for y in lo..hi {
            let off = y * stride;
            acc = fnv_bytes(acc, &pic.data[plane][off..off + row]);
        }
        acc
    }

    fn trace_masks(&self, y0: usize, count: usize) -> u64 {
        let mut acc = 0xcbf2_9ce4_8422_2325u64;
        for gy in y0..y0 + count {
            for gx in 1..self.bw {
                acc = fnv_bytes(acc, &[self.mask_y_col[gy * self.bw + gx]]);
            }
            for gx in 0..self.bw {
                acc = fnv_bytes(acc, &[self.mask_y_row[gy * self.bw + gx]]);
            }
        }
        acc
    }

    fn trace_level(&self, y0: usize, count: usize) -> u64 {
        let mut acc = 0xcbf2_9ce4_8422_2325u64;
        for gy in y0..y0 + count {
            for gx in 0..self.bw {
                acc = fnv_bytes(acc, &self.level[gy * self.bw + gx]);
            }
        }
        acc
    }

    fn trace_dump(
        &self,
        pic: &crate::decode::intra::PicBuf,
        frame_no: u32,
        sby: usize,
        stage: &str,
        y0: usize,
        count: usize,
    ) {
        if stage == "pre-cols" || stage == "pre-filter" {
            crate::trace_eprintln!(
                "TILE {} {} {} {:?} {:?}",
                frame_no,
                self.n_cols,
                self.n_rows,
                0,
                0
            );
            for gy in y0..y0 + count {
                let mut line = format!("MC {} {} {} {}", stage, frame_no, sby, gy);
                for gx in 1..self.bw {
                    let _ = core::fmt::Write::write_fmt(
                        &mut line,
                        format_args!(" {}", self.mask_y_col[gy * self.bw + gx]),
                    );
                }
                crate::trace_eprintln!("{line}");
                let mut line = format!("MR {} {} {} {}", stage, frame_no, sby, gy);
                for gx in 0..self.bw {
                    let _ = core::fmt::Write::write_fmt(
                        &mut line,
                        format_args!(" {}", self.mask_y_row[gy * self.bw + gx]),
                    );
                }
                crate::trace_eprintln!("{line}");
            }
        }
        if stage == "pre-cols" {
            for gy in y0..y0 + count {
                let mut line = format!("LV {} {} {} {}", stage, frame_no, sby, gy);
                for gx in 0..self.bw {
                    let e = self.level[gy * self.bw + gx];
                    let _ = core::fmt::Write::write_fmt(
                        &mut line,
                        format_args!(" {},{},{},{}", e[0], e[1], e[2], e[3]),
                    );
                }
                crate::trace_eprintln!("{line}");
            }
        }
        let mut lo = y0 * 4;
        let mut hi = (y0 + count) * 4;
        if stage.starts_with("post-rows") {
            lo = lo.saturating_sub(8);
            hi = (hi + 8).min((y0 + count) * 4);
        }
        crate::trace_eprintln!(
            "T {} {} {} off={} lv={},{},{},{} {:016x} {:016x} {:016x} lvl={:016x} msk={:016x}",
            frame_no,
            sby,
            stage,
            self.trace_off,
            self.trace_lv[0],
            self.trace_lv[1],
            self.trace_lv[2],
            self.trace_lv[3],
            self.trace_plane(pic, 0, lo, hi),
            self.trace_plane(pic, 1, lo, hi),
            self.trace_plane(pic, 2, lo, hi),
            self.trace_level(y0, count),
            self.trace_masks(y0, count)
        );
    }

    /// Kareyi dav1d sırasıyla süz: her SB satırında önce sütun (dikey)
    /// kenarlar, sonra satır (yatay) kenarlar; Y, U, V
    /// (`recon_tmpl.c:1987-2020,2099-2102`).
    pub fn apply(
        &mut self,
        pic: &mut crate::decode::intra::PicBuf,
        hdr: &FrameHeader,
        frame_no: u32,
        dsp: &LoopFilterDsp,
        pool: &crate::thread::WorkerPool,
    ) {
        if hdr.loopfilter.level_y[0] == 0 && hdr.loopfilter.level_y[1] == 0 {
            return;
        }
        // `start_of_tile_row` (`decode.c:2769-2775`).
        let mut start_of_tile_row = vec![0u8; self.sbh];
        let mut sby = 0usize;
        for tr in 0..self.n_rows {
            if sby < self.sbh {
                start_of_tile_row[sby] = tr as u8;
                sby += 1;
            }
            let end = usize::from(hdr.tiling.row_start_sb[tr + 1]).min(self.sbh);
            while sby < end {
                start_of_tile_row[sby] = 0;
                sby += 1;
            }
        }
        let has_chroma = pic.has_chroma;
        let uv_lvl = hdr.loopfilter.level_u != 0 || hdr.loopfilter.level_v != 0;
        #[cfg(feature = "std")]
        {
            static MASKCNT: OnceLock<bool> = OnceLock::new();
            if *MASKCNT.get_or_init(|| std::env::var_os("AV1OXIDE_LF_MASKCNT").is_some()) {
                let nc = self.mask_uv_col.iter().filter(|&&v| v != 0).count();
                let nr = self.mask_uv_row.iter().filter(|&&v| v != 0).count();
                let lc = self.level.iter().filter(|e| e[2] != 0 || e[3] != 0).count();
                crate::trace_eprintln!(
                    "MASKCNT off={} uv_lvl={} has={} col={nc} row={nr} lvl={lc}",
                    hdr.frame_offset,
                    uv_lvl,
                    has_chroma
                );
            }
        }
        let trace = trace_on() && frame_no < 12;
        if pool.is_parallel() && !trace && self.sbh > 0 && (has_chroma && uv_lvl) {
            self.apply_parallel_planes(pic, hdr, &start_of_tile_row, dsp, pool);
            return;
        }
        for sby in 0..self.sbh {
            let y0 = sby * self.sb_step;
            let count = (self.bh - y0).min(self.sb_step);
            let trace = trace_on() && frame_no < 12;
            if trace {
                self.trace_dump(pic, frame_no, sby, "pre-cols", y0, count);
            }
            // --- tile sütun sınırlarında filtre boyunu düzelt -----------------
            for tile_col in 1..self.n_cols {
                let gx = usize::from(hdr.tiling.col_start_sb[tile_col]) * self.sb_step;
                if gx >= self.bw {
                    break;
                }
                let base = (tile_col - 1) * self.align_h + y0;
                for i in 0..count {
                    let gy = y0 + i;
                    let v = self.mask_y_col[gy * self.bw + gx];
                    let idx = if v > 0 { usize::from(v - 1) } else { 0 };
                    let lp = usize::from(self.tx_right_y[base + i]);
                    self.mask_y_col[gy * self.bw + gx] = idx.min(lp) as u8 + 1;
                }
                if has_chroma {
                    let cgx = gx >> self.ss_hor;
                    let cy0 = y0 >> self.ss_ver;
                    let ccount = ((y0 + count + self.ss_ver) >> self.ss_ver) - cy0;
                    let base = (tile_col - 1) * (self.align_h >> self.ss_ver) + cy0;
                    for i in 0..ccount {
                        let cy = cy0 + i;
                        let v = self.mask_uv_col[cy * self.cw + cgx];
                        let idx = if v > 0 { usize::from(v - 1) } else { 0 };
                        let lp = usize::from(self.tx_right_uv[base + i]);
                        self.mask_uv_col[cy * self.cw + cgx] = idx.min(lp) as u8 + 1;
                    }
                }
            }
            // --- tile satırı başında üst kenarı düzelt -----------------------
            let tr = usize::from(start_of_tile_row[sby]);
            if tr != 0 {
                let prev = (tr - 1) * self.sb128w;
                for x in 0..self.sb128w {
                    let w = 32.min(self.bw - x * 32);
                    for i in 0..w {
                        let gx = x * 32 + i;
                        let v = self.mask_y_row[y0 * self.bw + gx];
                        let idx = if v > 0 { usize::from(v - 1) } else { 0 };
                        let lp = usize::from(self.row_lpf_y[prev + x][i]);
                        self.mask_y_row[y0 * self.bw + gx] = idx.min(lp) as u8 + 1;
                    }
                    if has_chroma {
                        let cw = (w + self.ss_hor) >> self.ss_hor;
                        let cgy = y0 >> self.ss_ver;
                        let cbx = (x * 32) >> self.ss_hor;
                        for i in 0..cw {
                            let cgx = cbx + i;
                            let v = self.mask_uv_row[cgy * self.cw + cgx];
                            let idx = if v > 0 { usize::from(v - 1) } else { 0 };
                            let lp = usize::from(self.row_lpf_uv[prev + x][i]);
                            self.mask_uv_row[cgy * self.cw + cgx] = idx.min(lp) as u8 + 1;
                        }
                    }
                }
            }
            // --- sütun (dikey) kenarlar, Y -----------------------------------
            if trace {
                self.trace_dump(pic, frame_no, sby, "pre-filter", y0, count);
            }
            // F3c: kenar kümeleri `loop_filter_sb[düzlem][yön]` yuvalarından
            // süzülür (dav1d `lf_apply_tmpl.c` `filter_plane_{cols,rows}_{y,uv}`).
            let [[h_y, v_y], [h_uv, v_uv]] = dsp.loop_filter_sb;
            let (bw, cw) = (self.bw, self.cw);
            crate::dsp::profil_olc!(
                LoopFilterSb,
                h_y(
                    &mut pic.data[0],
                    pic.stride[0],
                    &self.mask_y_col,
                    bw,
                    &self.level,
                    bw,
                    0,
                    &self.lim,
                    bw,
                    y0,
                    count,
                )
            );
            if trace {
                self.trace_dump(pic, frame_no, sby, "post-cols", y0, count);
            }
            crate::dsp::profil_olc!(
                LoopFilterSb,
                v_y(
                    &mut pic.data[0],
                    pic.stride[0],
                    &self.mask_y_row,
                    bw,
                    &self.level,
                    bw,
                    1,
                    &self.lim,
                    bw,
                    y0,
                    count,
                )
            );
            if trace {
                self.trace_dump(pic, frame_no, sby, "post-rows-y", y0, count);
            }
            if !has_chroma || !uv_lvl {
                continue;
            }
            let cy0 = y0 >> self.ss_ver;
            let ccount = ((y0 + count + self.ss_ver) >> self.ss_ver) - cy0;
            // Kroma: U ([2]) ve V ([3]) ayrı düzlemlerdir; eski döngü kenar
            // başına iki düzlemi sırayla süzüyordu, düzlemler birbirinden
            // bağımsız olduğundan düzlem başına çağrı aynı sonucu verir.
            for pl in 1..3 {
                crate::dsp::profil_olc!(
                    LoopFilterSb,
                    h_uv(
                        &mut pic.data[pl],
                        pic.stride[1],
                        &self.mask_uv_col,
                        cw,
                        &self.level,
                        bw,
                        pl + 1,
                        &self.lim,
                        cw,
                        cy0,
                        ccount,
                    )
                );
            }
            for pl in 1..3 {
                crate::dsp::profil_olc!(
                    LoopFilterSb,
                    v_uv(
                        &mut pic.data[pl],
                        pic.stride[1],
                        &self.mask_uv_row,
                        cw,
                        &self.level,
                        bw,
                        pl + 1,
                        &self.lim,
                        cw,
                        cy0,
                        ccount,
                    )
                );
            }
            if trace {
                self.trace_dump(pic, frame_no, sby, "post-rows", y0, count);
            }
        }
        if trace_on() && frame_no < 12 {
            crate::trace_eprintln!(
                "F {} off={} {:016x} {:016x} {:016x}",
                frame_no,
                self.trace_off,
                self.trace_plane(pic, 0, 0, pic.height),
                self.trace_plane(pic, 1, 0, pic.height),
                self.trace_plane(pic, 2, 0, pic.height)
            );
        }
    }

    /// Karo sınırı maske düzeltmelerini uyguladıktan sonra `Y`, `U` ve `V`
    /// düzlemlerini sıfır-kopya (`core::mem::take(&mut pic.data[pl])`) olarak
    /// `WorkerPool` üzerinde paralel süzer (`F8b-3`).
    fn apply_parallel_planes(
        &mut self,
        pic: &mut crate::decode::intra::PicBuf,
        hdr: &FrameHeader,
        start_of_tile_row: &[u8],
        dsp: &LoopFilterDsp,
        pool: &crate::thread::WorkerPool,
    ) {
        use alloc::sync::Arc;

        let has_chroma = pic.has_chroma;
        if self.n_cols > 1 || self.n_rows > 1 {
            for (sby, &tr_raw) in start_of_tile_row.iter().enumerate().take(self.sbh) {
                let y0 = sby * self.sb_step;
                let count = (self.bh - y0).min(self.sb_step);
                for tile_col in 1..self.n_cols {
                    let gx = usize::from(hdr.tiling.col_start_sb[tile_col]) * self.sb_step;
                    if gx >= self.bw {
                        break;
                    }
                    let base = (tile_col - 1) * self.align_h + y0;
                    for i in 0..count {
                        let gy = y0 + i;
                        let v = self.mask_y_col[gy * self.bw + gx];
                        let idx = if v > 0 { usize::from(v - 1) } else { 0 };
                        let lp = usize::from(self.tx_right_y[base + i]);
                        self.mask_y_col[gy * self.bw + gx] = idx.min(lp) as u8 + 1;
                    }
                    if has_chroma {
                        let cgx = gx >> self.ss_hor;
                        let cy0 = y0 >> self.ss_ver;
                        let ccount = ((y0 + count + self.ss_ver) >> self.ss_ver) - cy0;
                        let base = (tile_col - 1) * (self.align_h >> self.ss_ver) + cy0;
                        for i in 0..ccount {
                            let cy = cy0 + i;
                            let v = self.mask_uv_col[cy * self.cw + cgx];
                            let idx = if v > 0 { usize::from(v - 1) } else { 0 };
                            let lp = usize::from(self.tx_right_uv[base + i]);
                            self.mask_uv_col[cy * self.cw + cgx] = idx.min(lp) as u8 + 1;
                        }
                    }
                }
                let tr = usize::from(tr_raw);
                if tr != 0 {
                    let prev = (tr - 1) * self.sb128w;
                    for x in 0..self.sb128w {
                        let w = 32.min(self.bw - x * 32);
                        for i in 0..w {
                            let gx = x * 32 + i;
                            let v = self.mask_y_row[y0 * self.bw + gx];
                            let idx = if v > 0 { usize::from(v - 1) } else { 0 };
                            let lp = usize::from(self.row_lpf_y[prev + x][i]);
                            self.mask_y_row[y0 * self.bw + gx] = idx.min(lp) as u8 + 1;
                        }
                        if has_chroma {
                            let cw = (w + self.ss_hor) >> self.ss_hor;
                            let cgy = y0 >> self.ss_ver;
                            let cbx = (x * 32) >> self.ss_hor;
                            for i in 0..cw {
                                let cgx = cbx + i;
                                let v = self.mask_uv_row[cgy * self.cw + cgx];
                                let idx = if v > 0 { usize::from(v - 1) } else { 0 };
                                let lp = usize::from(self.row_lpf_uv[prev + x][i]);
                                self.mask_uv_row[cgy * self.cw + cgx] = idx.min(lp) as u8 + 1;
                            }
                        }
                    }
                }
            }
        }

        struct PlaneLfJob {
            pl: usize,
            stride: usize,
            data: Vec<u8>,
        }

        let mut jobs = vec![
            PlaneLfJob {
                pl: 0,
                stride: pic.stride[0],
                data: core::mem::take(&mut pic.data[0]),
            },
            PlaneLfJob {
                pl: 1,
                stride: pic.stride[1],
                data: core::mem::take(&mut pic.data[1]),
            },
            PlaneLfJob {
                pl: 2,
                stride: pic.stride[1],
                data: core::mem::take(&mut pic.data[2]),
            },
        ];

        let level_arc = Arc::new(core::mem::take(&mut self.level));
        let myc_arc = Arc::new(core::mem::take(&mut self.mask_y_col));
        let myr_arc = Arc::new(core::mem::take(&mut self.mask_y_row));
        let muc_arc = Arc::new(core::mem::take(&mut self.mask_uv_col));
        let mur_arc = Arc::new(core::mem::take(&mut self.mask_uv_row));

        let [[h_y, v_y], [h_uv, v_uv]] = dsp.loop_filter_sb;
        let (bw, bh, cw) = (self.bw, self.bh, self.cw);
        let (sbh, sb_step, ss_ver) = (self.sbh, self.sb_step, self.ss_ver);
        let lim = self.lim;

        let lvl_w = Arc::clone(&level_arc);
        let myc_w = Arc::clone(&myc_arc);
        let myr_w = Arc::clone(&myr_arc);
        let muc_w = Arc::clone(&muc_arc);
        let mur_w = Arc::clone(&mur_arc);

        let _ = pool.execute_tasks(&mut jobs, move |_wid, _idx, job: &mut PlaneLfJob| {
            if job.pl == 0 {
                for sby in 0..sbh {
                    let y0 = sby * sb_step;
                    let count = (bh - y0).min(sb_step);
                    crate::dsp::profil_olc!(
                        LoopFilterSb,
                        h_y(
                            &mut job.data,
                            job.stride,
                            &myc_w,
                            bw,
                            &lvl_w,
                            bw,
                            0,
                            &lim,
                            bw,
                            y0,
                            count,
                        )
                    );
                    crate::dsp::profil_olc!(
                        LoopFilterSb,
                        v_y(
                            &mut job.data,
                            job.stride,
                            &myr_w,
                            bw,
                            &lvl_w,
                            bw,
                            1,
                            &lim,
                            bw,
                            y0,
                            count,
                        )
                    );
                }
            } else {
                let pl = job.pl;
                for sby in 0..sbh {
                    let y0 = sby * sb_step;
                    let count = (bh - y0).min(sb_step);
                    let cy0 = y0 >> ss_ver;
                    let ccount = ((y0 + count + ss_ver) >> ss_ver) - cy0;
                    crate::dsp::profil_olc!(
                        LoopFilterSb,
                        h_uv(
                            &mut job.data,
                            job.stride,
                            &muc_w,
                            cw,
                            &lvl_w,
                            bw,
                            pl + 1,
                            &lim,
                            cw,
                            cy0,
                            ccount,
                        )
                    );
                    crate::dsp::profil_olc!(
                        LoopFilterSb,
                        v_uv(
                            &mut job.data,
                            job.stride,
                            &mur_w,
                            cw,
                            &lvl_w,
                            bw,
                            pl + 1,
                            &lim,
                            cw,
                            cy0,
                            ccount,
                        )
                    );
                }
            }
            Ok(())
        });

        for job in jobs {
            pic.data[job.pl] = job.data;
        }
        self.level = Arc::try_unwrap(level_arc).unwrap_or_else(|a| (*a).clone());
        self.mask_y_col = Arc::try_unwrap(myc_arc).unwrap_or_else(|a| (*a).clone());
        self.mask_y_row = Arc::try_unwrap(myr_arc).unwrap_or_else(|a| (*a).clone());
        self.mask_uv_col = Arc::try_unwrap(muc_arc).unwrap_or_else(|a| (*a).clone());
        self.mask_uv_row = Arc::try_unwrap(mur_arc).unwrap_or_else(|a| (*a).clone());
    }
}

#[cfg(all(test, feature = "std"))]
mod loopfilter_oracle_tests {
    //! S2c.1 çekirdek oracle differential'ı (`scripts/oracle/run_loopfilter_oracle.sh`).
    //!
    //! C oracle'ının ürettiği girdi dosyasını okur, aynı SB işlevlerini Rust
    //! çekirdeğiyle çalıştırır ve `loopfilter-rust.bin` yazar. Karşılaştırma
    //! betikte SHA-256 ile yapılır.

    use super::*;
    use crate::dsp::scalar::lf::loop_filter;

    fn run_sb(
        func: usize,
        data: &mut [u8],
        offp: usize,
        stride: isize,
        vmask: &[u32; 4],
        l: &[[u8; 4]],
        b4_stride: isize,
        lim: &LimLut,
        hw: usize,
        bpp: usize,
        bdmin: u32,
        maxpx: i32,
    ) {
        let vm = vmask[0] | vmask[1] | vmask[2];
        let uv = func >= 2;
        let h = func & 1 == 0;
        let mut off: isize = offp as isize;
        let mut li: isize = 32;
        let mut bit: u32 = 1;
        while vm & !(bit - 1) != 0 {
            if vm & bit != 0 {
                let lv = if h {
                    let a = l[li as usize][0];
                    if a != 0 { a } else { l[(li - 1) as usize][0] }
                } else {
                    let a = l[li as usize][0];
                    if a != 0 {
                        a
                    } else {
                        l[(li - b4_stride) as usize][0]
                    }
                };
                if lv != 0 {
                    let idx = if vmask[2] & bit != 0 {
                        2
                    } else {
                        usize::from(vmask[1] & bit != 0)
                    };
                    let wd = if uv { 4 + 2 * idx } else { 4 << idx };
                    let (sa, sb) = if h {
                        (stride, bpp as isize)
                    } else {
                        (bpp as isize, stride)
                    };
                    assert!(
                        off - 4 * sb >= 0 && off + 4 * sb < data.len() as isize,
                        "func={func} hw={hw} off={off} sb={sb} len={}",
                        data.len()
                    );
                    let _ = maxpx;
                    let (e, i, hh) = (lim.e[lv as usize], lim.i[lv as usize], i32::from(lv >> 4));
                    match bdmin {
                        0 => loop_filter::<8>(data, off, sa, sb, wd, e, i, hh),
                        2 => loop_filter::<10>(data, off, sa, sb, wd, e, i, hh),
                        _ => loop_filter::<12>(data, off, sa, sb, wd, e, i, hh),
                    }
                }
            }
            bit <<= 1;
            if h {
                off += 4 * stride;
                li += b4_stride;
            } else {
                off += 4 * bpp as isize;
                li += 1;
            }
        }
    }

    /// Oracle sürücüsündeki `lut` (seviye başına sharp = L & 7).
    fn oracle_lut() -> LimLut {
        let mut lut = LimLut::default();
        for level in 0..64i32 {
            let sharp = level & 7;
            let mut limit = level;
            if sharp > 0 {
                limit >>= (sharp + 3) >> 2;
                limit = limit.min(9 - sharp);
            }
            limit = limit.max(1);
            lut.i[level as usize] = limit;
            lut.e[level as usize] = 2 * (level + 2) + limit;
        }
        lut
    }

    #[test]
    fn loopfilter_oracle() {
        let Ok(dir) = std::env::var("LOOPFILTER_ORACLE_OUT") else {
            return;
        };
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let dir = base.join(&dir);
        let input = std::fs::read(dir.join("loopfilter-input.bin")).expect("girdi");
        let mut out = Vec::with_capacity(input.len());
        let mut p = 0usize;
        let mut calls = 0usize;
        while p + 12 <= input.len() {
            let h0u = u32::from_le_bytes([input[p], input[p + 1], input[p + 2], input[p + 3]]);
            let h1u = u32::from_le_bytes([input[p + 4], input[p + 5], input[p + 6], input[p + 7]]);
            let stride = usize::from(u16::from_le_bytes([input[p + 8], input[p + 9]]));
            let offp = usize::from(u16::from_le_bytes([input[p + 10], input[p + 11]]));
            p += 12;
            let bd = (h0u & 0xff) as usize;
            let func = ((h0u >> 8) & 0xff) as usize;
            let _wd = (h0u >> 16) as usize;
            let pattern = ((h1u >> 8) & 0xff) as usize;
            let n = (h1u >> 16) as usize;
            let _ = pattern;
            let mut vmask = [0u32; 4];
            for v in vmask.iter_mut() {
                *v = u32::from_le_bytes([input[p], input[p + 1], input[p + 2], input[p + 3]]);
                p += 4;
            }
            let lsize = usize::from(u16::from_le_bytes([input[p], input[p + 1]]));
            p += 2;
            let lb = &input[p..p + lsize];
            p += lsize;
            let bpp = if bd > 8 { 2 } else { 1 };
            let plane_len = 64 * 64 * bpp;
            let mut plane = input[p..p + plane_len].to_vec();
            p += plane_len;
            let mut l = [[0u8; 4]; 544];
            for (i, e) in l.iter_mut().enumerate() {
                e.copy_from_slice(&lb[i * 4..i * 4 + 4]);
            }
            let lim = oracle_lut();
            run_sb(
                func,
                plane.as_mut_slice(),
                offp,
                stride as isize,
                &vmask,
                &l,
                32,
                &lim,
                n,
                bpp,
                (bd as u32) - 8,
                (1i32 << bd) - 1,
            );
            out.extend_from_slice(&h0u.to_le_bytes());
            out.extend_from_slice(&h1u.to_le_bytes());
            out.extend_from_slice(&(stride as u16).to_le_bytes());
            out.extend_from_slice(&(offp as u16).to_le_bytes());
            for v in vmask {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(&(lsize as u16).to_le_bytes());
            out.extend_from_slice(lb);
            out.extend_from_slice(&plane);
            calls += 1;
        }
        std::fs::write(dir.join("loopfilter-rust.bin"), &out).expect("çıktı");
        println!("[loopfilter] {calls} çağrı");
        assert!(calls >= 1);
    }
}
