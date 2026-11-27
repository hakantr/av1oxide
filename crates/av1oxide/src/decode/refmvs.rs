//! Uzamsal hareket vektörü referans ızgarası — S2b.5.
//!
//! dav1d `src/refmvs.c` (salt-okunur referans @ `99f3354`) uzamsal yolunun
//! birebir karşılığı. Kapsam **yalnız uzamsal adaylardır**; temporal mv
//! (`save_tmvs`/`load_tmvs`, `use_ref_frame_mvs`) P15'tedir ve bu modülde
//! yoktur.
//!
//! ## Kaynak eşlemesi (dav1d @ `99f3354`)
//!
//! | Kural | dav1d satırı |
//! |---|---|
//! | `add_spatial_candidate` | `src/refmvs.c:42-86` |
//! | `scan_row` / `scan_col` | `src/refmvs.c:88-176` |
//! | `dav1d_refmvs_find` (uzamsal) | `src/refmvs.c:349-700` |
//! | tek referans yolu + kırpma | `src/refmvs.c:613-626` |
//! | `dav1d_refmvs_tile_sbrow_init` sınırları | `src/refmvs.c:682-712` |
//! | `splat_intrabc_mv` / `splat_intraref` | `src/decode.c:530-543`, `:562-574` |
//!
//! ## Tasarım notu: halka tampon yerine kare ızgarası
//!
//! dav1d `refmvs_frame.r` alanı, tile satırı başına 35 satırlık **halka
//! tampon** kullanır ve `rt->r[(by4 & 31) + 5]` ile adresler. Bu port,
//! davranışsal olarak eşdeğer olan **kare genişliğinde açık ızgara** kullanır
//! (4×4 blok başına bir kayıt). Taramalar `tile_row.start`/`tile_col.start`
//! ile sınırlandığından halka tamponun tuttuğu eski satırlar hiçbir zaman
//! okunmaz: bir bloğun üst/sol/sağ-üst adayları ya bu tile satırında daha
//! önce çözülmüştür ya da hiç okunmaz. `bs` alanı aday blok boyutunu taşır ve
//! `scan_row`/`scan_col` ilerlemesini belirler.
#![forbid(unsafe_code)]

use alloc::vec::Vec;

use crate::bitstream::frame_header::{
    FrameHeader, WarpedMotionParams, WarpedMotionType, get_poc_diff,
};
use crate::decode::refmvs_tmvs::{MfmvPlan, TemporalBlock, mv_projection};
use crate::dsp::refmvs::RefmvsDsp;
use crate::tables::decode as t;

/// `INVALID_MV` (`src/refmvs.h:40`): `0x80008000` → `y = x = -32768`.
pub(crate) const INVALID_MV: i16 = i16::MIN;

/// `union mv` (`dav1d/headers.h`): `y` alt 16 bit, `x` üst 16 bit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Mv {
    /// Dikey bileşen (1/8 piksel).
    pub y: i16,
    /// Yatay bileşen (1/8 piksel).
    pub x: i16,
}

impl Mv {
    /// Sıfır vektör.
    pub const ZERO: Self = Self { y: 0, x: 0 };
    /// Geçersiz (intra) işareti.
    pub const INVALID: Self = Self {
        y: INVALID_MV,
        x: INVALID_MV,
    };

    /// `union mv.n` — paketlenmiş 32-bit karşılaştırma anahtarı.
    #[must_use]
    pub fn n(self) -> i32 {
        ((u32::from(self.x as u16) << 16) | u32::from(self.y as u16)) as i32
    }

    /// `mv.n == INVALID_MV`.
    #[must_use]
    pub fn is_invalid(self) -> bool {
        self == Self::INVALID
    }
}

/// `refmvs_block` (`src/refmvs.h:57-62`) — 4×4 blok başına bir kayıt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct RefBlock {
    /// İki referans MV'si.
    pub mv: [Mv; 2],
    /// `[0] = 0: intra/intrabc, [1] = -1: tek referans`.
    pub ref_pair: [i8; 2],
    /// `enum BlockSize` — aday taramasında ilerleme adımı.
    pub bs: u8,
    /// `1 = globalmv+affine, 2 = newmv` (intrabc'de 0).
    pub mf: u8,
}

const _: () = assert!(core::mem::size_of::<Mv>() == 4 && core::mem::size_of::<RefBlock>() == 12);

/// `BS_8x8` — yazılmamış ızgara hücresinin güvenli varsayılan boyutu.
pub const BS_8X8: u8 = 4;

/// Yazılmamış ızgara hücresi: geçersiz MV, `ref_pair = [0, -1]`
/// (dav1d `refmvs_block` sıfır başlatması + `INVALID_MV`).
pub const INVALID_BLOCK: RefBlock = RefBlock {
    mv: [Mv::INVALID, Mv::INVALID],
    ref_pair: [0, -1],
    bs: BS_8X8,
    mf: 0,
};

impl Default for RefBlock {
    fn default() -> Self {
        INVALID_BLOCK
    }
}

/// `refmvs_candidate`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Candidate {
    /// Aday MV çifti (`[0]` = ilk referans; compound'da `[1]` = ikinci).
    pub mv: [Mv; 2],
    /// Ağırlık.
    pub weight: i32,
}

/// Tile sınırları (`refmvs_tile.tile_row/tile_col`, 4×4 biriminde).
#[derive(Clone, Copy, Default)]
pub(crate) struct TileBounds {
    /// Tile satırı başlangıcı.
    pub row_start: usize,
    /// Tile satırı bitişi (kare yüksekliğine kırpılmış).
    pub row_end: usize,
    /// Tile sütunu başlangıcı.
    pub col_start: usize,
    /// Tile sütunu bitişi (kare genişliğine kırpılmış).
    pub col_end: usize,
}

/// Kare genişliğinde uzamsal MV ızgarası + kare başına temporal durum.
pub(crate) struct RefMvs {
    /// F3c: `Dav1dRefmvsDSPContext` yuvaları (kurulum tablosunun kopyası).
    dsp: RefmvsDsp,
    pub(crate) grid: Vec<RefBlock>,
    pub(crate) iw4: usize,
    pub(crate) ih4: usize,
    pub(crate) x_base: usize,
    pub(crate) y_base: usize,
    pub(crate) stride: usize,
    /// Tanılama: çözülen kare sırası (`dav1d` `dav1d_decode_frame` sırası).
    /// Yalnız `AV1OXIDE_REFMVS_TRACE` açıkken okunur.
    frame: u32,
    /// Tanılama: kare içi çağrı sırası.
    seq: core::sync::atomic::AtomicU32,
    /// `rf->sign_bias[7]` — `poc_diff > 0` (`refmvs.c:879-887`).
    sign_bias: [u8; 7],
    /// `rf->mfmv_sign[7]` — `poc_diff < 0`; `save_tmvs` filtresi.
    mfmv_sign: [u8; 7],
    /// `rf->pocdiff[7]` — `iclip(get_poc_diff(poc, ref_poc), -31, 31)`.
    pocdiff: [i8; 7],
    /// `rf->use_ref_frame_mvs` (türetilmiş: `n_mfmvs > 0`).
    use_ref_frame_mvs: bool,
    /// `frm_hdr->use_ref_frame_mvs` — `globalmv_ctx` başlangıç değeri
    /// (`refmvs.c:400`); türetilmiş alandan **farklıdır**.
    hdr_use_ref_frame_mvs: bool,
    /// `rf->rp_proj` — geçerli karenin projeksiyonu (tam kare, 8×8; paralel tile'lar arasında sıfır-kopya paylaşılır).
    pub(crate) rp_proj: alloc::sync::Arc<Vec<TemporalBlock>>,
    /// `rf->rp_stride`.
    rp_stride: usize,
    /// `frm_hdr->gmv[7]`.
    gmv: [WarpedMotionParams; 7],
    /// `frm_hdr->hp`.
    hp: bool,
    /// `frm_hdr->force_integer_mv`.
    force_integer_mv: bool,
    /// `rf->n_mfmvs`, `mfmv_ref`, `mfmv_ref2cur`, `mfmv_ref2ref`.
    mfmv: MfmvPlan,
    /// Tanılama: temporal aday kullanım sayısı (araç sayacı).
    pub(crate) temporal_hits: core::sync::atomic::AtomicU64,
}

impl RefMvs {
    /// Kare boyutlarına göre (8 piksele yuvarlanmış) ızgara kurar.
    #[cfg(test)]
    pub(crate) fn new(
        width: usize,
        height: usize,
        dsp: RefmvsDsp,
        butce: &crate::frame::butce::Butce,
    ) -> crate::error::Result<Self> {
        Self::new_with_reuse(width, height, dsp, Vec::new(), butce, false)
    }

    /// `Key` / `Intra` (`!allow_intrabc`) karelerde tam `iw4 × ih4` ızgarası
    /// ayırmadan hafif `RefMvs` kabuğu kurar.
    pub(crate) fn new_empty(width: usize, height: usize, dsp: RefmvsDsp) -> Self {
        let iw4 = ((width + 7) >> 3) << 1;
        let ih4 = ((height + 7) >> 3) << 1;
        Self {
            dsp,
            grid: Vec::new(),
            iw4,
            ih4,
            x_base: 0,
            y_base: 0,
            stride: iw4,
            frame: 0,
            seq: core::sync::atomic::AtomicU32::new(0),
            sign_bias: [0; 7],
            mfmv_sign: [0; 7],
            pocdiff: [0; 7],
            use_ref_frame_mvs: false,
            hdr_use_ref_frame_mvs: false,
            rp_proj: alloc::sync::Arc::new(Vec::new()),
            rp_stride: 0,
            gmv: [WarpedMotionParams::default(); 7],
            hp: false,
            force_integer_mv: false,
            mfmv: MfmvPlan::default(),
            temporal_hits: core::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Havuzdan geri dönüştürülmüş `old_grid` varsa kapasitesi eşleştiğinde
    /// yeniden kullanarak `RefMvs` kurar (`F8b-1`).
    ///
    /// `skip_grid_fill == true` ise paralel çok-karolu birleştirmede tüm
    /// `iw4 × ih4` ızgarası üzerine yazılacağından gereksiz `INVALID_BLOCK`
    /// doldurması atlanır (`5.C.1`).
    pub(crate) fn new_with_reuse(
        width: usize,
        height: usize,
        dsp: RefmvsDsp,
        old_grid: Vec<RefBlock>,
        butce: &crate::frame::butce::Butce,
        skip_grid_fill: bool,
    ) -> crate::error::Result<Self> {
        let iw4 = ((width + 7) >> 3) << 1;
        let ih4 = ((height + 7) >> 3) << 1;
        let grid = if skip_grid_fill {
            butce.reuse_vec_dirty(old_grid, iw4 * ih4, INVALID_BLOCK)?.0
        } else {
            butce.reuse_vec(old_grid, iw4 * ih4, INVALID_BLOCK)?
        };
        Ok(Self {
            dsp,
            grid,
            iw4,
            ih4,
            x_base: 0,
            y_base: 0,
            stride: iw4,
            frame: 0,
            seq: core::sync::atomic::AtomicU32::new(0),
            sign_bias: [0; 7],
            mfmv_sign: [0; 7],
            pocdiff: [0; 7],
            use_ref_frame_mvs: false,
            hdr_use_ref_frame_mvs: false,
            rp_proj: alloc::sync::Arc::new(Vec::new()),
            rp_stride: 0,
            gmv: [WarpedMotionParams::default(); 7],
            hp: false,
            force_integer_mv: false,
            mfmv: MfmvPlan::default(),
            temporal_hits: core::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Paralel tile çözümü için salt-okunur temporal projeksiyonu (`Arc::clone(&self.rp_proj)`)
    /// paylaşan, `grid`'i yalnız tile dikdörtgeni (`tile`) boyutunda sıfır-doldurulmuş
    /// bağımsız tile `RefMvs` örneği oluşturur (`need_grid == false` ise 0 bayt).
    pub(crate) fn fork_for_tile(
        &self,
        tile: TileBounds,
        need_grid: bool,
    ) -> crate::error::Result<Self> {
        let (grid, x_base, y_base, stride) = if need_grid {
            let x0 = tile.col_start.min(self.iw4);
            let y0 = tile.row_start.min(self.ih4);
            let x1 = tile.col_end.min(self.iw4);
            let y1 = tile.row_end.min(self.ih4);
            let s = x1.saturating_sub(x0).max(32);
            let rows = y1.saturating_sub(y0).max(32);
            (
                crate::frame::butce::dolu_vec(s * rows, INVALID_BLOCK)?,
                x0,
                y0,
                s,
            )
        } else {
            (Vec::new(), 0, 0, 0)
        };
        Ok(Self {
            dsp: self.dsp,
            grid,
            iw4: self.iw4,
            ih4: self.ih4,
            x_base,
            y_base,
            stride,
            frame: self.frame,
            seq: core::sync::atomic::AtomicU32::new(0),
            sign_bias: self.sign_bias,
            mfmv_sign: self.mfmv_sign,
            pocdiff: self.pocdiff,
            use_ref_frame_mvs: self.use_ref_frame_mvs,
            hdr_use_ref_frame_mvs: self.hdr_use_ref_frame_mvs,
            rp_proj: alloc::sync::Arc::clone(&self.rp_proj),
            rp_stride: self.rp_stride,
            gmv: self.gmv,
            hp: self.hp,
            force_integer_mv: self.force_integer_mv,
            mfmv: self.mfmv,
            temporal_hits: core::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Tile yerel `other` ızgarasının `[row_start..row_end) × [col_start..col_end)`
    /// dikdörtgenini ve `temporal_hits` sayacını kare ızgarasına birleştirir.
    #[allow(dead_code)]
    pub(crate) fn merge_tile_from(&mut self, other: &Self, tile: TileBounds) {
        if !other.grid.is_empty() && other.stride > 0 {
            let y0 = tile.row_start.min(self.ih4);
            let y1 = tile.row_end.min(self.ih4);
            let x0 = tile.col_start.min(self.iw4);
            let x1 = tile.col_end.min(self.iw4);
            if x0 < x1 {
                let w = x1 - x0;
                for y in y0..y1 {
                    let dst_off = y * self.iw4 + x0;
                    let src_off = (y - other.y_base) * other.stride + (x0 - other.x_base);
                    self.grid[dst_off..dst_off + w]
                        .copy_from_slice(&other.grid[src_off..src_off + w]);
                }
            }
        }
        // ORDERING: Bariyer sonrası tek iş parçacıklı birleştirme.
        let hits = other
            .temporal_hits
            .load(core::sync::atomic::Ordering::Relaxed);
        self.temporal_hits
            .fetch_add(hits, core::sync::atomic::Ordering::Relaxed);
    }

    /// Bir 4×4 hücrenin ızgara kaydı (`find_matching_ref` için).
    #[must_use]
    pub(crate) fn block_at(&self, x: usize, y: usize) -> RefBlock {
        if x >= self.iw4 || y >= self.ih4 || x < self.x_base || y < self.y_base {
            return INVALID_BLOCK;
        }
        self.grid[(y - self.y_base) * self.stride + (x - self.x_base)]
    }

    /// `dav1d_refmvs_init_frame` (`refmvs.c:833-931`) — temporal durum.
    ///
    /// `ref_poc[i]` = `f->refpoc[i]` (7 referansın `frame_offset`'ı),
    /// `ref_ref_poc[i][j]` = `i` referansının `refpoc[j]`'si,
    /// `ref_tmvs[i]` = `i` referansının kaydedilmiş 8×8 MV ızgarası (yoksa
    /// `None`; boyut uyuşmazlığı çağıranın sorumluluğundadır).
    #[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
    pub(crate) fn init_frame(
        &mut self,
        hdr: &FrameHeader,
        order_hint_n_bits: u8,
        ref_poc: &[u8; 7],
        ref_ref_poc: &[[u8; 7]; 7],
        ref_tmvs: &[Option<&[TemporalBlock]>; 7],
        old_rp_proj: Vec<TemporalBlock>,
        butce: &crate::frame::butce::Butce,
    ) -> crate::error::Result<()> {
        self.hp = hdr.hp;
        self.force_integer_mv = hdr.force_integer_mv;
        self.hdr_use_ref_frame_mvs = hdr.use_ref_frame_mvs;
        self.gmv = hdr.gmv;
        let poc = i32::from(hdr.frame_offset);
        for i in 0..7 {
            let d = get_poc_diff(order_hint_n_bits, i32::from(ref_poc[i]), poc);
            self.sign_bias[i] = u8::from(d > 0);
            self.mfmv_sign[i] = u8::from(d < 0);
            self.pocdiff[i] =
                get_poc_diff(order_hint_n_bits, poc, i32::from(ref_poc[i])).clamp(-31, 31) as i8;
        }

        // `rp_stride`, kare genişliği 128'e hizalanmış 8×8 sütun sayısıdır.
        self.rp_stride = ((hdr.width[0] as usize + 127) & !127) >> 3;
        let ih8 = (hdr.height as usize + 7) >> 3;
        self.rp_proj = alloc::sync::Arc::new(butce.reuse_vec(
            old_rp_proj,
            self.rp_stride * ih8.max(1),
            TemporalBlock::default(),
        )?);

        // Temporal MV kurulumu (`refmvs.c:889-931`).
        let mut plan = MfmvPlan::default();
        if hdr.use_ref_frame_mvs && order_hint_n_bits != 0 {
            let has = |i: usize| ref_tmvs[i].is_some();
            let diff = |a: u8, b: i32| get_poc_diff(order_hint_n_bits, i32::from(a), b);
            let mut total = 2usize;
            if has(0) && ref_ref_poc[0][6] != ref_poc[3] {
                plan.mfmv_ref[plan.n_mfmvs] = 0;
                plan.n_mfmvs += 1;
                total = 3;
            }
            if has(4) && diff(ref_poc[4], poc) > 0 {
                plan.mfmv_ref[plan.n_mfmvs] = 4;
                plan.n_mfmvs += 1;
            }
            if has(5) && diff(ref_poc[5], poc) > 0 {
                plan.mfmv_ref[plan.n_mfmvs] = 5;
                plan.n_mfmvs += 1;
            }
            if plan.n_mfmvs < total && has(6) && diff(ref_poc[6], poc) > 0 {
                plan.mfmv_ref[plan.n_mfmvs] = 6;
                plan.n_mfmvs += 1;
            }
            if plan.n_mfmvs < total && has(1) {
                plan.mfmv_ref[plan.n_mfmvs] = 1;
                plan.n_mfmvs += 1;
            }
            for n in 0..plan.n_mfmvs {
                let r = usize::from(plan.mfmv_ref[n]);
                let rpoc = i32::from(ref_poc[r]);
                let diff1 = get_poc_diff(order_hint_n_bits, rpoc, poc);
                if diff1.abs() > 31 {
                    plan.mfmv_ref2cur[n] = crate::decode::refmvs_tmvs::INVALID_REF2CUR;
                } else {
                    plan.mfmv_ref2cur[n] = if r < 4 { -diff1 } else { diff1 } as i8;
                    for m in 0..7 {
                        let rrpoc = i32::from(ref_ref_poc[r][m]);
                        let diff2 = get_poc_diff(order_hint_n_bits, rpoc, rrpoc);
                        // dav1d `refmvs.c:924-926`: `(unsigned) diff2 > 31U`
                        // **negatif** farkı da eler (unsigned karşılaştırma).
                        // `unsigned_abs()` kullanmak -N'i geçirir ve
                        // `mfmv_ref2ref` 251 gibi geçersiz bir `den` üretir.
                        plan.mfmv_ref2ref[n][m] = if (diff2 as u32) > 31 { 0 } else { diff2 as u8 };
                    }
                }
            }
        }
        self.mfmv = plan;
        self.use_ref_frame_mvs = plan.n_mfmvs > 0;
        Ok(())
    }

    /// `load_tmvs` (`refmvs.c:719-790`) — kare başına bir kez, bütün satırlar.
    ///
    /// Tek tile satırı kullandığımız için projeksiyon tamponu tam karedir;
    /// 16 satırlık pencereler sırayla doldurulur (dav1d `tile_row_idx = 0`).
    pub(crate) fn load_temporal(&mut self, refs: &[Option<&[TemporalBlock]>; 7]) {
        if !self.use_ref_frame_mvs {
            return;
        }
        {
            static CALL: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
            let call = CALL.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            trace::maybe_dump_refs(call, refs, self.rp_stride, self.iw4 >> 1);
        }
        let rp_proj = alloc::sync::Arc::make_mut(&mut self.rp_proj);
        let ih8 = rp_proj.len() / self.rp_stride.max(1);
        let mut row = 0usize;
        while row < ih8 {
            let end = (row + 16).min(ih8);
            // `load_tmvs` 16 satırlık **pencere** yazar (`(py & 15)`); tam
            // kare tamponunda pencere başını dilime kaydırarak mutlak satıra
            // eşleriz. Aksi hâlde 16. satırdan sonrası 0..15'e yazılırdı.
            let base = row * self.rp_stride;
            // dav1d iş parçacıksız yolda `col_start8=0`,
            // `col_end8 = bw >> 1` verir (`decode.c:3227-3228`); `iw8` de
            // kare genişliğidir — hizalı `rp_stride` **değildir**. Hizalı
            // değer sağ kenarda kaynak taramasını 13 sütun fazla yapar.
            let iw8 = self.iw4 >> 1;
            crate::dsp::profil_olc!(
                RefmvsLoadTmvs,
                (self.dsp.load_tmvs)(
                    &mut rp_proj[base..],
                    self.rp_stride,
                    iw8,
                    ih8.max(1),
                    &self.mfmv,
                    refs,
                    0,
                    0,
                    iw8,
                    row,
                    end,
                )
            );
            row = end;
        }
    }

    /// `dav1d_refmvs_save_tmvs` (`refmvs.h:141-159`) — kare sonunda 8×8 ızgara.
    pub(crate) fn save_temporal(&self, out: &mut [TemporalBlock], stride: usize) {
        let ih8 = (self.ih4 + 1) >> 1;
        let iw8 = (self.iw4 + 1) >> 1;
        let _ = ih8;
        crate::dsp::profil_olc!(
            RefmvsSaveTmvs,
            (self.dsp.save_tmvs)(
                out,
                stride,
                &self.grid,
                self.iw4,
                &self.mfmv_sign,
                iw8,
                (self.ih4 >> 1).max(1),
                0,
                0,
            )
        );
    }

    /// `splat_oneref_mv` (`decode.c:530-543`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn splat_inter(
        &mut self,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        bs: u8,
        mv: Mv,
        ref0: i8,
        interintra: bool,
        inter_mode: u8,
    ) {
        // `mf = (mode == GLOBALMV && imin(bw4,bh4) >= 2) | ((mode == NEWMV) * 2)`
        const GLOBALMV: u8 = 2;
        const NEWMV: u8 = 3;
        let tmpl = RefBlock {
            mv: [mv, Mv::ZERO],
            ref_pair: [ref0 + 1, if interintra { 0 } else { -1 }],
            bs,
            mf: (u8::from(inter_mode == GLOBALMV) & u8::from(bw4.min(bh4) >= 2))
                | ((u8::from(inter_mode == NEWMV)) << 1),
        };
        self.splat(bx4, by4, bw4, bh4, tmpl);
    }

    /// `splat_tworef_mv` (`decode.c:545-560`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn splat_tworef(
        &mut self,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        bs: u8,
        mv: [Mv; 2],
        ref0: i8,
        ref1: i8,
        comp_mode: u8,
    ) {
        // `mf = (mode == GLOBALMV_GLOBALMV) | !!((1 << mode) & 0xbc) * 2`
        const COMP_GLOBALMV_GLOBALMV: u8 = 6;
        let mf = u8::from(comp_mode == COMP_GLOBALMV_GLOBALMV)
            | ((u8::from((1u8 << comp_mode) & 0xbc != 0)) << 1);
        let tmpl = RefBlock {
            mv,
            ref_pair: [ref0 + 1, ref1 + 1],
            bs,
            mf,
        };
        self.splat(bx4, by4, bw4, bh4, tmpl);
    }

    /// Tanılama izi için kare sırasını bildirir (`dec.frame_no`).
    pub(crate) fn set_frame(&mut self, frame: u32) {
        self.frame = frame;
        self.seq.store(0, core::sync::atomic::Ordering::Relaxed);
    }

    #[inline]
    fn get(&self, x: usize, y: usize) -> RefBlock {
        self.grid[(y - self.y_base) * self.stride + (x - self.x_base)]
    }

    /// `splat_intrabc_mv` (`decode.c:530-543`).
    pub(crate) fn splat_intrabc(
        &mut self,
        bx4: usize,
        by4: usize,
        bw4: usize,
        bh4: usize,
        bs: u8,
        mv: Mv,
    ) {
        let tmpl = RefBlock {
            mv: [mv, Mv::ZERO],
            ref_pair: [0, -1],
            bs,
            mf: 0,
        };
        self.splat(bx4, by4, bw4, bh4, tmpl);
    }

    /// `splat_intraref` (`decode.c:562-574`).
    pub(crate) fn splat_intra(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize, bs: u8) {
        let tmpl = RefBlock {
            mv: [Mv::INVALID, Mv::ZERO],
            ref_pair: [0, -1],
            bs,
            mf: 0,
        };
        self.splat(bx4, by4, bw4, bh4, tmpl);
    }

    fn splat(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize, tmpl: RefBlock) {
        if self.grid.is_empty() || self.stride == 0 {
            return;
        }
        let x0 = bx4.min(self.iw4);
        let y0 = by4.min(self.ih4);
        let x1 = (bx4 + bw4).min(self.iw4);
        let y1 = (by4 + bh4).min(self.ih4);
        let lx0 = x0.saturating_sub(self.x_base);
        let ly0 = y0.saturating_sub(self.y_base);
        // F3c: `splat_mv` yuvası; bölge kare sınırına kırpılmıştır (kare
        // geneli ızgara, dav1d'deki 32+5 satırlık halka yerine).
        crate::dsp::profil_olc!(
            RefmvsSplatMv,
            (self.dsp.splat_mv)(
                &mut self.grid,
                self.stride,
                tmpl,
                lx0,
                ly0,
                x1.saturating_sub(x0),
                y1.saturating_sub(y0),
            )
        );
    }

    /// `add_spatial_candidate` (`refmvs.c:42-86`).
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn add_spatial_candidate(
        stack: &mut [Candidate; 8],
        cnt: &mut usize,
        weight: i32,
        b: RefBlock,
        ref_pair: [i8; 2],
        gmv: [Mv; 2],
        have_newmv: &mut i32,
        have_refmv: &mut i32,
    ) {
        if b.mv[0].is_invalid() {
            return;
        }
        if ref_pair[1] == -1 {
            for n in 0..2 {
                if b.ref_pair[n] == ref_pair[0] {
                    let cand_mv = if (b.mf & 1) != 0 && !gmv[0].is_invalid() {
                        gmv[0]
                    } else {
                        b.mv[n]
                    };
                    *have_refmv = 1;
                    *have_newmv |= i32::from(b.mf >> 1);
                    let last = *cnt;
                    for c in stack.iter_mut().take(last) {
                        if c.mv[0].n() == cand_mv.n() {
                            c.weight += weight;
                            return;
                        }
                    }
                    if last < 8 {
                        stack[last].mv[0] = cand_mv;
                        stack[last].weight = weight;
                        *cnt = last + 1;
                    }
                    return;
                }
            }
        } else if b.ref_pair == ref_pair {
            let g = |n: usize| {
                if (b.mf & 1) != 0 && !gmv[n].is_invalid() {
                    gmv[n]
                } else {
                    b.mv[n]
                }
            };
            let cand = [g(0), g(1)];
            *have_refmv = 1;
            *have_newmv |= i32::from(b.mf >> 1);
            let last = *cnt;
            for c in stack.iter_mut().take(last) {
                if c.mv[0].n() == cand[0].n() && c.mv[1].n() == cand[1].n() {
                    c.weight += weight;
                    return;
                }
            }
            if last < 8 {
                stack[last].mv = cand;
                stack[last].weight = weight;
                *cnt = last + 1;
            }
        }
    }

    /// `scan_row` (`refmvs.c:88-135`).
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn scan_row(
        &self,
        stack: &mut [Candidate; 8],
        cnt: &mut usize,
        ref_pair: [i8; 2],
        gmv: [Mv; 2],
        b: usize,
        by: usize,
        bw4: usize,
        w4: usize,
        max_rows: usize,
        step: usize,
        have_newmv: &mut i32,
        have_refmv: &mut i32,
    ) -> usize {
        let mut cand = self.get(b, by);
        let first_dim = &t::BLOCK_DIMENSIONS[usize::from(cand.bs)];
        let mut cand_bw4 = usize::from(first_dim[0]);
        let mut len = step.max(bw4.min(cand_bw4));
        if bw4 <= cand_bw4 {
            let weight = if bw4 == 1 {
                2
            } else {
                2.max((2 * max_rows).min(usize::from(first_dim[1])))
            };
            Self::add_spatial_candidate(
                stack,
                cnt,
                (len * weight) as i32,
                cand,
                ref_pair,
                gmv,
                have_newmv,
                have_refmv,
            );
            return weight >> 1;
        }
        let mut x = 0usize;
        loop {
            Self::add_spatial_candidate(
                stack,
                cnt,
                (len * 2) as i32,
                cand,
                ref_pair,
                gmv,
                have_newmv,
                have_refmv,
            );
            x += len;
            if x >= w4 {
                return 1;
            }
            cand = self.get(b + x, by);
            cand_bw4 = usize::from(t::BLOCK_DIMENSIONS[usize::from(cand.bs)][0]);
            len = step.max(cand_bw4);
        }
    }

    /// `scan_col` (`refmvs.c:138-176`).
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn scan_col(
        &self,
        stack: &mut [Candidate; 8],
        cnt: &mut usize,
        ref_pair: [i8; 2],
        gmv: [Mv; 2],
        bx: usize,
        by: usize,
        bh4: usize,
        h4: usize,
        max_cols: usize,
        step: usize,
        have_newmv: &mut i32,
        have_refmv: &mut i32,
    ) -> usize {
        let mut cand = self.get(bx, by);
        let first_dim = &t::BLOCK_DIMENSIONS[usize::from(cand.bs)];
        let mut cand_bh4 = usize::from(first_dim[1]);
        let mut len = step.max(bh4.min(cand_bh4));
        if bh4 <= cand_bh4 {
            let weight = if bh4 == 1 {
                2
            } else {
                2.max((2 * max_cols).min(usize::from(first_dim[0])))
            };
            Self::add_spatial_candidate(
                stack,
                cnt,
                (len * weight) as i32,
                cand,
                ref_pair,
                gmv,
                have_newmv,
                have_refmv,
            );
            return weight >> 1;
        }
        let mut y = 0usize;
        loop {
            Self::add_spatial_candidate(
                stack,
                cnt,
                (len * 2) as i32,
                cand,
                ref_pair,
                gmv,
                have_newmv,
                have_refmv,
            );
            y += len;
            if y >= h4 {
                return 1;
            }
            cand = self.get(bx, by + y);
            cand_bh4 = usize::from(t::BLOCK_DIMENSIONS[usize::from(cand.bs)][1]);
            len = step.max(cand_bh4);
        }
    }

    /// `dav1d_refmvs_find` (`refmvs.c:349-700`) — yalnız uzamsal yol.
    ///
    /// `intra_edge_flags`: `EDGE_I444_TOP_HAS_RIGHT` biti kullanılır.
    /// Dönüş: `(aday yığını, aday sayısı, bağlam)`.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    pub(crate) fn find(
        &self,
        ref_pair: [i8; 2],
        bs: u8,
        intra_edge_flags: u8,
        by4: usize,
        bx4: usize,
        tile: TileBounds,
    ) -> ([Candidate; 8], usize, usize) {
        let b_dim = &t::BLOCK_DIMENSIONS[usize::from(bs)];
        let bw4 = usize::from(b_dim[0]);
        let bh4 = usize::from(b_dim[1]);
        let w4 = bw4.min(16).min(tile.col_end - bx4);
        let h4 = bh4.min(16).min(tile.row_end - by4);
        // `tgmv`/`gmv` (`refmvs.c:358-380`): global motion vektörü; `gmv`
        // yalnız öteleme dışı türlerde geçerlidir (spatial aday yerine geçer).
        let mut tgmv = [Mv::ZERO, Mv::ZERO];
        let mut gmv = [Mv::INVALID, Mv::INVALID];
        for n in 0..2 {
            let r = ref_pair[n];
            if r > 0 {
                let wp = &self.gmv[usize::from(r as u8) - 1];
                tgmv[n] = get_gmv_2d(wp, bx4, by4, bw4, bh4, self.hp, self.force_integer_mv);
                gmv[n] = if wp.kind.raw() > WarpedMotionType::Translation.raw() {
                    tgmv[n]
                } else {
                    Mv::INVALID
                };
            }
        }
        if trace::enabled() {
            trace::maybe_dump_grid(self.frame, by4, bx4, &self.grid, self.iw4);
            trace::maybe_dump_tmvs(self.frame, by4, bx4, &self.rp_proj, self.rp_stride);
        }

        let mut stack = [Candidate::default(); 8];
        let mut cnt = 0usize;
        let mut have_newmv = 0i32;
        let mut have_col_mvs = 0i32;
        let mut have_row_mvs = 0i32;
        // `~0U` nöbetçisi: geçerli tarama sayısı yerine kullanılan `None`.
        let mut max_rows = 0usize;
        let mut n_rows: Option<usize> = None;
        let mut top_y = 0usize;
        if by4 > tile.row_start {
            max_rows = ((by4 - tile.row_start + 1) >> 1).min(2 + usize::from(bh4 > 1));
            top_y = by4 - 1;
            let step = if bw4 >= 16 { 4 } else { 1 };
            let r = self.scan_row(
                &mut stack,
                &mut cnt,
                ref_pair,
                gmv,
                bx4,
                top_y,
                bw4,
                w4,
                max_rows,
                step,
                &mut have_newmv,
                &mut have_row_mvs,
            );
            n_rows = Some(r);
        }
        let mut max_cols = 0usize;
        let mut n_cols: Option<usize> = None;
        if bx4 > tile.col_start {
            max_cols = ((bx4 - tile.col_start + 1) >> 1).min(2 + usize::from(bw4 > 1));
            let step = if bh4 >= 16 { 4 } else { 1 };
            let r = self.scan_col(
                &mut stack,
                &mut cnt,
                ref_pair,
                gmv,
                bx4 - 1,
                by4,
                bh4,
                h4,
                max_cols,
                step,
                &mut have_newmv,
                &mut have_col_mvs,
            );
            n_cols = Some(r);
        }

        // top/right
        if n_rows.is_some()
            && intra_edge_flags & crate::decode::edge_tree::EDGE_I444_TOP_HAS_RIGHT != 0
            && bw4.max(bh4) <= 16
            && bw4 + bx4 < tile.col_end
        {
            let b = self.get(bx4 + bw4, top_y);
            Self::add_spatial_candidate(
                &mut stack,
                &mut cnt,
                4,
                b,
                ref_pair,
                gmv,
                &mut have_newmv,
                &mut have_row_mvs,
            );
        }

        let nearest_match = (have_col_mvs + have_row_mvs) as usize;
        let nearest_cnt = cnt;
        for n in 0..nearest_cnt {
            stack[n].weight += 640;
        }

        // temporal adaylar (`refmvs.c:404-467`).
        // `globalmv_ctx` **başlık** bayrağından gelir (`refmvs.c:400`), plan
        // türetilmiş `use_ref_frame_mvs`'ten değil.
        let mut globalmv_ctx = usize::from(self.hdr_use_ref_frame_mvs);
        if self.use_ref_frame_mvs {
            let stride = self.rp_stride;
            let by8 = by4 >> 1;
            let bx8 = bx4 >> 1;
            let step_h = if bw4 >= 16 { 2 } else { 1 };
            let step_v = if bh4 >= 16 { 2 } else { 1 };
            let w8 = ((w4 + 1) >> 1).min(8);
            let h8 = ((h4 + 1) >> 1).min(8);
            // dav1d burada **koşu** (run) tabanlı bir tarama yapar
            // (`refmvs.c:404-443`): `run = &rb[0]`, iç döngü `x = step_h`'tan
            // başlar ama `rb[0]` koşunun başlangıcıdır ve ilk farklı hücrede
            // (veya sonda) `2 * run_len` ağırlığıyla **eklenir**. Rust önce
            // hücreleri tek tek `x = step_h`'tan başlayarak ekliyordu; bu,
            // pencerenin ilk hücresini `(by8, bx8)` **tamamen atlıyordu**
            // (T40: `test30` k2, by4=0 bx4=54, `n_mvs` 1 ↔ 2). Ayrıca
            // `globalmv_ctx` yanlış hücreden hesaplanıyordu. Burada koşu
            // mantığı birebir kurulur.
            let mut cells = [TemporalBlock::default(); 64];
            let mut cells_len = 0usize;
            let mut y = 0usize;
            while y < h8 {
                let row = (by8 + y) * stride + bx8;
                let mut x = 0usize;
                while x < w8 {
                    cells[cells_len] = self.rp_proj.get(row + x).copied().unwrap_or_default();
                    cells_len += 1;
                    x += step_h;
                }
                y += step_v;
            }
            let cells = &cells[..cells_len];
            let mut idx = 0usize;
            let mut first = true;
            while idx < cells.len() {
                let run = cells[idx];
                let mut len = 1usize;
                while idx + len < cells.len() {
                    let b = cells[idx + len];
                    // `refmvs.c:426-430`: INVALID_MV eşitliğinde `.ref`
                    // okunmaz (iki geçersiz hücre aynı koşudur).
                    if b.mv.n() == run.mv.n() && (b.mv.is_invalid() || b.ref_ == run.ref_) {
                        len += 1;
                    } else {
                        break;
                    }
                }
                let gctx = if first {
                    first = false;
                    Some(&mut globalmv_ctx)
                } else {
                    None
                };
                add_temporal_candidate(
                    self,
                    &mut stack,
                    &mut cnt,
                    run,
                    ref_pair,
                    gctx,
                    tgmv,
                    (2 * len) as i32,
                );
                idx += len;
            }
            // Sağ-alt/sol-alt köşe adayları (`refmvs.c:445-465`).
            if bw4.min(bh4) >= 2 && bw4.max(bh4) < 16 {
                let bh8 = bh4 >> 1;
                let bw8 = bw4 >> 1;
                let row = (by8 + bh8) * stride;
                let has_bottom = by8 + bh8 < (tile.row_end >> 1).min((by8 & !7) + 8);
                if has_bottom && bx8 > (tile.col_start >> 1).max(bx8 & !7) {
                    let rb = self.rp_proj.get(row + bx8 - 1).copied().unwrap_or_default();
                    add_temporal_candidate(self, &mut stack, &mut cnt, rb, ref_pair, None, tgmv, 2);
                }
                if bx8 + bw8 < (tile.col_end >> 1).min((bx8 & !7) + 8) {
                    if has_bottom {
                        let rb = self
                            .rp_proj
                            .get(row + bx8 + bw8)
                            .copied()
                            .unwrap_or_default();
                        add_temporal_candidate(
                            self, &mut stack, &mut cnt, rb, ref_pair, None, tgmv, 2,
                        );
                    }
                    let has_prev =
                        by8 + bh8 >= 1 && by8 + bh8 - 1 < (tile.row_end >> 1).min((by8 & !7) + 8);
                    if has_prev {
                        // `rb[bw8 - stride]` — `rb` tabanı `bx8`'i içerir
                        // (`refmvs.c:459-460`: `rbi[... + bx8]`). Taban
                        // düşerse yanlış sütun (x=1) okunur.
                        // **Tuzak:** dav1d `rb` işaretçisi `(by8 & 15)` 16
                        // satırlık pencereye göre kurulur ve `rb[bw8 -
                        // stride]` geri sarar (`refmvs.c:455-462`). Rust
                        // tam-kare tamponunda ilk `rb` satırı kare alt
                        // sınırında **tampon dışına** düşer; eski kod bunu
                        // `get(row + bx8 + bw8)` ile ön kontrol edip adayı
                        // **düşürüyordu** (T40: `test5883_3500` k5, kare alt
                        // satırı, ağırlık 2 ↔ 4). Doğru indeks doğrudan
                        // `(by8 + bh8 - 1) * stride + bx8 + bw8`'dir.
                        let rb = self
                            .rp_proj
                            .get(row + bx8 + bw8 - stride)
                            .copied()
                            .unwrap_or_default();
                        add_temporal_candidate(
                            self, &mut stack, &mut cnt, rb, ref_pair, None, tgmv, 2,
                        );
                    }
                }
            }
        }

        // top/left (sol-üst çapraz) — `refmvs.c:484-489`.
        //
        // **Tuzak:** koşul `(n_rows | n_cols) != ~0U`'dur ve `~0U` ile OR
        // her zaman `~0U` verdiği için bu, "**iki tarama da yapıldı**"
        // demektir (tek tarama yetmez). Yalnız üst taraması yapıldığında
        // (ör. tile sütununun ilk bloğu: `bx4 == tile_col.start`) dav1d
        // çaprazı **eklemez**; Rust önceden ekliyordu (T33, `test10103`
        // kare 0 · by4=16 · bx4=384: dav1d `n_mvs=0`, Rust `n_mvs=1`).
        if n_rows.is_some() && n_cols.is_some() {
            let mut dummy_newmv = 0i32;
            let b = if bx4 > 0 {
                self.get(bx4 - 1, by4 - 1)
            } else {
                INVALID_BLOCK
            };
            Self::add_spatial_candidate(
                &mut stack,
                &mut cnt,
                4,
                b,
                ref_pair,
                gmv,
                &mut dummy_newmv,
                &mut have_row_mvs,
            );
        }

        // "secondary" (komşu olmayan) üst/sol — 8x8 çözünürlükte.
        let mut have_dummy = 0i32;
        for n in 2..=3usize {
            if n_rows.is_none_or(|v| n > v) && n <= max_rows {
                // **Halka tampon tuzağı:** dav1d `rt->r` dizisini
                // `(by4 & 31)` ile adresler (`refmvs.c:493-499`); satır
                // işaretçileri `tile_sbrow_init`'te (`refmvs.c:682-712`)
                // SB satırına göre kurulur. `by4 & 31 == 0` olan SB
                // satırlarında (sb64'te her ikinci SB satırı) ifade
                // **negatiftir** ve halka tamponda "extra" satıra düşer:
                // mantıksal satır `by4 - 3`/`by4 - 5`'tir. Rust önceden
                // `.max(0)` ile **kare başındaki 0. satıra** kırpıyordu
                // (T33'te `by4=32` için yanlış aday kaynağı).
                let local = (by4 & 31) as isize - 2 * n as isize + 1;
                let by = ((by4 - (by4 & 31)) as isize + (local | 1)).max(0) as usize;
                let by = by.min(self.ih4 - 1);
                let bx = (bx4 | 1).min(self.iw4 - 1);
                let step = if bw4 >= 16 { 4 } else { 2 };
                let r = self.scan_row(
                    &mut stack,
                    &mut cnt,
                    ref_pair,
                    gmv,
                    bx,
                    by,
                    bw4,
                    w4,
                    (1 + max_rows).saturating_sub(n),
                    step,
                    &mut have_dummy,
                    &mut have_row_mvs,
                );
                let cur = n_rows.unwrap_or(0);
                n_rows = Some(cur + r);
            }
            if n_cols.is_none_or(|v| n > v) && n <= max_cols {
                let by = (by4 | 1).min(self.ih4 - 1);
                let localx = bx4 as isize - (n as isize) * 2 + 1;
                let bx = ((localx | 1).max(0) as usize).min(self.iw4 - 1);
                let step = if bh4 >= 16 { 4 } else { 2 };
                let r = self.scan_col(
                    &mut stack,
                    &mut cnt,
                    ref_pair,
                    gmv,
                    bx,
                    by,
                    bh4,
                    h4,
                    (1 + max_cols).saturating_sub(n),
                    step,
                    &mut have_dummy,
                    &mut have_col_mvs,
                );
                let cur = n_cols.unwrap_or(0);
                n_cols = Some(cur + r);
            }
        }

        // Tek referans: komşu olmayan referanslardan genişletilmiş adaylar
        // (`refmvs.c:640-666`).
        if ref_pair[1] == -1 && cnt < 2 && ref_pair[0] > 0 {
            let sign = self.sign_bias[usize::from(ref_pair[0] as u8) - 1];
            let sz4 = w4.min(h4);
            if n_rows.is_some() {
                let top_y = by4 - 1;
                let mut x = 0usize;
                while x < sz4 && cnt < 2 {
                    let b = self.get(bx4 + x, top_y);
                    add_single_extended_candidate(&mut stack, &mut cnt, b, sign, &self.sign_bias);
                    x += usize::from(t::BLOCK_DIMENSIONS[usize::from(b.bs)][0]);
                }
            }
            if n_cols.is_some() {
                let mut y = 0usize;
                while y < sz4 && cnt < 2 {
                    let b = self.get(bx4 - 1, by4 + y);
                    add_single_extended_candidate(&mut stack, &mut cnt, b, sign, &self.sign_bias);
                    y += usize::from(t::BLOCK_DIMENSIONS[usize::from(b.bs)][1]);
                }
            }
        }

        let ref_match_count = have_col_mvs + have_row_mvs;

        // bağlam kurulumu (`refmvs.c:549-567`)
        let (refmv_ctx, newmv_ctx) = match nearest_match {
            0 => {
                let refmv_ctx = ref_match_count.min(2) as usize;
                let newmv_ctx = usize::from(ref_match_count > 0);
                (refmv_ctx, newmv_ctx)
            }
            1 => {
                let refmv_ctx = ((ref_match_count * 3) as usize).min(4);
                let newmv_ctx = (3 - have_newmv) as usize;
                (refmv_ctx, newmv_ctx)
            }
            _ => (5, (5 - have_newmv) as usize),
        };
        let ctx = (refmv_ctx << 4) | (globalmv_ctx << 3) | newmv_ctx;

        // sıralama (`refmvs.c:570-601`): önce "nearest", sonra "secondary".
        let mut len = nearest_cnt;
        while len > 0 {
            let mut last = 0usize;
            for n in 1..len {
                if stack[n - 1].weight < stack[n].weight {
                    stack.swap(n - 1, n);
                    last = n;
                }
            }
            len = last;
        }
        let mut len = cnt;
        while len > nearest_cnt {
            let mut last = nearest_cnt;
            for n in (nearest_cnt + 1)..len {
                if stack[n - 1].weight < stack[n].weight {
                    stack.swap(n - 1, n);
                    last = n;
                }
            }
            len = last;
        }

        // Kırpma sınırları (`refmvs.c:613-626`).
        let left = -((bx4 + bw4 + 4) as i32) * 4 * 8;
        let right = ((self.iw4 - bx4 + 4) as i32) * 4 * 8;
        let top = -((by4 + bh4 + 4) as i32) * 4 * 8;
        let bottom = ((self.ih4 - by4 + 4) as i32) * 4 * 8;

        let mut ctx = ctx;
        if ref_pair[1] > 0 {
            // Compound: genişletilmiş adaylar (`refmvs.c:603-666`).
            if cnt < 2 {
                let sign0 = self.sign_bias[usize::from(ref_pair[0] as u8) - 1];
                let sign1 = self.sign_bias[usize::from(ref_pair[1] as u8) - 1];
                let sz4 = w4.min(h4);
                let mut same = [Candidate::default(); 4];
                let mut same_count = [0usize; 4];
                if n_rows.is_some() {
                    let mut x = 0usize;
                    while x < sz4 {
                        let b = self.get(bx4 + x, by4 - 1);
                        add_compound_extended_candidate(
                            &mut same,
                            &mut same_count,
                            b,
                            sign0,
                            sign1,
                            ref_pair,
                            &self.sign_bias,
                        );
                        x += usize::from(t::BLOCK_DIMENSIONS[usize::from(b.bs)][0]);
                    }
                }
                if n_cols.is_some() {
                    let mut y = 0usize;
                    while y < sz4 {
                        let b = self.get(bx4 - 1, by4 + y);
                        add_compound_extended_candidate(
                            &mut same,
                            &mut same_count,
                            b,
                            sign0,
                            sign1,
                            ref_pair,
                            &self.sign_bias,
                        );
                        y += usize::from(t::BLOCK_DIMENSIONS[usize::from(b.bs)][1]);
                    }
                }
                // "same" ve "diff" adaylarını birleştir.
                for n in 0..2 {
                    let mut m = same_count[n];
                    if m >= 2 {
                        continue;
                    }
                    let l = same_count[2 + n];
                    if l != 0 {
                        same[m].mv[n] = same[2].mv[n];
                        m += 1;
                        if m == 2 {
                            continue;
                        }
                        if l == 2 {
                            same[1].mv[n] = same[3].mv[n];
                            continue;
                        }
                    }
                    while m < 2 {
                        same[m].mv[n] = tgmv[n];
                        m += 1;
                    }
                }
                // C kodu adayları doğrudan `mvstack[cnt..]` üzerinde tutar;
                // burada yerel `same` dizisi kullanıldığı için son değerler
                // dav1d'nin yazdığı yerlere birebir taşınır.
                if cnt == 0 {
                    stack[0].mv = same[0].mv;
                    stack[1].mv = same[1].mv;
                    stack[0].weight = 2;
                    stack[1].weight = 2;
                } else {
                    let eq = stack[0].mv[0].n() == same[0].mv[0].n()
                        && stack[0].mv[1].n() == same[0].mv[1].n();
                    stack[1].mv = if eq { same[1].mv } else { same[0].mv };
                    stack[1].weight = 2;
                }
                cnt = 2;
            }
            for c in stack.iter_mut().take(cnt) {
                c.mv[0].x = clip_i32(i32::from(c.mv[0].x), left, right) as i16;
                c.mv[0].y = clip_i32(i32::from(c.mv[0].y), top, bottom) as i16;
                c.mv[1].x = clip_i32(i32::from(c.mv[1].x), left, right) as i16;
                c.mv[1].y = clip_i32(i32::from(c.mv[1].y), top, bottom) as i16;
            }
            ctx = match refmv_ctx >> 1 {
                0 => newmv_ctx.min(1),
                1 => 1 + newmv_ctx.min(3),
                _ => clip_i32(3 + newmv_ctx as i32, 4, 7) as usize,
            };
        } else {
            for c in stack.iter_mut().take(cnt) {
                c.mv[0].x = clip_i32(i32::from(c.mv[0].x), left, right) as i16;
                c.mv[0].y = clip_i32(i32::from(c.mv[0].y), top, bottom) as i16;
            }
            for c in stack.iter_mut().skip(cnt).take(2 - cnt.min(2)) {
                c.mv[0] = tgmv[0];
            }
        }

        if trace::enabled() {
            let seq = self.seq.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            trace::emit(
                self.frame,
                seq,
                by4,
                bx4,
                bs,
                ref_pair,
                intra_edge_flags,
                &stack,
                cnt.min(8),
                ctx,
                tile,
            );
        }

        (stack, cnt.min(8), ctx)
    }
}

/// `fix_int_mv_precision` (`env.h:462-466`).
#[inline]
fn fix_int_mv_precision(mv: &mut Mv) {
    let x = i32::from(mv.x);
    let y = i32::from(mv.y);
    mv.x = ((x - (x >> 15) + 3) & !7) as i16;
    mv.y = ((y - (y >> 15) + 3) & !7) as i16;
}

/// `fix_mv_precision` (`env.h:467-474`).
#[inline]
fn fix_mv_precision(hdr: (bool, bool), mv: &mut Mv) {
    let (force_integer_mv, hp) = hdr;
    if force_integer_mv {
        fix_int_mv_precision(mv);
    } else if !hp {
        let x = i32::from(mv.x);
        let y = i32::from(mv.y);
        mv.x = ((x - (x >> 15)) & !1) as i16;
        mv.y = ((y - (y >> 15)) & !1) as i16;
    }
}

/// `get_gmv_2d` (`env.h:476-522`).
pub(crate) fn get_gmv_2d(
    gmv: &WarpedMotionParams,
    bx4: usize,
    by4: usize,
    bw4: usize,
    bh4: usize,
    hp: bool,
    force_integer_mv: bool,
) -> Mv {
    match gmv.kind {
        WarpedMotionType::Identity => Mv::ZERO,
        WarpedMotionType::Translation => {
            let mut res = Mv {
                y: (gmv.matrix[0] >> 13) as i16,
                x: (gmv.matrix[1] >> 13) as i16,
            };
            if force_integer_mv {
                fix_int_mv_precision(&mut res);
            }
            res
        }
        WarpedMotionType::RotZoom | WarpedMotionType::Affine => {
            let x = (bx4 * 4 + bw4 * 2) as i32 - 1;
            let y = (by4 * 4 + bh4 * 2) as i32 - 1;
            let xc = (gmv.matrix[2] - (1 << 16)).wrapping_mul(x)
                + gmv.matrix[3].wrapping_mul(y)
                + gmv.matrix[0];
            let yc = (gmv.matrix[5] - (1 << 16)).wrapping_mul(y)
                + gmv.matrix[4].wrapping_mul(x)
                + gmv.matrix[1];
            let shift = 16 - (3 - i32::from(!hp));
            let round = (1 << shift) >> 1;
            let sgn = |v: i32| if v < 0 { -1 } else { 1 };
            let mut res = Mv {
                y: (sgn(yc) * (((yc.abs() + round) >> shift) << i32::from(!hp))) as i16,
                x: (sgn(xc) * (((xc.abs() + round) >> shift) << i32::from(!hp))) as i16,
            };
            if force_integer_mv {
                fix_int_mv_precision(&mut res);
            }
            res
        }
    }
}

/// `add_compound_extended_candidate` (`refmvs.c:240-288`).
///
/// `same` dört adaylık yereldir: `[0..2]` "same" (bir bileşen), `[2..4]`
/// "diff" (öteki bileşen) adayları. `same_count` `[same0, same1, diff0,
/// diff1]` sayarlar.
fn add_compound_extended_candidate(
    same: &mut [Candidate; 4],
    same_count: &mut [usize; 4],
    cand_b: RefBlock,
    sign0: u8,
    sign1: u8,
    ref_pair: [i8; 2],
    sign_bias: &[u8; 7],
) {
    for n in 0..2 {
        let cand_ref = cand_b.ref_pair[n];
        if cand_ref <= 0 {
            break;
        }
        let mut cand_mv = cand_b.mv[n];
        if cand_ref == ref_pair[0] {
            if same_count[0] < 2 {
                same[same_count[0]].mv[0] = cand_mv;
                same_count[0] += 1;
            }
            if same_count[3] < 2 {
                if sign1 ^ sign_bias[usize::from(cand_ref as u8) - 1] != 0 {
                    cand_mv.y = -cand_mv.y;
                    cand_mv.x = -cand_mv.x;
                }
                same[2 + same_count[3]].mv[1] = cand_mv;
                same_count[3] += 1;
            }
        } else if cand_ref == ref_pair[1] {
            if same_count[1] < 2 {
                same[same_count[1]].mv[1] = cand_mv;
                same_count[1] += 1;
            }
            if same_count[2] < 2 {
                if sign0 ^ sign_bias[usize::from(cand_ref as u8) - 1] != 0 {
                    cand_mv.y = -cand_mv.y;
                    cand_mv.x = -cand_mv.x;
                }
                same[2 + same_count[2]].mv[0] = cand_mv;
                same_count[2] += 1;
            }
        } else {
            let i_cand_mv = Mv {
                y: -cand_mv.y,
                x: -cand_mv.x,
            };
            if same_count[2] < 2 {
                same[2 + same_count[2]].mv[0] =
                    if sign0 ^ sign_bias[usize::from(cand_ref as u8) - 1] != 0 {
                        i_cand_mv
                    } else {
                        cand_mv
                    };
                same_count[2] += 1;
            }
            if same_count[3] < 2 {
                same[2 + same_count[3]].mv[1] =
                    if sign1 ^ sign_bias[usize::from(cand_ref as u8) - 1] != 0 {
                        i_cand_mv
                    } else {
                        cand_mv
                    };
                same_count[3] += 1;
            }
        }
    }
}

/// `add_single_extended_candidate` (`refmvs.c:290-320`).
fn add_single_extended_candidate(
    stack: &mut [Candidate; 8],
    cnt: &mut usize,
    b: RefBlock,
    sign: u8,
    sign_bias: &[u8; 7],
) {
    for n in 0..2 {
        let cand_ref = b.ref_pair[n];
        if cand_ref <= 0 {
            break;
        }
        let mut cand_mv = b.mv[n];
        if sign ^ sign_bias[usize::from(cand_ref as u8) - 1] != 0 {
            cand_mv.y = -cand_mv.y;
            cand_mv.x = -cand_mv.x;
        }
        let last = *cnt;
        let mut m = 0usize;
        while m < last {
            if cand_mv.n() == stack[m].mv[0].n() {
                break;
            }
            m += 1;
        }
        if m == last && last < 8 {
            stack[last].mv[0] = cand_mv;
            stack[last].weight = 2;
            *cnt = last + 1;
        }
    }
}

/// `add_temporal_candidate` (`refmvs.c:194-238`) — tek referans dalı.
#[allow(clippy::too_many_arguments)]
fn add_temporal_candidate(
    rf: &RefMvs,
    stack: &mut [Candidate; 8],
    cnt: &mut usize,
    rb: TemporalBlock,
    ref_pair: [i8; 2],
    globalmv_ctx: Option<&mut usize>,
    tgmv: [Mv; 2],
    weight: i32,
) {
    if rb.mv == Mv::INVALID || rb.mv.is_invalid() {
        return;
    }
    if rb.ref_ == 0 || rb.ref_ > 31 {
        return;
    }
    let num0 = i32::from(rf.pocdiff[usize::from(ref_pair[0] as u8) - 1]);
    let mut mv0 = mv_projection(rb.mv, num0, i32::from(rb.ref_));
    fix_mv_precision((rf.force_integer_mv, rf.hp), &mut mv0);
    rf.temporal_hits
        .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    if ref_pair[1] == -1 {
        if let Some(ctx) = globalmv_ctx {
            *ctx = usize::from((mv0.x.abs_diff(tgmv[0].x) | mv0.y.abs_diff(tgmv[0].y)) >= 16);
        }
        let last = *cnt;
        for c in stack.iter_mut().take(last) {
            if c.mv[0].n() == mv0.n() {
                c.weight += weight;
                return;
            }
        }
        if last < 8 {
            stack[last].mv[0] = mv0;
            stack[last].weight = weight;
            *cnt = last + 1;
        }
    } else {
        let num1 = i32::from(rf.pocdiff[usize::from(ref_pair[1] as u8) - 1]);
        let mut mv1 = mv_projection(rb.mv, num1, i32::from(rb.ref_));
        fix_mv_precision((rf.force_integer_mv, rf.hp), &mut mv1);
        let last = *cnt;
        for c in stack.iter_mut().take(last) {
            if c.mv[0].n() == mv0.n() && c.mv[1].n() == mv1.n() {
                c.weight += weight;
                return;
            }
        }
        if last < 8 {
            stack[last].mv = [mv0, mv1];
            stack[last].weight = weight;
            *cnt = last + 1;
        }
    }
}

/// Tanılama: `dav1d_refmvs_find` çağrı izi — **varsayılan kapalı**.
///
/// `AV1OXIDE_REFMVS_TRACE` ortam değişkeni yoksa hiçbir şey yazılmaz; varlığı
/// bir kez (`OnceLock`) okunur, çağrı başına maliyet tek atomik yüktür.
/// Biçim `scripts/oracle/refmvs_trace.c` `V`/`W` satırlarıyla aynıdır:
///
/// ```text
/// V <kare> <sıra> <by4> <bx4> <bs> <ref0> <ref1> <kenar> <n_mvs> <ctx> <trs> <tre> <tcs> <tce>
/// W <kare> <sıra> <i> <mvy> <mvx> <ağırlık>
/// ```
///
/// `W` satırları yığının **ilk iki** girdisini yazar (`decode.c:1279-1291`
/// yalnız `mvstack[0]`/`mvstack[1]` okur). dav1d `mv.mv[1]` alanını tek
/// referans yolunda yazmadığı için (başlatılmamış) karşılaştırılmaz.
mod trace {
    use super::{Candidate, Mv, RefBlock, TileBounds};
    #[cfg(feature = "std")]
    use std::sync::OnceLock;

    /// `<kare>:<by4>:<bx4>:<satır0>:<satır1>`
    #[cfg(feature = "std")]
    type GridFilter = (u32, usize, usize, usize, usize);

    #[cfg(feature = "std")]
    static ON: OnceLock<bool> = OnceLock::new();
    #[cfg(feature = "std")]
    static FRAME: OnceLock<Option<u32>> = OnceLock::new();
    #[cfg(feature = "std")]
    static GRID: OnceLock<Option<GridFilter>> = OnceLock::new();
    #[cfg(feature = "std")]
    static TMVS: OnceLock<Option<(usize, usize)>> = OnceLock::new();
    #[cfg(feature = "std")]
    static BY: OnceLock<Option<usize>> = OnceLock::new();
    #[cfg(feature = "std")]
    static REFS: OnceLock<Option<(usize, usize)>> = OnceLock::new();
    #[cfg(feature = "std")]
    static BX: OnceLock<Option<usize>> = OnceLock::new();

    /// İz açık mı? (`AV1OXIDE_REFMVS_TRACE`)
    #[must_use]
    pub(super) fn enabled() -> bool {
        #[cfg(feature = "std")]
        {
            *ON.get_or_init(|| std::env::var_os("AV1OXIDE_REFMVS_TRACE").is_some())
        }
        #[cfg(not(feature = "std"))]
        {
            false
        }
    }

    #[cfg(feature = "std")]
    fn frame_filter() -> Option<u32> {
        *FRAME.get_or_init(|| {
            std::env::var("AV1OXIDE_REFMVS_FRAME")
                .ok()
                .and_then(|v| v.parse().ok())
        })
    }

    #[cfg(feature = "std")]
    fn by_filter() -> Option<usize> {
        *BY.get_or_init(|| std::env::var("AV1OXIDE_REFMVS_BY").ok()?.parse().ok())
    }

    #[cfg(feature = "std")]
    fn bx_filter() -> Option<usize> {
        *BX.get_or_init(|| std::env::var("AV1OXIDE_REFMVS_BX").ok()?.parse().ok())
    }

    #[cfg(feature = "std")]
    fn grid_filter() -> Option<GridFilter> {
        *GRID.get_or_init(|| {
            let v = std::env::var("AV1OXIDE_REFMVS_GRID").ok()?;
            let mut it = v.split(':').map(|p| p.parse::<u32>().ok());
            let frame = it.next()??;
            let by4 = it.next()?? as usize;
            let bx4 = it.next()?? as usize;
            let row0 = it.next()?? as usize;
            let row1 = it.next()?? as usize;
            Some((frame, by4, bx4, row0, row1))
        })
    }

    #[cfg(feature = "std")]
    fn refs_filter() -> Option<(usize, usize)> {
        *REFS.get_or_init(|| {
            let v = std::env::var("AV1OXIDE_REFMVS_REFS").ok()?;
            let mut it = v.split(':').map(|p| p.parse::<usize>().ok());
            Some((it.next()??, it.next()??))
        })
    }

    /// Referans karelerin kaydedilmiş 8×8 MV ızgaraları
    /// (`AV1OXIDE_REFMVS_REFS=<satır0>:<satır1>`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn maybe_dump_refs(
        call: u32,
        refs: &[Option<&[super::TemporalBlock]>; 7],
        stride: usize,
        iw8: usize,
    ) {
        #[cfg(feature = "std")]
        {
            let Some((row0, row1)) = refs_filter() else {
                return;
            };
            for (i, r) in refs.iter().enumerate() {
                let Some(r) = r else { continue };
                for row in row0..=row1 {
                    for x in 0..iw8 {
                        let Some(b) = r.get(row * stride + x) else {
                            continue;
                        };
                        crate::trace_eprintln!(
                            "S {call} {i} {row} {x} {} {} {}",
                            b.mv.y,
                            b.mv.x,
                            b.ref_
                        );
                    }
                }
            }
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (call, refs, stride, iw8);
        }
    }

    #[cfg(feature = "std")]
    fn tmvs_filter() -> Option<(usize, usize)> {
        *TMVS.get_or_init(|| {
            let v = std::env::var("AV1OXIDE_REFMVS_TMVS").ok()?;
            let mut it = v.split(':').map(|p| p.parse::<usize>().ok());
            Some((it.next()??, it.next()??))
        })
    }

    /// Temporal projeksiyon dökümü — `AV1OXIDE_REFMVS_TMVS=<satır0>:<satır1>`.
    ///
    /// `AV1OXIDE_REFMVS_FRAME`/`BY`/`BX` süzgeçleriyle aynı çağrıda yazılır;
    /// biçim `refmvs_trace.c` `T` satırıyla aynıdır:
    /// `T <satır> <x> <mvy> <mvx> <ref>`.
    pub(super) fn maybe_dump_tmvs(
        frame: u32,
        by4: usize,
        bx4: usize,
        rp_proj: &[super::TemporalBlock],
        stride: usize,
    ) {
        #[cfg(feature = "std")]
        {
            let Some((row0, row1)) = tmvs_filter() else {
                return;
            };
            if frame_filter().is_some_and(|f| f != frame) {
                return;
            }
            if by_filter().is_some_and(|y| y != by4) {
                return;
            }
            if bx_filter().is_some_and(|x| x != bx4) {
                return;
            }
            let rows = rp_proj.len() / stride.max(1);
            for row in row0..=row1.min(rows.saturating_sub(1)) {
                for x in 0..stride {
                    let b = rp_proj[row * stride + x];
                    crate::trace_eprintln!("T {row} {x} {} {} {}", b.mv.y, b.mv.x, b.ref_);
                }
            }
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (frame, by4, bx4, rp_proj, stride);
        }
    }

    /// Mantıksal ızgara dökümü — `AV1OXIDE_REFMVS_GRID=<kare>:<by4>:<bx4>:<satır0>:<satır1>`.
    ///
    /// Biçim `refmvs_trace.c` `G` satırıyla aynıdır:
    /// `G <satır> <x> <mvy> <mvx> <ref0> <ref1> <bs> <mf>`. Yalnız filtre
    /// eşleştiğinde yazar; aksi hâlde hiçbir maliyeti yoktur (iki `OnceLock`
    /// yüklemesi).
    pub(super) fn maybe_dump_grid(
        frame: u32,
        by4: usize,
        bx4: usize,
        grid: &[RefBlock],
        iw4: usize,
    ) {
        #[cfg(feature = "std")]
        {
            let Some((gf, gby, gbx, row0, row1)) = grid_filter() else {
                return;
            };
            if frame_filter().is_some_and(|f| f != frame) {
                return;
            }
            if gf != frame || gby != by4 || gbx != bx4 {
                return;
            }
            for row in row0..=row1.min(grid.len() / iw4 - 1) {
                for x in 0..iw4 {
                    let b = grid[row * iw4 + x];
                    crate::trace_eprintln!(
                        "G {row} {x} {} {} {} {} {} {}",
                        b.mv[0].y,
                        b.mv[0].x,
                        b.ref_pair[0],
                        b.ref_pair[1],
                        b.bs,
                        b.mf
                    );
                }
            }
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (frame, by4, bx4, grid, iw4);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit(
        frame: u32,
        seq: u32,
        by4: usize,
        bx4: usize,
        bs: u8,
        ref_pair: [i8; 2],
        edge_flags: u8,
        stack: &[Candidate; 8],
        cnt: usize,
        ctx: usize,
        tile: TileBounds,
    ) {
        #[cfg(feature = "std")]
        {
            if frame_filter().is_some_and(|f| f != frame) {
                return;
            }
            crate::trace_eprintln!(
                "V {frame} {seq} {by4} {bx4} {bs} {} {} {edge_flags} {cnt} {ctx} \
                 {} {} {} {}",
                ref_pair[0],
                ref_pair[1],
                tile.row_start,
                tile.row_end,
                tile.col_start,
                tile.col_end
            );
            for (i, c) in stack.iter().take(2).enumerate() {
                let mv: Mv = c.mv[0];
                // `mv[1]` dav1d'de tek referans yolunda yazılmaz; karşılaştırılmaz.
                crate::trace_eprintln!("W {frame} {seq} {i} {} {} {}", mv.y, mv.x, c.weight);
            }
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = (
                frame, seq, by4, bx4, bs, ref_pair, edge_flags, stack, cnt, ctx, tile,
            );
            let _: Option<Mv> = None;
        }
    }
}

/// `iclip` (`common/intops.h`).
#[inline]
fn clip_i32(v: i32, min: i32, max: i32) -> i32 {
    if v < min {
        min
    } else if v > max {
        max
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `union mv.n` paketlemesi: `y` alt, `x` üst 16 bit.
    #[test]
    fn mv_paketleme_dav1d_ile_ayni() {
        assert_eq!(Mv { y: 0, x: 0 }.n(), 0);
        assert_eq!(Mv { y: -1, x: 2 }.n(), (2 << 16) | 0xffff);
        assert_eq!(Mv::INVALID.n(), 0x8000_8000u32 as i32);
        assert!(Mv::INVALID.is_invalid());
        assert!(!Mv::ZERO.is_invalid());
    }

    /// `iclip` sınırları.
    #[test]
    fn kirpma_sinirlari() {
        assert_eq!(clip_i32(-10, -3, 7), -3);
        assert_eq!(clip_i32(10, -3, 7), 7);
        assert_eq!(clip_i32(5, -3, 7), 5);
    }

    /// `splat_intrabc_mv` yazılan blok aday olur; `splat_intraref` aday olmaz.
    #[test]
    fn splat_intrabc_aday_uretir_intraref_uretmez() {
        let mv = Mv { y: 8, x: -16 };
        let tile = TileBounds {
            row_start: 0,
            row_end: 8,
            col_start: 0,
            col_end: 8,
        };
        let mut rf = RefMvs::new(
            32,
            32,
            crate::dsp::refmvs::RefmvsDsp::SCALAR,
            &crate::frame::butce::Butce::sinirsiz(),
        )
        .expect("ızgara");
        // 8x8 (bs=4) intrabc bloğu (0,0)'a yazılır; (2,0)'daki 4x4 blok
        // sol komşusundan aday alır.
        rf.splat_intrabc(0, 0, 2, 2, 4, mv);
        let (stack, cnt, _ctx) = rf.find([0, -1], 21, 0, 0, 2, tile);
        assert_eq!(cnt, 1, "intrabc komşusu aday olmalı");
        assert_eq!(stack[0].mv[0], mv);
        // Aynı ızgaraya intraref yazılırsa aday kalmaz.
        rf.splat_intra(0, 0, 2, 2, 4);
        let (_, cnt, _) = rf.find([0, -1], 21, 0, 0, 2, tile);
        assert_eq!(cnt, 0, "intraref aday üretmemeli");
    }

    /// Sol-üst çapraz adayı **iki tarama da** yapıldığında eklenir
    /// (`refmvs.c:486`: `(n_rows | n_cols) != ~0U` — `~0U` ile OR her zaman
    /// `~0U` olduğu için bu "VE" demektir). Tile sütununun ilk bloğunda
    /// (`bx4 == tile_col.start`) sol tarama yoktur → çapraz **eklenmez**
    /// (T33'te `test10103` kare 0 · by4=16 · bx4=384'te `n_mvs` 0 ↔ 1 idi).
    #[test]
    fn capraz_aday_iki_tarama_da_gerekir() {
        let mv = Mv { y: 8, x: -16 };
        let mut rf = RefMvs::new(
            32,
            32,
            crate::dsp::refmvs::RefmvsDsp::SCALAR,
            &crate::frame::butce::Butce::sinirsiz(),
        )
        .expect("ızgara"); // iw4 = 8, ih4 = 8
        // Sol-üst çapraz: (by4-1, bx4-1) = (3, 3).
        rf.splat_intrabc(2, 2, 2, 2, 17, mv);
        // Tile sütunu bx4'te başlıyor: sol tarama yok.
        let tile = TileBounds {
            row_start: 0,
            row_end: 8,
            col_start: 4,
            col_end: 8,
        };
        let (_, cnt, _) = rf.find([0, -1], 17, 0, 4, 4, tile);
        assert_eq!(cnt, 0, "sol tarama yokken çapraz eklenmemeli");
        // Tile sütunu 0'dan başlarsa iki tarama da yapılır → çapraz eklenir.
        let tile0 = TileBounds {
            row_start: 0,
            row_end: 8,
            col_start: 0,
            col_end: 8,
        };
        let (stack, cnt, _) = rf.find([0, -1], 17, 0, 4, 4, tile0);
        assert_eq!(cnt, 1, "iki tarama varken çapraz eklenmeli");
        assert_eq!(stack[0].mv[0], mv);
    }

    /// Üst-sağ adayı `have_row_mvs`'i **yazar** (`refmvs.c:408`); Rust sahte
    /// sayaç kullandığı için `nearest_match` ve `ctx` kayıyordu
    /// (T33'te `test10103` kare 0 · by4=16 · bx4=1024: `ctx` 85 ↔ 51).
    #[test]
    fn ust_sag_aday_have_row_mvs_yazar() {
        let mv = Mv { y: -4, x: 12 };
        let mut rf = RefMvs::new(
            32,
            32,
            crate::dsp::refmvs::RefmvsDsp::SCALAR,
            &crate::frame::butce::Butce::sinirsiz(),
        )
        .expect("ızgara"); // iw4 = 8, ih4 = 8
        // Üst-sağ: (by4-1, bx4+bw4) = (3, 2).
        rf.splat_intrabc(2, 3, 2, 2, 17, mv);
        let tile = TileBounds {
            row_start: 0,
            row_end: 8,
            col_start: 0,
            col_end: 8,
        };
        // EDGE_I444_TOP_HAS_RIGHT = 1 << 0.
        let (stack, cnt, ctx) = rf.find([0, -1], 17, 1, 4, 0, tile);
        assert_eq!(cnt, 1, "üst-sağ aday eklenmeli");
        assert_eq!(stack[0].mv[0], mv);
        // nearest_match = have_col + have_row = 0 + 1 = 1;
        // ref_match_count = 1 → refmv_ctx = min(3,4) = 3, newmv_ctx = 3.
        assert_eq!(
            ctx,
            (3 << 4) | 3,
            "ctx üst-sağın have_row_mvs'i ile kurulur"
        );
    }

    /// İkincil satır taraması `by4 & 31 == 0` iken **`by4-3`** mantıksal
    /// satırını okur (`refmvs.c:493-499` + halka eşlemesi `:682-712`).
    /// Rust `.max(0)` ile kare başındaki 0. satıra kırpıyordu.
    #[test]
    fn ikincil_satir_halka_eslemesi() {
        // Kırpma sınırları (bx4=0, bw4=1): x ∈ [-160, 640], y ∈ [-1184, 128].
        let mv_row0 = Mv { y: 8, x: 16 };
        let mv_row29 = Mv { y: -8, x: -16 };
        let mut rf = RefMvs::new(
            64,
            128,
            crate::dsp::refmvs::RefmvsDsp::SCALAR,
            &crate::frame::butce::Butce::sinirsiz(),
        )
        .expect("ızgara"); // iw4 = 16, ih4 = 32
        rf.splat_intrabc(0, 0, 2, 2, 17, mv_row0);
        rf.splat_intrabc(0, 29, 2, 2, 17, mv_row29);
        let tile = TileBounds {
            row_start: 0,
            row_end: 32,
            col_start: 0,
            col_end: 16,
        };
        // 4x4 blok (by4=32, bx4=0): bw4 = bh4 = 1 → max_rows = 2,
        // üst tarama n_rows = 1 verir; n = 2 için ikincil satır = 32-3 = 29.
        let (stack, cnt, _) = rf.find([0, -1], 21, 0, 32, 0, tile);
        assert_eq!(cnt, 1, "ikincil satır adayı bulunmalı");
        assert_eq!(
            stack[0].mv[0], mv_row29,
            "by4=32 için ikincil satır 29 olmalı (0 değil)"
        );
    }
}

#[cfg(test)]
mod tasma_testleri {
    use super::{Mv, get_gmv_2d};
    use crate::bitstream::frame_header::{WarpedMotionParams, WarpedMotionType};

    /// P38 taşma sınıfı: global hareket izdüşümü (`get_gmv_2d`, dav1d
    /// `refmvs.c` `get_gmv_2d`) AV1 sınırında `i32`'de taşmaz. Bit akışının
    /// verebildiği en uç matris (dav1d `read_gm_params`: `mat[2]−2^16`,
    /// `mat[3]` ∈ [−2^13, 2^13], öteleme ∈ [−2^22, 2^22]) ve 65 536 piksellik
    /// karenin son bloğu: |xc| ≤ 2·2^13·65 595 + 2^22 < 2^31. Robust profilde
    /// (`overflow-checks`) taşma panik olurdu.
    #[test]
    fn tasma_gmv_izdusumu_av1_sinirinda_tasmaz() {
        for isaret in [1i32, -1] {
            let gmv = WarpedMotionParams {
                kind: WarpedMotionType::Affine,
                matrix: [
                    isaret * (1 << 22),
                    isaret * (1 << 22),
                    (1 << 16) + isaret * (1 << 13),
                    isaret * (1 << 13),
                    isaret * (1 << 13),
                    (1 << 16) + isaret * (1 << 13),
                ],
            };
            for hp in [false, true] {
                let mv: Mv = get_gmv_2d(&gmv, 16_383, 16_383, 32, 32, hp, false);
                let _ = (mv.x, mv.y);
            }
        }
    }
}
