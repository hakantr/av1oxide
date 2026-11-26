//! Loop restoration (LR) — S2c.2b / P20: sözdizimi ve kare üzerine uygulama.
//!
//! dav1d `src/decode.c` LR sözdizimi ve `src/lr_apply_tmpl.c` /
//! `src/lf_apply_tmpl.c` uygulama kurallarının birebir karşılığı. Çekirdekler
//! ([`crate::dsp::lr`]) bu modülde **değil**; burada yalnız birim
//! parametreleri, şerit/birim gezinmesi, kenar bayrakları ve şerit sınırı
//! satırları kurulur.
//!
//! ## dav1d eşlemesi (dav1d @ `99f3354`)
//!
//! | Rust | dav1d |
//! |---|---|
//! | [`LrUnit`] | `lf_mask.h:42-48` (`Av1RestorationUnit`) |
//! | [`LrState::mask`] | `internal.h:299` (`lr_mask`, `Av1Restoration.lr[3][4]`) |
//! | `LrState::new` `restore_planes` | `decode.c:3025-3040` |
//! | [`LrState::tile_init`] | `decode.c:2470-2515` (`lr_ref` varsayılanları) |
//! | [`LrState::read_sb`] | `decode.c:2669-2720` (SB başında birim kapsamı) |
//! | [`read_unit`] | `decode.c:2517-2580` (`read_restoration_info`) |
//! | `SGR_PARAMS` (`dsp::lr_tables`) | `tables.c:415-420` (`dav1d_sgr_params`) |
//!
//! Superres'li karede (`width[0] != width[1]`, S2c.3) birim ızgarası ve
//! uygulama **büyütülmüş** genişliktedir (`sr_sb128w`, `sr_cur`); tile başı
//! ve SB başı birim kapsamı dav1d'nin ayrı superres dallarıyla hesaplanır
//! (`decode.c:2484-2496`, `2681-2699`).
#![forbid(unsafe_code)]

use alloc::vec;
use alloc::vec::Vec;

use crate::bitstream::frame_header::{FrameHeader, RestorationType};
use crate::decode::intra::{PicBuf, ToolCounts};
use crate::dsp::lr::{LrBitDepth, LrEdgeFlags, LrFilterFn, LrParams};
use crate::entropy::cdf::CdfContext;
use crate::entropy::cdf_layout as layout;
use crate::entropy::msac::MsacContext;
use crate::frame::{BitDepth, ChromaFormat};

/// Birim türü `NONE` (`DAV1D_RESTORATION_NONE`).
pub(crate) const LR_NONE: u8 = 0;
/// Birim türü `WIENER` (`DAV1D_RESTORATION_WIENER`).
pub(crate) const LR_WIENER: u8 = 2;
/// Birim türü `SGRPROJ` tabanı; birimde `SGRPROJ + sgr_idx` saklanır
/// (`lf_mask.h:43`, `decode.c:2560`).
pub(crate) const LR_SGRPROJ: u8 = 3;

/// `dav1d_sgr_params[16][2]` (`tables.c:415-420`) — tek kaynak [`crate::dsp::lr_tables`].
pub(crate) use crate::dsp::lr_tables::SGR_PARAMS;

/// dav1d `Av1RestorationUnit` (`lf_mask.h:42-48`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LrUnit {
    /// `NONE` / `WIENER` / `SGRPROJ + sgr_idx`.
    pub(crate) kind: u8,
    /// Yatay Wiener katsayıları (dış → iç).
    pub(crate) filter_h: [i8; 3],
    /// Dikey Wiener katsayıları (dış → iç).
    pub(crate) filter_v: [i8; 3],
    /// SGR ağırlıkları (okunan `w0`, `w1`).
    pub(crate) sgr_weights: [i8; 2],
}

/// Tile başında `lr_ref` varsayılanları (`decode.c:2500-2507`).
const LR_REF_DEFAULT_FILTER: [i8; 3] = [3, -7, 15];
const LR_REF_DEFAULT_SGR: [i8; 2] = [-32, 31];

/// Paralel tile çözümünde tile başına kaydedilen LR ızgara işlemi.
#[derive(Clone, Copy, Debug)]
pub(crate) enum LrTileOp {
    /// Tile başında `mask[sb_idx][p][unit_idx]` yuvasına varsayılan `filter_h`,
    /// `filter_v`, `sgr_weights` yazımı (`tile_init`).
    TileInit {
        sb_idx: usize,
        p: usize,
        unit_idx: usize,
    },
    /// SB başında okunan `LrUnit`'in `mask[sb_idx][p][unit_idx]` yuvasına yazımı (`read_sb`).
    SetUnit {
        sb_idx: usize,
        p: usize,
        unit_idx: usize,
        unit: LrUnit,
    },
}

/// Kare düzeyi LR durumu: 128×128 başına `lr[3][4]` birim ızgarası ve tile
/// başına `lr_ref` (dav1d `ts->lr_ref[3]`).
///
/// dav1d `lr_ref` bir işaretçidir; tile başında kare ızgarasındaki bir
/// yuvayı gösterir ve o yuvaya varsayılanları **yazar**. Son okunan
/// Wiener/SGR birimini gösterdikten sonra yalnız değerleri okunur. Bu yüzden
/// burada değer kopyası tutulur; tile başındaki yazım da ızgaraya aynen
/// yapılır (yuva bu karede okunmazsa varsayılanlar ızgarada kalır).
pub(crate) struct LrState {
    /// `f->lf.restore_planes` (bit p = düzlem p).
    pub(crate) restore_planes: u8,
    /// `f->sr_sb128w` (superres kapalıyken `sb128w`).
    pub(crate) sb128w: usize,
    /// `f->lf.lr_mask` (`sr_sb128w * sb128h`).
    pub(crate) mask: Vec<[[LrUnit; 4]; 3]>,
    /// Paralel tile modunda kaydedilen işlemler (`None` = doğrudan `mask`'a yaz).
    tile_ops: Option<Vec<LrTileOp>>,
    /// `ts->lr_ref` (değer kopyası).
    lr_ref: [LrUnit; 3],
    /// Kare türü (düzlem başına `restoration.type`).
    kind: [RestorationType; 3],
    /// `restoration.unit_size` (log2; [0] luma, [1] kroma).
    unit_size: [u8; 2],
    /// Büyütülmüş kare genişliği (`width[1]`, `sr_cur.p.p.w`; LR bu
    /// genişlikte çalışır) ve yüksekliği (luma piksel).
    width: usize,
    height: usize,
    /// Kodlanan genişlik (`width[0]`, `cur.p.w`).
    coded_width: usize,
    /// Superres paydası (`width_scale_denominator`); `0` = superres yok.
    sr_denom: usize,
    /// Düzlem 1/2 için alt örnekleme.
    ss_hor: usize,
    ss_ver: usize,
}

impl LrState {
    /// Havuzdan geri dönüştürülmüş `old_mask` varsa kapasitesi eşleştiğinde
    /// yeniden kullanarak `LrState` kurar (`F8b-1`).
    pub(crate) fn new_with_reuse(
        hdr: &FrameHeader,
        chroma: ChromaFormat,
        old_mask: Vec<[[LrUnit; 4]; 3]>,
        butce: &crate::frame::butce::Butce,
    ) -> crate::error::Result<Option<Self>> {
        let r = &hdr.restoration;
        let restore_planes = u8::from(r.kind[0] != RestorationType::None)
            | (u8::from(r.kind[1] != RestorationType::None) << 1)
            | (u8::from(r.kind[2] != RestorationType::None) << 2);
        if restore_planes == 0 {
            return Ok(None);
        }
        // `f->sr_sb128w = (sr_cur.p.p.w + 127) >> 7` (`decode.c:3025`).
        let width = hdr.width[1] as usize;
        let coded_width = hdr.width[0] as usize;
        let height = hdr.height as usize;
        let sb128w = (width + 127) >> 7;
        let sr_denom = if coded_width != width {
            usize::from(hdr.super_res.width_scale_denominator)
        } else {
            0
        };
        let sb128h = ((((height + 7) >> 3) << 1) + 31) >> 5;
        let (ss_hor, ss_ver) = chroma.chroma_shift();
        Ok(Some(Self {
            restore_planes,
            sb128w,
            mask: butce.reuse_vec(
                old_mask,
                sb128w * sb128h.max(1),
                [[LrUnit::default(); 4]; 3],
            )?,
            tile_ops: None,
            lr_ref: [LrUnit::default(); 3],
            kind: r.kind,
            unit_size: r.unit_size,
            width,
            height,
            coded_width,
            sr_denom,
            ss_hor: ss_hor as usize,
            ss_ver: ss_ver as usize,
        }))
    }

    /// Paralel tile çözümü için bağımsız `lr_ref` ve işlem günlüğü (`tile_ops`)
    /// taşıyan hafif `LrState` kopyası oluşturur (`mask` ayırmaz).
    #[must_use]
    pub(crate) fn fork_for_tile(&self) -> Self {
        Self {
            restore_planes: self.restore_planes,
            sb128w: self.sb128w,
            mask: Vec::new(),
            tile_ops: Some(Vec::new()),
            lr_ref: [LrUnit::default(); 3],
            kind: self.kind,
            unit_size: self.unit_size,
            width: self.width,
            height: self.height,
            coded_width: self.coded_width,
            sr_denom: self.sr_denom,
            ss_hor: self.ss_hor,
            ss_ver: self.ss_ver,
        }
    }

    /// Paralel tile çözümünde kaydedilen `other.tile_ops` işlemlerini kare
    /// `self.mask` ızgarasına `(tile_row, tile_col)` sırasıyla uygular.
    pub(crate) fn merge_tile_from(&mut self, other: &Self) {
        let Some(ops) = other.tile_ops.as_ref() else {
            return;
        };
        for &op in ops {
            match op {
                LrTileOp::TileInit {
                    sb_idx,
                    p,
                    unit_idx,
                } => {
                    if let Some(slot) = self.mask.get_mut(sb_idx) {
                        slot[p][unit_idx].filter_v = LR_REF_DEFAULT_FILTER;
                        slot[p][unit_idx].filter_h = LR_REF_DEFAULT_FILTER;
                        slot[p][unit_idx].sgr_weights = LR_REF_DEFAULT_SGR;
                    }
                }
                LrTileOp::SetUnit {
                    sb_idx,
                    p,
                    unit_idx,
                    unit,
                } => {
                    if let Some(slot) = self.mask.get_mut(sb_idx) {
                        slot[p][unit_idx] = unit;
                    }
                }
            }
        }
    }

    /// Düzlem `p` için (yatay, dikey) alt örnekleme (`decode.c:2673-2674`):
    /// `ss_ver = p && I420`, `ss_hor = p && !I444`.
    fn plane_ss(&self, p: usize) -> (usize, usize) {
        if p == 0 {
            (0, 0)
        } else {
            (self.ss_hor, self.ss_ver)
        }
    }

    /// Tile başı (`decode.c:2470-2515`, superres kapalı dal): `lr_ref`
    /// tile'ın ilk 64×64'ünün yuvasını gösterir ve varsayılanlarla dolar.
    ///
    /// `row_start`/`col_start` 4×4 blok birimindedir (`ts->tiling`).
    pub(crate) fn tile_init(&mut self, row_start: usize, col_start: usize) {
        let superres = self.sr_denom != 0;
        let (sb_idx, unit_idx) = if superres {
            // Yalnız dikey bileşen (`decode.c:2474-2476`).
            ((row_start >> 5) * self.sb128w, (row_start & 16) >> 3)
        } else {
            (
                (row_start >> 5) * self.sb128w + (col_start >> 5),
                ((row_start & 16) >> 3) + ((col_start & 16) >> 4),
            )
        };
        for p in 0..3 {
            if (self.restore_planes >> p) & 1 == 0 {
                continue;
            }
            let (sb_idx, unit_idx) = if superres {
                // `decode.c:2484-2496`: tile'ın ilk sütunu büyütülmüş birim
                // ızgarasına yuvarlanır. Ilk birim kare ızgarası dışındaysa
                // `lr_ref` **değişmez** (dav1d `continue`).
                let (ss_hor, _) = self.plane_ss(p);
                let d = self.sr_denom;
                let unit_size_log2 = usize::from(self.unit_size[usize::from(p != 0)]);
                let rnd = (8usize << unit_size_log2) - 1;
                let shift = unit_size_log2 + 3;
                let x = (((4 * col_start * d) >> ss_hor) + rnd) >> shift;
                let px_x = x << (unit_size_log2 + ss_hor);
                let u_idx = unit_idx + ((px_x & 64) >> 6);
                let sb128x = px_x >> 7;
                if sb128x >= self.sb128w {
                    continue;
                }
                (sb_idx + sb128x, u_idx)
            } else {
                (sb_idx, unit_idx)
            };
            let mut def = self
                .mask
                .get(sb_idx)
                .map_or(LrUnit::default(), |s| s[p][unit_idx]);
            def.filter_v = LR_REF_DEFAULT_FILTER;
            def.filter_h = LR_REF_DEFAULT_FILTER;
            def.sgr_weights = LR_REF_DEFAULT_SGR;
            if let Some(ops) = self.tile_ops.as_mut() {
                ops.push(LrTileOp::TileInit {
                    sb_idx,
                    p,
                    unit_idx,
                });
            } else if let Some(slot) = self.mask.get_mut(sb_idx) {
                slot[p][unit_idx] = def;
            }
            self.lr_ref[p] = def;
        }
    }

    /// SB başında birim kapsamı ve okuma (`decode.c:2669-2720`, superres
    /// kapalı dal). `bx`/`by` 4×4 blok birimindedir; `decode_sb`'den
    /// **önce** çağrılır. Okunan birim sayısını türe göre `counts`'a ekler.
    pub(crate) fn read_sb(
        &mut self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        bx: usize,
        by: usize,
        sb_step: usize,
        counts: &mut LrSyntaxCounts,
    ) {
        for p in 0..3 {
            if (self.restore_planes >> p) & 1 == 0 {
                continue;
            }
            let (ss_hor, ss_ver) = self.plane_ss(p);
            let unit_size_log2 = u32::from(self.unit_size[usize::from(p != 0)]);
            let y = (by * 4) >> ss_ver;
            let h = (self.height + ss_ver) >> ss_ver;
            let unit_size = 1usize << unit_size_log2;
            let mask = unit_size - 1;
            if y & mask != 0 {
                continue;
            }
            let half_unit = unit_size >> 1;
            // Kare sınırında yarım birim yukarı yuvarlanır (birden çok birim
            // varsa son birim öncekine katılır).
            if y != 0 && y + half_unit > h {
                continue;
            }
            let frame_type = self.kind[p];
            if self.sr_denom != 0 {
                // Superres dalı (`decode.c:2681-2699`): SB'nin kapsadığı
                // büyütülmüş birimler `[x0, min(x1, n_units))`.
                let w = (self.width + ss_hor) >> ss_hor;
                let n_units = ((w + half_unit) >> unit_size_log2).max(1);
                let d = self.sr_denom;
                let rnd = unit_size * 8 - 1;
                let shift = unit_size_log2 + 3;
                let x0 = (((4 * bx * d) >> ss_hor) + rnd) >> shift;
                let x1 = (((4 * (bx + sb_step) * d) >> ss_hor) + rnd) >> shift;
                for x in x0..x1.min(n_units) {
                    let px_x = x << (unit_size_log2 as usize + ss_hor);
                    let sb_idx = (by >> 5) * self.sb128w + (px_x >> 7);
                    let unit_idx = ((by & 16) >> 3) + ((px_x & 64) >> 6);
                    let unit = read_unit(msac, cdf, &mut self.lr_ref[p], p, frame_type);
                    counts.add(frame_type, unit.kind);
                    if let Some(ops) = self.tile_ops.as_mut() {
                        ops.push(LrTileOp::SetUnit {
                            sb_idx,
                            p,
                            unit_idx,
                            unit,
                        });
                    } else if let Some(slot) = self.mask.get_mut(sb_idx) {
                        slot[p][unit_idx] = unit;
                    }
                }
                continue;
            }
            let x = (4 * bx) >> ss_hor;
            if x & mask != 0 {
                continue;
            }
            let w = (self.coded_width + ss_hor) >> ss_hor;
            if x != 0 && x + half_unit > w {
                continue;
            }
            let sb_idx = (by >> 5) * self.sb128w + (bx >> 5);
            let unit_idx = ((by & 16) >> 3) + ((bx & 16) >> 4);
            let unit = read_unit(msac, cdf, &mut self.lr_ref[p], p, frame_type);
            counts.add(frame_type, unit.kind);
            if let Some(ops) = self.tile_ops.as_mut() {
                ops.push(LrTileOp::SetUnit {
                    sb_idx,
                    p,
                    unit_idx,
                    unit,
                });
            } else if let Some(slot) = self.mask.get_mut(sb_idx) {
                slot[p][unit_idx] = unit;
            }
        }
    }
}

/// Sözdizimi sayaçları (tanılama; davranışı değiştirmez).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LrSyntaxCounts {
    /// Okunan birim sayısı.
    pub(crate) units: u64,
    /// `SWITCHABLE` karede okunan birim.
    pub(crate) switchable_units: u64,
    /// Türü `NONE` çıkan birim.
    pub(crate) none_units: u64,
    /// Wiener birimi.
    pub(crate) wiener_units: u64,
    /// SGR birimi.
    pub(crate) sgr_units: u64,
}

impl LrSyntaxCounts {
    fn add(&mut self, frame_type: RestorationType, kind: u8) {
        self.units += 1;
        if frame_type == RestorationType::Switchable {
            self.switchable_units += 1;
        }
        match kind {
            LR_NONE => self.none_units += 1,
            LR_WIENER => self.wiener_units += 1,
            _ => self.sgr_units += 1,
        }
    }
}

/// `read_restoration_info` (`decode.c:2517-2580`).
///
/// Wiener katsayıları ve SGR ağırlıkları `lr_ref`'e göre alt-üssel kodlanır;
/// türü Wiener ya da SGR olan birim yeni `lr_ref` olur (`NONE` değiştirmez).
pub(crate) fn read_unit(
    msac: &mut MsacContext<'_>,
    cdf: &mut CdfContext,
    lr_ref: &mut LrUnit,
    p: usize,
    frame_type: RestorationType,
) -> LrUnit {
    let mut lr = LrUnit::default();
    if frame_type == RestorationType::Switchable {
        let filter = msac.decode_symbol_adapt(
            &mut cdf.cells_mut()[layout::CDF_m + layout::MODE_restore_switchable..],
            2,
        ) as u8;
        // NONE / WIENER / SGRPROJ (`decode.c:2529`).
        lr.kind = filter + u8::from(filter != 0);
    } else {
        let off = if frame_type == RestorationType::Wiener {
            layout::MODE_restore_wiener
        } else {
            layout::MODE_restore_sgrproj
        };
        let ty = msac.decode_bool_adapt(&mut cdf.cells_mut()[layout::CDF_m + off..]);
        lr.kind = if ty != 0 { frame_type.raw() } else { LR_NONE };
    }

    if lr.kind == LR_WIENER {
        let r = *lr_ref;
        lr.filter_v[0] = if p != 0 {
            0
        } else {
            (msac.decode_subexp(i32::from(r.filter_v[0]) + 5, 16, 1) - 5) as i8
        };
        lr.filter_v[1] = (msac.decode_subexp(i32::from(r.filter_v[1]) + 23, 32, 2) - 23) as i8;
        lr.filter_v[2] = (msac.decode_subexp(i32::from(r.filter_v[2]) + 17, 64, 3) - 17) as i8;
        lr.filter_h[0] = if p != 0 {
            0
        } else {
            (msac.decode_subexp(i32::from(r.filter_h[0]) + 5, 16, 1) - 5) as i8
        };
        lr.filter_h[1] = (msac.decode_subexp(i32::from(r.filter_h[1]) + 23, 32, 2) - 23) as i8;
        lr.filter_h[2] = (msac.decode_subexp(i32::from(r.filter_h[2]) + 17, 64, 3) - 17) as i8;
        lr.sgr_weights = r.sgr_weights;
        *lr_ref = lr;
    } else if lr.kind == LR_SGRPROJ {
        let r = *lr_ref;
        let idx = msac.decode_bools(4) as usize;
        let sgr = SGR_PARAMS[idx];
        lr.kind += idx as u8;
        lr.sgr_weights[0] = if sgr[0] != 0 {
            (msac.decode_subexp(i32::from(r.sgr_weights[0]) + 96, 128, 4) - 96) as i8
        } else {
            0
        };
        lr.sgr_weights[1] = if sgr[1] != 0 {
            (msac.decode_subexp(i32::from(r.sgr_weights[1]) + 32, 128, 4) - 32) as i8
        } else {
            95
        };
        lr.filter_v = r.filter_v;
        lr.filter_h = r.filter_h;
        *lr_ref = lr;
    }
    lr
}

/// LR çekirdek tablosu (dav1d `Dav1dLoopRestorationDSPContext`):
/// `wiener[0]` 7 dokunuş, `wiener[1]` 5 dokunuş; `sgr[0]` 5×5, `sgr[1]` 3×3,
/// `sgr[2]` karma (`looprestoration_tmpl.c:1360-1365`).
pub(crate) struct LrKernels {
    pub(crate) wiener: [LrFilterFn; 2],
    pub(crate) sgr: [LrFilterFn; 3],
}

/// LR çağrı izi (tanılama; varsayılan kapalı).
///
/// `AV1OXIDE_LR_TRACE` varsa her çekirdek çağrısından sonra bir satır
/// yazılır; `AV1OXIDE_LR_TRACE_FRAME` kare, `AV1OXIDE_LR_TRACE_PLANE` düzlem
/// süzgecidir. Biçim `scripts/oracle/lr_trace.c` ile **aynıdır**:
/// `L <kare> <düzlem> <x> <y> <w> <h> <tür> <p0..p7> <kenar> <fnv>`.
#[derive(Clone, Copy)]
struct TraceCfg {
    frame: Option<u32>,
    plane: Option<usize>,
}

fn trace_cfg() -> Option<TraceCfg> {
    #[cfg(feature = "std")]
    {
        static CFG: std::sync::OnceLock<Option<TraceCfg>> = std::sync::OnceLock::new();
        *CFG.get_or_init(|| {
            std::env::var_os("AV1OXIDE_LR_TRACE")?;
            let frame = std::env::var("AV1OXIDE_LR_TRACE_FRAME")
                .ok()
                .and_then(|v| v.parse().ok());
            let plane = std::env::var("AV1OXIDE_LR_TRACE_PLANE")
                .ok()
                .and_then(|v| v.parse().ok());
            Some(TraceCfg { frame, plane })
        })
    }
    #[cfg(not(feature = "std"))]
    {
        None
    }
}

impl TraceCfg {
    fn on(&self, frame_no: u32, plane: usize) -> bool {
        self.frame.is_none_or(|f| f == frame_no) && self.plane.is_none_or(|p| p == plane)
    }
}

/// FNV-1a 64 — `lr_trace.c` ile aynı (çağrı sonrası bölge özeti).
fn fnv_block(data: &[u8], stride: usize, w: usize, h: usize, bpp: usize) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for y in 0..h {
        for &v in &data[y * stride..y * stride + w * bpp] {
            hash ^= u64::from(v);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

/// Deblocking sonrası, CDEF öncesi LR şerit sınırı satırları yedeği (`dav1d_copy_lpf`).
///
/// Tam kare kopyası yerine yalnız `(restore_planes >> p) & 1 != 0` olan düzlemlerde
/// her şerit sınırı `b_k = (56 >> ss_ver) + k * (64 >> ss_ver) < plane_h` için 4 satır
/// (`b_k - 2`, `b_k - 1`, `b_k`, `(b_k + 1).min(plane_h - 1)`) saklar (`4.A.1`).
#[derive(Default)]
pub(crate) struct LrCdefBackup {
    pub(crate) data: [Vec<u8>; 3],
    pub(crate) stride: [usize; 3],
    pub(crate) n_boundaries: [usize; 3],
}

impl LrCdefBackup {
    /// Deblocking sonrası `pic` düzlemlerinden yalnız LR şerit sınırı satırlarını yakalar.
    pub(crate) fn capture(
        pic: &PicBuf,
        restore_planes: u8,
        old_data: [Vec<u8>; 3],
        butce: &crate::frame::butce::Butce,
    ) -> crate::error::Result<Self> {
        let mut old_iter = old_data.into_iter();
        let mut data = [Vec::new(), Vec::new(), Vec::new()];
        let mut stride = [0usize; 3];
        let mut n_boundaries = [0usize; 3];
        for p in 0..3 {
            let old_p = old_iter.next().unwrap_or_default();
            if (restore_planes >> p) & 1 == 0 {
                let _ = butce.reuse_vec(old_p, 0, 0u8)?;
                continue;
            }
            let ss_ver = if p == 0 { 0 } else { pic.ss_ver };
            let plane_h = if p == 0 { pic.height } else { pic.ch };
            let first_b = 56 >> ss_ver;
            let step_b = 64 >> ss_ver;
            let n_b = if plane_h > first_b {
                1 + (plane_h - 1 - first_b) / step_b
            } else {
                0
            };
            let s = pic.stride[p];
            let mut buf = butce.reuse_vec(old_p, 4 * n_b * s, 0u8)?;
            for k in 0..n_b {
                let b = first_b + k * step_b;
                let src_rows = [b - 2, b - 1, b, (b + 1).min(plane_h - 1)];
                for (r, &sy) in src_rows.iter().enumerate() {
                    let doff = (4 * k + r) * s;
                    let soff = sy * s;
                    buf[doff..doff + s].copy_from_slice(&pic.data[p][soff..soff + s]);
                }
            }
            data[p] = buf;
            stride[p] = s;
            n_boundaries[p] = n_b;
        }
        Ok(Self {
            data,
            stride,
            n_boundaries,
        })
    }

    /// Yedek tamponlarının toplam bayt boyutu.
    #[must_use]
    pub(crate) fn bayt(&self) -> usize {
        self.data.iter().map(Vec::len).sum()
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn copy_top_slices(
        pre: &[u8],
        ps: usize,
        y: usize,
        ss_ver: usize,
        lpf: &mut [u8],
        stride: usize,
        col_lo: usize,
        col_hi: usize,
    ) {
        let k = (y - (56 >> ss_ver)) >> (6 - ss_ver);
        for r in 0..2 {
            let so = (4 * k + r) * ps;
            lpf[r * stride + col_lo..r * stride + col_hi]
                .copy_from_slice(&pre[so + col_lo..so + col_hi]);
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn copy_bottom_slices(
        pre: &[u8],
        ps: usize,
        b: usize,
        ss_ver: usize,
        lpf: &mut [u8],
        stride: usize,
        col_lo: usize,
        col_hi: usize,
    ) {
        let k = (b - (56 >> ss_ver)) >> (6 - ss_ver);
        for r in 0..2 {
            let so = (4 * k + 2 + r) * ps;
            lpf[(6 + r) * stride + col_lo..(6 + r) * stride + col_hi]
                .copy_from_slice(&pre[so + col_lo..so + col_hi]);
        }
    }
}

/// Kare düzeyi uygulama bağlamı (bir karelik).
struct Apply<'a> {
    st: &'a LrState,
    /// Deblocking sonrası, CDEF öncesi şerit sınırı satırları (`dav1d_copy_lpf`).
    pre_cdef: &'a LrCdefBackup,
    kernels: &'a LrKernels,
    bpp: usize,
    lbd: LrBitDepth,
    hbd: bool,
    sb128: usize,
    /// `f->sbh`.
    sbh: usize,
    trace: Option<TraceCfg>,
    frame_no: u32,
    /// P38: `lpf` şerit tamponunun bütçe defteri (geçici ayırma).
    butce: &'a crate::frame::butce::Butce,
}

/// Kare üzerine LR uygular (`dav1d_lr_sbrow`, `lr_apply_tmpl.c:169-204`).
///
/// dav1d LR'yi SB satırı başına uygular; burada da aynı sıra izlenir
/// (SB satırı → Y, U, V → birim → şerit). Şerit içi pikseller CDEF
/// çıktısıdır (`pic`, yerinde), şerit sınırının üstündeki/altındaki iki
/// satır `pre_cdef`'ten (deblocking sonrası, CDEF öncesi) okunur — dav1d
/// `backup_lpf` (`lf_apply_tmpl.c:41-102`) yedeğiyle birebir aynı değerler.
/// Sol sütunlar önceki birimin **LR öncesi** son dört sütunudur
/// (`backup4xU`, `lr_apply_tmpl.c:100-105`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply(
    pic: &mut PicBuf,
    pre_cdef: &mut LrCdefBackup,
    st: &LrState,
    sb128: bool,
    bd: BitDepth,
    kernels: &LrKernels,
    tools: &mut ToolCounts,
    frame_no: u32,
    butce: &crate::frame::butce::Butce,
    pool: &crate::thread::WorkerPool,
) -> crate::error::Result<()> {
    let lbd = match bd {
        BitDepth::Eight => LrBitDepth::B8,
        BitDepth::Ten => LrBitDepth::B10,
        BitDepth::Twelve => LrBitDepth::B12,
    };
    let sb128_u = usize::from(sb128);
    let sb_shift = 4 + sb128_u;
    let bh4 = ((st.height + 7) >> 3) << 1;
    let sbh = (bh4 + (1 << sb_shift) - 1) >> sb_shift;
    let trace = trace_cfg();
    if st.restore_planes & 1 != 0 {
        tools.lr_unit_y = 1u64 << st.unit_size[0];
    }
    if st.restore_planes & 6 != 0 {
        tools.lr_unit_uv = 1u64 << st.unit_size[1];
    }

    if pool.is_parallel() && sbh > 1 && trace.is_none() {
        return apply_parallel_sbrows(
            pic, pre_cdef, st, sb128_u, sbh, bd, lbd, kernels, tools, butce, pool,
        );
    }

    let a = Apply {
        st,
        pre_cdef,
        kernels,
        bpp: bd.bytes_per_sample(),
        lbd,
        hbd: bd != BitDepth::Eight,
        sb128: sb128_u,
        sbh,
        trace,
        frame_no,
        butce,
    };
    for sby in 0..sbh {
        let offset_y = if sby != 0 { 8 } else { 0 };
        let not_last = usize::from(sby + 1 < sbh);
        if st.restore_planes & 1 != 0 {
            let h = st.height;
            let w = st.width;
            let next_row_y = (sby + 1) << (6 + sb128_u);
            let row_h = (next_row_y - 8 * not_last).min(h);
            let y_stripe = (sby << (6 + sb128_u)) - offset_y;
            a.lr_sbrow(pic, tools, y_stripe, w, h, row_h, 0)?;
        }
        if st.restore_planes & 6 != 0 {
            let (ss_hor, ss_ver) = (st.ss_hor, st.ss_ver);
            let h = (st.height + ss_ver) >> ss_ver;
            let w = (st.width + ss_hor) >> ss_hor;
            let next_row_y = (sby + 1) << ((6 - ss_ver) + sb128_u);
            let row_h = (next_row_y - (8 >> ss_ver) * not_last).min(h);
            let offset_uv = offset_y >> ss_ver;
            let y_stripe = (sby << ((6 - ss_ver) + sb128_u)) - offset_uv;
            for p in 1..3 {
                if (st.restore_planes >> p) & 1 != 0 {
                    a.lr_sbrow(pic, tools, y_stripe, w, h, row_h, p)?;
                }
            }
        }
    }
    Ok(())
}

/// SB satırı şeritlerini (`sby = 0..sbh`) `WorkerPool` üzerinde sınırlı dalga
/// tamponlarıyla (`wave_size = n_threads.min(sbh)`) paralel süzüp deterministik
/// `sby` sırasıyla `pic.data` ve `tools` içine birleştirir.
#[allow(clippy::too_many_arguments)]
fn apply_parallel_sbrows(
    pic: &mut PicBuf,
    pre_cdef: &mut LrCdefBackup,
    st: &LrState,
    sb128: usize,
    sbh: usize,
    bd: BitDepth,
    lbd: LrBitDepth,
    kernels: &LrKernels,
    tools: &mut ToolCounts,
    butce: &crate::frame::butce::Butce,
    pool: &crate::thread::WorkerPool,
) -> crate::error::Result<()> {
    use crate::frame::butce::dolu_vec;
    use alloc::sync::Arc;

    // `n_threads = 1` ile birebir aynı tepe bütçeyi (`allocation_peak`) üretmek
    // için tek bir `8 * max_stride` geçici `lpf` ayırmasını defterden geçir.
    let mut max_lpf_stride = 0usize;
    for (p, &stride) in pic.stride.iter().enumerate() {
        if (st.restore_planes >> p) & 1 != 0 {
            max_lpf_stride = max_lpf_stride.max(stride);
        }
    }
    if max_lpf_stride > 0 {
        let _lpf_budget_token = butce.gecici(8 * max_lpf_stride, 0u8)?;
    }

    struct SbRowJob {
        sby: usize,
        y_range: [(usize, usize); 3],
        dst: [Vec<u8>; 3],
        tools: ToolCounts,
    }

    let wave_size = pool.n_threads().min(sbh).max(1);
    let mut jobs: Vec<SbRowJob> = (0..wave_size)
        .map(|_| SbRowJob {
            sby: 0,
            y_range: [(0, 0); 3],
            dst: [Vec::new(), Vec::new(), Vec::new()],
            tools: ToolCounts::default(),
        })
        .collect();

    let pre0 = Arc::new(core::mem::take(&mut pre_cdef.data[0]));
    let pre1 = Arc::new(core::mem::take(&mut pre_cdef.data[1]));
    let pre2 = Arc::new(core::mem::take(&mut pre_cdef.data[2]));
    let pre_strides = pre_cdef.stride;
    let mask_arc = Arc::new(st.mask.clone());

    let restore_planes = st.restore_planes;
    let sb128w = st.sb128w;
    let unit_size_arr = st.unit_size;
    let width = st.width;
    let height = st.height;
    let ss_hor = st.ss_hor;
    let ss_ver = st.ss_ver;
    let strides = pic.stride;
    let bpp = bd.bytes_per_sample();
    let hbd = bd != BitDepth::Eight;
    let wiener_fns = kernels.wiener;
    let sgr_fns = kernels.sgr;

    let mut wave_res: crate::error::Result<()> = Ok(());
    for wave_start in (0..sbh).step_by(wave_size) {
        let count = (sbh - wave_start).min(wave_size);
        if jobs.len() > count {
            jobs.truncate(count);
        }
        for (i, job) in jobs.iter_mut().enumerate() {
            let sby = wave_start + i;
            let offset_y = if sby != 0 { 8 } else { 0 };
            let not_last = usize::from(sby + 1 < sbh);
            let mut y_range = [(0usize, 0usize); 3];

            if restore_planes & 1 != 0 {
                let h = height;
                let next_row_y = (sby + 1) << (6 + sb128);
                let row_h = (next_row_y - 8 * not_last).min(h);
                let y_stripe = (sby << (6 + sb128)) - offset_y;
                y_range[0] = (y_stripe, row_h);
            }
            if restore_planes & 6 != 0 {
                let h = (height + ss_ver) >> ss_ver;
                let next_row_y = (sby + 1) << ((6 - ss_ver) + sb128);
                let row_h = (next_row_y - (8 >> ss_ver) * not_last).min(h);
                let offset_uv = offset_y >> ss_ver;
                let y_stripe = (sby << ((6 - ss_ver) + sb128)) - offset_uv;
                for (p, yr) in y_range.iter_mut().enumerate().skip(1) {
                    if (restore_planes >> p) & 1 != 0 {
                        *yr = (y_stripe, row_h);
                    }
                }
            }
            job.sby = sby;
            job.y_range = y_range;
            job.tools = ToolCounts::default();
            for p in 0..3 {
                if (restore_planes >> p) & 1 != 0 {
                    let (y0, y1) = y_range[p];
                    let len = y1.saturating_sub(y0) * strides[p];
                    if len > 0 {
                        if job.dst[p].capacity() < len {
                            job.dst[p] = match dolu_vec(len, 0u8) {
                                Ok(v) => v,
                                Err(e) => {
                                    wave_res = Err(e);
                                    break;
                                }
                            };
                        } else {
                            job.dst[p].resize(len, 0);
                        }
                        job.dst[p].copy_from_slice(&pic.data[p][y0 * strides[p]..y1 * strides[p]]);
                    }
                }
            }
            if wave_res.is_err() {
                break;
            }
        }
        if wave_res.is_err() {
            break;
        }

        let pre0_w = Arc::clone(&pre0);
        let pre1_w = Arc::clone(&pre1);
        let pre2_w = Arc::clone(&pre2);
        let mask_w = Arc::clone(&mask_arc);

        let res = pool.execute_tasks(&mut jobs, move |_wid, _idx, job: &mut SbRowJob| {
            let pre_planes: [&[u8]; 3] = [pre0_w.as_slice(), pre1_w.as_slice(), pre2_w.as_slice()];
            for p in 0..3 {
                if (restore_planes >> p) & 1 == 0 {
                    continue;
                }
                let (y_stripe, row_h) = job.y_range[p];
                if y_stripe >= row_h {
                    continue;
                }
                let stride = strides[p];
                let ps = pre_strides[p];
                let (psh, psv) = if p == 0 { (0, 0) } else { (ss_hor, ss_ver) };
                let w = if p == 0 { width } else { (width + psh) >> psh };
                let h = if p == 0 {
                    height
                } else {
                    (height + psv) >> psv
                };
                let unit_size_log2 = usize::from(unit_size_arr[usize::from(p != 0)]);
                let unit_size = 1usize << unit_size_log2;
                let half_unit_size = unit_size >> 1;
                let max_unit_size = unit_size + half_unit_size;
                let row_y = y_stripe + if y_stripe != 0 { 8 >> psv } else { 0 };
                let shift_hor = 7 - psh;
                let mut edges = LrEdgeFlags::HAVE_RIGHT;
                if y_stripe > 0 {
                    edges = edges.union(LrEdgeFlags::HAVE_TOP);
                }
                let mut aligned_unit_pos = row_y & !(unit_size - 1);
                if aligned_unit_pos != 0 && aligned_unit_pos + half_unit_size > h {
                    aligned_unit_pos -= unit_size;
                }
                aligned_unit_pos <<= psv;
                let sb_idx = (aligned_unit_pos >> 7) * sb128w;
                let unit_idx = ((aligned_unit_pos >> 6) & 1) << 1;
                let unit_at =
                    |sb: usize, u: usize| mask_w.get(sb).map_or(LrUnit::default(), |s| s[p][u]);
                let mut lr = [unit_at(sb_idx, unit_idx), LrUnit::default()];
                let mut restore = lr[0].kind != LR_NONE;
                let mut x = 0usize;
                let mut bit = 0usize;
                let backup_h = row_h - y_stripe;
                let mut border = [
                    dolu_vec(backup_h * 4 * bpp, 0u8)?,
                    dolu_vec(backup_h * 4 * bpp, 0u8)?,
                ];
                let mut lpf = dolu_vec(8 * stride, 0u8)?;

                let mut run_unit = |dst_buf: &mut [u8],
                                    tools_out: &mut ToolCounts,
                                    left_buf: &[u8],
                                    ux: usize,
                                    uw: usize,
                                    u_lr: &LrUnit,
                                    mut u_edges: LrEdgeFlags| {
                    let mut stripe_h =
                        ((64 - if y_stripe == 0 { 8 } else { 0 }) >> psv).min(row_h - y_stripe);
                    let (params, func, kind_no) = if u_lr.kind == LR_WIENER {
                        let mut filter = [[0i16; 8]; 2];
                        let fh = u_lr.filter_h.map(i16::from);
                        let fv = u_lr.filter_v.map(i16::from);
                        filter[0][0] = fh[0];
                        filter[0][6] = fh[0];
                        filter[0][1] = fh[1];
                        filter[0][5] = fh[1];
                        filter[0][2] = fh[2];
                        filter[0][4] = fh[2];
                        filter[0][3] = -(fh[0] + fh[1] + fh[2]) * 2;
                        if hbd {
                            filter[0][3] += 128;
                        }
                        filter[1][0] = fv[0];
                        filter[1][6] = fv[0];
                        filter[1][1] = fv[1];
                        filter[1][5] = fv[1];
                        filter[1][2] = fv[2];
                        filter[1][4] = fv[2];
                        filter[1][3] = 128 - (fv[0] + fv[1] + fv[2]) * 2;
                        let idx = usize::from((filter[0][0] | filter[1][0]) == 0);
                        (LrParams::Wiener { filter }, wiener_fns[idx], idx)
                    } else {
                        let sgr_idx = usize::from(u_lr.kind - LR_SGRPROJ);
                        let sgr = SGR_PARAMS[sgr_idx];
                        let w0 = i32::from(u_lr.sgr_weights[0]);
                        let w1 = 128 - (w0 + i32::from(u_lr.sgr_weights[1]));
                        let idx = usize::from(sgr[0] != 0) + usize::from(sgr[1] != 0) * 2 - 1;
                        (
                            LrParams::Sgr {
                                s0: u32::from(sgr[0]),
                                s1: u32::from(sgr[1]),
                                w0,
                                w1,
                            },
                            sgr_fns[idx],
                            2 + idx,
                        )
                    };

                    let mut y = y_stripe;
                    let mut left_off = 0usize;
                    let pre = pre_planes[p];
                    while y + stripe_h <= row_h {
                        let have_bottom = job.sby + 1 != sbh || y + stripe_h != row_h;
                        u_edges = LrEdgeFlags::from_bits(
                            (u_edges.bits() & !LrEdgeFlags::HAVE_BOTTOM.bits())
                                | if have_bottom {
                                    LrEdgeFlags::HAVE_BOTTOM.bits()
                                } else {
                                    0
                                },
                        );
                        let col_lo = ux.saturating_sub(16) * bpp;
                        let col_hi = ((ux + uw + 32) * bpp).min(stride);
                        if u_edges.contains(LrEdgeFlags::HAVE_TOP) {
                            LrCdefBackup::copy_top_slices(
                                pre, ps, y, psv, &mut lpf, stride, col_lo, col_hi,
                            );
                        }
                        if have_bottom {
                            let b = y + stripe_h;
                            LrCdefBackup::copy_bottom_slices(
                                pre, ps, b, psv, &mut lpf, stride, col_lo, col_hi,
                            );
                        }
                        let local_off = (y - y_stripe) * stride + ux * bpp;
                        crate::dsp::profil_olc!(
                            @ if kind_no < 2 {
                                crate::dsp::profil::Islev::LrWiener
                            } else {
                                crate::dsp::profil::Islev::LrSgr
                            },
                            (func)(
                                &mut dst_buf[local_off..],
                                stride,
                                &left_buf[left_off * 4 * bpp..],
                                &lpf,
                                stride,
                                ux,
                                uw,
                                stripe_h,
                                &params,
                                u_edges,
                                lbd,
                            )
                        );
                        match kind_no {
                            0 => tools_out.lr_wiener7 += 1,
                            1 => tools_out.lr_wiener5 += 1,
                            2 => tools_out.lr_sgr5x5 += 1,
                            3 => tools_out.lr_sgr3x3 += 1,
                            _ => tools_out.lr_sgrmix += 1,
                        }
                        left_off += stripe_h;
                        y += stripe_h;
                        u_edges = u_edges.union(LrEdgeFlags::HAVE_TOP);
                        stripe_h = (64 >> psv).min(row_h - y);
                        if stripe_h == 0 {
                            break;
                        }
                    }
                };

                while x + max_unit_size <= w {
                    let next_x = x + unit_size;
                    let next_u_idx = unit_idx + ((next_x >> (shift_hor - 1)) & 1);
                    lr[bit ^ 1] = unit_at(sb_idx + (next_x >> shift_hor), next_u_idx);
                    let restore_next = lr[bit ^ 1].kind != LR_NONE;
                    if restore_next {
                        let col = (x + unit_size - 4) * bpp;
                        for r in 0..backup_h {
                            let so = r * stride + col;
                            border[bit][r * 4 * bpp..(r + 1) * 4 * bpp]
                                .copy_from_slice(&job.dst[p][so..so + 4 * bpp]);
                        }
                    }
                    if restore {
                        run_unit(
                            &mut job.dst[p],
                            &mut job.tools,
                            &border[bit ^ 1],
                            x,
                            unit_size,
                            &lr[bit],
                            edges,
                        );
                    }
                    x = next_x;
                    restore = restore_next;
                    edges = edges.union(LrEdgeFlags::HAVE_LEFT);
                    bit ^= 1;
                }
                if restore {
                    edges = LrEdgeFlags::from_bits(edges.bits() & !LrEdgeFlags::HAVE_RIGHT.bits());
                    let unit_w = w - x;
                    run_unit(
                        &mut job.dst[p],
                        &mut job.tools,
                        &border[bit ^ 1],
                        x,
                        unit_w,
                        &lr[bit],
                        edges,
                    );
                }
            }
            Ok(())
        });

        if let Err(e) = res {
            wave_res = Err(e);
            break;
        }
        for job in &jobs {
            for p in 0..3 {
                if (restore_planes >> p) & 1 != 0 {
                    let (y0, y1) = job.y_range[p];
                    if y0 < y1 {
                        let stride = pic.stride[p];
                        pic.data[p][y0 * stride..y1 * stride].copy_from_slice(&job.dst[p]);
                    }
                }
            }
            tools.lr_wiener7 += job.tools.lr_wiener7;
            tools.lr_wiener5 += job.tools.lr_wiener5;
            tools.lr_sgr5x5 += job.tools.lr_sgr5x5;
            tools.lr_sgr3x3 += job.tools.lr_sgr3x3;
            tools.lr_sgrmix += job.tools.lr_sgrmix;
        }
    }

    pre_cdef.data[0] = Arc::try_unwrap(pre0).unwrap_or_else(|a| (*a).clone());
    pre_cdef.data[1] = Arc::try_unwrap(pre1).unwrap_or_else(|a| (*a).clone());
    pre_cdef.data[2] = Arc::try_unwrap(pre2).unwrap_or_else(|a| (*a).clone());
    wave_res
}

impl Apply<'_> {
    /// `lr_sbrow` (`lr_apply_tmpl.c:107-167`).
    #[allow(clippy::too_many_arguments)]
    fn lr_sbrow(
        &self,
        pic: &mut PicBuf,
        tools: &mut ToolCounts,
        y: usize,
        w: usize,
        h: usize,
        row_h: usize,
        p: usize,
    ) -> crate::error::Result<()> {
        let st = self.st;
        let (ss_hor, ss_ver) = st.plane_ss(p);
        let bpp = self.bpp;
        let unit_size_log2 = usize::from(st.unit_size[usize::from(p != 0)]);
        let unit_size = 1usize << unit_size_log2;
        let half_unit_size = unit_size >> 1;
        let max_unit_size = unit_size + half_unit_size;
        // SB satırının y'si (y, row_y'nin 8 luma satırı üstüdür).
        let row_y = y + if y != 0 { 8 >> ss_ver } else { 0 };
        let shift_hor = 7 - ss_hor;
        let mut edges = LrEdgeFlags::HAVE_RIGHT;
        if y > 0 {
            edges = edges.union(LrEdgeFlags::HAVE_TOP);
        }
        let mut aligned_unit_pos = row_y & !(unit_size - 1);
        if aligned_unit_pos != 0 && aligned_unit_pos + half_unit_size > h {
            aligned_unit_pos -= unit_size;
        }
        aligned_unit_pos <<= ss_ver;
        let sb_idx = (aligned_unit_pos >> 7) * st.sb128w;
        let unit_idx = ((aligned_unit_pos >> 6) & 1) << 1;
        let unit_at = |sb: usize, u: usize| st.mask.get(sb).map_or(LrUnit::default(), |s| s[p][u]);
        let mut lr = [unit_at(sb_idx, unit_idx), LrUnit::default()];
        let mut restore = lr[0].kind != LR_NONE;
        let mut x = 0usize;
        let mut bit = 0usize;
        let backup_h = row_h - y;
        // `pre_lr_border[2][128 + 8][4]`: birim başına sol dört sütun.
        let mut border = [vec![0u8; backup_h * 4 * bpp], vec![0u8; backup_h * 4 * bpp]];
        let stride = pic.stride[p];
        let mut lpf_opt = None;
        while x + max_unit_size <= w {
            let next_x = x + unit_size;
            let next_u_idx = unit_idx + ((next_x >> (shift_hor - 1)) & 1);
            lr[bit ^ 1] = unit_at(sb_idx + (next_x >> shift_hor), next_u_idx);
            let restore_next = lr[bit ^ 1].kind != LR_NONE;
            if restore_next {
                // `backup4xU(pre_lr_border[bit], p + unit_size - 4, ...)`.
                let src = &pic.data[p];
                let col = (x + unit_size - 4) * bpp;
                for r in 0..backup_h {
                    let so = (y + r) * stride + col;
                    border[bit][r * 4 * bpp..(r + 1) * 4 * bpp]
                        .copy_from_slice(&src[so..so + 4 * bpp]);
                }
            }
            if restore {
                let lpf = match &mut lpf_opt {
                    Some(b) => b,
                    None => lpf_opt.insert(self.butce.gecici(8 * stride, 0u8)?),
                };
                self.lr_stripe(
                    pic,
                    tools,
                    &border[bit ^ 1],
                    lpf,
                    x,
                    y,
                    p,
                    unit_size,
                    row_h,
                    &lr[bit],
                    edges,
                )?;
            }
            x = next_x;
            restore = restore_next;
            edges = edges.union(LrEdgeFlags::HAVE_LEFT);
            bit ^= 1;
        }
        if restore {
            edges = LrEdgeFlags::from_bits(edges.bits() & !LrEdgeFlags::HAVE_RIGHT.bits());
            let unit_w = w - x;
            let lpf = match &mut lpf_opt {
                Some(b) => b,
                None => lpf_opt.insert(self.butce.gecici(8 * stride, 0u8)?),
            };
            self.lr_stripe(
                pic,
                tools,
                &border[bit ^ 1],
                lpf,
                x,
                y,
                p,
                unit_w,
                row_h,
                &lr[bit],
                edges,
            )?;
        }
        Ok(())
    }

    /// `lr_stripe` (`lr_apply_tmpl.c:36-98`).
    #[allow(clippy::too_many_arguments)]
    fn lr_stripe(
        &self,
        pic: &mut PicBuf,
        tools: &mut ToolCounts,
        left: &[u8],
        lpf: &mut [u8],
        x: usize,
        y0: usize,
        p: usize,
        unit_w: usize,
        row_h: usize,
        lr: &LrUnit,
        mut edges: LrEdgeFlags,
    ) -> crate::error::Result<()> {
        let st = self.st;
        let (_, ss_ver) = st.plane_ss(p);
        let bpp = self.bpp;
        let stride = pic.stride[p];
        let plane_h = if p == 0 {
            st.height
        } else {
            (st.height + ss_ver) >> ss_ver
        };
        let sby = (y0 + if y0 != 0 { 8 << ss_ver } else { 0 }) >> (6 - ss_ver + self.sb128);
        // Karenin ilk şeridi 8 luma satırı kısadır.
        let mut stripe_h = ((64 - if y0 == 0 { 8 } else { 0 }) >> ss_ver).min(row_h - y0);

        let (params, func, kind_name, kind_no) = if lr.kind == LR_WIENER {
            let mut filter = [[0i16; 8]; 2];
            let fh = lr.filter_h.map(i16::from);
            let fv = lr.filter_v.map(i16::from);
            filter[0][0] = fh[0];
            filter[0][6] = fh[0];
            filter[0][1] = fh[1];
            filter[0][5] = fh[1];
            filter[0][2] = fh[2];
            filter[0][4] = fh[2];
            filter[0][3] = -(fh[0] + fh[1] + fh[2]) * 2;
            // 8 bit SIMD için `+128` ayrı ele alınır (`lr_apply_tmpl.c:63-66`).
            if self.hbd {
                filter[0][3] += 128;
            }
            filter[1][0] = fv[0];
            filter[1][6] = fv[0];
            filter[1][1] = fv[1];
            filter[1][5] = fv[1];
            filter[1][2] = fv[2];
            filter[1][4] = fv[2];
            filter[1][3] = 128 - (fv[0] + fv[1] + fv[2]) * 2;
            let idx = usize::from((filter[0][0] | filter[1][0]) == 0);
            (
                LrParams::Wiener { filter },
                self.kernels.wiener[idx],
                ["w7", "w5"][idx],
                idx,
            )
        } else {
            let sgr_idx = usize::from(lr.kind - LR_SGRPROJ);
            let sgr = SGR_PARAMS[sgr_idx];
            let w0 = i32::from(lr.sgr_weights[0]);
            let w1 = 128 - (w0 + i32::from(lr.sgr_weights[1]));
            let idx = usize::from(sgr[0] != 0) + usize::from(sgr[1] != 0) * 2 - 1;
            (
                LrParams::Sgr {
                    s0: u32::from(sgr[0]),
                    s1: u32::from(sgr[1]),
                    w0,
                    w1,
                },
                self.kernels.sgr[idx],
                ["s0", "s1", "s2"][idx],
                2 + idx,
            )
        };

        let mut y = y0;
        let mut left_off = 0usize;
        // dav1d `lpf` düzeni: satır 0,1 = şeridin üstündeki iki satır;
        // satır 6,7 = altındaki iki satır (`lpf + 6 * stride`).
        let col_lo = x.saturating_sub(16) * bpp;
        let col_hi = ((x + unit_w + 32) * bpp).min(stride);
        let pre = &self.pre_cdef.data[p];
        let ps = self.pre_cdef.stride[p];
        while y + stripe_h <= row_h {
            let have_bottom = sby + 1 != self.sbh || y + stripe_h != row_h;
            edges = LrEdgeFlags::from_bits(
                (edges.bits() & !LrEdgeFlags::HAVE_BOTTOM.bits())
                    | if have_bottom {
                        LrEdgeFlags::HAVE_BOTTOM.bits()
                    } else {
                        0
                    },
            );
            if edges.contains(LrEdgeFlags::HAVE_TOP) {
                LrCdefBackup::copy_top_slices(pre, ps, y, ss_ver, lpf, stride, col_lo, col_hi);
            }
            if have_bottom {
                let b = y + stripe_h;
                LrCdefBackup::copy_bottom_slices(pre, ps, b, ss_ver, lpf, stride, col_lo, col_hi);
            }
            let _ = plane_h;
            let off = y * stride + x * bpp;
            crate::dsp::profil_olc!(
                @ if kind_no < 2 {
                    crate::dsp::profil::Islev::LrWiener
                } else {
                    crate::dsp::profil::Islev::LrSgr
                },
                (func)(
                &mut pic.data[p][off..],
                stride,
                &left[left_off * 4 * bpp..],
                lpf,
                stride,
                x,
                unit_w,
                stripe_h,
                &params,
                edges,
                self.lbd,
            ));
            match kind_no {
                0 => tools.lr_wiener7 += 1,
                1 => tools.lr_wiener5 += 1,
                2 => tools.lr_sgr5x5 += 1,
                3 => tools.lr_sgr3x3 += 1,
                _ => tools.lr_sgrmix += 1,
            }
            if let Some(t) = self.trace.as_ref()
                && t.on(self.frame_no, p)
            {
                let hash = fnv_block(&pic.data[p][off..], stride, unit_w, stripe_h, bpp);
                let pv: [i32; 8] = match params {
                    LrParams::Wiener { filter } => [
                        filter[0][0].into(),
                        filter[0][1].into(),
                        filter[0][2].into(),
                        filter[0][3].into(),
                        filter[1][0].into(),
                        filter[1][1].into(),
                        filter[1][2].into(),
                        filter[1][3].into(),
                    ],
                    LrParams::Sgr { s0, s1, w0, w1 } => [s0 as i32, s1 as i32, w0, w1, 0, 0, 0, 0],
                };
                crate::trace_eprintln!(
                    "L {} {p} {x} {y} {unit_w} {stripe_h} {kind_name} {} {} {} {} {} {} {} {} {} {hash}",
                    self.frame_no,
                    pv[0],
                    pv[1],
                    pv[2],
                    pv[3],
                    pv[4],
                    pv[5],
                    pv[6],
                    pv[7],
                    edges.bits()
                );
            }
            left_off += stripe_h;
            y += stripe_h;
            edges = edges.union(LrEdgeFlags::HAVE_TOP);
            stripe_h = (64 >> ss_ver).min(row_h - y);
            if stripe_h == 0 {
                break;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `restore_switchable` sembolü 0/1/2 → NONE/WIENER/SGRPROJ (`decode.c:2529`).
    #[test]
    fn switchable_sembol_eslemesi() {
        for (filter, beklenen) in [(0u8, LR_NONE), (1, LR_WIENER), (2, LR_SGRPROJ)] {
            assert_eq!(filter + u8::from(filter != 0), beklenen);
        }
    }

    /// Tile başı varsayılanları yuvaya yazılır ve `lr_ref` olur
    /// (`decode.c:2500-2507`); tür alanı korunur.
    #[test]
    fn tile_basi_varsayilanlari() {
        let mut st = LrState {
            restore_planes: 0b101,
            sb128w: 2,
            mask: vec![[[LrUnit::default(); 4]; 3]; 4],
            tile_ops: None,
            lr_ref: [LrUnit::default(); 3],
            kind: [
                RestorationType::Wiener,
                RestorationType::None,
                RestorationType::Sgrproj,
            ],
            unit_size: [6, 6],
            width: 256,
            height: 256,
            coded_width: 256,
            sr_denom: 0,
            ss_hor: 1,
            ss_ver: 1,
        };
        st.mask[3][2][3].kind = 7;
        // row_start = 48 (4×4): 128 px satırı → sb satırı 1, alt yarı;
        // col_start = 48: sb sütunu 1, sağ yarı.
        st.tile_init(48, 48);
        let u = st.mask[3][2][3];
        assert_eq!(u.kind, 7, "tür korunur");
        assert_eq!(u.filter_h, [3, -7, 15]);
        assert_eq!(u.filter_v, [3, -7, 15]);
        assert_eq!(u.sgr_weights, [-32, 31]);
        assert_eq!(st.lr_ref[0].filter_h, [3, -7, 15]);
        assert_eq!(
            st.mask[3][1][3],
            LrUnit::default(),
            "kapalı düzleme yazılmaz"
        );
    }

    /// `dav1d_sgr_params`: s0 = 0 olan satırlar 10..=13, s1 = 0 olanlar 14..=15.
    #[test]
    fn sgr_parametre_tablosu() {
        let s0_sifir: Vec<usize> = (0..16).filter(|&i| SGR_PARAMS[i][0] == 0).collect();
        let s1_sifir: Vec<usize> = (0..16).filter(|&i| SGR_PARAMS[i][1] == 0).collect();
        assert_eq!(s0_sifir, vec![10, 11, 12, 13]);
        assert_eq!(s1_sifir, vec![14, 15]);
    }
}
