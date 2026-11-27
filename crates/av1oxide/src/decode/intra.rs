//! Intra kare çözme uçtan uca — S2b.3.
//!
//! Bu modül dav1d `src/decode.c` (tile/superblock/partition/intra blok
//! sözdizimi) ve `src/recon_tmpl.c` (`dav1d_recon_b_intra`, `decode_coefs`
//! çağrıları) C yolunun birebir karşılığıdır. Kapsam **yalnız intra
//! karelerdir** (KEY / INTRA_ONLY) ve seçim kuralı gereği filtreler kapalıdır.
//!
//! ## Kaynak eşlemesi (dav1d @ `99f3354`, salt-okunur)
//!
//! | Kural | dav1d satırı | AV1 § |
//! |---|---|---|
//! | kare boyutları, sb128, sbh | `decode.c` 3334-3528 | 5.9.1 |
//! | tile/sbrow döngüsü | `decode.c` 3203-3248, 2601-2756 | 5.10, 5.11 |
//! | `setup_tile`, CDF kopyası, MSAC | `decode.c` 2431-2516 | 5.10.1 |
//! | partition ağacı + bağlam | `decode.c` 2123-2395 | 5.11.4 |
//! | `decode_b` intra yolu | `decode.c` 683-1260 | 5.11.5-5.11.27 |
//! | segment id / skip / cdef / delta q-lf | `decode.c` 820-1000 | 5.11.8-5.11.11 |
//! | y/uv mod, açı deltası, CfL, palette | `decode.c` 1050-1215 | 5.11.12-5.11.20 |
//! | tx boyutu (implicit split) | `decode.c` 1218-1240, `recon_tmpl.c` 731-937 | 5.11.21 |
//! | `dav1d_recon_b_intra` | `recon_tmpl.c` 1176-1556 | 7.11.2, 7.11.3 |
//! | `read_pal_plane`/`read_pal_indices` | `recon_tmpl.c` 2172-2310 | 5.11.18 |
//! | `get_cur_frame_segid` | `env.h` 439-460 | 5.11.8 |
//! | kare çıkışı/yaşamı | `src/lib.c` 438-483, `decode.c` 3529-3555 | 7.20 |
//!
//! ## Bilinçli sınırlar
//!
//! - Araç reddi yoktur (S2d.1): inter tahmin, intrabc, deblocking, CDEF,
//!   LR, superres ve kuantizasyon matrisi çözülür. Film grain kare çözümünün
//!   parçası değildir; yalnız çıktı kopyasına uygulanır
//!   ([`crate::decode::filmgrain`]) ve referans grain'siz kalır.
//! - Kare çözümü **tek iş parçacıklı** ve **durumsuzdur**: intra kareler
//!   önceki kare piksellerine bağımlı değildir (primary_ref_frame = NONE).
//! - Loop filter kapalı olduğundan `ipred_edge` yedeği yerine doğrudan
//!   düzlem pikselleri okunur (dav1d `decode.c:2111-2135` ile eşdeğer).
//! - Pikseller 128'e hizalanmış geçici `PicBuf` üzerinde çözülür (dav1d
//!   `dav1d_default_picture_alloc`), görünür alan `Frame`'e kopyalanır.

#![forbid(unsafe_code)]

use crate::bitstream::SequenceHeader;
use crate::bitstream::frame_header::{FrameHeader, FrameType, SegmentationData, TxfmMode};
use crate::decode::cdef::cdef_idx_slot;
use crate::decode::coef::{CoefEnv, decode_coefs};
use crate::decode::edge_tree::{
    EDGE_ALL_LEFT_HAS_BOTTOM, EDGE_ALL_TOP_HAS_RIGHT, EDGE_ALL_TR_AND_BL,
    EDGE_I444_LEFT_HAS_BOTTOM, EDGE_I444_TOP_HAS_RIGHT, EdgeTree, NodeId,
};
use crate::decode::ipred_prepare::{
    EdgeCtx, chroma_edge_masks, prepare_intra_edges, prepare_intra_edges_ctx,
};
use crate::decode::probe;
use crate::decode::refmvs::{Candidate, Mv, RefMvs, TileBounds, get_gmv_2d};
use crate::decode::refmvs_tmvs::TemporalBlock;
use crate::decode::warpv;
use crate::dsp::ipred::Edge;
use crate::dsp::itx::{TxSize, TxType};
use crate::dsp::{BitDepthDsp, DspDispatch};
use crate::entropy::CdfContext;
use crate::entropy::cdf_layout as layout;
use crate::entropy::msac::{MsacContext, ulog2};
use crate::error::{Error, Result, SizeStage};
use crate::frame::butce::Butce;
use crate::frame::{BitDepth, ChromaFormat};
use crate::sync::OnceLock;
use crate::tables::decode as t;
use alloc::format;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

// --- kare düzeyi sabitler -------------------------------------------------

/// `BL_128X128`.
const BL_128X128: u8 = 0;
/// `BL_64X64`.
const BL_64X64: u8 = 1;
/// `BL_8X8`.
const BL_8X8: u8 = 4;

/// `BS_4x4`.
const BS_4X4: u8 = 21;

/// `TX_4X4`.
const TX_4X4: u8 = 0;
/// `TX_64X64`.
const TX_64X64: u8 = 4;

/// Inter tahmin modları (`enum InterPredMode`, `levels.h:200-206`).
const NEARESTMV: u8 = 0;
/// `NEARMV`.
const NEARMV: u8 = 1;
/// `GLOBALMV`.
const GLOBALMV: u8 = 2;
/// `NEWMV`.
const NEWMV: u8 = 3;

/// `enum DRL_PROXIMITY` (`levels.h:208-213`).
const NEAREST_DRL: u8 = 0;
/// `NEARER_DRL`.
const NEARER_DRL: u8 = 1;
/// `NEAR_DRL`.
const NEAR_DRL: u8 = 2;

/// `COMP_INTER_NONE`.
const COMP_INTER_NONE: u8 = 0;
/// `COMP_INTER_WEIGHTED_AVG`.
const COMP_INTER_WEIGHTED_AVG: u8 = 1;
/// `COMP_INTER_AVG`.
const COMP_INTER_AVG: u8 = 2;
/// `COMP_INTER_SEG`.
const COMP_INTER_SEG: u8 = 3;
/// `COMP_INTER_WEDGE`.
const COMP_INTER_WEDGE: u8 = 4;

/// `INTER_INTRA_BLEND` (`levels.h`).
const INTER_INTRA_BLEND: u8 = 1;
/// `INTER_INTRA_WEDGE`.
const INTER_INTRA_WEDGE: u8 = 2;
/// `II_SMOOTH_PRED`.
const II_SMOOTH_PRED: u8 = 3;
/// `SMOOTH_PRED` (`enum IntraPredMode`).
const SMOOTH_PRED: u8 = 9;

/// `enum CompInterPredMode` sırasıyla `dav1d_comp_inter_pred_modes`.
const COMP_INTER_PRED_MODES: [[u8; 2]; 8] = [
    [NEARESTMV, NEARESTMV],
    [NEARMV, NEARMV],
    [NEARESTMV, NEWMV],
    [NEWMV, NEARESTMV],
    [NEARMV, NEWMV],
    [NEWMV, NEARMV],
    [GLOBALMV, GLOBALMV],
    [NEWMV, NEWMV],
];

/// `enum MotionMode` (`levels.h`): öteleme.
const MM_TRANSLATION: u8 = 0;
/// OBMC.
const MM_OBMC: u8 = 1;
/// Yerel warp.
const MM_WARP: u8 = 2;

/// `N_SWITCHABLE_FILTERS` (`headers.h:71`).
const N_SWITCHABLE_FILTERS: u8 = 3;
/// `DAV1D_FILTER_8TAP_REGULAR`.
const FILTER_8TAP_REGULAR: u8 = 0;
/// `DAV1D_FILTER_8TAP_SMOOTH`.
const FILTER_8TAP_SMOOTH: u8 = 1;
/// `DAV1D_FILTER_8TAP_SHARP`.
const FILTER_8TAP_SHARP: u8 = 2;
/// `DAV1D_FILTER_BILINEAR` (`levels.h:72`).
const FILTER_BILINEAR: u8 = 3;

/// `DC_PRED`.
const DC_PRED: u8 = 0;
/// `VERT_PRED`.
const VERT_PRED: u8 = 1;
/// `VERT_LEFT_PRED`.
const VERT_LEFT_PRED: u8 = 8;
/// `FILTER_PRED`.
const FILTER_PRED: u8 = 13;
/// `CFL_PRED`.
const CFL_PRED: u8 = 13;

/// `cfl_allowed_mask` (`src/tables.h:70-84`): BS_32x32..BS_4x4.
// dav1d `src/tables.h:70-84`: BS_32x32..BS_4x4, ancak **BS_16x64 (bit 10)
// hariç**. `0x3fff80` bit 10'u da içerir; bu, 16x64 bloklarda CfL'e izin
// verip `uv_mode` CDF'sini 13 sembolle okur (dav1d 12) ve izi kaydırır.
const CFL_ALLOWED_MASK: u32 = 0x003f_fb80;

/// Kare içi renk önbelleği (3 düzlem × 8 renk).
type Palette = [[u16; 8]; 3];

/// 128'e hizalanmış geçici resim tamponu (dav1d varsayılan ayırıcısı).
///
/// Genişlik ve yükseklik 128'in katına yuvarlanır; böylece kenardaki bloklar
/// tam tx boyutunu görünür alanın dışına yazabilir (dav1d
/// `src/picture.c:46-78`).
#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct PicBuf {
    pub(crate) data: [Vec<u8>; 3],
    pub(crate) stride: [usize; 3],
    pub(crate) x_base: [usize; 3],
    pub(crate) y_base: [usize; 3],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) cw: usize,
    pub(crate) ch: usize,
    pub(crate) bpp: usize,
    pub(crate) has_chroma: bool,
    pub(crate) ss_hor: usize,
    pub(crate) ss_ver: usize,
}

#[allow(dead_code)]
impl PicBuf {
    pub(crate) fn new(
        width: usize,
        height: usize,
        chroma: ChromaFormat,
        bd: BitDepth,
        butce: &Butce,
    ) -> Result<Self> {
        Self::new_with_reuse(
            width,
            height,
            chroma,
            bd,
            [Vec::new(), Vec::new(), Vec::new()],
            butce,
        )
    }

    /// Havuzdan geri dönüştürülmüş düzlem tamponları (`old_data`) varsa kapasiteleri
    /// eşleştiğinde yeniden kullanarak `PicBuf` kurar (`F8b-1`).
    pub(crate) fn new_with_reuse(
        width: usize,
        height: usize,
        chroma: ChromaFormat,
        bd: BitDepth,
        old_data: [Vec<u8>; 3],
        butce: &Butce,
    ) -> Result<Self> {
        let bpp = bd.bytes_per_sample();
        let has_chroma = chroma.has_chroma();
        let (ss_hor, ss_ver) = chroma.chroma_shift();
        let (ss_hor, ss_ver) = (ss_hor as usize, ss_ver as usize);
        // P38 (taşma sınıfı): boyut zinciri denetimli; AV1 sınırında
        // (65 536², 16 bit) düzlem 8 590 065 680 bayttır — 32 bitlik `usize`'ı
        // aşar, 64 bitte sığar. Taşma `Error::SizeOverflow` döner.
        let tasma = |stage, lhs, rhs| Error::SizeOverflow { stage, lhs, rhs };
        let hizala = |v: usize| {
            v.checked_add(127)
                .map(|n| n & !127)
                .ok_or(tasma(SizeStage::Stride, v, 127))
        };
        let aligned_w = hizala(width)?;
        let aligned_h = hizala(height)?;
        let y_stride =
            aligned_w
                .checked_mul(bpp)
                .ok_or(tasma(SizeStage::RowBytes, aligned_w, bpp))?;
        let uv_stride = if has_chroma {
            (aligned_w >> ss_hor) * bpp
        } else {
            0
        };
        // CDEF (S2c.2a): son satırdaki 8x8 bloğun `dst` dilimi `(h-1)*stride
        // + (w+2)*bpp` bayt gerektirir (çekirdek sözleşmesi; `HAVE_RIGHT` yokken
        // fazladan iki piksel **okunmaz** ama `debug_assert` uzunluğu ister).
        // Bir fazla satır bu sınırı her genişlikte karşılar; görünür alan
        // kopyası yalnız `height` satır okur, davranış değişmez.
        // P32: MC SIMD gövdeleri okuma bölgesinin sonundan sonra en çok
        // `MC_KUYRUK` bayt okur (kopyasız sözleşme); referans olarak okunan her
        // düzlemin kuyruğuna eklenir. (Fazladan satır bugün de yeterlidir; kuyruk
        // sözleşmeyi açık ve satır adımından bağımsız kılar.)
        let kuyruk = crate::dsp::mc::MC_KUYRUK;
        let duzlem = |stride: usize, satir: usize| {
            stride
                .checked_mul(satir + 1)
                .and_then(|n| n.checked_add(kuyruk))
                .ok_or(tasma(SizeStage::PlaneArea, stride, satir + 1))
        };
        let y_len = duzlem(y_stride, aligned_h)?;
        let uv_len = duzlem(uv_stride, aligned_h >> ss_ver)?;
        let (cw, ch) = if has_chroma {
            ((width + ss_hor) >> ss_hor, (height + ss_ver) >> ss_ver)
        } else {
            (0, 0)
        };
        let uv_len = if has_chroma { uv_len } else { 0 };
        let [old_y, old_u, old_v] = old_data;
        let (mut y_buf, y_dirty) = butce.reuse_vec_dirty(old_y, y_len, 0u8)?;
        if y_dirty {
            let active_row_bytes = width * bpp;
            if active_row_bytes < y_stride {
                for row in 0..height {
                    let base = row * y_stride;
                    y_buf[base + active_row_bytes..base + y_stride].fill(0);
                }
            }
            let tail_off = height * y_stride;
            if tail_off < y_len {
                y_buf[tail_off..y_len].fill(0);
            }
        }
        let (mut u_buf, u_dirty) = butce.reuse_vec_dirty(old_u, uv_len, 0u8)?;
        let (mut v_buf, v_dirty) = butce.reuse_vec_dirty(old_v, uv_len, 0u8)?;
        if has_chroma {
            let active_uv_bytes = cw * bpp;
            let uv_tail_off = ch * uv_stride;
            for (buf, dirty) in [(&mut u_buf, u_dirty), (&mut v_buf, v_dirty)] {
                if dirty {
                    if active_uv_bytes < uv_stride {
                        for row in 0..ch {
                            let base = row * uv_stride;
                            buf[base + active_uv_bytes..base + uv_stride].fill(0);
                        }
                    }
                    if uv_tail_off < uv_len {
                        buf[uv_tail_off..uv_len].fill(0);
                    }
                }
            }
        }
        Ok(Self {
            data: [y_buf, u_buf, v_buf],
            stride: [y_stride, uv_stride, uv_stride],
            x_base: [0; 3],
            y_base: [0; 3],
            width,
            height,
            cw,
            ch,
            bpp,
            has_chroma,
            ss_hor,
            ss_ver,
        })
    }

    /// Düzlem depolarının ayrı ayrı bayt uzunlukları (`[y_len, uv_len, uv_len]`).
    pub(crate) fn plane_byte_lens(
        width: usize,
        height: usize,
        chroma: ChromaFormat,
        bd: BitDepth,
    ) -> Result<[usize; 3]> {
        let bpp = bd.bytes_per_sample();
        let has_chroma = chroma.has_chroma();
        let (ss_hor, ss_ver) = chroma.chroma_shift();
        let (ss_hor, ss_ver) = (ss_hor as usize, ss_ver as usize);
        let tasma = |stage, lhs, rhs| Error::SizeOverflow { stage, lhs, rhs };
        let hizala = |v: usize| {
            v.checked_add(127)
                .map(|n| n & !127)
                .ok_or(tasma(SizeStage::Stride, v, 127))
        };
        let aligned_w = hizala(width)?;
        let aligned_h = hizala(height)?;
        let y_stride =
            aligned_w
                .checked_mul(bpp)
                .ok_or(tasma(SizeStage::RowBytes, aligned_w, bpp))?;
        let uv_stride = if has_chroma {
            (aligned_w >> ss_hor) * bpp
        } else {
            0
        };
        let kuyruk = crate::dsp::mc::MC_KUYRUK;
        let duzlem = |stride: usize, satir: usize| {
            stride
                .checked_mul(satir + 1)
                .and_then(|n| n.checked_add(kuyruk))
                .ok_or(tasma(SizeStage::PlaneArea, stride, satir + 1))
        };
        let y_len = duzlem(y_stride, aligned_h)?;
        let uv_len = if has_chroma {
            duzlem(uv_stride, aligned_h >> ss_ver)?
        } else {
            0
        };
        Ok([y_len, uv_len, uv_len])
    }

    /// Düzlem depolarının toplam baytı (bütçe defteri).
    pub(crate) fn bayt(&self) -> usize {
        self.data.iter().map(Vec::len).sum()
    }

    /// Havuzdan geri dönüştürülmüş düzlem tamponlarıyla (`old_data`) düşebilir kopya (`F8b-1`).
    pub(crate) fn try_clone_with_reuse(
        &self,
        old_data: [Vec<u8>; 3],
        butce: &Butce,
    ) -> Result<Self> {
        let [old_y, old_u, old_v] = old_data;
        Ok(Self {
            data: [
                butce.reuse_kopya(old_y, &self.data[0])?,
                butce.reuse_kopya(old_u, &self.data[1])?,
                butce.reuse_kopya(old_v, &self.data[2])?,
            ],
            ..*self
        })
    }

    /// Paralel tile çözümü için yalnız tile dikdörtgeni (`[row_start_4..row_end_4) × [col_start_4..col_end_4)`)
    /// boyutunda (`allow_intrabc` ise + sol/üst 8 piksel IntraBC dolgu payı) geçici `PicBuf` oluşturur (`F8b-2`).
    pub(crate) fn fork_for_tile(
        &self,
        row_start_4: usize,
        row_end_4: usize,
        col_start_4: usize,
        col_end_4: usize,
        allow_intrabc: bool,
    ) -> Result<Self> {
        use crate::frame::butce::dolu_vec;
        let pad = if allow_intrabc { 8 } else { 0 };
        let x0_y = (col_start_4 * 4).saturating_sub(pad);
        let y0_y = (row_start_4 * 4).saturating_sub(pad);
        let x0_uv = x0_y >> self.ss_hor;
        let y0_uv = y0_y >> self.ss_ver;
        let tile_w_luma = ((col_end_4 * 4).saturating_sub(x0_y).max(128) + 127) & !127;
        let tile_h_luma = ((row_end_4 * 4).saturating_sub(y0_y).max(128) + 127) & !127;
        let y_stride = tile_w_luma * self.bpp;
        let uv_stride = if self.has_chroma {
            (tile_w_luma >> self.ss_hor) * self.bpp
        } else {
            0
        };
        let kuyruk = crate::dsp::mc::MC_KUYRUK;
        let y_len = y_stride * (tile_h_luma + 1) + kuyruk;
        let uv_len = if self.has_chroma {
            uv_stride * ((tile_h_luma >> self.ss_ver) + 1) + kuyruk
        } else {
            0
        };
        Ok(Self {
            data: [
                dolu_vec(y_len, 0u8)?,
                dolu_vec(uv_len, 0u8)?,
                dolu_vec(uv_len, 0u8)?,
            ],
            stride: [y_stride, uv_stride, uv_stride],
            x_base: [x0_y, x0_uv, x0_uv],
            y_base: [y0_y, y0_uv, y0_uv],
            ..*self
        })
    }

    /// Tile yerel `other` düzlem piksellerinin `[row_start..row_end_sb) × [col_start..col_end_sb)`
    /// (4×4 blok biriminde, SB sınırlarına kadar) dikdörtgenini kare `PicBuf` içine birleştirir.
    pub(crate) fn merge_tile_from(
        &mut self,
        other: &Self,
        row_start_4: usize,
        row_end_4: usize,
        col_start_4: usize,
        col_end_4: usize,
    ) {
        let n_planes = if self.has_chroma { 3 } else { 1 };
        for p in 0..n_planes {
            let sh = if p == 0 { 0 } else { self.ss_hor };
            let sv = if p == 0 { 0 } else { self.ss_ver };
            let dst_stride = self.stride[p];
            let src_stride = other.stride[p];
            if dst_stride == 0 || src_stride == 0 {
                continue;
            }
            let max_rows = self.data[p].len() / dst_stride;
            let y0 = ((row_start_4 * 4) >> sv).min(max_rows);
            let y1 = (((row_end_4 * 4) + sv) >> sv).min(max_rows);
            let x0_px = (col_start_4 * 4) >> sh;
            let x1_px = ((col_end_4 * 4) + sh) >> sh;
            let dst_x0_bytes = (x0_px * self.bpp).min(dst_stride);
            let dst_x1_bytes = (x1_px * self.bpp).min(dst_stride);
            if dst_x0_bytes < dst_x1_bytes {
                let copy_bytes = dst_x1_bytes - dst_x0_bytes;
                let src_x0_bytes = (x0_px - other.x_base[p]) * self.bpp;
                for y in y0..y1 {
                    let dst_off = y * dst_stride + dst_x0_bytes;
                    let src_off = (y - other.y_base[p]) * src_stride + src_x0_bytes;
                    self.data[p][dst_off..dst_off + copy_bytes]
                        .copy_from_slice(&other.data[p][src_off..src_off + copy_bytes]);
                }
            }
        }
    }

    #[inline]
    fn off(&self, plane: usize, x: usize, y: usize) -> usize {
        (y - self.y_base[plane]) * self.stride[plane] + (x - self.x_base[plane]) * self.bpp
    }
}

/// Çözülmüş bir karenin **referans kopyası** (piksel + başlık + CDF + segment
/// haritası + temporal MV ızgarası).
///
/// dav1d `Dav1dThreadPicture` + `c->refs[i]` (piksel, `frame_hdr`, `cdf`,
/// `segmap`, `refmvs`, `refpoc`) alanlarının Rust karşılığıdır.
pub(crate) struct RefFrame {
    /// 128'e hizalanmış yeniden kurulum tamponu (MC ve `show_existing_frame` kaynağı).
    pub pic: PicBuf,
    /// Kare başlığı (referans doğrulama, `svc`, `gmv`).
    pub hdr: FrameHeader,
    /// Kare sonu CDF'si (`refresh_context` güncellemesiyle).
    pub cdf: CdfContext,
    /// Segment haritası (`segmentation.update_map` sonrası).
    pub segmap: Vec<u8>,
    /// Kaydedilmiş 8×8 temporal MV ızgarası (`save_tmvs`).
    pub tmvs: Vec<TemporalBlock>,
    /// Temporal ızgara satır adımı.
    pub tmvs_stride: usize,
    /// Bu karenin 7 referansının `frame_offset`'ı (`refpoc`).
    pub refpoc: [u8; 7],
    /// Blok ızgarası yüksekliği (8 piksele yuvarlanmış 4×4 biriminde).
    /// Genişlik karşılığı `hdr.width[0]`'dan (`ref_coded_width`) türetilir;
    /// superres'li referansta `pic` büyütülmüş genişliktedir.
    pub bh: usize,
    /// Piksel düzeni.
    pub chroma: ChromaFormat,
    /// Bit derinliği.
    pub bd: BitDepth,
    /// Karenin dizi başlığında `mtrx == MC_IDENTITY` (film grain kroma
    /// kısıtlı aralığı; dav1d resmin kendi `seq_hdr`'ını kullanır,
    /// `src/fg_apply_tmpl.c:172`).
    pub mc_identity: bool,
}

impl RefFrame {
    /// dav1d `dav1d_thread_picture_ref` + `c->refs[i]`: çözülen karenin
    /// durumu yuvaya **taşınır** (P38 / `4.A.1`: ikincil `Frame` kopyası yok).
    #[must_use]
    pub fn from_decoded(decoded: DecodedFrame, hdr: FrameHeader, mc_identity: bool) -> Self {
        Self {
            pic: decoded.pic,
            hdr,
            cdf: decoded.cdf,
            segmap: decoded.segmap,
            tmvs: decoded.tmvs,
            tmvs_stride: decoded.tmvs_stride,
            refpoc: decoded.refpoc,
            bh: decoded.bh,
            chroma: decoded.chroma,
            bd: decoded.bd,
            mc_identity,
        }
    }

    /// Bütçe defterinde sayılan bayt: yapının kendisi (CDF ve başlık dâhil,
    /// `Arc` ayırması) + piksel, segment ve temporal MV depoları.
    pub(crate) fn bayt(&self) -> usize {
        core::mem::size_of::<Self>()
            + self.pic.bayt()
            + self.segmap.len()
            + core::mem::size_of_val(self.tmvs.as_slice())
    }
}

impl DecodedFrame {
    /// Yuvaya taşınınca [`RefFrame::bayt`] ile aynı sayıyı verir.
    pub(crate) fn bayt(&self) -> usize {
        core::mem::size_of::<RefFrame>()
            + self.pic.bayt()
            + self.segmap.len()
            + core::mem::size_of_val(self.tmvs.as_slice())
    }
}

/// Bir karenin çözüm çıktısı: yeniden kurulum tamponu + referans durumu.
pub(crate) struct DecodedFrame {
    /// Yeniden kurulum tamponu (referans veya yerinde çıktı).
    pub pic: PicBuf,
    /// Piksel düzeni.
    pub chroma: ChromaFormat,
    /// Bit derinliği.
    pub bd: BitDepth,
    /// Kare sonu CDF'si.
    pub cdf: CdfContext,
    /// Segment haritası.
    pub segmap: Vec<u8>,
    /// Temporal MV ızgarası (yalnız inter/switch karelerde dolu).
    pub tmvs: Vec<TemporalBlock>,
    /// Temporal ızgara satır adımı.
    pub tmvs_stride: usize,
    /// 7 referansın `frame_offset`'ı.
    pub refpoc: [u8; 7],
    /// Blok ızgarası yüksekliği.
    pub bh: usize,
    /// Araç sayaçları.
    pub tools: ToolCounts,
}

/// dav1d `BlockContext` — yalnız intra kare yolunda kullanılan alanlar.
#[derive(Clone, Copy, Debug)]
struct BlockContext {
    mode: [u8; 32],
    lcoef: [u8; 32],
    ccoef: [[u8; 32]; 2],
    seg_pred: [u8; 32],
    skip: [u8; 32],
    skip_mode: [u8; 32],
    intra: [u8; 32],
    tx_intra: [i8; 32],
    tx: [i8; 32],
    partition: [u8; 16],
    uvmode: [u8; 32],
    pal_sz: [u8; 32],
    /// `comp_type` (`COMP_INTER_NONE` = 0).
    comp_type: [u8; 32],
    /// `ref[0]`/`ref[1]` (−1 = intra).
    ref_pair: [[i8; 32]; 2],
    /// `filter[0]`/`filter[1]` (3 = atanmamış).
    filter: [[u8; 32]; 2],
    /// `tx_lpf_y` (`env.h:52`): sol/sağ kenar tx log2 boyu.
    tx_lpf_y: [u8; 32],
    /// `tx_lpf_uv` (`env.h:53`).
    tx_lpf_uv: [u8; 32],
}

impl Default for BlockContext {
    fn default() -> Self {
        Self {
            mode: [0; 32],
            lcoef: [0x40; 32],
            ccoef: [[0x40; 32]; 2],
            seg_pred: [0; 32],
            skip: [0; 32],
            skip_mode: [0; 32],
            intra: [0; 32],
            tx_intra: [-1; 32],
            tx: [TX_64X64 as i8; 32],
            partition: [0; 16],
            uvmode: [DC_PRED; 32],
            pal_sz: [0; 32],
            comp_type: [0; 32],
            ref_pair: [[-1; 32]; 2],
            filter: [[3; 32]; 2],
            tx_lpf_y: [2; 32],
            tx_lpf_uv: [1; 32],
        }
    }
}

/// `reset_context` (`decode.c:2396-2422`) — anahtar/intra kare dalı.
#[inline(always)]
fn splat_n<T: Copy, const N: usize>(slice: &mut [T], off: usize, val: T) {
    slice[off..off + N].copy_from_slice(&[val; N]);
}

/// Küçük komşuluk bağlam dilimlerini (`1..=32` eleman) `_memset` kütüphane
/// çağrısına düşmeden sabit genişlikli yazmaç saklamalarıyla doldurur.
#[inline(always)]
fn fill_small<T: Copy>(slice: &mut [T], off: usize, count: usize, val: T) {
    let dst = &mut slice[off..off + count];
    match count {
        0 => {}
        1 => dst[0] = val,
        2 => dst.copy_from_slice(&[val; 2]),
        4 => dst.copy_from_slice(&[val; 4]),
        8 => dst.copy_from_slice(&[val; 8]),
        16 => dst.copy_from_slice(&[val; 16]),
        32 => dst.copy_from_slice(&[val; 32]),
        _ => {
            let mut i = 0;
            while i < dst.len() {
                dst[i] = val;
                i += 1;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn fill_intra_side_n<const N: usize>(
    ctx: &mut BlockContext,
    pal_uv_sz: &mut [u8; 32],
    off: usize,
    lsz: i8,
    y_mode_nofilt: u8,
    pal_sz0: u8,
    seg_pred: u8,
    skip: u8,
    uv_pal_sz: u8,
    is_inter_frame: bool,
) {
    splat_n::<i8, N>(&mut ctx.tx_intra, off, lsz);
    splat_n::<i8, N>(&mut ctx.tx, off, lsz);
    splat_n::<u8, N>(&mut ctx.mode, off, y_mode_nofilt);
    splat_n::<u8, N>(&mut ctx.pal_sz, off, pal_sz0);
    splat_n::<u8, N>(&mut ctx.seg_pred, off, seg_pred);
    splat_n::<u8, N>(&mut ctx.skip_mode, off, 0);
    splat_n::<u8, N>(&mut ctx.intra, off, 1);
    splat_n::<u8, N>(&mut ctx.skip, off, skip);
    splat_n::<u8, N>(pal_uv_sz, off, uv_pal_sz);
    if is_inter_frame {
        splat_n::<u8, N>(&mut ctx.comp_type, off, 0);
        splat_n::<i8, N>(&mut ctx.ref_pair[0], off, -1);
        splat_n::<i8, N>(&mut ctx.ref_pair[1], off, -1);
        splat_n::<u8, N>(&mut ctx.filter[0], off, N_SWITCHABLE_FILTERS);
        splat_n::<u8, N>(&mut ctx.filter[1], off, N_SWITCHABLE_FILTERS);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn fill_intra_side_ctx(
    ctx: &mut BlockContext,
    pal_uv_sz: &mut [u8; 32],
    off: usize,
    log2_sz: u8,
    lsz: i8,
    y_mode_nofilt: u8,
    pal_sz0: u8,
    seg_pred: u8,
    skip: u8,
    uv_pal_sz: u8,
    is_inter_frame: bool,
) {
    match log2_sz {
        0 => fill_intra_side_n::<1>(
            ctx,
            pal_uv_sz,
            off,
            lsz,
            y_mode_nofilt,
            pal_sz0,
            seg_pred,
            skip,
            uv_pal_sz,
            is_inter_frame,
        ),
        1 => fill_intra_side_n::<2>(
            ctx,
            pal_uv_sz,
            off,
            lsz,
            y_mode_nofilt,
            pal_sz0,
            seg_pred,
            skip,
            uv_pal_sz,
            is_inter_frame,
        ),
        2 => fill_intra_side_n::<4>(
            ctx,
            pal_uv_sz,
            off,
            lsz,
            y_mode_nofilt,
            pal_sz0,
            seg_pred,
            skip,
            uv_pal_sz,
            is_inter_frame,
        ),
        3 => fill_intra_side_n::<8>(
            ctx,
            pal_uv_sz,
            off,
            lsz,
            y_mode_nofilt,
            pal_sz0,
            seg_pred,
            skip,
            uv_pal_sz,
            is_inter_frame,
        ),
        4 => fill_intra_side_n::<16>(
            ctx,
            pal_uv_sz,
            off,
            lsz,
            y_mode_nofilt,
            pal_sz0,
            seg_pred,
            skip,
            uv_pal_sz,
            is_inter_frame,
        ),
        _ => fill_intra_side_n::<32>(
            ctx,
            pal_uv_sz,
            off,
            lsz,
            y_mode_nofilt,
            pal_sz0,
            seg_pred,
            skip,
            uv_pal_sz,
            is_inter_frame,
        ),
    }
}

#[inline(always)]
fn fill_inter_side_n<const N: usize>(
    ctx: &mut BlockContext,
    pal_uv_sz: &mut [u8; 32],
    off: usize,
    val: i8,
    seg_pred: u8,
    b: &Av1Block,
) {
    splat_n::<u8, N>(&mut ctx.seg_pred, off, seg_pred);
    splat_n::<u8, N>(&mut ctx.skip_mode, off, u8::from(b.skip_mode));
    splat_n::<u8, N>(&mut ctx.intra, off, 0);
    splat_n::<u8, N>(&mut ctx.skip, off, u8::from(b.skip));
    splat_n::<u8, N>(&mut ctx.pal_sz, off, 0);
    splat_n::<i8, N>(&mut ctx.tx_intra, off, val);
    splat_n::<u8, N>(&mut ctx.comp_type, off, b.comp_type);
    splat_n::<u8, N>(&mut ctx.filter[0], off, b.filter[0]);
    splat_n::<u8, N>(&mut ctx.filter[1], off, b.filter[1]);
    splat_n::<u8, N>(&mut ctx.mode, off, b.inter_mode);
    splat_n::<i8, N>(&mut ctx.ref_pair[0], off, b.ref0);
    splat_n::<i8, N>(&mut ctx.ref_pair[1], off, b.ref1);
    splat_n::<u8, N>(pal_uv_sz, off, 0);
}

#[inline]
fn fill_inter_side_ctx(
    ctx: &mut BlockContext,
    pal_uv_sz: &mut [u8; 32],
    off: usize,
    log2_sz: u8,
    seg_pred: u8,
    b: &Av1Block,
) {
    let val = log2_sz as i8;
    match log2_sz {
        0 => fill_inter_side_n::<1>(ctx, pal_uv_sz, off, val, seg_pred, b),
        1 => fill_inter_side_n::<2>(ctx, pal_uv_sz, off, val, seg_pred, b),
        2 => fill_inter_side_n::<4>(ctx, pal_uv_sz, off, val, seg_pred, b),
        3 => fill_inter_side_n::<8>(ctx, pal_uv_sz, off, val, seg_pred, b),
        4 => fill_inter_side_n::<16>(ctx, pal_uv_sz, off, val, seg_pred, b),
        _ => fill_inter_side_n::<32>(ctx, pal_uv_sz, off, val, seg_pred, b),
    }
}

fn reset_context(ctx: &mut BlockContext, keyframe: bool) {
    ctx.intra.fill(u8::from(keyframe));
    ctx.uvmode.fill(DC_PRED);
    if keyframe {
        ctx.mode.fill(DC_PRED);
    }
    ctx.partition.fill(0);
    ctx.skip.fill(0);
    ctx.skip_mode.fill(0);
    ctx.tx_intra.fill(-1);
    ctx.tx.fill(TX_64X64 as i8);
    ctx.lcoef.fill(0x40);
    ctx.ccoef[0].fill(0x40);
    ctx.ccoef[1].fill(0x40);
    ctx.seg_pred.fill(0);
    ctx.pal_sz.fill(0);
    ctx.filter[0].fill(N_SWITCHABLE_FILTERS);
    ctx.filter[1].fill(N_SWITCHABLE_FILTERS);
    ctx.tx_lpf_y.fill(2);
    ctx.tx_lpf_uv.fill(1);
    if !keyframe {
        ctx.ref_pair[0].fill(-1);
        ctx.ref_pair[1].fill(-1);
        ctx.comp_type.fill(0);
        ctx.mode.fill(NEARESTMV);
    }
}

/// Blok bağlamları + renk önbellekleri (dav1d `f->a`, `t->l`, `t->al_pal`).
struct Blocks {
    above: Vec<BlockContext>,
    above_pal: Vec<[Palette; 32]>,
    above_pal_uv_sz: Vec<[u8; 32]>,
    left: BlockContext,
    left_pal: [Palette; 32],
    left_pal_uv_sz: [u8; 32],
}

impl Blocks {
    #[cfg(test)]
    fn new(columns: usize, tile_rows: usize, butce: &Butce) -> Result<Self> {
        Self::new_with_reuse(columns, tile_rows, None, butce)
    }

    fn new_with_reuse(
        columns: usize,
        tile_rows: usize,
        old: Option<Self>,
        butce: &Butce,
    ) -> Result<Self> {
        // dav1d `decode.c:2794`: above bağlamı `sb128w * tiling.rows` kadardır —
        // **her tile satırı** ayrı above bağlamı bölgesi kullanır
        // (`decode.c:2651` `t->a = f->a + col_sb128_start + tile_row * f->sb128w`).
        let columns = columns.max(1) * tile_rows.max(1);
        let (old_above, old_pal, old_uv) = match old {
            Some(o) => (o.above, o.above_pal, o.above_pal_uv_sz),
            None => (Vec::new(), Vec::new(), Vec::new()),
        };
        Ok(Self {
            above: butce.reuse_vec(old_above, columns, BlockContext::default())?,
            above_pal: butce.reuse_vec(old_pal, columns, [[[0u16; 8]; 3]; 32])?,
            above_pal_uv_sz: butce.reuse_vec(old_uv, columns, [0u8; 32])?,
            left: BlockContext::default(),
            left_pal: [[[0u16; 8]; 3]; 32],
            left_pal_uv_sz: [0u8; 32],
        })
    }

    fn fork_for_tile(sb128w: usize, keyframe: bool) -> Result<Self> {
        use crate::frame::butce::dolu_vec;
        let mut ctx = BlockContext::default();
        reset_context(&mut ctx, keyframe);
        let n = sb128w.max(1);
        Ok(Self {
            above: dolu_vec(n, ctx)?,
            above_pal: dolu_vec(n, [[[0u16; 8]; 3]; 32])?,
            above_pal_uv_sz: dolu_vec(n, [0u8; 32])?,
            left: BlockContext::default(),
            left_pal: [[[0u16; 8]; 3]; 32],
            left_pal_uv_sz: [0u8; 32],
        })
    }

    /// Geri dönüştürülen tile `Blocks` bağlamının yalnızca ilgili sütun aralığını sıfırlar.
    fn reset_for_tile(&mut self, col_start_sb128: usize, col_end_sb128: usize, keyframe: bool) {
        let mut ctx = BlockContext::default();
        reset_context(&mut ctx, keyframe);
        let lo = col_start_sb128.min(self.above.len());
        let hi = col_end_sb128.min(self.above.len());
        if lo < hi {
            self.above[lo..hi].fill(ctx);
            self.above_pal[lo..hi].fill([[[0u16; 8]; 3]; 32]);
            self.above_pal_uv_sz[lo..hi].fill([0u8; 32]);
        }
        self.left = BlockContext::default();
        self.left_pal = [[[0u16; 8]; 3]; 32];
        self.left_pal_uv_sz = [0u8; 32];
    }
}

/// `struct ScalableMotionParams` (`src/internal.h:236-241`) — ölçekli referans.
#[derive(Clone, Copy, Debug, Default)]
struct ScalableMotionParams {
    /// 14-bit sabit noktalı ölçek çarpanı (`0` = ölçek yok).
    scale: i32,
    /// Örnekleme adımı (`(scale + 8) >> 4`).
    step: i32,
}

/// MC çıktı hedefi (`mc()` C imzasındaki `dst8`/`dst16` ayrımı).
enum McOut<'a> {
    /// Geçerli kare düzlemine yaz (`x`, `y` piksel).
    Pic { plane: usize, x: usize, y: usize },
    /// Ham bayt dilimine yaz (OBMC `lap`).
    Bytes {
        dst: &'a mut [u8],
        stride: usize,
        off: usize,
    },
    /// `i16` ara tampona yaz (compound).
    Prep { tmp: &'a mut [i16] },
}

/// Warp için gereken türetilmiş parametreler (matris + shear).
#[derive(Clone, Copy, Debug, Default)]
struct WarpParams {
    /// 16.16 sabit noktalı 2x3 matris.
    matrix: [i32; 6],
    /// `u.p.alpha/beta/gamma/delta` (`u.abcd` ile aynı).
    abcd: [i16; 4],
}

/// Çözülen blok (`Av1Block`'un intra + intrabc alanları).
#[derive(Clone, Copy, Debug)]
struct Av1Block {
    skip: bool,
    /// `b->intra` (`allow_intrabc` karesinde blok başına okunur).
    intra: bool,
    /// `b->mv[0]` (intrabc dv'si) ve `b->mv[1]` (compound).
    mv: [Mv; 2],
    /// `b->max_ytx` (`read_vartx_tree`).
    max_ytx: u8,
    /// `b->tx_split0/1` — var-tx ağacı maskeleri.
    tx_split: [u16; 2],
    seg_id: u8,
    y_mode: u8,
    uv_mode: u8,
    y_angle: i32,
    uv_angle: i32,
    cfl_alpha: [i32; 2],
    pal_sz: [u8; 2],
    tx: u8,
    uvtx: u8,
    /// `b->ref[0]` (−1 = intra).
    ref0: i8,
    /// `b->ref[1]` (−1 = tek referans).
    ref1: i8,
    /// `b->skip_mode`.
    skip_mode: bool,
    /// `b->wedge_idx`.
    wedge_idx: u8,
    /// `b->mask_sign`.
    mask_sign: u8,
    /// `b->interintra_mode` (`II_DC_PRED` = 0).
    interintra_mode: u8,
    /// `b->comp_type`.
    comp_type: u8,
    /// `b->inter_mode` (`InterPredMode` / `CompInterPredMode`).
    inter_mode: u8,
    /// `b->drl_idx`.
    drl_idx: u8,
    /// `b->motion_mode`.
    motion_mode: u8,
    /// `b->interintra_type`.
    interintra_type: u8,
    /// `filter[0]` / `filter[1]`.
    filter: [u8; 2],
    /// `b->filter2d`.
    filter2d: u8,
    /// Yerel warp parametreleri (`t->warpmv`; yalnız `MM_WARP`).
    warp: Option<WarpParams>,
}

impl Default for Av1Block {
    fn default() -> Self {
        Self {
            skip: false,
            intra: true,
            mv: [Mv::ZERO, Mv::ZERO],
            max_ytx: TX_4X4,
            tx_split: [0; 2],
            seg_id: 0,
            y_mode: DC_PRED,
            uv_mode: DC_PRED,
            y_angle: 0,
            uv_angle: 0,
            cfl_alpha: [0; 2],
            pal_sz: [0; 2],
            tx: TX_4X4,
            uvtx: TX_4X4,
            ref0: -1,
            ref1: -1,
            skip_mode: false,
            wedge_idx: 0,
            mask_sign: 0,
            interintra_mode: 0,
            comp_type: 0,
            inter_mode: NEARESTMV,
            drl_idx: NEAREST_DRL,
            motion_mode: MM_TRANSLATION,
            interintra_type: 0,
            filter: [N_SWITCHABLE_FILTERS; 2],
            filter2d: 0,
            warp: None,
        }
    }
}

/// Geçici çözme tamponları (`t->scratch`).
struct Scratch {
    edge: Vec<u8>,
    levels: Vec<u8>,
    cf: Vec<i32>,
    ac: Vec<i16>,
    pal_tmp: Vec<u8>,
    pal_idx_y: Vec<u8>,
    pal_idx_uv: Vec<u8>,
    /// `t->scratch.emu_edge` — `192 * (128 + 7)` piksel.
    emu_edge: Vec<u8>,
    /// `t->scratch.txtp_map` — 32×32 luma tx türü haritası (inter kroma).
    txtp_map: Vec<u8>,
    /// `t->scratch.compinter` — compound ara tamponu (2 × 128×128, 16-bit).
    compinter: Vec<i16>,
    /// `t->scratch.seg_mask` — fark ağırlıklı compound maskesi (128×128).
    seg_mask: Vec<u8>,
    /// `t->scratch.interintra` — interintra tahmin tamponu (64×64).
    interintra: Vec<u8>,
    /// `t->scratch.lap` — OBMC ara tamponu (128×32).
    lap: Vec<u8>,
}

impl Scratch {
    fn new_with_reuse(bd: BitDepth, old: Option<Self>, butce: &Butce) -> Result<Self> {
        let bpp = bd.bytes_per_sample();
        let (
            old_edge,
            old_levels,
            old_cf,
            old_ac,
            old_pal_tmp,
            old_pal_idx_y,
            old_pal_idx_uv,
            old_emu_edge,
            old_txtp_map,
            old_compinter,
            old_seg_mask,
            old_interintra,
            old_lap,
        ) = match old {
            Some(o) => (
                o.edge,
                o.levels,
                o.cf,
                o.ac,
                o.pal_tmp,
                o.pal_idx_y,
                o.pal_idx_uv,
                o.emu_edge,
                o.txtp_map,
                o.compinter,
                o.seg_mask,
                o.interintra,
                o.lap,
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
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ),
        };
        Ok(Self {
            edge: butce.reuse_vec(old_edge, 257 * bpp, 0u8)?,
            levels: butce.reuse_vec(old_levels, 32 * 34 + 64, 0u8)?,
            cf: butce.reuse_vec(old_cf, 1024, 0i32)?,
            ac: butce.reuse_vec(old_ac, 32 * 32, 0i16)?,
            pal_tmp: butce.reuse_vec(old_pal_tmp, 64 * 64, 0u8)?,
            pal_idx_y: butce.reuse_vec(old_pal_idx_y, 64 * 64, 0u8)?,
            pal_idx_uv: butce.reuse_vec(old_pal_idx_uv, 64 * 64, 0u8)?,
            // dav1d `src/internal.h:417-418`: 320 × (256 + 7) örnek; P32
            // kopyasız MC sözleşmesi için kuyruk (`MC_KUYRUK`).
            emu_edge: butce.reuse_vec(
                old_emu_edge,
                320 * (256 + 7) * bpp + crate::dsp::mc::MC_KUYRUK,
                0u8,
            )?,
            txtp_map: butce.reuse_vec(old_txtp_map, 32 * 32, 0u8)?,
            compinter: butce.reuse_vec(old_compinter, 2 * 128 * 128, 0i16)?,
            seg_mask: butce.reuse_vec(old_seg_mask, 128 * 128, 0u8)?,
            interintra: butce.reuse_vec(old_interintra, 64 * 64 * bpp, 0u8)?,
            lap: butce.reuse_vec(old_lap, 128 * 32 * bpp, 0u8)?,
        })
    }

    fn fork_for_tile(bd: BitDepth) -> Result<Self> {
        use crate::frame::butce::dolu_vec;
        let bpp = bd.bytes_per_sample();
        Ok(Self {
            edge: dolu_vec(257 * bpp, 0u8)?,
            levels: dolu_vec(32 * 34 + 64, 0u8)?,
            cf: dolu_vec(1024, 0i32)?,
            ac: dolu_vec(32 * 32, 0i16)?,
            pal_tmp: dolu_vec(64 * 64, 0u8)?,
            pal_idx_y: dolu_vec(64 * 64, 0u8)?,
            pal_idx_uv: dolu_vec(64 * 64, 0u8)?,
            emu_edge: dolu_vec(320 * (256 + 7) * bpp + crate::dsp::mc::MC_KUYRUK, 0u8)?,
            txtp_map: dolu_vec(32 * 32, 0u8)?,
            compinter: dolu_vec(2 * 128 * 128, 0i16)?,
            seg_mask: dolu_vec(128 * 128, 0u8)?,
            interintra: dolu_vec(64 * 64 * bpp, 0u8)?,
            lap: dolu_vec(128 * 32 * bpp, 0u8)?,
        })
    }
}

/// Kareler arası geçici çalışma tamponları paketi (`F8b-1` / `4.A.1`).
#[derive(Default)]
struct WorkFrameBufs {
    lf: Option<crate::decode::loopfilter::LfFrame>,
    blocks: Option<Blocks>,
    cdef_state: Option<crate::decode::cdef::CdefState>,
    lr_mask: Vec<[[crate::decode::lr::LrUnit; 4]; 3]>,
    scratch: Option<Scratch>,
    refmvs_grid: Vec<crate::decode::refmvs::RefBlock>,
    refmvs_rp_proj: Vec<TemporalBlock>,
    lr_cdef_backup: Option<[Vec<u8>; 3]>,
}

struct BufPoolInner {
    work_bufs: Vec<WorkFrameBufs>,
    u8_vecs: Vec<Vec<u8>>,
    segmap_vecs: Vec<Vec<u8>>,
    tmvs_vecs: Vec<Vec<TemporalBlock>>,
}

/// Kareler arası sınırlı tampon geri dönüşüm havuzu (`F8b-1` / `4.A.1`, `%100 #![forbid(unsafe_code)]`).
///
/// Kare çözümü sonunda düşen çalışma tamponlarını (`LfFrame`, `Blocks`, `CdefState`,
/// `LrState::mask`, `Scratch`, `RefMvs::grid` ve `rp_proj`, `LrCdefBackup`) ve referans
/// penceresinden düşen `RefFrame` düzlem/segmap/tmvs vektörlerini iş parçacığı sayısına
/// orantılı sabit kapasiteyle saklayarak işletim sistemine tekrar tekrar `mmap`/`munmap`
/// + sayfa sıfırlama çağrısı yapılmasını önler.
///
/// `Config::allocation_limit` (`butce.has_limit() == true`) etkin olduğunda havuz
/// devre dışı bırakılır; böylece boşta bekleyen hiçbir tampon bütçe sınırını
/// daraltmaz.
pub(crate) struct DecodeBufPool {
    max_work_bufs: usize,
    max_u8_vecs: usize,
    max_segmap_vecs: usize,
    max_tmvs_vecs: usize,
    inner: crate::sync::Mutex<BufPoolInner>,
}

impl DecodeBufPool {
    pub(crate) fn new(n_threads: usize) -> Self {
        let max_work_bufs = if n_threads <= 1 {
            2
        } else {
            n_threads.clamp(2, 4)
        };
        let max_u8_vecs = if n_threads <= 1 {
            6
        } else {
            (n_threads * 3 / 2).clamp(6, 12)
        };
        let max_segmap_vecs = if n_threads <= 1 {
            4
        } else {
            n_threads.clamp(4, 8)
        };
        let max_tmvs_vecs = if n_threads <= 1 {
            2
        } else {
            n_threads.clamp(2, 4)
        };
        Self {
            max_work_bufs,
            max_u8_vecs,
            max_segmap_vecs,
            max_tmvs_vecs,
            inner: crate::sync::Mutex::new(BufPoolInner {
                work_bufs: Vec::with_capacity(max_work_bufs),
                u8_vecs: Vec::with_capacity(max_u8_vecs),
                segmap_vecs: Vec::with_capacity(max_segmap_vecs),
                tmvs_vecs: Vec::with_capacity(max_tmvs_vecs),
            }),
        }
    }

    pub(crate) fn clear(&self) {
        let mut g = self.inner.lock();
        g.work_bufs.clear();
        g.u8_vecs.clear();
        g.segmap_vecs.clear();
        g.tmvs_vecs.clear();
    }

    fn take_work_bufs(&self, has_limit: bool) -> WorkFrameBufs {
        if has_limit {
            return WorkFrameBufs::default();
        }
        let mut g = self.inner.lock();
        g.work_bufs.pop().unwrap_or_default()
    }

    fn put_work_bufs(&self, entry: WorkFrameBufs, has_limit: bool) {
        if has_limit {
            return;
        }
        let mut g = self.inner.lock();
        if g.work_bufs.len() < self.max_work_bufs {
            g.work_bufs.push(entry);
        }
    }

    pub(crate) fn take_segmap_vec(&self, len: usize, has_limit: bool) -> Vec<u8> {
        if has_limit || len == 0 {
            return Vec::new();
        }
        let mut g = self.inner.lock();
        if let Some(pos) = g.segmap_vecs.iter().rposition(|v| v.capacity() == len) {
            return g.segmap_vecs.swap_remove(pos);
        }
        if let Some(pos) = g
            .segmap_vecs
            .iter()
            .rposition(|v| v.capacity() >= len && v.capacity() <= len.saturating_mul(2))
        {
            return g.segmap_vecs.swap_remove(pos);
        }
        Vec::new()
    }

    pub(crate) fn put_segmap_vec(&self, v: Vec<u8>, has_limit: bool) {
        if has_limit || v.capacity() == 0 {
            return;
        }
        let mut g = self.inner.lock();
        if g.segmap_vecs.len() < self.max_segmap_vecs {
            g.segmap_vecs.push(v);
        }
    }

    pub(crate) fn take_u8_triple(&self, lens: [usize; 3], has_limit: bool) -> [Vec<u8>; 3] {
        if has_limit {
            return [Vec::new(), Vec::new(), Vec::new()];
        }
        let mut g = self.inner.lock();
        let mut take_one = |len: usize| -> Vec<u8> {
            if len == 0 {
                return Vec::new();
            }
            if let Some(pos) = g.u8_vecs.iter().rposition(|v| v.capacity() == len) {
                return g.u8_vecs.swap_remove(pos);
            }
            if let Some(pos) = g
                .u8_vecs
                .iter()
                .rposition(|v| v.capacity() >= len && v.capacity() <= len.saturating_mul(2))
            {
                return g.u8_vecs.swap_remove(pos);
            }
            Vec::new()
        };
        [take_one(lens[0]), take_one(lens[1]), take_one(lens[2])]
    }

    pub(crate) fn put_u8_vec(&self, v: Vec<u8>, has_limit: bool) {
        if has_limit || v.capacity() == 0 {
            return;
        }
        let mut g = self.inner.lock();
        if g.u8_vecs.len() < self.max_u8_vecs {
            g.u8_vecs.push(v);
        }
    }

    pub(crate) fn take_tmvs_vec(&self, len: usize, has_limit: bool) -> Vec<TemporalBlock> {
        if has_limit || len == 0 {
            return Vec::new();
        }
        let mut g = self.inner.lock();
        if let Some(pos) = g
            .tmvs_vecs
            .iter()
            .rposition(|v| v.capacity() >= len && v.capacity() <= len.saturating_mul(2))
        {
            return g.tmvs_vecs.swap_remove(pos);
        }
        Vec::new()
    }

    pub(crate) fn recycle_segmap_tmvs(
        &self,
        segmap: Vec<u8>,
        tmvs: Vec<TemporalBlock>,
        has_limit: bool,
    ) {
        if has_limit {
            return;
        }
        let mut g = self.inner.lock();
        if segmap.capacity() > 0 && g.segmap_vecs.len() < self.max_segmap_vecs {
            g.segmap_vecs.push(segmap);
        }
        if tmvs.capacity() > 0 && g.tmvs_vecs.len() < self.max_tmvs_vecs {
            g.tmvs_vecs.push(tmvs);
        }
    }

    pub(crate) fn recycle_pic(&self, pic: PicBuf, has_limit: bool) {
        if has_limit {
            return;
        }
        let mut g = self.inner.lock();
        for v in pic.data {
            if v.capacity() > 0 && g.u8_vecs.len() < self.max_u8_vecs {
                g.u8_vecs.push(v);
            }
        }
    }

    pub(crate) fn recycle_ref_frame(&self, rf: RefFrame, has_limit: bool) {
        if has_limit {
            return;
        }
        let mut g = self.inner.lock();
        for v in rf.pic.data {
            if v.capacity() > 0 && g.u8_vecs.len() < self.max_u8_vecs {
                g.u8_vecs.push(v);
            }
        }
        if rf.segmap.capacity() > 0 && g.segmap_vecs.len() < self.max_segmap_vecs {
            g.segmap_vecs.push(rf.segmap);
        }
        if rf.tmvs.capacity() > 0 && g.tmvs_vecs.len() < self.max_tmvs_vecs {
            g.tmvs_vecs.push(rf.tmvs);
        }
    }
}

/// Kare çözme durumu.
/// Araç kapsamı sayaçları (tanılama).
///
/// **Tanılama amaçlıdır**: sayaçlar çözme davranışını değiştirmez, üretim
/// crate'ine bağımlılık eklemez ve API'nin anlamını değiştirmez; yalnız
/// [`decode_frame`] dönüşüne eklenir. "Blok" sayıları çözülen blok
/// sayısıdır; "kare" bayrakları `ToolCounts::frame_flags` ile raporlanır.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ToolCounts {
    /// Palette (luma) kullanan blok sayısı.
    pub palette_y: u64,
    /// Palette (kroma) kullanan blok sayısı.
    pub palette_uv: u64,
    /// CfL tahmini kullanan kroma bloğu sayısı.
    pub cfl: u64,
    /// Filter intra kullanan blok sayısı.
    pub filter_intra: u64,
    /// Yönlü (Z1/Z2/Z3) tahmin kullanan blok sayısı.
    pub directional: u64,
    /// Sıfırdan farklı açı deltası taşıyan blok sayısı.
    pub angle_delta: u64,
    /// Kayıpsız segment (WHT) ile çözülen blok sayısı.
    pub lossless_wht: u64,
    /// Kuantizasyon matrisi (QM) etkinken çözülen blok sayısı.
    pub qmatrix: u64,
    /// Sıfırdan farklı delta q okuyan süperblok sayısı.
    pub delta_q: u64,
    /// Sıfırdan farklı delta lf okuyan süperblok sayısı.
    pub delta_lf: u64,
    /// `seg_id != 0` olan blok sayısı (segmentasyon fiilen kullanıldı).
    pub seg_block: u64,
    /// Tx boyutu blok için azami boyuttan **küçük** (örtük bölünme) blok sayısı.
    pub tx_split: u64,
    /// Intrabc (blok içi kopyalama) ile çözülen blok sayısı.
    pub intrabc: u64,
    /// Intrabc dv artık değeri okuyan blok sayısı.
    pub intrabc_residual: u64,
    /// Çözülen blok sayısı (payda).
    pub blocks: u64,
    /// Kare 128×128 superblock kullanıyor.
    pub sb128_frame: bool,
    /// Kare birden fazla tile içeriyor.
    pub multi_tile_frame: bool,
    /// Karede segmentasyon etkin.
    pub segmentation_frame: bool,
    /// Karede kuantizasyon matrisi etkin.
    pub qmatrix_frame: bool,
    /// Tek referanslı inter blok sayısı.
    pub inter_single_ref: u64,
    /// Temporal (mfmv) aday kullanan `refmvs` çağrısı sayısı.
    pub temporal_candidate: u64,
    /// NEWMV blok sayısı.
    pub newmv: u64,
    /// `dual_filter` ile okunan blok sayısı (ikinci filtre sembolü).
    pub dual_filter: u64,
    /// Alt-8×8 kroma türetimi yapan blok sayısı.
    pub sub8x8_chroma: u64,
    /// `primary_ref_frame`'den CDF yükleyen kare sayısı.
    pub cdf_load: u64,
    /// Skip mode kullanan blok sayısı.
    pub skip_mode: u64,
    /// Compound blok sayısı (bütün türler).
    pub compound: u64,
    /// Ortalama compound (`COMP_INTER_AVG`) blok sayısı.
    pub comp_avg: u64,
    /// Mesafe ağırlıklı compound (`COMP_INTER_WEIGHTED_AVG`) blok sayısı.
    pub comp_weighted: u64,
    /// Wedge compound (`COMP_INTER_WEDGE`) blok sayısı.
    pub comp_wedge: u64,
    /// Fark ağırlıklı compound (`COMP_INTER_SEG`) blok sayısı.
    pub comp_seg: u64,
    /// Interintra (smooth/blend) blok sayısı.
    pub interintra_blend: u64,
    /// Interintra wedge blok sayısı.
    pub interintra_wedge: u64,
    /// OBMC blok sayısı.
    pub obmc: u64,
    /// Yerel warp (MM_WARP) blok sayısı.
    pub warp_local: u64,
    /// GLOBALMV blok sayısı.
    pub globalmv: u64,
    /// NEARMV (tek referans) blok sayısı.
    pub nearmv: u64,
    /// Ölçekli referans (svc.scale != 0) kullanan blok sayısı.
    pub scaled_ref: u64,
    /// CDEF ile filtrelenen 8×8 blok sayısı (en az bir düzlem).
    pub cdef_8x8_filtered: u64,
    /// CDEF luma filtre çağrısı sayısı (`fb[0]`).
    pub cdef_luma_blocks: u64,
    /// CDEF kroma filtre çağrısı sayısı (`fb[1]`/`fb[2]`, U ve V ayrı).
    pub cdef_chroma_blocks: u64,
    /// `cdef_idx == -1` ya da iki güç de 0 olduğu için atlanan 64×64 sayısı.
    pub cdef_skipped64: u64,
    /// `noskip_mask` sıfır olduğu için atlanan 8×8 sayısı.
    pub cdef_skipped8: u64,
    /// `adjust_strength` sıfır döndürdüğü için luma filtresi atlanan 8×8.
    pub cdef_adj_zero: u64,
    /// 4:2:2 kroma birincil yön tablosu (`uv_dirs[1]`) kullanılan 8×8.
    pub cdef_uv422_pri: u64,
    /// `cdef_find_dir` yön dağılımı (0..7).
    pub cdef_dir_hist: [u64; 8],
    /// LR sözdizimi: okunan birim sayısı (`read_restoration_info`).
    pub lr_units: u64,
    /// LR sözdizimi: `SWITCHABLE` karede okunan birim sayısı.
    pub lr_switchable_units: u64,
    /// LR sözdizimi: türü `NONE` çıkan birim sayısı.
    pub lr_none_units: u64,
    /// LR sözdizimi: Wiener birimi sayısı.
    pub lr_wiener_units: u64,
    /// LR sözdizimi: SGR birimi sayısı.
    pub lr_sgr_units: u64,
    /// LR şerit çağrısı: Wiener 7 dokunuş (`lr.wiener[0]`).
    pub lr_wiener7: u64,
    /// LR şerit çağrısı: Wiener 5 dokunuş (`lr.wiener[1]`).
    pub lr_wiener5: u64,
    /// LR şerit çağrısı: SGR 5×5 (`lr.sgr[0]`).
    pub lr_sgr5x5: u64,
    /// LR şerit çağrısı: SGR 3×3 (`lr.sgr[1]`).
    pub lr_sgr3x3: u64,
    /// LR şerit çağrısı: SGR karma (`lr.sgr[2]`).
    pub lr_sgrmix: u64,
    /// LR luma birim boyutu (piksel; 0 = luma LR kapalı).
    pub lr_unit_y: u64,
    /// LR kroma birim boyutu (piksel; 0 = kroma LR kapalı).
    pub lr_unit_uv: u64,
    /// Superres karesi (1 = `width[0] != width[1]`, büyütme uygulandı).
    pub sr_frame: u64,
    /// Superres paydası (`width_scale_denominator`, 9..16; 0 = superres yok).
    pub sr_denom: u64,
    /// Superres + LR (büyütülmüş karede LR).
    pub sr_lr: u64,
    /// Superres'li (büyütülmüş) referanstan ölçekli MC yapan blok sayısı.
    pub sr_ref_scaled: u64,
    /// Film grain **bu çıktıya** uygulandı (1; `has_grain` ve
    /// `Config::apply_grain`). Kare çözümüne değil çıktıya bağlıdır:
    /// `show_existing` çıktısında da dolar (diğer sayaçlar orada sıfırdır).
    pub fg_applied: u64,
    /// Uygulanan grain parametreleri yuvadan yüklendi (`update = 0`).
    pub fg_update0: u64,
    /// Uygulanan grain'de örtüşme (`overlap_flag`).
    pub fg_overlap: u64,
    /// Uygulanan grain'de `chroma_scaling_from_luma`.
    pub fg_csfl: u64,
    /// Uygulanan grain'de kısıtlı aralık kırpması.
    pub fg_clip: u64,
    /// Uygulanan grain'in AR gecikmesi + 1 (0 = grain yok).
    pub fg_ar_lag1: u64,
    /// Uygulanan grain'de luma ölçekleme noktası var (`num_y_points != 0`).
    pub fg_y: u64,
    /// Uygulanan grain'de kroma ölçekleme noktası var.
    pub fg_uv: u64,
}

impl ToolCounts {
    /// Başka bir tile'ın blok/sözdizimi araç sayaçlarını bu sayaçlara ekler.
    pub(crate) fn merge_from(&mut self, other: &Self) {
        self.palette_y += other.palette_y;
        self.palette_uv += other.palette_uv;
        self.cfl += other.cfl;
        self.filter_intra += other.filter_intra;
        self.directional += other.directional;
        self.angle_delta += other.angle_delta;
        self.lossless_wht += other.lossless_wht;
        self.qmatrix += other.qmatrix;
        self.delta_q += other.delta_q;
        self.delta_lf += other.delta_lf;
        self.seg_block += other.seg_block;
        self.tx_split += other.tx_split;
        self.intrabc += other.intrabc;
        self.intrabc_residual += other.intrabc_residual;
        self.blocks += other.blocks;
        self.inter_single_ref += other.inter_single_ref;
        self.temporal_candidate += other.temporal_candidate;
        self.newmv += other.newmv;
        self.dual_filter += other.dual_filter;
        self.sub8x8_chroma += other.sub8x8_chroma;
        self.skip_mode += other.skip_mode;
        self.compound += other.compound;
        self.comp_avg += other.comp_avg;
        self.comp_weighted += other.comp_weighted;
        self.comp_wedge += other.comp_wedge;
        self.comp_seg += other.comp_seg;
        self.interintra_blend += other.interintra_blend;
        self.interintra_wedge += other.interintra_wedge;
        self.obmc += other.obmc;
        self.warp_local += other.warp_local;
        self.globalmv += other.globalmv;
        self.nearmv += other.nearmv;
        self.scaled_ref += other.scaled_ref;
        self.sr_ref_scaled += other.sr_ref_scaled;
    }
}

struct Dec<'a> {
    seq: &'a SequenceHeader,
    hdr: &'a FrameHeader,
    dsp: &'a DspDispatch,
    /// F3c: bu karenin bit derinliği tablosu — `Dec::new`'de **bir kez**
    /// seçilir; blok döngülerinde bit derinliği seçimi yoktur (DOG-05).
    bdsp: &'a BitDepthDsp,
    bd: BitDepth,
    chroma: ChromaFormat,
    ss_hor: usize,
    ss_ver: usize,
    has_chroma: bool,
    bw: usize,
    bh: usize,
    /// `f->w4` = (genişlik + 3) >> 2 (loop filter ızgarası).
    w4: usize,
    /// `f->h4` = (yükseklik + 3) >> 2.
    h4: usize,
    sb128: bool,
    sb_shift: u32,
    sb_step: usize,
    pic: PicBuf,
    blocks: Blocks,
    above_base: usize,
    sb128w: usize,
    bx: usize,
    by: usize,
    col_start: usize,
    col_end: usize,
    row_start: usize,
    row_end: usize,
    last_qidx: i32,
    last_delta_lf: [i8; 4],
    /// Kare geneli LF seviye tablosu (`f->lf.lvl`).
    frame_lflvl: crate::decode::loopfilter::LfLvlAll,
    /// Süper blok için geçerli LF seviye tablosu (`ts->lflvl`).
    lflvl: crate::decode::loopfilter::LfLvlAll,
    /// Deblocking durumu (loop filter seviyesi sıfırsa `None`).
    lf: Option<crate::decode::loopfilter::LfFrame>,
    /// CDEF kare düzeyi girdileri (`noskip_mask` + `cdef_idx`).
    cdef_state: crate::decode::cdef::CdefState,
    /// Loop restoration birim ızgarası ve `lr_ref` (S2c.2b; LR kapalıysa `None`).
    lr: Option<crate::decode::lr::LrState>,
    /// LR sözdizimi sayaçları (tanılama).
    lr_counts: crate::decode::lr::LrSyntaxCounts,
    dq: [[[u16; 2]; 3]; 8],
    frame_dq: [[[u16; 2]; 3]; 8],
    cdef_idx: [i32; 4],
    scratch: Scratch,
    segmap: Vec<u8>,
    segmap_stride: usize,
    segmap_x_base: usize,
    segmap_y_base: usize,
    frame_segmap_stride: usize,
    pal: Palette,
    /// Çözülen kare sırası (DSP izi; yalnız tanılama).
    frame_no: u32,
    /// Uzamsal MV ızgarası (`refmvs`).
    refmvs: RefMvs,
    /// Referans yuvaları (piksel + CDF + segment + tmvs).
    refs: &'a [Option<Arc<RefFrame>>; 8],
    /// `f->prev_segmap` (birincil referanstan, boyut uyuyorsa).
    prev_segmap: Option<Arc<Vec<u8>>>,
    /// `f->refpoc[7]` — 7 referansın `frame_offset`'ı.
    refpoc: [u8; 7],
    /// `f->refrefpoc[7][7]`.
    refrefpoc: [[u8; 7]; 7],
    /// `t->tl_4x4_filter` — önceki inter bloğun 2B filtresi.
    tl_4x4_filter: u8,
    /// `f->svc[7][2]` — ölçekli referans ölçek/adım çarpanları.
    svc: [[ScalableMotionParams; 2]; 7],
    /// `f->gmv_warp_allowed[7]`.
    gmv_warp_allowed: [bool; 7],
    /// Global motion shear parametreleri (`u.abcd`).
    gmv_warp: [[i16; 4]; 7],
    /// `f->jnt_weights[7][7]` — mesafe ağırlıklı compound ağırlıkları.
    jnt_weights: [[u8; 7]; 7],
    tools: ToolCounts,
    sb128_frame: bool,
    multi_tile_frame: bool,
    segmentation_frame: bool,
    qmatrix_frame: bool,
}

/// İki kenar ağacı (dav1d `dav1d_intra_edge_tree`) — süreçte bir kez kurulur.
static EDGE_TREES: OnceLock<[EdgeTree; 2]> = OnceLock::new();

fn edge_trees() -> &'static [EdgeTree; 2] {
    EDGE_TREES.get_or_init(|| [EdgeTree::build(BL_128X128), EdgeTree::build(BL_64X64)])
}

/// `iclip_u8`.
#[inline]
fn clip_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// `init_quant_tables` (`decode.c:54-74`).
#[allow(clippy::too_many_arguments)]
fn init_quant_tables(hdr: &FrameHeader, qidx: i32, out: &mut [[[u16; 2]; 3]; 8], hbd: usize) {
    let n_seg = if hdr.segmentation.enabled { 8 } else { 1 };
    for (i, row) in out.iter_mut().take(n_seg).enumerate() {
        let yac = if hdr.segmentation.enabled {
            clip_u8(qidx + i32::from(hdr.segmentation.seg_data.d[i].delta_q))
        } else {
            clip_u8(qidx)
        };
        let ydc = clip_u8(i32::from(yac) + i32::from(hdr.quant.ydc_delta));
        let uac = clip_u8(i32::from(yac) + i32::from(hdr.quant.uac_delta));
        let udc = clip_u8(i32::from(yac) + i32::from(hdr.quant.udc_delta));
        let vac = clip_u8(i32::from(yac) + i32::from(hdr.quant.vac_delta));
        let vdc = clip_u8(i32::from(yac) + i32::from(hdr.quant.vdc_delta));
        row[0] = [
            t::DQ_TBL[hbd][usize::from(ydc)][0],
            t::DQ_TBL[hbd][usize::from(yac)][1],
        ];
        row[1] = [
            t::DQ_TBL[hbd][usize::from(udc)][0],
            t::DQ_TBL[hbd][usize::from(uac)][1],
        ];
        row[2] = [
            t::DQ_TBL[hbd][usize::from(vdc)][0],
            t::DQ_TBL[hbd][usize::from(vac)][1],
        ];
    }
}

/// Kroma katsayı bağlamı indeksi (dav1d `recon_tmpl.c:831`):
/// `cbx4 = bx4 >> ss_hor`, `cby4 = by4 >> ss_ver`.
#[inline]
fn chroma_ctx4(b4: usize, ss: usize) -> usize {
    b4 >> ss
}

/// `neg_deinterleave` (`decode.c:169-189`).
fn neg_deinterleave(diff: i32, reference: i32, max: i32) -> i32 {
    if reference == 0 {
        return diff;
    }
    if reference >= max - 1 {
        return max - diff - 1;
    }
    if 2 * reference < max {
        if diff <= 2 * reference {
            if diff & 1 != 0 {
                reference + ((diff + 1) >> 1)
            } else {
                reference - (diff >> 1)
            }
        } else {
            diff
        }
    } else if diff <= 2 * (max - reference - 1) {
        if diff & 1 != 0 {
            reference + ((diff + 1) >> 1)
        } else {
            reference - (diff >> 1)
        }
    } else {
        max - (diff + 1)
    }
}

/// `get_cur_frame_segid` (`env.h:439-460`).
fn get_cur_frame_segid(
    segmap: &[u8],
    stride: usize,
    by: usize,
    bx: usize,
    have_top: bool,
    have_left: bool,
) -> (i32, u8) {
    let base = by * stride + bx;
    if have_left && have_top {
        let l = i32::from(segmap[base - 1]);
        let a = i32::from(segmap[base - stride]);
        let al = i32::from(segmap[base - stride - 1]);
        let ctx = if l == a && al == l {
            2u8
        } else if l == a || al == l || a == al {
            1
        } else {
            0
        };
        (if a == al { a } else { l }, ctx)
    } else {
        let value = if have_left {
            i32::from(segmap[base - 1])
        } else if have_top {
            i32::from(segmap[base - stride])
        } else {
            0
        };
        (value, 0)
    }
}

/// Tanılama: **kullanılan** kenar baytlarının özeti (varsayılan kapalı).
///
/// `prepare_intra_edges` yalnız modun gerektirdiği baytları yazar
/// (`ipred_prepare_tmpl.c:55-108`); kalan baytlar tamponda önceki bloktan
/// kalır ve iki tarafta farklı olabilir. Özet bu yüzden yalnız kullanılan
/// aralıkları kapsar (T42: VERT modunda sol kenar).
const IP_TOP: u8 = 1;
const IP_LEFT: u8 = 2;
const IP_TOPRIGHT: u8 = 4;
const IP_BOTTOMLEFT: u8 = 8;
const IP_TOPLEFT: u8 = 16;
const IPRED_EDGE_NEEDS: [u8; 14] = [
    IP_TOP | IP_LEFT,
    IP_TOP,
    IP_LEFT,
    IP_LEFT,
    IP_TOP,
    0,
    IP_TOP | IP_TOPRIGHT | IP_TOPLEFT,
    IP_LEFT | IP_TOP | IP_TOPLEFT,
    IP_LEFT | IP_BOTTOMLEFT | IP_TOPLEFT,
    IP_LEFT | IP_TOP,
    IP_LEFT | IP_TOP,
    IP_LEFT | IP_TOP,
    IP_LEFT | IP_TOP | IP_TOPLEFT,
    IP_LEFT | IP_TOP | IP_TOPLEFT,
];

fn edge_hash_used(
    bytes: &[u8],
    topleft: usize,
    w: usize,
    h: usize,
    mode: usize,
    bpp: usize,
) -> u64 {
    let needs = IPRED_EDGE_NEEDS
        .get(mode)
        .copied()
        .unwrap_or(IP_LEFT | IP_TOP | IP_TOPLEFT);
    let mut hash = 1_469_598_103_934_665_603u64;
    let mut mix = |i: isize| {
        let off = (topleft as isize + i * bpp as isize) as usize;
        let v = if bpp == 2 {
            u64::from(u16::from_le_bytes([bytes[off], bytes[off + 1]]))
        } else {
            u64::from(bytes[off])
        };
        hash ^= v;
        hash = hash.wrapping_mul(1_099_511_628_211);
    };
    if needs & IP_BOTTOMLEFT != 0 {
        for i in (h + 1..=2 * h).rev() {
            mix(-(i as isize));
        }
    }
    if needs & IP_LEFT != 0 {
        for i in (1..=h).rev() {
            mix(-(i as isize));
        }
    }
    if needs & IP_TOPLEFT != 0 {
        mix(0);
    }
    if needs & IP_TOP != 0 {
        for i in 1..=w {
            mix(i as isize);
        }
    }
    if needs & IP_TOPRIGHT != 0 {
        for i in w + 1..=2 * w {
            mix(i as isize);
        }
    }
    hash
}

/// Tanılama: 16-bit ara tampon (warp8x8t) özeti.
fn pred_hash_i16(data: &[i16], off: usize, stride: usize, w: usize, h: usize) -> u64 {
    let mut hash = 1_469_598_103_934_665_603u64;
    for y in 0..h {
        for x in 0..w {
            let v = u16::from_ne_bytes(data[off + y * stride + x].to_ne_bytes());
            hash ^= u64::from(v);
            hash = hash.wrapping_mul(1_099_511_628_211);
        }
    }
    hash
}

/// Tanılama: 16 bit ara tampon özeti (`mct`, `avg`, `mask`).
fn hash_i16(data: &[i16]) -> u64 {
    let mut hash = 1_469_598_103_934_665_603u64;
    for &v in data {
        hash ^= u64::from(v as u16);
        hash = hash.wrapping_mul(1_099_511_628_211);
    }
    hash
}

/// Tanılama: warp satırı (`dsp_trace.c` `W` biçimi).
#[allow(clippy::too_many_arguments)]
fn warp_trace_line(
    frame: u32,
    fname: &str,
    bx: usize,
    by: usize,
    hash: u64,
    x: i64,
    y: i64,
    plane: usize,
    mx: i32,
    my: i32,
    refslot: u32,
    sx: u32,
    sy: u32,
    abcd: [i16; 4],
    matrix: [i32; 6],
) {
    let _ = matrix;
    crate::trace_eprintln!(
        "W {frame} {fname} {bx} {by} {hash:016x} {x} {y} {plane} {mx} {my} {refslot} {sx} {sy} {} {} {} {}",
        abcd[0],
        abcd[1],
        abcd[2],
        abcd[3]
    );
}

/// Tanılama: OBMC blend satırı (`dsp_trace.c` `B` biçimi).
fn blend_trace_line(frame: u32, fname: &str, bx: usize, by: usize, w: usize, h: usize, hash: u64) {
    crate::trace_eprintln!("B {frame} {fname} {bx} {by} {w} {h} {hash:016x}");
}

/// Tanılama: MC çağrı satırı (`dsp_trace.c` `D` biçimi).
#[allow(clippy::too_many_arguments)]
fn mc_trace_line(
    frame: u32,
    fname: &str,
    filter: usize,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    p1: i32,
    p2: i32,
    refpoc: u32,
    sx: u32,
    sy: u32,
    hash: u64,
) {
    crate::trace_eprintln!(
        "D {frame} {fname} {filter} {bx} {by} {w} {h} {mx} {my} {p1} {p2} {refpoc} 0 {sx} {sy} {hash:016x}"
    );
}

/// Tanılama: tahmin edilen bloğun FNV-1a özeti (varsayılan kapalı).
fn pred_hash(data: &[u8], off: usize, stride: usize, w: usize, h: usize, bpp: usize) -> u64 {
    let mut hash = 1_469_598_103_934_665_603u64;
    for y in 0..h {
        for x in 0..w {
            let i = off + y * stride + x * bpp;
            let v = if bpp == 2 {
                u64::from(u16::from_le_bytes([data[i], data[i + 1]]))
            } else {
                u64::from(data[i])
            };
            hash ^= v;
            hash = hash.wrapping_mul(1_099_511_628_211);
        }
    }
    hash
}

/// dav1d `decode.c:281`: tek üst örnek özel yolu **yalnız alt 32 bite**
/// bakar; `masks[0]` bit 32 (sağ-üst örnek) özel yolu kapatmaz. İkinci
/// koşul sol-üst örneğin yokluğudur (`!(masks[1] >> 32)`).
fn warp_uses_single_top_sample(masks: &[u64; 2]) -> bool {
    masks[0] as u32 == 1 && (masks[1] >> 32) == 0
}

/// `derive_warpmv` örnek seçimi — dav1d `decode.c:315-329` birebir.
///
/// Eşik dışı örnekleri (`mvd[i] == -1`) dizinin sonundaki geçerli
/// örneklerle değiştirir. dav1d döngüsü her yer değiştirmeden sonra
/// **`i++`, `j--`** yapar (`for (…; k++, i++, j--)`); bu artırma/azaltma
/// eksikken aynı geçerli örnek iki kez kopyalanıyordu (iz: kare 1, blok
/// (1524, 560), `ret = 2` → dav1d `[(184,-72), (-72,56)]`, Rust
/// `[(184,-72), (184,-72)]`).
///
/// `assert(i != j)` ve `if (i > j) break;` sırası korunur. `i` daima
/// geçersiz (`-1`), `j` daima geçerli bir örneğe bakar; aynı indeks ikisi
/// olamaz, bu yüzden `debug_assert!` hiç düşmez. Kopyalama yalnız `i < j`
/// iken yapıldığından `j -= 1` öncesi `j >= 1`'dir; `usize` taşması
/// oluşmaz (`warp_ornek_secimi_dav1d_dongusu` testi bütün `mvd`
/// desenlerini dener).
fn select_warp_samples(pts: &mut [[[i32; 2]; 2]; 8], np: usize, ret: usize, mvd: &mut [i32; 8]) {
    let mut i = 0usize;
    let mut j = np - 1;
    for _ in 0..np - ret {
        while mvd[i] != -1 {
            i += 1;
        }
        while mvd[j] == -1 {
            j -= 1;
        }
        debug_assert!(i != j);
        if i > j {
            break;
        }
        // Eşik dışı örneğin yerine sonundaki geçerli örneği koy.
        mvd[i] = mvd[j];
        pts[i] = pts[j];
        i += 1;
        j -= 1;
    }
}

/// DSP çağrı izi ayarları — dav1d `DSP_TRACE`, `DSP_TRACE_FRAME`,
/// `DSP_TRACE_PLANE`, `DSP_TRACE_X/Y` karşılığı.
///
/// **Varsayılan kapalıdır**: `AV1OXIDE_DSP_TRACE` yoksa hiçbir satır yazılmaz.
/// Ayarlar bir kez (`OnceLock`) okunur; çağrı başına maliyet tek atomik yüktür.
/// Süzgeç adları Rust tarafında `AV1OXIDE_DSP_*`dir; `DSP_TRACE_*` de kabul
/// edilir, böylece iki taraf aynı ortam değişkenleriyle süzülebilir.
#[derive(Clone, Copy)]
struct DspTrace {
    on: bool,
    frame: i64,
    plane: i64,
    x: i64,
    y: i64,
    blk_trace: bool,
    lf_blk: bool,
    lf_dlf: bool,
    refmvs_trace: bool,
    lf_off: bool,
    cdef_off: bool,
    lr_off: bool,
}

#[cfg(feature = "std")]
fn dsp_trace_env(birincil: &str, ikincil: &str) -> i64 {
    std::env::var(birincil)
        .ok()
        .or_else(|| std::env::var(ikincil).ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(-1)
}

#[inline(always)]
fn dsp_trace() -> &'static DspTrace {
    #[cfg(feature = "std")]
    {
        static T: OnceLock<DspTrace> = OnceLock::new();
        T.get_or_init(|| DspTrace {
            on: std::env::var_os("AV1OXIDE_DSP_TRACE").is_some(),
            frame: dsp_trace_env("AV1OXIDE_DSP_FRAME", "DSP_TRACE_FRAME"),
            plane: dsp_trace_env("AV1OXIDE_DSP_PLANE", "DSP_TRACE_PLANE"),
            x: dsp_trace_env("AV1OXIDE_DSP_X", "DSP_TRACE_X"),
            y: dsp_trace_env("AV1OXIDE_DSP_Y", "DSP_TRACE_Y"),
            blk_trace: std::env::var_os("AV1OXIDE_TRACE").is_some(),
            lf_blk: std::env::var_os("AV1OXIDE_LF_BLK").is_some(),
            lf_dlf: std::env::var_os("AV1OXIDE_LF_DLF").is_some(),
            refmvs_trace: std::env::var_os("AV1OXIDE_REFMVS_TRACE").is_some(),
            lf_off: std::env::var_os("AV1OXIDE_LF_OFF").is_some(),
            cdef_off: std::env::var_os("AV1OXIDE_CDEF_OFF").is_some(),
            lr_off: std::env::var_os("AV1OXIDE_LR_OFF").is_some(),
        })
    }
    #[cfg(not(feature = "std"))]
    {
        static DEFAULT: DspTrace = DspTrace {
            on: false,
            frame: -1,
            plane: -1,
            x: -1,
            y: -1,
            blk_trace: false,
            lf_blk: false,
            lf_dlf: false,
            refmvs_trace: false,
            lf_off: false,
            cdef_off: false,
            lr_off: false,
        };
        &DEFAULT
    }
}

impl DspTrace {
    /// Kare süzgeci (`DSP_TRACE_FRAME`).
    fn frame_ok(&self, frame: u32) -> bool {
        self.on && (self.frame < 0 || self.frame == i64::from(frame))
    }

    /// Kare + düzlem + x/y süzgeci (`DSP_TRACE_PLANE`, `DSP_TRACE_X/Y`).
    fn pos_ok(&self, frame: u32, plane: usize, x: usize, y: usize) -> bool {
        self.frame_ok(frame)
            && (self.plane < 0 || self.plane == plane as i64)
            && (self.x < 0 || self.x == x as i64)
            && (self.y < 0 || self.y == y as i64)
    }
}

/// `mct` / `mct_scaled` satırı — `scripts/oracle/dsp_trace.c` `N` biçimi.
/// Çıktı **16 bit ara tampondur** (`prep`), özet `w*h` i16 üzerinden alınır.
#[allow(clippy::too_many_arguments)]
fn mct_trace_line(
    frame: u32,
    fname: &str,
    filter: usize,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    p1: i32,
    p2: i32,
    refpoc: u32,
    sx: u32,
    sy: u32,
    hash: u64,
) {
    crate::trace_eprintln!(
        "N {frame} {fname} {filter} {bx} {by} {w} {h} {mx} {my} {p1} {p2} {refpoc} 0 {sx} {sy} {hash:016x}"
    );
}

/// `avg` / `w_avg` satırı — `dsp_trace.c` `A` biçimi.
#[allow(clippy::too_many_arguments)]
fn avg_trace_line(
    frame: u32,
    fname: &str,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    weight: i32,
    hash_dst: u64,
    hash_t1: u64,
    hash_t2: u64,
) {
    crate::trace_eprintln!(
        "A {frame} {fname} {bx} {by} {w} {h} {weight} {hash_dst:016x} {hash_t1:016x} {hash_t2:016x}"
    );
}

/// `mask` / `w_mask` satırı — `dsp_trace.c` `C` biçimi.
#[allow(clippy::too_many_arguments)]
fn mask_trace_line(
    frame: u32,
    fname: &str,
    idx: usize,
    sign: i32,
    bx: usize,
    by: usize,
    w: usize,
    h: usize,
    hash_dst: u64,
    hash_t1: u64,
    hash_t2: u64,
    hash_mask: u64,
) {
    crate::trace_eprintln!(
        "C {frame} {fname} {idx} {sign} {bx} {by} {w} {h} {hash_dst:016x} {hash_t1:016x} {hash_t2:016x} {hash_mask:016x}"
    );
}

/// `emu_edge` satırı — `dsp_trace.c` `E` biçimi.
#[allow(clippy::too_many_arguments)]
fn emu_trace_line(
    frame: u32,
    bx: usize,
    by: usize,
    bw: usize,
    bh: usize,
    iw: usize,
    ih: usize,
    x: i32,
    y: i32,
    hash: u64,
) {
    crate::trace_eprintln!("E {frame} {bx} {by} {bw} {bh} {iw} {ih} {x} {y} {hash:016x}");
}

/// itx satırı — `dsp_trace.c` `T` biçimi. Konum **gerçek hedef** (düzlem
/// pikseli) ve `DSP_TRACE_PLANE`/`DSP_TRACE_X/Y` süzgeçleri uygulanır.
#[allow(clippy::too_many_arguments)]
fn itx_trace_line(
    frame: u32,
    plane: usize,
    x: usize,
    y: usize,
    tx: u8,
    txtp: u8,
    eob: i32,
    bpc: u8,
    n: usize,
    bdmax: u32,
    cf: &[i32],
) {
    if !dsp_trace().pos_ok(frame, plane, x, y) {
        return;
    }
    let mut line = format!("T {frame} {plane} {x} {y} {tx} {txtp} {eob} {bpc} {n} {bdmax}");
    for (i, c) in cf.iter().take(n).enumerate() {
        if *c != 0 {
            let _ = core::fmt::Write::write_fmt(&mut line, format_args!(" {i}:{c}"));
        }
    }
    crate::trace_eprintln!("{line}");
}

/// Blok satırı — `dsp_trace.c` `K` biçimi.
fn block_trace_line(frame: u32, bx: usize, by: usize, bs: u8, b: &Av1Block) {
    crate::trace_eprintln!(
        "K {frame} {bx} {by} {bs} {} {} {} {} {} {} {} {} {} {} {}",
        u8::from(b.intra),
        if b.intra { 0 } else { b.comp_type },
        if b.intra { 0 } else { b.interintra_type },
        if b.intra || b.comp_type != 0 {
            0
        } else {
            b.motion_mode
        },
        if b.intra { 0 } else { b.filter2d },
        if b.intra { 0 } else { b.ref0 },
        if b.intra { 0 } else { b.ref1 },
        if b.intra { 0 } else { b.mv[0].x },
        if b.intra { 0 } else { b.mv[0].y },
        u8::from(b.skip_mode),
        u8::from(b.skip)
    );
}

/// Kare segment haritasının başlangıcı — dav1d `decode.c:3618-3672` üç dalı.
///
/// - Segmentasyon kapalı: dav1d `cur_segmap = NULL` saklar; Rust sıfır dolu
///   vektör saklar. Sonraki kare bunu önceki harita olarak görse bile
///   kopyalanan baytlar sıfırdır — dav1d'nin sıfırlanmış yeni haritasıyla
///   aynı sonucu verir.
/// - `update_map = 1`: yeni harita tahsis edilir (`:3644-3651`); içerik
///   `decode_b` sonunda blok blok yazılır. Rust sıfırla başlatır (dav1d
///   havuzdan gelen tamponu sıfırlamaz, ama her blok yazıldığı için fark
///   yoktur).
/// - `update_map = 0` ve önceki harita var: **önceki harita paylaşılır**
///   (`:3652-3657`). Rust referans sayımı yerine bayt kopyası kullanır;
///   referans yuvasına da bu harita kaydedilir (`:3690-3693`).
/// - `update_map = 0` ve önceki harita yok: sıfırlanmış yeni harita
///   (`:3658-3667`).
#[allow(dead_code)]
fn init_segmap(
    hdr: &FrameHeader,
    stride: usize,
    height: usize,
    prev: Option<&[u8]>,
    butce: &Butce,
) -> Result<Vec<u8>> {
    init_segmap_with_reuse(hdr, stride, height, prev, Vec::new(), butce)
}

fn init_segmap_with_reuse(
    hdr: &FrameHeader,
    stride: usize,
    height: usize,
    prev: Option<&[u8]>,
    old_map: Vec<u8>,
    butce: &Butce,
) -> Result<Vec<u8>> {
    let len = stride * height.max(1);
    let mut map = butce.reuse_vec(old_map, len, 0u8)?;
    if hdr.segmentation.enabled
        && !hdr.segmentation.update_map
        && let Some(prev) = prev
    {
        let n = prev.len().min(len);
        map[..n].copy_from_slice(&prev[..n]);
    }
    Ok(map)
}

/// `get_prev_frame_segid` (`decode.c:489-505`) — bloktaki en küçük segment
/// kimliği; sıfır görülünce erken çıkış, harita dışına taşma yok.
fn prev_segid_in_map(map: &[u8], stride: usize, by: usize, bx: usize, w4: usize, h4: usize) -> u8 {
    let mut seg_id = 8u8;
    for y in 0..h4.max(1) {
        let base = (by + y) * stride + bx;
        if base + w4 > map.len() {
            break;
        }
        for x in 0..w4 {
            seg_id = seg_id.min(map[base + x]);
        }
        if seg_id == 0 {
            break;
        }
    }
    if seg_id >= 8 { 0 } else { seg_id }
}

impl<'a> Dec<'a> {
    /// Kare çözücüyü kurar (kare düzeyi tablolar + tamponlar).
    #[allow(clippy::too_many_arguments)]
    fn new(
        seq: &'a SequenceHeader,
        hdr: &'a FrameHeader,
        dsp: &'a DspDispatch,
        refs: &'a [Option<Arc<RefFrame>>; 8],
        prev_segmap: Option<Arc<Vec<u8>>>,
        work_bufs: &mut WorkFrameBufs,
        buf_pool: &DecodeBufPool,
        butce: &'a Butce,
        parallel_tiles: bool,
    ) -> Result<Self> {
        let bd = match seq.hbd {
            0 => BitDepth::Eight,
            1 => BitDepth::Ten,
            2 => BitDepth::Twelve,
            _ => return Err(Error::Unsupported("geçersiz seq hbd")),
        };
        let chroma = match seq.layout.raw() {
            0 => ChromaFormat::Monochrome,
            1 => ChromaFormat::C420,
            2 => ChromaFormat::C422,
            _ => ChromaFormat::C444,
        };
        let width = hdr.width[0] as usize;
        let height = hdr.height as usize;
        if width == 0 || height == 0 {
            return Err(Error::InvalidData("kare boyutu sıfır"));
        }
        let (ss_hor, ss_ver) = chroma.chroma_shift();
        let (ss_hor, ss_ver) = (ss_hor as usize, ss_ver as usize);
        // dav1d `decode.c:3561-3562`: blok gezinmesi ve blok bağlamı sınırları
        // 8 piksele yuvarlanmış `bw`/`bh` kullanır — 4 piksele yuvarlanmış
        // `w4`/`h4` **değil**. `w4`/`h4` yalnız loop filter maskesinde geçer.
        let bw = ((width + 7) >> 3) << 1;
        let bh = ((height + 7) >> 3) << 1;
        let sb128 = seq.sb128;
        let sb_shift = 4 + u32::from(sb128);
        let sb_step = 1usize << sb_shift;
        let sb128w = (bw + 31) >> 5;
        let w4 = (width + 3) >> 2;
        let h4 = (height + 3) >> 2;
        let segmap_stride = sb128w * 32;
        let segmap_height = ((bh + 31) >> 5) * 32;
        let hbd = usize::from(match bd {
            BitDepth::Eight => 0u8,
            BitDepth::Ten => 1,
            BitDepth::Twelve => 2,
        });
        let mut dq = [[[0u16; 2]; 3]; 8];
        init_quant_tables(hdr, i32::from(hdr.quant.yac), &mut dq, hbd);
        let has_limit = butce.has_limit();
        let pic_lens = PicBuf::plane_byte_lens(width, height, chroma, bd)?;
        let old_pic_data = buf_pool.take_u8_triple(pic_lens, has_limit);
        let pic = PicBuf::new_with_reuse(width, height, chroma, bd, old_pic_data, butce)?;
        let frame_lflvl = crate::decode::loopfilter::calc_lf_values(hdr, [0; 4]);
        let tiles_cover_frame = parallel_tiles
            && hdr.tiling.cols > 0
            && hdr.tiling.rows > 0
            && usize::from(hdr.tiling.col_start_sb[usize::from(hdr.tiling.cols)]) * sb_step >= bw
            && usize::from(hdr.tiling.row_start_sb[usize::from(hdr.tiling.rows)]) * sb_step >= bh;
        let need_refmvs =
            !matches!(hdr.frame_type, FrameType::Key | FrameType::Intra) || hdr.allow_intrabc;
        let old_lf = work_bufs.lf.take();
        let lf = if hdr.loopfilter.level_y[0] != 0 || hdr.loopfilter.level_y[1] != 0 {
            Some(crate::decode::loopfilter::LfFrame::new_with_reuse(
                hdr,
                w4,
                h4,
                ss_hor,
                ss_ver,
                sb128,
                old_lf,
                butce,
                tiles_cover_frame,
            )?)
        } else {
            work_bufs.lf = old_lf;
            None
        };
        let old_segmap = buf_pool.take_segmap_vec(segmap_stride * segmap_height.max(1), has_limit);
        crate::dsp::profil_bd!(bd);
        let mut dec = Self {
            seq,
            hdr,
            dsp,
            bdsp: dsp.for_bit_depth(bd),
            bd,
            chroma,
            ss_hor,
            ss_ver,
            has_chroma: chroma.has_chroma(),
            bw,
            bh,
            w4,
            h4,
            sb128,
            sb_shift,
            sb_step,
            pic,
            blocks: Blocks::new_with_reuse(
                sb128w,
                usize::from(hdr.tiling.rows),
                work_bufs.blocks.take(),
                butce,
            )?,
            above_base: 0,
            sb128w,
            bx: 0,
            by: 0,
            col_start: 0,
            col_end: 0,
            row_start: 0,
            row_end: 0,
            last_qidx: i32::from(hdr.quant.yac),
            last_delta_lf: [0; 4],
            frame_lflvl,
            lflvl: frame_lflvl,
            lf,
            dq,
            frame_dq: dq,
            cdef_idx: [-1; 4],
            cdef_state: crate::decode::cdef::CdefState::new_with_reuse(
                bw,
                bh,
                work_bufs.cdef_state.take(),
                butce,
            )?,
            lr: crate::decode::lr::LrState::new_with_reuse(
                hdr,
                chroma,
                core::mem::take(&mut work_bufs.lr_mask),
                butce,
            )?,
            lr_counts: crate::decode::lr::LrSyntaxCounts::default(),
            scratch: Scratch::new_with_reuse(bd, work_bufs.scratch.take(), butce)?,
            segmap: init_segmap_with_reuse(
                hdr,
                segmap_stride,
                segmap_height,
                prev_segmap.as_ref().map(|m| m.as_slice()),
                old_segmap,
                butce,
            )?,
            segmap_stride,
            segmap_x_base: 0,
            segmap_y_base: 0,
            frame_segmap_stride: segmap_stride,
            pal: [[0u16; 8]; 3],
            frame_no: 0,
            refmvs: if !need_refmvs {
                RefMvs::new_empty(width, height, *dsp.refmvs())
            } else {
                RefMvs::new_with_reuse(
                    width,
                    height,
                    *dsp.refmvs(),
                    core::mem::take(&mut work_bufs.refmvs_grid),
                    butce,
                    tiles_cover_frame,
                )?
            },
            refs,
            prev_segmap,
            refpoc: [0; 7],
            refrefpoc: [[0; 7]; 7],
            tl_4x4_filter: 0,
            svc: [[ScalableMotionParams::default(); 2]; 7],
            gmv_warp_allowed: [false; 7],
            gmv_warp: [[0; 4]; 7],
            jnt_weights: [[0; 7]; 7],
            tools: ToolCounts::default(),
            sb128_frame: false,
            multi_tile_frame: false,
            segmentation_frame: false,
            qmatrix_frame: false,
        };
        dec.sb128_frame = sb128;
        dec.multi_tile_frame = usize::from(hdr.tiling.cols) * usize::from(hdr.tiling.rows) > 1;
        dec.segmentation_frame = hdr.segmentation.enabled;
        dec.qmatrix_frame = hdr.quant.qm;
        // Above bağlamları kare başında sıfırlanır (`reset_context`).
        let keyframe = matches!(hdr.frame_type, FrameType::Key | FrameType::Intra);
        for ctx in &mut dec.blocks.above {
            reset_context(ctx, keyframe);
        }
        // Inter/switch karelerde referans türetilmiş durumu kur.
        if matches!(hdr.frame_type, FrameType::Inter | FrameType::Switch) {
            let mut refpoc = [0u8; 7];
            let mut refrefpoc = [[0u8; 7]; 7];
            let mut ref_tmvs: [Option<&[TemporalBlock]>; 7] = [None; 7];
            for i in 0..7 {
                let slot = usize::from(hdr.refidx[i] as u8);
                let Some(r) = refs.get(slot).and_then(|r| r.as_ref()) else {
                    return Err(Error::Unsupported("referans çözülmedi: inter referans"));
                };
                refpoc[i] = r.hdr.frame_offset;
                refrefpoc[i] = r.refpoc;
                // dav1d `decode.c:3592-3605`: `ref_mvs[i]` yalnız referans
                // karenin **kendi refmvs'i varsa** (inter/switch) ve
                // boyutlar uyuşuyorsa dolar; `c->refs[].refmvs == NULL`
                // olduğunda (anahtar/intra referans) temporal aday yoktur.
                // Rust yalnız boyuta bakıyordu: intra referansın sıfır
                // dolu `tmvs` tamponu da "var" sayılıyor, `mfmv` planına
                // fazladan referans giriyor ve projeksiyon kayıyordu
                // (T40: `test4331_4344` k9; ref 6 dav1d'de NULL).
                let ref_has_mvs = matches!(
                    r.hdr.frame_type,
                    crate::bitstream::frame_header::FrameType::Inter
                        | crate::bitstream::frame_header::FrameType::Switch
                );
                // dav1d `decode.c:3595-3597`: `ref_w` referansın kodlanan
                // genişliğinden (`ref_coded_width`); superres'li referansın
                // piksel tamponu büyütülmüş genişliktedir.
                let ref_w = ((r.hdr.width[0] as usize + 7) >> 3) << 1;
                if ref_has_mvs
                    && ref_w == dec.bw
                    && r.bh == dec.bh
                    && r.tmvs.len() == r.tmvs_stride * (r.bh / 2)
                {
                    ref_tmvs[i] = Some(&r.tmvs[..]);
                }
            }
            dec.refpoc = refpoc;
            dec.refrefpoc = refrefpoc;
            // Ölçekli referans (`decode.c:3465-3495`).
            for i in 0..7 {
                let slot = usize::from(hdr.refidx[i] as u8);
                let Some(r) = refs.get(slot).and_then(|r| r.as_ref()) else {
                    return Err(Error::Unsupported("referans çözülmedi: inter referans"));
                };
                let cur_w = hdr.width[0] as usize;
                let cur_h = hdr.height as usize;
                if cur_w != r.pic.width || cur_h != r.pic.height {
                    // `scale_fac(ref_sz, this_sz) = ((ref_sz << 14) + (this_sz >> 1)) / this_sz`
                    let scale_fac = |ref_sz: usize, this_sz: usize| -> i32 {
                        (((ref_sz << 14) + (this_sz >> 1)) / this_sz) as i32
                    };
                    dec.svc[i][0].scale = scale_fac(r.pic.width, cur_w);
                    dec.svc[i][1].scale = scale_fac(r.pic.height, cur_h);
                    dec.svc[i][0].step = (dec.svc[i][0].scale + 8) >> 4;
                    dec.svc[i][1].step = (dec.svc[i][1].scale + 8) >> 4;
                }
                let wm = &hdr.gmv[i];
                let shear = if wm.kind.raw()
                    > crate::bitstream::frame_header::WarpedMotionType::Translation.raw()
                {
                    warpv::get_shear_params(wm)
                } else {
                    None
                };
                if let Some(p) = shear {
                    dec.gmv_warp[i] = p;
                }
                dec.gmv_warp_allowed[i] = wm.kind.raw()
                    > crate::bitstream::frame_header::WarpedMotionType::Translation.raw()
                    && !hdr.force_integer_mv
                    && shear.is_some()
                    && dec.svc[i][0].scale == 0;
            }
            // Mesafe ağırlıklı compound ağırlıkları (`decode.c:3098-3131`).
            if hdr.switchable_comp_refs {
                const QUANT_DIST_WEIGHT: [[i32; 2]; 3] = [[2, 3], [2, 5], [2, 7]];
                const QUANT_DIST_LOOKUP: [[u8; 2]; 4] = [[9, 7], [11, 5], [12, 4], [13, 3]];
                for i in 0..7 {
                    let ref0poc = i32::from(refpoc[i]);
                    for (j, &p1) in refpoc.iter().enumerate().skip(i + 1) {
                        let ref1poc = i32::from(p1);
                        let cur = i32::from(hdr.frame_offset);
                        let d1 = crate::bitstream::frame_header::get_poc_diff(
                            seq.order_hint_n_bits,
                            ref0poc,
                            cur,
                        )
                        .abs()
                        .min(31);
                        let d0 = crate::bitstream::frame_header::get_poc_diff(
                            seq.order_hint_n_bits,
                            ref1poc,
                            cur,
                        )
                        .abs()
                        .min(31);
                        let order = usize::from(d0 <= d1);
                        let mut k = 0usize;
                        while k < 3 {
                            let c0 = QUANT_DIST_WEIGHT[k][order];
                            let c1 = QUANT_DIST_WEIGHT[k][1 - order];
                            let d0_c0 = d0 * c0;
                            let d1_c1 = d1 * c1;
                            if (d0 > d1 && d0_c0 < d1_c1) || (d0 <= d1 && d0_c0 > d1_c1) {
                                break;
                            }
                            k += 1;
                        }
                        dec.jnt_weights[i][j] = QUANT_DIST_LOOKUP[k.min(3)][order];
                    }
                }
            }
            dec.refmvs.init_frame(
                hdr,
                seq.order_hint_n_bits,
                &refpoc,
                &refrefpoc,
                &ref_tmvs,
                core::mem::take(&mut work_bufs.refmvs_rp_proj),
                butce,
            )?;
            dec.refmvs.load_temporal(&ref_tmvs);
        }
        Ok(dec)
    }

    /// `decode.c:1893-1900`: lossless segmentte maske tx boyları TX_4X4'e
    /// zorlanır (skip blokta bile; `read_vartx_tree` bunu yapmaz).
    fn lf_tx(&self, seg: usize, b: &Av1Block) -> (usize, usize) {
        if self.hdr.segmentation.lossless[seg] {
            (usize::from(TX_4X4), usize::from(TX_4X4))
        } else {
            (usize::from(b.max_ytx), usize::from(b.uvtx))
        }
    }

    /// `dav1d_create_lf_mask_intra/inter` çağrısı (`decode.c:1224,1893-1900`).
    ///
    /// Karede deblocking kapalıysa hiçbir şey yapmaz (dav1d de maskeyi yalnız
    /// `loopfilter.level_y` sıfırdan farklıysa üretir); aksi hâlde kenar
    /// maskelerini, 4×4 seviye önbelleğini ve `tx_lpf_*` bağlamlarını yazar.
    fn create_lf_mask(&mut self, bs: u8, b: &Av1Block, intra: bool) {
        if self.lf.is_none() {
            return;
        }
        let b_dim = t::BLOCK_DIMENSIONS[usize::from(bs)];
        let bw4 = usize::from(b_dim[0]);
        let bh4 = usize::from(b_dim[1]);
        let bx4 = self.bx & 31;
        let by4 = self.by & 31;
        let ss_hor = self.ss_hor;
        let ss_ver = self.ss_ver;
        let has_chroma = self.has_chroma
            && (bw4 > ss_hor || self.bx & 1 != 0)
            && (bh4 > ss_ver || self.by & 1 != 0);
        let seg = usize::from(b.seg_id);
        let t = &self.lflvl[seg];
        let lv = if intra {
            [t[0][0][0], t[1][0][0], t[2][0][0], t[3][0][0]]
        } else {
            if !(0..7).contains(&b.ref0) {
                return;
            }
            let gmv = if b.comp_type != COMP_INTER_NONE {
                b.inter_mode == 6 // GLOBALMV_GLOBALMV
            } else {
                b.inter_mode == GLOBALMV
            };
            let r = usize::from(b.ref0 as u8) + 1;
            // dav1d `decode.c:1893-1900`: `[!is_globalmv]` — GLOBALMV **0**,
            // diğer inter kipler **1**.
            let g = usize::from(!gmv);
            [t[0][r][g], t[1][r][g], t[2][r][g], t[3][r][g]]
        };
        if dsp_trace().lf_blk
            && self.by == 0
            && self.hdr.loopfilter.level_y[0] == 13
            && self.hdr.loopfilter.level_v == 61
        {
            crate::trace_eprintln!(
                "BLK bx={} by={} bs={} bw4={} bh4={} chroma={} intra={} seg={} lv={:?} ref0={} im={} ct={}",
                self.bx,
                self.by,
                bs,
                bw4,
                bh4,
                has_chroma,
                intra,
                b.seg_id,
                lv,
                b.ref0,
                b.inter_mode,
                b.comp_type
            );
        }
        let ai = self.bx >> 5;
        let above_idx = self.above_base + ai;
        let (ytx, uvtx) = self.lf_tx(seg, b);
        let lf = self.lf.as_mut().expect("lf");
        let above = &mut self.blocks.above[above_idx];
        let left = &mut self.blocks.left;
        let ay = &mut above.tx_lpf_y[bx4..];
        let ly = &mut left.tx_lpf_y[by4..];
        if has_chroma {
            let cbx4 = bx4 >> ss_hor;
            let cby4 = by4 >> ss_ver;
            let auv = &mut above.tx_lpf_uv[cbx4..];
            let luv = &mut left.tx_lpf_uv[cby4..];
            if intra {
                lf.create_mask_intra(
                    self.bx,
                    self.by,
                    self.w4,
                    self.h4,
                    bs,
                    usize::from(b.tx),
                    usize::from(b.uvtx),
                    lv,
                    ay,
                    ly,
                    Some((auv, luv)),
                );
            } else {
                lf.create_mask_inter(
                    self.bx,
                    self.by,
                    self.w4,
                    self.h4,
                    bs,
                    b.skip,
                    ytx,
                    &b.tx_split,
                    uvtx,
                    lv,
                    ay,
                    ly,
                    Some((auv, luv)),
                );
            }
        } else if intra {
            lf.create_mask_intra(
                self.bx,
                self.by,
                self.w4,
                self.h4,
                bs,
                usize::from(b.tx),
                usize::from(b.uvtx),
                lv,
                ay,
                ly,
                None,
            );
        } else {
            lf.create_mask_inter(
                self.bx,
                self.by,
                self.w4,
                self.h4,
                bs,
                b.skip,
                ytx,
                &b.tx_split,
                uvtx,
                lv,
                ay,
                ly,
                None,
            );
        }
    }

    /// Bir karonun bütün süper blok satırlarını çözer.
    fn decode_tile(
        &mut self,
        tile_row: usize,
        tile_col: usize,
        data: &[u8],
        cdf: &mut CdfContext,
    ) -> Result<()> {
        let mut msac =
            MsacContext::init_with_isa(data, self.hdr.disable_cdf_update, self.dsp.msac_isa());
        self.last_qidx = i32::from(self.hdr.quant.yac);
        self.last_delta_lf = [0; 4];
        self.dq = self.frame_dq;
        let row_start_sb = usize::from(self.hdr.tiling.row_start_sb[tile_row]);
        let row_end_sb = usize::from(self.hdr.tiling.row_start_sb[tile_row + 1])
            .min((self.bh + self.sb_step - 1) >> self.sb_shift);
        let col_start = usize::from(self.hdr.tiling.col_start_sb[tile_col]) << self.sb_shift;
        let col_end =
            (usize::from(self.hdr.tiling.col_start_sb[tile_col + 1]) << self.sb_shift).min(self.bw);
        self.col_start = col_start;
        self.col_end = col_end;
        self.above_base = if self.blocks.above.len() <= self.sb128w {
            0
        } else {
            tile_row * self.sb128w
        };
        self.row_start = row_start_sb << self.sb_shift;
        // dav1d `decode.c:2469`: `tiling.row_end = imin(row_sb_end << sb_shift,
        // f->bh)`. SB'ye yuvarlanmış uç, kare yüksekliğini aşarsa kırpılır;
        // aksi hâlde `prepare_intra_edges` görünür alan dışındaki dolgu
        // piksellerini "alt-sol komşu" sayar (dav1d saymaz).
        self.row_end = (row_end_sb << self.sb_shift).min(self.bh);
        // Tile başında `lr_ref` varsayılanları (`decode.c:2470-2515`).
        if let Some(lr) = self.lr.as_mut() {
            lr.tile_init(self.row_start, self.col_start);
        }

        let root_bl = if self.sb128 { BL_128X128 } else { BL_64X64 };
        for sby in row_start_sb..row_end_sb {
            self.by = sby << self.sb_shift;
            // dav1d `decode.c:2657`: her SB satırında `reset_context(&t->l,
            // keyframe)`; `keyframe` yalnız KEY karede 1'dir. İnter karelerde
            // 0 verilmezse `comp_type`/`ref`/`mode` **bayat** kalır ve
            // compound/jnt bağlamları komşu tile'dan sızar.
            let keyframe = matches!(self.hdr.frame_type, FrameType::Key | FrameType::Intra);
            reset_context(&mut self.blocks.left, keyframe);
            self.blocks.left_pal = [[[0u16; 8]; 3]; 32];
            self.blocks.left_pal_uv_sz = [0u8; 32];
            let mut bx = col_start;
            while bx < col_end {
                self.bx = bx;
                self.reset_cdef();
                // SB başında LR birimleri (`decode.c:2669-2720`): `decode_sb`
                // öncesi, CDEF indeks sıfırlamasından sonra.
                if let Some(lr) = self.lr.as_mut() {
                    lr.read_sb(
                        &mut msac,
                        cdf,
                        self.bx,
                        self.by,
                        self.sb_step,
                        &mut self.lr_counts,
                    );
                }
                let node = edge_trees()[usize::from(!self.sb128)].root();
                self.decode_sb(&mut msac, cdf, root_bl, node)?;
                // dav1d `decode.c:948-963` + `2657-2678`: SB'nin `cdef_idx`
                // değerleri kare ızgarasına taşınır.
                self.cdef_state
                    .store_sb(self.bx, self.by, self.sb128, &self.cdef_idx);
                bx += self.sb_step;
            }
            if msac.cnt() <= -15 {
                return Err(Error::InvalidData("MSAC veri sonu aşıldı"));
            }
            // Sağ kenar tx bağlamı (`decode.c:2739-2747`).
            if let Some(lf) = self.lf.as_mut() {
                let by = self.by;
                let step = self.sb_step;
                let yoff = by & 16;
                let rel_by = by - lf.y_base;
                let dst = if lf.tile_local {
                    rel_by
                } else {
                    tile_col * lf.align_h + by
                };
                lf.tx_right_y[dst..dst + step]
                    .copy_from_slice(&self.blocks.left.tx_lpf_y[yoff..yoff + step]);
                let uvoff = yoff >> self.ss_ver;
                let uvn = step >> self.ss_ver;
                let uvdst = if lf.tile_local {
                    rel_by >> self.ss_ver
                } else {
                    tile_col * (lf.align_h >> self.ss_ver) + (by >> self.ss_ver)
                };
                lf.tx_right_uv[uvdst..uvdst + uvn]
                    .copy_from_slice(&self.blocks.left.tx_lpf_uv[uvoff..uvoff + uvn]);
            }
        }
        // Tile sonu: bu tile satırının bittiği alt kenardaki `above.tx_lpf_*`
        // bağlamını bir sonraki tile satırının (`tile_row + 1`) deblocking
        // okuması için bu tile'ın sütun aralığında (`[col_start..col_end)`)
        // `lf.row_lpf_{y,uv}` içine kaydet (`lf_apply_tmpl.c:374-`).
        if let Some(lf) = self
            .lf
            .as_mut()
            .filter(|lf| tile_row + 1 < lf.n_rows && col_start < col_end)
        {
            let src_base = self.above_base;
            let dst_base = if lf.tile_local {
                0
            } else {
                tile_row * self.sb128w
            };
            let sb_x0 = col_start >> 5;
            let sb_x1 = col_end.div_ceil(32).min(self.sb128w);
            for sbx in sb_x0..sb_x1 {
                let lx0 = col_start.saturating_sub(sbx * 32).min(32);
                let lx1 = col_end.saturating_sub(sbx * 32).min(32);
                if lx0 < lx1 {
                    lf.row_lpf_y[dst_base + sbx][lx0..lx1]
                        .copy_from_slice(&self.blocks.above[src_base + sbx].tx_lpf_y[lx0..lx1]);
                }
                let uv_step = 32 >> self.ss_hor;
                let uv_x0 = (col_start >> self.ss_hor)
                    .saturating_sub(sbx * uv_step)
                    .min(32);
                let uv_x1 = ((col_end + self.ss_hor) >> self.ss_hor)
                    .saturating_sub(sbx * uv_step)
                    .min(32);
                if uv_x0 < uv_x1 {
                    lf.row_lpf_uv[dst_base + sbx][uv_x0..uv_x1].copy_from_slice(
                        &self.blocks.above[src_base + sbx].tx_lpf_uv[uv_x0..uv_x1],
                    );
                }
            }
        }
        Ok(())
    }

    /// SB başına CDEF indeks bağlamını sıfırlar (`decode.c:2662-2678`).
    ///
    /// sb128'de dizinin dört kadranı da sıfırlanır; sb64'te dav1d işaretçiyi
    /// süper bloğun **kendi yuvasına** kaydırıp `ptr[0] = -1` yapar
    /// (`decode.c:2664-2668`); mutlak yuva [`cdef_idx_slot`]'tur. Okuma
    /// (`decode_b`) ve kare ızgarasına yazma (`CdefState::store_sb`) **aynı**
    /// yuvayı kullanır.
    fn reset_cdef(&mut self) {
        if self.sb128 {
            self.cdef_idx = [-1; 4];
        } else {
            let idx = cdef_idx_slot(self.bx, self.by);
            self.cdef_idx[idx] = -1;
        }
    }

    /// `decode_sb` (`decode.c:2123-2395`).
    fn decode_sb(
        &mut self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        bl: u8,
        node: NodeId,
    ) -> Result<()> {
        #[cfg(feature = "msac-trace")]
        crate::trace::mark(
            "sb",
            self.bx as i32,
            self.by as i32,
            i32::from(bl),
            i32::from(self.sb128),
        );
        if bl > BL_8X8 {
            return Err(Error::InvalidData("partition ağacı 8x8'in altına indi"));
        }
        let hsz = 16usize >> bl;
        let have_h_split = self.bw > self.bx + hsz;
        let have_v_split = self.bh > self.by + hsz;
        if !have_h_split && !have_v_split {
            if bl >= BL_8X8 {
                return Err(Error::InvalidData("8x8 bölünmesi kare dışında"));
            }
            let child = edge_trees()[usize::from(!self.sb128)].split_node(node, 0);
            return self.decode_sb(msac, cdf, bl + 1, child);
        }
        if have_h_split && have_v_split {
            let bx8 = (self.bx & 31) >> 1;
            let by8 = (self.by & 31) >> 1;
            let ai = self.bx >> 5;
            let part_a = self.blocks.above[self.above_base + ai].partition[bx8];
            let part_l = self.blocks.left.partition[by8];
            let ctx = ((part_a >> (4 - bl)) & 1) + (((part_l >> (4 - bl)) & 1) << 1);
            let row = layout::CDF_m
                + layout::MODE_partition
                + (usize::from(bl) * 4 + usize::from(ctx)) * 16;
            let n = usize::from(t::PARTITION_TYPE_COUNT[usize::from(bl)]);
            #[cfg(feature = "msac-trace")]
            crate::trace::mark(
                "partition",
                self.bx as i32,
                self.by as i32,
                i32::from(bl),
                ctx as i32,
            );
            let bp = msac.decode_symbol_adapt(&mut cdf.cells_mut()[row..], n) as u8;
            // 4:2:2'de dikey bölmeler geçersizdir (dav1d `decode.c:2161-2165`):
            // `PARTITION_V`(2), `PARTITION_V4`(9), `PARTITION_T_LEFT_SPLIT`(6),
            // `PARTITION_T_RIGHT_SPLIT`(7). Daha önce yanlışlıkla `T_TOP_SPLIT`
            // (4) kontrol ediliyordu ve `T_LEFT_SPLIT` (6) atlanıyordu.
            if self.chroma == ChromaFormat::C422 && matches!(bp, 2 | 6 | 7 | 9) {
                return Err(Error::InvalidData("4:2:2 dikey partition"));
            }
            let bs = t::BLOCK_SIZES[usize::from(bl)][usize::from(bp)];
            match bp {
                0 => {
                    self.decode_b(
                        msac,
                        cdf,
                        bl,
                        bs[0],
                        bp,
                        edge_trees()[usize::from(!self.sb128)].o(node),
                    )?;
                }
                1 => {
                    let h = edge_trees()[usize::from(!self.sb128)].h(node, 0);
                    self.decode_b(msac, cdf, bl, bs[0], bp, h)?;
                    self.by += hsz;
                    let h = edge_trees()[usize::from(!self.sb128)].h(node, 1);
                    self.decode_b(msac, cdf, bl, bs[0], bp, h)?;
                    self.by -= hsz;
                }
                2 => {
                    let v = edge_trees()[usize::from(!self.sb128)].v(node, 0);
                    self.decode_b(msac, cdf, bl, bs[0], bp, v)?;
                    self.bx += hsz;
                    let v = edge_trees()[usize::from(!self.sb128)].v(node, 1);
                    self.decode_b(msac, cdf, bl, bs[0], bp, v)?;
                    self.bx -= hsz;
                }
                3 => {
                    if bl == BL_8X8 {
                        let tree = &edge_trees()[usize::from(!self.sb128)];
                        self.decode_b(msac, cdf, bl, BS_4X4, bp, EDGE_ALL_TR_AND_BL)?;
                        // dav1d `src/decode.c:2197-2206`: 8×8 bölmenin dört
                        // 4×4 bloğunda sol-üst 2B filtresi **ilk bloktan
                        // sonra saklanır** ve **dördüncü bloktan önce geri
                        // yüklenir**. Dördüncü blok 4:2:0/4:2:2'de kromayı
                        // taşır; kromanın sol-üst parçası ilk bloğun mv'si ve
                        // filtresiyle üretilir (`src/recon_tmpl.c:1658-1670`).
                        let tl_filter = self.tl_4x4_filter;
                        self.bx += 1;
                        let s = tree.split_tip(node, 0);
                        self.decode_b(msac, cdf, bl, BS_4X4, bp, s)?;
                        self.bx -= 1;
                        self.by += 1;
                        let s = tree.split_tip(node, 1);
                        self.decode_b(msac, cdf, bl, BS_4X4, bp, s)?;
                        self.bx += 1;
                        self.tl_4x4_filter = tl_filter;
                        let s = tree.split_tip(node, 2);
                        self.decode_b(msac, cdf, bl, BS_4X4, bp, s)?;
                        self.bx -= 1;
                        self.by -= 1;
                    } else {
                        let tree = &edge_trees()[usize::from(!self.sb128)];
                        let c0 = tree.split_node(node, 0);
                        let c1 = tree.split_node(node, 1);
                        let c2 = tree.split_node(node, 2);
                        let c3 = tree.split_node(node, 3);
                        self.decode_sb(msac, cdf, bl + 1, c0)?;
                        self.bx += hsz;
                        self.decode_sb(msac, cdf, bl + 1, c1)?;
                        self.bx -= hsz;
                        self.by += hsz;
                        self.decode_sb(msac, cdf, bl + 1, c2)?;
                        self.bx += hsz;
                        self.decode_sb(msac, cdf, bl + 1, c3)?;
                        self.bx -= hsz;
                        self.by -= hsz;
                    }
                }
                4 => {
                    let tree = &edge_trees()[usize::from(!self.sb128)];
                    let (h0, v1) = (tree.h(node, 0), tree.v(node, 1));
                    self.decode_b(msac, cdf, bl, bs[0], bp, EDGE_ALL_TR_AND_BL)?;
                    self.bx += hsz;
                    self.decode_b(msac, cdf, bl, bs[0], bp, v1)?;
                    self.bx -= hsz;
                    self.by += hsz;
                    let h1 = tree.h(node, 1);
                    self.decode_b(msac, cdf, bl, bs[1], bp, h1)?;
                    self.by -= hsz;
                    let _ = h0;
                }
                5 => {
                    let tree = &edge_trees()[usize::from(!self.sb128)];
                    let (h0, v0) = (tree.h(node, 0), tree.v(node, 0));
                    self.decode_b(msac, cdf, bl, bs[0], bp, h0)?;
                    self.by += hsz;
                    self.decode_b(msac, cdf, bl, bs[1], bp, v0)?;
                    self.bx += hsz;
                    self.decode_b(msac, cdf, bl, bs[1], bp, 0)?;
                    self.bx -= hsz;
                    self.by -= hsz;
                }
                6 => {
                    let tree = &edge_trees()[usize::from(!self.sb128)];
                    let h1 = tree.h(node, 1);
                    let v1 = tree.v(node, 1);
                    self.decode_b(msac, cdf, bl, bs[0], bp, EDGE_ALL_TR_AND_BL)?;
                    self.by += hsz;
                    self.decode_b(msac, cdf, bl, bs[0], bp, h1)?;
                    self.by -= hsz;
                    self.bx += hsz;
                    self.decode_b(msac, cdf, bl, bs[1], bp, v1)?;
                    self.bx -= hsz;
                }
                7 => {
                    let tree = &edge_trees()[usize::from(!self.sb128)];
                    let v0 = tree.v(node, 0);
                    let h0 = tree.h(node, 0);
                    self.decode_b(msac, cdf, bl, bs[0], bp, v0)?;
                    self.bx += hsz;
                    self.decode_b(msac, cdf, bl, bs[1], bp, h0)?;
                    self.by += hsz;
                    self.decode_b(msac, cdf, bl, bs[1], bp, 0)?;
                    self.by -= hsz;
                    self.bx -= hsz;
                }
                8 => {
                    let tree = &edge_trees()[usize::from(!self.sb128)];
                    let (h0, h1) = (tree.h(node, 0), tree.h(node, 1));
                    let h4 = tree.h4(node);
                    self.decode_b(msac, cdf, bl, bs[0], bp, h0)?;
                    self.by += hsz >> 1;
                    self.decode_b(msac, cdf, bl, bs[0], bp, h4)?;
                    self.by += hsz >> 1;
                    self.decode_b(msac, cdf, bl, bs[0], bp, EDGE_ALL_LEFT_HAS_BOTTOM)?;
                    self.by += hsz >> 1;
                    if self.by < self.bh {
                        self.decode_b(msac, cdf, bl, bs[0], bp, h1)?;
                    }
                    self.by -= (hsz * 3) >> 1;
                }
                _ => {
                    let tree = &edge_trees()[usize::from(!self.sb128)];
                    let (v0, v1) = (tree.v(node, 0), tree.v(node, 1));
                    let v4 = tree.v4(node);
                    self.decode_b(msac, cdf, bl, bs[0], bp, v0)?;
                    self.bx += hsz >> 1;
                    self.decode_b(msac, cdf, bl, bs[0], bp, v4)?;
                    self.bx += hsz >> 1;
                    self.decode_b(msac, cdf, bl, bs[0], bp, EDGE_ALL_TOP_HAS_RIGHT)?;
                    self.bx += hsz >> 1;
                    if self.bx < self.bw {
                        self.decode_b(msac, cdf, bl, bs[0], bp, v1)?;
                    }
                    self.bx -= (hsz * 3) >> 1;
                }
            }
            // partition bağlamı (split hariç; 8x8'de split de yazılır)
            if bp != 3 || bl == BL_8X8 {
                let al = t::AL_PART_CTX[0][usize::from(bl)][usize::from(bp)];
                let al_left = t::AL_PART_CTX[1][usize::from(bl)][usize::from(bp)];
                let bx8 = (self.bx & 31) >> 1;
                let by8 = (self.by & 31) >> 1;
                let ai = self.bx >> 5;
                fill_small(
                    &mut self.blocks.above[self.above_base + ai].partition,
                    bx8,
                    hsz,
                    al,
                );
                fill_small(&mut self.blocks.left.partition, by8, hsz, al_left);
            }
            return Ok(());
        }
        if have_h_split {
            let bx8 = (self.bx & 31) >> 1;
            let by8 = (self.by & 31) >> 1;
            let ai = self.bx >> 5;
            let part_a = self.blocks.above[self.above_base + ai].partition[bx8];
            let part_l = self.blocks.left.partition[by8];
            let ctx = ((part_a >> (4 - bl)) & 1) + (((part_l >> (4 - bl)) & 1) << 1);
            let row = layout::CDF_m
                + layout::MODE_partition
                + (usize::from(bl) * 4 + usize::from(ctx)) * 16;
            let f = gather_top_partition_prob(&cdf.cells()[row..], bl);
            #[cfg(feature = "msac-trace")]
            crate::trace::mark(
                "partition_split_h",
                self.bx as i32,
                self.by as i32,
                i32::from(bl),
                ctx as i32,
            );
            let is_split = msac.decode_bool(f) != 0;
            if is_split {
                let tree = &edge_trees()[usize::from(!self.sb128)];
                let c0 = tree.split_node(node, 0);
                let c1 = tree.split_node(node, 1);
                self.decode_sb(msac, cdf, bl + 1, c0)?;
                self.bx += hsz;
                self.decode_sb(msac, cdf, bl + 1, c1)?;
                self.bx -= hsz;
            } else {
                let bs = t::BLOCK_SIZES[usize::from(bl)][1][0];
                let h0 = edge_trees()[usize::from(!self.sb128)].h(node, 0);
                self.decode_b(msac, cdf, bl, bs, 1, h0)?;
            }
            let bp = if is_split { 3u8 } else { 1 };
            if bp != 3 || bl == BL_8X8 {
                let al = t::AL_PART_CTX[0][usize::from(bl)][usize::from(bp)];
                let al_left = t::AL_PART_CTX[1][usize::from(bl)][usize::from(bp)];
                let bx8 = (self.bx & 31) >> 1;
                let by8 = (self.by & 31) >> 1;
                let ai = self.bx >> 5;
                fill_small(
                    &mut self.blocks.above[self.above_base + ai].partition,
                    bx8,
                    hsz,
                    al,
                );
                fill_small(&mut self.blocks.left.partition, by8, hsz, al_left);
            }
            return Ok(());
        }
        // Yalnız dikey bölme var.
        let bx8 = (self.bx & 31) >> 1;
        let by8 = (self.by & 31) >> 1;
        let ai = self.bx >> 5;
        let part_a = self.blocks.above[self.above_base + ai].partition[bx8];
        let part_l = self.blocks.left.partition[by8];
        let ctx = ((part_a >> (4 - bl)) & 1) + (((part_l >> (4 - bl)) & 1) << 1);
        let row =
            layout::CDF_m + layout::MODE_partition + (usize::from(bl) * 4 + usize::from(ctx)) * 16;
        let f = gather_left_partition_prob(&cdf.cells()[row..], bl);
        #[cfg(feature = "msac-trace")]
        crate::trace::mark(
            "partition_split_v",
            self.bx as i32,
            self.by as i32,
            i32::from(bl),
            ctx as i32,
        );
        let is_split = msac.decode_bool(f) != 0;
        if is_split {
            let tree = &edge_trees()[usize::from(!self.sb128)];
            let c0 = tree.split_node(node, 0);
            let c2 = tree.split_node(node, 2);
            self.decode_sb(msac, cdf, bl + 1, c0)?;
            self.by += hsz;
            self.decode_sb(msac, cdf, bl + 1, c2)?;
            self.by -= hsz;
        } else {
            if self.chroma == ChromaFormat::C422 {
                return Err(Error::InvalidData("4:2:2 dikey partition"));
            }
            let bs = t::BLOCK_SIZES[usize::from(bl)][2][0];
            let v0 = edge_trees()[usize::from(!self.sb128)].v(node, 0);
            self.decode_b(msac, cdf, bl, bs, 2, v0)?;
        }
        let bp = if is_split { 3u8 } else { 2 };
        if bp != 3 || bl == BL_8X8 {
            let al = t::AL_PART_CTX[0][usize::from(bl)][usize::from(bp)];
            let al_left = t::AL_PART_CTX[1][usize::from(bl)][usize::from(bp)];
            let bx8 = (self.bx & 31) >> 1;
            let by8 = (self.by & 31) >> 1;
            let ai = self.bx >> 5;
            fill_small(
                &mut self.blocks.above[self.above_base + ai].partition,
                bx8,
                hsz,
                al,
            );
            fill_small(&mut self.blocks.left.partition, by8, hsz, al_left);
        }
        Ok(())
    }
}

/// `gather_top_partition_prob` (`env.h:120-132`).
fn gather_top_partition_prob(cdf: &[u16], bl: u8) -> u32 {
    let mut out = u32::from(cdf[1]).wrapping_sub(u32::from(cdf[4]));
    out = out.wrapping_add(u32::from(cdf[5]));
    if bl != BL_128X128 {
        out = out.wrapping_add(u32::from(cdf[8]).wrapping_sub(u32::from(cdf[7])));
    }
    out
}

/// `gather_left_partition_prob` (`env.h:104-117`).
fn gather_left_partition_prob(cdf: &[u16], bl: u8) -> u32 {
    let mut out = u32::from(cdf[0]).wrapping_sub(u32::from(cdf[1]));
    out = out.wrapping_add(u32::from(cdf[2]).wrapping_sub(u32::from(cdf[6])));
    if bl != BL_128X128 {
        out = out.wrapping_add(u32::from(cdf[7]).wrapping_sub(u32::from(cdf[8])));
    }
    out
}

impl<'a> Dec<'a> {
    /// `decode_b` intra yolu (`decode.c:683-1260`, `else` inter dalları hariç).
    #[allow(clippy::too_many_lines)]
    fn decode_b(
        &mut self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        _bl: u8,
        bs: u8,
        _bp: u8,
        edge_flags: u8,
    ) -> Result<()> {
        #[cfg(feature = "msac-trace")]
        crate::trace::mark(
            "blok",
            self.bx as i32,
            self.by as i32,
            i32::from(bs),
            i32::from(_bl),
        );
        let blk_bx = self.bx;
        let blk_by = self.by;
        let b_dim = &t::BLOCK_DIMENSIONS[usize::from(bs)];
        let bw4 = usize::from(b_dim[0]);
        let bh4 = usize::from(b_dim[1]);
        let bx4 = self.bx & 31;
        let by4 = self.by & 31;
        let ai = self.bx >> 5;
        let w4 = bw4.min(self.bw - self.bx);
        let h4 = bh4.min(self.bh - self.by);
        let ss_hor = self.ss_hor;
        let ss_ver = self.ss_ver;
        let cbx4 = bx4 >> ss_hor;
        let cby4 = by4 >> ss_ver;
        let cbw4 = (bw4 + ss_hor) >> ss_hor;
        let cbh4 = (bh4 + ss_ver) >> ss_ver;
        let has_chroma = self.has_chroma
            && (bw4 > ss_hor || self.bx & 1 != 0)
            && (bh4 > ss_ver || self.by & 1 != 0);
        let have_left = self.bx > self.col_start;
        let have_top = self.by > self.row_start;

        let is_inter_frame = matches!(self.hdr.frame_type, FrameType::Inter | FrameType::Switch);
        let mut b = Av1Block::default();
        let mut seg: Option<SegmentationData> = None;
        let mut seg_pred = 0u8;

        // --- segment_id (preskip) -----------------------------------------
        if self.hdr.segmentation.enabled {
            if !self.hdr.segmentation.update_map {
                b.seg_id = if self.prev_segmap.is_some() {
                    self.prev_frame_segid(self.by, self.bx, w4, h4)
                } else {
                    0
                };
                seg = Some(self.hdr.segmentation.seg_data.d[usize::from(b.seg_id)]);
            } else if self.hdr.segmentation.seg_data.preskip {
                if self.hdr.segmentation.temporal
                    && self.pal_bool(
                        msac,
                        cdf,
                        layout::MODE_seg_pred,
                        self.seg_pred_ctx(bx4, by4),
                    )
                {
                    seg_pred = 1;
                    b.seg_id = if self.prev_segmap.is_some() {
                        self.prev_frame_segid(self.by, self.bx, w4, h4)
                    } else {
                        0
                    };
                } else {
                    let (pred, ctx) = get_cur_frame_segid(
                        &self.segmap,
                        self.segmap_stride,
                        self.by - self.segmap_y_base,
                        self.bx - self.segmap_x_base,
                        have_top,
                        have_left,
                    );
                    let diff = msac.decode_symbol_adapt(
                        &mut cdf.cells_mut()
                            [layout::CDF_m + layout::MODE_seg_id + usize::from(ctx) * 8..],
                        7,
                    ) as i32;
                    let last = i32::from(self.hdr.segmentation.seg_data.last_active_segid);
                    let mut id = neg_deinterleave(diff, pred, last + 1);
                    if id > last {
                        id = 0;
                    }
                    if !(0..8).contains(&id) {
                        id = 0;
                    }
                    b.seg_id = id as u8;
                }
                // dav1d `decode.c:869-871`: preskip yolunda da `seg` işaretçisi
                // seçilen segmente bağlanır. Bu olmadan aşağıdaki `seg->skip`
                // denetimi kaçar ve segment "skip" özelliği taşıdığında
                // dav1d'nin **okumadığı** skip sembolü okunur (sözdizimi kayar).
                seg = Some(self.hdr.segmentation.seg_data.d[usize::from(b.seg_id)]);
            }
        } else {
            b.seg_id = 0;
        }

        // --- skip_mode -----------------------------------------------------
        let seg_allows = seg.is_none_or(|s| !(s.globalmv || s.reference != -1 || s.skip));
        let mut skip_mode = false;
        if seg_allows && self.hdr.skip_mode_enabled && bw4.min(bh4) > 1 {
            let ctx = usize::from(self.blocks.above[self.above_base + ai].skip_mode[bx4])
                + usize::from(self.blocks.left.skip_mode[by4]);
            let row = layout::CDF_m + layout::MODE_skip_mode + ctx * 2;
            skip_mode = msac.decode_bool_adapt(&mut cdf.cells_mut()[row..]) != 0;
        }
        b.skip_mode = skip_mode;

        // --- skip ----------------------------------------------------------
        // dav1d `decode.c:907-916`: `skip_mode` ise `skip` **okunmaz**, 1'dir.
        if b.skip_mode || seg.is_some_and(|s| s.skip) {
            b.skip = true;
        } else {
            let sctx = usize::from(self.blocks.above[self.above_base + ai].skip[bx4])
                + usize::from(self.blocks.left.skip[by4]);
            let row = layout::CDF_m + layout::MODE_skip + sctx * 2;
            b.skip = msac.decode_bool_adapt(&mut cdf.cells_mut()[row..]) != 0;
        }

        // --- segment_id (postskip) -----------------------------------------
        if self.hdr.segmentation.enabled
            && self.hdr.segmentation.update_map
            && !self.hdr.segmentation.seg_data.preskip
        {
            let mut temporal_pred = false;
            if !b.skip && self.hdr.segmentation.temporal {
                let ctx = usize::from(self.blocks.above[self.above_base + ai].seg_pred[bx4])
                    + usize::from(self.blocks.left.seg_pred[by4]);
                let row = layout::CDF_m + layout::MODE_seg_pred + ctx * 2;
                temporal_pred = msac.decode_bool_adapt(&mut cdf.cells_mut()[row..]) != 0;
            }
            if temporal_pred {
                seg_pred = 1;
                b.seg_id = if self.prev_segmap.is_some() {
                    self.prev_frame_segid(self.by, self.bx, w4, h4)
                } else {
                    0
                };
            } else {
                let (pred, ctx) = get_cur_frame_segid(
                    &self.segmap,
                    self.segmap_stride,
                    self.by - self.segmap_y_base,
                    self.bx - self.segmap_x_base,
                    have_top,
                    have_left,
                );
                if b.skip {
                    b.seg_id = pred as u8;
                } else {
                    let diff = msac.decode_symbol_adapt(
                        &mut cdf.cells_mut()
                            [layout::CDF_m + layout::MODE_seg_id + usize::from(ctx) * 8..],
                        7,
                    ) as i32;
                    let last = i32::from(self.hdr.segmentation.seg_data.last_active_segid);
                    let mut id = neg_deinterleave(diff, pred, last + 1);
                    if id > last {
                        id = 0;
                    }
                    if !(0..8).contains(&id) {
                        id = 0;
                    }
                    b.seg_id = id as u8;
                }
            }
            if b.seg_id >= 8 {
                b.seg_id = 0;
            }
        }

        // --- cdef index ----------------------------------------------------
        if !b.skip {
            // dav1d `decode.c:948-963`: sb128'de `idx = ((bx & 16) >> 4) +
            // ((by & 16) >> 3)`, sb64'te `cur_sb_cdef_idx_ptr` bu yuvaya
            // kaydırılmıştır ve `ptr[0]` okunur — mutlak yuva ikisinde de
            // aynıdır ([`cdef_idx_slot`]). sb64'te `idx = 0` kullanmak
            // ikinci ve alt sıradaki 64×64'lerde önceki süper bloğun değerini
            // okumak (bitleri hiç okumamak) demekti.
            let idx = cdef_idx_slot(self.bx, self.by);
            if self.cdef_idx[idx] == -1 {
                let v = msac.decode_bools(u32::from(self.hdr.cdef.n_bits)) as i32;
                self.cdef_idx[idx] = v;
                if bw4 > 16 {
                    self.cdef_idx[idx + 1] = v;
                }
                if bh4 > 16 {
                    self.cdef_idx[idx + 2] = v;
                }
                if bw4 == 32 && bh4 == 32 {
                    self.cdef_idx[idx + 3] = v;
                }
            }
            // dav1d `decode.c:1952-1961`: `noskip_mask` — `!b->skip` olan
            // bloğun kapsadığı bütün 8x8'ler işaretlenir. Konum **kare
            // mutlak**tır (`bx4/by4` SB-göreli maske için 31'e maskelenir).
            self.cdef_state.mark_noskip(self.bx, self.by, w4, h4);
        }

        // --- delta q / delta lf -------------------------------------------
        let sb_mask = 31u32 >> u32::from(!self.sb128);
        if (self.bx as u32 | self.by as u32) & sb_mask == 0 {
            let prev_qidx = self.last_qidx;
            let sb_bs = if self.sb128 { 0 } else { 3 };
            let have_delta_q = self.hdr.delta.q_present && (bs != sb_bs || !b.skip);
            let prev_delta_lf = self.last_delta_lf;
            if have_delta_q {
                let mut delta_q = msac.decode_symbol_adapt(
                    &mut cdf.cells_mut()[layout::CDF_m + layout::MODE_delta_q..],
                    3,
                ) as i32;
                if delta_q == 3 {
                    let n_bits = 1 + msac.decode_bools(3) as i32;
                    delta_q = msac.decode_bools(n_bits as u32) as i32 + 1 + (1 << n_bits);
                }
                if delta_q != 0 {
                    if msac.decode_bool_equi() != 0 {
                        delta_q = -delta_q;
                    }
                    delta_q *= 1 << self.hdr.delta.q_res_log2;
                }
                self.last_qidx = (self.last_qidx + delta_q).clamp(1, 255);
                if self.hdr.delta.lf_present {
                    let n_lfs = if self.hdr.delta.lf_multi {
                        if self.has_chroma { 4 } else { 2 }
                    } else {
                        1
                    };
                    for i in 0..n_lfs {
                        let row = layout::CDF_m
                            + layout::MODE_delta_lf
                            + (i + usize::from(self.hdr.delta.lf_multi)) * 4;
                        let mut delta_lf =
                            msac.decode_symbol_adapt(&mut cdf.cells_mut()[row..], 3) as i32;
                        if delta_lf == 3 {
                            let n_bits = 1 + msac.decode_bools(3) as i32;
                            delta_lf = msac.decode_bools(n_bits as u32) as i32 + 1 + (1 << n_bits);
                        }
                        if delta_lf != 0 {
                            if msac.decode_bool_equi() != 0 {
                                delta_lf = -delta_lf;
                            }
                            delta_lf *= 1 << self.hdr.delta.lf_res_log2;
                        }
                        self.last_delta_lf[i] =
                            (i32::from(self.last_delta_lf[i]) + delta_lf).clamp(-63, 63) as i8;
                    }
                }
            }
            if self.last_qidx == i32::from(self.hdr.quant.yac) {
                self.dq = self.frame_dq;
            } else if self.last_qidx != prev_qidx {
                init_quant_tables(
                    self.hdr,
                    self.last_qidx,
                    &mut self.dq,
                    match self.bd {
                        BitDepth::Eight => 0,
                        BitDepth::Ten => 1,
                        BitDepth::Twelve => 2,
                    },
                );
            }
            let _ = prev_delta_lf;
            if self.last_qidx != prev_qidx {
                self.tools.delta_q += 1;
            }
            if self.last_delta_lf != prev_delta_lf {
                self.tools.delta_lf += 1;
            }
            // `decode.c:1021-1030`: delta LF sıfırsa kare tablosu, değilse
            // süper blok tablosu.
            if self.last_delta_lf == [0; 4] {
                self.lflvl = self.frame_lflvl;
            } else if self.last_delta_lf != prev_delta_lf {
                self.lflvl =
                    crate::decode::loopfilter::calc_lf_values(self.hdr, self.last_delta_lf);
                if dsp_trace().lf_dlf {
                    crate::trace_eprintln!(
                        "DLF frame={} bx={} by={} d={:?} v={:?}",
                        self.frame_no,
                        self.bx,
                        self.by,
                        self.last_delta_lf,
                        [
                            self.lflvl[0][0][0][0],
                            self.lflvl[0][1][0][0],
                            self.lflvl[0][2][0][0],
                            self.lflvl[0][3][0][0]
                        ]
                    );
                }
            }
        }

        // --- intra / intrabc / inter bayrağı ------------------------------
        // dav1d `decode.c:1028-1052`: inter/switch karelerde segment referans
        // özelliği `b->intra`'yı zorlar; aksi hâlde `m.intra` CDF'si okunur.
        // Anahtar/intra karelerde `allow_intrabc` ise `m.intrabc` okunur.
        if is_inter_frame {
            // dav1d `decode.c:1028-1030`: `skip_mode` ise intra bayrağı
            // **okunmaz**, `b->intra = 0`.
            if b.skip_mode {
                b.intra = false;
            } else if seg.is_some_and(|s| s.reference >= 0 || s.globalmv) {
                b.intra = seg.is_some_and(|s| s.reference == 0);
            } else {
                let ictx = self.get_intra_ctx(bx4, by4, have_top, have_left);
                let row = layout::CDF_m + layout::MODE_intra + ictx * 2;
                b.intra = msac.decode_bool_adapt(&mut cdf.cells_mut()[row..]) == 0;
            }
            if b.intra {
                b.ref0 = -1;
            }
        } else if self.hdr.allow_intrabc {
            let row = layout::CDF_m + layout::MODE_intrabc;
            b.intra = msac.decode_bool_adapt(&mut cdf.cells_mut()[row..]) == 0;
        } else {
            b.intra = true;
        }
        if b.intra {
            // --- y mode --------------------------------------------------------
            let ymode_off = if is_inter_frame {
                // dav1d `decode.c:1055-1058`: inter karelerde `m.y_mode[sz]`.
                layout::CDF_m
                    + layout::MODE_y_mode
                    + usize::from(t::YMODE_SIZE_CONTEXT[usize::from(bs)]) * 16
            } else {
                let ymode_a = t::INTRA_MODE_CONTEXT
                    [usize::from(self.blocks.above[self.above_base + ai].mode[bx4])];
                let ymode_l = t::INTRA_MODE_CONTEXT[usize::from(self.blocks.left.mode[by4])];
                layout::CDF_kfym + (usize::from(ymode_a) * 5 + usize::from(ymode_l)) * 16
            };
            b.y_mode = msac.decode_symbol_adapt(&mut cdf.cells_mut()[ymode_off..], 12) as u8;

            if b_dim[2] + b_dim[3] >= 2 && (VERT_PRED..=VERT_LEFT_PRED).contains(&b.y_mode) {
                let row = layout::CDF_m
                    + layout::MODE_angle_delta
                    + usize::from(b.y_mode - VERT_PRED) * 8;
                let angle = msac.decode_symbol_adapt(&mut cdf.cells_mut()[row..], 6) as i32;
                b.y_angle = angle - 3;
            }

            if has_chroma {
                let lossless_seg = self.hdr.segmentation.lossless[usize::from(b.seg_id)];
                let cfl_allowed = if lossless_seg {
                    cbw4 == 1 && cbh4 == 1
                } else {
                    CFL_ALLOWED_MASK & (1u32 << bs) != 0
                };
                let row = layout::CDF_m
                    + layout::MODE_uv_mode
                    + (usize::from(cfl_allowed) * 13 + usize::from(b.y_mode)) * 16;
                let n = 13 - usize::from(!cfl_allowed);
                b.uv_mode = msac.decode_symbol_adapt(&mut cdf.cells_mut()[row..], n) as u8;
                b.uv_angle = 0;
                if b.uv_mode == CFL_PRED {
                    let sign = msac.decode_symbol_adapt(
                        &mut cdf.cells_mut()[layout::CDF_m + layout::MODE_cfl_sign..],
                        7,
                    ) as i32
                        + 1;
                    let sign_u = (sign * 0x56) >> 8;
                    let sign_v = sign - sign_u * 3;
                    if sign_u != 0 {
                        let ctx = usize::from(sign_u == 2) * 3 + sign_v as usize;
                        let idx = msac.decode_symbol_adapt(
                            &mut cdf.cells_mut()
                                [layout::CDF_m + layout::MODE_cfl_alpha + ctx * 16..],
                            15,
                        ) as i32
                            + 1;
                        b.cfl_alpha[0] = if sign_u == 1 { -idx } else { idx };
                    }
                    if sign_v != 0 {
                        let ctx = usize::from(sign_v == 2) * 3 + sign_u as usize;
                        let idx = msac.decode_symbol_adapt(
                            &mut cdf.cells_mut()
                                [layout::CDF_m + layout::MODE_cfl_alpha + ctx * 16..],
                            15,
                        ) as i32
                            + 1;
                        b.cfl_alpha[1] = if sign_v == 1 { -idx } else { idx };
                    }
                } else if b_dim[2] + b_dim[3] >= 2
                    && (VERT_PRED..=VERT_LEFT_PRED).contains(&b.uv_mode)
                {
                    let row = layout::CDF_m
                        + layout::MODE_angle_delta
                        + usize::from(b.uv_mode - VERT_PRED) * 8;
                    let angle = msac.decode_symbol_adapt(&mut cdf.cells_mut()[row..], 6) as i32;
                    b.uv_angle = angle - 3;
                }
            }

            // --- palette / filter intra ---------------------------------------
            b.pal_sz = [0, 0];
            if self.hdr.allow_screen_content_tools && bw4.max(bh4) <= 16 && bw4 + bh4 >= 4 {
                let sz_ctx = usize::from(b_dim[2] + b_dim[3] - 2);
                if b.y_mode == DC_PRED {
                    let pal_ctx =
                        usize::from(self.blocks.above[self.above_base + ai].pal_sz[bx4] > 0)
                            + usize::from(self.blocks.left.pal_sz[by4] > 0);
                    let row = layout::CDF_m + layout::MODE_pal_y + (sz_ctx * 3 + pal_ctx) * 2;
                    if msac.decode_bool_adapt(&mut cdf.cells_mut()[row..]) != 0 {
                        self.read_pal_plane(msac, cdf, 0, sz_ctx, bx4, by4, &mut b)?;
                    }
                }
                if has_chroma && b.uv_mode == DC_PRED {
                    let pal_ctx = usize::from(b.pal_sz[0] > 0);
                    let row = layout::CDF_m + layout::MODE_pal_uv + pal_ctx * 2;
                    if msac.decode_bool_adapt(&mut cdf.cells_mut()[row..]) != 0 {
                        self.read_pal_uv(msac, cdf, sz_ctx, bx4, by4, &mut b)?;
                    }
                }
            }

            if b.y_mode == DC_PRED
                && b.pal_sz[0] == 0
                && b_dim[2].max(b_dim[3]) <= 3
                && self.seq.filter_intra
            {
                let row = layout::CDF_m + layout::MODE_use_filter_intra + usize::from(bs) * 2;
                if msac.decode_bool_adapt(&mut cdf.cells_mut()[row..]) != 0 {
                    b.y_mode = FILTER_PRED;
                    b.y_angle = msac.decode_symbol_adapt(
                        &mut cdf.cells_mut()[layout::CDF_m + layout::MODE_filter_intra..],
                        4,
                    ) as i32;
                }
            }

            if dsp_trace().blk_trace {
                crate::trace_eprintln!(
                    "B bx={} by={} bs={} skip={} seg={} y={} uv={} tx={} cnt={} intra={} mv={} maxytx={} split={:x}/{:x}",
                    self.bx,
                    self.by,
                    bs,
                    b.skip,
                    b.seg_id,
                    b.y_mode,
                    b.uv_mode,
                    b.tx,
                    msac.cnt(),
                    u8::from(b.intra),
                    ((b.mv[0].y as i32) << 16) | (b.mv[0].x as u16 as i32),
                    b.max_ytx,
                    b.tx_split[0],
                    b.tx_split[1]
                );
            }
            if b.pal_sz[0] != 0 {
                self.read_pal_indices(msac, cdf, b.pal_sz[0] as usize, 0, w4, h4, bw4, bh4)?;
            }
            if has_chroma && b.pal_sz[1] != 0 {
                let cw4 = (w4 + ss_hor) >> ss_hor;
                let ch4 = (h4 + ss_ver) >> ss_ver;
                self.read_pal_indices(msac, cdf, b.pal_sz[1] as usize, 1, cw4, ch4, cbw4, cbh4)?;
            }

            // --- tx size -------------------------------------------------------
            if self.hdr.segmentation.lossless[usize::from(b.seg_id)] {
                b.tx = TX_4X4;
                b.uvtx = TX_4X4;
            } else {
                b.tx = t::MAX_TXFM_SIZE_FOR_BS[usize::from(bs)][0];
                b.uvtx = t::MAX_TXFM_SIZE_FOR_BS[usize::from(bs)][match self.chroma {
                    ChromaFormat::C420 => 1,
                    ChromaFormat::C422 => 2,
                    _ => 3,
                }];
                let mut t_dim = t::TXFM_DIMENSIONS[usize::from(b.tx)];
                if self.hdr.txfm_mode == TxfmMode::Switchable && t_dim[5] > 0 {
                    let tctx = usize::from(self.blocks.left.tx_intra[by4] >= t_dim[3] as i8)
                        + usize::from(
                            self.blocks.above[self.above_base + ai].tx_intra[bx4] >= t_dim[2] as i8,
                        );
                    let row = layout::CDF_m
                        + layout::MODE_txsz
                        + (usize::from(t_dim[5] - 1) * 3 + tctx) * 4;
                    let mut depth = msac.decode_symbol_adapt(
                        &mut cdf.cells_mut()[row..],
                        usize::from(t_dim[5].min(2)),
                    );
                    while depth > 0 {
                        depth -= 1;
                        b.tx = t_dim[6];
                        t_dim = t::TXFM_DIMENSIONS[usize::from(b.tx)];
                    }
                }
            }

            // --- yeniden yapılandırma -----------------------------------------
            self.recon_b_intra(msac, cdf, bs, edge_flags, &b)?;
            // dav1d `decode.c:1268-1270`: inter/switch karelerde ve intrabc'e
            // izin verilen karelerde intra bloklar da refmvs ızgarasına
            // `splat_intraref` ile yazılır.
            if self.hdr.allow_intrabc || is_inter_frame {
                self.refmvs.splat_intra(self.bx, self.by, bw4, bh4, bs);
            }
            self.create_lf_mask(bs, &b, true);
        } else if is_inter_frame {
            // --- tek referanslı inter blok (dav1d `decode.c:1390-1880`) ----
            self.decode_inter_block(msac, cdf, bs, edge_flags, &mut b, seg, seg_pred)?;
        } else {
            // --- intrabc (dav1d `decode.c:1271-1390`) ----------------------
            self.decode_intrabc(msac, cdf, bs, edge_flags, &mut b)?;
        }

        // --- araç kapsamı sayaçları (tanılama; davranışı değiştirmez) -------
        self.tools.blocks += 1;
        if !b.intra && !is_inter_frame {
            self.tools.intrabc += 1;
        }
        if b.pal_sz[0] != 0 {
            self.tools.palette_y += 1;
        }
        if b.pal_sz[1] != 0 {
            self.tools.palette_uv += 1;
        }
        if has_chroma && b.uv_mode == CFL_PRED {
            self.tools.cfl += 1;
        }
        if b.y_mode == FILTER_PRED {
            self.tools.filter_intra += 1;
        }
        if (VERT_PRED..=VERT_LEFT_PRED).contains(&b.y_mode) {
            self.tools.directional += 1;
            if b.y_angle != 0 {
                self.tools.angle_delta += 1;
            }
        }
        if self.hdr.segmentation.lossless[usize::from(b.seg_id)] {
            self.tools.lossless_wht += 1;
        }
        if self.hdr.quant.qm {
            self.tools.qmatrix += 1;
        }
        if b.seg_id != 0 {
            self.tools.seg_block += 1;
        }
        if b.intra && b.tx != t::MAX_TXFM_SIZE_FOR_BS[usize::from(bs)][0] {
            self.tools.tx_split += 1;
        }

        // --- bağlam güncellemesi ------------------------------------------
        let t_dim = t::TXFM_DIMENSIONS[usize::from(b.tx)];
        let y_mode_nofilt = if b.y_mode == FILTER_PRED {
            DC_PRED
        } else {
            b.y_mode
        };
        let uv_mode = b.uv_mode;
        let pal_sz0 = b.pal_sz[0];
        let pal_sz1 = b.pal_sz[1];
        if b.intra {
            let above_idx = self.above_base + ai;
            let uv_pal_sz = if has_chroma { pal_sz1 } else { 0 };
            fill_intra_side_ctx(
                &mut self.blocks.above[above_idx],
                &mut self.blocks.above_pal_uv_sz[above_idx],
                bx4,
                b_dim[2],
                t_dim[2] as i8,
                y_mode_nofilt,
                pal_sz0,
                seg_pred,
                u8::from(b.skip),
                uv_pal_sz,
                is_inter_frame,
            );
            fill_intra_side_ctx(
                &mut self.blocks.left,
                &mut self.blocks.left_pal_uv_sz,
                by4,
                b_dim[3],
                t_dim[3] as i8,
                y_mode_nofilt,
                pal_sz0,
                seg_pred,
                u8::from(b.skip),
                uv_pal_sz,
                is_inter_frame,
            );
            if pal_sz0 != 0 {
                for x in 0..bw4 {
                    self.blocks.above_pal[above_idx][bx4 + x] = self.pal;
                }
                for y in 0..bh4 {
                    self.blocks.left_pal[by4 + y] = self.pal;
                }
            }
            if has_chroma {
                fill_small(
                    &mut self.blocks.above[above_idx].uvmode,
                    cbx4,
                    cbw4,
                    uv_mode,
                );
                fill_small(&mut self.blocks.left.uvmode, cby4, cbh4, uv_mode);
                if pal_sz1 != 0 {
                    for pl in 1..3 {
                        for x in 0..bw4 {
                            self.blocks.above_pal[above_idx][bx4 + x][pl] = self.pal[pl];
                        }
                        for y in 0..bh4 {
                            self.blocks.left_pal[by4 + y][pl] = self.pal[pl];
                        }
                    }
                }
            }
        } else if !is_inter_frame {
            // dav1d `decode.c:1366-1389`: intrabc bağlam güncellemesi.
            // `tx` **yazılmaz** (`read_vartx_tree` yazdı); `tx_intra` blok
            // boyutunun log2'sidir; mod DC_PRED, palet 0, intra 0.
            //
            // Yalnız **gerçek intrabc** blokları içindir (`!is_inter_frame`):
            // inter karelerin inter blokları bağlamı `decode_inter_block`
            // içinde dav1d `decode.c:1909-1926` ile **zaten** günceller. Bu
            // dalın onlar için de çalışması `skip_mode`/`tx_intra`/`mode`/
            // `pal_sz`/`seg_pred` alanlarını sıfırlıyordu (T39: test2136
            // kare 2'de skip_mode CDF bağlamı 1 yerine 0 → MSAC kayması).
            let above_idx = self.above_base + ai;
            for i in 0..2 {
                let (off, count, val) = if i == 0 {
                    (bx4, 1usize << b_dim[2], b_dim[2])
                } else {
                    (by4, 1usize << b_dim[3], b_dim[3])
                };
                let ctx = if i == 0 {
                    &mut self.blocks.above[above_idx]
                } else {
                    &mut self.blocks.left
                };
                fill_small(&mut ctx.tx_intra, off, count, val as i8);
                fill_small(&mut ctx.mode, off, count, DC_PRED);
                fill_small(&mut ctx.pal_sz, off, count, 0);
                fill_small(&mut ctx.seg_pred, off, count, seg_pred);
                fill_small(&mut ctx.skip_mode, off, count, 0);
                fill_small(&mut ctx.intra, off, count, 0);
                fill_small(&mut ctx.skip, off, count, u8::from(b.skip));
                if i == 0 {
                    fill_small(&mut self.blocks.above_pal_uv_sz[above_idx], off, count, 0);
                } else {
                    fill_small(&mut self.blocks.left_pal_uv_sz, off, count, 0);
                }
            }
            if has_chroma {
                fill_small(
                    &mut self.blocks.above[above_idx].uvmode,
                    cbx4,
                    cbw4,
                    DC_PRED,
                );
                fill_small(&mut self.blocks.left.uvmode, cby4, cbh4, DC_PRED);
            }
        }
        if self.hdr.segmentation.enabled && self.hdr.segmentation.update_map {
            let stride = self.segmap_stride;
            let rel_bx = self.bx - self.segmap_x_base;
            let rel_by = self.by - self.segmap_y_base;
            for y in 0..bh4 {
                let row = rel_by + y;
                if row >= self.segmap.len() / stride {
                    break;
                }
                let base = row * stride + rel_bx;
                if base + bw4 > self.segmap.len() {
                    break;
                }
                self.segmap[base..base + bw4].fill(b.seg_id);
            }
        }
        if dsp_trace().blk_trace {
            // Blok düzeyi tanılama (varsayılan kapalı): kare, blok konumu,
            // araç seçimleri. Farklı karelerde ilk farklı bloğu bulmak için.
            crate::trace_eprintln!(
                "K {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
                self.frame_no,
                blk_bx,
                blk_by,
                bs,
                u8::from(b.intra),
                b.comp_type,
                b.interintra_type,
                b.motion_mode,
                b.filter2d,
                b.ref0,
                b.ref1,
                b.mv[0].x,
                b.mv[0].y,
                u8::from(b.skip_mode),
                u8::from(b.skip)
            );
        }
        Ok(())
    }

    /// `seg_pred` CDF bağlamı.
    fn seg_pred_ctx(&self, bx4: usize, by4: usize) -> usize {
        let ai = self.bx >> 5;
        let _ = ai;
        // Çağrı yeri bağlamı kendisi hesaplar; bu yardımcı yalnız indeks verir.
        usize::from(self.blocks.above[self.above_base + (self.bx >> 5)].seg_pred[bx4])
            + usize::from(self.blocks.left.seg_pred[by4])
    }

    /// `seg_pred` boolean'ı (bağlam hazır).
    fn pal_bool(
        &self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        mode_off: usize,
        ctx: usize,
    ) -> bool {
        let row = layout::CDF_m + mode_off + ctx * 2;
        msac.decode_bool_adapt(&mut cdf.cells_mut()[row..]) != 0
    }
}

struct TxLoopCtx<'a> {
    env: CoefEnv<'a>,
    tx_size: TxSize,
    bs: u8,
    y_mode_nofilt: u8,
    uv_mode: u8,
    plane: usize,
    tx: u8,
    txw: usize,
    txh: usize,
}

impl<'a> Dec<'a> {
    /// `dsp_trace.c` `P` satırının bağlam alanları — dav1d `t->a`/`t->l`
    /// **güncel** blok konumundan (`t->bx`/`t->by`, recon sırasında parça
    /// kadar ilerler).
    fn ipred_ctx(&self) -> (u8, u8, u8, u8) {
        let ai = self.bx >> 5;
        let cbx4 = (self.bx >> self.ss_hor) & 31;
        let cby4 = (self.by >> self.ss_ver) & 31;
        (
            self.blocks.above[self.above_base + ai].uvmode[cbx4],
            self.blocks.left.uvmode[cby4],
            self.blocks.above[self.above_base + ai].intra[self.bx & 31],
            self.blocks.left.intra[self.by & 31],
        )
    }

    /// `dav1d_recon_b_intra` (`recon_tmpl.c:1176-1556`).
    #[allow(clippy::too_many_lines)]
    fn recon_b_intra(
        &mut self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        bs: u8,
        intra_edge_flags: u8,
        b: &Av1Block,
    ) -> Result<()> {
        let trace_frame_ok = dsp_trace().frame_ok(self.frame_no);
        let probe_on = probe::enabled();
        if trace_frame_ok {
            block_trace_line(self.frame_no, self.bx, self.by, bs, b);
        }
        let b_dim = &t::BLOCK_DIMENSIONS[usize::from(bs)];
        let bw4 = usize::from(b_dim[0]);
        let bh4 = usize::from(b_dim[1]);
        let bx4 = self.bx & 31;
        let by4 = self.by & 31;
        let ai = self.bx >> 5;
        let w4 = bw4.min(self.bw - self.bx);
        let h4 = bh4.min(self.bh - self.by);
        let ss_hor = self.ss_hor;
        let ss_ver = self.ss_ver;
        let cbx4 = bx4 >> ss_hor;
        let cby4 = by4 >> ss_ver;
        let cbw4 = (bw4 + ss_hor) >> ss_hor;
        let cbh4 = (bh4 + ss_ver) >> ss_ver;
        let cw4 = (w4 + ss_hor) >> ss_hor;
        let ch4 = (h4 + ss_ver) >> ss_ver;
        let has_chroma = self.has_chroma
            && (bw4 > ss_hor || self.bx & 1 != 0)
            && (bh4 > ss_ver || self.by & 1 != 0);
        let t_dim = t::TXFM_DIMENSIONS[usize::from(b.tx)];
        let uv_t_dim = t::TXFM_DIMENSIONS[usize::from(b.uvtx)];
        let txw = usize::from(t_dim[0]);
        let txh = usize::from(t_dim[1]);
        let uv_txw = usize::from(uv_t_dim[0]);
        let uv_txh = usize::from(uv_t_dim[1]);
        let bpp = self.bd.bytes_per_sample();
        let bitdepth_max = self.bd.max_sample() as u16;
        let intra_edge_filter_flag = i32::from(self.seq.intra_edge_filter) << 10;
        let (chroma_tr_mask, chroma_bl_mask) = chroma_edge_masks(self.seq.layout.raw());
        let ipred = self.bdsp.ipred();
        let bx0 = self.bx;
        let by0 = self.by;
        let col_start = self.col_start;
        let row_start = self.row_start;
        let col_end = self.col_end;
        let row_end = self.row_end;
        let bw = self.bw;
        let bh = self.bh;
        let bd = self.bd;
        let seg_id = usize::from(b.seg_id);
        let dq_seg = self.dq[seg_id];
        let y_mode_nofilt = if b.y_mode == FILTER_PRED {
            t::FILTER_MODE_TO_Y_MODE[b.y_angle.clamp(0, 4) as usize]
        } else {
            b.y_mode
        };
        let fast_tx = !b.skip && !trace_frame_ok && !probe_on;
        let tx_size_y = if fast_tx {
            Some(
                TxSize::from_index(usize::from(b.tx))
                    .ok_or(Error::InvalidData("geçersiz tx boyutu"))?,
            )
        } else {
            None
        };
        let tx_size_uv = if fast_tx && has_chroma {
            Some(
                TxSize::from_index(usize::from(b.uvtx))
                    .ok_or(Error::InvalidData("geçersiz tx boyutu"))?,
            )
        } else {
            None
        };
        let env_base = CoefEnv {
            lossless: self.hdr.segmentation.lossless[seg_id],
            reduced_txtp: self.hdr.reduced_txtp_set,
            seg_qidx: self.hdr.segmentation.qidx[seg_id],
            bit_depth: bd,
            ss_ver: ss_ver != 0,
            ss_hor: ss_hor != 0,
            dq: &dq_seg,
            qm: if self.hdr.quant.qm {
                Some(crate::decode::coef::QmSel {
                    level: self.hdr.quant.qm_y,
                    class: 0,
                    tx: b.tx,
                })
            } else {
                None
            },
            inter: !b.intra,
            txtp_hint: 0,
        };

        // --- luma ----------------------------------------------------------
        // --- luma + kroma: dav1d parça (chunk) düzeni -----------------------
        // dav1d `recon_tmpl.c:1197-1556`: 64x64 px'lik parçalar `init_y` dış,
        // `init_x` iç döngüde; her parçada ÖNCE luma, SONRA kroma (U, V)
        // katsayıları çözülür. Önceki sürüm bütün luma'yı, sonra bütün
        // kroma'yı çözüyordu: bu, 64x64'ten büyük bloklarda MSAC sembol
        // sırasını (ve CDF uyarlamasını) dav1d'den kaydırıyordu.
        let y_stride = self.pic.stride[0];
        let uv_stride = self.pic.stride[1];
        let edge_ctx_y = EdgeCtx {
            col_start,
            row_start,
            col_end,
            row_end,
            stride: y_stride,
            mode_in: b.y_mode,
            angle_in: b.y_angle,
            tw: txw,
            th: txh,
            filter_edge: self.seq.intra_edge_filter,
            topleft_off: 128 * bpp,
            bd,
        };
        let tx_ctx_y = TxLoopCtx {
            env: env_base,
            tx_size: tx_size_y.unwrap_or(TxSize::Tx4x4),
            bs,
            y_mode_nofilt,
            uv_mode: b.uv_mode,
            plane: 0,
            tx: b.tx,
            txw,
            txh,
        };
        let sm_uv = sm_uv_flag(&self.blocks.above[self.above_base + ai], cbx4)
            | sm_uv_flag(&self.blocks.left, cby4);
        let mut init_y = 0usize;
        while init_y < h4 {
            let sub_h4 = h4.min(init_y + 16);
            let sub_ch4 = ch4.min((init_y + 16) >> ss_ver);
            let mut init_x = 0usize;
            while init_x < w4 {
                if b.pal_sz[0] != 0 {
                    let off = self.pic.off(0, 4 * self.bx, 4 * self.by);
                    let pal = self.pal_bytes(0);
                    let stride = self.pic.stride[0];
                    let dst = &mut self.pic.data[0];
                    crate::dsp::profil_olc!(
                        PalPred,
                        (ipred.pal_pred)(
                            &mut dst[off..],
                            stride,
                            &pal,
                            &self.scratch.pal_idx_y,
                            bw4 * 4,
                            bh4 * 4,
                            bd,
                        )
                    );
                    self.probe_pred(
                        2,
                        0,
                        4 * self.bx,
                        4 * self.by,
                        bw4 * 4,
                        bh4 * 4,
                        0,
                        0,
                        0,
                        (4 * bw) as i32,
                        (4 * bh) as i32,
                        0,
                    );
                }
                let sm = sm_flag(&self.blocks.above[self.above_base + ai], bx4)
                    | sm_flag(&self.blocks.left, by4);
                let intra_flags = sm | intra_edge_filter_flag;
                let sb_has_tr: u8 = if init_x + 16 < w4 {
                    1
                } else if init_y > 0 {
                    0
                } else {
                    intra_edge_flags & EDGE_I444_TOP_HAS_RIGHT
                };
                let sb_has_bl: u8 = if init_x > 0 {
                    0
                } else if init_y + 16 < h4 {
                    1
                } else {
                    intra_edge_flags & EDGE_I444_LEFT_HAS_BOTTOM
                };
                let sub_w4 = w4.min(init_x + 16);
                if b.pal_sz[0] == 0 || !b.skip || probe_on {
                    let mut y = init_y;
                    self.by = by0 + init_y;
                    let step_x_y = txw * 4 * bpp;
                    while y < sub_h4 {
                        let mut x = init_x;
                        self.bx = bx0 + init_x;
                        let mut dst_off = self.pic.off(0, 4 * self.bx, 4 * self.by);
                        while x < sub_w4 {
                            if b.pal_sz[0] == 0 {
                                let tr = if (y > init_y || sb_has_tr == 0) && x + txw >= sub_w4 {
                                    0
                                } else {
                                    EDGE_I444_TOP_HAS_RIGHT
                                };
                                let bl = if x > init_x || (sb_has_bl == 0 && y + txh >= sub_h4) {
                                    0
                                } else {
                                    EDGE_I444_LEFT_HAS_BOTTOM
                                };
                                let edge_flags2 = tr | bl;
                                let prepared = prepare_intra_edges_ctx(
                                    &edge_ctx_y,
                                    &self.pic.data[0],
                                    self.bx,
                                    self.by,
                                    edge_flags2,
                                    dst_off,
                                    &mut self.scratch.edge,
                                );
                                let angle = prepared.angle;
                                let dst = &mut self.pic.data[0];
                                let edge_bytes = &self.scratch.edge;
                                crate::dsp::profil_olc!(
                                    IntraPred,
                                    ipred.intra_pred[usize::from(prepared.mode)](
                                        &mut dst[dst_off..],
                                        y_stride,
                                        Edge::new(edge_bytes, 128 * bpp),
                                        txw * 4,
                                        txh * 4,
                                        (angle | intra_flags) as usize,
                                        4 * bw - 4 * self.bx,
                                        4 * bh - 4 * self.by,
                                        bd,
                                        bitdepth_max,
                                    )
                                );
                                if trace_frame_ok {
                                    let hash = pred_hash(
                                        &self.pic.data[0],
                                        dst_off,
                                        y_stride,
                                        txw * 4,
                                        txh * 4,
                                        bpp,
                                    );
                                    let (au, lu, ai_ctx, li) = self.ipred_ctx();
                                    crate::trace_eprintln!(
                                        "P {} {} {} {} {} {} {:016x} {:016x} {} {} {} {} {} {} {}",
                                        self.frame_no,
                                        prepared.mode,
                                        self.bx,
                                        self.by,
                                        txw * 4,
                                        txh * 4,
                                        hash,
                                        edge_hash_used(
                                            edge_bytes,
                                            128 * bpp,
                                            txw * 4,
                                            txh * 4,
                                            usize::from(prepared.mode),
                                            bpp
                                        ),
                                        (angle | intra_flags) as usize,
                                        4 * bw - 4 * self.bx,
                                        4 * bh - 4 * self.by,
                                        au,
                                        lu,
                                        ai_ctx,
                                        li
                                    );
                                }
                                if probe_on {
                                    self.probe_pred(
                                        0,
                                        0,
                                        4 * self.bx,
                                        4 * self.by,
                                        txw * 4,
                                        txh * 4,
                                        i32::from(prepared.mode),
                                        angle | intra_flags,
                                        intra_flags,
                                        (4 * bw - 4 * self.bx) as i32,
                                        (4 * bh - 4 * self.by) as i32,
                                        0,
                                    );
                                }
                            }
                            if fast_tx {
                                self.decode_tx_coefs_fast(
                                    msac, cdf, &tx_ctx_y, dst_off, self.bx, self.by,
                                )?;
                            } else if !b.skip || probe_on {
                                self.decode_tx_coefs(
                                    msac,
                                    cdf,
                                    bs,
                                    b,
                                    0,
                                    b.tx,
                                    dst_off,
                                    self.bx,
                                    self.by,
                                    trace_frame_ok,
                                    probe_on,
                                )?;
                            }
                            x += txw;
                            self.bx += txw;
                            dst_off += step_x_y;
                        }
                        self.bx = bx0 + init_x;
                        y += txh;
                        self.by += txh;
                    }
                    self.bx = bx0;
                    self.by = by0;
                }
                if has_chroma {
                    // CfL/palet tahmini parçadan önce yapılır (CFL ve palet
                    // yalnız tek parçalı bloklarda geçerlidir; dav1d de
                    // CFL için `assert(!init_x && !init_y)` yapar).
                    if init_x == 0 && init_y == 0 {
                        let uv_stride = self.pic.stride[1];
                        if b.uv_mode == CFL_PRED {
                            let y_src_off =
                                self.pic
                                    .off(0, 4 * (self.bx & !ss_hor), 4 * (self.by & !ss_ver));
                            let furthest_r = ((cw4 << ss_hor) + txw - 1) & !(txw - 1);
                            let furthest_b = ((ch4 << ss_ver) + txh - 1) & !(txh - 1);
                            // dav1d `recon_tmpl.c:1342-1357`: `y_src` bloğun luma sol-üst
                            // köşesidir; çekirdek ofset almaz, dilim oradan başlamalıdır.
                            crate::dsp::profil_olc!(
                                CflAc,
                                ipred.cfl_ac[match self.chroma {
                                    ChromaFormat::C420 => 0,
                                    ChromaFormat::C422 => 1,
                                    _ => 2,
                                }](
                                    &mut self.scratch.ac,
                                    &self.pic.data[0][y_src_off..],
                                    self.pic.stride[0],
                                    cbw4 - (furthest_r >> ss_hor),
                                    cbh4 - (furthest_b >> ss_ver),
                                    cbw4 * 4,
                                    cbh4 * 4,
                                    bd,
                                )
                            );
                            for pl in 0..2 {
                                if b.cfl_alpha[pl] == 0 {
                                    continue;
                                }
                                let px = self.bx >> ss_hor;
                                let py = self.by >> ss_ver;
                                let uv_off = self.pic.off(1 + pl, px * 4, py * 4);
                                let prepared = prepare_intra_edges(
                                    px,
                                    px > (col_start >> ss_hor),
                                    py,
                                    py > (row_start >> ss_ver),
                                    col_end >> ss_hor,
                                    row_end >> ss_ver,
                                    0,
                                    &self.pic.data[1 + pl],
                                    uv_off,
                                    uv_stride,
                                    DC_PRED,
                                    0,
                                    uv_txw,
                                    uv_txh,
                                    false,
                                    &mut self.scratch.edge,
                                    128 * bpp,
                                    bd,
                                );
                                let edge_bytes = &self.scratch.edge;
                                let dst = &mut self.pic.data[1 + pl];
                                let cfl_mode = match prepared.mode {
                                    DC_PRED => 0,
                                    3 => 1, // LEFT_DC_PRED
                                    4 => 2, // TOP_DC_PRED
                                    5 => 3, // DC_128_PRED
                                    _ => {
                                        return Err(Error::InvalidData(
                                            "CFL için geçersiz DC modu",
                                        ));
                                    }
                                };
                                crate::dsp::profil_olc!(
                                    CflPred,
                                    ipred.cfl_pred[cfl_mode](
                                        &mut dst[uv_off..],
                                        uv_stride,
                                        Edge::new(edge_bytes, 128 * bpp),
                                        uv_txw * 4,
                                        uv_txh * 4,
                                        &self.scratch.ac,
                                        b.cfl_alpha[pl],
                                        bd,
                                        bitdepth_max,
                                    )
                                );
                                self.probe_pred(
                                    1,
                                    1 + pl,
                                    px * 4,
                                    py * 4,
                                    uv_txw * 4,
                                    uv_txh * 4,
                                    cfl_mode as i32,
                                    0,
                                    0,
                                    ((4 * bw + ss_hor - 4 * (self.bx & !ss_hor)) >> ss_hor) as i32,
                                    ((4 * bh + ss_ver - 4 * (self.by & !ss_ver)) >> ss_ver) as i32,
                                    b.cfl_alpha[pl],
                                );
                            }
                        } else if b.pal_sz[1] != 0 {
                            let uv_off =
                                self.pic
                                    .off(1, 4 * (self.bx >> ss_hor), 4 * (self.by >> ss_ver));
                            // dav1d `recon_tmpl.c:1412-1416`: U için `pal[1]`, V için `pal[2]`.
                            // Düzlem başına ayrı 16 baytlık palet; `pal_bytes(1)` yalnız 1.
                            // düzlemi doldurur, bu yüzden iki ayrı çağrı gerekir.
                            let pal_u = self.pal_bytes(1);
                            let pal_v = self.pal_bytes(2);
                            for pl in 0..2 {
                                let stride = self.pic.stride[1];
                                let dst = &mut self.pic.data[1 + pl];
                                let pal = if pl == 0 {
                                    &pal_u[16..32]
                                } else {
                                    &pal_v[32..48]
                                };
                                crate::dsp::profil_olc!(
                                    PalPred,
                                    (ipred.pal_pred)(
                                        &mut dst[uv_off..],
                                        stride,
                                        pal,
                                        &self.scratch.pal_idx_uv,
                                        cbw4 * 4,
                                        cbh4 * 4,
                                        bd,
                                    )
                                );
                                self.probe_pred(
                                    2,
                                    1 + pl,
                                    4 * (self.bx >> ss_hor),
                                    4 * (self.by >> ss_ver),
                                    cbw4 * 4,
                                    cbh4 * 4,
                                    0,
                                    0,
                                    0,
                                    ((4 * bw + ss_hor - 4 * (self.bx & !ss_hor)) >> ss_hor) as i32,
                                    ((4 * bh + ss_ver - 4 * (self.by & !ss_ver)) >> ss_ver) as i32,
                                    0,
                                );
                            }
                        }
                    }
                    // dav1d `recon_tmpl.c:1428-1430`: üçlü koşul. `1 & ef & mask`
                    // yazımı önce `1 & ef` yapar ve yalnız bit0'ı bırakır;
                    // 4:2:2/4:2:0 maskesi (bit1/bit2) böylece **hep 0** olur ve
                    // kroma Z1/Z2/Z3 üst-sağ kenarı yanlış (doldurulmuş) olur.
                    let uv_sb_has_tr: u8 = if ((init_x + 16) >> ss_hor) < cw4 {
                        1
                    } else if init_y != 0 {
                        0
                    } else {
                        intra_edge_flags & chroma_tr_mask
                    };
                    let uv_sb_has_bl: u8 = if init_x > 0 {
                        0
                    } else {
                        let has_bl = ((init_y + 16) >> ss_ver) < ch4;
                        u8::from(has_bl) | (intra_edge_flags & chroma_bl_mask)
                    };
                    let sub_cw4 = cw4.min((init_x + 16) >> ss_hor);
                    let uv_mode_eff = if b.uv_mode == CFL_PRED {
                        DC_PRED
                    } else {
                        b.uv_mode
                    };
                    let edge_ctx_uv = EdgeCtx {
                        col_start: col_start >> ss_hor,
                        row_start: row_start >> ss_ver,
                        col_end: col_end >> ss_hor,
                        row_end: row_end >> ss_ver,
                        stride: uv_stride,
                        mode_in: uv_mode_eff,
                        angle_in: b.uv_angle,
                        tw: uv_txw,
                        th: uv_txh,
                        filter_edge: self.seq.intra_edge_filter,
                        topleft_off: 128 * bpp,
                        bd,
                    };
                    let step_x_uv = uv_txw * 4 * bpp;
                    for pl in 0..2 {
                        let has_pred =
                            !(b.uv_mode == CFL_PRED && b.cfl_alpha[pl] != 0) && b.pal_sz[1] == 0;
                        if !has_pred && b.skip && !probe_on {
                            continue;
                        }
                        let env_uv = CoefEnv {
                            qm: if self.hdr.quant.qm {
                                Some(crate::decode::coef::QmSel {
                                    level: if pl == 0 {
                                        self.hdr.quant.qm_u
                                    } else {
                                        self.hdr.quant.qm_v
                                    },
                                    class: 1,
                                    tx: b.uvtx,
                                })
                            } else {
                                None
                            },
                            ..env_base
                        };
                        let tx_ctx_uv = TxLoopCtx {
                            env: env_uv,
                            tx_size: tx_size_uv.unwrap_or(TxSize::Tx4x4),
                            bs,
                            y_mode_nofilt,
                            uv_mode: b.uv_mode,
                            plane: 1 + pl,
                            tx: b.uvtx,
                            txw: uv_txw,
                            txh: uv_txh,
                        };
                        let mut y = init_y >> ss_ver;
                        self.by = by0 + init_y;
                        self.bx = bx0;
                        while y < sub_ch4 {
                            let mut x = init_x >> ss_hor;
                            self.bx = bx0 + init_x;
                            let mut dst_off = self.pic.off(
                                1 + pl,
                                4 * (self.bx >> ss_hor),
                                4 * (self.by >> ss_ver),
                            );
                            while x < sub_cw4 {
                                if has_pred {
                                    let edge_flags2 = (if (y > (init_y >> ss_ver)
                                        || uv_sb_has_tr == 0)
                                        && (x + uv_txw >= sub_cw4)
                                    {
                                        0
                                    } else {
                                        EDGE_I444_TOP_HAS_RIGHT
                                    }) | (if x > (init_x >> ss_hor)
                                        || (uv_sb_has_bl == 0 && y + uv_txh >= sub_ch4)
                                    {
                                        0
                                    } else {
                                        EDGE_I444_LEFT_HAS_BOTTOM
                                    });
                                    let px = self.bx >> ss_hor;
                                    let py = self.by >> ss_ver;
                                    let prepared = prepare_intra_edges_ctx(
                                        &edge_ctx_uv,
                                        &self.pic.data[1 + pl],
                                        px,
                                        py,
                                        edge_flags2,
                                        dst_off,
                                        &mut self.scratch.edge,
                                    );
                                    let angle = prepared.angle;
                                    let edge_bytes = &self.scratch.edge;
                                    let dst = &mut self.pic.data[1 + pl];
                                    crate::dsp::profil_olc!(
                                        IntraPred,
                                        ipred.intra_pred[usize::from(prepared.mode)](
                                            &mut dst[dst_off..],
                                            uv_stride,
                                            Edge::new(edge_bytes, 128 * bpp),
                                            uv_txw * 4,
                                            uv_txh * 4,
                                            (angle | sm_uv | intra_edge_filter_flag) as usize,
                                            (4 * bw + ss_hor - 4 * (self.bx & !ss_hor)) >> ss_hor,
                                            (4 * bh + ss_ver - 4 * (self.by & !ss_ver)) >> ss_ver,
                                            bd,
                                            bitdepth_max,
                                        )
                                    );
                                    if trace_frame_ok {
                                        let hash = pred_hash(
                                            &self.pic.data[1 + pl],
                                            dst_off,
                                            uv_stride,
                                            uv_txw * 4,
                                            uv_txh * 4,
                                            bpp,
                                        );
                                        let (au, lu, ai_ctx, li) = self.ipred_ctx();
                                        crate::trace_eprintln!(
                                            "P {} {} {} {} {} {} {:016x} {:016x} {} {} {} {} {} {} {}",
                                            self.frame_no,
                                            prepared.mode,
                                            self.bx,
                                            self.by,
                                            uv_txw * 4,
                                            uv_txh * 4,
                                            hash,
                                            edge_hash_used(
                                                edge_bytes,
                                                128 * bpp,
                                                uv_txw * 4,
                                                uv_txh * 4,
                                                usize::from(prepared.mode),
                                                bpp
                                            ),
                                            (angle | sm_uv | intra_edge_filter_flag) as usize,
                                            (4 * bw + ss_hor - 4 * (self.bx & !ss_hor)) >> ss_hor,
                                            (4 * bh + ss_ver - 4 * (self.by & !ss_ver)) >> ss_ver,
                                            au,
                                            lu,
                                            ai_ctx,
                                            li
                                        );
                                    }
                                    if probe_on {
                                        self.probe_pred(
                                            0,
                                            1 + pl,
                                            4 * px,
                                            4 * py,
                                            uv_txw * 4,
                                            uv_txh * 4,
                                            i32::from(prepared.mode),
                                            angle | sm_uv | intra_edge_filter_flag,
                                            sm_uv | intra_edge_filter_flag,
                                            ((4 * bw + ss_hor - 4 * (self.bx & !ss_hor)) >> ss_hor)
                                                as i32,
                                            ((4 * bh + ss_ver - 4 * (self.by & !ss_ver)) >> ss_ver)
                                                as i32,
                                            0,
                                        );
                                    }
                                }
                                if fast_tx {
                                    self.decode_tx_coefs_fast(
                                        msac, cdf, &tx_ctx_uv, dst_off, self.bx, self.by,
                                    )?;
                                } else if !b.skip || probe_on {
                                    self.decode_tx_coefs(
                                        msac,
                                        cdf,
                                        bs,
                                        b,
                                        1 + pl,
                                        b.uvtx,
                                        dst_off,
                                        self.bx,
                                        self.by,
                                        trace_frame_ok,
                                        probe_on,
                                    )?;
                                }
                                x += uv_txw;
                                self.bx += uv_txw << ss_hor;
                                dst_off += step_x_uv;
                            }
                            self.bx = bx0 + init_x;
                            y += uv_txh;
                            self.by += uv_txh << ss_ver;
                        }
                        self.by = by0;
                    }
                }
                init_x += 16;
            }
            init_y += 16;
        }
        if b.skip && !probe_on {
            let above_idx = self.above_base + ai;
            fill_small(&mut self.blocks.above[above_idx].lcoef, bx4, w4, 0x40);
            fill_small(&mut self.blocks.left.lcoef, by4, h4, 0x40);
            if has_chroma {
                for pl in 0..2 {
                    fill_small(&mut self.blocks.above[above_idx].ccoef[pl], cbx4, cw4, 0x40);
                    fill_small(&mut self.blocks.left.ccoef[pl], cby4, ch4, 0x40);
                }
            }
        }
        self.bx = bx0;
        self.by = by0;
        Ok(())
    }
}

/// `sm_flag` (`ipred_prepare.h:95-100`).
fn sm_flag(ctx: &BlockContext, idx: usize) -> i32 {
    if ctx.intra[idx] == 0 {
        return 0;
    }
    match ctx.mode[idx] {
        9..=11 => 512,
        _ => 0,
    }
}

/// `sm_uv_flag` (`ipred_prepare.h:102-107`).
fn sm_uv_flag(ctx: &BlockContext, idx: usize) -> i32 {
    match ctx.uvmode[idx] {
        9..=11 => 512,
        _ => 0,
    }
}

/// `order_palette` (`decode.c:353-412`).
fn order_palette(
    pal_idx: &[u8],
    stride: usize,
    i: usize,
    first: usize,
    last: usize,
    order: &mut [[u8; 8]; 64],
    ctx: &mut [u8; 64],
) {
    let mut have_top = i > first;
    let mut base = first + (i - first) * stride;
    for (n, j) in (last..=first).rev().enumerate() {
        let have_left = j > 0;
        let mut mask = 0u16;
        let mut o_idx = 0usize;
        let mut add = |v: u8| {
            order[n][o_idx] = v;
            o_idx += 1;
            mask |= 1 << v;
        };
        if !have_left {
            ctx[n] = 0;
            add(pal_idx[base - stride]);
        } else if !have_top {
            ctx[n] = 0;
            add(pal_idx[base - 1]);
        } else {
            let l = pal_idx[base - 1];
            let t = pal_idx[base - stride];
            let tl = pal_idx[base - stride - 1];
            let same_t_l = t == l;
            let same_t_tl = t == tl;
            let same_l_tl = l == tl;
            let same_all = same_t_l & same_t_tl & same_l_tl;
            if same_all {
                ctx[n] = 4;
                add(t);
            } else if same_t_l {
                ctx[n] = 3;
                add(t);
                add(tl);
            } else if same_t_tl | same_l_tl {
                ctx[n] = 2;
                add(tl);
                add(if same_t_tl { l } else { t });
            } else {
                ctx[n] = 1;
                add(t.min(l));
                add(t.max(l));
                add(tl);
            }
        }
        for (bit, m) in (0u16..8).map(|b| (b, 1u16 << b)) {
            if mask & m == 0 {
                order[n][o_idx] = bit as u8;
                o_idx += 1;
            }
        }
        debug_assert_eq!(o_idx, 8);
        have_top = true;
        base = base + stride - 1;
        let _ = j;
    }
}

impl<'a> Dec<'a> {
    /// Paleti `pal_pred` bayt biçimine çevirir (düzlem başına 16 bayt).
    /// Yeniden oynatma probu (blok düzeyi tanılama) — tahmin kaydı.
    ///
    /// Yalnız `probe` alıcısı ve hedefi kuruluysa iş yapar; aksi hâlde tek bir
    /// `enabled()` denetimidir ve üretim davranışını **değiştirmez**.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn probe_pred(
        &self,
        kind: u8,
        plane: usize,
        x: usize,
        y: usize,
        w: usize,
        h: usize,
        mode: i32,
        angle: i32,
        flags: i32,
        max_width: i32,
        max_height: i32,
        alpha: i32,
    ) {
        if !probe::enabled() {
            return;
        }
        let pred = self.probe_gather(plane, x, y, w, h);
        let edge: &[u8] = if kind < 2 {
            &self.scratch.edge
        } else {
            &[][..]
        };
        let ac: &[i16] = if kind == 1 {
            &self.scratch.ac[..(w * h).min(self.scratch.ac.len())]
        } else {
            &[][..]
        };
        let (idx, pal): (&[u8], &[u16]) = if kind == 2 {
            let src = if plane == 0 {
                &self.scratch.pal_idx_y
            } else {
                &self.scratch.pal_idx_uv
            };
            (&src[..(w / 2 * h).min(src.len())], &self.pal[plane])
        } else {
            (&[][..], &[0u16; 8])
        };
        let layout = match self.chroma {
            ChromaFormat::C420 => 1,
            ChromaFormat::C422 => 2,
            ChromaFormat::C444 => 3,
            ChromaFormat::Monochrome => 0,
        };
        let rec = probe::Pred {
            kind,
            mode,
            angle,
            flags,
            max_width,
            max_height,
            alpha,
            bitdepth_max: if self.bd.bits() == 8 {
                255
            } else {
                (1 << self.bd.bits()) - 1
            },
            tx: 0,
            txtp: 0,
            eob: -1,
            x: x as u32,
            y: y as u32,
            w: w as u32,
            h: h as u32,
        };
        probe::pred_record(
            rec,
            plane as u8,
            self.bd.bits() as u8,
            layout,
            edge,
            ac,
            idx,
            pal,
            &pred,
        );
    }

    /// Düzlemden blok anlık görüntüsü (bayt, satır satır).
    fn probe_gather(&self, plane: usize, x: usize, y: usize, w: usize, h: usize) -> Vec<u8> {
        let bpp = self.bd.bytes_per_sample();
        let data = &self.pic.data[plane];
        let mut out = Vec::with_capacity(w * h * bpp);
        for row in 0..h {
            let start = self.pic.off(plane, x, y + row);
            out.extend_from_slice(&data[start..start + w * bpp]);
        }
        out
    }

    fn pal_bytes(&self, plane: usize) -> [u8; 48] {
        let mut out = [0u8; 48];
        let bpp = self.bd.bytes_per_sample();
        for (i, &v) in self.pal[plane].iter().enumerate() {
            // 8-bit'te `pal_pred` paleti **bayt** olarak okur (`pal[i]`), 16-bit'te
            // `u16` (`ipred_tmpl.c:723-735`, `PIXEL`). Her düzlem 16 baytlık
            // yuvaya yazılır; 8-bit'te yalnız ilk 8 baytı doludur.
            let dst = plane * 16 + i * bpp;
            out[dst..dst + bpp].copy_from_slice(&v.to_le_bytes()[..bpp]);
        }
        out
    }

    /// Hızlı intra katsayı çözüm yolu (`!b.skip && !trace && !probe`).
    #[allow(clippy::too_many_arguments)]
    fn decode_tx_coefs_fast(
        &mut self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        ctx: &TxLoopCtx<'_>,
        dst_off: usize,
        bx: usize,
        by: usize,
    ) -> Result<()> {
        let bx4 = bx & 31;
        let by4 = by & 31;
        let ai = bx >> 5;
        let plane = ctx.plane;
        let chroma = plane != 0;
        let (ctw, cth, cbx4, cby4) = if chroma {
            (
                ctx.txw.min((self.bw - bx + self.ss_hor) >> self.ss_hor),
                ctx.txh.min((self.bh - by + self.ss_ver) >> self.ss_ver),
                chroma_ctx4(bx4, self.ss_hor),
                chroma_ctx4(by4, self.ss_ver),
            )
        } else {
            (
                ctx.txw.min(self.bw - bx),
                ctx.txh.min(self.bh - by),
                bx4,
                by4,
            )
        };
        let (a, l): (&[u8], &[u8]) = if chroma {
            let pl = plane - 1;
            (
                &self.blocks.above[self.above_base + ai].ccoef[pl][cbx4..],
                &self.blocks.left.ccoef[pl][cby4..],
            )
        } else {
            (
                &self.blocks.above[self.above_base + ai].lcoef[bx4..],
                &self.blocks.left.lcoef[by4..],
            )
        };
        #[cfg(feature = "msac-trace")]
        crate::trace::mark("tx", plane as i32, bx as i32, by as i32, i32::from(ctx.tx));
        let result = crate::dsp::profil_olc!(
            KatsayiCozme,
            decode_coefs(
                msac,
                cdf,
                &ctx.env,
                ctx.tx,
                ctx.bs,
                ctx.y_mode_nofilt,
                ctx.uv_mode,
                plane,
                a,
                l,
                &mut self.scratch.cf,
                &mut self.scratch.levels,
            )
        );
        let res_ctx = result.res_ctx;
        let eob = result.eob;
        if chroma {
            let pl = plane - 1;
            let above_idx = self.above_base + ai;
            fill_small(
                &mut self.blocks.above[above_idx].ccoef[pl],
                cbx4,
                ctw,
                res_ctx,
            );
            fill_small(&mut self.blocks.left.ccoef[pl], cby4, cth, res_ctx);
        } else {
            let above_idx = self.above_base + ai;
            fill_small(&mut self.blocks.above[above_idx].lcoef, bx4, ctw, res_ctx);
            fill_small(&mut self.blocks.left.lcoef, by4, cth, res_ctx);
        }
        if eob >= 0 {
            let Some(tx_type) = TxType::from_index(usize::from(result.txtp)) else {
                return Err(Error::InvalidData("geçersiz tx türü"));
            };
            let itx = self.bdsp.itx().itxfm_add[ctx.tx_size.index()][tx_type.index()];
            let stride = self.pic.stride[plane];
            let dst = &mut self.pic.data[plane];
            crate::dsp::profil_olc!(
                ItxfmAdd,
                itx(
                    &mut dst[dst_off..],
                    stride,
                    &mut self.scratch.cf,
                    eob as usize,
                    ctx.tx_size,
                    tx_type,
                    self.bd,
                )
            );
        }
        Ok(())
    }

    /// Tek bir tx bloğunun katsayılarını çözer, dequant eder ve itx uygular.
    #[allow(clippy::too_many_arguments)]
    #[cold]
    #[inline(never)]
    fn decode_tx_coefs(
        &mut self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        bs: u8,
        b: &Av1Block,
        plane: usize,
        tx: u8,
        dst_off: usize,
        bx: usize,
        by: usize,
        trace_on: bool,
        probe_on: bool,
    ) -> Result<()> {
        let bx4 = bx & 31;
        let by4 = by & 31;
        // Kroma bağlam indeksleri kroma 4x4 birimindedir (dav1d
        // `recon_tmpl.c:831` `cbx4 = bx4 >> ss_hor`).
        let cbx4 = chroma_ctx4(bx4, self.ss_hor);
        let cby4 = chroma_ctx4(by4, self.ss_ver);
        let ai = bx >> 5;
        let chroma = plane != 0;
        let t_dim = t::TXFM_DIMENSIONS[usize::from(tx)];
        let txw = usize::from(t_dim[0]);
        let txh = usize::from(t_dim[1]);
        let (ctw, cth) = if chroma {
            (
                txw.min((self.bw - bx + self.ss_hor) >> self.ss_hor),
                txh.min((self.bh - by + self.ss_ver) >> self.ss_ver),
            )
        } else {
            (txw.min(self.bw - bx), txh.min(self.bh - by))
        };
        if b.skip {
            if chroma {
                let pl = plane - 1;
                let a_c = &mut self.blocks.above[self.above_base + ai].ccoef[pl];
                let l_c = &mut self.blocks.left.ccoef[pl];
                if ctw == 1 {
                    a_c[cbx4] = 0x40;
                } else {
                    a_c[cbx4..cbx4 + ctw].fill(0x40);
                }
                if cth == 1 {
                    l_c[cby4] = 0x40;
                } else {
                    l_c[cby4..cby4 + cth].fill(0x40);
                }
            } else {
                let a_l = &mut self.blocks.above[self.above_base + ai].lcoef;
                let l_l = &mut self.blocks.left.lcoef;
                if ctw == 1 {
                    a_l[bx4] = 0x40;
                } else {
                    a_l[bx4..bx4 + ctw].fill(0x40);
                }
                if cth == 1 {
                    l_l[by4] = 0x40;
                } else {
                    l_l[by4..by4 + cth].fill(0x40);
                }
            }
            if probe_on {
                let (px, py) = if chroma {
                    ((bx >> self.ss_hor) * 4, (by >> self.ss_ver) * 4)
                } else {
                    (bx * 4, by * 4)
                };
                let (pw, ph) = (ctw * 4, cth * 4);
                if pw > 0 && ph > 0 {
                    let pixels = self.probe_gather(plane, px, py, pw, ph);
                    let coeffs = vec![0i32; 1024];
                    let layout = match self.chroma {
                        ChromaFormat::C420 => 1,
                        ChromaFormat::C422 => 2,
                        ChromaFormat::C444 => 3,
                        ChromaFormat::Monochrome => 0,
                    };
                    probe::tx_record(
                        plane as u8,
                        self.bd.bits() as u8,
                        layout,
                        px as u32,
                        py as u32,
                        pw as u32,
                        ph as u32,
                        tx,
                        0,
                        -1,
                        px as u32,
                        py as u32,
                        &pixels,
                        &coeffs,
                        &pixels,
                    );
                }
            }
            return Ok(());
        }
        let seg_id = usize::from(b.seg_id);
        let env = CoefEnv {
            lossless: self.hdr.segmentation.lossless[seg_id],
            reduced_txtp: self.hdr.reduced_txtp_set,
            seg_qidx: self.hdr.segmentation.qidx[seg_id],
            bit_depth: self.bd,
            ss_ver: self.ss_ver != 0,
            ss_hor: self.ss_hor != 0,
            dq: &self.dq[seg_id],
            qm: if self.hdr.quant.qm {
                Some(crate::decode::coef::QmSel {
                    level: match plane {
                        0 => self.hdr.quant.qm_y,
                        1 => self.hdr.quant.qm_u,
                        _ => self.hdr.quant.qm_v,
                    },
                    class: usize::from(plane != 0),
                    tx,
                })
            } else {
                None
            },
            inter: !b.intra,
            txtp_hint: 0,
        };
        let y_mode_nofilt = if b.y_mode == FILTER_PRED {
            t::FILTER_MODE_TO_Y_MODE[b.y_angle.clamp(0, 4) as usize]
        } else {
            b.y_mode
        };
        let (a, l): (&[u8], &[u8]) = if chroma {
            let pl = plane - 1;
            (
                &self.blocks.above[self.above_base + ai].ccoef[pl][cbx4..],
                &self.blocks.left.ccoef[pl][cby4..],
            )
        } else {
            (
                &self.blocks.above[self.above_base + ai].lcoef[bx4..],
                &self.blocks.left.lcoef[by4..],
            )
        };
        #[cfg(feature = "msac-trace")]
        crate::trace::mark("tx", plane as i32, bx as i32, by as i32, i32::from(tx));
        let result = crate::dsp::profil_olc!(
            KatsayiCozme,
            decode_coefs(
                msac,
                cdf,
                &env,
                tx,
                bs,
                y_mode_nofilt,
                b.uv_mode,
                plane,
                a,
                l,
                &mut self.scratch.cf,
                &mut self.scratch.levels,
            )
        );
        let res_ctx = result.res_ctx;
        let eob = result.eob;
        let txtp = result.txtp;
        if chroma {
            let pl = plane - 1;
            let above_idx = self.above_base + ai;
            fill_small(
                &mut self.blocks.above[above_idx].ccoef[pl],
                cbx4,
                ctw,
                res_ctx,
            );
            fill_small(&mut self.blocks.left.ccoef[pl], cby4, cth, res_ctx);
        } else {
            let above_idx = self.above_base + ai;
            fill_small(&mut self.blocks.above[above_idx].lcoef, bx4, ctw, res_ctx);
            fill_small(&mut self.blocks.left.lcoef, by4, cth, res_ctx);
        }
        if eob >= 0 {
            let Some(tx_size) = TxSize::from_index(usize::from(tx)) else {
                return Err(Error::InvalidData("geçersiz tx boyutu"));
            };
            let Some(tx_type) = TxType::from_index(usize::from(txtp)) else {
                return Err(Error::InvalidData("geçersiz tx türü"));
            };
            let itx = self.bdsp.itx().itxfm_add[tx_size.index()][tx_type.index()];
            if !trace_on && !probe_on {
                let stride = self.pic.stride[plane];
                let dst = &mut self.pic.data[plane];
                crate::dsp::profil_olc!(
                    ItxfmAdd,
                    itx(
                        &mut dst[dst_off..],
                        stride,
                        &mut self.scratch.cf,
                        eob as usize,
                        tx_size,
                        tx_type,
                        self.bd,
                    )
                );
                return Ok(());
            }
            let (px, py) = if chroma {
                ((bx >> self.ss_hor) * 4, (by >> self.ss_ver) * 4)
            } else {
                (bx * 4, by * 4)
            };
            let (pw, ph) = (ctw * 4, cth * 4);
            // Intra katsayı yolu da `T` satırı yazar (dav1d `itx` trambolini
            // bütün `itxfm_add` çağrılarını görür; Rust önceden yalnız inter
            // dalı izliyordu).
            if dsp_trace().on {
                let dim = &t::TXFM_DIMENSIONS[usize::from(tx)];
                let n = (usize::from(dim[0]) * 4).min(32) * (usize::from(dim[1]) * 4).min(32);
                itx_trace_line(
                    self.frame_no,
                    plane,
                    px,
                    py,
                    tx,
                    txtp,
                    eob,
                    if self.bd.bytes_per_sample() > 1 {
                        16
                    } else {
                        8
                    },
                    n,
                    self.bd.max_sample(),
                    &self.scratch.cf,
                );
            }
            // Yeniden oynatma probu: itx **öncesi** hedef ve katsayılar.
            let before = if probe::enabled() {
                Some(self.probe_gather(plane, px, py, pw, ph))
            } else {
                None
            };
            let coeffs = if probe::enabled() {
                Some(self.scratch.cf.clone())
            } else {
                None
            };
            let stride = self.pic.stride[plane];
            let dst = &mut self.pic.data[plane];
            crate::dsp::profil_olc!(
                ItxfmAdd,
                itx(
                    &mut dst[dst_off..],
                    stride,
                    &mut self.scratch.cf,
                    eob as usize,
                    tx_size,
                    tx_type,
                    self.bd,
                )
            );
            if let (Some(before), Some(coeffs)) = (before, coeffs) {
                let after = self.probe_gather(plane, px, py, pw, ph);
                let layout = match self.chroma {
                    ChromaFormat::C420 => 1,
                    ChromaFormat::C422 => 2,
                    ChromaFormat::C444 => 3,
                    ChromaFormat::Monochrome => 0,
                };
                probe::tx_record(
                    plane as u8,
                    self.bd.bits() as u8,
                    layout,
                    px as u32,
                    py as u32,
                    pw as u32,
                    ph as u32,
                    tx,
                    txtp,
                    eob,
                    ((dst_off % stride) / self.bd.bytes_per_sample()) as u32,
                    (dst_off / stride) as u32,
                    &before,
                    &coeffs,
                    &after,
                );
            }
        } else if probe_on {
            // Atlanan blok: itx yok, hedef = nihai piksel (eob = -1).
            let (px, py) = if chroma {
                ((bx >> self.ss_hor) * 4, (by >> self.ss_ver) * 4)
            } else {
                (bx * 4, by * 4)
            };
            let (pw, ph) = (ctw * 4, cth * 4);
            if pw > 0 && ph > 0 {
                let pixels = self.probe_gather(plane, px, py, pw, ph);
                let coeffs = vec![0i32; 1024];
                let layout = match self.chroma {
                    ChromaFormat::C420 => 1,
                    ChromaFormat::C422 => 2,
                    ChromaFormat::C444 => 3,
                    ChromaFormat::Monochrome => 0,
                };
                probe::tx_record(
                    plane as u8,
                    self.bd.bits() as u8,
                    layout,
                    px as u32,
                    py as u32,
                    pw as u32,
                    ph as u32,
                    tx,
                    txtp,
                    -1,
                    px as u32,
                    py as u32,
                    &pixels,
                    &coeffs,
                    &pixels,
                );
            }
        }
        Ok(())
    }

    /// `read_pal_plane` (`recon_tmpl.c:2172-2277`).
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn read_pal_plane(
        &mut self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        pl: usize,
        sz_ctx: usize,
        bx4: usize,
        by4: usize,
        b: &mut Av1Block,
    ) -> Result<()> {
        let ai = self.bx >> 5;
        let row = layout::CDF_m + layout::MODE_pal_sz + (pl * 7 + sz_ctx) * 8;
        let pal_sz = msac.decode_symbol_adapt(&mut cdf.cells_mut()[row..], 6) + 2;
        b.pal_sz[pl] = pal_sz as u8;
        let l_cache = if pl > 0 {
            usize::from(self.blocks.left_pal_uv_sz[by4])
        } else {
            usize::from(self.blocks.left.pal_sz[by4])
        };
        let a_cache = if by4 & 15 != 0 {
            if pl > 0 {
                usize::from(self.blocks.above_pal_uv_sz[self.above_base + ai][bx4])
            } else {
                usize::from(self.blocks.above[self.above_base + ai].pal_sz[bx4])
            }
        } else {
            0
        };
        let l = self.blocks.left_pal[by4][pl];
        let a = self.blocks.above_pal[self.above_base + ai][bx4][pl];
        let mut cache = [0u16; 16];
        let mut n_cache = 0usize;
        let mut li = 0usize;
        let mut ai_x = 0usize;
        while li < l_cache && ai_x < a_cache {
            if l[li] < a[ai_x] {
                if n_cache == 0 || cache[n_cache - 1] != l[li] {
                    cache[n_cache] = l[li];
                    n_cache += 1;
                }
                li += 1;
            } else {
                if a[ai_x] == l[li] {
                    li += 1;
                }
                if n_cache == 0 || cache[n_cache - 1] != a[ai_x] {
                    cache[n_cache] = a[ai_x];
                    n_cache += 1;
                }
                ai_x += 1;
            }
        }
        while li < l_cache {
            if n_cache == 0 || cache[n_cache - 1] != l[li] {
                cache[n_cache] = l[li];
                n_cache += 1;
            }
            li += 1;
        }
        while ai_x < a_cache {
            if n_cache == 0 || cache[n_cache - 1] != a[ai_x] {
                cache[n_cache] = a[ai_x];
                n_cache += 1;
            }
            ai_x += 1;
        }
        let mut used_cache = [0u16; 8];
        let mut n_used = 0usize;
        for n in 0..n_cache {
            if n_used < pal_sz && msac.decode_bool_equi() != 0 {
                used_cache[n_used] = cache[n];
                n_used += 1;
            }
        }
        let mut pal = [0u16; 8];
        if n_used < pal_sz {
            let bpc = self.bd.bits();
            let mut idx = n_used;
            let mut prev = msac.decode_bools(bpc) as i32;
            pal[idx] = prev as u16;
            idx += 1;
            if idx < pal_sz {
                let mut bits = bpc as i32 - 3 + msac.decode_bools(2) as i32;
                let max = (1i32 << bpc) - 1;
                while idx < pal_sz {
                    let delta = msac.decode_bools(bits as u32) as i32;
                    let bump = i32::from(pl == 0);
                    prev = (prev + delta + bump).min(max);
                    pal[idx] = prev as u16;
                    idx += 1;
                    if prev + bump >= max {
                        while idx < pal_sz {
                            pal[idx] = max as u16;
                            idx += 1;
                        }
                        break;
                    }
                    bits = bits.min(1 + ulog2((max - prev - bump) as u32) as i32);
                }
            }
            let new_entries = pal;
            let mut n = 0usize;
            let mut m = n_used;
            for i in 0..pal_sz {
                if n < n_used && (m >= pal_sz || used_cache[n] <= new_entries[m]) {
                    pal[i] = used_cache[n];
                    n += 1;
                } else {
                    pal[i] = new_entries[m];
                    m += 1;
                }
            }
        } else {
            pal[..n_used].copy_from_slice(&used_cache[..n_used]);
        }
        self.pal[pl] = pal;
        Ok(())
    }

    /// `read_pal_uv` (`recon_tmpl.c:2278-2310`).
    #[allow(clippy::needless_range_loop)]
    fn read_pal_uv(
        &mut self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        sz_ctx: usize,
        bx4: usize,
        by4: usize,
        b: &mut Av1Block,
    ) -> Result<()> {
        self.read_pal_plane(msac, cdf, 1, sz_ctx, bx4, by4, b)?;
        let bpc = self.bd.bits();
        let pal_sz = usize::from(b.pal_sz[1]);
        let mut v = [0u16; 8];
        if msac.decode_bool_equi() != 0 {
            let bits = bpc as i32 - 4 + msac.decode_bools(2) as i32;
            let mut prev = msac.decode_bools(bpc) as i32;
            v[0] = prev as u16;
            let max = (1i32 << bpc) - 1;
            for i in 1..pal_sz {
                let mut delta = msac.decode_bools(bits as u32) as i32;
                if delta != 0 && msac.decode_bool_equi() != 0 {
                    delta = -delta;
                }
                prev = (prev + delta) & max;
                v[i] = prev as u16;
            }
        } else {
            for i in 0..pal_sz {
                v[i] = msac.decode_bools(bpc) as u16;
            }
        }
        self.pal[2] = v;
        Ok(())
    }

    /// `read_pal_indices` (`decode.c:414-443`).
    #[allow(clippy::too_many_arguments)]
    fn read_pal_indices(
        &mut self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        pal_sz: usize,
        pl: usize,
        w4: usize,
        h4: usize,
        bw4: usize,
        bh4: usize,
    ) -> Result<()> {
        let stride = bw4 * 4;
        let first = msac.decode_uniform(pal_sz as u32) as u8;
        let mut pal_tmp = core::mem::take(&mut self.scratch.pal_tmp);
        pal_tmp[0] = first;
        let cm_base = layout::CDF_m + layout::MODE_color_map + ((pl * 7 + pal_sz - 2) * 5) * 8;
        let mut order = [[0u8; 8]; 64];
        let mut ctxs = [0u8; 64];
        for i in 1..4 * (w4 + h4) - 1 {
            let first = i.min(w4 * 4 - 1);
            let last = i.saturating_sub(h4 * 4 - 1);
            order_palette(&pal_tmp, stride, i, first, last, &mut order, &mut ctxs);
            let mut j = first;
            let mut m = 0usize;
            while j >= last {
                let row = cm_base + usize::from(ctxs[m]) * 8;
                let color_idx = msac.decode_symbol_adapt(&mut cdf.cells_mut()[row..], pal_sz - 1);
                let value = order[m][color_idx];
                pal_tmp[(i - j) * stride + j] = value;
                if j == 0 {
                    break;
                }
                j -= 1;
                m += 1;
            }
        }
        let pal_dsp = self.dsp.pal();
        let (bw, bh, w, h) = (bw4 * 4, bh4 * 4, w4 * 4, h4 * 4);
        if pl == 0 {
            let mut dst = core::mem::take(&mut self.scratch.pal_idx_y);
            crate::dsp::profil_olc!(
                PalIdxFinish,
                (pal_dsp.pal_idx_finish)(&mut dst, &pal_tmp, bw, bh, w, h)
            );
            self.scratch.pal_idx_y = dst;
        } else {
            let mut dst = core::mem::take(&mut self.scratch.pal_idx_uv);
            crate::dsp::profil_olc!(
                PalIdxFinish,
                (pal_dsp.pal_idx_finish)(&mut dst, &pal_tmp, bw, bh, w, h)
            );
            self.scratch.pal_idx_uv = dst;
        }
        self.scratch.pal_tmp = pal_tmp;
        Ok(())
    }
}

/// Intrabc + refmvs yolu — S2b.5.
///
/// ## Kaynak eşlemesi (dav1d @ `99f3354`, salt-okunur)
///
/// | Kural | dav1d satırı |
/// |---|---|
/// | intrabc bayrağı | `decode.c:1047-1048` |
/// | dv öngörüsü / varsayılan dv | `decode.c:1271-1291` |
/// | `read_mv_residual` (tamsayı hassasiyet) | `decode.c:78-120`, `:1293-1294` |
/// | dv kırpma (tile + SB) | `decode.c:1296-1348` |
/// | `read_vartx_tree` / `read_tx_tree` | `decode.c:120-200`, `:445-492`, `:1356` |
/// | `recon_b_inter` (bilinear MC) | `recon_tmpl.c:1557-1620`, `:1584-1593` |
/// | `mc()` (emu_edge + bilinear) | `recon_tmpl.c:938-1048` |
/// | `read_coef_tree` (inter) | `recon_tmpl.c:731-822` |
/// | `decode_coefs` inter tx türü | `recon_tmpl.c:321-412` |
/// | `splat_intrabc_mv` | `decode.c:530-543`, `:1366` |
impl<'a> Dec<'a> {
    /// `read_mv_component_diff` (`decode.c:69-111`) — `mv_prec = -1`
    /// (intrabc: tamsayı hassasiyet; `fp`/`hp` bitleri okunmaz).
    fn read_mv_component_diff(
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        comp_off: usize,
        mv_prec: i32,
    ) -> i32 {
        let base = crate::entropy::cdf::CDF_MV_OFFSET + comp_off;
        let sign = msac.decode_bool_adapt(&mut cdf.cells_mut()[base + layout::MVC_sign..]) as i32;
        let cl =
            msac.decode_symbol_adapt(&mut cdf.cells_mut()[base + layout::MVC_classes..], 10) as i32;
        let mut fp = 3i32;
        let mut hp = 1i32;
        let up: i32;
        if cl == 0 {
            up = msac.decode_bool_adapt(&mut cdf.cells_mut()[base + layout::MVC_class0..]) as i32;
            if mv_prec >= 0 {
                fp = msac.decode_symbol_adapt(
                    &mut cdf.cells_mut()[base + layout::MVC_class0_fp + up as usize * 4..],
                    3,
                ) as i32;
                if mv_prec > 0 {
                    hp = msac
                        .decode_bool_adapt(&mut cdf.cells_mut()[base + layout::MVC_class0_hp..])
                        as i32;
                }
            }
        } else {
            let mut u = 1i32 << cl;
            for n in 0..cl as usize {
                let bit = msac
                    .decode_bool_adapt(&mut cdf.cells_mut()[base + layout::MVC_classN + n * 2..])
                    as i32;
                u |= bit << n;
            }
            up = u;
            if mv_prec >= 0 {
                fp = msac
                    .decode_symbol_adapt(&mut cdf.cells_mut()[base + layout::MVC_classN_fp..], 3)
                    as i32;
                if mv_prec > 0 {
                    hp = msac
                        .decode_bool_adapt(&mut cdf.cells_mut()[base + layout::MVC_classN_hp..])
                        as i32;
                }
            }
        }
        let diff = ((up << 3) | (fp << 1) | hp) + 1;
        if sign != 0 { -diff } else { diff }
    }

    /// `read_mv_residual` (`decode.c:113-120`). Dönüş: `mv_joint` sembolü.
    fn read_mv_residual(
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        mv: &mut Mv,
        mv_prec: i32,
    ) -> i32 {
        let joint = msac.decode_symbol_adapt(
            &mut cdf.cells_mut()[crate::entropy::cdf::CDF_MV_OFFSET + layout::MV_joint..],
            3,
        ) as i32;
        // `enum MVJoint` (`src/levels.h:199-203`): 2 = V (y, `comp[0]`),
        // 1 = H (x, `comp[1]`). dav1d `decode.c:113-120` **önce** dikey
        // bileşeni okur; sembol sırası buna bağlıdır.
        if joint & 2 != 0 {
            let d = Self::read_mv_component_diff(msac, cdf, layout::MV_comp, mv_prec);
            mv.y = (i32::from(mv.y) + d) as i16;
        }
        if joint & 1 != 0 {
            let d = Self::read_mv_component_diff(msac, cdf, layout::MV_comp + 64, mv_prec);
            mv.x = (i32::from(mv.x) + d) as i16;
        }
        joint
    }

    /// Intrabc blok sözdizimi + yeniden yapılandırma (`decode.c:1271-1390`).
    fn decode_intrabc(
        &mut self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        bs: u8,
        edge_flags: u8,
        b: &mut Av1Block,
    ) -> Result<()> {
        if dsp_trace().frame_ok(self.frame_no) {
            block_trace_line(self.frame_no, self.bx, self.by, bs, b);
        }
        let b_dim = &t::BLOCK_DIMENSIONS[usize::from(bs)];
        let bw4 = usize::from(b_dim[0]);
        let bh4 = usize::from(b_dim[1]);
        let has_chroma = self.has_chroma
            && (bw4 > self.ss_hor || self.bx & 1 != 0)
            && (bh4 > self.ss_ver || self.by & 1 != 0);
        let tile = TileBounds {
            row_start: self.row_start,
            row_end: self.row_end,
            col_start: self.col_start,
            col_end: self.col_end,
        };
        let (stack, _ncnt, _nctx) =
            self.refmvs
                .find([0, -1], bs, edge_flags, self.by, self.bx, tile);
        let sb128 = u32::from(self.sb128);
        let mut mv = if stack[0].mv[0].n() != 0 {
            stack[0].mv[0]
        } else if stack[1].mv[0].n() != 0 {
            stack[1].mv[0]
        } else if (self.by as isize) - ((16usize << sb128) as isize) < self.row_start as isize {
            Mv {
                y: 0,
                x: (-((512i32 << sb128) + 2048)) as i16,
            }
        } else {
            Mv {
                y: (-(512i32 << sb128)) as i16,
                x: 0,
            }
        };
        let joint = Self::read_mv_residual(msac, cdf, &mut mv, -1);
        if joint != 0 {
            self.tools.intrabc_residual += 1;
        }

        // dv kırpma (`decode.c:1296-1348`)
        let mut border_left = self.col_start as i32 * 4;
        let mut border_top = self.row_start as i32 * 4;
        if has_chroma {
            if bw4 < 2 && self.ss_hor == 1 {
                border_left += 4;
            }
            if bh4 < 2 && self.ss_ver == 1 {
                border_top += 4;
            }
        }
        let bw4i = bw4 as i32;
        let bh4i = bh4 as i32;
        let mut src_left = self.bx as i32 * 4 + (i32::from(mv.x) >> 3);
        let mut src_top = self.by as i32 * 4 + (i32::from(mv.y) >> 3);
        let mut src_right = src_left + bw4i * 4;
        let mut src_bottom = src_top + bh4i * 4;
        let border_right = ((self.col_end as i32 + (bw4i - 1)) & !(bw4i - 1)).wrapping_mul(4);
        if src_left < border_left {
            src_right += border_left - src_left;
            src_left = border_left;
        } else if src_right > border_right {
            src_left -= src_right - border_right;
            src_right = border_right;
        }
        if src_top < border_top {
            src_bottom += border_top - src_top;
            src_top = border_top;
        }
        let sb_shift = 4 + sb128;
        let sbx = ((self.bx as i32) >> sb_shift) << (sb_shift + 2);
        let sby = ((self.by as i32) >> sb_shift) << (sb_shift + 2);
        let sb_size = 1i32 << (sb_shift + 2);
        if src_bottom > sby && src_right > sbx {
            if src_top - border_top >= src_bottom - sby {
                let d = src_bottom - sby;
                src_top -= d;
                src_bottom -= d;
            } else if src_left - border_left >= src_right - sbx {
                let d = src_right - sbx;
                src_left -= d;
                src_right -= d;
            }
        }
        if src_bottom > sby + sb_size {
            let d = src_bottom - (sby + sb_size);
            src_top -= d;
            src_bottom -= d;
        }
        if src_bottom > sby && src_right > sbx {
            return Err(Error::InvalidData(
                "intrabc dv geçerli superblock ile çakışıyor",
            ));
        }
        mv = Mv {
            y: ((src_top - self.by as i32 * 4) << 3) as i16,
            x: ((src_left - self.bx as i32 * 4) << 3) as i16,
        };
        b.mv[0] = mv;
        self.read_vartx_tree(msac, cdf, bs, b)?;
        self.recon_b_inter_intrabc(msac, cdf, bs, b)?;
        self.refmvs
            .splat_intrabc(self.bx, self.by, bw4, bh4, bs, mv);
        Ok(())
    }

    /// `read_tx_tree` (`decode.c:120-200`).
    #[allow(clippy::too_many_arguments)]
    fn read_tx_tree(
        &mut self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        bx: usize,
        by: usize,
        from: u8,
        depth: usize,
        masks: &mut [u16; 2],
        x_off: usize,
        y_off: usize,
    ) {
        let dim = &t::TXFM_DIMENSIONS[usize::from(from)];
        let (txw, txh) = (dim[2], dim[3]);
        let bx4 = bx & 31;
        let by4 = by & 31;
        let ai = bx >> 5;
        let mut is_split = false;
        if depth < 2 && from > TX_4X4 {
            let cat = 2 * (usize::from(TX_64X64) - usize::from(dim[5])) - depth;
            let a = usize::from(self.blocks.above[self.above_base + ai].tx[bx4] < txw as i8);
            let l = usize::from(self.blocks.left.tx[by4] < txh as i8);
            let row = layout::CDF_m + layout::MODE_txpart + (cat * 3 + a + l) * 2;
            is_split = msac.decode_bool_adapt(&mut cdf.cells_mut()[row..]) != 0;
            if is_split {
                masks[depth] |= 1 << (y_off * 4 + x_off);
            }
        }
        if is_split && dim[5] > 1 {
            let sub = dim[6];
            let sub_dim = &t::TXFM_DIMENSIONS[usize::from(sub)];
            let txsw = usize::from(sub_dim[0]);
            let txsh = usize::from(sub_dim[1]);
            self.read_tx_tree(
                msac,
                cdf,
                bx,
                by,
                sub,
                depth + 1,
                masks,
                x_off * 2,
                y_off * 2,
            );
            if txw >= txh && bx + txsw < self.bw {
                self.read_tx_tree(
                    msac,
                    cdf,
                    bx + txsw,
                    by,
                    sub,
                    depth + 1,
                    masks,
                    x_off * 2 + 1,
                    y_off * 2,
                );
            }
            if txh >= txw && by + txsh < self.bh {
                self.read_tx_tree(
                    msac,
                    cdf,
                    bx,
                    by + txsh,
                    sub,
                    depth + 1,
                    masks,
                    x_off * 2,
                    y_off * 2 + 1,
                );
                if txw >= txh && bx + txsw < self.bw {
                    self.read_tx_tree(
                        msac,
                        cdf,
                        bx + txsw,
                        by + txsh,
                        sub,
                        depth + 1,
                        masks,
                        x_off * 2 + 1,
                        y_off * 2 + 1,
                    );
                }
            }
        } else {
            let valw = if is_split { TX_4X4 } else { txw };
            let valh = if is_split { TX_4X4 } else { txh };
            let nw = 1usize << dim[2];
            let nh = 1usize << dim[3];
            let above_idx = self.above_base + ai;
            fill_small(&mut self.blocks.above[above_idx].tx, bx4, nw, valw as i8);
            fill_small(&mut self.blocks.left.tx, by4, nh, valh as i8);
        }
    }

    /// `read_vartx_tree` (`decode.c:445-492`).
    fn read_vartx_tree(
        &mut self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        bs: u8,
        b: &mut Av1Block,
    ) -> Result<()> {
        let b_dim = &t::BLOCK_DIMENSIONS[usize::from(bs)];
        let bw4 = usize::from(b_dim[0]);
        let bh4 = usize::from(b_dim[1]);
        let bx4 = self.bx & 31;
        let by4 = self.by & 31;
        let ai = self.bx >> 5;
        let layout_idx = match self.chroma {
            ChromaFormat::C420 => 1,
            ChromaFormat::C422 => 2,
            ChromaFormat::C444 => 3,
            ChromaFormat::Monochrome => 0,
        };
        b.max_ytx = t::MAX_TXFM_SIZE_FOR_BS[usize::from(bs)][0];
        let switchable = self.hdr.txfm_mode == TxfmMode::Switchable;
        if !b.skip && (self.hdr.segmentation.lossless[usize::from(b.seg_id)] || b.max_ytx == TX_4X4)
        {
            b.max_ytx = TX_4X4;
            b.uvtx = TX_4X4;
            if switchable {
                let nw = 1usize << b_dim[2];
                let nh = 1usize << b_dim[3];
                let above_idx = self.above_base + ai;
                fill_small(&mut self.blocks.above[above_idx].tx, bx4, nw, TX_4X4 as i8);
                fill_small(&mut self.blocks.left.tx, by4, nh, TX_4X4 as i8);
            }
        } else if !switchable || b.skip {
            if switchable {
                let nw = 1usize << b_dim[2];
                let nh = 1usize << b_dim[3];
                let above_idx = self.above_base + ai;
                fill_small(
                    &mut self.blocks.above[above_idx].tx,
                    bx4,
                    nw,
                    b_dim[2] as i8,
                );
                fill_small(&mut self.blocks.left.tx, by4, nh, b_dim[3] as i8);
            }
            b.uvtx = t::MAX_TXFM_SIZE_FOR_BS[usize::from(bs)][layout_idx];
        } else {
            let ytx = &t::TXFM_DIMENSIONS[usize::from(b.max_ytx)];
            let (step_w, step_h) = (usize::from(ytx[0]), usize::from(ytx[1]));
            let mut masks = [0u16; 2];
            let mut y = 0usize;
            let mut y_off = 0usize;
            while y < bh4 {
                let mut x = 0usize;
                let mut x_off = 0usize;
                while x < bw4 {
                    self.read_tx_tree(
                        msac,
                        cdf,
                        self.bx + x,
                        self.by + y,
                        b.max_ytx,
                        0,
                        &mut masks,
                        x_off,
                        y_off,
                    );
                    x += step_w;
                    x_off += 1;
                }
                y += step_h;
                y_off += 1;
            }
            b.tx_split = masks;
            b.uvtx = t::MAX_TXFM_SIZE_FOR_BS[usize::from(bs)][layout_idx];
        }
        Ok(())
    }

    /// `mc()` intrabc dalı (`recon_tmpl.c:938-1048`) — **her zaman**
    /// `emu_edge` üzerinden kopyalanır; iç bölgelerde piksel değerleri
    /// birebir aynıdır, böylece Rust dilim ödünç kuralları da korunur.
    #[allow(clippy::too_many_arguments)]
    fn intrabc_mc_plane(
        &mut self,
        plane: usize,
        bw4: usize,
        bh4: usize,
        bx: usize,
        by: usize,
        mv: Mv,
    ) {
        let ss_hor = if plane != 0 { self.ss_hor } else { 0 };
        let ss_ver = if plane != 0 { self.ss_ver } else { 0 };
        let h_mul = 4 >> ss_hor;
        let v_mul = 4 >> ss_ver;
        let mx = i32::from(mv.x) & (15 >> u32::from(ss_hor == 0));
        let my = i32::from(mv.y) & (15 >> u32::from(ss_ver == 0));
        let dx = bx as i32 * h_mul as i32 + (i32::from(mv.x) >> (3 + ss_hor));
        let dy = by as i32 * v_mul as i32 + (i32::from(mv.y) >> (3 + ss_ver));
        let w = (self.bw * 4) >> ss_hor;
        let h = (self.bh * 4) >> ss_ver;
        let pw = bw4 * h_mul;
        let ph = bh4 * v_mul;
        let mxn = usize::from(mx != 0);
        let myn = usize::from(my != 0);
        let _ = (mxn, myn);
        let bpp = self.bd.bytes_per_sample();
        let stride = self.pic.stride[plane];
        let bd = self.bd;
        let bmax = bd.max_sample() as u16;
        let mc = self.bdsp.mc();
        let dst_off = self.pic.off(plane, bx * h_mul, by * v_mul);
        let trace_on = dsp_trace().frame_ok(self.frame_no);
        // dav1d `mc()` intrabc dalı (`recon_tmpl.c:974-990`): `emu_edge` yalnız
        // kaynak pencere kare dışına taşarken çağrılır. Rust her zaman emu
        // kullanır (ödünç kuralları); iz bu yüzden **dav1d koşulunu** izler.
        let mxn = i32::from(mx != 0);
        let myn = i32::from(my != 0);
        let emu_c = dx < 3 * mxn
            || dy < 3 * myn
            || dx + pw as i32 + 4 * mxn > w as i32
            || dy + ph as i32 + 4 * myn > h as i32;
        let (emu_bw, emu_bh) = (pw + 7 * mxn as usize, ph + 7 * myn as usize);
        let (emu_x, emu_y) = (dx - 3 * mxn, dy - 3 * myn);
        let (emu_hoff, emu_voff) = ((3 - 3 * mxn) as usize, (3 - 3 * myn) as usize);
        let mut buf = core::mem::take(&mut self.scratch.emu_edge);
        // Rust MC sözleşmesi (`dsp/scalar/mc.rs:150-157`, `:589`): `src`
        // **dolgu hücresini** (blok başlangıcının −3 satır/−3 sütunu)
        // gösterir. `emu_edge` tamponunun (0,0) hücresi tam olarak
        // `(dx-3, dy-3)`'tür, yani dolgu hücresidir: `ref_off` **0** olmalı.
        // T33'te burada `(192*3+3)*bpp` kullanılıyordu ve çekirdek kendi
        // −3'ünü eklediği için kaynak **(+3,+3) kayıyordu** (ilk intrabc
        // bloğu `test10103` kare 0: dav1d kaynağı (25,0), Rust (28,3)).
        let xb = self.pic.x_base[plane];
        let yb = self.pic.y_base[plane];
        crate::dsp::profil_olc!(
            McEmuEdge,
            (mc.emu_edge)(
                pw + 7,
                ph + 7,
                w - xb,
                h - yb,
                dx - 3 - xb as i32,
                dy - 3 - yb as i32,
                &mut buf,
                192 * bpp,
                &self.pic.data[plane],
                stride,
                bd,
                bmax,
            )
        );
        let ref_off = 0;
        let src_stride = 192 * bpp;
        let mxx = (mx << u32::from(ss_hor == 0)) as u8;
        let myy = (my << u32::from(ss_ver == 0)) as u8;
        let dst = &mut self.pic.data[plane];
        crate::dsp::profil_olc!(
            McMc,
            (mc.mc[9])(
                &mut dst[dst_off..],
                stride,
                &buf[ref_off..],
                src_stride,
                pw,
                ph,
                mxx,
                myy,
                bd,
                bmax,
            )
        );
        if trace_on {
            // dav1d intrabc'de `mc[FILTER_2D_BILINEAR]` çağrısını `D` satırıyla
            // izler; kaynak `cur`/emu olduğu için yuva 255'tir.
            let hash = pred_hash(&self.pic.data[plane], dst_off, stride, pw, ph, bpp);
            mc_trace_line(
                self.frame_no,
                "mc",
                9,
                self.bx,
                self.by,
                pw,
                ph,
                i32::from(mxx),
                i32::from(myy),
                0,
                0,
                255,
                0,
                0,
                hash,
            );
            if emu_c {
                let off = emu_hoff * bpp + emu_voff * src_stride;
                let hash = pred_hash(&buf, off, src_stride, emu_bw, emu_bh, bpp);
                emu_trace_line(
                    self.frame_no,
                    self.bx,
                    self.by,
                    emu_bw,
                    emu_bh,
                    w,
                    h,
                    emu_x,
                    emu_y,
                    hash,
                );
            }
        }
        self.scratch.emu_edge = buf;
    }

    /// `dav1d_recon_b_inter` intrabc dalı (`recon_tmpl.c:1557-1620`,
    /// `:1960-2180`).
    fn recon_b_inter_intrabc(
        &mut self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        bs: u8,
        b: &mut Av1Block,
    ) -> Result<()> {
        let b_dim = &t::BLOCK_DIMENSIONS[usize::from(bs)];
        let bw4 = usize::from(b_dim[0]);
        let bh4 = usize::from(b_dim[1]);
        let ss_hor = self.ss_hor;
        let ss_ver = self.ss_ver;
        let w4 = bw4.min(self.bw - self.bx);
        let h4 = bh4.min(self.bh - self.by);
        let cw4 = (w4 + ss_hor) >> ss_hor;
        let ch4 = (h4 + ss_ver) >> ss_ver;
        let _ = (cw4, ch4);
        let has_chroma = self.has_chroma
            && (bw4 > ss_hor || self.bx & 1 != 0)
            && (bh4 > ss_ver || self.by & 1 != 0);
        let mv = b.mv[0];

        // --- tahmin -------------------------------------------------------
        self.intrabc_mc_plane(0, bw4, bh4, self.bx, self.by, mv);
        if has_chroma {
            let cbw4 = bw4 << usize::from(bw4 == ss_hor);
            let cbh4 = bh4 << usize::from(bh4 == ss_ver);
            for pl in 1..3 {
                self.intrabc_mc_plane(pl, cbw4, cbh4, self.bx & !ss_hor, self.by & !ss_ver, mv);
            }
        }
        self.recon_b_inter_coefs(msac, cdf, bs, b)
    }

    /// Inter blok katsayıları + itx (`recon_tmpl.c:1880-2040`).
    fn recon_b_inter_coefs(
        &mut self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        bs: u8,
        b: &Av1Block,
    ) -> Result<()> {
        let b_dim = &t::BLOCK_DIMENSIONS[usize::from(bs)];
        let bw4 = usize::from(b_dim[0]);
        let bh4 = usize::from(b_dim[1]);
        let ss_hor = self.ss_hor;
        let ss_ver = self.ss_ver;
        let w4 = bw4.min(self.bw - self.bx);
        let h4 = bh4.min(self.bh - self.by);
        let cw4 = (w4 + ss_hor) >> ss_hor;
        let ch4 = (h4 + ss_ver) >> ss_ver;
        let _ = (cw4, ch4);
        let has_chroma = self.has_chroma
            && (bw4 > ss_hor || self.bx & 1 != 0)
            && (bh4 > ss_ver || self.by & 1 != 0);
        if b.skip {
            let bx4 = self.bx & 31;
            let by4 = self.by & 31;
            let ai = self.bx >> 5;
            let nw = 1usize << b_dim[2];
            let nh = 1usize << b_dim[3];
            let above_idx = self.above_base + ai;
            fill_small(&mut self.blocks.above[above_idx].lcoef, bx4, nw, 0x40);
            fill_small(&mut self.blocks.left.lcoef, by4, nh, 0x40);
            if has_chroma {
                let cbx4 = bx4 >> ss_hor;
                let cby4 = by4 >> ss_ver;
                let cbw4 = (bw4 + ss_hor) >> ss_hor;
                let cbh4 = (bh4 + ss_ver) >> ss_ver;
                for pl in 0..2 {
                    fill_small(
                        &mut self.blocks.above[above_idx].ccoef[pl],
                        cbx4,
                        cbw4,
                        0x40,
                    );
                    fill_small(&mut self.blocks.left.ccoef[pl], cby4, cbh4, 0x40);
                }
            }
            return Ok(());
        }

        let ytx = t::TXFM_DIMENSIONS[usize::from(b.max_ytx)];
        let uvtx = t::TXFM_DIMENSIONS[usize::from(b.uvtx)];
        let (ytx_w, ytx_h) = (usize::from(ytx[0]), usize::from(ytx[1]));
        let (uvtx_w, uvtx_h) = (usize::from(uvtx[0]), usize::from(uvtx[1]));
        let tx_split = b.tx_split;
        let bx0 = self.bx;
        let by0 = self.by;

        // `txtp_map` yalnız kroma varken okunur (`recon_tmpl.c:795-801`);
        // her `[init_y..sub_h4, init_x..sub_w4]` alt-bloğunda luma `read_coef_tree`
        // ilgili hücreleri kroma döngüsünden hemen önce eksiksiz yazar.

        let mut init_y = 0usize;
        while init_y < bh4 {
            let sub_h4 = h4.min(init_y + 16);
            let mut init_x = 0usize;
            while init_x < bw4 {
                let sub_w4 = w4.min(init_x + 16);
                // --- luma katsayıları -------------------------------------
                let mut y_off = usize::from(init_y != 0);
                let mut y = init_y;
                while y < sub_h4 {
                    let mut x_off = usize::from(init_x != 0);
                    let mut x = init_x;
                    while x < sub_w4 {
                        let dst_off = self.pic.off(0, (bx0 + x) * 4, (by0 + y) * 4);
                        self.read_coef_tree(
                            msac,
                            cdf,
                            bs,
                            b,
                            b.max_ytx,
                            0,
                            tx_split,
                            x_off,
                            y_off,
                            bx0 + x,
                            by0 + y,
                            dst_off,
                        )?;
                        x += ytx_w;
                        x_off += 1;
                    }
                    y += ytx_h;
                    y_off += 1;
                }
                // --- kroma katsayıları ------------------------------------
                if has_chroma {
                    for pl in 0..2 {
                        let plane = 1 + pl;
                        let mut y = init_y >> ss_ver;
                        let yend = ch4.min((init_y + 16) >> ss_ver);
                        while y < yend {
                            let mut x = init_x >> ss_hor;
                            let xend = cw4.min((init_x + 16) >> ss_hor);
                            while x < xend {
                                let luma_x = bx0 + (x << ss_hor);
                                let luma_y = by0 + (y << ss_ver);
                                let hint =
                                    self.scratch.txtp_map[(luma_y & 31) * 32 + (luma_x & 31)];
                                let dst_off = self.pic.off(
                                    plane,
                                    ((bx0 >> ss_hor) + x) * 4,
                                    ((by0 >> ss_ver) + y) * 4,
                                );
                                self.decode_inter_leaf(
                                    msac,
                                    cdf,
                                    bs,
                                    b,
                                    plane,
                                    b.uvtx,
                                    dst_off,
                                    bx0 + (x << ss_hor),
                                    by0 + (y << ss_ver),
                                    hint,
                                )?;
                                x += uvtx_w;
                            }
                            y += uvtx_h;
                        }
                    }
                }
                init_x += 16;
            }
            init_y += 16;
        }
        Ok(())
    }

    /// `read_coef_tree` (`recon_tmpl.c:731-822`) — inter (intrabc) yol.
    #[allow(clippy::too_many_arguments)]
    fn read_coef_tree(
        &mut self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        bs: u8,
        b: &Av1Block,
        ytx: u8,
        depth: usize,
        tx_split: [u16; 2],
        x_off: usize,
        y_off: usize,
        bx: usize,
        by: usize,
        dst_off: usize,
    ) -> Result<()> {
        let dim = &t::TXFM_DIMENSIONS[usize::from(ytx)];
        let (txw, txh) = (dim[2], dim[3]);
        // dav1d `recon_tmpl.c:743-747`: kaydırma yalnız `tx_split[depth]`
        // sıfırdan farklıyken değerlendirilir; `y_off` 3'ü aşabilir
        // (kayıpsız bloklarda TX_4X4 bölünemez). Bit 0..5 dışı daima 0'dır.
        let bit = y_off * 4 + x_off;
        if depth < 2 && bit < 16 && tx_split[depth] & (1 << bit) != 0 {
            let sub = dim[6];
            let sub_dim = &t::TXFM_DIMENSIONS[usize::from(sub)];
            let txsw = usize::from(sub_dim[0]);
            let txsh = usize::from(sub_dim[1]);
            self.read_coef_tree(
                msac,
                cdf,
                bs,
                b,
                sub,
                depth + 1,
                tx_split,
                x_off * 2,
                y_off * 2,
                bx,
                by,
                dst_off,
            )?;
            if txw >= txh && bx + txsw < self.bw {
                self.read_coef_tree(
                    msac,
                    cdf,
                    bs,
                    b,
                    sub,
                    depth + 1,
                    tx_split,
                    x_off * 2 + 1,
                    y_off * 2,
                    bx + txsw,
                    by,
                    dst_off + txsw * 4 * self.bd.bytes_per_sample(),
                )?;
            }
            if txh >= txw && by + txsh < self.bh {
                self.read_coef_tree(
                    msac,
                    cdf,
                    bs,
                    b,
                    sub,
                    depth + 1,
                    tx_split,
                    x_off * 2,
                    y_off * 2 + 1,
                    bx,
                    by + txsh,
                    dst_off + txsh * 4 * self.pic.stride[0],
                )?;
                if txw >= txh && bx + txsw < self.bw {
                    self.read_coef_tree(
                        msac,
                        cdf,
                        bs,
                        b,
                        sub,
                        depth + 1,
                        tx_split,
                        x_off * 2 + 1,
                        y_off * 2 + 1,
                        bx + txsw,
                        by + txsh,
                        dst_off
                            + txsw * 4 * self.bd.bytes_per_sample()
                            + txsh * 4 * self.pic.stride[0],
                    )?;
                }
            }
            return Ok(());
        }
        let (_eob, txtp) = self.decode_inter_leaf(msac, cdf, bs, b, 0, ytx, dst_off, bx, by, 0)?;
        // `txtp_map` (dav1d `recon_tmpl.c:795-801`): dav1d bunu `eob`'dan
        // **bağımsız** doldurur — `decode_coefs` atlanan blokta da `txtp`
        // yazar (`lossless ? WHT_WHT : DCT_DCT`). Kroma `txtp` ipucu bu
        // haritadan okunur; eob'a koşullamak 4:2:0/4:2:2 intrabc
        // karelerinde kroma tx türünü kaydırır (T33).
        if self.has_chroma {
            let bx4 = bx & 31;
            let by4 = by & 31;
            let tw = 1usize << dim[2];
            let th = 1usize << dim[3];
            if tw == 1 && th == 1 {
                self.scratch.txtp_map[by4 * 32 + bx4] = txtp;
            } else {
                for row in 0..th {
                    let base = (by4 + row) * 32 + bx4;
                    self.scratch.txtp_map[base..base + tw].fill(txtp);
                }
            }
        }
        Ok(())
    }

    /// Inter katsayı bloğu çözümü + itx (`decode_coefs` inter dalı).
    #[allow(clippy::too_many_arguments)]
    fn decode_inter_leaf(
        &mut self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        bs: u8,
        b: &Av1Block,
        plane: usize,
        tx: u8,
        dst_off: usize,
        bx: usize,
        by: usize,
        txtp_hint: u8,
    ) -> Result<(i32, u8)> {
        let bx4 = bx & 31;
        let by4 = by & 31;
        let ai = bx >> 5;
        let chroma = plane != 0;
        let cbx4 = chroma_ctx4(bx4, self.ss_hor);
        let cby4 = chroma_ctx4(by4, self.ss_ver);
        let t_dim = &t::TXFM_DIMENSIONS[usize::from(tx)];
        let txw = usize::from(t_dim[0]);
        let txh = usize::from(t_dim[1]);
        let seg_id = usize::from(b.seg_id);
        let env = CoefEnv {
            lossless: self.hdr.segmentation.lossless[seg_id],
            reduced_txtp: self.hdr.reduced_txtp_set,
            seg_qidx: self.hdr.segmentation.qidx[seg_id],
            bit_depth: self.bd,
            ss_ver: self.ss_ver != 0,
            ss_hor: self.ss_hor != 0,
            dq: &self.dq[seg_id],
            qm: if self.hdr.quant.qm {
                Some(crate::decode::coef::QmSel {
                    level: match plane {
                        0 => self.hdr.quant.qm_y,
                        1 => self.hdr.quant.qm_u,
                        _ => self.hdr.quant.qm_v,
                    },
                    class: usize::from(chroma),
                    tx,
                })
            } else {
                None
            },
            inter: true,
            txtp_hint,
        };
        let (a, l): (&[u8], &[u8]) = if chroma {
            let pl = plane - 1;
            (
                &self.blocks.above[self.above_base + ai].ccoef[pl][cbx4..],
                &self.blocks.left.ccoef[pl][cby4..],
            )
        } else {
            (
                &self.blocks.above[self.above_base + ai].lcoef[bx4..],
                &self.blocks.left.lcoef[by4..],
            )
        };
        let result = crate::dsp::profil_olc!(
            KatsayiCozme,
            decode_coefs(
                msac,
                cdf,
                &env,
                tx,
                bs,
                b.y_mode,
                b.uv_mode,
                plane,
                a,
                l,
                &mut self.scratch.cf,
                &mut self.scratch.levels,
            )
        );
        let (ctw, cth) = if chroma {
            (
                txw.min((self.bw - bx + self.ss_hor) >> self.ss_hor),
                txh.min((self.bh - by + self.ss_ver) >> self.ss_ver),
            )
        } else {
            (txw.min(self.bw - bx), txh.min(self.bh - by))
        };
        let above_idx = self.above_base + ai;
        if chroma {
            let pl = plane - 1;
            fill_small(
                &mut self.blocks.above[above_idx].ccoef[pl],
                cbx4,
                ctw,
                result.res_ctx,
            );
            fill_small(&mut self.blocks.left.ccoef[pl], cby4, cth, result.res_ctx);
        } else {
            fill_small(
                &mut self.blocks.above[above_idx].lcoef,
                bx4,
                ctw,
                result.res_ctx,
            );
            fill_small(&mut self.blocks.left.lcoef, by4, cth, result.res_ctx);
        }
        if result.eob >= 0 {
            let Some(tx_size) = TxSize::from_index(usize::from(tx)) else {
                return Err(Error::InvalidData("geçersiz tx boyutu"));
            };
            let Some(tx_type) = TxType::from_index(usize::from(result.txtp)) else {
                return Err(Error::InvalidData("geçersiz tx türü"));
            };
            let itx = self.bdsp.itx().itxfm_add[tx_size.index()][tx_type.index()];
            let stride = self.pic.stride[plane];
            if !dsp_trace().on && !probe::enabled() {
                let dst = &mut self.pic.data[plane];
                crate::dsp::profil_olc!(
                    ItxfmAdd,
                    itx(
                        &mut dst[dst_off..],
                        stride,
                        &mut self.scratch.cf,
                        result.eob as usize,
                        tx_size,
                        tx_type,
                        self.bd,
                    )
                );
                return Ok((result.eob, result.txtp));
            }
            // Yeniden oynatma probu (intrabc/intra yol): itx öncesi hedef ve
            // katsayılar; çağrı sonrası hedef. Tanılama; varsayılan kapalı.
            let (px, py) = if chroma {
                ((bx >> self.ss_hor) * 4, (by >> self.ss_ver) * 4)
            } else {
                (bx * 4, by * 4)
            };
            if dsp_trace().on {
                let dim = &t::TXFM_DIMENSIONS[usize::from(tx)];
                let n = (usize::from(dim[0]) * 4).min(32) * (usize::from(dim[1]) * 4).min(32);
                itx_trace_line(
                    self.frame_no,
                    plane,
                    px,
                    py,
                    tx,
                    result.txtp,
                    result.eob,
                    if self.bd.bytes_per_sample() > 1 {
                        16
                    } else {
                        8
                    },
                    n,
                    self.bd.max_sample(),
                    &self.scratch.cf,
                );
            }
            let (pw, ph) = (ctw * 4, cth * 4);
            let before = if probe::enabled() {
                Some(self.probe_gather(plane, px, py, pw, ph))
            } else {
                None
            };
            let coeffs = if probe::enabled() {
                Some(self.scratch.cf.clone())
            } else {
                None
            };
            let dst = &mut self.pic.data[plane];
            crate::dsp::profil_olc!(
                ItxfmAdd,
                itx(
                    &mut dst[dst_off..],
                    stride,
                    &mut self.scratch.cf,
                    result.eob as usize,
                    tx_size,
                    tx_type,
                    self.bd,
                )
            );
            if let (Some(before), Some(coeffs)) = (before, coeffs) {
                let after = self.probe_gather(plane, px, py, pw, ph);
                let layout = match self.chroma {
                    ChromaFormat::C420 => 1,
                    ChromaFormat::C422 => 2,
                    ChromaFormat::C444 => 3,
                    ChromaFormat::Monochrome => 0,
                };
                probe::tx_record(
                    plane as u8,
                    self.bd.bits() as u8,
                    layout,
                    px as u32,
                    py as u32,
                    pw as u32,
                    ph as u32,
                    tx,
                    result.txtp,
                    result.eob,
                    ((dst_off % stride) / self.bd.bytes_per_sample()) as u32,
                    (dst_off / stride) as u32,
                    &before,
                    &coeffs,
                    &after,
                );
            }
        }
        Ok((result.eob, result.txtp))
    }
}

/// Tek referanslı inter blok çözümü — dav1d `src/decode.c` 1390-2110 ve
/// `src/recon_tmpl.c` 1557-2040.
///
/// Kapsam: tek referans, çeviri hareketi (global motion yalnız öteleme),
/// NEWMV/NEARESTMV/NEARMV/GLOBALMV, uzamsal + temporal `refmvs` adayları,
/// interp filtresi, inter tx bölmesi, segment referans/skip/globalmv
/// özellikleri. P17 araçları (compound, skip mode, interintra, OBMC, yerel
/// warp, çeviri dışı global motion, ölçekli referans) **sembol okunduktan
/// hemen sonra** [`Error::Unsupported`] döner.
impl Dec<'_> {
    /// `get_intra_ctx` (`env.h:59-73`).
    fn get_intra_ctx(&self, bx4: usize, by4: usize, have_top: bool, have_left: bool) -> usize {
        let a = self.blocks.above[self.above_base + (self.bx >> 5)].intra[bx4];
        let l = self.blocks.left.intra[by4];
        match (have_left, have_top) {
            (true, true) => {
                let ctx = usize::from(l) + usize::from(a);
                ctx + usize::from(ctx == 2)
            }
            (true, false) => usize::from(l) * 2,
            (false, true) => usize::from(a) * 2,
            (false, false) => 0,
        }
    }

    /// `get_prev_frame_segid` (`decode.c:489-505`).
    fn prev_frame_segid(&self, by: usize, bx: usize, w4: usize, h4: usize) -> u8 {
        let Some(map) = self.prev_segmap.as_ref() else {
            return 0;
        };
        prev_segid_in_map(map, self.frame_segmap_stride, by, bx, w4, h4)
    }

    /// Komşu referans kayıtları üzerinde dolaşır (`env.h` inter bağlamları).
    fn for_each_neighbor_ref(
        &self,
        bx4: usize,
        by4: usize,
        have_top: bool,
        have_left: bool,
        mut f: impl FnMut(i8),
    ) {
        let a = &self.blocks.above[self.above_base + (self.bx >> 5)];
        let l = &self.blocks.left;
        if have_top && a.intra[bx4] == 0 {
            f(a.ref_pair[0][bx4]);
            if a.comp_type[bx4] != 0 {
                f(a.ref_pair[1][bx4]);
            }
        }
        if have_left && l.intra[by4] == 0 {
            f(l.ref_pair[0][by4]);
            if l.comp_type[by4] != 0 {
                f(l.ref_pair[1][by4]);
            }
        }
    }

    /// `av1_get_ref_ctx` (`env.h:283-305`).
    fn av1_get_ref_ctx(&self, bx4: usize, by4: usize, have_top: bool, have_left: bool) -> usize {
        let mut cnt = [0i32; 2];
        self.for_each_neighbor_ref(bx4, by4, have_top, have_left, |r| {
            cnt[usize::from(r >= 4)] += 1;
        });
        if cnt[0] == cnt[1] {
            1
        } else if cnt[0] < cnt[1] {
            0
        } else {
            2
        }
    }

    /// `av1_get_fwd_ref_ctx` (`env.h:307-333`).
    fn av1_get_fwd_ref_ctx(
        &self,
        bx4: usize,
        by4: usize,
        have_top: bool,
        have_left: bool,
    ) -> usize {
        let mut cnt = [0i32; 4];
        self.for_each_neighbor_ref(bx4, by4, have_top, have_left, |r| {
            if (0..4).contains(&r) {
                cnt[usize::from(r as u8)] += 1;
            }
        });
        cnt[0] += cnt[1];
        cnt[2] += cnt[3];
        if cnt[0] == cnt[2] {
            1
        } else if cnt[0] < cnt[2] {
            0
        } else {
            2
        }
    }

    /// `av1_get_fwd_ref_1_ctx` (`env.h:335-359`).
    fn av1_get_fwd_ref_1_ctx(
        &self,
        bx4: usize,
        by4: usize,
        have_top: bool,
        have_left: bool,
    ) -> usize {
        let mut cnt = [0i32; 2];
        self.for_each_neighbor_ref(bx4, by4, have_top, have_left, |r| {
            if (0..2).contains(&r) {
                cnt[usize::from(r as u8)] += 1;
            }
        });
        if cnt[0] == cnt[1] {
            1
        } else if cnt[0] < cnt[1] {
            0
        } else {
            2
        }
    }

    /// `av1_get_fwd_ref_2_ctx` (`env.h:361-385`).
    fn av1_get_fwd_ref_2_ctx(
        &self,
        bx4: usize,
        by4: usize,
        have_top: bool,
        have_left: bool,
    ) -> usize {
        let mut cnt = [0i32; 2];
        self.for_each_neighbor_ref(bx4, by4, have_top, have_left, |r| {
            if r == 2 || r == 3 {
                cnt[usize::from((r - 2) as u8)] += 1;
            }
        });
        if cnt[0] == cnt[1] {
            1
        } else if cnt[0] < cnt[1] {
            0
        } else {
            2
        }
    }

    /// `av1_get_bwd_ref_ctx` (`env.h:387-411`).
    fn av1_get_bwd_ref_ctx(
        &self,
        bx4: usize,
        by4: usize,
        have_top: bool,
        have_left: bool,
    ) -> usize {
        let mut cnt = [0i32; 3];
        self.for_each_neighbor_ref(bx4, by4, have_top, have_left, |r| {
            if r >= 4 {
                cnt[usize::from((r - 4) as u8)] += 1;
            }
        });
        cnt[1] += cnt[0];
        if cnt[2] == cnt[1] {
            1
        } else if cnt[1] < cnt[2] {
            0
        } else {
            2
        }
    }

    /// `av1_get_bwd_ref_1_ctx` (`env.h:413-437`).
    fn av1_get_bwd_ref_1_ctx(
        &self,
        bx4: usize,
        by4: usize,
        have_top: bool,
        have_left: bool,
    ) -> usize {
        let mut cnt = [0i32; 3];
        self.for_each_neighbor_ref(bx4, by4, have_top, have_left, |r| {
            if r >= 4 {
                cnt[usize::from((r - 4) as u8)] += 1;
            }
        });
        if cnt[0] == cnt[1] {
            1
        } else if cnt[0] < cnt[1] {
            0
        } else {
            2
        }
    }

    /// `get_comp_ctx` (`env.h:156-184`).
    fn get_comp_ctx(&self, bx4: usize, by4: usize, have_top: bool, have_left: bool) -> usize {
        let a = &self.blocks.above[self.above_base + (self.bx >> 5)];
        let l = &self.blocks.left;
        if have_top {
            if have_left {
                if a.comp_type[bx4] != 0 {
                    if l.comp_type[by4] != 0 {
                        return 4;
                    }
                    return 2 + usize::from((l.ref_pair[0][by4] as u8) >= 4);
                } else if l.comp_type[by4] != 0 {
                    return 2 + usize::from((a.ref_pair[0][bx4] as u8) >= 4);
                }
                // dav1d `env.h:170`: burada karşılaştırma **işaretli**dir.
                let ab = usize::from(a.ref_pair[0][bx4] >= 4);
                let lb = usize::from(l.ref_pair[0][by4] >= 4);
                usize::from(ab != lb)
            } else if a.comp_type[bx4] != 0 {
                3
            } else {
                // dav1d `env.h:174`: işaretli karşılaştırma.
                usize::from(a.ref_pair[0][bx4] >= 4)
            }
        } else if have_left {
            if l.comp_type[by4] != 0 {
                3
            } else {
                // dav1d `env.h:177`: işaretli karşılaştırma.
                usize::from(l.ref_pair[0][by4] >= 4)
            }
        } else {
            1
        }
    }

    /// `has_uni_comp` (`env.h:190-193`) — kenar bloğu tek yönlü mü?
    fn has_uni_comp(&self, above: bool, off: usize) -> bool {
        let e = if above {
            &self.blocks.above[self.above_base + (self.bx >> 5)]
        } else {
            &self.blocks.left
        };
        // dav1d `env.h:190-193`: `ref` **int8_t**; karşılaştırma işaretli.
        // (intra -1 `< 4` doğrudur.)
        (e.ref_pair[0][off] < 4) == (e.ref_pair[1][off] < 4)
    }

    /// `get_comp_dir_ctx` (`env.h:186-238`).
    fn get_comp_dir_ctx(&self, bx4: usize, by4: usize, have_top: bool, have_left: bool) -> usize {
        let a = &self.blocks.above[self.above_base + (self.bx >> 5)];
        let l = &self.blocks.left;
        if have_top && have_left {
            let a_intra = a.intra[bx4] != 0;
            let l_intra = l.intra[by4] != 0;
            if a_intra && l_intra {
                return 2;
            }
            if a_intra || l_intra {
                let (edge, off) = if a_intra { (l, by4) } else { (a, bx4) };
                if edge.comp_type[off] == COMP_INTER_NONE {
                    return 2;
                }
                return 1 + 2 * usize::from(self.has_uni_comp(!a_intra, off));
            }
            let a_comp = a.comp_type[bx4] != COMP_INTER_NONE;
            let l_comp = l.comp_type[by4] != COMP_INTER_NONE;
            let a_ref0 = a.ref_pair[0][bx4];
            let l_ref0 = l.ref_pair[0][by4];
            if !a_comp && !l_comp {
                return 1 + 2 * usize::from((a_ref0 >= 4) == (l_ref0 >= 4));
            }
            if !a_comp || !l_comp {
                let (edge, off, is_a) = if a_comp {
                    (a, bx4, true)
                } else {
                    (l, by4, false)
                };
                let _ = edge;
                if !self.has_uni_comp(is_a, off) {
                    return 1;
                }
                return 3 + usize::from((a_ref0 >= 4) == (l_ref0 >= 4));
            }
            let a_uni = self.has_uni_comp(true, bx4);
            let l_uni = self.has_uni_comp(false, by4);
            if !a_uni && !l_uni {
                return 0;
            }
            if !a_uni || !l_uni {
                return 2;
            }
            return 3 + usize::from((a_ref0 == 4) == (l_ref0 == 4));
        }
        if have_top || have_left {
            let (edge, off, is_a) = if have_left {
                (l, by4, false)
            } else {
                (a, bx4, true)
            };
            let _ = edge;
            if if is_a {
                a.intra[off] != 0
            } else {
                l.intra[off] != 0
            } {
                return 2;
            }
            if if is_a {
                a.comp_type[off] == COMP_INTER_NONE
            } else {
                l.comp_type[off] == COMP_INTER_NONE
            } {
                return 2;
            }
            return 4 * usize::from(self.has_uni_comp(is_a, off));
        }
        2
    }

    /// `av1_get_uni_p1_ctx` (`env.h:413-437`).
    fn av1_get_uni_p1_ctx(&self, bx4: usize, by4: usize, have_top: bool, have_left: bool) -> usize {
        let mut cnt = [0i32; 3];
        self.for_each_neighbor_ref(bx4, by4, have_top, have_left, |r| {
            if (1..4).contains(&r) {
                cnt[usize::from((r - 1) as u8)] += 1;
            }
        });
        cnt[1] += cnt[2];
        if cnt[0] == cnt[1] {
            1
        } else if cnt[0] < cnt[1] {
            0
        } else {
            2
        }
    }

    /// `get_jnt_comp_ctx` (`env.h:240-256`).
    fn get_jnt_comp_ctx(&self, bx4: usize, by4: usize, ref0: i8, ref1: i8) -> usize {
        let cur = i32::from(self.hdr.frame_offset);
        let ref0poc = i32::from(self.refpoc[usize::from(ref0 as u8)]);
        let ref1poc = i32::from(self.refpoc[usize::from(ref1 as u8)]);
        let d0 =
            crate::bitstream::frame_header::get_poc_diff(self.seq.order_hint_n_bits, ref0poc, cur)
                .abs();
        let d1 =
            crate::bitstream::frame_header::get_poc_diff(self.seq.order_hint_n_bits, cur, ref1poc)
                .abs();
        let offset = usize::from(d0 == d1);
        let a = &self.blocks.above[self.above_base + (self.bx >> 5)];
        let l = &self.blocks.left;
        let a_ctx = usize::from(a.comp_type[bx4] >= COMP_INTER_AVG || a.ref_pair[0][bx4] == 6);
        let l_ctx = usize::from(l.comp_type[by4] >= COMP_INTER_AVG || l.ref_pair[0][by4] == 6);
        3 * offset + a_ctx + l_ctx
    }

    /// `get_mask_comp_ctx` (`env.h:258-268`).
    fn get_mask_comp_ctx(&self, bx4: usize, by4: usize) -> usize {
        let a = &self.blocks.above[self.above_base + (self.bx >> 5)];
        let l = &self.blocks.left;
        let a_ctx = if a.comp_type[bx4] >= COMP_INTER_SEG {
            1
        } else if a.ref_pair[0][bx4] == 6 {
            3
        } else {
            0
        };
        let l_ctx = if l.comp_type[by4] >= COMP_INTER_SEG {
            1
        } else if l.ref_pair[0][by4] == 6 {
            3
        } else {
            0
        };
        (a_ctx + l_ctx).min(5)
    }

    /// `get_filter_ctx` (`env.h:135-154`).
    fn get_filter_ctx(
        &self,
        bx4: usize,
        by4: usize,
        comp: usize,
        dir: usize,
        reference: i8,
    ) -> usize {
        let a = &self.blocks.above[self.above_base + (self.bx >> 5)];
        let l = &self.blocks.left;
        let a_filter = if a.ref_pair[0][bx4] == reference || a.ref_pair[1][bx4] == reference {
            a.filter[dir][bx4]
        } else {
            N_SWITCHABLE_FILTERS
        };
        let l_filter = if l.ref_pair[0][by4] == reference || l.ref_pair[1][by4] == reference {
            l.filter[dir][by4]
        } else {
            N_SWITCHABLE_FILTERS
        };
        if a_filter == l_filter {
            comp * 4 + usize::from(a_filter)
        } else if a_filter == N_SWITCHABLE_FILTERS {
            comp * 4 + usize::from(l_filter)
        } else if l_filter == N_SWITCHABLE_FILTERS {
            comp * 4 + usize::from(a_filter)
        } else {
            comp * 4 + usize::from(N_SWITCHABLE_FILTERS)
        }
    }

    /// `find_matching_ref` (`decode.c:191-262`) — OBMC/warp aday maskesi.
    #[allow(clippy::too_many_arguments)]
    fn find_matching_ref(
        &self,
        edge_flags: u8,
        bw4: usize,
        bh4: usize,
        w4: usize,
        h4: usize,
        have_left: bool,
        have_top: bool,
        reference: i8,
    ) -> [u64; 2] {
        let mut masks = [0u64; 2];
        let mut count = 0usize;
        let matches = |b: crate::decode::refmvs::RefBlock| {
            b.ref_pair[0] == reference + 1 && b.ref_pair[1] == -1
        };
        let mut have_topleft = have_top && have_left;
        let mut have_topright = bw4.max(bh4) < 32
            && have_top
            && self.bx + bw4 < self.col_end
            && edge_flags & EDGE_I444_TOP_HAS_RIGHT != 0;
        if have_top {
            let mut b = self.refmvs.block_at(self.bx, self.by - 1);
            if matches(b) {
                masks[0] |= 1;
                count = 1;
            }
            let mut aw4 = usize::from(t::BLOCK_DIMENSIONS[usize::from(b.bs)][0]);
            if aw4 >= bw4 {
                let off = self.bx & (aw4 - 1);
                if off != 0 {
                    have_topleft = false;
                }
                if aw4 - off > bw4 {
                    have_topright = false;
                }
            } else {
                let mut mask = 1u64 << aw4;
                let mut x = aw4;
                while x < w4 {
                    b = self.refmvs.block_at(self.bx + x, self.by - 1);
                    if matches(b) {
                        masks[0] |= mask;
                        count += 1;
                        if count >= 8 {
                            return masks;
                        }
                    }
                    aw4 = usize::from(t::BLOCK_DIMENSIONS[usize::from(b.bs)][0]);
                    mask <<= aw4;
                    x += aw4;
                }
            }
        }
        if have_left {
            let mut b = self.refmvs.block_at(self.bx - 1, self.by);
            if matches(b) {
                masks[1] |= 1;
                count += 1;
                if count >= 8 {
                    return masks;
                }
            }
            let mut lh4 = usize::from(t::BLOCK_DIMENSIONS[usize::from(b.bs)][1]);
            if lh4 >= bh4 {
                if self.by & (lh4 - 1) != 0 {
                    have_topleft = false;
                }
            } else {
                let mut mask = 1u64 << lh4;
                let mut y = lh4;
                while y < h4 {
                    b = self.refmvs.block_at(self.bx - 1, self.by + y);
                    if matches(b) {
                        masks[1] |= mask;
                        count += 1;
                        if count >= 8 {
                            return masks;
                        }
                    }
                    lh4 = usize::from(t::BLOCK_DIMENSIONS[usize::from(b.bs)][1]);
                    mask <<= lh4;
                    y += lh4;
                }
            }
        }
        if have_topleft && matches(self.refmvs.block_at(self.bx - 1, self.by - 1)) {
            masks[1] |= 1u64 << 32;
            count += 1;
            if count >= 8 {
                return masks;
            }
        }
        if have_topright && matches(self.refmvs.block_at(self.bx + bw4, self.by - 1)) {
            masks[0] |= 1u64 << 32;
        }
        masks
    }

    /// `findoddzero` (`decode.c:185-189`) — `start` ofsetinden itibaren
    /// tek indeksli girdilerde sıfır arar.
    fn findoddzero(buf: &[u8], start: usize, len: usize) -> bool {
        (0..len).any(|n| buf.get(start + n * 2).copied().unwrap_or(1) == 0)
    }

    /// Tek referans seçimi (`decode.c:1620-1664`).
    fn read_single_ref(
        &self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        bx4: usize,
        by4: usize,
        have_top: bool,
        have_left: bool,
    ) -> i8 {
        let base = layout::CDF_m + layout::MODE_ref;
        let ctx1 = self.av1_get_ref_ctx(bx4, by4, have_top, have_left);
        let b0 = msac.decode_bool_adapt(&mut cdf.cells_mut()[base + ctx1 * 2..]) != 0;
        if b0 {
            let ctx2 = self.av1_get_bwd_ref_ctx(bx4, by4, have_top, have_left);
            let b1 = msac.decode_bool_adapt(&mut cdf.cells_mut()[base + (3 + ctx2) * 2..]) != 0;
            if b1 {
                6
            } else {
                let ctx3 = self.av1_get_bwd_ref_1_ctx(bx4, by4, have_top, have_left);
                let b2 =
                    msac.decode_bool_adapt(&mut cdf.cells_mut()[base + (15 + ctx3) * 2..]) != 0;
                4 + i8::from(b2)
            }
        } else {
            let ctx2 = self.av1_get_fwd_ref_ctx(bx4, by4, have_top, have_left);
            let b1 = msac.decode_bool_adapt(&mut cdf.cells_mut()[base + (6 + ctx2) * 2..]) != 0;
            if b1 {
                let ctx3 = self.av1_get_fwd_ref_2_ctx(bx4, by4, have_top, have_left);
                let b2 =
                    msac.decode_bool_adapt(&mut cdf.cells_mut()[base + (12 + ctx3) * 2..]) != 0;
                2 + i8::from(b2)
            } else {
                let ctx3 = self.av1_get_fwd_ref_1_ctx(bx4, by4, have_top, have_left);
                let b2 = msac.decode_bool_adapt(&mut cdf.cells_mut()[base + (9 + ctx3) * 2..]) != 0;
                i8::from(b2)
            }
        }
    }

    /// Inter blok sözdizimi + yeniden yapılandırma (dav1d `decode.c:1388-1980`).
    ///
    /// Kapsam: tek/çift referans, skip mode, compound (ortalama, mesafe
    /// ağırlıklı, wedge, fark ağırlıklı), interintra (smooth/wedge), OBMC,
    /// yerel warp, çeviri dışı global motion ve ölçekli referans.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn decode_inter_block(
        &mut self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        bs: u8,
        edge_flags: u8,
        b: &mut Av1Block,
        seg: Option<SegmentationData>,
        seg_pred: u8,
    ) -> Result<()> {
        let b_dim = &t::BLOCK_DIMENSIONS[usize::from(bs)];
        let bw4 = usize::from(b_dim[0]);
        let bh4 = usize::from(b_dim[1]);
        let bx4 = self.bx & 31;
        let by4 = self.by & 31;
        let ai = self.bx >> 5;
        let w4 = bw4.min(self.bw - self.bx);
        let h4 = bh4.min(self.bh - self.by);
        let have_left = self.bx > self.col_start;
        let have_top = self.by > self.row_start;
        // dav1d `decode.c:702-704`: kroma **blok düzeyinde** var mı? 4:2:0'da
        // 4x4 luma bloğunun kroması yalnız tek (bx,by) konumunda taşınır.
        // Bağlam güncellemesi bu bayrağa bağlıdır (`decode.c:1934`); yalnız
        // kare düzeyi bayrağa bakmak, kromasız inter bloğun `uvmode` üst
        // bağlamını DC_PRED ile **ezmesine** yol açar (T41: test6158 k5).
        let has_chroma = self.has_chroma
            && (bw4 > self.ss_hor || self.bx & 1 != 0)
            && (bh4 > self.ss_ver || self.by & 1 != 0);

        let tile = TileBounds {
            row_start: self.row_start,
            row_end: self.row_end,
            col_start: self.col_start,
            col_end: self.col_end,
        };

        // --- compound bayrağı (`decode.c:1388-1403`) ----------------------
        let seg_allows = seg.is_none_or(|s| s.reference == -1 && !s.globalmv && !s.skip);
        let mut is_comp = b.skip_mode;
        if !b.skip_mode && seg_allows && self.hdr.switchable_comp_refs && bw4.min(bh4) > 1 {
            let ctx = self.get_comp_ctx(bx4, by4, have_top, have_left);
            is_comp = msac.decode_bool_adapt(
                &mut cdf.cells_mut()[layout::CDF_m + layout::MODE_comp + ctx * 2..],
            ) != 0;
        }
        b.comp_type = COMP_INTER_NONE;
        b.ref1 = -1;
        let mut has_subpel_filter: bool;

        if b.skip_mode {
            // `decode.c:1405-1428`
            b.ref0 = self.hdr.skip_mode_refs[0];
            b.ref1 = self.hdr.skip_mode_refs[1];
            b.comp_type = COMP_INTER_AVG;
            b.inter_mode = 0; // NEARESTMV_NEARESTMV
            b.drl_idx = NEAREST_DRL;
            has_subpel_filter = false;
            let (stack, _n_mvs, _ctx) = self.refmvs.find(
                [b.ref0 + 1, b.ref1 + 1],
                bs,
                edge_flags,
                self.by,
                self.bx,
                tile,
            );
            b.mv[0] = stack[0].mv[0];
            b.mv[1] = stack[0].mv[1];
            fix_mv_precision(self.hdr, &mut b.mv[0]);
            fix_mv_precision(self.hdr, &mut b.mv[1]);
            self.tools.skip_mode += 1;
        } else if is_comp {
            // `decode.c:1429-1619`
            let base = layout::CDF_m;
            let dir_ctx = self.get_comp_dir_ctx(bx4, by4, have_top, have_left);
            let bidir = msac.decode_bool_adapt(
                &mut cdf.cells_mut()[base + layout::MODE_comp_dir + dir_ctx * 2..],
            ) != 0;
            if bidir {
                let ctx1 = self.av1_get_fwd_ref_ctx(bx4, by4, have_top, have_left);
                let fwd = msac.decode_bool_adapt(
                    &mut cdf.cells_mut()[base + layout::MODE_comp_fwd_ref + ctx1 * 2..],
                ) != 0;
                if fwd {
                    let ctx2 = self.av1_get_fwd_ref_2_ctx(bx4, by4, have_top, have_left);
                    b.ref0 = 2 + i8::from(
                        msac.decode_bool_adapt(
                            &mut cdf.cells_mut()
                                [base + layout::MODE_comp_fwd_ref + (6 + ctx2) * 2..],
                        ) != 0,
                    );
                } else {
                    let ctx2 = self.av1_get_fwd_ref_1_ctx(bx4, by4, have_top, have_left);
                    b.ref0 = i8::from(
                        msac.decode_bool_adapt(
                            &mut cdf.cells_mut()
                                [base + layout::MODE_comp_fwd_ref + (3 + ctx2) * 2..],
                        ) != 0,
                    );
                }
                let ctx3 = self.av1_get_bwd_ref_ctx(bx4, by4, have_top, have_left);
                let bwd = msac.decode_bool_adapt(
                    &mut cdf.cells_mut()[base + layout::MODE_comp_bwd_ref + ctx3 * 2..],
                ) != 0;
                if bwd {
                    b.ref1 = 6;
                } else {
                    let ctx4 = self.av1_get_bwd_ref_1_ctx(bx4, by4, have_top, have_left);
                    b.ref1 = 4 + i8::from(
                        msac.decode_bool_adapt(
                            &mut cdf.cells_mut()
                                [base + layout::MODE_comp_bwd_ref + (3 + ctx4) * 2..],
                        ) != 0,
                    );
                }
            } else {
                // tek yönlü compound referansları
                let uctx_p = self.av1_get_ref_ctx(bx4, by4, have_top, have_left);
                let uni = msac.decode_bool_adapt(
                    &mut cdf.cells_mut()[base + layout::MODE_comp_uni_ref + uctx_p * 2..],
                ) != 0;
                if uni {
                    b.ref0 = 4;
                    b.ref1 = 6;
                } else {
                    let uctx_p1 = self.av1_get_uni_p1_ctx(bx4, by4, have_top, have_left);
                    b.ref0 = 0;
                    b.ref1 = 1 + i8::from(
                        msac.decode_bool_adapt(
                            &mut cdf.cells_mut()
                                [base + layout::MODE_comp_uni_ref + (3 + uctx_p1) * 2..],
                        ) != 0,
                    );
                    if b.ref1 == 2 {
                        let uctx_p2 = self.av1_get_fwd_ref_2_ctx(bx4, by4, have_top, have_left);
                        b.ref1 += i8::from(
                            msac.decode_bool_adapt(
                                &mut cdf.cells_mut()
                                    [base + layout::MODE_comp_uni_ref + (6 + uctx_p2) * 2..],
                            ) != 0,
                        );
                    }
                }
            }

            let (stack, n_mvs, ctx) = self.refmvs.find(
                [b.ref0 + 1, b.ref1 + 1],
                bs,
                edge_flags,
                self.by,
                self.bx,
                tile,
            );
            b.inter_mode = msac.decode_symbol_adapt(
                &mut cdf.cells_mut()[base + layout::MODE_comp_inter_mode + ctx * 8..],
                7,
            ) as u8;
            let im = COMP_INTER_PRED_MODES[usize::from(b.inter_mode)];
            b.drl_idx = NEAREST_DRL;
            if b.inter_mode == 7 {
                // NEWMV_NEWMV
                if n_mvs > 1 {
                    let c = get_drl_context(&stack, 0);
                    b.drl_idx += u8::from(
                        msac.decode_bool_adapt(
                            &mut cdf.cells_mut()[base + layout::MODE_drl_bit + c * 2..],
                        ) != 0,
                    );
                    if b.drl_idx == NEARER_DRL && n_mvs > 2 {
                        let c = get_drl_context(&stack, 1);
                        b.drl_idx += u8::from(
                            msac.decode_bool_adapt(
                                &mut cdf.cells_mut()[base + layout::MODE_drl_bit + c * 2..],
                            ) != 0,
                        );
                    }
                }
            } else if im[0] == NEARMV || im[1] == NEARMV {
                b.drl_idx = NEARER_DRL;
                if n_mvs > 2 {
                    let c = get_drl_context(&stack, 1);
                    b.drl_idx += u8::from(
                        msac.decode_bool_adapt(
                            &mut cdf.cells_mut()[base + layout::MODE_drl_bit + c * 2..],
                        ) != 0,
                    );
                    if b.drl_idx == NEAR_DRL && n_mvs > 3 {
                        let c = get_drl_context(&stack, 2);
                        b.drl_idx += u8::from(
                            msac.decode_bool_adapt(
                                &mut cdf.cells_mut()[base + layout::MODE_drl_bit + c * 2..],
                            ) != 0,
                        );
                    }
                }
            }
            has_subpel_filter =
                bw4.min(bh4) == 1 || b.inter_mode != 6 /* GLOBALMV_GLOBALMV */;
            for (idx, &mode) in im.iter().enumerate() {
                let rref = if idx == 0 { b.ref0 } else { b.ref1 };
                if !(0..7).contains(&rref) {
                    return Err(Error::InvalidData("inter referans indeksi aralık dışında"));
                }
                match mode {
                    NEARMV | NEARESTMV => {
                        b.mv[idx] = stack[usize::from(b.drl_idx)].mv[idx];
                        fix_mv_precision(self.hdr, &mut b.mv[idx]);
                    }
                    GLOBALMV => {
                        let gmv_params = self.hdr.gmv[usize::from(rref as u8)];
                        if gmv_params.kind
                            == crate::bitstream::frame_header::WarpedMotionType::Translation
                        {
                            has_subpel_filter = true;
                        }
                        b.mv[idx] = get_gmv_2d(
                            &gmv_params,
                            self.bx,
                            self.by,
                            bw4,
                            bh4,
                            self.hdr.hp,
                            self.hdr.force_integer_mv,
                        );
                    }
                    NEWMV => {
                        b.mv[idx] = stack[usize::from(b.drl_idx)].mv[idx];
                        let mv_prec = i32::from(self.hdr.hp) - i32::from(self.hdr.force_integer_mv);
                        Self::read_mv_residual(msac, cdf, &mut b.mv[idx], mv_prec);
                    }
                    _ => return Err(Error::InvalidData("geçersiz compound inter modu")),
                }
            }

            // jnt_comp / seg / wedge seçimi (`decode.c:1588-1619`)
            let mut is_segwedge = false;
            if self.seq.masked_compound {
                let mask_ctx = self.get_mask_comp_ctx(bx4, by4);
                is_segwedge = msac.decode_bool_adapt(
                    &mut cdf.cells_mut()[base + layout::MODE_mask_comp + mask_ctx * 2..],
                ) != 0;
            }
            if !is_segwedge {
                if self.seq.jnt_comp {
                    let jnt_ctx = self.get_jnt_comp_ctx(bx4, by4, b.ref0, b.ref1);
                    let w = msac.decode_bool_adapt(
                        &mut cdf.cells_mut()[base + layout::MODE_jnt_comp + jnt_ctx * 2..],
                    ) != 0;
                    b.comp_type = COMP_INTER_WEIGHTED_AVG + u8::from(w);
                    self.tools.comp_weighted += u64::from(!w);
                    self.tools.comp_avg += u64::from(w);
                } else {
                    b.comp_type = COMP_INTER_AVG;
                    self.tools.comp_avg += 1;
                }
            } else {
                let wedge_ctx = usize::from(t::WEDGE_CTX_LUT[usize::from(bs)]);
                if t::WEDGE_ALLOWED_MASK & (1u32 << bs) != 0 {
                    let w = msac.decode_bool_adapt(
                        &mut cdf.cells_mut()[base + layout::MODE_wedge_comp + wedge_ctx * 2..],
                    ) != 0;
                    b.comp_type = COMP_INTER_WEDGE - u8::from(w);
                    if b.comp_type == COMP_INTER_WEDGE {
                        b.wedge_idx = msac.decode_symbol_adapt(
                            &mut cdf.cells_mut()[base + layout::MODE_wedge_idx + wedge_ctx * 16..],
                            15,
                        ) as u8;
                    }
                } else {
                    b.comp_type = COMP_INTER_SEG;
                }
                b.mask_sign = u8::from(msac.decode_bool_equi() != 0);
                if b.comp_type == COMP_INTER_WEDGE {
                    self.tools.comp_wedge += 1;
                } else {
                    self.tools.comp_seg += 1;
                }
            }
            self.tools.compound += 1;
        } else {
            // Tek referans (`decode.c:1620-1780`).
            b.comp_type = COMP_INTER_NONE;
            b.ref0 = match seg {
                Some(s) if s.reference > 0 => s.reference - 1,
                Some(s) if s.globalmv || s.skip => 0,
                _ => self.read_single_ref(msac, cdf, bx4, by4, have_top, have_left),
            };

            let (stack, n_mvs, ctx) =
                self.refmvs
                    .find([b.ref0 + 1, -1], bs, edge_flags, self.by, self.bx, tile);
            let seg_force = seg.is_some_and(|s| s.skip || s.globalmv);
            let gmv_params = self.hdr.gmv[usize::from(b.ref0 as u8)];

            let not_newmv = seg_force
                || msac.decode_bool_adapt(
                    &mut cdf.cells_mut()[layout::CDF_m + layout::MODE_newmv_mode + (ctx & 7) * 2..],
                ) != 0;
            if not_newmv {
                let not_globalmv = !seg_force
                    && msac.decode_bool_adapt(
                        &mut cdf.cells_mut()
                            [layout::CDF_m + layout::MODE_globalmv_mode + ((ctx >> 3) & 1) * 2..],
                    ) != 0;
                if !not_globalmv {
                    b.inter_mode = GLOBALMV;
                    b.mv[0] = get_gmv_2d(
                        &gmv_params,
                        self.bx,
                        self.by,
                        bw4,
                        bh4,
                        self.hdr.hp,
                        self.hdr.force_integer_mv,
                    );
                    has_subpel_filter = bw4.min(bh4) == 1
                        || gmv_params.kind
                            == crate::bitstream::frame_header::WarpedMotionType::Translation;
                    self.tools.globalmv += 1;
                } else {
                    has_subpel_filter = true;
                    let near = msac.decode_bool_adapt(
                        &mut cdf.cells_mut()
                            [layout::CDF_m + layout::MODE_refmv_mode + ((ctx >> 4) & 15) * 2..],
                    ) != 0;
                    if near {
                        b.inter_mode = NEARMV;
                        b.drl_idx = NEARER_DRL;
                        if n_mvs > 2 {
                            let c = get_drl_context(&stack, 1);
                            b.drl_idx += u8::from(
                                msac.decode_bool_adapt(
                                    &mut cdf.cells_mut()
                                        [layout::CDF_m + layout::MODE_drl_bit + c * 2..],
                                ) != 0,
                            );
                            if b.drl_idx == NEAR_DRL && n_mvs > 3 {
                                let c = get_drl_context(&stack, 2);
                                b.drl_idx += u8::from(
                                    msac.decode_bool_adapt(
                                        &mut cdf.cells_mut()
                                            [layout::CDF_m + layout::MODE_drl_bit + c * 2..],
                                    ) != 0,
                                );
                            }
                        }
                    } else {
                        b.inter_mode = NEARESTMV;
                        b.drl_idx = NEAREST_DRL;
                    }
                    b.mv[0] = stack[usize::from(b.drl_idx)].mv[0];
                    if b.drl_idx < NEAR_DRL {
                        fix_mv_precision(self.hdr, &mut b.mv[0]);
                    }
                    self.tools.nearmv += 1;
                }
            } else {
                has_subpel_filter = true;
                b.inter_mode = NEWMV;
                b.drl_idx = NEAREST_DRL;
                self.tools.newmv += 1;
                if n_mvs > 1 {
                    let c = get_drl_context(&stack, 0);
                    b.drl_idx += u8::from(
                        msac.decode_bool_adapt(
                            &mut cdf.cells_mut()[layout::CDF_m + layout::MODE_drl_bit + c * 2..],
                        ) != 0,
                    );
                    if b.drl_idx == NEARER_DRL && n_mvs > 2 {
                        let c = get_drl_context(&stack, 1);
                        b.drl_idx += u8::from(
                            msac.decode_bool_adapt(
                                &mut cdf.cells_mut()
                                    [layout::CDF_m + layout::MODE_drl_bit + c * 2..],
                            ) != 0,
                        );
                    }
                }
                if n_mvs > 1 {
                    b.mv[0] = stack[usize::from(b.drl_idx)].mv[0];
                } else {
                    b.mv[0] = stack[0].mv[0];
                    fix_mv_precision(self.hdr, &mut b.mv[0]);
                }
                let mv_prec = i32::from(self.hdr.hp) - i32::from(self.hdr.force_integer_mv);
                Self::read_mv_residual(msac, cdf, &mut b.mv[0], mv_prec);
            }

            // --- interintra bayrağı (`decode.c:1745-1776`) ----------------
            b.interintra_type = 0;
            if self.seq.inter_intra
                && t::INTERINTRA_ALLOWED_MASK & (1u32 << bs) != 0
                && msac.decode_bool_adapt(
                    &mut cdf.cells_mut()[layout::CDF_m
                        + layout::MODE_interintra
                        + usize::from(t::YMODE_SIZE_CONTEXT[usize::from(bs)]) * 2..],
                ) != 0
            {
                let grp = usize::from(t::YMODE_SIZE_CONTEXT[usize::from(bs)]);
                b.interintra_mode = msac.decode_symbol_adapt(
                    &mut cdf.cells_mut()[layout::CDF_m + layout::MODE_interintra_mode + grp * 4..],
                    3,
                ) as u8;
                let wedge_ctx = usize::from(t::WEDGE_CTX_LUT[usize::from(bs)]);
                let w = msac.decode_bool_adapt(
                    &mut cdf.cells_mut()
                        [layout::CDF_m + layout::MODE_interintra_wedge + wedge_ctx * 2..],
                ) != 0;
                b.interintra_type = INTER_INTRA_BLEND + u8::from(w);
                if b.interintra_type == INTER_INTRA_WEDGE {
                    b.wedge_idx = msac.decode_symbol_adapt(
                        &mut cdf.cells_mut()
                            [layout::CDF_m + layout::MODE_wedge_idx + wedge_ctx * 16..],
                        15,
                    ) as u8;
                }
                if b.interintra_type == INTER_INTRA_WEDGE {
                    self.tools.interintra_wedge += 1;
                } else {
                    self.tools.interintra_blend += 1;
                }
            }

            // --- hareket kipi (OBMC/warp) (`decode.c:1778-1832`) ----------
            b.motion_mode = MM_TRANSLATION;
            if self.hdr.switchable_motion_mode
                && b.interintra_type == 0
                && bw4.min(bh4) >= 2
                && !(!self.hdr.force_integer_mv
                    && b.inter_mode == GLOBALMV
                    && gmv_params.kind.raw()
                        > crate::bitstream::frame_header::WarpedMotionType::Translation.raw())
                && ((have_left && Self::findoddzero(&self.blocks.left.intra, by4 + 1, h4 >> 1))
                    || (have_top
                        && Self::findoddzero(
                            &self.blocks.above[self.above_base + ai].intra,
                            bx4 + 1,
                            w4 >> 1,
                        )))
            {
                let mask = self
                    .find_matching_ref(edge_flags, bw4, bh4, w4, h4, have_left, have_top, b.ref0);
                let allow_warp = self.svc[usize::from(b.ref0 as u8)][0].scale == 0
                    && !self.hdr.force_integer_mv
                    && self.hdr.warp_motion
                    && (mask[0] | mask[1]) != 0;
                let mode = if allow_warp {
                    msac.decode_symbol_adapt(
                        &mut cdf.cells_mut()
                            [layout::CDF_m + layout::MODE_motion_mode + usize::from(bs) * 4..],
                        2,
                    ) as u8
                } else {
                    if msac.decode_bool_adapt(
                        &mut cdf.cells_mut()
                            [layout::CDF_m + layout::MODE_obmc + usize::from(bs) * 2..],
                    ) != 0
                    {
                        MM_OBMC
                    } else {
                        MM_TRANSLATION
                    }
                };
                b.motion_mode = mode;
                match mode {
                    MM_WARP => {
                        has_subpel_filter = false;
                        self.tools.warp_local += 1;
                        let (wmp, ok) = self.derive_warpmv(bw4, bh4, &mask, b.mv[0]);
                        if ok {
                            let shear = warpv::get_shear_params(&wmp);
                            if let Some(abcd) = shear {
                                let mut wmp = wmp;
                                wmp.kind = crate::bitstream::frame_header::WarpedMotionType::Affine;
                                b.warp = Some(WarpParams {
                                    matrix: wmp.matrix,
                                    abcd,
                                });
                            }
                        }
                    }
                    MM_OBMC => self.tools.obmc += 1,
                    _ => {}
                }
            }
        }
        if b.ref0 < 0 || b.ref0 >= 7 || (is_comp && (b.ref1 < 0 || b.ref1 >= 7)) {
            return Err(Error::InvalidData("inter referans indeksi aralık dışında"));
        }
        if self.svc[usize::from(b.ref0 as u8)][0].scale != 0
            || (is_comp && self.svc[usize::from(b.ref1 as u8)][0].scale != 0)
        {
            self.tools.scaled_ref += 1;
            // Superres'li (büyütülmüş) referanstan ölçekli MC (tanılama).
            let sr_ref = |r: i8| {
                usize::try_from(r)
                    .ok()
                    .and_then(|i| self.refs.get(usize::from(self.hdr.refidx[i] as u8)))
                    .and_then(|r| r.as_ref())
                    .is_some_and(|r| r.hdr.width[0] != r.hdr.width[1])
            };
            if sr_ref(b.ref0) || (is_comp && sr_ref(b.ref1)) {
                self.tools.sr_ref_scaled += 1;
            }
        }

        // --- interp filtresi (`decode.c:1835-1874`) -----------------------
        use crate::bitstream::frame_header::FilterMode;
        if self.hdr.subpel_filter_mode == FilterMode::Switchable {
            if has_subpel_filter {
                let comp = usize::from(is_comp);
                let ctx1 = self.get_filter_ctx(bx4, by4, comp, 0, b.ref0);
                b.filter[0] = msac.decode_symbol_adapt(
                    &mut cdf.cells_mut()[layout::CDF_m + layout::MODE_filter + ctx1 * 4..],
                    2,
                ) as u8;
                if self.seq.dual_filter {
                    let ctx2 = self.get_filter_ctx(bx4, by4, comp, 1, b.ref0);
                    b.filter[1] = msac.decode_symbol_adapt(
                        &mut cdf.cells_mut()
                            [layout::CDF_m + layout::MODE_filter + (8 + ctx2) * 4..],
                        2,
                    ) as u8;
                    self.tools.dual_filter += 1;
                } else {
                    b.filter[1] = b.filter[0];
                }
            } else {
                b.filter = [FILTER_8TAP_REGULAR; 2];
            }
        } else {
            let f = match self.hdr.subpel_filter_mode {
                FilterMode::EightTapRegular => FILTER_8TAP_REGULAR,
                FilterMode::EightTapSmooth => FILTER_8TAP_SMOOTH,
                FilterMode::EightTapSharp => FILTER_8TAP_SHARP,
                // **Tuzak:** `FilterMode::Bilinear` (3) `_` dalına düşerse
                // 8-tap regular seçilir; bütün kare yanlış örneklenir
                // (T40: `test30` k5, `subpel_filter_mode = bilinear`).
                FilterMode::Bilinear => FILTER_BILINEAR,
                _ => FILTER_8TAP_REGULAR,
            };
            b.filter = [f, f];
        }
        b.filter2d = t::FILTER_2D[usize::from(b.filter[1])][usize::from(b.filter[0])];
        // --- tx ağacı (`decode.c:1876`) -----------------------------------
        self.read_vartx_tree(msac, cdf, bs, b)?;

        // --- yeniden yapılandırma (`recon_tmpl.c:1557-2040`) -------------
        self.recon_b_inter_real(msac, cdf, bs, b)?;
        self.create_lf_mask(bs, b, false);

        // --- bağlam güncellemesi (`decode.c:1927-1962`) ------------------
        if is_comp {
            self.refmvs.splat_tworef(
                self.bx,
                self.by,
                bw4,
                bh4,
                bs,
                b.mv,
                b.ref0,
                b.ref1,
                b.inter_mode,
            );
        } else {
            self.refmvs.splat_inter(
                self.bx,
                self.by,
                bw4,
                bh4,
                bs,
                b.mv[0],
                b.ref0,
                b.interintra_type != 0,
                b.inter_mode,
            );
        }
        let above_idx = self.above_base + ai;
        fill_inter_side_ctx(
            &mut self.blocks.above[above_idx],
            &mut self.blocks.above_pal_uv_sz[above_idx],
            bx4,
            b_dim[2],
            seg_pred,
            b,
        );
        fill_inter_side_ctx(
            &mut self.blocks.left,
            &mut self.blocks.left_pal_uv_sz,
            by4,
            b_dim[3],
            seg_pred,
            b,
        );
        if has_chroma {
            let cbx4 = bx4 >> self.ss_hor;
            let cby4 = by4 >> self.ss_ver;
            let cbw4 = (bw4 + self.ss_hor) >> self.ss_hor;
            let cbh4 = (bh4 + self.ss_ver) >> self.ss_ver;
            fill_small(
                &mut self.blocks.above[above_idx].uvmode,
                cbx4,
                cbw4,
                DC_PRED,
            );
            fill_small(&mut self.blocks.left.uvmode, cby4, cbh4, DC_PRED);
        }
        self.tools.inter_single_ref += u64::from(!is_comp);
        Ok(())
    }

    /// `mc()` (`recon_tmpl.c:938-1048`) — tek blok MC; ölçekli referans dâhil.
    ///
    /// `out` hedefi seçer: kare düzlemi (`dst8`), ham bayt dilimi (OBMC
    /// `lap`) ya da `i16` ara tampon (`tmp`, compound).
    #[allow(clippy::too_many_arguments)]
    fn mc_block(
        &mut self,
        plane: usize,
        ref_idx: usize,
        bx: usize,
        by: usize,
        bw4: usize,
        bh4: usize,
        mv: Mv,
        filter2d: u8,
        out: McOut<'_>,
    ) -> Result<()> {
        let ref_slot = usize::from(self.hdr.refidx[ref_idx] as u8);
        let ref_frame = self
            .refs
            .get(ref_slot)
            .and_then(|r| r.as_deref())
            .ok_or(Error::Unsupported("referans çözülmedi: inter referans"))?;
        let refp = &ref_frame.pic;
        let ss_hor = if plane != 0 { self.ss_hor } else { 0 };
        let ss_ver = if plane != 0 { self.ss_ver } else { 0 };
        let h_mul = 4 >> ss_hor;
        let v_mul = 4 >> ss_ver;
        let pw = bw4 * h_mul;
        let ph = bh4 * v_mul;
        let bpp = self.bd.bytes_per_sample();
        let bmax = self.bd.max_sample() as u16;
        let bd = self.bd;
        let mc = self.bdsp.mc();
        let Some(f2d) = crate::dsp::mc::Filter2d::from_index(usize::from(filter2d)) else {
            return Err(Error::InvalidData("geçersiz interp filtresi"));
        };
        let scaled = self.svc[ref_idx][0].scale != 0 || self.svc[ref_idx][1].scale != 0;

        // `src` = filtre penceresinin sol-üstü (blok kaynağı − 3).
        let mut emu: Option<Vec<u8>> = None;
        let mut src_off = 0usize;
        let src_stride;
        let mut mx_put: u8 = 0;
        let mut my_put: u8 = 0;
        let mut mx_i = 0i32;
        let mut my_i = 0i32;
        let mut step_x = 0i32;
        let mut step_y = 0i32;
        // dav1d `mc()` (`recon_tmpl.c:966-990,1015-1040`) `emu_edge` koşulu ve
        // parametreleri. İz iki tarafta aynı satırları üretsin diye Rust'ın
        // (daha korumacı) koşulundan ayrı tutulur; Rust `src` sözleşmesi
        // dolgu hücresini gösterdiğinden gerçek emu kullanımı değişmez
        // (emu iç bölgede de birebir aynı pikselleri kopyalar).
        let emu_c;
        let emu_cbw;
        let emu_cbh;
        let emu_ciw;
        let emu_cih;
        let emu_cx;
        let emu_cy;
        let emu_hoff;
        let emu_voff;
        let emu_cstride;
        let trace_sx_c;
        let trace_sy_c;
        if !scaled {
            let mvx = i32::from(mv.x);
            let mvy = i32::from(mv.y);
            let mx = mvx & (15 >> u32::from(ss_hor == 0));
            let my = mvy & (15 >> u32::from(ss_ver == 0));
            let dx = bx as i32 * h_mul as i32 + (mvx >> (3 + ss_hor));
            let dy = by as i32 * v_mul as i32 + (mvy >> (3 + ss_ver));
            let w = ((refp.width + ss_hor) >> ss_hor) as i32;
            let h = ((refp.height + ss_ver) >> ss_ver) as i32;
            mx_put = (mx << u32::from(ss_hor == 0)) as u8;
            my_put = (my << u32::from(ss_ver == 0)) as u8;
            trace_sx_c = dx;
            trace_sy_c = dy;
            let mxn = i32::from(mx_put != 0);
            let myn = i32::from(my_put != 0);
            emu_c = dx < 3 * mxn
                || dy < 3 * myn
                || dx + pw as i32 + 4 * mxn > w
                || dy + ph as i32 + 4 * myn > h;
            emu_cbw = pw + 7 * mxn as usize;
            emu_cbh = ph + 7 * myn as usize;
            emu_ciw = w as usize;
            emu_cih = h as usize;
            emu_cx = dx - 3 * mxn;
            emu_cy = dy - 3 * myn;
            emu_hoff = (3 - 3 * mxn) as usize;
            emu_voff = (3 - 3 * myn) as usize;
            emu_cstride = 192 * bpp;
            if dx < 3 || dy < 3 || dx + pw as i32 + 4 > w || dy + ph as i32 + 4 > h {
                let mut buf = core::mem::take(&mut self.scratch.emu_edge);
                crate::dsp::profil_olc!(
                    McEmuEdge,
                    (mc.emu_edge)(
                        pw + 7,
                        ph + 7,
                        w as usize,
                        h as usize,
                        dx - 3,
                        dy - 3,
                        &mut buf,
                        192 * bpp,
                        &refp.data[plane],
                        refp.stride[plane],
                        bd,
                        bmax,
                    )
                );
                src_stride = 192 * bpp;
                emu = Some(buf);
            } else {
                src_stride = refp.stride[plane];
                src_off = (dy as usize - 3) * refp.stride[plane] + (dx as usize - 3) * bpp;
            }
        } else {
            // `scale_mv` (`recon_tmpl.c:1001-1009`).
            let smp_x = self.svc[ref_idx][0];
            let smp_y = self.svc[ref_idx][1];
            let mvx = i32::from(mv.x);
            let mvy = i32::from(mv.y);
            let orig_pos_y = ((by * v_mul) << 4) as i32 + mvy * (1 << u32::from(ss_ver == 0));
            let orig_pos_x = ((bx * h_mul) << 4) as i32 + mvx * (1 << u32::from(ss_hor == 0));
            let scale_mv = |val: i32, scale: i32| -> i32 {
                let tmp = i64::from(val) * i64::from(scale) + i64::from(scale - 0x4000) * 8;
                let mag = ((tmp.abs() + 128) >> 8) as i32;
                (if tmp < 0 { -mag } else { mag }) + 32
            };
            let pos_x = scale_mv(orig_pos_x, smp_x.scale);
            let pos_y = scale_mv(orig_pos_y, smp_y.scale);
            let left = pos_x >> 10;
            let top = pos_y >> 10;
            let right = ((pos_x + (pw as i32 - 1) * smp_x.step) >> 10) + 1;
            let bottom = ((pos_y + (ph as i32 - 1) * smp_y.step) >> 10) + 1;
            let w = ((refp.width + ss_hor) >> ss_hor) as i32;
            let h = ((refp.height + ss_ver) >> ss_ver) as i32;
            trace_sx_c = left;
            trace_sy_c = top;
            emu_c = left < 3 || top < 3 || right + 4 > w || bottom + 4 > h;
            emu_cbw = (right - left + 7) as usize;
            emu_cbh = (bottom - top + 7) as usize;
            emu_ciw = w as usize;
            emu_cih = h as usize;
            emu_cx = left - 3;
            emu_cy = top - 3;
            emu_hoff = 0;
            emu_voff = 0;
            emu_cstride = 320 * bpp;
            if left < 3 || top < 3 || right + 4 > w || bottom + 4 > h {
                let mut buf = core::mem::take(&mut self.scratch.emu_edge);
                crate::dsp::profil_olc!(
                    McEmuEdge,
                    (mc.emu_edge)(
                        (right - left + 7) as usize,
                        (bottom - top + 7) as usize,
                        w as usize,
                        h as usize,
                        left - 3,
                        top - 3,
                        &mut buf,
                        320 * bpp,
                        &refp.data[plane],
                        refp.stride[plane],
                        bd,
                        bmax,
                    )
                );
                src_stride = 320 * bpp;
                emu = Some(buf);
            } else {
                src_stride = refp.stride[plane];
                src_off = (top as usize - 3) * refp.stride[plane] + (left as usize - 3) * bpp;
            }
            mx_i = pos_x & 0x3ff;
            my_i = pos_y & 0x3ff;
            step_x = smp_x.step;
            step_y = smp_y.step;
        }
        let src: &[u8] = match &emu {
            Some(buf) => buf,
            None => &refp.data[plane][src_off..],
        };
        let trace_on = dsp_trace().frame_ok(self.frame_no);
        // `scripts/oracle/dsp_trace.c` `D` satırıyla aynı alanlar: kare,
        // işlev, filtre, blok (4x4), w, h, subpel/ölçek parametreleri,
        // kaynak yuvası (255 = `emu_edge`), kaynak (x,y).
        let (trace_fname, trace_mx, trace_my, trace_p1, trace_p2, trace_refpoc, trace_sx, trace_sy) =
            if trace_on {
                let (mxv, myv, p1, p2, fname) = if scaled {
                    (mx_i, my_i, step_x, step_y, "mcs")
                } else {
                    (i32::from(mx_put), i32::from(my_put), 0, 0, "mc")
                };
                // dav1d `emu_edge` çağırmadıysa `src` referans yuvasındadır
                // ve `trace_ref_slot` kaynak konumunu **merkez** olarak verir
                // (Rust `src`'si dolgu hücresidir: +3 px).
                let (refpoc, sx, sy) = if emu_c {
                    (255u32, 0u32, 0u32)
                } else {
                    let poc = u32::from(ref_frame.hdr.frame_offset);
                    (
                        poc,
                        u32::try_from(trace_sx_c).unwrap_or(0),
                        u32::try_from(trace_sy_c).unwrap_or(0),
                    )
                };
                (fname, mxv, myv, p1, p2, refpoc, sx, sy)
            } else {
                ("", 0, 0, 0, 0, 255u32, 0u32, 0u32)
            };
        match out {
            McOut::Pic { plane, x, y } => {
                let dst_off = self.pic.off(plane, x, y);
                let stride = self.pic.stride[plane];
                let dst = &mut self.pic.data[plane][dst_off..];
                if scaled {
                    crate::dsp::profil_olc!(
                        McMcScaled,
                        (mc.mc_scaled[f2d.index()])(
                            dst, stride, src, src_stride, pw, ph, mx_i, my_i, step_x, step_y, bd,
                            bmax,
                        )
                    );
                } else {
                    crate::dsp::profil_olc!(
                        McMc,
                        (mc.mc[f2d.index()])(
                            dst, stride, src, src_stride, pw, ph, mx_put, my_put, bd, bmax,
                        )
                    );
                }
                if trace_on {
                    let hash = pred_hash(&self.pic.data[plane], dst_off, stride, pw, ph, bpp);
                    mc_trace_line(
                        self.frame_no,
                        trace_fname,
                        f2d.index(),
                        self.bx,
                        self.by,
                        pw,
                        ph,
                        trace_mx,
                        trace_my,
                        trace_p1,
                        trace_p2,
                        trace_refpoc,
                        trace_sx,
                        trace_sy,
                        hash,
                    );
                }
            }
            McOut::Bytes { dst, stride, off } => {
                if scaled {
                    crate::dsp::profil_olc!(
                        McMcScaled,
                        (mc.mc_scaled[f2d.index()])(
                            &mut dst[off..],
                            stride,
                            src,
                            src_stride,
                            pw,
                            ph,
                            mx_i,
                            my_i,
                            step_x,
                            step_y,
                            bd,
                            bmax,
                        )
                    );
                } else {
                    crate::dsp::profil_olc!(
                        McMc,
                        (mc.mc[f2d.index()])(
                            &mut dst[off..],
                            stride,
                            src,
                            src_stride,
                            pw,
                            ph,
                            mx_put,
                            my_put,
                            bd,
                            bmax,
                        )
                    );
                }
                if trace_on {
                    let hash = pred_hash(dst, off, stride, pw, ph, bpp);
                    mc_trace_line(
                        self.frame_no,
                        trace_fname,
                        f2d.index(),
                        self.bx,
                        self.by,
                        pw,
                        ph,
                        trace_mx,
                        trace_my,
                        trace_p1,
                        trace_p2,
                        trace_refpoc,
                        trace_sx,
                        trace_sy,
                        hash,
                    );
                }
            }
            McOut::Prep { tmp } => {
                if scaled {
                    crate::dsp::profil_olc!(
                        McMctScaled,
                        (mc.mct_scaled[f2d.index()])(
                            tmp, src, src_stride, pw, ph, mx_i, my_i, step_x, step_y, bd, bmax,
                        )
                    );
                } else {
                    crate::dsp::profil_olc!(
                        McMct,
                        (mc.mct[f2d.index()])(
                            tmp, src, src_stride, pw, ph, mx_put, my_put, bd, bmax
                        )
                    );
                }
                if trace_on {
                    let hash = hash_i16(&tmp[..pw * ph]);
                    mct_trace_line(
                        self.frame_no,
                        if scaled { "mcts" } else { "mct" },
                        f2d.index(),
                        self.bx,
                        self.by,
                        pw,
                        ph,
                        trace_mx,
                        trace_my,
                        trace_p1,
                        trace_p2,
                        trace_refpoc,
                        trace_sx,
                        trace_sy,
                        hash,
                    );
                }
            }
        }
        if let Some(buf) = emu.as_ref().filter(|_| trace_on && emu_c) {
            let off = emu_hoff * bpp + emu_voff * emu_cstride;
            let hash = pred_hash(buf, off, emu_cstride, emu_cbw, emu_cbh, bpp);
            emu_trace_line(
                self.frame_no,
                self.bx,
                self.by,
                emu_cbw,
                emu_cbh,
                emu_ciw,
                emu_cih,
                emu_cx,
                emu_cy,
                hash,
            );
        }
        if let Some(buf) = emu {
            self.scratch.emu_edge = buf;
        }
        Ok(())
    }

    /// Alt-8×8 kroma için komşu referanslı MC (`recon_tmpl.c:1636-1690`).
    #[allow(clippy::too_many_arguments)]
    fn mc_chroma_ref(
        &mut self,
        plane: usize,
        ref_idx: i8,
        bx: usize,
        by: usize,
        bw4: usize,
        bh4: usize,
        mv: Mv,
        filter2d: u8,
        dst_x: usize,
        dst_y: usize,
    ) -> Result<()> {
        self.mc_block(
            plane,
            usize::from(ref_idx as u8),
            bx,
            by,
            bw4,
            bh4,
            mv,
            filter2d,
            McOut::Pic {
                plane,
                x: dst_x,
                y: dst_y,
            },
        )
    }

    /// `warp_affine()` (`recon_tmpl.c:1115-1180`).
    #[allow(clippy::too_many_arguments)]
    fn warp_affine_block(
        &mut self,
        plane: usize,
        ref_slot: usize,
        bx: usize,
        by: usize,
        bw4: usize,
        bh4: usize,
        wp: &WarpParams,
        mut out: McOut<'_>,
    ) -> Result<()> {
        let ref_frame = self
            .refs
            .get(ref_slot)
            .and_then(|r| r.as_deref())
            .ok_or(Error::Unsupported("referans çözülmedi: inter referans"))?;
        let refp = &ref_frame.pic;
        let ss_hor = if plane != 0 { self.ss_hor } else { 0 };
        let ss_ver = if plane != 0 { self.ss_ver } else { 0 };
        let h_mul = 4 >> ss_hor;
        let v_mul = 4 >> ss_ver;
        let width = (refp.width + ss_hor) >> ss_hor;
        let height = (refp.height + ss_ver) >> ss_ver;
        let bpp = self.bd.bytes_per_sample();
        let bmax = self.bd.max_sample() as u16;
        let bd = self.bd;
        let mc = self.bdsp.mc();
        let mat = &wp.matrix;
        let mut src_stride;
        let mut src_off = 0usize;
        let (alpha, beta, gamma, delta) = (
            i32::from(wp.abcd[0]),
            i32::from(wp.abcd[1]),
            i32::from(wp.abcd[2]),
            i32::from(wp.abcd[3]),
        );
        let mut emu: Option<Vec<u8>> = None;
        let mut y = 0usize;
        while y < bh4 * v_mul {
            let src_y = by * 4 + ((y + 4) << ss_ver);
            let mat3_y = i64::from(mat[3]) * src_y as i64 + i64::from(mat[0]);
            let mat5_y = i64::from(mat[5]) * src_y as i64 + i64::from(mat[1]);
            let mut x = 0usize;
            while x < bw4 * h_mul {
                let src_x = bx * 4 + ((x + 4) << ss_hor);
                let mvx = ((i64::from(mat[2]) * src_x as i64 + mat3_y) >> ss_hor) as i32;
                let mvy = ((i64::from(mat[4]) * src_x as i64 + mat5_y) >> ss_ver) as i32;
                let dx = (mvx >> 16) - 4;
                let mx = ((mvx & 0xffff) - alpha * 4 - beta * 7) & !0x3f;
                let dy = (mvy >> 16) - 4;
                let my = ((mvy & 0xffff) - gamma * 4 - delta * 4) & !0x3f;

                if dx < 3 || dx + 12 > width as i32 || dy < 3 || dy + 12 > height as i32 {
                    let mut buf = core::mem::take(&mut self.scratch.emu_edge);
                    crate::dsp::profil_olc!(
                        McEmuEdge,
                        (mc.emu_edge)(
                            15,
                            15,
                            width,
                            height,
                            dx - 3,
                            dy - 3,
                            &mut buf,
                            32 * bpp,
                            &refp.data[plane],
                            refp.stride[plane],
                            bd,
                            bmax,
                        )
                    );
                    if dsp_trace().frame_ok(self.frame_no) {
                        let hash = pred_hash(&buf, 0, 32 * bpp, 15, 15, bpp);
                        emu_trace_line(
                            self.frame_no,
                            self.bx,
                            self.by,
                            15,
                            15,
                            width,
                            height,
                            dx - 3,
                            dy - 3,
                            hash,
                        );
                    }
                    src_stride = 32 * bpp;
                    emu = Some(buf);
                } else {
                    src_stride = refp.stride[plane];
                    src_off = (dy as usize - 3) * refp.stride[plane] + (dx as usize - 3) * bpp;
                }
                let src: &[u8] = match &emu {
                    Some(buf) => buf,
                    None => &refp.data[plane][src_off..],
                };
                // `dsp_trace.c` `W`: `trace_ref_slot` referans yuvasını ve
                // kaynak **merkez** konumunu yazar; emu tamponunda yuva 255.
                let (trace_refslot, trace_wsx, trace_wsy) = if emu.is_some() {
                    (255u32, 0u32, 0u32)
                } else {
                    // `trace_ref_slot`: `src` hangi `refp[]` yuvasındaysa o.
                    // (Aynı kare birden çok yuvada olabilir; dav1d ilk eşleşmeyi
                    // tutar.)
                    let slot = (0..8usize).find(|&i| {
                        self.refs
                            .get(i)
                            .and_then(|r| r.as_ref())
                            .is_some_and(|r| core::ptr::eq(refp, &r.pic))
                    });
                    (
                        slot.map_or(255u32, |i| i as u32),
                        u32::try_from(dx).unwrap_or(0),
                        u32::try_from(dy).unwrap_or(0),
                    )
                };
                match &mut out {
                    McOut::Pic {
                        plane,
                        x: px,
                        y: py,
                    } => {
                        let dst_off = self.pic.off(*plane, *px + x, *py + y);
                        let stride = self.pic.stride[*plane];
                        crate::dsp::profil_olc!(
                            McWarp8x8,
                            (mc.warp8x8)(
                                &mut self.pic.data[*plane][dst_off..],
                                stride,
                                src,
                                src_stride,
                                &wp.abcd,
                                mx,
                                my,
                                bd,
                                bmax,
                            )
                        );
                        if dsp_trace().frame_ok(self.frame_no) {
                            let hash =
                                pred_hash(&self.pic.data[*plane], dst_off, stride, 8, 8, bpp);
                            warp_trace_line(
                                self.frame_no,
                                "warp8x8",
                                self.bx,
                                self.by,
                                hash,
                                i64::try_from(*px + x).unwrap_or(-1),
                                i64::try_from(*py + y).unwrap_or(-1),
                                *plane,
                                mx,
                                my,
                                trace_refslot,
                                trace_wsx,
                                trace_wsy,
                                wp.abcd,
                                wp.matrix,
                            );
                        }
                    }
                    McOut::Prep { tmp } => {
                        let dstride = bw4 * h_mul;
                        crate::dsp::profil_olc!(
                            McWarp8x8t,
                            (mc.warp8x8t)(
                                &mut tmp[y * dstride + x..],
                                dstride,
                                src,
                                src_stride,
                                &wp.abcd,
                                mx,
                                my,
                                bd,
                                bmax,
                            )
                        );
                        if dsp_trace().frame_ok(self.frame_no) {
                            let hash = pred_hash_i16(tmp, y * dstride + x, dstride, 8, 8);
                            warp_trace_line(
                                self.frame_no,
                                "warpt8x8",
                                self.bx,
                                self.by,
                                hash,
                                -1,
                                -1,
                                0,
                                mx,
                                my,
                                trace_refslot,
                                trace_wsx,
                                trace_wsy,
                                wp.abcd,
                                wp.matrix,
                            );
                        }
                    }
                    McOut::Bytes { dst, stride, off } => {
                        let dstride = *stride;
                        crate::dsp::profil_olc!(
                            McWarp8x8,
                            (mc.warp8x8)(
                                &mut dst[*off + y * dstride + x * bpp..],
                                dstride,
                                src,
                                src_stride,
                                &wp.abcd,
                                mx,
                                my,
                                bd,
                                bmax,
                            )
                        );
                        if dsp_trace().frame_ok(self.frame_no) {
                            let hash =
                                pred_hash(dst, *off + y * dstride + x * bpp, dstride, 8, 8, bpp);
                            warp_trace_line(
                                self.frame_no,
                                "warp8x8",
                                self.bx,
                                self.by,
                                hash,
                                -1,
                                -1,
                                0,
                                mx,
                                my,
                                trace_refslot,
                                trace_wsx,
                                trace_wsy,
                                wp.abcd,
                                wp.matrix,
                            );
                        }
                    }
                }
                // Emu kenar tamponu her 8x8'den sonra iade edilir; aksi
                // hâlde sonraki 8x8 `take` ile boş tampon alır.
                if let Some(buf) = emu.take() {
                    self.scratch.emu_edge = buf;
                }
                x += 8;
            }
            y += 8;
        }
        Ok(())
    }

    /// `obmc()` (`recon_tmpl.c:1052-1113`).
    #[allow(clippy::too_many_arguments)]
    fn obmc_block(
        &mut self,
        plane: usize,
        dst: &mut [u8],
        dst_stride: usize,
        bs: u8,
        bx4: usize,
        by4: usize,
        w4: usize,
        h4: usize,
    ) -> Result<()> {
        let b_dim = &t::BLOCK_DIMENSIONS[usize::from(bs)];
        let ss_hor = if plane != 0 { self.ss_hor } else { 0 };
        let ss_ver = if plane != 0 { self.ss_ver } else { 0 };
        let h_mul = 4 >> ss_hor;
        let v_mul = 4 >> ss_ver;
        let bpp = self.bd.bytes_per_sample();
        let mut lap = core::mem::take(&mut self.scratch.lap);
        let lw = 128usize;
        if self.by > self.row_start
            && (plane == 0 || usize::from(b_dim[0]) * h_mul + usize::from(b_dim[1]) * v_mul >= 16)
        {
            let mut x = 0usize;
            let mut i = 0usize;
            let max_i = usize::from(b_dim[2]).min(4);
            while x < w4 && i < max_i {
                let a_r = self.refmvs.block_at(self.bx + x + 1, self.by - 1);
                let a_dim = &t::BLOCK_DIMENSIONS[usize::from(a_r.bs)];
                let step4 = usize::from(a_dim[0]).clamp(2, 16);
                if a_r.ref_pair[0] > 0 {
                    let ow4 = step4.min(usize::from(b_dim[0]));
                    let oh4 = usize::from(b_dim[1]).min(16) >> 1;
                    let ai = self.bx >> 5;
                    let f2d = t::FILTER_2D[usize::from(
                        self.blocks.above[self.above_base + ai].filter[1][bx4 + x + 1],
                    )][usize::from(self.blocks.above[self.above_base + ai].filter[0][bx4 + x + 1])];
                    self.mc_block(
                        plane,
                        usize::from(a_r.ref_pair[0] as u8 - 1),
                        self.bx + x,
                        self.by,
                        ow4,
                        (oh4 * 3 + 3) >> 2,
                        a_r.mv[0],
                        f2d,
                        McOut::Bytes {
                            dst: &mut lap,
                            stride: ow4 * h_mul * bpp,
                            off: 0,
                        },
                    )?;
                    crate::dsp::profil_olc!(
                        McBlendH,
                        (self.bdsp.mc().blend_h)(
                            &mut dst[x * h_mul * bpp..],
                            dst_stride,
                            &lap,
                            h_mul * ow4,
                            v_mul * oh4,
                            self.bd,
                            self.bd.max_sample() as u16,
                        )
                    );
                    if dsp_trace().frame_ok(self.frame_no) {
                        let hash = pred_hash(
                            dst,
                            x * h_mul * bpp,
                            dst_stride,
                            h_mul * ow4,
                            v_mul * oh4,
                            bpp,
                        );
                        blend_trace_line(
                            self.frame_no,
                            "blendh",
                            self.bx,
                            self.by,
                            h_mul * ow4,
                            v_mul * oh4,
                            hash,
                        );
                    }
                    i += 1;
                }
                x += step4;
            }
        }
        if self.bx > self.col_start {
            let mut y = 0usize;
            let mut i = 0usize;
            let max_i = usize::from(b_dim[3]).min(4);
            while y < h4 && i < max_i {
                // dav1d `recon_tmpl.c:1093`: `&r[y + 1][t->bx - 1]` — yalnız
                // tek (odd) bloklar örtüşme için dikkate alınır; komşu **bir
                // satır aşağıdan** okunur (`f2d` de `by4 + y + 1`'den gelir).
                let l_r = self.refmvs.block_at(self.bx - 1, self.by + y + 1);
                let l_dim = &t::BLOCK_DIMENSIONS[usize::from(l_r.bs)];
                let step4 = usize::from(l_dim[1]).clamp(2, 16);
                if l_r.ref_pair[0] > 0 {
                    let ow4 = usize::from(b_dim[0]).min(16) >> 1;
                    let oh4 = step4.min(usize::from(b_dim[1]));
                    let f2d = t::FILTER_2D[usize::from(self.blocks.left.filter[1][by4 + y + 1])]
                        [usize::from(self.blocks.left.filter[0][by4 + y + 1])];
                    self.mc_block(
                        plane,
                        usize::from(l_r.ref_pair[0] as u8 - 1),
                        self.bx,
                        self.by + y,
                        ow4,
                        oh4,
                        l_r.mv[0],
                        f2d,
                        McOut::Bytes {
                            dst: &mut lap,
                            stride: h_mul * ow4 * bpp,
                            off: 0,
                        },
                    )?;
                    crate::dsp::profil_olc!(
                        McBlendV,
                        (self.bdsp.mc().blend_v)(
                            &mut dst[y * v_mul * dst_stride..],
                            dst_stride,
                            &lap,
                            h_mul * ow4,
                            v_mul * oh4,
                            self.bd,
                            self.bd.max_sample() as u16,
                        )
                    );
                    if dsp_trace().frame_ok(self.frame_no) {
                        let hash = pred_hash(
                            dst,
                            y * v_mul * dst_stride,
                            dst_stride,
                            h_mul * ow4,
                            v_mul * oh4,
                            bpp,
                        );
                        blend_trace_line(
                            self.frame_no,
                            "blendv",
                            self.bx,
                            self.by,
                            h_mul * ow4,
                            v_mul * oh4,
                            hash,
                        );
                    }
                    i += 1;
                }
                y += step4;
            }
        }
        self.scratch.lap = lap;
        let _ = (lw, bpp);
        Ok(())
    }

    /// Interintra birleştirme (`recon_tmpl.c:1665-1703` / `1860-1900`).
    #[allow(clippy::too_many_arguments)]
    fn interintra_plane(
        &mut self,
        plane: usize,
        b: &Av1Block,
        bs: u8,
        x4: usize,
        y4: usize,
        bw4: usize,
        bh4: usize,
        dst_off: usize,
        dst_stride: usize,
    ) -> Result<()> {
        let bpp = self.bd.bytes_per_sample();
        let bd = self.bd;
        let bmax = bd.max_sample() as u16;
        let ss_hor = if plane != 0 { self.ss_hor } else { 0 };
        let ss_ver = if plane != 0 { self.ss_ver } else { 0 };
        let (tw, th) = (bw4, bh4);
        let mut edge = core::mem::take(&mut self.scratch.edge);
        let mode_in = if b.interintra_mode == II_SMOOTH_PRED {
            SMOOTH_PRED
        } else {
            b.interintra_mode
        };
        let prepared = prepare_intra_edges(
            x4,
            self.bx > self.col_start,
            y4,
            self.by > self.row_start,
            self.col_end >> ss_hor,
            self.row_end >> ss_ver,
            0,
            &self.pic.data[plane],
            dst_off,
            dst_stride,
            mode_in,
            0,
            tw,
            th,
            false,
            &mut edge,
            32 * bpp,
            bd,
        );
        let mut tmp = core::mem::take(&mut self.scratch.interintra);
        let ipred = self.bdsp.ipred();
        crate::dsp::profil_olc!(
            IntraPred,
            ipred.intra_pred[usize::from(prepared.mode)](
                &mut tmp,
                tw * 4 * bpp,
                Edge::new(&edge, 32 * bpp),
                tw * 4,
                th * 4,
                0,
                0,
                0,
                bd,
                bmax,
            )
        );
        if dsp_trace().frame_ok(self.frame_no) {
            let hash = pred_hash(&tmp, 0, tw * 4 * bpp, tw * 4, th * 4, bpp);
            let (au, lu, ai_ctx, li) = self.ipred_ctx();
            crate::trace_eprintln!(
                "P {} {} {} {} {} {} {:016x} {:016x} 0 0 0 {} {} {} {}",
                self.frame_no,
                prepared.mode,
                self.bx,
                self.by,
                tw * 4,
                th * 4,
                hash,
                edge_hash_used(
                    &edge,
                    32 * bpp,
                    tw * 4,
                    th * 4,
                    usize::from(prepared.mode),
                    bpp
                ),
                au,
                lu,
                ai_ctx,
                li
            );
        }
        let masks = crate::decode::wedge::Masks::get();
        let chroma_idx = if !self.has_chroma {
            0
        } else {
            match self.chroma {
                crate::frame::ChromaFormat::C420 => 2,
                crate::frame::ChromaFormat::C422 => 1,
                _ => 0,
            }
        };
        let cidx = if plane == 0 { 0 } else { chroma_idx };
        let mask = masks.interintra(
            cidx,
            usize::from(bs),
            b.interintra_type == INTER_INTRA_BLEND,
            usize::from(b.interintra_mode),
            usize::from(b.wedge_idx),
        );
        let mc = self.bdsp.mc();
        crate::dsp::profil_olc!(
            McBlend,
            (mc.blend)(
                &mut self.pic.data[plane][dst_off..],
                dst_stride,
                &tmp,
                tw * 4,
                th * 4,
                mask,
                bd,
                bmax,
            )
        );
        if dsp_trace().frame_ok(self.frame_no) {
            let hash = pred_hash(
                &self.pic.data[plane],
                dst_off,
                dst_stride,
                tw * 4,
                th * 4,
                bpp,
            );
            blend_trace_line(
                self.frame_no,
                "blend",
                self.bx,
                self.by,
                tw * 4,
                th * 4,
                hash,
            );
        }
        self.scratch.edge = edge;
        self.scratch.interintra = tmp;
        Ok(())
    }

    /// `recon_b_inter` (`recon_tmpl.c:1557-2040`) — tahmin + katsayılar.
    fn recon_b_inter_real(
        &mut self,
        msac: &mut MsacContext<'_>,
        cdf: &mut CdfContext,
        bs: u8,
        b: &Av1Block,
    ) -> Result<()> {
        if dsp_trace().frame_ok(self.frame_no) {
            block_trace_line(self.frame_no, self.bx, self.by, bs, b);
        }
        if b.comp_type != COMP_INTER_NONE {
            self.recon_b_inter_compound(bs, b)?;
        } else {
            self.recon_b_inter_single(bs, b)?;
            self.tl_4x4_filter = b.filter2d;
        }
        self.recon_b_inter_coefs(msac, cdf, bs, b)
    }

    /// Compound tahmin (`recon_tmpl.c:1780-1905`).
    fn recon_b_inter_compound(&mut self, bs: u8, b: &Av1Block) -> Result<()> {
        let b_dim = &t::BLOCK_DIMENSIONS[usize::from(bs)];
        let bw4 = usize::from(b_dim[0]);
        let bh4 = usize::from(b_dim[1]);
        let ss_hor = self.ss_hor;
        let ss_ver = self.ss_ver;
        let bmax = self.bd.max_sample() as u16;
        let bd = self.bd;
        let bpp = self.bd.bytes_per_sample();
        let has_chroma = self.has_chroma;
        let lw = bw4 * 4;
        let lh = bh4 * 4;
        let n = lw * lh;
        let mut tmp = core::mem::take(&mut self.scratch.compinter);
        let mc = self.bdsp.mc();

        // --- luma ---------------------------------------------------------
        for i in 0..2 {
            let ref_idx = if i == 0 { b.ref0 } else { b.ref1 };
            let slot = usize::from(self.hdr.refidx[usize::from(ref_idx as u8)] as u8);
            let use_warp = b.inter_mode == 6 && self.gmv_warp_allowed[usize::from(ref_idx as u8)];
            if use_warp {
                let wp = WarpParams {
                    matrix: self.hdr.gmv[usize::from(ref_idx as u8)].matrix,
                    abcd: self.gmv_warp[usize::from(ref_idx as u8)],
                };
                self.warp_affine_block(
                    0,
                    slot,
                    self.bx,
                    self.by,
                    bw4,
                    bh4,
                    &wp,
                    McOut::Prep {
                        tmp: &mut tmp[i * n..(i + 1) * n],
                    },
                )?;
            } else {
                self.mc_block(
                    0,
                    usize::from(ref_idx as u8),
                    self.bx,
                    self.by,
                    bw4,
                    bh4,
                    b.mv[i],
                    b.filter2d,
                    McOut::Prep {
                        tmp: &mut tmp[i * n..(i + 1) * n],
                    },
                )?;
            }
        }
        let dst_x = self.bx * 4;
        let dst_y = self.by * 4;
        let dst_off = self.pic.off(0, dst_x, dst_y);
        let stride = self.pic.stride[0];
        // dav1d `recon_tmpl.c:1843`: luma `w_mask` çağrısı **kroma düzen
        // indeksiyle** yapılır (`w_mask[chr_layout_idx]`) ve `ss_hor/ss_ver`
        // bayrakları maskeyi **kroma çözünürlüğünde** üretir
        // (`mc_tmpl.c:724-768`). Rust önce `w_mask[0]` + (false,false) ile
        // tam çözünürlüklü maske üretiyordu; kroma `mask` çağrısı bu luma
        // maskesinin ilk `cw*ch` baytını okuyunca **yalnız kroma** yanlış
        // çıkıyordu (T40: `test933` k1/k3, tümü COMP_INTER_SEG).
        let chroma_idx = match self.chroma {
            crate::frame::ChromaFormat::C420 => 2,
            crate::frame::ChromaFormat::C422 => 1,
            _ => 0,
        };
        let mut seg_mask_used = false;
        match b.comp_type {
            COMP_INTER_AVG => {
                crate::dsp::profil_olc!(
                    McAvg,
                    (mc.avg)(
                        &mut self.pic.data[0][dst_off..],
                        stride,
                        &tmp[..n],
                        &tmp[n..2 * n],
                        lw,
                        lh,
                        bd,
                        bmax,
                    )
                );
                if dsp_trace().frame_ok(self.frame_no) {
                    avg_trace_line(
                        self.frame_no,
                        "avg",
                        self.bx,
                        self.by,
                        lw,
                        lh,
                        0,
                        pred_hash(&self.pic.data[0], dst_off, stride, lw, lh, bpp),
                        hash_i16(&tmp[..n]),
                        hash_i16(&tmp[n..2 * n]),
                    );
                }
            }
            COMP_INTER_WEIGHTED_AVG => {
                let w = self.jnt_weights[usize::from(b.ref0 as u8)][usize::from(b.ref1 as u8)];
                crate::dsp::profil_olc!(
                    McWAvg,
                    (mc.w_avg)(
                        &mut self.pic.data[0][dst_off..],
                        stride,
                        &tmp[..n],
                        &tmp[n..2 * n],
                        lw,
                        lh,
                        i32::from(w),
                        bd,
                        bmax,
                    )
                );
                if dsp_trace().frame_ok(self.frame_no) {
                    avg_trace_line(
                        self.frame_no,
                        "wavg",
                        self.bx,
                        self.by,
                        lw,
                        lh,
                        i32::from(w),
                        pred_hash(&self.pic.data[0], dst_off, stride, lw, lh, bpp),
                        hash_i16(&tmp[..n]),
                        hash_i16(&tmp[n..2 * n]),
                    );
                }
            }
            COMP_INTER_SEG => {
                let mut mask = core::mem::take(&mut self.scratch.seg_mask);
                let sign = i32::from(b.mask_sign);
                let (t1, t2) = if b.mask_sign != 0 {
                    (&tmp[n..2 * n], &tmp[..n])
                } else {
                    (&tmp[..n], &tmp[n..2 * n])
                };
                crate::dsp::profil_olc!(
                    McWMask,
                    (mc.w_mask[chroma_idx])(
                        &mut self.pic.data[0][dst_off..],
                        stride,
                        t1,
                        t2,
                        lw,
                        lh,
                        &mut mask,
                        sign,
                        ss_hor != 0,
                        ss_ver != 0,
                        bd,
                        bmax,
                    )
                );
                if dsp_trace().frame_ok(self.frame_no) {
                    let mn = (lw * lh) >> (usize::from(ss_hor != 0) + usize::from(ss_ver != 0));
                    mask_trace_line(
                        self.frame_no,
                        "wmask",
                        chroma_idx,
                        sign,
                        self.bx,
                        self.by,
                        lw,
                        lh,
                        pred_hash(&self.pic.data[0], dst_off, stride, lw, lh, bpp),
                        hash_i16(t1),
                        hash_i16(t2),
                        pred_hash(&mask, 0, mn, mn, 1, 1),
                    );
                }
                self.scratch.seg_mask = mask;
                seg_mask_used = true;
            }
            COMP_INTER_WEDGE => {
                let masks = crate::decode::wedge::Masks::get();
                let mask = masks.wedge(
                    0,
                    usize::from(bs),
                    usize::from(b.mask_sign),
                    usize::from(b.wedge_idx),
                );
                let (t1, t2) = if b.mask_sign != 0 {
                    (&tmp[n..2 * n], &tmp[..n])
                } else {
                    (&tmp[..n], &tmp[n..2 * n])
                };
                crate::dsp::profil_olc!(
                    McMask,
                    (mc.mask)(
                        &mut self.pic.data[0][dst_off..],
                        stride,
                        t1,
                        t2,
                        lw,
                        lh,
                        mask,
                        bd,
                        bmax,
                    )
                );
                if dsp_trace().frame_ok(self.frame_no) {
                    mask_trace_line(
                        self.frame_no,
                        "mask",
                        0,
                        0,
                        self.bx,
                        self.by,
                        lw,
                        lh,
                        pred_hash(&self.pic.data[0], dst_off, stride, lw, lh, bpp),
                        hash_i16(t1),
                        hash_i16(t2),
                        pred_hash(mask, 0, lw * lh, lw * lh, 1, 1),
                    );
                }
            }
            _ => {}
        }

        // --- kroma --------------------------------------------------------
        if has_chroma {
            let cw = lw >> ss_hor;
            let ch = lh >> ss_ver;
            let cn = cw * ch;
            let seg_mask = if seg_mask_used {
                Some(core::mem::take(&mut self.scratch.seg_mask))
            } else {
                None
            };
            for pl in 1..3 {
                for i in 0..2 {
                    let ref_idx = if i == 0 { b.ref0 } else { b.ref1 };
                    let slot = usize::from(self.hdr.refidx[usize::from(ref_idx as u8)] as u8);
                    // dav1d: `imin(cbw4, cbh4) > 1 && gmv_warp_allowed`
                    let use_warp = b.inter_mode == 6
                        && self.gmv_warp_allowed[usize::from(ref_idx as u8)]
                        && (cw / 4).min(ch / 4) > 1;
                    if use_warp {
                        let wp = WarpParams {
                            matrix: self.hdr.gmv[usize::from(ref_idx as u8)].matrix,
                            abcd: self.gmv_warp[usize::from(ref_idx as u8)],
                        };
                        self.warp_affine_block(
                            pl,
                            slot,
                            self.bx,
                            self.by,
                            bw4,
                            bh4,
                            &wp,
                            McOut::Prep {
                                tmp: &mut tmp[i * cn..(i + 1) * cn],
                            },
                        )?;
                    } else {
                        self.mc_block(
                            pl,
                            usize::from(ref_idx as u8),
                            self.bx,
                            self.by,
                            bw4,
                            bh4,
                            b.mv[i],
                            b.filter2d,
                            McOut::Prep {
                                tmp: &mut tmp[i * cn..(i + 1) * cn],
                            },
                        )?;
                    }
                }
                let dst_x = (self.bx >> ss_hor) * 4;
                let dst_y = (self.by >> ss_ver) * 4;
                let dst_off = self.pic.off(pl, dst_x, dst_y);
                let stride = self.pic.stride[pl];
                match b.comp_type {
                    COMP_INTER_AVG => {
                        crate::dsp::profil_olc!(
                            McAvg,
                            (mc.avg)(
                                &mut self.pic.data[pl][dst_off..],
                                stride,
                                &tmp[..cn],
                                &tmp[cn..2 * cn],
                                cw,
                                ch,
                                bd,
                                bmax,
                            )
                        );
                        if dsp_trace().frame_ok(self.frame_no) {
                            avg_trace_line(
                                self.frame_no,
                                "avg",
                                self.bx,
                                self.by,
                                cw,
                                ch,
                                0,
                                pred_hash(&self.pic.data[pl], dst_off, stride, cw, ch, bpp),
                                hash_i16(&tmp[..cn]),
                                hash_i16(&tmp[cn..2 * cn]),
                            );
                        }
                    }
                    COMP_INTER_WEIGHTED_AVG => {
                        let w =
                            self.jnt_weights[usize::from(b.ref0 as u8)][usize::from(b.ref1 as u8)];
                        crate::dsp::profil_olc!(
                            McWAvg,
                            (mc.w_avg)(
                                &mut self.pic.data[pl][dst_off..],
                                stride,
                                &tmp[..cn],
                                &tmp[cn..2 * cn],
                                cw,
                                ch,
                                i32::from(w),
                                bd,
                                bmax,
                            )
                        );
                        if dsp_trace().frame_ok(self.frame_no) {
                            avg_trace_line(
                                self.frame_no,
                                "wavg",
                                self.bx,
                                self.by,
                                cw,
                                ch,
                                i32::from(w),
                                pred_hash(&self.pic.data[pl], dst_off, stride, cw, ch, bpp),
                                hash_i16(&tmp[..cn]),
                                hash_i16(&tmp[cn..2 * cn]),
                            );
                        }
                    }
                    COMP_INTER_WEDGE => {
                        let masks = crate::decode::wedge::Masks::get();
                        let mask = masks.wedge(
                            chroma_idx,
                            usize::from(bs),
                            usize::from(b.mask_sign),
                            usize::from(b.wedge_idx),
                        );
                        let (t1, t2) = if b.mask_sign != 0 {
                            (&tmp[cn..2 * cn], &tmp[..cn])
                        } else {
                            (&tmp[..cn], &tmp[cn..2 * cn])
                        };
                        crate::dsp::profil_olc!(
                            McMask,
                            (mc.mask)(
                                &mut self.pic.data[pl][dst_off..],
                                stride,
                                t1,
                                t2,
                                cw,
                                ch,
                                mask,
                                bd,
                                bmax,
                            )
                        );
                        if dsp_trace().frame_ok(self.frame_no) {
                            mask_trace_line(
                                self.frame_no,
                                "mask",
                                0,
                                0,
                                self.bx,
                                self.by,
                                cw,
                                ch,
                                pred_hash(&self.pic.data[pl], dst_off, stride, cw, ch, bpp),
                                hash_i16(t1),
                                hash_i16(t2),
                                pred_hash(mask, 0, cw * ch, cw * ch, 1, 1),
                            );
                        }
                    }
                    COMP_INTER_SEG => {
                        let mask = seg_mask
                            .as_deref()
                            .ok_or(Error::InvalidData("fark ağırlıklı compound maskesi yok"))?;
                        let (t1, t2) = if b.mask_sign != 0 {
                            (&tmp[cn..2 * cn], &tmp[..cn])
                        } else {
                            (&tmp[..cn], &tmp[cn..2 * cn])
                        };
                        crate::dsp::profil_olc!(
                            McMask,
                            (mc.mask)(
                                &mut self.pic.data[pl][dst_off..],
                                stride,
                                t1,
                                t2,
                                cw,
                                ch,
                                mask,
                                bd,
                                bmax,
                            )
                        );
                        if dsp_trace().frame_ok(self.frame_no) {
                            mask_trace_line(
                                self.frame_no,
                                "mask",
                                0,
                                0,
                                self.bx,
                                self.by,
                                cw,
                                ch,
                                pred_hash(&self.pic.data[pl], dst_off, stride, cw, ch, bpp),
                                hash_i16(t1),
                                hash_i16(t2),
                                pred_hash(mask, 0, cw * ch, cw * ch, 1, 1),
                            );
                        }
                    }
                    _ => {}
                }
            }
            if let Some(m) = seg_mask {
                self.scratch.seg_mask = m;
            }
        }
        self.scratch.compinter = tmp;
        Ok(())
    }

    /// Tek referans tahmini + interintra/OBMC/warp (`recon_tmpl.c:1567-1780`).
    fn recon_b_inter_single(&mut self, bs: u8, b: &Av1Block) -> Result<()> {
        let b_dim = &t::BLOCK_DIMENSIONS[usize::from(bs)];
        let bw4 = usize::from(b_dim[0]);
        let bh4 = usize::from(b_dim[1]);
        let bx4 = self.bx & 31;
        let by4 = self.by & 31;
        let ss_hor = self.ss_hor;
        let ss_ver = self.ss_ver;
        let bmax = self.bd.max_sample() as u16;
        let bd = self.bd;
        let has_chroma = self.has_chroma
            && (bw4 > ss_hor || self.bx & 1 != 0)
            && (bh4 > ss_ver || self.by & 1 != 0);
        let w4 = bw4.min(self.bw - self.bx);
        let h4 = bh4.min(self.bh - self.by);
        let ref_slot = usize::from(self.hdr.refidx[usize::from(b.ref0 as u8)] as u8);
        let warp_ok = bw4.min(bh4) > 1
            && ((b.inter_mode == GLOBALMV && self.gmv_warp_allowed[usize::from(b.ref0 as u8)])
                || (b.motion_mode == MM_WARP && b.warp.is_some()));
        let warp_params = if b.motion_mode == MM_WARP {
            b.warp
        } else if b.inter_mode == GLOBALMV && self.gmv_warp_allowed[usize::from(b.ref0 as u8)] {
            Some(WarpParams {
                matrix: self.hdr.gmv[usize::from(b.ref0 as u8)].matrix,
                abcd: self.gmv_warp[usize::from(b.ref0 as u8)],
            })
        } else {
            None
        };

        // --- luma ---------------------------------------------------------
        if warp_ok {
            let wp = warp_params.ok_or(Error::InvalidData("warp parametresi yok"))?;
            self.warp_affine_block(
                0,
                ref_slot,
                self.bx,
                self.by,
                bw4,
                bh4,
                &wp,
                McOut::Pic {
                    plane: 0,
                    x: self.bx * 4,
                    y: self.by * 4,
                },
            )?;
        } else {
            let dst_x = self.bx * 4;
            let dst_y = self.by * 4;
            self.mc_block(
                0,
                usize::from(b.ref0 as u8),
                self.bx,
                self.by,
                bw4,
                bh4,
                b.mv[0],
                b.filter2d,
                McOut::Pic {
                    plane: 0,
                    x: dst_x,
                    y: dst_y,
                },
            )?;
            if b.motion_mode == MM_OBMC {
                let dst_off = self.pic.off(0, dst_x, dst_y);
                let stride = self.pic.stride[0];
                let mut dst = core::mem::take(&mut self.pic.data[0]);
                let dst_slice = &mut dst[dst_off..];
                self.obmc_block(0, dst_slice, stride, bs, bx4, by4, w4, h4)?;
                self.pic.data[0] = dst;
            }
        }
        if b.interintra_type != 0 {
            let dst_off = self.pic.off(0, self.bx * 4, self.by * 4);
            let stride = self.pic.stride[0];
            self.interintra_plane(0, b, bs, self.bx, self.by, bw4, bh4, dst_off, stride)?;
        }

        if !has_chroma {
            return Ok(());
        }

        // --- kroma --------------------------------------------------------
        let mut is_sub8x8 = bw4 == ss_hor || bh4 == ss_ver;
        if is_sub8x8 {
            if bw4 == 1 {
                is_sub8x8 &= self.refmvs.block_at(self.bx - 1, self.by).ref_pair[0] > 0;
            }
            if bh4 == ss_ver {
                is_sub8x8 &= self.refmvs.block_at(self.bx, self.by - 1).ref_pair[0] > 0;
            }
            if bw4 == 1 && bh4 == ss_ver {
                is_sub8x8 &= self.refmvs.block_at(self.bx - 1, self.by - 1).ref_pair[0] > 0;
            }
        }
        if is_sub8x8 {
            // dav1d `recon_tmpl.c:1636-1712` — komşu MV'leriyle kroma.
            self.tools.sub8x8_chroma += 1;
            let mut h_off = 0usize;
            let mut v_off = 0usize;
            let base_x = (self.bx >> ss_hor) * 4;
            let base_y = (self.by >> ss_ver) * 4;
            if bw4 == 1 && bh4 == ss_ver {
                let nb = self.refmvs.block_at(self.bx - 1, self.by - 1);
                for pl in 1..3 {
                    self.mc_chroma_ref(
                        pl,
                        nb.ref_pair[0] - 1,
                        self.bx - 1,
                        self.by - 1,
                        bw4,
                        bh4,
                        nb.mv[0],
                        self.tl_4x4_filter,
                        base_x,
                        base_y,
                    )?;
                }
                v_off = 2;
                h_off = 2;
            }
            if bw4 == 1 {
                let nb = self.refmvs.block_at(self.bx - 1, self.by);
                let left_filter = t::FILTER_2D[usize::from(self.blocks.left.filter[1][by4])]
                    [usize::from(self.blocks.left.filter[0][by4])];
                for pl in 1..3 {
                    self.mc_chroma_ref(
                        pl,
                        nb.ref_pair[0] - 1,
                        self.bx - 1,
                        self.by,
                        bw4,
                        bh4,
                        nb.mv[0],
                        left_filter,
                        base_x,
                        base_y + v_off,
                    )?;
                }
                h_off = 2;
            }
            if bh4 == ss_ver {
                let nb = self.refmvs.block_at(self.bx, self.by - 1);
                let ai = self.bx >> 5;
                let top_filter = t::FILTER_2D
                    [usize::from(self.blocks.above[self.above_base + ai].filter[1][bx4])]
                    [usize::from(self.blocks.above[self.above_base + ai].filter[0][bx4])];
                for pl in 1..3 {
                    self.mc_chroma_ref(
                        pl,
                        nb.ref_pair[0] - 1,
                        self.bx,
                        self.by - 1,
                        bw4,
                        bh4,
                        nb.mv[0],
                        top_filter,
                        base_x + h_off,
                        base_y,
                    )?;
                }
                v_off = 2;
            }
            for pl in 1..3 {
                self.mc_block(
                    pl,
                    usize::from(b.ref0 as u8),
                    self.bx,
                    self.by,
                    bw4,
                    bh4,
                    b.mv[0],
                    b.filter2d,
                    McOut::Pic {
                        plane: pl,
                        x: base_x + h_off,
                        y: base_y + v_off,
                    },
                )?;
            }
        } else {
            let cbw4 = bw4 << usize::from(bw4 == ss_hor);
            let cbh4 = bh4 << usize::from(bh4 == ss_ver);
            // dav1d `recon_tmpl.c:1717`: kroma warp'ı yalnız
            // `imin(cbw4, cbh4) > 1` iken (yani ≥8 kroma pikseli) çağrılır;
            // `warp_affine` 8×8 bloklar yazar ve küçük kroma bloğunda
            // tampon sonunu aşar.
            let cbw4_ctx = (bw4 + ss_hor) >> ss_hor;
            let cbh4_ctx = (bh4 + ss_ver) >> ss_ver;
            let cwarp_ok = (bw4.min(bh4) > 1)
                && cbw4_ctx.min(cbh4_ctx) > 1
                && (((b.inter_mode == GLOBALMV)
                    && self.gmv_warp_allowed[usize::from(b.ref0 as u8)])
                    || (b.motion_mode == MM_WARP && b.warp.is_some()));
            for pl in 1..3 {
                if cwarp_ok {
                    let wp = warp_params.ok_or(Error::InvalidData("warp parametresi yok"))?;
                    self.warp_affine_block(
                        pl,
                        ref_slot,
                        self.bx,
                        self.by,
                        bw4,
                        bh4,
                        &wp,
                        McOut::Pic {
                            plane: pl,
                            x: (self.bx >> ss_hor) * 4,
                            y: (self.by >> ss_ver) * 4,
                        },
                    )?;
                } else {
                    let dst_x = (self.bx >> ss_hor) * 4;
                    let dst_y = (self.by >> ss_ver) * 4;
                    self.mc_block(
                        pl,
                        usize::from(b.ref0 as u8),
                        self.bx & !ss_hor,
                        self.by & !ss_ver,
                        cbw4,
                        cbh4,
                        b.mv[0],
                        b.filter2d,
                        McOut::Pic {
                            plane: pl,
                            x: dst_x,
                            y: dst_y,
                        },
                    )?;
                    if b.motion_mode == MM_OBMC {
                        let dst_off = self.pic.off(pl, dst_x, dst_y);
                        let stride = self.pic.stride[pl];
                        let mut dst = core::mem::take(&mut self.pic.data[pl]);
                        let dst_slice = &mut dst[dst_off..];
                        self.obmc_block(pl, dst_slice, stride, bs, bx4, by4, w4, h4)?;
                        self.pic.data[pl] = dst;
                    }
                }
            }
            if b.interintra_type != 0 {
                let cbw4 = (bw4 + ss_hor) >> ss_hor;
                let cbh4 = (bh4 + ss_ver) >> ss_ver;
                for pl in 1..3 {
                    let dst_off =
                        self.pic
                            .off(pl, (self.bx >> ss_hor) * 4, (self.by >> ss_ver) * 4);
                    let stride = self.pic.stride[pl];
                    self.interintra_plane(
                        pl,
                        b,
                        bs,
                        self.bx >> ss_hor,
                        self.by >> ss_ver,
                        cbw4,
                        cbh4,
                        dst_off,
                        stride,
                    )?;
                }
            }
        }
        let _ = (bmax, bd);
        Ok(())
    }

    /// `derive_warpmv` (`decode.c:264-340`) — komşu eşleşen MV'lerden affine.
    fn derive_warpmv(
        &self,
        bw4: usize,
        bh4: usize,
        masks: &[u64; 2],
        mv: Mv,
    ) -> (crate::bitstream::frame_header::WarpedMotionParams, bool) {
        use crate::bitstream::frame_header::{DEFAULT_WM_PARAMS, WarpedMotionParams};
        let mut pts = [[[0i32; 2]; 2]; 8];
        let mut np = 0usize;
        let dim = |bs: u8| -> [u8; 4] { t::BLOCK_DIMENSIONS[usize::from(bs)] };
        let add_sample = |pts: &mut [[[i32; 2]; 2]; 8],
                          np: &mut usize,
                          dx: i32,
                          dy: i32,
                          sx: i32,
                          sy: i32,
                          b: crate::decode::refmvs::RefBlock| {
            let d = dim(b.bs);
            pts[*np][0][0] = 16 * (2 * dx + sx * i32::from(d[0])) - 8;
            pts[*np][0][1] = 16 * (2 * dy + sy * i32::from(d[1])) - 8;
            pts[*np][1][0] = pts[*np][0][0] + i32::from(b.mv[0].x);
            pts[*np][1][1] = pts[*np][0][1] + i32::from(b.mv[0].y);
            *np += 1;
        };
        // dav1d `decode.c:281`: tek üst örnek özel yolu **yalnız alt 32
        // bite** bakar (`(unsigned) masks[0] == 1`). Bit 32 sağ-üst
        // örnektir ve özel yolu kapatmaz. 64 bit karşılaştırma sağ-üst
        // bit set iken genel yola düşürüyordu (`off = 0` → örnek x = 248;
        // dav1d'de `off = bx & (aw4-1)` → x = -8).
        if warp_uses_single_top_sample(masks) {
            let b = self.refmvs.block_at(self.bx, self.by - 1);
            let off = (self.bx & (usize::from(dim(b.bs)[0]) - 1)) as i32;
            add_sample(&mut pts, &mut np, -off, 0, 1, -1, b);
        } else {
            let mut off = 0usize;
            let mut xmask = masks[0] as u32;
            while np < 8 && xmask != 0 {
                let tz = xmask.trailing_zeros() as usize;
                off += tz;
                xmask >>= tz;
                let b = self.refmvs.block_at(self.bx + off, self.by - 1);
                add_sample(&mut pts, &mut np, off as i32, 0, 1, -1, b);
                xmask &= !1;
            }
        }
        if np < 8 && masks[1] == 1 {
            let b = self.refmvs.block_at(self.bx - 1, self.by);
            let off = (self.by & (usize::from(dim(b.bs)[1]) - 1)) as i32;
            add_sample(&mut pts, &mut np, 0, -off, -1, 1, b);
        } else {
            let mut off = 0usize;
            let mut ymask = masks[1] as u32;
            while np < 8 && ymask != 0 {
                let tz = ymask.trailing_zeros() as usize;
                off += tz;
                ymask >>= tz;
                let b = self.refmvs.block_at(self.bx - 1, self.by + off);
                add_sample(&mut pts, &mut np, 0, off as i32, -1, 1, b);
                ymask &= !1;
            }
        }
        if np < 8 && (masks[1] >> 32) != 0 {
            let b = self.refmvs.block_at(self.bx - 1, self.by - 1);
            add_sample(&mut pts, &mut np, 0, 0, -1, -1, b);
        }
        if np < 8 && (masks[0] >> 32) != 0 {
            let b = self.refmvs.block_at(self.bx + bw4, self.by - 1);
            add_sample(&mut pts, &mut np, bw4 as i32, 0, 1, -1, b);
        }
        // MV farkı eşiğine göre örnek seçimi (`decode.c:315-338`).
        let thresh = 4 * (bw4.max(bh4) as i32).clamp(4, 28);
        let mut mvd = [0i32; 8];
        let mut ret = 0usize;
        for i in 0..np {
            mvd[i] = (pts[i][1][0] - pts[i][0][0] - i32::from(mv.x)).abs()
                + (pts[i][1][1] - pts[i][0][1] - i32::from(mv.y)).abs();
            if mvd[i] > thresh {
                mvd[i] = -1;
            } else {
                ret += 1;
            }
        }
        if ret == 0 {
            ret = 1;
        } else {
            select_warp_samples(&mut pts, np, ret, &mut mvd);
        }
        let mut wm: WarpedMotionParams = DEFAULT_WM_PARAMS;
        let ok = warpv::find_affine_int(
            &pts[..ret],
            ret,
            bw4,
            bh4,
            (i32::from(mv.x), i32::from(mv.y)),
            &mut wm,
            self.bx,
            self.by,
        );
        // dav1d `dsp_trace.c` `Q`: `ret` örnek + sonuç + `matrix[2,3,5]`
        // (seçimden **sonraki** ilk `ret` örnek; Rust önceden seçimden önceki
        // `np` örneği basıyordu).
        if dsp_trace().frame_ok(self.frame_no) {
            let mut line = format!(
                "Q {} {} {} {} {} {}",
                self.frame_no,
                self.bx,
                self.by,
                ret,
                i32::from(mv.x),
                i32::from(mv.y)
            );
            for p in pts.iter().take(ret) {
                let _ = core::fmt::Write::write_fmt(
                    &mut line,
                    format_args!(" {}:{}:{}:{}", p[0][0], p[0][1], p[1][0], p[1][1]),
                );
            }
            let _ = core::fmt::Write::write_fmt(
                &mut line,
                format_args!(
                    " {} {} {} {}",
                    i32::from(ok),
                    wm.matrix[2],
                    wm.matrix[3],
                    wm.matrix[5]
                ),
            );
            crate::trace_eprintln!("{line}");
        }
        if ok {
            (wm, true)
        } else {
            (DEFAULT_WM_PARAMS, false)
        }
    }
}

/// `get_drl_context` (`env.h:437-445`).
fn get_drl_context(stack: &[Candidate; 8], idx: usize) -> usize {
    if stack[idx].weight >= 640 {
        usize::from(stack[idx + 1].weight < 640)
    } else if stack[idx + 1].weight < 640 {
        2
    } else {
        0
    }
}

/// `fix_mv_precision` (`env.h:462-474`).
fn fix_mv_precision(hdr: &FrameHeader, mv: &mut Mv) {
    let fix_int = |mv: &mut Mv| {
        let x = i32::from(mv.x);
        let y = i32::from(mv.y);
        mv.x = ((x - (x >> 15) + 3) & !7) as i16;
        mv.y = ((y - (y >> 15) + 3) & !7) as i16;
    };
    if hdr.force_integer_mv {
        fix_int(mv);
    } else if !hdr.hp {
        let x = i32::from(mv.x);
        let y = i32::from(mv.y);
        mv.x = ((x - (x >> 15)) & !1) as i16;
        mv.y = ((y - (y >> 15)) & !1) as i16;
    }
}

/// `dav1d_submit_frame` erken denetimleri (`src/decode.c:3452-3478`): inter
/// karede `primary_ref_frame` ve 7 referansın varlığı, boyut oranı ([1/2, 16])
/// ve piksel biçimi. Bu hatalarda dav1d referans yuvalarını **yenilemez**
/// (yenileme `decode.c:3676-3699` sonrasıdır); çağıran başlık yuvalarını geri
/// yükler ([`crate::api::Decoder`]).
pub(crate) fn validate_frame_refs(
    hdr: &FrameHeader,
    chroma: ChromaFormat,
    bd: BitDepth,
    refs: &[Option<Arc<RefFrame>>; 8],
) -> Result<()> {
    let is_inter = matches!(hdr.frame_type, FrameType::Inter | FrameType::Switch);
    if is_inter {
        for i in 0..7 {
            let slot = usize::from(hdr.refidx[i] as u8);
            let Some(r) = refs.get(slot).and_then(|r| r.as_ref()) else {
                return Err(Error::Unsupported("referans çözülmedi: inter referans"));
            };
            // dav1d `decode.c:3459-3464`: ölçekli referans serbesttir; yalnız
            // boyut oranı [1/2, 16] aralığında olmalıdır (SVC ölçeği).
            let cur_w = hdr.width[0] as usize;
            let cur_h = hdr.height as usize;
            if cur_w * 2 < r.pic.width
                || cur_h * 2 < r.pic.height
                || cur_w > r.pic.width * 16
                || cur_h > r.pic.height * 16
            {
                return Err(Error::InvalidData("ölçekli referans oranı aralık dışında"));
            }
            if r.chroma != chroma || r.bd != bd {
                return Err(Error::InvalidData("referans piksel biçimi uyuşmuyor"));
            }
        }
        if hdr.primary_ref_frame != 7 {
            let pr = usize::from(hdr.primary_ref_frame);
            let slot = usize::from(hdr.refidx[pr] as u8);
            if refs.get(slot).and_then(|r| r.as_ref()).is_none() {
                return Err(Error::Unsupported("referans çözülmedi: primary_ref_frame"));
            }
        }
    }

    Ok(())
}

/// Kare çözme girişi: seçim kuralını sağlayan kareleri çözer.
///
/// `tiles`, karo indeksine göre (satır-major) ham tile verisidir. Intra
/// kareler olduğu gibi, inter/switch kareler **tek referanslı** yolla çözülür.
/// Film grain kare çözümünü etkilemez: grain yalnız çıktı kopyasına
/// uygulanır (`crate::decode::filmgrain`, `Decoder`). Araç reddi yoktur;
/// kalan [`Error::Unsupported`] gerekçeleri yalnız eksik referans
/// ("referans çözülmedi"), `show_existing_frame` (bu yola gelmez) ve geçersiz
/// `seq.hbd`'dir; bu durumda **hiçbir çıktı üretilmez**.
pub(crate) fn decode_frame(
    seq: &SequenceHeader,
    hdr: &FrameHeader,
    tiles: &[&[u8]],
    dsp: &DspDispatch,
    refs: &[Option<Arc<RefFrame>>; 8],
    butce: &Butce,
    pool: &crate::thread::WorkerPool,
) -> Result<DecodedFrame> {
    if hdr.show_existing_frame {
        return Err(Error::Unsupported("show_existing_frame"));
    }
    let n_tiles = usize::from(hdr.tiling.cols) * usize::from(hdr.tiling.rows);
    if tiles.len() != n_tiles {
        return Err(Error::InvalidData(
            "tile verisi sayısı karo sayısına eşit değil",
        ));
    }
    let is_inter = matches!(hdr.frame_type, FrameType::Inter | FrameType::Switch);
    let chroma = match seq.layout.raw() {
        0 => ChromaFormat::Monochrome,
        1 => ChromaFormat::C420,
        2 => ChromaFormat::C422,
        _ => ChromaFormat::C444,
    };
    let bd = match seq.hbd {
        0 => BitDepth::Eight,
        1 => BitDepth::Ten,
        2 => BitDepth::Twelve,
        _ => return Err(Error::Unsupported("geçersiz seq hbd")),
    };
    let bw = ((hdr.width[0] as usize + 7) >> 3) << 1;
    let bh = ((hdr.height as usize + 7) >> 3) << 1;

    // --- referans doğrulaması (dav1d `decode.c:3446-3478`) ----------------
    validate_frame_refs(hdr, chroma, bd, refs)?;

    // --- CDF girişi (dav1d `decode.c:3481-3490`) -------------------------
    let qcat = crate::entropy::cdf::qcat_for_qidx(hdr.quant.yac);
    let cdf_in = if hdr.primary_ref_frame == 7 {
        CdfContext::from_default(qcat)
    } else {
        let pr = usize::from(hdr.primary_ref_frame);
        let slot = usize::from(hdr.refidx[pr] as u8);
        refs.get(slot)
            .and_then(|r| r.as_ref())
            .map(|r| r.cdf.clone())
            .ok_or(Error::Unsupported("referans çözülmedi: primary_ref_frame"))?
    };

    // --- önceki segment haritası (dav1d `decode.c:3617-3651`) -------------
    let has_limit = butce.has_limit();
    let buf_pool = pool.buf_pool();
    let mut work_bufs = buf_pool.take_work_bufs(has_limit);
    let mut prev_segmap: Option<Arc<Vec<u8>>> = None;
    if hdr.segmentation.enabled
        && (hdr.segmentation.temporal || !hdr.segmentation.update_map)
        && hdr.primary_ref_frame != 7
    {
        let pr = usize::from(hdr.primary_ref_frame);
        let slot = usize::from(hdr.refidx[pr] as u8);
        if let Some(r) = refs.get(slot).and_then(|r| r.as_ref()) {
            let stride = ((bw + 31) >> 5) * 32;
            let height = ((bh + 31) >> 5) * 32;
            // dav1d `decode.c:3629-3631`: uygunluk referansın **kodlanan**
            // genişliğiyle (`ref_coded_width`, superres öncesi) sınanır.
            let ref_w = ((r.hdr.width[0] as usize + 7) >> 3) << 1;
            if ref_w == bw && r.bh == bh && r.segmap.len() >= stride * height {
                let old_prev = buf_pool.take_segmap_vec(stride * height, has_limit);
                prev_segmap = Some(Arc::new(
                    butce.reuse_kopya(old_prev, &r.segmap[..stride * height])?,
                ));
            }
        }
    }

    let dt = dsp_trace();
    let trace_active = dt.on || dt.refmvs_trace || dt.lf_blk || cfg!(feature = "msac-trace");
    let parallel_tiles = pool.is_parallel() && n_tiles > 1 && !trace_active;

    let mut dec = Dec::new(
        seq,
        hdr,
        dsp,
        refs,
        prev_segmap,
        &mut work_bufs,
        buf_pool,
        butce,
        parallel_tiles,
    )?;
    if hdr.primary_ref_frame != 7 {
        dec.tools.cdf_load = 1;
    }
    // Yeniden oynatma probu: bu, çözülen kaçıncı karedir (tanılama).
    dec.frame_no = probe::frame_start(
        hdr.width[0] as u32,
        hdr.height as u32,
        [8u8, 10, 12][usize::from(seq.hbd.min(2))],
        match seq.layout.raw() {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => 3,
        },
    );
    // Tanılama: refmvs çağrı izi kare sırası (`AV1OXIDE_REFMVS_TRACE`).
    dec.refmvs.set_frame(dec.frame_no);
    let mut tile_cdfs: Vec<CdfContext> = butce.kapasite(n_tiles)?;
    if parallel_tiles {
        decode_tiles_parallel(&mut dec, hdr, tiles, &cdf_in, &mut tile_cdfs, pool)?;
    } else {
        for tile_row in 0..usize::from(hdr.tiling.rows) {
            for tile_col in 0..usize::from(hdr.tiling.cols) {
                let idx = tile_row * usize::from(hdr.tiling.cols) + tile_col;
                let mut cdf = cdf_in.clone();
                dec.decode_tile(tile_row, tile_col, tiles[idx], &mut cdf)?;
                tile_cdfs.push(cdf);
            }
        }
    }
    // --- deblocking (`recon_tmpl.c:1987-2020`): kare bittikten sonra,
    // SB satırı sırasıyla önce dikey sonra yatay kenarlar. Intra tahmin
    // filtrelenmemiş pikselleri kullandığı için sonuç dav1d ile aynıdır.
    if let Some(lf) = dec.lf.as_mut().filter(|_| !dt.lf_off) {
        lf.apply(&mut dec.pic, hdr, dec.frame_no, dec.bdsp.lf(), pool);
    }
    // LR şerit sınırı satırları deblocking sonrası, CDEF öncesi piksellerdir
    // (`dav1d_copy_lpf`, `lf_apply_tmpl.c:104-`); tam kare yerine yalnız LR
    // açık düzlemlerin 4-satırlık şerit sınırları saklanır (`4.A.1`).
    let mut pre_cdef = match dec.lr.as_ref() {
        Some(lr) => {
            let old_pre = work_bufs
                .lr_cdef_backup
                .take()
                .unwrap_or_else(|| [Vec::new(), Vec::new(), Vec::new()]);
            Some(crate::decode::lr::LrCdefBackup::capture(
                &dec.pic,
                lr.restore_planes,
                old_pre,
                butce,
            )?)
        }
        None => None,
    };
    // --- CDEF (`recon_tmpl.c:2099-2106`): deblocking'den **sonra**.
    // Filtre girdileri deblocking sonrası, CDEF öncesi piksellerdir; CDEF
    // kare sonunda 8 satırlık bantlar hâlinde uygulanır (bkz. `decode::cdef`).
    {
        if !dt.cdef_off {
            let (chroma, bd, frame_no) = (dec.chroma, dec.bd, dec.frame_no);
            crate::decode::cdef::apply(
                &mut dec.pic,
                hdr,
                chroma,
                bd,
                dec.bdsp.cdef(),
                &dec.cdef_state,
                &mut dec.tools,
                frame_no,
                butce,
                pool,
            )?;
        }
    }
    // --- superres (`recon_tmpl.c:2103-2104`): CDEF'ten **sonra**, LR'den önce.
    // Kare yatayda `width[1]`'e büyütülür; LR büyütülmüş karede çalışır ve
    // şerit sınırı satırları da büyütülmüş CDEF öncesi satırlardan gelir
    // (`lf_apply_tmpl.c:64-77`). Referans yuvasına ve çıktıya büyütülmüş kare
    // gider (bkz. `decode::superres`).
    if hdr.width[0] != hdr.width[1] {
        let sr = crate::decode::superres::Superres::new(
            hdr.width[0] as usize,
            hdr.width[1] as usize,
            dec.chroma,
        );
        let kernel = dsp.for_bit_depth(bd).resize();
        let (chroma, bd, bw, height) = (dec.chroma, dec.bd, dec.bw, dec.pic.height);
        let coded_stride = dec.pic.stride;
        let up = crate::decode::superres::upscale_planes(
            &mut dec.pic.data,
            coded_stride,
            height,
            bw,
            &sr,
            chroma,
            bd,
            kernel,
            butce,
            pool,
        )?;
        let up_pre = pre_cdef
            .as_ref()
            .map(|pre| {
                crate::decode::superres::upscale_lr_backup(
                    pre, bw, &sr, chroma, bd, up.stride, kernel, butce,
                )
            })
            .transpose()?;
        crate::decode::superres::trace(
            dec.frame_no,
            &up,
            up_pre.as_ref(),
            dec.lr.as_ref().map_or(0, |l| l.restore_planes),
            height,
            bw,
            dec.sb128,
            &sr,
            chroma,
            bd.bytes_per_sample(),
        );
        // Kodlanan boyuttaki düzlemler burada serbest kalır (defterden düşülür).
        butce.birak(dec.pic.bayt());
        for v in dec.pic.data {
            buf_pool.put_u8_vec(v, has_limit);
        }
        if let Some(pre) = pre_cdef.take() {
            butce.birak(pre.bayt());
            work_bufs.lr_cdef_backup = Some(pre.data);
        }
        dec.pic = up;
        pre_cdef = up_pre;
        dec.tools.sr_frame = 1;
        dec.tools.sr_denom = u64::from(hdr.super_res.width_scale_denominator);
        dec.tools.sr_lr = u64::from(dec.lr.is_some());
    }
    // --- loop restoration (`recon_tmpl.c:2105-2106`): CDEF'ten (ve superres
    // büyütmesinden) **sonra**. Referans yuvasına LR sonrası kare gider.
    if let (Some(lr), Some(pre)) = (dec.lr.as_ref(), pre_cdef.as_mut())
        && !dt.lr_off
    {
        let kernels = crate::decode::lr::LrKernels {
            wiener: dsp.for_bit_depth(bd).lr().wiener,
            sgr: dsp.for_bit_depth(bd).lr().sgr,
        };
        crate::decode::lr::apply(
            &mut dec.pic,
            pre,
            lr,
            dec.sb128,
            dec.bd,
            &kernels,
            &mut dec.tools,
            dec.frame_no,
            butce,
            pool,
        )?;
    }
    if let Some(pre) = pre_cdef.take() {
        butce.birak(pre.bayt());
        if work_bufs.lr_cdef_backup.is_none() {
            work_bufs.lr_cdef_backup = Some(pre.data);
        }
    }
    // --- kare sonu CDF (dav1d `decode.c:3317-3320`) -----------------------
    let cdf_out = if hdr.refresh_context {
        let mut out = cdf_in.clone();
        let idx = usize::from(hdr.tiling.update).min(tile_cdfs.len() - 1);
        out.update(
            &tile_cdfs[idx],
            crate::entropy::cdf::FrameKind::from_frame_type(hdr.frame_type.raw()),
        );
        out
    } else {
        cdf_in.clone()
    };
    // --- temporal MV ızgarası (dav1d `decode.c:3230-3236`) ---------------
    let tmvs_stride = ((hdr.width[0] as usize + 127) & !127) >> 3;
    let ih8 = (hdr.height as usize + 7) >> 3;
    let tmvs_len = tmvs_stride * ih8.max(1);
    let mut tmvs = if !is_inter {
        Vec::new()
    } else {
        let old_tmvs = buf_pool.take_tmvs_vec(tmvs_len, has_limit);
        butce.reuse_vec(old_tmvs, tmvs_len, TemporalBlock::default())?
    };
    if is_inter {
        dec.refmvs.save_temporal(&mut tmvs, tmvs_stride);
    }

    let bpp = dec.bd.bytes_per_sample();
    if probe::enabled() {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut bytes = 0u64;
        let n_planes = dec.frame_plane_count();
        for pl in 0..n_planes {
            let (pw, ph) = if pl == 0 {
                (dec.pic.width, dec.pic.height)
            } else {
                (dec.pic.cw, dec.pic.ch)
            };
            let row_bytes = pw * bpp;
            let stride = dec.pic.stride[pl];
            let plane_data = &dec.pic.data[pl];
            for y in 0..ph {
                let row = &plane_data[y * stride..y * stride + row_bytes];
                for &v in row {
                    hash ^= u64::from(v);
                    hash = hash.wrapping_mul(0x100_0000_01b3);
                }
                bytes += row_bytes as u64;
            }
        }
        probe::frame_end(hash, bytes);
    }
    let mut tools = dec.tools;
    tools.sb128_frame = dec.sb128_frame;
    tools.multi_tile_frame = dec.multi_tile_frame;
    tools.segmentation_frame = dec.segmentation_frame;
    tools.qmatrix_frame = dec.qmatrix_frame;
    tools.temporal_candidate = dec
        .refmvs
        .temporal_hits
        .load(core::sync::atomic::Ordering::Relaxed);
    tools.lr_units = dec.lr_counts.units;
    tools.lr_switchable_units = dec.lr_counts.switchable_units;
    tools.lr_none_units = dec.lr_counts.none_units;
    tools.lr_wiener_units = dec.lr_counts.wiener_units;
    tools.lr_sgr_units = dec.lr_counts.sgr_units;
    if let Some(arc) = dec.prev_segmap.take()
        && let Ok(v) = Arc::try_unwrap(arc)
    {
        buf_pool.put_segmap_vec(v, has_limit);
    }
    work_bufs.lf = dec.lf.take();
    work_bufs.blocks = Some(dec.blocks);
    work_bufs.cdef_state = Some(dec.cdef_state);
    if let Some(lr) = dec.lr.take() {
        work_bufs.lr_mask = lr.mask;
    }
    work_bufs.scratch = Some(dec.scratch);
    if !dec.refmvs.grid.is_empty() {
        work_bufs.refmvs_grid = dec.refmvs.grid;
    }
    if let Ok(rp) = Arc::try_unwrap(dec.refmvs.rp_proj) {
        work_bufs.refmvs_rp_proj = rp;
    }
    buf_pool.put_work_bufs(work_bufs, has_limit);
    let segmap = dec.segmap;
    let (refpoc, bh, pic, chroma, bd) = (dec.refpoc, dec.bh, dec.pic, dec.chroma, dec.bd);
    Ok(DecodedFrame {
        pic,
        chroma,
        bd,
        cdf: cdf_out,
        segmap,
        tmvs,
        tmvs_stride,
        refpoc,
        bh,
        tools,
    })
}

/// Karedeki tüm `(tile_row, tile_col)` karolarını kalıcı `WorkerPool` üzerinde
/// paralel çözer ve sonuçları deterministik `(tile_row, tile_col)` sırasıyla
/// `dec` ve `tile_cdfs` içine birleştirir (`ADR-0004` §3.1, `F8b-2`).
#[allow(clippy::too_many_lines)]
fn decode_tiles_parallel(
    dec: &mut Dec<'_>,
    hdr: &FrameHeader,
    tiles: &[&[u8]],
    cdf_in: &CdfContext,
    tile_cdfs: &mut Vec<CdfContext>,
    pool: &crate::thread::WorkerPool,
) -> Result<()> {
    use crate::sync::Mutex;

    struct TileWork {
        tile_row: usize,
        tile_col: usize,
        row_start: usize,
        row_end_sb4: usize,
        row_end: usize,
        col_start: usize,
        col_end_sb4: usize,
        col_end: usize,
        tile_segmap_stride: usize,
        data: Vec<u8>,
        cdf: CdfContext,
        pic_data: [Vec<u8>; 3],
        pic_stride: [usize; 3],
        pic_x_base: [usize; 3],
        pic_y_base: [usize; 3],
        lf: Option<crate::decode::loopfilter::LfFrame>,
        cdef_state: Option<crate::decode::cdef::CdefState>,
        lr: Option<crate::decode::lr::LrState>,
        lr_counts: crate::decode::lr::LrSyntaxCounts,
        segmap: Vec<u8>,
        refmvs: Option<RefMvs>,
        tools: ToolCounts,
    }

    let n_rows = usize::from(hdr.tiling.rows);
    let n_cols = usize::from(hdr.tiling.cols);
    let keyframe = matches!(hdr.frame_type, FrameType::Key | FrameType::Intra);
    let allow_intrabc = hdr.allow_intrabc;
    let need_refmvs = !keyframe || allow_intrabc;
    let need_segmap = hdr.segmentation.enabled && hdr.segmentation.update_map;
    let max_sb_rows = (dec.bh + dec.sb_step - 1) >> dec.sb_shift;
    let max_sb_cols = (dec.bw + dec.sb_step - 1) >> dec.sb_shift;

    let mut row_bounds_sb4: Vec<(usize, usize, usize)> = Vec::with_capacity(n_rows);
    for tile_row in 0..n_rows {
        let row_start_sb = usize::from(hdr.tiling.row_start_sb[tile_row]);
        let row_end_sb = usize::from(hdr.tiling.row_start_sb[tile_row + 1]).min(max_sb_rows);
        let row_start = row_start_sb << dec.sb_shift;
        let row_end_sb4 = row_end_sb << dec.sb_shift;
        let row_end = row_end_sb4.min(dec.bh);
        row_bounds_sb4.push((row_start, row_end_sb4, row_end));
    }

    let seq_owned = Arc::new(dec.seq.clone());
    let hdr_owned = Arc::new(dec.hdr.clone());
    let dsp_owned = Arc::new(*dec.dsp);
    let refs_owned: Arc<[Option<Arc<RefFrame>>; 8]> = Arc::new(dec.refs.clone());
    let prev_segmap = dec.prev_segmap.as_ref().map(Arc::clone);
    let pic_meta = Arc::new(PicBuf {
        data: [Vec::new(), Vec::new(), Vec::new()],
        ..dec.pic
    });
    let lf_meta = Arc::new(
        dec.lf
            .as_ref()
            .map(crate::decode::loopfilter::LfFrame::meta_template),
    );
    let cdef_meta = Arc::new(crate::decode::cdef::CdefState {
        noskip: Vec::new(),
        noskip_stride: dec.cdef_state.noskip_stride,
        noskip_x_base: 0,
        noskip_y_base: 0,
        idx64: Vec::new(),
        sb64w: dec.cdef_state.sb64w,
        sb64_x_base: 0,
        sb64_y_base: 0,
    });
    let lr_meta = Arc::new(
        dec.lr
            .as_ref()
            .map(crate::decode::lr::LrState::fork_for_tile),
    );
    let refmvs_meta = Arc::new(dec.refmvs.fork_for_tile(TileBounds::default(), false)?);
    let bd = dec.bd;
    let chroma = dec.chroma;
    let ss_hor = dec.ss_hor;
    let ss_ver = dec.ss_ver;
    let has_chroma = dec.has_chroma;
    let bw = dec.bw;
    let bh = dec.bh;
    let w4 = dec.w4;
    let h4 = dec.h4;
    let sb128 = dec.sb128;
    let sb_shift = dec.sb_shift;
    let sb_step = dec.sb_step;
    let sb128w = dec.sb128w;
    let frame_lflvl = dec.frame_lflvl;
    let frame_dq = dec.frame_dq;
    let frame_segmap_stride = dec.frame_segmap_stride;
    let frame_no = dec.frame_no;
    let refpoc = dec.refpoc;
    let refrefpoc = dec.refrefpoc;
    let svc = dec.svc;
    let gmv_warp_allowed = dec.gmv_warp_allowed;
    let gmv_warp = dec.gmv_warp;
    let jnt_weights = dec.jnt_weights;
    let sb128_frame = dec.sb128_frame;
    let multi_tile_frame = dec.multi_tile_frame;
    let segmentation_frame = dec.segmentation_frame;
    let qmatrix_frame = dec.qmatrix_frame;

    // Büyük çok-satırlı karelerde (`Y >= 4 MiB`, `n_rows >= 4`) karoları `tile_row`
    // dalgaları hâlinde çözüp anında birleştir (`5.C.1`): 120 karoluk geçici
    // tamponun tamamını aynı anda tutmak yerine dalga başına ~30 karo tutarak
    // tepe bellek/RSS tüketimini belirgin biçimde düşürür.
    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    let monotonic_rows = n_rows >= 2
        && dec.pic.data[0].len() >= 512 * 1024
        && row_bounds_sb4
            .windows(2)
            .all(|ab| ab[0].1 == ab[1].0 && ab[0].0 <= ab[0].1 && ab[0].2 == ab[1].0);
    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    let wave_rows = if monotonic_rows && n_rows >= 4 && dec.pic.data[0].len() >= 4 * 1024 * 1024 {
        let target_tiles = (pool.n_threads() * 3 / 2).max(16);
        target_tiles.div_ceil(n_cols.max(1)).clamp(1, n_rows)
    } else {
        n_rows
    };
    #[cfg(not(all(feature = "std", not(target_arch = "wasm32"))))]
    let wave_rows = n_rows;

    // Faz 1 işçileri için `(Scratch, Blocks)` yeniden kullanım havuzu (`5.C.1`):
    // tüm dalgalar boyunca paylaşılır; en fazla `n_threads` örnek oluşturulur.
    let tile_scratch_pool: Arc<Mutex<Vec<(Scratch, Blocks)>>> =
        Arc::new(Mutex::new(Vec::with_capacity(pool.n_threads())));

    let mut wave_r_start = 0usize;
    while wave_r_start < n_rows {
        let wave_r_end = (wave_r_start + wave_rows).min(n_rows);
        let wave_r_count = wave_r_end - wave_r_start;
        let mut works: Vec<TileWork> = Vec::with_capacity(wave_r_count * n_cols);
        for (tile_row, &(row_start, row_end_sb4, row_end)) in row_bounds_sb4
            .iter()
            .enumerate()
            .take(wave_r_end)
            .skip(wave_r_start)
        {
            for tile_col in 0..n_cols {
                let col_start_sb = usize::from(hdr.tiling.col_start_sb[tile_col]);
                let col_end_sb =
                    usize::from(hdr.tiling.col_start_sb[tile_col + 1]).min(max_sb_cols);
                let col_start = col_start_sb << dec.sb_shift;
                let col_end_sb4 = col_end_sb << dec.sb_shift;
                let col_end = col_end_sb4.min(dec.bw);
                let idx = tile_row * n_cols + tile_col;
                let tile_segmap_stride = col_end_sb4.saturating_sub(col_start).max(32);
                works.push(TileWork {
                    tile_row,
                    tile_col,
                    row_start,
                    row_end_sb4,
                    row_end,
                    col_start,
                    col_end_sb4,
                    col_end,
                    tile_segmap_stride,
                    data: crate::frame::butce::kopya_vec(tiles[idx])?,
                    cdf: cdf_in.clone(),
                    pic_data: [Vec::new(), Vec::new(), Vec::new()],
                    pic_stride: [0; 3],
                    pic_x_base: [0; 3],
                    pic_y_base: [0; 3],
                    lf: None,
                    cdef_state: None,
                    lr: None,
                    lr_counts: crate::decode::lr::LrSyntaxCounts::default(),
                    segmap: Vec::new(),
                    refmvs: None,
                    tools: ToolCounts::default(),
                });
            }
        }

        let seq_w = Arc::clone(&seq_owned);
        let hdr_w = Arc::clone(&hdr_owned);
        let dsp_w = Arc::clone(&dsp_owned);
        let refs_w = Arc::clone(&refs_owned);
        let prev_seg_w = prev_segmap.as_ref().map(Arc::clone);
        let pic_meta_w = Arc::clone(&pic_meta);
        let lf_meta_w = Arc::clone(&lf_meta);
        let cdef_meta_w = Arc::clone(&cdef_meta);
        let lr_meta_w = Arc::clone(&lr_meta);
        let refmvs_meta_w = Arc::clone(&refmvs_meta);
        let tile_scratch_pool_w = Arc::clone(&tile_scratch_pool);

        pool.execute_tasks(&mut works, move |_wid, _idx, w: &mut TileWork| {
            let pic = pic_meta_w.fork_for_tile(
                w.row_start,
                w.row_end_sb4,
                w.col_start,
                w.col_end_sb4,
                allow_intrabc,
            )?;
            let (scratch, blocks) = match tile_scratch_pool_w.lock().pop() {
                Some((s, mut b)) => {
                    b.reset_for_tile(w.col_start >> 5, w.col_end_sb4.div_ceil(32), keyframe);
                    (s, b)
                }
                None => (
                    Scratch::fork_for_tile(bd)?,
                    Blocks::fork_for_tile(sb128w, keyframe)?,
                ),
            };
            let lf = lf_meta_w
                .as_ref()
                .as_ref()
                .map(|lf| lf.fork_for_tile(w.row_start, w.row_end_sb4, w.col_start, w.col_end_sb4))
                .transpose()?;
            let cdef_state = cdef_meta_w.fork_for_tile(
                w.row_start,
                w.row_end_sb4,
                w.col_start,
                w.col_end_sb4,
            )?;
            let lr = lr_meta_w
                .as_ref()
                .as_ref()
                .map(crate::decode::lr::LrState::fork_for_tile);
            let segmap = if need_segmap {
                let tile_seg_h = w.row_end_sb4.saturating_sub(w.row_start).max(32);
                crate::frame::butce::dolu_vec(w.tile_segmap_stride * tile_seg_h, 0u8)?
            } else {
                Vec::new()
            };
            let refmvs = refmvs_meta_w.fork_for_tile(
                TileBounds {
                    row_start: w.row_start,
                    row_end: w.row_end,
                    col_start: w.col_start,
                    col_end: w.col_end,
                },
                need_refmvs,
            )?;
            let mut tdec = Dec {
                seq: &seq_w,
                hdr: &hdr_w,
                dsp: &dsp_w,
                bdsp: dsp_w.for_bit_depth(bd),
                bd,
                chroma,
                ss_hor,
                ss_ver,
                has_chroma,
                bw,
                bh,
                w4,
                h4,
                sb128,
                sb_shift,
                sb_step,
                pic,
                blocks,
                above_base: 0,
                sb128w,
                bx: 0,
                by: 0,
                col_start: 0,
                col_end: 0,
                row_start: 0,
                row_end: 0,
                last_qidx: i32::from(hdr_w.quant.yac),
                last_delta_lf: [0; 4],
                frame_lflvl,
                lflvl: frame_lflvl,
                lf,
                cdef_state,
                lr,
                lr_counts: w.lr_counts,
                dq: frame_dq,
                frame_dq,
                cdef_idx: [-1; 4],
                scratch,
                segmap,
                segmap_stride: w.tile_segmap_stride,
                segmap_x_base: w.col_start,
                segmap_y_base: w.row_start,
                frame_segmap_stride,
                pal: [[0u16; 8]; 3],
                frame_no,
                refmvs,
                refs: &refs_w,
                prev_segmap: prev_seg_w.as_ref().map(Arc::clone),
                refpoc,
                refrefpoc,
                tl_4x4_filter: 0,
                svc,
                gmv_warp_allowed,
                gmv_warp,
                jnt_weights,
                tools: ToolCounts::default(),
                sb128_frame,
                multi_tile_frame,
                segmentation_frame,
                qmatrix_frame,
            };
            let res = tdec.decode_tile(w.tile_row, w.tile_col, &w.data, &mut w.cdf);
            w.data = Vec::new();
            let Dec {
                pic,
                blocks,
                lf,
                cdef_state,
                lr,
                lr_counts,
                scratch,
                segmap,
                refmvs,
                tools,
                ..
            } = tdec;
            tile_scratch_pool_w.lock().push((scratch, blocks));
            let [py, pu, pv] = pic.data;
            w.pic_stride = pic.stride;
            w.pic_x_base = pic.x_base;
            w.pic_y_base = pic.y_base;
            w.pic_data[0] = py;
            if has_chroma {
                w.pic_data[1] = pu;
                w.pic_data[2] = pv;
            }
            w.lf = lf;
            w.cdef_state = Some(cdef_state);
            w.lr = lr;
            w.lr_counts = lr_counts;
            w.segmap = segmap;
            if need_refmvs {
                w.refmvs = Some(refmvs);
            }
            w.tools = tools;
            res
        })?;

        for w in &mut works {
            if let (Some(dst_lr), Some(src_lr)) = (dec.lr.as_mut(), w.lr.take()) {
                dst_lr.merge_tile_from(&src_lr);
            }
            dec.lr_counts.units += w.lr_counts.units;
            dec.lr_counts.switchable_units += w.lr_counts.switchable_units;
            dec.lr_counts.none_units += w.lr_counts.none_units;
            dec.lr_counts.wiener_units += w.lr_counts.wiener_units;
            dec.lr_counts.sgr_units += w.lr_counts.sgr_units;
            dec.tools.merge_from(&w.tools);
            tile_cdfs.push(w.cdf.clone());
        }

        // Faz 2 (`5.C.1`): bu dalganın karolarını `tile_row` şeritlerine `split_at_mut`
        // ile bölerek sıfır-kopya paralel birleştir ve hemen serbest bırak.
        #[cfg_attr(
            not(all(feature = "std", not(target_arch = "wasm32"))),
            allow(unused_mut)
        )]
        let mut merged_scoped = false;
        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        if monotonic_rows {
            let guard = pool.acquire_scope_workers(pool.n_threads().saturating_sub(1));
            if guard.total_threads() > 1 {
                merged_scoped = true;
                struct PicTileSrc {
                    data: Vec<u8>,
                    src_stride: usize,
                    x_base: usize,
                    y_base: usize,
                    row_start: usize,
                    row_end_sb4: usize,
                    col_start: usize,
                    col_end_sb4: usize,
                }
                struct RefMvsTileSrc {
                    rmv: RefMvs,
                    row_start: usize,
                    row_end: usize,
                    col_start: usize,
                    col_end: usize,
                }
                struct LfTileSrc {
                    lf: crate::decode::loopfilter::LfFrame,
                    tile_row: usize,
                    tile_col: usize,
                    row_start: usize,
                    row_end: usize,
                    col_start: usize,
                    col_end: usize,
                }
                struct SegCdefTileSrc {
                    segmap: Vec<u8>,
                    tile_segmap_stride: usize,
                    cdef_state: Option<crate::decode::cdef::CdefState>,
                    row_start: usize,
                    row_end: usize,
                    col_start: usize,
                    col_end: usize,
                }

                let mut plane_row_tiles: [Vec<Vec<PicTileSrc>>; 3] = [
                    (0..wave_r_count)
                        .map(|_| Vec::with_capacity(n_cols))
                        .collect(),
                    if dec.pic.has_chroma {
                        (0..wave_r_count)
                            .map(|_| Vec::with_capacity(n_cols))
                            .collect()
                    } else {
                        Vec::new()
                    },
                    if dec.pic.has_chroma {
                        (0..wave_r_count)
                            .map(|_| Vec::with_capacity(n_cols))
                            .collect()
                    } else {
                        Vec::new()
                    },
                ];
                let mut refmvs_row_tiles: Vec<Vec<RefMvsTileSrc>> = if need_refmvs {
                    (0..wave_r_count)
                        .map(|_| Vec::with_capacity(n_cols))
                        .collect()
                } else {
                    Vec::new()
                };
                let mut lf_tiles: Vec<LfTileSrc> = if dec.lf.is_some() {
                    Vec::with_capacity(works.len())
                } else {
                    Vec::new()
                };
                let mut seg_cdef_tiles: Vec<SegCdefTileSrc> = Vec::with_capacity(works.len());

                for w in works.drain(..) {
                    let local_r = w.tile_row - wave_r_start;
                    let [py, pu, pv] = w.pic_data;
                    plane_row_tiles[0][local_r].push(PicTileSrc {
                        data: py,
                        src_stride: w.pic_stride[0],
                        x_base: w.pic_x_base[0],
                        y_base: w.pic_y_base[0],
                        row_start: w.row_start,
                        row_end_sb4: w.row_end_sb4,
                        col_start: w.col_start,
                        col_end_sb4: w.col_end_sb4,
                    });
                    if dec.pic.has_chroma {
                        plane_row_tiles[1][local_r].push(PicTileSrc {
                            data: pu,
                            src_stride: w.pic_stride[1],
                            x_base: w.pic_x_base[1],
                            y_base: w.pic_y_base[1],
                            row_start: w.row_start,
                            row_end_sb4: w.row_end_sb4,
                            col_start: w.col_start,
                            col_end_sb4: w.col_end_sb4,
                        });
                        plane_row_tiles[2][local_r].push(PicTileSrc {
                            data: pv,
                            src_stride: w.pic_stride[2],
                            x_base: w.pic_x_base[2],
                            y_base: w.pic_y_base[2],
                            row_start: w.row_start,
                            row_end_sb4: w.row_end_sb4,
                            col_start: w.col_start,
                            col_end_sb4: w.col_end_sb4,
                        });
                    }
                    if need_refmvs && let Some(rmv) = w.refmvs {
                        refmvs_row_tiles[local_r].push(RefMvsTileSrc {
                            rmv,
                            row_start: w.row_start,
                            row_end: w.row_end,
                            col_start: w.col_start,
                            col_end: w.col_end,
                        });
                    }
                    if let Some(lf) = w.lf {
                        lf_tiles.push(LfTileSrc {
                            lf,
                            tile_row: w.tile_row,
                            tile_col: w.tile_col,
                            row_start: w.row_start,
                            row_end: w.row_end,
                            col_start: w.col_start,
                            col_end: w.col_end,
                        });
                    }
                    seg_cdef_tiles.push(SegCdefTileSrc {
                        segmap: w.segmap,
                        tile_segmap_stride: w.tile_segmap_stride,
                        cdef_state: w.cdef_state,
                        row_start: w.row_start,
                        row_end: w.row_end,
                        col_start: w.col_start,
                        col_end: w.col_end,
                    });
                }

                enum ScopedMergeTask<'a> {
                    PicStrip {
                        dst: &'a mut [u8],
                        dst_y0: usize,
                        dst_stride: usize,
                        bpp: usize,
                        sh: usize,
                        sv: usize,
                        tiles: Vec<PicTileSrc>,
                    },
                    RefMvsStrip {
                        dst: &'a mut [crate::decode::refmvs::RefBlock],
                        dst_y0: usize,
                        iw4: usize,
                        ih4: usize,
                        tiles: Vec<RefMvsTileSrc>,
                    },
                    Lf {
                        lf: &'a mut crate::decode::loopfilter::LfFrame,
                        w4: usize,
                        h4: usize,
                        tiles: Vec<LfTileSrc>,
                    },
                    SegmapAndCdef {
                        segmap: &'a mut [u8],
                        segmap_stride: usize,
                        bh: usize,
                        cdef_state: &'a mut crate::decode::cdef::CdefState,
                        tiles: Vec<SegCdefTileSrc>,
                    },
                }

                let mut tasks: Vec<ScopedMergeTask<'_>> = Vec::with_capacity(wave_r_count * 4 + 2);
                let total_hits = core::sync::atomic::AtomicU64::new(0);

                tasks.push(ScopedMergeTask::SegmapAndCdef {
                    segmap: dec.segmap.as_mut_slice(),
                    segmap_stride: dec.segmap_stride,
                    bh: dec.bh,
                    cdef_state: &mut dec.cdef_state,
                    tiles: seg_cdef_tiles,
                });

                if need_refmvs {
                    let iw4 = dec.refmvs.iw4;
                    let ih4 = dec.refmvs.ih4;
                    let wave_y0 = row_bounds_sb4[wave_r_start].0.min(ih4);
                    let start_elem = wave_y0.saturating_mul(iw4).min(dec.refmvs.grid.len());
                    let mut rem = &mut dec.refmvs.grid[start_elem..];
                    let mut cur_y = wave_y0;
                    for (local_r, row_tiles) in refmvs_row_tiles.into_iter().enumerate() {
                        let r = wave_r_start + local_r;
                        let (_, _, r_end) = row_bounds_sb4[r];
                        let y1 = r_end.min(ih4);
                        let take_elems = if r + 1 == n_rows {
                            rem.len()
                        } else {
                            y1.saturating_sub(cur_y).saturating_mul(iw4).min(rem.len())
                        };
                        let (band, rest) = rem.split_at_mut(take_elems);
                        rem = rest;
                        tasks.push(ScopedMergeTask::RefMvsStrip {
                            dst: band,
                            dst_y0: cur_y,
                            iw4,
                            ih4,
                            tiles: row_tiles,
                        });
                        cur_y = y1;
                    }
                }

                let [y_plane, u_plane, v_plane] = &mut dec.pic.data;
                let [y_tiles_by_row, u_tiles_by_row, v_tiles_by_row] = plane_row_tiles;
                let bpp = dec.pic.bpp;

                if dec.pic.has_chroma {
                    let sh = dec.pic.ss_hor;
                    let sv = dec.pic.ss_ver;
                    for (plane_slice, row_buckets, dst_stride) in [
                        (u_plane.as_mut_slice(), u_tiles_by_row, dec.pic.stride[1]),
                        (v_plane.as_mut_slice(), v_tiles_by_row, dec.pic.stride[2]),
                    ] {
                        let max_rows = plane_slice.len().checked_div(dst_stride).unwrap_or(0);
                        let wave_y0 = ((row_bounds_sb4[wave_r_start].0 * 4) >> sv).min(max_rows);
                        let start_byte = wave_y0.saturating_mul(dst_stride).min(plane_slice.len());
                        let mut rem = &mut plane_slice[start_byte..];
                        let mut cur_y = wave_y0;
                        for (local_r, row_tiles) in row_buckets.into_iter().enumerate() {
                            let r = wave_r_start + local_r;
                            let (_, row_end_sb4, _) = row_bounds_sb4[r];
                            let y1 = (((row_end_sb4 * 4) + sv) >> sv).min(max_rows);
                            let take_bytes = if r + 1 == n_rows {
                                rem.len()
                            } else {
                                y1.saturating_sub(cur_y)
                                    .saturating_mul(dst_stride)
                                    .min(rem.len())
                            };
                            let (band, rest) = rem.split_at_mut(take_bytes);
                            rem = rest;
                            tasks.push(ScopedMergeTask::PicStrip {
                                dst: band,
                                dst_y0: cur_y,
                                dst_stride,
                                bpp,
                                sh,
                                sv,
                                tiles: row_tiles,
                            });
                            cur_y = y1;
                        }
                    }
                }

                {
                    let dst_stride = dec.pic.stride[0];
                    let max_rows = y_plane.len().checked_div(dst_stride).unwrap_or(0);
                    let wave_y0 = (row_bounds_sb4[wave_r_start].0 * 4).min(max_rows);
                    let start_byte = wave_y0.saturating_mul(dst_stride).min(y_plane.len());
                    let mut rem = &mut y_plane[start_byte..];
                    let mut cur_y = wave_y0;
                    for (local_r, row_tiles) in y_tiles_by_row.into_iter().enumerate() {
                        let r = wave_r_start + local_r;
                        let (_, row_end_sb4, _) = row_bounds_sb4[r];
                        let y1 = (row_end_sb4 * 4).min(max_rows);
                        let take_bytes = if r + 1 == n_rows {
                            rem.len()
                        } else {
                            y1.saturating_sub(cur_y)
                                .saturating_mul(dst_stride)
                                .min(rem.len())
                        };
                        let (band, rest) = rem.split_at_mut(take_bytes);
                        rem = rest;
                        tasks.push(ScopedMergeTask::PicStrip {
                            dst: band,
                            dst_y0: cur_y,
                            dst_stride,
                            bpp,
                            sh: 0,
                            sv: 0,
                            tiles: row_tiles,
                        });
                        cur_y = y1;
                    }
                }

                if let Some(lf) = dec.lf.as_mut() {
                    tasks.push(ScopedMergeTask::Lf {
                        lf,
                        w4: dec.w4,
                        h4: dec.h4,
                        tiles: lf_tiles,
                    });
                }

                let n_merge_tasks = tasks.len();
                let queue = Mutex::new(tasks);
                let run_merge = || loop {
                    let Some(task) = queue.lock().pop() else {
                        break;
                    };
                    match task {
                        ScopedMergeTask::PicStrip {
                            dst,
                            dst_y0,
                            dst_stride,
                            bpp,
                            sh,
                            sv,
                            tiles,
                        } => {
                            let strip_rows = dst.len().checked_div(dst_stride).unwrap_or(0);
                            let strip_y1 = dst_y0 + strip_rows;
                            for t in tiles {
                                if t.src_stride == 0 {
                                    continue;
                                }
                                let y0 = ((t.row_start * 4) >> sv).max(dst_y0).min(strip_y1);
                                let y1 = (((t.row_end_sb4 * 4) + sv) >> sv).min(strip_y1);
                                let x0_px = (t.col_start * 4) >> sh;
                                let x1_px = ((t.col_end_sb4 * 4) + sh) >> sh;
                                let dst_x0_bytes = (x0_px * bpp).min(dst_stride);
                                let dst_x1_bytes = (x1_px * bpp).min(dst_stride);
                                if dst_x0_bytes < dst_x1_bytes {
                                    let copy_bytes = dst_x1_bytes - dst_x0_bytes;
                                    let src_x0_bytes = (x0_px - t.x_base) * bpp;
                                    for y in y0..y1 {
                                        let dst_off = (y - dst_y0) * dst_stride + dst_x0_bytes;
                                        let src_off = (y - t.y_base) * t.src_stride + src_x0_bytes;
                                        dst[dst_off..dst_off + copy_bytes].copy_from_slice(
                                            &t.data[src_off..src_off + copy_bytes],
                                        );
                                    }
                                }
                            }
                        }
                        ScopedMergeTask::RefMvsStrip {
                            dst,
                            dst_y0,
                            iw4,
                            ih4,
                            tiles,
                        } => {
                            let strip_rows = dst.len().checked_div(iw4).unwrap_or(0);
                            let strip_y1 = (dst_y0 + strip_rows).min(ih4);
                            let mut hits = 0u64;
                            for t in tiles {
                                let other_rmv = t.rmv;
                                if !other_rmv.grid.is_empty() && other_rmv.stride > 0 {
                                    let y0 = t.row_start.max(dst_y0).min(strip_y1);
                                    let y1 = t.row_end.min(strip_y1);
                                    let x0 = t.col_start.min(iw4);
                                    let x1 = t.col_end.min(iw4);
                                    if x0 < x1 {
                                        let copy_w = x1 - x0;
                                        let src_x0 = x0 - other_rmv.x_base;
                                        for y in y0..y1 {
                                            let dst_off = (y - dst_y0) * iw4 + x0;
                                            let src_off =
                                                (y - other_rmv.y_base) * other_rmv.stride + src_x0;
                                            dst[dst_off..dst_off + copy_w].copy_from_slice(
                                                &other_rmv.grid[src_off..src_off + copy_w],
                                            );
                                        }
                                    }
                                }
                                hits += other_rmv
                                    .temporal_hits
                                    .load(core::sync::atomic::Ordering::Relaxed);
                            }
                            if hits > 0 {
                                total_hits.fetch_add(hits, core::sync::atomic::Ordering::Relaxed);
                            }
                        }
                        ScopedMergeTask::Lf { lf, w4, h4, tiles } => {
                            for t in tiles {
                                lf.merge_tile_from(
                                    &t.lf,
                                    t.tile_row,
                                    t.tile_col,
                                    t.row_start.min(h4),
                                    t.row_end.min(h4),
                                    t.col_start.min(w4),
                                    t.col_end.min(w4),
                                );
                            }
                        }
                        ScopedMergeTask::SegmapAndCdef {
                            segmap,
                            segmap_stride,
                            bh,
                            cdef_state,
                            tiles,
                        } => {
                            for t in tiles {
                                if !t.segmap.is_empty() {
                                    let sy0 = t.row_start.min(bh);
                                    let sy1 = t.row_end.min(bh);
                                    let sx0 = t.col_start.min(segmap_stride);
                                    let sx1 = t.col_end.min(segmap_stride);
                                    if sx0 < sx1 {
                                        let copy_w = sx1 - sx0;
                                        let src_stride = t.tile_segmap_stride;
                                        for y in sy0..sy1 {
                                            let dst_off = y * segmap_stride + sx0;
                                            let src_off = (y - t.row_start) * src_stride;
                                            if dst_off + copy_w <= segmap.len()
                                                && src_off + copy_w <= t.segmap.len()
                                            {
                                                segmap[dst_off..dst_off + copy_w].copy_from_slice(
                                                    &t.segmap[src_off..src_off + copy_w],
                                                );
                                            }
                                        }
                                    }
                                }
                                if let Some(other_cdef) = t.cdef_state {
                                    cdef_state.merge_tile_from(
                                        &other_cdef,
                                        t.row_start,
                                        t.row_end,
                                        t.col_start,
                                        t.col_end,
                                    );
                                }
                            }
                        }
                    }
                };

                let n_spawn = guard.total_threads().min(n_merge_tasks).saturating_sub(1);
                std::thread::scope(|s| {
                    for _ in 0..n_spawn {
                        s.spawn(run_merge);
                    }
                    run_merge();
                });
                drop(queue);
                drop(guard);

                let hits = total_hits.into_inner();
                if hits > 0 {
                    dec.refmvs
                        .temporal_hits
                        .fetch_add(hits, core::sync::atomic::Ordering::Relaxed);
                }
            }
        }

        if !merged_scoped {
            // Geri-düşüş yolu (küçük kareler, tek satırlı tile ızgaraları veya `no_std`/`wasm32`):
            // 6 bağımsız hedef bileşeni (`Pic Y`, `Pic U`, `Pic V`, `RefMvs::grid`, `LfFrame`,
            // `segmap + CdefState`) `WorkerPool` üzerinde paralel birleştir.
            #[allow(clippy::large_enum_variant)]
            enum MergeComponent {
                PicPlane {
                    plane: usize,
                    data: Vec<u8>,
                    dst_stride: usize,
                    bpp: usize,
                    sh: usize,
                    sv: usize,
                },
                RefMvsGrid {
                    grid: Vec<crate::decode::refmvs::RefBlock>,
                    iw4: usize,
                    ih4: usize,
                    hits: u64,
                },
                Lf {
                    lf: crate::decode::loopfilter::LfFrame,
                    w4: usize,
                    h4: usize,
                },
                SegmapAndCdef {
                    segmap: Vec<u8>,
                    segmap_stride: usize,
                    bh: usize,
                    cdef_state: crate::decode::cdef::CdefState,
                },
            }

            let works_arc: Arc<Vec<Mutex<TileWork>>> =
                Arc::new(works.into_iter().map(Mutex::new).collect());
            let mut jobs: Vec<MergeComponent> = Vec::with_capacity(6);
            jobs.push(MergeComponent::PicPlane {
                plane: 0,
                data: core::mem::take(&mut dec.pic.data[0]),
                dst_stride: dec.pic.stride[0],
                bpp: dec.pic.bpp,
                sh: 0,
                sv: 0,
            });
            if dec.pic.has_chroma {
                jobs.push(MergeComponent::PicPlane {
                    plane: 1,
                    data: core::mem::take(&mut dec.pic.data[1]),
                    dst_stride: dec.pic.stride[1],
                    bpp: dec.pic.bpp,
                    sh: dec.pic.ss_hor,
                    sv: dec.pic.ss_ver,
                });
                jobs.push(MergeComponent::PicPlane {
                    plane: 2,
                    data: core::mem::take(&mut dec.pic.data[2]),
                    dst_stride: dec.pic.stride[2],
                    bpp: dec.pic.bpp,
                    sh: dec.pic.ss_hor,
                    sv: dec.pic.ss_ver,
                });
            }
            if need_refmvs {
                jobs.push(MergeComponent::RefMvsGrid {
                    grid: core::mem::take(&mut dec.refmvs.grid),
                    iw4: dec.refmvs.iw4,
                    ih4: dec.refmvs.ih4,
                    hits: 0,
                });
            }
            if let Some(lf) = dec.lf.take() {
                jobs.push(MergeComponent::Lf {
                    lf,
                    w4: dec.w4,
                    h4: dec.h4,
                });
            }
            jobs.push(MergeComponent::SegmapAndCdef {
                segmap: core::mem::take(&mut dec.segmap),
                segmap_stride: dec.segmap_stride,
                bh: dec.bh,
                cdef_state: core::mem::replace(
                    &mut dec.cdef_state,
                    crate::decode::cdef::CdefState {
                        noskip: Vec::new(),
                        noskip_stride: 1,
                        noskip_x_base: 0,
                        noskip_y_base: 0,
                        idx64: Vec::new(),
                        sb64w: 1,
                        sb64_x_base: 0,
                        sb64_y_base: 0,
                    },
                ),
            });

            let works_for_workers = Arc::clone(&works_arc);
            pool.execute_tasks(&mut jobs, move |_wid, _jidx, job: &mut MergeComponent| {
                match job {
                    MergeComponent::PicPlane {
                        plane,
                        data,
                        dst_stride,
                        bpp,
                        sh,
                        sv,
                    } => {
                        let p = *plane;
                        let dst_stride = *dst_stride;
                        let bpp = *bpp;
                        let sh = *sh;
                        let sv = *sv;
                        if let Some(max_rows) = data.len().checked_div(dst_stride) {
                            for w_lock in works_for_workers.iter() {
                                let (
                                    src_data,
                                    src_stride,
                                    x_base,
                                    y_base,
                                    row_start,
                                    row_end_sb4,
                                    col_start,
                                    col_end_sb4,
                                ) = {
                                    let mut w = w_lock.lock();
                                    (
                                        core::mem::take(&mut w.pic_data[p]),
                                        w.pic_stride[p],
                                        w.pic_x_base[p],
                                        w.pic_y_base[p],
                                        w.row_start,
                                        w.row_end_sb4,
                                        w.col_start,
                                        w.col_end_sb4,
                                    )
                                };
                                if src_data.is_empty() || src_stride == 0 {
                                    continue;
                                }
                                let y0 = ((row_start * 4) >> sv).min(max_rows);
                                let y1 = (((row_end_sb4 * 4) + sv) >> sv).min(max_rows);
                                let x0_px = (col_start * 4) >> sh;
                                let x1_px = ((col_end_sb4 * 4) + sh) >> sh;
                                let dst_x0_bytes = (x0_px * bpp).min(dst_stride);
                                let dst_x1_bytes = (x1_px * bpp).min(dst_stride);
                                if dst_x0_bytes < dst_x1_bytes {
                                    let copy_bytes = dst_x1_bytes - dst_x0_bytes;
                                    let src_x0_bytes = (x0_px - x_base) * bpp;
                                    for y in y0..y1 {
                                        let dst_off = y * dst_stride + dst_x0_bytes;
                                        let src_off = (y - y_base) * src_stride + src_x0_bytes;
                                        data[dst_off..dst_off + copy_bytes].copy_from_slice(
                                            &src_data[src_off..src_off + copy_bytes],
                                        );
                                    }
                                }
                            }
                        }
                    }
                    MergeComponent::RefMvsGrid {
                        grid,
                        iw4,
                        ih4,
                        hits,
                    } => {
                        let iw4 = *iw4;
                        let ih4 = *ih4;
                        for w_lock in works_for_workers.iter() {
                            let (other_rmv, row_start, row_end, col_start, col_end) = {
                                let mut w = w_lock.lock();
                                let Some(r) = w.refmvs.take() else {
                                    continue;
                                };
                                (r, w.row_start, w.row_end, w.col_start, w.col_end)
                            };
                            if !other_rmv.grid.is_empty() && other_rmv.stride > 0 {
                                let y0 = row_start.min(ih4);
                                let y1 = row_end.min(ih4);
                                let x0 = col_start.min(iw4);
                                let x1 = col_end.min(iw4);
                                if x0 < x1 {
                                    let copy_w = x1 - x0;
                                    let src_x0 = x0 - other_rmv.x_base;
                                    for y in y0..y1 {
                                        let dst_off = y * iw4 + x0;
                                        let src_off =
                                            (y - other_rmv.y_base) * other_rmv.stride + src_x0;
                                        grid[dst_off..dst_off + copy_w].copy_from_slice(
                                            &other_rmv.grid[src_off..src_off + copy_w],
                                        );
                                    }
                                }
                            }
                            *hits += other_rmv
                                .temporal_hits
                                .load(core::sync::atomic::Ordering::Relaxed);
                        }
                    }
                    MergeComponent::Lf { lf, w4, h4 } => {
                        let w4 = *w4;
                        let h4 = *h4;
                        for w_lock in works_for_workers.iter() {
                            let (
                                src_lf,
                                tile_row,
                                tile_col,
                                row_start,
                                row_end,
                                col_start,
                                col_end,
                            ) = {
                                let mut w = w_lock.lock();
                                let Some(l) = w.lf.take() else {
                                    continue;
                                };
                                (
                                    l,
                                    w.tile_row,
                                    w.tile_col,
                                    w.row_start,
                                    w.row_end,
                                    w.col_start,
                                    w.col_end,
                                )
                            };
                            lf.merge_tile_from(
                                &src_lf,
                                tile_row,
                                tile_col,
                                row_start.min(h4),
                                row_end.min(h4),
                                col_start.min(w4),
                                col_end.min(w4),
                            );
                        }
                    }
                    MergeComponent::SegmapAndCdef {
                        segmap,
                        segmap_stride,
                        bh,
                        cdef_state,
                    } => {
                        let segmap_stride = *segmap_stride;
                        let bh = *bh;
                        for w_lock in works_for_workers.iter() {
                            let (
                                other_seg,
                                other_cdef,
                                tile_segmap_stride,
                                row_start,
                                row_end,
                                col_start,
                                col_end,
                            ) = {
                                let mut w = w_lock.lock();
                                (
                                    core::mem::take(&mut w.segmap),
                                    w.cdef_state.take(),
                                    w.tile_segmap_stride,
                                    w.row_start,
                                    w.row_end,
                                    w.col_start,
                                    w.col_end,
                                )
                            };
                            if !other_seg.is_empty() {
                                let sy0 = row_start.min(bh);
                                let sy1 = row_end.min(bh);
                                let sx0 = col_start.min(segmap_stride);
                                let sx1 = col_end.min(segmap_stride);
                                if sx0 < sx1 {
                                    let copy_w = sx1 - sx0;
                                    let src_stride = tile_segmap_stride;
                                    for y in sy0..sy1 {
                                        let dst_off = y * segmap_stride + sx0;
                                        let src_off = (y - row_start) * src_stride;
                                        if dst_off + copy_w <= segmap.len()
                                            && src_off + copy_w <= other_seg.len()
                                        {
                                            segmap[dst_off..dst_off + copy_w].copy_from_slice(
                                                &other_seg[src_off..src_off + copy_w],
                                            );
                                        }
                                    }
                                }
                            }
                            if let Some(other_cdef) = other_cdef {
                                cdef_state.merge_tile_from(
                                    &other_cdef,
                                    row_start,
                                    row_end,
                                    col_start,
                                    col_end,
                                );
                            }
                        }
                    }
                }
                Ok(())
            })?;

            for job in jobs {
                match job {
                    MergeComponent::PicPlane { plane, data, .. } => {
                        dec.pic.data[plane] = data;
                    }
                    MergeComponent::RefMvsGrid { grid, hits, .. } => {
                        dec.refmvs.grid = grid;
                        dec.refmvs
                            .temporal_hits
                            .fetch_add(hits, core::sync::atomic::Ordering::Relaxed);
                    }
                    MergeComponent::Lf { lf, .. } => {
                        dec.lf = Some(lf);
                    }
                    MergeComponent::SegmapAndCdef {
                        segmap, cdef_state, ..
                    } => {
                        dec.segmap = segmap;
                        dec.cdef_state = cdef_state;
                    }
                }
            }
        }

        wave_r_start = wave_r_end;
    }
    drop(tile_scratch_pool);
    Ok(())
}

impl Dec<'_> {
    fn frame_plane_count(&self) -> usize {
        if self.has_chroma { 3 } else { 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// dav1d `recon_tmpl.c:831`: kroma bağlamı kroma 4x4 birimindedir.
    #[test]
    fn kroma_baglam_indeksi_kroma_biriminde() {
        assert_eq!(chroma_ctx4(5, 0), 5, "4:4:4'te kaydırma yok");
        assert_eq!(chroma_ctx4(5, 1), 2, "4:2:0/4:2:2 yatay yarı");
        assert_eq!(chroma_ctx4(0, 1), 0);
    }

    /// dav1d `recon_tmpl.c:1197-1200,1440`: 64x64 px parça (chunk) bölümlemesi
    /// bloğu **tam** örter ve parçalar çakışmaz; kroma parçaları aynı düzende
    /// `ss_hor`/`ss_ver` ile ölçeklenir (`sub_ch4`, `sub_cw4`).
    #[test]
    fn parca_bolumlemesi_luma_ve_kromayi_tam_orer() {
        let boyutlar = [
            (1usize, 1usize),
            (2, 2),
            (4, 4),
            (8, 8),
            (16, 16),
            (32, 32),
            (32, 16),
            (16, 32),
            (4, 16),
            (20, 7),
        ];
        let kromalar = [(0usize, 0usize), (1, 0), (1, 1)];
        for &(w4, h4) in &boyutlar {
            for &(ss_hor, ss_ver) in &kromalar {
                let cw4 = (w4 + ss_hor) >> ss_hor;
                let ch4 = (h4 + ss_ver) >> ss_ver;
                let mut luma = 0usize;
                let mut chroma = 0usize;
                let mut init_y = 0usize;
                while init_y < h4 {
                    let sub_h4 = h4.min(init_y + 16);
                    let sub_ch4 = ch4.min((init_y + 16) >> ss_ver);
                    let mut init_x = 0usize;
                    while init_x < w4 {
                        let sub_w4 = w4.min(init_x + 16);
                        let sub_cw4 = cw4.min((init_x + 16) >> ss_hor);
                        luma += (sub_h4 - init_y) * (sub_w4 - init_x);
                        let cy0 = init_y >> ss_ver;
                        let cx0 = init_x >> ss_hor;
                        if sub_ch4 > cy0 && sub_cw4 > cx0 {
                            chroma += (sub_ch4 - cy0) * (sub_cw4 - cx0);
                        }
                        init_x += 16;
                    }
                    init_y += 16;
                }
                assert_eq!(luma, w4 * h4, "luma kapsamı w4={w4} h4={h4}");
                assert_eq!(
                    chroma,
                    cw4 * ch4,
                    "kroma kapsamı w4={w4} h4={h4} ss=({ss_hor},{ss_ver})"
                );
            }
        }
    }

    /// dav1d `decode.c:2794,2651`: above bağlamı `sb128w * tiling.rows`
    /// kadardır; her tile satırı ayrı bölge kullanır.
    #[test]
    fn above_baglami_tile_satiri_basina_ayrilir() {
        let blocks = Blocks::new(3, 2, &Butce::sinirsiz()).expect("bağlam");
        assert_eq!(blocks.above.len(), 6);
        assert_eq!(blocks.above_pal.len(), 6);
        assert_eq!(blocks.above_pal_uv_sz.len(), 6);
    }

    /// P38 taşma sınıfı: AV1 sınırında (65 536², 16 bit, 4:4:4, 64 × 64 tile)
    /// kare düzeyi ayırma boyutları taşmadan hesaplanır. Dev kare ayrılmaz:
    /// 1 baytlık bütçe ilk düzlemde `BudgetExceeded` verir ve istenen bayt
    /// envanterdeki değere (`docs/baseline/kaynak-siniri.md`) eşittir.
    #[test]
    fn tasma_kare_ayirmalari_av1_sinirinda_dogru_boyutlu() {
        let b = Butce::yeni(Some(1), 0);
        let butce_hatasi = |requested: usize| Error::BudgetExceeded {
            requested,
            limit: 1,
        };
        let pic = PicBuf::new(65536, 65536, ChromaFormat::C444, BitDepth::Twelve, &b).err();
        #[cfg(target_pointer_width = "64")]
        assert_eq!(pic, Some(butce_hatasi(131_072 * 65_537 + 16)));
        #[cfg(not(target_pointer_width = "64"))]
        assert!(matches!(pic, Some(Error::SizeOverflow { .. })));
        let mut h = FrameHeader::default();
        h.tiling.cols = 64;
        h.tiling.rows = 64;
        let lf = crate::decode::loopfilter::LfFrame::new(&h, 16_384, 16_384, 0, 0, true, &b).err();
        assert_eq!(lf, Some(butce_hatasi(16_384 * 16_384 * 4)));
        let mv = RefMvs::new(65536, 65536, crate::dsp::refmvs::RefmvsDsp::SCALAR, &b).err();
        assert_eq!(mv, Some(butce_hatasi(16_384 * 16_384 * 12)));
        let bloklar = Blocks::new(512, 64, &b).err();
        assert_eq!(bloklar, Some(butce_hatasi(512 * 64 * 624)));
        // `usize` sınırında: taşma hatası, panik yok.
        for (w, h) in [
            (usize::MAX - 64, 64),
            (64, usize::MAX - 64),
            (usize::MAX / 2, 3),
        ] {
            assert!(
                matches!(
                    PicBuf::new(w, h, ChromaFormat::C420, BitDepth::Ten, &b),
                    Err(Error::SizeOverflow { .. })
                ),
                "{w}×{h}"
            );
        }
    }

    /// dav1d `decode.c:281`: tek üst örnek özel yolu `(unsigned) masks[0] == 1`
    /// (yalnız alt 32 bit) ve `!(masks[1] >> 32)`. Sağ-üst örnek (bit 32)
    /// özel yolu **kapatmaz**; sol-üst örnek kapatır.
    #[test]
    fn warp_tek_ust_ornek_kontrolu_alt_32_bit() {
        // Sağ-üst bit set: dav1d özel yola girer.
        assert!(warp_uses_single_top_sample(&[0x1_0000_0001, 1]));
        // Yalnız alt örnek: özel yol.
        assert!(warp_uses_single_top_sample(&[1, 0]));
        // Sol-üst örnek var: özel yol değil.
        assert!(!warp_uses_single_top_sample(&[1, 1 << 32]));
        // Alt 32 bit 1 değil: genel yol.
        assert!(!warp_uses_single_top_sample(&[2, 0]));
        assert!(!warp_uses_single_top_sample(&[0, 0]));
        // Yalnız bit 32: alt 32 bit 0 → genel yol.
        assert!(!warp_uses_single_top_sample(&[1 << 32, 0]));
    }

    /// dav1d `decode.c:315-329` örnek seçim döngüsü. Uygulama notu:
    /// `np = 4`, `mvd = [-1, -1, geçerli, geçerli]` → dav1d sonucu
    /// `[p3, p2, …]` (her yer değiştirmeden sonra `i++`, `j--`).
    #[test]
    fn warp_ornek_secimi_dav1d_dongusu() {
        let mut pts = noktalar();
        let mut mvd = [-1, -1, 7, 9, 0, 0, 0, 0];
        select_warp_samples(&mut pts, 4, 2, &mut mvd);
        assert_eq!(pts[0], nokta(3), "ilk eşik dışı örnek p3 ile değişir");
        assert_eq!(pts[1], nokta(2), "ikinci eşik dışı örnek p2 ile değişir");
        assert!(mvd[0] != -1 && mvd[1] != -1);
    }

    /// Bütün `mvd` desenleri: `i`/`j` çakışmaz, `usize` taşmaz, seçilen
    /// `ret` örnek tam olarak geçerli örneklerin kümesidir.
    #[test]
    fn warp_ornek_secimi_butun_desenler() {
        for np in 1..=8usize {
            for bits in 0u32..(1u32 << np) {
                let mut pts = noktalar();
                let mut mvd = [0i32; 8];
                let mut gecerli = Vec::new();
                for (i, m) in mvd.iter_mut().enumerate().take(np) {
                    if bits & (1 << i) != 0 {
                        *m = -1;
                    } else {
                        *m = i as i32;
                        gecerli.push(nokta(i));
                    }
                }
                if gecerli.is_empty() {
                    continue;
                }
                select_warp_samples(&mut pts, np, gecerli.len(), &mut mvd);
                for (i, p) in pts.iter().take(gecerli.len()).enumerate() {
                    assert!(mvd[i] != -1, "np={np} bits={bits} i={i} geçersiz");
                    assert!(
                        gecerli.contains(p),
                        "np={np} bits={bits} i={i} seçilen örnek kaynak kümede değil"
                    );
                }
                let mut secilen: Vec<_> = pts.iter().take(gecerli.len()).copied().collect();
                secilen.sort_unstable();
                let mut beklenen = gecerli.clone();
                beklenen.sort_unstable();
                assert_eq!(secilen, beklenen, "np={np} bits={bits}");
            }
        }
    }

    fn nokta(i: usize) -> [[i32; 2]; 2] {
        [[i as i32, 10 + i as i32], [100 + i as i32, 200 + i as i32]]
    }

    /// dav1d `decode.c:3618-3672`: kare segment haritası başlangıcının üç
    /// dalı — güncelleme (sıfırdan yeni harita), **devir** (önceki harita
    /// paylaşılır) ve önceki harita yoksa sıfırlanmış yeni harita.
    ///
    /// Devir dalı aynı zamanda referans yuvasına yazılan haritayı belirler
    /// (`:3690-3693`): `Dec::new` bu baytları `DecodedFrame.segmap` ve
    /// `RefFrame.segmap` üzerinden sonraki kareye taşır.
    #[test]
    fn segment_haritasi_devri_dav1d_dallari() {
        let prev = [0u8, 1, 2, 3, 4, 5, 6, 0];

        // Dal 1: update_map=1 → önceki harita kullanılmaz; yeni harita sıfır
        // başlar ve `decode_b` doldurur.
        let mut guncelle = FrameHeader::default();
        guncelle.segmentation.enabled = true;
        guncelle.segmentation.update_map = true;
        let m = init_segmap(&guncelle, 4, 2, Some(&prev), &Butce::sinirsiz()).expect("harita");
        assert_eq!(m, [0u8; 8], "update_map=1: önceki harita devralınmaz");

        // Dal 2: update_map=0 ve önceki harita var → harita **paylaşılır**.
        let mut devir = FrameHeader::default();
        devir.segmentation.enabled = true;
        devir.segmentation.update_map = false;
        let m = init_segmap(&devir, 4, 2, Some(&prev), &Butce::sinirsiz()).expect("harita");
        assert_eq!(m, prev, "update_map=0: önceki harita birebir devralınır");

        // Dal 3: update_map=0 ve önceki harita yok → sıfırlanmış yeni harita.
        let m = init_segmap(&devir, 4, 2, None, &Butce::sinirsiz()).expect("harita");
        assert_eq!(m, [0u8; 8], "önceki harita yok: sıfır harita");

        // Segmentasyon kapalı: dav1d `cur_segmap = NULL`; Rust sıfır vektör
        // saklar (yuva boyut denetimi devri engeller).
        let mut kapali = FrameHeader::default();
        kapali.segmentation.enabled = false;
        let m = init_segmap(&kapali, 4, 2, Some(&prev), &Butce::sinirsiz()).expect("harita");
        assert_eq!(m, [0u8; 8], "segmentasyon kapalı: harita yok");

        // Kare yüksekliği dolgusu: devralınan bölgenin dışı sıfır kalır.
        let m = init_segmap(&devir, 4, 3, Some(&prev), &Butce::sinirsiz()).expect("harita");
        assert_eq!(&m[..8], &prev[..], "devralınan bölge");
        assert_eq!(&m[8..], &[0u8; 4], "dolgu sıfır");
    }

    /// `get_prev_frame_segid` (`decode.c:489-505`): blok üzerindeki en küçük
    /// segment kimliği; sıfır görüldüğünde erken çıkış, taşma yok.
    #[test]
    fn onceki_kare_segment_kimligi_blok_minimumu() {
        // Satır adımı 4; blok (0,0) 2×2 4x4 birim.
        let map: Vec<u8> = vec![
            7, 7, 7, 7, //
            7, 3, 7, 7, //
            7, 7, 7, 7, //
            7, 7, 7, 7,
        ];
        assert_eq!(
            prev_segid_in_map(&map, 4, 0, 0, 2, 2),
            3,
            "blok içi minimum"
        );
        assert_eq!(
            prev_segid_in_map(&map, 4, 0, 2, 2, 2),
            7,
            "blok sınırı dışına taşmaz"
        );
        assert_eq!(
            prev_segid_in_map(&[0; 16], 4, 0, 0, 4, 4),
            0,
            "sıfır kimlik"
        );
    }

    fn noktalar() -> [[[i32; 2]; 2]; 8] {
        let mut pts = [[[0i32; 2]; 2]; 8];
        for (i, p) in pts.iter_mut().enumerate() {
            *p = nokta(i);
        }
        pts
    }
}
