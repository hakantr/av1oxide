//! CDEF uygulaması (kare üzerine) — S2c.2a / P19.
//!
//! dav1d `src/cdef_apply_tmpl.c` `dav1d_cdef_brow` (97-) C yolunun birebir
//! karşılığı. Çekirdek ([`crate::dsp::scalar::cdef`]) **değiştirilmez**;
//! bu modül yalnız çağrı kurallarını (atlanan 64×64/8×8, güçler, `damping`,
//! `adjust_strength`, 4:2:2 yön tablosu, kenar bayrakları, filtre öncesi
//! komşu pikselleri) uygular.
//!
//! ## dav1d eşlemesi (dav1d @ `99f3354`)
//!
//! | Rust | dav1d |
//! |---|---|
//! | [`CdefState::mark_noskip`] | `decode.c:1952-1961` |
//! | [`CdefState`] `idx64` | `decode.c:948-963` (`lflvl->cdef_idx[4]`) |
//! | [`adjust_strength`] | `cdef_apply_tmpl.c:91-95` |
//! | [`apply`] | `cdef_apply_tmpl.c:97-` (`dav1d_cdef_brow`) |
//! | [`UV_DIRS`] | `cdef_apply_tmpl.c:120-121` |
//! | güçler / `damping` | `cdef_apply_tmpl.c:167-181` |
//! | `dir` + varyans | `cdef_apply_tmpl.c:210-212` |
//!
//! ## Neden kare sonunda?
//!
//! dav1d CDEF'i SB satırı başına uygular; filtre girdileri **deblocking
//! sonrası, CDEF öncesi** piksellerdir. Bu port deblocking'i kare sonunda
//! bir kez uyguladığı için CDEF de kare sonunda **8 satırlık bantlar**
//! hâlinde raster sırayla uygulanır. Üst iki satır ve sol iki sütun
//! (filtrelenmiş bölge) yedekten, alt iki satır ve sağ iki sütun (henüz
//! filtrelenmemiş bölge) canlı kareden okunur; dav1d'nin `cdef_line` ve
//! `lr_bak` yedekleriyle birebir aynı değerlerdir. Tile sınırı CDEF'i
//! **etkilemez** (`cdef_apply_tmpl.c:97-`).
#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use alloc::vec::Vec;

use crate::bitstream::frame_header::FrameHeader;
use crate::decode::intra::{PicBuf, ToolCounts};
use crate::dsp::cdef::{
    CdefBitDepth, CdefBlockSize, CdefDsp, CdefEdgeFlags, CdefFilterFn, CdefFilterParams,
};
use crate::frame::{BitDepth, ChromaFormat};

/// `uv_dirs[2][8]` (`cdef_apply_tmpl.c:120-121`): satır 1 yalnız 4:2:2.
const UV_DIRS: [[u8; 8]; 2] = [[0, 1, 2, 3, 4, 5, 6, 7], [7, 0, 2, 4, 5, 6, 6, 6]];

/// CDEF çağrı izi (tanılama; varsayılan kapalı).
///
/// `AV1OXIDE_CDEF_TRACE` varsa `cdef_apply_tmpl.c` çağrıları satır satır
/// yazılır; `AV1OXIDE_CDEF_TRACE_FRAME` kare, `AV1OXIDE_CDEF_TRACE_PLANE`
/// düzlem süzgecidir. Biçim `scripts/oracle/cdef_trace.c` ile **aynıdır**.
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
            std::env::var_os("AV1OXIDE_CDEF_TRACE")?;
            let frame = std::env::var("AV1OXIDE_CDEF_TRACE_FRAME")
                .ok()
                .and_then(|v| v.parse().ok());
            let plane = std::env::var("AV1OXIDE_CDEF_TRACE_PLANE")
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

/// FNV-1a 64 — `cdef_trace.c` ile aynı (filtre sonrası blok özeti).
fn fnv_block(data: &[u8], stride: usize, w: usize, h: usize, bpp: usize) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for y in 0..h {
        let row = &data[y * stride..y * stride + w * bpp];
        for &v in row {
            hash ^= u64::from(v);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

/// `adjust_strength` (`cdef_apply_tmpl.c:91-95`): yalnız luma birincil gücü.
///
/// `var == 0` → 0 (`ulog2(0)` çağrılmaz). `i = min(ulog2(var >> 6), 12)`.
#[inline]
fn adjust_strength(strength: i32, var: u32) -> i32 {
    if var == 0 {
        return 0;
    }
    let v = var >> 6;
    let i = if v != 0 {
        (31 - v.leading_zeros()).min(12) as i32
    } else {
        0
    };
    (strength * (4 + i) + 8) >> 4
}

/// Blok konumunun `cdef_idx[4]` yuvası (`decode.c:2662-2668`, `948-963`).
///
/// dav1d sb128'de `cur_sb_cdef_idx_ptr`'yi dört kadranlık dizinin başına
/// alır ve okumada `idx = ((bx & 16) >> 4) + ((by & 16) >> 3)` ekler; sb64'te
/// işaretçiyi **doğrudan bu yuvaya** kaydırır ve `ptr[0]` okur
/// (`decode.c:2664-2668`). İki durumda da dizideki **mutlak** yuva aynıdır,
/// bu yüzden sıfırlama (`decode/intra.rs` `reset_cdef`), okuma (`decode_b`) ve
/// kare ızgarasına yazma ([`CdefState::store_sb`]) tek bu formülü kullanır.
#[inline]
pub(crate) const fn cdef_idx_slot(bx: usize, by: usize) -> usize {
    ((bx & 16) >> 4) + ((by & 16) >> 3)
}

/// CDEF'in kare düzeyi girdileri: 8×8 `noskip_mask` ızgarası ve 64×64
/// `cdef_idx` ızgarası.
///
/// dav1d `noskip_mask` bitleri 4×4 sütun çözünürlüğündedir; 8×8 sınaması iki
/// bitin OR'udur (`decode.c:1952-1961`, `cdef_apply_tmpl.c:196-201`). Burada
/// aynı bilgi **8×8 başına tek bayt** olarak tutulur: bloktaki dört 4×4'ten
/// biri bile `!skip` ise bayt 1'dir. `cdef_idx` ise 64×64 başına `i8`'dir
/// (-1 = okunmadı/atlanır).
pub(crate) struct CdefState {
    /// 8×8 başına "bu blokta kodlanmış katsayı var" bayrağı.
    pub(crate) noskip: Vec<u8>,
    /// `noskip` satır adımı (8×8 sütun sayısı).
    pub(crate) noskip_stride: usize,
    /// Karo-yerel `noskip` X tabanı (8×8 biriminde; tam karede `0`).
    pub(crate) noskip_x_base: usize,
    /// Karo-yerel `noskip` Y tabanı (8×8 biriminde; tam karede `0`).
    pub(crate) noskip_y_base: usize,
    /// 64×64 başına `cdef_idx` (`-1` = unset).
    pub(crate) idx64: Vec<i32>,
    /// 64×64 sütun sayısı (tam karede veya karo-yerel).
    pub(crate) sb64w: usize,
    /// Karo-yerel `idx64` X tabanı (64×64 biriminde; tam karede `0`).
    pub(crate) sb64_x_base: usize,
    /// Karo-yerel `idx64` Y tabanı (64×64 biriminde; tam karede `0`).
    pub(crate) sb64_y_base: usize,
}

impl CdefState {
    #[cfg(test)]
    pub(crate) fn new(
        bw: usize,
        bh: usize,
        butce: &crate::frame::butce::Butce,
    ) -> crate::error::Result<Self> {
        Self::new_with_reuse(bw, bh, None, butce)
    }

    pub(crate) fn new_with_reuse(
        bw: usize,
        bh: usize,
        old: Option<Self>,
        butce: &crate::frame::butce::Butce,
    ) -> crate::error::Result<Self> {
        let noskip_stride = ((bw + 1) >> 1).max(1);
        let noskip_h = ((bh + 1) >> 1).max(1);
        let sb64w = ((bw + 15) >> 4).max(1);
        let sb64h = ((bh + 15) >> 4).max(1);
        let (old_noskip, old_idx64) = match old {
            Some(o) => (o.noskip, o.idx64),
            None => (Vec::new(), Vec::new()),
        };
        Ok(Self {
            noskip: butce.reuse_vec(old_noskip, noskip_stride * noskip_h, 0)?,
            noskip_stride,
            noskip_x_base: 0,
            noskip_y_base: 0,
            idx64: butce.reuse_vec(old_idx64, sb64w * sb64h, -1)?,
            sb64w,
            sb64_x_base: 0,
            sb64_y_base: 0,
        })
    }

    /// Paralel tile çözümü için yalnız karo dikdörtgeni (`[row_start..row_end) × [col_start..col_end)`)
    /// boyutunda geçici `CdefState` oluşturur (`F8b-2`).
    pub(crate) fn fork_for_tile(
        &self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) -> crate::error::Result<Self> {
        use crate::frame::butce::dolu_vec;
        let _ = self;
        let noskip_x_base = col_start >> 1;
        let noskip_y_base = row_start >> 1;
        let nx1 = (col_end + 1) >> 1;
        let ny1 = (row_end + 1) >> 1;
        let noskip_stride = nx1.saturating_sub(noskip_x_base).max(1);
        let noskip_h = ny1.saturating_sub(noskip_y_base).max(1);

        let sb64_x_base = col_start >> 4;
        let sb64_y_base = row_start >> 4;
        let sx1 = (col_end + 15) >> 4;
        let sy1 = (row_end + 15) >> 4;
        let sb64w = sx1.saturating_sub(sb64_x_base).max(1);
        let sb64h = sy1.saturating_sub(sb64_y_base).max(1);

        Ok(Self {
            noskip: dolu_vec(noskip_stride * noskip_h, 0)?,
            noskip_stride,
            noskip_x_base,
            noskip_y_base,
            idx64: dolu_vec(sb64w * sb64h, -1)?,
            sb64w,
            sb64_x_base,
            sb64_y_base,
        })
    }

    /// Tile yerel `other` `noskip` ve `idx64` alanlarını (`[row_start..row_end) × [col_start..col_end)`)
    /// kare `CdefState` içine birleştirir.
    pub(crate) fn merge_tile_from(
        &mut self,
        other: &Self,
        row_start: usize,
        row_end: usize,
        col_start: usize,
        col_end: usize,
    ) {
        let noskip_h = self.noskip.len() / self.noskip_stride;
        let ny0 = (row_start >> 1).min(noskip_h);
        let ny1 = ((row_end + 1) >> 1).min(noskip_h);
        let nx0 = (col_start >> 1).min(self.noskip_stride);
        let nx1 = ((col_end + 1) >> 1).min(self.noskip_stride);
        if nx0 < nx1 {
            let w = nx1 - nx0;
            let src_x0 = nx0 - other.noskip_x_base;
            for y in ny0..ny1 {
                let dst_off = y * self.noskip_stride + nx0;
                let src_off = (y - other.noskip_y_base) * other.noskip_stride + src_x0;
                self.noskip[dst_off..dst_off + w]
                    .copy_from_slice(&other.noskip[src_off..src_off + w]);
            }
        }
        let sb64h = self.idx64.len() / self.sb64w;
        let sy0 = (row_start >> 4).min(sb64h);
        let sy1 = ((row_end + 15) >> 4).min(sb64h);
        let sx0 = (col_start >> 4).min(self.sb64w);
        let sx1 = ((col_end + 15) >> 4).min(self.sb64w);
        if sx0 < sx1 {
            let w = sx1 - sx0;
            let src_x0 = sx0 - other.sb64_x_base;
            for y in sy0..sy1 {
                let dst_off = y * self.sb64w + sx0;
                let src_off = (y - other.sb64_y_base) * other.sb64w + src_x0;
                self.idx64[dst_off..dst_off + w]
                    .copy_from_slice(&other.idx64[src_off..src_off + w]);
            }
        }
    }

    /// `decode.c:1952-1961` — `!b->skip` olan bloğun kapsadığı 8×8'ler.
    ///
    /// dav1d maskeyi 4×4 sütun bitleri olarak yazar; CDEF 8×8 sınamasında
    /// `3 << (bx & 30)` kullanır, yani iki 4×4 sütununun **herhangi biri**
    /// işaretliyse 8×8 filtrelenir.
    pub(crate) fn mark_noskip(&mut self, bx4: usize, by4: usize, bw4: usize, bh4: usize) {
        if bw4 == 0 || bh4 == 0 {
            return;
        }
        let noskip_h = self.noskip.len() / self.noskip_stride;
        let x0 = (bx4 >> 1).saturating_sub(self.noskip_x_base);
        let x1 = ((bx4 + bw4 - 1) >> 1)
            .saturating_sub(self.noskip_x_base)
            .min(self.noskip_stride - 1);
        let y0 = (by4 >> 1).saturating_sub(self.noskip_y_base);
        let y1 = ((by4 + bh4 - 1) >> 1)
            .saturating_sub(self.noskip_y_base)
            .min(noskip_h - 1);
        for y in y0..=y1 {
            let base = y * self.noskip_stride;
            self.noskip[base + x0..=base + x1].fill(1);
        }
    }

    /// `decode_sb` sonrası `lflvl->cdef_idx[4]` değerlerini kare ızgarasına
    /// taşır (`decode.c:2657-2678` sıfırlama + `948-963` yazma).
    pub(crate) fn store_sb(&mut self, bx: usize, by: usize, sb128: bool, idx: &[i32; 4]) {
        let sb64h = self.idx64.len() / self.sb64w;
        if sb128 {
            let base_x = ((bx >> 5) << 1).saturating_sub(self.sb64_x_base);
            let base_y = ((by >> 5) << 1).saturating_sub(self.sb64_y_base);
            for q in 0..4 {
                let x = base_x + (q & 1);
                let y = base_y + (q >> 1);
                // 128'lik SB kare kenarını taşabilir; kare ızgarası 8'e
                // yuvarlanmış boyuttadır ve taşan kadran CDEF tarafından
                // okunmaz (`bx < bw`, `by < bh`).
                if x < self.sb64w && y < sb64h {
                    self.idx64[y * self.sb64w + x] = idx[q];
                }
            }
        } else {
            // dav1d `decode.c:2664-2668` + `948-963`: sb64'te değer süper
            // bloğun **kendi yuvasından** okunur; `idx[0]` komşu 64×64'ün
            // (ya da önceki SB'nin) değerini taşır ve bitler okunmaz.
            let x = (bx >> 4).saturating_sub(self.sb64_x_base);
            let y = (by >> 4).saturating_sub(self.sb64_y_base);
            if x < self.sb64w && y < sb64h {
                self.idx64[y * self.sb64w + x] = idx[cdef_idx_slot(bx, by)];
            }
        }
    }
}

/// Tek düzlem filtresini çağırır; dilimleri kare tamponundan ve yedeklerden
/// kurar.
///
/// `top`/`bottom` tampon düzeni: `2*bpp` bayt önek (mantıksal x = −2,−1) +
/// 2 satır × `stride`; dilim `[x*bpp..]`'den başlar, böylece filtre
/// sözleşmesindeki "indeks 0 = x−2" sağlanır (`bx == 0` iken önek okunmaz).
#[allow(clippy::too_many_arguments)]
#[inline]
fn filter_plane(
    pic: &mut PicBuf,
    filter_fn: CdefFilterFn,
    p: usize,
    bpp: usize,
    bit_depth: CdefBitDepth,
    stride: usize,
    x: usize,
    y: usize,
    size: CdefBlockSize,
    pri: u8,
    sec: u8,
    dir: u8,
    damping: u8,
    edges: CdefEdgeFlags,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    have_left: bool,
    have_top: bool,
    have_bottom: bool,
    trace: Option<&TraceCfg>,
    frame_no: u32,
) {
    let off = y * stride + x * bpp;
    let h = size.h();
    let left = if have_left { &left[..h * 2 * bpp] } else { &[] };
    let top = if have_top { &top[x * bpp..] } else { &[] };
    let bottom = if have_bottom { &bottom[x * bpp..] } else { &[] };
    let params = CdefFilterParams {
        bit_depth,
        size,
        pri_strength: pri,
        sec_strength: sec,
        dir: dir & 7,
        damping,
        edges,
    };
    crate::dsp::profil_olc!(
        CdefFb,
        filter_fn(&mut pic.data[p][off..], stride, left, top, bottom, params)
    );
    if let Some(t) = trace
        && t.on(frame_no, p)
    {
        let hash = fnv_block(&pic.data[p][off..], stride, size.w(), size.h(), bpp);
        crate::trace_eprintln!(
            "C {frame_no} {p} {x} {y} {} {} {pri} {sec} {} {damping} {} {hash}",
            size.w(),
            size.h(),
            params.dir,
            edges.bits()
        );
    }
}

/// Kare üzerine CDEF uygular (`dav1d_cdef_brow`).
///
/// `state` çözme sırasında doldurulmuş `noskip_mask` ve `cdef_idx`
/// ızgaralarıdır. Sayaçlar [`ToolCounts`] içine yazılır (davranışı değiştirmez).
pub(crate) fn apply(
    pic: &mut PicBuf,
    hdr: &FrameHeader,
    chroma: ChromaFormat,
    bd: BitDepth,
    dsp: &CdefDsp,
    state: &CdefState,
    tools: &mut ToolCounts,
    frame_no: u32,
    butce: &crate::frame::butce::Butce,
    pool: &crate::thread::WorkerPool,
) -> crate::error::Result<()> {
    let trace = trace_cfg();
    let cdef = &hdr.cdef;
    // Karede hiçbir güç sıfırdan farklı değilse CDEF fiilen kapalıdır
    // (`dav1d_filter_sbrow` yalnız `seq_hdr->cdef` iken çağırır).
    if !cdef.y_strength.iter().any(|&v| v != 0) && !cdef.uv_strength.iter().any(|&v| v != 0) {
        return Ok(());
    }
    let bit_depth = match bd {
        BitDepth::Eight => CdefBitDepth::B8,
        BitDepth::Ten => CdefBitDepth::B10,
        BitDepth::Twelve => CdefBitDepth::B12,
    };
    let bpp = bd.bytes_per_sample();
    let bitdepth_min_8 = bit_depth.min_8();
    let damping = u32::from(cdef.damping) + bitdepth_min_8;
    let has_chroma = chroma.has_chroma();
    let (ss_hor, ss_ver) = chroma.chroma_shift();
    let (ss_hor, ss_ver) = (ss_hor as usize, ss_ver as usize);
    let uv_idx = match chroma {
        ChromaFormat::C444 => 0,
        ChromaFormat::C422 => 1,
        _ => 2,
    };
    let uv_size = CdefBlockSize::ALL[uv_idx];
    let uv_dir = &UV_DIRS[usize::from(chroma == ChromaFormat::C422)];

    // F3c: `dir` ve `fb[blok boyutu]` yuvaları (karenin bit derinliği tablosu).
    let dir_fn = dsp.dir;
    let fb_y: CdefFilterFn = dsp.fb[CdefBlockSize::W8H8.index()];
    let fb_uv: CdefFilterFn = dsp.fb[uv_size.index()];

    let bw = ((pic.width + 7) >> 3) << 1;
    let bh = ((pic.height + 7) >> 3) << 1;
    let n_planes = if has_chroma { 3 } else { 1 };
    let y_stride = pic.stride[0];
    let uv_stride = pic.stride[1];

    let plane_stride = |p: usize| if p == 0 { y_stride } else { uv_stride };
    let plane_y = |p: usize, by4: usize| {
        if p == 0 { by4 * 4 } else { (by4 * 4) >> ss_ver }
    };
    let plane_h = |p: usize| if p == 0 { 8usize } else { 8usize >> ss_ver };

    // Üst iki satır: bir önceki bandın **filtre öncesi** son iki satırı
    // (dav1d `backup2lines`). Alt iki satır: henüz filtrelenmemiş canlı
    // satırlar (dav1d `bot = bptrs + h`); bant başında kopyalanır.
    // P38: satır tamponları (en çok 9 × (2 · bpp + 2 · stride) bayt) bütçe
    // defterinden geçici ayrılır; işlev dönünce defterden düşer (`n_threads`
    // değerinden bağımsız olarak aynı tepe bütçeyi üretir).
    let mk_lines = |stride: usize| butce.gecici(2 * bpp + 2 * stride, 0u8);
    let mut top_cur = [
        mk_lines(y_stride)?,
        mk_lines(uv_stride)?,
        mk_lines(uv_stride)?,
    ];
    let mut top_next = [
        mk_lines(y_stride)?,
        mk_lines(uv_stride)?,
        mk_lines(uv_stride)?,
    ];
    let mut bot = [
        mk_lines(y_stride)?,
        mk_lines(uv_stride)?,
        mk_lines(uv_stride)?,
    ];

    let n_bands = (bh + 1) >> 1;
    if pool.is_parallel() && n_bands > 1 && trace.is_none() {
        return apply_parallel_bands(
            pic,
            cdef,
            chroma,
            bit_depth,
            bpp,
            bitdepth_min_8,
            damping,
            has_chroma,
            ss_hor,
            ss_ver,
            uv_size,
            uv_dir,
            dir_fn,
            fb_y,
            fb_uv,
            bw,
            bh,
            n_planes,
            y_stride,
            uv_stride,
            n_bands,
            state,
            tools,
            pool,
        );
    }

    // Sol/sağ ikişer sütun (paketli): en çok 8 satır × 2 piksel × 2 bayt.
    let plane_x = |p: usize, bx4: usize| {
        if p == 0 { bx4 * 4 } else { (bx4 * 4) >> ss_hor }
    };
    let plane_w = |p: usize| if p == 0 { 8usize } else { 8usize >> ss_hor };
    let mut left_bak = [[0u8; 32]; 3];
    let mut right_save = [[0u8; 32]; 3];

    for band in 0..n_bands {
        let by = band * 2;
        let have_top = by > 0;
        let have_bottom = by + 2 < bh;
        // Bant başı: son iki satırı (filtre öncesi) ve alt iki canlı satırı
        // kopyala. `top_cur` bir önceki bandın yedeğidir.
        for p in 0..n_planes {
            let stride = plane_stride(p);
            let y0 = plane_y(p, by);
            let h = plane_h(p);
            let src = &pic.data[p];
            let prefix = 2 * bpp;
            for r in 0..2 {
                let srow = y0 + h - 2 + r;
                let so = srow * stride;
                let dof = prefix + r * stride;
                top_next[p][dof..dof + stride].copy_from_slice(&src[so..so + stride]);
            }
            if have_bottom {
                for r in 0..2 {
                    let srow = y0 + h + r;
                    let so = srow * stride;
                    let dof = prefix + r * stride;
                    bot[p][dof..dof + stride].copy_from_slice(&src[so..so + stride]);
                }
            }
        }

        // `edges` (`cdef_apply_tmpl.c:128-131`): kare kenarında bayrak yok;
        // tile sınırı CDEF'i etkilemez.
        let mut band_edges = CdefEdgeFlags::NONE;
        if have_top {
            band_edges = band_edges.union(CdefEdgeFlags::HAVE_TOP);
        }
        if have_bottom {
            band_edges = band_edges.union(CdefEdgeFlags::HAVE_BOTTOM);
        }

        let sb64y = by >> 4;
        // `prev_flag` bandın tamamı boyunca yaşar (`cdef_apply_tmpl.c:143`);
        // 64×64 atlanınca sıfırlanır (`:154`).
        let mut prev_saved = [false; 2];
        for sbx in 0..state.sb64w {
            let row = sb64y * state.sb64w + sbx;
            let idx_raw = state.idx64.get(row).copied().unwrap_or(-1);
            let Some(idx) = usize::try_from(idx_raw)
                .ok()
                .filter(|&i| i < cdef.y_strength.len())
            else {
                tools.cdef_skipped64 += 1;
                prev_saved = [false; 2];
                continue;
            };
            let y_lvl = cdef.y_strength[idx];
            let uv_lvl = if has_chroma { cdef.uv_strength[idx] } else { 0 };
            if y_lvl == 0 && uv_lvl == 0 {
                tools.cdef_skipped64 += 1;
                prev_saved = [false; 2];
                continue;
            }
            let y_pri_lvl = (i32::from(y_lvl) >> 2) << bitdepth_min_8;
            let mut y_sec_lvl = i32::from(y_lvl & 3);
            y_sec_lvl += i32::from(y_sec_lvl == 3);
            y_sec_lvl <<= bitdepth_min_8;
            let uv_pri_lvl = (i32::from(uv_lvl) >> 2) << bitdepth_min_8;
            let mut uv_sec_lvl = i32::from(uv_lvl & 3);
            uv_sec_lvl += i32::from(uv_sec_lvl == 3);
            uv_sec_lvl <<= bitdepth_min_8;

            let bx_end = ((sbx + 1) << 4).min(bw);
            let mut bx = sbx << 4;
            while bx < bx_end {
                let have_left = bx > 0;
                let have_right = bx + 2 < bw;
                let noskip_ok = state
                    .noskip
                    .get((by >> 1) * state.noskip_stride + (bx >> 1))
                    .copied()
                    .unwrap_or(0)
                    != 0;
                if !noskip_ok {
                    tools.cdef_skipped8 += 1;
                    prev_saved = [false; 2];
                    bx += 2;
                    continue;
                }

                let flag = [y_lvl != 0, has_chroma && uv_lvl != 0];
                // Sol iki sütun: önceki blok filtrelendiyse yedekten, değilse
                // canlıdan (`do_left` — `cdef_apply_tmpl.c:184-202`).
                if have_left {
                    for p in 0..n_planes {
                        let group = usize::from(p != 0);
                        let h = plane_h(p);
                        if prev_saved[group] {
                            left_bak[p][..h * 2 * bpp]
                                .copy_from_slice(&right_save[p][..h * 2 * bpp]);
                        } else {
                            let stride = plane_stride(p);
                            let x0 = plane_x(p, bx) - 2;
                            let y0 = plane_y(p, by);
                            let src = &pic.data[p];
                            for r in 0..h {
                                let so = (y0 + r) * stride + x0 * bpp;
                                let dof = r * 2 * bpp;
                                left_bak[p][dof..dof + 2 * bpp]
                                    .copy_from_slice(&src[so..so + 2 * bpp]);
                            }
                        }
                    }
                }

                // Sağ iki sütun: filtre öncesi canlı değerlerden yedekle
                // (`backup2x8`, `lr_bak[!bit]` — `cdef_apply_tmpl.c:203-207`).
                let mut new_saved = [false; 2];
                if have_right {
                    for p in 0..n_planes {
                        let group = usize::from(p != 0);
                        if !flag[group] {
                            continue;
                        }
                        let stride = plane_stride(p);
                        let x0 = plane_x(p, bx) + plane_w(p) - 2;
                        let y0 = plane_y(p, by);
                        let h = plane_h(p);
                        let src = &pic.data[p];
                        for r in 0..h {
                            let so = (y0 + r) * stride + x0 * bpp;
                            let dof = r * 2 * bpp;
                            right_save[p][dof..dof + 2 * bpp]
                                .copy_from_slice(&src[so..so + 2 * bpp]);
                        }
                        new_saved[group] = true;
                    }
                }
                // Yön + varyans yalnız luma ya da kroma birincil gücü sıfırdan
                // farklıysa hesaplanır (`cdef_apply_tmpl.c:210-212`).
                let mut dir = 0u8;
                let mut variance = 0u32;
                if y_pri_lvl != 0 || uv_pri_lvl != 0 {
                    let off = plane_y(0, by) * y_stride + plane_x(0, bx) * bpp;
                    let res = crate::dsp::profil_olc!(
                        CdefDir,
                        dir_fn(&pic.data[0][off..], y_stride, bit_depth)
                    );
                    dir = res.dir;
                    variance = res.var;
                    tools.cdef_dir_hist[usize::from(dir & 7)] += 1;
                    if let Some(t) = trace.as_ref()
                        && t.on(frame_no, 0)
                    {
                        crate::trace_eprintln!(
                            "D {frame_no} {dir} {variance} {} {}",
                            plane_x(0, bx),
                            plane_y(0, by)
                        );
                    }
                }

                let mut edges = band_edges;
                if have_left {
                    edges = edges.union(CdefEdgeFlags::HAVE_LEFT);
                }
                if have_right {
                    edges = edges.union(CdefEdgeFlags::HAVE_RIGHT);
                }

                let mut any = false;
                // --- luma (`fb[0]`) -------------------------------------
                if y_pri_lvl != 0 {
                    let adj = adjust_strength(y_pri_lvl, variance);
                    if adj != 0 || y_sec_lvl != 0 {
                        filter_plane(
                            pic,
                            fb_y,
                            0,
                            bpp,
                            bit_depth,
                            y_stride,
                            plane_x(0, bx),
                            plane_y(0, by),
                            CdefBlockSize::W8H8,
                            adj as u8,
                            y_sec_lvl as u8,
                            dir,
                            damping as u8,
                            edges,
                            &left_bak[0],
                            &top_cur[0],
                            &bot[0],
                            have_left,
                            have_top,
                            have_bottom,
                            trace.as_ref(),
                            frame_no,
                        );
                        tools.cdef_luma_blocks += 1;
                        any = true;
                    } else {
                        tools.cdef_adj_zero += 1;
                    }
                } else if y_sec_lvl != 0 {
                    filter_plane(
                        pic,
                        fb_y,
                        0,
                        bpp,
                        bit_depth,
                        y_stride,
                        plane_x(0, bx),
                        plane_y(0, by),
                        CdefBlockSize::W8H8,
                        0,
                        y_sec_lvl as u8,
                        0,
                        damping as u8,
                        edges,
                        &left_bak[0],
                        &top_cur[0],
                        &bot[0],
                        have_left,
                        have_top,
                        have_bottom,
                        trace.as_ref(),
                        frame_no,
                    );
                    tools.cdef_luma_blocks += 1;
                    any = true;
                }

                // --- kroma (`fb[uv_idx]`, damping − 1) -----------------
                if has_chroma && uv_lvl != 0 {
                    let uvdir = if uv_pri_lvl != 0 {
                        uv_dir[usize::from(dir & 7)]
                    } else {
                        0
                    };
                    if uv_pri_lvl != 0 && chroma == ChromaFormat::C422 {
                        tools.cdef_uv422_pri += 1;
                    }
                    for p in 1..3 {
                        filter_plane(
                            pic,
                            fb_uv,
                            p,
                            bpp,
                            bit_depth,
                            uv_stride,
                            plane_x(p, bx),
                            plane_y(p, by),
                            uv_size,
                            uv_pri_lvl as u8,
                            uv_sec_lvl as u8,
                            uvdir,
                            (damping - 1) as u8,
                            edges,
                            &left_bak[p],
                            &top_cur[p],
                            &bot[p],
                            have_left,
                            have_top,
                            have_bottom,
                            trace.as_ref(),
                            frame_no,
                        );
                        tools.cdef_chroma_blocks += 1;
                        any = true;
                    }
                }
                if any {
                    tools.cdef_8x8_filtered += 1;
                }

                prev_saved = new_saved;
                bx += 2;
            }
        }

        core::mem::swap(&mut top_cur, &mut top_next);
    }
    Ok(())
}

/// 8-satırlık CDEF bantlarını `WorkerPool` üzerinde paralel süzüp
/// deterministik bant sırasıyla `pic.data` ve `tools` içine birleştirir.
fn apply_parallel_bands(
    pic: &mut PicBuf,
    cdef: &crate::bitstream::frame_header::CdefParams,
    chroma: ChromaFormat,
    bit_depth: CdefBitDepth,
    bpp: usize,
    bitdepth_min_8: u32,
    damping: u32,
    has_chroma: bool,
    ss_hor: usize,
    ss_ver: usize,
    uv_size: CdefBlockSize,
    uv_dir: &'static [u8; 8],
    dir_fn: crate::dsp::cdef::CdefDirFn,
    fb_y: CdefFilterFn,
    fb_uv: CdefFilterFn,
    bw: usize,
    bh: usize,
    n_planes: usize,
    y_stride: usize,
    uv_stride: usize,
    n_bands: usize,
    state: &CdefState,
    tools: &mut ToolCounts,
    pool: &crate::thread::WorkerPool,
) -> crate::error::Result<()> {
    use crate::frame::butce::dolu_vec;
    use alloc::sync::Arc;

    struct BandChunk {
        band_start: usize,
        band_end: usize,
        dst: [Vec<u8>; 3],
        tools: ToolCounts,
    }

    let plane_stride = |p: usize| if p == 0 { y_stride } else { uv_stride };
    let plane_y = |p: usize, by4: usize| {
        if p == 0 { by4 * 4 } else { (by4 * 4) >> ss_ver }
    };
    let plane_h = |p: usize| if p == 0 { 8usize } else { 8usize >> ss_ver };

    let n_chunks = (pool.n_threads() * 2).clamp(1, n_bands);
    let bands_per_chunk = n_bands.div_ceil(n_chunks);

    // Her bant dilimi için filtre öncesi satırların yerel kopyasını (+ `2 * bpp` sağ
    // kenar SIMD dolgu payı) oluştur; `pic.data` ise komşu `top`/`bottom`
    // okumaları için salt-okunur kalır.
    let mut jobs: Vec<BandChunk> = Vec::with_capacity(n_chunks);
    let mut band_cur = 0usize;
    while band_cur < n_bands {
        let band_end = (band_cur + bands_per_chunk).min(n_bands);
        jobs.push(BandChunk {
            band_start: band_cur,
            band_end,
            dst: [Vec::new(), Vec::new(), Vec::new()],
            tools: ToolCounts::default(),
        });
        band_cur = band_end;
    }

    // Salt-okunur pre-CDEF düzlemlerini ve CDEF durumunu sıfır-kopya paylaşmak
    // için geçici olarak `Arc` içine taşı, paralel bantlar bittiğinde geri al.
    let pre_p0 = Arc::new(core::mem::take(&mut pic.data[0]));
    let pre_p1 = Arc::new(core::mem::take(&mut pic.data[1]));
    let pre_p2 = Arc::new(core::mem::take(&mut pic.data[2]));
    let noskip = Arc::new(state.noskip.clone());
    let idx64 = Arc::new(state.idx64.clone());
    let sb64w = state.sb64w;
    let noskip_stride = state.noskip_stride;
    let cdef_hdr = *cdef;

    let p0_w = Arc::clone(&pre_p0);
    let p1_w = Arc::clone(&pre_p1);
    let p2_w = Arc::clone(&pre_p2);

    let res = pool.execute_tasks(&mut jobs, move |_wid, _idx, job: &mut BandChunk| {
        let pre_planes: [&[u8]; 3] = [p0_w.as_slice(), p1_w.as_slice(), p2_w.as_slice()];
        let chunk_bands = job.band_end - job.band_start;
        let by0 = job.band_start * 2;
        for p in 0..n_planes {
            let stride = if p == 0 { y_stride } else { uv_stride };
            let y0 = if p == 0 { by0 * 4 } else { (by0 * 4) >> ss_ver };
            let h = (if p == 0 { 8usize } else { 8usize >> ss_ver }) * chunk_bands;
            let byte_len = h * stride;
            let mut buf = dolu_vec(byte_len + 2 * bpp, 0u8)?;
            let src_end = (y0 * stride + byte_len).min(pre_planes[p].len());
            if y0 * stride < src_end {
                let copy_len = src_end - y0 * stride;
                buf[..copy_len].copy_from_slice(&pre_planes[p][y0 * stride..src_end]);
            }
            job.dst[p] = buf;
        }

        let mut top_buf = [
            dolu_vec(2 * bpp + 2 * y_stride, 0u8)?,
            dolu_vec(2 * bpp + 2 * uv_stride, 0u8)?,
            dolu_vec(2 * bpp + 2 * uv_stride, 0u8)?,
        ];
        let mut bot_buf = [
            dolu_vec(2 * bpp + 2 * y_stride, 0u8)?,
            dolu_vec(2 * bpp + 2 * uv_stride, 0u8)?,
            dolu_vec(2 * bpp + 2 * uv_stride, 0u8)?,
        ];
        let mut left_bak = [[0u8; 32]; 3];
        let mut right_save = [[0u8; 32]; 3];
        let prefix = 2 * bpp;

        for band in job.band_start..job.band_end {
            let local_band = band - job.band_start;
            let band_y_off = [
                local_band * 8 * y_stride,
                local_band * (8 >> ss_ver) * uv_stride,
                local_band * (8 >> ss_ver) * uv_stride,
            ];
            let by = band * 2;
            let have_top = by > 0;
            let have_bottom = by + 2 < bh;

            for p in 0..n_planes {
                let stride = if p == 0 { y_stride } else { uv_stride };
                let y0 = if p == 0 { by * 4 } else { (by * 4) >> ss_ver };
                let h = if p == 0 { 8usize } else { 8usize >> ss_ver };
                let src = pre_planes[p];
                if have_top {
                    for r in 0..2 {
                        let srow = y0 - 2 + r;
                        let so = srow * stride;
                        let dof = prefix + r * stride;
                        top_buf[p][dof..dof + stride].copy_from_slice(&src[so..so + stride]);
                    }
                }
                if have_bottom {
                    for r in 0..2 {
                        let srow = y0 + h + r;
                        let so = srow * stride;
                        let dof = prefix + r * stride;
                        bot_buf[p][dof..dof + stride].copy_from_slice(&src[so..so + stride]);
                    }
                }
            }

            let mut band_edges = CdefEdgeFlags::NONE;
            if have_top {
                band_edges = band_edges.union(CdefEdgeFlags::HAVE_TOP);
            }
            if have_bottom {
                band_edges = band_edges.union(CdefEdgeFlags::HAVE_BOTTOM);
            }

            let sb64y = by >> 4;
            let mut prev_saved = [false; 2];
            for sbx in 0..sb64w {
                let row = sb64y * sb64w + sbx;
                let idx_raw = idx64.get(row).copied().unwrap_or(-1);
                let Some(idx) = usize::try_from(idx_raw)
                    .ok()
                    .filter(|&i| i < cdef_hdr.y_strength.len())
                else {
                    job.tools.cdef_skipped64 += 1;
                    prev_saved = [false; 2];
                    continue;
                };
                let y_lvl = cdef_hdr.y_strength[idx];
                let uv_lvl = if has_chroma {
                    cdef_hdr.uv_strength[idx]
                } else {
                    0
                };
                if y_lvl == 0 && uv_lvl == 0 {
                    job.tools.cdef_skipped64 += 1;
                    prev_saved = [false; 2];
                    continue;
                };
                let y_pri_lvl = (i32::from(y_lvl) >> 2) << bitdepth_min_8;
                let mut y_sec_lvl = i32::from(y_lvl & 3);
                y_sec_lvl += i32::from(y_sec_lvl == 3);
                y_sec_lvl <<= bitdepth_min_8;
                let uv_pri_lvl = (i32::from(uv_lvl) >> 2) << bitdepth_min_8;
                let mut uv_sec_lvl = i32::from(uv_lvl & 3);
                uv_sec_lvl += i32::from(uv_sec_lvl == 3);
                uv_sec_lvl <<= bitdepth_min_8;

                let bx_end = ((sbx + 1) << 4).min(bw);
                let mut bx = sbx << 4;
                while bx < bx_end {
                    let have_left = bx > 0;
                    let have_right = bx + 2 < bw;
                    let noskip_ok = noskip
                        .get((by >> 1) * noskip_stride + (bx >> 1))
                        .copied()
                        .unwrap_or(0)
                        != 0;
                    if !noskip_ok {
                        job.tools.cdef_skipped8 += 1;
                        prev_saved = [false; 2];
                        bx += 2;
                        continue;
                    }

                    let flag = [y_lvl != 0, has_chroma && uv_lvl != 0];
                    if have_left {
                        for p in 0..n_planes {
                            let group = usize::from(p != 0);
                            let h = if p == 0 { 8usize } else { 8usize >> ss_ver };
                            if prev_saved[group] {
                                left_bak[p][..h * 2 * bpp]
                                    .copy_from_slice(&right_save[p][..h * 2 * bpp]);
                            } else {
                                let stride = if p == 0 { y_stride } else { uv_stride };
                                let x0 = (if p == 0 { bx * 4 } else { (bx * 4) >> ss_hor }) - 2;
                                for r in 0..h {
                                    let so = band_y_off[p] + r * stride + x0 * bpp;
                                    let dof = r * 2 * bpp;
                                    left_bak[p][dof..dof + 2 * bpp]
                                        .copy_from_slice(&job.dst[p][so..so + 2 * bpp]);
                                }
                            }
                        }
                    }

                    let mut new_saved = [false; 2];
                    if have_right {
                        for p in 0..n_planes {
                            let group = usize::from(p != 0);
                            if !flag[group] {
                                continue;
                            }
                            let stride = if p == 0 { y_stride } else { uv_stride };
                            let x0 = (if p == 0 { bx * 4 } else { (bx * 4) >> ss_hor })
                                + (if p == 0 { 8usize } else { 8usize >> ss_hor })
                                - 2;
                            let h = if p == 0 { 8usize } else { 8usize >> ss_ver };
                            for r in 0..h {
                                let so = band_y_off[p] + r * stride + x0 * bpp;
                                let dof = r * 2 * bpp;
                                right_save[p][dof..dof + 2 * bpp]
                                    .copy_from_slice(&job.dst[p][so..so + 2 * bpp]);
                            }
                            new_saved[group] = true;
                        }
                    }

                    let mut dir = 0u8;
                    let mut variance = 0u32;
                    if y_pri_lvl != 0 || uv_pri_lvl != 0 {
                        let off = band_y_off[0] + (bx * 4) * bpp;
                        let res = crate::dsp::profil_olc!(
                            CdefDir,
                            dir_fn(&job.dst[0][off..], y_stride, bit_depth)
                        );
                        dir = res.dir;
                        variance = res.var;
                        job.tools.cdef_dir_hist[usize::from(dir & 7)] += 1;
                    }

                    let mut edges = band_edges;
                    if have_left {
                        edges = edges.union(CdefEdgeFlags::HAVE_LEFT);
                    }
                    if have_right {
                        edges = edges.union(CdefEdgeFlags::HAVE_RIGHT);
                    }

                    let mut any = false;
                    let x_y = bx * 4;
                    if y_pri_lvl != 0 {
                        let adj = adjust_strength(y_pri_lvl, variance);
                        if adj != 0 || y_sec_lvl != 0 {
                            let left_s = if have_left {
                                &left_bak[0][..8 * 2 * bpp]
                            } else {
                                &[]
                            };
                            let top_s = if have_top {
                                &top_buf[0][x_y * bpp..]
                            } else {
                                &[]
                            };
                            let bot_s = if have_bottom {
                                &bot_buf[0][x_y * bpp..]
                            } else {
                                &[]
                            };
                            let params = CdefFilterParams {
                                bit_depth,
                                size: CdefBlockSize::W8H8,
                                pri_strength: adj as u8,
                                sec_strength: y_sec_lvl as u8,
                                dir: dir & 7,
                                damping: damping as u8,
                                edges,
                            };
                            crate::dsp::profil_olc!(
                                CdefFb,
                                fb_y(
                                    &mut job.dst[0][band_y_off[0] + x_y * bpp..],
                                    y_stride,
                                    left_s,
                                    top_s,
                                    bot_s,
                                    params
                                )
                            );
                            job.tools.cdef_luma_blocks += 1;
                            any = true;
                        } else {
                            job.tools.cdef_adj_zero += 1;
                        }
                    } else if y_sec_lvl != 0 {
                        let left_s = if have_left {
                            &left_bak[0][..8 * 2 * bpp]
                        } else {
                            &[]
                        };
                        let top_s = if have_top {
                            &top_buf[0][x_y * bpp..]
                        } else {
                            &[]
                        };
                        let bot_s = if have_bottom {
                            &bot_buf[0][x_y * bpp..]
                        } else {
                            &[]
                        };
                        let params = CdefFilterParams {
                            bit_depth,
                            size: CdefBlockSize::W8H8,
                            pri_strength: 0,
                            sec_strength: y_sec_lvl as u8,
                            dir: 0,
                            damping: damping as u8,
                            edges,
                        };
                        crate::dsp::profil_olc!(
                            CdefFb,
                            fb_y(
                                &mut job.dst[0][band_y_off[0] + x_y * bpp..],
                                y_stride,
                                left_s,
                                top_s,
                                bot_s,
                                params
                            )
                        );
                        job.tools.cdef_luma_blocks += 1;
                        any = true;
                    }

                    if has_chroma && uv_lvl != 0 {
                        let uvdir = if uv_pri_lvl != 0 {
                            uv_dir[usize::from(dir & 7)]
                        } else {
                            0
                        };
                        if uv_pri_lvl != 0 && chroma == ChromaFormat::C422 {
                            job.tools.cdef_uv422_pri += 1;
                        }
                        let x_uv = (bx * 4) >> ss_hor;
                        let h_uv = uv_size.h();
                        for p in 1..3 {
                            let left_s = if have_left {
                                &left_bak[p][..h_uv * 2 * bpp]
                            } else {
                                &[]
                            };
                            let top_s = if have_top {
                                &top_buf[p][x_uv * bpp..]
                            } else {
                                &[]
                            };
                            let bot_s = if have_bottom {
                                &bot_buf[p][x_uv * bpp..]
                            } else {
                                &[]
                            };
                            let params = CdefFilterParams {
                                bit_depth,
                                size: uv_size,
                                pri_strength: uv_pri_lvl as u8,
                                sec_strength: uv_sec_lvl as u8,
                                dir: uvdir & 7,
                                damping: (damping - 1) as u8,
                                edges,
                            };
                            crate::dsp::profil_olc!(
                                CdefFb,
                                fb_uv(
                                    &mut job.dst[p][band_y_off[p] + x_uv * bpp..],
                                    uv_stride,
                                    left_s,
                                    top_s,
                                    bot_s,
                                    params
                                )
                            );
                            job.tools.cdef_chroma_blocks += 1;
                            any = true;
                        }
                    }
                    if any {
                        job.tools.cdef_8x8_filtered += 1;
                    }
                    prev_saved = new_saved;
                    bx += 2;
                }
            }
        }
        Ok(())
    });

    // `Arc` sahipliklerini `pic.data` içine geri taşı (kopyasız).
    pic.data[0] = Arc::try_unwrap(pre_p0).unwrap_or_else(|a| (*a).clone());
    pic.data[1] = Arc::try_unwrap(pre_p1).unwrap_or_else(|a| (*a).clone());
    pic.data[2] = Arc::try_unwrap(pre_p2).unwrap_or_else(|a| (*a).clone());
    res?;

    // Bant dilimi çıktılarını ve araç sayaçlarını deterministik sırayla birleştir.
    for job in jobs {
        let chunk_bands = job.band_end - job.band_start;
        let by0 = job.band_start * 2;
        for p in 0..n_planes {
            let stride = plane_stride(p);
            let y0 = plane_y(p, by0);
            let h = plane_h(p) * chunk_bands;
            let byte_len = h * stride;
            let dst_end = (y0 * stride + byte_len).min(pic.data[p].len());
            if y0 * stride < dst_end {
                let copy_len = dst_end - y0 * stride;
                pic.data[p][y0 * stride..dst_end].copy_from_slice(&job.dst[p][..copy_len]);
            }
        }
        tools.cdef_skipped64 += job.tools.cdef_skipped64;
        tools.cdef_skipped8 += job.tools.cdef_skipped8;
        tools.cdef_8x8_filtered += job.tools.cdef_8x8_filtered;
        tools.cdef_luma_blocks += job.tools.cdef_luma_blocks;
        tools.cdef_chroma_blocks += job.tools.cdef_chroma_blocks;
        tools.cdef_adj_zero += job.tools.cdef_adj_zero;
        tools.cdef_uv422_pri += job.tools.cdef_uv422_pri;
        for d in 0..8 {
            tools.cdef_dir_hist[d] += job.tools.cdef_dir_hist[d];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    /// `adjust_strength` (`cdef_apply_tmpl.c:91-95`): `var == 0` → 0; aksi
    /// hâlde `(strength * (4 + min(ulog2(var >> 6), 12)) + 8) >> 4`.
    #[test]
    fn adjust_strength_dav1d_formulu() {
        assert_eq!(adjust_strength(15, 0), 0);
        // var >> 6 == 1 → i = 0 → (15*4 + 8) >> 4 = 4
        assert_eq!(adjust_strength(15, 64), 4);
        // var >> 6 == 2 → i = 1 → (15*5 + 8) >> 4 = 5
        assert_eq!(adjust_strength(15, 128), 5);
        // i büyük değerde 12'ye kırpılır: (15*16 + 8) >> 4 = 15
        assert_eq!(adjust_strength(15, u32::MAX), 15);
        // 8-bit güç 4 ve var 0 → 0; var 64 → (16 + 8) >> 4 = 1
        assert_eq!(adjust_strength(4, 64), 1);
    }

    /// `noskip_mask`: bir 4×4 blok yalnız **kendi** 8×8'ini işaretler
    /// (`decode.c:1952-1961`; CDEF sınaması iki 4×4 sütununun OR'udur).
    #[test]
    fn noskip_8x8_kaplamasi() {
        let mut s =
            CdefState::new(4, 4, &crate::frame::butce::Butce::sinirsiz()).expect("cdef durumu"); // bw=bh=4 (16 px)
        s.mark_noskip(3, 2, 1, 1);
        assert_eq!(s.noskip_stride, 2);
        assert_eq!(s.noskip[3], 1, "8x8 (1,1) işaretlenmeli");
        assert_eq!(s.noskip[2], 0);
        assert_eq!(s.noskip[1], 0);
        // 8×8'i tamamen kapsayan blok dört hücreyi de işaretler.
        let mut s =
            CdefState::new(4, 4, &crate::frame::butce::Butce::sinirsiz()).expect("cdef durumu");
        s.mark_noskip(0, 0, 4, 4);
        assert!(s.noskip.iter().all(|&v| v == 1));
    }

    /// `cdef_idx` kare ızgarası: 64×64 SB'si kendi hücresine, 128×128 SB'si
    /// dört kadrana yazılır; kare dışına taşan kadran yazılmaz.
    #[test]
    fn cdef_idx_kare_izgarasi() {
        let mut s =
            CdefState::new(32, 4, &crate::frame::butce::Butce::sinirsiz()).expect("cdef durumu"); // bw=32 (128 px), bh=4 (16 px)
        assert_eq!(s.sb64w, 2);
        s.store_sb(0, 0, false, &[5, -1, -1, -1]);
        assert_eq!(s.idx64[0], 5);
        // sb64'te ikinci 64×64'ün yuvası `cdef_idx_slot(16, 0) == 1`'dir;
        // `idx[0]` yazmak önceki (burada 5) değeri taşırdı.
        s.store_sb(16, 0, false, &[-1, 3, -1, -1]);
        assert_eq!(s.idx64[1], 3);
        // 128 SB: 32×16 px karede 4 kadranın yalnız 2'si ızgarada.
        let mut s =
            CdefState::new(32, 4, &crate::frame::butce::Butce::sinirsiz()).expect("cdef durumu");
        s.store_sb(0, 0, true, &[1, 2, 3, 4]);
        assert_eq!(s.idx64.len(), 2);
        assert_eq!(s.idx64[0], 1);
        assert_eq!(s.idx64[1], 2);
        // Taşma yok (panik yok) ve kare sınırı dışındaki kadran yazılmaz.
        let mut s =
            CdefState::new(16, 16, &crate::frame::butce::Butce::sinirsiz()).expect("cdef durumu"); // 1×1 64x64
        s.store_sb(0, 0, true, &[1, 2, 3, 4]);
        assert_eq!(s.idx64, vec![1]);
    }

    /// sb64'te `cdef_idx` yuvası: **yan yana ve alt alta** 64×64 süper
    /// bloklar. Sıfırlama → okuma → kare ızgarasına yazma aynı yuvayı
    /// kullanmalıdır (`decode.c:2664-2668` + `948-963`).
    ///
    /// Dizi gerçek çözücüde olduğu gibi süper bloklar arasında **kalıcıdır**;
    /// yalnız o süper bloğun yuvası sıfırlanır. Eski hata (okuma/yazma
    /// `idx = 0`) bu testte ikinci süper bloktan itibaren bayat değer üretir.
    #[test]
    fn cdef_idx_yuvasi_sb64_yan_yana_alt_alta() {
        // Yuva formülü: dav1d sb64 işaretçi kaydırması.
        assert_eq!(cdef_idx_slot(0, 0), 0);
        assert_eq!(cdef_idx_slot(16, 0), 1, "yan yana 64×64");
        assert_eq!(cdef_idx_slot(32, 0), 0);
        assert_eq!(cdef_idx_slot(48, 0), 1);
        assert_eq!(cdef_idx_slot(0, 16), 2, "alt alta 64×64");
        assert_eq!(cdef_idx_slot(16, 16), 3);
        assert_eq!(cdef_idx_slot(32, 16), 2);
        assert_eq!(cdef_idx_slot(48, 16), 3);

        // 4×2 adet 64×64 (256×128 px): dört yuva da sınanır.
        let mut s = CdefState::new(4 * 16, 2 * 16, &crate::frame::butce::Butce::sinirsiz())
            .expect("cdef durumu");
        assert_eq!(s.sb64w, 4);
        let mut cdef_idx = [-1i32; 4];
        let mut sirali = 0i32;
        for by in [0usize, 16] {
            for bx in [0usize, 16, 32, 48] {
                sirali += 1;
                // reset_cdef (sb64 dalı): yalnız süper bloğun yuvası.
                let slot = cdef_idx_slot(bx, by);
                cdef_idx[slot] = -1;
                // decode_b: yuva boşsa `n_bits > 0` bitleri okunur (burada
                // okunan değer sıra numarasıdır) ve yuvaya yazılır.
                if cdef_idx[slot] == -1 {
                    cdef_idx[slot] = sirali;
                }
                // store_sb: kare ızgarasına **aynı** yuvadan yazılır.
                s.store_sb(bx, by, false, &cdef_idx);
                assert_eq!(
                    s.idx64[(by >> 4) * s.sb64w + (bx >> 4)],
                    sirali,
                    "sb64 ({bx},{by}) yuvası {slot}"
                );
            }
        }
    }

    /// 4:2:2 kroma yön tablosu `uv_dirs[1]` (`cdef_apply_tmpl.c:120-121`).
    #[test]
    fn uv_dirs_422_tablosu() {
        assert_eq!(UV_DIRS[0], [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(UV_DIRS[1], [7, 0, 2, 4, 5, 6, 6, 6]);
    }
}
