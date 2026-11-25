//! Superres (yatay büyütme) — S2c.3 / P21: kare üzerine uygulama.
//!
//! dav1d superres karesini kodlanan genişlikte (`width[0]`) çözer; deblocking ve
//! CDEF bu genişlikte çalışır, sonra her SB satırı yatayda `width[1]`'e büyütülür
//! (`dav1d_filter_sbrow_resize`), LR büyütülmüş karede çalışır ve referans
//! yuvasına büyütülmüş kare gider. Bu port deblocking/CDEF'i kare sonunda
//! uyguladığı için büyütme de kare sonunda yapılır: büyütme satır başına
//! bağımsızdır ve SB satırı bölümlemesinin birleşimi `[0, h)`'yi tam bir kez
//! örter, sonuç birebir aynıdır. LR'nin şerit sınırı satırları da büyütülmüş
//! deblocking sonrası (CDEF öncesi) satırlardır (`backup_lpf` içinde
//! `mc.resize`); bu yüzden CDEF öncesi kopya da aynı adım/başlangıçla büyütülür.
//!
//! ## dav1d eşlemesi (dav1d @ `99f3354`)
//!
//! | Rust | dav1d |
//! |---|---|
//! | [`scale_fac`] | `decode.c:3537-3538` (`scale_fac` makrosu) |
//! | [`upscale_x0`] | `decode.c:3328-3332` (`get_upscale_x0`) |
//! | [`Superres::new`] (luma/kroma ayrı adım ve başlangıç) | `decode.c:3536-3547` |
//! | [`upscale_planes`] | `recon_tmpl.c:2053-2086` (`dav1d_filter_sbrow_resize`) |
//! | kaynak kırpma genişliği `(4·bw + ss_hor) >> ss_hor` | `recon_tmpl.c:2078` |
//! | LR şerit sınırı satırlarının büyütülmesi | `lf_apply_tmpl.c:64-77,108-131` |
//! | iz sırası (SB satırı: `copy_lpf` → CDEF → büyütme) | `recon_tmpl.c:2004-2023,2100-2108` |
#![forbid(unsafe_code)]

use crate::decode::intra::PicBuf;
use crate::dsp::resize::{ResizeBitDepth, ResizeFn};
use crate::frame::{BitDepth, ChromaFormat};

/// dav1d `scale_fac(ref_sz, this_sz)` (`decode.c:3537-3538`).
#[inline]
#[must_use]
pub(crate) const fn scale_fac(ref_sz: i32, this_sz: i32) -> i32 {
    ((ref_sz << 14) + (this_sz >> 1)) / this_sz
}

/// dav1d `get_upscale_x0` (`decode.c:3328-3332`).
///
/// Pay negatif olabilir; C tamsayı bölmesi sıfıra doğru keser — Rust `/` da
/// öyle (`>>` kullanmak sonucu değiştirir).
#[inline]
#[must_use]
pub(crate) const fn upscale_x0(in_w: i32, out_w: i32, step: i32) -> i32 {
    let err = out_w * step - (in_w << 14);
    let x0 = (-((out_w - in_w) << 13) + (out_w >> 1)) / out_w + 128 - (err / 2);
    x0 & 0x3fff
}

/// Kare düzeyi superres parametreleri (`f->resize_step[2]`, `f->resize_start[2]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Superres {
    /// `[luma, kroma]` adım (14 bit kesir).
    pub(crate) step: [i32; 2],
    /// `[luma, kroma]` başlangıç konumu (`& 0x3fff`).
    pub(crate) start: [i32; 2],
    /// Kodlanan genişlik (`width[0]`).
    pub(crate) coded_w: usize,
    /// Büyütülmüş genişlik (`width[1]`).
    pub(crate) sr_w: usize,
}

impl Superres {
    /// `decode.c:3536-3547`: kroma adım ve başlangıcı **kroma genişlikleriyle**
    /// (`in_cw`, `out_cw`) ayrı hesaplanır; `ss_hor = layout != I444`.
    #[must_use]
    pub(crate) fn new(coded_w: usize, sr_w: usize, chroma: ChromaFormat) -> Self {
        let (in_w, out_w) = (coded_w as i32, sr_w as i32);
        let ss_hor = i32::from(chroma != ChromaFormat::C444);
        let in_cw = (in_w + ss_hor) >> ss_hor;
        let out_cw = (out_w + ss_hor) >> ss_hor;
        let step = [scale_fac(in_w, out_w), scale_fac(in_cw, out_cw)];
        let start = [
            upscale_x0(in_w, out_w, step[0]),
            upscale_x0(in_cw, out_cw, step[1]),
        ];
        Self {
            step,
            start,
            coded_w,
            sr_w,
        }
    }
}

fn resize_bd(bd: BitDepth) -> ResizeBitDepth {
    match bd {
        BitDepth::Eight => ResizeBitDepth::B8,
        BitDepth::Ten => ResizeBitDepth::B10,
        BitDepth::Twelve => ResizeBitDepth::B12,
    }
}

/// Düzlem `p` için (yatay, dikey) alt örnekleme (`recon_tmpl.c:2070,2076`).
fn plane_ss(p: usize, chroma: ChromaFormat) -> (usize, usize) {
    if p == 0 {
        (0, 0)
    } else {
        (
            usize::from(chroma != ChromaFormat::C444),
            usize::from(chroma == ChromaFormat::C420),
        )
    }
}

/// Kodlanan düzlemleri `sr.sr_w` genişliğine büyütür (`dav1d_filter_sbrow_resize`).
///
/// `src`/`src_stride` kodlanan [`PicBuf`] düzenindedir; `bw` blok ızgarası
/// genişliği (4×4 biriminde, 8 piksele yuvarlanmış). Kaynak kırpma genişliği
/// dav1d gibi `(4·bw + ss_hor) >> ss_hor`'dur (görünür genişlik değil).
#[allow(clippy::too_many_arguments)]
pub(crate) fn upscale_planes(
    src: &mut [Vec<u8>; 3],
    src_stride: [usize; 3],
    height: usize,
    bw: usize,
    sr: &Superres,
    chroma: ChromaFormat,
    bd: BitDepth,
    kernel: ResizeFn,
    butce: &crate::frame::butce::Butce,
    pool: &crate::thread::WorkerPool,
) -> crate::error::Result<PicBuf> {
    let mut dst = PicBuf::new(sr.sr_w, height, chroma, bd, butce)?;
    let n_planes = if chroma.has_chroma() { 3 } else { 1 };
    let rbd = resize_bd(bd);
    if !pool.is_parallel() {
        for p in 0..n_planes {
            let (ss_hor, ss_ver) = plane_ss(p, chroma);
            let dst_w = (sr.sr_w + ss_hor) >> ss_hor;
            let src_w = (4 * bw + ss_hor) >> ss_hor;
            let rows = (height + ss_ver) >> ss_ver;
            let ds = dst.stride[p];
            crate::dsp::profil_olc!(
                Resize,
                kernel(
                    &mut dst.data[p],
                    ds,
                    &src[p],
                    src_stride[p],
                    dst_w,
                    rows,
                    src_w,
                    sr.step[usize::from(p != 0)],
                    sr.start[usize::from(p != 0)],
                    rbd,
                )
            );
        }
        return Ok(dst);
    }

    use crate::frame::butce::dolu_vec;
    use std::sync::Arc;

    struct ResizeJob {
        y0: usize,
        rows: usize,
        buf: Vec<u8>,
    }

    let stripe_rows = 32usize;
    for p in 0..n_planes {
        let (ss_hor, ss_ver) = plane_ss(p, chroma);
        let dst_w = (sr.sr_w + ss_hor) >> ss_hor;
        let src_w = (4 * bw + ss_hor) >> ss_hor;
        let total_rows = (height + ss_ver) >> ss_ver;
        let ds = dst.stride[p];
        let ss = src_stride[p];
        let step = sr.step[usize::from(p != 0)];
        let start = sr.start[usize::from(p != 0)];
        let n_stripes = total_rows.div_ceil(stripe_rows);
        if n_stripes <= 1 {
            crate::dsp::profil_olc!(
                Resize,
                kernel(
                    &mut dst.data[p],
                    ds,
                    &src[p],
                    ss,
                    dst_w,
                    total_rows,
                    src_w,
                    step,
                    start,
                    rbd,
                )
            );
            continue;
        }
        let mut jobs = Vec::with_capacity(n_stripes);
        for s in 0..n_stripes {
            let y0 = s * stripe_rows;
            let r = stripe_rows.min(total_rows - y0);
            jobs.push(ResizeJob {
                y0,
                rows: r,
                buf: dolu_vec(r * ds, 0u8)?,
            });
        }
        let src_arc = Arc::new(core::mem::take(&mut src[p]));
        let src_w_arc = Arc::clone(&src_arc);
        let res = pool.execute_tasks(&mut jobs, move |_wid, _idx, job: &mut ResizeJob| {
            let src_slice = &src_w_arc[job.y0 * ss..];
            crate::dsp::profil_olc!(
                Resize,
                kernel(
                    &mut job.buf,
                    ds,
                    src_slice,
                    ss,
                    dst_w,
                    job.rows,
                    src_w,
                    step,
                    start,
                    rbd,
                )
            );
            Ok(())
        });
        src[p] = Arc::try_unwrap(src_arc).unwrap_or_else(|a| (*a).clone());
        res?;
        for job in jobs {
            let dst_off = job.y0 * ds;
            dst.data[p][dst_off..dst_off + job.rows * ds].copy_from_slice(&job.buf);
        }
    }
    Ok(dst)
}

/// FNV-1a 64 — `resize_trace.c` ile aynı.
fn fnv_rows(data: &[u8], stride: usize, y0: usize, rows: usize, w_bytes: usize) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for y in y0..y0 + rows {
        for &v in &data[y * stride..y * stride + w_bytes] {
            hash ^= u64::from(v);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

/// Büyütme çağrı izi (tanılama; varsayılan kapalı, `AV1OXIDE_SR_TRACE`).
///
/// dav1d'nin `mc.resize` çağrılarını **aynı sırayla** yeniden üretir: her SB
/// satırında önce `copy_lpf` (LR açık düzlemlerin şerit sınırı satırları, `L`),
/// sonra `dav1d_filter_sbrow_resize` (`S`). Satır biçimi
/// `scripts/oracle/resize_trace.c` ile aynıdır:
/// `Z <kare> <S|L> <düzlem> <kaynak_y> <h> <dst_w> <src_w> <dx> <mx0> <fnv>`.
/// Özet büyütülmüş tampondaki aynı satırlardan alınır (`L` için büyütülmüş
/// CDEF öncesi kopya; `n_lines == 3` iken yinelenen satır özete girmez).
#[allow(clippy::too_many_arguments)]
pub(crate) fn trace(
    frame_no: u32,
    sr_pic: &PicBuf,
    sr_pre: Option<&PicBuf>,
    restore_planes: u8,
    height: usize,
    bw: usize,
    sb128: bool,
    sr: &Superres,
    chroma: ChromaFormat,
    bpp: usize,
) {
    if std::env::var_os("AV1OXIDE_SR_TRACE").is_none() {
        return;
    }
    let fsel: Option<u32> = std::env::var("AV1OXIDE_SR_TRACE_FRAME")
        .ok()
        .and_then(|v| v.parse().ok());
    if fsel.is_some_and(|f| f != frame_no) {
        return;
    }
    let sb128 = usize::from(sb128);
    let sbsz = 16usize << sb128; // 4×4 biriminde
    let bh4 = ((height + 7) >> 3) << 1;
    let sbh = bh4.div_ceil(sbsz);
    let n_planes = if chroma.has_chroma() { 3 } else { 1 };
    for sby in 0..sbh {
        // --- `dav1d_copy_lpf` (`lf_apply_tmpl.c:104-`), yalnız LR düzlemleri.
        if let Some(pre) = sr_pre {
            let offset = if sby != 0 { 8 } else { 0 };
            for p in 0..n_planes {
                if (restore_planes >> p) & 1 == 0 {
                    continue;
                }
                let (ss_hor, ss_ver) = plane_ss(p, chroma);
                let h = (height + ss_ver) >> ss_ver;
                let row_h = ((sby + 1) << ((6 - ss_ver) + sb128)).min(h - 1);
                let mut row = (sby << ((6 - ss_ver) + sb128)) - (offset >> ss_ver);
                let mut stripe_h = (64 - if row == 0 { 8 } else { 0 }) >> ss_ver;
                let dst_w = (sr.sr_w + ss_hor) >> ss_hor;
                let src_w = (4 * bw + ss_hor) >> ss_hor;
                while row + stripe_h <= row_h {
                    let n_lines = 4 - usize::from(row + stripe_h + 1 == h);
                    let y = row + stripe_h - 2;
                    let hash = fnv_rows(&pre.data[p], pre.stride[p], y, n_lines, dst_w * bpp);
                    eprintln!(
                        "Z {frame_no} L {p} {y} {n_lines} {dst_w} {src_w} {} {} {hash}",
                        sr.step[ss_hor], sr.start[ss_hor]
                    );
                    row += stripe_h;
                    stripe_h = 64 >> ss_ver;
                }
            }
        }
        // --- `dav1d_filter_sbrow_resize` (`recon_tmpl.c:2053-2086`).
        let y = sby * sbsz * 4;
        for p in 0..n_planes {
            let (ss_hor, ss_ver) = plane_ss(p, chroma);
            let h_start = (if sby != 0 { 8 } else { 0 }) >> ss_ver;
            let h_end = (4 * (sbsz - 2 * usize::from(sby + 1 < sbh))) >> ss_ver;
            let img_h = (height - sbsz * 4 * sby + ss_ver) >> ss_ver;
            let rows = img_h.min(h_end) + h_start;
            let y0 = (y >> ss_ver) - h_start;
            let dst_w = (sr.sr_w + ss_hor) >> ss_hor;
            let src_w = (4 * bw + ss_hor) >> ss_hor;
            let hash = fnv_rows(&sr_pic.data[p], sr_pic.stride[p], y0, rows, dst_w * bpp);
            eprintln!(
                "Z {frame_no} S {p} {y0} {rows} {dst_w} {src_w} {} {} {hash}",
                sr.step[usize::from(p != 0)],
                sr.start[usize::from(p != 0)]
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `get_upscale_x0`: payın negatif olduğu durumda sıfıra doğru kesme.
    #[test]
    fn upscale_x0_sifira_dogru_kesme() {
        // in 9, out 16: pay = −(7 << 13) + 8 = −57336 → / 16 = −3583 (C ile aynı;
        // `>> 4` −3584 verirdi).
        let step = scale_fac(9, 16);
        assert_eq!(step, 9216);
        let err = 16 * step - (9 << 14);
        assert_eq!(err, 0);
        assert_eq!(upscale_x0(9, 16, step), (-3583 + 128) & 0x3fff);
        assert_ne!(upscale_x0(9, 16, step), ((-57336 >> 4) + 128) & 0x3fff);
    }

    /// Kroma adım/başlangıcı kroma genişlikleriyle ayrı hesaplanır.
    #[test]
    fn kroma_adim_ayri() {
        let s = Superres::new(1001, 1777, ChromaFormat::C420);
        assert_eq!(s.step[0], scale_fac(1001, 1777));
        assert_eq!(s.step[1], scale_fac(501, 889));
        assert_eq!(s.start[1], upscale_x0(501, 889, s.step[1]));
        let t = Superres::new(1001, 1777, ChromaFormat::C444);
        assert_eq!(t.step[0], t.step[1]);
        assert_eq!(t.start[0], t.start[1]);
    }
}
