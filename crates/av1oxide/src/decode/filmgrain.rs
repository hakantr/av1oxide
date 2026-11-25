//! Film grain'in çıktı karesine uygulanması — S2d.1 / P22.
//!
//! dav1d karşılıkları (@ `99f3354`, salt-okunur):
//!
//! | Kural | Kaynak |
//! |---|---|
//! | `has_grain`: `num_y_points ∥ num_uv_points[0] ∥ num_uv_points[1] ∥ (clip_to_restricted_range ∧ chroma_scaling_from_luma)` | `src/lib.c:303-309` |
//! | Grain **yalnız çıktı kopyasına**; referans grain'siz kalır | `src/lib.c:311-330`, `485-` |
//! | LUT ve ölçekleme üretimi, değişmeyen düzlemlerin kopyası | `src/fg_apply_tmpl.c:100-159` (`dav1d_prep_grain`) |
//! | 32 satırlık şerit başına `fgy`/`fguv`, luma dolgusu `ptr[w] = ptr[w − 1]` | `src/fg_apply_tmpl.c:161-223` (`dav1d_apply_grain_row`) |
//! | Şerit sayısı `(h + 31) / 32` | `src/fg_apply_tmpl.c:225-240` (`dav1d_apply_grain`) |
//! | `is_id` = `seq_hdr->mtrx == DAV1D_MC_IDENTITY` | `src/fg_apply_tmpl.c:173` |
//!
//! Çıktı kopyası girdinin tam kopyasıdır: grain uygulanan düzlemlerin görünür
//! alanı çekirdeklerce baştan sona yeniden yazılır, uygulanmayanlar dav1d'nin
//! `memcpy`'si gibi aynen kalır. dav1d luma dolgusunu girdi (referans)
//! resmine yazar; o sütun görünür genişliğin ötesindedir ve sonraki MC
//! okumaları kenar öykünmesiyle (`emu_edge`) görünür alana kırpıldığı için
//! referansı etkilemez. Burada dolgu ayrı bir luma kopyasına yazılır;
//! referans tamponuna dokunulmaz.

#![forbid(unsafe_code)]

use crate::bitstream::frame_header::FilmGrainData;
use crate::dsp::fg::{
    EMPTY_LUT, FG_BLOCK_SIZE, FgBitDepth, FgDsp, FgLayout, GRAIN_HEIGHT, GRAIN_WIDTH, GrainLut,
    SCALING_SIZE_MAX, SUB_GRAIN_HEIGHT, SUB_GRAIN_WIDTH,
};
use crate::frame::{BitDepth, ChromaFormat, Frame, PlaneType};

/// `generate_scaling` (`src/fg_apply_tmpl.c:41-97`).
///
/// `points[i] = [x, scaling]` artan `x` sırasıyla; ilk `points.len()` nokta
/// kullanılır. 8 bitte 256, 16bpc'de `1 << bd` giriş doldurulur: ilk noktaya
/// kadar ilk değer, noktalar arası `0x10000 / dx` adımlı doğrusal ara değer,
/// son noktadan sonra son değer; >8 bitte noktalar `<< (bd − 8)` konumuna
/// yazılır ve aradaki `2^(bd−8) − 1` giriş yuvarlanmış ara değerle dolar.
pub fn generate_scaling(bd: FgBitDepth, points: &[[u8; 2]], scaling: &mut [u8]) {
    let shift_x = bd.min_8();
    let scaling_size = bd.scaling_size();
    let ok = scaling.len() >= scaling_size && points.windows(2).all(|p| p[0][0] < p[1][0]);
    debug_assert!(ok, "generate_scaling sözleşme dışı girdi");
    if !ok {
        return;
    }
    let num = points.len();
    if num == 0 {
        scaling[..scaling_size].fill(0);
        return;
    }
    // İlk noktaya kadar ilk değer.
    scaling[..usize::from(points[0][0]) << shift_x].fill(points[0][1]);
    // Noktalar arası doğrusal ara değer.
    for i in 0..num - 1 {
        let bx = i32::from(points[i][0]);
        let by = i32::from(points[i][1]);
        let ex = i32::from(points[i + 1][0]);
        let ey = i32::from(points[i + 1][1]);
        let dx = ex - bx;
        let dy = ey - by;
        let delta = dy * ((0x10000 + (dx >> 1)) / dx);
        let mut d = 0x8000i32;
        for x in 0..dx {
            scaling[((bx + x) as usize) << shift_x] = (by + (d >> 16)) as u8;
            d += delta;
        }
    }
    // Son noktadan sonra son değer.
    let n = usize::from(points[num - 1][0]) << shift_x;
    scaling[n..scaling_size].fill(points[num - 1][1]);

    if shift_x != 0 {
        let pad = 1usize << shift_x;
        let rnd = (pad >> 1) as i32;
        for i in 0..num - 1 {
            let bx = usize::from(points[i][0]) << shift_x;
            let ex = usize::from(points[i + 1][0]) << shift_x;
            let dx = ex - bx;
            let mut x = 0usize;
            while x < dx {
                let base = i32::from(scaling[bx + x]);
                let range = i32::from(scaling[bx + x + pad]) - base;
                let mut r = rnd;
                for n in 1..pad {
                    r += range;
                    scaling[bx + x + n] = (base + (r >> shift_x)) as u8;
                }
                x += pad;
            }
        }
    }
}

/// dav1d `has_grain` (`src/lib.c:303-309`).
#[must_use]
pub(crate) fn has_grain(d: &FilmGrainData) -> bool {
    d.num_y_points != 0
        || d.num_uv_points[0] != 0
        || d.num_uv_points[1] != 0
        || (d.clip_to_restricted_range && d.chroma_scaling_from_luma)
}

fn fg_bd(bd: BitDepth) -> FgBitDepth {
    match bd {
        BitDepth::Eight => FgBitDepth::B8,
        BitDepth::Ten => FgBitDepth::B10,
        BitDepth::Twelve => FgBitDepth::B12,
    }
}

fn fg_layout(c: ChromaFormat) -> Option<FgLayout> {
    match c {
        ChromaFormat::Monochrome => None,
        ChromaFormat::C420 => Some(FgLayout::I420),
        ChromaFormat::C422 => Some(FgLayout::I422),
        ChromaFormat::C444 => Some(FgLayout::I444),
    }
}

/// `num` noktanın dilimi (sözdizimi sınırı: luma 14, kroma 10).
fn points<const N: usize>(p: &[[u8; 2]; N], num: i32) -> &[[u8; 2]] {
    &p[..usize::try_from(num).unwrap_or(0).min(N)]
}

/// FNV-1a 64 (iz özetleri; `scripts/oracle/fg_trace.c` ile aynı).
fn fnv(bytes: impl Iterator<Item = u8>, mut hash: u64) -> u64 {
    for b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

const FNV_INIT: u64 = 0xcbf2_9ce4_8422_2325;

fn lut_hash(lut: &GrainLut, w: usize, h: usize) -> u64 {
    fnv(
        lut.iter()
            .take(h)
            .flat_map(|row| row.iter().take(w).flat_map(|v| v.to_le_bytes())),
        FNV_INIT,
    )
}

fn rows_hash(data: &[u8], stride: usize, rows: usize, row_bytes: usize) -> u64 {
    let mut h = FNV_INIT;
    for y in 0..rows {
        h = fnv(data[y * stride..y * stride + row_bytes].iter().copied(), h);
    }
    h
}

/// Grain uygulanmış çıktı kopyası (`dav1d_apply_grain`).
///
/// `trace` verilirse (çıktı resmi sırası) dav1d iz aracıyla aynı biçimde
/// çağrı satırları standart hataya yazılır (bkz. [`trace_picture`]).
///
/// P38: çıktı kopyası ve luma dolgu tamponu `butce` defterinden ayrılır
/// (düşebilir); hata durumunda çıktı yoktur.
pub(crate) fn apply_grain(
    input: &Frame,
    data: &FilmGrainData,
    is_id: bool,
    dsp: &FgDsp,
    trace: Option<u64>,
    butce: &crate::frame::butce::Butce,
    pool: &crate::thread::WorkerPool,
) -> crate::error::Result<Frame> {
    let lens = [
        input
            .plane(PlaneType::Y)
            .map_or(0, |p| p.allocation_bytes()),
        input
            .plane(PlaneType::U)
            .map_or(0, |p| p.allocation_bytes()),
        input
            .plane(PlaneType::V)
            .map_or(0, |p| p.allocation_bytes()),
    ];
    let bufs = pool.buf_pool().take_u8_triple(lens, butce.has_limit());
    let mut out = input.try_clone_with_bufs(bufs, butce)?;
    let bd = fg_bd(input.bit_depth());
    let layout = fg_layout(input.chroma_format());
    let bpp = bd.bytes();
    let w = input.width();
    let h = input.height();
    if w == 0 || h == 0 {
        return Ok(out);
    }
    if let Some(pic) = trace {
        eprintln!(
            "P {pic} {w} {h} {} {} {} {} {} {} {} {} {} {} {} {} {}",
            bd.bits(),
            layout.map_or(0, |l| l.index() + 1),
            data.seed,
            data.num_y_points,
            data.num_uv_points[0],
            data.num_uv_points[1],
            u8::from(data.chroma_scaling_from_luma),
            u8::from(data.overlap_flag),
            u8::from(data.clip_to_restricted_range),
            data.ar_coeff_lag,
            data.scaling_shift,
            data.ar_coeff_shift,
            data.grain_scale_shift,
        );
    }

    // --- `dav1d_prep_grain` (`src/fg_apply_tmpl.c:100-159`) ---------------
    let mut luts: Vec<GrainLut> = vec![EMPTY_LUT; 3];
    crate::dsp::profil_olc!(FgGrainY, (dsp.generate_grain_y)(&mut luts[0], data, bd));
    if let Some(pic) = trace {
        eprintln!(
            "G {pic} 0 {}",
            lut_hash(&luts[0], GRAIN_WIDTH, GRAIN_HEIGHT)
        );
    }
    if let Some(l) = layout {
        for pl in 0..2 {
            if data.num_uv_points[pl] != 0 || data.chroma_scaling_from_luma {
                let (y, uv) = luts.split_at_mut(1);
                crate::dsp::profil_olc!(
                    FgGrainUv,
                    (dsp.generate_grain_uv[l.index()])(&mut uv[pl], &y[0], data, pl, bd)
                );
                if let Some(pic) = trace {
                    let cw = if l.ss_x() != 0 {
                        SUB_GRAIN_WIDTH
                    } else {
                        GRAIN_WIDTH
                    };
                    let ch = if l.ss_y() != 0 {
                        SUB_GRAIN_HEIGHT
                    } else {
                        GRAIN_HEIGHT
                    };
                    eprintln!("G {pic} {} {}", pl + 1, lut_hash(&uv[pl], cw, ch));
                }
            }
        }
    }
    let mut scaling = vec![vec![0u8; SCALING_SIZE_MAX]; 3];
    if data.num_y_points != 0 || data.chroma_scaling_from_luma {
        generate_scaling(
            bd,
            points(&data.y_points, data.num_y_points),
            &mut scaling[0],
        );
    }
    for pl in 0..2 {
        if data.num_uv_points[pl] != 0 {
            generate_scaling(
                bd,
                points(&data.uv_points[pl], data.num_uv_points[pl]),
                &mut scaling[1 + pl],
            );
        }
    }

    // --- `dav1d_apply_grain_row` (`src/fg_apply_tmpl.c:161-223`) ----------
    let Some(y_in) = input.plane(PlaneType::Y) else {
        return Ok(out);
    };
    let y_stride = y_in.stride();
    let do_uv = layout.is_some()
        && (data.num_uv_points[0] != 0
            || data.num_uv_points[1] != 0
            || data.chroma_scaling_from_luma);
    let (ss_x, ss_y) = layout.map_or((0, 0), |l| (l.ss_x(), l.ss_y()));
    // Luma dolgusu (`src/fg_apply_tmpl.c:195-202`): tek genişlikte yatay alt
    // örneklemeli kroma, son sütun için luma `w`'yi okur.
    let padded = if do_uv && (w & ss_x) != 0 {
        // P38 (taşma sınıfı): AV1 sınırında `(w+1)·bpp·h` 8 590 065 664 bayt
        // (32 bitlik `usize`'ı aşar); denetimli.
        let lstride = (w + 1) * bpp;
        let n = lstride
            .checked_mul(h)
            .ok_or(crate::error::Error::SizeOverflow {
                stage: crate::error::SizeStage::PlaneArea,
                lhs: lstride,
                rhs: h,
            })?;
        let mut buf = butce.gecici(n, 0u8)?;
        for y in 0..h {
            let row = &y_in.data()[y * y_stride..y * y_stride + w * bpp];
            let dst = &mut buf[y * lstride..(y + 1) * lstride];
            dst[..w * bpp].copy_from_slice(row);
            let (head, tail) = dst.split_at_mut(w * bpp);
            tail.copy_from_slice(&head[(w - 1) * bpp..]);
        }
        Some((buf, lstride))
    } else {
        None
    };
    let (luma, luma_stride): (&[u8], usize) = match &padded {
        Some((buf, s)) => (buf.as_slice(), *s),
        None => (y_in.data(), y_stride),
    };
    let cpw = (w + ss_x) >> ss_x;
    let rows = h.div_ceil(FG_BLOCK_SIZE);
    if pool.is_parallel() && rows > 1 && trace.is_none() {
        apply_grain_parallel_rows(
            input,
            &mut out,
            data,
            is_id,
            dsp,
            bd,
            layout,
            w,
            h,
            cpw,
            ss_y,
            do_uv,
            y_stride,
            luma,
            luma_stride,
            &luts,
            &scaling,
            rows,
            pool,
        )?;
        out.metadata_mut().film_grain_applied = true;
        return Ok(out);
    }
    for row in 0..rows {
        let band = (h - row * FG_BLOCK_SIZE).min(FG_BLOCK_SIZE);
        if data.num_y_points != 0
            && let Some(y_out) = out.plane_mut(PlaneType::Y)
        {
            let off = row * FG_BLOCK_SIZE * y_stride;
            crate::dsp::profil_olc!(
                FgY,
                (dsp.fgy_32x32xn)(
                    &mut y_out.data_mut()[off..],
                    &y_in.data()[off..],
                    y_stride,
                    data,
                    w,
                    &scaling[0],
                    &luts[0],
                    band,
                    row,
                    bd,
                )
            );
            if let Some(pic) = trace {
                let hsh = rows_hash(&y_out.data()[off..], y_stride, band, w * bpp);
                eprintln!("Y {pic} {row} {band} {w} {hsh}");
            }
        }
        let Some(l) = layout.filter(|_| do_uv) else {
            continue;
        };
        let bh = (band + ss_y) >> ss_y;
        let loff = row * FG_BLOCK_SIZE * luma_stride;
        for pl in 0..2 {
            if !data.chroma_scaling_from_luma && data.num_uv_points[pl] == 0 {
                continue;
            }
            let ptype = if pl == 0 { PlaneType::U } else { PlaneType::V };
            let (Some(c_in), Some(c_out)) = (input.plane(ptype), out.plane_mut(ptype)) else {
                continue;
            };
            let c_stride = c_in.stride();
            let uv_off = (row * FG_BLOCK_SIZE * c_stride) >> ss_y;
            let si = if data.chroma_scaling_from_luma {
                0
            } else {
                1 + pl
            };
            crate::dsp::profil_olc!(
                FgUv,
                (dsp.fguv_32x32xn[l.index()])(
                    &mut c_out.data_mut()[uv_off..],
                    &c_in.data()[uv_off..],
                    c_stride,
                    data,
                    cpw,
                    &scaling[si],
                    &luts[1 + pl],
                    bh,
                    row,
                    &luma[loff..],
                    luma_stride,
                    pl,
                    is_id,
                    bd,
                )
            );
            if let Some(pic) = trace {
                let hsh = rows_hash(&c_out.data()[uv_off..], c_stride, bh, cpw * bpp);
                eprintln!(
                    "C {pic} {} {row} {bh} {cpw} {} {hsh}",
                    pl + 1,
                    u8::from(is_id)
                );
            }
        }
    }
    out.metadata_mut().film_grain_applied = true;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn apply_grain_parallel_rows(
    input: &Frame,
    out: &mut Frame,
    data: &FilmGrainData,
    is_id: bool,
    dsp: &FgDsp,
    bd: FgBitDepth,
    layout: Option<FgLayout>,
    w: usize,
    h: usize,
    cpw: usize,
    ss_y: usize,
    do_uv: bool,
    y_stride: usize,
    luma: &[u8],
    luma_stride: usize,
    luts: &[GrainLut],
    scaling: &[Vec<u8>],
    rows: usize,
    pool: &crate::thread::WorkerPool,
) -> crate::error::Result<()> {
    use crate::frame::butce::dolu_vec;
    use std::sync::Arc;

    struct FgRowJob {
        row: usize,
        band: usize,
        bh: usize,
        y_dst: Vec<u8>,
        uv_dst: [Vec<u8>; 2],
    }

    let do_y = data.num_y_points != 0;
    let do_u = do_uv && (data.chroma_scaling_from_luma || data.num_uv_points[0] != 0);
    let do_v = do_uv && (data.chroma_scaling_from_luma || data.num_uv_points[1] != 0);

    let u_stride = out.plane(PlaneType::U).map_or(0, |p| p.stride());
    let v_stride = out.plane(PlaneType::V).map_or(0, |p| p.stride());
    let luma_is_y = input
        .plane(PlaneType::Y)
        .is_some_and(|p| core::ptr::eq(luma, p.data()));
    let y_arc = Arc::new(
        out.plane_mut(PlaneType::Y)
            .map_or_else(Vec::new, |p| p.take_data()),
    );
    let u_arc = Arc::new(
        out.plane_mut(PlaneType::U)
            .map_or_else(Vec::new, |p| p.take_data()),
    );
    let v_arc = Arc::new(
        out.plane_mut(PlaneType::V)
            .map_or_else(Vec::new, |p| p.take_data()),
    );
    let luma_arc = if luma_is_y {
        Arc::clone(&y_arc)
    } else if do_u || do_v {
        Arc::new(dolu_vec(luma.len(), 0u8).map(|mut b| {
            b.copy_from_slice(luma);
            b
        })?)
    } else {
        Arc::new(Vec::new())
    };

    let mut jobs: Vec<FgRowJob> = Vec::with_capacity(rows);
    for row in 0..rows {
        let band = (h - row * FG_BLOCK_SIZE).min(FG_BLOCK_SIZE);
        let bh = (band + ss_y) >> ss_y;
        jobs.push(FgRowJob {
            row,
            band,
            bh,
            y_dst: Vec::new(),
            uv_dst: [Vec::new(), Vec::new()],
        });
    }

    let luts_arc = Arc::new(luts.to_vec());
    let scaling_arc = Arc::new(scaling.to_vec());
    let fg_data = *data;
    let fgy_fn = dsp.fgy_32x32xn;
    let fguv_fns = dsp.fguv_32x32xn;
    let y_w = Arc::clone(&y_arc);
    let u_w = Arc::clone(&u_arc);
    let v_w = Arc::clone(&v_arc);
    let luma_w = Arc::clone(&luma_arc);

    let res = pool.execute_tasks(&mut jobs, move |_wid, _idx, job: &mut FgRowJob| {
        if do_y {
            let off = job.row * FG_BLOCK_SIZE * y_stride;
            let len = job.band * y_stride;
            let mut y_dst = dolu_vec(len, 0u8)?;
            y_dst.copy_from_slice(&y_w[off..off + len]);
            job.y_dst = y_dst;
            crate::dsp::profil_olc!(
                FgY,
                (fgy_fn)(
                    &mut job.y_dst,
                    &y_w[off..off + len],
                    y_stride,
                    &fg_data,
                    w,
                    &scaling_arc[0],
                    &luts_arc[0],
                    job.band,
                    job.row,
                    bd,
                )
            );
        }
        if let Some(l) = layout.filter(|_| do_uv) {
            let fguv_fn = fguv_fns[l.index()];
            let loff = job.row * FG_BLOCK_SIZE * luma_stride;
            let luma_slice = &luma_w[loff..];
            for (pl, (do_pl, c_stride, c_src)) in [
                (do_u, u_stride, u_w.as_slice()),
                (do_v, v_stride, v_w.as_slice()),
            ]
            .into_iter()
            .enumerate()
            {
                if !do_pl || c_stride == 0 {
                    continue;
                }
                let off = (job.row * FG_BLOCK_SIZE * c_stride) >> ss_y;
                let len = job.bh * c_stride;
                let mut uv_dst = dolu_vec(len, 0u8)?;
                uv_dst.copy_from_slice(&c_src[off..off + len]);
                job.uv_dst[pl] = uv_dst;
                let si = if fg_data.chroma_scaling_from_luma {
                    0
                } else {
                    1 + pl
                };
                crate::dsp::profil_olc!(
                    FgUv,
                    (fguv_fn)(
                        &mut job.uv_dst[pl],
                        &c_src[off..off + len],
                        c_stride,
                        &fg_data,
                        cpw,
                        &scaling_arc[si],
                        &luts_arc[1 + pl],
                        job.bh,
                        job.row,
                        luma_slice,
                        luma_stride,
                        pl,
                        is_id,
                        bd,
                    )
                );
            }
        }
        Ok(())
    });

    drop(luma_arc);
    if let Some(p) = out.plane_mut(PlaneType::Y) {
        p.put_data(Arc::try_unwrap(y_arc).unwrap_or_else(|a| (*a).clone()));
    }
    if let Some(p) = out.plane_mut(PlaneType::U) {
        p.put_data(Arc::try_unwrap(u_arc).unwrap_or_else(|a| (*a).clone()));
    }
    if let Some(p) = out.plane_mut(PlaneType::V) {
        p.put_data(Arc::try_unwrap(v_arc).unwrap_or_else(|a| (*a).clone()));
    }
    res?;

    for job in jobs {
        if do_y && let Some(y_out) = out.plane_mut(PlaneType::Y) {
            let off = job.row * FG_BLOCK_SIZE * y_stride;
            let len = job.band * y_stride;
            y_out.data_mut()[off..off + len].copy_from_slice(&job.y_dst);
        }
        if do_u && let Some(u_out) = out.plane_mut(PlaneType::U) {
            let off = (job.row * FG_BLOCK_SIZE * u_stride) >> ss_y;
            let len = job.bh * u_stride;
            u_out.data_mut()[off..off + len].copy_from_slice(&job.uv_dst[0]);
        }
        if do_v && let Some(v_out) = out.plane_mut(PlaneType::V) {
            let off = (job.row * FG_BLOCK_SIZE * v_stride) >> ss_y;
            let len = job.bh * v_stride;
            v_out.data_mut()[off..off + len].copy_from_slice(&job.uv_dst[1]);
        }
    }
    Ok(())
}

/// İz seçimi: `AV1OXIDE_FG_TRACE` tanımlıysa (tanılama; varsayılan kapalı)
/// ve `AV1OXIDE_FG_TRACE_PIC` yoksa ya da `pic`'e eşitse `Some(pic)`.
///
/// Satır biçimi (`scripts/oracle/fg_trace.c` ile aynı; `pic` = çıktı resmi
/// sırası, gizli kareler dâhil):
///
/// ```text
/// P <pic> <w> <h> <bpc> <layout> <seed> <num_y> <num_uv0> <num_uv1> <csfl> <overlap> <clip> <lag> <scaling_shift> <ar_coeff_shift> <grain_scale_shift>
/// G <pic> <0|1|2> <fnv(LUT, i16 LE)>
/// Y <pic> <row> <bh> <pw> <fnv(çıktı satırları)>
/// C <pic> <1|2> <row> <bh> <pw> <is_id> <fnv(çıktı satırları)>
/// ```
#[must_use]
pub(crate) fn trace_picture(pic: u64) -> Option<u64> {
    use std::sync::OnceLock;
    static CFG: OnceLock<Option<Option<u64>>> = OnceLock::new();
    let cfg = *CFG.get_or_init(|| {
        std::env::var_os("AV1OXIDE_FG_TRACE")?;
        let sel: Option<u64> = std::env::var("AV1OXIDE_FG_TRACE_PIC")
            .ok()
            .and_then(|v| v.parse().ok());
        Some(sel)
    });
    let sel = cfg?;
    if sel.is_some_and(|s| s != pic) {
        return None;
    }
    Some(pic)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ölçekleme: tek nokta → bütün tablo o değer; nokta yok → sıfır.
    #[test]
    fn olcekleme_uc_durumlar() {
        for bd in FgBitDepth::ALL {
            let mut t = vec![0xAAu8; 4096];
            generate_scaling(bd, &[], &mut t);
            assert!(t[..bd.scaling_size()].iter().all(|&v| v == 0));
            generate_scaling(bd, &[[100, 77]], &mut t);
            assert!(t[..bd.scaling_size()].iter().all(|&v| v == 77));
        }
    }

    /// 8 bit ara değer: 0→0 ile 255→255 arası birim eğim.
    #[test]
    fn olcekleme_dogrusal() {
        let mut t = vec![0u8; 256];
        generate_scaling(FgBitDepth::B8, &[[0, 0], [255, 255]], &mut t);
        for (i, &v) in t.iter().enumerate() {
            assert_eq!(usize::from(v), i);
        }
    }

    fn data() -> FilmGrainData {
        let mut d = FilmGrainData {
            seed: 0xbeef,
            num_y_points: 2,
            num_uv_points: [2, 2],
            scaling_shift: 8,
            ar_coeff_lag: 2,
            ar_coeff_shift: 7,
            uv_mult: [10, -20],
            uv_luma_mult: [64, 30],
            uv_offset: [5, -7],
            overlap_flag: true,
            ..FilmGrainData::default()
        };
        d.y_points[0] = [0, 40];
        d.y_points[1] = [255, 90];
        d.uv_points[0][0] = [16, 20];
        d.uv_points[0][1] = [200, 120];
        d.uv_points[1][0] = [0, 60];
        d.uv_points[1][1] = [128, 10];
        d
    }

    /// Grain yalnız kopyaya uygulanır; girdi değişmez; tek genişlik ve
    /// kısmi blok (kenar) panik üretmez.
    #[test]
    fn kopya_uzerine_uygulama() {
        for (bd, chroma) in [
            (BitDepth::Eight, ChromaFormat::C420),
            (BitDepth::Ten, ChromaFormat::C422),
            (BitDepth::Twelve, ChromaFormat::C444),
            (BitDepth::Eight, ChromaFormat::Monochrome),
        ] {
            let mut f = Frame::new(37, 35, chroma, bd).expect("kare");
            for p in [PlaneType::Y, PlaneType::U, PlaneType::V] {
                if let Some(pl) = f.plane_mut(p) {
                    for (i, b) in pl.data_mut().iter_mut().enumerate() {
                        *b = (i * 7 % 3) as u8;
                    }
                }
            }
            let before = f.clone();
            let fg_bd = match bd {
                BitDepth::Eight => FgBitDepth::B8,
                BitDepth::Ten => FgBitDepth::B10,
                BitDepth::Twelve => FgBitDepth::B12,
            };
            let out = apply_grain(
                &f,
                &data(),
                false,
                &FgDsp::scalar_for(fg_bd),
                None,
                &crate::frame::butce::Butce::sinirsiz(),
                &crate::thread::WorkerPool::new(1),
            )
            .expect("grain");
            assert_eq!(f, before);
            assert_ne!(out.plane(PlaneType::Y), f.plane(PlaneType::Y));
            assert!(out.metadata().film_grain_applied);
            let out_mt = apply_grain(
                &f,
                &data(),
                false,
                &FgDsp::scalar_for(fg_bd),
                None,
                &crate::frame::butce::Butce::sinirsiz(),
                &crate::thread::WorkerPool::new(4),
            )
            .expect("grain mt");
            assert_eq!(out, out_mt);
        }
    }

    #[test]
    fn has_grain_kurali() {
        let mut d = FilmGrainData::default();
        assert!(!has_grain(&d));
        d.clip_to_restricted_range = true;
        assert!(!has_grain(&d));
        d.chroma_scaling_from_luma = true;
        assert!(has_grain(&d));
    }
}
