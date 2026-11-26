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

struct GrainPrep {
    luts: [GrainLut; 3],
    scaling: [[u8; SCALING_SIZE_MAX]; 3],
}

fn prep_grain(
    w: usize,
    h: usize,
    bd: FgBitDepth,
    layout: Option<FgLayout>,
    data: &FilmGrainData,
    dsp: &FgDsp,
    trace: Option<u64>,
) -> GrainPrep {
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
    let mut luts: [GrainLut; 3] = [EMPTY_LUT; 3];
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
    let mut scaling = [[0u8; SCALING_SIZE_MAX]; 3];
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
    GrainPrep { luts, scaling }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn fill_luma_strip(
    dst: &mut [u8],
    dst_stride: usize,
    src_y: &[u8],
    src_y_stride: usize,
    y0: usize,
    band: usize,
    w: usize,
    bpp: usize,
    pad_last_col: bool,
) {
    let row_bytes = w * bpp;
    for r in 0..band {
        let s = (y0 + r) * src_y_stride;
        let d = r * dst_stride;
        dst[d..d + row_bytes].copy_from_slice(&src_y[s..s + row_bytes]);
        if pad_last_col {
            let (head, tail) = dst[d..d + row_bytes + bpp].split_at_mut(row_bytes);
            tail.copy_from_slice(&head[row_bytes - bpp..]);
        }
    }
}

#[inline]
fn copy_rows_to_frame(
    dst_plane: &mut [u8],
    dst_off: usize,
    dst_stride: usize,
    src_strip: &[u8],
    rows_n: usize,
    row_bytes: usize,
) {
    if dst_stride == row_bytes {
        let n = rows_n * dst_stride;
        dst_plane[dst_off..dst_off + n].copy_from_slice(&src_strip[..n]);
    } else {
        for r in 0..rows_n {
            let d = dst_off + r * dst_stride;
            let s = r * dst_stride;
            dst_plane[d..d + row_bytes].copy_from_slice(&src_strip[s..s + row_bytes]);
        }
    }
}

/// Grain uygulanmış çıktı kopyası (`dav1d_apply_grain`, C-ABI ve birim testi yolu).
///
/// `trace` verilirse (çıktı resmi sırası) dav1d iz aracıyla aynı biçimde
/// çağrı satırları standart hataya yazılır (bkz. [`trace_picture`]).
///
/// P38 / `4.A.1`: çıktı düzlemleri ön-kopyasız hazırlanır; tek genişlik luma
/// dolgusu tam-kare yerine 32 satırlık şerit tamponuyla yapılır; paralel modda
/// dalga tamponları (`wave_size = n_threads.min(rows)`) yeniden kullanılır.
pub(crate) fn apply_grain(
    input: &Frame,
    data: &FilmGrainData,
    is_id: bool,
    dsp: &FgDsp,
    trace: Option<u64>,
    butce: &crate::frame::butce::Butce,
    pool: &crate::thread::WorkerPool,
) -> crate::error::Result<Frame> {
    let bd = fg_bd(input.bit_depth());
    let layout = fg_layout(input.chroma_format());
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

    let w = input.width();
    let h = input.height();
    if w == 0 || h == 0 {
        return Ok(out);
    }
    let Some(y_in) = input.plane(PlaneType::Y) else {
        return Ok(out);
    };
    let src_data = [
        y_in.data(),
        input.plane(PlaneType::U).map_or(&[][..], |p| p.data()),
        input.plane(PlaneType::V).map_or(&[][..], |p| p.data()),
    ];
    let src_stride = [
        y_in.stride(),
        input.plane(PlaneType::U).map_or(0, |p| p.stride()),
        input.plane(PlaneType::V).map_or(0, |p| p.stride()),
    ];
    apply_grain_into_out_slices(
        src_data, src_stride, &mut out, data, is_id, dsp, bd, layout, trace, butce, pool,
    )?;
    out.metadata_mut().film_grain_applied = true;
    Ok(out)
}

/// Referans olarak saklanmayan (`refresh_frame_flags == 0`) görünür karelerde
/// tam-kare ikincil [`Frame`] kopyası ayırmadan, 32 satırlık kayan şerit tamponları
/// üzerinden doğrudan yerinde (`in-place`) Film Grain uygular (`4.A.1`).
///
/// Kroma sentezinin (`fguv_32x32xn`) grain'siz `luma` satırlarına bağımlılığı,
/// her 32 satırlık şeritte `Y` düzlemi üzerine yazılmadan önce grain'siz `Y`
/// şeridinin saklanmasıyla (veya paralel dalgada `U`/`V` şeritlerinin `Y`'den
/// önce işlenmesiyle) `%100` bit-birebir korunur.
pub(crate) fn apply_grain_in_place(
    frame: &mut Frame,
    data: &FilmGrainData,
    is_id: bool,
    dsp: &FgDsp,
    trace: Option<u64>,
    butce: &crate::frame::butce::Butce,
    pool: &crate::thread::WorkerPool,
) -> crate::error::Result<()> {
    let w = frame.width();
    let h = frame.height();
    if w == 0 || h == 0 || frame.plane(PlaneType::Y).is_none() {
        return Ok(());
    }
    let bd = fg_bd(frame.bit_depth());
    let layout = fg_layout(frame.chroma_format());
    let bpp = bd.bytes();
    let (ss_x, ss_y) = layout.map_or((0, 0), |l| (l.ss_x(), l.ss_y()));
    let do_y = data.num_y_points != 0;
    let do_uv = layout.is_some()
        && (data.num_uv_points[0] != 0
            || data.num_uv_points[1] != 0
            || data.chroma_scaling_from_luma);
    let do_u = do_uv && (data.chroma_scaling_from_luma || data.num_uv_points[0] != 0);
    let do_v = do_uv && (data.chroma_scaling_from_luma || data.num_uv_points[1] != 0);

    let prep = prep_grain(w, h, bd, layout, data, dsp, trace);
    if !do_y && !do_u && !do_v {
        frame.metadata_mut().film_grain_applied = true;
        return Ok(());
    }

    let y_stride = frame.plane(PlaneType::Y).map_or(0, |p| p.stride());
    let u_stride = frame.plane(PlaneType::U).map_or(0, |p| p.stride());
    let v_stride = frame.plane(PlaneType::V).map_or(0, |p| p.stride());
    let pad_luma = (do_u || do_v) && (w & ss_x) != 0;
    let need_luma_save = (do_u || do_v) && (do_y || pad_luma);

    let band_max = FG_BLOCK_SIZE.min(h);
    let bh_max = (band_max + ss_y) >> ss_y;
    let mut max_src_bytes = 0usize;
    if do_y {
        max_src_bytes = max_src_bytes.max(band_max * y_stride);
    }
    if do_u {
        max_src_bytes = max_src_bytes.max(bh_max * u_stride);
    }
    if do_v {
        max_src_bytes = max_src_bytes.max(bh_max * v_stride);
    }
    let luma_save_stride = if pad_luma {
        w.checked_add(1).and_then(|wp| wp.checked_mul(bpp)).ok_or(
            crate::error::Error::SizeOverflow {
                stage: crate::error::SizeStage::RowBytes,
                lhs: w,
                rhs: bpp,
            },
        )?
    } else {
        y_stride
    };
    let mut src_strip = butce.gecici(max_src_bytes, 0u8)?;
    let mut luma_strip = if need_luma_save {
        let n =
            luma_save_stride
                .checked_mul(band_max)
                .ok_or(crate::error::Error::SizeOverflow {
                    stage: crate::error::SizeStage::PlaneArea,
                    lhs: luma_save_stride,
                    rhs: band_max,
                })?;
        Some(butce.gecici(n, 0u8)?)
    } else {
        None
    };

    let cpw = (w + ss_x) >> ss_x;
    let rows = h.div_ceil(FG_BLOCK_SIZE);
    if pool.is_parallel() && rows > 1 && trace.is_none() {
        apply_grain_in_place_waves(
            frame,
            data,
            is_id,
            dsp,
            bd,
            layout,
            w,
            h,
            cpw,
            ss_y,
            bpp,
            do_y,
            do_u,
            do_v,
            pad_luma,
            y_stride,
            u_stride,
            v_stride,
            luma_save_stride,
            &prep,
            rows,
            pool,
        )?;
        frame.metadata_mut().film_grain_applied = true;
        return Ok(());
    }

    for row in 0..rows {
        let y0 = row * FG_BLOCK_SIZE;
        let band = (h - y0).min(FG_BLOCK_SIZE);
        if let Some(lbuf) = luma_strip.as_deref_mut()
            && let Some(y_pl) = frame.plane(PlaneType::Y)
        {
            fill_luma_strip(
                lbuf,
                luma_save_stride,
                y_pl.data(),
                y_stride,
                y0,
                band,
                w,
                bpp,
                pad_luma,
            );
        }
        if do_y && let Some(y_pl) = frame.plane_mut(PlaneType::Y) {
            let off = y0 * y_stride;
            let len = band * y_stride;
            src_strip[..len].copy_from_slice(&y_pl.data()[off..off + len]);
            crate::dsp::profil_olc!(
                FgY,
                (dsp.fgy_32x32xn)(
                    &mut y_pl.data_mut()[off..off + len],
                    &src_strip[..len],
                    y_stride,
                    data,
                    w,
                    &prep.scaling[0],
                    &prep.luts[0],
                    band,
                    row,
                    bd,
                )
            );
            if let Some(pic) = trace {
                let hsh = rows_hash(&y_pl.data()[off..], y_stride, band, w * bpp);
                eprintln!("Y {pic} {row} {band} {w} {hsh}");
            }
        }
        let Some(l) = layout.filter(|_| do_u || do_v) else {
            continue;
        };
        let bh = (band + ss_y) >> ss_y;
        let planes = frame.planes_mut();
        let Some((y_pl, uv_pls)) = planes.split_first_mut() else {
            continue;
        };
        let (luma_ref, lstride_ref): (&[u8], usize) = match luma_strip.as_deref() {
            Some(lbuf) => (lbuf, luma_save_stride),
            None => (&y_pl.data()[y0 * y_stride..], y_stride),
        };
        for pl in 0..2 {
            let active = if pl == 0 { do_u } else { do_v };
            if !active {
                continue;
            }
            let Some(c_pl) = uv_pls.get_mut(pl) else {
                continue;
            };
            let c_stride = c_pl.stride();
            let uv_off = (y0 * c_stride) >> ss_y;
            let len = bh * c_stride;
            src_strip[..len].copy_from_slice(&c_pl.data()[uv_off..uv_off + len]);
            let si = if data.chroma_scaling_from_luma {
                0
            } else {
                1 + pl
            };
            crate::dsp::profil_olc!(
                FgUv,
                (dsp.fguv_32x32xn[l.index()])(
                    &mut c_pl.data_mut()[uv_off..uv_off + len],
                    &src_strip[..len],
                    c_stride,
                    data,
                    cpw,
                    &prep.scaling[si],
                    &prep.luts[1 + pl],
                    bh,
                    row,
                    luma_ref,
                    lstride_ref,
                    pl,
                    is_id,
                    bd,
                )
            );
            if let Some(pic) = trace {
                let hsh = rows_hash(&c_pl.data()[uv_off..], c_stride, bh, cpw * bpp);
                eprintln!(
                    "C {pic} {} {row} {bh} {cpw} {} {hsh}",
                    pl + 1,
                    u8::from(is_id)
                );
            }
        }
    }
    frame.metadata_mut().film_grain_applied = true;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_grain_in_place_waves(
    frame: &mut Frame,
    data: &FilmGrainData,
    is_id: bool,
    dsp: &FgDsp,
    bd: FgBitDepth,
    layout: Option<FgLayout>,
    w: usize,
    h: usize,
    cpw: usize,
    ss_y: usize,
    bpp: usize,
    do_y: bool,
    do_u: bool,
    do_v: bool,
    pad_luma: bool,
    y_stride: usize,
    u_stride: usize,
    v_stride: usize,
    luma_pad_stride: usize,
    prep: &GrainPrep,
    rows: usize,
    pool: &crate::thread::WorkerPool,
) -> crate::error::Result<()> {
    use crate::frame::butce::dolu_vec;
    use std::sync::Arc;

    struct InPlaceWaveJob {
        row: usize,
        band: usize,
        bh: usize,
        y_buf: Vec<u8>,
        u_buf: Vec<u8>,
        v_buf: Vec<u8>,
        luma_pad: Vec<u8>,
    }

    let need_y_in_job = do_y || do_u || do_v;
    let bh_alloc = (FG_BLOCK_SIZE + ss_y) >> ss_y;
    let wave_size = pool.n_threads().min(rows).max(1);
    let mut jobs: Vec<InPlaceWaveJob> = (0..wave_size)
        .map(|_| {
            Ok(InPlaceWaveJob {
                row: 0,
                band: 0,
                bh: 0,
                y_buf: if need_y_in_job {
                    dolu_vec(FG_BLOCK_SIZE * y_stride, 0u8)?
                } else {
                    Vec::new()
                },
                u_buf: if do_u {
                    dolu_vec(bh_alloc * u_stride, 0u8)?
                } else {
                    Vec::new()
                },
                v_buf: if do_v {
                    dolu_vec(bh_alloc * v_stride, 0u8)?
                } else {
                    Vec::new()
                },
                luma_pad: if pad_luma {
                    dolu_vec(FG_BLOCK_SIZE * luma_pad_stride, 0u8)?
                } else {
                    Vec::new()
                },
            })
        })
        .collect::<crate::error::Result<Vec<_>>>()?;

    let mut max_scratch = 0usize;
    if do_y {
        max_scratch = max_scratch.max(FG_BLOCK_SIZE * y_stride);
    }
    if do_u {
        max_scratch = max_scratch.max(bh_alloc * u_stride);
    }
    if do_v {
        max_scratch = max_scratch.max(bh_alloc * v_stride);
    }

    let luts_arc = Arc::new(prep.luts);
    let scaling_arc = Arc::new(prep.scaling);
    let fg_data = *data;
    let fgy_fn = dsp.fgy_32x32xn;
    let fguv_fns = dsp.fguv_32x32xn;
    let scratch_pool = Arc::new(pool.handle());

    for wave_start in (0..rows).step_by(wave_size) {
        let count = (rows - wave_start).min(wave_size);
        if jobs.len() > count {
            jobs.truncate(count);
        }
        for (i, job) in jobs.iter_mut().enumerate() {
            let row = wave_start + i;
            let y0 = row * FG_BLOCK_SIZE;
            let band = (h - y0).min(FG_BLOCK_SIZE);
            let bh = (band + ss_y) >> ss_y;
            job.row = row;
            job.band = band;
            job.bh = bh;
            if need_y_in_job && let Some(y_pl) = frame.plane(PlaneType::Y) {
                let off = y0 * y_stride;
                let len = band * y_stride;
                job.y_buf[..len].copy_from_slice(&y_pl.data()[off..off + len]);
            }
            if do_u && let Some(u_pl) = frame.plane(PlaneType::U) {
                let off = (y0 * u_stride) >> ss_y;
                let len = bh * u_stride;
                job.u_buf[..len].copy_from_slice(&u_pl.data()[off..off + len]);
            }
            if do_v && let Some(v_pl) = frame.plane(PlaneType::V) {
                let off = (y0 * v_stride) >> ss_y;
                let len = bh * v_stride;
                job.v_buf[..len].copy_from_slice(&v_pl.data()[off..off + len]);
            }
        }

        let luts_w = Arc::clone(&luts_arc);
        let scaling_w = Arc::clone(&scaling_arc);
        let sp_w = Arc::clone(&scratch_pool);
        pool.execute_tasks(&mut jobs, move |wid, _idx, job: &mut InPlaceWaveJob| {
            sp_w.scratch().with_scratch(wid, |ws| {
                let tmp = ws.ensure_bytes(max_scratch)?;
                // 1) Kroma düzlemlerini grain'siz `job.y_buf` okunurken sentezle.
                if let Some(l) = layout.filter(|_| do_u || do_v) {
                    let fguv_fn = fguv_fns[l.index()];
                    let (luma_ref, lstride): (&[u8], usize) = if pad_luma {
                        fill_luma_strip(
                            &mut job.luma_pad,
                            luma_pad_stride,
                            &job.y_buf,
                            y_stride,
                            0,
                            job.band,
                            w,
                            bpp,
                            true,
                        );
                        (&job.luma_pad, luma_pad_stride)
                    } else {
                        (&job.y_buf, y_stride)
                    };
                    if do_u && u_stride > 0 {
                        let len = job.bh * u_stride;
                        tmp[..len].copy_from_slice(&job.u_buf[..len]);
                        let si = if fg_data.chroma_scaling_from_luma {
                            0
                        } else {
                            1
                        };
                        crate::dsp::profil_olc!(
                            FgUv,
                            (fguv_fn)(
                                &mut job.u_buf[..len],
                                &tmp[..len],
                                u_stride,
                                &fg_data,
                                cpw,
                                &scaling_w[si],
                                &luts_w[1],
                                job.bh,
                                job.row,
                                luma_ref,
                                lstride,
                                0,
                                is_id,
                                bd,
                            )
                        );
                    }
                    if do_v && v_stride > 0 {
                        let len = job.bh * v_stride;
                        tmp[..len].copy_from_slice(&job.v_buf[..len]);
                        let si = if fg_data.chroma_scaling_from_luma {
                            0
                        } else {
                            2
                        };
                        crate::dsp::profil_olc!(
                            FgUv,
                            (fguv_fn)(
                                &mut job.v_buf[..len],
                                &tmp[..len],
                                v_stride,
                                &fg_data,
                                cpw,
                                &scaling_w[si],
                                &luts_w[2],
                                job.bh,
                                job.row,
                                luma_ref,
                                lstride,
                                1,
                                is_id,
                                bd,
                            )
                        );
                    }
                }
                // 2) Kroma tamamlandıktan sonra Luma şeridini yerinde sentezle.
                if do_y && y_stride > 0 {
                    let len = job.band * y_stride;
                    tmp[..len].copy_from_slice(&job.y_buf[..len]);
                    crate::dsp::profil_olc!(
                        FgY,
                        (fgy_fn)(
                            &mut job.y_buf[..len],
                            &tmp[..len],
                            y_stride,
                            &fg_data,
                            w,
                            &scaling_w[0],
                            &luts_w[0],
                            job.band,
                            job.row,
                            bd,
                        )
                    );
                }
                Ok(())
            })
        })?;

        for job in &jobs {
            let y0 = job.row * FG_BLOCK_SIZE;
            if do_y && let Some(y_pl) = frame.plane_mut(PlaneType::Y) {
                copy_rows_to_frame(
                    y_pl.data_mut(),
                    y0 * y_stride,
                    y_stride,
                    &job.y_buf,
                    job.band,
                    w * bpp,
                );
            }
            if do_u && let Some(u_pl) = frame.plane_mut(PlaneType::U) {
                copy_rows_to_frame(
                    u_pl.data_mut(),
                    (y0 * u_stride) >> ss_y,
                    u_stride,
                    &job.u_buf,
                    job.bh,
                    cpw * bpp,
                );
            }
            if do_v && let Some(v_pl) = frame.plane_mut(PlaneType::V) {
                copy_rows_to_frame(
                    v_pl.data_mut(),
                    (y0 * v_stride) >> ss_y,
                    v_stride,
                    &job.v_buf,
                    job.bh,
                    cpw * bpp,
                );
            }
        }
    }
    Ok(())
}

/// Referans olarak saklanan bir karenin `&mut PicBuf` düzlemlerinden ara
/// grain'siz [`Frame`] kopyası oluşturmadan tek geçişte grain'li çıktı
/// [`Frame`]'ini üretir (`n_fc == 1`, `refresh_frame_flags != 0` yolu; `4.A.1`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_grain_from_pic_mut(
    pic: &mut crate::decode::intra::PicBuf,
    chroma: ChromaFormat,
    bd_in: BitDepth,
    data: &FilmGrainData,
    is_id: bool,
    dsp: &FgDsp,
    trace: Option<u64>,
    recycled: [Vec<u8>; 3],
    butce: &crate::frame::butce::Butce,
    pool: &crate::thread::WorkerPool,
) -> crate::error::Result<Frame> {
    let bd = fg_bd(bd_in);
    let layout = fg_layout(chroma);
    let do_y = data.num_y_points != 0;
    let do_uv = layout.is_some()
        && (data.num_uv_points[0] != 0
            || data.num_uv_points[1] != 0
            || data.chroma_scaling_from_luma);
    let do_u = do_uv && (data.chroma_scaling_from_luma || data.num_uv_points[0] != 0);
    let do_v = do_uv && (data.chroma_scaling_from_luma || data.num_uv_points[1] != 0);

    let mut out = Frame::new_for_grain_from_pic_in(
        pic.width,
        pic.height,
        chroma,
        bd_in,
        &pic.data,
        pic.stride,
        [do_y, do_u, do_v],
        recycled,
        butce,
    )?;

    let rows = pic.height.div_ceil(FG_BLOCK_SIZE);
    if pool.is_parallel() && rows > 1 && trace.is_none() {
        use std::sync::Arc;
        let prep = prep_grain(pic.width, pic.height, bd, layout, data, dsp, None);
        alloc_pic_grain_temp_budget(
            pic.width, pic.height, pic.stride, &out, bd, layout, do_y, do_u, do_v, butce,
        )?;
        let src_arc = Arc::new([
            core::mem::take(&mut pic.data[0]),
            core::mem::take(&mut pic.data[1]),
            core::mem::take(&mut pic.data[2]),
        ]);
        let src_stride = pic.stride;
        let res = apply_grain_parallel_waves_shared(
            &src_arc,
            move |arr, idx| (arr[idx].as_slice(), src_stride[idx]),
            &mut out,
            data,
            is_id,
            dsp,
            bd,
            layout,
            do_y,
            do_u,
            do_v,
            &prep,
            rows,
            pool,
        );
        let [p0, p1, p2] = Arc::try_unwrap(src_arc).unwrap_or_else(|a| (*a).clone());
        pic.data = [p0, p1, p2];
        res?;
        out.metadata_mut().film_grain_applied = true;
        return Ok(out);
    }

    let src_data = [
        pic.data[0].as_slice(),
        pic.data[1].as_slice(),
        pic.data[2].as_slice(),
    ];
    apply_grain_into_out_slices(
        src_data, pic.stride, &mut out, data, is_id, dsp, bd, layout, trace, butce, pool,
    )?;
    out.metadata_mut().film_grain_applied = true;
    Ok(out)
}

/// Paylaşılan `Arc<RefFrame>` içindeki `PicBuf` düzlemlerinden tek geçişte
/// grain'li çıktı [`Frame`]'ini üretir (`show_existing_frame` ve boru hattı yolu; `4.A.1`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_grain_from_ref_pic(
    rf: &std::sync::Arc<crate::decode::intra::RefFrame>,
    data: &FilmGrainData,
    is_id: bool,
    dsp: &FgDsp,
    trace: Option<u64>,
    recycled: [Vec<u8>; 3],
    butce: &crate::frame::butce::Butce,
    pool: &crate::thread::WorkerPool,
) -> crate::error::Result<Frame> {
    let pic = &rf.pic;
    let bd = fg_bd(rf.bd);
    let layout = fg_layout(rf.chroma);
    let do_y = data.num_y_points != 0;
    let do_uv = layout.is_some()
        && (data.num_uv_points[0] != 0
            || data.num_uv_points[1] != 0
            || data.chroma_scaling_from_luma);
    let do_u = do_uv && (data.chroma_scaling_from_luma || data.num_uv_points[0] != 0);
    let do_v = do_uv && (data.chroma_scaling_from_luma || data.num_uv_points[1] != 0);

    let mut out = Frame::new_for_grain_from_pic_in(
        pic.width,
        pic.height,
        rf.chroma,
        rf.bd,
        &pic.data,
        pic.stride,
        [do_y, do_u, do_v],
        recycled,
        butce,
    )?;

    let rows = pic.height.div_ceil(FG_BLOCK_SIZE);
    if pool.is_parallel() && rows > 1 && trace.is_none() {
        let prep = prep_grain(pic.width, pic.height, bd, layout, data, dsp, None);
        alloc_pic_grain_temp_budget(
            pic.width, pic.height, pic.stride, &out, bd, layout, do_y, do_u, do_v, butce,
        )?;
        apply_grain_parallel_waves_shared(
            rf,
            |r, idx| (r.pic.data[idx].as_slice(), r.pic.stride[idx]),
            &mut out,
            data,
            is_id,
            dsp,
            bd,
            layout,
            do_y,
            do_u,
            do_v,
            &prep,
            rows,
            pool,
        )?;
        out.metadata_mut().film_grain_applied = true;
        return Ok(out);
    }

    let src_data = [
        pic.data[0].as_slice(),
        pic.data[1].as_slice(),
        pic.data[2].as_slice(),
    ];
    apply_grain_into_out_slices(
        src_data, pic.stride, &mut out, data, is_id, dsp, bd, layout, trace, butce, pool,
    )?;
    out.metadata_mut().film_grain_applied = true;
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn alloc_pic_grain_temp_budget(
    w: usize,
    h: usize,
    src_stride: [usize; 3],
    out: &Frame,
    bd: FgBitDepth,
    layout: Option<FgLayout>,
    do_y: bool,
    do_u: bool,
    do_v: bool,
    butce: &crate::frame::butce::Butce,
) -> crate::error::Result<()> {
    let bpp = bd.bytes();
    let (ss_x, ss_y) = layout.map_or((0, 0), |l| (l.ss_x(), l.ss_y()));
    let band_max = FG_BLOCK_SIZE.min(h);
    let bh_max = (band_max + ss_y) >> ss_y;
    let out_strides = [
        out.plane(PlaneType::Y).map_or(0, |p| p.stride()),
        out.plane(PlaneType::U).map_or(0, |p| p.stride()),
        out.plane(PlaneType::V).map_or(0, |p| p.stride()),
    ];
    let mut max_mismatch = 0usize;
    if do_y && src_stride[0] != out_strides[0] {
        max_mismatch = max_mismatch.max(band_max * out_strides[0]);
    }
    if do_u && src_stride[1] != out_strides[1] {
        max_mismatch = max_mismatch.max(bh_max * out_strides[1]);
    }
    if do_v && src_stride[2] != out_strides[2] {
        max_mismatch = max_mismatch.max(bh_max * out_strides[2]);
    }
    let _src_tok = if max_mismatch > 0 {
        Some(butce.gecici(max_mismatch, 0u8)?)
    } else {
        None
    };
    if (do_u || do_v) && (w & ss_x) != 0 {
        let lstride = w.checked_add(1).and_then(|wp| wp.checked_mul(bpp)).ok_or(
            crate::error::Error::SizeOverflow {
                stage: crate::error::SizeStage::RowBytes,
                lhs: w,
                rhs: bpp,
            },
        )?;
        let n = lstride
            .checked_mul(band_max)
            .ok_or(crate::error::Error::SizeOverflow {
                stage: crate::error::SizeStage::PlaneArea,
                lhs: lstride,
                rhs: band_max,
            })?;
        let _luma_tok = butce.gecici(n, 0u8)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_grain_into_out_slices(
    src_data: [&[u8]; 3],
    src_stride: [usize; 3],
    out: &mut Frame,
    data: &FilmGrainData,
    is_id: bool,
    dsp: &FgDsp,
    bd: FgBitDepth,
    layout: Option<FgLayout>,
    trace: Option<u64>,
    butce: &crate::frame::butce::Butce,
    pool: &crate::thread::WorkerPool,
) -> crate::error::Result<()> {
    let w = out.width();
    let h = out.height();
    let bpp = bd.bytes();
    let (ss_x, ss_y) = layout.map_or((0, 0), |l| (l.ss_x(), l.ss_y()));
    let do_y = data.num_y_points != 0;
    let do_uv = layout.is_some()
        && (data.num_uv_points[0] != 0
            || data.num_uv_points[1] != 0
            || data.chroma_scaling_from_luma);
    let do_u = do_uv && (data.chroma_scaling_from_luma || data.num_uv_points[0] != 0);
    let do_v = do_uv && (data.chroma_scaling_from_luma || data.num_uv_points[1] != 0);

    let prep = prep_grain(w, h, bd, layout, data, dsp, trace);
    if !do_y && !do_u && !do_v {
        return Ok(());
    }

    let out_strides = [
        out.plane(PlaneType::Y).map_or(0, |p| p.stride()),
        out.plane(PlaneType::U).map_or(0, |p| p.stride()),
        out.plane(PlaneType::V).map_or(0, |p| p.stride()),
    ];
    let band_max = FG_BLOCK_SIZE.min(h);
    let bh_max = (band_max + ss_y) >> ss_y;
    let mut max_mismatch = 0usize;
    if do_y && src_stride[0] != out_strides[0] {
        max_mismatch = max_mismatch.max(band_max * out_strides[0]);
    }
    if do_u && src_stride[1] != out_strides[1] {
        max_mismatch = max_mismatch.max(bh_max * out_strides[1]);
    }
    if do_v && src_stride[2] != out_strides[2] {
        max_mismatch = max_mismatch.max(bh_max * out_strides[2]);
    }
    let mut src_strip = if max_mismatch > 0 {
        Some(butce.gecici(max_mismatch, 0u8)?)
    } else {
        None
    };

    let pad_luma = (do_u || do_v) && (w & ss_x) != 0;
    let luma_pad_stride = if pad_luma {
        w.checked_add(1).and_then(|wp| wp.checked_mul(bpp)).ok_or(
            crate::error::Error::SizeOverflow {
                stage: crate::error::SizeStage::RowBytes,
                lhs: w,
                rhs: bpp,
            },
        )?
    } else {
        0
    };
    let mut luma_strip = if pad_luma {
        let n = luma_pad_stride
            .checked_mul(band_max)
            .ok_or(crate::error::Error::SizeOverflow {
                stage: crate::error::SizeStage::PlaneArea,
                lhs: luma_pad_stride,
                rhs: band_max,
            })?;
        Some(butce.gecici(n, 0u8)?)
    } else {
        None
    };

    let cpw = (w + ss_x) >> ss_x;
    let rows = h.div_ceil(FG_BLOCK_SIZE);
    if pool.is_parallel() && rows > 1 && trace.is_none() {
        use crate::frame::butce::kopya_vec;
        use std::sync::Arc;
        drop(src_strip);
        drop(luma_strip);
        let shared = Arc::new([
            if do_y || do_u || do_v {
                kopya_vec(src_data[0])?
            } else {
                Vec::new()
            },
            if do_u {
                kopya_vec(src_data[1])?
            } else {
                Vec::new()
            },
            if do_v {
                kopya_vec(src_data[2])?
            } else {
                Vec::new()
            },
        ]);
        return apply_grain_parallel_waves_shared(
            &shared,
            move |arr, idx| (arr[idx].as_slice(), src_stride[idx]),
            out,
            data,
            is_id,
            dsp,
            bd,
            layout,
            do_y,
            do_u,
            do_v,
            &prep,
            rows,
            pool,
        );
    }

    for row in 0..rows {
        let y0 = row * FG_BLOCK_SIZE;
        let band = (h - y0).min(FG_BLOCK_SIZE);
        if do_y && let Some(y_out) = out.plane_mut(PlaneType::Y) {
            let dst_stride = out_strides[0];
            let dst_off = y0 * dst_stride;
            let src_off = y0 * src_stride[0];
            let row_bytes = w * bpp;
            let src_ref: &[u8] = if src_stride[0] == dst_stride {
                &src_data[0][src_off..]
            } else {
                let st = src_strip.as_deref_mut().expect("src_strip");
                for r in 0..band {
                    let d = r * dst_stride;
                    let s = src_off + r * src_stride[0];
                    st[d..d + row_bytes].copy_from_slice(&src_data[0][s..s + row_bytes]);
                }
                &st[..band * dst_stride]
            };
            crate::dsp::profil_olc!(
                FgY,
                (dsp.fgy_32x32xn)(
                    &mut y_out.data_mut()[dst_off..],
                    src_ref,
                    dst_stride,
                    data,
                    w,
                    &prep.scaling[0],
                    &prep.luts[0],
                    band,
                    row,
                    bd,
                )
            );
            if let Some(pic) = trace {
                let hsh = rows_hash(&y_out.data()[dst_off..], dst_stride, band, row_bytes);
                eprintln!("Y {pic} {row} {band} {w} {hsh}");
            }
        }
        let Some(l) = layout.filter(|_| do_u || do_v) else {
            continue;
        };
        let bh = (band + ss_y) >> ss_y;
        let (luma_ref, lstride): (&[u8], usize) = if let Some(lbuf) = luma_strip.as_deref_mut() {
            fill_luma_strip(
                lbuf,
                luma_pad_stride,
                src_data[0],
                src_stride[0],
                y0,
                band,
                w,
                bpp,
                true,
            );
            (lbuf, luma_pad_stride)
        } else {
            (&src_data[0][y0 * src_stride[0]..], src_stride[0])
        };
        let row_bytes_uv = cpw * bpp;
        for pl in 0..2 {
            let active = if pl == 0 { do_u } else { do_v };
            if !active {
                continue;
            }
            let ptype = if pl == 0 { PlaneType::U } else { PlaneType::V };
            let Some(c_out) = out.plane_mut(ptype) else {
                continue;
            };
            let dst_stride = out_strides[1 + pl];
            let s_stride = src_stride[1 + pl];
            let dst_off = (y0 * dst_stride) >> ss_y;
            let src_off = (y0 * s_stride) >> ss_y;
            let src_ref: &[u8] = if s_stride == dst_stride {
                &src_data[1 + pl][src_off..]
            } else {
                let st = src_strip.as_deref_mut().expect("src_strip");
                for r in 0..bh {
                    let d = r * dst_stride;
                    let s = src_off + r * s_stride;
                    st[d..d + row_bytes_uv].copy_from_slice(&src_data[1 + pl][s..s + row_bytes_uv]);
                }
                &st[..bh * dst_stride]
            };
            let si = if data.chroma_scaling_from_luma {
                0
            } else {
                1 + pl
            };
            crate::dsp::profil_olc!(
                FgUv,
                (dsp.fguv_32x32xn[l.index()])(
                    &mut c_out.data_mut()[dst_off..],
                    src_ref,
                    dst_stride,
                    data,
                    cpw,
                    &prep.scaling[si],
                    &prep.luts[1 + pl],
                    bh,
                    row,
                    luma_ref,
                    lstride,
                    pl,
                    is_id,
                    bd,
                )
            );
            if let Some(pic) = trace {
                let hsh = rows_hash(&c_out.data()[dst_off..], dst_stride, bh, row_bytes_uv);
                eprintln!(
                    "C {pic} {} {row} {bh} {cpw} {} {hsh}",
                    pl + 1,
                    u8::from(is_id)
                );
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_grain_parallel_waves_shared<P, G>(
    holder: &std::sync::Arc<P>,
    get_plane: G,
    out: &mut Frame,
    data: &FilmGrainData,
    is_id: bool,
    dsp: &FgDsp,
    bd: FgBitDepth,
    layout: Option<FgLayout>,
    do_y: bool,
    do_u: bool,
    do_v: bool,
    prep: &GrainPrep,
    rows: usize,
    pool: &crate::thread::WorkerPool,
) -> crate::error::Result<()>
where
    P: Send + Sync + 'static,
    G: Fn(&P, usize) -> (&[u8], usize) + Copy + Send + Sync + 'static,
{
    use crate::frame::butce::dolu_vec;
    use std::sync::Arc;

    struct WaveJob {
        row: usize,
        band: usize,
        bh: usize,
        y_dst: Vec<u8>,
        u_dst: Vec<u8>,
        v_dst: Vec<u8>,
        luma_pad: Vec<u8>,
    }

    let w = out.width();
    let h = out.height();
    let bpp = bd.bytes();
    let (ss_x, ss_y) = layout.map_or((0, 0), |l| (l.ss_x(), l.ss_y()));
    let cpw = (w + ss_x) >> ss_x;
    let y_dst_stride = out.plane(PlaneType::Y).map_or(0, |p| p.stride());
    let u_dst_stride = out.plane(PlaneType::U).map_or(0, |p| p.stride());
    let v_dst_stride = out.plane(PlaneType::V).map_or(0, |p| p.stride());
    let pad_luma = (do_u || do_v) && (w & ss_x) != 0;
    let luma_pad_stride = if pad_luma { (w + 1) * bpp } else { 0 };
    let bh_alloc = (FG_BLOCK_SIZE + ss_y) >> ss_y;

    let (_, s0_stride) = get_plane(holder, 0);
    let (_, s1_stride) = get_plane(holder, 1);
    let (_, s2_stride) = get_plane(holder, 2);
    let mut max_mismatch = 0usize;
    if do_y && s0_stride != y_dst_stride {
        max_mismatch = max_mismatch.max(FG_BLOCK_SIZE * y_dst_stride);
    }
    if do_u && s1_stride != u_dst_stride {
        max_mismatch = max_mismatch.max(bh_alloc * u_dst_stride);
    }
    if do_v && s2_stride != v_dst_stride {
        max_mismatch = max_mismatch.max(bh_alloc * v_dst_stride);
    }

    let wave_size = pool.n_threads().min(rows).max(1);
    let mut jobs: Vec<WaveJob> = (0..wave_size)
        .map(|_| {
            Ok(WaveJob {
                row: 0,
                band: 0,
                bh: 0,
                y_dst: if do_y {
                    dolu_vec(FG_BLOCK_SIZE * y_dst_stride, 0u8)?
                } else {
                    Vec::new()
                },
                u_dst: if do_u {
                    dolu_vec(bh_alloc * u_dst_stride, 0u8)?
                } else {
                    Vec::new()
                },
                v_dst: if do_v {
                    dolu_vec(bh_alloc * v_dst_stride, 0u8)?
                } else {
                    Vec::new()
                },
                luma_pad: if pad_luma {
                    dolu_vec(FG_BLOCK_SIZE * luma_pad_stride, 0u8)?
                } else {
                    Vec::new()
                },
            })
        })
        .collect::<crate::error::Result<Vec<_>>>()?;

    let luts_arc = Arc::new(prep.luts);
    let scaling_arc = Arc::new(prep.scaling);
    let fg_data = *data;
    let fgy_fn = dsp.fgy_32x32xn;
    let fguv_fns = dsp.fguv_32x32xn;
    let scratch_pool = Arc::new(pool.handle());

    for wave_start in (0..rows).step_by(wave_size) {
        let count = (rows - wave_start).min(wave_size);
        if jobs.len() > count {
            jobs.truncate(count);
        }
        for (i, job) in jobs.iter_mut().enumerate() {
            let row = wave_start + i;
            let band = (h - row * FG_BLOCK_SIZE).min(FG_BLOCK_SIZE);
            job.row = row;
            job.band = band;
            job.bh = (band + ss_y) >> ss_y;
        }

        let holder_w = Arc::clone(holder);
        let luts_w = Arc::clone(&luts_arc);
        let scaling_w = Arc::clone(&scaling_arc);
        let sp_w = Arc::clone(&scratch_pool);
        pool.execute_tasks(&mut jobs, move |wid, _idx, job: &mut WaveJob| {
            let y0 = job.row * FG_BLOCK_SIZE;
            let (src_y, src_y_stride) = get_plane(&holder_w, 0);
            sp_w.scratch().with_scratch(wid, |ws| {
                let tmp = if max_mismatch > 0 {
                    ws.ensure_bytes(max_mismatch)?
                } else {
                    &mut [][..]
                };
                if do_y && y_dst_stride > 0 {
                    let row_bytes = w * bpp;
                    let src_off = y0 * src_y_stride;
                    let src_ref: &[u8] = if src_y_stride == y_dst_stride {
                        &src_y[src_off..]
                    } else {
                        for r in 0..job.band {
                            let d = r * y_dst_stride;
                            let s = src_off + r * src_y_stride;
                            tmp[d..d + row_bytes].copy_from_slice(&src_y[s..s + row_bytes]);
                        }
                        &tmp[..job.band * y_dst_stride]
                    };
                    crate::dsp::profil_olc!(
                        FgY,
                        (fgy_fn)(
                            &mut job.y_dst[..job.band * y_dst_stride],
                            src_ref,
                            y_dst_stride,
                            &fg_data,
                            w,
                            &scaling_w[0],
                            &luts_w[0],
                            job.band,
                            job.row,
                            bd,
                        )
                    );
                }
                if let Some(l) = layout.filter(|_| do_u || do_v) {
                    let fguv_fn = fguv_fns[l.index()];
                    let (luma_ref, lstride): (&[u8], usize) = if pad_luma {
                        fill_luma_strip(
                            &mut job.luma_pad,
                            luma_pad_stride,
                            src_y,
                            src_y_stride,
                            y0,
                            job.band,
                            w,
                            bpp,
                            true,
                        );
                        (&job.luma_pad, luma_pad_stride)
                    } else {
                        (&src_y[y0 * src_y_stride..], src_y_stride)
                    };
                    let row_bytes_uv = cpw * bpp;
                    for pl in 0..2 {
                        let (active, dst_stride, dst_buf) = if pl == 0 {
                            (do_u, u_dst_stride, &mut job.u_dst)
                        } else {
                            (do_v, v_dst_stride, &mut job.v_dst)
                        };
                        if !active || dst_stride == 0 {
                            continue;
                        }
                        let (src_c, src_c_stride) = get_plane(&holder_w, 1 + pl);
                        let src_off = (y0 * src_c_stride) >> ss_y;
                        let src_ref: &[u8] = if src_c_stride == dst_stride {
                            &src_c[src_off..]
                        } else {
                            for r in 0..job.bh {
                                let d = r * dst_stride;
                                let s = src_off + r * src_c_stride;
                                tmp[d..d + row_bytes_uv]
                                    .copy_from_slice(&src_c[s..s + row_bytes_uv]);
                            }
                            &tmp[..job.bh * dst_stride]
                        };
                        let si = if fg_data.chroma_scaling_from_luma {
                            0
                        } else {
                            1 + pl
                        };
                        crate::dsp::profil_olc!(
                            FgUv,
                            (fguv_fn)(
                                &mut dst_buf[..job.bh * dst_stride],
                                src_ref,
                                dst_stride,
                                &fg_data,
                                cpw,
                                &scaling_w[si],
                                &luts_w[1 + pl],
                                job.bh,
                                job.row,
                                luma_ref,
                                lstride,
                                pl,
                                is_id,
                                bd,
                            )
                        );
                    }
                }
                Ok(())
            })
        })?;

        for job in &jobs {
            let y0 = job.row * FG_BLOCK_SIZE;
            if do_y && let Some(y_out) = out.plane_mut(PlaneType::Y) {
                copy_rows_to_frame(
                    y_out.data_mut(),
                    y0 * y_dst_stride,
                    y_dst_stride,
                    &job.y_dst,
                    job.band,
                    w * bpp,
                );
            }
            if do_u && let Some(u_out) = out.plane_mut(PlaneType::U) {
                copy_rows_to_frame(
                    u_out.data_mut(),
                    (y0 * u_dst_stride) >> ss_y,
                    u_dst_stride,
                    &job.u_dst,
                    job.bh,
                    cpw * bpp,
                );
            }
            if do_v && let Some(v_out) = out.plane_mut(PlaneType::V) {
                copy_rows_to_frame(
                    v_out.data_mut(),
                    (y0 * v_dst_stride) >> ss_y,
                    v_dst_stride,
                    &job.v_dst,
                    job.bh,
                    cpw * bpp,
                );
            }
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

            // Yerinde (`apply_grain_in_place`) tek ve çok iş parçacıklı şerit
            // sentezi `apply_grain` ile bit-birebir özdeş olmalıdır.
            let mut f_in_place_st = f.clone();
            apply_grain_in_place(
                &mut f_in_place_st,
                &data(),
                false,
                &FgDsp::scalar_for(fg_bd),
                None,
                &crate::frame::butce::Butce::sinirsiz(),
                &crate::thread::WorkerPool::new(1),
            )
            .expect("in_place st");
            assert_eq!(f_in_place_st, out);

            let mut f_in_place_mt = f.clone();
            apply_grain_in_place(
                &mut f_in_place_mt,
                &data(),
                false,
                &FgDsp::scalar_for(fg_bd),
                None,
                &crate::frame::butce::Butce::sinirsiz(),
                &crate::thread::WorkerPool::new(4),
            )
            .expect("in_place mt");
            assert_eq!(f_in_place_mt, out);

            // `PicBuf` kaynaklı (`apply_grain_from_pic_mut`) tek geçişli sentez
            // (`128` hizalı `PicBuf` adımı ≠ `64` hizalı `Frame` adımı dâhil)
            // bit-birebir özdeş olmalıdır.
            let bpp = bd.bytes_per_sample();
            let mut pic = crate::decode::intra::PicBuf::new(
                37,
                35,
                chroma,
                bd,
                &crate::frame::butce::Butce::sinirsiz(),
            )
            .expect("pic");
            for (idx, p) in [PlaneType::Y, PlaneType::U, PlaneType::V]
                .into_iter()
                .enumerate()
            {
                if let Some(pl) = f.plane(p) {
                    let rb = pl.width() * bpp;
                    for y in 0..pl.height() {
                        let s = y * pl.stride();
                        let d = y * pic.stride[idx];
                        pic.data[idx][d..d + rb].copy_from_slice(&pl.data()[s..s + rb]);
                    }
                }
            }
            for nth in [1usize, 4] {
                let out_pic = apply_grain_from_pic_mut(
                    &mut pic,
                    chroma,
                    bd,
                    &data(),
                    false,
                    &FgDsp::scalar_for(fg_bd),
                    None,
                    [Vec::new(), Vec::new(), Vec::new()],
                    &crate::frame::butce::Butce::sinirsiz(),
                    &crate::thread::WorkerPool::new(nth),
                )
                .expect("from_pic");
                assert_eq!(out_pic.metadata(), out.metadata());
                assert_eq!(out_pic.plane_count(), out.plane_count());
                for p in [PlaneType::Y, PlaneType::U, PlaneType::V] {
                    if let (Some(a), Some(b)) = (out_pic.plane(p), out.plane(p)) {
                        assert_eq!(a.layout(), b.layout());
                        for y in 0..a.height() {
                            assert_eq!(a.row(y), b.row(y), "from_pic nth={nth} p={p:?} y={y}");
                        }
                    }
                }
            }
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
