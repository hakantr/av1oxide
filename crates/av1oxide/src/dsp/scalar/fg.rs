//! Film grain — dav1d `src/filmgrain_tmpl.c` skaler gövdeleri
//! (`Dav1dFilmGrainDSPContext`). `generate_scaling` dav1d'de DSP bağlamında
//! değil, uygulama katmanındadır (`src/fg_apply_tmpl.c:41-97`); bu port onu
//! [`crate::decode::filmgrain::generate_scaling`]'e taşır (F3c).
//!
//! Sözleşme: [`crate::dsp::fg`]. 8/10/12 bit tek gövde; piksel okuma ve yazma
//! bayt dilimleri üzerindendir (16bpc küçük sonlu).

#![forbid(unsafe_code)]
// Döngüler dav1d C döngülerini birebir izler (indeks = dav1d `x`/`y`/`i`).
#![allow(clippy::needless_range_loop)]

use crate::bitstream::frame_header::FilmGrainData;
use crate::dsp::fg::{
    FG_BLOCK_SIZE, FgBitDepth, GRAIN_HEIGHT, GRAIN_WIDTH, GrainLut, SUB_GRAIN_HEIGHT,
    SUB_GRAIN_WIDTH,
};
use crate::dsp::fg_tables::GAUSSIAN_SEQUENCE;

/// F3c: `BPC`'nin bit derinliği (derleme zamanı sabiti). Girişler çağıranın
/// `bd` parametresini yalnız `debug_assert` ile denetler ve gövdeyi bu sabitle
/// çalıştırır; `bpp`, `bitdepth_max` ve kaydırmalar böylece monomorfta sabittir.
#[inline(always)]
const fn bpc_bd<const BPC: u32>() -> FgBitDepth {
    match BPC {
        8 => FgBitDepth::B8,
        10 => FgBitDepth::B10,
        _ => FgBitDepth::B12,
    }
}

/// Piksel başına bayt — `BPC` derleme zamanı sabiti (kapanışlarda da katlanır).
#[inline(always)]
const fn bpp_of<const BPC: u32>() -> usize {
    if BPC == 8 { 1 } else { 2 }
}

/// Piksel okuma — `BPC` ile monomorf (F3c): genişlik derleme zamanı sabitidir;
/// kapanışlardan çağrıldığında da çalışma zamanı `bpp` dalı oluşmaz.
#[inline(always)]
fn load<const BPC: u32>(buf: &[u8], idx: usize) -> i32 {
    if BPC == 8 {
        i32::from(buf[idx])
    } else {
        i32::from(u16::from_le_bytes([buf[2 * idx], buf[2 * idx + 1]]))
    }
}

/// Piksel yazma — `BPC` ile monomorf (F3c).
#[inline(always)]
fn store<const BPC: u32>(buf: &mut [u8], idx: usize, v: i32) {
    if BPC == 8 {
        buf[idx] = v as u8;
    } else {
        buf[2 * idx..2 * idx + 2].copy_from_slice(&(v as u16).to_le_bytes());
    }
}

/// `get_random_number` (`src/filmgrain_tmpl.c:38-44`): 16 bit LFSR.
#[inline]
pub fn get_random_number(bits: u32, state: &mut u32) -> i32 {
    let r = *state;
    let bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
    *state = (r >> 1) | (bit << 15);
    ((*state >> (16 - bits)) & ((1 << bits) - 1)) as i32
}

/// `round2` (`src/filmgrain_tmpl.c:46-48`).
#[inline]
pub(crate) fn round2(x: i32, shift: u32) -> i32 {
    (x + ((1 << shift) >> 1)) >> shift
}

/// `generate_grain_y_c` (`src/filmgrain_tmpl.c:50-86`).
///
/// 73 × 82 Gauss örneği (`get_random_number(11)`, `round2(·, 4 − bd_min_8 +
/// grain_scale_shift)`), sonra `ar_coeff_lag` gecikmeli AR süzgeci
/// (`y ≥ 3`, `3 ≤ x < 79`), `[−128 << bd_min_8, (128 << bd_min_8) − 1]`'e
/// kırpılır.
pub fn generate_grain_y<const BPC: u32>(buf: &mut GrainLut, data: &FilmGrainData, bd: FgBitDepth) {
    debug_assert_eq!(bd.bits(), BPC, "yuva bit derinliği ≠ çağrı");
    let bd = bpc_bd::<BPC>();
    let ok = (0..=3).contains(&data.ar_coeff_lag)
        && (0..=3).contains(&data.grain_scale_shift)
        && data.ar_coeff_shift <= 16;
    debug_assert!(ok, "generate_grain_y sözleşme dışı girdi");
    if !ok {
        return;
    }
    let min_8 = bd.min_8();
    let mut seed = data.seed;
    let shift = 4 - min_8 + data.grain_scale_shift as u32;
    let grain_ctr = 128i32 << min_8;
    let (grain_min, grain_max) = (-grain_ctr, grain_ctr - 1);

    for row in buf.iter_mut() {
        for v in row.iter_mut() {
            let value = get_random_number(11, &mut seed) as usize;
            *v = round2(i32::from(GAUSSIAN_SEQUENCE[value]), shift) as i16;
        }
    }

    let ar_pad = 3usize;
    let ar_lag = data.ar_coeff_lag as usize;
    let ar_shift = data.ar_coeff_shift as u32;
    for y in ar_pad..GRAIN_HEIGHT {
        for x in ar_pad..GRAIN_WIDTH - ar_pad {
            let mut coeff = data.ar_coeffs_y.iter();
            let mut sum = 0i32;
            'outer: for yy in y - ar_lag..=y {
                for xx in x - ar_lag..=x + ar_lag {
                    if yy == y && xx == x {
                        break 'outer;
                    }
                    let c = coeff.next().copied().unwrap_or(0);
                    sum += i32::from(c) * i32::from(buf[yy][xx]);
                }
            }
            let grain = i32::from(buf[y][x]) + round2(sum, ar_shift);
            buf[y][x] = grain.clamp(grain_min, grain_max) as i16;
        }
    }
}

/// `generate_grain_uv_c` (`src/filmgrain_tmpl.c:88-145`).
///
/// Tohum `seed ^ (uv ? 0x49d8 : 0xb524)`; alt örneklenmiş boyut 44 × 38.
/// AR süzgecinin son katsayısı (`num_y_points ≠ 0` iken) luma grain'inin
/// `(1 + subx) × (1 + suby)` ortalamasına uygulanır.
fn generate_grain_uv<const BPC: u32>(
    buf: &mut GrainLut,
    buf_y: &GrainLut,
    data: &FilmGrainData,
    uv: usize,
    subx: usize,
    suby: usize,
    bd: FgBitDepth,
) {
    debug_assert_eq!(bd.bits(), BPC, "yuva bit derinliği ≠ çağrı");
    let bd = bpc_bd::<BPC>();
    let ok = uv < 2
        && (0..=3).contains(&data.ar_coeff_lag)
        && (0..=3).contains(&data.grain_scale_shift)
        && data.ar_coeff_shift <= 16;
    debug_assert!(ok, "generate_grain_uv sözleşme dışı girdi");
    if !ok {
        return;
    }
    let min_8 = bd.min_8();
    let mut seed = data.seed ^ if uv != 0 { 0x49d8 } else { 0xb524 };
    let shift = 4 - min_8 + data.grain_scale_shift as u32;
    let grain_ctr = 128i32 << min_8;
    let (grain_min, grain_max) = (-grain_ctr, grain_ctr - 1);

    let chroma_w = if subx != 0 {
        SUB_GRAIN_WIDTH
    } else {
        GRAIN_WIDTH
    };
    let chroma_h = if suby != 0 {
        SUB_GRAIN_HEIGHT
    } else {
        GRAIN_HEIGHT
    };

    for row in buf.iter_mut().take(chroma_h) {
        for v in row.iter_mut().take(chroma_w) {
            let value = get_random_number(11, &mut seed) as usize;
            *v = round2(i32::from(GAUSSIAN_SEQUENCE[value]), shift) as i16;
        }
    }

    let ar_pad = 3usize;
    let ar_lag = data.ar_coeff_lag as usize;
    let ar_shift = data.ar_coeff_shift as u32;
    let coeffs = &data.ar_coeffs_uv[uv];
    for y in ar_pad..chroma_h {
        for x in ar_pad..chroma_w - ar_pad {
            let mut ci = 0usize;
            let mut sum = 0i32;
            'outer: for yy in y - ar_lag..=y {
                for xx in x - ar_lag..=x + ar_lag {
                    if yy == y && xx == x {
                        // Son (geçerli) piksel: luma grain katkısı.
                        if data.num_y_points == 0 {
                            break 'outer;
                        }
                        let mut luma = 0i32;
                        let luma_x = ((x - ar_pad) << subx) + ar_pad;
                        let luma_y = ((y - ar_pad) << suby) + ar_pad;
                        for i in 0..=suby {
                            for j in 0..=subx {
                                luma += i32::from(buf_y[luma_y + i][luma_x + j]);
                            }
                        }
                        let luma = round2(luma, (subx + suby) as u32);
                        sum += luma * i32::from(coeffs[ci]);
                        break 'outer;
                    }
                    sum += i32::from(coeffs[ci]) * i32::from(buf[yy][xx]);
                    ci += 1;
                }
            }
            let grain = i32::from(buf[y][x]) + round2(sum, ar_shift);
            buf[y][x] = grain.clamp(grain_min, grain_max) as i16;
        }
    }
}

/// `generate_grain_uv_420_c` (`src/filmgrain_tmpl.c:147-150`, `152`).
pub fn generate_grain_uv_420<const BPC: u32>(
    buf: &mut GrainLut,
    buf_y: &GrainLut,
    data: &FilmGrainData,
    uv: usize,
    bd: FgBitDepth,
) {
    generate_grain_uv::<BPC>(buf, buf_y, data, uv, 1, 1, bd);
}

/// `generate_grain_uv_422_c` (`src/filmgrain_tmpl.c:147-150`, `153`).
pub fn generate_grain_uv_422<const BPC: u32>(
    buf: &mut GrainLut,
    buf_y: &GrainLut,
    data: &FilmGrainData,
    uv: usize,
    bd: FgBitDepth,
) {
    generate_grain_uv::<BPC>(buf, buf_y, data, uv, 1, 0, bd);
}

/// `generate_grain_uv_444_c` (`src/filmgrain_tmpl.c:147-150`, `154`).
pub fn generate_grain_uv_444<const BPC: u32>(
    buf: &mut GrainLut,
    buf_y: &GrainLut,
    data: &FilmGrainData,
    uv: usize,
    bd: FgBitDepth,
) {
    generate_grain_uv::<BPC>(buf, buf_y, data, uv, 0, 0, bd);
}

/// `sample_lut` (`src/filmgrain_tmpl.c:158-167`). P39: SIMD gövdelerinin
/// kenar pikselleri de bu işlevi kullanır.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn sample_lut(
    lut: &GrainLut,
    offsets: &[[i32; 2]; 2],
    subx: usize,
    suby: usize,
    bx: usize,
    by: usize,
    x: usize,
    y: usize,
) -> i32 {
    let randval = offsets[bx][by];
    let offx = 3 + (2 >> subx) * (3 + (randval >> 4)) as usize;
    let offy = 3 + (2 >> suby) * (3 + (randval & 0xF)) as usize;
    i32::from(lut[offy + y + (FG_BLOCK_SIZE >> suby) * by][offx + x + (FG_BLOCK_SIZE >> subx) * bx])
}

/// Satır başı tohumları (`src/filmgrain_tmpl.c:190-196`, `301-307`):
/// `seed[0]` geçerli satır, `seed[1]` bir önceki. P39: SIMD gövdeleri de kullanır.
#[inline]
pub(crate) fn row_seeds(data: &FilmGrainData, row_num: usize, rows: usize) -> [u32; 2] {
    let mut seed = [0u32; 2];
    for (i, s) in seed.iter_mut().enumerate().take(rows) {
        let r = row_num as i64 - i as i64;
        *s = data.seed;
        *s ^= (((r * 37 + 178) & 0xFF) as u32) << 8;
        *s ^= ((r * 173 + 105) & 0xFF) as u32;
    }
    seed
}

/// `fgy_32x32xn_c` (`src/filmgrain_tmpl.c:169-276`).
///
/// Bir 32 satırlık grain satırını (`row_num`) `FG_BLOCK_SIZE` genişlikli
/// bloklarla işler: blok başına `get_random_number(8)` ofseti, örtüşme
/// açıkken sol 2 sütun ve üst 2 satırda `{27, 17}`/`{17, 27}` karışımı;
/// gürültü `round2(scaling[src] · grain, scaling_shift)`, çıktı
/// `[min_value, max_value]`'e (kısıtlı aralıkta `16 << b`, `235 << b`)
/// kırpılır.
#[allow(clippy::too_many_arguments)]
pub fn fgy_32x32xn<const BPC: u32>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8],
    grain_lut: &GrainLut,
    bh: usize,
    row_num: usize,
    bd: FgBitDepth,
) {
    debug_assert_eq!(bd.bits(), BPC, "yuva bit derinliği ≠ çağrı");
    let bd = bpc_bd::<BPC>();
    let bpp = bd.bytes();
    let max = bd.bitdepth_max();
    let ok = pw > 0
        && bh > 0
        && bh <= FG_BLOCK_SIZE
        && stride.is_multiple_of(bpp)
        && (8..=11).contains(&data.scaling_shift)
        && scaling.len() > max as usize
        && src.len() >= (bh - 1) * stride + pw * bpp
        && dst.len() >= (bh - 1) * stride + pw * bpp;
    debug_assert!(ok, "fgy_32x32xn sözleşme dışı girdi");
    if !ok {
        return;
    }
    let overlap = data.overlap_flag;
    let rows = 1 + usize::from(overlap && row_num > 0);
    let min_8 = bd.min_8();
    let grain_ctr = 128i32 << min_8;
    let (grain_min, grain_max) = (-grain_ctr, grain_ctr - 1);
    let (min_value, max_value) = if data.clip_to_restricted_range {
        (16 << min_8, 235 << min_8)
    } else {
        (0, max)
    };
    let shift = data.scaling_shift as u32;
    let mut seed = row_seeds(data, row_num, rows);
    let mut offsets = [[0i32; 2]; 2];
    const W: [[i32; 2]; 2] = [[27, 17], [17, 27]];

    let mut add_noise = |x: usize, y: usize, bx: usize, grain: i32| {
        let idx = y * stride / bpp_of::<BPC>() + x + bx;
        let s = load::<BPC>(src, idx);
        let noise = round2(i32::from(scaling[s as usize]) * grain, shift);
        store::<BPC>(dst, idx, (s + noise).clamp(min_value, max_value));
    };

    let mut bx = 0usize;
    while bx < pw {
        let bw = FG_BLOCK_SIZE.min(pw - bx);
        if overlap && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = get_random_number(8, &mut seed[i]);
        }
        let ystart = if overlap && row_num != 0 {
            bh.min(2)
        } else {
            0
        };
        let xstart = if overlap && bx != 0 { bw.min(2) } else { 0 };

        for y in ystart..bh {
            // Örtüşmesiz bölge.
            for x in xstart..bw {
                let grain = sample_lut(grain_lut, &offsets, 0, 0, 0, 0, x, y);
                add_noise(x, y, bx, grain);
            }
            // Örtüşen sütun.
            for x in 0..xstart {
                let grain = sample_lut(grain_lut, &offsets, 0, 0, 0, 0, x, y);
                let old = sample_lut(grain_lut, &offsets, 0, 0, 1, 0, x, y);
                let grain = round2(old * W[x][0] + grain * W[x][1], 5);
                add_noise(x, y, bx, grain.clamp(grain_min, grain_max));
            }
        }
        for y in 0..ystart {
            // Örtüşen satır (köşe hariç).
            for x in xstart..bw {
                let grain = sample_lut(grain_lut, &offsets, 0, 0, 0, 0, x, y);
                let old = sample_lut(grain_lut, &offsets, 0, 0, 0, 1, x, y);
                let grain = round2(old * W[y][0] + grain * W[y][1], 5);
                add_noise(x, y, bx, grain.clamp(grain_min, grain_max));
            }
            // Çift örtüşen köşe.
            for x in 0..xstart {
                let top = sample_lut(grain_lut, &offsets, 0, 0, 0, 1, x, y);
                let old = sample_lut(grain_lut, &offsets, 0, 0, 1, 1, x, y);
                let top = round2(old * W[x][0] + top * W[x][1], 5).clamp(grain_min, grain_max);

                let grain = sample_lut(grain_lut, &offsets, 0, 0, 0, 0, x, y);
                let old = sample_lut(grain_lut, &offsets, 0, 0, 1, 0, x, y);
                let grain = round2(old * W[x][0] + grain * W[x][1], 5).clamp(grain_min, grain_max);

                let grain = round2(top * W[y][0] + grain * W[y][1], 5);
                add_noise(x, y, bx, grain.clamp(grain_min, grain_max));
            }
        }
        bx += FG_BLOCK_SIZE;
    }
}

/// `fguv_32x32xn_c` (`src/filmgrain_tmpl.c:278-402`).
///
/// Kroma bloğu `FG_BLOCK_SIZE >> sx` genişliğindedir. Ölçekleme indeksi
/// luma ortalamasıdır (`sx` iken yatay iki pikselin yuvarlanmış ortalaması,
/// dikeyde yalnız üst satır); `chroma_scaling_from_luma` kapalıysa
/// `iclip_pixel(((avg · uv_luma_mult + src · uv_mult) >> 6) + uv_offset ·
/// 2^bd_min_8)`. Alt örneklenmiş eksende örtüşme ağırlığı `{23, 22}`.
#[allow(clippy::too_many_arguments)]
fn fguv_32x32xn<const BPC: u32>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8],
    grain_lut: &GrainLut,
    bh: usize,
    row_num: usize,
    luma: &[u8],
    luma_stride: usize,
    uv: usize,
    is_id: bool,
    sx: usize,
    sy: usize,
    bd: FgBitDepth,
) {
    debug_assert_eq!(bd.bits(), BPC, "yuva bit derinliği ≠ çağrı");
    let bd = bpc_bd::<BPC>();
    let bpp = bd.bytes();
    let max = bd.bitdepth_max();
    let ok = pw > 0
        && bh > 0
        && bh <= FG_BLOCK_SIZE >> sy
        && stride.is_multiple_of(bpp)
        && luma_stride.is_multiple_of(bpp)
        && uv < 2
        && (8..=11).contains(&data.scaling_shift)
        && scaling.len() > max as usize
        && src.len() >= (bh - 1) * stride + pw * bpp
        && dst.len() >= (bh - 1) * stride + pw * bpp
        && luma.len() >= ((bh - 1) << sy) * luma_stride + (((pw - 1) << sx) + sx + 1) * bpp;
    debug_assert!(ok, "fguv_32x32xn sözleşme dışı girdi");
    if !ok {
        return;
    }
    let overlap = data.overlap_flag;
    let rows = 1 + usize::from(overlap && row_num > 0);
    let min_8 = bd.min_8();
    let grain_ctr = 128i32 << min_8;
    let (grain_min, grain_max) = (-grain_ctr, grain_ctr - 1);
    let (min_value, max_value) = if data.clip_to_restricted_range {
        (16 << min_8, (if is_id { 235 } else { 240 }) << min_8)
    } else {
        (0, max)
    };
    let shift = data.scaling_shift as u32;
    let mut seed = row_seeds(data, row_num, rows);
    let mut offsets = [[0i32; 2]; 2];
    const W: [[[i32; 2]; 2]; 2] = [[[27, 17], [17, 27]], [[23, 22], [0, 0]]];
    let csfl = data.chroma_scaling_from_luma;
    let uv_luma_mult = data.uv_luma_mult[uv];
    let uv_mult = data.uv_mult[uv];
    let uv_offset = data.uv_offset[uv] * (1 << min_8);

    let mut add_noise = |x: usize, y: usize, bx: usize, grain: i32| {
        let lx = (bx + x) << sx;
        let ly = y << sy;
        let lrow = ly * luma_stride / bpp_of::<BPC>();
        let mut avg = load::<BPC>(luma, lrow + lx);
        if sx != 0 {
            avg = (avg + load::<BPC>(luma, lrow + lx + 1) + 1) >> 1;
        }
        let idx = y * stride / bpp_of::<BPC>() + bx + x;
        let s = load::<BPC>(src, idx);
        let mut val = avg;
        if !csfl {
            let combined = avg * uv_luma_mult + s * uv_mult;
            val = ((combined >> 6) + uv_offset).clamp(0, max);
        }
        let noise = round2(i32::from(scaling[val as usize]) * grain, shift);
        store::<BPC>(dst, idx, (s + noise).clamp(min_value, max_value));
    };

    let bstep = FG_BLOCK_SIZE >> sx;
    let mut bx = 0usize;
    while bx < pw {
        let bw = bstep.min(pw - bx);
        if overlap && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = get_random_number(8, &mut seed[i]);
        }
        let ystart = if overlap && row_num != 0 {
            bh.min(2 >> sy)
        } else {
            0
        };
        let xstart = if overlap && bx != 0 {
            bw.min(2 >> sx)
        } else {
            0
        };

        for y in ystart..bh {
            for x in xstart..bw {
                let grain = sample_lut(grain_lut, &offsets, sx, sy, 0, 0, x, y);
                add_noise(x, y, bx, grain);
            }
            for x in 0..xstart {
                let grain = sample_lut(grain_lut, &offsets, sx, sy, 0, 0, x, y);
                let old = sample_lut(grain_lut, &offsets, sx, sy, 1, 0, x, y);
                let grain = round2(old * W[sx][x][0] + grain * W[sx][x][1], 5);
                add_noise(x, y, bx, grain.clamp(grain_min, grain_max));
            }
        }
        for y in 0..ystart {
            for x in xstart..bw {
                let grain = sample_lut(grain_lut, &offsets, sx, sy, 0, 0, x, y);
                let old = sample_lut(grain_lut, &offsets, sx, sy, 0, 1, x, y);
                let grain = round2(old * W[sy][y][0] + grain * W[sy][y][1], 5);
                add_noise(x, y, bx, grain.clamp(grain_min, grain_max));
            }
            for x in 0..xstart {
                let top = sample_lut(grain_lut, &offsets, sx, sy, 0, 1, x, y);
                let old = sample_lut(grain_lut, &offsets, sx, sy, 1, 1, x, y);
                let top =
                    round2(old * W[sx][x][0] + top * W[sx][x][1], 5).clamp(grain_min, grain_max);

                let grain = sample_lut(grain_lut, &offsets, sx, sy, 0, 0, x, y);
                let old = sample_lut(grain_lut, &offsets, sx, sy, 1, 0, x, y);
                let grain =
                    round2(old * W[sx][x][0] + grain * W[sx][x][1], 5).clamp(grain_min, grain_max);

                let grain = round2(top * W[sy][y][0] + grain * W[sy][y][1], 5);
                add_noise(x, y, bx, grain.clamp(grain_min, grain_max));
            }
        }
        bx += bstep;
    }
}

macro_rules! fguv_ss_fn {
    ($name:ident, $sx:expr, $sy:expr, $doc:literal) => {
        #[doc = $doc]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<const BPC: u32>(
            dst: &mut [u8],
            src: &[u8],
            stride: usize,
            data: &FilmGrainData,
            pw: usize,
            scaling: &[u8],
            grain_lut: &GrainLut,
            bh: usize,
            row_num: usize,
            luma: &[u8],
            luma_stride: usize,
            uv: usize,
            is_id: bool,
            bd: FgBitDepth,
        ) {
            fguv_32x32xn::<BPC>(
                dst,
                src,
                stride,
                data,
                pw,
                scaling,
                grain_lut,
                bh,
                row_num,
                luma,
                luma_stride,
                uv,
                is_id,
                $sx,
                $sy,
                bd,
            );
        }
    };
}

fguv_ss_fn!(
    fguv_32x32xn_420,
    1,
    1,
    "`fguv_32x32xn_420_c` (`src/filmgrain_tmpl.c:404-409`, `411`)."
);
fguv_ss_fn!(
    fguv_32x32xn_422,
    1,
    0,
    "`fguv_32x32xn_422_c` (`src/filmgrain_tmpl.c:404-409`, `412`)."
);
fguv_ss_fn!(
    fguv_32x32xn_444,
    0,
    0,
    "`fguv_32x32xn_444_c` (`src/filmgrain_tmpl.c:404-409`, `413`)."
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::fg::EMPTY_LUT;

    /// LFSR: dav1d bit sırası (`r ^ r>>1 ^ r>>3 ^ r>>12`), en üst `bits` bit.
    #[test]
    fn lfsr_adimi() {
        let mut s = 0x0001u32;
        // bit = 1 → state = 0x8000; 11 bit → 0x8000 >> 5 = 0x400.
        assert_eq!(get_random_number(11, &mut s), 0x400);
        assert_eq!(s, 0x8000);
        let mut z = 0u32;
        assert_eq!(get_random_number(8, &mut z), 0);
        assert_eq!(z, 0);
    }

    /// Grain değerleri bit derinliği aralığında kalır.
    #[test]
    fn grain_araligi() {
        let mut data = FilmGrainData {
            seed: 0x1234,
            num_y_points: 1,
            ar_coeff_lag: 3,
            ar_coeff_shift: 6,
            scaling_shift: 8,
            ..FilmGrainData::default()
        };
        data.ar_coeffs_y = [127; 24];
        data.ar_coeffs_uv = [[-128; 28]; 2];
        for bd in FgBitDepth::ALL {
            let lim = 128i16 << bd.min_8();
            let mut y = EMPTY_LUT;
            let dsp = crate::dsp::fg::FgDsp::scalar_for(bd);
            (dsp.generate_grain_y)(&mut y, &data, bd);
            assert!(y.iter().flatten().all(|&v| (-lim..lim).contains(&v)));
            let mut u = EMPTY_LUT;
            (dsp.generate_grain_uv[0])(&mut u, &y, &data, 0, bd);
            assert!(u.iter().flatten().all(|&v| (-lim..lim).contains(&v)));
        }
    }
}
