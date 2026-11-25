//! ITU-T H.273 / CICP renk uzayı ve kroma alt örnekleme dönüşümleri.

#![forbid(unsafe_code)]

use crate::avif::image::{AvifImage, Rgb16Image, RgbImage, Rgba16Image, RgbaImage};
use crate::error::{Error, Result, SizeStage};
use crate::frame::{BitDepth, ChromaFormat};

/// `matrix_coefficients` değerine karşılık gelen `(Kr, Kb)` katsayılarını döndürür (`ITU-T H.273`).
#[inline]
fn matrix_kr_kb(matrix_coefficients: u16, color_primaries: u16) -> (f32, f32) {
    match matrix_coefficients {
        1 => (0.2126, 0.0722),      // BT.709
        4 => (0.30, 0.11),          // FCC
        5 | 6 => (0.299, 0.114),    // BT.470BG / BT.601
        7 => (0.212, 0.087),        // SMPTE ST 240
        9 | 10 => (0.2627, 0.0593), // BT.2020 NCL
        12 => match color_primaries {
            9 => (0.2627, 0.0593),
            _ => (0.2126, 0.0722),
        },
        _ => (0.299, 0.114), // Varsayılan (2 = Unspecified -> BT.601)
    }
}

/// Y ve UV örneklerini `[0.0, 1.0]` / `[-0.5, 0.5]` aralığına dönüştüren arama tablolarını oluşturur.
fn build_unorm_tables(
    bit_depth: BitDepth,
    full_range: bool,
    identity_mode: bool,
) -> (Vec<f32>, Vec<f32>) {
    let bits = bit_depth.bits();
    let count = 1usize << bits;
    let max_chan = ((1u32 << bits) - 1) as f32;
    let shift = bits - 8;

    let bias_y = if full_range {
        0.0f32
    } else {
        (16u32 << shift) as f32
    };
    let range_y = if full_range {
        max_chan
    } else {
        (219u32 << shift) as f32
    };

    let mut table_y = Vec::with_capacity(count);
    for cp in 0..count {
        table_y.push(((cp as f32) - bias_y) / range_y);
    }

    if identity_mode {
        let table_uv = table_y.clone();
        (table_y, table_uv)
    } else {
        let bias_uv = (1u32 << (bits - 1)) as f32;
        let range_uv = if full_range {
            max_chan
        } else {
            (224u32 << shift) as f32
        };
        let mut table_uv = Vec::with_capacity(count);
        for cp in 0..count {
            table_uv.push(((cp as f32) - bias_uv) / range_uv);
        }
        (table_y, table_uv)
    }
}

/// `4:2:0`, `4:2:2` veya `4:4:4` kroma düzlemlerinden `(i, j)` luma koordinatına karşılık gelen
/// iki doğrusal (`bilinear`) enterpolasyonlu `(Cb, Cr)` çiftini hesaplar (`libavif` `avifImageYUVAnyToRGBAnySlow` uyumlu).
#[inline]
#[allow(clippy::too_many_arguments)]
fn sample_cb_cr_bilinear(
    u_plane: &[u16],
    v_plane: &[u16],
    table_uv: &[f32],
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    i: usize,
    j: usize,
    chroma_format: ChromaFormat,
    max_val: usize,
) -> (f32, f32) {
    if cw == 0 || ch == 0 {
        return (0.0, 0.0);
    }
    let shift_x = usize::from(matches!(
        chroma_format,
        ChromaFormat::C420 | ChromaFormat::C422
    ));
    let shift_y = usize::from(matches!(chroma_format, ChromaFormat::C420));
    let uv_i = (i >> shift_x).min(cw - 1);
    let uv_j = (j >> shift_y).min(ch - 1);

    if chroma_format == ChromaFormat::C444 {
        let idx = uv_j * cw + uv_i;
        let u0 = usize::from(u_plane[idx]).min(max_val);
        let v0 = usize::from(v_plane[idx]).min(max_val);
        return (table_uv[u0], table_uv[v0]);
    }

    let adj_col: isize = if i == 0 || (i == w - 1 && (i & 1) != 0) {
        0
    } else if (i & 1) != 0 {
        if uv_i + 1 < cw { 1 } else { 0 }
    } else if uv_i > 0 {
        -1
    } else {
        0
    };

    let adj_row: isize =
        if j == 0 || (j == h - 1 && (j & 1) != 0) || chroma_format == ChromaFormat::C422 {
            0
        } else if (j & 1) != 0 {
            if uv_j + 1 < ch { 1 } else { 0 }
        } else if uv_j > 0 {
            -1
        } else {
            0
        };

    let c0 = uv_i;
    let c1 = ((uv_i as isize) + adj_col) as usize;
    let r0 = uv_j;
    let r1 = ((uv_j as isize) + adj_row) as usize;

    let i00 = r0 * cw + c0;
    let i10 = r0 * cw + c1;
    let i01 = r1 * cw + c0;
    let i11 = r1 * cw + c1;

    let u00 = usize::from(u_plane[i00]).min(max_val);
    let u10 = usize::from(u_plane[i10]).min(max_val);
    let u01 = usize::from(u_plane[i01]).min(max_val);
    let u11 = usize::from(u_plane[i11]).min(max_val);

    let v00 = usize::from(v_plane[i00]).min(max_val);
    let v10 = usize::from(v_plane[i10]).min(max_val);
    let v01 = usize::from(v_plane[i01]).min(max_val);
    let v11 = usize::from(v_plane[i11]).min(max_val);

    let cb = (table_uv[u00] * (9.0 / 16.0))
        + (table_uv[u10] * (3.0 / 16.0))
        + (table_uv[u01] * (3.0 / 16.0))
        + (table_uv[u11] * (1.0 / 16.0));
    let cr = (table_uv[v00] * (9.0 / 16.0))
        + (table_uv[v10] * (3.0 / 16.0))
        + (table_uv[v01] * (3.0 / 16.0))
        + (table_uv[v11] * (1.0 / 16.0));
    (cb, cr)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn convert_pixel_f32(
    unorm_y: u16,
    y_norm: f32,
    cb: f32,
    cr: f32,
    monochrome: bool,
    matrix_coefficients: u16,
    kr: f32,
    kg: f32,
    kb: f32,
    yuv_max_f: f32,
    rgb_max_f: f32,
) -> (f32, f32, f32) {
    if monochrome {
        let g = y_norm.clamp(0.0, 1.0);
        return (g, g, g);
    }
    let (r, g, b) = if matrix_coefficients == 0 {
        // Identity (GBR)
        (cr, y_norm, cb)
    } else if matrix_coefficients == 8 {
        // YCgCo
        let t = y_norm - cb;
        (t + cr, y_norm + cb, t - cr)
    } else if matrix_coefficients == 16 || matrix_coefficients == 17 {
        // YCgCo-Re / YCgCo-Ro
        let yy = i32::from(unorm_y);
        let cg = (cb * yuv_max_f).round() as i32;
        let co = (cr * yuv_max_f).round() as i32;
        let t = yy - (cg >> 1);
        let max_rgb = rgb_max_f as i32;
        let g_i = (t + cg).clamp(0, max_rgb);
        let b_i = (t - (co >> 1)).clamp(0, max_rgb);
        let r_i = (b_i + co).clamp(0, max_rgb);
        (
            (r_i as f32) / rgb_max_f,
            (g_i as f32) / rgb_max_f,
            (b_i as f32) / rgb_max_f,
        )
    } else {
        let r = y_norm + (2.0 * (1.0 - kr)) * cr;
        let b = y_norm + (2.0 * (1.0 - kb)) * cb;
        let g = y_norm - ((2.0 * ((kr * (1.0 - kr) * cr) + (kb * (1.0 - kb) * cb))) / kg);
        (r, g, b)
    };
    (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

#[inline]
fn rescale_alpha_u8(src_alpha: u16, bit_depth: BitDepth) -> u8 {
    match bit_depth {
        BitDepth::Eight => (src_alpha & 0xFF) as u8,
        BitDepth::Ten | BitDepth::Twelve => {
            let src_max_f = (bit_depth.max_sample() as u16) as f32;
            let alpha_f = (src_alpha.min(bit_depth.max_sample() as u16) as f32) / src_max_f;
            ((0.5f32 + (alpha_f * 255.0f32)) as i32).clamp(0, 255) as u8
        }
    }
}

/// [`AvifImage`] düzlemlerini 16-bit RGB tamponuna dönüştürür.
pub(crate) fn image_to_rgb16(img: &AvifImage) -> Result<Rgb16Image> {
    let w = img.width as usize;
    let h = img.height as usize;
    let cw = img.chroma_width as usize;
    let ch = img.chroma_height as usize;
    let num_pixels = w.checked_mul(h).ok_or(Error::SizeOverflow {
        stage: SizeStage::PlaneArea,
        lhs: w,
        rhs: h,
    })?;
    let total_samples = num_pixels.checked_mul(3).ok_or(Error::SizeOverflow {
        stage: SizeStage::TotalAllocation,
        lhs: num_pixels,
        rhs: 3,
    })?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(total_samples)
        .map_err(|_| Error::AllocationFailed {
            requested: total_samples * core::mem::size_of::<u16>(),
        })?;

    let monochrome = img.chroma_format == ChromaFormat::Monochrome;
    let mtrx = img.color.matrix_coefficients;
    let (kr, kb) = matrix_kr_kb(mtrx, img.color.color_primaries);
    let kg = 1.0f32 - kr - kb;
    let (table_y, table_uv) = build_unorm_tables(img.bit_depth, img.color.full_range, mtrx == 0);
    let max_val = img.bit_depth.max_sample() as usize;
    let max_f = max_val as f32;

    for j in 0..h {
        let y_row_off = j * w;
        for i in 0..w {
            let unorm_y = img.y_plane[y_row_off + i].min(max_val as u16);
            let y_norm = table_y[usize::from(unorm_y)];
            let (cb, cr) = if monochrome {
                (0.0, 0.0)
            } else {
                sample_cb_cr_bilinear(
                    &img.u_plane,
                    &img.v_plane,
                    &table_uv,
                    w,
                    h,
                    cw,
                    ch,
                    i,
                    j,
                    img.chroma_format,
                    max_val,
                )
            };
            let (rc, gc, bc) = convert_pixel_f32(
                unorm_y, y_norm, cb, cr, monochrome, mtrx, kr, kg, kb, max_f, max_f,
            );
            pixels.push((0.5f32 + (rc * max_f)) as u16);
            pixels.push((0.5f32 + (gc * max_f)) as u16);
            pixels.push((0.5f32 + (bc * max_f)) as u16);
        }
    }

    Ok(Rgb16Image {
        width: img.width,
        height: img.height,
        bit_depth: img.bit_depth,
        pixels,
    })
}

/// [`AvifImage`] düzlemlerini 16-bit RGBA tamponuna dönüştürür.
pub(crate) fn image_to_rgba16(img: &AvifImage) -> Result<Rgba16Image> {
    let w = img.width as usize;
    let h = img.height as usize;
    let cw = img.chroma_width as usize;
    let ch = img.chroma_height as usize;
    let num_pixels = w.checked_mul(h).ok_or(Error::SizeOverflow {
        stage: SizeStage::PlaneArea,
        lhs: w,
        rhs: h,
    })?;
    let total_samples = num_pixels.checked_mul(4).ok_or(Error::SizeOverflow {
        stage: SizeStage::TotalAllocation,
        lhs: num_pixels,
        rhs: 4,
    })?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(total_samples)
        .map_err(|_| Error::AllocationFailed {
            requested: total_samples * core::mem::size_of::<u16>(),
        })?;

    let monochrome = img.chroma_format == ChromaFormat::Monochrome;
    let mtrx = img.color.matrix_coefficients;
    let (kr, kb) = matrix_kr_kb(mtrx, img.color.color_primaries);
    let kg = 1.0f32 - kr - kb;
    let (table_y, table_uv) = build_unorm_tables(img.bit_depth, img.color.full_range, mtrx == 0);
    let max_val = img.bit_depth.max_sample() as usize;
    let max_f = max_val as f32;
    let opaque_alpha = max_val as u16;

    for j in 0..h {
        let y_row_off = j * w;
        for i in 0..w {
            let idx = y_row_off + i;
            let unorm_y = img.y_plane[idx].min(opaque_alpha);
            let y_norm = table_y[usize::from(unorm_y)];
            let (cb, cr) = if monochrome {
                (0.0, 0.0)
            } else {
                sample_cb_cr_bilinear(
                    &img.u_plane,
                    &img.v_plane,
                    &table_uv,
                    w,
                    h,
                    cw,
                    ch,
                    i,
                    j,
                    img.chroma_format,
                    max_val,
                )
            };
            let (rc, gc, bc) = convert_pixel_f32(
                unorm_y, y_norm, cb, cr, monochrome, mtrx, kr, kg, kb, max_f, max_f,
            );
            let a = img
                .alpha_plane
                .as_ref()
                .map_or(opaque_alpha, |ap| ap[idx].min(opaque_alpha));
            pixels.push((0.5f32 + (rc * max_f)) as u16);
            pixels.push((0.5f32 + (gc * max_f)) as u16);
            pixels.push((0.5f32 + (bc * max_f)) as u16);
            pixels.push(a);
        }
    }

    Ok(Rgba16Image {
        width: img.width,
        height: img.height,
        bit_depth: img.bit_depth,
        pixels,
    })
}

/// [`AvifImage`] düzlemlerini 8-bit RGB tamponuna dönüştürür.
pub(crate) fn image_to_rgb8(img: &AvifImage) -> Result<RgbImage> {
    let w = img.width as usize;
    let h = img.height as usize;
    let cw = img.chroma_width as usize;
    let ch = img.chroma_height as usize;
    let num_pixels = w.checked_mul(h).ok_or(Error::SizeOverflow {
        stage: SizeStage::PlaneArea,
        lhs: w,
        rhs: h,
    })?;
    let total_bytes = num_pixels.checked_mul(3).ok_or(Error::SizeOverflow {
        stage: SizeStage::TotalAllocation,
        lhs: num_pixels,
        rhs: 3,
    })?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(total_bytes)
        .map_err(|_| Error::AllocationFailed {
            requested: total_bytes,
        })?;

    let monochrome = img.chroma_format == ChromaFormat::Monochrome;
    let mtrx = img.color.matrix_coefficients;
    let (kr, kb) = matrix_kr_kb(mtrx, img.color.color_primaries);
    let kg = 1.0f32 - kr - kb;
    let (table_y, table_uv) = build_unorm_tables(img.bit_depth, img.color.full_range, mtrx == 0);
    let max_val = img.bit_depth.max_sample() as usize;
    let yuv_max_f = max_val as f32;

    for j in 0..h {
        let y_row_off = j * w;
        for i in 0..w {
            let unorm_y = img.y_plane[y_row_off + i].min(max_val as u16);
            let y_norm = table_y[usize::from(unorm_y)];
            let (cb, cr) = if monochrome {
                (0.0, 0.0)
            } else {
                sample_cb_cr_bilinear(
                    &img.u_plane,
                    &img.v_plane,
                    &table_uv,
                    w,
                    h,
                    cw,
                    ch,
                    i,
                    j,
                    img.chroma_format,
                    max_val,
                )
            };
            let (rc, gc, bc) = convert_pixel_f32(
                unorm_y, y_norm, cb, cr, monochrome, mtrx, kr, kg, kb, yuv_max_f, 255.0,
            );
            pixels.push((0.5f32 + (rc * 255.0f32)) as u8);
            pixels.push((0.5f32 + (gc * 255.0f32)) as u8);
            pixels.push((0.5f32 + (bc * 255.0f32)) as u8);
        }
    }

    Ok(RgbImage {
        width: img.width,
        height: img.height,
        pixels,
    })
}

/// [`AvifImage`] düzlemlerini 8-bit RGBA tamponuna dönüştürür.
pub(crate) fn image_to_rgba8(img: &AvifImage) -> Result<RgbaImage> {
    let w = img.width as usize;
    let h = img.height as usize;
    let cw = img.chroma_width as usize;
    let ch = img.chroma_height as usize;
    let num_pixels = w.checked_mul(h).ok_or(Error::SizeOverflow {
        stage: SizeStage::PlaneArea,
        lhs: w,
        rhs: h,
    })?;
    let total_bytes = num_pixels.checked_mul(4).ok_or(Error::SizeOverflow {
        stage: SizeStage::TotalAllocation,
        lhs: num_pixels,
        rhs: 4,
    })?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(total_bytes)
        .map_err(|_| Error::AllocationFailed {
            requested: total_bytes,
        })?;

    let monochrome = img.chroma_format == ChromaFormat::Monochrome;
    let mtrx = img.color.matrix_coefficients;
    let (kr, kb) = matrix_kr_kb(mtrx, img.color.color_primaries);
    let kg = 1.0f32 - kr - kb;
    let (table_y, table_uv) = build_unorm_tables(img.bit_depth, img.color.full_range, mtrx == 0);
    let max_val = img.bit_depth.max_sample() as usize;
    let yuv_max_f = max_val as f32;
    let opaque_alpha = max_val as u16;

    for j in 0..h {
        let y_row_off = j * w;
        for i in 0..w {
            let idx = y_row_off + i;
            let unorm_y = img.y_plane[idx].min(opaque_alpha);
            let y_norm = table_y[usize::from(unorm_y)];
            let (cb, cr) = if monochrome {
                (0.0, 0.0)
            } else {
                sample_cb_cr_bilinear(
                    &img.u_plane,
                    &img.v_plane,
                    &table_uv,
                    w,
                    h,
                    cw,
                    ch,
                    i,
                    j,
                    img.chroma_format,
                    max_val,
                )
            };
            let (rc, gc, bc) = convert_pixel_f32(
                unorm_y, y_norm, cb, cr, monochrome, mtrx, kr, kg, kb, yuv_max_f, 255.0,
            );
            let a16 = img
                .alpha_plane
                .as_ref()
                .map_or(opaque_alpha, |ap| ap[idx].min(opaque_alpha));
            pixels.push((0.5f32 + (rc * 255.0f32)) as u8);
            pixels.push((0.5f32 + (gc * 255.0f32)) as u8);
            pixels.push((0.5f32 + (bc * 255.0f32)) as u8);
            pixels.push(rescale_alpha_u8(a16, img.bit_depth));
        }
    }

    Ok(RgbaImage {
        width: img.width,
        height: img.height,
        pixels,
    })
}
