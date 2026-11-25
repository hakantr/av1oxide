//! ITU-T H.273 / CICP renk uzayı ve kroma alt örnekleme dönüşümleri.

#![forbid(unsafe_code)]

use crate::avif::image::{AvifImage, Rgb16Image, RgbImage, Rgba16Image, RgbaImage};
use crate::error::{Error, Result, SizeStage};
use crate::frame::{BitDepth, ChromaFormat};

/// `color_primaries` kod noktasına karşılık gelen CIE 1931 `(rX, rY, gX, gY, bX, bY, wX, wY)`
/// birincil renk ve beyaz nokta koordinatlarını döndürür (`ITU-T H.273` Tablo 2).
#[inline]
fn color_primaries_cie_xy(color_primaries: u16) -> [f32; 8] {
    match color_primaries {
        1 => [0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290], // BT.709
        4 => [0.67, 0.33, 0.21, 0.71, 0.14, 0.08, 0.310, 0.316],   // BT.470M
        5 => [0.64, 0.33, 0.29, 0.60, 0.15, 0.06, 0.3127, 0.3290], // BT.470BG
        6 | 7 => [0.630, 0.340, 0.310, 0.595, 0.155, 0.070, 0.3127, 0.3290], // BT.601 / SMPTE 240M
        8 => [0.681, 0.319, 0.243, 0.692, 0.145, 0.049, 0.310, 0.316], // Generic film
        9 => [0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290], // BT.2020
        10 => [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3333, 0.3333],      // SMPTE ST 428-1 (XYZ)
        11 => [0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.314, 0.351], // SMPTE RP 431-2 (DCI P3)
        12 => [0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3127, 0.3290], // SMPTE EG 432-1 (Display P3)
        22 => [0.630, 0.340, 0.295, 0.605, 0.155, 0.077, 0.3127, 0.3290], // EBU Tech. 3213-E
        _ => [0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290],        // Varsayılan (BT.709)
    }
}

/// Birincil renk koordinatlarından parlaklık katsayılarını `(Kr, Kb)` türetir (`ITU-T H.273` Denkl. 32–37).
#[inline]
fn chroma_derived_kr_kb(color_primaries: u16) -> (f32, f32) {
    let [rx, ry, gx, gy, bx, by, wx, wy] = color_primaries_cie_xy(color_primaries);
    let rz = 1.0f32 - (rx + ry);
    let gz = 1.0f32 - (gx + gy);
    let bz = 1.0f32 - (bx + by);
    let wz = 1.0f32 - (wx + wy);
    let denom =
        wy * (rx * (gy * bz - by * gz) + gx * (by * rz - ry * bz) + bx * (ry * gz - gy * rz));
    let kr = (ry
        * (wx * (gy * bz - by * gz) + wy * (bx * gz - gx * bz) + wz * (gx * by - bx * gy)))
        / denom;
    let kb = (by
        * (wx * (ry * gz - gy * rz) + wy * (gx * rz - rx * gz) + wz * (rx * gy - gx * ry)))
        / denom;
    (kr, kb)
}

/// Doğrusal YUV → RGB dönüşümünde desteklenmeyen veya ayrılmış `ITU-T H.273` `matrix_coefficients`
/// kod noktalarını denetler.
#[inline]
fn validate_matrix_coefficients(matrix_coefficients: u16) -> Result<()> {
    match matrix_coefficients {
        0 | 1 | 2 | 4..=9 | 12 | 16 | 17 => Ok(()),
        _ => Err(Error::Unsupported(
            "doğrusal olmayan veya ayrılmış H.273 matrix_coefficients değeri doğrusal RGB dönüşümünde desteklenmiyor",
        )),
    }
}

/// `matrix_coefficients` değerine karşılık gelen `(Kr, Kb)` katsayılarını döndürür (`ITU-T H.273`).
#[inline]
fn matrix_kr_kb(matrix_coefficients: u16, color_primaries: u16) -> (f32, f32) {
    match matrix_coefficients {
        1 => (0.2126, 0.0722),   // BT.709
        4 => (0.30, 0.11),       // FCC
        5 | 6 => (0.299, 0.114), // BT.470BG / BT.601
        7 => (0.212, 0.087),     // SMPTE ST 240
        9 => (0.2627, 0.0593),   // BT.2020 NCL
        12 => chroma_derived_kr_kb(color_primaries),
        _ => (0.299, 0.114), // Varsayılan (2 = Unspecified -> MIAF §7.3.6.4 BT.601)
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
    validate_matrix_coefficients(img.color.matrix_coefficients)?;
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
    validate_matrix_coefficients(img.color.matrix_coefficients)?;
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
    validate_matrix_coefficients(img.color.matrix_coefficients)?;
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
    validate_matrix_coefficients(img.color.matrix_coefficients)?;
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

#[cfg(test)]
mod tests {
    use super::{chroma_derived_kr_kb, validate_matrix_coefficients};
    use crate::error::Error;

    #[test]
    fn h273_chroma_derived_kr_kb_matches_cie_primaries() {
        // BT.709 (1): (0.2126, 0.0722)
        let (kr709, kb709) = chroma_derived_kr_kb(1);
        assert!((kr709 - 0.212639).abs() < 1e-4);
        assert!((kb709 - 0.072192).abs() < 1e-4);

        // BT.2020 (9): (0.2627, 0.0593)
        let (kr2020, kb2020) = chroma_derived_kr_kb(9);
        assert!((kr2020 - 0.262700).abs() < 1e-4);
        assert!((kb2020 - 0.059302).abs() < 1e-4);

        // Display P3 / SMPTE EG 432-1 (12): (0.228975, 0.079287)
        let (kr_p3, kb_p3) = chroma_derived_kr_kb(12);
        assert!((kr_p3 - 0.228975).abs() < 1e-4);
        assert!((kb_p3 - 0.079287).abs() < 1e-4);
    }

    #[test]
    fn h273_nonlinear_and_reserved_matrices_return_unsupported() {
        for supported in [0u16, 1, 2, 4, 5, 6, 7, 8, 9, 12, 16, 17] {
            assert_eq!(validate_matrix_coefficients(supported), Ok(()));
        }
        for unsupported in [3u16, 10, 11, 13, 14, 15, 18, 255] {
            assert!(matches!(
                validate_matrix_coefficients(unsupported),
                Err(Error::Unsupported(_))
            ));
        }
    }
}
