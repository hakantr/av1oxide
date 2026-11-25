//! ITU-T H.273 / CICP renk uzayı ve kroma alt örnekleme dönüşümleri.

#![forbid(unsafe_code)]

use crate::avif::image::{AvifImage, Rgb16Image, RgbImage, Rgba16Image, RgbaImage};
use crate::error::{Error, Result, SizeStage};
use crate::frame::{BitDepth, ChromaFormat};

/// `4:2:0` veya `4:2:2` kroma düzleminden `(x, y)` luma koordinatına karşılık gelen
/// iki doğrusal (`bilinear`) enterpolasyonlu kroma örneğini okur.
#[inline]
fn sample_chroma_bilinear(
    plane: &[u16],
    cw: usize,
    ch: usize,
    x: usize,
    y: usize,
    sub_x: bool,
    sub_y: bool,
) -> i32 {
    if cw == 0 || ch == 0 {
        return 0;
    }
    if !sub_x && !sub_y {
        let cx = x.min(cw - 1);
        let cy = y.min(ch - 1);
        return i32::from(plane[cy * cw + cx]);
    }

    // Merkez hizalı 2x alt örnekleme için (2*x - 1) / 4 ağırlıklı komşuluk.
    let (cx0, cx1, wx) = if sub_x {
        if x == 0 {
            (0, 0, 0)
        } else {
            let base = (x - 1) / 2;
            let c0 = base.min(cw - 1);
            let c1 = (base + 1).min(cw - 1);
            let w = if ((x - 1) & 1) == 0 { 1 } else { 3 };
            (c0, c1, w)
        }
    } else {
        let c = x.min(cw - 1);
        (c, c, 0)
    };

    let (cy0, cy1, wy) = if sub_y {
        if y == 0 {
            (0, 0, 0)
        } else {
            let base = (y - 1) / 2;
            let c0 = base.min(ch - 1);
            let c1 = (base + 1).min(ch - 1);
            let w = if ((y - 1) & 1) == 0 { 1 } else { 3 };
            (c0, c1, w)
        }
    } else {
        let c = y.min(ch - 1);
        (c, c, 0)
    };

    let p00 = i32::from(plane[cy0 * cw + cx0]);
    let p10 = i32::from(plane[cy0 * cw + cx1]);
    let p01 = i32::from(plane[cy1 * cw + cx0]);
    let p11 = i32::from(plane[cy1 * cw + cx1]);

    let top = p00 * (4 - wx) + p10 * wx;
    let bot = p01 * (4 - wx) + p11 * wx;
    (top * (4 - wy) + bot * wy + 8) >> 4
}

/// Tek bir `(Y, U, V)` örneğini H.273 `matrix_coefficients` ve `full_range` değerine göre
/// `[0, max_val]` aralığında `(R, G, B)` üçlüsüne dönüştürür.
#[inline]
fn yuv_to_rgb_sample(
    y_val: i32,
    u_val: i32,
    v_val: i32,
    bit_depth: BitDepth,
    matrix_coefficients: u16,
    full_range: bool,
    monochrome: bool,
) -> (u16, u16, u16) {
    let bits = bit_depth.bits();
    let max_val = (1i32 << bits) - 1;
    let shift = bits - 8;
    let uv_bias = 128i32 << shift;

    if monochrome {
        let luma = if full_range {
            y_val.clamp(0, max_val)
        } else {
            let y_min = 16i32 << shift;
            let y_span = 219i32 << shift;
            (((y_val - y_min) * max_val + (y_span / 2)) / y_span).clamp(0, max_val)
        } as u16;
        return (luma, luma, luma);
    }

    // matrix_coefficients == 0: Identity (GBR)
    // H.273 §8.3: G = Y, B = Cb (U), R = Cr (V).
    if matrix_coefficients == 0 {
        if full_range {
            let g = y_val.clamp(0, max_val) as u16;
            let b = u_val.clamp(0, max_val) as u16;
            let r = v_val.clamp(0, max_val) as u16;
            return (r, g, b);
        }
        let y_min = 16i32 << shift;
        let y_span = 219i32 << shift;
        let g = (((y_val - y_min) * max_val + (y_span / 2)) / y_span).clamp(0, max_val) as u16;
        let b = (((u_val - y_min) * max_val + (y_span / 2)) / y_span).clamp(0, max_val) as u16;
        let r = (((v_val - y_min) * max_val + (y_span / 2)) / y_span).clamp(0, max_val) as u16;
        return (r, g, b);
    }

    // matrix_coefficients == 8: YCgCo
    if matrix_coefficients == 8 {
        let (yy, cg, co) = if full_range {
            (y_val, u_val - uv_bias, v_val - uv_bias)
        } else {
            let y_min = 16i32 << shift;
            let y_span = 219i32 << shift;
            let c_span = 224i32 << shift;
            (
                ((y_val - y_min) * max_val + (y_span / 2)) / y_span,
                ((u_val - uv_bias) * max_val + (c_span / 2)) / c_span,
                ((v_val - uv_bias) * max_val + (c_span / 2)) / c_span,
            )
        };
        let t = yy - cg;
        let g = (yy + cg).clamp(0, max_val) as u16;
        let b = (t - co).clamp(0, max_val) as u16;
        let r = (t + co).clamp(0, max_val) as u16;
        return (r, g, b);
    }

    // H.273 Kr, Kb sabitleri (16384 ölçekli sabit nokta):
    // - BT.709 (1): Kr = 0.2126, Kb = 0.0722
    // - BT.601 (5, 6) ve varsayılan: Kr = 0.299, Kb = 0.114
    // - BT.2020 NCL (9, 10): Kr = 0.2627, Kb = 0.0593
    // R = Y + c_rv * Cr
    // G = Y - c_gu * Cb - c_gv * Cr
    // B = Y + c_bu * Cb
    let (c_rv, c_gu, c_gv, c_bu): (i64, i64, i64, i64) = match matrix_coefficients {
        1 => (25_802, 3_069, 7_669, 30_402),      // BT.709
        9 | 10 => (24_160, 2_696, 9_361, 30_828), // BT.2020
        _ => (22_970, 5_638, 11_700, 29_032),     // BT.601 (5, 6, 2)
    };

    let (y_norm, cb_norm, cr_norm): (i64, i64, i64) = if full_range {
        (
            i64::from(y_val) * 16_384,
            i64::from(u_val - uv_bias),
            i64::from(v_val - uv_bias),
        )
    } else {
        let y_min = 16i32 << shift;
        let y_span = i64::from(219i32 << shift);
        let c_span = i64::from(224i32 << shift);
        let max_i64 = i64::from(max_val);
        (
            (i64::from(y_val - y_min) * max_i64 * 16_384 + (y_span / 2)) / y_span,
            (i64::from(u_val - uv_bias) * max_i64 + (c_span / 2)) / c_span,
            (i64::from(v_val - uv_bias) * max_i64 + (c_span / 2)) / c_span,
        )
    };

    let r = ((y_norm + c_rv * cr_norm + 8_192) >> 14).clamp(0, i64::from(max_val)) as u16;
    let g = ((y_norm - c_gu * cb_norm - c_gv * cr_norm + 8_192) >> 14).clamp(0, i64::from(max_val))
        as u16;
    let b = ((y_norm + c_bu * cb_norm + 8_192) >> 14).clamp(0, i64::from(max_val)) as u16;
    (r, g, b)
}

#[inline]
fn sample_to_u8(val: u16, bit_depth: BitDepth) -> u8 {
    match bit_depth {
        BitDepth::Eight => (val & 0xFF) as u8,
        BitDepth::Ten => (((u32::from(val) * 255) + 511) / 1023).min(255) as u8,
        BitDepth::Twelve => (((u32::from(val) * 255) + 2047) / 4095).min(255) as u8,
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
    let sub_x = matches!(img.chroma_format, ChromaFormat::C420 | ChromaFormat::C422);
    let sub_y = matches!(img.chroma_format, ChromaFormat::C420);

    for y in 0..h {
        let y_row_off = y * w;
        for x in 0..w {
            let y_val = i32::from(img.y_plane[y_row_off + x]);
            let (u_val, v_val) = if monochrome {
                (0, 0)
            } else {
                (
                    sample_chroma_bilinear(&img.u_plane, cw, ch, x, y, sub_x, sub_y),
                    sample_chroma_bilinear(&img.v_plane, cw, ch, x, y, sub_x, sub_y),
                )
            };
            let (r, g, b) = yuv_to_rgb_sample(
                y_val,
                u_val,
                v_val,
                img.bit_depth,
                img.color.matrix_coefficients,
                img.color.full_range,
                monochrome,
            );
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
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
    let sub_x = matches!(img.chroma_format, ChromaFormat::C420 | ChromaFormat::C422);
    let sub_y = matches!(img.chroma_format, ChromaFormat::C420);
    let opaque_alpha = img.bit_depth.max_sample() as u16;

    for y in 0..h {
        let y_row_off = y * w;
        for x in 0..w {
            let idx = y_row_off + x;
            let y_val = i32::from(img.y_plane[idx]);
            let (u_val, v_val) = if monochrome {
                (0, 0)
            } else {
                (
                    sample_chroma_bilinear(&img.u_plane, cw, ch, x, y, sub_x, sub_y),
                    sample_chroma_bilinear(&img.v_plane, cw, ch, x, y, sub_x, sub_y),
                )
            };
            let (r, g, b) = yuv_to_rgb_sample(
                y_val,
                u_val,
                v_val,
                img.bit_depth,
                img.color.matrix_coefficients,
                img.color.full_range,
                monochrome,
            );
            let a = img
                .alpha_plane
                .as_ref()
                .map_or(opaque_alpha, |ap| ap[idx].min(opaque_alpha));
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
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
    let sub_x = matches!(img.chroma_format, ChromaFormat::C420 | ChromaFormat::C422);
    let sub_y = matches!(img.chroma_format, ChromaFormat::C420);

    for y in 0..h {
        let y_row_off = y * w;
        for x in 0..w {
            let y_val = i32::from(img.y_plane[y_row_off + x]);
            let (u_val, v_val) = if monochrome {
                (0, 0)
            } else {
                (
                    sample_chroma_bilinear(&img.u_plane, cw, ch, x, y, sub_x, sub_y),
                    sample_chroma_bilinear(&img.v_plane, cw, ch, x, y, sub_x, sub_y),
                )
            };
            let (r, g, b) = yuv_to_rgb_sample(
                y_val,
                u_val,
                v_val,
                img.bit_depth,
                img.color.matrix_coefficients,
                img.color.full_range,
                monochrome,
            );
            pixels.push(sample_to_u8(r, img.bit_depth));
            pixels.push(sample_to_u8(g, img.bit_depth));
            pixels.push(sample_to_u8(b, img.bit_depth));
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
    let sub_x = matches!(img.chroma_format, ChromaFormat::C420 | ChromaFormat::C422);
    let sub_y = matches!(img.chroma_format, ChromaFormat::C420);
    let opaque_alpha = img.bit_depth.max_sample() as u16;

    for y in 0..h {
        let y_row_off = y * w;
        for x in 0..w {
            let idx = y_row_off + x;
            let y_val = i32::from(img.y_plane[idx]);
            let (u_val, v_val) = if monochrome {
                (0, 0)
            } else {
                (
                    sample_chroma_bilinear(&img.u_plane, cw, ch, x, y, sub_x, sub_y),
                    sample_chroma_bilinear(&img.v_plane, cw, ch, x, y, sub_x, sub_y),
                )
            };
            let (r, g, b) = yuv_to_rgb_sample(
                y_val,
                u_val,
                v_val,
                img.bit_depth,
                img.color.matrix_coefficients,
                img.color.full_range,
                monochrome,
            );
            let a16 = img
                .alpha_plane
                .as_ref()
                .map_or(opaque_alpha, |ap| ap[idx].min(opaque_alpha));
            pixels.push(sample_to_u8(r, img.bit_depth));
            pixels.push(sample_to_u8(g, img.bit_depth));
            pixels.push(sample_to_u8(b, img.bit_depth));
            pixels.push(sample_to_u8(a16, img.bit_depth));
        }
    }

    Ok(RgbaImage {
        width: img.width,
        height: img.height,
        pixels,
    })
}
