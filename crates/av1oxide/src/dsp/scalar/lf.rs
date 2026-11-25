//! Deblocking (loop filter) skaler gövdeleri — dav1d `src/loopfilter_tmpl.c`.
//!
//! Sözleşme: [`crate::dsp::lf`]. `loop_filter` dav1d `loop_filter`'ın
//! (`loopfilter_tmpl.c:37-163`) birebir karşılığıdır; dört SB girişi
//! (`loop_filter_sb[düzlem][yön]`) dav1d `loop_filter_{h,v}_sb128{y,uv}_c`
//! (`loopfilter_tmpl.c:165-255`) gibi bir düzlem/yön kenar kümesini işler:
//! kenar maskesi, seviye geri düşüşü (seviye 0 ise soldaki/üstteki 4×4) ve
//! filtre boyu seçimi girişin içindedir. Kenar sırası çözme hattındaki eski
//! döngüyle aynıdır (F3c taşıması, algoritma değişmedi).
//!
//! Bütün girişler `BPC` (8/10/12) ile monomorfiktir.

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use crate::dsp::lf::LimLut;

/// `loop_filter` (`loopfilter_tmpl.c:37-163`).
///
/// `BPC` (8/10/12) ile monomorfiktir (F3c, DOG-05): `bitdepth_min_8`, piksel
/// genişliği ve en büyük değer derleme zamanı sabitidir; döngüde bit derinliği
/// dalı yoktur.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn loop_filter<const BPC: u32>(
    data: &mut [u8],
    mut off: isize,
    stride_a: isize,
    stride_b: isize,
    wd: usize,
    e: i32,
    i: i32,
    h: i32,
) {
    let bitdepth_min_8 = BPC - 8;
    let max_px = (1i32 << BPC) - 1;
    let f = 1i32 << bitdepth_min_8;
    let e = e << bitdepth_min_8;
    let i = i << bitdepth_min_8;
    let h = h << bitdepth_min_8;
    let bpp: isize = if BPC == 8 { 1 } else { 2 };

    let get = |d: &[u8], o: isize| -> i32 {
        let o = o as usize;
        if bpp == 2 {
            i32::from(u16::from_le_bytes([d[o], d[o + 1]]))
        } else {
            i32::from(d[o])
        }
    };
    let put = |d: &mut [u8], o: isize, v: i32| {
        let v = v.clamp(0, max_px);
        let o = o as usize;
        if bpp == 2 {
            d[o..o + 2].copy_from_slice(&(v as u16).to_le_bytes());
        } else {
            d[o] = v as u8;
        }
    };

    for _ in 0..4 {
        let p1 = get(data, off - 2 * stride_b);
        let p0 = get(data, off - stride_b);
        let q0 = get(data, off);
        let q1 = get(data, off + stride_b);
        let mut p2 = 0;
        let mut p3 = 0;
        let mut q2 = 0;
        let mut q3 = 0;
        let mut fm = (p1 - p0).abs() <= i
            && (q1 - q0).abs() <= i
            && (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1) <= e;
        if wd > 4 {
            p2 = get(data, off - 3 * stride_b);
            q2 = get(data, off + 2 * stride_b);
            fm &= (p2 - p1).abs() <= i && (q2 - q1).abs() <= i;
            if wd > 6 {
                p3 = get(data, off - 4 * stride_b);
                q3 = get(data, off + 3 * stride_b);
                fm &= (p3 - p2).abs() <= i && (q3 - q2).abs() <= i;
            }
        }
        if !fm {
            off += stride_a;
            continue;
        }
        let mut flat8in = true;
        let mut flat8out = true;
        let mut p4 = 0;
        let mut p5 = 0;
        let mut p6 = 0;
        let mut q4 = 0;
        let mut q5 = 0;
        let mut q6 = 0;
        if wd >= 16 {
            p6 = get(data, off - 7 * stride_b);
            p5 = get(data, off - 6 * stride_b);
            p4 = get(data, off - 5 * stride_b);
            q4 = get(data, off + 4 * stride_b);
            q5 = get(data, off + 5 * stride_b);
            q6 = get(data, off + 6 * stride_b);
            flat8out = (p6 - p0).abs() <= f
                && (p5 - p0).abs() <= f
                && (p4 - p0).abs() <= f
                && (q4 - q0).abs() <= f
                && (q5 - q0).abs() <= f
                && (q6 - q0).abs() <= f;
        }
        if wd >= 6 {
            flat8in = (p2 - p0).abs() <= f
                && (p1 - p0).abs() <= f
                && (q1 - q0).abs() <= f
                && (q2 - q0).abs() <= f;
        }
        if wd >= 8 {
            flat8in &= (p3 - p0).abs() <= f && (q3 - q0).abs() <= f;
        }
        if wd >= 16 && flat8out && flat8in {
            let v = |x: i32| x;
            put(
                data,
                off - 6 * stride_b,
                (p6 * 7 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + 8) >> 4,
            );
            let _ = v;
            put(
                data,
                off - 5 * stride_b,
                (p6 * 5 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + 8) >> 4,
            );
            put(
                data,
                off - 4 * stride_b,
                (p6 * 4 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + 8) >> 4,
            );
            put(
                data,
                off - 3 * stride_b,
                (p6 * 3 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + 8) >> 4,
            );
            put(
                data,
                off - 2 * stride_b,
                (p6 * 2 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + 8)
                    >> 4,
            );
            put(
                data,
                off - stride_b,
                (p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + 8)
                    >> 4,
            );
            put(
                data,
                off,
                (p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + 8)
                    >> 4,
            );
            put(
                data,
                off + stride_b,
                (p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 * 2 + q2 * 2 + q3 + q4 + q5 + q6 * 2 + 8)
                    >> 4,
            );
            put(
                data,
                off + 2 * stride_b,
                (p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 * 2 + q3 * 2 + q4 + q5 + q6 * 3 + 8) >> 4,
            );
            put(
                data,
                off + 3 * stride_b,
                (p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 * 2 + q4 * 2 + q5 + q6 * 4 + 8) >> 4,
            );
            put(
                data,
                off + 4 * stride_b,
                (p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 * 2 + q5 * 2 + q6 * 5 + 8) >> 4,
            );
            put(
                data,
                off + 5 * stride_b,
                (p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 * 2 + q6 * 7 + 8) >> 4,
            );
        } else if wd >= 8 && flat8in {
            put(
                data,
                off - 3 * stride_b,
                (p3 * 3 + p2 * 2 + p1 + p0 + q0 + 4) >> 3,
            );
            put(
                data,
                off - 2 * stride_b,
                (p3 * 2 + p2 + p1 * 2 + p0 + q0 + q1 + 4) >> 3,
            );
            put(
                data,
                off - stride_b,
                (p3 + p2 + p1 + p0 * 2 + q0 + q1 + q2 + 4) >> 3,
            );
            put(data, off, (p2 + p1 + p0 + q0 * 2 + q1 + q2 + q3 + 4) >> 3);
            put(
                data,
                off + stride_b,
                (p1 + p0 + q0 + q1 * 2 + q2 + q3 * 2 + 4) >> 3,
            );
            put(
                data,
                off + 2 * stride_b,
                (p0 + q0 + q1 + q2 * 2 + q3 * 3 + 4) >> 3,
            );
        } else if wd == 6 && flat8in {
            put(
                data,
                off - 2 * stride_b,
                (p2 * 3 + p1 * 2 + p0 * 2 + q0 + 4) >> 3,
            );
            put(
                data,
                off - stride_b,
                (p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + 4) >> 3,
            );
            put(data, off, (p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + 4) >> 3);
            put(
                data,
                off + stride_b,
                (p0 + q0 * 2 + q1 * 2 + q2 * 3 + 4) >> 3,
            );
        } else {
            let hev = (p1 - p0).abs() > h || (q1 - q0).abs() > h;
            let iclip_diff = |v: i32| {
                v.clamp(
                    -128 * (1 << bitdepth_min_8),
                    128 * (1 << bitdepth_min_8) - 1,
                )
            };
            if hev {
                let mut fv = iclip_diff(p1 - q1);
                fv = iclip_diff(3 * (q0 - p0) + fv);
                let f1 = (fv + 4).min((128 << bitdepth_min_8) - 1) >> 3;
                let f2 = (fv + 3).min((128 << bitdepth_min_8) - 1) >> 3;
                put(data, off - stride_b, p0 + f2);
                put(data, off, q0 - f1);
            } else {
                let fv = iclip_diff(3 * (q0 - p0));
                let f1 = (fv + 4).min((128 << bitdepth_min_8) - 1) >> 3;
                let f2 = (fv + 3).min((128 << bitdepth_min_8) - 1) >> 3;
                put(data, off - stride_b, p0 + f2);
                put(data, off, q0 - f1);
                let fv = (f1 + 1) >> 1;
                put(data, off - 2 * stride_b, p1 + fv);
                put(data, off + stride_b, q1 - fv);
            }
        }
        off += stride_a;
    }
}

/// `level` hücresinin süzme seviyesi; 0 ise `fallback` hücresi (dav1d
/// `loopfilter_tmpl.c:186-188`, `221-223`).
#[inline(always)]
fn level_or(level: &[[u8; 4]], at: usize, fallback: usize, idx: usize) -> u32 {
    let l = level[at][idx];
    u32::from(if l != 0 { l } else { level[fallback][idx] })
}

/// Bir kenarın filtresi: `wd` filtre boyu, `l` seviye.
#[inline(always)]
fn edge<const BPC: u32>(
    dst: &mut [u8],
    off: usize,
    stride_a: isize,
    stride_b: isize,
    wd: usize,
    l: u32,
    lut: &LimLut,
) {
    loop_filter::<BPC>(
        dst,
        off as isize,
        stride_a,
        stride_b,
        wd,
        lut.e[l as usize],
        lut.i[l as usize],
        (l >> 4) as i32,
    );
}

/// `loop_filter_sb[0][0]` — luma dikey kenarlar (yatay süzme;
/// dav1d `loop_filter_h_sb128y_c`). Sütun `1..w4`, satır `y0..y0+count`.
pub fn lpf_h_sb_y<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    mask: &[u8],
    mask_stride: usize,
    level: &[[u8; 4]],
    lvl_stride: usize,
    lvl_idx: usize,
    lut: &LimLut,
    w4: usize,
    y0: usize,
    count: usize,
) {
    let bpp = if BPC == 8 { 1 } else { 2 };
    for gx in 1..w4 {
        for gy in y0..y0 + count {
            let v = mask[gy * mask_stride + gx];
            if v == 0 {
                continue;
            }
            let l = level_or(
                level,
                gy * lvl_stride + gx,
                gy * lvl_stride + gx - 1,
                lvl_idx,
            );
            if l == 0 {
                continue;
            }
            let off = gy * 4 * stride + gx * 4 * bpp;
            let wd = 4 << (v - 1);
            edge::<BPC>(dst, off, stride as isize, bpp as isize, wd, l, lut);
        }
    }
}

/// `loop_filter_sb[0][1]` — luma yatay kenarlar (dikey süzme;
/// dav1d `loop_filter_v_sb128y_c`). Satır `y0..y0+count` (0. satır hariç).
pub fn lpf_v_sb_y<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    mask: &[u8],
    mask_stride: usize,
    level: &[[u8; 4]],
    lvl_stride: usize,
    lvl_idx: usize,
    lut: &LimLut,
    w4: usize,
    y0: usize,
    count: usize,
) {
    let bpp = if BPC == 8 { 1 } else { 2 };
    for gy in y0..y0 + count {
        if gy == 0 {
            continue;
        }
        for gx in 0..w4 {
            let v = mask[gy * mask_stride + gx];
            if v == 0 {
                continue;
            }
            let l = level_or(
                level,
                gy * lvl_stride + gx,
                (gy - 1) * lvl_stride + gx,
                lvl_idx,
            );
            if l == 0 {
                continue;
            }
            let off = gy * 4 * stride + gx * 4 * bpp;
            let wd = 4 << (v - 1);
            edge::<BPC>(dst, off, bpp as isize, stride as isize, wd, l, lut);
        }
    }
}

/// `loop_filter_sb[1][0]` — kroma dikey kenarlar, tek düzlem (dav1d
/// `loop_filter_h_sb128uv_c`); filtre boyu `4 + 2·indeks`.
pub fn lpf_h_sb_uv<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    mask: &[u8],
    mask_stride: usize,
    level: &[[u8; 4]],
    lvl_stride: usize,
    lvl_idx: usize,
    lut: &LimLut,
    w4: usize,
    y0: usize,
    count: usize,
) {
    let bpp = if BPC == 8 { 1 } else { 2 };
    for gx in 1..w4 {
        for gy in y0..y0 + count {
            let v = mask[gy * mask_stride + gx];
            if v == 0 {
                continue;
            }
            let l = level_or(
                level,
                gy * lvl_stride + gx,
                gy * lvl_stride + gx - 1,
                lvl_idx,
            );
            if l == 0 {
                continue;
            }
            let off = gy * 4 * stride + gx * 4 * bpp;
            let wd = 4 + 2 * usize::from(v - 1);
            edge::<BPC>(dst, off, stride as isize, bpp as isize, wd, l, lut);
        }
    }
}

/// `loop_filter_sb[1][1]` — kroma yatay kenarlar, tek düzlem (dav1d
/// `loop_filter_v_sb128uv_c`); filtre boyu `4 + 2·indeks`.
pub fn lpf_v_sb_uv<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    mask: &[u8],
    mask_stride: usize,
    level: &[[u8; 4]],
    lvl_stride: usize,
    lvl_idx: usize,
    lut: &LimLut,
    w4: usize,
    y0: usize,
    count: usize,
) {
    let bpp = if BPC == 8 { 1 } else { 2 };
    for gy in y0..y0 + count {
        if gy == 0 {
            continue;
        }
        for gx in 0..w4 {
            let v = mask[gy * mask_stride + gx];
            if v == 0 {
                continue;
            }
            let l = level_or(
                level,
                gy * lvl_stride + gx,
                (gy - 1) * lvl_stride + gx,
                lvl_idx,
            );
            if l == 0 {
                continue;
            }
            let off = gy * 4 * stride + gx * 4 * bpp;
            let wd = 4 + 2 * usize::from(v - 1);
            edge::<BPC>(dst, off, bpp as isize, stride as isize, wd, l, lut);
        }
    }
}
