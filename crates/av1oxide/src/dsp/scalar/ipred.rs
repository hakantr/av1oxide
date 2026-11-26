//! Skaler intra tahmin çekirdekleri — S2b.2.
//!
//! Bu modül dav1d'nin **C yolunu** (`src/ipred_tmpl.c` + `src/pal.c`, asm
//! olmadan) bit-exact yeniden gerçekleştirir. Üst düzey sözleşme, mod
//! numaralandırması, kenar okuma aralıkları ve kaynak satır eşlemesi
//! [`crate::dsp::ipred`] belgesindedir; burada yalnız gövde ve aritmetik
//! ayrıntıları vardır.
//!
//! ## `const` generic monomorfizasyon
//!
//! dav1d her bit derinliği için `BITDEPTH=8` ve `BITDEPTH=16` olmak üzere
//! **iki kez** derlenir (`src/ipred_tmpl.c`); 10 ve 12-bit aynı 16bpc gövdesini
//! `bitdepth_max` ile paylaşır. Bu port aynı ayrımı `const HIGHBD: bool` ile
//! yapar: [`dispatch`] 8-bit'te `false`, 10/12-bit'te `true` monomorfizasyonunu
//! seçer. Böylece sıcak yol tek bir sabit dallanma taşır ve `u16` depolama
//! yalnız 16bpc'de okunur (`include/common/bitdepth.h:44-82`).
//!
//! ## Tanımlı aritmetik
//!
//! C'deki `int` taşması/kaydırması tanımsız davranıştır. Burada ara değerler
//! `i32` (konumlar `isize`) tutulur; geçerli AV1 piksel/açı aralığında sonuç C
//! ile bit-aynıdır. Piksel yazımı C'deki `pixel = (pixel) x` dönüşümü gibi
//! **kırparak** yazar (`x as u8` / `(x as u16)`); `iclip_pixel` açıkça
//! `[0, bitdepth_max]` aralığına kırpar.
//!
//! ## Açı bayrakları
//!
//! `angle & 511` temel açı; `(angle >> 9) & 1` yumuşak blok; `angle >> 10`
//! (sıfırdan farklıysa) kenar filtresi/upsample etkin. Kaynak:
//! `src/ipred_tmpl.c:409-412` (Z1), `462-466` (Z2), `542-546` (Z3).
//!
//! ## AK/AC ve palette
//!
//! `cfl_ac[3]` biçimleri `ss_hor`/`ss_ver` ile ayrılır (`src/ipred_tmpl.c:672-740`);
//! `cfl_pred` DC üretimi `src/ipred_tmpl.c:96-137, 207-226` ile aynıdır;
//! `pal_pred` iki nibble'ı iki piksele açar (`src/ipred_tmpl.c:723-735`).
//! `pal_idx_finish` [`crate::dsp::scalar::pal`] içindedir.

#![forbid(unsafe_code)]
// Mekanik dav1d portu: döngü indeksleri C'deki gibi açıkça kullanılır ve
// çekirdek imzaları dav1d `decl_*` sözleşmesiyle birebir aynı sayıda
// parametre taşır (ADR-0003).
#![allow(clippy::needless_range_loop)]

use crate::dsp::ipred::{CflPredMode, Edge, IntraPredMode};
use crate::dsp::ipred_tables::{DR_INTRA_DERIVATIVE, FILTER_INTRA_TAPS, SM_WEIGHTS};
use crate::frame::{BitDepth, ChromaFormat};

/// En büyük blok kenarı (AV1 tx üst sınırı).
const MAX_DIM: usize = 64;

/// Yerel kenar tamponunun uzunluğu: dav1d'nin 257 pikselik `scratch.edge`
/// (`src/internal.h:438`) ile aynı bütçe; yerel dizilerde `TMP_BASE`
/// çevresinde indekslenir.
const TMP_LEN: usize = 257;

/// Yerel kenar tamponunda sol-üst pikselin konumu (dav1d `edge + 128`).
const TMP_BASE: isize = 128;

/// `dav1d_filter_intra_taps` x86 yerleşiminde çıktı başına katsayı adımı
/// (`src/ipred_tmpl.c:586-601`: `FLT_INCR 2`).
const FLT_INCR: usize = 2;

#[inline(always)]
const fn pixel_bytes<const HIGHBD: bool>() -> usize {
    if HIGHBD { 2 } else { 1 }
}

#[inline(always)]
fn clip_isize(value: isize, low: isize, high: isize) -> isize {
    value.max(low).min(high)
}

/// F3c: `BPC` bit derinliğinin en büyük piksel değeri (derleme zamanı sabiti).
#[inline(always)]
const fn bpc_max<const BPC: u32>() -> u32 {
    (1u32 << BPC) - 1
}

#[inline(always)]
fn clip_pixel(value: i32, bitdepth_max: i32) -> i32 {
    value.max(0).min(bitdepth_max)
}

#[inline(always)]
fn read_edge<const HIGHBD: bool>(edge: Edge<'_>, relative: isize) -> i32 {
    let offset = edge.topleft as isize + relative * pixel_bytes::<HIGHBD>() as isize;
    debug_assert!(offset >= 0);
    let offset = offset as usize;
    if HIGHBD {
        i32::from(u16::from_le_bytes([
            edge.bytes[offset],
            edge.bytes[offset + 1],
        ]))
    } else {
        i32::from(edge.bytes[offset])
    }
}

#[inline(always)]
fn write_pixel<const HIGHBD: bool>(dst: &mut [u8], offset: usize, value: i32) {
    if HIGHBD {
        dst[offset..offset + 2].copy_from_slice(&(value as u16).to_le_bytes());
    } else {
        dst[offset] = value as u8;
    }
}

#[inline(always)]
fn read_pixel<const HIGHBD: bool>(buf: &[u8], offset: usize) -> i32 {
    if HIGHBD {
        i32::from(u16::from_le_bytes([buf[offset], buf[offset + 1]]))
    } else {
        i32::from(buf[offset])
    }
}

/// `get_filter_strength` (`src/ipred_tmpl.c:333-368`).
#[must_use]
fn get_filter_strength(wh: usize, angle: usize, is_sm: bool) -> usize {
    if is_sm {
        if wh <= 8 {
            if angle >= 64 {
                return 2;
            }
            if angle >= 40 {
                return 1;
            }
        } else if wh <= 16 {
            if angle >= 48 {
                return 2;
            }
            if angle >= 20 {
                return 1;
            }
        } else if wh <= 24 {
            if angle >= 4 {
                return 3;
            }
        } else {
            return 3;
        }
    } else if wh <= 8 {
        if angle >= 56 {
            return 1;
        }
    } else if wh <= 16 {
        if angle >= 40 {
            return 1;
        }
    } else if wh <= 24 {
        if angle >= 32 {
            return 3;
        }
        if angle >= 16 {
            return 2;
        }
        if angle >= 8 {
            return 1;
        }
    } else if wh <= 32 {
        if angle >= 32 {
            return 3;
        }
        if angle >= 4 {
            return 2;
        }
        return 1;
    } else {
        return 3;
    }
    0
}

/// `get_upsample` (`src/ipred_tmpl.c:396-398`): `angle < 40 && wh <= 16>>is_sm`.
#[must_use]
fn get_upsample(wh: usize, angle: usize, is_sm: bool) -> bool {
    angle < 40 && wh <= (16 >> usize::from(is_sm))
}

/// `angle` parametresinin çözümlenmiş biçimi (temel açı + iki bayrak).
#[derive(Clone, Copy)]
struct AngleFlags {
    angle: usize,
    is_sm: bool,
    enable: bool,
}

#[must_use]
fn parse_angle(angle: usize) -> AngleFlags {
    AngleFlags {
        angle: angle & 511,
        is_sm: (angle >> 9) & 1 == 1,
        enable: angle >> 10 != 0,
    }
}

/// `filter_edge` (`src/ipred_tmpl.c:370-394`).
///
/// `out[i]`, `edge` üzerinde `in_base + i` konumuna karşılık gelir; ilk
/// `min(sz, lim_from)` örnek kopyalanır, `min(lim_to, sz)`'e kadar 5'li
/// çekirdekle süzülür, kalanı kopyalanır.
#[allow(clippy::too_many_arguments)]
fn filter_edge<const HIGHBD: bool>(
    out: &mut [i32],
    sz: usize,
    lim_from: isize,
    lim_to: isize,
    edge: Edge<'_>,
    in_base: isize,
    from: isize,
    to: isize,
    strength: usize,
) {
    const KERNEL: [[i32; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
    assert!((1..=3).contains(&strength), "filtre gücü 1..3 olmalı");
    assert!(out.len() >= sz, "filtre çıktısı kısa");
    let copy_end = if lim_from < 0 {
        0
    } else {
        sz.min(lim_from as usize)
    };
    let mut i = 0;
    while i < copy_end {
        out[i] = read_edge::<HIGHBD>(edge, in_base + clip_isize(i as isize, from, to - 1));
        i += 1;
    }
    let filter_end = if lim_to < 0 {
        0
    } else {
        sz.min(lim_to as usize)
    };
    while i < filter_end {
        let mut sum = 0;
        for (j, weight) in KERNEL[strength - 1].iter().enumerate() {
            let position = clip_isize(i as isize - 2 + j as isize, from, to - 1);
            sum += read_edge::<HIGHBD>(edge, in_base + position) * weight;
        }
        out[i] = (sum + 8) >> 4;
        i += 1;
    }
    while i < sz {
        out[i] = read_edge::<HIGHBD>(edge, in_base + clip_isize(i as isize, from, to - 1));
        i += 1;
    }
}

/// `upsample_edge` (`src/ipred_tmpl.c:400-418`).
///
/// `out[2i]` kaynak örneği, `out[2i+1]` `{-1,9,9,-1}` çekirdeğiyle üretilen
/// ara örnektir; son örnek `out[2(hsz-1)]`'dir.
fn upsample_edge<const HIGHBD: bool>(
    out: &mut [i32],
    hsz: usize,
    edge: Edge<'_>,
    in_base: isize,
    from: isize,
    to: isize,
    bitdepth_max: i32,
) {
    const KERNEL: [i32; 4] = [-1, 9, 9, -1];
    assert!(out.len() >= 2 * hsz - 1, "upsample çıktısı kısa");
    let mut i = 0;
    while i < hsz - 1 {
        out[i * 2] = read_edge::<HIGHBD>(edge, in_base + clip_isize(i as isize, from, to - 1));
        let mut sum = 0;
        for (j, weight) in KERNEL.iter().enumerate() {
            let position = clip_isize(i as isize + j as isize - 1, from, to - 1);
            sum += read_edge::<HIGHBD>(edge, in_base + position) * weight;
        }
        out[i * 2 + 1] = clip_pixel((sum + 8) >> 4, bitdepth_max);
        i += 1;
    }
    out[(hsz - 1) * 2] =
        read_edge::<HIGHBD>(edge, in_base + clip_isize(hsz as isize - 1, from, to - 1));
}

// ---------------------------------------------------------------------------
// Doğrulama (ADR-0003: her çağrıda O(1) sözleşme denetimi)
// ---------------------------------------------------------------------------

fn validate_geometry(width: usize, height: usize) {
    assert!(
        (4..=MAX_DIM).contains(&width) && width.is_power_of_two(),
        "genişlik 4..=64 ve 2'nin kuvveti olmalı: {width}"
    );
    assert!(
        (4..=MAX_DIM).contains(&height) && height.is_power_of_two(),
        "yükseklik 4..=64 ve 2'nin kuvveti olmalı: {height}"
    );
}

fn validate_dst<const HIGHBD: bool>(dst: &[u8], stride: usize, width: usize, height: usize) {
    let bytes = pixel_bytes::<HIGHBD>();
    assert!(
        stride >= width * bytes,
        "dst_stride satırdan küçük: {stride} < {}",
        width * bytes
    );
    if HIGHBD {
        assert!(
            stride.is_multiple_of(2),
            "16bpc dst_stride çift olmalı: {stride}"
        );
    }
    let required = (height - 1) * stride + width * bytes;
    assert!(
        dst.len() >= required,
        "dst dilimi kısa: {} < {required}",
        dst.len()
    );
}

fn validate_edge<const HIGHBD: bool>(edge: Edge<'_>, back: usize, front: usize) {
    let bytes = pixel_bytes::<HIGHBD>();
    assert!(
        edge.topleft >= back * bytes,
        "edge.topleft geri aralığı kapsamıyor: {} < {}",
        edge.topleft,
        back * bytes
    );
    assert!(
        edge.topleft.is_multiple_of(bytes),
        "edge.topleft piksel hizasında değil: {}",
        edge.topleft
    );
    let required = edge.topleft + (front + 1) * bytes;
    assert!(
        edge.bytes.len() >= required,
        "edge dilimi ileri aralığı kapsamıyor: {} < {required}",
        edge.bytes.len()
    );
}

/// Modun `edge` üzerinde okuduğu **(geri, ileri) piksel aralığı**.
///
/// Ayrıntılı tablo [`crate::dsp::ipred`] belgesindedir. Bu işlev çekirdeğin
/// ön denetiminin **tek kaynağıdır** ve sentinel testleri aynı sınırı kullanır.
#[must_use]
pub fn required_range(
    mode: IntraPredMode,
    width: usize,
    height: usize,
    angle: usize,
) -> (usize, usize) {
    let flags = parse_angle(angle);
    match mode {
        IntraPredMode::Dc
        | IntraPredMode::Paeth
        | IntraPredMode::Smooth
        | IntraPredMode::SmoothV
        | IntraPredMode::SmoothH
        | IntraPredMode::Filter => (height, width),
        IntraPredMode::TopDc | IntraPredMode::Vert => (0, width),
        IntraPredMode::LeftDc | IntraPredMode::Hor => (height, 0),
        IntraPredMode::Dc128 => (0, 0),
        IntraPredMode::Z1 => {
            assert!(flags.angle < 90, "Z1 açısı 90'dan küçük olmalı");
            (0, width + width.min(height))
        }
        IntraPredMode::Z2 => {
            assert!(
                flags.angle > 90 && flags.angle < 180,
                "Z2 açısı 90..180 aralığında olmalı"
            );
            (height, width)
        }
        IntraPredMode::Z3 => {
            assert!(flags.angle > 180, "Z3 açısı 180'den büyük olmalı");
            let upsample =
                flags.enable && get_upsample(width + height, flags.angle - 180, flags.is_sm);
            let strength = if flags.enable && !upsample {
                get_filter_strength(width + height, flags.angle - 180, flags.is_sm)
            } else {
                0
            };
            if upsample || strength > 0 {
                (width + height, 0)
            } else {
                (height + width.min(height), 0)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ortak tabanlar
// ---------------------------------------------------------------------------

fn splat_dc<const HIGHBD: bool>(
    dst: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
    dc: i32,
) {
    let bytes = pixel_bytes::<HIGHBD>();
    for y in 0..height {
        for x in 0..width {
            write_pixel::<HIGHBD>(dst, y * stride + x * bytes, dc);
        }
    }
}

fn dc_gen_top<const HIGHBD: bool>(edge: Edge<'_>, width: usize) -> u32 {
    let mut dc = (width >> 1) as u32;
    for i in 0..width {
        dc += read_edge::<HIGHBD>(edge, 1 + i as isize) as u32;
    }
    dc >> width.trailing_zeros()
}

fn dc_gen_left<const HIGHBD: bool>(edge: Edge<'_>, height: usize) -> u32 {
    let mut dc = (height >> 1) as u32;
    for i in 0..height {
        dc += read_edge::<HIGHBD>(edge, -(1 + i as isize)) as u32;
    }
    dc >> height.trailing_zeros()
}

fn dc_gen<const HIGHBD: bool>(edge: Edge<'_>, width: usize, height: usize) -> u32 {
    let mut dc = ((width + height) >> 1) as u32;
    for i in 0..width {
        dc += read_edge::<HIGHBD>(edge, 1 + i as isize) as u32;
    }
    for i in 0..height {
        dc += read_edge::<HIGHBD>(edge, -(1 + i as isize)) as u32;
    }
    dc >>= (width + height).trailing_zeros();
    if width != height {
        let (multiplier, shift) = if HIGHBD {
            if width > height * 2 || height > width * 2 {
                (0x6667u64, 17)
            } else {
                (0xAAABu64, 17)
            }
        } else if width > height * 2 || height > width * 2 {
            (0x3334u64, 16)
        } else {
            (0x5556u64, 16)
        };
        dc = ((u64::from(dc) * multiplier) >> shift) as u32;
    }
    dc
}

fn paeth<const HIGHBD: bool>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let bytes = pixel_bytes::<HIGHBD>();
    let top_left = read_edge::<HIGHBD>(edge, 0);
    for y in 0..height {
        let left = read_edge::<HIGHBD>(edge, -(y as isize + 1));
        for x in 0..width {
            let top = read_edge::<HIGHBD>(edge, 1 + x as isize);
            let base = left + top - top_left;
            let ldiff = (left - base).abs();
            let tdiff = (top - base).abs();
            let tldiff = (top_left - base).abs();
            let value = if ldiff <= tdiff && ldiff <= tldiff {
                left
            } else if tdiff <= tldiff {
                top
            } else {
                top_left
            };
            write_pixel::<HIGHBD>(dst, y * stride + x * bytes, value);
        }
    }
}

fn smooth<const HIGHBD: bool>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let bytes = pixel_bytes::<HIGHBD>();
    let weights_hor = &SM_WEIGHTS[width..width + width];
    let weights_ver = &SM_WEIGHTS[height..height + height];
    let right = read_edge::<HIGHBD>(edge, width as isize);
    let bottom = read_edge::<HIGHBD>(edge, -(height as isize));
    for y in 0..height {
        for x in 0..width {
            let prediction = i32::from(weights_ver[y]) * read_edge::<HIGHBD>(edge, 1 + x as isize)
                + (256 - i32::from(weights_ver[y])) * bottom
                + i32::from(weights_hor[x]) * read_edge::<HIGHBD>(edge, -(y as isize + 1))
                + (256 - i32::from(weights_hor[x])) * right;
            write_pixel::<HIGHBD>(dst, y * stride + x * bytes, (prediction + 256) >> 9);
        }
    }
}

fn smooth_v<const HIGHBD: bool>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let bytes = pixel_bytes::<HIGHBD>();
    let weights_ver = &SM_WEIGHTS[height..height + height];
    let bottom = read_edge::<HIGHBD>(edge, -(height as isize));
    for y in 0..height {
        for x in 0..width {
            let prediction = i32::from(weights_ver[y]) * read_edge::<HIGHBD>(edge, 1 + x as isize)
                + (256 - i32::from(weights_ver[y])) * bottom;
            write_pixel::<HIGHBD>(dst, y * stride + x * bytes, (prediction + 128) >> 8);
        }
    }
}

fn smooth_h<const HIGHBD: bool>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let bytes = pixel_bytes::<HIGHBD>();
    let weights_hor = &SM_WEIGHTS[width..width + width];
    let right = read_edge::<HIGHBD>(edge, width as isize);
    for y in 0..height {
        let left = read_edge::<HIGHBD>(edge, -(y as isize + 1));
        for x in 0..width {
            let prediction =
                i32::from(weights_hor[x]) * left + (256 - i32::from(weights_hor[x])) * right;
            write_pixel::<HIGHBD>(dst, y * stride + x * bytes, (prediction + 128) >> 8);
        }
    }
}

fn z1<const HIGHBD: bool>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
    bitdepth_max: i32,
) {
    let bytes = pixel_bytes::<HIGHBD>();
    let flags = parse_angle(angle);
    let angle = flags.angle;
    assert!(angle < 90, "Z1 açısı 90'dan küçük olmalı: {angle}");
    let mut dx = i32::from(
        *DR_INTRA_DERIVATIVE
            .get(angle >> 1)
            .unwrap_or_else(|| panic!("Z1 açısı tablo dışı: {angle}")),
    );
    let mut buffer = [0i32; TMP_LEN];
    let upsample = flags.enable && get_upsample(width + height, 90 - angle, flags.is_sm);
    let (base_inc, max_base_x) = if upsample {
        upsample_edge::<HIGHBD>(
            &mut buffer,
            width + height,
            edge,
            1,
            -1,
            (width + width.min(height)) as isize,
            bitdepth_max,
        );
        dx <<= 1;
        (2isize, 2 * (width + height) - 2)
    } else {
        let strength = if flags.enable {
            get_filter_strength(width + height, 90 - angle, flags.is_sm)
        } else {
            0
        };
        if strength > 0 {
            filter_edge::<HIGHBD>(
                &mut buffer,
                width + height,
                0,
                (width + height) as isize,
                edge,
                1,
                -1,
                (width + width.min(height)) as isize,
                strength,
            );
            (1isize, width + height - 1)
        } else {
            for i in 0..width + width.min(height) {
                buffer[i] = read_edge::<HIGHBD>(edge, 1 + i as isize);
            }
            (1isize, width + width.min(height) - 1)
        }
    };
    let mut xpos = dx;
    for y in 0..height {
        let frac = xpos & 0x3E;
        let mut base = xpos >> 6;
        let mut x = 0;
        while x < width {
            if base < max_base_x as i32 {
                let value = buffer[base as usize] * (64 - frac) + buffer[base as usize + 1] * frac;
                write_pixel::<HIGHBD>(dst, y * stride + x * bytes, (value + 32) >> 6);
            } else {
                let fill = buffer[max_base_x];
                while x < width {
                    write_pixel::<HIGHBD>(dst, y * stride + x * bytes, fill);
                    x += 1;
                }
                break;
            }
            x += 1;
            base += base_inc as i32;
        }
        xpos += dx;
    }
}

#[allow(clippy::too_many_arguments)]
fn z2<const HIGHBD: bool>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
    max_width: usize,
    max_height: usize,
    bitdepth_max: i32,
) {
    let bytes = pixel_bytes::<HIGHBD>();
    let flags = parse_angle(angle);
    let angle = flags.angle;
    assert!(
        angle > 90 && angle < 180,
        "Z2 açısı 90..180 aralığında olmalı: {angle}"
    );
    let mut dy = i32::from(
        *DR_INTRA_DERIVATIVE
            .get((angle - 90) >> 1)
            .unwrap_or_else(|| panic!("Z2 açısı tablo dışı: {angle}")),
    );
    let mut dx = i32::from(
        *DR_INTRA_DERIVATIVE
            .get((180 - angle) >> 1)
            .unwrap_or_else(|| panic!("Z2 açısı tablo dışı: {angle}")),
    );
    let upsample_left = flags.enable && get_upsample(width + height, 180 - angle, flags.is_sm);
    let upsample_above = flags.enable && get_upsample(width + height, angle - 90, flags.is_sm);
    let mut buffer = [0i32; TMP_LEN];
    let base = TMP_BASE;

    if upsample_above {
        upsample_edge::<HIGHBD>(
            &mut buffer[TMP_BASE as usize..],
            width + 1,
            edge,
            0,
            0,
            (width + 1) as isize,
            bitdepth_max,
        );
        dx <<= 1;
    } else {
        let strength = if flags.enable {
            get_filter_strength(width + height, angle - 90, flags.is_sm)
        } else {
            0
        };
        if strength > 0 {
            filter_edge::<HIGHBD>(
                &mut buffer[(TMP_BASE + 1) as usize..],
                width,
                0,
                max_width as isize,
                edge,
                1,
                -1,
                width as isize,
                strength,
            );
        } else {
            for i in 0..width {
                buffer[(base + 1 + i as isize) as usize] =
                    read_edge::<HIGHBD>(edge, 1 + i as isize);
            }
        }
    }
    if upsample_left {
        upsample_edge::<HIGHBD>(
            &mut buffer[(TMP_BASE - 2 * height as isize) as usize..],
            height + 1,
            edge,
            -(height as isize),
            0,
            (height + 1) as isize,
            bitdepth_max,
        );
        dy <<= 1;
    } else {
        let strength = if flags.enable {
            get_filter_strength(width + height, 180 - angle, flags.is_sm)
        } else {
            0
        };
        if strength > 0 {
            filter_edge::<HIGHBD>(
                &mut buffer[(TMP_BASE - height as isize) as usize..],
                height,
                height as isize - max_height as isize,
                height as isize,
                edge,
                -(height as isize),
                0,
                (height + 1) as isize,
                strength,
            );
        } else {
            for i in 0..height {
                buffer[(base - height as isize + i as isize) as usize] =
                    read_edge::<HIGHBD>(edge, -(height as isize) + i as isize);
            }
        }
    }
    buffer[base as usize] = read_edge::<HIGHBD>(edge, 0);

    let base_inc_x = 1 + isize::from(upsample_above);
    let left = base - (1 + isize::from(upsample_left));
    let mut xpos = ((1 + isize::from(upsample_above)) << 6) - dx as isize;
    for y in 0..height {
        let mut base_x = xpos >> 6;
        let frac_x = (xpos & 0x3E) as i32;
        let mut ypos = ((y as isize) << (6 + u32::from(upsample_left))) - dy as isize;
        for x in 0..width {
            let value = if base_x >= 0 {
                buffer[(base + base_x) as usize] * (64 - frac_x)
                    + buffer[(base + base_x + 1) as usize] * frac_x
            } else {
                let base_y = ypos >> 6;
                let frac_y = (ypos & 0x3E) as i32;
                buffer[(left - base_y) as usize] * (64 - frac_y)
                    + buffer[(left - base_y - 1) as usize] * frac_y
            };
            write_pixel::<HIGHBD>(dst, y * stride + x * bytes, (value + 32) >> 6);
            base_x += base_inc_x;
            ypos -= dy as isize;
        }
        xpos -= dx as isize;
    }
}

fn z3<const HIGHBD: bool>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
    bitdepth_max: i32,
) {
    let bytes = pixel_bytes::<HIGHBD>();
    let flags = parse_angle(angle);
    let angle = flags.angle;
    assert!(angle > 180, "Z3 açısı 180'den büyük olmalı: {angle}");
    let mut dy = i32::from(
        *DR_INTRA_DERIVATIVE
            .get((270 - angle) >> 1)
            .unwrap_or_else(|| panic!("Z3 açısı tablo dışı: {angle}")),
    );
    let mut left_out = [0i32; TMP_LEN];
    let upsample = flags.enable && get_upsample(width + height, angle - 180, flags.is_sm);
    let strength = if flags.enable && !upsample {
        get_filter_strength(width + height, angle - 180, flags.is_sm)
    } else {
        0
    };
    let direct = !upsample && strength == 0;
    let mut left_base = 0isize;
    let mut max_base_y;
    let base_inc;
    if upsample {
        upsample_edge::<HIGHBD>(
            &mut left_out,
            width + height,
            edge,
            -((width + height) as isize),
            (width.saturating_sub(height)) as isize,
            (width + height + 1) as isize,
            bitdepth_max,
        );
        left_base = (2 * (width + height) - 2) as isize;
        max_base_y = 2 * (width + height) - 2;
        dy <<= 1;
        base_inc = 2isize;
    } else {
        if strength > 0 {
            filter_edge::<HIGHBD>(
                &mut left_out,
                width + height,
                0,
                (width + height) as isize,
                edge,
                -((width + height) as isize),
                width.saturating_sub(height) as isize,
                (width + height + 1) as isize,
                strength,
            );
            left_base = (width + height - 1) as isize;
        }
        max_base_y = width + height - 1;
        base_inc = 1;
        if direct {
            max_base_y = height + width.min(height) - 1;
        }
    }
    let mut ypos = dy as isize;
    for x in 0..width {
        let frac = (ypos & 0x3E) as i32;
        let mut base = ypos >> 6;
        let mut y = 0;
        while y < height {
            if base < max_base_y as isize {
                let value = if direct {
                    read_edge::<HIGHBD>(edge, -(1 + base)) * (64 - frac)
                        + read_edge::<HIGHBD>(edge, -(2 + base)) * frac
                } else {
                    left_out[(left_base - base) as usize] * (64 - frac)
                        + left_out[(left_base - base - 1) as usize] * frac
                };
                write_pixel::<HIGHBD>(dst, y * stride + x * bytes, (value + 32) >> 6);
            } else {
                let fill = if direct {
                    read_edge::<HIGHBD>(edge, -(1 + max_base_y as isize))
                } else {
                    left_out[(left_base - max_base_y as isize) as usize]
                };
                while y < height {
                    write_pixel::<HIGHBD>(dst, y * stride + x * bytes, fill);
                    y += 1;
                }
                break;
            }
            y += 1;
            base += base_inc;
        }
        ypos += dy as isize;
    }
}

fn filter<const HIGHBD: bool>(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
    bitdepth_max: i32,
) {
    let bytes = pixel_bytes::<HIGHBD>();
    let filter_index = angle & 511;
    assert!(
        filter_index < 5,
        "filtre indeksi 0..4 olmalı: {filter_index}"
    );
    assert!(
        width <= 32 && height <= 32,
        "FILTER_PRED yalnız ≤32×32: {width}×{height}"
    );
    assert!(
        width.is_multiple_of(4) && height.is_multiple_of(2),
        "FILTER_PRED blok boyutu 4/2'nin katı olmalı"
    );
    let taps = &FILTER_INTRA_TAPS[filter_index];
    let mut y = 0;
    while y < height {
        let mut x = 0;
        while x < width {
            let group = x / 4;
            // dav1d `ipred_filter_c` (`src/ipred_tmpl.c:618-670`): ilk satır
            // çiftinde `top` üst kenardır; sonraki çiftlerde bir önceki
            // çiftin **ikinci** satırıdır (`top = &dst[stride]`). İlk grubun
            // p0'ı her zaman sol kenardır; sonraki gruplarda p0 = top[-1].
            let top_from_edge = y == 0;
            let edge_topleft = -(y as isize);
            let (p0, p1, p2, p3, p4) = if top_from_edge {
                if group == 0 {
                    (
                        read_edge::<HIGHBD>(edge, edge_topleft),
                        read_edge::<HIGHBD>(edge, 1),
                        read_edge::<HIGHBD>(edge, 2),
                        read_edge::<HIGHBD>(edge, 3),
                        read_edge::<HIGHBD>(edge, 4),
                    )
                } else {
                    let column = 4 * group as isize;
                    (
                        read_edge::<HIGHBD>(edge, column),
                        read_edge::<HIGHBD>(edge, column + 1),
                        read_edge::<HIGHBD>(edge, column + 2),
                        read_edge::<HIGHBD>(edge, column + 3),
                        read_edge::<HIGHBD>(edge, column + 4),
                    )
                }
            } else {
                let top_row = (y - 1) * stride;
                let column = 4 * group;
                let p0 = if group == 0 {
                    read_edge::<HIGHBD>(edge, edge_topleft)
                } else {
                    read_pixel::<HIGHBD>(dst, top_row + (column - 1) * bytes)
                };
                (
                    p0,
                    read_pixel::<HIGHBD>(dst, top_row + column * bytes),
                    read_pixel::<HIGHBD>(dst, top_row + (column + 1) * bytes),
                    read_pixel::<HIGHBD>(dst, top_row + (column + 2) * bytes),
                    read_pixel::<HIGHBD>(dst, top_row + (column + 3) * bytes),
                )
            };
            let (p5, p6) = if group == 0 {
                (
                    read_edge::<HIGHBD>(edge, edge_topleft - 1),
                    read_edge::<HIGHBD>(edge, edge_topleft - 2),
                )
            } else {
                (
                    read_pixel::<HIGHBD>(dst, y * stride + (x - 1) * bytes),
                    read_pixel::<HIGHBD>(dst, (y + 1) * stride + (x - 1) * bytes),
                )
            };
            for yy in 0..2 {
                for xx in 0..4 {
                    let slot = (yy * 4 + xx) * FLT_INCR;
                    let accum = i32::from(taps[slot]) * p0
                        + i32::from(taps[slot + 1]) * p1
                        + i32::from(taps[slot + 16]) * p2
                        + i32::from(taps[slot + 17]) * p3
                        + i32::from(taps[slot + 32]) * p4
                        + i32::from(taps[slot + 33]) * p5
                        + i32::from(taps[slot + 48]) * p6;
                    write_pixel::<HIGHBD>(
                        dst,
                        (y + yy) * stride + (x + xx) * bytes,
                        clip_pixel((accum + 8) >> 4, bitdepth_max),
                    );
                }
            }
            x += 4;
        }
        y += 2;
    }
}

// ---------------------------------------------------------------------------
// Genel giriş noktaları
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn dispatch<const HIGHBD: bool>(
    mode: IntraPredMode,
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
    max_width: usize,
    max_height: usize,
    bitdepth_max: u16,
) {
    validate_geometry(width, height);
    validate_dst::<HIGHBD>(dst, dst_stride, width, height);
    let (back, front) = required_range(mode, width, height, angle);
    validate_edge::<HIGHBD>(edge, back, front);
    if HIGHBD {
        assert!(
            bitdepth_max <= 4095,
            "16bpc bitdepth_max 4095'i aşamaz: {bitdepth_max}"
        );
    }
    let max = if HIGHBD { i32::from(bitdepth_max) } else { 255 };
    match mode {
        IntraPredMode::Dc => splat_dc::<HIGHBD>(
            dst,
            dst_stride,
            width,
            height,
            dc_gen::<HIGHBD>(edge, width, height) as i32,
        ),
        IntraPredMode::Vert => {
            for y in 0..height {
                for x in 0..width {
                    let value = read_edge::<HIGHBD>(edge, 1 + x as isize);
                    write_pixel::<HIGHBD>(dst, y * dst_stride + x * pixel_bytes::<HIGHBD>(), value);
                }
            }
        }
        IntraPredMode::Hor => {
            for y in 0..height {
                let value = read_edge::<HIGHBD>(edge, -(y as isize + 1));
                for x in 0..width {
                    write_pixel::<HIGHBD>(dst, y * dst_stride + x * pixel_bytes::<HIGHBD>(), value);
                }
            }
        }
        IntraPredMode::LeftDc => splat_dc::<HIGHBD>(
            dst,
            dst_stride,
            width,
            height,
            dc_gen_left::<HIGHBD>(edge, height) as i32,
        ),
        IntraPredMode::TopDc => splat_dc::<HIGHBD>(
            dst,
            dst_stride,
            width,
            height,
            dc_gen_top::<HIGHBD>(edge, width) as i32,
        ),
        IntraPredMode::Dc128 => {
            let dc = if HIGHBD { (max + 1) >> 1 } else { 128 };
            splat_dc::<HIGHBD>(dst, dst_stride, width, height, dc);
        }
        IntraPredMode::Z1 => z1::<HIGHBD>(dst, dst_stride, edge, width, height, angle, max),
        IntraPredMode::Z2 => z2::<HIGHBD>(
            dst, dst_stride, edge, width, height, angle, max_width, max_height, max,
        ),
        IntraPredMode::Z3 => z3::<HIGHBD>(dst, dst_stride, edge, width, height, angle, max),
        IntraPredMode::Smooth => smooth::<HIGHBD>(dst, dst_stride, edge, width, height),
        IntraPredMode::SmoothV => smooth_v::<HIGHBD>(dst, dst_stride, edge, width, height),
        IntraPredMode::SmoothH => smooth_h::<HIGHBD>(dst, dst_stride, edge, width, height),
        IntraPredMode::Paeth => paeth::<HIGHBD>(dst, dst_stride, edge, width, height),
        IntraPredMode::Filter => filter::<HIGHBD>(dst, dst_stride, edge, width, height, angle, max),
    }
}

/// Bir intra tahmin modunu skaler olarak uygular.
///
/// Ayrıntılı sözleşme [`crate::dsp::ipred`] belgesindedir.
///
/// # Panikler
///
/// Boyut/stride/kenar aralığı sözleşmeyi ihlal ederse panikler.
#[allow(clippy::too_many_arguments)]
pub fn intra_pred<const BPC: u32>(
    mode: IntraPredMode,
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
    max_width: usize,
    max_height: usize,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    debug_assert_eq!(bit_depth.bits(), BPC, "yuva bit derinliği ≠ çağrı");
    debug_assert_eq!(
        u32::from(bitdepth_max),
        bpc_max::<BPC>(),
        "bitdepth_max ≠ yuva"
    );
    let bitdepth_max = bpc_max::<BPC>() as u16;
    if BPC == 8 {
        dispatch::<false>(
            mode,
            dst,
            dst_stride,
            edge,
            width,
            height,
            angle,
            max_width,
            max_height,
            bitdepth_max,
        );
    } else {
        dispatch::<true>(
            mode,
            dst,
            dst_stride,
            edge,
            width,
            height,
            angle,
            max_width,
            max_height,
            bitdepth_max,
        );
    }
}

macro_rules! angular_wrapper {
    ($name:ident, $mode:expr, $doc:literal) => {
        #[doc = $doc]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<const BPC: u32>(
            dst: &mut [u8],
            dst_stride: usize,
            edge: Edge<'_>,
            width: usize,
            height: usize,
            angle: usize,
            max_width: usize,
            max_height: usize,
            bit_depth: BitDepth,
            bitdepth_max: u16,
        ) {
            intra_pred::<BPC>(
                $mode,
                dst,
                dst_stride,
                edge,
                width,
                height,
                angle,
                max_width,
                max_height,
                bit_depth,
                bitdepth_max,
            );
        }
    };
}

angular_wrapper!(ipred_dc, IntraPredMode::Dc, "`DC_PRED` skaler çekirdeği.");
angular_wrapper!(
    ipred_vert,
    IntraPredMode::Vert,
    "`VERT_PRED` skaler çekirdeği."
);
angular_wrapper!(
    ipred_hor,
    IntraPredMode::Hor,
    "`HOR_PRED` skaler çekirdeği."
);
angular_wrapper!(
    ipred_left_dc,
    IntraPredMode::LeftDc,
    "`LEFT_DC_PRED` skaler çekirdeği."
);
angular_wrapper!(
    ipred_top_dc,
    IntraPredMode::TopDc,
    "`TOP_DC_PRED` skaler çekirdeği."
);
angular_wrapper!(
    ipred_dc_128,
    IntraPredMode::Dc128,
    "`DC_128_PRED` skaler çekirdeği."
);
angular_wrapper!(ipred_z1, IntraPredMode::Z1, "`Z1_PRED` skaler çekirdeği.");
angular_wrapper!(ipred_z2, IntraPredMode::Z2, "`Z2_PRED` skaler çekirdeği.");
angular_wrapper!(ipred_z3, IntraPredMode::Z3, "`Z3_PRED` skaler çekirdeği.");
angular_wrapper!(
    ipred_smooth,
    IntraPredMode::Smooth,
    "`SMOOTH_PRED` skaler çekirdeği."
);
angular_wrapper!(
    ipred_smooth_v,
    IntraPredMode::SmoothV,
    "`SMOOTH_V_PRED` skaler çekirdeği."
);
angular_wrapper!(
    ipred_smooth_h,
    IntraPredMode::SmoothH,
    "`SMOOTH_H_PRED` skaler çekirdeği."
);
angular_wrapper!(
    ipred_paeth,
    IntraPredMode::Paeth,
    "`PAETH_PRED` skaler çekirdeği."
);
angular_wrapper!(
    ipred_filter,
    IntraPredMode::Filter,
    "`FILTER_PRED` skaler çekirdeği."
);

// ---------------------------------------------------------------------------
// CfL
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn cfl_ac_impl<const HIGHBD: bool>(
    format: ChromaFormat,
    ac: &mut [i16],
    y: &[u8],
    y_stride: usize,
    w_pad: usize,
    h_pad: usize,
    cw: usize,
    ch: usize,
) {
    let (ss_hor, ss_ver) = match format {
        ChromaFormat::C420 => (true, true),
        ChromaFormat::C422 => (true, false),
        ChromaFormat::C444 => (false, false),
        ChromaFormat::Monochrome => return,
    };
    // Bozuk girdi söz dizimi çözümünü geçersiz parametrelerle buraya
    // getirebilir; panik yerine sessizce dön (geçerli akışlarda değişmez).
    if !(4..=32).contains(&cw)
        || !cw.is_power_of_two()
        || !(4..=32).contains(&ch)
        || !ch.is_power_of_two()
        || w_pad.saturating_mul(4) >= cw
        || h_pad.saturating_mul(4) >= ch
    {
        return;
    }
    debug_assert!(
        w_pad * 4 < cw && h_pad * 4 < ch,
        "cfl_ac w_pad/h_pad sınır dışı: {w_pad}/{h_pad}"
    );
    if ac.len() < cw * ch {
        return;
    }
    let bytes = pixel_bytes::<HIGHBD>();
    let luma_width = cw << usize::from(ss_hor);
    let luma_height = ch << usize::from(ss_ver);
    if y_stride < luma_width * bytes {
        return;
    }
    let required = (luma_height - 1) * y_stride + luma_width * bytes;
    if y.len() < required {
        return;
    }

    let visible_w = cw - 4 * w_pad;
    let visible_h = ch - 4 * h_pad;
    for row in 0..visible_h {
        let luma_row = row << usize::from(ss_ver);
        let mut x = 0;
        while x < visible_w {
            let column = (x << usize::from(ss_hor)) * bytes;
            let mut sum = read_pixel::<HIGHBD>(y, luma_row * y_stride + column);
            if ss_hor {
                sum += read_pixel::<HIGHBD>(y, luma_row * y_stride + column + bytes);
            }
            if ss_ver {
                let below = (luma_row + 1) * y_stride;
                sum += read_pixel::<HIGHBD>(y, below + column);
                if ss_hor {
                    sum += read_pixel::<HIGHBD>(y, below + column + bytes);
                }
            }
            // `ipred_tmpl.c:672`: `ac[x] = ac_sum << (1 + !ss_ver + !ss_hor)`.
            // C'de `ac_sum` `int`'tir ve sonuç `int16_t`'ye daralır. Kaydırma
            // kroma biçimine **bağlıdır**, bu yüzden en büyük değer her zaman
            // aynıdır: 4:4:4 `4095<<3`, 4:2:2 `8190<<2`, 4:2:0 `16380<<1`
            // → hepsi **32 760 < 32768**. Yani `u16` ara değer de taşmaz ve
            // C'deki daraltma kayıpsızdır; `wrapping`/`checked` gerekmez.
            ac[row * cw + x] = (sum << (1 + usize::from(!ss_ver) + usize::from(!ss_hor))) as i16;
            x += 1;
        }
        while x < cw {
            ac[row * cw + x] = ac[row * cw + x - 1];
            x += 1;
        }
    }
    for row in visible_h..ch {
        for x in 0..cw {
            ac[row * cw + x] = ac[(row - 1) * cw + x];
        }
    }

    // `ipred_tmpl.c:684-691`: `int sum`; `log2sz ≤ 6+6 = 12` için en büyük
    // `2^11 + 1024·32767 ≈ 3,36e7 < 2^31−1` ⇒ taşma yok.
    let log2sz = cw.trailing_zeros() + ch.trailing_zeros();
    let mut sum = (1i32 << log2sz) >> 1;
    for value in ac.iter().take(cw * ch) {
        sum += i32::from(*value);
    }
    sum >>= log2sz;
    for value in ac.iter_mut().take(cw * ch) {
        *value = (i32::from(*value) - sum) as i16;
    }
}

macro_rules! cfl_ac_wrapper {
    ($name:ident, $format:expr, $doc:literal) => {
        #[doc = $doc]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<const BPC: u32>(
            ac: &mut [i16],
            y: &[u8],
            y_stride: usize,
            w_pad: usize,
            h_pad: usize,
            cw: usize,
            ch: usize,
            bit_depth: BitDepth,
        ) {
            debug_assert_eq!(bit_depth.bits(), BPC, "yuva bit derinliği ≠ çağrı");
            if BPC == 8 {
                cfl_ac_impl::<false>($format, ac, y, y_stride, w_pad, h_pad, cw, ch);
            } else {
                cfl_ac_impl::<true>($format, ac, y, y_stride, w_pad, h_pad, cw, ch);
            }
        }
    };
}

cfl_ac_wrapper!(cfl_ac_420, ChromaFormat::C420, "4:2:0 `cfl_ac` çekirdeği.");
cfl_ac_wrapper!(cfl_ac_422, ChromaFormat::C422, "4:2:2 `cfl_ac` çekirdeği.");
cfl_ac_wrapper!(cfl_ac_444, ChromaFormat::C444, "4:4:4 `cfl_ac` çekirdeği.");

#[allow(clippy::too_many_arguments)]
fn cfl_pred_impl<const HIGHBD: bool>(
    mode: CflPredMode,
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    ac: &[i16],
    alpha: i32,
    bitdepth_max: u16,
) {
    assert!(
        (4..=32).contains(&width) && width.is_power_of_two(),
        "cfl_pred genişliği 4..=32 ve 2'nin kuvveti olmalı: {width}"
    );
    assert!(
        (4..=32).contains(&height) && height.is_power_of_two(),
        "cfl_pred yüksekliği 4..=32 ve 2'nin kuvveti olmalı: {height}"
    );
    assert!(
        (-16..=16).contains(&alpha),
        "cfl_pred alpha q3 [-16,16] dışında: {alpha}"
    );
    validate_dst::<HIGHBD>(dst, dst_stride, width, height);
    validate_edge::<HIGHBD>(edge, height, width);
    assert!(ac.len() >= width * height, "cfl_pred ac dilimi kısa");
    let max = if HIGHBD {
        assert!(bitdepth_max <= 4095, "16bpc bitdepth_max 4095'i aşamaz");
        i32::from(bitdepth_max)
    } else {
        255
    };
    let dc = match mode {
        CflPredMode::Dc => dc_gen::<HIGHBD>(edge, width, height),
        CflPredMode::LeftDc => dc_gen_left::<HIGHBD>(edge, height),
        CflPredMode::TopDc => dc_gen_top::<HIGHBD>(edge, width),
        CflPredMode::Dc128 => {
            if HIGHBD {
                ((max + 1) >> 1) as u32
            } else {
                128
            }
        }
    };
    let bytes = pixel_bytes::<HIGHBD>();
    for y in 0..height {
        for x in 0..width {
            // `ipred_tmpl.c:145-148` (`cfl_pred`): `alpha` `int` (`|α| ≤ 16`),
            // `ac` `int16_t` ⇒ `|diff| ≤ 524 288`; `+ 32` taşmaz.
            let diff = alpha * i32::from(ac[y * width + x]);
            let absolute = (diff.abs() + 32) >> 6;
            let signed = if diff < 0 { -absolute } else { absolute };
            let value = clip_pixel(dc as i32 + signed, max);
            write_pixel::<HIGHBD>(dst, y * dst_stride + x * bytes, value);
        }
    }
}

macro_rules! cfl_pred_wrapper {
    ($name:ident, $mode:expr, $doc:literal) => {
        #[doc = $doc]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<const BPC: u32>(
            dst: &mut [u8],
            dst_stride: usize,
            edge: Edge<'_>,
            width: usize,
            height: usize,
            ac: &[i16],
            alpha: i32,
            bit_depth: BitDepth,
            bitdepth_max: u16,
        ) {
            debug_assert_eq!(bit_depth.bits(), BPC, "yuva bit derinliği ≠ çağrı");
            debug_assert_eq!(
                u32::from(bitdepth_max),
                bpc_max::<BPC>(),
                "bitdepth_max ≠ yuva"
            );
            let bitdepth_max = bpc_max::<BPC>() as u16;
            if BPC == 8 {
                cfl_pred_impl::<false>(
                    $mode,
                    dst,
                    dst_stride,
                    edge,
                    width,
                    height,
                    ac,
                    alpha,
                    bitdepth_max,
                );
            } else {
                cfl_pred_impl::<true>(
                    $mode,
                    dst,
                    dst_stride,
                    edge,
                    width,
                    height,
                    ac,
                    alpha,
                    bitdepth_max,
                );
            }
        }
    };
}

cfl_pred_wrapper!(
    cfl_pred_dc,
    CflPredMode::Dc,
    "`cfl_pred[DC_PRED]` skaler çekirdeği."
);
cfl_pred_wrapper!(
    cfl_pred_left_dc,
    CflPredMode::LeftDc,
    "`cfl_pred[LEFT_DC_PRED]` skaler çekirdeği."
);
cfl_pred_wrapper!(
    cfl_pred_top_dc,
    CflPredMode::TopDc,
    "`cfl_pred[TOP_DC_PRED]` skaler çekirdeği."
);
cfl_pred_wrapper!(
    cfl_pred_dc_128,
    CflPredMode::Dc128,
    "`cfl_pred[DC_128_PRED]` skaler çekirdeği."
);

/// `pal_pred` skaler çekirdeği (dav1d `src/ipred_tmpl.c:723-735`).
///
/// # Panikler
///
/// Boyut, palet/indeks dilimi veya nibble sözleşmesi ihlal edilirse panikler.
pub fn pal_pred<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
    bit_depth: BitDepth,
) {
    assert!(
        (4..=MAX_DIM).contains(&w) && w.is_power_of_two(),
        "pal_pred genişliği 4..=64 ve 2'nin kuvveti olmalı: {w}"
    );
    assert!(
        (4..=MAX_DIM).contains(&h) && h.is_power_of_two(),
        "pal_pred yüksekliği 4..=64 ve 2'nin kuvveti olmalı: {h}"
    );
    debug_assert_eq!(bit_depth.bits(), BPC, "yuva bit derinliği ≠ çağrı");
    let bytes = if BPC == 8 { 1 } else { 2 };
    assert!(pal.len() >= 8 * bytes, "pal_pred palet dilimi kısa");
    assert!(
        idx.len() >= w * h / 2,
        "pal_pred indeks dilimi kısa: {} < {}",
        idx.len(),
        w * h / 2
    );
    let required = (h - 1) * dst_stride + w * bytes;
    assert!(dst.len() >= required, "pal_pred hedef dilimi kısa");
    let highbd = BPC > 8;
    let pal_get = |index: usize| -> i32 {
        if highbd {
            i32::from(u16::from_le_bytes([pal[index * 2], pal[index * 2 + 1]]))
        } else {
            i32::from(pal[index])
        }
    };
    let palette = [
        pal_get(0),
        pal_get(1),
        pal_get(2),
        pal_get(3),
        pal_get(4),
        pal_get(5),
        pal_get(6),
        pal_get(7),
    ];
    for y in 0..h {
        for x in (0..w).step_by(2) {
            let packed = idx[y * (w / 2) + x / 2];
            assert!(
                packed & 0x88 == 0,
                "pal_pred indeks nibble'ı 0..7 dışında: {packed:#04x}"
            );
            let left = palette[usize::from(packed & 7)];
            let right = palette[usize::from(packed >> 4)];
            if highbd {
                write_pixel::<true>(dst, y * dst_stride + x * 2, left);
                write_pixel::<true>(dst, y * dst_stride + (x + 1) * 2, right);
            } else {
                write_pixel::<false>(dst, y * dst_stride + x, left);
                write_pixel::<false>(dst, y * dst_stride + x + 1, right);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{get_filter_strength, get_upsample, required_range};
    #[cfg(feature = "std")]
    use crate::dsp::ipred::Edge;
    use crate::dsp::ipred::IntraPredMode;
    #[cfg(feature = "std")]
    use crate::frame::BitDepth;

    #[test]
    fn filtre_gucu_esikleri_dav1d_ile_ayni() {
        // `src/ipred_tmpl.c:333-368` tablosunun sınır örnekleri.
        assert_eq!(get_filter_strength(8, 56, false), 1);
        assert_eq!(get_filter_strength(8, 55, false), 0);
        assert_eq!(get_filter_strength(16, 40, false), 1);
        assert_eq!(get_filter_strength(24, 32, false), 3);
        assert_eq!(get_filter_strength(24, 16, false), 2);
        assert_eq!(get_filter_strength(24, 8, false), 1);
        assert_eq!(get_filter_strength(32, 32, false), 3);
        assert_eq!(get_filter_strength(32, 4, false), 2);
        assert_eq!(get_filter_strength(32, 3, false), 1);
        assert_eq!(get_filter_strength(33, 0, false), 3);
        assert_eq!(get_filter_strength(8, 64, true), 2);
        assert_eq!(get_filter_strength(8, 40, true), 1);
        assert_eq!(get_filter_strength(16, 48, true), 2);
        assert_eq!(get_filter_strength(16, 20, true), 1);
        assert_eq!(get_filter_strength(24, 4, true), 3);
        assert_eq!(get_filter_strength(25, 0, true), 3);
    }

    #[test]
    fn upsample_kosulu_dav1d_ile_ayni() {
        assert!(get_upsample(8, 39, false));
        assert!(!get_upsample(8, 40, false));
        assert!(get_upsample(16, 0, false));
        assert!(!get_upsample(17, 0, false));
        assert!(get_upsample(8, 0, true));
        assert!(!get_upsample(9, 0, true));
    }

    #[test]
    fn kenar_araligi_tablosu() {
        use IntraPredMode as M;
        assert_eq!(required_range(M::Dc, 8, 4, 0), (4, 8));
        assert_eq!(required_range(M::Dc128, 64, 64, 0), (0, 0));
        assert_eq!(required_range(M::Z1, 8, 4, 0), (0, 12));
        assert_eq!(required_range(M::Z2, 8, 4, 93), (4, 8));
        // Z3 bayraksız: h + min(w,h).
        assert_eq!(required_range(M::Z3, 8, 4, 183), (8, 0));
        // Z3 kenar filtresi etkin ve upsample uygun (angle-180 = 3): w + h.
        assert_eq!(required_range(M::Z3, 8, 4, 183 | (1 << 10)), (12, 0));
        assert_eq!(required_range(M::Z3, 8, 8, 183 | (1 << 10)), (16, 0));
        // Z3 upsample uygun değil (wh = 40 > 16), kenar filtresi gücü 3: w + h.
        assert_eq!(required_range(M::Z3, 32, 8, 203 | (1 << 10)), (40, 0));
    }

    #[cfg(feature = "std")]
    #[test]
    fn gecersiz_kenar_dilimi_panikler() {
        let bytes = [0u8; 4];
        let edge = Edge::new(&bytes, 2);
        let mut dst = [0u8; 16];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            super::intra_pred::<8>(
                IntraPredMode::Dc,
                &mut dst,
                4,
                edge,
                4,
                4,
                0,
                0,
                0,
                BitDepth::Eight,
                255,
            );
        }));
        assert!(result.is_err(), "kısa kenar dilimi panik üretmeliydi");
    }
}
