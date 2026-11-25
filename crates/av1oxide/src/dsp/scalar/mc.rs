//! MC skaler (referans) çekirdekleri — S2b.4: `Dav1dMCDSPContext`'in tamamı
//! (`resize` hariç).
//!
//! dav1d `src/mc_tmpl.c`'nin (`99f3354`, asm yok) portudur; her kuralın kaynak
//! satırı aşağıdaki tabloda ve kodun yanındadır. Amaç **bit-exact** davranıştır:
//! C'nin işaretli kaydırması (`>>` negatif değerde de aritmetiktir), `int16_t`
//! ara tamponu, yuvarlama sabitleri ve kırpmalar birebir taşınır.
//!
//! ## Bit derinliği monomorfizasyonu (dav1d `BITDEPTH`)
//!
//! Gövde `const BPC: u32` (8/10/12) ile **üç kez** monomorfize edilir; dav1d'de
//! 8-bit ve 16bpc olmak üzere iki derleme vardır ve 10/12 ayrımı yalnız
//! `bitdepth_max` ile yapılır. Bu portta 10/12 de ayrı monomorfize edilir, ama
//! sabitler dav1d'nin `bitdepth_max` türevleriyle **aynı**dır:
//!
//! | Büyüklük | 8-bit | 10-bit | 12-bit | Kaynak |
//! |---|---|---|---|---|
//! | `bpp` (bayt/piksel) | 1 | 2 | 2 | `include/common/bitdepth.h:31-79` |
//! | `intermediate_bits` (`ib`) | 4 | 4 | 2 | `src/mc_tmpl.c:40-45` |
//! | `PREP_BIAS` | 0 | 8192 | 8192 | `src/mc_tmpl.c:39-48` |
//! | `bitdepth_max` | 255 | 1023 | 4095 | `bitdepth_from_max` |
//!
//! ## Kaynak okuma aralıkları (test (b) — sentinel/over-read sözleşmesi)
//!
//! `src` **dolgu hücresinin** (blok satırı −3 / sütunu −3) başlangıcıdır:
//! dav1d `put_8tap_c` `src -= 3·src_stride` yapar ve 8-tap çekirdekleri
//! sütunda `x−3..x+4` okur; bu port aynı ofsetleri doğrudan uygular.
//! Tek istisna [`emu_edge`]'dir: orada `src` **referans düzleminin** sol
//! üstüdür (dav1d `emu_edge_c` sözleşmesi).
//!
//! | Çekirdek | Okunan kaynak (dolgu hücresi koordinatları) | Yazılan hedef |
//! |---|---|---|
//! | `put_8tap` / `prep_8tap` | satır `0..h+6`, sütun `0..w+6` (`h+7`×`w+7`) → dilim uzunluğu `(h+6)·src_stride + (w+7)·bpp` | `h` satır × `w` piksel |
//! | `put_8tap_scaled` / `prep_8tap_scaled` | satır `0 .. ((my+(h−1)·dy)>>10)+7`, sütun `0 .. ((mx+(w−1)·dx)>>10)+8` | `h` satır × `w` piksel |
//! | `put_bilin` / `prep_bilin` | satır `0..h+7`, sütun `0..w+7` | `h` satır × `w` piksel |
//! | `put_bilin_scaled` / `prep_bilin_scaled` | satır `0 .. ((my+(h−1)·dy)>>10)+4`, sütun `0 .. ((mx+(w−1)·dx)>>10)+5` | `h` satır × `w` piksel |
//! | `warp8x8` / `warp8x8t` | 15×15 pencere (dolgu hücresi `0..14`) | 8×8 (`t`: 8×8 `i16`) |
//! | `emu_edge` | `iclip(y,0,ih−1) .. +center_h`, `iclip(x,0,iw−1) .. +center_w` (düzlem koordinatı) | `bw`×`bh` |
//! | `avg` / `w_avg` / `mask` / `w_mask` | `w·h` `i16` (tmp1, tmp2), maske `w·h` bayt | `h`×`w` piksel (+ maske) |
//! | `blend` | `dst` `w·h`, `tmp` `w·h`, maske `w·h` | `h`×`w` piksel |
//! | `blend_v` | `dst`, `tmp`, `OBMC_MASKS[w..]` (`(w·3)>>2` bayt) | sol `(w·3)>>2` sütun |
//! | `blend_h` | `dst`, `tmp`, `OBMC_MASKS[h..]` (`(h·3)>>2` bayt) | üst `(h·3)>>2` satır |
//!
//! Sözleşme dışı girdi **panikler** (güvenli indeksleme).
//!
//! ## Tanımlı aritmetik (S2b.3 dersi)
//!
//! - dav1d'nin **işaretsiz/sarmalı** işlemleri burada açıkça `wrapping_*` ile
//!   yazılır (`abs` yerine `unsigned_abs`, `iclip` yerine `clamp`).
//! - Ara değerlerin `i16`'ya sığdığı **aralık yorumuyla** belgelenir ve
//!   `debug_assert` ile denetlenir: dav1d `src/mc_tmpl.c:41-48` yorumu 16bpc
//!   çıktının `[−20588, 36956]` (10-bit) / `[−20602, 36983]` (12-bit)
//!   aralığında olduğunu, `PREP_BIAS = 8192` çıkarınca `i16`'ya sığdığını
//!   söyler. 8-bit'te ara değerler `[−2140, 4100]` aralığındadır.
//! - `>>` negatif değerlerde Rust'ta da aritmetiktir (C ile aynı).
//!
//! ## Kapsam dışı
//!
//! `resize` (superres, P16) bu modülde **yoktur**.

#![forbid(unsafe_code)]

use crate::dsp::mc::{Filter2d, MC_MAX_W, McBitDepth};
use crate::dsp::mc_tables::{
    MC_SUBPEL_FILTERS, MC_TAPS, MC_WARP_FILTER, N_WARP_FILTERS, OBMC_MASKS,
};
use crate::frame::BitDepth;

/// Piksel başına bayt (8-bit → 1, 16bpc → 2).
#[inline(always)]
const fn bpp_of(bits: u32) -> usize {
    if bits == 8 { 1 } else { 2 }
}

/// dav1d `get_intermediate_bits`: 8-bit → 4, 16bpc → `14 − bitdepth`.
#[inline(always)]
const fn intermediate_bits_of(bits: u32) -> u32 {
    if bits == 8 { 4 } else { 14 - bits }
}

/// dav1d `PREP_BIAS`: 8-bit → 0, 16bpc → 8192.
#[inline(always)]
const fn prep_bias_of(bits: u32) -> i32 {
    if bits == 8 { 0 } else { 8192 }
}

/// Piksel baytı → `i32` (16bpc'de **yerel** bayt sırası; dav1d `pixel`).
#[inline(always)]
fn load<const BPC: u32>(buf: &[u8], off: usize) -> i32 {
    if BPC == 8 {
        i32::from(buf[off])
    } else {
        i32::from(u16::from_ne_bytes([buf[off], buf[off + 1]]))
    }
}

/// `i32` → piksel baytı (değer önce kırpılır; dav1d `iclip_pixel`).
#[inline(always)]
fn store<const BPC: u32>(buf: &mut [u8], off: usize, value: i32) {
    debug_assert!((0..(1i32 << BPC)).contains(&value), "piksel aralık dışı");
    if BPC == 8 {
        buf[off] = value as u8;
    } else {
        buf[off..off + 2].copy_from_slice(&(value as u16).to_ne_bytes());
    }
}

/// dav1d `iclip_pixel`.
#[inline(always)]
fn iclip_pixel(value: i32, max: i32) -> i32 {
    value.clamp(0, max)
}

/// `DAV1D_FILTER_8TAP_RND(src,x,F,stride,sh) = (sum + ((1<<sh)>>1)) >> sh`.
#[inline(always)]
fn rnd_shift(sum: i32, shift: u32) -> i32 {
    (sum.wrapping_add((1i32 << shift) >> 1)) >> shift
}

/// `DAV1D_FILTER_8TAP_RND2 = (sum + rnd) >> sh` (açık yuvarlama sabiti).
#[inline(always)]
fn rnd2_shift(sum: i32, rnd: i32, shift: u32) -> i32 {
    sum.wrapping_add(rnd) >> shift
}

/// dav1d `GET_H_FILTER(mx)` (`src/mc_tmpl.c:115-118`).
#[inline]
fn h_filter(filter: Filter2d, w: usize, mx: u8) -> Option<&'static [i8; MC_TAPS]> {
    if mx == 0 {
        return None;
    }
    let k = filter.horizontal_table();
    let table = if w > 4 { k } else { 3 + (k & 1) };
    Some(&MC_SUBPEL_FILTERS[table][usize::from(mx) - 1])
}

/// dav1d `GET_V_FILTER(my)` (`src/mc_tmpl.c:120-123`).
#[inline]
fn v_filter(filter: Filter2d, h: usize, my: u8) -> Option<&'static [i8; MC_TAPS]> {
    if my == 0 {
        return None;
    }
    let k = filter.vertical_table();
    let table = if h > 4 { k } else { 3 + (k & 1) };
    Some(&MC_SUBPEL_FILTERS[table][usize::from(my) - 1])
}

/// `FILTER_8TAP(src, x, F, 1)` — `off` tap 0'ın (blok sütunu `x−3`) ofsetidir.
#[inline(always)]
fn filter_h<const BPC: u32>(src: &[u8], off: usize, f: &[i8; MC_TAPS]) -> i32 {
    let bpp = bpp_of(BPC);
    let mut sum = 0i32;
    for (k, &coef) in f.iter().enumerate() {
        sum = sum.wrapping_add(i32::from(coef).wrapping_mul(load::<BPC>(src, off + k * bpp)));
    }
    sum
}

/// `FILTER_8TAP(src, x, F, stride)` — tap 0 `off`, satır adımı `stride`.
#[inline(always)]
fn filter_v<const BPC: u32>(src: &[u8], off: usize, stride: usize, f: &[i8; MC_TAPS]) -> i32 {
    let mut sum = 0i32;
    for (k, &coef) in f.iter().enumerate() {
        sum = sum.wrapping_add(i32::from(coef).wrapping_mul(load::<BPC>(src, off + k * stride)));
    }
    sum
}

/// `mid` halkası: 8 satır (dav1d `int16_t mid[128*135]`'in kayan penceresi).
const MID_RING: usize = 8;

/// `mid` halkasının bir satırını üretir — dav1d `src/mc_tmpl.c:149-151,262-264`.
///
/// `row` **dolgu hücresi** satırıdır (0 = blok satırı −3). Yatay tap'ler
/// dolgu sütunu `x..x+7`'dir (blok sütunu `x−3..x+4`).
#[allow(clippy::too_many_arguments)]
#[inline]
fn fill_mid_row<const BPC: u32>(
    mid: &mut [i16; MID_RING * MC_MAX_W],
    row: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    filter: &[i8; MC_TAPS],
    rnd: i32,
    shift: u32,
) {
    let slot = row & (MID_RING - 1);
    let base = row * src_stride;
    for x in 0..w {
        let value = rnd2_shift(
            filter_h::<BPC>(src, base + x * bpp_of(BPC), filter),
            rnd,
            shift,
        );
        debug_assert!(
            (i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&value),
            "ara değer int16'ya sığmalı (dav1d int16_t mid)"
        );
        mid[slot * MC_MAX_W + x] = value as i16;
    }
}

/// `put_8tap_c` — dav1d `src/mc_tmpl.c:129-187`.
#[allow(clippy::too_many_arguments)]
fn put_8tap_bpc<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    filter: Filter2d,
) {
    if w == 0 || h == 0 {
        return;
    }
    let ib = intermediate_bits_of(BPC);
    let shift_mid = 6 - ib;
    let rnd_mid = (1i32 << shift_mid) >> 1;
    let shift_vert = 6 + ib;
    let rnd_vert = (1i32 << shift_vert) >> 1;
    let rnd_h_only = 32 + ((1i32 << shift_mid) >> 1);
    let max_pixel = (1i32 << BPC) - 1;
    let bpp = bpp_of(BPC);

    let fh = h_filter(filter, w, mx);
    let fv = v_filter(filter, h, my);

    match (fh, fv) {
        // ---- iki boyutlu: yatay ara tampon + dikey ----
        // dav1d `src/mc_tmpl.c:144-172`.
        (Some(fh), Some(fv)) => {
            let mut mid = [0i16; MID_RING * MC_MAX_W];
            for row in 0..MID_RING {
                fill_mid_row::<BPC>(&mut mid, row, src, src_stride, w, fh, rnd_mid, shift_mid);
            }
            for y in 0..h {
                let drow = y * dst_stride;
                for x in 0..w {
                    let mut sum = 0i32;
                    for k in 0..MID_RING {
                        let m = i32::from(mid[((y + k) & (MID_RING - 1)) * MC_MAX_W + x]);
                        sum = sum.wrapping_add(i32::from(fv[k]).wrapping_mul(m));
                    }
                    let value = rnd2_shift(sum, rnd_vert, shift_vert);
                    store::<BPC>(dst, drow + x * bpp, iclip_pixel(value, max_pixel));
                }
                let next = y + MID_RING;
                if next < h + 7 {
                    fill_mid_row::<BPC>(&mut mid, next, src, src_stride, w, fh, rnd_mid, shift_mid);
                }
            }
        }
        // ---- yalnız yatay: `(sum + intermediate_rnd) >> 6` ----
        // dav1d `src/mc_tmpl.c:173-181`.
        (Some(fh), None) => {
            for y in 0..h {
                let srow = (y + 3) * src_stride;
                let drow = y * dst_stride;
                for x in 0..w {
                    let value = rnd2_shift(filter_h::<BPC>(src, srow + x * bpp, fh), rnd_h_only, 6);
                    store::<BPC>(dst, drow + x * bpp, value.clamp(0, max_pixel));
                }
            }
        }
        // ---- yalnız dikey: `(sum + 32) >> 6` ----
        // dav1d `src/mc_tmpl.c:182-185`.
        (None, Some(fv)) => {
            for y in 0..h {
                let drow = y * dst_stride;
                for x in 0..w {
                    let base = y * src_stride + (x + 3) * bpp;
                    let value = rnd_shift(filter_v::<BPC>(src, base, src_stride, fv), 6);
                    store::<BPC>(dst, drow + x * bpp, value.clamp(0, max_pixel));
                }
            }
        }
        // ---- kopya: dav1d `put_c` ----
        // dav1d `src/mc_tmpl.c:51-60,186`.
        (None, None) => {
            let row = w * bpp;
            for y in 0..h {
                let s = (y + 3) * src_stride + 3 * bpp;
                let d = y * dst_stride;
                dst[d..d + row].copy_from_slice(&src[s..s + row]);
            }
        }
    }
}

/// `prep_8tap_c` — dav1d `src/mc_tmpl.c:246-305`.
#[allow(clippy::too_many_arguments)]
fn prep_8tap_bpc<const BPC: u32>(
    tmp: &mut [i16],
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    filter: Filter2d,
) {
    if w == 0 || h == 0 {
        return;
    }
    let ib = intermediate_bits_of(BPC);
    let shift_mid = 6 - ib;
    let rnd_mid = (1i32 << shift_mid) >> 1;
    let bias = prep_bias_of(BPC);
    let bpp = bpp_of(BPC);

    let fh = h_filter(filter, w, mx);
    let fv = v_filter(filter, h, my);

    match (fh, fv) {
        (Some(fh), Some(fv)) => {
            let mut mid = [0i16; MID_RING * MC_MAX_W];
            for row in 0..MID_RING {
                fill_mid_row::<BPC>(&mut mid, row, src, src_stride, w, fh, rnd_mid, shift_mid);
            }
            for y in 0..h {
                for x in 0..w {
                    let mut sum = 0i32;
                    for k in 0..MID_RING {
                        let m = i32::from(mid[((y + k) & (MID_RING - 1)) * MC_MAX_W + x]);
                        sum = sum.wrapping_add(i32::from(fv[k]).wrapping_mul(m));
                    }
                    let value = rnd2_shift(sum, 32, 6).wrapping_sub(bias);
                    store_tmp(tmp, y * w + x, value);
                }
                let next = y + MID_RING;
                if next < h + 7 {
                    fill_mid_row::<BPC>(&mut mid, next, src, src_stride, w, fh, rnd_mid, shift_mid);
                }
            }
        }
        (Some(fh), None) => {
            for y in 0..h {
                let srow = (y + 3) * src_stride;
                for x in 0..w {
                    let value =
                        rnd2_shift(filter_h::<BPC>(src, srow + x * bpp, fh), rnd_mid, shift_mid)
                            .wrapping_sub(bias);
                    store_tmp(tmp, y * w + x, value);
                }
            }
        }
        (None, Some(fv)) => {
            for y in 0..h {
                for x in 0..w {
                    let base = y * src_stride + (x + 3) * bpp;
                    let value = rnd2_shift(
                        filter_v::<BPC>(src, base, src_stride, fv),
                        rnd_mid,
                        shift_mid,
                    )
                    .wrapping_sub(bias);
                    store_tmp(tmp, y * w + x, value);
                }
            }
        }
        (None, None) => {
            // `prep_c`: `tmp = (src << ib) - PREP_BIAS` (8-bit'te bias 0).
            for y in 0..h {
                let srow = (y + 3) * src_stride + 3 * bpp;
                for x in 0..w {
                    let p = load::<BPC>(src, srow + x * bpp);
                    store_tmp(tmp, y * w + x, (p << ib) - bias);
                }
            }
        }
    }
}

/// Geçici `i16` yazımı (aralık denetimli).
#[inline(always)]
fn store_tmp(tmp: &mut [i16], index: usize, value: i32) {
    debug_assert!(
        (i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&value),
        "ara değer int16'ya sığmalı (dav1d int16_t tmp)"
    );
    tmp[index] = value as i16;
}

/// `put_8tap_scaled_c` — dav1d `src/mc_tmpl.c:189-244`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // x = piksel sütunu (dav1d döngüsü birebir)
fn put_8tap_scaled_bpc<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    mut my: i32,
    dx: i32,
    dy: i32,
    filter: Filter2d,
) {
    if w == 0 || h == 0 {
        return;
    }
    let ib = intermediate_bits_of(BPC);
    let shift_mid = 6 - ib;
    let rnd_mid = (1i32 << shift_mid) >> 1;
    let shift_vert = 6 + ib;
    let rnd_vert = (1i32 << shift_vert) >> 1;
    let intermediate_rnd = (1i32 << ib) >> 1;
    let max_pixel = (1i32 << BPC) - 1;
    let bpp = bpp_of(BPC);
    let mut mid = [[0i16; MC_MAX_W]; MID_RING];
    let mut start = 0usize;
    let mut in_y: i32 = -8;
    let imx = mx;

    for y in 0..h {
        let src_y = my >> 10;
        let fv = v_filter(filter, h, ((my & 0x3ff) >> 6) as u8);
        while in_y < src_y {
            let mut gen_x = imx;
            let mut ioff: i32 = 0;
            // Yeni satır, eski mantıksal 0'ın (en eski) yerine yazılır
            // (dav1d `mid_ptrs[7] = mid_ptr` döndürmesi).
            let slot = start;
            // Üretim satırı `in_y` → dolgu satırı `in_y + 8` (dav1d
            // `src -= 3*src_stride` sonrası blok satırı `in_y + 5`).
            let base = (in_y + 8) as usize * src_stride;
            for x in 0..w {
                let fh = h_filter(filter, w, (gen_x >> 6) as u8);
                let value = match fh {
                    // `FILTER_8TAP(src, ioff, fh, 1)`: tap'ler dolgu sütunu
                    // `ioff .. ioff+7` (blok sütunu `ioff−3 .. ioff+4`).
                    Some(f) => rnd2_shift(
                        filter_h::<BPC>(src, base + ioff as usize * bpp, f),
                        rnd_mid,
                        shift_mid,
                    ),
                    // Filtre yok (alt piksel 0): `src[ioff]` = blok sütunu
                    // `ioff` = dolgu sütunu `ioff + 3`.
                    None => load::<BPC>(src, base + (ioff as usize + 3) * bpp) << ib,
                };
                debug_assert!(
                    (i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&value),
                    "ara değer int16'ya sığmalı"
                );
                mid[slot][x] = value as i16;
                gen_x += dx;
                ioff += gen_x >> 10;
                gen_x &= 0x3ff;
            }
            start = (start + 1) % MID_RING;
            in_y += 1;
        }
        let drow = y * dst_stride;
        for x in 0..w {
            let value = match fv {
                Some(f) => {
                    let mut sum = 0i32;
                    for (k, &coef) in f.iter().enumerate() {
                        let m = i32::from(mid[(start + k) % MID_RING][x]);
                        sum = sum.wrapping_add(i32::from(coef).wrapping_mul(m));
                    }
                    rnd2_shift(sum, rnd_vert, shift_vert)
                }
                None => {
                    let m = i32::from(mid[(start + 3) % MID_RING][x]);
                    (m.wrapping_add(intermediate_rnd)) >> ib
                }
            };
            store::<BPC>(dst, drow + x * bpp, iclip_pixel(value, max_pixel));
        }
        my += dy;
    }
}

/// `prep_8tap_scaled_c` — dav1d `src/mc_tmpl.c:307-358`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // x = piksel sütunu (dav1d döngüsü birebir)
fn prep_8tap_scaled_bpc<const BPC: u32>(
    tmp: &mut [i16],
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    mut my: i32,
    dx: i32,
    dy: i32,
    filter: Filter2d,
) {
    if w == 0 || h == 0 {
        return;
    }
    let ib = intermediate_bits_of(BPC);
    let shift_mid = 6 - ib;
    let rnd_mid = (1i32 << shift_mid) >> 1;
    let bias = prep_bias_of(BPC);
    let bpp = bpp_of(BPC);
    let mut mid = [[0i16; MC_MAX_W]; MID_RING];
    let mut start = 0usize;
    let mut in_y: i32 = -8;
    let imx = mx;

    for y in 0..h {
        let src_y = my >> 10;
        let fv = v_filter(filter, h, ((my & 0x3ff) >> 6) as u8);
        while in_y < src_y {
            let mut gen_x = imx;
            let mut ioff: i32 = 0;
            let slot = start;
            let base = (in_y + 8) as usize * src_stride;
            for x in 0..w {
                let fh = h_filter(filter, w, (gen_x >> 6) as u8);
                let value = match fh {
                    Some(f) => rnd2_shift(
                        filter_h::<BPC>(src, base + ioff as usize * bpp, f),
                        rnd_mid,
                        shift_mid,
                    ),
                    // Filtre yok (alt piksel 0): `src[ioff]` = blok sütunu
                    // `ioff` = dolgu sütunu `ioff + 3`.
                    None => load::<BPC>(src, base + (ioff as usize + 3) * bpp) << ib,
                };
                mid[slot][x] = value as i16;
                gen_x += dx;
                ioff += gen_x >> 10;
                gen_x &= 0x3ff;
            }
            start = (start + 1) % MID_RING;
            in_y += 1;
        }
        for x in 0..w {
            let value = match fv {
                Some(f) => {
                    let mut sum = 0i32;
                    for (k, &coef) in f.iter().enumerate() {
                        let m = i32::from(mid[(start + k) % MID_RING][x]);
                        sum = sum.wrapping_add(i32::from(coef).wrapping_mul(m));
                    }
                    rnd2_shift(sum, 32, 6)
                }
                None => i32::from(mid[(start + 3) % MID_RING][x]),
            }
            .wrapping_sub(bias);
            store_tmp(tmp, y * w + x, value);
        }
        my += dy;
    }
}

/// `FILTER_BILIN_RND(src,x,mxy,stride,sh)` — `16·a + m·(b−a)`, yuvarlanmış.
#[inline(always)]
fn bilin_rnd(a: i32, b: i32, m: i32, shift: u32) -> i32 {
    let v = 16 * a + m * (b - a);
    (v.wrapping_add((1i32 << shift) >> 1)) >> shift
}

/// `put_bilin_c` — dav1d `src/mc_tmpl.c:434-489`.
#[allow(clippy::too_many_arguments)]
fn put_bilin_bpc<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    if w == 0 || h == 0 {
        return;
    }
    let ib = intermediate_bits_of(BPC);
    let intermediate_rnd = (1i32 << ib) >> 1;
    let max_pixel = (1i32 << BPC) - 1;
    let bpp = bpp_of(BPC);

    if mx != 0 {
        let mxv = i32::from(mx);
        if my != 0 {
            let myv = i32::from(my);
            // dav1d `int16_t mid[128*129]`: `h+1` ara satır.
            let mut mid = vec![0i16; (h + 1) * w];
            for row in 0..=h {
                let base = (row + 3) * src_stride + 3 * bpp;
                for x in 0..w {
                    let p0 = load::<BPC>(src, base + x * bpp);
                    let p1 = load::<BPC>(src, base + (x + 1) * bpp);
                    mid[row * w + x] = bilin_rnd(p0, p1, mxv, 4 - ib) as i16;
                }
            }
            for y in 0..h {
                let drow = y * dst_stride;
                for x in 0..w {
                    let p0 = i32::from(mid[y * w + x]);
                    let p1 = i32::from(mid[(y + 1) * w + x]);
                    let value = bilin_rnd(p0, p1, myv, 4 + ib);
                    store::<BPC>(dst, drow + x * bpp, iclip_pixel(value, max_pixel));
                }
            }
        } else {
            for y in 0..h {
                let srow = (y + 3) * src_stride + 3 * bpp;
                let drow = y * dst_stride;
                for x in 0..w {
                    let p0 = load::<BPC>(src, srow + x * bpp);
                    let p1 = load::<BPC>(src, srow + (x + 1) * bpp);
                    let px = bilin_rnd(p0, p1, mxv, 4 - ib);
                    let value = (px.wrapping_add(intermediate_rnd)) >> ib;
                    store::<BPC>(dst, drow + x * bpp, iclip_pixel(value, max_pixel));
                }
            }
        }
    } else if my != 0 {
        let myv = i32::from(my);
        for y in 0..h {
            let drow = y * dst_stride;
            for x in 0..w {
                let base = (y + 3) * src_stride + (x + 3) * bpp;
                let p0 = load::<BPC>(src, base);
                let p1 = load::<BPC>(src, base + src_stride);
                let value = bilin_rnd(p0, p1, myv, 4);
                store::<BPC>(dst, drow + x * bpp, iclip_pixel(value, max_pixel));
            }
        }
    } else {
        let row = w * bpp;
        for y in 0..h {
            let s = (y + 3) * src_stride + 3 * bpp;
            let d = y * dst_stride;
            dst[d..d + row].copy_from_slice(&src[s..s + row]);
        }
    }
}

/// `prep_bilin_c` — dav1d `src/mc_tmpl.c:533-586`.
#[allow(clippy::too_many_arguments)]
fn prep_bilin_bpc<const BPC: u32>(
    tmp: &mut [i16],
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    if w == 0 || h == 0 {
        return;
    }
    let ib = intermediate_bits_of(BPC);
    let bias = prep_bias_of(BPC);
    let bpp = bpp_of(BPC);

    if mx != 0 {
        let mxv = i32::from(mx);
        if my != 0 {
            let myv = i32::from(my);
            let mut mid = vec![0i16; (h + 1) * w];
            for row in 0..=h {
                let base = (row + 3) * src_stride + 3 * bpp;
                for x in 0..w {
                    let p0 = load::<BPC>(src, base + x * bpp);
                    let p1 = load::<BPC>(src, base + (x + 1) * bpp);
                    mid[row * w + x] = bilin_rnd(p0, p1, mxv, 4 - ib) as i16;
                }
            }
            for y in 0..h {
                for x in 0..w {
                    let p0 = i32::from(mid[y * w + x]);
                    let p1 = i32::from(mid[(y + 1) * w + x]);
                    let value = bilin_rnd(p0, p1, myv, 4).wrapping_sub(bias);
                    store_tmp(tmp, y * w + x, value);
                }
            }
        } else {
            for y in 0..h {
                let srow = (y + 3) * src_stride + 3 * bpp;
                for x in 0..w {
                    let p0 = load::<BPC>(src, srow + x * bpp);
                    let p1 = load::<BPC>(src, srow + (x + 1) * bpp);
                    let value = bilin_rnd(p0, p1, mxv, 4 - ib).wrapping_sub(bias);
                    store_tmp(tmp, y * w + x, value);
                }
            }
        }
    } else if my != 0 {
        let myv = i32::from(my);
        for y in 0..h {
            for x in 0..w {
                let base = (y + 3) * src_stride + (x + 3) * bpp;
                let p0 = load::<BPC>(src, base);
                let p1 = load::<BPC>(src, base + src_stride);
                let value = bilin_rnd(p0, p1, myv, 4 - ib).wrapping_sub(bias);
                store_tmp(tmp, y * w + x, value);
            }
        }
    } else {
        for y in 0..h {
            let srow = (y + 3) * src_stride + 3 * bpp;
            for x in 0..w {
                let p = load::<BPC>(src, srow + x * bpp);
                store_tmp(tmp, y * w + x, (p << ib) - bias);
            }
        }
    }
}

/// `put_bilin_scaled_c` — dav1d `src/mc_tmpl.c:491-531`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // x = piksel sütunu (dav1d döngüsü birebir)
fn put_bilin_scaled_bpc<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    mut my: i32,
    dx: i32,
    dy: i32,
) {
    if w == 0 || h == 0 {
        return;
    }
    let ib = intermediate_bits_of(BPC);
    let max_pixel = (1i32 << BPC) - 1;
    let bpp = bpp_of(BPC);
    let mut mid = [[0i16; MC_MAX_W]; 2];
    let mut in_y: i32 = -2;
    let imx = mx;

    for y in 0..h {
        let src_y = my >> 10;
        let dmy = my & 0x3ff;
        while in_y < src_y {
            let slot = (in_y & 1) as usize;
            let mut gen_x = imx;
            let mut ioff: i32 = 0;
            // Üretim satırı `in_y` → dolgu satırı `in_y + 5` (dav1d bu yolda
            // `src`'yi geri almaz; blok satırı `in_y + 2`).
            let base = (in_y + 5) as usize * src_stride + 3 * bpp;
            for x in 0..w {
                let p0 = load::<BPC>(src, base + ioff as usize * bpp);
                let p1 = load::<BPC>(src, base + (ioff as usize + 1) * bpp);
                mid[slot][x] = bilin_rnd(p0, p1, gen_x >> 6, 4 - ib) as i16;
                gen_x += dx;
                ioff += gen_x >> 10;
                gen_x &= 0x3ff;
            }
            in_y += 1;
        }
        let drow = y * dst_stride;
        for x in 0..w {
            let p0 = i32::from(mid[(src_y & 1) as usize][x]);
            let p1 = i32::from(mid[((src_y + 1) & 1) as usize][x]);
            let value = bilin_rnd(p0, p1, dmy >> 6, 4 + ib);
            store::<BPC>(dst, drow + x * bpp, iclip_pixel(value, max_pixel));
        }
        my += dy;
    }
}

/// `prep_bilin_scaled_c` — dav1d `src/mc_tmpl.c:588-626`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // x = piksel sütunu (dav1d döngüsü birebir)
fn prep_bilin_scaled_bpc<const BPC: u32>(
    tmp: &mut [i16],
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    mut my: i32,
    dx: i32,
    dy: i32,
) {
    if w == 0 || h == 0 {
        return;
    }
    let ib = intermediate_bits_of(BPC);
    let bias = prep_bias_of(BPC);
    let bpp = bpp_of(BPC);
    let mut mid = [[0i16; MC_MAX_W]; 2];
    let mut in_y: i32 = -2;
    let imx = mx;

    for y in 0..h {
        let src_y = my >> 10;
        let dmy = my & 0x3ff;
        while in_y < src_y {
            let slot = (in_y & 1) as usize;
            let mut gen_x = imx;
            let mut ioff: i32 = 0;
            let base = (in_y + 5) as usize * src_stride + 3 * bpp;
            for x in 0..w {
                let p0 = load::<BPC>(src, base + ioff as usize * bpp);
                let p1 = load::<BPC>(src, base + (ioff as usize + 1) * bpp);
                mid[slot][x] = bilin_rnd(p0, p1, gen_x >> 6, 4 - ib) as i16;
                gen_x += dx;
                ioff += gen_x >> 10;
                gen_x &= 0x3ff;
            }
            in_y += 1;
        }
        for x in 0..w {
            let p0 = i32::from(mid[(src_y & 1) as usize][x]);
            let p1 = i32::from(mid[((src_y + 1) & 1) as usize][x]);
            let value = bilin_rnd(p0, p1, dmy >> 6, 4).wrapping_sub(bias);
            store_tmp(tmp, y * w + x, value);
        }
        my += dy;
    }
}

/// `avg_c` — dav1d `src/mc_tmpl.c:628-643`.
fn avg_bpc<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
) {
    let ib = intermediate_bits_of(BPC);
    let shift = ib + 1;
    let rnd = (1i32 << ib) + prep_bias_of(BPC) * 2;
    let max_pixel = (1i32 << BPC) - 1;
    let bpp = bpp_of(BPC);
    for y in 0..h {
        let drow = y * dst_stride;
        for x in 0..w {
            let a = i32::from(tmp1[y * w + x]);
            let b = i32::from(tmp2[y * w + x]);
            let value = (a.wrapping_add(b).wrapping_add(rnd)) >> shift;
            store::<BPC>(dst, drow + x * bpp, iclip_pixel(value, max_pixel));
        }
    }
}

/// `w_avg_c` — dav1d `src/mc_tmpl.c:645-661`.
fn w_avg_bpc<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
) {
    let ib = intermediate_bits_of(BPC);
    let shift = ib + 4;
    let rnd = (8i32 << ib) + prep_bias_of(BPC) * 16;
    let max_pixel = (1i32 << BPC) - 1;
    let bpp = bpp_of(BPC);
    for y in 0..h {
        let drow = y * dst_stride;
        for x in 0..w {
            let a = i32::from(tmp1[y * w + x]);
            let b = i32::from(tmp2[y * w + x]);
            let value = (a
                .wrapping_mul(weight)
                .wrapping_add(b.wrapping_mul(16 - weight))
                .wrapping_add(rnd))
                >> shift;
            store::<BPC>(dst, drow + x * bpp, iclip_pixel(value, max_pixel));
        }
    }
}

/// `mask_c` — dav1d `src/mc_tmpl.c:663-680`.
#[allow(clippy::too_many_arguments)]
fn mask_bpc<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &[u8],
) {
    let ib = intermediate_bits_of(BPC);
    let shift = ib + 6;
    let rnd = (32i32 << ib) + prep_bias_of(BPC) * 64;
    let max_pixel = (1i32 << BPC) - 1;
    let bpp = bpp_of(BPC);
    for y in 0..h {
        let drow = y * dst_stride;
        for x in 0..w {
            let m = i32::from(mask[y * w + x]);
            let a = i32::from(tmp1[y * w + x]);
            let b = i32::from(tmp2[y * w + x]);
            let value = (a
                .wrapping_mul(m)
                .wrapping_add(b.wrapping_mul(64 - m))
                .wrapping_add(rnd))
                >> shift;
            store::<BPC>(dst, drow + x * bpp, iclip_pixel(value, max_pixel));
        }
    }
}

/// `w_mask_c` — dav1d `src/mc_tmpl.c:724-767`.
///
/// `ss_hor` / `ss_ver`: 4:4:4 → (0,0), 4:2:2 → (1,0), 4:2:0 → (1,1).
/// Maske tamponu **sıfırlanmış** olmalıdır (dav1d çağıranı da öyle verir).
#[allow(clippy::too_many_arguments)]
fn w_mask_bpc<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &mut [u8],
    sign: i32,
    ss_hor: bool,
    ss_ver: bool,
) {
    let ib = intermediate_bits_of(BPC);
    let shift = ib + 6;
    let rnd = (32i32 << ib) + prep_bias_of(BPC) * 64;
    let max_pixel = (1i32 << BPC) - 1;
    let bpp = bpp_of(BPC);
    let mask_sh = BPC + ib - 4;
    let mask_rnd = 1i32 << (mask_sh - 5);
    let mask_row = w >> usize::from(ss_hor);
    let mut mask_off = 0usize;
    let mut rows = h;
    while rows > 0 {
        let y = h - rows;
        let drow = y * dst_stride;
        let mut x = 0;
        while x < w {
            let t1 = i32::from(tmp1[y * w + x]);
            let t2 = i32::from(tmp2[y * w + x]);
            let tmpdiff = t1 - t2;
            let m = (38 + ((tmpdiff.unsigned_abs() as i32 + mask_rnd) >> mask_sh)).min(64);
            let value = (tmpdiff
                .wrapping_mul(m)
                .wrapping_add(t2.wrapping_mul(64))
                .wrapping_add(rnd))
                >> shift;
            store::<BPC>(dst, drow + x * bpp, iclip_pixel(value, max_pixel));

            if ss_hor {
                x += 1;
                let t1b = i32::from(tmp1[y * w + x]);
                let t2b = i32::from(tmp2[y * w + x]);
                let diff_b = t1b - t2b;
                let n = (38 + ((diff_b.unsigned_abs() as i32 + mask_rnd) >> mask_sh)).min(64);
                let value_b = (diff_b
                    .wrapping_mul(n)
                    .wrapping_add(t2b.wrapping_mul(64))
                    .wrapping_add(rnd))
                    >> shift;
                store::<BPC>(dst, drow + x * bpp, iclip_pixel(value_b, max_pixel));

                // dav1d `if (h & ss_ver)` — `ss_ver == 0` iken **her zaman**
                // yanlıştır (h'nin tekliği tek başına yetmez).
                if ss_ver && rows & 1 != 0 {
                    let idx = mask_off + (x >> 1);
                    mask[idx] = ((m + n + i32::from(mask[idx]) + 2 - sign) >> 2) as u8;
                } else if ss_ver {
                    mask[mask_off + (x >> 1)] = (m + n) as u8;
                } else {
                    mask[mask_off + (x >> 1)] = ((m + n + 1 - sign) >> 1) as u8;
                }
            } else {
                mask[mask_off + x] = m as u8;
            }
            x += 1;
        }
        if !ss_ver || rows & 1 != 0 {
            mask_off += mask_row;
        }
        rows -= 1;
    }
}

/// `blend_px` — dav1d `src/mc_tmpl.c:682`.
#[inline(always)]
fn blend_px(a: i32, b: i32, m: i32) -> i32 {
    (a * (64 - m) + b * m + 32) >> 6
}

/// `blend_c` — dav1d `src/mc_tmpl.c:683-694`.
#[allow(clippy::too_many_arguments)]
fn blend_bpc<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    mask: &[u8],
) {
    let bpp = bpp_of(BPC);
    for y in 0..h {
        let drow = y * dst_stride;
        for x in 0..w {
            let a = load::<BPC>(dst, drow + x * bpp);
            let b = load::<BPC>(tmp, (y * w + x) * bpp);
            let m = i32::from(mask[y * w + x]);
            store::<BPC>(dst, drow + x * bpp, blend_px(a, b, m));
        }
    }
}

/// `blend_v_c` — dav1d `src/mc_tmpl.c:696-707` (sol `(w·3)>>2` sütun).
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn blend_v_bpc<const BPC: u32>(dst: &mut [u8], dst_stride: usize, tmp: &[u8], w: usize, h: usize) {
    let bpp = bpp_of(BPC);
    debug_assert!(w <= 32, "OBMC maskesi w ≤ 32 için tanımlı");
    let mask = &OBMC_MASKS[w..];
    let cols = (w * 3) >> 2;
    for y in 0..h {
        let drow = y * dst_stride;
        for x in 0..cols {
            let a = load::<BPC>(dst, drow + x * bpp);
            let b = load::<BPC>(tmp, (y * w + x) * bpp);
            store::<BPC>(dst, drow + x * bpp, blend_px(a, b, i32::from(mask[x])));
        }
    }
}

/// `blend_h_c` — dav1d `src/mc_tmpl.c:709-722` (üst `(h·3)>>2` satır).
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn blend_h_bpc<const BPC: u32>(dst: &mut [u8], dst_stride: usize, tmp: &[u8], w: usize, h: usize) {
    let bpp = bpp_of(BPC);
    debug_assert!(h <= 32, "OBMC maskesi h ≤ 32 için tanımlı");
    let mask = &OBMC_MASKS[h..];
    let rows = (h * 3) >> 2;
    for y in 0..rows {
        let m = i32::from(mask[y]);
        let drow = y * dst_stride;
        for x in 0..w {
            let a = load::<BPC>(dst, drow + x * bpp);
            let b = load::<BPC>(tmp, (y * w + x) * bpp);
            store::<BPC>(dst, drow + x * bpp, blend_px(a, b, m));
        }
    }
}

/// `FILTER_WARP_RND` — dav1d `src/mc_tmpl.c:785-794`.
#[inline(always)]
fn warp_filter_h<const BPC: u32>(src: &[u8], off: usize, f: &[i8; 8], shift: u32) -> i32 {
    let bpp = bpp_of(BPC);
    let mut sum = 0i32;
    for (k, &coef) in f.iter().enumerate() {
        sum = sum.wrapping_add(i32::from(coef).wrapping_mul(load::<BPC>(src, off + k * bpp)));
    }
    (sum.wrapping_add((1i32 << shift) >> 1)) >> shift
}

/// `FILTER_WARP_RND(mid_ptr, x, F, 8, sh)` — `i16` ara satırlarda dikey.
#[inline(always)]
fn warp_filter_v(mid: &[i16; 15 * 8], row: usize, x: usize, f: &[i8; 8], shift: u32) -> i32 {
    let mut sum = 0i32;
    for (k, &coef) in f.iter().enumerate() {
        sum = sum.wrapping_add(i32::from(coef).wrapping_mul(i32::from(mid[(row + k) * 8 + x])));
    }
    (sum.wrapping_add((1i32 << shift) >> 1)) >> shift
}

/// Warp filtre indeksi: `64 + ((tmx + 512) >> 10)`, `[0, 192]` sözleşmesi.
#[inline(always)]
fn warp_filter_index(tmx: i32) -> usize {
    let idx = 64 + ((tmx + 512) >> 10);
    debug_assert!(
        (0..N_WARP_FILTERS as i32).contains(&idx),
        "warp filtre indeksi [0,192] dışında: {idx}"
    );
    idx as usize
}

/// `warp_affine_8x8_c` — dav1d `src/mc_tmpl.c:799-832`.
#[allow(clippy::too_many_arguments)]
fn warp8x8_bpc<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
) {
    let ib = intermediate_bits_of(BPC);
    let max_pixel = (1i32 << BPC) - 1;
    let bpp = bpp_of(BPC);
    let mut mid = [0i16; 15 * 8];
    let mut mx = mx;
    let mut my = my;
    for y in 0..15 {
        let mut tmx = mx;
        for x in 0..8 {
            let f = &MC_WARP_FILTER[warp_filter_index(tmx)];
            // 15×15 pencere: dolgu satırı `y` (blok satırı `y−3`), tap'ler
            // dolgu sütunu `x..x+7`.
            mid[y * 8 + x] = warp_filter_h::<BPC>(src, y * src_stride + x * bpp, f, 7 - ib) as i16;
            tmx += i32::from(abcd[0]);
        }
        mx += i32::from(abcd[1]);
    }
    for y in 0..8 {
        let mut tmy = my;
        let drow = y * dst_stride;
        for x in 0..8 {
            let f = &MC_WARP_FILTER[warp_filter_index(tmy)];
            let value = warp_filter_v(&mid, y, x, f, 7 + ib);
            store::<BPC>(dst, drow + x * bpp, iclip_pixel(value, max_pixel));
            tmy += i32::from(abcd[2]);
        }
        my += i32::from(abcd[3]);
    }
}

/// `warp_affine_8x8t_c` — dav1d `src/mc_tmpl.c:834-866`.
#[allow(clippy::too_many_arguments)]
fn warp8x8t_bpc<const BPC: u32>(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
) {
    let ib = intermediate_bits_of(BPC);
    let bias = prep_bias_of(BPC);
    let bpp = bpp_of(BPC);
    let mut mid = [0i16; 15 * 8];
    let mut mx = mx;
    let mut my = my;
    for y in 0..15 {
        let mut tmx = mx;
        for x in 0..8 {
            let f = &MC_WARP_FILTER[warp_filter_index(tmx)];
            mid[y * 8 + x] = warp_filter_h::<BPC>(src, y * src_stride + x * bpp, f, 7 - ib) as i16;
            tmx += i32::from(abcd[0]);
        }
        mx += i32::from(abcd[1]);
    }
    for y in 0..8 {
        let mut tmy = my;
        for x in 0..8 {
            let f = &MC_WARP_FILTER[warp_filter_index(tmy)];
            let value = warp_filter_v(&mid, y, x, f, 7).wrapping_sub(bias);
            store_tmp(tmp, y * tmp_stride + x, value);
            tmy += i32::from(abcd[2]);
        }
        my += i32::from(abcd[3]);
    }
}

/// `emu_edge_c` — dav1d `src/mc_tmpl.c:868-916`.
///
/// **Kaynak sözleşmesi farklıdır:** `src` referans düzleminin sol üstüdür
/// (dolgu hücresi değil); `(x, y)` bloğun düzlemdeki konumudur ve negatif
/// olabilir. Okuma penceresi
/// `iclip(y,0,ih−1) .. +center_h` × `iclip(x,0,iw−1) .. +center_w`'dir;
/// bu pencerenin dışına **okunmaz** (test (b)).
#[allow(clippy::too_many_arguments)]
fn emu_edge_bpc<const BPC: u32>(
    bw: usize,
    bh: usize,
    iw: usize,
    ih: usize,
    x: i32,
    y: i32,
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
) {
    let bpp = bpp_of(BPC);
    debug_assert!(iw >= 1 && ih >= 1 && bw >= 1 && bh >= 1);
    // `ref += iclip(y,0,ih-1)*stride + iclip(x,0,iw-1)`.
    let ref_row =
        y.clamp(0, ih as i32 - 1) as usize * src_stride + x.clamp(0, iw as i32 - 1) as usize * bpp;
    let left_ext = (-x).clamp(0, bw as i32 - 1) as usize;
    let right_ext = (x + bw as i32 - iw as i32).clamp(0, bw as i32 - 1) as usize;
    debug_assert!(left_ext + right_ext < bw);
    let top_ext = (-y).clamp(0, bh as i32 - 1) as usize;
    let bottom_ext = (y + bh as i32 - ih as i32).clamp(0, bh as i32 - 1) as usize;
    debug_assert!(top_ext + bottom_ext < bh);

    let center_w = bw - left_ext - right_ext;
    let center_h = bh - top_ext - bottom_ext;
    for row in 0..center_h {
        let s = ref_row + row * src_stride;
        let d = (top_ext + row) * dst_stride;
        dst[d + left_ext * bpp..d + (left_ext + center_w) * bpp]
            .copy_from_slice(&src[s..s + center_w * bpp]);
        // Sol kenar: `pixel_set(blk, blk[left_ext], left_ext)`.
        let edge_left = load::<BPC>(dst, d + left_ext * bpp);
        for k in 0..left_ext {
            store::<BPC>(dst, d + k * bpp, edge_left);
        }
        // Sağ kenar: `pixel_set(blk + left_ext + center_w, blk[...−1], right_ext)`.
        let edge_right = load::<BPC>(dst, d + (left_ext + center_w - 1) * bpp);
        for k in 0..right_ext {
            store::<BPC>(dst, d + (left_ext + center_w + k) * bpp, edge_right);
        }
    }
    // Üst: görünür bloğun ilk satırını `top_ext` kez kopyala.
    for row in 0..top_ext {
        let s = top_ext * dst_stride;
        let d = row * dst_stride;
        for k in 0..bw {
            let p = load::<BPC>(dst, s + k * bpp);
            store::<BPC>(dst, d + k * bpp, p);
        }
    }
    // Alt: son görünür satırı `bottom_ext` kez kopyala.
    let last = (top_ext + center_h - 1) * dst_stride;
    for row in 0..bottom_ext {
        let d = (top_ext + center_h + row) * dst_stride;
        for k in 0..bw {
            let p = load::<BPC>(dst, last + k * bpp);
            store::<BPC>(dst, d + k * bpp, p);
        }
    }
}

// ---------------------------------------------------------------------------
// Bit derinliğine göre ince sarmalayıcılar (dav1d `bitfn` deseni).
// ---------------------------------------------------------------------------

/// F3c: giriş `BPC` ile monomorftur; çağrı başına bit derinliği seçimi yoktur.
/// `bit_depth` parametresi imzada kalır (ADR-0003 §2.2) ve yalnız yuva ile
/// çağıranın uyumu `debug_assert` ile denetlenir.
macro_rules! bpc_dispatch {
    ($bit_depth:expr, $f:ident $(, $arg:expr)*) => {{
        debug_assert_eq!($bit_depth.bits(), BPC, "yuva bit derinliği ≠ çağrı");
        $f::<BPC>($($arg),*)
    }};
}

/// 16bpc yuvada `bitdepth_max` gerçekten 10/12-bit mi (S2b.2 dersi)?
#[inline(always)]
fn check_max(bit_depth: BitDepth, bitdepth_max: u16) {
    debug_assert_eq!(
        u32::from(bitdepth_max),
        (1u32 << bit_depth.bits()) - 1,
        "bitdepth_max bit derinliğiyle tutarsız"
    );
}

/// `put_8tap` — dav1d `put_8tap_*_c` (10 filtre birleşimi).
#[allow(clippy::too_many_arguments)]
pub fn put_8tap<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    filter: Filter2d,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    check_max(bit_depth, bitdepth_max);
    bpc_dispatch!(
        bit_depth,
        put_8tap_bpc,
        dst,
        dst_stride,
        src,
        src_stride,
        w,
        h,
        mx,
        my,
        filter
    );
}

/// `prep_8tap` — dav1d `prep_8tap_*_c`.
#[allow(clippy::too_many_arguments)]
pub fn prep_8tap<const BPC: u32>(
    tmp: &mut [i16],
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    filter: Filter2d,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    check_max(bit_depth, bitdepth_max);
    bpc_dispatch!(
        bit_depth,
        prep_8tap_bpc,
        tmp,
        src,
        src_stride,
        w,
        h,
        mx,
        my,
        filter
    );
}

/// `put_8tap_scaled` — dav1d `put_8tap_*_scaled_c`.
#[allow(clippy::too_many_arguments)]
pub fn put_8tap_scaled<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
    filter: Filter2d,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    check_max(bit_depth, bitdepth_max);
    bpc_dispatch!(
        bit_depth,
        put_8tap_scaled_bpc,
        dst,
        dst_stride,
        src,
        src_stride,
        w,
        h,
        mx,
        my,
        dx,
        dy,
        filter
    );
}

/// `prep_8tap_scaled` — dav1d `prep_8tap_*_scaled_c`.
#[allow(clippy::too_many_arguments)]
pub fn prep_8tap_scaled<const BPC: u32>(
    tmp: &mut [i16],
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
    filter: Filter2d,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    check_max(bit_depth, bitdepth_max);
    bpc_dispatch!(
        bit_depth,
        prep_8tap_scaled_bpc,
        tmp,
        src,
        src_stride,
        w,
        h,
        mx,
        my,
        dx,
        dy,
        filter
    );
}

/// `put_bilin` — dav1d `put_bilin_c`.
#[allow(clippy::too_many_arguments)]
pub fn put_bilin<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    check_max(bit_depth, bitdepth_max);
    bpc_dispatch!(
        bit_depth,
        put_bilin_bpc,
        dst,
        dst_stride,
        src,
        src_stride,
        w,
        h,
        mx,
        my
    );
}

/// `prep_bilin` — dav1d `prep_bilin_c`.
#[allow(clippy::too_many_arguments)]
pub fn prep_bilin<const BPC: u32>(
    tmp: &mut [i16],
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    check_max(bit_depth, bitdepth_max);
    bpc_dispatch!(
        bit_depth,
        prep_bilin_bpc,
        tmp,
        src,
        src_stride,
        w,
        h,
        mx,
        my
    );
}

/// `put_bilin_scaled` — dav1d `put_bilin_scaled_c`.
#[allow(clippy::too_many_arguments)]
pub fn put_bilin_scaled<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    check_max(bit_depth, bitdepth_max);
    bpc_dispatch!(
        bit_depth,
        put_bilin_scaled_bpc,
        dst,
        dst_stride,
        src,
        src_stride,
        w,
        h,
        mx,
        my,
        dx,
        dy
    );
}

/// `prep_bilin_scaled` — dav1d `prep_bilin_scaled_c`.
#[allow(clippy::too_many_arguments)]
pub fn prep_bilin_scaled<const BPC: u32>(
    tmp: &mut [i16],
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    check_max(bit_depth, bitdepth_max);
    bpc_dispatch!(
        bit_depth,
        prep_bilin_scaled_bpc,
        tmp,
        src,
        src_stride,
        w,
        h,
        mx,
        my,
        dx,
        dy
    );
}

/// `avg` — dav1d `avg_c`.
#[allow(clippy::too_many_arguments)]
pub fn avg<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    check_max(bit_depth, bitdepth_max);
    bpc_dispatch!(bit_depth, avg_bpc, dst, dst_stride, tmp1, tmp2, w, h);
}

/// `w_avg` — dav1d `w_avg_c`.
#[allow(clippy::too_many_arguments)]
pub fn w_avg<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    check_max(bit_depth, bitdepth_max);
    bpc_dispatch!(
        bit_depth, w_avg_bpc, dst, dst_stride, tmp1, tmp2, w, h, weight
    );
}

/// `mask` — dav1d `mask_c`.
#[allow(clippy::too_many_arguments)]
pub fn mask<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &[u8],
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    check_max(bit_depth, bitdepth_max);
    bpc_dispatch!(bit_depth, mask_bpc, dst, dst_stride, tmp1, tmp2, w, h, mask);
}

/// `w_mask` — dav1d `w_mask_{444,422,420}_c`.
#[allow(clippy::too_many_arguments)]
pub fn w_mask<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &mut [u8],
    sign: i32,
    ss_hor: bool,
    ss_ver: bool,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    check_max(bit_depth, bitdepth_max);
    bpc_dispatch!(
        bit_depth, w_mask_bpc, dst, dst_stride, tmp1, tmp2, w, h, mask, sign, ss_hor, ss_ver
    );
}

/// `blend` — dav1d `blend_c`.
#[allow(clippy::too_many_arguments)]
pub fn blend<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    mask: &[u8],
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    check_max(bit_depth, bitdepth_max);
    bpc_dispatch!(bit_depth, blend_bpc, dst, dst_stride, tmp, w, h, mask);
}

/// `blend_v` — dav1d `blend_v_c` (OBMC dikey kenar).
#[allow(clippy::too_many_arguments)]
pub fn blend_v<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    check_max(bit_depth, bitdepth_max);
    bpc_dispatch!(bit_depth, blend_v_bpc, dst, dst_stride, tmp, w, h);
}

/// `blend_h` — dav1d `blend_h_c` (OBMC yatay kenar).
#[allow(clippy::too_many_arguments)]
pub fn blend_h<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    check_max(bit_depth, bitdepth_max);
    bpc_dispatch!(bit_depth, blend_h_bpc, dst, dst_stride, tmp, w, h);
}

/// `warp8x8` — dav1d `warp_affine_8x8_c`.
#[allow(clippy::too_many_arguments)]
pub fn warp8x8<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    check_max(bit_depth, bitdepth_max);
    bpc_dispatch!(
        bit_depth,
        warp8x8_bpc,
        dst,
        dst_stride,
        src,
        src_stride,
        abcd,
        mx,
        my
    );
}

/// `warp8x8t` — dav1d `warp_affine_8x8t_c`.
#[allow(clippy::too_many_arguments)]
pub fn warp8x8t<const BPC: u32>(
    tmp: &mut [i16],
    tmp_stride: usize,
    src: &[u8],
    src_stride: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    check_max(bit_depth, bitdepth_max);
    bpc_dispatch!(
        bit_depth,
        warp8x8t_bpc,
        tmp,
        tmp_stride,
        src,
        src_stride,
        abcd,
        mx,
        my
    );
}

/// `emu_edge` — dav1d `emu_edge_c`.
#[allow(clippy::too_many_arguments)]
pub fn emu_edge<const BPC: u32>(
    bw: usize,
    bh: usize,
    iw: usize,
    ih: usize,
    x: i32,
    y: i32,
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    check_max(bit_depth, bitdepth_max);
    bpc_dispatch!(
        bit_depth,
        emu_edge_bpc,
        bw,
        bh,
        iw,
        ih,
        x,
        y,
        dst,
        dst_stride,
        src,
        src_stride
    );
}

/// S15.5 pilotunun giriş noktası — yeni genel skaler çekirdekle **tutarlıdır**.
///
/// `put_8tap_regular` artık [`put_8tap`]'in `FILTER_2D_8TAP_REGULAR` için ince
/// bir sarmalayıcısıdır; davranış (bit-exact) değişmemiştir ve AVX2 yuvası
/// ([`crate::dsp::x86::mc`]) aynı işlevi kullanmaya devam eder. Sözleşme
/// S15.5'teki gibidir: `src` dolgu hücresinden (satır −3, sütun −3) başlar.
#[allow(clippy::too_many_arguments)]
pub fn put_8tap_regular(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bit_depth: McBitDepth,
) {
    let (bd, max) = (bit_depth.to_bit_depth(), bit_depth.max_pixel() as u16);
    let f = Filter2d::Regular;
    match bit_depth {
        McBitDepth::B8 => put_8tap::<8>(dst, dst_stride, src, src_stride, w, h, mx, my, f, bd, max),
        McBitDepth::B10 => {
            put_8tap::<10>(dst, dst_stride, src, src_stride, w, h, mx, my, f, bd, max)
        }
        McBitDepth::B12 => {
            put_8tap::<12>(dst, dst_stride, src, src_stride, w, h, mx, my, f, bd, max)
        }
    }
}
