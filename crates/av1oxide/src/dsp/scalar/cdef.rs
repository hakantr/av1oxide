//! CDEF skaler (referans) çekirdekleri — S15.4 pilotu.
//!
//! Bu modül dav1d `src/cdef_tmpl.c`'nin **birebir** portudur; satır referansları
//! ilgili kodun yanındadır. Amaç bit-exact davranıştır: C'nin tanımsız davranışı
//! (negatif kaydırma, `ulog2(0)`) **taşınmaz**, yerine tanımlı ve belgelenmiş
//! davranış konur; geri kalan her aritmetik açık genişletme (`i32`) ya da
//! `wrapping_*` ile tanımlıdır.
//!
//! ## Bit derinliği monomorfizasyonu
//!
//! Gövdeler `const BPC: u32` ile tek yazılır; `8`/`10`/`12` için üç ayrı
//! örnekleme üretilir. Böylece piksel erişimi ([`px_at`], [`px_put`]) ve
//! `bitdepth_min_8` dalları derleme zamanında kaybolur.
//!
//! ## Sözleşme dışı girdiler
//!
//! | Girdi | dav1d C | Bu port |
//! |---|---|---|
//! | `pri_strength == 0 && sec_strength == 0` | `assert` düşer; `NDEBUG`'da `ulog2(0)` + negatif kaydırma → **UB** | `dst` **değişmeden** bırakılır (tanımlı) |
//! | `sec_shift < 0` (`damping < ulog2(sec)`) | negatif kaydırma → **UB** | 0'a kırpılır (tanımlı) |
//! | `dir > 7`, güçler aralık dışı | tanımsız | hata ayıklamada `debug_assert`, serbest'te maskelenir/kırpılır |

// Döngüler dav1d kaynağındaki indekslemeyi **birebir** yansıtır
// (`partial_sum_*[n]`, `cdef_dirs[k]`); denetim izi bu yüzden korunur.
#![allow(clippy::needless_range_loop)]
#![forbid(unsafe_code)]

use crate::dsp::cdef::{
    CDEF_DIRECTIONS, CdefBitDepth, CdefDirResult, CdefEdgeFlags, CdefFilterParams,
    DIR_CENTER_WEIGHT, DIV_TABLE, TMP_BUF_LEN, TMP_FILL, clamped_shift, constrain, px_at, px_put,
    tmp_index, ulog2,
};

/// `apply_sign`in `umin`/`imax`/`iclip` yardımcıları (dav1d `include/common/intops.h:35-53`).
///
/// `umin` C'de **unsigned** parametre alır; `INT16_MIN` dolgusunun "unsigned'da
/// büyük" davranışı buna dayanır (`src/cdef_tmpl.c:47-48`), bu yüzden burada da
/// bit deseni `u32` üzerinden karşılaştırılır.
#[inline]
fn umin(a: i32, b: i32) -> i32 {
    if (a as u32) < (b as u32) { a } else { b }
}

#[inline]
fn imax(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}

#[inline]
fn iclip(v: i32, min: i32, max: i32) -> i32 {
    if v < min {
        min
    } else if v > max {
        max
    } else {
        v
    }
}

/// Skaler `dir` çekirdeği — dav1d `cdef_find_dir_c`; `BPC` (8/10/12) ile
/// monomorfize yuva girişi (F3c). `bit_depth` imzada kalır ve yalnız yuva ile
/// çağıranın uyumu `debug_assert` ile denetlenir.
pub fn find_dir<const BPC: u32>(
    img: &[u8],
    stride: usize,
    bit_depth: CdefBitDepth,
) -> CdefDirResult {
    debug_assert_eq!(bit_depth.bits(), BPC, "yuva bit derinliği ≠ çağrı");
    find_dir_bpc::<BPC>(img, stride)
}

/// `dir` gövdesi — dav1d `src/cdef_tmpl.c:239-305`.
fn find_dir_bpc<const BPC: u32>(img: &[u8], stride: usize) -> CdefDirResult {
    let bpp = if BPC == 8 { 1usize } else { 2 };
    let bd_min_8 = BPC - 8;
    debug_assert!(img.len() >= 7 * stride + 8 * bpp, "dir: 8x8 blok okunmalı");

    // Kısmi toplamlar (dav1d `src/cdef_tmpl.c:243-245`).
    let mut partial_sum_hv = [[0i32; 8]; 2];
    let mut partial_sum_diag = [[0i32; 15]; 2];
    let mut partial_sum_alt = [[0i32; 11]; 4];

    for y in 0..8usize {
        for x in 0..8usize {
            // dav1d `src/cdef_tmpl.c:249`
            let px = (i32::from(px_at::<BPC>(img, y * stride + x * bpp)) >> bd_min_8) - 128;

            partial_sum_diag[0][y + x] = partial_sum_diag[0][y + x].wrapping_add(px);
            partial_sum_alt[0][y + (x >> 1)] = partial_sum_alt[0][y + (x >> 1)].wrapping_add(px);
            partial_sum_hv[0][y] = partial_sum_hv[0][y].wrapping_add(px);
            partial_sum_alt[1][3 + y - (x >> 1)] =
                partial_sum_alt[1][3 + y - (x >> 1)].wrapping_add(px);
            partial_sum_diag[1][7 + y - x] = partial_sum_diag[1][7 + y - x].wrapping_add(px);
            partial_sum_alt[2][3 - (y >> 1) + x] =
                partial_sum_alt[2][3 - (y >> 1) + x].wrapping_add(px);
            partial_sum_hv[1][x] = partial_sum_hv[1][x].wrapping_add(px);
            partial_sum_alt[3][(y >> 1) + x] = partial_sum_alt[3][(y >> 1) + x].wrapping_add(px);
        }
    }

    // Maliyetler (dav1d `src/cdef_tmpl.c:263-292`). `unsigned` semantiği:
    // `(int * int) as u32` + `wrapping_*` — C'de taşma tanımlıdır (mod 2^32).
    let mut cost = [0u32; 8];

    for n in 0..8usize {
        let hv0 = partial_sum_hv[0][n];
        let hv1 = partial_sum_hv[1][n];
        cost[2] = cost[2].wrapping_add(hv0.wrapping_mul(hv0) as u32);
        cost[6] = cost[6].wrapping_add(hv1.wrapping_mul(hv1) as u32);
    }
    cost[2] = cost[2].wrapping_mul(DIR_CENTER_WEIGHT);
    cost[6] = cost[6].wrapping_mul(DIR_CENTER_WEIGHT);

    for n in 0..7usize {
        let d = DIV_TABLE[n];
        let a = partial_sum_diag[0][n];
        let b = partial_sum_diag[0][14 - n];
        cost[0] = cost[0].wrapping_add(
            (a.wrapping_mul(a).wrapping_add(b.wrapping_mul(b)) as u32).wrapping_mul(d),
        );
        let a = partial_sum_diag[1][n];
        let b = partial_sum_diag[1][14 - n];
        cost[4] = cost[4].wrapping_add(
            (a.wrapping_mul(a).wrapping_add(b.wrapping_mul(b)) as u32).wrapping_mul(d),
        );
    }
    let c0 = partial_sum_diag[0][7];
    let c1 = partial_sum_diag[1][7];
    cost[0] = cost[0].wrapping_add((c0.wrapping_mul(c0) as u32).wrapping_mul(DIR_CENTER_WEIGHT));
    cost[4] = cost[4].wrapping_add((c1.wrapping_mul(c1) as u32).wrapping_mul(DIR_CENTER_WEIGHT));

    for n in 0..4usize {
        let mut acc = 0u32;
        for m in 0..5usize {
            let v = partial_sum_alt[n][3 + m];
            acc = acc.wrapping_add(v.wrapping_mul(v) as u32);
        }
        acc = acc.wrapping_mul(DIR_CENTER_WEIGHT);
        for m in 0..3usize {
            let d = DIV_TABLE[2 * m + 1];
            let a = partial_sum_alt[n][m];
            let b = partial_sum_alt[n][10 - m];
            acc = acc.wrapping_add(
                (a.wrapping_mul(a).wrapping_add(b.wrapping_mul(b)) as u32).wrapping_mul(d),
            );
        }
        cost[n * 2 + 1] = acc;
    }

    // En iyi yön: **katı** `>` (dav1d `src/cdef_tmpl.c:294-301`).
    let mut best_dir = 0usize;
    let mut best_cost = cost[0];
    for n in 1..8usize {
        if cost[n] > best_cost {
            best_cost = cost[n];
            best_dir = n;
        }
    }

    // dav1d `src/cdef_tmpl.c:303`
    CdefDirResult {
        dir: best_dir as u8,
        var: best_cost.wrapping_sub(cost[best_dir ^ 4]) >> 10,
    }
}

/// Skaler `filter` çekirdeği — dav1d `cdef_filter_block_c`; `BPC` ile
/// monomorfize yuva girişi (F3c). Blok boyutu (`fb[0..2]`) `params.size`
/// içindedir; `params.bit_depth` yalnız `debug_assert` ile denetlenir.
pub fn filter<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    debug_assert_eq!(params.bit_depth.bits(), BPC, "yuva bit derinliği ≠ çağrı");
    filter_bpc::<BPC>(dst, stride, left, top, bottom, params);
}

/// `filter` gövdesi — dav1d `src/cdef_tmpl.c:104-216`.
fn filter_bpc<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    let bpp = if BPC == 8 { 1usize } else { 2 };
    let w = params.size.w();
    let h = params.size.h();
    debug_assert!(params.dir < 8, "dir 0..7 olmalı");
    debug_assert!(
        u32::from(params.pri_strength) <= 15 << (BPC - 8),
        "pri_strength aralık dışı"
    );
    debug_assert!(
        u32::from(params.sec_strength) <= 4 << (BPC - 8),
        "sec_strength aralık dışı"
    );
    debug_assert!(
        dst.len() >= (h - 1) * stride + (w + 2) * bpp,
        "dst: satırlar x = w+1'e kadar okunabilmeli"
    );

    let has_left = params.edges.contains(CdefEdgeFlags::HAVE_LEFT);
    let has_right = params.edges.contains(CdefEdgeFlags::HAVE_RIGHT);
    let has_top = params.edges.contains(CdefEdgeFlags::HAVE_TOP);
    let has_bottom = params.edges.contains(CdefEdgeFlags::HAVE_BOTTOM);

    // dav1d `src/cdef_tmpl.c:63-79` — sınırlar.
    let x_start: i32 = if has_left { -2 } else { 0 };
    let x_end: i32 = if has_right { w as i32 + 2 } else { w as i32 };

    // Padding (dav1d `src/cdef_tmpl.c:44-102`).
    //
    // C'deki dört `fill()` çağrısı yerine **tamponun tamamı** `TMP_FILL` ile
    // başlatılır: kopyalama döngüleri okuma penceresi dışında kalan her konumu
    // zaten doldurur, dolgu bölgeleri ise tanım gereği kopyalanmaz. Bu yüzden
    // "tümü dolgu" ile C'nin "seçili bölgeyi dolgu" davranışı **aynıdır**.
    let mut tmp = [TMP_FILL; TMP_BUF_LEN];
    let top_rows = if has_top { 2usize } else { 0 };
    let bottom_rows = if has_bottom { 2usize } else { 0 };

    for row in 0..top_rows {
        // dav1d `src/cdef_tmpl.c:81-85`
        let y = -2 + row as i32;
        for x in x_start..x_end {
            let v = px_at::<BPC>(top, row * stride + (x + 2) as usize * bpp);
            tmp[tmp_index(x, y)] = v as i16;
        }
    }
    if has_left {
        // dav1d `src/cdef_tmpl.c:86-88` — `left[y][2 + x]`
        for y in 0..h {
            for x in -2i32..0 {
                let v = px_at::<BPC>(left, y * 2 * bpp + (x + 2) as usize * bpp);
                tmp[tmp_index(x, y as i32)] = v as i16;
            }
        }
    }
    for y in 0..h {
        // dav1d `src/cdef_tmpl.c:89-94` — `tmp[x] = src[x]`, x ∈ [0, x_end)
        for x in 0..x_end {
            let v = px_at::<BPC>(dst, y * stride + x as usize * bpp);
            tmp[tmp_index(x, y as i32)] = v as i16;
        }
    }
    for row in 0..bottom_rows {
        // dav1d `src/cdef_tmpl.c:95-100`
        let y = h as i32 + row as i32;
        for x in x_start..x_end {
            let v = px_at::<BPC>(bottom, row * stride + (x + 2) as usize * bpp);
            tmp[tmp_index(x, y)] = v as i16;
        }
    }

    let dir = usize::from(params.dir & 7);
    let pri_strength = i32::from(params.pri_strength);
    let sec_strength = i32::from(params.sec_strength);

    // dav1d `src/cdef_tmpl.c:190-191`: `pri == 0 && sec == 0` sözleşme dışıdır
    // (C'de `assert` + `NDEBUG`'da UB). Matematiksel sonuç "değişiklik yok"
    // olduğu için burada tanımlı biçimde erken dönülür.
    if pri_strength == 0 && sec_strength == 0 {
        return;
    }

    /// `tmp` üzerinden tap okuması: `tmp[x + off]`, taban `(x, y)` satırında.
    #[inline(always)]
    fn tap(tmp: &[i16; TMP_BUF_LEN], x: i32, y: i32, off: i32) -> i32 {
        let index = tmp_index(x, y) as i32 + off;
        debug_assert!(
            (0..TMP_BUF_LEN as i32).contains(&index),
            "tap indeksi taştı"
        );
        i32::from(tmp[index as usize])
    }

    if pri_strength != 0 {
        // dav1d `src/cdef_tmpl.c:121-123`
        let bd_min_8 = BPC - 8;
        let pri_tap = 4 - (((params.pri_strength as u32) >> bd_min_8) & 1) as i32;
        let pri_shift =
            clamped_shift(params.damping as i32 - ulog2(params.pri_strength as u32) as i32);

        if sec_strength != 0 {
            // dav1d `src/cdef_tmpl.c:124-169` — öncelikli + ikincil (kırpmalı dal)
            let sec_shift =
                clamped_shift(params.damping as i32 - ulog2(params.sec_strength as u32) as i32);
            for y in 0..h {
                for x in 0..w {
                    let px = i32::from(px_at::<BPC>(dst, y * stride + x * bpp));
                    let mut sum = 0i32;
                    let (mut min, mut max) = (px, px);
                    let mut pri_tap_k = pri_tap;
                    for k in 0..2usize {
                        let off1 = i32::from(CDEF_DIRECTIONS[dir + 2][k]); // dir
                        let p0 = tap(&tmp, x as i32, y as i32, off1);
                        let p1 = tap(&tmp, x as i32, y as i32, -off1);
                        sum = sum.wrapping_add(pri_tap_k.wrapping_mul(constrain(
                            p0 - px,
                            pri_strength,
                            pri_shift,
                        )));
                        sum = sum.wrapping_add(pri_tap_k.wrapping_mul(constrain(
                            p1 - px,
                            pri_strength,
                            pri_shift,
                        )));
                        // dav1d `src/cdef_tmpl.c:138-139`: pri_tap_k == 4 ise 2, değilse 3
                        pri_tap_k = (pri_tap_k & 3) | 2;
                        min = umin(p0, min);
                        max = imax(p0, max);
                        min = umin(p1, min);
                        max = imax(p1, max);

                        let off2 = i32::from(CDEF_DIRECTIONS[dir + 4][k]); // dir + 2
                        let off3 = i32::from(CDEF_DIRECTIONS[dir][k]); // dir - 2
                        let s0 = tap(&tmp, x as i32, y as i32, off2);
                        let s1 = tap(&tmp, x as i32, y as i32, -off2);
                        let s2 = tap(&tmp, x as i32, y as i32, off3);
                        let s3 = tap(&tmp, x as i32, y as i32, -off3);
                        let sec_tap = 2 - k as i32; // dav1d `src/cdef_tmpl.c:151`
                        sum = sum.wrapping_add(sec_tap.wrapping_mul(constrain(
                            s0 - px,
                            sec_strength,
                            sec_shift,
                        )));
                        sum = sum.wrapping_add(sec_tap.wrapping_mul(constrain(
                            s1 - px,
                            sec_strength,
                            sec_shift,
                        )));
                        sum = sum.wrapping_add(sec_tap.wrapping_mul(constrain(
                            s2 - px,
                            sec_strength,
                            sec_shift,
                        )));
                        sum = sum.wrapping_add(sec_tap.wrapping_mul(constrain(
                            s3 - px,
                            sec_strength,
                            sec_shift,
                        )));
                        min = umin(s0, min);
                        max = imax(s0, max);
                        min = umin(s1, min);
                        max = imax(s1, max);
                        min = umin(s2, min);
                        max = imax(s2, max);
                        min = umin(s3, min);
                        max = imax(s3, max);
                    }
                    // dav1d `src/cdef_tmpl.c:165`
                    let v = px.wrapping_add(round_shift(sum));
                    let v = iclip(v, min, max);
                    px_put::<BPC>(dst, y * stride + x * bpp, v as u16);
                }
            }
        } else {
            // dav1d `src/cdef_tmpl.c:170-189` — yalnız öncelikli (**kırpma yok**)
            for y in 0..h {
                for x in 0..w {
                    let px = i32::from(px_at::<BPC>(dst, y * stride + x * bpp));
                    let mut sum = 0i32;
                    let mut pri_tap_k = pri_tap;
                    for k in 0..2usize {
                        let off = i32::from(CDEF_DIRECTIONS[dir + 2][k]);
                        let p0 = tap(&tmp, x as i32, y as i32, off);
                        let p1 = tap(&tmp, x as i32, y as i32, -off);
                        sum = sum.wrapping_add(pri_tap_k.wrapping_mul(constrain(
                            p0 - px,
                            pri_strength,
                            pri_shift,
                        )));
                        sum = sum.wrapping_add(pri_tap_k.wrapping_mul(constrain(
                            p1 - px,
                            pri_strength,
                            pri_shift,
                        )));
                        pri_tap_k = (pri_tap_k & 3) | 2;
                    }
                    // Kırpma yok: `pixel` tipine dönüşüm (8-bit'te mod 256).
                    let v = px.wrapping_add(round_shift(sum));
                    px_put::<BPC>(dst, y * stride + x * bpp, v as u16);
                }
            }
        }
    } else {
        // dav1d `src/cdef_tmpl.c:190-215` — yalnız ikincil (**kırpma yok**)
        let sec_shift =
            clamped_shift(params.damping as i32 - ulog2(params.sec_strength as u32) as i32);
        for y in 0..h {
            for x in 0..w {
                let px = i32::from(px_at::<BPC>(dst, y * stride + x * bpp));
                let mut sum = 0i32;
                for k in 0..2usize {
                    let off1 = i32::from(CDEF_DIRECTIONS[dir + 4][k]); // dir + 2
                    let off2 = i32::from(CDEF_DIRECTIONS[dir][k]); // dir - 2
                    let s0 = tap(&tmp, x as i32, y as i32, off1);
                    let s1 = tap(&tmp, x as i32, y as i32, -off1);
                    let s2 = tap(&tmp, x as i32, y as i32, off2);
                    let s3 = tap(&tmp, x as i32, y as i32, -off2);
                    let sec_tap = 2 - k as i32;
                    sum = sum.wrapping_add(sec_tap.wrapping_mul(constrain(
                        s0 - px,
                        sec_strength,
                        sec_shift,
                    )));
                    sum = sum.wrapping_add(sec_tap.wrapping_mul(constrain(
                        s1 - px,
                        sec_strength,
                        sec_shift,
                    )));
                    sum = sum.wrapping_add(sec_tap.wrapping_mul(constrain(
                        s2 - px,
                        sec_strength,
                        sec_shift,
                    )));
                    sum = sum.wrapping_add(sec_tap.wrapping_mul(constrain(
                        s3 - px,
                        sec_strength,
                        sec_shift,
                    )));
                }
                let v = px.wrapping_add(round_shift(sum));
                px_put::<BPC>(dst, y * stride + x * bpp, v as u16);
            }
        }
    }
}

/// `(sum - (sum < 0) + 8) >> 4` — dav1d `src/cdef_tmpl.c:165,184,210`.
///
/// `sum < 0` iken bir çıkarılır, sonra 8 eklenip **aritmetik** 4 bit kaydırılır
/// (C'de `>>` negatifte implementation-defined; gcc/clang aritmetik uygular,
/// Rust `>>` de aritmetiktir).
#[inline(always)]
fn round_shift(sum: i32) -> i32 {
    (sum - i32::from(sum < 0) + 8) >> 4
}
