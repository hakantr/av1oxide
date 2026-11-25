//! Loop restoration skaler (referans) çekirdekleri — S2c.2b.
//!
//! Bu modül dav1d `src/looprestoration_tmpl.c` C yolunun **birebir** portudur
//! (Wiener 7/5 dokunuş, SGR 5×5 / 3×3 / karma). Satır referansları ilgili
//! kodun yanındadır; sözleşme ve tam kaynak tablosu [`crate::dsp::lr`]
//! başlığındadır.
//!
//! ## Çeviri kuralları
//!
//! - **İşaretçi pencereleri** (`ptrs[7]`, `sumsq_ptrs[5]`, `A_ptrs[4]` …) satır
//!   **indeksi** dizileridir; dav1d'deki her atama/döndürme aynı sırayla yapılır
//!   (`rotate` = `rotate_left(1)`, `rotate5_x2` = `rotate_left(2)`). dav1d'de
//!   `sumsq_ptrs`/`sum_ptrs` ve `A_ptrs`/`B_ptrs` her zaman **birlikte**
//!   atanır/döndürülür; bu yüzden her çift tek bir indeks dizisi paylaşır.
//! - `goto` kuyrukları (`v1/v2/v3`, `vert_1/vert_2/odd/output_1/output_2`)
//!   etiketli blok + `enum` ile aynı sırada yürütülür.
//! - **Yerinde** çalışma: kaynak satırlar `dst`'ten okunur, çıktı aynı `dst`'e
//!   yazılır. Okuma/yazma sırası dav1d ile aynı olduğu için sonuç da aynıdır.
//! - `coef` tipi 8 bit'te `int16_t`, 16bpc'de `int32_t`'dir (`common/bitdepth.h`);
//!   `coef` depolamaları özel `coef()` yardımcısıyla aynı daraltmayı yapar (geçerli girdide
//!   değerler sığar, daraltma no-op'tur).
//! - `unsigned` aritmetik (`p * s`, `x * BB * one_by_x`) `u32` `wrapping_*`
//!   ile yapılır (C'de tanımlı sarma); geri kalan aritmetik `i32`'dir ve
//!   geçerli piksel aralığında taşmaz (`robust` profilli oracle koşusu bunu
//!   taşma denetimiyle doğrular).
//! - C'nin başlatılmamış yığın tamponları burada sıfırla başlar; dav1d hiçbir
//!   zaman yazmadığı bir satırı okumadığı için (oracle kanıtı) gözlemlenebilir
//!   fark yoktur.

// Döngüler dav1d kaynağındaki indekslemeyi **birebir** yansıtır (`tmp[i]`,
// `B[i - 1]` …); denetim izi bu yüzden korunur.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
#![forbid(unsafe_code)]

use crate::dsp::lr::{LrBitDepth, LrEdgeFlags, LrParams};
use crate::dsp::lr_tables::SGR_X_BY_X;

/// dav1d `REST_UNIT_STRIDE` (`looprestoration_tmpl.c:41-42`).
const REST_UNIT_STRIDE: usize = 390;
/// dav1d `BUF_STRIDE` (`looprestoration_tmpl.c:679`).
const BUF_STRIDE: usize = 384 + 16;
/// dav1d `FILTER_OUT_STRIDE` (`looprestoration_tmpl.c:566`).
const FILTER_OUT_STRIDE: usize = 384;
/// Desteklenen en büyük birim genişliği (`FILTER_OUT_STRIDE` sınırı).
pub const LR_MAX_W: usize = 384;

/// Wiener / 5×5 / karma çekirdeklerin kenar ötesine uzanımı (`src[-3..w+3)`).
const REACH_3: usize = 3;
/// 3×3 çekirdeğin uzanımı (`src[-2..w+2)`).
const REACH_2: usize = 2;

// ---------------------------------------------------------------------------
// Ortak yardımcılar
// ---------------------------------------------------------------------------

/// Piksel başına bayt.
const fn bpp_of(bpc: u32) -> usize {
    if bpc == 8 { 1 } else { 2 }
}

/// Piksel okuma (8 bit: 1 bayt; 16bpc: 2 bayt küçük sonlu).
#[inline(always)]
fn px<const BPC: u32>(buf: &[u8], off: usize) -> i32 {
    if BPC == 8 {
        i32::from(buf[off])
    } else {
        i32::from(u16::from_le_bytes([buf[off], buf[off + 1]]))
    }
}

/// Piksel yazma; `v` önceden `[0, bitdepth_max]` aralığına kırpılmıştır.
#[inline(always)]
fn put<const BPC: u32>(buf: &mut [u8], off: usize, v: i32) {
    if BPC == 8 {
        buf[off] = v as u8;
    } else {
        buf[off..off + 2].copy_from_slice(&(v as u16).to_le_bytes());
    }
}

/// `iclip(v, min, max)` (dav1d `include/common/intops.h:51-53`).
#[inline(always)]
fn iclip(v: i32, min: i32, max: i32) -> i32 {
    if v < min {
        min
    } else if v > max {
        max
    } else {
        v
    }
}

/// `iclip_pixel(v)` = `iclip(v, 0, bitdepth_max)` (`common/bitdepth.h`).
#[inline(always)]
fn iclip_pixel<const BPC: u32>(v: i32) -> i32 {
    iclip(v, 0, (1 << BPC) - 1)
}

/// `coef` depolaması: 8 bit'te `int16_t` (`common/bitdepth.h`), 16bpc'de `int32_t`.
#[inline(always)]
fn coef<const BPC: u32>(v: i32) -> i32 {
    if BPC == 8 { i32::from(v as i16) } else { v }
}

/// Sözleşme denetimi: hata ayıklamada düşer, serbest derlemede `false` döner.
#[inline]
fn sozlesme(ok: bool, what: &str) -> bool {
    debug_assert!(ok, "loop restoration sözleşme dışı girdi: {what}");
    ok
}

/// Bir piksel satırının görünümü: `base` = sütun 0'ın bayt ofseti.
///
/// dav1d `const pixel *src`'nin karşılığıdır; negatif sütunlar yalnız `lpf`
/// satırlarında (`HAVE_LEFT`, `lpf_x >= reach`) okunur.
#[derive(Clone, Copy)]
struct Src<'a> {
    buf: &'a [u8],
    base: usize,
}

impl Src<'_> {
    #[inline(always)]
    fn at<const BPC: u32>(self, x: isize) -> i32 {
        let off = self.base as isize + x * bpp_of(BPC) as isize;
        px::<BPC>(self.buf, off as usize)
    }
}

/// Kaynak satır: `dst` satırı (sol komşu `left` tamponunda) ya da `lpf` satırı
/// (sol komşu aynı tamponda; dav1d `left == NULL`).
#[derive(Clone, Copy)]
enum RowRef {
    /// `dst` satırı `r` (`left[r]` ile).
    Main(usize),
    /// `lpf` satırı `k` (0,1 üst; 6,7 alt).
    Lpf(usize),
}

/// Çağrı bağlamı (değişmeyen girdiler).
struct Ctx<'l> {
    left: &'l [u8],
    lpf: &'l [u8],
    stride: usize,
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    edges: LrEdgeFlags,
}

impl<'l> Ctx<'l> {
    /// Bellek sözleşmesi (`dsp::lr` başlığı) sağlanıyor mu?
    fn gecerli(&self, dst_len: usize, h: usize, reach: usize, bpp: usize) -> bool {
        if self.w == 0 || self.w > LR_MAX_W || h == 0 {
            return false;
        }
        let right = if self.edges.contains(LrEdgeFlags::HAVE_RIGHT) {
            reach
        } else {
            0
        };
        let row_bytes = (self.w + right) * bpp;
        let dst_need = (h - 1)
            .checked_mul(self.stride)
            .and_then(|v| v.checked_add(row_bytes));
        if !matches!(dst_need, Some(n) if n <= dst_len) {
            return false;
        }
        let have_left = self.edges.contains(LrEdgeFlags::HAVE_LEFT);
        if have_left {
            let left_need = h.checked_mul(4 * bpp);
            if !matches!(left_need, Some(n) if n <= self.left.len()) {
                return false;
            }
        }
        let last_lpf_row = if self.edges.contains(LrEdgeFlags::HAVE_BOTTOM) {
            Some(7usize)
        } else if self.edges.contains(LrEdgeFlags::HAVE_TOP) {
            Some(1)
        } else {
            None
        };
        if let Some(r) = last_lpf_row {
            if have_left && self.lpf_x < reach {
                return false;
            }
            let need = r.checked_mul(self.lpf_stride).and_then(|v| {
                self.lpf_x
                    .checked_add(self.w + right)
                    .and_then(|p| p.checked_mul(bpp))
                    .and_then(|b| v.checked_add(b))
            });
            if !matches!(need, Some(n) if n <= self.lpf.len()) {
                return false;
            }
        }
        true
    }

    /// Satırın kaynak görünümü ve (varsa) sol komşu satırı.
    ///
    /// dav1d'de `dst` satırları için `left` her zaman geçerli bir işaretçidir
    /// ama yalnız `HAVE_LEFT` iken okunur; `lpf` satırları `left = NULL` ile
    /// çağrılır (`looprestoration_tmpl.c:276-278,366-372`).
    #[inline(always)]
    fn row<'a, const BPC: u32>(&'a self, dst: &'a [u8], row: RowRef) -> (Option<Src<'a>>, Src<'a>) {
        let bpp = bpp_of(BPC);
        match row {
            RowRef::Main(r) => {
                let left = if self.edges.contains(LrEdgeFlags::HAVE_LEFT) {
                    Some(Src {
                        buf: self.left,
                        base: r * 4 * bpp,
                    })
                } else {
                    None
                };
                (
                    left,
                    Src {
                        buf: dst,
                        base: r * self.stride,
                    },
                )
            }
            RowRef::Lpf(k) => (
                None,
                Src {
                    buf: self.lpf,
                    base: k * self.lpf_stride + self.lpf_x * bpp,
                },
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Genel giriş noktaları (`LrFilterFn` imzası)
// ---------------------------------------------------------------------------

/// Bağlamı kurar ve sözleşmeyi denetler; geçersizse `None`.
fn kur<'l>(
    dst_len: usize,
    stride: usize,
    left: &'l [u8],
    lpf: &'l [u8],
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
    reach: usize,
) -> Option<Ctx<'l>> {
    let c = Ctx {
        left,
        lpf,
        stride,
        lpf_stride,
        lpf_x,
        w,
        edges,
    };
    if sozlesme(c.gecerli(dst_len, h, reach, bd.bytes()), "dilim/boyut") {
        Some(c)
    } else {
        None
    }
}

/// Wiener 7 dokunuş — dav1d `lr.wiener[0]` = `wiener_c` (`looprestoration_tmpl.c:1362`).
pub fn wiener_7tap<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    lpf: &[u8],
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    params: &LrParams,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
) {
    let LrParams::Wiener { filter } = params else {
        sozlesme(false, "wiener: LrParams::Wiener bekleniyor");
        return;
    };
    let Some(c) = kur(
        dst.len(),
        stride,
        left,
        lpf,
        lpf_stride,
        lpf_x,
        w,
        h,
        edges,
        bd,
        REACH_3,
    ) else {
        return;
    };
    debug_assert_eq!(bd.bits(), BPC, "yuva bit derinliği ≠ çağrı");
    wiener::<BPC>(dst, &c, h, filter);
}

/// Wiener 5 dokunuş — dav1d `lr.wiener[1]`; C yolunda bu yuva **da**
/// `wiener_c`'dir (`looprestoration_tmpl.c:1362`), yani gövde 7 dokunuşla
/// aynıdır ve dış katsayılar (`filter[x][0] = filter[x][6] = 0`) sıfırdır.
/// Bu yüzden okuma aralığı da dav1d C ile aynıdır (`src[-3..w+3)`).
pub fn wiener_5tap<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    lpf: &[u8],
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    params: &LrParams,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
) {
    if let LrParams::Wiener { filter } = params {
        // Seçim kuralı `wiener[!(filter[0][0] | filter[1][0])]` (lr_apply_tmpl.c:72).
        debug_assert!(
            filter[0][0] == 0 && filter[1][0] == 0 && filter[0][6] == 0 && filter[1][6] == 0,
            "wiener_5tap: dış katsayılar sıfır olmalı"
        );
    }
    wiener_7tap::<BPC>(
        dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
    );
}

/// SGR 5×5 — dav1d `lr.sgr[0]` = `sgr_5x5_c` (`looprestoration_tmpl.c:819-1019`).
/// `s0` ve `w0` kullanılır.
pub fn sgr_5x5<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    lpf: &[u8],
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    params: &LrParams,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
) {
    let LrParams::Sgr { s0, w0, .. } = *params else {
        sozlesme(false, "sgr: LrParams::Sgr bekleniyor");
        return;
    };
    let Some(c) = kur(
        dst.len(),
        stride,
        left,
        lpf,
        lpf_stride,
        lpf_x,
        w,
        h,
        edges,
        bd,
        REACH_3,
    ) else {
        return;
    };
    debug_assert_eq!(bd.bits(), BPC, "yuva bit derinliği ≠ çağrı");
    sgr_5x5_bpc::<BPC>(dst, &c, h, s0, w0);
}

/// SGR 3×3 — dav1d `lr.sgr[1]` = `sgr_3x3_c` (`looprestoration_tmpl.c:673-817`).
/// `s1` ve `w1` kullanılır.
pub fn sgr_3x3<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    lpf: &[u8],
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    params: &LrParams,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
) {
    let LrParams::Sgr { s1, w1, .. } = *params else {
        sozlesme(false, "sgr: LrParams::Sgr bekleniyor");
        return;
    };
    let Some(c) = kur(
        dst.len(),
        stride,
        left,
        lpf,
        lpf_stride,
        lpf_x,
        w,
        h,
        edges,
        bd,
        REACH_2,
    ) else {
        return;
    };
    debug_assert_eq!(bd.bits(), BPC, "yuva bit derinliği ≠ çağrı");
    sgr_3x3_bpc::<BPC>(dst, &c, h, s1, w1);
}

/// SGR karma (5×5 + 3×3) — dav1d `lr.sgr[2]` = `sgr_mix_c`
/// (`looprestoration_tmpl.c:1021-1345`). `s0`, `s1`, `w0`, `w1` kullanılır.
pub fn sgr_mix<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    lpf: &[u8],
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    params: &LrParams,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
) {
    let LrParams::Sgr { s0, s1, w0, w1 } = *params else {
        sozlesme(false, "sgr: LrParams::Sgr bekleniyor");
        return;
    };
    let Some(c) = kur(
        dst.len(),
        stride,
        left,
        lpf,
        lpf_stride,
        lpf_x,
        w,
        h,
        edges,
        bd,
        REACH_3,
    ) else {
        return;
    };
    debug_assert_eq!(bd.bits(), BPC, "yuva bit derinliği ≠ çağrı");
    sgr_mix_bpc::<BPC>(dst, &c, h, s0, s1, w0, w1);
}

// ---------------------------------------------------------------------------
// Wiener (looprestoration_tmpl.c:44-386)
// ---------------------------------------------------------------------------

/// `wiener_filter_h` — `looprestoration_tmpl.c:44-161`.
fn wiener_filter_h<const BPC: u32>(
    out: &mut [u16],
    left: Option<Src<'_>>,
    src: Src<'_>,
    fh: &[i16; 8],
    w: usize,
    edges: LrEdgeFlags,
) {
    let bitdepth = BPC as i32;
    let round_bits_h = 3 + if BPC == 12 { 2 } else { 0 };
    let rounding_off_h = 1 << (round_bits_h - 1);
    let clip_limit = 1 << (bitdepth + 1 + 7 - round_bits_h);
    let have_left = edges.contains(LrEdgeFlags::HAVE_LEFT);
    let have_right = edges.contains(LrEdgeFlags::HAVE_RIGHT);
    let f: [i32; 7] = [
        i32::from(fh[0]),
        i32::from(fh[1]),
        i32::from(fh[2]),
        i32::from(fh[3]),
        i32::from(fh[4]),
        i32::from(fh[5]),
        i32::from(fh[6]),
    ];
    let wi = w as isize;
    // `int sum = (1 << (bitdepth + 6)); #if BITDEPTH == 8 sum += src[x] * 128;`
    let start_sum = |x: isize| -> i32 {
        let mut sum = 1 << (bitdepth + 6);
        if BPC == 8 {
            sum += src.at::<BPC>(x) * 128;
        }
        sum
    };
    let finish = |sum: i32| -> u16 {
        iclip((sum + rounding_off_h) >> round_bits_h, 0, clip_limit - 1) as u16
    };

    if w < 6 {
        // Küçük genişlik: her erişimde koşul (looprestoration_tmpl.c:55-81).
        for x in 0..wi {
            let mut sum = start_sum(x);
            for (i, &fi) in f.iter().enumerate() {
                let idx = x + i as isize - 3;
                let v = if idx < 0 {
                    if !have_left {
                        src.at::<BPC>(0)
                    } else if let Some(l) = left {
                        l.at::<BPC>(4 + idx)
                    } else {
                        src.at::<BPC>(idx)
                    }
                } else if idx >= wi && !have_right {
                    src.at::<BPC>(wi - 1)
                } else {
                    src.at::<BPC>(idx)
                };
                sum += v * fi;
            }
            out[x as usize] = finish(sum);
        }
        return;
    }

    // Satır başı (looprestoration_tmpl.c:83-125).
    let mut start = 3isize;
    if !have_left {
        for x in 0..3isize {
            let mut sum = start_sum(x);
            for (i, &fi) in f.iter().enumerate() {
                let idx = x + i as isize - 3;
                let v = if idx < 0 {
                    src.at::<BPC>(0)
                } else {
                    src.at::<BPC>(idx)
                };
                sum += v * fi;
            }
            out[x as usize] = finish(sum);
        }
    } else if let Some(l) = left {
        for x in 0..3isize {
            let mut sum = start_sum(x);
            for (i, &fi) in f.iter().enumerate() {
                let idx = x + i as isize - 3;
                let v = if idx < 0 {
                    l.at::<BPC>(4 + idx)
                } else {
                    src.at::<BPC>(idx)
                };
                sum += v * fi;
            }
            out[x as usize] = finish(sum);
        }
    } else {
        // `lpf` satırı: sol kenar aynı tamponda (looprestoration_tmpl.c:120-124).
        start = 0;
    }
    let end = if have_right { wi } else { wi - 3 };

    // Koşulsuz gövde (looprestoration_tmpl.c:130-142).
    for x in start..end {
        let mut sum = start_sum(x);
        for (i, &fi) in f.iter().enumerate() {
            sum += src.at::<BPC>(x + i as isize - 3) * fi;
        }
        out[x as usize] = finish(sum);
    }

    // Sağ kenar dolgusu (looprestoration_tmpl.c:144-160).
    for x in end..wi {
        let mut sum = start_sum(x);
        for (i, &fi) in f.iter().enumerate() {
            let idx = x + i as isize - 3;
            let v = if idx >= wi {
                src.at::<BPC>(wi - 1)
            } else {
                src.at::<BPC>(idx)
            };
            sum += v * fi;
        }
        out[x as usize] = finish(sum);
    }
}

/// Wiener dikey parametreleri (`looprestoration_tmpl.c:168-170,200-202`).
const fn wiener_v_params(bpc: u32) -> (i32, i32, i32) {
    let round_bits_v = 11 - if bpc == 12 { 2 } else { 0 };
    let rounding_off_v = 1 << (round_bits_v - 1);
    let round_offset = 1 << (bpc as i32 + (round_bits_v - 1));
    (round_bits_v, rounding_off_v, round_offset)
}

/// Wiener durum: `hor[6 * REST_UNIT_STRIDE]`, `ptrs[7]` (satır indeksleri).
struct WienerState {
    hor: [u16; 6 * REST_UNIT_STRIDE],
    tmp: [u16; REST_UNIT_STRIDE],
    ptrs: [usize; 7],
}

impl WienerState {
    #[inline(always)]
    fn row(&self, r: usize) -> &[u16] {
        &self.hor[r * REST_UNIT_STRIDE..(r + 1) * REST_UNIT_STRIDE]
    }

    /// `wiener_filter_h(rows[r], left, src, fh, w, edges)`.
    fn filter_h<const BPC: u32>(
        &mut self,
        c: &Ctx<'_>,
        dst: &[u8],
        row: RowRef,
        r: usize,
        fh: &[i16; 8],
    ) {
        let (left, src) = c.row::<BPC>(dst, row);
        let out = &mut self.hor[r * REST_UNIT_STRIDE..(r + 1) * REST_UNIT_STRIDE];
        wiener_filter_h::<BPC>(out, left, src, fh, c.w, c.edges);
    }

    /// `wiener_filter_v` — `looprestoration_tmpl.c:163-191`.
    fn filter_v<const BPC: u32>(
        &mut self,
        dst: &mut [u8],
        c: &Ctx<'_>,
        p_row: usize,
        fv: &[i16; 8],
    ) {
        let (round_bits_v, rounding_off_v, round_offset) = wiener_v_params(BPC);
        let bpp = bpp_of(BPC);
        let p = p_row * c.stride;
        let rows = [
            self.row(self.ptrs[0]),
            self.row(self.ptrs[1]),
            self.row(self.ptrs[2]),
            self.row(self.ptrs[3]),
            self.row(self.ptrs[4]),
            self.row(self.ptrs[5]),
        ];
        for i in 0..c.w {
            let mut sum = -round_offset;
            // Yalnız 6 girdi satırı; 7. satır 6. ile aynı kabul edilir.
            for (k, row) in rows.iter().enumerate() {
                sum += i32::from(row[i]) * i32::from(fv[k]);
            }
            sum += i32::from(rows[5][i]) * i32::from(fv[6]);
            put::<BPC>(
                dst,
                p + i * bpp,
                iclip_pixel::<BPC>((sum + rounding_off_v) >> round_bits_v),
            );
        }
        // Yalnız ilk 5 işaretçi kayar (`ptrs[i] = ptrs[i + 1]`, i < 5); 6. olduğu gibi kalır.
        self.ptrs.copy_within(1..6, 0);
    }

    /// `wiener_filter_hv` — `looprestoration_tmpl.c:193-246`.
    fn filter_hv<const BPC: u32>(
        &mut self,
        dst: &mut [u8],
        c: &Ctx<'_>,
        p_row: usize,
        row: RowRef,
        filter: &[[i16; 8]; 2],
    ) {
        let (round_bits_v, rounding_off_v, round_offset) = wiener_v_params(BPC);
        let bpp = bpp_of(BPC);
        let fv = &filter[1];
        {
            let (left, src) = c.row::<BPC>(dst, row);
            wiener_filter_h::<BPC>(&mut self.tmp, left, src, &filter[0], c.w, c.edges);
        }
        let p = p_row * c.stride;
        for i in 0..c.w {
            let mut sum = -round_offset;
            for k in 0..6 {
                sum += i32::from(self.row(self.ptrs[k])[i]) * i32::from(fv[k]);
            }
            sum += i32::from(self.tmp[i]) * i32::from(fv[6]);
            put::<BPC>(
                dst,
                p + i * bpp,
                iclip_pixel::<BPC>((sum + rounding_off_v) >> round_bits_v),
            );
        }
        // `memcpy(ptrs[6], tmp, ...)`: yalnız [0, w) sonradan okunur.
        let dst_row = self.ptrs[6];
        let w = c.w;
        self.hor[dst_row * REST_UNIT_STRIDE..dst_row * REST_UNIT_STRIDE + w]
            .copy_from_slice(&self.tmp[..w]);
        // `ptrs[i] = ptrs[i + 1]` (i < 6), sonra `ptrs[6] = ptrs[0]`.
        self.ptrs.copy_within(1..7, 0);
        self.ptrs[6] = self.ptrs[0];
    }
}

/// `wiener_c` — `looprestoration_tmpl.c:250-386`.
fn wiener<const BPC: u32>(dst: &mut [u8], c: &Ctx<'_>, h: usize, filter: &[[i16; 8]; 2]) {
    let fh = &filter[0];
    let fv = &filter[1];
    let mut st = WienerState {
        hor: [0; 6 * REST_UNIT_STRIDE],
        tmp: [0; REST_UNIT_STRIDE],
        ptrs: [0; 7],
    };
    let mut h = h;
    let mut src_row = 0usize;
    let mut p_row = 0usize;

    // Kuyruk: kaç tane yalnız-dikey (`wiener_filter_v`) satırı kaldı (v1/v2/v3).
    let tail: usize = 'akis: {
        if c.edges.contains(LrEdgeFlags::HAVE_TOP) {
            st.ptrs[..6].copy_from_slice(&[0, 0, 1, 2, 2, 2]);
            st.filter_h::<BPC>(c, dst, RowRef::Lpf(0), 0, fh);
            st.filter_h::<BPC>(c, dst, RowRef::Lpf(1), 1, fh);
            st.filter_h::<BPC>(c, dst, RowRef::Main(src_row), 2, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 1;
            }
            st.ptrs[4] = 3;
            st.ptrs[5] = 3;
            st.filter_h::<BPC>(c, dst, RowRef::Main(src_row), 3, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 2;
            }
            st.ptrs[5] = 4;
            st.filter_h::<BPC>(c, dst, RowRef::Main(src_row), 4, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
        } else {
            st.ptrs[..6].copy_from_slice(&[0; 6]);
            st.filter_h::<BPC>(c, dst, RowRef::Main(src_row), 0, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 1;
            }
            st.ptrs[4] = 1;
            st.ptrs[5] = 1;
            st.filter_h::<BPC>(c, dst, RowRef::Main(src_row), 1, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 2;
            }
            st.ptrs[5] = 2;
            st.filter_h::<BPC>(c, dst, RowRef::Main(src_row), 2, fh);
            src_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
            st.ptrs[6] = 3;
            st.filter_hv::<BPC>(dst, c, p_row, RowRef::Main(src_row), filter);
            src_row += 1;
            p_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
            st.ptrs[6] = 4;
            st.filter_hv::<BPC>(dst, c, p_row, RowRef::Main(src_row), filter);
            src_row += 1;
            p_row += 1;
            h -= 1;
            if h == 0 {
                break 'akis 3;
            }
        }

        // `ptrs[6] = ptrs[5] + REST_UNIT_STRIDE` (looprestoration_tmpl.c:354).
        debug_assert_eq!(st.ptrs[5], 4);
        st.ptrs[6] = st.ptrs[5] + 1;
        loop {
            st.filter_hv::<BPC>(dst, c, p_row, RowRef::Main(src_row), filter);
            src_row += 1;
            p_row += 1;
            h -= 1;
            if h == 0 {
                break;
            }
        }

        if !c.edges.contains(LrEdgeFlags::HAVE_BOTTOM) {
            break 'akis 3;
        }
        st.filter_hv::<BPC>(dst, c, p_row, RowRef::Lpf(6), filter);
        p_row += 1;
        st.filter_hv::<BPC>(dst, c, p_row, RowRef::Lpf(7), filter);
        p_row += 1;
        1
    };

    // v3 → v2 → v1 (looprestoration_tmpl.c:374-385).
    for _ in 0..tail {
        st.filter_v::<BPC>(dst, c, p_row, fv);
        p_row += 1;
    }
}

// ---------------------------------------------------------------------------
// SGR (looprestoration_tmpl.c:388-1345)
// ---------------------------------------------------------------------------

/// `N` satırlık çift tampon: `sumsq`/`sum` ya da `A`/`B` (dav1d `*_buf[BUF_STRIDE * N + 16]`).
struct Rows<const N: usize> {
    /// `int32_t` satırları (`sumsq` / `A`).
    sq: [[i32; BUF_STRIDE]; N],
    /// `coef` satırları (`sum` / `B`).
    s: [[i32; BUF_STRIDE]; N],
}

impl<const N: usize> Rows<N> {
    fn new() -> Self {
        Self {
            sq: [[0; BUF_STRIDE]; N],
            s: [[0; BUF_STRIDE]; N],
        }
    }
}

/// `sgr_box3_row_h` — `looprestoration_tmpl.c:413-429`.
fn box3_row_h<const BPC: u32>(
    sumsq: &mut [i32; BUF_STRIDE],
    sum: &mut [i32; BUF_STRIDE],
    left: Option<Src<'_>>,
    src: Src<'_>,
    w: usize,
    edges: LrEdgeFlags,
) {
    let have_left = edges.contains(LrEdgeFlags::HAVE_LEFT);
    let have_right = edges.contains(LrEdgeFlags::HAVE_RIGHT);
    let side = |k: isize| -> i32 {
        if have_left {
            match left {
                Some(l) => l.at::<BPC>(4 + k),
                None => src.at::<BPC>(k),
            }
        } else {
            src.at::<BPC>(0)
        }
    };
    let mut a = side(-2);
    let mut b = side(-1);
    let wi = w as isize;
    // `sumsq++; sum++;` → indeks `x + 1`.
    for x in -1..wi + 1 {
        let c = if x + 1 < wi || have_right {
            src.at::<BPC>(x + 1)
        } else {
            src.at::<BPC>(wi - 1)
        };
        let j = (x + 1) as usize;
        sum[j] = coef::<BPC>(a + b + c);
        sumsq[j] = a * a + b * b + c * c;
        a = b;
        b = c;
    }
}

/// `sgr_box5_row_h` — `looprestoration_tmpl.c:431-451`.
fn box5_row_h<const BPC: u32>(
    sumsq: &mut [i32; BUF_STRIDE],
    sum: &mut [i32; BUF_STRIDE],
    left: Option<Src<'_>>,
    src: Src<'_>,
    w: usize,
    edges: LrEdgeFlags,
) {
    let have_left = edges.contains(LrEdgeFlags::HAVE_LEFT);
    let have_right = edges.contains(LrEdgeFlags::HAVE_RIGHT);
    let side = |k: isize| -> i32 {
        if have_left {
            match left {
                Some(l) => l.at::<BPC>(4 + k),
                None => src.at::<BPC>(k),
            }
        } else {
            src.at::<BPC>(0)
        }
    };
    let mut a = side(-3);
    let mut b = side(-2);
    let mut c = side(-1);
    let mut d = src.at::<BPC>(0);
    let wi = w as isize;
    for x in -1..wi + 1 {
        let e = if x + 2 < wi || have_right {
            src.at::<BPC>(x + 2)
        } else {
            src.at::<BPC>(wi - 1)
        };
        let j = (x + 1) as usize;
        sum[j] = coef::<BPC>(a + b + c + d + e);
        sumsq[j] = a * a + b * b + c * c + d * d + e * e;
        a = b;
        b = c;
        c = d;
        d = e;
    }
}

/// `sgr_calc_row_ab` — `looprestoration_tmpl.c:499-517`.
fn calc_row_ab<const BPC: u32>(
    aa: &mut [i32; BUF_STRIDE],
    bb: &mut [i32; BUF_STRIDE],
    w: usize,
    s: u32,
    n: i32,
    one_by_x: u32,
) {
    let bitdepth_min_8 = BPC - 8;
    for i in 0..w + 2 {
        let a = (aa[i] + ((1 << (2 * bitdepth_min_8)) >> 1)) >> (2 * bitdepth_min_8);
        let b = (bb[i] + ((1 << bitdepth_min_8) >> 1)) >> bitdepth_min_8;

        let p = (a * n - b * b).max(0) as u32;
        let z = p.wrapping_mul(s).wrapping_add(1 << 19) >> 20;
        let x = u32::from(SGR_X_BY_X[z.min(255) as usize]);

        // A ile B burada yer değiştirir; B `coef` boyutuna sığar.
        aa[i] = (x
            .wrapping_mul(bb[i] as u32)
            .wrapping_mul(one_by_x)
            .wrapping_add(1 << 11)
            >> 12) as i32;
        bb[i] = coef::<BPC>(x as i32);
    }
}

/// `sgr_box3_vert` (+`sgr_box3_row_v`) — `looprestoration_tmpl.c:463-477,519-526`.
fn box3_vert<const BPC: u32, const K: usize>(
    sums: &Rows<3>,
    sp: &mut [usize; 3],
    ab: &mut Rows<K>,
    out: usize,
    w: usize,
    s: u32,
) {
    let (oa, ob) = (&mut ab.sq[out], &mut ab.s[out]);
    let (q0, q1, q2) = (&sums.sq[sp[0]], &sums.sq[sp[1]], &sums.sq[sp[2]]);
    let (s0, s1, s2) = (&sums.s[sp[0]], &sums.s[sp[1]], &sums.s[sp[2]]);
    for x in 0..w + 2 {
        oa[x] = q0[x] + q1[x] + q2[x];
        ob[x] = coef::<BPC>(s0[x] + s1[x] + s2[x]);
    }
    calc_row_ab::<BPC>(oa, ob, w, s, 9, 455);
    sp.rotate_left(1);
}

/// `sgr_box5_vert` (+`sgr_box5_row_v`) — `looprestoration_tmpl.c:479-497,528-535`.
fn box5_vert<const BPC: u32>(
    sums: &Rows<5>,
    sp: &mut [usize; 5],
    ab: &mut Rows<2>,
    out: usize,
    w: usize,
    s: u32,
) {
    let (oa, ob) = (&mut ab.sq[out], &mut ab.s[out]);
    let q = [
        &sums.sq[sp[0]],
        &sums.sq[sp[1]],
        &sums.sq[sp[2]],
        &sums.sq[sp[3]],
        &sums.sq[sp[4]],
    ];
    let t = [
        &sums.s[sp[0]],
        &sums.s[sp[1]],
        &sums.s[sp[2]],
        &sums.s[sp[3]],
        &sums.s[sp[4]],
    ];
    for x in 0..w + 2 {
        oa[x] = q[0][x] + q[1][x] + q[2][x] + q[3][x] + q[4][x];
        ob[x] = coef::<BPC>(t[0][x] + t[1][x] + t[2][x] + t[3][x] + t[4][x]);
    }
    calc_row_ab::<BPC>(oa, ob, w, s, 25, 164);
    // `rotate5_x2` (looprestoration_tmpl.c:401-411).
    sp.rotate_left(2);
}

/// `sgr_finish_filter_row1` (`EIGHT_NEIGHBORS`) — `looprestoration_tmpl.c:549-564`.
fn finish_filter_row1<const BPC: u32, const K: usize>(
    tmp: &mut [i32],
    src: Src<'_>,
    ab: &Rows<K>,
    ap: [usize; 3],
    w: usize,
) {
    let (b0, b1, b2) = (&ab.s[ap[0]], &ab.s[ap[1]], &ab.s[ap[2]]);
    let (a0, a1, a2) = (&ab.sq[ap[0]], &ab.sq[ap[1]], &ab.sq[ap[2]]);
    for i in 0..w {
        let j = i + 1;
        let a = (b1[j] + b1[j - 1] + b1[j + 1] + b0[j] + b2[j]) * 4
            + (b0[j - 1] + b2[j - 1] + b0[j + 1] + b2[j + 1]) * 3;
        let b = (a1[j] + a1[j - 1] + a1[j + 1] + a0[j] + a2[j]) * 4
            + (a0[j - 1] + a2[j - 1] + a0[j + 1] + a2[j + 1]) * 3;
        tmp[i] = coef::<BPC>((b - a * src.at::<BPC>(i as isize) + (1 << 8)) >> 9);
    }
}

/// `sgr_finish_filter2` (`SIX_NEIGHBORS`) — `looprestoration_tmpl.c:568-595`.
fn finish_filter2<const BPC: u32>(
    tmp: &mut [i32; 2 * FILTER_OUT_STRIDE],
    src0: Src<'_>,
    src1: Src<'_>,
    ab: &Rows<2>,
    ap: [usize; 2],
    w: usize,
    h: usize,
) {
    let (b0, b1) = (&ab.s[ap[0]], &ab.s[ap[1]]);
    let (a0, a1) = (&ab.sq[ap[0]], &ab.sq[ap[1]]);
    for i in 0..w {
        let j = i + 1;
        let a = (b0[j] + b1[j]) * 6 + (b0[j - 1] + b1[j - 1] + b0[j + 1] + b1[j + 1]) * 5;
        let b = (a0[j] + a1[j]) * 6 + (a0[j - 1] + a1[j - 1] + a0[j + 1] + a1[j + 1]) * 5;
        tmp[i] = coef::<BPC>((b - a * src0.at::<BPC>(i as isize) + (1 << 8)) >> 9);
    }
    if h <= 1 {
        return;
    }
    for i in 0..w {
        let j = i + 1;
        let a = b1[j] * 6 + (b1[j - 1] + b1[j + 1]) * 5;
        let b = a1[j] * 6 + (a1[j - 1] + a1[j + 1]) * 5;
        tmp[FILTER_OUT_STRIDE + i] =
            coef::<BPC>((b - a * src1.at::<BPC>(i as isize) + (1 << 7)) >> 8);
    }
}

/// `sgr_weighted_row1` — `looprestoration_tmpl.c:597-604`.
fn weighted_row1<const BPC: u32>(dst: &mut [u8], off: usize, t1: &[i32], w: usize, w1: i32) {
    let bpp = bpp_of(BPC);
    for i in 0..w {
        let v = w1 * t1[i];
        let o = off + i * bpp;
        let d = px::<BPC>(dst, o);
        put::<BPC>(dst, o, iclip_pixel::<BPC>(d + ((v + (1 << 10)) >> 11)));
    }
}

/// `sgr_weighted2` — `looprestoration_tmpl.c:606-620`.
fn weighted2<const BPC: u32>(
    dst: &mut [u8],
    off: usize,
    stride: usize,
    t1: &[i32; 2 * FILTER_OUT_STRIDE],
    t2: &[i32; 2 * FILTER_OUT_STRIDE],
    w: usize,
    h: usize,
    w0: i32,
    w1: i32,
) {
    let bpp = bpp_of(BPC);
    for j in 0..h {
        let row = off + j * stride;
        let t = j * FILTER_OUT_STRIDE;
        for i in 0..w {
            let v = w0 * t1[t + i] + w1 * t2[t + i];
            let o = row + i * bpp;
            let d = px::<BPC>(dst, o);
            put::<BPC>(dst, o, iclip_pixel::<BPC>(d + ((v + (1 << 10)) >> 11)));
        }
    }
}

/// `dst` satırının kaynak görünümü (sonlandırma süzgeçleri `*dst`'i okur).
#[inline(always)]
fn dst_src<'a>(dst: &'a [u8], c: &Ctx<'_>, r: usize) -> Src<'a> {
    Src {
        buf: dst,
        base: r * c.stride,
    }
}

/// `sgr_finish1` — `looprestoration_tmpl.c:622-633`.
fn finish1<const BPC: u32>(
    dst: &mut [u8],
    c: &Ctx<'_>,
    dst_row: &mut usize,
    ab: &Rows<3>,
    ap: &mut [usize; 3],
    tmp: &mut [i32; 2 * FILTER_OUT_STRIDE],
    w1: i32,
) {
    finish_filter_row1::<BPC, 3>(tmp, dst_src(dst, c, *dst_row), ab, *ap, c.w);
    weighted_row1::<BPC>(dst, *dst_row * c.stride, &tmp[..], c.w, w1);
    *dst_row += 1;
    ap.rotate_left(1);
}

/// `sgr_finish2` — `looprestoration_tmpl.c:635-650`.
fn finish2<const BPC: u32>(
    dst: &mut [u8],
    c: &Ctx<'_>,
    dst_row: &mut usize,
    ab: &Rows<2>,
    ap: &mut [usize; 2],
    tmp: &mut [i32; 2 * FILTER_OUT_STRIDE],
    h: usize,
    w0: i32,
) {
    let r = *dst_row;
    // `h <= 1` iken ikinci satır okunmaz (finish_filter2 erken döner).
    let r1 = if h > 1 { r + 1 } else { r };
    finish_filter2::<BPC>(
        tmp,
        dst_src(dst, c, r),
        dst_src(dst, c, r1),
        ab,
        *ap,
        c.w,
        h,
    );
    weighted_row1::<BPC>(dst, r * c.stride, &tmp[..], c.w, w0);
    *dst_row += 1;
    if h > 1 {
        weighted_row1::<BPC>(dst, (r + 1) * c.stride, &tmp[FILTER_OUT_STRIDE..], c.w, w0);
        *dst_row += 1;
    }
    ap.rotate_left(1);
}

/// `sgr_finish_mix` — `looprestoration_tmpl.c:652-670`.
fn finish_mix<const BPC: u32>(
    dst: &mut [u8],
    c: &Ctx<'_>,
    dst_row: &mut usize,
    ab5: &Rows<2>,
    ap5: &mut [usize; 2],
    ab3: &Rows<4>,
    ap3: &mut [usize; 4],
    tmp5: &mut [i32; 2 * FILTER_OUT_STRIDE],
    tmp3: &mut [i32; 2 * FILTER_OUT_STRIDE],
    h: usize,
    w0: i32,
    w1: i32,
) {
    let r = *dst_row;
    let r1 = if h > 1 { r + 1 } else { r };
    finish_filter2::<BPC>(
        tmp5,
        dst_src(dst, c, r),
        dst_src(dst, c, r1),
        ab5,
        *ap5,
        c.w,
        h,
    );
    finish_filter_row1::<BPC, 4>(
        &mut tmp3[..],
        dst_src(dst, c, r),
        ab3,
        [ap3[0], ap3[1], ap3[2]],
        c.w,
    );
    if h > 1 {
        finish_filter_row1::<BPC, 4>(
            &mut tmp3[FILTER_OUT_STRIDE..],
            dst_src(dst, c, r + 1),
            ab3,
            [ap3[1], ap3[2], ap3[3]],
            c.w,
        );
    }
    weighted2::<BPC>(dst, r * c.stride, c.stride, tmp5, tmp3, c.w, h, w0, w1);
    *dst_row += h;
    ap5.rotate_left(1);
    ap3.rotate_left(1);
}

/// `sgr_3x3_c` — `looprestoration_tmpl.c:673-817`.
fn sgr_3x3_bpc<const BPC: u32>(dst: &mut [u8], c: &Ctx<'_>, h: usize, s1: u32, w1: i32) {
    /// `goto` hedefleri.
    enum Kuyruk {
        /// Normal dönüş (BOTTOM satırları işlendi).
        Bitti,
        /// `vert_2`.
        Vert2,
        /// `vert_1`.
        Vert1,
    }
    let w = c.w;
    let mut sums = Rows::<3>::new();
    let mut ab = Rows::<3>::new();
    let mut sp: [usize; 3];
    let mut ap = [0usize, 1, 2];
    let mut tmp = [0i32; 2 * FILTER_OUT_STRIDE];
    let mut h = h;
    let mut sr = 0usize;
    let mut dr = 0usize;

    // `sgr_box3_row_h(sumsq[r], sum[r], left?, src, w, edges)`.
    let row_h = |sums: &mut Rows<3>, dst: &[u8], row: RowRef, r: usize| {
        let (left, src) = c.row::<BPC>(dst, row);
        box3_row_h::<BPC>(&mut sums.sq[r], &mut sums.s[r], left, src, w, c.edges);
    };

    let kuyruk = 'akis: {
        if c.edges.contains(LrEdgeFlags::HAVE_TOP) {
            sp = [0, 1, 2];
            row_h(&mut sums, dst, RowRef::Lpf(0), 0);
            row_h(&mut sums, dst, RowRef::Lpf(1), 1);
            // sgr_box3_hv
            row_h(&mut sums, dst, RowRef::Main(sr), sp[2]);
            box3_vert::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            row_h(&mut sums, dst, RowRef::Main(sr), sp[2]);
            box3_vert::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
        } else {
            sp = [0, 0, 0];
            row_h(&mut sums, dst, RowRef::Main(sr), 0);
            sr += 1;
            box3_vert::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp[2] = 1;
            row_h(&mut sums, dst, RowRef::Main(sr), sp[2]);
            box3_vert::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp[2] = 2;
        }

        loop {
            row_h(&mut sums, dst, RowRef::Main(sr), sp[2]);
            box3_vert::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            sr += 1;
            finish1::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
            h -= 1;
            if h == 0 {
                break;
            }
        }

        if !c.edges.contains(LrEdgeFlags::HAVE_BOTTOM) {
            break 'akis Kuyruk::Vert2;
        }
        row_h(&mut sums, dst, RowRef::Lpf(6), sp[2]);
        box3_vert::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
        finish1::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        row_h(&mut sums, dst, RowRef::Lpf(7), sp[2]);
        box3_vert::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
        finish1::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        Kuyruk::Bitti
    };

    match kuyruk {
        Kuyruk::Bitti => return,
        Kuyruk::Vert2 => {
            sp[2] = sp[1];
            box3_vert::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            finish1::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
        }
        Kuyruk::Vert1 => {
            sp[2] = sp[1];
            box3_vert::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
            ap.rotate_left(1);
        }
    }
    // output_1
    sp[2] = sp[1];
    box3_vert::<BPC, 3>(&sums, &mut sp, &mut ab, ap[2], w, s1);
    finish1::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, w1);
}

/// `sgr_5x5_c` — `looprestoration_tmpl.c:819-1019`.
fn sgr_5x5_bpc<const BPC: u32>(dst: &mut [u8], c: &Ctx<'_>, h: usize, s0: u32, w0: i32) {
    /// `goto` hedefleri.
    enum Kuyruk {
        /// `output_2` (BOTTOM satırları okundu).
        Output2,
        /// `vert_2`.
        Vert2,
        /// `odd`.
        Odd,
        /// `vert_1`.
        Vert1,
    }
    let w = c.w;
    let mut sums = Rows::<5>::new();
    let mut ab = Rows::<2>::new();
    let mut sp: [usize; 5];
    let mut ap = [0usize, 1];
    let mut tmp = [0i32; 2 * FILTER_OUT_STRIDE];
    let mut h = h;
    let mut sr = 0usize;
    let mut dr = 0usize;

    let row_h = |sums: &mut Rows<5>, dst: &[u8], row: RowRef, r: usize| {
        let (left, src) = c.row::<BPC>(dst, row);
        box5_row_h::<BPC>(&mut sums.sq[r], &mut sums.s[r], left, src, w, c.edges);
    };

    let kuyruk = 'akis: {
        if c.edges.contains(LrEdgeFlags::HAVE_TOP) {
            sp = [0, 0, 1, 2, 3];
            row_h(&mut sums, dst, RowRef::Lpf(0), 0);
            row_h(&mut sums, dst, RowRef::Lpf(1), 1);
            row_h(&mut sums, dst, RowRef::Main(sr), 2);
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            row_h(&mut sums, dst, RowRef::Main(sr), 3);
            sr += 1;
            box5_vert::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            // İşaretçiler 2 döndü; [3] ve [4] satır 0'ı gösterir, [3] boş satır 4'e.
            sp[3] = 4;
        } else {
            sp = [0; 5];
            row_h(&mut sums, dst, RowRef::Main(sr), 0);
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp[4] = 1;
            row_h(&mut sums, dst, RowRef::Main(sr), 1);
            sr += 1;
            box5_vert::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
            ap.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp[3] = 2;
            sp[4] = 3;
            row_h(&mut sums, dst, RowRef::Main(sr), 2);
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            row_h(&mut sums, dst, RowRef::Main(sr), 3);
            sr += 1;
            box5_vert::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp[3] = 4;
        }

        loop {
            row_h(&mut sums, dst, RowRef::Main(sr), sp[3]);
            sr += 1;
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            row_h(&mut sums, dst, RowRef::Main(sr), sp[4]);
            sr += 1;
            box5_vert::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
            h -= 1;
            if h == 0 {
                break;
            }
        }

        if !c.edges.contains(LrEdgeFlags::HAVE_BOTTOM) {
            break 'akis Kuyruk::Vert2;
        }
        row_h(&mut sums, dst, RowRef::Lpf(6), sp[3]);
        row_h(&mut sums, dst, RowRef::Lpf(7), sp[4]);
        Kuyruk::Output2
    };

    let output_1 = match kuyruk {
        Kuyruk::Output2 => false,
        Kuyruk::Vert2 => {
            // Son satırı iki kez daha çoğalt.
            sp[3] = sp[2];
            sp[4] = sp[2];
            false
        }
        Kuyruk::Odd => {
            sp[4] = sp[3];
            box5_vert::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
            finish2::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
            true
        }
        Kuyruk::Vert1 => {
            sp[4] = sp[3];
            box5_vert::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
            ap.rotate_left(1);
            true
        }
    };
    if output_1 {
        sp[3] = sp[2];
        sp[4] = sp[2];
        box5_vert::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
        // Yalnız bir satır çıkar.
        finish2::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 1, w0);
    } else {
        // output_2
        box5_vert::<BPC>(&sums, &mut sp, &mut ab, ap[1], w, s0);
        finish2::<BPC>(dst, c, &mut dr, &ab, &mut ap, &mut tmp, 2, w0);
    }
}

/// `sgr_mix_c` — `looprestoration_tmpl.c:1021-1345`.
fn sgr_mix_bpc<const BPC: u32>(
    dst: &mut [u8],
    c: &Ctx<'_>,
    h: usize,
    s0: u32,
    s1: u32,
    w0: i32,
    w1: i32,
) {
    /// `goto` hedefleri.
    enum Kuyruk {
        /// `output_2` (BOTTOM satırları okundu).
        Output2,
        /// `vert_2`.
        Vert2,
        /// `odd`.
        Odd,
        /// `vert_1`.
        Vert1,
    }
    let w = c.w;
    let mut sums5 = Rows::<5>::new();
    let mut sums3 = Rows::<3>::new();
    let mut ab5 = Rows::<2>::new();
    let mut ab3 = Rows::<4>::new();
    let mut sp5: [usize; 5];
    let mut sp3: [usize; 3];
    let mut ap5 = [0usize, 1];
    let mut ap3 = [0usize, 1, 2, 3];
    let mut tmp5 = [0i32; 2 * FILTER_OUT_STRIDE];
    let mut tmp3 = [0i32; 2 * FILTER_OUT_STRIDE];
    let mut h = h;
    let mut sr = 0usize;
    let mut dr = 0usize;

    // `sgr_box35_row_h(sumsq3[r3], sum3[r3], sumsq5[r5], sum5[r5], left?, src, w, edges)`.
    let row_h35 =
        |s3: &mut Rows<3>, s5: &mut Rows<5>, dst: &[u8], row: RowRef, r3: usize, r5: usize| {
            let (left, src) = c.row::<BPC>(dst, row);
            box3_row_h::<BPC>(&mut s3.sq[r3], &mut s3.s[r3], left, src, w, c.edges);
            box5_row_h::<BPC>(&mut s5.sq[r5], &mut s5.s[r5], left, src, w, c.edges);
        };

    let kuyruk = 'akis: {
        if c.edges.contains(LrEdgeFlags::HAVE_TOP) {
            sp5 = [0, 0, 1, 2, 3];
            sp3 = [0, 1, 2];
            row_h35(&mut sums3, &mut sums5, dst, RowRef::Lpf(0), 0, 0);
            row_h35(&mut sums3, &mut sums5, dst, RowRef::Lpf(1), 1, 1);
            row_h35(&mut sums3, &mut sums5, dst, RowRef::Main(sr), 2, 2);
            sr += 1;
            box3_vert::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            row_h35(&mut sums3, &mut sums5, dst, RowRef::Main(sr), sp3[2], 3);
            sr += 1;
            box5_vert::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            ap5.rotate_left(1);
            box3_vert::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp5[3] = 4;
        } else {
            sp5 = [0; 5];
            sp3 = [0; 3];
            row_h35(&mut sums3, &mut sums5, dst, RowRef::Main(sr), 0, 0);
            sr += 1;
            box3_vert::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert1;
            }
            sp5[4] = 1;
            sp3[2] = 1;
            row_h35(&mut sums3, &mut sums5, dst, RowRef::Main(sr), 1, 1);
            sr += 1;
            box5_vert::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            ap5.rotate_left(1);
            box3_vert::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp5[3] = 2;
            sp5[4] = 3;
            sp3[2] = 2;
            row_h35(&mut sums3, &mut sums5, dst, RowRef::Main(sr), 2, 2);
            sr += 1;
            box3_vert::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            row_h35(&mut sums3, &mut sums5, dst, RowRef::Main(sr), sp3[2], 3);
            sr += 1;
            box5_vert::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix::<BPC>(
                dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 2, w0, w1,
            );
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Vert2;
            }
            sp5[3] = 4;
        }

        loop {
            row_h35(
                &mut sums3,
                &mut sums5,
                dst,
                RowRef::Main(sr),
                sp3[2],
                sp5[3],
            );
            sr += 1;
            box3_vert::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            h -= 1;
            if h == 0 {
                break 'akis Kuyruk::Odd;
            }
            row_h35(
                &mut sums3,
                &mut sums5,
                dst,
                RowRef::Main(sr),
                sp3[2],
                sp5[4],
            );
            sr += 1;
            box5_vert::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix::<BPC>(
                dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 2, w0, w1,
            );
            h -= 1;
            if h == 0 {
                break;
            }
        }

        if !c.edges.contains(LrEdgeFlags::HAVE_BOTTOM) {
            break 'akis Kuyruk::Vert2;
        }
        row_h35(&mut sums3, &mut sums5, dst, RowRef::Lpf(6), sp3[2], sp5[3]);
        box3_vert::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
        ap3.rotate_left(1);
        row_h35(&mut sums3, &mut sums5, dst, RowRef::Lpf(7), sp3[2], sp5[4]);
        Kuyruk::Output2
    };

    let output_1 = match kuyruk {
        Kuyruk::Output2 => false,
        Kuyruk::Vert2 => {
            // Son satırı iki kez daha çoğalt.
            sp5[3] = sp5[2];
            sp5[4] = sp5[2];
            sp3[2] = sp3[1];
            box3_vert::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            sp3[2] = sp3[1];
            false
        }
        Kuyruk::Odd => {
            sp5[4] = sp5[3];
            sp3[2] = sp3[1];
            box5_vert::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            box3_vert::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            finish_mix::<BPC>(
                dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 2, w0, w1,
            );
            true
        }
        Kuyruk::Vert1 => {
            sp5[4] = sp5[3];
            sp3[2] = sp3[1];
            box5_vert::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
            ap5.rotate_left(1);
            box3_vert::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
            ap3.rotate_left(1);
            true
        }
    };
    if output_1 {
        sp5[3] = sp5[2];
        sp5[4] = sp5[2];
        sp3[2] = sp3[1];
        box5_vert::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
        box3_vert::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
        ap3.rotate_left(1);
        // Yalnız bir satır çıkar.
        finish_mix::<BPC>(
            dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 1, w0, w1,
        );
    } else {
        // output_2
        box5_vert::<BPC>(&sums5, &mut sp5, &mut ab5, ap5[1], w, s0);
        box3_vert::<BPC, 4>(&sums3, &mut sp3, &mut ab3, ap3[3], w, s1);
        finish_mix::<BPC>(
            dst, c, &mut dr, &ab5, &mut ap5, &ab3, &mut ap3, &mut tmp5, &mut tmp3, 2, w0, w1,
        );
    }
}

#[cfg(test)]
mod tests {
    //! Oracle'dan bağımsız sözleşme testleri (dav1d differential'ı:
    //! `tests/lr_scalar_oracle.rs` + `scripts/oracle/run_lr_oracle.sh`).

    use super::{REACH_2, REACH_3};
    use crate::dsp::lr::{LrBitDepth, LrDsp, LrEdgeFlags, LrFilterFn, LrParams};
    use crate::dsp::lr_tables::SGR_PARAMS;

    /// Basit xorshift (yalnız test).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        fn range(&mut self, lo: i32, hi: i32) -> i32 {
            lo + self.below((hi - lo + 1) as u64) as i32
        }
    }

    fn put_px(buf: &mut [u8], off: usize, bpp: usize, v: u16) {
        if bpp == 1 {
            buf[off] = v as u8;
        } else {
            buf[off..off + 2].copy_from_slice(&v.to_le_bytes());
        }
    }

    /// Beş çekirdek: (ad, işlev, uzanım, 5 dokunuş mu, SGR mi).
    fn kernels(bd: LrBitDepth) -> [(&'static str, LrFilterFn, usize, bool, bool); 5] {
        let d = LrDsp::scalar_for(bd);
        [
            ("wiener7", d.wiener[0], REACH_3, false, false),
            ("wiener5", d.wiener[1], REACH_3, true, false),
            ("sgr5x5", d.sgr[0], REACH_3, false, true),
            ("sgr3x3", d.sgr[1], REACH_2, false, true),
            ("sgrmix", d.sgr[2], REACH_3, false, true),
        ]
    }

    fn wiener_params(rng: &mut Rng, five: bool, bd: LrBitDepth) -> LrParams {
        let (lo, hi) = ([-5, -23, -17], [10, 8, 46]);
        let mut f = [[0i16; 8]; 2];
        for (pass, row) in f.iter_mut().enumerate() {
            let mut t = [0i32; 3];
            for i in 0..3 {
                t[i] = rng.range(lo[i], hi[i]);
            }
            if five {
                t[0] = 0;
            }
            row[0] = t[0] as i16;
            row[6] = t[0] as i16;
            row[1] = t[1] as i16;
            row[5] = t[1] as i16;
            row[2] = t[2] as i16;
            row[4] = t[2] as i16;
            row[3] = if pass == 0 {
                (-(t[0] + t[1] + t[2]) * 2 + if bd == LrBitDepth::B8 { 0 } else { 128 }) as i16
            } else {
                (128 - (t[0] + t[1] + t[2]) * 2) as i16
            };
        }
        LrParams::Wiener { filter: f }
    }

    fn sgr_params(rng: &mut Rng) -> LrParams {
        let idx = rng.below(16) as usize;
        let w0 = rng.range(-96, 31);
        let w1r = rng.range(-32, 95);
        LrParams::Sgr {
            s0: u32::from(SGR_PARAMS[idx][0]),
            s1: u32::from(SGR_PARAMS[idx][1]),
            w0,
            w1: 128 - (w0 + w1r),
        }
    }

    /// Sözleşme-sıkı tamponlar: okunabilir bölge `base`'ten, geri kalan `poison`'dan.
    struct Bufs {
        dst: Vec<u8>,
        stride: usize,
        left: Vec<u8>,
        lpf: Vec<u8>,
        lpf_stride: usize,
        lpf_x: usize,
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        base_seed: u64,
        poison_seed: u64,
        bd: LrBitDepth,
        edges: LrEdgeFlags,
        w: usize,
        h: usize,
        reach: usize,
    ) -> Bufs {
        let bpp = bd.bytes();
        let max = u64::from(bd.bitdepth_max()) + 1;
        let mut base = Rng(base_seed);
        let mut poison = Rng(poison_seed);
        let have_left = edges.contains(LrEdgeFlags::HAVE_LEFT);
        let have_right = edges.contains(LrEdgeFlags::HAVE_RIGHT);
        let lreach = if have_left { reach } else { 0 };
        let rreach = if have_right { reach } else { 0 };
        // dst: satır adımı readable + 2 piksel; dilim tam sınırda biter.
        let stride = (w + rreach + 2) * bpp;
        let dst_len = (h - 1) * stride + (w + rreach) * bpp;
        let mut dst = vec![0u8; dst_len];
        for y in 0..h {
            for x in 0..stride / bpp {
                let o = y * stride + x * bpp;
                if o + bpp > dst_len {
                    break;
                }
                let readable = x < w + rreach;
                let v = if readable {
                    base.below(max)
                } else {
                    poison.below(max)
                };
                put_px(&mut dst, o, bpp, v as u16);
            }
        }
        let mut left = Vec::new();
        if have_left {
            left = vec![0u8; h * 4 * bpp];
            for y in 0..h {
                for k in 0..4 {
                    let v = if k >= 4 - reach {
                        base.below(max)
                    } else {
                        poison.below(max)
                    };
                    put_px(&mut left, (y * 4 + k) * bpp, bpp, v as u16);
                }
            }
        }
        let lpf_x = lreach + 1;
        let lpf_stride = (lpf_x + w + rreach + 3) * bpp;
        let have_top = edges.contains(LrEdgeFlags::HAVE_TOP);
        let have_bottom = edges.contains(LrEdgeFlags::HAVE_BOTTOM);
        let last = if have_bottom {
            Some(7)
        } else if have_top {
            Some(1)
        } else {
            None
        };
        let mut lpf = Vec::new();
        if let Some(last) = last {
            let len = last * lpf_stride + (lpf_x + w + rreach) * bpp;
            lpf = vec![0u8; len];
            for r in 0..8usize {
                let flagged = (r < 2 && have_top) || (r >= 6 && have_bottom);
                for col in 0..lpf_stride / bpp {
                    let o = r * lpf_stride + col * bpp;
                    if o + bpp > len {
                        break;
                    }
                    let x = col as isize - lpf_x as isize;
                    let readable = flagged && x >= -(lreach as isize) && x < (w + rreach) as isize;
                    let v = if readable {
                        base.below(max)
                    } else {
                        poison.below(max)
                    };
                    put_px(&mut lpf, o, bpp, v as u16);
                }
            }
        }
        Bufs {
            dst,
            stride,
            left,
            lpf,
            lpf_stride,
            lpf_x,
        }
    }

    /// Taslak bellek koruması: dilimler sözleşmenin tam sınırında; okunmasına
    /// izin verilmeyen her bayt iki koşuda **farklı** zehir taşır. Çıktı aynı
    /// olmalı (zehir okunmadı) ve `[0, w)` dışı hiçbir bayt değişmemeli
    /// (yazılmadı). `HAVE_LEFT` yoksa `left`, `HAVE_TOP/BOTTOM` yoksa `lpf` boştur.
    #[test]
    fn sinir_dilim_zehir_okunmaz_yazilmaz() {
        let sizes = [
            (1usize, 1usize),
            (2, 3),
            (5, 4),
            (6, 5),
            (7, 2),
            (33, 7),
            (64, 9),
        ];
        let mut rng = Rng(0x00c0_ffee_1234_5678);
        let mut runs = 0usize;
        for k in 0..5 {
            for bd in LrBitDepth::ALL {
                let (name, kernel, reach, five, sgr) = kernels(bd)[k];
                for e in 0..16u8 {
                    let edges = LrEdgeFlags::from_bits(e);
                    for &(w, h) in &sizes {
                        let params = if sgr {
                            sgr_params(&mut rng)
                        } else {
                            wiener_params(&mut rng, five, bd)
                        };
                        let seed = rng.next() | 1;
                        let mut a = build(seed, 0x1111_2222_3333_4445, bd, edges, w, h, reach);
                        let mut b = build(seed, 0x9999_8888_7777_6667, bd, edges, w, h, reach);
                        let (a0, b0) = (a.dst.clone(), b.dst.clone());
                        for x in [&mut a, &mut b] {
                            kernel(
                                &mut x.dst,
                                x.stride,
                                &x.left,
                                &x.lpf,
                                x.lpf_stride,
                                x.lpf_x,
                                w,
                                h,
                                &params,
                                edges,
                                bd,
                            );
                        }
                        let bpp = bd.bytes();
                        for y in 0..h {
                            let row = y * a.stride;
                            let n = w * bpp;
                            assert_eq!(
                                a.dst[row..row + n],
                                b.dst[row..row + n],
                                "{name} {bd:?} edges={e:04b} {w}x{h}: zehir okundu (satır {y})"
                            );
                            let end = if y + 1 == h {
                                a.dst.len()
                            } else {
                                row + a.stride
                            };
                            assert_eq!(
                                a.dst[row + n..end],
                                a0[row + n..end],
                                "{name} {bd:?} edges={e:04b} {w}x{h}: [0,w) dışına yazıldı"
                            );
                            assert_eq!(b.dst[row + n..end], b0[row + n..end]);
                        }
                        runs += 1;
                    }
                }
            }
        }
        assert_eq!(runs, 5 * 3 * 16 * sizes.len());
    }

    /// SGR ağırlığı 0 iken `dst` değişmez: `v = 0 → (0 + 2^10) >> 11 = 0`
    /// (`looprestoration_tmpl.c:601-602,613-614`).
    #[test]
    fn sgr_sifir_agirlik_kimlik() {
        let mut rng = Rng(0x5151_7272_9393_b4b5);
        for bd in LrBitDepth::ALL {
            let d = LrDsp::scalar_for(bd);
            for e in 0..16u8 {
                let edges = LrEdgeFlags::from_bits(e);
                for (k, kernel) in d.sgr.iter().enumerate() {
                    let reach = if k == 1 { REACH_2 } else { REACH_3 };
                    let (w, h) = (1 + rng.below(70) as usize, 1 + rng.below(12) as usize);
                    let mut x = build(rng.next() | 1, rng.next() | 1, bd, edges, w, h, reach);
                    let before = x.dst.clone();
                    let idx = rng.below(16) as usize;
                    let params = LrParams::Sgr {
                        s0: u32::from(SGR_PARAMS[idx][0]),
                        s1: u32::from(SGR_PARAMS[idx][1]),
                        w0: 0,
                        w1: 0,
                    };
                    kernel(
                        &mut x.dst,
                        x.stride,
                        &x.left,
                        &x.lpf,
                        x.lpf_stride,
                        x.lpf_x,
                        w,
                        h,
                        &params,
                        edges,
                        bd,
                    );
                    assert_eq!(x.dst, before, "sgr[{k}] {bd:?} edges={e:04b} {w}x{h}");
                }
            }
        }
    }

    /// Düz girdi düz kalır: katsayı toplamları yatayda 0 (+8-bit `src*128`),
    /// dikeyde 128 olduğu için Wiener çıktısı her bit derinliğinde girdiye
    /// eşittir (yuvarlama kalıntısı sıfır; `looprestoration_tmpl.c:50-52,168-170`).
    #[test]
    fn wiener_duz_girdi_degismez() {
        let mut rng = Rng(0xabcd_ef01_2345_6789);
        for bd in LrBitDepth::ALL {
            let bpp = bd.bytes();
            for e in 0..16u8 {
                let edges = LrEdgeFlags::from_bits(e);
                for five in [false, true] {
                    let kernel = LrDsp::scalar_for(bd).wiener[usize::from(five)];
                    let (w, h) = (1 + rng.below(80) as usize, 1 + rng.below(10) as usize);
                    let v = rng.below(u64::from(bd.bitdepth_max()) + 1) as u16;
                    let stride = (w + 5) * bpp;
                    let mut dst = vec![0u8; h * stride];
                    let mut left = vec![0u8; h * 4 * bpp];
                    let lpf_x = 4;
                    let lpf_stride = (lpf_x + w + 4) * bpp;
                    let mut lpf = vec![0u8; 8 * lpf_stride];
                    for buf in [&mut dst, &mut left, &mut lpf] {
                        for o in (0..buf.len()).step_by(bpp) {
                            put_px(buf, o, bpp, v);
                        }
                    }
                    let params = wiener_params(&mut rng, five, bd);
                    kernel(
                        &mut dst, stride, &left, &lpf, lpf_stride, lpf_x, w, h, &params, edges, bd,
                    );
                    for o in (0..dst.len()).step_by(bpp) {
                        let got = if bpp == 1 {
                            u16::from(dst[o])
                        } else {
                            u16::from_le_bytes([dst[o], dst[o + 1]])
                        };
                        assert_eq!(got, v, "{bd:?} edges={e:04b} five={five} {w}x{h}");
                    }
                }
            }
        }
    }

    /// Sözleşme dışı girdi: serbest derlemede `dst` değişmez, panik yok
    /// (hata ayıklamada `debug_assert` düşer, bu yüzden yalnız serbestte koşar).
    #[cfg(not(debug_assertions))]
    #[test]
    fn sozlesme_disi_girdi_dst_degismez() {
        let d = LrDsp::scalar::<8>();
        let wiener = LrParams::Wiener {
            filter: [[0, 0, 0, 128, 0, 0, 0, 0], [0, 0, 0, 128, 0, 0, 0, 0]],
        };
        let sgr = LrParams::Sgr {
            s0: 140,
            s1: 3236,
            w0: 10,
            w1: 50,
        };
        let all = LrEdgeFlags::ALL;
        let mut dst = vec![7u8; 64 * 400];
        let before = dst.clone();
        let left = vec![0u8; 64 * 4];
        let lpf = vec![0u8; 8 * 400];
        // Yanlış parametre türü.
        (d.wiener[0])(
            &mut dst,
            400,
            &left,
            &lpf,
            400,
            3,
            16,
            4,
            &sgr,
            all,
            LrBitDepth::B8,
        );
        (d.sgr[0])(
            &mut dst,
            400,
            &left,
            &lpf,
            400,
            3,
            16,
            4,
            &wiener,
            all,
            LrBitDepth::B8,
        );
        // w = 0, w > 384, h = 0.
        (d.sgr[2])(
            &mut dst,
            400,
            &left,
            &lpf,
            400,
            3,
            0,
            4,
            &sgr,
            all,
            LrBitDepth::B8,
        );
        (d.sgr[2])(
            &mut dst,
            400,
            &left,
            &lpf,
            400,
            3,
            super::LR_MAX_W + 1,
            4,
            &sgr,
            all,
            LrBitDepth::B8,
        );
        (d.sgr[1])(
            &mut dst,
            400,
            &left,
            &lpf,
            400,
            3,
            16,
            0,
            &sgr,
            all,
            LrBitDepth::B8,
        );
        // Kısa dst / left / lpf, lpf_x < reach.
        (d.wiener[1])(
            &mut dst[..100],
            400,
            &left,
            &lpf,
            400,
            3,
            16,
            4,
            &wiener,
            all,
            LrBitDepth::B8,
        );
        (d.sgr[1])(
            &mut dst,
            400,
            &left[..10],
            &lpf,
            400,
            3,
            16,
            4,
            &sgr,
            all,
            LrBitDepth::B8,
        );
        (d.sgr[0])(
            &mut dst,
            400,
            &left,
            &lpf[..2000],
            400,
            3,
            16,
            4,
            &sgr,
            all,
            LrBitDepth::B8,
        );
        (d.sgr[0])(
            &mut dst,
            400,
            &left,
            &lpf,
            400,
            2,
            16,
            4,
            &sgr,
            all,
            LrBitDepth::B8,
        );
        // 16bpc'de iki kat bayt gerekir.
        (d.sgr[2])(
            &mut dst,
            400,
            &left,
            &lpf,
            400,
            3,
            384,
            64,
            &sgr,
            all,
            LrBitDepth::B10,
        );
        assert_eq!(dst, before);
    }
}
