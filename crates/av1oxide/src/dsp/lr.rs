//! Loop restoration (Wiener / SGRPROJ) çekirdek sözleşmesi — S2c.2b.
//!
//! Bu modül **davranışı** tanımlar, gövde içermez: kenar bayrakları, bit
//! derinliği, parametre tipi, çekirdek imzası ve dispatch tablosu ([`LrDsp`]).
//! Gövdeler [`crate::dsp::scalar::lr`] içindedir (8/10/12 bit, tek kaynak);
//! tablolar [`crate::dsp::lr_tables`] içindedir. SIMD yolu yoktur.
//!
//! ## Kaynak eşlemesi (dav1d @ `99f3354`, salt-okunur)
//!
//! Aşağıdaki tablo, taşınan **her** kuralın dav1d kaynağındaki satırını verir.
//! Çelişki hâlinde kaynak esastır; bu tablo denetim izidir.
//!
//! | Kural | Kaynak |
//! |---|---|
//! | Kenar bayrakları `LR_HAVE_LEFT=1<<0`, `RIGHT=1<<1`, `TOP=1<<2`, `BOTTOM=1<<3` | `src/looprestoration.h:36-41` |
//! | `LooprestorationParams`: `filter[2][8]` ya da `sgr {s0, s1, w0, w1}` | `src/looprestoration.h:49-55` |
//! | Çekirdek imzası (`dst, stride, left[][4], lpf, w, h, params, edges`) | `src/looprestoration.h:64-69` |
//! | `wiener[2]` (7 / 5 dokunuş), `sgr[3]` (5×5 / 3×3 / karma) | `src/looprestoration.h:72-75` |
//! | `wiener[0] = wiener[1] = wiener_c`; `sgr[0..3] = 5x5, 3x3, mix` | `src/looprestoration_tmpl.c:1362-1365` |
//! | `REST_UNIT_STRIDE = 390` (`256*1.5 + 3 + 3`) | `src/looprestoration_tmpl.c:41-42` |
//! | Wiener yatay: `w < 6` tam koşullu döngü; `w >= 6` baş/gövde/son döngüleri | `src/looprestoration_tmpl.c:44-161` |
//! | Wiener yatay yuvarlama `round_bits_h = 3 + 2*(bd==12)`, kırpma `[0, 1<<(bd+8-round_bits_h))` | `src/looprestoration_tmpl.c:50-52,77` |
//! | 8 bit'te yatay toplama `src[x] * 128` eklenir (16bpc'de `filter[0][3] += 128`) | `src/looprestoration_tmpl.c:59-61,90-92,107-109,133-135,148-150`, `src/lr_apply_tmpl.c:61-65` |
//! | Wiener dikey `round_bits_v = 11 - 2*(bd==12)`, `round_offset = 1 << (bd + round_bits_v - 1)` | `src/looprestoration_tmpl.c:163-191` |
//! | Wiener `_v` işaretçi kaydırması (yalnız ilk 5) / `_hv` (6 + `ptrs[6] = ptrs[0]`) | `src/looprestoration_tmpl.c:186-190,193-246` |
//! | Wiener satır akışı: TOP/TOP'suz başlangıç, gövde, BOTTOM, `v1/v2/v3` kuyrukları | `src/looprestoration_tmpl.c:250-386` |
//! | SGR `rotate`, `rotate5_x2` | `src/looprestoration_tmpl.c:389-411` |
//! | Kutu toplamları `sgr_box3_row_h` / `sgr_box5_row_h` / `sgr_box35_row_h` (sol: `left` ya da `src[-k]`) | `src/looprestoration_tmpl.c:413-461` |
//! | Dikey kutu toplamları `sgr_box3_row_v` / `sgr_box5_row_v` (`w + 2` sütun) | `src/looprestoration_tmpl.c:463-497` |
//! | `sgr_calc_row_ab`: `p = max(a*n - b*b, 0)`, `z = (p*s + 2^19) >> 20`, `x = x_by_x[min(z,255)]` | `src/looprestoration_tmpl.c:499-517` |
//! | `n = 9, one_by_x = 455` (3×3); `n = 25, one_by_x = 164` (5×5); `sgr_box3_hv` | `src/looprestoration_tmpl.c:519-547` |
//! | `EIGHT_NEIGHBORS` (4/3 ağırlık, `>> 9`) | `src/looprestoration_tmpl.c:549-564` |
//! | `SIX_NEIGHBORS` (6/5 ağırlık, `>> 9`; ikinci satır `>> 8`) | `src/looprestoration_tmpl.c:568-595` |
//! | `FILTER_OUT_STRIDE = 384` | `src/looprestoration_tmpl.c:566` |
//! | Ağırlıklı karışım `dst + ((w*t + 2^10) >> 11)` / `w0*t1 + w1*t2` | `src/looprestoration_tmpl.c:597-620` |
//! | `sgr_finish1/2/mix` (satır çıkışı + `rotate`) | `src/looprestoration_tmpl.c:622-670` |
//! | `sgr_3x3_c` / `sgr_5x5_c` / `sgr_mix_c` satır akışı ve kuyrukları | `src/looprestoration_tmpl.c:673-817`, `819-1019`, `1021-1345` |
//! | `BUF_STRIDE = 384 + 16` | `src/looprestoration_tmpl.c:679` |
//! | `dav1d_sgr_params[16][2]`, `dav1d_sgr_x_by_x[256]` | `src/tables.c:415-442` |
//! | Çağıran: `filter` kurulumu, `wiener[!(f[0][0] \| f[1][0])]`, `sgr[!!s0 + 2*!!s1 - 1]` | `src/lr_apply_tmpl.c:54-83` |
//!
//! ## Bellek sözleşmesi
//!
//! `bpp` = piksel başına bayt (8 bit → 1, 10/12 bit → 2, küçük sonlu `u16`).
//! `reach` = çekirdeğin kenar ötesine uzanımı: Wiener / 5×5 / karma **3**,
//! 3×3 **2** piksel (dav1d `src[-3..w+3)` / `src[-2..w+2)`).
//!
//! | Tampon | Okunur | Gereken uzunluk |
//! |---|---|---|
//! | `dst` | satır `0..h`, sütun `0..w` (+`reach` sağda, yalnız `HAVE_RIGHT`) | `(h−1)·stride + (w + reach·R)·bpp` |
//! | `left` | yalnız `HAVE_LEFT`; satır `r` piksel `1..4` (`4−reach..4`) | `h·4·bpp` |
//! | `lpf` satır 0,1 | yalnız `HAVE_TOP` | `1·lpf_stride + (lpf_x + w + reach·R)·bpp` |
//! | `lpf` satır 6,7 | yalnız `HAVE_BOTTOM` | `7·lpf_stride + (lpf_x + w + reach·R)·bpp` |
//!
//! `R = 1` yalnız `HAVE_RIGHT` iken. `HAVE_LEFT` iken `lpf` satırlarında
//! `lpf_x − reach .. lpf_x` sütunları da okunur ("left edge existing in the
//! same buffer", `looprestoration_tmpl.c:112-117`); bu yüzden `lpf_x >= reach`
//! gerekir. `dst` satırlarında **sol** komşu hiçbir zaman okunmaz (her zaman
//! `left` tamponu ya da `src[0]` dolgusu kullanılır).
//!
//! Yazma: yalnız `dst` satır `0..h`, sütun `0..w` (dav1d C ile aynı). dav1d
//! başlık yorumundaki "hizalı taşan yazma" serbestliği **C yolunda kullanılmaz**
//! (`looprestoration.h:61-63`); bu port da kullanmaz.
//!
//! Piksel değerleri `[0, bitdepth_max]` aralığındadır (çözücünün yazdığı her
//! piksel kırpılmıştır). Aralık dışı değerde dav1d C'de `int` taşması
//! tanımsızdır; bu port taşma denetimli profillerde (`dev`, `robust`) bunu
//! taşma panikiyle gösterir, serbest derlemede sarar.
//!
//! Sözleşme dışı girdi (yetersiz dilim, `w ∉ 1..=384`, `h == 0`, yanlış
//! [`LrParams`] türü): hata ayıklamada `debug_assert` düşer, serbest derlemede
//! `dst` **değişmeden** bırakılır — panik yoktur.

#![forbid(unsafe_code)]

use crate::dsp::scalar::lr as scalar;

/// dav1d `enum LrEdgeFlags` (`src/looprestoration.h:36-41`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LrEdgeFlags(u8);

impl LrEdgeFlags {
    /// Kenar yok.
    pub const NONE: Self = Self(0);
    /// Sol komşu mevcut (`LR_HAVE_LEFT = 1 << 0`).
    pub const HAVE_LEFT: Self = Self(1 << 0);
    /// Sağ komşu mevcut (`LR_HAVE_RIGHT = 1 << 1`).
    pub const HAVE_RIGHT: Self = Self(1 << 1);
    /// Üst `lpf` satırları mevcut (`LR_HAVE_TOP = 1 << 2`).
    pub const HAVE_TOP: Self = Self(1 << 2);
    /// Alt `lpf` satırları mevcut (`LR_HAVE_BOTTOM = 1 << 3`).
    pub const HAVE_BOTTOM: Self = Self(1 << 3);
    /// Dört bayrağın birleşimi (`0b1111`).
    pub const ALL: Self = Self(0b1111);

    /// Ham bit deseni.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Bitlerden küme kurar; tanımsız (4..7) bitler **atılır** (dav1d
    /// çekirdekleri yalnız dört biti sınar).
    #[must_use]
    pub const fn from_bits(b: u8) -> Self {
        Self(b & Self::ALL.0)
    }

    /// Birleşim.
    #[must_use]
    pub const fn union(self, o: Self) -> Self {
        Self(self.0 | o.0)
    }

    /// `o`'nun **tüm** bitleri bu kümede var mı?
    #[must_use]
    pub const fn contains(self, o: Self) -> bool {
        self.0 & o.0 == o.0
    }
}

/// Desteklenen bit derinlikleri; `bitdepth_max = (1 << bd) − 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LrBitDepth {
    /// 8 bit (1 bayt/piksel).
    B8,
    /// 10 bit (2 bayt/piksel, küçük sonlu).
    B10,
    /// 12 bit (2 bayt/piksel, küçük sonlu).
    B12,
}

impl LrBitDepth {
    /// Üç bit derinliği (test taraması için tek kaynak).
    pub const ALL: [Self; 3] = [Self::B8, Self::B10, Self::B12];

    /// Bit cinsinden derinlik (8 / 10 / 12).
    #[must_use]
    pub const fn bits(self) -> u32 {
        match self {
            Self::B8 => 8,
            Self::B10 => 10,
            Self::B12 => 12,
        }
    }

    /// Piksel başına bayt (8 bit → 1, 16bpc → 2).
    #[must_use]
    pub const fn bytes(self) -> usize {
        match self {
            Self::B8 => 1,
            Self::B10 | Self::B12 => 2,
        }
    }

    /// `bitdepth_max = (1 << bd) − 1` (255 / 1023 / 4095).
    #[must_use]
    pub const fn bitdepth_max(self) -> u16 {
        (1u16 << self.bits()) - 1
    }
}

/// dav1d `LooprestorationParams` (`src/looprestoration.h:49-55`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LrParams {
    /// `filter[0]` yatay, `filter[1]` dikey; değerler `lr_stripe`te kurulmuş hâliyle
    /// (merkez katsayı ve 8-bit dışı `+128` dahil — çağıran kurar).
    Wiener {
        /// dav1d `params->filter[2][8]` (`[x][7]` kullanılmaz).
        filter: [[i16; 8]; 2],
    },
    /// `s0`,`s1` = `dav1d_sgr_params[idx]`, `w0`, `w1` (`w1 = 128 - (w0 + w1_okunan)` çağıran kurar).
    Sgr {
        /// 5×5 (r=2) gücü (`dav1d_sgr_params[idx][0]`; 0 → 5×5 geçişi yok).
        s0: u32,
        /// 3×3 (r=1) gücü (`dav1d_sgr_params[idx][1]`; 0 → 3×3 geçişi yok).
        s1: u32,
        /// 5×5 ağırlığı (`sgr_weights[0]`).
        w0: i32,
        /// 3×3 ağırlığı (`128 − (sgr_weights[0] + sgr_weights[1])`).
        w1: i32,
    },
}

/// Piksel tamponları bayt dizisidir; 8 bit → 1 bayt/piksel, 10/12 bit → 2 bayt LE (u16).
///
/// Bellek sözleşmesi modül başlığındaki tablodadır. Sonuç `dst` içine
/// **yerinde** yazılır (dav1d `decl_lr_filter_fn`, `src/looprestoration.h:64-69`).
pub type LrFilterFn = fn(
    dst: &mut [u8], // dst[0] = şeridin sol üst pikseli (dav1d `p`); satır r = r*stride bayt.
    stride: usize,  // dst satır adımı (bayt)
    left: &[u8],    // dav1d `left[h][4]`: h satır × 4 piksel, paketli (satır r = r*4*bpp bayt).
    lpf: &[u8], // dav1d `lpf`: satır 0,1 = üst iki satır (HAVE_TOP); satır 6,7 = alt iki satır (HAVE_BOTTOM).
    lpf_stride: usize, // lpf satır adımı (bayt)
    lpf_x: usize, // lpf satırlarında dst sütun 0'a karşılık gelen piksel sütunu (HAVE_LEFT → lpf_x >= 3).
    w: usize, // birim genişliği (1..=384; dav1d REST_UNIT_STRIDE = 390, FILTER_OUT_STRIDE = 384)
    h: usize, // şerit yüksekliği (>= 1; dav1d'de <= 64)
    params: &LrParams,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
);

/// Loop restoration çekirdek ailesi — dav1d `Dav1dLoopRestorationDSPContext`
/// (`src/looprestoration.h:72-75`).
///
/// Seçim **çağırandadır** (dav1d `src/lr_apply_tmpl.c:72,83`):
/// `wiener[!(filter[0][0] | filter[1][0]) as usize]`,
/// `sgr[(s0 != 0) as usize + 2 * (s1 != 0) as usize - 1]`. Çekirdekler bu
/// indekslemeyi bilmez.
#[derive(Clone, Copy)]
pub struct LrDsp {
    /// `[0]` = 7 dokunuş, `[1]` = 5 dokunuş (dav1d C'de ikisi de `wiener_c`).
    pub wiener: [LrFilterFn; 2],
    /// `[0]` = 5×5, `[1]` = 3×3, `[2]` = karma.
    pub sgr: [LrFilterFn; 3],
}

impl LrDsp {
    /// `BPC` bit derinliğinin skaler tablosu (F3c: her giriş o bit derinliğinin
    /// monomorfudur).
    #[must_use]
    pub const fn scalar<const BPC: u32>() -> Self {
        Self {
            wiener: [scalar::wiener_7tap::<BPC>, scalar::wiener_5tap::<BPC>],
            sgr: [
                scalar::sgr_5x5::<BPC>,
                scalar::sgr_3x3::<BPC>,
                scalar::sgr_mix::<BPC>,
            ],
        }
    }

    /// Çalışma zamanı bit derinliğinin skaler tablosu (kurulum ve testler).
    #[must_use]
    pub const fn scalar_for(bd: LrBitDepth) -> Self {
        match bd {
            LrBitDepth::B8 => Self::scalar::<8>(),
            LrBitDepth::B10 => Self::scalar::<10>(),
            LrBitDepth::B12 => Self::scalar::<12>(),
        }
    }
}

impl std::fmt::Debug for LrDsp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LrDsp { wiener: [fn; 2], sgr: [fn; 3] }")
    }
}

/// P46: SIMD `looprestoration` (`wiener`, `sgr`) girişlerinin sözleşme dışı
/// girdide skaler gövdeye devrettiği çağrı sayısı (süreç başından beri).
/// Çözme hattında 0 olmalıdır.
static LR_SKALER_DEVIR: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// [`LR_SKALER_DEVIR`] değeri.
#[must_use]
pub fn lr_skaler_devir_sayisi() -> u64 {
    LR_SKALER_DEVIR.load(core::sync::atomic::Ordering::Relaxed)
}

/// Skaler devri sayar (yalnız SIMD girişlerinin soğuk yolu).
#[cold]
pub fn lr_skaler_devri_say() {
    LR_SKALER_DEVIR.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
}

/// P46: Ortak LR bellek sözleşmesi (`reach ∈ {2, 3}`):
/// - `bd.bits() == BPC`
/// - `w ∈ 1..=384` (`LR_MAX_W`), `h >= 1`
/// - `BPC > 8` ise `stride % 2 == 0` ve (`HAVE_TOP` veya `HAVE_BOTTOM` varsa) `lpf_stride % 2 == 0`
/// - `right = if HAVE_RIGHT { reach } else { 0 }`, `row_bytes = (w + right) * bpp`
/// - `stride >= w * bpp` ve `dst_len >= (h - 1) * stride + row_bytes`
/// - `HAVE_LEFT` ise `left_len >= h * 4 * bpp`
/// - `HAVE_TOP` veya `HAVE_BOTTOM` ise (`last_row = if HAVE_BOTTOM { 7 } else { 1 }`):
///   - `HAVE_LEFT` ise `lpf_x >= reach`
///   - `lpf_len >= last_row * lpf_stride + (lpf_x + w + right) * bpp`
#[must_use]
#[allow(clippy::too_many_arguments)]
fn lr_bellek_sozlesmesi<const BPC: u32>(
    reach: usize,
    dst_len: usize,
    stride: usize,
    left_len: usize,
    lpf_len: usize,
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
) -> bool {
    let bpp = if BPC == 8 { 1 } else { 2 };
    if bd.bits() != BPC || w == 0 || w > scalar::LR_MAX_W || h == 0 {
        return false;
    }
    if BPC > 8 && !stride.is_multiple_of(2) {
        return false;
    }
    let right = if edges.contains(LrEdgeFlags::HAVE_RIGHT) {
        reach
    } else {
        0
    };
    let row_bytes = (w + right) * bpp;
    if stride < w * bpp {
        return false;
    }
    let Some(dst_need) = (h - 1)
        .checked_mul(stride)
        .and_then(|v| v.checked_add(row_bytes))
    else {
        return false;
    };
    if dst_len < dst_need {
        return false;
    }
    let have_left = edges.contains(LrEdgeFlags::HAVE_LEFT);
    if have_left {
        let Some(left_need) = h.checked_mul(4 * bpp) else {
            return false;
        };
        if left_len < left_need {
            return false;
        }
    }
    let last_lpf_row = if edges.contains(LrEdgeFlags::HAVE_BOTTOM) {
        Some(7usize)
    } else if edges.contains(LrEdgeFlags::HAVE_TOP) {
        Some(1usize)
    } else {
        None
    };
    if let Some(r) = last_lpf_row {
        if BPC > 8 && !lpf_stride.is_multiple_of(2) {
            return false;
        }
        if have_left && lpf_x < reach {
            return false;
        }
        let Some(lpf_need) = r.checked_mul(lpf_stride).and_then(|v| {
            lpf_x
                .checked_add(w + right)
                .and_then(|p| p.checked_mul(bpp))
                .and_then(|b| v.checked_add(b))
        }) else {
            return false;
        };
        if lpf_len < lpf_need {
            return false;
        }
    }
    true
}

/// P46: `wiener[tap_idx]` (`0 = 7-tap`, `1 = 5-tap`) SIMD girişinin O(1) sözleşmesi.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn wiener_sozlesmesi<const BPC: u32>(
    tap_idx: usize,
    dst_len: usize,
    stride: usize,
    left_len: usize,
    lpf_len: usize,
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    params: &LrParams,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
) -> bool {
    if tap_idx >= 2 || !matches!(params, LrParams::Wiener { .. }) {
        return false;
    }
    lr_bellek_sozlesmesi::<BPC>(
        3, dst_len, stride, left_len, lpf_len, lpf_stride, lpf_x, w, h, edges, bd,
    )
}

/// P46: `sgr[sgr_idx]` (`0 = 5x5`, `1 = 3x3`, `2 = mix`) SIMD girişinin O(1) sözleşmesi.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn sgr_sozlesmesi<const BPC: u32>(
    sgr_idx: usize,
    dst_len: usize,
    stride: usize,
    left_len: usize,
    lpf_len: usize,
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    params: &LrParams,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
) -> bool {
    if sgr_idx >= 3 || !matches!(params, LrParams::Sgr { .. }) {
        return false;
    }
    let reach = if sgr_idx == 1 { 2 } else { 3 };
    lr_bellek_sozlesmesi::<BPC>(
        reach, dst_len, stride, left_len, lpf_len, lpf_stride, lpf_x, w, h, edges, bd,
    )
}

/// P46: Sözleşme dışı `wiener` girdisinde skaler devir sayacını artırır;
/// skaler gövdenin önkoşulları sağlanıyorsa ilgili skaler gövdeyi çağırır,
/// değilse paniklemeden döner.
#[cold]
#[allow(clippy::too_many_arguments)]
pub fn wiener_skaler_devret<const BPC: u32>(
    tap_idx: usize,
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
    lr_skaler_devri_say();
    if wiener_sozlesmesi::<BPC>(
        tap_idx,
        dst.len(),
        stride,
        left.len(),
        lpf.len(),
        lpf_stride,
        lpf_x,
        w,
        h,
        params,
        edges,
        bd,
    ) {
        let f = LrDsp::scalar::<BPC>().wiener[tap_idx];
        f(
            dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
        );
    }
}

/// P46: Sözleşme dışı `sgr` girdisinde skaler devir sayacını artırır;
/// skaler gövdenin önkoşulları sağlanıyorsa ilgili skaler gövdeyi çağırır,
/// değilse paniklemeden döner.
#[cold]
#[allow(clippy::too_many_arguments)]
pub fn sgr_skaler_devret<const BPC: u32>(
    sgr_idx: usize,
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
    lr_skaler_devri_say();
    if sgr_sozlesmesi::<BPC>(
        sgr_idx,
        dst.len(),
        stride,
        left.len(),
        lpf.len(),
        lpf_stride,
        lpf_x,
        w,
        h,
        params,
        edges,
        bd,
    ) {
        let f = LrDsp::scalar::<BPC>().sgr[sgr_idx];
        f(
            dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{LrBitDepth, LrEdgeFlags};

    #[test]
    fn kenar_bayraklari_dav1d_bitleri() {
        assert_eq!(LrEdgeFlags::NONE.bits(), 0);
        assert_eq!(LrEdgeFlags::HAVE_LEFT.bits(), 1);
        assert_eq!(LrEdgeFlags::HAVE_RIGHT.bits(), 2);
        assert_eq!(LrEdgeFlags::HAVE_TOP.bits(), 4);
        assert_eq!(LrEdgeFlags::HAVE_BOTTOM.bits(), 8);
        let lt = LrEdgeFlags::HAVE_LEFT.union(LrEdgeFlags::HAVE_TOP);
        assert!(lt.contains(LrEdgeFlags::HAVE_LEFT));
        assert!(lt.contains(LrEdgeFlags::HAVE_TOP));
        assert!(!lt.contains(LrEdgeFlags::HAVE_RIGHT));
        assert!(lt.contains(LrEdgeFlags::NONE));
        assert_eq!(LrEdgeFlags::from_bits(0xff), LrEdgeFlags::ALL);
        assert_eq!(LrEdgeFlags::from_bits(5), lt);
    }

    /// `scalar_for(bd)` o bit derinliğinin monomorf tablosudur. P30: işlev
    /// adresi karşılaştırılmaz; iki tablonun her yuvası aynı girdide aynı
    /// çıktıyı verir ve 10 ile 12 bit gövdeleri ayırt edici girdide farklıdır.
    #[test]
    fn skaler_tablo_bit_derinligi_basina_ayri() {
        use super::{LrEdgeFlags, LrFilterFn, LrParams};
        let (w, h) = (16usize, 4usize);
        // wiener[0] 7 dokunuş; wiener[1] 5 dokunuş (dış katsayılar 0).
        let wiener = [
            LrParams::Wiener {
                filter: [
                    [3, -7, 15, 106 + 128, 15, -7, 3, 0],
                    [3, -7, 15, 106, 15, -7, 3, 0],
                ],
            },
            LrParams::Wiener {
                filter: [
                    [0, -7, 15, 112 + 128, 15, -7, 0, 0],
                    [0, -7, 15, 112, 15, -7, 0, 0],
                ],
            },
        ];
        let sgr = LrParams::Sgr {
            s0: 140,
            s1: 3236,
            w0: -32,
            w1: 129,
        };
        let kos = |f: LrFilterFn, bd: LrBitDepth, params: &LrParams, maks: u32| -> Vec<u8> {
            let bpp = bd.bytes();
            let stride = w * bpp;
            let mut dst = vec![0u8; stride * h];
            for (i, px) in dst.chunks_exact_mut(bpp).enumerate() {
                let v = ((i as u32).wrapping_mul(2_654_435_761) >> 9) & maks;
                if bpp == 1 {
                    px[0] = v as u8;
                } else {
                    px.copy_from_slice(&(v as u16).to_le_bytes());
                }
            }
            f(
                &mut dst,
                stride,
                &[],
                &[],
                0,
                0,
                w,
                h,
                params,
                LrEdgeFlags::NONE,
                bd,
            );
            dst
        };
        for bd in LrBitDepth::ALL {
            let d = super::LrDsp::scalar_for(bd);
            let s = match bd {
                LrBitDepth::B8 => super::LrDsp::scalar::<8>(),
                LrBitDepth::B10 => super::LrDsp::scalar::<10>(),
                LrBitDepth::B12 => super::LrDsp::scalar::<12>(),
            };
            let maks = bd.bitdepth_max() as u32;
            for ((a, b), p) in d.wiener.iter().zip(s.wiener.iter()).zip(&wiener) {
                assert_eq!(kos(*a, bd, p, maks), kos(*b, bd, p, maks));
            }
            for (a, b) in d.sgr.iter().zip(s.sgr.iter()) {
                assert_eq!(kos(*a, bd, &sgr, maks), kos(*b, bd, &sgr, maks));
            }
        }
        let (d10, d12) = (super::LrDsp::scalar::<10>(), super::LrDsp::scalar::<12>());
        assert_ne!(
            kos(d10.wiener[0], LrBitDepth::B10, &wiener[0], 1023),
            kos(d12.wiener[0], LrBitDepth::B12, &wiener[0], 1023),
            "ayırt edici girdi"
        );
    }

    #[test]
    fn bit_derinligi_sabitleri() {
        assert_eq!(LrBitDepth::B8.bitdepth_max(), 255);
        assert_eq!(LrBitDepth::B10.bitdepth_max(), 1023);
        assert_eq!(LrBitDepth::B12.bitdepth_max(), 4095);
        assert_eq!(LrBitDepth::B8.bytes(), 1);
        assert_eq!(LrBitDepth::B10.bytes(), 2);
        assert_eq!(LrBitDepth::B12.bytes(), 2);
    }

    #[test]
    fn p46_lr_sozlesmesi_ve_paniksiz_skaler_devir() {
        use super::{
            LrEdgeFlags, LrParams, lr_skaler_devir_sayisi, sgr_sozlesmesi, wiener_skaler_devret,
            wiener_sozlesmesi,
        };
        let wp = LrParams::Wiener {
            filter: [[0; 8]; 2],
        };
        let sp = LrParams::Sgr {
            s0: 140,
            s1: 3236,
            w0: -32,
            w1: 129,
        };
        assert!(wiener_sozlesmesi::<8>(
            0,
            64 * 64,
            64,
            0,
            0,
            0,
            0,
            64,
            64,
            &wp,
            LrEdgeFlags::NONE,
            LrBitDepth::B8,
        ));
        // Kısa dst / yanlış params / HAVE_RIGHT için +3 eksikliği reddedilir.
        assert!(!wiener_sozlesmesi::<8>(
            0,
            64 * 64,
            64,
            0,
            0,
            0,
            0,
            64,
            64,
            &wp,
            LrEdgeFlags::HAVE_RIGHT,
            LrBitDepth::B8,
        ));
        assert!(!wiener_sozlesmesi::<8>(
            0,
            64 * 64,
            64,
            0,
            0,
            0,
            0,
            64,
            64,
            &sp,
            LrEdgeFlags::NONE,
            LrBitDepth::B8,
        ));
        assert!(sgr_sozlesmesi::<8>(
            1,
            63 * 64 + 66,
            64,
            0,
            0,
            0,
            0,
            64,
            64,
            &sp,
            LrEdgeFlags::HAVE_RIGHT,
            LrBitDepth::B8,
        ));
        assert!(!sgr_sozlesmesi::<8>(
            0,
            63 * 64 + 66,
            64,
            0,
            0,
            0,
            0,
            64,
            64,
            &sp,
            LrEdgeFlags::HAVE_RIGHT,
            LrBitDepth::B8,
        ));

        let once = lr_skaler_devir_sayisi();
        let mut kisa_dst = [0u8; 8];
        wiener_skaler_devret::<8>(
            0,
            &mut kisa_dst,
            16,
            &[],
            &[],
            0,
            0,
            16,
            4,
            &wp,
            LrEdgeFlags::NONE,
            LrBitDepth::B8,
        );
        assert!(lr_skaler_devir_sayisi() > once);
    }
}
