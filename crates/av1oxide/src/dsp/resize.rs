//! Superres yatay büyütme çekirdeği sözleşmesi (dav1d `mc.resize`) — S2c.3 / P21.
//!
//! Bu modül **davranışı** tanımlar: bit derinliği, çekirdek imzası ve bellek
//! sözleşmesi. Gövde [`crate::dsp::scalar::resize`] içindedir (8/10/12 bit, tek
//! kaynak); filtre tablosu [`crate::dsp::resize_tables`]. SIMD yolu yoktur;
//! yuva ([`crate::dsp::BitDepthDsp::resize`], F3c) her bit derinliğinde o bit
//! derinliğinin skaler monomorfunu taşır.
//! MC ailesi ([`crate::dsp::mc::McDsp`]) bilinçli olarak **değişmez**: AVX2
//! tablolarına alan eklemek mevcut AVX2 koduna dokunmayı gerektirirdi.
//!
//! ## Kaynak eşlemesi (dav1d @ `99f3354`, salt-okunur)
//!
//! | Kural | Kaynak |
//! |---|---|
//! | İmza `(dst, dst_stride, src, src_stride, dst_w, h, src_w, dx, mx0)` | `src/mc.h:117-121` |
//! | Satır başı `mx = mx0`, `src_x = -1` | `src/mc_tmpl.c:924` |
//! | 8 dokunuş `F = dav1d_resize_filter[mx >> 8]`, kaynak `iclip(src_x + k − 3, 0, src_w − 1)` | `src/mc_tmpl.c:926-934` |
//! | Çıkış `iclip_pixel((−Σ F[k]·src + 64) >> 7)` | `src/mc_tmpl.c:927,935` |
//! | İlerleme `mx += dx; src_x += mx >> 14; mx &= 0x3fff` | `src/mc_tmpl.c:936-938` |
//! | DSP kaydı `c->resize = resize_c` | `src/mc_tmpl.c:991` |
//! | Tablo `dav1d_resize_filter[64][8]` | `src/tables.c:655-688` |
//!
//! ## Bellek sözleşmesi
//!
//! `bpp` = piksel başına bayt (8 bit → 1, 10/12 bit → 2, küçük sonlu `u16`).
//!
//! | Tampon | Okunur / yazılır | Gereken uzunluk |
//! |---|---|---|
//! | `src` | satır `0..h`, sütun `0..src_w` (kırpılmış; kenar ötesi okunmaz) | `(h−1)·src_stride + src_w·bpp` |
//! | `dst` | satır `0..h`, sütun `0..dst_w` yazılır | `(h−1)·dst_stride + dst_w·bpp` |
//!
//! `mx0` `0..=0x3fff` aralığındadır (dav1d `get_upscale_x0` `& 0x3fff`);
//! `dx > 0`. Sözleşme dışı girdi hata ayıklamada `debug_assert` ile düşer,
//! serbest derlemede `dst` değişmeden kalır — panik yoktur.

#![forbid(unsafe_code)]

/// Desteklenen bit derinlikleri; `bitdepth_max = (1 << bd) − 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeBitDepth {
    /// 8 bit (1 bayt/piksel).
    B8,
    /// 10 bit (2 bayt/piksel, küçük sonlu).
    B10,
    /// 12 bit (2 bayt/piksel, küçük sonlu).
    B12,
}

impl ResizeBitDepth {
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
    pub const fn bitdepth_max(self) -> i32 {
        (1 << self.bits()) - 1
    }
}

/// dav1d `resize_fn` (`src/mc.h:117-121`).
///
/// `dst_stride` ve `src_stride` bayt cinsindendir.
pub type ResizeFn = fn(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    dst_w: usize,
    h: usize,
    src_w: usize,
    dx: i32,
    mx0: i32,
    bd: ResizeBitDepth,
);

/// `BPC` bit derinliğinin skaler büyütme çekirdeği (`resize_c`; F3c: monomorf).
#[must_use]
pub const fn scalar<const BPC: u32>() -> ResizeFn {
    crate::dsp::scalar::resize::resize::<BPC>
}

/// Çalışma zamanı bit derinliğinin skaler büyütme çekirdeği (kurulum ve testler).
#[must_use]
pub const fn scalar_for(bd: ResizeBitDepth) -> ResizeFn {
    match bd {
        ResizeBitDepth::B8 => scalar::<8>(),
        ResizeBitDepth::B10 => scalar::<10>(),
        ResizeBitDepth::B12 => scalar::<12>(),
    }
}

/// SIMD `resize` girişinin O(1) sözleşme denetimi (8 bpc, P33).
///
/// Skaler gövdenin koşulları: `h`, `dst_w`, `src_w` ≥ 1, `mx0` 0…0x3fff, `dx`
/// 1…2¹⁶; kaynak dilimi `(h−1)·ss + src_w`, hedef `(h−1)·ds + dst_w` bayt;
/// satır adımları genişlikten küçük değil (satırlar örtüşmez).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn buyutme_sozlesmesi(
    dst_len: usize,
    dst_stride: usize,
    src_len: usize,
    src_stride: usize,
    dst_w: usize,
    h: usize,
    src_w: usize,
    dx: i32,
    mx0: i32,
) -> bool {
    h >= 1
        && dst_w >= 1
        && src_w >= 1
        && (0..=0x3fff).contains(&mx0)
        && (1..=1 << 16).contains(&dx)
        && src_stride >= src_w
        && dst_stride >= dst_w
        && src_len >= (h - 1) * src_stride + src_w
        && dst_len >= (h - 1) * dst_stride + dst_w
}

/// SIMD 16 bpc `resize` girişinin O(1) sözleşme denetimi (10/12 bit, P34):
/// [`buyutme_sozlesmesi`]'nin piksel başına 2 baytlık karşılığı — satır
/// adımları `≥ 2·genişlik` bayt, kaynak dilimi `(h−1)·ss + 2·src_w`, hedef
/// `(h−1)·ds + 2·dst_w` bayt; bit derinliği 10/12.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn buyutme16_sozlesmesi(
    dst_len: usize,
    dst_stride: usize,
    src_len: usize,
    src_stride: usize,
    dst_w: usize,
    h: usize,
    src_w: usize,
    dx: i32,
    mx0: i32,
    bd: ResizeBitDepth,
) -> bool {
    bd != ResizeBitDepth::B8
        && h >= 1
        && dst_w >= 1
        && src_w >= 1
        && (0..=0x3fff).contains(&mx0)
        && (1..=1 << 16).contains(&dx)
        && src_stride >= 2 * src_w
        && dst_stride >= 2 * dst_w
        && src_len >= (h - 1) * src_stride + 2 * src_w
        && dst_len >= (h - 1) * dst_stride + 2 * dst_w
}
