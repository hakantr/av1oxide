//! Film grain çekirdek sözleşmesi (dav1d `Dav1dFilmGrainDSPContext`) — S2d.1 / P22.
//!
//! Bu modül **davranışı** tanımlar: sabitler, bit derinliği, kroma düzeni,
//! çekirdek imzaları ve bellek sözleşmesi. Gövdeler
//! [`crate::dsp::scalar::fg`] içindedir (8/10/12 bit, tek kaynak); Gauss
//! dizisi [`crate::dsp::fg_tables`]. SIMD yolu yoktur; yuvalar
//! ([`crate::dsp::BitDepthDsp::fg`], F3c) her bit derinliğinde o bit
//! derinliğinin skaler monomorflarını taşır.
//!
//! ## Kaynak eşlemesi (dav1d @ `99f3354`, salt-okunur)
//!
//! | Kural | Kaynak |
//! |---|---|
//! | `GRAIN_WIDTH` 82, `GRAIN_HEIGHT` 73, `FG_BLOCK_SIZE` 32 | `src/filmgrain.h:35-37` |
//! | `SUB_GRAIN_WIDTH` 44, `SUB_GRAIN_HEIGHT` 38 | `src/filmgrain_tmpl.c:35-36` |
//! | `SCALING_SIZE` (8 bit 256, 16bpc 4096) | `src/filmgrain.h:38-44` |
//! | `get_random_number` (16 bit LFSR), `round2` | `src/filmgrain_tmpl.c:38-48` |
//! | `generate_grain_y_c` | `src/filmgrain_tmpl.c:50-86` |
//! | `generate_grain_uv_c` (420/422/444) | `src/filmgrain_tmpl.c:88-154` |
//! | `sample_lut` | `src/filmgrain_tmpl.c:158-167` |
//! | `fgy_32x32xn_c` | `src/filmgrain_tmpl.c:169-276` |
//! | `fguv_32x32xn_c` (420/422/444) | `src/filmgrain_tmpl.c:278-413` |
//! | DSP kaydı | `src/filmgrain_tmpl.c:425-445` |
//! | `generate_scaling` | `src/fg_apply_tmpl.c:41-97` |
//!
//! ## Bellek sözleşmesi
//!
//! `bpp` = piksel başına bayt (8 bit → 1, 10/12 bit → 2, küçük sonlu `u16`).
//! Bütün adımlar (`stride`, `luma_stride`) **bayt** cinsindendir.
//!
//! | Tampon | Okunur / yazılır | Gereken uzunluk |
//! |---|---|---|
//! | `src` (`fgy`/`fguv`) | satır `0..bh`, sütun `0..pw` | `(bh−1)·stride + pw·bpp` |
//! | `dst` (`fgy`/`fguv`) | satır `0..bh`, sütun `0..pw` yazılır | `(bh−1)·stride + pw·bpp` |
//! | `luma` (`fguv`) | satır `y << sy`, sütun `(x << sx) + {0, sx}` | `((bh−1) << sy)·luma_stride + (((pw−1) << sx) + sx + 1)·bpp` |
//! | `scaling` | indeks `0..=bitdepth_max` | `bitdepth_max + 1` |
//!
//! `fguv` 4:2:0/4:2:2'de son kroma sütunu için luma sütunu `w`'yi (görünür
//! genişliğin bir ötesi) okuyabilir; çağıran dav1d gibi o sütunu `w − 1`
//! ile doldurur (`src/fg_apply_tmpl.c:195-202`). Sözleşme dışı girdi hata
//! ayıklamada `debug_assert` ile düşer, serbest derlemede `dst` değişmeden
//! kalır — panik yoktur.

#![forbid(unsafe_code)]

use crate::bitstream::frame_header::FilmGrainData;

/// Grain LUT genişliği (`GRAIN_WIDTH`).
pub const GRAIN_WIDTH: usize = 82;
/// Grain LUT yüksekliği (`GRAIN_HEIGHT`).
pub const GRAIN_HEIGHT: usize = 73;
/// Alt örneklenmiş kroma LUT genişliği (`SUB_GRAIN_WIDTH`).
pub const SUB_GRAIN_WIDTH: usize = 44;
/// Alt örneklenmiş kroma LUT yüksekliği (`SUB_GRAIN_HEIGHT`).
pub const SUB_GRAIN_HEIGHT: usize = 38;
/// Grain blok boyu (`FG_BLOCK_SIZE`).
pub const FG_BLOCK_SIZE: usize = 32;
/// En büyük ölçekleme tablosu boyu (16bpc `SCALING_SIZE`).
pub const SCALING_SIZE_MAX: usize = 4096;

/// Grain LUT'u: `[GRAIN_HEIGHT][GRAIN_WIDTH]` (dav1d `entry`; 8 bitte
/// `int8_t`, 16bpc'de `int16_t`). 8 bitte bütün değerler `[−128, 127]`
/// aralığında kaldığı için `i16` saklama dav1d ile aynı değeri verir.
pub type GrainLut = [[i16; GRAIN_WIDTH]; GRAIN_HEIGHT];

/// Boş (sıfır) grain LUT'u.
pub const EMPTY_LUT: GrainLut = [[0; GRAIN_WIDTH]; GRAIN_HEIGHT];

/// Desteklenen bit derinlikleri; `bitdepth_max = (1 << bd) − 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FgBitDepth {
    /// 8 bit (1 bayt/piksel).
    B8,
    /// 10 bit (2 bayt/piksel, küçük sonlu).
    B10,
    /// 12 bit (2 bayt/piksel, küçük sonlu).
    B12,
}

impl FgBitDepth {
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

    /// `bitdepth_min_8 = bd − 8` (0 / 2 / 4).
    #[must_use]
    pub const fn min_8(self) -> u32 {
        self.bits() - 8
    }

    /// `generate_scaling`'in doldurduğu giriş sayısı (8 bit `SCALING_SIZE`
    /// = 256; 16bpc `1 << bd`).
    #[must_use]
    pub const fn scaling_size(self) -> usize {
        1 << self.bits()
    }
}

/// Kroma düzeni (dav1d `layout − 1` dizini: 420 → 0, 422 → 1, 444 → 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FgLayout {
    /// 4:2:0 (`ss_x = ss_y = 1`).
    I420,
    /// 4:2:2 (`ss_x = 1`, `ss_y = 0`).
    I422,
    /// 4:4:4 (`ss_x = ss_y = 0`).
    I444,
}

impl FgLayout {
    /// Üç düzen (test taraması için tek kaynak).
    pub const ALL: [Self; 3] = [Self::I420, Self::I422, Self::I444];

    /// dav1d dizini (`layout − 1`).
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::I420 => 0,
            Self::I422 => 1,
            Self::I444 => 2,
        }
    }

    /// Yatay alt örnekleme (`ss_x`).
    #[must_use]
    pub const fn ss_x(self) -> usize {
        match self {
            Self::I420 | Self::I422 => 1,
            Self::I444 => 0,
        }
    }

    /// Dikey alt örnekleme (`ss_y`).
    #[must_use]
    pub const fn ss_y(self) -> usize {
        match self {
            Self::I420 => 1,
            Self::I422 | Self::I444 => 0,
        }
    }
}

/// dav1d `generate_grain_y_fn` (`src/filmgrain.h:46-49`).
pub type GenerateGrainYFn = fn(buf: &mut GrainLut, data: &FilmGrainData, bd: FgBitDepth);

/// dav1d `generate_grain_uv_fn` (`src/filmgrain.h:51-55`); `uv` 0 = U, 1 = V.
pub type GenerateGrainUvFn =
    fn(buf: &mut GrainLut, buf_y: &GrainLut, data: &FilmGrainData, uv: usize, bd: FgBitDepth);

/// dav1d `fgy_32x32xn_fn` (`src/filmgrain.h:57-63`).
pub type FgyFn = fn(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8],
    grain_lut: &GrainLut,
    bh: usize,
    row_num: usize,
    bd: FgBitDepth,
);

/// dav1d `fguv_32x32xn_fn` (`src/filmgrain.h:65-72`).
pub type FguvFn = fn(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8],
    grain_lut: &GrainLut,
    bh: usize,
    row_num: usize,
    luma: &[u8],
    luma_stride: usize,
    uv: usize,
    is_id: bool,
    bd: FgBitDepth,
);

/// Film grain çekirdek tablosu (dav1d `Dav1dFilmGrainDSPContext`).
#[derive(Clone, Copy)]
pub struct FgDsp {
    /// `generate_grain_y`.
    pub generate_grain_y: GenerateGrainYFn,
    /// `generate_grain_uv[layout − 1]`.
    pub generate_grain_uv: [GenerateGrainUvFn; 3],
    /// `fgy_32x32xn`.
    pub fgy_32x32xn: FgyFn,
    /// `fguv_32x32xn[layout − 1]`.
    pub fguv_32x32xn: [FguvFn; 3],
}

impl core::fmt::Debug for FgDsp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("FgDsp(skaler)")
    }
}

/// P39: SIMD film grain girişlerinin sözleşme dışı girdide skaler gövdeye
/// devrettiği çağrı sayısı (süreç başından beri). Çözme hattında 0 olmalıdır.
static FG_SKALER_DEVIR: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// [`FG_SKALER_DEVIR`] değeri.
#[must_use]
pub fn fg_skaler_devir_sayisi() -> u64 {
    FG_SKALER_DEVIR.load(core::sync::atomic::Ordering::Relaxed)
}

/// Skaler devri sayar (yalnız SIMD girişlerinin soğuk yolu).
#[cold]
pub fn fg_skaler_devri_say() {
    FG_SKALER_DEVIR.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
}

/// `(bh − 1)·stride + son` (taşmada `None`).
fn bolge_sonu(bh: usize, stride: usize, son: usize) -> Option<usize> {
    (bh - 1).checked_mul(stride)?.checked_add(son)
}

/// P39: 8 bpc SIMD `fgy_32x32xn` girişinin O(1) sözleşmesi. `pw ≥ 1`, `bh`
/// `1..=32`, `stride ≥ pw`, `scaling_shift` `8..=11`; `src` ve `dst`
/// `(bh − 1)·stride + pw` bayt. Ölçekleme tablosunun boyu girişte ayrıca
/// denetlenir (sabit boyutlu dizi). Grain LUT'u sabit boyutlu türdür.
#[must_use]
pub fn fgy_sozlesmesi(
    dst_len: usize,
    src_len: usize,
    stride: usize,
    pw: usize,
    bh: usize,
    data: &FilmGrainData,
) -> bool {
    pw > 0
        && (1..=FG_BLOCK_SIZE).contains(&bh)
        && stride >= pw
        && (8..=11).contains(&data.scaling_shift)
        && bolge_sonu(bh, stride, pw).is_some_and(|n| src_len >= n && dst_len >= n)
}

/// P39: 8 bpc SIMD `fguv_32x32xn` girişinin O(1) sözleşmesi ([`fgy_sozlesmesi`]
/// koşulları, `bh ≤ 32 >> sy`, `uv < 2`, `luma_stride ≥ ((pw − 1) << sx) + sx + 1`;
/// luma `((bh − 1) << sy)·luma_stride + ((pw − 1) << sx) + sx + 1` bayt).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fguv_sozlesmesi(
    dst_len: usize,
    src_len: usize,
    stride: usize,
    pw: usize,
    bh: usize,
    luma_len: usize,
    luma_stride: usize,
    uv: usize,
    sx: usize,
    sy: usize,
    data: &FilmGrainData,
) -> bool {
    let luma_son = pw
        .checked_sub(1)
        .and_then(|v| v.checked_shl(sx as u32))
        .map(|v| v + sx + 1);
    fgy_sozlesmesi(dst_len, src_len, stride, pw, bh, data)
        && bh <= FG_BLOCK_SIZE >> sy
        && uv < 2
        && luma_son.is_some_and(|son| {
            luma_stride >= son
                && ((bh - 1) << sy)
                    .checked_mul(luma_stride)
                    .and_then(|v| v.checked_add(son))
                    .is_some_and(|n| luma_len >= n)
        })
}

/// P42: 16 bpc (`10`/`12` bit, `bpp = 2`) SIMD `fgy_32x32xn` girişinin O(1)
/// sözleşmesi. `bd` `B10` veya `B12`, `pw ≥ 1`, `bh` `1..=32`, `stride` çift
/// ve `stride ≥ pw·2`, `scaling_shift` `8..=11`; `src` ve `dst`
/// `(bh − 1)·stride + pw·2` bayt. Ölçekleme tablosunun boyu (`1024` / `4096`)
/// girişte ayrıca sabit boyutlu dilimle denetlenir.
#[must_use]
pub fn fgy16_sozlesmesi(
    dst_len: usize,
    src_len: usize,
    stride: usize,
    pw: usize,
    bh: usize,
    data: &FilmGrainData,
    bd: FgBitDepth,
) -> bool {
    let son_bayt = pw.checked_mul(2);
    matches!(bd, FgBitDepth::B10 | FgBitDepth::B12)
        && pw > 0
        && (1..=FG_BLOCK_SIZE).contains(&bh)
        && stride.is_multiple_of(2)
        && (8..=11).contains(&data.scaling_shift)
        && son_bayt.is_some_and(|son| {
            stride >= son
                && bolge_sonu(bh, stride, son).is_some_and(|n| src_len >= n && dst_len >= n)
        })
}

/// P42: 16 bpc (`10`/`12` bit, `bpp = 2`) SIMD `fguv_32x32xn` girişinin O(1)
/// sözleşmesi ([`fgy16_sozlesmesi`] koşulları, `bh ≤ 32 >> sy`, `uv < 2`,
/// `luma_stride` çift ve `luma_stride ≥ (((pw − 1) << sx) + sx + 1)·2`; luma
/// `((bh − 1) << sy)·luma_stride + (((pw − 1) << sx) + sx + 1)·2` bayt).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn fguv16_sozlesmesi(
    dst_len: usize,
    src_len: usize,
    stride: usize,
    pw: usize,
    bh: usize,
    luma_len: usize,
    luma_stride: usize,
    uv: usize,
    sx: usize,
    sy: usize,
    data: &FilmGrainData,
    bd: FgBitDepth,
) -> bool {
    let luma_son = pw
        .checked_sub(1)
        .and_then(|v| v.checked_shl(sx as u32))
        .and_then(|v| v.checked_add(sx + 1))
        .and_then(|v| v.checked_mul(2));
    fgy16_sozlesmesi(dst_len, src_len, stride, pw, bh, data, bd)
        && bh <= FG_BLOCK_SIZE >> sy
        && uv < 2
        && luma_stride.is_multiple_of(2)
        && luma_son.is_some_and(|son| {
            luma_stride >= son
                && ((bh - 1) << sy)
                    .checked_mul(luma_stride)
                    .and_then(|v| v.checked_add(son))
                    .is_some_and(|n| luma_len >= n)
        })
}

impl FgDsp {
    /// `BPC` bit derinliğinin skaler tablosu (`dav1d_film_grain_dsp_init`, asm
    /// yok); F3c: her giriş o bit derinliğinin monomorfudur.
    #[must_use]
    pub const fn scalar<const BPC: u32>() -> Self {
        use crate::dsp::scalar::fg;
        Self {
            generate_grain_y: fg::generate_grain_y::<BPC>,
            generate_grain_uv: [
                fg::generate_grain_uv_420::<BPC>,
                fg::generate_grain_uv_422::<BPC>,
                fg::generate_grain_uv_444::<BPC>,
            ],
            fgy_32x32xn: fg::fgy_32x32xn::<BPC>,
            fguv_32x32xn: [
                fg::fguv_32x32xn_420::<BPC>,
                fg::fguv_32x32xn_422::<BPC>,
                fg::fguv_32x32xn_444::<BPC>,
            ],
        }
    }

    /// Çalışma zamanı bit derinliğinin skaler tablosu (kurulum ve testler).
    #[must_use]
    pub const fn scalar_for(bd: FgBitDepth) -> Self {
        match bd {
            FgBitDepth::B8 => Self::scalar::<8>(),
            FgBitDepth::B10 => Self::scalar::<10>(),
            FgBitDepth::B12 => Self::scalar::<12>(),
        }
    }
}
