//! Ters dönüşüm (inverse transform, itx) sözleşmesi — S2b.1.
//!
//! Bu modül **davranışı ve bellek sözleşmesini** tanımlar; gövdeler
//! [`crate::dsp::scalar::itx`] (bit derinliği başına monomorfize skaler
//! çekirdek) ve [`crate::dsp::scalar::itx_1d`] (1 boyutlu çekirdekler)
//! içindedir. Kapsam dav1d `itxfm_add` karşılığı **skaler** çekirdeklerdir:
//! bütün kare/dikdörtgen tx boyutları, bütün tx türleri, WHT ve eob hızlı
//! yolları (SIMD, katsayı çözme ve dequant bu adımın dışındadır).
//!
//! ## Kaynak eşlemesi (dav1d @ `99f3354`, salt-okunur)
//!
//! Taşınan her kuralın dav1d kaynağındaki satırı. Çelişki hâlinde kaynak
//! esastır; bu tablo denetim izidir.
//!
//! | Kural | Kaynak |
//! |---|---|
//! | `inv_txfm_add_c` gövdesi (DC yolu, toplama/sıfırlama) | `src/itx_tmpl.c:43-118` |
//! | `has_dconly = (txtp == DCT_DCT)`; `eob < has_dconly` → DC yolu | `src/itx_tmpl.c:50,58-70` |
//! | `is_rect2 = (w·2 == h \|\| h·2 == w)`; DC'de ikinci `×181` çarpanı | `src/itx_tmpl.c:55,61-65` |
//! | `rnd = (1 << shift) >> 1` | `src/itx_tmpl.c:56` |
//! | DC: `dc=(dc·181+128)>>8` [rect2 ise iki kez], `dc=(dc+rnd)>>shift`, `dc=(dc·181+128+2048)>>12`, `dst=iclip_pixel(dst+dc)` | `src/itx_tmpl.c:59-68` |
//! | İlk/son 1B işlev seçimi: `tx1d_fns[lw][h]`, `tx1d_fns[lh][v]` | `src/itx_tmpl.c:72-74`; `src/itx_1d.c:1019-1041` |
//! | 8-bit: `row_clip_min = col_clip_min = INT16_MIN` (her ikisi de −32768) | `src/itx_tmpl.c:76-78` |
//! | 16bpc: `row_clip_min = (int)((unsigned)~bitdepth_max << 7)`, `col_clip_min = (int)((unsigned)~bitdepth_max << 5)` | `src/itx_tmpl.c:79-81` |
//! | `row_clip_max = ~row_clip_min`, `col_clip_max = ~col_clip_min` | `src/itx_tmpl.c:83-84` |
//! | `sh = min(h,32)`, `sw = min(w,32)` (64 noktalı üst yarı okunmaz) | `src/itx_tmpl.c:75,102;109-127` |
//! | eob hızlı yolları: `IDENTITY`/ikinci eksen, `IDENTITY`/birinci eksen, genel `lnzc[tx][eob]` | `src/itx_tmpl.c:87-95` |
//! | Toplama: `coeff[y + x·sh]` (satır = birinci/bitişik eksen); rect2'de `(c·181+128)>>8` | `src/itx_tmpl.c:96-104` |
//! | `last_nonzero_col+1 < sh` ise kalan satırlar sıfırlanır | `src/itx_tmpl.c:105-106` |
//! | Katsayı tamponunun **tamamı** (`sw·sh`) sıfırlanır | `src/itx_tmpl.c:108` |
//! | `tmp[i] = iclip((tmp[i]+rnd)>>shift, col_clip_min, col_clip_max)` | `src/itx_tmpl.c:109-110` |
//! | İkinci 1B: her sütun, stride `w` | `src/itx_tmpl.c:112-113` |
//! | Son: `dst[x] = iclip_pixel(dst[x] + ((c+8)>>4))` | `src/itx_tmpl.c:115-118` |
//! | `shift` tablosu (tx boyutu → shift) | `src/itx_tmpl.c:160-178` |
//! | WHT: `coeff>>2`, iki WHT geçişi, `memset(coeff, 0, 4·4)`, doğrudan `dst += *c` | `src/itx_tmpl.c:184-202` |
//! | WHT yalnız `TX_4X4` yuvasına atanır | `src/itx_tmpl.c:268` |
//! | Destek matrisi (hangi boyut hangi türleri alır) | `src/itx_tmpl.c:240-302` |
//! | 8-bit `coef = int16_t`, 16bpc `coef = int32_t` | `include/common/bitdepth.h:44,62` |
//! | 16bpc `PXSTRIDE(x) = x >> 1` (çağrı stride'ı **bayt**) | `include/common/bitdepth.h:79-82` |
//! | 1B çekirdekler ve 64 noktalı sıfır bölgeleri | `src/itx_1d.c:65-1081` |
//!
//! ## Katsayı yerleşimi ve 64 noktalı sıfır bölgeleri
//!
//! Ters dönüşüm katsayı tamponu dav1d'de `coeff[satır + sütun·sh]` olarak
//! okunur (`src/itx_tmpl.c:96-104`); burada `sh = min(h,32)` ve satır
//! **dikey** frekanstır. Yani tampon, yüksekliği `sh` olan sütun-majör
//! yerleşimdir. `w > 32` veya `h > 32` olduğunda yalnız sol üst 32×32 bölge
//! kullanılır; 64 noktalı 1B çekirdek yalnız ilk 32 girişi okur (bileşik
//! stride'lar nedeniyle en büyük erişilen indeks 31'dir, `src/itx_1d.c:436-781`).
//! Bu adımda üretim tarafı katsayı çözmediği için yerleşim yalnız **çağrı
//! sözleşmesidir**; F2b'de katsayı çözücü bu yerleşimi üretecektir.
//!
//! ## eob hızlı yolları
//!
//! `last_nonzero_col` (ilk 1B geçişte işlenecek en büyük satır) üç durumda
//! farklı hesaplanır (`src/itx_tmpl.c:87-95`):
//!
//! 1. yalnız dikey dönüşüm (`h = IDENTITY`, `v ≠ IDENTITY`): `min(sh−1, eob)`;
//! 2. yalnız yatay dönüşüm (`h ≠ IDENTITY`, `v = IDENTITY`): `eob >> (lw+2)`;
//! 3. diğerleri: `dav1d_last_nonzero_col_from_eob[tx][eob]` — dav1d bunu
//!    `scan.c:319-331` `init_tbl` ile üretir; bu port aynı tabloyu
//!    [`crate::dsp::itx_tables`] içinde derleme zamanı sabiti olarak taşır.
//!
//! `eob = 0` ve `txtp = DCT_DCT` **DC-only** yoludur (`has_dconly`); bu yolda
//! yalnız `coeff[0]` sıfırlanır, diğer katsayılara dokunulmaz. Genel yolda
//! `coeff[0..sw·sh]` tamamen sıfırlanır (`src/itx_tmpl.c:108`).
//!
//! ## Aritmetik
//!
//! 1B çekirdekler tüm ara değerleri `i64` tutar (tanımlı aritmetik; C'nin
//! işaretli taşma UB'si taşınmaz). Geçerli AV1 katsayı aralığında sonuç C ile
//! bit-aynıdır; ayrıntı [`crate::dsp::scalar::itx_1d`] belgesindedir.
//!
//! ## Dispatch sözleşmesi (ADR-0003)
//!
//! [`ItxFn`] = **dilim + bayt stride + açık tx boyutu + açık tx türü + açık
//! bit derinliği**. Bit derinliği başına **ayrı yuva** vardır
//! (`DspDispatch::itx: [Option<ItxFn>; 3]`; 8/10/12 → 0/1/2). `None` →
//! skaler yol ([`crate::dsp::scalar::itx::inverse_transform_add`]). Çekirdek
//! hedef dilimin `(h−1)·stride + w·bpp` baytını okur/yazar; dışına taşmaz,
//! sözleşme dışı girdide panikler (güvenli indeksleme).

#![forbid(unsafe_code)]

use crate::dsp::scalar::itx_1d::Tx1dType;

/// Bir ters dönüşüm çekirdeğinin imzası (ADR-0003: dilim + bayt stride +
/// açık tx boyutu/türü + açık bit derinliği).
///
/// - `dst`: hedef düzlem dilimi (**bayt**); 16bpc'de piksel 2 bayttır.
/// - `dst_stride`: satır adımı **bayt** (dav1d `PXSTRIDE` ile böler).
/// - `coeff`: 32×32 = 1024 girdilik katsayı tamponu (dav1d yerleşimi).
/// - `eob`: katsayı sonu (0..=sw·sh−1).
/// - `tx`: kare/dikdörtgen tx boyutu.
/// - `txtp`: tx türü (WHT dâhil).
/// - `bit_depth`: açık bit derinliği; yuva anahtarının parçası.
pub type ItxFn = fn(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
    bit_depth: crate::frame::BitDepth,
);

/// dav1d `N_RECT_TX_SIZES` (`src/levels.h:62-78`).
pub const N_RECT_TX_SIZES: usize = 19;

/// dav1d `N_TX_TYPES_PLUS_LL` (`src/levels.h:80-100`): 16 tür + `WHT_WHT`.
pub const N_TX_TYPES_PLUS_LL: usize = 17;

/// itx çekirdek ailesi — dav1d `Dav1dInvTxfmDSPContext` (`src/itx.h:43-45`):
/// `itxfm_add[N_RECT_TX_SIZES][N_TX_TYPES_PLUS_LL]`.
///
/// Her bit derinliğinin kendi tablosu vardır (F3c). Skaler tabloda her hücre
/// o bit derinliğinin monomorfu
/// ([`crate::dsp::scalar::itx::inverse_transform_add`]) olur; desteklenmeyen
/// hücreler ([`is_supported`] yanlış) çözme hattında çağrılmaz, çağrılırsa
/// skaler gövde panikler.
#[derive(Clone, Copy)]
pub struct ItxDsp {
    /// `itxfm_add[tx][txtp]`.
    pub itxfm_add: [[ItxFn; N_TX_TYPES_PLUS_LL]; N_RECT_TX_SIZES],
}

impl ItxDsp {
    /// `BPC` bit derinliğinin skaler tablosu.
    #[must_use]
    pub const fn scalar<const BPC: u32>() -> Self {
        let f: ItxFn = crate::dsp::scalar::itx::inverse_transform_add::<BPC>;
        Self {
            itxfm_add: [[f; N_TX_TYPES_PLUS_LL]; N_RECT_TX_SIZES],
        }
    }

    /// Çalışma zamanı bit derinliğinin skaler tablosu (kurulum ve testler için;
    /// çözme hattı tabloyu [`crate::dsp::DspDispatch::for_bit_depth`] ile alır).
    #[must_use]
    pub const fn scalar_for(bit_depth: crate::frame::BitDepth) -> Self {
        match bit_depth {
            crate::frame::BitDepth::Eight => Self::scalar::<8>(),
            crate::frame::BitDepth::Ten => Self::scalar::<10>(),
            crate::frame::BitDepth::Twelve => Self::scalar::<12>(),
        }
    }
}

impl core::fmt::Debug for ItxDsp {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "ItxDsp {{ itxfm_add: [[fn; {N_TX_TYPES_PLUS_LL}]; {N_RECT_TX_SIZES}] }}"
        )
    }
}

/// dav1d `enum RectTxfmSize` adları, diskriminant sırasıyla (`src/levels.h:52-78`).
const DAV1D_TX_SIZE_NAMES: [&str; N_RECT_TX_SIZES] = [
    "TX_4X4",
    "TX_8X8",
    "TX_16X16",
    "TX_32X32",
    "TX_64X64",
    "RTX_4X8",
    "RTX_8X4",
    "RTX_8X16",
    "RTX_16X8",
    "RTX_16X32",
    "RTX_32X16",
    "RTX_32X64",
    "RTX_64X32",
    "RTX_4X16",
    "RTX_16X4",
    "RTX_8X32",
    "RTX_32X8",
    "RTX_16X64",
    "RTX_64X16",
];

/// dav1d `enum TxfmType` adları, diskriminant sırasıyla (`src/levels.h:80-100`).
const DAV1D_TX_TYPE_NAMES: [&str; N_TX_TYPES_PLUS_LL] = [
    "DCT_DCT",
    "ADST_DCT",
    "DCT_ADST",
    "ADST_ADST",
    "FLIPADST_DCT",
    "DCT_FLIPADST",
    "FLIPADST_FLIPADST",
    "ADST_FLIPADST",
    "FLIPADST_ADST",
    "IDTX",
    "V_DCT",
    "H_DCT",
    "V_ADST",
    "H_ADST",
    "V_FLIPADST",
    "H_FLIPADST",
    "WHT_WHT",
];

/// Ters dönüşüm boyutu — dav1d `enum RectTxfmSize` (`src/levels.h:62-78`).
///
/// Diskriminant sırası dav1d ile **birebir** aynıdır (`N_TX_SIZES = 5`, sonra
/// 14 dikdörtgen boyut): tablo indeksleri bu sıraya dayanır.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TxSize {
    /// 4×4.
    Tx4x4 = 0,
    /// 8×8.
    Tx8x8 = 1,
    /// 16×16.
    Tx16x16 = 2,
    /// 32×32.
    Tx32x32 = 3,
    /// 64×64.
    Tx64x64 = 4,
    /// 4×8.
    Rtx4x8 = 5,
    /// 8×4.
    Rtx8x4 = 6,
    /// 8×16.
    Rtx8x16 = 7,
    /// 16×8.
    Rtx16x8 = 8,
    /// 16×32.
    Rtx16x32 = 9,
    /// 32×16.
    Rtx32x16 = 10,
    /// 32×64.
    Rtx32x64 = 11,
    /// 64×32.
    Rtx64x32 = 12,
    /// 4×16.
    Rtx4x16 = 13,
    /// 16×4.
    Rtx16x4 = 14,
    /// 8×32.
    Rtx8x32 = 15,
    /// 32×8.
    Rtx32x8 = 16,
    /// 16×64.
    Rtx16x64 = 17,
    /// 64×16.
    Rtx64x16 = 18,
}

/// Tüm tx boyutları, dav1d diskriminant sırasıyla.
pub const TX_SIZES: [TxSize; 19] = [
    TxSize::Tx4x4,
    TxSize::Tx8x8,
    TxSize::Tx16x16,
    TxSize::Tx32x32,
    TxSize::Tx64x64,
    TxSize::Rtx4x8,
    TxSize::Rtx8x4,
    TxSize::Rtx8x16,
    TxSize::Rtx16x8,
    TxSize::Rtx16x32,
    TxSize::Rtx32x16,
    TxSize::Rtx32x64,
    TxSize::Rtx64x32,
    TxSize::Rtx4x16,
    TxSize::Rtx16x4,
    TxSize::Rtx8x32,
    TxSize::Rtx32x8,
    TxSize::Rtx16x64,
    TxSize::Rtx64x16,
];

/// dav1d `TxfmInfo` (`src/tables.h:42-46`) alanlarından yalnız bu adımın
/// kullandıkları: 4 piksel biriminde `w`,`h` ve log2 karşılıkları.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TxfmInfo {
    /// Genişlik / 4.
    pub w: u8,
    /// Yükseklik / 4.
    pub h: u8,
    /// `log2(w)`.
    pub lw: u8,
    /// `log2(h)`.
    pub lh: u8,
}

impl TxSize {
    /// dav1d diskriminantı (`N_RECT_TX_SIZES = 19`).
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// dav1d sabit adı (`TX_4X4`, `RTX_4X8`, …) — yuva anahtarı ve matris eşlemesi.
    #[must_use]
    pub const fn dav1d_name(self) -> &'static str {
        DAV1D_TX_SIZE_NAMES[self as usize]
    }

    /// Diskriminanttan boyut; `19..=255` → `None`.
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Self> {
        if index < 19 {
            Some(TX_SIZES[index])
        } else {
            None
        }
    }

    /// dav1d `dav1d_txfm_dimensions[tx]` (`src/tables.c:129-168`).
    #[must_use]
    pub const fn info(self) -> TxfmInfo {
        match self {
            Self::Tx4x4 => TxfmInfo {
                w: 1,
                h: 1,
                lw: 0,
                lh: 0,
            },
            Self::Tx8x8 => TxfmInfo {
                w: 2,
                h: 2,
                lw: 1,
                lh: 1,
            },
            Self::Tx16x16 => TxfmInfo {
                w: 4,
                h: 4,
                lw: 2,
                lh: 2,
            },
            Self::Tx32x32 => TxfmInfo {
                w: 8,
                h: 8,
                lw: 3,
                lh: 3,
            },
            Self::Tx64x64 => TxfmInfo {
                w: 16,
                h: 16,
                lw: 4,
                lh: 4,
            },
            Self::Rtx4x8 => TxfmInfo {
                w: 1,
                h: 2,
                lw: 0,
                lh: 1,
            },
            Self::Rtx8x4 => TxfmInfo {
                w: 2,
                h: 1,
                lw: 1,
                lh: 0,
            },
            Self::Rtx8x16 => TxfmInfo {
                w: 2,
                h: 4,
                lw: 1,
                lh: 2,
            },
            Self::Rtx16x8 => TxfmInfo {
                w: 4,
                h: 2,
                lw: 2,
                lh: 1,
            },
            Self::Rtx16x32 => TxfmInfo {
                w: 4,
                h: 8,
                lw: 2,
                lh: 3,
            },
            Self::Rtx32x16 => TxfmInfo {
                w: 8,
                h: 4,
                lw: 3,
                lh: 2,
            },
            Self::Rtx32x64 => TxfmInfo {
                w: 8,
                h: 16,
                lw: 3,
                lh: 4,
            },
            Self::Rtx64x32 => TxfmInfo {
                w: 16,
                h: 8,
                lw: 4,
                lh: 3,
            },
            Self::Rtx4x16 => TxfmInfo {
                w: 1,
                h: 4,
                lw: 0,
                lh: 2,
            },
            Self::Rtx16x4 => TxfmInfo {
                w: 4,
                h: 1,
                lw: 2,
                lh: 0,
            },
            Self::Rtx8x32 => TxfmInfo {
                w: 2,
                h: 8,
                lw: 1,
                lh: 3,
            },
            Self::Rtx32x8 => TxfmInfo {
                w: 8,
                h: 2,
                lw: 3,
                lh: 1,
            },
            Self::Rtx16x64 => TxfmInfo {
                w: 4,
                h: 16,
                lw: 2,
                lh: 4,
            },
            Self::Rtx64x16 => TxfmInfo {
                w: 16,
                h: 4,
                lw: 4,
                lh: 2,
            },
        }
    }

    /// Piksel cinsinden genişlik (dav1d `4 * t_dim->w`).
    #[must_use]
    pub const fn width(self) -> usize {
        4 * self.info().w as usize
    }

    /// Piksel cinsinden yükseklik (dav1d `4 * t_dim->h`).
    #[must_use]
    pub const fn height(self) -> usize {
        4 * self.info().h as usize
    }

    /// İlk 1B geçişte kullanılan shift — dav1d `src/itx_tmpl.c:160-178`.
    #[must_use]
    pub const fn shift(self) -> u32 {
        // dav1d `src/itx_tmpl.c:160-178` sırasıyla birebir.
        match self {
            Self::Tx4x4 | Self::Rtx4x8 | Self::Rtx8x4 => 0,
            Self::Rtx4x16
            | Self::Tx8x8
            | Self::Rtx8x16
            | Self::Rtx16x4
            | Self::Rtx16x8
            | Self::Rtx16x32
            | Self::Rtx32x16
            | Self::Rtx32x64
            | Self::Rtx64x32 => 1,
            Self::Rtx8x32
            | Self::Rtx32x8
            | Self::Tx16x16
            | Self::Tx32x32
            | Self::Rtx16x64
            | Self::Rtx64x16
            | Self::Tx64x64 => 2,
        }
    }
}

/// Ters dönüşüm türü — dav1d `enum TxfmType` (`src/levels.h:80-100`).
///
/// Diskriminant sırası dav1d ile aynıdır: 0–15 `N_TX_TYPES`, `WHT_WHT = 16`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TxType {
    /// DCT / DCT.
    DctDct = 0,
    /// Dikey ADST, yatay DCT.
    AdstDct = 1,
    /// Dikey DCT, yatay ADST.
    DctAdst = 2,
    /// ADST / ADST.
    AdstAdst = 3,
    /// Dikey flipADST, yatay DCT.
    FlipadstDct = 4,
    /// Dikey DCT, yatay flipADST.
    DctFlipadst = 5,
    /// flipADST / flipADST.
    FlipadstFlipadst = 6,
    /// Dikey ADST, yatay flipADST.
    AdstFlipadst = 7,
    /// Dikey flipADST, yatay ADST.
    FlipadstAdst = 8,
    /// Özdeşlik / özdeşlik.
    Idtx = 9,
    /// Dikey DCT, yatay özdeşlik.
    VDct = 10,
    /// Yatay DCT, dikey özdeşlik.
    HDct = 11,
    /// Dikey ADST, yatay özdeşlik.
    VAdst = 12,
    /// Yatay ADST, dikey özdeşlik.
    HAdst = 13,
    /// Dikey flipADST, yatay özdeşlik.
    VFlipadst = 14,
    /// Yatay flipADST, dikey özdeşlik.
    HFlipadst = 15,
    /// Kayıpsız 4×4 WHT (`WHT_WHT = N_TX_TYPES`).
    WhtWht = 16,
}

/// On altı normal tx türü (WHT hariç), dav1d sırasıyla.
pub const TX_TYPES: [TxType; 16] = [
    TxType::DctDct,
    TxType::AdstDct,
    TxType::DctAdst,
    TxType::AdstAdst,
    TxType::FlipadstDct,
    TxType::DctFlipadst,
    TxType::FlipadstFlipadst,
    TxType::AdstFlipadst,
    TxType::FlipadstAdst,
    TxType::Idtx,
    TxType::VDct,
    TxType::HDct,
    TxType::VAdst,
    TxType::HAdst,
    TxType::VFlipadst,
    TxType::HFlipadst,
];

impl TxType {
    /// dav1d diskriminantı (WHT = 16).
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// dav1d sabit adı (`DCT_DCT`, …, `WHT_WHT`) — yuva anahtarı.
    #[must_use]
    pub const fn dav1d_name(self) -> &'static str {
        DAV1D_TX_TYPE_NAMES[self as usize]
    }

    /// Diskriminanttan tür; `16` → WHT, `17..=255` → `None`.
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::DctDct),
            1 => Some(Self::AdstDct),
            2 => Some(Self::DctAdst),
            3 => Some(Self::AdstAdst),
            4 => Some(Self::FlipadstDct),
            5 => Some(Self::DctFlipadst),
            6 => Some(Self::FlipadstFlipadst),
            7 => Some(Self::AdstFlipadst),
            8 => Some(Self::FlipadstAdst),
            9 => Some(Self::Idtx),
            10 => Some(Self::VDct),
            11 => Some(Self::HDct),
            12 => Some(Self::VAdst),
            13 => Some(Self::HAdst),
            14 => Some(Self::VFlipadst),
            15 => Some(Self::HFlipadst),
            16 => Some(Self::WhtWht),
            _ => None,
        }
    }

    /// Etkin (yatay, dikey) 1B dönüşüm türleri.
    ///
    /// dav1d yuvayı (vertical, horizontal) adlandırır ama `itxfm_add` tablosuna
    /// atarken işlev adlarını çevirir (`src/itx_tmpl.c:240-302`); **etkin**
    /// eşleme `tests/checkasm/itx.c:57-73` tablosunun aynısıdır. Bu port
    /// doğrudan etkin eşlemeyi kullanır.
    #[must_use]
    pub const fn tx1d(self) -> (Tx1dType, Tx1dType) {
        match self {
            Self::DctDct => (Tx1dType::Dct, Tx1dType::Dct),
            Self::AdstDct => (Tx1dType::Dct, Tx1dType::Adst),
            Self::DctAdst => (Tx1dType::Adst, Tx1dType::Dct),
            Self::AdstAdst => (Tx1dType::Adst, Tx1dType::Adst),
            Self::FlipadstDct => (Tx1dType::Dct, Tx1dType::FlipAdst),
            Self::DctFlipadst => (Tx1dType::FlipAdst, Tx1dType::Dct),
            Self::FlipadstFlipadst => (Tx1dType::FlipAdst, Tx1dType::FlipAdst),
            Self::AdstFlipadst => (Tx1dType::FlipAdst, Tx1dType::Adst),
            Self::FlipadstAdst => (Tx1dType::Adst, Tx1dType::FlipAdst),
            Self::Idtx => (Tx1dType::Identity, Tx1dType::Identity),
            Self::VDct => (Tx1dType::Identity, Tx1dType::Dct),
            Self::HDct => (Tx1dType::Dct, Tx1dType::Identity),
            Self::VAdst => (Tx1dType::Identity, Tx1dType::Adst),
            Self::HAdst => (Tx1dType::Adst, Tx1dType::Identity),
            Self::VFlipadst => (Tx1dType::Identity, Tx1dType::FlipAdst),
            Self::HFlipadst => (Tx1dType::FlipAdst, Tx1dType::Identity),
            Self::WhtWht => (Tx1dType::Identity, Tx1dType::Identity),
        }
    }
}

/// Verilen (tx boyutu, tx türü) hücresi dav1d tarafından **atanmış mı**?
///
/// dav1d `dav1d_itx_dsp_init` (`src/itx_tmpl.c:240-302`): 4×4–16×8 sekiz
/// boyutta 16 türün tamamı; `16×16` boyutunda 12 tür (`inv_txfm_fn16`; ADST/
/// flipADST'in 1B türleri yok); `8×32,16×32,32×8,32×16,32×32` boyutlarında
/// yalnız `DCT_DCT` + `IDTX`; `16×64,32×64,64×16,64×32,64×64` boyutlarında
/// yalnız `DCT_DCT`; WHT yalnız `4×4`.
#[must_use]
pub const fn is_supported(tx: TxSize, txtp: TxType) -> bool {
    if matches!(txtp, TxType::WhtWht) {
        return matches!(tx, TxSize::Tx4x4);
    }
    match tx {
        TxSize::Tx4x4
        | TxSize::Rtx4x8
        | TxSize::Rtx4x16
        | TxSize::Rtx8x4
        | TxSize::Tx8x8
        | TxSize::Rtx8x16
        | TxSize::Rtx16x4
        | TxSize::Rtx16x8 => true,
        TxSize::Tx16x16 => matches!(
            txtp,
            TxType::DctDct
                | TxType::AdstDct
                | TxType::DctAdst
                | TxType::AdstAdst
                | TxType::FlipadstDct
                | TxType::DctFlipadst
                | TxType::FlipadstFlipadst
                | TxType::AdstFlipadst
                | TxType::FlipadstAdst
                | TxType::Idtx
                | TxType::VDct
                | TxType::HDct
        ),
        TxSize::Rtx8x32
        | TxSize::Rtx16x32
        | TxSize::Rtx32x8
        | TxSize::Rtx32x16
        | TxSize::Tx32x32 => matches!(txtp, TxType::DctDct | TxType::Idtx),
        TxSize::Rtx16x64
        | TxSize::Rtx32x64
        | TxSize::Rtx64x16
        | TxSize::Rtx64x32
        | TxSize::Tx64x64 => matches!(txtp, TxType::DctDct),
    }
}

/// Yatay/dikey 1B işlevi tablodan seçer; boyutun kendi 1B çekirdeği yoksa
/// `None` (dav1d'de boş tablo hücresi; desteklenen hücrelerde oluşmaz).
#[must_use]
pub fn tx1d_fn(square_index: usize, kind: Tx1dType) -> Option<crate::dsp::scalar::itx_1d::Itx1dFn> {
    crate::dsp::scalar::itx_1d::TX1D_FNS[square_index][kind as usize]
}

/// P43: SIMD ITX girişlerinin sözleşme dışı girdide skaler gövdeye devrettiği
/// çağrı sayısı (süreç başından beri). Çözme hattında 0 olmalıdır.
static ITX_SKALER_DEVIR: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// [`ITX_SKALER_DEVIR`] değeri.
#[must_use]
pub fn itx_skaler_devir_sayisi() -> u64 {
    ITX_SKALER_DEVIR.load(core::sync::atomic::Ordering::Relaxed)
}

/// Skaler devri sayar (yalnız SIMD girişlerinin soğuk yolu).
#[cold]
pub fn itx_skaler_devri_say() {
    ITX_SKALER_DEVIR.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
}

/// `(h − 1)·stride + w_bayt` (taşmada `None`).
fn itx_bolge_sonu(h: usize, stride: usize, w_bayt: usize) -> Option<usize> {
    (h - 1).checked_mul(stride)?.checked_add(w_bayt)
}

/// P43: 8 bpc (`bpp = 1`) SIMD `itxfm_add` girişinin O(1) sözleşmesi:
/// `is_supported(tx, txtp)`, `coeff_len ≥ 1024`, `dst_stride ≥ w`,
/// `dst_len ≥ (h − 1)·dst_stride + w`, ve WHT/DC-only dışındaki genel yolda
/// `eob < sw·sh` (`sw = min(w, 32)`, `sh = min(h, 32)`).
#[must_use]
pub fn itx8_sozlesmesi(
    dst_len: usize,
    dst_stride: usize,
    coeff_len: usize,
    eob: usize,
    tx: TxSize,
    txtp: TxType,
) -> bool {
    if !is_supported(tx, txtp) || coeff_len < 1024 {
        return false;
    }
    let w = tx.width();
    let h = tx.height();
    let sw = w.min(32);
    let sh = h.min(32);
    let eob_ok = matches!(txtp, TxType::WhtWht)
        || (matches!(txtp, TxType::DctDct) && eob == 0)
        || eob < sw * sh;
    eob_ok && dst_stride >= w && itx_bolge_sonu(h, dst_stride, w).is_some_and(|n| dst_len >= n)
}

/// P43: 16 bpc (`10`/`12` bit, `bpp = 2`) SIMD `itxfm_add` girişinin O(1)
/// sözleşmesi: `is_supported(tx, txtp)`, `coeff_len ≥ 1024`, `dst_stride` çift
/// ve `dst_stride ≥ w·2`, `dst_len ≥ (h − 1)·dst_stride + w·2`, ve WHT/DC-only
/// dışındaki genel yolda `eob < sw·sh`.
#[must_use]
pub fn itx16_sozlesmesi(
    dst_len: usize,
    dst_stride: usize,
    coeff_len: usize,
    eob: usize,
    tx: TxSize,
    txtp: TxType,
) -> bool {
    if !is_supported(tx, txtp) || coeff_len < 1024 || !dst_stride.is_multiple_of(2) {
        return false;
    }
    let w = tx.width();
    let h = tx.height();
    let w_bayt = w * 2;
    let sw = w.min(32);
    let sh = h.min(32);
    let eob_ok = matches!(txtp, TxType::WhtWht)
        || (matches!(txtp, TxType::DctDct) && eob == 0)
        || eob < sw * sh;
    eob_ok
        && dst_stride >= w_bayt
        && itx_bolge_sonu(h, dst_stride, w_bayt).is_some_and(|n| dst_len >= n)
}

/// P43: Sözleşme dışı girdide skaler devir sayacını artırır; tampon ve hücre
/// skaler gövdenin güvenli önkoşullarını sağlıyorsa
/// [`crate::dsp::scalar::itx::inverse_transform_add`] çalıştırır, sağlamıyorsa
/// paniklemeden döner.
#[cold]
pub fn itx_skaler_devret<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
) {
    itx_skaler_devri_say();
    let bpp = if BPC == 8 { 1 } else { 2 };
    let w_bayt = tx.width() * bpp;
    let h = tx.height();
    let sw = tx.width().min(32);
    let sh = h.min(32);
    let eob_ok = matches!(txtp, TxType::WhtWht)
        || (matches!(txtp, TxType::DctDct) && eob == 0)
        || eob < sw * sh;
    if is_supported(tx, txtp)
        && coeff.len() >= 1024
        && eob_ok
        && dst_stride >= w_bayt
        && itx_bolge_sonu(h, dst_stride, w_bayt).is_some_and(|n| dst.len() >= n)
    {
        let bd = match BPC {
            8 => crate::frame::BitDepth::Eight,
            10 => crate::frame::BitDepth::Ten,
            _ => crate::frame::BitDepth::Twelve,
        };
        crate::dsp::scalar::itx::inverse_transform_add::<BPC>(
            dst, dst_stride, coeff, eob, tx, txtp, bd,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tx_boyutlari_dav1d_sirasiyla_tam() {
        for (index, tx) in TX_SIZES.iter().enumerate() {
            assert_eq!(tx.index(), index);
            assert_eq!(TxSize::from_index(index), Some(*tx));
            // dav1d tablosundaki w,h: 4..64 ve 2'nin kuvveti.
            assert!((4..=64).contains(&tx.width()));
            assert!((4..=64).contains(&tx.height()));
            assert_eq!(1usize << tx.info().lw, tx.width() / 4);
            assert_eq!(1usize << tx.info().lh, tx.height() / 4);
        }
        assert_eq!(TxSize::from_index(19), None);
    }

    #[test]
    fn tx_turleri_dav1d_sirasiyla_tam() {
        for (index, txtp) in TX_TYPES.iter().enumerate() {
            assert_eq!(txtp.index(), index);
            assert_eq!(TxType::from_index(index), Some(*txtp));
            // WHT dâhil her türün 1B eşlemesi vardır.
            let (h, v) = txtp.tx1d();
            assert!(h as usize <= 3 && v as usize <= 3);
        }
        assert_eq!(TxType::from_index(16), Some(TxType::WhtWht));
        assert_eq!(TxType::from_index(17), None);
    }

    #[test]
    fn destek_matrisi_dav1d_init_ile_ayni() {
        // 16 türün tamamını alan 8 boyut.
        const FULL: [TxSize; 8] = [
            TxSize::Tx4x4,
            TxSize::Rtx4x8,
            TxSize::Rtx4x16,
            TxSize::Rtx8x4,
            TxSize::Tx8x8,
            TxSize::Rtx8x16,
            TxSize::Rtx16x4,
            TxSize::Rtx16x8,
        ];
        // 16×16: 12 tür (ADST/flipADST 1B türleri atanmamış).
        const TWELVE: [TxType; 12] = [
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
            TxType::FlipadstDct,
            TxType::DctFlipadst,
            TxType::FlipadstFlipadst,
            TxType::AdstFlipadst,
            TxType::FlipadstAdst,
            TxType::Idtx,
            TxType::VDct,
            TxType::HDct,
        ];
        // Yalnız DCT_DCT + IDTX.
        const TWO: [TxSize; 5] = [
            TxSize::Rtx8x32,
            TxSize::Rtx16x32,
            TxSize::Rtx32x8,
            TxSize::Rtx32x16,
            TxSize::Tx32x32,
        ];
        // Yalnız DCT_DCT.
        const ONE: [TxSize; 5] = [
            TxSize::Rtx16x64,
            TxSize::Rtx32x64,
            TxSize::Rtx64x16,
            TxSize::Rtx64x32,
            TxSize::Tx64x64,
        ];
        let mut cells = 0usize;
        for tx in TX_SIZES {
            for txtp in TX_TYPES {
                let expected = FULL.contains(&tx)
                    || (tx == TxSize::Tx16x16 && TWELVE.contains(&txtp))
                    || (TWO.contains(&tx) && matches!(txtp, TxType::DctDct | TxType::Idtx))
                    || (ONE.contains(&tx) && txtp == TxType::DctDct);
                assert_eq!(is_supported(tx, txtp), expected, "{tx:?}/{txtp:?}");
                if expected {
                    cells += 1;
                }
            }
            assert_eq!(is_supported(tx, TxType::WhtWht), tx == TxSize::Tx4x4);
            if tx == TxSize::Tx4x4 {
                cells += 1;
            }
        }
        assert_eq!(cells, 156, "hücre sayısı dav1d init ile eşleşmeli");
    }

    #[test]
    fn shift_tablosu_dav1d_ile_ayni() {
        // dav1d src/itx_tmpl.c:160-178.
        let expected: [(TxSize, u32); 19] = [
            (TxSize::Tx4x4, 0),
            (TxSize::Rtx4x8, 0),
            (TxSize::Rtx4x16, 1),
            (TxSize::Rtx8x4, 0),
            (TxSize::Tx8x8, 1),
            (TxSize::Rtx8x16, 1),
            (TxSize::Rtx8x32, 2),
            (TxSize::Rtx16x4, 1),
            (TxSize::Rtx16x8, 1),
            (TxSize::Tx16x16, 2),
            (TxSize::Rtx16x32, 1),
            (TxSize::Rtx16x64, 2),
            (TxSize::Rtx32x8, 2),
            (TxSize::Rtx32x16, 1),
            (TxSize::Tx32x32, 2),
            (TxSize::Rtx32x64, 1),
            (TxSize::Rtx64x16, 2),
            (TxSize::Rtx64x32, 1),
            (TxSize::Tx64x64, 2),
        ];
        for (tx, shift) in expected {
            assert_eq!(tx.shift(), shift, "{tx:?}");
        }
    }

    #[test]
    fn p43_itx_sozlesmesi_sinir_denetimi() {
        assert!(itx8_sozlesmesi(
            16,
            4,
            1024,
            100,
            TxSize::Tx4x4,
            TxType::WhtWht
        ));
        assert!(itx8_sozlesmesi(
            16,
            4,
            1024,
            0,
            TxSize::Tx4x4,
            TxType::DctDct
        ));
        assert!(itx8_sozlesmesi(
            16,
            4,
            1024,
            15,
            TxSize::Tx4x4,
            TxType::DctDct
        ));
        assert!(!itx8_sozlesmesi(
            16,
            4,
            1024,
            16,
            TxSize::Tx4x4,
            TxType::DctDct
        ));
        assert!(!itx8_sozlesmesi(
            15,
            4,
            1024,
            0,
            TxSize::Tx4x4,
            TxType::WhtWht
        ));
        assert!(!itx8_sozlesmesi(
            16,
            3,
            1024,
            0,
            TxSize::Tx4x4,
            TxType::WhtWht
        ));
        assert!(!itx8_sozlesmesi(
            16,
            4,
            1023,
            0,
            TxSize::Tx4x4,
            TxType::WhtWht
        ));
        assert!(!itx8_sozlesmesi(
            64,
            8,
            1024,
            0,
            TxSize::Tx8x8,
            TxType::WhtWht
        ));

        assert!(itx16_sozlesmesi(
            32,
            8,
            1024,
            100,
            TxSize::Tx4x4,
            TxType::WhtWht
        ));
        assert!(!itx16_sozlesmesi(
            32,
            9,
            1024,
            0,
            TxSize::Tx4x4,
            TxType::WhtWht
        ));
        assert!(!itx16_sozlesmesi(
            31,
            8,
            1024,
            0,
            TxSize::Tx4x4,
            TxType::WhtWht
        ));
        assert!(!itx16_sozlesmesi(
            32,
            6,
            1024,
            0,
            TxSize::Tx4x4,
            TxType::WhtWht
        ));
    }
}
