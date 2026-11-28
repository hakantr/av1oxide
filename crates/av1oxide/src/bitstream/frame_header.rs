//! Frame header OBU'sunun (uncompressed header) tam ayrıştırması (F2a / S2a.2).
//!
//! ## Sorumluluk
//!
//! [`FrameHeader::parse`] AV1 `uncompressed_header()` sözdiziminin **tamamını**
//! ayrıştırır ve `Dav1dFrameHeader` (`include/dav1d/headers.h` 335–434) yapısının
//! alan alan Rust karşılığını üretir:
//!
//! - kare türü, `show_frame`/`showable_frame`, `error_resilient_mode`,
//!   `disable_cdf_update`, `allow_screen_content_tools`/`force_integer_mv`,
//!   `frame_id`, `frame_size_override`, `order_hint`, `primary_ref_frame`,
//! - decoder model `buffer_removal_time` alanları (operating point başına),
//! - `refresh_frame_flags`, `ref_order_hint` (dav1d'de `frame_offset`),
//!   `allow_intrabc`, `frame_refs_short_signaling` **ve referans seçim
//!   algoritması**, `ref_frame_idx` + `delta_frame_id`,
//! - `frame_size()` / `frame_size_with_refs()` / `render_size()` / superres,
//! - yüksek hassasiyetli mv, interpolasyon filtresi, motion mode,
//!   `use_ref_frame_mvs`, `refresh_context`,
//! - tile bilgisi: uniform/non-uniform bölümleme, `context_update_tile_id`,
//!   `tile_size_bytes`,
//! - kuantizasyon (delta'lar + qm), segmentasyon (primary ref'ten devralma
//!   dâhil), delta q / delta lf, loop filter (delta devralma dâhil), CDEF,
//!   loop restoration, tx mode, compound ref mode, skip mode, warped motion,
//!   reduced tx set, global motion (önceki parametrelere göre), film grain
//!   (load/update).
//!
//! [`parse_tile_group`] tile group / frame OBU'sunun **başlığını** (tile
//! `start`/`end`) ve tile **boyut alanlarını** ayrıştırır; tile verisi
//! **çözülmez**, yalnız sınırlarıyla atlanır (F2b kapsamı).
//!
//! ## Upstream karşılığı (dav1d `99f3354`, salt-okunur) ve AV1 spesifikasyonu
//!
//! Satır numaraları `src/obu.c` içindeki `parse_frame_hdr()` gövdesine aittir
//! (fonksiyon 409–1166); tile group başlığı `parse_tile_hdr()` 1157–1166.
//!
//! | Alan grubu | dav1d `src/obu.c` | AV1 spesifikasyonu |
//! |---|---|---|
//! | `show_existing_frame`, `existing_frame_idx`, `frame_presentation_delay`, `frame_id` (gösterilen kare) | 416–433 | §5.9.2, §6.8.2; §7.8 (referans yenileme) |
//! | kare türü, `show_frame`, `showable_frame`, `error_resilient_mode` | 435–453 | §5.9.2 `uncompressed_header()`, §6.8.2 |
//! | `disable_cdf_update`, `allow_screen_content_tools`, `force_integer_mv` | 455–465 | §5.9.2, §6.8.2 |
//! | `frame_id` | 467–468 | §5.9.2, §6.8.2 |
//! | `frame_size_override`, `order_hint`, `primary_ref_frame` | 470–478 | §5.9.2, §6.8.2 |
//! | `buffer_removal_time_present` + operating point zamanları | 480–493 | §5.9.2, §6.8.2 |
//! | `refresh_frame_flags`, `ref_order_hint`, `frame_size()` (key/intra) | 495–509 | §5.9.2–§5.9.4, §6.8.2–§6.8.4 |
//! | `refresh_frame_flags`, `ref_order_hint`, `frame_refs_short_signaling` + referans seçimi, `ref_frame_idx`, `delta_frame_id`, `frame_size_with_refs()`, `hp`, `subpel_filter_mode`, `switchable_motion_mode`, `use_ref_frame_mvs` (inter/switch) | 510–597 | §5.9.2, §5.9.5, §6.8.2, §6.8.5 |
//! | `refresh_context` | 600–603 | §5.9.2, §6.8.2 |
//! | tile bilgisi (`uniform`, `log2_cols/rows`, `col_start_sb`, `row_start_sb`, `context_update_tile_id`, `tile_size_bytes`) | 606–677 | §5.9.6 `tile_info()`, §6.8.6 |
//! | kuantizasyon (`base_q_idx`, delta'lar, qm) | 680–718 | §5.9.7 `quantization_params()`, §6.8.7 |
//! | segmentasyon (update_map/temporal/update_data, primary ref'ten devralma) | 721–786 | §5.9.8 `segmentation_params()`, §6.8.8 |
//! | delta q / delta lf | 789–803 | §5.9.9–§5.9.10, §6.8.9–§6.8.10 |
//! | lossless/qidx türetimi (`all_lossless`) | 806–816 | §6.8.8, §7.11.1 |
//! | loop filter (delta devralma dâhil) | 819–853 | §5.9.11 `loop_filter_params()`, §6.8.11 |
//! | CDEF | 856–868 | §5.9.12 `cdef_params()`, §6.8.12 |
//! | loop restoration | 871–900 | §5.9.13 `lr_params()`, §6.8.13 |
//! | tx mode | 903–905 | §5.9.14 `read_tx_mode()`, §6.8.14 |
//! | `switchable_comp_refs` + skip mode | 907–998 | §5.9.15–§5.9.16, §6.8.15–§6.8.16 |
//! | warped motion izni, reduced tx set | 1000–1008 | §5.9.2, §6.8.2 |
//! | global motion (önceki parametrelere göre) | 1010–1066 | §5.9.17 `global_motion_params()`, §6.8.17 |
//! | film grain (present/update/load) | 1068–1159 | §5.9.18 `film_grain_params()`, §6.8.18 |
//! | tile group başlığı (start/end) | 1157–1166 | §5.10.1 `tile_group_obu()`, §6.10.1 |
//! | tile boyut alanları (veri çözülmez) | `src/decode.c` 2900–2960 (tile size okuma) | §5.10.1, §6.10.1 |
//!
//! `dav1d_bytealign_get_bits`, `check_trailing_bits`, `dav1d_get_uniform`,
//! `dav1d_get_bits_subexp` ve `get_poc_diff` yardımcılarının karşılıkları
//! [`BitReader`] ve bu modülde birebir uygulanır (`src/env.h` 237–244,
//! `src/getbits.c` 114–165, `src/obu.c` 48–70).
//!
//! ## Reddetme kuralları (dav1d ile aynı)
//!
//! | Kural | dav1d satırı |
//! |---|---|
//! | `show_existing_frame` + `frame_id_numbers_present`: referans yok ya da `frame_id` uyuşmuyor | 425–431 |
//! | katı kipte INTRA + `refresh_frame_flags == 0xff` | 501–505 |
//! | `frame_size_with_refs`: referans kare yok | `read_frame_size` 349–352 |
//! | `frame_size_with_refs` / `frame_size_override`: `width[1] > max_width` ya da `height > max_height` | `read_frame_size` 355–359, 380–384 |
//! | referans seçim algoritmasında referans kare yok | 529–530 |
//! | `delta_frame_id` ile hesaplanan `frame_id` uyuşmuyor | 585–590 |
//! | `context_update_tile_id >= TileCols*TileRows` | 669–671 |
//! | `segmentation.update_data == 0` iken primary ref yok | 776–778 |
//! | loop filter delta devralmada primary ref yok | 837–839 |
//! | global motion devralmada primary ref yok | 1029–1031 |
//! | skip mode referansında kare yok | 946, 973 |
//! | film grain `num_y_points > 14`, `num_uv_points > 10`, artan olmayan noktalar, 4:2:0 UV tutarsızlığı | 1090–1132 |
//! | film grain `update == 0` iken referans yok / `refidx` 7 referansın hiçbiri değil | 1077–1081 |
//! | tile group: `start > end` ya da `start != n_tiles_so_far` | 1300–1306 |
//!
//! ## Bilinçli davranış farkları (gerekçeli)
//!
//! 1. **Paket atomikliği:** dav1d bir frame header ayrıştırma hatasında
//!    `c->frame_hdr = NULL` yapar ve yarım kalan referans durumunu bırakır;
//!    av1oxide bozuk paketin **hiçbir** durum değişikliğini uygulamaz
//!    ([`crate::api::Decoder::send_packet`]). Geçerli akışlarda fark yoktur;
//!    bozuk girdide kurtarma noktası tanımlıdır (GUV-01).
//! 2. **Tile group sınır denetimi:** dav1d tile boyut alanlarını tile çözme
//!    sırasında okur; burada OBU sınırında okunur ve tile verisi atlanır.
//!    Geçerli akışlarda tüketilen bayt sayısı aynıdır; bozuk girdide hata
//!    daha erken (ve konumlu) bildirilir. Tile verisi **çözülmez** (F2b).
//! 3. **Tanımsız alan varsayılanları:** dav1d `Dav1dFrameHeader`'ı `memset(0)`
//!    ile sıfırlar; buradaki [`FrameHeader::default`] aynı sıfır durumunu
//!    üretir (gmv matrisleri dâhil). `show_existing_frame` erken dönüşünde
//!    global motion varsayılanları **atanmaz** — dav1d ile aynı.
//!
//! ## Sınır
//!
//! MSAC/CDF (S2a.3), tile/blok çözme ve kare üretimi (F2b), film grain
//! uygulaması (F2d) bu modülde yoktur.

#![forbid(unsafe_code)]

use alloc::vec::Vec;

use crate::bitstream::reader::BitReader;
use crate::bitstream::seq_header::{AdaptiveBoolean, SequenceHeader};
use crate::error::{BitstreamError, BitstreamErrorKind, Error, Result};

/// `DAV1D_MAX_SEGMENTS` (headers.h 43).
pub const MAX_SEGMENTS: usize = 8;
/// `DAV1D_MAX_TILE_COLS` (headers.h 41).
pub const MAX_TILE_COLS: usize = 64;
/// `DAV1D_MAX_TILE_ROWS` (headers.h 42).
pub const MAX_TILE_ROWS: usize = 64;
/// `DAV1D_REFS_PER_FRAME` (headers.h 46).
pub const REFS_PER_FRAME: usize = 7;
/// `DAV1D_TOTAL_REFS_PER_FRAME` (headers.h 47).
pub const TOTAL_REFS_PER_FRAME: usize = REFS_PER_FRAME + 1;
/// Referans yuvası sayısı (AV1 `NUM_REF_FRAMES`).
pub const NUM_REF_FRAMES: usize = 8;
/// `DAV1D_PRIMARY_REF_NONE` (headers.h 45).
pub const PRIMARY_REF_NONE: u8 = 7;
/// `DAV1D_MAX_CDEF_STRENGTHS` (headers.h 39).
pub const MAX_CDEF_STRENGTHS: usize = 8;
/// `DAV1D_MAX_OPERATING_POINTS` (headers.h 40).
pub const MAX_OPERATING_POINTS: usize = 32;
/// `DAV1D_MAX_SEGMENTS` (headers.h 43).
pub const MAX_SEGMENT_ID: usize = MAX_SEGMENTS;

/// dav1d `enum Dav1dFrameType` (headers.h 303–308).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FrameType {
    /// 0: anahtar kare (KEY_FRAME).
    #[default]
    Key,
    /// 1: ara kare (INTER_FRAME).
    Inter,
    /// 2: anahtar olmayan intra kare (INTRA_ONLY_FRAME).
    Intra,
    /// 3: geçiş karesi (SWITCH_FRAME).
    Switch,
}

impl FrameType {
    /// 2 bit ham değerden tür.
    #[must_use]
    pub const fn from_raw(raw: u8) -> Self {
        match raw & 0x03 {
            1 => Self::Inter,
            2 => Self::Intra,
            3 => Self::Switch,
            _ => Self::Key,
        }
    }

    /// dav1d enum değeri (0–3).
    #[must_use]
    pub const fn raw(self) -> u8 {
        match self {
            Self::Key => 0,
            Self::Inter => 1,
            Self::Intra => 2,
            Self::Switch => 3,
        }
    }

    /// `IS_KEY_OR_INTRA()` (dav1d `src/levels.h` 58–59).
    #[must_use]
    pub const fn is_key_or_intra(self) -> bool {
        matches!(self, Self::Key | Self::Intra)
    }

    /// `IS_INTER_OR_SWITCH()` (dav1d `src/levels.h` 60–61).
    #[must_use]
    pub const fn is_inter_or_switch(self) -> bool {
        matches!(self, Self::Inter | Self::Switch)
    }
}

/// dav1d `enum Dav1dTxfmMode` (headers.h 62–67).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TxfmMode {
    /// 0: yalnız 4x4.
    #[default]
    FourByFour,
    /// 1: en büyük dönüşüm.
    Largest,
    /// 2: seçilebilir.
    Switchable,
}

impl TxfmMode {
    /// dav1d enum değeri.
    #[must_use]
    pub const fn raw(self) -> u8 {
        match self {
            Self::FourByFour => 0,
            Self::Largest => 1,
            Self::Switchable => 2,
        }
    }
}

/// dav1d `enum Dav1dFilterMode` (headers.h 69–77).
///
/// Alt örnekleme için `Switchable` dav1d'de `DAV1D_N_FILTERS` (4) değerini
/// taşır; döküm bu ham değeri kullanır.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FilterMode {
    /// 0: 8-tap regular.
    #[default]
    EightTapRegular,
    /// 1: 8-tap smooth.
    EightTapSmooth,
    /// 2: 8-tap sharp.
    EightTapSharp,
    /// 3: bilinear (switchable olmayan yol).
    Bilinear,
    /// 4: switchable (`DAV1D_FILTER_SWITCHABLE`).
    Switchable,
}

impl FilterMode {
    /// 2 bit ham değerden mod (yalnız `switchable` biti 0 iken okunur).
    #[must_use]
    pub const fn from_raw(raw: u8) -> Self {
        match raw & 0x03 {
            0 => Self::EightTapRegular,
            1 => Self::EightTapSmooth,
            2 => Self::EightTapSharp,
            _ => Self::Bilinear,
        }
    }

    /// dav1d enum değeri.
    #[must_use]
    pub const fn raw(self) -> u8 {
        match self {
            Self::EightTapRegular => 0,
            Self::EightTapSmooth => 1,
            Self::EightTapSharp => 2,
            Self::Bilinear => 3,
            Self::Switchable => 4,
        }
    }
}

/// dav1d `enum Dav1dRestorationType` (headers.h 83–88).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RestorationType {
    /// 0: yok.
    #[default]
    None,
    /// 1: seçilebilir.
    Switchable,
    /// 2: Wiener.
    Wiener,
    /// 3: SGRPROJ.
    Sgrproj,
}

impl RestorationType {
    /// 2 bit ham değerden tür.
    #[must_use]
    pub const fn from_raw(raw: u8) -> Self {
        match raw & 0x03 {
            1 => Self::Switchable,
            2 => Self::Wiener,
            3 => Self::Sgrproj,
            _ => Self::None,
        }
    }

    /// dav1d enum değeri.
    #[must_use]
    pub const fn raw(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Switchable => 1,
            Self::Wiener => 2,
            Self::Sgrproj => 3,
        }
    }

    /// `NONE` değil mi?
    #[must_use]
    pub const fn is_present(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// dav1d `enum Dav1dWarpedMotionType` (headers.h 90–95).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WarpedMotionType {
    /// 0: özdeşlik.
    #[default]
    Identity,
    /// 1: öteleme.
    Translation,
    /// 2: dönme + yakınlaştırma.
    RotZoom,
    /// 3: afin.
    Affine,
}

impl WarpedMotionType {
    /// dav1d enum değeri.
    #[must_use]
    pub const fn raw(self) -> u8 {
        match self {
            Self::Identity => 0,
            Self::Translation => 1,
            Self::RotZoom => 2,
            Self::Affine => 3,
        }
    }
}

/// `Dav1dWarpedMotionParams` (headers.h 97–107).
///
/// `default()` **sıfır matris** üretir: dav1d `Dav1dFrameHeader`'ı `memset(0)`
/// ile sıfırladığı için `show_existing_frame` erken dönüşünde matris sıfırdır.
/// Kimlik matrisi [`DEFAULT_WM_PARAMS`] ile ayrıca atanır (dav1d `tables.c`
/// 388–397, `dav1d_default_wm_params`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WarpedMotionParams {
    /// Hareket türü.
    pub kind: WarpedMotionType,
    /// 16.16 sabit noktalı 2x3 matris (dav1d `matrix[6]`).
    pub matrix: [i32; 6],
}

/// dav1d `dav1d_default_wm_params` (tables.c 388–397).
pub const DEFAULT_WM_PARAMS: WarpedMotionParams = WarpedMotionParams {
    kind: WarpedMotionType::Identity,
    matrix: [0, 0, 1 << 16, 0, 0, 1 << 16],
};

/// `Dav1dFilmGrainData` (headers.h 315–333).
///
/// `ar_coeffs_uv` dav1d'de `[2][25 + 3]`'tür (hizalama dolgusu dâhil); döküm
/// bütün 28 girişi karşılaştırdığı için Rust tarafı da 28 tutar.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FilmGrainData {
    /// `grain_seed`.
    pub seed: u32,
    /// `num_y_points` (0–14).
    pub num_y_points: i32,
    /// Luma noktaları: `[value, scaling]`.
    pub y_points: [[u8; 2]; 14],
    /// `chroma_scaling_from_luma`.
    pub chroma_scaling_from_luma: bool,
    /// UV düzlemi başına nokta sayısı (0–10).
    pub num_uv_points: [i32; 2],
    /// UV noktaları: `[plane][index][value, scaling]`.
    pub uv_points: [[[u8; 2]; 10]; 2],
    /// `scaling_shift` (8–11).
    pub scaling_shift: i32,
    /// `ar_coeff_lag` (0–3).
    pub ar_coeff_lag: i32,
    /// Luma AR katsayıları (−128..=127).
    pub ar_coeffs_y: [i8; 24],
    /// UV AR katsayıları (dav1d `[2][28]`).
    pub ar_coeffs_uv: [[i8; 28]; 2],
    /// `ar_coeff_shift` (6–9).
    pub ar_coeff_shift: u64,
    /// `grain_scale_shift` (0–3).
    pub grain_scale_shift: i32,
    /// UV çarpanları.
    pub uv_mult: [i32; 2],
    /// UV luma çarpanları.
    pub uv_luma_mult: [i32; 2],
    /// UV ofsetleri.
    pub uv_offset: [i32; 2],
    /// `overlap_flag`.
    pub overlap_flag: bool,
    /// `clip_to_restricted_range`.
    pub clip_to_restricted_range: bool,
}

/// Film grain başlığı: `present` + `update` + veri (headers.h 337–340).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FilmGrainHeader {
    /// `film_grain_params_present` (bu kare için).
    pub present: bool,
    /// `update_grain`.
    pub update: bool,
    /// Film grain parametreleri.
    pub data: FilmGrainData,
}

/// `Dav1dSegmentationData` (headers.h 296–302).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SegmentationData {
    /// `delta_q` (−255..=255).
    pub delta_q: i16,
    /// `delta_lf_y_v`.
    pub delta_lf_y_v: i8,
    /// `delta_lf_y_h`.
    pub delta_lf_y_h: i8,
    /// `delta_lf_u`.
    pub delta_lf_u: i8,
    /// `delta_lf_v`.
    pub delta_lf_v: i8,
    /// `ref` (−1 = segment için özel referans yok).
    pub reference: i8,
    /// `skip`.
    pub skip: bool,
    /// `globalmv`.
    pub globalmv: bool,
}

/// `Dav1dSegmentationDataSet` (headers.h 304–308).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SegmentationDataSet {
    /// Segment başına veri.
    pub d: [SegmentationData; MAX_SEGMENTS],
    /// `preskip`.
    pub preskip: bool,
    /// `last_active_segid` (−1 = etkin segment yok).
    pub last_active_segid: i8,
}

/// `Dav1dLoopfilterModeRefDeltas` (headers.h 310–313).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LoopfilterModeRefDeltas {
    /// `mode_delta[2]`.
    pub mode_delta: [i8; 2],
    /// `ref_delta[8]`.
    pub ref_delta: [i8; TOTAL_REFS_PER_FRAME],
}

/// dav1d `default_mode_ref_deltas` (obu.c 404–407).
pub const DEFAULT_MODE_REF_DELTAS: LoopfilterModeRefDeltas = LoopfilterModeRefDeltas {
    mode_delta: [0, 0],
    ref_delta: [1, 0, 0, 0, -1, 0, -1, -1],
};

/// `Dav1dFrameHeader.tiling` (headers.h 375–386).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TilingInfo {
    /// `uniform_tile_spacing_flag`.
    pub uniform: bool,
    /// `tile_size_bytes` (1–4).
    pub n_bytes: u8,
    /// `min_log2_cols`.
    pub min_log2_cols: u8,
    /// `max_log2_cols`.
    pub max_log2_cols: u8,
    /// `log2_cols`.
    pub log2_cols: u8,
    /// `TileCols`.
    pub cols: u8,
    /// `min_log2_rows`.
    pub min_log2_rows: u8,
    /// `max_log2_rows`.
    pub max_log2_rows: u8,
    /// `log2_rows`.
    pub log2_rows: u8,
    /// `TileRows`.
    pub rows: u8,
    /// Süper blok sütun başlangıçları (`TileCols + 1` anlamlı).
    pub col_start_sb: [u16; MAX_TILE_COLS + 1],
    /// Süper blok satır başlangıçları (`TileRows + 1` anlamlı).
    pub row_start_sb: [u16; MAX_TILE_ROWS + 1],
    /// `context_update_tile_id`.
    pub update: u16,
}

impl Default for TilingInfo {
    fn default() -> Self {
        Self {
            uniform: false,
            n_bytes: 0,
            min_log2_cols: 0,
            max_log2_cols: 0,
            log2_cols: 0,
            cols: 0,
            min_log2_rows: 0,
            max_log2_rows: 0,
            log2_rows: 0,
            rows: 0,
            col_start_sb: [0; MAX_TILE_COLS + 1],
            row_start_sb: [0; MAX_TILE_ROWS + 1],
            update: 0,
        }
    }
}

/// `Dav1dFrameHeader.quant` (headers.h 388–393).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct QuantizationParams {
    /// `base_q_idx`.
    pub yac: u8,
    /// `delta_q_y_dc`.
    pub ydc_delta: i8,
    /// `delta_q_u_dc`.
    pub udc_delta: i8,
    /// `delta_q_u_ac`.
    pub uac_delta: i8,
    /// `delta_q_v_dc`.
    pub vdc_delta: i8,
    /// `delta_q_v_ac`.
    pub vac_delta: i8,
    /// `using_qmatrix`.
    pub qm: bool,
    /// `qm_y`.
    pub qm_y: u8,
    /// `qm_u`.
    pub qm_u: u8,
    /// `qm_v`.
    pub qm_v: u8,
}

/// `Dav1dFrameHeader.segmentation` (headers.h 394–399).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SegmentationParams {
    /// `segmentation_enabled`.
    pub enabled: bool,
    /// `segmentation_update_map`.
    pub update_map: bool,
    /// `segmentation_temporal_update`.
    pub temporal: bool,
    /// `segmentation_update_data`.
    pub update_data: bool,
    /// Segment verisi.
    pub seg_data: SegmentationDataSet,
    /// Segment başına lossless bayrağı (türetilmiş).
    pub lossless: [bool; MAX_SEGMENTS],
    /// Segment başına `qindex` (türetilmiş).
    pub qidx: [u8; MAX_SEGMENTS],
}

/// `Dav1dFrameHeader.delta` (headers.h 400–410).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DeltaParams {
    /// `delta_q_present`.
    pub q_present: bool,
    /// `delta_q_res`.
    pub q_res_log2: u8,
    /// `delta_lf_present`.
    pub lf_present: bool,
    /// `delta_lf_res`.
    pub lf_res_log2: u8,
    /// `delta_lf_multi`.
    pub lf_multi: bool,
}

/// `Dav1dFrameHeader.loopfilter` (headers.h 412–418).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LoopFilterParams {
    /// `loop_filter_level[0]` (dikey).
    pub level_y: [u8; 2],
    /// `loop_filter_level_u`.
    pub level_u: u8,
    /// `loop_filter_level_v`.
    pub level_v: u8,
    /// `loop_filter_delta_enabled`.
    pub mode_ref_delta_enabled: bool,
    /// `loop_filter_delta_update`.
    pub mode_ref_delta_update: bool,
    /// Mod/ref delta'ları (devralınabilir).
    pub mode_ref_deltas: LoopfilterModeRefDeltas,
    /// `loop_filter_sharpness`.
    pub sharpness: u8,
}

/// `Dav1dFrameHeader.cdef` (headers.h 419–423).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CdefParams {
    /// `cdef_damping_minus_3 + 3`.
    pub damping: u8,
    /// `cdef_bits`.
    pub n_bits: u8,
    /// Luma strength'leri.
    pub y_strength: [u8; MAX_CDEF_STRENGTHS],
    /// Chroma strength'leri.
    pub uv_strength: [u8; MAX_CDEF_STRENGTHS],
}

/// `Dav1dFrameHeader.restoration` (headers.h 424–427).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RestorationParams {
    /// Düzlem başına restoration türü.
    pub kind: [RestorationType; 3],
    /// `lr_unit_shift` ile türetilen birim boyutu (log2).
    pub unit_size: [u8; 2],
}

/// `Dav1dFrameHeaderOperatingPoint` (headers.h 357–359).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FrameHeaderOperatingPoint {
    /// `buffer_removal_time`.
    pub buffer_removal_time: u32,
}

/// `Dav1dFrameHeader`'ın Rust karşılığı (headers.h 335–434).
///
/// Alan adları ve varsayılanları dav1d yapısıyla birebir eşlenir; döküm
/// (`framehdr_diff.py`) bütün alanları adlarıyla karşılaştırır.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    /// Film grain başlığı.
    pub film_grain: FilmGrainHeader,
    /// Kare türü.
    pub frame_type: FrameType,
    /// `[coded_width, upscaled_width]` (dav1d `width[2]`).
    pub width: [i32; 2],
    /// Kare yüksekliği (superres öncesi).
    pub height: i32,
    /// `order_hint` (dav1d `frame_offset`).
    pub frame_offset: u8,
    /// OBU uzantısından gelen temporal katman kimliği.
    pub temporal_id: u8,
    /// OBU uzantısından gelen spatial katman kimliği.
    pub spatial_id: u8,
    /// `show_existing_frame`.
    pub show_existing_frame: bool,
    /// `frame_to_show_map_idx`.
    pub existing_frame_idx: u8,
    /// `current_frame_id`.
    pub frame_id: u32,
    /// `frame_presentation_delay`.
    pub frame_presentation_delay: u32,
    /// `show_frame`.
    pub show_frame: bool,
    /// `showable_frame`.
    pub showable_frame: bool,
    /// `error_resilient_mode`.
    pub error_resilient_mode: bool,
    /// `disable_cdf_update`.
    pub disable_cdf_update: bool,
    /// `allow_screen_content_tools`.
    pub allow_screen_content_tools: bool,
    /// `force_integer_mv`.
    pub force_integer_mv: bool,
    /// `frame_size_override_flag`.
    pub frame_size_override: bool,
    /// `primary_ref_frame` (7 = `PRIMARY_REF_NONE`).
    pub primary_ref_frame: u8,
    /// `buffer_removal_time_present_flag`.
    pub buffer_removal_time_present: bool,
    /// Operating point başına `buffer_removal_time`.
    pub operating_points: [FrameHeaderOperatingPoint; MAX_OPERATING_POINTS],
    /// `refresh_frame_flags`.
    pub refresh_frame_flags: u8,
    /// `render_width`.
    pub render_width: i32,
    /// `render_height`.
    pub render_height: i32,
    /// Superres: `width_scale_denominator` + `enabled`.
    pub super_res: SuperResParams,
    /// `render_and_frame_size_different`.
    pub have_render_size: bool,
    /// `allow_intrabc`.
    pub allow_intrabc: bool,
    /// `frame_refs_short_signaling`.
    pub frame_ref_short_signaling: bool,
    /// `ref_frame_idx[7]` (−1 = atanmadı).
    pub refidx: [i8; REFS_PER_FRAME],
    /// `allow_high_precision_mv`.
    pub hp: bool,
    /// `interpolation_filter`.
    pub subpel_filter_mode: FilterMode,
    /// `is_motion_mode_switchable`.
    pub switchable_motion_mode: bool,
    /// `use_ref_frame_mvs`.
    pub use_ref_frame_mvs: bool,
    /// `disable_frame_end_update_cdf`'in tersi (dav1d `refresh_context`).
    pub refresh_context: bool,
    /// Tile bilgisi.
    pub tiling: TilingInfo,
    /// Kuantizasyon.
    pub quant: QuantizationParams,
    /// Segmentasyon.
    pub segmentation: SegmentationParams,
    /// Delta q / delta lf.
    pub delta: DeltaParams,
    /// Bütün segmentler lossless mı?
    pub all_lossless: bool,
    /// Loop filter.
    pub loopfilter: LoopFilterParams,
    /// CDEF.
    pub cdef: CdefParams,
    /// Loop restoration.
    pub restoration: RestorationParams,
    /// `tx_mode`.
    pub txfm_mode: TxfmMode,
    /// `reference_select`.
    pub switchable_comp_refs: bool,
    /// `skip_mode_allowed`.
    pub skip_mode_allowed: bool,
    /// `skip_mode_present`.
    pub skip_mode_enabled: bool,
    /// `skip_mode_frame[2]`.
    pub skip_mode_refs: [i8; 2],
    /// `allow_warped_motion`.
    pub warp_motion: bool,
    /// `reduced_tx_set`.
    pub reduced_txtp_set: bool,
    /// Referans başına global motion parametreleri.
    pub gmv: [WarpedMotionParams; REFS_PER_FRAME],
}

/// Superres alanları (headers.h 365–368).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SuperResParams {
    /// `width_scale_denominator` (8 = kapalı).
    pub width_scale_denominator: u8,
    /// `enable_superres` (bu kare için).
    pub enabled: bool,
}

impl Default for FrameHeader {
    fn default() -> Self {
        Self {
            film_grain: FilmGrainHeader::default(),
            frame_type: FrameType::default(),
            width: [0, 0],
            height: 0,
            frame_offset: 0,
            temporal_id: 0,
            spatial_id: 0,
            show_existing_frame: false,
            existing_frame_idx: 0,
            frame_id: 0,
            frame_presentation_delay: 0,
            show_frame: false,
            showable_frame: false,
            error_resilient_mode: false,
            disable_cdf_update: false,
            allow_screen_content_tools: false,
            force_integer_mv: false,
            frame_size_override: false,
            primary_ref_frame: 0,
            buffer_removal_time_present: false,
            operating_points: [FrameHeaderOperatingPoint::default(); MAX_OPERATING_POINTS],
            refresh_frame_flags: 0,
            render_width: 0,
            render_height: 0,
            super_res: SuperResParams::default(),
            have_render_size: false,
            allow_intrabc: false,
            frame_ref_short_signaling: false,
            refidx: [0; REFS_PER_FRAME],
            hp: false,
            subpel_filter_mode: FilterMode::default(),
            switchable_motion_mode: false,
            use_ref_frame_mvs: false,
            refresh_context: false,
            tiling: TilingInfo::default(),
            quant: QuantizationParams::default(),
            segmentation: SegmentationParams::default(),
            delta: DeltaParams::default(),
            all_lossless: false,
            loopfilter: LoopFilterParams::default(),
            cdef: CdefParams::default(),
            restoration: RestorationParams::default(),
            txfm_mode: TxfmMode::default(),
            switchable_comp_refs: false,
            skip_mode_allowed: false,
            skip_mode_enabled: false,
            skip_mode_refs: [0; 2],
            warp_motion: false,
            reduced_txtp_set: false,
            gmv: [WarpedMotionParams::default(); REFS_PER_FRAME],
        }
    }
}

/// Frame header ayrıştırma bağlamı.
#[derive(Clone, Copy)]
pub struct FrameHeaderContext<'a> {
    /// Etkin sequence header.
    pub sequence_header: &'a SequenceHeader,
    /// Başlık düzeyi referans yuvaları (dav1d `c->refs[i].p.p.frame_hdr`).
    pub refs: &'a [Option<FrameHeader>; NUM_REF_FRAMES],
    /// OBU uzantısından gelen temporal katman kimliği (yoksa 0).
    pub temporal_id: u8,
    /// OBU uzantısından gelen spatial katman kimliği (yoksa 0).
    pub spatial_id: u8,
    /// dav1d `strict_std_compliance` (varsayılan `false`).
    pub strict_std_compliance: bool,
}

fn invalid(reader: &BitReader<'_>, why: &'static str) -> Error {
    Error::Bitstream(BitstreamError::new(
        BitstreamErrorKind::InvalidFrameHeader(why),
        reader.bit_position(),
    ))
}

fn invalid_tile(bit_offset: usize, why: &'static str) -> Error {
    Error::Bitstream(BitstreamError::new(
        BitstreamErrorKind::InvalidTileGroup(why),
        bit_offset,
    ))
}

/// `get_poc_diff()` (dav1d `src/env.h` 237–244).
#[must_use]
pub fn get_poc_diff(order_hint_n_bits: u8, poc0: i32, poc1: i32) -> i32 {
    if order_hint_n_bits == 0 {
        return 0;
    }
    let mask = 1i32 << (order_hint_n_bits - 1);
    let diff = poc0 - poc1;
    (diff & (mask - 1)) - (diff & mask)
}

#[inline]
fn tile_log2(mut sz: i32, tgt: i32) -> u8 {
    let mut k = 0u8;
    while sz < tgt {
        sz = sz.saturating_mul(2);
        k = k.saturating_add(1);
        if k > 30 {
            break;
        }
    }
    k
}

#[inline]
fn iclip_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

impl FrameHeader {
    /// OBU yükünü (başlık + boyut alanı hariç) ayrıştırır.
    ///
    /// `trailing_bits()` **çağıranın** sorumluluğundadır: frame header
    /// OBU'sunda okunur, frame OBU'sunda okunmaz (dav1d `src/obu.c` 1290–1300).
    ///
    /// # Hatalar
    ///
    /// [`Error::Bitstream`]: kesik veri ya da dav1d'nin reddettiği bir alan
    /// değeri ([`BitstreamErrorKind::InvalidFrameHeader`]).
    pub fn parse(reader: &mut BitReader<'_>, ctx: &FrameHeaderContext<'_>) -> Result<Self> {
        let seq = ctx.sequence_header;
        let mut h = Self {
            temporal_id: ctx.temporal_id,
            spatial_id: ctx.spatial_id,
            ..Self::default()
        };

        if !seq.reduced_still_picture_header {
            h.show_existing_frame = reader.read_bit()? != 0;
        }
        if h.show_existing_frame {
            h.existing_frame_idx = reader.read_bits(3)? as u8;
            if seq.decoder_model_info_present && !seq.equal_picture_interval {
                h.frame_presentation_delay =
                    reader.read_bits(u32::from(seq.frame_presentation_delay_length))?;
            }
            if seq.frame_id_numbers_present {
                h.frame_id = reader.read_bits(u32::from(seq.frame_id_n_bits))?;
                let ref_hdr = ctx.refs[usize::from(h.existing_frame_idx)].as_ref();
                let Some(ref_hdr) = ref_hdr else {
                    return Err(invalid(reader, "show_existing_frame: referans yuva boş"));
                };
                if ref_hdr.frame_id != h.frame_id {
                    return Err(invalid(
                        reader,
                        "show_existing_frame: frame_id referansla uyuşmuyor",
                    ));
                }
            }
            return Ok(h);
        }

        if seq.reduced_still_picture_header {
            h.frame_type = FrameType::Key;
            h.show_frame = true;
        } else {
            h.frame_type = FrameType::from_raw(reader.read_bits(2)? as u8);
            h.show_frame = reader.read_bit()? != 0;
        }
        if h.show_frame {
            if seq.decoder_model_info_present && !seq.equal_picture_interval {
                h.frame_presentation_delay =
                    reader.read_bits(u32::from(seq.frame_presentation_delay_length))?;
            }
            h.showable_frame = h.frame_type != FrameType::Key;
        } else {
            h.showable_frame = reader.read_bit()? != 0;
        }
        h.error_resilient_mode = (h.frame_type == FrameType::Key && h.show_frame)
            || h.frame_type == FrameType::Switch
            || seq.reduced_still_picture_header
            || reader.read_bit()? != 0;

        h.disable_cdf_update = reader.read_bit()? != 0;
        h.allow_screen_content_tools = match seq.screen_content_tools {
            AdaptiveBoolean::Adaptive => reader.read_bit()? != 0,
            AdaptiveBoolean::On => true,
            AdaptiveBoolean::Off => false,
        };
        if h.allow_screen_content_tools {
            h.force_integer_mv = match seq.force_integer_mv {
                AdaptiveBoolean::Adaptive => reader.read_bit()? != 0,
                AdaptiveBoolean::On => true,
                AdaptiveBoolean::Off => false,
            };
        }
        if h.frame_type.is_key_or_intra() {
            h.force_integer_mv = true;
        }

        if seq.frame_id_numbers_present {
            h.frame_id = reader.read_bits(u32::from(seq.frame_id_n_bits))?;
        }

        if !seq.reduced_still_picture_header {
            h.frame_size_override = h.frame_type == FrameType::Switch || reader.read_bit()? != 0;
        }
        if seq.order_hint {
            h.frame_offset = reader.read_bits(u32::from(seq.order_hint_n_bits))? as u8;
        }
        h.primary_ref_frame = if !h.error_resilient_mode && h.frame_type.is_inter_or_switch() {
            reader.read_bits(3)? as u8
        } else {
            PRIMARY_REF_NONE
        };

        if seq.decoder_model_info_present {
            h.buffer_removal_time_present = reader.read_bit()? != 0;
            if h.buffer_removal_time_present {
                for i in 0..usize::from(seq.num_operating_points) {
                    let seq_op = &seq.operating_points[i];
                    if !seq_op.decoder_model_param_present {
                        continue;
                    }
                    let in_temporal_layer = (seq_op.idc >> h.temporal_id) & 1 != 0;
                    let in_spatial_layer = (seq_op.idc >> (h.spatial_id + 8)) & 1 != 0;
                    if seq_op.idc == 0 || (in_temporal_layer && in_spatial_layer) {
                        h.operating_points[i].buffer_removal_time =
                            reader.read_bits(u32::from(seq.buffer_removal_delay_length))?;
                    }
                }
            }
        }

        if h.frame_type.is_key_or_intra() {
            h.refresh_frame_flags = if h.frame_type == FrameType::Key && h.show_frame {
                0xff
            } else {
                reader.read_bits(8)? as u8
            };
            if h.refresh_frame_flags != 0xff && h.error_resilient_mode && seq.order_hint {
                for _ in 0..NUM_REF_FRAMES {
                    let _ = reader.read_bits(u32::from(seq.order_hint_n_bits))?;
                }
            }
            if ctx.strict_std_compliance
                && h.frame_type == FrameType::Intra
                && h.refresh_frame_flags == 0xff
            {
                return Err(invalid(
                    reader,
                    "katı kip: INTRA karede refresh_frame_flags == 0xff (obu.c 501-505)",
                ));
            }
            read_frame_size(reader, ctx, &mut h, false)?;
            if h.allow_screen_content_tools && !h.super_res.enabled {
                h.allow_intrabc = reader.read_bit()? != 0;
            }
        } else {
            h.refresh_frame_flags = if h.frame_type == FrameType::Switch {
                0xff
            } else {
                reader.read_bits(8)? as u8
            };
            if h.error_resilient_mode && seq.order_hint {
                for _ in 0..NUM_REF_FRAMES {
                    let _ = reader.read_bits(u32::from(seq.order_hint_n_bits))?;
                }
            }
            if seq.order_hint {
                h.frame_ref_short_signaling = reader.read_bit()? != 0;
                if h.frame_ref_short_signaling {
                    read_frame_refs(reader, ctx, &mut h)?;
                }
            }
            for i in 0..REFS_PER_FRAME {
                if !h.frame_ref_short_signaling {
                    h.refidx[i] = reader.read_bits(3)? as i8;
                }
                if seq.frame_id_numbers_present {
                    let delta_ref_frame_id =
                        reader.read_bits(u32::from(seq.delta_frame_id_n_bits))? + 1;
                    let frame_id_bits = u32::from(seq.frame_id_n_bits);
                    let mask = (1u32 << frame_id_bits) - 1;
                    let ref_frame_id = h
                        .frame_id
                        .wrapping_add(1u32 << frame_id_bits)
                        .wrapping_sub(delta_ref_frame_id)
                        & mask;
                    let slot = usize::try_from(h.refidx[i].max(0)).unwrap_or(0);
                    let ref_hdr = ctx.refs[slot].as_ref();
                    let Some(ref_hdr) = ref_hdr else {
                        return Err(invalid(reader, "delta_frame_id: referans yuva boş"));
                    };
                    if ref_hdr.frame_id != ref_frame_id {
                        return Err(invalid(
                            reader,
                            "delta_frame_id: hesaplanan frame_id referansla uyuşmuyor",
                        ));
                    }
                }
            }
            let use_ref = !h.error_resilient_mode && h.frame_size_override;
            read_frame_size(reader, ctx, &mut h, use_ref)?;
            if !h.force_integer_mv {
                h.hp = reader.read_bit()? != 0;
            }
            h.subpel_filter_mode = if reader.read_bit()? != 0 {
                FilterMode::Switchable
            } else {
                FilterMode::from_raw(reader.read_bits(2)? as u8)
            };
            h.switchable_motion_mode = reader.read_bit()? != 0;
            if !h.error_resilient_mode
                && seq.ref_frame_mvs
                && seq.order_hint
                && h.frame_type.is_inter_or_switch()
            {
                h.use_ref_frame_mvs = reader.read_bit()? != 0;
            }
        }

        if !seq.reduced_still_picture_header && !h.disable_cdf_update {
            h.refresh_context = reader.read_bit()? == 0;
        }

        read_tile_info(reader, seq, &mut h)?;
        read_quantization(reader, seq, &mut h)?;
        read_segmentation(reader, ctx, &mut h)?;
        read_delta_q_lf(reader, &mut h)?;
        derive_lossless(&mut h);
        read_loop_filter(reader, ctx, seq, &mut h)?;
        read_cdef(reader, seq, &mut h)?;
        read_restoration(reader, seq, &mut h)?;
        if !h.all_lossless {
            h.txfm_mode = if reader.read_bit()? != 0 {
                TxfmMode::Switchable
            } else {
                TxfmMode::Largest
            };
        }
        if h.frame_type.is_inter_or_switch() {
            h.switchable_comp_refs = reader.read_bit()? != 0;
        }
        read_skip_mode(reader, ctx, seq, &mut h)?;
        if !h.error_resilient_mode && h.frame_type.is_inter_or_switch() && seq.warped_motion {
            h.warp_motion = reader.read_bit()? != 0;
        }
        h.reduced_txtp_set = reader.read_bit()? != 0;

        h.gmv = [DEFAULT_WM_PARAMS; REFS_PER_FRAME];
        if h.frame_type.is_inter_or_switch() {
            read_global_motion(reader, ctx, &mut h)?;
        }
        read_film_grain(reader, ctx, seq, &mut h)?;
        Ok(h)
    }

    /// Bir referans yuvasının kare türü `KEY` mi? (`show_existing_frame`
    /// anahtar kare yenilemesi için).
    #[must_use]
    pub const fn is_key(&self) -> bool {
        matches!(self.frame_type, FrameType::Key)
    }
}

/// `read_frame_size()` (dav1d `src/obu.c` 341–402): `frame_size()`,
/// `render_size()` ve `frame_size_with_refs()`.
fn read_frame_size(
    reader: &mut BitReader<'_>,
    ctx: &FrameHeaderContext<'_>,
    h: &mut FrameHeader,
    use_ref: bool,
) -> Result<()> {
    let seq = ctx.sequence_header;
    if use_ref {
        for i in 0..REFS_PER_FRAME {
            if reader.read_bit()? != 0 {
                let slot = usize::try_from(h.refidx[i].max(0)).unwrap_or(0);
                let Some(ref_hdr) = ctx.refs[slot].as_ref() else {
                    return Err(invalid(reader, "frame_size_with_refs: referans yuva boş"));
                };
                h.width[1] = ref_hdr.width[1];
                h.height = ref_hdr.height;
                if h.width[1] > seq.max_width || h.height > seq.max_height {
                    return Err(invalid(
                        reader,
                        "frame_size_with_refs: kare boyutu sequence max_width/max_height sınırını aşıyor",
                    ));
                }
                h.render_width = ref_hdr.render_width;
                h.render_height = ref_hdr.render_height;
                h.super_res.enabled = seq.super_res && reader.read_bit()? != 0;
                if h.super_res.enabled {
                    let d = 9 + reader.read_bits(3)? as u8;
                    h.super_res.width_scale_denominator = d;
                    h.width[0] = ((h.width[1] * 8 + i32::from(d >> 1)) / i32::from(d))
                        .max(h.width[1].min(16));
                } else {
                    h.super_res.width_scale_denominator = 8;
                    h.width[0] = h.width[1];
                }
                return Ok(());
            }
        }
    }

    if h.frame_size_override {
        h.width[1] = reader.read_bits(u32::from(seq.width_n_bits))? as i32 + 1;
        h.height = reader.read_bits(u32::from(seq.height_n_bits))? as i32 + 1;
        if h.width[1] > seq.max_width || h.height > seq.max_height {
            return Err(invalid(
                reader,
                "frame_size_override: kare boyutu sequence max_width/max_height sınırını aşıyor",
            ));
        }
    } else {
        h.width[1] = seq.max_width;
        h.height = seq.max_height;
    }
    h.super_res.enabled = seq.super_res && reader.read_bit()? != 0;
    if h.super_res.enabled {
        let d = 9 + reader.read_bits(3)? as u8;
        h.super_res.width_scale_denominator = d;
        h.width[0] = ((h.width[1] * 8 + i32::from(d >> 1)) / i32::from(d)).max(h.width[1].min(16));
    } else {
        h.super_res.width_scale_denominator = 8;
        h.width[0] = h.width[1];
    }
    h.have_render_size = reader.read_bit()? != 0;
    if h.have_render_size {
        h.render_width = reader.read_bits(16)? as i32 + 1;
        h.render_height = reader.read_bits(16)? as i32 + 1;
    } else {
        h.render_width = h.width[1];
        h.render_height = h.height;
    }
    Ok(())
}

/// AV1 §5.9.2 `frame_refs_short_signaling` referans seçim algoritması
/// (dav1d `src/obu.c` 516–583).
fn read_frame_refs(
    reader: &mut BitReader<'_>,
    ctx: &FrameHeaderContext<'_>,
    h: &mut FrameHeader,
) -> Result<()> {
    let seq = ctx.sequence_header;
    h.refidx[0] = reader.read_bits(3)? as i8;
    h.refidx[1] = -1;
    h.refidx[2] = -1;
    h.refidx[3] = reader.read_bits(3)? as i8;

    // dav1d: `int frame_offset_mem[8 + 1]; int *const frame_offset = &mem[1];`
    // — `refidx == -1` durumunda yazılan `INT_MIN` değeri indeks 0'a düşer.
    let mut frame_offset_mem = [0i32; NUM_REF_FRAMES + 1];
    let mut earliest_ref: i32 = -1;
    let mut earliest_offset = i32::MAX;
    for i in 0..NUM_REF_FRAMES {
        let Some(ref_hdr) = ctx.refs[i].as_ref() else {
            return Err(invalid(
                reader,
                "frame_refs_short_signaling: referans yuva boş",
            ));
        };
        let diff = get_poc_diff(
            seq.order_hint_n_bits,
            i32::from(ref_hdr.frame_offset),
            i32::from(h.frame_offset),
        );
        frame_offset_mem[i + 1] = diff;
        if diff < earliest_offset {
            earliest_offset = diff;
            earliest_ref = i as i32;
        }
    }
    let fo = |index: i32| -> usize { usize::try_from(index + 1).unwrap_or(0) };
    frame_offset_mem[fo(i32::from(h.refidx[0]))] = i32::MIN;
    frame_offset_mem[fo(i32::from(h.refidx[3]))] = i32::MIN;

    let mut refidx: i32 = -1;
    let mut latest_offset: i32 = 0;
    for i in 0..NUM_REF_FRAMES {
        let hint = frame_offset_mem[i + 1];
        if hint >= latest_offset {
            latest_offset = hint;
            refidx = i as i32;
        }
    }
    frame_offset_mem[fo(refidx)] = i32::MIN;
    h.refidx[6] = refidx as i8;

    for refidx_slot in &mut h.refidx[4..6] {
        // dav1d: `unsigned earliest_offset = UINT8_MAX;` ve `unsigned hint`
        // karşılaştırması; negatif POC farkları büyük işaretsiz değer olur.
        let mut earliest_offset_u: u32 = u32::from(u8::MAX);
        let mut picked: i32 = -1;
        for j in 0..NUM_REF_FRAMES {
            let hint = frame_offset_mem[j + 1] as u32;
            if hint < earliest_offset_u {
                earliest_offset_u = hint;
                picked = j as i32;
            }
        }
        frame_offset_mem[fo(picked)] = i32::MIN;
        *refidx_slot = picked as i8;
    }

    for refidx_slot in &mut h.refidx[1..REFS_PER_FRAME] {
        let mut picked = i32::from(*refidx_slot);
        if picked < 0 {
            let mut latest_offset_u: u32 = !u32::from(u8::MAX);
            picked = -1;
            for j in 0..NUM_REF_FRAMES {
                let hint = frame_offset_mem[j + 1] as u32;
                if hint >= latest_offset_u {
                    latest_offset_u = hint;
                    picked = j as i32;
                }
            }
            frame_offset_mem[fo(picked)] = i32::MIN;
            *refidx_slot = if picked >= 0 {
                picked as i8
            } else {
                earliest_ref.max(0) as i8
            };
        }
    }
    Ok(())
}

/// `tile_info()` (dav1d `src/obu.c` 606–677).
fn read_tile_info(
    reader: &mut BitReader<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
) -> Result<()> {
    h.tiling.uniform = reader.read_bit()? != 0;
    let sbsz_min1: i32 = (64i32 << u8::from(seq.sb128)) - 1;
    let sbsz_log2 = 6 + u32::from(seq.sb128);
    let sbw = (h.width[0] + sbsz_min1) >> sbsz_log2;
    let sbh = (h.height + sbsz_min1) >> sbsz_log2;
    let max_tile_width_sb = 4096i32 >> sbsz_log2;
    let max_tile_area_sb_base = (4096i32 * 2304) >> (2 * sbsz_log2);
    h.tiling.min_log2_cols = tile_log2(max_tile_width_sb, sbw);
    h.tiling.max_log2_cols = tile_log2(1, sbw.min(MAX_TILE_COLS as i32));
    h.tiling.max_log2_rows = tile_log2(1, sbh.min(MAX_TILE_ROWS as i32));
    let min_log2_tiles = tile_log2(max_tile_area_sb_base, sbw * sbh).max(h.tiling.min_log2_cols);

    if h.tiling.uniform {
        h.tiling.log2_cols = h.tiling.min_log2_cols;
        while h.tiling.log2_cols < h.tiling.max_log2_cols && reader.read_bit()? != 0 {
            h.tiling.log2_cols += 1;
        }
        let tile_w = 1 + ((sbw - 1) >> h.tiling.log2_cols);
        h.tiling.cols = 0;
        let mut sbx = 0;
        while sbx < sbw {
            if usize::from(h.tiling.cols) >= MAX_TILE_COLS {
                return Err(invalid(
                    reader,
                    "tile_info: TileCols DAV1D_MAX_TILE_COLS aştı",
                ));
            }
            h.tiling.col_start_sb[usize::from(h.tiling.cols)] = sbx as u16;
            sbx += tile_w;
            h.tiling.cols += 1;
        }
        h.tiling.min_log2_rows =
            (i32::from(min_log2_tiles) - i32::from(h.tiling.log2_cols)).max(0) as u8;

        h.tiling.log2_rows = h.tiling.min_log2_rows;
        while h.tiling.log2_rows < h.tiling.max_log2_rows && reader.read_bit()? != 0 {
            h.tiling.log2_rows += 1;
        }
        let tile_h = 1 + ((sbh - 1) >> h.tiling.log2_rows);
        h.tiling.rows = 0;
        let mut sby = 0;
        while sby < sbh {
            if usize::from(h.tiling.rows) >= MAX_TILE_ROWS {
                return Err(invalid(
                    reader,
                    "tile_info: TileRows DAV1D_MAX_TILE_ROWS aştı",
                ));
            }
            h.tiling.row_start_sb[usize::from(h.tiling.rows)] = sby as u16;
            sby += tile_h;
            h.tiling.rows += 1;
        }
    } else {
        h.tiling.cols = 0;
        let mut widest_tile = 0i32;
        let mut max_tile_area_sb = sbw * sbh;
        let mut sbx = 0;
        while sbx < sbw && usize::from(h.tiling.cols) < MAX_TILE_COLS {
            let tile_width_sb = (sbw - sbx).min(max_tile_width_sb);
            let tile_w = if tile_width_sb > 1 {
                1 + reader.read_uniform(tile_width_sb as u32)? as i32
            } else {
                1
            };
            h.tiling.col_start_sb[usize::from(h.tiling.cols)] = sbx as u16;
            sbx += tile_w;
            widest_tile = widest_tile.max(tile_w);
            h.tiling.cols += 1;
        }
        h.tiling.log2_cols = tile_log2(1, i32::from(h.tiling.cols));
        if min_log2_tiles != 0 {
            max_tile_area_sb >>= min_log2_tiles + 1;
        }
        let max_tile_height_sb = (max_tile_area_sb / widest_tile.max(1)).max(1);

        h.tiling.rows = 0;
        let mut sby = 0;
        while sby < sbh && usize::from(h.tiling.rows) < MAX_TILE_ROWS {
            let tile_height_sb = (sbh - sby).min(max_tile_height_sb);
            let tile_h = if tile_height_sb > 1 {
                1 + reader.read_uniform(tile_height_sb as u32)? as i32
            } else {
                1
            };
            h.tiling.row_start_sb[usize::from(h.tiling.rows)] = sby as u16;
            sby += tile_h;
            h.tiling.rows += 1;
        }
        h.tiling.log2_rows = tile_log2(1, i32::from(h.tiling.rows));
    }
    h.tiling.col_start_sb[usize::from(h.tiling.cols)] = sbw as u16;
    h.tiling.row_start_sb[usize::from(h.tiling.rows)] = sbh as u16;
    if h.tiling.log2_cols != 0 || h.tiling.log2_rows != 0 {
        let n_bits = u32::from(h.tiling.log2_cols) + u32::from(h.tiling.log2_rows);
        h.tiling.update = reader.read_bits(n_bits)? as u16;
        if u32::from(h.tiling.update) >= u32::from(h.tiling.cols) * u32::from(h.tiling.rows) {
            return Err(invalid(
                reader,
                "tile_info: context_update_tile_id TileCols*TileRows sınırını aştı",
            ));
        }
        h.tiling.n_bytes = reader.read_bits(2)? as u8 + 1;
    }
    Ok(())
}

/// `quantization_params()` (dav1d `src/obu.c` 680–718).
fn read_quantization(
    reader: &mut BitReader<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
) -> Result<()> {
    h.quant.yac = reader.read_bits(8)? as u8;
    if reader.read_bit()? != 0 {
        h.quant.ydc_delta = reader.read_sbits(7)? as i8;
    }
    if !seq.monochrome {
        let diff_uv_delta = seq.separate_uv_delta_q && reader.read_bit()? != 0;
        if reader.read_bit()? != 0 {
            h.quant.udc_delta = reader.read_sbits(7)? as i8;
        }
        if reader.read_bit()? != 0 {
            h.quant.uac_delta = reader.read_sbits(7)? as i8;
        }
        if diff_uv_delta {
            if reader.read_bit()? != 0 {
                h.quant.vdc_delta = reader.read_sbits(7)? as i8;
            }
            if reader.read_bit()? != 0 {
                h.quant.vac_delta = reader.read_sbits(7)? as i8;
            }
        } else {
            h.quant.vdc_delta = h.quant.udc_delta;
            h.quant.vac_delta = h.quant.uac_delta;
        }
    }
    h.quant.qm = reader.read_bit()? != 0;
    if h.quant.qm {
        h.quant.qm_y = reader.read_bits(4)? as u8;
        h.quant.qm_u = reader.read_bits(4)? as u8;
        h.quant.qm_v = if seq.separate_uv_delta_q {
            reader.read_bits(4)? as u8
        } else {
            h.quant.qm_u
        };
    }
    Ok(())
}

/// `segmentation_params()` (dav1d `src/obu.c` 721–786).
fn read_segmentation(
    reader: &mut BitReader<'_>,
    ctx: &FrameHeaderContext<'_>,
    h: &mut FrameHeader,
) -> Result<()> {
    h.segmentation.enabled = reader.read_bit()? != 0;
    if h.segmentation.enabled {
        if h.primary_ref_frame == PRIMARY_REF_NONE {
            h.segmentation.update_map = true;
            h.segmentation.update_data = true;
        } else {
            h.segmentation.update_map = reader.read_bit()? != 0;
            if h.segmentation.update_map {
                h.segmentation.temporal = reader.read_bit()? != 0;
            }
            h.segmentation.update_data = reader.read_bit()? != 0;
        }

        if h.segmentation.update_data {
            h.segmentation.seg_data.last_active_segid = -1;
            for i in 0..MAX_SEGMENTS {
                let seg = &mut h.segmentation.seg_data.d[i];
                if reader.read_bit()? != 0 {
                    seg.delta_q = reader.read_sbits(9)? as i16;
                    h.segmentation.seg_data.last_active_segid = i as i8;
                }
                if reader.read_bit()? != 0 {
                    seg.delta_lf_y_v = reader.read_sbits(7)? as i8;
                    h.segmentation.seg_data.last_active_segid = i as i8;
                }
                if reader.read_bit()? != 0 {
                    seg.delta_lf_y_h = reader.read_sbits(7)? as i8;
                    h.segmentation.seg_data.last_active_segid = i as i8;
                }
                if reader.read_bit()? != 0 {
                    seg.delta_lf_u = reader.read_sbits(7)? as i8;
                    h.segmentation.seg_data.last_active_segid = i as i8;
                }
                if reader.read_bit()? != 0 {
                    seg.delta_lf_v = reader.read_sbits(7)? as i8;
                    h.segmentation.seg_data.last_active_segid = i as i8;
                }
                if reader.read_bit()? != 0 {
                    seg.reference = reader.read_bits(3)? as i8;
                    h.segmentation.seg_data.last_active_segid = i as i8;
                    h.segmentation.seg_data.preskip = true;
                } else {
                    seg.reference = -1;
                }
                seg.skip = reader.read_bit()? != 0;
                if seg.skip {
                    h.segmentation.seg_data.last_active_segid = i as i8;
                    h.segmentation.seg_data.preskip = true;
                }
                seg.globalmv = reader.read_bit()? != 0;
                if seg.globalmv {
                    h.segmentation.seg_data.last_active_segid = i as i8;
                    h.segmentation.seg_data.preskip = true;
                }
            }
        } else {
            let primary = usize::from(h.primary_ref_frame.min(REFS_PER_FRAME as u8 - 1));
            let slot = usize::try_from(h.refidx[primary].max(0)).unwrap_or(0);
            let Some(ref_hdr) = ctx.refs[slot].as_ref() else {
                return Err(invalid(
                    reader,
                    "segmentation: primary_ref_frame referansı boş",
                ));
            };
            h.segmentation.seg_data = ref_hdr.segmentation.seg_data;
        }
    } else {
        for seg in &mut h.segmentation.seg_data.d {
            seg.reference = -1;
        }
    }
    Ok(())
}

/// `delta_q_params()` + `delta_lf_params()` (dav1d `src/obu.c` 789–803).
fn read_delta_q_lf(reader: &mut BitReader<'_>, h: &mut FrameHeader) -> Result<()> {
    if h.quant.yac != 0 {
        h.delta.q_present = reader.read_bit()? != 0;
        if h.delta.q_present {
            h.delta.q_res_log2 = reader.read_bits(2)? as u8;
            if !h.allow_intrabc {
                h.delta.lf_present = reader.read_bit()? != 0;
                if h.delta.lf_present {
                    h.delta.lf_res_log2 = reader.read_bits(2)? as u8;
                    h.delta.lf_multi = reader.read_bit()? != 0;
                }
            }
        }
    }
    Ok(())
}

/// Lossless/qindex türetimi (dav1d `src/obu.c` 806–816).
fn derive_lossless(h: &mut FrameHeader) {
    let delta_lossless = h.quant.ydc_delta == 0
        && h.quant.udc_delta == 0
        && h.quant.uac_delta == 0
        && h.quant.vdc_delta == 0
        && h.quant.vac_delta == 0;
    h.all_lossless = true;
    for i in 0..MAX_SEGMENTS {
        h.segmentation.qidx[i] = if h.segmentation.enabled {
            iclip_u8(i32::from(h.quant.yac) + i32::from(h.segmentation.seg_data.d[i].delta_q))
        } else {
            h.quant.yac
        };
        h.segmentation.lossless[i] = h.segmentation.qidx[i] == 0 && delta_lossless;
        h.all_lossless &= h.segmentation.lossless[i];
    }
}

/// `loop_filter_params()` (dav1d `src/obu.c` 819–853).
fn read_loop_filter(
    reader: &mut BitReader<'_>,
    ctx: &FrameHeaderContext<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
) -> Result<()> {
    if h.all_lossless || h.allow_intrabc {
        h.loopfilter.mode_ref_delta_enabled = true;
        h.loopfilter.mode_ref_delta_update = true;
        h.loopfilter.mode_ref_deltas = DEFAULT_MODE_REF_DELTAS;
        return Ok(());
    }
    h.loopfilter.level_y[0] = reader.read_bits(6)? as u8;
    h.loopfilter.level_y[1] = reader.read_bits(6)? as u8;
    if !seq.monochrome && (h.loopfilter.level_y[0] != 0 || h.loopfilter.level_y[1] != 0) {
        h.loopfilter.level_u = reader.read_bits(6)? as u8;
        h.loopfilter.level_v = reader.read_bits(6)? as u8;
    }
    h.loopfilter.sharpness = reader.read_bits(3)? as u8;

    if h.primary_ref_frame == PRIMARY_REF_NONE {
        h.loopfilter.mode_ref_deltas = DEFAULT_MODE_REF_DELTAS;
    } else {
        let primary = usize::from(h.primary_ref_frame.min(REFS_PER_FRAME as u8 - 1));
        let slot = usize::try_from(h.refidx[primary].max(0)).unwrap_or(0);
        let Some(ref_hdr) = ctx.refs[slot].as_ref() else {
            return Err(invalid(
                reader,
                "loop_filter: primary_ref_frame referansı boş",
            ));
        };
        h.loopfilter.mode_ref_deltas = ref_hdr.loopfilter.mode_ref_deltas;
    }
    h.loopfilter.mode_ref_delta_enabled = reader.read_bit()? != 0;
    if h.loopfilter.mode_ref_delta_enabled {
        h.loopfilter.mode_ref_delta_update = reader.read_bit()? != 0;
        if h.loopfilter.mode_ref_delta_update {
            for ref_delta in &mut h.loopfilter.mode_ref_deltas.ref_delta[..TOTAL_REFS_PER_FRAME] {
                if reader.read_bit()? != 0 {
                    *ref_delta = reader.read_sbits(7)? as i8;
                }
            }
            for mode_delta in &mut h.loopfilter.mode_ref_deltas.mode_delta {
                if reader.read_bit()? != 0 {
                    *mode_delta = reader.read_sbits(7)? as i8;
                }
            }
        }
    }
    Ok(())
}

/// `cdef_params()` (dav1d `src/obu.c` 856–868).
fn read_cdef(reader: &mut BitReader<'_>, seq: &SequenceHeader, h: &mut FrameHeader) -> Result<()> {
    if h.all_lossless || !seq.cdef || h.allow_intrabc {
        return Ok(());
    }
    h.cdef.damping = reader.read_bits(2)? as u8 + 3;
    h.cdef.n_bits = reader.read_bits(2)? as u8;
    for i in 0..(1usize << h.cdef.n_bits) {
        h.cdef.y_strength[i] = reader.read_bits(6)? as u8;
        if !seq.monochrome {
            h.cdef.uv_strength[i] = reader.read_bits(6)? as u8;
        }
    }
    Ok(())
}

/// `lr_params()` (dav1d `src/obu.c` 871–900).
fn read_restoration(
    reader: &mut BitReader<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
) -> Result<()> {
    if (!h.all_lossless || h.super_res.enabled) && seq.restoration && !h.allow_intrabc {
        h.restoration.kind[0] = RestorationType::from_raw(reader.read_bits(2)? as u8);
        if !seq.monochrome {
            h.restoration.kind[1] = RestorationType::from_raw(reader.read_bits(2)? as u8);
            h.restoration.kind[2] = RestorationType::from_raw(reader.read_bits(2)? as u8);
        }
        if h.restoration.kind.iter().any(|t| t.is_present()) {
            h.restoration.unit_size[0] = 6 + u8::from(seq.sb128);
            if reader.read_bit()? != 0 {
                h.restoration.unit_size[0] += 1;
                if !seq.sb128 {
                    h.restoration.unit_size[0] += reader.read_bit()? as u8;
                }
            }
            h.restoration.unit_size[1] = h.restoration.unit_size[0];
            if (h.restoration.kind[1].is_present() || h.restoration.kind[2].is_present())
                && seq.ss_hor == 1
                && seq.ss_ver == 1
            {
                h.restoration.unit_size[1] -= reader.read_bit()? as u8;
            }
        } else {
            h.restoration.unit_size[0] = 8;
        }
    }
    Ok(())
}

/// `frame_reference_mode()` + `skip_mode_params()` (dav1d `src/obu.c` 907–998).
fn read_skip_mode(
    reader: &mut BitReader<'_>,
    ctx: &FrameHeaderContext<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
) -> Result<()> {
    if !(h.switchable_comp_refs && h.frame_type.is_inter_or_switch() && seq.order_hint) {
        if h.skip_mode_allowed {
            h.skip_mode_enabled = reader.read_bit()? != 0;
        }
        return Ok(());
    }
    let poc = i32::from(h.frame_offset);
    let mut off_before: i32 = -1;
    let mut off_after: i32 = -1;
    let mut off_before_idx: usize = 0;
    let mut off_after_idx: usize = 0;
    for (i, &refidx_val) in h.refidx[..REFS_PER_FRAME].iter().enumerate() {
        let slot = usize::try_from(refidx_val.max(0)).unwrap_or(0);
        let Some(ref_hdr) = ctx.refs[slot].as_ref() else {
            return Err(invalid(reader, "skip_mode: referans yuva boş"));
        };
        let refpoc = i32::from(ref_hdr.frame_offset);
        let diff = get_poc_diff(seq.order_hint_n_bits, refpoc, poc);
        if diff > 0 {
            if off_after < 0 || get_poc_diff(seq.order_hint_n_bits, off_after, refpoc) > 0 {
                off_after = refpoc;
                off_after_idx = i;
            }
        } else if diff < 0
            && (off_before < 0 || get_poc_diff(seq.order_hint_n_bits, refpoc, off_before) > 0)
        {
            off_before = refpoc;
            off_before_idx = i;
        }
    }

    if off_before >= 0 && off_after >= 0 {
        h.skip_mode_refs[0] = off_before_idx.min(off_after_idx) as i8;
        h.skip_mode_refs[1] = off_before_idx.max(off_after_idx) as i8;
        h.skip_mode_allowed = true;
    } else if off_before >= 0 {
        let mut off_before2: i32 = -1;
        let mut off_before2_idx: usize = 0;
        for (i, &refidx_val) in h.refidx[..REFS_PER_FRAME].iter().enumerate() {
            let slot = usize::try_from(refidx_val.max(0)).unwrap_or(0);
            let Some(ref_hdr) = ctx.refs[slot].as_ref() else {
                return Err(invalid(reader, "skip_mode: referans yuva boş"));
            };
            let refpoc = i32::from(ref_hdr.frame_offset);
            if get_poc_diff(seq.order_hint_n_bits, refpoc, off_before) < 0
                && (off_before2 < 0 || get_poc_diff(seq.order_hint_n_bits, refpoc, off_before2) > 0)
            {
                off_before2 = refpoc;
                off_before2_idx = i;
            }
        }
        if off_before2 >= 0 {
            h.skip_mode_refs[0] = off_before_idx.min(off_before2_idx) as i8;
            h.skip_mode_refs[1] = off_before_idx.max(off_before2_idx) as i8;
            h.skip_mode_allowed = true;
        }
    }
    if h.skip_mode_allowed {
        h.skip_mode_enabled = reader.read_bit()? != 0;
    }
    Ok(())
}

/// `global_motion_params()` (dav1d `src/obu.c` 1010–1066).
fn read_global_motion(
    reader: &mut BitReader<'_>,
    ctx: &FrameHeaderContext<'_>,
    h: &mut FrameHeader,
) -> Result<()> {
    for i in 0..REFS_PER_FRAME {
        let kind = if reader.read_bit()? == 0 {
            WarpedMotionType::Identity
        } else if reader.read_bit()? != 0 {
            WarpedMotionType::RotZoom
        } else if reader.read_bit()? != 0 {
            WarpedMotionType::Translation
        } else {
            WarpedMotionType::Affine
        };
        h.gmv[i].kind = kind;
        if kind == WarpedMotionType::Identity {
            continue;
        }

        let ref_gmv = if h.primary_ref_frame == PRIMARY_REF_NONE {
            DEFAULT_WM_PARAMS
        } else {
            let primary = usize::from(h.primary_ref_frame.min(REFS_PER_FRAME as u8 - 1));
            let slot = usize::try_from(h.refidx[primary].max(0)).unwrap_or(0);
            let Some(ref_hdr) = ctx.refs[slot].as_ref() else {
                return Err(invalid(
                    reader,
                    "global_motion: primary_ref_frame referansı boş",
                ));
            };
            ref_hdr.gmv[i]
        };
        let ref_mat = ref_gmv.matrix;
        let (bits, shift) = if kind.raw() >= WarpedMotionType::RotZoom.raw() {
            let m2 = (1 << 16) + 2 * reader.read_subexp((ref_mat[2] - (1 << 16)) >> 1, 12)?;
            h.gmv[i].matrix[2] = m2;
            h.gmv[i].matrix[3] = 2 * reader.read_subexp(ref_mat[3] >> 1, 12)?;
            (12u32, 10u32)
        } else {
            // dav1d: `bits = 9 - !hdr->hp; shift = 13 + !hdr->hp;`
            let not_hp = u32::from(!h.hp);
            (9 - not_hp, 13 + not_hp)
        };

        if kind == WarpedMotionType::Affine {
            h.gmv[i].matrix[4] = 2 * reader.read_subexp(ref_mat[4] >> 1, 12)?;
            h.gmv[i].matrix[5] =
                (1 << 16) + 2 * reader.read_subexp((ref_mat[5] - (1 << 16)) >> 1, 12)?;
        } else {
            h.gmv[i].matrix[4] = -h.gmv[i].matrix[3];
            h.gmv[i].matrix[5] = h.gmv[i].matrix[2];
        }
        h.gmv[i].matrix[0] = reader.read_subexp(ref_mat[0] >> shift, bits)? * (1 << shift);
        h.gmv[i].matrix[1] = reader.read_subexp(ref_mat[1] >> shift, bits)? * (1 << shift);
    }
    Ok(())
}

/// `film_grain_params()` (dav1d `src/obu.c` 1068–1159).
fn read_film_grain(
    reader: &mut BitReader<'_>,
    ctx: &FrameHeaderContext<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
) -> Result<()> {
    if !(seq.film_grain_present && (h.show_frame || h.showable_frame)) {
        return Ok(());
    }
    h.film_grain.present = reader.read_bit()? != 0;
    if !h.film_grain.present {
        return Ok(());
    }
    let seed = reader.read_bits(16)?;
    h.film_grain.update = h.frame_type != FrameType::Inter || reader.read_bit()? != 0;
    if !h.film_grain.update {
        let refidx = reader.read_bits(3)? as usize;
        let found = h.refidx[..REFS_PER_FRAME]
            .iter()
            .any(|&idx| usize::try_from(idx).unwrap_or(usize::MAX) == refidx);
        if !found {
            return Err(invalid(
                reader,
                "film_grain: load refidx refidx[] içindeki 7 referanstan biri değil",
            ));
        }
        let Some(ref_hdr) = ctx.refs[refidx].as_ref() else {
            return Err(invalid(reader, "film_grain: load referans yuva boş"));
        };
        h.film_grain.data = ref_hdr.film_grain.data;
        h.film_grain.data.seed = seed;
        return Ok(());
    }

    let fgd = &mut h.film_grain.data;
    fgd.seed = seed;
    fgd.num_y_points = reader.read_bits(4)? as i32;
    if fgd.num_y_points > 14 {
        return Err(invalid(reader, "film_grain: num_y_points > 14"));
    }
    for i in 0..fgd.num_y_points as usize {
        fgd.y_points[i][0] = reader.read_bits(8)? as u8;
        if i != 0 && fgd.y_points[i - 1][0] >= fgd.y_points[i][0] {
            return Err(invalid(reader, "film_grain: y_points artan değil"));
        }
        fgd.y_points[i][1] = reader.read_bits(8)? as u8;
    }

    if !seq.monochrome {
        fgd.chroma_scaling_from_luma = reader.read_bit()? != 0;
    }
    if seq.monochrome
        || fgd.chroma_scaling_from_luma
        || (seq.ss_ver == 1 && seq.ss_hor == 1 && fgd.num_y_points == 0)
    {
        fgd.num_uv_points = [0, 0];
    } else {
        for plane in 0..2 {
            fgd.num_uv_points[plane] = reader.read_bits(4)? as i32;
            if fgd.num_uv_points[plane] > 10 {
                return Err(invalid(reader, "film_grain: num_uv_points > 10"));
            }
            for i in 0..fgd.num_uv_points[plane] as usize {
                fgd.uv_points[plane][i][0] = reader.read_bits(8)? as u8;
                if i != 0 && fgd.uv_points[plane][i - 1][0] >= fgd.uv_points[plane][i][0] {
                    return Err(invalid(reader, "film_grain: uv_points artan değil"));
                }
                fgd.uv_points[plane][i][1] = reader.read_bits(8)? as u8;
            }
        }
    }

    if seq.ss_hor == 1
        && seq.ss_ver == 1
        && (fgd.num_uv_points[0] != 0) != (fgd.num_uv_points[1] != 0)
    {
        return Err(invalid(
            reader,
            "film_grain: 4:2:0 UV nokta sayıları tutarsız",
        ));
    }

    fgd.scaling_shift = reader.read_bits(2)? as i32 + 8;
    fgd.ar_coeff_lag = reader.read_bits(2)? as i32;
    let num_y_pos = 2 * fgd.ar_coeff_lag * (fgd.ar_coeff_lag + 1);
    if fgd.num_y_points != 0 {
        for coeff in &mut fgd.ar_coeffs_y[..num_y_pos as usize] {
            *coeff = (reader.read_bits(8)? as i32 - 128) as i8;
        }
    }
    for plane in 0..2 {
        if fgd.num_uv_points[plane] != 0 || fgd.chroma_scaling_from_luma {
            let num_uv_pos = num_y_pos + i32::from(fgd.num_y_points != 0);
            for coeff in &mut fgd.ar_coeffs_uv[plane][..num_uv_pos as usize] {
                *coeff = (reader.read_bits(8)? as i32 - 128) as i8;
            }
            if fgd.num_y_points == 0 {
                fgd.ar_coeffs_uv[plane][num_uv_pos as usize] = 0;
            }
        }
    }
    fgd.ar_coeff_shift = u64::from(reader.read_bits(2)?) + 6;
    fgd.grain_scale_shift = reader.read_bits(2)? as i32;
    for plane in 0..2 {
        if fgd.num_uv_points[plane] != 0 {
            fgd.uv_mult[plane] = reader.read_bits(8)? as i32 - 128;
            fgd.uv_luma_mult[plane] = reader.read_bits(8)? as i32 - 128;
            fgd.uv_offset[plane] = reader.read_bits(9)? as i32 - 256;
        }
    }
    fgd.overlap_flag = reader.read_bit()? != 0;
    fgd.clip_to_restricted_range = reader.read_bit()? != 0;
    Ok(())
}

/// Tile group / frame OBU'sunun tile başlığı ve tile boyutları.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TileGroupInfo {
    /// `tile_start_and_end_present_flag` okundu mu?
    pub have_tile_pos: bool,
    /// İlk tile indeksi.
    pub start: u16,
    /// Son tile indeksi (dâhil).
    pub end: u16,
    /// Tile grubundaki tile'ların bayt cinsinden boyutları.
    pub tile_sizes: Vec<u32>,
    /// Bu OBU'daki tile verisi toplam baytı.
    pub data_bytes: usize,
    /// Tile verisinin OBU yükü içindeki bayt ofseti (başlıktan sonra).
    pub data_offset: usize,
}

/// `parse_tile_hdr()` + tile boyut alanları (dav1d `src/obu.c` 1157–1166,
/// `src/decode.c` tile size okuma; AV1 §5.10.1).
///
/// `payload`, OBU başlığından **sonraki** ve frame header'dan sonra bayt
/// hizalanmış tile verisidir. Tile verisi çözülmez; boyut alanları okunur ve
/// veri atlanır.
///
/// # Hatalar
///
/// [`BitstreamErrorKind::InvalidTileGroup`]: `start > end`, `start` beklenen
/// tile indeksinden farklı ya da tile boyutu kalan baytı aşıyor.
pub fn parse_tile_group(payload: &[u8], header: &FrameHeader) -> Result<TileGroupInfo> {
    let n_tiles = u32::from(header.tiling.cols) * u32::from(header.tiling.rows);
    let mut reader = BitReader::new(payload);
    let have_tile_pos = n_tiles > 1 && reader.read_bit()? != 0;
    let (start, end) = if have_tile_pos {
        let n_bits = u32::from(header.tiling.log2_cols) + u32::from(header.tiling.log2_rows);
        let start = reader.read_bits(n_bits)? as u16;
        let end = reader.read_bits(n_bits)? as u16;
        (start, end)
    } else {
        (0, n_tiles.saturating_sub(1) as u16)
    };
    reader.align_to_byte();
    let data_start = reader.byte_position();
    if data_start > payload.len() {
        return Err(invalid_tile(
            reader.bit_position(),
            "tile group başlığı OBU'yu aştı",
        ));
    }
    if start > end {
        return Err(invalid_tile(reader.bit_position(), "tile_start > tile_end"));
    }
    let mut pos = data_start;
    let mut tile_sizes = Vec::new();
    for tile in start..=end {
        if tile < end {
            let size_bytes = usize::from(header.tiling.n_bytes);
            if size_bytes == 0 || pos.saturating_add(size_bytes) > payload.len() {
                return Err(invalid_tile(pos * 8, "tile boyut alanı OBU'yu aştı"));
            }
            // AV1 §5.10.1: `tile_size_minus_1 = le(TileSizeBytes)` — baytlar
            // küçük uçlu (dav1d decode.c 3170–3171: `*data++ << (k * 8)`).
            let mut size_minus_1: u32 = 0;
            for k in 0..size_bytes {
                size_minus_1 |= u32::from(payload[pos + k]) << (k * 8);
            }
            pos += size_bytes;
            let size = (size_minus_1 as usize).saturating_add(1);
            if size > payload.len() - pos {
                return Err(invalid_tile(
                    pos * 8,
                    "tile boyutu kalan OBU baytını aşıyor",
                ));
            }
            // Tile verisi çözülmez; yalnız sınırıyla atlanır (F2b).
            pos += size;
            tile_sizes.push(size as u32);
        } else {
            tile_sizes.push((payload.len() - pos) as u32);
        }
    }
    let data_bytes = tile_sizes.iter().map(|&size| size as usize).sum();
    Ok(TileGroupInfo {
        have_tile_pos,
        start,
        end,
        tile_sizes,
        data_bytes,
        data_offset: data_start,
    })
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use alloc::format;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::{
        DEFAULT_MODE_REF_DELTAS, DEFAULT_WM_PARAMS, FilmGrainData, FilmGrainHeader, FrameHeader,
        FrameHeaderContext, FrameType, LoopfilterModeRefDeltas, NUM_REF_FRAMES, PRIMARY_REF_NONE,
        WarpedMotionType, get_poc_diff, parse_tile_group, read_film_grain, read_frame_refs,
        read_frame_size, read_global_motion, read_loop_filter, read_segmentation, read_tile_info,
    };
    use crate::bitstream::BitReader;
    use crate::bitstream::seq_header::SequenceHeader;

    /// Testler için MSB-first bit yazıcı; `ns(max)` kodlayıcısı dâhil.
    struct BitWriter {
        bits: Vec<u8>,
    }

    impl BitWriter {
        fn new() -> Self {
            Self { bits: Vec::new() }
        }

        fn bit(&mut self, value: u32) {
            self.bits.push((value & 1) as u8);
        }

        fn bits(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                self.bit((value >> i) & 1);
            }
        }

        /// dav1d `dav1d_get_uniform` tersi (`ns(max)` kodlayıcısı).
        fn uniform(&mut self, max: u32, value: u32) {
            let l = 32 - max.leading_zeros();
            let m = (1u32 << l) - max;
            if value < m {
                self.bits(value, l - 1);
            } else {
                self.bits(value + m, l);
            }
        }

        fn bytes(self) -> Vec<u8> {
            let mut out = vec![0u8; self.bits.len().div_ceil(8)];
            for (i, bit) in self.bits.iter().enumerate() {
                if *bit != 0 {
                    out[i / 8] |= 1 << (7 - (i % 8));
                }
            }
            out
        }
    }

    #[test]
    fn poc_farki_isaretli_ve_moduler() {
        assert_eq!(get_poc_diff(0, 5, 3), 0);
        assert_eq!(get_poc_diff(8, 5, 3), 2);
        assert_eq!(get_poc_diff(8, 3, 5), -2);
        // 8 bit order hint: 255 -> -1, 1 -> -255 (modüler).
        assert_eq!(get_poc_diff(8, 255, 0), -1);
        assert_eq!(get_poc_diff(8, 0, 255), 1);
    }

    fn empty_ctx<'a>(
        seq: &'a SequenceHeader,
        refs: &'a [Option<FrameHeader>; 8],
    ) -> FrameHeaderContext<'a> {
        FrameHeaderContext {
            sequence_header: seq,
            refs,
            temporal_id: 0,
            spatial_id: 0,
            strict_std_compliance: false,
        }
    }

    #[test]
    fn gosterme_mevcut_kare_referans_yoksa_reddedilir() {
        // reduced olmayan, order_hint'siz minimal sequence header.
        let seq = SequenceHeader {
            num_operating_points: 1,
            max_width: 64,
            max_height: 64,
            width_n_bits: 1,
            height_n_bits: 1,
            frame_id_numbers_present: true,
            frame_id_n_bits: 8,
            ..SequenceHeader::default()
        };
        let refs = [const { None }; NUM_REF_FRAMES];
        let ctx = empty_ctx(&seq, &refs);
        // show_existing_frame=1, existing_frame_idx=0; trailing bit yok.
        let payload = [0b1000_0000u8, 0x00];
        let mut reader = BitReader::new(&payload);
        let err = FrameHeader::parse(&mut reader, &ctx).expect_err("referanssız gösterim");
        assert!(format!("{err}").contains("referans"));
    }

    #[test]
    fn varsayilanlar_dav1d_memset_ile_ayni() {
        let h = FrameHeader::default();
        assert_eq!(h.frame_type, FrameType::Key);
        assert_eq!(h.primary_ref_frame, 0);
        assert_eq!(PRIMARY_REF_NONE, 7);
        assert_eq!(h.refidx, [0; 7]);
        // show_existing erken dönüşü öncesi gmv matrisleri sıfırdır.
        assert_eq!(h.gmv[0].matrix, [0; 6]);
        // Parser'ın atadığı varsayılan ise kimlik matrisidir.
        assert_eq!(DEFAULT_WM_PARAMS.matrix, [0, 0, 1 << 16, 0, 0, 1 << 16]);
        assert_eq!(DEFAULT_MODE_REF_DELTAS.ref_delta[0], 1);
    }

    #[test]
    fn tile_grubu_basligi_ve_boyutlari() {
        let mut header = FrameHeader::default();
        header.tiling.cols = 2;
        header.tiling.rows = 1;
        header.tiling.log2_cols = 1;
        header.tiling.log2_rows = 0;
        header.tiling.n_bytes = 1;
        // tile_start_and_end_present=0; tile 0 boyutu 3 (0x02 + 1), tile 1 kalan 2.
        let payload = [0x00, 0x02, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE];
        let info = parse_tile_group(&payload, &header).expect("tile group");
        assert!(!info.have_tile_pos);
        assert_eq!((info.start, info.end), (0, 1));
        assert_eq!(info.tile_sizes, vec![3, 2]);
        assert_eq!(info.data_bytes, 5);
    }

    #[test]
    fn tile_grubu_sinir_ihlalleri_reddedilir() {
        let mut header = FrameHeader::default();
        header.tiling.cols = 2;
        header.tiling.rows = 1;
        header.tiling.log2_cols = 1;
        header.tiling.n_bytes = 1;
        // start=1 > end=0.
        let payload = [0b1100_0000, 0b0100_0000];
        assert!(parse_tile_group(&payload, &header).is_err());
        // Tile boyutu kalandan büyük: 0x40 + 1 = 65 > 2.
        let payload = [0x00, 0x40, 0xAA, 0xBB];
        assert!(parse_tile_group(&payload, &header).is_err());
    }

    #[test]
    fn kisa_sinyal_referans_secimi_dav1d_algoritmasi() {
        let mut refs = [const { None }; NUM_REF_FRAMES];
        for (i, slot) in refs.iter_mut().enumerate() {
            let mut header = FrameHeader::default();
            header.frame_offset = ((i as u8) + 1) * 10;
            *slot = Some(header);
        }
        let seq = SequenceHeader {
            order_hint: true,
            order_hint_n_bits: 8,
            ..SequenceHeader::default()
        };
        let ctx = FrameHeaderContext {
            sequence_header: &seq,
            refs: &refs,
            temporal_id: 0,
            spatial_id: 0,
            strict_std_compliance: false,
        };
        let mut writer = BitWriter::new();
        writer.bits(2, 3); // refidx[0] = 2
        writer.bits(5, 3); // refidx[3] = 5
        let payload = writer.bytes();
        let mut reader = BitReader::new(&payload);
        let mut header = FrameHeader {
            frame_offset: 100,
            ..FrameHeader::default()
        };
        read_frame_refs(&mut reader, &ctx, &mut header).expect("referans seçimi");
        assert!(!header.frame_ref_short_signaling);
        assert_eq!(header.refidx, [2, 7, 6, 5, 4, 3, 1]);

        // Bütün referanslar mevcut olmalı; boş yuva hatadır.
        let refs_missing = [const { None }; NUM_REF_FRAMES];
        let ctx_missing = FrameHeaderContext {
            refs: &refs_missing,
            ..ctx
        };
        let mut reader = BitReader::new(&payload);
        let mut header = FrameHeader {
            frame_offset: 100,
            ..FrameHeader::default()
        };
        assert!(read_frame_refs(&mut reader, &ctx_missing, &mut header).is_err());
    }

    #[test]
    fn segmentasyon_verisi_primary_reften_devralinir() {
        let mut reference = FrameHeader::default();
        reference.segmentation.seg_data.last_active_segid = 3;
        reference.segmentation.seg_data.preskip = true;
        reference.segmentation.seg_data.d[2].delta_q = -7;
        reference.segmentation.seg_data.d[2].reference = 4;
        let mut refs = [const { None }; NUM_REF_FRAMES];
        refs[4] = Some(reference);
        let seq = SequenceHeader {
            monochrome: true,
            ..SequenceHeader::default()
        };
        let ctx = FrameHeaderContext {
            sequence_header: &seq,
            refs: &refs,
            temporal_id: 0,
            spatial_id: 0,
            strict_std_compliance: false,
        };
        let mut header = FrameHeader::default();
        header.primary_ref_frame = 0;
        header.refidx[0] = 4;
        // enabled=1, update_map=0, update_data=0 → devral.
        let payload = [0b1000_0000u8];
        let mut reader = BitReader::new(&payload);
        read_segmentation(&mut reader, &ctx, &mut header).expect("segmentasyon devralma");
        assert!(header.segmentation.enabled);
        assert!(!header.segmentation.update_data);
        assert_eq!(
            header.segmentation.seg_data,
            refs[4].as_ref().expect("referans").segmentation.seg_data
        );

        // update_data=1: veri ayrıştırılır, devralınmaz.
        let mut header = FrameHeader::default();
        header.primary_ref_frame = 0;
        header.refidx[0] = 4;
        let mut writer = BitWriter::new();
        writer.bit(1); // enabled
        writer.bit(0); // update_map
        writer.bit(1); // update_data
        for _ in 0..8 {
            writer.bit(0); // delta_q yok
            writer.bit(0); // delta_lf_y_v yok
            writer.bit(0); // delta_lf_y_h yok
            writer.bit(0); // delta_lf_u yok
            writer.bit(0); // delta_lf_v yok
            writer.bit(0); // ref yok
            writer.bit(0); // skip yok
            writer.bit(0); // globalmv yok
        }
        let payload = writer.bytes();
        let mut reader = BitReader::new(&payload);
        read_segmentation(&mut reader, &ctx, &mut header).expect("segmentasyon güncelleme");
        assert!(header.segmentation.update_data);
        assert_eq!(header.segmentation.seg_data.last_active_segid, -1);
        assert_eq!(header.segmentation.seg_data.d[2].reference, -1);
    }

    #[test]
    fn loop_filter_deltalari_primary_reften_devralinir() {
        let mut reference = FrameHeader::default();
        reference.loopfilter.mode_ref_deltas = LoopfilterModeRefDeltas {
            mode_delta: [-2, 3],
            ref_delta: [1, -1, 2, -3, 4, -5, 6, -7],
        };
        let mut refs = [const { None }; NUM_REF_FRAMES];
        refs[4] = Some(reference);
        let seq = SequenceHeader {
            monochrome: true,
            ..SequenceHeader::default()
        };
        let ctx = FrameHeaderContext {
            sequence_header: &seq,
            refs: &refs,
            temporal_id: 0,
            spatial_id: 0,
            strict_std_compliance: false,
        };
        let mut header = FrameHeader::default();
        header.primary_ref_frame = 0;
        header.refidx[0] = 4;
        // level_y[0]=0 (6), level_y[1]=0 (6), sharpness=0 (3), enabled=0 (1).
        let payload = [0u8, 0u8];
        let mut reader = BitReader::new(&payload);
        read_loop_filter(&mut reader, &ctx, &seq, &mut header).expect("loop filter devralma");
        assert_eq!(
            header.loopfilter.mode_ref_deltas,
            refs[4]
                .as_ref()
                .expect("referans")
                .loopfilter
                .mode_ref_deltas
        );
        assert!(!header.loopfilter.mode_ref_delta_enabled);
    }

    #[test]
    fn tek_tip_olmayan_tile_bolumleme_sinirlari() {
        let seq = SequenceHeader {
            monochrome: true,
            max_width: 192,
            max_height: 100,
            ..SequenceHeader::default()
        };
        let mut header = FrameHeader::default();
        header.width = [192, 192];
        header.height = 100;
        let mut writer = BitWriter::new();
        writer.bit(0); // uniform_tile_spacing_flag = 0
        writer.uniform(3, 1); // 3 süper blok genişliğinden tile_w = 2
        // Kalan 1 süper blok: tile_width_sb = 1, bit yok.
        writer.uniform(2, 1); // 2 süper blok yüksekliğinden tile_h = 2
        writer.bit(0); // context_update_tile_id (1 bit)
        writer.bits(1, 2); // tile_size_bytes_minus_1 = 1 → 2 bayt
        let payload = writer.bytes();
        let mut reader = BitReader::new(&payload);
        read_tile_info(&mut reader, &seq, &mut header).expect("tile bilgisi");
        assert!(!header.tiling.uniform);
        assert_eq!(header.tiling.cols, 2);
        assert_eq!(header.tiling.rows, 1);
        assert_eq!(header.tiling.log2_cols, 1);
        assert_eq!(header.tiling.log2_rows, 0);
        assert_eq!(header.tiling.n_bytes, 2);
        assert_eq!(header.tiling.update, 0);
        assert_eq!(header.tiling.col_start_sb[..3], [0, 2, 3]);
        assert_eq!(header.tiling.row_start_sb[..2], [0, 2]);
    }

    #[test]
    fn global_motion_parametre_kodlamasi() {
        let seq = SequenceHeader::default();
        let refs = [const { None }; NUM_REF_FRAMES];
        let ctx = FrameHeaderContext {
            sequence_header: &seq,
            refs: &refs,
            temporal_id: 0,
            spatial_id: 0,
            strict_std_compliance: false,
        };
        let mut header = FrameHeader::default();
        header.hp = true;
        header.primary_ref_frame = PRIMARY_REF_NONE;
        header.gmv = [DEFAULT_WM_PARAMS; 7];
        let mut writer = BitWriter::new();
        // gmv[0]: type = TRANSLATION (1, 0, 1)
        writer.bit(1);
        writer.bit(0);
        writer.bit(1);
        // mat[0], mat[1]: su değeri -1 (0 + 3 bit 001 → recenter 511 → -1)
        writer.bit(0);
        writer.bits(1, 3);
        writer.bit(0);
        writer.bits(1, 3);
        // gmv[1..7]: IDENTITY
        for _ in 0..6 {
            writer.bit(0);
        }
        let payload = writer.bytes();
        let mut reader = BitReader::new(&payload);
        read_global_motion(&mut reader, &ctx, &mut header).expect("global motion");
        assert_eq!(header.gmv[0].kind, WarpedMotionType::Translation);
        assert_eq!(header.gmv[0].matrix[0], -8192);
        assert_eq!(header.gmv[0].matrix[1], -8192);
        // Öteleme yolunda mat[2], mat[3] kimlik varsayılanından gelir;
        // mat[4] = -mat[3], mat[5] = mat[2] (dav1d obu.c 1054–1056).
        assert_eq!(header.gmv[0].matrix[2], 1 << 16);
        assert_eq!(header.gmv[0].matrix[3], 0);
        assert_eq!(header.gmv[0].matrix[4], 0);
        assert_eq!(header.gmv[0].matrix[5], 1 << 16);
        for gmv in &header.gmv[1..7] {
            assert_eq!(gmv.kind, WarpedMotionType::Identity);
            assert_eq!(gmv.matrix, DEFAULT_WM_PARAMS.matrix);
        }
        assert_eq!(reader.bit_position(), 17);
    }

    #[test]
    fn film_grain_guncelleme_ve_yukleme() {
        let seq = SequenceHeader {
            film_grain_present: true,
            monochrome: true,
            ss_hor: 1,
            ss_ver: 1,
            ..SequenceHeader::default()
        };
        let mut refs = [const { None }; NUM_REF_FRAMES];
        let ctx = FrameHeaderContext {
            sequence_header: &seq,
            refs: &refs,
            temporal_id: 0,
            spatial_id: 0,
            strict_std_compliance: false,
        };

        // --- update: bütün parametreler sıfır, seed 0x1234 ---
        let mut header = FrameHeader::default();
        header.frame_type = FrameType::Inter;
        header.show_frame = true;
        let mut writer = BitWriter::new();
        writer.bit(1); // present
        writer.bits(0x1234, 16); // seed
        writer.bit(1); // update
        writer.bits(0, 4); // num_y_points = 0
        writer.bits(0, 2); // scaling_shift → 8
        writer.bits(0, 2); // ar_coeff_lag = 0
        writer.bits(0, 2); // ar_coeff_shift → 6
        writer.bits(0, 2); // grain_scale_shift = 0
        writer.bit(0); // overlap_flag
        writer.bit(0); // clip_to_restricted_range
        let payload = writer.bytes();
        let mut reader = BitReader::new(&payload);
        read_film_grain(&mut reader, &ctx, &seq, &mut header).expect("film grain update");
        assert!(header.film_grain.present);
        assert!(header.film_grain.update);
        assert_eq!(header.film_grain.data.seed, 0x1234);
        assert_eq!(header.film_grain.data.num_y_points, 0);
        assert_eq!(header.film_grain.data.scaling_shift, 8);
        assert_eq!(header.film_grain.data.ar_coeff_shift, 6);
        assert!(!header.film_grain.data.overlap_flag);

        // --- load: referans kareden devral, yalnız seed değişir ---
        let mut reference = FrameHeader::default();
        reference.film_grain = FilmGrainHeader {
            present: true,
            update: true,
            data: FilmGrainData {
                seed: 111,
                num_y_points: 2,
                scaling_shift: 11,
                ar_coeff_lag: 2,
                ..FilmGrainData::default()
            },
        };
        refs[3] = Some(reference);
        let ctx = FrameHeaderContext {
            sequence_header: &seq,
            refs: &refs,
            temporal_id: 0,
            spatial_id: 0,
            strict_std_compliance: false,
        };
        let mut header = FrameHeader::default();
        header.frame_type = FrameType::Inter;
        header.show_frame = true;
        header.refidx[0] = 3;
        let mut writer = BitWriter::new();
        writer.bit(1); // present
        writer.bits(0x2222, 16); // seed
        writer.bit(0); // update = 0 → load
        writer.bits(3, 3); // refidx
        let payload = writer.bytes();
        let mut reader = BitReader::new(&payload);
        read_film_grain(&mut reader, &ctx, &seq, &mut header).expect("film grain load");
        assert!(!header.film_grain.update);
        assert_eq!(header.film_grain.data.seed, 0x2222);
        assert_eq!(header.film_grain.data.num_y_points, 2);
        assert_eq!(header.film_grain.data.scaling_shift, 11);
        assert_eq!(header.film_grain.data.ar_coeff_lag, 2);
    }

    #[test]
    fn frame_size_override_ve_with_refs_max_boyut_asimini_reddeder() {
        let seq = SequenceHeader {
            max_width: 64,
            max_height: 48,
            width_n_bits: 8,
            height_n_bits: 8,
            super_res: true,
            ..SequenceHeader::default()
        };
        let refs_empty = [const { None }; NUM_REF_FRAMES];
        let ctx_empty = empty_ctx(&seq, &refs_empty);

        // 1. `frame_size_override`: (64, 48) geçerli.
        {
            let mut w = BitWriter::new();
            w.bits(63, 8); // width = 64
            w.bits(47, 8); // height = 48
            w.bit(0); // super_res.enabled = 0
            w.bit(0); // have_render_size = 0
            let bytes = w.bytes();
            let mut br = BitReader::new(&bytes);
            let mut h = FrameHeader {
                frame_size_override: true,
                ..FrameHeader::default()
            };
            read_frame_size(&mut br, &ctx_empty, &mut h, false)
                .expect("sınır içindeki frame_size_override kabul edilmeli");
            assert_eq!((h.width[1], h.height), (64, 48));
        }

        // 2. `frame_size_override`: width = 65 > max_width = 64 -> super_res bitinden önce hata.
        for (w_m1, h_m1) in [(64u32, 47u32), (63u32, 48u32)] {
            let mut w = BitWriter::new();
            w.bits(w_m1, 8);
            w.bits(h_m1, 8);
            w.bit(1); // super_res.enabled = 1 (okunmamalı!)
            let bytes = w.bytes();
            let mut br = BitReader::new(&bytes);
            let mut h = FrameHeader {
                frame_size_override: true,
                ..FrameHeader::default()
            };
            let err = read_frame_size(&mut br, &ctx_empty, &mut h, false)
                .expect_err("max_width/max_height aşan frame_size_override reddedilmeli");
            let msg = format!("{err}");
            assert!(
                msg.contains("frame_size_override") && !msg.contains("frame_size_limit"),
                "beklenmeyen hata iletisi: {msg}"
            );
            assert_eq!(
                br.bit_position(),
                16,
                "hata super_res.enabled biti okunmadan tam 16. bitte üretilmeli"
            );
        }

        // 3. `frame_size_with_refs`: referans kare boyutu `seq.max_width` veya `seq.max_height`
        //    sınırını aşıyorsa `super_res.enabled` biti okunmadan reddedilmeli (`dav1d` `obu.c:355-359`).
        for (ref_w, ref_h, should_ok) in [(64, 48, true), (65, 48, false), (64, 49, false)] {
            let mut refs = [const { None }; NUM_REF_FRAMES];
            refs[2] = Some(FrameHeader {
                width: [ref_w, ref_w],
                height: ref_h,
                render_width: ref_w,
                render_height: ref_h,
                ..FrameHeader::default()
            });
            let ctx = empty_ctx(&seq, &refs);
            let mut w = BitWriter::new();
            w.bit(1); // found_ref[0] = 1 -> refidx[0] = 2
            w.bit(0); // super_res.enabled = 0
            let bytes = w.bytes();
            let mut br = BitReader::new(&bytes);
            let mut h = FrameHeader::default();
            h.refidx[0] = 2;
            let res = read_frame_size(&mut br, &ctx, &mut h, true);
            if should_ok {
                res.expect("sınır içindeki frame_size_with_refs kabul edilmeli");
                assert_eq!((h.width[1], h.height), (64, 48));
            } else {
                let err = res.expect_err("sınırı aşan frame_size_with_refs reddedilmeli");
                let msg = format!("{err}");
                assert!(
                    msg.contains("frame_size_with_refs") && !msg.contains("frame_size_limit"),
                    "beklenmeyen hata iletisi: {msg}"
                );
                assert_eq!(
                    br.bit_position(),
                    1,
                    "hata super_res.enabled biti okunmadan tam 1. bitte üretilmeli"
                );
            }
        }
    }
}
