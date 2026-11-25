//! `libdav1d` (`1.5.4`, API `7.0.0`) C-ABI veri yapıları, sabitler ve hata kodları.
//!
//! Bu modül `include/dav1d/{common,version,headers,picture,data,dav1d}.h` başlıklarındaki
//! tüm C yapılarını (`#[repr(C)]`), sabitleri ve geri çağrı imzalarını sıfır dış
//! bağımlılıkla (`core::ffi`) tanımlar.

#![forbid(unsafe_code)]

use core::ffi::{c_int, c_uint, c_void};

/// `DAV1D_API_VERSION_MAJOR` (`include/dav1d/version.h`).
pub const DAV1D_API_VERSION_MAJOR: c_uint = 7;
/// `DAV1D_API_VERSION_MINOR` (`include/dav1d/version.h`).
pub const DAV1D_API_VERSION_MINOR: c_uint = 0;
/// `DAV1D_API_VERSION_PATCH` (`include/dav1d/version.h`).
pub const DAV1D_API_VERSION_PATCH: c_uint = 0;

/// `DAV1D_PICTURE_ALIGNMENT` (`include/dav1d/picture.h`).
pub const DAV1D_PICTURE_ALIGNMENT: usize = 64;
/// `DAV1D_MAX_THREADS` (`include/dav1d/dav1d.h`).
pub const DAV1D_MAX_THREADS: c_int = 256;
/// `DAV1D_MAX_FRAME_DELAY` (`include/dav1d/dav1d.h`).
pub const DAV1D_MAX_FRAME_DELAY: c_int = 256;

/// `DAV1D_MAX_CDEF_STRENGTHS` (`include/dav1d/headers.h`).
pub const DAV1D_MAX_CDEF_STRENGTHS: usize = 8;
/// `DAV1D_MAX_OPERATING_POINTS` (`include/dav1d/headers.h`).
pub const DAV1D_MAX_OPERATING_POINTS: usize = 32;
/// `DAV1D_MAX_TILE_COLS` (`include/dav1d/headers.h`).
pub const DAV1D_MAX_TILE_COLS: usize = 64;
/// `DAV1D_MAX_TILE_ROWS` (`include/dav1d/headers.h`).
pub const DAV1D_MAX_TILE_ROWS: usize = 64;
/// `DAV1D_MAX_SEGMENTS` (`include/dav1d/headers.h`).
pub const DAV1D_MAX_SEGMENTS: usize = 8;
/// `DAV1D_NUM_REF_FRAMES` (`include/dav1d/headers.h`).
pub const DAV1D_NUM_REF_FRAMES: usize = 8;
/// `DAV1D_PRIMARY_REF_NONE` (`include/dav1d/headers.h`).
pub const DAV1D_PRIMARY_REF_NONE: u8 = 7;
/// `DAV1D_REFS_PER_FRAME` (`include/dav1d/headers.h`).
pub const DAV1D_REFS_PER_FRAME: usize = 7;
/// `DAV1D_TOTAL_REFS_PER_FRAME` (`include/dav1d/headers.h`).
pub const DAV1D_TOTAL_REFS_PER_FRAME: usize = DAV1D_REFS_PER_FRAME + 1;

/// POSIX `ENOENT` hata kodu.
pub const ENOENT: c_int = 2;
/// POSIX `EIO` hata kodu.
pub const EIO: c_int = 5;
/// POSIX `ENOMEM` hata kodu.
pub const ENOMEM: c_int = 12;
/// POSIX `EINVAL` hata kodu.
pub const EINVAL: c_int = 22;
/// POSIX `ERANGE` hata kodu.
pub const ERANGE: c_int = 34;

/// Platforma duyarlı POSIX `EAGAIN` hata kodu (Apple / *BSD üzerinde `35`, Linux / Windows / WASI üzerinde `11`).
#[cfg(any(
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "dragonfly",
))]
pub const EAGAIN: c_int = 35;

/// Platforma duyarlı POSIX `EAGAIN` hata kodu (Linux / Windows / WASI üzerinde `11`).
#[cfg(not(any(
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "dragonfly",
)))]
pub const EAGAIN: c_int = 11;

/// `DAV1D_ERR(e)` makrosu karşılığı: POSIX hata kodunu negatif `c_int`'e çevirir.
#[must_use]
pub const fn dav1d_err(errno: c_int) -> c_int {
    -errno
}

/// Opak çözücü bağlamı (`typedef struct Dav1dContext Dav1dContext`).
pub struct Dav1dContext {
    pub(crate) inner: crate::capi::ffi::ContextInner,
}

/// Opak referans sayımlı bellek nesnesi (`typedef struct Dav1dRef Dav1dRef`).
pub struct Dav1dRef {
    #[allow(dead_code)]
    pub(crate) inner: crate::capi::ffi::RefInner,
}

/// `enum Dav1dObuType` (`include/dav1d/headers.h`).
pub type Dav1dObuType = c_uint;
/// `DAV1D_OBU_SEQ_HDR`.
pub const DAV1D_OBU_SEQ_HDR: Dav1dObuType = 1;
/// `DAV1D_OBU_TD`.
pub const DAV1D_OBU_TD: Dav1dObuType = 2;
/// `DAV1D_OBU_FRAME_HDR`.
pub const DAV1D_OBU_FRAME_HDR: Dav1dObuType = 3;
/// `DAV1D_OBU_TILE_GRP`.
pub const DAV1D_OBU_TILE_GRP: Dav1dObuType = 4;
/// `DAV1D_OBU_METADATA`.
pub const DAV1D_OBU_METADATA: Dav1dObuType = 5;
/// `DAV1D_OBU_FRAME`.
pub const DAV1D_OBU_FRAME: Dav1dObuType = 6;
/// `DAV1D_OBU_REDUNDANT_FRAME_HDR`.
pub const DAV1D_OBU_REDUNDANT_FRAME_HDR: Dav1dObuType = 7;
/// `DAV1D_OBU_PADDING`.
pub const DAV1D_OBU_PADDING: Dav1dObuType = 15;

/// `enum Dav1dTxfmMode` (`include/dav1d/headers.h`).
pub type Dav1dTxfmMode = c_uint;
/// `DAV1D_TX_4X4_ONLY`.
pub const DAV1D_TX_4X4_ONLY: Dav1dTxfmMode = 0;
/// `DAV1D_TX_LARGEST`.
pub const DAV1D_TX_LARGEST: Dav1dTxfmMode = 1;
/// `DAV1D_TX_SWITCHABLE`.
pub const DAV1D_TX_SWITCHABLE: Dav1dTxfmMode = 2;
/// `DAV1D_N_TX_MODES`.
pub const DAV1D_N_TX_MODES: Dav1dTxfmMode = 3;

/// `enum Dav1dFilterMode` (`include/dav1d/headers.h`).
pub type Dav1dFilterMode = c_uint;
/// `DAV1D_FILTER_8TAP_REGULAR`.
pub const DAV1D_FILTER_8TAP_REGULAR: Dav1dFilterMode = 0;
/// `DAV1D_FILTER_8TAP_SMOOTH`.
pub const DAV1D_FILTER_8TAP_SMOOTH: Dav1dFilterMode = 1;
/// `DAV1D_FILTER_8TAP_SHARP`.
pub const DAV1D_FILTER_8TAP_SHARP: Dav1dFilterMode = 2;
/// `DAV1D_N_SWITCHABLE_FILTERS`.
pub const DAV1D_N_SWITCHABLE_FILTERS: Dav1dFilterMode = 3;
/// `DAV1D_FILTER_BILINEAR`.
pub const DAV1D_FILTER_BILINEAR: Dav1dFilterMode = 3;
/// `DAV1D_N_FILTERS`.
pub const DAV1D_N_FILTERS: Dav1dFilterMode = 4;
/// `DAV1D_FILTER_SWITCHABLE`.
pub const DAV1D_FILTER_SWITCHABLE: Dav1dFilterMode = 4;

/// `enum Dav1dAdaptiveBoolean` (`include/dav1d/headers.h`).
pub type Dav1dAdaptiveBoolean = c_uint;
/// `DAV1D_OFF`.
pub const DAV1D_OFF: Dav1dAdaptiveBoolean = 0;
/// `DAV1D_ON`.
pub const DAV1D_ON: Dav1dAdaptiveBoolean = 1;
/// `DAV1D_ADAPTIVE`.
pub const DAV1D_ADAPTIVE: Dav1dAdaptiveBoolean = 2;

/// `enum Dav1dRestorationType` (`include/dav1d/headers.h`).
pub type Dav1dRestorationType = c_uint;
/// `DAV1D_RESTORATION_NONE`.
pub const DAV1D_RESTORATION_NONE: Dav1dRestorationType = 0;
/// `DAV1D_RESTORATION_SWITCHABLE`.
pub const DAV1D_RESTORATION_SWITCHABLE: Dav1dRestorationType = 1;
/// `DAV1D_RESTORATION_WIENER`.
pub const DAV1D_RESTORATION_WIENER: Dav1dRestorationType = 2;
/// `DAV1D_RESTORATION_SGRPROJ`.
pub const DAV1D_RESTORATION_SGRPROJ: Dav1dRestorationType = 3;

/// `enum Dav1dWarpedMotionType` (`include/dav1d/headers.h`).
pub type Dav1dWarpedMotionType = c_uint;
/// `DAV1D_WM_TYPE_IDENTITY`.
pub const DAV1D_WM_TYPE_IDENTITY: Dav1dWarpedMotionType = 0;
/// `DAV1D_WM_TYPE_TRANSLATION`.
pub const DAV1D_WM_TYPE_TRANSLATION: Dav1dWarpedMotionType = 1;
/// `DAV1D_WM_TYPE_ROT_ZOOM`.
pub const DAV1D_WM_TYPE_ROT_ZOOM: Dav1dWarpedMotionType = 2;
/// `DAV1D_WM_TYPE_AFFINE`.
pub const DAV1D_WM_TYPE_AFFINE: Dav1dWarpedMotionType = 3;

/// `Dav1dWarpedMotionParams` (`include/dav1d/headers.h`).
///
/// C tarafındaki `union { struct { int16_t alpha, beta, gamma, delta; } p; int16_t abcd[4]; } u`
/// alanı `abcd: [i16; 4]` ile bayt-birebir aynı boyut ve hizalamaya sahiptir.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dWarpedMotionParams {
    /// Warped motion tipi.
    pub r#type: Dav1dWarpedMotionType,
    /// 6 elemanlı global motion matrisi.
    pub matrix: [i32; 6],
    /// `[alpha, beta, gamma, delta]` shear parametreleri (`u.abcd`).
    pub abcd: [i16; 4],
}

/// `enum Dav1dPixelLayout` (`include/dav1d/headers.h`).
pub type Dav1dPixelLayout = c_uint;
/// `DAV1D_PIXEL_LAYOUT_I400` (Monochrome).
pub const DAV1D_PIXEL_LAYOUT_I400: Dav1dPixelLayout = 0;
/// `DAV1D_PIXEL_LAYOUT_I420` (4:2:0).
pub const DAV1D_PIXEL_LAYOUT_I420: Dav1dPixelLayout = 1;
/// `DAV1D_PIXEL_LAYOUT_I422` (4:2:2).
pub const DAV1D_PIXEL_LAYOUT_I422: Dav1dPixelLayout = 2;
/// `DAV1D_PIXEL_LAYOUT_I444` (4:4:4).
pub const DAV1D_PIXEL_LAYOUT_I444: Dav1dPixelLayout = 3;

/// `enum Dav1dFrameType` (`include/dav1d/headers.h`).
pub type Dav1dFrameType = c_uint;
/// `DAV1D_FRAME_TYPE_KEY`.
pub const DAV1D_FRAME_TYPE_KEY: Dav1dFrameType = 0;
/// `DAV1D_FRAME_TYPE_INTER`.
pub const DAV1D_FRAME_TYPE_INTER: Dav1dFrameType = 1;
/// `DAV1D_FRAME_TYPE_INTRA`.
pub const DAV1D_FRAME_TYPE_INTRA: Dav1dFrameType = 2;
/// `DAV1D_FRAME_TYPE_SWITCH`.
pub const DAV1D_FRAME_TYPE_SWITCH: Dav1dFrameType = 3;

/// `enum Dav1dColorPrimaries` (`include/dav1d/headers.h`).
pub type Dav1dColorPrimaries = c_uint;
/// `enum Dav1dTransferCharacteristics` (`include/dav1d/headers.h`).
pub type Dav1dTransferCharacteristics = c_uint;
/// `enum Dav1dMatrixCoefficients` (`include/dav1d/headers.h`).
pub type Dav1dMatrixCoefficients = c_uint;
/// `enum Dav1dChromaSamplePosition` (`include/dav1d/headers.h`).
pub type Dav1dChromaSamplePosition = c_uint;

/// `Dav1dContentLightLevel` (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dContentLightLevel {
    /// Maksimum içerik ışık düzeyi.
    pub max_content_light_level: u16,
    /// Maksimum kare ortalama ışık düzeyi.
    pub max_frame_average_light_level: u16,
}

/// `Dav1dMasteringDisplay` (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dMasteringDisplay {
    /// RGB ana renk koordinatları (`0.16` sabit nokta).
    pub primaries: [[u16; 2]; 3],
    /// Beyaz nokta koordinatları (`0.16` sabit nokta).
    pub white_point: [u16; 2],
    /// Maksimum parlaklık (`24.8` sabit nokta).
    pub max_luminance: u32,
    /// Minimum parlaklık (`18.14` sabit nokta).
    pub min_luminance: u32,
}

/// `Dav1dITUTT35` (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Dav1dITUTT35 {
    /// Ülke kodu.
    pub country_code: u8,
    /// Ülke kodu uzantı baytı.
    pub country_code_extension_byte: u8,
    /// Yük boyutu (bayt).
    pub payload_size: usize,
    /// Yük işaretçisi.
    pub payload: *mut u8,
}

/// `struct Dav1dSequenceHeaderOperatingPoint` (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dSequenceHeaderOperatingPoint {
    /// Major seviye.
    pub major_level: u8,
    /// Minor seviye.
    pub minor_level: u8,
    /// İlk gösterim gecikmesi.
    pub initial_display_delay: u8,
    /// Operating point IDC maskesi.
    pub idc: u16,
    /// Tier bayrağı.
    pub tier: u8,
    /// Decoder model parametresi var mı?
    pub decoder_model_param_present: u8,
    /// Display model parametresi var mı?
    pub display_model_param_present: u8,
}

/// `struct Dav1dSequenceHeaderOperatingParameterInfo` (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dSequenceHeaderOperatingParameterInfo {
    /// Çözücü tampon gecikmesi.
    pub decoder_buffer_delay: u32,
    /// Kodlayıcı tampon gecikmesi.
    pub encoder_buffer_delay: u32,
    /// Düşük gecikme kipi bayrağı.
    pub low_delay_mode: u8,
}

/// `Dav1dSequenceHeader` (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dav1dSequenceHeader {
    /// AV1 profili (`0`, `1` veya `2`).
    pub profile: u8,
    /// Maksimum genişlik.
    pub max_width: c_int,
    /// Maksimum yükseklik.
    pub max_height: c_int,
    /// Piksel yerleşimi (`Dav1dPixelLayout`).
    pub layout: Dav1dPixelLayout,
    /// Renk ana değerleri (`Dav1dColorPrimaries`).
    pub pri: Dav1dColorPrimaries,
    /// Transfer karakteristiği (`Dav1dTransferCharacteristics`).
    pub trc: Dav1dTransferCharacteristics,
    /// Matris katsayıları (`Dav1dMatrixCoefficients`).
    pub mtrx: Dav1dMatrixCoefficients,
    /// Kroma örnek konumu (`Dav1dChromaSamplePosition`).
    pub chr: Dav1dChromaSamplePosition,
    /// Yüksek bit derinliği (`0 = 8b`, `1 = 10b`, `2 = 12b`).
    pub hbd: u8,
    /// Renk aralığı (`0 = sınırlı`, `1 = tam`).
    pub color_range: u8,
    /// Operating point sayısı.
    pub num_operating_points: u8,
    /// Operating point dizisi.
    pub operating_points: [Dav1dSequenceHeaderOperatingPoint; DAV1D_MAX_OPERATING_POINTS],
    /// Still picture bayrağı.
    pub still_picture: u8,
    /// Reduced still picture header bayrağı.
    pub reduced_still_picture_header: u8,
    /// Timing info varlık bayrağı.
    pub timing_info_present: u8,
    /// Tick başına birim sayısı.
    pub num_units_in_tick: u32,
    /// Zaman ölçeği.
    pub time_scale: u32,
    /// Eşit resim aralığı bayrağı.
    pub equal_picture_interval: u8,
    /// Resim başına tick sayısı.
    pub num_ticks_per_picture: u32,
    /// Decoder model info varlık bayrağı.
    pub decoder_model_info_present: u8,
    /// Tampon gecikmesi bit uzunluğu.
    pub encoder_decoder_buffer_delay_length: u8,
    /// Çözme tick'i başına birim sayısı.
    pub num_units_in_decoding_tick: u32,
    /// Tampon kaldırma gecikmesi bit uzunluğu.
    pub buffer_removal_delay_length: u8,
    /// Kare sunum gecikmesi bit uzunluğu.
    pub frame_presentation_delay_length: u8,
    /// Display model info varlık bayrağı.
    pub display_model_info_present: u8,
    /// Genişlik bit sayısı.
    pub width_n_bits: u8,
    /// Yükseklik bit sayısı.
    pub height_n_bits: u8,
    /// Frame ID varlık bayrağı.
    pub frame_id_numbers_present: u8,
    /// Delta frame ID bit sayısı.
    pub delta_frame_id_n_bits: u8,
    /// Frame ID bit sayısı.
    pub frame_id_n_bits: u8,
    /// 128×128 süperblok bayrağı.
    pub sb128: u8,
    /// Filter intra bayrağı.
    pub filter_intra: u8,
    /// Intra edge filter bayrağı.
    pub intra_edge_filter: u8,
    /// Inter-intra bayrağı.
    pub inter_intra: u8,
    /// Masked compound bayrağı.
    pub masked_compound: u8,
    /// Warped motion bayrağı.
    pub warped_motion: u8,
    /// Dual filter bayrağı.
    pub dual_filter: u8,
    /// Order hint bayrağı.
    pub order_hint: u8,
    /// Joint compound bayrağı.
    pub jnt_comp: u8,
    /// Ref frame MVs bayrağı.
    pub ref_frame_mvs: u8,
    /// Screen content tools (`Dav1dAdaptiveBoolean`).
    pub screen_content_tools: Dav1dAdaptiveBoolean,
    /// Force integer MV (`Dav1dAdaptiveBoolean`).
    pub force_integer_mv: Dav1dAdaptiveBoolean,
    /// Order hint bit sayısı.
    pub order_hint_n_bits: u8,
    /// Superres bayrağı.
    pub super_res: u8,
    /// CDEF bayrağı.
    pub cdef: u8,
    /// Loop restoration bayrağı.
    pub restoration: u8,
    /// Yatay alt örnekleme.
    pub ss_hor: u8,
    /// Dikey alt örnekleme.
    pub ss_ver: u8,
    /// Monochrome bayrağı.
    pub monochrome: u8,
    /// Color description varlık bayrağı.
    pub color_description_present: u8,
    /// Ayrı UV delta-Q bayrağı.
    pub separate_uv_delta_q: u8,
    /// Film grain varlık bayrağı.
    pub film_grain_present: u8,
    /// Operating parameter info dizisi.
    pub operating_parameter_info:
        [Dav1dSequenceHeaderOperatingParameterInfo; DAV1D_MAX_OPERATING_POINTS],
}

/// `Dav1dSegmentationData` (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dSegmentationData {
    /// Kuantizasyon deltası.
    pub delta_q: i16,
    /// Dikey Y loop filter deltası.
    pub delta_lf_y_v: i8,
    /// Yatay Y loop filter deltası.
    pub delta_lf_y_h: i8,
    /// U loop filter deltası.
    pub delta_lf_u: i8,
    /// V loop filter deltası.
    pub delta_lf_v: i8,
    /// Referans kare indeksi.
    pub r#ref: i8,
    /// Skip bayrağı.
    pub skip: u8,
    /// Global MV bayrağı.
    pub globalmv: u8,
}

/// `Dav1dSegmentationDataSet` (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dSegmentationDataSet {
    /// 8 segmentin verileri.
    pub d: [Dav1dSegmentationData; DAV1D_MAX_SEGMENTS],
    /// Pre-skip bayrağı.
    pub preskip: u8,
    /// Son etkin segment kimliği.
    pub last_active_segid: i8,
}

/// `Dav1dLoopfilterModeRefDeltas` (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dLoopfilterModeRefDeltas {
    /// Mod deltaları.
    pub mode_delta: [i8; 2],
    /// Referans deltaları.
    pub ref_delta: [i8; DAV1D_TOTAL_REFS_PER_FRAME],
}

/// `Dav1dFilmGrainData` (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dFilmGrainData {
    /// Rastgelelik tohumu.
    pub seed: c_uint,
    /// Luma nokta sayısı.
    pub num_y_points: c_int,
    /// Luma ölçekleme noktaları.
    pub y_points: [[u8; 2]; 14],
    /// Kroma ölçeklemesi lumadan mı?
    pub chroma_scaling_from_luma: c_int,
    /// UV nokta sayıları.
    pub num_uv_points: [c_int; 2],
    /// UV ölçekleme noktaları.
    pub uv_points: [[[u8; 2]; 10]; 2],
    /// Ölçekleme kaydırma miktarı.
    pub scaling_shift: c_int,
    /// AR katsayı gecikmesi.
    pub ar_coeff_lag: c_int,
    /// Luma AR katsayıları.
    pub ar_coeffs_y: [i8; 24],
    /// UV AR katsayıları (`25 + 3` hizalama dolgulu).
    pub ar_coeffs_uv: [[i8; 28]; 2],
    /// AR katsayı kaydırma miktarı.
    pub ar_coeff_shift: u64,
    /// Grain ölçek kaydırma miktarı.
    pub grain_scale_shift: c_int,
    /// UV çarpanları.
    pub uv_mult: [c_int; 2],
    /// UV-luma çarpanları.
    pub uv_luma_mult: [c_int; 2],
    /// UV ofsetleri.
    pub uv_offset: [c_int; 2],
    /// Örtüşme bayrağı.
    pub overlap_flag: c_int,
    /// Kısıtlı aralığa kırpma bayrağı.
    pub clip_to_restricted_range: c_int,
}

/// `Dav1dFrameHeader.film_grain` anonim yapısı (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dFrameHeaderFilmGrain {
    /// Film grain verisi.
    pub data: Dav1dFilmGrainData,
    /// Film grain mevcut mu?
    pub present: u8,
    /// Film grain güncellendi mi?
    pub update: u8,
}

/// `struct Dav1dFrameHeaderOperatingPoint` (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dFrameHeaderOperatingPoint {
    /// Tampon kaldırma zamanı.
    pub buffer_removal_time: u32,
}

/// `Dav1dFrameHeader.super_res` anonim yapısı (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dFrameHeaderSuperRes {
    /// Genişlik ölçek paydası.
    pub width_scale_denominator: u8,
    /// Superres etkin mi?
    pub enabled: u8,
}

/// `Dav1dFrameHeader.tiling` anonim yapısı (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dav1dFrameHeaderTiling {
    /// Tek tip tile bayrağı.
    pub uniform: u8,
    /// Tile boyutu bayt sayısı.
    pub n_bytes: u8,
    /// Minimum log2 sütun.
    pub min_log2_cols: u8,
    /// Maksimum log2 sütun.
    pub max_log2_cols: u8,
    /// Log2 sütun.
    pub log2_cols: u8,
    /// Sütun sayısı.
    pub cols: u8,
    /// Minimum log2 satır.
    pub min_log2_rows: u8,
    /// Maksimum log2 satır.
    pub max_log2_rows: u8,
    /// Log2 satır.
    pub log2_rows: u8,
    /// Satır sayısı.
    pub rows: u8,
    /// Sütun süperblok başlangıçları.
    pub col_start_sb: [u16; DAV1D_MAX_TILE_COLS + 1],
    /// Satır süperblok başlangıçları.
    pub row_start_sb: [u16; DAV1D_MAX_TILE_ROWS + 1],
    /// CDF güncelleme tile indeksi.
    pub update: u16,
}

/// `Dav1dFrameHeader.quant` anonim yapısı (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dFrameHeaderQuant {
    /// Luma AC temel Q indeksi.
    pub yac: u8,
    /// Luma DC deltası.
    pub ydc_delta: i8,
    /// U DC deltası.
    pub udc_delta: i8,
    /// U AC deltası.
    pub uac_delta: i8,
    /// V DC deltası.
    pub vdc_delta: i8,
    /// V AC deltası.
    pub vac_delta: i8,
    /// Kuantizasyon matrisi bayrağı.
    pub qm: u8,
    /// Y kuantizasyon matrisi seviyesi.
    pub qm_y: u8,
    /// U kuantizasyon matrisi seviyesi.
    pub qm_u: u8,
    /// V kuantizasyon matrisi seviyesi.
    pub qm_v: u8,
}

/// `Dav1dFrameHeader.segmentation` anonim yapısı (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dFrameHeaderSegmentation {
    /// Segmentasyon etkin mi?
    pub enabled: u8,
    /// Harita güncelleniyor mu?
    pub update_map: u8,
    /// Zamansal güncelleme mi?
    pub temporal: u8,
    /// Veri güncelleniyor mu?
    pub update_data: u8,
    /// Segment veri kümesi.
    pub seg_data: Dav1dSegmentationDataSet,
    /// Segment başına kayıpsız bayrağı.
    pub lossless: [u8; DAV1D_MAX_SEGMENTS],
    /// Segment başına Q indeksi.
    pub qidx: [u8; DAV1D_MAX_SEGMENTS],
}

/// `Dav1dFrameHeader.delta.q` anonim yapısı (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dFrameHeaderDeltaQ {
    /// Delta-Q mevcut mu?
    pub present: u8,
    /// Çözünürlük log2 değeri.
    pub res_log2: u8,
}

/// `Dav1dFrameHeader.delta.lf` anonim yapısı (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dFrameHeaderDeltaLf {
    /// Delta-LF mevcut mu?
    pub present: u8,
    /// Çözünürlük log2 değeri.
    pub res_log2: u8,
    /// Çoklu düzlem bayrağı.
    pub multi: u8,
}

/// `Dav1dFrameHeader.delta` anonim yapısı (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dFrameHeaderDelta {
    /// Delta-Q parametreleri.
    pub q: Dav1dFrameHeaderDeltaQ,
    /// Delta-LF parametreleri.
    pub lf: Dav1dFrameHeaderDeltaLf,
}

/// `Dav1dFrameHeader.loopfilter` anonim yapısı (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dFrameHeaderLoopfilter {
    /// Luma filtre seviyeleri (`[dikey, yatay]`).
    pub level_y: [u8; 2],
    /// U filtre seviyesi.
    pub level_u: u8,
    /// V filtre seviyesi.
    pub level_v: u8,
    /// Mod/referans deltası etkin mi?
    pub mode_ref_delta_enabled: u8,
    /// Mod/referans deltası güncelleniyor mu?
    pub mode_ref_delta_update: u8,
    /// Mod/referans deltaları.
    pub mode_ref_deltas: Dav1dLoopfilterModeRefDeltas,
    /// Keskinlik parametresi.
    pub sharpness: u8,
}

/// `Dav1dFrameHeader.cdef` anonim yapısı (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dFrameHeaderCdef {
    /// Sönümleme (`damping`).
    pub damping: u8,
    /// Güç indeksi bit sayısı.
    pub n_bits: u8,
    /// Luma güç değerleri.
    pub y_strength: [u8; DAV1D_MAX_CDEF_STRENGTHS],
    /// UV güç değerleri.
    pub uv_strength: [u8; DAV1D_MAX_CDEF_STRENGTHS],
}

/// `Dav1dFrameHeader.restoration` anonim yapısı (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dFrameHeaderRestoration {
    /// Düzlem başına loop restoration tipi.
    pub r#type: [Dav1dRestorationType; 3],
    /// Birim boyutu log2 (`[y, uv]`).
    pub unit_size: [u8; 2],
}

/// `Dav1dFrameHeader` (`include/dav1d/headers.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dav1dFrameHeader {
    /// Film grain parametreleri.
    pub film_grain: Dav1dFrameHeaderFilmGrain,
    /// Kare tipi (`Dav1dFrameType`).
    pub frame_type: Dav1dFrameType,
    /// `[coded_width, superres_upscaled_width]`.
    pub width: [c_int; 2],
    /// Kare yüksekliği.
    pub height: c_int,
    /// Kare sıra ipucu (`order_hint`).
    pub frame_offset: u8,
    /// SVC temporal katman kimliği.
    pub temporal_id: u8,
    /// SVC spatial katman kimliği.
    pub spatial_id: u8,
    /// Mevcut kareyi göster bayrağı.
    pub show_existing_frame: u8,
    /// Gösterilecek mevcut referans yuvası indeksi.
    pub existing_frame_idx: u8,
    /// Kare kimlik numarası.
    pub frame_id: u32,
    /// Sunum gecikmesi.
    pub frame_presentation_delay: u32,
    /// Kare görünür mü?
    pub show_frame: u8,
    /// Kare daha sonra gösterilebilir mi?
    pub showable_frame: u8,
    /// Hata dayanıklılık kipi bayrağı.
    pub error_resilient_mode: u8,
    /// CDF güncelleme devre dışı bayrağı.
    pub disable_cdf_update: u8,
    /// Ekran içeriği araçları izni.
    pub allow_screen_content_tools: u8,
    /// Tamsayı hareket vektörü zorlaması.
    pub force_integer_mv: u8,
    /// Kare boyutu geçersiz kılma bayrağı.
    pub frame_size_override: u8,
    /// Birincil referans kare (`7 = NONE`).
    pub primary_ref_frame: u8,
    /// Tampon kaldırma zamanı varlık bayrağı.
    pub buffer_removal_time_present: u8,
    /// Operating point başına tampon kaldırma zamanları.
    pub operating_points: [Dav1dFrameHeaderOperatingPoint; DAV1D_MAX_OPERATING_POINTS],
    /// Yenilenen referans yuvası bit maskesi.
    pub refresh_frame_flags: u8,
    /// Render genişliği.
    pub render_width: c_int,
    /// Render yüksekliği.
    pub render_height: c_int,
    /// Superres parametreleri.
    pub super_res: Dav1dFrameHeaderSuperRes,
    /// Render boyutu farklı mı?
    pub have_render_size: u8,
    /// IntraBC izni.
    pub allow_intrabc: u8,
    /// Kısa referans sinyallemesi.
    pub frame_ref_short_signaling: u8,
    /// 7 aktif referansın yuva indeksleri.
    pub refidx: [i8; DAV1D_REFS_PER_FRAME],
    /// Yüksek hassasiyetli MV bayrağı.
    pub hp: u8,
    /// Alt-piksel filtre kipi (`Dav1dFilterMode`).
    pub subpel_filter_mode: Dav1dFilterMode,
    /// Seçilebilir hareket kipi bayrağı.
    pub switchable_motion_mode: u8,
    /// Zamansal referans MV kullanımı.
    pub use_ref_frame_mvs: u8,
    /// Kare sonu bağlam yenileme bayrağı.
    pub refresh_context: u8,
    /// Tile bölümleme parametreleri.
    pub tiling: Dav1dFrameHeaderTiling,
    /// Kuantizasyon parametreleri.
    pub quant: Dav1dFrameHeaderQuant,
    /// Segmentasyon parametreleri.
    pub segmentation: Dav1dFrameHeaderSegmentation,
    /// Delta-Q / Delta-LF parametreleri.
    pub delta: Dav1dFrameHeaderDelta,
    /// Tüm segmentler kayıpsız mı?
    pub all_lossless: u8,
    /// Deblocking loop filter parametreleri.
    pub loopfilter: Dav1dFrameHeaderLoopfilter,
    /// CDEF parametreleri.
    pub cdef: Dav1dFrameHeaderCdef,
    /// Loop restoration parametreleri.
    pub restoration: Dav1dFrameHeaderRestoration,
    /// Dönüşüm boyutu kipi (`Dav1dTxfmMode`).
    pub txfm_mode: Dav1dTxfmMode,
    /// Bileşik referans seçimi bayrağı.
    pub switchable_comp_refs: u8,
    /// Skip mode izni.
    pub skip_mode_allowed: u8,
    /// Skip mode etkin mi?
    pub skip_mode_enabled: u8,
    /// Skip mode referans indeksleri.
    pub skip_mode_refs: [i8; 2],
    /// Yerel warped motion izni.
    pub warp_motion: u8,
    /// Kısıtlı dönüşüm kümesi bayrağı.
    pub reduced_txtp_set: u8,
    /// 7 referans için global warped motion parametreleri.
    pub gmv: [Dav1dWarpedMotionParams; DAV1D_REFS_PER_FRAME],
}

/// `Dav1dUserData` (`include/dav1d/common.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Dav1dUserData {
    /// Kullanıcı verisi işaretçisi.
    pub data: *const u8,
    /// Referans sayımlı yaşam süresi nesnesi.
    pub r#ref: *mut Dav1dRef,
}

impl Default for Dav1dUserData {
    fn default() -> Self {
        Self {
            data: core::ptr::null(),
            r#ref: core::ptr::null_mut(),
        }
    }
}

/// `Dav1dDataProps` (`include/dav1d/common.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Dav1dDataProps {
    /// Kapsayıcı zaman damgası (varsayılan `i64::MIN`).
    pub timestamp: i64,
    /// Kapsayıcı süresi (varsayılan `0`).
    pub duration: i64,
    /// Akış bayt ofseti (varsayılan `-1`).
    pub offset: i64,
    /// Paket boyutu (bayt).
    pub size: usize,
    /// Kullanıcı tanımlı referans sayımlı veri.
    pub user_data: Dav1dUserData,
}

impl Default for Dav1dDataProps {
    fn default() -> Self {
        Self {
            timestamp: i64::MIN,
            duration: 0,
            offset: -1,
            size: 0,
            user_data: Dav1dUserData::default(),
        }
    }
}

/// `Dav1dPictureParameters` (`include/dav1d/picture.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dav1dPictureParameters {
    /// Genişlik (piksel).
    pub w: c_int,
    /// Yükseklik (piksel).
    pub h: c_int,
    /// Piksel yerleşimi (`Dav1dPixelLayout`).
    pub layout: Dav1dPixelLayout,
    /// Bileşen başına bit sayısı (`8`, `10` veya `12`).
    pub bpc: c_int,
}

/// `Dav1dPicture` (`include/dav1d/picture.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Dav1dPicture {
    /// Etkin sequence header işaretçisi.
    pub seq_hdr: *mut Dav1dSequenceHeader,
    /// Kare başlığı işaretçisi.
    pub frame_hdr: *mut Dav1dFrameHeader,
    /// Düzlemsel piksel verisi işaretçileri (`[Y, U, V]`).
    pub data: [*mut c_void; 3],
    /// Satır adımları (bayt cinsinden: `[luma, chroma]`).
    pub stride: [isize; 2],
    /// Resim boyut ve biçim parametreleri.
    pub p: Dav1dPictureParameters,
    /// Girdi paketinden aktarılan metaveri.
    pub m: Dav1dDataProps,
    /// HDR Content Light Level metaverisi (varsa).
    pub content_light: *mut Dav1dContentLightLevel,
    /// HDR Mastering Display metaverisi (varsa).
    pub mastering_display: *mut Dav1dMasteringDisplay,
    /// ITU-T T.35 metaveri dizisi (varsa).
    pub itut_t35: *mut Dav1dITUTT35,
    /// ITU-T T.35 girdi sayısı.
    pub n_itut_t35: usize,
    /// İleride kullanım için ayrılmış alan.
    pub reserved: [usize; 4],
    /// `frame_hdr` tahsis referansı.
    pub frame_hdr_ref: *mut Dav1dRef,
    /// `seq_hdr` tahsis referansı.
    pub seq_hdr_ref: *mut Dav1dRef,
    /// `content_light` tahsis referansı.
    pub content_light_ref: *mut Dav1dRef,
    /// `mastering_display` tahsis referansı.
    pub mastering_display_ref: *mut Dav1dRef,
    /// `itut_t35` tahsis referansı.
    pub itut_t35_ref: *mut Dav1dRef,
    /// İleride kullanım için ayrılmış referans alanı.
    pub reserved_ref: [usize; 4],
    /// Düzlem piksel verisi tahsis referansı.
    pub r#ref: *mut Dav1dRef,
    /// Özel resim ayırıcı tarafından yönetilen işaretçi.
    pub allocator_data: *mut c_void,
}

impl Default for Dav1dPicture {
    fn default() -> Self {
        Self {
            seq_hdr: core::ptr::null_mut(),
            frame_hdr: core::ptr::null_mut(),
            data: [core::ptr::null_mut(); 3],
            stride: [0; 2],
            p: Dav1dPictureParameters::default(),
            m: Dav1dDataProps::default(),
            content_light: core::ptr::null_mut(),
            mastering_display: core::ptr::null_mut(),
            itut_t35: core::ptr::null_mut(),
            n_itut_t35: 0,
            reserved: [0; 4],
            frame_hdr_ref: core::ptr::null_mut(),
            seq_hdr_ref: core::ptr::null_mut(),
            content_light_ref: core::ptr::null_mut(),
            mastering_display_ref: core::ptr::null_mut(),
            itut_t35_ref: core::ptr::null_mut(),
            reserved_ref: [0; 4],
            r#ref: core::ptr::null_mut(),
            allocator_data: core::ptr::null_mut(),
        }
    }
}

/// Özel resim ayırma geri çağrısı (`int (*alloc_picture_callback)(Dav1dPicture *pic, void *cookie)`).
pub type Dav1dAllocPictureCallback =
    Option<extern "C" fn(pic: *mut Dav1dPicture, cookie: *mut c_void) -> c_int>;

/// Özel resim serbest bırakma geri çağrısı (`void (*release_picture_callback)(Dav1dPicture *pic, void *cookie)`).
pub type Dav1dReleasePictureCallback =
    Option<extern "C" fn(pic: *mut Dav1dPicture, cookie: *mut c_void)>;

/// `Dav1dPicAllocator` (`include/dav1d/picture.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Dav1dPicAllocator {
    /// Geri çağrılara iletilen opak kullanıcı işaretçisi.
    pub cookie: *mut c_void,
    /// Resim tamponu ayırma geri çağrısı.
    pub alloc_picture_callback: Dav1dAllocPictureCallback,
    /// Resim tamponu serbest bırakma geri çağrısı.
    pub release_picture_callback: Dav1dReleasePictureCallback,
}

impl Default for Dav1dPicAllocator {
    fn default() -> Self {
        Self {
            cookie: core::ptr::null_mut(),
            alloc_picture_callback: None,
            release_picture_callback: None,
        }
    }
}

/// Sarılmış tampon veya kullanıcı verisi serbest bırakma geri çağrısı (`void (*free_callback)(const uint8_t *buf, void *cookie)`).
pub type Dav1dFreeCallback = Option<extern "C" fn(data: *const u8, cookie: *mut c_void)>;

/// `Dav1dData` (`include/dav1d/data.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Dav1dData {
    /// OBU veri işaretçisi.
    pub data: *const u8,
    /// Veri boyutu (bayt).
    pub sz: usize,
    /// Referans sayımlı yaşam süresi nesnesi.
    pub r#ref: *mut Dav1dRef,
    /// Kullanıcı metaverisi.
    pub m: Dav1dDataProps,
}

impl Default for Dav1dData {
    fn default() -> Self {
        Self {
            data: core::ptr::null(),
            sz: 0,
            r#ref: core::ptr::null_mut(),
            m: Dav1dDataProps::default(),
        }
    }
}

/// `Dav1dLogger` (`include/dav1d/dav1d.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Dav1dLogger {
    /// Günlük geri çağrısına iletilen kullanıcı işaretçisi.
    pub cookie: *mut c_void,
    /// `vprintf` uyumlu C geri çağrı işaretçisi (`NULL` = kapalı).
    pub callback: *const c_void,
}

impl Default for Dav1dLogger {
    fn default() -> Self {
        Self {
            cookie: core::ptr::null_mut(),
            callback: core::ptr::null(),
        }
    }
}

/// `enum Dav1dInloopFilterType` (`include/dav1d/dav1d.h`).
pub type Dav1dInloopFilterType = c_uint;
/// `DAV1D_INLOOPFILTER_NONE`.
pub const DAV1D_INLOOPFILTER_NONE: Dav1dInloopFilterType = 0;
/// `DAV1D_INLOOPFILTER_DEBLOCK`.
pub const DAV1D_INLOOPFILTER_DEBLOCK: Dav1dInloopFilterType = 1 << 0;
/// `DAV1D_INLOOPFILTER_CDEF`.
pub const DAV1D_INLOOPFILTER_CDEF: Dav1dInloopFilterType = 1 << 1;
/// `DAV1D_INLOOPFILTER_RESTORATION`.
pub const DAV1D_INLOOPFILTER_RESTORATION: Dav1dInloopFilterType = 1 << 2;
/// `DAV1D_INLOOPFILTER_ALL`.
pub const DAV1D_INLOOPFILTER_ALL: Dav1dInloopFilterType =
    DAV1D_INLOOPFILTER_DEBLOCK | DAV1D_INLOOPFILTER_CDEF | DAV1D_INLOOPFILTER_RESTORATION;

/// `enum Dav1dDecodeFrameType` (`include/dav1d/dav1d.h`).
pub type Dav1dDecodeFrameType = c_uint;
/// `DAV1D_DECODEFRAMETYPE_ALL`.
pub const DAV1D_DECODEFRAMETYPE_ALL: Dav1dDecodeFrameType = 0;
/// `DAV1D_DECODEFRAMETYPE_REFERENCE`.
pub const DAV1D_DECODEFRAMETYPE_REFERENCE: Dav1dDecodeFrameType = 1;
/// `DAV1D_DECODEFRAMETYPE_INTRA`.
pub const DAV1D_DECODEFRAMETYPE_INTRA: Dav1dDecodeFrameType = 2;
/// `DAV1D_DECODEFRAMETYPE_KEY`.
pub const DAV1D_DECODEFRAMETYPE_KEY: Dav1dDecodeFrameType = 3;

/// `enum Dav1dEventFlags` (`include/dav1d/dav1d.h`).
pub type Dav1dEventFlags = c_uint;
/// `DAV1D_EVENT_FLAG_NEW_SEQUENCE`.
pub const DAV1D_EVENT_FLAG_NEW_SEQUENCE: Dav1dEventFlags = 1 << 0;
/// `DAV1D_EVENT_FLAG_NEW_OP_PARAMS_INFO`.
pub const DAV1D_EVENT_FLAG_NEW_OP_PARAMS_INFO: Dav1dEventFlags = 1 << 1;

/// `Dav1dSettings` (`include/dav1d/dav1d.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Dav1dSettings {
    /// İşçi thread sayısı (`0 = otomatik`).
    pub n_threads: c_int,
    /// Maksimum kare gecikmesi (`0 = otomatik`, `1 = düşük gecikme`).
    pub max_frame_delay: c_int,
    /// Çıktı karelerine film grain uygula (`varsayılan = 1`).
    pub apply_grain: c_int,
    /// Ölçeklenebilir akışlar için operating point seçimi (`0..=31`).
    pub operating_point: c_int,
    /// Tüm spatial katmanları çıkar (`varsayılan = 1`).
    pub all_layers: c_int,
    /// Piksel cinsinden maksimum kare boyutu (`0 = sınırsız`).
    pub frame_size_limit: c_uint,
    /// Özel resim tamponu ayırıcı.
    pub allocator: Dav1dPicAllocator,
    /// Günlük geri çağrısı.
    pub logger: Dav1dLogger,
    /// Katı standart uyumluluğu (`varsayılan = 0`).
    pub strict_std_compliance: c_int,
    /// Görünmez kareleri de çıktı olarak ver (`varsayılan = 0`).
    pub output_invisible_frames: c_int,
    /// Etkin halka içi filtreler (`varsayılan = DAV1D_INLOOPFILTER_ALL`).
    pub inloop_filters: Dav1dInloopFilterType,
    /// Çözülecek kare tipleri (`varsayılan = DAV1D_DECODEFRAMETYPE_ALL`).
    pub decode_frame_type: Dav1dDecodeFrameType,
    /// İleride kullanım için ayrılmış alan.
    pub reserved: [u8; 16],
}

impl Default for Dav1dSettings {
    fn default() -> Self {
        Self {
            n_threads: 0,
            max_frame_delay: 0,
            apply_grain: 1,
            operating_point: 0,
            all_layers: 1,
            frame_size_limit: 0,
            allocator: Dav1dPicAllocator::default(),
            logger: Dav1dLogger::default(),
            strict_std_compliance: 0,
            output_invisible_frames: 0,
            inloop_filters: DAV1D_INLOOPFILTER_ALL,
            decode_frame_type: DAV1D_DECODEFRAMETYPE_ALL,
            reserved: [0; 16],
        }
    }
}
