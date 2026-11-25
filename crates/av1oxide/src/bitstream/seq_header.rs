//! AV1 sequence header OBU'sunun tam ayrıştırması (F2a / S2a.1).
//!
//! ## Sorumluluk
//!
//! `sequence_header_obu()` yükünün **bütün anlamsal alanları**; dav1d
//! `include/dav1d/headers.h` içindeki `Dav1dSequenceHeader` yapısının Rust
//! karşılığı. Operating point'ler, timing/decoder model bilgisi, renk
//! yapılandırması, araç bayrakları ve film grain varlığı dâhildir.
//!
//! ## Upstream karşılığı ve spesifikasyon eşlemesi
//!
//! Aşağıdaki tablo, **her ayrıştırılan alan grubu** için dav1d kaynak satırını
//! (`99f3354ed0b9b4bf3c7c9347318dbcc522c94af5`, salt-okunur) ve AV1
//! spesifikasyonunun ilgili bölümünü verir. Satır numaraları `src/obu.c`
//! içindeki `parse_seq_hdr()` gövdesine aittir (fonksiyon 72–300).
//!
//! | Alan grubu | dav1d `src/obu.c` | AV1 spesifikasyonu |
//! |---|---|---|
//! | `seq_profile`, `still_picture`, `reduced_still_picture_header` | 83–92 | §5.5.1 `sequence_header_obu()`, §6.5.1 |
//! | `seq_level_idx`, `seq_tier` (reduced yol) | 99–102 | §5.5.1, §6.5.1 |
//! | `timing_info_present_flag`, `num_units_in_display_tick`, `time_scale` | 104–109 | §5.5.3 `timing_info()`, §6.5.3 |
//! | `equal_picture_interval`, `num_ticks_per_picture_minus_1` (`uvlc()`) | 110–116 | §5.5.3, §6.5.3 |
//! | `decoder_model_info_present_flag`, gecikme uzunlukları, `num_units_in_decoding_tick` | 118–126 | §5.5.4 `decoder_model_info()`, §6.5.4 |
//! | `display_model_info_present_flag` | 133 | §5.5.1, §6.5.1 |
//! | `operating_points_cnt_minus_1` | 134 | §5.5.1, §6.5.1 |
//! | `operating_point_idc` | 138–139 | §5.5.1, §6.5.1 |
//! | `seq_level_idx`, `seq_tier` | 141–144 | §5.5.1, §6.5.1 |
//! | `decoder_model_present_for_this_op`, `operating_parameters_info()` | 145–153 | §5.5.5, §6.5.5 |
//! | `display_model_present_for_this_op`, `initial_display_delay_minus_1` | 157–160 | §5.5.1, §6.5.1 |
//! | `frame_width_bits_minus_1`, `frame_height_bits_minus_1`, `max_frame_width_minus_1`, `max_frame_height_minus_1` | 168–171 | §5.5.1, §6.5.1 |
//! | `frame_id_numbers_present_flag`, `delta_frame_id_minus_1`, `additional_frame_id_length_minus_1` | 177–180 | §5.5.1, §6.5.1 |
//! | `use_128x128_superblock`, `enable_filter_intra`, `enable_intra_edge_filter` | 188–190 | §5.5.1, §6.5.1 |
//! | inter/intra araç bayrakları (`enable_interintra_compound` … `enable_ref_frame_mvs`) | 195–202 | §5.5.1, §6.5.1 |
//! | `seq_choose_screen_content_tools`, `seq_force_screen_content_tools` | 204 | §5.5.1, §6.5.1 |
//! | `seq_force_integer_mv` | 209–210 | §5.5.1, §6.5.1 |
//! | `order_hint_bits_minus_1` | 212 | §5.5.1, §6.5.1 |
//! | `enable_superres`, `enable_cdef`, `enable_restoration` | 214–216 | §5.5.1, §6.5.1 |
//! | `high_bitdepth`, `twelve_bit` (profil 2) | 222–225 | §5.5.1, §6.5.1 |
//! | `mono_chrome` | 226 | §5.5.1, §6.5.1 |
//! | `color_description_present_flag`, `color_primaries`, `transfer_characteristics`, `matrix_coefficients` | 227–235 | §5.5.2 `color_config()`, §6.5.2 |
//! | `color_range`, `subsampling_x`, `subsampling_y`, `chroma_sample_position` | 237–273 | §5.5.2, §6.5.2 |
//! | `separate_uv_delta_q` | 280–281 | §5.5.1, §6.5.1 |
//! | `film_grain_params_present` | 286 | §5.5.1, §6.5.1 |
//! | `trailing_bits()` | 298, 48–70 | §5.3.5, §6.4.4 |
//!
//! ## Reddetme kuralları (dav1d ile aynı)
//!
//! `strict_std_compliance = 0` dav1d'nin genel API'sinin varsayılanıdır
//! (`dav1d_parse_sequence_header`, obu.c 302–339). Bu kipte dav1d'nin
//! **her zaman** reddettiği değerler:
//!
//! - `profile > 2` (satır 83–84),
//! - `reduced_still_picture_header && !still_picture` (92),
//! - sıfır olmayan `operating_point_idc` için alt/üst baytın sıfır olması
//!   (138–139),
//! - BT.709 + sRGB + identity renk yapılandırmasında profile uyumsuzluğu
//!   (248–249).
//!
//! Yalnız `strict_std_compliance = 1` iken reddedilenler:
//! `num_units_in_display_tick == 0`, `time_scale == 0`,
//! `num_units_in_decoding_tick == 0` (108–109, 122–123) ve identity matrix ile
//! 4:4:4 dışı layout (274–279). `strict` seçeneği [`SequenceHeaderParseOptions`]
//! ile verilir; varsayılan `false` (dav1d genel API'siyle aynı).

#![forbid(unsafe_code)]

use crate::bitstream::reader::BitReader;
use crate::error::{BitstreamError, BitstreamErrorKind, Error, Result};

/// `Dav1dSequenceHeader` içindeki operating point dizisinin boyutu
/// (dav1d `DAV1D_MAX_OPERATING_POINTS`, headers.h 240).
pub const MAX_OPERATING_POINTS: usize = 32;

/// Bilinmeyen renk primaries değeri (dav1d `DAV1D_COLOR_PRI_UNKNOWN`).
pub const COLOR_PRI_UNKNOWN: u8 = 2;
/// BT.709 renk primaries değeri (dav1d `DAV1D_COLOR_PRI_BT709`).
pub const COLOR_PRI_BT709: u8 = 1;
/// Bilinmeyen transfer karakteristiği (dav1d `DAV1D_TRC_UNKNOWN`).
pub const TRC_UNKNOWN: u8 = 2;
/// sRGB transfer karakteristiği (dav1d `DAV1D_TRC_SRGB`).
pub const TRC_SRGB: u8 = 13;
/// Identity matrix katsayısı (dav1d `DAV1D_MC_IDENTITY`).
pub const MC_IDENTITY: u8 = 0;
/// Bilinmeyen matrix katsayısı (dav1d `DAV1D_MC_UNKNOWN`).
pub const MC_UNKNOWN: u8 = 2;

/// dav1d `enum Dav1dAdaptiveBoolean` (headers.h 78–83).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AdaptiveBoolean {
    /// 0: kapalı.
    #[default]
    Off,
    /// 1: açık.
    On,
    /// 2: uyarlanabilir (kare başlığı karar verir).
    Adaptive,
}

impl AdaptiveBoolean {
    /// dav1d enum değeri (0/1/2).
    #[must_use]
    pub const fn raw(self) -> u8 {
        match self {
            Self::Off => 0,
            Self::On => 1,
            Self::Adaptive => 2,
        }
    }
}

/// dav1d `enum Dav1dPixelLayout` (headers.h 114–119).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PixelLayout {
    /// 0: monochrome.
    #[default]
    I400,
    /// 1: 4:2:0.
    I420,
    /// 2: 4:2:2.
    I422,
    /// 3: 4:4:4.
    I444,
}

impl PixelLayout {
    /// dav1d enum değeri (0/1/2/3).
    #[must_use]
    pub const fn raw(self) -> u8 {
        match self {
            Self::I400 => 0,
            Self::I420 => 1,
            Self::I422 => 2,
            Self::I444 => 3,
        }
    }
}

/// dav1d `enum Dav1dChromaSamplePosition` (headers.h 177–182).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ChromaSamplePosition {
    /// 0: bilinmiyor.
    #[default]
    Unknown,
    /// 1: dikey konumlu.
    Vertical,
    /// 2: luma (0,0) ile eş konumlu.
    Colocated,
    /// 3: ayrılmış değer (dav1d enum'unda karşılığı yok, kabul edilir).
    Reserved3,
}

impl ChromaSamplePosition {
    /// 2 bit ham değerden konum.
    #[must_use]
    pub const fn from_raw(raw: u8) -> Self {
        match raw & 0x03 {
            0 => Self::Unknown,
            1 => Self::Vertical,
            2 => Self::Colocated,
            _ => Self::Reserved3,
        }
    }

    /// dav1d enum değeri (0/1/2/3).
    #[must_use]
    pub const fn raw(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::Vertical => 1,
            Self::Colocated => 2,
            Self::Reserved3 => 3,
        }
    }
}

/// Bir operating point'in sequence header bilgisi
/// (dav1d `Dav1dSequenceHeaderOperatingPoint`, headers.h 234–241).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct OperatingPoint {
    /// `seq_level_idx` üst 3 biti + 2 (reduced yolda 3 bitin kendisi).
    pub major_level: u8,
    /// `seq_level_idx` alt 2 biti.
    pub minor_level: u8,
    /// `initial_display_delay_minus_1 + 1`; parametre yoksa 10.
    pub initial_display_delay: u8,
    /// `operating_point_idc` (12 bit).
    pub idc: u16,
    /// `seq_tier` (yalnız `major_level > 3` iken okunur).
    pub tier: u8,
    /// Bu operating point için decoder model parametreleri var mı?
    pub decoder_model_param_present: bool,
    /// Bu operating point için display model parametreleri var mı?
    pub display_model_param_present: bool,
}

/// `operating_parameters_info()` (dav1d `Dav1dSequenceHeaderOperatingParameterInfo`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct OperatingParameterInfo {
    /// `decoder_buffer_delay` (encoder_decoder_buffer_delay_length bit).
    pub decoder_buffer_delay: u32,
    /// `encoder_buffer_delay` (encoder_decoder_buffer_delay_length bit).
    pub encoder_buffer_delay: u32,
    /// `low_delay_mode_flag`.
    pub low_delay_mode: bool,
}

/// Sequence header ayrıştırma seçenekleri.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SequenceHeaderParseOptions {
    /// dav1d `strict_std_compliance`: yalnız katı kipte reddedilen kuralları açar.
    pub strict_std_compliance: bool,
}

/// `Dav1dSequenceHeader`'ın Rust karşılığı (headers.h 194–290).
///
/// Alan adları ve anlamları dav1d yapısıyla **bire bir** eşlenir; böylece
/// differential karşılaştırıcı alan alan eşleştirebilir. `hbd` dav1d'deki gibi
/// 0/1/2'dir (spec'in `high_bitdepth`/`twelve_bit` ikilisinin birleşimi).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceHeader {
    /// `seq_profile` (0–2).
    pub profile: u8,
    /// `max_frame_width_minus_1 + 1`.
    pub max_width: i32,
    /// `max_frame_height_minus_1 + 1`.
    pub max_height: i32,
    /// Piksel düzeni (dav1d türetilmiş alan).
    pub layout: PixelLayout,
    /// `color_primaries` (dav1d enum değeri; renk tanımı yoksa 2).
    pub pri: u8,
    /// `transfer_characteristics` (renk tanımı yoksa 2).
    pub trc: u8,
    /// `matrix_coefficients` (renk tanımı yoksa 2).
    pub mtrx: u8,
    /// `chroma_sample_position` (yalnız 4:2:0'da okunur).
    pub chr: ChromaSamplePosition,
    /// 0 = 8 bit, 1 = 10 bit, 2 = 12 bit (dav1d birleşik alanı).
    pub hbd: u8,
    /// `color_range` (0 = MPEG, 1 = JPEG).
    pub color_range: u8,
    /// Operating point sayısı (1–32).
    pub num_operating_points: u8,
    /// Operating point tablosu (yalnız ilk `num_operating_points` anlamlı).
    pub operating_points: [OperatingPoint; MAX_OPERATING_POINTS],
    /// `still_picture`.
    pub still_picture: bool,
    /// `reduced_still_picture_header`.
    pub reduced_still_picture_header: bool,
    /// `timing_info_present_flag`.
    pub timing_info_present: bool,
    /// `num_units_in_display_tick`.
    pub num_units_in_tick: u32,
    /// `time_scale`.
    pub time_scale: u32,
    /// `equal_picture_interval`.
    pub equal_picture_interval: bool,
    /// `num_ticks_per_picture_minus_1 + 1`.
    pub num_ticks_per_picture: u32,
    /// `decoder_model_info_present_flag`.
    pub decoder_model_info_present: bool,
    /// `encoder_decoder_buffer_delay_length_minus_1 + 1`.
    pub encoder_decoder_buffer_delay_length: u8,
    /// `num_units_in_decoding_tick`.
    pub num_units_in_decoding_tick: u32,
    /// `buffer_removal_delay_length_minus_1 + 1`.
    pub buffer_removal_delay_length: u8,
    /// `frame_presentation_delay_length_minus_1 + 1`.
    pub frame_presentation_delay_length: u8,
    /// `display_model_info_present_flag`.
    pub display_model_info_present: bool,
    /// `frame_width_bits_minus_1 + 1`.
    pub width_n_bits: u8,
    /// `frame_height_bits_minus_1 + 1`.
    pub height_n_bits: u8,
    /// `frame_id_numbers_present_flag`.
    pub frame_id_numbers_present: bool,
    /// `delta_frame_id_minus_1 + 2`.
    pub delta_frame_id_n_bits: u8,
    /// `additional_frame_id_length_minus_1 + delta_frame_id_n_bits + 1`.
    pub frame_id_n_bits: u8,
    /// `use_128x128_superblock`.
    pub sb128: bool,
    /// `enable_filter_intra`.
    pub filter_intra: bool,
    /// `enable_intra_edge_filter`.
    pub intra_edge_filter: bool,
    /// `enable_interintra_compound`.
    pub inter_intra: bool,
    /// `enable_masked_compound`.
    pub masked_compound: bool,
    /// `enable_warped_motion`.
    pub warped_motion: bool,
    /// `enable_dual_filter`.
    pub dual_filter: bool,
    /// `enable_order_hint`.
    pub order_hint: bool,
    /// `enable_jnt_comp` (yalnız `order_hint` ile).
    pub jnt_comp: bool,
    /// `enable_ref_frame_mvs` (yalnız `order_hint` ile).
    pub ref_frame_mvs: bool,
    /// `seq_choose_screen_content_tools`/`seq_force_screen_content_tools`.
    pub screen_content_tools: AdaptiveBoolean,
    /// `seq_force_integer_mv`.
    pub force_integer_mv: AdaptiveBoolean,
    /// `order_hint_bits_minus_1 + 1` (yalnız `order_hint` ile).
    pub order_hint_n_bits: u8,
    /// `enable_superres`.
    pub super_res: bool,
    /// `enable_cdef`.
    pub cdef: bool,
    /// `enable_restoration`.
    pub restoration: bool,
    /// `subsampling_x` (dav1d türetilmiş alan).
    pub ss_hor: u8,
    /// `subsampling_y` (dav1d türetilmiş alan).
    pub ss_ver: u8,
    /// `mono_chrome`.
    pub monochrome: bool,
    /// `color_description_present_flag`.
    pub color_description_present: bool,
    /// `separate_uv_delta_q`.
    pub separate_uv_delta_q: bool,
    /// `film_grain_params_present`.
    pub film_grain_present: bool,
    /// Operating point başına decoder model parametreleri.
    pub operating_parameter_info: [OperatingParameterInfo; MAX_OPERATING_POINTS],
}

impl Default for SequenceHeader {
    fn default() -> Self {
        Self {
            profile: 0,
            max_width: 0,
            max_height: 0,
            layout: PixelLayout::I400,
            pri: 0,
            trc: 0,
            mtrx: 0,
            chr: ChromaSamplePosition::Unknown,
            hbd: 0,
            color_range: 0,
            num_operating_points: 0,
            operating_points: [OperatingPoint::default(); MAX_OPERATING_POINTS],
            still_picture: false,
            reduced_still_picture_header: false,
            timing_info_present: false,
            num_units_in_tick: 0,
            time_scale: 0,
            equal_picture_interval: false,
            num_ticks_per_picture: 0,
            decoder_model_info_present: false,
            encoder_decoder_buffer_delay_length: 0,
            num_units_in_decoding_tick: 0,
            buffer_removal_delay_length: 0,
            frame_presentation_delay_length: 0,
            display_model_info_present: false,
            width_n_bits: 0,
            height_n_bits: 0,
            frame_id_numbers_present: false,
            delta_frame_id_n_bits: 0,
            frame_id_n_bits: 0,
            sb128: false,
            filter_intra: false,
            intra_edge_filter: false,
            inter_intra: false,
            masked_compound: false,
            warped_motion: false,
            dual_filter: false,
            order_hint: false,
            jnt_comp: false,
            ref_frame_mvs: false,
            screen_content_tools: AdaptiveBoolean::Off,
            force_integer_mv: AdaptiveBoolean::Off,
            order_hint_n_bits: 0,
            super_res: false,
            cdef: false,
            restoration: false,
            ss_hor: 0,
            ss_ver: 0,
            monochrome: false,
            color_description_present: false,
            separate_uv_delta_q: false,
            film_grain_present: false,
            operating_parameter_info: [OperatingParameterInfo::default(); MAX_OPERATING_POINTS],
        }
    }
}

/// Sequence header kural ihlali (konumlu, `&'static str` gerekçeli).
fn invalid(reason: &'static str, bit_offset: usize) -> Error {
    Error::Bitstream(BitstreamError::new(
        BitstreamErrorKind::InvalidSequenceHeader(reason),
        bit_offset,
    ))
}

impl SequenceHeader {
    /// Sequence header yükünü dav1d'nin varsayılan (strict olmayan) kuralıyla ayrıştırır.
    ///
    /// `payload` OBU başlığı ve boyut alanı **hariç** yüktür (dav1d
    /// `parse_seq_hdr` çağrısındaki `gb` konumu, obu.c 1210–1216).
    ///
    /// # Hatalar
    ///
    /// Alan sınırı/kesik veri [`BitstreamErrorKind::UnexpectedEnd`]; dav1d'nin
    /// reddettiği değerler [`BitstreamErrorKind::InvalidSequenceHeader`].
    pub fn parse(payload: &[u8]) -> Result<Self> {
        Self::parse_with(payload, SequenceHeaderParseOptions::default())
    }

    /// Sequence header yükünü verilen seçeneklerle ayrıştırır.
    ///
    /// # Hatalar
    ///
    /// [`Self::parse`] ile aynı; `strict_std_compliance` ek katı kuralları açar.
    pub fn parse_with(payload: &[u8], options: SequenceHeaderParseOptions) -> Result<Self> {
        let mut br = BitReader::new(payload);
        let mut hdr = Self::default();
        let strict = options.strict_std_compliance;

        hdr.profile = br.read_bits(3)? as u8;
        if hdr.profile > 2 {
            return Err(invalid(
                "seq_profile > 2 (AV1 §6.5.1; dav1d obu.c 83-84)",
                br.bit_position(),
            ));
        }
        hdr.still_picture = br.read_bit()? != 0;
        hdr.reduced_still_picture_header = br.read_bit()? != 0;
        if hdr.reduced_still_picture_header && !hdr.still_picture {
            return Err(invalid(
                "reduced_still_picture_header && !still_picture (dav1d obu.c 92)",
                br.bit_position(),
            ));
        }

        if hdr.reduced_still_picture_header {
            hdr.num_operating_points = 1;
            hdr.operating_points[0].major_level = br.read_bits(3)? as u8;
            hdr.operating_points[0].minor_level = br.read_bits(2)? as u8;
            hdr.operating_points[0].initial_display_delay = 10;
        } else {
            hdr.timing_info_present = br.read_bit()? != 0;
            if hdr.timing_info_present {
                hdr.num_units_in_tick = br.read_bits(32)?;
                hdr.time_scale = br.read_bits(32)?;
                if strict && (hdr.num_units_in_tick == 0 || hdr.time_scale == 0) {
                    return Err(invalid(
                        "timing_info: num_units_in_display_tick/time_scale 0 (dav1d obu.c 108-109)",
                        br.bit_position(),
                    ));
                }
                hdr.equal_picture_interval = br.read_bit()? != 0;
                if hdr.equal_picture_interval {
                    let num_ticks_per_picture = br.read_uvlc()?;
                    hdr.num_ticks_per_picture = num_ticks_per_picture + 1;
                }
                hdr.decoder_model_info_present = br.read_bit()? != 0;
                if hdr.decoder_model_info_present {
                    hdr.encoder_decoder_buffer_delay_length = br.read_bits(5)? as u8 + 1;
                    hdr.num_units_in_decoding_tick = br.read_bits(32)?;
                    if strict && hdr.num_units_in_decoding_tick == 0 {
                        return Err(invalid(
                            "num_units_in_decoding_tick 0 (dav1d obu.c 122-123)",
                            br.bit_position(),
                        ));
                    }
                    hdr.buffer_removal_delay_length = br.read_bits(5)? as u8 + 1;
                    hdr.frame_presentation_delay_length = br.read_bits(5)? as u8 + 1;
                }
            }
            hdr.display_model_info_present = br.read_bit()? != 0;
            hdr.num_operating_points = br.read_bits(5)? as u8 + 1;
            for index in 0..usize::from(hdr.num_operating_points) {
                let op = &mut hdr.operating_points[index];
                op.idc = br.read_bits(12)? as u16;
                if op.idc != 0 && ((op.idc & 0x00ff) == 0 || (op.idc & 0x0f00) == 0) {
                    return Err(invalid(
                        "operating_point_idc: alt/üst katman baytı 0 (dav1d obu.c 138-139)",
                        br.bit_position(),
                    ));
                }
                op.major_level = 2 + br.read_bits(3)? as u8;
                op.minor_level = br.read_bits(2)? as u8;
                if op.major_level > 3 {
                    op.tier = br.read_bit()? as u8;
                }
                if hdr.decoder_model_info_present {
                    op.decoder_model_param_present = br.read_bit()? != 0;
                    if op.decoder_model_param_present {
                        let delay_len = u32::from(hdr.encoder_decoder_buffer_delay_length);
                        let info = &mut hdr.operating_parameter_info[index];
                        info.decoder_buffer_delay = br.read_bits(delay_len)?;
                        info.encoder_buffer_delay = br.read_bits(delay_len)?;
                        info.low_delay_mode = br.read_bit()? != 0;
                    }
                }
                if hdr.display_model_info_present {
                    op.display_model_param_present = br.read_bit()? != 0;
                }
                op.initial_display_delay = if op.display_model_param_present {
                    br.read_bits(4)? as u8 + 1
                } else {
                    10
                };
            }
        }

        hdr.width_n_bits = br.read_bits(4)? as u8 + 1;
        hdr.height_n_bits = br.read_bits(4)? as u8 + 1;
        hdr.max_width = br.read_bits(u32::from(hdr.width_n_bits))? as i32 + 1;
        hdr.max_height = br.read_bits(u32::from(hdr.height_n_bits))? as i32 + 1;

        if !hdr.reduced_still_picture_header {
            hdr.frame_id_numbers_present = br.read_bit()? != 0;
            if hdr.frame_id_numbers_present {
                hdr.delta_frame_id_n_bits = br.read_bits(4)? as u8 + 2;
                hdr.frame_id_n_bits = br.read_bits(3)? as u8 + hdr.delta_frame_id_n_bits + 1;
            }
        }

        hdr.sb128 = br.read_bit()? != 0;
        hdr.filter_intra = br.read_bit()? != 0;
        hdr.intra_edge_filter = br.read_bit()? != 0;
        if hdr.reduced_still_picture_header {
            hdr.screen_content_tools = AdaptiveBoolean::Adaptive;
            hdr.force_integer_mv = AdaptiveBoolean::Adaptive;
        } else {
            hdr.inter_intra = br.read_bit()? != 0;
            hdr.masked_compound = br.read_bit()? != 0;
            hdr.warped_motion = br.read_bit()? != 0;
            hdr.dual_filter = br.read_bit()? != 0;
            hdr.order_hint = br.read_bit()? != 0;
            if hdr.order_hint {
                hdr.jnt_comp = br.read_bit()? != 0;
                hdr.ref_frame_mvs = br.read_bit()? != 0;
            }
            hdr.screen_content_tools = if br.read_bit()? != 0 {
                AdaptiveBoolean::Adaptive
            } else if br.read_bit()? != 0 {
                AdaptiveBoolean::On
            } else {
                AdaptiveBoolean::Off
            };
            // dav1d obu.c 209-210: screen_content_tools == OFF ise
            // force_integer_mv ADAPTIVE(2) yapılır ve kareden bit okunmaz.
            hdr.force_integer_mv = if hdr.screen_content_tools != AdaptiveBoolean::Off {
                if br.read_bit()? != 0 {
                    AdaptiveBoolean::Adaptive
                } else if br.read_bit()? != 0 {
                    AdaptiveBoolean::On
                } else {
                    AdaptiveBoolean::Off
                }
            } else {
                AdaptiveBoolean::Adaptive
            };
            if hdr.order_hint {
                hdr.order_hint_n_bits = br.read_bits(3)? as u8 + 1;
            }
        }
        hdr.super_res = br.read_bit()? != 0;
        hdr.cdef = br.read_bit()? != 0;
        hdr.restoration = br.read_bit()? != 0;

        hdr.hbd = br.read_bit()? as u8;
        if hdr.profile == 2 && hdr.hbd == 1 {
            hdr.hbd += br.read_bit()? as u8;
        }
        if hdr.profile != 1 {
            hdr.monochrome = br.read_bit()? != 0;
        }
        hdr.color_description_present = br.read_bit()? != 0;
        if hdr.color_description_present {
            hdr.pri = br.read_bits(8)? as u8;
            hdr.trc = br.read_bits(8)? as u8;
            hdr.mtrx = br.read_bits(8)? as u8;
        } else {
            hdr.pri = COLOR_PRI_UNKNOWN;
            hdr.trc = TRC_UNKNOWN;
            hdr.mtrx = MC_UNKNOWN;
        }
        if hdr.monochrome {
            hdr.color_range = br.read_bit()? as u8;
            hdr.layout = PixelLayout::I400;
            hdr.ss_hor = 1;
            hdr.ss_ver = 1;
            hdr.chr = ChromaSamplePosition::Unknown;
        } else if hdr.pri == COLOR_PRI_BT709 && hdr.trc == TRC_SRGB && hdr.mtrx == MC_IDENTITY {
            hdr.layout = PixelLayout::I444;
            hdr.color_range = 1;
            if hdr.profile != 1 && !(hdr.profile == 2 && hdr.hbd == 2) {
                return Err(invalid(
                    "BT.709/sRGB/identity renk: profile 1 veya 2+12-bit değil (dav1d obu.c 248-249)",
                    br.bit_position(),
                ));
            }
        } else {
            hdr.color_range = br.read_bit()? as u8;
            match hdr.profile {
                0 => {
                    hdr.layout = PixelLayout::I420;
                    hdr.ss_hor = 1;
                    hdr.ss_ver = 1;
                }
                1 => hdr.layout = PixelLayout::I444,
                _ => {
                    if hdr.hbd == 2 {
                        hdr.ss_hor = br.read_bit()? as u8;
                        if hdr.ss_hor == 1 {
                            hdr.ss_ver = br.read_bit()? as u8;
                        }
                    } else {
                        hdr.ss_hor = 1;
                    }
                    hdr.layout = match (hdr.ss_hor, hdr.ss_ver) {
                        (1, 1) => PixelLayout::I420,
                        (1, 0) => PixelLayout::I422,
                        _ => PixelLayout::I444,
                    };
                }
            }
            hdr.chr = if hdr.ss_hor == 1 && hdr.ss_ver == 1 {
                ChromaSamplePosition::from_raw(br.read_bits(2)? as u8)
            } else {
                ChromaSamplePosition::Unknown
            };
        }
        if strict && hdr.mtrx == MC_IDENTITY && hdr.layout != PixelLayout::I444 {
            return Err(invalid(
                "identity matrix ile 4:4:4 dışı layout (dav1d obu.c 274-279)",
                br.bit_position(),
            ));
        }
        if !hdr.monochrome {
            hdr.separate_uv_delta_q = br.read_bit()? != 0;
        }
        hdr.film_grain_present = br.read_bit()? != 0;
        br.read_trailing_bits(strict)?;
        Ok(hdr)
    }

    /// Seçilen operating point dizini: dav1d `c->operating_point` semantiği.
    ///
    /// İstenen dizin `num_operating_points` sınırındaysa **0** kullanılır
    /// (dav1d obu.c 1224–1226).
    #[must_use]
    pub fn operating_point_index(&self, requested: u8) -> usize {
        if usize::from(requested) < usize::from(self.num_operating_points) {
            usize::from(requested)
        } else {
            0
        }
    }

    /// "Aynı kodlanmış video dizisi" karşılaştırması (dav1d `obu.c` 1238–1241).
    ///
    /// dav1d yeni bir sequence header ile öncekini `memcmp(seq_hdr, c->seq_hdr,
    /// offsetof(Dav1dSequenceHeader, operating_parameter_info))` ile karşılaştırır:
    /// `operating_parameter_info` **dışındaki** bütün alanlar aynıysa dizi
    /// sürüyordur; değişen bir alan varsa dav1d bütün referans yuvalarını ve
    /// bekleyen frame header'ı **sıfırlar** (yeni dizi). Bu yöntem aynı kuralı
    /// alan alan uygular.
    #[must_use]
    pub fn same_coded_sequence(&self, other: &Self) -> bool {
        let mut normalized = self.clone();
        normalized.operating_parameter_info = other.operating_parameter_info;
        normalized == *other
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AdaptiveBoolean, ChromaSamplePosition, PixelLayout, SequenceHeader,
        SequenceHeaderParseOptions,
    };
    use crate::error::{BitstreamErrorKind, Error};

    /// Testler için MSB-first bit yazıcı (spec §5.3 söz dizimi kodlayıcısı).
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

        fn trailing(&mut self) {
            self.bit(1);
            while !self.bits.len().is_multiple_of(8) {
                self.bit(0);
            }
        }

        fn bytes(&self) -> Vec<u8> {
            let mut out = vec![0u8; self.bits.len().div_ceil(8)];
            for (i, bit) in self.bits.iter().enumerate() {
                if *bit != 0 {
                    out[i / 8] |= 1 << (7 - (i % 8));
                }
            }
            out
        }
    }

    fn kind(err: Error) -> BitstreamErrorKind {
        match err {
            Error::Bitstream(err) => err.kind,
            other => panic!("bit akışı hatası bekleniyordu: {other:?}"),
        }
    }

    /// Reduced still picture header'ın gövdesi (trailing hariç).
    fn reduced_header(w: &mut BitWriter) {
        w.bits(0, 3); // profile 0
        w.bit(1); // still_picture
        w.bit(1); // reduced_still_picture_header
        w.bits(2, 3); // major_level
        w.bits(1, 2); // minor_level
        w.bits(3, 4); // width_n_bits = 4
        w.bits(2, 4); // height_n_bits = 3
        w.bits(14, 4); // max_width = 15
        w.bits(6, 3); // max_height = 7
        w.bit(0); // sb128
        w.bit(1); // filter_intra
        w.bit(0); // intra_edge_filter
        w.bit(0); // super_res
        w.bit(1); // cdef
        w.bit(0); // restoration
        w.bit(0); // high_bitdepth
        w.bit(0); // mono_chrome
        w.bit(0); // color_description_present
        w.bit(1); // color_range
        w.bits(2, 2); // chroma_sample_position
        w.bit(1); // separate_uv_delta_q
        w.bit(1); // film_grain_params_present
    }

    #[test]
    fn reduced_still_picture_tum_alanlar() {
        let mut w = BitWriter::new();
        reduced_header(&mut w);
        w.trailing();
        let hdr = SequenceHeader::parse(&w.bytes()).expect("reduced header");
        assert_eq!(hdr.profile, 0);
        assert!(hdr.still_picture);
        assert!(hdr.reduced_still_picture_header);
        assert_eq!(hdr.num_operating_points, 1);
        assert_eq!(hdr.operating_points[0].major_level, 2);
        assert_eq!(hdr.operating_points[0].minor_level, 1);
        assert_eq!(hdr.operating_points[0].initial_display_delay, 10);
        assert_eq!(hdr.operating_points[0].idc, 0);
        assert_eq!(hdr.width_n_bits, 4);
        assert_eq!(hdr.height_n_bits, 3);
        assert_eq!(hdr.max_width, 15);
        assert_eq!(hdr.max_height, 7);
        assert!(hdr.filter_intra);
        assert!(!hdr.sb128);
        assert!(!hdr.intra_edge_filter);
        assert!(hdr.cdef);
        assert!(!hdr.super_res);
        assert!(!hdr.restoration);
        assert!(!hdr.frame_id_numbers_present);
        assert_eq!(hdr.screen_content_tools, AdaptiveBoolean::Adaptive);
        assert_eq!(hdr.force_integer_mv, AdaptiveBoolean::Adaptive);
        assert_eq!(hdr.layout, PixelLayout::I420);
        assert_eq!((hdr.ss_hor, hdr.ss_ver), (1, 1));
        assert!(!hdr.monochrome);
        assert_eq!(hdr.hbd, 0);
        assert_eq!(hdr.color_range, 1);
        assert_eq!(hdr.chr, ChromaSamplePosition::Colocated);
        assert_eq!((hdr.pri, hdr.trc, hdr.mtrx), (2, 2, 2));
        assert!(hdr.separate_uv_delta_q);
        assert!(hdr.film_grain_present);
        assert!(!hdr.timing_info_present);
        assert!(!hdr.display_model_info_present);
        assert!(!hdr.decoder_model_info_present);
        assert_eq!(hdr.order_hint_n_bits, 0);
    }

    /// Tam (reduced olmayan) header: timing + decoder/display model + 2 operating point.
    fn full_header(w: &mut BitWriter) {
        w.bits(2, 3); // profile 2
        w.bit(0); // still_picture
        w.bit(0); // reduced
        w.bit(1); // timing_info_present
        w.bits(1, 32); // num_units_in_display_tick
        w.bits(30, 32); // time_scale
        w.bit(1); // equal_picture_interval
        // uvlc(2): 3 = 0b11 -> bir sıfır, bir 1, ardından 1 bit (0b011).
        w.bit(0);
        w.bit(1);
        w.bit(1);
        w.bit(1); // decoder_model_info_present
        w.bits(4, 5); // encoder_decoder_buffer_delay_length = 5
        w.bits(1, 32); // num_units_in_decoding_tick
        w.bits(0, 5); // buffer_removal_delay_length = 1
        w.bits(1, 5); // frame_presentation_delay_length = 2
        w.bit(1); // display_model_info_present
        w.bits(1, 5); // num_operating_points = 2
        // op 0
        w.bits(0, 12); // idc
        w.bits(3, 3); // major_level = 5
        w.bits(1, 2); // minor_level
        w.bit(1); // tier
        w.bit(1); // decoder_model_param_present
        w.bits(7, 5); // decoder_buffer_delay
        w.bits(8, 5); // encoder_buffer_delay
        w.bit(1); // low_delay_mode
        w.bit(1); // display_model_param_present
        w.bits(5, 4); // initial_display_delay = 6
        // op 1
        w.bits(0x101, 12); // idc
        w.bits(2, 3); // major_level = 4
        w.bits(0, 2); // minor_level
        w.bit(0); // tier
        w.bit(0); // decoder_model_param_present
        w.bit(0); // display_model_param_present
        // boyutlar
        w.bits(15, 4); // width_n_bits = 16
        w.bits(8, 4); // height_n_bits = 9
        w.bits(0xffff, 16); // max_width = 65536
        w.bits(479, 9); // max_height = 480
        // frame id
        w.bit(1); // frame_id_numbers_present
        w.bits(0, 4); // delta = 2
        w.bits(2, 3); // frame_id_n_bits = 2 + 2 + 1 = 5
        // araç bayrakları
        w.bit(1); // sb128
        w.bit(1); // filter_intra
        w.bit(0); // intra_edge_filter
        w.bit(1); // inter_intra
        w.bit(1); // masked_compound
        w.bit(1); // warped_motion
        w.bit(1); // dual_filter
        w.bit(1); // order_hint
        w.bit(1); // jnt_comp
        w.bit(1); // ref_frame_mvs
        w.bit(1); // screen_content_tools -> adaptive
        w.bit(1); // force_integer_mv -> adaptive
        w.bits(4, 3); // order_hint_n_bits = 5
        w.bit(1); // super_res
        w.bit(1); // cdef
        w.bit(1); // restoration
        // renk
        w.bit(1); // high_bitdepth
        w.bit(1); // twelve_bit -> hbd = 2
        w.bit(0); // mono_chrome
        w.bit(1); // color_description_present
        w.bits(9, 8); // pri = BT2020
        w.bits(16, 8); // trc = PQ
        w.bits(9, 8); // mtrx = BT2020_NCL
        w.bit(1); // color_range
        w.bit(1); // subsampling_x
        w.bit(0); // subsampling_y
        w.bit(1); // separate_uv_delta_q
        w.bit(1); // film_grain_present
    }

    #[test]
    fn tam_header_butun_alanlar() {
        let mut w = BitWriter::new();
        full_header(&mut w);
        w.trailing();
        let hdr = SequenceHeader::parse(&w.bytes()).expect("tam header");
        assert_eq!(hdr.profile, 2);
        assert!(!hdr.still_picture);
        assert!(!hdr.reduced_still_picture_header);
        assert!(hdr.timing_info_present);
        assert_eq!(hdr.num_units_in_tick, 1);
        assert_eq!(hdr.time_scale, 30);
        assert!(hdr.equal_picture_interval);
        assert_eq!(hdr.num_ticks_per_picture, 3, "uvlc(2) + 1");
        assert!(hdr.decoder_model_info_present);
        assert_eq!(hdr.encoder_decoder_buffer_delay_length, 5);
        assert_eq!(hdr.num_units_in_decoding_tick, 1);
        assert_eq!(hdr.buffer_removal_delay_length, 1);
        assert_eq!(hdr.frame_presentation_delay_length, 2);
        assert!(hdr.display_model_info_present);
        assert_eq!(hdr.num_operating_points, 2);
        assert_eq!(hdr.operating_points[0].idc, 0);
        assert_eq!(hdr.operating_points[0].major_level, 5);
        assert_eq!(hdr.operating_points[0].minor_level, 1);
        assert_eq!(hdr.operating_points[0].tier, 1);
        assert!(hdr.operating_points[0].decoder_model_param_present);
        assert_eq!(hdr.operating_points[0].initial_display_delay, 6);
        assert!(hdr.operating_points[0].display_model_param_present);
        assert_eq!(hdr.operating_parameter_info[0].decoder_buffer_delay, 7);
        assert_eq!(hdr.operating_parameter_info[0].encoder_buffer_delay, 8);
        assert!(hdr.operating_parameter_info[0].low_delay_mode);
        assert_eq!(hdr.operating_points[1].idc, 0x101);
        assert_eq!(hdr.operating_points[1].major_level, 4);
        assert_eq!(hdr.operating_points[1].tier, 0);
        assert!(!hdr.operating_points[1].decoder_model_param_present);
        assert_eq!(hdr.operating_points[1].initial_display_delay, 10);
        assert_eq!(hdr.width_n_bits, 16);
        assert_eq!(hdr.height_n_bits, 9);
        assert_eq!(hdr.max_width, 65536);
        assert_eq!(hdr.max_height, 480);
        assert!(hdr.frame_id_numbers_present);
        assert_eq!(hdr.delta_frame_id_n_bits, 2);
        assert_eq!(hdr.frame_id_n_bits, 5);
        assert!(hdr.sb128);
        assert!(hdr.filter_intra);
        assert!(!hdr.intra_edge_filter);
        assert!(hdr.inter_intra);
        assert!(hdr.masked_compound);
        assert!(hdr.warped_motion);
        assert!(hdr.dual_filter);
        assert!(hdr.order_hint);
        assert!(hdr.jnt_comp);
        assert!(hdr.ref_frame_mvs);
        assert_eq!(hdr.screen_content_tools, AdaptiveBoolean::Adaptive);
        assert_eq!(hdr.force_integer_mv, AdaptiveBoolean::Adaptive);
        assert_eq!(hdr.order_hint_n_bits, 5);
        assert!(hdr.super_res);
        assert!(hdr.cdef);
        assert!(hdr.restoration);
        assert_eq!(hdr.hbd, 2);
        assert!(!hdr.monochrome);
        assert!(hdr.color_description_present);
        assert_eq!(hdr.pri, 9);
        assert_eq!(hdr.trc, 16);
        assert_eq!(hdr.mtrx, 9);
        assert_eq!(hdr.color_range, 1);
        assert_eq!((hdr.ss_hor, hdr.ss_ver), (1, 0));
        assert_eq!(hdr.layout, PixelLayout::I422);
        assert_eq!(hdr.chr, ChromaSamplePosition::Unknown);
        assert!(hdr.separate_uv_delta_q);
        assert!(hdr.film_grain_present);
        assert_eq!(hdr.operating_point_index(1), 1);
        assert_eq!(
            hdr.operating_point_index(31),
            0,
            "sınır dışı istek 0'a düşer (dav1d obu.c 1224-1226)"
        );
    }

    /// Profil 1 + BT.709/sRGB/identity renk yapılandırması.
    fn identity_header(w: &mut BitWriter) {
        w.bits(1, 3); // profile 1
        w.bit(0); // still
        w.bit(0); // reduced
        w.bit(0); // timing_info_present
        w.bit(0); // display_model_info_present
        w.bits(0, 5); // 1 operating point
        w.bits(0, 12); // idc
        w.bits(0, 3); // major_level = 2
        w.bits(0, 2); // minor_level
        w.bits(0, 4); // width_n_bits = 1
        w.bits(0, 4); // height_n_bits = 1
        w.bits(0, 1); // max_width = 1
        w.bits(0, 1); // max_height = 1
        w.bit(0); // frame_id_numbers_present
        w.bit(0); // sb128
        w.bit(0); // filter_intra
        w.bit(0); // intra_edge_filter
        w.bit(0); // inter_intra
        w.bit(0); // masked_compound
        w.bit(0); // warped_motion
        w.bit(0); // dual_filter
        w.bit(0); // order_hint
        w.bit(0); // screen_content_tools: ilk bit 0
        w.bit(1); // ikinci bit 1 -> ON
        w.bit(0); // force_integer_mv: ilk bit 0
        w.bit(1); // ikinci bit 1 -> ON
        w.bit(0); // super_res
        w.bit(0); // cdef
        w.bit(0); // restoration
        w.bit(0); // high_bitdepth
        // profil 1: mono_chrome okunmaz
        w.bit(1); // color_description_present
        w.bits(1, 8); // pri BT709
        w.bits(13, 8); // trc sRGB
        w.bits(0, 8); // mtrx identity
        // identity dalı: color_range/subsumpling/chroma bitleri YOK
        w.bit(1); // separate_uv_delta_q
        w.bit(0); // film_grain_present
    }

    #[test]
    fn identity_renk_dali_44x4_ve_profil1() {
        let mut w = BitWriter::new();
        identity_header(&mut w);
        w.trailing();
        let hdr = SequenceHeader::parse(&w.bytes()).expect("identity header");
        assert_eq!(hdr.profile, 1);
        assert_eq!(hdr.layout, PixelLayout::I444);
        assert_eq!(hdr.color_range, 1);
        assert_eq!((hdr.ss_hor, hdr.ss_ver), (0, 0));
        assert!(!hdr.monochrome);
        assert_eq!(hdr.max_width, 1);
        assert_eq!(hdr.max_height, 1);
        assert_eq!(hdr.screen_content_tools, AdaptiveBoolean::On);
        assert_eq!(hdr.force_integer_mv, AdaptiveBoolean::On);
        assert_eq!(hdr.num_operating_points, 1);
        assert_eq!(hdr.operating_points[0].major_level, 2);
        assert_eq!(hdr.operating_points[0].tier, 0);
        assert_eq!(hdr.operating_points[0].initial_display_delay, 10);
    }

    #[test]
    fn kesik_veri_panik_uretmez() {
        let mut w = BitWriter::new();
        reduced_header(&mut w);
        let bytes = w.bytes();
        for cut in 0..bytes.len() {
            // Trailing bit yok; her kesim noktası ya hata ya da (son bayt
            // tamamen doluysa) geçerli olabilir. Önemli olan panik olmaması.
            let _ = SequenceHeader::parse(&bytes[..cut]);
        }
        let err = SequenceHeader::parse(&[]).expect_err("boş");
        assert_eq!(kind(err), BitstreamErrorKind::UnexpectedEnd);
    }

    #[test]
    fn gecersiz_profil_ve_reduced_kombinasyonu() {
        let mut w = BitWriter::new();
        w.bits(3, 3); // profile 3
        w.trailing();
        let err = SequenceHeader::parse(&w.bytes()).expect_err("profil");
        assert!(matches!(
            kind(err),
            BitstreamErrorKind::InvalidSequenceHeader(_)
        ));

        let mut w = BitWriter::new();
        w.bits(0, 3); // profile 0
        w.bit(0); // still_picture
        w.bit(1); // reduced -> geçersiz
        w.trailing();
        let err = SequenceHeader::parse(&w.bytes()).expect_err("reduced");
        assert!(matches!(
            kind(err),
            BitstreamErrorKind::InvalidSequenceHeader(_)
        ));
    }

    #[test]
    fn gecersiz_operating_point_idc() {
        // idc = 0x100 -> alt bayt 0: reddedilir.
        let mut w = BitWriter::new();
        w.bits(0, 3);
        w.bit(0);
        w.bit(0);
        w.bit(0); // timing
        w.bit(0); // display model
        w.bits(0, 5); // 1 op
        w.bits(0x100, 12);
        w.trailing();
        let err = SequenceHeader::parse(&w.bytes()).expect_err("idc");
        assert!(matches!(
            kind(err),
            BitstreamErrorKind::InvalidSequenceHeader(_)
        ));
    }

    #[test]
    fn kati_kip_ek_kurallari() {
        let mut w = BitWriter::new();
        full_header(&mut w);
        w.trailing();
        let bytes = w.bytes();
        SequenceHeader::parse(&bytes).expect("gevşek kipte geçerli");
        let strict = SequenceHeaderParseOptions {
            strict_std_compliance: true,
        };
        SequenceHeader::parse_with(&bytes, strict).expect("bu header katı kipte de geçerli");

        // time_scale = 0: gevşek kabul, katı red.
        let mut w = BitWriter::new();
        w.bits(0, 3);
        w.bit(0);
        w.bit(0);
        w.bit(1); // timing_info_present
        w.bits(1, 32); // num_units
        w.bits(0, 32); // time_scale = 0
        w.bit(0); // equal_picture_interval
        w.bit(0); // decoder_model_info_present
        w.bit(0); // display_model_info_present
        w.bits(0, 5); // 1 op
        w.bits(0, 12);
        w.bits(0, 3); // major 2
        w.bits(0, 2);
        w.bits(0, 4);
        w.bits(0, 4);
        w.bits(0, 1);
        w.bits(0, 1);
        w.bit(0); // frame id
        w.bit(0);
        w.bit(0);
        w.bit(0);
        w.bit(0); // inter
        w.bit(0); // masked
        w.bit(0); // warped
        w.bit(0); // dual
        w.bit(0); // order hint
        w.bit(0); // sct ilk
        w.bit(0); // sct ikinci -> OFF
        // force_integer_mv okunmaz (screen_content_tools OFF)
        w.bit(0); // super_res
        w.bit(0); // cdef
        w.bit(0); // restoration
        w.bit(0); // high bitdepth
        w.bit(0); // mono
        w.bit(0); // color desc
        w.bit(0); // color_range
        w.bits(0, 2); // chr
        w.bit(0); // separate uv
        w.bit(0); // film grain
        w.trailing();
        let bytes = w.bytes();
        let hdr = SequenceHeader::parse(&bytes).expect("gevşek");
        assert_eq!(hdr.time_scale, 0);
        assert_eq!(hdr.screen_content_tools, AdaptiveBoolean::Off);
        assert_eq!(
            hdr.force_integer_mv,
            AdaptiveBoolean::Adaptive,
            "dav1d: screen_content_tools OFF iken force_integer_mv = 2"
        );
        let err = SequenceHeader::parse_with(&bytes, strict).expect_err("katı");
        assert!(matches!(
            kind(err),
            BitstreamErrorKind::InvalidSequenceHeader(_)
        ));
    }

    #[test]
    fn trailing_bits_gevsek_ve_kati_kontrol() {
        let strict = SequenceHeaderParseOptions {
            strict_std_compliance: true,
        };
        let mut w = BitWriter::new();
        reduced_header(&mut w);
        // Trailing bit eklenmedi: dolgu bitleri 0 olduğu için gevşek kip kabul
        // eder (dav1d de yalnız okumanın taşmadığını denetler); katı kip reddeder.
        assert!(SequenceHeader::parse(&w.bytes()).is_ok(), "gevşek kabul");
        assert!(SequenceHeader::parse_with(&w.bytes(), strict).is_err());

        w.trailing();
        SequenceHeader::parse(&w.bytes()).expect("trailing tamam");
        SequenceHeader::parse_with(&w.bytes(), strict).expect("katı kipte de tam");

        // Trailing sonrasına bir dolgu biti ekle; katı kipte bu bit kirletilir.
        w.bit(0);
        let mut bytes = w.bytes();
        if let Some(last) = bytes.last_mut() {
            *last |= 1;
        }
        assert!(SequenceHeader::parse(&bytes).is_ok(), "gevşek kabul");
        let err = SequenceHeader::parse_with(&bytes, strict).expect_err("kirli");
        assert_eq!(kind(err), BitstreamErrorKind::TrailingBits);
    }
}
