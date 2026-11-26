//! AVIF görüntü yapılandırması, metaveri ve düzlem/piksel taşıyıcı tipleri.

#![forbid(unsafe_code)]

use crate::avif::color;
use crate::avif::gainmap::{GainMap, GainMapMetadata};
use crate::error::Result;
use crate::frame::{BitDepth, ChromaFormat};

/// AVIF çözme yapılandırması.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AvifDecodeConfig {
    /// İş parçacığı sayısı (`0` = sistemdeki mantıksal çekirdek sayısına göre otomatik).
    pub n_threads: usize,
    /// Tahsis öncesi izin verilen en büyük piksel alanı (`genişlik × yükseklik`).
    pub max_pixels: u64,
    /// İsteğe bağlı toplam bellek bütçesi üst sınırı (bayt).
    pub allocation_limit: Option<usize>,
    /// AV1 akışında bulunan Film Grain parametrelerinin uygulanıp uygulanmayacağı.
    pub apply_grain: bool,
    /// Katı standart uyumluluğu denetimi.
    pub strict: bool,
    /// Varsa ISO 21496-1 `tmap` kazanç haritası görüntüsünün de çözülüp çözülmeyeceği.
    ///
    /// Varsayılan `false` değerinde `tmap` üst verisi ve kutu doğrulaması yürütülür ancak
    /// ikincil kazanç haritası AV1 akışı çözülmez.
    pub decode_gain_map: bool,
}

impl Default for AvifDecodeConfig {
    fn default() -> Self {
        Self {
            n_threads: 0,
            max_pixels: 16_384 * 16_384,
            allocation_limit: None,
            apply_grain: true,
            strict: false,
            decode_gain_map: false,
        }
    }
}

/// CICP / ITU-T H.273 renk tanımlayıcıları (`colr` `nclx` veya AV1 Sequence Header).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorInfo {
    /// Renk ana bileşenleri (`color_primaries`, ör. `1` = BT.709, `9` = BT.2020).
    pub color_primaries: u16,
    /// Aktarım karakteristiği (`transfer_characteristics`, ör. `13` = sRGB, `16` = PQ).
    pub transfer_characteristics: u16,
    /// Matris katsayıları (`matrix_coefficients`, ör. `0` = Identity/GBR, `1` = BT.709, `6` = BT.601).
    pub matrix_coefficients: u16,
    /// Örnek değerlerinin tam aralık (`true`: `[0, 2^N - 1]`) veya sınırlı aralık (`false`) olduğu.
    pub full_range: bool,
}

impl Default for ColorInfo {
    fn default() -> Self {
        // MIAF / AVIF varsayılanı: BT.709 (1), sRGB (13), BT.601 (6), tam aralık (true).
        Self {
            color_primaries: 1,
            transfer_characteristics: 13,
            matrix_coefficients: 6,
            full_range: true,
        }
    }
}

/// `clap` (Clean Aperture) kesirli kırpma parametreleri.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CleanAperture {
    /// Kırpılmış genişlik payı.
    pub width_n: u32,
    /// Kırpılmış genişlik paydası.
    pub width_d: u32,
    /// Kırpılmış yükseklik payı.
    pub height_n: u32,
    /// Kırpılmış yükseklik paydası.
    pub height_d: u32,
    /// Yatay merkez ofseti payı.
    pub horiz_off_n: i32,
    /// Yatay merkez ofseti paydası.
    pub horiz_off_d: u32,
    /// Dikey merkez ofseti payı.
    pub vert_off_n: i32,
    /// Dikey merkez ofseti paydası.
    pub vert_off_d: u32,
}

/// `pasp` (Pixel Aspect Ratio) piksel en-boy oranı.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelAspectRatio {
    /// Piksel yatay aralığı.
    pub h_spacing: u32,
    /// Piksel dikey aralığı.
    pub v_spacing: u32,
}

/// `clli` (Content Light Level Information) HDR içerik ışık düzeyi.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContentLightLevel {
    /// En yüksek içerik ışık düzeyi (`MaxCLL`, cd/m²).
    pub max_content_light_level: u16,
    /// En yüksek kare ortalama ışık düzeyi (`MaxFALL`, cd/m²).
    pub max_pic_average_light_level: u16,
}

/// `mdcv` (Mastering Display Colour Volume) HDR referans ekran renk hacmi.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MasteringDisplay {
    /// Ekran ana renk koordinatları (`0.00002` biriminde `[x, y] × 3`).
    pub primaries: [[u16; 2]; 3],
    /// Beyaz nokta koordinatları (`0.00002` biriminde `[x, y]`).
    pub white_point: [u16; 2],
    /// En yüksek parlaklık (`0.0001` cd/m² biriminde).
    pub max_luminance: u32,
    /// En düşük parlaklık (`0.0001` cd/m² biriminde).
    pub min_luminance: u32,
}

/// Piksel çözme işlemi yapılmadan `meta` kutusundan çıkarılan AVIF başlık bilgisi.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AvifInfo {
    /// Dönüşümler (`clap`, `irot`, `imir`) sonrası nihai görüntü genişliği.
    pub width: u32,
    /// Dönüşümler (`clap`, `irot`, `imir`) sonrası nihai görüntü yüksekliği.
    pub height: u32,
    /// Kanal başına bit derinliği (`8`, `10` veya `12` bit).
    pub bit_depth: BitDepth,
    /// Kroma alt örnekleme biçimi (`4:2:0`, `4:2:2`, `4:4:4` veya `Monochrome`).
    pub chroma_format: ChromaFormat,
    /// Yardımcı alfa kanalının bulunup bulunmadığı.
    pub has_alpha: bool,
    /// Renk kanallarının alfa ile ön-çarpımlı (`prem`) olup olmadığı.
    pub premultiplied_alpha: bool,
    /// CICP / H.273 renk tanımlayıcıları.
    pub color: ColorInfo,
    /// Varsa gömülü ICC renk profili baytları.
    pub icc_profile: Option<Vec<u8>>,
    /// Varsa TIFF başlığından başlayan ham Exif metaverisi.
    pub exif: Option<Vec<u8>>,
    /// Varsa ham XMP (`application/rdf+xml`) metaverisi.
    pub xmp: Option<Vec<u8>>,
    /// Varsa içerik ışık düzeyi (`clli`) bilgisi.
    pub clli: Option<ContentLightLevel>,
    /// Varsa referans ekran renk hacmi (`mdcv`) bilgisi.
    pub mdcv: Option<MasteringDisplay>,
    /// Varsa piksel en-boy oranı (`pasp`) bilgisi.
    pub pasp: Option<PixelAspectRatio>,
    /// Görüntüde geçerli ve desteklenen bir ISO 21496-1 `tmap` kazanç haritası bulunup bulunmadığı.
    pub has_gain_map: bool,
    /// Varsa ISO 21496-1 `tmap` kazanç haritası üst verisi ve alternatif görüntü özellikleri.
    pub gain_map_metadata: Option<GainMapMetadata>,
}

/// 8-bit sıkı paketlenmiş (`R, G, B`) RGB görüntü tamponu.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbImage {
    /// Görüntü genişliği (piksel).
    pub width: u32,
    /// Görüntü yüksekliği (piksel).
    pub height: u32,
    /// Satır-büyük (`row-major`) sırada `width * height * 3` baytlık `[R, G, B, ...]` verisi.
    pub pixels: Vec<u8>,
}

/// 8-bit sıkı paketlenmiş (`R, G, B, A`) RGBA görüntü tamponu.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbaImage {
    /// Görüntü genişliği (piksel).
    pub width: u32,
    /// Görüntü yüksekliği (piksel).
    pub height: u32,
    /// Satır-büyük (`row-major`) sırada `width * height * 4` baytlık `[R, G, B, A, ...]` verisi.
    pub pixels: Vec<u8>,
}

/// 16-bit (`8/10/12-bit` örnek derinliğinde) sıkı paketlenmiş (`R, G, B`) RGB görüntü tamponu.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb16Image {
    /// Görüntü genişliği (piksel).
    pub width: u32,
    /// Görüntü yüksekliği (piksel).
    pub height: u32,
    /// Örneklerin etkin bit derinliği (`8`, `10` veya `12` bit).
    pub bit_depth: BitDepth,
    /// Satır-büyük sırada `width * height * 3` elemanlı `[R, G, B, ...]` örnekleri.
    pub pixels: Vec<u16>,
}

/// 16-bit (`8/10/12-bit` örnek derinliğinde) sıkı paketlenmiş (`R, G, B, A`) RGBA görüntü tamponu.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgba16Image {
    /// Görüntü genişliği (piksel).
    pub width: u32,
    /// Görüntü yüksekliği (piksel).
    pub height: u32,
    /// Örneklerin etkin bit derinliği (`8`, `10` veya `12` bit).
    pub bit_depth: BitDepth,
    /// Satır-büyük sırada `width * height * 4` elemanlı `[R, G, B, A, ...]` örnekleri.
    pub pixels: Vec<u16>,
}

/// Bir AVIF düzleminin veya düzlem satırının bit derinliğine özgü doğal dilim görünümü.
///
/// `BitDepth::Eight` görüntülerde [`AvifPlane::U8`], `BitDepth::Ten` ve `BitDepth::Twelve`
/// görüntülerde ise [`AvifPlane::U16`] varyantı döndürülür.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AvifPlane<'a> {
    /// 8-bit doğal örnek dilimi (`1` bayt/örnek).
    U8(&'a [u8]),
    /// 10-bit veya 12-bit doğal örnek dilimi (`2` bayt/örnek).
    U16(&'a [u16]),
}

impl<'a> AvifPlane<'a> {
    /// Dilimdeki toplam örnek sayısı.
    #[must_use]
    pub const fn len(&self) -> usize {
        match self {
            Self::U8(s) => s.len(),
            Self::U16(s) => s.len(),
        }
    }

    /// Dilimin boş olup olmadığı.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Düzlem 8-bit ise ham `&[u8]` dilimini döndürür.
    #[must_use]
    pub const fn as_u8(self) -> Option<&'a [u8]> {
        match self {
            Self::U8(s) => Some(s),
            Self::U16(_) => None,
        }
    }

    /// Düzlem 10/12-bit ise ham `&[u16]` dilimini döndürür.
    #[must_use]
    pub const fn as_u16(self) -> Option<&'a [u16]> {
        match self {
            Self::U8(_) => None,
            Self::U16(s) => Some(s),
        }
    }

    /// Belirtilen indeksteki örneği `u16` olarak döndürür (sınır dışıysa `None`).
    #[must_use]
    pub fn get(&self, index: usize) -> Option<u16> {
        match self {
            Self::U8(s) => s.get(index).copied().map(u16::from),
            Self::U16(s) => s.get(index).copied(),
        }
    }
}

/// Düzlem örneklerinin iç depolama biçimi (`8-bit` için `Vec<u8>`, `10/12-bit` için `Vec<u16>`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PlaneStorage {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

impl PlaneStorage {
    #[inline]
    pub(crate) const fn bytes_per_sample(bit_depth: BitDepth) -> usize {
        match bit_depth {
            BitDepth::Eight => 1,
            BitDepth::Ten | BitDepth::Twelve => 2,
        }
    }

    #[inline]
    pub(crate) const fn empty_for(bit_depth: BitDepth) -> Self {
        match bit_depth {
            BitDepth::Eight => Self::U8(Vec::new()),
            BitDepth::Ten | BitDepth::Twelve => Self::U16(Vec::new()),
        }
    }

    #[inline]
    pub(crate) fn as_plane(&self) -> AvifPlane<'_> {
        match self {
            Self::U8(v) => AvifPlane::U8(v),
            Self::U16(v) => AvifPlane::U16(v),
        }
    }

    #[inline]
    pub(crate) fn as_u8(&self) -> Option<&[u8]> {
        match self {
            Self::U8(v) => Some(v),
            Self::U16(_) => None,
        }
    }

    #[inline]
    pub(crate) fn as_u16(&self) -> Option<&[u16]> {
        match self {
            Self::U8(_) => None,
            Self::U16(v) => Some(v),
        }
    }

    #[inline]
    pub(crate) fn row(&self, y: usize, width: usize) -> Option<AvifPlane<'_>> {
        let start = y.checked_mul(width)?;
        let end = start.checked_add(width)?;
        match self {
            Self::U8(v) => v.get(start..end).map(AvifPlane::U8),
            Self::U16(v) => v.get(start..end).map(AvifPlane::U16),
        }
    }
}

/// Çözülmüş ve dönüşümleri (`clap`, `irot`, `imir`) uygulanmış düzlemsel AVIF görüntüsü.
#[derive(Clone, Debug)]
pub struct AvifImage {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) chroma_width: u32,
    pub(crate) chroma_height: u32,
    pub(crate) bit_depth: BitDepth,
    pub(crate) chroma_format: ChromaFormat,
    pub(crate) y_plane: PlaneStorage,
    pub(crate) u_plane: PlaneStorage,
    pub(crate) v_plane: PlaneStorage,
    pub(crate) alpha_plane: Option<PlaneStorage>,
    pub(crate) premultiplied_alpha: bool,
    pub(crate) color: ColorInfo,
    pub(crate) icc_profile: Option<Vec<u8>>,
    pub(crate) exif: Option<Vec<u8>>,
    pub(crate) xmp: Option<Vec<u8>>,
    pub(crate) clli: Option<ContentLightLevel>,
    pub(crate) mdcv: Option<MasteringDisplay>,
    pub(crate) pasp: Option<PixelAspectRatio>,
    pub(crate) gain_map_metadata: Option<GainMapMetadata>,
    pub(crate) gain_map: Option<GainMap>,
    pub(crate) effective_threads: usize,
}

impl PartialEq for AvifImage {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && self.chroma_width == other.chroma_width
            && self.chroma_height == other.chroma_height
            && self.bit_depth == other.bit_depth
            && self.chroma_format == other.chroma_format
            && self.y_plane == other.y_plane
            && self.u_plane == other.u_plane
            && self.v_plane == other.v_plane
            && self.alpha_plane == other.alpha_plane
            && self.premultiplied_alpha == other.premultiplied_alpha
            && self.color == other.color
            && self.icc_profile == other.icc_profile
            && self.exif == other.exif
            && self.xmp == other.xmp
            && self.clli == other.clli
            && self.mdcv == other.mdcv
            && self.pasp == other.pasp
            && self.gain_map_metadata == other.gain_map_metadata
            && self.gain_map == other.gain_map
    }
}

impl Eq for AvifImage {}

impl AvifImage {
    /// Luma (ve alfa) düzleminin piksel cinsinden genişliği.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Luma (ve alfa) düzleminin piksel cinsinden yüksekliği.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Kroma (`U` ve `V`) düzlemlerinin `(genişlik, yükseklik)` boyutları.
    ///
    /// `ChromaFormat::Monochrome` durumunda `(0, 0)` döner.
    #[must_use]
    pub const fn chroma_dimensions(&self) -> (u32, u32) {
        (self.chroma_width, self.chroma_height)
    }

    /// Etkin bit derinliği (`8`, `10` veya `12` bit).
    #[must_use]
    pub const fn bit_depth(&self) -> BitDepth {
        self.bit_depth
    }

    /// Kroma alt örnekleme biçimi.
    #[must_use]
    pub const fn chroma_format(&self) -> ChromaFormat {
        self.chroma_format
    }

    /// Görüntüde çözülmüş bir alfa düzlemi bulunup bulunmadığı.
    #[must_use]
    pub const fn has_alpha(&self) -> bool {
        self.alpha_plane.is_some()
    }

    /// Renk kanallarının alfa kanalı ile ön-çarpımlı (`prem`) olup olmadığı.
    #[must_use]
    pub const fn premultiplied_alpha(&self) -> bool {
        self.premultiplied_alpha
    }

    /// Sıkı paketlenmiş (`width * height`) Y (veya Identity GBR kipinde G) düzlemi görünümü.
    #[must_use]
    pub fn y_plane(&self) -> AvifPlane<'_> {
        self.y_plane.as_plane()
    }

    /// Sıkı paketlenmiş (`chroma_width * chroma_height`) U (veya Identity GBR kipinde B) düzlemi görünümü.
    #[must_use]
    pub fn u_plane(&self) -> AvifPlane<'_> {
        self.u_plane.as_plane()
    }

    /// Sıkı paketlenmiş (`chroma_width * chroma_height`) V (veya Identity GBR kipinde R) düzlemi görünümü.
    #[must_use]
    pub fn v_plane(&self) -> AvifPlane<'_> {
        self.v_plane.as_plane()
    }

    /// Varsa sıkı paketlenmiş (`width * height`) alfa düzlemi görünümü.
    #[must_use]
    pub fn alpha_plane(&self) -> Option<AvifPlane<'_>> {
        self.alpha_plane.as_ref().map(PlaneStorage::as_plane)
    }

    /// Görüntü 8-bit ise sıkı paketlenmiş Y düzlemi `&[u8]` dilimini döndürür.
    #[must_use]
    pub fn y_plane_u8(&self) -> Option<&[u8]> {
        self.y_plane.as_u8()
    }

    /// Görüntü 10/12-bit ise sıkı paketlenmiş Y düzlemi `&[u16]` dilimini döndürür.
    #[must_use]
    pub fn y_plane_u16(&self) -> Option<&[u16]> {
        self.y_plane.as_u16()
    }

    /// Görüntü 8-bit ise sıkı paketlenmiş U düzlemi `&[u8]` dilimini döndürür.
    #[must_use]
    pub fn u_plane_u8(&self) -> Option<&[u8]> {
        self.u_plane.as_u8()
    }

    /// Görüntü 10/12-bit ise sıkı paketlenmiş U düzlemi `&[u16]` dilimini döndürür.
    #[must_use]
    pub fn u_plane_u16(&self) -> Option<&[u16]> {
        self.u_plane.as_u16()
    }

    /// Görüntü 8-bit ise sıkı paketlenmiş V düzlemi `&[u8]` dilimini döndürür.
    #[must_use]
    pub fn v_plane_u8(&self) -> Option<&[u8]> {
        self.v_plane.as_u8()
    }

    /// Görüntü 10/12-bit ise sıkı paketlenmiş V düzlemi `&[u16]` dilimini döndürür.
    #[must_use]
    pub fn v_plane_u16(&self) -> Option<&[u16]> {
        self.v_plane.as_u16()
    }

    /// Görüntü 8-bit ve alfa düzlemli ise sıkı paketlenmiş alfa `&[u8]` dilimini döndürür.
    #[must_use]
    pub fn alpha_plane_u8(&self) -> Option<&[u8]> {
        self.alpha_plane.as_ref().and_then(PlaneStorage::as_u8)
    }

    /// Görüntü 10/12-bit ve alfa düzlemli ise sıkı paketlenmiş alfa `&[u16]` dilimini döndürür.
    #[must_use]
    pub fn alpha_plane_u16(&self) -> Option<&[u16]> {
        self.alpha_plane.as_ref().and_then(PlaneStorage::as_u16)
    }

    /// Y düzleminin `y` indeksli satır görünümünü döndürür.
    #[must_use]
    pub fn y_row(&self, y: usize) -> Option<AvifPlane<'_>> {
        if y >= self.height as usize {
            return None;
        }
        self.y_plane.row(y, self.width as usize)
    }

    /// U düzleminin `y` indeksli satır görünümünü döndürür.
    #[must_use]
    pub fn u_row(&self, y: usize) -> Option<AvifPlane<'_>> {
        let cw = self.chroma_width as usize;
        if cw == 0 || y >= self.chroma_height as usize {
            return None;
        }
        self.u_plane.row(y, cw)
    }

    /// V düzleminin `y` indeksli satır görünümünü döndürür.
    #[must_use]
    pub fn v_row(&self, y: usize) -> Option<AvifPlane<'_>> {
        let cw = self.chroma_width as usize;
        if cw == 0 || y >= self.chroma_height as usize {
            return None;
        }
        self.v_plane.row(y, cw)
    }

    /// Alfa düzleminin `y` indeksli satır görünümünü döndürür.
    #[must_use]
    pub fn alpha_row(&self, y: usize) -> Option<AvifPlane<'_>> {
        if y >= self.height as usize {
            return None;
        }
        let alpha = self.alpha_plane.as_ref()?;
        alpha.row(y, self.width as usize)
    }

    /// Görüntü 8-bit ise Y düzleminin `y` indeksli `&[u8]` satır dilimini döndürür.
    #[must_use]
    pub fn y_row_u8(&self, y: usize) -> Option<&[u8]> {
        self.y_row(y).and_then(AvifPlane::as_u8)
    }

    /// Görüntü 10/12-bit ise Y düzleminin `y` indeksli `&[u16]` satır dilimini döndürür.
    #[must_use]
    pub fn y_row_u16(&self, y: usize) -> Option<&[u16]> {
        self.y_row(y).and_then(AvifPlane::as_u16)
    }

    /// Görüntü 8-bit ise U düzleminin `y` indeksli `&[u8]` satır dilimini döndürür.
    #[must_use]
    pub fn u_row_u8(&self, y: usize) -> Option<&[u8]> {
        self.u_row(y).and_then(AvifPlane::as_u8)
    }

    /// Görüntü 10/12-bit ise U düzleminin `y` indeksli `&[u16]` satır dilimini döndürür.
    #[must_use]
    pub fn u_row_u16(&self, y: usize) -> Option<&[u16]> {
        self.u_row(y).and_then(AvifPlane::as_u16)
    }

    /// Görüntü 8-bit ise V düzleminin `y` indeksli `&[u8]` satır dilimini döndürür.
    #[must_use]
    pub fn v_row_u8(&self, y: usize) -> Option<&[u8]> {
        self.v_row(y).and_then(AvifPlane::as_u8)
    }

    /// Görüntü 10/12-bit ise V düzleminin `y` indeksli `&[u16]` satır dilimini döndürür.
    #[must_use]
    pub fn v_row_u16(&self, y: usize) -> Option<&[u16]> {
        self.v_row(y).and_then(AvifPlane::as_u16)
    }

    /// Görüntü 8-bit ve alfa düzlemli ise alfa düzleminin `y` indeksli `&[u8]` satır dilimini döndürür.
    #[must_use]
    pub fn alpha_row_u8(&self, y: usize) -> Option<&[u8]> {
        self.alpha_row(y).and_then(AvifPlane::as_u8)
    }

    /// Görüntü 10/12-bit ve alfa düzlemli ise alfa düzleminin `y` indeksli `&[u16]` satır dilimini döndürür.
    #[must_use]
    pub fn alpha_row_u16(&self, y: usize) -> Option<&[u16]> {
        self.alpha_row(y).and_then(AvifPlane::as_u16)
    }

    /// CICP / H.273 renk tanımlayıcıları.
    #[must_use]
    pub const fn color(&self) -> ColorInfo {
        self.color
    }

    /// Varsa gömülü ICC renk profili baytları.
    #[must_use]
    pub fn icc_profile(&self) -> Option<&[u8]> {
        self.icc_profile.as_deref()
    }

    /// Varsa TIFF başlığından başlayan ham Exif metaverisi.
    #[must_use]
    pub fn exif(&self) -> Option<&[u8]> {
        self.exif.as_deref()
    }

    /// Varsa ham XMP (`application/rdf+xml`) metaverisi.
    #[must_use]
    pub fn xmp(&self) -> Option<&[u8]> {
        self.xmp.as_deref()
    }

    /// Varsa içerik ışık düzeyi (`clli`) bilgisi.
    #[must_use]
    pub const fn clli(&self) -> Option<ContentLightLevel> {
        self.clli
    }

    /// Varsa referans ekran renk hacmi (`mdcv`) bilgisi.
    #[must_use]
    pub const fn mdcv(&self) -> Option<MasteringDisplay> {
        self.mdcv
    }

    /// Varsa piksel en-boy oranı (`pasp`) bilgisi.
    #[must_use]
    pub const fn pasp(&self) -> Option<PixelAspectRatio> {
        self.pasp
    }

    /// Görüntüde geçerli ve desteklenen bir ISO 21496-1 `tmap` kazanç haritası bulunup bulunmadığı.
    #[must_use]
    pub const fn has_gain_map(&self) -> bool {
        self.gain_map_metadata.is_some()
    }

    /// Varsa ISO 21496-1 `tmap` kazanç haritası üst verisi ve alternatif görüntü özellikleri.
    #[must_use]
    pub fn gain_map_metadata(&self) -> Option<&GainMapMetadata> {
        self.gain_map_metadata.as_ref()
    }

    /// `AvifDecodeConfig::decode_gain_map == true` ile çözülmüşse kazanç haritası görüntüsü ve üst verisi.
    #[must_use]
    pub fn gain_map(&self) -> Option<&GainMap> {
        self.gain_map.as_ref()
    }

    /// Düzlemleri H.273 renk matrisine göre 8-bit RGB (`[R, G, B]`) tamponuna dönüştürür.
    pub fn to_rgb8(&self) -> Result<RgbImage> {
        color::image_to_rgb8(self)
    }

    /// Düzlemleri H.273 renk matrisine göre 8-bit RGBA (`[R, G, B, A]`) tamponuna dönüştürür.
    ///
    /// Görüntüde alfa düzlemi yoksa `A` kanalı `255` ile doldurulur.
    pub fn to_rgba8(&self) -> Result<RgbaImage> {
        color::image_to_rgba8(self)
    }

    /// Düzlemleri H.273 renk matrisine göre etkin bit derinliğinde (`8/10/12-bit`) 16-bit RGB tamponuna dönüştürür.
    pub fn to_rgb16(&self) -> Result<Rgb16Image> {
        color::image_to_rgb16(self)
    }

    /// Düzlemleri H.273 renk matrisine göre etkin bit derinliğinde (`8/10/12-bit`) 16-bit RGBA tamponuna dönüştürür.
    ///
    /// Görüntüde alfa düzlemi yoksa `A` kanalı `(1 << bit_depth) - 1` ile doldurulur.
    pub fn to_rgba16(&self) -> Result<Rgba16Image> {
        color::image_to_rgba16(self)
    }
}
