//! AVIF görüntü yapılandırması, metaveri ve düzlem/piksel taşıyıcı tipleri.

#![forbid(unsafe_code)]

use crate::avif::color;
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
}

impl Default for AvifDecodeConfig {
    fn default() -> Self {
        Self {
            n_threads: 0,
            max_pixels: 16_384 * 16_384,
            allocation_limit: None,
            apply_grain: true,
            strict: false,
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

/// Çözülmüş ve dönüşümleri (`clap`, `irot`, `imir`) uygulanmış düzlemsel AVIF görüntüsü.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AvifImage {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) chroma_width: u32,
    pub(crate) chroma_height: u32,
    pub(crate) bit_depth: BitDepth,
    pub(crate) chroma_format: ChromaFormat,
    pub(crate) y_plane: Vec<u16>,
    pub(crate) u_plane: Vec<u16>,
    pub(crate) v_plane: Vec<u16>,
    pub(crate) alpha_plane: Option<Vec<u16>>,
    pub(crate) premultiplied_alpha: bool,
    pub(crate) color: ColorInfo,
    pub(crate) icc_profile: Option<Vec<u8>>,
    pub(crate) exif: Option<Vec<u8>>,
    pub(crate) xmp: Option<Vec<u8>>,
    pub(crate) clli: Option<ContentLightLevel>,
    pub(crate) mdcv: Option<MasteringDisplay>,
    pub(crate) pasp: Option<PixelAspectRatio>,
}

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

    /// Sıkı paketlenmiş (`width * height`) Y (veya Identity GBR kipinde G) düzlemi örnekleri.
    #[must_use]
    pub fn y_plane(&self) -> &[u16] {
        &self.y_plane
    }

    /// Sıkı paketlenmiş (`chroma_width * chroma_height`) U (veya Identity GBR kipinde B) düzlemi örnekleri.
    #[must_use]
    pub fn u_plane(&self) -> &[u16] {
        &self.u_plane
    }

    /// Sıkı paketlenmiş (`chroma_width * chroma_height`) V (veya Identity GBR kipinde R) düzlemi örnekleri.
    #[must_use]
    pub fn v_plane(&self) -> &[u16] {
        &self.v_plane
    }

    /// Varsa sıkı paketlenmiş (`width * height`) alfa düzlemi örnekleri.
    #[must_use]
    pub fn alpha_plane(&self) -> Option<&[u16]> {
        self.alpha_plane.as_deref()
    }

    /// Y düzleminin `y` indeksli satır dilimini döndürür.
    #[must_use]
    pub fn y_row(&self, y: usize) -> Option<&[u16]> {
        let w = self.width as usize;
        let start = y.checked_mul(w)?;
        self.y_plane.get(start..start.checked_add(w)?)
    }

    /// U düzleminin `y` indeksli satır dilimini döndürür.
    #[must_use]
    pub fn u_row(&self, y: usize) -> Option<&[u16]> {
        let cw = self.chroma_width as usize;
        if cw == 0 {
            return None;
        }
        let start = y.checked_mul(cw)?;
        self.u_plane.get(start..start.checked_add(cw)?)
    }

    /// V düzleminin `y` indeksli satır dilimini döndürür.
    #[must_use]
    pub fn v_row(&self, y: usize) -> Option<&[u16]> {
        let cw = self.chroma_width as usize;
        if cw == 0 {
            return None;
        }
        let start = y.checked_mul(cw)?;
        self.v_plane.get(start..start.checked_add(cw)?)
    }

    /// Alfa düzleminin `y` indeksli satır dilimini döndürür.
    #[must_use]
    pub fn alpha_row(&self, y: usize) -> Option<&[u16]> {
        let alpha = self.alpha_plane.as_ref()?;
        let w = self.width as usize;
        let start = y.checked_mul(w)?;
        alpha.get(start..start.checked_add(w)?)
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
