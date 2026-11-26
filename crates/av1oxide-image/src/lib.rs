//! `av1oxide::avif` için [`image::ImageDecoder`] (`image 0.25`) köprü paketi.
//!
//! Bu paket `%100 #![forbid(unsafe_code)]` altında derlenir, ana `av1oxide`
//! paketine `image` bağımlılığı eklemeden AVIF görüntü çözme desteğini `image`
//! ekosistemine bağlar ve `image` paketini `default-features = false` ile kullanır.

#![forbid(unsafe_code)]
#![deny(unused_crate_dependencies)]
#![warn(clippy::all)]

use core::marker::PhantomData;
use std::io::{Cursor, Read};

use av1oxide::avif::{AvifDecodeConfig, AvifInfo, AvifPrepared};
use av1oxide::{BitDepth, Error};
use image::error::{DecodingError, ImageFormatHint, LimitError, LimitErrorKind};
use image::{
    ColorType, DynamicImage, ExtendedColorType, ImageDecoder, ImageError, ImageFormat, ImageResult,
    Limits,
};

/// [`av1oxide::avif`] arka ucunu kullanan [`image::ImageDecoder`] gerçekleştirimi.
///
/// `8-bit` AVIF görüntüleri `ColorType::Rgb8` / `ColorType::Rgba8` olarak, `10-bit`
/// ve `12-bit` AVIF görüntüleri ise `image` sözleşmesi uyarınca tam `[0..=65535]`
/// dinamik aralığına ölçeklenerek `ColorType::Rgb16` / `ColorType::Rgba16` (yerel
/// bayt sırası) biçiminde döndürülür.
///
/// ## Tek Ayrıştırma ve Doğrudan Çıktı Yazımı
///
/// - [`AvifDecoder::with_config`] akışı yalnızca bir kez okuyup [`AvifPrepared`] ile
///   kapsayıcı/üst veri yapısını (`cmin` açma dâhil) tek seferde ayrıştırır.
/// - [`ImageDecoder::read_image`] ikinci bir kutu ayrıştırması yapmadan düzlemleri çözer,
///   ardından ham girdi baytlarını serbest bırakıp `AvifImage::write_*_into` metodlarıyla
///   doğrudan çağıranın `buf: &mut [u8]` dilimine (`0` ara RGB/RGBA vektörüyle) yazar.
///
/// ## Bellek Bütçesi Kapsamı (`allocation_limit` ve `Limits::max_alloc`)
///
/// - [`AvifDecoder::with_config`] akıştan okuduğu iç `raw_bytes` tamponunu okuma başlamadan
///   önce `Read::take(limit + 1)` ile sınırlar; ardından kalan bütçeyi (`limit - raw_bytes.len()`)
///   [`AvifPrepared::from_vec`] çağrısına aktararak `cmin`/`infe` açma ve `ICC`/`Exif`/`XMP`
///   başlık tahsislerini aynı tavan içinde tutar.
/// - [`ImageDecoder::set_limits`] çağrısı hem çıktı boyutunu (`total_bytes()`) hem de adaptörün
///   halihazırda tuttuğu girdi + kapsayıcı + üst veri belleğini `limits.max_alloc` tavanına
///   karşı denetler.
/// - [`ImageDecoder::read_image`] aşamasında iç çözücü tamponları toplam tavan içinden
///   `raw_bytes.len()` düşülerek sınırlandırılır; çağıranın dışarıda ayırıp `read_image`
///   çağrısına geçirdiği `&mut [u8]` çıktı dilimi kütüphanenin iç tahsis sayacına dâhil değildir.
#[derive(Clone, Debug)]
pub struct AvifDecoder<R: Read> {
    prepared: AvifPrepared,
    config: AvifDecodeConfig,
    limits: Limits,
    _marker: PhantomData<R>,
}

impl<R: Read> AvifDecoder<R> {
    /// Varsayılan [`AvifDecodeConfig`] ile yeni bir [`AvifDecoder`] oluşturur.
    ///
    /// Akış baytları okunur ve piksel çözümüne geçilmeden önce kapsayıcı ve başlık bilgisi
    /// ([`AvifPrepared::from_vec`]) tek geçişte ayrıştırılır; böylece `dimensions`, `color_type`,
    /// metaveri ve `set_limits` kontrolleri piksel tahsisi yapılmadan yürütülebilir.
    pub fn new(reader: R) -> ImageResult<Self> {
        Self::with_config(reader, AvifDecodeConfig::default())
    }

    /// Özel [`AvifDecodeConfig`] yapılandırmasıyla yeni bir [`AvifDecoder`] oluşturur.
    ///
    /// `config.allocation_limit` tanımlıysa okuma başlamadan önce `Read::take(limit + 1)` ile
    /// girdi boyutu sınırlandırılır; girdi `limit` bayttan büyükse başlık ayrıştırılmadan
    /// `LimitErrorKind::InsufficientMemory` döndürülür. Kalan bütçe (`limit - raw_bytes.len()`)
    /// [`AvifPrepared::from_vec`] çağrısına iletilir.
    pub fn with_config(mut reader: R, config: AvifDecodeConfig) -> ImageResult<Self> {
        let mut raw_bytes = Vec::new();
        let header_cfg = if let Some(limit) = config.allocation_limit {
            let max_read = (limit as u64).saturating_add(1);
            (&mut reader).take(max_read).read_to_end(&mut raw_bytes)?;
            if raw_bytes.len() > limit {
                return Err(ImageError::Limits(LimitError::from_kind(
                    LimitErrorKind::InsufficientMemory,
                )));
            }
            AvifDecodeConfig {
                allocation_limit: Some(limit - raw_bytes.len()),
                ..config.clone()
            }
        } else {
            reader.read_to_end(&mut raw_bytes)?;
            config.clone()
        };
        let prepared =
            AvifPrepared::from_vec(raw_bytes, &header_cfg).map_err(map_av1oxide_error)?;
        Ok(Self {
            prepared,
            config,
            limits: Limits::no_limits(),
            _marker: PhantomData,
        })
    }

    /// Ayrıştırılmış AVIF üst verisine salt-okunur erişim döndürür.
    #[must_use]
    pub const fn info(&self) -> &AvifInfo {
        self.prepared.info()
    }

    fn held_header_bytes(&self) -> usize {
        self.prepared.held_bytes()
    }

    fn effective_allocation_limit(&self) -> Option<usize> {
        let from_limits = self
            .limits
            .max_alloc
            .map(|v| usize::try_from(v).unwrap_or(usize::MAX));
        match (self.config.allocation_limit, from_limits) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }
}

impl<R: Read> ImageDecoder for AvifDecoder<R> {
    fn dimensions(&self) -> (u32, u32) {
        let info = self.prepared.info();
        (info.width, info.height)
    }

    fn color_type(&self) -> ColorType {
        let info = self.prepared.info();
        match (info.bit_depth, info.has_alpha) {
            (BitDepth::Eight, false) => ColorType::Rgb8,
            (BitDepth::Eight, true) => ColorType::Rgba8,
            (BitDepth::Ten | BitDepth::Twelve, false) => ColorType::Rgb16,
            (BitDepth::Ten | BitDepth::Twelve, true) => ColorType::Rgba16,
        }
    }

    fn original_color_type(&self) -> ExtendedColorType {
        self.color_type().into()
    }

    fn icc_profile(&mut self) -> ImageResult<Option<Vec<u8>>> {
        Ok(self.prepared.info().icc_profile.clone())
    }

    fn exif_metadata(&mut self) -> ImageResult<Option<Vec<u8>>> {
        Ok(self.prepared.info().exif.clone())
    }

    fn xmp_metadata(&mut self) -> ImageResult<Option<Vec<u8>>> {
        Ok(self.prepared.info().xmp.clone())
    }

    fn set_limits(&mut self, limits: Limits) -> ImageResult<()> {
        limits.check_support(&image::LimitSupport::default())?;
        let (width, height) = self.dimensions();
        limits.check_dimensions(width, height)?;
        if let Some(max_alloc) = limits.max_alloc {
            let held = self.held_header_bytes() as u64;
            if self.total_bytes() > max_alloc || held > max_alloc {
                return Err(ImageError::Limits(LimitError::from_kind(
                    LimitErrorKind::InsufficientMemory,
                )));
            }
        }
        self.limits = limits;
        Ok(())
    }

    fn read_image(self, buf: &mut [u8]) -> ImageResult<()>
    where
        Self: Sized,
    {
        let expected_bytes = self.total_bytes();
        assert_eq!(
            buf.len() as u64,
            expected_bytes,
            "çıktı tampon boyutu image::ImageDecoder::total_bytes() ile eşleşmiyor"
        );

        let (w, h) = self.dimensions();
        self.limits.check_dimensions(w, h)?;
        let held_total = self.held_header_bytes() as u64;
        if let Some(max_alloc) = self.limits.max_alloc
            && (expected_bytes > max_alloc || held_total > max_alloc)
        {
            return Err(ImageError::Limits(LimitError::from_kind(
                LimitErrorKind::InsufficientMemory,
            )));
        }

        let effective_limit = self.effective_allocation_limit();
        let bit_depth = self.prepared.info().bit_depth;
        let has_alpha = self.prepared.info().has_alpha;
        let raw_len = self.prepared.raw_len();
        let mut cfg = self.config;
        if let Some(max_w) = self.limits.max_image_width
            && let Some(max_h) = self.limits.max_image_height
        {
            let pixel_limit = u64::from(max_w).saturating_mul(u64::from(max_h));
            cfg.max_pixels = cfg.max_pixels.min(pixel_limit);
        }

        // `raw_bytes` tamponunu toplam bütçeden düşerek tek ayrıştırmalı çözücüye geçir;
        // `decode_with(self, ...)` tamamlandığında ham baytlar ve kapsayıcı anında bırakılır.
        if let Some(limit) = effective_limit {
            if raw_len > limit {
                return Err(ImageError::Limits(LimitError::from_kind(
                    LimitErrorKind::InsufficientMemory,
                )));
            }
            cfg.allocation_limit = Some(limit - raw_len);
        } else {
            cfg.allocation_limit = None;
        }

        let img = self
            .prepared
            .decode_with(&cfg)
            .map_err(map_av1oxide_error)?;

        match (bit_depth, has_alpha) {
            (BitDepth::Eight, false) => img.write_rgb8_into(buf).map_err(map_av1oxide_error)?,
            (BitDepth::Eight, true) => img.write_rgba8_into(buf).map_err(map_av1oxide_error)?,
            (BitDepth::Ten | BitDepth::Twelve, false) => img
                .write_rgb16_scaled_ne_into(buf)
                .map_err(map_av1oxide_error)?,
            (BitDepth::Ten | BitDepth::Twelve, true) => img
                .write_rgba16_scaled_ne_into(buf)
                .map_err(map_av1oxide_error)?,
        }
        Ok(())
    }

    fn read_image_boxed(self: Box<Self>, buf: &mut [u8]) -> ImageResult<()> {
        (*self).read_image(buf)
    }
}

fn map_av1oxide_error(err: Error) -> ImageError {
    match err {
        Error::BudgetExceeded { .. } | Error::AllocationFailed { .. } => {
            ImageError::Limits(LimitError::from_kind(LimitErrorKind::InsufficientMemory))
        }
        Error::SizeOverflow { .. } => {
            ImageError::Limits(LimitError::from_kind(LimitErrorKind::DimensionError))
        }
        other => ImageError::Decoding(DecodingError::new(
            ImageFormatHint::Exact(ImageFormat::Avif),
            other,
        )),
    }
}

/// Bellek içi AVIF bayt dilimini doğrudan [`DynamicImage`] olarak çözer.
pub fn decode_avif_bytes(data: &[u8]) -> ImageResult<DynamicImage> {
    let decoder = AvifDecoder::new(Cursor::new(data))?;
    DynamicImage::from_decoder(decoder)
}

/// `image::hooks` sistemine `.avif` uzantısı için `av1oxide-image` çözücü kancasını kaydeder.
///
/// Kanca ilk kez eklendiyse `true`, `.avif` için daha önce bir kanca tanımlanmışsa `false` döner.
pub fn register_decoding_hook() -> bool {
    image::hooks::register_decoding_hook(
        "avif".into(),
        Box::new(|reader| Ok(Box::new(AvifDecoder::new(reader)?))),
    )
}

#[cfg(test)]
mod tests {
    use super::AvifDecoder;
    use av1oxide::BitDepth;
    use image::ColorType;

    #[test]
    fn color_type_mapping_covers_8_10_12_bit_and_alpha() {
        let cases = [
            (BitDepth::Eight, false, ColorType::Rgb8),
            (BitDepth::Eight, true, ColorType::Rgba8),
            (BitDepth::Ten, false, ColorType::Rgb16),
            (BitDepth::Ten, true, ColorType::Rgba16),
            (BitDepth::Twelve, false, ColorType::Rgb16),
            (BitDepth::Twelve, true, ColorType::Rgba16),
        ];
        for (bd, has_alpha, expected) in cases {
            let ct = match (bd, has_alpha) {
                (BitDepth::Eight, false) => ColorType::Rgb8,
                (BitDepth::Eight, true) => ColorType::Rgba8,
                (BitDepth::Ten | BitDepth::Twelve, false) => ColorType::Rgb16,
                (BitDepth::Ten | BitDepth::Twelve, true) => ColorType::Rgba16,
            };
            assert_eq!(ct, expected);
        }
        let _ = core::any::type_name::<AvifDecoder<std::io::Empty>>();
    }
}
