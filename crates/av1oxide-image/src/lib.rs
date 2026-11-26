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

use av1oxide::avif::{AvifDecodeConfig, AvifInfo, decode_with, parse_info};
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
#[derive(Clone, Debug)]
pub struct AvifDecoder<R: Read> {
    raw_bytes: Vec<u8>,
    info: AvifInfo,
    config: AvifDecodeConfig,
    limits: Limits,
    _marker: PhantomData<R>,
}

impl<R: Read> AvifDecoder<R> {
    /// Varsayılan [`AvifDecodeConfig`] ile yeni bir [`AvifDecoder`] oluşturur.
    ///
    /// Akış baytları okunur ve piksel çözümüne geçilmeden önce yalnızca başlık
    /// ([`parse_info`]) ayrıştırılır; böylece `dimensions`, `color_type`, metaveri
    /// ve `set_limits` kontrolleri piksel tahsisi yapılmadan yürütülebilir.
    pub fn new(reader: R) -> ImageResult<Self> {
        Self::with_config(reader, AvifDecodeConfig::default())
    }

    /// Özel [`AvifDecodeConfig`] yapılandırmasıyla yeni bir [`AvifDecoder`] oluşturur.
    pub fn with_config(mut reader: R, config: AvifDecodeConfig) -> ImageResult<Self> {
        let mut raw_bytes = Vec::new();
        reader.read_to_end(&mut raw_bytes)?;
        let info = parse_info(&raw_bytes).map_err(map_av1oxide_error)?;
        Ok(Self {
            raw_bytes,
            info,
            config,
            limits: Limits::no_limits(),
            _marker: PhantomData,
        })
    }

    /// Ayrıştırılmış AVIF üst verisine salt-okunur erişim döndürür.
    #[must_use]
    pub const fn info(&self) -> &AvifInfo {
        &self.info
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
        (self.info.width, self.info.height)
    }

    fn color_type(&self) -> ColorType {
        match (self.info.bit_depth, self.info.has_alpha) {
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
        Ok(self.info.icc_profile.clone())
    }

    fn exif_metadata(&mut self) -> ImageResult<Option<Vec<u8>>> {
        Ok(self.info.exif.clone())
    }

    fn xmp_metadata(&mut self) -> ImageResult<Option<Vec<u8>>> {
        Ok(self.info.xmp.clone())
    }

    fn set_limits(&mut self, limits: Limits) -> ImageResult<()> {
        limits.check_support(&image::LimitSupport::default())?;
        let (width, height) = self.dimensions();
        limits.check_dimensions(width, height)?;
        if let Some(max_alloc) = limits.max_alloc
            && self.total_bytes() > max_alloc
        {
            return Err(ImageError::Limits(LimitError::from_kind(
                LimitErrorKind::InsufficientMemory,
            )));
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
        if let Some(max_alloc) = self.limits.max_alloc
            && expected_bytes > max_alloc
        {
            return Err(ImageError::Limits(LimitError::from_kind(
                LimitErrorKind::InsufficientMemory,
            )));
        }

        let mut cfg = self.config.clone();
        cfg.allocation_limit = self.effective_allocation_limit();
        if let Some(max_w) = self.limits.max_image_width
            && let Some(max_h) = self.limits.max_image_height
        {
            let pixel_limit = u64::from(max_w).saturating_mul(u64::from(max_h));
            cfg.max_pixels = cfg.max_pixels.min(pixel_limit);
        }

        let img = decode_with(&self.raw_bytes, &cfg).map_err(map_av1oxide_error)?;

        match (self.info.bit_depth, self.info.has_alpha) {
            (BitDepth::Eight, false) => {
                let rgb = img.to_rgb8().map_err(map_av1oxide_error)?;
                buf.copy_from_slice(&rgb.pixels);
            }
            (BitDepth::Eight, true) => {
                let rgba = img.to_rgba8().map_err(map_av1oxide_error)?;
                buf.copy_from_slice(&rgba.pixels);
            }
            (BitDepth::Ten, false) => {
                let rgb16 = img.to_rgb16().map_err(map_av1oxide_error)?;
                write_scaled_u16_ne(&rgb16.pixels, scale_10_to_16, buf);
            }
            (BitDepth::Ten, true) => {
                let rgba16 = img.to_rgba16().map_err(map_av1oxide_error)?;
                write_scaled_u16_ne(&rgba16.pixels, scale_10_to_16, buf);
            }
            (BitDepth::Twelve, false) => {
                let rgb16 = img.to_rgb16().map_err(map_av1oxide_error)?;
                write_scaled_u16_ne(&rgb16.pixels, scale_12_to_16, buf);
            }
            (BitDepth::Twelve, true) => {
                let rgba16 = img.to_rgba16().map_err(map_av1oxide_error)?;
                write_scaled_u16_ne(&rgba16.pixels, scale_12_to_16, buf);
            }
        }
        Ok(())
    }

    fn read_image_boxed(self: Box<Self>, buf: &mut [u8]) -> ImageResult<()> {
        (*self).read_image(buf)
    }
}

/// 10-bit (`0..=1023`) örneği tam 16-bit (`0..=65535`) aralığına yuvarlayarak ölçekler.
#[inline]
const fn scale_10_to_16(v: u16) -> u16 {
    let clamped = if v > 1023 { 1023 } else { v };
    ((clamped as u32 * 65_535 + 511) / 1023) as u16
}

/// 12-bit (`0..=4095`) örneği tam 16-bit (`0..=65535`) aralığına yuvarlayarak ölçekler.
#[inline]
const fn scale_12_to_16(v: u16) -> u16 {
    let clamped = if v > 4095 { 4095 } else { v };
    ((clamped as u32 * 65_535 + 2047) / 4095) as u16
}

fn write_scaled_u16_ne(samples: &[u16], scale: fn(u16) -> u16, out: &mut [u8]) {
    for (&sample, dst) in samples.iter().zip(out.as_chunks_mut::<2>().0.iter_mut()) {
        *dst = scale(sample).to_ne_bytes();
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
    use super::{scale_10_to_16, scale_12_to_16};

    #[test]
    fn high_bitdepth_scaling_endpoints_and_monotonicity() {
        assert_eq!(scale_10_to_16(0), 0);
        assert_eq!(scale_10_to_16(1023), 65_535);
        assert_eq!(scale_10_to_16(512), 32_800);
        for v in 0..1023u16 {
            assert!(scale_10_to_16(v) < scale_10_to_16(v + 1));
        }

        assert_eq!(scale_12_to_16(0), 0);
        assert_eq!(scale_12_to_16(4095), 65_535);
        assert_eq!(scale_12_to_16(2048), 32_776);
        for v in 0..4095u16 {
            assert!(scale_12_to_16(v) < scale_12_to_16(v + 1));
        }
    }
}
