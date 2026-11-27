//! [`image::ImageDecoder`] (`image 0.25`) integration crate for [`av1oxide::avif`].
//!
//! This crate compiles under `#![forbid(unsafe_code)]`, integrates AVIF decoding
//! with the `image` crate ecosystem without adding an `image` dependency to the
//! core `av1oxide` crate, and depends on `image` with `default-features = false`.

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

/// [`image::ImageDecoder`] implementation backed by [`av1oxide::avif`].
///
/// `8-bit` AVIF images decode as `ColorType::Rgb8` / `ColorType::Rgba8`. `10-bit`
/// and `12-bit` AVIF images are scaled to the full `[0..=65535]` dynamic range
/// required by the `image` crate contract and returned as `ColorType::Rgb16` /
/// `ColorType::Rgba16` in native byte order.
///
/// ## Single-Pass Parsing and Direct Output Writing
///
/// - [`AvifDecoder::with_config`] reads the stream once and parses container boxes
///   and metadata (including `cmin` decompression) in a single pass via [`AvifPrepared`].
/// - [`ImageDecoder::read_image`] decodes image planes without a second container parse,
///   releases the raw input bytes immediately, and writes converted pixels directly into
///   the caller-provided `buf: &mut [u8]` slice via `AvifImage::write_*_into` with zero
///   intermediate RGB/RGBA allocation.
///
/// ## Memory Budget Scope (`allocation_limit` and `Limits::max_alloc`)
///
/// - [`AvifDecoder::with_config`] reads the input stream into a `Vec<u8>` whose capacity
///   is grown with `try_reserve_exact` capped at `limit` (never doubling past `limit`),
///   then forwards the remaining budget (`limit - raw_bytes.capacity()`) to
///   [`AvifPrepared::from_vec`] so `cmin`/`infe` decompression and `ICC`/`Exif`/`XMP`
///   metadata allocations remain within the same ceiling.
/// - [`ImageDecoder::set_limits`] validates both the output size (`total_bytes()`) and
///   the input/container/metadata memory already held by the adapter against `limits.max_alloc`.
/// - Metadata copies returned by [`ImageDecoder::icc_profile`],
///   [`ImageDecoder::exif_metadata`], and [`ImageDecoder::xmp_metadata`] are also checked
///   against the effective allocation limit.
/// - During [`ImageDecoder::read_image`], internal decoder allocations are capped by
///   subtracting `raw_capacity()` from the effective ceiling; the caller-owned `&mut [u8]`
///   output buffer passed to `read_image` is outside the library's internal allocation counter.
#[derive(Clone, Debug)]
pub struct AvifDecoder<R: Read> {
    prepared: AvifPrepared,
    config: AvifDecodeConfig,
    limits: Limits,
    _marker: PhantomData<R>,
}

impl<R: Read> AvifDecoder<R> {
    /// Creates a new [`AvifDecoder`] using the default [`AvifDecodeConfig`].
    ///
    /// Stream bytes are read and container/header metadata are parsed in a single pass
    /// ([`AvifPrepared::from_vec`]) before pixel decoding starts, allowing `dimensions`,
    /// `color_type`, metadata access, and `set_limits` checks prior to pixel buffer allocation.
    pub fn new(reader: R) -> ImageResult<Self> {
        Self::with_config(reader, AvifDecodeConfig::default())
    }

    /// Creates a new [`AvifDecoder`] with a custom [`AvifDecodeConfig`].
    ///
    /// When `config.allocation_limit` is set, input reading uses a fixed stack buffer and
    /// `try_reserve_exact` so the input vector's capacity never exceeds `limit`; if the
    /// stream exceeds `limit` bytes, `LimitErrorKind::InsufficientMemory` is returned before
    /// header parsing. The remaining budget (`limit - raw_bytes.capacity()`) is passed to
    /// [`AvifPrepared::from_vec`].
    pub fn with_config(mut reader: R, config: AvifDecodeConfig) -> ImageResult<Self> {
        let (raw_bytes, header_cfg) = if let Some(limit) = config.allocation_limit {
            let raw_bytes = read_bounded_exact_cap(&mut reader, limit)?;
            let remaining = limit.checked_sub(raw_bytes.capacity()).ok_or_else(|| {
                ImageError::Limits(LimitError::from_kind(LimitErrorKind::InsufficientMemory))
            })?;
            (
                raw_bytes,
                AvifDecodeConfig {
                    allocation_limit: Some(remaining),
                    ..config.clone()
                },
            )
        } else {
            let mut raw_bytes = Vec::new();
            reader.read_to_end(&mut raw_bytes)?;
            (raw_bytes, config.clone())
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

    /// Returns read-only access to the parsed AVIF header and metadata.
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
        clone_metadata_budgeted(
            self.prepared.info().icc_profile.as_deref(),
            self.held_header_bytes(),
            self.effective_allocation_limit(),
        )
    }

    fn exif_metadata(&mut self) -> ImageResult<Option<Vec<u8>>> {
        clone_metadata_budgeted(
            self.prepared.info().exif.as_deref(),
            self.held_header_bytes(),
            self.effective_allocation_limit(),
        )
    }

    fn xmp_metadata(&mut self) -> ImageResult<Option<Vec<u8>>> {
        clone_metadata_budgeted(
            self.prepared.info().xmp.as_deref(),
            self.held_header_bytes(),
            self.effective_allocation_limit(),
        )
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
        let raw_cap = self.prepared.raw_capacity();
        let mut cfg = self.config;
        if let Some(max_w) = self.limits.max_image_width
            && let Some(max_h) = self.limits.max_image_height
        {
            let pixel_limit = u64::from(max_w).saturating_mul(u64::from(max_h));
            cfg.max_pixels = cfg.max_pixels.min(pixel_limit);
        }

        // `raw_bytes` tamponunun gerçek kapasitesini toplam bütçeden düşerek tek ayrıştırmalı
        // çözücüye geçir; `decode_with(self, ...)` tamamlandığında ham baytlar ve kapsayıcı bırakılır.
        if let Some(limit) = effective_limit {
            if raw_cap > limit {
                return Err(ImageError::Limits(LimitError::from_kind(
                    LimitErrorKind::InsufficientMemory,
                )));
            }
            cfg.allocation_limit = Some(limit - raw_cap);
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

fn insufficient_memory() -> ImageError {
    ImageError::Limits(LimitError::from_kind(LimitErrorKind::InsufficientMemory))
}

/// Reads up to `limit` bytes from `reader` without allowing the output `Vec`'s capacity
/// to exceed `limit`, returning `InsufficientMemory` if the stream contains more than `limit` bytes.
fn read_bounded_exact_cap<R: Read>(reader: &mut R, limit: usize) -> ImageResult<Vec<u8>> {
    if limit == 0 {
        let mut probe = [0u8; 1];
        loop {
            match reader.read(&mut probe) {
                Ok(0) => return Ok(Vec::new()),
                Ok(_) => return Err(insufficient_memory()),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
    }

    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    while buf.len() < limit {
        let to_read = (limit - buf.len()).min(chunk.len());
        let n = match reader.read(&mut chunk[..to_read]) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        };
        if buf.capacity() - buf.len() < n {
            let needed = buf.len() + n;
            let target_cap = buf
                .capacity()
                .saturating_mul(2)
                .max(256)
                .min(limit)
                .max(needed);
            buf.try_reserve_exact(target_cap - buf.len())
                .map_err(|_| insufficient_memory())?;
        }
        buf.extend_from_slice(&chunk[..n]);
    }

    if buf.len() == limit {
        let mut probe = [0u8; 1];
        loop {
            match reader.read(&mut probe) {
                Ok(0) => break,
                Ok(_) => return Err(insufficient_memory()),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
    }

    buf.shrink_to_fit();
    Ok(buf)
}

fn clone_metadata_budgeted(
    meta: Option<&[u8]>,
    held_header_bytes: usize,
    effective_limit: Option<usize>,
) -> ImageResult<Option<Vec<u8>>> {
    let Some(bytes) = meta else {
        return Ok(None);
    };
    if let Some(limit) = effective_limit
        && held_header_bytes.saturating_add(bytes.len()) > limit
    {
        return Err(insufficient_memory());
    }
    let mut out = Vec::new();
    out.try_reserve_exact(bytes.len())
        .map_err(|_| insufficient_memory())?;
    out.extend_from_slice(bytes);
    Ok(Some(out))
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

/// Decodes an in-memory AVIF byte slice directly into a [`DynamicImage`].
pub fn decode_avif_bytes(data: &[u8]) -> ImageResult<DynamicImage> {
    let decoder = AvifDecoder::new(Cursor::new(data))?;
    DynamicImage::from_decoder(decoder)
}

/// Registers the `av1oxide-image` decoding hook for the `.avif` extension with `image::hooks`.
///
/// Returns `true` if the hook was newly installed, or `false` if a hook for `.avif` was already registered.
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
