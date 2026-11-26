//! Saf Rust, `%100 #![forbid(unsafe_code)]` `.avif` (AV1 Image File Format)
//! okuyucu ve görüntü çözücü katmanı.
//!
//! Bu modül `ISO/IEC 14496-12` (ISOBMFF), `ISO/IEC 23008-12` (HEIF/MIAF) ve
//! `AOMedia AVIF v1.1.0` spesifikasyonlarına uygun olarak:
//!
//! - `ftyp` marka doğrulaması ([`probe`]) ve `meta` kutusu başlık incelemesi ([`parse_info`]),
//! - Tekil `av01` görüntü öğeleri, `grid` karo ızgaraları (tek ve çok-thread) ve `iden` türetilmiş görüntüleri,
//! - Yardımcı alfa (`auxl` / `auxC`) kanalı ve ön-çarpımlı alfa (`prem`) metaverisi,
//! - `clap` (temiz açıklık kırpma), `irot` (`90°/180°/270°` döndürme) ve `imir` (aynalama) dönüşümleri,
//! - ITU-T H.273 renk uzayı dönüşümleri (`Identity/GBR`, `BT.709`, `BT.601`, `BT.2020`, `YCgCo`, `Monochrome`)
//!   ile düzlemsel ([`AvifImage`]) ve paketlenmiş RGB/RGBA ([`RgbImage`], [`RgbaImage`], [`Rgb16Image`], [`Rgba16Image`])
//!   çıktıları üretir.

#![forbid(unsafe_code)]

pub(crate) mod boxes;
pub(crate) mod color;
pub(crate) mod decode;
pub mod image;

pub use boxes::probe;
pub use decode::{decode, decode_rgb8, decode_rgba8, decode_with, parse_info};
pub use image::{
    AvifDecodeConfig, AvifImage, AvifInfo, AvifPlane, CleanAperture, ColorInfo, ContentLightLevel,
    MasteringDisplay, PixelAspectRatio, Rgb16Image, RgbImage, Rgba16Image, RgbaImage,
};
