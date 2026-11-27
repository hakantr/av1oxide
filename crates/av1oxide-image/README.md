**English** | **[Türkçe](README.tr.md)**

# `av1oxide-image`

`av1oxide-image` is a standalone, 100% `#![forbid(unsafe_code)]` adapter crate that integrates the [`av1oxide::avif`](https://crates.io/crates/av1oxide) AVIF image decoder with the Rust [`image`](https://crates.io/crates/image) (`0.25`) `ImageDecoder` trait.

Keeping the adapter in a separate crate preserves the dependency isolation of the core `av1oxide` crate (no `image` dependency in `av1oxide`), while `av1oxide-image` depends on `image` with `default-features = false` so no additional third-party image codecs are pulled in.

## Features

- `image::ImageDecoder` implementation (`AvifDecoder<R: std::io::Read>`):
  - **Pre-read memory budget enforcement:** When `AvifDecodeConfig::allocation_limit` is configured, input ingestion is bounded with `Read::take(limit + 1)` before reading into memory, and `image::Limits` (`max_image_width`, `max_image_height`, `max_alloc`) are validated before pixel buffers are allocated.
  - **Single-pass container parsing:** `AvifDecoder::with_config` parses container boxes and metadata (including `cmin` decompression) once via `AvifPrepared`, avoiding any redundant container parse during `ImageDecoder::read_image`.
  - **Zero-copy-intermediate direct output writing:** Decoded planes are converted and written directly into the caller-supplied `buf: &mut [u8]` slice (`write_rgb8_into`, `write_rgba8_into`, `write_rgb16_scaled_ne_into`, `write_rgba16_scaled_ne_into`) without allocating an intermediate RGB/RGBA vector.
  - **Color types:** `8-bit` images decode to `ColorType::Rgb8` / `ColorType::Rgba8`; `10-bit` and `12-bit` images are scaled to the full `[0..=65535]` dynamic range required by `image` and returned as `ColorType::Rgb16` / `ColorType::Rgba16` in native endianness.
  - **Embedded metadata:** Exposes embedded `ICC`, `Exif`, and `XMP` payloads via `icc_profile`, `exif_metadata`, and `xmp_metadata`.
- `decode_avif_bytes(&[u8])` helper for decoding an in-memory AVIF byte slice directly into a `DynamicImage`.
- `register_decoding_hook()` helper for registering the `.avif` extension with `image::hooks::register_decoding_hook`.

## Usage

```rust,no_run
use std::fs::File;
use av1oxide_image::{AvifDecoder, decode_avif_bytes, register_decoding_hook};
use image::DynamicImage;

// 1. Decode directly from an in-memory byte slice
let bytes = std::fs::read("example.avif")?;
let img: DynamicImage = decode_avif_bytes(&bytes)?;
println!("{}x{}", img.width(), img.height());

// 2. Decode from a reader via the ImageDecoder trait
let decoder = AvifDecoder::new(File::open("example.avif")?)?;
let img2 = DynamicImage::from_decoder(decoder)?;
println!("{}x{}", img2.width(), img2.height());

// 3. Register the .avif decoding hook for image::ImageReader
let _ = register_decoding_hook();
# Ok::<(), Box<dyn std::error::Error>>(())
```

## License

`BSD-2-Clause` — see [LICENSE](LICENSE).
