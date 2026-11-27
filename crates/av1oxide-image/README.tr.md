**[English](README.md)** | **Türkçe**

# `av1oxide-image`

`av1oxide-image`, [`av1oxide::avif`](https://crates.io/crates/av1oxide) AVIF görüntü çözücüsünü Rust [`image`](https://crates.io/crates/image) (`0.25`) ekosisteminin `ImageDecoder` arayüzüne bağlayan bağımsız ve `%100 #![forbid(unsafe_code)]` bir adaptör paketidir.

Ana `av1oxide` paketinin bağımlılık ağacı `image` bağımlılığı almadan yalıtılmış kalır; `av1oxide-image` ise `image` paketini `default-features = false` ile çekerek ek üçüncü taraf görüntü kodeği getirmez.

## Özellikler

- `image::ImageDecoder` gerçekleştirimi (`AvifDecoder<R: std::io::Read>`):
  - **Okuma öncesi bellek bütçesi denetimi:** `AvifDecodeConfig::allocation_limit` tanımlıysa akış okunmadan önce `Read::take(limit + 1)` ile sınırlandırılır; `Limits` (`max_image_width`, `max_image_height`, `max_alloc`) piksel tahsisinden önce denetlenir.
  - **Tek geçişli ayrıştırma:** `AvifPrepared` ile kapsayıcı ve üst veri (`cmin` açma dâhil) tek seferde ayrıştırılır; `read_image` aşamasında ikinci kez kutu ayrıştırması yapılmaz.
  - **Doğrudan çıktı tamponuna yazım:** Çözülen piksel düzlemleri ara RGB/RGBA vektörü ayrılmadan doğrudan çağıranın `buf: &mut [u8]` dilimine yazılır (`write_rgb8_into`, `write_rgba8_into`, `write_rgb16_scaled_ne_into`, `write_rgba16_scaled_ne_into`).
  - **Renk biçimleri:** `8-bit` görüntüler için `ColorType::Rgb8` / `ColorType::Rgba8`; `10-bit` ve `12-bit` görüntüler için `image` sözleşmesine uygun tam `[0..=65535]` dinamik aralığına ölçeklenmiş `ColorType::Rgb16` / `ColorType::Rgba16` (yerel bayt sırası).
  - **Üst veri erişimi:** Gömülü `ICC`, `Exif` ve `XMP` verileri (`icc_profile`, `exif_metadata`, `xmp_metadata`).
- Doğrudan bayt diliminden `DynamicImage` çözen `decode_avif_bytes(&[u8])` yardımcısı.
- `image::hooks::register_decoding_hook` üzerinden `.avif` uzantı kancası kaydı (`register_decoding_hook()`).

## Kullanım

```rust,no_run
use std::fs::File;
use av1oxide_image::{AvifDecoder, decode_avif_bytes, register_decoding_hook};
use image::DynamicImage;

// 1. Doğrudan bayt diliminden çözme
let bytes = std::fs::read("example.avif")?;
let img: DynamicImage = decode_avif_bytes(&bytes)?;
println!("{}x{}", img.width(), img.height());

// 2. ImageDecoder arayüzü ile dosya akışından çözme
let decoder = AvifDecoder::new(File::open("example.avif")?)?;
let img2 = DynamicImage::from_decoder(decoder)?;
println!("{}x{}", img2.width(), img2.height());

// 3. image::ImageReader için .avif kancasını kaydetme
let _ = register_decoding_hook();
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Lisans

`BSD-2-Clause` — bkz. [LICENSE](LICENSE).
