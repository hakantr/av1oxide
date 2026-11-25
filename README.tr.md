# av1oxide (`v0.4.0`)

[English](README.md) | **Türkçe**

`av1oxide`, VideoLAN [`dav1d`](https://code.videolan.org/videolan/dav1d) (`1.5.4`) AV1 video çözücüsünün **%100 kararlı Rust** (`rustc 1.100.0`, Edition 2024), **sıfır üçüncü taraf medya veya kodek bağımlılığı**, **`libdav1d` (`7.0.0`) ikili uyumlu C-ABI katmanı** ve **saf Rust `.avif` görüntü çözücü katmanı** ile yeniden gerçekleştirimidir.

- **Profiller ve Bit Derinlikleri:** Main (`0`), High (`1`), Professional (`2`); `8-bit`, `10-bit`, `12-bit`; `4:2:0`, `4:2:2`, `4:4:4` ve `Monochrome` (`4:0:0`).
- **AV1 Hattı:** OBU ayrıştırma, Sequence/Frame Header, MSAC aritmetik entropi çözücü, ters dönüşümler (`ITX`), intra tahmin (`IPRED`, CfL, Palette), hareket telafisi (`MC`, Warp, Superres/Resize, IntraBC, `REFMVS`), halka içi filtreler (`LF`, `CDEF`, `LR` Wiener/SGR) ve Film Grain (`FG`) sentezi.
- **Bit-Birebir Doğruluk:** `dav1d-test-data` (`19/19` akış, `964` kare) ve Argon Streams `v2.1.1` (`2 762/2 762` akış, `36 700` kare) olmak üzere toplam **`2 781 / 2 781` akış (`37 664 / 37 664` kare)** üzerinde tüm CPU yetenek düzeylerinde ve çok-thread kiplerinde `dav1d 1.5.4` ile `%100` MD5 bit-birebir eşitlik.
- **Donanım Yetenek Jetonlu SIMD (`767` DSP Yuvası):**
  - `x86_64` (`SSE2`, `SSSE3`, `SSE4.1`, `AVX2`, `AVX-512ICL`): `756 / 767` SIMD yuvası
  - `aarch64` (`NEON`, `DOTPROD`): `701 / 767` SIMD yuvası
  - `wasm32` (`+simd128`): `%100 #![forbid(unsafe_code)]` ile `31 / 767` SIMD yuvası
  - İkincil mimariler (`arm32`, `riscv64`, `loongarch64`, `powerpc64le`): `%100` güvenli skaler geri-düşüş
- **Çok-Thread ve Bellek Bütçesi:** Kare içi (`Tile`, süperblok satırı, halka içi filtre) ve kareler arası (`FrameSlot`) paralellik (`Config::n_threads`, `Config::max_frame_delay`); tahsis öncesi piksel ve bellek tavanı (`Config::frame_size_limit`, `Config::allocation_limit`).
- **`libdav1d` (`7.0.0`) Uyumlu C-ABI (`capi`):** 19 resmî `DAV1D_API` fonksiyonunun tamamını (`dav1d_open`, `dav1d_send_data`, `dav1d_get_picture`, `dav1d_apply_grain`, `dav1d_parse_sequence_header`, vb.) ve resmî `include/dav1d/*.h` C başlıklarını içerir.
- **Saf Rust `.avif` Görüntü Okuyucu (`avif`):** `%100 #![forbid(unsafe_code)]` ISOBMFF/HEIF/MIAF/AVIF kapsayıcı ayrıştırıcısı; tekil `av01`, çok-thread `grid` karo birleştirme, `iden` türetilmiş görüntüler, yardımcı alfa (`auxl`/`auxC`) kanalı, `clap`/`irot`/`imir` dönüşümleri, `ICC`/`Exif`/`XMP` metaverisi (bütçe ve oran korumalı `defl`/`zlib`/`gzip` sıkıştırılmış HEIF metaveri öğeleri ve `cmin` kutuları dahil) ve ITU-T H.273 renk dönüşümleri (`RGB8`/`RGBA8`/`RGB16`/`RGBA16`) — lisans denetimli `AOMediaCodec/av1-avif` ve `AOMediaCodec/libavif` test korpusundaki `156 / 156` geçerli `.avif` görüntüsünde referans C `libavif` ile düzlemsel (`Y`, `U`, `V`, `Alpha`), `RGBA8` ve `RGBA16` çıktılarında `%100` MD5 bit-birebir eşitlik.

---

## Bağımlılık Politikası

`av1oxide`, hiçbir üçüncü taraf medya, kodek, görüntü işleme veya C medya kütüphanesine bağımlı değildir; tüm AV1 bit akışı/entropi/DSP çözme, ISOBMFF/AVIF kapsayıcı ayrıştırma, ITU-T H.273 renk dönüşümü ve `libdav1d` C-ABI katmanları depo içinde gerçekleştirilmiştir. Kullanılan tüm çekirdek kütüphaneler iç uygulama ayrıntısı (`pub(crate)` / özel) olarak yalıtılmıştır; `av1oxide` genel (`pub`) API imzalarına hiçbir bağımlılık tipi yeniden dışa aktarılmaz veya sızdırılmaz.

Düşük seviyeli eşzamanlılık, bellek yerleşimi ve sıkıştırma açma ilkel tipleri için yalnızca `rust-lang` organizasyonu ve Rust çekirdek ekip üyeleri tarafından geliştirilen aşağıdaki temel kütüphaneler `default-features = false` ile kullanılır:

| Crate | Sürüm | Lisans | Kaynak | Özellikler (`default-features = false`) | Görev | Unsafe / Yerel Kod |
|---|---|---|---|---|---|---|
| `crossbeam-utils` | `0.8` (`0.8.23`) | `MIT OR Apache-2.0` | [crossbeam-rs/crossbeam](https://github.com/crossbeam-rs/crossbeam) | `["std"]` | Paylaşılan atomik sayaçların ve işçi yuvalarının (`thread::TaskBatch`, `thread::ScratchPool`, `frame::butce::Butce`) `CachePadded` ile önbellek satırı izolasyonu ve kilitsiz CAS döngülerinde `Backoff` bekleme kontrolü | Pure Rust (iç atomik/hizalama `unsafe`, `0` C kodu) |
| `parking_lot` | `0.12` (`0.12.5`) | `MIT OR Apache-2.0` | [Amanieu/parking_lot](https://github.com/Amanieu/parking_lot) | `[]` (`not(target_arch = "wasm32")`) | `thread::WorkerPool`, `thread::FrameSlot`, `api::Decoder` ve `decode::intra` içinde zehirlenmesiz (`non-poisoning`) kompakt kullanıcı uzayı `Mutex` ve `Condvar` (`sync`) | Pure Rust + Wasm dışı hedeflerde işletim sistemi futex / `libc` sistem çağrısı (`0` derlenen C dosyası) |
| `memchr` | `2` (`2.8.3`) | `Unlicense OR MIT` | [BurntSushi/memchr](https://github.com/BurntSushi/memchr) | `[]` | ISOBMFF NUL-sonlu dize ayrıştırma (`avif::boxes`) ve OBU metaveri kuyruk baytı taramasında (`api::Decoder`) SIMD/SWAR hızlandırmalı bayt arama (`memchr`, `memrchr`) | Pure Rust (iç SIMD/işaretçi `unsafe`, `0` C kodu) |
| `rustc-hash` | `2` (`2.1.3`) | `Apache-2.0 OR MIT` | [rust-lang/rustc-hash](https://github.com/rust-lang/rustc-hash) | `[]` | ISOBMFF `item_id` tabloları (`avif`) ve DSP tablo kurulumunda (`dsp`) hızlı deterministik tamsayı karması (`FxBuildHasher`) | Safe Rust (`0` `unsafe` blok, `0` C kodu) |
| `indexmap` | `2` (`2.14.2`) | `Apache-2.0 OR MIT` | [indexmap-rs/indexmap](https://github.com/indexmap-rs/indexmap) | `[]` | ISOBMFF `iinf`, `ipma`, `iloc` ve `iref` öğe tablolarında (`avif::boxes`, `avif::decode`, `dsp`) ekleme sırasını koruyan $O(1)$ tekil tablo ve küme motoru (`IndexMap`, `IndexSet`) | Pure Rust (`indexmap` `#![forbid(unsafe_code)]` + Miri onaylı `hashbrown`, `0` C kodu) |
| `flate2` | `1.1` (`1.1.11`) | `MIT OR Apache-2.0` | [rust-lang/flate2-rs](https://github.com/rust-lang/flate2-rs) | `["rust_backend"]` | Sıkıştırılmış HEIF metaveri öğeleri (`infe` `content_encoding`, `cmpC`) ve `cmin` kutuları (`avif::boxes`) için sınırlı `deflate`/`zlib`/`gzip` açıcısı | Safe Rust (`miniz_oxide` `#![forbid(unsafe_code)]` arka ucu + `crc32fast`, `0` C/`libz-sys` kodu) |

---

## Kurulum

`Cargo.toml` dosyanıza ekleyin:

```toml
[dependencies]
av1oxide = "0.4"
```

Kütüphane API belgelerini (`rustdoc`) yerel olarak üretmek için:

```bash
cargo doc -p av1oxide --no-deps --open
```

---

## Kütüphane Kullanımı

### 1. Temel OBU Paket Çözme ve YUV Düzlemlerini Okuma

```rust
use av1oxide::{BitDepth, Config, Decoder, Frame, Packet, PlaneType, Result};

fn decode_packets(packets: &[&[u8]]) -> Result<()> {
    let config = Config {
        n_threads: 4,
        max_frame_delay: 2,
        ..Config::default()
    };
    let mut decoder = Decoder::new(config)?;

    for (pts, payload) in packets.iter().enumerate() {
        decoder.send_packet(&Packet::with_timestamp(payload, pts as i64))?;
        while let Some(frame) = decoder.receive_frame()? {
            process_frame(&frame);
        }
    }

    // Akış sonunda boru hattında bekleyen kareleri boşalt.
    decoder.flush();
    while let Some(frame) = decoder.receive_frame()? {
        process_frame(&frame);
    }

    Ok(())
}

fn process_frame(frame: &Frame) {
    let y_plane = frame.plane(PlaneType::Y).expect("Y düzlemi");
    match frame.bit_depth() {
        BitDepth::Eight => {
            for y in 0..y_plane.height() {
                let _row: &[u8] = y_plane.row(y).expect("8-bit satır dilimi");
            }
        }
        BitDepth::Ten | BitDepth::Twelve => {
            // 10/12-bit örnekler satır içinde küçük-endian u16 çiftleri olarak saklanır.
            for y in 0..y_plane.height() {
                let row_bytes = y_plane.row(y).expect("16-bit satır baytları");
                for pair in row_bytes.chunks_exact(2) {
                    let _sample = u16::from_le_bytes([pair[0], pair[1]]);
                }
            }
        }
    }
}
```

### 2. Güvenilmeyen Girdi İçin Kaynak Bütçesi ve Hata Kurtarma (`reset`)

```rust
use av1oxide::{Config, Decoder, Error, Packet, Result};

fn decode_untrusted(payload: &[u8]) -> Result<()> {
    let config = Config {
        n_threads: 2,
        max_frame_delay: 1,
        frame_size_limit: 3840 * 2160,              // Maksimum 4K kare boyutu (piksel)
        allocation_limit: Some(128 * 1024 * 1024),  // Maksimum 128 MiB çalışma belleği
        ..Config::default()
    };
    let mut decoder = Decoder::new(config)?;

    match decoder.send_packet(&Packet::new(payload)) {
        Ok(()) => {
            while let Some(frame) = decoder.receive_frame()? {
                println!(
                    "Kare: {}x{}, tepe bellek: {} bayt",
                    frame.width(),
                    frame.height(),
                    decoder.allocation_peak()
                );
            }
        }
        Err(Error::BudgetExceeded { requested, limit }) => {
            eprintln!("Bellek bütçesi aşıldı: istenen={requested}, sınır={limit}");
            decoder.reset();
        }
        Err(Error::Bitstream(msg)) => {
            eprintln!("Bozuk bit akışı: {msg}");
            decoder.reset();
        }
        Err(err) => return Err(err),
    }

    Ok(())
}
```

### 3. `libdav1d` Uyumlu C-ABI Kullanımı (`--features capi`)

19 `dav1d_*` C sembolünü dışa aktaran paylaşımlı (`libav1oxide.so` / `.dylib` / `.dll`) ve statik (`libav1oxide.a`) kütüphaneleri üretmek için:

```bash
cargo rustc --release -p av1oxide --features capi --crate-type cdylib,staticlib
```

Herhangi bir C/C++ programını `crates/av1oxide/include` (`#include <dav1d/dav1d.h>`) ve `libav1oxide.a` / `libav1oxide.so` ile doğrudan bağlayabilirsiniz:

```c
#include <dav1d/dav1d.h>

int decode_obu_c(const uint8_t *obu_buf, size_t obu_len) {
    Dav1dSettings s;
    dav1d_default_settings(&s);
    s.n_threads = 4;

    Dav1dContext *c = NULL;
    if (dav1d_open(&c, &s) < 0) return -1;

    Dav1dData data = { 0 };
    uint8_t *dst = dav1d_data_create(&data, obu_len);
    memcpy(dst, obu_buf, obu_len);

    if (dav1d_send_data(c, &data) < 0) {
        dav1d_data_unref(&data);
        dav1d_close(&c);
        return -1;
    }

    Dav1dPicture pic = { 0 };
    while (dav1d_get_picture(c, &pic) == 0) {
        /* pic.data[0..3], pic.stride[0..2], pic.p.w, pic.p.h, pic.p.bpc */
        dav1d_picture_unref(&pic);
    }

    dav1d_close(&c);
    return 0;
}
```

### 4. `.avif` Görüntü Çözme (`av1oxide::avif`)

```rust
use av1oxide::Result;
use av1oxide::avif::{AvifDecodeConfig, decode_rgba8, decode_with, parse_info, probe};

fn avif_coz(avif_baytlari: &[u8]) -> Result<()> {
    if !probe(avif_baytlari) {
        return Ok(());
    }

    // Piksel çözmeden görüntü boyutlarını, renk tanımlayıcılarını, alfa ve metaveriyi incele.
    let info = parse_info(avif_baytlari)?;
    println!(
        "AVIF başlığı: {}x{}, {:?}, {:?}, alfa={}",
        info.width, info.height, info.bit_depth, info.chroma_format, info.has_alpha
    );

    // Doğrudan sıkı paketlenmiş 8-bit RGBA ([R, G, B, A, ...]) tamponuna çöz.
    let rgba = decode_rgba8(avif_baytlari)?;
    assert_eq!(rgba.pixels.len(), (rgba.width as usize) * (rgba.height as usize) * 4);

    // Veya açık iş parçacığı ve bellek bütçesiyle 8/10/12-bit düzlemsel örneklere çöz.
    let cfg = AvifDecodeConfig {
        n_threads: 4,
        max_pixels: 8192 * 8192,
        allocation_limit: Some(256 * 1024 * 1024),
        ..AvifDecodeConfig::default()
    };
    let planar = decode_with(avif_baytlari, &cfg)?;
    let _rgb16 = planar.to_rgb16()?;
    Ok(())
}
```

### 5. Çalıştırılabilir Örnekler (`examples/`)

Depoda doğrudan çalıştırılabilen beş kullanım örneği bulunur:

- **[`examples/decode_obu.rs`](crates/av1oxide/examples/decode_obu.rs):** Ham AV1 OBU paketlerini zaman damgasıyla (`Packet::with_timestamp`) çözme, kare metaverisini ve düzlem satırlarını okuma:
  ```bash
  cargo run --release -p av1oxide --example decode_obu
  cargo run --release -p av1oxide --example decode_obu -- <dosya.obu>
  ```
- **[`examples/decode_ivf.rs`](crates/av1oxide/examples/decode_ivf.rs):** Çok-thread (`n_threads: 4, max_frame_delay: 2`) çözücü ile `.ivf` akışını kare kare çözme ve isteğe bağlı düzlemsel YUV dosyasına yazma:
  ```bash
  cargo run --release -p av1oxide --example decode_ivf
  cargo run --release -p av1oxide --example decode_ivf -- <girdi.ivf> [cikti.yuv]
  ```
- **[`examples/bounded_decoder.rs`](crates/av1oxide/examples/bounded_decoder.rs):** `frame_size_limit` ve `allocation_limit` ile bellek bütçesi sınırlandırma, bozuk paket yakalama ve `decoder.reset()` ile aynı çözücü örneğini kurtarma:
  ```bash
  cargo run --release -p av1oxide --example bounded_decoder
  ```
- **[`examples/c_abi_decode.rs`](crates/av1oxide/examples/c_abi_decode.rs):** `libdav1d` (`7.0.0`) C-ABI üzerinden uçtan uca çözme (`dav1d_default_settings` → `dav1d_open` → `dav1d_data_create` → `dav1d_send_data` → `dav1d_get_picture` → `dav1d_picture_unref` → `dav1d_close`):
  ```bash
  cargo run --release -p av1oxide --example c_abi_decode
  ```
- **[`examples/decode_avif.rs`](crates/av1oxide/examples/decode_avif.rs):** `avif::parse_info` ile `.avif` başlıklarını inceleme ve `avif::decode` / `avif::decode_rgba8` ile düzlemsel ve 8-bit RGBA pikselleri çözme:
  ```bash
  cargo run --release -p av1oxide --example decode_avif
  cargo run --release -p av1oxide --example decode_avif -- <girdi.avif>
  ```

---

## `dav1d 1.5.4` ve `libavif 1.4.2` ile Süre ve Bellek (RAM) Karşılaştırması

Tüm ölçümler `%100` MD5 bit-birebir eşitlik altında `av1oxide` (`v0.4.0`, `rustc 1.100.0 --release`, `opt-level = 3`, `lto = "thin"`, `strip = "symbols"`), referans C/ASM `dav1d 1.5.4` (`99f3354ed0b9b4bf3c7c9347318dbcc522c94af5`) ve referans C `libavif 1.4.2` (`b8551fe61aaf92733234efc5e11cdfac5a5f54b3`, `-DAVIF_LIBYUV=OFF -DAVIF_LIBSHARPYUV=OFF` + `dav1d 1.5.4`) arasında gerçekleştirilmiştir. Dosya okuma, kapsayıcı ayrıştırma ve MD5 doğrulaması zamanlama penceresinin dışındadır; her iş yükü ve çözücü izole alt süreçlerde `5–7` dönüşümlü koşu ile yürütülüp ortanca (`p50`) duvar süresi, işletim sistemi Tepe RSS (`MiB`) ve `av1oxide` canlı yığın tepesi (`MiB`) raporlanmıştır.

### 1. `x86_64` — AMD Ryzen 9 9950X3D (Zen 5, `16C / 32T`, `AVX-512ICL`)

#### Tek-Thread (`n_threads = 1`, `max_frame_delay = 1`) Küme Özeti

| Küme | Akış / Kare | `av1oxide` Toplam (`ms`) | `dav1d` Toplam (`ms`) | Duvar Süresi Oranı (`av1oxide / dav1d`) | Geometrik Ortalama | Ortanca (`p50`) | Küme Maks. `av1oxide` Tepe RSS (`MiB`) | Küme Maks. `dav1d` Tepe RSS (`MiB`) | RSS Oranı (`av1oxide / dav1d`) | MD5 Eşitliği |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`korpus`** | `19` akış / `950` çözülen kare | `645,5 ms` | `430,0 ms` | **`1,501×`** | `1,552×` | `1,741×` | `19,01 MiB` (`10,90 MiB` canlı yığın) | `9,16 MiB` | `2,08×` | `19 / 19` (`%100`) |
| **`temsilci`** | `8` akış / `485` çözülen kare | `1 963,6 ms` | `1 063,1 ms` | **`1,847×`** | `1,593×` | `1,704×` | `1 231,36 MiB` (`1 212,62 MiB` canlı yığın) | `608,98 MiB` | `2,02×` | `8 / 8` (`%100`) |

#### Tek-Thread (`n_threads = 1`) Temsilci Akışlar Detayı

| Akış | Çözünürlük | Bit / Kroma | Kare | `av1oxide` (`ms/kare`) | `av1oxide` (`FPS`) | `dav1d` (`ms/kare`) | `dav1d` (`FPS`) | Süre Oranı (`av1oxide / dav1d`) | Skaler'e Göre Hızlanma | `av1oxide` Tepe RSS / Canlı Yığın (`MiB`) | `dav1d` Tepe RSS (`MiB`) | RSS Oranı (`av1oxide / dav1d`) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `422_oddheight_8b` | `64×63` | `8b` `4:2:2` | `80` | `0,293` | `3 415,8` | `0,253` | `3 951,8` | **`1,157×`** | `2,60×` | `6,92` / `0,37` | `3,66` | `1,89×` |
| `422_oddheight_10b` | `64×63` | `10b` `4:2:2` | `80` | `0,296` | `3 373,9` | `0,255` | `3 919,0` | **`1,162×`** | `2,79×` | `7,48` / `0,56` | `3,91` | `1,91×` |
| `vq002_inter_432x240` | `432×240` | `8b` `4:2:0` | `50` | `2,845` | `351,5` | `2,169` | `461,0` | **`1,312×`** | `4,08×` | `11,20` / `3,40` | `5,41` | `2,07×` |
| `test10755_cdef10_sb64` | `1024×512` | `10b` `4:2:0` | `24` | `2,674` | `374,0` | `1,649` | `606,5` | **`1,622×`** | `4,17×` | `23,41` / `13,29` | `11,39` | `2,05×` |
| `fg10_film_grain_50` | `352×288` | `10b` `4:2:0` | `100` | `0,835` | `1 197,8` | `0,467` | `2 139,8` | **`1,786×`** | `4,48×` | `14,66` / `6,79` | `6,40` | `2,29×` |
| `test25_8k_444_fg` | `7680×4320` | `8b` `4:4:4` | `6` | `246,217` | `4,1` | `125,826` | `8,0` | **`1,957×`** | `1,89×` | `1 231,36` / `1 212,62` | `608,98` | `2,02×` |
| `test1472_lr8_444` | `306×255` | `8b` `4:4:4` | `45` | `0,308` | `3 243,4` | `0,153` | `6 532,7` | **`2,014×`** | `4,12×` | `10,45` / `2,90` | `5,41` | `1,93×` |
| `ccvb_film_grain` | `640×480` | `8b` `4:2:0` | `100` | `1,354` | `738,5` | `0,659` | `1 517,9` | **`2,055×`** | `5,42×` | `19,04` / `10,90` | `9,16` | `2,08×` |

#### Çok-Thread (`AMD Ryzen 9 9950X3D`) AV1 Karşılaştırması

| Akış | Kip (`max_frame_delay`, `n_threads`) | `av1oxide` (`t=1`) | `av1oxide` (Çok-Thread) | `av1oxide` Öz-Hızlanma | `dav1d` (Çok-Thread) | Süre Oranı (`av1oxide / dav1d`) | `av1oxide` Tepe RSS / Canlı Yığın (`MiB`) | `dav1d` Tepe RSS (`MiB`) | RSS Oranı (`av1oxide / dav1d`) | MD5 Eşitliği |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vq002_inter_gop` (`432×240`, `8b`, `10×10=100` kare) | `mfd=1, t=8` | `272,9 ms` | `257,7 ms` | `1,06×` | `242,5 ms` | **`1,06×`** | `11,36` / `3,40` | `6,72` | `1,69×` | `%100` |
| `fg10_film_grain_50` (`352×288`, `10b`, `10×15=150` kare) | `mfd=1, t=8` | `119,4 ms` | `103,4 ms` | `1,15×` | `71,5 ms` | **`1,45×`** | `14,81` / `7,14` | `7,46` | `1,98×` | `%100` |
| `ccvb_film_grain` (`640×480`, `8b`, `100×3=300` kare) | `mfd=1, t=8` | `384,2 ms` | `298,4 ms` | `1,29×` | `182,2 ms` | **`1,64×`** | `19,46` / `11,39` | `9,97` | `1,95×` | `%100` |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tile, `8×15=120` kare) | `mfd=0, t=16` | `1 210,3 ms` | `258,6 ms` | **`4,68×`** | `141,1 ms` | **`1,83×`** | `196,18` / `94,59` | `315,22` | **`0,62×`** | `%100` |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tile, `8×15=120` kare) | `mfd=4, t=16` | `1 210,3 ms` | `241,6 ms` | **`5,01×`** | `145,3 ms` | **`1,66×`** | `219,47` / `95,27` | `306,50` | **`0,72×`** | `%100` |
| `test25_8k_444_fg` (`7680×4320`, `8b` `4:4:4`, `120` tile + FG, `6` kare) | `mfd=0, t=8` | `1 500,1 ms` | `429,5 ms` | **`3,49×`** | `186,9 ms` | **`2,30×`** (`t=4`: `2,58×`) | `1 990,94` / `1 416,23` | `1 416,41` | `1,41×` | `%100` |
| `test25_8k_120_tile` (`7680×4320`, `12b`, `120` tile, `10×2=20` kare) | `mfd=0, t=16` | `1 590,1 ms` | `570,0 ms` | **`2,79×`** | `165,5 ms` | **`3,44×`** | `1 321,86` / `811,50` | `1 020,45` | `1,30×` | `%100` |

#### `.avif` Görüntü Çözme + `RGBA8` Dönüşümü (`av1oxide::avif` vs C `libavif 1.4.2 + dav1d 1.5.4`)

| İş Yükü | Görüntü / Kip | `av1oxide` Süre (`ms`) | C `libavif` Süre (`ms`) | Süre Oranı (`av1oxide / libavif`) | `av1oxide` Tepe RSS (`MiB`) | `av1oxide` Canlı Yığın (`MiB`) | C `libavif` Tepe RSS (`MiB`) | RSS Oranı (`av1oxide / libavif`) | `RGBA8` MD5 Eşitliği |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`avif_corpus_156_t4`** | `156` geçerli `.avif` görüntüsü, `n_threads = 4` (`1` tur) | `3 617,2 ms` | `2 477,6 ms` | **`1,460×`** | `183,33 MiB` | `99,15 MiB` | `103,39 MiB` | **`1,77×`** | `156 / 156` (`%100`) |
| **`avif_bbb_4k_t8`** | `bbb_4k.avif` (`3840×2160`), `n_threads = 8` (`5` tekrar) | `407,2 ms` | `273,7 ms` | **`1,488×`** | `75,30 MiB` | `67,06 MiB` | `51,23 MiB` | **`1,47×`** | `5 / 5` (`%100`) |

---

### 2. `aarch64` — Apple M4 Pro (`14C = 10P + 4E`, `NEON + DOTPROD`)

#### Tek-Thread (`n_threads = 1`, `max_frame_delay = 1`) Küme Özeti

| Küme | SIMD Kipi | Akış / Kare | `av1oxide` Toplam (`ms`) | `dav1d` Toplam (`ms`) | Duvar Süresi Oranı (`av1oxide / dav1d`) | Geometrik Ortalama | Ortanca (`p50`) | Küme Maks. `av1oxide` Tepe RSS (`MiB`) | Küme Maks. `dav1d` Tepe RSS (`MiB`) | RSS Oranı (`av1oxide / dav1d`) | MD5 Eşitliği |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`korpus`** | `NEON + DOTPROD` (aynı-ISA) | `19` akış / `950` çözülen kare | `738,5 ms` | `474,1 ms` | **`1,558×`** | `1,623×` | `1,599×` | `16,52 MiB` (`10,89 MiB` canlı yığın) | `8,91 MiB` | `1,85×` | `19 / 19` (`%100`) |
| **`temsilci`** | `NEON + DOTPROD` (aynı-ISA) | `8` akış / `485` çözülen kare | `1 994,9 ms` | `1 123,8 ms` | **`1,775×`** | `1,560×` | `1,632×` | `1 234,05 MiB` (`1 212,62 MiB` canlı yığın) | `610,30 MiB` | `2,02×` | `8 / 8` (`%100`) |
| **`korpus`** | `tespit` (`dav1d` + `I8MM`) | `19` akış / `950` çözülen kare | `734,2 ms` | `474,5 ms` | **`1,547×`** | `1,647×` | `1,585×` | `16,50 MiB` (`10,89 MiB` canlı yığın) | `8,89 MiB` | `1,86×` | `19 / 19` (`%100`) |
| **`temsilci`** | `tespit` (`dav1d` + `I8MM`) | `8` akış / `485` çözülen kare | `1 982,5 ms` | `1 122,2 ms` | **`1,767×`** | `1,570×` | `1,622×` | `1 233,30 MiB` (`1 212,62 MiB` canlı yığın) | `610,31 MiB` | `2,02×` | `8 / 8` (`%100`) |

#### Tek-Thread (`n_threads = 1`) Temsilci Akışlar Detayı (`NEON + DOTPROD`)

| Akış | Çözünürlük | Bit / Kroma | Kare | `av1oxide` (`ms/kare`) | `av1oxide` (`FPS`) | `dav1d` (`ms/kare`) | `dav1d` (`FPS`) | Süre Oranı (`av1oxide / dav1d`) | Skaler'e Göre Hızlanma | `av1oxide` Tepe RSS / Canlı Yığın (`MiB`) | `dav1d` Tepe RSS (`MiB`) | RSS Oranı (`av1oxide / dav1d`) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `422_oddheight_8b` | `64×63` | `8b` `4:2:2` | `80` | `0,292` | `3 423,5` | `0,274` | `3 645,0` | **`1,065×`** | `1,52×` | `4,20` / `0,36` | `2,91` | `1,45×` |
| `422_oddheight_10b` | `64×63` | `10b` `4:2:2` | `80` | `0,310` | `3 225,8` | `0,283` | `3 538,6` | **`1,097×`** | `1,74×` | `4,58` / `0,55` | `3,00` | `1,53×` |
| `vq002_inter_432x240` | `432×240` | `8b` `4:2:0` | `50` | `3,263` | `306,5` | `2,526` | `395,8` | **`1,291×`** | `2,34×` | `8,64` / `3,39` | `5,08` | `1,70×` |
| `test10755_cdef10_sb64` | `1024×512` | `10b` `4:2:0` | `24` | `2,760` | `362,4` | `1,947` | `513,5` | **`1,417×`** | `2,64×` | `20,70` / `13,28` | `11,56` | `1,79×` |
| `fg10_film_grain_50` | `352×288` | `10b` `4:2:0` | `100` | `0,980` | `1 020,7` | `0,509` | `1 965,6` | **`1,926×`** | `2,98×` | `12,00` / `6,78` | `6,02` | `1,99×` |
| `test25_8k_444_fg` | `7680×4320` | `8b` `4:4:4` | `6` | `234,857` | `4,3` | `127,112` | `7,9` | **`1,848×`** | `1,68×` | `1 234,05` / `1 212,62` | `610,30` | `2,02×` |
| `test1472_lr8_444` | `306×255` | `8b` `4:4:4` | `45` | `0,366` | `2 735,2` | `0,182` | `5 501,9` | **`2,011×`** | `2,32×` | `8,34` / `2,90` | `4,95` | `1,68×` |
| `ccvb_film_grain` | `640×480` | `8b` `4:2:0` | `100` | `1,938` | `516,0` | `0,845` | `1 183,6` | **`2,294×`** | `2,98×` | `16,44` / `10,89` | `8,91` | `1,85×` |

#### Çok-Thread (`Apple M4 Pro`) AV1 Karşılaştırması

| Akış | Kip (`max_frame_delay`, `n_threads`) | `av1oxide` (`t=1`) | `av1oxide` (Çok-Thread) | `av1oxide` Öz-Hızlanma | `dav1d` (Çok-Thread) | Süre Oranı (`av1oxide / dav1d`) | `av1oxide` Tepe RSS / Canlı Yığın (`MiB`) | `dav1d` Tepe RSS (`MiB`) | RSS Oranı (`av1oxide / dav1d`) | MD5 Eşitliği |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vq002_inter_gop` (`432×240`, `8b`, `10×10=100` kare) | `mfd=1, t=4` | `311,1 ms` | `277,9 ms` | `1,12×` | `231,1 ms` | **`1,20×`** | `9,50` / `3,39` | `5,97` | `1,59×` | `%100` |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tile, `8×15=120` kare) | `mfd=0, t=14` | `1 550,6 ms` | `238,2 ms` | **`6,51×`** | `137,5 ms` | **`1,73×`** | `140,00` / `94,42` | `102,77` | `1,36×` | `%100` |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tile, `8×15=120` kare) | `mfd=4, t=14` | `1 550,6 ms` | `239,4 ms` | **`6,48×`** | `136,9 ms` | **`1,75×`** | `139,89` / `94,69` | `103,03` | `1,36×` | `%100` |
| `test25_8k_444_fg` (`7680×4320`, `8b` `4:4:4`, `120` tile + FG, `6` kare) | `mfd=0, t=14` | `1 418,8 ms` | `147,7 ms` | **`9,60×`** | `66,2 ms` | **`2,23×`** | `2 055,83` / `1 705,67` | `1 398,83` | `1,47×` | `%100` |
| `test25_8k_120_tile` (`7680×4320`, `12b`, `120` tile, `10×2=20` kare) | `mfd=0, t=14` | `1 694,6 ms` | `238,7 ms` | **`7,10×`** | `97,3 ms` | **`2,45×`** | `1 410,64` / `1 131,39` | `1 097,81` | `1,28×` | `%100` |

#### `.avif` Görüntü Çözme + `RGBA8` Dönüşümü (`av1oxide::avif` vs C `libavif 1.4.2 + dav1d 1.5.4`)

| İş Yükü | Görüntü / Kip | `av1oxide` Süre (`ms`) | C `libavif` Süre (`ms`) | Süre Oranı (`av1oxide / libavif`) | `av1oxide` Tepe RSS (`MiB`) | `av1oxide` Canlı Yığın (`MiB`) | C `libavif` Tepe RSS (`MiB`) | RSS Oranı (`av1oxide / libavif`) | `RGBA8` MD5 Eşitliği |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`avif_corpus_156_t4`** | `156` geçerli `.avif` görüntüsü, `n_threads = 4` (`1` tur) | `3 348,8 ms` | `2 600,8 ms` | **`1,288×`** | `335,95 MiB` | `106,74 MiB` | `205,16 MiB` | **`1,64×`** | `156 / 156` (`%100`) |
| **`avif_bbb_4k_t8`** | `bbb_4k.avif` (`3840×2160`), `n_threads = 8` (`5` tekrar) | `366,1 ms` | `222,8 ms` | **`1,644×`** | `104,80 MiB` | `67,06 MiB` | `51,31 MiB` | **`2,04×`** | `5 / 5` (`%100`) |

---

### 3. Süre ve Bellek Farkının Teknik Analizi

#### Çalışma Zamanı Hızını Etkilemeyen Kontroller (`~%0` Etki)

- Bellek bütçesi ve kare boyutu kontrolleri (`Config::allocation_limit`, `Config::frame_size_limit`) piksel veya blok döngülerinin içinde değil, yalnızca kare ve tampon tahsisi sırasında çalıştırılır; kare çözme hızı üzerinde ölçülebilir bir etkisi yoktur.

#### `dav1d` ile Süre Farkının Ölçülmüş Teknik Nedenleri (`~1,50×–1,70×`)

- **Dizi Sınır Kontrolleri (Çözme süresinin `%23–%50`'si):** Yüksek çözünürlüklü karelerde milyonlarca `4×4` / `8×8` blok ve komşuluk tablosu gezilir. C `dav1d` bu yapılara sınır kontrolsüz ham işaretçi aritmetiğiyle (`ptr += offset`) erişirken, `av1oxide` içinde DSP dışındaki tüm çözücü çekirdeği `#![forbid(unsafe_code)]` altında derlenir ve her dilim erişiminde (`slice[i]`) çalışma zamanı sınır kontrolü yürütülür (iki tarafta da SIMD kapatıldığında saf C ile saf Rust arasında ölçülen `~1,65×–1,71×` oranının temel kaynağıdır).
- **SIMD Giriş Doğrulaması ve Sıfır Taşan Okuma (`~2–5 ns` / çağrı):** C `dav1d` Assembly rutinleri girişte uzunluk kontrolü yapmaz ve tampon sonundan dışarı 16–64 bayt dolgu bölgesine taşan okuma (over-read) gerçekleştirir. `av1oxide` içindeki her SIMD fonksiyonu girişte dilim boyutu önkoşullarını doğrular ve dilim sınırlarının dışına okuma yapmamak için kenarlarda maskeli/sınırlı yükleme kullanır; `10–20 ns` süren küçük (`4×4`) blok çağrılarında bu doğrulama `%15–%25` ek süre oluşturur.
- **El Yazısı Assembly ve Derleyici (`LLVM`) Kod Üretimi:** `dav1d` yaklaşık `265 bin` satır el yazısı NASM/GAS Assembly kullanarak standart fonksiyon çağrı kurallarını (ABI) atlayıp CPU bayrak yazmaçlarını ve 32 adet `zmm` yazmacını doğrudan yönetir; `av1oxide` ise kararlı Rust `core::arch` intrinsiklerini standart LLVM ABI çerçevesi üzerinden derler.

#### Bellek (`Peak RSS`) Farkının Teknik Nedenleri

- **Tek-Thread Taban İzi ve Yüksek Çözünürlüklü Düzlem Tamponları (`~1,85×–2,08×` RSS):** Küçük ve orta çözünürlüklü akışlarda (`16×16` – `640×480`) `av1oxide` canlı yığın kullanımı `0,21–10,90 MiB` iken işletim sistemi süreç yerleşik küme boyutu (`x86_64` Linux üzerinde `6,9–19,0 MiB`, `aarch64` macOS üzerinde `4,2–16,5 MiB`; C `dav1d`: `2,9–9,2 MiB`), paylaşımlı kütüphane bağlaması yerine statik bağlanan Rust çalışma zamanı ile çoklu-ISA SIMD kod sayfalarını (`x86_64` ELF: `9,7 MiB`, `aarch64` Mach-O: `2,8 MiB`) içerir. Film Grain içeren `8K` `4:4:4` akışında (`test25_8k_444_fg`: `1 231–1 234 MiB` vs `609–610 MiB`, `2,02×`) `av1oxide`, ham işaretçi örtüşmesi kullanmamak amacıyla referans kare düzlemlerini, CDEF/LR şerit ara tamponlarını ve referans yuvalarını değiştirmeden üretilen Film Grain çıktı kopyasını ayrık sahiplikli dilimler olarak tutar.
- **8K Çok-Thread Kare Gecikmeli Ölçeklenme (`~1,28×–1,47×` RSS):** `8K` `120` karolu çok-thread çözmede (`test25_8k_120_tile` `1,28×–1,30×` RSS; `test25_8k_444_fg` `1,41×–1,47×` RSS) her iki çözücü de eşzamanlı `FrameSlot` bağlamlarını (`n_fc = 3..4`) ölçekler; referans karelerin iş parçacıkları arasında `Arc` ile paylaşılması ve `DecodeBufPool` üzerinden geri dönüştürülmesi nedeniyle göreli RSS oranı tek-thread durumuna kıyasla daralır.
- **16-Thread Çok-Karolu Akışlarda Paylaşımlı Geçici Havuz (`x86_64` Üzerinde `0,62×–0,72×` RSS):** `1024×1024` `16` karolu `10-bit` akışta `n_threads = 16` kipinde (`test10103_coklu_tile`), `av1oxide` Linux `x86_64` üzerinde `196,2–219,5 MiB` Tepe RSS (`94,6–95,3 MiB` canlı yığın) kullanırken C `dav1d` `306,5–315,2 MiB` (`0,62×–0,72×`) kullanır; bunun nedeni `av1oxide` tarafında her kare bağlamı için tam karo/katsayı arenalarını peşin tahsis etmek yerine sınırlı `ScratchPool` işçi yuvalarının ve `DecodeBufPool` düzlemlerinin talep üzerine paylaşılmasıdır. macOS `aarch64` (`t=14`) üzerinde `16 KiB` sanal bellek sayfası boyutu ve sistem `libmalloc` iş parçacığı önbelleklemesi ile bu değer `140,0 MiB` vs `102,8 MiB` (`1,36×`, `94,4 MiB` canlı yığın) olarak ölçülür.
- **`.avif` Kapsayıcı ve `RGBA8` Dönüşümü (`1,47×–2,04×` RSS):** `.avif` çözme sırasında (`4K` `bbb_4k.avif` üzerinde `67,1 MiB` canlı yığın, `156` görüntülük korpusta `99,2–106,7 MiB`) `av1oxide::avif`, hedef `RGBA8` tamponunu (`3840×2160` için `31,64 MiB`) ayırmadan önce düzlemsel YUVA kanallarını `u16` dilimleri olarak saklar; C `libavif` ise `8-bit` kaynak kanallarını `avifImageYUVToRGB` öncesinde `uint8_t` düzlemleri olarak tutar.

---

## Lisans

[BSD-2-Clause](LICENSE)
