# av1oxide (`v0.5.0`)

[English](README.md) | **Türkçe**

`av1oxide`, VideoLAN [`dav1d`](https://code.videolan.org/videolan/dav1d) (`1.5.4`) AV1 video çözücüsünün **%100 kararlı Rust** (`rustc 1.100.0`, Edition 2024), **sıfır üçüncü taraf medya veya kodek bağımlılığı**, isteğe bağlı **`#![no_std] + alloc`** desteği, **`libdav1d` (`1.5.4` / API `7.0.0`) drop-in ikili uyumlu C-ABI katmanı** ve **saf Rust `.avif` / `.avis` görüntü ve dizi çözücü katmanı** ile yeniden gerçekleştirimidir.

- **Profiller ve Bit Derinlikleri:** Main (`0`), High (`1`), Professional (`2`); `8-bit`, `10-bit`, `12-bit`; `4:2:0`, `4:2:2`, `4:4:4` ve `Monochrome` (`4:0:0`).
- **AV1 Hattı:** OBU ayrıştırma, Sequence/Frame Header, MSAC aritmetik entropi çözücü, ters dönüşümler (`ITX`), intra tahmin (`IPRED`, CfL, Palette), hareket telafisi (`MC`, Warp, Superres/Resize, IntraBC, `REFMVS`), halka içi filtreler (`LF`, `CDEF`, `4` satırlık `LrCdefBackup` ile `LR` Wiener/SGR) ve şerit bazlı (`32` satır) Film Grain (`FG`) sentezi.
- **Bit-Birebir Doğruluk:** `dav1d-test-data` (`19/19` akış, `964` kare) ve Argon Streams `v2.1.1` (`2 762/2 762` akış, `36 700` kare) olmak üzere toplam **`2 781 / 2 781` akış (`37 664 / 37 664` kare)** üzerinde tüm CPU yetenek düzeylerinde ve çok-thread kiplerinde `dav1d 1.5.4` ile `%100` MD5 bit-birebir eşitlik.
- **Donanım Yetenek Jetonlu SIMD (`767` DSP Yuvası):**
  - `x86_64` (`SSE2`, `SSSE3`, `SSE4.1`, `AVX2`, `AVX-512ICL`): `767 / 767` (`%100`) SIMD yuvası
  - `aarch64` (`NEON`, `DOTPROD`): `767 / 767` (`%100`) SIMD yuvası
  - `wasm32` (`+simd128`): `%100 #![forbid(unsafe_code)]` ile `682 / 767` (`%88,9`) SIMD yuvası (Node.js ve Deno üzerinde skaler WASM'a göre `1,88×–2,40×` hızlanma; skaler derlemede `0` `v128` komutu)
  - İkincil mimariler (`arm32`, `riscv64`, `loongarch64`, `powerpc64le`): `%100` güvenli skaler geri-düşüş
- **Çok-Thread ve Bellek Bütçesi:** Kare içi (`Tile`, süperblok satırı, halka içi filtre) ve kareler arası (`FrameSlot`) paralellik (`Config::n_threads`, `Config::max_frame_delay`), düzlem geri dönüşüm havuzu (`DecodeBufPool`) ve tahsis öncesi piksel/bellek tavanı (`Config::frame_size_limit`, `Config::allocation_limit`).
- **İsteğe Bağlı `#![no_std] + alloc` (`default = ["std"]`):** `--no-default-features` ile tam tek-thread AV1 + AVIF çekirdeği sıfır `std` bağımlılığıyla `core + alloc` üzerinde derlenir (`wasm32-unknown-unknown`, `aarch64`, `armv7` ve `riscv64gc` hedeflerinde doğrulandı).
- **Kapsayıcı Yardımcıları (`container`):** `%100 #![forbid(unsafe_code)]` sıfır kopyalı IVF (`container::ivf::{IvfParser, IvfReader}`) ve AV1 Annex-B (`container::annexb::{AnnexBDemuxer, read_leb128}`) ayrıştırıcıları.
- **`libdav1d` (`1.5.4` / API `7.0.0`) Drop-In C-ABI (`capi`):** 19 resmî `DAV1D_API` fonksiyonunun tamamını (`dav1d_open`, `dav1d_send_data`, `dav1d_get_picture`, `dav1d_apply_grain`, `dav1d_parse_sequence_header`, vb.), resmî `include/dav1d/*.h` C başlıklarını ve `[package.metadata.capi]` / `dav1d.pc` sistem paketleme üst verisini içerir — değiştirilmemiş C `libavif 1.4.2` ikilisinin dinamik olarak `libdav1d.so.7` kütüphanesine bağlanması (`162 / 162` korpus paritesi) ve bağımsız çok-thread C IVF istemcisi ile doğrulanmıştır.
- **Saf Rust `.avif` / `.avis` Görüntü Okuyucu (`avif`):** `%100 #![forbid(unsafe_code)]` ISOBMFF/HEIF/MIAF/AVIF ayrıştırıcısı; doğal `u8` 8-bit düzlem temsili (`AvifPlane::U8`/`U16`) ve şerit-paralel YUVA→RGB dönüşümü, tekil `av01`, çok-thread `grid` karo birleştirme, `iden` türetilmiş görüntüler, yardımcı alfa (`auxl`/`auxC`), `clap`/`irot`/`imir` dönüşümleri, ISO 21496-1 / Ultra HDR `tmap` Gain-Map (`9 / 9` parite), akışlı animasyonlu `.avis` çözme (`AvifSequenceDecoder`, `6 / 6` animasyonlu akış + `nth_frame` rastgele erişim), `ICC`/`Exif`/`XMP` metaverisi (`std` altında bütçe korumalı `defl`/`zlib`/`gzip` sıkıştırılmış HEIF metaveri öğeleri ve `cmin` kutuları dahil) ve doğrusal/doğrusal olmayan ITU-T H.273 dönüşümleri (`ICtCp` `mc = 14` PQ/HLG ve `BT.2020 CL` `mc = 10` dahil; `RGB8`/`RGBA8`/`RGB16`/`RGBA16`) — lisans denetimli `AOMediaCodec/av1-avif` ve `AOMediaCodec/libavif` test korpusundaki `162 / 162` durağan (`156` geçerli + `6` negatif, `9 / 9` Gain-Map) ve `6 / 6` animasyonlu `.avif` dosyasında referans C `libavif 1.4.2` ile `%100` MD5 bit-birebir eşitlik.
- **Ayrı `image` Ekosistem Köprüsü (`av1oxide-image`):** Çalışma alanı içindeki bağımsız [`crates/av1oxide-image`](crates/av1oxide-image/README.md) (`v0.1.0`, `%100 #![forbid(unsafe_code)]`) paketi, ana `av1oxide` bağımlılık ağacına `image` eklemeden `image::ImageDecoder` (`Rgb8`/`Rgba8`/`Rgb16`/`Rgba16`, `icc_profile`, `exif_metadata`, `xmp_metadata`, `Limits`) ve `register_decoding_hook` uyumluluğu sunar.

---

## Bağımlılık Politikası

`av1oxide`, hiçbir üçüncü taraf medya, kodek, görüntü işleme veya C medya kütüphanesine bağımlı değildir; tüm AV1 bit akışı/entropi/DSP çözme, ISOBMFF/AVIF/IVF/Annex-B kapsayıcı ayrıştırma, ITU-T H.273 renk dönüşümü ve `libdav1d` C-ABI katmanları depo içinde gerçekleştirilmiştir. Kullanılan tüm çekirdek kütüphaneler iç uygulama ayrıntısı (`pub(crate)` / özel) olarak yalıtılmıştır; `av1oxide` genel (`pub`) API imzalarına hiçbir bağımlılık tipi yeniden dışa aktarılmaz veya sızdırılmaz.

Düşük seviyeli eşzamanlılık, bellek yerleşimi ve sıkıştırma açma ilkel tipleri için yalnızca `rust-lang` organizasyonu ve Rust çekirdek ekip üyeleri tarafından geliştirilen aşağıdaki temel kütüphaneler `default-features = false` ile kullanılır:

| Crate | Sürüm | Lisans | Kaynak | Özellikler (`default-features = false`) | Görev | Unsafe / Yerel Kod |
|---|---|---|---|---|---|---|
| `crossbeam-utils` | `0.8` (`0.8.23`) | `MIT OR Apache-2.0` | [crossbeam-rs/crossbeam](https://github.com/crossbeam-rs/crossbeam) | `[]` (`std` özelliği `av1oxide/std` ile açılır) | Paylaşılan atomik sayaç/işçi yuvalarının (`thread::TaskBatch`, `thread::ScratchPool`, `frame::butce::Butce`) `CachePadded` önbellek satırı izolasyonu, `no_std` `AtomicCell` ve kilitsiz CAS döngülerinde `Backoff` | Pure Rust (iç atomik/hizalama `unsafe`, `0` C kodu) |
| `parking_lot` | `0.12` (`0.12.5`) | `MIT OR Apache-2.0` | [Amanieu/parking_lot](https://github.com/Amanieu/parking_lot) | `[]` (isteğe bağlı: `std` & `not(target_arch = "wasm32")`) | `thread::WorkerPool`, `thread::FrameSlot`, `api::Decoder` ve `decode::intra` içinde zehirlenmesiz (`non-poisoning`) kompakt kullanıcı uzayı `Mutex` ve `Condvar` (`sync`) | Pure Rust + Wasm dışı hedeflerde işletim sistemi futex / `libc` sistem çağrısı (`0` derlenen C dosyası) |
| `memchr` | `2` (`2.8.3`) | `Unlicense OR MIT` | [BurntSushi/memchr](https://github.com/BurntSushi/memchr) | `[]` | ISOBMFF NUL-sonlu dize ayrıştırma (`avif::boxes`) ve OBU metaveri kuyruk baytı taramasında (`api::Decoder`) SIMD/SWAR hızlandırmalı bayt arama (`memchr`, `memrchr`) | Pure Rust (iç SIMD/işaretçi `unsafe`, `0` C kodu) |
| `rustc-hash` | `2` (`2.1.3`) | `Apache-2.0 OR MIT` | [rust-lang/rustc-hash](https://github.com/rust-lang/rustc-hash) | `[]` | ISOBMFF `item_id` tabloları (`avif`) ve DSP tablo kurulumunda (`dsp`) hızlı deterministik tamsayı karması (`FxBuildHasher`) | Safe Rust (`0` `unsafe` blok, `0` C kodu) |
| `indexmap` | `2` (`2.14.2`) | `Apache-2.0 OR MIT` | [indexmap-rs/indexmap](https://github.com/indexmap-rs/indexmap) | `[]` | ISOBMFF `iinf`, `ipma`, `iloc` ve `iref` öğe tablolarında (`avif::boxes`, `avif::decode`, `dsp`) ekleme sırasını koruyan $O(1)$ tekil tablo ve küme motoru (`IndexMap`, `IndexSet`) | Pure Rust (`indexmap` `#![forbid(unsafe_code)]` + Miri onaylı `hashbrown`, `0` C kodu) |
| `flate2` | `1.1` (`1.1.11`) | `MIT OR Apache-2.0` | [rust-lang/flate2-rs](https://github.com/rust-lang/flate2-rs) | `["rust_backend"]` (isteğe bağlı: `std`) | Sıkıştırılmış HEIF metaveri öğeleri (`infe` `content_encoding`, `cmpC`) ve `cmin` kutuları (`avif::boxes`) için sınırlı `deflate`/`zlib`/`gzip` açıcısı | Safe Rust (`miniz_oxide` `#![forbid(unsafe_code)]` arka ucu + `crc32fast`, `0` C/`libz-sys` kodu) |

---

## Kurulum

`Cargo.toml` dosyanıza ekleyin:

```toml
[dependencies]
av1oxide = "0.5"
```

`#![no_std] + alloc` ortamları için (`std`, `parking_lot` ve `flate2` olmadan tek-thread AV1 + AVIF):

```toml
[dependencies]
av1oxide = { version = "0.5", default-features = false }
```

`av1oxide` bağımlılık ağacını değiştirmeden `image` (`0.25`) ekosistemiyle bütünleşmek için ayrı `av1oxide-image` paketini ([`crates/av1oxide-image/README.md`](crates/av1oxide-image/README.md)) kullanabilirsiniz.

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

19 `dav1d_*` C sembolünü dışa aktaran paylaşımlı (`libav1oxide.so` / `.dylib` / `.dll`) ve statik (`libav1oxide.a`) kütüphaneleri üretmek için (`[package.metadata.capi]` üzerinden `dav1d.pc` ile drop-in kurulum desteklenir):

```bash
cargo rustc --release -p av1oxide --features capi --crate-type cdylib,staticlib
```

Herhangi bir C/C++ programını `crates/av1oxide/include` (`#include <dav1d/dav1d.h>`) ve `libav1oxide.a` / `libav1oxide.so` (veya `pkg-config --cflags --libs dav1d`) ile doğrudan bağlayabilirsiniz:

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

### 4. Durağan `.avif`, Gain-Map ve Animasyonlu `.avis` Çözme (`av1oxide::avif`)

```rust
use av1oxide::Result;
use av1oxide::avif::{
    AvifDecodeConfig, AvifSequenceDecoder, decode_rgba8, decode_with, parse_info, probe,
};

fn avif_coz(avif_baytlari: &[u8]) -> Result<()> {
    if !probe(avif_baytlari) {
        return Ok(());
    }

    // Piksel çözmeden görüntü boyutlarını, renk tanımlayıcılarını, alfa, kazanç haritası ve metaveriyi incele.
    let info = parse_info(avif_baytlari)?;
    println!(
        "AVIF başlığı: {}x{}, {:?}, {:?}, alfa={}, gain_map={}",
        info.width, info.height, info.bit_depth, info.chroma_format, info.has_alpha, info.has_gain_map
    );

    // Doğrudan sıkı paketlenmiş 8-bit RGBA ([R, G, B, A, ...]) tamponuna çöz.
    let rgba = decode_rgba8(avif_baytlari)?;
    assert_eq!(rgba.pixels.len(), (rgba.width as usize) * (rgba.height as usize) * 4);

    // Açık iş parçacığı, bellek bütçesi ve isteğe bağlı ISO 21496-1 kazanç haritası ile çöz.
    let cfg = AvifDecodeConfig {
        n_threads: 4,
        max_pixels: 8192 * 8192,
        allocation_limit: Some(256 * 1024 * 1024),
        decode_gain_map: true,
        ..AvifDecodeConfig::default()
    };
    let planar = decode_with(avif_baytlari, &cfg)?;
    let _rgb16 = planar.to_rgb16()?;
    let _gain_map = planar.gain_map();

    // Animasyonlu .avis karelerini bellekte biriktirmeden akışlı olarak çöz.
    let mut seq = AvifSequenceDecoder::with_config(avif_baytlari, cfg)?;
    while let Some(frame) = seq.next_frame()? {
        let _frame_rgba = frame.image.to_rgba8()?;
    }
    Ok(())
}
```

### 5. IVF ve Annex-B Ayrıştırma (`av1oxide::container`)

```rust
use av1oxide::container::annexb::AnnexBDemuxer;
use av1oxide::container::ivf::IvfParser;
use av1oxide::{Config, Decoder, Packet, Result};

fn ivf_coz(ivf_baytlari: &[u8]) -> Result<()> {
    let mut parser = IvfParser::new(ivf_baytlari)?;
    let mut decoder = Decoder::new(Config::default())?;
    for pkt in &mut parser {
        let pkt = pkt?;
        decoder.send_packet(&Packet::with_timestamp(pkt.data, pkt.timestamp as i64))?;
        while let Some(_frame) = decoder.receive_frame()? {}
    }
    decoder.flush();
    while let Some(_frame) = decoder.receive_frame()? {}
    Ok(())
}

fn annexb_ayristir(annexb_baytlari: &[u8]) -> Result<Vec<Vec<u8>>> {
    AnnexBDemuxer::new(annexb_baytlari).demux_all()
}
```

### 6. Çalıştırılabilir Örnekler (`examples/`)

Depoda doğrudan çalıştırılabilen beş kullanım örneği bulunur:

- **[`examples/decode_obu.rs`](crates/av1oxide/examples/decode_obu.rs):** Ham AV1 OBU paketlerini zaman damgasıyla (`Packet::with_timestamp`) çözme, kare metaverisini ve düzlem satırlarını okuma:
  ```bash
  cargo run --release -p av1oxide --example decode_obu
  cargo run --release -p av1oxide --example decode_obu -- <dosya.obu>
  ```
- **[`examples/decode_ivf.rs`](crates/av1oxide/examples/decode_ivf.rs):** `container::ivf::IvfParser` ve çok-thread (`n_threads: 4, max_frame_delay: 2`) çözücü ile `.ivf` akışını kare kare çözme ve isteğe bağlı düzlemsel YUV dosyasına yazma:
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

Tüm ölçümler `%100` MD5 bit-birebir eşitlik altında `av1oxide` (`rustc 1.100.0 --release`, `opt-level = 3`, `lto = "thin"`, `strip = "symbols"`), referans C/ASM `dav1d 1.5.4` (`99f3354ed0b9b4bf3c7c9347318dbcc522c94af5`) ve referans C `libavif 1.4.2` (`b8551fe61aaf92733234efc5e11cdfac5a5f54b3`, `-DAVIF_LIBYUV=OFF -DAVIF_LIBSHARPYUV=OFF` + `dav1d 1.5.4`) arasında gerçekleştirilmiştir. Dosya okuma, kapsayıcı ayrıştırma ve MD5 doğrulaması zamanlama penceresinin dışındadır; her iş yükü ve çözücü izole alt süreçlerde `5–7` dönüşümlü koşu ile yürütülüp ortanca (`p50`) duvar süresi, işletim sistemi Tepe RSS (`MiB`) ve `av1oxide` canlı yığın tepesi (`MiB`) raporlanmıştır. Aşağıdaki `x86_64` tabloları **`v0.5.0`** ölçümlerini, Apple M4 Pro (`aarch64`) tabloları ise donanım üzerindeki `v0.5.0` eşitlemesi öncesindeki **`v0.4.0` taban çizgisini** yansıtır.

### 1. `x86_64` — AMD Ryzen 9 9950X3D (Zen 5, `16C / 32T`, `AVX-512ICL`, `v0.5.0`)

#### Tek-Thread (`n_threads = 1`, `max_frame_delay = 1`) Küme Özeti

| Küme | Akış / Kare | `av1oxide` Toplam (`ms`) | `dav1d` Toplam (`ms`) | Duvar Süresi Oranı (`av1oxide / dav1d`) | Geometrik Ortalama | Ortanca (`p50`) | Küme Maks. `av1oxide` Tepe RSS (`MiB`) | Küme Maks. `dav1d` Tepe RSS (`MiB`) | RSS Oranı (`av1oxide / dav1d`) | MD5 Eşitliği |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`korpus`** | `19` akış / `950` çözülen kare | `641,8 ms` | `431,1 ms` | **`1,489×`** | `1,539×` | `1,717×` | `15,04 MiB` (`6,42 MiB` canlı yığın) | `9,16 MiB` | `1,64×` | `19 / 19` (`%100`) |
| **`temsilci`** | `8` akış / `485` çözülen kare | `1 900,3 ms` | `1 075,9 ms` | **`1,766×`** | `1,565×` | `1,676×` | `667,45 MiB` (`644,76 MiB` canlı yığın) | `608,98 MiB` | `1,10×` | `8 / 8` (`%100`) |

#### Tek-Thread (`n_threads = 1`) Temsilci Akışlar Detayı

| Akış | Çözünürlük | Bit / Kroma | Kare | `av1oxide` (`ms/kare`) | `av1oxide` (`FPS`) | `dav1d` (`ms/kare`) | `dav1d` (`FPS`) | Süre Oranı (`av1oxide / dav1d`) | Skaler'e Göre Hızlanma | `av1oxide` Tepe RSS / Canlı Yığın (`MiB`) | `dav1d` Tepe RSS (`MiB`) | RSS Oranı (`av1oxide / dav1d`) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `422_oddheight_8b` | `64×63` | `8b` `4:2:2` | `80` | `0,292` | `3 419,7` | `0,254` | `3 943,7` | **`1,153×`** | `2,62×` | `7,52` / `0,29` | `3,66` | `2,05×` |
| `422_oddheight_10b` | `64×63` | `10b` `4:2:2` | `80` | `0,295` | `3 394,2` | `0,256` | `3 907,1` | **`1,151×`** | `2,84×` | `7,77` / `0,41` | `3,91` | `1,99×` |
| `vq002_inter_432x240` | `432×240` | `8b` `4:2:0` | `50` | `2,843` | `351,7` | `2,172` | `460,4` | **`1,309×`** | `4,07×` | `10,53` / `2,22` | `5,41` | `1,95×` |
| `test10755_cdef10_sb64` | `1024×512` | `10b` `4:2:0` | `24` | `2,634` | `379,6` | `1,649` | `606,4` | **`1,597×`** | `4,23×` | `17,02` / `8,25` | `11,39` | `1,49×` |
| `fg10_film_grain_50` | `352×288` | `10b` `4:2:0` | `100` | `0,821` | `1 217,4` | `0,468` | `2 135,2` | **`1,754×`** | `4,56×` | `12,20` / `4,22` | `6,41` | `1,90×` |
| `test25_8k_444_fg` | `7680×4320` | `8b` `4:4:4` | `6` | `236,363` | `4,2` | `127,867` | `7,8` | **`1,849×`** | `1,93×` | `667,45` / `644,76` | `608,98` | `1,10×` |
| `test1472_lr8_444` | `306×255` | `8b` `4:4:4` | `45` | `0,301` | `3 317,3` | `0,153` | `6 534,6` | **`1,970×`** | `4,19×` | `9,85` / `1,78` | `5,39` | `1,83×` |
| `ccvb_film_grain` | `640×480` | `8b` `4:2:0` | `100` | `1,341` | `745,6` | `0,660` | `1 514,7` | **`2,031×`** | `5,48×` | `14,84` / `6,42` | `9,16` | `1,62×` |

#### Çok-Thread (`AMD Ryzen 9 9950X3D`) AV1 Karşılaştırması

| Akış | Kip (`max_frame_delay`, `n_threads`) | `av1oxide` (`t=1`) | `av1oxide` (Çok-Thread) | `av1oxide` Öz-Hızlanma | `dav1d` (Çok-Thread) | Süre Oranı (`av1oxide / dav1d`) | `av1oxide` Tepe RSS / Canlı Yığın (`MiB`) | `dav1d` Tepe RSS (`MiB`) | RSS Oranı (`av1oxide / dav1d`) | MD5 Eşitliği |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vq002_inter_gop` (`432×240`, `8b`, `10×10=100` kare) | `mfd=1, t=8` | `272,1 ms` | `256,7 ms` | `1,06×` | `248,8 ms` | **`1,03×`** | `10,26` / `2,31` | `6,70` | `1,53×` | `%100` |
| `fg10_film_grain_50` (`352×288`, `10b`, `10×15=150` kare) | `mfd=1, t=8` | `118,8 ms` | `104,0 ms` | `1,14×` | `73,4 ms` | **`1,42×`** | `12,83` / `4,71` | `7,22` | `1,78×` | `%100` |
| `ccvb_film_grain` (`640×480`, `8b`, `100×3=300` kare) | `mfd=1, t=8` | `382,3 ms` | `301,4 ms` | `1,27×` | `179,3 ms` | **`1,68×`** | `14,94` / `6,87` | `9,97` | `1,50×` | `%100` |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tile, `8×15=120` kare) | `mfd=0, t=16` | `1 163,9 ms` | `205,9 ms` | **`5,65×`** | `143,7 ms` | **`1,43×`** | `158,48` / `67,90` | `284,57` | **`0,56×`** | `%100` |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tile, `8×15=120` kare) | `mfd=4, t=16` | `1 163,9 ms` | `208,7 ms` | **`5,58×`** | `140,5 ms` | **`1,49×`** | `161,47` / `68,71` | `291,93` | **`0,55×`** | `%100` |
| `test25_8k_444_fg` (`7680×4320`, `8b` `4:4:4`, `120` tile + FG, `6` kare) | `mfd=0, t=8` | `1 406,9 ms` | `384,8 ms` | **`3,66×`** | `191,4 ms` | **`2,01×`** (`t=4`: `2,23×`) | `1 207,20` / `861,96` | `1 414,59` | **`0,85×`** (`t=4`: `0,96×`) | `%100` |
| `test25_8k_120_tile` (`7680×4320`, `12b`, `120` tile, `10×2=20` kare) | `mfd=0, t=16` | `1 445,1 ms` | `477,7 ms` | **`3,03×`** | `168,7 ms` | **`2,83×`** | `883,56` / `686,83` | `1 020,79` | **`0,87×`** | `%100` |

#### `.avif` Görüntü Çözme + `RGBA8` Dönüşümü (`av1oxide::avif` vs C `libavif 1.4.2 + dav1d 1.5.4`)

| İş Yükü | Görüntü / Kip | `av1oxide` Süre (`ms`) | C `libavif` Süre (`ms`) | Süre Oranı (`av1oxide / libavif`) | `av1oxide` Tepe RSS (`MiB`) | `av1oxide` Canlı Yığın (`MiB`) | C `libavif` Tepe RSS (`MiB`) | RSS Oranı (`av1oxide / libavif`) | `RGBA8` MD5 Eşitliği |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`avif_corpus_156_t4`** | `156` geçerli `.avif` görüntüsü, `n_threads = 4` (`1` tur) | `2 362,4 ms` | `2 467,9 ms` | **`0,957×`** | `154,21 MiB` | `74,75 MiB` | `103,49 MiB` | **`1,49×`** | `156 / 156` (`%100`) |
| **`avif_bbb_4k_t8`** | `bbb_4k.avif` (`3840×2160`), `n_threads = 8` (`5` tekrar) | `126,9 ms` | `273,7 ms` | **`0,464×`** | `89,29 MiB` | `43,53 MiB` | `51,43 MiB` | **`1,74×`** | `5 / 5` (`%100`) |

---

### 2. `aarch64` — Apple M4 Pro (`14C = 10P + 4E`, `NEON + DOTPROD`, `v0.4.0` Taban Çizgisi)

> Bu bölümdeki tablolar Apple M4 Pro (`aarch64-apple-darwin`) donanımında ölçülen `v0.4.0` taban çizgisini içerir; `v0.5.0` Apple M4 Pro ölçümleri macOS donanım doğrulaması ardından güncellenecektir.

#### Tek-Thread (`n_threads = 1`, `max_frame_delay = 1`) Küme Özeti (`v0.4.0`)

| Küme | SIMD Kipi | Akış / Kare | `av1oxide` Toplam (`ms`) | `dav1d` Toplam (`ms`) | Duvar Süresi Oranı (`av1oxide / dav1d`) | Geometrik Ortalama | Ortanca (`p50`) | Küme Maks. `av1oxide` Tepe RSS (`MiB`) | Küme Maks. `dav1d` Tepe RSS (`MiB`) | RSS Oranı (`av1oxide / dav1d`) | MD5 Eşitliği |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`korpus`** | `NEON + DOTPROD` (aynı-ISA) | `19` akış / `950` çözülen kare | `738,5 ms` | `474,1 ms` | **`1,558×`** | `1,623×` | `1,599×` | `16,52 MiB` (`10,89 MiB` canlı yığın) | `8,91 MiB` | `1,85×` | `19 / 19` (`%100`) |
| **`temsilci`** | `NEON + DOTPROD` (aynı-ISA) | `8` akış / `485` çözülen kare | `1 994,9 ms` | `1 123,8 ms` | **`1,775×`** | `1,560×` | `1,632×` | `1 234,05 MiB` (`1 212,62 MiB` canlı yığın) | `610,30 MiB` | `2,02×` | `8 / 8` (`%100`) |
| **`korpus`** | `tespit` (`dav1d` + `I8MM`) | `19` akış / `950` çözülen kare | `734,2 ms` | `474,5 ms` | **`1,547×`** | `1,647×` | `1,585×` | `16,50 MiB` (`10,89 MiB` canlı yığın) | `8,89 MiB` | `1,86×` | `19 / 19` (`%100`) |
| **`temsilci`** | `tespit` (`dav1d` + `I8MM`) | `8` akış / `485` çözülen kare | `1 982,5 ms` | `1 122,2 ms` | **`1,767×`** | `1,570×` | `1,622×` | `1 233,30 MiB` (`1 212,62 MiB` canlı yığın) | `610,31 MiB` | `2,02×` | `8 / 8` (`%100`) |

#### Tek-Thread (`n_threads = 1`) Temsilci Akışlar Detayı (`NEON + DOTPROD`, `v0.4.0`)

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

#### Çok-Thread (`Apple M4 Pro`, `v0.4.0`) AV1 Karşılaştırması

| Akış | Kip (`max_frame_delay`, `n_threads`) | `av1oxide` (`t=1`) | `av1oxide` (Çok-Thread) | `av1oxide` Öz-Hızlanma | `dav1d` (Çok-Thread) | Süre Oranı (`av1oxide / dav1d`) | `av1oxide` Tepe RSS / Canlı Yığın (`MiB`) | `dav1d` Tepe RSS (`MiB`) | RSS Oranı (`av1oxide / dav1d`) | MD5 Eşitliği |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `vq002_inter_gop` (`432×240`, `8b`, `10×10=100` kare) | `mfd=1, t=4` | `311,1 ms` | `277,9 ms` | `1,12×` | `231,1 ms` | **`1,20×`** | `9,50` / `3,39` | `5,97` | `1,59×` | `%100` |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tile, `8×15=120` kare) | `mfd=0, t=14` | `1 550,6 ms` | `238,2 ms` | **`6,51×`** | `137,5 ms` | **`1,73×`** | `140,00` / `94,42` | `102,77` | `1,36×` | `%100` |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tile, `8×15=120` kare) | `mfd=4, t=14` | `1 550,6 ms` | `239,4 ms` | **`6,48×`** | `136,9 ms` | **`1,75×`** | `139,89` / `94,69` | `103,03` | `1,36×` | `%100` |
| `test25_8k_444_fg` (`7680×4320`, `8b` `4:4:4`, `120` tile + FG, `6` kare) | `mfd=0, t=14` | `1 418,8 ms` | `147,7 ms` | **`9,60×`** | `66,2 ms` | **`2,23×`** | `2 055,83` / `1 705,67` | `1 398,83` | `1,47×` | `%100` |
| `test25_8k_120_tile` (`7680×4320`, `12b`, `120` tile, `10×2=20` kare) | `mfd=0, t=14` | `1 694,6 ms` | `238,7 ms` | **`7,10×`** | `97,3 ms` | **`2,45×`** | `1 410,64` / `1 131,39` | `1 097,81` | `1,28×` | `%100` |

#### `.avif` Görüntü Çözme + `RGBA8` Dönüşümü (`Apple M4 Pro`, `v0.4.0`)

| İş Yükü | Görüntü / Kip | `av1oxide` Süre (`ms`) | C `libavif` Süre (`ms`) | Süre Oranı (`av1oxide / libavif`) | `av1oxide` Tepe RSS (`MiB`) | `av1oxide` Canlı Yığın (`MiB`) | C `libavif` Tepe RSS (`MiB`) | RSS Oranı (`av1oxide / libavif`) | `RGBA8` MD5 Eşitliği |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`avif_corpus_156_t4`** | `156` geçerli `.avif` görüntüsü, `n_threads = 4` (`1` tur) | `3 348,8 ms` | `2 600,8 ms` | **`1,288×`** | `335,95 MiB` | `106,74 MiB` | `205,16 MiB` | **`1,64×`** | `156 / 156` (`%100`) |
| **`avif_bbb_4k_t8`** | `bbb_4k.avif` (`3840×2160`), `n_threads = 8` (`5` tekrar) | `366,1 ms` | `222,8 ms` | **`1,644×`** | `104,80 MiB` | `67,06 MiB` | `51,31 MiB` | **`2,04×`** | `5 / 5` (`%100`) |

---

### 3. Süre ve Bellek Farkının Teknik Analizi

#### Çalışma Zamanı Hızını Etkilemeyen Kontroller (`~%0` Etki)

- Bellek bütçesi ve kare boyutu kontrolleri (`Config::allocation_limit`, `Config::frame_size_limit`) piksel veya blok döngülerinin içinde değil, yalnızca kare ve tampon tahsisi sırasında çalıştırılır; kare çözme hızı üzerinde ölçülebilir bir etkisi yoktur.

#### `dav1d` ile AV1 Süre Farkının Ölçülmüş Teknik Nedenleri (`~1,49×–1,77×`)

- **Dizi Sınır Kontrolleri (Çözme süresinin `%23–%50`'si):** Yüksek çözünürlüklü karelerde milyonlarca `4×4` / `8×8` blok ve komşuluk tablosu gezilir. C `dav1d` bu yapılara sınır kontrolsüz ham işaretçi aritmetiğiyle (`ptr += offset`) erişirken, `av1oxide` içinde DSP dışındaki tüm çözücü çekirdeği `#![forbid(unsafe_code)]` altında derlenir ve her dilim erişiminde (`slice[i]`) çalışma zamanı sınır kontrolü yürütülür (iki tarafta da SIMD kapatıldığında saf C ile saf Rust arasında ölçülen `~1,65×–1,71×` oranının temel kaynağıdır).
- **SIMD Giriş Doğrulaması ve Sıfır Taşan Okuma (`~2–5 ns` / çağrı):** C `dav1d` Assembly rutinleri girişte uzunluk kontrolü yapmaz ve tampon sonundan dışarı 16–64 bayt dolgu bölgesine taşan okuma (over-read) gerçekleştirir. `av1oxide` içindeki her SIMD fonksiyonu girişte dilim boyutu önkoşullarını doğrular ve dilim sınırlarının dışına okuma yapmamak için kenarlarda maskeli/sınırlı yükleme kullanır; `10–20 ns` süren küçük (`4×4`) blok çağrılarında bu doğrulama `%15–%25` ek süre oluşturur.
- **El Yazısı Assembly ve Derleyici (`LLVM`) Kod Üretimi:** `dav1d` yaklaşık `265 bin` satır el yazısı NASM/GAS Assembly kullanarak standart fonksiyon çağrı kurallarını (ABI) atlayıp CPU bayrak yazmaçlarını ve 32 adet `zmm` yazmacını doğrudan yönetir; `av1oxide` ise kararlı Rust `core::arch` intrinsiklerini standart LLVM ABI çerçevesi üzerinden derler.

#### Bellek (`Peak RSS`) ve `.avif` Davranışının Teknik Nedenleri

- **Tek-Thread Taban İzi ve İkili Kod Sayfaları (Küçük akışlarda `1,49×–2,05×` RSS):** Küçük ve orta çözünürlüklü akışlarda (`16×16` – `640×480`) `av1oxide` `v0.5.0` canlı yığın kullanımı `0,29–8,25 MiB` (`x86_64`) iken işletim sistemi süreç yerleşik küme boyutu (`x86_64` Linux üzerinde `7,5–17,0 MiB`; C `dav1d`: `3,7–11,4 MiB`), paylaşımlı kütüphane bağlaması yerine statik bağlanan Rust çalışma zamanı ile çoklu-ISA SIMD kod sayfalarını (`x86_64` ELF: `10,2 MiB`, `aarch64` Mach-O `v0.4.0`: `2,8 MiB`) içerir.
- **8K Şerit Bazlı Film Grain ve `LrCdefBackup` (`x86_64` tek-thread `1,10×`, çok-thread `0,85×–0,96×` RSS):** Tam kare `pre_cdef` kopyası yerine `4` satırlık `LrCdefBackup` şerit sınır tamponu ve `32` satırlık kayan şerit tamponu üzerinde yerinde Film Grain sentezi (`refresh_frame_flags == 0` için `Frame::from_owned_pic`) sayesinde `test25_8k_444_fg` (`7680×4320`, `4:4:4` + FG) tek-thread Tepe RSS değeri `v0.4.0` sürümündeki `1 231,4 MiB` (`2,02×`) düzeyinden `667,5 MiB` (`1,10×`, `dav1d` `609,0 MiB`) düzeyine, çok-thread `t=8` Tepe RSS değeri ise `1 990,9 MiB` (`1,41×`) düzeyinden `1 207,2 MiB` (`0,85×`, `dav1d` `1 414,6 MiB` değerinden `%14,7` daha düşük; `t=4`: `0,96×`; `test25_8k_120_tile` `t=16`: `0,87×`) düzeyine inmiştir.
- **16-Thread Çok-Karolu Akışlarda Paylaşımlı Geçici Havuz (`x86_64` Üzerinde `0,55×–0,56×` RSS):** `1024×1024` `16` karolu `10-bit` akışta `n_threads = 16` kipinde (`test10103_coklu_tile`), `av1oxide` Linux `x86_64` üzerinde `158,5–161,5 MiB` Tepe RSS (`67,9–68,7 MiB` canlı yığın) kullanırken C `dav1d` `284,6–291,9 MiB` (`0,55×–0,56×`) kullanır; bunun nedeni `av1oxide` tarafında her kare bağlamı için tam karo/katsayı arenalarını peşin tahsis etmek yerine sınırlı `ScratchPool` işçi yuvalarının ve `DecodeBufPool` düzlemlerinin talep üzerine paylaşılmasıdır.
- **`.avif` Doğal `u8` Düzlemler ve Şerit-Paralel YUVA→RGB (`x86_64` Üzerinde `0,464×–0,957×` Süre):** `8-bit` AVIF kanallarının doğal `u8` (`AvifPlane::U8`) olarak saklanması ve özelleştirilmiş `(9, 3, 3, 1)/16` satır-çifti YUVA→RGB dönüşümünün `std::thread::scope` şeritleriyle paralelleştirilmesi sonucunda `x86_64` `avif_corpus_156_t4` süresi `3 617,2 ms` (`1,460×`) değerinden `2 362,4 ms` (`0,957×`, C `libavif 1.4.2 + dav1d 1.5.4` toplamından `%4,3` daha hızlı) değerine, canlı yığın tepesi `99,15 MiB` değerinden `74,75 MiB` değerine inmiştir; `4K` `bbb_4k.avif` (`t=8`, `5` tekrar) iş yükünde süre `407,2 ms` (`1,488×`) değerinden `126,9 ms` (`0,464×`, `-DAVIF_LIBYUV=OFF` derlenmiş C `libavif` sürümünden `2,16×` daha hızlı) değerine, canlı yığın tepesi ise `67,06 MiB` değerinden `43,53 MiB` değerine düşmüştür.

---

## Lisans

[BSD-2-Clause](LICENSE)
