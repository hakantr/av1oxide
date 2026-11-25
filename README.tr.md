# av1oxide (`v0.3.1`)

[English](README.md) | **Türkçe**

`av1oxide`, VideoLAN [`dav1d`](https://code.videolan.org/videolan/dav1d) (`1.5.4`) AV1 video çözücüsünün **%100 kararlı Rust** (`rustc 1.100.0`, Edition 2024), **sıfır çalışma zamanı dış bağımlılığı**, **`libdav1d` (`7.0.0`) ikili uyumlu C-ABI katmanı** ve **saf Rust `.avif` görüntü çözücü katmanı** ile yeniden gerçekleştirimidir.

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
- **Sıfır Bağımlılıklı `.avif` Görüntü Okuyucu (`avif`):** `%100 #![forbid(unsafe_code)]` ISOBMFF/HEIF/MIAF/AVIF kapsayıcı ayrıştırıcısı; tekil `av01`, çok-thread `grid` karo birleştirme, `iden` türetilmiş görüntüler, yardımcı alfa (`auxl`/`auxC`) kanalı, `clap`/`irot`/`imir` dönüşümleri, `ICC`/`Exif`/`XMP` metaverisi ve ITU-T H.273 renk dönüşümleri (`RGB8`/`RGBA8`/`RGB16`/`RGBA16`) — lisans denetimli `AOMediaCodec/av1-avif` ve `AOMediaCodec/libavif` test korpusundaki `156 / 156` geçerli `.avif` görüntüsünde referans C `libavif` ile düzlemsel (`Y`, `U`, `V`, `Alpha`), `RGBA8` ve `RGBA16` çıktılarında `%100` MD5 bit-birebir eşitlik.

---

## Kurulum

`Cargo.toml` dosyanıza ekleyin:

```toml
[dependencies]
av1oxide = "0.3"
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

## `dav1d 1.5.4` ile Son Performans Karşılaştırma Tabloları

Tüm ölçümler `%100` MD5 bit-birebir eşitlik altında `av1oxide` (`rustc 1.100.0 --release`) ile referans C/ASM `dav1d 1.5.4` (`99f3354ed0b9b4bf3c7c9347318dbcc522c94af5`) arasında gerçekleştirilmiştir.

### 1. `x86_64` — AMD Ryzen 9 9950X3D (Zen 5, `16C / 32T`, `AVX-512ICL`)

#### Tek-Thread (`n_threads = 1`) Küme Özeti

| Küme | Akış / Kare | `av1oxide` Toplam (`ms`) | `dav1d` Toplam (`ms`) | Duvar Süresi Oranı (`av1oxide / dav1d`) | Geometrik Ortalama | Ortanca (`p50`) | MD5 Eşitliği |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`korpus`** | `19` akış / `964` kare | `647,7 ms` | `431,3 ms` | **`1,502×`** | `1,555×` | `1,767×` | `19 / 19` (`%100`) |
| **`temsilci`** | `8` akış / `485` kare | `1 980,4 ms` | `1 070,7 ms` | **`1,850×`** | `1,589×` | `1,699×` | `8 / 8` (`%100`) |

#### Tek-Thread (`n_threads = 1`) Temsilci Akışlar Detayı

| Akış | Çözünürlük | Bit / Kroma | Kare | `av1oxide` (`ms/kare`) | `av1oxide` (`FPS`) | `dav1d` (`ms/kare`) | `dav1d` (`FPS`) | Oran (`av1oxide / dav1d`) | Skaler'e Göre Hızlanma |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `422_oddheight_8b` | `64×63` | `8b` `4:2:2` | `80` | `0,292` | `3 418,8` | `0,253` | `3 957,3` | **`1,156×`** | `2,59×` |
| `422_oddheight_10b` | `64×63` | `10b` `4:2:2` | `80` | `0,297` | `3 370,4` | `0,255` | `3 915,4` | **`1,165×`** | `2,79×` |
| `vq002_inter_432x240` | `432×240` | `8b` `4:2:0` | `50` | `2,837` | `352,5` | `2,168` | `461,3` | **`1,309×`** | `4,06×` |
| `test10755_cdef10_sb64` | `1024×512` | `10b` `4:2:0` | `24` | `2,660` | `375,9` | `1,645` | `607,8` | **`1,617×`** | `4,19×` |
| `fg10_film_grain_50` | `352×288` | `10b` `4:2:0` | `100` | `0,835` | `1 197,5` | `0,469` | `2 131,7` | **`1,780×`** | `4,47×` |
| `test25_8k_444_fg` | `7680×4320` | `8b` `4:4:4` | `6` | `248,828` | `4,0` | `126,890` | `7,9` | **`1,959×`** | `1,89×` |
| `test1472_lr8_444` | `306×255` | `8b` `4:4:4` | `45` | `0,307` | `3 254,1` | `0,154` | `6 502,0` | **`1,996×`** | `4,14×` |
| `ccvb_film_grain` | `640×480` | `8b` `4:2:0` | `100` | `1,372` | `728,8` | `0,670` | `1 493,2` | **`2,047×`** | `5,34×` |

#### Çok-Thread (`AMD Ryzen 9 9950X3D`) Karşılaştırması

| Akış | Kip (`max_frame_delay`, `n_threads`) | `av1oxide` (`t=1`) | `av1oxide` (Çok-Thread) | `av1oxide` Öz-Hızlanma | `dav1d` (Çok-Thread) | Oran (`av1oxide / dav1d`) |
|---|---|---:|---:|---:|---:|---:|
| `vq002_inter_gop` (`432×240`, `8b`, `100×10` kare) | `mfd=1, t=8` | `272,6 ms` | `257,8 ms` | `1,06×` | `251,7 ms` | **`1,02×`** |
| `fg10_film_grain_50` (`352×288`, `10b`, `150×15` kare) | `mfd=1, t=8` | `122,4 ms` | `104,4 ms` | `1,17×` | `75,5 ms` | **`1,38×`** |
| `ccvb_film_grain` (`640×480`, `8b`, `300×3` kare) | `mfd=1, t=8` | `387,1 ms` | `310,5 ms` | `1,25×` | `203,0 ms` | **`1,53×`** |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tile) | `mfd=0, t=16` | `1 217,5 ms` | `234,3 ms` | **`5,20×`** | `136,9 ms` | **`1,71×`** |
| `test25_8k_444_fg` (`7680×4320`, `8b` `4:4:4`, `120` tile + FG) | `mfd=0, t=8` | `1 476,5 ms` | `159,6 ms` | **`9,25×`** | `63,1 ms` | **`2,53×`** (`t=4`: `2,18×`) |
| `test25_8k_120_tile` (`7680×4320`, `12b`, `120` tile) | `mfd=0, t=16` | `1 589,0 ms` | `401,0 ms` | **`3,96×`** | `99,9 ms` | **`4,01×`** |

---

### 2. `aarch64` — Apple M4 Pro (`14C = 10P + 4E`, `NEON + DOTPROD`)

#### Tek-Thread (`n_threads = 1`) Küme Özeti

| Küme | Akış / Kare | SIMD Düzeyi | `av1oxide` Toplam (`ms`) | `dav1d` Toplam (`ms`) | Duvar Süresi Oranı (`av1oxide / dav1d`) | Geometrik Ortalama | Ortanca (`p50`) | MD5 Eşitliği |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| **`korpus`** | `19` akış / `964` kare | `NEON + DOTPROD` | `796,9 ms` | `486,4 ms` | **`1,638×`** | `1,680×` | `1,786×` | `19 / 19` (`%100`) |
| **`temsilci`** | `8` akış / `485` kare | `NEON + DOTPROD` | `1 995,2 ms` | `1 117,0 ms` | **`1,786×`** | `1,675×` | `1,649×` | `8 / 8` (`%100`) |

#### Çok-Thread (`Apple M4 Pro`) Karşılaştırması

| Akış | Kip (`max_frame_delay`, `n_threads`) | `av1oxide` (`t=1`) | `av1oxide` (Çok-Thread) | `av1oxide` Öz-Hızlanma | `dav1d` (Çok-Thread) | Oran (`av1oxide / dav1d`) |
|---|---|---:|---:|---:|---:|---:|
| `vq002_inter_gop` (`432×240`, `8b`) | `mfd=1, t=4` | `318,0 ms` | `290,7 ms` | `1,09×` | `236,9 ms` | **`1,23×`** |
| `test10103_coklu_tile` (`1024×1024`, `10b`, `16` tile) | `mfd=0, t=14` | `1 558,9 ms` | `236,2 ms` | **`6,60×`** | `135,7 ms` | **`1,74×`** |
| `test25_8k_444_fg` (`7680×4320`, `8b` `4:4:4`, `120` tile + FG) | `mfd=0, t=14` | `1 410,1 ms` | `149,6 ms` | **`9,43×`** | `68,0 ms` | **`2,20×`** |
| `test25_8k_120_tile` (`7680×4320`, `12b`, `120` tile) | `mfd=0, t=14` | `1 687,5 ms` | `233,0 ms` | **`7,24×`** | `95,0 ms` | **`2,45×`** |

---

### 3. Süre Farkının Teknik Analizi

#### Çalışma Zamanı Hızını Etkilemeyen Kontroller (`~%0` Etki)

- Bellek bütçesi ve kare boyutu kontrolleri (`Config::allocation_limit`, `Config::frame_size_limit`) piksel veya blok döngülerinin içinde değil, yalnızca kare ve tampon tahsisi sırasında çalıştırılır; kare çözme hızı üzerinde ölçülebilir bir etkisi yoktur.

#### `dav1d` ile Süre Farkının Ölçülmüş Teknik Nedenleri (`~1,50×–1,70×`)

- **Dizi Sınır Kontrolleri (Çözme süresinin `%23–%50`'si):** Yüksek çözünürlüklü karelerde milyonlarca `4×4` / `8×8` blok ve komşuluk tablosu gezilir. C `dav1d` bu yapılara sınır kontrolsüz ham işaretçi aritmetiğiyle (`ptr += offset`) erişirken, `av1oxide` içinde DSP dışındaki tüm çözücü çekirdeği `#![forbid(unsafe_code)]` altında derlenir ve her dilim erişiminde (`slice[i]`) çalışma zamanı sınır kontrolü yürütülür (iki tarafta da SIMD kapatıldığında saf C ile saf Rust arasında ölçülen `~1,65×–1,71×` oranının temel kaynağıdır).
- **SIMD Giriş Doğrulaması ve Sıfır Taşan Okuma (`~2–5 ns` / çağrı):** C `dav1d` Assembly rutinleri girişte uzunluk kontrolü yapmaz ve tampon sonundan dışarı 16–64 bayt dolgu bölgesine taşan okuma (over-read) gerçekleştirir. `av1oxide` içindeki her SIMD fonksiyonu girişte dilim boyutu önkoşullarını doğrular ve dilim sınırlarının dışına okuma yapmamak için kenarlarda maskeli/sınırlı yükleme kullanır; `10–20 ns` süren küçük (`4×4`) blok çağrılarında bu doğrulama `%15–%25` ek süre oluşturur.
- **El Yazısı Assembly ve Derleyici (`LLVM`) Kod Üretimi:** `dav1d` yaklaşık `265 bin` satır el yazısı NASM/GAS Assembly kullanarak standart fonksiyon çağrı kurallarını (ABI) atlayıp CPU bayrak yazmaçlarını ve 32 adet `zmm` yazmacını doğrudan yönetir; `av1oxide` ise kararlı Rust `core::arch` intrinsiklerini standart LLVM ABI çerçevesi üzerinden derler.

---

## Lisans

[BSD-2-Clause](LICENSE)
