//! # av1oxide — %100 Kararlı Rust AV1 Video ve AVIF Görüntü Çözücüsü (`v0.5.0`)
//!
//! `av1oxide`, VideoLAN [`dav1d`](https://code.videolan.org/videolan/dav1d) (`1.5.4`)
//! AV1 video çözücüsünün **%100 kararlı Rust** (`rustc 1.100.0`, Edition 2024) ve
//! **sıfır üçüncü taraf medya veya kodek bağımlılığı** ile yeniden gerçekleştirimidir.
//!
//! ## Özellikler
//!
//! - **AV1 Bit Akışı ve Profil Kapsamı:** Main (`0`), High (`1`) ve Professional (`2`)
//!   profilleri; `8`, `10` ve `12` bit derinliği (`4:2:0`, `4:2:2`, `4:4:4`, `Monochrome`);
//!   OBU katmanı, Sequence/Frame Header, MSAC entropi, ters dönüşüm (`ITX`), intra tahmin
//!   (`IPRED`, CfL, Palette), hareket telafisi (`MC`, Warp, Superres/Resize, IntraBC, `REFMVS`),
//!   halka içi filtreler (`LF`, `CDEF`, `LR` Wiener/SGR) ve şerit bazlı Film Grain (`FG`) sentezi.
//! - **Bit-Birebir Doğruluk:** `dav1d-test-data` (`19/19` akış, `964` kare) ve
//!   Argon Streams `v2.1.1` (`2 762/2 762` akış, `36 700` kare) olmak üzere toplam
//!   **`2 781 / 2 781` akış (`37 664 / 37 664` kare)** üzerinde tüm CPU maskeleri ve
//!   çok-thread kiplerinde `dav1d` ile `%100` MD5 bit-birebir eşitlik.
//! - **Donanım Yetenek Jetonlu SIMD (`DspDispatch`, `767` Yuva):**
//!   - `x86_64` (`SSE2`, `SSSE3`, `SSE4.1`, `AVX2`, `AVX-512ICL`): `767 / 767` (`%100`) SIMD yuvası,
//!   - `aarch64` (`NEON`, `DOTPROD`): `767 / 767` (`%100`) SIMD yuvası,
//!   - `wasm32` (`+simd128`): `%100 #![forbid(unsafe_code)]` ile `682 / 767` (`%88,9`) SIMD yuvası,
//!   - İkincil mimariler (`arm32`, `riscv64`, `loongarch64`, `powerpc64le`): `%100` güvenli
//!     ve bit-birebir skaler geri-düşüş.
//! - **Çok-Thread Çözme ve Kaynak Bütçesi:** `%100 #![forbid(unsafe_code)]`
//!   kalıcı `WorkerPool` (kare içi tile/SB-satırı/halka-içi filtre paralelliği + kareler arası
//!   `FrameSlot` boru hattı), `DecodeBufPool` düzlem geri dönüşümü, 4-satırlık `LrCdefBackup`
//!   şerit tamponu, 32-satırlık yerinde Film Grain sentezi ve tahsis öncesi atomik bellek bütçesi
//!   ([`Config::allocation_limit`], [`Config::frame_size_limit`]).
//! - **İsteğe Bağlı `#![no_std] + alloc` Desteği (`default = ["std"]`):** `--no-default-features`
//!   ile tam tek-thread AV1 + AVIF çekirdeği `core + alloc` üzerinde derlenir (`wasm32-unknown-unknown`,
//!   `aarch64`, `armv7`, `riscv64gc` dahil).
//! - **Kapsayıcı Yardımcıları ([`container`]):** `%100 #![forbid(unsafe_code)]` sıfır kopyalı
//!   IVF (`container::ivf::IvfParser`, `std` altında `IvfReader`) ve AV1 Annex-B
//!   (`container::annexb::AnnexBDemuxer`, `read_leb128`) ayrıştırıcıları.
//! - **`libdav1d` (`1.5.4` / API `7.0.0`) Uyumlu C-ABI Katmanı (`capi`):** `include/dav1d/dav1d.h`
//!   ile ikili uyumlu 19 C fonksiyonu (`dav1d_open`, `dav1d_send_data`, `dav1d_get_picture`,
//!   `dav1d_apply_grain`, `dav1d_parse_sequence_header`, vb.) ve `[package.metadata.capi]` /
//!   `dav1d.pc` drop-in sistem paketlemesi (değiştirilmemiş C `libavif 1.4.2` ile `162 / 162` parite).
//! - **Saf Rust `.avif` / `.avis` Görüntü Okuyucu Katmanı ([`avif`]):** `%100 #![forbid(unsafe_code)]`
//!   ISOBMFF/HEIF/MIAF/AVIF kutu ayrıştırıcısı; 8-bit doğal `u8` düzlem temsili ve şerit-paralel
//!   YUVA→RGB dönüşümü, tekil `av01`, `grid` karo birleştirme, `iden`, yardımcı alfa (`auxl`/`auxC`),
//!   `clap`/`irot`/`imir` dönüşümleri, ISO 21496-1 / Ultra HDR `tmap` Gain-Map (`9 / 9` parite),
//!   akışlı animasyonlu `.avis` çözme ([`avif::AvifSequenceDecoder`], `6 / 6` parite), `ICC`/`Exif`/`XMP`
//!   metaverisi ve doğrusal/doğrusal olmayan ITU-T H.273 renk dönüşümleri (`ICtCp` `mc=14` ve
//!   `BT.2020 CL` `mc=10` dahil; `RGB8`/`RGBA8`/`RGB16`/`RGBA16`); lisans denetimli `AOMediaCodec/av1-avif`
//!   ve `AOMediaCodec/libavif` korpusundaki `162 / 162` (durağan: `156` geçerli + `6` negatif) dosyada
//!   referans C `libavif 1.4.2` ile `%100` MD5 bit-birebir eşitlik.
//! - **Ayrı `image` Ekosistem Köprüsü (`av1oxide-image`):** Çalışma alanı içindeki bağımsız
//!   `av1oxide-image` paketi ana `av1oxide` bağımlılık ağacını değiştirmeden `image::ImageDecoder`
//!   uyumluluğu sunar.
//!
//! ## Hızlı Başlangıç Örneği
//!
//! ```rust
//! use av1oxide::{BitDepth, Config, Decoder, Packet, PlaneType};
//!
//! // Tek karelik (1x1, 8-bit) geçerli AV1 OBU paketi (TD + Sequence Header + Keyframe).
//! const SAMPLE_OBU: &[u8] = &[
//!     0x12, 0x00, 0x0A, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7D, 0xFC, 0xC0, 0x20, 0x32,
//!     0x17, 0x10, 0x01, 0x91, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68,
//!     0xD2, 0x16, 0x8C, 0x64, 0xBE, 0x2E, 0xC7, 0xDA, 0xEE, 0x96,
//! ];
//!
//! let config = Config {
//!     n_threads: 1,
//!     max_frame_delay: 1,
//!     ..Config::default()
//! };
//! let mut decoder = Decoder::new(config).expect("çözücü kurulumu");
//!
//! decoder.send_packet(&Packet::new(SAMPLE_OBU)).expect("OBU paketi kabul edilmeli");
//! let frame = decoder
//!     .receive_frame()
//!     .expect("çözme hatasız tamamlanmalı")
//!     .expect("görünür kare üretilmeli");
//!
//! assert_eq!((frame.width(), frame.height()), (1, 1));
//! assert_eq!(frame.bit_depth(), BitDepth::Eight);
//! let y_plane = frame.plane(PlaneType::Y).expect("Y düzlemi mevcut olmalı");
//! assert_eq!(y_plane.row(0).map(<[u8]>::len), Some(1));
//!
//! decoder.flush();
//! assert!(decoder.receive_frame().expect("boşaltma hatasız").is_none());
//! ```
//!
//! ## Modül Haritası
//!
//! | Modül | İçerik |
//! |---|---|
//! | [`api`] | [`Config`], [`Packet`], [`Decoder`] ve [`DecoderState`] durum makinesi |
//! | [`avif`] | `.avif` / `.avis` (ISOBMFF/HEIF/MIAF) okuyucu, `grid`, `tmap` Gain-Map, [`avif::AvifSequenceDecoder`] ve H.273 RGB/RGBA dönüşümleri |
//! | `capi` | (`std`) `libdav1d` (`1.5.4` / API `7.0.0`) uyumlu C-ABI fonksiyonları ve `#[repr(C)]` yapıları |
//! | [`container`] | Sıfır kopyalı IVF (`container::ivf`) ve AV1 Annex-B (`container::annexb`) ayrıştırıcıları |
//! | [`frame`] | [`Frame`], [`FramePlane`], [`PlaneLayout`], [`PlaneType`], [`BitDepth`], [`ChromaFormat`] |
//! | [`error`] | [`Error`], [`Result`], [`SizeStage`] |
//! | [`dsp`] | [`dsp::DspDispatch`] (`767` yuva; `scalar`, `x86`, `arm`, `wasm`) |
//! | [`cpu`] | [`cpu::CpuFlags`], `detect()`, [`cpu::CpuLevel`] ve mimari yetenek jetonları |
//! | [`audit`] | [`audit::UNSAFE_EXEMPTIONS`] muafiyet defteri |
//! | [`bitstream`] | Sınır denetimli bit okuyucu, OBU, Sequence Header ve Frame Header |
//! | [`entropy`] | MSAC aritmetik çözücü ve CDF güncelleme |
//! | [`decode`] | Tile/blok/katsayı çözme, yeniden kurulum, halka içi filtreler ve referans durumu |
//! | [`tables`] | Normatif AV1 sabit tabloları |
//! | [`thread`] | `%100 #![forbid(unsafe_code)]` `WorkerPool`, `TaskBatch` ve `FrameSlot` |
//!
//! ## Kapsam ve Güvenlik Modeli
//!
//! - Çözücü çekirdeği ([`api`]) AV1 bit akışı → YUV düzlemleri + metaveri üretir; [`avif`]
//!   modülü durağan/animasyonlu `.avif`/`.avis` görüntü kapsayıcısı ayrıştırma ve H.273
//!   RGB/RGBA dönüşümünü, [`container`] modülü ise IVF ve AV1 Annex-B ayrıştırmayı sağlar.
//!   Genel amaçlı video kapsayıcıları (MP4/Matroska/WebM), GPU yüklemesi ve AV1 kodlama bu
//!   kütüphanenin kapsamı dışındadır.
//! - Üretim kaynaklarında haricî `.asm` / `.S` veya satır içi `asm!` / `global_asm!` kullanılmaz.
//! - `crates/av1oxide/src` altındaki `218` kaynak dosyadan `124`'ü doğrudan `#![forbid(unsafe_code)]`
//!   taşır (kalan `5` üst modül kökü `0` `unsafe` bloğu içerir); yalnız [`audit::UNSAFE_EXEMPTIONS`]
//!   içindeki `88` `x86_64`/`aarch64` SIMD dosyası ve `1` C-ABI sınır dosyası (`capi/ffi.rs`)
//!   dar kapsamlı işlemler için muaf tutulur (`0` güvensiz işlev imzası, `392` blok).
//!
//! ## Bağımlılık Politikası
//!
//! `av1oxide` hiçbir üçüncü taraf medya, kodek, görüntü işleme veya C medya kütüphanesine
//! bağımlı değildir; kullanılan tüm çekirdek sistem ve veri yapısı kütüphaneleri iç uygulama
//! ayrıntısı (`pub(crate)` / özel) olarak yalıtılmıştır ve genel (`pub`) API imzalarına sızdırılmaz:
//!
//! | Crate | Sürüm | Lisans | Kaynak | Özellikler (`default-features = false`) | Görev | Unsafe / Yerel Kod |
//! |---|---|---|---|---|---|---|
//! | `crossbeam-utils` | `0.8` | `MIT OR Apache-2.0` | `crossbeam-rs/crossbeam` | `[]` (`std` özelliği `av1oxide/std` ile açılır) | `CachePadded` önbellek satırı izolasyonu, `AtomicCell` ve CAS `Backoff` (`sync`, `thread`, `frame::butce`) | Pure Rust (iç atomik/hizalama `unsafe`, `0` C kodu) |
//! | `parking_lot` | `0.12` | `MIT OR Apache-2.0` | `Amanieu/parking_lot` | `[]` (isteğe bağlı: `std` & `not(wasm32)`) | Zehirlenmesiz kompakt kullanıcı uzayı `Mutex` ve `Condvar` (`sync`) | Pure Rust + işletim sistemi futex / `libc` bağlaması (`0` C dosyası) |
//! | `memchr` | `2` | `Unlicense OR MIT` | `BurntSushi/memchr` | `[]` | SIMD/SWAR NUL-dize ve OBU kuyruk baytı taraması (`avif::boxes`, `api::Decoder`) | Pure Rust (iç SIMD/işaretçi `unsafe`, `0` C kodu) |
//! | `rustc-hash` | `2` | `Apache-2.0 OR MIT` | `rust-lang/rustc-hash` | `[]` | Deterministik tamsayı karması (`FxBuildHasher`, `avif`, `dsp`) | Safe Rust (`0` `unsafe` blok, `0` C kodu) |
//! | `indexmap` | `2` | `Apache-2.0 OR MIT` | `indexmap-rs/indexmap` | `[]` | Ekleme sırasını koruyan $O(1)$ tekil tablo/küme motoru (`avif::boxes`, `avif::decode`, `dsp`) | Pure Rust (`indexmap` `#![forbid(unsafe_code)]` + Miri onaylı `hashbrown`, `0` C kodu) |
//! | `flate2` | `1.1` | `MIT OR Apache-2.0` | `rust-lang/flate2-rs` | `["rust_backend"]` (isteğe bağlı: `std`) | Sınırlı `deflate`/`zlib`/`gzip` HEIF metaveri ve `cmin` açıcısı (`avif::boxes`) | Safe Rust (`miniz_oxide` `#![forbid(unsafe_code)]` arka ucu, `0` C kodu) |

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unused_crate_dependencies)]
#![warn(clippy::all)]

extern crate alloc;

/// Yalnız `std` kipinde tanılama izlerini `stderr`'e yazar; `no_std` kipinde sessizdir.
#[cfg(feature = "std")]
macro_rules! trace_eprintln {
    ($($arg:tt)*) => {
        ::std::eprintln!($($arg)*);
    };
}

#[cfg(not(feature = "std"))]
macro_rules! trace_eprintln {
    ($($arg:tt)*) => {{
        let _ = ::core::format_args!($($arg)*);
    }};
}

pub(crate) use trace_eprintln;

pub mod api;
pub mod audit;
pub mod avif;
pub mod bitstream;
#[cfg(feature = "std")]
pub mod capi;
pub mod container;
pub mod cpu;
pub mod decode;
pub mod dsp;
pub mod entropy;
pub mod error;
pub mod frame;
pub(crate) mod sync;
pub mod tables;
pub mod thread;

#[cfg(feature = "msac-trace")]
pub mod trace;

pub use api::{Config, Decoder, DecoderState, Packet};
pub use error::{Error, Result, SizeStage};
pub use frame::{BitDepth, ChromaFormat, Frame, FrameMetadata, FramePlane, PlaneLayout, PlaneType};

/// Kütüphane sürümü (`Cargo.toml` içindeki `package.version`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn surum_dizesi_bos_degil() {
        assert!(!super::VERSION.is_empty(), "CARGO_PKG_VERSION boş olmamalı");
    }

    #[test]
    fn cozucu_obu_ayristirir_eksik_pakette_kare_uretmez() {
        use crate::api::{Config, Packet};

        let mut decoder = super::Decoder::new(Config::default()).expect("kurulum başarılı olmalı");
        // Temporal delimiter OBU (tip 2, boyut 0): geçerli ve hatasız.
        assert_eq!(decoder.send_packet(&Packet::new(&[0x12, 0x00])), Ok(()));
        // Boyut alanı kalan bayttan büyük OBU hata döner (panik yok).
        assert!(matches!(
            decoder.send_packet(&Packet::new(&[0x0A, 0x05])),
            Err(super::Error::Bitstream(_))
        ));
        // Paket kare tamamlamaz.
        assert!(matches!(decoder.receive_frame(), Ok(None)));
    }
}
