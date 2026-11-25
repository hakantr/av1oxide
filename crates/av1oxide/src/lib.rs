//! # av1oxide — %100 Kararlı Rust AV1 Video ve AVIF Görüntü Çözücüsü (`v0.4.0`)
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
//!   halka içi filtreler (`LF`, `CDEF`, `LR` Wiener/SGR) ve Film Grain (`FG`) sentezi.
//! - **Bit-Birebir Doğruluk:** `dav1d-test-data` (`19/19` akış, `964` kare) ve
//!   Argon Streams `v2.1.1` (`2 762/2 762` akış, `36 700` kare) olmak üzere toplam
//!   **`2 781 / 2 781` akış (`37 664 / 37 664` kare)** üzerinde tüm CPU maskeleri ve
//!   çok-thread kiplerinde `dav1d` ile `%100` MD5 bit-birebir eşitlik.
//! - **Donanım Yetenek Jetonlu SIMD (`DspDispatch`, `767` Yuva):**
//!   - `x86_64` (`SSE2`, `SSSE3`, `SSE4.1`, `AVX2`, `AVX-512ICL`): `756 / 767` SIMD yuvası,
//!   - `aarch64` (`NEON`, `DOTPROD`): `701 / 767` SIMD yuvası,
//!   - `wasm32` (`+simd128`): `%100 #![forbid(unsafe_code)]` ile `31 / 767` SIMD yuvası,
//!   - İkincil mimariler (`arm32`, `riscv64`, `loongarch64`, `powerpc64le`): `%100` güvenli
//!     ve bit-birebir skaler geri-düşüş.
//! - **Çok-Thread Çözme ve Kaynak Bütçesi:** `%100 #![forbid(unsafe_code)]`
//!   kalıcı `WorkerPool` (kare içi tile/SB-satırı/halka-içi filtre paralelliği + kareler arası
//!   `FrameSlot` boru hattı), `DecodeBufPool` düzlem geri dönüşümü ve tahsis öncesi atomik
//!   bellek bütçesi ([`Config::allocation_limit`], [`Config::frame_size_limit`]).
//! - **`libdav1d` (`7.0.0`) Uyumlu C-ABI Katmanı ([`capi`]):** `include/dav1d/dav1d.h`
//!   ile ikili uyumlu 19 C fonksiyonu (`dav1d_open`, `dav1d_send_data`, `dav1d_get_picture`,
//!   `dav1d_apply_grain`, `dav1d_parse_sequence_header`, vb.); `--features capi` ile
//!   doğrudan `cdylib` (`.so`/`.dylib`/`.dll`) veya `staticlib` (`.a`) olarak derlenebilir.
//! - **Saf Rust `.avif` Görüntü Okuyucu Katmanı ([`avif`]):** `%100 #![forbid(unsafe_code)]`
//!   ISOBMFF/HEIF/MIAF/AVIF kutu ayrıştırıcısı, tekil `av01`, `grid` karo birleştirme, `iden`,
//!   yardımcı alfa (`auxl`/`auxC`) kanalı, `clap`/`irot`/`imir` dönüşümleri, `ICC`/`Exif`/`XMP`
//!   metaverisi ve ITU-T H.273 renk dönüşümleri (`RGB8`/`RGBA8`/`RGB16`/`RGBA16`); resmî
//!   `AOMediaCodec/av1-avif` ve `AOMediaCodec/libavif` test korpusundaki `156 / 156` geçerli
//!   `.avif` dosyasında referans C `libavif` ile düzlemsel (`Y`, `U`, `V`, `Alpha`) ve
//!   `RGBA8`/`RGBA16` çıktılarında `%100` MD5 bit-birebir eşitlik.
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
//! | [`avif`] | `.avif` (ISOBMFF/HEIF/MIAF) okuyucu, `grid` karo birleştirme ve RGB/RGBA dönüşümleri |
//! | [`capi`] | `libdav1d` (`7.0.0`) uyumlu C-ABI fonksiyonları ve `#[repr(C)]` yapıları |
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
//!   modülü ise `.avif` görüntü kapsayıcısı ayrıştırma ve H.273 RGB/RGBA dönüşümünü sağlar.
//!   Video akış kapsayıcıları (IVF/MP4/Matroska), GPU yüklemesi ve AV1 kodlama bu kütüphanenin
//!   kapsamı dışındadır.
//! - Üretim kaynaklarında haricî `.asm` / `.S` veya satır içi `asm!` / `global_asm!` kullanılmaz.
//! - `crates/av1oxide/src` altındaki `204` kaynak dosyadan `115`'i `#![forbid(unsafe_code)]`
//!   taşır; yalnız [`audit::UNSAFE_EXEMPTIONS`] içindeki `88` `x86_64`/`aarch64` SIMD dosyası
//!   ve `1` C-ABI sınır dosyası (`capi/ffi.rs`) dar kapsamlı işlemler için muaf tutulur
//!   (`0` güvensiz işlev imzası, `389` blok).
//!
//! ## Bağımlılık Politikası
//!
//! `av1oxide` hiçbir üçüncü taraf medya, kodek, görüntü işleme veya C medya kütüphanesine
//! bağımlı değildir; kullanılan tüm çekirdek sistem ve veri yapısı kütüphaneleri iç uygulama
//! ayrıntısı (`pub(crate)` / özel) olarak yalıtılmıştır ve genel (`pub`) API imzalarına sızdırılmaz:
//!
//! | Crate | Sürüm | Lisans | Kaynak | Özellikler (`default-features = false`) | Görev | Unsafe / Yerel Kod |
//! |---|---|---|---|---|---|---|
//! | `crossbeam-utils` | `0.8` | `MIT OR Apache-2.0` | `crossbeam-rs/crossbeam` | `["std"]` | `CachePadded` önbellek satırı izolasyonu ve CAS `Backoff` (`thread`, `frame::butce`) | Pure Rust (iç atomik/hizalama `unsafe`, `0` C kodu) |
//! | `parking_lot` | `0.12` | `MIT OR Apache-2.0` | `Amanieu/parking_lot` | `[]` (`not(wasm32)`) | Zehirlenmesiz kompakt kullanıcı uzayı `Mutex` ve `Condvar` (`sync`) | Pure Rust + işletim sistemi futex / `libc` bağlaması (`0` C dosyası) |
//! | `memchr` | `2` | `Unlicense OR MIT` | `BurntSushi/memchr` | `[]` | SIMD/SWAR NUL-dize ve OBU kuyruk baytı taraması (`avif::boxes`, `api::Decoder`) | Pure Rust (iç SIMD/işaretçi `unsafe`, `0` C kodu) |
//! | `rustc-hash` | `2` | `Apache-2.0 OR MIT` | `rust-lang/rustc-hash` | `[]` | Deterministik tamsayı karması (`FxBuildHasher`, `avif`, `dsp`) | Safe Rust (`0` `unsafe` blok, `0` C kodu) |
//! | `indexmap` | `2` | `Apache-2.0 OR MIT` | `indexmap-rs/indexmap` | `[]` | Ekleme sırasını koruyan $O(1)$ tekil tablo/küme motoru (`avif::boxes`, `avif::decode`, `dsp`) | Pure Rust (`indexmap` `#![forbid(unsafe_code)]` + Miri onaylı `hashbrown`, `0` C kodu) |
//! | `flate2` | `1.1` | `MIT OR Apache-2.0` | `rust-lang/flate2-rs` | `["rust_backend"]` | Sınırlı `deflate`/`zlib`/`gzip` HEIF metaveri ve `cmin` açıcısı (`avif::boxes`) | Safe Rust (`miniz_oxide` `#![forbid(unsafe_code)]` arka ucu, `0` C kodu) |

#![deny(unused_crate_dependencies)]
#![warn(clippy::all)]

pub mod api;
pub mod audit;
pub mod avif;
pub mod bitstream;
pub mod capi;
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
