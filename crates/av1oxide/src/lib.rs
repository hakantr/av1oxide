//! # av1oxide — %100 Kararlı Rust AV1 Video Çözücüsü (`v0.1.0`)
//!
//! `av1oxide`, VideoLAN [`dav1d`](https://code.videolan.org/videolan/dav1d) (`1.5.4`)
//! AV1 video çözücüsünün **%100 kararlı Rust** (`rustc 1.100.0`, Edition 2024) ve
//! **sıfır çalışma zamanı dış bağımlılığı** ile yeniden gerçekleştirimidir.
//!
//! ## Özellikler
//!
//! - **Tam AV1 Bit Akışı ve Profil Kapsamı:** Main (`0`), High (`1`) ve Professional (`2`)
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
//! - Çözücü çekirdeği **yalnızca** AV1 bit akışı → YUV düzlemleri + metaveri üretir; kapsayıcı
//!   (IVF/MP4/Matroska/AVIF) ayrıştırma, YUV→RGB dönüşümü, GPU yüklemesi ve AV1 kodlama
//!   bu kütüphanenin kapsamı dışındadır.
//! - Üretim kaynaklarında haricî `.asm` / `.S` veya satır içi `asm!` / `global_asm!` kullanılmaz.
//! - `crates/av1oxide/src` altındaki `194` kaynak dosyadan `107`'si `#![forbid(unsafe_code)]`
//!   taşır; yalnız [`audit::UNSAFE_EXEMPTIONS`] içindeki `87` `x86_64`/`aarch64` SIMD dosyası
//!   dar kapsamlı `core::arch` intrinsikleri için muaf tutulur (`0` güvensiz işlev imzası, `362` blok).

#![warn(clippy::all)]

pub mod api;
pub mod audit;
pub mod bitstream;
pub mod cpu;
pub mod decode;
pub mod dsp;
pub mod entropy;
pub mod error;
pub mod frame;
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
