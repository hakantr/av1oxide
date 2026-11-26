//! Genel API yüzeyi (BLK-01).
//!
//! Mimari tasarım bu dizini "Config, Packet, Frame, Error" olarak tanımlar. Tipler
//! sorumluluklarına göre ayrı modüllerde yaşar, burada **tek yüzeyde** yeniden
//! dışa aktarılır:
//!
//! - [`Config`] — çözücü yapılandırması (`api/config.rs`, BLK-01/BLK-07)
//! - [`Packet`] — girdi paketi ve sahiplik sözleşmesi (`api/packet.rs`, BLK-02)
//! - [`Decoder`] — çözücü ve durum makinesi (`api/decoder.rs`, BLK-01)
//! - [`Frame`] — çıktı karesi (`crate::frame`, BLK-03)
//! - [`Error`] / [`Result`] — hata modeli (`crate::error`, BLK-05/BLK-07)
//!
//! Kullanım sırası (F2'de doldurulacak hat):
//!
//! ```text
//! Decoder::new(Config::default())?
//!   → send_packet(&Packet::new(&obu_bytes))?   // F2a: OBU + başlıklar
//!   → receive_frame()?  (Ok(Some(frame)) | Ok(None))  // F2b'de kare üretir
//!   → flush() / reset()
//! ```

#![forbid(unsafe_code)]

mod config;
mod decoder;
mod packet;

pub use config::{Config, MAX_FRAME_DELAY, MAX_THREADS};
#[cfg(feature = "std")]
pub(crate) use decoder::OutputCapiMeta;
pub use decoder::{Decoder, DecoderState};
pub use packet::Packet;

// Mimari yapı: `api/` yüzeyi Frame ve Error'ı da sunar; tanımları kendi
// modüllerinde kalır (tek kaynak ilkesi).
pub use crate::error::{Error, Result};
pub use crate::frame::{BitDepth, ChromaFormat, Frame, FramePlane, PlaneType};
