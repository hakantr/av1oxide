//! AV1 bit akışı kapsayıcı ayrıştırıcıları (`ivf` ve `annexb`).
//!
//! Bu modül, [`Decoder::send_packet`](crate::Decoder::send_packet) girişine doğrudan
//! beslenebilecek AV1 paketlerini üretmek için `%100 #![forbid(unsafe_code)]` ve
//! `#![no_std]` + `alloc` uyumlu iki hafif ayrıştırıcı sunar:
//!
//! - [`ivf`]: Sıfır kopyalı dilim okuyucusu ([`IvfParser`]) ve `std` özelliği etkinken
//!   akış tabanlı [`IvfReader`].
//! - [`annexb`]: AV1 Spesifikasyonu Annex B uzunluk-ön-ekli (`leb128`) bit akışlarını
//!   standart Section 5 düşük-ek-yüklü OBU paketlerine dönüştüren [`AnnexBDemuxer`].

#![forbid(unsafe_code)]

pub mod annexb;
pub mod ivf;

pub use annexb::{AnnexBDemuxer, read_leb128};
#[cfg(feature = "std")]
pub use ivf::IvfReader;
pub use ivf::{IvfHeader, IvfPacket, IvfParser};
