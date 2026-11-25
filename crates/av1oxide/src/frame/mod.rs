//! Kare, düzlemler, bit derinliği ve chroma biçimi (F1.5 / S15.2).
//!
//! Bu modül BLK-03 ve BLK-04'ün tip düzeyindeki karşılığıdır:
//!
//! | Tip | Sorumluluk | İlke |
//! |---|---|---|
//! | [`BitDepth`] | 8/10/12 bit; bayt/örnek ve en büyük geçerli örnek | BLK-04 |
//! | [`ChromaFormat`] | 4:2:0 / 4:2:2 / 4:4:4 / monochrome + chroma boyut türetme | BLK-04 |
//! | [`PlaneType`] | Y/U/V düzlem kimliği | BLK-03 |
//! | [`PlaneLayout`] | `checked_*` boyut zinciri (genişlik → stride → tahsis) | BLK-05 |
//! | [`FramePlane`] | Hizalı, padding'li, sahiplenilmiş düzlem deposu | BLK-05, BLK-06 |
//! | [`Frame`] | Düzlemler + bit derinliği + chroma + metaveri | BLK-03, BLK-07 |
//!
//! **Sahiplik (BLK-02):** Düzlem depoları sahiplenilmiştir (`Vec<u8>`); kare,
//! yaşam süresi boyunca geçerlidir ve girdi paketine ödünç bağlı değildir.

#![forbid(unsafe_code)]

mod bit_depth;
pub(crate) mod butce;
mod chroma;
mod picture;
mod plane;

pub use bit_depth::BitDepth;
pub use chroma::ChromaFormat;
pub use picture::{Frame, FrameMetadata};
pub use plane::{FramePlane, PlaneLayout, PlaneType};
