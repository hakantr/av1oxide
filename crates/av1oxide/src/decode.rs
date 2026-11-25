//! Çözme hattı: kare/tile, referanslar, yeniden kurulum (F1.5 iskeleti: **boş**, F2b–F2d).
//!
//! ## Sorumluluk
//!
//! - kare ve tile düzenekleri (frame header → tile group → tile decode),
//! - referans kare yönetimi ve yaşam süresi (BLK-02, BLK-08),
//! - yeniden kurulum (reconstruction), intra/inter tahmin, dönüşüm,
//!   döngü filtreleri, loop restoration, film grain (skaler),
//! - çıktı sırası, delayed/**show-existing** kare ve flush/reset durum
//!   makineleri (BLK-01, BLK-08).
//!
//! ## Upstream karşılığı
//!
//! dav1d `src/decode.c`, `src/decode.h`, `src/refmvs.c` (salt-okunur referans;
//! kayıtlar `docs/source-manifest.csv` içinde). Dizin düzeni bire bir taklit
//! **edilmez** (CEK-10); davranış ve hız hedeflenir.
//!
//! F1.5'te bu modül **bilinçli olarak boştu**; F2a / S2a.2'de başlık düzeyi
//! referans durumu ([`refs`]) eklendi. Piksel düzeyi referans yönetimi,
//! yeniden kurulum ve kare üretimi F2b–F2d'nin işidir.

#![forbid(unsafe_code)]

pub mod cdef;
pub mod coef;
pub mod edge_tree;
pub mod filmgrain;
pub mod intra;
pub mod ipred_prepare;
pub mod loopfilter;
pub mod lr;
pub mod probe;
pub mod refmvs;
pub mod refmvs_tmvs;
pub mod refs;
pub mod superres;
pub mod warpv;
pub mod wedge;

pub use refs::HeaderReferenceState;
