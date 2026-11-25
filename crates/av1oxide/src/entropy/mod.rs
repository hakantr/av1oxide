//! Entropi kodlama: MSAC ve CDF (S2a.3 ✅).
//!
//! ## Sorumluluk
//!
//! - MSAC (çok sembollü aritmetik kod çözücü) çekirdeği ve durum makinesi
//!   ([`msac`]),
//! - CDF (kümülatif dağılım fonksiyonu) bağlamları, dört kuantizasyon
//!   kategorisi ve güncelleme semantiği ([`cdf`]),
//! - varsayılan tablolar [`crate::tables::cdf`] içinde **üretilmiş** olarak.
//!
//! ## Referans ve doğrulama
//!
//! Tek referans dav1d `99f3354ed0b9b4bf3c7c9347318dbcc522c94af5` **C
//! yoludur** (`src/msac.c`, `src/cdf.c`, `asm = OFF`). MSAC'ın asm yolları
//! (`src/x86/msac.asm`, `src/arm/msac.S`) referans **alınmaz**; SIMD
//! intrinsic'leri F2 sınırı gereği yasaktır.
//!
//! Bit-exact doğrulama, üretim crate'inin **dışında** derlenen oracle ile
//! yapılır:
//!
//! | Doğrulama | Oracle | Karşılaştırıcı |
//! |---|---|---|
//! | ilkel düzeyde MSAC (≥1 000 000 çağrı) | `scripts/oracle/msac_oracle.c` | `scripts/oracle/msac_diff.py` |
//! | CDF tabloları + `copy`/`update` (bayt bayt) | `scripts/oracle/cdf_oracle.c` | `scripts/oracle/cdf_diff.py` |
//! | sağlamlık (≥200 000 oturum) | — (yalnız Rust) | `scripts/oracle/run_msac_robustness.sh` |
//!
//! ## Kaynak satırları (özet)
//!
//! - sabitler: `EC_PROB_SHIFT = 6`, `EC_MIN_PROB = 4` — `src/msac.c:36-37`
//! - dolgu/veri sonu: `ctx_refill` — `src/msac.c:41-58`; AV1 spec §8.2.6
//! - yeniden ölçekleme: `ctx_norm` — `src/msac.c:85-97`
//! - ilkeller: `msac.c:60-201`; `msac.h:94-108`; AV1 spec §8.2.6, §8.3.2
//! - başlatma: `dav1d_msac_init` — `src/msac.c:204-220`
//! - CDF varsayılanları: `src/cdf.c:77` ve `:699`; `update` `:3915`,
//!   `init_static` `:4023`, `copy` `:4028`; AV1 spec §7.20
//!
//! ## İlkeler
//!
//! - **CEK-02:** skaler Rust referansı kalıcıdır; hiçbir algoritma yalnız SIMD
//!   ile teslim edilmez.
//! - **SMD-01:** taşma/kaydırma/yuvarlama davranışı sözleşme tablosunda
//!   yazılıdır ([`msac`] modül belgesi).
//! - **DSP-01:** MSAC yuvası [`crate::dsp::DspDispatch`] üzerinden
//!   seçilir; piksel başına CPU tespiti yapılmaz. (F3/F4'te skaler yuva bu
//!   modülün ilkelleriyle doldurulur; bu adımda `dsp` deposu değişmez.)
//!
//! Ölçüm notu (S0.6.2a): `msac_decode_hi_tok` SIMD'in C'den **yavaş** çıktığı
//! çekirdeklerden biridir (0,76×) — F4'te öncelik sırası ölçümle belirlenir.

#![forbid(unsafe_code)]

pub mod cdf;
pub mod cdf_layout;
pub mod msac;

pub use cdf::{CdfContext, CdfThreadContext, FrameKind, qcat_for_qidx};
pub use msac::{MsacContext, MsacIsa, inv_recenter, ulog2};
