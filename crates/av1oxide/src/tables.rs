//! Sabit tablolar (F2: üretilmiş ve kaynak izlenebilir).
//!
//! ## Sorumluluk
//!
//! AV1 normatif sabitleri: tarama sıraları (scan), kuantizasyon/işaret
//! tabloları, `qctx`, `tx` boyut eşlemeleri, palet ve CDEF yön tabloları,
//! **CDF varsayılanları** (S2a.3).
//!
//! ## Kaynak izlenebilirliği (SRC-07)
//!
//! Her tablo, `docs/source-manifest.csv` içindeki upstream kaydıyla
//! eşleştirilir (dav1d: `src/tables.c`, `src/scan.c`, `src/cdf.c` içindeki
//! başlangıç tabloları). Tablo eklerken:
//!
//! - kaynak dosya + sembol adı manifestte bulunmalı,
//! - tablo değerleri **ölçümle** doğrulanmalı (golden/differential test),
//! - sabit tablolar üretim derlemesinde `static`/`const` olarak gömülmeli,
//!   çalışma zamanında kurulmamalıdır (F2'de ölçülür).
//!
//! ## Durum
//!
//! | Alt modül | İçerik | Kaynak | Durum |
//! |---|---|---|---|
//! | `cdf` | `default_cdf` + `default_coef_cdf[4]` + `update` sayaç konumları | dav1d `src/cdf.c:77,699,3915` | ✅ S2a.3 |
//!
//! `cdf` **üretilmiştir** (`scripts/oracle/cdf_oracle.c`): değerler elle
//! yazılmaz, dav1d C yapılarından okunur ve `scripts/oracle/cdf_diff.py` ile
//! bayt bayt doğrulanır. Diğer tablolar (scan, qctx, …) F2b'de eklenecektir.

#![forbid(unsafe_code)]

pub mod cdf;
pub mod decode;
