//! Ortak test yardımcıları (SMD-07 / TST-01).
//!
//! `tests/*.rs` dosyalarının `mod common;` ile paylaştığı yardımcılar burada
//! durur. Alt modül [`sentinel`] SMD-07'nin zorunlu kıldığı üç şeyi sağlar:
//!
//! 1. çıktı tamponunun **önüne ve arkasına sentinel** (yazma taşması tespiti),
//! 2. girdinin **sıralı/padding'li** yerleştirilmesi (over-read tespiti),
//! 3. hata veren girdinin **diske kaydedilmesi** (regresyon korpusu).
//!
//! `#![allow(dead_code)]`: bu modül birden çok entegrasyon test crate'ine
//! derlenir; her crate yalnız ihtiyacı olan yardımcıyı kullandığı için
//! kullanılmayan öğeler normaldir ve uyarı üretmemelidir.
//!
//! S15.4 ile [`cdef_fixture`] eklendi: dav1d C oracle fixture'larının okuyucusu,
//! kayıt koşum yardımcıları ve sabit tohumlu test üreteci.
//!
//! P29 ile [`on_kosul`] eklendi: ön koşulu (oracle verisi, AVX2 / x86) eksik
//! test sessizce geçmez; tek yardımcı [`on_kosul::eksik`].

#![allow(dead_code)]
#![forbid(unsafe_code)]

pub mod cdef_fixture;
pub mod ipred_slots;
pub mod mc_fixture;
pub mod on_kosul;
pub mod sentinel;
