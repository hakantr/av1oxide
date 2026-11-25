//! x86_64 ISA backend'i — S15.4 pilotu (AVX2, 8-bit CDEF).
//!
//! ## `unsafe` sınırı (GUV-02/GUV-04)
//!
//! Bu, üretim kaynak ağacındaki **tek** `unsafe` bölgesidir ve `audit`
//! muafiyet defterinde kayıtlıdır ([`crate::audit::UNSAFE_EXEMPTIONS`]).
//! Sınır bilinçli olarak dardır:
//!
//! - `unsafe` yalnız işaretçi alan yükleme/saklama intrinsic'lerinde (`mem`)
//!   ve belirteçli girişlerdeki tek `#[target_feature]` çağrısındadır.
//! - Her `unsafe` bloğu üç cümleli `// SAFETY:` gerekçesi taşır (ISA, sınır,
//!   hizalama/takma ad; GUV-04, ADR-0003 §2.4).
//! - ISA yetkisi kurulumda **tespit + maske** ile üretilen belirteçtir
//!   ([`crate::cpu::Avx2Token`]); belirteç olmadan AVX2 gövdesine giden yol
//!   açılamaz, böylece "yanlış tespitle illegal instruction" (DSP-01/7) yolu
//!   **tür düzeyinde** kapatılır.
//! - `-C target-cpu=native` ile genel derleme **yapılmaz**; ISA seçimi
//!   çalışma zamanındadır.
//!
//! ## Kapsam
//!
//! S15.4 kapsamı **8-bit AVX2**'dir; S15.5'te MC `put_8tap_regular` (8-bit)
//! eklendi. 16bpc girdide güvenli sarmalayıcılar
//! skaler gövdeye devreder (F3'te bit derinliği başına yuva ayrılır); SSSE3/
//! SSE4.1/AVX-512 yolları kapsam dışıdır (F4).
//!
//! Bu dosya `forbid` değil `deny` taşır: `unsafe` yalnız [`cdef`], [`mc`] ve
//! `mem` içindedir ve orada `#![allow(unsafe_code)]` ile açılır (forbid alt
//! modüllerce gevşetilemez). Bu dosyada **hiç** `unsafe` yoktur.
//!
//! ## F3b (P25) sözleşmesi
//!
//! - ISA yetkisi [`crate::cpu::Avx2Token`] ile verilir; belirteç yalnız
//!   `DspDispatch::select` kurulumunda, gerçek tespit + maske AVX2'ye izin
//!   verdiğinde üretilir. Çekirdek girişleri belirteci parametre olarak ister
//!   ya da yuva işaretçisini yalnız belirteçle verir; çağrı başına ISA
//!   denetimi yoktur.
//! - Çekirdekler ve yardımcıları güvenli `#[target_feature(enable = "avx2")]`
//!   işlevleridir. `unsafe` yalnız iki yerdedir: `mem` içindeki işaretçi
//!   alan yükleme/saklama intrinsic'leri ve her belirteçli girişteki tek
//!   `#[target_feature]` çağrısı.

#![deny(unsafe_code)]

pub mod buyutme;
pub mod buyutme16;
pub mod buyutme16_avx512;
pub mod buyutme16_ssse3;
pub mod buyutme_avx512;
pub mod buyutme_ssse3;
pub mod cdef;
pub mod cdef_avx512;
pub mod cdef_sse;
pub mod fg;
pub mod fg16;
pub mod fg16_avx512;
pub mod fg16_ssse3;
pub mod fg_avx512;
pub mod fg_ssse3;
pub mod ipred;
pub mod ipred16;
pub mod ipred16_avx512;
pub mod ipred16_ssse3;
pub mod ipred_avx512;
pub mod ipred_ssse3;
pub mod itx;
pub mod itx16_10;
pub mod itx16_10_buyuk;
pub mod itx16_12;
pub mod itx16_12_buyuk;
pub mod itx16_avx512;
pub mod itx16_buyuk_avx512;
pub mod itx16_buyuk_sse4;
pub mod itx16_sse4;
pub mod itx_avx512;
pub mod itx_buyuk;
pub mod itx_buyuk_avx512;
pub mod itx_buyuk_ssse3;
pub mod itx_ssse3;
pub mod itx_wht;
pub mod lf;
pub mod lf16;
pub mod lf16_avx512;
pub mod lf16_ssse3;
pub mod lf_avx512;
pub mod lf_ssse3;
pub mod lr;
pub mod lr16;
pub mod lr16_avx512;
pub mod lr16_ssse3;
pub mod lr_avx512;
pub mod lr_sse2;
pub mod lr_ssse3;
pub mod mc;
pub mod mc16;
pub mod mc16_avx512;
pub mod mc16_bilesik;
pub mod mc16_bilesik_avx512;
pub mod mc16_bilesik_ssse3;
pub mod mc16_karisim;
pub mod mc16_karisim_avx512;
pub mod mc16_karisim_ssse3;
pub mod mc16_kenar;
pub mod mc16_kenar_ssse3;
pub mod mc16_olcekli;
pub mod mc16_olcekli_ssse3;
pub mod mc16_ssse3;
pub mod mc16_warp;
pub mod mc16_warp_avx512;
pub mod mc16_warp_ssse3;
pub mod mc_avx512;
pub mod mc_bilesik;
pub mod mc_bilesik_avx512;
pub mod mc_bilesik_ssse3;
pub mod mc_karisim;
pub mod mc_karisim_avx512;
pub mod mc_karisim_ssse3;
pub mod mc_kenar;
pub mod mc_kenar_ssse3;
pub mod mc_olcekli;
pub mod mc_olcekli_ssse3;
pub mod mc_ssse3;
pub mod mc_warp;
pub mod mc_warp_avx512;
pub mod mc_warp_sse41;
pub mod mc_warp_ssse3;
mod mem;
pub mod msac;
pub mod msac_avx2;
pub mod pal;
pub mod pal_avx512;
pub mod pal_ssse3;
pub mod refmvs;
pub mod refmvs_avx512;
pub mod refmvs_sse2;
pub mod refmvs_sse41;
pub mod refmvs_ssse3;
#[cfg(test)]
mod sozluk;
