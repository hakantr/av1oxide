//! Denetim (audit) defteri: `unsafe` muafiyetleri ve doğrulama izleri (F1.5 / S15.2).
//!
//! ## Neden gerekli?
//!
//! - **GUV-02:** Mümkün olan modüllerde `#![forbid(unsafe_code)]`; SIMD/FFI ve
//!   özel tampon modülleri için **ayrı `unsafe` sınırı** tanımlanır. Hangi
//!   modülün muaf olduğu ve **neden** olduğu yazılı olmadan hiçbir modül muaf
//!   sayılmaz.
//! - **GUV-04:** `unsafe` sınırı küçük, tek yönlü ve gerekçeli olur; her
//!   `unsafe` bloğu `// SAFETY:` yorumu taşır.
//! - **GUV-08 (dürüstlük):** "kendi kodumuzda `unsafe` yok" demek, "binary'de
//!   hiç `unsafe` yok" demek **değildir** (`std` içindeki `unsafe` kullanımı
//!   bu defterin kapsamı dışındadır, ama raporda ayrıca belirtilir).
//!
//! ## F1.5 durumu
//!
//! **S15.2**'de kayıt boştu (hiç `unsafe` yoktu). **S15.4** ile x86_64 CDEF AVX2
//! pilotu eklendi ve deftere **tek** kayıt girdi; **S15.5** MC'yi ekledi.
//! **F3b (P25):** Rust 1.99.1'de işaretçisiz intrinsic'ler güvenli
//! `#[target_feature]` bağlamında güvenlidir (target_feature 1.1); `unsafe`
//! yalnız işaretçi alan yükleme/saklama intrinsic'lerinde (`dsp/x86/mem.rs`)
//! ve belirteçli girişlerdeki tek `#[target_feature]` çağrısında
//! (`dsp/x86/cdef.rs`, `dsp/x86/mc.rs`) kalır. Kayıt gerekçe + faz taşır
//! (GUV-03). Testler, `src/` altındaki her dosyanın bir `unsafe` politikası
//! (yasak ya da kayıtlı muafiyet) bildirdiğini ve her bloğun üç cümleli
//! `// SAFETY:` şablonunu (ADR-0003 §2.4) taşıdığını **ölçer**.

#![forbid(unsafe_code)]

/// Envanter belgesi ile kaynak ağacını senkron tutan denetim (S15.6).
#[cfg(test)]
mod envanter;

/// `unsafe` kullanımına izin verilen modülün kaydı.
///
/// Kayıt, "izin var" demekle kalmaz; **gerekçeyi** ve **hangi fazda** verildiğini
/// de taşır (GUV-03: gerekçe "daha hızlı olabilir" olamaz — profil sonucu veya
/// donanım zorunluluğu olmalıdır).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnsafeExemption {
    /// Crate'e göre modül yolu (ör. `dsp/x86/mod.rs`).
    pub module: &'static str,
    /// Muafiyetin gerekçesi (profil sonucu veya donanım zorunluluğu).
    pub reason: &'static str,
    /// Muafiyetin verildiği adım/faz (ör. `S15.6`, `F4`).
    pub phase: &'static str,
}

/// `unsafe` muafiyet defteri.
///
/// **S15.4:** x86_64 CDEF AVX2 pilotu. **S15.5:** aynı sınır içinde MC
/// (`put_8tap_regular`, 8-bit AVX2) eklendi. **F3b (P25):** gerekçeler Rust
/// 1.99.1'e göre yenilendi ve işaretçili yükleme/saklama `dsp/x86/mem.rs`'te
/// toplandı. **F3d (P27):** yalnız test derlemesine giren x86 sözlük lane
/// testleri (`dsp/x86/sozluk.rs`, `cfg(test)`) eklendi. Kapsam bilinçli olarak
/// dardır: `unsafe` yalnız bu dört dosyadadır;
/// kardeş modüller (`dsp/x86/mod.rs`, `dsp/mod.rs`, `dsp/cdef.rs`,
/// `dsp/mc.rs`, `dsp/scalar/*`) ve diğer tüm modüller
/// `#![forbid(unsafe_code)]` (ya da `deny`) taşır.
pub const UNSAFE_EXEMPTIONS: &[UnsafeExemption] = &[
    UnsafeExemption {
        module: "capi/ffi.rs",
        reason: "libdav1d (7.0.0) C-ABI uyumluluk katmanı FFI sınır dosyası: 19 `extern \"C\"` giriş noktasında \
                 C çağıranı tarafından verilen ham işaretçilerin null/boyut doğrulaması sonrası okunması/yazılması ve \
                 `Arc::into_raw`/`Arc::from_raw` tabanlı `Dav1dRef` referans sayımı; tüm FFI girişleri `catch_unwind` \
                 ile sarılıdır ve kardeş modüller (`capi/mod.rs`, `capi/types.rs`, `capi/convert.rs`) `%100 #![forbid(unsafe_code)]` \
                 altındadır.",
        phase: "v0.2.0",
    },
    UnsafeExemption {
        module: "dsp/arm/mem.rs",
        reason: "ARM64 NEON (`aarch64`) sabit boyutlu dizi yükleme/saklama yardımcıları (`vld1*`/`vst1*` üzerinde `&[T; N]` / `&mut [T; N]`) \
                 ve `NeonToken`-kapılı `fn(...)` yuva giriş köprüleri (`cdef`, `cdef16`, `pal`, `refmvs` çekirdekleri `%100 #![forbid(unsafe_code)]` \
                 ile derlenir; `dsp/arm/` altındaki tek `unsafe` sınır dosyasıdır, P51 / F5a).",
        phase: "P51",
    },
    UnsafeExemption {
        module: "dsp/arm/giris.rs",
        reason: "ARM64 NEON / DOTPROD (`aarch64`) yuva giriş köprüleri ve sabit boyutlu dizi saklama yardımcıları (F5b–F5d): \
                 her köprü güvenli kodla O(1) sözleşmeyi denetler (dışı skalere devreder), sonra `%100 #![forbid(unsafe_code)]` \
                 `#[target_feature]` gövdesini çağırır; adresler yalnız `NeonToken` / `DotprodToken` ile verilir (P52+).",
        phase: "P52",
    },
    UnsafeExemption {
        module: "dsp/x86/buyutme.rs",
        reason: "AVX2 superres büyütme girişi (buyut_giris): güvenli #[target_feature(enable = \"avx2\")] \
                 gövdesini AVX2'siz bağlamdan çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız Avx2Token ile verilen \
                 girişlerdedir, gövde unsafe içermez (P33). P36: AVX512ICL girişleri (buyut_giris_icl) de bu modüldedir (paylaşılan AVX2 gövdelerinin iç bağlantısı için); \
                 tek çağrı yalnız Avx512IclToken ile verilen girişlerdedir.",
        phase: "P33/P36",
    },
    UnsafeExemption {
        module: "dsp/x86/buyutme_ssse3.rs",
        reason: "SSSE3 superres büyütme girişi (buyut_giris): \
                 güvenli #[target_feature(enable = \"ssse3\")] gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.99.1'de unsafe \
                 gerektirir; tek çağrı yalnız Ssse3Token ile verilen girişlerdedir, gövde unsafe içermez (P35).",
        phase: "P35",
    },
    UnsafeExemption {
        module: "dsp/x86/buyutme16.rs",
        reason: "AVX2 MC 16 bpc girişleri (buyut16_giris): güvenli #[target_feature(enable = \"avx2\")] \
                 gövdesini AVX2'siz bağlamdan çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız Avx2Token \
                 ile verilen girişlerdedir, gövdeler unsafe içermez (P34). P36: AVX512ICL girişleri (buyut16_giris_icl) de bu modüldedir (paylaşılan AVX2 gövdelerinin iç bağlantısı için); \
                 tek çağrı yalnız Avx512IclToken ile verilen girişlerdedir.",
        phase: "P34/P36",
    },
    UnsafeExemption {
        module: "dsp/x86/buyutme16_ssse3.rs",
        reason: "SSSE3 16 bpc superres büyütme girişi (buyut16_giris): \
                 güvenli #[target_feature(enable = \"ssse3\")] gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.99.1'de unsafe \
                 gerektirir; tek çağrı yalnız Ssse3Token ile verilen girişlerdedir, gövde unsafe içermez (P35).",
        phase: "P35",
    },
    UnsafeExemption {
        module: "dsp/x86/cdef.rs",
        reason: "AVX2 CDEF girişleri (find_dir_entry, filter_entry): \
                 güvenli #[target_feature(enable = \"avx2\")] çekirdeğini AVX2'siz bağlamdan çağırmak \
                 Rust 1.99.1'de unsafe gerektirir (target_feature 1.1); bu tek çağrı yalnız \
                 Avx2Token ile verilen girişlerdedir, çekirdek gövdeleri unsafe içermez \
                 (S15.4 pilotu, F3b).",
        phase: "S15.4/P25",
    },
    UnsafeExemption {
        module: "dsp/x86/cdef_avx512.rs",
        reason: "AVX512ICL CDEF girişleri (filter_avx512_entry, filter16_avx512_entry): \
                 güvenli #[target_feature(enable = \"avx512f,avx512bw,avx512vl\")] çekirdeğini \
                 o özellikler olmadan çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı \
                 yalnız Avx512IclToken ile verilen girişlerdedir, gövdeler unsafe içermez \
                 (F4-CDEF c).",
        phase: "P30",
    },
    UnsafeExemption {
        module: "dsp/x86/cdef_sse.rs",
        reason: "SSSE3 ve SSE4.1 CDEF girişleri (find_dir/find_dir16/filter/filter16 \
                 *_ssse3_entry, *_sse41_entry): güvenli #[target_feature(enable = \"ssse3\" | \
                 \"sse4.1\")] çekirdeğini o özellik olmadan çağırmak Rust 1.99.1'de unsafe \
                 gerektirir; tek çağrı yalnız Ssse3Token / Sse41Token ile verilen \
                 girişlerdedir, gövdeler unsafe içermez (F4-CDEF c).",
        phase: "P30",
    },
    UnsafeExemption {
        module: "dsp/x86/fg.rs",
        reason: "AVX2 film grain girişleri (grain_y_giris, grain_uv_*_giris, fgy_giris, fguv_*_giris): \
                 güvenli #[target_feature(enable = \"avx2\")] gövdelerini AVX2'siz bağlamdan çağırmak Rust 1.99.1'de \
                 unsafe gerektirir; tek çağrı yalnız Avx2Token ile verilen girişlerdedir, gövdeler unsafe içermez (P39).",
        phase: "P39",
    },
    UnsafeExemption {
        module: "dsp/x86/fg16.rs",
        reason: "AVX2 film grain 16 bpc (10/12 bit) girişleri (grain_y_giris, grain_uv_*_giris, fgy_giris, fguv_*_giris): \
                 güvenli #[target_feature(enable = \"avx2\")] gövdelerini AVX2'siz bağlamdan çağırmak Rust 1.99.1'de \
                 unsafe gerektirir; tek çağrı yalnız Avx2Token ile verilen girişlerdedir, gövdeler unsafe içermez (P42).",
        phase: "P42",
    },
    UnsafeExemption {
        module: "dsp/x86/fg16_avx512.rs",
        reason: "AVX512ICL film grain 16 bpc (10/12 bit) girişleri (fgy_giris, fguv_*_giris): \
                 güvenli #[target_feature(enable = \"avx512f,avx512bw,avx512vl,avx512vbmi\")] gövdelerini AVX-512'siz bağlamdan çağırmak Rust 1.99.1'de \
                 unsafe gerektirir; tek çağrı yalnız Avx512IclToken ile verilen girişlerdedir, gövdeler unsafe içermez (P42).",
        phase: "P42",
    },
    UnsafeExemption {
        module: "dsp/x86/fg16_ssse3.rs",
        reason: "SSSE3 film grain 16 bpc (10/12 bit) girişleri (grain_y_giris, grain_uv_*_giris, fgy_giris, fguv_*_giris): \
                 güvenli #[target_feature(enable = \"ssse3\")] gövdelerini SSSE3'süz bağlamdan çağırmak Rust 1.99.1'de \
                 unsafe gerektirir; tek çağrı yalnız Ssse3Token ile verilen girişlerdedir, gövdeler unsafe içermez (P42).",
        phase: "P42",
    },
    UnsafeExemption {
        module: "dsp/x86/fg_avx512.rs",
        reason: "AVX512ICL film grain 8 bpc girişleri (fgy_giris, fguv_*_giris): \
                 güvenli #[target_feature(enable = \"avx512f,avx512bw,avx512vl,avx512vbmi\")] gövdelerini AVX-512'siz bağlamdan çağırmak Rust 1.99.1'de \
                 unsafe gerektirir; tek çağrı yalnız Avx512IclToken ile verilen girişlerdedir, gövdeler unsafe içermez (P41).",
        phase: "P41",
    },
    UnsafeExemption {
        module: "dsp/x86/fg_ssse3.rs",
        reason: "SSSE3 film grain girişleri (grain_y_giris, grain_uv_*_giris, fgy_giris, fguv_*_giris): \
                 güvenli #[target_feature(enable = \"ssse3\")] gövdelerini SSSE3'süz bağlamdan çağırmak Rust 1.99.1'de \
                 unsafe gerektirir; tek çağrı yalnız Ssse3Token ile verilen girişlerdedir, gövdeler unsafe içermez (P41).",
        phase: "P41",
    },
    UnsafeExemption {
        module: "dsp/x86/ipred.rs",
        reason: "AVX2 8 bpc intra tahmin girişleri (intra_pred[14], cfl_ac[3], cfl_pred[4], pal_pred; 22 kayıt): \
                 güvenli #[target_feature(enable = \"avx2\")] gövdelerini AVX2'siz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Avx2Token ile verilen girişlerdedir, gövdeler unsafe içermez (P45).",
        phase: "P45",
    },
    UnsafeExemption {
        module: "dsp/x86/ipred16.rs",
        reason: "AVX2 16 bpc (10/12 bit ortak) intra tahmin girişleri (intra_pred[14], cfl_ac[3], cfl_pred[4], pal_pred; 22 kayıt / 44 yuva): \
                 güvenli #[target_feature(enable = \"avx2\")] gövdelerini AVX2'siz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Avx2Token ile verilen girişlerdedir, gövdeler unsafe içermez (P45).",
        phase: "P45",
    },
    UnsafeExemption {
        module: "dsp/x86/ipred16_avx512.rs",
        reason: "AVX512ICL 16 bpc (10/12 bit ortak) intra tahmin girişleri (Paeth, Smooth, SmoothH, SmoothV, Z1, Z2, Z3, Filter + pal_pred; 9 kayıt / 18 yuva): \
                 güvenli #[target_feature(enable = \"avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni\")] gövdelerini AVX-512'siz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Avx512IclToken ile verilen girişlerdedir, gövdeler unsafe içermez (P45).",
        phase: "P45",
    },
    UnsafeExemption {
        module: "dsp/x86/ipred16_ssse3.rs",
        reason: "SSSE3 16 bpc (10/12 bit ortak) intra tahmin girişleri (intra_pred[14], cfl_ac[3], cfl_pred[4], pal_pred; 22 kayıt / 44 yuva): \
                 güvenli #[target_feature(enable = \"ssse3\")] gövdelerini SSSE3'süz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Ssse3Token ile verilen girişlerdedir, gövdeler unsafe içermez (P45).",
        phase: "P45",
    },
    UnsafeExemption {
        module: "dsp/x86/ipred_avx512.rs",
        reason: "AVX512ICL 8 bpc intra tahmin girişleri (intra_pred[14] + pal_pred; 15 kayıt): \
                 güvenli #[target_feature(enable = \"avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni\")] gövdelerini AVX-512'siz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Avx512IclToken ile verilen girişlerdedir, gövdeler unsafe içermez (P45).",
        phase: "P45",
    },
    UnsafeExemption {
        module: "dsp/x86/ipred_ssse3.rs",
        reason: "SSSE3 8 bpc intra tahmin girişleri (intra_pred[14], cfl_ac[3], cfl_pred[4], pal_pred; 22 kayıt): \
                 güvenli #[target_feature(enable = \"ssse3\")] gövdelerini SSSE3'süz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Ssse3Token ile verilen girişlerdedir, gövdeler unsafe içermez (P45).",
        phase: "P45",
    },
    UnsafeExemption {
        module: "dsp/x86/itx.rs",
        reason: "AVX2 8 bpc w, h <= 16 WHT-dışı ters dönüşüm girişleri (itx8_avx2_giris): \
                 güvenli #[target_feature(enable = \"avx2\")] gövdesini AVX2'siz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Avx2Token ile verilen girişlerdedir, gövdeler unsafe içermez (P43).",
        phase: "P43",
    },
    UnsafeExemption {
        module: "dsp/x86/itx16_10.rs",
        reason: "AVX2 10 bpc WHT-dışı (w, h <= 16, 140 kayıt) ters dönüşüm girişi (itx10_avx2_giris): \
                 güvenli #[target_feature(enable = \"avx2\")] gövdesini AVX2'siz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Avx2Token ile verilen girişlerdedir, gövdeler unsafe içermez (P44).",
        phase: "P44",
    },
    UnsafeExemption {
        module: "dsp/x86/itx16_10_buyuk.rs",
        reason: "AVX2 10 bpc büyük boyutlu (32/64 nokta, 15 kayıt) ters dönüşüm girişi (itx10_buyuk_avx2_giris): \
                 güvenli #[target_feature(enable = \"avx2\")] gövdesini AVX2'siz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Avx2Token ile verilen girişlerdedir, gövdeler unsafe içermez (P44).",
        phase: "P44",
    },
    UnsafeExemption {
        module: "dsp/x86/itx16_12.rs",
        reason: "AVX2 12 bpc WHT-dışı (w, h <= 16, 140 kayıt) ters dönüşüm girişi (itx12_avx2_giris): \
                 güvenli #[target_feature(enable = \"avx2\")] gövdesini AVX2'siz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Avx2Token ile verilen girişlerdedir, gövdeler unsafe içermez (P44).",
        phase: "P44",
    },
    UnsafeExemption {
        module: "dsp/x86/itx16_12_buyuk.rs",
        reason: "AVX2 12 bpc büyük boyutlu (32 nokta, 7 kayıt) ters dönüşüm girişi (itx12_buyuk_avx2_giris): \
                 güvenli #[target_feature(enable = \"avx2\")] gövdesini AVX2'siz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Avx2Token ile verilen girişlerdedir, gövdeler unsafe içermez (P44).",
        phase: "P44",
    },
    UnsafeExemption {
        module: "dsp/x86/itx16_avx512.rs",
        reason: "AVX512ICL 10 bpc WHT-dışı (8 <= w, h <= 16, 60 kayıt) ters dönüşüm girişi (itx10_avx512_giris): \
                 güvenli #[target_feature(enable = \"avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni\")] gövdesini AVX-512'siz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Avx512IclToken ile verilen girişlerdedir, gövdeler unsafe içermez (P44).",
        phase: "P44",
    },
    UnsafeExemption {
        module: "dsp/x86/itx16_buyuk_avx512.rs",
        reason: "AVX512ICL 10 bpc büyük boyutlu (32/64 nokta, 15 kayıt) ters dönüşüm girişi (itx10_buyuk_avx512_giris): \
                 güvenli #[target_feature(enable = \"avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni\")] gövdesini AVX-512'siz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Avx512IclToken ile verilen girişlerdedir, gövdeler unsafe içermez (P44).",
        phase: "P44",
    },
    UnsafeExemption {
        module: "dsp/x86/itx16_buyuk_sse4.rs",
        reason: "SSE4.1 10 bpc büyük boyutlu (32/64 nokta, 15 kayıt) ters dönüşüm girişi (itx10_buyuk_sse4_giris): \
                 güvenli #[target_feature(enable = \"sse4.1\")] gövdesini SSE4.1'siz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Sse41Token ile verilen girişlerdedir, gövdeler unsafe içermez (P43).",
        phase: "P43",
    },
    UnsafeExemption {
        module: "dsp/x86/itx16_sse4.rs",
        reason: "SSE4.1 10 bpc WHT-dışı (w, h <= 16, 140 kayıt) ters dönüşüm girişi (itx10_sse4_giris): \
                 güvenli #[target_feature(enable = \"sse4.1\")] gövdesini SSE4.1'siz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Sse41Token ile verilen girişlerdedir, gövdeler unsafe içermez (P43).",
        phase: "P43",
    },
    UnsafeExemption {
        module: "dsp/x86/itx_avx512.rs",
        reason: "AVX512ICL 8 bpc w, h <= 16 WHT-dışı ters dönüşüm girişleri (itx8_avx512_giris, 140 kayıt): \
                 güvenli #[target_feature(enable = \"avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni\")] gövdesini AVX-512'siz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Avx512IclToken ile verilen girişlerdedir, gövdeler unsafe içermez (P43).",
        phase: "P43",
    },
    UnsafeExemption {
        module: "dsp/x86/itx_buyuk.rs",
        reason: "AVX2 8 bpc max(w, h) in {32, 64} büyük ters dönüşüm girişleri (itx8_buyuk_avx2_giris): \
                 güvenli #[target_feature(enable = \"avx2\")] gövdesini AVX2'siz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Avx2Token ile verilen girişlerdedir, gövdeler unsafe içermez (P43).",
        phase: "P43",
    },
    UnsafeExemption {
        module: "dsp/x86/itx_buyuk_avx512.rs",
        reason: "AVX512ICL 8 bpc max(w, h) in {32, 64} büyük ters dönüşüm girişleri (itx8_buyuk_avx512_giris, 15 kayıt): \
                 güvenli #[target_feature(enable = \"avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni\")] gövdesini AVX-512'siz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Avx512IclToken ile verilen girişlerdedir, gövdeler unsafe içermez (P43).",
        phase: "P43",
    },
    UnsafeExemption {
        module: "dsp/x86/itx_buyuk_ssse3.rs",
        reason: "SSSE3 8 bpc büyük boyutlu (32/64 nokta, 15 kayıt) ters dönüşüm girişi (itx8_buyuk_ssse3_giris): \
                 güvenli #[target_feature(enable = \"ssse3\")] gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Ssse3Token ile verilen girişlerdedir, gövdeler unsafe içermez (P43).",
        phase: "P43",
    },
    UnsafeExemption {
        module: "dsp/x86/itx_ssse3.rs",
        reason: "SSSE3 8 bpc WHT-dışı (w, h <= 16, 140 kayıt) ters dönüşüm girişi (itx8_ssse3_giris): \
                 güvenli #[target_feature(enable = \"ssse3\")] gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.100.0'da \
                 unsafe gerektirir; tek çağrı yalnız Ssse3Token ile verilen girişlerdedir, gövdeler unsafe içermez (P43).",
        phase: "P43",
    },
    UnsafeExemption {
        module: "dsp/x86/itx_wht.rs",
        reason: "SSE2 ve AVX2 kayıpsız TX_4X4 × WHT_WHT ters dönüşüm girişleri (wht8_sse2_giris, wht16_sse2_giris, wht8_avx2_giris, wht16_avx2_giris): \
                 güvenli #[target_feature(enable = \"sse2\" | \"avx2\")] gövdelerini o özellikler olmadan çağırmak Rust 1.99.1'de \
                 unsafe gerektirir; tek çağrı yalnız Sse2Token / Avx2Token ile verilen girişlerdedir, gövdeler unsafe içermez (P43).",
        phase: "P43",
    },
    UnsafeExemption {
        module: "dsp/x86/lf.rs",
        reason: "AVX2 8 bpc loop_filter_sb girişi (lpf_sb_8bpc_avx2, 4 kayıt): güvenli #[target_feature(enable = \"avx2\")] \
                 gövdesini AVX2'siz bağlamdan çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Avx2Token \
                 ile verilen girişlerdedir, gövde unsafe içermez (P46).",
        phase: "P46",
    },
    UnsafeExemption {
        module: "dsp/x86/lf16.rs",
        reason: "AVX2 16 bpc loop_filter_sb girişleri (lpf_{h,v}_sb_{y,uv}_16bpc_avx2, 4 kayıt / 8 yuva @10+@12): güvenli #[target_feature(enable = \"avx2\")] \
                 gövdelerini AVX2'siz bağlamdan çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Avx2Token \
                 ile verilen girişlerdedir, gövdeler unsafe içermez (P46).",
        phase: "P46",
    },
    UnsafeExemption {
        module: "dsp/x86/lf16_avx512.rs",
        reason: "AVX512ICL 16 bpc loop_filter_sb girişleri (lpf_{h,v}_sb_{y,uv}_16bpc_avx512icl, 4 kayıt / 8 yuva @10+@12): güvenli #[target_feature(enable = \"avx512f,avx512bw,avx512vl,avx512vbmi\")] \
                 gövdelerini AVX-512'siz bağlamdan çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Avx512IclToken \
                 ile verilen girişlerdedir, gövdeler unsafe içermez (P46).",
        phase: "P46",
    },
    UnsafeExemption {
        module: "dsp/x86/lf16_ssse3.rs",
        reason: "SSSE3 16 bpc loop_filter_sb girişleri (lpf_{h,v}_sb_{y,uv}_16bpc_ssse3, 4 kayıt / 8 yuva @10+@12): güvenli #[target_feature(enable = \"ssse3\")] \
                 gövdelerini SSSE3'süz bağlamdan çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Ssse3Token \
                 ile verilen girişlerdedir, gövdeler unsafe içermez (P46).",
        phase: "P46",
    },
    UnsafeExemption {
        module: "dsp/x86/lf_avx512.rs",
        reason: "AVX512ICL 8 bpc loop_filter_sb girişi (lpf_sb_8bpc_avx512icl, 4 kayıt): güvenli #[target_feature(enable = \"avx512f,avx512bw,avx512vl,avx512vbmi\")] \
                 gövdesini AVX-512'siz bağlamdan çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Avx512IclToken \
                 ile verilen girişlerdedir, gövde unsafe içermez (P46).",
        phase: "P46",
    },
    UnsafeExemption {
        module: "dsp/x86/lf_ssse3.rs",
        reason: "SSSE3 8 bpc loop_filter_sb girişi (lpf_sb_8bpc_ssse3, 4 kayıt): güvenli #[target_feature(enable = \"ssse3\")] \
                 gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Ssse3Token \
                 ile verilen girişlerdedir, gövde unsafe içermez (P46).",
        phase: "P46",
    },
    UnsafeExemption {
        module: "dsp/x86/lr.rs",
        reason: "AVX2 8 bpc loop restoration girişleri (wiener_8bpc_avx2, sgr_8bpc_avx2, 5 kayıt): güvenli #[target_feature(enable = \"avx2\")] \
                 gövdelerini AVX2'siz bağlamdan çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Avx2Token \
                 ile verilen girişlerdedir, gövdeler unsafe içermez (P46).",
        phase: "P46",
    },
    UnsafeExemption {
        module: "dsp/x86/lr16.rs",
        reason: "AVX2 16 bpc loop restoration girişleri (wiener_16bpc_avx2, sgr_*_16bpc_avx2, 5 kayıt / 7 yuva): güvenli #[target_feature(enable = \"avx2\")] \
                 gövdelerini AVX2'siz bağlamdan çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Avx2Token \
                 ile verilen girişlerdedir, gövdeler unsafe içermez (P46).",
        phase: "P46",
    },
    UnsafeExemption {
        module: "dsp/x86/lr16_avx512.rs",
        reason: "AVX512ICL 16 bpc loop restoration girişleri (wiener_16bpc_avx512icl, sgr_*_16bpc_avx512icl, 5 kayıt / 7 yuva): güvenli #[target_feature(enable = \"avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni\")] \
                 gövdelerini AVX-512'siz bağlamdan çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Avx512IclToken \
                 ile verilen girişlerdedir, gövdeler unsafe içermez (P46).",
        phase: "P46",
    },
    UnsafeExemption {
        module: "dsp/x86/lr16_ssse3.rs",
        reason: "SSSE3 16 bpc loop restoration girişleri (wiener_16bpc_ssse3, sgr_*_16bpc_ssse3, 5 kayıt / 7 yuva): güvenli #[target_feature(enable = \"ssse3\")] \
                 gövdelerini SSSE3'süz bağlamdan çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Ssse3Token \
                 ile verilen girişlerdedir, gövdeler unsafe içermez (P46).",
        phase: "P46",
    },
    UnsafeExemption {
        module: "dsp/x86/lr_avx512.rs",
        reason: "AVX512ICL 8 bpc loop restoration girişleri (wiener_filter7_8bpc_avx512icl, sgr_8bpc_avx512, 4 kayıt / 5 yuva): güvenli #[target_feature(enable = \"avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni\")] \
                 gövdelerini AVX-512'siz bağlamdan çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Avx512IclToken \
                 ile verilen girişlerdedir, gövdeler unsafe içermez (P46).",
        phase: "P46",
    },
    UnsafeExemption {
        module: "dsp/x86/lr_sse2.rs",
        reason: "SSE2 8 bpc loop restoration Wiener girişi (wiener_8bpc_sse2, 2 kayıt): güvenli #[target_feature(enable = \"sse2\")] \
                 gövdesini SSE2'siz bağlamdan çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Sse2Token \
                 ile verilen girişlerdedir, gövde unsafe içermez (P46).",
        phase: "P46",
    },
    UnsafeExemption {
        module: "dsp/x86/lr_ssse3.rs",
        reason: "SSSE3 8 bpc loop restoration girişleri (wiener_8bpc_ssse3, sgr_8bpc_ssse3, 5 kayıt): güvenli #[target_feature(enable = \"ssse3\")] \
                 gövdelerini SSSE3'süz bağlamdan çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Ssse3Token \
                 ile verilen girişlerdedir, gövdeler unsafe içermez (P46).",
        phase: "P46",
    },
    UnsafeExemption {
        module: "dsp/x86/mc.rs",
        reason: "AVX2 MC girişleri (put_giris, prep_giris, put_8tap_regular_prepared): \
                 güvenli #[target_feature(enable = \"avx2\")] çekirdeğini AVX2'siz bağlamdan \
                 çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız Avx2Token ile \
                 verilen girişlerdedir, gövdeler unsafe içermez (S15.5 pilotu, F3b; P32 \
                 kopyasız 8 bpc put/prep). P36: AVX512ICL girişleri (put_giris_icl, prep_giris_icl) de bu modüldedir (paylaşılan AVX2 gövdelerinin iç bağlantısı için); \
                 tek çağrı yalnız Avx512IclToken ile verilen girişlerdedir.",
        phase: "S15.5/P25/P32/P36",
    },
    UnsafeExemption {
        module: "dsp/x86/mc16.rs",
        reason: "AVX2 MC 16 bpc girişleri (put16_giris, prep16_giris): güvenli #[target_feature(enable = \"avx2\")] \
                 gövdesini AVX2'siz bağlamdan çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız Avx2Token ile verilen \
                 girişlerdedir, gövdeler unsafe içermez (P34). P36: AVX512ICL girişleri (put16_giris_icl, prep16_giris_icl) de bu modüldedir (paylaşılan AVX2 gövdelerinin iç bağlantısı için); \
                 tek çağrı yalnız Avx512IclToken ile verilen girişlerdedir.",
        phase: "P34/P36",
    },
    UnsafeExemption {
        module: "dsp/x86/mc16_bilesik.rs",
        reason: "AVX2 MC 16 bpc bileşik girişleri (avg16_giris, w_avg16_giris, mask16_giris, w_mask16_giris): \
                 güvenli #[target_feature(enable = \"avx2\")] gövdesini AVX2'siz bağlamdan çağırmak Rust 1.99.1'de unsafe \
                 gerektirir; tek çağrı yalnız Avx2Token ile verilen girişlerdedir, gövdeler unsafe içermez (P34). P36: AVX512ICL girişleri (avg16_giris_icl, w_avg16_giris_icl, mask16_giris_icl, w_mask16_giris_icl) de bu modüldedir (paylaşılan AVX2 gövdelerinin iç bağlantısı için); \
                 tek çağrı yalnız Avx512IclToken ile verilen girişlerdedir.",
        phase: "P34/P36",
    },
    UnsafeExemption {
        module: "dsp/x86/mc16_bilesik_ssse3.rs",
        reason: "SSSE3 MC 16 bpc bileşik girişleri (avg16_giris, w_avg16_giris, mask16_giris, w_mask16_giris): \
                 güvenli #[target_feature(enable = \"ssse3\")] gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.99.1'de unsafe \
                 gerektirir; tek çağrı yalnız Ssse3Token ile verilen girişlerdedir, gövdeler unsafe içermez (P35).",
        phase: "P35",
    },
    UnsafeExemption {
        module: "dsp/x86/mc16_karisim.rs",
        reason: "AVX2 MC 16 bpc karışım girişleri (blend16_giris, blend_v16_giris, blend_h16_giris): \
                 güvenli #[target_feature(enable = \"avx2\")] gövdesini AVX2'siz bağlamdan çağırmak Rust 1.99.1'de unsafe \
                 gerektirir; tek çağrı yalnız Avx2Token ile verilen girişlerdedir, gövdeler unsafe içermez (P34). P36: AVX512ICL girişleri (blend16_giris_icl, blend_v16_giris_icl, blend_h16_giris_icl) de bu modüldedir (paylaşılan AVX2 gövdelerinin iç bağlantısı için); \
                 tek çağrı yalnız Avx512IclToken ile verilen girişlerdedir.",
        phase: "P34/P36",
    },
    UnsafeExemption {
        module: "dsp/x86/mc16_karisim_ssse3.rs",
        reason: "SSSE3 MC 16 bpc karışım girişleri (blend16_giris, blend_v16_giris, blend_h16_giris): \
                 güvenli #[target_feature(enable = \"ssse3\")] gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.99.1'de unsafe \
                 gerektirir; tek çağrı yalnız Ssse3Token ile verilen girişlerdedir, gövdeler unsafe içermez (P35).",
        phase: "P35",
    },
    UnsafeExemption {
        module: "dsp/x86/mc16_kenar.rs",
        reason: "AVX2 MC 16 bpc girişleri (emu_edge16_giris): güvenli #[target_feature(enable = \"avx2\")] \
                 gövdesini AVX2'siz bağlamdan çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız Avx2Token \
                 ile verilen girişlerdedir, gövdeler unsafe içermez (P34).",
        phase: "P34",
    },
    UnsafeExemption {
        module: "dsp/x86/mc16_kenar_ssse3.rs",
        reason: "SSSE3 MC 16 bpc emu_edge girişi (emu_edge16_giris): \
                 güvenli #[target_feature(enable = \"ssse3\")] gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.99.1'de unsafe \
                 gerektirir; tek çağrı yalnız Ssse3Token ile verilen girişlerdedir, gövde unsafe içermez (P35).",
        phase: "P35",
    },
    UnsafeExemption {
        module: "dsp/x86/mc16_olcekli.rs",
        reason: "AVX2 MC 16 bpc girişleri (put16_giris, prep16_giris): güvenli #[target_feature(enable = \"avx2\")] \
                 gövdesini AVX2'siz bağlamdan çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız Avx2Token \
                 ile verilen girişlerdedir, gövdeler unsafe içermez (P34).",
        phase: "P34",
    },
    UnsafeExemption {
        module: "dsp/x86/mc16_olcekli_ssse3.rs",
        reason: "SSSE3 MC 16 bpc ölçekli girişleri (put16_giris, prep16_giris; 8-tap ve bilinear): \
                 güvenli #[target_feature(enable = \"ssse3\")] gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.99.1'de unsafe \
                 gerektirir; tek çağrı yalnız Ssse3Token ile verilen girişlerdedir, gövdeler unsafe içermez (P35).",
        phase: "P35",
    },
    UnsafeExemption {
        module: "dsp/x86/mc16_ssse3.rs",
        reason: "SSSE3 MC 16 bpc girişleri (put16_giris, prep16_giris): güvenli #[target_feature(enable = \"ssse3\")] \
                 gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız Ssse3Token \
                 ile verilen girişlerdedir, gövdeler unsafe içermez (P35).",
        phase: "P35",
    },
    UnsafeExemption {
        module: "dsp/x86/mc16_warp.rs",
        reason: "AVX2 MC 16 bpc girişleri (warp8x8_16_giris, warp8x8t_16_giris): güvenli #[target_feature(enable = \"avx2\")] \
                 gövdesini AVX2'siz bağlamdan çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız Avx2Token \
                 ile verilen girişlerdedir, gövdeler unsafe içermez (P34).",
        phase: "P34",
    },
    UnsafeExemption {
        module: "dsp/x86/mc16_warp_avx512.rs",
        reason: "AVX512ICL MC 16 bpc warp girişleri (warp8x8_16_giris, warp8x8t_16_giris): güvenli #[target_feature(enable = \"avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni\")] \
                 gövdesini AVX-512'siz bağlamdan çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız \
                 Avx512IclToken ile verilen girişlerdedir, gövdeler unsafe içermez (P36).",
        phase: "P36",
    },
    UnsafeExemption {
        module: "dsp/x86/mc16_warp_ssse3.rs",
        reason: "SSSE3 MC 16 bpc warp girişleri (warp8x8_16_giris, warp8x8t_16_giris): \
                 güvenli #[target_feature(enable = \"ssse3\")] gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.99.1'de unsafe \
                 gerektirir; tek çağrı yalnız Ssse3Token ile verilen girişlerdedir, gövdeler unsafe içermez (P35).",
        phase: "P35",
    },
    UnsafeExemption {
        module: "dsp/x86/mc_bilesik.rs",
        reason: "AVX2 MC bileşik girişleri (avg_giris, w_avg_giris, mask_giris, w_mask_giris): \
                 güvenli #[target_feature(enable = \"avx2\")] gövdesini AVX2'siz bağlamdan \
                 çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız Avx2Token ile \
                 verilen girişlerdedir, gövdeler unsafe içermez (P33). P36: AVX512ICL girişleri (avg_giris_icl, w_avg_giris_icl, mask_giris_icl, w_mask_giris_icl) de bu modüldedir (paylaşılan AVX2 gövdelerinin iç bağlantısı için); \
                 tek çağrı yalnız Avx512IclToken ile verilen girişlerdedir.",
        phase: "P33/P36",
    },
    UnsafeExemption {
        module: "dsp/x86/mc_bilesik_ssse3.rs",
        reason: "SSSE3 MC bileşik girişleri (avg_giris, w_avg_giris, mask_giris, w_mask_giris): \
                 güvenli #[target_feature(enable = \"ssse3\")] gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.99.1'de unsafe \
                 gerektirir; tek çağrı yalnız Ssse3Token ile verilen girişlerdedir, gövdeler unsafe içermez (P35).",
        phase: "P35",
    },
    UnsafeExemption {
        module: "dsp/x86/mc_karisim.rs",
        reason: "AVX2 MC karışım girişleri (blend_giris, blend_v_giris, blend_h_giris): \
                 güvenli #[target_feature(enable = \"avx2\")] gövdesini AVX2'siz bağlamdan \
                 çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız Avx2Token ile \
                 verilen girişlerdedir, gövdeler unsafe içermez (P33). P36: AVX512ICL girişleri (blend_giris_icl, blend_v_giris_icl, blend_h_giris_icl) de bu modüldedir (paylaşılan AVX2 gövdelerinin iç bağlantısı için); \
                 tek çağrı yalnız Avx512IclToken ile verilen girişlerdedir.",
        phase: "P33/P36",
    },
    UnsafeExemption {
        module: "dsp/x86/mc_karisim_ssse3.rs",
        reason: "SSSE3 MC karışım girişleri (blend_giris, blend_v_giris, blend_h_giris): \
                 güvenli #[target_feature(enable = \"ssse3\")] gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.99.1'de unsafe \
                 gerektirir; tek çağrı yalnız Ssse3Token ile verilen girişlerdedir, gövdeler unsafe içermez (P35).",
        phase: "P35",
    },
    UnsafeExemption {
        module: "dsp/x86/mc_kenar.rs",
        reason: "AVX2 MC emu_edge girişi (emu_edge_giris): güvenli #[target_feature(enable = \"avx2\")] \
                 gövdesini AVX2'siz bağlamdan çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız Avx2Token ile verilen \
                 girişlerdedir, gövde unsafe içermez (P33).",
        phase: "P33",
    },
    UnsafeExemption {
        module: "dsp/x86/mc_kenar_ssse3.rs",
        reason: "SSSE3 MC emu_edge girişi (emu_edge_giris): \
                 güvenli #[target_feature(enable = \"ssse3\")] gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.99.1'de unsafe \
                 gerektirir; tek çağrı yalnız Ssse3Token ile verilen girişlerdedir, gövde unsafe içermez (P35).",
        phase: "P35",
    },
    UnsafeExemption {
        module: "dsp/x86/mc_olcekli.rs",
        reason: "AVX2 MC ölçekli girişleri (put_giris, prep_giris; 8-tap ve bilinear): güvenli #[target_feature(enable = \"avx2\")] \
                 gövdesini AVX2'siz bağlamdan çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız Avx2Token ile verilen \
                 girişlerdedir, gövdeler unsafe içermez (P33).",
        phase: "P33",
    },
    UnsafeExemption {
        module: "dsp/x86/mc_olcekli_ssse3.rs",
        reason: "SSSE3 MC ölçekli girişleri (put_giris, prep_giris; 8-tap ve bilinear): \
                 güvenli #[target_feature(enable = \"ssse3\")] gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.99.1'de unsafe \
                 gerektirir; tek çağrı yalnız Ssse3Token ile verilen girişlerdedir, gövdeler unsafe içermez (P35).",
        phase: "P35",
    },
    UnsafeExemption {
        module: "dsp/x86/mc_ssse3.rs",
        reason: "SSSE3 MC 8 bpc girişleri (put_giris, prep_giris): güvenli #[target_feature(enable = \"ssse3\")] \
                 gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız Ssse3Token \
                 ile verilen girişlerdedir, gövdeler unsafe içermez (P35).",
        phase: "P35",
    },
    UnsafeExemption {
        module: "dsp/x86/mc_warp.rs",
        reason: "AVX2 MC warp girişleri (warp8x8_giris, warp8x8t_giris): güvenli #[target_feature(enable = \"avx2\")] \
                 gövdesini AVX2'siz bağlamdan çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız Avx2Token ile verilen \
                 girişlerdedir, gövdeler unsafe içermez (P33).",
        phase: "P33",
    },
    UnsafeExemption {
        module: "dsp/x86/mc_warp_avx512.rs",
        reason: "AVX512ICL MC warp girişleri (warp8x8_giris, warp8x8t_giris): güvenli #[target_feature(enable = \"avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni\")] \
                 gövdesini AVX-512'siz bağlamdan çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız \
                 Avx512IclToken ile verilen girişlerdedir, gövdeler unsafe içermez (P36).",
        phase: "P36",
    },
    UnsafeExemption {
        module: "dsp/x86/mc_warp_sse41.rs",
        reason: "SSE4.1 MC warp girişleri (warp8x8_giris, warp8x8t_giris): \
                 güvenli #[target_feature(enable = \"sse4.1\")] gövdesini SSE4.1'süz bağlamdan çağırmak Rust 1.99.1'de unsafe \
                 gerektirir; tek çağrı yalnız Sse41Token ile verilen girişlerdedir, gövdeler unsafe içermez (P35).",
        phase: "P35",
    },
    UnsafeExemption {
        module: "dsp/x86/mc_warp_ssse3.rs",
        reason: "SSSE3 MC warp girişleri (warp8x8_giris, warp8x8t_giris): \
                 güvenli #[target_feature(enable = \"ssse3\")] gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.99.1'de unsafe \
                 gerektirir; tek çağrı yalnız Ssse3Token ile verilen girişlerdedir, gövdeler unsafe içermez (P35).",
        phase: "P35",
    },
    UnsafeExemption {
        module: "dsp/x86/mem.rs",
        reason: "İşaretçi alan yükleme/saklama intrinsic'leri (loadu/storeu/storel/\
                 storeu_si32, 128 ve 256 bit; AVX-512 maskeli saklama): Rust 1.99.1'de target_feature \
                 bağlamında bile unsafe'tir (ham işaretçi). Her yardımcı sabit boyutlu dizi \
                 referansı alır; erişilen baytlar tür düzeyinde dizinin içindedir (F3b).",
        phase: "P25",
    },
    UnsafeExemption {
        module: "dsp/x86/msac.rs",
        reason: "SSE2 MSAC entropi çözücü girişleri (decode_symbol_adapt4_sse2, decode_symbol_adapt8_sse2, \
                 decode_symbol_adapt16_sse2, decode_hi_tok_sse2, decode_bool_adapt_sse2, decode_bool_equi_sse2, \
                 decode_bool_sse2): güvenli #[target_feature(enable = \"sse2\")] gövdelerini SSE2'siz bağlamdan \
                 çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Sse2Token ile verilen \
                 girişlerdedir, gövdeler unsafe içermez (P47).",
        phase: "P47",
    },
    UnsafeExemption {
        module: "dsp/x86/msac_avx2.rs",
        reason: "AVX2 MSAC 16-simgeli entropi çözücü girişi (decode_symbol_adapt16_avx2): \
                 güvenli #[target_feature(enable = \"avx2\")] gövdesini AVX2'siz bağlamdan çağırmak \
                 Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Avx2Token ile verilen \
                 giriştedir, gövde unsafe içermez (P47).",
        phase: "P47",
    },
    UnsafeExemption {
        module: "dsp/x86/pal.rs",
        reason: "AVX2 pal_idx_finish girişi (pal_idx_finish_avx2): güvenli #[target_feature(enable = \"avx2\")] \
                 gövdesini AVX2'siz bağlamdan çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız Avx2Token \
                 ile verilen giriştedir, gövde unsafe içermez (P45).",
        phase: "P45",
    },
    UnsafeExemption {
        module: "dsp/x86/pal_avx512.rs",
        reason: "AVX512ICL pal_idx_finish girişi (pal_idx_finish_avx512icl): güvenli #[target_feature(enable = \"avx512f,avx512bw,avx512vl,avx512vbmi\")] \
                 gövdesini AVX-512'siz bağlamdan çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız Avx512IclToken \
                 ile verilen giriştedir, gövde unsafe içermez (P45).",
        phase: "P45",
    },
    UnsafeExemption {
        module: "dsp/x86/pal_ssse3.rs",
        reason: "SSSE3 pal_idx_finish girişi (pal_idx_finish_ssse3): güvenli #[target_feature(enable = \"ssse3\")] \
                 gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı yalnız Ssse3Token \
                 ile verilen giriştedir, gövde unsafe içermez (P45).",
        phase: "P45",
    },
    UnsafeExemption {
        module: "dsp/x86/refmvs.rs",
        reason: "AVX2 save_tmvs ve splat_mv girişleri (save_tmvs_avx2, splat_mv_avx2): güvenli #[target_feature(enable = \"avx2\")] \
                 gövdelerini AVX2'siz bağlamdan çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Avx2Token \
                 ile verilen girişlerdedir, gövdeler unsafe içermez (P46).",
        phase: "P46",
    },
    UnsafeExemption {
        module: "dsp/x86/refmvs_avx512.rs",
        reason: "AVX512ICL save_tmvs ve splat_mv girişleri (save_tmvs_avx512icl, splat_mv_avx512icl): güvenli #[target_feature(enable = \"avx512f,avx512bw,avx512vl,avx512vbmi\")] \
                 gövdelerini AVX-512'siz bağlamdan çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Avx512IclToken \
                 ile verilen girişlerdedir, gövdeler unsafe içermez (P46).",
        phase: "P46",
    },
    UnsafeExemption {
        module: "dsp/x86/refmvs_sse2.rs",
        reason: "SSE2 splat_mv girişi (splat_mv_sse2): güvenli #[target_feature(enable = \"sse2\")] \
                 gövdesini SSE2'siz bağlamdan çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Sse2Token \
                 ile verilen giriştedir, gövde unsafe içermez (P46).",
        phase: "P46",
    },
    UnsafeExemption {
        module: "dsp/x86/refmvs_sse41.rs",
        reason: "SSE4.1 load_tmvs girişi (load_tmvs_sse41): güvenli #[target_feature(enable = \"sse4.1\")] \
                 gövdesini SSE4.1'siz bağlamdan çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Sse41Token \
                 ile verilen giriştedir, gövde unsafe içermez (P46).",
        phase: "P46",
    },
    UnsafeExemption {
        module: "dsp/x86/refmvs_ssse3.rs",
        reason: "SSSE3 save_tmvs girişi (save_tmvs_ssse3): güvenli #[target_feature(enable = \"ssse3\")] \
                 gövdesini SSSE3'süz bağlamdan çağırmak Rust 1.100.0'da unsafe gerektirir; tek çağrı yalnız Ssse3Token \
                 ile verilen giriştedir, gövde unsafe içermez (P46).",
        phase: "P46",
    },
    UnsafeExemption {
        module: "dsp/x86/sozluk.rs",
        reason: "Yalnız cfg(test): x86 SIMD sözlüğünün lane davranış testleri (SMD-03). \
                 Güvenli #[target_feature(enable = \"avx2\")] test gövdesini AVX2'siz test \
                 bağlamından çağırmak Rust 1.99.1'de unsafe gerektirir; tek çağrı Avx2Token \
                 isteyen avx2_ile girişindedir, gövde bellek erişimli intrinsic kullanmaz \
                 ve üretim ikilisine girmez (F3d).",
        phase: "P27",
    },
];

#[cfg(test)]
mod tests {
    use super::UNSAFE_EXEMPTIONS;
    use std::fs;
    use std::path::{Path, PathBuf};

    /// `src/` altındaki tüm `.rs` dosyalarını toplar.
    fn kaynak_dosyalari(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = fs::read_dir(dir).expect("kaynak dizini okunmalı");
        for entry in entries {
            let path = entry.expect("dizin girdisi okunmalı").path();
            if path.is_dir() {
                kaynak_dosyalari(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }

    #[test]
    fn muafiyet_defteri_kayitli_ve_gerekceli() {
        // Yalnız capi/ffi.rs ile dsp/x86/ ve dsp/arm/ ISA backend'i muaf; her kayıt gerekçe + faz taşır.
        assert_eq!(
            UNSAFE_EXEMPTIONS.len(),
            89,
            "seksen dokuz muafiyet beklenir (capi/ffi.rs + dsp/arm/mem + dsp/arm/giris + 86 dsp/x86/*)"
        );
        for exemption in UNSAFE_EXEMPTIONS {
            assert!(
                exemption.module == "capi/ffi.rs"
                    || exemption.module.starts_with("dsp/x86/")
                    || exemption.module.starts_with("dsp/arm/"),
                "unsafe yalnız capi/ffi.rs veya dsp/<isa>/ altında: {}",
                exemption.module
            );
        }
        for exemption in UNSAFE_EXEMPTIONS {
            assert!(
                std::path::Path::new(exemption.module).exists()
                    || std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("src")
                        .join(exemption.module)
                        .exists(),
                "muafiyet kaydı var olmayan modülü gösteriyor: {}",
                exemption.module
            );
            assert!(
                exemption.reason.len() > 30,
                "gerekçe anlamlı uzunlukta olmalı (GUV-03): {}",
                exemption.module
            );
            assert!(!exemption.phase.is_empty(), "faz boş olamaz");
        }
    }

    /// Her `unsafe` bloğu `// SAFETY:` gerekçesi taşımalıdır (GUV-04).
    #[test]
    fn her_unsafe_blokta_safety_gerekcesi_var() {
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        kaynak_dosyalari(&src, &mut files);
        let kelime = denetlenen_kelime();
        let mut bloksuz: Vec<(String, usize)> = Vec::new();
        for file in files {
            let rel = file
                .strip_prefix(&src)
                .expect("src altında olmalı")
                .to_string_lossy()
                .replace('\\', "/");
            // Bu dosya (audit/mod.rs) politikayı **tanımlar** ve anahtar
            // sözcüğü zorunlu olarak adlandırır; kendi metnini taramak yanlış
            // pozitif üretir. Orada yaptırım `#![forbid(...)]`tir — derleyici
            // kesin kaynaktır.
            if rel == "audit/mod.rs" {
                continue;
            }
            let text = fs::read_to_string(&file).expect("kaynak okunmalı");
            let blok = format!("{kelime} {{");
            let lines: Vec<&str> = text.lines().collect();
            for (index, line) in lines.iter().enumerate() {
                if !line.contains(&blok) {
                    continue;
                }
                // Geriye doğru en çok 12 satır: araya **başka bir blok**
                // girerse arama durur (her blok kendi gerekçesini taşımalı).
                let mut bulundu = false;
                for earlier in lines[index.saturating_sub(12)..index].iter().rev() {
                    if earlier.contains(&blok) {
                        break;
                    }
                    if earlier.contains("// SAFETY:") {
                        bulundu = true;
                        break;
                    }
                }
                if !bulundu {
                    bloksuz.push((rel.clone(), index + 1));
                }
            }
        }
        assert!(
            bloksuz.is_empty(),
            "// SAFETY: gerekçesi olmayan blok(lar) (dosya, satır): {bloksuz:?}"
        );
    }

    /// F3b (P25): her `// SAFETY:` gerekçesi ADR-0003 §2.4'ün üç cümlesini
    /// taşır — `(1) ISA`, `(2) sınır`, `(3) hizalama/takma ad`. Gerekçe, bloğun
    /// önündeki kesintisiz yorum satırlarıdır.
    #[test]
    fn her_safety_gerekcesi_uc_cumle_sablonunda() {
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        kaynak_dosyalari(&src, &mut files);
        let blok = format!("{} {{", denetlenen_kelime());
        let mut eksik: Vec<(String, usize, &str)> = Vec::new();
        let mut sayac = 0usize;
        for file in files {
            let rel = file
                .strip_prefix(&src)
                .expect("src altında olmalı")
                .to_string_lossy()
                .replace('\\', "/");
            if rel == "audit/mod.rs" {
                continue;
            }
            let text = fs::read_to_string(&file).expect("kaynak okunmalı");
            let lines: Vec<&str> = text.lines().collect();
            for (index, line) in lines.iter().enumerate() {
                if !line.contains(&blok) {
                    continue;
                }
                // Bloğun önündeki kesintisiz `//` yorum satırları.
                let mut start = index;
                while start > 0 && lines[start - 1].trim_start().starts_with("//") {
                    start -= 1;
                }
                let yorum: String = lines[start..index]
                    .iter()
                    .map(|l| l.trim_start().trim_start_matches('/').trim())
                    .collect::<Vec<_>>()
                    .join(" ");
                sayac += 1;
                for parca in ["SAFETY:", "(1) ISA", "(2) sınır", "(3) hizalama/takma ad"] {
                    if !yorum.contains(parca) {
                        eksik.push((rel.clone(), index + 1, parca));
                    }
                }
            }
        }
        assert!(sayac > 0, "denetlenecek blok bulunamadı");
        assert!(
            eksik.is_empty(),
            "üç cümle şablonu eksik (dosya, satır, eksik parça): {eksik:?}"
        );
    }

    /// Denetlenen anahtar sözcük: parçalardan kurulur, böylece denetim kendi
    /// metnini yakalamaz (`scripts/kabul/channel_check.py` ile aynı teknik).
    fn denetlenen_kelime() -> String {
        ["uns", "afe"].concat()
    }

    /// `//` ve `/* … */` yorumlarını kabaca temizler (denetim taraması için).
    fn yorumsuz_kod(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut chars = text.chars().peekable();
        let (mut blok, mut satir) = (false, false);
        while let Some(c) = chars.next() {
            if satir {
                if c == '\n' {
                    satir = false;
                    out.push(c);
                }
                continue;
            }
            if blok {
                if c == '*' && chars.peek() == Some(&'/') {
                    chars.next();
                    blok = false;
                }
                continue;
            }
            if c == '/' {
                match chars.peek() {
                    Some('/') => {
                        chars.next();
                        satir = true;
                        continue;
                    }
                    Some('*') => {
                        chars.next();
                        blok = true;
                        continue;
                    }
                    _ => {}
                }
            }
            out.push(c);
        }
        out
    }

    #[test]
    fn her_kaynak_dosyasi_politika_bildirir() {
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        kaynak_dosyalari(&src, &mut files);
        assert!(
            files.len() > 10,
            "kaynak ağacı beklenenden küçük: {files:?}"
        );

        let kelime = denetlenen_kelime();
        for file in files {
            let rel = file
                .strip_prefix(&src)
                .expect("src altında olmalı")
                .to_string_lossy()
                .replace('\\', "/");
            // Bu dosya (audit/mod.rs) politikayı **tanımlar** ve anahtar
            // sözcüğü zorunlu olarak adlandırır; kendi metnini taramak yanlış
            // pozitif üretir.
            if rel == "audit/mod.rs" {
                continue;
            }
            let text = fs::read_to_string(&file).expect("kaynak okunmalı");

            // Politikayı `#[forbid(...)]`/`#[deny(...)]` özniteliği bildirir; öznitelik
            // metni aşağıdaki "çıplak kullanım" taramasından ayıklanır.
            let politika_eki = format!("{kelime}_code");
            let politikali = text.contains(&format!("#![forbid({politika_eki})]"))
                || text.contains(&format!("#![deny({politika_eki})]"));
            let kayitli = UNSAFE_EXEMPTIONS.iter().any(|e| e.module == rel);
            assert!(
                politikali || kayitli,
                "{rel}: `#![forbid({politika_eki})]` yok ve muafiyet defterinde kaydı yok (GUV-02)"
            );

            // Yorumlar ve politika özniteliği ayıklandıktan sonra kalan çıplak kelime
            // gerçek kullanımdır (blok/fn/impl/trait) ve defterde kayıtlı olmalıdır.
            let kod = yorumsuz_kod(&text).replace(&politika_eki, "");
            if kod.contains(&kelime) {
                assert!(
                    kayitli,
                    "{rel}: muafiyet defterinde kaydı olmayan `{kelime}` kullanımı var (GUV-04)"
                );
            }
        }
    }
}
