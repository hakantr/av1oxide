//! Bellek erişimli x86 intrinsic'lerinin dilim tabanlı sarmalayıcıları (F3b / P25).
//!
//! Rust 1.99.1'de `core::arch` intrinsic'lerinin çoğu, uygun
//! `#[target_feature]` bağlamında **güvenlidir** (target_feature 1.1). Yalnız
//! işaretçi alan yükleme/saklama intrinsic'leri `unsafe` kalır; bu modül
//! onların **tek** yeridir. Her yardımcı sabit boyutlu bir dizi referansı
//! (`&[T; N]`) alır: erişilen baytlar tür düzeyinde dizinin içindedir, sınır
//! denetimi çağıranın dilimden diziyi ayırdığı yerde ([`chunk`],
//! [`chunk_mut`]) bir kez yapılır ve aşımda panik verir (bellek dışına erişim
//! yoktur).
//!
//! 128-bit yardımcılar `#[target_feature(enable = "sse2")]`, 256-bit
//! yardımcılar `#[target_feature(enable = "avx2")]`, maskeli saklamalar (P30)
//! `#[target_feature(enable = "avx512bw,avx512vl")]` taşır (Rust 1.99.1, SSE2
//! derleme yapılandırmasında etkin olsa bile özelliğin işlevde listelenmesini
//! ister). Hepsi güvenli target_feature işlevleridir; 128-bit yardımcılar
//! SSSE3/SSE4.1 ve AVX2 gövdelerinden, 256-bit yardımcılar yalnız AVX2
//! gövdelerinden çağrılır (her ikisi SSE2'yi kapsar). P36: 512-bit yardımcılar
//! `#[target_feature(enable = "avx512f,avx512bw")]` taşır ve yalnız AVX512ICL
//! gövdelerinden çağrılır.

#![allow(unsafe_code, clippy::items_after_test_module)]

use core::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm_loadu_si128, _mm_mask_storeu_epi8, _mm_mask_storeu_epi16,
    _mm_storel_epi64, _mm_storeu_si32, _mm_storeu_si128, _mm256_cmpge_epu32_mask,
    _mm256_loadu_si256, _mm256_mask_storeu_epi8, _mm256_set1_epi32, _mm256_storeu_si256,
    _mm512_i32gather_epi64, _mm512_loadu_si512, _mm512_maskz_loadu_epi8, _mm512_storeu_si512,
};
use core::arch::x86_64::{
    __m128i as M128, _mm_setr_epi32, _mm256_and_si256, _mm256_blend_epi16, _mm256_cvtepu8_epi16,
    _mm256_i32gather_epi32, _mm256_set1_epi16, _mm256_setzero_si256, _mm256_slli_epi32,
    _mm256_srli_epi32, _mm256_srlv_epi32, _mm256_unpackhi_epi8, _mm256_unpacklo_epi8,
    _mm512_and_si512, _mm512_broadcast_i32x4, _mm512_i32gather_epi32, _mm512_or_si512,
    _mm512_set1_epi16, _mm512_set1_epi32, _mm512_shuffle_epi8, _mm512_srli_epi32,
    _mm512_ternarylogic_epi32,
};

/// `s[at .. at + N]`'i sabit boyutlu diziye ayırır; aşımda panik.
#[inline(always)]
#[must_use]
pub fn chunk<T, const N: usize>(s: &[T], at: usize) -> &[T; N] {
    s[at..]
        .first_chunk::<N>()
        .expect("SIMD yüklemesi dilim sınırını aşıyor")
}

/// `s[at .. at + N]`'i değiştirilebilir sabit boyutlu diziye ayırır; aşımda panik.
#[inline(always)]
#[must_use]
pub fn chunk_mut<T, const N: usize>(s: &mut [T], at: usize) -> &mut [T; N] {
    s[at..]
        .first_chunk_mut::<N>()
        .expect("SIMD saklaması dilim sınırını aşıyor")
}

/// 16 baytı hizasız yükler.
#[target_feature(enable = "sse2")]
#[inline]
#[must_use]
pub fn load_u8x16(src: &[u8; 16]) -> __m128i {
    // SAFETY: (1) ISA: `_mm_loadu_si128` SSE2'dir; işlev SSE2
    // `#[target_feature]` taşır. (2) sınır: `src` tam 16 bayttır, komut tam 16
    // bayt okur. (3) hizalama/takma ad: hizasız yüklemedir; paylaşılan
    // referanstan salt okuma.
    unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) }
}

/// 32 baytı hizasız yükler (P34: 16 bpc satırları bayt dilimindedir).
#[target_feature(enable = "avx2")]
#[inline]
#[must_use]
pub fn load_u8x32(src: &[u8; 32]) -> __m256i {
    // SAFETY: (1) ISA: `_mm256_loadu_si256` AVX'tir; işlev AVX2
    // `#[target_feature]` taşır. (2) sınır: `src` tam 32 bayttır, komut tam 32
    // bayt okur. (3) hizalama/takma ad: hizasız yüklemedir; paylaşılan
    // referanstan salt okuma.
    unsafe { _mm256_loadu_si256(src.as_ptr().cast::<__m256i>()) }
}

/// 16 baytı hizasız saklar.
#[target_feature(enable = "sse2")]
#[inline]
pub fn store_u8x16(dst: &mut [u8; 16], v: __m128i) {
    // SAFETY: (1) ISA: `_mm_storeu_si128` SSE2'dir; işlev SSE2
    // `#[target_feature]` taşır. (2) sınır: `dst` tam 16 bayttır, komut tam 16
    // bayt yazar. (3) hizalama/takma ad: hizasız saklamadır; `&mut` tek erişim
    // yoludur.
    unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast::<__m128i>(), v) }
}

/// Alt 64 bit lane'i 8 bayt olarak saklar (`movq`).
#[target_feature(enable = "sse2")]
#[inline]
pub fn store_u8x8(dst: &mut [u8; 8], v: __m128i) {
    // SAFETY: (1) ISA: `_mm_storel_epi64` SSE2'dir; işlev SSE2
    // `#[target_feature]` taşır. (2) sınır: `dst` tam 8 bayttır, komut yalnız
    // alt 8 baytı yazar. (3) hizalama/takma ad: hizasız saklamadır; `&mut` tek
    // erişim yoludur.
    unsafe { _mm_storel_epi64(dst.as_mut_ptr().cast::<__m128i>(), v) }
}

/// Alt 32 bit lane'i 4 bayt olarak saklar.
#[target_feature(enable = "sse2")]
#[inline]
pub fn store_u8x4(dst: &mut [u8; 4], v: __m128i) {
    // SAFETY: (1) ISA: `_mm_storeu_si32` SSE2'dir; işlev SSE2
    // `#[target_feature]` taşır. (2) sınır: `dst` tam 4 bayttır, komut yalnız 4
    // bayt yazar. (3) hizalama/takma ad: hizasız saklamadır; `&mut` tek erişim
    // yoludur.
    unsafe { _mm_storeu_si32(dst.as_mut_ptr(), v) }
}

/// P30 — AVX-512 maskeli 8 bit saklama: `v`'nin `k` maskesindeki baytları
/// `dst`'ye yazar. `k`'nın yalnız `N`'den küçük bitleri kullanılır; yazılan
/// her bayt dizinin içindedir (fazladan açılan bir lane bile `dst` dışına
/// yazamaz, yalnız bloğun komşu sütununu değiştirir — sentinel testi bunu
/// yakalar).
#[target_feature(enable = "avx512bw,avx512vl")]
#[inline]
pub fn store_u8_k<const N: usize>(dst: &mut [u8; N], v: __m128i, k: u16) {
    const { assert!(N >= 1 && N <= 16, "maskeli 8 bit saklama en çok 16 bayt") };
    let k = k & ((1u32 << N) - 1) as u16;
    // SAFETY: (1) ISA: `_mm_mask_storeu_epi8` AVX512BW + VL'dir; işlev bu
    // özellikleri `#[target_feature]` ile taşır. (2) sınır: maske yalnız
    // `i < N` baytlarını seçer ve `dst` tam `N` bayttır; maskesiz bayta
    // yazılmaz. (3) hizalama/takma ad: hizasız maskeli saklamadır; `&mut` tek
    // erişim yoludur.
    unsafe { _mm_mask_storeu_epi8(dst.as_mut_ptr().cast::<i8>(), k, v) }
}

/// P30 — AVX-512 maskeli 16 bit saklama: `v`'nin `k` maskesindeki 16 bit
/// lane'lerini `dst`'ye (küçük sonlu) yazar. `k`'nın yalnız `N / 2`'den küçük
/// bitleri kullanılır (en çok 8 lane); yazılan her bayt dizinin içindedir.
#[target_feature(enable = "avx512bw,avx512vl")]
#[inline]
pub fn store_u16_k<const N: usize>(dst: &mut [u8; N], v: __m128i, k: u8) {
    const { assert!(N >= 2, "maskeli 16 bit saklama en az bir lane") };
    let lanes = if N / 2 < 8 { N / 2 } else { 8 };
    let k = k & ((1u32 << lanes) - 1) as u8;
    // SAFETY: (1) ISA: `_mm_mask_storeu_epi16` AVX512BW + VL'dir; işlev bu
    // özellikleri taşır. (2) sınır: maske yalnız `i < min(N/2, 8)` lane'lerini
    // seçer; lane `i` `dst[2i .. 2i+2]`'ye yazar ve `2i + 2 <= N`. (3)
    // hizalama/takma ad: hizasız maskeli saklamadır (işaretçi `i16`'ya
    // dönüştürülür ama yalnız komuta verilir); `&mut` tek erişim yoludur.
    unsafe { _mm_mask_storeu_epi16(dst.as_mut_ptr().cast::<i16>(), k, v) }
}

/// 16 i16'yı hizasız yükler.
#[target_feature(enable = "avx2")]
#[inline]
#[must_use]
pub fn load_i16x16(src: &[i16; 16]) -> __m256i {
    // SAFETY: (1) ISA: işlev `#[target_feature(enable = "avx2")]` taşır ve
    // yalnız AVX2 bağlamından çağrılabilir (güvenli target_feature işlevi).
    // (2) sınır: `src` tam 32 bayttır, komut tam 32 bayt okur.
    // (3) hizalama/takma ad: hizasız yüklemedir; paylaşılan referanstan salt
    // okuma.
    unsafe { _mm256_loadu_si256(src.as_ptr().cast::<__m256i>()) }
}

/// 16 i16'yı hizasız saklar.
#[target_feature(enable = "avx2")]
#[inline]
pub fn store_i16x16(dst: &mut [i16; 16], v: __m256i) {
    // SAFETY: (1) ISA: işlev `#[target_feature(enable = "avx2")]` taşır ve
    // yalnız AVX2 bağlamından çağrılabilir. (2) sınır: `dst` tam 32 bayttır,
    // komut tam 32 bayt yazar. (3) hizalama/takma ad: hizasız saklamadır;
    // `&mut` tek erişim yoludur.
    unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast::<__m256i>(), v) }
}

/// 32 baytı hizasız saklar (P32, MC `put` `w ≥ 32` yatay yolu).
#[target_feature(enable = "avx2")]
#[inline]
pub fn store_u8x32(dst: &mut [u8; 32], v: __m256i) {
    // SAFETY: (1) ISA: işlev `#[target_feature(enable = "avx2")]` taşır ve
    // yalnız AVX2 bağlamından çağrılabilir. (2) sınır: `dst` tam 32 bayttır,
    // komut tam 32 bayt yazar. (3) hizalama/takma ad: hizasız saklamadır;
    // `&mut` tek erişim yoludur.
    unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast::<__m256i>(), v) }
}

/// 8 i16'yı hizasız saklar (P32, MC `prep` çıkışı).
#[target_feature(enable = "sse2")]
#[inline]
pub fn store_i16x8(dst: &mut [i16; 8], v: __m128i) {
    // SAFETY: (1) ISA: `_mm_storeu_si128` SSE2'dir; işlev SSE2
    // `#[target_feature]` taşır. (2) sınır: `dst` tam 16 bayttır, komut tam 16
    // bayt yazar. (3) hizalama/takma ad: hizasız saklamadır; `&mut` tek erişim
    // yoludur.
    unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast::<__m128i>(), v) }
}

/// 8 i32'yi hizasız saklar.
#[target_feature(enable = "avx2")]
#[inline]
pub fn store_i32x8(dst: &mut [i32; 8], v: __m256i) {
    // SAFETY: (1) ISA: işlev `#[target_feature(enable = "avx2")]` taşır ve
    // yalnız AVX2 bağlamından çağrılabilir. (2) sınır: `dst` tam 32 bayttır,
    // komut tam 32 bayt yazar. (3) hizalama/takma ad: hizasız saklamadır;
    // `&mut` tek erişim yoludur.
    unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast::<__m256i>(), v) }
}

/// 8 i16'yı hizasız yükler (P35, SSSE3 MC bileşik girdileri).
#[target_feature(enable = "sse2")]
#[inline]
#[must_use]
pub fn load_i16x8(src: &[i16; 8]) -> __m128i {
    // SAFETY: (1) ISA: `_mm_loadu_si128` SSE2'dir; işlev SSE2
    // `#[target_feature]` taşır. (2) sınır: `src` tam 16 bayttır, komut tam 16
    // bayt okur. (3) hizalama/takma ad: hizasız yüklemedir; paylaşılan
    // referanstan salt okuma.
    unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) }
}

/// P36: 64 baytı hizasız yükler.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
#[must_use]
pub fn load_u8x64(src: &[u8; 64]) -> __m512i {
    // SAFETY: (1) ISA: `_mm512_loadu_si512` AVX-512F'dir; işlev bu özelliği
    // `#[target_feature]` ile taşır. (2) sınır: `src` tam 64 bayttır, komut tam
    // 64 bayt okur. (3) hizalama/takma ad: hizasız yüklemedir; paylaşılan
    // referanstan salt okuma.
    unsafe { _mm512_loadu_si512(src.as_ptr().cast::<__m512i>()) }
}

/// P36: 64 baytı hizasız saklar.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
pub fn store_u8x64(dst: &mut [u8; 64], v: __m512i) {
    // SAFETY: (1) ISA: `_mm512_storeu_si512` AVX-512F'dir; işlev bu özelliği
    // taşır. (2) sınır: `dst` tam 64 bayttır, komut tam 64 bayt yazar. (3)
    // hizalama/takma ad: hizasız saklamadır; `&mut` tek erişim yoludur.
    unsafe { _mm512_storeu_si512(dst.as_mut_ptr().cast::<__m512i>(), v) }
}

/// P36: 32 i16'yı hizasız yükler.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
#[must_use]
pub fn load_i16x32(src: &[i16; 32]) -> __m512i {
    // SAFETY: (1) ISA: `_mm512_loadu_si512` AVX-512F'dir; işlev bu özelliği
    // taşır. (2) sınır: `src` tam 32 öğe = 64 bayttır, komut tam 64 bayt okur.
    // (3) hizalama/takma ad: hizasız yüklemedir; paylaşılan referanstan salt
    // okuma.
    unsafe { _mm512_loadu_si512(src.as_ptr().cast::<__m512i>()) }
}

/// P36: 32 i16'yı hizasız saklar.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
pub fn store_i16x32(dst: &mut [i16; 32], v: __m512i) {
    // SAFETY: (1) ISA: `_mm512_storeu_si512` AVX-512F'dir; işlev bu özelliği
    // taşır. (2) sınır: `dst` tam 32 öğe = 64 bayttır, komut tam 64 bayt yazar.
    // (3) hizalama/takma ad: hizasız saklamadır; `&mut` tek erişim yoludur.
    unsafe { _mm512_storeu_si512(dst.as_mut_ptr().cast::<__m512i>(), v) }
}

/// P36: dizinin ilk `N` baytını (`N ≤ 64`) 64 baytlık yazmacın alt baytlarına
/// yükler, üst baytlar sıfır (`vmovdqu8` sıfırlayan maske). Okuma yalnız dizinin
/// içindedir (maskeli lane'ler belleğe dokunmaz).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
#[must_use]
pub fn load_u8_kz<const N: usize>(src: &[u8; N]) -> __m512i {
    const { assert!(N >= 1 && N <= 64, "maskeli yükleme en çok 64 bayt") };
    let k = if N == 64 { u64::MAX } else { (1u64 << N) - 1 };
    // SAFETY: (1) ISA: `_mm512_maskz_loadu_epi8` AVX-512BW'dir; işlev bu
    // özelliği taşır. (2) sınır: maske yalnız `i < N` baytlarını seçer ve `src`
    // tam `N` bayttır; maskesiz lane'ler okunmaz (hata bastırmalı yükleme). (3)
    // hizalama/takma ad: hizasız yüklemedir; paylaşılan referanstan salt okuma.
    unsafe { _mm512_maskz_loadu_epi8(k, src.as_ptr().cast::<i8>()) }
}

/// P36: dilimin ilk `n` baytını (`n ≤ 64`, çalışma zamanı uzunluğu) 64
/// baytlık yazmacın alt baytlarına yükler, üst baytlar sıfır. `n > 64` ya da
/// `n > src.len()` panik; maskeli lane'ler belleğe dokunmaz.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
#[must_use]
pub fn load_u8_kn(src: &[u8], n: usize) -> __m512i {
    assert!(n <= 64 && n <= src.len(), "maskeli yükleme dilim dışında");
    let k = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
    // SAFETY: (1) ISA: `_mm512_maskz_loadu_epi8` AVX-512BW'dir; işlev bu
    // özelliği taşır. (2) sınır: maske yalnız `i < n` baytlarını seçer ve
    // `n ≤ src.len()` (yukarıdaki denetim); maskesiz lane'ler okunmaz (hata
    // bastırmalı yükleme, `n = 0`'da hiçbir bayt). (3) hizalama/takma ad:
    // hizasız yüklemedir; paylaşılan referanstan salt okuma.
    unsafe { _mm512_maskz_loadu_epi8(k, src.as_ptr().cast::<i8>()) }
}

/// P36: `v`'nin ilk `n` baytını (`n ≤ 16`, çalışma zamanı uzunluğu) `dst`'nin
/// başına yazar (maskeli saklama); `dst`'nin öbür baytları değişmez. `n > 16`
/// ya da `n > dst.len()` panik.
#[target_feature(enable = "avx512bw,avx512vl")]
#[inline]
pub fn store_u8_kn(dst: &mut [u8], v: __m128i, n: usize) {
    assert!(n <= 16 && n <= dst.len(), "maskeli saklama dilim dışında");
    let k = ((1u32 << n) - 1) as u16;
    // SAFETY: (1) ISA: `_mm_mask_storeu_epi8` AVX512BW + VL'dir; işlev bu
    // özellikleri taşır. (2) sınır: maske yalnız `i < n ≤ dst.len()` baytlarını
    // seçer (yukarıdaki denetim); maskesiz bayta yazılmaz. (3) hizalama/takma
    // ad: hizasız maskeli saklamadır; `&mut` tek erişim yoludur.
    unsafe { _mm_mask_storeu_epi8(dst.as_mut_ptr().cast::<i8>(), k, v) }
}

/// P36: [`store_u8_kn`]'nin 32 baytlık (`ymm`) karşılığı (`n ≤ 32`).
#[target_feature(enable = "avx512bw,avx512vl")]
#[inline]
pub fn store_u8x32_kn(dst: &mut [u8], v: __m256i, n: usize) {
    assert!(n <= 32 && n <= dst.len(), "maskeli saklama dilim dışında");
    let k = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
    // SAFETY: (1) ISA: `_mm256_mask_storeu_epi8` AVX512BW + VL'dir; işlev bu
    // özellikleri taşır. (2) sınır: maske yalnız `i < n ≤ dst.len()` baytlarını
    // seçer (yukarıdaki denetim); maskesiz bayta yazılmaz. (3) hizalama/takma
    // ad: hizasız maskeli saklamadır; `&mut` tek erişim yoludur.
    unsafe { _mm256_mask_storeu_epi8(dst.as_mut_ptr().cast::<i8>(), k, v) }
}

/// P36: `tablo[idx[i]]` (`i` 0…7) 8 `i64` toplama (`vpgatherdq`). Dizinler
/// vektörle denetlenir: biri `tablo.len()`'den küçük değilse (negatif dizin
/// işaretsiz karşılaştırmada büyüktür) panik; bellek dışına erişim yoktur.
#[target_feature(enable = "avx512f,avx512vl")]
#[inline]
#[must_use]
pub fn gather_i64x8(tablo: &[i64], idx: __m256i) -> __m512i {
    let n = _mm256_set1_epi32(i32::try_from(tablo.len()).unwrap_or(i32::MAX));
    assert!(
        _mm256_cmpge_epu32_mask(idx, n) == 0,
        "toplama dizini tablo dışında"
    );
    // SAFETY: (1) ISA: `_mm512_i32gather_epi64` AVX-512F'dir; işlev bu özelliği
    // taşır. (2) sınır: her dizin `[0, tablo.len())` içindedir (yukarıdaki
    // işaretsiz karşılaştırma, aksi hâlde panik) ve ölçek 8 `i64` boyudur; okunan
    // her öğe `tablo`'nun içindedir. (3) hizalama/takma ad: doğal hizalı `i64`
    // dizisinden, paylaşılan referans üzerinden salt okuma.
    unsafe { _mm512_i32gather_epi64::<8>(idx, tablo.as_ptr()) }
}

/// P39: 8 bpc film grain ölçekleme tablosu araması — `tablo[idx[i] & 0xFF]`
/// (`i` 0…7, 32 bitlik şeritler; sonuç `0..=255`). `vpgatherdd` her şeritte
/// dizinden başlayarak 4 bayt okur; tablo bu yüzden 256 + 3 = 259 bayttır
/// (çözme hattının ölçekleme tamponu `SCALING_SIZE_MAX` bayttır). Dizinler
/// `& 0xFF` ile tablonun ilk 256 girişine sınırlanır.
pub const OLCEKLEME_TOPLAMA: usize = 259;

/// P39: [`OLCEKLEME_TOPLAMA`] baytlık tablodan 8 bayt değer (32 bitlik şeritlere).
#[target_feature(enable = "avx2")]
#[inline]
#[must_use]
pub fn gather_u8_256(tablo: &[u8; OLCEKLEME_TOPLAMA], idx: __m256i) -> __m256i {
    let m = _mm256_set1_epi32(0xFF);
    let i = _mm256_and_si256(idx, m);
    // SAFETY: (1) ISA: `_mm256_i32gather_epi32` AVX2'dir; işlev AVX2
    // `#[target_feature]` taşır. (2) sınır: her dizin `& 0xFF` ile `0..=255`'tir,
    // ölçek 1; şerit `tablo[i .. i + 4]`'ü okur, en büyük okunan bayt 258 <
    // `tablo.len()` = 259. (3) hizalama/takma ad: hizasız 32 bitlik okumalar,
    // paylaşılan referans üzerinden salt okuma.
    let g = unsafe { _mm256_i32gather_epi32::<1>(tablo.as_ptr().cast::<i32>(), i) };
    _mm256_and_si256(g, m)
}

/// `tablo[v[i]]` (16 `u16` şerit, aynı sıra); `v`'nin şeritleri `≤ 255`
/// olmalıdır (çağıranlar sıfır genişletilmiş baytlar ya da `& 0xFF` maskesi verir).
/// Çift şeritler 32 bitlik şeridin alt, tek şeritler üst yarısından iki
/// `vpgatherdd` ile okunur ve `vpblendw` ile birleşir.
#[target_feature(enable = "avx2")]
#[inline]
fn olcekleme_u16(tablo: &[u8; OLCEKLEME_TOPLAMA], v: __m256i) -> __m256i {
    let cift = _mm256_and_si256(v, _mm256_set1_epi32(0xFFFF));
    let tek = _mm256_srli_epi32::<16>(v);
    // SAFETY: (1) ISA: `_mm256_i32gather_epi32` AVX2'dir; işlev AVX2
    // `#[target_feature]` taşır. (2) sınır: dizinler `v`'nin `u16` şeritleridir
    // ve bu işlevin üç çağıranı onları sıfır genişletilmiş baytlardan ya da
    // `& 0xFF` maskesiyle kurar (`0..=255`), ölçek 1; şerit `tablo[i .. i + 4]`'ü okur, en büyük okunan
    // bayt 258 < 259. (3) hizalama/takma ad: hizasız 32 bitlik okumalar,
    // paylaşılan referans üzerinden salt okuma.
    let (gc, gt) = unsafe {
        (
            _mm256_i32gather_epi32::<1>(tablo.as_ptr().cast::<i32>(), cift),
            _mm256_i32gather_epi32::<1>(tablo.as_ptr().cast::<i32>(), tek),
        )
    };
    let w = _mm256_blend_epi16::<0xAA>(gc, _mm256_slli_epi32::<16>(gt));
    _mm256_and_si256(w, _mm256_set1_epi16(0xFF))
}

/// P39: `tablo[v[i] & 0xFF]` (16 `u16` şerit, aynı sıra); maske dizinleri tablonun
/// ilk 256 girişine sınırlar.
#[target_feature(enable = "avx2")]
#[inline]
#[must_use]
pub fn olcekleme16_u16(tablo: &[u8; OLCEKLEME_TOPLAMA], v: __m256i) -> __m256i {
    olcekleme_u16(tablo, _mm256_and_si256(v, _mm256_set1_epi16(0xFF)))
}

/// P39: 16 baytın `u16` genişlemesi ve ölçekleme değerleri — `(s, tablo[s])`
/// (16 şerit, doğal sıra; dizinler sıfır genişletilmiş baytlar, yapı gereği `≤ 255`).
#[target_feature(enable = "avx2")]
#[inline]
#[must_use]
pub fn olcekleme16(tablo: &[u8; OLCEKLEME_TOPLAMA], s: M128) -> (__m256i, __m256i) {
    let s16 = _mm256_cvtepu8_epi16(s);
    (s16, olcekleme_u16(tablo, s16))
}

/// P39: 32 baytın `u16` genişlemeleri ve ölçekleme değerleri —
/// `(lo, hi, tablo[lo], tablo[hi])`; `lo` = `punpcklbw` (baytlar 0–7, 16–23),
/// `hi` = `punpckhbw` (8–15, 24–31) şerit sırası.
#[target_feature(enable = "avx2")]
#[inline]
#[must_use]
pub fn olcekleme32(
    tablo: &[u8; OLCEKLEME_TOPLAMA],
    s: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let z = _mm256_setzero_si256();
    let (lo, hi) = (_mm256_unpacklo_epi8(s, z), _mm256_unpackhi_epi8(s, z));
    (lo, hi, olcekleme_u16(tablo, lo), olcekleme_u16(tablo, hi))
}

/// P42: `N` (`1024` ya da `4096`) baytlık 16 bpc film grain ölçekleme
/// tablosundan 16 `u16` dizin için 16 ölçekleme baytı (`0..=255`, aynı sıra).
///
/// `dav1d`'nin `filmgrain16_avx2.asm` dosyası `vpgatherdd` ile `[scalingq + m4]`
/// (`4095`'te +3 bayt taşma) ve `[scalingq + m3 - 2]` (`0`'da -2 bayt taşma)
/// okur; bu modülde tablo dışına erişim yasaktır. Her dizin `& (N − 1)` ile
/// `[0, N)` aralığına maskelenir, `idx >> 2` dword diziniyle (`SCALE = 4`,
/// 4 bayt hizalı) `tablo[(idx & !3) .. (idx & !3) + 4]` okunur ve `(idx & 3) * 8`
/// bit sağa kaydırılarak hedef bayt seçilir.
#[target_feature(enable = "avx2")]
#[inline]
#[must_use]
pub fn olcekleme16_gather<const N: usize>(tablo: &[u8; N], v: __m256i) -> __m256i {
    const {
        assert!(
            N == 1024 || N == 4096,
            "16 bpc ölçekleme tablosu 1024 ya da 4096 bayt"
        );
    }
    let mask = _mm256_set1_epi32((N - 1) as i32);
    let cift = _mm256_and_si256(v, mask);
    let tek = _mm256_and_si256(_mm256_srli_epi32::<16>(v), mask);
    let dc = _mm256_srli_epi32::<2>(cift);
    let dt = _mm256_srli_epi32::<2>(tek);
    // SAFETY: (1) ISA: `_mm256_i32gather_epi32` AVX2'dir; işlev AVX2
    // `#[target_feature]` taşır. (2) sınır: `cift` ve `tek` `& (N - 1)` ile
    // `0..N` aralığındadır, `>> 2` sonrası her dword dizini `d` `0..N/4`
    // içindedir ve `SCALE = 4` ile `tablo[d * 4 .. d * 4 + 4]` okunur (`d * 4 + 4 <= N`).
    // (3) hizalama/takma ad: 4 bayt adımlı 32 bitlik okumalar, paylaşılan
    // referans üzerinden salt okuma.
    let (gc, gt) = unsafe {
        (
            _mm256_i32gather_epi32::<4>(tablo.as_ptr().cast::<i32>(), dc),
            _mm256_i32gather_epi32::<4>(tablo.as_ptr().cast::<i32>(), dt),
        )
    };
    let uc = _mm256_set1_epi32(3);
    let sc = _mm256_slli_epi32::<3>(_mm256_and_si256(cift, uc));
    let st = _mm256_slli_epi32::<3>(_mm256_and_si256(tek, uc));
    let bc = _mm256_srlv_epi32(gc, sc);
    let bt = _mm256_slli_epi32::<16>(_mm256_srlv_epi32(gt, st));
    let w = _mm256_blend_epi16::<0xAA>(bc, bt);
    _mm256_and_si256(w, _mm256_set1_epi16(0xFF))
}

/// P42: `N` (`1024` ya da `4096`) baytlık 16 bpc film grain ölçekleme
/// tablosundan 32 `u16` dizin için 32 ölçekleme baytı (`0..=255`, aynı sıra).
///
/// `dav1d`'nin `filmgrain16_avx512.asm` dosyası `vpgatherdd` ile `[scalingq + m16]`
/// (`SCALE = 1`, `N − 1`'de +3 bayt taşma) okur; bu modülde tablo dışına erişim
/// yasaktır. Her dizin `(idx >> 2) & ((N − 1) >> 2)` ile `[0, N / 4)` aralığındaki
/// 4 bayt hizalı dword dizinine indirgenir (`SCALE = 4`), `tablo[(idx & !3) .. (idx & !3) + 4]`
/// okunur ve `vpshufb` + `vpblendmw` ile `(idx & 3)` baytı seçilip üst bayt sıfırlanır.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
#[must_use]
pub fn olcekleme16_gather_512<const N: usize>(tablo: &[u8; N], v: __m512i) -> __m512i {
    const {
        assert!(
            N == 1024 || N == 4096,
            "16 bpc ölçekleme tablosu 1024 ya da 4096 bayt"
        );
    }
    let dmask = _mm512_set1_epi32(((N - 1) >> 2) as i32);
    let dc = _mm512_and_si512(_mm512_srli_epi32::<2>(v), dmask);
    let dt = _mm512_and_si512(_mm512_srli_epi32::<18>(v), dmask);
    // SAFETY: (1) ISA: `_mm512_i32gather_epi32` AVX-512F'dir; işlev AVX-512F/BW/VL
    // `#[target_feature]` taşır. (2) sınır: `dc` ve `dt` `& ((N - 1) >> 2)` ile
    // `0..N/4` aralığındadır ve `SCALE = 4` ile `tablo[d * 4 .. d * 4 + 4]`
    // okunur (`d * 4 + 4 <= N`). (3) hizalama/takma ad: 4 bayt adımlı 32 bitlik
    // okumalar, paylaşılan referans üzerinden salt okuma.
    let (gc, gt) = unsafe {
        (
            _mm512_i32gather_epi32::<4>(dc, tablo.as_ptr().cast::<i32>()),
            _mm512_i32gather_epi32::<4>(dt, tablo.as_ptr().cast::<i32>()),
        )
    };
    let taban = _mm512_broadcast_i32x4(_mm_setr_epi32(
        0x8000_8000u32 as i32,
        0x8004_8004u32 as i32,
        0x8008_8008u32 as i32,
        0x800C_800Cu32 as i32,
    ));
    let shuf = _mm512_or_si512(_mm512_and_si512(v, _mm512_set1_epi16(3)), taban);
    let bc = _mm512_shuffle_epi8(gc, shuf);
    let bt = _mm512_shuffle_epi8(gt, shuf);
    _mm512_ternarylogic_epi32::<0xD8>(bt, bc, dmask)
}

#[cfg(test)]
mod tests {
    use super::{chunk, chunk_mut};

    /// Dizi ayırma sınırda döner, aşımda panik verir (bellek dışına erişim
    /// yok). Intrinsic yardımcıları güvenli `#[target_feature]` işlevleridir;
    /// onlar çekirdek testleri (sentinel, bit-aynılık) üzerinden sınanır.
    #[test]
    fn chunk_sinirda_doner_asimda_panik() {
        let v = [1u8, 2, 3, 4, 5];
        assert_eq!(chunk::<u8, 2>(&v, 3), &[4, 5]);
        assert!(std::panic::catch_unwind(|| chunk::<u8, 2>(&v, 4)).is_err());
        assert!(std::panic::catch_unwind(|| chunk::<u8, 1>(&v, 6)).is_err());
        let mut w = [0u8; 4];
        *chunk_mut::<u8, 2>(&mut w, 2) = [7, 8];
        assert_eq!(w, [0, 0, 7, 8]);
    }

    /// P36: çalışma zamanı uzunluklu maskeli yükleme yalnız ilk `n` baytı okur
    /// (dilimin hemen ötesindeki bayt 0xFF olsa da lane `n` sıfırdır) ve
    /// maskeli saklama yalnız ilk `n` baytı yazar (komşu nöbetçiler aynı).
    /// AVX512ICL yoksa atlanır (belirteç yok).
    #[test]
    fn p36_maskeli_yukleme_ve_saklama_siniri() {
        if !crate::cpu::CpuFlags::detect().contains(crate::cpu::CpuFlags::AVX512ICL) {
            return;
        }
        // SAFETY: (1) ISA: yukarıda AVX512ICL tespit edildi (AVX-512F/BW/VL
        // dahil). (2) sınır: yardımcılar dilim uzunluğunu kendileri denetler; çağrı
        // yalnız bu dizilerin içindedir. (3) hizalama/takma ad: diziler yereldir,
        // hizasız erişim yardımcıların kendisindedir.
        unsafe { p36_sinir_govdesi() }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    fn p36_sinir_govdesi() {
        use super::{load_u8_kn, store_u8_kn, store_u8x32_kn, store_u8x64};
        use core::arch::x86_64::{_mm_set1_epi8, _mm256_set1_epi8};
        let kaynak: [u8; 65] = core::array::from_fn(|i| if i < 40 { i as u8 + 1 } else { 0xFF });
        for n in 0..=40 {
            let mut b = [0u8; 64];
            store_u8x64(&mut b, load_u8_kn(&kaynak[..n], n));
            for (i, &v) in b.iter().enumerate() {
                assert_eq!(v, if i < n { i as u8 + 1 } else { 0 }, "n={n} lane {i}");
            }
        }
        for n in 0..=16 {
            let mut d = [0xA5u8; 18];
            store_u8_kn(&mut d[1..=n], _mm_set1_epi8(7), n);
            for (i, &v) in d.iter().enumerate() {
                assert_eq!(
                    v,
                    if (1..=n).contains(&i) { 7 } else { 0xA5 },
                    "n={n} bayt {i}"
                );
            }
        }
        for n in 0..=32 {
            let mut d = [0xA5u8; 34];
            store_u8x32_kn(&mut d[1..=n], _mm256_set1_epi8(7), n);
            for (i, &v) in d.iter().enumerate() {
                assert_eq!(
                    v,
                    if (1..=n).contains(&i) { 7 } else { 0xA5 },
                    "n={n} bayt {i}"
                );
            }
        }
    }
}

use crate::decode::refmvs::RefBlock;
use crate::decode::refmvs_tmvs::TemporalBlock;

/// P46: 4 adet `RefBlock`'u (`4 × 12 = 48` bayt, `#[repr(C)]`) ardışık üç
/// `__m128i` yazmacı (`3 × 16 = 48` bayt) ile yazar (`splat_mv` SSE2/AVX2/AVX512ICL).
#[target_feature(enable = "sse2")]
#[inline]
pub fn store_refblock_x4(dst: &mut [RefBlock; 4], m0: __m128i, m1: __m128i, m2: __m128i) {
    const { assert!(core::mem::size_of::<[RefBlock; 4]>() == 48) };
    let p = dst.as_mut_ptr().cast::<__m128i>();
    // SAFETY: (1) ISA: `_mm_storeu_si128` SSE2'dir; işlev SSE2 `#[target_feature]`
    // taşır. (2) sınır: `dst` tam `4 × 12 = 48` bayttır; üç `storeu_si128`
    // `[0..16)`, `[16..32)` ve `[32..48)` aralıklarına tam 48 bayt yazar,
    // dizi dışına taşmaz. (3) hizalama/takma ad: hizasız saklamadır, `RefBlock`
    // her bayt deseninde geçerli `#[repr(C)]` tam sayı alanlarından oluşur ve
    // `&mut` tek erişim yoludur.
    unsafe {
        _mm_storeu_si128(p, m0);
        _mm_storeu_si128(p.add(1), m1);
        _mm_storeu_si128(p.add(2), m2);
    }
}

/// P46: 8 adet `RefBlock`'u (`8 × 12 = 96` bayt, `#[repr(C)]`) ardışık üç
/// `__m256i` yazmacı (`3 × 32 = 96` bayt) ile yazar (`splat_mv` AVX2).
#[target_feature(enable = "avx2")]
#[inline]
pub fn store_refblock_x8_avx2(dst: &mut [RefBlock; 8], m0: __m256i, m1: __m256i, m2: __m256i) {
    const { assert!(core::mem::size_of::<[RefBlock; 8]>() == 96) };
    let p = dst.as_mut_ptr().cast::<__m256i>();
    // SAFETY: (1) ISA: `_mm256_storeu_si256` AVX'tir; işlev AVX2 `#[target_feature]`
    // taşır. (2) sınır: `dst` tam `8 × 12 = 96` bayttır; üç `storeu_si256`
    // `[0..32)`, `[32..64)` ve `[64..96)` aralıklarına tam 96 bayt yazar,
    // dizi dışına taşmaz. (3) hizalama/takma ad: hizasız saklamadır, `RefBlock`
    // her bayt deseninde geçerli `#[repr(C)]` tam sayı alanlarından oluşur ve
    // `&mut` tek erişim yoludur.
    unsafe {
        _mm256_storeu_si256(p, m0);
        _mm256_storeu_si256(p.add(1), m1);
        _mm256_storeu_si256(p.add(2), m2);
    }
}

/// P46: 16 adet `RefBlock`'u (`16 × 12 = 192` bayt, `#[repr(C)]`) ardışık üç
/// `__m512i` yazmacı (`3 × 64 = 192` bayt) ile yazar (`splat_mv` AVX512ICL).
#[target_feature(enable = "avx512f")]
#[inline]
pub fn store_refblock_x16_avx512(dst: &mut [RefBlock; 16], m0: __m512i, m1: __m512i, m2: __m512i) {
    const { assert!(core::mem::size_of::<[RefBlock; 16]>() == 192) };
    let p = dst.as_mut_ptr().cast::<__m512i>();
    // SAFETY: (1) ISA: `_mm512_storeu_si512` AVX-512F'dir; işlev bu özelliği
    // `#[target_feature]` ile taşır. (2) sınır: `dst` tam `16 × 12 = 192`
    // bayttır; üç `storeu_si512` `[0..64)`, `[64..128)` ve `[128..192)`
    // aralıklarına tam 192 bayt yazar, dizi dışına taşmaz. (3) hizalama/takma
    // ad: hizasız saklamadır, `RefBlock` her bayt deseninde geçerli `#[repr(C)]`
    // tam sayı alanlarından oluşur ve `&mut` tek erişim yoludur.
    unsafe {
        _mm512_storeu_si512(p, m0);
        _mm512_storeu_si512(p.add(1), m1);
        _mm512_storeu_si512(p.add(2), m2);
    }
}

/// P46: 4 adet `TemporalBlock`'u (`4 × 6 = 24` bayt, `#[repr(C)]`) bir `__m128i`
/// (`16` bayt) ve bir 64-bit (`8` bayt) saklama ile yazar (`save_tmvs` SSSE3/AVX2/AVX-512).
#[target_feature(enable = "sse2")]
#[inline]
pub fn store_temporal_x4(dst: &mut [TemporalBlock; 4], m0: __m128i, m1: __m128i) {
    const { assert!(core::mem::size_of::<[TemporalBlock; 4]>() == 24) };
    let p = dst.as_mut_ptr().cast::<u8>();
    // SAFETY: (1) ISA: `_mm_storeu_si128` ve `_mm_storel_epi64` SSE2'dir; işlev
    // SSE2 `#[target_feature]` taşır. (2) sınır: `dst` tam `4 × 6 = 24`
    // bayttır; `[0..16)` ve `[16..24)` aralıklarına tam 24 bayt yazılır, dizi
    // dışına taşmaz. (3) hizalama/takma ad: hizasız saklamadır, `TemporalBlock`
    // her bayt deseninde geçerli `#[repr(C)]` tam sayı alanlarından (`Mv` + `u8`
    // + sıfırlanmış dolgu baytı) oluşur ve `&mut` tek erişim yoludur.
    unsafe {
        _mm_storeu_si128(p.cast::<__m128i>(), m0);
        _mm_storel_epi64(p.add(16).cast::<__m128i>(), m1);
    }
}

/// P46: 8 adet `TemporalBlock`'u (`8 × 6 = 48` bayt, `#[repr(C)]`) ardışık üç
/// `__m128i` yazmacı (`3 × 16 = 48` bayt) ile yazar (`save_tmvs` SSSE3/AVX2/AVX-512).
#[target_feature(enable = "sse2")]
#[inline]
pub fn store_temporal_x8(dst: &mut [TemporalBlock; 8], m0: __m128i, m1: __m128i, m2: __m128i) {
    const { assert!(core::mem::size_of::<[TemporalBlock; 8]>() == 48) };
    let p = dst.as_mut_ptr().cast::<__m128i>();
    // SAFETY: (1) ISA: `_mm_storeu_si128` SSE2'dir; işlev SSE2 `#[target_feature]`
    // taşır. (2) sınır: `dst` tam `8 × 6 = 48` bayttır; üç `storeu_si128`
    // `[0..16)`, `[16..32)` ve `[32..48)` aralıklarına tam 48 bayt yazar,
    // dizi dışına taşmaz. (3) hizalama/takma ad: hizasız saklamadır, `TemporalBlock`
    // her bayt deseninde geçerli `#[repr(C)]` tam sayı alanlarından oluşur ve
    // `&mut` tek erişim yoludur.
    unsafe {
        _mm_storeu_si128(p, m0);
        _mm_storeu_si128(p.add(1), m1);
        _mm_storeu_si128(p.add(2), m2);
    }
}

/// P46: 4 i32'yi (`16` bayt) hizasız yükler (`unsafe` yok: `load_u8x16` üzerinden).
#[target_feature(enable = "sse2")]
#[inline]
#[must_use]
pub fn load_i32x4(src: &[i32; 4]) -> __m128i {
    let mut b = [0u8; 16];
    let mut i = 0;
    while i < 4 {
        let le = src[i].to_le_bytes();
        b[4 * i] = le[0];
        b[4 * i + 1] = le[1];
        b[4 * i + 2] = le[2];
        b[4 * i + 3] = le[3];
        i += 1;
    }
    load_u8x16(&b)
}

/// P46: 4 i32'yi (`16` bayt) hizasız saklar (`unsafe` yok: `store_u8x16` üzerinden).
#[target_feature(enable = "sse2")]
#[inline]
pub fn store_i32x4(dst: &mut [i32; 4], v: __m128i) {
    let mut b = [0u8; 16];
    store_u8x16(&mut b, v);
    let mut i = 0;
    while i < 4 {
        dst[i] = i32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        i += 1;
    }
}

/// P46: 8 i32'yi (`32` bayt) hizasız yükler (`unsafe` yok: `load_u8x32` üzerinden).
#[target_feature(enable = "avx2")]
#[inline]
#[must_use]
pub fn load_i32x8(src: &[i32; 8]) -> __m256i {
    let mut b = [0u8; 32];
    let mut i = 0;
    while i < 8 {
        let le = src[i].to_le_bytes();
        b[4 * i] = le[0];
        b[4 * i + 1] = le[1];
        b[4 * i + 2] = le[2];
        b[4 * i + 3] = le[3];
        i += 1;
    }
    load_u8x32(&b)
}

/// P46: 16 i32'yi (`64` bayt) hizasız yükler (`unsafe` yok: `load_u8x64` üzerinden).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
#[must_use]
pub fn load_i32x16(src: &[i32; 16]) -> __m512i {
    let mut b = [0u8; 64];
    let mut i = 0;
    while i < 16 {
        let le = src[i].to_le_bytes();
        b[4 * i] = le[0];
        b[4 * i + 1] = le[1];
        b[4 * i + 2] = le[2];
        b[4 * i + 3] = le[3];
        i += 1;
    }
    load_u8x64(&b)
}

/// P46: 16 i32'yi (`64` bayt) hizasız saklar (`unsafe` yok: `store_u8x64` üzerinden).
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
pub fn store_i32x16(dst: &mut [i32; 16], v: __m512i) {
    let mut b = [0u8; 64];
    store_u8x64(&mut b, v);
    let mut i = 0;
    while i < 16 {
        dst[i] = i32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        i += 1;
    }
}

/// P47: 4 u16'yı (`8` bayt) alt 64-bit şeride yükler (`movq`, `unsafe` yok).
#[target_feature(enable = "sse2")]
#[inline]
#[must_use]
pub fn load_u16x4(src: &[u16; 4]) -> __m128i {
    let w = u64::from(src[0])
        | (u64::from(src[1]) << 16)
        | (u64::from(src[2]) << 32)
        | (u64::from(src[3]) << 48);
    core::arch::x86_64::_mm_cvtsi64_si128(w as i64)
}

/// P47: alt 64-bit şeridi 4 u16 (`8` bayt) olarak saklar (`movq`, `unsafe` yok).
#[target_feature(enable = "sse2")]
#[inline]
pub fn store_u16x4(dst: &mut [u16; 4], v: __m128i) {
    let w = core::arch::x86_64::_mm_cvtsi128_si64(v) as u64;
    dst[0] = w as u16;
    dst[1] = (w >> 16) as u16;
    dst[2] = (w >> 32) as u16;
    dst[3] = (w >> 48) as u16;
}

/// P47: 8 u16'yı (`16` bayt) hizasız yükler (`unsafe` yok: `load_u8x16` üzerinden).
#[target_feature(enable = "sse2")]
#[inline]
#[must_use]
pub fn load_u16x8(src: &[u16; 8]) -> __m128i {
    let mut b = [0u8; 16];
    let mut i = 0;
    while i < 8 {
        let le = src[i].to_le_bytes();
        b[2 * i] = le[0];
        b[2 * i + 1] = le[1];
        i += 1;
    }
    load_u8x16(&b)
}

/// P47: 8 u16'yı (`16` bayt) hizasız saklar (`unsafe` yok: `store_u8x16` üzerinden).
#[target_feature(enable = "sse2")]
#[inline]
pub fn store_u16x8(dst: &mut [u16; 8], v: __m128i) {
    let mut b = [0u8; 16];
    store_u8x16(&mut b, v);
    let mut i = 0;
    while i < 8 {
        dst[i] = u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
        i += 1;
    }
}

/// P47: 16 u16'yı (`32` bayt) hizasız yükler (`unsafe` yok: `load_u8x32` üzerinden).
#[target_feature(enable = "avx2")]
#[inline]
#[must_use]
pub fn load_u16x16(src: &[u16; 16]) -> __m256i {
    let mut b = [0u8; 32];
    let mut i = 0;
    while i < 16 {
        let le = src[i].to_le_bytes();
        b[2 * i] = le[0];
        b[2 * i + 1] = le[1];
        i += 1;
    }
    load_u8x32(&b)
}

/// P47: 16 u16'yı (`32` bayt) hizasız saklar (`unsafe` yok: `store_u8x32` üzerinden).
#[target_feature(enable = "avx2")]
#[inline]
pub fn store_u16x16(dst: &mut [u16; 16], v: __m256i) {
    let mut b = [0u8; 32];
    store_u8x32(&mut b, v);
    let mut i = 0;
    while i < 16 {
        dst[i] = u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
        i += 1;
    }
}
