//! CDEF AVX2 çekirdekleri — 8 bpc: S15.4 pilotu, F4-CDEF a (P28) yeniden yazımı;
//! 16 bpc (10 ve 12 bit): F4-CDEF b (P29).
//!
//! ## `filter`: dolgu kopyası yok (P28)
//!
//! S15.4 pilotu bloğu önce 192 öğelik bir i16 dolgu tamponuna skaler döngüyle
//! kopyalıyordu (dav1d `src/cdef_tmpl.c:56-102` C yolu). Bu gövde tampon kurmaz:
//! kaynak satırları (`top` 2, blok `h`, `bottom` 2) `dst`/`left`/`top`/`bottom`
//! dilimlerinden **vektör yüklemeleriyle** doğrudan okunur — her satır bir kez,
//! 16 baytlık bir yazmaçta (bayt `j` = mantıksal sütun `j − 2`). Bir tapın
//! komşuları satır yazmacından tek `pshufb` ile alınır: aynı maske baytları
//! seçer **ve** `u8 → i16` genişletir. 8 genişlikte iki satır, 4 genişlikte dört
//! satır tek 256 bit yazmaçtadır (ADR-0003 §2.6 madde 4–5).
//!
//! Okuma sözleşmesi değişmedi ([`crate::dsp::cdef`] modül belgesi): `dst`
//! satırlarında sütun `0 .. w+2`, `left` `2h` bayt, `top`/`bottom`
//! `stride + w + 4` bayt; bütün yüklemeler bu aralıklarda denetimli dizi
//! referanslarıyla yapılır — dilim dışına 0 bayt.
//!
//! ## Kenarlar: geçersiz komşu = merkez piksel
//!
//! dav1d C yolu kenarda eksik pikseli `INT16_MIN` ile doldurur; bu değerin
//! `constrain` katkısı 0'dır ve min/max'i değiştirmez. Bu gövde eksik komşunun
//! yerine **o çıktı pikselinin kendi değerini** (`px`) koyar: `p − px = 0` ⟹
//! `constrain = 0` (katkı 0), `min(px, min) = min` ve `max(px, max) = max`
//! (min/max `px` ile başlar) — sonuç bit-aynıdır ve `damping` koşulundan
//! bağımsızdır. Geçerlilik, satır yazmacının eşi olan bir **maske yazmacından**
//! aynı `pshufb` ile alınır ve `blendv` ile uygulanır. Dört kenarın da
//! bulunduğu (çözme hattında en sık) durumda maske kodu derleme zamanında
//! kalkar (`EDGES = false`).
//!
//! ## `dir`: 64 piksel tek geçiş (P28)
//!
//! Sekiz satır bir kez yüklenir; kısmi toplamlar (`diag`, `alt`, `hv`)
//! bellekten geçmeden yazmaçlarda bayt kaydırmalarıyla (`pslldq`/`psrldq`)
//! birikir, maliyetler `pmaddwd` + `pmulld` ile hesaplanır ve sekiz yön
//! `phaddd` ile birleştirilir (dav1d `src/x86/cdef_avx2.asm` `cdef_dir`
//! yaklaşımı; kod kopyalanmadı).
//!
//! ## 16 bpc (P29)
//!
//! Aynı yöntem, `u16` piksellerle: dolgu tamponu yok, satırlar dilimlerden
//! doğrudan (`load_u8x16`, `cvtsi64`, `pinsrd`), eksik kenar = merkez piksel
//! (maske). Piksel zaten 16 bit olduğundan genişletme yoktur; satır kaynağı 12
//! (`W = 8`) ya da 8 (`W = 4`) piksel, yani 24/16 bayttır ve bir tap penceresi
//! iki `pshufb`'un `or`'udur (8 bpc'de tek `pshufb`). Tap listesi yön başına
//! derleme zamanı tablosundadır (`TAPS16`); aynı katsayılı tap'ların
//! `constrain` toplamı bir kez çarpılır, yuvarlama `pmulhrsw` ile (dav1d
//! `src/x86/cdef16_avx2.asm` düzeni; kod kopyalanmadı). `i16` yeterlidir:
//! `|sum| ≤ 12·240 + 12·64 = 3 648` (12 bit, en büyük güç). `dir` satırları
//! `(px >> bitdepth_min_8) − 128` ile 8 bit'e indirgenir ve 8 bpc maliyet gövdesi
//! (`dir_maliyet!`) aynen kullanılır (dav1d AVX2 de 16 bpc'de 8 bpc gövdesini
//! çağırır); 8 bpc gövdesinin makine kodu bu paylaşımla değişmedi.
//!
//! ## Yetki ve `unsafe` (F3b / P25, ADR-0003 §2.3-2.4)
//!
//! - **ISA yetkisi kurulumdadır:** AVX2 gövdelerine giden yol yalnız bir
//!   [`Avx2Token`] ile açılır — [`find_dir`]/[`filter`] belirteci parametre
//!   olarak ister, dispatch yuvası işaretçisi [`find_dir_fn`]/[`filter_fn`]
//!   ile yalnız belirteçle verilir. Çağrı başına `is_x86_feature_detected!`
//!   yoktur; O(1) uzunluk/stride/parametre denetimi her çağrıda kalır.
//! - **Çekirdekler güvenli `#[target_feature]` işlevleridir** (target_feature
//!   1.1): işaretçisiz intrinsic'ler güvenlidir; yüklemeler tamsayı
//!   dönüşümleriyle (`from_le_bytes` + `cvtsi64`), saklamalar [`super::mem`]
//!   yardımcılarıyla, sınırı denetlenen dizi referanslarıyla yapılır.
//! - Bu dosyada `unsafe` yalnız dört girişteki **tek çağrıdır**
//!   (`find_dir_entry`, `filter_entry`, `find_dir16_entry`, `filter16_entry`);
//!   dosya `audit::UNSAFE_EXEMPTIONS` defterinde kayıtlıdır.
//!
//! ```compile_fail,E0061
//! // Belirteçsiz çekirdek çağrısı derlenmez.
//! let img = [0u8; 64];
//! let _ = av1oxide::dsp::x86::cdef::find_dir(
//!     &img, 8, av1oxide::dsp::cdef::CdefBitDepth::B8,
//! );
//! ```

// Döngüler dav1d kaynağındaki indekslemeyi **birebir** yansıtır
// (`cdef_dirs[k]`); denetim izi bu yüzden korunur.
#![allow(clippy::needless_range_loop)]
#![allow(unsafe_code)]

use super::mem::{
    chunk, chunk_mut, load_i16x16, load_u8x16, store_i32x8, store_u8x4, store_u8x8, store_u8x16,
};
use crate::cpu::Avx2Token;
use crate::dsp::cdef::{
    CDEF_DIRECTIONS, CdefBitDepth, CdefDirFn, CdefDirResult, CdefDsp, CdefEdgeFlags, CdefFilterFn,
    CdefFilterParams, DIR_CENTER_WEIGHT, DIV_TABLE,
};
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_cvtepu8_epi16, _mm_cvtsi32_si128, _mm_cvtsi64_si128,
    _mm_hadd_epi16, _mm_insert_epi16, _mm_insert_epi32, _mm_set1_epi16, _mm_setr_epi8,
    _mm_setzero_si128, _mm_slli_si128, _mm_srl_epi16, _mm_srli_si128, _mm_sub_epi16,
    _mm_unpacklo_epi64, _mm256_abs_epi16, _mm256_add_epi16, _mm256_add_epi32, _mm256_blendv_epi8,
    _mm256_bslli_epi128, _mm256_bsrli_epi128, _mm256_castsi256_ps, _mm256_castsi256_si128,
    _mm256_cmpeq_epi32, _mm256_cmpgt_epi16, _mm256_extracti128_si256, _mm256_hadd_epi16,
    _mm256_hadd_epi32, _mm256_madd_epi16, _mm256_max_epi16, _mm256_max_epu32, _mm256_min_epi16,
    _mm256_min_epu16, _mm256_movemask_ps, _mm256_mulhrs_epi16, _mm256_mullo_epi16,
    _mm256_mullo_epi32, _mm256_or_si256, _mm256_permute2x128_si256, _mm256_permutevar8x32_epi32,
    _mm256_set_m128i, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi8, _mm256_setr_epi32,
    _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_shuffle_epi32, _mm256_sign_epi16,
    _mm256_slli_epi16, _mm256_srai_epi16, _mm256_srl_epi16, _mm256_srli_epi16, _mm256_srli_epi32,
    _mm256_sub_epi16, _mm256_sub_epi32, _mm256_subs_epu16, _mm256_unpackhi_epi16,
    _mm256_unpacklo_epi16,
};

/// Kaynak satır sayısı en çok `h + 4 = 12` (mantıksal −2 .. h+1).
pub(super) const MAX_ROWS: usize = 12;

/// Bir tap: kaynak satır kayması, sütun kayması indeksi ve katsayı.
#[derive(Clone, Copy, Default)]
pub(super) struct Tap {
    /// Satır kayması `a + 2 ∈ [0, 4]`.
    pub(super) row: usize,
    /// Sütun kayması `b + 2 ∈ [0, 4]` (seçme maskesi tablosu indeksi).
    pub(super) bi: usize,
    /// Tap katsayısı (`pri_tap` ya da `sec_tap`).
    pub(super) k: i16,
}

/// Yön tablosunun `12` tabanlı paketlenmiş ofsetini `(satır, sütun)`'a çözer.
///
/// `off = a*12 + b`, `b ∈ [−2, 2]`. `a = (off + 2) / 12` (Öklid bölmesi)
/// seçilir çünkü `b + 2 ∈ [0, 4] ⊂ [0, 12)` olur; sonra `b = off − a*12`.
const fn unpack_offset(off: i16) -> (i32, i32) {
    let off = off as i32;
    let a = (off + 2).div_euclid(12);
    let b = off - a * 12;
    assert!(a >= -2 && a <= 2 && b >= -2 && b <= 2, "ofset aralık dışı");
    (a, b)
}

/// [`CDEF_DIRECTIONS`]'ın çözülmüş `(a, b)` biçimi (derleme zamanı; çağrı
/// başına bölme yok).
pub(super) const DIR_AB: [[(i32, i32); 2]; 12] = {
    let mut t = [[(0, 0); 2]; 12];
    let mut i = 0;
    while i < 12 {
        t[i][0] = unpack_offset(CDEF_DIRECTIONS[i][0]);
        t[i][1] = unpack_offset(CDEF_DIRECTIONS[i][1]);
        i += 1;
    }
    t
};

/// Güç değerinin `ulog2`'si (dav1d `intops.h:67`); `v > 0` beklenir.
#[must_use]
pub(super) fn ilog2(v: i32) -> i32 {
    debug_assert!(v > 0, "ulog2(0) tanımsız");
    31 - (v as u32).leading_zeros() as i32
}

/// Belirteçli `dir` girişi: AVX2 gövdesi yalnız [`Avx2Token`] ile çağrılır.
///
/// Sözleşme dışı girdide (16bpc, kısa tampon) skaler yola düşer.
pub fn find_dir(
    _avx2: Avx2Token,
    img: &[u8],
    stride: usize,
    bit_depth: CdefBitDepth,
) -> CdefDirResult {
    find_dir_entry(img, stride, bit_depth)
}

/// Dispatch yuvası işaretçisi (`dir`): yalnız [`Avx2Token`] ile verilir.
#[must_use]
pub const fn find_dir_fn(_avx2: Avx2Token) -> CdefDirFn {
    find_dir_entry
}

/// Belirteçli `filter` girişi: AVX2 gövdesi yalnız [`Avx2Token`] ile çağrılır.
///
/// Sözleşme dışı girdide skaler yola düşer. `damping > 6` durumu da skaler
/// yola düşer (AV1'de `damping ∈ [3,6]`, `cdef_damping ∈ [0,3]` + 3); gövde
/// bu sınıra dayanmaz ama S15.4 sözleşmesi korunur.
pub fn filter(
    _avx2: Avx2Token,
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    filter_entry(dst, stride, left, top, bottom, params);
}

/// Dispatch yuvası işaretçisi (`filter`): yalnız [`Avx2Token`] ile verilir.
#[must_use]
pub const fn filter_fn(_avx2: Avx2Token) -> CdefFilterFn {
    filter_entry
}

/// `dir` girişi. Adresi yalnız [`find_dir`]/[`find_dir_fn`] (belirteçli) ile
/// dışarı verilir.
fn find_dir_entry(img: &[u8], stride: usize, bit_depth: CdefBitDepth) -> CdefDirResult {
    if bit_depth != CdefBitDepth::B8 || img.len() < 7 * stride + 8 {
        // Sözleşme dışı / 16bpc: o bit derinliğinin skaler monomorfu.
        return (CdefDsp::scalar_for(bit_depth).dir)(img, stride, bit_depth);
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`find_dir`, `find_dir_fn`); belirteç AVX2 tespiti + maske izninin
    // kanıtıdır (`cpu::token`). (2) sınır: `img.len() >= 7*stride + 8`; gövdenin
    // en büyük okuması `img[7*stride .. 7*stride + 8]`'dir ve ayrıca denetlenir.
    // (3) hizalama/takma ad: gövde `img`'yi yalnız okur, yerel dizilere yazar.
    unsafe { find_dir_avx2(img, stride) }
}

/// `filter` girişi. Adresi yalnız [`filter`]/[`filter_fn`] (belirteçli) ile
/// dışarı verilir.
fn filter_entry(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    let w = params.size.w();
    let h = params.size.h();
    // Sözleşme kontrolleri: yalnız **okuma** sınırlarını güvenceye alır.
    // Eksik kenarın dilimi okunmaz (çözme hattı onu boş geçirir); varsa tam
    // satır aralığı gerekir.
    let e = params.edges;
    let ok = params.bit_depth == CdefBitDepth::B8
        && params.damping <= 6
        && params.dir < 8
        && stride >= w + 4
        && dst.len() >= (h - 1) * stride + w + 2
        && (!e.contains(CdefEdgeFlags::HAVE_LEFT) || left.len() >= h * 2)
        && (!e.contains(CdefEdgeFlags::HAVE_TOP) || top.len() >= stride + w + 4)
        && (!e.contains(CdefEdgeFlags::HAVE_BOTTOM) || bottom.len() >= stride + w + 4);
    if !ok {
        let fb = CdefDsp::scalar_for(params.bit_depth).fb[params.size.index()];
        fb(dst, stride, left, top, bottom, params);
        return;
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`filter`, `filter_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır.
    // (2) sınır: yukarıdaki denetimler gövdenin okuduğu her baytı kapsar (dst
    // satırları sütun `0..w+2`; bayrağı olan kenarda top/bottom
    // `row*stride + w + 4`, left `2h`; bayraksız kenar dilimi okunmaz);
    // gövdenin yüklemeleri bu aralıklarda denetimli dizi referanslarıdır.
    // (3) hizalama/takma ad: `dst` tek `&mut`, kaynaklar `&`; ödünç kuralları
    // çakışmayı dışlar.
    unsafe { filter_avx2(dst, stride, left, top, bottom, params) }
}

// ---------------------------------------------------------------------------
// dir
// ---------------------------------------------------------------------------

/// Bir satırın 8 pikseli, `i16` lane'lerde `px − 128` (dav1d `cdef_tmpl.c:249`).
#[target_feature(enable = "avx2")]
#[inline]
fn dir_row(img: &[u8], off: usize) -> __m128i {
    let raw = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(img, off)));
    _mm_sub_epi16(_mm_cvtepu8_epi16(raw), _mm_set1_epi16(128))
}

/// `(lo, hi) += v`'nin `N` lane (bayt `B = 2N`) kaydırılmış katkısı, iki 128
/// bit yarıda ayrı ayrı: `lo` lane `n` ← `v[n − N]`, `hi` lane `n` ←
/// `v[n + 8 − N]` (`R = 16 − 2N`; immediate'ler derleme zamanı sabitidir).
#[target_feature(enable = "avx2")]
#[inline]
fn acc_shift<const B: i32, const R: i32>(lo: &mut __m256i, hi: &mut __m256i, v: __m256i) {
    *lo = _mm256_add_epi16(*lo, _mm256_bslli_epi128::<B>(v));
    *hi = _mm256_add_epi16(*hi, _mm256_bsrli_epi128::<R>(v));
}

/// Her 128 bit yarıda `Σ (lo[n]² + hi[n]²) · w[n]` (8 lane), yarı başına
/// 4 × i32 kısmi toplam (sarmalı; C `unsigned` semantiği).
#[target_feature(enable = "avx2")]
#[inline]
fn cost_terms(lo: __m256i, hi: __m256i, w_lo: __m256i, w_hi: __m256i) -> __m256i {
    let a = _mm256_unpacklo_epi16(lo, hi);
    let b = _mm256_unpackhi_epi16(lo, hi);
    _mm256_add_epi32(
        _mm256_mullo_epi32(_mm256_madd_epi16(a, a), w_lo),
        _mm256_mullo_epi32(_mm256_madd_epi16(b, b), w_hi),
    )
}

/// 8x8 blok üzerinde yön taraması (AVX2) — dav1d `cdef_find_dir_c` ile bit-exact.
///
/// Yazmaç düzeni (iki yön tek 256 bit yazmaçta):
/// - `x_y = [r_y | ters(r_y)]`: alt yarı `diag[0][y + x]`, üst yarı
///   `diag[1][7 + y − x]` — ikisi de `y` lane kaydırmasıyla birikir;
/// - `hadd(x_y, 0) = [r_y ikilileri | ters ikililer]`: `alt[0][y + (x>>1)]` ve
///   `alt[1][3 + y − (x>>1)]` — yine `y` kaydırması;
/// - `s_m = r_{2m} + r_{2m+1}`, `[s_m | s_{3−m}]` `3 − m` kaydırmasıyla:
///   alt yarı `alt[2][3 − m + x]`, üst yarı `alt[3][m' + x]` (`m' = 3 − m`).
///
/// Güvenli `#[target_feature]` işlevidir; `img` okumaları dizi referanslarıyla
/// denetlenir (`y*stride + 8 <= img.len()`).
/// 8 satırdan (`i16` lane'lerde `px − 128`) yön sonucu: 8 ve 16 bpc'nin ortak
/// maliyet gövdesi (dav1d `src/cdef_tmpl.c:247-303`; dav1d AVX2 de 16 bpc'de 8
/// bpc maliyet gövdesini çağırır). Makro, 8 bpc gövdesinin token dizisini
/// değiştirmeden paylaşmak içindir (P29: 8 bpc makine kodu aynı kalır).
macro_rules! dir_maliyet {
    ($r0:ident, $r1:ident, $r2:ident, $r3:ident, $r4:ident, $r5:ident, $r6:ident, $r7:ident) => {{
        let (r0, r1, r2, r3, r4, r5, r6, r7) = ($r0, $r1, $r2, $r3, $r4, $r5, $r6, $r7);

        let z = _mm256_setzero_si256();
        // 8 i16 lane'i ters çeviren maske (x → 7 − x), alt yarıda kimlik.
        let rev = _mm256_setr_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 15, 12, 13, 10, 11, 8, 9, 6,
            7, 4, 5, 2, 3, 0, 1,
        );

        // ---- kısmi toplamlar (dav1d `src/cdef_tmpl.c:247-260`) ---------------
        let (mut dl, mut dh, mut al, mut ah) = (z, z, z, z);
        macro_rules! satir {
            ($r:expr, $b:literal, $rr:literal) => {{
                let x = _mm256_shuffle_epi8(_mm256_set_m128i($r, $r), rev);
                acc_shift::<$b, $rr>(&mut dl, &mut dh, x);
                acc_shift::<$b, $rr>(&mut al, &mut ah, _mm256_hadd_epi16(x, z));
            }};
        }
        satir!(r0, 0, 16);
        satir!(r1, 2, 14);
        satir!(r2, 4, 12);
        satir!(r3, 6, 10);
        satir!(r4, 8, 8);
        satir!(r5, 10, 6);
        satir!(r6, 12, 4);
        satir!(r7, 14, 2);
        let s0 = _mm_add_epi16(r0, r1);
        let s1 = _mm_add_epi16(r2, r3);
        let s2 = _mm_add_epi16(r4, r5);
        let s3 = _mm_add_epi16(r6, r7);
        let (mut bl, mut bh) = (z, z);
        acc_shift::<6, 10>(&mut bl, &mut bh, _mm256_set_m128i(s3, s0));
        acc_shift::<4, 12>(&mut bl, &mut bh, _mm256_set_m128i(s2, s1));
        acc_shift::<2, 14>(&mut bl, &mut bh, _mm256_set_m128i(s1, s2));
        acc_shift::<0, 16>(&mut bl, &mut bh, _mm256_set_m128i(s0, s3));
        // hv: satır toplamları (hadd ağacı; |Σ| ≤ 1024 i16'ya sığar) ve sütunlar.
        let hv1 = _mm_add_epi16(_mm_add_epi16(s0, s1), _mm_add_epi16(s2, s3));
        let hv0 = _mm_hadd_epi16(
            _mm_hadd_epi16(_mm_hadd_epi16(r0, r1), _mm_hadd_epi16(r2, r3)),
            _mm_hadd_epi16(_mm_hadd_epi16(r4, r5), _mm_hadd_epi16(r6, r7)),
        );

        // ---- maliyetler (dav1d `src/cdef_tmpl.c:263-292`) ---------------------
        // diag: (d[n]² + d[14−n]²)·div[n] (n < 7), d[7]²·105 — `hi` lane n ←
        // hi[6 − n], son lane 0. alt: (a[n]² + a[10−n]²)·div[2n+1] (n < 3),
        // a[n]²·105 — `hi` lane n ← hi[2 − n], kalanı 0.
        let rev7 = _mm256_setr_epi8(
            12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, -128, -128, 12, 13, 10, 11, 8, 9, 6, 7,
            4, 5, 2, 3, 0, 1, -128, -128,
        );
        let rev3 = _mm256_setr_epi8(
            4, 5, 2, 3, 0, 1, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 4, 5, 2,
            3, 0, 1, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
        );
        let d = |i: usize| DIV_TABLE[i] as i32;
        let c = DIR_CENTER_WEIGHT as i32;
        let w_d_lo = _mm256_setr_epi32(d(0), d(1), d(2), d(3), d(0), d(1), d(2), d(3));
        let w_d_hi = _mm256_setr_epi32(d(4), d(5), d(6), c, d(4), d(5), d(6), c);
        let w_a_lo = _mm256_setr_epi32(d(1), d(3), d(5), c, d(1), d(3), d(5), c);
        let w_c = _mm256_set1_epi32(c);
        let c04 = cost_terms(dl, _mm256_shuffle_epi8(dh, rev7), w_d_lo, w_d_hi); // [c0 | c4]
        let c13 = cost_terms(al, _mm256_shuffle_epi8(ah, rev3), w_a_lo, w_c); // [c1 | c3]
        let c57 = cost_terms(bl, _mm256_shuffle_epi8(bh, rev3), w_a_lo, w_c); // [c5 | c7]
        let hv = _mm256_set_m128i(hv1, hv0);
        let c26 = _mm256_mullo_epi32(_mm256_madd_epi16(hv, hv), w_c); // [c2 | c6]
        // Yarı başına yatay toplam: alt [c0, c1, c2, c5], üst [c4, c3, c6, c7].
        let t = _mm256_hadd_epi32(_mm256_hadd_epi32(c04, c13), _mm256_hadd_epi32(c26, c57));
        // Doğal sıraya diz: lane n = cost[n].
        let cost = _mm256_permutevar8x32_epi32(t, _mm256_setr_epi32(0, 1, 2, 5, 4, 3, 6, 7));

        // En iyi yön dalsız (dav1d `src/cdef_tmpl.c:294-301`, katı `>`): en büyük
        // (işaretsiz) maliyet, ona eşit **ilk** lane. `cost[dir ^ 4]` = yarıları
        // takas edilmiş vektörün aynı lane'i.
        let m = _mm256_max_epu32(cost, _mm256_permute2x128_si256::<0x01>(cost, cost));
        let m = _mm256_max_epu32(m, _mm256_shuffle_epi32::<0x4e>(m));
        let m = _mm256_max_epu32(m, _mm256_shuffle_epi32::<0xb1>(m));
        let eq = _mm256_cmpeq_epi32(cost, m);
        let best_dir =
            (_mm256_movemask_ps(_mm256_castsi256_ps(eq)) as u32).trailing_zeros() as usize;
        let swapped = _mm256_permute2x128_si256::<0x01>(cost, cost);
        let mut var = [0i32; 8];
        store_i32x8(
            &mut var,
            _mm256_srli_epi32::<10>(_mm256_sub_epi32(cost, swapped)),
        );
        CdefDirResult {
            dir: best_dir as u8,
            var: var[best_dir & 7] as u32,
        }
    }};
}

#[target_feature(enable = "avx2")]
fn find_dir_avx2(img: &[u8], stride: usize) -> CdefDirResult {
    let r0 = dir_row(img, 0);
    let r1 = dir_row(img, stride);
    let r2 = dir_row(img, 2 * stride);
    let r3 = dir_row(img, 3 * stride);
    let r4 = dir_row(img, 4 * stride);
    let r5 = dir_row(img, 5 * stride);
    let r6 = dir_row(img, 6 * stride);
    let r7 = dir_row(img, 7 * stride);
    dir_maliyet!(r0, r1, r2, r3, r4, r5, r6, r7)
}

// ---------------------------------------------------------------------------
// filter
// ---------------------------------------------------------------------------

/// `s[off..off+N]` küçük sonlu tamsayısı (denetimli dizi referansıyla).
macro_rules! le {
    ($t:ty, $n:literal, $s:expr, $off:expr) => {
        <$t>::from_le_bytes(*chunk::<u8, $n>($s, $off))
    };
}

/// Kenar satırının (`top`/`bottom`, indeks 0 = sütun −2) yazmacı: sütun
/// `−2 .. w+2` (8 bayt; `W = 8` iken + 4 bayt).
#[target_feature(enable = "avx2")]
#[inline]
fn edge_row<const W: usize>(s: &[u8], o: usize) -> __m128i {
    let v = _mm_cvtsi64_si128(le!(i64, 8, s, o));
    if W == 8 {
        _mm_insert_epi32::<2>(v, le!(i32, 4, s, o + 8))
    } else {
        v
    }
}

/// Blok satırının yazmacı: sütun −2, −1 `left`'ten (`l`), `0 .. w+2` `dst`'den.
///
/// `W = 8`: sütun 0..7 (8 bayt) + 8, 9 (2 bayt) — `o + 10 ≤ (h−1)·stride + w + 2`.
/// `W = 4`: sütun 0..3 (4 bayt) + 4, 5 (2 bayt) — `o + 6 ≤ (h−1)·stride + w + 2`.
#[target_feature(enable = "avx2")]
#[inline]
fn block_row<const W: usize>(dst: &[u8], o: usize, l: i32) -> __m128i {
    if W == 8 {
        let v = _mm_slli_si128::<2>(_mm_cvtsi64_si128(le!(i64, 8, dst, o)));
        let v = _mm_insert_epi16::<0>(v, l);
        _mm_insert_epi16::<5>(v, i32::from(le!(u16, 2, dst, o + 8)))
    } else {
        let v = _mm_slli_si128::<2>(_mm_cvtsi32_si128(le!(i32, 4, dst, o)));
        let v = _mm_insert_epi16::<0>(v, l);
        _mm_insert_epi16::<3>(v, i32::from(le!(u16, 2, dst, o + 4)))
    }
}

/// Tap seçme maskeleri, sütun kayması `b ∈ [−2, 2]` (indeks `b + 2`); her
/// 128 bit yarıda aynı. `W = 8`: lane `x` ← bayt `b + 2 + x` (bir satır).
/// `W = 4`: lane `x < 4` ← bayt `b + 2 + x`, lane `4 + x` ← bayt
/// `8 + b + 2 + x` (yarıda iki satır). `dup = false`: üst bayt `0x80` → 0
/// (`u8 → i16`); `dup = true`: üst bayt aynı bayt (geçerlilik `0xFFFF`/0).
pub(super) const fn sel_table(w: usize, dup: bool) -> [[i16; 16]; 5] {
    let mut t = [[0i16; 16]; 5];
    let mut bi = 0;
    while bi < 5 {
        let mut half = 0;
        while half < 2 {
            let mut x = 0;
            while x < 8 {
                let src = if w == 8 || x < 4 {
                    bi + x
                } else {
                    8 + bi + x - 4
                };
                let hi = if dup { src } else { 0x80 };
                t[bi][half * 8 + x] = (src | (hi << 8)) as u16 as i16;
                x += 1;
            }
            half += 1;
        }
        bi += 1;
    }
    t
}

/// [`sel_table`] örnekleri (derleme zamanı).
const SEL8: [[i16; 16]; 5] = sel_table(8, false);
const VSEL8: [[i16; 16]; 5] = sel_table(8, true);
const SEL4: [[i16; 16]; 5] = sel_table(4, false);
const VSEL4: [[i16; 16]; 5] = sel_table(4, true);

/// 8-bit CDEF filtresi (AVX2) — dav1d `cdef_filter_block_c` ile bit-exact.
///
/// Güvenli `#[target_feature]` işlevidir. Doğruluk için çağıranın sözleşmesi
/// (`filter_entry`): `stride >= w+4`, `dst.len() >= (h-1)*stride + w + 2`,
/// bayrağı olan kenarda `left.len() >= h*2`, `top/bottom.len() >= stride + w
/// + 4`; 8-bit, `damping <= 6`, `dir < 8`; bellek erişimleri ayrıca denetimlidir.
#[target_feature(enable = "avx2")]
fn filter_avx2(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    let pri = i32::from(params.pri_strength);
    let sec = i32::from(params.sec_strength);
    if pri == 0 && sec == 0 {
        return; // sözleşme dışı; skaler yolla aynı karar (dst değişmez)
    }
    let all = params.edges.bits() == 0b1111;
    match (params.size.w(), all, pri != 0 && sec != 0) {
        (8, true, true) => filter_w::<8, false, true>(dst, stride, left, top, bottom, params),
        (8, true, false) => filter_w::<8, false, false>(dst, stride, left, top, bottom, params),
        (8, false, true) => filter_w::<8, true, true>(dst, stride, left, top, bottom, params),
        (8, false, false) => filter_w::<8, true, false>(dst, stride, left, top, bottom, params),
        (_, true, true) => filter_w::<4, false, true>(dst, stride, left, top, bottom, params),
        (_, true, false) => filter_w::<4, false, false>(dst, stride, left, top, bottom, params),
        (_, false, true) => filter_w::<4, true, true>(dst, stride, left, top, bottom, params),
        (_, false, false) => filter_w::<4, true, false>(dst, stride, left, top, bottom, params),
    }
}

/// Genişlik `W`, kenar maskesi `EDGES`, kırpma `MINMAX` için gövde.
///
/// `W = 8`: yazmaç = iki satır (`y`, `y+1`); `W = 4`: dört satır (alt yarı
/// `y`, `y+1`; üst yarı `y+2`, `y+3`). Monomorflar `filter_avx2` içine satır
/// içi açılır: P28'de `inline(never)` denendi, skaler kipte yerleşim etkisiyle
/// vq002 +%1,9 (satır içi: +%0,5) ve 8×8 çekirdek +%2 ölçüldü; geri alındı.
#[target_feature(enable = "avx2")]
fn filter_w<const W: usize, const EDGES: bool, const MINMAX: bool>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    let h = params.size.h();
    let n_rows = h + 4;
    let e = params.edges;
    let has_top = !EDGES || e.contains(CdefEdgeFlags::HAVE_TOP);
    let has_bottom = !EDGES || e.contains(CdefEdgeFlags::HAVE_BOTTOM);
    let has_left = !EDGES || e.contains(CdefEdgeFlags::HAVE_LEFT);

    // ---- 1) kaynak satırları (dolgu yok: doğrudan dilimlerden) ------------
    // rows[i] = mantıksal satır i − 2. Bayrağı olmayan kenarın dilimi okunmaz.
    let z128 = _mm_setzero_si128();
    let mut rows = [z128; MAX_ROWS];
    if has_top {
        rows[0] = edge_row::<W>(top, 0);
        rows[1] = edge_row::<W>(top, stride);
    }
    for r in 0..h {
        let l = if has_left {
            i32::from(le!(u16, 2, left, 2 * r))
        } else {
            0
        };
        rows[r + 2] = block_row::<W>(dst, r * stride, l);
    }
    if has_bottom {
        rows[h + 2] = edge_row::<W>(bottom, 0);
        rows[h + 3] = edge_row::<W>(bottom, stride);
    }

    // Grup kaynakları bir kez kurulur (tap başına `vinserti128` yok):
    //   W = 8 → srcs[i] = [rows[i] | rows[i+1]]
    //   W = 4 → srcs[i] = [rows[i] ‖ rows[i+1] | rows[i+2] ‖ rows[i+3]]
    let z = _mm256_setzero_si256();
    let n_src = if W == 8 { n_rows - 1 } else { n_rows - 3 };
    let pack = |rs: &[__m128i; MAX_ROWS], i: usize| -> __m256i {
        if W == 8 {
            _mm256_set_m128i(rs[i + 1], rs[i])
        } else {
            _mm256_set_m128i(
                _mm_unpacklo_epi64(rs[i + 2], rs[i + 3]),
                _mm_unpacklo_epi64(rs[i], rs[i + 1]),
            )
        }
    };
    let mut srcs = [z; MAX_ROWS];
    for (i, v) in srcs.iter_mut().enumerate().take(n_src) {
        *v = pack(&rows, i);
    }
    // Geçerlilik (yalnız kenar yolunda): bayt 0xFF = komşu yok.
    let mut vsrcs = [z; MAX_ROWS];
    if EDGES {
        let mut col = [0i8; 16];
        if !e.contains(CdefEdgeFlags::HAVE_LEFT) {
            col[0] = -1;
            col[1] = -1;
        }
        if !e.contains(CdefEdgeFlags::HAVE_RIGHT) {
            for c in col.iter_mut().skip(W + 2) {
                *c = -1;
            }
        }
        let colv = v16(&col);
        let allv = v16(&[-1i8; 16]);
        let mut vrows = [z128; MAX_ROWS];
        for (i, v) in vrows.iter_mut().enumerate().take(n_rows) {
            let missing = (i < 2 && !has_top) || (i >= h + 2 && !has_bottom);
            *v = if missing { allv } else { colv };
        }
        for (i, v) in vsrcs.iter_mut().enumerate().take(n_src) {
            *v = pack(&vrows, i);
        }
    }

    // ---- 2) tap listesi (küçük indeksler; eşik/kaydırma grup başına) -------
    let (sel, vsel) = if W == 8 {
        (&SEL8, &VSEL8)
    } else {
        (&SEL4, &VSEL4)
    };
    let pri = i32::from(params.pri_strength);
    let sec = i32::from(params.sec_strength);
    let dir = usize::from(params.dir);
    let mut pt = [Tap::default(); 4];
    let mut st = [Tap::default(); 8];
    let (mut np, mut ns) = (0usize, 0usize);
    let tap = |a: i32, b: i32, k: i32| Tap {
        row: (a + 2) as usize,
        bi: (b + 2) as usize,
        k: k as i16,
    };
    if pri != 0 {
        // dav1d `src/cdef_tmpl.c:121-123,131,139` — 8-bit'te `bitdepth_min_8 = 0`.
        let mut k = 4 - (params.pri_strength & 1) as i32;
        for j in 0..2usize {
            let (a, b) = DIR_AB[dir + 2][j]; // dir
            // `tmp[x + off]` **ve** `tmp[x - off]`: negatif ofset satırı da çevirir.
            pt[np] = tap(a, b, k);
            pt[np + 1] = tap(-a, -b, k);
            np += 2;
            k = (k & 3) | 2; // dav1d `src/cdef_tmpl.c:139`
        }
    }
    if sec != 0 {
        // dav1d `src/cdef_tmpl.c:125,144-155`
        for j in 0..2usize {
            let k = 2 - j as i32; // sec_tap (dav1d `:151`)
            let (a2, b2) = DIR_AB[dir + 4][j]; // dir + 2
            let (a3, b3) = DIR_AB[dir][j]; // dir - 2
            st[ns] = tap(a2, b2, k);
            st[ns + 1] = tap(-a2, -b2, k);
            st[ns + 2] = tap(a3, b3, k);
            st[ns + 3] = tap(-a3, -b3, k);
            ns += 4;
        }
    }
    let pri_thr = _mm256_set1_epi16(pri as i16);
    let sec_thr = _mm256_set1_epi16(sec as i16);
    let pri_cnt = _mm_cvtsi32_si128(if pri != 0 {
        (i32::from(params.damping) - ilog2(pri)).clamp(0, 31)
    } else {
        0
    });
    let sec_cnt = _mm_cvtsi32_si128(if sec != 0 {
        (i32::from(params.damping) - ilog2(sec)).clamp(0, 31)
    } else {
        0
    });
    let center = load_i16x16(&sel[2]);
    let pick = _mm256_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8, 10, 12, 14, -1,
        -1, -1, -1, -1, -1, -1, -1,
    );
    let eight = _mm256_set1_epi16(8);

    // ---- 3) satır grupları (W = 8: 2 satır; W = 4: 4 satır) -------------
    let group = if W == 8 { 2 } else { 4 };
    let mut y = 0usize;
    while y < h {
        let px = _mm256_shuffle_epi8(srcs[y + 2], center);
        let mut sum = z;
        let mut minv = px;
        let mut maxv = px;
        // Bir tapın katkısı (dav1d `src/cdef_tmpl.c:37-42`, 16-bit lane):
        //   d = p − px; ad = |d|; t = max(0, thr − (ad >> shift));
        //   c = min_epu16(ad, t); c = sign_epi16(c, d)  [= apply_sign]
        let mut step = |t: &Tap, thr: __m256i, cnt: __m128i| {
            let i = y + t.row;
            let mut p = _mm256_shuffle_epi8(srcs[i], load_i16x16(&sel[t.bi]));
            if EDGES {
                // Komşu yoksa merkez piksel: katkı 0, min/max değişmez.
                let inv = _mm256_shuffle_epi8(vsrcs[i], load_i16x16(&vsel[t.bi]));
                p = _mm256_blendv_epi8(p, px, inv);
            }
            let d = _mm256_sub_epi16(p, px);
            let ad = _mm256_abs_epi16(d);
            let c = _mm256_min_epu16(ad, _mm256_subs_epu16(thr, _mm256_srl_epi16(ad, cnt)));
            let cs = _mm256_sign_epi16(c, d);
            sum = _mm256_add_epi16(sum, _mm256_mullo_epi16(cs, _mm256_set1_epi16(t.k)));
            if MINMAX {
                minv = _mm256_min_epi16(p, minv);
                maxv = _mm256_max_epi16(p, maxv);
            }
        };
        for t in &pt[..np] {
            step(t, pri_thr, pri_cnt);
        }
        for t in &st[..ns] {
            step(t, sec_thr, sec_cnt);
        }
        // `(sum - (sum < 0) + 8) >> 4` (dav1d `src/cdef_tmpl.c:165,184,210`)
        let neg = _mm256_cmpgt_epi16(z, sum);
        let r = _mm256_sub_epi16(sum, _mm256_srli_epi16::<15>(neg));
        let r = _mm256_srai_epi16::<4>(_mm256_add_epi16(r, eight));
        let mut r = _mm256_add_epi16(r, px);
        if MINMAX {
            // dav1d `src/cdef_tmpl.c:165` — `iclip(v, min, max)`, işaretli
            r = _mm256_min_epi16(_mm256_max_epi16(r, minv), maxv);
        }
        // i16 → u8: alt bayt (değer 0..255).
        let packed = _mm256_shuffle_epi8(r, pick);
        let lo = _mm256_castsi256_si128(packed);
        let hi = _mm256_extracti128_si256::<1>(packed);
        if W == 8 {
            store_u8x8(chunk_mut(dst, y * stride), lo);
            store_u8x8(chunk_mut(dst, (y + 1) * stride), hi);
        } else {
            store_u8x4(chunk_mut(dst, y * stride), lo);
            store_u8x4(chunk_mut(dst, (y + 1) * stride), _mm_srli_si128::<4>(lo));
            store_u8x4(chunk_mut(dst, (y + 2) * stride), hi);
            store_u8x4(chunk_mut(dst, (y + 3) * stride), _mm_srli_si128::<4>(hi));
        }
        y += group;
    }
}

/// `[i8; 16]` → yazmaç.
#[target_feature(enable = "avx2")]
#[inline]
fn v16(m: &[i8; 16]) -> __m128i {
    _mm_setr_epi8(
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13],
        m[14], m[15],
    )
}

// ---------------------------------------------------------------------------
// 16 bpc (10 ve 12 bit) — F4-CDEF b (P29)
// ---------------------------------------------------------------------------

/// Belirteçli 16 bpc `dir` dispatch yuvası işaretçisi: yalnız [`Avx2Token`] ile.
#[must_use]
pub const fn find_dir16_fn(_avx2: Avx2Token) -> CdefDirFn {
    find_dir16_entry
}

/// Belirteçli 16 bpc `filter` dispatch yuvası işaretçisi: yalnız [`Avx2Token`] ile.
#[must_use]
pub const fn filter16_fn(_avx2: Avx2Token) -> CdefFilterFn {
    filter16_entry
}

/// 16 bpc `dir` girişi. Adresi yalnız [`find_dir16_fn`] (belirteçli) ile verilir.
fn find_dir16_entry(img: &[u8], stride: usize, bit_depth: CdefBitDepth) -> CdefDirResult {
    if bit_depth == CdefBitDepth::B8 || img.len() < 7 * stride + 16 {
        // Sözleşme dışı: o bit derinliğinin skaler monomorfu.
        return (CdefDsp::scalar_for(bit_depth).dir)(img, stride, bit_depth);
    }
    let shift = bit_depth.min_8() as i32;
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`find_dir16_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır
    // (`cpu::token`). (2) sınır: `img.len() >= 7*stride + 16`; gövdenin en büyük
    // okuması `img[7*stride .. 7*stride + 16]`'dır ve ayrıca denetlenir.
    // (3) hizalama/takma ad: gövde `img`'yi yalnız okur, yerel dizilere yazar.
    unsafe { find_dir16_avx2(img, stride, shift) }
}

/// 16 bpc `filter` girişi. Adresi yalnız [`filter16_fn`] (belirteçli) ile verilir.
fn filter16_entry(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    let w = params.size.w();
    let h = params.size.h();
    let e = params.edges;
    // Sözleşme (bayt; `bpp = 2`): yalnız okuma sınırlarını güvenceye alır.
    // `damping` 16 bpc'de `cdef_damping + 3 + bitdepth_min_8` ≤ 10.
    let ok = params.bit_depth != CdefBitDepth::B8
        && params.damping <= 6 + params.bit_depth.min_8() as u8
        && params.dir < 8
        && stride >= 2 * (w + 4)
        && dst.len() >= (h - 1) * stride + 2 * (w + 2)
        && (!e.contains(CdefEdgeFlags::HAVE_LEFT) || left.len() >= h * 4)
        && (!e.contains(CdefEdgeFlags::HAVE_TOP) || top.len() >= stride + 2 * (w + 4))
        && (!e.contains(CdefEdgeFlags::HAVE_BOTTOM) || bottom.len() >= stride + 2 * (w + 4));
    if !ok {
        let fb = CdefDsp::scalar_for(params.bit_depth).fb[params.size.index()];
        fb(dst, stride, left, top, bottom, params);
        return;
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`filter16_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır.
    // (2) sınır: yukarıdaki denetimler gövdenin okuduğu her baytı kapsar (dst
    // satırları sütun `0..w+2`; bayrağı olan kenarda top/bottom
    // `row*stride + 2(w+4)`, left `4h`; bayraksız kenar dilimi okunmaz);
    // gövdenin yüklemeleri bu aralıklarda denetimli dizi referanslarıdır.
    // (3) hizalama/takma ad: `dst` tek `&mut`, kaynaklar `&`.
    unsafe { filter16_avx2(dst, stride, left, top, bottom, params) }
}

/// 16 bpc satırın 8 pikseli, `i16` lane'lerde `(px >> bitdepth_min_8) − 128`
/// (dav1d `cdef_tmpl.c:249`).
#[target_feature(enable = "avx2")]
#[inline]
fn dir_row16(img: &[u8], off: usize, sh: __m128i) -> __m128i {
    let v = load_u8x16(chunk(img, off));
    _mm_sub_epi16(_mm_srl_epi16(v, sh), _mm_set1_epi16(128))
}

/// 16 bpc yön taraması (AVX2): satırlar 8 bit'e indirgendikten sonra 8 bpc
/// gövdesiyle aynı maliyet hesabı ([`dir_maliyet!`]).
#[target_feature(enable = "avx2")]
fn find_dir16_avx2(img: &[u8], stride: usize, shift: i32) -> CdefDirResult {
    let sh = _mm_cvtsi32_si128(shift);
    let r0 = dir_row16(img, 0, sh);
    let r1 = dir_row16(img, stride, sh);
    let r2 = dir_row16(img, 2 * stride, sh);
    let r3 = dir_row16(img, 3 * stride, sh);
    let r4 = dir_row16(img, 4 * stride, sh);
    let r5 = dir_row16(img, 5 * stride, sh);
    let r6 = dir_row16(img, 6 * stride, sh);
    let r7 = dir_row16(img, 7 * stride, sh);
    dir_maliyet!(r0, r1, r2, r3, r4, r5, r6, r7)
}

/// 16 bpc kenar satırı (`top`/`bottom`, indeks 0 = sütun −2): `(lo, hi)`,
/// `lo` = sütun −2..5; `W = 8` iken `hi` = sütun 6..9 (8 bayt), `W = 4` iken 0.
#[target_feature(enable = "avx2")]
#[inline]
fn edge_row16<const W: usize>(s: &[u8], o: usize) -> (__m128i, __m128i) {
    let lo = load_u8x16(chunk(s, o));
    if W == 8 {
        (lo, _mm_cvtsi64_si128(le!(i64, 8, s, o + 16)))
    } else {
        (lo, _mm_setzero_si128())
    }
}

/// 16 bpc blok satırı: sütun −2, −1 `left`'ten (`l`, iki piksel 32 bit),
/// `0 .. w+2` `dst`'den. `W = 8`: `lo` = sütun −2..5, `hi` = 6..9 —
/// `o + 20 ≤ (h−1)·stride + 2(w+2)`. `W = 4`: tek yazmaç sütun −2..5 —
/// `o + 12 ≤ (h−1)·stride + 2(w+2)`.
#[target_feature(enable = "avx2")]
#[inline]
fn block_row16<const W: usize>(dst: &[u8], o: usize, l: i32) -> (__m128i, __m128i) {
    if W == 8 {
        let v = _mm_slli_si128::<4>(load_u8x16(chunk(dst, o)));
        let lo = _mm_insert_epi32::<0>(v, l);
        (lo, _mm_cvtsi64_si128(le!(i64, 8, dst, o + 12)))
    } else {
        let v = _mm_slli_si128::<4>(_mm_cvtsi64_si128(le!(i64, 8, dst, o)));
        let v = _mm_insert_epi32::<0>(v, l);
        (
            _mm_insert_epi32::<3>(v, le!(i32, 4, dst, o + 8)),
            _mm_setzero_si128(),
        )
    }
}

/// 16 bpc tap seçme maskeleri, sütun kayması `s = b + 2 ∈ [0, 4]`; her 128 bit
/// yarıda aynı. Pencere = sanal satırın (`W = 8`: `lo ‖ hi`, 12 piksel;
/// `W = 4`: tek yazmaç, 8 piksel) `s .. s+7` (`W = 8`) ya da `s .. s+3`
/// (`W = 4`) pikselleri. `parca`: `W = 8` → 0 = `lo`'dan, 1 = `hi`'dan; `W = 4`
/// → 0 = çıktının alt yarısı (lane 0..3), 1 = üst yarısı (lane 4..7). Seçilmeyen
/// lane `0x8080` (→ 0); iki parçanın `or`'u pencereyi verir.
pub(super) const fn sel16_table(w: usize, parca: usize) -> [[i16; 16]; 8] {
    let mut t = [[0i16; 16]; 8];
    let mut s = 0;
    while s < 5 {
        let mut half = 0;
        while half < 2 {
            let mut x = 0;
            while x < 8 {
                let src: i32 = if w == 8 {
                    let p = (s + x) as i32;
                    if parca == 0 {
                        if p < 8 { p } else { -1 }
                    } else if p >= 8 {
                        p - 8
                    } else {
                        -1
                    }
                } else if parca == 0 {
                    if x < 4 { (s + x) as i32 } else { -1 }
                } else if x >= 4 {
                    (s + x - 4) as i32
                } else {
                    -1
                };
                t[s][half * 8 + x] = if src < 0 {
                    0x8080u16 as i16
                } else {
                    ((2 * src) | ((2 * src + 1) << 8)) as u16 as i16
                };
                x += 1;
            }
            half += 1;
        }
        s += 1;
    }
    t
}

/// [`sel16_table`] örnekleri (derleme zamanı).
const SEL16_8A: [[i16; 16]; 8] = sel16_table(8, 0);
const SEL16_8B: [[i16; 16]; 8] = sel16_table(8, 1);
const SEL16_4A: [[i16; 16]; 8] = sel16_table(4, 0);
const SEL16_4B: [[i16; 16]; 8] = sel16_table(4, 1);

/// 16 bpc CDEF filtresi (AVX2) — dav1d `cdef_filter_block_c` ile bit-exact.
///
/// Güvenli `#[target_feature]` işlevidir; çağıranın sözleşmesi
/// `filter16_entry`'dedir.
#[target_feature(enable = "avx2")]
fn filter16_avx2(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    let pri = i32::from(params.pri_strength);
    let sec = i32::from(params.sec_strength);
    if pri == 0 && sec == 0 {
        return; // sözleşme dışı; skaler yolla aynı karar (dst değişmez)
    }
    let all = params.edges.bits() == 0b1111;
    match (params.size.w(), all, pri != 0 && sec != 0) {
        (8, true, true) => filter16_w::<8, false, true>(dst, stride, left, top, bottom, params),
        (8, true, false) => filter16_w::<8, false, false>(dst, stride, left, top, bottom, params),
        (8, false, true) => filter16_w::<8, true, true>(dst, stride, left, top, bottom, params),
        (8, false, false) => filter16_w::<8, true, false>(dst, stride, left, top, bottom, params),
        (_, true, true) => filter16_w::<4, false, true>(dst, stride, left, top, bottom, params),
        (_, true, false) => filter16_w::<4, false, false>(dst, stride, left, top, bottom, params),
        (_, false, true) => filter16_w::<4, true, true>(dst, stride, left, top, bottom, params),
        (_, false, false) => filter16_w::<4, true, false>(dst, stride, left, top, bottom, params),
    }
}

/// 16 bpc tap tablosu, yön başına (derleme zamanı; çağrı başına hesap yok):
/// `[0..4]` birincil (`k = 0` çifti, `k = 1` çifti), `[4..12]` ikincil (`k = 0`
/// dörtlüsü, `k = 1` dörtlüsü). Katsayılar çağrıda (`pri_tap` bit derinliğine
/// ve güce bağlı); `k` alanı kullanılmaz.
pub(super) const TAPS16: [[Tap; 12]; 8] = {
    let mut t = [[Tap {
        row: 0,
        bi: 0,
        k: 0,
    }; 12]; 8];
    let mut dir = 0;
    while dir < 8 {
        let mut j = 0;
        while j < 2 {
            let (a, b) = DIR_AB[dir + 2][j];
            t[dir][2 * j] = Tap {
                row: (a + 2) as usize,
                bi: (b + 2) as usize,
                k: 0,
            };
            t[dir][2 * j + 1] = Tap {
                row: (2 - a) as usize,
                bi: (2 - b) as usize,
                k: 0,
            };
            let (a2, b2) = DIR_AB[dir + 4][j];
            let (a3, b3) = DIR_AB[dir][j];
            t[dir][4 + 4 * j] = Tap {
                row: (a2 + 2) as usize,
                bi: (b2 + 2) as usize,
                k: 0,
            };
            t[dir][4 + 4 * j + 1] = Tap {
                row: (2 - a2) as usize,
                bi: (2 - b2) as usize,
                k: 0,
            };
            t[dir][4 + 4 * j + 2] = Tap {
                row: (a3 + 2) as usize,
                bi: (b3 + 2) as usize,
                k: 0,
            };
            t[dir][4 + 4 * j + 3] = Tap {
                row: (2 - a3) as usize,
                bi: (2 - b3) as usize,
                k: 0,
            };
            j += 1;
        }
        dir += 1;
    }
    t
};

/// 16 bpc kaynak dizilerinin boyutu: 2'nin kuvveti, dizin `& (N16 − 1)` ile
/// sınır denetimsiz (`MAX_ROWS = 12 ≤ 16`).
pub(super) const N16: usize = 16;

/// 16 bpc gövde: genişlik `W`, kenar maskesi `EDGES`, kırpma `MINMAX`.
///
/// Piksel zaten 16 bittir (genişletme yok); `i16` aritmetiği yeterlidir:
/// `|constrain| ≤ eşik` (12 bit'te birincil ≤ 240, ikincil ≤ 64), `|sum| ≤
/// 12·240 + 12·64 = 3 648`, `px + adj ∈ (−32 768, 32 767)`. Yazmaç düzeni
/// 8 bpc'yle aynıdır: `W = 8` iki satır (`y`, `y+1`), `W = 4` dört satır.
/// Satır kaynağı 12 (`W = 8`) ya da 8 (`W = 4`) piksel olduğu için pencere iki
/// `pshufb`'un `or`'udur: `W = 8` → `[lo(y) | lo(y+1)]` ve `[hi(y) | hi(y+1)]`;
/// `W = 4` → `[r(y) | r(y+2)]` ve `[r(y+1) | r(y+3)]`.
#[target_feature(enable = "avx2")]
fn filter16_w<const W: usize, const EDGES: bool, const MINMAX: bool>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    let h = params.size.h();
    let n_rows = h + 4;
    let e = params.edges;
    let has_top = !EDGES || e.contains(CdefEdgeFlags::HAVE_TOP);
    let has_bottom = !EDGES || e.contains(CdefEdgeFlags::HAVE_BOTTOM);
    let has_left = !EDGES || e.contains(CdefEdgeFlags::HAVE_LEFT);

    // ---- 1) kaynak satırları (dolgu yok: doğrudan dilimlerden) ------------
    let z128 = _mm_setzero_si128();
    let mut lo = [z128; N16];
    let mut hi = [z128; N16];
    if has_top {
        (lo[0], hi[0]) = edge_row16::<W>(top, 0);
        (lo[1], hi[1]) = edge_row16::<W>(top, stride);
    }
    for r in 0..h {
        let l = if has_left {
            le!(i32, 4, left, 4 * r)
        } else {
            0
        };
        (lo[(r + 2) & (N16 - 1)], hi[(r + 2) & (N16 - 1)]) = block_row16::<W>(dst, r * stride, l);
    }
    if has_bottom {
        (lo[h + 2], hi[h + 2]) = edge_row16::<W>(bottom, 0);
        (lo[h + 3], hi[h + 3]) = edge_row16::<W>(bottom, stride);
    }

    // Grup kaynakları (W = 8: srcs_a = lo çiftleri, srcs_b = hi çiftleri;
    // W = 4: srcs_a = [r(i) | r(i+2)], srcs_b = [r(i+1) | r(i+3)]).
    let z = _mm256_setzero_si256();
    let n_src = if W == 8 { n_rows - 1 } else { n_rows - 3 };
    let pack = |l: &[__m128i; N16], hh: &[__m128i; N16], i: usize| -> (__m256i, __m256i) {
        let m = N16 - 1;
        if W == 8 {
            (
                _mm256_set_m128i(l[(i + 1) & m], l[i & m]),
                _mm256_set_m128i(hh[(i + 1) & m], hh[i & m]),
            )
        } else {
            (
                _mm256_set_m128i(l[(i + 2) & m], l[i & m]),
                _mm256_set_m128i(l[(i + 3) & m], l[(i + 1) & m]),
            )
        }
    };
    let mut srcs_a = [z; N16];
    let mut srcs_b = [z; N16];
    for i in 0..n_src {
        (srcs_a[i & (N16 - 1)], srcs_b[i & (N16 - 1)]) = pack(&lo, &hi, i);
    }
    // Geçerlilik (yalnız kenar yolunda): piksel 0xFFFF = komşu yok.
    let mut vsrcs_a = [z; N16];
    let mut vsrcs_b = [z; N16];
    if EDGES {
        let mut col_lo = [0i8; 16];
        let mut col_hi = [0i8; 16];
        if !e.contains(CdefEdgeFlags::HAVE_LEFT) {
            col_lo[..4].fill(-1);
        }
        if !e.contains(CdefEdgeFlags::HAVE_RIGHT) {
            // sütun w, w+1: W = 8 → hi piksel 2, 3; W = 4 → lo piksel 6, 7.
            if W == 8 {
                col_hi[4..8].fill(-1);
            } else {
                col_lo[12..16].fill(-1);
            }
        }
        let (cl, ch) = (v16(&col_lo), v16(&col_hi));
        let allv = v16(&[-1i8; 16]);
        let mut vlo = [z128; N16];
        let mut vhi = [z128; N16];
        for i in 0..n_rows {
            let missing = (i < 2 && !has_top) || (i >= h + 2 && !has_bottom);
            (vlo[i & (N16 - 1)], vhi[i & (N16 - 1)]) =
                if missing { (allv, allv) } else { (cl, ch) };
        }
        for i in 0..n_src {
            (vsrcs_a[i & (N16 - 1)], vsrcs_b[i & (N16 - 1)]) = pack(&vlo, &vhi, i);
        }
    }

    // ---- 2) tap listesi (dav1d `src/cdef_tmpl.c:121-155`) ------------------
    let (sel_a, sel_b) = if W == 8 {
        (&SEL16_8A, &SEL16_8B)
    } else {
        (&SEL16_4A, &SEL16_4B)
    };
    let sel_av: [__m256i; 8] = core::array::from_fn(|i| load_i16x16(&sel_a[i]));
    let sel_bv: [__m256i; 8] = core::array::from_fn(|i| load_i16x16(&sel_b[i]));
    let bd_min8 = params.bit_depth.min_8() as i32;
    let pri = i32::from(params.pri_strength);
    let sec = i32::from(params.sec_strength);
    // pt[2j], pt[2j+1]: birincil `k = j` çifti (katsayı pk[j]); st[4j..4j+4]:
    // ikincil `k = j` dörtlüsü (katsayı 2 − j: 2 → `<< 1`, 1 → kendisi).
    let (pt, st) = TAPS16[usize::from(params.dir) & 7].split_at(4);
    // `pri_tap = 4 − ((pri_strength >> bitdepth_min_8) & 1)` (dav1d `:122`);
    // k = 1'de `(pri_tap & 3) | 2` (dav1d `:139`).
    let k0 = 4 - ((pri >> bd_min8) & 1);
    let pk = [
        _mm256_set1_epi16(k0 as i16),
        _mm256_set1_epi16(((k0 & 3) | 2) as i16),
    ];
    let pri_thr = _mm256_set1_epi16(pri as i16);
    let sec_thr = _mm256_set1_epi16(sec as i16);
    let pri_cnt = _mm_cvtsi32_si128(if pri != 0 {
        (i32::from(params.damping) - ilog2(pri)).clamp(0, 31)
    } else {
        0
    });
    let sec_cnt = _mm_cvtsi32_si128(if sec != 0 {
        (i32::from(params.damping) - ilog2(sec)).clamp(0, 31)
    } else {
        0
    });
    let (center_a, center_b) = (sel_av[2], sel_bv[2]);
    // `(x + 8) >> 4` = `pmulhrsw(x, 2048)` (|x| küçük; dav1d `cdef16_avx2.asm`).
    let r2048 = _mm256_set1_epi16(2048);

    // ---- 3) satır grupları (W = 8: 2 satır; W = 4: 4 satır) -------------
    let group = if W == 8 { 2 } else { 4 };
    let mut y = 0usize;
    while y < h {
        let px = _mm256_or_si256(
            _mm256_shuffle_epi8(srcs_a[(y + 2) & (N16 - 1)], center_a),
            _mm256_shuffle_epi8(srcs_b[(y + 2) & (N16 - 1)], center_b),
        );
        let mut minv = px;
        let mut maxv = px;
        // Bir tapın `constrain` katkısı (dav1d `src/cdef_tmpl.c:37-42`).
        let mut cons = |t: &Tap, thr: __m256i, cnt: __m128i| -> __m256i {
            let i = (y + t.row) & (N16 - 1);
            let bi = t.bi & 7;
            let (ma, mb) = (sel_av[bi], sel_bv[bi]);
            let mut p = _mm256_or_si256(
                _mm256_shuffle_epi8(srcs_a[i], ma),
                _mm256_shuffle_epi8(srcs_b[i], mb),
            );
            if EDGES {
                // Komşu yoksa merkez piksel: katkı 0, min/max değişmez.
                let inv = _mm256_or_si256(
                    _mm256_shuffle_epi8(vsrcs_a[i], ma),
                    _mm256_shuffle_epi8(vsrcs_b[i], mb),
                );
                p = _mm256_blendv_epi8(p, px, inv);
            }
            let d = _mm256_sub_epi16(p, px);
            let ad = _mm256_abs_epi16(d);
            let c = _mm256_min_epu16(ad, _mm256_subs_epu16(thr, _mm256_srl_epi16(ad, cnt)));
            if MINMAX {
                minv = _mm256_min_epi16(p, minv);
                maxv = _mm256_max_epi16(p, maxv);
            }
            _mm256_sign_epi16(c, d)
        };
        let mut sum = z;
        if pri != 0 {
            for j in 0..2usize {
                let c = _mm256_add_epi16(
                    cons(&pt[2 * j], pri_thr, pri_cnt),
                    cons(&pt[2 * j + 1], pri_thr, pri_cnt),
                );
                sum = _mm256_add_epi16(sum, _mm256_mullo_epi16(c, pk[j]));
            }
        }
        if sec != 0 {
            for j in 0..2usize {
                let c = _mm256_add_epi16(
                    _mm256_add_epi16(
                        cons(&st[4 * j], sec_thr, sec_cnt),
                        cons(&st[4 * j + 1], sec_thr, sec_cnt),
                    ),
                    _mm256_add_epi16(
                        cons(&st[4 * j + 2], sec_thr, sec_cnt),
                        cons(&st[4 * j + 3], sec_thr, sec_cnt),
                    ),
                );
                // sec_tap: k = 0 → 2, k = 1 → 1
                let c = if j == 0 { _mm256_slli_epi16::<1>(c) } else { c };
                sum = _mm256_add_epi16(sum, c);
            }
        }
        // `(sum - (sum < 0) + 8) >> 4` (dav1d `src/cdef_tmpl.c:165,184,210`):
        // `sum + (sum >> 15)` sonra `pmulhrsw(·, 2048)` = `(· + 8) >> 4`.
        let r = _mm256_add_epi16(sum, _mm256_srai_epi16::<15>(sum));
        let mut r = _mm256_add_epi16(_mm256_mulhrs_epi16(r, r2048), px);
        if MINMAX {
            // dav1d `src/cdef_tmpl.c:165` — `iclip(v, min, max)`
            r = _mm256_min_epi16(_mm256_max_epi16(r, minv), maxv);
        }
        // i16 → u16 (kırpmasız dallarda C'nin `pixel` dönüşümü: mod 2^16).
        let lo_r = _mm256_castsi256_si128(r);
        let hi_r = _mm256_extracti128_si256::<1>(r);
        if W == 8 {
            store_u8x16(chunk_mut(dst, y * stride), lo_r);
            store_u8x16(chunk_mut(dst, (y + 1) * stride), hi_r);
        } else {
            store_u8x8(chunk_mut(dst, y * stride), lo_r);
            store_u8x8(chunk_mut(dst, (y + 1) * stride), _mm_srli_si128::<8>(lo_r));
            store_u8x8(chunk_mut(dst, (y + 2) * stride), hi_r);
            store_u8x8(chunk_mut(dst, (y + 3) * stride), _mm_srli_si128::<8>(hi_r));
        }
        y += group;
    }
}
