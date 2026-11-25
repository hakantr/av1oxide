//! MC AVX2 çekirdekleri (8 bpc) — F4-MC a (P32): `put` (`mc[0..9]`) ve `prep`
//! (`mct[0..9]`); 9 adet 8-tap filtre birleşimi + bilinear.
//!
//! ## Kopyasız sözleşme (P32)
//!
//! Çekirdekler çağrı başına yerel dolgu kopyası **yapmaz**; kaynağın dolgu
//! hücresine (blok satırı −3, sütunu −3; [`crate::dsp::mc`] bellek
//! sözleşmesi) doğrudan okur. Girişte O(1) denetim
//! ([`crate::dsp::mc::simd_sozlesmesi`]): kaynak dilimi `(h+7)` satır ×
//! `(w+7)` piksellik okuma bölgesinin sonundan sonra en az
//! [`MC_KUYRUK`] bayt sürer. Sözleşme dışı girdi (bit derinliği, boyut, kısa
//! dilim) skaler gövdeye devreder ve sayılır
//! ([`crate::dsp::mc::skaler_devir_sayisi`]); çözme hattında bu sayı 0'dır.
//! `put` yalnız `w` piksel yazar (dar bloklarda kısmi saklama); `prep` `tmp`'ye
//! tam `w·h` öğe yazar. Bütün erişimler dilim/dizi denetimlidir: bir okuma
//! sözleşmenin dışına taşarsa panik olur, bellek dışına erişim olmaz.
//!
//! ## Yöntem (dav1d `mc_avx2.asm` yaklaşımı; asm kopyalanmadı)
//!
//! - **Tap sayısı derleme zamanında:** `sharp` içeren birleşimler 8 tap;
//!   `regular`/`smooth` tablolarının dış iki katsayısı sıfır olduğu için 6 tap
//!   (dav1d `put_6tap`/`prep_6tap`); `w ≤ 4` (yatay) / `h ≤ 4` (dikey) 4 tap
//!   tablosu (dav1d `GET_H_FILTER`/`GET_V_FILTER`); bilinear ofset 3'te 2 tap
//!   (`16 − m`, `m`). Generic çekirdek `N` tap'ı `S = (8 − N) / 2`. sütundan
//!   (satırdan) başlayan `N/2` ikiliyle hesaplar.
//! - **Yatay:** `pshufb` ile her çıktı pikseli için `(s[x+S+2k], s[x+S+2k+1])`
//!   ikilileri, `pmaddubsw` ile `i16`; 16 çıktı = iki 16 baytlık lane (ikinci
//!   lane bölgenin dışına taşmayacak tabandan yüklenir).
//! - **Dikey:** iki satırın baytları `punpck*bw` ile ikili yapılır,
//!   `pmaddubsw`; iki çıkış satırı birlikte, ikililer kayan pencerede.
//! - **İki boyutlu:** yatay geçiş `i16` ara satır (8 bpc `(Σ+2)>>2`, bilinear
//!   kaydırmasız), dikey geçiş `punpck*wd` + `pmaddwd` (`i32`).
//! - **Yuvarlama** (8 bpc, dav1d `src/mc_tmpl.c`): yalnız yatay `put`
//!   `(Σ+34)>>6`; `pmulhrsw` ile `(Σ+2)>>2` (×8192), `(Σ+32)>>6` (×512),
//!   `(Σ+8)>>4` (×2048); iki boyutlu `put` `(Σ+512)>>10`, `prep` `(Σ+32)>>6`,
//!   bilinear `put` `(Σ+128)>>8`, `prep` `(Σ+8)>>4`. `packuswb` doyurmalıdır
//!   (`iclip_pixel(0, 255)`).
//! - **Genişlik sınıfları:** `w ≥ 16` 16 piksellik sütun şeritleri (ymm);
//!   `w = 8` iki satır bir ymm'de; `w = 4`, `w = 2` iki satır bir xmm'de.
//!
//! ## `unsafe` politikası (GUV-02/GUV-04)
//!
//! `unsafe` yalnız belirteçli girişlerdeki `#[target_feature]` çağrısıdır
//! (generic `put_giris`/`prep_giris` ve [`put_8tap_regular_prepared`]); dosya
//! `audit::UNSAFE_EXEMPTIONS` defterinde kayıtlıdır. Yükleme/saklama
//! [`super::mem`] yardımcılarıyla ya da `cvtsi` dönüşümleriyle yapılır.
//!
//! ```compile_fail,E0061
//! // Belirteçsiz çekirdek çağrısı derlenmez.
//! let src = [0u8; 64];
//! let mut dst = [0u8; 4];
//! av1oxide::dsp::x86::mc::put_8tap_regular(
//!     &mut dst, 2, &src, 9, 2, 2, 0, 0, av1oxide::dsp::mc::McBitDepth::B8,
//! );
//! ```

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mem::{
    chunk, chunk_mut, load_u8x16, store_i16x8, store_i16x16, store_u8x8, store_u8x16, store_u8x32,
};
use crate::cpu::Avx2Token;
use crate::dsp::mc::{
    Filter2d, MC_KUYRUK, McBitDepth, McFn, McPrepFn, McPutFn, simd_sozlesmesi, skaler_devri_say,
};
use crate::dsp::mc_tables::MC_SUBPEL_FILTERS;
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_cvtsi32_si128, _mm_cvtsi64_si128, _mm_cvtsi128_si32,
    _mm_cvtsi128_si64, _mm_maddubs_epi16, _mm_mulhrs_epi16, _mm_mullo_epi16, _mm_packus_epi16,
    _mm_set_epi64x, _mm_set1_epi16, _mm_setr_epi8, _mm_setzero_si128, _mm_shuffle_epi8,
    _mm_srai_epi16, _mm_srli_epi16, _mm_srli_epi64, _mm_srli_si128, _mm_unpackhi_epi16,
    _mm_unpackhi_epi64, _mm_unpacklo_epi8, _mm_unpacklo_epi16, _mm_unpacklo_epi32,
    _mm_unpacklo_epi64, _mm256_add_epi16, _mm256_add_epi32, _mm256_castsi256_si128,
    _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_madd_epi16, _mm256_maddubs_epi16,
    _mm256_mulhrs_epi16, _mm256_mullo_epi16, _mm256_packs_epi32, _mm256_packus_epi16,
    _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set_m128i, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_slli_epi16,
    _mm256_srai_epi16, _mm256_srai_epi32, _mm256_srli_epi16, _mm256_unpackhi_epi8,
    _mm256_unpackhi_epi16, _mm256_unpacklo_epi8, _mm256_unpacklo_epi16,
};

// ---------------------------------------------------------------------------
// Dış yüzey (belirteçli)
// ---------------------------------------------------------------------------

/// `mc[filtre]` (8 bpc) yuvasının AVX2 girişi; yalnız [`Avx2Token`] ile.
#[must_use]
pub const fn put_fn(_avx2: Avx2Token, filtre: Filter2d) -> McPutFn {
    match filtre {
        Filter2d::Regular => put_giris::<0, 0>,
        Filter2d::RegularSmooth => put_giris::<0, 1>,
        Filter2d::RegularSharp => put_giris::<0, 2>,
        Filter2d::SharpRegular => put_giris::<2, 0>,
        Filter2d::SharpSmooth => put_giris::<2, 1>,
        Filter2d::Sharp => put_giris::<2, 2>,
        Filter2d::SmoothRegular => put_giris::<1, 0>,
        Filter2d::Smooth => put_giris::<1, 1>,
        Filter2d::SmoothSharp => put_giris::<1, 2>,
        Filter2d::Bilinear => put_giris::<BILIN, BILIN>,
    }
}

/// `mct[filtre]` (8 bpc) yuvasının AVX2 girişi; yalnız [`Avx2Token`] ile.
#[must_use]
pub const fn prep_fn(_avx2: Avx2Token, filtre: Filter2d) -> McPrepFn {
    match filtre {
        Filter2d::Regular => prep_giris::<0, 0>,
        Filter2d::RegularSmooth => prep_giris::<0, 1>,
        Filter2d::RegularSharp => prep_giris::<0, 2>,
        Filter2d::SharpRegular => prep_giris::<2, 0>,
        Filter2d::SharpSmooth => prep_giris::<2, 1>,
        Filter2d::Sharp => prep_giris::<2, 2>,
        Filter2d::SmoothRegular => prep_giris::<1, 0>,
        Filter2d::Smooth => prep_giris::<1, 1>,
        Filter2d::SmoothSharp => prep_giris::<1, 2>,
        Filter2d::Bilinear => prep_giris::<BILIN, BILIN>,
    }
}

/// S15.5 pilot yüzeyi: `put_8tap_regular` (`McFn` imzası); belirteçli.
pub fn put_8tap_regular(
    _avx2: Avx2Token,
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bit_depth: McBitDepth,
) {
    put_8tap_regular_entry(dst, dst_stride, src, src_stride, w, h, mx, my, bit_depth);
}

/// Pilot `McFn` işaretçisi: yalnız [`Avx2Token`] ile.
#[must_use]
pub const fn put_8tap_regular_fn(_avx2: Avx2Token) -> McFn {
    put_8tap_regular_entry
}

/// F3c: `mc[FILTER_2D_8TAP_REGULAR]` (8 bpc) yuvası — [`put_fn`] ile aynı giriş.
#[must_use]
pub const fn put_8tap_regular_slot_fn(avx2: Avx2Token) -> McPutFn {
    put_fn(avx2, Filter2d::Regular)
}

/// Pilot `McFn` girişi: P32 `put_giris::<0, 0>`'a ince uyarlayıcı.
fn put_8tap_regular_entry(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bit_depth: McBitDepth,
) {
    let bd = bit_depth.to_bit_depth();
    put_giris::<0, 0>(
        dst,
        dst_stride,
        src,
        src_stride,
        w,
        h,
        mx,
        my,
        bd,
        bit_depth.max_pixel() as u16,
    );
}

/// Ölçüm/test girişi: dolgu hücresi düzenindeki tamponda doğrudan çekirdek
/// (P32'de üretim girişiyle aynı gövde; yerel kopya yoktur).
///
/// # Panics
///
/// Sözleşme ([`crate::dsp::mc::simd_sozlesmesi`]) tutmazsa panikler (test
/// yardımcısı; üretim girişi skalere devreder).
#[doc(hidden)]
pub fn put_8tap_regular_prepared(
    _avx2: Avx2Token,
    dst: &mut [u8],
    dst_stride: usize,
    padded: &[u8],
    padded_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    assert!(dst_stride >= w && h >= 1, "prepared: hedef adımı kısa");
    assert!(
        simd_sozlesmesi(
            padded.len(),
            padded_stride,
            w,
            h,
            mx,
            my,
            dst.len(),
            (h - 1) * dst_stride + w
        ),
        "prepared: sözleşme dışı girdi"
    );
    // Belirteçli giriş (bu işlev `Avx2Token` ister): üretim girişiyle aynı gövde.
    put_giris::<0, 0>(
        dst,
        dst_stride,
        padded,
        padded_stride,
        w,
        h,
        mx,
        my,
        BitDepth::Eight,
        255,
    );
}

/// Bilinear yuvası için sözde filtre türü (8-tap tablosu kullanılmaz).
pub(super) const BILIN: usize = 9;

/// `(FH, FV)` filtre türleri (0 regular, 1 smooth, 2 sharp; [`BILIN`]) →
/// [`Filter2d`] (skaler devir için).
pub(super) const fn filtre2d(fh: usize, fv: usize) -> Filter2d {
    match (fh, fv) {
        (0, 0) => Filter2d::Regular,
        (0, 1) => Filter2d::RegularSmooth,
        (0, 2) => Filter2d::RegularSharp,
        (2, 0) => Filter2d::SharpRegular,
        (2, 1) => Filter2d::SharpSmooth,
        (2, 2) => Filter2d::Sharp,
        (1, 0) => Filter2d::SmoothRegular,
        (1, 1) => Filter2d::Smooth,
        (1, 2) => Filter2d::SmoothSharp,
        _ => Filter2d::Bilinear,
    }
}

/// `put` girişi: sözleşme denetimi + AVX2 gövdesi ya da skaler devir.
fn put_giris<const FH: usize, const FV: usize>(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    let ok = bit_depth == BitDepth::Eight
        && dst_stride >= w
        && h >= 1
        && simd_sozlesmesi(
            src.len(),
            src_stride,
            w,
            h,
            mx,
            my,
            dst.len(),
            (h - 1) * dst_stride + w,
        );
    if !ok {
        skaler_devri_say();
        put_skaler::<FH, FV>(
            dst,
            dst_stride,
            src,
            src_stride,
            w,
            h,
            mx,
            my,
            bit_depth,
            bitdepth_max,
        );
        return;
    }
    // Yol girişte seçilir: kopya ISA gerektirmez; öbür yollar kendi küçük
    // `#[target_feature]` gövdesine gider (çağrı başına sabit maliyet, P32).
    let mut hd = Hedef {
        dst,
        ds: dst_stride,
        tmp: &mut [],
    };
    let (ss, hd) = (src_stride, &mut hd);
    match (mx != 0, my != 0) {
        (false, false) if w < 16 => kopya_put(hd, src, ss, w, h),
        // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
        // (`put_fn`, `put_8tap_regular_fn`); belirteç AVX2 tespitinin kanıtıdır.
        // (2) sınır: `simd_sozlesmesi` kaynak bölgesini + `MC_KUYRUK`'u ve hedefte
        // `h` satır × `w` baytı kapsar; erişimler denetimlidir. (3) hizalama/takma
        // ad: `dst` tek `&mut`, `src` `&`.
        (false, false) => unsafe { kopya_genis::<false>(hd, src, ss, w, h) },
        // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
        // (`put_fn`, `put_8tap_regular_fn`); belirteç AVX2 tespiti + maske
        // izninin kanıtıdır (`cpu::token`). (2) sınır: `simd_sozlesmesi` kaynak
        // bölgesini + `MC_KUYRUK`'u ve hedefte `h` satır × `w` baytı kapsar; gövde
        // erişimleri dilim/dizi denetimlidir. (3) hizalama/takma ad: `dst` tek
        // `&mut`, `src` `&`; ödünç kuralları çakışmayı dışlar.
        (true, false) => unsafe { yol_h::<FH, false>(hd, src, ss, w, h, mx) },
        // SAFETY: (1) ISA: adres yalnız `Avx2Token` ile verilir (`put_fn`);
        // belirteç AVX2 tespitinin kanıtıdır. (2) sınır: `simd_sozlesmesi`
        // kaynak bölgesini + `MC_KUYRUK`'u ve hedefi kapsar; erişimler
        // denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
        (false, true) => unsafe { yol_v::<FV, false>(hd, src, ss, w, h, my) },
        // SAFETY: (1) ISA: adres yalnız `Avx2Token` ile verilir (`put_fn`);
        // belirteç AVX2 tespitinin kanıtıdır. (2) sınır: `simd_sozlesmesi`
        // kaynak bölgesini + `MC_KUYRUK`'u ve hedefi kapsar; erişimler
        // denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
        (true, true) => unsafe { yol_hv::<FH, FV, false>(hd, src, ss, w, h, mx, my) },
    }
}

/// `prep` girişi: sözleşme denetimi + AVX2 gövdesi ya da skaler devir.
fn prep_giris<const FH: usize, const FV: usize>(
    tmp: &mut [i16],
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    let ok = bit_depth == BitDepth::Eight
        && simd_sozlesmesi(src.len(), src_stride, w, h, mx, my, tmp.len(), w * h);
    if !ok {
        skaler_devri_say();
        prep_skaler::<FH, FV>(tmp, src, src_stride, w, h, mx, my, bit_depth, bitdepth_max);
        return;
    }
    let mut hd = Hedef {
        dst: &mut [],
        ds: 0,
        tmp,
    };
    let (ss, hd) = (src_stride, &mut hd);
    match (mx != 0, my != 0) {
        (false, false) if w < 16 => kopya_prep(hd, src, ss, w, h),
        // SAFETY: (1) ISA: adres yalnız `Avx2Token` ile verilir (`prep_fn`);
        // belirteç AVX2 tespitinin kanıtıdır. (2) sınır: `simd_sozlesmesi`
        // kaynak bölgesini + `MC_KUYRUK`'u ve `tmp`'yi kapsar; erişimler
        // denetimlidir. (3) hizalama/takma ad: `tmp` tek `&mut`, `src` `&`.
        (false, false) => unsafe { kopya_genis::<true>(hd, src, ss, w, h) },
        // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
        // (`prep_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır.
        // (2) sınır: `simd_sozlesmesi` kaynak bölgesini + `MC_KUYRUK`'u ve
        // `tmp`'de `w·h` öğeyi kapsar; erişimler denetimlidir. (3) hizalama/takma
        // ad: `tmp` tek `&mut`, `src` `&`.
        (true, false) => unsafe { yol_h::<FH, true>(hd, src, ss, w, h, mx) },
        // SAFETY: (1) ISA: adres yalnız `Avx2Token` ile verilir (`prep_fn`);
        // belirteç AVX2 tespitinin kanıtıdır. (2) sınır: `simd_sozlesmesi`
        // kaynak bölgesini + `MC_KUYRUK`'u ve `tmp`'yi kapsar; erişimler
        // denetimlidir. (3) hizalama/takma ad: `tmp` tek `&mut`, `src` `&`.
        (false, true) => unsafe { yol_v::<FV, true>(hd, src, ss, w, h, my) },
        // SAFETY: (1) ISA: adres yalnız `Avx2Token` ile verilir (`prep_fn`);
        // belirteç AVX2 tespitinin kanıtıdır. (2) sınır: `simd_sozlesmesi`
        // kaynak bölgesini + `MC_KUYRUK`'u ve `tmp`'yi kapsar; erişimler
        // denetimlidir. (3) hizalama/takma ad: `tmp` tek `&mut`, `src` `&`.
        (true, true) => unsafe { yol_hv::<FH, FV, true>(hd, src, ss, w, h, mx, my) },
    }
}

/// P36: AVX512ICL `put` girişi (adres yalnız [`super::mc_avx512::put_fn`] ile,
/// [`Avx512IclToken`](crate::cpu::Avx512IclToken)). AVX2 girişiyle aynı
/// sözleşme denetimi ve `w < 32`'de aynı dar dağıtım (bu modülün gövdeleri);
/// `w ≥ 32` [`super::mc_avx512::genis`]. Giriş bu modüldedir: paylaşılan AVX2
/// gövdelerinin bütün çağırıcıları aynı kod üretim biriminde kalır ve iç
/// bağlantıları korunur (başka birimden çağrılan tek kopya dış bağlantıya geçip
/// LLVM'nin yerel işlev iyileştirmelerini yitiriyordu: AVX2 hücrelerinde
/// +%10–20, P36 hız bütçesi tanı koşusu).
pub(super) fn put_giris_icl<const FH: usize, const FV: usize>(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    let ok = bit_depth == BitDepth::Eight
        && dst_stride >= w
        && h >= 1
        && simd_sozlesmesi(
            src.len(),
            src_stride,
            w,
            h,
            mx,
            my,
            dst.len(),
            (h - 1) * dst_stride + w,
        );
    if !ok {
        skaler_devri_say();
        put_skaler::<FH, FV>(
            dst,
            dst_stride,
            src,
            src_stride,
            w,
            h,
            mx,
            my,
            bit_depth,
            bitdepth_max,
        );
        return;
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`mc_avx512::put_fn`); belirteç AVX512ICL (VBMI dahil, AVX2'yi kapsar)
    // tespiti + maske izninin kanıtıdır. (2) sınır: `simd_sozlesmesi` kaynak
    // bölgesini + `MC_KUYRUK`'u ve hedefte `h` satır × `w` baytı kapsar (AVX2 ve
    // AVX512ICL gövdelerinin sözleşmesi); maskeli yüklemeler yalnız bölge
    // baytlarını seçer, erişimler dilim/dizi denetimlidir. (3) hizalama/takma ad:
    // hizasız erişim; `dst` tek `&mut`, `src` `&`.
    unsafe {
        // Geniş dal `Hedef`'ini kendisi kurar: dar dalda yapı yazmaçlarda kalır
        // (AVX2 girişiyle aynı kod; ortak kurulum dar dalda yığına yazılıyordu).
        if w >= 32 {
            let mut hd = Hedef {
                dst,
                ds: dst_stride,
                tmp: &mut [],
            };
            return super::mc_avx512::genis::<FH, FV, false>(
                &mut hd, src, src_stride, w, h, mx, my,
            );
        }
        let mut hd = Hedef {
            dst,
            ds: dst_stride,
            tmp: &mut [],
        };
        let (ss, hd) = (src_stride, &mut hd);
        match (mx != 0, my != 0) {
            (false, false) if w < 16 => kopya_put(hd, src, ss, w, h),
            (false, false) => kopya_genis::<false>(hd, src, ss, w, h),
            (true, false) => yol_h::<FH, false>(hd, src, ss, w, h, mx),
            (false, true) => yol_v::<FV, false>(hd, src, ss, w, h, my),
            (true, true) => yol_hv::<FH, FV, false>(hd, src, ss, w, h, mx, my),
        }
    }
}

/// P36: AVX512ICL `prep` girişi ([`put_giris_icl`] gibi; adres yalnız
/// [`super::mc_avx512::prep_fn`] ile).
pub(super) fn prep_giris_icl<const FH: usize, const FV: usize>(
    tmp: &mut [i16],
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    let ok = bit_depth == BitDepth::Eight
        && simd_sozlesmesi(src.len(), src_stride, w, h, mx, my, tmp.len(), w * h);
    if !ok {
        skaler_devri_say();
        prep_skaler::<FH, FV>(tmp, src, src_stride, w, h, mx, my, bit_depth, bitdepth_max);
        return;
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`mc_avx512::prep_fn`); belirteç AVX512ICL (VBMI dahil, AVX2'yi kapsar)
    // tespiti + maske izninin kanıtıdır. (2) sınır: `simd_sozlesmesi` kaynak
    // bölgesini + `MC_KUYRUK`'u ve `tmp`'de `w·h` öğeyi kapsar; maskeli
    // yüklemeler yalnız bölge baytlarını seçer, erişimler denetimlidir. (3)
    // hizalama/takma ad: hizasız erişim; `tmp` tek `&mut`, `src` `&`.
    unsafe {
        // Geniş dal `Hedef`'ini kendisi kurar: dar dalda yapı yazmaçlarda kalır
        // (AVX2 girişiyle aynı kod; ortak kurulum dar dalda yığına yazılıyordu).
        if w >= 32 {
            let mut hd = Hedef {
                dst: &mut [],
                ds: 0,
                tmp,
            };
            return super::mc_avx512::genis::<FH, FV, true>(&mut hd, src, src_stride, w, h, mx, my);
        }
        let mut hd = Hedef {
            dst: &mut [],
            ds: 0,
            tmp,
        };
        let (ss, hd) = (src_stride, &mut hd);
        match (mx != 0, my != 0) {
            (false, false) if w < 16 => kopya_prep(hd, src, ss, w, h),
            (false, false) => kopya_genis::<true>(hd, src, ss, w, h),
            (true, false) => yol_h::<FH, true>(hd, src, ss, w, h, mx),
            (false, true) => yol_v::<FV, true>(hd, src, ss, w, h, my),
            (true, true) => yol_hv::<FH, FV, true>(hd, src, ss, w, h, mx, my),
        }
    }
}

#[cold]
pub(super) fn put_skaler<const FH: usize, const FV: usize>(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    let f = filtre2d(FH, FV);
    // Skaler gövde `BPC` monomorfudur: bit derinliğine göre seçilir.
    macro_rules! cagir {
        ($bpc:literal) => {
            if f.is_bilinear() {
                scalar::put_bilin::<$bpc>(
                    dst,
                    dst_stride,
                    src,
                    src_stride,
                    w,
                    h,
                    mx,
                    my,
                    bit_depth,
                    bitdepth_max,
                )
            } else {
                scalar::put_8tap::<$bpc>(
                    dst,
                    dst_stride,
                    src,
                    src_stride,
                    w,
                    h,
                    mx,
                    my,
                    f,
                    bit_depth,
                    bitdepth_max,
                )
            }
        };
    }
    match bit_depth {
        BitDepth::Eight => cagir!(8),
        BitDepth::Ten => cagir!(10),
        BitDepth::Twelve => cagir!(12),
    }
}

#[cold]
pub(super) fn prep_skaler<const FH: usize, const FV: usize>(
    tmp: &mut [i16],
    src: &[u8],
    src_stride: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    let f = filtre2d(FH, FV);
    macro_rules! cagir {
        ($bpc:literal) => {
            if f.is_bilinear() {
                scalar::prep_bilin::<$bpc>(
                    tmp,
                    src,
                    src_stride,
                    w,
                    h,
                    mx,
                    my,
                    bit_depth,
                    bitdepth_max,
                )
            } else {
                scalar::prep_8tap::<$bpc>(
                    tmp,
                    src,
                    src_stride,
                    w,
                    h,
                    mx,
                    my,
                    f,
                    bit_depth,
                    bitdepth_max,
                )
            }
        };
    }
    match bit_depth {
        BitDepth::Eight => cagir!(8),
        BitDepth::Ten => cagir!(10),
        BitDepth::Twelve => cagir!(12),
    }
}

// ---------------------------------------------------------------------------
// Yol seçimi
// ---------------------------------------------------------------------------

/// Hedef: `put` için bayt düzlemi, `prep` için `tmp` (adım `w`).
pub(super) struct Hedef<'a> {
    pub(super) dst: &'a mut [u8],
    pub(super) ds: usize,
    pub(super) tmp: &'a mut [i16],
}

/// Filtre satırı: tür `t` (0 regular, 1 smooth, 2 sharp), boyut (`w` ya da
/// `h`), alt piksel `m ≥ 1` — dav1d `GET_H_FILTER`/`GET_V_FILTER`.
pub(super) fn filtre_satiri(t: usize, boyut: usize, m: u8) -> &'static [i8; 8] {
    let tablo = if boyut > 4 { t } else { 3 + (t & 1) };
    &MC_SUBPEL_FILTERS[tablo][usize::from(m) - 1]
}

/// Tap sayısı: `boyut ≤ 4` → 4; `sharp` → 8; aksi hâlde 6.
const fn tap(t: usize, boyut: usize) -> usize {
    if boyut <= 4 {
        4
    } else if t == 2 {
        8
    } else {
        6
    }
}

/// Yalnız yatay yol: filtre satırı ve tap sayısı seçimi.
#[target_feature(enable = "avx2")]
fn yol_h<const FH: usize, const PREP: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
) {
    if FH == BILIN {
        return yatay::<2, PREP, true>(hd, src, ss, w, h, &kat_bilin(mx));
    }
    let f = filtre_satiri(FH, w, mx);
    match tap(FH, w) {
        8 => yatay::<8, PREP, false>(hd, src, ss, w, h, &kat_b::<8>(f)),
        6 => yatay::<6, PREP, false>(hd, src, ss, w, h, &kat_b::<6>(f)),
        _ => yatay::<4, PREP, false>(hd, src, ss, w, h, &kat_b::<4>(f)),
    }
}

/// Yalnız dikey yol: filtre satırı ve tap sayısı seçimi.
#[target_feature(enable = "avx2")]
fn yol_v<const FV: usize, const PREP: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    my: u8,
) {
    if FV == BILIN {
        return dikey::<2, PREP, true>(hd, src, ss, w, h, &kat_bilin(my));
    }
    let f = filtre_satiri(FV, h, my);
    match tap(FV, h) {
        8 => dikey::<8, PREP, false>(hd, src, ss, w, h, &kat_b::<8>(f)),
        6 => dikey::<6, PREP, false>(hd, src, ss, w, h, &kat_b::<6>(f)),
        _ => dikey::<4, PREP, false>(hd, src, ss, w, h, &kat_b::<4>(f)),
    }
}

/// İki boyutlu yol: iki eksenin filtre satırı ve tap sayısı seçimi.
#[target_feature(enable = "avx2")]
fn yol_hv<const FH: usize, const FV: usize, const PREP: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    if FH == BILIN {
        return iki::<2, 2, PREP, true>(hd, src, ss, w, h, &kat_bilin(mx), &kat_bilin_w(my));
    }
    let fh = filtre_satiri(FH, w, mx);
    let fv = filtre_satiri(FV, h, my);
    macro_rules! hv {
        ($nh:literal, $nv:literal) => {
            iki::<$nh, $nv, PREP, false>(hd, src, ss, w, h, &kat_b::<$nh>(fh), &kat_w::<$nv>(fv))
        };
    }
    match (tap(FH, w), tap(FV, h)) {
        (8, 8) => hv!(8, 8),
        (8, 6) => hv!(8, 6),
        (8, _) => hv!(8, 4),
        (6, 8) => hv!(6, 8),
        (6, 6) => hv!(6, 6),
        (6, _) => hv!(6, 4),
        (_, 8) => hv!(4, 8),
        (_, 6) => hv!(4, 6),
        _ => hv!(4, 4),
    }
}

// ---------------------------------------------------------------------------
// Katsayılar ve karıştırma maskeleri
// ---------------------------------------------------------------------------

/// `pmaddubsw` ikilileri: `k < N/2` için `(f[S+2k], f[S+2k+1])` baytları
/// (`S = (8−N)/2`), her 16 bit lane'de.
#[target_feature(enable = "avx2")]
#[inline]
fn kat_b<const N: usize>(f: &[i8; 8]) -> [__m256i; 4] {
    let s = (8 - N) / 2;
    let mut c = [_mm256_setzero_si256(); 4];
    for (k, ck) in c.iter_mut().enumerate().take(N / 2) {
        let lo = f[s + 2 * k] as u8;
        let hi = f[s + 2 * k + 1] as u8;
        *ck = _mm256_set1_epi16(i16::from_le_bytes([lo, hi]));
    }
    c
}

/// Bilinear `pmaddubsw` ikilisi `(16 − m, m)`.
#[target_feature(enable = "avx2")]
#[inline]
fn kat_bilin(m: u8) -> [__m256i; 4] {
    let mut c = [_mm256_setzero_si256(); 4];
    c[0] = _mm256_set1_epi16(i16::from_le_bytes([16 - m, m]));
    c
}

/// `pmaddwd` ikilileri (iki boyutlu dikey geçiş): `(f[S+2k], f[S+2k+1])`
/// `i16` olarak her 32 bit lane'de.
#[target_feature(enable = "avx2")]
#[inline]
fn kat_w<const N: usize>(f: &[i8; 8]) -> [__m256i; 4] {
    let s = (8 - N) / 2;
    let mut c = [_mm256_setzero_si256(); 4];
    for (k, ck) in c.iter_mut().enumerate().take(N / 2) {
        let lo = i32::from(f[s + 2 * k]) & 0xffff;
        let hi = i32::from(f[s + 2 * k + 1]) << 16;
        *ck = _mm256_set1_epi32(hi | lo);
    }
    c
}

/// Bilinear `pmaddwd` ikilisi `(16 − m, m)`.
#[target_feature(enable = "avx2")]
#[inline]
fn kat_bilin_w(m: u8) -> [__m256i; 4] {
    // [0]: `pmaddwd` ikilisi; [1], [2]: `u16` yolu için `16 − m` ve `m` (P32).
    let mut c = [_mm256_setzero_si256(); 4];
    c[0] = _mm256_set1_epi32((i32::from(m) << 16) | (16 - i32::from(m)));
    c[1] = _mm256_set1_epi16(16 - i16::from(m));
    c[2] = _mm256_set1_epi16(i16::from(m));
    c
}

/// Bilinear iki boyutlu dikey adım `u16`'da: `t = (16−my)·m0 + my·m1`
/// (`m ∈ [0, 4080]` → `t ≤ 65 280`, işaretsiz 16 bite sığar); `put`
/// `(t+128)>>8` (`≤ 65 408`), `prep` `(t+8)>>4` — mantıksal kaydırma, tam sonuç.
#[target_feature(enable = "avx2")]
#[inline]
fn bilin_dikey<const PREP: bool>(a: __m256i, b: __m256i, cv: &[__m256i; 4]) -> __m256i {
    let t = _mm256_add_epi16(_mm256_mullo_epi16(a, cv[1]), _mm256_mullo_epi16(b, cv[2]));
    if PREP {
        _mm256_srli_epi16::<4>(_mm256_add_epi16(t, _mm256_set1_epi16(8)))
    } else {
        _mm256_srli_epi16::<8>(_mm256_add_epi16(t, _mm256_set1_epi16(128)))
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn bilin_dikey128<const PREP: bool>(a: __m128i, b: __m128i, cv: &[__m256i; 4]) -> __m128i {
    let (c1, c2) = (_mm256_castsi256_si128(cv[1]), _mm256_castsi256_si128(cv[2]));
    let t = _mm_add_epi16(_mm_mullo_epi16(a, c1), _mm_mullo_epi16(b, c2));
    if PREP {
        _mm_srli_epi16::<4>(_mm_add_epi16(t, _mm_set1_epi16(8)))
    } else {
        _mm_srli_epi16::<8>(_mm_add_epi16(t, _mm_set1_epi16(128)))
    }
}

/// 16 baytlık lane maskesi: çıktı `i` (0..8), ikili `k` → baytlar `ofs+i+2k`,
/// `ofs+i+2k+1`.
const fn maske8(ofs: usize, k: usize) -> [i8; 16] {
    let mut m = [0i8; 16];
    let mut i = 0;
    while i < 8 {
        m[2 * i] = (ofs + i + 2 * k) as i8;
        m[2 * i + 1] = (ofs + i + 2 * k + 1) as i8;
        i += 1;
    }
    m
}

/// İki satır (yarım lane başına 8 bayt), satır başına `n` çıktı (`n` = 4 ya
/// da 2): satır 0 çıktıları kelime `0..n`, satır 1 çıktıları `n..2n`; kalanı
/// sıfır (`0x80`).
const fn maske_iki(ofs: usize, k: usize, n: usize) -> [i8; 16] {
    let mut m = [-128i8; 16];
    let mut r = 0;
    while r < 2 {
        let mut i = 0;
        while i < n {
            let o = r * n + i;
            m[2 * o] = (8 * r + ofs + i + 2 * k) as i8;
            m[2 * o + 1] = (8 * r + ofs + i + 2 * k + 1) as i8;
            i += 1;
        }
        r += 1;
    }
    m
}

#[target_feature(enable = "avx2")]
#[inline]
fn v128(m: [i8; 16]) -> __m128i {
    _mm_setr_epi8(
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13],
        m[14], m[15],
    )
}

#[target_feature(enable = "avx2")]
#[inline]
fn v256(lo: [i8; 16], hi: [i8; 16]) -> __m256i {
    _mm256_set_m128i(v128(hi), v128(lo))
}

// ---------------------------------------------------------------------------
// Yükleme / saklama yardımcıları (denetimli)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
fn yukle8(src: &[u8], o: usize) -> i64 {
    i64::from_le_bytes(*chunk::<u8, 8>(src, o))
}

#[target_feature(enable = "avx2")]
#[inline]
fn yukle4(src: &[u8], o: usize) -> __m128i {
    _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(src, o)))
}

#[target_feature(enable = "avx2")]
#[inline]
fn yukle2(src: &[u8], o: usize) -> __m128i {
    _mm_cvtsi32_si128(i32::from(u16::from_le_bytes(*chunk::<u8, 2>(src, o))))
}

/// 16 `i16` (lane 0 sütun 0..7, lane 1 sütun 8..15) → `put` 16 bayt ya da
/// `prep` 16 öğe, satır `y` sütun `x`.
#[target_feature(enable = "avx2")]
#[inline]
fn sakla16<const PREP: bool>(hd: &mut Hedef<'_>, w: usize, y: usize, x: usize, v: __m256i) {
    if PREP {
        store_i16x16(chunk_mut(hd.tmp, y * w + x), v);
    } else {
        let b = _mm_packus_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        store_u8x16(chunk_mut(hd.dst, y * hd.ds + x), b);
    }
}

/// İki satır × 8 (lane 0 satır `y`, lane 1 satır `y+1`).
#[target_feature(enable = "avx2")]
#[inline]
fn sakla8x2<const PREP: bool>(hd: &mut Hedef<'_>, y: usize, v: __m256i) {
    if PREP {
        store_i16x16(chunk_mut(hd.tmp, y * 8), v);
    } else {
        let b = _mm_packus_epi16(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        let ds = hd.ds;
        store_u8x8(chunk_mut(hd.dst, y * ds), b);
        store_u8x8(chunk_mut(hd.dst, (y + 1) * ds), _mm_srli_si128::<8>(b));
    }
}

/// İki satır × `n` (`n` = 4 ya da 2) xmm'de: satır `y` kelime `0..n`, satır
/// `y+1` kelime `n..2n`.
#[target_feature(enable = "avx2")]
#[inline]
fn sakla_dar<const PREP: bool, const N: usize>(hd: &mut Hedef<'_>, y: usize, v: __m128i) {
    if PREP {
        if N == 4 {
            store_i16x8(chunk_mut(hd.tmp, y * 4), v);
        } else {
            let q = _mm_cvtsi128_si64(v) as u64;
            let t: &mut [i16; 4] = chunk_mut(hd.tmp, y * 2);
            for (i, o) in t.iter_mut().enumerate() {
                *o = (q >> (16 * i)) as i16;
            }
        }
    } else {
        let b = _mm_packus_epi16(v, v);
        let ds = hd.ds;
        if N == 4 {
            let q = (_mm_cvtsi128_si64(b) as u64).to_le_bytes();
            *chunk_mut::<u8, 4>(hd.dst, y * ds) = [q[0], q[1], q[2], q[3]];
            *chunk_mut::<u8, 4>(hd.dst, (y + 1) * ds) = [q[4], q[5], q[6], q[7]];
        } else {
            let q = (_mm_cvtsi128_si32(b) as u32).to_le_bytes();
            *chunk_mut::<u8, 2>(hd.dst, y * ds) = [q[0], q[1]];
            *chunk_mut::<u8, 2>(hd.dst, (y + 1) * ds) = [q[2], q[3]];
        }
    }
}

// ---------------------------------------------------------------------------
// Yuvarlama (8 bpc)
// ---------------------------------------------------------------------------

/// Yalnız yatay: `put` `(Σ+34)>>6`, `prep` `(Σ+2)>>2`; bilinear `put`
/// `(Σ+8)>>4`, `prep` `Σ`.
#[target_feature(enable = "avx2")]
#[inline]
fn yuv_h<const PREP: bool, const BIL: bool>(v: __m256i) -> __m256i {
    match (PREP, BIL) {
        (false, false) => _mm256_srai_epi16::<6>(_mm256_add_epi16(v, _mm256_set1_epi16(34))),
        (true, false) => _mm256_mulhrs_epi16(v, _mm256_set1_epi16(8192)),
        (false, true) => _mm256_mulhrs_epi16(v, _mm256_set1_epi16(2048)),
        (true, true) => v,
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn yuv_h128<const PREP: bool, const BIL: bool>(v: __m128i) -> __m128i {
    match (PREP, BIL) {
        (false, false) => _mm_srai_epi16::<6>(_mm_add_epi16(v, _mm_set1_epi16(34))),
        (true, false) => _mm_mulhrs_epi16(v, _mm_set1_epi16(8192)),
        (false, true) => _mm_mulhrs_epi16(v, _mm_set1_epi16(2048)),
        (true, true) => v,
    }
}

/// Yalnız dikey: `put` `(Σ+32)>>6`, `prep` `(Σ+2)>>2`; bilinear `put`
/// `(Σ+8)>>4`, `prep` `Σ`.
#[target_feature(enable = "avx2")]
#[inline]
fn yuv_v<const PREP: bool, const BIL: bool>(v: __m256i) -> __m256i {
    match (PREP, BIL) {
        (false, false) => _mm256_mulhrs_epi16(v, _mm256_set1_epi16(512)),
        (true, false) => _mm256_mulhrs_epi16(v, _mm256_set1_epi16(8192)),
        (false, true) => _mm256_mulhrs_epi16(v, _mm256_set1_epi16(2048)),
        (true, true) => v,
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn yuv_v128<const PREP: bool, const BIL: bool>(v: __m128i) -> __m128i {
    match (PREP, BIL) {
        (false, false) => _mm_mulhrs_epi16(v, _mm_set1_epi16(512)),
        (true, false) => _mm_mulhrs_epi16(v, _mm_set1_epi16(8192)),
        (false, true) => _mm_mulhrs_epi16(v, _mm_set1_epi16(2048)),
        (true, true) => v,
    }
}

/// İki boyutlu ara satır: 8-tap `(Σ+2)>>2`, bilinear `Σ`.
#[target_feature(enable = "avx2")]
#[inline]
fn yuv_ara<const BIL: bool>(v: __m256i) -> __m256i {
    if BIL {
        v
    } else {
        _mm256_mulhrs_epi16(v, _mm256_set1_epi16(8192))
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn yuv_ara128<const BIL: bool>(v: __m128i) -> __m128i {
    if BIL {
        v
    } else {
        _mm_mulhrs_epi16(v, _mm_set1_epi16(8192))
    }
}

/// İki boyutlu son adım (`i32`): `put` `(Σ+512)>>10`, `prep` `(Σ+32)>>6`;
/// bilinear `put` `(Σ+128)>>8`, `prep` `(Σ+8)>>4`.
#[target_feature(enable = "avx2")]
#[inline]
fn yuv_son<const PREP: bool, const BIL: bool>(v: __m256i) -> __m256i {
    match (PREP, BIL) {
        (false, false) => _mm256_srai_epi32::<10>(_mm256_add_epi32(v, _mm256_set1_epi32(512))),
        (true, false) => _mm256_srai_epi32::<6>(_mm256_add_epi32(v, _mm256_set1_epi32(32))),
        (false, true) => _mm256_srai_epi32::<8>(_mm256_add_epi32(v, _mm256_set1_epi32(128))),
        (true, true) => _mm256_srai_epi32::<4>(_mm256_add_epi32(v, _mm256_set1_epi32(8))),
    }
}

// ---------------------------------------------------------------------------
// Yatay toplamlar (yuvarlamadan önce, `i16`)
// ---------------------------------------------------------------------------

/// 16 çıktı: `base` = satırın dolgu ofseti + ilk çıktının blok sütunu (`x`).
/// Okunan pencere `base+S .. base+S+15+N` (`W = 15+N` bayt): lane 0 çıktı
/// `0..7` pencerenin başından, lane 1 çıktı `8..15` pencerenin son 16
/// baytından (bölge dışına okumaz).
#[target_feature(enable = "avx2")]
#[inline]
fn h16<const N: usize, const W: usize>(src: &[u8], base: usize, c: &[__m256i; 4]) -> __m256i {
    const { assert!(W == 15 + N && N.is_multiple_of(2) && N >= 2 && N <= 8) };
    let s = (8 - N) / 2;
    let win: &[u8; W] = chunk(src, base + s);
    let lo = load_u8x16(chunk(win, 0));
    let hi = load_u8x16(chunk(win, W - 16));
    let v = _mm256_set_m128i(hi, lo);
    let mut acc = _mm256_setzero_si256();
    for (k, ck) in c.iter().enumerate().take(N / 2) {
        let m = v256(maske8(0, k), maske8(9 - N, k));
        acc = _mm256_add_epi16(acc, _mm256_maddubs_epi16(_mm256_shuffle_epi8(v, m), *ck));
    }
    acc
}

/// İki satır × 8 çıktı (`w = 8`): lane 0 satır `o0`, lane 1 satır `o1`
/// (dolgu satır ofsetleri). Satır başına 16 bayt (sütun 0..15; son satırda
/// bölgenin 1 bayt ötesi, [`MC_KUYRUK`] içinde).
#[target_feature(enable = "avx2")]
#[inline]
fn h8x2<const N: usize>(src: &[u8], o0: usize, o1: usize, c: &[__m256i; 4]) -> __m256i {
    let s = (8 - N) / 2;
    let v = _mm256_set_m128i(load_u8x16(chunk(src, o1)), load_u8x16(chunk(src, o0)));
    let mut acc = _mm256_setzero_si256();
    for (k, ck) in c.iter().enumerate().take(N / 2) {
        let m = v256(maske8(s, k), maske8(s, k));
        acc = _mm256_add_epi16(acc, _mm256_maddubs_epi16(_mm256_shuffle_epi8(v, m), *ck));
    }
    acc
}

/// Tek satır × 8 çıktı (`w = 8`, iki boyutlu ara satır).
#[target_feature(enable = "avx2")]
#[inline]
fn h8<const N: usize>(src: &[u8], o: usize, c: &[__m256i; 4]) -> __m128i {
    let s = (8 - N) / 2;
    let v = load_u8x16(chunk(src, o));
    let mut acc = _mm_setzero_si128();
    for (k, ck) in c.iter().enumerate().take(N / 2) {
        let m = v128(maske8(s, k));
        acc = _mm_add_epi16(
            acc,
            _mm_maddubs_epi16(_mm_shuffle_epi8(v, m), _mm256_castsi256_si128(*ck)),
        );
    }
    acc
}

/// İki satır × `n` çıktı (`w = n` ∈ {4, 2}) xmm'de. `w = 4`: satır başına 8
/// bayt sütun `S..S+7`; `w = 2`: sütun `0..7` (bölge `0..8` içinde).
#[target_feature(enable = "avx2")]
#[inline]
fn h_dar<const N: usize, const W: usize>(
    src: &[u8],
    o0: usize,
    o1: usize,
    c: &[__m256i; 4],
) -> __m128i {
    let s = (8 - N) / 2;
    let (bas, ofs) = if W == 4 { (s, 0) } else { (0, s) };
    let v = _mm_set_epi64x(yukle8(src, o1 + bas), yukle8(src, o0 + bas));
    let mut acc = _mm_setzero_si128();
    for (k, ck) in c.iter().enumerate().take(N / 2) {
        let m = v128(maske_iki(ofs, k, W));
        acc = _mm_add_epi16(
            acc,
            _mm_maddubs_epi16(_mm_shuffle_epi8(v, m), _mm256_castsi256_si128(*ck)),
        );
    }
    acc
}

/// Kayan pencere: ilk `n` öğe bir sola kayar, `n-1`. öğe `yeni` olur.
///
/// Döngüyle kaydırma (`a[k] = a[k+1]`) LLVM'de bellek kaydırmasına (yığın)
/// dönüşüyordu; yeni dizi kurmak öğeleri yazmaçta tutar (P32 disassembly).
#[inline(always)]
fn kaydir<T: Copy, const K: usize>(a: [T; K], n: usize, yeni: T) -> [T; K] {
    core::array::from_fn(|k| {
        if k + 1 < n {
            a[k + 1]
        } else if k + 1 == n {
            yeni
        } else {
            a[k]
        }
    })
}

// ---------------------------------------------------------------------------
// Kopya
// ---------------------------------------------------------------------------

/// `put` kopya yolu (dav1d `put_c`): satır başına sabit boyutlu kopya; ISA
/// gerektirmez. Genişlik dalı satır döngüsünün dışındadır.
#[inline]
fn kopya_put(hd: &mut Hedef<'_>, src: &[u8], ss: usize, w: usize, h: usize) {
    fn satirlar<const W: usize>(hd: &mut Hedef<'_>, src: &[u8], ss: usize, w: usize, h: usize) {
        let ds = hd.ds;
        for y in 0..h {
            let so = (y + 3) * ss + 3;
            for x in (0..w).step_by(W) {
                *chunk_mut::<u8, W>(hd.dst, y * ds + x) = *chunk(src, so + x);
            }
        }
    }
    match w {
        2 => satirlar::<2>(hd, src, ss, w, h),
        4 => satirlar::<4>(hd, src, ss, w, h),
        8 => satirlar::<8>(hd, src, ss, w, h),
        16 => satirlar::<16>(hd, src, ss, w, h),
        _ => satirlar::<32>(hd, src, ss, w, h),
    }
}

/// `prep` kopya yolu (dav1d `prep_c`): `tmp = s << 4`; ISA gerektirmez (LLVM
/// SSE2 ile vektörleştirir).
#[inline]
fn kopya_prep(hd: &mut Hedef<'_>, src: &[u8], ss: usize, w: usize, h: usize) {
    fn satirlar<const W: usize>(hd: &mut Hedef<'_>, src: &[u8], ss: usize, w: usize, h: usize) {
        for y in 0..h {
            let so = (y + 3) * ss + 3;
            for x in (0..w).step_by(W) {
                let s: &[u8; W] = chunk(src, so + x);
                let t: &mut [i16; W] = chunk_mut(hd.tmp, y * w + x);
                for (o, &p) in t.iter_mut().zip(s) {
                    *o = i16::from(p) << 4;
                }
            }
        }
    }
    match w {
        2 => satirlar::<2>(hd, src, ss, w, h),
        4 => satirlar::<4>(hd, src, ss, w, h),
        8 => satirlar::<8>(hd, src, ss, w, h),
        _ => satirlar::<16>(hd, src, ss, w, h),
    }
}

/// Kopya yolu, `w ≥ 16` (AVX2): `put` satır başına 32 baytlık (`w = 16`'da
/// 16) kopya, `prep` `vpmovzxbw` + `vpsllw 4` (16 piksel).
#[target_feature(enable = "avx2")]
fn kopya_genis<const PREP: bool>(hd: &mut Hedef<'_>, src: &[u8], ss: usize, w: usize, h: usize) {
    let mut so = 3 * ss + 3;
    if PREP {
        for y in 0..h {
            for x in (0..w).step_by(16) {
                let v = _mm256_cvtepu8_epi16(load_u8x16(chunk(src, so + x)));
                store_i16x16(chunk_mut(hd.tmp, y * w + x), _mm256_slli_epi16::<4>(v));
            }
            so += ss;
        }
    } else {
        let ds = hd.ds;
        let mut d = 0;
        if w == 16 {
            for _ in 0..h {
                *chunk_mut::<u8, 16>(hd.dst, d) = *chunk(src, so);
                so += ss;
                d += ds;
            }
        } else {
            for _ in 0..h {
                for x in (0..w).step_by(32) {
                    *chunk_mut::<u8, 32>(hd.dst, d + x) = *chunk(src, so + x);
                }
                so += ss;
                d += ds;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Yalnız yatay
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
fn yatay<const N: usize, const PREP: bool, const BIL: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    c: &[__m256i; 4],
) {
    match w {
        2 | 4 => {
            for y in (0..h).step_by(2) {
                let (o0, o1) = ((y + 3) * ss, (y + 4) * ss);
                let v = if w == 4 {
                    yuv_h128::<PREP, BIL>(h_dar::<N, 4>(src, o0, o1, c))
                } else {
                    yuv_h128::<PREP, BIL>(h_dar::<N, 2>(src, o0, o1, c))
                };
                if w == 4 {
                    sakla_dar::<PREP, 4>(hd, y, v);
                } else {
                    sakla_dar::<PREP, 2>(hd, y, v);
                }
            }
        }
        8 => {
            for y in (0..h).step_by(2) {
                let v = h8x2::<N>(src, (y + 3) * ss, (y + 4) * ss, c);
                sakla8x2::<PREP>(hd, y, yuv_h::<PREP, BIL>(v));
            }
        }
        16 => {
            for y in 0..h {
                let v = h16w::<N>(src, (y + 3) * ss, c);
                sakla16::<PREP>(hd, w, y, 0, yuv_h::<PREP, BIL>(v));
            }
        }
        _ => {
            for y in 0..h {
                let o = (y + 3) * ss;
                for x in (0..w).step_by(32) {
                    let a = yuv_h::<PREP, BIL>(h16w::<N>(src, o + x, c));
                    let b = yuv_h::<PREP, BIL>(h16w::<N>(src, o + x + 16, c));
                    if PREP {
                        store_i16x16(chunk_mut(hd.tmp, y * w + x), a);
                        store_i16x16(chunk_mut(hd.tmp, y * w + x + 16), b);
                    } else {
                        // `packuswb` lane içi: [x..x+7, x+16..x+23 | x+8..x+15, x+24..x+31]
                        // → `vpermq` 0xD8 ile sıralı 32 bayt.
                        let p = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(a, b));
                        store_u8x32(chunk_mut(hd.dst, y * hd.ds + x), p);
                    }
                }
            }
        }
    }
}

/// [`h16`]'yı `N`'ye uyan pencere boyuyla çağırır.
#[target_feature(enable = "avx2")]
#[inline]
fn h16w<const N: usize>(src: &[u8], base: usize, c: &[__m256i; 4]) -> __m256i {
    match N {
        8 => h16::<8, 23>(src, base, c),
        6 => h16::<6, 21>(src, base, c),
        4 => h16::<4, 19>(src, base, c),
        _ => h16::<2, 17>(src, base, c),
    }
}

// ---------------------------------------------------------------------------
// Yalnız dikey
// ---------------------------------------------------------------------------

/// İki çıkış satırı × 16 (`i16`): `a0` sütun 0..7, `a1` sütun 8..15; lane 0
/// satır `y`, lane 1 satır `y+1`.
#[target_feature(enable = "avx2")]
#[inline]
fn sakla16x2<const PREP: bool>(
    hd: &mut Hedef<'_>,
    w: usize,
    y: usize,
    x: usize,
    a0: __m256i,
    a1: __m256i,
) {
    if PREP {
        store_i16x16(
            chunk_mut(hd.tmp, y * w + x),
            _mm256_permute2x128_si256::<0x20>(a0, a1),
        );
        store_i16x16(
            chunk_mut(hd.tmp, (y + 1) * w + x),
            _mm256_permute2x128_si256::<0x31>(a0, a1),
        );
    } else {
        let p = _mm256_packus_epi16(a0, a1);
        let ds = hd.ds;
        store_u8x16(chunk_mut(hd.dst, y * ds + x), _mm256_castsi256_si128(p));
        store_u8x16(
            chunk_mut(hd.dst, (y + 1) * ds + x),
            _mm256_extracti128_si256::<1>(p),
        );
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn dikey<const N: usize, const PREP: bool, const BIL: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    c: &[__m256i; 4],
) {
    let s = (8 - N) / 2;
    let n = N / 2;
    // Dolgu satırı `j`, blok sütunu `x` → ofset `j*ss + 3 + x`.
    match w {
        2 | 4 => {
            let yuk = |j: usize| {
                if w == 4 {
                    yukle4(src, j * ss + 3)
                } else {
                    yukle2(src, j * ss + 3)
                }
            };
            // `R[k]` = (çıkış satırı y ikilisi k) ⊕ (y+1 ikilisi k).
            let birlestir = |a: __m128i, b: __m128i| {
                if w == 4 {
                    _mm_unpacklo_epi64(a, b)
                } else {
                    _mm_unpacklo_epi32(a, b)
                }
            };
            let mut r = [_mm_setzero_si128(); 4];
            let mut satir = [_mm_setzero_si128(); 9];
            for (j, sj) in satir.iter_mut().enumerate().take(N + 1) {
                *sj = yuk(s + j);
            }
            for k in 0..n {
                let p0 = _mm_unpacklo_epi8(satir[2 * k], satir[2 * k + 1]);
                let p1 = _mm_unpacklo_epi8(satir[2 * k + 1], satir[2 * k + 2]);
                r[k] = birlestir(p0, p1);
            }
            let mut son = satir[N];
            let mut y = 0;
            loop {
                let mut acc = _mm_setzero_si128();
                for k in 0..n {
                    acc = _mm_add_epi16(acc, _mm_maddubs_epi16(r[k], _mm256_castsi256_si128(c[k])));
                }
                let v = yuv_v128::<PREP, BIL>(acc);
                if w == 4 {
                    sakla_dar::<PREP, 4>(hd, y, v);
                } else {
                    sakla_dar::<PREP, 2>(hd, y, v);
                }
                y += 2;
                if y >= h {
                    break;
                }
                let a = yuk(y + s + N - 1);
                let b = yuk(y + s + N);
                r = kaydir(
                    r,
                    n,
                    birlestir(_mm_unpacklo_epi8(son, a), _mm_unpacklo_epi8(a, b)),
                );
                son = b;
            }
        }
        8 => {
            let yuk = |j: usize| _mm_cvtsi64_si128(yukle8(src, j * ss + 3));
            let mut r = [_mm256_setzero_si256(); 4];
            let mut satir = [_mm_setzero_si128(); 9];
            for (j, sj) in satir.iter_mut().enumerate().take(N + 1) {
                *sj = yuk(s + j);
            }
            for k in 0..n {
                let p0 = _mm_unpacklo_epi8(satir[2 * k], satir[2 * k + 1]);
                let p1 = _mm_unpacklo_epi8(satir[2 * k + 1], satir[2 * k + 2]);
                r[k] = _mm256_set_m128i(p1, p0);
            }
            let mut son = satir[N];
            let mut y = 0;
            loop {
                let mut acc = _mm256_setzero_si256();
                for k in 0..n {
                    acc = _mm256_add_epi16(acc, _mm256_maddubs_epi16(r[k], c[k]));
                }
                sakla8x2::<PREP>(hd, y, yuv_v::<PREP, BIL>(acc));
                y += 2;
                if y >= h {
                    break;
                }
                let a = yuk(y + s + N - 1);
                let b = yuk(y + s + N);
                r = kaydir(
                    r,
                    n,
                    _mm256_set_m128i(_mm_unpacklo_epi8(a, b), _mm_unpacklo_epi8(son, a)),
                );
                son = b;
            }
        }
        _ => {
            // 16 sütunluk şerit, iki çıkış satırı bir ymm'de (lane 0 satır `y`,
            // lane 1 satır `y+1`): `X_j = [satır j | satır j+1]`; ikili `k`
            // `unpack(X_j, X_{j+1})` (`j = y+S+2k`) iki satırın ikilisini birden
            // kurar; ikililer iki satırlık kayan pencerede.
            for x in (0..w).step_by(16) {
                let yuk = |j: usize| load_u8x16(chunk(src, j * ss + 3 + x));
                let mut lo = [_mm256_setzero_si256(); 4];
                let mut hi = [_mm256_setzero_si256(); 4];
                let mut satir = [_mm_setzero_si128(); 9];
                for (j, sj) in satir.iter_mut().enumerate().take(N + 1) {
                    *sj = yuk(s + j);
                }
                for k in 0..n {
                    let x0 = _mm256_set_m128i(satir[2 * k + 1], satir[2 * k]);
                    let x1 = _mm256_set_m128i(satir[2 * k + 2], satir[2 * k + 1]);
                    lo[k] = _mm256_unpacklo_epi8(x0, x1);
                    hi[k] = _mm256_unpackhi_epi8(x0, x1);
                }
                let mut son = satir[N];
                let mut y = 0;
                // Sonraki iki yeni satırın ofseti (satır `y+S+N−1`, `y = 2`'de).
                let mut oa = (s + N + 1) * ss + 3 + x;
                loop {
                    let mut a0 = _mm256_setzero_si256();
                    let mut a1 = _mm256_setzero_si256();
                    for k in 0..n {
                        a0 = _mm256_add_epi16(a0, _mm256_maddubs_epi16(lo[k], c[k]));
                        a1 = _mm256_add_epi16(a1, _mm256_maddubs_epi16(hi[k], c[k]));
                    }
                    sakla16x2::<PREP>(hd, w, y, x, yuv_v::<PREP, BIL>(a0), yuv_v::<PREP, BIL>(a1));
                    y += 2;
                    if y >= h {
                        break;
                    }
                    let a = load_u8x16(chunk(src, oa));
                    let b = load_u8x16(chunk(src, oa + ss));
                    oa += 2 * ss;
                    let x0 = _mm256_set_m128i(a, son);
                    let x1 = _mm256_set_m128i(b, a);
                    lo = kaydir(lo, n, _mm256_unpacklo_epi8(x0, x1));
                    hi = kaydir(hi, n, _mm256_unpackhi_epi8(x0, x1));
                    son = b;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// İki boyutlu
// ---------------------------------------------------------------------------

/// İki boyutlu yol, `w = W` ∈ {2, 4}: ara satırlar xmm'nin alt `W`
/// kelimesinde kayan pencerede; ilk ara satır tek, sonrakiler bir `h_dar`
/// çağrısında ikişer (satır `j`, `j+1`) üretilir — okunan son satır tam gereken
/// satırdır (`h+SV+NV−2`). İki çıkış satırı bir ymm'de (lane 0 satır `y`, lane 1
/// satır `y+1`). Kapanış yok: satır içine alınmayan kapanış çağrısı (P32
/// disassembly) sabit maliyeti artırıyordu.
#[target_feature(enable = "avx2")]
#[inline]
fn iki_dar<const NH: usize, const NV: usize, const PREP: bool, const BIL: bool, const W: usize>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    h: usize,
    ch: &[__m256i; 4],
    cv: &[__m256i; 4],
) {
    let sv = (8 - NV) / 2;
    let n = NV / 2;
    // Pencere: satır `y+SV .. y+SV+NV` (NV+1 satır).
    let mut m = [_mm_setzero_si128(); 10];
    m[0] = yuv_ara128::<BIL>(h_dar::<NH, W>(src, sv * ss, sv * ss, ch));
    for k in 0..n {
        let j = sv + 1 + 2 * k;
        let v = yuv_ara128::<BIL>(h_dar::<NH, W>(src, j * ss, (j + 1) * ss, ch));
        m[1 + 2 * k] = v;
        m[2 + 2 * k] = if W == 4 {
            _mm_unpackhi_epi64(v, v)
        } else {
            _mm_srli_epi64::<32>(v)
        };
    }
    let mut y = 0;
    loop {
        if BIL {
            // `u16` yolu: satır `y` = (m0, m1), satır `y+1` = (m1, m2) bir xmm'de.
            let (x0, x1) = if W == 4 {
                (
                    _mm_unpacklo_epi64(m[0], m[1]),
                    _mm_unpacklo_epi64(m[1], m[2]),
                )
            } else {
                (
                    _mm_unpacklo_epi32(m[0], m[1]),
                    _mm_unpacklo_epi32(m[1], m[2]),
                )
            };
            let v = bilin_dikey128::<PREP>(x0, x1, cv);
            if W == 4 {
                sakla_dar::<PREP, 4>(hd, y, v);
            } else {
                sakla_dar::<PREP, 2>(hd, y, v);
            }
        } else {
            let mut acc = _mm256_setzero_si256();
            for k in 0..n {
                let (a, b, c) = (m[2 * k], m[2 * k + 1], m[2 * k + 2]);
                let p = _mm256_set_m128i(_mm_unpacklo_epi16(b, c), _mm_unpacklo_epi16(a, b));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, cv[k]));
            }
            let r = _mm256_packs_epi32(yuv_son::<PREP, BIL>(acc), _mm256_setzero_si256());
            let (lo, hi) = (_mm256_castsi256_si128(r), _mm256_extracti128_si256::<1>(r));
            if W == 4 {
                sakla_dar::<PREP, 4>(hd, y, _mm_unpacklo_epi64(lo, hi));
            } else {
                sakla_dar::<PREP, 2>(hd, y, _mm_unpacklo_epi32(lo, hi));
            }
        }
        y += 2;
        if y >= h {
            break;
        }
        let j = y + sv + NV - 1;
        let v = yuv_ara128::<BIL>(h_dar::<NH, W>(src, j * ss, (j + 1) * ss, ch));
        let b = if W == 4 {
            _mm_unpackhi_epi64(v, v)
        } else {
            _mm_srli_epi64::<32>(v)
        };
        m = core::array::from_fn(|k| {
            if k + 2 <= NV {
                m[k + 2]
            } else if k == NV - 1 {
                v
            } else if k == NV {
                b
            } else {
                _mm_setzero_si128()
            }
        });
    }
}

/// İki boyutlu yol: `NH` tap yatay → `i16` ara satır, `NV` tap dikey
/// (`pmaddwd`). Çıkış satırı `y`, ara satırlar `y+SV .. y+SV+NV−1`
/// (`SV = (8−NV)/2`); ara satır `j` = dolgu satırı `j`'nin yatay filtresi.
#[target_feature(enable = "avx2")]
fn iki<const NH: usize, const NV: usize, const PREP: bool, const BIL: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    ch: &[__m256i; 4],
    cv: &[__m256i; 4],
) {
    let sv = (8 - NV) / 2;
    let n = NV / 2;
    match w {
        4 => iki_dar::<NH, NV, PREP, BIL, 4>(hd, src, ss, h, ch, cv),
        2 => iki_dar::<NH, NV, PREP, BIL, 2>(hd, src, ss, h, ch, cv),
        8 => {
            // Ara satırlar xmm (8 kelime); ilk ara satır tek, sonrakiler `h8x2`
            // ile ikişer. İki çıkış satırı bir ymm'de.
            let ara2 = |j: usize| -> (__m128i, __m128i) {
                let v = yuv_ara::<BIL>(h8x2::<NH>(src, j * ss, (j + 1) * ss, ch));
                (_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v))
            };
            let mut m = [_mm_setzero_si128(); 10];
            m[0] = yuv_ara128::<BIL>(h8::<NH>(src, sv * ss, ch));
            for k in 0..n {
                let (a, b) = ara2(sv + 1 + 2 * k);
                m[1 + 2 * k] = a;
                m[2 + 2 * k] = b;
            }
            let mut y = 0;
            loop {
                let r = if BIL {
                    // `u16` yolu: lane 0 satır `y` (m0, m1), lane 1 satır `y+1` (m1, m2).
                    let x0 = _mm256_set_m128i(m[1], m[0]);
                    let x1 = _mm256_set_m128i(m[2], m[1]);
                    bilin_dikey::<PREP>(x0, x1, cv)
                } else {
                    let mut lo = _mm256_setzero_si256();
                    let mut hi = _mm256_setzero_si256();
                    for k in 0..n {
                        let (a, b, c) = (m[2 * k], m[2 * k + 1], m[2 * k + 2]);
                        let pl =
                            _mm256_set_m128i(_mm_unpacklo_epi16(b, c), _mm_unpacklo_epi16(a, b));
                        let ph =
                            _mm256_set_m128i(_mm_unpackhi_epi16(b, c), _mm_unpackhi_epi16(a, b));
                        lo = _mm256_add_epi32(lo, _mm256_madd_epi16(pl, cv[k]));
                        hi = _mm256_add_epi32(hi, _mm256_madd_epi16(ph, cv[k]));
                    }
                    _mm256_packs_epi32(yuv_son::<PREP, BIL>(lo), yuv_son::<PREP, BIL>(hi))
                };
                sakla8x2::<PREP>(hd, y, r);
                y += 2;
                if y >= h {
                    break;
                }
                let (a, b) = ara2(y + sv + NV - 1);
                m = core::array::from_fn(|k| {
                    if k + 2 <= NV {
                        m[k + 2]
                    } else if k == NV - 1 {
                        a
                    } else if k == NV {
                        b
                    } else {
                        _mm_setzero_si128()
                    }
                });
            }
        }
        _ => {
            for x in (0..w).step_by(16) {
                let ara = |j: usize| yuv_ara::<BIL>(h16w::<NH>(src, j * ss + x, ch));
                let mut m = [_mm256_setzero_si256(); 8];
                for (j, mj) in m.iter_mut().enumerate().take(NV - 1) {
                    *mj = ara(sv + j);
                }
                for y in 0..h {
                    m[NV - 1] = ara(y + sv + NV - 1);
                    let r = if BIL {
                        bilin_dikey::<PREP>(m[0], m[1], cv)
                    } else {
                        let mut lo = _mm256_setzero_si256();
                        let mut hi = _mm256_setzero_si256();
                        for k in 0..n {
                            let a = m[2 * k];
                            let b = m[2 * k + 1];
                            lo = _mm256_add_epi32(
                                lo,
                                _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), cv[k]),
                            );
                            hi = _mm256_add_epi32(
                                hi,
                                _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), cv[k]),
                            );
                        }
                        _mm256_packs_epi32(yuv_son::<PREP, BIL>(lo), yuv_son::<PREP, BIL>(hi))
                    };
                    sakla16::<PREP>(hd, w, y, x, r);
                    m = kaydir(m, NV, m[NV - 1]);
                }
            }
        }
    }
}

// Derleme zamanı denetimi: kuyruk, `w = 8` yatay geçişinin 1 baytlık aşımını
// kapsar.
const _: () = assert!(MC_KUYRUK >= 1);
