//! MC 16 bpc (10/12 bit) AVX2 `put`/`prep` çekirdekleri — F4-MC c (P34):
//! `mc[0..9]`, `mct[0..9]` (9 adet 8-tap birleşimi + bilinear).
//!
//! ## Sözleşme (P32 kopyasız sözleşmesinin 16 bpc karşılığı)
//!
//! `src` dolgu hücresinin başıdır (blok satırı −3, sütunu −3); pikseller `u16`
//! (yerel bayt sırası), satır adımları **bayt**. Girişte O(1)
//! [`crate::dsp::mc::simd16_sozlesmesi`]: `w` 2'nin kuvveti 2…128, `h` çift
//! 2…128, `mx, my < 16`, kaynak satır adımı `≥ (w+7)·2`, kaynak dilimi
//! `(h+6)·ss + (w+7)·2 + K` bayt, bit derinliği 10/12 ve `bitdepth_max` tutarlı;
//! `put` hedefi `(h−1)·ds + 2w` bayt, `prep` `tmp` `w·h` öğe. Sözleşme dışı
//! girdi skalere devreder ve sayılır.
//!
//! **Okuma bölgesi:** satır `0..=h+6`, sütun `0..=w+6` (`(h+7)` × `(w+7)` piksel).
//! `w ≥ 16` yolunun 16 piksellik yüklemeleri bölgenin içinde biter; `w ≤ 8`
//! yolu 8 piksellik yüklemelerle 8 çıktı hesaplar ve satır bölgesini en çok 6
//! piksel (12 bayt ≤ `K`) aşar (`w = 2`: yatay sütun 14'e kadar, bölge 8'e
//! kadar). Aşan baytlar çıktıya karışmaz (yalnız yazılmayan sütunlara gider).
//! **Yazma** yalnız `h` satır × `w` piksel (`prep`: `w·h` öğe).
//!
//! ## Yöntem
//!
//! - **Tap ikilileri:** `N`-tap filtre `S = (8−N)/2`. sütundan başlayan `P = N/2`
//!   ikiliyle (`pmaddwd`, katsayı ikilisi `(f[S+2i], f[S+2i+1])`). 8-tap tabloları
//!   (6/4-tap tabloların dış katsayıları sıfır) `P = 4, S = 0`; bilinear
//!   `(16−m, m)` `P = 1, S = 3`. Pikseller ≤ 4095 olduğundan `i16` olarak çarpılır.
//! - **Yatay:** 16 çıktı için çift çıktılar `x + S + 2i`, tek çıktılar
//!   `x + S + 2i + 1` piksel ofsetinden 32 baytlık yüklemelerle (`dword j` =
//!   çıktı `x + 2j` / `x + 2j + 1`'in tap ikilisi); `punpck{l,h}dq` toplamları
//!   şerit düzenine `[0..3 | 8..11]`, `[4..7 | 12..15]` getirir (dikey yolla aynı),
//!   `packusdw`/`packssdw` sıralı 16 değer verir.
//! - **Dikey:** satır ikilileri `punpck{l,h}wd` + `pmaddwd`.
//! - **İki boyutlu:** yatay ara satır `i16` (dav1d `int16_t mid`; 64 ölçekli
//!   filtrelerle ara değer `i16`'ya sığar), 8 satırlık halka (genişlik sınıfına
//!   göre monomorf), dikey `pmaddwd`.
//! - **Yuvarlama** (`ib` = 10 bitte 4, 12 bitte 2; dav1d `mc_tmpl.c`): 8-tap ara
//!   `(Σ + r) >> (6−ib)`, `put` yalnız yatay `(Σ + 32 + r) >> 6`, yalnız dikey
//!   `(Σ + 32) >> 6`, iki boyutlu `(Σ + r') >> (6+ib)`; `prep` yalnız yatay/dikey
//!   `(Σ + r) >> (6−ib)`, iki boyutlu `(Σ + 32) >> 6`, sonra `− PREP_BIAS`
//!   (8192); kopya `(p << ib) − 8192`. Bilinear aynı yapıda 4'lük kaydırmalarla.
//!   Kaydırma sayıları çağrı başına bir kez yazmaca alınır (`psrad xmm`).
//!   `put` kırpması `packusdw` + `pminuw(bitdepth_max)`.
//!
//! `unsafe` yalnız belirteçli girişlerdeki `#[target_feature]` çağrısıdır
//! (`put16_giris`/`prep16_giris`); dosya `audit::UNSAFE_EXEMPTIONS` defterinde
//! kayıtlıdır. Yükleme/saklama [`super::mem`] yardımcılarıyla yapılır.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mc::{BILIN, filtre_satiri, prep_skaler, put_skaler};
use super::mem::{
    chunk, chunk_mut, load_u8x16, load_u8x32, store_i16x8, store_i16x16, store_u8x4, store_u8x8,
    store_u8x16, store_u8x32,
};
use crate::cpu::Avx2Token;
use crate::dsp::mc::{McPrepFn, McPutFn, simd16_sozlesmesi, skaler_devri_say};
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi32, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_cvtsi128_si64,
    _mm_madd_epi16, _mm_min_epu16, _mm_packs_epi32, _mm_packus_epi32, _mm_set1_epi16,
    _mm_set1_epi32, _mm_setzero_si128, _mm_sll_epi16, _mm_sra_epi32, _mm_sub_epi16, _mm_sub_epi32,
    _mm_unpackhi_epi16, _mm_unpackhi_epi32, _mm_unpacklo_epi16, _mm_unpacklo_epi32,
    _mm256_add_epi32, _mm256_madd_epi16, _mm256_min_epu16, _mm256_packs_epi32, _mm256_packus_epi32,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setzero_si256, _mm256_sll_epi16, _mm256_sra_epi32,
    _mm256_sub_epi16, _mm256_sub_epi32, _mm256_unpackhi_epi16, _mm256_unpackhi_epi32,
    _mm256_unpacklo_epi16, _mm256_unpacklo_epi32,
};

/// `prep` ara değerinden çıkarılan sabit (dav1d `PREP_BIAS`, 16 bpc).
const PREP_BIAS: i32 = 8192;

/// Hedef: `put` için bayt düzlemi (adım bayt), `prep` için `tmp` (adım `w`).
pub(super) struct Hedef<'a> {
    pub(super) dst: &'a mut [u8],
    pub(super) ds: usize,
    pub(super) tmp: &'a mut [i16],
}

/// Bir geçişin yuvarlaması: `(Σ + r) >> sh` (kaydırma çağrı başına sabit).
#[derive(Clone, Copy)]
struct Yuv {
    r: i32,
    sh: i32,
}

impl Yuv {
    const fn yeni(sh: i32) -> Self {
        Yuv {
            r: (1 << sh) >> 1,
            sh,
        }
    }
}

/// Çıktıya özgü değerler: `bias` (`prep`), `max` (`put` kırpma sınırı).
#[derive(Clone, Copy)]
struct Cikti {
    bias: i32,
    max: i32,
}

/// Tap ikilisi katsayıları `(f[S+2i], f[S+2i+1])` → `pmaddwd` sözcükleri.
fn ikililer<const P: usize, const S: usize>(f: &[i8; 8]) -> [i32; 4] {
    let mut k = [0i32; 4];
    for (i, ki) in k.iter_mut().enumerate().take(P) {
        let a = i32::from(f[S + 2 * i]);
        let b = i32::from(f[S + 2 * i + 1]);
        *ki = (a & 0xffff) | (b << 16);
    }
    k
}

/// Bilinear katsayı ikilisi `(16−m, m)`.
const fn bilin_ikili(m: u8) -> [i32; 4] {
    let m = m as i32;
    [((16 - m) & 0xffff) | (m << 16), 0, 0, 0]
}

// ---------------------------------------------------------------------------
// Giriş
// ---------------------------------------------------------------------------

/// `put` girişi: sözleşme denetimi + AVX2 gövdesi ya da skaler devir.
fn put16_giris<const FH: usize, const FV: usize>(
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
    let ok = dst_stride >= 2 * w
        && h >= 1
        && simd16_sozlesmesi(
            src.len(),
            src_stride,
            w,
            h,
            mx,
            my,
            bit_depth,
            bitdepth_max,
            dst.len(),
            (h - 1) * dst_stride + 2 * w,
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
    let mut hd = Hedef {
        dst,
        ds: dst_stride,
        tmp: &mut [],
    };
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`put16_fn`); belirteç AVX2 tespitinin kanıtıdır. (2) sınır:
    // `simd16_sozlesmesi` kaynak bölgesini + `MC_KUYRUK`'u ve hedefte `h` satır ×
    // `2w` baytı kapsar; gövde erişimleri dilim/dizi denetimlidir. (3)
    // hizalama/takma ad: `dst` tek `&mut`, `src` `&`; ödünç kuralları çakışmayı dışlar.
    unsafe { govde::<FH, FV, false>(&mut hd, src, src_stride, w, h, mx, my, bitdepth_max) }
}

/// `prep` girişi: sözleşme denetimi + AVX2 gövdesi ya da skaler devir.
fn prep16_giris<const FH: usize, const FV: usize>(
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
    let ok = simd16_sozlesmesi(
        src.len(),
        src_stride,
        w,
        h,
        mx,
        my,
        bit_depth,
        bitdepth_max,
        tmp.len(),
        w * h,
    );
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`prep16_fn`); belirteç AVX2 tespitinin kanıtıdır. (2) sınır:
    // `simd16_sozlesmesi` kaynak bölgesini + `MC_KUYRUK`'u ve `tmp`'de `w·h` öğeyi
    // kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `tmp` tek `&mut`,
    // `src` `&`.
    unsafe { govde::<FH, FV, true>(&mut hd, src, src_stride, w, h, mx, my, bitdepth_max) }
}

/// Yol seçimi: kopya / yalnız yatay / yalnız dikey / iki boyutlu; filtre
/// satırları ve yuvarlamalar bir kez.
#[target_feature(enable = "avx2")]
fn govde<const FH: usize, const FV: usize, const PREP: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bitdepth_max: u16,
) {
    let max = i32::from(bitdepth_max);
    let ib = if max == 4095 { 2 } else { 4 };
    let cikti = Cikti {
        bias: PREP_BIAS,
        max,
    };
    let bil = FH == BILIN;
    // Ara/tek geçiş kaydırması: 8-tap 6 − ib, bilinear 4 − ib.
    let sh_ara = if bil { 4 - ib } else { 6 - ib };
    let tam = if bil { 4 } else { 6 };
    match (mx != 0, my != 0) {
        (false, false) => kopya::<PREP>(hd, src, ss, w, h, ib),
        (true, false) => {
            // `put`: (Σ + r_ara + (r_ib << sh_ara)) >> tam ; `prep`: (Σ + r_ara) >> sh_ara − bias.
            let yuv = if PREP {
                Yuv::yeni(sh_ara)
            } else {
                let ara = Yuv::yeni(sh_ara);
                Yuv {
                    r: ara.r + (((1 << ib) >> 1) << sh_ara),
                    sh: tam,
                }
            };
            if bil {
                yol_h::<PREP, 1, 3>(hd, src, ss, w, h, bilin_ikili(mx), yuv, cikti);
            } else {
                let f = filtre_satiri(FH, w, mx);
                yol_h::<PREP, 4, 0>(hd, src, ss, w, h, ikililer::<4, 0>(f), yuv, cikti);
            }
        }
        (false, true) => {
            // `put`: (Σ + r) >> tam ; `prep`: (Σ + r_ara) >> sh_ara − bias.
            let yuv = if PREP {
                Yuv::yeni(sh_ara)
            } else {
                Yuv::yeni(tam)
            };
            if bil {
                yol_v::<PREP, 1, 3>(hd, src, ss, w, h, bilin_ikili(my), yuv, cikti);
            } else {
                let f = filtre_satiri(FV, h, my);
                yol_v::<PREP, 4, 0>(hd, src, ss, w, h, ikililer::<4, 0>(f), yuv, cikti);
            }
        }
        (true, true) => {
            let ara = Yuv::yeni(sh_ara);
            // `put`: (Σ + r) >> (tam + ib) ; `prep`: (Σ + r) >> tam − bias.
            let son = if PREP {
                Yuv::yeni(tam)
            } else {
                Yuv::yeni(tam + ib)
            };
            if bil {
                yol_hv::<PREP, 1, 3>(
                    hd,
                    src,
                    ss,
                    w,
                    h,
                    bilin_ikili(mx),
                    bilin_ikili(my),
                    ara,
                    son,
                    cikti,
                );
            } else {
                let fh = filtre_satiri(FH, w, mx);
                let fv = filtre_satiri(FV, h, my);
                yol_hv::<PREP, 4, 0>(
                    hd,
                    src,
                    ss,
                    w,
                    h,
                    ikililer::<4, 0>(fh),
                    ikililer::<4, 0>(fv),
                    ara,
                    son,
                    cikti,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ortak yardımcılar
// ---------------------------------------------------------------------------

/// 16 çıktının yatay tap toplamları (`i32`, şerit düzeni `[0..3 | 8..11]`,
/// `[4..7 | 12..15]`); `o` dolgu satırı + sütun `x`'in bayt ofseti.
#[target_feature(enable = "avx2")]
#[inline]
fn yatay16<const P: usize, const S: usize>(
    src: &[u8],
    o: usize,
    k: &[__m256i; 4],
) -> (__m256i, __m256i) {
    let mut e = _mm256_setzero_si256();
    let mut t = _mm256_setzero_si256();
    for (i, ki) in k.iter().enumerate().take(P) {
        let a = load_u8x32(chunk(src, o + 2 * (S + 2 * i)));
        let b = load_u8x32(chunk(src, o + 2 * (S + 2 * i + 1)));
        e = _mm256_add_epi32(e, _mm256_madd_epi16(a, *ki));
        t = _mm256_add_epi32(t, _mm256_madd_epi16(b, *ki));
    }
    (_mm256_unpacklo_epi32(e, t), _mm256_unpackhi_epi32(e, t))
}

/// 8 çıktının yatay tap toplamları (`lo` çıktı 0..3, `hi` 4..7).
#[target_feature(enable = "avx2")]
#[inline]
fn yatay8<const P: usize, const S: usize>(
    src: &[u8],
    o: usize,
    k: &[__m128i; 4],
) -> (__m128i, __m128i) {
    let mut e = _mm_setzero_si128();
    let mut t = _mm_setzero_si128();
    for (i, ki) in k.iter().enumerate().take(P) {
        let a = load_u8x16(chunk(src, o + 2 * (S + 2 * i)));
        let b = load_u8x16(chunk(src, o + 2 * (S + 2 * i + 1)));
        e = _mm_add_epi32(e, _mm_madd_epi16(a, *ki));
        t = _mm_add_epi32(t, _mm_madd_epi16(b, *ki));
    }
    (_mm_unpacklo_epi32(e, t), _mm_unpackhi_epi32(e, t))
}

/// `(v + r) >> sh` (ymm).
#[target_feature(enable = "avx2")]
#[inline]
fn yuv256(v: __m256i, r: __m256i, sh: __m128i) -> __m256i {
    _mm256_sra_epi32(_mm256_add_epi32(v, r), sh)
}

/// `(v + r) >> sh` (xmm).
#[target_feature(enable = "avx2")]
#[inline]
fn yuv128(v: __m128i, r: __m128i, sh: __m128i) -> __m128i {
    _mm_sra_epi32(_mm_add_epi32(v, r), sh)
}

/// 16 çıktıyı yazar: `put` `packusdw` + `pminuw(max)`, `prep` `− bias` + `packssdw`.
#[target_feature(enable = "avx2")]
#[inline]
fn yaz16<const PREP: bool>(
    hd: &mut Hedef<'_>,
    w: usize,
    y: usize,
    x: usize,
    lo: __m256i,
    hi: __m256i,
    c: (__m256i, __m256i),
) {
    if PREP {
        let v = _mm256_packs_epi32(_mm256_sub_epi32(lo, c.0), _mm256_sub_epi32(hi, c.0));
        store_i16x16(chunk_mut(hd.tmp, y * w + x), v);
    } else {
        let v = _mm256_min_epu16(_mm256_packus_epi32(lo, hi), c.1);
        store_u8x32(chunk_mut(hd.dst, y * hd.ds + 2 * x), v);
    }
}

/// 8 çıktıdan `w` (2/4/8) tanesini yazar.
#[target_feature(enable = "avx2")]
#[inline]
fn yaz8<const PREP: bool>(
    hd: &mut Hedef<'_>,
    w: usize,
    y: usize,
    lo: __m128i,
    hi: __m128i,
    c: (__m128i, __m128i),
) {
    if PREP {
        let v = _mm_packs_epi32(_mm_sub_epi32(lo, c.0), _mm_sub_epi32(hi, c.0));
        let o = y * w;
        match w {
            8 => store_i16x8(chunk_mut(hd.tmp, o), v),
            4 => {
                let b = _mm_cvtsi128_si64(v).to_le_bytes();
                for (k, t) in hd.tmp[o..o + 4].iter_mut().enumerate() {
                    *t = i16::from_le_bytes([b[2 * k], b[2 * k + 1]]);
                }
            }
            _ => {
                let b = _mm_cvtsi128_si32(v).to_le_bytes();
                hd.tmp[o] = i16::from_le_bytes([b[0], b[1]]);
                hd.tmp[o + 1] = i16::from_le_bytes([b[2], b[3]]);
            }
        }
    } else {
        let v = _mm_min_epu16(_mm_packus_epi32(lo, hi), c.1);
        let o = y * hd.ds;
        match w {
            8 => store_u8x16(chunk_mut(hd.dst, o), v),
            4 => store_u8x8(chunk_mut(hd.dst, o), v),
            _ => store_u8x4(chunk_mut(hd.dst, o), v),
        }
    }
}

/// Katsayı ikilileri, yuvarlama ve çıktı değerleri vektörlerde (ymm, xmm).
#[target_feature(enable = "avx2")]
#[inline]
fn kat256(k: [i32; 4]) -> [__m256i; 4] {
    [
        _mm256_set1_epi32(k[0]),
        _mm256_set1_epi32(k[1]),
        _mm256_set1_epi32(k[2]),
        _mm256_set1_epi32(k[3]),
    ]
}

#[target_feature(enable = "avx2")]
#[inline]
fn kat128(k: [i32; 4]) -> [__m128i; 4] {
    [
        _mm_set1_epi32(k[0]),
        _mm_set1_epi32(k[1]),
        _mm_set1_epi32(k[2]),
        _mm_set1_epi32(k[3]),
    ]
}

// ---------------------------------------------------------------------------
// Yollar
// ---------------------------------------------------------------------------

/// Kopya: `put` satır kopyası, `prep` `(p << ib) − 8192`.
#[target_feature(enable = "avx2")]
fn kopya<const PREP: bool>(hd: &mut Hedef<'_>, src: &[u8], ss: usize, w: usize, h: usize, ib: i32) {
    let ibv = _mm_cvtsi32_si128(ib);
    for y in 0..h {
        let s0 = (y + 3) * ss + 6;
        if PREP {
            let bias = _mm256_set1_epi16(PREP_BIAS as i16);
            let o = y * w;
            if w >= 16 {
                let mut x = 0;
                while x < w {
                    let v = _mm256_sll_epi16(load_u8x32(chunk(src, s0 + 2 * x)), ibv);
                    store_i16x16(chunk_mut(hd.tmp, o + x), _mm256_sub_epi16(v, bias));
                    x += 16;
                }
            } else {
                let b8 = _mm_set1_epi16(PREP_BIAS as i16);
                let v = _mm_sub_epi16(_mm_sll_epi16(load_u8x16(chunk(src, s0)), ibv), b8);
                match w {
                    8 => store_i16x8(chunk_mut(hd.tmp, o), v),
                    4 => {
                        let b = _mm_cvtsi128_si64(v).to_le_bytes();
                        for (k, t) in hd.tmp[o..o + 4].iter_mut().enumerate() {
                            *t = i16::from_le_bytes([b[2 * k], b[2 * k + 1]]);
                        }
                    }
                    _ => {
                        let b = _mm_cvtsi128_si32(v).to_le_bytes();
                        hd.tmp[o] = i16::from_le_bytes([b[0], b[1]]);
                        hd.tmp[o + 1] = i16::from_le_bytes([b[2], b[3]]);
                    }
                }
            }
        } else {
            let d0 = y * hd.ds;
            match w {
                2 => *chunk_mut::<u8, 4>(hd.dst, d0) = *chunk(src, s0),
                4 => *chunk_mut::<u8, 8>(hd.dst, d0) = *chunk(src, s0),
                8 => store_u8x16(chunk_mut(hd.dst, d0), load_u8x16(chunk(src, s0))),
                16 => kopya_satir::<32>(hd.dst, d0, src, s0),
                32 => kopya_satir::<64>(hd.dst, d0, src, s0),
                64 => kopya_satir::<128>(hd.dst, d0, src, s0),
                _ => kopya_satir::<256>(hd.dst, d0, src, s0),
            }
        }
    }
}

/// `put` kopyasının bir satırı: `B` bayt (32'nin katı), satır başına tek sınır
/// denetimi, içte sabit ofsetli 32 baytlık parçalar.
#[target_feature(enable = "avx2")]
#[inline]
fn kopya_satir<const B: usize>(dst: &mut [u8], d0: usize, src: &[u8], s0: usize) {
    let d: &mut [u8; B] = chunk_mut(dst, d0);
    let s: &[u8; B] = chunk(src, s0);
    let mut i = 0;
    while i < B {
        store_u8x32(chunk_mut(d, i), load_u8x32(chunk(s, i)));
        i += 32;
    }
}

/// Yalnız yatay: dolgu satırı `y + 3`.
#[target_feature(enable = "avx2")]
fn yol_h<const PREP: bool, const P: usize, const S: usize>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    k: [i32; 4],
    yuv: Yuv,
    c: Cikti,
) {
    let sh = _mm_cvtsi32_si128(yuv.sh);
    if w >= 16 {
        let k = kat256(k);
        let r = _mm256_set1_epi32(yuv.r);
        let cv = (_mm256_set1_epi32(c.bias), _mm256_set1_epi16(c.max as i16));
        for y in 0..h {
            let o = (y + 3) * ss;
            let mut x = 0;
            while x < w {
                let (lo, hi) = yatay16::<P, S>(src, o + 2 * x, &k);
                yaz16::<PREP>(hd, w, y, x, yuv256(lo, r, sh), yuv256(hi, r, sh), cv);
                x += 16;
            }
        }
    } else {
        let k = kat128(k);
        let r = _mm_set1_epi32(yuv.r);
        let cv = (_mm_set1_epi32(c.bias), _mm_set1_epi16(c.max as i16));
        for y in 0..h {
            let (lo, hi) = yatay8::<P, S>(src, (y + 3) * ss, &k);
            yaz8::<PREP>(hd, w, y, yuv128(lo, r, sh), yuv128(hi, r, sh), cv);
        }
    }
}

/// Yalnız dikey: dolgu sütunu `x + 3`, satırlar `y + S + 2i`, `+1`.
#[target_feature(enable = "avx2")]
fn yol_v<const PREP: bool, const P: usize, const S: usize>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    k: [i32; 4],
    yuv: Yuv,
    c: Cikti,
) {
    let sh = _mm_cvtsi32_si128(yuv.sh);
    if w >= 16 {
        let k = kat256(k);
        let r = _mm256_set1_epi32(yuv.r);
        let cv = (_mm256_set1_epi32(c.bias), _mm256_set1_epi16(c.max as i16));
        for y in 0..h {
            let mut x = 0;
            while x < w {
                let o = (y + S) * ss + 2 * (x + 3);
                let mut lo = r;
                let mut hi = r;
                for (i, ki) in k.iter().enumerate().take(P) {
                    let a = load_u8x32(chunk(src, o + 2 * i * ss));
                    let b = load_u8x32(chunk(src, o + (2 * i + 1) * ss));
                    lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), *ki));
                    hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), *ki));
                }
                let (lo, hi) = (_mm256_sra_epi32(lo, sh), _mm256_sra_epi32(hi, sh));
                yaz16::<PREP>(hd, w, y, x, lo, hi, cv);
                x += 16;
            }
        }
    } else {
        let k = kat128(k);
        let r = _mm_set1_epi32(yuv.r);
        let cv = (_mm_set1_epi32(c.bias), _mm_set1_epi16(c.max as i16));
        for y in 0..h {
            let o = (y + S) * ss + 6;
            let mut lo = r;
            let mut hi = r;
            for (i, ki) in k.iter().enumerate().take(P) {
                let a = load_u8x16(chunk(src, o + 2 * i * ss));
                let b = load_u8x16(chunk(src, o + (2 * i + 1) * ss));
                lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), *ki));
                hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), *ki));
            }
            let (lo, hi) = (_mm_sra_epi32(lo, sh), _mm_sra_epi32(hi, sh));
            yaz8::<PREP>(hd, w, y, lo, hi, cv);
        }
    }
}

/// İki boyutlu: genişlik sınıfı seçimi (halka boyu).
#[target_feature(enable = "avx2")]
fn yol_hv<const PREP: bool, const P: usize, const S: usize>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    kh: [i32; 4],
    kv: [i32; 4],
    ara: Yuv,
    son: Yuv,
    c: Cikti,
) {
    match w {
        ..=8 => hv_dar::<PREP, P, S>(hd, src, ss, w, h, kh, kv, ara, son, c),
        16 => hv_genis::<PREP, P, S, 16>(hd, src, ss, w, h, kh, kv, ara, son, c),
        32 => hv_genis::<PREP, P, S, 32>(hd, src, ss, w, h, kh, kv, ara, son, c),
        64 => hv_genis::<PREP, P, S, 64>(hd, src, ss, w, h, kh, kv, ara, son, c),
        _ => hv_genis::<PREP, P, S, 128>(hd, src, ss, w, h, kh, kv, ara, son, c),
    }
}

/// İki boyutlu, `w ≥ 16`: 8 satırlık ara halka (`G` sütun), dolgu satırı `j`
/// halka yuvası `j & 7`; çıktı satırı `y` ara satırları `y + S + 2i`, `+1`.
#[target_feature(enable = "avx2")]
fn hv_genis<const PREP: bool, const P: usize, const S: usize, const G: usize>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    kh: [i32; 4],
    kv: [i32; 4],
    ara: Yuv,
    son: Yuv,
    c: Cikti,
) {
    let (kh, kv) = (kat256(kh), kat256(kv));
    let (ra, sa) = (_mm256_set1_epi32(ara.r), _mm_cvtsi32_si128(ara.sh));
    let (rs, ssh) = (_mm256_set1_epi32(son.r), _mm_cvtsi32_si128(son.sh));
    let cv = (_mm256_set1_epi32(c.bias), _mm256_set1_epi16(c.max as i16));
    let mut halka = [[0i16; G]; 8];
    // Üretilen sonraki dolgu satırı.
    let mut j = S;
    for y in 0..h {
        while j < y + S + 2 * P {
            let o = j * ss;
            let satir = &mut halka[j & 7];
            let mut x = 0;
            while x < w {
                let (lo, hi) = yatay16::<P, S>(src, o + 2 * x, &kh);
                let v = _mm256_packs_epi32(yuv256(lo, ra, sa), yuv256(hi, ra, sa));
                store_i16x16(chunk_mut(satir, x), v);
                x += 16;
            }
            j += 1;
        }
        let mut x = 0;
        while x < w {
            let mut lo = rs;
            let mut hi = rs;
            for (i, ki) in kv.iter().enumerate().take(P) {
                let a = super::mem::load_i16x16(chunk(&halka[(y + S + 2 * i) & 7], x));
                let b = super::mem::load_i16x16(chunk(&halka[(y + S + 2 * i + 1) & 7], x));
                lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), *ki));
                hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), *ki));
            }
            let (lo, hi) = (_mm256_sra_epi32(lo, ssh), _mm256_sra_epi32(hi, ssh));
            yaz16::<PREP>(hd, w, y, x, lo, hi, cv);
            x += 16;
        }
    }
}

/// İki boyutlu, `w ≤ 8`: 8 çıktılık xmm satırları (ara halka 8 × 8).
#[target_feature(enable = "avx2")]
fn hv_dar<const PREP: bool, const P: usize, const S: usize>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    kh: [i32; 4],
    kv: [i32; 4],
    ara: Yuv,
    son: Yuv,
    c: Cikti,
) {
    let (kh, kv) = (kat128(kh), kat128(kv));
    let (ra, sa) = (_mm_set1_epi32(ara.r), _mm_cvtsi32_si128(ara.sh));
    let (rs, ssh) = (_mm_set1_epi32(son.r), _mm_cvtsi32_si128(son.sh));
    let cv = (_mm_set1_epi32(c.bias), _mm_set1_epi16(c.max as i16));
    let mut halka = [_mm_setzero_si128(); 8];
    let mut j = S;
    for y in 0..h {
        while j < y + S + 2 * P {
            let (lo, hi) = yatay8::<P, S>(src, j * ss, &kh);
            halka[j & 7] = _mm_packs_epi32(yuv128(lo, ra, sa), yuv128(hi, ra, sa));
            j += 1;
        }
        let mut lo = rs;
        let mut hi = rs;
        for (i, ki) in kv.iter().enumerate().take(P) {
            let a = halka[(y + S + 2 * i) & 7];
            let b = halka[(y + S + 2 * i + 1) & 7];
            lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), *ki));
            hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), *ki));
        }
        let (lo, hi) = (_mm_sra_epi32(lo, ssh), _mm_sra_epi32(hi, ssh));
        yaz8::<PREP>(hd, w, y, lo, hi, cv);
    }
}

// ---------------------------------------------------------------------------
// Yuvalar
// ---------------------------------------------------------------------------

/// 16 bpc `mc[f]` AVX2 yuvası (yalnız belirteçle); `f` dav1d `Filter2d` sırası.
#[must_use]
pub fn put16_fn(_avx2: Avx2Token, f: crate::dsp::mc::Filter2d) -> McPutFn {
    use crate::dsp::mc::Filter2d as F;
    match f {
        F::Regular => put16_giris::<0, 0>,
        F::RegularSmooth => put16_giris::<0, 1>,
        F::RegularSharp => put16_giris::<0, 2>,
        F::SharpRegular => put16_giris::<2, 0>,
        F::SharpSmooth => put16_giris::<2, 1>,
        F::Sharp => put16_giris::<2, 2>,
        F::SmoothRegular => put16_giris::<1, 0>,
        F::Smooth => put16_giris::<1, 1>,
        F::SmoothSharp => put16_giris::<1, 2>,
        F::Bilinear => put16_giris::<BILIN, BILIN>,
    }
}

/// 16 bpc `mct[f]` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn prep16_fn(_avx2: Avx2Token, f: crate::dsp::mc::Filter2d) -> McPrepFn {
    use crate::dsp::mc::Filter2d as F;
    match f {
        F::Regular => prep16_giris::<0, 0>,
        F::RegularSmooth => prep16_giris::<0, 1>,
        F::RegularSharp => prep16_giris::<0, 2>,
        F::SharpRegular => prep16_giris::<2, 0>,
        F::SharpSmooth => prep16_giris::<2, 1>,
        F::Sharp => prep16_giris::<2, 2>,
        F::SmoothRegular => prep16_giris::<1, 0>,
        F::Smooth => prep16_giris::<1, 1>,
        F::SmoothSharp => prep16_giris::<1, 2>,
        F::Bilinear => prep16_giris::<BILIN, BILIN>,
    }
}

// ---------------------------------------------------------------------------
// P36: AVX512ICL girişleri (adres yalnız [`super::mc16_avx512`] belirteçli işaretçileriyle)
// ---------------------------------------------------------------------------
//
// Girişler bu modüldedir: dar bloklar bu modülün AVX2 gövdelerini, geniş bloklar
// `mc16_avx512`'in gövdelerini çağırır. Paylaşılan AVX2 gövdelerinin bütün çağırıcıları
// aynı kod üretim biriminde kalır ve iç bağlantıları korunur (başka birimden
// çağrılan tek kopya dış bağlantıya geçip LLVM'nin yerel işlev iyileştirmelerini
// yitiriyordu: AVX2 hücrelerinde +%10–20, P36 hız bütçesi tanı koşusu).

/// `put` girişi: AVX2 girişiyle aynı sözleşme denetimi (aksi hâlde skaler devir);
/// `w < 32` ve kopya (`mx = my = 0`) AVX2 gövdesi, aksi hâlde AVX512ICL gövdesi.
/// Tek giriş, tek `#[target_feature]` çağrısı (P36 hücre ölçümü: AVX2 girişine
/// devretmek çağrı başına sabit maliyet ekliyordu; bellek bağlı 16 bpc kopyada
/// AVX-512 yolu AVX2'den yavaştı).
pub(super) fn put16_giris_icl<const FH: usize, const FV: usize>(
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
    let ok = dst_stride >= 2 * w
        && h >= 1
        && simd16_sozlesmesi(
            src.len(),
            src_stride,
            w,
            h,
            mx,
            my,
            bit_depth,
            bitdepth_max,
            dst.len(),
            (h - 1) * dst_stride + 2 * w,
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
    // (`mc16_avx512::put16_fn`); belirteç AVX512ICL (AVX2'yi kapsar) tespiti + maske izninin
    // kanıtıdır. (2) sınır: `simd16_sozlesmesi` kaynak bölgesini + `MC_KUYRUK`'u ve
    // hedefte `h` satır × `2w` baytı kapsar (iki gövdenin de sözleşmesi); gövde
    // erişimleri dilim/dizi denetimlidir. (3) hizalama/takma ad: hizasız erişim;
    // `dst` tek `&mut`, `src` `&`.
    unsafe {
        // Her dal `Hedef`'ini kendisi kurar: dar dalda yapı yazmaçlarda kalır
        // (AVX2 girişiyle aynı kod; ortak kurulum dar dalda yığına yazılıyordu).
        if w < 32 || (mx == 0 && my == 0) {
            let mut hd = Hedef {
                dst,
                ds: dst_stride,
                tmp: &mut [],
            };
            govde::<FH, FV, false>(&mut hd, src, src_stride, w, h, mx, my, bitdepth_max);
        } else {
            let mut hd = Hedef {
                dst,
                ds: dst_stride,
                tmp: &mut [],
            };
            super::mc16_avx512::govde::<FH, FV, false>(
                &mut hd,
                src,
                src_stride,
                w,
                h,
                mx,
                my,
                bitdepth_max,
            );
        }
    }
}

/// `prep` girişi: [`put16_giris`] gibi (`w < 32` AVX2 gövdesi).
pub(super) fn prep16_giris_icl<const FH: usize, const FV: usize>(
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
    let ok = simd16_sozlesmesi(
        src.len(),
        src_stride,
        w,
        h,
        mx,
        my,
        bit_depth,
        bitdepth_max,
        tmp.len(),
        w * h,
    );
    if !ok {
        skaler_devri_say();
        prep_skaler::<FH, FV>(tmp, src, src_stride, w, h, mx, my, bit_depth, bitdepth_max);
        return;
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`mc16_avx512::prep16_fn`); belirteç AVX512ICL (AVX2'yi kapsar) tespitinin kanıtıdır.
    // (2) sınır: `simd16_sozlesmesi` kaynak bölgesini + `MC_KUYRUK`'u ve `tmp`'de
    // `w·h` öğeyi kapsar; erişimler denetimlidir. (3) hizalama/takma ad: hizasız
    // erişim; `tmp` tek `&mut`, `src` `&`.
    unsafe {
        // Her dal `Hedef`'ini kendisi kurar: dar dalda yapı yazmaçlarda kalır
        // (AVX2 girişiyle aynı kod; ortak kurulum dar dalda yığına yazılıyordu).
        if w < 32 {
            let mut hd = Hedef {
                dst: &mut [],
                ds: 0,
                tmp,
            };
            govde::<FH, FV, true>(&mut hd, src, src_stride, w, h, mx, my, bitdepth_max);
        } else {
            let mut hd = Hedef {
                dst: &mut [],
                ds: 0,
                tmp,
            };
            super::mc16_avx512::govde::<FH, FV, true>(
                &mut hd,
                src,
                src_stride,
                w,
                h,
                mx,
                my,
                bitdepth_max,
            );
        }
    }
}
