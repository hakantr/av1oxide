//! MC SSSE3 çekirdekleri (8 bpc) — F4-MC d (P35): `put` (`mc[0..9]`) ve `prep`
//! (`mct[0..9]`); 9 adet 8-tap filtre birleşimi + bilinear.
//!
//! ## Kopyasız sözleşme (P32)
//!
//! AVX2 gövdesiyle ([`super::mc`]) aynı sözleşme: çağrı başına yerel dolgu
//! kopyası yok; girişte O(1) denetim ([`crate::dsp::mc::simd_sozlesmesi`]);
//! sözleşme dışı girdi skalere devreder ve sayılır. Okuma bölgesinin ötesi
//! yalnız yatay geçişte satır başına en çok 1 bayttır ([`MC_KUYRUK`] içinde).
//!
//! ## Yöntem
//!
//! AVX2 gövdesinin yapısı xmm'de (dav1d `mc_sse.asm` yaklaşımı; asm
//! kopyalanmadı): `pshufb` + `pmaddubsw` yatay geçiş, `punpck*bw` + `pmaddubsw`
//! dikey geçiş (çift ve tek çıkış satırının ikilileri ayrı kayan pencerelerde),
//! iki boyutlu yolda `i16` ara satır + `pmaddwd`. AVX2'de iki satırı taşıyan ymm
//! şeritleri burada iki xmm'dir; `w ≥ 16` 8 piksellik parçalarla işlenir.
//! SSE4.1 ve üstü komut kullanılmaz (`pmovzx` yerine `punpcklbw` + sıfır).
//! Yuvarlama ve tap seçimi AVX2 gövdesiyle aynıdır.
//!
//! ## `unsafe` politikası (GUV-02/GUV-04)
//!
//! `unsafe` yalnız belirteçli girişlerdeki `#[target_feature]` çağrısıdır; dosya
//! `audit::UNSAFE_EXEMPTIONS` defterinde kayıtlıdır. Yükleme/saklama
//! [`super::mem`] yardımcılarıyla ya da `cvtsi` dönüşümleriyle yapılır.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mc::{BILIN, prep_skaler, put_skaler};
use super::mem::{chunk, chunk_mut, load_u8x16, store_i16x8, store_u8x8, store_u8x16};
use crate::cpu::Ssse3Token;
use crate::dsp::mc::{Filter2d, MC_KUYRUK, McPrepFn, McPutFn, simd_sozlesmesi, skaler_devri_say};
use crate::dsp::mc_tables::MC_SUBPEL_FILTERS;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_add_epi32, _mm_cvtsi32_si128, _mm_cvtsi64_si128, _mm_cvtsi128_si32,
    _mm_cvtsi128_si64, _mm_madd_epi16, _mm_maddubs_epi16, _mm_mulhrs_epi16, _mm_mullo_epi16,
    _mm_packs_epi32, _mm_packus_epi16, _mm_set_epi64x, _mm_set1_epi16, _mm_set1_epi32,
    _mm_setr_epi8, _mm_setzero_si128, _mm_shuffle_epi8, _mm_slli_epi16, _mm_srai_epi16,
    _mm_srai_epi32, _mm_srli_epi16, _mm_srli_epi64, _mm_unpackhi_epi8, _mm_unpackhi_epi16,
    _mm_unpackhi_epi64, _mm_unpacklo_epi8, _mm_unpacklo_epi16, _mm_unpacklo_epi32,
    _mm_unpacklo_epi64,
};

// ---------------------------------------------------------------------------
// Dış yüzey (belirteçli)
// ---------------------------------------------------------------------------

/// `mc[filtre]` (8 bpc) yuvasının SSSE3 girişi; yalnız [`Ssse3Token`] ile.
#[must_use]
pub const fn put_fn(_ssse3: Ssse3Token, filtre: Filter2d) -> McPutFn {
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

/// `mct[filtre]` (8 bpc) yuvasının SSSE3 girişi; yalnız [`Ssse3Token`] ile.
#[must_use]
pub const fn prep_fn(_ssse3: Ssse3Token, filtre: Filter2d) -> McPrepFn {
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

/// `put` girişi: sözleşme denetimi + SSSE3 gövdesi ya da skaler devir.
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
    let mut hd = Hedef {
        dst,
        ds: dst_stride,
        tmp: &mut [],
    };
    let (ss, hd) = (src_stride, &mut hd);
    match (mx != 0, my != 0) {
        (false, false) => kopya_put(hd, src, ss, w, h),
        // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
        // (`put_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır.
        // (2) sınır: `simd_sozlesmesi` kaynak bölgesini + `MC_KUYRUK`'u ve
        // hedefte `h` satır × `w` baytı kapsar; erişimler denetimlidir.
        // (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
        (true, false) => unsafe { yol_h::<FH, false>(hd, src, ss, w, h, mx) },
        // SAFETY: (1) ISA: adres yalnız `Ssse3Token` ile verilir (`put_fn`);
        // belirteç SSSE3 tespitinin kanıtıdır. (2) sınır: `simd_sozlesmesi`
        // kaynak bölgesini + `MC_KUYRUK`'u ve hedefi kapsar; erişimler
        // denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
        (false, true) => unsafe { yol_v::<FV, false>(hd, src, ss, w, h, my) },
        // SAFETY: (1) ISA: adres yalnız `Ssse3Token` ile verilir (`put_fn`);
        // belirteç SSSE3 tespitinin kanıtıdır. (2) sınır: `simd_sozlesmesi`
        // kaynak bölgesini + `MC_KUYRUK`'u ve hedefi kapsar; erişimler
        // denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
        (true, true) => unsafe { yol_hv::<FH, FV, false>(hd, src, ss, w, h, mx, my) },
    }
}

/// `prep` girişi: sözleşme denetimi + SSSE3 gövdesi ya da skaler devir.
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
        // SAFETY: (1) ISA: adres yalnız `Ssse3Token` ile verilir (`prep_fn`);
        // belirteç SSSE3 tespitinin kanıtıdır. (2) sınır: `simd_sozlesmesi`
        // kaynak bölgesini + `MC_KUYRUK`'u ve `tmp`'yi kapsar; erişimler
        // denetimlidir. (3) hizalama/takma ad: `tmp` tek `&mut`, `src` `&`.
        (false, false) => unsafe { kopya_genis_prep(hd, src, ss, w, h) },
        // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
        // (`prep_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır.
        // (2) sınır: `simd_sozlesmesi` kaynak bölgesini + `MC_KUYRUK`'u ve
        // `tmp`'de `w·h` öğeyi kapsar; erişimler denetimlidir. (3) hizalama/takma
        // ad: `tmp` tek `&mut`, `src` `&`.
        (true, false) => unsafe { yol_h::<FH, true>(hd, src, ss, w, h, mx) },
        // SAFETY: (1) ISA: adres yalnız `Ssse3Token` ile verilir (`prep_fn`);
        // belirteç SSSE3 tespitinin kanıtıdır. (2) sınır: `simd_sozlesmesi`
        // kaynak bölgesini + `MC_KUYRUK`'u ve `tmp`'yi kapsar; erişimler
        // denetimlidir. (3) hizalama/takma ad: `tmp` tek `&mut`, `src` `&`.
        (false, true) => unsafe { yol_v::<FV, true>(hd, src, ss, w, h, my) },
        // SAFETY: (1) ISA: adres yalnız `Ssse3Token` ile verilir (`prep_fn`);
        // belirteç SSSE3 tespitinin kanıtıdır. (2) sınır: `simd_sozlesmesi`
        // kaynak bölgesini + `MC_KUYRUK`'u ve `tmp`'yi kapsar; erişimler
        // denetimlidir. (3) hizalama/takma ad: `tmp` tek `&mut`, `src` `&`.
        (true, true) => unsafe { yol_hv::<FH, FV, true>(hd, src, ss, w, h, mx, my) },
    }
}

// ---------------------------------------------------------------------------
// Yol seçimi
// ---------------------------------------------------------------------------

/// Hedef: `put` için bayt düzlemi, `prep` için `tmp` (adım `w`).
struct Hedef<'a> {
    dst: &'a mut [u8],
    ds: usize,
    tmp: &'a mut [i16],
}

/// Filtre satırı: tür `t` (0 regular, 1 smooth, 2 sharp), boyut (`w` ya da
/// `h`), alt piksel `m ≥ 1` — dav1d `GET_H_FILTER`/`GET_V_FILTER`.
fn filtre_satiri(t: usize, boyut: usize, m: u8) -> &'static [i8; 8] {
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

#[target_feature(enable = "ssse3")]
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

#[target_feature(enable = "ssse3")]
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

#[target_feature(enable = "ssse3")]
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

/// `pmaddubsw` ikilileri: `k < N/2` için `(f[S+2k], f[S+2k+1])` baytları.
#[target_feature(enable = "ssse3")]
#[inline]
fn kat_b<const N: usize>(f: &[i8; 8]) -> [__m128i; 4] {
    let s = (8 - N) / 2;
    let mut c = [_mm_setzero_si128(); 4];
    for (k, ck) in c.iter_mut().enumerate().take(N / 2) {
        let lo = f[s + 2 * k] as u8;
        let hi = f[s + 2 * k + 1] as u8;
        *ck = _mm_set1_epi16(i16::from_le_bytes([lo, hi]));
    }
    c
}

/// Bilinear `pmaddubsw` ikilisi `(16 − m, m)`.
#[target_feature(enable = "ssse3")]
#[inline]
fn kat_bilin(m: u8) -> [__m128i; 4] {
    let mut c = [_mm_setzero_si128(); 4];
    c[0] = _mm_set1_epi16(i16::from_le_bytes([16 - m, m]));
    c
}

/// `pmaddwd` ikilileri (iki boyutlu dikey geçiş).
#[target_feature(enable = "ssse3")]
#[inline]
fn kat_w<const N: usize>(f: &[i8; 8]) -> [__m128i; 4] {
    let s = (8 - N) / 2;
    let mut c = [_mm_setzero_si128(); 4];
    for (k, ck) in c.iter_mut().enumerate().take(N / 2) {
        let lo = i32::from(f[s + 2 * k]) & 0xffff;
        let hi = i32::from(f[s + 2 * k + 1]) << 16;
        *ck = _mm_set1_epi32(hi | lo);
    }
    c
}

/// Bilinear iki boyutlu dikey katsayılar: [1] `16 − m`, [2] `m` (`u16` yolu).
#[target_feature(enable = "ssse3")]
#[inline]
fn kat_bilin_w(m: u8) -> [__m128i; 4] {
    let mut c = [_mm_setzero_si128(); 4];
    c[1] = _mm_set1_epi16(16 - i16::from(m));
    c[2] = _mm_set1_epi16(i16::from(m));
    c
}

/// Bilinear iki boyutlu dikey adım `u16`'da (AVX2 gövdesiyle aynı aritmetik):
/// `t = (16−my)·a + my·b ≤ 65 280`; `put` `(t+128)>>8`, `prep` `(t+8)>>4`.
#[target_feature(enable = "ssse3")]
#[inline]
fn bilin_dikey<const PREP: bool>(a: __m128i, b: __m128i, cv: &[__m128i; 4]) -> __m128i {
    let t = _mm_add_epi16(_mm_mullo_epi16(a, cv[1]), _mm_mullo_epi16(b, cv[2]));
    if PREP {
        _mm_srli_epi16::<4>(_mm_add_epi16(t, _mm_set1_epi16(8)))
    } else {
        _mm_srli_epi16::<8>(_mm_add_epi16(t, _mm_set1_epi16(128)))
    }
}

/// 16 baytlık maske: çıktı `i` (0..8), ikili `k` → baytlar `ofs+i+2k`,
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

/// İki satır (yarım başına 8 bayt), satır başına `n` çıktı: satır 0 kelime
/// `0..n`, satır 1 kelime `n..2n`; kalanı sıfır (`0x80`).
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

#[target_feature(enable = "ssse3")]
#[inline]
fn v128(m: [i8; 16]) -> __m128i {
    _mm_setr_epi8(
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13],
        m[14], m[15],
    )
}

// ---------------------------------------------------------------------------
// Yükleme / saklama yardımcıları (denetimli)
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
#[inline]
fn yukle8(src: &[u8], o: usize) -> i64 {
    i64::from_le_bytes(*chunk::<u8, 8>(src, o))
}

#[target_feature(enable = "ssse3")]
#[inline]
fn yukle4(src: &[u8], o: usize) -> __m128i {
    _mm_cvtsi32_si128(i32::from_le_bytes(*chunk::<u8, 4>(src, o)))
}

#[target_feature(enable = "ssse3")]
#[inline]
fn yukle2(src: &[u8], o: usize) -> __m128i {
    _mm_cvtsi32_si128(i32::from(u16::from_le_bytes(*chunk::<u8, 2>(src, o))))
}

/// 8 `i16` → `put` 8 bayt ya da `prep` 8 öğe, satır `y` sütun `x`.
#[target_feature(enable = "ssse3")]
#[inline]
fn sakla8<const PREP: bool>(hd: &mut Hedef<'_>, w: usize, y: usize, x: usize, v: __m128i) {
    if PREP {
        store_i16x8(chunk_mut(hd.tmp, y * w + x), v);
    } else {
        let b = _mm_packus_epi16(v, v);
        store_u8x8(chunk_mut(hd.dst, y * hd.ds + x), b);
    }
}

/// 16 `i16` (`a` sütun `x..x+7`, `b` sütun `x+8..x+15`), satır `y`.
#[target_feature(enable = "ssse3")]
#[inline]
fn sakla16<const PREP: bool>(
    hd: &mut Hedef<'_>,
    w: usize,
    y: usize,
    x: usize,
    a: __m128i,
    b: __m128i,
) {
    if PREP {
        store_i16x8(chunk_mut(hd.tmp, y * w + x), a);
        store_i16x8(chunk_mut(hd.tmp, y * w + x + 8), b);
    } else {
        store_u8x16(chunk_mut(hd.dst, y * hd.ds + x), _mm_packus_epi16(a, b));
    }
}

/// İki satır × `n` (`n` = 4 ya da 2): satır `y` kelime `0..n`, satır `y+1`
/// kelime `n..2n`.
#[target_feature(enable = "ssse3")]
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
// Yuvarlama (8 bpc; AVX2 gövdesiyle aynı)
// ---------------------------------------------------------------------------

/// Yalnız yatay: `put` `(Σ+34)>>6`, `prep` `(Σ+2)>>2`; bilinear `put`
/// `(Σ+8)>>4`, `prep` `Σ`.
#[target_feature(enable = "ssse3")]
#[inline]
fn yuv_h<const PREP: bool, const BIL: bool>(v: __m128i) -> __m128i {
    match (PREP, BIL) {
        (false, false) => _mm_srai_epi16::<6>(_mm_add_epi16(v, _mm_set1_epi16(34))),
        (true, false) => _mm_mulhrs_epi16(v, _mm_set1_epi16(8192)),
        (false, true) => _mm_mulhrs_epi16(v, _mm_set1_epi16(2048)),
        (true, true) => v,
    }
}

/// Yalnız dikey: `put` `(Σ+32)>>6`, `prep` `(Σ+2)>>2`; bilinear `put`
/// `(Σ+8)>>4`, `prep` `Σ`.
#[target_feature(enable = "ssse3")]
#[inline]
fn yuv_v<const PREP: bool, const BIL: bool>(v: __m128i) -> __m128i {
    match (PREP, BIL) {
        (false, false) => _mm_mulhrs_epi16(v, _mm_set1_epi16(512)),
        (true, false) => _mm_mulhrs_epi16(v, _mm_set1_epi16(8192)),
        (false, true) => _mm_mulhrs_epi16(v, _mm_set1_epi16(2048)),
        (true, true) => v,
    }
}

/// İki boyutlu ara satır: 8-tap `(Σ+2)>>2`, bilinear `Σ`.
#[target_feature(enable = "ssse3")]
#[inline]
fn yuv_ara<const BIL: bool>(v: __m128i) -> __m128i {
    if BIL {
        v
    } else {
        _mm_mulhrs_epi16(v, _mm_set1_epi16(8192))
    }
}

/// İki boyutlu son adım (`i32`): `put` `(Σ+512)>>10`, `prep` `(Σ+32)>>6`.
#[target_feature(enable = "ssse3")]
#[inline]
fn yuv_son<const PREP: bool>(v: __m128i) -> __m128i {
    if PREP {
        _mm_srai_epi32::<6>(_mm_add_epi32(v, _mm_set1_epi32(32)))
    } else {
        _mm_srai_epi32::<10>(_mm_add_epi32(v, _mm_set1_epi32(512)))
    }
}

// ---------------------------------------------------------------------------
// Yatay toplamlar (yuvarlamadan önce, `i16`)
// ---------------------------------------------------------------------------

/// Tek satır × 8 çıktı: satırın dolgu ofseti + blok sütunu `o`'dan 16 bayt
/// (sütun `x..x+15`; `w`'nin son 8'lisinde bölgenin 1 bayt ötesi, [`MC_KUYRUK`]
/// içinde).
#[target_feature(enable = "ssse3")]
#[inline]
fn h8<const N: usize>(src: &[u8], o: usize, c: &[__m128i; 4]) -> __m128i {
    let s = (8 - N) / 2;
    let v = load_u8x16(chunk(src, o));
    let mut acc = _mm_setzero_si128();
    for (k, ck) in c.iter().enumerate().take(N / 2) {
        let m = v128(maske8(s, k));
        acc = _mm_add_epi16(acc, _mm_maddubs_epi16(_mm_shuffle_epi8(v, m), *ck));
    }
    acc
}

/// İki satır × `n` çıktı (`w = n` ∈ {4, 2}). `w = 4`: satır başına 8 bayt sütun
/// `S..S+7`; `w = 2`: sütun `0..7`.
#[target_feature(enable = "ssse3")]
#[inline]
fn h_dar<const N: usize, const W: usize>(
    src: &[u8],
    o0: usize,
    o1: usize,
    c: &[__m128i; 4],
) -> __m128i {
    let s = (8 - N) / 2;
    let (bas, ofs) = if W == 4 { (s, 0) } else { (0, s) };
    let v = _mm_set_epi64x(yukle8(src, o1 + bas), yukle8(src, o0 + bas));
    let mut acc = _mm_setzero_si128();
    for (k, ck) in c.iter().enumerate().take(N / 2) {
        let m = v128(maske_iki(ofs, k, W));
        acc = _mm_add_epi16(acc, _mm_maddubs_epi16(_mm_shuffle_epi8(v, m), *ck));
    }
    acc
}

/// Kayan pencere: ilk `n` öğe bir sola kayar, `n-1`. öğe `yeni` olur.
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
/// gerektirmez.
#[inline]
fn kopya_put(hd: &mut Hedef<'_>, src: &[u8], ss: usize, w: usize, h: usize) {
    fn satirlar<const W: usize>(hd: &mut Hedef<'_>, src: &[u8], ss: usize, w: usize, h: usize) {
        let ds = hd.ds;
        for y in 0..h {
            let so = (y + 3) * ss + 3;
            if W < 16 {
                *chunk_mut::<u8, W>(hd.dst, y * ds) = *chunk(src, so);
            } else {
                // `w ≥ 16`: satır dilimleri bir kez ayrılır, 16 baytlık parçalarda
                // sınır denetimi yok.
                let d = &mut hd.dst[y * ds..y * ds + w];
                for (dp, sp) in d
                    .as_chunks_mut::<16>()
                    .0
                    .iter_mut()
                    .zip(src[so..so + w].as_chunks::<16>().0)
                {
                    *dp = *sp;
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

/// `prep` kopya yolu `w < 16` (dav1d `prep_c`): `tmp = s << 4`; ISA gerektirmez.
#[inline]
fn kopya_prep(hd: &mut Hedef<'_>, src: &[u8], ss: usize, w: usize, h: usize) {
    fn satirlar<const W: usize>(hd: &mut Hedef<'_>, src: &[u8], ss: usize, w: usize, h: usize) {
        for y in 0..h {
            let so = (y + 3) * ss + 3;
            let s: &[u8; W] = chunk(src, so);
            let t: &mut [i16; W] = chunk_mut(hd.tmp, y * w);
            for (o, &p) in t.iter_mut().zip(s) {
                *o = i16::from(p) << 4;
            }
        }
    }
    match w {
        2 => satirlar::<2>(hd, src, ss, w, h),
        4 => satirlar::<4>(hd, src, ss, w, h),
        _ => satirlar::<8>(hd, src, ss, w, h),
    }
}

/// `prep` kopya yolu `w ≥ 16`: `punpck{l,h}bw` sıfırla + `psllw 4` (16 piksel).
#[target_feature(enable = "ssse3")]
fn kopya_genis_prep(hd: &mut Hedef<'_>, src: &[u8], ss: usize, w: usize, h: usize) {
    let z = _mm_setzero_si128();
    let mut so = 3 * ss + 3;
    for y in 0..h {
        // Satır dilimleri bir kez ayrılır; 16 piksellik parçalarda sınır denetimi yok.
        let t = &mut hd.tmp[y * w..(y + 1) * w];
        for (sp, tp) in src[so..so + w]
            .as_chunks::<16>()
            .0
            .iter()
            .zip(t.as_chunks_mut::<16>().0)
        {
            let v = load_u8x16(sp);
            store_i16x8(
                chunk_mut(tp, 0),
                _mm_slli_epi16::<4>(_mm_unpacklo_epi8(v, z)),
            );
            store_i16x8(
                chunk_mut(tp, 8),
                _mm_slli_epi16::<4>(_mm_unpackhi_epi8(v, z)),
            );
        }
        so += ss;
    }
}

// ---------------------------------------------------------------------------
// Yalnız yatay
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
#[inline]
fn yatay<const N: usize, const PREP: bool, const BIL: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    c: &[__m128i; 4],
) {
    match w {
        2 | 4 => {
            for y in (0..h).step_by(2) {
                let (o0, o1) = ((y + 3) * ss, (y + 4) * ss);
                if w == 4 {
                    let v = yuv_h::<PREP, BIL>(h_dar::<N, 4>(src, o0, o1, c));
                    sakla_dar::<PREP, 4>(hd, y, v);
                } else {
                    let v = yuv_h::<PREP, BIL>(h_dar::<N, 2>(src, o0, o1, c));
                    sakla_dar::<PREP, 2>(hd, y, v);
                }
            }
        }
        8 => {
            for y in 0..h {
                let v = yuv_h::<PREP, BIL>(h8::<N>(src, (y + 3) * ss, c));
                sakla8::<PREP>(hd, w, y, 0, v);
            }
        }
        _ => {
            for y in 0..h {
                let o = (y + 3) * ss;
                for x in (0..w).step_by(16) {
                    let a = yuv_h::<PREP, BIL>(h8::<N>(src, o + x, c));
                    let b = yuv_h::<PREP, BIL>(h8::<N>(src, o + x + 8, c));
                    sakla16::<PREP>(hd, w, y, x, a, b);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Yalnız dikey
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
#[inline]
fn dikey<const N: usize, const PREP: bool, const BIL: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    c: &[__m128i; 4],
) {
    let s = (8 - N) / 2;
    let n = N / 2;
    match w {
        2 | 4 => {
            let yuk = |j: usize| {
                if w == 4 {
                    yukle4(src, j * ss + 3)
                } else {
                    yukle2(src, j * ss + 3)
                }
            };
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
                    acc = _mm_add_epi16(acc, _mm_maddubs_epi16(r[k], c[k]));
                }
                let v = yuv_v::<PREP, BIL>(acc);
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
            // Çift (`e`) ve tek (`o`) çıkış satırının ikilileri ayrı pencerelerde.
            let yuk = |j: usize| _mm_cvtsi64_si128(yukle8(src, j * ss + 3));
            let mut re = [_mm_setzero_si128(); 4];
            let mut ro = [_mm_setzero_si128(); 4];
            let mut satir = [_mm_setzero_si128(); 9];
            for (j, sj) in satir.iter_mut().enumerate().take(N + 1) {
                *sj = yuk(s + j);
            }
            for k in 0..n {
                re[k] = _mm_unpacklo_epi8(satir[2 * k], satir[2 * k + 1]);
                ro[k] = _mm_unpacklo_epi8(satir[2 * k + 1], satir[2 * k + 2]);
            }
            let mut son = satir[N];
            let mut y = 0;
            loop {
                let mut ae = _mm_setzero_si128();
                let mut ao = _mm_setzero_si128();
                for k in 0..n {
                    ae = _mm_add_epi16(ae, _mm_maddubs_epi16(re[k], c[k]));
                    ao = _mm_add_epi16(ao, _mm_maddubs_epi16(ro[k], c[k]));
                }
                sakla8::<PREP>(hd, w, y, 0, yuv_v::<PREP, BIL>(ae));
                sakla8::<PREP>(hd, w, y + 1, 0, yuv_v::<PREP, BIL>(ao));
                y += 2;
                if y >= h {
                    break;
                }
                let a = yuk(y + s + N - 1);
                let b = yuk(y + s + N);
                re = kaydir(re, n, _mm_unpacklo_epi8(son, a));
                ro = kaydir(ro, n, _mm_unpacklo_epi8(a, b));
                son = b;
            }
        }
        _ => {
            // 16 sütunluk şerit: çift/tek çıkış satırının alt (`l`) ve üst (`u`)
            // 8 sütununun ikilileri ayrı kayan pencerelerde.
            for x in (0..w).step_by(16) {
                let yuk = |j: usize| load_u8x16(chunk(src, j * ss + 3 + x));
                let mut le = [_mm_setzero_si128(); 4];
                let mut ue = [_mm_setzero_si128(); 4];
                let mut lo = [_mm_setzero_si128(); 4];
                let mut uo = [_mm_setzero_si128(); 4];
                let mut satir = [_mm_setzero_si128(); 9];
                for (j, sj) in satir.iter_mut().enumerate().take(N + 1) {
                    *sj = yuk(s + j);
                }
                for k in 0..n {
                    let (a, b, c2) = (satir[2 * k], satir[2 * k + 1], satir[2 * k + 2]);
                    le[k] = _mm_unpacklo_epi8(a, b);
                    ue[k] = _mm_unpackhi_epi8(a, b);
                    lo[k] = _mm_unpacklo_epi8(b, c2);
                    uo[k] = _mm_unpackhi_epi8(b, c2);
                }
                let mut son = satir[N];
                let mut y = 0;
                let mut oa = (s + N + 1) * ss + 3 + x;
                loop {
                    let mut a0 = _mm_setzero_si128();
                    let mut a1 = _mm_setzero_si128();
                    let mut b0 = _mm_setzero_si128();
                    let mut b1 = _mm_setzero_si128();
                    for k in 0..n {
                        a0 = _mm_add_epi16(a0, _mm_maddubs_epi16(le[k], c[k]));
                        a1 = _mm_add_epi16(a1, _mm_maddubs_epi16(ue[k], c[k]));
                        b0 = _mm_add_epi16(b0, _mm_maddubs_epi16(lo[k], c[k]));
                        b1 = _mm_add_epi16(b1, _mm_maddubs_epi16(uo[k], c[k]));
                    }
                    sakla16::<PREP>(hd, w, y, x, yuv_v::<PREP, BIL>(a0), yuv_v::<PREP, BIL>(a1));
                    sakla16::<PREP>(
                        hd,
                        w,
                        y + 1,
                        x,
                        yuv_v::<PREP, BIL>(b0),
                        yuv_v::<PREP, BIL>(b1),
                    );
                    y += 2;
                    if y >= h {
                        break;
                    }
                    let a = load_u8x16(chunk(src, oa));
                    let b = load_u8x16(chunk(src, oa + ss));
                    oa += 2 * ss;
                    le = kaydir(le, n, _mm_unpacklo_epi8(son, a));
                    ue = kaydir(ue, n, _mm_unpackhi_epi8(son, a));
                    lo = kaydir(lo, n, _mm_unpacklo_epi8(a, b));
                    uo = kaydir(uo, n, _mm_unpackhi_epi8(a, b));
                    son = b;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// İki boyutlu
// ---------------------------------------------------------------------------

/// İki boyutlu dikey geçiş (`i32`), bir çıkış satırının 8 sütunu: `m` ara
/// satırlar (`m[0]` en üst), `p` düzen (`unpacklo`/`unpackhi`).
#[target_feature(enable = "ssse3")]
#[inline]
fn hv_sutun<const NV: usize, const PREP: bool>(m: &[__m128i], cv: &[__m128i; 4]) -> __m128i {
    let mut lo = _mm_setzero_si128();
    let mut hi = _mm_setzero_si128();
    for k in 0..NV / 2 {
        let (a, b) = (m[2 * k], m[2 * k + 1]);
        lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), cv[k]));
        hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), cv[k]));
    }
    _mm_packs_epi32(yuv_son::<PREP>(lo), yuv_son::<PREP>(hi))
}

/// İki boyutlu yol, `w = W` ∈ {2, 4}: ara satırlar xmm'nin alt `W`
/// kelimesinde kayan pencerede; ilk ara satır tek, sonrakiler bir `h_dar`
/// çağrısında ikişer (satır `j`, `j+1`).
#[target_feature(enable = "ssse3")]
#[inline]
fn iki_dar<const NH: usize, const NV: usize, const PREP: bool, const BIL: bool, const W: usize>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    h: usize,
    ch: &[__m128i; 4],
    cv: &[__m128i; 4],
) {
    let sv = (8 - NV) / 2;
    let n = NV / 2;
    let ust = |v: __m128i| {
        if W == 4 {
            _mm_unpackhi_epi64(v, v)
        } else {
            _mm_srli_epi64::<32>(v)
        }
    };
    let mut m = [_mm_setzero_si128(); 10];
    m[0] = yuv_ara::<BIL>(h_dar::<NH, W>(src, sv * ss, sv * ss, ch));
    for k in 0..n {
        let j = sv + 1 + 2 * k;
        let v = yuv_ara::<BIL>(h_dar::<NH, W>(src, j * ss, (j + 1) * ss, ch));
        m[1 + 2 * k] = v;
        m[2 + 2 * k] = ust(v);
    }
    let mut y = 0;
    loop {
        if BIL {
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
            let v = bilin_dikey::<PREP>(x0, x1, cv);
            if W == 4 {
                sakla_dar::<PREP, 4>(hd, y, v);
            } else {
                sakla_dar::<PREP, 2>(hd, y, v);
            }
        } else {
            // Satır `y`: ikililer (m[2k], m[2k+1]); satır `y+1`: (m[2k+1], m[2k+2]).
            let mut ae = _mm_setzero_si128();
            let mut ao = _mm_setzero_si128();
            for k in 0..n {
                let (a, b, c) = (m[2 * k], m[2 * k + 1], m[2 * k + 2]);
                ae = _mm_add_epi32(ae, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), cv[k]));
                ao = _mm_add_epi32(ao, _mm_madd_epi16(_mm_unpacklo_epi16(b, c), cv[k]));
            }
            let z = _mm_setzero_si128();
            let re = _mm_packs_epi32(yuv_son::<PREP>(ae), z);
            let ro = _mm_packs_epi32(yuv_son::<PREP>(ao), z);
            if W == 4 {
                sakla_dar::<PREP, 4>(hd, y, _mm_unpacklo_epi64(re, ro));
            } else {
                sakla_dar::<PREP, 2>(hd, y, _mm_unpacklo_epi32(re, ro));
            }
        }
        y += 2;
        if y >= h {
            break;
        }
        let j = y + sv + NV - 1;
        let v = yuv_ara::<BIL>(h_dar::<NH, W>(src, j * ss, (j + 1) * ss, ch));
        let b = ust(v);
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
/// (`pmaddwd`). Çıkış satırı `y`, ara satırlar `y+SV .. y+SV+NV−1`.
#[target_feature(enable = "ssse3")]
fn iki<const NH: usize, const NV: usize, const PREP: bool, const BIL: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    ch: &[__m128i; 4],
    cv: &[__m128i; 4],
) {
    let sv = (8 - NV) / 2;
    match w {
        4 => iki_dar::<NH, NV, PREP, BIL, 4>(hd, src, ss, h, ch, cv),
        2 => iki_dar::<NH, NV, PREP, BIL, 2>(hd, src, ss, h, ch, cv),
        _ => {
            // 8 sütunluk şeritler; ara satırlar şerit başına kayan pencerede.
            for x in (0..w).step_by(8) {
                let ara = |j: usize| yuv_ara::<BIL>(h8::<NH>(src, j * ss + x, ch));
                let mut m = [_mm_setzero_si128(); 8];
                for (j, mj) in m.iter_mut().enumerate().take(NV - 1) {
                    *mj = ara(sv + j);
                }
                for y in 0..h {
                    m[NV - 1] = ara(y + sv + NV - 1);
                    let r = if BIL {
                        bilin_dikey::<PREP>(m[0], m[1], cv)
                    } else {
                        hv_sutun::<NV, PREP>(&m, cv)
                    };
                    sakla8::<PREP>(hd, w, y, x, r);
                    m = kaydir(m, NV, m[NV - 1]);
                }
            }
        }
    }
}

// Derleme zamanı denetimi: kuyruk, yatay geçişin 1 baytlık aşımını kapsar.
const _: () = assert!(MC_KUYRUK >= 1);
