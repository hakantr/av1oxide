//! MC karışım AVX2 çekirdekleri (8 bpc) — F4-MC b (P33): `blend`, `blend_v`,
//! `blend_h` (inter-intra / maske ve OBMC).
//!
//! ## Sözleşme
//!
//! Girişte O(1) denetim ([`crate::dsp::mc::karisim_sozlesmesi`]): `w`, `h`
//! 2'nin kuvveti — `blend` `w` 4…32, `h` 4…32 (`w·h ≥ 16`); `blend_v` `w`
//! 2…32, `h` 2…128; `blend_h` `w` 2…128, `h` 2…32 (OBMC maske tablosu);
//! `tmp` (ve `blend`'de `mask`) en az `w·h` öğe, satırlar bitişik;
//! `dst_stride ≥ w`, `dst` yazılan son satırın sonuna kadar. Sözleşme dışı
//! girdi skalere devreder ve sayılır. Okuma/yazma yalnız `w` sütunluk satırlar
//! (`blend_h` yalnız ilk `(h·3) >> 2` satır).
//!
//! ## Yöntem (dav1d `mc_avx2.asm` yaklaşımı; asm kopyalanmadı)
//!
//! `(d·(64−m) + t·m + 32) >> 6`: hedef ve `tmp` baytları `punpck{l,h}bw` ile
//! ikili, ağırlıklar `(64−m, m)` ikilisi, `pmaddubsw` (en çok 255·64, taşmaz),
//! `pmulhrsw(·, 512)` = `(x + 32) >> 6`, `packuswb` sırayı geri kurar.
//! `w ≥ 32` satır başına ymm; `w` 4…16'da bir xmm'e `16/w` satır toplanır (kısmi grupta
//! yalnız yazılan satırlar okunur); `w = 2` doğrudan tamsayı formülüyle.
//! `blend_v`'de maske sütuna (`OBMC_MASKS[w + x]`; son çeyrek 0, hedef aynı
//! kalır), `blend_h`'de satıra (`OBMC_MASKS[h + y]`) bağlıdır.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mem::{chunk, chunk_mut, load_u8x16, store_u8x4, store_u8x8, store_u8x16, store_u8x32};
use crate::cpu::Avx2Token;
use crate::dsp::mc::{BlendDirFn, BlendFn, karisim_sozlesmesi, skaler_devri_say};
use crate::dsp::mc_tables::OBMC_MASKS;
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, __m256i, _mm_maddubs_epi16, _mm_mulhrs_epi16, _mm_packus_epi16, _mm_set_epi64x,
    _mm_set1_epi8, _mm_set1_epi16, _mm_setr_epi32, _mm_srli_si128, _mm_sub_epi8, _mm_unpackhi_epi8,
    _mm_unpacklo_epi8, _mm256_maddubs_epi16, _mm256_mulhrs_epi16, _mm256_packus_epi16,
    _mm256_set_m128i, _mm256_set1_epi8, _mm256_set1_epi16, _mm256_sub_epi8, _mm256_unpackhi_epi8,
    _mm256_unpacklo_epi8,
};

/// 32 baytlık iki yarıdan (`lo`, `hi`) yazmaç.
#[target_feature(enable = "avx2")]
#[inline]
fn yazmac(b: &[u8; 32]) -> __m256i {
    _mm256_set_m128i(load_u8x16(chunk(b, 16)), load_u8x16(chunk(b, 0)))
}

/// `s[o..o+32]` (32 bayt, iki 16'lık yükleme).
#[target_feature(enable = "avx2")]
#[inline]
fn yukle32(s: &[u8], o: usize) -> __m256i {
    _mm256_set_m128i(load_u8x16(chunk(s, o + 16)), load_u8x16(chunk(s, o)))
}

/// `(d·(64−m) + t·m + 32) >> 6`, 32 piksel; `m` baytları `mb`.
#[target_feature(enable = "avx2")]
#[inline]
fn harmanla(d: __m256i, t: __m256i, mb: __m256i) -> __m256i {
    let ters = _mm256_sub_epi8(_mm256_set1_epi8(64), mb);
    let wlo = _mm256_unpacklo_epi8(ters, mb);
    let whi = _mm256_unpackhi_epi8(ters, mb);
    let k = _mm256_set1_epi16(512);
    let lo = _mm256_mulhrs_epi16(_mm256_maddubs_epi16(_mm256_unpacklo_epi8(d, t), wlo), k);
    let hi = _mm256_mulhrs_epi16(_mm256_maddubs_epi16(_mm256_unpackhi_epi8(d, t), whi), k);
    _mm256_packus_epi16(lo, hi)
}

/// 128 bit karışım (16 piksel): `(d·(64−m) + t·m + 32) >> 6`.
#[target_feature(enable = "avx2")]
#[inline]
fn harmanla128(d: __m128i, t: __m128i, mb: __m128i) -> __m128i {
    let ters = _mm_sub_epi8(_mm_set1_epi8(64), mb);
    let k = _mm_set1_epi16(512);
    let lo = _mm_maddubs_epi16(_mm_unpacklo_epi8(d, t), _mm_unpacklo_epi8(ters, mb));
    let hi = _mm_maddubs_epi16(_mm_unpackhi_epi8(d, t), _mm_unpackhi_epi8(ters, mb));
    _mm_packus_epi16(_mm_mulhrs_epi16(lo, k), _mm_mulhrs_epi16(hi, k))
}

/// `W` (4, 8, 16) sütunlu `satir ≤ 16/W` satırı (`y`'den) bir xmm'e toplar;
/// eksik satırlar 0 (okunmaz).
#[target_feature(enable = "avx2")]
#[inline]
fn topla16<const W: usize>(d: &[u8], ds: usize, y: usize, satir: usize) -> __m128i {
    let r32 = |r: usize| {
        if r < satir {
            i32::from_le_bytes(*chunk(d, (y + r) * ds))
        } else {
            0
        }
    };
    let r64 = |r: usize| {
        if r < satir {
            i64::from_le_bytes(*chunk(d, (y + r) * ds))
        } else {
            0
        }
    };
    match W {
        4 => _mm_setr_epi32(r32(0), r32(1), r32(2), r32(3)),
        8 => _mm_set_epi64x(r64(1), r64(0)),
        _ => load_u8x16(chunk(d, y * ds)),
    }
}

/// [`topla16`]'nın tersi: yalnız ilk `satir` satırı yazar.
#[target_feature(enable = "avx2")]
#[inline]
fn dagit16<const W: usize>(d: &mut [u8], ds: usize, y: usize, satir: usize, v: __m128i) {
    match W {
        4 => {
            store_u8x4(chunk_mut(d, y * ds), v);
            if satir > 1 {
                store_u8x4(chunk_mut(d, (y + 1) * ds), _mm_srli_si128::<4>(v));
            }
            if satir > 2 {
                store_u8x4(chunk_mut(d, (y + 2) * ds), _mm_srli_si128::<8>(v));
            }
            if satir > 3 {
                store_u8x4(chunk_mut(d, (y + 3) * ds), _mm_srli_si128::<12>(v));
            }
        }
        8 => {
            store_u8x8(chunk_mut(d, y * ds), v);
            if satir > 1 {
                store_u8x8(chunk_mut(d, (y + 1) * ds), _mm_srli_si128::<8>(v));
            }
        }
        _ => store_u8x16(chunk_mut(d, y * ds), v),
    }
}

/// `tmp`'nin `o`'dan `n ≤ 16` baytı (eksik baytlar 0; dilim dışı okunmaz).
#[target_feature(enable = "avx2")]
#[inline]
fn bitisik16(s: &[u8], o: usize, n: usize) -> __m128i {
    if n == 16 {
        load_u8x16(chunk(s, o))
    } else {
        let mut b = [0u8; 16];
        b[..n].copy_from_slice(&s[o..o + n]);
        load_u8x16(&b)
    }
}

/// `blend` gövdesi, `W` sütun: `W ≥ 32` satır başına ymm, `W ≤ 16` xmm başına
/// `16/W` satır (`h` 2'nin kuvveti ≥ 4 olduğundan tam gruplar).
#[target_feature(enable = "avx2")]
fn blend_govde<const W: usize>(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize, mask: &[u8]) {
    if W >= 32 {
        for y in 0..h {
            let o = y * W;
            let d = yukle32(dst, y * ds);
            let r = harmanla(d, yukle32(tmp, o), yukle32(mask, o));
            store_u8x32(chunk_mut(dst, y * ds), r);
        }
        return;
    }
    let rpx = 16 / W;
    for y in (0..h).step_by(rpx) {
        let o = y * W;
        let r = harmanla128(
            topla16::<W>(dst, ds, y, rpx),
            load_u8x16(chunk(tmp, o)),
            load_u8x16(chunk(mask, o)),
        );
        dagit16::<W>(dst, ds, y, rpx, r);
    }
}

/// `blend_v` gövdesi: sütun maskesi `OBMC_MASKS[W + x]`, satırlara yayılı.
#[target_feature(enable = "avx2")]
fn blend_v_govde<const W: usize>(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize) {
    if W == 2 {
        // İki piksellik satırda yalnız sütun 0 karışır (`(2·3) >> 2 = 1`;
        // sütun 1'in maskesi 0): doğrudan tamsayı formülü.
        let m = i32::from(OBMC_MASKS[2]);
        for y in 0..h {
            let o = y * ds;
            dst[o] = px(dst[o], tmp[y * 2], m);
        }
        return;
    }
    let tablo = &BLEND_V_MASKE[W.trailing_zeros() as usize - 2];
    if W >= 32 {
        let m = yazmac(tablo);
        for y in 0..h {
            let d = yukle32(dst, y * ds);
            let r = harmanla(d, yukle32(tmp, y * W), m);
            store_u8x32(chunk_mut(dst, y * ds), r);
        }
        return;
    }
    let m = load_u8x16(chunk(tablo, 0));
    let rpx = 16 / W;
    let mut y = 0;
    while y < h {
        // `W = 4`, `h = 2`'de tek kısmi grup.
        let satir = rpx.min(h - y);
        let t = bitisik16(tmp, y * W, satir * W);
        let r = harmanla128(topla16::<W>(dst, ds, y, satir), t, m);
        dagit16::<W>(dst, ds, y, satir, r);
        y += satir;
    }
}

/// `blend_h` gövdesi: satır maskesi `OBMC_MASKS[h + y]`, ilk `(h·3) >> 2` satır.
#[target_feature(enable = "avx2")]
fn blend_h_govde<const W: usize>(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize) {
    let satirlar = (h * 3) >> 2;
    let tablo = &OBMC_MASKS[h..];
    if W >= 32 {
        for (y, &my) in tablo.iter().enumerate().take(satirlar) {
            let m = _mm256_set1_epi8(my as i8);
            for x in (0..W).step_by(32) {
                let d = yukle32(dst, y * ds + x);
                let r = harmanla(d, yukle32(tmp, y * W + x), m);
                store_u8x32(chunk_mut(dst, y * ds + x), r);
            }
        }
        return;
    }
    if W == 2 {
        for (y, &m) in tablo.iter().enumerate().take(satirlar) {
            let (o, t) = (y * ds, y * 2);
            let m = i32::from(m);
            dst[o] = px(dst[o], tmp[t], m);
            dst[o + 1] = px(dst[o + 1], tmp[t + 1], m);
        }
        return;
    }
    let rpx = 16 / W;
    let mut y = 0;
    while y < satirlar {
        let satir = rpx.min(satirlar - y);
        // Satır maskesi yayını: satır `r` için `W` bayt `tablo[y + r]` (eksik satır 0).
        let m = |r: usize| if r < satir { tablo[y + r] } else { 0 };
        let mb = match W {
            4 => {
                let k = |r: usize| (u32::from(m(r)) * 0x0101_0101) as i32;
                _mm_setr_epi32(k(0), k(1), k(2), k(3))
            }
            8 => {
                let k = |r: usize| (u64::from(m(r)) * 0x0101_0101_0101_0101) as i64;
                _mm_set_epi64x(k(1), k(0))
            }
            _ => _mm_set1_epi8(m(0) as i8),
        };
        let t = bitisik16(tmp, y * W, satir * W);
        let r = harmanla128(topla16::<W>(dst, ds, y, satir), t, mb);
        dagit16::<W>(dst, ds, y, satir, r);
        y += satir;
    }
}

/// Tek piksel karışımı `(d·(64−m) + t·m + 32) >> 6` (dar satırlar).
#[inline(always)]
fn px(d: u8, t: u8, m: i32) -> u8 {
    ((i32::from(d) * (64 - m) + i32::from(t) * m + 32) >> 6) as u8
}

/// `blend_v` sütun maskesi `OBMC_MASKS[W + x]`, 32 bayta yayılı (`W` = 4, 8, 16, 32).
static BLEND_V_MASKE: [[u8; 32]; 4] = [
    blend_v_maske(4),
    blend_v_maske(8),
    blend_v_maske(16),
    blend_v_maske(32),
];

const fn blend_v_maske(w: usize) -> [u8; 32] {
    let mut b = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        b[i] = OBMC_MASKS[w + i % w];
        i += 1;
    }
    b
}

macro_rules! skaler {
    ($bd:expr, $f:ident ( $($a:expr),* )) => {
        match $bd {
            BitDepth::Eight => scalar::$f::<8>($($a),*),
            BitDepth::Ten => scalar::$f::<10>($($a),*),
            BitDepth::Twelve => scalar::$f::<12>($($a),*),
        }
    };
}

fn blend_giris(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    mask: &[u8],
    bd: BitDepth,
    bdmax: u16,
) {
    if bd != BitDepth::Eight
        || mask.len() < w * h
        || w * h < 16
        || !karisim_sozlesmesi(w, (4, 32), h, (4, 32), h, tmp.len(), dst.len(), ds)
    {
        skaler_devri_say();
        return skaler!(bd, blend(dst, ds, tmp, w, h, mask, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`blend_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `karisim_sozlesmesi` `tmp`'nin `w·h` öğesini ve hedefte `h` satır × `w`
    // baytı, `mask.len() ≥ w·h` maskeyi kapsar; erişimler denetimlidir. (3)
    // hizalama/takma ad: `dst` tek `&mut`, girdiler `&`; ödünç kuralları
    // çakışmayı dışlar.
    unsafe {
        match w {
            4 => blend_govde::<4>(dst, ds, tmp, h, mask),
            8 => blend_govde::<8>(dst, ds, tmp, h, mask),
            16 => blend_govde::<16>(dst, ds, tmp, h, mask),
            _ => blend_govde::<32>(dst, ds, tmp, h, mask),
        }
    }
}

fn blend_v_giris(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    if bd != BitDepth::Eight
        || !karisim_sozlesmesi(w, (2, 32), h, (2, 128), h, tmp.len(), dst.len(), ds)
    {
        skaler_devri_say();
        return skaler!(bd, blend_v(dst, ds, tmp, w, h, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`blend_v_fn`); belirteç AVX2 tespitinin kanıtıdır. (2) sınır:
    // `karisim_sozlesmesi` `tmp`'nin `w·h` öğesini ve hedefte `h` satır × `w`
    // baytı kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek
    // `&mut`, `tmp` `&`.
    unsafe {
        match w {
            2 => blend_v_govde::<2>(dst, ds, tmp, h),
            4 => blend_v_govde::<4>(dst, ds, tmp, h),
            8 => blend_v_govde::<8>(dst, ds, tmp, h),
            16 => blend_v_govde::<16>(dst, ds, tmp, h),
            _ => blend_v_govde::<32>(dst, ds, tmp, h),
        }
    }
}

fn blend_h_giris(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    if bd != BitDepth::Eight
        || !karisim_sozlesmesi(
            w,
            (2, 128),
            h,
            (2, 32),
            (h * 3) >> 2,
            tmp.len(),
            dst.len(),
            ds,
        )
    {
        skaler_devri_say();
        return skaler!(bd, blend_h(dst, ds, tmp, w, h, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`blend_h_fn`); belirteç AVX2 tespitinin kanıtıdır. (2) sınır:
    // `karisim_sozlesmesi` `tmp`'nin `w·h` öğesini ve hedefte yazılan
    // `(h·3) >> 2` satır × `w` baytı kapsar; erişimler denetimlidir. (3)
    // hizalama/takma ad: `dst` tek `&mut`, `tmp` `&`.
    unsafe {
        match w {
            2 => blend_h_govde::<2>(dst, ds, tmp, h),
            4 => blend_h_govde::<4>(dst, ds, tmp, h),
            8 => blend_h_govde::<8>(dst, ds, tmp, h),
            16 => blend_h_govde::<16>(dst, ds, tmp, h),
            32 => blend_h_govde::<32>(dst, ds, tmp, h),
            64 => blend_h_govde::<64>(dst, ds, tmp, h),
            _ => blend_h_govde::<128>(dst, ds, tmp, h),
        }
    }
}

/// `blend` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn blend_fn(_avx2: Avx2Token) -> BlendFn {
    blend_giris
}

/// `blend_v` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn blend_v_fn(_avx2: Avx2Token) -> BlendDirFn {
    blend_v_giris
}

/// `blend_h` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn blend_h_fn(_avx2: Avx2Token) -> BlendDirFn {
    blend_h_giris
}

// ---------------------------------------------------------------------------
// P36: AVX512ICL girişleri (adres yalnız [`super::mc_karisim_avx512`] belirteçli işaretçileriyle)
// ---------------------------------------------------------------------------
//
// Girişler bu modüldedir: dar bloklar bu modülün AVX2 gövdelerini, geniş bloklar
// `mc_karisim_avx512`'in gövdelerini çağırır. Paylaşılan AVX2 gövdelerinin bütün çağırıcıları
// aynı kod üretim biriminde kalır ve iç bağlantıları korunur (başka birimden
// çağrılan tek kopya dış bağlantıya geçip LLVM'nin yerel işlev iyileştirmelerini
// yitiriyordu: AVX2 hücrelerinde +%10–20, P36 hız bütçesi tanı koşusu).

/// `blend` girişi: AVX2 girişiyle aynı sözleşme denetimi (aksi hâlde skaler
/// devir); `w < 32` AVX2 gövdeleri, `w = 32` AVX512ICL gövdesi. Tek giriş, tek
/// `#[target_feature]` çağrısı (P36 hücre ölçümü: AVX2 girişine devretmek çağrı
/// başına sabit maliyet ekliyordu; `w = 16`'da dört satırı zmm'e toplamak AVX2
/// yolundan yavaştı).
pub(super) fn blend_giris_icl(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    mask: &[u8],
    bd: BitDepth,
    bdmax: u16,
) {
    if bd != BitDepth::Eight
        || mask.len() < w * h
        || w * h < 16
        || !karisim_sozlesmesi(w, (4, 32), h, (4, 32), h, tmp.len(), dst.len(), ds)
    {
        skaler_devri_say();
        return skaler!(bd, blend(dst, ds, tmp, w, h, mask, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`mc_karisim_avx512::blend_fn`); belirteç AVX512ICL (AVX2'yi kapsar) tespiti + maske izninin
    // kanıtıdır. (2) sınır: `karisim_sozlesmesi` `tmp`'nin `w·h` öğesini ve
    // hedefte `h` satır × `w` baytı, `mask.len() ≥ w·h` maskeyi kapsar; `h` 2'nin
    // kuvveti ≥ 4 olduğundan satır çiftleri tamdır; erişimler denetimlidir. (3)
    // hizalama/takma ad: `dst` tek `&mut`, girdiler `&`.
    unsafe {
        match w {
            4 => blend_govde::<4>(dst, ds, tmp, h, mask),
            8 => blend_govde::<8>(dst, ds, tmp, h, mask),
            16 => blend_govde::<16>(dst, ds, tmp, h, mask),
            _ => super::mc_karisim_avx512::blend_govde(dst, ds, tmp, h, mask),
        }
    }
}

/// `blend_v` girişi: [`blend_giris`] gibi (`w < 32` AVX2 gövdeleri).
pub(super) fn blend_v_giris_icl(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    if bd != BitDepth::Eight
        || !karisim_sozlesmesi(w, (2, 32), h, (2, 128), h, tmp.len(), dst.len(), ds)
    {
        skaler_devri_say();
        return skaler!(bd, blend_v(dst, ds, tmp, w, h, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`mc_karisim_avx512::blend_v_fn`); belirteç AVX512ICL (AVX2'yi kapsar) tespitinin kanıtıdır.
    // (2) sınır: `karisim_sozlesmesi` `tmp`'nin `w·h` öğesini ve hedefte `h` satır
    // × `w` baytı kapsar; `h` 2'nin kuvveti olduğundan satır çiftleri tamdır;
    // erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `tmp` `&`.
    unsafe {
        match w {
            2 => blend_v_govde::<2>(dst, ds, tmp, h),
            4 => blend_v_govde::<4>(dst, ds, tmp, h),
            8 => blend_v_govde::<8>(dst, ds, tmp, h),
            16 => blend_v_govde::<16>(dst, ds, tmp, h),
            _ => super::mc_karisim_avx512::blend_v_govde(dst, ds, tmp, h),
        }
    }
}

/// `blend_h` girişi: [`blend_giris`] gibi (`w < 32` AVX2 gövdeleri).
pub(super) fn blend_h_giris_icl(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    if bd != BitDepth::Eight
        || !karisim_sozlesmesi(
            w,
            (2, 128),
            h,
            (2, 32),
            (h * 3) >> 2,
            tmp.len(),
            dst.len(),
            ds,
        )
    {
        skaler_devri_say();
        return skaler!(bd, blend_h(dst, ds, tmp, w, h, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`mc_karisim_avx512::blend_h_fn`); belirteç AVX512ICL (AVX2'yi kapsar) tespitinin kanıtıdır.
    // (2) sınır: `karisim_sozlesmesi` `tmp`'nin `w·h` öğesini ve hedefte yazılan
    // `(h·3) >> 2` satır × `w` baytı kapsar; erişimler denetimlidir. (3)
    // hizalama/takma ad: `dst` tek `&mut`, `tmp` `&`.
    unsafe {
        match w {
            2 => blend_h_govde::<2>(dst, ds, tmp, h),
            4 => blend_h_govde::<4>(dst, ds, tmp, h),
            8 => blend_h_govde::<8>(dst, ds, tmp, h),
            16 => blend_h_govde::<16>(dst, ds, tmp, h),
            32 => super::mc_karisim_avx512::blend_h_govde::<32>(dst, ds, tmp, h),
            64 => super::mc_karisim_avx512::blend_h_govde::<64>(dst, ds, tmp, h),
            _ => super::mc_karisim_avx512::blend_h_govde::<128>(dst, ds, tmp, h),
        }
    }
}
