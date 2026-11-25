//! MC karışım SSSE3 çekirdekleri (8 bpc) — F4-MC d (P35): `blend`, `blend_v`,
//! `blend_h` (inter-intra / maske ve OBMC).
//!
//! ## Sözleşme
//!
//! AVX2 gövdeleriyle ([`super::mc_karisim`]) aynı: girişte O(1) denetim
//! ([`crate::dsp::mc::karisim_sozlesmesi`]); sözleşme dışı girdi skalere
//! devreder ve sayılır. Okuma/yazma yalnız `w` sütunluk satırlar (`blend_h`
//! yalnız ilk `(h·3) >> 2` satır).
//!
//! ## Yöntem
//!
//! Aritmetik AVX2 gövdesiyle aynıdır: `(d·(64−m) + t·m + 32) >> 6` hedef ve
//! `tmp` baytlarının `punpck{l,h}bw` ikilisi ile `(64−m, m)` ağırlık
//! ikilisinin `pmaddubsw`'si, `pmulhrsw(·, 512)`, `packuswb`. Yazmaç 128
//! bittir: `w ≥ 32` satır başına 16 piksellik parçalar; `w` 4…16'da bir xmm'e
//! `16/w` satır toplanır (kısmi grupta yalnız yazılan satırlar okunur); `w = 2`
//! doğrudan tamsayı formülüyle. SSE4.1 ve üstü komut yoktur.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mem::{chunk, chunk_mut, load_u8x16, store_u8x4, store_u8x8, store_u8x16};
use crate::cpu::Ssse3Token;
use crate::dsp::mc::{BlendDirFn, BlendFn, karisim_sozlesmesi, skaler_devri_say};
use crate::dsp::mc_tables::OBMC_MASKS;
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, _mm_maddubs_epi16, _mm_mulhrs_epi16, _mm_packus_epi16, _mm_set_epi64x, _mm_set1_epi8,
    _mm_set1_epi16, _mm_setr_epi32, _mm_srli_si128, _mm_sub_epi8, _mm_unpackhi_epi8,
    _mm_unpacklo_epi8,
};

/// Karışım (16 piksel): `(d·(64−m) + t·m + 32) >> 6`.
#[target_feature(enable = "ssse3")]
#[inline]
fn harmanla(d: __m128i, t: __m128i, mb: __m128i) -> __m128i {
    let ters = _mm_sub_epi8(_mm_set1_epi8(64), mb);
    let k = _mm_set1_epi16(512);
    let lo = _mm_maddubs_epi16(_mm_unpacklo_epi8(d, t), _mm_unpacklo_epi8(ters, mb));
    let hi = _mm_maddubs_epi16(_mm_unpackhi_epi8(d, t), _mm_unpackhi_epi8(ters, mb));
    _mm_packus_epi16(_mm_mulhrs_epi16(lo, k), _mm_mulhrs_epi16(hi, k))
}

/// `W` (4, 8, 16) sütunlu `satir ≤ 16/W` satırı (`y`'den) bir xmm'e toplar;
/// eksik satırlar 0 (okunmaz).
#[target_feature(enable = "ssse3")]
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
#[target_feature(enable = "ssse3")]
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
#[target_feature(enable = "ssse3")]
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

/// Hedef satır parçası (`do`, 16 piksel) ile `tmp` parçasının (`to`) karışımı.
#[target_feature(enable = "ssse3")]
#[inline]
fn parca16(dst: &mut [u8], o: usize, tmp: &[u8], to: usize, mb: __m128i) {
    let r = harmanla(load_u8x16(chunk(dst, o)), load_u8x16(chunk(tmp, to)), mb);
    store_u8x16(chunk_mut(dst, o), r);
}

/// `blend` gövdesi, `W` sütun: `W ≥ 32` satır başına 16 piksellik parçalar,
/// `W ≤ 16` xmm başına `16/W` satır (`h` 2'nin kuvveti ≥ 4 olduğundan tam
/// gruplar).
#[target_feature(enable = "ssse3")]
fn blend_govde<const W: usize>(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize, mask: &[u8]) {
    if W >= 32 {
        for y in 0..h {
            for x in (0..W).step_by(16) {
                let o = y * W + x;
                parca16(dst, y * ds + x, tmp, o, load_u8x16(chunk(mask, o)));
            }
        }
        return;
    }
    let rpx = 16 / W;
    for y in (0..h).step_by(rpx) {
        let o = y * W;
        let r = harmanla(
            topla16::<W>(dst, ds, y, rpx),
            load_u8x16(chunk(tmp, o)),
            load_u8x16(chunk(mask, o)),
        );
        dagit16::<W>(dst, ds, y, rpx, r);
    }
}

/// `blend_v` gövdesi: sütun maskesi `OBMC_MASKS[W + x]`, satırlara yayılı.
#[target_feature(enable = "ssse3")]
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
        let m0 = load_u8x16(chunk(tablo, 0));
        let m1 = load_u8x16(chunk(tablo, 16));
        for y in 0..h {
            parca16(dst, y * ds, tmp, y * W, m0);
            parca16(dst, y * ds + 16, tmp, y * W + 16, m1);
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
        let r = harmanla(topla16::<W>(dst, ds, y, satir), t, m);
        dagit16::<W>(dst, ds, y, satir, r);
        y += satir;
    }
}

/// `blend_h` gövdesi: satır maskesi `OBMC_MASKS[h + y]`, ilk `(h·3) >> 2` satır.
#[target_feature(enable = "ssse3")]
fn blend_h_govde<const W: usize>(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize) {
    let satirlar = (h * 3) >> 2;
    let tablo = &OBMC_MASKS[h..];
    if W >= 32 {
        for (y, &my) in tablo.iter().enumerate().take(satirlar) {
            let m = _mm_set1_epi8(my as i8);
            for x in (0..W).step_by(16) {
                parca16(dst, y * ds + x, tmp, y * W + x, m);
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
        let r = harmanla(topla16::<W>(dst, ds, y, satir), t, mb);
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`blend_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır. (2) sınır:
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`blend_v_fn`); belirteç SSSE3 tespitinin kanıtıdır. (2) sınır:
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`blend_h_fn`); belirteç SSSE3 tespitinin kanıtıdır. (2) sınır:
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

/// `blend` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn blend_fn(_ssse3: Ssse3Token) -> BlendFn {
    blend_giris
}

/// `blend_v` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn blend_v_fn(_ssse3: Ssse3Token) -> BlendDirFn {
    blend_v_giris
}

/// `blend_h` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn blend_h_fn(_ssse3: Ssse3Token) -> BlendDirFn {
    blend_h_giris
}
