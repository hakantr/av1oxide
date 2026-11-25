//! MC karışım SSSE3 çekirdekleri (16 bpc, 10/12 bit) — F4-MC d (P35):
//! `blend`, `blend_v`, `blend_h` (inter-intra / maske ve OBMC).
//!
//! ## Sözleşme
//!
//! AVX2 gövdeleriyle ([`super::mc16_karisim`]) aynı: girişte O(1) denetim
//! ([`crate::dsp::mc::karisim16_sozlesmesi`]); sözleşme dışı girdi skalere
//! devreder ve sayılır. Okuma/yazma yalnız `w` piksellik satırlar (`blend_h`
//! yalnız ilk `(h·3) >> 2` satır). Değer tanım kümesi AVX2 gövdesindeki gibi:
//! hedef ve `tmp` pikselleri `≤ bitdepth_max ≤ 4095`, maske 0…64.
//!
//! ## Yöntem
//!
//! Aritmetik AVX2 gövdesiyle aynıdır: hedef ve `tmp` pikselleri
//! `punpck{l,h}wd` ile ikili, `(64−m, m)` ağırlık ikilisiyle `pmaddwd` →
//! `i32`, `+ 32`, `psrad 6`. Sonuç tanım kümesinde `[0, 4095]` içinde
//! olduğundan `packssdw` (SSE2) `packusdw` (SSE4.1) ile aynı baytları verir.
//! Yazmaç 128 bittir; bir adım 16 piksel (iki yazmaç). `w ≥ 16` satır başına
//! 16'lık parçalar; `w` 4 ve 8'de iki yazmaca `16/w` satır toplanır (kısmi
//! grupta yalnız yazılan satırlar okunur); `w = 2` girişte doğrudan tamsayı
//! formülüyle (`#[target_feature]` çağrısı olmadan). Maske baytları
//! `punpck{l,h}bw` sıfırla genişler (`pmovzxbw` SSE4.1 olduğundan
//! kullanılmaz). SSE4.1 ve üstü komut yoktur.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mem::{chunk, chunk_mut, load_u8x16, store_u8x8, store_u8x16};
use crate::cpu::Ssse3Token;
use crate::dsp::mc::{BlendDirFn, BlendFn, bd16_tutarli, karisim16_sozlesmesi, skaler_devri_say};
use crate::dsp::mc_tables::OBMC_MASKS;
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_madd_epi16, _mm_packs_epi32, _mm_set_epi64x, _mm_set1_epi16,
    _mm_set1_epi32, _mm_setzero_si128, _mm_srai_epi32, _mm_srli_si128, _mm_sub_epi16,
    _mm_unpackhi_epi8, _mm_unpackhi_epi16, _mm_unpacklo_epi8, _mm_unpacklo_epi16,
};

/// 16 piksel: iki yazmaçlık sıralı `u16` (ya da `i16` maske) ikilisi.
type Iki = (__m128i, __m128i);

/// 8 pikselin karışımı `(d·(64−m) + t·m + 32) >> 6`: `d`, `t` sıralı `u16`,
/// `m` sıralı `i16` maske.
#[target_feature(enable = "ssse3")]
#[inline]
fn harmanla8(d: __m128i, t: __m128i, m: __m128i) -> __m128i {
    let n = _mm_sub_epi16(_mm_set1_epi16(64), m);
    let k = _mm_set1_epi32(32);
    let lo = _mm_madd_epi16(_mm_unpacklo_epi16(d, t), _mm_unpacklo_epi16(n, m));
    let hi = _mm_madd_epi16(_mm_unpackhi_epi16(d, t), _mm_unpackhi_epi16(n, m));
    let lo = _mm_srai_epi32::<6>(_mm_add_epi32(lo, k));
    let hi = _mm_srai_epi32::<6>(_mm_add_epi32(hi, k));
    _mm_packs_epi32(lo, hi)
}

/// 16 pikselin karışımı.
#[target_feature(enable = "ssse3")]
#[inline]
fn harmanla(d: Iki, t: Iki, m: Iki) -> Iki {
    (harmanla8(d.0, t.0, m.0), harmanla8(d.1, t.1, m.1))
}

/// 16 maske baytı `s[o..o+16]` → sıralı `i16` (iki yazmaç).
#[target_feature(enable = "ssse3")]
#[inline]
fn maske16(s: &[u8], o: usize) -> Iki {
    let b = load_u8x16(chunk(s, o));
    let z = _mm_setzero_si128();
    (_mm_unpacklo_epi8(b, z), _mm_unpackhi_epi8(b, z))
}

/// `s[o..o+32]` (iki 16'lık yükleme).
#[target_feature(enable = "ssse3")]
#[inline]
fn yukle32(s: &[u8], o: usize) -> Iki {
    (load_u8x16(chunk(s, o)), load_u8x16(chunk(s, o + 16)))
}

/// [`yukle32`]'nin tersi.
#[target_feature(enable = "ssse3")]
#[inline]
fn sakla32(s: &mut [u8], o: usize, v: Iki) {
    store_u8x16(chunk_mut(s, o), v.0);
    store_u8x16(chunk_mut(s, o + 16), v.1);
}

/// `W` (4, 8) sütunlu `satir ≤ 16/W` satırı (`y`'den) iki yazmaca toplar;
/// eksik satırlar 0 (okunmaz).
#[target_feature(enable = "ssse3")]
#[inline]
fn topla<const W: usize>(d: &[u8], ds: usize, y: usize, satir: usize) -> Iki {
    if W == 4 {
        let mut r = [0i64; 4];
        for (i, v) in r.iter_mut().enumerate().take(satir) {
            *v = i64::from_le_bytes(*chunk(d, (y + i) * ds));
        }
        (_mm_set_epi64x(r[1], r[0]), _mm_set_epi64x(r[3], r[2]))
    } else {
        let r0 = load_u8x16(chunk(d, y * ds));
        let r1 = if satir > 1 {
            load_u8x16(chunk(d, (y + 1) * ds))
        } else {
            _mm_setzero_si128()
        };
        (r0, r1)
    }
}

/// [`topla`]'nın tersi: yalnız ilk `satir` satırı yazar.
#[target_feature(enable = "ssse3")]
#[inline]
fn dagit<const W: usize>(d: &mut [u8], ds: usize, y: usize, satir: usize, v: Iki) {
    if W == 4 {
        store_u8x8(chunk_mut(d, y * ds), v.0);
        if satir > 1 {
            store_u8x8(chunk_mut(d, (y + 1) * ds), _mm_srli_si128::<8>(v.0));
        }
        if satir > 2 {
            store_u8x8(chunk_mut(d, (y + 2) * ds), v.1);
        }
        if satir > 3 {
            store_u8x8(chunk_mut(d, (y + 3) * ds), _mm_srli_si128::<8>(v.1));
        }
    } else {
        store_u8x16(chunk_mut(d, y * ds), v.0);
        if satir > 1 {
            store_u8x16(chunk_mut(d, (y + 1) * ds), v.1);
        }
    }
}

/// `s`'nin `o`'dan `n ≤ 16` baytı; dilimde 16 bayt varsa doğrudan yüklenir
/// (fazla baytlar `tmp`'nin kendi bölgesindedir ve yazılmayan satırlara gider),
/// yoksa eksik baytlar 0 (dilim dışı okunmaz).
#[target_feature(enable = "ssse3")]
#[inline]
fn bitisik16(s: &[u8], o: usize, n: usize) -> __m128i {
    if n == 16 || o + 16 <= s.len() {
        load_u8x16(chunk(s, o))
    } else if n == 0 {
        _mm_setzero_si128()
    } else {
        let mut b = [0u8; 16];
        b[..n].copy_from_slice(&s[o..o + n]);
        load_u8x16(&b)
    }
}

/// `s`'nin `o`'dan `n ≤ 32` baytı (iki yazmaç; bkz. [`bitisik16`]).
#[target_feature(enable = "ssse3")]
#[inline]
fn bitisik32(s: &[u8], o: usize, n: usize) -> Iki {
    (
        bitisik16(s, o, n.min(16)),
        bitisik16(s, o + 16, n.saturating_sub(16)),
    )
}

/// `blend` gövdesi, `W` sütun: `W ≥ 16` satır başına 16'lık parçalar, `W ≤ 8`
/// iki yazmaca `16/W` satır (`h` 2'nin kuvveti ≥ 4 olduğundan tam gruplar).
#[target_feature(enable = "ssse3")]
fn blend_govde<const W: usize>(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize, mask: &[u8]) {
    if W >= 16 {
        for y in 0..h {
            for x in (0..W).step_by(16) {
                let (o, od) = (y * W + x, y * ds + 2 * x);
                let r = harmanla(yukle32(dst, od), yukle32(tmp, 2 * o), maske16(mask, o));
                sakla32(dst, od, r);
            }
        }
        return;
    }
    let rpx = 16 / W;
    for y in (0..h).step_by(rpx) {
        let o = y * W;
        let r = harmanla(
            topla::<W>(dst, ds, y, rpx),
            yukle32(tmp, 2 * o),
            maske16(mask, o),
        );
        dagit::<W>(dst, ds, y, rpx, r);
    }
}

/// Tek piksel karışımı `(d·(64−m) + t·m + 32) >> 6` (`w = 2`).
#[inline(always)]
fn px(dst: &mut [u8], od: usize, tmp: &[u8], ot: usize, m: i32) {
    let d = i32::from(u16::from_ne_bytes(*chunk(dst, od)));
    let t = i32::from(u16::from_ne_bytes(*chunk(tmp, ot)));
    let v = ((d * (64 - m) + t * m + 32) >> 6) as u16;
    *chunk_mut(dst, od) = v.to_ne_bytes();
}

/// `blend_v`, `w = 2`: yalnız sütun 0 karışır (`(2·3) >> 2 = 1`; sütun 1'in
/// maskesi 0). Girişte doğrudan tamsayı formülü (AVX2 gövdesindeki gibi).
fn blend_v_iki(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize) {
    let m = i32::from(OBMC_MASKS[2]);
    for y in 0..h {
        px(dst, y * ds, tmp, y * 4, m);
    }
}

/// `blend_h`, `w = 2`: ilk `(h·3) >> 2` satır, satır maskesi `OBMC_MASKS[h + y]`
/// (girişte doğrudan tamsayı formülü; bkz. [`blend_v_iki`]).
fn blend_h_iki(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize) {
    for (y, &m) in OBMC_MASKS[h..].iter().enumerate().take((h * 3) >> 2) {
        // Satırın iki pikseli tek 4 baytlık yükleme/saklamayla.
        let m = u32::from(m);
        let d = u32::from_ne_bytes(*chunk(dst, y * ds));
        let t = u32::from_ne_bytes(*chunk(tmp, y * 4));
        let k = |d: u32, t: u32| (d * (64 - m) + t * m + 32) >> 6;
        let v0 = k(d & 0xffff, t & 0xffff);
        let v1 = k(d >> 16, t >> 16);
        *chunk_mut(dst, y * ds) = (v0 | (v1 << 16)).to_ne_bytes();
    }
}

/// `blend_v` gövdesi: sütun maskesi `OBMC_MASKS[W + x]`, satırlara yayılı.
#[target_feature(enable = "ssse3")]
fn blend_v_govde<const W: usize>(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize) {
    let tablo = &BLEND_V_MASKE[W.trailing_zeros() as usize - 2];
    let m0 = maske16(tablo, 0);
    if W >= 16 {
        let m1 = maske16(tablo, 16);
        for y in 0..h {
            for x in (0..W).step_by(16) {
                let od = y * ds + 2 * x;
                let m = if x == 0 { m0 } else { m1 };
                let r = harmanla(yukle32(dst, od), yukle32(tmp, 2 * (y * W + x)), m);
                sakla32(dst, od, r);
            }
        }
        return;
    }
    let rpx = 16 / W;
    let mut y = 0;
    while y < h {
        // `W = 4`, `h = 2`'de tek kısmi grup.
        let satir = rpx.min(h - y);
        let t = bitisik32(tmp, 2 * y * W, 2 * satir * W);
        let r = harmanla(topla::<W>(dst, ds, y, satir), t, m0);
        dagit::<W>(dst, ds, y, satir, r);
        y += satir;
    }
}

/// `blend_h` gövdesi: satır maskesi `OBMC_MASKS[h + y]`, ilk `(h·3) >> 2` satır.
#[target_feature(enable = "ssse3")]
fn blend_h_govde<const W: usize>(dst: &mut [u8], ds: usize, tmp: &[u8], h: usize) {
    let satirlar = (h * 3) >> 2;
    let tablo = &OBMC_MASKS[h..];
    if W >= 16 {
        for (y, &my) in tablo.iter().enumerate().take(satirlar) {
            let m = _mm_set1_epi16(i16::from(my));
            for x in (0..W).step_by(16) {
                let od = y * ds + 2 * x;
                let r = harmanla(yukle32(dst, od), yukle32(tmp, 2 * (y * W + x)), (m, m));
                sakla32(dst, od, r);
            }
        }
        return;
    }
    let rpx = 16 / W;
    let mut y = 0;
    while y < satirlar {
        let satir = rpx.min(satirlar - y);
        // Satır maskesi yayını: satır `r` için `W` öğe `tablo[y + r]` (eksik satır 0).
        let mut m = [0u16; 4];
        for (r, v) in m.iter_mut().enumerate().take(satir) {
            *v = u16::from(tablo[y + r]);
        }
        let mb = if W == 4 {
            let k = |r: usize| (u64::from(m[r]) * 0x0001_0001_0001_0001) as i64;
            (_mm_set_epi64x(k(1), k(0)), _mm_set_epi64x(k(3), k(2)))
        } else {
            (_mm_set1_epi16(m[0] as i16), _mm_set1_epi16(m[1] as i16))
        };
        let t = bitisik32(tmp, 2 * y * W, 2 * satir * W);
        let r = harmanla(topla::<W>(dst, ds, y, satir), t, mb);
        dagit::<W>(dst, ds, y, satir, r);
        y += satir;
    }
}

/// `blend_v` sütun maskesi `OBMC_MASKS[W + x]`, 32 bayta yayılı (`W` = 4, 8, 16, 32;
/// AVX2 gövdelerinin tablosuyla aynı, kendi kopyası).
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

fn blend16_giris(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    mask: &[u8],
    bd: BitDepth,
    bdmax: u16,
) {
    if mask.len() < w * h
        || w * h < 16
        || !karisim16_sozlesmesi(
            w,
            (4, 32),
            h,
            (4, 32),
            h,
            tmp.len(),
            dst.len(),
            ds,
            bd,
            bdmax,
        )
    {
        skaler_devri_say();
        return skaler!(bd, blend(dst, ds, tmp, w, h, mask, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`blend_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `karisim16_sozlesmesi` `tmp`'nin `w·h` pikselini ve hedefte `h` satır × `2w`
    // baytı, `mask.len() ≥ w·h` maskeyi kapsar; erişimler denetimlidir. (3)
    // hizalama/takma ad: `dst` tek `&mut`, girdiler `&`; ödünç kuralları çakışmayı dışlar.
    unsafe {
        match w {
            4 => blend_govde::<4>(dst, ds, tmp, h, mask),
            8 => blend_govde::<8>(dst, ds, tmp, h, mask),
            16 => blend_govde::<16>(dst, ds, tmp, h, mask),
            _ => blend_govde::<32>(dst, ds, tmp, h, mask),
        }
    }
}

fn blend_v16_giris(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    // `w = 2`: girişte doğrudan tamsayı formülü (güvenli kod; skalerle aynı sırada
    // aynı erişimler, sözleşme denetimine gerek yok — yalnız bit derinliği).
    if w == 2 && bd16_tutarli(bd, bdmax) {
        return blend_v_iki(dst, ds, tmp, h);
    }
    if !karisim16_sozlesmesi(
        w,
        (2, 32),
        h,
        (2, 128),
        h,
        tmp.len(),
        dst.len(),
        ds,
        bd,
        bdmax,
    ) {
        skaler_devri_say();
        return skaler!(bd, blend_v(dst, ds, tmp, w, h, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`blend_v_fn`); belirteç SSSE3 tespitinin kanıtıdır. (2) sınır:
    // `karisim16_sozlesmesi` `tmp`'nin `w·h` pikselini ve hedefte `h` satır × `2w`
    // baytı kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek
    // `&mut`, `tmp` `&`.
    unsafe {
        match w {
            4 => blend_v_govde::<4>(dst, ds, tmp, h),
            8 => blend_v_govde::<8>(dst, ds, tmp, h),
            16 => blend_v_govde::<16>(dst, ds, tmp, h),
            _ => blend_v_govde::<32>(dst, ds, tmp, h),
        }
    }
}

fn blend_h16_giris(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    // `w = 2`: girişte doğrudan tamsayı formülü (güvenli kod; skalerle aynı sırada
    // aynı erişimler, sözleşme denetimine gerek yok — yalnız bit derinliği).
    if w == 2 && bd16_tutarli(bd, bdmax) {
        return blend_h_iki(dst, ds, tmp, h);
    }
    if !karisim16_sozlesmesi(
        w,
        (2, 128),
        h,
        (2, 32),
        (h * 3) >> 2,
        tmp.len(),
        dst.len(),
        ds,
        bd,
        bdmax,
    ) {
        skaler_devri_say();
        return skaler!(bd, blend_h(dst, ds, tmp, w, h, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`blend_h_fn`); belirteç SSSE3 tespitinin kanıtıdır. (2) sınır:
    // `karisim16_sozlesmesi` `tmp`'nin `w·h` pikselini ve hedefte yazılan
    // `(h·3) >> 2` satır × `2w` baytı kapsar; erişimler denetimlidir. (3)
    // hizalama/takma ad: `dst` tek `&mut`, `tmp` `&`.
    unsafe {
        match w {
            4 => blend_h_govde::<4>(dst, ds, tmp, h),
            8 => blend_h_govde::<8>(dst, ds, tmp, h),
            16 => blend_h_govde::<16>(dst, ds, tmp, h),
            32 => blend_h_govde::<32>(dst, ds, tmp, h),
            64 => blend_h_govde::<64>(dst, ds, tmp, h),
            _ => blend_h_govde::<128>(dst, ds, tmp, h),
        }
    }
}

/// 16 bpc `blend` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn blend_fn(_ssse3: Ssse3Token) -> BlendFn {
    blend16_giris
}

/// 16 bpc `blend_v` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn blend_v_fn(_ssse3: Ssse3Token) -> BlendDirFn {
    blend_v16_giris
}

/// 16 bpc `blend_h` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn blend_h_fn(_ssse3: Ssse3Token) -> BlendDirFn {
    blend_h16_giris
}
