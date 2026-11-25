//! MC bileşik AVX2 çekirdekleri (8 bpc) — F4-MC b (P33): `avg`, `w_avg`,
//! `mask`, `w_mask[0..2]` (4:4:4, 4:2:2, 4:2:0).
//!
//! ## Sözleşme
//!
//! Girişte O(1) denetim ([`crate::dsp::mc::bilesik_sozlesmesi`]): `w`, `h`
//! 2'nin kuvveti (`avg`/`w_avg`/`mask` 4…128, `w_mask` 8…128), `tmp1`/`tmp2`
//! (ve `mask`) en az `w·h` öğe, `dst_stride ≥ w`, `dst` en az
//! `(h−1)·dst_stride + w` bayt; `w_avg` ağırlığı 0…16; `w_mask`'ta `sign` 0/1,
//! `(ss_hor, ss_ver)` yuvanın alt türüyle aynı ve maske en az
//! `(w >> ss_hor)·(h >> ss_ver)` bayt. Sözleşme dışı girdi skalere devreder ve
//! sayılır ([`crate::dsp::mc::skaler_devir_sayisi`]). Okuma tam `w·h` öğedir
//! (`tmp` satırları bitişik), yazma yalnız `h` satır × `w` bayt (+ maske).
//!
//! **Değer tanım kümesi:** girdiler `prep`/`warp8x8t` çıktılarıdır (8 bpc
//! [`PREP_ARALIK`]). Bu kümede `a + b` ve `2·(a − b)` `i16`'ya sığar; 16 bit
//! aritmetik skaler `i32` formülüyle bit-exact'tir (dav1d AVX2 aynı varsayımı
//! kullanır). Testler rastgele girdiyi bu aralıktan çeker.
//!
//! ## Yöntem (dav1d `mc_avx2.asm` yaklaşımı; asm kopyalanmadı)
//!
//! - `avg`: `(a + b + 16) >> 5` = `pmulhrsw(a + b, 1024)`.
//! - `w_avg`: `(a·w + b·(16−w) + 128) >> 8` = `(⌊(a−b)(w−16)/16⌋ + a + 8) >> 4`;
//!   `⌊x·k/2¹⁶⌋` `pmulhw` ile, `k = (w−16)·2¹²` (`w ≤ 7`'de girdiler yer
//!   değiştirir, `k = −w·2¹²`; ikisi de `i16`'ya sığar), son `>> 4` yuvarlamalı
//!   `pmulhrsw(·, 2048)`.
//! - `mask`: `(a·m + b·(64−m) + 512) >> 10` = `(⌊2(b−a)·(−m·2⁹)/2¹⁶⌋ + b + 8) >> 4`
//!   (`m = 64`'te `−m·2⁹ = −2¹⁵` tam sığar).
//! - `w_mask`: `64 − m` = `(6903 ⊖ |t₂−t₁|) >> 8` (doyurmalı çıkarma; 6903 =
//!   26·256 + 247), değer `(⌊(t₂−t₁)·(64−m)·2¹⁰/2¹⁶⌋ + t₁ + 8) >> 4`; maske 4:2:2
//!   ikili toplam `pmaddubsw`, 4:2:0 iki satırın ikili toplamı.
//! - 32 öğe bir adım: iki `i16` yazmacı `packuswb` + `vpermq` ile sıralı 32 bayt;
//!   `tmp` satırları bitişik olduğundan `w < 32`'de adım birden çok satır yazar.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mem::{
    chunk, chunk_mut, load_i16x16, load_u8x16, store_u8x4, store_u8x8, store_u8x16, store_u8x32,
};
use crate::cpu::Avx2Token;
use crate::dsp::mc::{AvgFn, MaskFn, WAvgFn, WMaskFn, bilesik_sozlesmesi, skaler_devri_say};
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m256i, _mm_add_epi16, _mm_packus_epi16, _mm_set1_epi16, _mm_srli_epi16, _mm_srli_si128,
    _mm_unpacklo_epi64, _mm256_abs_epi16, _mm256_add_epi16, _mm256_castsi256_si128,
    _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_maddubs_epi16, _mm256_mulhi_epi16,
    _mm256_mulhrs_epi16, _mm256_packus_epi16, _mm256_permute4x64_epi64, _mm256_set1_epi8,
    _mm256_set1_epi16, _mm256_setzero_si256, _mm256_slli_epi16, _mm256_srli_epi16,
    _mm256_sub_epi16, _mm256_subs_epu16,
};

/// 8 bpc `prep` çıktı aralığı — dav1d `src/mc_tmpl.c:40` ("Output in interval
/// [-5132, 9212]"); filtre tablolarından bağımsız hesap da aynı sınırı verir
/// (`warp8x8t` `[−4096, 8176]`, kopya ve bilinear `[0, 4080]` içinde kalır).
/// Bu aralıkta `a + b` ve `2·(a − b)` `i16`'ya sığar.
pub const PREP_ARALIK: (i16, i16) = (-5132, 9212);

/// Bileşik işlem (derleme zamanı).
pub(super) const AVG: u8 = 0;
pub(super) const W_AVG: u8 = 1;
pub(super) const MASK: u8 = 2;

/// Bir adımın girdileri: `t1`, `t2` (`w_avg`'da gerekirse yer değiştirmiş),
/// `mask` (`mask`'ta), `k` (`w_avg` çarpanı).
pub(super) struct Girdi<'a> {
    pub(super) t1: &'a [i16],
    pub(super) t2: &'a [i16],
    pub(super) mask: &'a [u8],
    pub(super) k: i16,
}

/// 16 öğenin `i16` sonucu (doyurmalı `packuswb`'ye hazır), öğe ofseti `o`.
#[target_feature(enable = "avx2")]
#[inline]
fn hesap16<const OP: u8>(g: &Girdi<'_>, o: usize) -> __m256i {
    let a = load_i16x16(chunk(g.t1, o));
    let b = load_i16x16(chunk(g.t2, o));
    match OP {
        AVG => _mm256_mulhrs_epi16(_mm256_add_epi16(a, b), _mm256_set1_epi16(1024)),
        W_AVG => {
            let r = _mm256_add_epi16(
                a,
                _mm256_mulhi_epi16(_mm256_sub_epi16(a, b), _mm256_set1_epi16(g.k)),
            );
            _mm256_mulhrs_epi16(r, _mm256_set1_epi16(2048))
        }
        _ => {
            let m = _mm256_cvtepu8_epi16(load_u8x16(chunk(g.mask, o)));
            let k = _mm256_slli_epi16::<9>(_mm256_sub_epi16(_mm256_setzero_si256(), m));
            let d = _mm256_slli_epi16::<1>(_mm256_sub_epi16(b, a));
            let r = _mm256_add_epi16(b, _mm256_mulhi_epi16(d, k));
            _mm256_mulhrs_epi16(r, _mm256_set1_epi16(2048))
        }
    }
}

/// İki 16'lık `i16` sonuçtan sıralı 32 bayt.
#[target_feature(enable = "avx2")]
#[inline]
fn paketle32(r0: __m256i, r1: __m256i) -> __m256i {
    _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi16(r0, r1))
}

/// Öğe ofseti `o`'dan başlayan sıralı 32 pikseli (`W` sütunlu bitişik satırlar)
/// hedefe yazar; `W ≥ 32`'de tek satır parçası.
#[target_feature(enable = "avx2")]
#[inline]
fn sakla32<const W: usize>(dst: &mut [u8], ds: usize, o: usize, p: __m256i) {
    let (y, x) = (o / W, o % W);
    let lo = _mm256_castsi256_si128(p);
    let hi = _mm256_extracti128_si256::<1>(p);
    match W {
        4 => {
            let mut v = lo;
            for r in 0..4 {
                store_u8x4(chunk_mut(dst, (y + r) * ds), v);
                v = _mm_srli_si128::<4>(v);
            }
            let mut v = hi;
            for r in 4..8 {
                store_u8x4(chunk_mut(dst, (y + r) * ds), v);
                v = _mm_srli_si128::<4>(v);
            }
        }
        8 => {
            store_u8x8(chunk_mut(dst, y * ds), lo);
            store_u8x8(chunk_mut(dst, (y + 1) * ds), _mm_srli_si128::<8>(lo));
            store_u8x8(chunk_mut(dst, (y + 2) * ds), hi);
            store_u8x8(chunk_mut(dst, (y + 3) * ds), _mm_srli_si128::<8>(hi));
        }
        16 => {
            store_u8x16(chunk_mut(dst, y * ds), lo);
            store_u8x16(chunk_mut(dst, (y + 1) * ds), hi);
        }
        _ => store_u8x32(chunk_mut(dst, y * ds + x), p),
    }
}

/// `avg` / `w_avg` / `mask` gövdesi: `W` sütun (`W = 32` ⇒ `w ≥ 32`).
#[target_feature(enable = "avx2")]
fn bidir<const OP: u8, const W: usize>(dst: &mut [u8], ds: usize, n: usize, g: &Girdi<'_>) {
    if n == 16 {
        // 4×4: tek yazmaç, dört satır.
        let r = hesap16::<OP>(g, 0);
        let p = _mm256_packus_epi16(r, r);
        let lo = _mm256_castsi256_si128(p);
        let hi = _mm256_extracti128_si256::<1>(p);
        store_u8x4(chunk_mut(dst, 0), lo);
        store_u8x4(chunk_mut(dst, ds), _mm_srli_si128::<4>(lo));
        store_u8x4(chunk_mut(dst, 2 * ds), hi);
        store_u8x4(chunk_mut(dst, 3 * ds), _mm_srli_si128::<4>(hi));
        return;
    }
    let mut o = 0;
    while o < n {
        let p = paketle32(hesap16::<OP>(g, o), hesap16::<OP>(g, o + 16));
        sakla32::<W>(dst, ds, o, p);
        o += 32;
    }
}

/// Genişliğe göre `bidir` monomorfu.
#[target_feature(enable = "avx2")]
fn bidir_w<const OP: u8>(dst: &mut [u8], ds: usize, w: usize, h: usize, g: &Girdi<'_>) {
    let n = w * h;
    match w {
        4 => bidir::<OP, 4>(dst, ds, n, g),
        8 => bidir::<OP, 8>(dst, ds, n, g),
        16 => bidir::<OP, 16>(dst, ds, n, g),
        32 => bidir::<OP, 32>(dst, ds, n, g),
        64 => bidir::<OP, 64>(dst, ds, n, g),
        _ => bidir::<OP, 128>(dst, ds, n, g),
    }
}

/// `w_mask` 16 öğesi: (`i16` piksel sonucu, `64 − m`).
#[target_feature(enable = "avx2")]
#[inline]
fn wm16(t1: &[i16], t2: &[i16], o: usize) -> (__m256i, __m256i) {
    let a = load_i16x16(chunk(t1, o));
    let b = load_i16x16(chunk(t2, o));
    let d = _mm256_sub_epi16(b, a);
    let k = _mm256_srli_epi16::<8>(_mm256_subs_epu16(
        _mm256_set1_epi16(6903),
        _mm256_abs_epi16(d),
    ));
    let r = _mm256_add_epi16(a, _mm256_mulhi_epi16(d, _mm256_slli_epi16::<10>(k)));
    (_mm256_mulhrs_epi16(r, _mm256_set1_epi16(2048)), k)
}

/// 32 öğenin `m` baytları (sıralı) ve 16 ikili toplamı (`i16`, sıralı).
#[target_feature(enable = "avx2")]
#[inline]
fn m_bayt_ve_ikili(k0: __m256i, k1: __m256i) -> (__m256i, __m256i) {
    let altmis4 = _mm256_set1_epi16(64);
    let m = paketle32(_mm256_sub_epi16(altmis4, k0), _mm256_sub_epi16(altmis4, k1));
    (m, _mm256_maddubs_epi16(m, _mm256_set1_epi8(1)))
}

/// `w_mask` gövdesi: alt tür `SUB` (0: 4:4:4, 1: 4:2:2, 2: 4:2:0), `W` sütun.
#[target_feature(enable = "avx2")]
fn w_mask_govde<const SUB: usize, const W: usize>(
    dst: &mut [u8],
    ds: usize,
    t1: &[i16],
    t2: &[i16],
    h: usize,
    mask: &mut [u8],
    sign: i16,
) {
    let n = W * h;
    if SUB < 2 {
        // 4:4:4 / 4:2:2: öğe sırasıyla; maske satırları da bitişik.
        let ek = _mm256_set1_epi16(1 - sign);
        let mut o = 0;
        while o < n {
            let (r0, k0) = wm16(t1, t2, o);
            let (r1, k1) = wm16(t1, t2, o + 16);
            sakla32::<W>(dst, ds, o, paketle32(r0, r1));
            let (m, ikili) = m_bayt_ve_ikili(k0, k1);
            if SUB == 0 {
                store_u8x32(chunk_mut(mask, o), m);
            } else {
                let v = _mm256_srli_epi16::<1>(_mm256_add_epi16(ikili, ek));
                let p = paketle32(v, v);
                store_u8x16(chunk_mut(mask, o / 2), _mm256_castsi256_si128(p));
            }
            o += 32;
        }
        return;
    }
    // 4:2:0: satır ikilileri; maske satırı `W/2` bayt.
    let ek = _mm_set1_epi16(2 - sign);
    let mut o = 0;
    while o < n {
        if W >= 32 {
            // Satır `y` ve `y+1`'in aynı 32 sütunu.
            let (y, x) = (o / W, o % W);
            let o1 = o + W;
            let (r0, k0) = wm16(t1, t2, o);
            let (r1, k1) = wm16(t1, t2, o + 16);
            let (r2, k2) = wm16(t1, t2, o1);
            let (r3, k3) = wm16(t1, t2, o1 + 16);
            sakla32::<W>(dst, ds, o, paketle32(r0, r1));
            sakla32::<W>(dst, ds, o1, paketle32(r2, r3));
            let (_, s0) = m_bayt_ve_ikili(k0, k1);
            let (_, s1) = m_bayt_ve_ikili(k2, k3);
            let t = _mm256_add_epi16(s0, s1);
            let lo = _mm_srli_epi16::<2>(_mm_add_epi16(_mm256_castsi256_si128(t), ek));
            let hi = _mm_srli_epi16::<2>(_mm_add_epi16(_mm256_extracti128_si256::<1>(t), ek));
            store_u8x16(
                chunk_mut(mask, (y / 2) * (W / 2) + x / 2),
                _mm_packus_epi16(lo, hi),
            );
            o += if x + 32 >= W { W + 32 } else { 32 };
        } else {
            // `W = 16`: 32 öğe = satır ikilisi; `W = 8`: iki satır ikilisi.
            let (r0, k0) = wm16(t1, t2, o);
            let (r1, k1) = wm16(t1, t2, o + 16);
            sakla32::<W>(dst, ds, o, paketle32(r0, r1));
            let (_, s) = m_bayt_ve_ikili(k0, k1);
            let lo = _mm256_castsi256_si128(s);
            let hi = _mm256_extracti128_si256::<1>(s);
            let v = if W == 16 {
                // lo = satır y'nin 8 ikilisi, hi = satır y+1'in.
                _mm_add_epi16(lo, hi)
            } else {
                // lo = satır y (4) | y+1 (4), hi = y+2 | y+3.
                let a = _mm_add_epi16(lo, _mm_srli_si128::<8>(lo));
                let b = _mm_add_epi16(hi, _mm_srli_si128::<8>(hi));
                _mm_unpacklo_epi64(a, b)
            };
            let v = _mm_srli_epi16::<2>(_mm_add_epi16(v, ek));
            let p = _mm_packus_epi16(v, v);
            store_u8x8(chunk_mut(mask, o / 4), p);
            o += 32;
        }
    }
}

/// `w_mask` genişlik monomorfu.
#[target_feature(enable = "avx2")]
fn w_mask_w<const SUB: usize>(
    dst: &mut [u8],
    ds: usize,
    t1: &[i16],
    t2: &[i16],
    w: usize,
    h: usize,
    mask: &mut [u8],
    sign: i16,
) {
    match w {
        8 => w_mask_govde::<SUB, 8>(dst, ds, t1, t2, h, mask, sign),
        16 => w_mask_govde::<SUB, 16>(dst, ds, t1, t2, h, mask, sign),
        32 => w_mask_govde::<SUB, 32>(dst, ds, t1, t2, h, mask, sign),
        64 => w_mask_govde::<SUB, 64>(dst, ds, t1, t2, h, mask, sign),
        _ => w_mask_govde::<SUB, 128>(dst, ds, t1, t2, h, mask, sign),
    }
}

/// Skaler gövde (bit derinliğine göre monomorf).
macro_rules! skaler {
    ($bd:expr, $f:ident ( $($a:expr),* )) => {
        match $bd {
            BitDepth::Eight => scalar::$f::<8>($($a),*),
            BitDepth::Ten => scalar::$f::<10>($($a),*),
            BitDepth::Twelve => scalar::$f::<12>($($a),*),
        }
    };
}

fn avg_giris(
    dst: &mut [u8],
    ds: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    if bd != BitDepth::Eight
        || !bilesik_sozlesmesi(w, h, 4, tmp1.len(), tmp2.len(), w * h, dst.len(), ds)
    {
        skaler_devri_say();
        return skaler!(bd, avg(dst, ds, tmp1, tmp2, w, h, bd, bdmax));
    }
    let g = Girdi {
        t1: tmp1,
        t2: tmp2,
        mask: &[],
        k: 0,
    };
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`avg_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `bilesik_sozlesmesi` `tmp` dilimlerinin `w·h` öğesini ve hedefte `h` satır ×
    // `w` baytı kapsar; gövde erişimleri dilim/dizi denetimlidir. (3)
    // hizalama/takma ad: `dst` tek `&mut`, girdiler `&`; ödünç kuralları çakışmayı
    // dışlar.
    unsafe { bidir_w::<AVG>(dst, ds, w, h, &g) }
}

fn w_avg_giris(
    dst: &mut [u8],
    ds: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
    bd: BitDepth,
    bdmax: u16,
) {
    if bd != BitDepth::Eight
        || !(0..=16).contains(&weight)
        || !bilesik_sozlesmesi(w, h, 4, tmp1.len(), tmp2.len(), w * h, dst.len(), ds)
    {
        skaler_devri_say();
        return skaler!(bd, w_avg(dst, ds, tmp1, tmp2, w, h, weight, bd, bdmax));
    }
    // `w > 7`: k = (w − 16)·2¹²; aksi hâlde girdiler yer değiştirir, k = −w·2¹².
    let g = if weight > 7 {
        Girdi {
            t1: tmp1,
            t2: tmp2,
            mask: &[],
            k: ((weight - 16) << 12) as i16,
        }
    } else {
        Girdi {
            t1: tmp2,
            t2: tmp1,
            mask: &[],
            k: (-weight << 12) as i16,
        }
    };
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`w_avg_fn`); belirteç AVX2 tespitinin kanıtıdır. (2) sınır:
    // `bilesik_sozlesmesi` `tmp` dilimlerinin `w·h` öğesini ve hedefi kapsar;
    // erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, girdiler `&`.
    unsafe { bidir_w::<W_AVG>(dst, ds, w, h, &g) }
}

fn mask_giris(
    dst: &mut [u8],
    ds: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &[u8],
    bd: BitDepth,
    bdmax: u16,
) {
    if bd != BitDepth::Eight
        || mask.len() < w * h
        || !bilesik_sozlesmesi(w, h, 4, tmp1.len(), tmp2.len(), w * h, dst.len(), ds)
    {
        skaler_devri_say();
        return skaler!(bd, mask(dst, ds, tmp1, tmp2, w, h, mask, bd, bdmax));
    }
    let g = Girdi {
        t1: tmp1,
        t2: tmp2,
        mask,
        k: 0,
    };
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`mask_fn`); belirteç AVX2 tespitinin kanıtıdır. (2) sınır:
    // `bilesik_sozlesmesi` ve `mask.len() ≥ w·h` okunan bütün öğeleri ve hedefi
    // kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`.
    unsafe { bidir_w::<MASK>(dst, ds, w, h, &g) }
}

fn w_mask_giris<const SUB: usize>(
    dst: &mut [u8],
    ds: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &mut [u8],
    sign: i32,
    ss_hor: bool,
    ss_ver: bool,
    bd: BitDepth,
    bdmax: u16,
) {
    let alt = (SUB != 0, SUB == 2);
    if bd != BitDepth::Eight
        || (ss_hor, ss_ver) != alt
        || !(0..=1).contains(&sign)
        || mask.len() < (w >> usize::from(ss_hor)) * (h >> usize::from(ss_ver))
        || !bilesik_sozlesmesi(w, h, 8, tmp1.len(), tmp2.len(), w * h, dst.len(), ds)
    {
        skaler_devri_say();
        return skaler!(
            bd,
            w_mask(
                dst, ds, tmp1, tmp2, w, h, mask, sign, ss_hor, ss_ver, bd, bdmax
            )
        );
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`w_mask_fn`); belirteç AVX2 tespitinin kanıtıdır. (2) sınır:
    // `bilesik_sozlesmesi` `tmp` öğelerini ve hedefi, maske denetimi alt türün
    // maske boyunu kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `dst`
    // ve `mask` ayrı `&mut`, girdiler `&`.
    unsafe { w_mask_w::<SUB>(dst, ds, tmp1, tmp2, w, h, mask, sign as i16) }
}

/// `avg` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn avg_fn(_avx2: Avx2Token) -> AvgFn {
    avg_giris
}

/// `w_avg` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn w_avg_fn(_avx2: Avx2Token) -> WAvgFn {
    w_avg_giris
}

/// `mask` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn mask_fn(_avx2: Avx2Token) -> MaskFn {
    mask_giris
}

/// `w_mask[alt]` AVX2 yuvası (yalnız belirteçle); `alt` 0: 4:4:4, 1: 4:2:2,
/// 2: 4:2:0.
#[must_use]
pub fn w_mask_fn(_avx2: Avx2Token, alt: usize) -> WMaskFn {
    match alt {
        0 => w_mask_giris::<0>,
        1 => w_mask_giris::<1>,
        _ => w_mask_giris::<2>,
    }
}

// ---------------------------------------------------------------------------
// P36: AVX512ICL girişleri (adres yalnız [`super::mc_bilesik_avx512`] belirteçli işaretçileriyle)
// ---------------------------------------------------------------------------
//
// Girişler bu modüldedir: dar bloklar bu modülün AVX2 gövdelerini, geniş bloklar
// `mc_bilesik_avx512`'in gövdelerini çağırır. Paylaşılan AVX2 gövdelerinin bütün çağırıcıları
// aynı kod üretim biriminde kalır ve iç bağlantıları korunur (başka birimden
// çağrılan tek kopya dış bağlantıya geçip LLVM'nin yerel işlev iyileştirmelerini
// yitiriyordu: AVX2 hücrelerinde +%10–20, P36 hız bütçesi tanı koşusu).

pub(super) fn avg_giris_icl(
    dst: &mut [u8],
    ds: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    if bd != BitDepth::Eight
        || !bilesik_sozlesmesi(w, h, 4, tmp1.len(), tmp2.len(), w * h, dst.len(), ds)
    {
        skaler_devri_say();
        return skaler!(bd, avg(dst, ds, tmp1, tmp2, w, h, bd, bdmax));
    }
    let g = Girdi {
        t1: tmp1,
        t2: tmp2,
        mask: &[],
        k: 0,
    };
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`mc_bilesik_avx512::avg_fn`); belirteç AVX512ICL (AVX2'yi kapsar) tespiti + maske izninin kanıtıdır. (2)
    // sınır: `bilesik_sozlesmesi` `tmp` dilimlerinin `w·h` öğesini ve hedefte `h`
    // satır × `w` baytı kapsar; gövde erişimleri dilim/dizi denetimlidir. (3)
    // hizalama/takma ad: `dst` tek `&mut`, girdiler `&`.
    unsafe {
        if w < 32 {
            bidir_w::<AVG>(dst, ds, w, h, &g)
        } else {
            super::mc_bilesik_avx512::bidir_w::<AVG>(dst, ds, w, h, &g)
        }
    }
}

pub(super) fn w_avg_giris_icl(
    dst: &mut [u8],
    ds: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
    bd: BitDepth,
    bdmax: u16,
) {
    if bd != BitDepth::Eight
        || !(0..=16).contains(&weight)
        || !bilesik_sozlesmesi(w, h, 4, tmp1.len(), tmp2.len(), w * h, dst.len(), ds)
    {
        skaler_devri_say();
        return skaler!(bd, w_avg(dst, ds, tmp1, tmp2, w, h, weight, bd, bdmax));
    }
    let g = if weight > 7 {
        Girdi {
            t1: tmp1,
            t2: tmp2,
            mask: &[],
            k: ((weight - 16) << 12) as i16,
        }
    } else {
        Girdi {
            t1: tmp2,
            t2: tmp1,
            mask: &[],
            k: (-weight << 12) as i16,
        }
    };
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`mc_bilesik_avx512::w_avg_fn`); belirteç AVX512ICL (AVX2'yi kapsar) tespitinin kanıtıdır. (2) sınır:
    // `bilesik_sozlesmesi` `tmp` dilimlerinin `w·h` öğesini ve hedefi kapsar;
    // erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, girdiler `&`.
    unsafe {
        if w < 32 {
            bidir_w::<W_AVG>(dst, ds, w, h, &g)
        } else {
            super::mc_bilesik_avx512::bidir_w::<W_AVG>(dst, ds, w, h, &g)
        }
    }
}

pub(super) fn mask_giris_icl(
    dst: &mut [u8],
    ds: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &[u8],
    bd: BitDepth,
    bdmax: u16,
) {
    if bd != BitDepth::Eight
        || mask.len() < w * h
        || !bilesik_sozlesmesi(w, h, 4, tmp1.len(), tmp2.len(), w * h, dst.len(), ds)
    {
        skaler_devri_say();
        return skaler!(bd, mask(dst, ds, tmp1, tmp2, w, h, mask, bd, bdmax));
    }
    let g = Girdi {
        t1: tmp1,
        t2: tmp2,
        mask,
        k: 0,
    };
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`mc_bilesik_avx512::mask_fn`); belirteç AVX512ICL (AVX2'yi kapsar) tespitinin kanıtıdır. (2) sınır:
    // `bilesik_sozlesmesi` ve `mask.len() ≥ w·h` okunan bütün öğeleri ve hedefi
    // kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`.
    unsafe {
        if w < 32 {
            bidir_w::<MASK>(dst, ds, w, h, &g)
        } else {
            super::mc_bilesik_avx512::bidir_w::<MASK>(dst, ds, w, h, &g)
        }
    }
}

pub(super) fn w_mask_giris_icl<const SUB: usize>(
    dst: &mut [u8],
    ds: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &mut [u8],
    sign: i32,
    ss_hor: bool,
    ss_ver: bool,
    bd: BitDepth,
    bdmax: u16,
) {
    let alt = (SUB != 0, SUB == 2);
    if bd != BitDepth::Eight
        || (ss_hor, ss_ver) != alt
        || !(0..=1).contains(&sign)
        || mask.len() < (w >> usize::from(ss_hor)) * (h >> usize::from(ss_ver))
        || !bilesik_sozlesmesi(w, h, 8, tmp1.len(), tmp2.len(), w * h, dst.len(), ds)
    {
        skaler_devri_say();
        return skaler!(
            bd,
            w_mask(
                dst, ds, tmp1, tmp2, w, h, mask, sign, ss_hor, ss_ver, bd, bdmax
            )
        );
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`mc_bilesik_avx512::w_mask_fn`); belirteç AVX512ICL (AVX2'yi kapsar) tespitinin kanıtıdır. (2) sınır:
    // `bilesik_sozlesmesi` `tmp` öğelerini ve hedefi, maske denetimi alt türün
    // maske boyunu kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `dst`
    // ve `mask` ayrı `&mut`, girdiler `&`.
    unsafe {
        if w < 32 {
            w_mask_w::<SUB>(dst, ds, tmp1, tmp2, w, h, mask, sign as i16)
        } else {
            super::mc_bilesik_avx512::w_mask_w::<SUB>(dst, ds, tmp1, tmp2, w, h, mask, sign as i16)
        }
    }
}
