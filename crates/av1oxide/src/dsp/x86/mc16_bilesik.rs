//! MC bileşik AVX2 çekirdekleri (16 bpc, 10/12 bit) — F4-MC c (P34): `avg`,
//! `w_avg`, `mask`, `w_mask[0..2]` (4:4:4, 4:2:2, 4:2:0).
//!
//! ## Sözleşme
//!
//! Girişte O(1) denetim ([`crate::dsp::mc::bilesik16_sozlesmesi`]): bit
//! derinliği 10/12 ve `bitdepth_max` tutarlı; `w`, `h` 2'nin kuvveti
//! (`avg`/`w_avg`/`mask` 4…128, `w_mask` 8…128); `tmp1`/`tmp2` (ve `mask`) en
//! az `w·h` öğe; hedef satır adımı `≥ 2w` bayt, `dst` en az `(h−1)·ds + 2w`
//! bayt; `w_avg` ağırlığı 0…16; `w_mask`'ta `sign` 0/1, `(ss_hor, ss_ver)`
//! yuvanın alt türüyle aynı ve maske en az `(w >> ss_hor)·(h >> ss_ver)` bayt.
//! Sözleşme dışı girdi skalere devreder ve sayılır. Okuma tam `w·h` öğedir,
//! yazma yalnız `h` satır × `2w` bayt (+ maske).
//!
//! **Değer tanım kümesi varsayımı yok:** 16 bpc `prep` çıktısı `i16`'nın
//! neredeyse tamamını kullanır (10 bit `[−28780, 28764]`), `a + b` `i16`'ya
//! sığmaz. Ağırlıklı toplam `pmaddwd` ile `i32`'de tam hesaplanır: her `i16`
//! girdi ve her `u8` maske değeri için skalerle bit-exact.
//!
//! ## Yöntem
//!
//! - İki girdi `punpck{l,h}wd` ile ikililenir ve katsayı ikilisiyle `pmaddwd`:
//!   `avg` `(1, 1)`, `w_avg` `(w, 16−w)`, `mask` `(m, 64−m)` → `i32` toplam;
//!   `+ rnd`, `psrad` (sayı çağrı başına yazmaçta), `packusdw` +
//!   `pminuw(bitdepth_max)` = `iclip_pixel`. `punpck`/`pack` çifti şerit
//!   düzenini korur: bir adım 16 sıralı piksel (32 bayt).
//! - Sabitler (dav1d `mc_tmpl.c`; `ib` 10 bitte 4, 12 bitte 2; `PREP_BIAS`
//!   8192): `avg` `>> (ib+1)`, `rnd = 2^ib + 2·8192`; `w_avg` `>> (ib+4)`,
//!   `rnd = 8·2^ib + 16·8192`; `mask`/`w_mask` `>> (ib+6)`,
//!   `rnd = 32·2^ib + 64·8192`.
//! - `w_mask`: `m = min(38 + ((|t₁−t₂| + 32) >> 10), 64)` (`mask_sh =
//!   bpc + ib − 4` iki bit derinliğinde 10). Fark doyurmalı `psubsw` ile
//!   alınır: `|t₁−t₂| ≥ 32767`'de gerçek ve doymuş fark ikisi de `m = 64`
//!   verir. Maske baytları 8 bpc gövdesinin yolundan: 32 öğe `packuswb` +
//!   `vpermq`, 4:2:2 ikili toplam `pmaddubsw`, 4:2:0 iki satırın ikili toplamı.
//!
//! `unsafe` yalnız belirteçli girişlerdeki `#[target_feature]` çağrısıdır;
//! dosya `audit::UNSAFE_EXEMPTIONS` defterinde kayıtlıdır. Yükleme/saklama
//! [`super::mem`] yardımcılarıyla yapılır.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mem::{chunk, chunk_mut, load_i16x16, load_u8x16, store_u8x8, store_u8x16, store_u8x32};
use crate::cpu::Avx2Token;
use crate::dsp::mc::{AvgFn, MaskFn, WAvgFn, WMaskFn, bilesik16_sozlesmesi, skaler_devri_say};
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_cvtsi32_si128, _mm_packus_epi16, _mm_set1_epi16,
    _mm_srli_epi16, _mm_srli_si128, _mm_unpacklo_epi64, _mm256_abs_epi16, _mm256_add_epi16,
    _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepu8_epi16, _mm256_extracti128_si256,
    _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_min_epi16, _mm256_min_epu16,
    _mm256_packus_epi16, _mm256_packus_epi32, _mm256_permute4x64_epi64, _mm256_set1_epi8,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_sra_epi32, _mm256_srli_epi16, _mm256_sub_epi16,
    _mm256_subs_epi16, _mm256_unpackhi_epi16, _mm256_unpacklo_epi16,
};

/// `prep` ara değerinden çıkarılan sabit (dav1d `PREP_BIAS`, 16 bpc).
const PREP_BIAS: i32 = 8192;

/// Bileşik işlem (derleme zamanı).
pub(super) const AVG: u8 = 0;
pub(super) const W_AVG: u8 = 1;
pub(super) const MASK: u8 = 2;

/// Çağrı sabitleri ve girdiler.
struct Sabit<'a> {
    t1: &'a [i16],
    t2: &'a [i16],
    mask: &'a [u8],
    /// `avg` `(1, 1)`, `w_avg` `(w, 16−w)` katsayı ikilisi (`i32` şeritte).
    kat: __m256i,
    rnd: __m256i,
    sh: __m128i,
    max: __m256i,
}

impl<'a> Sabit<'a> {
    /// İşlem `op` için yuvarlama ve kaydırma (`ib` `bitdepth_max`'tan).
    #[target_feature(enable = "avx2")]
    fn yeni(
        op: u8,
        t1: &'a [i16],
        t2: &'a [i16],
        mask: &'a [u8],
        kat: i32,
        bitdepth_max: u16,
    ) -> Self {
        let ib = if bitdepth_max == 4095 { 2 } else { 4 };
        let (sh, rnd) = match op {
            AVG => (ib + 1, (1 << ib) + 2 * PREP_BIAS),
            W_AVG => (ib + 4, (8 << ib) + 16 * PREP_BIAS),
            _ => (ib + 6, (32 << ib) + 64 * PREP_BIAS),
        };
        Sabit {
            t1,
            t2,
            mask,
            kat: _mm256_set1_epi32(kat),
            rnd: _mm256_set1_epi32(rnd),
            sh: _mm_cvtsi32_si128(sh),
            max: _mm256_set1_epi16(bitdepth_max as i16),
        }
    }
}

/// `i32` ağırlıklı toplamlardan (şerit düzeni `[0..3 | 8..11]`,
/// `[4..7 | 12..15]`) 16 sıralı kırpılmış piksel.
#[target_feature(enable = "avx2")]
#[inline]
fn bitir(lo: __m256i, hi: __m256i, s: &Sabit<'_>) -> __m256i {
    let lo = _mm256_sra_epi32(_mm256_add_epi32(lo, s.rnd), s.sh);
    let hi = _mm256_sra_epi32(_mm256_add_epi32(hi, s.rnd), s.sh);
    _mm256_min_epu16(_mm256_packus_epi32(lo, hi), s.max)
}

/// Öğe ofseti `o`'dan 16 öğenin sonucu.
#[target_feature(enable = "avx2")]
#[inline]
fn hesap16<const OP: u8>(s: &Sabit<'_>, o: usize) -> __m256i {
    let a = load_i16x16(chunk(s.t1, o));
    let b = load_i16x16(chunk(s.t2, o));
    let (klo, khi) = if OP == MASK {
        let m = _mm256_cvtepu8_epi16(load_u8x16(chunk(s.mask, o)));
        let n = _mm256_sub_epi16(_mm256_set1_epi16(64), m);
        (_mm256_unpacklo_epi16(m, n), _mm256_unpackhi_epi16(m, n))
    } else {
        (s.kat, s.kat)
    };
    let lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), klo);
    let hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), khi);
    bitir(lo, hi, s)
}

/// Öğe ofseti `o`'dan başlayan 16 sıralı pikseli (`W` sütunlu bitişik
/// satırlar) hedefe yazar.
#[target_feature(enable = "avx2")]
#[inline]
fn sakla16<const W: usize>(dst: &mut [u8], ds: usize, o: usize, p: __m256i) {
    let (y, x) = (o / W, o % W);
    let lo = _mm256_castsi256_si128(p);
    let hi = _mm256_extracti128_si256::<1>(p);
    match W {
        4 => {
            store_u8x8(chunk_mut(dst, y * ds), lo);
            store_u8x8(chunk_mut(dst, (y + 1) * ds), _mm_srli_si128::<8>(lo));
            store_u8x8(chunk_mut(dst, (y + 2) * ds), hi);
            store_u8x8(chunk_mut(dst, (y + 3) * ds), _mm_srli_si128::<8>(hi));
        }
        8 => {
            store_u8x16(chunk_mut(dst, y * ds), lo);
            store_u8x16(chunk_mut(dst, (y + 1) * ds), hi);
        }
        _ => store_u8x32(chunk_mut(dst, y * ds + 2 * x), p),
    }
}

/// `avg` / `w_avg` / `mask` gövdesi: `W` sütun, `n = W·h` öğe.
#[target_feature(enable = "avx2")]
fn bidir<const OP: u8, const W: usize>(dst: &mut [u8], ds: usize, n: usize, s: &Sabit<'_>) {
    let mut o = 0;
    while o < n {
        sakla16::<W>(dst, ds, o, hesap16::<OP>(s, o));
        o += 16;
    }
}

/// Genişliğe göre `bidir` monomorfu.
#[target_feature(enable = "avx2")]
fn bidir_w<const OP: u8>(dst: &mut [u8], ds: usize, w: usize, h: usize, s: &Sabit<'_>) {
    let n = w * h;
    match w {
        4 => bidir::<OP, 4>(dst, ds, n, s),
        8 => bidir::<OP, 8>(dst, ds, n, s),
        16 => bidir::<OP, 16>(dst, ds, n, s),
        32 => bidir::<OP, 32>(dst, ds, n, s),
        64 => bidir::<OP, 64>(dst, ds, n, s),
        _ => bidir::<OP, 128>(dst, ds, n, s),
    }
}

/// Giriş: `avg`/`w_avg`/`mask` sabitleri ve gövde.
#[target_feature(enable = "avx2")]
fn bidir_giris<const OP: u8>(
    dst: &mut [u8],
    ds: usize,
    t1: &[i16],
    t2: &[i16],
    mask: &[u8],
    w: usize,
    h: usize,
    kat: i32,
    bitdepth_max: u16,
) {
    let s = Sabit::yeni(OP, t1, t2, mask, kat, bitdepth_max);
    bidir_w::<OP>(dst, ds, w, h, &s);
}

/// `w_mask` 16 öğesi: (sıralı piksel sonucu, `m` `i16`).
#[target_feature(enable = "avx2")]
#[inline]
fn wm16(s: &Sabit<'_>, o: usize) -> (__m256i, __m256i) {
    let a = load_i16x16(chunk(s.t1, o));
    let b = load_i16x16(chunk(s.t2, o));
    // |t₁ − t₂| işaretsiz (doymuş fark; `abs(−32768)` = 0x8000).
    let fark = _mm256_abs_epi16(_mm256_subs_epi16(a, b));
    let q = _mm256_srli_epi16::<10>(_mm256_add_epi16(fark, _mm256_set1_epi16(32)));
    let m = _mm256_min_epi16(
        _mm256_add_epi16(q, _mm256_set1_epi16(38)),
        _mm256_set1_epi16(64),
    );
    let n = _mm256_sub_epi16(_mm256_set1_epi16(64), m);
    let lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), _mm256_unpacklo_epi16(m, n));
    let hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), _mm256_unpackhi_epi16(m, n));
    (bitir(lo, hi, s), m)
}

/// 32 öğenin `m` baytları (sıralı) ve 16 ikili toplamı (`i16`, sıralı).
#[target_feature(enable = "avx2")]
#[inline]
fn m_bayt_ve_ikili(m0: __m256i, m1: __m256i) -> (__m256i, __m256i) {
    let m = _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi16(m0, m1));
    (m, _mm256_maddubs_epi16(m, _mm256_set1_epi8(1)))
}

/// `w_mask` gövdesi: alt tür `SUB` (0: 4:4:4, 1: 4:2:2, 2: 4:2:0), `W` sütun.
#[target_feature(enable = "avx2")]
fn w_mask_govde<const SUB: usize, const W: usize>(
    dst: &mut [u8],
    ds: usize,
    h: usize,
    mask: &mut [u8],
    sign: i16,
    s: &Sabit<'_>,
) {
    let n = W * h;
    if SUB < 2 {
        // 4:4:4 / 4:2:2: öğe sırasıyla; maske satırları da bitişik.
        let ek = _mm256_set1_epi16(1 - sign);
        let mut o = 0;
        while o < n {
            let (r0, m0) = wm16(s, o);
            let (r1, m1) = wm16(s, o + 16);
            sakla16::<W>(dst, ds, o, r0);
            sakla16::<W>(dst, ds, o + 16, r1);
            let (m, ikili) = m_bayt_ve_ikili(m0, m1);
            if SUB == 0 {
                store_u8x32(chunk_mut(mask, o), m);
            } else {
                let v = _mm256_srli_epi16::<1>(_mm256_add_epi16(ikili, ek));
                let p = _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi16(v, v));
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
            let (r0, m0) = wm16(s, o);
            let (r1, m1) = wm16(s, o + 16);
            let (r2, m2) = wm16(s, o1);
            let (r3, m3) = wm16(s, o1 + 16);
            sakla16::<W>(dst, ds, o, r0);
            sakla16::<W>(dst, ds, o + 16, r1);
            sakla16::<W>(dst, ds, o1, r2);
            sakla16::<W>(dst, ds, o1 + 16, r3);
            let (_, s0) = m_bayt_ve_ikili(m0, m1);
            let (_, s1) = m_bayt_ve_ikili(m2, m3);
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
            let (r0, m0) = wm16(s, o);
            let (r1, m1) = wm16(s, o + 16);
            sakla16::<W>(dst, ds, o, r0);
            sakla16::<W>(dst, ds, o + 16, r1);
            let (_, t) = m_bayt_ve_ikili(m0, m1);
            let lo = _mm256_castsi256_si128(t);
            let hi = _mm256_extracti128_si256::<1>(t);
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
            store_u8x8(chunk_mut(mask, o / 4), _mm_packus_epi16(v, v));
            o += 32;
        }
    }
}

/// `w_mask` sabitleri ve genişlik monomorfu.
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
    bitdepth_max: u16,
) {
    let s = Sabit::yeni(MASK, t1, t2, &[], 0, bitdepth_max);
    match w {
        8 => w_mask_govde::<SUB, 8>(dst, ds, h, mask, sign, &s),
        16 => w_mask_govde::<SUB, 16>(dst, ds, h, mask, sign, &s),
        32 => w_mask_govde::<SUB, 32>(dst, ds, h, mask, sign, &s),
        64 => w_mask_govde::<SUB, 64>(dst, ds, h, mask, sign, &s),
        _ => w_mask_govde::<SUB, 128>(dst, ds, h, mask, sign, &s),
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

fn avg16_giris(
    dst: &mut [u8],
    ds: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    if !bilesik16_sozlesmesi(w, h, 4, tmp1.len(), tmp2.len(), dst.len(), ds, bd, bdmax) {
        skaler_devri_say();
        return skaler!(bd, avg(dst, ds, tmp1, tmp2, w, h, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`avg_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `bilesik16_sozlesmesi` `tmp` dilimlerinin `w·h` öğesini ve hedefte `h` satır
    // × `2w` baytı kapsar; gövde erişimleri dilim/dizi denetimlidir. (3)
    // hizalama/takma ad: `dst` tek `&mut`, girdiler `&`; ödünç kuralları çakışmayı dışlar.
    unsafe { bidir_giris::<AVG>(dst, ds, tmp1, tmp2, &[], w, h, 0x0001_0001, bdmax) }
}

fn w_avg16_giris(
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
    if !(0..=16).contains(&weight)
        || !bilesik16_sozlesmesi(w, h, 4, tmp1.len(), tmp2.len(), dst.len(), ds, bd, bdmax)
    {
        skaler_devri_say();
        return skaler!(bd, w_avg(dst, ds, tmp1, tmp2, w, h, weight, bd, bdmax));
    }
    // Katsayı ikilisi: düşük yarı `tmp1`'in `w`'si, yüksek yarı `tmp2`'nin `16 − w`'si.
    let kat = weight | ((16 - weight) << 16);
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`w_avg_fn`); belirteç AVX2 tespitinin kanıtıdır. (2) sınır:
    // `bilesik16_sozlesmesi` `tmp` dilimlerinin `w·h` öğesini ve hedefi kapsar;
    // erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, girdiler `&`.
    unsafe { bidir_giris::<W_AVG>(dst, ds, tmp1, tmp2, &[], w, h, kat, bdmax) }
}

fn mask16_giris(
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
    if mask.len() < w * h
        || !bilesik16_sozlesmesi(w, h, 4, tmp1.len(), tmp2.len(), dst.len(), ds, bd, bdmax)
    {
        skaler_devri_say();
        return skaler!(bd, mask(dst, ds, tmp1, tmp2, w, h, mask, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`mask_fn`); belirteç AVX2 tespitinin kanıtıdır. (2) sınır:
    // `bilesik16_sozlesmesi` ve `mask.len() ≥ w·h` okunan bütün öğeleri ve hedefi
    // kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`.
    unsafe { bidir_giris::<MASK>(dst, ds, tmp1, tmp2, mask, w, h, 0, bdmax) }
}

fn w_mask16_giris<const SUB: usize>(
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
    if (ss_hor, ss_ver) != alt
        || !(0..=1).contains(&sign)
        || mask.len() < (w >> usize::from(ss_hor)) * (h >> usize::from(ss_ver))
        || !bilesik16_sozlesmesi(w, h, 8, tmp1.len(), tmp2.len(), dst.len(), ds, bd, bdmax)
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
    // `bilesik16_sozlesmesi` `tmp` öğelerini ve hedefi, maske denetimi alt türün
    // maske boyunu kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `dst`
    // ve `mask` ayrı `&mut`, girdiler `&`.
    unsafe { w_mask_w::<SUB>(dst, ds, tmp1, tmp2, w, h, mask, sign as i16, bdmax) }
}

/// 16 bpc `avg` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn avg_fn(_avx2: Avx2Token) -> AvgFn {
    avg16_giris
}

/// 16 bpc `w_avg` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn w_avg_fn(_avx2: Avx2Token) -> WAvgFn {
    w_avg16_giris
}

/// 16 bpc `mask` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn mask_fn(_avx2: Avx2Token) -> MaskFn {
    mask16_giris
}

/// 16 bpc `w_mask[alt]` AVX2 yuvası (yalnız belirteçle); `alt` 0: 4:4:4,
/// 1: 4:2:2, 2: 4:2:0.
#[must_use]
pub fn w_mask_fn(_avx2: Avx2Token, alt: usize) -> WMaskFn {
    match alt {
        0 => w_mask16_giris::<0>,
        1 => w_mask16_giris::<1>,
        _ => w_mask16_giris::<2>,
    }
}

// ---------------------------------------------------------------------------
// P36: AVX512ICL girişleri (adres yalnız [`super::mc16_bilesik_avx512`] belirteçli işaretçileriyle)
// ---------------------------------------------------------------------------
//
// Girişler bu modüldedir: dar bloklar bu modülün AVX2 gövdelerini, geniş bloklar
// `mc16_bilesik_avx512`'in gövdelerini çağırır. Paylaşılan AVX2 gövdelerinin bütün çağırıcıları
// aynı kod üretim biriminde kalır ve iç bağlantıları korunur (başka birimden
// çağrılan tek kopya dış bağlantıya geçip LLVM'nin yerel işlev iyileştirmelerini
// yitiriyordu: AVX2 hücrelerinde +%10–20, P36 hız bütçesi tanı koşusu).

pub(super) fn avg16_giris_icl(
    dst: &mut [u8],
    ds: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    if !bilesik16_sozlesmesi(w, h, 4, tmp1.len(), tmp2.len(), dst.len(), ds, bd, bdmax) {
        skaler_devri_say();
        return skaler!(bd, avg(dst, ds, tmp1, tmp2, w, h, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`mc16_bilesik_avx512::avg_fn`); belirteç AVX512ICL (AVX2'yi kapsar) tespiti + maske izninin kanıtıdır. (2)
    // sınır: `bilesik16_sozlesmesi` `tmp` dilimlerinin `w·h` öğesini ve hedefte
    // `h` satır × `2w` baytı kapsar; gövde erişimleri dilim/dizi denetimlidir.
    // (3) hizalama/takma ad: `dst` tek `&mut`, girdiler `&`.
    unsafe {
        if w < 16 {
            bidir_giris::<AVG>(dst, ds, tmp1, tmp2, &[], w, h, 0x0001_0001, bdmax)
        } else {
            super::mc16_bilesik_avx512::bidir_giris::<AVG>(
                dst,
                ds,
                tmp1,
                tmp2,
                &[],
                w,
                h,
                0x0001_0001,
                bdmax,
            )
        }
    }
}

pub(super) fn w_avg16_giris_icl(
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
    if !(0..=16).contains(&weight)
        || !bilesik16_sozlesmesi(w, h, 4, tmp1.len(), tmp2.len(), dst.len(), ds, bd, bdmax)
    {
        skaler_devri_say();
        return skaler!(bd, w_avg(dst, ds, tmp1, tmp2, w, h, weight, bd, bdmax));
    }
    let kat = weight | ((16 - weight) << 16);
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`mc16_bilesik_avx512::w_avg_fn`); belirteç AVX512ICL (AVX2'yi kapsar) tespitinin kanıtıdır. (2) sınır:
    // `bilesik16_sozlesmesi` `tmp` dilimlerinin `w·h` öğesini ve hedefi kapsar;
    // erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, girdiler `&`.
    unsafe {
        if w < 16 {
            bidir_giris::<W_AVG>(dst, ds, tmp1, tmp2, &[], w, h, kat, bdmax)
        } else {
            super::mc16_bilesik_avx512::bidir_giris::<W_AVG>(
                dst,
                ds,
                tmp1,
                tmp2,
                &[],
                w,
                h,
                kat,
                bdmax,
            )
        }
    }
}

pub(super) fn mask16_giris_icl(
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
    if mask.len() < w * h
        || !bilesik16_sozlesmesi(w, h, 4, tmp1.len(), tmp2.len(), dst.len(), ds, bd, bdmax)
    {
        skaler_devri_say();
        return skaler!(bd, mask(dst, ds, tmp1, tmp2, w, h, mask, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`mc16_bilesik_avx512::mask_fn`); belirteç AVX512ICL (AVX2'yi kapsar) tespitinin kanıtıdır. (2) sınır:
    // `bilesik16_sozlesmesi` ve `mask.len() ≥ w·h` okunan bütün öğeleri ve hedefi
    // kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`.
    unsafe {
        if w < 16 {
            bidir_giris::<MASK>(dst, ds, tmp1, tmp2, mask, w, h, 0, bdmax)
        } else {
            super::mc16_bilesik_avx512::bidir_giris::<MASK>(
                dst, ds, tmp1, tmp2, mask, w, h, 0, bdmax,
            )
        }
    }
}

pub(super) fn w_mask16_giris_icl<const SUB: usize>(
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
    if (ss_hor, ss_ver) != alt
        || !(0..=1).contains(&sign)
        || mask.len() < (w >> usize::from(ss_hor)) * (h >> usize::from(ss_ver))
        || !bilesik16_sozlesmesi(w, h, 8, tmp1.len(), tmp2.len(), dst.len(), ds, bd, bdmax)
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
    // (`mc16_bilesik_avx512::w_mask_fn`); belirteç AVX512ICL (AVX2'yi kapsar) tespitinin kanıtıdır. (2) sınır:
    // `bilesik16_sozlesmesi` `tmp` öğelerini ve hedefi, maske denetimi alt türün
    // maske boyunu kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `dst`
    // ve `mask` ayrı `&mut`, girdiler `&`.
    unsafe {
        if w < 32 {
            w_mask_w::<SUB>(dst, ds, tmp1, tmp2, w, h, mask, sign as i16, bdmax)
        } else {
            super::mc16_bilesik_avx512::w_mask_w::<SUB>(
                dst,
                ds,
                tmp1,
                tmp2,
                w,
                h,
                mask,
                sign as i16,
                bdmax,
            )
        }
    }
}
