//! MC bileşik SSSE3 çekirdekleri (16 bpc, 10/12 bit) — F4-MC d (P35): `avg`,
//! `w_avg`, `mask`, `w_mask[0..2]` (4:4:4, 4:2:2, 4:2:0).
//!
//! ## Sözleşme
//!
//! AVX2 gövdeleriyle ([`super::mc16_bilesik`]) aynı: girişte O(1) denetim
//! ([`crate::dsp::mc::bilesik16_sozlesmesi`]); sözleşme dışı girdi skalere
//! devreder ve sayılır. Okuma tam `w·h` öğedir, yazma yalnız `h` satır × `2w`
//! bayt (+ maske). Değer tanım kümesi varsayımı yoktur: ağırlıklı toplam
//! `pmaddwd` ile `i32`'de tam hesaplanır.
//!
//! ## Yöntem
//!
//! Aritmetik AVX2 gövdesiyle aynıdır (`punpck{l,h}wd` + `pmaddwd`, `+ rnd`,
//! `psrad`); yazmaç 128 bittir ve bir adım 16 öğedir (iki 8 piksellik
//! yazmaç). Kırpma `packssdw` + `pmaxsw(0)` + `pminsw(bitdepth_max)`'tir
//! (`packusdw`/`pminuw` SSE4.1 olduğundan kullanılmaz; `i32` sonuç `> 32767`
//! iken iki yol da `bitdepth_max`, `< 0` iken 0 verir). `mask` baytları
//! `punpck{l,h}bw` sıfırla genişler. `w_mask` maskesi 8 bpc SSSE3 gövdesinin
//! yolundan: 16 öğe `packuswb` (sıralı), 4:2:2 ikili toplam `pmaddubsw`,
//! 4:2:0 iki satırın ikili toplamı. SSE4.1 ve üstü komut yoktur.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mem::{chunk, chunk_mut, load_i16x8, load_u8x16, store_u8x4, store_u8x8, store_u8x16};
use crate::cpu::Ssse3Token;
use crate::dsp::mc::{AvgFn, MaskFn, WAvgFn, WMaskFn, bilesik16_sozlesmesi, skaler_devri_say};
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, _mm_abs_epi16, _mm_add_epi16, _mm_add_epi32, _mm_cvtsi32_si128, _mm_madd_epi16,
    _mm_maddubs_epi16, _mm_max_epi16, _mm_min_epi16, _mm_packs_epi32, _mm_packus_epi16,
    _mm_set1_epi8, _mm_set1_epi16, _mm_set1_epi32, _mm_setzero_si128, _mm_sra_epi32,
    _mm_srli_epi16, _mm_srli_si128, _mm_sub_epi16, _mm_subs_epi16, _mm_unpackhi_epi8,
    _mm_unpackhi_epi16, _mm_unpacklo_epi8, _mm_unpacklo_epi16,
};

/// `prep` ara değerinden çıkarılan sabit (dav1d `PREP_BIAS`, 16 bpc).
const PREP_BIAS: i32 = 8192;

/// Bileşik işlem (derleme zamanı).
const AVG: u8 = 0;
const W_AVG: u8 = 1;
const MASK: u8 = 2;

/// Çağrı sabitleri ve girdiler.
struct Sabit<'a> {
    t1: &'a [i16],
    t2: &'a [i16],
    mask: &'a [u8],
    /// `avg` `(1, 1)`, `w_avg` `(w, 16−w)` katsayı ikilisi (`i32` şeritte).
    kat: __m128i,
    rnd: __m128i,
    sh: __m128i,
    max: __m128i,
}

impl<'a> Sabit<'a> {
    /// İşlem `op` için yuvarlama ve kaydırma (`ib` `bitdepth_max`'tan).
    #[target_feature(enable = "ssse3")]
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
            kat: _mm_set1_epi32(kat),
            rnd: _mm_set1_epi32(rnd),
            sh: _mm_cvtsi32_si128(sh),
            max: _mm_set1_epi16(bitdepth_max as i16),
        }
    }
}

/// `i32` ağırlıklı toplamlardan (öğe `0..3`, `4..7`) 8 sıralı kırpılmış piksel.
#[target_feature(enable = "ssse3")]
#[inline]
fn bitir(lo: __m128i, hi: __m128i, s: &Sabit<'_>) -> __m128i {
    let lo = _mm_sra_epi32(_mm_add_epi32(lo, s.rnd), s.sh);
    let hi = _mm_sra_epi32(_mm_add_epi32(hi, s.rnd), s.sh);
    _mm_min_epi16(
        _mm_max_epi16(_mm_packs_epi32(lo, hi), _mm_setzero_si128()),
        s.max,
    )
}

/// Öğe ofseti `o`'dan 8 öğenin sonucu; `m` `mask`'ta 8 maske değeri (`i16`).
#[target_feature(enable = "ssse3")]
#[inline]
fn hesap8<const OP: u8>(s: &Sabit<'_>, o: usize, m: __m128i) -> __m128i {
    let a = load_i16x8(chunk(s.t1, o));
    let b = load_i16x8(chunk(s.t2, o));
    let (klo, khi) = if OP == MASK {
        let n = _mm_sub_epi16(_mm_set1_epi16(64), m);
        (_mm_unpacklo_epi16(m, n), _mm_unpackhi_epi16(m, n))
    } else {
        (s.kat, s.kat)
    };
    let lo = _mm_madd_epi16(_mm_unpacklo_epi16(a, b), klo);
    let hi = _mm_madd_epi16(_mm_unpackhi_epi16(a, b), khi);
    bitir(lo, hi, s)
}

/// Öğe ofseti `o`'dan başlayan 8 sıralı pikseli (`W` sütunlu bitişik
/// satırlar) hedefe yazar.
#[target_feature(enable = "ssse3")]
#[inline]
fn sakla8<const W: usize>(dst: &mut [u8], ds: usize, o: usize, p: __m128i) {
    let (y, x) = (o / W, o % W);
    if W == 4 {
        store_u8x8(chunk_mut(dst, y * ds), p);
        store_u8x8(chunk_mut(dst, (y + 1) * ds), _mm_srli_si128::<8>(p));
    } else {
        store_u8x16(chunk_mut(dst, y * ds + 2 * x), p);
    }
}

/// `avg` / `w_avg` / `mask` gövdesi: `W` sütun, `n = W·h` öğe; adım 16 öğe.
#[target_feature(enable = "ssse3")]
fn bidir<const OP: u8, const W: usize>(dst: &mut [u8], ds: usize, n: usize, s: &Sabit<'_>) {
    let mut o = 0;
    while o < n {
        let (m0, m1) = if OP == MASK {
            let mb = load_u8x16(chunk(s.mask, o));
            let z = _mm_setzero_si128();
            (_mm_unpacklo_epi8(mb, z), _mm_unpackhi_epi8(mb, z))
        } else {
            (_mm_setzero_si128(), _mm_setzero_si128())
        };
        sakla8::<W>(dst, ds, o, hesap8::<OP>(s, o, m0));
        sakla8::<W>(dst, ds, o + 8, hesap8::<OP>(s, o + 8, m1));
        o += 16;
    }
}

/// Genişliğe göre `bidir` monomorfu.
#[target_feature(enable = "ssse3")]
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
#[target_feature(enable = "ssse3")]
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

/// `w_mask` 8 öğesi: (sıralı piksel sonucu, `m` `i16`).
#[target_feature(enable = "ssse3")]
#[inline]
fn wm8(s: &Sabit<'_>, o: usize) -> (__m128i, __m128i) {
    let a = load_i16x8(chunk(s.t1, o));
    let b = load_i16x8(chunk(s.t2, o));
    // |t₁ − t₂| işaretsiz (doymuş fark; `abs(−32768)` = 0x8000).
    let fark = _mm_abs_epi16(_mm_subs_epi16(a, b));
    let q = _mm_srli_epi16::<10>(_mm_add_epi16(fark, _mm_set1_epi16(32)));
    let m = _mm_min_epi16(_mm_add_epi16(q, _mm_set1_epi16(38)), _mm_set1_epi16(64));
    let n = _mm_sub_epi16(_mm_set1_epi16(64), m);
    let lo = _mm_madd_epi16(_mm_unpacklo_epi16(a, b), _mm_unpacklo_epi16(m, n));
    let hi = _mm_madd_epi16(_mm_unpackhi_epi16(a, b), _mm_unpackhi_epi16(m, n));
    (bitir(lo, hi, s), m)
}

/// 16 öğe: pikselleri hedefe yazar; (sıralı 16 `m` baytı, 8 ikili toplam
/// `i16`). **Makro:** gövde çağıranın döngüsüne açılır (işlev olarak satır
/// içine alınmıyor ve 16 öğe başına çağrıya dönüşüyordu).
macro_rules! wm16 {
    ($w:expr, $dst:expr, $ds:expr, $o:expr, $s:expr) => {{
        let (o, s) = ($o, $s);
        let (r0, m0) = wm8(s, o);
        let (r1, m1) = wm8(s, o + 8);
        sakla8::<{ $w }>($dst, $ds, o, r0);
        sakla8::<{ $w }>($dst, $ds, o + 8, r1);
        let m = _mm_packus_epi16(m0, m1);
        (m, _mm_maddubs_epi16(m, _mm_set1_epi8(1)))
    }};
}

/// `w_mask` gövdesi: alt tür `SUB` (0: 4:4:4, 1: 4:2:2, 2: 4:2:0), `W` sütun.
#[target_feature(enable = "ssse3")]
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
        let ek = _mm_set1_epi16(1 - sign);
        let mut o = 0;
        while o < n {
            let (m, ikili) = wm16!(W, dst, ds, o, s);
            if SUB == 0 {
                store_u8x16(chunk_mut(mask, o), m);
            } else {
                let v = _mm_srli_epi16::<1>(_mm_add_epi16(ikili, ek));
                store_u8x8(chunk_mut(mask, o / 2), _mm_packus_epi16(v, v));
            }
            o += 16;
        }
        return;
    }
    // 4:2:0: satır ikilileri; maske satırı `W/2` bayt.
    let ek = _mm_set1_epi16(2 - sign);
    let mut o = 0;
    while o < n {
        if W >= 16 {
            // Satır `y` ve `y+1`'in aynı 16 sütunu.
            let (y, x) = (o / W, o % W);
            let (_, s0) = wm16!(W, dst, ds, o, s);
            let (_, s1) = wm16!(W, dst, ds, o + W, s);
            let v = _mm_srli_epi16::<2>(_mm_add_epi16(_mm_add_epi16(s0, s1), ek));
            store_u8x8(
                chunk_mut(mask, (y / 2) * (W / 2) + x / 2),
                _mm_packus_epi16(v, v),
            );
            o += if x + 16 >= W { W + 16 } else { 16 };
        } else {
            // `W = 8`: 16 öğe = satır ikilisi; ikili toplamların alt yarısı
            // satır `y`, üst yarısı satır `y+1`.
            let (_, t) = wm16!(W, dst, ds, o, s);
            let v = _mm_add_epi16(t, _mm_srli_si128::<8>(t));
            let v = _mm_srli_epi16::<2>(_mm_add_epi16(v, ek));
            store_u8x4(chunk_mut(mask, o / 4), _mm_packus_epi16(v, v));
            o += 16;
        }
    }
}

/// `w_mask` sabitleri ve genişlik monomorfu.
#[target_feature(enable = "ssse3")]
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`avg_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır. (2) sınır:
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`w_avg_fn`); belirteç SSSE3 tespitinin kanıtıdır. (2) sınır:
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`mask_fn`); belirteç SSSE3 tespitinin kanıtıdır. (2) sınır:
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`w_mask_fn`); belirteç SSSE3 tespitinin kanıtıdır. (2) sınır:
    // `bilesik16_sozlesmesi` `tmp` öğelerini ve hedefi, maske denetimi alt türün
    // maske boyunu kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `dst`
    // ve `mask` ayrı `&mut`, girdiler `&`.
    unsafe { w_mask_w::<SUB>(dst, ds, tmp1, tmp2, w, h, mask, sign as i16, bdmax) }
}

/// 16 bpc `avg` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn avg_fn(_ssse3: Ssse3Token) -> AvgFn {
    avg16_giris
}

/// 16 bpc `w_avg` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn w_avg_fn(_ssse3: Ssse3Token) -> WAvgFn {
    w_avg16_giris
}

/// 16 bpc `mask` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn mask_fn(_ssse3: Ssse3Token) -> MaskFn {
    mask16_giris
}

/// 16 bpc `w_mask[alt]` SSSE3 yuvası (yalnız belirteçle); `alt` 0: 4:4:4,
/// 1: 4:2:2, 2: 4:2:0.
#[must_use]
pub fn w_mask_fn(_ssse3: Ssse3Token, alt: usize) -> WMaskFn {
    match alt {
        0 => w_mask16_giris::<0>,
        1 => w_mask16_giris::<1>,
        _ => w_mask16_giris::<2>,
    }
}
