//! MC bileşik SSSE3 çekirdekleri (8 bpc) — F4-MC d (P35): `avg`, `w_avg`,
//! `mask`, `w_mask[0..2]` (4:4:4, 4:2:2, 4:2:0).
//!
//! ## Sözleşme
//!
//! AVX2 gövdeleriyle ([`super::mc_bilesik`]) aynı: girişte O(1) denetim
//! ([`crate::dsp::mc::bilesik_sozlesmesi`]); sözleşme dışı girdi skalere
//! devreder ve sayılır. Okuma tam `w·h` öğedir (`tmp` satırları bitişik),
//! yazma yalnız `h` satır × `w` bayt (+ maske). Değer tanım kümesi 8 bpc
//! [`PREP_ARALIK`](super::mc_bilesik::PREP_ARALIK); `a + b` ve `2·(a − b)` `i16`'ya sığar.
//!
//! ## Yöntem
//!
//! Aritmetik AVX2 gövdesiyle aynıdır (`pmulhrsw`, `pmulhw`, `psubusw`,
//! `pabsw`, `pmaddubsw`); yazmaç 128 bittir. 16 öğe bir adım: iki `i16`
//! yazmacı `packuswb` ile sıralı 16 bayt (AVX2'deki `vpermq` gerekmez);
//! `tmp` satırları bitişik olduğundan `w < 16`'da adım birden çok satır
//! yazar. `mask` baytları `punpck{l,h}bw` sıfırla genişler (`pmovzxbw`
//! SSE4.1 olduğundan kullanılmaz). SSE4.1 ve üstü komut yoktur.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mem::{chunk, chunk_mut, load_i16x8, load_u8x16, store_u8x4, store_u8x8, store_u8x16};
use crate::cpu::Ssse3Token;
use crate::dsp::mc::{AvgFn, MaskFn, WAvgFn, WMaskFn, bilesik_sozlesmesi, skaler_devri_say};
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, _mm_abs_epi16, _mm_add_epi16, _mm_maddubs_epi16, _mm_mulhi_epi16, _mm_mulhrs_epi16,
    _mm_packus_epi16, _mm_set1_epi8, _mm_set1_epi16, _mm_setzero_si128, _mm_slli_epi16,
    _mm_srli_epi16, _mm_srli_si128, _mm_sub_epi16, _mm_subs_epu16, _mm_unpackhi_epi8,
    _mm_unpacklo_epi8,
};

/// Bileşik işlem (derleme zamanı).
const AVG: u8 = 0;
const W_AVG: u8 = 1;
const MASK: u8 = 2;

/// Bir adımın girdileri: `t1`, `t2` (`w_avg`'da gerekirse yer değiştirmiş),
/// `mask` (`mask`'ta), `k` (`w_avg` çarpanı).
struct Girdi<'a> {
    t1: &'a [i16],
    t2: &'a [i16],
    mask: &'a [u8],
    k: i16,
}

/// 8 öğenin `i16` sonucu; `m` `mask`'ta 8 maske değeri (`i16`).
#[target_feature(enable = "ssse3")]
#[inline]
fn hesap8<const OP: u8>(g: &Girdi<'_>, o: usize, m: __m128i) -> __m128i {
    let a = load_i16x8(chunk(g.t1, o));
    let b = load_i16x8(chunk(g.t2, o));
    match OP {
        AVG => _mm_mulhrs_epi16(_mm_add_epi16(a, b), _mm_set1_epi16(1024)),
        W_AVG => {
            let r = _mm_add_epi16(a, _mm_mulhi_epi16(_mm_sub_epi16(a, b), _mm_set1_epi16(g.k)));
            _mm_mulhrs_epi16(r, _mm_set1_epi16(2048))
        }
        _ => {
            let k = _mm_slli_epi16::<9>(_mm_sub_epi16(_mm_setzero_si128(), m));
            let d = _mm_slli_epi16::<1>(_mm_sub_epi16(b, a));
            let r = _mm_add_epi16(b, _mm_mulhi_epi16(d, k));
            _mm_mulhrs_epi16(r, _mm_set1_epi16(2048))
        }
    }
}

/// Öğe ofseti `o`'dan başlayan 16 öğenin sıralı 16 baytı. **Makro:** gövde
/// çağıranın döngüsüne açılır (işlev olarak `w_avg`/`mask`'ta satır içine
/// alınmıyor ve 16 öğe başına çağrıya dönüşüyordu; `#[target_feature]` işlevi
/// `#[inline(always)]` olamaz).
macro_rules! hesap16 {
    ($op:expr, $g:expr, $o:expr) => {{
        let (g, o) = ($g, $o);
        let (m0, m1) = if $op == MASK {
            let mb = load_u8x16(chunk(g.mask, o));
            let z = _mm_setzero_si128();
            (_mm_unpacklo_epi8(mb, z), _mm_unpackhi_epi8(mb, z))
        } else {
            (_mm_setzero_si128(), _mm_setzero_si128())
        };
        _mm_packus_epi16(hesap8::<{ $op }>(g, o, m0), hesap8::<{ $op }>(g, o + 8, m1))
    }};
}

/// Öğe ofseti `o`'dan başlayan sıralı 16 pikseli (`W` sütunlu bitişik satırlar)
/// hedefe yazar; `W ≥ 16`'da tek satır parçası.
#[target_feature(enable = "ssse3")]
#[inline]
fn sakla16<const W: usize>(dst: &mut [u8], ds: usize, o: usize, p: __m128i) {
    let (y, x) = (o / W, o % W);
    match W {
        4 => {
            let mut v = p;
            for r in 0..4 {
                store_u8x4(chunk_mut(dst, (y + r) * ds), v);
                v = _mm_srli_si128::<4>(v);
            }
        }
        8 => {
            store_u8x8(chunk_mut(dst, y * ds), p);
            store_u8x8(chunk_mut(dst, (y + 1) * ds), _mm_srli_si128::<8>(p));
        }
        _ => store_u8x16(chunk_mut(dst, y * ds + x), p),
    }
}

/// `avg` / `w_avg` / `mask` gövdesi: `W` sütun.
#[target_feature(enable = "ssse3")]
fn bidir<const OP: u8, const W: usize>(dst: &mut [u8], ds: usize, n: usize, g: &Girdi<'_>) {
    let mut o = 0;
    while o < n {
        sakla16::<W>(dst, ds, o, hesap16!(OP, g, o));
        o += 16;
    }
}

/// Genişliğe göre `bidir` monomorfu.
#[target_feature(enable = "ssse3")]
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

/// `w_mask` 8 öğesi: (`i16` piksel sonucu, `64 − m`).
#[target_feature(enable = "ssse3")]
#[inline]
fn wm8(t1: &[i16], t2: &[i16], o: usize) -> (__m128i, __m128i) {
    let a = load_i16x8(chunk(t1, o));
    let b = load_i16x8(chunk(t2, o));
    let d = _mm_sub_epi16(b, a);
    let k = _mm_srli_epi16::<8>(_mm_subs_epu16(_mm_set1_epi16(6903), _mm_abs_epi16(d)));
    let r = _mm_add_epi16(a, _mm_mulhi_epi16(d, _mm_slli_epi16::<10>(k)));
    (_mm_mulhrs_epi16(r, _mm_set1_epi16(2048)), k)
}

/// 16 öğe: (sıralı 16 piksel baytı, sıralı 16 `m` baytı, 8 ikili toplam `i16`).
#[target_feature(enable = "ssse3")]
#[inline]
fn wm16(t1: &[i16], t2: &[i16], o: usize) -> (__m128i, __m128i, __m128i) {
    let (r0, k0) = wm8(t1, t2, o);
    let (r1, k1) = wm8(t1, t2, o + 8);
    let altmis4 = _mm_set1_epi16(64);
    let m = _mm_packus_epi16(_mm_sub_epi16(altmis4, k0), _mm_sub_epi16(altmis4, k1));
    (
        _mm_packus_epi16(r0, r1),
        m,
        _mm_maddubs_epi16(m, _mm_set1_epi8(1)),
    )
}

/// `w_mask` gövdesi: alt tür `SUB` (0: 4:4:4, 1: 4:2:2, 2: 4:2:0), `W` sütun.
#[target_feature(enable = "ssse3")]
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
        let ek = _mm_set1_epi16(1 - sign);
        let mut o = 0;
        while o < n {
            let (p, m, ikili) = wm16(t1, t2, o);
            sakla16::<W>(dst, ds, o, p);
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
            let o1 = o + W;
            let (p0, _, s0) = wm16(t1, t2, o);
            let (p1, _, s1) = wm16(t1, t2, o1);
            sakla16::<W>(dst, ds, o, p0);
            sakla16::<W>(dst, ds, o1, p1);
            let v = _mm_srli_epi16::<2>(_mm_add_epi16(_mm_add_epi16(s0, s1), ek));
            store_u8x8(
                chunk_mut(mask, (y / 2) * (W / 2) + x / 2),
                _mm_packus_epi16(v, v),
            );
            o += if x + 16 >= W { W + 16 } else { 16 };
        } else {
            // `W = 8`: 16 öğe = satır ikilisi; ikili toplamların alt yarısı
            // satır `y`, üst yarısı satır `y+1`.
            let (p, _, s) = wm16(t1, t2, o);
            sakla16::<W>(dst, ds, o, p);
            let v = _mm_add_epi16(s, _mm_srli_si128::<8>(s));
            let v = _mm_srli_epi16::<2>(_mm_add_epi16(v, ek));
            store_u8x4(chunk_mut(mask, o / 4), _mm_packus_epi16(v, v));
            o += 16;
        }
    }
}

/// `w_mask` genişlik monomorfu.
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`avg_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır. (2) sınır:
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`w_avg_fn`); belirteç SSSE3 tespitinin kanıtıdır. (2) sınır:
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`mask_fn`); belirteç SSSE3 tespitinin kanıtıdır. (2) sınır:
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`w_mask_fn`); belirteç SSSE3 tespitinin kanıtıdır. (2) sınır:
    // `bilesik_sozlesmesi` `tmp` öğelerini ve hedefi, maske denetimi alt türün
    // maske boyunu kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `dst`
    // ve `mask` ayrı `&mut`, girdiler `&`.
    unsafe { w_mask_w::<SUB>(dst, ds, tmp1, tmp2, w, h, mask, sign as i16) }
}

/// `avg` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn avg_fn(_ssse3: Ssse3Token) -> AvgFn {
    avg_giris
}

/// `w_avg` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn w_avg_fn(_ssse3: Ssse3Token) -> WAvgFn {
    w_avg_giris
}

/// `mask` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn mask_fn(_ssse3: Ssse3Token) -> MaskFn {
    mask_giris
}

/// `w_mask[alt]` SSSE3 yuvası (yalnız belirteçle); `alt` 0: 4:4:4, 1: 4:2:2,
/// 2: 4:2:0.
#[must_use]
pub fn w_mask_fn(_ssse3: Ssse3Token, alt: usize) -> WMaskFn {
    match alt {
        0 => w_mask_giris::<0>,
        1 => w_mask_giris::<1>,
        _ => w_mask_giris::<2>,
    }
}
