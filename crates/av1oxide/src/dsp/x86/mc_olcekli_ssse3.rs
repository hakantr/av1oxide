//! MC ölçekli SSSE3 çekirdekleri (8 bpc) — F4-MC d (P35): `mc_scaled[0..9]`
//! (`put`), `mct_scaled[0..9]` (`prep`); 9 adet 8-tap birleşimi + bilinear.
//!
//! ## Sözleşme
//!
//! AVX2 gövdeleriyle ([`super::mc_olcekli`]) aynı: okuma bölgesi (8-tap satır
//! ve sütun `son + 7`, bilinear `3 ..= son + 4`; sütun başına 8 / 2 baytlık
//! yükleme, aşım yok), girişte O(1) denetim
//! ([`crate::dsp::mc::olcekli_sozlesmesi`]). Sözleşme dışı girdi skalere
//! devreder ve sayılır.
//!
//! ## Yöntem
//!
//! Aritmetik ve yapı AVX2 gövdesiyle aynıdır; yazmaç 128 bittir.
//!
//! - **Sütun önhesabı** (çağrı başına bir kez): sütun ofseti ve filtresi
//!   (`u64`, AVX2 gövdesinin tablosu; alt piksel 0'da birim filtre).
//! - **Yatay** (8-tap): iki sütunun 8 baytlık pencereleri ve filtreleri bir
//!   yazmaçta, `pmaddubsw` (komşu tap ikilileri) + `pmaddwd(·, 1)` + `phaddd`
//!   sıralı `[x0..x3]`, `[x4..x7]`; `(Σ + 2) >> 2`. Bilinear: sütun başına 2
//!   bayt ve `(16−m, m)` ağırlığı, `pmaddubsw`.
//! - **Dikey**: 8-tap 8 satırlık halka, satır ikilileri `punpck*wd` +
//!   `pmaddwd`, 8 sütunluk parçalar; bilinear 2 satırlık halka (`w ≤ 8` dar
//!   yolu yazmaçta, AVX2 gövdesiyle aynı). `dy` 1024/2048 sabit dallar.
//!
//! Yatay ve dikey adım makroyla döngüye açılır (satır başına çağrı yok). SSE4.1
//! ve üstü komut yoktur.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mc_olcekli::{filtre, kat_ikili, prep_skaler, put_skaler, tablolar};
use super::mem::{chunk, chunk_mut, load_i16x8, store_i16x8, store_u8x8};
use crate::cpu::Ssse3Token;
use crate::dsp::mc::{
    Filter2d, McPrepScaledFn, McPutScaledFn, olcekli_sozlesmesi, skaler_devri_say,
};
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_cvtsi32_si128, _mm_cvtsi64_si128, _mm_cvtsi128_si32,
    _mm_cvtsi128_si64, _mm_hadd_epi32, _mm_madd_epi16, _mm_maddubs_epi16, _mm_packs_epi32,
    _mm_packus_epi16, _mm_set_epi64x, _mm_set1_epi16, _mm_set1_epi32, _mm_setr_epi16,
    _mm_setzero_si128, _mm_srai_epi32, _mm_unpackhi_epi16, _mm_unpacklo_epi16,
};

/// Bilinear yuva indeksi.
const BILIN: usize = 9;

/// Çıktı hedefi: `put` (`dst`, adım) ya da `prep` (`tmp`, adım `w`).
struct Hedef<'a> {
    dst: &'a mut [u8],
    tmp: &'a mut [i16],
    ds: usize,
}

/// Bir çıktı satırının 8 sütununu (`i32` iki yarı, sıralı; yuvarlanmış ve
/// kaydırılmış) yazar: `x` sütun başı, `w` blok genişliği.
#[target_feature(enable = "ssse3")]
#[inline]
fn sakla8<const PREP: bool>(
    hd: &mut Hedef<'_>,
    y: usize,
    x: usize,
    w: usize,
    lo: __m128i,
    hi: __m128i,
) {
    let v = _mm_packs_epi32(lo, hi);
    let n = (w - x).min(8);
    if PREP {
        let o = y * w + x;
        match n {
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
        let p = _mm_packus_epi16(v, v);
        let o = y * hd.ds + x;
        match n {
            8 => store_u8x8(chunk_mut(hd.dst, o), p),
            4 => *chunk_mut::<u8, 4>(hd.dst, o) = _mm_cvtsi128_si32(p).to_le_bytes(),
            _ => {
                let b = _mm_cvtsi128_si32(p).to_le_bytes();
                *chunk_mut::<u8, 2>(hd.dst, o) = [b[0], b[1]];
            }
        }
    }
}

/// 8 baytlık sütun penceresi (`u64` olarak).
#[inline]
fn oku8(src: &[u8], o: usize) -> i64 {
    i64::from_le_bytes(*chunk(src, o))
}

/// 8-tap yatay satır: `ara` ← dolgu satırı (bayt tabanı `taban`); `ofs`/`fil`
/// sütun önhesabı (`gw` 8'in katı; `w` ötesi sütunlar son sütunun ofseti,
/// filtre 0). **Makro:** gövde çağıranın döngüsüne açılır (AVX2 gövdesi gibi).
macro_rules! yatay8_satir {
    ($ara:expr, $src:expr, $taban:expr, $ofs:expr, $fil:expr, $gw:expr) => {{
        let (ara, src, taban, ofs, fil, gw) = ($ara, $src, $taban, $ofs, $fil, $gw);
        let bir = _mm_set1_epi16(1);
        let iki = _mm_set1_epi32(2);
        let mut g = 0;
        while g < gw {
            let mut m = [_mm_setzero_si128(); 4];
            for (j, mj) in m.iter_mut().enumerate() {
                // Sütun `g + 2j` alt 8 bayt, `g + 2j + 1` üst 8 bayt.
                let (c0, c1) = (g + 2 * j, g + 2 * j + 1);
                let s = _mm_set_epi64x(oku8(src, taban + ofs[c1]), oku8(src, taban + ofs[c0]));
                let f = _mm_set_epi64x(fil[c1], fil[c0]);
                *mj = _mm_madd_epi16(_mm_maddubs_epi16(s, f), bir);
            }
            let lo = _mm_srai_epi32::<2>(_mm_add_epi32(_mm_hadd_epi32(m[0], m[1]), iki));
            let hi = _mm_srai_epi32::<2>(_mm_add_epi32(_mm_hadd_epi32(m[2], m[3]), iki));
            store_i16x8(chunk_mut(ara, g), _mm_packs_epi32(lo, hi));
            g += 8;
        }
    }};
}

/// Dar 8-tap yatay satır (`w` 2 ya da 4): yalnız `n` sütun hesaplanır (bir ya da
/// iki yazmaç); ara satırın kalan sütunları kullanılmaz. **Makro** (bkz. `yatay8_satir`).
macro_rules! yatay8_dar_satir {
    ($ara:expr, $src:expr, $taban:expr, $ofs:expr, $fil:expr, $n:expr) => {{
        let (ara, src, taban, ofs, fil, n) = ($ara, $src, $taban, $ofs, $fil, $n);
        let bir = _mm_set1_epi16(1);
        let iki = _mm_set1_epi32(2);
        let ikili = |c0: usize| {
            let s = _mm_set_epi64x(oku8(src, taban + ofs[c0 + 1]), oku8(src, taban + ofs[c0]));
            let f = _mm_set_epi64x(fil[c0 + 1], fil[c0]);
            _mm_madd_epi16(_mm_maddubs_epi16(s, f), bir)
        };
        let m0 = ikili(0);
        let m1 = if n == 4 { ikili(2) } else { m0 };
        let lo = _mm_srai_epi32::<2>(_mm_add_epi32(_mm_hadd_epi32(m0, m1), iki));
        store_i16x8(chunk_mut(ara, 0), _mm_packs_epi32(lo, lo));
    }};
}

/// 8-tap dikey çıktı satırı `y`: halkanın `bas`'tan başlayan 8 satırı (en eski
/// önce), katsayı ikilileri `k`; 8 sütunluk parçalar. **Makro** (bkz. `yatay8_satir`).
macro_rules! dikey8_satir {
    ($prep:expr, $hd:expr, $y:expr, $w:expr, $halka:expr, $bas:expr, $k:expr) => {{
        let (hd, y, w, halka, bas, k) = (&mut *$hd, $y, $w, $halka, $bas, $k);
        let yuv = _mm_set1_epi32(if $prep { 32 } else { 512 });
        let mut x = 0;
        while x < w {
            let mut lo = yuv;
            let mut hi = yuv;
            for j in 0..4 {
                let a = load_i16x8(chunk(&halka[(bas + 2 * j) & 7], x));
                let b = load_i16x8(chunk(&halka[(bas + 2 * j + 1) & 7], x));
                lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), k[j]));
                hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), k[j]));
            }
            let (lo, hi) = if $prep {
                (_mm_srai_epi32::<6>(lo), _mm_srai_epi32::<6>(hi))
            } else {
                (_mm_srai_epi32::<10>(lo), _mm_srai_epi32::<10>(hi))
            };
            sakla8::<PREP>(hd, y, x, w, lo, hi);
            x += 8;
        }
    }};
}

/// 8-tap ölçekli gövde. `G`: sütun kapasitesi ve ara satır uzunluğu
/// (`w.max(8)`'i kapsayan sınıf; yüklemeler 8'lik, `G`'yi aşmaz).
///
/// Tek döngü (AVX2 gövdesiyle aynı yapı): çıktı satırı `y` için kaynak satırı
/// `(my + y·dy) >> 10`'a kadar halkaya yatay satır üretilir, sonra dikey satır
/// yazılır; `dy` 1024/2048'de satır filtresi bir kez seçilir.
#[target_feature(enable = "ssse3")]
fn olcekli8<const PREP: bool, const G: usize>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
    f: Filter2d,
) {
    let (th, tv) = tablolar(f, w, h);
    let gw = w.max(8);
    let mut ofs = [0usize; G];
    let mut fil = [0i64; G];
    for x in 0..gw {
        if x < w {
            let t = mx + x as i32 * dx;
            ofs[x] = (t >> 10) as usize;
            fil[x] = filtre(th, ((t & 1023) >> 6) as usize);
        } else {
            ofs[x] = ofs[w - 1];
        }
    }
    let mut halka = [[0i16; G]; 8];
    let mut bas = 0usize;
    let mut in_y: i32 = -8;
    let mut my = my;
    let sabit = dy == 1024 || dy == 2048;
    let adim = dy >> 10;
    let kat = |fv: i64| {
        [
            _mm_set1_epi32(kat_ikili(fv, 0)),
            _mm_set1_epi32(kat_ikili(fv, 1)),
            _mm_set1_epi32(kat_ikili(fv, 2)),
            _mm_set1_epi32(kat_ikili(fv, 3)),
        ]
    };
    let mut k = kat(filtre(tv, ((my & 1023) >> 6) as usize));
    for y in 0..h {
        let hedef_y = if sabit && y > 0 {
            // `dy` 1024/2048: satır filtresi sabit (`my & 1023` değişmez).
            in_y + adim
        } else {
            if !sabit {
                k = kat(filtre(tv, ((my & 1023) >> 6) as usize));
            }
            my >> 10
        };
        while in_y < hedef_y {
            if w <= 4 {
                yatay8_dar_satir!(
                    &mut halka[bas],
                    src,
                    (in_y + 8) as usize * ss,
                    &ofs,
                    &fil,
                    w
                );
            } else {
                yatay8_satir!(
                    &mut halka[bas],
                    src,
                    (in_y + 8) as usize * ss,
                    &ofs,
                    &fil,
                    gw
                );
            }
            bas = (bas + 1) & 7;
            in_y += 1;
        }
        dikey8_satir!(PREP, hd, y, w, &halka, bas, &k);
        my += dy;
    }
}

/// Bilinear yatay: sütun başına `(s[o+3], s[o+4])` ve `(16−m, m)`; 8 sütunluk
/// parçalar.
#[target_feature(enable = "ssse3")]
#[inline]
fn yatay2<const G: usize>(
    ara: &mut [i16; G],
    src: &[u8],
    taban: usize,
    ofs: &[usize],
    agr: &[i16],
    gw: usize,
) {
    let p = |x: usize| i16::from_le_bytes(*chunk(src, taban + ofs[x] + 3));
    let mut g = 0;
    while g < gw {
        let s = _mm_setr_epi16(
            p(g),
            p(g + 1),
            p(g + 2),
            p(g + 3),
            p(g + 4),
            p(g + 5),
            p(g + 6),
            p(g + 7),
        );
        let a = load_i16x8(chunk(agr, g));
        store_i16x8(chunk_mut(ara, g), _mm_maddubs_epi16(s, a));
        g += 8;
    }
}

/// Bilinear ölçekli gövde (`w ≥ 16`). `G` sütun kapasitesi ve ara satır
/// uzunluğu (`w`).
#[target_feature(enable = "ssse3")]
fn olcekli2<const PREP: bool, const G: usize>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
) {
    let gw = w;
    let mut ofs = [0usize; G];
    // `(16 − m) | m << 8`: `pmaddubsw` ağırlık ikilisi.
    let mut agr = [0i16; G];
    for x in 0..gw {
        let t = mx + x as i32 * dx;
        ofs[x] = (t >> 10) as usize;
        let m = (t & 1023) >> 6;
        agr[x] = ((16 - m) | (m << 8)) as i16;
    }
    let mut halka = [[0i16; G]; 2];
    let mut in_y: i32 = -2;
    let mut my = my;
    let yuv = _mm_set1_epi32(if PREP { 8 } else { 128 });
    for y in 0..h {
        let src_y = my >> 10;
        while in_y < src_y {
            // Ara satır `in_y` → dolgu satırı `in_y + 5`.
            let slot = (in_y & 1) as usize;
            yatay2(
                &mut halka[slot],
                src,
                (in_y + 5) as usize * ss,
                &ofs,
                &agr,
                gw,
            );
            in_y += 1;
        }
        let m = (my & 1023) >> 6;
        let k = _mm_set1_epi32((16 - m) | (m << 16));
        let (ra, rb) = (
            &halka[(src_y & 1) as usize],
            &halka[((src_y + 1) & 1) as usize],
        );
        let mut x = 0;
        while x < w {
            let a = load_i16x8(chunk(ra, x));
            let b = load_i16x8(chunk(rb, x));
            let lo = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(a, b), k), yuv);
            let hi = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(a, b), k), yuv);
            let (lo, hi) = if PREP {
                (_mm_srai_epi32::<4>(lo), _mm_srai_epi32::<4>(hi))
            } else {
                (_mm_srai_epi32::<8>(lo), _mm_srai_epi32::<8>(hi))
            };
            sakla8::<PREP>(hd, y, x, w, lo, hi);
            x += 8;
        }
        my += dy;
    }
}

/// 8-tap genişlik sınıfı seçimi.
#[target_feature(enable = "ssse3")]
fn olcekli8_w<const PREP: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
    f: Filter2d,
) {
    match w {
        ..=8 => olcekli8::<PREP, 8>(hd, src, ss, w, h, mx, my, dx, dy, f),
        16 => olcekli8::<PREP, 16>(hd, src, ss, w, h, mx, my, dx, dy, f),
        32 => olcekli8::<PREP, 32>(hd, src, ss, w, h, mx, my, dx, dy, f),
        64 => olcekli8::<PREP, 64>(hd, src, ss, w, h, mx, my, dx, dy, f),
        _ => olcekli8::<PREP, 128>(hd, src, ss, w, h, mx, my, dx, dy, f),
    }
}

/// Bilinear dar yol (`w = N` ∈ {2, 4, 8}): sütun başına `(s[o+3], s[o+4])`
/// bayt ikilisi tek `u32`/`u64` kelimede toplanır (`N` sütun, fazlası yok),
/// `pmaddubsw` ağırlığı `(16−m, m)`, dikey `pmaddwd`; 2 satırlık halka
/// yazmaçta. AVX2 gövdesinin dar yoluyla aynı.
#[target_feature(enable = "ssse3")]
fn olcekli2_dar<const PREP: bool, const N: usize>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
) {
    let mut ofs = [0usize; N];
    let mut agr = [0i16; 8];
    for x in 0..N {
        let t = mx + x as i32 * dx;
        ofs[x] = (t >> 10) as usize;
        let m = (t & 1023) >> 6;
        agr[x] = ((16 - m) | (m << 8)) as i16;
    }
    let a = _mm_setr_epi16(
        agr[0], agr[1], agr[2], agr[3], agr[4], agr[5], agr[6], agr[7],
    );
    // Dolgu satırı tabanından `N` sütunun bayt ikilileri → `pmaddubsw`.
    let yatay = |taban: usize| -> __m128i {
        let p = |x: usize| u64::from(u16::from_le_bytes(*chunk(src, taban + ofs[x] + 3)));
        let s = match N {
            2 => _mm_cvtsi32_si128((p(0) | p(1) << 16) as i32),
            4 => _mm_cvtsi64_si128((p(0) | p(1) << 16 | p(2) << 32 | p(3) << 48) as i64),
            _ => _mm_set_epi64x(
                (p(4) | p(5) << 16 | p(6) << 32 | p(7) << 48) as i64,
                (p(0) | p(1) << 16 | p(2) << 32 | p(3) << 48) as i64,
            ),
        };
        _mm_maddubs_epi16(s, a)
    };
    let mut halka = [_mm_setzero_si128(); 2];
    let mut in_y: i32 = -2;
    let mut my = my;
    let yuv = _mm_set1_epi32(if PREP { 8 } else { 128 });
    for y in 0..h {
        let src_y = my >> 10;
        while in_y < src_y {
            // Ara satır `in_y` → dolgu satırı `in_y + 5`.
            halka[(in_y & 1) as usize] = yatay((in_y + 5) as usize * ss);
            in_y += 1;
        }
        let m = (my & 1023) >> 6;
        let k = _mm_set1_epi32((16 - m) | (m << 16));
        let (ra, rb) = (
            halka[(src_y & 1) as usize],
            halka[((src_y + 1) & 1) as usize],
        );
        let lo = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(ra, rb), k), yuv);
        let hi = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(ra, rb), k), yuv);
        let (lo, hi) = if PREP {
            (_mm_srai_epi32::<4>(lo), _mm_srai_epi32::<4>(hi))
        } else {
            (_mm_srai_epi32::<8>(lo), _mm_srai_epi32::<8>(hi))
        };
        sakla8::<PREP>(hd, y, 0, N, lo, hi);
        my += dy;
    }
}

/// Bilinear genişlik sınıfı seçimi.
#[target_feature(enable = "ssse3")]
fn olcekli2_w<const PREP: bool>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
) {
    match w {
        ..=2 => olcekli2_dar::<PREP, 2>(hd, src, ss, h, mx, my, dx, dy),
        4 => olcekli2_dar::<PREP, 4>(hd, src, ss, h, mx, my, dx, dy),
        8 => olcekli2_dar::<PREP, 8>(hd, src, ss, h, mx, my, dx, dy),
        16 => olcekli2::<PREP, 16>(hd, src, ss, w, h, mx, my, dx, dy),
        32 => olcekli2::<PREP, 32>(hd, src, ss, w, h, mx, my, dx, dy),
        64 => olcekli2::<PREP, 64>(hd, src, ss, w, h, mx, my, dx, dy),
        _ => olcekli2::<PREP, 128>(hd, src, ss, w, h, mx, my, dx, dy),
    }
}

fn put_giris<const F: usize>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
    bd: BitDepth,
    bdmax: u16,
) {
    let ok = bd == BitDepth::Eight
        && ds >= w
        && olcekli_sozlesmesi(F == BILIN, src.len(), ss, w, h, mx, my, dx, dy)
        && h >= 1
        && dst.len() >= (h - 1) * ds + w;
    if !ok {
        skaler_devri_say();
        return put_skaler::<F>(dst, ds, src, ss, w, h, mx, my, dx, dy, bd, bdmax);
    }
    let mut hd = Hedef {
        dst,
        tmp: &mut [],
        ds,
    };
    let f = Filter2d::from_index(F).unwrap_or(Filter2d::Regular);
    if F == BILIN {
        // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
        // (`put_scaled_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır.
        // (2) sınır: `olcekli_sozlesmesi` okuma bölgesinin son baytını kaynak
        // diliminde, hedef denetimi `h` satır × `w` baytı kapsar; erişimler
        // denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
        unsafe { olcekli2_w::<false>(&mut hd, src, ss, w, h, mx, my, dx, dy) }
    } else {
        // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
        // (`put_scaled_fn`); belirteç SSSE3 tespitinin kanıtıdır. (2) sınır:
        // `olcekli_sozlesmesi` 8-tap okuma bölgesini, hedef denetimi `h` satır ×
        // `w` baytı kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `dst`
        // tek `&mut`, `src` `&`.
        unsafe { olcekli8_w::<false>(&mut hd, src, ss, w, h, mx, my, dx, dy, f) }
    }
}

fn prep_giris<const F: usize>(
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
    bd: BitDepth,
    bdmax: u16,
) {
    let ok = bd == BitDepth::Eight
        && olcekli_sozlesmesi(F == BILIN, src.len(), ss, w, h, mx, my, dx, dy)
        && tmp.len() >= w * h;
    if !ok {
        skaler_devri_say();
        return prep_skaler::<F>(tmp, src, ss, w, h, mx, my, dx, dy, bd, bdmax);
    }
    let mut hd = Hedef {
        dst: &mut [],
        tmp,
        ds: w,
    };
    let f = Filter2d::from_index(F).unwrap_or(Filter2d::Regular);
    if F == BILIN {
        // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
        // (`prep_scaled_fn`); belirteç SSSE3 tespitinin kanıtıdır. (2) sınır:
        // `olcekli_sozlesmesi` bilinear okuma bölgesini, `tmp.len() ≥ w·h` çıktıyı
        // kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `tmp` tek `&mut`,
        // `src` `&`.
        unsafe { olcekli2_w::<true>(&mut hd, src, ss, w, h, mx, my, dx, dy) }
    } else {
        // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
        // (`prep_scaled_fn`); belirteç SSSE3 tespitinin kanıtıdır. (2) sınır:
        // `olcekli_sozlesmesi` 8-tap okuma bölgesini, `tmp.len() ≥ w·h` çıktıyı
        // kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `tmp` tek `&mut`,
        // `src` `&`.
        unsafe { olcekli8_w::<true>(&mut hd, src, ss, w, h, mx, my, dx, dy, f) }
    }
}

/// `mc_scaled[filtre]` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn put_scaled_fn(_ssse3: Ssse3Token, f: Filter2d) -> McPutScaledFn {
    match f.index() {
        0 => put_giris::<0>,
        1 => put_giris::<1>,
        2 => put_giris::<2>,
        3 => put_giris::<3>,
        4 => put_giris::<4>,
        5 => put_giris::<5>,
        6 => put_giris::<6>,
        7 => put_giris::<7>,
        8 => put_giris::<8>,
        _ => put_giris::<BILIN>,
    }
}

/// `mct_scaled[filtre]` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn prep_scaled_fn(_ssse3: Ssse3Token, f: Filter2d) -> McPrepScaledFn {
    match f.index() {
        0 => prep_giris::<0>,
        1 => prep_giris::<1>,
        2 => prep_giris::<2>,
        3 => prep_giris::<3>,
        4 => prep_giris::<4>,
        5 => prep_giris::<5>,
        6 => prep_giris::<6>,
        7 => prep_giris::<7>,
        8 => prep_giris::<8>,
        _ => prep_giris::<BILIN>,
    }
}
