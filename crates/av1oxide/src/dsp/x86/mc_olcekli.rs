//! MC ölçekli AVX2 çekirdekleri (8 bpc) — F4-MC b (P33): `mc_scaled[0..9]`
//! (`put`), `mct_scaled[0..9]` (`prep`); 9 adet 8-tap birleşimi + bilinear.
//!
//! ## Sözleşme
//!
//! Kaynak dolgu hücresi (blok satırı −3, sütunu −3). Çıktı sütunu `x`'in kaynak
//! ofseti `(mx + x·dx) >> 10`, alt piksel `((mx + x·dx) & 1023) >> 6`; çıktı
//! satırı `y` için `(my + y·dy) >> 10` ve `((my + y·dy) & 1023) >> 6`.
//! **Okuma bölgesi** (8-tap): dolgu satırları `0 ..= ((my + dy·(h−1)) >> 10) + 7`,
//! sütunları `0 ..= ((mx + dx·(w−1)) >> 10) + 7`; bilinear: satırlar
//! `3 ..= son + 4`, sütunlar `3 ..= son + 4`. Gövde yalnız bu bölgeyi okur
//! (sütun başına 8 / 2 baytlık yükleme, aşım yok). Girişte O(1) denetim
//! ([`crate::dsp::mc::olcekli_sozlesmesi`]): `w` 2'nin kuvveti 2…128, `h`
//! 1…128 (OBMC üst komşusu `3·2^k` yükseklik; satır döngüsü `h`'den bağımsız),
//! `mx`, `my` 0…1023, `dx`, `dy` 1…2048 (AV1 ölçek sınırı), bölgenin son
//! baytı kaynak diliminde, hedef `h` satır × `w` (`prep` `w·h`). Sözleşme dışı
//! girdi skalere devreder ve sayılır.
//!
//! ## Yöntem
//!
//! - **Sütun önhesabı** (çağrı başına bir kez): her sütunun kaynak ofseti ve
//!   filtresi (`u64`; alt piksel 0'da `[0, 0, 0, 64, 0, 0, 0, 0]`:
//!   `(64·s + 2) >> 2 = s << 4`, skaler "filtre yok" yoluyla aynı).
//! - **Yatay** (8-tap): 8 sütunluk grupta sütun başına 8 baytlık pencere ve
//!   filtre `u64` olarak toplanır, `pmaddubsw` (komşu tap ikilileri) +
//!   `pmaddwd(·, 1)` + `phaddd`, `(Σ + 2) >> 2`. Bilinear: sütun başına 2 bayt
//!   ve `(16−m, m)` ağırlığı, `pmaddubsw` (kaydırmasız).
//! - **Dikey**: 8-tap 8 satırlık halka (skalerle aynı döndürme), satır
//!   ikilileri `punpck*wd` + `pmaddwd`; satır filtresi alt piksel 0'da
//!   `[0, 0, 0, 64, …]` (`put` `(64·m + 512) >> 10 = (m + 8) >> 4`, `prep` `m`).
//!   Bilinear 2 satırlık halka. `put` `(Σ + 512) >> 10` (bilinear
//!   `(Σ + 128) >> 8`), `prep` `(Σ + 32) >> 6` (bilinear `(Σ + 8) >> 4`).
//! - **`dy = 1024` ve `dy = 2048`** ayrı dallardır (dav1d gibi): satır filtresi
//!   bir kez seçilir, her çıktı satırı tam 1 / 2 yeni ara satır üretir.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mem::{
    chunk, chunk_mut, load_i16x16, store_i16x8, store_i16x16, store_u8x8, store_u8x16,
};
use crate::cpu::Avx2Token;
use crate::dsp::mc::{
    Filter2d, McPrepScaledFn, McPutScaledFn, olcekli_sozlesmesi, skaler_devri_say,
};
use crate::dsp::mc_tables::MC_SUBPEL_FILTERS;
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi32, _mm_cvtsi32_si128, _mm_cvtsi64_si128, _mm_cvtsi128_si32,
    _mm_cvtsi128_si64, _mm_madd_epi16, _mm_maddubs_epi16, _mm_packs_epi32, _mm_packus_epi16,
    _mm_set_epi64x, _mm_set1_epi32, _mm_setr_epi16, _mm_setzero_si128, _mm_srai_epi32,
    _mm_unpackhi_epi16, _mm_unpacklo_epi16, _mm_unpacklo_epi32, _mm256_add_epi32,
    _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_hadd_epi32, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_packs_epi32, _mm256_packus_epi16, _mm256_permute4x64_epi64,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi16, _mm256_setr_epi64x, _mm256_srai_epi32,
    _mm256_unpackhi_epi16, _mm256_unpacklo_epi16,
};

/// Bilinear yuva indeksi.
const BILIN: usize = 9;
/// Alt piksel 0 ("filtre yok") yerine geçen birim filtre: tap 3 = 64.
const BIRIM: i64 = 64 << 24;

/// Bütün alt piksel filtreleri `u64` olarak (derleme zamanında): `[tablo][p]`,
/// `p` 0…15 (0 → birim). Satır ve sütun başına filtre seçimi tek tablo okuması
/// (P34: dizi `map` kapanışlı çevirme satır içine alınmıyordu).
static FILTRE_U64: [[i64; 16]; 6] = filtre_tablosu();

const fn filtre_tablosu() -> [[i64; 16]; 6] {
    let mut t = [[BIRIM; 16]; 6];
    let mut k = 0;
    while k < 6 {
        let mut p = 1;
        while p < 16 {
            let f = &MC_SUBPEL_FILTERS[k][p - 1];
            let mut v = 0i64;
            let mut i = 0;
            while i < 8 {
                v |= ((f[i] as u8) as i64) << (8 * i);
                i += 1;
            }
            t[k][p] = v;
            p += 1;
        }
        k += 1;
    }
    t
}

/// Filtre: `tablo` (0…5) ve alt piksel `p` (0…15; 0 → birim).
#[inline]
pub(super) fn filtre(tablo: usize, p: usize) -> i64 {
    FILTRE_U64[tablo][p]
}

/// `w`'ye göre yatay, `h`'ye göre dikey tablo (dav1d `GET_H/V_FILTER`).
pub(super) fn tablolar(f: Filter2d, w: usize, h: usize) -> (usize, usize) {
    let (kh, kv) = (f.horizontal_table(), f.vertical_table());
    (
        if w > 4 { kh } else { 3 + (kh & 1) },
        if h > 4 { kv } else { 3 + (kv & 1) },
    )
}

/// Çıktı hedefi: `put` (`dst`, adım) ya da `prep` (`tmp`, adım `w`).
struct Hedef<'a> {
    dst: &'a mut [u8],
    tmp: &'a mut [i16],
    ds: usize,
}

/// Bir çıktı satırının 16 sütununu (`i32` iki yarı, lane düzeni) yuvarlar ve
/// yazar: `x` sütun başı, `w` blok genişliği.
#[target_feature(enable = "avx2")]
#[inline]
fn sakla<const PREP: bool>(
    hd: &mut Hedef<'_>,
    y: usize,
    x: usize,
    w: usize,
    lo: __m256i,
    hi: __m256i,
) {
    let v = _mm256_packs_epi32(lo, hi); // 16 × i16, sütun sırası (lane başına)
    if PREP {
        let o = y * w + x;
        let n = (w - x).min(16);
        if n == 16 {
            store_i16x16(chunk_mut(hd.tmp, o), v);
        } else {
            let l = _mm256_castsi256_si128(v);
            match n {
                8 => store_i16x8(chunk_mut(hd.tmp, o), l),
                4 => {
                    let b = _mm_cvtsi128_si64(l).to_le_bytes();
                    for (k, t) in hd.tmp[o..o + 4].iter_mut().enumerate() {
                        *t = i16::from_le_bytes([b[2 * k], b[2 * k + 1]]);
                    }
                }
                _ => {
                    let b = _mm_cvtsi128_si32(l).to_le_bytes();
                    hd.tmp[o] = i16::from_le_bytes([b[0], b[1]]);
                    hd.tmp[o + 1] = i16::from_le_bytes([b[2], b[3]]);
                }
            }
        }
    } else {
        let p = _mm256_permute4x64_epi64::<0xd8>(_mm256_packus_epi16(v, v));
        let l = _mm256_castsi256_si128(p);
        let o = y * hd.ds + x;
        match (w - x).min(16) {
            16 => store_u8x16(chunk_mut(hd.dst, o), l),
            8 => store_u8x8(chunk_mut(hd.dst, o), l),
            4 => chunk_mut::<u8, 4>(hd.dst, o).copy_from_slice(&_mm_cvtsi128_si32(l).to_le_bytes()),
            _ => chunk_mut::<u8, 2>(hd.dst, o)
                .copy_from_slice(&_mm_cvtsi128_si32(l).to_le_bytes()[..2]),
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
/// filtre 0). **Makro:** gövde çağıranın döngüsüne açılır. `#[target_feature]`
/// işlevi `#[inline(always)]` olamaz (Rust 1.99.1); P33'te satır başına çağrı ve
/// yığın yüklemesi genel dalda (`dy` 1024/2048 dışı) iki kipli hıza yol açıyordu.
macro_rules! yatay8_satir {
    ($ara:expr, $src:expr, $taban:expr, $ofs:expr, $fil:expr, $gw:expr) => {{
        let (ara, src, taban, ofs, fil, gw) = ($ara, $src, $taban, $ofs, $fil, $gw);
        let bir = _mm256_set1_epi16(1);
        let iki = _mm256_set1_epi32(2);
        let mut g = 0;
        while g < gw {
            let s0 = _mm256_setr_epi64x(
                oku8(src, taban + ofs[g]),
                oku8(src, taban + ofs[g + 1]),
                oku8(src, taban + ofs[g + 2]),
                oku8(src, taban + ofs[g + 3]),
            );
            let s1 = _mm256_setr_epi64x(
                oku8(src, taban + ofs[g + 4]),
                oku8(src, taban + ofs[g + 5]),
                oku8(src, taban + ofs[g + 6]),
                oku8(src, taban + ofs[g + 7]),
            );
            let f0 = _mm256_setr_epi64x(fil[g], fil[g + 1], fil[g + 2], fil[g + 3]);
            let f1 = _mm256_setr_epi64x(fil[g + 4], fil[g + 5], fil[g + 6], fil[g + 7]);
            let a = _mm256_madd_epi16(_mm256_maddubs_epi16(s0, f0), bir);
            let b = _mm256_madd_epi16(_mm256_maddubs_epi16(s1, f1), bir);
            // [x0, x1, x4, x5 | x2, x3, x6, x7]
            let t = _mm256_srai_epi32::<2>(_mm256_add_epi32(_mm256_hadd_epi32(a, b), iki));
            let lo = _mm256_castsi256_si128(t);
            let hi = _mm256_extracti128_si256::<1>(t);
            let v = _mm_unpacklo_epi32(_mm_packs_epi32(lo, lo), _mm_packs_epi32(hi, hi));
            store_i16x8(chunk_mut(ara, g), v);
            g += 8;
        }
    }};
}

/// Satır filtresi `fv`'nin (`u64`, 8 × `i8`) `pmaddwd` katsayı ikilileri.
#[inline]
pub(super) const fn kat_ikili(fv: i64, j: usize) -> i32 {
    let a = (fv >> (16 * j)) as i8 as i32;
    let b = (fv >> (16 * j + 8)) as i8 as i32;
    (a & 0xffff) | (b << 16)
}

/// 8-tap dikey çıktı satırı `y`: halkanın `bas`'tan başlayan 8 satırı (en eski
/// önce), katsayı ikilileri `k`; 16 sütunluk parçalar. **Makro** (bkz. `yatay8_satir`).
macro_rules! dikey8_satir {
    ($prep:expr, $hd:expr, $y:expr, $w:expr, $halka:expr, $bas:expr, $k:expr) => {{
        let (hd, y, w, halka, bas, k) = (&mut *$hd, $y, $w, $halka, $bas, $k);
        let yuv = _mm256_set1_epi32(if $prep { 32 } else { 512 });
        let mut x = 0;
        while x < w {
            let mut lo = yuv;
            let mut hi = yuv;
            for j in 0..4 {
                let a = load_i16x16(chunk(&halka[(bas + 2 * j) & 7], x));
                let b = load_i16x16(chunk(&halka[(bas + 2 * j + 1) & 7], x));
                lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), k[j]));
                hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), k[j]));
            }
            let (lo, hi) = if $prep {
                (_mm256_srai_epi32::<6>(lo), _mm256_srai_epi32::<6>(hi))
            } else {
                (_mm256_srai_epi32::<10>(lo), _mm256_srai_epi32::<10>(hi))
            };
            sakla::<PREP>(hd, y, x, w, lo, hi);
            x += 16;
        }
    }};
}

/// 8-tap ölçekli gövde. `G`: sütun kapasitesi (`w.max(8)`'i kapsayan sınıf),
/// `R = G + 16` ara satır uzunluğu (16'lık yükleme payı).
///
/// Tek döngü: çıktı satırı `y` için kaynak satırı `(my + y·dy) >> 10`'a kadar
/// halkaya yatay satır üretilir, sonra dikey satır yazılır. `dy` 1024/2048'de
/// satır filtresi bir kez seçilir ve her satır tam `dy >> 10` yeni ara satır
/// ekler (dav1d'nin ayrı dalları); genel `dy`'de filtre satır başına seçilir.
/// Yatay ve dikey adım makroyla döngüye açılır: döngüde çağrı yok.
#[target_feature(enable = "avx2")]
fn olcekli8<const PREP: bool, const G: usize, const R: usize>(
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
    let mut halka = [[0i16; R]; 8];
    let mut bas = 0usize;
    let mut in_y: i32 = -8;
    let mut my = my;
    let sabit = dy == 1024 || dy == 2048;
    let adim = dy >> 10;
    let kat = |fv: i64| {
        [
            _mm256_set1_epi32(kat_ikili(fv, 0)),
            _mm256_set1_epi32(kat_ikili(fv, 1)),
            _mm256_set1_epi32(kat_ikili(fv, 2)),
            _mm256_set1_epi32(kat_ikili(fv, 3)),
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
            yatay8_satir!(
                &mut halka[bas],
                src,
                (in_y + 8) as usize * ss,
                &ofs,
                &fil,
                gw
            );
            bas = (bas + 1) & 7;
            in_y += 1;
        }
        dikey8_satir!(PREP, hd, y, w, &halka, bas, &k);
        my += dy;
    }
}

/// Bilinear yatay: sütun başına `(s[o+3], s[o+4])` ve `(16−m, m)`.
#[target_feature(enable = "avx2")]
#[inline]
fn yatay2<const R: usize>(
    ara: &mut [i16; R],
    src: &[u8],
    taban: usize,
    ofs: &[usize],
    agr: &[i16],
    gw: usize,
) {
    let p = |x: usize| i16::from_le_bytes(*chunk(src, taban + ofs[x] + 3));
    let mut g = 0;
    while g < gw {
        let s = _mm256_setr_epi16(
            p(g),
            p(g + 1),
            p(g + 2),
            p(g + 3),
            p(g + 4),
            p(g + 5),
            p(g + 6),
            p(g + 7),
            p(g + 8),
            p(g + 9),
            p(g + 10),
            p(g + 11),
            p(g + 12),
            p(g + 13),
            p(g + 14),
            p(g + 15),
        );
        let a = load_i16x16(chunk(agr, g));
        store_i16x16(chunk_mut(ara, g), _mm256_maddubs_epi16(s, a));
        g += 16;
    }
}

/// Bilinear ölçekli gövde.
#[target_feature(enable = "avx2")]
/// `G` sütun kapasitesi (`w.max(16)`), `R = G + 16`.
fn olcekli2<const PREP: bool, const G: usize, const R: usize>(
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
    let gw = w.max(16);
    let mut ofs = [0usize; G];
    // `(16 − m) | m << 8`: `pmaddubsw` ağırlık ikilisi.
    let mut agr = [0i16; G];
    for x in 0..gw {
        if x < w {
            let t = mx + x as i32 * dx;
            ofs[x] = (t >> 10) as usize;
            let m = (t & 1023) >> 6;
            agr[x] = ((16 - m) | (m << 8)) as i16;
        } else {
            ofs[x] = ofs[w - 1];
        }
    }
    let mut halka = [[0i16; R]; 2];
    let mut in_y: i32 = -2;
    let mut my = my;
    let yuv = _mm256_set1_epi32(if PREP { 8 } else { 128 });
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
        let k = _mm256_set1_epi32((16 - m) | (m << 16));
        let (ra, rb) = (
            &halka[(src_y & 1) as usize],
            &halka[((src_y + 1) & 1) as usize],
        );
        let mut x = 0;
        while x < w {
            let a = load_i16x16(chunk(ra, x));
            let b = load_i16x16(chunk(rb, x));
            let lo = _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), k), yuv);
            let hi = _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), k), yuv);
            let (lo, hi) = if PREP {
                (_mm256_srai_epi32::<4>(lo), _mm256_srai_epi32::<4>(hi))
            } else {
                (_mm256_srai_epi32::<8>(lo), _mm256_srai_epi32::<8>(hi))
            };
            sakla::<PREP>(hd, y, x, w, lo, hi);
            x += 16;
        }
        my += dy;
    }
}

/// 8-tap genişlik sınıfı seçimi.
#[target_feature(enable = "avx2")]
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
        ..=8 => olcekli8::<PREP, 8, 24>(hd, src, ss, w, h, mx, my, dx, dy, f),
        16 => olcekli8::<PREP, 16, 32>(hd, src, ss, w, h, mx, my, dx, dy, f),
        32 => olcekli8::<PREP, 32, 48>(hd, src, ss, w, h, mx, my, dx, dy, f),
        64 => olcekli8::<PREP, 64, 80>(hd, src, ss, w, h, mx, my, dx, dy, f),
        _ => olcekli8::<PREP, 128, 144>(hd, src, ss, w, h, mx, my, dx, dy, f),
    }
}

/// Bilinear dar yol (`w = N` ∈ {2, 4, 8}): xmm; sütun başına `(s[o+3], s[o+4])`
/// bayt ikilisi tek `u32`/`u64` kelimede toplanır (`N` sütun, fazlası yok),
/// `pmaddubsw` ağırlığı `(16−m, m)`, dikey `pmaddwd`. Aritmetik geniş yolla aynı.
#[target_feature(enable = "avx2")]
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
        let v = _mm_packs_epi32(lo, hi);
        if PREP {
            let o = y * N;
            match N {
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
            let o = y * hd.ds;
            match N {
                8 => store_u8x8(chunk_mut(hd.dst, o), p),
                4 => *chunk_mut::<u8, 4>(hd.dst, o) = _mm_cvtsi128_si32(p).to_le_bytes(),
                _ => {
                    let b = _mm_cvtsi128_si32(p).to_le_bytes();
                    *chunk_mut::<u8, 2>(hd.dst, o) = [b[0], b[1]];
                }
            }
        }
        my += dy;
    }
}

/// Bilinear genişlik sınıfı seçimi.
#[target_feature(enable = "avx2")]
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
        16 => olcekli2::<PREP, 16, 32>(hd, src, ss, w, h, mx, my, dx, dy),
        32 => olcekli2::<PREP, 32, 48>(hd, src, ss, w, h, mx, my, dx, dy),
        64 => olcekli2::<PREP, 64, 80>(hd, src, ss, w, h, mx, my, dx, dy),
        _ => olcekli2::<PREP, 128, 144>(hd, src, ss, w, h, mx, my, dx, dy),
    }
}

/// Bit derinliğine göre skaler gövde (P34 16 bpc gövdesinin devri de).
pub(super) fn put_skaler<const F: usize>(
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
    let f = Filter2d::from_index(F).unwrap_or(Filter2d::Regular);
    macro_rules! cagir {
        ($b:literal) => {
            if F == BILIN {
                scalar::put_bilin_scaled::<$b>(dst, ds, src, ss, w, h, mx, my, dx, dy, bd, bdmax)
            } else {
                scalar::put_8tap_scaled::<$b>(dst, ds, src, ss, w, h, mx, my, dx, dy, f, bd, bdmax)
            }
        };
    }
    match bd {
        BitDepth::Eight => cagir!(8),
        BitDepth::Ten => cagir!(10),
        BitDepth::Twelve => cagir!(12),
    }
}

pub(super) fn prep_skaler<const F: usize>(
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
    let f = Filter2d::from_index(F).unwrap_or(Filter2d::Regular);
    macro_rules! cagir {
        ($b:literal) => {
            if F == BILIN {
                scalar::prep_bilin_scaled::<$b>(tmp, src, ss, w, h, mx, my, dx, dy, bd, bdmax)
            } else {
                scalar::prep_8tap_scaled::<$b>(tmp, src, ss, w, h, mx, my, dx, dy, f, bd, bdmax)
            }
        };
    }
    match bd {
        BitDepth::Eight => cagir!(8),
        BitDepth::Ten => cagir!(10),
        BitDepth::Twelve => cagir!(12),
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
        // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
        // (`put_scaled_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır.
        // (2) sınır: `olcekli_sozlesmesi` okuma bölgesinin son baytını kaynak
        // diliminde, hedef denetimi `h` satır × `w` baytı kapsar; erişimler
        // denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
        unsafe { olcekli2_w::<false>(&mut hd, src, ss, w, h, mx, my, dx, dy) }
    } else {
        // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
        // (`put_scaled_fn`); belirteç AVX2 tespitinin kanıtıdır. (2) sınır:
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
        // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
        // (`prep_scaled_fn`); belirteç AVX2 tespitinin kanıtıdır. (2) sınır:
        // `olcekli_sozlesmesi` bilinear okuma bölgesini, `tmp.len() ≥ w·h` çıktıyı
        // kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `tmp` tek `&mut`,
        // `src` `&`.
        unsafe { olcekli2_w::<true>(&mut hd, src, ss, w, h, mx, my, dx, dy) }
    } else {
        // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
        // (`prep_scaled_fn`); belirteç AVX2 tespitinin kanıtıdır. (2) sınır:
        // `olcekli_sozlesmesi` 8-tap okuma bölgesini, `tmp.len() ≥ w·h` çıktıyı
        // kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `tmp` tek `&mut`,
        // `src` `&`.
        unsafe { olcekli8_w::<true>(&mut hd, src, ss, w, h, mx, my, dx, dy, f) }
    }
}

/// `mc_scaled[filtre]` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn put_scaled_fn(_avx2: Avx2Token, f: Filter2d) -> McPutScaledFn {
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

/// `mct_scaled[filtre]` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn prep_scaled_fn(_avx2: Avx2Token, f: Filter2d) -> McPrepScaledFn {
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
