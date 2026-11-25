//! MC ölçekli AVX2 çekirdekleri (16 bpc, 10/12 bit) — F4-MC c (P34):
//! `mc_scaled[0..9]` (`put`), `mct_scaled[0..9]` (`prep`); 9 adet 8-tap
//! birleşimi + bilinear.
//!
//! ## Sözleşme
//!
//! Kaynak dolgu hücresi (blok satırı −3, sütunu −3); pikseller `u16`, satır
//! adımları bayt. Çıktı sütunu `x`'in kaynak ofseti `(mx + x·dx) >> 10`, alt
//! piksel `((mx + x·dx) & 1023) >> 6`; çıktı satırı `y` için
//! `(my + y·dy) >> 10` ve `((my + y·dy) & 1023) >> 6`. **Okuma bölgesi** 8 bpc
//! gövdesiyle aynı piksellerdir (8-tap satır/sütun `son + 7`, bilinear
//! `3 ..= son + 4`; sütun başına 16 / 4 baytlık yükleme, aşım yok). Girişte
//! O(1) denetim ([`crate::dsp::mc::olcekli16_sozlesmesi`]): bit derinliği
//! 10/12 ve `bitdepth_max` tutarlı, `w` 2'nin kuvveti 2…128, `h` 1…128, `mx`,
//! `my` 0…1023, `dx`, `dy` 1…2048, bölgenin son pikseli kaynak diliminde,
//! hedef `h` satır × `2w` bayt (`prep` `w·h`). Sözleşme dışı girdi skalere
//! devreder ve sayılır.
//!
//! ## Yöntem
//!
//! - **Sütun önhesabı** (çağrı başına bir kez): her sütunun kaynak ofseti ve
//!   filtresi (`i16` × 8, derleme zamanı tablosu [`FILTRE16`]; alt piksel 0'da
//!   birim filtre `[0, 0, 0, 64, 0, …]`: `(64·p + r) >> (6−ib) = p << ib`,
//!   skaler "filtre yok" yoluyla aynı).
//! - **Yatay** (8-tap): sütun başına 8 piksel (16 bayt) × filtre `pmaddwd`;
//!   sütun `x` ve `x + 4` bir ymm'in iki şeridinde, `phaddd` ağacı 8 sıralı
//!   toplam; `(Σ + r) >> (6−ib)` (`ib` 10 bitte 4, 12 bitte 2; kaydırma
//!   yazmaçta). Bilinear: sütun başına piksel ikilisi `(p, p+1)` (4 bayt) ×
//!   `(16−m, m)` `pmaddwd`, `(Σ + r) >> (4−ib)`; `w ≤ 8` xmm dar yolu, halka
//!   yazmaçta.
//! - **Dikey**: 8-tap 8 satırlık `i16` halka (skalerle aynı döndürme), satır
//!   ikilileri `punpck*wd` + `pmaddwd` (katsayılar 8 bpc tablosundan); satır
//!   filtresi alt piksel 0'da birim (`put` `(64·m + r) >> (6+ib) =
//!   (m + 2^(ib−1)) >> ib`, `prep` `m`). Bilinear 2 satırlık halka. `put`
//!   `(Σ + r) >> (6+ib)` (bilinear `>> (4+ib)`), `packusdw` +
//!   `pminuw(bitdepth_max)`; `prep` `(Σ + 32) >> 6` (bilinear `(Σ + 8) >> 4`)
//!   `− 8192`, `packssdw`.
//! - **`dy = 1024` ve `dy = 2048`** sabit daldır (8 bpc gövdesi gibi): satır
//!   filtresi bir kez seçilir, her çıktı satırı tam 1 / 2 yeni ara satır üretir.
//!   Yatay ve dikey adım makroyla döngüye açılır (satır başına çağrı yok).
//!
//! `unsafe` yalnız belirteçli girişlerdeki `#[target_feature]` çağrısıdır;
//! dosya `audit::UNSAFE_EXEMPTIONS` defterinde kayıtlıdır. Yükleme/saklama
//! [`super::mem`] yardımcılarıyla yapılır.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mc_olcekli::{filtre, kat_ikili, prep_skaler, put_skaler, tablolar};
use super::mem::{
    chunk, chunk_mut, load_i16x16, load_u8x16, store_i16x8, store_i16x16, store_u8x8, store_u8x16,
    store_u8x32,
};
use crate::cpu::Avx2Token;
use crate::dsp::mc::{
    Filter2d, McPrepScaledFn, McPutScaledFn, olcekli16_sozlesmesi, skaler_devri_say,
};
use crate::dsp::mc_tables::MC_SUBPEL_FILTERS;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi32, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_cvtsi128_si64,
    _mm_madd_epi16, _mm_min_epu16, _mm_packs_epi32, _mm_packus_epi32, _mm_set1_epi16,
    _mm_set1_epi32, _mm_setr_epi32, _mm_setzero_si128, _mm_sra_epi32, _mm_sub_epi32,
    _mm_unpackhi_epi16, _mm_unpacklo_epi16, _mm_unpacklo_epi64, _mm256_add_epi32,
    _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_hadd_epi32, _mm256_madd_epi16,
    _mm256_min_epu16, _mm256_packs_epi32, _mm256_packus_epi32, _mm256_set_m128i, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_setr_epi32, _mm256_setzero_si256, _mm256_sra_epi32, _mm256_sub_epi32,
    _mm256_unpackhi_epi16, _mm256_unpacklo_epi16,
};

/// Bilinear yuva indeksi.
const BILIN: usize = 9;
/// `prep` ara değerinden çıkarılan sabit (dav1d `PREP_BIAS`, 16 bpc).
const PREP_BIAS: i32 = 8192;

/// Bütün alt piksel filtreleri `i16` × 8 (küçük sonlu 16 bayt): `[tablo][p]`,
/// `p` 0…15 (0 → birim filtre, tap 3 = 64).
static FILTRE16: [[[u8; 16]; 16]; 6] = filtre16_tablosu();

const fn filtre16_tablosu() -> [[[u8; 16]; 16]; 6] {
    let mut t = [[[0u8; 16]; 16]; 6];
    let mut k = 0;
    while k < 6 {
        let mut p = 0;
        while p < 16 {
            let mut i = 0;
            while i < 8 {
                let c: i16 = if p == 0 {
                    if i == 3 { 64 } else { 0 }
                } else {
                    MC_SUBPEL_FILTERS[k][p - 1][i] as i16
                };
                let b = c.to_le_bytes();
                t[k][p][2 * i] = b[0];
                t[k][p][2 * i + 1] = b[1];
                i += 1;
            }
            p += 1;
        }
        k += 1;
    }
    t
}

/// Çıktı hedefi: `put` (`dst`, adım bayt) ya da `prep` (`tmp`, adım `w`).
struct Hedef<'a> {
    dst: &'a mut [u8],
    tmp: &'a mut [i16],
    ds: usize,
}

/// Çağrı sabitleri: yatay ve dikey yuvarlama/kaydırma, `prep` sapması,
/// `put` üst sınırı.
struct Sabit {
    yuv_h: __m256i,
    sh_h: __m128i,
    yuv_v: __m256i,
    sh_v: __m128i,
    bias: __m256i,
    max: __m256i,
}

impl Sabit {
    /// `sh_h`/`sh_v` kaydırma, yuvarlama `(1 << sh) >> 1`.
    #[target_feature(enable = "avx2")]
    fn yeni(sh_h: i32, sh_v: i32, bitdepth_max: u16) -> Self {
        Sabit {
            yuv_h: _mm256_set1_epi32((1 << sh_h) >> 1),
            sh_h: _mm_cvtsi32_si128(sh_h),
            yuv_v: _mm256_set1_epi32((1 << sh_v) >> 1),
            sh_v: _mm_cvtsi32_si128(sh_v),
            bias: _mm256_set1_epi32(PREP_BIAS),
            max: _mm256_set1_epi16(bitdepth_max as i16),
        }
    }
}

/// Bir çıktı satırının 16 sütununu (`i32` iki yarı, şerit düzeni `[0..3 |
/// 8..11]`, `[4..7 | 12..15]`; yuvarlanmış ve kaydırılmış) yazar: `x` sütun
/// başı, `w` blok genişliği.
#[target_feature(enable = "avx2")]
#[inline]
fn sakla16<const PREP: bool>(
    hd: &mut Hedef<'_>,
    y: usize,
    x: usize,
    w: usize,
    lo: __m256i,
    hi: __m256i,
    s: &Sabit,
) {
    let n = (w - x).min(16);
    if PREP {
        let v = _mm256_packs_epi32(_mm256_sub_epi32(lo, s.bias), _mm256_sub_epi32(hi, s.bias));
        let o = y * w + x;
        let l = _mm256_castsi256_si128(v);
        match n {
            16 => store_i16x16(chunk_mut(hd.tmp, o), v),
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
    } else {
        let v = _mm256_min_epu16(_mm256_packus_epi32(lo, hi), s.max);
        let o = y * hd.ds + 2 * x;
        let l = _mm256_castsi256_si128(v);
        match n {
            16 => store_u8x32(chunk_mut(hd.dst, o), v),
            8 => store_u8x16(chunk_mut(hd.dst, o), l),
            4 => store_u8x8(chunk_mut(hd.dst, o), l),
            _ => *chunk_mut::<u8, 4>(hd.dst, o) = _mm_cvtsi128_si32(l).to_le_bytes(),
        }
    }
}

/// 8 sıralı `i32` → 8 sıralı `i16` (`[x0..x3 | x4..x7]` şerit düzeninden).
#[target_feature(enable = "avx2")]
#[inline]
fn sirala8(t: __m256i) -> __m128i {
    let p = _mm256_packs_epi32(t, t);
    _mm_unpacklo_epi64(_mm256_castsi256_si128(p), _mm256_extracti128_si256::<1>(p))
}

/// 8-tap yatay satır: `ara` ← dolgu satırı (bayt tabanı `taban`); `ofs`/`fil`
/// sütun önhesabı (`gw` 8'in katı). **Makro:** gövde çağıranın döngüsüne
/// açılır (`#[target_feature]` işlevi `#[inline(always)]` olamaz).
macro_rules! yatay16_satir {
    ($ara:expr, $src:expr, $taban:expr, $ofs:expr, $fil:expr, $gw:expr, $s:expr) => {{
        let (ara, src, taban, ofs, fil, gw, s) = ($ara, $src, $taban, $ofs, $fil, $gw, $s);
        let mut g = 0;
        while g < gw {
            let mut m = [_mm256_setzero_si256(); 4];
            for (j, mj) in m.iter_mut().enumerate() {
                // Şerit 0: sütun `g + j`, şerit 1: sütun `g + j + 4`.
                let p = _mm256_set_m128i(
                    load_u8x16(chunk(src, taban + 2 * ofs[g + j + 4])),
                    load_u8x16(chunk(src, taban + 2 * ofs[g + j])),
                );
                let f = _mm256_set_m128i(load_u8x16(&fil[g + j + 4]), load_u8x16(&fil[g + j]));
                *mj = _mm256_madd_epi16(p, f);
            }
            let t = _mm256_hadd_epi32(_mm256_hadd_epi32(m[0], m[1]), _mm256_hadd_epi32(m[2], m[3]));
            let t = _mm256_sra_epi32(_mm256_add_epi32(t, s.yuv_h), s.sh_h);
            store_i16x8(chunk_mut(ara, g), sirala8(t));
            g += 8;
        }
    }};
}

/// 8-tap dikey çıktı satırı `y`: halkanın `bas`'tan başlayan 8 satırı (en eski
/// önce), katsayı ikilileri `k`; 16 sütunluk parçalar. **Makro**.
macro_rules! dikey16_satir {
    ($prep:expr, $hd:expr, $y:expr, $w:expr, $halka:expr, $bas:expr, $k:expr, $s:expr) => {{
        let (hd, y, w, halka, bas, k, s) = (&mut *$hd, $y, $w, $halka, $bas, $k, $s);
        let mut x = 0;
        while x < w {
            let mut lo = s.yuv_v;
            let mut hi = s.yuv_v;
            for j in 0..4 {
                let a = load_i16x16(chunk(&halka[(bas + 2 * j) & 7], x));
                let b = load_i16x16(chunk(&halka[(bas + 2 * j + 1) & 7], x));
                lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), k[j]));
                hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), k[j]));
            }
            let lo = _mm256_sra_epi32(lo, s.sh_v);
            let hi = _mm256_sra_epi32(hi, s.sh_v);
            sakla16::<$prep>(hd, y, x, w, lo, hi, s);
            x += 16;
        }
    }};
}

/// 8-tap ölçekli gövde. `G`: sütun kapasitesi (`w.max(8)`'i kapsayan sınıf),
/// `R = G + 16` ara satır uzunluğu (16'lık yükleme payı).
///
/// Tek döngü (8 bpc gövdesiyle aynı yapı): çıktı satırı `y` için kaynak satırı
/// `(my + y·dy) >> 10`'a kadar halkaya yatay satır üretilir, sonra dikey satır
/// yazılır; `dy` 1024/2048'de satır filtresi bir kez seçilir ve her satır tam
/// `dy >> 10` yeni ara satır ekler.
#[target_feature(enable = "avx2")]
fn olcekli16_8<const PREP: bool, const G: usize, const R: usize>(
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
    bitdepth_max: u16,
) {
    let ib = if bitdepth_max == 4095 { 2 } else { 4 };
    let s = Sabit::yeni(6 - ib, if PREP { 6 } else { 6 + ib }, bitdepth_max);
    let (th, tv) = tablolar(f, w, h);
    let gw = w.max(8);
    let mut ofs = [0usize; G];
    let mut fil = [[0u8; 16]; G];
    for x in 0..gw {
        if x < w {
            let t = mx + x as i32 * dx;
            ofs[x] = (t >> 10) as usize;
            fil[x] = FILTRE16[th][((t & 1023) >> 6) as usize];
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
            yatay16_satir!(
                &mut halka[bas],
                src,
                (in_y + 8) as usize * ss,
                &ofs,
                &fil,
                gw,
                &s
            );
            bas = (bas + 1) & 7;
            in_y += 1;
        }
        dikey16_satir!(PREP, hd, y, w, &halka, bas, &k, &s);
        my += dy;
    }
}

/// Bilinear ölçekli gövde. `G` sütun kapasitesi (`w.max(8)`), `R = G + 16`.
#[target_feature(enable = "avx2")]
fn olcekli16_2<const PREP: bool, const G: usize, const R: usize>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
    bitdepth_max: u16,
) {
    let ib = if bitdepth_max == 4095 { 2 } else { 4 };
    let s = Sabit::yeni(4 - ib, if PREP { 4 } else { 4 + ib }, bitdepth_max);
    let gw = w.max(8);
    let mut ofs = [0usize; G];
    // `(16 − m, m)`: `pmaddwd` ağırlık ikilisi (sütun başına iki `i16`; en çok 128 sütun).
    let mut agr = [0i16; 2 * 128];
    for x in 0..gw {
        if x < w {
            let t = mx + x as i32 * dx;
            ofs[x] = (t >> 10) as usize;
            let m = ((t & 1023) >> 6) as i16;
            agr[2 * x] = 16 - m;
            agr[2 * x + 1] = m;
        } else {
            ofs[x] = ofs[w - 1];
        }
    }
    let mut halka = [[0i16; R]; 2];
    let mut in_y: i32 = -2;
    let mut my = my;
    for y in 0..h {
        let src_y = my >> 10;
        while in_y < src_y {
            // Ara satır `in_y` → dolgu satırı `in_y + 5`, sütun `ofs + 3`.
            let ara = &mut halka[(in_y & 1) as usize];
            let taban = (in_y + 5) as usize * ss + 6;
            let p = |x: usize| i32::from_le_bytes(*chunk(src, taban + 2 * ofs[x]));
            let mut g = 0;
            while g < gw {
                let v = _mm256_setr_epi32(
                    p(g),
                    p(g + 1),
                    p(g + 2),
                    p(g + 3),
                    p(g + 4),
                    p(g + 5),
                    p(g + 6),
                    p(g + 7),
                );
                let t = _mm256_madd_epi16(v, load_i16x16(chunk(&agr, 2 * g)));
                let t = _mm256_sra_epi32(_mm256_add_epi32(t, s.yuv_h), s.sh_h);
                store_i16x8(chunk_mut(ara, g), sirala8(t));
                g += 8;
            }
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
            let lo = _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), k), s.yuv_v);
            let hi = _mm256_add_epi32(_mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), k), s.yuv_v);
            let lo = _mm256_sra_epi32(lo, s.sh_v);
            let hi = _mm256_sra_epi32(hi, s.sh_v);
            sakla16::<PREP>(hd, y, x, w, lo, hi, &s);
            x += 16;
        }
        my += dy;
    }
}

/// Bilinear dar yol (`w = N` ∈ {2, 4, 8}): xmm; sütun başına piksel ikilisi
/// `(p, p+1)` (4 bayt, `u32`) `_mm_setr_epi32` ile toplanır (`N` sütun, fazlası
/// yok), ağırlık `(16−m, m)` `pmaddwd`; 2 satırlık halka yazmaçta, dikey
/// `pmaddwd`. Aritmetik geniş yolla aynı.
#[target_feature(enable = "avx2")]
fn olcekli16_2_dar<const PREP: bool, const N: usize>(
    hd: &mut Hedef<'_>,
    src: &[u8],
    ss: usize,
    h: usize,
    mx: i32,
    my: i32,
    dx: i32,
    dy: i32,
    bitdepth_max: u16,
) {
    let ib = if bitdepth_max == 4095 { 2 } else { 4 };
    let (sh_h, sh_v) = (4 - ib, if PREP { 4 } else { 4 + ib });
    let (yuv_h, shh) = (_mm_set1_epi32((1 << sh_h) >> 1), _mm_cvtsi32_si128(sh_h));
    let (yuv_v, shv) = (_mm_set1_epi32((1 << sh_v) >> 1), _mm_cvtsi32_si128(sh_v));
    let mut ofs = [0usize; 8];
    let mut agr = [0i32; 8];
    for x in 0..N {
        let t = mx + x as i32 * dx;
        ofs[x] = (t >> 10) as usize;
        let m = (t & 1023) >> 6;
        agr[x] = (16 - m) | (m << 16);
    }
    let a0 = _mm_setr_epi32(agr[0], agr[1], agr[2], agr[3]);
    let a1 = _mm_setr_epi32(agr[4], agr[5], agr[6], agr[7]);
    // Ara satır: dolgu satırı tabanından `N` sütunun ikilileri → 8 `i16` (ilk `N`'si geçerli).
    let yatay = |taban: usize| -> __m128i {
        let p = |x: usize| {
            if x < N {
                i32::from_le_bytes(*chunk(src, taban + 2 * ofs[x]))
            } else {
                0
            }
        };
        let lo = _mm_madd_epi16(_mm_setr_epi32(p(0), p(1), p(2), p(3)), a0);
        let lo = _mm_sra_epi32(_mm_add_epi32(lo, yuv_h), shh);
        if N == 8 {
            let hi = _mm_madd_epi16(_mm_setr_epi32(p(4), p(5), p(6), p(7)), a1);
            let hi = _mm_sra_epi32(_mm_add_epi32(hi, yuv_h), shh);
            _mm_packs_epi32(lo, hi)
        } else {
            _mm_packs_epi32(lo, lo)
        }
    };
    let bias = _mm_set1_epi32(PREP_BIAS);
    let max = _mm_set1_epi16(bitdepth_max as i16);
    let mut halka = [_mm_setzero_si128(); 2];
    let mut in_y: i32 = -2;
    let mut my = my;
    for y in 0..h {
        let src_y = my >> 10;
        while in_y < src_y {
            // Ara satır `in_y` → dolgu satırı `in_y + 5`, sütun `ofs + 3`.
            halka[(in_y & 1) as usize] = yatay((in_y + 5) as usize * ss + 6);
            in_y += 1;
        }
        let m = (my & 1023) >> 6;
        let k = _mm_set1_epi32((16 - m) | (m << 16));
        let (ra, rb) = (
            halka[(src_y & 1) as usize],
            halka[((src_y + 1) & 1) as usize],
        );
        let lo = _mm_madd_epi16(_mm_unpacklo_epi16(ra, rb), k);
        let lo = _mm_sra_epi32(_mm_add_epi32(lo, yuv_v), shv);
        let hi = if N == 8 {
            let hi = _mm_madd_epi16(_mm_unpackhi_epi16(ra, rb), k);
            _mm_sra_epi32(_mm_add_epi32(hi, yuv_v), shv)
        } else {
            lo
        };
        if PREP {
            let v = _mm_packs_epi32(_mm_sub_epi32(lo, bias), _mm_sub_epi32(hi, bias));
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
            let v = _mm_min_epu16(_mm_packus_epi32(lo, hi), max);
            let o = y * hd.ds;
            match N {
                8 => store_u8x16(chunk_mut(hd.dst, o), v),
                4 => store_u8x8(chunk_mut(hd.dst, o), v),
                _ => *chunk_mut::<u8, 4>(hd.dst, o) = _mm_cvtsi128_si32(v).to_le_bytes(),
            }
        }
        my += dy;
    }
}

/// Genişlik sınıfı seçimi (8-tap ya da bilinear).
#[target_feature(enable = "avx2")]
fn olcekli16_w<const PREP: bool>(
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
    bitdepth_max: u16,
) {
    let bm = bitdepth_max;
    if f.is_bilinear() {
        match w {
            ..=2 => olcekli16_2_dar::<PREP, 2>(hd, src, ss, h, mx, my, dx, dy, bm),
            4 => olcekli16_2_dar::<PREP, 4>(hd, src, ss, h, mx, my, dx, dy, bm),
            8 => olcekli16_2_dar::<PREP, 8>(hd, src, ss, h, mx, my, dx, dy, bm),
            16 => olcekli16_2::<PREP, 16, 32>(hd, src, ss, w, h, mx, my, dx, dy, bm),
            32 => olcekli16_2::<PREP, 32, 48>(hd, src, ss, w, h, mx, my, dx, dy, bm),
            64 => olcekli16_2::<PREP, 64, 80>(hd, src, ss, w, h, mx, my, dx, dy, bm),
            _ => olcekli16_2::<PREP, 128, 144>(hd, src, ss, w, h, mx, my, dx, dy, bm),
        }
    } else {
        match w {
            ..=8 => olcekli16_8::<PREP, 8, 24>(hd, src, ss, w, h, mx, my, dx, dy, f, bm),
            16 => olcekli16_8::<PREP, 16, 32>(hd, src, ss, w, h, mx, my, dx, dy, f, bm),
            32 => olcekli16_8::<PREP, 32, 48>(hd, src, ss, w, h, mx, my, dx, dy, f, bm),
            64 => olcekli16_8::<PREP, 64, 80>(hd, src, ss, w, h, mx, my, dx, dy, f, bm),
            _ => olcekli16_8::<PREP, 128, 144>(hd, src, ss, w, h, mx, my, dx, dy, f, bm),
        }
    }
}

fn put16_giris<const F: usize>(
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
    let ok = olcekli16_sozlesmesi(F == BILIN, src.len(), ss, w, h, mx, my, dx, dy, bd, bdmax)
        && ds >= 2 * w
        && dst.len() >= (h - 1) * ds + 2 * w;
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`put_scaled_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `olcekli16_sozlesmesi` okuma bölgesinin son pikselini kaynak diliminde, hedef
    // denetimi `h` satır × `2w` baytı kapsar; erişimler denetimlidir. (3)
    // hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
    unsafe { olcekli16_w::<false>(&mut hd, src, ss, w, h, mx, my, dx, dy, f, bdmax) }
}

fn prep16_giris<const F: usize>(
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
    let ok = olcekli16_sozlesmesi(F == BILIN, src.len(), ss, w, h, mx, my, dx, dy, bd, bdmax)
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`prep_scaled_fn`); belirteç AVX2 tespitinin kanıtıdır. (2) sınır:
    // `olcekli16_sozlesmesi` okuma bölgesini, `tmp.len() ≥ w·h` çıktıyı kapsar;
    // erişimler denetimlidir. (3) hizalama/takma ad: `tmp` tek `&mut`, `src` `&`.
    unsafe { olcekli16_w::<true>(&mut hd, src, ss, w, h, mx, my, dx, dy, f, bdmax) }
}

/// 16 bpc `mc_scaled[filtre]` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn put_scaled_fn(_avx2: Avx2Token, f: Filter2d) -> McPutScaledFn {
    match f.index() {
        0 => put16_giris::<0>,
        1 => put16_giris::<1>,
        2 => put16_giris::<2>,
        3 => put16_giris::<3>,
        4 => put16_giris::<4>,
        5 => put16_giris::<5>,
        6 => put16_giris::<6>,
        7 => put16_giris::<7>,
        8 => put16_giris::<8>,
        _ => put16_giris::<BILIN>,
    }
}

/// 16 bpc `mct_scaled[filtre]` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn prep_scaled_fn(_avx2: Avx2Token, f: Filter2d) -> McPrepScaledFn {
    match f.index() {
        0 => prep16_giris::<0>,
        1 => prep16_giris::<1>,
        2 => prep16_giris::<2>,
        3 => prep16_giris::<3>,
        4 => prep16_giris::<4>,
        5 => prep16_giris::<5>,
        6 => prep16_giris::<6>,
        7 => prep16_giris::<7>,
        8 => prep16_giris::<8>,
        _ => prep16_giris::<BILIN>,
    }
}
