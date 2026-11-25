//! MC ölçekli SSSE3 çekirdekleri (16 bpc, 10/12 bit) — F4-MC d (P35):
//! `mc_scaled[0..9]` (`put`), `mct_scaled[0..9]` (`prep`); 9 adet 8-tap
//! birleşimi + bilinear.
//!
//! ## Sözleşme
//!
//! AVX2 gövdeleriyle ([`super::mc16_olcekli`]) aynı: okuma bölgesi 8 bpc
//! gövdesiyle aynı pikseller (sütun başına 16 / 4 baytlık yükleme, aşım yok),
//! girişte O(1) denetim ([`crate::dsp::mc::olcekli16_sozlesmesi`]). Sözleşme
//! dışı girdi skalere devreder ve sayılır.
//!
//! ## Yöntem
//!
//! Aritmetik ve yapı AVX2 gövdesiyle aynıdır; yazmaç 128 bittir.
//!
//! - **Yatay** (8-tap): sütun başına 8 piksel (16 bayt) × filtre (`i16` × 8,
//!   derleme zamanı tablosu; alt piksel 0'da birim) `pmaddwd`; `phaddd` ağacı
//!   sıralı `[x0..x3]`, `[x4..x7]`; `(Σ + r) >> (6−ib)` (kaydırma yazmaçta).
//!   Bilinear: sütun başına piksel ikilisi `(p, p+1)` × `(16−m, m)` `pmaddwd`,
//!   `(Σ + r) >> (4−ib)`; `w ≤ 8` dar yolu, halka yazmaçta.
//! - **Dikey**: 8-tap 8 satırlık `i16` halka, satır ikilileri `punpck*wd` +
//!   `pmaddwd` (katsayılar 8 bpc tablosundan), 8 sütunluk parçalar; bilinear 2
//!   satırlık halka. `put` kırpması `packssdw` + `pmaxsw(0)` +
//!   `pminsw(bitdepth_max)` (`packusdw`/`pminuw` SSE4.1 yerine; `i32` sonuç
//!   `> 32767` iken iki yol da `bitdepth_max`, `< 0` iken 0 verir); `prep`
//!   `− 8192`, `packssdw`. `dy` 1024/2048 sabit dallar.
//!
//! Yatay ve dikey adım makroyla döngüye açılır (satır başına çağrı yok). SSE4.1
//! ve üstü komut yoktur.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mc_olcekli::{filtre, kat_ikili, prep_skaler, put_skaler, tablolar};
use super::mem::{chunk, chunk_mut, load_i16x8, load_u8x16, store_i16x8, store_u8x8, store_u8x16};
use crate::cpu::Ssse3Token;
use crate::dsp::mc::{
    Filter2d, McPrepScaledFn, McPutScaledFn, olcekli16_sozlesmesi, skaler_devri_say,
};
use crate::dsp::mc_tables::MC_SUBPEL_FILTERS;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_cvtsi32_si128, _mm_cvtsi128_si32, _mm_cvtsi128_si64,
    _mm_hadd_epi32, _mm_madd_epi16, _mm_max_epi16, _mm_min_epi16, _mm_packs_epi32, _mm_set1_epi16,
    _mm_set1_epi32, _mm_setr_epi32, _mm_setzero_si128, _mm_sra_epi32, _mm_sub_epi32,
    _mm_unpackhi_epi16, _mm_unpacklo_epi16,
};

/// Bilinear yuva indeksi.
const BILIN: usize = 9;
/// `prep` ara değerinden çıkarılan sabit (dav1d `PREP_BIAS`, 16 bpc).
const PREP_BIAS: i32 = 8192;

/// Bütün alt piksel filtreleri `i16` × 8 (küçük sonlu 16 bayt): `[tablo][p]`,
/// `p` 0…15 (0 → birim filtre, tap 3 = 64); AVX2 gövdesinin tablosuyla aynı,
/// kendi kopyası.
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
    yuv_h: __m128i,
    sh_h: __m128i,
    yuv_v: __m128i,
    sh_v: __m128i,
    bias: __m128i,
    max: __m128i,
}

impl Sabit {
    /// `sh_h`/`sh_v` kaydırma, yuvarlama `(1 << sh) >> 1`.
    #[target_feature(enable = "ssse3")]
    fn yeni(sh_h: i32, sh_v: i32, bitdepth_max: u16) -> Self {
        Sabit {
            yuv_h: _mm_set1_epi32((1 << sh_h) >> 1),
            sh_h: _mm_cvtsi32_si128(sh_h),
            yuv_v: _mm_set1_epi32((1 << sh_v) >> 1),
            sh_v: _mm_cvtsi32_si128(sh_v),
            bias: _mm_set1_epi32(PREP_BIAS),
            max: _mm_set1_epi16(bitdepth_max as i16),
        }
    }
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
    s: &Sabit,
) {
    let n = (w - x).min(8);
    if PREP {
        let v = _mm_packs_epi32(_mm_sub_epi32(lo, s.bias), _mm_sub_epi32(hi, s.bias));
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
        let v = _mm_max_epi16(_mm_packs_epi32(lo, hi), _mm_setzero_si128());
        let v = _mm_min_epi16(v, s.max);
        let o = y * hd.ds + 2 * x;
        match n {
            8 => store_u8x16(chunk_mut(hd.dst, o), v),
            4 => store_u8x8(chunk_mut(hd.dst, o), v),
            _ => *chunk_mut::<u8, 4>(hd.dst, o) = _mm_cvtsi128_si32(v).to_le_bytes(),
        }
    }
}

/// 8-tap yatay satır: `ara` ← dolgu satırı (bayt tabanı `taban`); `ofs`/`fil`
/// sütun önhesabı (`gw` 8'in katı). **Makro:** gövde çağıranın döngüsüne
/// açılır (AVX2 gövdesi gibi).
macro_rules! yatay16_satir {
    ($ara:expr, $src:expr, $taban:expr, $ofs:expr, $fil:expr, $gw:expr, $s:expr) => {{
        let (ara, src, taban, ofs, fil, gw, s) = ($ara, $src, $taban, $ofs, $fil, $gw, $s);
        let mut g = 0;
        while g < gw {
            let mut m = [_mm_setzero_si128(); 8];
            for (j, mj) in m.iter_mut().enumerate() {
                let p = load_u8x16(chunk(src, taban + 2 * ofs[g + j]));
                *mj = _mm_madd_epi16(p, load_u8x16(&fil[g + j]));
            }
            let lo = _mm_hadd_epi32(_mm_hadd_epi32(m[0], m[1]), _mm_hadd_epi32(m[2], m[3]));
            let hi = _mm_hadd_epi32(_mm_hadd_epi32(m[4], m[5]), _mm_hadd_epi32(m[6], m[7]));
            let lo = _mm_sra_epi32(_mm_add_epi32(lo, s.yuv_h), s.sh_h);
            let hi = _mm_sra_epi32(_mm_add_epi32(hi, s.yuv_h), s.sh_h);
            store_i16x8(chunk_mut(ara, g), _mm_packs_epi32(lo, hi));
            g += 8;
        }
    }};
}

/// Dar 8-tap yatay satır (`w` 2 ya da 4): yalnız `n` sütun hesaplanır; ara
/// satırın kalan sütunları kullanılmaz. **Makro**.
macro_rules! yatay16_dar_satir {
    ($ara:expr, $src:expr, $taban:expr, $ofs:expr, $fil:expr, $n:expr, $s:expr) => {{
        let (ara, src, taban, ofs, fil, n, s) = ($ara, $src, $taban, $ofs, $fil, $n, $s);
        let m = |j: usize| {
            _mm_madd_epi16(
                load_u8x16(chunk(src, taban + 2 * ofs[j])),
                load_u8x16(&fil[j]),
            )
        };
        let t01 = _mm_hadd_epi32(m(0), m(1));
        let t = if n == 4 {
            _mm_hadd_epi32(t01, _mm_hadd_epi32(m(2), m(3)))
        } else {
            _mm_hadd_epi32(t01, t01)
        };
        let t = _mm_sra_epi32(_mm_add_epi32(t, s.yuv_h), s.sh_h);
        store_i16x8(chunk_mut(ara, 0), _mm_packs_epi32(t, t));
    }};
}

/// 8-tap dikey çıktı satırı `y`: halkanın `bas`'tan başlayan 8 satırı (en eski
/// önce), katsayı ikilileri `k`; 8 sütunluk parçalar. **Makro**.
macro_rules! dikey16_satir {
    ($prep:expr, $hd:expr, $y:expr, $w:expr, $halka:expr, $bas:expr, $k:expr, $s:expr) => {{
        let (hd, y, w, halka, bas, k, s) = (&mut *$hd, $y, $w, $halka, $bas, $k, $s);
        let mut x = 0;
        while x < w {
            let mut lo = s.yuv_v;
            let mut hi = s.yuv_v;
            for j in 0..4 {
                let a = load_i16x8(chunk(&halka[(bas + 2 * j) & 7], x));
                let b = load_i16x8(chunk(&halka[(bas + 2 * j + 1) & 7], x));
                lo = _mm_add_epi32(lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), k[j]));
                hi = _mm_add_epi32(hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), k[j]));
            }
            let lo = _mm_sra_epi32(lo, s.sh_v);
            let hi = _mm_sra_epi32(hi, s.sh_v);
            sakla8::<$prep>(hd, y, x, w, lo, hi, s);
            x += 8;
        }
    }};
}

/// 8-tap ölçekli gövde. `G`: sütun kapasitesi ve ara satır uzunluğu
/// (`w.max(8)`'i kapsayan sınıf; yüklemeler 8'lik, `G`'yi aşmaz).
#[target_feature(enable = "ssse3")]
fn olcekli16_8<const PREP: bool, const G: usize>(
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
                yatay16_dar_satir!(
                    &mut halka[bas],
                    src,
                    (in_y + 8) as usize * ss,
                    &ofs,
                    &fil,
                    w,
                    &s
                );
            } else {
                yatay16_satir!(
                    &mut halka[bas],
                    src,
                    (in_y + 8) as usize * ss,
                    &ofs,
                    &fil,
                    gw,
                    &s
                );
            }
            bas = (bas + 1) & 7;
            in_y += 1;
        }
        dikey16_satir!(PREP, hd, y, w, &halka, bas, &k, &s);
        my += dy;
    }
}

/// Bilinear ölçekli gövde (`w ≥ 16`). `G` sütun kapasitesi ve ara satır
/// uzunluğu (`w`).
#[target_feature(enable = "ssse3")]
fn olcekli16_2<const PREP: bool, const G: usize>(
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
    let gw = w;
    let mut ofs = [0usize; G];
    // `(16 − m, m)`: `pmaddwd` ağırlık ikilisi (sütun başına iki `i16`; en çok 128 sütun).
    let mut agr = [0i16; 2 * 128];
    for x in 0..gw {
        let t = mx + x as i32 * dx;
        ofs[x] = (t >> 10) as usize;
        let m = ((t & 1023) >> 6) as i16;
        agr[2 * x] = 16 - m;
        agr[2 * x + 1] = m;
    }
    let mut halka = [[0i16; G]; 2];
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
                let v0 = _mm_setr_epi32(p(g), p(g + 1), p(g + 2), p(g + 3));
                let v1 = _mm_setr_epi32(p(g + 4), p(g + 5), p(g + 6), p(g + 7));
                let lo = _mm_madd_epi16(v0, load_i16x8(chunk(&agr, 2 * g)));
                let hi = _mm_madd_epi16(v1, load_i16x8(chunk(&agr, 2 * g + 8)));
                let lo = _mm_sra_epi32(_mm_add_epi32(lo, s.yuv_h), s.sh_h);
                let hi = _mm_sra_epi32(_mm_add_epi32(hi, s.yuv_h), s.sh_h);
                store_i16x8(chunk_mut(ara, g), _mm_packs_epi32(lo, hi));
                g += 8;
            }
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
            let lo = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(a, b), k), s.yuv_v);
            let hi = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(a, b), k), s.yuv_v);
            let lo = _mm_sra_epi32(lo, s.sh_v);
            let hi = _mm_sra_epi32(hi, s.sh_v);
            sakla8::<PREP>(hd, y, x, w, lo, hi, &s);
            x += 8;
        }
        my += dy;
    }
}

/// Bilinear dar yol (`w = N` ∈ {2, 4, 8}): sütun başına piksel ikilisi
/// `(p, p+1)` (4 bayt, `u32`) `_mm_setr_epi32` ile toplanır (`N` sütun, fazlası
/// yok), ağırlık `(16−m, m)` `pmaddwd`; 2 satırlık halka yazmaçta, dikey
/// `pmaddwd`. AVX2 gövdesinin dar yoluyla aynı.
#[target_feature(enable = "ssse3")]
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
    let s = Sabit::yeni(4 - ib, if PREP { 4 } else { 4 + ib }, bitdepth_max);
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
        let lo = _mm_sra_epi32(_mm_add_epi32(lo, s.yuv_h), s.sh_h);
        if N == 8 {
            let hi = _mm_madd_epi16(_mm_setr_epi32(p(4), p(5), p(6), p(7)), a1);
            let hi = _mm_sra_epi32(_mm_add_epi32(hi, s.yuv_h), s.sh_h);
            _mm_packs_epi32(lo, hi)
        } else {
            _mm_packs_epi32(lo, lo)
        }
    };
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
        let lo = _mm_sra_epi32(_mm_add_epi32(lo, s.yuv_v), s.sh_v);
        let hi = if N == 8 {
            let hi = _mm_madd_epi16(_mm_unpackhi_epi16(ra, rb), k);
            _mm_sra_epi32(_mm_add_epi32(hi, s.yuv_v), s.sh_v)
        } else {
            lo
        };
        sakla8::<PREP>(hd, y, 0, N, lo, hi, &s);
        my += dy;
    }
}

/// Genişlik sınıfı seçimi (8-tap ya da bilinear).
#[target_feature(enable = "ssse3")]
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
            16 => olcekli16_2::<PREP, 16>(hd, src, ss, w, h, mx, my, dx, dy, bm),
            32 => olcekli16_2::<PREP, 32>(hd, src, ss, w, h, mx, my, dx, dy, bm),
            64 => olcekli16_2::<PREP, 64>(hd, src, ss, w, h, mx, my, dx, dy, bm),
            _ => olcekli16_2::<PREP, 128>(hd, src, ss, w, h, mx, my, dx, dy, bm),
        }
    } else {
        match w {
            ..=8 => olcekli16_8::<PREP, 8>(hd, src, ss, w, h, mx, my, dx, dy, f, bm),
            16 => olcekli16_8::<PREP, 16>(hd, src, ss, w, h, mx, my, dx, dy, f, bm),
            32 => olcekli16_8::<PREP, 32>(hd, src, ss, w, h, mx, my, dx, dy, f, bm),
            64 => olcekli16_8::<PREP, 64>(hd, src, ss, w, h, mx, my, dx, dy, f, bm),
            _ => olcekli16_8::<PREP, 128>(hd, src, ss, w, h, mx, my, dx, dy, f, bm),
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`put_scaled_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır. (2) sınır:
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`prep_scaled_fn`); belirteç SSSE3 tespitinin kanıtıdır. (2) sınır:
    // `olcekli16_sozlesmesi` okuma bölgesini, `tmp.len() ≥ w·h` çıktıyı kapsar;
    // erişimler denetimlidir. (3) hizalama/takma ad: `tmp` tek `&mut`, `src` `&`.
    unsafe { olcekli16_w::<true>(&mut hd, src, ss, w, h, mx, my, dx, dy, f, bdmax) }
}

/// 16 bpc `mc_scaled[filtre]` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn put_scaled_fn(_ssse3: Ssse3Token, f: Filter2d) -> McPutScaledFn {
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

/// 16 bpc `mct_scaled[filtre]` SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn prep_scaled_fn(_ssse3: Ssse3Token, f: Filter2d) -> McPrepScaledFn {
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
