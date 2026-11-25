//! Film grain 16 bpc (`10`/`12` bit) x86_64 SSSE3 çekirdekleri — F4-FG c (P42 Adım 1).
//!
//! Bu modül dav1d'nin `src/x86/filmgrain16_sse.asm` dosyasındaki 8 SSSE3 sembolünü
//! (`10` ve `12` bpc = 16 yuva) **saf Rust intrinsics** ile gerçekler; dış `.asm`/`.S`,
//! `asm!` ve `global_asm!` yoktur (CEK-02). Yalnız SSE2 ve SSSE3 yönergeleri
//! kullanılır (`#[target_feature(enable = "ssse3")]`, [`Ssse3Token`] kanıtı);
//! SSE4.1+ ve VEX yönergesi üretilmez.
//!
//! ## P34 modül izolasyonu
//!
//! 8 bpc modülleriyle (`fg.rs`, `fg_ssse3.rs`, `fg_avx512.rs`) sıcak yardımcı
//! paylaşılmaz; 8 bpc CGU bölümlemesi ve makine kodu değişmez.
//!
//! ## 16 bpc aritmetik kuralları (dav1d `filmgrain16_sse.asm` uyumu)
//!
//! 1. `generate_grain_*`: `12` bpc ve `grain_scale_shift == 0` iken
//!    `shift = 4 − min_8 + grain_scale_shift = 0` olur; `1 << (15 − shift)`
//!    `i16` taşıracağından `shift == 0` dalında Gauss örneği doğrudan yazılır.
//! 2. `fgy` / `fguv` gürültü hesabı: `10`/`12` bpc'de `scaling · grain` çarpımı
//!    (`255 · 2048 = 522 240`) `i16`'ya sığmaz. Önce `scaling` (`0..=255`)
//!    `1 << (15 − scaling_shift)` (`scaling_shift ∈ 8..=11` → `≤ 255 · 128 = 32 640`)
//!    ile `i16` içinde sola kaydırılır, ardından `_mm_mulhrs_epi16(sc_shifted, grain)`
//!    uygulanır (`round2(scaling · grain, scaling_shift)` ile bit-özdeş).
//! 3. Örtüşme (`overlap`) harmanlaması: `12` bpc'de `|old · 27 + new · 17| ≤ 90 112`
//!    `i16`'yı taşırır; `_mm_madd_epi16` (`pmaddwd`) ile `i32` içinde toplanıp
//!    `+ 16`, `>> 5` (`_mm_srai_epi32::<5>`) ve `_mm_packs_epi32` + `[grain_min, grain_max]`
//!    kırpması uygulanır.
//!
//! ## Sembol eşlemesi (dav1d @ `99f3354`)
//!
//! | ID | İşlev | dav1d sembolü |
//! |---|---|---|
//! | D01589 | `generate_grain_y` | `dav1d_generate_grain_y_16bpc_ssse3` |
//! | D01577 | `generate_grain_uv[I420]` | `dav1d_generate_grain_uv_420_16bpc_ssse3` |
//! | D01581 | `generate_grain_uv[I422]` | `dav1d_generate_grain_uv_422_16bpc_ssse3` |
//! | D01585 | `generate_grain_uv[I444]` | `dav1d_generate_grain_uv_444_16bpc_ssse3` |
//! | D01573 | `fgy_32x32xn` | `dav1d_fgy_32x32xn_16bpc_ssse3` |
//! | D01555 | `fguv_32x32xn[I420]` | `dav1d_fguv_32x32xn_i420_16bpc_ssse3` |
//! | D01561 | `fguv_32x32xn[I422]` | `dav1d_fguv_32x32xn_i422_16bpc_ssse3` |
//! | D01567 | `fguv_32x32xn[I444]` | `dav1d_fguv_32x32xn_i444_16bpc_ssse3` |

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use core::arch::x86_64::*;

use super::mem::{chunk, chunk_mut, load_i16x8, load_u8x16, store_i16x8, store_u8x16};
use crate::bitstream::frame_header::FilmGrainData;
use crate::cpu::Ssse3Token;
use crate::dsp::fg::{
    FG_BLOCK_SIZE, FgBitDepth, FguvFn, FgyFn, GRAIN_HEIGHT, GRAIN_WIDTH, GenerateGrainUvFn,
    GenerateGrainYFn, GrainLut, SUB_GRAIN_HEIGHT, SUB_GRAIN_WIDTH, fg_skaler_devri_say,
    fguv16_sozlesmesi, fgy16_sozlesmesi,
};
use crate::dsp::fg_tables::GAUSSIAN_SEQUENCE;
use crate::dsp::scalar::fg::{self as scalar, get_random_number, round2, row_seeds, sample_lut};

/// AR süzgeci yarıçapı (`src/filmgrain_tmpl.c:66`: `ar_pad = 3`).
const AR_PAD: usize = 3;

/// `generate_grain_*` 16 bpc girişlerinin O(1) parametre sözleşmesi.
#[inline]
fn ar16_sozlesmesi(data: &FilmGrainData, bd: FgBitDepth) -> bool {
    matches!(bd, FgBitDepth::B10 | FgBitDepth::B12)
        && (0..=3).contains(&data.ar_coeff_lag)
        && (0..=3).contains(&data.grain_scale_shift)
        && (6..=9).contains(&data.ar_coeff_shift)
        && data.seed < 0x1_0000
}

/// `fguv` kroma ölçekleme katsayılarının AV1 sözleşmesi (`i16` taşmasızlığı).
#[inline]
fn uv_katsayi_sozlesmesi(data: &FilmGrainData, uv: usize) -> bool {
    data.chroma_scaling_from_luma
        || ((-128..=127).contains(&data.uv_luma_mult[uv])
            && (-128..=127).contains(&data.uv_mult[uv])
            && (-256..=255).contains(&data.uv_offset[uv]))
}

/// 16 bit LFSR Gauss dizisi üreteci (`get_random_number(11)`).
struct GaussUreteci {
    tohum: u32,
}

impl GaussUreteci {
    #[inline(always)]
    fn sonraki(&mut self) -> usize {
        get_random_number(11, &mut self.tohum) as usize
    }
}

/// `W × H` Gauss örneklerini üretir ve `shift = (12 − BPC) + grain_scale_shift`
/// ile yuvarlayarak `buf`'un ilk `H` satırının ilk `W` sütununa yazar.
///
/// Kural 1: `12` bpc ve `grain_scale_shift == 0` iken `shift == 0` olur;
/// `1 << (15 − shift)` `i16` taşıracağından `shift == 0` dalında Gauss örnekleri
/// doğrudan yazılır (`round2(x, 0) = x`).
#[target_feature(enable = "ssse3")]
fn gauss_doldur_16<const BPC: u32, const W: usize, const H: usize>(
    buf: &mut GrainLut,
    tohum: u32,
    grain_scale_shift: i32,
) {
    let mut g = GaussUreteci { tohum };
    let shift = (12 - BPC) + grain_scale_shift as u32;
    if shift == 0 {
        for row in buf.iter_mut().take(H) {
            for x in 0..W {
                row[x] = GAUSSIAN_SEQUENCE[g.sonraki()];
            }
        }
        return;
    }
    let carpan = _mm_set1_epi16(1 << (15 - shift));
    let simd_son = (W / 8) * 8;
    for row in buf.iter_mut().take(H) {
        let mut x = 0usize;
        while x < simd_son {
            let v = _mm_setr_epi16(
                GAUSSIAN_SEQUENCE[g.sonraki()],
                GAUSSIAN_SEQUENCE[g.sonraki()],
                GAUSSIAN_SEQUENCE[g.sonraki()],
                GAUSSIAN_SEQUENCE[g.sonraki()],
                GAUSSIAN_SEQUENCE[g.sonraki()],
                GAUSSIAN_SEQUENCE[g.sonraki()],
                GAUSSIAN_SEQUENCE[g.sonraki()],
                GAUSSIAN_SEQUENCE[g.sonraki()],
            );
            let yuv = _mm_mulhrs_epi16(v, carpan);
            store_i16x8(chunk_mut(row, x), yuv);
            x += 8;
        }
        while x < W {
            row[x] = round2(i32::from(GAUSSIAN_SEQUENCE[g.sonraki()]), shift) as i16;
            x += 1;
        }
    }
}

/// 4 × `i32` şeritli birikim çiftini (`x..x + 4`, `x + 4..x + 8`) `[i32; 8]` dizisine açar.
#[target_feature(enable = "ssse3")]
#[inline]
fn cikar_i32x8(lo: __m128i, hi: __m128i) -> [i32; 8] {
    let lo01 = _mm_cvtsi128_si64(lo) as u64;
    let lo23 = _mm_cvtsi128_si64(_mm_srli_si128::<8>(lo)) as u64;
    let hi01 = _mm_cvtsi128_si64(hi) as u64;
    let hi23 = _mm_cvtsi128_si64(_mm_srli_si128::<8>(hi)) as u64;
    [
        lo01 as i32,
        (lo01 >> 32) as i32,
        lo23 as i32,
        (lo23 >> 32) as i32,
        hi01 as i32,
        (hi01 >> 32) as i32,
        hi23 as i32,
        (hi23 >> 32) as i32,
    ]
}

/// İki komşu dizisini (`a`, `b`) katsayı çiftiyle (`c0`, `c1`) çarparak `(lo, hi)`
/// 32 bit birikimlerine ekler (`_mm_madd_epi16` = `pmaddwd`).
#[target_feature(enable = "ssse3")]
#[inline]
fn biriktir_cift(a: __m128i, b: __m128i, c0: i8, c1: i8, lo: &mut __m128i, hi: &mut __m128i) {
    let ck = (i32::from(c0) & 0xFFFF) | (i32::from(c1) << 16);
    let vc = _mm_set1_epi32(ck);
    let p_lo = _mm_unpacklo_epi16(a, b);
    let p_hi = _mm_unpackhi_epi16(a, b);
    *lo = _mm_add_epi32(*lo, _mm_madd_epi16(p_lo, vc));
    *hi = _mm_add_epi32(*hi, _mm_madd_epi16(p_hi, vc));
}

/// Tek bir komşu dizisini (`a`) tek katsayıyla (`c0`) çarparak `(lo, hi)`
/// birikimlerine ekler.
#[target_feature(enable = "ssse3")]
#[inline]
fn biriktir_tek(a: __m128i, c0: i8, lo: &mut __m128i, hi: &mut __m128i) {
    biriktir_cift(a, _mm_setzero_si128(), c0, 0, lo, hi);
}

/// Luma grain LUT'undan `(((y − 3) << SUBY) + 3, ((x − 3) << SUBX) + 3)` konumundaki
/// 8 kroma örneğine karşılık gelen `(1 + SUBX) × (1 + SUBY)` ortalamalı luma grain
/// vektörünü (`8 × i16`) hesaplar (`12` bpc 4:2:0 toplamı `[-8192, 8188]` `i16`'ya sığar).
#[target_feature(enable = "ssse3")]
#[inline]
fn luma_ort_8<const SUBX: usize, const SUBY: usize>(
    buf_y: &GrainLut,
    y: usize,
    x: usize,
) -> __m128i {
    let ly = ((y - AR_PAD) << SUBY) + AR_PAD;
    let lx = ((x - AR_PAD) << SUBX) + AR_PAD;
    if SUBX == 0 && SUBY == 0 {
        load_i16x8(chunk(&buf_y[ly], lx))
    } else if SUBX == 1 && SUBY == 0 {
        let r0_0 = load_i16x8(chunk(&buf_y[ly], lx));
        let r0_1 = load_i16x8(chunk(&buf_y[ly], lx + 8));
        let toplam = _mm_hadd_epi16(r0_0, r0_1);
        _mm_srai_epi16::<1>(_mm_add_epi16(toplam, _mm_set1_epi16(1)))
    } else {
        let r0_0 = load_i16x8(chunk(&buf_y[ly], lx));
        let r0_1 = load_i16x8(chunk(&buf_y[ly], lx + 8));
        let r1_0 = load_i16x8(chunk(&buf_y[ly + 1], lx));
        let r1_1 = load_i16x8(chunk(&buf_y[ly + 1], lx + 8));
        let s0 = _mm_hadd_epi16(r0_0, r0_1);
        let s1 = _mm_hadd_epi16(r1_0, r1_1);
        let toplam = _mm_add_epi16(s0, s1);
        _mm_srai_epi16::<2>(_mm_add_epi16(toplam, _mm_set1_epi16(2)))
    }
}

/// `LAG ∈ {1, 2, 3}` için AR süzgeci: üst satırların (`yy < y`) katkısını 8'li
/// SSSE3 `_mm_madd_epi16` şeritleriyle biriktirir, aynı satırın (`yy == y`)
/// ardışık bağımlı `LAG` sol komşusunu skaler tamamlar.
#[target_feature(enable = "ssse3")]
fn ar_suz_lag_16<
    const BPC: u32,
    const W: usize,
    const H: usize,
    const LAG: usize,
    const SUBX: usize,
    const SUBY: usize,
>(
    buf: &mut GrainLut,
    buf_y: Option<&GrainLut>,
    coeffs: &[i8],
    luma_katsayi: i8,
    ar_shift: u32,
) {
    let gmin = -(1i32 << (BPC - 1));
    let gmax = (1i32 << (BPC - 1)) - 1;
    let satir_gen = 2 * LAG + 1;
    let ust_sayisi = LAG * satir_gen;
    let x_son = W - AR_PAD;
    let simd_son = AR_PAD + ((x_son - AR_PAD) / 8) * 8;

    for y in AR_PAD..H {
        let mut x = AR_PAD;
        while x < simd_son {
            let mut lo = _mm_setzero_si128();
            let mut hi = _mm_setzero_si128();
            let mut ci = 0usize;
            for dy in 0..LAG {
                let yy = y - LAG + dy;
                let row = &buf[yy];
                let mut dx = 0usize;
                while dx + 1 < satir_gen {
                    let a = load_i16x8(chunk(row, x - LAG + dx));
                    let b = load_i16x8(chunk(row, x - LAG + dx + 1));
                    biriktir_cift(a, b, coeffs[ci], coeffs[ci + 1], &mut lo, &mut hi);
                    ci += 2;
                    dx += 2;
                }
                let a = load_i16x8(chunk(row, x + LAG));
                if let Some(by) = buf_y.filter(|_| dy + 1 == LAG && luma_katsayi != 0) {
                    let l = luma_ort_8::<SUBX, SUBY>(by, y, x);
                    biriktir_cift(a, l, coeffs[ci], luma_katsayi, &mut lo, &mut hi);
                } else {
                    biriktir_tek(a, coeffs[ci], &mut lo, &mut hi);
                }
                ci += 1;
            }
            let ust = cikar_i32x8(lo, hi);
            let row = &mut buf[y];
            for (i, &ust_toplam) in ust.iter().enumerate() {
                let xx = x + i;
                let mut toplam = ust_toplam;
                for dx in 0..LAG {
                    toplam += i32::from(coeffs[ust_sayisi + dx]) * i32::from(row[xx - LAG + dx]);
                }
                let g = i32::from(row[xx]) + round2(toplam, ar_shift);
                row[xx] = g.clamp(gmin, gmax) as i16;
            }
            x += 8;
        }
        while x < x_son {
            let mut toplam = 0i32;
            let mut ci = 0usize;
            for dy in 0..LAG {
                let yy = y - LAG + dy;
                for dx in 0..satir_gen {
                    toplam += i32::from(coeffs[ci]) * i32::from(buf[yy][x - LAG + dx]);
                    ci += 1;
                }
            }
            if let Some(by) = buf_y.filter(|_| luma_katsayi != 0) {
                let mut luma = 0i32;
                let luma_y = ((y - AR_PAD) << SUBY) + AR_PAD;
                let luma_x = ((x - AR_PAD) << SUBX) + AR_PAD;
                for i in 0..=SUBY {
                    for j in 0..=SUBX {
                        luma += i32::from(by[luma_y + i][luma_x + j]);
                    }
                }
                let luma = round2(luma, (SUBX + SUBY) as u32);
                toplam += i32::from(luma_katsayi) * luma;
            }
            for dx in 0..LAG {
                toplam += i32::from(coeffs[ust_sayisi + dx]) * i32::from(buf[y][x - LAG + dx]);
            }
            let g = i32::from(buf[y][x]) + round2(toplam, ar_shift);
            buf[y][x] = g.clamp(gmin, gmax) as i16;
            x += 1;
        }
    }
}

/// `LAG == 0` kroma durumu (`num_y_points > 0`): yalnız luma grain katkısını
/// 8'li `_mm_madd_epi16` (32 bit çarpım, `12` bpc'de `2048 · 128` taşmasız) ile uygular.
#[target_feature(enable = "ssse3")]
fn ar_suz_lag0_16<
    const BPC: u32,
    const W: usize,
    const H: usize,
    const SUBX: usize,
    const SUBY: usize,
>(
    buf: &mut GrainLut,
    buf_y: &GrainLut,
    luma_katsayi: i8,
    ar_shift: u32,
) {
    if luma_katsayi == 0 {
        return;
    }
    let gmin = -(1i32 << (BPC - 1));
    let gmax = (1i32 << (BPC - 1)) - 1;
    let x_son = W - AR_PAD;
    let simd_son = AR_PAD + ((x_son - AR_PAD) / 8) * 8;
    let vc32 = _mm_set1_epi32(i32::from(luma_katsayi) & 0xFFFF);
    let vyuv32 = _mm_set1_epi32((1 << ar_shift) >> 1);
    let vshift = _mm_cvtsi32_si128(ar_shift as i32);
    let vmin = _mm_set1_epi16(gmin as i16);
    let vmax = _mm_set1_epi16(gmax as i16);
    let sifir = _mm_setzero_si128();

    for y in AR_PAD..H {
        let mut x = AR_PAD;
        while x < simd_son {
            let l = luma_ort_8::<SUBX, SUBY>(buf_y, y, x);
            let p_lo = _mm_madd_epi16(_mm_unpacklo_epi16(l, sifir), vc32);
            let p_hi = _mm_madd_epi16(_mm_unpackhi_epi16(l, sifir), vc32);
            let d_lo = _mm_sra_epi32(_mm_add_epi32(p_lo, vyuv32), vshift);
            let d_hi = _mm_sra_epi32(_mm_add_epi32(p_hi, vyuv32), vshift);
            let delta = _mm_packs_epi32(d_lo, d_hi);
            let cur = load_i16x8(chunk(&buf[y], x));
            let toplam = _mm_max_epi16(_mm_min_epi16(_mm_add_epi16(cur, delta), vmax), vmin);
            store_i16x8(chunk_mut(&mut buf[y], x), toplam);
            x += 8;
        }
        while x < x_son {
            let mut luma = 0i32;
            let luma_y = ((y - AR_PAD) << SUBY) + AR_PAD;
            let luma_x = ((x - AR_PAD) << SUBX) + AR_PAD;
            for i in 0..=SUBY {
                for j in 0..=SUBX {
                    luma += i32::from(buf_y[luma_y + i][luma_x + j]);
                }
            }
            let luma = round2(luma, (SUBX + SUBY) as u32);
            let g = i32::from(buf[y][x]) + round2(i32::from(luma_katsayi) * luma, ar_shift);
            buf[y][x] = g.clamp(gmin, gmax) as i16;
            x += 1;
        }
    }
}

#[target_feature(enable = "ssse3")]
fn grain_y_govde<const BPC: u32>(buf: &mut GrainLut, data: &FilmGrainData) {
    gauss_doldur_16::<BPC, GRAIN_WIDTH, GRAIN_HEIGHT>(buf, data.seed, data.grain_scale_shift);
    let s = data.ar_coeff_shift as u32;
    let c = &data.ar_coeffs_y;
    match data.ar_coeff_lag {
        0 => {}
        1 => ar_suz_lag_16::<BPC, GRAIN_WIDTH, GRAIN_HEIGHT, 1, 0, 0>(buf, None, c, 0, s),
        2 => ar_suz_lag_16::<BPC, GRAIN_WIDTH, GRAIN_HEIGHT, 2, 0, 0>(buf, None, c, 0, s),
        _ => ar_suz_lag_16::<BPC, GRAIN_WIDTH, GRAIN_HEIGHT, 3, 0, 0>(buf, None, c, 0, s),
    }
}

#[target_feature(enable = "ssse3")]
fn grain_uv_govde<
    const BPC: u32,
    const W: usize,
    const H: usize,
    const SUBX: usize,
    const SUBY: usize,
>(
    buf: &mut GrainLut,
    buf_y: &GrainLut,
    data: &FilmGrainData,
    uv: usize,
) {
    let tohum = data.seed ^ if uv != 0 { 0x49d8 } else { 0xb524 };
    gauss_doldur_16::<BPC, W, H>(buf, tohum, data.grain_scale_shift);
    let s = data.ar_coeff_shift as u32;
    let c = &data.ar_coeffs_uv[uv];
    let lag = data.ar_coeff_lag as usize;
    let luma_konum = 2 * lag * (lag + 1);
    let lk = if data.num_y_points > 0 {
        c[luma_konum]
    } else {
        0
    };
    let by = (data.num_y_points > 0).then_some(buf_y);
    match lag {
        0 => {
            if let Some(y_lut) = by {
                ar_suz_lag0_16::<BPC, W, H, SUBX, SUBY>(buf, y_lut, lk, s);
            }
        }
        1 => ar_suz_lag_16::<BPC, W, H, 1, SUBX, SUBY>(buf, by, c, lk, s),
        2 => ar_suz_lag_16::<BPC, W, H, 2, SUBX, SUBY>(buf, by, c, lk, s),
        _ => ar_suz_lag_16::<BPC, W, H, 3, SUBX, SUBY>(buf, by, c, lk, s),
    }
}

fn grain_y_giris(buf: &mut GrainLut, data: &FilmGrainData, bd: FgBitDepth) {
    if !ar16_sozlesmesi(data, bd) {
        fg_skaler_devri_say();
        let f = match bd {
            FgBitDepth::B8 => scalar::generate_grain_y::<8>,
            FgBitDepth::B10 => scalar::generate_grain_y::<10>,
            FgBitDepth::B12 => scalar::generate_grain_y::<12>,
        };
        return f(buf, data, bd);
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`generate_grain_y_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır.
    // (2) sınır: `buf` sabit boyutlu LUT türüdür (`[[i16; 82]; 73]`); erişimler
    // denetimli dizi/dilim erişimidir. (3) hizalama/takma ad: `buf` tek `&mut`.
    unsafe {
        if bd == FgBitDepth::B10 {
            grain_y_govde::<10>(buf, data);
        } else {
            grain_y_govde::<12>(buf, data);
        }
    }
}

macro_rules! grain_uv_giris {
    ($ad:ident, $w:ident, $h:ident, $sx:literal, $sy:literal, $skaler:ident) => {
        fn $ad(
            buf: &mut GrainLut,
            buf_y: &GrainLut,
            data: &FilmGrainData,
            uv: usize,
            bd: FgBitDepth,
        ) {
            if uv >= 2 || !ar16_sozlesmesi(data, bd) {
                fg_skaler_devri_say();
                let f = match bd {
                    FgBitDepth::B8 => scalar::$skaler::<8>,
                    FgBitDepth::B10 => scalar::$skaler::<10>,
                    FgBitDepth::B12 => scalar::$skaler::<12>,
                };
                return f(buf, buf_y, data, uv, bd);
            }
            // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
            // (`generate_grain_uv_fn`); belirteç SSSE3 tespiti + maske izninin
            // kanıtıdır. (2) sınır: `buf`/`buf_y` sabit boyutlu LUT türüdür; `uv <
            // 2`; erişimler denetimlidir. (3) hizalama/takma ad: `buf` tek `&mut`,
            // `buf_y` `&` (ayrı nesneler).
            unsafe {
                if bd == FgBitDepth::B10 {
                    grain_uv_govde::<10, $w, $h, $sx, $sy>(buf, buf_y, data, uv);
                } else {
                    grain_uv_govde::<12, $w, $h, $sx, $sy>(buf, buf_y, data, uv);
                }
            }
        }
    };
}

grain_uv_giris!(
    grain_uv_420_giris,
    SUB_GRAIN_WIDTH,
    SUB_GRAIN_HEIGHT,
    1,
    1,
    generate_grain_uv_420
);
grain_uv_giris!(
    grain_uv_422_giris,
    SUB_GRAIN_WIDTH,
    GRAIN_HEIGHT,
    1,
    0,
    generate_grain_uv_422
);
grain_uv_giris!(
    grain_uv_444_giris,
    GRAIN_WIDTH,
    GRAIN_HEIGHT,
    0,
    0,
    generate_grain_uv_444
);

/// `generate_grain_y` 16 bpc SSSE3 yuvası (yalnız belirteçle; D01589).
#[must_use]
pub fn generate_grain_y_fn(_ssse3: Ssse3Token) -> GenerateGrainYFn {
    grain_y_giris
}

/// `generate_grain_uv[düzen]` 16 bpc SSSE3 yuvası (yalnız belirteçle; D01577/D01581/D01585).
#[must_use]
pub fn generate_grain_uv_fn(_ssse3: Ssse3Token, duzen: usize) -> GenerateGrainUvFn {
    match duzen {
        0 => grain_uv_420_giris,
        1 => grain_uv_422_giris,
        _ => grain_uv_444_giris,
    }
}

// ============================================================================
// fgy_32x32xn ve fguv_32x32xn — 16 bpc SSSE3
// ============================================================================

/// Grain LUT'undan `(offy, offx)` konumundaki 8 ardışık `i16` değerini XMM'e yükler.
#[target_feature(enable = "ssse3")]
#[inline]
fn lut_8(lut: &GrainLut, y: usize, x: usize) -> __m128i {
    load_i16x8(chunk(&lut[y], x))
}

/// LUT ofset çifti `(offx, offy)`.
#[inline(always)]
fn lut_ofset(
    offsets: &[[i32; 2]; 2],
    subx: usize,
    suby: usize,
    bx: usize,
    by: usize,
) -> (usize, usize) {
    let randval = offsets[bx][by];
    let offx = 3 + (2 >> subx) * (3 + (randval >> 4)) as usize;
    let offy = 3 + (2 >> suby) * (3 + (randval & 0xF)) as usize;
    (
        offx + (FG_BLOCK_SIZE >> subx) * bx,
        offy + (FG_BLOCK_SIZE >> suby) * by,
    )
}

/// 4 adet `u16` indeks için sabit boyutlu `scaling: &[u8; N]` (`N = 1024` veya `4096`)
/// tablosundan 4 ölçekleme baytı okur ve 4 × `u16` olarak `u64` içinde paketler.
#[inline(always)]
fn olcekle4_16<const N: usize>(scaling: &[u8; N], q: u64) -> u64 {
    let mask = N - 1;
    let s0 = u64::from(scaling[(q as u16 as usize) & mask]);
    let s1 = u64::from(scaling[((q >> 16) as u16 as usize) & mask]);
    let s2 = u64::from(scaling[((q >> 32) as u16 as usize) & mask]);
    let s3 = u64::from(scaling[((q >> 48) as u16 as usize) & mask]);
    s0 | (s1 << 16) | (s2 << 32) | (s3 << 48)
}

/// XMM'deki 8 `u16` indeks için `scaling` değerlerini (`8 × i16`) döner.
#[target_feature(enable = "ssse3")]
#[inline]
fn olcekleme16_xmm<const N: usize>(scaling: &[u8; N], idx: __m128i) -> __m128i {
    let q0 = _mm_cvtsi128_si64(idx) as u64;
    let q1 = _mm_cvtsi128_si64(_mm_srli_si128::<8>(idx)) as u64;
    let r0 = olcekle4_16::<N>(scaling, q0);
    let r1 = olcekle4_16::<N>(scaling, q1);
    _mm_set_epi64x(r1 as i64, r0 as i64)
}

/// Kural 2: 8 adet `16` bit piksele (`s`) ölçeklenmiş grain gürültüsü ekler ve
/// `[vmin, vmax]`'a kırpar.
#[target_feature(enable = "ssse3")]
#[inline]
fn gurultu_ekle_8(
    s: __m128i,
    sc: __m128i,
    g: __m128i,
    carpan: __m128i,
    vmin: __m128i,
    vmax: __m128i,
) -> __m128i {
    let sc_sh = _mm_mullo_epi16(sc, carpan);
    let gurultu = _mm_mulhrs_epi16(sc_sh, g);
    let toplam = _mm_add_epi16(s, gurultu);
    _mm_max_epi16(_mm_min_epi16(toplam, vmax), vmin)
}

/// Kural 3: İki 8'li `i16` grain vektörünü (`eski`, `yeni`) `(w0, w1)` ağırlıklarıyla
/// `_mm_madd_epi16` (`i32` birikim, `12` bpc'de taşmasız) üzerinden harmanlar ve
/// `[GRAIN_MIN, GRAIN_MAX]`'a kırpar.
#[target_feature(enable = "ssse3")]
#[inline]
fn harmanla_16<const BPC: u32>(eski: __m128i, yeni: __m128i, w: [i16; 2]) -> __m128i {
    let gmin = -(1i16 << (BPC - 1));
    let gmax = (1i16 << (BPC - 1)) - 1;
    let w32 = _mm_set1_epi32((i32::from(w[0]) & 0xFFFF) | (i32::from(w[1]) << 16));
    let yuv = _mm_set1_epi32(16);
    let lo = _mm_srai_epi32::<5>(_mm_add_epi32(
        _mm_madd_epi16(_mm_unpacklo_epi16(eski, yeni), w32),
        yuv,
    ));
    let hi = _mm_srai_epi32::<5>(_mm_add_epi32(
        _mm_madd_epi16(_mm_unpackhi_epi16(eski, yeni), w32),
        yuv,
    ));
    let p = _mm_packs_epi32(lo, hi);
    _mm_max_epi16(_mm_min_epi16(p, _mm_set1_epi16(gmax)), _mm_set1_epi16(gmin))
}

/// Blok sol kenarındaki ilk 2 sütunu (`x = 0, 1`) sol bloktan gelen `sol2` ile
/// `[[27, 17], [17, 27]]` ağırlıklarıyla `i32` üzerinden harmanlar ve `g`'nin
/// ilk iki şeridine yazar.
#[target_feature(enable = "ssse3")]
#[inline]
fn yatay_harmanla_16<const BPC: u32>(sol2: __m128i, g: __m128i) -> __m128i {
    let gmin = -(1i16 << (BPC - 1));
    let gmax = (1i16 << (BPC - 1)) - 1;
    let cift = _mm_unpacklo_epi16(sol2, g);
    let w = _mm_set_epi16(0, 0, 0, 0, 27, 17, 17, 27);
    let m = _mm_srai_epi32::<5>(_mm_add_epi32(_mm_madd_epi16(cift, w), _mm_set1_epi32(16)));
    let p = _mm_packs_epi32(m, m);
    let k = _mm_max_epi16(_mm_min_epi16(p, _mm_set1_epi16(gmax)), _mm_set1_epi16(gmin));
    _mm_alignr_epi8::<12>(_mm_srli_si128::<4>(g), _mm_slli_si128::<12>(k))
}

/// 8 piksellik (`16` bayt) sabit boyutlu luma parçasını işler ve `d16`'ya yazar.
#[target_feature(enable = "ssse3")]
#[inline]
fn fgy_8_parca<const N: usize>(
    d16: &mut [u8; 16],
    s16: &[u8; 16],
    scaling: &[u8; N],
    g: __m128i,
    carpan: __m128i,
    vmin: __m128i,
    vmax: __m128i,
) {
    let s = load_u8x16(s16);
    let sc = olcekleme16_xmm::<N>(scaling, s);
    let out = gurultu_ekle_8(s, sc, g, carpan, vmin, vmax);
    store_u8x16(d16, out);
}

/// 8 piksellik (`16` bayt) luma satır parçasını işler ve `dst`'ye yazar.
#[target_feature(enable = "ssse3")]
#[inline]
fn fgy_8<const N: usize>(
    dst: &mut [u8],
    src: &[u8],
    off: usize,
    scaling: &[u8; N],
    g: __m128i,
    carpan: __m128i,
    vmin: __m128i,
    vmax: __m128i,
) {
    fgy_8_parca::<N>(
        chunk_mut(dst, off),
        chunk(src, off),
        scaling,
        g,
        carpan,
        vmin,
        vmax,
    );
}

/// Dikey örtüşme (`0..ystart`) veya kısmi (`10 <= bw < 32`) luma bloğunu 8'li SIMD
/// pencereleriyle (`0`, `8`, `16`, `bw - 8`) işler.
#[target_feature(enable = "ssse3")]
#[inline(never)]
fn fgy_ust_veya_kismi_16<const BPC: u32, const N: usize>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    scaling: &[u8; N],
    grain_lut: &GrainLut,
    offsets: &[[i32; 2]; 2],
    bx: usize,
    bw: usize,
    y_bas: usize,
    y_son: usize,
    xstart: usize,
    ystart: usize,
    carpan: __m128i,
    vmin: __m128i,
    vmax: __m128i,
) {
    const W: [[i16; 2]; 2] = [[27, 17], [17, 27]];
    let (ox00, oy00) = lut_ofset(offsets, 0, 0, 0, 0);
    let (ox10, oy10) = if xstart != 0 {
        lut_ofset(offsets, 0, 0, 1, 0)
    } else {
        (0, 0)
    };
    let (ox01, oy01) = if ystart != 0 {
        lut_ofset(offsets, 0, 0, 0, 1)
    } else {
        (0, 0)
    };
    let (ox11, oy11) = if xstart != 0 && ystart != 0 {
        lut_ofset(offsets, 0, 0, 1, 1)
    } else {
        (0, 0)
    };

    let mut dxs = [0usize; 4];
    let mut n = 0usize;
    let mut x0 = 0usize;
    while x0 + 8 <= bw {
        dxs[n] = x0;
        n += 1;
        x0 += 8;
    }
    if x0 < bw {
        dxs[n] = bw - 8;
        n += 1;
    }

    for y in y_bas..y_son {
        let off = y * stride + bx * 2;
        if y < ystart {
            for &dx in &dxs[..n] {
                let mut ust = lut_8(grain_lut, oy01 + y, ox01 + dx);
                let mut cur = lut_8(grain_lut, oy00 + y, ox00 + dx);
                if dx == 0 && xstart != 0 {
                    let sol_ust = lut_8(grain_lut, oy11 + y, ox11);
                    let sol_cur = lut_8(grain_lut, oy10 + y, ox10);
                    ust = yatay_harmanla_16::<BPC>(sol_ust, ust);
                    cur = yatay_harmanla_16::<BPC>(sol_cur, cur);
                }
                let g = harmanla_16::<BPC>(ust, cur, W[y]);
                fgy_8::<N>(dst, src, off + dx * 2, scaling, g, carpan, vmin, vmax);
            }
        } else {
            for &dx in &dxs[..n] {
                let mut g = lut_8(grain_lut, oy00 + y, ox00 + dx);
                if dx == 0 && xstart != 0 {
                    let sol = lut_8(grain_lut, oy10 + y, ox10);
                    g = yatay_harmanla_16::<BPC>(sol, g);
                }
                fgy_8::<N>(dst, src, off + dx * 2, scaling, g, carpan, vmin, vmax);
            }
        }
    }
}

/// Tam 32 piksellik (`64` bayt) luma bloğunun sıcak satırlarını (`ystart..bh`) SSSE3 ile işler.
#[target_feature(enable = "ssse3")]
#[inline(never)]
fn fgy_blok_32_16<const BPC: u32, const N: usize>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    scaling: &[u8; N],
    grain_lut: &GrainLut,
    offsets: &[[i32; 2]; 2],
    bx: usize,
    bh: usize,
    xstart: usize,
    ystart: usize,
    carpan: __m128i,
    vmin: __m128i,
    vmax: __m128i,
) {
    let (ox00, oy00) = lut_ofset(offsets, 0, 0, 0, 0);
    if xstart != 0 {
        let (ox10, oy10) = lut_ofset(offsets, 0, 0, 1, 0);
        for y in ystart..bh {
            let off = y * stride + bx * 2;
            let g32 = chunk::<i16, 32>(&grain_lut[oy00 + y], ox00);
            let sol = lut_8(grain_lut, oy10 + y, ox10);
            let g0 = yatay_harmanla_16::<BPC>(sol, load_i16x8(chunk(g32, 0)));
            let g1 = load_i16x8(chunk(g32, 8));
            let g2 = load_i16x8(chunk(g32, 16));
            let g3 = load_i16x8(chunk(g32, 24));
            let s64 = chunk::<u8, 64>(src, off);
            let d64 = chunk_mut::<u8, 64>(dst, off);
            fgy_8_parca::<N>(
                chunk_mut(d64, 0),
                chunk(s64, 0),
                scaling,
                g0,
                carpan,
                vmin,
                vmax,
            );
            fgy_8_parca::<N>(
                chunk_mut(d64, 16),
                chunk(s64, 16),
                scaling,
                g1,
                carpan,
                vmin,
                vmax,
            );
            fgy_8_parca::<N>(
                chunk_mut(d64, 32),
                chunk(s64, 32),
                scaling,
                g2,
                carpan,
                vmin,
                vmax,
            );
            fgy_8_parca::<N>(
                chunk_mut(d64, 48),
                chunk(s64, 48),
                scaling,
                g3,
                carpan,
                vmin,
                vmax,
            );
        }
    } else {
        for y in ystart..bh {
            let off = y * stride + bx * 2;
            let g32 = chunk::<i16, 32>(&grain_lut[oy00 + y], ox00);
            let g0 = load_i16x8(chunk(g32, 0));
            let g1 = load_i16x8(chunk(g32, 8));
            let g2 = load_i16x8(chunk(g32, 16));
            let g3 = load_i16x8(chunk(g32, 24));
            let s64 = chunk::<u8, 64>(src, off);
            let d64 = chunk_mut::<u8, 64>(dst, off);
            fgy_8_parca::<N>(
                chunk_mut(d64, 0),
                chunk(s64, 0),
                scaling,
                g0,
                carpan,
                vmin,
                vmax,
            );
            fgy_8_parca::<N>(
                chunk_mut(d64, 16),
                chunk(s64, 16),
                scaling,
                g1,
                carpan,
                vmin,
                vmax,
            );
            fgy_8_parca::<N>(
                chunk_mut(d64, 32),
                chunk(s64, 32),
                scaling,
                g2,
                carpan,
                vmin,
                vmax,
            );
            fgy_8_parca::<N>(
                chunk_mut(d64, 48),
                chunk(s64, 48),
                scaling,
                g3,
                carpan,
                vmin,
                vmax,
            );
        }
    }
}

/// Kısmi (`bw < 10`) luma bloğunu tam sınırlar içinde (`0..bw`) işler.
#[inline(never)]
fn fgy_kismi_blok_16<const BPC: u32, const N: usize>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    scaling: &[u8; N],
    grain_lut: &GrainLut,
    offsets: &[[i32; 2]; 2],
    bx: usize,
    bw: usize,
    bh: usize,
    xstart: usize,
    ystart: usize,
    shift: u32,
    min_value: i32,
    max_value: i32,
) {
    let gmin = -(1i32 << (BPC - 1));
    let gmax = (1i32 << (BPC - 1)) - 1;
    const W: [[i32; 2]; 2] = [[27, 17], [17, 27]];
    let mask = N - 1;
    let mut uygula = |x: usize, y: usize, grain: i32| {
        let off = y * stride + (bx + x) * 2;
        let s = i32::from(u16::from_le_bytes([src[off], src[off + 1]]));
        let sc = i32::from(scaling[(s as usize) & mask]);
        let noise = round2(sc * grain, shift);
        let out = (s + noise).clamp(min_value, max_value) as u16;
        dst[off..off + 2].copy_from_slice(&out.to_le_bytes());
    };
    for y in ystart..bh {
        for x in xstart..bw {
            let grain = sample_lut(grain_lut, offsets, 0, 0, 0, 0, x, y);
            uygula(x, y, grain);
        }
        for x in 0..xstart {
            let grain = sample_lut(grain_lut, offsets, 0, 0, 0, 0, x, y);
            let old = sample_lut(grain_lut, offsets, 0, 0, 1, 0, x, y);
            let grain = round2(old * W[x][0] + grain * W[x][1], 5).clamp(gmin, gmax);
            uygula(x, y, grain);
        }
    }
    for y in 0..ystart {
        for x in xstart..bw {
            let grain = sample_lut(grain_lut, offsets, 0, 0, 0, 0, x, y);
            let old = sample_lut(grain_lut, offsets, 0, 0, 0, 1, x, y);
            let grain = round2(old * W[y][0] + grain * W[y][1], 5).clamp(gmin, gmax);
            uygula(x, y, grain);
        }
        for x in 0..xstart {
            let top = sample_lut(grain_lut, offsets, 0, 0, 0, 1, x, y);
            let old = sample_lut(grain_lut, offsets, 0, 0, 1, 1, x, y);
            let top = round2(old * W[x][0] + top * W[x][1], 5).clamp(gmin, gmax);
            let grain = sample_lut(grain_lut, offsets, 0, 0, 0, 0, x, y);
            let old = sample_lut(grain_lut, offsets, 0, 0, 1, 0, x, y);
            let grain = round2(old * W[x][0] + grain * W[x][1], 5).clamp(gmin, gmax);
            let grain = round2(top * W[y][0] + grain * W[y][1], 5).clamp(gmin, gmax);
            uygula(x, y, grain);
        }
    }
}

#[target_feature(enable = "ssse3")]
fn fgy_govde<const BPC: u32, const N: usize>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8; N],
    grain_lut: &GrainLut,
    bh: usize,
    row_num: usize,
) {
    let overlap = data.overlap_flag;
    let rows = 1 + usize::from(overlap && row_num > 0);
    let min_8 = BPC - 8;
    let max = (1i32 << BPC) - 1;
    let (min_value, max_value) = if data.clip_to_restricted_range {
        (16i32 << min_8, 235i32 << min_8)
    } else {
        (0, max)
    };
    let shift = data.scaling_shift as u32;
    let mut seed = row_seeds(data, row_num, rows);
    let mut offsets = [[0i32; 2]; 2];
    let carpan = _mm_set1_epi16(1 << (15 - shift));
    let vmin = _mm_set1_epi16(min_value as i16);
    let vmax = _mm_set1_epi16(max_value as i16);

    let mut bx = 0usize;
    while bx < pw {
        let bw = FG_BLOCK_SIZE.min(pw - bx);
        if overlap && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = get_random_number(8, &mut seed[i]);
        }
        let ystart = if overlap && row_num != 0 {
            bh.min(2)
        } else {
            0
        };
        let xstart = if overlap && bx != 0 { bw.min(2) } else { 0 };

        if bw == FG_BLOCK_SIZE {
            if ystart > 0 {
                fgy_ust_veya_kismi_16::<BPC, N>(
                    dst, src, stride, scaling, grain_lut, &offsets, bx, bw, 0, ystart, xstart,
                    ystart, carpan, vmin, vmax,
                );
            }
            if ystart < bh {
                fgy_blok_32_16::<BPC, N>(
                    dst, src, stride, scaling, grain_lut, &offsets, bx, bh, xstart, ystart, carpan,
                    vmin, vmax,
                );
            }
        } else if bw >= 10 {
            fgy_ust_veya_kismi_16::<BPC, N>(
                dst, src, stride, scaling, grain_lut, &offsets, bx, bw, 0, bh, xstart, ystart,
                carpan, vmin, vmax,
            );
        } else {
            fgy_kismi_blok_16::<BPC, N>(
                dst, src, stride, scaling, grain_lut, &offsets, bx, bw, bh, xstart, ystart, shift,
                min_value, max_value,
            );
        }
        bx += FG_BLOCK_SIZE;
    }
}

fn fgy_giris(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8],
    grain_lut: &GrainLut,
    bh: usize,
    row_num: usize,
    bd: FgBitDepth,
) {
    let ok = fgy16_sozlesmesi(dst.len(), src.len(), stride, pw, bh, data, bd);
    let t10 = if ok && bd == FgBitDepth::B10 {
        scaling.first_chunk::<1024>()
    } else {
        None
    };
    let t12 = if ok && bd == FgBitDepth::B12 {
        scaling.first_chunk::<4096>()
    } else {
        None
    };
    if t10.is_none() && t12.is_none() {
        fg_skaler_devri_say();
        let f = match bd {
            FgBitDepth::B8 => scalar::fgy_32x32xn::<8>,
            FgBitDepth::B10 => scalar::fgy_32x32xn::<10>,
            FgBitDepth::B12 => scalar::fgy_32x32xn::<12>,
        };
        return f(
            dst, src, stride, data, pw, scaling, grain_lut, bh, row_num, bd,
        );
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`fgy_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `fgy16_sozlesmesi` `src`/`dst`'nin `(bh−1)·stride + pw·2` baytı kapsadığını,
    // `stride`'ın çift ve `≥ pw·2` olduğunu, `first_chunk` tablonun `1024`/`4096`
    // baytını garanti eder; bütün erişimler denetimli dilim/dizi erişimidir.
    // (3) hizalama/takma ad: `dst` tek `&mut`, diğerleri `&`.
    unsafe {
        if let Some(tablo) = t10 {
            fgy_govde::<10, 1024>(dst, src, stride, data, pw, tablo, grain_lut, bh, row_num);
        } else if let Some(tablo) = t12 {
            fgy_govde::<12, 4096>(dst, src, stride, data, pw, tablo, grain_lut, bh, row_num);
        }
    }
}

/// 8 kroma pikseli için luma satırından ortalama luma değerini (`8 × i16`) hesaplar.
#[target_feature(enable = "ssse3")]
#[inline]
fn luma_ort_uv8<const SX: usize>(luma: &[u8], l_baz: usize) -> __m128i {
    if SX == 0 {
        load_u8x16(chunk(luma, l_baz))
    } else {
        let l0 = load_u8x16(chunk(luma, l_baz));
        let l1 = load_u8x16(chunk(luma, l_baz + 16));
        let toplam = _mm_hadd_epi16(l0, l1);
        _mm_avg_epu16(toplam, _mm_setzero_si128())
    }
}

/// Kroma blok sol kenarında yatay örtüşme: `SX == 1`'de yalnız 0. sütun `[23, 22]`
/// ile, `SX == 0`'da 0. ve 1. sütunlar `[[27, 17], [17, 27]]` ile harmanlanır.
#[target_feature(enable = "ssse3")]
#[inline]
fn yatay_harmanla_uv_16<const BPC: u32, const SX: usize>(sol: __m128i, g: __m128i) -> __m128i {
    if SX == 0 {
        yatay_harmanla_16::<BPC>(sol, g)
    } else {
        let gmin = -(1i16 << (BPC - 1));
        let gmax = (1i16 << (BPC - 1)) - 1;
        let cift = _mm_unpacklo_epi16(sol, g);
        let w = _mm_set1_epi32(23 | (22 << 16));
        let m = _mm_srai_epi32::<5>(_mm_add_epi32(_mm_madd_epi16(cift, w), _mm_set1_epi32(16)));
        let p = _mm_packs_epi32(m, m);
        let k = _mm_max_epi16(_mm_min_epi16(p, _mm_set1_epi16(gmax)), _mm_set1_epi16(gmin));
        _mm_alignr_epi8::<14>(_mm_srli_si128::<2>(g), _mm_slli_si128::<14>(k))
    }
}

/// 8 kroma pikselini (`16` bayt) işler ve `dst`'ye yazar.
#[target_feature(enable = "ssse3")]
#[inline]
fn fguv_8_16<const N: usize, const SX: usize, const CSFL: bool>(
    dst: &mut [u8],
    src: &[u8],
    s_off: usize,
    luma: &[u8],
    l_off: usize,
    scaling: &[u8; N],
    g: __m128i,
    carpan: __m128i,
    vmin: __m128i,
    vmax: __m128i,
    vbdmax: __m128i,
    carpan_uv: __m128i,
    v_ofs: __m128i,
) {
    let s = load_u8x16(chunk(src, s_off));
    let avg = luma_ort_uv8::<SX>(luma, l_off);
    let idx = if CSFL {
        _mm_min_epi16(_mm_max_epi16(avg, _mm_setzero_si128()), vbdmax)
    } else {
        let lo = _mm_srai_epi32::<6>(_mm_madd_epi16(_mm_unpacklo_epi16(avg, s), carpan_uv));
        let hi = _mm_srai_epi32::<6>(_mm_madd_epi16(_mm_unpackhi_epi16(avg, s), carpan_uv));
        let birlesik = _mm_packs_epi32(lo, hi);
        let toplam = _mm_add_epi16(birlesik, v_ofs);
        _mm_min_epi16(_mm_max_epi16(toplam, _mm_setzero_si128()), vbdmax)
    };
    let sc = olcekleme16_xmm::<N>(scaling, idx);
    let out = gurultu_ekle_8(s, sc, g, carpan, vmin, vmax);
    store_u8x16(chunk_mut(dst, s_off), out);
}

/// Tam genişlikli (`32 >> SX`: 16 veya 32 piksel) kroma bloğunu SSSE3 ile işler.
#[target_feature(enable = "ssse3")]
#[inline(never)]
fn fguv_blok_tam_16<
    const BPC: u32,
    const N: usize,
    const SX: usize,
    const SY: usize,
    const CSFL: bool,
>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    luma: &[u8],
    luma_stride: usize,
    scaling: &[u8; N],
    grain_lut: &GrainLut,
    offsets: &[[i32; 2]; 2],
    bx: usize,
    bh: usize,
    xstart: usize,
    ystart: usize,
    carpan: __m128i,
    vmin: __m128i,
    vmax: __m128i,
    vbdmax: __m128i,
    carpan_uv: __m128i,
    v_ofs: __m128i,
) {
    const W_Y: [[[i16; 2]; 2]; 2] = [[[27, 17], [17, 27]], [[23, 22], [0, 0]]];
    let (ox00, oy00) = lut_ofset(offsets, SX, SY, 0, 0);
    let (ox10, oy10) = if xstart != 0 {
        lut_ofset(offsets, SX, SY, 1, 0)
    } else {
        (0, 0)
    };
    let (ox01, oy01) = if ystart != 0 {
        lut_ofset(offsets, SX, SY, 0, 1)
    } else {
        (0, 0)
    };
    let (ox11, oy11) = if xstart != 0 && ystart != 0 {
        lut_ofset(offsets, SX, SY, 1, 1)
    } else {
        (0, 0)
    };
    let bstep = FG_BLOCK_SIZE >> SX;

    for y in 0..ystart {
        let s_baz = y * stride + bx * 2;
        let l_baz = (y << SY) * luma_stride + (bx << SX) * 2;
        let wy = W_Y[SY][y];
        let mut dx = 0usize;
        while dx < bstep {
            let mut ust = lut_8(grain_lut, oy01 + y, ox01 + dx);
            let mut cur = lut_8(grain_lut, oy00 + y, ox00 + dx);
            if dx == 0 && xstart != 0 {
                let sol_ust = lut_8(grain_lut, oy11 + y, ox11);
                let sol_cur = lut_8(grain_lut, oy10 + y, ox10);
                ust = yatay_harmanla_uv_16::<BPC, SX>(sol_ust, ust);
                cur = yatay_harmanla_uv_16::<BPC, SX>(sol_cur, cur);
            }
            let g = harmanla_16::<BPC>(ust, cur, wy);
            fguv_8_16::<N, SX, CSFL>(
                dst,
                src,
                s_baz + dx * 2,
                luma,
                l_baz + (dx << SX) * 2,
                scaling,
                g,
                carpan,
                vmin,
                vmax,
                vbdmax,
                carpan_uv,
                v_ofs,
            );
            dx += 8;
        }
    }

    if SX == 0 {
        for y in ystart..bh {
            let s_baz = y * stride + bx * 2;
            let l_baz = (y << SY) * luma_stride + bx * 2;
            let g32 = chunk::<i16, 32>(&grain_lut[oy00 + y], ox00);
            let mut g0 = load_i16x8(chunk(g32, 0));
            if xstart != 0 {
                let sol = lut_8(grain_lut, oy10 + y, ox10);
                g0 = yatay_harmanla_uv_16::<BPC, 0>(sol, g0);
            }
            let g1 = load_i16x8(chunk(g32, 8));
            let g2 = load_i16x8(chunk(g32, 16));
            let g3 = load_i16x8(chunk(g32, 24));
            let s64 = chunk::<u8, 64>(src, s_baz);
            let d64 = chunk_mut::<u8, 64>(dst, s_baz);
            let l64 = chunk::<u8, 64>(luma, l_baz);
            fguv_8_16::<N, 0, CSFL>(
                d64, s64, 0, l64, 0, scaling, g0, carpan, vmin, vmax, vbdmax, carpan_uv, v_ofs,
            );
            fguv_8_16::<N, 0, CSFL>(
                d64, s64, 16, l64, 16, scaling, g1, carpan, vmin, vmax, vbdmax, carpan_uv, v_ofs,
            );
            fguv_8_16::<N, 0, CSFL>(
                d64, s64, 32, l64, 32, scaling, g2, carpan, vmin, vmax, vbdmax, carpan_uv, v_ofs,
            );
            fguv_8_16::<N, 0, CSFL>(
                d64, s64, 48, l64, 48, scaling, g3, carpan, vmin, vmax, vbdmax, carpan_uv, v_ofs,
            );
        }
    } else {
        for y in ystart..bh {
            let s_baz = y * stride + bx * 2;
            let l_baz = (y << SY) * luma_stride + (bx << 1) * 2;
            let g16 = chunk::<i16, 16>(&grain_lut[oy00 + y], ox00);
            let mut g0 = load_i16x8(chunk(g16, 0));
            if xstart != 0 {
                let sol = lut_8(grain_lut, oy10 + y, ox10);
                g0 = yatay_harmanla_uv_16::<BPC, SX>(sol, g0);
            }
            let g1 = load_i16x8(chunk(g16, 8));
            let s32 = chunk::<u8, 32>(src, s_baz);
            let d32 = chunk_mut::<u8, 32>(dst, s_baz);
            let l64 = chunk::<u8, 64>(luma, l_baz);
            fguv_8_16::<N, SX, CSFL>(
                d32, s32, 0, l64, 0, scaling, g0, carpan, vmin, vmax, vbdmax, carpan_uv, v_ofs,
            );
            fguv_8_16::<N, SX, CSFL>(
                d32, s32, 16, l64, 32, scaling, g1, carpan, vmin, vmax, vbdmax, carpan_uv, v_ofs,
            );
        }
    }
}

/// Kısmi (`bw < 32 >> SX`) kroma bloğunu tam sınırlar içinde (`0..bw`) işler.
#[inline(never)]
fn fguv_kismi_blok_16<const BPC: u32, const N: usize, const SX: usize, const SY: usize>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    scaling: &[u8; N],
    grain_lut: &GrainLut,
    offsets: &[[i32; 2]; 2],
    luma: &[u8],
    luma_stride: usize,
    uv: usize,
    bx: usize,
    bw: usize,
    bh: usize,
    xstart: usize,
    ystart: usize,
    shift: u32,
    min_value: i32,
    max_value: i32,
) {
    let gmin = -(1i32 << (BPC - 1));
    let gmax = (1i32 << (BPC - 1)) - 1;
    let max = (1i32 << BPC) - 1;
    let min_8 = BPC - 8;
    const W: [[[i32; 2]; 2]; 2] = [[[27, 17], [17, 27]], [[23, 22], [0, 0]]];
    let csfl = data.chroma_scaling_from_luma;
    let uv_luma_mult = data.uv_luma_mult[uv];
    let uv_mult = data.uv_mult[uv];
    let uv_offset = data.uv_offset[uv] * (1 << min_8);
    let mask = N - 1;

    let mut uygula = |x: usize, y: usize, grain: i32| {
        let lx = ((bx + x) << SX) * 2;
        let ly = y << SY;
        let lrow = ly * luma_stride;
        let mut avg = i32::from(u16::from_le_bytes([luma[lrow + lx], luma[lrow + lx + 1]]));
        if SX != 0 {
            let l1 = i32::from(u16::from_le_bytes([
                luma[lrow + lx + 2],
                luma[lrow + lx + 3],
            ]));
            avg = (avg + l1 + 1) >> 1;
        }
        let s_off = y * stride + (bx + x) * 2;
        let s = i32::from(u16::from_le_bytes([src[s_off], src[s_off + 1]]));
        let val = if csfl {
            avg.clamp(0, max)
        } else {
            let combined = avg * uv_luma_mult + s * uv_mult;
            ((combined >> 6) + uv_offset).clamp(0, max)
        };
        let sc = i32::from(scaling[(val as usize) & mask]);
        let noise = round2(sc * grain, shift);
        let out = (s + noise).clamp(min_value, max_value) as u16;
        dst[s_off..s_off + 2].copy_from_slice(&out.to_le_bytes());
    };

    for y in ystart..bh {
        for x in xstart..bw {
            let grain = sample_lut(grain_lut, offsets, SX, SY, 0, 0, x, y);
            uygula(x, y, grain);
        }
        for x in 0..xstart {
            let grain = sample_lut(grain_lut, offsets, SX, SY, 0, 0, x, y);
            let old = sample_lut(grain_lut, offsets, SX, SY, 1, 0, x, y);
            let grain = round2(old * W[SX][x][0] + grain * W[SX][x][1], 5).clamp(gmin, gmax);
            uygula(x, y, grain);
        }
    }
    for y in 0..ystart {
        for x in xstart..bw {
            let grain = sample_lut(grain_lut, offsets, SX, SY, 0, 0, x, y);
            let old = sample_lut(grain_lut, offsets, SX, SY, 0, 1, x, y);
            let grain = round2(old * W[SY][y][0] + grain * W[SY][y][1], 5).clamp(gmin, gmax);
            uygula(x, y, grain);
        }
        for x in 0..xstart {
            let top = sample_lut(grain_lut, offsets, SX, SY, 0, 1, x, y);
            let old = sample_lut(grain_lut, offsets, SX, SY, 1, 1, x, y);
            let top = round2(old * W[SX][x][0] + top * W[SX][x][1], 5).clamp(gmin, gmax);
            let grain = sample_lut(grain_lut, offsets, SX, SY, 0, 0, x, y);
            let old = sample_lut(grain_lut, offsets, SX, SY, 1, 0, x, y);
            let grain = round2(old * W[SX][x][0] + grain * W[SX][x][1], 5).clamp(gmin, gmax);
            let grain = round2(top * W[SY][y][0] + grain * W[SY][y][1], 5).clamp(gmin, gmax);
            uygula(x, y, grain);
        }
    }
}

#[target_feature(enable = "ssse3")]
fn fguv_govde<const BPC: u32, const N: usize, const SX: usize, const SY: usize>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8; N],
    grain_lut: &GrainLut,
    bh: usize,
    row_num: usize,
    luma: &[u8],
    luma_stride: usize,
    uv: usize,
    is_id: bool,
) {
    let overlap = data.overlap_flag;
    let rows = 1 + usize::from(overlap && row_num > 0);
    let min_8 = BPC - 8;
    let max = (1i32 << BPC) - 1;
    let (min_value, max_value) = if data.clip_to_restricted_range {
        (
            16i32 << min_8,
            (if is_id { 235i32 } else { 240i32 }) << min_8,
        )
    } else {
        (0, max)
    };
    let shift = data.scaling_shift as u32;
    let mut seed = row_seeds(data, row_num, rows);
    let mut offsets = [[0i32; 2]; 2];
    let lm = data.uv_luma_mult[uv];
    let m = data.uv_mult[uv];
    let ofs = (data.uv_offset[uv] * (1 << min_8)) as i16;
    let carpan = _mm_set1_epi16(1 << (15 - shift));
    let vmin = _mm_set1_epi16(min_value as i16);
    let vmax = _mm_set1_epi16(max_value as i16);
    let vbdmax = _mm_set1_epi16(max as i16);
    let carpan_uv = _mm_set1_epi32((lm & 0xFFFF) | (m << 16));
    let v_ofs = _mm_set1_epi16(ofs);

    let bstep = FG_BLOCK_SIZE >> SX;
    let mut bx = 0usize;
    while bx < pw {
        let bw = bstep.min(pw - bx);
        if overlap && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = get_random_number(8, &mut seed[i]);
        }
        let ystart = if overlap && row_num != 0 {
            bh.min(2 >> SY)
        } else {
            0
        };
        let xstart = if overlap && bx != 0 {
            bw.min(2 >> SX)
        } else {
            0
        };

        if bw == bstep {
            if data.chroma_scaling_from_luma {
                fguv_blok_tam_16::<BPC, N, SX, SY, true>(
                    dst,
                    src,
                    stride,
                    luma,
                    luma_stride,
                    scaling,
                    grain_lut,
                    &offsets,
                    bx,
                    bh,
                    xstart,
                    ystart,
                    carpan,
                    vmin,
                    vmax,
                    vbdmax,
                    carpan_uv,
                    v_ofs,
                );
            } else {
                fguv_blok_tam_16::<BPC, N, SX, SY, false>(
                    dst,
                    src,
                    stride,
                    luma,
                    luma_stride,
                    scaling,
                    grain_lut,
                    &offsets,
                    bx,
                    bh,
                    xstart,
                    ystart,
                    carpan,
                    vmin,
                    vmax,
                    vbdmax,
                    carpan_uv,
                    v_ofs,
                );
            }
        } else {
            fguv_kismi_blok_16::<BPC, N, SX, SY>(
                dst,
                src,
                stride,
                data,
                scaling,
                grain_lut,
                &offsets,
                luma,
                luma_stride,
                uv,
                bx,
                bw,
                bh,
                xstart,
                ystart,
                shift,
                min_value,
                max_value,
            );
        }
        bx += bstep;
    }
}

macro_rules! fguv_giris {
    ($ad:ident, $sx:literal, $sy:literal, $skaler:ident) => {
        fn $ad(
            dst: &mut [u8],
            src: &[u8],
            stride: usize,
            data: &FilmGrainData,
            pw: usize,
            scaling: &[u8],
            grain_lut: &GrainLut,
            bh: usize,
            row_num: usize,
            luma: &[u8],
            luma_stride: usize,
            uv: usize,
            is_id: bool,
            bd: FgBitDepth,
        ) {
            let ok = fguv16_sozlesmesi(
                dst.len(),
                src.len(),
                stride,
                pw,
                bh,
                luma.len(),
                luma_stride,
                uv,
                $sx,
                $sy,
                data,
                bd,
            ) && uv_katsayi_sozlesmesi(data, uv);
            let t10 = if ok && bd == FgBitDepth::B10 {
                scaling.first_chunk::<1024>()
            } else {
                None
            };
            let t12 = if ok && bd == FgBitDepth::B12 {
                scaling.first_chunk::<4096>()
            } else {
                None
            };
            if t10.is_none() && t12.is_none() {
                fg_skaler_devri_say();
                let f = match bd {
                    FgBitDepth::B8 => scalar::$skaler::<8>,
                    FgBitDepth::B10 => scalar::$skaler::<10>,
                    FgBitDepth::B12 => scalar::$skaler::<12>,
                };
                return f(
                    dst,
                    src,
                    stride,
                    data,
                    pw,
                    scaling,
                    grain_lut,
                    bh,
                    row_num,
                    luma,
                    luma_stride,
                    uv,
                    is_id,
                    bd,
                );
            }
            // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
            // (`fguv_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır. (2) sınır:
            // `fguv16_sozlesmesi` `src`/`dst`'nin `(bh−1)·stride + pw·2` baytı ve lumanın
            // `((bh−1) << sy)·luma_stride + (((pw−1) << sx) + sx + 1)·2` baytı kapsadığını,
            // `first_chunk` tablonun `1024`/`4096` baytını garanti eder; erişimler denetimlidir.
            // (3) hizalama/takma ad: `dst` tek `&mut`, diğerleri `&`.
            unsafe {
                if let Some(tablo) = t10 {
                    fguv_govde::<10, 1024, $sx, $sy>(
                        dst,
                        src,
                        stride,
                        data,
                        pw,
                        tablo,
                        grain_lut,
                        bh,
                        row_num,
                        luma,
                        luma_stride,
                        uv,
                        is_id,
                    );
                } else if let Some(tablo) = t12 {
                    fguv_govde::<12, 4096, $sx, $sy>(
                        dst,
                        src,
                        stride,
                        data,
                        pw,
                        tablo,
                        grain_lut,
                        bh,
                        row_num,
                        luma,
                        luma_stride,
                        uv,
                        is_id,
                    );
                }
            }
        }
    };
}

fguv_giris!(fguv_420_giris, 1, 1, fguv_32x32xn_420);
fguv_giris!(fguv_422_giris, 1, 0, fguv_32x32xn_422);
fguv_giris!(fguv_444_giris, 0, 0, fguv_32x32xn_444);

/// `fgy_32x32xn` 16 bpc SSSE3 yuvası (yalnız belirteçle; D01573).
#[must_use]
pub fn fgy_fn(_ssse3: Ssse3Token) -> FgyFn {
    fgy_giris
}

/// `fguv_32x32xn[düzen]` 16 bpc SSSE3 yuvası (yalnız belirteçle; D01555/D01561/D01567).
#[must_use]
pub fn fguv_fn(_ssse3: Ssse3Token, duzen: usize) -> FguvFn {
    match duzen {
        0 => fguv_420_giris,
        1 => fguv_422_giris,
        _ => fguv_444_giris,
    }
}
