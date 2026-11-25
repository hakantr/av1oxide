//! MC warp AVX2 çekirdekleri (16 bpc, 10/12 bit) — F4-MC c (P34): `warp8x8`,
//! `warp8x8t`.
//!
//! ## Sözleşme
//!
//! Kaynak dolgu hücresi (blok satırı −3, sütunu −3) `15 × 15` piksellik okuma
//! bölgesinin başıdır; pikseller `u16`, satır adımları bayt (`warp8x8t`'nin
//! `tmp` satır adımı öğe). Girişte O(1) denetim
//! ([`crate::dsp::mc::warp16_sozlesmesi`]): bit derinliği 10/12 ve
//! `bitdepth_max` tutarlı, `src_stride ≥ 30`, kaynak dilimi `14·ss + 30 +
//! MC_KUYRUK` bayt, hedef 8 satır × 8 piksel ve filtre indeksinin
//! `64 + ((t + 512) >> 10)` bütün `(satır, sütun)` için `[0, 192]` içinde
//! olması. Sözleşme dışı girdi skalere devreder ve sayılır. Yatay yüklemeler
//! okuma bölgesinin içinde biter (son yükleme sütun 7…14).
//!
//! ## Yöntem
//!
//! - **Yatay** (15 satır × 8 çıktı, her çıktının filtresi ayrı): çıktı `x`
//!   için `s[x..x+8]` (8 piksel, 16 bayt) ve filtre `f_x` (`i16`, derleme
//!   zamanı tablosu [`WARP16_H`]) `pmaddwd`; `x` ve `x + 4` bir ymm'in iki
//!   şeridinde, `phaddd` ağacı `[x0..x3 | x4..x7]` verir; `(Σ + r) >> (7−ib)`
//!   (`ib` 10 bitte 4, 12 bitte 2; kaydırma yazmaçta). Ara değer
//!   `[−6015, 22395]` (tablo: pozitif tap toplamı en çok 175, negatif en az
//!   −47) `i16`'ya sığar.
//! - **Dikey** (8 satır × 8 çıktı): 8 bpc gövdesinin yolu (`i16` ara satırlar,
//!   8×8 aktarım, `palignr`, `pmovsxbw` filtreler, `pmaddwd`, `phaddd`,
//!   `vpermd`; [`super::mc_warp`]). `warp8x8` `(Σ + r) >> (7+ib)`,
//!   `packusdw` + `pminuw(bitdepth_max)`; `warp8x8t` `(Σ + 64) >> 7 − 8192`
//!   (`PREP_BIAS`), `packssdw`.
//!
//! `unsafe` yalnız belirteçli girişlerdeki `#[target_feature]` çağrısıdır;
//! dosya `audit::UNSAFE_EXEMPTIONS` defterinde kayıtlıdır. Yükleme/saklama
//! [`super::mem`] yardımcılarıyla yapılır.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mc_warp::{WARP_V, aktar, indeks, kaydir};
use super::mem::{chunk, chunk_mut, load_u8x16, store_i16x8, store_u8x16};
use crate::cpu::Avx2Token;
use crate::dsp::mc::{Warp8x8Fn, Warp8x8tFn, skaler_devri_say, warp16_sozlesmesi};
use crate::dsp::mc_tables::{MC_WARP_FILTER, N_WARP_FILTERS};
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, __m256i, _mm_cvtsi32_si128, _mm_min_epu16, _mm_packs_epi32, _mm_packus_epi32,
    _mm_set_epi64x, _mm_set1_epi16, _mm_setzero_si128, _mm_unpacklo_epi64, _mm256_add_epi32,
    _mm256_castsi256_si128, _mm256_cvtepi8_epi16, _mm256_extracti128_si256, _mm256_hadd_epi32,
    _mm256_madd_epi16, _mm256_packs_epi32, _mm256_permutevar8x32_epi32, _mm256_set_m128i,
    _mm256_set1_epi32, _mm256_setr_epi32, _mm256_setzero_si256, _mm256_sra_epi32, _mm256_sub_epi32,
};

/// `prep` ara değerinden çıkarılan sabit (dav1d `PREP_BIAS`, 16 bpc).
const PREP_BIAS: i32 = 8192;

/// Yatay geçiş filtre tablosu: `(f0, …, f7)` `i16` küçük sonlu (16 bayt).
static WARP16_H: [[u8; 16]; N_WARP_FILTERS] = tablo16();

const fn tablo16() -> [[u8; 16]; N_WARP_FILTERS] {
    let mut t = [[0u8; 16]; N_WARP_FILTERS];
    let mut i = 0;
    while i < N_WARP_FILTERS {
        let mut k = 0;
        while k < 8 {
            let b = (MC_WARP_FILTER[i][k] as i16).to_le_bytes();
            t[i][2 * k] = b[0];
            t[i][2 * k + 1] = b[1];
            k += 1;
        }
        i += 1;
    }
    t
}

/// Yatay geçiş: 15 ara satır (`i16` × 8, `xmm`); `(Σ + yuv) >> sh`.
#[target_feature(enable = "avx2")]
#[inline]
fn yatay16(
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    yuv: __m256i,
    sh: __m128i,
) -> [__m128i; 16] {
    let (a, b) = (i32::from(abcd[0]), i32::from(abcd[1]));
    let mut ara = [_mm_setzero_si128(); 16];
    let mut t0 = mx;
    for (r, satir) in ara.iter_mut().enumerate().take(15) {
        let o = r * ss;
        let mut m = [_mm256_setzero_si256(); 4];
        for (x, mp) in m.iter_mut().enumerate() {
            // Şerit 0: çıktı `x`, şerit 1: çıktı `x + 4`.
            let p = _mm256_set_m128i(
                load_u8x16(chunk(src, o + 2 * (x + 4))),
                load_u8x16(chunk(src, o + 2 * x)),
            );
            let tx = t0 + x as i32 * a;
            let f = _mm256_set_m128i(
                load_u8x16(&WARP16_H[indeks(tx + 4 * a)]),
                load_u8x16(&WARP16_H[indeks(tx)]),
            );
            *mp = _mm256_madd_epi16(p, f);
        }
        // [x0, x1, x2, x3 | x4, x5, x6, x7]
        let s = _mm256_hadd_epi32(_mm256_hadd_epi32(m[0], m[1]), _mm256_hadd_epi32(m[2], m[3]));
        let t = _mm256_sra_epi32(_mm256_add_epi32(s, yuv), sh);
        let p = _mm256_packs_epi32(t, t);
        *satir = _mm_unpacklo_epi64(_mm256_castsi256_si128(p), _mm256_extracti128_si256::<1>(p));
        t0 += b;
    }
    ara
}

/// Warp gövdesi: `T = false` → `warp8x8` (`u16` piksel), `true` → `warp8x8t`
/// (`i16`).
#[target_feature(enable = "avx2")]
fn warp16_govde<const T: bool>(
    dst: &mut [u8],
    tmp: &mut [i16],
    ds: usize,
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bitdepth_max: u16,
) {
    let ib = if bitdepth_max == 4095 { 2 } else { 4 };
    let sh_h = 7 - ib;
    let ara = yatay16(
        src,
        ss,
        abcd,
        mx,
        _mm256_set1_epi32(1 << (sh_h - 1)),
        _mm_cvtsi32_si128(sh_h),
    );
    let ust = aktar(&ara, 0);
    let alt = aktar(&ara, 8);
    // Sütun ikilileri (x, x+1): üst 8 satır ve alt 8 satır (son satır 0).
    let lo: [__m256i; 4] = core::array::from_fn(|p| _mm256_set_m128i(ust[2 * p + 1], ust[2 * p]));
    let hi: [__m256i; 4] = core::array::from_fn(|p| _mm256_set_m128i(alt[2 * p + 1], alt[2 * p]));
    let sira = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let (c, d) = (i32::from(abcd[2]), i32::from(abcd[3]));
    let sh_v = if T { 7 } else { 7 + ib };
    let yuv = _mm256_set1_epi32(1 << (sh_v - 1));
    let sh_v = _mm_cvtsi32_si128(sh_v);
    let bias = _mm256_set1_epi32(PREP_BIAS);
    let max = _mm_set1_epi16(bitdepth_max as i16);
    let mut t0 = my;
    for y in 0..8 {
        let f = |x: i32| WARP_V[indeks(t0 + x * c)];
        let mut m = [_mm256_setzero_si256(); 4];
        for (p, mp) in m.iter_mut().enumerate() {
            let x = 2 * p as i32;
            let k = _mm256_cvtepi8_epi16(_mm_set_epi64x(f(x + 1), f(x)));
            *mp = _mm256_madd_epi16(kaydir(hi[p], lo[p], y), k);
        }
        // [x0, x2, x4, x6 | x1, x3, x5, x7] → x sırası.
        let s = _mm256_hadd_epi32(_mm256_hadd_epi32(m[0], m[1]), _mm256_hadd_epi32(m[2], m[3]));
        let s = _mm256_add_epi32(_mm256_permutevar8x32_epi32(s, sira), yuv);
        let v = _mm256_sra_epi32(s, sh_v);
        if T {
            let v = _mm256_sub_epi32(v, bias);
            let p = _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
            store_i16x8(chunk_mut(tmp, y * ds), p);
        } else {
            let p = _mm_packus_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
            store_u8x16(chunk_mut(dst, y * ds), _mm_min_epu16(p, max));
        }
        t0 += d;
    }
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

fn warp8x8_16_giris(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bd: BitDepth,
    bdmax: u16,
) {
    if !warp16_sozlesmesi(src.len(), ss, abcd, mx, my, dst.len(), ds, 2, bd, bdmax) {
        skaler_devri_say();
        return skaler!(bd, warp8x8(dst, ds, src, ss, abcd, mx, my, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`warp8x8_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `warp16_sozlesmesi` kaynak bölgesini (15 satır × 30 bayt, `MC_KUYRUK`),
    // hedefte 8 satır × 16 baytı ve filtre indeksinin `[0, 192]` aralığını kapsar;
    // erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
    unsafe { warp16_govde::<false>(dst, &mut [], ds, src, ss, abcd, mx, my, bdmax) }
}

fn warp8x8t_16_giris(
    tmp: &mut [i16],
    ts: usize,
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bd: BitDepth,
    bdmax: u16,
) {
    if !warp16_sozlesmesi(src.len(), ss, abcd, mx, my, tmp.len(), ts, 1, bd, bdmax) {
        skaler_devri_say();
        return skaler!(bd, warp8x8t(tmp, ts, src, ss, abcd, mx, my, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`warp8x8t_fn`); belirteç AVX2 tespitinin kanıtıdır. (2) sınır:
    // `warp16_sozlesmesi` kaynak bölgesini, `tmp`'de 8 satır × 8 öğeyi ve filtre
    // indeksini kapsar; erişimler denetimlidir. (3) hizalama/takma ad: `tmp`
    // tek `&mut`, `src` `&`.
    unsafe { warp16_govde::<true>(&mut [], tmp, ts, src, ss, abcd, mx, my, bdmax) }
}

/// 16 bpc `warp8x8` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn warp8x8_fn(_avx2: Avx2Token) -> Warp8x8Fn {
    warp8x8_16_giris
}

/// 16 bpc `warp8x8t` AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn warp8x8t_fn(_avx2: Avx2Token) -> Warp8x8tFn {
    warp8x8t_16_giris
}
