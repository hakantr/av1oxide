//! MC warp AVX512ICL çekirdekleri (16 bpc, 10/12 bit) — F4-MC e (P36):
//! `warp8x8`, `warp8x8t`.
//!
//! ## Sözleşme
//!
//! AVX2 gövdesinin sözleşmesi ([`super::mc16_warp`],
//! [`crate::dsp::mc::warp16_sozlesmesi`]): kaynak 15 satır × 30 bayt, hedef
//! 8 × 8, filtre indeksi `[0, 192]`. Satırlar **maskeli** yüklenir (30 bayt;
//! okuma bölgenin içinde biter). Filtreler `vpgatherdq` ile toplanır (dizinler
//! ayrıca denetlenir); yuva `SLOW_GATHER` bayrağı olan işlemcide kurulmaz.
//!
//! ## Yöntem
//!
//! - **Yatay** (iki satır bir zmm'de; 15. satır çifti satır 14'ü yineler):
//!   lane `(h, p)` (`h` satır, `p` çıktı paritesi) dword `i` = çıktı `x = 2i+p`
//!   için `(s[x+2j], s[x+2j+1])` ikilisi (`vpermw`), filtre ikilisi
//!   `(f_x[2j], f_x[2j+1])` toplanan qword'lerden ([`super::mc_warp_avx512`]
//!   `filtre_ikilisi`), `vpdpwssd` (doymasız); `vpermd` sütun sırasını kurar;
//!   `(Σ + r) >> (7−ib)`, `vpmovsdw` (ara değer `[−6015, 22395]`, AVX2
//!   gövdesindeki gerekçe).
//! - **Dikey:** 8 bpc gövdesiyle ortak ([`super::mc_warp_avx512`] `dikey`);
//!   `warp8x8` `(Σ + r) >> (7+ib)`, `vpmovusdw` + `pminuw(bitdepth_max)`;
//!   `warp8x8t` `(Σ + 64) >> 7 − 8192`, `vpmovsdw`.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mc_warp::WARP_V;
use super::mc_warp_avx512::{INDEKS_EK, dikey, filtre_ikilisi};
use super::mem::{
    chunk, chunk_mut, gather_i64x8, load_i16x32, load_u8_kz, load_u8x64, store_i16x8, store_i16x16,
    store_u8x16,
};
use crate::cpu::Avx512IclToken;
use crate::dsp::mc::{Warp8x8Fn, Warp8x8tFn, skaler_devri_say, warp16_sozlesmesi};
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m128i, __m512i, _mm_cvtsi32_si128, _mm256_add_epi32, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_min_epu16, _mm256_mullo_epi32, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_setr_epi32, _mm256_srai_epi32, _mm512_castsi512_si256,
    _mm512_cvtsepi32_epi16, _mm512_cvtusepi32_epi16, _mm512_dpwssd_epi32, _mm512_inserti64x4,
    _mm512_max_epi32, _mm512_permutexvar_epi16, _mm512_permutexvar_epi32, _mm512_set1_epi32,
    _mm512_setr_epi32, _mm512_setzero_si512, _mm512_sra_epi32, _mm512_sub_epi32,
};

/// `prep` ara değerinden çıkarılan sabit (dav1d `PREP_BIAS`, 16 bpc).
const PREP_BIAS: i32 = 8192;

/// `vpermw` dizinleri (yatay ikili `j`): word `w`, lane `(h, p)`, dword `i`,
/// sözcük `b` → satır `h`'nin pikseli `2i + p + 2j + b` (`16h +` piksel).
static YATAY_SIRA: [[i16; 32]; 4] = [yatay_sira(0), yatay_sira(1), yatay_sira(2), yatay_sira(3)];

const fn yatay_sira(j: usize) -> [i16; 32] {
    let mut m = [0i16; 32];
    let mut w = 0;
    while w < 32 {
        let serit = w / 8;
        let (h, p) = (serit / 2, serit % 2);
        let (i, b) = ((w % 8) / 2, w % 2);
        m[w] = (16 * h + 2 * i + p + 2 * j + b) as i16;
        w += 1;
    }
    m
}

/// `vpermt2b` dizinleri (yatay filtre ikilisi `j`): dword (lane `(h, p)`, `i`)
/// bayt 1 = `f_x[2j]`, bayt 3 = `f_x[2j+1]` (`x = 2i + p`; `h = 1` ikinci
/// kaynak, +64).
static YATAY_FILTRE: [[u8; 64]; 4] = [
    yatay_filtre(0),
    yatay_filtre(1),
    yatay_filtre(2),
    yatay_filtre(3),
];

const fn yatay_filtre(j: usize) -> [u8; 64] {
    let mut m = [0u8; 64];
    let mut q = 0;
    while q < 64 {
        let d = q / 4;
        let serit = d / 4;
        let (h, p) = (serit / 2, serit % 2);
        let x = 2 * (d % 4) + p;
        m[q] = match q % 4 {
            1 => (64 * h + 8 * x + 2 * j) as u8,
            3 => (64 * h + 8 * x + 2 * j + 1) as u8,
            _ => 0,
        };
        q += 1;
    }
    m
}

/// `vpermd`: lane `(h, p)` dword `i` (çıktı `2i + p`) → satır `h` sütun sırası.
static SUTUN_SIRA: [u8; 16] = {
    let mut m = [0u8; 16];
    let mut k = 0;
    while k < 16 {
        let (h, x) = (k / 8, k % 8);
        m[k] = (4 * (2 * h + x % 2) + x / 2) as u8;
        k += 1;
    }
    m
};

/// Yatay geçiş: ara satırlar `mid[8r..8r+8]` (`r` 0…14; satır 15 = satır 14),
/// `(Σ + r) >> sh`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yatay16(
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    yuv: __m512i,
    sh: __m128i,
    mid: &mut [i16; 128],
) {
    let (a, b) = (i32::from(abcd[0]), i32::from(abcd[1]));
    let xa = _mm256_mullo_epi32(
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
        _mm256_set1_epi32(a),
    );
    let zs: [__m512i; 4] = core::array::from_fn(|j| load_i16x32(&YATAY_SIRA[j]));
    let gs: [__m512i; 4] = core::array::from_fn(|j| load_u8x64(&YATAY_FILTRE[j]));
    let s = SUTUN_SIRA.map(i32::from);
    let sira = _mm512_setr_epi32(
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], s[13],
        s[14], s[15],
    );
    let indeks = |r: usize| {
        let t = mx + r as i32 * b + INDEKS_EK;
        _mm256_srai_epi32::<10>(_mm256_add_epi32(xa, _mm256_set1_epi32(t)))
    };
    for q in 0..8 {
        let (r0, r1) = (2 * q, (2 * q + 1).min(14));
        let f0 = gather_i64x8(&WARP_V, indeks(r0));
        let f1 = gather_i64x8(&WARP_V, indeks(r1));
        let v = _mm512_inserti64x4::<1>(
            load_u8_kz::<30>(chunk(src, r0 * ss)),
            _mm512_castsi512_si256(load_u8_kz::<30>(chunk(src, r1 * ss))),
        );
        let mut acc = yuv;
        for j in 0..4 {
            let z = _mm512_permutexvar_epi16(zs[j], v);
            acc = _mm512_dpwssd_epi32(acc, z, filtre_ikilisi(f0, gs[j], f1));
        }
        let s = _mm512_sra_epi32(_mm512_permutexvar_epi32(sira, acc), sh);
        store_i16x16(chunk_mut(mid, 16 * q), _mm512_cvtsepi32_epi16(s));
    }
}

/// Warp gövdesi: `T = false` → `warp8x8` (`u16` piksel), `true` → `warp8x8t`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
    let mut mid = [0i16; 128];
    yatay16(
        src,
        ss,
        abcd,
        mx,
        _mm512_set1_epi32(1 << (sh_h - 1)),
        _mm_cvtsi32_si128(sh_h),
        &mut mid,
    );
    let sh_v = if T { 7 } else { 7 + ib };
    let toplam = dikey(&mid, abcd, my, _mm512_set1_epi32(1 << (sh_v - 1)));
    let sh_v = _mm_cvtsi32_si128(sh_v);
    let bias = _mm512_set1_epi32(PREP_BIAS);
    let max = _mm256_set1_epi16(bitdepth_max as i16);
    for (q, s) in toplam.into_iter().enumerate() {
        let y = 2 * q;
        let v = _mm512_sra_epi32(s, sh_v);
        if T {
            let p = _mm512_cvtsepi32_epi16(_mm512_sub_epi32(v, bias));
            store_i16x8(chunk_mut(tmp, y * ds), _mm256_castsi256_si128(p));
            store_i16x8(
                chunk_mut(tmp, (y + 1) * ds),
                _mm256_extracti128_si256::<1>(p),
            );
        } else {
            let p = _mm512_cvtusepi32_epi16(_mm512_max_epi32(v, _mm512_setzero_si512()));
            let p = _mm256_min_epu16(p, max);
            store_u8x16(chunk_mut(dst, y * ds), _mm256_castsi256_si128(p));
            store_u8x16(
                chunk_mut(dst, (y + 1) * ds),
                _mm256_extracti128_si256::<1>(p),
            );
        }
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`warp8x8_fn`); belirteç AVX512ICL (VBMI, VNNI dahil) tespitinin
    // kanıtıdır. (2) sınır: `warp16_sozlesmesi` kaynak bölgesini (15 satır × 30
    // bayt, `MC_KUYRUK`), hedefte 8 satır × 16 baytı ve filtre indeksinin
    // `[0, 192]` aralığını kapsar; toplama dizinleri ayrıca denetlenir. (3)
    // hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`warp8x8t_fn`); belirteç AVX512ICL tespitinin kanıtıdır. (2) sınır:
    // `warp16_sozlesmesi` kaynak bölgesini, `tmp`'de 8 satır × 8 öğeyi ve filtre
    // indeksini kapsar; toplama dizinleri ayrıca denetlenir. (3) hizalama/takma
    // ad: `tmp` tek `&mut`, `src` `&`.
    unsafe { warp16_govde::<true>(&mut [], tmp, ts, src, ss, abcd, mx, my, bdmax) }
}

/// 16 bpc `warp8x8` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn warp8x8_fn(_avx512: Avx512IclToken) -> Warp8x8Fn {
    warp8x8_16_giris
}

/// 16 bpc `warp8x8t` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn warp8x8t_fn(_avx512: Avx512IclToken) -> Warp8x8tFn {
    warp8x8t_16_giris
}
