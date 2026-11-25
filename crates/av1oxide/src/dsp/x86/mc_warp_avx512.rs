//! MC warp AVX512ICL çekirdekleri (8 bpc) — F4-MC e (P36): `warp8x8`,
//! `warp8x8t`.
//!
//! ## Sözleşme
//!
//! AVX2 gövdesinin sözleşmesi ([`super::mc_warp`],
//! [`crate::dsp::mc::warp_sozlesmesi`]): kaynak 15 satır × 16 bayt
//! (`MC_KUYRUK` içinde), hedef 8 × 8, filtre indeksi `[0, 192]`. Sözleşme dışı
//! girdi skalere devreder ve sayılır. Filtreler `vpgatherdq` ile toplanır;
//! toplama yardımcısı ([`super::mem::gather_i64x8`]) dizinleri ayrıca vektörle
//! denetler. Yuva `SLOW_GATHER` bayrağı olan işlemcide kurulmaz
//! (`dsp::candidate_entry`).
//!
//! ## Yöntem
//!
//! - **Yatay** (15 satır × 8 çıktı): satırın 16 baytından `vpermb` ile qword
//!   `x` = `s[x..x+8]`; çıktı başına filtre (`f0..f7`, [`super::mc_warp`]
//!   `WARP_V` tablosu) qword olarak toplanır; `vpdpbusd` (VNNI, `u8 × i8`,
//!   doymasız) qword'ün iki dword'üne `Σ_{k<4}` ve `Σ_{k≥4}` verir, ikisi
//!   toplanır; `(Σ + 4) >> 3`, `vpmovqw` → `i16` ara satır (yığında 16 × 8).
//! - **Dikey** ([`dikey`], 16 bpc gövdesiyle ortak): iki çıkış satırı bir
//!   zmm'de; ara satır ikilisi `(r, r+1)` sütun başına bir dword
//!   (`vpermw`), filtre ikilisi `(f[2j], f[2j+1])` toplanan qword'lerden
//!   `vpermt2b` (sıfırlayan maske) + `psraw 8` ile işaret genişletilmiş
//!   `i16`; `vpdpwssd` (doymasız `i32` birikim). `warp8x8` `(Σ + 1024) >> 11`
//!   → `[0, 255]`, `warp8x8t` `(Σ + 64) >> 7` (8 bpc'de `PREP_BIAS` 0).
//!
//! Filtre indeksi `64 + ((t + 512) >> 10) = (t + 66048) >> 10` (aritmetik
//! kaydırma; `64·1024` 1024'ün katı).

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use super::mc_warp::WARP_V;
use super::mem::{
    chunk, chunk_mut, gather_i64x8, load_i16x32, load_u8x16, load_u8x64, store_i16x8, store_u8x8,
};
use crate::cpu::Avx512IclToken;
use crate::dsp::mc::{Warp8x8Fn, Warp8x8tFn, skaler_devri_say, warp_sozlesmesi};
use crate::dsp::scalar::mc as scalar;
use crate::frame::BitDepth;
use core::arch::x86_64::{
    __m256i, __m512i, _mm_srli_si128, _mm256_add_epi32, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_mullo_epi32, _mm256_set1_epi32, _mm256_setr_epi32,
    _mm256_srai_epi32, _mm512_add_epi32, _mm512_castsi512_si256, _mm512_cvtepi64_epi16,
    _mm512_cvtsepi32_epi16, _mm512_cvtusepi32_epi8, _mm512_dpbusd_epi32, _mm512_dpwssd_epi32,
    _mm512_extracti64x4_epi64, _mm512_maskz_permutex2var_epi8, _mm512_max_epi32,
    _mm512_mullo_epi32, _mm512_permutexvar_epi8, _mm512_permutexvar_epi16, _mm512_set1_epi32,
    _mm512_setr_epi32, _mm512_setzero_si512, _mm512_srai_epi16, _mm512_srai_epi32,
    _mm512_srli_epi64, _mm512_zextsi128_si512,
};

/// Filtre indeksi `64 + ((t + 512) >> 10)`'un toplamsal sabiti (`t`'ye
/// eklenip 10 bit kaydırılır).
pub(super) const INDEKS_EK: i32 = 512 + (64 << 10);

/// `vpermb` dizinleri: qword `x` = satır baytları `x..x+8`.
static YATAY_SIRA: [u8; 64] = {
    let mut m = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        m[i] = (i / 8 + i % 8) as u8;
        i += 1;
    }
    m
};

/// `vpermw` dizinleri (dikey): 32 `i16`'lık pencere ara satır `b..b+4`'ten;
/// alt yarı dword `x` = `(mid[b][x], mid[b+1][x])`, üst yarı `(mid[b+1][x],
/// mid[b+2][x])`.
static DIKEY_SIRA: [i16; 32] = {
    let mut m = [0i16; 32];
    let mut w = 0;
    while w < 32 {
        let (h, i) = (w / 16, w % 16);
        m[w] = (8 * (h + i % 2) + i / 2) as i16;
        w += 1;
    }
    m
};

/// `vpermt2b` dizinleri (dikey filtre ikilisi `j`): dword `(h, x)` (`h` satır
/// yarısı, `x` sütun) bayt 1 = `f_x[2j]`, bayt 3 = `f_x[2j+1]` (ikinci kaynak
/// `h = 1`, +64); bayt 0 ve 2 sıfırlayan maskeyle 0 → `psraw 8` işaret
/// genişletir.
static DIKEY_FILTRE: [[u8; 64]; 4] = [
    dikey_filtre(0),
    dikey_filtre(1),
    dikey_filtre(2),
    dikey_filtre(3),
];

const fn dikey_filtre(j: usize) -> [u8; 64] {
    let mut m = [0u8; 64];
    let mut p = 0;
    while p < 64 {
        let d = p / 4;
        let (h, x) = (d / 8, d % 8);
        m[p] = match p % 4 {
            1 => (64 * h + 8 * x + 2 * j) as u8,
            3 => (64 * h + 8 * x + 2 * j + 1) as u8,
            _ => 0,
        };
        p += 1;
    }
    m
}

/// Tek baytları seçen sıfırlayan maske.
pub(super) const TEK_BAYT: u64 = 0xAAAA_AAAA_AAAA_AAAA;

/// `vpermt2b` + `psraw 8`: toplanmış qword filtrelerden (`f0`, `f1`: iki satır)
/// işaret genişletilmiş `i16` ikilileri.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
#[inline]
pub(super) fn filtre_ikilisi(f0: __m512i, dizin: __m512i, f1: __m512i) -> __m512i {
    _mm512_srai_epi16::<8>(_mm512_maskz_permutex2var_epi8(TEK_BAYT, f0, dizin, f1))
}

/// Yatay geçiş (8 bpc): 15 ara satır `mid[8r..8r+8]`, `(Σ + 4) >> 3`.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yatay(src: &[u8], ss: usize, abcd: &[i16; 4], mx: i32, mid: &mut [i16; 128]) {
    let (a, b) = (i32::from(abcd[0]), i32::from(abcd[1]));
    let xa = _mm256_mullo_epi32(
        _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
        _mm256_set1_epi32(a),
    );
    let sira = load_u8x64(&YATAY_SIRA);
    let yuv = _mm512_set1_epi32(4);
    let mut t = mx + INDEKS_EK;
    for r in 0..15 {
        let ix = _mm256_srai_epi32::<10>(_mm256_add_epi32(xa, _mm256_set1_epi32(t)));
        let f = gather_i64x8(&WARP_V, ix);
        let s =
            _mm512_permutexvar_epi8(sira, _mm512_zextsi128_si512(load_u8x16(chunk(src, r * ss))));
        let d = _mm512_dpbusd_epi32(_mm512_setzero_si512(), s, f);
        // qword `x`'in alt dword'ü: `Σ_{k<4} + Σ_{k≥4}`.
        let v = _mm512_add_epi32(d, _mm512_srli_epi64::<32>(d));
        let v = _mm512_srai_epi32::<3>(_mm512_add_epi32(v, yuv));
        store_i16x8(chunk_mut(mid, 8 * r), _mm512_cvtepi64_epi16(v));
        t += b;
    }
}

/// Dikey geçiş toplamları: çıkış satırı çifti `(2q, 2q+1)` için 16 `i32`
/// (dword `x` satır `2q`, `8 + x` satır `2q+1`), birikim `yuv`'dan başlar.
/// Ara satırlar `mid[8r..8r+8]` (`r` 0…14; satır 15 okunur, kullanılmaz).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
pub(super) fn dikey(mid: &[i16; 128], abcd: &[i16; 4], my: i32, yuv: __m512i) -> [__m512i; 4] {
    let (c, d) = (i32::from(abcd[2]), i32::from(abcd[3]));
    // dword `(h, x)` = `x·c + h·d`.
    let xc = _mm512_add_epi32(
        _mm512_mullo_epi32(
            _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7),
            _mm512_set1_epi32(c),
        ),
        _mm512_mullo_epi32(
            _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1),
            _mm512_set1_epi32(d),
        ),
    );
    let zs = load_i16x32(&DIKEY_SIRA);
    let gs: [__m512i; 4] = core::array::from_fn(|j| load_u8x64(&DIKEY_FILTRE[j]));
    let mut cikti = [_mm512_setzero_si512(); 4];
    let mut t = my + INDEKS_EK;
    for (q, cq) in cikti.iter_mut().enumerate() {
        let ix = _mm512_srai_epi32::<10>(_mm512_add_epi32(xc, _mm512_set1_epi32(t)));
        let f0 = gather_i64x8(&WARP_V, _mm512_castsi512_si256(ix));
        let f1 = gather_i64x8(&WARP_V, _mm512_extracti64x4_epi64::<1>(ix));
        let mut acc = yuv;
        for (j, g) in gs.iter().enumerate() {
            let z = _mm512_permutexvar_epi16(zs, load_i16x32(chunk(mid, 8 * (2 * q + 2 * j))));
            acc = _mm512_dpwssd_epi32(acc, z, filtre_ikilisi(f0, *g, f1));
        }
        *cq = acc;
        t += 2 * d;
    }
    cikti
}

/// Warp gövdesi: `T = false` → `warp8x8` (bayt), `true` → `warp8x8t` (`i16`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn warp_govde<const T: bool>(
    dst: &mut [u8],
    tmp: &mut [i16],
    ds: usize,
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
) {
    let mut mid = [0i16; 128];
    yatay(src, ss, abcd, mx, &mut mid);
    let toplam = dikey(&mid, abcd, my, _mm512_set1_epi32(if T { 64 } else { 1024 }));
    for (q, s) in toplam.into_iter().enumerate() {
        let y = 2 * q;
        if T {
            let v: __m256i = _mm512_cvtsepi32_epi16(_mm512_srai_epi32::<7>(s));
            store_i16x8(chunk_mut(tmp, y * ds), _mm256_castsi256_si128(v));
            store_i16x8(
                chunk_mut(tmp, (y + 1) * ds),
                _mm256_extracti128_si256::<1>(v),
            );
        } else {
            let v = _mm512_max_epi32(_mm512_srai_epi32::<11>(s), _mm512_setzero_si512());
            let b = _mm512_cvtusepi32_epi8(v);
            store_u8x8(chunk_mut(dst, y * ds), b);
            store_u8x8(chunk_mut(dst, (y + 1) * ds), _mm_srli_si128::<8>(b));
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

fn warp8x8_giris(
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
    if bd != BitDepth::Eight || !warp_sozlesmesi(src.len(), ss, abcd, mx, my, dst.len(), ds) {
        skaler_devri_say();
        return skaler!(bd, warp8x8(dst, ds, src, ss, abcd, mx, my, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`warp8x8_fn`); belirteç AVX512ICL (VBMI, VNNI dahil) tespitinin
    // kanıtıdır. (2) sınır: `warp_sozlesmesi` kaynak bölgesini (15 satır × 16
    // bayt, `MC_KUYRUK`), hedefte 8 satır × 8 baytı ve filtre indeksinin
    // `[0, 192]` aralığını kapsar; toplama dizinleri ayrıca denetlenir, erişimler
    // denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
    unsafe { warp_govde::<false>(dst, &mut [], ds, src, ss, abcd, mx, my) }
}

fn warp8x8t_giris(
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
    if bd != BitDepth::Eight || !warp_sozlesmesi(src.len(), ss, abcd, mx, my, tmp.len(), ts) {
        skaler_devri_say();
        return skaler!(bd, warp8x8t(tmp, ts, src, ss, abcd, mx, my, bd, bdmax));
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`warp8x8t_fn`); belirteç AVX512ICL tespitinin kanıtıdır. (2) sınır:
    // `warp_sozlesmesi` kaynak bölgesini, `tmp`'de 8 satır × 8 öğeyi ve filtre
    // indeksini kapsar; toplama dizinleri ayrıca denetlenir. (3) hizalama/takma
    // ad: `tmp` tek `&mut`, `src` `&`.
    unsafe { warp_govde::<true>(&mut [], tmp, ts, src, ss, abcd, mx, my) }
}

/// `warp8x8` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn warp8x8_fn(_avx512: Avx512IclToken) -> Warp8x8Fn {
    warp8x8_giris
}

/// `warp8x8t` AVX512ICL yuvası (yalnız belirteçle).
#[must_use]
pub fn warp8x8t_fn(_avx512: Avx512IclToken) -> Warp8x8tFn {
    warp8x8t_giris
}
