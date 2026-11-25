//! Film grain 16 bpc (`10`/`12` bit) x86_64 AVX512ICL çekirdekleri — F4-FG c (P42 Adım 3).
//!
//! Bu modül dav1d'nin `src/x86/filmgrain16_avx512.asm` dosyasındaki 4 AVX512ICL sembolünü
//! (`10` ve `12` bpc = 8 yuva) **saf Rust intrinsics** ile gerçekler; dış `.asm`/`.S`,
//! `asm!` ve `global_asm!` yoktur (CEK-02). Yalnız
//! `#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]` ve
//! [`Avx512IclToken`] kanıtı kullanılır.
//! (`dav1d`'de 16 bpc `generate_grain_*` için AVX512ICL çekirdeği yoktur.)
//!
//! ## P34 modül izolasyonu
//!
//! 8 bpc modülleriyle (`fg.rs`, `fg_ssse3.rs`, `fg_avx512.rs`) ve 16 bpc `fg16_ssse3.rs` /
//! `fg16.rs` modülleriyle sıcak yardımcı paylaşılmaz; mevcut modüllerin CGU bölümlemesi
//! ve makine kodu değişmez.
//!
//! ## `SLOW_GATHER` paritesi (`dav1d` `src/x86/filmgrain.h:76`)
//!
//! 8 bpc AVX512ICL `fgy`/`fguv` ölçekleme tablosu aramasını `vpermb` ile yaptığından
//! `SLOW_GATHER` açıkken de kurulur; 16 bpc (`10`/`12` bpc) AVX512ICL `fgy`/`fguv` ise
//! `1024`/`4096` baytlık tabloyu `vpgatherdd` ile topladığından `SLOW_GATHER` (`0x20`,
//! örn. Zen 4) açıkken **kurulmaz**.
//!
//! ## Sınır güvenli 512-bit `vpgatherdd` (`&[u8; 1024]` / `&[u8; 4096]`)
//!
//! `dav1d`'nin `filmgrain16_avx512.asm` dosyası `vpgatherdd` ile `[scalingq + m16]`
//! (`SCALE = 1`, `N − 1`'de +3 bayt taşma) okur; bu modülde tablo dışına okuma yasaktır.
//! Ölçekleme tablosu [`olcekleme16_gather_512`] ve [`olcekleme16_gather`] üzerinden her
//! dizin `& (N − 1)` ile maskelenip `idx >> 2` dword diziniyle (`SCALE = 4`, 4 bayt hizalı)
//! tam `&[u8; 1024]` (`B10`) ve `&[u8; 4096]` (`B12`) sınırları içinde toplanır.
//!
//! ## Sembol eşlemesi (dav1d @ `99f3354`)
//!
//! | ID | İşlev | dav1d sembolü |
//! |---|---|---|
//! | D01571 | `fgy_32x32xn` | `dav1d_fgy_32x32xn_16bpc_avx512icl` |
//! | D01553 | `fguv_32x32xn[I420]` | `dav1d_fguv_32x32xn_i420_16bpc_avx512icl` |
//! | D01559 | `fguv_32x32xn[I422]` | `dav1d_fguv_32x32xn_i422_16bpc_avx512icl` |
//! | D01565 | `fguv_32x32xn[I444]` | `dav1d_fguv_32x32xn_i444_16bpc_avx512icl` |

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use core::arch::x86_64::*;

use super::mem::{
    chunk, chunk_mut, load_i16x8, load_i16x16, load_i16x32, load_u8x32, load_u8x64,
    olcekleme16_gather, olcekleme16_gather_512, store_u8x32, store_u8x64,
};
use crate::bitstream::frame_header::FilmGrainData;
use crate::cpu::Avx512IclToken;
use crate::dsp::fg::{
    FG_BLOCK_SIZE, FgBitDepth, FguvFn, FgyFn, GrainLut, fg_skaler_devri_say, fguv16_sozlesmesi,
    fgy16_sozlesmesi,
};
use crate::dsp::scalar::fg::{self as scalar, get_random_number, round2, sample_lut};

/// `fguv` kroma ölçekleme katsayılarının AV1 sözleşmesi (`i16` taşmasızlığı).
#[inline]
fn uv_katsayi_sozlesmesi(data: &FilmGrainData, uv: usize) -> bool {
    data.chroma_scaling_from_luma
        || ((-128..=127).contains(&data.uv_luma_mult[uv])
            && (-128..=127).contains(&data.uv_mult[uv])
            && (-256..=255).contains(&data.uv_offset[uv]))
}

/// Satır PRNG başlangıç tohumlarını (`0` = bu satır, `1` = üst satır) döngüsüz üretir.
#[inline(always)]
const fn satir_tohumlari(data: &FilmGrainData, row_num: usize, overlap: bool) -> (u32, u32) {
    let r0 = row_num & 0xFFFF;
    let s0 = data.seed ^ ((((r0 * 37 + 178) & 0xFF) << 8) | ((r0 * 173 + 105) & 0xFF)) as u32;
    let s1 = if overlap && row_num != 0 {
        let r1 = (row_num - 1) & 0xFFFF;
        data.seed ^ ((((r1 * 37 + 178) & 0xFF) << 8) | ((r1 * 173 + 105) & 0xFF)) as u32
    } else {
        0
    };
    (s0, s1)
}

/// Grain LUT'undan `(offy, offx)` konumundaki 32 ardışık `i16` değerini ZMM'e yükler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn lut_32(lut: &GrainLut, y: usize, x: usize) -> __m512i {
    load_i16x32(chunk(&lut[y], x))
}

/// Grain LUT'undan `(offy, offx)` konumundaki 16 ardışık `i16` değerini YMM'e yükler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn lut_16(lut: &GrainLut, y: usize, x: usize) -> __m256i {
    load_i16x16(chunk(&lut[y], x))
}

/// Grain LUT'undan `(offy, offx)` konumundaki 8 ardışık `i16` değerini XMM'e yükler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn lut_8(lut: &GrainLut, y: usize, x: usize) -> __m128i {
    load_i16x8(chunk(&lut[y], x))
}

/// LUT ofset çifti `(offx, offy)`. `randval as u8` üzerinden derleme zamanında
/// `offx <= 39`, `offy <= 39` üst sınırını kanıtlar (`grain_lut` sınır denetimini düşürür).
#[inline(always)]
const fn lut_ofset(
    offsets: &[[i32; 2]; 2],
    subx: usize,
    suby: usize,
    bx: usize,
    by: usize,
) -> (usize, usize) {
    let randval = offsets[bx][by] as u8;
    let offx = 3 + (2 >> subx) * (3 + ((randval >> 4) as usize));
    let offy = 3 + (2 >> suby) * (3 + ((randval & 0xF) as usize));
    (
        offx + (FG_BLOCK_SIZE >> subx) * bx,
        offy + (FG_BLOCK_SIZE >> suby) * by,
    )
}

/// Kural 2 (512-bit): 32 adet `16` bit piksele (`s`) ölçeklenmiş grain gürültüsü ekler ve
/// `[vmin, vmax]`'a kırpar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn gurultu_ekle_32(
    s: __m512i,
    sc: __m512i,
    g: __m512i,
    carpan: __m512i,
    vmin: __m512i,
    vmax: __m512i,
) -> __m512i {
    let sc_sh = _mm512_mullo_epi16(sc, carpan);
    let gurultu = _mm512_mulhrs_epi16(sc_sh, g);
    let toplam = _mm512_add_epi16(s, gurultu);
    _mm512_max_epi16(_mm512_min_epi16(toplam, vmax), vmin)
}

/// Kural 2 (256-bit): 16 adet `16` bit piksele (`s`) ölçeklenmiş grain gürültüsü ekler ve
/// `[vmin, vmax]`'a kırpar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn gurultu_ekle_16(
    s: __m256i,
    sc: __m256i,
    g: __m256i,
    carpan: __m256i,
    vmin: __m256i,
    vmax: __m256i,
) -> __m256i {
    let sc_sh = _mm256_mullo_epi16(sc, carpan);
    let gurultu = _mm256_mulhrs_epi16(sc_sh, g);
    let toplam = _mm256_add_epi16(s, gurultu);
    _mm256_max_epi16(_mm256_min_epi16(toplam, vmax), vmin)
}

/// Kural 3 (512-bit): İki 32'li `i16` grain vektörünü (`eski`, `yeni`) `(w0, w1)` ağırlıklarıyla
/// `_mm512_madd_epi16` (`i32` birikim, `12` bpc'de taşmasız) üzerinden harmanlar ve
/// `[GRAIN_MIN, GRAIN_MAX]`'a kırpar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn harmanla_32<const BPC: u32>(eski: __m512i, yeni: __m512i, w: [i16; 2]) -> __m512i {
    let gmin = -(1i16 << (BPC - 1));
    let gmax = (1i16 << (BPC - 1)) - 1;
    let w32 = _mm512_set1_epi32((i32::from(w[0]) & 0xFFFF) | (i32::from(w[1]) << 16));
    let yuv = _mm512_set1_epi32(16);
    let lo = _mm512_srai_epi32::<5>(_mm512_add_epi32(
        _mm512_madd_epi16(_mm512_unpacklo_epi16(eski, yeni), w32),
        yuv,
    ));
    let hi = _mm512_srai_epi32::<5>(_mm512_add_epi32(
        _mm512_madd_epi16(_mm512_unpackhi_epi16(eski, yeni), w32),
        yuv,
    ));
    let p = _mm512_packs_epi32(lo, hi);
    _mm512_max_epi16(
        _mm512_min_epi16(p, _mm512_set1_epi16(gmax)),
        _mm512_set1_epi16(gmin),
    )
}

/// Kural 3 (`422` dikey örtüşme, 512-bit): Alt 256 bitte `y = 0` (`[27, 17]`), üst 256 bitte
/// `y = 1` (`[17, 27]`) ağırlıklarıyla iki satırı tek ZMM geçişinde harmanlar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn harmanla_32_422<const BPC: u32>(eski: __m512i, yeni: __m512i) -> __m512i {
    let gmin = -(1i16 << (BPC - 1));
    let gmax = (1i16 << (BPC - 1)) - 1;
    let w0 = _mm256_set1_epi32(27 | (17 << 16));
    let w1 = _mm256_set1_epi32(17 | (27 << 16));
    let w32 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(w0), w1);
    let yuv = _mm512_set1_epi32(16);
    let lo = _mm512_srai_epi32::<5>(_mm512_add_epi32(
        _mm512_madd_epi16(_mm512_unpacklo_epi16(eski, yeni), w32),
        yuv,
    ));
    let hi = _mm512_srai_epi32::<5>(_mm512_add_epi32(
        _mm512_madd_epi16(_mm512_unpackhi_epi16(eski, yeni), w32),
        yuv,
    ));
    let p = _mm512_packs_epi32(lo, hi);
    _mm512_max_epi16(
        _mm512_min_epi16(p, _mm512_set1_epi16(gmax)),
        _mm512_set1_epi16(gmin),
    )
}

/// İki 32'li ZMM satırının (`g0`, `g1`) sol kenarındaki ilk 2 sütunu (`x = 0, 1`)
/// `sol0`, `sol1` ile tek XMM geçişinde (`[[27, 17], [17, 27]]`) harmanlar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn yatay_harmanla_cift_32<const BPC: u32>(
    sol0: __m128i,
    sol1: __m128i,
    g0: __m512i,
    g1: __m512i,
) -> (__m512i, __m512i) {
    let gmin = -(1i16 << (BPC - 1));
    let gmax = (1i16 << (BPC - 1)) - 1;
    let sol = _mm_unpacklo_epi32(sol0, sol1);
    let g_lo = _mm_unpacklo_epi32(_mm512_castsi512_si128(g0), _mm512_castsi512_si128(g1));
    let cift = _mm_unpacklo_epi16(sol, g_lo);
    let w = _mm_setr_epi32(
        27 | (17 << 16),
        17 | (27 << 16),
        27 | (17 << 16),
        17 | (27 << 16),
    );
    let m = _mm_srai_epi32::<5>(_mm_add_epi32(_mm_madd_epi16(cift, w), _mm_set1_epi32(16)));
    let p = _mm_packs_epi32(m, m);
    let k = _mm512_castsi128_si512(_mm_max_epi16(
        _mm_min_epi16(p, _mm_set1_epi16(gmax)),
        _mm_set1_epi16(gmin),
    ));
    (
        _mm512_mask_blend_epi32(0x0001, g0, k),
        _mm512_mask_shuffle_epi32::<0x55>(g1, 0x0001, k),
    )
}

/// Dört 32'li ZMM satırının (`g0..g3`) sol kenarındaki ilk 2 sütunu (`x = 0, 1`)
/// `sol0..sol3` ile tek YMM geçişinde (`[[27, 17], [17, 27]]`) harmanlar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn yatay_harmanla_dort_32<const BPC: u32>(
    sol0: __m128i,
    sol1: __m128i,
    sol2: __m128i,
    sol3: __m128i,
    g0: __m512i,
    g1: __m512i,
    g2: __m512i,
    g3: __m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
    let gmin = -(1i16 << (BPC - 1));
    let gmax = (1i16 << (BPC - 1)) - 1;
    let sol02 = _mm_unpacklo_epi32(sol0, sol2);
    let sol13 = _mm_unpacklo_epi32(sol1, sol3);
    let sol = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(sol02), sol13);
    let g02 = _mm_unpacklo_epi32(_mm512_castsi512_si128(g0), _mm512_castsi512_si128(g2));
    let g13 = _mm_unpacklo_epi32(_mm512_castsi512_si128(g1), _mm512_castsi512_si128(g3));
    let g_lo = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(g02), g13);
    let cift = _mm256_unpacklo_epi16(sol, g_lo);
    let w = _mm256_broadcastsi128_si256(_mm_setr_epi32(
        27 | (17 << 16),
        17 | (27 << 16),
        27 | (17 << 16),
        17 | (27 << 16),
    ));
    let m = _mm256_srai_epi32::<5>(_mm256_add_epi32(
        _mm256_madd_epi16(cift, w),
        _mm256_set1_epi32(16),
    ));
    let p = _mm256_packs_epi32(m, m);
    let k = _mm256_max_epi16(
        _mm256_min_epi16(p, _mm256_set1_epi16(gmax)),
        _mm256_set1_epi16(gmin),
    );
    let k02 = _mm512_castsi256_si512(k);
    let k13 = _mm512_castsi128_si512(_mm256_extracti128_si256::<1>(k));
    (
        _mm512_mask_blend_epi32(0x0001, g0, k02),
        _mm512_mask_blend_epi32(0x0001, g1, k13),
        _mm512_mask_shuffle_epi32::<0x55>(g2, 0x0001, k02),
        _mm512_mask_shuffle_epi32::<0x55>(g3, 0x0001, k13),
    )
}

/// Kural 3 (256-bit): İki 16'lı `i16` grain vektörünü (`eski`, `yeni`) `(w0, w1)` ağırlıklarıyla
/// `_mm256_madd_epi16` üzerinden harmanlar ve `[GRAIN_MIN, GRAIN_MAX]`'a kırpar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn harmanla_16<const BPC: u32>(eski: __m256i, yeni: __m256i, w: [i16; 2]) -> __m256i {
    let gmin = -(1i16 << (BPC - 1));
    let gmax = (1i16 << (BPC - 1)) - 1;
    let w32 = _mm256_set1_epi32((i32::from(w[0]) & 0xFFFF) | (i32::from(w[1]) << 16));
    let yuv = _mm256_set1_epi32(16);
    let lo = _mm256_srai_epi32::<5>(_mm256_add_epi32(
        _mm256_madd_epi16(_mm256_unpacklo_epi16(eski, yeni), w32),
        yuv,
    ));
    let hi = _mm256_srai_epi32::<5>(_mm256_add_epi32(
        _mm256_madd_epi16(_mm256_unpackhi_epi16(eski, yeni), w32),
        yuv,
    ));
    let p = _mm256_packs_epi32(lo, hi);
    _mm256_max_epi16(
        _mm256_min_epi16(p, _mm256_set1_epi16(gmax)),
        _mm256_set1_epi16(gmin),
    )
}

/// Blok sol kenarındaki ilk 2 sütunu (`x = 0, 1`) sol bloktan gelen `sol2` ile
/// `[[27, 17], [17, 27]]` ağırlıklarıyla `i32` üzerinden harmanlar ve 32'li `g` ZMM
/// vektörünün ilk iki `i16` şeridine yazar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn yatay_harmanla_32<const BPC: u32>(sol2: __m128i, g: __m512i) -> __m512i {
    let gmin = -(1i16 << (BPC - 1));
    let gmax = (1i16 << (BPC - 1)) - 1;
    let g_lo = _mm512_castsi512_si128(g);
    let cift = _mm_unpacklo_epi16(sol2, g_lo);
    let w = _mm_set_epi16(0, 0, 0, 0, 27, 17, 17, 27);
    let m = _mm_srai_epi32::<5>(_mm_add_epi32(_mm_madd_epi16(cift, w), _mm_set1_epi32(16)));
    let p = _mm_packs_epi32(m, m);
    let k = _mm_max_epi16(_mm_min_epi16(p, _mm_set1_epi16(gmax)), _mm_set1_epi16(gmin));
    _mm512_mask_blend_epi32(0x0001, g, _mm512_castsi128_si512(k))
}

/// Blok sol kenarındaki ilk 2 sütunu (`x = 0, 1`) 16'lı `g` YMM vektörüne harmanlar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn yatay_harmanla_16<const BPC: u32>(sol2: __m128i, g: __m256i) -> __m256i {
    let gmin = -(1i16 << (BPC - 1));
    let gmax = (1i16 << (BPC - 1)) - 1;
    let g_lo = _mm256_castsi256_si128(g);
    let cift = _mm_unpacklo_epi16(sol2, g_lo);
    let w = _mm_set_epi16(0, 0, 0, 0, 27, 17, 17, 27);
    let m = _mm_srai_epi32::<5>(_mm_add_epi32(_mm_madd_epi16(cift, w), _mm_set1_epi32(16)));
    let p = _mm_packs_epi32(m, m);
    let k = _mm_max_epi16(_mm_min_epi16(p, _mm_set1_epi16(gmax)), _mm_set1_epi16(gmin));
    _mm256_blend_epi32::<0x01>(g, _mm256_castsi128_si256(k))
}

/// 32 piksellik (`64` bayt) tam luma satırını 512-bit ZMM ile işler ve `d64`'e yazar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn fgy_32_parca<const N: usize>(
    d64: &mut [u8; 64],
    s64: &[u8; 64],
    scaling: &[u8; N],
    g: __m512i,
    carpan: __m512i,
    vmin: __m512i,
    vmax: __m512i,
) {
    let s = load_u8x64(s64);
    let sc = olcekleme16_gather_512::<N>(scaling, s);
    let out = gurultu_ekle_32(s, sc, g, carpan, vmin, vmax);
    store_u8x64(d64, out);
}

/// 16 piksellik (`32` bayt) luma parçasını 256-bit YMM ile işler ve `d32`'ye yazar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn fgy_16_parca<const N: usize>(
    d32: &mut [u8; 32],
    s32: &[u8; 32],
    scaling: &[u8; N],
    g: __m256i,
    carpan: __m256i,
    vmin: __m256i,
    vmax: __m256i,
) {
    let s = load_u8x32(s32);
    let sc = olcekleme16_gather::<N>(scaling, s);
    let out = gurultu_ekle_16(s, sc, g, carpan, vmin, vmax);
    store_u8x32(d32, out);
}

/// Tam 32 piksellik (`64` bayt) luma bloğunu (`0..ystart` dikey örtüşme dahil) 512-bit ZMM
/// ile işler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline(never)]
fn fgy_blok_32_512<const BPC: u32, const N: usize>(
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
    carpan: __m512i,
    vmin: __m512i,
    vmax: __m512i,
) {
    assert!((1..=FG_BLOCK_SIZE).contains(&bh) && bx + FG_BLOCK_SIZE <= (1 << 16));
    assert!(stride >= (bx + FG_BLOCK_SIZE) * 2 && stride <= (1 << 20));
    let gerekli = (bh - 1) * stride + (bx + FG_BLOCK_SIZE) * 2;
    assert!(src.len() >= gerekli && dst.len() >= gerekli);
    assert!(ystart <= 2 && ystart <= bh);
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

    for y in 0..ystart {
        let off = y * stride + bx * 2;
        let mut ust = lut_32(grain_lut, oy01 + y, ox01);
        let mut cur = lut_32(grain_lut, oy00 + y, ox00);
        if xstart != 0 {
            let sol_ust = lut_8(grain_lut, oy11 + y, ox11);
            let sol_cur = lut_8(grain_lut, oy10 + y, ox10);
            ust = yatay_harmanla_32::<BPC>(sol_ust, ust);
            cur = yatay_harmanla_32::<BPC>(sol_cur, cur);
        }
        let g = harmanla_32::<BPC>(ust, cur, W[y]);
        fgy_32_parca::<N>(
            chunk_mut(dst, off),
            chunk(src, off),
            scaling,
            g,
            carpan,
            vmin,
            vmax,
        );
    }

    if xstart != 0 {
        for y in ystart..bh {
            let off = y * stride + bx * 2;
            let sol = lut_8(grain_lut, oy10 + y, ox10);
            let g = yatay_harmanla_32::<BPC>(sol, lut_32(grain_lut, oy00 + y, ox00));
            fgy_32_parca::<N>(
                chunk_mut(dst, off),
                chunk(src, off),
                scaling,
                g,
                carpan,
                vmin,
                vmax,
            );
        }
    } else {
        for y in ystart..bh {
            let off = y * stride + bx * 2;
            let g = lut_32(grain_lut, oy00 + y, ox00);
            fgy_32_parca::<N>(
                chunk_mut(dst, off),
                chunk(src, off),
                scaling,
                g,
                carpan,
                vmin,
                vmax,
            );
        }
    }
}

/// Kısmi (`16 <= bw < 32`) luma bloğunu 16'lı YMM pencereleriyle (`0` ve `bw - 16`) işler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline(never)]
fn fgy_kismi_16<const BPC: u32, const N: usize>(
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
    carpan: __m256i,
    vmin: __m256i,
    vmax: __m256i,
) {
    assert!((16..FG_BLOCK_SIZE).contains(&bw) && (1..=FG_BLOCK_SIZE).contains(&bh));
    assert!(bx + bw <= (1 << 16) && stride >= (bx + bw) * 2 && stride <= (1 << 20));
    let gerekli = (bh - 1) * stride + (bx + bw) * 2;
    assert!(src.len() >= gerekli && dst.len() >= gerekli);
    assert!(ystart <= 2 && ystart <= bh);
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

    let dx1 = bw - 16;
    for y in 0..bh {
        let off = y * stride + bx * 2;
        if y < ystart {
            let mut ust0 = lut_16(grain_lut, oy01 + y, ox01);
            let mut cur0 = lut_16(grain_lut, oy00 + y, ox00);
            if xstart != 0 {
                let sol_ust = lut_8(grain_lut, oy11 + y, ox11);
                let sol_cur = lut_8(grain_lut, oy10 + y, ox10);
                ust0 = yatay_harmanla_16::<BPC>(sol_ust, ust0);
                cur0 = yatay_harmanla_16::<BPC>(sol_cur, cur0);
            }
            let g0 = harmanla_16::<BPC>(ust0, cur0, W[y]);
            fgy_16_parca::<N>(
                chunk_mut(dst, off),
                chunk(src, off),
                scaling,
                g0,
                carpan,
                vmin,
                vmax,
            );
            if dx1 > 0 {
                let ust1 = lut_16(grain_lut, oy01 + y, ox01 + dx1);
                let cur1 = lut_16(grain_lut, oy00 + y, ox00 + dx1);
                let mut g1 = harmanla_16::<BPC>(ust1, cur1, W[y]);
                if dx1 < xstart {
                    let g0_s1 = _mm_srli_epi32::<16>(_mm256_castsi256_si128(g0));
                    let g1_lo = _mm_blend_epi16::<0x01>(_mm256_castsi256_si128(g1), g0_s1);
                    g1 = _mm256_blend_epi32::<0x01>(g1, _mm256_castsi128_si256(g1_lo));
                }
                fgy_16_parca::<N>(
                    chunk_mut(dst, off + dx1 * 2),
                    chunk(src, off + dx1 * 2),
                    scaling,
                    g1,
                    carpan,
                    vmin,
                    vmax,
                );
            }
        } else {
            let mut g0 = lut_16(grain_lut, oy00 + y, ox00);
            if xstart != 0 {
                let sol = lut_8(grain_lut, oy10 + y, ox10);
                g0 = yatay_harmanla_16::<BPC>(sol, g0);
            }
            fgy_16_parca::<N>(
                chunk_mut(dst, off),
                chunk(src, off),
                scaling,
                g0,
                carpan,
                vmin,
                vmax,
            );
            if dx1 > 0 {
                let mut g1 = lut_16(grain_lut, oy00 + y, ox00 + dx1);
                if dx1 < xstart {
                    let g0_s1 = _mm_srli_epi32::<16>(_mm256_castsi256_si128(g0));
                    let g1_lo = _mm_blend_epi16::<0x01>(_mm256_castsi256_si128(g1), g0_s1);
                    g1 = _mm256_blend_epi32::<0x01>(g1, _mm256_castsi128_si256(g1_lo));
                }
                fgy_16_parca::<N>(
                    chunk_mut(dst, off + dx1 * 2),
                    chunk(src, off + dx1 * 2),
                    scaling,
                    g1,
                    carpan,
                    vmin,
                    vmax,
                );
            }
        }
    }
}

/// Kısmi (`bw < 16`) luma bloğunu tam sınırlar içinde (`0..bw`) işler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
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
    let ust_var = overlap && row_num > 0;
    let min_8 = BPC - 8;
    let max = (1i32 << BPC) - 1;
    let (min_value, max_value) = if data.clip_to_restricted_range {
        (16i32 << min_8, 235i32 << min_8)
    } else {
        (0, max)
    };
    let shift = data.scaling_shift as u32;
    let (mut s0, mut s1) = satir_tohumlari(data, row_num, overlap);
    let mut offsets = [[0i32; 2]; 2];
    let carpan512 = _mm512_set1_epi16(1 << (15 - shift));
    let vmin512 = _mm512_set1_epi16(min_value as i16);
    let vmax512 = _mm512_set1_epi16(max_value as i16);
    let carpan256 = _mm512_castsi512_si256(carpan512);
    let vmin256 = _mm512_castsi512_si256(vmin512);
    let vmax256 = _mm512_castsi512_si256(vmax512);

    let ystart = if ust_var { bh.min(2) } else { 0 };
    let mut bx = 0usize;
    while bx + FG_BLOCK_SIZE <= pw {
        if overlap && bx != 0 {
            offsets[1][0] = offsets[0][0];
            if ust_var {
                offsets[1][1] = offsets[0][1];
            }
        }
        offsets[0][0] = get_random_number(8, &mut s0);
        if ust_var {
            offsets[0][1] = get_random_number(8, &mut s1);
        }
        let xstart = if overlap && bx != 0 { 2 } else { 0 };
        fgy_blok_32_512::<BPC, N>(
            dst, src, stride, scaling, grain_lut, &offsets, bx, bh, xstart, ystart, carpan512,
            vmin512, vmax512,
        );
        bx += FG_BLOCK_SIZE;
    }
    if bx < pw {
        let bw = pw - bx;
        if overlap && bx != 0 {
            offsets[1][0] = offsets[0][0];
            if ust_var {
                offsets[1][1] = offsets[0][1];
            }
        }
        offsets[0][0] = get_random_number(8, &mut s0);
        if ust_var {
            offsets[0][1] = get_random_number(8, &mut s1);
        }
        let xstart = if overlap && bx != 0 { bw.min(2) } else { 0 };
        if bw >= 16 {
            fgy_kismi_16::<BPC, N>(
                dst, src, stride, scaling, grain_lut, &offsets, bx, bw, bh, xstart, ystart,
                carpan256, vmin256, vmax256,
            );
        } else {
            fgy_kismi_blok_16::<BPC, N>(
                dst, src, stride, scaling, grain_lut, &offsets, bx, bw, bh, xstart, ystart, shift,
                min_value, max_value,
            );
        }
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`fgy_fn`); belirteç AVX512ICL tespiti + maske izninin kanıtıdır. (2) sınır:
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

/// Kroma blok sol kenarında yatay örtüşme (16'lı YMM): `SX == 1`'de yalnız 0. sütun
/// `[23, 22]` ile, `SX == 0`'da 0. ve 1. sütunlar `[[27, 17], [17, 27]]` ile harmanlanır.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn yatay_harmanla_uv_16<const BPC: u32, const SX: usize>(sol: __m128i, g: __m256i) -> __m256i {
    if SX == 0 {
        yatay_harmanla_16::<BPC>(sol, g)
    } else {
        let gmin = -(1i16 << (BPC - 1));
        let gmax = (1i16 << (BPC - 1)) - 1;
        let g_lo = _mm256_castsi256_si128(g);
        let cift = _mm_unpacklo_epi16(sol, g_lo);
        let w = _mm_set1_epi32(23 | (22 << 16));
        let m = _mm_srai_epi32::<5>(_mm_add_epi32(_mm_madd_epi16(cift, w), _mm_set1_epi32(16)));
        let p = _mm_packs_epi32(m, m);
        let k = _mm_max_epi16(_mm_min_epi16(p, _mm_set1_epi16(gmax)), _mm_set1_epi16(gmin));
        let b = _mm_blend_epi16::<0x01>(g_lo, k);
        _mm256_blend_epi32::<0x01>(g, _mm256_castsi128_si256(b))
    }
}

/// Kroma `SX == 1` blok sol kenarında iki satırın (`g0`, `g1`) 0. sütunlarını
/// sol bloktan gelen `sol0`, `sol1` ile `[23, 22]` ağırlıklarıyla 128-bit XMM
/// geçişinde harmanlar ve `512`-bit birleşik `(g0, g1)` vektörünü döndürür.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn yatay_harmanla_uv_cift_16<const BPC: u32>(
    sol0: __m128i,
    sol1: __m128i,
    g0: __m256i,
    g1: __m256i,
) -> __m512i {
    let gmin = -(1i16 << (BPC - 1));
    let gmax = (1i16 << (BPC - 1)) - 1;
    let g = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(g0), g1);
    let g0_lo = _mm512_castsi512_si128(g);
    let g1_lo = _mm256_castsi256_si128(g1);
    let s01 = _mm_unpacklo_epi16(sol0, sol1);
    let g01 = _mm_unpacklo_epi16(g0_lo, g1_lo);
    let cift = _mm_unpacklo_epi16(s01, g01);
    let w = _mm_set1_epi32(23 | (22 << 16));
    let m = _mm_srai_epi32::<5>(_mm_add_epi32(_mm_madd_epi16(cift, w), _mm_set1_epi32(16)));
    let p = _mm_packs_epi32(m, m);
    let k = _mm_max_epi16(_mm_min_epi16(p, _mm_set1_epi16(gmax)), _mm_set1_epi16(gmin));
    let k_hi = _mm_srli_epi32::<16>(k);
    let k512 = _mm512_inserti32x4::<2>(_mm512_castsi128_si512(k), k_hi);
    let kmask = _mm512_setr_epi32(0xFFFF, 0, 0, 0, 0, 0, 0, 0, 0xFFFF, 0, 0, 0, 0, 0, 0, 0);
    _mm512_ternarylogic_epi32::<0xD8>(g, k512, kmask)
}

/// Kroma `4:2:2` (`SX == 1, SY == 0`) köşe örtüşmesinde 4 satırın (`ust0`, `ust1`,
/// `cur0`, `cur1`) 0. sütunlarını sol komşularıyla (`sol_ust0..sol_cur1`) `[23, 22]`
/// ağırlıklarıyla tek 128-bit XMM geçişinde harmanlar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn yatay_harmanla_uv_dort_16<const BPC: u32>(
    sol_ust0: __m128i,
    sol_ust1: __m128i,
    sol_cur0: __m128i,
    sol_cur1: __m128i,
    ust0: __m256i,
    ust1: __m256i,
    cur0: __m256i,
    cur1: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let gmin = -(1i16 << (BPC - 1));
    let gmax = (1i16 << (BPC - 1)) - 1;
    let s01 = _mm_unpacklo_epi16(sol_ust0, sol_ust1);
    let s23 = _mm_unpacklo_epi16(sol_cur0, sol_cur1);
    let sol_4 = _mm_unpacklo_epi32(s01, s23);
    let u0_lo = _mm256_castsi256_si128(ust0);
    let u1_lo = _mm256_castsi256_si128(ust1);
    let c0_lo = _mm256_castsi256_si128(cur0);
    let c1_lo = _mm256_castsi256_si128(cur1);
    let g01 = _mm_unpacklo_epi16(u0_lo, u1_lo);
    let g23 = _mm_unpacklo_epi16(c0_lo, c1_lo);
    let g_4 = _mm_unpacklo_epi32(g01, g23);
    let cift = _mm_unpacklo_epi16(sol_4, g_4);
    let w = _mm_set1_epi32(23 | (22 << 16));
    let m = _mm_srai_epi32::<5>(_mm_add_epi32(_mm_madd_epi16(cift, w), _mm_set1_epi32(16)));
    let p = _mm_packs_epi32(m, m);
    let k = _mm_max_epi16(_mm_min_epi16(p, _mm_set1_epi16(gmax)), _mm_set1_epi16(gmin));
    let b_u0 = _mm_blend_epi16::<0x01>(u0_lo, k);
    let b_u1 = _mm_blend_epi16::<0x01>(u1_lo, _mm_srli_epi64::<16>(k));
    let b_c0 = _mm_blend_epi16::<0x01>(c0_lo, _mm_srli_epi64::<32>(k));
    let b_c1 = _mm_blend_epi16::<0x01>(c1_lo, _mm_srli_epi64::<48>(k));
    (
        _mm256_blend_epi32::<0x01>(ust0, _mm256_castsi128_si256(b_u0)),
        _mm256_blend_epi32::<0x01>(ust1, _mm256_castsi128_si256(b_u1)),
        _mm256_blend_epi32::<0x01>(cur0, _mm256_castsi128_si256(b_c0)),
        _mm256_blend_epi32::<0x01>(cur1, _mm256_castsi128_si256(b_c1)),
    )
}

/// 32 kroma pikselinin (`s` ve `avg` 512-bit ZMM) ölçekleme tablosu dizinlerini hesaplar ve
/// 512-bit `vpgatherdd` ile ölçekleme değerlerini toplar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn fguv_32_olcek<const N: usize, const CSFL: bool>(
    s: __m512i,
    avg: __m512i,
    scaling: &[u8; N],
    vbdmax: __m512i,
    carpan_uv: __m512i,
    v_ofs32: __m512i,
) -> __m512i {
    let idx = if CSFL {
        _mm512_min_epu16(avg, vbdmax)
    } else {
        let lo = _mm512_srai_epi32::<6>(_mm512_add_epi32(
            _mm512_madd_epi16(_mm512_unpacklo_epi16(avg, s), carpan_uv),
            v_ofs32,
        ));
        let hi = _mm512_srai_epi32::<6>(_mm512_add_epi32(
            _mm512_madd_epi16(_mm512_unpackhi_epi16(avg, s), carpan_uv),
            v_ofs32,
        ));
        _mm512_min_epu16(_mm512_packus_epi32(lo, hi), vbdmax)
    };
    olcekleme16_gather_512::<N>(scaling, idx)
}

/// 32 kroma pikselinin (`s` ve `avg` 512-bit ZMM) ölçekleme tablosu dizinlerini hesaplar,
/// 512-bit `vpgatherdd` ile ölçekleme değerlerini toplar ve gürültü eklenmiş pikselleri döndürür.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn fguv_32_hesapla<const N: usize, const CSFL: bool>(
    s: __m512i,
    avg: __m512i,
    scaling: &[u8; N],
    g: __m512i,
    carpan: __m512i,
    vmin: __m512i,
    vmax: __m512i,
    vbdmax: __m512i,
    carpan_uv: __m512i,
    v_ofs32: __m512i,
) -> __m512i {
    let sc = fguv_32_olcek::<N, CSFL>(s, avg, scaling, vbdmax, carpan_uv, v_ofs32);
    gurultu_ekle_32(s, sc, g, carpan, vmin, vmax)
}

/// `luma` yatay 2:1 alt-örneklemesi için çift `u16` dizinleri (`[0, 2, 4, ..., 62]`).
const PW_EVEN: [i16; 32] = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48,
    50, 52, 54, 56, 58, 60, 62,
];

/// `luma` yatay 2:1 alt-örneklemesi için tek `u16` dizinleri (`[1, 3, 5, ..., 63]`).
const PW_ODD: [i16; 32] = [
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49,
    51, 53, 55, 57, 59, 61, 63,
];

/// Tam genişlikli (`32 >> SX`: 16 veya 32 piksel) ve tam yükseklikli (`32 >> SY`: 16 veya 32 satır)
/// kroma bloğunu 512-bit ZMM ile işler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline]
fn fguv_blok_tam_512<
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
    xstart: usize,
    ystart: usize,
    carpan: __m512i,
    vmin: __m512i,
    vmax: __m512i,
    vbdmax: __m512i,
    carpan_uv: __m512i,
    v_ofs32: __m512i,
    pw_even: __m512i,
    pw_odd: __m512i,
) {
    let bstep = FG_BLOCK_SIZE >> SX;
    let bh = FG_BLOCK_SIZE >> SY;
    assert!(bx + bstep <= (1 << 16));
    assert!(stride >= (bx + bstep) * 2 && stride <= 0xFFFFF);
    assert!(luma_stride >= ((bx + bstep) << SX) * 2 && luma_stride <= 0xFFFFF);
    let stride = stride & 0xFFFFF;
    let luma_stride = luma_stride & 0xFFFFF;
    let s_span = stride + bstep * 2;
    let lstep = luma_stride << SY;
    let l_span = lstep + (bstep << SX) * 2;
    let s_max = (bh - 2) * stride;
    let l_max = (bh - 2) * lstep;
    let gerekli = s_max + s_span;
    let luma_gerekli = l_max + l_span;
    let s_off0 = bx * 2;
    let l_off0 = (bx << SX) * 2;
    assert!(src.len() >= s_off0 + gerekli && dst.len() >= s_off0 + gerekli);
    assert!(luma.len() >= l_off0 + luma_gerekli);
    let dst = &mut dst[s_off0..s_off0 + gerekli];
    let src = &src[s_off0..s_off0 + gerekli];
    let luma = &luma[l_off0..l_off0 + luma_gerekli];
    const W_Y: [[[i16; 2]; 2]; 2] = [[[27, 17], [17, 27]], [[23, 22], [0, 0]]];
    let (ox00, oy00) = lut_ofset(offsets, SX, SY, 0, 0);
    let (ox10, oy10) = lut_ofset(offsets, SX, SY, 1, 0);
    let (ox01, oy01) = lut_ofset(offsets, SX, SY, 0, 1);
    let (ox11, oy11) = lut_ofset(offsets, SX, SY, 1, 1);
    let lut00: &[[i16; 82]; 32] = chunk(grain_lut, oy00);
    let lut10: &[[i16; 82]; 32] = chunk(grain_lut, oy10);

    if SX == 0 {
        assert!(s_span >= 64 && l_span >= 64);
        macro_rules! cift32_uygula {
            ($y:expr, $g0:expr, $g1:expr) => {{
                let y_v: usize = $y;
                let s_baz0 = (y_v * stride).min(s_max);
                let l_baz0 = (y_v * lstep).min(l_max);
                let s_win = &src[s_baz0..s_baz0 + s_span];
                let s0 = load_u8x64(s_win.first_chunk::<64>().unwrap());
                let s1 = load_u8x64(s_win[stride..].first_chunk::<64>().unwrap());
                let l_win = &luma[l_baz0..l_baz0 + l_span];
                let l0 = load_u8x64(l_win.first_chunk::<64>().unwrap());
                let l1 = load_u8x64(l_win[lstep..].first_chunk::<64>().unwrap());
                let out0 = fguv_32_hesapla::<N, CSFL>(
                    s0, l0, scaling, $g0, carpan, vmin, vmax, vbdmax, carpan_uv, v_ofs32,
                );
                let out1 = fguv_32_hesapla::<N, CSFL>(
                    s1, l1, scaling, $g1, carpan, vmin, vmax, vbdmax, carpan_uv, v_ofs32,
                );
                let d_win = &mut dst[s_baz0..s_baz0 + s_span];
                store_u8x64(d_win.first_chunk_mut::<64>().unwrap(), out0);
                store_u8x64(d_win[stride..].first_chunk_mut::<64>().unwrap(), out1);
            }};
        }
        let (g0_ilk, g1_ilk) = if ystart > 0 {
            let (ust0, ust1, cur0, cur1) = if xstart != 0 {
                yatay_harmanla_dort_32::<BPC>(
                    lut_8(grain_lut, oy11, ox11),
                    lut_8(grain_lut, oy11 + 1, ox11),
                    load_i16x8(chunk(&lut10[0], ox10)),
                    load_i16x8(chunk(&lut10[1], ox10)),
                    lut_32(grain_lut, oy01, ox01),
                    lut_32(grain_lut, oy01 + 1, ox01),
                    load_i16x32(chunk(&lut00[0], ox00)),
                    load_i16x32(chunk(&lut00[1], ox00)),
                )
            } else {
                (
                    lut_32(grain_lut, oy01, ox01),
                    lut_32(grain_lut, oy01 + 1, ox01),
                    load_i16x32(chunk(&lut00[0], ox00)),
                    load_i16x32(chunk(&lut00[1], ox00)),
                )
            };
            (
                harmanla_32::<BPC>(ust0, cur0, W_Y[SY][0]),
                harmanla_32::<BPC>(ust1, cur1, W_Y[SY][1]),
            )
        } else if xstart != 0 {
            yatay_harmanla_cift_32::<BPC>(
                load_i16x8(chunk(&lut10[0], ox10)),
                load_i16x8(chunk(&lut10[1], ox10)),
                load_i16x32(chunk(&lut00[0], ox00)),
                load_i16x32(chunk(&lut00[1], ox00)),
            )
        } else {
            (
                load_i16x32(chunk(&lut00[0], ox00)),
                load_i16x32(chunk(&lut00[1], ox00)),
            )
        };
        cift32_uygula!(0, g0_ilk, g1_ilk);
        if xstart != 0 {
            for pair in 1..(bh >> 1) {
                let y = pair * 2;
                let (g0, g1) = yatay_harmanla_cift_32::<BPC>(
                    load_i16x8(chunk(&lut10[y], ox10)),
                    load_i16x8(chunk(&lut10[y + 1], ox10)),
                    load_i16x32(chunk(&lut00[y], ox00)),
                    load_i16x32(chunk(&lut00[y + 1], ox00)),
                );
                cift32_uygula!(y, g0, g1);
            }
        } else {
            for pair in 1..(bh >> 1) {
                let y = pair * 2;
                let g0 = load_i16x32(chunk(&lut00[y], ox00));
                let g1 = load_i16x32(chunk(&lut00[y + 1], ox00));
                cift32_uygula!(y, g0, g1);
            }
        }
    } else {
        assert!(s_span >= 32 && l_span >= 64);
        macro_rules! cift16_uygula {
            ($y:expr, $g:expr) => {{
                let y_v: usize = $y;
                let l_baz0 = (y_v * lstep).min(l_max);
                let l_win = &luma[l_baz0..l_baz0 + l_span];
                let l0 = load_u8x64(l_win.first_chunk::<64>().unwrap());
                let l1 = load_u8x64(l_win[lstep..].first_chunk::<64>().unwrap());
                let cift = _mm512_permutex2var_epi16(l0, pw_even, l1);
                let tek = _mm512_permutex2var_epi16(l0, pw_odd, l1);
                let avg = _mm512_avg_epu16(cift, tek);
                let s_baz0 = (y_v * stride).min(s_max);
                let s_win = &src[s_baz0..s_baz0 + s_span];
                let s0 = load_u8x32(s_win.first_chunk::<32>().unwrap());
                let s1 = load_u8x32(s_win[stride..].first_chunk::<32>().unwrap());
                let s = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(s0), s1);
                let g_v: __m512i = $g;
                let sc = fguv_32_olcek::<N, CSFL>(s, avg, scaling, vbdmax, carpan_uv, v_ofs32);
                let out = gurultu_ekle_32(s, sc, g_v, carpan, vmin, vmax);
                let d_win = &mut dst[s_baz0..s_baz0 + s_span];
                store_u8x32(
                    d_win.first_chunk_mut::<32>().unwrap(),
                    _mm512_castsi512_si256(out),
                );
                store_u8x32(
                    d_win[stride..].first_chunk_mut::<32>().unwrap(),
                    _mm512_extracti64x4_epi64::<1>(out),
                );
            }};
        }
        let cur0_ilk = load_i16x16(chunk(&lut00[0], ox00));
        let cur1_ilk = load_i16x16(chunk(&lut00[1], ox00));
        let g_ilk = if ystart > 0 {
            if SY == 1 {
                let mut ust0 = lut_16(grain_lut, oy01, ox01);
                let mut c0 = cur0_ilk;
                let mut c1 = cur1_ilk;
                if xstart != 0 {
                    let sol_ust0 = lut_8(grain_lut, oy11, ox11);
                    let sol_cur0 = load_i16x8(chunk(&lut10[0], ox10));
                    let sol_cur1 = load_i16x8(chunk(&lut10[1], ox10));
                    ust0 = yatay_harmanla_uv_16::<BPC, SX>(sol_ust0, ust0);
                    c0 = yatay_harmanla_uv_16::<BPC, SX>(sol_cur0, c0);
                    c1 = yatay_harmanla_uv_16::<BPC, SX>(sol_cur1, c1);
                }
                let g0 = harmanla_16::<BPC>(ust0, c0, [23, 22]);
                _mm512_inserti64x4::<1>(_mm512_castsi256_si512(g0), c1)
            } else {
                let (ust0, ust1, c0, c1) = if xstart != 0 {
                    yatay_harmanla_uv_dort_16::<BPC>(
                        lut_8(grain_lut, oy11, ox11),
                        lut_8(grain_lut, oy11 + 1, ox11),
                        load_i16x8(chunk(&lut10[0], ox10)),
                        load_i16x8(chunk(&lut10[1], ox10)),
                        lut_16(grain_lut, oy01, ox01),
                        lut_16(grain_lut, oy01 + 1, ox01),
                        cur0_ilk,
                        cur1_ilk,
                    )
                } else {
                    (
                        lut_16(grain_lut, oy01, ox01),
                        lut_16(grain_lut, oy01 + 1, ox01),
                        cur0_ilk,
                        cur1_ilk,
                    )
                };
                harmanla_32_422::<BPC>(
                    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(ust0), ust1),
                    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(c0), c1),
                )
            }
        } else if xstart != 0 {
            yatay_harmanla_uv_cift_16::<BPC>(
                load_i16x8(chunk(&lut10[0], ox10)),
                load_i16x8(chunk(&lut10[1], ox10)),
                cur0_ilk,
                cur1_ilk,
            )
        } else {
            _mm512_inserti64x4::<1>(_mm512_castsi256_si512(cur0_ilk), cur1_ilk)
        };
        cift16_uygula!(0, g_ilk);
        if xstart != 0 {
            for pair in 1..(bh >> 1) {
                let y = pair * 2;
                cift16_uygula!(
                    y,
                    yatay_harmanla_uv_cift_16::<BPC>(
                        load_i16x8(chunk(&lut10[y], ox10)),
                        load_i16x8(chunk(&lut10[y + 1], ox10)),
                        load_i16x16(chunk(&lut00[y], ox00)),
                        load_i16x16(chunk(&lut00[y + 1], ox00)),
                    )
                );
            }
        } else {
            for pair in 1..(bh >> 1) {
                let y = pair * 2;
                cift16_uygula!(
                    y,
                    _mm512_inserti64x4::<1>(
                        _mm512_castsi256_si512(load_i16x16(chunk(&lut00[y], ox00))),
                        load_i16x16(chunk(&lut00[y + 1], ox00)),
                    )
                );
            }
        }
    }
}

/// Kısmi (`bw < 32 >> SX`) kroma bloğunu tam sınırlar içinde (`0..bw`) işler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
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

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline(never)]
fn fguv_govde<
    const BPC: u32,
    const N: usize,
    const SX: usize,
    const SY: usize,
    const CSFL: bool,
>(
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
    let bh_k = FG_BLOCK_SIZE >> SY;
    let bstep = FG_BLOCK_SIZE >> SX;
    assert!((1..=bh_k).contains(&bh) && (1..=(1 << 16)).contains(&pw));
    assert!(stride >= pw * 2 && stride <= (1 << 20));
    let luma_min_w = (((pw - 1) << SX) + (1 << SX)) * 2;
    assert!(luma_stride >= luma_min_w && luma_stride <= (1 << 20));
    let gerekli = (bh - 1) * stride + pw * 2;
    let luma_gerekli = ((bh - 1) << SY) * luma_stride + luma_min_w;
    assert!(src.len() >= gerekli && dst.len() >= gerekli && luma.len() >= luma_gerekli);
    let overlap = data.overlap_flag;
    let ust_var = overlap && row_num > 0;
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
    let (mut s0, mut s1) = satir_tohumlari(data, row_num, overlap);
    let mut offsets = [[0i32; 2]; 2];
    let lm = data.uv_luma_mult[uv];
    let m = data.uv_mult[uv];
    let ofs = (data.uv_offset[uv] * (1 << min_8)) as i16;
    let carpan = _mm512_set1_epi16(1 << (15 - shift));
    let vmin = _mm512_set1_epi16(min_value as i16);
    let vmax = _mm512_set1_epi16(max_value as i16);
    let vbdmax = _mm512_set1_epi16(max as i16);
    let carpan_uv = _mm512_set1_epi32((lm & 0xFFFF) | (m << 16));
    let v_ofs32 = _mm512_set1_epi32(i32::from(ofs) << 6);

    let ystart = if ust_var { bh.min(2 >> SY) } else { 0 };
    let pw_even = load_i16x32(&PW_EVEN);
    let pw_odd = load_i16x32(&PW_ODD);
    let mut bx = 0usize;
    if bh == bh_k {
        while bx + bstep <= pw {
            if overlap && bx != 0 {
                offsets[1][0] = offsets[0][0];
                if ust_var {
                    offsets[1][1] = offsets[0][1];
                }
            }
            offsets[0][0] = get_random_number(8, &mut s0);
            if ust_var {
                offsets[0][1] = get_random_number(8, &mut s1);
            }
            let xstart = if overlap && bx != 0 { 2 >> SX } else { 0 };
            fguv_blok_tam_512::<BPC, N, SX, SY, CSFL>(
                dst,
                src,
                stride,
                luma,
                luma_stride,
                scaling,
                grain_lut,
                &offsets,
                bx,
                xstart,
                ystart,
                carpan,
                vmin,
                vmax,
                vbdmax,
                carpan_uv,
                v_ofs32,
                pw_even,
                pw_odd,
            );
            bx += bstep;
        }
    } else {
        while bx + bstep <= pw {
            if overlap && bx != 0 {
                offsets[1][0] = offsets[0][0];
                if ust_var {
                    offsets[1][1] = offsets[0][1];
                }
            }
            offsets[0][0] = get_random_number(8, &mut s0);
            if ust_var {
                offsets[0][1] = get_random_number(8, &mut s1);
            }
            let xstart = if overlap && bx != 0 { 2 >> SX } else { 0 };
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
                bstep,
                bh,
                xstart,
                ystart,
                shift,
                min_value,
                max_value,
            );
            bx += bstep;
        }
    }
    if bx < pw {
        let bw = pw - bx;
        if overlap && bx != 0 {
            offsets[1][0] = offsets[0][0];
            if ust_var {
                offsets[1][1] = offsets[0][1];
            }
        }
        offsets[0][0] = get_random_number(8, &mut s0);
        if ust_var {
            offsets[0][1] = get_random_number(8, &mut s1);
        }
        let xstart = if overlap && bx != 0 {
            bw.min(2 >> SX)
        } else {
            0
        };
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
            // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
            // (`fguv_fn`); belirteç AVX512ICL tespiti + maske izninin kanıtıdır. (2) sınır:
            // `fguv16_sozlesmesi` `src`/`dst`'nin `(bh−1)·stride + pw·2` baytı ve lumanın
            // `((bh−1) << sy)·luma_stride + (((pw−1) << sx) + sx + 1)·2` baytı kapsadığını,
            // `first_chunk` tablonun `1024`/`4096` baytını garanti eder; erişimler denetimlidir.
            // (3) hizalama/takma ad: `dst` tek `&mut`, diğerleri `&`.
            unsafe {
                if let Some(tablo) = t10 {
                    if data.chroma_scaling_from_luma {
                        fguv_govde::<10, 1024, $sx, $sy, true>(
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
                    } else {
                        fguv_govde::<10, 1024, $sx, $sy, false>(
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
                } else if let Some(tablo) = t12 {
                    if data.chroma_scaling_from_luma {
                        fguv_govde::<12, 4096, $sx, $sy, true>(
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
                    } else {
                        fguv_govde::<12, 4096, $sx, $sy, false>(
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
        }
    };
}

fguv_giris!(fguv_420_giris, 1, 1, fguv_32x32xn_420);
fguv_giris!(fguv_422_giris, 1, 0, fguv_32x32xn_422);
fguv_giris!(fguv_444_giris, 0, 0, fguv_32x32xn_444);

/// `fgy_32x32xn` 16 bpc AVX512ICL yuvası (yalnız belirteçle; D01571).
#[must_use]
pub fn fgy_fn(_icl: Avx512IclToken) -> FgyFn {
    fgy_giris
}

/// `fguv_32x32xn[düzen]` 16 bpc AVX512ICL yuvası (yalnız belirteçle; D01553/D01559/D01565).
#[must_use]
pub fn fguv_fn(_icl: Avx512IclToken, duzen: usize) -> FguvFn {
    match duzen {
        0 => fguv_420_giris,
        1 => fguv_422_giris,
        _ => fguv_444_giris,
    }
}
