//! Film grain 8 bpc AVX512ICL (`vpermb`) çekirdekleri — F4-FG b (P41 — Adım 2).
//!
//! ## Kapsam (4 kayıt) ve `SLOW_GATHER` Paritesi
//!
//! `dav1d` (`src/x86/filmgrain.h:74-82`, `src/x86/filmgrain_avx512.asm`) 8 bpc
//! için yalnız uygulama yuvalarında (`fgy_32x32xn` ve
//! `fguv_32x32xn[420/422/444]`) AVX512ICL gövdesi tanımlar; `generate_grain_*`
//! yuvaları AVX2'de kalır.
//!
//! 8 bpc AVX512ICL gövdeleri ölçekleme tablosu aramasını `vpgatherdd` yerine
//! 4 × `__m512i` tablo yazmacı üzerinde `vpermi2b` + `vpermt2b`
//! (`_mm512_permutex2var_epi8`) + `vpmovb2m` + `vpblendmb` ile yaptığı için
//! `SLOW_GATHER` (`0x20`, örn. Zen 4) bayrağından **etkilenmez** ve
//! `SLOW_GATHER` açıkken de kurulur (`filmgrain.h:76`:
//! `if (BITDEPTH == 8 || !(flags & DAV1D_X86_CPU_FLAG_SLOW_GATHER))`).
//!
//! ## Sözleşme (kopyasız)
//!
//! Çekirdekler çağrı başına yerel kopya yapmaz; `src`, `dst` ve luma
//! dilimlerine doğrudan erişir ve yalnız sözleşme bölgesini okur/yazar.
//! Kısmi genişlikler (`pw = 145` → `bw = 17` / kromada `bw = 9`) ve kalan
//! satırlar [`super::mem::load_u8_kn`], [`super::mem::store_u8x32_kn`] ve
//! [`super::mem::store_u8_kn`] maskeli yardımcılarıyla ZMM'de işlenir.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use super::fg::{GRAIN_MAX, GRAIN_MIN, W};
use super::fg_ssse3::{OLCEKLEME_BOYU, uv_katsayi_sozlesmesi};
use super::mem::{
    chunk, chunk_mut, load_i16x16, load_i16x32, load_u8_kn, load_u8x16, load_u8x32, load_u8x64,
    store_u8_kn, store_u8x16, store_u8x32, store_u8x32_kn,
};
use crate::bitstream::frame_header::FilmGrainData;
use crate::cpu::Avx512IclToken;
use crate::dsp::fg::{
    FG_BLOCK_SIZE, FgBitDepth, FguvFn, FgyFn, GrainLut, fg_skaler_devri_say, fguv_sozlesmesi,
    fgy_sozlesmesi,
};
use crate::dsp::scalar::fg as scalar;
use core::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm256_add_epi16, _mm256_castsi128_si256, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_inserti128_si256, _mm256_mask_blend_epi16, _mm256_max_epi16,
    _mm256_min_epi16, _mm256_mulhrs_epi16, _mm256_mullo_epi16, _mm256_set1_epi16,
    _mm256_setzero_si256, _mm512_add_epi16, _mm512_avg_epu8, _mm512_castsi256_si512,
    _mm512_castsi512_si128, _mm512_castsi512_si256, _mm512_extracti64x4_epi64, _mm512_inserti64x4,
    _mm512_maddubs_epi16, _mm512_mask_blend_epi8, _mm512_mask_blend_epi16, _mm512_max_epi16,
    _mm512_max_epu8, _mm512_min_epi16, _mm512_min_epu8, _mm512_movepi8_mask, _mm512_mulhrs_epi16,
    _mm512_mullo_epi16, _mm512_packus_epi16, _mm512_permutex2var_epi8, _mm512_set1_epi8,
    _mm512_set1_epi16, _mm512_set1_epi32, _mm512_setr_epi32, _mm512_setzero_si512,
    _mm512_shuffle_i32x4, _mm512_srai_epi16, _mm512_unpackhi_epi8, _mm512_unpacklo_epi8,
};

/// `luma` yatay 2:1 alt-örneklemesi için çift bayt dizinleri (`[0, 2, 4, ..., 126]`).
const PB_EVEN: [u8; 64] = [
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48,
    50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96,
    98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126,
];

/// `luma` yatay 2:1 alt-örneklemesi için tek bayt dizinleri (`[1, 3, 5, ..., 127]`).
const PB_ODD: [u8; 64] = [
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49,
    51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97,
    99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127,
];

/// Döngüsüz blok satırı tohum hesabı (LLVM'in 1..=2 elemanlık döngüyü
/// vektörleştirip fonksiyon girişinde XMM prologu üretmesini önler).
#[inline(always)]
const fn satir_tohumlari(data: &FilmGrainData, row_num: usize, overlap: bool) -> [u32; 2] {
    let r0 = row_num & 0xFFFF;
    let s0 = data.seed ^ ((((r0 * 37 + 178) & 0xFF) << 8) | ((r0 * 173 + 105) & 0xFF)) as u32;
    let s1 = if overlap && row_num != 0 {
        let r1 = (row_num - 1) & 0xFFFF;
        data.seed ^ ((((r1 * 37 + 178) & 0xFF) << 8) | ((r1 * 173 + 105) & 0xFF)) as u32
    } else {
        0
    };
    [s0, s1]
}

/// Bloğun 8-bit rastgele değerinden (`0..=255`) derleme zamanında üst sınırı
/// (`<= 39`) kanıtlanabilir LUT ofseti `(offx, offy)`.
#[inline(always)]
const fn ofset512(randval: i32, sx: usize, sy: usize) -> (usize, usize) {
    let rv = (randval as u8) as usize;
    (
        3 + (2 >> sx) * (3 + (rv >> 4)),
        3 + (2 >> sy) * (3 + (rv & 0xF)),
    )
}

/// Bloğun dört LUT ofseti (`o`, `ust`, `sol`, `kose`); her bileşen `6..=39`
/// aralığında olduğu için `y < 32` ve `x < 32` erişimleri derleme zamanında
/// `73 × 82` LUT sınırları içindedir.
#[derive(Clone, Copy)]
struct Ofsetler512 {
    o: (usize, usize),
    ust: (usize, usize),
    sol: (usize, usize),
    kose: (usize, usize),
}

impl Ofsetler512 {
    #[inline(always)]
    fn yeni(offsets: &[[i32; 2]; 2], sx: usize, sy: usize) -> Self {
        Self {
            o: ofset512(offsets[0][0], sx, sy),
            ust: ofset512(offsets[0][1], sx, sy),
            sol: ofset512(offsets[1][0], sx, sy),
            kose: ofset512(offsets[1][1], sx, sy),
        }
    }
}

/// 256 baytlık ölçekleme tablosunun 4 × `__m512i` yazmaçtaki gösterimi.
#[derive(Clone, Copy)]
struct OlcekTablosu {
    s0: __m512i,
    s1: __m512i,
    s2: __m512i,
    s3: __m512i,
}

impl OlcekTablosu {
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
    #[inline]
    fn yeni(scaling: &[u8; OLCEKLEME_BOYU]) -> Self {
        Self {
            s0: load_u8x64(chunk(scaling, 0)),
            s1: load_u8x64(chunk(scaling, 64)),
            s2: load_u8x64(chunk(scaling, 128)),
            s3: load_u8x64(chunk(scaling, 192)),
        }
    }
}

/// LUT satırındaki ardışık iki `i16` örneğini tek `i32` olarak paketler.
#[inline(always)]
fn paket_i16x2(satir: &[i16; 82], x: usize) -> i32 {
    let a = chunk::<i16, 2>(satir, x);
    (a[0] as u16 as i32) | ((a[1] as u16 as i32) << 16)
}

/// 64 baytlık `idx` vektörünün ölçekleme tablosundaki karşılıklarını
/// `vpermi2b` + `vpermt2b` + `vpmovb2m` + `vpblendmb` ile çağrısız üretir.
macro_rules! olcekle64 {
    ($t:expr, $idx:expr) => {{
        let idx_v: __m512i = $idx;
        let lo = _mm512_permutex2var_epi8(($t).s0, idx_v, ($t).s1);
        let hi = _mm512_permutex2var_epi8(($t).s2, idx_v, ($t).s3);
        let k = _mm512_movepi8_mask(idx_v);
        _mm512_mask_blend_epi8(k, lo, hi)
    }};
}

/// 64 piksellik `fgy` çekirdek hesabı (`s`: 64 × `u8`, `g_lo`/`g_hi`: 2 × 32 × `i16`).
macro_rules! fgy_hesapla64 {
    ($s:expr, $g_lo:expr, $g_hi:expr, $t:expr, $carpan:expr, $vmin:expr, $vmax:expr, $z:expr) => {{
        let s_v: __m512i = $s;
        let sc = olcekle64!($t, s_v);
        let s_lo = _mm512_unpacklo_epi8(s_v, $z);
        let s_hi = _mm512_unpackhi_epi8(s_v, $z);
        let sc_lo = _mm512_unpacklo_epi8(sc, $z);
        let sc_hi = _mm512_unpackhi_epi8(sc, $z);
        let n_lo = _mm512_mulhrs_epi16(_mm512_mullo_epi16(sc_lo, $g_lo), $carpan);
        let n_hi = _mm512_mulhrs_epi16(_mm512_mullo_epi16(sc_hi, $g_hi), $carpan);
        let b = _mm512_packus_epi16(_mm512_add_epi16(s_lo, n_lo), _mm512_add_epi16(s_hi, n_hi));
        _mm512_max_epu8(_mm512_min_epu8(b, $vmax), $vmin)
    }};
}

/// 32 örnekli (`SX = 0`) tek LUT satırını yükler ve gerekiyorsa ilk 2 örneğini
/// sol blokla yatay harmanlar.
macro_rules! satir32_yatay_m {
    ($lut:expr, $oy:expr, $ox:expr, $sol_oy:expr, $sol_ox:expr, $xstart:expr, $w0:expr, $w1:expr, $bin24:expr, $gmin:expr, $gmax:expr) => {{
        let mut g = load_i16x32(chunk(&($lut)[$oy], $ox));
        if $xstart > 0 {
            let eski = _mm512_set1_epi32(paket_i16x2(&($lut)[$sol_oy], $sol_ox + FG_BLOCK_SIZE));
            let x = _mm512_add_epi16(_mm512_mullo_epi16(eski, $w0), _mm512_mullo_epi16(g, $w1));
            let r = _mm512_max_epi16(
                _mm512_min_epi16(_mm512_mulhrs_epi16(x, $bin24), $gmax),
                $gmin,
            );
            g = _mm512_mask_blend_epi16(0x0003, g, r);
        }
        g
    }};
}

/// 32 örnekli (`SX = 0, SY = 0`) dikey örtüşme satırını (`y < ystart`)
/// yatay + dikey harman sırasıyla üretir.
macro_rules! satir32_ust_m {
    ($lut:expr, $of:expr, $y:expr, $xstart:expr, $w0:expr, $w1:expr, $bin24:expr, $gmin:expr, $gmax:expr) => {{
        let cur = satir32_yatay_m!(
            $lut,
            ($of).o.1 + $y,
            ($of).o.0,
            ($of).sol.1 + $y,
            ($of).sol.0,
            $xstart,
            $w0,
            $w1,
            $bin24,
            $gmin,
            $gmax
        );
        let top = satir32_yatay_m!(
            $lut,
            ($of).ust.1 + $y + FG_BLOCK_SIZE,
            ($of).ust.0,
            ($of).kose.1 + $y + FG_BLOCK_SIZE,
            ($of).kose.0,
            $xstart,
            $w0,
            $w1,
            $bin24,
            $gmin,
            $gmax
        );
        let wy = W[0][$y];
        let x = _mm512_add_epi16(
            _mm512_mullo_epi16(top, _mm512_set1_epi16(wy[0] as i16)),
            _mm512_mullo_epi16(cur, _mm512_set1_epi16(wy[1] as i16)),
        );
        _mm512_max_epi16(
            _mm512_min_epi16(_mm512_mulhrs_epi16(x, $bin24), $gmax),
            $gmin,
        )
    }};
}

/// `fgy` kısmi genişlik (`bw < 32`) soğuk bloğu: sıcak döngüdeki ZMM yazmaç
/// tahsisini bozmamak için sabitlerini kendi içinde kurar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline(never)]
fn fgy_kismi_blok(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    scaling: &[u8; OLCEKLEME_BOYU],
    lut: &GrainLut,
    of: &Ofsetler512,
    bx: usize,
    bw: usize,
    bh: usize,
    xstart: usize,
    ystart: usize,
) {
    assert!((1..=FG_BLOCK_SIZE).contains(&bh) && bw < FG_BLOCK_SIZE);
    let sinir = if data.clip_to_restricted_range {
        (16, 235)
    } else {
        (0, 255)
    };
    let shift = data.scaling_shift as u32;
    let carpan = _mm512_set1_epi16(1 << (15 - shift));
    let vmin = _mm512_set1_epi8(sinir.0 as i8);
    let vmax = _mm512_set1_epi8(sinir.1 as u8 as i8);
    let t = OlcekTablosu::yeni(scaling);
    let z = _mm512_setzero_si512();
    let w0 = _mm512_set1_epi32(27 | (17 << 16));
    let w1 = _mm512_set1_epi32(17 | (27 << 16));
    let bin24 = _mm512_set1_epi16(1 << 10);
    let gmin = _mm512_set1_epi16(GRAIN_MIN as i16);
    let gmax = _mm512_set1_epi16(GRAIN_MAX as i16);

    macro_rules! cift_kismi {
        ($y:expr, $ga:expr, $gb:expr) => {{
            let i0 = ($y) * stride + bx;
            let i1 = i0 + stride;
            let g_lo = _mm512_shuffle_i32x4::<0x88>($ga, $gb);
            let g_hi = _mm512_shuffle_i32x4::<0xDD>($ga, $gb);
            let s0 = load_u8_kn(&src[i0..i0 + bw], bw);
            let s1 = _mm512_castsi512_si256(load_u8_kn(&src[i1..i1 + bw], bw));
            let s = _mm512_inserti64x4::<1>(s0, s1);
            let out = fgy_hesapla64!(s, g_lo, g_hi, t, carpan, vmin, vmax, z);
            store_u8x32_kn(&mut dst[i0..i0 + bw], _mm512_castsi512_si256(out), bw);
            store_u8x32_kn(
                &mut dst[i1..i1 + bw],
                _mm512_extracti64x4_epi64::<1>(out),
                bw,
            );
        }};
    }
    macro_rules! tek_kismi {
        ($y:expr, $ga:expr) => {{
            let i0 = ($y) * stride + bx;
            let g_lo = _mm512_shuffle_i32x4::<0x88>($ga, z);
            let g_hi = _mm512_shuffle_i32x4::<0xDD>($ga, z);
            let s = load_u8_kn(&src[i0..i0 + bw], bw);
            let out = fgy_hesapla64!(s, g_lo, g_hi, t, carpan, vmin, vmax, z);
            store_u8x32_kn(&mut dst[i0..i0 + bw], _mm512_castsi512_si256(out), bw);
        }};
    }

    if ystart == 2 {
        let g0 = satir32_ust_m!(lut, of, 0, xstart, w0, w1, bin24, gmin, gmax);
        let g1 = satir32_ust_m!(lut, of, 1, xstart, w0, w1, bin24, gmin, gmax);
        cift_kismi!(0, g0, g1);
    } else if ystart == 1 {
        let g0 = satir32_ust_m!(lut, of, 0, xstart, w0, w1, bin24, gmin, gmax);
        tek_kismi!(0, g0);
    }
    let mut y = ystart;
    while y + 2 <= bh {
        let g0 = satir32_yatay_m!(
            lut,
            of.o.1 + y,
            of.o.0,
            of.sol.1 + y,
            of.sol.0,
            xstart,
            w0,
            w1,
            bin24,
            gmin,
            gmax
        );
        let g1 = satir32_yatay_m!(
            lut,
            of.o.1 + y + 1,
            of.o.0,
            of.sol.1 + y + 1,
            of.sol.0,
            xstart,
            w0,
            w1,
            bin24,
            gmin,
            gmax
        );
        cift_kismi!(y, g0, g1);
        y += 2;
    }
    if y < bh {
        let g0 = satir32_yatay_m!(
            lut,
            of.o.1 + y,
            of.o.0,
            of.sol.1 + y,
            of.sol.0,
            xstart,
            w0,
            w1,
            bin24,
            gmin,
            gmax
        );
        tek_kismi!(y, g0);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline(never)]
fn fgy_govde(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8; OLCEKLEME_BOYU],
    lut: &GrainLut,
    bh: usize,
    row_num: usize,
) {
    assert!((1..=FG_BLOCK_SIZE).contains(&bh) && (1..=(1 << 16)).contains(&pw));
    assert!(stride >= pw && stride <= (1 << 20));
    let gerekli = (bh - 1) * stride + pw;
    assert!(src.len() >= gerekli && dst.len() >= gerekli);
    let overlap = data.overlap_flag;
    let ust_var = overlap && row_num != 0;
    let sinir = if data.clip_to_restricted_range {
        (16, 235)
    } else {
        (0, 255)
    };
    let shift = data.scaling_shift as u32;
    let carpan = _mm512_set1_epi16(1 << (15 - shift));
    let vmin = _mm512_set1_epi8(sinir.0 as i8);
    let vmax = _mm512_set1_epi8(sinir.1 as u8 as i8);
    let z = _mm512_setzero_si512();
    let w0 = _mm512_set1_epi32(27 | (17 << 16));
    let w1 = _mm512_set1_epi32(17 | (27 << 16));
    let vw0 = _mm512_mask_blend_epi16(0xFFFF_0000, _mm512_set1_epi16(27), _mm512_set1_epi16(17));
    let vw1 = _mm512_mask_blend_epi16(0xFFFF_0000, _mm512_set1_epi16(17), _mm512_set1_epi16(27));
    let bin24 = _mm512_set1_epi16(1 << 10);
    let gmin = _mm512_set1_epi16(GRAIN_MIN as i16);
    let gmax = _mm512_set1_epi16(GRAIN_MAX as i16);
    let tablo = OlcekTablosu::yeni(scaling);
    let ystart = if ust_var { bh.min(2) } else { 0 };
    let s_span = stride + FG_BLOCK_SIZE;
    let mut seed = satir_tohumlari(data, row_num, overlap);
    let mut offsets = [[0i32; 2]; 2];
    let mut bx = 0usize;

    macro_rules! cift_uygula {
        ($i0:expr, $g_lo:expr, $g_hi:expr) => {{
            let i0_v: usize = $i0;
            let s_win = &src[i0_v..i0_v + s_span];
            let s0 = load_u8x32(s_win.first_chunk::<32>().unwrap());
            let s1 = load_u8x32(s_win.last_chunk::<32>().unwrap());
            let s = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(s0), s1);
            let out = fgy_hesapla64!(s, $g_lo, $g_hi, tablo, carpan, vmin, vmax, z);
            let d_win = &mut dst[i0_v..i0_v + s_span];
            store_u8x32(
                d_win.first_chunk_mut::<32>().unwrap(),
                _mm512_castsi512_si256(out),
            );
            store_u8x32(
                d_win.last_chunk_mut::<32>().unwrap(),
                _mm512_extracti64x4_epi64::<1>(out),
            );
        }};
    }
    macro_rules! tek_tam {
        ($i0:expr, $ga:expr) => {{
            let i0_v: usize = $i0;
            let g_lo = _mm512_shuffle_i32x4::<0x88>($ga, z);
            let g_hi = _mm512_shuffle_i32x4::<0xDD>($ga, z);
            let s = _mm512_castsi256_si512(load_u8x32(chunk(src, i0_v)));
            let out = fgy_hesapla64!(s, g_lo, g_hi, tablo, carpan, vmin, vmax, z);
            store_u8x32(chunk_mut(dst, i0_v), _mm512_castsi512_si256(out));
        }};
    }

    // Tam blok (`bw == 32`) döngüsü: sıfır iç işlev çağrısı (tüm ZMM sabitleri yazmaçta kalır).
    while bx + FG_BLOCK_SIZE <= pw {
        if overlap && bx != 0 {
            offsets[1] = offsets[0];
        }
        offsets[0][0] = scalar::get_random_number(8, &mut seed[0]);
        if ust_var {
            offsets[0][1] = scalar::get_random_number(8, &mut seed[1]);
        }
        let xstart = if overlap && bx != 0 { 2 } else { 0 };
        let of = Ofsetler512::yeni(&offsets, 0, 0);
        if ystart == 2 {
            let g_a = load_i16x32(chunk(&lut[of.o.1], of.o.0));
            let g_b = load_i16x32(chunk(&lut[of.o.1 + 1], of.o.0));
            let mut g_lo = _mm512_shuffle_i32x4::<0x88>(g_a, g_b);
            let g_hi = _mm512_shuffle_i32x4::<0xDD>(g_a, g_b);
            let uy = of.ust.1 + FG_BLOCK_SIZE;
            let t_a = load_i16x32(chunk(&lut[uy], of.ust.0));
            let t_b = load_i16x32(chunk(&lut[uy + 1], of.ust.0));
            let mut t_lo = _mm512_shuffle_i32x4::<0x88>(t_a, t_b);
            let t_hi = _mm512_shuffle_i32x4::<0xDD>(t_a, t_b);
            if xstart > 0 {
                let s_sol0 = paket_i16x2(&lut[of.sol.1], of.sol.0 + FG_BLOCK_SIZE);
                let s_sol1 = paket_i16x2(&lut[of.sol.1 + 1], of.sol.0 + FG_BLOCK_SIZE);
                let eski =
                    _mm512_setr_epi32(s_sol0, 0, 0, 0, 0, 0, 0, 0, s_sol1, 0, 0, 0, 0, 0, 0, 0);
                let x =
                    _mm512_add_epi16(_mm512_mullo_epi16(eski, w0), _mm512_mullo_epi16(g_lo, w1));
                let r =
                    _mm512_max_epi16(_mm512_min_epi16(_mm512_mulhrs_epi16(x, bin24), gmax), gmin);
                g_lo = _mm512_mask_blend_epi16(0x0003_0003, g_lo, r);
                let ky = of.kose.1 + FG_BLOCK_SIZE;
                let k_sol0 = paket_i16x2(&lut[ky], of.kose.0 + FG_BLOCK_SIZE);
                let k_sol1 = paket_i16x2(&lut[ky + 1], of.kose.0 + FG_BLOCK_SIZE);
                let eski_t =
                    _mm512_setr_epi32(k_sol0, 0, 0, 0, 0, 0, 0, 0, k_sol1, 0, 0, 0, 0, 0, 0, 0);
                let xt =
                    _mm512_add_epi16(_mm512_mullo_epi16(eski_t, w0), _mm512_mullo_epi16(t_lo, w1));
                let rt =
                    _mm512_max_epi16(_mm512_min_epi16(_mm512_mulhrs_epi16(xt, bin24), gmax), gmin);
                t_lo = _mm512_mask_blend_epi16(0x0003_0003, t_lo, rt);
            }
            let xv_lo =
                _mm512_add_epi16(_mm512_mullo_epi16(t_lo, vw0), _mm512_mullo_epi16(g_lo, vw1));
            let xv_hi =
                _mm512_add_epi16(_mm512_mullo_epi16(t_hi, vw0), _mm512_mullo_epi16(g_hi, vw1));
            let gv_lo = _mm512_max_epi16(
                _mm512_min_epi16(_mm512_mulhrs_epi16(xv_lo, bin24), gmax),
                gmin,
            );
            let gv_hi = _mm512_max_epi16(
                _mm512_min_epi16(_mm512_mulhrs_epi16(xv_hi, bin24), gmax),
                gmin,
            );
            cift_uygula!(bx, gv_lo, gv_hi);
        } else if ystart == 1 {
            let g0 = satir32_ust_m!(lut, of, 0, xstart, w0, w1, bin24, gmin, gmax);
            tek_tam!(bx, g0);
        }
        let mut y = ystart;
        let mut i0 = y * stride + bx;
        if xstart > 0 {
            while y + 2 <= bh {
                let g_a = load_i16x32(chunk(&lut[of.o.1 + y], of.o.0));
                let g_b = load_i16x32(chunk(&lut[of.o.1 + y + 1], of.o.0));
                let mut g_lo = _mm512_shuffle_i32x4::<0x88>(g_a, g_b);
                let g_hi = _mm512_shuffle_i32x4::<0xDD>(g_a, g_b);
                let s_sol0 = paket_i16x2(&lut[of.sol.1 + y], of.sol.0 + FG_BLOCK_SIZE);
                let s_sol1 = paket_i16x2(&lut[of.sol.1 + y + 1], of.sol.0 + FG_BLOCK_SIZE);
                let eski =
                    _mm512_setr_epi32(s_sol0, 0, 0, 0, 0, 0, 0, 0, s_sol1, 0, 0, 0, 0, 0, 0, 0);
                let x =
                    _mm512_add_epi16(_mm512_mullo_epi16(eski, w0), _mm512_mullo_epi16(g_lo, w1));
                let r =
                    _mm512_max_epi16(_mm512_min_epi16(_mm512_mulhrs_epi16(x, bin24), gmax), gmin);
                g_lo = _mm512_mask_blend_epi16(0x0003_0003, g_lo, r);
                cift_uygula!(i0, g_lo, g_hi);
                i0 += 2 * stride;
                y += 2;
            }
        } else {
            while y + 2 <= bh {
                let g_a = load_i16x32(chunk(&lut[of.o.1 + y], of.o.0));
                let g_b = load_i16x32(chunk(&lut[of.o.1 + y + 1], of.o.0));
                let g_lo = _mm512_shuffle_i32x4::<0x88>(g_a, g_b);
                let g_hi = _mm512_shuffle_i32x4::<0xDD>(g_a, g_b);
                cift_uygula!(i0, g_lo, g_hi);
                i0 += 2 * stride;
                y += 2;
            }
        }
        if y < bh {
            let g0 = satir32_yatay_m!(
                lut,
                of.o.1 + y,
                of.o.0,
                of.sol.1 + y,
                of.sol.0,
                xstart,
                w0,
                w1,
                bin24,
                gmin,
                gmax
            );
            tek_tam!(i0, g0);
        }
        bx += FG_BLOCK_SIZE;
    }

    // Varsa en sondaki tek kısmi genişlikli blok (`bw < 32`).
    if bx < pw {
        let bw = pw - bx;
        if overlap && bx != 0 {
            offsets[1] = offsets[0];
        }
        offsets[0][0] = scalar::get_random_number(8, &mut seed[0]);
        if ust_var {
            offsets[0][1] = scalar::get_random_number(8, &mut seed[1]);
        }
        let xstart = if overlap && bx != 0 { bw.min(2) } else { 0 };
        let of = Ofsetler512::yeni(&offsets, 0, 0);
        fgy_kismi_blok(
            dst, src, stride, data, scaling, lut, &of, bx, bw, bh, xstart, ystart,
        );
    }
}

/// 64 kroma pikseli (`s`: 64 × `u8`) ve 64 luma ortalaması (`avg_u8`: 64 × `u8`)
/// üzerinden `CSFL` / `!CSFL` ölçekleme araması, gürültü ve kırpma.
macro_rules! fguv_hesapla64 {
    ($csfl:expr, $s:expr, $avg_u8:expr, $g_lo:expr, $g_hi:expr, $t:expr, $carpan:expr, $carpan_uv:expr, $ofs:expr, $vmin:expr, $vmax:expr, $z:expr) => {{
        let s_v: __m512i = $s;
        let avg_v: __m512i = $avg_u8;
        let idx = if $csfl {
            avg_v
        } else {
            let p_lo = _mm512_unpacklo_epi8(avg_v, s_v);
            let p_hi = _mm512_unpackhi_epi8(avg_v, s_v);
            let v_lo = _mm512_add_epi16(
                _mm512_srai_epi16::<6>(_mm512_maddubs_epi16(p_lo, $carpan_uv)),
                $ofs,
            );
            let v_hi = _mm512_add_epi16(
                _mm512_srai_epi16::<6>(_mm512_maddubs_epi16(p_hi, $carpan_uv)),
                $ofs,
            );
            _mm512_packus_epi16(v_lo, v_hi)
        };
        let sc = olcekle64!($t, idx);
        let s_lo = _mm512_unpacklo_epi8(s_v, $z);
        let s_hi = _mm512_unpackhi_epi8(s_v, $z);
        let sc_lo = _mm512_unpacklo_epi8(sc, $z);
        let sc_hi = _mm512_unpackhi_epi8(sc, $z);
        let n_lo = _mm512_mulhrs_epi16(_mm512_mullo_epi16(sc_lo, $g_lo), $carpan);
        let n_hi = _mm512_mulhrs_epi16(_mm512_mullo_epi16(sc_hi, $g_hi), $carpan);
        let b = _mm512_packus_epi16(_mm512_add_epi16(s_lo, n_lo), _mm512_add_epi16(s_hi, n_hi));
        _mm512_max_epu8(_mm512_min_epu8(b, $vmax), $vmin)
    }};
}

/// 16 örnekli (`SX = 1`) tek LUT satırını (`y`) yatay ve gerekiyorsa (`y < ystart`)
/// dikey harman sırasıyla 256-bit yazmaca yükler.
macro_rules! satir16_yukle_m {
    ($sy:expr, $lut:expr, $of:expr, $y:expr, $xstart:expr, $ystart:expr) => {{
        let bstep = FG_BLOCK_SIZE >> 1;
        let bh_k = FG_BLOCK_SIZE >> ($sy);
        let yatay = |oy: usize, ox: usize, sol_oy: usize, sol_ox: usize| -> __m256i {
            let mut g = load_i16x16(chunk(&($lut)[oy], ox));
            if $xstart > 0 {
                let eski = _mm256_set1_epi16(($lut)[sol_oy][sol_ox + bstep]);
                let x = _mm256_add_epi16(
                    _mm256_mullo_epi16(eski, _mm256_set1_epi16(23)),
                    _mm256_mullo_epi16(g, _mm256_set1_epi16(22)),
                );
                let r = _mm256_mulhrs_epi16(x, _mm256_set1_epi16(1 << 10));
                let r = _mm256_max_epi16(
                    _mm256_min_epi16(r, _mm256_set1_epi16(GRAIN_MAX as i16)),
                    _mm256_set1_epi16(GRAIN_MIN as i16),
                );
                g = _mm256_mask_blend_epi16(0x0001, g, r);
            }
            g
        };
        let y_v: usize = $y;
        let mut g = yatay(($of).o.1 + y_v, ($of).o.0, ($of).sol.1 + y_v, ($of).sol.0);
        if y_v < $ystart {
            let top = yatay(
                ($of).ust.1 + y_v + bh_k,
                ($of).ust.0,
                ($of).kose.1 + y_v + bh_k,
                ($of).kose.0,
            );
            let w = W[$sy][y_v & 1];
            let x = _mm256_add_epi16(
                _mm256_mullo_epi16(top, _mm256_set1_epi16(w[0] as i16)),
                _mm256_mullo_epi16(g, _mm256_set1_epi16(w[1] as i16)),
            );
            let r = _mm256_mulhrs_epi16(x, _mm256_set1_epi16(1 << 10));
            g = _mm256_max_epi16(
                _mm256_min_epi16(r, _mm256_set1_epi16(GRAIN_MAX as i16)),
                _mm256_set1_epi16(GRAIN_MIN as i16),
            );
        }
        g
    }};
}

/// `fguv 444` (`SX = 0, SY = 0`) kısmi genişlik (`bw < 32`) soğuk bloğu.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline(never)]
fn fguv444_kismi_blok<const CSFL: bool>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    luma: &[u8],
    luma_stride: usize,
    data: &FilmGrainData,
    scaling: &[u8; OLCEKLEME_BOYU],
    lut: &GrainLut,
    of: &Ofsetler512,
    bx: usize,
    bw: usize,
    bh: usize,
    xstart: usize,
    ystart: usize,
    uv: usize,
    is_id: bool,
) {
    assert!((1..=FG_BLOCK_SIZE).contains(&bh) && bw < FG_BLOCK_SIZE);
    let sinir = if data.clip_to_restricted_range {
        (16, if is_id { 235 } else { 240 })
    } else {
        (0, 255)
    };
    let shift = data.scaling_shift as u32;
    let (lm, m, off) = (data.uv_luma_mult[uv], data.uv_mult[uv], data.uv_offset[uv]);
    let carpan = _mm512_set1_epi16(1 << (15 - shift));
    let carpan_uv = _mm512_set1_epi16(((lm & 0xFF) as i16) | ((m as i16) << 8));
    let ofs = _mm512_set1_epi16(off as i16);
    let vmin = _mm512_set1_epi8(sinir.0 as i8);
    let vmax = _mm512_set1_epi8(sinir.1 as u8 as i8);
    let t = OlcekTablosu::yeni(scaling);
    let z = _mm512_setzero_si512();
    let w0 = _mm512_set1_epi32(27 | (17 << 16));
    let w1 = _mm512_set1_epi32(17 | (27 << 16));
    let bin24 = _mm512_set1_epi16(1 << 10);
    let gmin = _mm512_set1_epi16(GRAIN_MIN as i16);
    let gmax = _mm512_set1_epi16(GRAIN_MAX as i16);

    macro_rules! cift_kismi {
        ($y:expr, $ga:expr, $gb:expr) => {{
            let i0 = ($y) * stride + bx;
            let i1 = i0 + stride;
            let lx0 = ($y) * luma_stride + bx;
            let lx1 = lx0 + luma_stride;
            let g_lo = _mm512_shuffle_i32x4::<0x88>($ga, $gb);
            let g_hi = _mm512_shuffle_i32x4::<0xDD>($ga, $gb);
            let s0 = load_u8_kn(&src[i0..i0 + bw], bw);
            let s1 = _mm512_castsi512_si256(load_u8_kn(&src[i1..i1 + bw], bw));
            let s = _mm512_inserti64x4::<1>(s0, s1);
            let l0 = load_u8_kn(&luma[lx0..lx0 + bw], bw);
            let l1 = _mm512_castsi512_si256(load_u8_kn(&luma[lx1..lx1 + bw], bw));
            let l = _mm512_inserti64x4::<1>(l0, l1);
            let out = fguv_hesapla64!(
                CSFL, s, l, g_lo, g_hi, t, carpan, carpan_uv, ofs, vmin, vmax, z
            );
            store_u8x32_kn(&mut dst[i0..i0 + bw], _mm512_castsi512_si256(out), bw);
            store_u8x32_kn(
                &mut dst[i1..i1 + bw],
                _mm512_extracti64x4_epi64::<1>(out),
                bw,
            );
        }};
    }
    macro_rules! tek_kismi {
        ($y:expr, $ga:expr) => {{
            let i0 = ($y) * stride + bx;
            let lx0 = ($y) * luma_stride + bx;
            let g_lo = _mm512_shuffle_i32x4::<0x88>($ga, z);
            let g_hi = _mm512_shuffle_i32x4::<0xDD>($ga, z);
            let s = load_u8_kn(&src[i0..i0 + bw], bw);
            let l = load_u8_kn(&luma[lx0..lx0 + bw], bw);
            let out = fguv_hesapla64!(
                CSFL, s, l, g_lo, g_hi, t, carpan, carpan_uv, ofs, vmin, vmax, z
            );
            store_u8x32_kn(&mut dst[i0..i0 + bw], _mm512_castsi512_si256(out), bw);
        }};
    }

    if ystart == 2 {
        let g0 = satir32_ust_m!(lut, of, 0, xstart, w0, w1, bin24, gmin, gmax);
        let g1 = satir32_ust_m!(lut, of, 1, xstart, w0, w1, bin24, gmin, gmax);
        cift_kismi!(0, g0, g1);
    } else if ystart == 1 {
        let g0 = satir32_ust_m!(lut, of, 0, xstart, w0, w1, bin24, gmin, gmax);
        tek_kismi!(0, g0);
    }
    let mut y = ystart;
    while y + 2 <= bh {
        let g0 = satir32_yatay_m!(
            lut,
            of.o.1 + y,
            of.o.0,
            of.sol.1 + y,
            of.sol.0,
            xstart,
            w0,
            w1,
            bin24,
            gmin,
            gmax
        );
        let g1 = satir32_yatay_m!(
            lut,
            of.o.1 + y + 1,
            of.o.0,
            of.sol.1 + y + 1,
            of.sol.0,
            xstart,
            w0,
            w1,
            bin24,
            gmin,
            gmax
        );
        cift_kismi!(y, g0, g1);
        y += 2;
    }
    if y < bh {
        let g0 = satir32_yatay_m!(
            lut,
            of.o.1 + y,
            of.o.0,
            of.sol.1 + y,
            of.sol.0,
            xstart,
            w0,
            w1,
            bin24,
            gmin,
            gmax
        );
        tek_kismi!(y, g0);
    }
}

/// `fguv 420 / 422` (`SX = 1`) kısmi genişlik (`bw < 16`) veya `bh % 4 != 0` kalan satırlar
/// soğuk yolu.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline(never)]
fn fguv42x_kismi_adim<const SY: usize, const CSFL: bool>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    luma: &[u8],
    luma_stride: usize,
    data: &FilmGrainData,
    scaling: &[u8; OLCEKLEME_BOYU],
    lut: &GrainLut,
    of: &Ofsetler512,
    bx: usize,
    bw: usize,
    y_bas: usize,
    bh: usize,
    xstart: usize,
    ystart: usize,
    uv: usize,
    is_id: bool,
) {
    assert!(bh >= 1 && bh <= (FG_BLOCK_SIZE >> SY) && (1..=(FG_BLOCK_SIZE >> 1)).contains(&bw));
    let sinir = if data.clip_to_restricted_range {
        (16, if is_id { 235 } else { 240 })
    } else {
        (0, 255)
    };
    let shift = data.scaling_shift as u32;
    let (lm, m, off) = (data.uv_luma_mult[uv], data.uv_mult[uv], data.uv_offset[uv]);
    let carpan = _mm512_set1_epi16(1 << (15 - shift));
    let carpan_uv = _mm512_set1_epi16(((lm & 0xFF) as i16) | ((m as i16) << 8));
    let ofs = _mm512_set1_epi16(off as i16);
    let vmin = _mm512_set1_epi8(sinir.0 as i8);
    let vmax = _mm512_set1_epi8(sinir.1 as u8 as i8);
    let pb_even = load_u8x64(&PB_EVEN);
    let pb_odd = load_u8x64(&PB_ODD);
    let t = OlcekTablosu::yeni(scaling);
    let z = _mm512_setzero_si512();
    let lw = bw << 1;
    let mut y = y_bas;
    while y < bh {
        let n_satir = (bh - y).min(4);
        let mut s_rows = [_mm512_castsi512_si128(z); 4];
        let mut l_rows = [_mm256_setzero_si256(); 4];
        let mut g_rows = [_mm256_setzero_si256(); 4];
        for r in 0..n_satir {
            let yr = y + r;
            let i = yr * stride + bx;
            let lx = (yr << SY) * luma_stride + (bx << 1);
            s_rows[r] = if bw == 16 {
                load_u8x16(chunk(src, i))
            } else {
                _mm512_castsi512_si128(load_u8_kn(&src[i..i + bw], bw))
            };
            l_rows[r] = if lw == 32 {
                load_u8x32(chunk(luma, lx))
            } else {
                _mm512_castsi512_si256(load_u8_kn(&luma[lx..lx + lw], lw))
            };
            g_rows[r] = satir16_yukle_m!(SY, lut, of, yr, xstart, ystart);
        }
        let s01 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(s_rows[0]), s_rows[1]);
        let s23 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(s_rows[2]), s_rows[3]);
        let s = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(s01), s23);
        let l01 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(l_rows[0]), l_rows[1]);
        let l23 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(l_rows[2]), l_rows[3]);
        let even = _mm512_permutex2var_epi8(l01, pb_even, l23);
        let odd = _mm512_permutex2var_epi8(l01, pb_odd, l23);
        let avg_u8 = _mm512_avg_epu8(even, odd);

        let g_a = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(g_rows[0]), g_rows[1]);
        let g_b = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(g_rows[2]), g_rows[3]);
        let g_lo = _mm512_shuffle_i32x4::<0x88>(g_a, g_b);
        let g_hi = _mm512_shuffle_i32x4::<0xDD>(g_a, g_b);
        let out = fguv_hesapla64!(
            CSFL, s, avg_u8, g_lo, g_hi, t, carpan, carpan_uv, ofs, vmin, vmax, z
        );
        let out_lo = _mm512_castsi512_si256(out);
        let out_hi = _mm512_extracti64x4_epi64::<1>(out);
        let cikti: [__m128i; 4] = [
            _mm256_castsi256_si128(out_lo),
            _mm256_extracti128_si256::<1>(out_lo),
            _mm256_castsi256_si128(out_hi),
            _mm256_extracti128_si256::<1>(out_hi),
        ];
        for r in 0..n_satir {
            let i = (y + r) * stride + bx;
            if bw == 16 {
                store_u8x16(chunk_mut(dst, i), cikti[r]);
            } else {
                store_u8_kn(&mut dst[i..i + bw], cikti[r], bw);
            }
        }
        y += n_satir;
    }
}

/// `fguv 444` (`SX = 0, SY = 0`) sıcak gövdesi.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline(never)]
fn fguv444_govde_ic<const CSFL: bool>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8; OLCEKLEME_BOYU],
    lut: &GrainLut,
    bh: usize,
    row_num: usize,
    luma: &[u8],
    luma_stride: usize,
    uv: usize,
    is_id: bool,
) {
    assert!((1..=FG_BLOCK_SIZE).contains(&bh) && (1..=(1 << 16)).contains(&pw));
    assert!(stride >= pw && stride <= (1 << 20));
    assert!(luma_stride >= pw && luma_stride <= (1 << 20));
    let gerekli = (bh - 1) * stride + pw;
    let luma_gerekli = (bh - 1) * luma_stride + pw;
    assert!(src.len() >= gerekli && dst.len() >= gerekli && luma.len() >= luma_gerekli);
    let overlap = data.overlap_flag;
    let ust_var = overlap && row_num != 0;
    let sinir = if data.clip_to_restricted_range {
        (16, if is_id { 235 } else { 240 })
    } else {
        (0, 255)
    };
    let shift = data.scaling_shift as u32;
    let (lm, m, off) = (data.uv_luma_mult[uv], data.uv_mult[uv], data.uv_offset[uv]);
    let carpan = _mm512_set1_epi16(1 << (15 - shift));
    let carpan_uv = _mm512_set1_epi16(((lm & 0xFF) as i16) | ((m as i16) << 8));
    let ofs = _mm512_set1_epi16(off as i16);
    let vmin = _mm512_set1_epi8(sinir.0 as i8);
    let vmax = _mm512_set1_epi8(sinir.1 as u8 as i8);
    let z = _mm512_setzero_si512();
    let w0 = _mm512_set1_epi32(27 | (17 << 16));
    let w1 = _mm512_set1_epi32(17 | (27 << 16));
    let vw0 = _mm512_mask_blend_epi16(0xFFFF_0000, _mm512_set1_epi16(27), _mm512_set1_epi16(17));
    let vw1 = _mm512_mask_blend_epi16(0xFFFF_0000, _mm512_set1_epi16(17), _mm512_set1_epi16(27));
    let bin24 = _mm512_set1_epi16(1 << 10);
    let gmin = _mm512_set1_epi16(GRAIN_MIN as i16);
    let gmax = _mm512_set1_epi16(GRAIN_MAX as i16);
    let tablo = OlcekTablosu::yeni(scaling);
    let ystart = if ust_var { bh.min(2) } else { 0 };
    let s_span = stride + FG_BLOCK_SIZE;
    let l_span = luma_stride + FG_BLOCK_SIZE;
    let mut seed = satir_tohumlari(data, row_num, overlap);
    let mut offsets = [[0i32; 2]; 2];
    let mut bx = 0usize;

    macro_rules! cift444_uygula {
        ($i0:expr, $lx0:expr, $g_lo:expr, $g_hi:expr) => {{
            let i0_v: usize = $i0;
            let lx0_v: usize = $lx0;
            let s_win = &src[i0_v..i0_v + s_span];
            let s0 = load_u8x32(s_win.first_chunk::<32>().unwrap());
            let s1 = load_u8x32(s_win.last_chunk::<32>().unwrap());
            let s = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(s0), s1);
            let l_win = &luma[lx0_v..lx0_v + l_span];
            let l0 = load_u8x32(l_win.first_chunk::<32>().unwrap());
            let l1 = load_u8x32(l_win.last_chunk::<32>().unwrap());
            let l = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(l0), l1);
            let out = fguv_hesapla64!(
                CSFL, s, l, $g_lo, $g_hi, tablo, carpan, carpan_uv, ofs, vmin, vmax, z
            );
            let d_win = &mut dst[i0_v..i0_v + s_span];
            store_u8x32(
                d_win.first_chunk_mut::<32>().unwrap(),
                _mm512_castsi512_si256(out),
            );
            store_u8x32(
                d_win.last_chunk_mut::<32>().unwrap(),
                _mm512_extracti64x4_epi64::<1>(out),
            );
        }};
    }
    macro_rules! tek444 {
        ($i0:expr, $lx0:expr, $ga:expr) => {{
            let i0_v: usize = $i0;
            let lx0_v: usize = $lx0;
            let g_lo = _mm512_shuffle_i32x4::<0x88>($ga, z);
            let g_hi = _mm512_shuffle_i32x4::<0xDD>($ga, z);
            let s = _mm512_castsi256_si512(load_u8x32(chunk(src, i0_v)));
            let l = _mm512_castsi256_si512(load_u8x32(chunk(luma, lx0_v)));
            let out = fguv_hesapla64!(
                CSFL, s, l, g_lo, g_hi, tablo, carpan, carpan_uv, ofs, vmin, vmax, z
            );
            store_u8x32(chunk_mut(dst, i0_v), _mm512_castsi512_si256(out));
        }};
    }

    while bx + FG_BLOCK_SIZE <= pw {
        if overlap && bx != 0 {
            offsets[1] = offsets[0];
        }
        offsets[0][0] = scalar::get_random_number(8, &mut seed[0]);
        if ust_var {
            offsets[0][1] = scalar::get_random_number(8, &mut seed[1]);
        }
        let xstart = if overlap && bx != 0 { 2 } else { 0 };
        let of = Ofsetler512::yeni(&offsets, 0, 0);
        if ystart == 2 {
            let g_a = load_i16x32(chunk(&lut[of.o.1], of.o.0));
            let g_b = load_i16x32(chunk(&lut[of.o.1 + 1], of.o.0));
            let mut g_lo = _mm512_shuffle_i32x4::<0x88>(g_a, g_b);
            let g_hi = _mm512_shuffle_i32x4::<0xDD>(g_a, g_b);
            let uy = of.ust.1 + FG_BLOCK_SIZE;
            let t_a = load_i16x32(chunk(&lut[uy], of.ust.0));
            let t_b = load_i16x32(chunk(&lut[uy + 1], of.ust.0));
            let mut t_lo = _mm512_shuffle_i32x4::<0x88>(t_a, t_b);
            let t_hi = _mm512_shuffle_i32x4::<0xDD>(t_a, t_b);
            if xstart > 0 {
                let s_sol0 = paket_i16x2(&lut[of.sol.1], of.sol.0 + FG_BLOCK_SIZE);
                let s_sol1 = paket_i16x2(&lut[of.sol.1 + 1], of.sol.0 + FG_BLOCK_SIZE);
                let eski =
                    _mm512_setr_epi32(s_sol0, 0, 0, 0, 0, 0, 0, 0, s_sol1, 0, 0, 0, 0, 0, 0, 0);
                let x =
                    _mm512_add_epi16(_mm512_mullo_epi16(eski, w0), _mm512_mullo_epi16(g_lo, w1));
                let r =
                    _mm512_max_epi16(_mm512_min_epi16(_mm512_mulhrs_epi16(x, bin24), gmax), gmin);
                g_lo = _mm512_mask_blend_epi16(0x0003_0003, g_lo, r);
                let ky = of.kose.1 + FG_BLOCK_SIZE;
                let k_sol0 = paket_i16x2(&lut[ky], of.kose.0 + FG_BLOCK_SIZE);
                let k_sol1 = paket_i16x2(&lut[ky + 1], of.kose.0 + FG_BLOCK_SIZE);
                let eski_t =
                    _mm512_setr_epi32(k_sol0, 0, 0, 0, 0, 0, 0, 0, k_sol1, 0, 0, 0, 0, 0, 0, 0);
                let xt =
                    _mm512_add_epi16(_mm512_mullo_epi16(eski_t, w0), _mm512_mullo_epi16(t_lo, w1));
                let rt =
                    _mm512_max_epi16(_mm512_min_epi16(_mm512_mulhrs_epi16(xt, bin24), gmax), gmin);
                t_lo = _mm512_mask_blend_epi16(0x0003_0003, t_lo, rt);
            }
            let xv_lo =
                _mm512_add_epi16(_mm512_mullo_epi16(t_lo, vw0), _mm512_mullo_epi16(g_lo, vw1));
            let xv_hi =
                _mm512_add_epi16(_mm512_mullo_epi16(t_hi, vw0), _mm512_mullo_epi16(g_hi, vw1));
            let gv_lo = _mm512_max_epi16(
                _mm512_min_epi16(_mm512_mulhrs_epi16(xv_lo, bin24), gmax),
                gmin,
            );
            let gv_hi = _mm512_max_epi16(
                _mm512_min_epi16(_mm512_mulhrs_epi16(xv_hi, bin24), gmax),
                gmin,
            );
            cift444_uygula!(bx, bx, gv_lo, gv_hi);
        } else if ystart == 1 {
            let g0 = satir32_ust_m!(lut, of, 0, xstart, w0, w1, bin24, gmin, gmax);
            tek444!(bx, bx, g0);
        }
        let mut y = ystart;
        let mut i0 = y * stride + bx;
        let mut lx0 = y * luma_stride + bx;
        if xstart > 0 {
            while y + 2 <= bh {
                let g_a = load_i16x32(chunk(&lut[of.o.1 + y], of.o.0));
                let g_b = load_i16x32(chunk(&lut[of.o.1 + y + 1], of.o.0));
                let mut g_lo = _mm512_shuffle_i32x4::<0x88>(g_a, g_b);
                let g_hi = _mm512_shuffle_i32x4::<0xDD>(g_a, g_b);
                let s_sol0 = paket_i16x2(&lut[of.sol.1 + y], of.sol.0 + FG_BLOCK_SIZE);
                let s_sol1 = paket_i16x2(&lut[of.sol.1 + y + 1], of.sol.0 + FG_BLOCK_SIZE);
                let eski =
                    _mm512_setr_epi32(s_sol0, 0, 0, 0, 0, 0, 0, 0, s_sol1, 0, 0, 0, 0, 0, 0, 0);
                let x =
                    _mm512_add_epi16(_mm512_mullo_epi16(eski, w0), _mm512_mullo_epi16(g_lo, w1));
                let r =
                    _mm512_max_epi16(_mm512_min_epi16(_mm512_mulhrs_epi16(x, bin24), gmax), gmin);
                g_lo = _mm512_mask_blend_epi16(0x0003_0003, g_lo, r);
                cift444_uygula!(i0, lx0, g_lo, g_hi);
                i0 += 2 * stride;
                lx0 += 2 * luma_stride;
                y += 2;
            }
        } else {
            while y + 2 <= bh {
                let g_a = load_i16x32(chunk(&lut[of.o.1 + y], of.o.0));
                let g_b = load_i16x32(chunk(&lut[of.o.1 + y + 1], of.o.0));
                let g_lo = _mm512_shuffle_i32x4::<0x88>(g_a, g_b);
                let g_hi = _mm512_shuffle_i32x4::<0xDD>(g_a, g_b);
                cift444_uygula!(i0, lx0, g_lo, g_hi);
                i0 += 2 * stride;
                lx0 += 2 * luma_stride;
                y += 2;
            }
        }
        if y < bh {
            let g0 = satir32_yatay_m!(
                lut,
                of.o.1 + y,
                of.o.0,
                of.sol.1 + y,
                of.sol.0,
                xstart,
                w0,
                w1,
                bin24,
                gmin,
                gmax
            );
            tek444!(i0, lx0, g0);
        }
        bx += FG_BLOCK_SIZE;
    }
    if bx < pw {
        let bw = pw - bx;
        if overlap && bx != 0 {
            offsets[1] = offsets[0];
        }
        offsets[0][0] = scalar::get_random_number(8, &mut seed[0]);
        if ust_var {
            offsets[0][1] = scalar::get_random_number(8, &mut seed[1]);
        }
        let xstart = if overlap && bx != 0 { bw.min(2) } else { 0 };
        let of = Ofsetler512::yeni(&offsets, 0, 0);
        fguv444_kismi_blok::<CSFL>(
            dst,
            src,
            stride,
            luma,
            luma_stride,
            data,
            scaling,
            lut,
            &of,
            bx,
            bw,
            bh,
            xstart,
            ystart,
            uv,
            is_id,
        );
    }
}

/// `fguv 420 / 422` (`SX = 1`) sıcak gövdesi.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
#[inline(never)]
fn fguv42x_govde_ic<const SY: usize, const CSFL: bool>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8; OLCEKLEME_BOYU],
    lut: &GrainLut,
    bh: usize,
    row_num: usize,
    luma: &[u8],
    luma_stride: usize,
    uv: usize,
    is_id: bool,
) {
    let bh_k = FG_BLOCK_SIZE >> SY;
    let bstep = FG_BLOCK_SIZE >> 1;
    assert!(bh >= 1 && bh <= bh_k && (1..=(1 << 16)).contains(&pw));
    assert!(stride >= pw && stride <= (1 << 20));
    let luma_min_w = ((pw - 1) << 1) + 2;
    assert!(luma_stride >= luma_min_w && luma_stride <= (1 << 20));
    let gerekli = (bh - 1) * stride + pw;
    let luma_gerekli = ((bh - 1) << SY) * luma_stride + luma_min_w;
    assert!(src.len() >= gerekli && dst.len() >= gerekli && luma.len() >= luma_gerekli);
    let overlap = data.overlap_flag;
    let ust_var = overlap && row_num != 0;
    let sinir = if data.clip_to_restricted_range {
        (16, if is_id { 235 } else { 240 })
    } else {
        (0, 255)
    };
    let shift = data.scaling_shift as u32;
    let (lm, m, off) = (data.uv_luma_mult[uv], data.uv_mult[uv], data.uv_offset[uv]);
    let carpan = _mm512_set1_epi16(1 << (15 - shift));
    let carpan_uv = _mm512_set1_epi16(((lm & 0xFF) as i16) | ((m as i16) << 8));
    let ofs = _mm512_set1_epi16(off as i16);
    let vmin = _mm512_set1_epi8(sinir.0 as i8);
    let vmax = _mm512_set1_epi8(sinir.1 as u8 as i8);
    let pb_even = load_u8x64(&PB_EVEN);
    let pb_odd = load_u8x64(&PB_ODD);
    let z = _mm512_setzero_si512();
    let w0 = _mm512_set1_epi16(23);
    let w1 = _mm512_set1_epi16(22);
    let bin24 = _mm512_set1_epi16(1 << 10);
    let gmin = _mm512_set1_epi16(GRAIN_MIN as i16);
    let gmax = _mm512_set1_epi16(GRAIN_MAX as i16);
    let tablo = OlcekTablosu::yeni(scaling);
    let ystart = if ust_var { bh.min(2 >> SY) } else { 0 };
    let lstep = luma_stride << SY;
    let s_span4 = 3 * stride + 16;
    let s_span2 = stride + 16;
    let l_span4 = 3 * lstep + 32;
    let l_span2 = lstep + 32;
    let mut seed = satir_tohumlari(data, row_num, overlap);
    let mut offsets = [[0i32; 2]; 2];
    let mut bx = 0usize;

    macro_rules! dortlu16 {
        ($i0:expr, $lx0:expr, $g_lo:expr, $g_hi:expr) => {{
            let i0_v: usize = $i0;
            let s_win = &src[i0_v..i0_v + s_span4];
            let s_mid = &s_win[stride..stride + s_span2];
            let s01 = _mm256_inserti128_si256::<1>(
                _mm256_castsi128_si256(load_u8x16(s_win.first_chunk::<16>().unwrap())),
                load_u8x16(s_mid.first_chunk::<16>().unwrap()),
            );
            let s23 = _mm256_inserti128_si256::<1>(
                _mm256_castsi128_si256(load_u8x16(s_mid.last_chunk::<16>().unwrap())),
                load_u8x16(s_win.last_chunk::<16>().unwrap()),
            );
            let s = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(s01), s23);
            let lx0_v: usize = $lx0;
            let l_win = &luma[lx0_v..lx0_v + l_span4];
            let l_mid = &l_win[lstep..lstep + l_span2];
            let l01 = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(load_u8x32(l_win.first_chunk::<32>().unwrap())),
                load_u8x32(l_mid.first_chunk::<32>().unwrap()),
            );
            let l23 = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(load_u8x32(l_mid.last_chunk::<32>().unwrap())),
                load_u8x32(l_win.last_chunk::<32>().unwrap()),
            );
            let even = _mm512_permutex2var_epi8(l01, pb_even, l23);
            let odd = _mm512_permutex2var_epi8(l01, pb_odd, l23);
            let avg_u8 = _mm512_avg_epu8(even, odd);
            let out = fguv_hesapla64!(
                CSFL, s, avg_u8, $g_lo, $g_hi, tablo, carpan, carpan_uv, ofs, vmin, vmax, z
            );
            let out_lo = _mm512_castsi512_si256(out);
            let out_hi = _mm512_extracti64x4_epi64::<1>(out);
            let d_win = &mut dst[i0_v..i0_v + s_span4];
            store_u8x16(
                d_win.first_chunk_mut::<16>().unwrap(),
                _mm256_castsi256_si128(out_lo),
            );
            store_u8x16(
                d_win.last_chunk_mut::<16>().unwrap(),
                _mm256_extracti128_si256::<1>(out_hi),
            );
            let d_mid = &mut d_win[stride..stride + s_span2];
            store_u8x16(
                d_mid.first_chunk_mut::<16>().unwrap(),
                _mm256_extracti128_si256::<1>(out_lo),
            );
            store_u8x16(
                d_mid.last_chunk_mut::<16>().unwrap(),
                _mm256_castsi256_si128(out_hi),
            );
        }};
    }

    // Tam blok (`bw == 16`, `bh % 4 == 0`) sıcak döngüsü: sıfır iç `call`.
    if (bh & 3) == 0 {
        let v_mask: u32 = if SY == 1 { 0x00FF } else { 0xFFFF };
        let vw0 = if SY == 1 {
            _mm512_set1_epi16(23)
        } else {
            _mm512_mask_blend_epi16(0xFF00, _mm512_set1_epi16(27), _mm512_set1_epi16(17))
        };
        let vw1 = if SY == 1 {
            _mm512_set1_epi16(22)
        } else {
            _mm512_mask_blend_epi16(0xFF00, _mm512_set1_epi16(17), _mm512_set1_epi16(27))
        };
        while bx + bstep <= pw {
            if overlap && bx != 0 {
                offsets[1] = offsets[0];
            }
            offsets[0][0] = scalar::get_random_number(8, &mut seed[0]);
            if ust_var {
                offsets[0][1] = scalar::get_random_number(8, &mut seed[1]);
            }
            let xstart = usize::from(overlap && bx != 0);
            let of = Ofsetler512::yeni(&offsets, 1, SY);
            let mut y = 0usize;
            let mut i0 = bx;
            let mut lx0 = bx << 1;
            let ox_s = of.sol.0 + bstep;
            if ystart > 0 {
                let oy = of.o.1;
                let g0 = load_i16x16(chunk(&lut[oy], of.o.0));
                let g1 = load_i16x16(chunk(&lut[oy + 1], of.o.0));
                let g2 = load_i16x16(chunk(&lut[oy + 2], of.o.0));
                let g3 = load_i16x16(chunk(&lut[oy + 3], of.o.0));
                let g_a = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(g0), g1);
                let g_b = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(g2), g3);
                let mut g_lo = _mm512_shuffle_i32x4::<0x88>(g_a, g_b);
                let mut g_hi = _mm512_shuffle_i32x4::<0xDD>(g_a, g_b);
                let uy = of.ust.1 + bh_k;
                let t0 = load_i16x16(chunk(&lut[uy], of.ust.0));
                let t1 = if SY == 0 {
                    load_i16x16(chunk(&lut[uy + 1], of.ust.0))
                } else {
                    _mm256_setzero_si256()
                };
                let t_a = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(t0), t1);
                let mut t_lo = _mm512_shuffle_i32x4::<0x88>(t_a, z);
                let t_hi = _mm512_shuffle_i32x4::<0xDD>(t_a, z);
                if xstart > 0 {
                    let oy_s = of.sol.1;
                    let eski = _mm512_setr_epi32(
                        i32::from(lut[oy_s][ox_s]),
                        0,
                        0,
                        0,
                        i32::from(lut[oy_s + 1][ox_s]),
                        0,
                        0,
                        0,
                        i32::from(lut[oy_s + 2][ox_s]),
                        0,
                        0,
                        0,
                        i32::from(lut[oy_s + 3][ox_s]),
                        0,
                        0,
                        0,
                    );
                    let x = _mm512_add_epi16(
                        _mm512_mullo_epi16(eski, w0),
                        _mm512_mullo_epi16(g_lo, w1),
                    );
                    let r = _mm512_max_epi16(
                        _mm512_min_epi16(_mm512_mulhrs_epi16(x, bin24), gmax),
                        gmin,
                    );
                    g_lo = _mm512_mask_blend_epi16(0x0101_0101, g_lo, r);
                    let ky = of.kose.1 + bh_k;
                    let kx = of.kose.0 + bstep;
                    let k1 = if SY == 0 {
                        i32::from(lut[ky + 1][kx])
                    } else {
                        0
                    };
                    let eski_t = _mm512_setr_epi32(
                        i32::from(lut[ky][kx]),
                        0,
                        0,
                        0,
                        k1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    );
                    let xt = _mm512_add_epi16(
                        _mm512_mullo_epi16(eski_t, w0),
                        _mm512_mullo_epi16(t_lo, w1),
                    );
                    let rt = _mm512_max_epi16(
                        _mm512_min_epi16(_mm512_mulhrs_epi16(xt, bin24), gmax),
                        gmin,
                    );
                    t_lo = _mm512_mask_blend_epi16(0x0101, t_lo, rt);
                }
                let xv_lo =
                    _mm512_add_epi16(_mm512_mullo_epi16(t_lo, vw0), _mm512_mullo_epi16(g_lo, vw1));
                let rv_lo = _mm512_max_epi16(
                    _mm512_min_epi16(_mm512_mulhrs_epi16(xv_lo, bin24), gmax),
                    gmin,
                );
                g_lo = _mm512_mask_blend_epi16(v_mask, g_lo, rv_lo);
                let xv_hi =
                    _mm512_add_epi16(_mm512_mullo_epi16(t_hi, vw0), _mm512_mullo_epi16(g_hi, vw1));
                let rv_hi = _mm512_max_epi16(
                    _mm512_min_epi16(_mm512_mulhrs_epi16(xv_hi, bin24), gmax),
                    gmin,
                );
                g_hi = _mm512_mask_blend_epi16(v_mask, g_hi, rv_hi);
                dortlu16!(i0, lx0, g_lo, g_hi);
                i0 += 4 * stride;
                lx0 += 4 * lstep;
                y = 4;
            }
            if xstart > 0 {
                while y + 4 <= bh {
                    let oy = of.o.1 + y;
                    let g0 = load_i16x16(chunk(&lut[oy], of.o.0));
                    let g1 = load_i16x16(chunk(&lut[oy + 1], of.o.0));
                    let g2 = load_i16x16(chunk(&lut[oy + 2], of.o.0));
                    let g3 = load_i16x16(chunk(&lut[oy + 3], of.o.0));
                    let g_a = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(g0), g1);
                    let g_b = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(g2), g3);
                    let mut g_lo = _mm512_shuffle_i32x4::<0x88>(g_a, g_b);
                    let g_hi = _mm512_shuffle_i32x4::<0xDD>(g_a, g_b);
                    let oy_s = of.sol.1 + y;
                    let eski = _mm512_setr_epi32(
                        i32::from(lut[oy_s][ox_s]),
                        0,
                        0,
                        0,
                        i32::from(lut[oy_s + 1][ox_s]),
                        0,
                        0,
                        0,
                        i32::from(lut[oy_s + 2][ox_s]),
                        0,
                        0,
                        0,
                        i32::from(lut[oy_s + 3][ox_s]),
                        0,
                        0,
                        0,
                    );
                    let x = _mm512_add_epi16(
                        _mm512_mullo_epi16(eski, w0),
                        _mm512_mullo_epi16(g_lo, w1),
                    );
                    let r = _mm512_max_epi16(
                        _mm512_min_epi16(_mm512_mulhrs_epi16(x, bin24), gmax),
                        gmin,
                    );
                    g_lo = _mm512_mask_blend_epi16(0x0101_0101, g_lo, r);
                    dortlu16!(i0, lx0, g_lo, g_hi);
                    i0 += 4 * stride;
                    lx0 += 4 * lstep;
                    y += 4;
                }
            } else {
                while y + 4 <= bh {
                    let oy = of.o.1 + y;
                    let g0 = load_i16x16(chunk(&lut[oy], of.o.0));
                    let g1 = load_i16x16(chunk(&lut[oy + 1], of.o.0));
                    let g2 = load_i16x16(chunk(&lut[oy + 2], of.o.0));
                    let g3 = load_i16x16(chunk(&lut[oy + 3], of.o.0));
                    let g_a = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(g0), g1);
                    let g_b = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(g2), g3);
                    let g_lo = _mm512_shuffle_i32x4::<0x88>(g_a, g_b);
                    let g_hi = _mm512_shuffle_i32x4::<0xDD>(g_a, g_b);
                    dortlu16!(i0, lx0, g_lo, g_hi);
                    i0 += 4 * stride;
                    lx0 += 4 * lstep;
                    y += 4;
                }
            }
            bx += bstep;
        }
    }
    while bx < pw {
        let bw = bstep.min(pw - bx);
        if overlap && bx != 0 {
            offsets[1] = offsets[0];
        }
        offsets[0][0] = scalar::get_random_number(8, &mut seed[0]);
        if ust_var {
            offsets[0][1] = scalar::get_random_number(8, &mut seed[1]);
        }
        let xstart = usize::from(overlap && bx != 0);
        let of = Ofsetler512::yeni(&offsets, 1, SY);
        fguv42x_kismi_adim::<SY, CSFL>(
            dst,
            src,
            stride,
            luma,
            luma_stride,
            data,
            scaling,
            lut,
            &of,
            bx,
            bw,
            0,
            bh,
            xstart,
            ystart,
            uv,
            is_id,
        );
        bx += bstep;
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi")]
fn fguv_govde<const SX: usize, const SY: usize>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8; OLCEKLEME_BOYU],
    lut: &GrainLut,
    bh: usize,
    row_num: usize,
    luma: &[u8],
    luma_stride: usize,
    uv: usize,
    is_id: bool,
) {
    if SX == 0 {
        if data.chroma_scaling_from_luma {
            fguv444_govde_ic::<true>(
                dst,
                src,
                stride,
                data,
                pw,
                scaling,
                lut,
                bh,
                row_num,
                luma,
                luma_stride,
                uv,
                is_id,
            );
        } else {
            fguv444_govde_ic::<false>(
                dst,
                src,
                stride,
                data,
                pw,
                scaling,
                lut,
                bh,
                row_num,
                luma,
                luma_stride,
                uv,
                is_id,
            );
        }
    } else if data.chroma_scaling_from_luma {
        fguv42x_govde_ic::<SY, true>(
            dst,
            src,
            stride,
            data,
            pw,
            scaling,
            lut,
            bh,
            row_num,
            luma,
            luma_stride,
            uv,
            is_id,
        );
    } else {
        fguv42x_govde_ic::<SY, false>(
            dst,
            src,
            stride,
            data,
            pw,
            scaling,
            lut,
            bh,
            row_num,
            luma,
            luma_stride,
            uv,
            is_id,
        );
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
    let tablo = scaling.first_chunk::<OLCEKLEME_BOYU>();
    let Some(tablo) = tablo.filter(|_| {
        bd == FgBitDepth::B8 && fgy_sozlesmesi(dst.len(), src.len(), stride, pw, bh, data)
    }) else {
        fg_skaler_devri_say();
        let f = match bd {
            FgBitDepth::B8 => scalar::fgy_32x32xn::<8>,
            FgBitDepth::B10 => scalar::fgy_32x32xn::<10>,
            FgBitDepth::B12 => scalar::fgy_32x32xn::<12>,
        };
        return f(
            dst, src, stride, data, pw, scaling, grain_lut, bh, row_num, bd,
        );
    };
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`fgy_fn`); belirteç AVX512ICL (F/BW/VL/VBMI) tespiti + maske izninin
    // kanıtıdır. (2) sınır: `fgy_sozlesmesi` `src`/`dst`'nin `(bh−1)·stride + pw`
    // baytı kapsadığını ve `stride ≥ pw`'yi, `first_chunk` tablonun 256 baytını
    // garanti eder; kısmi genişlikler maskeli yükleme/saklama ile yalnız `bw`
    // bayta dokunur. (3) hizalama/takma ad: `dst` tek `&mut`, diğerleri `&`.
    unsafe { fgy_govde(dst, src, stride, data, pw, tablo, grain_lut, bh, row_num) }
}

/// `fguv` girişlerinin ortak sözleşme + skaler devir denetimi.
fn fguv_on_denetim<'a>(
    dst: &[u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &'a [u8],
    bh: usize,
    luma: &[u8],
    luma_stride: usize,
    uv: usize,
    sx: usize,
    sy: usize,
    bd: FgBitDepth,
) -> Option<&'a [u8; OLCEKLEME_BOYU]> {
    scaling.first_chunk::<OLCEKLEME_BOYU>().filter(|_| {
        bd == FgBitDepth::B8
            && fguv_sozlesmesi(
                dst.len(),
                src.len(),
                stride,
                pw,
                bh,
                luma.len(),
                luma_stride,
                uv,
                sx,
                sy,
                data,
            )
            && uv_katsayi_sozlesmesi(data, uv)
    })
}

fn fguv_420_giris(
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
    let Some(tablo) = fguv_on_denetim(
        dst,
        src,
        stride,
        data,
        pw,
        scaling,
        bh,
        luma,
        luma_stride,
        uv,
        1,
        1,
        bd,
    ) else {
        fg_skaler_devri_say();
        let f = match bd {
            FgBitDepth::B8 => scalar::fguv_32x32xn_420::<8>,
            FgBitDepth::B10 => scalar::fguv_32x32xn_420::<10>,
            FgBitDepth::B12 => scalar::fguv_32x32xn_420::<12>,
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
    };
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`fguv_fn`); belirteç AVX512ICL (F/BW/VL/VBMI) tespiti + maske izninin
    // kanıtıdır. (2) sınır: `fguv_on_denetim` `src`/`dst` ve `luma` (4:2:0)
    // dilim sınırlarını, `uv < 2`'yi ve 256 baytlık ölçekleme tablosunu garanti
    // eder; kısmi genişlikler maskeli yükleme/saklama ile sınırlıdır.
    // (3) hizalama/takma ad: `dst` tek `&mut`, diğerleri `&`.
    unsafe {
        fguv_govde::<1, 1>(
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
        )
    }
}

fn fguv_422_giris(
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
    let Some(tablo) = fguv_on_denetim(
        dst,
        src,
        stride,
        data,
        pw,
        scaling,
        bh,
        luma,
        luma_stride,
        uv,
        1,
        0,
        bd,
    ) else {
        fg_skaler_devri_say();
        let f = match bd {
            FgBitDepth::B8 => scalar::fguv_32x32xn_422::<8>,
            FgBitDepth::B10 => scalar::fguv_32x32xn_422::<10>,
            FgBitDepth::B12 => scalar::fguv_32x32xn_422::<12>,
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
    };
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`fguv_fn`); belirteç AVX512ICL (F/BW/VL/VBMI) tespiti + maske izninin
    // kanıtıdır. (2) sınır: `fguv_on_denetim` `src`/`dst` ve `luma` (4:2:2)
    // dilim sınırlarını, `uv < 2`'yi ve 256 baytlık ölçekleme tablosunu garanti
    // eder; kısmi genişlikler maskeli yükleme/saklama ile sınırlıdır.
    // (3) hizalama/takma ad: `dst` tek `&mut`, diğerleri `&`.
    unsafe {
        fguv_govde::<1, 0>(
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
        )
    }
}

fn fguv_444_giris(
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
    let Some(tablo) = fguv_on_denetim(
        dst,
        src,
        stride,
        data,
        pw,
        scaling,
        bh,
        luma,
        luma_stride,
        uv,
        0,
        0,
        bd,
    ) else {
        fg_skaler_devri_say();
        let f = match bd {
            FgBitDepth::B8 => scalar::fguv_32x32xn_444::<8>,
            FgBitDepth::B10 => scalar::fguv_32x32xn_444::<10>,
            FgBitDepth::B12 => scalar::fguv_32x32xn_444::<12>,
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
    };
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`fguv_fn`); belirteç AVX512ICL (F/BW/VL/VBMI) tespiti + maske izninin
    // kanıtıdır. (2) sınır: `fguv_on_denetim` `src`/`dst` ve `luma` (4:4:4)
    // dilim sınırlarını, `uv < 2`'yi ve 256 baytlık ölçekleme tablosunu garanti
    // eder; kısmi genişlikler maskeli yükleme/saklama ile sınırlıdır.
    // (3) hizalama/takma ad: `dst` tek `&mut`, diğerleri `&`.
    unsafe {
        fguv_govde::<0, 0>(
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
        )
    }
}

/// `fgy_32x32xn` 8 bpc AVX512ICL yuvası (yalnız belirteçle; `SLOW_GATHER`'dan etkilenmez).
#[must_use]
pub fn fgy_fn(_icl: Avx512IclToken) -> FgyFn {
    fgy_giris
}

/// `fguv_32x32xn[düzen]` 8 bpc AVX512ICL yuvası (yalnız belirteçle; `SLOW_GATHER`'dan etkilenmez).
#[must_use]
pub fn fguv_fn(_icl: Avx512IclToken, duzen: usize) -> FguvFn {
    match duzen {
        0 => fguv_420_giris,
        1 => fguv_422_giris,
        _ => fguv_444_giris,
    }
}
