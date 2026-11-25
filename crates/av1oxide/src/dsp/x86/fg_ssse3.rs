//! Film grain 8 bpc SSSE3 çekirdekleri — F4-FG b (P41).
//!
//! ## Sözleşme (kopyasız)
//!
//! Çekirdekler çağrı başına yerel kopya yapmaz; `src`, `dst` ve luma
//! dilimlerine doğrudan erişir ve yalnız sözleşme bölgesini okur/yazar
//! (`fgy`: satır `0..bh` × sütun `0..pw`; `fguv` luma: satır `y << sy`, sütun
//! `0 ..= ((pw − 1) << sx) + sx`). Girişte O(1) denetim
//! ([`crate::dsp::fg::fgy_sozlesmesi`], [`crate::dsp::fg::fguv_sozlesmesi`]) ve
//! ölçekleme tablosunun en az [`OLCEKLEME_BOYU`] (256) bayt olması aranır;
//! sözleşme dışı girdi skaler gövdeye devreder ve sayılır
//! ([`crate::dsp::fg::fg_skaler_devri_say`]). Grain LUT'u sabit boyutlu türdür
//! (`[[i16; 82]; 73]`).
//!
//! ## Yöntem ve ISA sınırı (yalnız SSE2 + SSSE3)
//!
//! Bu modüldeki bütün SIMD gövdeleri yalnız `#[target_feature(enable = "ssse3")]`
//! altında derlenir ve SSE4.1+ komutu içermez:
//! - Satır 16 piksellik parçalarla (iki `__m128i` `8 × i16` yarı: `lo`, `hi`)
//!   işlenir; kaynak baytları `_mm_unpacklo_epi8`/`_mm_unpackhi_epi8` ile
//!   `i16`'ya genişletilir.
//! - Ölçekleme tablosu `&[u8; 256]` üzerinde `usize::from(b: u8)` ile
//!   sınır denetimsiz okunur (`olcekle4` + `_mm_set_epi64x`; XMM'deki
//!   dizinler iki 64-bit `_mm_cvtsi128_si64` ile çıkarılır).
//! - `fguv` luma ortalaması `_mm_maddubs_epi16` + `_mm_avg_epu16` (`pavgw`),
//!   kroma ölçekleme dizini `_mm_maddubs_epi16` + `_mm_srai_epi16::<6>` +
//!   `_mm_add_epi16` + `_mm_packus_epi16` ile hesaplanır (`dav1d`
//!   `filmgrain_sse.asm` ile birebir).
//! - `generate_grain_*` AR süzgecinde üst satır dokunuş çiftleri
//!   `_mm_madd_epi16` (`pmaddwd`) ile 32-bit akümülatöre toplanır, luma
//!   katkısı `_mm_hadd_epi16` (`phaddw`) ile 16-bitte ortalanır.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use super::fg::{
    GRAIN_MAX, GRAIN_MIN, Ofsetler, W, gauss_doldur, grain_sozlesmesi, kenar_grain, ofset,
    parcalar, skaler_piksel,
};
use super::mem::{chunk, chunk_mut, load_i16x8, load_u8x16, store_i16x8, store_u8x16};
use crate::bitstream::frame_header::FilmGrainData;
use crate::cpu::Ssse3Token;
use crate::dsp::fg::{
    FG_BLOCK_SIZE, FgBitDepth, FguvFn, FgyFn, GRAIN_HEIGHT, GRAIN_WIDTH, GenerateGrainUvFn,
    GenerateGrainYFn, GrainLut, SUB_GRAIN_HEIGHT, SUB_GRAIN_WIDTH, fg_skaler_devri_say,
    fguv_sozlesmesi, fgy_sozlesmesi,
};
use crate::dsp::scalar::fg as scalar;
use core::arch::x86_64::{
    __m128i, _mm_add_epi16, _mm_add_epi32, _mm_avg_epu16, _mm_cvtsi32_si128, _mm_cvtsi128_si64,
    _mm_hadd_epi16, _mm_insert_epi16, _mm_madd_epi16, _mm_maddubs_epi16, _mm_max_epi16,
    _mm_max_epu8, _mm_min_epi16, _mm_min_epu8, _mm_mulhrs_epi16, _mm_mullo_epi16, _mm_packus_epi16,
    _mm_set_epi64x, _mm_set1_epi8, _mm_set1_epi16, _mm_set1_epi32, _mm_setr_epi16,
    _mm_setzero_si128, _mm_srai_epi16, _mm_srli_si128, _mm_unpackhi_epi8, _mm_unpackhi_epi16,
    _mm_unpacklo_epi8, _mm_unpacklo_epi16,
};

/// 8 bpc SSSE3/AVX512ICL ölçekleme tablosu boyu (`0..=255` tek bayt okuma).
pub(super) const OLCEKLEME_BOYU: usize = 256;

/// 32-bit yazmaçtaki 4 baytlık dizinden 4 × `u16` ölçekleme değerini tek `i64`'te
/// paketler (`(b >> k) & 0xFF < 256` olduğu için sınır denetimi üretilmez).
#[inline(always)]
fn olcekle4(scaling: &[u8; OLCEKLEME_BOYU], w: u32) -> i64 {
    let s0 = i64::from(scaling[(w & 0xFF) as usize]);
    let s1 = i64::from(scaling[((w >> 8) & 0xFF) as usize]);
    let s2 = i64::from(scaling[((w >> 16) & 0xFF) as usize]);
    let s3 = i64::from(scaling[(w >> 24) as usize]);
    s0 | (s1 << 16) | (s2 << 32) | (s3 << 48)
}

/// 64-bit yazmaçtaki 8 baytlık dizinlerden 8 × `i16` ölçekleme değeri.
#[target_feature(enable = "ssse3")]
#[inline]
fn olcekleme_u64(scaling: &[u8; OLCEKLEME_BOYU], w: u64) -> __m128i {
    _mm_set_epi64x(
        olcekle4(scaling, (w >> 32) as u32),
        olcekle4(scaling, w as u32),
    )
}

/// XMM yazmacındaki 16 × `u8` dizinden `(lo, hi)` (2 × `8 × i16`) ölçekleme değeri.
#[target_feature(enable = "ssse3")]
#[inline]
fn olcekleme_xmm(scaling: &[u8; OLCEKLEME_BOYU], idx: __m128i) -> (__m128i, __m128i) {
    let lo = _mm_cvtsi128_si64(idx) as u64;
    let hi = _mm_cvtsi128_si64(_mm_srli_si128::<8>(idx)) as u64;
    (olcekleme_u64(scaling, lo), olcekleme_u64(scaling, hi))
}

/// `clamp(round2(eski · w0 + yeni · w1, 5), −128, 127)` (8 şerit; `pmulhrsw(x, 1024)`).
#[target_feature(enable = "ssse3")]
#[inline]
fn harmanla(eski: __m128i, yeni: __m128i, w: [i32; 2]) -> __m128i {
    let x = _mm_add_epi16(
        _mm_mullo_epi16(eski, _mm_set1_epi16(w[0] as i16)),
        _mm_mullo_epi16(yeni, _mm_set1_epi16(w[1] as i16)),
    );
    let r = _mm_mulhrs_epi16(x, _mm_set1_epi16(1 << 10));
    _mm_max_epi16(
        _mm_min_epi16(r, _mm_set1_epi16(GRAIN_MAX as i16)),
        _mm_set1_epi16(GRAIN_MIN as i16),
    )
}

/// Yatay örtüşme (ilk 8 şerit): `x < 2 >> sx` şeritleri `W[sx][x]` ile harmanlanır,
/// kalan şeritler `(0, 32)` ağırlığıyla değişmez.
#[target_feature(enable = "ssse3")]
#[inline]
fn yatay_harmanla(eski: __m128i, yeni: __m128i, sx: usize) -> __m128i {
    let (w0, w1) = if sx == 0 {
        (
            _mm_setr_epi16(27, 17, 0, 0, 0, 0, 0, 0),
            _mm_setr_epi16(17, 27, 32, 32, 32, 32, 32, 32),
        )
    } else {
        (
            _mm_setr_epi16(23, 0, 0, 0, 0, 0, 0, 0),
            _mm_setr_epi16(22, 32, 32, 32, 32, 32, 32, 32),
        )
    };
    let x = _mm_add_epi16(_mm_mullo_epi16(eski, w0), _mm_mullo_epi16(yeni, w1));
    let r = _mm_mulhrs_epi16(x, _mm_set1_epi16(1 << 10));
    _mm_max_epi16(
        _mm_min_epi16(r, _mm_set1_epi16(GRAIN_MAX as i16)),
        _mm_set1_epi16(GRAIN_MIN as i16),
    )
}

/// LUT satırının `x` ve `x + 1` örnekleri ilk iki şeritte.
#[target_feature(enable = "ssse3")]
#[inline]
fn iki_ornek(satir: &[i16], x: usize) -> __m128i {
    let a = chunk::<i16, 2>(satir, x);
    let w = (u32::from(a[0] as u16) | (u32::from(a[1] as u16) << 16)) as i32;
    _mm_cvtsi32_si128(w)
}

/// `g`'nin `serit` (0 ya da 1) şeridine `v` yazar.
#[target_feature(enable = "ssse3")]
#[inline]
fn serit_yaz(g: __m128i, serit: usize, v: i32) -> __m128i {
    if serit == 0 {
        _mm_insert_epi16::<0>(g, v)
    } else {
        _mm_insert_epi16::<1>(g, v)
    }
}

/// Satırın 16 piksellik grain parçası `(lo, hi)`: yatay örtüşme yalnız `lo`
/// yarısındadır (`xstart ≤ 2`), dikey örtüşme her iki yarıyı harmanlar.
#[target_feature(enable = "ssse3")]
#[inline(never)]
fn grain_parca(
    lut: &GrainLut,
    offsets: &[[i32; 2]; 2],
    of: &Ofsetler,
    sx: usize,
    sy: usize,
    x0: usize,
    y: usize,
    xstart: usize,
    ystart: usize,
) -> (__m128i, __m128i) {
    let bh_k = FG_BLOCK_SIZE >> sy;
    let bw_k = FG_BLOCK_SIZE >> sx;
    let mut g_lo = load_i16x8(chunk(&lut[of.o.1 + y], of.o.0 + x0));
    let mut g_hi = load_i16x8(chunk(&lut[of.o.1 + y], of.o.0 + x0 + 8));
    let yatay = x0 == 0 && xstart > 0;
    if yatay {
        g_lo = yatay_harmanla(iki_ornek(&lut[of.sol.1 + y], of.sol.0 + bw_k), g_lo, sx);
    }
    if y < ystart {
        let mut t_lo = load_i16x8(chunk(&lut[of.ust.1 + y + bh_k], of.ust.0 + x0));
        let t_hi = load_i16x8(chunk(&lut[of.ust.1 + y + bh_k], of.ust.0 + x0 + 8));
        if yatay {
            t_lo = yatay_harmanla(
                iki_ornek(&lut[of.kose.1 + y + bh_k], of.kose.0 + bw_k),
                t_lo,
                sx,
            );
        }
        g_lo = harmanla(t_lo, g_lo, W[sy][y]);
        g_hi = harmanla(t_hi, g_hi, W[sy][y]);
    }
    if x0 > 0 {
        for x in x0..xstart.min(x0 + 8) {
            g_lo = serit_yaz(
                g_lo,
                x - x0,
                kenar_grain(lut, offsets, sx, sy, x, y, xstart, ystart),
            );
        }
    }
    (g_lo, g_hi)
}

/// `fgy` 16 piksel: `d16` ← `clamp(s16 + round2(scaling[s16] · g, shift), min, max)`.
#[target_feature(enable = "ssse3")]
#[inline]
fn fgy_parca(
    d16: &mut [u8; 16],
    s16: &[u8; 16],
    g: (__m128i, __m128i),
    scaling: &[u8; OLCEKLEME_BOYU],
    carpan: __m128i,
    vmin: __m128i,
    vmax: __m128i,
) {
    let s = load_u8x16(s16);
    let z = _mm_setzero_si128();
    let (s_lo, s_hi) = (_mm_unpacklo_epi8(s, z), _mm_unpackhi_epi8(s, z));
    let (sc_lo, sc_hi) = olcekleme_xmm(scaling, s);
    let n_lo = _mm_mulhrs_epi16(_mm_mullo_epi16(sc_lo, g.0), carpan);
    let n_hi = _mm_mulhrs_epi16(_mm_mullo_epi16(sc_hi, g.1), carpan);
    let b = _mm_packus_epi16(_mm_add_epi16(s_lo, n_lo), _mm_add_epi16(s_hi, n_hi));
    store_u8x16(d16, _mm_max_epu8(_mm_min_epu8(b, vmax), vmin));
}

/// `fgy` soğuk yol: dikey örtüşme satırları (`y < ystart`) veya kısmi blok (`bw < 32`).
#[target_feature(enable = "ssse3")]
#[inline(never)]
fn fgy_kismi_veya_ust(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    scaling: &[u8; OLCEKLEME_BOYU],
    lut: &GrainLut,
    offsets: &[[i32; 2]; 2],
    of: &Ofsetler,
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
    let (bas, n) = parcalar(bw);
    let sol = |y: usize| iki_ornek(&lut[of.sol.1 + y], of.sol.0 + FG_BLOCK_SIZE);
    for y in y_bas..y_son {
        let lut_satir = &lut[of.o.1 + y];
        for &x0 in &bas[..n] {
            let g = if y < ystart || (x0 > 0 && x0 < xstart) {
                grain_parca(lut, offsets, of, 0, 0, x0, y, xstart, ystart)
            } else {
                let mut g_lo = load_i16x8(chunk(lut_satir, of.o.0 + x0));
                let g_hi = load_i16x8(chunk(lut_satir, of.o.0 + x0 + 8));
                if x0 == 0 && xstart > 0 {
                    g_lo = yatay_harmanla(sol(y), g_lo, 0);
                }
                (g_lo, g_hi)
            };
            let i = y * stride + bx + x0;
            fgy_parca(
                chunk_mut(dst, i),
                chunk(src, i),
                g,
                scaling,
                carpan,
                vmin,
                vmax,
            );
        }
    }
}

/// `fgy` tam blok (`bw = 32`) sıcak döngüsü: içinde `call` olmadığı için
/// XMM sabitleri (`carpan`, `vmin`, `vmax`) yazmaçta kalır.
#[target_feature(enable = "ssse3")]
#[inline(never)]
fn fgy_tam_blok(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    scaling: &[u8; OLCEKLEME_BOYU],
    lut: &GrainLut,
    of: &Ofsetler,
    bx: usize,
    ystart: usize,
    bh: usize,
    xstart: usize,
    carpan: __m128i,
    vmin: __m128i,
    vmax: __m128i,
) {
    if xstart > 0 {
        for y in ystart..bh {
            let g32 = chunk::<i16, 32>(&lut[of.o.1 + y], of.o.0);
            let sol = iki_ornek(&lut[of.sol.1 + y], of.sol.0 + FG_BLOCK_SIZE);
            let g0_lo = yatay_harmanla(sol, load_i16x8(chunk(g32, 0)), 0);
            let g0_hi = load_i16x8(chunk(g32, 8));
            let i = y * stride + bx;
            let s32 = chunk::<u8, 32>(src, i);
            let d32 = chunk_mut::<u8, 32>(dst, i);
            fgy_parca(
                chunk_mut(d32, 0),
                chunk(s32, 0),
                (g0_lo, g0_hi),
                scaling,
                carpan,
                vmin,
                vmax,
            );
            let g1_lo = load_i16x8(chunk(g32, 16));
            let g1_hi = load_i16x8(chunk(g32, 24));
            fgy_parca(
                chunk_mut(d32, 16),
                chunk(s32, 16),
                (g1_lo, g1_hi),
                scaling,
                carpan,
                vmin,
                vmax,
            );
        }
    } else {
        for y in ystart..bh {
            let g32 = chunk::<i16, 32>(&lut[of.o.1 + y], of.o.0);
            let g0_lo = load_i16x8(chunk(g32, 0));
            let g0_hi = load_i16x8(chunk(g32, 8));
            let i = y * stride + bx;
            let s32 = chunk::<u8, 32>(src, i);
            let d32 = chunk_mut::<u8, 32>(dst, i);
            fgy_parca(
                chunk_mut(d32, 0),
                chunk(s32, 0),
                (g0_lo, g0_hi),
                scaling,
                carpan,
                vmin,
                vmax,
            );
            let g1_lo = load_i16x8(chunk(g32, 16));
            let g1_hi = load_i16x8(chunk(g32, 24));
            fgy_parca(
                chunk_mut(d32, 16),
                chunk(s32, 16),
                (g1_lo, g1_hi),
                scaling,
                carpan,
                vmin,
                vmax,
            );
        }
    }
}

#[target_feature(enable = "ssse3")]
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
    let overlap = data.overlap_flag;
    let rows = 1 + usize::from(overlap && row_num > 0);
    let sinir = if data.clip_to_restricted_range {
        (16, 235)
    } else {
        (0, 255)
    };
    let shift = data.scaling_shift as u32;
    let carpan = _mm_set1_epi16(1 << (15 - shift));
    let (vmin, vmax) = (
        _mm_set1_epi8(sinir.0 as i8),
        _mm_set1_epi8(sinir.1 as u8 as i8),
    );
    let mut seed = scalar::row_seeds(data, row_num, rows);
    let mut offsets = [[0i32; 2]; 2];
    let mut bx = 0usize;
    while bx < pw {
        let bw = FG_BLOCK_SIZE.min(pw - bx);
        if overlap && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = scalar::get_random_number(8, &mut seed[i]);
        }
        let ystart = if overlap && row_num != 0 {
            bh.min(2)
        } else {
            0
        };
        let xstart = if overlap && bx != 0 { bw.min(2) } else { 0 };
        if bw < 16 {
            let (ox, oy) = ofset(offsets[0][0], 0, 0);
            for y in 0..bh {
                for x in 0..bw {
                    let g = if x < xstart || y < ystart {
                        kenar_grain(lut, &offsets, 0, 0, x, y, xstart, ystart)
                    } else {
                        i32::from(lut[oy + y][ox + x])
                    };
                    let i = y * stride + bx + x;
                    let s = i32::from(src[i]);
                    dst[i] = skaler_piksel(s, src[i] as usize, g, scaling, shift, sinir);
                }
            }
        } else {
            let of = Ofsetler::yeni(&offsets, 0, 0);
            if bw == FG_BLOCK_SIZE {
                if ystart > 0 {
                    fgy_kismi_veya_ust(
                        dst, src, stride, scaling, lut, &offsets, &of, bx, bw, 0, ystart, xstart,
                        ystart, carpan, vmin, vmax,
                    );
                }
                if ystart < bh {
                    fgy_tam_blok(
                        dst, src, stride, scaling, lut, &of, bx, ystart, bh, xstart, carpan, vmin,
                        vmax,
                    );
                }
            } else {
                fgy_kismi_veya_ust(
                    dst, src, stride, scaling, lut, &offsets, &of, bx, bw, 0, bh, xstart, ystart,
                    carpan, vmin, vmax,
                );
            }
        }
        bx += FG_BLOCK_SIZE;
    }
}

/// `fguv` parçalarının değişmeyen bağlamı.
struct FguvBaglam {
    stride: usize,
    luma_stride: usize,
    carpan: __m128i,
    carpan_uv: __m128i,
    ofs: __m128i,
    vmin: __m128i,
    vmax: __m128i,
}

/// `fguv` 16 kroma pikseli: luma ortalaması (`l0`/`l1`), ölçekleme dizini,
/// gürültü ve kırpma.
#[target_feature(enable = "ssse3")]
#[inline]
fn fguv_parca<const SX: usize, const CSFL: bool>(
    d16: &mut [u8; 16],
    s16: &[u8; 16],
    l0: &[u8; 16],
    l1: &[u8; 16],
    k: &FguvBaglam,
    scaling: &[u8; OLCEKLEME_BOYU],
    g: (__m128i, __m128i),
) {
    let s = load_u8x16(s16);
    let z = _mm_setzero_si128();
    let (s_lo, s_hi) = (_mm_unpacklo_epi8(s, z), _mm_unpackhi_epi8(s, z));
    let (sc_lo, sc_hi) = if SX == 0 && CSFL {
        olcekleme_xmm(scaling, load_u8x16(l0))
    } else {
        let avg_u8 = if SX != 0 {
            let bir8 = _mm_set1_epi8(1);
            let a0 = _mm_avg_epu16(_mm_maddubs_epi16(load_u8x16(l0), bir8), z);
            let a1 = _mm_avg_epu16(_mm_maddubs_epi16(load_u8x16(l1), bir8), z);
            _mm_packus_epi16(a0, a1)
        } else {
            load_u8x16(l0)
        };
        let idx = if CSFL {
            avg_u8
        } else {
            let p_lo = _mm_unpacklo_epi8(avg_u8, s);
            let p_hi = _mm_unpackhi_epi8(avg_u8, s);
            let v_lo = _mm_add_epi16(
                _mm_srai_epi16::<6>(_mm_maddubs_epi16(p_lo, k.carpan_uv)),
                k.ofs,
            );
            let v_hi = _mm_add_epi16(
                _mm_srai_epi16::<6>(_mm_maddubs_epi16(p_hi, k.carpan_uv)),
                k.ofs,
            );
            _mm_packus_epi16(v_lo, v_hi)
        };
        olcekleme_xmm(scaling, idx)
    };
    let n_lo = _mm_mulhrs_epi16(_mm_mullo_epi16(sc_lo, g.0), k.carpan);
    let n_hi = _mm_mulhrs_epi16(_mm_mullo_epi16(sc_hi, g.1), k.carpan);
    let b = _mm_packus_epi16(_mm_add_epi16(s_lo, n_lo), _mm_add_epi16(s_hi, n_hi));
    store_u8x16(d16, _mm_max_epu8(_mm_min_epu8(b, k.vmax), k.vmin));
}

/// `fguv` soğuk yol: dikey örtüşme satırları (`y < ystart`) veya kısmi blok (`bw < bstep`).
#[target_feature(enable = "ssse3")]
#[inline(never)]
fn fguv_kismi_veya_ust<const SX: usize, const SY: usize, const CSFL: bool>(
    dst: &mut [u8],
    src: &[u8],
    luma: &[u8],
    k: &FguvBaglam,
    scaling: &[u8; OLCEKLEME_BOYU],
    lut: &GrainLut,
    offsets: &[[i32; 2]; 2],
    of: &Ofsetler,
    bx: usize,
    bw: usize,
    y_bas: usize,
    y_son: usize,
    xstart: usize,
    ystart: usize,
) {
    let bstep = FG_BLOCK_SIZE >> SX;
    let (bas, n) = parcalar(bw);
    let sol = |y: usize| iki_ornek(&lut[of.sol.1 + y], of.sol.0 + bstep);
    for y in y_bas..y_son {
        let lut_satir = &lut[of.o.1 + y];
        for &x0 in &bas[..n] {
            let g = if y < ystart || (x0 > 0 && x0 < xstart) {
                grain_parca(lut, offsets, of, SX, SY, x0, y, xstart, ystart)
            } else {
                let mut g_lo = load_i16x8(chunk(lut_satir, of.o.0 + x0));
                let g_hi = load_i16x8(chunk(lut_satir, of.o.0 + x0 + 8));
                if x0 == 0 && xstart > 0 {
                    g_lo = yatay_harmanla(sol(y), g_lo, SX);
                }
                (g_lo, g_hi)
            };
            let x = bx + x0;
            let i = y * k.stride + x;
            let lx = (y << SY) * k.luma_stride + (x << SX);
            if SX != 0 {
                let l32 = chunk::<u8, 32>(luma, lx);
                fguv_parca::<SX, CSFL>(
                    chunk_mut(dst, i),
                    chunk(src, i),
                    chunk(l32, 0),
                    chunk(l32, 16),
                    k,
                    scaling,
                    g,
                );
            } else {
                let l16 = chunk::<u8, 16>(luma, lx);
                fguv_parca::<0, CSFL>(chunk_mut(dst, i), chunk(src, i), l16, l16, k, scaling, g);
            }
        }
    }
}

/// `fguv` tam blok (`bw = bstep`) sıcak döngüsü: içinde `call` olmadığı için
/// XMM sabitleri yazmaçta kalır.
#[target_feature(enable = "ssse3")]
#[inline(never)]
fn fguv_tam_blok<const SX: usize, const SY: usize, const CSFL: bool>(
    dst: &mut [u8],
    src: &[u8],
    luma: &[u8],
    k: &FguvBaglam,
    scaling: &[u8; OLCEKLEME_BOYU],
    lut: &GrainLut,
    of: &Ofsetler,
    bx: usize,
    ystart: usize,
    bh: usize,
    xstart: usize,
) {
    let bstep = FG_BLOCK_SIZE >> SX;
    if SX == 0 {
        if xstart > 0 {
            for y in ystart..bh {
                let g32 = chunk::<i16, 32>(&lut[of.o.1 + y], of.o.0);
                let sol = iki_ornek(&lut[of.sol.1 + y], of.sol.0 + bstep);
                let g0_lo = yatay_harmanla(sol, load_i16x8(chunk(g32, 0)), 0);
                let g0_hi = load_i16x8(chunk(g32, 8));
                let i = y * k.stride + bx;
                let lx = (y << SY) * k.luma_stride + bx;
                let s32 = chunk::<u8, 32>(src, i);
                let d32 = chunk_mut::<u8, 32>(dst, i);
                let l32 = chunk::<u8, 32>(luma, lx);
                let l0 = chunk(l32, 0);
                let l1 = chunk(l32, 16);
                fguv_parca::<0, CSFL>(
                    chunk_mut(d32, 0),
                    chunk(s32, 0),
                    l0,
                    l0,
                    k,
                    scaling,
                    (g0_lo, g0_hi),
                );
                let g1_lo = load_i16x8(chunk(g32, 16));
                let g1_hi = load_i16x8(chunk(g32, 24));
                fguv_parca::<0, CSFL>(
                    chunk_mut(d32, 16),
                    chunk(s32, 16),
                    l1,
                    l1,
                    k,
                    scaling,
                    (g1_lo, g1_hi),
                );
            }
        } else {
            for y in ystart..bh {
                let g32 = chunk::<i16, 32>(&lut[of.o.1 + y], of.o.0);
                let g0_lo = load_i16x8(chunk(g32, 0));
                let g0_hi = load_i16x8(chunk(g32, 8));
                let i = y * k.stride + bx;
                let lx = (y << SY) * k.luma_stride + bx;
                let s32 = chunk::<u8, 32>(src, i);
                let d32 = chunk_mut::<u8, 32>(dst, i);
                let l32 = chunk::<u8, 32>(luma, lx);
                let l0 = chunk(l32, 0);
                let l1 = chunk(l32, 16);
                fguv_parca::<0, CSFL>(
                    chunk_mut(d32, 0),
                    chunk(s32, 0),
                    l0,
                    l0,
                    k,
                    scaling,
                    (g0_lo, g0_hi),
                );
                let g1_lo = load_i16x8(chunk(g32, 16));
                let g1_hi = load_i16x8(chunk(g32, 24));
                fguv_parca::<0, CSFL>(
                    chunk_mut(d32, 16),
                    chunk(s32, 16),
                    l1,
                    l1,
                    k,
                    scaling,
                    (g1_lo, g1_hi),
                );
            }
        }
    } else if xstart > 0 {
        for y in ystart..bh {
            let g16 = chunk::<i16, 16>(&lut[of.o.1 + y], of.o.0);
            let sol = iki_ornek(&lut[of.sol.1 + y], of.sol.0 + bstep);
            let g_lo = yatay_harmanla(sol, load_i16x8(chunk(g16, 0)), SX);
            let g_hi = load_i16x8(chunk(g16, 8));
            let i = y * k.stride + bx;
            let lx = (y << SY) * k.luma_stride + (bx << 1);
            let l32 = chunk::<u8, 32>(luma, lx);
            fguv_parca::<SX, CSFL>(
                chunk_mut(dst, i),
                chunk(src, i),
                chunk(l32, 0),
                chunk(l32, 16),
                k,
                scaling,
                (g_lo, g_hi),
            );
        }
    } else {
        for y in ystart..bh {
            let g16 = chunk::<i16, 16>(&lut[of.o.1 + y], of.o.0);
            let g_lo = load_i16x8(chunk(g16, 0));
            let g_hi = load_i16x8(chunk(g16, 8));
            let i = y * k.stride + bx;
            let lx = (y << SY) * k.luma_stride + (bx << 1);
            let l32 = chunk::<u8, 32>(luma, lx);
            fguv_parca::<SX, CSFL>(
                chunk_mut(dst, i),
                chunk(src, i),
                chunk(l32, 0),
                chunk(l32, 16),
                k,
                scaling,
                (g_lo, g_hi),
            );
        }
    }
}

#[target_feature(enable = "ssse3")]
#[inline(never)]
fn fguv_govde_ic<const SX: usize, const SY: usize, const CSFL: bool>(
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
    let overlap = data.overlap_flag;
    let rows = 1 + usize::from(overlap && row_num > 0);
    let sinir = if data.clip_to_restricted_range {
        (16, if is_id { 235 } else { 240 })
    } else {
        (0, 255)
    };
    let shift = data.scaling_shift as u32;
    let carpan = _mm_set1_epi16(1 << (15 - shift));
    let (vmin, vmax) = (
        _mm_set1_epi8(sinir.0 as i8),
        _mm_set1_epi8(sinir.1 as u8 as i8),
    );
    let (lm, m, off) = (data.uv_luma_mult[uv], data.uv_mult[uv], data.uv_offset[uv]);
    let carpan_uv = _mm_set1_epi16(((lm & 0xFF) | ((m & 0xFF) << 8)) as i16);
    let ofs = _mm_set1_epi16(off as i16);
    let bstep = FG_BLOCK_SIZE >> SX;
    let mut seed = scalar::row_seeds(data, row_num, rows);
    let mut offsets = [[0i32; 2]; 2];
    let mut bx = 0usize;
    while bx < pw {
        let bw = bstep.min(pw - bx);
        if overlap && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = scalar::get_random_number(8, &mut seed[i]);
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
        if bw < 16 {
            let (ox, oy) = ofset(offsets[0][0], SX, SY);
            for y in 0..bh {
                let lrow = (y << SY) * luma_stride;
                for x in 0..bw {
                    let g = if x < xstart || y < ystart {
                        kenar_grain(lut, &offsets, SX, SY, x, y, xstart, ystart)
                    } else {
                        i32::from(lut[oy + y][ox + x])
                    };
                    let lx = (bx + x) << SX;
                    let mut avg = i32::from(luma[lrow + lx]);
                    if SX != 0 {
                        avg = (avg + i32::from(luma[lrow + lx + 1]) + 1) >> 1;
                    }
                    let i = y * stride + bx + x;
                    let s = i32::from(src[i]);
                    let val = if CSFL {
                        avg
                    } else {
                        (((avg * lm + s * m) >> 6) + off).clamp(0, 255)
                    };
                    dst[i] = skaler_piksel(s, val as usize, g, scaling, shift, sinir);
                }
            }
        } else {
            let of = Ofsetler::yeni(&offsets, SX, SY);
            let k = FguvBaglam {
                stride,
                luma_stride,
                carpan,
                carpan_uv,
                ofs,
                vmin,
                vmax,
            };
            if bw == bstep {
                if ystart > 0 {
                    fguv_kismi_veya_ust::<SX, SY, CSFL>(
                        dst, src, luma, &k, scaling, lut, &offsets, &of, bx, bw, 0, ystart, xstart,
                        ystart,
                    );
                }
                if ystart < bh {
                    fguv_tam_blok::<SX, SY, CSFL>(
                        dst, src, luma, &k, scaling, lut, &of, bx, ystart, bh, xstart,
                    );
                }
            } else {
                fguv_kismi_veya_ust::<SX, SY, CSFL>(
                    dst, src, luma, &k, scaling, lut, &offsets, &of, bx, bw, 0, bh, xstart, ystart,
                );
            }
        }
        bx += bstep;
    }
}

#[target_feature(enable = "ssse3")]
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
    if data.chroma_scaling_from_luma {
        fguv_govde_ic::<SX, SY, true>(
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
        fguv_govde_ic::<SX, SY, false>(
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

/// AR süzgeci (`LAG` ∈ `0..=3`): üst satırların dokunuş çiftleri `_mm_madd_epi16`
/// (`pmaddwd`) ile 32-bit `(acc_lo, acc_hi)` akümülatörlerine toplanır; luma
/// katkısı `_mm_hadd_epi16` (`phaddw`) ile 16-bitte ortalanıp eklenir.
#[target_feature(enable = "ssse3")]
fn ar_suz_lag<const LAG: usize>(
    buf: &mut GrainLut,
    w: usize,
    h: usize,
    katsayi: &[i8],
    ar_shift: u32,
    luma: Option<(&GrainLut, usize, usize, i32)>,
) {
    let son = w - 3;
    let n_ust = LAG * (2 * LAG + 1);
    let mut ust = [0i32; 82];
    for y in 3..h {
        let mut x0 = 3;
        while x0 < son {
            let xb = x0.min(son - 8);
            let mut acc_lo = _mm_setzero_si128();
            let mut acc_hi = _mm_setzero_si128();
            for r_idx in 0..LAG {
                let r = LAG - r_idx;
                let satir = &buf[y - r];
                let base_k = r_idx * (2 * LAG + 1);
                let mut dx = 0;
                while dx < 2 * LAG {
                    let c0 = i32::from(katsayi[base_k + dx]) & 0xFFFF;
                    let c1 = i32::from(katsayi[base_k + dx + 1]) << 16;
                    let cp = _mm_set1_epi32(c0 | c1);
                    let v0 = load_i16x8(chunk(satir, xb + dx - LAG));
                    let v1 = load_i16x8(chunk(satir, xb + dx + 1 - LAG));
                    acc_lo = _mm_add_epi32(acc_lo, _mm_madd_epi16(_mm_unpacklo_epi16(v0, v1), cp));
                    acc_hi = _mm_add_epi32(acc_hi, _mm_madd_epi16(_mm_unpackhi_epi16(v0, v1), cp));
                    dx += 2;
                }
                let cl = _mm_set1_epi32(i32::from(katsayi[base_k + 2 * LAG]) & 0xFFFF);
                let vl = load_i16x8(chunk(satir, xb + LAG));
                acc_lo = _mm_add_epi32(acc_lo, _mm_madd_epi16(_mm_unpacklo_epi16(vl, vl), cl));
                acc_hi = _mm_add_epi32(acc_hi, _mm_madd_epi16(_mm_unpackhi_epi16(vl, vl), cl));
            }
            if let Some((by, subx, suby, c)) = luma {
                let lx = ((xb - 3) << subx) + 3;
                let ly = ((y - 3) << suby) + 3;
                let satir_topla = |r: &[i16; 82]| {
                    if subx != 0 {
                        _mm_hadd_epi16(load_i16x8(chunk(r, lx)), load_i16x8(chunk(r, lx + 8)))
                    } else {
                        load_i16x8(chunk(r, lx))
                    }
                };
                let mut l = satir_topla(&by[ly]);
                if suby != 0 {
                    l = _mm_add_epi16(l, satir_topla(&by[ly + 1]));
                }
                let l = match subx + suby {
                    1 => _mm_srai_epi16::<1>(_mm_add_epi16(l, _mm_set1_epi16(1))),
                    2 => _mm_srai_epi16::<2>(_mm_add_epi16(l, _mm_set1_epi16(2))),
                    _ => l,
                };
                let cp = _mm_set1_epi32(c & 0xFFFF);
                acc_lo = _mm_add_epi32(acc_lo, _mm_madd_epi16(_mm_unpacklo_epi16(l, l), cp));
                acc_hi = _mm_add_epi32(acc_hi, _mm_madd_epi16(_mm_unpackhi_epi16(l, l), cp));
            }
            let lo01 = _mm_cvtsi128_si64(acc_lo) as u64;
            let lo23 = _mm_cvtsi128_si64(_mm_srli_si128::<8>(acc_lo)) as u64;
            let hi01 = _mm_cvtsi128_si64(acc_hi) as u64;
            let hi23 = _mm_cvtsi128_si64(_mm_srli_si128::<8>(acc_hi)) as u64;
            let u = chunk_mut::<i32, 8>(&mut ust, xb);
            u[0] = lo01 as i32;
            u[1] = (lo01 >> 32) as i32;
            u[2] = lo23 as i32;
            u[3] = (lo23 >> 32) as i32;
            u[4] = hi01 as i32;
            u[5] = (hi01 >> 32) as i32;
            u[6] = hi23 as i32;
            u[7] = (hi23 >> 32) as i32;
            x0 = xb + 8;
        }
        let yuvarla = (1i32 << ar_shift) >> 1;
        let satir = &mut buf[y];
        for x in 3..son {
            let mut sum = ust[x];
            for k in 1..=LAG {
                sum += i32::from(katsayi[n_ust + LAG - k]) * i32::from(satir[x - k]);
            }
            let g = i32::from(satir[x]) + ((sum + yuvarla) >> ar_shift);
            satir[x] = g.clamp(GRAIN_MIN, GRAIN_MAX) as i16;
        }
    }
}

/// `lag == 0` kroma luma-katkılı yol: yatay/dikey geri-bağlantı olmadığı için
/// 8 şerit doğrudan 16-bit `_mm_mullo_epi16` + `_mm_mulhrs_epi16` ile güncellenir.
#[target_feature(enable = "ssse3")]
fn ar_suz_lag0(
    buf: &mut GrainLut,
    w: usize,
    h: usize,
    ar_shift: u32,
    (by, subx, suby, c): (&GrainLut, usize, usize, i32),
) {
    let son = w - 3;
    let ck = _mm_set1_epi16(c as i16);
    let olcek = _mm_set1_epi16(1 << (15 - ar_shift));
    let vmin = _mm_set1_epi16(GRAIN_MIN as i16);
    let vmax = _mm_set1_epi16(GRAIN_MAX as i16);
    for y in 3..h {
        let ly = ((y - 3) << suby) + 3;
        let son8 = load_i16x8(chunk(&buf[y], son - 8));
        let mut x0 = 3;
        while x0 < son {
            let (xb, orig) = if x0 + 8 <= son {
                (x0, load_i16x8(chunk(&buf[y], x0)))
            } else {
                (son - 8, son8)
            };
            let lx = ((xb - 3) << subx) + 3;
            let satir_topla = |r: &[i16; 82]| {
                if subx != 0 {
                    _mm_hadd_epi16(load_i16x8(chunk(r, lx)), load_i16x8(chunk(r, lx + 8)))
                } else {
                    load_i16x8(chunk(r, lx))
                }
            };
            let mut l = satir_topla(&by[ly]);
            if suby != 0 {
                l = _mm_add_epi16(l, satir_topla(&by[ly + 1]));
            }
            let l = match subx + suby {
                1 => _mm_srai_epi16::<1>(_mm_add_epi16(l, _mm_set1_epi16(1))),
                2 => _mm_srai_epi16::<2>(_mm_add_epi16(l, _mm_set1_epi16(2))),
                _ => l,
            };
            let d = _mm_mulhrs_epi16(_mm_mullo_epi16(l, ck), olcek);
            let g = _mm_max_epi16(_mm_min_epi16(_mm_add_epi16(orig, d), vmax), vmin);
            store_i16x8(chunk_mut(&mut buf[y], xb), g);
            x0 = xb + 8;
        }
    }
}

#[target_feature(enable = "ssse3")]
fn ar_suz(
    buf: &mut GrainLut,
    w: usize,
    h: usize,
    lag: usize,
    katsayi: &[i8],
    ar_shift: u32,
    luma: Option<(&GrainLut, usize, usize, i32)>,
) {
    match lag {
        0 => {
            if let Some(l) = luma {
                ar_suz_lag0(buf, w, h, ar_shift, l);
            }
        }
        1 => ar_suz_lag::<1>(buf, w, h, katsayi, ar_shift, luma),
        2 => ar_suz_lag::<2>(buf, w, h, katsayi, ar_shift, luma),
        _ => ar_suz_lag::<3>(buf, w, h, katsayi, ar_shift, luma),
    }
}

#[target_feature(enable = "ssse3")]
fn grain_y_govde(buf: &mut GrainLut, data: &FilmGrainData) {
    let shift = 4 + data.grain_scale_shift as u32;
    gauss_doldur(buf, data.seed, GRAIN_WIDTH, GRAIN_HEIGHT, shift);
    let lag = data.ar_coeff_lag as usize;
    if lag > 0 {
        ar_suz(
            buf,
            GRAIN_WIDTH,
            GRAIN_HEIGHT,
            lag,
            &data.ar_coeffs_y,
            data.ar_coeff_shift as u32,
            None,
        );
    }
}

#[target_feature(enable = "ssse3")]
fn grain_uv_govde<const SX: usize, const SY: usize>(
    buf: &mut GrainLut,
    buf_y: &GrainLut,
    data: &FilmGrainData,
    uv: usize,
) {
    let (w, h) = (
        if SX != 0 {
            SUB_GRAIN_WIDTH
        } else {
            GRAIN_WIDTH
        },
        if SY != 0 {
            SUB_GRAIN_HEIGHT
        } else {
            GRAIN_HEIGHT
        },
    );
    let seed = data.seed ^ if uv != 0 { 0x49d8 } else { 0xb524 };
    gauss_doldur(buf, seed, w, h, 4 + data.grain_scale_shift as u32);
    let lag = data.ar_coeff_lag as usize;
    let katsayi = &data.ar_coeffs_uv[uv];
    let luma = (data.num_y_points != 0)
        .then(|| (buf_y, SX, SY, i32::from(katsayi[lag * (2 * lag + 1) + lag])));
    if lag > 0 || luma.is_some() {
        ar_suz(buf, w, h, lag, katsayi, data.ar_coeff_shift as u32, luma);
    }
}

fn grain_y_giris(buf: &mut GrainLut, data: &FilmGrainData, bd: FgBitDepth) {
    if bd != FgBitDepth::B8 || !grain_sozlesmesi(data) {
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
    unsafe { grain_y_govde(buf, data) }
}

macro_rules! grain_uv_giris {
    ($ad:ident, $sx:literal, $sy:literal, $skaler:ident) => {
        fn $ad(
            buf: &mut GrainLut,
            buf_y: &GrainLut,
            data: &FilmGrainData,
            uv: usize,
            bd: FgBitDepth,
        ) {
            if bd != FgBitDepth::B8 || uv >= 2 || !grain_sozlesmesi(data) {
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
            unsafe { grain_uv_govde::<$sx, $sy>(buf, buf_y, data, uv) }
        }
    };
}

grain_uv_giris!(grain_uv_420_giris, 1, 1, generate_grain_uv_420);
grain_uv_giris!(grain_uv_422_giris, 1, 0, generate_grain_uv_422);
grain_uv_giris!(grain_uv_444_giris, 0, 0, generate_grain_uv_444);

/// `generate_grain_y` 8 bpc SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn generate_grain_y_fn(_ssse3: Ssse3Token) -> GenerateGrainYFn {
    grain_y_giris
}

/// `generate_grain_uv[düzen]` 8 bpc SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn generate_grain_uv_fn(_ssse3: Ssse3Token, duzen: usize) -> GenerateGrainUvFn {
    match duzen {
        0 => grain_uv_420_giris,
        1 => grain_uv_422_giris,
        _ => grain_uv_444_giris,
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`fgy_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `fgy_sozlesmesi` `src`/`dst`'nin `(bh−1)·stride + pw` baytı kapsadığını ve
    // `stride ≥ pw`'yi, `first_chunk` tablonun 256 baytını garanti eder; bütün
    // erişimler denetimli dilim/dizi erişimidir. (3) hizalama/takma ad: `dst`
    // tek `&mut`, diğerleri `&`; ödünç kuralları çakışmayı dışlar.
    unsafe { fgy_govde(dst, src, stride, data, pw, tablo, grain_lut, bh, row_num) }
}

/// Kroma ölçekleme katsayılarının 8/9 bitlik AV1 sözleşme aralığında olup olmadığı.
pub(super) fn uv_katsayi_sozlesmesi(data: &FilmGrainData, uv: usize) -> bool {
    data.chroma_scaling_from_luma
        || ((-128..=127).contains(&data.uv_mult[uv])
            && (-128..=127).contains(&data.uv_luma_mult[uv])
            && (-256..=255).contains(&data.uv_offset[uv]))
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
            let tablo = scaling.first_chunk::<OLCEKLEME_BOYU>();
            let Some(tablo) = tablo.filter(|_| {
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
                        $sx,
                        $sy,
                        data,
                    )
                    && uv_katsayi_sozlesmesi(data, uv)
            }) else {
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
            };
            // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
            // (`fguv_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır. (2) sınır:
            // `fguv_sozlesmesi` `src`/`dst`'nin `(bh−1)·stride + pw` baytı ve lumanın
            // `((bh−1) << sy)·luma_stride + ((pw−1) << sx) + sx + 1` baytı kapsadığını,
            // `first_chunk` tablonun 256 baytını garanti eder; erişimler denetimlidir.
            // (3) hizalama/takma ad: `dst` tek `&mut`, diğerleri `&`.
            unsafe {
                fguv_govde::<$sx, $sy>(
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
    };
}

fguv_giris!(fguv_420_giris, 1, 1, fguv_32x32xn_420);
fguv_giris!(fguv_422_giris, 1, 0, fguv_32x32xn_422);
fguv_giris!(fguv_444_giris, 0, 0, fguv_32x32xn_444);

/// `fgy_32x32xn` 8 bpc SSSE3 yuvası (yalnız belirteçle).
#[must_use]
pub fn fgy_fn(_ssse3: Ssse3Token) -> FgyFn {
    fgy_giris
}

/// `fguv_32x32xn[düzen]` 8 bpc SSSE3 yuvası (yalnız belirteçle; düzen dizini
/// 0 = 420, 1 = 422, 2 = 444).
#[must_use]
pub fn fguv_fn(_ssse3: Ssse3Token, duzen: usize) -> FguvFn {
    match duzen {
        0 => fguv_420_giris,
        1 => fguv_422_giris,
        _ => fguv_444_giris,
    }
}
