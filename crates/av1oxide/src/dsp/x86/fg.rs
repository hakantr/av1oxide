//! Film grain 8 bpc AVX2 çekirdekleri — F4-FG a (P39).
//!
//! ## Sözleşme (kopyasız)
//!
//! Çekirdekler çağrı başına yerel kopya yapmaz; `src`, `dst` ve luma
//! dilimlerine doğrudan erişir ve yalnız sözleşme bölgesini okur/yazar
//! (`fgy`: satır `0..bh` × sütun `0..pw`; `fguv` luma: satır `y << sy`, sütun
//! `0 ..= ((pw − 1) << sx) + sx`). Girişte O(1) denetim
//! ([`crate::dsp::fg::fgy_sozlesmesi`], [`crate::dsp::fg::fguv_sozlesmesi`]) ve
//! ölçekleme tablosunun en az [`OLCEKLEME_TOPLAMA`] bayt olması (çözme hattı
//! `SCALING_SIZE_MAX` = 4096 bayt verir) aranır; sözleşme dışı girdi skaler
//! gövdeye devreder ve sayılır ([`crate::dsp::fg::fg_skaler_devri_say`]).
//! Grain LUT'u sabit boyutlu türdür (`[[i16; 82]; 73]`).
//!
//! ## Yöntem
//!
//! Satır 16 piksellik parçalarla işlenir: kaynak baytları `u16`'ya
//! genişletilir; ölçekleme tablosu `vpgatherdd` ile okunur
//! ([`gather_u8_256`]; dav1d `filmgrain_avx2.asm` de toplama kullanır);
//! gürültü `pmullw` + `pmulhrsw` ile `round2(scaling · grain, scaling_shift)`
//! (çarpan `1 << (15 − scaling_shift)`; çarpım `|scaling · grain| ≤ 32 640`
//! `i16`'ya sığar), sonuç `packuswb` + `pmaxub`/`pminub` ile kırpılır. Tam blok
//! (32 piksel; kroma 4:2:x'te 16) iki (bir) parçadır; `16 ≤ bw < 32` kısmi blokta
//! ikinci parça `bw − 16`'dan **örtüşerek** işlenir (aynı pikseller aynı değeri
//! alır), `bw < 16` blok skaler piksel yoluyla işlenir. Örtüşmenin üst iki satırı
//! (dikey) vektörle (`{27, 17}`/`{17, 27}`, kromada `{23, 22}`; `round2(·, 5)`
//! `pmulhrsw` çarpanı 1024), sol iki sütunu (yatay) şerit ağırlık vektörüyle
//! (`yatay_harmanla`; köşe skaler gövdeyle aynı sırayla) harmanlanır.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments)]
// Ofset/tohum döngüleri skaler gövdenin (dav1d C) döngülerini birebir izler.
#![allow(clippy::needless_range_loop)]

use super::mem::{
    OLCEKLEME_TOPLAMA, chunk, chunk_mut, gather_u8_256, load_i16x8, load_i16x16, load_u8x16,
    load_u8x32, olcekleme16, olcekleme16_u16, olcekleme32, store_i32x8, store_u8x16, store_u8x32,
};
use crate::bitstream::frame_header::FilmGrainData;
use crate::cpu::Avx2Token;
use crate::dsp::fg::{
    FG_BLOCK_SIZE, FgBitDepth, FguvFn, FgyFn, GRAIN_HEIGHT, GRAIN_WIDTH, GenerateGrainUvFn,
    GenerateGrainYFn, GrainLut, SUB_GRAIN_HEIGHT, SUB_GRAIN_WIDTH, fg_skaler_devri_say,
    fguv_sozlesmesi, fgy_sozlesmesi,
};
use crate::dsp::fg_tables::GAUSSIAN_SEQUENCE;
use crate::dsp::scalar::fg as scalar;
use core::arch::x86_64::{
    __m128i, __m256i, _mm_max_epu8, _mm_min_epu8, _mm_packus_epi16, _mm_set1_epi8,
    _mm256_add_epi16, _mm256_add_epi32, _mm256_broadcastsi128_si256, _mm256_castsi256_si128,
    _mm256_cvtepi16_epi32, _mm256_cvtepu8_epi16, _mm256_extracti128_si256, _mm256_insert_epi16,
    _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_max_epi16, _mm256_max_epi32, _mm256_max_epu8,
    _mm256_min_epi16, _mm256_min_epi32, _mm256_min_epu8, _mm256_mulhrs_epi16, _mm256_mullo_epi16,
    _mm256_mullo_epi32, _mm256_packus_epi16, _mm256_packus_epi32, _mm256_permute2x128_si256,
    _mm256_set1_epi8, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi16,
    _mm256_setzero_si256, _mm256_srai_epi32, _mm256_srav_epi32, _mm256_srli_epi16,
    _mm256_unpackhi_epi8, _mm256_unpackhi_epi16, _mm256_unpacklo_epi8, _mm256_unpacklo_epi16,
};

/// Örtüşme ağırlıkları `[alt örnekleme][konum] = [eski, yeni]` (skaler
/// gövdelerin `W` tablosu).
pub(super) const W: [[[i32; 2]; 2]; 2] = [[[27, 17], [17, 27]], [[23, 22], [0, 0]]];

/// 8 bitte grain aralığı `[−128, 127]`.
pub(super) const GRAIN_MIN: i32 = -128;
pub(super) const GRAIN_MAX: i32 = 127;

/// Bloğun LUT ofseti `(offx, offy)` (`sample_lut`, `src/filmgrain_tmpl.c:158-167`).
pub(super) const fn ofset(randval: i32, sx: usize, sy: usize) -> (usize, usize) {
    (
        3 + (2 >> sx) * (3 + (randval >> 4)) as usize,
        3 + (2 >> sy) * (3 + (randval & 0xF)) as usize,
    )
}

/// Örtüşme bölgesindeki (`x < xstart` ya da `y < ystart`) pikselin grain'i —
/// skaler gövdenin formülü ve sırası (yatay, dikey, köşe).
pub(super) fn kenar_grain(
    lut: &GrainLut,
    offsets: &[[i32; 2]; 2],
    sx: usize,
    sy: usize,
    x: usize,
    y: usize,
    xstart: usize,
    ystart: usize,
) -> i32 {
    let s = |bx, by| scalar::sample_lut(lut, offsets, sx, sy, bx, by, x, y);
    let harman = |eski: i32, yeni: i32, w: [i32; 2]| {
        scalar::round2(eski * w[0] + yeni * w[1], 5).clamp(GRAIN_MIN, GRAIN_MAX)
    };
    if y >= ystart {
        harman(s(1, 0), s(0, 0), W[sx][x])
    } else if x >= xstart {
        harman(s(0, 1), s(0, 0), W[sy][y])
    } else {
        let top = harman(s(1, 1), s(0, 1), W[sx][x]);
        let grain = harman(s(1, 0), s(0, 0), W[sx][x]);
        harman(top, grain, W[sy][y])
    }
}

/// Gürültü ve kırpma (skaler `add_noise`): `clamp(s + round2(scaling[val] ·
/// grain, shift), min, max)`.
#[inline]
pub(super) fn skaler_piksel(
    s: i32,
    val: usize,
    grain: i32,
    scaling: &[u8],
    shift: u32,
    sinir: (i32, i32),
) -> u8 {
    let noise = scalar::round2(i32::from(scaling[val]) * grain, shift);
    (s + noise).clamp(sinir.0, sinir.1) as u8
}

/// `clamp(round2(eski · w0 + yeni · w1, 5), −128, 127)` (16 şerit; `|toplam| ≤
/// 128 · 45` `i16`'ya sığar, `pmulhrsw(x, 1024)` = `(x + 16) >> 5`).
#[target_feature(enable = "avx2")]
#[inline]
fn harmanla(eski: __m256i, yeni: __m256i, w: [i32; 2]) -> __m256i {
    let x = _mm256_add_epi16(
        _mm256_mullo_epi16(eski, _mm256_set1_epi16(w[0] as i16)),
        _mm256_mullo_epi16(yeni, _mm256_set1_epi16(w[1] as i16)),
    );
    let r = _mm256_mulhrs_epi16(x, _mm256_set1_epi16(1 << 10));
    _mm256_max_epi16(
        _mm256_min_epi16(r, _mm256_set1_epi16(GRAIN_MAX as i16)),
        _mm256_set1_epi16(GRAIN_MIN as i16),
    )
}

/// Yatay örtüşme: `x < 2 >> sx` şeritleri `clamp(round2(eski · W[sx][x][0] +
/// yeni · W[sx][x][1], 5))`, diğer şeritler ağırlık `(0, 32)` ile değişmez
/// (`round2(32 · g, 5) = g`, kırpma etkisiz). `eski` yalnız ilk iki şeritte
/// anlamlıdır.
#[target_feature(enable = "avx2")]
#[inline]
fn yatay_harmanla(eski: __m256i, yeni: __m256i, sx: usize) -> __m256i {
    let (w0, w1) = if sx == 0 {
        (
            _mm256_setr_epi16(27, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            _mm256_setr_epi16(
                17, 27, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
            ),
        )
    } else {
        (
            _mm256_setr_epi16(23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            _mm256_setr_epi16(
                22, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
            ),
        )
    };
    let x = _mm256_add_epi16(_mm256_mullo_epi16(eski, w0), _mm256_mullo_epi16(yeni, w1));
    let r = _mm256_mulhrs_epi16(x, _mm256_set1_epi16(1 << 10));
    _mm256_max_epi16(
        _mm256_min_epi16(r, _mm256_set1_epi16(GRAIN_MAX as i16)),
        _mm256_set1_epi16(GRAIN_MIN as i16),
    )
}

/// LUT satırının `x` ve `x + 1` örnekleri ilk iki şeritte (yatay örtüşmenin
/// "eski" bloğu; diğer şeritler sıfır).
#[target_feature(enable = "avx2")]
#[inline]
fn iki_ornek(satir: &[i16], x: usize) -> __m256i {
    let a = chunk::<i16, 2>(satir, x);
    _mm256_setr_epi16(a[0], a[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
}

/// `g`'nin `serit` (0 ya da 1) şeridine `v`.
#[target_feature(enable = "avx2")]
#[inline]
fn serit_yaz(g: __m256i, serit: usize, v: i32) -> __m256i {
    if serit == 0 {
        _mm256_insert_epi16::<0>(g, v as i16)
    } else {
        _mm256_insert_epi16::<1>(g, v as i16)
    }
}

/// `fgy` 16 piksel: `dst[i..i+16]` ← `clamp(src + round2(scaling[src] · g,
/// shift), min, max)`.
#[target_feature(enable = "avx2")]
#[inline]
fn fgy_parca(
    dst: &mut [u8],
    src: &[u8],
    i: usize,
    g: __m256i,
    scaling: &[u8; OLCEKLEME_TOPLAMA],
    carpan: __m256i,
    vmin: __m128i,
    vmax: __m128i,
) {
    let (s16, sc) = olcekleme16(scaling, load_u8x16(chunk(src, i)));
    let n = _mm256_mulhrs_epi16(_mm256_mullo_epi16(sc, g), carpan);
    let r = _mm256_add_epi16(s16, n);
    let b = _mm_packus_epi16(_mm256_castsi256_si128(r), _mm256_extracti128_si256::<1>(r));
    store_u8x16(chunk_mut(dst, i), _mm_max_epu8(_mm_min_epu8(b, vmax), vmin));
}

/// `fgy` tam blok (32 piksel): `g0` pikseller 0–15, `g1` 16–31 grain'i. Kaynak
/// `punpck{l,h}bw` sırasında (0–7/16–23, 8–15/24–31) işlenir; grain aynı sıraya
/// `vperm2i128` ile getirilir, `packuswb` sonucu doğal sıraya döndürür.
#[target_feature(enable = "avx2")]
#[inline]
fn fgy_parca32(
    dst: &mut [u8],
    src: &[u8],
    i: usize,
    g0: __m256i,
    g1: __m256i,
    scaling: &[u8; OLCEKLEME_TOPLAMA],
    carpan: __m256i,
    vmin: __m256i,
    vmax: __m256i,
) {
    let (lo, hi, sl, sh) = olcekleme32(scaling, load_u8x32(chunk(src, i)));
    let glo = _mm256_permute2x128_si256::<0x20>(g0, g1);
    let ghi = _mm256_permute2x128_si256::<0x31>(g0, g1);
    let rlo = _mm256_add_epi16(lo, _mm256_mulhrs_epi16(_mm256_mullo_epi16(sl, glo), carpan));
    let rhi = _mm256_add_epi16(hi, _mm256_mulhrs_epi16(_mm256_mullo_epi16(sh, ghi), carpan));
    let b = _mm256_packus_epi16(rlo, rhi);
    store_u8x32(
        chunk_mut(dst, i),
        _mm256_max_epu8(_mm256_min_epu8(b, vmax), vmin),
    );
}

/// Blok parçalarının başlangıç sütunları: tam blok `[0, 16]` (`bw = 32`), kısmi
/// `16 ≤ bw < 32` `[0, bw − 16]`, `bw = 16` `[0]`.
pub(super) fn parcalar(bw: usize) -> ([usize; 2], usize) {
    match bw {
        16 => ([0, 0], 1),
        _ => ([0, bw - 16], 2),
    }
}

/// Bloğun dört LUT ofseti (`sample_lut`'un `(bx, by)` = (0,0), (0,1), (1,0),
/// (1,1); geçersiz olanlar kullanılmaz) — blok başına bir kez.
#[derive(Clone, Copy)]
pub(super) struct Ofsetler {
    pub(super) o: (usize, usize),
    pub(super) ust: (usize, usize),
    pub(super) sol: (usize, usize),
    pub(super) kose: (usize, usize),
}

impl Ofsetler {
    pub(super) fn yeni(offsets: &[[i32; 2]; 2], sx: usize, sy: usize) -> Self {
        Self {
            o: ofset(offsets[0][0], sx, sy),
            ust: ofset(offsets[0][1], sx, sy),
            sol: ofset(offsets[1][0], sx, sy),
            kose: ofset(offsets[1][1], sx, sy),
        }
    }
}

/// Satırın grain parçası (skaler gövdenin sırası): yatay örtüşmede sol blokla
/// harman (`x0 = 0`), dikey örtüşmede üst blok (köşede o da yatay harmanlı)
/// ile harman. `0 < x0 < xstart` (yalnız `bw = 17` kısmi blokta) şeritleri
/// skaler formülle yazılır.
#[target_feature(enable = "avx2")]
#[inline]
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
) -> __m256i {
    let bh_k = FG_BLOCK_SIZE >> sy;
    let bw_k = FG_BLOCK_SIZE >> sx;
    let mut g = load_i16x16(chunk(&lut[of.o.1 + y], of.o.0 + x0));
    let yatay = x0 == 0 && xstart > 0;
    if yatay {
        g = yatay_harmanla(iki_ornek(&lut[of.sol.1 + y], of.sol.0 + bw_k), g, sx);
    }
    if y < ystart {
        let mut t = load_i16x16(chunk(&lut[of.ust.1 + y + bh_k], of.ust.0 + x0));
        if yatay {
            t = yatay_harmanla(
                iki_ornek(&lut[of.kose.1 + y + bh_k], of.kose.0 + bw_k),
                t,
                sx,
            );
        }
        g = harmanla(t, g, W[sy][y]);
    }
    if x0 > 0 {
        for x in x0..xstart.min(x0 + 16) {
            g = serit_yaz(
                g,
                x - x0,
                kenar_grain(lut, offsets, sx, sy, x, y, xstart, ystart),
            );
        }
    }
    g
}

#[target_feature(enable = "avx2")]
fn fgy_govde(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8; OLCEKLEME_TOPLAMA],
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
    let carpan = _mm256_set1_epi16(1 << (15 - shift));
    let (vmin, vmax) = (
        _mm_set1_epi8(sinir.0 as i8),
        _mm_set1_epi8(sinir.1 as u8 as i8),
    );
    let (vmin32, vmax32) = (
        _mm256_set1_epi8(sinir.0 as i8),
        _mm256_set1_epi8(sinir.1 as u8 as i8),
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
            // Dar kısmi blok: skaler piksel yolu.
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
            let (bas, n) = parcalar(bw);
            let of = Ofsetler::yeni(&offsets, 0, 0);
            // Dikey örtüşme satırları (`y < ystart`) ve `bw = 17`'nin örtüşen
            // parçası ayrı döngüde (`grain_parca` çağrısı): kalan satırların
            // döngüsü çağrısız kalır (vektör sabitleri yazmaçta).
            for y in 0..ystart {
                for &x0 in &bas[..n] {
                    let g = grain_parca(lut, &offsets, &of, 0, 0, x0, y, xstart, ystart);
                    fgy_parca(
                        dst,
                        src,
                        y * stride + bx + x0,
                        g,
                        scaling,
                        carpan,
                        vmin,
                        vmax,
                    );
                }
            }
            let sol = |y: usize| iki_ornek(&lut[of.sol.1 + y], of.sol.0 + FG_BLOCK_SIZE);
            if bw == FG_BLOCK_SIZE {
                for y in ystart..bh {
                    let lut_satir = &lut[of.o.1 + y];
                    let mut g0 = load_i16x16(chunk(lut_satir, of.o.0));
                    if xstart > 0 {
                        g0 = yatay_harmanla(sol(y), g0, 0);
                    }
                    let g1 = load_i16x16(chunk(lut_satir, of.o.0 + 16));
                    fgy_parca32(
                        dst,
                        src,
                        y * stride + bx,
                        g0,
                        g1,
                        scaling,
                        carpan,
                        vmin32,
                        vmax32,
                    );
                }
            } else {
                for y in ystart..bh {
                    let lut_satir = &lut[of.o.1 + y];
                    for &x0 in &bas[..n] {
                        let g = if x0 > 0 && x0 < xstart {
                            grain_parca(lut, &offsets, &of, 0, 0, x0, y, xstart, ystart)
                        } else {
                            let g = load_i16x16(chunk(lut_satir, of.o.0 + x0));
                            if x0 == 0 && xstart > 0 {
                                yatay_harmanla(sol(y), g, 0)
                            } else {
                                g
                            }
                        };
                        fgy_parca(
                            dst,
                            src,
                            y * stride + bx + x0,
                            g,
                            scaling,
                            carpan,
                            vmin,
                            vmax,
                        );
                    }
                }
            }
        }
        bx += FG_BLOCK_SIZE;
    }
}

/// `fguv` parçalarının değişmeyen bağlamı.
struct FguvBaglam {
    stride: usize,
    luma_stride: usize,
    csfl: bool,
    carpan: __m256i,
    carpan_uv: __m256i,
    ofs: __m256i,
    vmin: __m128i,
    vmax: __m128i,
}

/// `fguv` 16 kroma pikseli (sütun `x`, satır `y`): luma ortalaması, ölçekleme
/// dizini (`csfl` ise ortalama, değilse `clamp(((avg · lm + src · m) >> 6) +
/// off, 0, 255)`), gürültü ve kırpma.
#[target_feature(enable = "avx2")]
#[inline]
fn fguv_parca<const SX: usize, const SY: usize>(
    dst: &mut [u8],
    src: &[u8],
    luma: &[u8],
    k: &FguvBaglam,
    scaling: &[u8; OLCEKLEME_TOPLAMA],
    x: usize,
    y: usize,
    g: __m256i,
) {
    let lx = (y << SY) * k.luma_stride + (x << SX);
    let avg = if SX != 0 {
        let toplam = _mm256_maddubs_epi16(load_u8x32(chunk(luma, lx)), _mm256_set1_epi8(1));
        _mm256_srli_epi16::<1>(_mm256_add_epi16(toplam, _mm256_set1_epi16(1)))
    } else {
        _mm256_cvtepu8_epi16(load_u8x16(chunk(luma, lx)))
    };
    let i = y * k.stride + x;
    let s16 = _mm256_cvtepu8_epi16(load_u8x16(chunk(src, i)));
    let sc = if k.csfl {
        olcekleme16_u16(scaling, avg)
    } else {
        let birlesik = |p: __m256i| {
            let v = _mm256_add_epi32(
                _mm256_srai_epi32::<6>(_mm256_madd_epi16(p, k.carpan_uv)),
                k.ofs,
            );
            _mm256_min_epi32(
                _mm256_max_epi32(v, _mm256_setzero_si256()),
                _mm256_set1_epi32(255),
            )
        };
        let (lo, hi) = (
            birlesik(_mm256_unpacklo_epi16(avg, s16)),
            birlesik(_mm256_unpackhi_epi16(avg, s16)),
        );
        _mm256_packus_epi32(gather_u8_256(scaling, lo), gather_u8_256(scaling, hi))
    };
    let n = _mm256_mulhrs_epi16(_mm256_mullo_epi16(sc, g), k.carpan);
    let r = _mm256_add_epi16(s16, n);
    let b = _mm_packus_epi16(_mm256_castsi256_si128(r), _mm256_extracti128_si256::<1>(r));
    store_u8x16(
        chunk_mut(dst, i),
        _mm_max_epu8(_mm_min_epu8(b, k.vmax), k.vmin),
    );
}

/// `fguv` 4:4:4 tam blok (32 piksel; `g0` pikseller 0–15, `g1` 16–31): luma,
/// kaynak ve grain `punpck{l,h}bw` sırasında ([`fgy_parca32`] gibi).
#[target_feature(enable = "avx2")]
#[inline]
fn fguv444_parca32(
    dst: &mut [u8],
    src: &[u8],
    luma: &[u8],
    k: &FguvBaglam,
    scaling: &[u8; OLCEKLEME_TOPLAMA],
    x: usize,
    y: usize,
    g0: __m256i,
    g1: __m256i,
) {
    let z = _mm256_setzero_si256();
    let l = load_u8x32(chunk(luma, y * k.luma_stride + x));
    let i = y * k.stride + x;
    let s = load_u8x32(chunk(src, i));
    let (slo, shi) = (_mm256_unpacklo_epi8(s, z), _mm256_unpackhi_epi8(s, z));
    let (sl, sh) = if k.csfl {
        let (_, _, sl, sh) = olcekleme32(scaling, l);
        (sl, sh)
    } else {
        let (llo, lhi) = (_mm256_unpacklo_epi8(l, z), _mm256_unpackhi_epi8(l, z));
        let birlesik = |p: __m256i| {
            let v = _mm256_add_epi32(
                _mm256_srai_epi32::<6>(_mm256_madd_epi16(p, k.carpan_uv)),
                k.ofs,
            );
            gather_u8_256(
                scaling,
                _mm256_min_epi32(_mm256_max_epi32(v, z), _mm256_set1_epi32(255)),
            )
        };
        let yarim = |a: __m256i, b: __m256i| {
            _mm256_packus_epi32(
                birlesik(_mm256_unpacklo_epi16(a, b)),
                birlesik(_mm256_unpackhi_epi16(a, b)),
            )
        };
        (yarim(llo, slo), yarim(lhi, shi))
    };
    let glo = _mm256_permute2x128_si256::<0x20>(g0, g1);
    let ghi = _mm256_permute2x128_si256::<0x31>(g0, g1);
    let rlo = _mm256_add_epi16(
        slo,
        _mm256_mulhrs_epi16(_mm256_mullo_epi16(sl, glo), k.carpan),
    );
    let rhi = _mm256_add_epi16(
        shi,
        _mm256_mulhrs_epi16(_mm256_mullo_epi16(sh, ghi), k.carpan),
    );
    let b = _mm256_packus_epi16(rlo, rhi);
    let (vmin, vmax) = (
        _mm256_broadcastsi128_si256(k.vmin),
        _mm256_broadcastsi128_si256(k.vmax),
    );
    store_u8x32(
        chunk_mut(dst, i),
        _mm256_max_epu8(_mm256_min_epu8(b, vmax), vmin),
    );
}

#[target_feature(enable = "avx2")]
fn fguv_govde<const SX: usize, const SY: usize>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8; OLCEKLEME_TOPLAMA],
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
    let carpan = _mm256_set1_epi16(1 << (15 - shift));
    let (vmin, vmax) = (
        _mm_set1_epi8(sinir.0 as i8),
        _mm_set1_epi8(sinir.1 as u8 as i8),
    );
    let csfl = data.chroma_scaling_from_luma;
    let (lm, m, off) = (data.uv_luma_mult[uv], data.uv_mult[uv], data.uv_offset[uv]);
    // `madd` çarpanı: `[avg, src]` çiftleri × `[uv_luma_mult, uv_mult]`.
    let carpan_uv = _mm256_set1_epi32((lm & 0xFFFF) | (m << 16));
    let ofs = _mm256_set1_epi32(off);
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
            // Dar kısmi blok: skaler piksel yolu (skaler `add_noise`).
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
                    let val = if csfl {
                        avg
                    } else {
                        (((avg * lm + s * m) >> 6) + off).clamp(0, 255)
                    };
                    dst[i] = skaler_piksel(s, val as usize, g, scaling, shift, sinir);
                }
            }
        } else {
            let (bas, n) = parcalar(bw);
            let of = Ofsetler::yeni(&offsets, SX, SY);
            let k = FguvBaglam {
                stride,
                luma_stride,
                csfl,
                carpan,
                carpan_uv,
                ofs,
                vmin,
                vmax,
            };
            // Dikey örtüşme satırları ve `bw = 17`'nin örtüşen parçası ayrı
            // döngüde (`grain_parca` çağrısı); kalan döngü çağrısızdır.
            for y in 0..ystart {
                for &x0 in &bas[..n] {
                    let g = grain_parca(lut, &offsets, &of, SX, SY, x0, y, xstart, ystart);
                    fguv_parca::<SX, SY>(dst, src, luma, &k, scaling, bx + x0, y, g);
                }
            }
            let sol = |y: usize| iki_ornek(&lut[of.sol.1 + y], of.sol.0 + bstep);
            if SX == 0 && bw == FG_BLOCK_SIZE {
                for y in ystart..bh {
                    let lut_satir = &lut[of.o.1 + y];
                    let mut g0 = load_i16x16(chunk(lut_satir, of.o.0));
                    if xstart > 0 {
                        g0 = yatay_harmanla(sol(y), g0, 0);
                    }
                    let g1 = load_i16x16(chunk(lut_satir, of.o.0 + 16));
                    fguv444_parca32(dst, src, luma, &k, scaling, bx, y, g0, g1);
                }
                bx += bstep;
                continue;
            }
            for y in ystart..bh {
                let lut_satir = &lut[of.o.1 + y];
                for &x0 in &bas[..n] {
                    let g = if x0 > 0 && x0 < xstart {
                        grain_parca(lut, &offsets, &of, SX, SY, x0, y, xstart, ystart)
                    } else {
                        let g = load_i16x16(chunk(lut_satir, of.o.0 + x0));
                        if x0 == 0 && xstart > 0 {
                            yatay_harmanla(sol(y), g, SX)
                        } else {
                            g
                        }
                    };
                    fguv_parca::<SX, SY>(dst, src, luma, &k, scaling, bx + x0, y, g);
                }
            }
        }
        bx += bstep;
    }
}

/// `get_random_number(11)` dizisi, dörder adımlık çözümle: 16 bitlik durum `s`
/// için sonraki dört geri besleme biti `t = s ^ s>>1 ^ s>>3 ^ s>>12`'nin 0–3.
/// bitleridir (musluklar 0, 1, 3, 12; en yakın geri besleme 4 adım ötede),
/// `x = s | (t & 0xF) << 16` ve `k`. adımın değeri `(x >> (k + 5)) & 0x7FF`
/// (`k` 1–4), yeni durum `(x >> 4) & 0xFFFF`. Durum 16 bittir (giriş `seed <
/// 65536` ister; bitstream 16 bit okur).
struct Gauss {
    s: u32,
    x: u32,
    kalan: u32,
}

impl Gauss {
    fn yeni(seed: u32) -> Self {
        Self {
            s: seed,
            x: 0,
            kalan: 0,
        }
    }

    #[inline]
    fn sonraki(&mut self) -> usize {
        if self.kalan == 0 {
            let s = self.s;
            let t = s ^ (s >> 1) ^ (s >> 3) ^ (s >> 12);
            self.x = s | ((t & 0xF) << 16);
            self.s = (self.x >> 4) & 0xFFFF;
            self.kalan = 4;
        }
        let k = 5 - self.kalan;
        self.kalan -= 1;
        ((self.x >> (k + 5)) & 0x7FF) as usize
    }
}

/// Grain LUT'unun `w × h` bölgesini Gauss örnekleriyle doldurur (satır sırası;
/// `round2(GAUSSIAN_SEQUENCE[v], 4 + grain_scale_shift)`).
pub(super) fn gauss_doldur(buf: &mut GrainLut, seed: u32, w: usize, h: usize, shift: u32) {
    let mut g = Gauss::yeni(seed);
    let yuvarla = (1 << shift) >> 1;
    for row in buf.iter_mut().take(h) {
        for v in row.iter_mut().take(w) {
            *v = ((i32::from(GAUSSIAN_SEQUENCE[g.sonraki()]) + yuvarla) >> shift) as i16;
        }
    }
}

/// AR süzgeci (`ar_coeff_lag` `L` ≥ 1 ya da luma katkısı): satır `y` için üst
/// satırların ve luma'nın katkısı (sütunlar `3..w−3`, 8'li parçalar, son parça
/// örtüşür) AVX2 ile `ust`'e, sonra satır içi bağımlı kısım sıralı.
/// `luma`: `(buf_y, subx, suby, katsayı)` — `None`: luma katkısı yok.
#[target_feature(enable = "avx2")]
fn ar_suz(
    buf: &mut GrainLut,
    w: usize,
    h: usize,
    lag: usize,
    katsayi: &[i8],
    ar_shift: u32,
    luma: Option<(&GrainLut, usize, usize, i32)>,
) {
    let son = w - 3;
    let n_ust = lag * (2 * lag + 1);
    let mut ust = [0i32; 82];
    let parca_bas = |x0: usize| x0.min(son - 8);
    for y in 3..h {
        let mut x0 = 3;
        while x0 < son {
            let xb = parca_bas(x0);
            let mut acc = _mm256_setzero_si256();
            for r in 1..=lag {
                let satir = &buf[y - r];
                for dx in 0..=2 * lag {
                    let c = i32::from(katsayi[(lag - r) * (2 * lag + 1) + dx]);
                    let v = _mm256_cvtepi16_epi32(load_i16x8(chunk(satir, xb + dx - lag)));
                    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(v, _mm256_set1_epi32(c)));
                }
            }
            if let Some((by, subx, suby, c)) = luma {
                let lx = ((xb - 3) << subx) + 3;
                let ly = ((y - 3) << suby) + 3;
                let satir_topla = |r: &[i16; 82]| {
                    if subx != 0 {
                        _mm256_madd_epi16(load_i16x16(chunk(r, lx)), _mm256_set1_epi16(1))
                    } else {
                        _mm256_cvtepi16_epi32(load_i16x8(chunk(r, lx)))
                    }
                };
                let mut l = satir_topla(&by[ly]);
                if suby != 0 {
                    l = _mm256_add_epi32(l, satir_topla(&by[ly + 1]));
                }
                let sh = (subx + suby) as i32;
                let l = _mm256_srav_epi32(
                    _mm256_add_epi32(l, _mm256_set1_epi32((1 << sh) >> 1)),
                    _mm256_set1_epi32(sh),
                );
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(l, _mm256_set1_epi32(c)));
            }
            store_i32x8(chunk_mut(&mut ust, xb), acc);
            x0 = xb + 8;
        }
        let yuvarla = (1i32 << ar_shift) >> 1;
        let satir = &mut buf[y];
        for x in 3..son {
            let mut sum = ust[x];
            for k in 1..=lag {
                sum += i32::from(katsayi[n_ust + lag - k]) * i32::from(satir[x - k]);
            }
            let g = i32::from(satir[x]) + ((sum + yuvarla) >> ar_shift);
            satir[x] = g.clamp(GRAIN_MIN, GRAIN_MAX) as i16;
        }
    }
}

/// Grain üretiminin O(1) sözleşmesi (skaler gövdenin koşulları ve `seed < 65536`).
pub(super) fn grain_sozlesmesi(data: &FilmGrainData) -> bool {
    (0..=3).contains(&data.ar_coeff_lag)
        && (0..=3).contains(&data.grain_scale_shift)
        && data.ar_coeff_shift <= 16
        && data.seed < 0x1_0000
}

#[target_feature(enable = "avx2")]
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

#[target_feature(enable = "avx2")]
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`generate_grain_y_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır.
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
            // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
            // (`generate_grain_uv_fn`); belirteç AVX2 tespiti + maske izninin
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

/// `generate_grain_y` 8 bpc AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn generate_grain_y_fn(_avx2: Avx2Token) -> GenerateGrainYFn {
    grain_y_giris
}

/// `generate_grain_uv[düzen]` 8 bpc AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn generate_grain_uv_fn(_avx2: Avx2Token, duzen: usize) -> GenerateGrainUvFn {
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
    let tablo = scaling.first_chunk::<OLCEKLEME_TOPLAMA>();
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
    // (`fgy_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır. (2) sınır:
    // `fgy_sozlesmesi` `src`/`dst`'nin `(bh−1)·stride + pw` baytı kapsadığını ve
    // `stride ≥ pw`'yi, `first_chunk` tablonun 259 baytını garanti eder; bütün
    // erişimler denetimli dilim/dizi erişimidir. (3) hizalama/takma ad: `dst`
    // tek `&mut`, diğerleri `&`; ödünç kuralları çakışmayı dışlar.
    unsafe { fgy_govde(dst, src, stride, data, pw, tablo, grain_lut, bh, row_num) }
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
            let tablo = scaling.first_chunk::<OLCEKLEME_TOPLAMA>();
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
            // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir
            // (`fguv_fn`); belirteç AVX2 tespiti + maske izninin kanıtıdır. (2) sınır:
            // `fguv_sozlesmesi` `src`/`dst`'nin `(bh−1)·stride + pw` baytı ve lumanın
            // `((bh−1) << sy)·luma_stride + ((pw−1) << sx) + sx + 1` baytı kapsadığını,
            // `first_chunk` tablonun 259 baytını garanti eder; erişimler denetimlidir.
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

/// `fgy_32x32xn` 8 bpc AVX2 yuvası (yalnız belirteçle).
#[must_use]
pub fn fgy_fn(_avx2: Avx2Token) -> FgyFn {
    fgy_giris
}

/// `fguv_32x32xn[düzen]` 8 bpc AVX2 yuvası (yalnız belirteçle; düzen dizini
/// 0 = 420, 1 = 422, 2 = 444).
#[must_use]
pub fn fguv_fn(_avx2: Avx2Token, duzen: usize) -> FguvFn {
    match duzen {
        0 => fguv_420_giris,
        1 => fguv_422_giris,
        _ => fguv_444_giris,
    }
}
