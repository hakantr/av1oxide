//! 8 bpc intra tahmin (`intra_pred[14]`, `cfl_ac[3]`, `cfl_pred[4]`, `pal_pred`;
//! 22 kayıt) x86_64 SSSE3 çekirdekleri — P45 Adım 2 (Paket A).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Kayıtlar | Yuva | Kaynak |
//! |---|---|---|
//! | `D01621`, `D01626`, `D01632`, `D01637`, `D01642`, `D01648` | `intra_pred[DC/VERT/HOR/LEFT_DC/TOP_DC/DC_128]` | `src/x86/ipred_sse.asm:341-724`, `src/x86/ipred.h:76-81` |
//! | `D01654`, `D01660`, `D01666` | `intra_pred[Z1/Z2/Z3]` | `src/x86/ipred_sse.asm:1174-2240`, `src/x86/ipred.h:82-84` |
//! | `D01671`, `D01676`, `D01682` | `intra_pred[SMOOTH/SMOOTH_V/SMOOTH_H]` | `src/x86/ipred_sse.asm:2598-3230`, `src/x86/ipred.h:85-87` |
//! | `D01688` | `intra_pred[PAETH]` | `src/x86/ipred_sse.asm:2241-2597`, `src/x86/ipred.h:88` |
//! | `D01694` | `intra_pred[FILTER]` | `src/x86/ipred_sse.asm:5219-5414`, `src/x86/ipred.h:89` |
//! | `D01592`, `D01596`, `D01600` | `cfl_ac[420/422/444]` | `src/x86/ipred_sse.asm:3626-4475`, `src/x86/ipred.h:91-93` |
//! | `D01604`, `D01608`, `D01612`, `D01616` | `cfl_pred[DC/LEFT_DC/TOP_DC/DC_128]` | `src/x86/ipred_sse.asm:3231-3625`, `src/x86/ipred.h:95-98` |
//! | `D01700` | `pal_pred` | `src/x86/ipred_sse.asm:5154-5218`, `src/x86/ipred.h:100` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bu modülde tüm SIMD işlevleri yalnız `#[target_feature(enable = "ssse3")]`
//!   taşır; SSE4.1+, VEX ve EVEX yönergesi üretilmez (`0` SSE4.1+/VEX/EVEX).
//! - Kopyasız O(1) sözleşme: girişte [`angular_sozlesmesi::<8>`],
//!   [`cfl_ac_sozlesmesi::<8>`], [`cfl_pred_sozlesmesi::<8>`],
//!   [`pal_pred_sozlesmesi::<8>`] denetlenir; sözleşme dışı girdide ilgili
//!   `*_skaler_devret::<8>` ile skaler gövdeye devredilir.
//! - `OverReadProbe` güvenliği: `edge.bytes` üzerinde yalnız izinli
//!   `[topleft - back .. topleft + 1 + front]` aralığı okunur; `cfl_ac` içinde
//!   yalnız görünür luma bölgesi okunur; `dst` üzerinde her satırda yalnız
//!   `dst[y * dst_stride .. y * dst_stride + w]` baytlarına yazılır.

#![allow(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use core::arch::x86_64::{
    __m128i, _mm_abs_epi16, _mm_add_epi16, _mm_add_epi32, _mm_and_si128, _mm_andnot_si128,
    _mm_avg_epu16, _mm_cmpgt_epi16, _mm_cvtsi32_si128, _mm_cvtsi64_si128, _mm_cvtsi128_si32,
    _mm_madd_epi16, _mm_maddubs_epi16, _mm_mulhrs_epi16, _mm_mullo_epi16, _mm_or_si128,
    _mm_packus_epi16, _mm_sad_epu8, _mm_set1_epi8, _mm_set1_epi16, _mm_setr_epi8,
    _mm_setzero_si128, _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_sign_epi16, _mm_slli_epi16,
    _mm_srai_epi16, _mm_srli_epi16, _mm_sub_epi16, _mm_unpackhi_epi8, _mm_unpackhi_epi64,
    _mm_unpacklo_epi8,
};

use crate::cpu::Ssse3Token;
use crate::dsp::ipred::{
    AngularPredFn, CflAcFn, CflPredFn, CflPredMode, Edge, IntraPredMode, PalPredFn,
    angular_skaler_devret, angular_sozlesmesi, cfl_ac_skaler_devret, cfl_ac_sozlesmesi,
    cfl_pred_skaler_devret, cfl_pred_sozlesmesi, pal_pred_skaler_devret, pal_pred_sozlesmesi,
};
use crate::dsp::ipred_tables::{DR_INTRA_DERIVATIVE, FILTER_INTRA_TAPS, SM_WEIGHTS};
use crate::dsp::x86::mem;
use crate::frame::BitDepth;

const TMP_LEN: usize = 257;
const TMP_BASE: isize = 128;

// ---------------------------------------------------------------------------
// Kurucular (`Ssse3Token` kapılı, çalışma zamanında `Token::new` çağrılmaz)
// ---------------------------------------------------------------------------

/// 8 bpc `intra_pred[mode]` SSSE3 yuva işaretçisi (`D01621`..`D01694`).
#[must_use]
pub const fn intra_pred_fn(_token: Ssse3Token, mode: IntraPredMode) -> AngularPredFn {
    match mode {
        IntraPredMode::Dc => ipred_dc_ssse3,
        IntraPredMode::Vert => ipred_vert_ssse3,
        IntraPredMode::Hor => ipred_hor_ssse3,
        IntraPredMode::LeftDc => ipred_left_dc_ssse3,
        IntraPredMode::TopDc => ipred_top_dc_ssse3,
        IntraPredMode::Dc128 => ipred_dc_128_ssse3,
        IntraPredMode::Z1 => ipred_z1_ssse3,
        IntraPredMode::Z2 => ipred_z2_ssse3,
        IntraPredMode::Z3 => ipred_z3_ssse3,
        IntraPredMode::Smooth => ipred_smooth_ssse3,
        IntraPredMode::SmoothV => ipred_smooth_v_ssse3,
        IntraPredMode::SmoothH => ipred_smooth_h_ssse3,
        IntraPredMode::Paeth => ipred_paeth_ssse3,
        IntraPredMode::Filter => ipred_filter_ssse3,
    }
}

/// 8 bpc `cfl_ac[layout]` SSSE3 yuva işaretçisi (`D01592`, `D01596`, `D01600`).
#[must_use]
pub const fn cfl_ac_fn(_token: Ssse3Token, layout: usize) -> Option<CflAcFn> {
    match layout {
        0 => Some(cfl_ac_420_ssse3),
        1 => Some(cfl_ac_422_ssse3),
        2 => Some(cfl_ac_444_ssse3),
        _ => None,
    }
}

/// 8 bpc `cfl_pred[mode]` SSSE3 yuva işaretçisi (`D01604`, `D01608`, `D01612`, `D01616`).
#[must_use]
pub const fn cfl_pred_fn(_token: Ssse3Token, mode: CflPredMode) -> CflPredFn {
    match mode {
        CflPredMode::Dc => cfl_pred_dc_ssse3,
        CflPredMode::LeftDc => cfl_pred_left_dc_ssse3,
        CflPredMode::TopDc => cfl_pred_top_dc_ssse3,
        CflPredMode::Dc128 => cfl_pred_dc_128_ssse3,
    }
}

/// 8 bpc `pal_pred` SSSE3 yuva işaretçisi (`D01700`).
#[must_use]
pub const fn pal_pred_fn(_token: Ssse3Token) -> PalPredFn {
    pal_pred_ssse3
}

// ---------------------------------------------------------------------------
// Ortak yardımcılar (`#[target_feature(enable = "ssse3")]`)
// ---------------------------------------------------------------------------

/// `src[..n]` (`n ∈ {4, 8, 16, 32, 64}`) baytlarının toplamını SSE2 `psadbw` ile hesaplar.
/// Yalnız `src[..n]` baytlarını okur (`OverReadProbe` güvenli).
#[target_feature(enable = "ssse3")]
#[inline]
fn sum_bytes_ssse3(src: &[u8], n: usize) -> u32 {
    let zero = _mm_setzero_si128();
    match n {
        4 => {
            let v = u32::from_le_bytes(*mem::chunk::<u8, 4>(src, 0));
            let m = _mm_cvtsi32_si128(v as i32);
            _mm_cvtsi128_si32(_mm_sad_epu8(m, zero)) as u32
        }
        8 => {
            let v = u64::from_le_bytes(*mem::chunk::<u8, 8>(src, 0));
            let m = _mm_cvtsi64_si128(v as i64);
            _mm_cvtsi128_si32(_mm_sad_epu8(m, zero)) as u32
        }
        16 => {
            let s = _mm_sad_epu8(mem::load_u8x16(mem::chunk::<u8, 16>(src, 0)), zero);
            let s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, zero));
            _mm_cvtsi128_si32(s) as u32
        }
        32 => {
            let s0 = _mm_sad_epu8(mem::load_u8x16(mem::chunk::<u8, 16>(src, 0)), zero);
            let s1 = _mm_sad_epu8(mem::load_u8x16(mem::chunk::<u8, 16>(src, 16)), zero);
            let s = _mm_add_epi32(s0, s1);
            let s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, zero));
            _mm_cvtsi128_si32(s) as u32
        }
        64 => {
            let s0 = _mm_sad_epu8(mem::load_u8x16(mem::chunk::<u8, 16>(src, 0)), zero);
            let s1 = _mm_sad_epu8(mem::load_u8x16(mem::chunk::<u8, 16>(src, 16)), zero);
            let s2 = _mm_sad_epu8(mem::load_u8x16(mem::chunk::<u8, 16>(src, 32)), zero);
            let s3 = _mm_sad_epu8(mem::load_u8x16(mem::chunk::<u8, 16>(src, 48)), zero);
            let s = _mm_add_epi32(_mm_add_epi32(s0, s1), _mm_add_epi32(s2, s3));
            let s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, zero));
            _mm_cvtsi128_si32(s) as u32
        }
        _ => unreachable!(),
    }
}

#[target_feature(enable = "ssse3")]
#[inline]
fn dc_gen_top_ssse3(edge: Edge<'_>, width: usize) -> u32 {
    let top = &edge.bytes[edge.topleft + 1..edge.topleft + 1 + width];
    let sum = sum_bytes_ssse3(top, width) + (width as u32 >> 1);
    sum >> width.trailing_zeros()
}

#[target_feature(enable = "ssse3")]
#[inline]
fn dc_gen_left_ssse3(edge: Edge<'_>, height: usize) -> u32 {
    let left = &edge.bytes[edge.topleft - height..edge.topleft];
    let sum = sum_bytes_ssse3(left, height) + (height as u32 >> 1);
    sum >> height.trailing_zeros()
}

#[target_feature(enable = "ssse3")]
#[inline]
fn dc_gen_ssse3(edge: Edge<'_>, width: usize, height: usize) -> u32 {
    let top = &edge.bytes[edge.topleft + 1..edge.topleft + 1 + width];
    let left = &edge.bytes[edge.topleft - height..edge.topleft];
    let mut dc = sum_bytes_ssse3(top, width)
        + sum_bytes_ssse3(left, height)
        + ((width + height) as u32 >> 1);
    dc >>= (width + height).trailing_zeros();
    if width != height {
        let mul = if width > height * 2 || height > width * 2 {
            0x3334u64
        } else {
            0x5556u64
        };
        dc = ((u64::from(dc) * mul) >> 16) as u32;
    }
    dc
}

/// Sabit bir `v` (16×`u8`) vektörünü `dst` tamponunun `width × height` bölgesine yazar.
#[target_feature(enable = "ssse3")]
#[inline]
fn splat_row_ssse3(dst: &mut [u8], stride: usize, width: usize, height: usize, v: __m128i) {
    match width {
        4 => {
            for y in 0..height {
                mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, y * stride), v);
            }
        }
        8 => {
            for y in 0..height {
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, y * stride), v);
            }
        }
        16 => {
            for y in 0..height {
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, y * stride), v);
            }
        }
        32 => {
            for y in 0..height {
                let row = y * stride;
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row), v);
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row + 16), v);
            }
        }
        64 => {
            for y in 0..height {
                let row = y * stride;
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row), v);
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row + 16), v);
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row + 32), v);
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row + 48), v);
            }
        }
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// 1. DC / VERT / HOR / LEFT_DC / TOP_DC / DC_128 (`D01621`..`D01648`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
fn ipred_dc_ssse3_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let dc = dc_gen_ssse3(edge, width, height) as i8;
    splat_row_ssse3(dst, dst_stride, width, height, _mm_set1_epi8(dc));
}

#[target_feature(enable = "ssse3")]
fn ipred_top_dc_ssse3_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let dc = dc_gen_top_ssse3(edge, width) as i8;
    splat_row_ssse3(dst, dst_stride, width, height, _mm_set1_epi8(dc));
}

#[target_feature(enable = "ssse3")]
fn ipred_left_dc_ssse3_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let dc = dc_gen_left_ssse3(edge, height) as i8;
    splat_row_ssse3(dst, dst_stride, width, height, _mm_set1_epi8(dc));
}

#[target_feature(enable = "ssse3")]
fn ipred_dc_128_ssse3_giris(dst: &mut [u8], dst_stride: usize, width: usize, height: usize) {
    splat_row_ssse3(dst, dst_stride, width, height, _mm_set1_epi8(-128i8));
}

#[target_feature(enable = "ssse3")]
fn ipred_vert_ssse3_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let top = &edge.bytes[edge.topleft + 1..edge.topleft + 1 + width];
    match width {
        4 => {
            let v = _mm_cvtsi32_si128(u32::from_le_bytes(*mem::chunk::<u8, 4>(top, 0)) as i32);
            splat_row_ssse3(dst, dst_stride, 4, height, v);
        }
        8 => {
            let v = _mm_cvtsi64_si128(u64::from_le_bytes(*mem::chunk::<u8, 8>(top, 0)) as i64);
            splat_row_ssse3(dst, dst_stride, 8, height, v);
        }
        16 => {
            let v = mem::load_u8x16(mem::chunk::<u8, 16>(top, 0));
            splat_row_ssse3(dst, dst_stride, 16, height, v);
        }
        32 => {
            let v0 = mem::load_u8x16(mem::chunk::<u8, 16>(top, 0));
            let v1 = mem::load_u8x16(mem::chunk::<u8, 16>(top, 16));
            for y in 0..height {
                let row = y * dst_stride;
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row), v0);
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row + 16), v1);
            }
        }
        64 => {
            let v0 = mem::load_u8x16(mem::chunk::<u8, 16>(top, 0));
            let v1 = mem::load_u8x16(mem::chunk::<u8, 16>(top, 16));
            let v2 = mem::load_u8x16(mem::chunk::<u8, 16>(top, 32));
            let v3 = mem::load_u8x16(mem::chunk::<u8, 16>(top, 48));
            for y in 0..height {
                let row = y * dst_stride;
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row), v0);
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row + 16), v1);
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row + 32), v2);
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row + 48), v3);
            }
        }
        _ => unreachable!(),
    }
}

#[target_feature(enable = "ssse3")]
fn ipred_hor_ssse3_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let left = &edge.bytes[edge.topleft - height..edge.topleft];
    let shuf3 = _mm_set1_epi8(3);
    let shuf2 = _mm_set1_epi8(2);
    let shuf1 = _mm_set1_epi8(1);
    let shuf0 = _mm_set1_epi8(0);
    let mut y = 0;
    while y < height {
        // left[height - 1 - y] = edge[-(y + 1)]
        let chunk4 = u32::from_le_bytes(*mem::chunk::<u8, 4>(left, height - 4 - y));
        let base = _mm_cvtsi32_si128(chunk4 as i32);
        let r0 = _mm_shuffle_epi8(base, shuf3);
        let r1 = _mm_shuffle_epi8(base, shuf2);
        let r2 = _mm_shuffle_epi8(base, shuf1);
        let r3 = _mm_shuffle_epi8(base, shuf0);
        for (dy, rv) in [(0, r0), (1, r1), (2, r2), (3, r3)] {
            let row = (y + dy) * dst_stride;
            match width {
                4 => mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, row), rv),
                8 => mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, row), rv),
                16 => mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row), rv),
                32 => {
                    mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row), rv);
                    mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row + 16), rv);
                }
                64 => {
                    mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row), rv);
                    mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row + 16), rv);
                    mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row + 32), rv);
                    mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row + 48), rv);
                }
                _ => unreachable!(),
            }
        }
        y += 4;
    }
}

// ---------------------------------------------------------------------------
// 2. PAETH (`D01688`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
#[inline]
fn paeth_row8_ssse3(
    t16: __m128i,
    ldiff16: __m128i,
    l16: __m128i,
    tl16: __m128i,
    tdiff16: __m128i,
) -> __m128i {
    // base = l + t - tl; tldiff = |tl - base| = |(l - tl) + (t - tl)|
    let t_minus_tl = _mm_sub_epi16(t16, tl16);
    let l_minus_tl = _mm_sub_epi16(l16, tl16);
    let tldiff16 = _mm_abs_epi16(_mm_add_epi16(l_minus_tl, t_minus_tl));
    // Koşul 1: ldiff <= tdiff && ldiff <= tldiff  <=>  !(ldiff > tdiff || ldiff > tldiff)
    let gt_lt = _mm_cmpgt_epi16(ldiff16, tdiff16);
    let gt_ltl = _mm_cmpgt_epi16(ldiff16, tldiff16);
    let not_l_mask = _mm_or_si128(gt_lt, gt_ltl);
    // Koşul 2: tdiff <= tldiff  <=>  !(tdiff > tldiff)
    let not_t_mask = _mm_cmpgt_epi16(tdiff16, tldiff16);
    let t_or_tl = _mm_or_si128(
        _mm_and_si128(not_t_mask, tl16),
        _mm_andnot_si128(not_t_mask, t16),
    );
    _mm_or_si128(
        _mm_and_si128(not_l_mask, t_or_tl),
        _mm_andnot_si128(not_l_mask, l16),
    )
}

#[target_feature(enable = "ssse3")]
fn ipred_paeth_ssse3_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let zero = _mm_setzero_si128();
    let tl = i16::from(edge.bytes[edge.topleft]);
    let tl16 = _mm_set1_epi16(tl);
    let top_slice = &edge.bytes[edge.topleft + 1..edge.topleft + 1 + width];
    let left_slice = &edge.bytes[edge.topleft - height..edge.topleft];

    // Top örneklerini 64 baytlık yığın tamponuna kopyala (16 baytlık güvenli yükleme için)
    let mut top_buf = [0u8; 64];
    top_buf[..width].copy_from_slice(top_slice);

    let mut top16 = [_mm_setzero_si128(); 8];
    let mut ldiff16 = [_mm_setzero_si128(); 8];
    let chunks8 = width.div_ceil(8);
    for c in 0..chunks8 {
        let raw8 = u64::from_le_bytes(*mem::chunk::<u8, 8>(&top_buf, c * 8));
        let t8 = _mm_cvtsi64_si128(raw8 as i64);
        let t16 = _mm_unpacklo_epi8(t8, zero);
        top16[c] = t16;
        ldiff16[c] = _mm_abs_epi16(_mm_sub_epi16(t16, tl16));
    }

    for y in 0..height {
        let l = i16::from(left_slice[height - 1 - y]);
        let l16 = _mm_set1_epi16(l);
        let tdiff16 = _mm_set1_epi16((l - tl).abs());
        let row = y * dst_stride;
        match width {
            4 => {
                let p0 = paeth_row8_ssse3(top16[0], ldiff16[0], l16, tl16, tdiff16);
                let packed = _mm_packus_epi16(p0, p0);
                mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, row), packed);
            }
            8 => {
                let p0 = paeth_row8_ssse3(top16[0], ldiff16[0], l16, tl16, tdiff16);
                let packed = _mm_packus_epi16(p0, p0);
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, row), packed);
            }
            16 => {
                let p0 = paeth_row8_ssse3(top16[0], ldiff16[0], l16, tl16, tdiff16);
                let p1 = paeth_row8_ssse3(top16[1], ldiff16[1], l16, tl16, tdiff16);
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row), _mm_packus_epi16(p0, p1));
            }
            32 => {
                for g in 0..2 {
                    let p0 = paeth_row8_ssse3(top16[2 * g], ldiff16[2 * g], l16, tl16, tdiff16);
                    let p1 =
                        paeth_row8_ssse3(top16[2 * g + 1], ldiff16[2 * g + 1], l16, tl16, tdiff16);
                    mem::store_u8x16(
                        mem::chunk_mut::<u8, 16>(dst, row + g * 16),
                        _mm_packus_epi16(p0, p1),
                    );
                }
            }
            64 => {
                for g in 0..4 {
                    let p0 = paeth_row8_ssse3(top16[2 * g], ldiff16[2 * g], l16, tl16, tdiff16);
                    let p1 =
                        paeth_row8_ssse3(top16[2 * g + 1], ldiff16[2 * g + 1], l16, tl16, tdiff16);
                    mem::store_u8x16(
                        mem::chunk_mut::<u8, 16>(dst, row + g * 16),
                        _mm_packus_epi16(p0, p1),
                    );
                }
            }
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------
// 3. SMOOTH / SMOOTH_V / SMOOTH_H (`D01671`, `D01676`, `D01682`)
// ---------------------------------------------------------------------------

/// `wv * top + (256 - wv) * bottom + add` değerini 16-bit işaretsiz şeritlerde hesaplar.
/// Not: `wv ∈ [4, 255]`, `top, bottom ∈ [0, 255]` olduğu için
/// `wv * top + (256 - wv) * bottom <= 256 * 255 = 65280`; `+ 128` veya `+ 255`
/// ile `65535`'i aşmaz (`u16` içinde taşmasızdır).
#[target_feature(enable = "ssse3")]
#[inline]
fn smooth_term8_ssse3(
    a16: __m128i,
    b16: __m128i,
    w16: __m128i,
    inv_w16: __m128i,
    bias16: __m128i,
) -> __m128i {
    // `_mm_mullo_epi16` u16 mod 65536 çarpımı yaptığı için taşmasız u16 toplamı verir.
    let p0 = _mm_mullo_epi16(a16, w16);
    let p1 = _mm_mullo_epi16(b16, inv_w16);
    _mm_add_epi16(_mm_add_epi16(p0, p1), bias16)
}

#[target_feature(enable = "ssse3")]
fn ipred_smooth_v_ssse3_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let zero = _mm_setzero_si128();
    let weights_ver = &SM_WEIGHTS[height..2 * height];
    let bottom = i16::from(edge.bytes[edge.topleft - height]);
    let bottom16 = _mm_set1_epi16(bottom);
    let bias128 = _mm_set1_epi16(128);

    let mut top_buf = [0u8; 64];
    top_buf[..width].copy_from_slice(&edge.bytes[edge.topleft + 1..edge.topleft + 1 + width]);
    let mut top16 = [_mm_setzero_si128(); 8];
    let chunks8 = width.div_ceil(8);
    for c in 0..chunks8 {
        let raw8 = u64::from_le_bytes(*mem::chunk::<u8, 8>(&top_buf, c * 8));
        top16[c] = _mm_unpacklo_epi8(_mm_cvtsi64_si128(raw8 as i64), zero);
    }

    for y in 0..height {
        let wv = i16::from(weights_ver[y]);
        let wv16 = _mm_set1_epi16(wv);
        let inv_wv16 = _mm_set1_epi16(256 - wv);
        let row = y * dst_stride;
        match width {
            4 => {
                let s0 = _mm_srli_epi16::<8>(smooth_term8_ssse3(
                    top16[0], bottom16, wv16, inv_wv16, bias128,
                ));
                mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, row), _mm_packus_epi16(s0, s0));
            }
            8 => {
                let s0 = _mm_srli_epi16::<8>(smooth_term8_ssse3(
                    top16[0], bottom16, wv16, inv_wv16, bias128,
                ));
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, row), _mm_packus_epi16(s0, s0));
            }
            16 => {
                let s0 = _mm_srli_epi16::<8>(smooth_term8_ssse3(
                    top16[0], bottom16, wv16, inv_wv16, bias128,
                ));
                let s1 = _mm_srli_epi16::<8>(smooth_term8_ssse3(
                    top16[1], bottom16, wv16, inv_wv16, bias128,
                ));
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row), _mm_packus_epi16(s0, s1));
            }
            32 => {
                for g in 0..2 {
                    let s0 = _mm_srli_epi16::<8>(smooth_term8_ssse3(
                        top16[2 * g],
                        bottom16,
                        wv16,
                        inv_wv16,
                        bias128,
                    ));
                    let s1 = _mm_srli_epi16::<8>(smooth_term8_ssse3(
                        top16[2 * g + 1],
                        bottom16,
                        wv16,
                        inv_wv16,
                        bias128,
                    ));
                    mem::store_u8x16(
                        mem::chunk_mut::<u8, 16>(dst, row + g * 16),
                        _mm_packus_epi16(s0, s1),
                    );
                }
            }
            64 => {
                for g in 0..4 {
                    let s0 = _mm_srli_epi16::<8>(smooth_term8_ssse3(
                        top16[2 * g],
                        bottom16,
                        wv16,
                        inv_wv16,
                        bias128,
                    ));
                    let s1 = _mm_srli_epi16::<8>(smooth_term8_ssse3(
                        top16[2 * g + 1],
                        bottom16,
                        wv16,
                        inv_wv16,
                        bias128,
                    ));
                    mem::store_u8x16(
                        mem::chunk_mut::<u8, 16>(dst, row + g * 16),
                        _mm_packus_epi16(s0, s1),
                    );
                }
            }
            _ => unreachable!(),
        }
    }
}

#[target_feature(enable = "ssse3")]
fn ipred_smooth_h_ssse3_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let zero = _mm_setzero_si128();
    let weights_hor = &SM_WEIGHTS[width..2 * width];
    let right = i16::from(edge.bytes[edge.topleft + width]);
    let right16 = _mm_set1_epi16(right);
    let bias128 = _mm_set1_epi16(128);
    let c256 = _mm_set1_epi16(256);

    let mut wh_buf = [0u8; 64];
    wh_buf[..width].copy_from_slice(weights_hor);
    let mut wh16 = [_mm_setzero_si128(); 8];
    let mut inv_wh16 = [_mm_setzero_si128(); 8];
    let chunks8 = width.div_ceil(8);
    for c in 0..chunks8 {
        let raw8 = u64::from_le_bytes(*mem::chunk::<u8, 8>(&wh_buf, c * 8));
        let w = _mm_unpacklo_epi8(_mm_cvtsi64_si128(raw8 as i64), zero);
        wh16[c] = w;
        inv_wh16[c] = _mm_sub_epi16(c256, w);
    }

    let left_slice = &edge.bytes[edge.topleft - height..edge.topleft];
    for y in 0..height {
        let l16 = _mm_set1_epi16(i16::from(left_slice[height - 1 - y]));
        let row = y * dst_stride;
        match width {
            4 => {
                let s0 = _mm_srli_epi16::<8>(smooth_term8_ssse3(
                    l16,
                    right16,
                    wh16[0],
                    inv_wh16[0],
                    bias128,
                ));
                mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, row), _mm_packus_epi16(s0, s0));
            }
            8 => {
                let s0 = _mm_srli_epi16::<8>(smooth_term8_ssse3(
                    l16,
                    right16,
                    wh16[0],
                    inv_wh16[0],
                    bias128,
                ));
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, row), _mm_packus_epi16(s0, s0));
            }
            16 => {
                let s0 = _mm_srli_epi16::<8>(smooth_term8_ssse3(
                    l16,
                    right16,
                    wh16[0],
                    inv_wh16[0],
                    bias128,
                ));
                let s1 = _mm_srli_epi16::<8>(smooth_term8_ssse3(
                    l16,
                    right16,
                    wh16[1],
                    inv_wh16[1],
                    bias128,
                ));
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row), _mm_packus_epi16(s0, s1));
            }
            32 => {
                for g in 0..2 {
                    let s0 = _mm_srli_epi16::<8>(smooth_term8_ssse3(
                        l16,
                        right16,
                        wh16[2 * g],
                        inv_wh16[2 * g],
                        bias128,
                    ));
                    let s1 = _mm_srli_epi16::<8>(smooth_term8_ssse3(
                        l16,
                        right16,
                        wh16[2 * g + 1],
                        inv_wh16[2 * g + 1],
                        bias128,
                    ));
                    mem::store_u8x16(
                        mem::chunk_mut::<u8, 16>(dst, row + g * 16),
                        _mm_packus_epi16(s0, s1),
                    );
                }
            }
            64 => {
                for g in 0..4 {
                    let s0 = _mm_srli_epi16::<8>(smooth_term8_ssse3(
                        l16,
                        right16,
                        wh16[2 * g],
                        inv_wh16[2 * g],
                        bias128,
                    ));
                    let s1 = _mm_srli_epi16::<8>(smooth_term8_ssse3(
                        l16,
                        right16,
                        wh16[2 * g + 1],
                        inv_wh16[2 * g + 1],
                        bias128,
                    ));
                    mem::store_u8x16(
                        mem::chunk_mut::<u8, 16>(dst, row + g * 16),
                        _mm_packus_epi16(s0, s1),
                    );
                }
            }
            _ => unreachable!(),
        }
    }
}

#[target_feature(enable = "ssse3")]
fn ipred_smooth_ssse3_giris(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
) {
    let zero = _mm_setzero_si128();
    let weights_hor = &SM_WEIGHTS[width..2 * width];
    let weights_ver = &SM_WEIGHTS[height..2 * height];
    let right16 = _mm_set1_epi16(i16::from(edge.bytes[edge.topleft + width]));
    let bottom16 = _mm_set1_epi16(i16::from(edge.bytes[edge.topleft - height]));
    let bias255 = _mm_set1_epi16(255);
    let c256 = _mm_set1_epi16(256);

    let mut top_buf = [0u8; 64];
    top_buf[..width].copy_from_slice(&edge.bytes[edge.topleft + 1..edge.topleft + 1 + width]);
    let mut wh_buf = [0u8; 64];
    wh_buf[..width].copy_from_slice(weights_hor);

    let mut top16 = [_mm_setzero_si128(); 8];
    let mut wh16 = [_mm_setzero_si128(); 8];
    let mut inv_wh16 = [_mm_setzero_si128(); 8];
    let chunks8 = width.div_ceil(8);
    for c in 0..chunks8 {
        let raw_t = u64::from_le_bytes(*mem::chunk::<u8, 8>(&top_buf, c * 8));
        top16[c] = _mm_unpacklo_epi8(_mm_cvtsi64_si128(raw_t as i64), zero);
        let raw_w = u64::from_le_bytes(*mem::chunk::<u8, 8>(&wh_buf, c * 8));
        let w = _mm_unpacklo_epi8(_mm_cvtsi64_si128(raw_w as i64), zero);
        wh16[c] = w;
        inv_wh16[c] = _mm_sub_epi16(c256, w);
    }

    let left_slice = &edge.bytes[edge.topleft - height..edge.topleft];
    for y in 0..height {
        let wv = i16::from(weights_ver[y]);
        let wv16 = _mm_set1_epi16(wv);
        let inv_wv16 = _mm_set1_epi16(256 - wv);
        let l16 = _mm_set1_epi16(i16::from(left_slice[height - 1 - y]));
        let row = y * dst_stride;
        // v_term = wv*top + (256-wv)*bottom + 255; h_term = wh*left + (256-wh)*right
        // pavgw(v_term, h_term) >> 8 == (v_term + h_term + 1) >> 9 == (.. + 256) >> 9
        match width {
            4 => {
                let v = smooth_term8_ssse3(top16[0], bottom16, wv16, inv_wv16, bias255);
                let h = smooth_term8_ssse3(l16, right16, wh16[0], inv_wh16[0], zero);
                let s0 = _mm_srli_epi16::<8>(_mm_avg_epu16(v, h));
                mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, row), _mm_packus_epi16(s0, s0));
            }
            8 => {
                let v = smooth_term8_ssse3(top16[0], bottom16, wv16, inv_wv16, bias255);
                let h = smooth_term8_ssse3(l16, right16, wh16[0], inv_wh16[0], zero);
                let s0 = _mm_srli_epi16::<8>(_mm_avg_epu16(v, h));
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, row), _mm_packus_epi16(s0, s0));
            }
            16 => {
                let v0 = smooth_term8_ssse3(top16[0], bottom16, wv16, inv_wv16, bias255);
                let h0 = smooth_term8_ssse3(l16, right16, wh16[0], inv_wh16[0], zero);
                let s0 = _mm_srli_epi16::<8>(_mm_avg_epu16(v0, h0));
                let v1 = smooth_term8_ssse3(top16[1], bottom16, wv16, inv_wv16, bias255);
                let h1 = smooth_term8_ssse3(l16, right16, wh16[1], inv_wh16[1], zero);
                let s1 = _mm_srli_epi16::<8>(_mm_avg_epu16(v1, h1));
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row), _mm_packus_epi16(s0, s1));
            }
            32 => {
                for g in 0..2 {
                    let v0 = smooth_term8_ssse3(top16[2 * g], bottom16, wv16, inv_wv16, bias255);
                    let h0 = smooth_term8_ssse3(l16, right16, wh16[2 * g], inv_wh16[2 * g], zero);
                    let s0 = _mm_srli_epi16::<8>(_mm_avg_epu16(v0, h0));
                    let v1 =
                        smooth_term8_ssse3(top16[2 * g + 1], bottom16, wv16, inv_wv16, bias255);
                    let h1 = smooth_term8_ssse3(
                        l16,
                        right16,
                        wh16[2 * g + 1],
                        inv_wh16[2 * g + 1],
                        zero,
                    );
                    let s1 = _mm_srli_epi16::<8>(_mm_avg_epu16(v1, h1));
                    mem::store_u8x16(
                        mem::chunk_mut::<u8, 16>(dst, row + g * 16),
                        _mm_packus_epi16(s0, s1),
                    );
                }
            }
            64 => {
                for g in 0..4 {
                    let v0 = smooth_term8_ssse3(top16[2 * g], bottom16, wv16, inv_wv16, bias255);
                    let h0 = smooth_term8_ssse3(l16, right16, wh16[2 * g], inv_wh16[2 * g], zero);
                    let s0 = _mm_srli_epi16::<8>(_mm_avg_epu16(v0, h0));
                    let v1 =
                        smooth_term8_ssse3(top16[2 * g + 1], bottom16, wv16, inv_wv16, bias255);
                    let h1 = smooth_term8_ssse3(
                        l16,
                        right16,
                        wh16[2 * g + 1],
                        inv_wh16[2 * g + 1],
                        zero,
                    );
                    let s1 = _mm_srli_epi16::<8>(_mm_avg_epu16(v1, h1));
                    mem::store_u8x16(
                        mem::chunk_mut::<u8, 16>(dst, row + g * 16),
                        _mm_packus_epi16(s0, s1),
                    );
                }
            }
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------
// 4. FILTER_PRED (`D01694`) — `pmaddubsw` 4×2 blok çekirdeği
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
fn ipred_filter_ssse3_giris(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
) {
    let filter_index = angle & 511;
    let taps_i8 = &FILTER_INTRA_TAPS[filter_index];
    let mut taps_bytes = [0u8; 64];
    for i in 0..64 {
        taps_bytes[i] = taps_i8[i] as u8;
    }
    let t0 = mem::load_u8x16(mem::chunk::<u8, 16>(&taps_bytes, 0));
    let t1 = mem::load_u8x16(mem::chunk::<u8, 16>(&taps_bytes, 16));
    let t2 = mem::load_u8x16(mem::chunk::<u8, 16>(&taps_bytes, 32));
    let t3 = mem::load_u8x16(mem::chunk::<u8, 16>(&taps_bytes, 48));
    let round8 = _mm_set1_epi16(8);

    let mut y = 0;
    while y < height {
        let top_from_edge = y == 0;
        let edge_tl = edge.topleft - y;
        let mut x = 0;
        while x < width {
            let group = x / 4;
            let (p0, p1, p2, p3, p4) = if top_from_edge {
                if group == 0 {
                    (
                        edge.bytes[edge_tl],
                        edge.bytes[edge.topleft + 1],
                        edge.bytes[edge.topleft + 2],
                        edge.bytes[edge.topleft + 3],
                        edge.bytes[edge.topleft + 4],
                    )
                } else {
                    let col = edge.topleft + 4 * group;
                    (
                        edge.bytes[col],
                        edge.bytes[col + 1],
                        edge.bytes[col + 2],
                        edge.bytes[col + 3],
                        edge.bytes[col + 4],
                    )
                }
            } else {
                let top_row = (y - 1) * stride;
                let col = 4 * group;
                let p0 = if group == 0 {
                    edge.bytes[edge_tl]
                } else {
                    dst[top_row + col - 1]
                };
                (
                    p0,
                    dst[top_row + col],
                    dst[top_row + col + 1],
                    dst[top_row + col + 2],
                    dst[top_row + col + 3],
                )
            };
            let (p5, p6) = if group == 0 {
                (edge.bytes[edge_tl - 1], edge.bytes[edge_tl - 2])
            } else {
                (dst[y * stride + x - 1], dst[(y + 1) * stride + x - 1])
            };

            let v01 = _mm_set1_epi16(u16::from_le_bytes([p0, p1]) as i16);
            let v23 = _mm_set1_epi16(u16::from_le_bytes([p2, p3]) as i16);
            let v45 = _mm_set1_epi16(u16::from_le_bytes([p4, p5]) as i16);
            let v60 = _mm_set1_epi16(u16::from_le_bytes([p6, 0]) as i16);

            let acc0 = _mm_add_epi16(_mm_maddubs_epi16(v01, t0), _mm_maddubs_epi16(v23, t1));
            let acc1 = _mm_add_epi16(_mm_maddubs_epi16(v45, t2), _mm_maddubs_epi16(v60, t3));
            let sum = _mm_srai_epi16::<4>(_mm_add_epi16(_mm_add_epi16(acc0, acc1), round8));
            let packed = _mm_packus_epi16(sum, sum);
            let row1 = _mm_shuffle_epi32::<0x55>(packed);

            mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, y * stride + x), packed);
            mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, (y + 1) * stride + x), row1);
            x += 4;
        }
        y += 2;
    }
}

// ---------------------------------------------------------------------------
// 5. Z1 / Z2 / Z3 (`D01654`, `D01660`, `D01666`)
// ---------------------------------------------------------------------------

#[inline(always)]
fn clip_isize(value: isize, low: isize, high: isize) -> isize {
    value.max(low).min(high)
}

#[inline(always)]
fn read_edge8(edge: Edge<'_>, relative: isize) -> i32 {
    i32::from(edge.bytes[(edge.topleft as isize + relative) as usize])
}

#[inline(always)]
fn get_filter_strength(wh: usize, angle: usize, is_sm: bool) -> usize {
    if is_sm {
        if wh <= 8 {
            if angle >= 64 {
                return 2;
            }
            if angle >= 40 {
                return 1;
            }
        } else if wh <= 16 {
            if angle >= 48 {
                return 2;
            }
            if angle >= 20 {
                return 1;
            }
        } else if wh <= 24 {
            if angle >= 4 {
                return 3;
            }
        } else {
            return 3;
        }
    } else if wh <= 8 {
        if angle >= 56 {
            return 1;
        }
    } else if wh <= 16 {
        if angle >= 40 {
            return 1;
        }
    } else if wh <= 24 {
        if angle >= 32 {
            return 3;
        }
        if angle >= 16 {
            return 2;
        }
        if angle >= 8 {
            return 1;
        }
    } else if wh <= 32 {
        if angle >= 32 {
            return 3;
        }
        if angle >= 4 {
            return 2;
        }
        return 1;
    } else {
        return 3;
    }
    0
}

#[inline(always)]
fn get_upsample(wh: usize, angle: usize, is_sm: bool) -> bool {
    angle < 40 && wh <= (16 >> usize::from(is_sm))
}

fn filter_edge8(
    out: &mut [u8],
    sz: usize,
    lim_from: isize,
    lim_to: isize,
    edge: Edge<'_>,
    in_base: isize,
    from: isize,
    to: isize,
    strength: usize,
) {
    const KERNEL: [[i32; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
    let copy_end = if lim_from < 0 {
        0
    } else {
        sz.min(lim_from as usize)
    };
    let mut i = 0;
    while i < copy_end {
        out[i] = read_edge8(edge, in_base + clip_isize(i as isize, from, to - 1)) as u8;
        i += 1;
    }
    let filter_end = if lim_to < 0 {
        0
    } else {
        sz.min(lim_to as usize)
    };
    while i < filter_end {
        let mut sum = 0;
        for (j, weight) in KERNEL[strength - 1].iter().enumerate() {
            let position = clip_isize(i as isize - 2 + j as isize, from, to - 1);
            sum += read_edge8(edge, in_base + position) * weight;
        }
        out[i] = ((sum + 8) >> 4) as u8;
        i += 1;
    }
    while i < sz {
        out[i] = read_edge8(edge, in_base + clip_isize(i as isize, from, to - 1)) as u8;
        i += 1;
    }
}

fn upsample_edge8(
    out: &mut [u8],
    hsz: usize,
    edge: Edge<'_>,
    in_base: isize,
    from: isize,
    to: isize,
) {
    const KERNEL: [i32; 4] = [-1, 9, 9, -1];
    let mut i = 0;
    while i < hsz - 1 {
        out[i * 2] = read_edge8(edge, in_base + clip_isize(i as isize, from, to - 1)) as u8;
        let mut sum = 0;
        for (j, weight) in KERNEL.iter().enumerate() {
            let position = clip_isize(i as isize + j as isize - 1, from, to - 1);
            sum += read_edge8(edge, in_base + position) * weight;
        }
        out[i * 2 + 1] = ((sum + 8) >> 4).clamp(0, 255) as u8;
        i += 1;
    }
    out[(hsz - 1) * 2] =
        read_edge8(edge, in_base + clip_isize(hsz as isize - 1, from, to - 1)) as u8;
}

/// Z1: `buf[0..=max_base_x]` hazırlandıktan sonra `max_base_x` sonrasını
/// `buf[max_base_x]` ile doldurursak, `base >= max_base_x` taşma durumunda
/// `(fill * (64 - frac) + fill * frac + 32) >> 6 == fill` olduğu için tüm satır
/// dallanmasız `pmaddubsw` + `pmulhrsw` ile hesaplanır!
#[target_feature(enable = "ssse3")]
fn ipred_z1_ssse3_giris(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
) {
    let is_sm = (angle >> 9) & 1 == 1;
    let enable = (angle >> 10) != 0;
    let base_angle = angle & 511;
    let mut dx = i32::from(DR_INTRA_DERIVATIVE[base_angle >> 1]);
    let mut buf = [0u8; 256];
    let upsample = enable && get_upsample(width + height, 90 - base_angle, is_sm);
    let (base_inc, max_base_x) = if upsample {
        upsample_edge8(
            &mut buf,
            width + height,
            edge,
            1,
            -1,
            (width + width.min(height)) as isize,
        );
        dx <<= 1;
        (2usize, 2 * (width + height) - 2)
    } else {
        let strength = if enable {
            get_filter_strength(width + height, 90 - base_angle, is_sm)
        } else {
            0
        };
        if strength > 0 {
            filter_edge8(
                &mut buf,
                width + height,
                0,
                (width + height) as isize,
                edge,
                1,
                -1,
                (width + width.min(height)) as isize,
                strength,
            );
            (1usize, width + height - 1)
        } else {
            let n = width + width.min(height);
            buf[..n].copy_from_slice(&edge.bytes[edge.topleft + 1..edge.topleft + 1 + n]);
            (1usize, n - 1)
        }
    };
    let fill = buf[max_base_x];
    buf[max_base_x + 1..max_base_x + 1 + 64].fill(fill);

    // b0_b1 çiftlerini pshufb ile seçmek için maske:
    // base_inc == 1: [0,1, 1,2, 2,3, 3,4, 4,5, 5,6, 6,7, 7,8]
    // base_inc == 2: [0,1, 2,3, 4,5, 6,7, 8,9, 10,11, 12,13, 14,15] (shuf bile gerekmez!)
    let shuf_inc1 = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);
    let mul512 = _mm_set1_epi16(512);
    let mut xpos = dx;
    for y in 0..height {
        let frac = (xpos & 0x3E) as i8;
        let wpair = _mm_set1_epi16(u16::from_le_bytes([(64 - frac) as u8, frac as u8]) as i16);
        let base = ((xpos >> 6) as usize).min(max_base_x);
        let row = y * stride;
        if base_inc == 1 {
            match width {
                4 => {
                    let s0 = mem::load_u8x16(mem::chunk::<u8, 16>(&buf, base));
                    let pairs = _mm_shuffle_epi8(s0, shuf_inc1);
                    let prod = _mm_mulhrs_epi16(_mm_maddubs_epi16(pairs, wpair), mul512);
                    mem::store_u8x4(
                        mem::chunk_mut::<u8, 4>(dst, row),
                        _mm_packus_epi16(prod, prod),
                    );
                }
                8 => {
                    let s0 = mem::load_u8x16(mem::chunk::<u8, 16>(&buf, base));
                    let pairs = _mm_shuffle_epi8(s0, shuf_inc1);
                    let prod = _mm_mulhrs_epi16(_mm_maddubs_epi16(pairs, wpair), mul512);
                    mem::store_u8x8(
                        mem::chunk_mut::<u8, 8>(dst, row),
                        _mm_packus_epi16(prod, prod),
                    );
                }
                16 => {
                    let b0 = base;
                    let b1 = (base + 8).min(max_base_x);
                    let p0 = _mm_shuffle_epi8(
                        mem::load_u8x16(mem::chunk::<u8, 16>(&buf, b0)),
                        shuf_inc1,
                    );
                    let p1 = _mm_shuffle_epi8(
                        mem::load_u8x16(mem::chunk::<u8, 16>(&buf, b1)),
                        shuf_inc1,
                    );
                    let r0 = _mm_mulhrs_epi16(_mm_maddubs_epi16(p0, wpair), mul512);
                    let r1 = _mm_mulhrs_epi16(_mm_maddubs_epi16(p1, wpair), mul512);
                    mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, row), _mm_packus_epi16(r0, r1));
                }
                32 => {
                    for g in 0..2 {
                        let b0 = (base + g * 16).min(max_base_x);
                        let b1 = (base + g * 16 + 8).min(max_base_x);
                        let p0 = _mm_shuffle_epi8(
                            mem::load_u8x16(mem::chunk::<u8, 16>(&buf, b0)),
                            shuf_inc1,
                        );
                        let p1 = _mm_shuffle_epi8(
                            mem::load_u8x16(mem::chunk::<u8, 16>(&buf, b1)),
                            shuf_inc1,
                        );
                        let r0 = _mm_mulhrs_epi16(_mm_maddubs_epi16(p0, wpair), mul512);
                        let r1 = _mm_mulhrs_epi16(_mm_maddubs_epi16(p1, wpair), mul512);
                        mem::store_u8x16(
                            mem::chunk_mut::<u8, 16>(dst, row + g * 16),
                            _mm_packus_epi16(r0, r1),
                        );
                    }
                }
                64 => {
                    for g in 0..4 {
                        let b0 = (base + g * 16).min(max_base_x);
                        let b1 = (base + g * 16 + 8).min(max_base_x);
                        let p0 = _mm_shuffle_epi8(
                            mem::load_u8x16(mem::chunk::<u8, 16>(&buf, b0)),
                            shuf_inc1,
                        );
                        let p1 = _mm_shuffle_epi8(
                            mem::load_u8x16(mem::chunk::<u8, 16>(&buf, b1)),
                            shuf_inc1,
                        );
                        let r0 = _mm_mulhrs_epi16(_mm_maddubs_epi16(p0, wpair), mul512);
                        let r1 = _mm_mulhrs_epi16(_mm_maddubs_epi16(p1, wpair), mul512);
                        mem::store_u8x16(
                            mem::chunk_mut::<u8, 16>(dst, row + g * 16),
                            _mm_packus_epi16(r0, r1),
                        );
                    }
                }
                _ => unreachable!(),
            }
        } else {
            // base_inc == 2: yalnız width + height <= 16 iken etkin (width ∈ {4, 8})
            match width {
                4 => {
                    let p0 = mem::load_u8x16(mem::chunk::<u8, 16>(&buf, base));
                    let r0 = _mm_mulhrs_epi16(_mm_maddubs_epi16(p0, wpair), mul512);
                    mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, row), _mm_packus_epi16(r0, r0));
                }
                8 => {
                    let p0 = mem::load_u8x16(mem::chunk::<u8, 16>(&buf, base));
                    let r0 = _mm_mulhrs_epi16(_mm_maddubs_epi16(p0, wpair), mul512);
                    mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, row), _mm_packus_epi16(r0, r0));
                }
                _ => unreachable!(),
            }
        }
        xpos += dx;
    }
}

#[target_feature(enable = "ssse3")]
fn ipred_z2_ssse3_giris(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
    max_width: usize,
    max_height: usize,
) {
    let is_sm = (angle >> 9) & 1 == 1;
    let enable = (angle >> 10) != 0;
    let base_angle = angle & 511;
    let mut dy = i32::from(DR_INTRA_DERIVATIVE[(base_angle - 90) >> 1]);
    let mut dx = i32::from(DR_INTRA_DERIVATIVE[(180 - base_angle) >> 1]);
    let upsample_left = enable && get_upsample(width + height, 180 - base_angle, is_sm);
    let upsample_above = enable && get_upsample(width + height, base_angle - 90, is_sm);
    let mut buf = [0u8; TMP_LEN + 64];
    let base = TMP_BASE;

    if upsample_above {
        upsample_edge8(
            &mut buf[TMP_BASE as usize..],
            width + 1,
            edge,
            0,
            0,
            (width + 1) as isize,
        );
        dx <<= 1;
    } else {
        let strength = if enable {
            get_filter_strength(width + height, base_angle - 90, is_sm)
        } else {
            0
        };
        if strength > 0 {
            filter_edge8(
                &mut buf[(TMP_BASE + 1) as usize..],
                width,
                0,
                max_width as isize,
                edge,
                1,
                -1,
                width as isize,
                strength,
            );
        } else {
            buf[(base + 1) as usize..(base + 1) as usize + width]
                .copy_from_slice(&edge.bytes[edge.topleft + 1..edge.topleft + 1 + width]);
        }
    }

    if upsample_left {
        upsample_edge8(
            &mut buf[(TMP_BASE - 2 * height as isize) as usize..],
            height + 1,
            edge,
            -(height as isize),
            0,
            (height + 1) as isize,
        );
        dy <<= 1;
    } else {
        let strength = if enable {
            get_filter_strength(width + height, 180 - base_angle, is_sm)
        } else {
            0
        };
        if strength > 0 {
            filter_edge8(
                &mut buf[(TMP_BASE - height as isize) as usize..],
                height,
                height as isize - max_height as isize,
                height as isize,
                edge,
                -(height as isize),
                0,
                (height + 1) as isize,
                strength,
            );
        } else {
            buf[(base - height as isize) as usize..base as usize]
                .copy_from_slice(&edge.bytes[edge.topleft - height..edge.topleft]);
        }
    }
    buf[base as usize] = edge.bytes[edge.topleft];

    let shuf_inc1 = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);
    let mul512 = _mm_set1_epi16(512);
    let base_inc_x = 1 + isize::from(upsample_above);
    let left = base - (1 + isize::from(upsample_left));
    let mut xpos = ((1 + isize::from(upsample_above)) << 6) - dx as isize;

    for y in 0..height {
        let mut base_x = xpos >> 6;
        let frac_x = (xpos & 0x3E) as i32;
        let wpair_x =
            _mm_set1_epi16(u16::from_le_bytes([(64 - frac_x) as u8, frac_x as u8]) as i16);
        let mut ypos = ((y as isize) << (6 + u32::from(upsample_left))) - dy as isize;
        let row = y * stride;

        let mut x = 0;
        while x + 8 <= width {
            if base_x >= 0 {
                // Bu 8 pikselin tamamı üst kenardan (`base_x >= 0`) gelir: SIMD hızlı yolu
                let idx = (base + base_x) as usize;
                let pairs = if base_inc_x == 1 {
                    _mm_shuffle_epi8(mem::load_u8x16(mem::chunk::<u8, 16>(&buf, idx)), shuf_inc1)
                } else {
                    mem::load_u8x16(mem::chunk::<u8, 16>(&buf, idx))
                };
                let prod = _mm_mulhrs_epi16(_mm_maddubs_epi16(pairs, wpair_x), mul512);
                mem::store_u8x8(
                    mem::chunk_mut::<u8, 8>(dst, row + x),
                    _mm_packus_epi16(prod, prod),
                );
                base_x += 8 * base_inc_x;
                ypos -= 8 * dy as isize;
                x += 8;
            } else if base_x + 7 * base_inc_x < 0 {
                // 8 pikselin tamamı sol kenardan gelir: 8 çift ve 8 ağırlığı paketleyip `pmaddubsw`
                let mut p_bytes = [0u8; 16];
                let mut w_bytes = [0u8; 16];
                for k in 0..8 {
                    let by = ypos >> 6;
                    let fy = (ypos & 0x3E) as u8;
                    let l_idx = (left - by) as usize;
                    p_bytes[2 * k] = buf[l_idx];
                    p_bytes[2 * k + 1] = buf[l_idx - 1];
                    w_bytes[2 * k] = 64 - fy;
                    w_bytes[2 * k + 1] = fy;
                    base_x += base_inc_x;
                    ypos -= dy as isize;
                }
                let pv = mem::load_u8x16(&p_bytes);
                let wv = mem::load_u8x16(&w_bytes);
                let prod = _mm_mulhrs_epi16(_mm_maddubs_epi16(pv, wv), mul512);
                mem::store_u8x8(
                    mem::chunk_mut::<u8, 8>(dst, row + x),
                    _mm_packus_epi16(prod, prod),
                );
                x += 8;
            } else {
                // Geçiş bloğu (sol -> üst sınırı): 8 çifti birleştirip tek `pmaddubsw` ile hesapla
                let mut p_bytes = [0u8; 16];
                let mut w_bytes = [0u8; 16];
                for k in 0..8 {
                    if base_x >= 0 {
                        let t_idx = (base + base_x) as usize;
                        p_bytes[2 * k] = buf[t_idx];
                        p_bytes[2 * k + 1] = buf[t_idx + 1];
                        w_bytes[2 * k] = (64 - frac_x) as u8;
                        w_bytes[2 * k + 1] = frac_x as u8;
                    } else {
                        let by = ypos >> 6;
                        let fy = (ypos & 0x3E) as u8;
                        let l_idx = (left - by) as usize;
                        p_bytes[2 * k] = buf[l_idx];
                        p_bytes[2 * k + 1] = buf[l_idx - 1];
                        w_bytes[2 * k] = 64 - fy;
                        w_bytes[2 * k + 1] = fy;
                    }
                    base_x += base_inc_x;
                    ypos -= dy as isize;
                }
                let pv = mem::load_u8x16(&p_bytes);
                let wv = mem::load_u8x16(&w_bytes);
                let prod = _mm_mulhrs_epi16(_mm_maddubs_epi16(pv, wv), mul512);
                mem::store_u8x8(
                    mem::chunk_mut::<u8, 8>(dst, row + x),
                    _mm_packus_epi16(prod, prod),
                );
                x += 8;
            }
        }
        if width == 4 {
            let mut p_bytes = [0u8; 16];
            let mut w_bytes = [0u8; 16];
            for k in 0..4 {
                if base_x >= 0 {
                    let t_idx = (base + base_x) as usize;
                    p_bytes[2 * k] = buf[t_idx];
                    p_bytes[2 * k + 1] = buf[t_idx + 1];
                    w_bytes[2 * k] = (64 - frac_x) as u8;
                    w_bytes[2 * k + 1] = frac_x as u8;
                } else {
                    let by = ypos >> 6;
                    let fy = (ypos & 0x3E) as u8;
                    let l_idx = (left - by) as usize;
                    p_bytes[2 * k] = buf[l_idx];
                    p_bytes[2 * k + 1] = buf[l_idx - 1];
                    w_bytes[2 * k] = 64 - fy;
                    w_bytes[2 * k + 1] = fy;
                }
                base_x += base_inc_x;
                ypos -= dy as isize;
            }
            let pv = mem::load_u8x16(&p_bytes);
            let wv = mem::load_u8x16(&w_bytes);
            let prod = _mm_mulhrs_epi16(_mm_maddubs_epi16(pv, wv), mul512);
            mem::store_u8x4(
                mem::chunk_mut::<u8, 4>(dst, row),
                _mm_packus_epi16(prod, prod),
            );
        }
        xpos -= dx as isize;
    }
}

/// Z3: Sol kenar ters sırada (`rev_buf[k] = left_edge[-(1 + k)]`) düzleştirilip
/// `max_base_y` sonrası `fill` ile doldurulur, ardından sütunlar `pmaddubsw` + `pmulhrsw`
/// ile 4/8 piksel olarak hesaplanır.
#[target_feature(enable = "ssse3")]
fn ipred_z3_ssse3_giris(
    dst: &mut [u8],
    stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
) {
    let is_sm = (angle >> 9) & 1 == 1;
    let enable = (angle >> 10) != 0;
    let base_angle = angle & 511;
    let mut dy = i32::from(DR_INTRA_DERIVATIVE[(270 - base_angle) >> 1]);
    let mut left_out = [0u8; TMP_LEN];
    let upsample = enable && get_upsample(width + height, base_angle - 180, is_sm);
    let strength = if enable && !upsample {
        get_filter_strength(width + height, base_angle - 180, is_sm)
    } else {
        0
    };
    let direct = !upsample && strength == 0;

    // `rev_buf[base]` = `left[-(1 + base)]`, `rev_buf[base + 1]` = `left[-(2 + base)]`
    let mut rev_buf = [0u8; 512];
    let (base_inc, max_base_y) = if upsample {
        upsample_edge8(
            &mut left_out,
            width + height,
            edge,
            -((width + height) as isize),
            width.saturating_sub(height) as isize,
            (width + height + 1) as isize,
        );
        let lb = 2 * (width + height) - 2;
        for b in 0..=lb {
            rev_buf[b] = left_out[lb - b];
        }
        dy <<= 1;
        (2usize, lb)
    } else if !direct {
        filter_edge8(
            &mut left_out,
            width + height,
            0,
            (width + height) as isize,
            edge,
            -((width + height) as isize),
            width.saturating_sub(height) as isize,
            (width + height + 1) as isize,
            strength,
        );
        let lb = width + height - 1;
        for b in 0..=lb {
            rev_buf[b] = left_out[lb - b];
        }
        (1usize, lb)
    } else {
        let mb = height + width.min(height) - 1;
        for b in 0..=mb {
            rev_buf[b] = edge.bytes[edge.topleft - 1 - b];
        }
        (1usize, mb)
    };
    let fill = rev_buf[max_base_y];
    rev_buf[max_base_y + 1..].fill(fill);

    // `col_buf[x * 64 + y]` içine her `x` sütununu SIMD ile üret, sonra `dst` satırlarına yaz
    let shuf_inc1 = _mm_setr_epi8(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8);
    let mul512 = _mm_set1_epi16(512);
    let mut col_buf = [[0u8; 64]; 64];
    let mut ypos = dy as usize;
    for x in 0..width {
        let frac = (ypos & 0x3E) as i8;
        let wpair = _mm_set1_epi16(u16::from_le_bytes([(64 - frac) as u8, frac as u8]) as i16);
        let base = (ypos >> 6).min(max_base_y);
        if base_inc == 1 {
            let mut y = 0;
            while y + 8 <= height {
                let b0 = (base + y).min(max_base_y);
                let p0 = _mm_shuffle_epi8(
                    mem::load_u8x16(mem::chunk::<u8, 16>(&rev_buf, b0)),
                    shuf_inc1,
                );
                let r0 = _mm_mulhrs_epi16(_mm_maddubs_epi16(p0, wpair), mul512);
                mem::store_u8x8(
                    mem::chunk_mut::<u8, 8>(&mut col_buf[x], y),
                    _mm_packus_epi16(r0, r0),
                );
                y += 8;
            }
            if height == 4 {
                let p0 = _mm_shuffle_epi8(
                    mem::load_u8x16(mem::chunk::<u8, 16>(&rev_buf, base)),
                    shuf_inc1,
                );
                let r0 = _mm_mulhrs_epi16(_mm_maddubs_epi16(p0, wpair), mul512);
                mem::store_u8x4(
                    mem::chunk_mut::<u8, 4>(&mut col_buf[x], 0),
                    _mm_packus_epi16(r0, r0),
                );
            }
        } else {
            // base_inc == 2 (height ∈ {4, 8})
            let p0 = mem::load_u8x16(mem::chunk::<u8, 16>(&rev_buf, base));
            let r0 = _mm_mulhrs_epi16(_mm_maddubs_epi16(p0, wpair), mul512);
            if height == 4 {
                mem::store_u8x4(
                    mem::chunk_mut::<u8, 4>(&mut col_buf[x], 0),
                    _mm_packus_epi16(r0, r0),
                );
            } else {
                mem::store_u8x8(
                    mem::chunk_mut::<u8, 8>(&mut col_buf[x], 0),
                    _mm_packus_epi16(r0, r0),
                );
            }
        }
        ypos += dy as usize;
    }

    for y in 0..height {
        let row = y * stride;
        for x in 0..width {
            dst[row + x] = col_buf[x][y];
        }
    }
}

// ---------------------------------------------------------------------------
// 6. CFL_AC (`D01592`, `D01596`, `D01600`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
fn cfl_ac_dc_subtract_ssse3(ac: &mut [i16], cw: usize, ch: usize) {
    let total = cw * ch;
    let log2sz = cw.trailing_zeros() + ch.trailing_zeros();
    let ones16 = _mm_set1_epi16(1);
    let mut sum_v = _mm_setzero_si128();
    let mut i = 0;
    while i < total {
        let v = mem::load_i16x8(mem::chunk::<i16, 8>(ac, i));
        sum_v = _mm_add_epi32(sum_v, _mm_madd_epi16(v, ones16));
        i += 8;
    }
    sum_v = _mm_add_epi32(sum_v, _mm_unpackhi_epi64(sum_v, _mm_setzero_si128()));
    sum_v = _mm_add_epi32(sum_v, _mm_shuffle_epi32::<1>(sum_v));
    let mut sum = _mm_cvtsi128_si32(sum_v) + ((1i32 << log2sz) >> 1);
    sum >>= log2sz;

    let avg_v = _mm_set1_epi16(sum as i16);
    let mut j = 0;
    while j < total {
        let v = mem::load_i16x8(mem::chunk::<i16, 8>(ac, j));
        mem::store_i16x8(mem::chunk_mut::<i16, 8>(ac, j), _mm_sub_epi16(v, avg_v));
        j += 8;
    }
}

#[target_feature(enable = "ssse3")]
fn cfl_ac_core_ssse3<const SS_HOR: usize, const SS_VER: usize>(
    ac: &mut [i16],
    y_src: &[u8],
    y_stride: usize,
    w_pad: usize,
    h_pad: usize,
    cw: usize,
    ch: usize,
) {
    let visible_w = cw - 4 * w_pad;
    let visible_h = ch - 4 * h_pad;
    let zero = _mm_setzero_si128();
    let madd_w = if SS_VER == 1 {
        _mm_set1_epi8(2)
    } else {
        _mm_set1_epi8(4)
    };

    for row in 0..visible_h {
        let luma_row = row << SS_VER;
        let r0_off = luma_row * y_stride;
        let r1_off = (luma_row + 1) * y_stride;
        let ac_row = row * cw;

        let mut x = 0;
        while x + 8 <= visible_w {
            if SS_HOR == 1 {
                let col = x * 2;
                let v0 = mem::load_u8x16(mem::chunk::<u8, 16>(y_src, r0_off + col));
                let s0 = _mm_maddubs_epi16(v0, madd_w);
                let res = if SS_VER == 1 {
                    let v1 = mem::load_u8x16(mem::chunk::<u8, 16>(y_src, r1_off + col));
                    _mm_add_epi16(s0, _mm_maddubs_epi16(v1, madd_w))
                } else {
                    s0
                };
                mem::store_i16x8(mem::chunk_mut::<i16, 8>(ac, ac_row + x), res);
            } else {
                // 4:4:4 (SS_HOR == 0, SS_VER == 0): 8 luma baytını oku, << 3
                let raw8 = u64::from_le_bytes(*mem::chunk::<u8, 8>(y_src, r0_off + x));
                let v16 = _mm_unpacklo_epi8(_mm_cvtsi64_si128(raw8 as i64), zero);
                let res = _mm_slli_epi16::<3>(v16);
                mem::store_i16x8(mem::chunk_mut::<i16, 8>(ac, ac_row + x), res);
            }
            x += 8;
        }
        if x + 4 <= visible_w {
            if SS_HOR == 1 {
                let col = x * 2;
                let raw0 = u64::from_le_bytes(*mem::chunk::<u8, 8>(y_src, r0_off + col));
                let s0 = _mm_maddubs_epi16(_mm_cvtsi64_si128(raw0 as i64), madd_w);
                let res = if SS_VER == 1 {
                    let raw1 = u64::from_le_bytes(*mem::chunk::<u8, 8>(y_src, r1_off + col));
                    _mm_add_epi16(
                        s0,
                        _mm_maddubs_epi16(_mm_cvtsi64_si128(raw1 as i64), madd_w),
                    )
                } else {
                    s0
                };
                let mut tmp = [0i16; 8];
                mem::store_i16x8(&mut tmp, res);
                ac[ac_row + x..ac_row + x + 4].copy_from_slice(&tmp[..4]);
            } else {
                let raw4 = u32::from_le_bytes(*mem::chunk::<u8, 4>(y_src, r0_off + x));
                let v16 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(raw4 as i32), zero);
                let res = _mm_slli_epi16::<3>(v16);
                let mut tmp = [0i16; 8];
                mem::store_i16x8(&mut tmp, res);
                ac[ac_row + x..ac_row + x + 4].copy_from_slice(&tmp[..4]);
            }
            x += 4;
        }
        if x < cw {
            let fill = ac[ac_row + x - 1];
            while x < cw {
                ac[ac_row + x] = fill;
                x += 1;
            }
        }
    }

    for row in visible_h..ch {
        let src_row = (visible_h - 1) * cw;
        let dst_row = row * cw;
        for x in 0..cw {
            ac[dst_row + x] = ac[src_row + x];
        }
    }

    cfl_ac_dc_subtract_ssse3(ac, cw, ch);
}

// ---------------------------------------------------------------------------
// 7. CFL_PRED (`D01604`, `D01608`, `D01612`, `D01616`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
#[inline]
fn cfl_pred_vec8_ssse3(
    v_ac: __m128i,
    v_alpha: __m128i,
    v_abs_alpha_shl9: __m128i,
    v_dc: __m128i,
) -> __m128i {
    // dav1d `ipred_cfl_ssse3` (`src/x86/ipred_sse.asm:3590-3596`):
    // psignw sign, ac, alpha -> pabsw abs_ac, ac -> pmulhrsw scaled, abs_ac, |alpha|<<9
    // -> psignw signed, scaled, sign -> paddw res, signed, dc
    let sign = _mm_sign_epi16(v_ac, v_alpha);
    let abs_ac = _mm_abs_epi16(v_ac);
    let scaled = _mm_mulhrs_epi16(abs_ac, v_abs_alpha_shl9);
    let signed = _mm_sign_epi16(scaled, sign);
    _mm_add_epi16(signed, v_dc)
}

#[target_feature(enable = "ssse3")]
fn cfl_pred_core_ssse3(
    mode: CflPredMode,
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    ac: &[i16],
    alpha: i32,
) {
    let dc = match mode {
        CflPredMode::Dc => dc_gen_ssse3(edge, width, height) as i16,
        CflPredMode::LeftDc => dc_gen_left_ssse3(edge, height) as i16,
        CflPredMode::TopDc => dc_gen_top_ssse3(edge, width) as i16,
        CflPredMode::Dc128 => 128,
    };
    if alpha == 0 {
        splat_row_ssse3(dst, dst_stride, width, height, _mm_set1_epi8(dc as i8));
        return;
    }

    let v_dc = _mm_set1_epi16(dc);
    let v_alpha = _mm_set1_epi16(alpha as i16);
    let v_abs_alpha_shl9 = _mm_slli_epi16::<9>(_mm_abs_epi16(v_alpha));

    match width {
        4 => {
            let mut y = 0;
            while y < height {
                // 2 satırı (2 × 4 = 8 i16) tek 128-bit vektörde işle
                let v_ac = mem::load_i16x8(mem::chunk::<i16, 8>(ac, y * 4));
                let res = cfl_pred_vec8_ssse3(v_ac, v_alpha, v_abs_alpha_shl9, v_dc);
                let packed = _mm_packus_epi16(res, res);
                let row1 = _mm_shuffle_epi32::<0x55>(packed);
                mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, y * dst_stride), packed);
                mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, (y + 1) * dst_stride), row1);
                y += 2;
            }
        }
        8 => {
            for y in 0..height {
                let v_ac = mem::load_i16x8(mem::chunk::<i16, 8>(ac, y * 8));
                let res = cfl_pred_vec8_ssse3(v_ac, v_alpha, v_abs_alpha_shl9, v_dc);
                mem::store_u8x8(
                    mem::chunk_mut::<u8, 8>(dst, y * dst_stride),
                    _mm_packus_epi16(res, res),
                );
            }
        }
        16 => {
            for y in 0..height {
                let ac_off = y * 16;
                let r0 = cfl_pred_vec8_ssse3(
                    mem::load_i16x8(mem::chunk::<i16, 8>(ac, ac_off)),
                    v_alpha,
                    v_abs_alpha_shl9,
                    v_dc,
                );
                let r1 = cfl_pred_vec8_ssse3(
                    mem::load_i16x8(mem::chunk::<i16, 8>(ac, ac_off + 8)),
                    v_alpha,
                    v_abs_alpha_shl9,
                    v_dc,
                );
                mem::store_u8x16(
                    mem::chunk_mut::<u8, 16>(dst, y * dst_stride),
                    _mm_packus_epi16(r0, r1),
                );
            }
        }
        32 => {
            for y in 0..height {
                let ac_off = y * 32;
                let row = y * dst_stride;
                for g in 0..2 {
                    let r0 = cfl_pred_vec8_ssse3(
                        mem::load_i16x8(mem::chunk::<i16, 8>(ac, ac_off + g * 16)),
                        v_alpha,
                        v_abs_alpha_shl9,
                        v_dc,
                    );
                    let r1 = cfl_pred_vec8_ssse3(
                        mem::load_i16x8(mem::chunk::<i16, 8>(ac, ac_off + g * 16 + 8)),
                        v_alpha,
                        v_abs_alpha_shl9,
                        v_dc,
                    );
                    mem::store_u8x16(
                        mem::chunk_mut::<u8, 16>(dst, row + g * 16),
                        _mm_packus_epi16(r0, r1),
                    );
                }
            }
        }
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// 8. PAL_PRED (`D01700`) — `pshufb` 8-renk palet açma
// ---------------------------------------------------------------------------

#[target_feature(enable = "ssse3")]
fn pal_pred_ssse3_giris(
    dst: &mut [u8],
    dst_stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
) {
    let pal64 = u64::from_le_bytes(*mem::chunk::<u8, 8>(pal, 0));
    let v_pal = _mm_cvtsi64_si128(pal64 as i64);
    let mask_lo = _mm_set1_epi8(0x07);

    match w {
        4 => {
            let mut y = 0;
            while y < h {
                // 4 satır = 8 idx baytı = 16 piksel
                let raw8 = u64::from_le_bytes(*mem::chunk::<u8, 8>(idx, y * 2));
                let v_idx = _mm_cvtsi64_si128(raw8 as i64);
                let lo = _mm_and_si128(v_idx, mask_lo);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(v_idx), mask_lo);
                let even = _mm_shuffle_epi8(v_pal, lo);
                let odd = _mm_shuffle_epi8(v_pal, hi);
                let out16 = _mm_unpacklo_epi8(even, odd);
                mem::store_u8x4(mem::chunk_mut::<u8, 4>(dst, y * dst_stride), out16);
                mem::store_u8x4(
                    mem::chunk_mut::<u8, 4>(dst, (y + 1) * dst_stride),
                    _mm_shuffle_epi32::<0x55>(out16),
                );
                mem::store_u8x4(
                    mem::chunk_mut::<u8, 4>(dst, (y + 2) * dst_stride),
                    _mm_shuffle_epi32::<0xAA>(out16),
                );
                mem::store_u8x4(
                    mem::chunk_mut::<u8, 4>(dst, (y + 3) * dst_stride),
                    _mm_shuffle_epi32::<0xFF>(out16),
                );
                y += 4;
            }
        }
        8 => {
            let mut y = 0;
            while y < h {
                // 2 satır = 8 idx baytı = 16 piksel
                let raw8 = u64::from_le_bytes(*mem::chunk::<u8, 8>(idx, y * 4));
                let v_idx = _mm_cvtsi64_si128(raw8 as i64);
                let lo = _mm_and_si128(v_idx, mask_lo);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(v_idx), mask_lo);
                let even = _mm_shuffle_epi8(v_pal, lo);
                let odd = _mm_shuffle_epi8(v_pal, hi);
                let out16 = _mm_unpacklo_epi8(even, odd);
                mem::store_u8x8(mem::chunk_mut::<u8, 8>(dst, y * dst_stride), out16);
                mem::store_u8x8(
                    mem::chunk_mut::<u8, 8>(dst, (y + 1) * dst_stride),
                    _mm_unpackhi_epi64(out16, out16),
                );
                y += 2;
            }
        }
        16 => {
            for y in 0..h {
                let raw8 = u64::from_le_bytes(*mem::chunk::<u8, 8>(idx, y * 8));
                let v_idx = _mm_cvtsi64_si128(raw8 as i64);
                let lo = _mm_and_si128(v_idx, mask_lo);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(v_idx), mask_lo);
                let even = _mm_shuffle_epi8(v_pal, lo);
                let odd = _mm_shuffle_epi8(v_pal, hi);
                let out16 = _mm_unpacklo_epi8(even, odd);
                mem::store_u8x16(mem::chunk_mut::<u8, 16>(dst, y * dst_stride), out16);
            }
        }
        32 => {
            for y in 0..h {
                let v_idx = mem::load_u8x16(mem::chunk::<u8, 16>(idx, y * 16));
                let lo = _mm_and_si128(v_idx, mask_lo);
                let hi = _mm_and_si128(_mm_srli_epi16::<4>(v_idx), mask_lo);
                let even = _mm_shuffle_epi8(v_pal, lo);
                let odd = _mm_shuffle_epi8(v_pal, hi);
                let row = y * dst_stride;
                mem::store_u8x16(
                    mem::chunk_mut::<u8, 16>(dst, row),
                    _mm_unpacklo_epi8(even, odd),
                );
                mem::store_u8x16(
                    mem::chunk_mut::<u8, 16>(dst, row + 16),
                    _mm_unpackhi_epi8(even, odd),
                );
            }
        }
        64 => {
            for y in 0..h {
                let row = y * dst_stride;
                let idx_off = y * 32;
                for g in 0..2 {
                    let v_idx = mem::load_u8x16(mem::chunk::<u8, 16>(idx, idx_off + g * 16));
                    let lo = _mm_and_si128(v_idx, mask_lo);
                    let hi = _mm_and_si128(_mm_srli_epi16::<4>(v_idx), mask_lo);
                    let even = _mm_shuffle_epi8(v_pal, lo);
                    let odd = _mm_shuffle_epi8(v_pal, hi);
                    mem::store_u8x16(
                        mem::chunk_mut::<u8, 16>(dst, row + g * 32),
                        _mm_unpacklo_epi8(even, odd),
                    );
                    mem::store_u8x16(
                        mem::chunk_mut::<u8, 16>(dst, row + g * 32 + 16),
                        _mm_unpackhi_epi8(even, odd),
                    );
                }
            }
        }
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// Güvenli giriş sarmalayıcıları (O(1) sözleşme + tek `unsafe` blok)
// ---------------------------------------------------------------------------

macro_rules! ssse3_angular_entry {
    ($fn_name:ident, $mode:expr, $call:expr) => {
        fn $fn_name(
            dst: &mut [u8],
            dst_stride: usize,
            edge: Edge<'_>,
            width: usize,
            height: usize,
            angle: usize,
            max_width: usize,
            max_height: usize,
            bit_depth: BitDepth,
            bitdepth_max: u16,
        ) {
            if !angular_sozlesmesi::<8>(
                $mode,
                dst.len(),
                dst_stride,
                edge,
                width,
                height,
                angle,
                bit_depth,
                bitdepth_max,
            ) {
                angular_skaler_devret::<8>(
                    $mode,
                    dst,
                    dst_stride,
                    edge,
                    width,
                    height,
                    angle,
                    max_width,
                    max_height,
                    bit_depth,
                    bitdepth_max,
                );
                return;
            }
            let call_fn = $call;
            // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
            // (`intra_pred_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır.
            // (2) sınır: `angular_sozlesmesi::<8>` blok boyutunu, `edge` okuma aralığını
            // ve `dst` yazma bölgesini O(1) doğruladı. (3) hizalama/takma ad: tüm SIMD
            // yükleme/saklamalar `mem` içindeki hizasız sabit boyutlu dizi yardımcılarıyla
            // yapılır; `dst` `&mut [u8]` ve `edge.bytes` `&[u8]` ayrık referanslardır.
            unsafe {
                call_fn(
                    dst, dst_stride, edge, width, height, angle, max_width, max_height,
                )
            }
        }
    };
}

#[target_feature(enable = "ssse3")]
fn call_dc(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_dc_ssse3_giris(dst, s, e, w, h);
}
#[target_feature(enable = "ssse3")]
fn call_vert(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_vert_ssse3_giris(dst, s, e, w, h);
}
#[target_feature(enable = "ssse3")]
fn call_hor(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_hor_ssse3_giris(dst, s, e, w, h);
}
#[target_feature(enable = "ssse3")]
fn call_left_dc(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_left_dc_ssse3_giris(dst, s, e, w, h);
}
#[target_feature(enable = "ssse3")]
fn call_top_dc(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_top_dc_ssse3_giris(dst, s, e, w, h);
}
#[target_feature(enable = "ssse3")]
fn call_dc_128(
    dst: &mut [u8],
    s: usize,
    _e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_dc_128_ssse3_giris(dst, s, w, h);
}
#[target_feature(enable = "ssse3")]
fn call_z1(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_z1_ssse3_giris(dst, s, e, w, h, a);
}
#[target_feature(enable = "ssse3")]
fn call_z2(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    a: usize,
    mw: usize,
    mh: usize,
) {
    ipred_z2_ssse3_giris(dst, s, e, w, h, a, mw, mh);
}
#[target_feature(enable = "ssse3")]
fn call_z3(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_z3_ssse3_giris(dst, s, e, w, h, a);
}
#[target_feature(enable = "ssse3")]
fn call_smooth(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_smooth_ssse3_giris(dst, s, e, w, h);
}
#[target_feature(enable = "ssse3")]
fn call_smooth_v(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_smooth_v_ssse3_giris(dst, s, e, w, h);
}
#[target_feature(enable = "ssse3")]
fn call_smooth_h(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_smooth_h_ssse3_giris(dst, s, e, w, h);
}
#[target_feature(enable = "ssse3")]
fn call_paeth(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    _a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_paeth_ssse3_giris(dst, s, e, w, h);
}
#[target_feature(enable = "ssse3")]
fn call_filter(
    dst: &mut [u8],
    s: usize,
    e: Edge<'_>,
    w: usize,
    h: usize,
    a: usize,
    _mw: usize,
    _mh: usize,
) {
    ipred_filter_ssse3_giris(dst, s, e, w, h, a);
}

ssse3_angular_entry!(ipred_dc_ssse3, IntraPredMode::Dc, call_dc);
ssse3_angular_entry!(ipred_vert_ssse3, IntraPredMode::Vert, call_vert);
ssse3_angular_entry!(ipred_hor_ssse3, IntraPredMode::Hor, call_hor);
ssse3_angular_entry!(ipred_left_dc_ssse3, IntraPredMode::LeftDc, call_left_dc);
ssse3_angular_entry!(ipred_top_dc_ssse3, IntraPredMode::TopDc, call_top_dc);
ssse3_angular_entry!(ipred_dc_128_ssse3, IntraPredMode::Dc128, call_dc_128);
ssse3_angular_entry!(ipred_z1_ssse3, IntraPredMode::Z1, call_z1);
ssse3_angular_entry!(ipred_z2_ssse3, IntraPredMode::Z2, call_z2);
ssse3_angular_entry!(ipred_z3_ssse3, IntraPredMode::Z3, call_z3);
ssse3_angular_entry!(ipred_smooth_ssse3, IntraPredMode::Smooth, call_smooth);
ssse3_angular_entry!(ipred_smooth_v_ssse3, IntraPredMode::SmoothV, call_smooth_v);
ssse3_angular_entry!(ipred_smooth_h_ssse3, IntraPredMode::SmoothH, call_smooth_h);
ssse3_angular_entry!(ipred_paeth_ssse3, IntraPredMode::Paeth, call_paeth);
ssse3_angular_entry!(ipred_filter_ssse3, IntraPredMode::Filter, call_filter);

macro_rules! ssse3_cfl_ac_entry {
    ($fn_name:ident, $layout:expr, $ss_hor:expr, $ss_ver:expr) => {
        fn $fn_name(
            ac: &mut [i16],
            y: &[u8],
            y_stride: usize,
            w_pad: usize,
            h_pad: usize,
            cw: usize,
            ch: usize,
            bit_depth: BitDepth,
        ) {
            if !cfl_ac_sozlesmesi::<8>(
                $layout,
                ac.len(),
                y.len(),
                y_stride,
                w_pad,
                h_pad,
                cw,
                ch,
                bit_depth,
            ) {
                cfl_ac_skaler_devret::<8>(
                    $layout, ac, y, y_stride, w_pad, h_pad, cw, ch, bit_depth,
                );
                return;
            }
            // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
            // (`cfl_ac_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır.
            // (2) sınır: `cfl_ac_sozlesmesi::<8>` `cw, ch`, `w_pad, h_pad`, `ac.len()`
            // ve `y` luma dilim sınırlarını O(1) doğruladı. (3) hizalama/takma ad: tüm
            // yükleme/saklamalar `mem` içindeki hizasız sabit boyutlu dizi yardımcılarıyla
            // yapılır; `ac` `&mut [i16]` ve `y` `&[u8]` ayrık referanslardır.
            unsafe {
                cfl_ac_core_ssse3::<$ss_hor, $ss_ver>(ac, y, y_stride, w_pad, h_pad, cw, ch);
            }
        }
    };
}

ssse3_cfl_ac_entry!(cfl_ac_420_ssse3, 0, 1, 1);
ssse3_cfl_ac_entry!(cfl_ac_422_ssse3, 1, 1, 0);
ssse3_cfl_ac_entry!(cfl_ac_444_ssse3, 2, 0, 0);

macro_rules! ssse3_cfl_pred_entry {
    ($fn_name:ident, $mode:expr) => {
        fn $fn_name(
            dst: &mut [u8],
            dst_stride: usize,
            edge: Edge<'_>,
            width: usize,
            height: usize,
            ac: &[i16],
            alpha: i32,
            bit_depth: BitDepth,
            bitdepth_max: u16,
        ) {
            if !cfl_pred_sozlesmesi::<8>(
                $mode,
                dst.len(),
                dst_stride,
                edge,
                width,
                height,
                ac.len(),
                alpha,
                bit_depth,
                bitdepth_max,
            ) {
                cfl_pred_skaler_devret::<8>(
                    $mode,
                    dst,
                    dst_stride,
                    edge,
                    width,
                    height,
                    ac,
                    alpha,
                    bit_depth,
                    bitdepth_max,
                );
                return;
            }
            // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
            // (`cfl_pred_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır.
            // (2) sınır: `cfl_pred_sozlesmesi::<8>` `width, height ∈ 4..=32`,
            // `edge`, `ac` ve `dst` sınırlarını O(1) doğruladı. (3) hizalama/takma ad:
            // tüm yükleme/saklamalar `mem` yardımcılarıyla hizasız yapılır; `dst`,
            // `edge.bytes` ve `ac` ayrık referanslardır.
            unsafe {
                cfl_pred_core_ssse3($mode, dst, dst_stride, edge, width, height, ac, alpha);
            }
        }
    };
}

ssse3_cfl_pred_entry!(cfl_pred_dc_ssse3, CflPredMode::Dc);
ssse3_cfl_pred_entry!(cfl_pred_left_dc_ssse3, CflPredMode::LeftDc);
ssse3_cfl_pred_entry!(cfl_pred_top_dc_ssse3, CflPredMode::TopDc);
ssse3_cfl_pred_entry!(cfl_pred_dc_128_ssse3, CflPredMode::Dc128);

fn pal_pred_ssse3(
    dst: &mut [u8],
    dst_stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
    bit_depth: BitDepth,
) {
    if !pal_pred_sozlesmesi::<8>(dst.len(), dst_stride, pal.len(), idx.len(), w, h, bit_depth) {
        pal_pred_skaler_devret::<8>(dst, dst_stride, pal, idx, w, h, bit_depth);
        return;
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Ssse3Token` ile verilir
    // (`pal_pred_fn`); belirteç SSSE3 tespiti + maske izninin kanıtıdır.
    // (2) sınır: `pal_pred_sozlesmesi::<8>` `w, h ∈ 4..=64`, `pal.len() >= 8`,
    // `idx.len() >= (w*h)/2` ve `dst` sınırlarını O(1) doğruladı. (3) hizalama/takma ad:
    // tüm yükleme/saklamalar `mem` yardımcılarıyla hizasız yapılır; `dst`, `pal`
    // ve `idx` ayrık referanslardır.
    unsafe { pal_pred_ssse3_giris(dst, dst_stride, pal, idx, w, h) }
}
