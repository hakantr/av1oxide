//! Büyük boyutlu (`7` kayıt: `8×32`/`32×8` `DCT_DCT+IDTX` ve `16×32`/`32×16`/`32×32` `IDTX`)
//! 12 bpc ters dönüşüm x86_64 AVX2 çekirdekleri — P44 Adım 1.
//!
//! Bu modül dav1d'nin `src/x86/itx.h:255-262` ve `src/x86/itx16_avx2.asm` dosyalarındaki
//! `max(w, h) == 32` boyutlu **7** adet **yalnız 12 bpc** (`BitDepth::Twelve`) AVX2 `itxfm_add`
//! kaydını **saf Rust intrinsics** ile gerçekler; dış `.asm`/`.S`, `asm!` ve `global_asm!`
//! yoktur (CEK-02):
//!
//! - `RTX_8X32`: `DCT_DCT` ve `IDTX` (`2` kayıt),
//! - `RTX_32X8`: `DCT_DCT` ve `IDTX` (`2` kayıt),
//! - `RTX_16X32`: `IDTX` (`1` kayıt),
//! - `RTX_32X16`: `IDTX` (`1` kayıt),
//! - `TX_32X32`: `IDTX` (`1` kayıt).
//!
//! En büyük 8 `DCT_DCT` dönüşümü (`RTX_16X32`, `RTX_32X16`, `TX_32X32`, `RTX_16X64`,
//! `RTX_32X64`, `RTX_64X16`, `RTX_64X32`, `TX_64X64` × `DCT_DCT`) dav1d'de de 12 bpc AVX2
//! için tanımlı değildir (`src/x86/itx.h:249-263`) ve skalerde kalır.
//!
//! ## Sözleşme, yığın çerçevesi (`< 0x400`) ve ISA saflığı (P34 / P44)
//!
//! - Girişte O(1) sözleşme denetimi ([`itx16_sozlesmesi`]) ile `(tx, txtp, bit_depth == Twelve)`
//!   uyumu aranır; sözleşme dışı çağrıda paniklemeden [`itx_skaler_devret`] üzerinden sayaç
//!   artırılır ve ilgili bit derinliğinin skaler monomorfuna devredilir.
//! - `8×32` ve `32×8` `DCT_DCT` çekirdeklerinde ara `32 × 8` `i32` matrisi yığın üzerinde
//!   `[__m256i; 32]` (`1024` bayt) açmak yerine çağıranın `coeff[0..256]` tamponunda yerinde
//!   tutulur ve dönüş öncesi sıfırlanır; böylece tüm sembollerde `sub $N, %rsp` değeri
//!   `< 0x400` (`1024` bayt altı) kalır.
//! - Bütün çekirdek ve yardımcılar yalnız `#[target_feature(enable = "avx2")]` taşır;
//!   `itx16_12.rs` ve diğer `dsp/x86/*` modülleriyle `#[target_feature]` yardımcı paylaşmaz
//!   (P34) ve EVEX (`%zmm`, `{%k`) yönergesi üretmez.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, __m256i, _mm_add_epi16, _mm_max_epi16, _mm_min_epi16, _mm_packs_epi32, _mm_set1_epi16,
    _mm_setzero_si128, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi64, _mm256_and_si256,
    _mm256_blend_epi32, _mm256_blendv_epi8, _mm256_castsi256_si128, _mm256_cmpgt_epi64,
    _mm256_extracti128_si256, _mm256_max_epi16, _mm256_max_epi32, _mm256_min_epi16,
    _mm256_min_epi32, _mm256_mul_epi32, _mm256_mullo_epi32, _mm256_permute2x128_si256,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_epi64x, _mm256_setr_epi32,
    _mm256_setzero_si256, _mm256_slli_epi32, _mm256_slli_epi64, _mm256_srai_epi32,
    _mm256_srli_epi64, _mm256_sub_epi32, _mm256_sub_epi64, _mm256_unpackhi_epi32,
    _mm256_unpackhi_epi64, _mm256_unpacklo_epi32, _mm256_unpacklo_epi64, _mm256_xor_si256,
};

use super::mem::{chunk, chunk_mut, load_u8x16, load_u8x32, store_i32x8, store_u8x16, store_u8x32};
use crate::cpu::Avx2Token;
use crate::dsp::itx::{ItxFn, TxSize, TxType, itx_skaler_devret, itx16_sozlesmesi};
use crate::dsp::itx_tables::LNZC_TABLES;
use crate::frame::BitDepth;
#[cfg(feature = "std")]
use core::cell::RefCell;

const ROW_MIN: i32 = -524_288;
const ROW_MAX: i32 = 524_287;
const COL_MIN: i32 = -131_072;
const COL_MAX: i32 = 131_071;

// ---------------------------------------------------------------------------
// Katsayı yükleme, 64-bit çarpımlı `i32` aritmetik ve devrikleme yardımcıları
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
fn yukle_i32x8(coeff: &[i32], off: usize) -> __m256i {
    let c = chunk::<i32, 8>(coeff, off);
    _mm256_setr_epi32(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7])
}

#[target_feature(enable = "avx2")]
#[inline]
fn sifirla_katsayi<const N: usize>(coeff: &mut [i32]) {
    const { assert!(N.is_multiple_of(8) && N <= 1024) };
    let z = _mm256_setzero_si256();
    let mut i = 0;
    while i < N {
        store_i32x8(chunk_mut(coeff, i), z);
        i += 8;
    }
}

#[derive(Clone, Copy)]
struct I64x8 {
    ev: __m256i,
    od: __m256i,
}

#[target_feature(enable = "avx2")]
#[inline]
fn srai64<const SHIFT: i32>(x: __m256i) -> __m256i {
    _mm256_blend_epi32::<0xAA>(_mm256_srli_epi64::<SHIFT>(x), _mm256_srai_epi32::<SHIFT>(x))
}

#[target_feature(enable = "avx2")]
#[inline]
fn clamp64(v: __m256i, min_v: __m256i, max_v: __m256i) -> __m256i {
    let v = _mm256_blendv_epi8(v, min_v, _mm256_cmpgt_epi64(min_v, v));
    _mm256_blendv_epi8(v, max_v, _mm256_cmpgt_epi64(v, max_v))
}

#[target_feature(enable = "avx2")]
#[inline]
fn kirp<const MIN: i32, const MAX: i32>(v: __m256i) -> __m256i {
    _mm256_min_epi32(
        _mm256_max_epi32(v, _mm256_set1_epi32(MIN)),
        _mm256_set1_epi32(MAX),
    )
}

#[target_feature(enable = "avx2")]
#[inline]
fn kirp_64<const MIN: i32, const MAX: i32>(x: I64x8) -> __m256i {
    let min_v = _mm256_set1_epi64x(i64::from(MIN));
    let max_v = _mm256_set1_epi64x(i64::from(MAX));
    let ev = clamp64(x.ev, min_v, max_v);
    let od = clamp64(x.od, min_v, max_v);
    _mm256_blend_epi32::<0xAA>(ev, _mm256_slli_epi64::<32>(od))
}

#[target_feature(enable = "avx2")]
#[inline]
fn add_sub_kirp_64<const MIN: i32, const MAX: i32>(u: I64x8, v: I64x8) -> (__m256i, __m256i) {
    if MIN >= -131_072 {
        return (
            kirp::<MIN, MAX>(_mm256_add_epi32(u.ev, v.ev)),
            kirp::<MIN, MAX>(_mm256_sub_epi32(u.ev, v.ev)),
        );
    }
    let sum = I64x8 {
        ev: _mm256_add_epi64(u.ev, v.ev),
        od: _mm256_add_epi64(u.od, v.od),
    };
    let diff = I64x8 {
        ev: _mm256_sub_epi64(u.ev, v.ev),
        od: _mm256_sub_epi64(u.od, v.od),
    };
    (kirp_64::<MIN, MAX>(sum), kirp_64::<MIN, MAX>(diff))
}

#[target_feature(enable = "avx2")]
#[inline]
fn sub_add_kirp_64<const MIN: i32, const MAX: i32>(u: I64x8, v: I64x8) -> (__m256i, __m256i) {
    if MIN >= -131_072 {
        return (
            kirp::<MIN, MAX>(_mm256_sub_epi32(u.ev, v.ev)),
            kirp::<MIN, MAX>(_mm256_add_epi32(u.ev, v.ev)),
        );
    }
    let diff = I64x8 {
        ev: _mm256_sub_epi64(u.ev, v.ev),
        od: _mm256_sub_epi64(u.od, v.od),
    };
    let sum = I64x8 {
        ev: _mm256_add_epi64(u.ev, v.ev),
        od: _mm256_add_epi64(u.od, v.od),
    };
    (kirp_64::<MIN, MAX>(diff), kirp_64::<MIN, MAX>(sum))
}

#[target_feature(enable = "avx2")]
#[inline]
fn mul1_64<const SHIFT: i32>(a: __m256i, c0: i32, rnd: i64) -> I64x8 {
    let a_od = _mm256_srli_epi64::<32>(a);
    let kc0 = _mm256_set1_epi32(c0);
    let rnd_v = _mm256_set1_epi64x(rnd);
    let ev = srai64::<SHIFT>(_mm256_add_epi64(_mm256_mul_epi32(a, kc0), rnd_v));
    let od = srai64::<SHIFT>(_mm256_add_epi64(_mm256_mul_epi32(a_od, kc0), rnd_v));
    I64x8 { ev, od }
}

#[target_feature(enable = "avx2")]
#[inline]
fn mul1_64_min<const SHIFT: i32, const MIN: i32>(a: __m256i, c0: i32, rnd: i64) -> I64x8 {
    if MIN >= -131_072 {
        let u = _mm256_srai_epi32::<SHIFT>(_mm256_add_epi32(
            _mm256_mullo_epi32(a, _mm256_set1_epi32(c0)),
            _mm256_set1_epi32(rnd as i32),
        ));
        return I64x8 {
            ev: u,
            od: _mm256_setzero_si256(),
        };
    }
    mul1_64::<SHIFT>(a, c0, rnd)
}

#[target_feature(enable = "avx2")]
#[inline]
fn mul2_64<const SHIFT: i32, const MIN: i32>(
    a: __m256i,
    b: __m256i,
    c0: i32,
    c1: i32,
    d0: i32,
    d1: i32,
    rnd: i64,
) -> (I64x8, I64x8) {
    if MIN >= -131_072 {
        let kc0 = _mm256_set1_epi32(c0);
        let kc1 = _mm256_set1_epi32(c1);
        let kd0 = _mm256_set1_epi32(d0);
        let kd1 = _mm256_set1_epi32(d1);
        let rnd_v = _mm256_set1_epi32(rnd as i32);
        let u = _mm256_srai_epi32::<SHIFT>(_mm256_add_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(a, kc0), _mm256_mullo_epi32(b, kc1)),
            rnd_v,
        ));
        let v = _mm256_srai_epi32::<SHIFT>(_mm256_add_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(a, kd0), _mm256_mullo_epi32(b, kd1)),
            rnd_v,
        ));
        let z = _mm256_setzero_si256();
        return (I64x8 { ev: u, od: z }, I64x8 { ev: v, od: z });
    }
    let a_od = _mm256_srli_epi64::<32>(a);
    let b_od = _mm256_srli_epi64::<32>(b);
    let kc0 = _mm256_set1_epi32(c0);
    let kc1 = _mm256_set1_epi32(c1);
    let kd0 = _mm256_set1_epi32(d0);
    let kd1 = _mm256_set1_epi32(d1);
    let rnd_v = _mm256_set1_epi64x(rnd);

    let u_ev = srai64::<SHIFT>(_mm256_add_epi64(
        _mm256_add_epi64(_mm256_mul_epi32(a, kc0), _mm256_mul_epi32(b, kc1)),
        rnd_v,
    ));
    let u_od = srai64::<SHIFT>(_mm256_add_epi64(
        _mm256_add_epi64(_mm256_mul_epi32(a_od, kc0), _mm256_mul_epi32(b_od, kc1)),
        rnd_v,
    ));
    let v_ev = srai64::<SHIFT>(_mm256_add_epi64(
        _mm256_add_epi64(_mm256_mul_epi32(a, kd0), _mm256_mul_epi32(b, kd1)),
        rnd_v,
    ));
    let v_od = srai64::<SHIFT>(_mm256_add_epi64(
        _mm256_add_epi64(_mm256_mul_epi32(a_od, kd0), _mm256_mul_epi32(b_od, kd1)),
        rnd_v,
    ));
    (I64x8 { ev: u_ev, od: u_od }, I64x8 { ev: v_ev, od: v_od })
}

#[target_feature(enable = "avx2")]
#[inline]
fn olcekle_181(v: __m256i) -> __m256i {
    let v_od = _mm256_srli_epi64::<32>(v);
    let k181 = _mm256_set1_epi32(181);
    let rnd = _mm256_set1_epi64x(128);
    let ev = _mm256_srli_epi64::<8>(_mm256_add_epi64(_mm256_mul_epi32(v, k181), rnd));
    let od = _mm256_slli_epi64::<24>(_mm256_add_epi64(_mm256_mul_epi32(v_od, k181), rnd));
    _mm256_blend_epi32::<0xAA>(ev, od)
}

#[target_feature(enable = "avx2")]
#[inline]
fn olcekle_181_ic(v: __m256i) -> __m256i {
    _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_mullo_epi32(v, _mm256_set1_epi32(181)),
        _mm256_set1_epi32(128),
    ))
}

#[target_feature(enable = "avx2")]
#[inline]
fn mul2<const MIN: i32>(
    a: __m256i,
    b: __m256i,
    c0: i32,
    c1: i32,
    d0: i32,
    d1: i32,
) -> (__m256i, __m256i) {
    let kc0 = _mm256_set1_epi32(c0);
    let kc1 = _mm256_set1_epi32(c1);
    let kd0 = _mm256_set1_epi32(d0);
    let kd1 = _mm256_set1_epi32(d1);
    let rnd = _mm256_set1_epi32(2048);

    let u0 = _mm256_add_epi32(_mm256_mullo_epi32(a, kc0), rnd);
    let u1 = _mm256_mullo_epi32(b, kc1);
    let v0 = _mm256_add_epi32(_mm256_mullo_epi32(a, kd0), rnd);
    let v1 = _mm256_mullo_epi32(b, kd1);

    if MIN >= -131_072 {
        (
            _mm256_srai_epi32::<12>(_mm256_add_epi32(u0, u1)),
            _mm256_srai_epi32::<12>(_mm256_add_epi32(v0, v1)),
        )
    } else {
        let u_avg = _mm256_add_epi32(
            _mm256_and_si256(u0, u1),
            _mm256_srai_epi32::<1>(_mm256_xor_si256(u0, u1)),
        );
        let v_avg = _mm256_add_epi32(
            _mm256_and_si256(v0, v1),
            _mm256_srai_epi32::<1>(_mm256_xor_si256(v0, v1)),
        );
        (
            _mm256_srai_epi32::<11>(u_avg),
            _mm256_srai_epi32::<11>(v_avg),
        )
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn kaydir_kirp_18b<const SHIFT: u32>(v: __m256i) -> __m256i {
    let s = match SHIFT {
        1 => _mm256_srai_epi32::<1>(_mm256_add_epi32(v, _mm256_set1_epi32(1))),
        2 => _mm256_srai_epi32::<2>(_mm256_add_epi32(v, _mm256_set1_epi32(2))),
        _ => unreachable!(),
    };
    kirp::<COL_MIN, COL_MAX>(s)
}

#[target_feature(enable = "avx2")]
#[inline]
fn devrik_8x8_i32(r: [__m256i; 8]) -> [__m256i; 8] {
    let t0 = _mm256_unpacklo_epi32(r[0], r[1]);
    let t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    let t2 = _mm256_unpacklo_epi32(r[2], r[3]);
    let t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    let t4 = _mm256_unpacklo_epi32(r[4], r[5]);
    let t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    let t6 = _mm256_unpacklo_epi32(r[6], r[7]);
    let t7 = _mm256_unpackhi_epi32(r[6], r[7]);

    let u0 = _mm256_unpacklo_epi64(t0, t2);
    let u1 = _mm256_unpackhi_epi64(t0, t2);
    let u2 = _mm256_unpacklo_epi64(t1, t3);
    let u3 = _mm256_unpackhi_epi64(t1, t3);
    let u4 = _mm256_unpacklo_epi64(t4, t6);
    let u5 = _mm256_unpackhi_epi64(t4, t6);
    let u6 = _mm256_unpacklo_epi64(t5, t7);
    let u7 = _mm256_unpackhi_epi64(t5, t7);

    [
        _mm256_permute2x128_si256::<0x20>(u0, u4),
        _mm256_permute2x128_si256::<0x20>(u1, u5),
        _mm256_permute2x128_si256::<0x20>(u2, u6),
        _mm256_permute2x128_si256::<0x20>(u3, u7),
        _mm256_permute2x128_si256::<0x31>(u0, u4),
        _mm256_permute2x128_si256::<0x31>(u1, u5),
        _mm256_permute2x128_si256::<0x31>(u2, u6),
        _mm256_permute2x128_si256::<0x31>(u3, u7),
    ]
}

// ---------------------------------------------------------------------------
// Hedef `u16` düzlem piksel ekleme + `[0, 4095]` doyurmalı yazma yardımcıları
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
fn ekle_yaz_8x1(dst: &mut [u8], off: usize, r: __m256i) {
    let s = _mm256_srai_epi32::<4>(_mm256_add_epi32(r, _mm256_set1_epi32(8)));
    let s_lo = _mm256_castsi256_si128(s);
    let s_hi = _mm256_extracti128_si256::<1>(s);
    let res16 = _mm_min_epi16(
        _mm_max_epi16(_mm_packs_epi32(s_lo, s_hi), _mm_set1_epi16(-4096)),
        _mm_set1_epi16(4095),
    );
    let dst16 = load_u8x16(chunk::<u8, 16>(dst, off));
    let out = _mm_min_epi16(
        _mm_max_epi16(_mm_add_epi16(dst16, res16), _mm_setzero_si128()),
        _mm_set1_epi16(4095),
    );
    store_u8x16(chunk_mut(dst, off), out);
}

#[target_feature(enable = "avx2")]
#[inline]
fn dc_yalniz<const W: usize, const H: usize, const SHIFT: u32, const RECT2: bool>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32],
) {
    let mut dc = i64::from(coeff[0]);
    coeff[0] = 0;
    if RECT2 {
        dc = (dc * 181 + 128) >> 8;
    }
    dc = (dc * 181 + 128) >> 8;
    let rnd = (1i64 << SHIFT) >> 1;
    dc = (dc + rnd) >> SHIFT;
    dc = (dc * 181 + 128 + 2048) >> 12;
    let dc16 = dc.clamp(-4096, 4095) as i16;

    if W == 8 {
        let dc_v: __m128i = _mm_set1_epi16(dc16);
        let sifir = _mm_setzero_si128();
        let tavan = _mm_set1_epi16(4095);
        let mut y = 0;
        while y < H {
            let off = y * stride;
            let d = load_u8x16(chunk::<u8, 16>(dst, off));
            let out = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(d, dc_v), sifir), tavan);
            store_u8x16(chunk_mut(dst, off), out);
            y += 1;
        }
    } else {
        let dc_v = _mm256_set1_epi16(dc16);
        let sifir = _mm256_setzero_si256();
        let tavan = _mm256_set1_epi16(4095);
        let mut y = 0;
        while y < H {
            let taban = y * stride;
            let mut x = 0;
            while x < W {
                let off = taban + x * 2;
                let d = load_u8x32(chunk::<u8, 32>(dst, off));
                let out =
                    _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(d, dc_v), sifir), tavan);
                store_u8x32(chunk_mut(dst, off), out);
                x += 16;
            }
            y += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Büyük `IDTX` gövdeleri (`5` kayıt: `8×32`, `32×8`, `16×32`, `32×16`, `32×32`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline(never)]
fn idtx_8x32_karo(dst: &mut [u8], stride: usize, coeff: &[i32], y0: usize) {
    let bir = _mm256_set1_epi32(1);
    let mut giris = [_mm256_setzero_si256(); 8];
    let mut x = 0;
    while x < 8 {
        let v = kirp::<-16_777_216, 16_777_215>(yukle_i32x8(coeff, y0 + x * 32));
        giris[x] = kirp::<COL_MIN, COL_MAX>(_mm256_srai_epi32::<1>(_mm256_add_epi32(v, bir)));
        x += 1;
    }
    let t = devrik_8x8_i32(giris);
    let mut dy = 0;
    while dy < 8 {
        ekle_yaz_8x1(dst, (y0 + dy) * stride, _mm256_slli_epi32::<2>(t[dy]));
        dy += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn idtx_8x32(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let y_blok = if eob <= 42 {
        1
    } else if eob <= 106 {
        2
    } else {
        4
    };
    let mut yb = 0;
    while yb < y_blok {
        idtx_8x32_karo(dst, stride, coeff, yb * 8);
        yb += 1;
    }
    sifirla_katsayi::<256>(coeff);
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn idtx_32x8_karo(dst: &mut [u8], stride: usize, coeff: &[i32], xb: usize) {
    let x0 = xb * 8;
    let x_bayt = xb * 16;
    let mut giris = [_mm256_setzero_si256(); 8];
    let mut dx = 0;
    while dx < 8 {
        giris[dx] = kirp::<COL_MIN, COL_MAX>(yukle_i32x8(coeff, (x0 + dx) * 8));
        dx += 1;
    }
    let t = devrik_8x8_i32(giris);
    let mut y = 0;
    while y < 8 {
        ekle_yaz_8x1(dst, y * stride + x_bayt, _mm256_add_epi32(t[y], t[y]));
        y += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn idtx_32x8(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let x_blok = if eob <= 42 {
        1
    } else if eob <= 106 {
        2
    } else {
        4
    };
    let mut xb = 0;
    while xb < x_blok {
        idtx_32x8_karo(dst, stride, coeff, xb);
        xb += 1;
    }
    sifirla_katsayi::<256>(coeff);
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn idtx_16x32_karo(
    dst: &mut [u8],
    stride: usize,
    coeff: &[i32],
    x0: usize,
    x_bayt: usize,
    y0: usize,
) {
    let mut giris = [_mm256_setzero_si256(); 8];
    let mut dx = 0;
    while dx < 8 {
        let v = olcekle_181(yukle_i32x8(coeff, y0 + (x0 + dx) * 32));
        giris[dx] = kirp_64::<COL_MIN, COL_MAX>(mul1_64::<12>(v, 5793, 3072));
        dx += 1;
    }
    let t = devrik_8x8_i32(giris);
    let mut dy = 0;
    while dy < 8 {
        ekle_yaz_8x1(
            dst,
            (y0 + dy) * stride + x_bayt,
            _mm256_slli_epi32::<2>(t[dy]),
        );
        dy += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn idtx_16x32(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let y_blok = if eob <= 43 {
        1
    } else if eob <= 150 {
        2
    } else if eob <= 278 {
        3
    } else {
        4
    };
    let x_blok = if eob <= 35 { 1 } else { 2 };
    let mut xb = 0;
    while xb < x_blok {
        let x0 = xb * 8;
        let x_bayt = xb * 16;
        let mut yb = 0;
        while yb < y_blok {
            idtx_16x32_karo(dst, stride, coeff, x0, x_bayt, yb * 8);
            yb += 1;
        }
        xb += 1;
    }
    sifirla_katsayi::<512>(coeff);
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn idtx_32x16_karo(
    dst: &mut [u8],
    stride: usize,
    coeff: &[i32],
    x0: usize,
    x_bayt: usize,
    y0: usize,
) {
    let k5793 = _mm256_set1_epi32(5793);
    let rnd11 = _mm256_set1_epi32(1024);
    let mut giris = [_mm256_setzero_si256(); 8];
    let mut dx = 0;
    while dx < 8 {
        let v =
            kirp::<-16_777_216, 16_777_215>(olcekle_181(yukle_i32x8(coeff, y0 + (x0 + dx) * 16)));
        giris[dx] = kirp::<COL_MIN, COL_MAX>(_mm256_add_epi32(v, v));
        dx += 1;
    }
    let t = devrik_8x8_i32(giris);
    let mut dy = 0;
    while dy < 8 {
        let c = _mm256_srai_epi32::<11>(_mm256_add_epi32(_mm256_mullo_epi32(t[dy], k5793), rnd11));
        ekle_yaz_8x1(dst, (y0 + dy) * stride + x_bayt, c);
        dy += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn idtx_32x16(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let x_blok = if eob <= 43 {
        1
    } else if eob <= 150 {
        2
    } else if eob <= 278 {
        3
    } else {
        4
    };
    let y_blok = if eob <= 35 { 1 } else { 2 };
    let mut yb = 0;
    while yb < y_blok {
        let y0 = yb * 8;
        let mut xb = 0;
        while xb < x_blok {
            idtx_32x16_karo(dst, stride, coeff, xb * 8, xb * 16, y0);
            xb += 1;
        }
        yb += 1;
    }
    sifirla_katsayi::<512>(coeff);
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn idtx_32x32_karo(
    dst: &mut [u8],
    stride: usize,
    coeff: &[i32],
    x0: usize,
    x_bayt: usize,
    y0: usize,
) {
    let mut giris = [_mm256_setzero_si256(); 8];
    let mut dx = 0;
    while dx < 8 {
        giris[dx] = kirp::<COL_MIN, COL_MAX>(yukle_i32x8(coeff, y0 + (x0 + dx) * 32));
        dx += 1;
    }
    let t = devrik_8x8_i32(giris);
    let mut dy = 0;
    while dy < 8 {
        ekle_yaz_8x1(
            dst,
            (y0 + dy) * stride + x_bayt,
            _mm256_slli_epi32::<2>(t[dy]),
        );
        dy += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn idtx_32x32(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let blok = if eob <= 35 {
        1
    } else if eob <= 135 {
        2
    } else {
        4
    };
    let mut yb = 0;
    while yb < blok {
        let y0 = yb * 8;
        let mut xb = 0;
        while xb < blok {
            idtx_32x32_karo(dst, stride, coeff, xb * 8, xb * 16, y0);
            xb += 1;
        }
        yb += 1;
    }
    sifirla_katsayi::<1024>(coeff);
}

// ---------------------------------------------------------------------------
// 1B 8/16/32 noktalı DCT çekirdekleri ve `8×32` / `32×8` `DCT_DCT` gövdeleri
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx2")]
#[inline]
fn idct4<const MIN: i32, const MAX: i32>(c: [__m256i; 4]) -> [__m256i; 4] {
    let (t0, t1) = mul2_64::<8, MIN>(c[0], c[2], 181, 181, 181, -181, 128);
    let (t2, t3) = mul2_64::<12, MIN>(c[1], c[3], 1567, -3784, 3784, 1567, 2048);
    let (o0, o3) = add_sub_kirp_64::<MIN, MAX>(t0, t3);
    let (o1, o2) = add_sub_kirp_64::<MIN, MAX>(t1, t2);
    [o0, o1, o2, o3]
}

#[target_feature(enable = "avx2")]
#[inline]
fn idct8<const MIN: i32, const MAX: i32>(c: [__m256i; 8]) -> [__m256i; 8] {
    let [t0, t1, t2, t3] = idct4::<MIN, MAX>([c[0], c[2], c[4], c[6]]);
    let (t4a, t7a) = mul2_64::<12, MIN>(c[1], c[7], 799, -4017, 4017, 799, 2048);
    let (t5a, t6a) = mul2_64::<11, MIN>(c[5], c[3], 1703, -1138, 1138, 1703, 1024);
    let (t4, t5a) = add_sub_kirp_64::<MIN, MAX>(t4a, t5a);
    let (t7, t6a) = add_sub_kirp_64::<MIN, MAX>(t7a, t6a);
    let t5 = olcekle_181_ic(_mm256_sub_epi32(t6a, t5a));
    let t6 = olcekle_181_ic(_mm256_add_epi32(t6a, t5a));
    [
        kirp::<MIN, MAX>(_mm256_add_epi32(t0, t7)),
        kirp::<MIN, MAX>(_mm256_add_epi32(t1, t6)),
        kirp::<MIN, MAX>(_mm256_add_epi32(t2, t5)),
        kirp::<MIN, MAX>(_mm256_add_epi32(t3, t4)),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t3, t4)),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t2, t5)),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t1, t6)),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t0, t7)),
    ]
}

#[inline]
const fn slot_off<const PERM: bool>(idx: usize) -> usize {
    if PERM {
        ((idx >> 3) + (idx & 7) * 4) * 8
    } else {
        idx * 8
    }
}

/// `coeff` çift yuvaları (`k = 0..16`, `idx = 2 * k`) üzerinde yerinde 16 noktalı DCT — 1. aşama (tek indeksli çift yuvalar `2, 6, ..., 30`).
#[target_feature(enable = "avx2")]
#[inline(never)]
fn idct32_cift_yari_asama1<const MIN: i32, const MAX: i32, const PERM: bool>(coeff: &mut [i32]) {
    let (t8a, t15a) = mul2_64::<12, MIN>(
        yukle_i32x8(coeff, slot_off::<PERM>(2)),
        yukle_i32x8(coeff, slot_off::<PERM>(30)),
        401,
        -4076,
        4076,
        401,
        2048,
    );
    let (t9a, t14a) = mul2_64::<11, MIN>(
        yukle_i32x8(coeff, slot_off::<PERM>(18)),
        yukle_i32x8(coeff, slot_off::<PERM>(14)),
        1583,
        -1299,
        1299,
        1583,
        1024,
    );
    let (t8, t9) = add_sub_kirp_64::<MIN, MAX>(t8a, t9a);
    let (t14, t15) = sub_add_kirp_64::<MIN, MAX>(t15a, t14a);
    let (t9a, t14a) = mul2::<MIN>(t14, t9, 1567, -3784, 3784, 1567);

    let (t10a, t13a) = mul2_64::<12, MIN>(
        yukle_i32x8(coeff, slot_off::<PERM>(10)),
        yukle_i32x8(coeff, slot_off::<PERM>(22)),
        1931,
        -3612,
        3612,
        1931,
        2048,
    );
    let (t11a, t12a) = mul2_64::<12, MIN>(
        yukle_i32x8(coeff, slot_off::<PERM>(26)),
        yukle_i32x8(coeff, slot_off::<PERM>(6)),
        3920,
        -1189,
        1189,
        3920,
        2048,
    );
    let (t10, t11) = sub_add_kirp_64::<MIN, MAX>(t11a, t10a);
    let (t12, t13) = add_sub_kirp_64::<MIN, MAX>(t12a, t13a);
    let (t10a, t13a) = mul2::<MIN>(t13, t10, -3784, -1567, 1567, -3784);

    let t8a = kirp::<MIN, MAX>(_mm256_add_epi32(t8, t11));
    let t9 = kirp::<MIN, MAX>(_mm256_add_epi32(t9a, t10a));
    let t10 = kirp::<MIN, MAX>(_mm256_sub_epi32(t9a, t10a));
    let t11a = kirp::<MIN, MAX>(_mm256_sub_epi32(t8, t11));
    let t12a = kirp::<MIN, MAX>(_mm256_sub_epi32(t15, t12));
    let t13 = kirp::<MIN, MAX>(_mm256_sub_epi32(t14a, t13a));
    let t14 = kirp::<MIN, MAX>(_mm256_add_epi32(t14a, t13a));
    let t15a = kirp::<MIN, MAX>(_mm256_add_epi32(t15, t12));

    let t10a = olcekle_181_ic(_mm256_sub_epi32(t13, t10));
    let t13a = olcekle_181_ic(_mm256_add_epi32(t13, t10));
    let t11 = olcekle_181_ic(_mm256_sub_epi32(t12a, t11a));
    let t12 = olcekle_181_ic(_mm256_add_epi32(t12a, t11a));

    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(2)), t8a);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(6)), t9);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(10)), t10a);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(14)), t11);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(18)), t12);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(22)), t13a);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(26)), t14);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(30)), t15a);
}

/// `coeff` çift yuvaları (`k = 0..16`, `idx = 2 * k`) üzerinde yerinde 16 noktalı DCT — 2. aşama (`idct8` + kelebek birleştirme).
#[target_feature(enable = "avx2")]
#[inline(never)]
fn idct32_cift_yari_asama2<const MIN: i32, const MAX: i32, const PERM: bool>(coeff: &mut [i32]) {
    let e = idct8::<MIN, MAX>([
        yukle_i32x8(coeff, slot_off::<PERM>(0)),
        yukle_i32x8(coeff, slot_off::<PERM>(4)),
        yukle_i32x8(coeff, slot_off::<PERM>(8)),
        yukle_i32x8(coeff, slot_off::<PERM>(12)),
        yukle_i32x8(coeff, slot_off::<PERM>(16)),
        yukle_i32x8(coeff, slot_off::<PERM>(20)),
        yukle_i32x8(coeff, slot_off::<PERM>(24)),
        yukle_i32x8(coeff, slot_off::<PERM>(28)),
    ]);
    let odd = [
        yukle_i32x8(coeff, slot_off::<PERM>(2)),
        yukle_i32x8(coeff, slot_off::<PERM>(6)),
        yukle_i32x8(coeff, slot_off::<PERM>(10)),
        yukle_i32x8(coeff, slot_off::<PERM>(14)),
        yukle_i32x8(coeff, slot_off::<PERM>(18)),
        yukle_i32x8(coeff, slot_off::<PERM>(22)),
        yukle_i32x8(coeff, slot_off::<PERM>(26)),
        yukle_i32x8(coeff, slot_off::<PERM>(30)),
    ];
    let mut i = 0;
    while i < 8 {
        store_i32x8(
            chunk_mut(coeff, slot_off::<PERM>(i * 2)),
            kirp::<MIN, MAX>(_mm256_add_epi32(e[i], odd[7 - i])),
        );
        store_i32x8(
            chunk_mut(coeff, slot_off::<PERM>((15 - i) * 2)),
            kirp::<MIN, MAX>(_mm256_sub_epi32(e[i], odd[7 - i])),
        );
        i += 1;
    }
}

/// 32 noktalı DCT tek yarı 1. aşama: ilk iki kelebek katmanını işleyip 16 tek yuvaya yazar.
#[target_feature(enable = "avx2")]
#[inline(never)]
fn idct32_tek_yari_asama1<const MIN: i32, const MAX: i32, const PERM: bool>(coeff: &mut [i32]) {
    // Grup A+B: (1, 31, 17, 15) ve (9, 23, 25, 7)
    let (t16a, t31a) = mul2_64::<12, MIN>(
        yukle_i32x8(coeff, slot_off::<PERM>(1)),
        yukle_i32x8(coeff, slot_off::<PERM>(31)),
        201,
        -4091,
        4091,
        201,
        2048,
    );
    let (t17a, t30a) = mul2_64::<12, MIN>(
        yukle_i32x8(coeff, slot_off::<PERM>(17)),
        yukle_i32x8(coeff, slot_off::<PERM>(15)),
        3035,
        -2751,
        2751,
        3035,
        2048,
    );
    let (t16, t17) = add_sub_kirp_64::<MIN, MAX>(t16a, t17a);
    let (t30, t31) = sub_add_kirp_64::<MIN, MAX>(t31a, t30a);
    let (t17a, t30a) = mul2::<MIN>(t30, t17, 799, -4017, 4017, 799);

    let (t18a, t29a) = mul2_64::<12, MIN>(
        yukle_i32x8(coeff, slot_off::<PERM>(9)),
        yukle_i32x8(coeff, slot_off::<PERM>(23)),
        1751,
        -3703,
        3703,
        1751,
        2048,
    );
    let (t19a, t28a) = mul2_64::<12, MIN>(
        yukle_i32x8(coeff, slot_off::<PERM>(25)),
        yukle_i32x8(coeff, slot_off::<PERM>(7)),
        3857,
        -1380,
        1380,
        3857,
        2048,
    );
    let (t18, t19) = sub_add_kirp_64::<MIN, MAX>(t19a, t18a);
    let (t28, t29) = add_sub_kirp_64::<MIN, MAX>(t28a, t29a);
    let (t18a, t29a) = mul2::<MIN>(t29, t18, -4017, -799, 799, -4017);

    store_i32x8(
        chunk_mut(coeff, slot_off::<PERM>(1)),
        kirp::<MIN, MAX>(_mm256_add_epi32(t16, t19)),
    );
    store_i32x8(
        chunk_mut(coeff, slot_off::<PERM>(31)),
        kirp::<MIN, MAX>(_mm256_add_epi32(t17a, t18a)),
    );
    store_i32x8(
        chunk_mut(coeff, slot_off::<PERM>(17)),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t17a, t18a)),
    );
    store_i32x8(
        chunk_mut(coeff, slot_off::<PERM>(15)),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t16, t19)),
    );
    store_i32x8(
        chunk_mut(coeff, slot_off::<PERM>(9)),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t31, t28)),
    );
    store_i32x8(
        chunk_mut(coeff, slot_off::<PERM>(23)),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t30a, t29a)),
    );
    store_i32x8(
        chunk_mut(coeff, slot_off::<PERM>(25)),
        kirp::<MIN, MAX>(_mm256_add_epi32(t30a, t29a)),
    );
    store_i32x8(
        chunk_mut(coeff, slot_off::<PERM>(7)),
        kirp::<MIN, MAX>(_mm256_add_epi32(t31, t28)),
    );

    // Grup C+D: (5, 27, 21, 11) ve (13, 19, 29, 3)
    let (t20a, t27a) = mul2_64::<12, MIN>(
        yukle_i32x8(coeff, slot_off::<PERM>(5)),
        yukle_i32x8(coeff, slot_off::<PERM>(27)),
        995,
        -3973,
        3973,
        995,
        2048,
    );
    let (t21a, t26a) = mul2_64::<12, MIN>(
        yukle_i32x8(coeff, slot_off::<PERM>(21)),
        yukle_i32x8(coeff, slot_off::<PERM>(11)),
        3513,
        -2106,
        2106,
        3513,
        2048,
    );
    let (t20, t21) = add_sub_kirp_64::<MIN, MAX>(t20a, t21a);
    let (t26, t27) = sub_add_kirp_64::<MIN, MAX>(t27a, t26a);
    let (t21a, t26a) = mul2::<MIN>(t26, t21, 3406, -2276, 2276, 3406);

    let (t22a, t25a) = mul2_64::<11, MIN>(
        yukle_i32x8(coeff, slot_off::<PERM>(13)),
        yukle_i32x8(coeff, slot_off::<PERM>(19)),
        1220,
        -1645,
        1645,
        1220,
        1024,
    );
    let (t23a, t24a) = mul2_64::<12, MIN>(
        yukle_i32x8(coeff, slot_off::<PERM>(29)),
        yukle_i32x8(coeff, slot_off::<PERM>(3)),
        4052,
        -601,
        601,
        4052,
        2048,
    );
    let (t22, t23) = sub_add_kirp_64::<MIN, MAX>(t23a, t22a);
    let (t24, t25) = add_sub_kirp_64::<MIN, MAX>(t24a, t25a);
    let (t22a, t25a) = mul2::<MIN>(t25, t22, -2276, -3406, 3406, -2276);

    store_i32x8(
        chunk_mut(coeff, slot_off::<PERM>(5)),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t23, t20)),
    );
    store_i32x8(
        chunk_mut(coeff, slot_off::<PERM>(27)),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t22a, t21a)),
    );
    store_i32x8(
        chunk_mut(coeff, slot_off::<PERM>(21)),
        kirp::<MIN, MAX>(_mm256_add_epi32(t22a, t21a)),
    );
    store_i32x8(
        chunk_mut(coeff, slot_off::<PERM>(11)),
        kirp::<MIN, MAX>(_mm256_add_epi32(t23, t20)),
    );
    store_i32x8(
        chunk_mut(coeff, slot_off::<PERM>(13)),
        kirp::<MIN, MAX>(_mm256_add_epi32(t24, t27)),
    );
    store_i32x8(
        chunk_mut(coeff, slot_off::<PERM>(19)),
        kirp::<MIN, MAX>(_mm256_add_epi32(t25a, t26a)),
    );
    store_i32x8(
        chunk_mut(coeff, slot_off::<PERM>(29)),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t25a, t26a)),
    );
    store_i32x8(
        chunk_mut(coeff, slot_off::<PERM>(3)),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t24, t27)),
    );
}

/// 32 noktalı DCT tek yarı 2. aşama: son kelebekleri işleyip `1, 3, 5, ..., 31` tek yuvalara yazar.
#[target_feature(enable = "avx2")]
#[inline(never)]
fn idct32_tek_yari_asama2<const MIN: i32, const MAX: i32, const PERM: bool>(coeff: &mut [i32]) {
    let t16a = yukle_i32x8(coeff, slot_off::<PERM>(1));
    let t17 = yukle_i32x8(coeff, slot_off::<PERM>(31));
    let t18 = yukle_i32x8(coeff, slot_off::<PERM>(17));
    let t19a = yukle_i32x8(coeff, slot_off::<PERM>(15));
    let t28a = yukle_i32x8(coeff, slot_off::<PERM>(9));
    let t29 = yukle_i32x8(coeff, slot_off::<PERM>(23));
    let t30 = yukle_i32x8(coeff, slot_off::<PERM>(25));
    let t31a = yukle_i32x8(coeff, slot_off::<PERM>(7));
    let t20a = yukle_i32x8(coeff, slot_off::<PERM>(5));
    let t21 = yukle_i32x8(coeff, slot_off::<PERM>(27));
    let t22 = yukle_i32x8(coeff, slot_off::<PERM>(21));
    let t23a = yukle_i32x8(coeff, slot_off::<PERM>(11));
    let t24a = yukle_i32x8(coeff, slot_off::<PERM>(13));
    let t25 = yukle_i32x8(coeff, slot_off::<PERM>(19));
    let t26 = yukle_i32x8(coeff, slot_off::<PERM>(29));
    let t27a = yukle_i32x8(coeff, slot_off::<PERM>(3));

    let (t18a, t29a) = mul2::<MIN>(t29, t18, 1567, -3784, 3784, 1567);
    let (t19, t28) = mul2::<MIN>(t28a, t19a, 1567, -3784, 3784, 1567);
    let (t20, t27) = mul2::<MIN>(t27a, t20a, -3784, -1567, 1567, -3784);
    let (t21a, t26a) = mul2::<MIN>(t26, t21, -3784, -1567, 1567, -3784);

    let t16 = kirp::<MIN, MAX>(_mm256_add_epi32(t16a, t23a));
    let t17a = kirp::<MIN, MAX>(_mm256_add_epi32(t17, t22));
    let t18 = kirp::<MIN, MAX>(_mm256_add_epi32(t18a, t21a));
    let t19a = kirp::<MIN, MAX>(_mm256_add_epi32(t19, t20));
    let t20a = kirp::<MIN, MAX>(_mm256_sub_epi32(t19, t20));
    let t21 = kirp::<MIN, MAX>(_mm256_sub_epi32(t18a, t21a));
    let t22a = kirp::<MIN, MAX>(_mm256_sub_epi32(t17, t22));
    let t23 = kirp::<MIN, MAX>(_mm256_sub_epi32(t16a, t23a));
    let t24 = kirp::<MIN, MAX>(_mm256_sub_epi32(t31a, t24a));
    let t25a = kirp::<MIN, MAX>(_mm256_sub_epi32(t30, t25));
    let t26 = kirp::<MIN, MAX>(_mm256_sub_epi32(t29a, t26a));
    let t27a = kirp::<MIN, MAX>(_mm256_sub_epi32(t28, t27));
    let t28a = kirp::<MIN, MAX>(_mm256_add_epi32(t28, t27));
    let t29 = kirp::<MIN, MAX>(_mm256_add_epi32(t29a, t26a));
    let t30a = kirp::<MIN, MAX>(_mm256_add_epi32(t30, t25));
    let t31 = kirp::<MIN, MAX>(_mm256_add_epi32(t31a, t24a));

    let t20 = olcekle_181_ic(_mm256_sub_epi32(t27a, t20a));
    let t27 = olcekle_181_ic(_mm256_add_epi32(t27a, t20a));
    let t21a = olcekle_181_ic(_mm256_sub_epi32(t26, t21));
    let t26a = olcekle_181_ic(_mm256_add_epi32(t26, t21));
    let t22 = olcekle_181_ic(_mm256_sub_epi32(t25a, t22a));
    let t25 = olcekle_181_ic(_mm256_add_epi32(t25a, t22a));
    let t23a = olcekle_181_ic(_mm256_sub_epi32(t24, t23));
    let t24a = olcekle_181_ic(_mm256_add_epi32(t24, t23));

    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(1)), t16);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(3)), t17a);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(5)), t18);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(7)), t19a);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(9)), t20);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(11)), t21a);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(13)), t22);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(15)), t23a);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(17)), t24a);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(19)), t25);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(21)), t26a);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(23)), t27);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(25)), t28a);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(27)), t29);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(29)), t30a);
    store_i32x8(chunk_mut(coeff, slot_off::<PERM>(31)), t31);
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn dct_8x32_gecis1(coeff: &mut [i32], y_blok: usize) {
    let mut yb = 0;
    while yb < y_blok {
        let y0 = yb * 8;
        let g: [__m256i; 8] = core::array::from_fn(|x| yukle_i32x8(coeff, y0 + x * 32));
        let p1 = idct8::<ROW_MIN, ROW_MAX>(g);
        let k: [__m256i; 8] = core::array::from_fn(|x| kaydir_kirp_18b::<2>(p1[x]));
        let t = devrik_8x8_i32(k);
        let mut dy = 0;
        while dy < 8 {
            store_i32x8(chunk_mut(coeff, y0 + dy * 32), t[dy]);
            dy += 1;
        }
        yb += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn dct_8x32(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<8, 32, 2, false>(dst, stride, coeff);
        return;
    }
    let y_blok = if eob <= 42 {
        1
    } else if eob <= 106 {
        2
    } else {
        4
    };
    dct_8x32_gecis1(coeff, y_blok);
    idct32_tek_yari_asama1::<COL_MIN, COL_MAX, true>(coeff);
    idct32_tek_yari_asama2::<COL_MIN, COL_MAX, true>(coeff);
    idct32_cift_yari_asama1::<COL_MIN, COL_MAX, true>(coeff);
    idct32_cift_yari_asama2::<COL_MIN, COL_MAX, true>(coeff);

    let mut i = 0;
    while i < 16 {
        let e_i = yukle_i32x8(coeff, slot_off::<true>(i * 2));
        let odd_i = yukle_i32x8(coeff, slot_off::<true>((15 - i) * 2 + 1));
        let ust = kirp::<COL_MIN, COL_MAX>(_mm256_add_epi32(e_i, odd_i));
        let alt = kirp::<COL_MIN, COL_MAX>(_mm256_sub_epi32(e_i, odd_i));
        ekle_yaz_8x1(dst, i * stride, ust);
        ekle_yaz_8x1(dst, (31 - i) * stride, alt);
        i += 1;
    }
    sifirla_katsayi::<256>(coeff);
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn dct_32x8_gecis2_karo(dst: &mut [u8], stride: usize, coeff: &[i32], xb: usize) {
    let mut karo = [_mm256_setzero_si256(); 8];
    let mut k = 0;
    while k < 8 {
        let idx = match xb {
            0 => k,
            1 => 8 + k,
            2 => 15 - k,
            _ => 7 - k,
        };
        let e_k = yukle_i32x8(coeff, idx * 16);
        let odd_k = yukle_i32x8(coeff, ((15 - idx) * 2 + 1) * 8);
        let v = if xb < 2 {
            _mm256_add_epi32(e_k, odd_k)
        } else {
            _mm256_sub_epi32(e_k, odd_k)
        };
        karo[k] = kaydir_kirp_18b::<2>(kirp::<ROW_MIN, ROW_MAX>(v));
        k += 1;
    }
    let c = idct8::<COL_MIN, COL_MAX>(devrik_8x8_i32(karo));
    let x_bayt = xb * 16;
    let mut y = 0;
    while y < 8 {
        ekle_yaz_8x1(dst, y * stride + x_bayt, c[y]);
        y += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn dct_32x8(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<32, 8, 2, false>(dst, stride, coeff);
        return;
    }
    idct32_tek_yari_asama1::<ROW_MIN, ROW_MAX, false>(coeff);
    idct32_tek_yari_asama2::<ROW_MIN, ROW_MAX, false>(coeff);
    idct32_cift_yari_asama1::<ROW_MIN, ROW_MAX, false>(coeff);
    idct32_cift_yari_asama2::<ROW_MIN, ROW_MAX, false>(coeff);
    dct_32x8_gecis2_karo(dst, stride, coeff, 0);
    dct_32x8_gecis2_karo(dst, stride, coeff, 1);
    dct_32x8_gecis2_karo(dst, stride, coeff, 2);
    dct_32x8_gecis2_karo(dst, stride, coeff, 3);
    sifirla_katsayi::<256>(coeff);
}

// ---------------------------------------------------------------------------
// Büyük `16×32`..`64×64` `DCT_DCT` (`8` kayıt) — iş parçacığı yerel ara tampon
// (`< 0x400` yığın çerçevesi, `coeff[sw*sh..]` koruma, tam 64-bit doyurma paritesi)
// ---------------------------------------------------------------------------

struct BuyukAra {
    /// Sütun-şerit düzenli `W × sh` ara matris: `(xb * sh + y) * 8` (`en çok 64 × 32 = 2048`).
    mat: [i32; 2048],
    /// Tek 1B şerit çalışma alanı (`64 × 8 = 512`).
    serit: [i32; 512],
    /// 64 noktalı DCT tek yarı giriş kopyası (`16 × 8 = 128`).
    tek: [i32; 128],
}

#[cfg(feature = "std")]
std::thread_local! {
    static BUYUK_ARA: RefCell<BuyukAra> = const {
        RefCell::new(BuyukAra {
            mat: [0i32; 2048],
            serit: [0i32; 512],
            tek: [0i32; 128],
        })
    };
}

#[inline]
fn with_buyuk_ara(f: impl FnOnce(&mut BuyukAra)) {
    #[cfg(feature = "std")]
    {
        BUYUK_ARA.with_borrow_mut(f);
    }
    #[cfg(not(feature = "std"))]
    {
        let mut ara = alloc::boxed::Box::new(BuyukAra {
            mat: [0i32; 2048],
            serit: [0i32; 512],
            tek: [0i32; 128],
        });
        f(&mut ara);
    }
}

/// `serit[0..128]` (`16 × 8` `i32`) üzerinde yerinde 16 noktalı 1B ters DCT.
#[target_feature(enable = "avx2")]
#[inline(never)]
fn idct16_serit<const MIN: i32, const MAX: i32>(serit: &mut [i32]) {
    let e = idct8::<MIN, MAX>([
        yukle_i32x8(serit, 0),
        yukle_i32x8(serit, 2 * 8),
        yukle_i32x8(serit, 4 * 8),
        yukle_i32x8(serit, 6 * 8),
        yukle_i32x8(serit, 8 * 8),
        yukle_i32x8(serit, 10 * 8),
        yukle_i32x8(serit, 12 * 8),
        yukle_i32x8(serit, 14 * 8),
    ]);

    let (t8a, t15a) = mul2_64::<12, MIN>(
        yukle_i32x8(serit, 8),
        yukle_i32x8(serit, 15 * 8),
        401,
        -4076,
        4076,
        401,
        2048,
    );
    let (t9a, t14a) = mul2_64::<11, MIN>(
        yukle_i32x8(serit, 9 * 8),
        yukle_i32x8(serit, 7 * 8),
        1583,
        -1299,
        1299,
        1583,
        1024,
    );
    let (t8, t9) = add_sub_kirp_64::<MIN, MAX>(t8a, t9a);
    let (t14, t15) = sub_add_kirp_64::<MIN, MAX>(t15a, t14a);
    let (t9a, t14a) = mul2::<MIN>(t14, t9, 1567, -3784, 3784, 1567);

    let (t10a, t13a) = mul2_64::<12, MIN>(
        yukle_i32x8(serit, 5 * 8),
        yukle_i32x8(serit, 11 * 8),
        1931,
        -3612,
        3612,
        1931,
        2048,
    );
    let (t11a, t12a) = mul2_64::<12, MIN>(
        yukle_i32x8(serit, 13 * 8),
        yukle_i32x8(serit, 3 * 8),
        3920,
        -1189,
        1189,
        3920,
        2048,
    );
    let (t10, t11) = sub_add_kirp_64::<MIN, MAX>(t11a, t10a);
    let (t12, t13) = add_sub_kirp_64::<MIN, MAX>(t12a, t13a);
    let (t10a, t13a) = mul2::<MIN>(t13, t10, -3784, -1567, 1567, -3784);

    let t8a = kirp::<MIN, MAX>(_mm256_add_epi32(t8, t11));
    let t9 = kirp::<MIN, MAX>(_mm256_add_epi32(t9a, t10a));
    let t10 = kirp::<MIN, MAX>(_mm256_sub_epi32(t9a, t10a));
    let t11a = kirp::<MIN, MAX>(_mm256_sub_epi32(t8, t11));
    let t12a = kirp::<MIN, MAX>(_mm256_sub_epi32(t15, t12));
    let t13 = kirp::<MIN, MAX>(_mm256_sub_epi32(t14a, t13a));
    let t14 = kirp::<MIN, MAX>(_mm256_add_epi32(t14a, t13a));
    let t15a = kirp::<MIN, MAX>(_mm256_add_epi32(t15, t12));

    let t10a = olcekle_181_ic(_mm256_sub_epi32(t13, t10));
    let t13a = olcekle_181_ic(_mm256_add_epi32(t13, t10));
    let t11 = olcekle_181_ic(_mm256_sub_epi32(t12a, t11a));
    let t12 = olcekle_181_ic(_mm256_add_epi32(t12a, t11a));

    let odd = [t8a, t9, t10a, t11, t12, t13a, t14, t15a];
    let mut i = 0;
    while i < 8 {
        let o = odd[7 - i];
        store_i32x8(
            chunk_mut(serit, i * 8),
            kirp::<MIN, MAX>(_mm256_add_epi32(e[i], o)),
        );
        store_i32x8(
            chunk_mut(serit, (15 - i) * 8),
            kirp::<MIN, MAX>(_mm256_sub_epi32(e[i], o)),
        );
        i += 1;
    }
}

/// `serit[0..256]` girişinden 32 noktalı 1B ters DCT hesaplar, çıktıyı `serit[256..512]`'ye yazar.
#[target_feature(enable = "avx2")]
#[inline(never)]
fn idct32_serit<const MIN: i32, const MAX: i32>(serit: &mut [i32]) {
    idct32_tek_yari_asama1::<MIN, MAX, false>(&mut serit[..256]);
    idct32_tek_yari_asama2::<MIN, MAX, false>(&mut serit[..256]);
    idct32_cift_yari_asama1::<MIN, MAX, false>(&mut serit[..256]);
    idct32_cift_yari_asama2::<MIN, MAX, false>(&mut serit[..256]);
    let mut i = 0;
    while i < 16 {
        let e_i = yukle_i32x8(serit, i * 16);
        let odd_i = yukle_i32x8(serit, ((15 - i) * 2 + 1) * 8);
        store_i32x8(
            chunk_mut(serit, 256 + i * 8),
            kirp::<MIN, MAX>(_mm256_add_epi32(e_i, odd_i)),
        );
        store_i32x8(
            chunk_mut(serit, 256 + (31 - i) * 8),
            kirp::<MIN, MAX>(_mm256_sub_epi32(e_i, odd_i)),
        );
        i += 1;
    }
}

/// `inv_dct8_1d_internal(..., tx64 = true)`: `serit[0, 8, 16, 24]` → `serit[256..320]`.
#[target_feature(enable = "avx2")]
#[inline(never)]
fn idct8_tx64_serit<const MIN: i32, const MAX: i32>(serit: &mut [i32]) {
    let t0 = mul1_64_min::<8, MIN>(yukle_i32x8(serit, 0), 181, 128);
    let in1_4 = yukle_i32x8(serit, 16 * 8);
    let t2 = mul1_64_min::<12, MIN>(in1_4, 1567, 2048);
    let t3 = mul1_64_min::<12, MIN>(in1_4, 3784, 2048);
    let (e4_0, e4_3) = add_sub_kirp_64::<MIN, MAX>(t0, t3);
    let (e4_1, e4_2) = add_sub_kirp_64::<MIN, MAX>(t0, t2);

    let in1_8 = yukle_i32x8(serit, 8 * 8);
    let in3_8 = yukle_i32x8(serit, 24 * 8);
    let t4a = mul1_64_min::<12, MIN>(in1_8, 799, 2048);
    let t5a = mul1_64_min::<12, MIN>(in3_8, -2276, 2048);
    let t6a = mul1_64_min::<12, MIN>(in3_8, 3406, 2048);
    let t7a = mul1_64_min::<12, MIN>(in1_8, 4017, 2048);
    let (t4, t5a) = add_sub_kirp_64::<MIN, MAX>(t4a, t5a);
    let (t7, t6a) = add_sub_kirp_64::<MIN, MAX>(t7a, t6a);
    let t5 = olcekle_181_ic(_mm256_sub_epi32(t6a, t5a));
    let t6 = olcekle_181_ic(_mm256_add_epi32(t6a, t5a));

    store_i32x8(
        chunk_mut(serit, 256),
        kirp::<MIN, MAX>(_mm256_add_epi32(e4_0, t7)),
    );
    store_i32x8(
        chunk_mut(serit, 256 + 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(e4_1, t6)),
    );
    store_i32x8(
        chunk_mut(serit, 256 + 2 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(e4_2, t5)),
    );
    store_i32x8(
        chunk_mut(serit, 256 + 3 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(e4_3, t4)),
    );
    store_i32x8(
        chunk_mut(serit, 256 + 4 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(e4_3, t4)),
    );
    store_i32x8(
        chunk_mut(serit, 256 + 5 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(e4_2, t5)),
    );
    store_i32x8(
        chunk_mut(serit, 256 + 6 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(e4_1, t6)),
    );
    store_i32x8(
        chunk_mut(serit, 256 + 7 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(e4_0, t7)),
    );
}

/// `inv_dct16_1d_internal(..., tx64 = true)` tek yarı + birleştirme:
/// `serit[4, 12, 20, 28]` ve `serit[256..320]` → `serit[256..384]`.
#[target_feature(enable = "avx2")]
#[inline(never)]
fn idct16_tx64_serit<const MIN: i32, const MAX: i32>(serit: &mut [i32]) {
    idct8_tx64_serit::<MIN, MAX>(serit);

    let in1 = yukle_i32x8(serit, 4 * 8);
    let in3 = yukle_i32x8(serit, 12 * 8);
    let in5 = yukle_i32x8(serit, 20 * 8);
    let in7 = yukle_i32x8(serit, 28 * 8);
    let t8a = mul1_64_min::<12, MIN>(in1, 401, 2048);
    let t9a = mul1_64_min::<12, MIN>(in7, -2598, 2048);
    let t10a = mul1_64_min::<12, MIN>(in5, 1931, 2048);
    let t11a = mul1_64_min::<12, MIN>(in3, -1189, 2048);
    let t12a = mul1_64_min::<12, MIN>(in3, 3920, 2048);
    let t13a = mul1_64_min::<12, MIN>(in5, 3612, 2048);
    let t14a = mul1_64_min::<12, MIN>(in7, 3166, 2048);
    let t15a = mul1_64_min::<12, MIN>(in1, 4076, 2048);

    let (t8, t9) = add_sub_kirp_64::<MIN, MAX>(t8a, t9a);
    let (t10, t11) = sub_add_kirp_64::<MIN, MAX>(t11a, t10a);
    let (t12, t13) = add_sub_kirp_64::<MIN, MAX>(t12a, t13a);
    let (t14, t15) = sub_add_kirp_64::<MIN, MAX>(t15a, t14a);

    let (t9a, t14a) = mul2::<MIN>(t14, t9, 1567, -3784, 3784, 1567);
    let (t10a, t13a) = mul2::<MIN>(t13, t10, -3784, -1567, 1567, -3784);

    let t8a = kirp::<MIN, MAX>(_mm256_add_epi32(t8, t11));
    let t9 = kirp::<MIN, MAX>(_mm256_add_epi32(t9a, t10a));
    let t10 = kirp::<MIN, MAX>(_mm256_sub_epi32(t9a, t10a));
    let t11a = kirp::<MIN, MAX>(_mm256_sub_epi32(t8, t11));
    let t12a = kirp::<MIN, MAX>(_mm256_sub_epi32(t15, t12));
    let t13 = kirp::<MIN, MAX>(_mm256_sub_epi32(t14a, t13a));
    let t14 = kirp::<MIN, MAX>(_mm256_add_epi32(t14a, t13a));
    let t15a = kirp::<MIN, MAX>(_mm256_add_epi32(t15, t12));

    let t10a = olcekle_181_ic(_mm256_sub_epi32(t13, t10));
    let t13a = olcekle_181_ic(_mm256_add_epi32(t13, t10));
    let t11 = olcekle_181_ic(_mm256_sub_epi32(t12a, t11a));
    let t12 = olcekle_181_ic(_mm256_add_epi32(t12a, t11a));

    let odd = [t8a, t9, t10a, t11, t12, t13a, t14, t15a];
    let mut i = 0;
    while i < 8 {
        let ev = yukle_i32x8(serit, 256 + i * 8);
        let o = odd[7 - i];
        store_i32x8(
            chunk_mut(serit, 256 + i * 8),
            kirp::<MIN, MAX>(_mm256_add_epi32(ev, o)),
        );
        store_i32x8(
            chunk_mut(serit, 256 + (15 - i) * 8),
            kirp::<MIN, MAX>(_mm256_sub_epi32(ev, o)),
        );
        i += 1;
    }
}

/// `inv_dct32_1d_internal(..., tx64 = true)` tek yarı 1. aşama:
/// `serit[2, 6, 10, ..., 30]` → ara 16 vektörü `serit[384..512]`'ye yazar.
#[target_feature(enable = "avx2")]
#[inline(never)]
fn idct32_tx64_tek_asama1<const MIN: i32, const MAX: i32>(serit: &mut [i32]) {
    let in1 = yukle_i32x8(serit, 2 * 8);
    let in15 = yukle_i32x8(serit, 30 * 8);
    let in9 = yukle_i32x8(serit, 18 * 8);
    let in7 = yukle_i32x8(serit, 14 * 8);
    let t16a = mul1_64_min::<12, MIN>(in1, 201, 2048);
    let t17a = mul1_64_min::<12, MIN>(in15, -2751, 2048);
    let t18a = mul1_64_min::<12, MIN>(in9, 1751, 2048);
    let t19a = mul1_64_min::<12, MIN>(in7, -1380, 2048);
    let t28a = mul1_64_min::<12, MIN>(in7, 3857, 2048);
    let t29a = mul1_64_min::<12, MIN>(in9, 3703, 2048);
    let t30a = mul1_64_min::<12, MIN>(in15, 3035, 2048);
    let t31a = mul1_64_min::<12, MIN>(in1, 4091, 2048);

    let (t16, t17) = add_sub_kirp_64::<MIN, MAX>(t16a, t17a);
    let (t18, t19) = sub_add_kirp_64::<MIN, MAX>(t19a, t18a);
    let (t28, t29) = add_sub_kirp_64::<MIN, MAX>(t28a, t29a);
    let (t30, t31) = sub_add_kirp_64::<MIN, MAX>(t31a, t30a);

    let (t17a, t30a) = mul2::<MIN>(t30, t17, 799, -4017, 4017, 799);
    let (t18a, t29a) = mul2::<MIN>(t29, t18, -4017, -799, 799, -4017);

    store_i32x8(
        chunk_mut(serit, 384),
        kirp::<MIN, MAX>(_mm256_add_epi32(t16, t19)),
    );
    store_i32x8(
        chunk_mut(serit, 384 + 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t17a, t18a)),
    );
    store_i32x8(
        chunk_mut(serit, 384 + 2 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t17a, t18a)),
    );
    store_i32x8(
        chunk_mut(serit, 384 + 3 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t16, t19)),
    );
    store_i32x8(
        chunk_mut(serit, 384 + 12 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t31, t28)),
    );
    store_i32x8(
        chunk_mut(serit, 384 + 13 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t30a, t29a)),
    );
    store_i32x8(
        chunk_mut(serit, 384 + 14 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t30a, t29a)),
    );
    store_i32x8(
        chunk_mut(serit, 384 + 15 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t31, t28)),
    );

    let in5 = yukle_i32x8(serit, 10 * 8);
    let in11 = yukle_i32x8(serit, 22 * 8);
    let in13 = yukle_i32x8(serit, 26 * 8);
    let in3 = yukle_i32x8(serit, 6 * 8);
    let t20a = mul1_64_min::<12, MIN>(in5, 995, 2048);
    let t21a = mul1_64_min::<12, MIN>(in11, -2106, 2048);
    let t22a = mul1_64_min::<12, MIN>(in13, 2440, 2048);
    let t23a = mul1_64_min::<12, MIN>(in3, -601, 2048);
    let t24a = mul1_64_min::<12, MIN>(in3, 4052, 2048);
    let t25a = mul1_64_min::<12, MIN>(in13, 3290, 2048);
    let t26a = mul1_64_min::<12, MIN>(in11, 3513, 2048);
    let t27a = mul1_64_min::<12, MIN>(in5, 3973, 2048);

    let (t20, t21) = add_sub_kirp_64::<MIN, MAX>(t20a, t21a);
    let (t22, t23) = sub_add_kirp_64::<MIN, MAX>(t23a, t22a);
    let (t24, t25) = add_sub_kirp_64::<MIN, MAX>(t24a, t25a);
    let (t26, t27) = sub_add_kirp_64::<MIN, MAX>(t27a, t26a);

    let (t21a, t26a) = mul2::<MIN>(t26, t21, 3406, -2276, 2276, 3406);
    let (t22a, t25a) = mul2::<MIN>(t25, t22, -2276, -3406, 3406, -2276);

    store_i32x8(
        chunk_mut(serit, 384 + 4 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t23, t20)),
    );
    store_i32x8(
        chunk_mut(serit, 384 + 5 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t22a, t21a)),
    );
    store_i32x8(
        chunk_mut(serit, 384 + 6 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t22a, t21a)),
    );
    store_i32x8(
        chunk_mut(serit, 384 + 7 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t23, t20)),
    );
    store_i32x8(
        chunk_mut(serit, 384 + 8 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t24, t27)),
    );
    store_i32x8(
        chunk_mut(serit, 384 + 9 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t25a, t26a)),
    );
    store_i32x8(
        chunk_mut(serit, 384 + 10 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t25a, t26a)),
    );
    store_i32x8(
        chunk_mut(serit, 384 + 11 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t24, t27)),
    );
}

/// `inv_dct32_1d_internal(..., tx64 = true)` tek yarı 2. aşama + birleştirme:
/// `serit[256..384]` (çift 16) ve `serit[384..512]` (tek 16) → 32 çıktı `serit[256..512]`.
#[target_feature(enable = "avx2")]
#[inline(never)]
fn idct32_tx64_tek_asama2<const MIN: i32, const MAX: i32>(serit: &mut [i32]) {
    let t16a = yukle_i32x8(serit, 384);
    let t17 = yukle_i32x8(serit, 384 + 8);
    let t18 = yukle_i32x8(serit, 384 + 2 * 8);
    let t19a = yukle_i32x8(serit, 384 + 3 * 8);
    let t20a = yukle_i32x8(serit, 384 + 4 * 8);
    let t21 = yukle_i32x8(serit, 384 + 5 * 8);
    let t22 = yukle_i32x8(serit, 384 + 6 * 8);
    let t23a = yukle_i32x8(serit, 384 + 7 * 8);
    let t24a = yukle_i32x8(serit, 384 + 8 * 8);
    let t25 = yukle_i32x8(serit, 384 + 9 * 8);
    let t26 = yukle_i32x8(serit, 384 + 10 * 8);
    let t27a = yukle_i32x8(serit, 384 + 11 * 8);
    let t28a = yukle_i32x8(serit, 384 + 12 * 8);
    let t29 = yukle_i32x8(serit, 384 + 13 * 8);
    let t30 = yukle_i32x8(serit, 384 + 14 * 8);
    let t31a = yukle_i32x8(serit, 384 + 15 * 8);

    let (t18a, t29a) = mul2::<MIN>(t29, t18, 1567, -3784, 3784, 1567);
    let (t19, t28) = mul2::<MIN>(t28a, t19a, 1567, -3784, 3784, 1567);
    let (t20, t27) = mul2::<MIN>(t27a, t20a, -3784, -1567, 1567, -3784);
    let (t21a, t26a) = mul2::<MIN>(t26, t21, -3784, -1567, 1567, -3784);

    let t16 = kirp::<MIN, MAX>(_mm256_add_epi32(t16a, t23a));
    let t17a = kirp::<MIN, MAX>(_mm256_add_epi32(t17, t22));
    let t18 = kirp::<MIN, MAX>(_mm256_add_epi32(t18a, t21a));
    let t19a = kirp::<MIN, MAX>(_mm256_add_epi32(t19, t20));
    let t20a = kirp::<MIN, MAX>(_mm256_sub_epi32(t19, t20));
    let t21 = kirp::<MIN, MAX>(_mm256_sub_epi32(t18a, t21a));
    let t22a = kirp::<MIN, MAX>(_mm256_sub_epi32(t17, t22));
    let t23 = kirp::<MIN, MAX>(_mm256_sub_epi32(t16a, t23a));
    let t24 = kirp::<MIN, MAX>(_mm256_sub_epi32(t31a, t24a));
    let t25a = kirp::<MIN, MAX>(_mm256_sub_epi32(t30, t25));
    let t26 = kirp::<MIN, MAX>(_mm256_sub_epi32(t29a, t26a));
    let t27a = kirp::<MIN, MAX>(_mm256_sub_epi32(t28, t27));
    let t28a = kirp::<MIN, MAX>(_mm256_add_epi32(t28, t27));
    let t29 = kirp::<MIN, MAX>(_mm256_add_epi32(t29a, t26a));
    let t30a = kirp::<MIN, MAX>(_mm256_add_epi32(t30, t25));
    let t31 = kirp::<MIN, MAX>(_mm256_add_epi32(t31a, t24a));

    let t20 = olcekle_181_ic(_mm256_sub_epi32(t27a, t20a));
    let t27 = olcekle_181_ic(_mm256_add_epi32(t27a, t20a));
    let t21a = olcekle_181_ic(_mm256_sub_epi32(t26, t21));
    let t26a = olcekle_181_ic(_mm256_add_epi32(t26, t21));
    let t22 = olcekle_181_ic(_mm256_sub_epi32(t25a, t22a));
    let t25 = olcekle_181_ic(_mm256_add_epi32(t25a, t22a));
    let t23a = olcekle_181_ic(_mm256_sub_epi32(t24, t23));
    let t24a = olcekle_181_ic(_mm256_add_epi32(t24, t23));

    store_i32x8(chunk_mut(serit, 384), t16);
    store_i32x8(chunk_mut(serit, 384 + 8), t17a);
    store_i32x8(chunk_mut(serit, 384 + 2 * 8), t18);
    store_i32x8(chunk_mut(serit, 384 + 3 * 8), t19a);
    store_i32x8(chunk_mut(serit, 384 + 4 * 8), t20);
    store_i32x8(chunk_mut(serit, 384 + 5 * 8), t21a);
    store_i32x8(chunk_mut(serit, 384 + 6 * 8), t22);
    store_i32x8(chunk_mut(serit, 384 + 7 * 8), t23a);
    store_i32x8(chunk_mut(serit, 384 + 8 * 8), t24a);
    store_i32x8(chunk_mut(serit, 384 + 9 * 8), t25);
    store_i32x8(chunk_mut(serit, 384 + 10 * 8), t26a);
    store_i32x8(chunk_mut(serit, 384 + 11 * 8), t27);
    store_i32x8(chunk_mut(serit, 384 + 12 * 8), t28a);
    store_i32x8(chunk_mut(serit, 384 + 13 * 8), t29);
    store_i32x8(chunk_mut(serit, 384 + 14 * 8), t30a);
    store_i32x8(chunk_mut(serit, 384 + 15 * 8), t31);

    let mut i = 0;
    while i < 16 {
        let e_i = yukle_i32x8(serit, 256 + i * 8);
        let o_i = yukle_i32x8(serit, 384 + (15 - i) * 8);
        store_i32x8(
            chunk_mut(serit, 256 + i * 8),
            kirp::<MIN, MAX>(_mm256_add_epi32(e_i, o_i)),
        );
        store_i32x8(
            chunk_mut(serit, 256 + (31 - i) * 8),
            kirp::<MIN, MAX>(_mm256_sub_epi32(e_i, o_i)),
        );
        i += 1;
    }
}

/// 64 noktalı DCT tek yarı 1a aşaması: `tek[0, 15, 8, 7, 4, 11, 12, 3]` → `serit[0..8]` ve `serit[24..32]`.
#[target_feature(enable = "avx2")]
#[inline(never)]
fn idct64_tek_asama1a<const MIN: i32, const MAX: i32>(tek: &[i32; 128], serit: &mut [i32]) {
    let in1 = yukle_i32x8(tek, 0);
    let in31 = yukle_i32x8(tek, 15 * 8);
    let in17 = yukle_i32x8(tek, 8 * 8);
    let in15 = yukle_i32x8(tek, 7 * 8);
    let t32a = mul1_64_min::<12, MIN>(in1, 101, 2048);
    let t33a = mul1_64_min::<12, MIN>(in31, -2824, 2048);
    let t34a = mul1_64_min::<12, MIN>(in17, 1660, 2048);
    let t35a = mul1_64_min::<12, MIN>(in15, -1474, 2048);
    let t60a = mul1_64_min::<12, MIN>(in15, 3822, 2048);
    let t61a = mul1_64_min::<12, MIN>(in17, 3745, 2048);
    let t62a = mul1_64_min::<12, MIN>(in31, 2967, 2048);
    let t63a = mul1_64_min::<12, MIN>(in1, 4095, 2048);

    let (t32, t33) = add_sub_kirp_64::<MIN, MAX>(t32a, t33a);
    let (t34, t35) = sub_add_kirp_64::<MIN, MAX>(t35a, t34a);
    let (t60, t61) = add_sub_kirp_64::<MIN, MAX>(t60a, t61a);
    let (t62, t63) = sub_add_kirp_64::<MIN, MAX>(t63a, t62a);

    let (t33a, t62a) = mul2::<MIN>(t62, t33, 401, -4076, 4076, 401);
    let (t34a, t61a) = mul2::<MIN>(t61, t34, -4076, -401, 401, -4076);

    store_i32x8(
        chunk_mut(serit, 0),
        kirp::<MIN, MAX>(_mm256_add_epi32(t32, t35)),
    );
    store_i32x8(
        chunk_mut(serit, 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t33a, t34a)),
    );
    store_i32x8(
        chunk_mut(serit, 2 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t33a, t34a)),
    );
    store_i32x8(
        chunk_mut(serit, 3 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t32, t35)),
    );
    store_i32x8(
        chunk_mut(serit, 28 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t63, t60)),
    );
    store_i32x8(
        chunk_mut(serit, 29 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t62a, t61a)),
    );
    store_i32x8(
        chunk_mut(serit, 30 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t62a, t61a)),
    );
    store_i32x8(
        chunk_mut(serit, 31 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t63, t60)),
    );

    let in9 = yukle_i32x8(tek, 4 * 8);
    let in23 = yukle_i32x8(tek, 11 * 8);
    let in25 = yukle_i32x8(tek, 12 * 8);
    let in7 = yukle_i32x8(tek, 3 * 8);
    let t36a = mul1_64_min::<12, MIN>(in9, 897, 2048);
    let t37a = mul1_64_min::<12, MIN>(in23, -2191, 2048);
    let t38a = mul1_64_min::<12, MIN>(in25, 2359, 2048);
    let t39a = mul1_64_min::<12, MIN>(in7, -700, 2048);
    let t56a = mul1_64_min::<12, MIN>(in7, 4036, 2048);
    let t57a = mul1_64_min::<12, MIN>(in25, 3349, 2048);
    let t58a = mul1_64_min::<12, MIN>(in23, 3461, 2048);
    let t59a = mul1_64_min::<12, MIN>(in9, 3996, 2048);

    let (t36, t37) = add_sub_kirp_64::<MIN, MAX>(t36a, t37a);
    let (t38, t39) = sub_add_kirp_64::<MIN, MAX>(t39a, t38a);
    let (t56, t57) = add_sub_kirp_64::<MIN, MAX>(t56a, t57a);
    let (t58, t59) = sub_add_kirp_64::<MIN, MAX>(t59a, t58a);

    let (t37a, t58a) = mul2::<MIN>(t58, t37, 3166, -2598, 2598, 3166);
    let (t38a, t57a) = mul2::<MIN>(t57, t38, -2598, -3166, 3166, -2598);

    store_i32x8(
        chunk_mut(serit, 4 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t39, t36)),
    );
    store_i32x8(
        chunk_mut(serit, 5 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t38a, t37a)),
    );
    store_i32x8(
        chunk_mut(serit, 6 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t38a, t37a)),
    );
    store_i32x8(
        chunk_mut(serit, 7 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t39, t36)),
    );
    store_i32x8(
        chunk_mut(serit, 24 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t56, t59)),
    );
    store_i32x8(
        chunk_mut(serit, 25 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t57a, t58a)),
    );
    store_i32x8(
        chunk_mut(serit, 26 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t57a, t58a)),
    );
    store_i32x8(
        chunk_mut(serit, 27 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t56, t59)),
    );
}

/// 64 noktalı DCT tek yarı 1b aşaması: `tek[2, 13, 10, 5, 6, 9, 14, 1]` → `serit[8..24]`.
#[target_feature(enable = "avx2")]
#[inline(never)]
fn idct64_tek_asama1b<const MIN: i32, const MAX: i32>(tek: &[i32; 128], serit: &mut [i32]) {
    let in5 = yukle_i32x8(tek, 2 * 8);
    let in27 = yukle_i32x8(tek, 13 * 8);
    let in21 = yukle_i32x8(tek, 10 * 8);
    let in11 = yukle_i32x8(tek, 5 * 8);
    let t40a = mul1_64_min::<12, MIN>(in5, 501, 2048);
    let t41a = mul1_64_min::<12, MIN>(in27, -2520, 2048);
    let t42a = mul1_64_min::<12, MIN>(in21, 2019, 2048);
    let t43a = mul1_64_min::<12, MIN>(in11, -1092, 2048);
    let t52a = mul1_64_min::<12, MIN>(in11, 3948, 2048);
    let t53a = mul1_64_min::<12, MIN>(in21, 3564, 2048);
    let t54a = mul1_64_min::<12, MIN>(in27, 3229, 2048);
    let t55a = mul1_64_min::<12, MIN>(in5, 4065, 2048);

    let (t40, t41) = add_sub_kirp_64::<MIN, MAX>(t40a, t41a);
    let (t42, t43) = sub_add_kirp_64::<MIN, MAX>(t43a, t42a);
    let (t52, t53) = add_sub_kirp_64::<MIN, MAX>(t52a, t53a);
    let (t54, t55) = sub_add_kirp_64::<MIN, MAX>(t55a, t54a);

    let (t41a, t54a) = mul2::<MIN>(t54, t41, 1931, -3612, 3612, 1931);
    let (t42a, t53a) = mul2::<MIN>(t53, t42, -3612, -1931, 1931, -3612);

    store_i32x8(
        chunk_mut(serit, 8 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t40, t43)),
    );
    store_i32x8(
        chunk_mut(serit, 9 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t41a, t42a)),
    );
    store_i32x8(
        chunk_mut(serit, 10 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t41a, t42a)),
    );
    store_i32x8(
        chunk_mut(serit, 11 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t40, t43)),
    );
    store_i32x8(
        chunk_mut(serit, 20 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t55, t52)),
    );
    store_i32x8(
        chunk_mut(serit, 21 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t54a, t53a)),
    );
    store_i32x8(
        chunk_mut(serit, 22 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t54a, t53a)),
    );
    store_i32x8(
        chunk_mut(serit, 23 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t55, t52)),
    );

    let in13 = yukle_i32x8(tek, 6 * 8);
    let in19 = yukle_i32x8(tek, 9 * 8);
    let in29 = yukle_i32x8(tek, 14 * 8);
    let in3 = yukle_i32x8(tek, 8);
    let t44a = mul1_64_min::<12, MIN>(in13, 1285, 2048);
    let t45a = mul1_64_min::<12, MIN>(in19, -1842, 2048);
    let t46a = mul1_64_min::<12, MIN>(in29, 2675, 2048);
    let t47a = mul1_64_min::<12, MIN>(in3, -301, 2048);
    let t48a = mul1_64_min::<12, MIN>(in3, 4085, 2048);
    let t49a = mul1_64_min::<12, MIN>(in29, 3102, 2048);
    let t50a = mul1_64_min::<12, MIN>(in19, 3659, 2048);
    let t51a = mul1_64_min::<12, MIN>(in13, 3889, 2048);

    let (t44, t45) = add_sub_kirp_64::<MIN, MAX>(t44a, t45a);
    let (t46, t47) = sub_add_kirp_64::<MIN, MAX>(t47a, t46a);
    let (t48, t49) = add_sub_kirp_64::<MIN, MAX>(t48a, t49a);
    let (t50, t51) = sub_add_kirp_64::<MIN, MAX>(t51a, t50a);

    let (t45a, t50a) = mul2::<MIN>(t50, t45, 3920, -1189, 1189, 3920);
    let (t46a, t49a) = mul2::<MIN>(t49, t46, -1189, -3920, 3920, -1189);

    store_i32x8(
        chunk_mut(serit, 12 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t47, t44)),
    );
    store_i32x8(
        chunk_mut(serit, 13 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t46a, t45a)),
    );
    store_i32x8(
        chunk_mut(serit, 14 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t46a, t45a)),
    );
    store_i32x8(
        chunk_mut(serit, 15 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t47, t44)),
    );
    store_i32x8(
        chunk_mut(serit, 16 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t48, t51)),
    );
    store_i32x8(
        chunk_mut(serit, 17 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t49a, t50a)),
    );
    store_i32x8(
        chunk_mut(serit, 18 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t49a, t50a)),
    );
    store_i32x8(
        chunk_mut(serit, 19 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t48, t51)),
    );
}

/// 64 noktalı DCT tek yarı 2a aşaması: `serit[0..8]` ve `serit[24..32]` yerinde kelebek.
#[target_feature(enable = "avx2")]
#[inline(never)]
fn idct64_tek_asama2a<const MIN: i32, const MAX: i32>(serit: &mut [i32]) {
    let t32a = yukle_i32x8(serit, 0);
    let t33 = yukle_i32x8(serit, 8);
    let t34 = yukle_i32x8(serit, 2 * 8);
    let t35a = yukle_i32x8(serit, 3 * 8);
    let t36a = yukle_i32x8(serit, 4 * 8);
    let t37 = yukle_i32x8(serit, 5 * 8);
    let t38 = yukle_i32x8(serit, 6 * 8);
    let t39a = yukle_i32x8(serit, 7 * 8);
    let t56a = yukle_i32x8(serit, 24 * 8);
    let t57 = yukle_i32x8(serit, 25 * 8);
    let t58 = yukle_i32x8(serit, 26 * 8);
    let t59a = yukle_i32x8(serit, 27 * 8);
    let t60a = yukle_i32x8(serit, 28 * 8);
    let t61 = yukle_i32x8(serit, 29 * 8);
    let t62 = yukle_i32x8(serit, 30 * 8);
    let t63a = yukle_i32x8(serit, 31 * 8);

    let (t34a, t61a) = mul2::<MIN>(t61, t34, 799, -4017, 4017, 799);
    let (t35, t60) = mul2::<MIN>(t60a, t35a, 799, -4017, 4017, 799);
    let (t36, t59) = mul2::<MIN>(t59a, t36a, -4017, -799, 799, -4017);
    let (t37a, t58a) = mul2::<MIN>(t58, t37, -4017, -799, 799, -4017);

    store_i32x8(
        chunk_mut(serit, 0),
        kirp::<MIN, MAX>(_mm256_add_epi32(t32a, t39a)),
    );
    store_i32x8(
        chunk_mut(serit, 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t33, t38)),
    );
    store_i32x8(
        chunk_mut(serit, 2 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t34a, t37a)),
    );
    store_i32x8(
        chunk_mut(serit, 3 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t35, t36)),
    );
    store_i32x8(
        chunk_mut(serit, 4 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t35, t36)),
    );
    store_i32x8(
        chunk_mut(serit, 5 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t34a, t37a)),
    );
    store_i32x8(
        chunk_mut(serit, 6 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t33, t38)),
    );
    store_i32x8(
        chunk_mut(serit, 7 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t32a, t39a)),
    );
    store_i32x8(
        chunk_mut(serit, 24 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t63a, t56a)),
    );
    store_i32x8(
        chunk_mut(serit, 25 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t62, t57)),
    );
    store_i32x8(
        chunk_mut(serit, 26 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t61a, t58a)),
    );
    store_i32x8(
        chunk_mut(serit, 27 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t60, t59)),
    );
    store_i32x8(
        chunk_mut(serit, 28 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t60, t59)),
    );
    store_i32x8(
        chunk_mut(serit, 29 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t61a, t58a)),
    );
    store_i32x8(
        chunk_mut(serit, 30 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t62, t57)),
    );
    store_i32x8(
        chunk_mut(serit, 31 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t63a, t56a)),
    );
}

/// 64 noktalı DCT tek yarı 2b aşaması: `serit[8..24]` yerinde kelebek.
#[target_feature(enable = "avx2")]
#[inline(never)]
fn idct64_tek_asama2b<const MIN: i32, const MAX: i32>(serit: &mut [i32]) {
    let t40a = yukle_i32x8(serit, 8 * 8);
    let t41 = yukle_i32x8(serit, 9 * 8);
    let t42 = yukle_i32x8(serit, 10 * 8);
    let t43a = yukle_i32x8(serit, 11 * 8);
    let t44a = yukle_i32x8(serit, 12 * 8);
    let t45 = yukle_i32x8(serit, 13 * 8);
    let t46 = yukle_i32x8(serit, 14 * 8);
    let t47a = yukle_i32x8(serit, 15 * 8);
    let t48a = yukle_i32x8(serit, 16 * 8);
    let t49 = yukle_i32x8(serit, 17 * 8);
    let t50 = yukle_i32x8(serit, 18 * 8);
    let t51a = yukle_i32x8(serit, 19 * 8);
    let t52a = yukle_i32x8(serit, 20 * 8);
    let t53 = yukle_i32x8(serit, 21 * 8);
    let t54 = yukle_i32x8(serit, 22 * 8);
    let t55a = yukle_i32x8(serit, 23 * 8);

    let (t42a, t53a) = mul2::<MIN>(t53, t42, 3406, -2276, 2276, 3406);
    let (t43, t52) = mul2::<MIN>(t52a, t43a, 3406, -2276, 2276, 3406);
    let (t44, t51) = mul2::<MIN>(t51a, t44a, -2276, -3406, 3406, -2276);
    let (t45a, t50a) = mul2::<MIN>(t50, t45, -2276, -3406, 3406, -2276);

    store_i32x8(
        chunk_mut(serit, 8 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t47a, t40a)),
    );
    store_i32x8(
        chunk_mut(serit, 9 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t46, t41)),
    );
    store_i32x8(
        chunk_mut(serit, 10 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t45a, t42a)),
    );
    store_i32x8(
        chunk_mut(serit, 11 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t44, t43)),
    );
    store_i32x8(
        chunk_mut(serit, 12 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t44, t43)),
    );
    store_i32x8(
        chunk_mut(serit, 13 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t45a, t42a)),
    );
    store_i32x8(
        chunk_mut(serit, 14 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t46, t41)),
    );
    store_i32x8(
        chunk_mut(serit, 15 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t47a, t40a)),
    );
    store_i32x8(
        chunk_mut(serit, 16 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t48a, t55a)),
    );
    store_i32x8(
        chunk_mut(serit, 17 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t49, t54)),
    );
    store_i32x8(
        chunk_mut(serit, 18 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t50a, t53a)),
    );
    store_i32x8(
        chunk_mut(serit, 19 * 8),
        kirp::<MIN, MAX>(_mm256_add_epi32(t51, t52)),
    );
    store_i32x8(
        chunk_mut(serit, 20 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t51, t52)),
    );
    store_i32x8(
        chunk_mut(serit, 21 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t50a, t53a)),
    );
    store_i32x8(
        chunk_mut(serit, 22 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t49, t54)),
    );
    store_i32x8(
        chunk_mut(serit, 23 * 8),
        kirp::<MIN, MAX>(_mm256_sub_epi32(t48a, t55a)),
    );
}

/// 64 noktalı DCT tek yarı 3. (son) aşama: `serit[0..256]` üzerinde yerinde dönme + kelebek + `181/256`.
#[target_feature(enable = "avx2")]
#[inline(never)]
fn idct64_tek_asama3<const MIN: i32, const MAX: i32>(serit: &mut [i32]) {
    let mut j = 0;
    while j < 4 {
        let lo = yukle_i32x8(serit, (4 + j) * 8);
        let hi = yukle_i32x8(serit, (27 - j) * 8);
        let (a, b) = mul2::<MIN>(hi, lo, 1567, -3784, 3784, 1567);
        store_i32x8(chunk_mut(serit, (4 + j) * 8), a);
        store_i32x8(chunk_mut(serit, (27 - j) * 8), b);
        j += 1;
    }
    let mut j = 0;
    while j < 4 {
        let lo = yukle_i32x8(serit, (8 + j) * 8);
        let hi = yukle_i32x8(serit, (23 - j) * 8);
        let (a, b) = mul2::<MIN>(hi, lo, -3784, -1567, 1567, -3784);
        store_i32x8(chunk_mut(serit, (8 + j) * 8), a);
        store_i32x8(chunk_mut(serit, (23 - j) * 8), b);
        j += 1;
    }
    let mut j = 0;
    while j < 8 {
        let a = yukle_i32x8(serit, j * 8);
        let b = yukle_i32x8(serit, (15 - j) * 8);
        store_i32x8(
            chunk_mut(serit, j * 8),
            kirp::<MIN, MAX>(_mm256_add_epi32(a, b)),
        );
        store_i32x8(
            chunk_mut(serit, (15 - j) * 8),
            kirp::<MIN, MAX>(_mm256_sub_epi32(a, b)),
        );
        let c = yukle_i32x8(serit, (16 + j) * 8);
        let d = yukle_i32x8(serit, (31 - j) * 8);
        store_i32x8(
            chunk_mut(serit, (16 + j) * 8),
            kirp::<MIN, MAX>(_mm256_sub_epi32(d, c)),
        );
        store_i32x8(
            chunk_mut(serit, (31 - j) * 8),
            kirp::<MIN, MAX>(_mm256_add_epi32(d, c)),
        );
        j += 1;
    }
    let mut j = 0;
    while j < 8 {
        let lo = yukle_i32x8(serit, (8 + j) * 8);
        let hi = yukle_i32x8(serit, (23 - j) * 8);
        store_i32x8(
            chunk_mut(serit, (8 + j) * 8),
            olcekle_181_ic(_mm256_sub_epi32(hi, lo)),
        );
        store_i32x8(
            chunk_mut(serit, (23 - j) * 8),
            olcekle_181_ic(_mm256_add_epi32(hi, lo)),
        );
        j += 1;
    }
}

/// 64 noktalı 1B ters DCT: `ara.serit[0..256]` (32 giriş) →
/// çift 32 çıktı `ara.serit[256..512]`, tek 32 çıktı `ara.serit[0..256]`.
#[target_feature(enable = "avx2")]
#[inline(never)]
fn idct64_serit<const MIN: i32, const MAX: i32>(ara: &mut BuyukAra) {
    let mut i = 0;
    while i < 16 {
        let v = yukle_i32x8(&ara.serit, (2 * i + 1) * 8);
        store_i32x8(chunk_mut(&mut ara.tek, i * 8), v);
        i += 1;
    }
    idct16_tx64_serit::<MIN, MAX>(&mut ara.serit);
    idct32_tx64_tek_asama1::<MIN, MAX>(&mut ara.serit);
    idct32_tx64_tek_asama2::<MIN, MAX>(&mut ara.serit);
    idct64_tek_asama1a::<MIN, MAX>(&ara.tek, &mut ara.serit);
    idct64_tek_asama1b::<MIN, MAX>(&ara.tek, &mut ara.serit);
    idct64_tek_asama2a::<MIN, MAX>(&mut ara.serit);
    idct64_tek_asama2b::<MIN, MAX>(&mut ara.serit);
    idct64_tek_asama3::<MIN, MAX>(&mut ara.serit);
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn dct_buyuk_gecis1<const W: usize, const SH: usize, const SHIFT: u32, const RECT2: bool>(
    ara: &mut BuyukAra,
    coeff: &[i32],
    y_blok: usize,
) {
    let sw = if W > 32 { 32 } else { W };
    let w_blok = W / 8;
    let mut yb = 0;
    while yb < y_blok {
        let y0 = yb * 8;
        let mut x = 0;
        while x < sw {
            let v = yukle_i32x8(coeff, y0 + x * SH);
            let v = if RECT2 { olcekle_181(v) } else { v };
            store_i32x8(chunk_mut(&mut ara.serit, x * 8), v);
            x += 1;
        }
        if W == 16 {
            idct16_serit::<ROW_MIN, ROW_MAX>(&mut ara.serit);
            let mut xb = 0;
            while xb < 2 {
                let karo: [__m256i; 8] = core::array::from_fn(|dx| {
                    kaydir_kirp_18b::<SHIFT>(yukle_i32x8(&ara.serit, (xb * 8 + dx) * 8))
                });
                let t = devrik_8x8_i32(karo);
                let mut dy = 0;
                while dy < 8 {
                    store_i32x8(chunk_mut(&mut ara.mat, (xb * SH + y0 + dy) * 8), t[dy]);
                    dy += 1;
                }
                xb += 1;
            }
        } else if W == 32 {
            idct32_serit::<ROW_MIN, ROW_MAX>(&mut ara.serit);
            let mut xb = 0;
            while xb < 4 {
                let karo: [__m256i; 8] = core::array::from_fn(|dx| {
                    kaydir_kirp_18b::<SHIFT>(yukle_i32x8(&ara.serit, 256 + (xb * 8 + dx) * 8))
                });
                let t = devrik_8x8_i32(karo);
                let mut dy = 0;
                while dy < 8 {
                    store_i32x8(chunk_mut(&mut ara.mat, (xb * SH + y0 + dy) * 8), t[dy]);
                    dy += 1;
                }
                xb += 1;
            }
        } else {
            idct64_serit::<ROW_MIN, ROW_MAX>(ara);
            let mut xb = 0;
            while xb < 8 {
                let karo: [__m256i; 8] = core::array::from_fn(|dx| {
                    let col = xb * 8 + dx;
                    let v = if col < 32 {
                        let e = yukle_i32x8(&ara.serit, 256 + col * 8);
                        let o = yukle_i32x8(&ara.serit, (31 - col) * 8);
                        kirp::<ROW_MIN, ROW_MAX>(_mm256_add_epi32(e, o))
                    } else {
                        let idx = 63 - col;
                        let e = yukle_i32x8(&ara.serit, 256 + idx * 8);
                        let o = yukle_i32x8(&ara.serit, (31 - idx) * 8);
                        kirp::<ROW_MIN, ROW_MAX>(_mm256_sub_epi32(e, o))
                    };
                    kaydir_kirp_18b::<SHIFT>(v)
                });
                let t = devrik_8x8_i32(karo);
                let mut dy = 0;
                while dy < 8 {
                    store_i32x8(chunk_mut(&mut ara.mat, (xb * SH + y0 + dy) * 8), t[dy]);
                    dy += 1;
                }
                xb += 1;
            }
        }
        yb += 1;
    }
    let h_blok = SH / 8;
    if y_blok < h_blok {
        let z = _mm256_setzero_si256();
        let mut xb = 0;
        while xb < w_blok {
            let mut y = y_blok * 8;
            while y < SH {
                store_i32x8(chunk_mut(&mut ara.mat, (xb * SH + y) * 8), z);
                y += 1;
            }
            xb += 1;
        }
    }
}

#[target_feature(enable = "avx2")]
#[inline(never)]
fn dct_buyuk_gecis2<const W: usize, const H: usize, const SH: usize>(
    ara: &mut BuyukAra,
    dst: &mut [u8],
    stride: usize,
) {
    let w_blok = W / 8;
    let mut xb = 0;
    while xb < w_blok {
        let x_bayt = xb * 16;
        let mat_taban = xb * SH * 8;
        let mut y = 0;
        while y < SH {
            let v = yukle_i32x8(&ara.mat, mat_taban + y * 8);
            store_i32x8(chunk_mut(&mut ara.serit, y * 8), v);
            y += 1;
        }
        if H == 16 {
            idct16_serit::<COL_MIN, COL_MAX>(&mut ara.serit);
            let mut r = 0;
            while r < 16 {
                ekle_yaz_8x1(dst, r * stride + x_bayt, yukle_i32x8(&ara.serit, r * 8));
                r += 1;
            }
        } else if H == 32 {
            idct32_serit::<COL_MIN, COL_MAX>(&mut ara.serit);
            let mut r = 0;
            while r < 32 {
                ekle_yaz_8x1(
                    dst,
                    r * stride + x_bayt,
                    yukle_i32x8(&ara.serit, 256 + r * 8),
                );
                r += 1;
            }
        } else {
            idct64_serit::<COL_MIN, COL_MAX>(ara);
            let mut r = 0;
            while r < 32 {
                let e_r = yukle_i32x8(&ara.serit, 256 + r * 8);
                let o_r = yukle_i32x8(&ara.serit, (31 - r) * 8);
                let ust = kirp::<COL_MIN, COL_MAX>(_mm256_add_epi32(e_r, o_r));
                let alt = kirp::<COL_MIN, COL_MAX>(_mm256_sub_epi32(e_r, o_r));
                ekle_yaz_8x1(dst, r * stride + x_bayt, ust);
                ekle_yaz_8x1(dst, (63 - r) * stride + x_bayt, alt);
                r += 1;
            }
        }
        xb += 1;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn dct_buyuk_2d<
    const W: usize,
    const H: usize,
    const SHIFT: u32,
    const RECT2: bool,
    const TX_IDX: usize,
>(
    ara: &mut BuyukAra,
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32],
    eob: usize,
) {
    if eob == 0 {
        dc_yalniz::<W, H, SHIFT, RECT2>(dst, stride, coeff);
        return;
    }
    const { assert!(W == 16 || W == 32 || W == 64) };
    const { assert!(H == 16 || H == 32 || H == 64) };
    let sw = if W > 32 { 32 } else { W };
    let sh = if H > 32 { 32 } else { H };
    let y_blok = (usize::from(LNZC_TABLES[TX_IDX][eob]) >> 3) + 1;
    if sh == 16 {
        dct_buyuk_gecis1::<W, 16, SHIFT, RECT2>(ara, coeff, y_blok);
    } else {
        dct_buyuk_gecis1::<W, 32, SHIFT, RECT2>(ara, coeff, y_blok);
    }
    if sw * sh == 512 {
        sifirla_katsayi::<512>(coeff);
    } else {
        sifirla_katsayi::<1024>(coeff);
    }
    if sh == 16 {
        dct_buyuk_gecis2::<W, H, 16>(ara, dst, stride);
    } else {
        dct_buyuk_gecis2::<W, H, 32>(ara, dst, stride);
    }
}

/// `(TX, TXTP)` çiftine özelleştirilmiş büyük (`15` kayıt) 12 bpc AVX2 çekirdek gövdesi.
#[target_feature(enable = "avx2")]
#[inline(never)]
fn itx12_buyuk_avx2_cekirdek<const TX: u8, const TXTP: u8>(
    ara: &mut BuyukAra,
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
) {
    let tx = TxSize::from_index(TX as usize).expect("geçerli tx");
    let txtp = TxType::from_index(TXTP as usize).expect("geçerli txtp");
    match (tx, txtp) {
        (TxSize::Rtx8x32, TxType::DctDct) => dct_8x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx8x32, TxType::Idtx) => idtx_8x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x8, TxType::DctDct) => dct_32x8(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x8, TxType::Idtx) => idtx_32x8(dst, dst_stride, coeff, eob),
        (TxSize::Rtx16x32, TxType::DctDct) => {
            dct_buyuk_2d::<16, 32, 1, true, { TxSize::Rtx16x32 as usize }>(
                ara, dst, dst_stride, coeff, eob,
            );
        }
        (TxSize::Rtx16x32, TxType::Idtx) => idtx_16x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x16, TxType::DctDct) => {
            dct_buyuk_2d::<32, 16, 1, true, { TxSize::Rtx32x16 as usize }>(
                ara, dst, dst_stride, coeff, eob,
            );
        }
        (TxSize::Rtx32x16, TxType::Idtx) => idtx_32x16(dst, dst_stride, coeff, eob),
        (TxSize::Tx32x32, TxType::DctDct) => {
            dct_buyuk_2d::<32, 32, 2, false, { TxSize::Tx32x32 as usize }>(
                ara, dst, dst_stride, coeff, eob,
            );
        }
        (TxSize::Tx32x32, TxType::Idtx) => idtx_32x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx16x64, TxType::DctDct) => {
            dct_buyuk_2d::<16, 64, 2, false, { TxSize::Rtx16x64 as usize }>(
                ara, dst, dst_stride, coeff, eob,
            );
        }
        (TxSize::Rtx32x64, TxType::DctDct) => {
            dct_buyuk_2d::<32, 64, 1, true, { TxSize::Rtx32x64 as usize }>(
                ara, dst, dst_stride, coeff, eob,
            );
        }
        (TxSize::Rtx64x16, TxType::DctDct) => {
            dct_buyuk_2d::<64, 16, 2, false, { TxSize::Rtx64x16 as usize }>(
                ara, dst, dst_stride, coeff, eob,
            );
        }
        (TxSize::Rtx64x32, TxType::DctDct) => {
            dct_buyuk_2d::<64, 32, 1, true, { TxSize::Rtx64x32 as usize }>(
                ara, dst, dst_stride, coeff, eob,
            );
        }
        (TxSize::Tx64x64, TxType::DctDct) => {
            dct_buyuk_2d::<64, 64, 2, false, { TxSize::Tx64x64 as usize }>(
                ara, dst, dst_stride, coeff, eob,
            );
        }
        _ => unreachable!(),
    }
}

/// `(TX, TXTP)` çiftinin sözleşme denetimli güvenli büyük 12 bpc AVX2 yuva girişi.
fn itx12_buyuk_avx2_giris<const TX: u8, const TXTP: u8>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
    bit_depth: BitDepth,
) {
    if bit_depth != BitDepth::Twelve
        || tx as u8 != TX
        || txtp as u8 != TXTP
        || !itx16_sozlesmesi(dst.len(), dst_stride, coeff.len(), eob, tx, txtp)
    {
        match bit_depth {
            BitDepth::Eight => itx_skaler_devret::<8>(dst, dst_stride, coeff, eob, tx, txtp),
            BitDepth::Ten => itx_skaler_devret::<10>(dst, dst_stride, coeff, eob, tx, txtp),
            BitDepth::Twelve => itx_skaler_devret::<12>(dst, dst_stride, coeff, eob, tx, txtp),
        }
        return;
    }
    with_buyuk_ara(|ara| {
        // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir (`itx12_buyuk_avx2_fn` /
        // `itx12_buyuk_avx2_lookup`); belirteç AVX2 tespiti ve maske izninin kanıtıdır. (2) sınır:
        // `itx16_sozlesmesi` `dst_stride`'ın çift ve `≥ w·2` olduğunu, `dst`'nin `(h − 1)·dst_stride + w·2`
        // baytı ve `coeff`'un `1024` öğeyi kapsadığını garanti eder; bütün bellek erişimleri sabit
        // boyutlu `chunk`/`chunk_mut` dizileri üzerindendir. (3) hizalama/takma ad: `ara`, `dst` ve
        // `coeff` ayrık `&mut` referanslardır, hizasız erişim kullanılır.
        unsafe {
            itx12_buyuk_avx2_cekirdek::<TX, TXTP>(ara, dst, dst_stride, coeff, eob);
        }
    });
}

/// `max(w, h) >= 32` boyutlu 15 adet 12 bpc AVX2 `itxfm_add[tx][txtp]` yuvası
/// (yalnız [`Avx2Token`] ile): 5 `IDTX` + 10 `DCT_DCT`.
#[must_use]
pub const fn itx12_buyuk_avx2_fn(_token: Avx2Token, tx: TxSize, txtp: TxType) -> Option<ItxFn> {
    match (tx, txtp) {
        (TxSize::Rtx8x32, TxType::DctDct) => {
            Some(itx12_buyuk_avx2_giris::<{ TxSize::Rtx8x32 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx8x32, TxType::Idtx) => {
            Some(itx12_buyuk_avx2_giris::<{ TxSize::Rtx8x32 as u8 }, { TxType::Idtx as u8 }>)
        }
        (TxSize::Rtx32x8, TxType::DctDct) => {
            Some(itx12_buyuk_avx2_giris::<{ TxSize::Rtx32x8 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx32x8, TxType::Idtx) => {
            Some(itx12_buyuk_avx2_giris::<{ TxSize::Rtx32x8 as u8 }, { TxType::Idtx as u8 }>)
        }
        (TxSize::Rtx16x32, TxType::DctDct) => {
            Some(itx12_buyuk_avx2_giris::<{ TxSize::Rtx16x32 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx16x32, TxType::Idtx) => {
            Some(itx12_buyuk_avx2_giris::<{ TxSize::Rtx16x32 as u8 }, { TxType::Idtx as u8 }>)
        }
        (TxSize::Rtx32x16, TxType::DctDct) => {
            Some(itx12_buyuk_avx2_giris::<{ TxSize::Rtx32x16 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx32x16, TxType::Idtx) => {
            Some(itx12_buyuk_avx2_giris::<{ TxSize::Rtx32x16 as u8 }, { TxType::Idtx as u8 }>)
        }
        (TxSize::Tx32x32, TxType::DctDct) => {
            Some(itx12_buyuk_avx2_giris::<{ TxSize::Tx32x32 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Tx32x32, TxType::Idtx) => {
            Some(itx12_buyuk_avx2_giris::<{ TxSize::Tx32x32 as u8 }, { TxType::Idtx as u8 }>)
        }
        (TxSize::Rtx16x64, TxType::DctDct) => {
            Some(itx12_buyuk_avx2_giris::<{ TxSize::Rtx16x64 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx32x64, TxType::DctDct) => {
            Some(itx12_buyuk_avx2_giris::<{ TxSize::Rtx32x64 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx64x16, TxType::DctDct) => {
            Some(itx12_buyuk_avx2_giris::<{ TxSize::Rtx64x16 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx64x32, TxType::DctDct) => {
            Some(itx12_buyuk_avx2_giris::<{ TxSize::Rtx64x32 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Tx64x64, TxType::DctDct) => {
            Some(itx12_buyuk_avx2_giris::<{ TxSize::Tx64x64 as u8 }, { TxType::DctDct as u8 }>)
        }
        _ => None,
    }
}

/// [`itx12_buyuk_avx2_fn`] eşdeğer arama girişi.
#[must_use]
pub const fn itx12_buyuk_avx2_lookup(token: Avx2Token, tx: TxSize, txtp: TxType) -> Option<ItxFn> {
    itx12_buyuk_avx2_fn(token, tx, txtp)
}
