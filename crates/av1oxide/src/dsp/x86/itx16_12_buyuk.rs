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
use crate::frame::BitDepth;

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
fn dc_yalniz<const W: usize, const H: usize, const SHIFT: u32>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32],
) {
    let mut dc = i64::from(coeff[0]);
    coeff[0] = 0;
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
        dc_yalniz::<8, 32, 2>(dst, stride, coeff);
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
        dc_yalniz::<32, 8, 2>(dst, stride, coeff);
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

/// `(TX, TXTP)` çiftine özelleştirilmiş büyük (`7` kayıt) 12 bpc AVX2 çekirdek gövdesi.
#[target_feature(enable = "avx2")]
#[inline(never)]
fn itx12_buyuk_avx2_cekirdek<const TX: u8, const TXTP: u8>(
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
        (TxSize::Rtx16x32, TxType::Idtx) => idtx_16x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x16, TxType::Idtx) => idtx_32x16(dst, dst_stride, coeff, eob),
        (TxSize::Tx32x32, TxType::Idtx) => idtx_32x32(dst, dst_stride, coeff, eob),
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx2Token` ile verilir (`itx12_buyuk_avx2_fn` /
    // `itx12_buyuk_avx2_lookup`); belirteç AVX2 tespiti ve maske izninin kanıtıdır. (2) sınır:
    // `itx16_sozlesmesi` `dst_stride`'ın çift ve `≥ w·2` olduğunu, `dst`'nin `(h − 1)·dst_stride + w·2`
    // baytı ve `coeff`'un `1024` öğeyi kapsadığını garanti eder; bütün bellek erişimleri sabit
    // boyutlu `chunk`/`chunk_mut` dizileri üzerindendir. (3) hizalama/takma ad: `dst` ve `coeff`
    // ayrık `&mut` referanslardır, hizasız erişim kullanılır.
    unsafe {
        itx12_buyuk_avx2_cekirdek::<TX, TXTP>(dst, dst_stride, coeff, eob);
    }
}

/// `max(w, h) == 32` boyutlu 7 adet 12 bpc AVX2 `itxfm_add[tx][txtp]` yuvası
/// (yalnız [`Avx2Token`] ile). En büyük 8 `DCT_DCT` dönüşümü `None` döner (skalerde kalır).
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
        (TxSize::Rtx16x32, TxType::Idtx) => {
            Some(itx12_buyuk_avx2_giris::<{ TxSize::Rtx16x32 as u8 }, { TxType::Idtx as u8 }>)
        }
        (TxSize::Rtx32x16, TxType::Idtx) => {
            Some(itx12_buyuk_avx2_giris::<{ TxSize::Rtx32x16 as u8 }, { TxType::Idtx as u8 }>)
        }
        (TxSize::Tx32x32, TxType::Idtx) => {
            Some(itx12_buyuk_avx2_giris::<{ TxSize::Tx32x32 as u8 }, { TxType::Idtx as u8 }>)
        }
        _ => None,
    }
}

/// [`itx12_buyuk_avx2_fn`] eşdeğer arama girişi.
#[must_use]
pub const fn itx12_buyuk_avx2_lookup(token: Avx2Token, tx: TxSize, txtp: TxType) -> Option<ItxFn> {
    itx12_buyuk_avx2_fn(token, tx, txtp)
}
