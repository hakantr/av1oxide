//! Büyük boyutlu (`max(w, h) ∈ {32, 64}`, 15 kayıt) 10 bpc ters dönüşüm x86_64 AVX512ICL çekirdekleri — P44 Adım 2.
//!
//! Bu modül dav1d'nin `src/x86/itx.h:292-304` ve `src/x86/itx16_avx512.asm` dosyalarındaki
//! `max(w, h) ∈ {32, 64}` boyutlu **15** adet **yalnız 10 bpc** (`BitDepth::Ten`) AVX512ICL
//! `itxfm_add` kaydını **saf Rust intrinsics** ile gerçekler; dış `.asm`/`.S`, `asm!` ve
//! `global_asm!` yoktur (CEK-02):
//!
//! - `RTX_8X32`, `RTX_32X8`, `RTX_16X32`, `RTX_32X16`, `TX_32X32`: `DCT_DCT` ve `IDTX` (`5 × 2 = 10`),
//! - `RTX_16X64`, `RTX_64X16`, `RTX_32X64`, `RTX_64X32`, `TX_64X64`: `DCT_DCT` (`5 × 1 = 5`).
//!
//! ## Mimari, Zen 4 (`0` gather/scatter), VNNI + 512-bit ZMM ve yığın çerçevesi (`< 0x400`)
//!
//! - **P34 modül izolasyonu:** Başka `dsp/x86/*` çekirdek modülleriyle hiçbir `#[target_feature]`
//!   yardımcısı paylaşılmaz; bütün SIMD işlevleri
//!   `#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]` taşır.
//! - **Zen 4 kuralı (`0` gather/scatter):** `vpgather*` ve `vpscatter*` yönergeleri kesinlikle
//!   üretilmez (`0` adet); `SLOW_GATHER` altında da güvenle kurulur.
//! - **Giriş sözleşmesi:** [`itx16_sozlesmesi`] ile `(tx, txtp, bit_depth == Ten)` uyumu aranır;
//!   `BitDepth::Eight`, `BitDepth::Twelve` veya sözleşme dışı girdide paniklemeden
//!   [`itx_skaler_devret`] üzerinden ilgili bit derinliğinin skaler monomorfuna devredilir.
//! - **Pass 1 (`8 × i32` YMM):** `[-131072, +131071]` (`18` bit `row_clip`) ile 32-bit şeritlerde
//!   çalışır; çıkışında `(v + rnd) >> shift` ve `_mm256_packs_epi32` ile `[-32768, +32767]`
//!   (`col_clip`) doyurmalı `i16` düzenine daraltılıp transpoz edilerek `coeff[..sw * sh]`
//!   içinde aşamalandırılır.
//! - **Pass 2 (`32 × i16` ZMM `z32`, `16 × i16` YMM `x16` ve `8 × i16` Split-YMM `x8`):**
//!   AVX-512 VNNI (`_mm512_dpwssd_epi32` / `_mm256_dpwssd_epi32`) kelebekleriyle 16-bit
//!   aritmetikte tamamlanıp `(c + 8) >> 4` yuvarlaması ve `[0, 1023]` kırpmasıyla hedef `u16`
//!   düzleme eklenir (`W >= 32` DC-only ve `h ∈ {8, 16}` geçişlerinde `64` baytlık ZMM
//!   `load_u8x64` / `store_u8x64` kullanılır).
//! - **Yığın disiplini (`sub < $0x400, %rsp`):** Çoklu vektör 1B dönüşüm ve transpoz adımları
//!   yazmaç değişkenleri üzerinde makrolarla açılır ve ara veriler `coeff[..sw * sh]` içinde
//!   tutulup sonunda sıfırlanır.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm_adds_epi16, _mm_max_epi16, _mm_min_epi16, _mm_set1_epi16,
    _mm_setzero_si128, _mm_unpacklo_epi64, _mm256_add_epi32, _mm256_adds_epi16,
    _mm256_castsi256_si128, _mm256_dpwssd_epi32, _mm256_extracti128_si256, _mm256_max_epi16,
    _mm256_max_epi32, _mm256_min_epi16, _mm256_min_epi32, _mm256_mulhrs_epi16, _mm256_mullo_epi32,
    _mm256_packs_epi32, _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_setr_epi32, _mm256_setzero_si256, _mm256_srai_epi32,
    _mm256_srli_si256, _mm256_sub_epi32, _mm256_subs_epi16, _mm256_unpackhi_epi16,
    _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi16, _mm256_unpacklo_epi32,
    _mm256_unpacklo_epi64, _mm512_add_epi32, _mm512_adds_epi16, _mm512_castsi256_si512,
    _mm512_dpwssd_epi32, _mm512_inserti64x4, _mm512_max_epi16, _mm512_min_epi16,
    _mm512_mulhrs_epi16, _mm512_packs_epi32, _mm512_set1_epi16, _mm512_set1_epi32,
    _mm512_setzero_si512, _mm512_srai_epi32, _mm512_subs_epi16, _mm512_unpackhi_epi16,
    _mm512_unpacklo_epi16,
};

use super::mem::{
    chunk, chunk_mut, load_u8x16, load_u8x32, load_u8x64, store_i32x8, store_u8x16, store_u8x32,
    store_u8x64,
};
use crate::cpu::Avx512IclToken;
use crate::dsp::itx::{ItxFn, TxSize, TxType, itx_skaler_devret, itx16_sozlesmesi};
use crate::frame::BitDepth;

#[inline(always)]
const fn cift_katsayi(c0: i16, c1: i16) -> i32 {
    (c0 as u16 as i32) | ((c1 as u16 as i32) << 16)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_i32x8(coeff: &[i32], off: usize) -> __m256i {
    let c = chunk::<i32, 8>(coeff, off);
    _mm256_setr_epi32(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7])
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn sakla_i32x8(coeff: &mut [i32], off: usize, v: __m256i) {
    store_i32x8(chunk_mut(coeff, off), v);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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

// ---------------------------------------------------------------------------
// Pass 1 (`8 × i32` YMM) yaprak yardımcıları
// ---------------------------------------------------------------------------

mod p1 {
    use super::*;

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn zero() -> __m256i {
        _mm256_setzero_si256()
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn olcekle_181(v: __m256i) -> __m256i {
        _mm256_srai_epi32::<8>(_mm256_add_epi32(
            _mm256_mullo_epi32(v, _mm256_set1_epi32(181)),
            _mm256_set1_epi32(128),
        ))
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn yukle_opt<const RECT2: bool>(coeff: &[i32], off: usize) -> __m256i {
        let v = yukle_i32x8(coeff, off);
        if RECT2 { olcekle_181(v) } else { v }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn kirp(v: __m256i) -> __m256i {
        _mm256_min_epi32(
            _mm256_max_epi32(v, _mm256_set1_epi32(-131_072)),
            _mm256_set1_epi32(131_071),
        )
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn adds(a: __m256i, b: __m256i) -> __m256i {
        kirp(_mm256_add_epi32(a, b))
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn subs(a: __m256i, b: __m256i) -> __m256i {
        kirp(_mm256_sub_epi32(a, b))
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn mul_r2(v: __m256i) -> __m256i {
        olcekle_181(v)
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn mul12(a: __m256i, k: i16) -> __m256i {
        _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_mullo_epi32(a, _mm256_set1_epi32(i32::from(k))),
            _mm256_set1_epi32(2048),
        ))
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn mulsub(a: __m256i, b: __m256i, c1: i16, c2: i16) -> (__m256i, __m256i) {
        let rnd = _mm256_set1_epi32(2048);
        let k1 = _mm256_set1_epi32(i32::from(c1));
        let k2 = _mm256_set1_epi32(i32::from(c2));
        let u = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_sub_epi32(_mm256_mullo_epi32(a, k1), _mm256_mullo_epi32(b, k2)),
            rnd,
        ));
        let v = _mm256_srai_epi32::<12>(_mm256_add_epi32(
            _mm256_add_epi32(_mm256_mullo_epi32(a, k2), _mm256_mullo_epi32(b, k1)),
            rnd,
        ));
        (u, v)
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn rot_r2(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
        (
            olcekle_181(_mm256_sub_epi32(a, b)),
            olcekle_181(_mm256_add_epi32(a, b)),
        )
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn add_raw_r2(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
        (
            olcekle_181(_mm256_sub_epi32(a, b)),
            olcekle_181(_mm256_add_epi32(a, b)),
        )
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn kaydir<const SHIFT: u32>(v: __m256i) -> __m256i {
        match SHIFT {
            1 => _mm256_srai_epi32::<1>(_mm256_add_epi32(v, _mm256_set1_epi32(1))),
            2 => _mm256_srai_epi32::<2>(_mm256_add_epi32(v, _mm256_set1_epi32(2))),
            _ => unreachable!(),
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn paket_2col<const SHIFT: u32>(c_lo: __m256i, c_hi: __m256i) -> __m256i {
        _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(
            kaydir::<SHIFT>(c_lo),
            kaydir::<SHIFT>(c_hi),
        ))
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn paket_split<const SHIFT: u32>(c: __m256i) -> __m256i {
        _mm256_packs_epi32(kaydir::<SHIFT>(c), _mm256_setzero_si256())
    }
}

// ---------------------------------------------------------------------------
// Pass 2 (`32 × i16` ZMM `z32`, `16 × i16` YMM `x16`, `8 × i16` Split-YMM `x8`)
// ---------------------------------------------------------------------------

macro_rules! p2_yaprak {
    ($mod_ad:ident, $HALF_LANE:literal) => {
        #[allow(dead_code)]
        mod $mod_ad {
            use super::*;

            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn zero() -> __m256i {
                _mm256_setzero_si256()
            }

            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn adds(a: __m256i, b: __m256i) -> __m256i {
                _mm256_adds_epi16(a, b)
            }

            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn subs(a: __m256i, b: __m256i) -> __m256i {
                _mm256_subs_epi16(a, b)
            }

            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn mul_r2(v: __m256i) -> __m256i {
                _mm256_mulhrs_epi16(v, _mm256_set1_epi16(2896 * 8))
            }

            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn mul12(v: __m256i, c: i16) -> __m256i {
                _mm256_mulhrs_epi16(v, _mm256_set1_epi16(c * 8))
            }

            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn mulsub(a: __m256i, b: __m256i, c1: i16, c2: i16) -> (__m256i, __m256i) {
                let lo = _mm256_unpacklo_epi16(b, a);
                let k1 = _mm256_set1_epi32(cift_katsayi(c2.wrapping_neg(), c1));
                let k2 = _mm256_set1_epi32(cift_katsayi(c1, c2));
                let rnd = _mm256_set1_epi32(2048);
                let z0 = _mm256_setzero_si256();
                let r1_lo =
                    _mm256_srai_epi32::<12>(_mm256_add_epi32(_mm256_dpwssd_epi32(z0, lo, k1), rnd));
                let r2_lo =
                    _mm256_srai_epi32::<12>(_mm256_add_epi32(_mm256_dpwssd_epi32(z0, lo, k2), rnd));
                if $HALF_LANE {
                    (
                        _mm256_packs_epi32(r1_lo, r1_lo),
                        _mm256_packs_epi32(r2_lo, r2_lo),
                    )
                } else {
                    let hi = _mm256_unpackhi_epi16(b, a);
                    let r1_hi = _mm256_srai_epi32::<12>(_mm256_add_epi32(
                        _mm256_dpwssd_epi32(z0, hi, k1),
                        rnd,
                    ));
                    let r2_hi = _mm256_srai_epi32::<12>(_mm256_add_epi32(
                        _mm256_dpwssd_epi32(z0, hi, k2),
                        rnd,
                    ));
                    (
                        _mm256_packs_epi32(r1_lo, r1_hi),
                        _mm256_packs_epi32(r2_lo, r2_hi),
                    )
                }
            }

            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn rot_r2(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
                mulsub(a, b, 2896, 2896)
            }

            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn add_raw_r2(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
                mulsub(a, b, 2896, 2896)
            }

            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn round_p2(v: __m256i) -> __m256i {
                _mm256_mulhrs_epi16(v, _mm256_set1_epi16(2048))
            }
        }
    };
}

p2_yaprak!(x8, true);
p2_yaprak!(x16, false);

mod z32 {
    use super::*;

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn adds(a: __m512i, b: __m512i) -> __m512i {
        _mm512_adds_epi16(a, b)
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn subs(a: __m512i, b: __m512i) -> __m512i {
        _mm512_subs_epi16(a, b)
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn mul_r2(v: __m512i) -> __m512i {
        _mm512_mulhrs_epi16(v, _mm512_set1_epi16(2896 * 8))
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn mul12(v: __m512i, c: i16) -> __m512i {
        _mm512_mulhrs_epi16(v, _mm512_set1_epi16(c * 8))
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn mulsub(a: __m512i, b: __m512i, c1: i16, c2: i16) -> (__m512i, __m512i) {
        let lo = _mm512_unpacklo_epi16(b, a);
        let hi = _mm512_unpackhi_epi16(b, a);
        let k1 = _mm512_set1_epi32(cift_katsayi(c2.wrapping_neg(), c1));
        let k2 = _mm512_set1_epi32(cift_katsayi(c1, c2));
        let rnd = _mm512_set1_epi32(2048);
        let z0 = _mm512_setzero_si512();
        let r1_lo = _mm512_srai_epi32::<12>(_mm512_add_epi32(_mm512_dpwssd_epi32(z0, lo, k1), rnd));
        let r2_lo = _mm512_srai_epi32::<12>(_mm512_add_epi32(_mm512_dpwssd_epi32(z0, lo, k2), rnd));
        let r1_hi = _mm512_srai_epi32::<12>(_mm512_add_epi32(_mm512_dpwssd_epi32(z0, hi, k1), rnd));
        let r2_hi = _mm512_srai_epi32::<12>(_mm512_add_epi32(_mm512_dpwssd_epi32(z0, hi, k2), rnd));
        (
            _mm512_packs_epi32(r1_lo, r1_hi),
            _mm512_packs_epi32(r2_lo, r2_hi),
        )
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn rot_r2(a: __m512i, b: __m512i) -> (__m512i, __m512i) {
        mulsub(a, b, 2896, 2896)
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn add_raw_r2(a: __m512i, b: __m512i) -> (__m512i, __m512i) {
        mulsub(a, b, 2896, 2896)
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[inline]
    pub(super) fn round_p2(v: __m512i) -> __m512i {
        _mm512_mulhrs_epi16(v, _mm512_set1_epi16(2048))
    }
}

// ---------------------------------------------------------------------------
// Yazmaç-değişkeni 1B ters DCT makroları (`p1`, `x8`, `x16`, `z32` ortak)
// ---------------------------------------------------------------------------

macro_rules! idct4_v {
    ($m:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident) => {{
        let (t1, t0) = $m::add_raw_r2($c0, $c2);
        let (t2, t3) = $m::mulsub($c1, $c3, 1567, 3784);
        $c0 = $m::adds(t0, t3);
        $c1 = $m::adds(t1, t2);
        $c2 = $m::subs(t1, t2);
        $c3 = $m::subs(t0, t3);
    }};
}

macro_rules! idct4_half_v {
    ($m:ident, $i0:expr, $i1:expr => $o0:ident, $o1:ident, $o2:ident, $o3:ident) => {
        let i0_v = $i0;
        let i1_v = $i1;
        let t0 = $m::mul_r2(i0_v);
        let t2 = $m::mul12(i1_v, 1567);
        let t3 = $m::mul12(i1_v, 3784);
        let $o0 = $m::adds(t0, t3);
        let $o1 = $m::adds(t0, t2);
        let $o2 = $m::subs(t0, t2);
        let $o3 = $m::subs(t0, t3);
    };
}

macro_rules! idct8_finish_v {
    ($m:ident, $e0:expr, $e1:expr, $e2:expr, $e3:expr, $t4a:expr, $t5a:expr, $t6a:expr, $t7a:expr
     => $o0:ident, $o1:ident, $o2:ident, $o3:ident, $o4:ident, $o5:ident, $o6:ident, $o7:ident) => {
        let (e0_v, e1_v, e2_v, e3_v) = ($e0, $e1, $e2, $e3);
        let (t4a_v, t5a_v, t6a_v, t7a_v) = ($t4a, $t5a, $t6a, $t7a);
        let t4 = $m::adds(t4a_v, t5a_v);
        let t5_in = $m::subs(t4a_v, t5a_v);
        let t7 = $m::adds(t7a_v, t6a_v);
        let t6_in = $m::subs(t7a_v, t6a_v);
        let (t5, t6) = $m::rot_r2(t6_in, t5_in);
        let $o0 = $m::adds(e0_v, t7);
        let $o1 = $m::adds(e1_v, t6);
        let $o2 = $m::adds(e2_v, t5);
        let $o3 = $m::adds(e3_v, t4);
        let $o4 = $m::subs(e3_v, t4);
        let $o5 = $m::subs(e2_v, t5);
        let $o6 = $m::subs(e1_v, t6);
        let $o7 = $m::subs(e0_v, t7);
    };
}

macro_rules! idct8_v {
    ($m:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident) => {{
        let (mut e0, mut e1, mut e2, mut e3) = ($c0, $c2, $c4, $c6);
        idct4_v!($m, e0, e1, e2, e3);
        let (t4a, t7a) = $m::mulsub($c1, $c7, 799, 4017);
        let (t5a, t6a) = $m::mulsub($c5, $c3, 3406, 2276);
        idct8_finish_v!($m, e0, e1, e2, e3, t4a, t5a, t6a, t7a => r0, r1, r2, r3, r4, r5, r6, r7);
        $c0 = r0;
        $c1 = r1;
        $c2 = r2;
        $c3 = r3;
        $c4 = r4;
        $c5 = r5;
        $c6 = r6;
        $c7 = r7;
    }};
}

macro_rules! idct8_half_v {
    ($m:ident, $i0:expr, $i1:expr, $i2:expr, $i3:expr
     => $o0:ident, $o1:ident, $o2:ident, $o3:ident, $o4:ident, $o5:ident, $o6:ident, $o7:ident) => {
        let (in0, in1, in2, in3) = ($i0, $i1, $i2, $i3);
        idct4_half_v!($m, in0, in2 => ev0, ev1, ev2, ev3);
        let t4a = $m::mul12(in1, 799);
        let t5a = $m::mul12(in3, -2276);
        let t6a = $m::mul12(in3, 3406);
        let t7a = $m::mul12(in1, 4017);
        idct8_finish_v!($m, ev0, ev1, ev2, ev3, t4a, t5a, t6a, t7a
            => $o0, $o1, $o2, $o3, $o4, $o5, $o6, $o7);
    };
}

macro_rules! idct16_finish_v {
    ($m:ident,
     $e0:expr, $e1:expr, $e2:expr, $e3:expr, $e4:expr, $e5:expr, $e6:expr, $e7:expr,
     $t8a:expr, $t9a:expr, $t10a:expr, $t11a:expr, $t12a:expr, $t13a:expr, $t14a:expr, $t15a:expr
     => $o0:ident, $o1:ident, $o2:ident, $o3:ident, $o4:ident, $o5:ident, $o6:ident, $o7:ident,
        $o8:ident, $o9:ident, $o10:ident, $o11:ident, $o12:ident, $o13:ident, $o14:ident, $o15:ident) => {
        let (t0, t1, t2, t3, t4, t5, t6, t7) = ($e0, $e1, $e2, $e3, $e4, $e5, $e6, $e7);
        let (a8, a9, a10, a11, a12, a13, a14, a15) =
            ($t8a, $t9a, $t10a, $t11a, $t12a, $t13a, $t14a, $t15a);
        let t8 = $m::adds(a8, a9);
        let t9 = $m::subs(a8, a9);
        let t10 = $m::subs(a11, a10);
        let t11 = $m::adds(a11, a10);
        let t12 = $m::adds(a12, a13);
        let t13 = $m::subs(a12, a13);
        let t14 = $m::subs(a15, a14);
        let t15 = $m::adds(a15, a14);

        let (t9a, t14a) = $m::mulsub(t14, t9, 1567, 3784);
        let (t10a, t13a) = $m::mulsub(t13, t10, -3784, 1567);

        let t8a = $m::adds(t8, t11);
        let t9_2 = $m::adds(t9a, t10a);
        let t10_2 = $m::subs(t9a, t10a);
        let t11a = $m::subs(t8, t11);
        let t12a = $m::subs(t15, t12);
        let t13_2 = $m::subs(t14a, t13a);
        let t14_2 = $m::adds(t14a, t13a);
        let t15a = $m::adds(t15, t12);

        let (t10_f, t13_f) = $m::rot_r2(t13_2, t10_2);
        let (t11_f, t12_f) = $m::rot_r2(t12a, t11a);

        let $o0 = $m::adds(t0, t15a);
        let $o1 = $m::adds(t1, t14_2);
        let $o2 = $m::adds(t2, t13_f);
        let $o3 = $m::adds(t3, t12_f);
        let $o4 = $m::adds(t4, t11_f);
        let $o5 = $m::adds(t5, t10_f);
        let $o6 = $m::adds(t6, t9_2);
        let $o7 = $m::adds(t7, t8a);
        let $o8 = $m::subs(t7, t8a);
        let $o9 = $m::subs(t6, t9_2);
        let $o10 = $m::subs(t5, t10_f);
        let $o11 = $m::subs(t4, t11_f);
        let $o12 = $m::subs(t3, t12_f);
        let $o13 = $m::subs(t2, t13_f);
        let $o14 = $m::subs(t1, t14_2);
        let $o15 = $m::subs(t0, t15a);
    };
}

macro_rules! idct16_v {
    ($m:ident,
     $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident,
     $c8:ident, $c9:ident, $c10:ident, $c11:ident, $c12:ident, $c13:ident, $c14:ident, $c15:ident) => {{
        let (mut e0, mut e1, mut e2, mut e3, mut e4, mut e5, mut e6, mut e7) =
            ($c0, $c2, $c4, $c6, $c8, $c10, $c12, $c14);
        idct8_v!($m, e0, e1, e2, e3, e4, e5, e6, e7);
        let (t8a, t15a) = $m::mulsub($c1, $c15, 401, 4076);
        let (t9a, t14a) = $m::mulsub($c9, $c7, 3166, 2598);
        let (t10a, t13a) = $m::mulsub($c5, $c11, 1931, 3612);
        let (t11a, t12a) = $m::mulsub($c13, $c3, 3920, 1189);
        idct16_finish_v!(
            $m, e0, e1, e2, e3, e4, e5, e6, e7,
            t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a
            => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
        );
        $c0 = r0;
        $c1 = r1;
        $c2 = r2;
        $c3 = r3;
        $c4 = r4;
        $c5 = r5;
        $c6 = r6;
        $c7 = r7;
        $c8 = r8;
        $c9 = r9;
        $c10 = r10;
        $c11 = r11;
        $c12 = r12;
        $c13 = r13;
        $c14 = r14;
        $c15 = r15;
    }};
}

macro_rules! idct16_half_v {
    ($m:ident,
     $i0:expr, $i1:expr, $i2:expr, $i3:expr, $i4:expr, $i5:expr, $i6:expr, $i7:expr
     => $o0:ident, $o1:ident, $o2:ident, $o3:ident, $o4:ident, $o5:ident, $o6:ident, $o7:ident,
        $o8:ident, $o9:ident, $o10:ident, $o11:ident, $o12:ident, $o13:ident, $o14:ident, $o15:ident) => {
        let (in0, in1, in2, in3, in4, in5, in6, in7) = ($i0, $i1, $i2, $i3, $i4, $i5, $i6, $i7);
        idct8_half_v!($m, in0, in2, in4, in6 => e0, e1, e2, e3, e4, e5, e6, e7);
        let t8a = $m::mul12(in1, 401);
        let t9a = $m::mul12(in7, -2598);
        let t10a = $m::mul12(in5, 1931);
        let t11a = $m::mul12(in3, -1189);
        let t12a = $m::mul12(in3, 3920);
        let t13a = $m::mul12(in5, 3612);
        let t14a = $m::mul12(in7, 3166);
        let t15a = $m::mul12(in1, 4076);
        idct16_finish_v!(
            $m, e0, e1, e2, e3, e4, e5, e6, e7,
            t8a, t9a, t10a, t11a, t12a, t13a, t14a, t15a
            => $o0, $o1, $o2, $o3, $o4, $o5, $o6, $o7,
               $o8, $o9, $o10, $o11, $o12, $o13, $o14, $o15
        );
    };
}

macro_rules! idct16_qtr_v {
    ($m:ident, $i0:expr, $i1:expr, $i2:expr, $i3:expr
     => $o0:ident, $o1:ident, $o2:ident, $o3:ident, $o4:ident, $o5:ident, $o6:ident, $o7:ident,
        $o8:ident, $o9:ident, $o10:ident, $o11:ident, $o12:ident, $o13:ident, $o14:ident, $o15:ident) => {
        let (in0, in1, in2, in3) = ($i0, $i1, $i2, $i3);
        let z = $m::zero();
        idct8_half_v!($m, in0, in2, z, z => e0, e1, e2, e3, e4, e5, e6, e7);
        let t8a = $m::mul12(in1, 401);
        let t11a = $m::mul12(in3, -1189);
        let t12a = $m::mul12(in3, 3920);
        let t15a = $m::mul12(in1, 4076);
        idct16_finish_v!(
            $m, e0, e1, e2, e3, e4, e5, e6, e7,
            t8a, z, z, t11a, t12a, z, z, t15a
            => $o0, $o1, $o2, $o3, $o4, $o5, $o6, $o7,
               $o8, $o9, $o10, $o11, $o12, $o13, $o14, $o15
        );
    };
}

macro_rules! idct32_odd_finish_v {
    ($m:ident,
     $o0:ident, $o1:ident, $o2:ident, $o3:ident, $o4:ident, $o5:ident, $o6:ident, $o7:ident,
     $o8:ident, $o9:ident, $o10:ident, $o11:ident, $o12:ident, $o13:ident, $o14:ident, $o15:ident) => {{
        let t16 = $m::adds($o0, $o1);
        let t17 = $m::subs($o0, $o1);
        let t18 = $m::subs($o3, $o2);
        let t19 = $m::adds($o3, $o2);
        let t20 = $m::adds($o4, $o5);
        let t21 = $m::subs($o4, $o5);
        let t22 = $m::subs($o7, $o6);
        let t23 = $m::adds($o7, $o6);
        let t24 = $m::adds($o8, $o9);
        let t25 = $m::subs($o8, $o9);
        let t26 = $m::subs($o11, $o10);
        let t27 = $m::adds($o11, $o10);
        let t28 = $m::adds($o12, $o13);
        let t29 = $m::subs($o12, $o13);
        let t30 = $m::subs($o15, $o14);
        let t31 = $m::adds($o15, $o14);

        let (t17a, t30a) = $m::mulsub(t30, t17, 799, 4017);
        let (t18a, t29a) = $m::mulsub(t29, t18, -4017, 799);
        let (t21a, t26a) = $m::mulsub(t26, t21, 3406, 2276);
        let (t22a, t25a) = $m::mulsub(t25, t22, -2276, 3406);

        let t16a = $m::adds(t16, t19);
        let t17 = $m::adds(t17a, t18a);
        let t18 = $m::subs(t17a, t18a);
        let t19a = $m::subs(t16, t19);
        let t20a = $m::subs(t23, t20);
        let t21 = $m::subs(t22a, t21a);
        let t22 = $m::adds(t22a, t21a);
        let t23a = $m::adds(t23, t20);
        let t24a = $m::adds(t24, t27);
        let t25 = $m::adds(t25a, t26a);
        let t26 = $m::subs(t25a, t26a);
        let t27a = $m::subs(t24, t27);
        let t28a = $m::subs(t31, t28);
        let t29 = $m::subs(t30a, t29a);
        let t30 = $m::adds(t30a, t29a);
        let t31a = $m::adds(t31, t28);

        let (t18a, t29a) = $m::mulsub(t29, t18, 1567, 3784);
        let (t19, t28) = $m::mulsub(t28a, t19a, 1567, 3784);
        let (t20, t27) = $m::mulsub(t27a, t20a, -3784, 1567);
        let (t21a, t26a) = $m::mulsub(t26, t21, -3784, 1567);

        let t16 = $m::adds(t16a, t23a);
        let t17a = $m::adds(t17, t22);
        let t18 = $m::adds(t18a, t21a);
        let t19a = $m::adds(t19, t20);
        let t20a = $m::subs(t19, t20);
        let t21 = $m::subs(t18a, t21a);
        let t22a = $m::subs(t17, t22);
        let t23 = $m::subs(t16a, t23a);
        let t24 = $m::subs(t31a, t24a);
        let t25a = $m::subs(t30, t25);
        let t26 = $m::subs(t29a, t26a);
        let t27a = $m::subs(t28, t27);
        let t28a = $m::adds(t28, t27);
        let t29 = $m::adds(t29a, t26a);
        let t30a = $m::adds(t30, t25);
        let t31 = $m::adds(t31a, t24a);

        let (t20, t27) = $m::rot_r2(t27a, t20a);
        let (t21a, t26a) = $m::rot_r2(t26, t21);
        let (t22, t25) = $m::rot_r2(t25a, t22a);
        let (t23a, t24a) = $m::rot_r2(t24, t23);

        $o0 = t31;
        $o1 = t30a;
        $o2 = t29;
        $o3 = t28a;
        $o4 = t27;
        $o5 = t26a;
        $o6 = t25;
        $o7 = t24a;
        $o8 = t23a;
        $o9 = t22;
        $o10 = t21a;
        $o11 = t20;
        $o12 = t19a;
        $o13 = t18;
        $o14 = t17a;
        $o15 = t16;
    }};
}

macro_rules! idct32_full_odd_v {
    ($m:ident,
     $t1:expr, $t3:expr, $t5:expr, $t7:expr, $t9:expr, $t11:expr, $t13:expr, $t15:expr,
     $b1:expr, $b3:expr, $b5:expr, $b7:expr, $b9:expr, $b11:expr, $b13:expr, $b15:expr
     => $o0:ident, $o1:ident, $o2:ident, $o3:ident, $o4:ident, $o5:ident, $o6:ident, $o7:ident,
        $o8:ident, $o9:ident, $o10:ident, $o11:ident, $o12:ident, $o13:ident, $o14:ident, $o15:ident) => {
        let (mut $o0, mut $o15) = $m::mulsub($t1, $b15, 201, 4091);
        let (mut $o1, mut $o14) = $m::mulsub($b1, $t15, 3035, 2751);
        let (mut $o2, mut $o13) = $m::mulsub($t9, $b7, 1751, 3703);
        let (mut $o3, mut $o12) = $m::mulsub($b9, $t7, 3857, 1380);
        let (mut $o4, mut $o11) = $m::mulsub($t5, $b11, 995, 3973);
        let (mut $o5, mut $o10) = $m::mulsub($b5, $t11, 3513, 2106);
        let (mut $o6, mut $o9) = $m::mulsub($t13, $b3, 2440, 3290);
        let (mut $o7, mut $o8) = $m::mulsub($b13, $t3, 4052, 601);
        idct32_odd_finish_v!(
            $m, $o0, $o1, $o2, $o3, $o4, $o5, $o6, $o7, $o8, $o9, $o10, $o11, $o12, $o13, $o14,
            $o15
        );
    };
}

macro_rules! idct32_half_odd_v {
    ($m:ident, $i1:expr, $i3:expr, $i5:expr, $i7:expr, $i9:expr, $i11:expr, $i13:expr, $i15:expr
     => $o0:ident, $o1:ident, $o2:ident, $o3:ident, $o4:ident, $o5:ident, $o6:ident, $o7:ident,
        $o8:ident, $o9:ident, $o10:ident, $o11:ident, $o12:ident, $o13:ident, $o14:ident, $o15:ident) => {
        let (q1, q3, q5, q7, q9, q11, q13, q15) = ($i1, $i3, $i5, $i7, $i9, $i11, $i13, $i15);
        let mut $o0 = $m::mul12(q1, 201);
        let mut $o1 = $m::mul12(q15, -2751);
        let mut $o2 = $m::mul12(q9, 1751);
        let mut $o3 = $m::mul12(q7, -1380);
        let mut $o4 = $m::mul12(q5, 995);
        let mut $o5 = $m::mul12(q11, -2106);
        let mut $o6 = $m::mul12(q13, 2440);
        let mut $o7 = $m::mul12(q3, -601);
        let mut $o8 = $m::mul12(q3, 4052);
        let mut $o9 = $m::mul12(q13, 3290);
        let mut $o10 = $m::mul12(q11, 3513);
        let mut $o11 = $m::mul12(q5, 3973);
        let mut $o12 = $m::mul12(q7, 3857);
        let mut $o13 = $m::mul12(q9, 3703);
        let mut $o14 = $m::mul12(q15, 3035);
        let mut $o15 = $m::mul12(q1, 4091);
        idct32_odd_finish_v!(
            $m, $o0, $o1, $o2, $o3, $o4, $o5, $o6, $o7, $o8, $o9, $o10, $o11, $o12, $o13, $o14,
            $o15
        );
    };
}

macro_rules! idct32_qtr_odd_v {
    ($m:ident, $i1:expr, $i3:expr, $i5:expr, $i7:expr
     => $o0:ident, $o1:ident, $o2:ident, $o3:ident, $o4:ident, $o5:ident, $o6:ident, $o7:ident,
        $o8:ident, $o9:ident, $o10:ident, $o11:ident, $o12:ident, $o13:ident, $o14:ident, $o15:ident) => {
        let (q1, q3, q5, q7) = ($i1, $i3, $i5, $i7);
        let z = $m::zero();
        let mut $o0 = $m::mul12(q1, 201);
        let mut $o1 = z;
        let mut $o2 = z;
        let mut $o3 = $m::mul12(q7, -1380);
        let mut $o4 = $m::mul12(q5, 995);
        let mut $o5 = z;
        let mut $o6 = z;
        let mut $o7 = $m::mul12(q3, -601);
        let mut $o8 = $m::mul12(q3, 4052);
        let mut $o9 = z;
        let mut $o10 = z;
        let mut $o11 = $m::mul12(q5, 3973);
        let mut $o12 = $m::mul12(q7, 3857);
        let mut $o13 = z;
        let mut $o14 = z;
        let mut $o15 = $m::mul12(q1, 4091);
        idct32_odd_finish_v!(
            $m, $o0, $o1, $o2, $o3, $o4, $o5, $o6, $o7, $o8, $o9, $o10, $o11, $o12, $o13, $o14,
            $o15
        );
    };
}

macro_rules! idct64_odd_ic_core_v {
    ($m:ident,
     $o8:expr, $o9:expr, $o10:expr, $o11:expr, $o12:expr, $o13:expr, $o14:expr, $o15:expr,
     $o16:expr, $o17:expr, $o18:expr, $o19:expr, $o20:expr, $o21:expr, $o22:expr, $o23:expr,
     $tek:ident) => {{
        let (o8, o9, o10, o11, o12, o13, o14, o15) = ($o8, $o9, $o10, $o11, $o12, $o13, $o14, $o15);
        let (o16, o17, o18, o19, o20, o21, o22, o23) =
            ($o16, $o17, $o18, $o19, $o20, $o21, $o22, $o23);
        let t40 = $m::adds(o8, o9);
        let t41 = $m::subs(o8, o9);
        let t42 = $m::subs(o11, o10);
        let t43 = $m::adds(o11, o10);
        let t44 = $m::adds(o12, o13);
        let t45 = $m::subs(o12, o13);
        let t46 = $m::subs(o15, o14);
        let t47 = $m::adds(o15, o14);
        let t48 = $m::adds(o16, o17);
        let t49 = $m::subs(o16, o17);
        let t50 = $m::subs(o19, o18);
        let t51 = $m::adds(o19, o18);
        let t52 = $m::adds(o20, o21);
        let t53 = $m::subs(o20, o21);
        let t54 = $m::subs(o23, o22);
        let t55 = $m::adds(o23, o22);

        let (t41a, t54a) = $m::mulsub(t54, t41, 1931, 3612);
        let (t42a, t53a) = $m::mulsub(t53, t42, -3612, 1931);
        let (t45a, t50a) = $m::mulsub(t50, t45, 3920, 1189);
        let (t46a, t49a) = $m::mulsub(t49, t46, -1189, 3920);

        let t40a = $m::adds(t40, t43);
        let t41 = $m::adds(t41a, t42a);
        let t42 = $m::subs(t41a, t42a);
        let t43a = $m::subs(t40, t43);
        let t44a = $m::subs(t47, t44);
        let t45 = $m::subs(t46a, t45a);
        let t46 = $m::adds(t46a, t45a);
        let t47a = $m::adds(t47, t44);
        let t48a = $m::adds(t48, t51);
        let t49 = $m::adds(t49a, t50a);
        let t50 = $m::subs(t49a, t50a);
        let t51a = $m::subs(t48, t51);
        let t52a = $m::subs(t55, t52);
        let t53 = $m::subs(t54a, t53a);
        let t54 = $m::adds(t54a, t53a);
        let t55a = $m::adds(t55, t52);

        let (t42a, t53a) = $m::mulsub(t53, t42, 3406, 2276);
        let (t43, t52) = $m::mulsub(t52a, t43a, 3406, 2276);
        let (t44, t51) = $m::mulsub(t51a, t44a, -2276, 3406);
        let (t45a, t50a) = $m::mulsub(t50, t45, -2276, 3406);

        let t40 = $m::subs(t47a, t40a);
        let t41a = $m::subs(t46, t41);
        let t42 = $m::subs(t45a, t42a);
        let t43a = $m::subs(t44, t43);
        let t44a = $m::adds(t44, t43);
        let t45 = $m::adds(t45a, t42a);
        let t46a = $m::adds(t46, t41);
        let t47 = $m::adds(t47a, t40a);
        let t48 = $m::adds(t48a, t55a);
        let t49a = $m::adds(t49, t54);
        let t50 = $m::adds(t50a, t53a);
        let t51a = $m::adds(t51, t52);
        let t52a = $m::subs(t51, t52);
        let t53 = $m::subs(t50a, t53a);
        let t54a = $m::subs(t49, t54);
        let t55 = $m::subs(t48a, t55a);

        let (t40a, t55a) = $m::mulsub(t55, t40, -3784, 1567);
        let (t41, t54) = $m::mulsub(t54a, t41a, -3784, 1567);
        let (t42a, t53a) = $m::mulsub(t53, t42, -3784, 1567);
        let (t43, t52) = $m::mulsub(t52a, t43a, -3784, 1567);

        $tek[0] = t40a;
        $tek[1] = t41;
        $tek[2] = t42a;
        $tek[3] = t43;
        $tek[4] = t44a;
        $tek[5] = t45;
        $tek[6] = t46a;
        $tek[7] = t47;
        $tek[8] = t48;
        $tek[9] = t49a;
        $tek[10] = t50;
        $tek[11] = t51a;
        $tek[12] = t52;
        $tek[13] = t53a;
        $tek[14] = t54;
        $tek[15] = t55a;
    }};
}

macro_rules! idct64_odd_dis_stage1_v {
    ($m:ident,
     $o0:expr, $o1:expr, $o2:expr, $o3:expr, $o4:expr, $o5:expr, $o6:expr, $o7:expr,
     $o24:expr, $o25:expr, $o26:expr, $o27:expr, $o28:expr, $o29:expr, $o30:expr, $o31:expr
     => $t32:ident, $t33a:ident, $t34:ident, $t35a:ident, $t36:ident, $t37a:ident, $t38:ident, $t39a:ident,
        $t56a:ident, $t57:ident, $t58a:ident, $t59:ident, $t60a:ident, $t61:ident, $t62a:ident, $t63:ident) => {
        let (o0, o1, o2, o3, o4, o5, o6, o7) = ($o0, $o1, $o2, $o3, $o4, $o5, $o6, $o7);
        let (o24, o25, o26, o27, o28, o29, o30, o31) =
            ($o24, $o25, $o26, $o27, $o28, $o29, $o30, $o31);

        let t32 = $m::adds(o0, o1);
        let t33 = $m::subs(o0, o1);
        let t34 = $m::subs(o3, o2);
        let t35 = $m::adds(o3, o2);
        let t36 = $m::adds(o4, o5);
        let t37 = $m::subs(o4, o5);
        let t38 = $m::subs(o7, o6);
        let t39 = $m::adds(o7, o6);
        let t56 = $m::adds(o24, o25);
        let t57 = $m::subs(o24, o25);
        let t58 = $m::subs(o27, o26);
        let t59 = $m::adds(o27, o26);
        let t60 = $m::adds(o28, o29);
        let t61 = $m::subs(o28, o29);
        let t62 = $m::subs(o31, o30);
        let t63 = $m::adds(o31, o30);

        let (t33a, t62a) = $m::mulsub(t62, t33, 401, 4076);
        let (t34a, t61a) = $m::mulsub(t61, t34, -4076, 401);
        let (t37a, t58a) = $m::mulsub(t58, t37, 3166, 2598);
        let (t38a, t57a) = $m::mulsub(t57, t38, -2598, 3166);

        let t32a = $m::adds(t32, t35);
        let t33 = $m::adds(t33a, t34a);
        let t34 = $m::subs(t33a, t34a);
        let t35a = $m::subs(t32, t35);
        let t36a = $m::subs(t39, t36);
        let t37 = $m::subs(t38a, t37a);
        let t38 = $m::adds(t38a, t37a);
        let t39a = $m::adds(t39, t36);
        let t56a = $m::adds(t56, t59);
        let t57 = $m::adds(t57a, t58a);
        let t58 = $m::subs(t57a, t58a);
        let t59a = $m::subs(t56, t59);
        let t60a = $m::subs(t63, t60);
        let t61 = $m::subs(t62a, t61a);
        let t62 = $m::adds(t62a, t61a);
        let t63a = $m::adds(t63, t60);

        let (t34a, t61a) = $m::mulsub(t61, t34, 799, 4017);
        let (t35, t60) = $m::mulsub(t60a, t35a, 799, 4017);
        let (t36, t59) = $m::mulsub(t59a, t36a, -4017, 799);
        let (t37a, t58a) = $m::mulsub(t58, t37, -4017, 799);

        let $t32 = $m::adds(t32a, t39a);
        let $t33a = $m::adds(t33, t38);
        let $t34 = $m::adds(t34a, t37a);
        let $t35a = $m::adds(t35, t36);
        let t36a = $m::subs(t35, t36);
        let t37 = $m::subs(t34a, t37a);
        let t38a = $m::subs(t33, t38);
        let t39 = $m::subs(t32a, t39a);
        let t56 = $m::subs(t63a, t56a);
        let t57a = $m::subs(t62, t57);
        let t58 = $m::subs(t61a, t58a);
        let t59a = $m::subs(t60, t59);
        let $t60a = $m::adds(t60, t59);
        let $t61 = $m::adds(t61a, t58a);
        let $t62a = $m::adds(t62, t57);
        let $t63 = $m::adds(t63a, t56a);

        let ($t36, $t59) = $m::mulsub(t59a, t36a, 1567, 3784);
        let ($t37a, $t58a) = $m::mulsub(t58, t37, 1567, 3784);
        let ($t38, $t57) = $m::mulsub(t57a, t38a, 1567, 3784);
        let ($t39a, $t56a) = $m::mulsub(t56, t39, 1567, 3784);
    };
}

macro_rules! idct64_odd_ic_yari_fn {
    ($fn_ic:ident, $m:ident) => {
        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
        #[inline(never)]
        fn $fn_ic(yarim: bool, tek: &mut [__m256i; 16], dis_girdi: &mut [__m256i; 8]) {
            dis_girdi[0] = tek[0];
            dis_girdi[1] = tek[3];
            dis_girdi[2] = tek[4];
            dis_girdi[3] = tek[7];
            dis_girdi[4] = tek[8];
            dis_girdi[5] = tek[11];
            dis_girdi[6] = tek[12];
            dis_girdi[7] = tek[15];
            let (q1, q2, q5, q6) = (tek[1], tek[2], tek[5], tek[6]);
            if yarim {
                let z = $m::zero();
                idct64_odd_ic_core_v!(
                    $m,
                    $m::mul12(q2, 501),
                    z,
                    z,
                    $m::mul12(q5, -1092),
                    $m::mul12(q6, 1285),
                    z,
                    z,
                    $m::mul12(q1, -301),
                    $m::mul12(q1, 4085),
                    z,
                    z,
                    $m::mul12(q6, 3889),
                    $m::mul12(q5, 3948),
                    z,
                    z,
                    $m::mul12(q2, 4065),
                    tek
                );
            } else {
                let (q9, q10, q13, q14) = (tek[9], tek[10], tek[13], tek[14]);
                idct64_odd_ic_core_v!(
                    $m,
                    $m::mul12(q2, 501),
                    $m::mul12(q13, -2520),
                    $m::mul12(q10, 2019),
                    $m::mul12(q5, -1092),
                    $m::mul12(q6, 1285),
                    $m::mul12(q9, -1842),
                    $m::mul12(q14, 2675),
                    $m::mul12(q1, -301),
                    $m::mul12(q1, 4085),
                    $m::mul12(q14, 3102),
                    $m::mul12(q9, 3659),
                    $m::mul12(q6, 3889),
                    $m::mul12(q5, 3948),
                    $m::mul12(q10, 3564),
                    $m::mul12(q13, 3229),
                    $m::mul12(q2, 4065),
                    tek
                );
            }
        }
    };
}

macro_rules! idct64_odd_dis_full_stage1_v {
    ($m:ident, $dis:ident
     => $t32:ident, $t33a:ident, $t34:ident, $t35a:ident, $t36:ident, $t37a:ident, $t38:ident, $t39a:ident,
        $t56a:ident, $t57:ident, $t58a:ident, $t59:ident, $t60a:ident, $t61:ident, $t62a:ident, $t63:ident) => {
        let (q0, q3, q4, q7) = ($dis[0], $dis[1], $dis[2], $dis[3]);
        let (q8, q11, q12, q15) = ($dis[4], $dis[5], $dis[6], $dis[7]);
        idct64_odd_dis_stage1_v!(
            $m,
            $m::mul12(q0, 101), $m::mul12(q15, -2824), $m::mul12(q8, 1660), $m::mul12(q7, -1474),
            $m::mul12(q4, 897), $m::mul12(q11, -2191), $m::mul12(q12, 2359), $m::mul12(q3, -700),
            $m::mul12(q3, 4036), $m::mul12(q12, 3349), $m::mul12(q11, 3461), $m::mul12(q4, 3996),
            $m::mul12(q7, 3822), $m::mul12(q8, 3745), $m::mul12(q15, 2967), $m::mul12(q0, 4095)
            => $t32, $t33a, $t34, $t35a, $t36, $t37a, $t38, $t39a,
               $t56a, $t57, $t58a, $t59, $t60a, $t61, $t62a, $t63
        );
    };
}

macro_rules! idct64_odd_dis_qtr_stage1_v {
    ($m:ident, $dis:ident
     => $t32:ident, $t33a:ident, $t34:ident, $t35a:ident, $t36:ident, $t37a:ident, $t38:ident, $t39a:ident,
        $t56a:ident, $t57:ident, $t58a:ident, $t59:ident, $t60a:ident, $t61:ident, $t62a:ident, $t63:ident) => {
        let (q0, q3, q4, q7) = ($dis[0], $dis[1], $dis[2], $dis[3]);
        let z = $m::zero();
        idct64_odd_dis_stage1_v!(
            $m,
            $m::mul12(q0, 101), z, z, $m::mul12(q7, -1474),
            $m::mul12(q4, 897), z, z, $m::mul12(q3, -700),
            $m::mul12(q3, 4036), z, z, $m::mul12(q4, 3996),
            $m::mul12(q7, 3822), z, z, $m::mul12(q0, 4095)
            => $t32, $t33a, $t34, $t35a, $t36, $t37a, $t38, $t39a,
               $t56a, $t57, $t58a, $t59, $t60a, $t61, $t62a, $t63
        );
    };
}

idct64_odd_ic_yari_fn!(gecis1_dct64_odd_ic_yari, p1);
idct64_odd_ic_yari_fn!(gecis2_dct64_odd_ic_yari, x16);

macro_rules! transpose_8x8_ymm_v {
    ($r0:ident, $r1:ident, $r2:ident, $r3:ident, $r4:ident, $r5:ident, $r6:ident, $r7:ident) => {{
        let t0 = _mm256_unpacklo_epi16($r0, $r1);
        let t1 = _mm256_unpackhi_epi16($r0, $r1);
        let t2 = _mm256_unpacklo_epi16($r2, $r3);
        let t3 = _mm256_unpackhi_epi16($r2, $r3);
        let t4 = _mm256_unpacklo_epi16($r4, $r5);
        let t5 = _mm256_unpackhi_epi16($r4, $r5);
        let t6 = _mm256_unpacklo_epi16($r6, $r7);
        let t7 = _mm256_unpackhi_epi16($r6, $r7);

        let s0 = _mm256_unpacklo_epi32(t0, t2);
        let s1 = _mm256_unpackhi_epi32(t0, t2);
        let s2 = _mm256_unpacklo_epi32(t1, t3);
        let s3 = _mm256_unpackhi_epi32(t1, t3);
        let s4 = _mm256_unpacklo_epi32(t4, t6);
        let s5 = _mm256_unpackhi_epi32(t4, t6);
        let s6 = _mm256_unpacklo_epi32(t5, t7);
        let s7 = _mm256_unpackhi_epi32(t5, t7);

        $r0 = _mm256_unpacklo_epi64(s0, s4);
        $r1 = _mm256_unpackhi_epi64(s0, s4);
        $r2 = _mm256_unpacklo_epi64(s1, s5);
        $r3 = _mm256_unpackhi_epi64(s1, s5);
        $r4 = _mm256_unpacklo_epi64(s2, s6);
        $r5 = _mm256_unpackhi_epi64(s2, s6);
        $r6 = _mm256_unpacklo_epi64(s3, s7);
        $r7 = _mm256_unpackhi_epi64(s3, s7);
    }};
}

macro_rules! transpose_8x8_split_ymm_v {
    ($r0:ident, $r1:ident, $r2:ident, $r3:ident, $r4:ident, $r5:ident, $r6:ident, $r7:ident) => {{
        let a0 = _mm256_unpacklo_epi16($r0, $r1);
        let a1 = _mm256_unpacklo_epi16($r2, $r3);
        let b0 = _mm256_unpacklo_epi32(a0, a1);
        let b1 = _mm256_unpackhi_epi32(a0, a1);
        let (t0, t1, t2, t3) = (
            b0,
            _mm256_srli_si256::<8>(b0),
            b1,
            _mm256_srli_si256::<8>(b1),
        );

        let c0 = _mm256_unpacklo_epi16($r4, $r5);
        let c1 = _mm256_unpacklo_epi16($r6, $r7);
        let d0 = _mm256_unpacklo_epi32(c0, c1);
        let d1 = _mm256_unpackhi_epi32(c0, c1);
        let (u0, u1, u2, u3) = (
            d0,
            _mm256_srli_si256::<8>(d0),
            d1,
            _mm256_srli_si256::<8>(d1),
        );

        $r0 = _mm256_permute2x128_si256::<0x20>(t0, u0);
        $r1 = _mm256_permute2x128_si256::<0x20>(t1, u1);
        $r2 = _mm256_permute2x128_si256::<0x20>(t2, u2);
        $r3 = _mm256_permute2x128_si256::<0x20>(t3, u3);
        $r4 = _mm256_permute2x128_si256::<0x31>(t0, u0);
        $r5 = _mm256_permute2x128_si256::<0x31>(t1, u1);
        $r6 = _mm256_permute2x128_si256::<0x31>(t2, u2);
        $r7 = _mm256_permute2x128_si256::<0x31>(t3, u3);
    }};
}

// ---------------------------------------------------------------------------
// Hedef `u16` düzlem piksel ekleme + `[0, 1023]` kırpma yardımcıları
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_8x1_xmm(dst: &mut [u8], off: usize, r: __m128i) {
    let d = load_u8x16(chunk::<u8, 16>(dst, off));
    let s = _mm_min_epi16(
        _mm_max_epi16(_mm_adds_epi16(d, r), _mm_setzero_si128()),
        _mm_set1_epi16(1023),
    );
    store_u8x16(chunk_mut(dst, off), s);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_8x1_split(dst: &mut [u8], off: usize, r: __m256i) {
    let d = load_u8x16(chunk::<u8, 16>(dst, off));
    let lo = _mm256_castsi256_si128(r);
    let hi = _mm256_extracti128_si256::<1>(r);
    let del = _mm_unpacklo_epi64(lo, hi);
    let s = _mm_min_epi16(
        _mm_max_epi16(_mm_adds_epi16(d, del), _mm_setzero_si128()),
        _mm_set1_epi16(1023),
    );
    store_u8x16(chunk_mut(dst, off), s);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_16x1_ymm(dst: &mut [u8], off: usize, r: __m256i) {
    let d = load_u8x32(chunk::<u8, 32>(dst, off));
    let s = _mm256_min_epi16(
        _mm256_max_epi16(_mm256_adds_epi16(d, r), _mm256_setzero_si256()),
        _mm256_set1_epi16(1023),
    );
    store_u8x32(chunk_mut(dst, off), s);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_32x1_zmm(dst: &mut [u8], off: usize, r: __m512i) {
    let d = load_u8x64(chunk::<u8, 64>(dst, off));
    let s = _mm512_min_epi16(
        _mm512_max_epi16(_mm512_adds_epi16(d, r), _mm512_setzero_si512()),
        _mm512_set1_epi16(1023),
    );
    store_u8x64(chunk_mut(dst, off), s);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_w8_split_kelebek32(dst: &mut [u8], stride: usize, k: usize, ek: __m256i, ok: __m256i) {
    let r4 = _mm256_set1_epi16(2048);
    ekle_yaz_8x1_split(
        dst,
        k * stride,
        _mm256_mulhrs_epi16(_mm256_adds_epi16(ek, ok), r4),
    );
    ekle_yaz_8x1_split(
        dst,
        (31 - k) * stride,
        _mm256_mulhrs_epi16(_mm256_subs_epi16(ek, ok), r4),
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_w16_kelebek32(
    dst: &mut [u8],
    stride: usize,
    x_bayt: usize,
    k: usize,
    ek: __m256i,
    ok: __m256i,
) {
    let r4 = _mm256_set1_epi16(2048);
    ekle_yaz_16x1_ymm(
        dst,
        k * stride + x_bayt,
        _mm256_mulhrs_epi16(_mm256_adds_epi16(ek, ok), r4),
    );
    ekle_yaz_16x1_ymm(
        dst,
        (31 - k) * stride + x_bayt,
        _mm256_mulhrs_epi16(_mm256_subs_epi16(ek, ok), r4),
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_w16_kelebek64(
    dst: &mut [u8],
    stride: usize,
    x_bayt: usize,
    k: usize,
    ek: __m256i,
    ok: __m256i,
) {
    let r4 = _mm256_set1_epi16(2048);
    ekle_yaz_16x1_ymm(
        dst,
        k * stride + x_bayt,
        _mm256_mulhrs_epi16(_mm256_adds_epi16(ek, ok), r4),
    );
    ekle_yaz_16x1_ymm(
        dst,
        (63 - k) * stride + x_bayt,
        _mm256_mulhrs_epi16(_mm256_subs_epi16(ek, ok), r4),
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
    let dc16 = dc.clamp(-16_384, 16_384) as i16;

    if W == 8 {
        let dc_v = _mm_set1_epi16(dc16);
        let mut y = 0;
        while y < H {
            ekle_yaz_8x1_xmm(dst, y * stride, dc_v);
            y += 1;
        }
    } else if W == 16 {
        let dc_v = _mm256_set1_epi16(dc16);
        let mut y = 0;
        while y < H {
            ekle_yaz_16x1_ymm(dst, y * stride, dc_v);
            y += 1;
        }
    } else {
        let dc_v = _mm512_set1_epi16(dc16);
        let mut y = 0;
        while y < H {
            let taban = y * stride;
            let mut x = 0;
            while x < W {
                ekle_yaz_32x1_zmm(dst, taban + x * 2, dc_v);
                x += 32;
            }
            y += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 1 şerit (`yb`, 8 satır) dönüştürücüleri (`coeff` içinde yerinde aşamalama)
// ---------------------------------------------------------------------------

macro_rules! sakla_serit8 {
    ($coeff:expr, $y0:expr, $sh:expr, $g_off:expr,
     $r0:expr, $r1:expr, $r2:expr, $r3:expr, $r4:expr, $r5:expr, $r6:expr, $r7:expr) => {{
        let y0_s = $y0;
        let sh_s = $sh;
        let go = $g_off;
        sakla_i32x8($coeff, y0_s + (go) * sh_s, $r0);
        sakla_i32x8($coeff, y0_s + (go + 1) * sh_s, $r1);
        sakla_i32x8($coeff, y0_s + (go + 2) * sh_s, $r2);
        sakla_i32x8($coeff, y0_s + (go + 3) * sh_s, $r3);
        sakla_i32x8($coeff, y0_s + (go + 4) * sh_s, $r4);
        sakla_i32x8($coeff, y0_s + (go + 5) * sh_s, $r5);
        sakla_i32x8($coeff, y0_s + (go + 6) * sh_s, $r6);
        sakla_i32x8($coeff, y0_s + (go + 7) * sh_s, $r7);
    }};
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn gecis1_dct8_serit(coeff: &mut [i32], y0: usize) {
    let mut c0 = yukle_i32x8(coeff, y0);
    let mut c1 = yukle_i32x8(coeff, y0 + 32);
    let mut c2 = yukle_i32x8(coeff, y0 + 2 * 32);
    let mut c3 = yukle_i32x8(coeff, y0 + 3 * 32);
    let mut c4 = yukle_i32x8(coeff, y0 + 4 * 32);
    let mut c5 = yukle_i32x8(coeff, y0 + 5 * 32);
    let mut c6 = yukle_i32x8(coeff, y0 + 6 * 32);
    let mut c7 = yukle_i32x8(coeff, y0 + 7 * 32);
    idct8_v!(p1, c0, c1, c2, c3, c4, c5, c6, c7);
    c0 = p1::paket_split::<2>(c0);
    c1 = p1::paket_split::<2>(c1);
    c2 = p1::paket_split::<2>(c2);
    c3 = p1::paket_split::<2>(c3);
    c4 = p1::paket_split::<2>(c4);
    c5 = p1::paket_split::<2>(c5);
    c6 = p1::paket_split::<2>(c6);
    c7 = p1::paket_split::<2>(c7);
    transpose_8x8_split_ymm_v!(c0, c1, c2, c3, c4, c5, c6, c7);
    sakla_serit8!(coeff, y0, 32, 0, c0, c1, c2, c3, c4, c5, c6, c7);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn gecis1_dct16_serit<const SHIFT: u32, const RECT2: bool>(
    coeff: &mut [i32],
    y0: usize,
    sh: usize,
    yarim: bool,
) {
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) = if yarim {
        let i0 = p1::yukle_opt::<RECT2>(coeff, y0);
        let i1 = p1::yukle_opt::<RECT2>(coeff, y0 + sh);
        let i2 = p1::yukle_opt::<RECT2>(coeff, y0 + 2 * sh);
        let i3 = p1::yukle_opt::<RECT2>(coeff, y0 + 3 * sh);
        let i4 = p1::yukle_opt::<RECT2>(coeff, y0 + 4 * sh);
        let i5 = p1::yukle_opt::<RECT2>(coeff, y0 + 5 * sh);
        let i6 = p1::yukle_opt::<RECT2>(coeff, y0 + 6 * sh);
        let i7 = p1::yukle_opt::<RECT2>(coeff, y0 + 7 * sh);
        idct16_half_v!(
            p1, i0, i1, i2, i3, i4, i5, i6, i7
            => c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15
        );
        (
            p1::paket_2col::<SHIFT>(c0, c8),
            p1::paket_2col::<SHIFT>(c1, c9),
            p1::paket_2col::<SHIFT>(c2, c10),
            p1::paket_2col::<SHIFT>(c3, c11),
            p1::paket_2col::<SHIFT>(c4, c12),
            p1::paket_2col::<SHIFT>(c5, c13),
            p1::paket_2col::<SHIFT>(c6, c14),
            p1::paket_2col::<SHIFT>(c7, c15),
        )
    } else {
        let mut c0 = p1::yukle_opt::<RECT2>(coeff, y0);
        let mut c1 = p1::yukle_opt::<RECT2>(coeff, y0 + sh);
        let mut c2 = p1::yukle_opt::<RECT2>(coeff, y0 + 2 * sh);
        let mut c3 = p1::yukle_opt::<RECT2>(coeff, y0 + 3 * sh);
        let mut c4 = p1::yukle_opt::<RECT2>(coeff, y0 + 4 * sh);
        let mut c5 = p1::yukle_opt::<RECT2>(coeff, y0 + 5 * sh);
        let mut c6 = p1::yukle_opt::<RECT2>(coeff, y0 + 6 * sh);
        let mut c7 = p1::yukle_opt::<RECT2>(coeff, y0 + 7 * sh);
        let mut c8 = p1::yukle_opt::<RECT2>(coeff, y0 + 8 * sh);
        let mut c9 = p1::yukle_opt::<RECT2>(coeff, y0 + 9 * sh);
        let mut c10 = p1::yukle_opt::<RECT2>(coeff, y0 + 10 * sh);
        let mut c11 = p1::yukle_opt::<RECT2>(coeff, y0 + 11 * sh);
        let mut c12 = p1::yukle_opt::<RECT2>(coeff, y0 + 12 * sh);
        let mut c13 = p1::yukle_opt::<RECT2>(coeff, y0 + 13 * sh);
        let mut c14 = p1::yukle_opt::<RECT2>(coeff, y0 + 14 * sh);
        let mut c15 = p1::yukle_opt::<RECT2>(coeff, y0 + 15 * sh);
        idct16_v!(
            p1, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15
        );
        (
            p1::paket_2col::<SHIFT>(c0, c8),
            p1::paket_2col::<SHIFT>(c1, c9),
            p1::paket_2col::<SHIFT>(c2, c10),
            p1::paket_2col::<SHIFT>(c3, c11),
            p1::paket_2col::<SHIFT>(c4, c12),
            p1::paket_2col::<SHIFT>(c5, c13),
            p1::paket_2col::<SHIFT>(c6, c14),
            p1::paket_2col::<SHIFT>(c7, c15),
        )
    };
    transpose_8x8_ymm_v!(p0, p1, p2, p3, p4, p5, p6, p7);
    sakla_serit8!(coeff, y0, sh, 0, p0, p1, p2, p3, p4, p5, p6, p7);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn gecis1_dct32_even<const RECT2: bool>(
    coeff: &mut [i32],
    y0: usize,
    sh: usize,
    ceyrek: bool,
    yarim: bool,
) {
    let (e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15) = if ceyrek {
        let i0 = p1::yukle_opt::<RECT2>(coeff, y0);
        let i2 = p1::yukle_opt::<RECT2>(coeff, y0 + 2 * sh);
        let i4 = p1::yukle_opt::<RECT2>(coeff, y0 + 4 * sh);
        let i6 = p1::yukle_opt::<RECT2>(coeff, y0 + 6 * sh);
        idct16_qtr_v!(
            p1, i0, i2, i4, i6
            => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
        );
        (
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        )
    } else if yarim {
        let i0 = p1::yukle_opt::<RECT2>(coeff, y0);
        let i2 = p1::yukle_opt::<RECT2>(coeff, y0 + 2 * sh);
        let i4 = p1::yukle_opt::<RECT2>(coeff, y0 + 4 * sh);
        let i6 = p1::yukle_opt::<RECT2>(coeff, y0 + 6 * sh);
        let i8 = p1::yukle_opt::<RECT2>(coeff, y0 + 8 * sh);
        let i10 = p1::yukle_opt::<RECT2>(coeff, y0 + 10 * sh);
        let i12 = p1::yukle_opt::<RECT2>(coeff, y0 + 12 * sh);
        let i14 = p1::yukle_opt::<RECT2>(coeff, y0 + 14 * sh);
        idct16_half_v!(
            p1, i0, i2, i4, i6, i8, i10, i12, i14
            => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
        );
        (
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        )
    } else {
        let mut c0 = p1::yukle_opt::<RECT2>(coeff, y0);
        let mut c1 = p1::yukle_opt::<RECT2>(coeff, y0 + 2 * sh);
        let mut c2 = p1::yukle_opt::<RECT2>(coeff, y0 + 4 * sh);
        let mut c3 = p1::yukle_opt::<RECT2>(coeff, y0 + 6 * sh);
        let mut c4 = p1::yukle_opt::<RECT2>(coeff, y0 + 8 * sh);
        let mut c5 = p1::yukle_opt::<RECT2>(coeff, y0 + 10 * sh);
        let mut c6 = p1::yukle_opt::<RECT2>(coeff, y0 + 12 * sh);
        let mut c7 = p1::yukle_opt::<RECT2>(coeff, y0 + 14 * sh);
        let mut c8 = p1::yukle_opt::<RECT2>(coeff, y0 + 16 * sh);
        let mut c9 = p1::yukle_opt::<RECT2>(coeff, y0 + 18 * sh);
        let mut c10 = p1::yukle_opt::<RECT2>(coeff, y0 + 20 * sh);
        let mut c11 = p1::yukle_opt::<RECT2>(coeff, y0 + 22 * sh);
        let mut c12 = p1::yukle_opt::<RECT2>(coeff, y0 + 24 * sh);
        let mut c13 = p1::yukle_opt::<RECT2>(coeff, y0 + 26 * sh);
        let mut c14 = p1::yukle_opt::<RECT2>(coeff, y0 + 28 * sh);
        let mut c15 = p1::yukle_opt::<RECT2>(coeff, y0 + 30 * sh);
        idct16_v!(
            p1, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15
        );
        (
            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15,
        )
    };
    sakla_i32x8(coeff, y0, e0);
    sakla_i32x8(coeff, y0 + 2 * sh, e1);
    sakla_i32x8(coeff, y0 + 4 * sh, e2);
    sakla_i32x8(coeff, y0 + 6 * sh, e3);
    sakla_i32x8(coeff, y0 + 8 * sh, e4);
    sakla_i32x8(coeff, y0 + 10 * sh, e5);
    sakla_i32x8(coeff, y0 + 12 * sh, e6);
    sakla_i32x8(coeff, y0 + 14 * sh, e7);
    sakla_i32x8(coeff, y0 + 16 * sh, e8);
    sakla_i32x8(coeff, y0 + 18 * sh, e9);
    sakla_i32x8(coeff, y0 + 20 * sh, e10);
    sakla_i32x8(coeff, y0 + 22 * sh, e11);
    sakla_i32x8(coeff, y0 + 24 * sh, e12);
    sakla_i32x8(coeff, y0 + 26 * sh, e13);
    sakla_i32x8(coeff, y0 + 28 * sh, e14);
    sakla_i32x8(coeff, y0 + 30 * sh, e15);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn gecis1_dct32_serit<const SHIFT: u32, const RECT2: bool>(
    coeff: &mut [i32],
    y0: usize,
    sh: usize,
    ceyrek: bool,
    yarim: bool,
) {
    gecis1_dct32_even::<RECT2>(coeff, y0, sh, ceyrek, yarim);
    let (o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15) = if ceyrek {
        let i1 = p1::yukle_opt::<RECT2>(coeff, y0 + sh);
        let i3 = p1::yukle_opt::<RECT2>(coeff, y0 + 3 * sh);
        let i5 = p1::yukle_opt::<RECT2>(coeff, y0 + 5 * sh);
        let i7 = p1::yukle_opt::<RECT2>(coeff, y0 + 7 * sh);
        idct32_qtr_odd_v!(
            p1, i1, i3, i5, i7
            => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
        );
        (
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        )
    } else if yarim {
        let i1 = p1::yukle_opt::<RECT2>(coeff, y0 + sh);
        let i3 = p1::yukle_opt::<RECT2>(coeff, y0 + 3 * sh);
        let i5 = p1::yukle_opt::<RECT2>(coeff, y0 + 5 * sh);
        let i7 = p1::yukle_opt::<RECT2>(coeff, y0 + 7 * sh);
        let i9 = p1::yukle_opt::<RECT2>(coeff, y0 + 9 * sh);
        let i11 = p1::yukle_opt::<RECT2>(coeff, y0 + 11 * sh);
        let i13 = p1::yukle_opt::<RECT2>(coeff, y0 + 13 * sh);
        let i15 = p1::yukle_opt::<RECT2>(coeff, y0 + 15 * sh);
        idct32_half_odd_v!(
            p1, i1, i3, i5, i7, i9, i11, i13, i15
            => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
        );
        (
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        )
    } else {
        let t1 = p1::yukle_opt::<RECT2>(coeff, y0 + sh);
        let t3 = p1::yukle_opt::<RECT2>(coeff, y0 + 3 * sh);
        let t5 = p1::yukle_opt::<RECT2>(coeff, y0 + 5 * sh);
        let t7 = p1::yukle_opt::<RECT2>(coeff, y0 + 7 * sh);
        let t9 = p1::yukle_opt::<RECT2>(coeff, y0 + 9 * sh);
        let t11 = p1::yukle_opt::<RECT2>(coeff, y0 + 11 * sh);
        let t13 = p1::yukle_opt::<RECT2>(coeff, y0 + 13 * sh);
        let t15 = p1::yukle_opt::<RECT2>(coeff, y0 + 15 * sh);
        let b1 = p1::yukle_opt::<RECT2>(coeff, y0 + 17 * sh);
        let b3 = p1::yukle_opt::<RECT2>(coeff, y0 + 19 * sh);
        let b5 = p1::yukle_opt::<RECT2>(coeff, y0 + 21 * sh);
        let b7 = p1::yukle_opt::<RECT2>(coeff, y0 + 23 * sh);
        let b9 = p1::yukle_opt::<RECT2>(coeff, y0 + 25 * sh);
        let b11 = p1::yukle_opt::<RECT2>(coeff, y0 + 27 * sh);
        let b13 = p1::yukle_opt::<RECT2>(coeff, y0 + 29 * sh);
        let b15 = p1::yukle_opt::<RECT2>(coeff, y0 + 31 * sh);
        idct32_full_odd_v!(
            p1, t1, t3, t5, t7, t9, t11, t13, t15, b1, b3, b5, b7, b9, b11, b13, b15
            => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
        );
        (
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        )
    };

    macro_rules! cift_paket {
        ($k:literal, $ok:ident, $ok8:ident => $pk:ident, $qm:ident) => {
            let ek = yukle_i32x8(coeff, y0 + (2 * $k) * sh);
            let ek8 = yukle_i32x8(coeff, y0 + (2 * ($k + 8)) * sh);
            let mut $pk = p1::paket_2col::<SHIFT>(p1::adds(ek, $ok), p1::adds(ek8, $ok8));
            let mut $qm = p1::paket_2col::<SHIFT>(p1::subs(ek8, $ok8), p1::subs(ek, $ok));
        };
    }
    cift_paket!(0, o0, o8 => p0, q7);
    cift_paket!(1, o1, o9 => p1, q6);
    cift_paket!(2, o2, o10 => p2, q5);
    cift_paket!(3, o3, o11 => p3, q4);
    cift_paket!(4, o4, o12 => p4, q3);
    cift_paket!(5, o5, o13 => p5, q2);
    cift_paket!(6, o6, o14 => p6, q1);
    cift_paket!(7, o7, o15 => p7, q0);

    transpose_8x8_ymm_v!(p0, p1, p2, p3, p4, p5, p6, p7);
    sakla_serit8!(coeff, y0, sh, 0, p0, p1, p2, p3, p4, p5, p6, p7);
    transpose_8x8_ymm_v!(q0, q1, q2, q3, q4, q5, q6, q7);
    sakla_serit8!(coeff, y0, sh, 8, q0, q1, q2, q3, q4, q5, q6, q7);
}

#[inline(always)]
const fn slot_e64(k: usize) -> usize {
    if (k & 2) == 0 { k } else { 31 - k }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn gecis1_dct64_even_ust<const RECT2: bool>(
    coeff: &mut [i32],
    y0: usize,
    sh: usize,
    yarim: bool,
    tek_sakla: &mut [__m256i; 16],
) {
    let mut k = 0;
    while k < 16 {
        tek_sakla[k] = p1::yukle_opt::<RECT2>(coeff, y0 + (2 * k + 1) * sh);
        k += 1;
    }

    let (e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15) = if yarim {
        let i0 = p1::yukle_opt::<RECT2>(coeff, y0);
        let i4 = p1::yukle_opt::<RECT2>(coeff, y0 + 4 * sh);
        let i8 = p1::yukle_opt::<RECT2>(coeff, y0 + 8 * sh);
        let i12 = p1::yukle_opt::<RECT2>(coeff, y0 + 12 * sh);
        idct16_qtr_v!(
            p1, i0, i4, i8, i12
            => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
        );
        (
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        )
    } else {
        let i0 = p1::yukle_opt::<RECT2>(coeff, y0);
        let i4 = p1::yukle_opt::<RECT2>(coeff, y0 + 4 * sh);
        let i8 = p1::yukle_opt::<RECT2>(coeff, y0 + 8 * sh);
        let i12 = p1::yukle_opt::<RECT2>(coeff, y0 + 12 * sh);
        let i16 = p1::yukle_opt::<RECT2>(coeff, y0 + 16 * sh);
        let i20 = p1::yukle_opt::<RECT2>(coeff, y0 + 20 * sh);
        let i24 = p1::yukle_opt::<RECT2>(coeff, y0 + 24 * sh);
        let i28 = p1::yukle_opt::<RECT2>(coeff, y0 + 28 * sh);
        idct16_half_v!(
            p1, i0, i4, i8, i12, i16, i20, i24, i28
            => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
        );
        (
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        )
    };

    sakla_i32x8(coeff, y0 + slot_e64(0) * sh, e0);
    sakla_i32x8(coeff, y0 + slot_e64(1) * sh, e1);
    sakla_i32x8(coeff, y0 + slot_e64(2) * sh, e2);
    sakla_i32x8(coeff, y0 + slot_e64(3) * sh, e3);
    sakla_i32x8(coeff, y0 + slot_e64(4) * sh, e4);
    sakla_i32x8(coeff, y0 + slot_e64(5) * sh, e5);
    sakla_i32x8(coeff, y0 + slot_e64(6) * sh, e6);
    sakla_i32x8(coeff, y0 + slot_e64(7) * sh, e7);
    sakla_i32x8(coeff, y0 + slot_e64(8) * sh, e8);
    sakla_i32x8(coeff, y0 + slot_e64(9) * sh, e9);
    sakla_i32x8(coeff, y0 + slot_e64(10) * sh, e10);
    sakla_i32x8(coeff, y0 + slot_e64(11) * sh, e11);
    sakla_i32x8(coeff, y0 + slot_e64(12) * sh, e12);
    sakla_i32x8(coeff, y0 + slot_e64(13) * sh, e13);
    sakla_i32x8(coeff, y0 + slot_e64(14) * sh, e14);
    sakla_i32x8(coeff, y0 + slot_e64(15) * sh, e15);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn gecis1_dct64_even_alt<const RECT2: bool>(coeff: &mut [i32], y0: usize, sh: usize, yarim: bool) {
    let (o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15) = if yarim {
        let i2 = p1::yukle_opt::<RECT2>(coeff, y0 + 2 * sh);
        let i6 = p1::yukle_opt::<RECT2>(coeff, y0 + 6 * sh);
        let i10 = p1::yukle_opt::<RECT2>(coeff, y0 + 10 * sh);
        let i14 = p1::yukle_opt::<RECT2>(coeff, y0 + 14 * sh);
        idct32_qtr_odd_v!(
            p1, i2, i6, i10, i14
            => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
        );
        (
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        )
    } else {
        let i2 = p1::yukle_opt::<RECT2>(coeff, y0 + 2 * sh);
        let i6 = p1::yukle_opt::<RECT2>(coeff, y0 + 6 * sh);
        let i10 = p1::yukle_opt::<RECT2>(coeff, y0 + 10 * sh);
        let i14 = p1::yukle_opt::<RECT2>(coeff, y0 + 14 * sh);
        let i18 = p1::yukle_opt::<RECT2>(coeff, y0 + 18 * sh);
        let i22 = p1::yukle_opt::<RECT2>(coeff, y0 + 22 * sh);
        let i26 = p1::yukle_opt::<RECT2>(coeff, y0 + 26 * sh);
        let i30 = p1::yukle_opt::<RECT2>(coeff, y0 + 30 * sh);
        idct32_half_odd_v!(
            p1, i2, i6, i10, i14, i18, i22, i26, i30
            => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
        );
        (
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        )
    };

    macro_rules! sakla_e32 {
        ($idx:literal, $ok:ident) => {{
            let ek = yukle_i32x8(coeff, y0 + slot_e64($idx) * sh);
            sakla_i32x8(coeff, y0 + ($idx) * sh, p1::adds(ek, $ok));
            sakla_i32x8(coeff, y0 + (31 - $idx) * sh, p1::subs(ek, $ok));
        }};
    }
    sakla_e32!(0, o0);
    sakla_e32!(1, o1);
    sakla_e32!(2, o2);
    sakla_e32!(3, o3);
    sakla_e32!(4, o4);
    sakla_e32!(5, o5);
    sakla_e32!(6, o6);
    sakla_e32!(7, o7);
    sakla_e32!(8, o8);
    sakla_e32!(9, o9);
    sakla_e32!(10, o10);
    sakla_e32!(11, o11);
    sakla_e32!(12, o12);
    sakla_e32!(13, o13);
    sakla_e32!(14, o14);
    sakla_e32!(15, o15);
}

macro_rules! p1_dis_ciftler_sakla {
    ($SHIFT:ident, $coeff:ident, $y0:ident, $sh:ident, $tek:ident,
     $t32:ident, $t33a:ident, $t34:ident, $t35a:ident, $t36:ident, $t37a:ident, $t38:ident, $t39a:ident,
     $t56a:ident, $t57:ident, $t58a:ident, $t59:ident, $t60a:ident, $t61:ident, $t62a:ident, $t63:ident) => {{
        macro_rules! cift_k {
            ($k:literal, $m:literal, $tk:ident, $bk:ident, $tm:ident, $bm:ident) => {{
                let ik_top = $tek[$m];
                let ik_bot = $tek[8 + $k];
                let im_top = $tek[$k];
                let im_bot = $tek[15 - $k];

                let hi_15_k = p1::adds($tk, ik_top);
                let lo_k = p1::adds($bk, ik_bot);
                let (hi_k, lo_15_k) = p1::rot_r2(p1::subs($bk, ik_bot), p1::subs($tk, ik_top));

                let hi_8_k = p1::adds($tm, im_top);
                let lo_m = p1::adds($bm, im_bot);
                let (hi_m, lo_8_k) = p1::rot_r2(p1::subs($bm, im_bot), p1::subs($tm, im_top));

                $tek[$k] = hi_k;
                $tek[$m] = hi_m;
                $tek[8 + $k] = hi_8_k;
                $tek[15 - $k] = hi_15_k;

                let ek = yukle_i32x8($coeff, $y0 + ($k) * $sh);
                let ek8 = yukle_i32x8($coeff, $y0 + ($k + 8) * $sh);
                let pk = p1::paket_2col::<$SHIFT>(p1::adds(ek, lo_k), p1::adds(ek8, lo_8_k));
                let wm = p1::paket_2col::<$SHIFT>(p1::subs(ek8, lo_8_k), p1::subs(ek, lo_k));
                sakla_i32x8($coeff, $y0 + ($k) * $sh, pk);
                sakla_i32x8($coeff, $y0 + ($k + 8) * $sh, wm);

                let em = yukle_i32x8($coeff, $y0 + ($m) * $sh);
                let em8 = yukle_i32x8($coeff, $y0 + ($m + 8) * $sh);
                let pm = p1::paket_2col::<$SHIFT>(p1::adds(em, lo_m), p1::adds(em8, lo_15_k));
                let wmm = p1::paket_2col::<$SHIFT>(p1::subs(em8, lo_15_k), p1::subs(em, lo_m));
                sakla_i32x8($coeff, $y0 + ($m) * $sh, pm);
                sakla_i32x8($coeff, $y0 + ($m + 8) * $sh, wmm);
            }};
        }
        cift_k!(0, 7, $t32, $t63, $t39a, $t56a);
        cift_k!(1, 6, $t33a, $t62a, $t38, $t57);
        cift_k!(2, 5, $t34, $t61, $t37a, $t58a);
        cift_k!(3, 4, $t35a, $t60a, $t36, $t59);
    }};
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn gecis1_dct64_odd_qtr_dis<const SHIFT: u32>(
    coeff: &mut [i32],
    y0: usize,
    sh: usize,
    dis_girdi: &[__m256i; 8],
    tek: &mut [__m256i; 16],
) {
    idct64_odd_dis_qtr_stage1_v!(
        p1, dis_girdi
        => t32, t33a, t34, t35a, t36, t37a, t38, t39a,
           t56a, t57, t58a, t59, t60a, t61, t62a, t63
    );
    p1_dis_ciftler_sakla!(
        SHIFT, coeff, y0, sh, tek, t32, t33a, t34, t35a, t36, t37a, t38, t39a, t56a, t57, t58a,
        t59, t60a, t61, t62a, t63
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn gecis1_dct64_odd_full_dis<const SHIFT: u32>(
    coeff: &mut [i32],
    y0: usize,
    sh: usize,
    dis_girdi: &[__m256i; 8],
    tek: &mut [__m256i; 16],
) {
    idct64_odd_dis_full_stage1_v!(
        p1, dis_girdi
        => t32, t33a, t34, t35a, t36, t37a, t38, t39a,
           t56a, t57, t58a, t59, t60a, t61, t62a, t63
    );
    p1_dis_ciftler_sakla!(
        SHIFT, coeff, y0, sh, tek, t32, t33a, t34, t35a, t36, t37a, t38, t39a, t56a, t57, t58a,
        t59, t60a, t61, t62a, t63
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn gecis1_dct64_dis_ve_ic_devrik<const SHIFT: u32>(
    coeff: &mut [i32],
    y0: usize,
    sh: usize,
    hi: &[__m256i; 16],
) {
    let mut g0_0 = yukle_i32x8(coeff, y0);
    let mut g0_1 = yukle_i32x8(coeff, y0 + sh);
    let mut g0_2 = yukle_i32x8(coeff, y0 + 2 * sh);
    let mut g0_3 = yukle_i32x8(coeff, y0 + 3 * sh);
    let mut g0_4 = yukle_i32x8(coeff, y0 + 4 * sh);
    let mut g0_5 = yukle_i32x8(coeff, y0 + 5 * sh);
    let mut g0_6 = yukle_i32x8(coeff, y0 + 6 * sh);
    let mut g0_7 = yukle_i32x8(coeff, y0 + 7 * sh);
    transpose_8x8_ymm_v!(g0_0, g0_1, g0_2, g0_3, g0_4, g0_5, g0_6, g0_7);
    sakla_serit8!(
        coeff, y0, sh, 0, g0_0, g0_1, g0_2, g0_3, g0_4, g0_5, g0_6, g0_7
    );

    macro_rules! ic_paket {
        ($k:literal, $qk:ident, $rm:ident) => {
            let ek = yukle_i32x8(coeff, y0 + (16 + $k) * sh);
            let ek8 = yukle_i32x8(coeff, y0 + (24 + $k) * sh);
            let mut $qk = p1::paket_2col::<SHIFT>(p1::adds(ek, hi[$k]), p1::adds(ek8, hi[$k + 8]));
            let mut $rm = p1::paket_2col::<SHIFT>(p1::subs(ek8, hi[$k + 8]), p1::subs(ek, hi[$k]));
        };
    }
    ic_paket!(0, g1_0, g2_7);
    ic_paket!(1, g1_1, g2_6);
    ic_paket!(2, g1_2, g2_5);
    ic_paket!(3, g1_3, g2_4);
    ic_paket!(4, g1_4, g2_3);
    ic_paket!(5, g1_5, g2_2);
    ic_paket!(6, g1_6, g2_1);
    ic_paket!(7, g1_7, g2_0);

    transpose_8x8_ymm_v!(g2_0, g2_1, g2_2, g2_3, g2_4, g2_5, g2_6, g2_7);
    sakla_serit8!(
        coeff, y0, sh, 16, g2_0, g2_1, g2_2, g2_3, g2_4, g2_5, g2_6, g2_7
    );

    let mut g3_7 = yukle_i32x8(coeff, y0 + 8 * sh);
    let mut g3_6 = yukle_i32x8(coeff, y0 + 9 * sh);
    let mut g3_5 = yukle_i32x8(coeff, y0 + 10 * sh);
    let mut g3_4 = yukle_i32x8(coeff, y0 + 11 * sh);
    let mut g3_3 = yukle_i32x8(coeff, y0 + 12 * sh);
    let mut g3_2 = yukle_i32x8(coeff, y0 + 13 * sh);
    let mut g3_1 = yukle_i32x8(coeff, y0 + 14 * sh);
    let mut g3_0 = yukle_i32x8(coeff, y0 + 15 * sh);
    transpose_8x8_ymm_v!(g3_0, g3_1, g3_2, g3_3, g3_4, g3_5, g3_6, g3_7);
    sakla_serit8!(
        coeff, y0, sh, 24, g3_0, g3_1, g3_2, g3_3, g3_4, g3_5, g3_6, g3_7
    );

    transpose_8x8_ymm_v!(g1_0, g1_1, g1_2, g1_3, g1_4, g1_5, g1_6, g1_7);
    sakla_serit8!(
        coeff, y0, sh, 8, g1_0, g1_1, g1_2, g1_3, g1_4, g1_5, g1_6, g1_7
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn gecis1_dct64_serit<const SHIFT: u32, const RECT2: bool>(
    coeff: &mut [i32],
    y0: usize,
    sh: usize,
    yarim: bool,
) {
    let mut tek = [_mm256_setzero_si256(); 16];
    let mut dis_girdi = [_mm256_setzero_si256(); 8];
    gecis1_dct64_even_ust::<RECT2>(coeff, y0, sh, yarim, &mut tek);
    gecis1_dct64_even_alt::<RECT2>(coeff, y0, sh, yarim);
    gecis1_dct64_odd_ic_yari(yarim, &mut tek, &mut dis_girdi);
    if yarim {
        gecis1_dct64_odd_qtr_dis::<SHIFT>(coeff, y0, sh, &dis_girdi, &mut tek);
    } else {
        gecis1_dct64_odd_full_dis::<SHIFT>(coeff, y0, sh, &dis_girdi, &mut tek);
    }
    gecis1_dct64_dis_ve_ic_devrik::<SHIFT>(coeff, y0, sh, &tek);
}

// ---------------------------------------------------------------------------
// Pass 2 sütun dönüştürücüleri (`h ∈ {8, 16, 32, 64}`, `coeff` üzerinden)
// ---------------------------------------------------------------------------

#[inline(always)]
const fn satir_ofseti(g_off: usize, y: usize, sh: usize) -> usize {
    (y / 8) * 8 + (g_off + (y % 8)) * sh
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_z32_satir(coeff: &[i32], g_off: usize, y: usize, sh: usize) -> __m512i {
    let lo = yukle_i32x8(coeff, satir_ofseti(g_off, y, sh));
    let hi = yukle_i32x8(coeff, satir_ofseti(g_off + 8, y, sh));
    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn gecis2_dct8_w32(dst: &mut [u8], stride: usize, coeff: &[i32], g_off: usize, x_bayt: usize) {
    let mut c0 = yukle_z32_satir(coeff, g_off, 0, 8);
    let mut c1 = yukle_z32_satir(coeff, g_off, 1, 8);
    let mut c2 = yukle_z32_satir(coeff, g_off, 2, 8);
    let mut c3 = yukle_z32_satir(coeff, g_off, 3, 8);
    let mut c4 = yukle_z32_satir(coeff, g_off, 4, 8);
    let mut c5 = yukle_z32_satir(coeff, g_off, 5, 8);
    let mut c6 = yukle_z32_satir(coeff, g_off, 6, 8);
    let mut c7 = yukle_z32_satir(coeff, g_off, 7, 8);
    idct8_v!(z32, c0, c1, c2, c3, c4, c5, c6, c7);
    ekle_yaz_32x1_zmm(dst, x_bayt, z32::round_p2(c0));
    ekle_yaz_32x1_zmm(dst, stride + x_bayt, z32::round_p2(c1));
    ekle_yaz_32x1_zmm(dst, 2 * stride + x_bayt, z32::round_p2(c2));
    ekle_yaz_32x1_zmm(dst, 3 * stride + x_bayt, z32::round_p2(c3));
    ekle_yaz_32x1_zmm(dst, 4 * stride + x_bayt, z32::round_p2(c4));
    ekle_yaz_32x1_zmm(dst, 5 * stride + x_bayt, z32::round_p2(c5));
    ekle_yaz_32x1_zmm(dst, 6 * stride + x_bayt, z32::round_p2(c6));
    ekle_yaz_32x1_zmm(dst, 7 * stride + x_bayt, z32::round_p2(c7));
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn gecis2_dct16_w32(
    dst: &mut [u8],
    stride: usize,
    coeff: &[i32],
    g_off: usize,
    x_bayt: usize,
    yarim: bool,
) {
    let (c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15) = if yarim {
        let i0 = yukle_z32_satir(coeff, g_off, 0, 16);
        let i1 = yukle_z32_satir(coeff, g_off, 1, 16);
        let i2 = yukle_z32_satir(coeff, g_off, 2, 16);
        let i3 = yukle_z32_satir(coeff, g_off, 3, 16);
        let i4 = yukle_z32_satir(coeff, g_off, 4, 16);
        let i5 = yukle_z32_satir(coeff, g_off, 5, 16);
        let i6 = yukle_z32_satir(coeff, g_off, 6, 16);
        let i7 = yukle_z32_satir(coeff, g_off, 7, 16);
        idct16_half_v!(
            z32, i0, i1, i2, i3, i4, i5, i6, i7
            => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
        );
        (
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        )
    } else {
        let mut r0 = yukle_z32_satir(coeff, g_off, 0, 16);
        let mut r1 = yukle_z32_satir(coeff, g_off, 1, 16);
        let mut r2 = yukle_z32_satir(coeff, g_off, 2, 16);
        let mut r3 = yukle_z32_satir(coeff, g_off, 3, 16);
        let mut r4 = yukle_z32_satir(coeff, g_off, 4, 16);
        let mut r5 = yukle_z32_satir(coeff, g_off, 5, 16);
        let mut r6 = yukle_z32_satir(coeff, g_off, 6, 16);
        let mut r7 = yukle_z32_satir(coeff, g_off, 7, 16);
        let mut r8 = yukle_z32_satir(coeff, g_off, 8, 16);
        let mut r9 = yukle_z32_satir(coeff, g_off, 9, 16);
        let mut r10 = yukle_z32_satir(coeff, g_off, 10, 16);
        let mut r11 = yukle_z32_satir(coeff, g_off, 11, 16);
        let mut r12 = yukle_z32_satir(coeff, g_off, 12, 16);
        let mut r13 = yukle_z32_satir(coeff, g_off, 13, 16);
        let mut r14 = yukle_z32_satir(coeff, g_off, 14, 16);
        let mut r15 = yukle_z32_satir(coeff, g_off, 15, 16);
        idct16_v!(
            z32, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
        );
        (
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        )
    };
    ekle_yaz_32x1_zmm(dst, x_bayt, z32::round_p2(c0));
    ekle_yaz_32x1_zmm(dst, stride + x_bayt, z32::round_p2(c1));
    ekle_yaz_32x1_zmm(dst, 2 * stride + x_bayt, z32::round_p2(c2));
    ekle_yaz_32x1_zmm(dst, 3 * stride + x_bayt, z32::round_p2(c3));
    ekle_yaz_32x1_zmm(dst, 4 * stride + x_bayt, z32::round_p2(c4));
    ekle_yaz_32x1_zmm(dst, 5 * stride + x_bayt, z32::round_p2(c5));
    ekle_yaz_32x1_zmm(dst, 6 * stride + x_bayt, z32::round_p2(c6));
    ekle_yaz_32x1_zmm(dst, 7 * stride + x_bayt, z32::round_p2(c7));
    ekle_yaz_32x1_zmm(dst, 8 * stride + x_bayt, z32::round_p2(c8));
    ekle_yaz_32x1_zmm(dst, 9 * stride + x_bayt, z32::round_p2(c9));
    ekle_yaz_32x1_zmm(dst, 10 * stride + x_bayt, z32::round_p2(c10));
    ekle_yaz_32x1_zmm(dst, 11 * stride + x_bayt, z32::round_p2(c11));
    ekle_yaz_32x1_zmm(dst, 12 * stride + x_bayt, z32::round_p2(c12));
    ekle_yaz_32x1_zmm(dst, 13 * stride + x_bayt, z32::round_p2(c13));
    ekle_yaz_32x1_zmm(dst, 14 * stride + x_bayt, z32::round_p2(c14));
    ekle_yaz_32x1_zmm(dst, 15 * stride + x_bayt, z32::round_p2(c15));
}

macro_rules! gecis2_dct32_govde {
    ($fn_even:ident, $fn_ana:ident, $m:ident, $yaz_kelebek:ident $(, $x_bayt_arg:ident : usize)?) => {
        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
        #[inline(never)]
        fn $fn_even(coeff: &mut [i32], g_off: usize, ceyrek: bool, yarim: bool) {
            let (e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15) = if ceyrek
            {
                let i0 = yukle_i32x8(coeff, satir_ofseti(g_off, 0, 32));
                let i2 = yukle_i32x8(coeff, satir_ofseti(g_off, 2, 32));
                let i4 = yukle_i32x8(coeff, satir_ofseti(g_off, 4, 32));
                let i6 = yukle_i32x8(coeff, satir_ofseti(g_off, 6, 32));
                idct16_qtr_v!(
                    $m, i0, i2, i4, i6
                    => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
                );
                (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)
            } else if yarim {
                let i0 = yukle_i32x8(coeff, satir_ofseti(g_off, 0, 32));
                let i2 = yukle_i32x8(coeff, satir_ofseti(g_off, 2, 32));
                let i4 = yukle_i32x8(coeff, satir_ofseti(g_off, 4, 32));
                let i6 = yukle_i32x8(coeff, satir_ofseti(g_off, 6, 32));
                let i8 = yukle_i32x8(coeff, satir_ofseti(g_off, 8, 32));
                let i10 = yukle_i32x8(coeff, satir_ofseti(g_off, 10, 32));
                let i12 = yukle_i32x8(coeff, satir_ofseti(g_off, 12, 32));
                let i14 = yukle_i32x8(coeff, satir_ofseti(g_off, 14, 32));
                idct16_half_v!(
                    $m, i0, i2, i4, i6, i8, i10, i12, i14
                    => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
                );
                (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)
            } else {
                let mut r0 = yukle_i32x8(coeff, satir_ofseti(g_off, 0, 32));
                let mut r1 = yukle_i32x8(coeff, satir_ofseti(g_off, 2, 32));
                let mut r2 = yukle_i32x8(coeff, satir_ofseti(g_off, 4, 32));
                let mut r3 = yukle_i32x8(coeff, satir_ofseti(g_off, 6, 32));
                let mut r4 = yukle_i32x8(coeff, satir_ofseti(g_off, 8, 32));
                let mut r5 = yukle_i32x8(coeff, satir_ofseti(g_off, 10, 32));
                let mut r6 = yukle_i32x8(coeff, satir_ofseti(g_off, 12, 32));
                let mut r7 = yukle_i32x8(coeff, satir_ofseti(g_off, 14, 32));
                let mut r8 = yukle_i32x8(coeff, satir_ofseti(g_off, 16, 32));
                let mut r9 = yukle_i32x8(coeff, satir_ofseti(g_off, 18, 32));
                let mut r10 = yukle_i32x8(coeff, satir_ofseti(g_off, 20, 32));
                let mut r11 = yukle_i32x8(coeff, satir_ofseti(g_off, 22, 32));
                let mut r12 = yukle_i32x8(coeff, satir_ofseti(g_off, 24, 32));
                let mut r13 = yukle_i32x8(coeff, satir_ofseti(g_off, 26, 32));
                let mut r14 = yukle_i32x8(coeff, satir_ofseti(g_off, 28, 32));
                let mut r15 = yukle_i32x8(coeff, satir_ofseti(g_off, 30, 32));
                idct16_v!(
                    $m, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
                );
                (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)
            };
            sakla_i32x8(coeff, satir_ofseti(g_off, 0, 32), e0);
            sakla_i32x8(coeff, satir_ofseti(g_off, 2, 32), e1);
            sakla_i32x8(coeff, satir_ofseti(g_off, 4, 32), e2);
            sakla_i32x8(coeff, satir_ofseti(g_off, 6, 32), e3);
            sakla_i32x8(coeff, satir_ofseti(g_off, 8, 32), e4);
            sakla_i32x8(coeff, satir_ofseti(g_off, 10, 32), e5);
            sakla_i32x8(coeff, satir_ofseti(g_off, 12, 32), e6);
            sakla_i32x8(coeff, satir_ofseti(g_off, 14, 32), e7);
            sakla_i32x8(coeff, satir_ofseti(g_off, 16, 32), e8);
            sakla_i32x8(coeff, satir_ofseti(g_off, 18, 32), e9);
            sakla_i32x8(coeff, satir_ofseti(g_off, 20, 32), e10);
            sakla_i32x8(coeff, satir_ofseti(g_off, 22, 32), e11);
            sakla_i32x8(coeff, satir_ofseti(g_off, 24, 32), e12);
            sakla_i32x8(coeff, satir_ofseti(g_off, 26, 32), e13);
            sakla_i32x8(coeff, satir_ofseti(g_off, 28, 32), e14);
            sakla_i32x8(coeff, satir_ofseti(g_off, 30, 32), e15);
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
        #[inline(never)]
        fn $fn_ana(
            dst: &mut [u8],
            stride: usize,
            coeff: &mut [i32],
            g_off: usize,
            $($x_bayt_arg: usize,)?
            ceyrek: bool,
            yarim: bool,
        ) {
            $fn_even(coeff, g_off, ceyrek, yarim);
            let (o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15) = if ceyrek
            {
                let i1 = yukle_i32x8(coeff, satir_ofseti(g_off, 1, 32));
                let i3 = yukle_i32x8(coeff, satir_ofseti(g_off, 3, 32));
                let i5 = yukle_i32x8(coeff, satir_ofseti(g_off, 5, 32));
                let i7 = yukle_i32x8(coeff, satir_ofseti(g_off, 7, 32));
                idct32_qtr_odd_v!(
                    $m, i1, i3, i5, i7
                    => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
                );
                (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)
            } else if yarim {
                let i1 = yukle_i32x8(coeff, satir_ofseti(g_off, 1, 32));
                let i3 = yukle_i32x8(coeff, satir_ofseti(g_off, 3, 32));
                let i5 = yukle_i32x8(coeff, satir_ofseti(g_off, 5, 32));
                let i7 = yukle_i32x8(coeff, satir_ofseti(g_off, 7, 32));
                let i9 = yukle_i32x8(coeff, satir_ofseti(g_off, 9, 32));
                let i11 = yukle_i32x8(coeff, satir_ofseti(g_off, 11, 32));
                let i13 = yukle_i32x8(coeff, satir_ofseti(g_off, 13, 32));
                let i15 = yukle_i32x8(coeff, satir_ofseti(g_off, 15, 32));
                idct32_half_odd_v!(
                    $m, i1, i3, i5, i7, i9, i11, i13, i15
                    => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
                );
                (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)
            } else {
                let t1 = yukle_i32x8(coeff, satir_ofseti(g_off, 1, 32));
                let t3 = yukle_i32x8(coeff, satir_ofseti(g_off, 3, 32));
                let t5 = yukle_i32x8(coeff, satir_ofseti(g_off, 5, 32));
                let t7 = yukle_i32x8(coeff, satir_ofseti(g_off, 7, 32));
                let t9 = yukle_i32x8(coeff, satir_ofseti(g_off, 9, 32));
                let t11 = yukle_i32x8(coeff, satir_ofseti(g_off, 11, 32));
                let t13 = yukle_i32x8(coeff, satir_ofseti(g_off, 13, 32));
                let t15 = yukle_i32x8(coeff, satir_ofseti(g_off, 15, 32));
                let b1 = yukle_i32x8(coeff, satir_ofseti(g_off, 17, 32));
                let b3 = yukle_i32x8(coeff, satir_ofseti(g_off, 19, 32));
                let b5 = yukle_i32x8(coeff, satir_ofseti(g_off, 21, 32));
                let b7 = yukle_i32x8(coeff, satir_ofseti(g_off, 23, 32));
                let b9 = yukle_i32x8(coeff, satir_ofseti(g_off, 25, 32));
                let b11 = yukle_i32x8(coeff, satir_ofseti(g_off, 27, 32));
                let b13 = yukle_i32x8(coeff, satir_ofseti(g_off, 29, 32));
                let b15 = yukle_i32x8(coeff, satir_ofseti(g_off, 31, 32));
                idct32_full_odd_v!(
                    $m, t1, t3, t5, t7, t9, t11, t13, t15, b1, b3, b5, b7, b9, b11, b13, b15
                    => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
                );
                (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)
            };

            macro_rules! yaz_k {
                ($k:literal, $ok:ident) => {
                    let ek = yukle_i32x8(coeff, satir_ofseti(g_off, 2 * $k, 32));
                    $yaz_kelebek(dst, stride, $($x_bayt_arg,)? $k, ek, $ok);
                };
            }
            yaz_k!(0, o0);
            yaz_k!(1, o1);
            yaz_k!(2, o2);
            yaz_k!(3, o3);
            yaz_k!(4, o4);
            yaz_k!(5, o5);
            yaz_k!(6, o6);
            yaz_k!(7, o7);
            yaz_k!(8, o8);
            yaz_k!(9, o9);
            yaz_k!(10, o10);
            yaz_k!(11, o11);
            yaz_k!(12, o12);
            yaz_k!(13, o13);
            yaz_k!(14, o14);
            yaz_k!(15, o15);
        }
    };
}

gecis2_dct32_govde!(
    gecis2_dct32_w8_even,
    gecis2_dct32_w8,
    x8,
    ekle_yaz_w8_split_kelebek32
);
gecis2_dct32_govde!(
    gecis2_dct32_w16_even,
    gecis2_dct32_w16,
    x16,
    ekle_yaz_w16_kelebek32,
    x_bayt: usize
);

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn gecis2_dct64_w16_even_ust(
    coeff: &mut [i32],
    g_off: usize,
    yarim: bool,
    tek_sakla: &mut [__m256i; 16],
) {
    let mut k = 0;
    while k < 16 {
        tek_sakla[k] = yukle_i32x8(coeff, satir_ofseti(g_off, 2 * k + 1, 32));
        k += 1;
    }

    let (e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15) = if yarim {
        let i0 = yukle_i32x8(coeff, satir_ofseti(g_off, 0, 32));
        let i4 = yukle_i32x8(coeff, satir_ofseti(g_off, 4, 32));
        let i8 = yukle_i32x8(coeff, satir_ofseti(g_off, 8, 32));
        let i12 = yukle_i32x8(coeff, satir_ofseti(g_off, 12, 32));
        idct16_qtr_v!(
            x16, i0, i4, i8, i12
            => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
        );
        (
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        )
    } else {
        let i0 = yukle_i32x8(coeff, satir_ofseti(g_off, 0, 32));
        let i4 = yukle_i32x8(coeff, satir_ofseti(g_off, 4, 32));
        let i8 = yukle_i32x8(coeff, satir_ofseti(g_off, 8, 32));
        let i12 = yukle_i32x8(coeff, satir_ofseti(g_off, 12, 32));
        let i16 = yukle_i32x8(coeff, satir_ofseti(g_off, 16, 32));
        let i20 = yukle_i32x8(coeff, satir_ofseti(g_off, 20, 32));
        let i24 = yukle_i32x8(coeff, satir_ofseti(g_off, 24, 32));
        let i28 = yukle_i32x8(coeff, satir_ofseti(g_off, 28, 32));
        idct16_half_v!(
            x16, i0, i4, i8, i12, i16, i20, i24, i28
            => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
        );
        (
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        )
    };

    sakla_i32x8(coeff, satir_ofseti(g_off, slot_e64(0), 32), e0);
    sakla_i32x8(coeff, satir_ofseti(g_off, slot_e64(1), 32), e1);
    sakla_i32x8(coeff, satir_ofseti(g_off, slot_e64(2), 32), e2);
    sakla_i32x8(coeff, satir_ofseti(g_off, slot_e64(3), 32), e3);
    sakla_i32x8(coeff, satir_ofseti(g_off, slot_e64(4), 32), e4);
    sakla_i32x8(coeff, satir_ofseti(g_off, slot_e64(5), 32), e5);
    sakla_i32x8(coeff, satir_ofseti(g_off, slot_e64(6), 32), e6);
    sakla_i32x8(coeff, satir_ofseti(g_off, slot_e64(7), 32), e7);
    sakla_i32x8(coeff, satir_ofseti(g_off, slot_e64(8), 32), e8);
    sakla_i32x8(coeff, satir_ofseti(g_off, slot_e64(9), 32), e9);
    sakla_i32x8(coeff, satir_ofseti(g_off, slot_e64(10), 32), e10);
    sakla_i32x8(coeff, satir_ofseti(g_off, slot_e64(11), 32), e11);
    sakla_i32x8(coeff, satir_ofseti(g_off, slot_e64(12), 32), e12);
    sakla_i32x8(coeff, satir_ofseti(g_off, slot_e64(13), 32), e13);
    sakla_i32x8(coeff, satir_ofseti(g_off, slot_e64(14), 32), e14);
    sakla_i32x8(coeff, satir_ofseti(g_off, slot_e64(15), 32), e15);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn gecis2_dct64_w16_even_alt(coeff: &mut [i32], g_off: usize, yarim: bool) {
    let (o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15) = if yarim {
        let i2 = yukle_i32x8(coeff, satir_ofseti(g_off, 2, 32));
        let i6 = yukle_i32x8(coeff, satir_ofseti(g_off, 6, 32));
        let i10 = yukle_i32x8(coeff, satir_ofseti(g_off, 10, 32));
        let i14 = yukle_i32x8(coeff, satir_ofseti(g_off, 14, 32));
        idct32_qtr_odd_v!(
            x16, i2, i6, i10, i14
            => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
        );
        (
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        )
    } else {
        let i2 = yukle_i32x8(coeff, satir_ofseti(g_off, 2, 32));
        let i6 = yukle_i32x8(coeff, satir_ofseti(g_off, 6, 32));
        let i10 = yukle_i32x8(coeff, satir_ofseti(g_off, 10, 32));
        let i14 = yukle_i32x8(coeff, satir_ofseti(g_off, 14, 32));
        let i18 = yukle_i32x8(coeff, satir_ofseti(g_off, 18, 32));
        let i22 = yukle_i32x8(coeff, satir_ofseti(g_off, 22, 32));
        let i26 = yukle_i32x8(coeff, satir_ofseti(g_off, 26, 32));
        let i30 = yukle_i32x8(coeff, satir_ofseti(g_off, 30, 32));
        idct32_half_odd_v!(
            x16, i2, i6, i10, i14, i18, i22, i26, i30
            => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
        );
        (
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
        )
    };

    macro_rules! sakla_p2_e32 {
        ($idx:literal, $ok:ident) => {{
            let ek = yukle_i32x8(coeff, satir_ofseti(g_off, slot_e64($idx), 32));
            sakla_i32x8(coeff, satir_ofseti(g_off, $idx, 32), x16::adds(ek, $ok));
            sakla_i32x8(
                coeff,
                satir_ofseti(g_off, 31 - $idx, 32),
                x16::subs(ek, $ok),
            );
        }};
    }
    sakla_p2_e32!(0, o0);
    sakla_p2_e32!(1, o1);
    sakla_p2_e32!(2, o2);
    sakla_p2_e32!(3, o3);
    sakla_p2_e32!(4, o4);
    sakla_p2_e32!(5, o5);
    sakla_p2_e32!(6, o6);
    sakla_p2_e32!(7, o7);
    sakla_p2_e32!(8, o8);
    sakla_p2_e32!(9, o9);
    sakla_p2_e32!(10, o10);
    sakla_p2_e32!(11, o11);
    sakla_p2_e32!(12, o12);
    sakla_p2_e32!(13, o13);
    sakla_p2_e32!(14, o14);
    sakla_p2_e32!(15, o15);
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn gecis2_dct64_w16_odd_yaz(
    dst: &mut [u8],
    stride: usize,
    coeff: &[i32],
    g_off: usize,
    x_bayt: usize,
    yarim: bool,
    dis_girdi: &[__m256i; 8],
    tek: &[__m256i; 16],
) {
    let (t32, t33a, t34, t35a, t36, t37a, t38, t39a, t56a, t57, t58a, t59, t60a, t61, t62a, t63) =
        if yarim {
            idct64_odd_dis_qtr_stage1_v!(
                x16, dis_girdi
                => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
            );
            (
                r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
            )
        } else {
            idct64_odd_dis_full_stage1_v!(
                x16, dis_girdi
                => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
            );
            (
                r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
            )
        };

    macro_rules! yaz_cift64 {
        ($k:literal, $tk:ident, $bk:ident) => {{
            let ik_top = tek[7 - $k];
            let ik_bot = tek[8 + $k];
            let hi_15_k = x16::adds($tk, ik_top);
            let lo_k = x16::adds($bk, ik_bot);
            let (hi_k, lo_15_k) = x16::rot_r2(x16::subs($bk, ik_bot), x16::subs($tk, ik_top));

            let e_lo_k = yukle_i32x8(coeff, satir_ofseti(g_off, $k, 32));
            ekle_yaz_w16_kelebek64(dst, stride, x_bayt, $k, e_lo_k, lo_k);
            let e_lo_m = yukle_i32x8(coeff, satir_ofseti(g_off, 15 - $k, 32));
            ekle_yaz_w16_kelebek64(dst, stride, x_bayt, 15 - $k, e_lo_m, lo_15_k);
            let e_hi_k = yukle_i32x8(coeff, satir_ofseti(g_off, 16 + $k, 32));
            ekle_yaz_w16_kelebek64(dst, stride, x_bayt, 16 + $k, e_hi_k, hi_k);
            let e_hi_m = yukle_i32x8(coeff, satir_ofseti(g_off, 31 - $k, 32));
            ekle_yaz_w16_kelebek64(dst, stride, x_bayt, 31 - $k, e_hi_m, hi_15_k);
        }};
    }
    yaz_cift64!(0, t32, t63);
    yaz_cift64!(1, t33a, t62a);
    yaz_cift64!(2, t34, t61);
    yaz_cift64!(3, t35a, t60a);
    yaz_cift64!(4, t36, t59);
    yaz_cift64!(5, t37a, t58a);
    yaz_cift64!(6, t38, t57);
    yaz_cift64!(7, t39a, t56a);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn gecis2_dct64_w16(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32],
    g_off: usize,
    x_bayt: usize,
    yarim: bool,
) {
    let mut tek = [_mm256_setzero_si256(); 16];
    let mut dis_girdi = [_mm256_setzero_si256(); 8];
    gecis2_dct64_w16_even_ust(coeff, g_off, yarim, &mut tek);
    gecis2_dct64_w16_even_alt(coeff, g_off, yarim);
    gecis2_dct64_odd_ic_yari(yarim, &mut tek, &mut dis_girdi);
    gecis2_dct64_w16_odd_yaz(dst, stride, coeff, g_off, x_bayt, yarim, &dis_girdi, &tek);
}

// ---------------------------------------------------------------------------
// `IDTX` (`Identity × Identity`) blok gövdeleri (5 kayıt, 10 bpc)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn idtx_8x32(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let y_blok = if eob <= 42 {
        1
    } else if eob <= 106 {
        2
    } else {
        4
    };
    let m_p2 = _mm256_set1_epi16(8192);
    let mut yb = 0;
    while yb < y_blok {
        let y0 = yb * 8;
        let mut r0 = p1::paket_split::<1>(yukle_i32x8(coeff, y0));
        let mut r1 = p1::paket_split::<1>(yukle_i32x8(coeff, y0 + 32));
        let mut r2 = p1::paket_split::<1>(yukle_i32x8(coeff, y0 + 2 * 32));
        let mut r3 = p1::paket_split::<1>(yukle_i32x8(coeff, y0 + 3 * 32));
        let mut r4 = p1::paket_split::<1>(yukle_i32x8(coeff, y0 + 4 * 32));
        let mut r5 = p1::paket_split::<1>(yukle_i32x8(coeff, y0 + 5 * 32));
        let mut r6 = p1::paket_split::<1>(yukle_i32x8(coeff, y0 + 6 * 32));
        let mut r7 = p1::paket_split::<1>(yukle_i32x8(coeff, y0 + 7 * 32));
        transpose_8x8_split_ymm_v!(r0, r1, r2, r3, r4, r5, r6, r7);
        ekle_yaz_8x1_split(dst, y0 * stride, _mm256_mulhrs_epi16(r0, m_p2));
        ekle_yaz_8x1_split(dst, (y0 + 1) * stride, _mm256_mulhrs_epi16(r1, m_p2));
        ekle_yaz_8x1_split(dst, (y0 + 2) * stride, _mm256_mulhrs_epi16(r2, m_p2));
        ekle_yaz_8x1_split(dst, (y0 + 3) * stride, _mm256_mulhrs_epi16(r3, m_p2));
        ekle_yaz_8x1_split(dst, (y0 + 4) * stride, _mm256_mulhrs_epi16(r4, m_p2));
        ekle_yaz_8x1_split(dst, (y0 + 5) * stride, _mm256_mulhrs_epi16(r5, m_p2));
        ekle_yaz_8x1_split(dst, (y0 + 6) * stride, _mm256_mulhrs_epi16(r6, m_p2));
        ekle_yaz_8x1_split(dst, (y0 + 7) * stride, _mm256_mulhrs_epi16(r7, m_p2));
        yb += 1;
    }
    sifirla_katsayi::<256>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn idtx_32x8(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let x_blok = if eob <= 42 {
        1
    } else if eob <= 106 {
        2
    } else {
        4
    };
    let m_p2 = _mm256_set1_epi16(4096);
    let mut xb = 0;
    while xb < x_blok {
        let x0 = xb * 8;
        let x_bayt = xb * 16;
        let mut r0 = _mm256_packs_epi32(yukle_i32x8(coeff, x0 * 8), _mm256_setzero_si256());
        let mut r1 = _mm256_packs_epi32(yukle_i32x8(coeff, (x0 + 1) * 8), _mm256_setzero_si256());
        let mut r2 = _mm256_packs_epi32(yukle_i32x8(coeff, (x0 + 2) * 8), _mm256_setzero_si256());
        let mut r3 = _mm256_packs_epi32(yukle_i32x8(coeff, (x0 + 3) * 8), _mm256_setzero_si256());
        let mut r4 = _mm256_packs_epi32(yukle_i32x8(coeff, (x0 + 4) * 8), _mm256_setzero_si256());
        let mut r5 = _mm256_packs_epi32(yukle_i32x8(coeff, (x0 + 5) * 8), _mm256_setzero_si256());
        let mut r6 = _mm256_packs_epi32(yukle_i32x8(coeff, (x0 + 6) * 8), _mm256_setzero_si256());
        let mut r7 = _mm256_packs_epi32(yukle_i32x8(coeff, (x0 + 7) * 8), _mm256_setzero_si256());
        transpose_8x8_split_ymm_v!(r0, r1, r2, r3, r4, r5, r6, r7);
        ekle_yaz_8x1_split(dst, x_bayt, _mm256_mulhrs_epi16(r0, m_p2));
        ekle_yaz_8x1_split(dst, stride + x_bayt, _mm256_mulhrs_epi16(r1, m_p2));
        ekle_yaz_8x1_split(dst, 2 * stride + x_bayt, _mm256_mulhrs_epi16(r2, m_p2));
        ekle_yaz_8x1_split(dst, 3 * stride + x_bayt, _mm256_mulhrs_epi16(r3, m_p2));
        ekle_yaz_8x1_split(dst, 4 * stride + x_bayt, _mm256_mulhrs_epi16(r4, m_p2));
        ekle_yaz_8x1_split(dst, 5 * stride + x_bayt, _mm256_mulhrs_epi16(r5, m_p2));
        ekle_yaz_8x1_split(dst, 6 * stride + x_bayt, _mm256_mulhrs_epi16(r6, m_p2));
        ekle_yaz_8x1_split(dst, 7 * stride + x_bayt, _mm256_mulhrs_epi16(r7, m_p2));
        xb += 1;
    }
    sifirla_katsayi::<256>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn idtx_16x32_olcekle(coeff: &[i32], off: usize) -> __m256i {
    let v = p1::yukle_opt::<true>(coeff, off);
    let s = _mm256_srai_epi32::<12>(_mm256_add_epi32(
        _mm256_mullo_epi32(v, _mm256_set1_epi32(5793)),
        _mm256_set1_epi32(3072),
    ));
    _mm256_packs_epi32(s, _mm256_setzero_si256())
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
    let m_p2 = _mm256_set1_epi16(8192);

    let mut xb = 0;
    while xb < x_blok {
        let x0 = xb * 8;
        let x_bayt = xb * 16;
        let mut yb = 0;
        while yb < y_blok {
            let y0 = yb * 8;
            let mut r0 = idtx_16x32_olcekle(coeff, y0 + x0 * 32);
            let mut r1 = idtx_16x32_olcekle(coeff, y0 + (x0 + 1) * 32);
            let mut r2 = idtx_16x32_olcekle(coeff, y0 + (x0 + 2) * 32);
            let mut r3 = idtx_16x32_olcekle(coeff, y0 + (x0 + 3) * 32);
            let mut r4 = idtx_16x32_olcekle(coeff, y0 + (x0 + 4) * 32);
            let mut r5 = idtx_16x32_olcekle(coeff, y0 + (x0 + 5) * 32);
            let mut r6 = idtx_16x32_olcekle(coeff, y0 + (x0 + 6) * 32);
            let mut r7 = idtx_16x32_olcekle(coeff, y0 + (x0 + 7) * 32);
            transpose_8x8_split_ymm_v!(r0, r1, r2, r3, r4, r5, r6, r7);
            ekle_yaz_8x1_split(dst, y0 * stride + x_bayt, _mm256_mulhrs_epi16(r0, m_p2));
            ekle_yaz_8x1_split(
                dst,
                (y0 + 1) * stride + x_bayt,
                _mm256_mulhrs_epi16(r1, m_p2),
            );
            ekle_yaz_8x1_split(
                dst,
                (y0 + 2) * stride + x_bayt,
                _mm256_mulhrs_epi16(r2, m_p2),
            );
            ekle_yaz_8x1_split(
                dst,
                (y0 + 3) * stride + x_bayt,
                _mm256_mulhrs_epi16(r3, m_p2),
            );
            ekle_yaz_8x1_split(
                dst,
                (y0 + 4) * stride + x_bayt,
                _mm256_mulhrs_epi16(r4, m_p2),
            );
            ekle_yaz_8x1_split(
                dst,
                (y0 + 5) * stride + x_bayt,
                _mm256_mulhrs_epi16(r5, m_p2),
            );
            ekle_yaz_8x1_split(
                dst,
                (y0 + 6) * stride + x_bayt,
                _mm256_mulhrs_epi16(r6, m_p2),
            );
            ekle_yaz_8x1_split(
                dst,
                (y0 + 7) * stride + x_bayt,
                _mm256_mulhrs_epi16(r7, m_p2),
            );
            yb += 1;
        }
        xb += 1;
    }
    sifirla_katsayi::<512>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn idtx_32x16_yukle(coeff: &[i32], off: usize) -> __m256i {
    let v = p1::yukle_opt::<true>(coeff, off);
    _mm256_packs_epi32(_mm256_add_epi32(v, v), _mm256_setzero_si256())
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn idtx_32x16_p2_yaz(dst: &mut [u8], off: usize, t: __m256i) {
    let k16 = _mm256_set1_epi16(1697 * 16);
    let iki_bir = _mm256_set1_epi32(cift_katsayi(2, 1));
    let sekiz = _mm256_set1_epi32(8);
    let u = _mm256_mulhrs_epi16(t, k16);
    let lo = _mm256_unpacklo_epi16(t, u);
    let s = _mm256_srai_epi32::<4>(_mm256_add_epi32(
        _mm256_dpwssd_epi32(_mm256_setzero_si256(), lo, iki_bir),
        sekiz,
    ));
    let r = _mm256_packs_epi32(s, s);
    ekle_yaz_8x1_split(dst, off, r);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
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
            let x0 = xb * 8;
            let x_bayt = xb * 16;
            let mut r0 = idtx_32x16_yukle(coeff, y0 + x0 * 16);
            let mut r1 = idtx_32x16_yukle(coeff, y0 + (x0 + 1) * 16);
            let mut r2 = idtx_32x16_yukle(coeff, y0 + (x0 + 2) * 16);
            let mut r3 = idtx_32x16_yukle(coeff, y0 + (x0 + 3) * 16);
            let mut r4 = idtx_32x16_yukle(coeff, y0 + (x0 + 4) * 16);
            let mut r5 = idtx_32x16_yukle(coeff, y0 + (x0 + 5) * 16);
            let mut r6 = idtx_32x16_yukle(coeff, y0 + (x0 + 6) * 16);
            let mut r7 = idtx_32x16_yukle(coeff, y0 + (x0 + 7) * 16);
            transpose_8x8_split_ymm_v!(r0, r1, r2, r3, r4, r5, r6, r7);
            idtx_32x16_p2_yaz(dst, y0 * stride + x_bayt, r0);
            idtx_32x16_p2_yaz(dst, (y0 + 1) * stride + x_bayt, r1);
            idtx_32x16_p2_yaz(dst, (y0 + 2) * stride + x_bayt, r2);
            idtx_32x16_p2_yaz(dst, (y0 + 3) * stride + x_bayt, r3);
            idtx_32x16_p2_yaz(dst, (y0 + 4) * stride + x_bayt, r4);
            idtx_32x16_p2_yaz(dst, (y0 + 5) * stride + x_bayt, r5);
            idtx_32x16_p2_yaz(dst, (y0 + 6) * stride + x_bayt, r6);
            idtx_32x16_p2_yaz(dst, (y0 + 7) * stride + x_bayt, r7);
            xb += 1;
        }
        yb += 1;
    }
    sifirla_katsayi::<512>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn idtx_32x32(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    let blok = if eob <= 35 {
        1
    } else if eob <= 135 {
        2
    } else {
        4
    };
    let m_p2 = _mm256_set1_epi16(8192);
    let mut yb = 0;
    while yb < blok {
        let y0 = yb * 8;
        let mut xb = 0;
        while xb < blok {
            let x0 = xb * 8;
            let x_bayt = xb * 16;
            let mut r0 =
                _mm256_packs_epi32(yukle_i32x8(coeff, y0 + x0 * 32), _mm256_setzero_si256());
            let mut r1 = _mm256_packs_epi32(
                yukle_i32x8(coeff, y0 + (x0 + 1) * 32),
                _mm256_setzero_si256(),
            );
            let mut r2 = _mm256_packs_epi32(
                yukle_i32x8(coeff, y0 + (x0 + 2) * 32),
                _mm256_setzero_si256(),
            );
            let mut r3 = _mm256_packs_epi32(
                yukle_i32x8(coeff, y0 + (x0 + 3) * 32),
                _mm256_setzero_si256(),
            );
            let mut r4 = _mm256_packs_epi32(
                yukle_i32x8(coeff, y0 + (x0 + 4) * 32),
                _mm256_setzero_si256(),
            );
            let mut r5 = _mm256_packs_epi32(
                yukle_i32x8(coeff, y0 + (x0 + 5) * 32),
                _mm256_setzero_si256(),
            );
            let mut r6 = _mm256_packs_epi32(
                yukle_i32x8(coeff, y0 + (x0 + 6) * 32),
                _mm256_setzero_si256(),
            );
            let mut r7 = _mm256_packs_epi32(
                yukle_i32x8(coeff, y0 + (x0 + 7) * 32),
                _mm256_setzero_si256(),
            );
            transpose_8x8_split_ymm_v!(r0, r1, r2, r3, r4, r5, r6, r7);
            ekle_yaz_8x1_split(dst, y0 * stride + x_bayt, _mm256_mulhrs_epi16(r0, m_p2));
            ekle_yaz_8x1_split(
                dst,
                (y0 + 1) * stride + x_bayt,
                _mm256_mulhrs_epi16(r1, m_p2),
            );
            ekle_yaz_8x1_split(
                dst,
                (y0 + 2) * stride + x_bayt,
                _mm256_mulhrs_epi16(r2, m_p2),
            );
            ekle_yaz_8x1_split(
                dst,
                (y0 + 3) * stride + x_bayt,
                _mm256_mulhrs_epi16(r3, m_p2),
            );
            ekle_yaz_8x1_split(
                dst,
                (y0 + 4) * stride + x_bayt,
                _mm256_mulhrs_epi16(r4, m_p2),
            );
            ekle_yaz_8x1_split(
                dst,
                (y0 + 5) * stride + x_bayt,
                _mm256_mulhrs_epi16(r5, m_p2),
            );
            ekle_yaz_8x1_split(
                dst,
                (y0 + 6) * stride + x_bayt,
                _mm256_mulhrs_epi16(r6, m_p2),
            );
            ekle_yaz_8x1_split(
                dst,
                (y0 + 7) * stride + x_bayt,
                _mm256_mulhrs_epi16(r7, m_p2),
            );
            xb += 1;
        }
        yb += 1;
    }
    sifirla_katsayi::<1024>(coeff);
}

// ---------------------------------------------------------------------------
// `DCT_DCT` büyük blok gövdeleri (10 kayıt, 10 bpc)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_8x32(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<8, 32, 2, false>(dst, stride, coeff);
        return;
    }
    let ceyrek = eob <= 42;
    let yarim = eob <= 106;
    let y_blok = if ceyrek {
        1
    } else if yarim {
        2
    } else {
        4
    };
    let mut yb = 0;
    while yb < y_blok {
        gecis1_dct8_serit(coeff, yb * 8);
        yb += 1;
    }
    gecis2_dct32_w8(dst, stride, coeff, 0, ceyrek, yarim);
    sifirla_katsayi::<256>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_32x8(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<32, 8, 2, false>(dst, stride, coeff);
        return;
    }
    gecis1_dct32_serit::<2, false>(coeff, 0, 8, eob <= 42, eob <= 106);
    gecis2_dct8_w32(dst, stride, coeff, 0, 0);
    sifirla_katsayi::<256>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_16x32(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<16, 32, 1, true>(dst, stride, coeff);
        return;
    }
    let ceyrek = eob <= 35;
    let yarim = eob <= 150;
    let y_blok = if ceyrek {
        1
    } else if yarim {
        2
    } else {
        4
    };
    let mut yb = 0;
    while yb < y_blok {
        gecis1_dct16_serit::<1, true>(coeff, yb * 8, 32, ceyrek);
        yb += 1;
    }
    gecis2_dct32_w16(dst, stride, coeff, 0, 0, ceyrek, yarim);
    sifirla_katsayi::<512>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_32x16(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<32, 16, 1, true>(dst, stride, coeff);
        return;
    }
    let ceyrek = eob <= 35;
    let yarim = eob <= 150;
    let y_blok = if ceyrek { 1 } else { 2 };
    let mut yb = 0;
    while yb < y_blok {
        gecis1_dct32_serit::<1, true>(coeff, yb * 8, 16, ceyrek, yarim);
        yb += 1;
    }
    gecis2_dct16_w32(dst, stride, coeff, 0, 0, ceyrek);
    sifirla_katsayi::<512>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_32x32(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<32, 32, 2, false>(dst, stride, coeff);
        return;
    }
    let ceyrek = eob <= 35;
    let yarim = eob <= 135;
    let y_blok = if ceyrek {
        1
    } else if yarim {
        2
    } else {
        4
    };
    let mut yb = 0;
    while yb < y_blok {
        gecis1_dct32_serit::<2, false>(coeff, yb * 8, 32, ceyrek, yarim);
        yb += 1;
    }
    gecis2_dct32_w16(dst, stride, coeff, 0, 0, ceyrek, yarim);
    gecis2_dct32_w16(dst, stride, coeff, 8, 32, ceyrek, yarim);
    sifirla_katsayi::<1024>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_16x64(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<16, 64, 2, false>(dst, stride, coeff);
        return;
    }
    let ceyrek = eob <= 35;
    let yarim = eob <= 150;
    let y_blok = if ceyrek {
        1
    } else if yarim {
        2
    } else {
        4
    };
    let mut yb = 0;
    while yb < y_blok {
        gecis1_dct16_serit::<2, false>(coeff, yb * 8, 32, ceyrek);
        yb += 1;
    }
    gecis2_dct64_w16(dst, stride, coeff, 0, 0, yarim);
    sifirla_katsayi::<512>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_64x16(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<64, 16, 2, false>(dst, stride, coeff);
        return;
    }
    let ceyrek = eob <= 35;
    let yarim = eob <= 150;
    let y_blok = if ceyrek { 1 } else { 2 };
    let mut yb = 0;
    while yb < y_blok {
        gecis1_dct64_serit::<2, false>(coeff, yb * 8, 16, yarim);
        yb += 1;
    }
    gecis2_dct16_w32(dst, stride, coeff, 0, 0, ceyrek);
    gecis2_dct16_w32(dst, stride, coeff, 16, 64, ceyrek);
    sifirla_katsayi::<512>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_32x64(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<32, 64, 1, true>(dst, stride, coeff);
        return;
    }
    let ceyrek = eob <= 35;
    let yarim = eob <= 135;
    let y_blok = if ceyrek {
        1
    } else if yarim {
        2
    } else {
        4
    };
    let mut yb = 0;
    while yb < y_blok {
        gecis1_dct32_serit::<1, true>(coeff, yb * 8, 32, ceyrek, yarim);
        yb += 1;
    }
    gecis2_dct64_w16(dst, stride, coeff, 0, 0, yarim);
    gecis2_dct64_w16(dst, stride, coeff, 8, 32, yarim);
    sifirla_katsayi::<1024>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_64x32(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<64, 32, 1, true>(dst, stride, coeff);
        return;
    }
    let ceyrek = eob <= 35;
    let yarim = eob <= 135;
    let y_blok = if ceyrek {
        1
    } else if yarim {
        2
    } else {
        4
    };
    let mut yb = 0;
    while yb < y_blok {
        gecis1_dct64_serit::<1, true>(coeff, yb * 8, 32, yarim);
        yb += 1;
    }
    let mut g = 0;
    while g < 4 {
        gecis2_dct32_w16(dst, stride, coeff, g * 8, g * 32, ceyrek, yarim);
        g += 1;
    }
    sifirla_katsayi::<1024>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_64x64(dst: &mut [u8], stride: usize, coeff: &mut [i32], eob: usize) {
    if eob == 0 {
        dc_yalniz::<64, 64, 2, false>(dst, stride, coeff);
        return;
    }
    let ceyrek = eob <= 35;
    let yarim = eob <= 135;
    let y_blok = if ceyrek {
        1
    } else if yarim {
        2
    } else {
        4
    };
    let mut yb = 0;
    while yb < y_blok {
        gecis1_dct64_serit::<2, false>(coeff, yb * 8, 32, yarim);
        yb += 1;
    }
    let mut g = 0;
    while g < 4 {
        gecis2_dct64_w16(dst, stride, coeff, g * 8, g * 32, yarim);
        g += 1;
    }
    sifirla_katsayi::<1024>(coeff);
}

/// `(TX, TXTP)` çiftine özelleştirilmiş büyük (`max(w, h) ∈ {32, 64}`) 10 bpc AVX512ICL çekirdek gövdesi.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn itx10_buyuk_avx512_cekirdek<const TX: u8, const TXTP: u8>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
) {
    let tx = const { TxSize::from_index(TX as usize).unwrap() };
    let txtp = const { TxType::from_index(TXTP as usize).unwrap() };
    match (tx, txtp) {
        (TxSize::Rtx8x32, TxType::DctDct) => dct_8x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx8x32, TxType::Idtx) => idtx_8x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x8, TxType::DctDct) => dct_32x8(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x8, TxType::Idtx) => idtx_32x8(dst, dst_stride, coeff, eob),
        (TxSize::Rtx16x32, TxType::DctDct) => dct_16x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx16x32, TxType::Idtx) => idtx_16x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x16, TxType::DctDct) => dct_32x16(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x16, TxType::Idtx) => idtx_32x16(dst, dst_stride, coeff, eob),
        (TxSize::Tx32x32, TxType::DctDct) => dct_32x32(dst, dst_stride, coeff, eob),
        (TxSize::Tx32x32, TxType::Idtx) => idtx_32x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx16x64, TxType::DctDct) => dct_16x64(dst, dst_stride, coeff, eob),
        (TxSize::Rtx64x16, TxType::DctDct) => dct_64x16(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x64, TxType::DctDct) => dct_32x64(dst, dst_stride, coeff, eob),
        (TxSize::Rtx64x32, TxType::DctDct) => dct_64x32(dst, dst_stride, coeff, eob),
        (TxSize::Tx64x64, TxType::DctDct) => dct_64x64(dst, dst_stride, coeff, eob),
        _ => unreachable!(),
    }
}

/// `(TX, TXTP)` çiftinin sözleşme denetimli güvenli büyük 10 bpc AVX512ICL yuva girişi.
fn itx10_buyuk_avx512_giris<const TX: u8, const TXTP: u8>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
    bit_depth: BitDepth,
) {
    if bit_depth != BitDepth::Ten
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
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`itx10_buyuk_avx512_fn`); belirteç AVX512ICL (F/BW/VL/VBMI/VNNI) tespiti ve maske
    // izninin kanıtıdır. (2) sınır: `itx16_sozlesmesi` `dst_stride`'ın çift ve `≥ w·2`
    // olduğunu, `dst`'nin `(h − 1)·dst_stride + w·2` baytı ve `coeff`'un `1024` öğeyi
    // kapsadığını garanti eder; bütün bellek erişimleri sabit boyutlu `chunk`/`chunk_mut`
    // dizileri üzerindendir. (3) hizalama/takma ad: `dst` ve `coeff` ayrık `&mut`
    // referanslardır, hizasız erişim kullanılır.
    unsafe {
        itx10_buyuk_avx512_cekirdek::<TX, TXTP>(dst, dst_stride, coeff, eob);
    }
}

/// Çalışma zamanı `(tx, txtp)` çifti için belirteç kapılı `max(w, h) ∈ {32, 64}` 10 bpc AVX512ICL `ItxFn` üreticisi (`15` kayıt).
#[must_use]
pub const fn itx10_buyuk_avx512_fn(
    _token: Avx512IclToken,
    tx: TxSize,
    txtp: TxType,
) -> Option<ItxFn> {
    match (tx, txtp) {
        (TxSize::Rtx8x32, TxType::DctDct) => {
            Some(itx10_buyuk_avx512_giris::<{ TxSize::Rtx8x32 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx8x32, TxType::Idtx) => {
            Some(itx10_buyuk_avx512_giris::<{ TxSize::Rtx8x32 as u8 }, { TxType::Idtx as u8 }>)
        }
        (TxSize::Rtx32x8, TxType::DctDct) => {
            Some(itx10_buyuk_avx512_giris::<{ TxSize::Rtx32x8 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx32x8, TxType::Idtx) => {
            Some(itx10_buyuk_avx512_giris::<{ TxSize::Rtx32x8 as u8 }, { TxType::Idtx as u8 }>)
        }
        (TxSize::Rtx16x32, TxType::DctDct) => {
            Some(itx10_buyuk_avx512_giris::<{ TxSize::Rtx16x32 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx16x32, TxType::Idtx) => {
            Some(itx10_buyuk_avx512_giris::<{ TxSize::Rtx16x32 as u8 }, { TxType::Idtx as u8 }>)
        }
        (TxSize::Rtx32x16, TxType::DctDct) => {
            Some(itx10_buyuk_avx512_giris::<{ TxSize::Rtx32x16 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx32x16, TxType::Idtx) => {
            Some(itx10_buyuk_avx512_giris::<{ TxSize::Rtx32x16 as u8 }, { TxType::Idtx as u8 }>)
        }
        (TxSize::Tx32x32, TxType::DctDct) => {
            Some(itx10_buyuk_avx512_giris::<{ TxSize::Tx32x32 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Tx32x32, TxType::Idtx) => {
            Some(itx10_buyuk_avx512_giris::<{ TxSize::Tx32x32 as u8 }, { TxType::Idtx as u8 }>)
        }
        (TxSize::Rtx16x64, TxType::DctDct) => {
            Some(itx10_buyuk_avx512_giris::<{ TxSize::Rtx16x64 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx64x16, TxType::DctDct) => {
            Some(itx10_buyuk_avx512_giris::<{ TxSize::Rtx64x16 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx32x64, TxType::DctDct) => {
            Some(itx10_buyuk_avx512_giris::<{ TxSize::Rtx32x64 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Rtx64x32, TxType::DctDct) => {
            Some(itx10_buyuk_avx512_giris::<{ TxSize::Rtx64x32 as u8 }, { TxType::DctDct as u8 }>)
        }
        (TxSize::Tx64x64, TxType::DctDct) => {
            Some(itx10_buyuk_avx512_giris::<{ TxSize::Tx64x64 as u8 }, { TxType::DctDct as u8 }>)
        }
        _ => None,
    }
}

/// Çalışma zamanı `(tx, txtp)` çifti için `max(w, h) ∈ {32, 64}` 10 bpc AVX512ICL yuva tablosu (`15` kayıt).
#[must_use]
pub const fn itx10_buyuk_avx512_lookup(
    token: Avx512IclToken,
    tx: TxSize,
    txtp: TxType,
) -> Option<ItxFn> {
    itx10_buyuk_avx512_fn(token, tx, txtp)
}
