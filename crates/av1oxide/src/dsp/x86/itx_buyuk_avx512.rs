//! 8 bpc büyük blok ters dönüşüm (`max(w, h) ∈ {32, 64}`) x86_64 AVX512ICL çekirdekleri — F4-ITX a (P43 Adım 3).
//!
//! Bu modül dav1d'nin `src/x86/itx_avx512.asm` ve `src/x86/itx.h:236-244` dosyalarındaki
//! `max(w, h) ∈ {32, 64}` boyutlu **15** adet 8 bpc AVX512ICL ters dönüşüm kaydını saf kararlı
//! Rust `core::arch::x86_64` intrinsics ile gerçekler (`.asm`, `.S`, `asm!`, `global_asm!`
//! ve dış C bağımlılığı yoktur; CEK-02):
//!
//! - `RTX_8X32`, `RTX_32X8`, `RTX_16X32`, `RTX_32X16`, `TX_32X32`: `DctDct` ve `Idtx` (`5 × 2 = 10`),
//! - `RTX_16X64`, `RTX_64X16`, `RTX_32X64`, `RTX_64X32`, `TX_64X64`: `DctDct` (`5 × 1 = 5`).
//!
//! ## Mimari kurallar, Zen 4 (`SLOW_GATHER`) uyumu ve sıfır yığın taşması
//!
//! - **P34 modül izolasyonu:** Bu modül `itx_avx512`, `itx`, `itx_buyuk`, `itx_ssse3` ya da
//!   `itx_wht` ile hiçbir `#[target_feature]` yardımcısı paylaşmaz.
//! - **Zen 4 kuralı (`0` gather/scatter):** `dav1d` 8 bpc AVX512ICL ITX yuvalarını `SLOW_GATHER`
//!   (`0x3f` Zen 4) altında da canlı tabloya kurar (`src/x86/itx.h:221-244`). Bu modülde
//!   `vpgather*` ve `vpscatter*` yönergeleri kesinlikle kullanılmaz (`0` adet); `idtx_8x32`
//!   dâhil bütün çekirdekler yazmaç içi ZMM/YMM transpoz + güvenli dilim indekslemesi kullanır.
//! - **512-bit ZMM (`z32`) + sıfır dev yığın çerçevesi (`sub < $0x400, %rsp`):**
//!   32 genişlikli satır/sütun geçişleri ve `32`/`64` baytlık hedef eklemeleri `_mm512_*`
//!   (`__m512i`, tek yazmaçta `32 × i16`) ve `_mm512_dpwssd_epi32` (AVX-512 VNNI) ile işlenir.
//!   Çoklu vektör 1B dönüşüm ve transpoz adımları yerel yazmaç değişkenleri üzerinde makrolarla
//!   açılır; hiçbir işlev sınırından vektör dizisi (`[V; 16/32]`) geçirilmez ve ara aşamalar
//!   `coeff[..sw * sh]` (`sw = min(w, 32)`, `sh = min(h, 32)`) içinde aşamalandırılıp sıfırlanır.
//! - **Katsayı sıfırlama sözleşmesi:**
//!   - `DctDct && eob == 0`: yalnız `coeff[0] = 0` sıfırlanır; `coeff[1..1024]` korunur.
//!   - Genel yol (`eob ≥ 1` veya `txtp != DctDct`): `eob` alt-blok hızlı yolunda dahi
//!     `coeff[..sw * sh]` sıfırlanır; `coeff[sw * sh..1024]` kuyruğuna dokunulmaz.
//! - **Tek `unsafe` bloğu:** Yalnız belirteçli giriş sarmalayıcısındadır ([`itx8_buyuk_avx512_giris`]).

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm_add_epi16, _mm_adds_epi16, _mm_adds_epu8, _mm_cvtepu8_epi16,
    _mm_cvtsi64_si128, _mm_cvtsi128_si64, _mm_dpwssd_epi32, _mm_mulhrs_epi16, _mm_packs_epi32,
    _mm_packus_epi16, _mm_set1_epi8, _mm_set1_epi16, _mm_set1_epi32, _mm_setr_epi32,
    _mm_setzero_si128, _mm_srai_epi32, _mm_srli_si128, _mm_subs_epi16, _mm_subs_epu8,
    _mm_unpackhi_epi16, _mm_unpacklo_epi16, _mm_unpacklo_epi64, _mm256_add_epi16,
    _mm256_adds_epi16, _mm256_adds_epu8, _mm256_castsi256_si128, _mm256_cvtepu8_epi16,
    _mm256_dpwssd_epi32, _mm256_extracti128_si256, _mm256_mulhrs_epi16, _mm256_packs_epi32,
    _mm256_packus_epi16, _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set1_epi8,
    _mm256_set1_epi16, _mm256_set1_epi32, _mm256_setr_epi32, _mm256_setzero_si256,
    _mm256_srai_epi32, _mm256_srli_si256, _mm256_subs_epi16, _mm256_subs_epu8,
    _mm256_unpackhi_epi16, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi16,
    _mm256_unpacklo_epi32, _mm256_unpacklo_epi64, _mm512_add_epi16, _mm512_adds_epi16,
    _mm512_adds_epu8, _mm512_castsi256_si512, _mm512_castsi512_si128, _mm512_castsi512_si256,
    _mm512_cvtepu8_epi16, _mm512_dpwssd_epi32, _mm512_extracti32x4_epi32,
    _mm512_extracti64x4_epi64, _mm512_inserti64x4, _mm512_mulhrs_epi16, _mm512_packs_epi32,
    _mm512_permutexvar_epi64, _mm512_set1_epi8, _mm512_set1_epi16, _mm512_set1_epi32,
    _mm512_setr_epi32, _mm512_setr_epi64, _mm512_setzero_si512, _mm512_srai_epi32,
    _mm512_subs_epi16, _mm512_subs_epu8, _mm512_unpackhi_epi16, _mm512_unpackhi_epi32,
    _mm512_unpackhi_epi64, _mm512_unpacklo_epi16, _mm512_unpacklo_epi32, _mm512_unpacklo_epi64,
};

use super::mem::{
    chunk, chunk_mut, load_u8x16, load_u8x32, load_u8x64, store_i32x8, store_u8x8, store_u8x16,
    store_u8x32, store_u8x64,
};
use crate::cpu::Avx512IclToken;
use crate::dsp::itx::{ItxFn, TxSize, TxType, itx_skaler_devret, itx8_sozlesmesi};
use crate::frame::BitDepth;

/// İki `i16` katsayısını `_mm*_dpwssd_epi32` için tek `i32` sabitine paketler (`[lo, hi]`).
#[inline(always)]
const fn pair(lo: i16, hi: i16) -> i32 {
    (lo as u16 as i32) | ((hi as u16 as i32) << 16)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_i32x4(c: &[i32; 4]) -> __m128i {
    _mm_setr_epi32(c[0], c[1], c[2], c[3])
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_i16x8_128(c: &[i32; 8]) -> __m128i {
    _mm_packs_epi32(yukle_i32x4(chunk(c, 0)), yukle_i32x4(chunk(c, 4)))
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn sakla_i32x4(dst: &mut [i32; 4], v: __m128i) {
    let q0 = _mm_cvtsi128_si64(v) as u64;
    let q1 = _mm_cvtsi128_si64(_mm_srli_si128::<8>(v)) as u64;
    *dst = [q0 as i32, (q0 >> 32) as i32, q1 as i32, (q1 >> 32) as i32];
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_i32x8(c: &[i32; 8]) -> __m256i {
    _mm256_setr_epi32(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7])
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_i32x16(c: &[i32; 16]) -> __m512i {
    _mm512_setr_epi32(
        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12], c[13],
        c[14], c[15],
    )
}

/// 8 × `i32` dilimini Split-YMM düzeninde (`[c0..c3, 0..0 || c4..c7, 0..0]`) doyurmalı `i16` olarak yükler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_i16x8(c: &[i32; 8]) -> __m256i {
    _mm256_packs_epi32(yukle_i32x8(c), _mm256_setzero_si256())
}

/// İki ayrı `8 × i32` sütun dilimini `[c0 || c1]` düzeninde `16 × i16` (`__m256i`) olarak yükler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_2col_8(c0: &[i32; 8], c1: &[i32; 8]) -> __m256i {
    _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi32(yukle_i32x8(c0), yukle_i32x8(c1)))
}

/// 16 × `i32` dilimini doğal sırada doyurmalı `16 × i16` (`__m256i`) olarak yükler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_i16x16(c: &[i32; 16]) -> __m256i {
    yukle_2col_8(chunk(c, 0), chunk(c, 8))
}

/// 32 × `i32` dilimini doğal sırada doyurmalı `32 × i16` (`__m512i`) olarak yükler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_z32(c: &[i32; 32]) -> __m512i {
    let lo = yukle_i32x16(chunk(c, 0));
    let hi = yukle_i32x16(chunk(c, 16));
    let idx = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    _mm512_permutexvar_epi64(idx, _mm512_packs_epi32(lo, hi))
}

/// Dört ayrı `8 × i32` sütun dilimini `[c0 || c1 || c2 || c3]` düzeninde `32 × i16` (`__m512i`) yükler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn yukle_4col_8_zmm(c0: &[i32; 8], c1: &[i32; 8], c2: &[i32; 8], c3: &[i32; 8]) -> __m512i {
    let lo = yukle_2col_8(c0, c1);
    let hi = yukle_2col_8(c2, c3);
    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
}

/// İki `__m256i` vektörünü `[lo || hi]` olarak tek `__m512i`'de birleştirir.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn birlestir_zmm(lo: __m256i, hi: __m256i) -> __m512i {
    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
}

/// `__m512i` yazmacını `16 × i32` dilimine saklar (`mem.rs` güvenli yardımcısıyla).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn sakla_zmm(dst: &mut [i32; 16], v: __m512i) {
    store_i32x8(chunk_mut(dst, 0), _mm512_castsi512_si256(v));
    store_i32x8(chunk_mut(dst, 8), _mm512_extracti64x4_epi64::<1>(v));
}

/// `16 × i32` diliminden `__m512i` yazmacını geri okur.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn oku_zmm(src: &[i32; 16]) -> __m512i {
    yukle_i32x16(src)
}

/// `coeff[..N]` bölgesini (`N` 8'in katı) 256-bit sıfır saklamalarıyla temizler.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn sifirla_katsayi<const N: usize>(coeff: &mut [i32; 1024]) {
    const { assert!(N.is_multiple_of(8) && N <= 1024) };
    let z = _mm256_setzero_si256();
    let mut i = 0;
    while i < N {
        store_i32x8(chunk_mut(coeff, i), z);
        i += 8;
    }
}

/// Şerit-paralel 1B yaprak yardımcılar (yalnız 1–2 skaler vektör değer döndürür, `0` yığın baytı):
/// - `x8`: Split-YMM `__m256i` (`$HALF_LANE = true`, 8 şerit),
/// - `x16`: YMM `__m256i` (`$HALF_LANE = false`, 16 şerit),
/// - `z32`: Tam ZMM `__m512i` (`$HALF_LANE = false`, 32 şerit).
macro_rules! buyuk_1d_avx512_yaprak {
    (
        $mod_ad:ident,
        $V:ty,
        $HALF_LANE:literal,
        $set1_16:ident,
        $set1_32:ident,
        $zero:ident,
        $unpacklo16:ident,
        $unpackhi16:ident,
        $dpwssd:ident,
        $srai32:ident,
        $packs32:ident,
        $adds16:ident,
        $subs16:ident,
        $mulhrs16:ident
    ) => {
        #[allow(dead_code)]
        mod $mod_ad {
            use super::*;

            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn zero() -> $V {
                $zero()
            }

            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn adds(a: $V, b: $V) -> $V {
                $adds16(a, b)
            }

            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn subs(a: $V, b: $V) -> $V {
                $subs16(a, b)
            }

            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn mul_r2(v: $V) -> $V {
                $mulhrs16(v, $set1_16(23168))
            }

            /// `(v * c + 2048) >> 12 == vpmulhrsw(v, c * 8)` (`|c| <= 4095`).
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn mul12(v: $V, c: i16) -> $V {
                $mulhrs16(v, $set1_16(c * 8))
            }

            /// `(((a*c1 - b*c2 + 2048) >> 12), ((a*c2 + b*c1 + 2048) >> 12))` (`vpdpwssd`).
            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn mulsub(a: $V, b: $V, c1: i16, c2: i16) -> ($V, $V) {
                let lo = $unpacklo16(b, a);
                let k1 = $set1_32(pair(c2.wrapping_neg(), c1));
                let k2 = $set1_32(pair(c1, c2));
                let rnd = $set1_32(2048);
                let r1_lo = $srai32::<12>($dpwssd(rnd, lo, k1));
                let r2_lo = $srai32::<12>($dpwssd(rnd, lo, k2));
                if $HALF_LANE {
                    ($packs32(r1_lo, r1_lo), $packs32(r2_lo, r2_lo))
                } else {
                    let hi = $unpackhi16(b, a);
                    let r1_hi = $srai32::<12>($dpwssd(rnd, hi, k1));
                    let r2_hi = $srai32::<12>($dpwssd(rnd, hi, k2));
                    ($packs32(r1_lo, r1_hi), $packs32(r2_lo, r2_hi))
                }
            }

            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn round_one<const SHIFT: u32>(v: $V) -> $V {
                if SHIFT == 1 {
                    $mulhrs16(v, $set1_16(16384))
                } else if SHIFT == 2 {
                    $mulhrs16(v, $set1_16(8192))
                } else {
                    v
                }
            }

            #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
            #[inline]
            pub(super) fn round_p2(v: $V) -> $V {
                $mulhrs16(v, $set1_16(2048))
            }
        }
    };
}

buyuk_1d_avx512_yaprak!(
    n8,
    __m128i,
    false,
    _mm_set1_epi16,
    _mm_set1_epi32,
    _mm_setzero_si128,
    _mm_unpacklo_epi16,
    _mm_unpackhi_epi16,
    _mm_dpwssd_epi32,
    _mm_srai_epi32,
    _mm_packs_epi32,
    _mm_adds_epi16,
    _mm_subs_epi16,
    _mm_mulhrs_epi16
);

buyuk_1d_avx512_yaprak!(
    x8,
    __m256i,
    true,
    _mm256_set1_epi16,
    _mm256_set1_epi32,
    _mm256_setzero_si256,
    _mm256_unpacklo_epi16,
    _mm256_unpackhi_epi16,
    _mm256_dpwssd_epi32,
    _mm256_srai_epi32,
    _mm256_packs_epi32,
    _mm256_adds_epi16,
    _mm256_subs_epi16,
    _mm256_mulhrs_epi16
);

buyuk_1d_avx512_yaprak!(
    x16,
    __m256i,
    false,
    _mm256_set1_epi16,
    _mm256_set1_epi32,
    _mm256_setzero_si256,
    _mm256_unpacklo_epi16,
    _mm256_unpackhi_epi16,
    _mm256_dpwssd_epi32,
    _mm256_srai_epi32,
    _mm256_packs_epi32,
    _mm256_adds_epi16,
    _mm256_subs_epi16,
    _mm256_mulhrs_epi16
);

buyuk_1d_avx512_yaprak!(
    z32,
    __m512i,
    false,
    _mm512_set1_epi16,
    _mm512_set1_epi32,
    _mm512_setzero_si512,
    _mm512_unpacklo_epi16,
    _mm512_unpackhi_epi16,
    _mm512_dpwssd_epi32,
    _mm512_srai_epi32,
    _mm512_packs_epi32,
    _mm512_adds_epi16,
    _mm512_subs_epi16,
    _mm512_mulhrs_epi16
);

// ---------------------------------------------------------------------------
// Yazmaç-değişkeni 1B ters DCT makroları (`sret` ve dizi `alloca` maliyeti = 0)
// ---------------------------------------------------------------------------

macro_rules! idct4_v {
    ($m:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident) => {{
        let (t1, t0) = $m::mulsub($c0, $c2, 2896, 2896);
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
        let t1 = t0;
        let t2 = $m::mul12(i1_v, 1567);
        let t3 = $m::mul12(i1_v, 3784);
        let $o0 = $m::adds(t0, t3);
        let $o1 = $m::adds(t1, t2);
        let $o2 = $m::subs(t1, t2);
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
        let (t5, t6) = $m::mulsub(t6_in, t5_in, 2896, 2896);
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
        $c0 = r0; $c1 = r1; $c2 = r2; $c3 = r3;
        $c4 = r4; $c5 = r5; $c6 = r6; $c7 = r7;
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

        let (t10_f, t13_f) = $m::mulsub(t13_2, t10_2, 2896, 2896);
        let (t11_f, t12_f) = $m::mulsub(t12a, t11a, 2896, 2896);

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
        $c0 = r0; $c1 = r1; $c2 = r2; $c3 = r3;
        $c4 = r4; $c5 = r5; $c6 = r6; $c7 = r7;
        $c8 = r8; $c9 = r9; $c10 = r10; $c11 = r11;
        $c12 = r12; $c13 = r13; $c14 = r14; $c15 = r15;
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

        let (t20, t27) = $m::mulsub(t27a, t20a, 2896, 2896);
        let (t21a, t26a) = $m::mulsub(t26, t21, 2896, 2896);
        let (t22, t25) = $m::mulsub(t25a, t22a, 2896, 2896);
        let (t23a, t24a) = $m::mulsub(t24, t23, 2896, 2896);

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

macro_rules! idct64_odd_core_v {
    ($m:ident,
     $i0:expr, $i1:expr, $i2:expr, $i3:expr, $i4:expr, $i5:expr, $i6:expr, $i7:expr,
     $i8:expr, $i9:expr, $i10:expr, $i11:expr, $i12:expr, $i13:expr, $i14:expr, $i15:expr,
     $i16:expr, $i17:expr, $i18:expr, $i19:expr, $i20:expr, $i21:expr, $i22:expr, $i23:expr,
     $i24:expr, $i25:expr, $i26:expr, $i27:expr, $i28:expr, $i29:expr, $i30:expr, $i31:expr
     => $lo0:ident, $lo1:ident, $lo2:ident, $lo3:ident, $lo4:ident, $lo5:ident, $lo6:ident, $lo7:ident,
        $lo8:ident, $lo9:ident, $lo10:ident, $lo11:ident, $lo12:ident, $lo13:ident, $lo14:ident, $lo15:ident;
        $hi0:ident, $hi1:ident, $hi2:ident, $hi3:ident, $hi4:ident, $hi5:ident, $hi6:ident, $hi7:ident,
        $hi8:ident, $hi9:ident, $hi10:ident, $hi11:ident, $hi12:ident, $hi13:ident, $hi14:ident, $hi15:ident) => {
        let (o0, o1, o2, o3, o4, o5, o6, o7) = ($i0, $i1, $i2, $i3, $i4, $i5, $i6, $i7);
        let (o8, o9, o10, o11, o12, o13, o14, o15) = ($i8, $i9, $i10, $i11, $i12, $i13, $i14, $i15);
        let (o16, o17, o18, o19, o20, o21, o22, o23) =
            ($i16, $i17, $i18, $i19, $i20, $i21, $i22, $i23);
        let (o24, o25, o26, o27, o28, o29, o30, o31) =
            ($i24, $i25, $i26, $i27, $i28, $i29, $i30, $i31);

        let mut t32 = $m::adds(o0, o1);
        let mut t33 = $m::subs(o0, o1);
        let mut t34 = $m::subs(o3, o2);
        let t35 = $m::adds(o3, o2);
        let t36 = $m::adds(o4, o5);
        let mut t37 = $m::subs(o4, o5);
        let t38 = $m::subs(o7, o6);
        let t39 = $m::adds(o7, o6);
        let t40 = $m::adds(o8, o9);
        let mut t41 = $m::subs(o8, o9);
        let mut t42 = $m::subs(o11, o10);
        let t43 = $m::adds(o11, o10);
        let t44 = $m::adds(o12, o13);
        let mut t45 = $m::subs(o12, o13);
        let t46 = $m::subs(o15, o14);
        let t47 = $m::adds(o15, o14);
        let t48 = $m::adds(o16, o17);
        let mut t49 = $m::subs(o16, o17);
        let mut t50 = $m::subs(o19, o18);
        let t51 = $m::adds(o19, o18);
        let t52 = $m::adds(o20, o21);
        let mut t53 = $m::subs(o20, o21);
        let mut t54 = $m::subs(o23, o22);
        let t55 = $m::adds(o23, o22);
        let t56 = $m::adds(o24, o25);
        let mut t57 = $m::subs(o24, o25);
        let mut t58 = $m::subs(o27, o26);
        let t59 = $m::adds(o27, o26);
        let t60 = $m::adds(o28, o29);
        let mut t61 = $m::subs(o28, o29);
        let mut t62 = $m::subs(o31, o30);
        let t63 = $m::adds(o31, o30);

        let (t33a, t62a) = $m::mulsub(t62, t33, 401, 4076);
        let (t34a, t61a) = $m::mulsub(t61, t34, -4076, 401);
        let (t37a, t58a) = $m::mulsub(t58, t37, 3166, 2598);
        let (t38a, t57a) = $m::mulsub(t57, t38, -2598, 3166);
        let (t41a, t54a) = $m::mulsub(t54, t41, 1931, 3612);
        let (t42a, t53a) = $m::mulsub(t53, t42, -3612, 1931);
        let (t45a, t50a) = $m::mulsub(t50, t45, 3920, 1189);
        let (t46a, t49a) = $m::mulsub(t49, t46, -1189, 3920);

        let t32a = $m::adds(t32, t35);
        t33 = $m::adds(t33a, t34a);
        t34 = $m::subs(t33a, t34a);
        let t35a = $m::subs(t32, t35);
        let t36a = $m::subs(t39, t36);
        t37 = $m::subs(t38a, t37a);
        let t38 = $m::adds(t38a, t37a);
        let t39a = $m::adds(t39, t36);
        let t40a = $m::adds(t40, t43);
        t41 = $m::adds(t41a, t42a);
        t42 = $m::subs(t41a, t42a);
        let t43a = $m::subs(t40, t43);
        let t44a = $m::subs(t47, t44);
        t45 = $m::subs(t46a, t45a);
        let t46 = $m::adds(t46a, t45a);
        let t47a = $m::adds(t47, t44);
        let t48a = $m::adds(t48, t51);
        t49 = $m::adds(t49a, t50a);
        t50 = $m::subs(t49a, t50a);
        let t51a = $m::subs(t48, t51);
        let t52a = $m::subs(t55, t52);
        t53 = $m::subs(t54a, t53a);
        t54 = $m::adds(t54a, t53a);
        let t55a = $m::adds(t55, t52);
        let t56a = $m::adds(t56, t59);
        t57 = $m::adds(t57a, t58a);
        t58 = $m::subs(t57a, t58a);
        let t59a = $m::subs(t56, t59);
        let t60a = $m::subs(t63, t60);
        t61 = $m::subs(t62a, t61a);
        t62 = $m::adds(t62a, t61a);
        let t63a = $m::adds(t63, t60);

        let (t34a, t61a) = $m::mulsub(t61, t34, 799, 4017);
        let (t35, t60) = $m::mulsub(t60a, t35a, 799, 4017);
        let (t36, t59) = $m::mulsub(t59a, t36a, -4017, 799);
        let (t37a, t58a) = $m::mulsub(t58, t37, -4017, 799);
        let (t42a, t53a) = $m::mulsub(t53, t42, 3406, 2276);
        let (t43, t52) = $m::mulsub(t52a, t43a, 3406, 2276);
        let (t44, t51) = $m::mulsub(t51a, t44a, -2276, 3406);
        let (t45a, t50a) = $m::mulsub(t50, t45, -2276, 3406);

        t32 = $m::adds(t32a, t39a);
        let t33a = $m::adds(t33, t38);
        t34 = $m::adds(t34a, t37a);
        let t35a = $m::adds(t35, t36);
        let t36a = $m::subs(t35, t36);
        t37 = $m::subs(t34a, t37a);
        let t38a = $m::subs(t33, t38);
        let t39 = $m::subs(t32a, t39a);
        let t40 = $m::subs(t47a, t40a);
        let t41a = $m::subs(t46, t41);
        t42 = $m::subs(t45a, t42a);
        let t43a = $m::subs(t44, t43);
        let t44a = $m::adds(t44, t43);
        t45 = $m::adds(t45a, t42a);
        let t46a = $m::adds(t46, t41);
        let t47 = $m::adds(t47a, t40a);
        let t48 = $m::adds(t48a, t55a);
        let t49a = $m::adds(t49, t54);
        t50 = $m::adds(t50a, t53a);
        let t51a = $m::adds(t51, t52);
        let t52a = $m::subs(t51, t52);
        t53 = $m::subs(t50a, t53a);
        let t54a = $m::subs(t49, t54);
        let t55 = $m::subs(t48a, t55a);
        let t56 = $m::subs(t63a, t56a);
        let t57a = $m::subs(t62, t57);
        t58 = $m::subs(t61a, t58a);
        let t59a = $m::subs(t60, t59);
        let t60a = $m::adds(t60, t59);
        t61 = $m::adds(t61a, t58a);
        let t62a = $m::adds(t62, t57);
        let t63 = $m::adds(t63a, t56a);

        let (t36, t59) = $m::mulsub(t59a, t36a, 1567, 3784);
        let (t37a, t58a) = $m::mulsub(t58, t37, 1567, 3784);
        let (t38, t57) = $m::mulsub(t57a, t38a, 1567, 3784);
        let (t39a, t56a) = $m::mulsub(t56, t39, 1567, 3784);
        let (t40a, t55a) = $m::mulsub(t55, t40, -3784, 1567);
        let (t41, t54) = $m::mulsub(t54a, t41a, -3784, 1567);
        let (t42a, t53a) = $m::mulsub(t53, t42, -3784, 1567);
        let (t43, t52) = $m::mulsub(t52a, t43a, -3784, 1567);

        let t32a = $m::adds(t32, t47);
        t33 = $m::adds(t33a, t46a);
        let t34a = $m::adds(t34, t45);
        let t35 = $m::adds(t35a, t44a);
        let t36a = $m::adds(t36, t43);
        t37 = $m::adds(t37a, t42a);
        let t38a = $m::adds(t38, t41);
        let t39 = $m::adds(t39a, t40a);
        let t40 = $m::subs(t39a, t40a);
        let t41a = $m::subs(t38, t41);
        t42 = $m::subs(t37a, t42a);
        let t43a = $m::subs(t36, t43);
        let t44 = $m::subs(t35a, t44a);
        let t45a = $m::subs(t34, t45);
        let t46 = $m::subs(t33a, t46a);
        let t47a = $m::subs(t32, t47);
        let t48a = $m::subs(t63, t48);
        t49 = $m::subs(t62a, t49a);
        let t50a = $m::subs(t61, t50);
        let t51 = $m::subs(t60a, t51a);
        let t52a = $m::subs(t59, t52);
        t53 = $m::subs(t58a, t53a);
        let t54a = $m::subs(t57, t54);
        let t55 = $m::subs(t56a, t55a);
        let t56 = $m::adds(t56a, t55a);
        let t57a = $m::adds(t57, t54);
        t58 = $m::adds(t58a, t53a);
        let t59a = $m::adds(t59, t52);
        let t60 = $m::adds(t60a, t51a);
        let t61a = $m::adds(t61, t50);
        t62 = $m::adds(t62a, t49a);
        let t63a = $m::adds(t63, t48);

        let (t40a, t55a) = $m::mulsub(t55, t40, 2896, 2896);
        let (t41, t54) = $m::mulsub(t54a, t41a, 2896, 2896);
        let (t42a, t53a) = $m::mulsub(t53, t42, 2896, 2896);
        let (t43, t52) = $m::mulsub(t52a, t43a, 2896, 2896);
        let (t44a, t51a) = $m::mulsub(t51, t44, 2896, 2896);
        let (t45, t50) = $m::mulsub(t50a, t45a, 2896, 2896);
        let (t46a, t49a) = $m::mulsub(t49, t46, 2896, 2896);
        let (t47, t48) = $m::mulsub(t48a, t47a, 2896, 2896);

        let ($lo0, $lo1, $lo2, $lo3, $lo4, $lo5, $lo6, $lo7) =
            (t63a, t62, t61a, t60, t59a, t58, t57a, t56);
        let ($lo8, $lo9, $lo10, $lo11, $lo12, $lo13, $lo14, $lo15) =
            (t55a, t54, t53a, t52, t51a, t50, t49a, t48);
        let ($hi0, $hi1, $hi2, $hi3, $hi4, $hi5, $hi6, $hi7) =
            (t47, t46a, t45, t44a, t43, t42a, t41, t40a);
        let ($hi8, $hi9, $hi10, $hi11, $hi12, $hi13, $hi14, $hi15) =
            (t39, t38a, t37, t36a, t35, t34a, t33, t32a);
    };
}

macro_rules! round_shift8_v {
    ($m:ident, $sh:literal, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident) => {{
        $c0 = $m::round_one::<$sh>($c0);
        $c1 = $m::round_one::<$sh>($c1);
        $c2 = $m::round_one::<$sh>($c2);
        $c3 = $m::round_one::<$sh>($c3);
        $c4 = $m::round_one::<$sh>($c4);
        $c5 = $m::round_one::<$sh>($c5);
        $c6 = $m::round_one::<$sh>($c6);
        $c7 = $m::round_one::<$sh>($c7);
    }};
}

macro_rules! round_shift16_v {
    ($m:ident, $sh:literal,
     $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident,
     $c8:ident, $c9:ident, $c10:ident, $c11:ident, $c12:ident, $c13:ident, $c14:ident, $c15:ident) => {{
        round_shift8_v!($m, $sh, $c0, $c1, $c2, $c3, $c4, $c5, $c6, $c7);
        round_shift8_v!($m, $sh, $c8, $c9, $c10, $c11, $c12, $c13, $c14, $c15);
    }};
}

macro_rules! round_p2_8_v {
    ($m:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident) => {{
        $c0 = $m::round_p2($c0);
        $c1 = $m::round_p2($c1);
        $c2 = $m::round_p2($c2);
        $c3 = $m::round_p2($c3);
        $c4 = $m::round_p2($c4);
        $c5 = $m::round_p2($c5);
        $c6 = $m::round_p2($c6);
        $c7 = $m::round_p2($c7);
    }};
}

macro_rules! round_p2_16_v {
    ($m:ident,
     $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident,
     $c8:ident, $c9:ident, $c10:ident, $c11:ident, $c12:ident, $c13:ident, $c14:ident, $c15:ident) => {{
        round_p2_8_v!($m, $c0, $c1, $c2, $c3, $c4, $c5, $c6, $c7);
        round_p2_8_v!($m, $c8, $c9, $c10, $c11, $c12, $c13, $c14, $c15);
    }};
}

// ---------------------------------------------------------------------------
// Yazmaç-değişkeni matris transpoz makroları (`i16`, YMM ve 512-bit ZMM)
// ---------------------------------------------------------------------------

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

macro_rules! transpose_8x8_zmm_v {
    ($r0:ident, $r1:ident, $r2:ident, $r3:ident, $r4:ident, $r5:ident, $r6:ident, $r7:ident) => {{
        let t0 = _mm512_unpacklo_epi16($r0, $r1);
        let t1 = _mm512_unpackhi_epi16($r0, $r1);
        let t2 = _mm512_unpacklo_epi16($r2, $r3);
        let t3 = _mm512_unpackhi_epi16($r2, $r3);
        let t4 = _mm512_unpacklo_epi16($r4, $r5);
        let t5 = _mm512_unpackhi_epi16($r4, $r5);
        let t6 = _mm512_unpacklo_epi16($r6, $r7);
        let t7 = _mm512_unpackhi_epi16($r6, $r7);

        let s0 = _mm512_unpacklo_epi32(t0, t2);
        let s1 = _mm512_unpackhi_epi32(t0, t2);
        let s2 = _mm512_unpacklo_epi32(t1, t3);
        let s3 = _mm512_unpackhi_epi32(t1, t3);
        let s4 = _mm512_unpacklo_epi32(t4, t6);
        let s5 = _mm512_unpackhi_epi32(t4, t6);
        let s6 = _mm512_unpacklo_epi32(t5, t7);
        let s7 = _mm512_unpackhi_epi32(t5, t7);

        $r0 = _mm512_unpacklo_epi64(s0, s4);
        $r1 = _mm512_unpackhi_epi64(s0, s4);
        $r2 = _mm512_unpacklo_epi64(s1, s5);
        $r3 = _mm512_unpackhi_epi64(s1, s5);
        $r4 = _mm512_unpacklo_epi64(s2, s6);
        $r5 = _mm512_unpackhi_epi64(s2, s6);
        $r6 = _mm512_unpacklo_epi64(s3, s7);
        $r7 = _mm512_unpackhi_epi64(s3, s7);
    }};
}

macro_rules! transpose_16x16_ymm_v {
    ($r0:ident, $r1:ident, $r2:ident, $r3:ident, $r4:ident, $r5:ident, $r6:ident, $r7:ident,
     $r8:ident, $r9:ident, $r10:ident, $r11:ident, $r12:ident, $r13:ident, $r14:ident, $r15:ident) => {{
        transpose_8x8_ymm_v!($r0, $r1, $r2, $r3, $r4, $r5, $r6, $r7);
        transpose_8x8_ymm_v!($r8, $r9, $r10, $r11, $r12, $r13, $r14, $r15);
        let (l0, l1, l2, l3, l4, l5, l6, l7) = ($r0, $r1, $r2, $r3, $r4, $r5, $r6, $r7);
        let (h0, h1, h2, h3, h4, h5, h6, h7) = ($r8, $r9, $r10, $r11, $r12, $r13, $r14, $r15);
        $r0 = _mm256_permute2x128_si256::<0x20>(l0, h0);
        $r1 = _mm256_permute2x128_si256::<0x20>(l1, h1);
        $r2 = _mm256_permute2x128_si256::<0x20>(l2, h2);
        $r3 = _mm256_permute2x128_si256::<0x20>(l3, h3);
        $r4 = _mm256_permute2x128_si256::<0x20>(l4, h4);
        $r5 = _mm256_permute2x128_si256::<0x20>(l5, h5);
        $r6 = _mm256_permute2x128_si256::<0x20>(l6, h6);
        $r7 = _mm256_permute2x128_si256::<0x20>(l7, h7);
        $r8 = _mm256_permute2x128_si256::<0x31>(l0, h0);
        $r9 = _mm256_permute2x128_si256::<0x31>(l1, h1);
        $r10 = _mm256_permute2x128_si256::<0x31>(l2, h2);
        $r11 = _mm256_permute2x128_si256::<0x31>(l3, h3);
        $r12 = _mm256_permute2x128_si256::<0x31>(l4, h4);
        $r13 = _mm256_permute2x128_si256::<0x31>(l5, h5);
        $r14 = _mm256_permute2x128_si256::<0x31>(l6, h6);
        $r15 = _mm256_permute2x128_si256::<0x31>(l7, h7);
    }};
}

macro_rules! transpose_8x16_to_split_ymm_v {
    ($c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident
     => $r0:ident, $r1:ident, $r2:ident, $r3:ident, $r4:ident, $r5:ident, $r6:ident, $r7:ident,
        $r8:ident, $r9:ident, $r10:ident, $r11:ident, $r12:ident, $r13:ident, $r14:ident, $r15:ident) => {
        transpose_8x8_ymm_v!($c0, $c1, $c2, $c3, $c4, $c5, $c6, $c7);
        let $r0 = _mm256_permute4x64_epi64::<0x50>($c0);
        let $r1 = _mm256_permute4x64_epi64::<0x50>($c1);
        let $r2 = _mm256_permute4x64_epi64::<0x50>($c2);
        let $r3 = _mm256_permute4x64_epi64::<0x50>($c3);
        let $r4 = _mm256_permute4x64_epi64::<0x50>($c4);
        let $r5 = _mm256_permute4x64_epi64::<0x50>($c5);
        let $r6 = _mm256_permute4x64_epi64::<0x50>($c6);
        let $r7 = _mm256_permute4x64_epi64::<0x50>($c7);
        let $r8 = _mm256_permute4x64_epi64::<0xFA>($c0);
        let $r9 = _mm256_permute4x64_epi64::<0xFA>($c1);
        let $r10 = _mm256_permute4x64_epi64::<0xFA>($c2);
        let $r11 = _mm256_permute4x64_epi64::<0xFA>($c3);
        let $r12 = _mm256_permute4x64_epi64::<0xFA>($c4);
        let $r13 = _mm256_permute4x64_epi64::<0xFA>($c5);
        let $r14 = _mm256_permute4x64_epi64::<0xFA>($c6);
        let $r15 = _mm256_permute4x64_epi64::<0xFA>($c7);
    };
}

macro_rules! transpose_16x8_split_to_ymm_v {
    ($c0:expr, $c1:expr, $c2:expr, $c3:expr, $c4:expr, $c5:expr, $c6:expr, $c7:expr,
     $c8:expr, $c9:expr, $c10:expr, $c11:expr, $c12:expr, $c13:expr, $c14:expr, $c15:expr
     => $r0:ident, $r1:ident, $r2:ident, $r3:ident, $r4:ident, $r5:ident, $r6:ident, $r7:ident) => {
        macro_rules! pack_pair {
            ($lo:expr, $hi:expr) => {{
                let a = _mm256_permute4x64_epi64::<0xD8>($lo);
                let b = _mm256_permute4x64_epi64::<0xD8>($hi);
                _mm256_permute2x128_si256::<0x20>(a, b)
            }};
        }
        let mut $r0 = pack_pair!($c0, $c8);
        let mut $r1 = pack_pair!($c1, $c9);
        let mut $r2 = pack_pair!($c2, $c10);
        let mut $r3 = pack_pair!($c3, $c11);
        let mut $r4 = pack_pair!($c4, $c12);
        let mut $r5 = pack_pair!($c5, $c13);
        let mut $r6 = pack_pair!($c6, $c14);
        let mut $r7 = pack_pair!($c7, $c15);
        transpose_8x8_ymm_v!($r0, $r1, $r2, $r3, $r4, $r5, $r6, $r7);
    };
}

// ---------------------------------------------------------------------------
// Sınır-güvenli piksel ekleme ve saklama yardımcıları (`0` gather/scatter)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_w8_cift(dst: &mut [u8], stride: usize, y: usize, r0: __m128i, r1: __m128i) {
    let d0 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, y * stride)));
    let d1 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, (y + 1) * stride)));
    let s0 = _mm_add_epi16(_mm_cvtepu8_epi16(d0), r0);
    let s1 = _mm_add_epi16(_mm_cvtepu8_epi16(d1), r1);
    let p = _mm_packus_epi16(s0, s1);
    store_u8x8(chunk_mut(dst, y * stride), p);
    store_u8x8(chunk_mut(dst, (y + 1) * stride), _mm_srli_si128::<8>(p));
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_w8_split_satir(dst: &mut [u8], off: usize, r: __m256i) {
    let d = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, off)));
    let del = _mm_unpacklo_epi64(_mm256_castsi256_si128(r), _mm256_extracti128_si256::<1>(r));
    let s = _mm_add_epi16(_mm_cvtepu8_epi16(d), del);
    store_u8x8(chunk_mut(dst, off), _mm_packus_epi16(s, s));
}

/// `8×32` 2. geçiş kelebek çiftini (`y = k` ve `y = 31 - k`) doğrudan hedefe yazar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_w8_split_kelebek32(dst: &mut [u8], stride: usize, k: usize, ek: __m256i, ok: __m256i) {
    let r4 = _mm256_set1_epi16(2048);
    ekle_yaz_w8_split_satir(
        dst,
        k * stride,
        _mm256_mulhrs_epi16(_mm256_adds_epi16(ek, ok), r4),
    );
    ekle_yaz_w8_split_satir(
        dst,
        (31 - k) * stride,
        _mm256_mulhrs_epi16(_mm256_subs_epi16(ek, ok), r4),
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_w16_satir(dst: &mut [u8], off: usize, r: __m256i) {
    let px = _mm256_cvtepu8_epi16(load_u8x16(chunk(dst, off)));
    let s = _mm256_add_epi16(px, r);
    let p = _mm_packus_epi16(_mm256_castsi256_si128(s), _mm256_extracti128_si256::<1>(s));
    store_u8x16(chunk_mut(dst, off), p);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_w16_cift(
    dst: &mut [u8],
    stride: usize,
    y: usize,
    x_off: usize,
    r0: __m256i,
    r1: __m256i,
) {
    let o0 = y * stride + x_off;
    let o1 = (y + 1) * stride + x_off;
    let px0 = _mm256_cvtepu8_epi16(load_u8x16(chunk(dst, o0)));
    let px1 = _mm256_cvtepu8_epi16(load_u8x16(chunk(dst, o1)));
    let s0 = _mm256_add_epi16(px0, r0);
    let s1 = _mm256_add_epi16(px1, r1);
    let p = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(s0, s1));
    store_u8x16(chunk_mut(dst, o0), _mm256_castsi256_si128(p));
    store_u8x16(chunk_mut(dst, o1), _mm256_extracti128_si256::<1>(p));
}

/// `w = 16, h = 32` 2. geçiş kelebek çiftini (`y = k` ve `y = 31 - k`) doğrudan hedefe yazar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_w16_kelebek32(
    dst: &mut [u8],
    stride: usize,
    x_off: usize,
    k: usize,
    ek: __m256i,
    ok: __m256i,
) {
    let r4 = _mm256_set1_epi16(2048);
    ekle_yaz_w16_satir(
        dst,
        k * stride + x_off,
        _mm256_mulhrs_epi16(_mm256_adds_epi16(ek, ok), r4),
    );
    ekle_yaz_w16_satir(
        dst,
        (31 - k) * stride + x_off,
        _mm256_mulhrs_epi16(_mm256_subs_epi16(ek, ok), r4),
    );
}

/// `h = 64` 2. geçiş 8 piksel genişlikli (`n8` `__m128i`) kelebek çiftini (`y` ve `63 - y`) hedefe yazar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_w8_kelebek64(
    dst: &mut [u8],
    stride: usize,
    x_off: usize,
    y: usize,
    ey: __m128i,
    oy: __m128i,
) {
    let r4 = _mm_set1_epi16(2048);
    let o0 = y * stride + x_off;
    let o1 = (63 - y) * stride + x_off;
    let d0 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, o0)));
    let d1 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, o1)));
    let del0 = _mm_mulhrs_epi16(_mm_adds_epi16(ey, oy), r4);
    let del1 = _mm_mulhrs_epi16(_mm_subs_epi16(ey, oy), r4);
    let s0 = _mm_add_epi16(_mm_cvtepu8_epi16(d0), del0);
    let s1 = _mm_add_epi16(_mm_cvtepu8_epi16(d1), del1);
    let p = _mm_packus_epi16(s0, s1);
    store_u8x8(chunk_mut(dst, o0), p);
    store_u8x8(chunk_mut(dst, o1), _mm_srli_si128::<8>(p));
}

/// Tek satırda 32 pikseli 512-bit ZMM `delta` (`32 × i16`) ile toplayıp `[0, 255]` kırparak yazar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn ekle_yaz_z32_satir(dst: &mut [u8], off: usize, delta: __m512i) {
    let px = _mm512_cvtepu8_epi16(load_u8x32(chunk(dst, off)));
    let s = _mm512_add_epi16(px, delta);
    let p = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(
        _mm512_castsi512_si256(s),
        _mm512_extracti64x4_epi64::<1>(s),
    ));
    store_u8x32(chunk_mut(dst, off), p);
}

macro_rules! ekle_yaz_z32_8_v {
    ($dst:expr, $stride:expr, $y_off:expr, $x_off:expr,
     $r0:expr, $r1:expr, $r2:expr, $r3:expr, $r4:expr, $r5:expr, $r6:expr, $r7:expr) => {{
        let y0 = $y_off;
        let x0 = $x_off;
        let st = $stride;
        ekle_yaz_z32_satir($dst, (y0) * st + x0, $r0);
        ekle_yaz_z32_satir($dst, (y0 + 1) * st + x0, $r1);
        ekle_yaz_z32_satir($dst, (y0 + 2) * st + x0, $r2);
        ekle_yaz_z32_satir($dst, (y0 + 3) * st + x0, $r3);
        ekle_yaz_z32_satir($dst, (y0 + 4) * st + x0, $r4);
        ekle_yaz_z32_satir($dst, (y0 + 5) * st + x0, $r5);
        ekle_yaz_z32_satir($dst, (y0 + 6) * st + x0, $r6);
        ekle_yaz_z32_satir($dst, (y0 + 7) * st + x0, $r7);
    }};
}

/// DC-only hızlı yolu (`txtp == DCT_DCT && eob == 0`): `W = 64` için 512-bit `load_u8x64`/`store_u8x64` kullanır.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn dc_only<const W: usize, const H: usize, const SHIFT: u32, const RECT2: bool>(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32; 1024],
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
    let add8 = dc.clamp(0, 255) as u8 as i8;
    let sub8 = (-dc).clamp(0, 255) as u8 as i8;

    if W == 8 {
        let va = _mm_set1_epi8(add8);
        let vs = _mm_set1_epi8(sub8);
        let mut y = 0;
        while y < H {
            let d0 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, y * stride)));
            let d1 = _mm_cvtsi64_si128(i64::from_le_bytes(*chunk::<u8, 8>(dst, (y + 1) * stride)));
            let p = _mm_subs_epu8(_mm_adds_epu8(_mm_unpacklo_epi64(d0, d1), va), vs);
            store_u8x8(chunk_mut(dst, y * stride), p);
            store_u8x8(chunk_mut(dst, (y + 1) * stride), _mm_srli_si128::<8>(p));
            y += 2;
        }
    } else if W == 16 {
        let va = _mm_set1_epi8(add8);
        let vs = _mm_set1_epi8(sub8);
        for y in 0..H {
            let off = y * stride;
            let d = load_u8x16(chunk(dst, off));
            store_u8x16(chunk_mut(dst, off), _mm_subs_epu8(_mm_adds_epu8(d, va), vs));
        }
    } else if W == 32 {
        let va = _mm256_set1_epi8(add8);
        let vs = _mm256_set1_epi8(sub8);
        for y in 0..H {
            let off = y * stride;
            let d = load_u8x32(chunk(dst, off));
            store_u8x32(
                chunk_mut(dst, off),
                _mm256_subs_epu8(_mm256_adds_epu8(d, va), vs),
            );
        }
    } else {
        let va = _mm512_set1_epi8(add8);
        let vs = _mm512_set1_epi8(sub8);
        for y in 0..H {
            let off = y * stride;
            let d = load_u8x64(chunk(dst, off));
            store_u8x64(
                chunk_mut(dst, off),
                _mm512_subs_epu8(_mm512_adds_epu8(d, va), vs),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Katsayı tamponu aşamalama ve üst seviye tek-indis kelebek makroları
// ---------------------------------------------------------------------------

macro_rules! sakla_ymm16 {
    ($buf:expr, $base:expr, $step:expr,
     $v0:expr, $v1:expr, $v2:expr, $v3:expr, $v4:expr, $v5:expr, $v6:expr, $v7:expr,
     $v8:expr, $v9:expr, $v10:expr, $v11:expr, $v12:expr, $v13:expr, $v14:expr, $v15:expr) => {{
        let b = $base;
        let st = $step;
        store_i32x8(chunk_mut($buf, b), $v0);
        store_i32x8(chunk_mut($buf, b + st), $v1);
        store_i32x8(chunk_mut($buf, b + 2 * st), $v2);
        store_i32x8(chunk_mut($buf, b + 3 * st), $v3);
        store_i32x8(chunk_mut($buf, b + 4 * st), $v4);
        store_i32x8(chunk_mut($buf, b + 5 * st), $v5);
        store_i32x8(chunk_mut($buf, b + 6 * st), $v6);
        store_i32x8(chunk_mut($buf, b + 7 * st), $v7);
        store_i32x8(chunk_mut($buf, b + 8 * st), $v8);
        store_i32x8(chunk_mut($buf, b + 9 * st), $v9);
        store_i32x8(chunk_mut($buf, b + 10 * st), $v10);
        store_i32x8(chunk_mut($buf, b + 11 * st), $v11);
        store_i32x8(chunk_mut($buf, b + 12 * st), $v12);
        store_i32x8(chunk_mut($buf, b + 13 * st), $v13);
        store_i32x8(chunk_mut($buf, b + 14 * st), $v14);
        store_i32x8(chunk_mut($buf, b + 15 * st), $v15);
    }};
}

macro_rules! yukle_ymm16 {
    ($buf:expr, $base:expr, $step:expr
     => $v0:ident, $v1:ident, $v2:ident, $v3:ident, $v4:ident, $v5:ident, $v6:ident, $v7:ident,
        $v8:ident, $v9:ident, $v10:ident, $v11:ident, $v12:ident, $v13:ident, $v14:ident, $v15:ident) => {
        let b_ld = $base;
        let st_ld = $step;
        #[allow(unused_mut)]
        let (mut $v0, mut $v1, mut $v2, mut $v3, mut $v4, mut $v5, mut $v6, mut $v7) = (
            yukle_i32x8(chunk($buf, b_ld)),
            yukle_i32x8(chunk($buf, b_ld + st_ld)),
            yukle_i32x8(chunk($buf, b_ld + 2 * st_ld)),
            yukle_i32x8(chunk($buf, b_ld + 3 * st_ld)),
            yukle_i32x8(chunk($buf, b_ld + 4 * st_ld)),
            yukle_i32x8(chunk($buf, b_ld + 5 * st_ld)),
            yukle_i32x8(chunk($buf, b_ld + 6 * st_ld)),
            yukle_i32x8(chunk($buf, b_ld + 7 * st_ld)),
        );
        #[allow(unused_mut)]
        let (mut $v8, mut $v9, mut $v10, mut $v11, mut $v12, mut $v13, mut $v14, mut $v15) = (
            yukle_i32x8(chunk($buf, b_ld + 8 * st_ld)),
            yukle_i32x8(chunk($buf, b_ld + 9 * st_ld)),
            yukle_i32x8(chunk($buf, b_ld + 10 * st_ld)),
            yukle_i32x8(chunk($buf, b_ld + 11 * st_ld)),
            yukle_i32x8(chunk($buf, b_ld + 12 * st_ld)),
            yukle_i32x8(chunk($buf, b_ld + 13 * st_ld)),
            yukle_i32x8(chunk($buf, b_ld + 14 * st_ld)),
            yukle_i32x8(chunk($buf, b_ld + 15 * st_ld)),
        );
    };
}

macro_rules! sakla_zmm16 {
    ($buf:expr, $base:expr, $step:expr,
     $v0:expr, $v1:expr, $v2:expr, $v3:expr, $v4:expr, $v5:expr, $v6:expr, $v7:expr,
     $v8:expr, $v9:expr, $v10:expr, $v11:expr, $v12:expr, $v13:expr, $v14:expr, $v15:expr) => {{
        let b = $base;
        let st = $step;
        sakla_zmm(chunk_mut($buf, b), $v0);
        sakla_zmm(chunk_mut($buf, b + st), $v1);
        sakla_zmm(chunk_mut($buf, b + 2 * st), $v2);
        sakla_zmm(chunk_mut($buf, b + 3 * st), $v3);
        sakla_zmm(chunk_mut($buf, b + 4 * st), $v4);
        sakla_zmm(chunk_mut($buf, b + 5 * st), $v5);
        sakla_zmm(chunk_mut($buf, b + 6 * st), $v6);
        sakla_zmm(chunk_mut($buf, b + 7 * st), $v7);
        sakla_zmm(chunk_mut($buf, b + 8 * st), $v8);
        sakla_zmm(chunk_mut($buf, b + 9 * st), $v9);
        sakla_zmm(chunk_mut($buf, b + 10 * st), $v10);
        sakla_zmm(chunk_mut($buf, b + 11 * st), $v11);
        sakla_zmm(chunk_mut($buf, b + 12 * st), $v12);
        sakla_zmm(chunk_mut($buf, b + 13 * st), $v13);
        sakla_zmm(chunk_mut($buf, b + 14 * st), $v14);
        sakla_zmm(chunk_mut($buf, b + 15 * st), $v15);
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

macro_rules! idct64_odd_full_v {
    ($m:ident,
     $p0:expr, $p1:expr, $p2:expr, $p3:expr, $p4:expr, $p5:expr, $p6:expr, $p7:expr,
     $p8:expr, $p9:expr, $p10:expr, $p11:expr, $p12:expr, $p13:expr, $p14:expr, $p15:expr
     => $lo0:ident, $lo1:ident, $lo2:ident, $lo3:ident, $lo4:ident, $lo5:ident, $lo6:ident, $lo7:ident,
        $lo8:ident, $lo9:ident, $lo10:ident, $lo11:ident, $lo12:ident, $lo13:ident, $lo14:ident, $lo15:ident;
        $hi0:ident, $hi1:ident, $hi2:ident, $hi3:ident, $hi4:ident, $hi5:ident, $hi6:ident, $hi7:ident,
        $hi8:ident, $hi9:ident, $hi10:ident, $hi11:ident, $hi12:ident, $hi13:ident, $hi14:ident, $hi15:ident) => {
        let (q0, q1, q2, q3, q4, q5, q6, q7) = ($p0, $p1, $p2, $p3, $p4, $p5, $p6, $p7);
        let (q8, q9, q10, q11, q12, q13, q14, q15) =
            ($p8, $p9, $p10, $p11, $p12, $p13, $p14, $p15);
        idct64_odd_core_v!(
            $m,
            $m::mul12(q0, 101), $m::mul12(q15, -2824), $m::mul12(q8, 1660), $m::mul12(q7, -1474),
            $m::mul12(q4, 897), $m::mul12(q11, -2191), $m::mul12(q12, 2359), $m::mul12(q3, -700),
            $m::mul12(q2, 501), $m::mul12(q13, -2520), $m::mul12(q10, 2019), $m::mul12(q5, -1092),
            $m::mul12(q6, 1285), $m::mul12(q9, -1842), $m::mul12(q14, 2675), $m::mul12(q1, -301),
            $m::mul12(q1, 4085), $m::mul12(q14, 3102), $m::mul12(q9, 3659), $m::mul12(q6, 3889),
            $m::mul12(q5, 3948), $m::mul12(q10, 3564), $m::mul12(q13, 3229), $m::mul12(q2, 4065),
            $m::mul12(q3, 4036), $m::mul12(q12, 3349), $m::mul12(q11, 3461), $m::mul12(q4, 3996),
            $m::mul12(q7, 3822), $m::mul12(q8, 3745), $m::mul12(q15, 2967), $m::mul12(q0, 4095)
            => $lo0, $lo1, $lo2, $lo3, $lo4, $lo5, $lo6, $lo7,
               $lo8, $lo9, $lo10, $lo11, $lo12, $lo13, $lo14, $lo15;
               $hi0, $hi1, $hi2, $hi3, $hi4, $hi5, $hi6, $hi7,
               $hi8, $hi9, $hi10, $hi11, $hi12, $hi13, $hi14, $hi15
        );
    };
}

macro_rules! idct64_odd_qtr_v {
    ($m:ident, $p0:expr, $p1:expr, $p2:expr, $p3:expr, $p4:expr, $p5:expr, $p6:expr, $p7:expr
     => $lo0:ident, $lo1:ident, $lo2:ident, $lo3:ident, $lo4:ident, $lo5:ident, $lo6:ident, $lo7:ident,
        $lo8:ident, $lo9:ident, $lo10:ident, $lo11:ident, $lo12:ident, $lo13:ident, $lo14:ident, $lo15:ident;
        $hi0:ident, $hi1:ident, $hi2:ident, $hi3:ident, $hi4:ident, $hi5:ident, $hi6:ident, $hi7:ident,
        $hi8:ident, $hi9:ident, $hi10:ident, $hi11:ident, $hi12:ident, $hi13:ident, $hi14:ident, $hi15:ident) => {
        let (q0, q1, q2, q3, q4, q5, q6, q7) = ($p0, $p1, $p2, $p3, $p4, $p5, $p6, $p7);
        let z = $m::zero();
        idct64_odd_core_v!(
            $m,
            $m::mul12(q0, 101), z, z, $m::mul12(q7, -1474),
            $m::mul12(q4, 897), z, z, $m::mul12(q3, -700),
            $m::mul12(q2, 501), z, z, $m::mul12(q5, -1092),
            $m::mul12(q6, 1285), z, z, $m::mul12(q1, -301),
            $m::mul12(q1, 4085), z, z, $m::mul12(q6, 3889),
            $m::mul12(q5, 3948), z, z, $m::mul12(q2, 4065),
            $m::mul12(q3, 4036), z, z, $m::mul12(q4, 3996),
            $m::mul12(q7, 3822), z, z, $m::mul12(q0, 4095)
            => $lo0, $lo1, $lo2, $lo3, $lo4, $lo5, $lo6, $lo7,
               $lo8, $lo9, $lo10, $lo11, $lo12, $lo13, $lo14, $lo15;
               $hi0, $hi1, $hi2, $hi3, $hi4, $hi5, $hi6, $hi7,
               $hi8, $hi9, $hi10, $hi11, $hi12, $hi13, $hi14, $hi15
        );
    };
}

macro_rules! idct64_odd_eighth_v {
    ($m:ident, $p0:expr, $p1:expr, $p2:expr, $p3:expr
     => $lo0:ident, $lo1:ident, $lo2:ident, $lo3:ident, $lo4:ident, $lo5:ident, $lo6:ident, $lo7:ident,
        $lo8:ident, $lo9:ident, $lo10:ident, $lo11:ident, $lo12:ident, $lo13:ident, $lo14:ident, $lo15:ident;
        $hi0:ident, $hi1:ident, $hi2:ident, $hi3:ident, $hi4:ident, $hi5:ident, $hi6:ident, $hi7:ident,
        $hi8:ident, $hi9:ident, $hi10:ident, $hi11:ident, $hi12:ident, $hi13:ident, $hi14:ident, $hi15:ident) => {
        let (q0, q1, q2, q3) = ($p0, $p1, $p2, $p3);
        let z = $m::zero();
        idct64_odd_core_v!(
            $m,
            $m::mul12(q0, 101), z, z, z,
            z, z, z, $m::mul12(q3, -700),
            $m::mul12(q2, 501), z, z, z,
            z, z, z, $m::mul12(q1, -301),
            $m::mul12(q1, 4085), z, z, z,
            z, z, z, $m::mul12(q2, 4065),
            $m::mul12(q3, 4036), z, z, z,
            z, z, z, $m::mul12(q0, 4095)
            => $lo0, $lo1, $lo2, $lo3, $lo4, $lo5, $lo6, $lo7,
               $lo8, $lo9, $lo10, $lo11, $lo12, $lo13, $lo14, $lo15;
               $hi0, $hi1, $hi2, $hi3, $hi4, $hi5, $hi6, $hi7,
               $hi8, $hi9, $hi10, $hi11, $hi12, $hi13, $hi14, $hi15
        );
    };
}

// ---------------------------------------------------------------------------
// Boyut × tür bazında büyük 2B AVX512ICL çekirdek gövdeleri (`8x32` .. `32x16`)
// ---------------------------------------------------------------------------

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_8x32_qtr(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    let mut c0 = yukle_i16x8(chunk(coeff, 0));
    let mut c1 = yukle_i16x8(chunk(coeff, 32));
    let mut c2 = yukle_i16x8(chunk(coeff, 64));
    let mut c3 = yukle_i16x8(chunk(coeff, 96));
    let mut c4 = yukle_i16x8(chunk(coeff, 128));
    let mut c5 = yukle_i16x8(chunk(coeff, 160));
    let mut c6 = yukle_i16x8(chunk(coeff, 192));
    let mut c7 = yukle_i16x8(chunk(coeff, 224));
    idct8_v!(x8, c0, c1, c2, c3, c4, c5, c6, c7);
    round_shift8_v!(x8, 2, c0, c1, c2, c3, c4, c5, c6, c7);
    transpose_8x8_split_ymm_v!(c0, c1, c2, c3, c4, c5, c6, c7);
    idct16_qtr_v!(x8, c0, c2, c4, c6 => e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15);
    sakla_ymm16!(
        coeff, 0, 8, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    );
    idct32_qtr_odd_v!(x8, c1, c3, c5, c7 => o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15);
    macro_rules! w8p {
        ($k:literal, $ok:expr) => {
            ekle_yaz_w8_split_kelebek32(dst, stride, $k, yukle_i32x8(chunk(coeff, $k * 8)), $ok)
        };
    }
    w8p!(0, o0);
    w8p!(1, o1);
    w8p!(2, o2);
    w8p!(3, o3);
    w8p!(4, o4);
    w8p!(5, o5);
    w8p!(6, o6);
    w8p!(7, o7);
    w8p!(8, o8);
    w8p!(9, o9);
    w8p!(10, o10);
    w8p!(11, o11);
    w8p!(12, o12);
    w8p!(13, o13);
    w8p!(14, o14);
    w8p!(15, o15);
    sifirla_katsayi::<256>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_8x32_half(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    let mut c0 = yukle_i16x16(chunk(coeff, 0));
    let mut c1 = yukle_i16x16(chunk(coeff, 32));
    let mut c2 = yukle_i16x16(chunk(coeff, 64));
    let mut c3 = yukle_i16x16(chunk(coeff, 96));
    let mut c4 = yukle_i16x16(chunk(coeff, 128));
    let mut c5 = yukle_i16x16(chunk(coeff, 160));
    let mut c6 = yukle_i16x16(chunk(coeff, 192));
    let mut c7 = yukle_i16x16(chunk(coeff, 224));
    idct8_v!(x16, c0, c1, c2, c3, c4, c5, c6, c7);
    round_shift8_v!(x16, 2, c0, c1, c2, c3, c4, c5, c6, c7);
    transpose_8x16_to_split_ymm_v!(c0, c1, c2, c3, c4, c5, c6, c7
        => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15);
    idct16_half_v!(x8, r0, r2, r4, r6, r8, r10, r12, r14
        => e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15);
    sakla_ymm16!(
        coeff, 0, 8, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    );
    idct32_half_odd_v!(x8, r1, r3, r5, r7, r9, r11, r13, r15
        => o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15);
    macro_rules! w8p {
        ($k:literal, $ok:expr) => {
            ekle_yaz_w8_split_kelebek32(dst, stride, $k, yukle_i32x8(chunk(coeff, $k * 8)), $ok)
        };
    }
    w8p!(0, o0);
    w8p!(1, o1);
    w8p!(2, o2);
    w8p!(3, o3);
    w8p!(4, o4);
    w8p!(5, o5);
    w8p!(6, o6);
    w8p!(7, o7);
    w8p!(8, o8);
    w8p!(9, o9);
    w8p!(10, o10);
    w8p!(11, o11);
    w8p!(12, o12);
    w8p!(13, o13);
    w8p!(14, o14);
    w8p!(15, o15);
    sifirla_katsayi::<256>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_8x32_full(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    let mut c0 = yukle_z32(chunk(coeff, 0));
    let mut c1 = yukle_z32(chunk(coeff, 32));
    let mut c2 = yukle_z32(chunk(coeff, 64));
    let mut c3 = yukle_z32(chunk(coeff, 96));
    let mut c4 = yukle_z32(chunk(coeff, 128));
    let mut c5 = yukle_z32(chunk(coeff, 160));
    let mut c6 = yukle_z32(chunk(coeff, 192));
    let mut c7 = yukle_z32(chunk(coeff, 224));
    idct8_v!(z32, c0, c1, c2, c3, c4, c5, c6, c7);
    round_shift8_v!(z32, 2, c0, c1, c2, c3, c4, c5, c6, c7);

    let (mut t0, mut t1, mut t2, mut t3, mut t4, mut t5, mut t6, mut t7) = (
        _mm512_castsi512_si256(c0),
        _mm512_castsi512_si256(c1),
        _mm512_castsi512_si256(c2),
        _mm512_castsi512_si256(c3),
        _mm512_castsi512_si256(c4),
        _mm512_castsi512_si256(c5),
        _mm512_castsi512_si256(c6),
        _mm512_castsi512_si256(c7),
    );
    let (mut b0, mut b1, mut b2, mut b3, mut b4, mut b5, mut b6, mut b7) = (
        _mm512_extracti64x4_epi64::<1>(c0),
        _mm512_extracti64x4_epi64::<1>(c1),
        _mm512_extracti64x4_epi64::<1>(c2),
        _mm512_extracti64x4_epi64::<1>(c3),
        _mm512_extracti64x4_epi64::<1>(c4),
        _mm512_extracti64x4_epi64::<1>(c5),
        _mm512_extracti64x4_epi64::<1>(c6),
        _mm512_extracti64x4_epi64::<1>(c7),
    );
    transpose_8x16_to_split_ymm_v!(t0, t1, t2, t3, t4, t5, t6, t7
        => a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15);
    transpose_8x16_to_split_ymm_v!(b0, b1, b2, b3, b4, b5, b6, b7
        => d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15);

    // Tek satırları `coeff[128..256]` içine sakla, çift satırların `idct16`'sını `coeff[0..128]` içine yaz.
    sakla_ymm16!(
        coeff, 128, 8, a1, a3, a5, a7, a9, a11, a13, a15, d1, d3, d5, d7, d9, d11, d13, d15
    );
    let (mut e0, mut e1, mut e2, mut e3, mut e4, mut e5, mut e6, mut e7) =
        (a0, a2, a4, a6, a8, a10, a12, a14);
    let (mut e8, mut e9, mut e10, mut e11, mut e12, mut e13, mut e14, mut e15) =
        (d0, d2, d4, d6, d8, d10, d12, d14);
    idct16_v!(
        x8, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    );
    sakla_ymm16!(
        coeff, 0, 8, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    );

    yukle_ymm16!(coeff, 128, 8
        => p1, p3, p5, p7, p9, p11, p13, p15, q1, q3, q5, q7, q9, q11, q13, q15);
    idct32_full_odd_v!(x8,
        p1, p3, p5, p7, p9, p11, p13, p15,
        q1, q3, q5, q7, q9, q11, q13, q15
        => o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15);
    macro_rules! w8p {
        ($k:literal, $ok:expr) => {
            ekle_yaz_w8_split_kelebek32(dst, stride, $k, yukle_i32x8(chunk(coeff, $k * 8)), $ok)
        };
    }
    w8p!(0, o0);
    w8p!(1, o1);
    w8p!(2, o2);
    w8p!(3, o3);
    w8p!(4, o4);
    w8p!(5, o5);
    w8p!(6, o6);
    w8p!(7, o7);
    w8p!(8, o8);
    w8p!(9, o9);
    w8p!(10, o10);
    w8p!(11, o11);
    w8p!(12, o12);
    w8p!(13, o13);
    w8p!(14, o14);
    w8p!(15, o15);
    sifirla_katsayi::<256>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn dct_dct_8x32(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    if eob <= 42 {
        dct_dct_8x32_qtr(dst, stride, coeff);
    } else if eob <= 106 {
        dct_dct_8x32_half(dst, stride, coeff);
    } else {
        dct_dct_8x32_full(dst, stride, coeff);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn idtx_8x32(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    if eob <= 106 {
        let r1 = _mm256_set1_epi16(16384);
        let r2 = _mm256_set1_epi16(8192);
        macro_rules! sc {
            ($off:expr) => {
                _mm256_mulhrs_epi16(
                    _mm256_mulhrs_epi16(yukle_i16x16(chunk(coeff, $off)), r1),
                    r2,
                )
            };
        }
        let (mut t0, mut t1, mut t2, mut t3, mut t4, mut t5, mut t6, mut t7) = (
            sc!(0),
            sc!(32),
            sc!(64),
            sc!(96),
            sc!(128),
            sc!(160),
            sc!(192),
            sc!(224),
        );
        sifirla_katsayi::<256>(coeff);
        transpose_8x8_ymm_v!(t0, t1, t2, t3, t4, t5, t6, t7);
        ekle_yaz_w8_cift(
            dst,
            stride,
            0,
            _mm256_castsi256_si128(t0),
            _mm256_castsi256_si128(t1),
        );
        ekle_yaz_w8_cift(
            dst,
            stride,
            2,
            _mm256_castsi256_si128(t2),
            _mm256_castsi256_si128(t3),
        );
        ekle_yaz_w8_cift(
            dst,
            stride,
            4,
            _mm256_castsi256_si128(t4),
            _mm256_castsi256_si128(t5),
        );
        ekle_yaz_w8_cift(
            dst,
            stride,
            6,
            _mm256_castsi256_si128(t6),
            _mm256_castsi256_si128(t7),
        );
        ekle_yaz_w8_cift(
            dst,
            stride,
            8,
            _mm256_extracti128_si256::<1>(t0),
            _mm256_extracti128_si256::<1>(t1),
        );
        ekle_yaz_w8_cift(
            dst,
            stride,
            10,
            _mm256_extracti128_si256::<1>(t2),
            _mm256_extracti128_si256::<1>(t3),
        );
        ekle_yaz_w8_cift(
            dst,
            stride,
            12,
            _mm256_extracti128_si256::<1>(t4),
            _mm256_extracti128_si256::<1>(t5),
        );
        ekle_yaz_w8_cift(
            dst,
            stride,
            14,
            _mm256_extracti128_si256::<1>(t6),
            _mm256_extracti128_si256::<1>(t7),
        );
        return;
    }

    let r1 = _mm512_set1_epi16(16384);
    let r2 = _mm512_set1_epi16(8192);
    macro_rules! scz {
        ($off:expr) => {
            _mm512_mulhrs_epi16(_mm512_mulhrs_epi16(yukle_z32(chunk(coeff, $off)), r1), r2)
        };
    }
    let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5, mut c6, mut c7) = (
        scz!(0),
        scz!(32),
        scz!(64),
        scz!(96),
        scz!(128),
        scz!(160),
        scz!(192),
        scz!(224),
    );
    sifirla_katsayi::<256>(coeff);
    transpose_8x8_zmm_v!(c0, c1, c2, c3, c4, c5, c6, c7);
    macro_rules! w8z {
        ($k:literal, $a:ident, $b:ident) => {{
            ekle_yaz_w8_cift(
                dst,
                stride,
                $k,
                _mm512_castsi512_si128($a),
                _mm512_castsi512_si128($b),
            );
            ekle_yaz_w8_cift(
                dst,
                stride,
                8 + $k,
                _mm512_extracti32x4_epi32::<1>($a),
                _mm512_extracti32x4_epi32::<1>($b),
            );
            ekle_yaz_w8_cift(
                dst,
                stride,
                16 + $k,
                _mm512_extracti32x4_epi32::<2>($a),
                _mm512_extracti32x4_epi32::<2>($b),
            );
            ekle_yaz_w8_cift(
                dst,
                stride,
                24 + $k,
                _mm512_extracti32x4_epi32::<3>($a),
                _mm512_extracti32x4_epi32::<3>($b),
            );
        }};
    }
    w8z!(0, c0, c1);
    w8z!(2, c2, c3);
    w8z!(4, c4, c5);
    w8z!(6, c6, c7);
}

/// `even` (`ebase + k*estep`) ile `odd` (`o0..o15`) çiftlerini `SHIFT` ile yuvarlayıp `coeff[0..256]` içine yazar.
macro_rules! kelebek_sakla_lo_hi16 {
    ($m:ident, $sh:literal, $coeff:expr, $ebase:expr, $estep:expr,
     $o0:expr, $o1:expr, $o2:expr, $o3:expr, $o4:expr, $o5:expr, $o6:expr, $o7:expr,
     $o8:expr, $o9:expr, $o10:expr, $o11:expr, $o12:expr, $o13:expr, $o14:expr, $o15:expr) => {{
        let eb = $ebase;
        let es = $estep;
        macro_rules! stp {
            ($k:literal, $ok:expr) => {{
                let ek = yukle_i32x8(chunk($coeff, eb + $k * es));
                let ok_v = $ok;
                store_i32x8(
                    chunk_mut($coeff, $k * 8),
                    $m::round_one::<$sh>($m::adds(ek, ok_v)),
                );
                store_i32x8(
                    chunk_mut($coeff, 128 + (15 - $k) * 8),
                    $m::round_one::<$sh>($m::subs(ek, ok_v)),
                );
            }};
        }
        stp!(0, $o0);
        stp!(1, $o1);
        stp!(2, $o2);
        stp!(3, $o3);
        stp!(4, $o4);
        stp!(5, $o5);
        stp!(6, $o6);
        stp!(7, $o7);
        stp!(8, $o8);
        stp!(9, $o9);
        stp!(10, $o10);
        stp!(11, $o11);
        stp!(12, $o12);
        stp!(13, $o13);
        stp!(14, $o14);
        stp!(15, $o15);
    }};
}

/// `32×8` 1. geçiş (`coeff[0..128]` `lo` ve `coeff[128..256]` `hi`) çıktısını 2. geçiş ZMM `idct8` ile yazar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_32x8_bitir(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    yukle_ymm16!(coeff, 0, 8 => c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15);
    transpose_16x8_split_to_ymm_v!(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15
        => lo0, lo1, lo2, lo3, lo4, lo5, lo6, lo7);
    yukle_ymm16!(coeff, 128, 8 => h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15);
    transpose_16x8_split_to_ymm_v!(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15
        => hi0, hi1, hi2, hi3, hi4, hi5, hi6, hi7);

    let mut r0 = birlestir_zmm(lo0, hi0);
    let mut r1 = birlestir_zmm(lo1, hi1);
    let mut r2 = birlestir_zmm(lo2, hi2);
    let mut r3 = birlestir_zmm(lo3, hi3);
    let mut r4 = birlestir_zmm(lo4, hi4);
    let mut r5 = birlestir_zmm(lo5, hi5);
    let mut r6 = birlestir_zmm(lo6, hi6);
    let mut r7 = birlestir_zmm(lo7, hi7);
    idct8_v!(z32, r0, r1, r2, r3, r4, r5, r6, r7);
    round_p2_8_v!(z32, r0, r1, r2, r3, r4, r5, r6, r7);
    ekle_yaz_z32_8_v!(dst, stride, 0, 0, r0, r1, r2, r3, r4, r5, r6, r7);
    sifirla_katsayi::<256>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_32x8_qtr(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    let q0 = yukle_i16x8(chunk(coeff, 0));
    let q1 = yukle_i16x8(chunk(coeff, 8));
    let q2 = yukle_i16x8(chunk(coeff, 16));
    let q3 = yukle_i16x8(chunk(coeff, 24));
    let q4 = yukle_i16x8(chunk(coeff, 32));
    let q5 = yukle_i16x8(chunk(coeff, 40));
    let q6 = yukle_i16x8(chunk(coeff, 48));
    let q7 = yukle_i16x8(chunk(coeff, 56));
    idct16_qtr_v!(x8, q0, q2, q4, q6 => e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15);
    sakla_ymm16!(
        coeff, 0, 8, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    );
    idct32_qtr_odd_v!(x8, q1, q3, q5, q7 => o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15);
    kelebek_sakla_lo_hi16!(
        x8, 2, coeff, 0, 8, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15
    );
    dct_dct_32x8_bitir(dst, stride, coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_32x8_half(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    let o_in1 = yukle_i16x8(chunk(coeff, 8));
    let o_in3 = yukle_i16x8(chunk(coeff, 24));
    let o_in5 = yukle_i16x8(chunk(coeff, 40));
    let o_in7 = yukle_i16x8(chunk(coeff, 56));
    let o_in9 = yukle_i16x8(chunk(coeff, 72));
    let o_in11 = yukle_i16x8(chunk(coeff, 88));
    let o_in13 = yukle_i16x8(chunk(coeff, 104));
    let o_in15 = yukle_i16x8(chunk(coeff, 120));
    idct16_half_v!(x8,
        yukle_i16x8(chunk(coeff, 0)), yukle_i16x8(chunk(coeff, 16)),
        yukle_i16x8(chunk(coeff, 32)), yukle_i16x8(chunk(coeff, 48)),
        yukle_i16x8(chunk(coeff, 64)), yukle_i16x8(chunk(coeff, 80)),
        yukle_i16x8(chunk(coeff, 96)), yukle_i16x8(chunk(coeff, 112))
        => e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15);
    sakla_ymm16!(
        coeff, 0, 8, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    );
    idct32_half_odd_v!(x8, o_in1, o_in3, o_in5, o_in7, o_in9, o_in11, o_in13, o_in15
        => o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15);
    kelebek_sakla_lo_hi16!(
        x8, 2, coeff, 0, 8, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15
    );
    dct_dct_32x8_bitir(dst, stride, coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_32x8_full(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    let mut e0 = yukle_i16x8(chunk(coeff, 0));
    let mut e1 = yukle_i16x8(chunk(coeff, 16));
    let mut e2 = yukle_i16x8(chunk(coeff, 32));
    let mut e3 = yukle_i16x8(chunk(coeff, 48));
    let mut e4 = yukle_i16x8(chunk(coeff, 64));
    let mut e5 = yukle_i16x8(chunk(coeff, 80));
    let mut e6 = yukle_i16x8(chunk(coeff, 96));
    let mut e7 = yukle_i16x8(chunk(coeff, 112));
    let mut e8 = yukle_i16x8(chunk(coeff, 128));
    let mut e9 = yukle_i16x8(chunk(coeff, 144));
    let mut e10 = yukle_i16x8(chunk(coeff, 160));
    let mut e11 = yukle_i16x8(chunk(coeff, 176));
    let mut e12 = yukle_i16x8(chunk(coeff, 192));
    let mut e13 = yukle_i16x8(chunk(coeff, 208));
    let mut e14 = yukle_i16x8(chunk(coeff, 224));
    let mut e15 = yukle_i16x8(chunk(coeff, 240));
    idct16_v!(
        x8, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    );
    // Çift sütunların okunmuş yuvalarına (`2*k*8`) `even` sonuçlarını sakla (`2*k*8 + 8` tek sütunlara dokunmaz!).
    sakla_ymm16!(
        coeff, 0, 16, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    );

    idct32_full_odd_v!(x8,
        yukle_i16x8(chunk(coeff, 8)), yukle_i16x8(chunk(coeff, 24)),
        yukle_i16x8(chunk(coeff, 40)), yukle_i16x8(chunk(coeff, 56)),
        yukle_i16x8(chunk(coeff, 72)), yukle_i16x8(chunk(coeff, 88)),
        yukle_i16x8(chunk(coeff, 104)), yukle_i16x8(chunk(coeff, 120)),
        yukle_i16x8(chunk(coeff, 136)), yukle_i16x8(chunk(coeff, 152)),
        yukle_i16x8(chunk(coeff, 168)), yukle_i16x8(chunk(coeff, 184)),
        yukle_i16x8(chunk(coeff, 200)), yukle_i16x8(chunk(coeff, 216)),
        yukle_i16x8(chunk(coeff, 232)), yukle_i16x8(chunk(coeff, 248))
        => o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15);
    // Önce `even` değerlerini yazmaçlara al ki `coeff[0..256]` üzerine yazarken üst üste binmesin!
    yukle_ymm16!(coeff, 0, 16 => v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
    macro_rules! st_pair {
        ($k:literal, $ev:ident, $od:ident) => {{
            store_i32x8(
                chunk_mut(coeff, $k * 8),
                x8::round_one::<2>(x8::adds($ev, $od)),
            );
            store_i32x8(
                chunk_mut(coeff, 128 + (15 - $k) * 8),
                x8::round_one::<2>(x8::subs($ev, $od)),
            );
        }};
    }
    st_pair!(0, v0, o0);
    st_pair!(1, v1, o1);
    st_pair!(2, v2, o2);
    st_pair!(3, v3, o3);
    st_pair!(4, v4, o4);
    st_pair!(5, v5, o5);
    st_pair!(6, v6, o6);
    st_pair!(7, v7, o7);
    st_pair!(8, v8, o8);
    st_pair!(9, v9, o9);
    st_pair!(10, v10, o10);
    st_pair!(11, v11, o11);
    st_pair!(12, v12, o12);
    st_pair!(13, v13, o13);
    st_pair!(14, v14, o14);
    st_pair!(15, v15, o15);
    dct_dct_32x8_bitir(dst, stride, coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn dct_dct_32x8(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    if eob <= 42 {
        dct_dct_32x8_qtr(dst, stride, coeff);
    } else if eob <= 106 {
        dct_dct_32x8_half(dst, stride, coeff);
    } else {
        dct_dct_32x8_full(dst, stride, coeff);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn idtx_32x8(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    if eob <= 106 {
        let r3 = _mm256_set1_epi16(4096);
        macro_rules! ld2 {
            ($k:expr) => {
                _mm256_mulhrs_epi16(
                    yukle_2col_8(chunk(coeff, $k * 8), chunk(coeff, ($k + 8) * 8)),
                    r3,
                )
            };
        }
        let (mut l0, mut l1, mut l2, mut l3, mut l4, mut l5, mut l6, mut l7) = (
            ld2!(0),
            ld2!(1),
            ld2!(2),
            ld2!(3),
            ld2!(4),
            ld2!(5),
            ld2!(6),
            ld2!(7),
        );
        sifirla_katsayi::<256>(coeff);
        transpose_8x8_ymm_v!(l0, l1, l2, l3, l4, l5, l6, l7);
        ekle_yaz_w16_cift(dst, stride, 0, 0, l0, l1);
        ekle_yaz_w16_cift(dst, stride, 2, 0, l2, l3);
        ekle_yaz_w16_cift(dst, stride, 4, 0, l4, l5);
        ekle_yaz_w16_cift(dst, stride, 6, 0, l6, l7);
        return;
    }

    let r3 = _mm512_set1_epi16(4096);
    macro_rules! ld4 {
        ($k:expr) => {
            _mm512_mulhrs_epi16(
                yukle_4col_8_zmm(
                    chunk(coeff, $k * 8),
                    chunk(coeff, ($k + 8) * 8),
                    chunk(coeff, ($k + 16) * 8),
                    chunk(coeff, ($k + 24) * 8),
                ),
                r3,
            )
        };
    }
    let (mut m0, mut m1, mut m2, mut m3, mut m4, mut m5, mut m6, mut m7) = (
        ld4!(0),
        ld4!(1),
        ld4!(2),
        ld4!(3),
        ld4!(4),
        ld4!(5),
        ld4!(6),
        ld4!(7),
    );
    sifirla_katsayi::<256>(coeff);
    transpose_8x8_zmm_v!(m0, m1, m2, m3, m4, m5, m6, m7);
    ekle_yaz_z32_8_v!(dst, stride, 0, 0, m0, m1, m2, m3, m4, m5, m6, m7);
}

macro_rules! kelebek_w16_32_yaz16 {
    ($dst:expr, $stride:expr, $x_off:expr, $coeff:expr, $ebase:expr, $estep:expr,
     $o0:expr, $o1:expr, $o2:expr, $o3:expr, $o4:expr, $o5:expr, $o6:expr, $o7:expr,
     $o8:expr, $o9:expr, $o10:expr, $o11:expr, $o12:expr, $o13:expr, $o14:expr, $o15:expr) => {{
        let eb = $ebase;
        let es = $estep;
        let xo = $x_off;
        ekle_yaz_w16_kelebek32($dst, $stride, xo, 0, yukle_i32x8(chunk($coeff, eb)), $o0);
        ekle_yaz_w16_kelebek32(
            $dst,
            $stride,
            xo,
            1,
            yukle_i32x8(chunk($coeff, eb + es)),
            $o1,
        );
        ekle_yaz_w16_kelebek32(
            $dst,
            $stride,
            xo,
            2,
            yukle_i32x8(chunk($coeff, eb + 2 * es)),
            $o2,
        );
        ekle_yaz_w16_kelebek32(
            $dst,
            $stride,
            xo,
            3,
            yukle_i32x8(chunk($coeff, eb + 3 * es)),
            $o3,
        );
        ekle_yaz_w16_kelebek32(
            $dst,
            $stride,
            xo,
            4,
            yukle_i32x8(chunk($coeff, eb + 4 * es)),
            $o4,
        );
        ekle_yaz_w16_kelebek32(
            $dst,
            $stride,
            xo,
            5,
            yukle_i32x8(chunk($coeff, eb + 5 * es)),
            $o5,
        );
        ekle_yaz_w16_kelebek32(
            $dst,
            $stride,
            xo,
            6,
            yukle_i32x8(chunk($coeff, eb + 6 * es)),
            $o6,
        );
        ekle_yaz_w16_kelebek32(
            $dst,
            $stride,
            xo,
            7,
            yukle_i32x8(chunk($coeff, eb + 7 * es)),
            $o7,
        );
        ekle_yaz_w16_kelebek32(
            $dst,
            $stride,
            xo,
            8,
            yukle_i32x8(chunk($coeff, eb + 8 * es)),
            $o8,
        );
        ekle_yaz_w16_kelebek32(
            $dst,
            $stride,
            xo,
            9,
            yukle_i32x8(chunk($coeff, eb + 9 * es)),
            $o9,
        );
        ekle_yaz_w16_kelebek32(
            $dst,
            $stride,
            xo,
            10,
            yukle_i32x8(chunk($coeff, eb + 10 * es)),
            $o10,
        );
        ekle_yaz_w16_kelebek32(
            $dst,
            $stride,
            xo,
            11,
            yukle_i32x8(chunk($coeff, eb + 11 * es)),
            $o11,
        );
        ekle_yaz_w16_kelebek32(
            $dst,
            $stride,
            xo,
            12,
            yukle_i32x8(chunk($coeff, eb + 12 * es)),
            $o12,
        );
        ekle_yaz_w16_kelebek32(
            $dst,
            $stride,
            xo,
            13,
            yukle_i32x8(chunk($coeff, eb + 13 * es)),
            $o13,
        );
        ekle_yaz_w16_kelebek32(
            $dst,
            $stride,
            xo,
            14,
            yukle_i32x8(chunk($coeff, eb + 14 * es)),
            $o14,
        );
        ekle_yaz_w16_kelebek32(
            $dst,
            $stride,
            xo,
            15,
            yukle_i32x8(chunk($coeff, eb + 15 * es)),
            $o15,
        );
    }};
}

/// `coeff[0..128]` (`top`) ve `coeff[128..256]` (`bot`) içindeki transpozlanmış `16×32` YMM verisini
/// `x16::idct32` ile işleyip `dst` (`x_off`) üzerine yazar (`coeff[256..384]` `even` scratch).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn idct32_yaz_w16_from_coeff(dst: &mut [u8], stride: usize, x_off: usize, coeff: &mut [i32; 1024]) {
    yukle_ymm16!(coeff, 0, 16
        => e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15);
    idct16_v!(
        x16, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    );
    sakla_ymm16!(
        coeff, 256, 8, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    );

    idct32_full_odd_v!(x16,
        yukle_i32x8(chunk(coeff, 8)), yukle_i32x8(chunk(coeff, 24)),
        yukle_i32x8(chunk(coeff, 40)), yukle_i32x8(chunk(coeff, 56)),
        yukle_i32x8(chunk(coeff, 72)), yukle_i32x8(chunk(coeff, 88)),
        yukle_i32x8(chunk(coeff, 104)), yukle_i32x8(chunk(coeff, 120)),
        yukle_i32x8(chunk(coeff, 136)), yukle_i32x8(chunk(coeff, 152)),
        yukle_i32x8(chunk(coeff, 168)), yukle_i32x8(chunk(coeff, 184)),
        yukle_i32x8(chunk(coeff, 200)), yukle_i32x8(chunk(coeff, 216)),
        yukle_i32x8(chunk(coeff, 232)), yukle_i32x8(chunk(coeff, 248))
        => o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15);
    kelebek_w16_32_yaz16!(
        dst, stride, x_off, coeff, 256, 8, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12,
        o13, o14, o15
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_16x32_half(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    macro_rules! ld {
        ($k:expr) => {
            x16::mul_r2(yukle_i16x16(chunk(coeff, $k * 32)))
        };
    }
    let (mut t0, mut t1, mut t2, mut t3, mut t4, mut t5, mut t6, mut t7) = (
        ld!(0),
        ld!(1),
        ld!(2),
        ld!(3),
        ld!(4),
        ld!(5),
        ld!(6),
        ld!(7),
    );
    let (mut t8, mut t9, mut t10, mut t11, mut t12, mut t13, mut t14, mut t15) = (
        ld!(8),
        ld!(9),
        ld!(10),
        ld!(11),
        ld!(12),
        ld!(13),
        ld!(14),
        ld!(15),
    );
    idct16_v!(
        x16, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15
    );
    round_shift16_v!(
        x16, 1, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15
    );
    transpose_16x16_ymm_v!(
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15
    );
    idct16_half_v!(x16, t0, t2, t4, t6, t8, t10, t12, t14
        => e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15);
    sakla_ymm16!(
        coeff, 0, 8, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    );
    idct32_half_odd_v!(x16, t1, t3, t5, t7, t9, t11, t13, t15
        => o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15);
    kelebek_w16_32_yaz16!(
        dst, stride, 0, coeff, 0, 8, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13,
        o14, o15
    );
    sifirla_katsayi::<512>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_16x32_full(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    macro_rules! ldz {
        ($k:expr) => {
            z32::mul_r2(yukle_z32(chunk(coeff, $k * 32)))
        };
    }
    let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5, mut c6, mut c7) = (
        ldz!(0),
        ldz!(1),
        ldz!(2),
        ldz!(3),
        ldz!(4),
        ldz!(5),
        ldz!(6),
        ldz!(7),
    );
    let (mut c8, mut c9, mut c10, mut c11, mut c12, mut c13, mut c14, mut c15) = (
        ldz!(8),
        ldz!(9),
        ldz!(10),
        ldz!(11),
        ldz!(12),
        ldz!(13),
        ldz!(14),
        ldz!(15),
    );
    idct16_v!(
        z32, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15
    );
    round_shift16_v!(
        z32, 1, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15
    );

    // Alt yarıları (`bot`) `coeff[128..256]` içine sakla, üst yarıyı (`top`) transpozlayıp `coeff[0..128]` içine yaz.
    sakla_ymm16!(
        coeff,
        128,
        8,
        _mm512_extracti64x4_epi64::<1>(c0),
        _mm512_extracti64x4_epi64::<1>(c1),
        _mm512_extracti64x4_epi64::<1>(c2),
        _mm512_extracti64x4_epi64::<1>(c3),
        _mm512_extracti64x4_epi64::<1>(c4),
        _mm512_extracti64x4_epi64::<1>(c5),
        _mm512_extracti64x4_epi64::<1>(c6),
        _mm512_extracti64x4_epi64::<1>(c7),
        _mm512_extracti64x4_epi64::<1>(c8),
        _mm512_extracti64x4_epi64::<1>(c9),
        _mm512_extracti64x4_epi64::<1>(c10),
        _mm512_extracti64x4_epi64::<1>(c11),
        _mm512_extracti64x4_epi64::<1>(c12),
        _mm512_extracti64x4_epi64::<1>(c13),
        _mm512_extracti64x4_epi64::<1>(c14),
        _mm512_extracti64x4_epi64::<1>(c15)
    );
    let (mut t0, mut t1, mut t2, mut t3, mut t4, mut t5, mut t6, mut t7) = (
        _mm512_castsi512_si256(c0),
        _mm512_castsi512_si256(c1),
        _mm512_castsi512_si256(c2),
        _mm512_castsi512_si256(c3),
        _mm512_castsi512_si256(c4),
        _mm512_castsi512_si256(c5),
        _mm512_castsi512_si256(c6),
        _mm512_castsi512_si256(c7),
    );
    let (mut t8, mut t9, mut t10, mut t11, mut t12, mut t13, mut t14, mut t15) = (
        _mm512_castsi512_si256(c8),
        _mm512_castsi512_si256(c9),
        _mm512_castsi512_si256(c10),
        _mm512_castsi512_si256(c11),
        _mm512_castsi512_si256(c12),
        _mm512_castsi512_si256(c13),
        _mm512_castsi512_si256(c14),
        _mm512_castsi512_si256(c15),
    );
    transpose_16x16_ymm_v!(
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15
    );
    sakla_ymm16!(
        coeff, 0, 8, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15
    );

    yukle_ymm16!(coeff, 128, 8 => b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15);
    transpose_16x16_ymm_v!(
        b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15
    );
    sakla_ymm16!(
        coeff, 128, 8, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15
    );

    idct32_yaz_w16_from_coeff(dst, stride, 0, coeff);
    sifirla_katsayi::<512>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn dct_dct_16x32(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    if eob <= 150 {
        dct_dct_16x32_half(dst, stride, coeff);
    } else {
        dct_dct_16x32_full(dst, stride, coeff);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn idtx_16x32(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    let strips = if eob <= 43 {
        1usize
    } else if eob <= 150 {
        2
    } else if eob <= 278 {
        3
    } else {
        4
    };

    let pairs = strips / 2;
    if pairs > 0 {
        let r_rect = _mm512_set1_epi16(23168);
        let k16 = _mm512_set1_epi16(27152);
        let r1 = _mm512_set1_epi16(16384);
        let r2 = _mm512_set1_epi16(8192);
        for p in 0..pairs {
            let y0 = p * 16;
            let y1 = y0 + 8;
            macro_rules! ldz {
                ($k:expr) => {{
                    let v0 = _mm512_mulhrs_epi16(
                        yukle_4col_8_zmm(
                            chunk(coeff, $k * 32 + y0),
                            chunk(coeff, ($k + 8) * 32 + y0),
                            chunk(coeff, $k * 32 + y1),
                            chunk(coeff, ($k + 8) * 32 + y1),
                        ),
                        r_rect,
                    );
                    let t = _mm512_mulhrs_epi16(v0, k16);
                    let v1 = _mm512_adds_epi16(v0, _mm512_mulhrs_epi16(t, r1));
                    _mm512_mulhrs_epi16(v1, r2)
                }};
            }
            let (mut m0, mut m1, mut m2, mut m3, mut m4, mut m5, mut m6, mut m7) = (
                ldz!(0),
                ldz!(1),
                ldz!(2),
                ldz!(3),
                ldz!(4),
                ldz!(5),
                ldz!(6),
                ldz!(7),
            );
            transpose_8x8_zmm_v!(m0, m1, m2, m3, m4, m5, m6, m7);
            macro_rules! w16z {
                ($k:literal, $a:ident, $b:ident) => {{
                    ekle_yaz_w16_cift(
                        dst,
                        stride,
                        y0 + $k,
                        0,
                        _mm512_castsi512_si256($a),
                        _mm512_castsi512_si256($b),
                    );
                    ekle_yaz_w16_cift(
                        dst,
                        stride,
                        y1 + $k,
                        0,
                        _mm512_extracti64x4_epi64::<1>($a),
                        _mm512_extracti64x4_epi64::<1>($b),
                    );
                }};
            }
            w16z!(0, m0, m1);
            w16z!(2, m2, m3);
            w16z!(4, m4, m5);
            w16z!(6, m6, m7);
        }
    }

    if strips & 1 != 0 {
        let y_off = (strips - 1) * 8;
        let r_rect = _mm256_set1_epi16(23168);
        let k16 = _mm256_set1_epi16(27152);
        let r1 = _mm256_set1_epi16(16384);
        let r2 = _mm256_set1_epi16(8192);
        macro_rules! ldy {
            ($k:expr) => {{
                let v0 = _mm256_mulhrs_epi16(
                    yukle_2col_8(
                        chunk(coeff, $k * 32 + y_off),
                        chunk(coeff, ($k + 8) * 32 + y_off),
                    ),
                    r_rect,
                );
                let t = _mm256_mulhrs_epi16(v0, k16);
                let v1 = _mm256_adds_epi16(v0, _mm256_mulhrs_epi16(t, r1));
                _mm256_mulhrs_epi16(v1, r2)
            }};
        }
        let (mut m0, mut m1, mut m2, mut m3, mut m4, mut m5, mut m6, mut m7) = (
            ldy!(0),
            ldy!(1),
            ldy!(2),
            ldy!(3),
            ldy!(4),
            ldy!(5),
            ldy!(6),
            ldy!(7),
        );
        transpose_8x8_ymm_v!(m0, m1, m2, m3, m4, m5, m6, m7);
        ekle_yaz_w16_cift(dst, stride, y_off, 0, m0, m1);
        ekle_yaz_w16_cift(dst, stride, y_off + 2, 0, m2, m3);
        ekle_yaz_w16_cift(dst, stride, y_off + 4, 0, m4, m5);
        ekle_yaz_w16_cift(dst, stride, y_off + 6, 0, m6, m7);
    }
    sifirla_katsayi::<512>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn transpoz_yerinde_16x16(coeff: &mut [i32; 1024], off: usize) {
    yukle_ymm16!(coeff, off, 8 => m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15);
    transpose_16x16_ymm_v!(
        m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15
    );
    sakla_ymm16!(
        coeff, off, 8, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15
    );
}

/// `coeff[base .. base+128]` (`lo16`) ve `coeff[base+128 .. base+256]` (`hi16`) içindeki `32×16` verisini
/// transpozlayıp 2. geçiş ZMM `idct16` ile `dst` üzerine yazar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_32x16_bitir(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], base: usize) {
    transpoz_yerinde_16x16(coeff, base);
    transpoz_yerinde_16x16(coeff, base + 128);

    let mut r0 = birlestir_zmm(
        yukle_i32x8(chunk(coeff, base)),
        yukle_i32x8(chunk(coeff, base + 128)),
    );
    let mut r1 = birlestir_zmm(
        yukle_i32x8(chunk(coeff, base + 8)),
        yukle_i32x8(chunk(coeff, base + 136)),
    );
    let mut r2 = birlestir_zmm(
        yukle_i32x8(chunk(coeff, base + 16)),
        yukle_i32x8(chunk(coeff, base + 144)),
    );
    let mut r3 = birlestir_zmm(
        yukle_i32x8(chunk(coeff, base + 24)),
        yukle_i32x8(chunk(coeff, base + 152)),
    );
    let mut r4 = birlestir_zmm(
        yukle_i32x8(chunk(coeff, base + 32)),
        yukle_i32x8(chunk(coeff, base + 160)),
    );
    let mut r5 = birlestir_zmm(
        yukle_i32x8(chunk(coeff, base + 40)),
        yukle_i32x8(chunk(coeff, base + 168)),
    );
    let mut r6 = birlestir_zmm(
        yukle_i32x8(chunk(coeff, base + 48)),
        yukle_i32x8(chunk(coeff, base + 176)),
    );
    let mut r7 = birlestir_zmm(
        yukle_i32x8(chunk(coeff, base + 56)),
        yukle_i32x8(chunk(coeff, base + 184)),
    );
    let mut r8 = birlestir_zmm(
        yukle_i32x8(chunk(coeff, base + 64)),
        yukle_i32x8(chunk(coeff, base + 192)),
    );
    let mut r9 = birlestir_zmm(
        yukle_i32x8(chunk(coeff, base + 72)),
        yukle_i32x8(chunk(coeff, base + 200)),
    );
    let mut r10 = birlestir_zmm(
        yukle_i32x8(chunk(coeff, base + 80)),
        yukle_i32x8(chunk(coeff, base + 208)),
    );
    let mut r11 = birlestir_zmm(
        yukle_i32x8(chunk(coeff, base + 88)),
        yukle_i32x8(chunk(coeff, base + 216)),
    );
    let mut r12 = birlestir_zmm(
        yukle_i32x8(chunk(coeff, base + 96)),
        yukle_i32x8(chunk(coeff, base + 224)),
    );
    let mut r13 = birlestir_zmm(
        yukle_i32x8(chunk(coeff, base + 104)),
        yukle_i32x8(chunk(coeff, base + 232)),
    );
    let mut r14 = birlestir_zmm(
        yukle_i32x8(chunk(coeff, base + 112)),
        yukle_i32x8(chunk(coeff, base + 240)),
    );
    let mut r15 = birlestir_zmm(
        yukle_i32x8(chunk(coeff, base + 120)),
        yukle_i32x8(chunk(coeff, base + 248)),
    );

    idct16_v!(
        z32, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
    );
    round_p2_16_v!(
        z32, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
    );
    ekle_yaz_z32_8_v!(dst, stride, 0, 0, r0, r1, r2, r3, r4, r5, r6, r7);
    ekle_yaz_z32_8_v!(dst, stride, 8, 0, r8, r9, r10, r11, r12, r13, r14, r15);
    sifirla_katsayi::<512>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_32x16_half(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    macro_rules! ld {
        ($k:expr) => {
            x16::mul_r2(yukle_i16x16(chunk(coeff, $k * 16)))
        };
    }
    idct16_half_v!(x16, ld!(0), ld!(2), ld!(4), ld!(6), ld!(8), ld!(10), ld!(12), ld!(14)
        => e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15);
    sakla_ymm16!(
        coeff, 256, 8, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    );
    idct32_half_odd_v!(x16, ld!(1), ld!(3), ld!(5), ld!(7), ld!(9), ld!(11), ld!(13), ld!(15)
        => o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15);
    kelebek_sakla_lo_hi16!(
        x16, 1, coeff, 256, 8, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15
    );
    dct_dct_32x16_bitir(dst, stride, coeff, 0);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_32x16_full_even(coeff: &mut [i32; 1024]) {
    macro_rules! ld {
        ($k:expr) => {
            x16::mul_r2(yukle_i16x16(chunk(coeff, $k * 16)))
        };
    }
    let (mut e0, mut e1, mut e2, mut e3, mut e4, mut e5, mut e6, mut e7) = (
        ld!(0),
        ld!(2),
        ld!(4),
        ld!(6),
        ld!(8),
        ld!(10),
        ld!(12),
        ld!(14),
    );
    let (mut e8, mut e9, mut e10, mut e11, mut e12, mut e13, mut e14, mut e15) = (
        ld!(16),
        ld!(18),
        ld!(20),
        ld!(22),
        ld!(24),
        ld!(26),
        ld!(28),
        ld!(30),
    );
    idct16_v!(
        x16, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    );
    sakla_ymm16!(
        coeff, 0, 32, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn transpoz_32x16_full(coeff: &mut [i32; 1024]) {
    yukle_ymm16!(coeff, 0, 32 => l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15);
    transpose_16x16_ymm_v!(
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15
    );
    sakla_ymm16!(
        coeff, 0, 32, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15
    );

    yukle_ymm16!(coeff, 8, 32 => h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15);
    core::mem::swap(&mut h0, &mut h15);
    core::mem::swap(&mut h1, &mut h14);
    core::mem::swap(&mut h2, &mut h13);
    core::mem::swap(&mut h3, &mut h12);
    core::mem::swap(&mut h4, &mut h11);
    core::mem::swap(&mut h5, &mut h10);
    core::mem::swap(&mut h6, &mut h9);
    core::mem::swap(&mut h7, &mut h8);
    transpose_16x16_ymm_v!(
        h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15
    );
    sakla_ymm16!(
        coeff, 8, 32, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_32x16_full_gecis2(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    transpoz_32x16_full(coeff);
    let mut r0 = oku_zmm(chunk(coeff, 0));
    let mut r1 = oku_zmm(chunk(coeff, 32));
    let mut r2 = oku_zmm(chunk(coeff, 2 * 32));
    let mut r3 = oku_zmm(chunk(coeff, 3 * 32));
    let mut r4 = oku_zmm(chunk(coeff, 4 * 32));
    let mut r5 = oku_zmm(chunk(coeff, 5 * 32));
    let mut r6 = oku_zmm(chunk(coeff, 6 * 32));
    let mut r7 = oku_zmm(chunk(coeff, 7 * 32));
    let mut r8 = oku_zmm(chunk(coeff, 8 * 32));
    let mut r9 = oku_zmm(chunk(coeff, 9 * 32));
    let mut r10 = oku_zmm(chunk(coeff, 10 * 32));
    let mut r11 = oku_zmm(chunk(coeff, 11 * 32));
    let mut r12 = oku_zmm(chunk(coeff, 12 * 32));
    let mut r13 = oku_zmm(chunk(coeff, 13 * 32));
    let mut r14 = oku_zmm(chunk(coeff, 14 * 32));
    let mut r15 = oku_zmm(chunk(coeff, 15 * 32));
    idct16_v!(
        z32, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
    );
    round_p2_16_v!(
        z32, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
    );
    ekle_yaz_z32_8_v!(dst, stride, 0, 0, r0, r1, r2, r3, r4, r5, r6, r7);
    ekle_yaz_z32_8_v!(dst, stride, 8, 0, r8, r9, r10, r11, r12, r13, r14, r15);
    sifirla_katsayi::<512>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_32x16_full(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    dct_dct_32x16_full_even(coeff);
    macro_rules! ld {
        ($k:expr) => {
            x16::mul_r2(yukle_i16x16(chunk(coeff, $k * 16)))
        };
    }
    idct32_full_odd_v!(x16,
        ld!(1), ld!(3), ld!(5), ld!(7), ld!(9), ld!(11), ld!(13), ld!(15),
        ld!(17), ld!(19), ld!(21), ld!(23), ld!(25), ld!(27), ld!(29), ld!(31)
        => o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15);
    macro_rules! st_in_place {
        ($k:literal, $od:ident) => {{
            let ev = yukle_i32x8(chunk(coeff, $k * 32));
            store_i32x8(
                chunk_mut(coeff, $k * 32),
                x16::round_one::<1>(x16::adds(ev, $od)),
            );
            store_i32x8(
                chunk_mut(coeff, $k * 32 + 8),
                x16::round_one::<1>(x16::subs(ev, $od)),
            );
        }};
    }
    st_in_place!(0, o0);
    st_in_place!(1, o1);
    st_in_place!(2, o2);
    st_in_place!(3, o3);
    st_in_place!(4, o4);
    st_in_place!(5, o5);
    st_in_place!(6, o6);
    st_in_place!(7, o7);
    st_in_place!(8, o8);
    st_in_place!(9, o9);
    st_in_place!(10, o10);
    st_in_place!(11, o11);
    st_in_place!(12, o12);
    st_in_place!(13, o13);
    st_in_place!(14, o14);
    st_in_place!(15, o15);
    dct_dct_32x16_full_gecis2(dst, stride, coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn dct_dct_32x16(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    if eob <= 150 {
        dct_dct_32x16_half(dst, stride, coeff);
    } else {
        dct_dct_32x16_full(dst, stride, coeff);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn idtx_32x16(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    if eob <= 150 {
        let r_rect = _mm256_set1_epi16(23168);
        let k16 = _mm256_set1_epi16(27152);
        let r4 = _mm256_set1_epi16(2048);
        let n_strips = if eob <= 35 { 1usize } else { 2 };
        for s in 0..n_strips {
            let y_off = s * 8;
            macro_rules! ldy {
                ($k:expr) => {{
                    let v0 = _mm256_mulhrs_epi16(
                        yukle_2col_8(
                            chunk(coeff, $k * 16 + y_off),
                            chunk(coeff, ($k + 8) * 16 + y_off),
                        ),
                        r_rect,
                    );
                    let v1 = _mm256_adds_epi16(v0, v0);
                    let t = _mm256_mulhrs_epi16(v1, k16);
                    let s2 = _mm256_adds_epi16(_mm256_adds_epi16(v1, v1), t);
                    _mm256_mulhrs_epi16(s2, r4)
                }};
            }
            let (mut m0, mut m1, mut m2, mut m3, mut m4, mut m5, mut m6, mut m7) = (
                ldy!(0),
                ldy!(1),
                ldy!(2),
                ldy!(3),
                ldy!(4),
                ldy!(5),
                ldy!(6),
                ldy!(7),
            );
            transpose_8x8_ymm_v!(m0, m1, m2, m3, m4, m5, m6, m7);
            ekle_yaz_w16_cift(dst, stride, y_off, 0, m0, m1);
            ekle_yaz_w16_cift(dst, stride, y_off + 2, 0, m2, m3);
            ekle_yaz_w16_cift(dst, stride, y_off + 4, 0, m4, m5);
            ekle_yaz_w16_cift(dst, stride, y_off + 6, 0, m6, m7);
        }
        sifirla_katsayi::<512>(coeff);
        return;
    }

    let r_rect = _mm512_set1_epi16(23168);
    let k16 = _mm512_set1_epi16(27152);
    let r4 = _mm512_set1_epi16(2048);
    for y_off in [0usize, 8] {
        macro_rules! ldz {
            ($k:expr) => {{
                let v0 = _mm512_mulhrs_epi16(
                    yukle_4col_8_zmm(
                        chunk(coeff, $k * 16 + y_off),
                        chunk(coeff, ($k + 8) * 16 + y_off),
                        chunk(coeff, ($k + 16) * 16 + y_off),
                        chunk(coeff, ($k + 24) * 16 + y_off),
                    ),
                    r_rect,
                );
                let v1 = _mm512_adds_epi16(v0, v0);
                let t = _mm512_mulhrs_epi16(v1, k16);
                let s2 = _mm512_adds_epi16(_mm512_adds_epi16(v1, v1), t);
                _mm512_mulhrs_epi16(s2, r4)
            }};
        }
        let (mut m0, mut m1, mut m2, mut m3, mut m4, mut m5, mut m6, mut m7) = (
            ldz!(0),
            ldz!(1),
            ldz!(2),
            ldz!(3),
            ldz!(4),
            ldz!(5),
            ldz!(6),
            ldz!(7),
        );
        transpose_8x8_zmm_v!(m0, m1, m2, m3, m4, m5, m6, m7);
        ekle_yaz_z32_8_v!(dst, stride, y_off, 0, m0, m1, m2, m3, m4, m5, m6, m7);
    }
    sifirla_katsayi::<512>(coeff);
}

// ---------------------------------------------------------------------------
// Boyut × tür bazında büyük 2B AVX512ICL çekirdek gövdeleri (`32x32` .. `64x64`)
// ---------------------------------------------------------------------------

macro_rules! kelebek_z32_32_yaz16 {
    ($dst:expr, $stride:expr, $x_off:expr, $coeff:expr, $ebase:expr,
     $o0:expr, $o1:expr, $o2:expr, $o3:expr, $o4:expr, $o5:expr, $o6:expr, $o7:expr,
     $o8:expr, $o9:expr, $o10:expr, $o11:expr, $o12:expr, $o13:expr, $o14:expr, $o15:expr) => {{
        let eb = $ebase;
        let xo = $x_off;
        macro_rules! pair_z {
            ($k:literal, $ok:expr) => {{
                let ek = oku_zmm(chunk($coeff, eb + $k * 16));
                let ok_v = $ok;
                ekle_yaz_z32_satir($dst, $k * $stride + xo, z32::round_p2(z32::adds(ek, ok_v)));
                ekle_yaz_z32_satir(
                    $dst,
                    (31 - $k) * $stride + xo,
                    z32::round_p2(z32::subs(ek, ok_v)),
                );
            }};
        }
        pair_z!(0, $o0);
        pair_z!(1, $o1);
        pair_z!(2, $o2);
        pair_z!(3, $o3);
        pair_z!(4, $o4);
        pair_z!(5, $o5);
        pair_z!(6, $o6);
        pair_z!(7, $o7);
        pair_z!(8, $o8);
        pair_z!(9, $o9);
        pair_z!(10, $o10);
        pair_z!(11, $o11);
        pair_z!(12, $o12);
        pair_z!(13, $o13);
        pair_z!(14, $o14);
        pair_z!(15, $o15);
    }};
}

/// `32×32` ve `32×64` tam yolları için 512-bit ZMM 1. geçiş (`idct32`) + `32×32` transpoz:
/// transpozlanmış `32` satırı (`[x0..31]`, her biri 1 ZMM = `16 × i32`) `coeff[512..1024]` içine yazar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn pass1_ve_transpoz_32x32_z32<const SHIFT: u32, const RECT2: bool>(coeff: &mut [i32; 1024]) {
    macro_rules! colz {
        ($off:expr) => {{
            let v = yukle_z32(chunk(coeff, $off));
            if RECT2 { z32::mul_r2(v) } else { v }
        }};
    }

    // Adım 1: 16 çift sütun (`x = 2*k`) -> `coeff[k*64 .. +16]`.
    {
        let (mut e0, mut e1, mut e2, mut e3, mut e4, mut e5, mut e6, mut e7) = (
            colz!(0),
            colz!(64),
            colz!(128),
            colz!(192),
            colz!(256),
            colz!(320),
            colz!(384),
            colz!(448),
        );
        let (mut e8, mut e9, mut e10, mut e11, mut e12, mut e13, mut e14, mut e15) = (
            colz!(512),
            colz!(576),
            colz!(640),
            colz!(704),
            colz!(768),
            colz!(832),
            colz!(896),
            colz!(960),
        );
        idct16_v!(
            z32, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
        );
        sakla_zmm16!(
            coeff, 0, 64, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
        );
    }

    // Adım 2: 16 tek sütun (`m*64 + 32`, `m = 0..16`) -> `even ± odd` -> `coeff[0..512]` içine sıkıştır (`k*32` ve `k*32+16`)!
    {
        idct32_full_odd_v!(z32,
            colz!(32), colz!(64 + 32), colz!(2 * 64 + 32), colz!(3 * 64 + 32),
            colz!(4 * 64 + 32), colz!(5 * 64 + 32), colz!(6 * 64 + 32), colz!(7 * 64 + 32),
            colz!(8 * 64 + 32), colz!(9 * 64 + 32), colz!(10 * 64 + 32), colz!(11 * 64 + 32),
            colz!(12 * 64 + 32), colz!(13 * 64 + 32), colz!(14 * 64 + 32), colz!(15 * 64 + 32)
            => o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15);
        macro_rules! comb {
            ($k:literal, $ok:expr) => {{
                let ek = oku_zmm(chunk(coeff, $k * 64));
                let ok_v = $ok;
                sakla_zmm(
                    chunk_mut(coeff, $k * 32),
                    z32::round_one::<SHIFT>(z32::adds(ek, ok_v)),
                );
                sakla_zmm(
                    chunk_mut(coeff, $k * 32 + 16),
                    z32::round_one::<SHIFT>(z32::subs(ek, ok_v)),
                );
            }};
        }
        comb!(0, o0);
        comb!(1, o1);
        comb!(2, o2);
        comb!(3, o3);
        comb!(4, o4);
        comb!(5, o5);
        comb!(6, o6);
        comb!(7, o7);
        comb!(8, o8);
        comb!(9, o9);
        comb!(10, o10);
        comb!(11, o11);
        comb!(12, o12);
        comb!(13, o13);
        comb!(14, o14);
        comb!(15, o15);
    }

    // Adım 3 (`y = 0..16` üst yarı) ve Adım 4 (`y = 16..32` alt yarı): `coeff[0..512]`'den oku, `coeff[512..1024]`'e yaz!
    for half_y in 0..2usize {
        let sub = half_y * 8;
        let out_base = 512 + half_y * 256;
        yukle_ymm16!(coeff, sub, 32
            => l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15);
        transpose_16x16_ymm_v!(
            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15
        );
        sakla_ymm16!(
            coeff, out_base, 16, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14,
            l15
        );

        // `rt[15 - k]` = `coeff[k * 32 + 16 + sub]` (`j = 15 - k` -> `(15 - j) * 32 + 16 + sub`).
        let mut r0 = yukle_i32x8(chunk(coeff, 15 * 32 + 16 + sub));
        let mut r1 = yukle_i32x8(chunk(coeff, 14 * 32 + 16 + sub));
        let mut r2 = yukle_i32x8(chunk(coeff, 13 * 32 + 16 + sub));
        let mut r3 = yukle_i32x8(chunk(coeff, 12 * 32 + 16 + sub));
        let mut r4 = yukle_i32x8(chunk(coeff, 11 * 32 + 16 + sub));
        let mut r5 = yukle_i32x8(chunk(coeff, 10 * 32 + 16 + sub));
        let mut r6 = yukle_i32x8(chunk(coeff, 9 * 32 + 16 + sub));
        let mut r7 = yukle_i32x8(chunk(coeff, 8 * 32 + 16 + sub));
        let mut r8 = yukle_i32x8(chunk(coeff, 7 * 32 + 16 + sub));
        let mut r9 = yukle_i32x8(chunk(coeff, 6 * 32 + 16 + sub));
        let mut r10 = yukle_i32x8(chunk(coeff, 5 * 32 + 16 + sub));
        let mut r11 = yukle_i32x8(chunk(coeff, 4 * 32 + 16 + sub));
        let mut r12 = yukle_i32x8(chunk(coeff, 3 * 32 + 16 + sub));
        let mut r13 = yukle_i32x8(chunk(coeff, 2 * 32 + 16 + sub));
        let mut r14 = yukle_i32x8(chunk(coeff, 32 + 16 + sub));
        let mut r15 = yukle_i32x8(chunk(coeff, 16 + sub));
        transpose_16x16_ymm_v!(
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
        );
        sakla_ymm16!(
            coeff,
            out_base + 8,
            16,
            r0,
            r1,
            r2,
            r3,
            r4,
            r5,
            r6,
            r7,
            r8,
            r9,
            r10,
            r11,
            r12,
            r13,
            r14,
            r15
        );
    }
}

/// `coeff[in_base .. +256]` içindeki `16` ZMM satırı (`y = 0..16`) 512-bit ZMM `idct32_half` ile işleyip
/// `dst` (`x_off`) üzerine yazar (`coeff[scr_base .. +256]` `even` scratch olarak kullanılır).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn idct32_yaz_z32_half_from_coeff(
    dst: &mut [u8],
    stride: usize,
    x_off: usize,
    coeff: &mut [i32; 1024],
    in_base: usize,
    scr_base: usize,
) {
    idct16_half_v!(z32,
        oku_zmm(chunk(coeff, in_base)),
        oku_zmm(chunk(coeff, in_base + 2 * 16)),
        oku_zmm(chunk(coeff, in_base + 4 * 16)),
        oku_zmm(chunk(coeff, in_base + 6 * 16)),
        oku_zmm(chunk(coeff, in_base + 8 * 16)),
        oku_zmm(chunk(coeff, in_base + 10 * 16)),
        oku_zmm(chunk(coeff, in_base + 12 * 16)),
        oku_zmm(chunk(coeff, in_base + 14 * 16))
        => e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15);
    sakla_zmm16!(
        coeff, scr_base, 16, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    );

    idct32_half_odd_v!(z32,
        oku_zmm(chunk(coeff, in_base + 16)),
        oku_zmm(chunk(coeff, in_base + 3 * 16)),
        oku_zmm(chunk(coeff, in_base + 5 * 16)),
        oku_zmm(chunk(coeff, in_base + 7 * 16)),
        oku_zmm(chunk(coeff, in_base + 9 * 16)),
        oku_zmm(chunk(coeff, in_base + 11 * 16)),
        oku_zmm(chunk(coeff, in_base + 13 * 16)),
        oku_zmm(chunk(coeff, in_base + 15 * 16))
        => o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15);
    kelebek_z32_32_yaz16!(
        dst, stride, x_off, coeff, scr_base, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12,
        o13, o14, o15
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn transpoz_cift_16x16(coeff: &mut [i32; 1024], dst_lo: usize, dst_step: usize, dst_hi: usize) {
    yukle_ymm16!(coeff, 0, 8 => l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15);
    transpose_16x16_ymm_v!(
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15
    );
    sakla_ymm16!(
        coeff, dst_lo, dst_step, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14,
        l15
    );

    yukle_ymm16!(coeff, 128, 8 => h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15);
    transpose_16x16_ymm_v!(
        h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15
    );
    sakla_ymm16!(
        coeff, dst_hi, dst_step, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
        h15
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_32x32_qtr(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    idct16_half_v!(x16,
        yukle_i16x16(chunk(coeff, 0)),
        yukle_i16x16(chunk(coeff, 2 * 32)),
        yukle_i16x16(chunk(coeff, 4 * 32)),
        yukle_i16x16(chunk(coeff, 6 * 32)),
        yukle_i16x16(chunk(coeff, 8 * 32)),
        yukle_i16x16(chunk(coeff, 10 * 32)),
        yukle_i16x16(chunk(coeff, 12 * 32)),
        yukle_i16x16(chunk(coeff, 14 * 32))
        => e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15);
    sakla_ymm16!(
        coeff, 512, 8, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    );
    idct32_half_odd_v!(x16,
        yukle_i16x16(chunk(coeff, 32)),
        yukle_i16x16(chunk(coeff, 3 * 32)),
        yukle_i16x16(chunk(coeff, 5 * 32)),
        yukle_i16x16(chunk(coeff, 7 * 32)),
        yukle_i16x16(chunk(coeff, 9 * 32)),
        yukle_i16x16(chunk(coeff, 11 * 32)),
        yukle_i16x16(chunk(coeff, 13 * 32)),
        yukle_i16x16(chunk(coeff, 15 * 32))
        => o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15);
    kelebek_sakla_lo_hi16!(
        x16, 2, coeff, 512, 8, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15
    );

    transpoz_cift_16x16(coeff, 512, 16, 520);
    idct32_yaz_z32_half_from_coeff(dst, stride, 0, coeff, 512, 0);
    sifirla_katsayi::<1024>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_32x32_full(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    pass1_ve_transpoz_32x32_z32::<2, false>(coeff);

    {
        let (mut e0, mut e1, mut e2, mut e3, mut e4, mut e5, mut e6, mut e7) = (
            oku_zmm(chunk(coeff, 512)),
            oku_zmm(chunk(coeff, 512 + 2 * 16)),
            oku_zmm(chunk(coeff, 512 + 4 * 16)),
            oku_zmm(chunk(coeff, 512 + 6 * 16)),
            oku_zmm(chunk(coeff, 512 + 8 * 16)),
            oku_zmm(chunk(coeff, 512 + 10 * 16)),
            oku_zmm(chunk(coeff, 512 + 12 * 16)),
            oku_zmm(chunk(coeff, 512 + 14 * 16)),
        );
        let (mut e8, mut e9, mut e10, mut e11, mut e12, mut e13, mut e14, mut e15) = (
            oku_zmm(chunk(coeff, 512 + 16 * 16)),
            oku_zmm(chunk(coeff, 512 + 18 * 16)),
            oku_zmm(chunk(coeff, 512 + 20 * 16)),
            oku_zmm(chunk(coeff, 512 + 22 * 16)),
            oku_zmm(chunk(coeff, 512 + 24 * 16)),
            oku_zmm(chunk(coeff, 512 + 26 * 16)),
            oku_zmm(chunk(coeff, 512 + 28 * 16)),
            oku_zmm(chunk(coeff, 512 + 30 * 16)),
        );
        idct16_v!(
            z32, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
        );
        sakla_zmm16!(
            coeff, 0, 16, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
        );
    }

    idct32_full_odd_v!(z32,
        oku_zmm(chunk(coeff, 512 + 16)),
        oku_zmm(chunk(coeff, 512 + 3 * 16)),
        oku_zmm(chunk(coeff, 512 + 5 * 16)),
        oku_zmm(chunk(coeff, 512 + 7 * 16)),
        oku_zmm(chunk(coeff, 512 + 9 * 16)),
        oku_zmm(chunk(coeff, 512 + 11 * 16)),
        oku_zmm(chunk(coeff, 512 + 13 * 16)),
        oku_zmm(chunk(coeff, 512 + 15 * 16)),
        oku_zmm(chunk(coeff, 512 + 17 * 16)),
        oku_zmm(chunk(coeff, 512 + 19 * 16)),
        oku_zmm(chunk(coeff, 512 + 21 * 16)),
        oku_zmm(chunk(coeff, 512 + 23 * 16)),
        oku_zmm(chunk(coeff, 512 + 25 * 16)),
        oku_zmm(chunk(coeff, 27 * 16 + 512)),
        oku_zmm(chunk(coeff, 29 * 16 + 512)),
        oku_zmm(chunk(coeff, 31 * 16 + 512))
        => o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15);
    kelebek_z32_32_yaz16!(
        dst, stride, 0, coeff, 0, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14,
        o15
    );
    sifirla_katsayi::<1024>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn dct_dct_32x32(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    if eob < 136 {
        dct_dct_32x32_qtr(dst, stride, coeff);
    } else {
        dct_dct_32x32_full(dst, stride, coeff);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn idtx_32x32(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    if eob < 136 {
        let r2 = _mm256_set1_epi16(8192);
        for y_off in [0usize, 8] {
            macro_rules! ldy {
                ($k:expr) => {
                    _mm256_mulhrs_epi16(
                        yukle_2col_8(
                            chunk(coeff, $k * 32 + y_off),
                            chunk(coeff, ($k + 8) * 32 + y_off),
                        ),
                        r2,
                    )
                };
            }
            let (mut m0, mut m1, mut m2, mut m3, mut m4, mut m5, mut m6, mut m7) = (
                ldy!(0),
                ldy!(1),
                ldy!(2),
                ldy!(3),
                ldy!(4),
                ldy!(5),
                ldy!(6),
                ldy!(7),
            );
            transpose_8x8_ymm_v!(m0, m1, m2, m3, m4, m5, m6, m7);
            ekle_yaz_w16_cift(dst, stride, y_off, 0, m0, m1);
            ekle_yaz_w16_cift(dst, stride, y_off + 2, 0, m2, m3);
            ekle_yaz_w16_cift(dst, stride, y_off + 4, 0, m4, m5);
            ekle_yaz_w16_cift(dst, stride, y_off + 6, 0, m6, m7);
        }
        sifirla_katsayi::<1024>(coeff);
        return;
    }

    let r2 = _mm512_set1_epi16(8192);
    for y_off in [0usize, 8, 16, 24] {
        macro_rules! ldz {
            ($k:expr) => {
                _mm512_mulhrs_epi16(
                    yukle_4col_8_zmm(
                        chunk(coeff, $k * 32 + y_off),
                        chunk(coeff, ($k + 8) * 32 + y_off),
                        chunk(coeff, ($k + 16) * 32 + y_off),
                        chunk(coeff, ($k + 24) * 32 + y_off),
                    ),
                    r2,
                )
            };
        }
        let (mut m0, mut m1, mut m2, mut m3, mut m4, mut m5, mut m6, mut m7) = (
            ldz!(0),
            ldz!(1),
            ldz!(2),
            ldz!(3),
            ldz!(4),
            ldz!(5),
            ldz!(6),
            ldz!(7),
        );
        transpose_8x8_zmm_v!(m0, m1, m2, m3, m4, m5, m6, m7);
        ekle_yaz_z32_8_v!(dst, stride, y_off, 0, m0, m1, m2, m3, m4, m5, m6, m7);
    }
    sifirla_katsayi::<1024>(coeff);
}

macro_rules! kelebek_w8_64_yaz {
    ($dst:expr, $stride:expr, $x_off:expr, $coeff:expr, $lo_base:expr, $lo_step:expr, $hi_base:expr, $hi_step:expr,
     $lo0:expr, $lo1:expr, $lo2:expr, $lo3:expr, $lo4:expr, $lo5:expr, $lo6:expr, $lo7:expr,
     $lo8:expr, $lo9:expr, $lo10:expr, $lo11:expr, $lo12:expr, $lo13:expr, $lo14:expr, $lo15:expr,
     $hi0:expr, $hi1:expr, $hi2:expr, $hi3:expr, $hi4:expr, $hi5:expr, $hi6:expr, $hi7:expr,
     $hi8:expr, $hi9:expr, $hi10:expr, $hi11:expr, $hi12:expr, $hi13:expr, $hi14:expr, $hi15:expr) => {{
        let lb = $lo_base;
        let ls = $lo_step;
        let hb = $hi_base;
        let hs = $hi_step;
        let xo = $x_off;
        macro_rules! wl {
            ($k:literal, $ok:expr) => {
                ekle_yaz_w8_kelebek64(
                    $dst,
                    $stride,
                    xo,
                    $k,
                    yukle_i32x4(chunk($coeff, lb + $k * ls)),
                    $ok,
                )
            };
        }
        macro_rules! wh {
            ($k:literal, $ok:expr) => {
                ekle_yaz_w8_kelebek64(
                    $dst,
                    $stride,
                    xo,
                    16 + $k,
                    yukle_i32x4(chunk($coeff, hb + $k * hs)),
                    $ok,
                )
            };
        }
        wl!(0, $lo0);
        wl!(1, $lo1);
        wl!(2, $lo2);
        wl!(3, $lo3);
        wl!(4, $lo4);
        wl!(5, $lo5);
        wl!(6, $lo6);
        wl!(7, $lo7);
        wl!(8, $lo8);
        wl!(9, $lo9);
        wl!(10, $lo10);
        wl!(11, $lo11);
        wl!(12, $lo12);
        wl!(13, $lo13);
        wl!(14, $lo14);
        wl!(15, $lo15);
        wh!(0, $hi0);
        wh!(1, $hi1);
        wh!(2, $hi2);
        wh!(3, $hi3);
        wh!(4, $hi4);
        wh!(5, $hi5);
        wh!(6, $hi6);
        wh!(7, $hi7);
        wh!(8, $hi8);
        wh!(9, $hi9);
        wh!(10, $hi10);
        wh!(11, $hi11);
        wh!(12, $hi12);
        wh!(13, $hi13);
        wh!(14, $hi14);
        wh!(15, $hi15);
    }};
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn idct64_qtr_even_w16(coeff: &mut [i32; 1024], in_base: usize, scr_base: usize) {
    idct16_qtr_v!(x16,
        yukle_i32x8(chunk(coeff, in_base)),
        yukle_i32x8(chunk(coeff, in_base + 4 * 8)),
        yukle_i32x8(chunk(coeff, in_base + 8 * 8)),
        yukle_i32x8(chunk(coeff, in_base + 12 * 8))
        => ee0, ee1, ee2, ee3, ee4, ee5, ee6, ee7, ee8, ee9, ee10, ee11, ee12, ee13, ee14, ee15);
    sakla_ymm16!(
        coeff, scr_base, 8, ee0, ee1, ee2, ee3, ee4, ee5, ee6, ee7, ee8, ee9, ee10, ee11, ee12,
        ee13, ee14, ee15
    );

    idct32_qtr_odd_v!(x16,
        yukle_i32x8(chunk(coeff, in_base + 2 * 8)),
        yukle_i32x8(chunk(coeff, in_base + 6 * 8)),
        yukle_i32x8(chunk(coeff, in_base + 10 * 8)),
        yukle_i32x8(chunk(coeff, in_base + 14 * 8))
        => eo0, eo1, eo2, eo3, eo4, eo5, eo6, eo7, eo8, eo9, eo10, eo11, eo12, eo13, eo14, eo15);
    macro_rules! st_ev {
        ($k:literal, $ok:expr) => {{
            let ek = yukle_i32x8(chunk(coeff, scr_base + $k * 8));
            let ok_v = $ok;
            store_i32x8(chunk_mut(coeff, scr_base + $k * 8), x16::adds(ek, ok_v));
            store_i32x8(
                chunk_mut(coeff, scr_base + 128 + (15 - $k) * 8),
                x16::subs(ek, ok_v),
            );
        }};
    }
    st_ev!(0, eo0);
    st_ev!(1, eo1);
    st_ev!(2, eo2);
    st_ev!(3, eo3);
    st_ev!(4, eo4);
    st_ev!(5, eo5);
    st_ev!(6, eo6);
    st_ev!(7, eo7);
    st_ev!(8, eo8);
    st_ev!(9, eo9);
    st_ev!(10, eo10);
    st_ev!(11, eo11);
    st_ev!(12, eo12);
    st_ev!(13, eo13);
    st_ev!(14, eo14);
    st_ev!(15, eo15);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn idct64_qtr_odd_yaz_w16(
    dst: &mut [u8],
    stride: usize,
    x_off: usize,
    half_x: usize,
    coeff: &[i32; 1024],
    in_base: usize,
    scr_base: usize,
) {
    let sub4 = half_x / 2;
    idct64_odd_qtr_v!(n8,
        yukle_i32x4(chunk(coeff, in_base + 8 + sub4)),
        yukle_i32x4(chunk(coeff, in_base + 3 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, in_base + 5 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, in_base + 7 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, in_base + 9 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, in_base + 11 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, in_base + 13 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, in_base + 15 * 8 + sub4))
        => lo0, lo1, lo2, lo3, lo4, lo5, lo6, lo7, lo8, lo9, lo10, lo11, lo12, lo13, lo14, lo15;
           hi0, hi1, hi2, hi3, hi4, hi5, hi6, hi7, hi8, hi9, hi10, hi11, hi12, hi13, hi14, hi15);
    kelebek_w8_64_yaz!(
        dst,
        stride,
        x_off + half_x,
        coeff,
        scr_base + sub4,
        8,
        scr_base + 128 + sub4,
        8,
        lo0,
        lo1,
        lo2,
        lo3,
        lo4,
        lo5,
        lo6,
        lo7,
        lo8,
        lo9,
        lo10,
        lo11,
        lo12,
        lo13,
        lo14,
        lo15,
        hi0,
        hi1,
        hi2,
        hi3,
        hi4,
        hi5,
        hi6,
        hi7,
        hi8,
        hi9,
        hi10,
        hi11,
        hi12,
        hi13,
        hi14,
        hi15
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn idct64_yaz_w16_qtr_from_coeff(
    dst: &mut [u8],
    stride: usize,
    x_off: usize,
    coeff: &mut [i32; 1024],
    in_base: usize,
    scr_base: usize,
) {
    idct64_qtr_even_w16(coeff, in_base, scr_base);
    idct64_qtr_odd_yaz_w16(dst, stride, x_off, 0, coeff, in_base, scr_base);
    idct64_qtr_odd_yaz_w16(dst, stride, x_off, 8, coeff, in_base, scr_base);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn idct64_tam_even_ee_w16(coeff: &mut [i32; 1024]) {
    idct16_half_v!(x16,
        yukle_i32x8(chunk(coeff, 0)),
        yukle_i32x8(chunk(coeff, 4 * 8)),
        yukle_i32x8(chunk(coeff, 8 * 8)),
        yukle_i32x8(chunk(coeff, 12 * 8)),
        yukle_i32x8(chunk(coeff, 128)),
        yukle_i32x8(chunk(coeff, 128 + 4 * 8)),
        yukle_i32x8(chunk(coeff, 128 + 8 * 8)),
        yukle_i32x8(chunk(coeff, 128 + 12 * 8))
        => ee0, ee1, ee2, ee3, ee4, ee5, ee6, ee7, ee8, ee9, ee10, ee11, ee12, ee13, ee14, ee15);
    sakla_ymm16!(
        coeff, 256, 8, ee0, ee1, ee2, ee3, ee4, ee5, ee6, ee7, ee8, ee9, ee10, ee11, ee12, ee13,
        ee14, ee15
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn idct64_tam_even_w16(coeff: &mut [i32; 1024]) {
    idct64_tam_even_ee_w16(coeff);

    idct32_half_odd_v!(x16,
        yukle_i32x8(chunk(coeff, 2 * 8)),
        yukle_i32x8(chunk(coeff, 6 * 8)),
        yukle_i32x8(chunk(coeff, 10 * 8)),
        yukle_i32x8(chunk(coeff, 14 * 8)),
        yukle_i32x8(chunk(coeff, 128 + 2 * 8)),
        yukle_i32x8(chunk(coeff, 128 + 6 * 8)),
        yukle_i32x8(chunk(coeff, 128 + 10 * 8)),
        yukle_i32x8(chunk(coeff, 128 + 14 * 8))
        => eo0, eo1, eo2, eo3, eo4, eo5, eo6, eo7, eo8, eo9, eo10, eo11, eo12, eo13, eo14, eo15);
    macro_rules! st_ev {
        ($k:literal, $ok:expr) => {{
            let ek = yukle_i32x8(chunk(coeff, 256 + $k * 8));
            let ok_v = $ok;
            store_i32x8(chunk_mut(coeff, 256 + $k * 8), x16::adds(ek, ok_v));
            store_i32x8(chunk_mut(coeff, 384 + (15 - $k) * 8), x16::subs(ek, ok_v));
        }};
    }
    st_ev!(0, eo0);
    st_ev!(1, eo1);
    st_ev!(2, eo2);
    st_ev!(3, eo3);
    st_ev!(4, eo4);
    st_ev!(5, eo5);
    st_ev!(6, eo6);
    st_ev!(7, eo7);
    st_ev!(8, eo8);
    st_ev!(9, eo9);
    st_ev!(10, eo10);
    st_ev!(11, eo11);
    st_ev!(12, eo12);
    st_ev!(13, eo13);
    st_ev!(14, eo14);
    st_ev!(15, eo15);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn idct64_tam_odd_yaz_w16(
    dst: &mut [u8],
    stride: usize,
    x_off: usize,
    half_x: usize,
    coeff: &[i32; 1024],
) {
    let sub4 = half_x / 2;
    idct64_odd_full_v!(n8,
        yukle_i32x4(chunk(coeff, 8 + sub4)),
        yukle_i32x4(chunk(coeff, 3 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, 5 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, 7 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, 9 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, 11 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, 13 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, 15 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, 128 + 8 + sub4)),
        yukle_i32x4(chunk(coeff, 128 + 3 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, 128 + 5 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, 128 + 7 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, 128 + 9 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, 128 + 11 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, 128 + 13 * 8 + sub4)),
        yukle_i32x4(chunk(coeff, 128 + 15 * 8 + sub4))
        => lo0, lo1, lo2, lo3, lo4, lo5, lo6, lo7, lo8, lo9, lo10, lo11, lo12, lo13, lo14, lo15;
           hi0, hi1, hi2, hi3, hi4, hi5, hi6, hi7, hi8, hi9, hi10, hi11, hi12, hi13, hi14, hi15);
    kelebek_w8_64_yaz!(
        dst,
        stride,
        x_off + half_x,
        coeff,
        256 + sub4,
        8,
        384 + sub4,
        8,
        lo0,
        lo1,
        lo2,
        lo3,
        lo4,
        lo5,
        lo6,
        lo7,
        lo8,
        lo9,
        lo10,
        lo11,
        lo12,
        lo13,
        lo14,
        lo15,
        hi0,
        hi1,
        hi2,
        hi3,
        hi4,
        hi5,
        hi6,
        hi7,
        hi8,
        hi9,
        hi10,
        hi11,
        hi12,
        hi13,
        hi14,
        hi15
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn idct64_yaz_w16_tam_from_coeff(
    dst: &mut [u8],
    stride: usize,
    x_off: usize,
    coeff: &mut [i32; 1024],
) {
    idct64_tam_even_w16(coeff);
    idct64_tam_odd_yaz_w16(dst, stride, x_off, 0, coeff);
    idct64_tam_odd_yaz_w16(dst, stride, x_off, 8, coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_16x64_qtr(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    let (mut t0, mut t1, mut t2, mut t3, mut t4, mut t5, mut t6, mut t7) = (
        yukle_i16x16(chunk(coeff, 0)),
        yukle_i16x16(chunk(coeff, 32)),
        yukle_i16x16(chunk(coeff, 2 * 32)),
        yukle_i16x16(chunk(coeff, 3 * 32)),
        yukle_i16x16(chunk(coeff, 4 * 32)),
        yukle_i16x16(chunk(coeff, 5 * 32)),
        yukle_i16x16(chunk(coeff, 6 * 32)),
        yukle_i16x16(chunk(coeff, 7 * 32)),
    );
    let (mut t8, mut t9, mut t10, mut t11, mut t12, mut t13, mut t14, mut t15) = (
        yukle_i16x16(chunk(coeff, 8 * 32)),
        yukle_i16x16(chunk(coeff, 9 * 32)),
        yukle_i16x16(chunk(coeff, 10 * 32)),
        yukle_i16x16(chunk(coeff, 11 * 32)),
        yukle_i16x16(chunk(coeff, 12 * 32)),
        yukle_i16x16(chunk(coeff, 13 * 32)),
        yukle_i16x16(chunk(coeff, 14 * 32)),
        yukle_i16x16(chunk(coeff, 15 * 32)),
    );
    idct16_v!(
        x16, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15
    );
    round_shift16_v!(
        x16, 2, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15
    );
    transpose_16x16_ymm_v!(
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15
    );
    sakla_ymm16!(
        coeff, 0, 8, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15
    );
    idct64_yaz_w16_qtr_from_coeff(dst, stride, 0, coeff, 0, 256);
    sifirla_katsayi::<512>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_16x64_full(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    {
        let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5, mut c6, mut c7) = (
            yukle_z32(chunk(coeff, 0)),
            yukle_z32(chunk(coeff, 32)),
            yukle_z32(chunk(coeff, 2 * 32)),
            yukle_z32(chunk(coeff, 3 * 32)),
            yukle_z32(chunk(coeff, 4 * 32)),
            yukle_z32(chunk(coeff, 5 * 32)),
            yukle_z32(chunk(coeff, 6 * 32)),
            yukle_z32(chunk(coeff, 7 * 32)),
        );
        let (mut c8, mut c9, mut c10, mut c11, mut c12, mut c13, mut c14, mut c15) = (
            yukle_z32(chunk(coeff, 8 * 32)),
            yukle_z32(chunk(coeff, 9 * 32)),
            yukle_z32(chunk(coeff, 10 * 32)),
            yukle_z32(chunk(coeff, 11 * 32)),
            yukle_z32(chunk(coeff, 12 * 32)),
            yukle_z32(chunk(coeff, 13 * 32)),
            yukle_z32(chunk(coeff, 14 * 32)),
            yukle_z32(chunk(coeff, 15 * 32)),
        );
        idct16_v!(
            z32, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15
        );
        round_shift16_v!(
            z32, 2, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15
        );
        sakla_ymm16!(
            coeff,
            0,
            8,
            _mm512_castsi512_si256(c0),
            _mm512_castsi512_si256(c1),
            _mm512_castsi512_si256(c2),
            _mm512_castsi512_si256(c3),
            _mm512_castsi512_si256(c4),
            _mm512_castsi512_si256(c5),
            _mm512_castsi512_si256(c6),
            _mm512_castsi512_si256(c7),
            _mm512_castsi512_si256(c8),
            _mm512_castsi512_si256(c9),
            _mm512_castsi512_si256(c10),
            _mm512_castsi512_si256(c11),
            _mm512_castsi512_si256(c12),
            _mm512_castsi512_si256(c13),
            _mm512_castsi512_si256(c14),
            _mm512_castsi512_si256(c15)
        );
        sakla_ymm16!(
            coeff,
            128,
            8,
            _mm512_extracti64x4_epi64::<1>(c0),
            _mm512_extracti64x4_epi64::<1>(c1),
            _mm512_extracti64x4_epi64::<1>(c2),
            _mm512_extracti64x4_epi64::<1>(c3),
            _mm512_extracti64x4_epi64::<1>(c4),
            _mm512_extracti64x4_epi64::<1>(c5),
            _mm512_extracti64x4_epi64::<1>(c6),
            _mm512_extracti64x4_epi64::<1>(c7),
            _mm512_extracti64x4_epi64::<1>(c8),
            _mm512_extracti64x4_epi64::<1>(c9),
            _mm512_extracti64x4_epi64::<1>(c10),
            _mm512_extracti64x4_epi64::<1>(c11),
            _mm512_extracti64x4_epi64::<1>(c12),
            _mm512_extracti64x4_epi64::<1>(c13),
            _mm512_extracti64x4_epi64::<1>(c14),
            _mm512_extracti64x4_epi64::<1>(c15)
        );
    }

    transpoz_cift_16x16(coeff, 0, 8, 128);
    idct64_yaz_w16_tam_from_coeff(dst, stride, 0, coeff);
    sifirla_katsayi::<512>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn dct_dct_16x64(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    if eob <= 150 {
        dct_dct_16x64_qtr(dst, stride, coeff);
    } else {
        dct_dct_16x64_full(dst, stride, coeff);
    }
}

macro_rules! birlestir_64_serit4_n8 {
    ($sh:tt, $coeff:expr, $sub4:expr, $elo_base:expr, $elo_step:expr, $ehi_base:expr, $ehi_step:expr,
     $lo0:expr, $lo1:expr, $lo2:expr, $lo3:expr, $lo4:expr, $lo5:expr, $lo6:expr, $lo7:expr,
     $lo8:expr, $lo9:expr, $lo10:expr, $lo11:expr, $lo12:expr, $lo13:expr, $lo14:expr, $lo15:expr,
     $hi0:expr, $hi1:expr, $hi2:expr, $hi3:expr, $hi4:expr, $hi5:expr, $hi6:expr, $hi7:expr,
     $hi8:expr, $hi9:expr, $hi10:expr, $hi11:expr, $hi12:expr, $hi13:expr, $hi14:expr, $hi15:expr) => {{
        let s4 = $sub4;
        let elb = $elo_base + s4;
        let els = $elo_step;
        let ehb = $ehi_base + s4;
        let ehs = $ehi_step;
        macro_rules! c4 {
            ($k:literal, $ol:expr, $oh:expr) => {{
                let el = yukle_i32x4(chunk($coeff, elb + $k * els));
                let eh = yukle_i32x4(chunk($coeff, ehb + $k * ehs));
                let ol_v = $ol;
                let oh_v = $oh;
                sakla_i32x4(
                    chunk_mut($coeff, $k * 32 + s4),
                    n8::round_one::<$sh>(n8::adds(el, ol_v)),
                );
                sakla_i32x4(
                    chunk_mut($coeff, $k * 32 + 8 + s4),
                    n8::round_one::<$sh>(n8::adds(eh, oh_v)),
                );
                sakla_i32x4(
                    chunk_mut($coeff, $k * 32 + 16 + s4),
                    n8::round_one::<$sh>(n8::subs(eh, oh_v)),
                );
                sakla_i32x4(
                    chunk_mut($coeff, $k * 32 + 24 + s4),
                    n8::round_one::<$sh>(n8::subs(el, ol_v)),
                );
            }};
        }
        c4!(0, $lo0, $hi15);
        c4!(1, $lo1, $hi14);
        c4!(2, $lo2, $hi13);
        c4!(3, $lo3, $hi12);
        c4!(4, $lo4, $hi11);
        c4!(5, $lo5, $hi10);
        c4!(6, $lo6, $hi9);
        c4!(7, $lo7, $hi8);
        c4!(8, $lo8, $hi7);
        c4!(9, $lo9, $hi6);
        c4!(10, $lo10, $hi5);
        c4!(11, $lo11, $hi4);
        c4!(12, $lo12, $hi3);
        c4!(13, $lo13, $hi2);
        c4!(14, $lo14, $hi1);
        c4!(15, $lo15, $hi0);
    }};
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn pass1_64x16_even_eighth(coeff: &mut [i32; 1024]) {
    let z = _mm256_setzero_si256();
    idct16_qtr_v!(x16,
        yukle_i16x16(chunk(coeff, 0)),
        yukle_i16x16(chunk(coeff, 2 * 32)),
        z, z
        => ee0, ee1, ee2, ee3, ee4, ee5, ee6, ee7, ee8, ee9, ee10, ee11, ee12, ee13, ee14, ee15);
    // Tek sütun yuvaları (`k * 32 + 16 .. +24`) zaten `odd_lo/odd_hi`'ye okunduğu için serbesttir!
    sakla_ymm16!(
        coeff, 16, 32, ee0, ee1, ee2, ee3, ee4, ee5, ee6, ee7, ee8, ee9, ee10, ee11, ee12, ee13,
        ee14, ee15
    );
    idct32_qtr_odd_v!(x16,
        yukle_i16x16(chunk(coeff, 32)),
        yukle_i16x16(chunk(coeff, 3 * 32)),
        z, z
        => eo0, eo1, eo2, eo3, eo4, eo5, eo6, eo7, eo8, eo9, eo10, eo11, eo12, eo13, eo14, eo15);
    macro_rules! st_ev {
        ($k:literal, $ok:expr) => {{
            let ek = yukle_i32x8(chunk(coeff, $k * 32 + 16));
            let ok_v = $ok;
            store_i32x8(chunk_mut(coeff, $k * 32), x16::adds(ek, ok_v));
            store_i32x8(chunk_mut(coeff, $k * 32 + 8), x16::subs(ek, ok_v));
        }};
    }
    st_ev!(0, eo0);
    st_ev!(1, eo1);
    st_ev!(2, eo2);
    st_ev!(3, eo3);
    st_ev!(4, eo4);
    st_ev!(5, eo5);
    st_ev!(6, eo6);
    st_ev!(7, eo7);
    st_ev!(8, eo8);
    st_ev!(9, eo9);
    st_ev!(10, eo10);
    st_ev!(11, eo11);
    st_ev!(12, eo12);
    st_ev!(13, eo13);
    st_ev!(14, eo14);
    st_ev!(15, eo15);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn pass1_64x16_even_qtr(coeff: &mut [i32; 1024]) {
    let z = _mm256_setzero_si256();
    idct16_qtr_v!(x16,
        yukle_i16x16(chunk(coeff, 0)),
        yukle_i16x16(chunk(coeff, 2 * 32)),
        yukle_i16x16(chunk(coeff, 4 * 32)),
        yukle_i16x16(chunk(coeff, 6 * 32))
        => ee0, ee1, ee2, ee3, ee4, ee5, ee6, ee7, ee8, ee9, ee10, ee11, ee12, ee13, ee14, ee15);
    sakla_ymm16!(
        coeff, 16, 32, ee0, ee1, ee2, ee3, ee4, ee5, ee6, ee7, ee8, ee9, ee10, ee11, ee12, ee13,
        ee14, ee15
    );
    idct32_half_odd_v!(x16,
        yukle_i16x16(chunk(coeff, 32)),
        yukle_i16x16(chunk(coeff, 3 * 32)),
        yukle_i16x16(chunk(coeff, 5 * 32)),
        yukle_i16x16(chunk(coeff, 7 * 32)),
        z, z, z, z
        => eo0, eo1, eo2, eo3, eo4, eo5, eo6, eo7, eo8, eo9, eo10, eo11, eo12, eo13, eo14, eo15);
    macro_rules! st_ev {
        ($k:literal, $ok:expr) => {{
            let ek = yukle_i32x8(chunk(coeff, $k * 32 + 16));
            let ok_v = $ok;
            store_i32x8(chunk_mut(coeff, $k * 32), x16::adds(ek, ok_v));
            store_i32x8(chunk_mut(coeff, $k * 32 + 8), x16::subs(ek, ok_v));
        }};
    }
    st_ev!(0, eo0);
    st_ev!(1, eo1);
    st_ev!(2, eo2);
    st_ev!(3, eo3);
    st_ev!(4, eo4);
    st_ev!(5, eo5);
    st_ev!(6, eo6);
    st_ev!(7, eo7);
    st_ev!(8, eo8);
    st_ev!(9, eo9);
    st_ev!(10, eo10);
    st_ev!(11, eo11);
    st_ev!(12, eo12);
    st_ev!(13, eo13);
    st_ev!(14, eo14);
    st_ev!(15, eo15);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn pass1_64x16_even_full_ee(coeff: &mut [i32; 1024]) {
    idct16_half_v!(x16,
        yukle_i16x16(chunk(coeff, 0)),
        yukle_i16x16(chunk(coeff, 2 * 32)),
        yukle_i16x16(chunk(coeff, 4 * 32)),
        yukle_i16x16(chunk(coeff, 6 * 32)),
        yukle_i16x16(chunk(coeff, 8 * 32)),
        yukle_i16x16(chunk(coeff, 10 * 32)),
        yukle_i16x16(chunk(coeff, 12 * 32)),
        yukle_i16x16(chunk(coeff, 14 * 32))
        => ee0, ee1, ee2, ee3, ee4, ee5, ee6, ee7, ee8, ee9, ee10, ee11, ee12, ee13, ee14, ee15);
    sakla_ymm16!(
        coeff, 16, 32, ee0, ee1, ee2, ee3, ee4, ee5, ee6, ee7, ee8, ee9, ee10, ee11, ee12, ee13,
        ee14, ee15
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn pass1_64x16_even_full(coeff: &mut [i32; 1024]) {
    pass1_64x16_even_full_ee(coeff);
    idct32_half_odd_v!(x16,
        yukle_i16x16(chunk(coeff, 32)),
        yukle_i16x16(chunk(coeff, 3 * 32)),
        yukle_i16x16(chunk(coeff, 5 * 32)),
        yukle_i16x16(chunk(coeff, 7 * 32)),
        yukle_i16x16(chunk(coeff, 9 * 32)),
        yukle_i16x16(chunk(coeff, 11 * 32)),
        yukle_i16x16(chunk(coeff, 13 * 32)),
        yukle_i16x16(chunk(coeff, 15 * 32))
        => eo0, eo1, eo2, eo3, eo4, eo5, eo6, eo7, eo8, eo9, eo10, eo11, eo12, eo13, eo14, eo15);
    macro_rules! st_ev {
        ($k:literal, $ok:expr) => {{
            let ek = yukle_i32x8(chunk(coeff, $k * 32 + 16));
            let ok_v = $ok;
            store_i32x8(chunk_mut(coeff, $k * 32), x16::adds(ek, ok_v));
            store_i32x8(chunk_mut(coeff, $k * 32 + 8), x16::subs(ek, ok_v));
        }};
    }
    st_ev!(0, eo0);
    st_ev!(1, eo1);
    st_ev!(2, eo2);
    st_ev!(3, eo3);
    st_ev!(4, eo4);
    st_ev!(5, eo5);
    st_ev!(6, eo6);
    st_ev!(7, eo7);
    st_ev!(8, eo8);
    st_ev!(9, eo9);
    st_ev!(10, eo10);
    st_ev!(11, eo11);
    st_ev!(12, eo12);
    st_ev!(13, eo13);
    st_ev!(14, eo14);
    st_ev!(15, eo15);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn pass1_64x16_odd_eighth_n8(coeff: &mut [i32; 1024], s4: usize, odd_lane: &[i32; 16]) {
    idct64_odd_eighth_v!(n8,
        yukle_i32x4(chunk(odd_lane, 0)),
        yukle_i32x4(chunk(odd_lane, 4)),
        yukle_i32x4(chunk(odd_lane, 8)),
        yukle_i32x4(chunk(odd_lane, 12))
        => lo0, lo1, lo2, lo3, lo4, lo5, lo6, lo7, lo8, lo9, lo10, lo11, lo12, lo13, lo14, lo15;
           hi0, hi1, hi2, hi3, hi4, hi5, hi6, hi7, hi8, hi9, hi10, hi11, hi12, hi13, hi14, hi15);
    birlestir_64_serit4_n8!(
        2, coeff, s4, 0, 32, 8, 32, lo0, lo1, lo2, lo3, lo4, lo5, lo6, lo7, lo8, lo9, lo10, lo11,
        lo12, lo13, lo14, lo15, hi0, hi1, hi2, hi3, hi4, hi5, hi6, hi7, hi8, hi9, hi10, hi11, hi12,
        hi13, hi14, hi15
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn pass1_64x16_odd_qtr_n8(coeff: &mut [i32; 1024], s4: usize, odd_lane: &[i32; 32]) {
    idct64_odd_qtr_v!(n8,
        yukle_i32x4(chunk(odd_lane, 0)),
        yukle_i32x4(chunk(odd_lane, 4)),
        yukle_i32x4(chunk(odd_lane, 8)),
        yukle_i32x4(chunk(odd_lane, 12)),
        yukle_i32x4(chunk(odd_lane, 16)),
        yukle_i32x4(chunk(odd_lane, 20)),
        yukle_i32x4(chunk(odd_lane, 24)),
        yukle_i32x4(chunk(odd_lane, 28))
        => lo0, lo1, lo2, lo3, lo4, lo5, lo6, lo7, lo8, lo9, lo10, lo11, lo12, lo13, lo14, lo15;
           hi0, hi1, hi2, hi3, hi4, hi5, hi6, hi7, hi8, hi9, hi10, hi11, hi12, hi13, hi14, hi15);
    birlestir_64_serit4_n8!(
        2, coeff, s4, 0, 32, 8, 32, lo0, lo1, lo2, lo3, lo4, lo5, lo6, lo7, lo8, lo9, lo10, lo11,
        lo12, lo13, lo14, lo15, hi0, hi1, hi2, hi3, hi4, hi5, hi6, hi7, hi8, hi9, hi10, hi11, hi12,
        hi13, hi14, hi15
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn pass1_64x16_odd_full_n8(coeff: &mut [i32; 1024], s4: usize, odd_lane: &[i32; 64]) {
    idct64_odd_full_v!(n8,
        yukle_i32x4(chunk(odd_lane, 0)),
        yukle_i32x4(chunk(odd_lane, 4)),
        yukle_i32x4(chunk(odd_lane, 8)),
        yukle_i32x4(chunk(odd_lane, 12)),
        yukle_i32x4(chunk(odd_lane, 16)),
        yukle_i32x4(chunk(odd_lane, 20)),
        yukle_i32x4(chunk(odd_lane, 24)),
        yukle_i32x4(chunk(odd_lane, 28)),
        yukle_i32x4(chunk(odd_lane, 32)),
        yukle_i32x4(chunk(odd_lane, 36)),
        yukle_i32x4(chunk(odd_lane, 40)),
        yukle_i32x4(chunk(odd_lane, 44)),
        yukle_i32x4(chunk(odd_lane, 48)),
        yukle_i32x4(chunk(odd_lane, 52)),
        yukle_i32x4(chunk(odd_lane, 56)),
        yukle_i32x4(chunk(odd_lane, 60))
        => lo0, lo1, lo2, lo3, lo4, lo5, lo6, lo7, lo8, lo9, lo10, lo11, lo12, lo13, lo14, lo15;
           hi0, hi1, hi2, hi3, hi4, hi5, hi6, hi7, hi8, hi9, hi10, hi11, hi12, hi13, hi14, hi15);
    birlestir_64_serit4_n8!(
        2, coeff, s4, 0, 32, 8, 32, lo0, lo1, lo2, lo3, lo4, lo5, lo6, lo7, lo8, lo9, lo10, lo11,
        lo12, lo13, lo14, lo15, hi0, hi1, hi2, hi3, hi4, hi5, hi6, hi7, hi8, hi9, hi10, hi11, hi12,
        hi13, hi14, hi15
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_64x16_transpoz_cift(coeff: &mut [i32; 1024], half: usize) {
    let s0 = half * 2;
    let s1 = s0 + 1;
    yukle_ymm16!(coeff, s0 * 8, 32
        => l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15);
    transpose_16x16_ymm_v!(
        l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15
    );
    sakla_ymm16!(
        coeff,
        s0 * 8,
        32,
        l0,
        l1,
        l2,
        l3,
        l4,
        l5,
        l6,
        l7,
        l8,
        l9,
        l10,
        l11,
        l12,
        l13,
        l14,
        l15
    );

    let (mut h0, mut h1, mut h2, mut h3, mut h4, mut h5, mut h6, mut h7) = (
        yukle_i32x8(chunk(coeff, 15 * 32 + s1 * 8)),
        yukle_i32x8(chunk(coeff, 14 * 32 + s1 * 8)),
        yukle_i32x8(chunk(coeff, 13 * 32 + s1 * 8)),
        yukle_i32x8(chunk(coeff, 12 * 32 + s1 * 8)),
        yukle_i32x8(chunk(coeff, 11 * 32 + s1 * 8)),
        yukle_i32x8(chunk(coeff, 10 * 32 + s1 * 8)),
        yukle_i32x8(chunk(coeff, 9 * 32 + s1 * 8)),
        yukle_i32x8(chunk(coeff, 8 * 32 + s1 * 8)),
    );
    let (mut h8, mut h9, mut h10, mut h11, mut h12, mut h13, mut h14, mut h15) = (
        yukle_i32x8(chunk(coeff, 7 * 32 + s1 * 8)),
        yukle_i32x8(chunk(coeff, 6 * 32 + s1 * 8)),
        yukle_i32x8(chunk(coeff, 5 * 32 + s1 * 8)),
        yukle_i32x8(chunk(coeff, 4 * 32 + s1 * 8)),
        yukle_i32x8(chunk(coeff, 3 * 32 + s1 * 8)),
        yukle_i32x8(chunk(coeff, 2 * 32 + s1 * 8)),
        yukle_i32x8(chunk(coeff, 32 + s1 * 8)),
        yukle_i32x8(chunk(coeff, s1 * 8)),
    );
    transpose_16x16_ymm_v!(
        h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15
    );
    sakla_ymm16!(
        coeff,
        s1 * 8,
        32,
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        h8,
        h9,
        h10,
        h11,
        h12,
        h13,
        h14,
        h15
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_64x16_gecis2(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], half_y: bool) {
    for half in 0..2usize {
        dct_dct_64x16_transpoz_cift(coeff, half);
        let s0 = half * 2;
        let s1 = s0 + 1;
        if half_y {
            idct16_half_v!(z32,
                birlestir_zmm(yukle_i32x8(chunk(coeff, s0 * 8)), yukle_i32x8(chunk(coeff, s1 * 8))),
                birlestir_zmm(yukle_i32x8(chunk(coeff, 32 + s0 * 8)), yukle_i32x8(chunk(coeff, 32 + s1 * 8))),
                birlestir_zmm(yukle_i32x8(chunk(coeff, 2 * 32 + s0 * 8)), yukle_i32x8(chunk(coeff, 2 * 32 + s1 * 8))),
                birlestir_zmm(yukle_i32x8(chunk(coeff, 3 * 32 + s0 * 8)), yukle_i32x8(chunk(coeff, 3 * 32 + s1 * 8))),
                birlestir_zmm(yukle_i32x8(chunk(coeff, 4 * 32 + s0 * 8)), yukle_i32x8(chunk(coeff, 4 * 32 + s1 * 8))),
                birlestir_zmm(yukle_i32x8(chunk(coeff, 5 * 32 + s0 * 8)), yukle_i32x8(chunk(coeff, 5 * 32 + s1 * 8))),
                birlestir_zmm(yukle_i32x8(chunk(coeff, 6 * 32 + s0 * 8)), yukle_i32x8(chunk(coeff, 6 * 32 + s1 * 8))),
                birlestir_zmm(yukle_i32x8(chunk(coeff, 7 * 32 + s0 * 8)), yukle_i32x8(chunk(coeff, 7 * 32 + s1 * 8)))
                => r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15);
            let (mut r0, mut r1, mut r2, mut r3, mut r4, mut r5, mut r6, mut r7) =
                (r0, r1, r2, r3, r4, r5, r6, r7);
            let (mut r8, mut r9, mut r10, mut r11, mut r12, mut r13, mut r14, mut r15) =
                (r8, r9, r10, r11, r12, r13, r14, r15);
            round_p2_16_v!(
                z32, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
            );
            ekle_yaz_z32_8_v!(dst, stride, 0, half * 32, r0, r1, r2, r3, r4, r5, r6, r7);
            ekle_yaz_z32_8_v!(
                dst,
                stride,
                8,
                half * 32,
                r8,
                r9,
                r10,
                r11,
                r12,
                r13,
                r14,
                r15
            );
        } else {
            macro_rules! rz {
                ($y:expr) => {
                    birlestir_zmm(
                        yukle_i32x8(chunk(coeff, $y * 32 + s0 * 8)),
                        yukle_i32x8(chunk(coeff, $y * 32 + s1 * 8)),
                    )
                };
            }
            let (mut r0, mut r1, mut r2, mut r3, mut r4, mut r5, mut r6, mut r7) = (
                rz!(0),
                rz!(1),
                rz!(2),
                rz!(3),
                rz!(4),
                rz!(5),
                rz!(6),
                rz!(7),
            );
            let (mut r8, mut r9, mut r10, mut r11, mut r12, mut r13, mut r14, mut r15) = (
                rz!(8),
                rz!(9),
                rz!(10),
                rz!(11),
                rz!(12),
                rz!(13),
                rz!(14),
                rz!(15),
            );
            idct16_v!(
                z32, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
            );
            round_p2_16_v!(
                z32, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15
            );
            ekle_yaz_z32_8_v!(dst, stride, 0, half * 32, r0, r1, r2, r3, r4, r5, r6, r7);
            ekle_yaz_z32_8_v!(
                dst,
                stride,
                8,
                half * 32,
                r8,
                r9,
                r10,
                r11,
                r12,
                r13,
                r14,
                r15
            );
        }
    }
    sifirla_katsayi::<512>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn dct_dct_64x16(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    if eob <= 35 {
        let mut odd_lo = [0i32; 16];
        let mut odd_hi = [0i32; 16];
        for m in 0..4 {
            sakla_i32x4(
                chunk_mut(&mut odd_lo, m * 4),
                yukle_i16x8_128(chunk(coeff, (2 * m + 1) * 16)),
            );
            sakla_i32x4(
                chunk_mut(&mut odd_hi, m * 4),
                yukle_i16x8_128(chunk(coeff, (2 * m + 1) * 16 + 8)),
            );
        }
        pass1_64x16_even_eighth(coeff);
        pass1_64x16_odd_eighth_n8(coeff, 0, &odd_lo);
        pass1_64x16_odd_eighth_n8(coeff, 4, &odd_hi);
        dct_dct_64x16_gecis2(dst, stride, coeff, true);
    } else if eob <= 150 {
        let mut odd_lo = [0i32; 32];
        let mut odd_hi = [0i32; 32];
        for m in 0..8 {
            sakla_i32x4(
                chunk_mut(&mut odd_lo, m * 4),
                yukle_i16x8_128(chunk(coeff, (2 * m + 1) * 16)),
            );
            sakla_i32x4(
                chunk_mut(&mut odd_hi, m * 4),
                yukle_i16x8_128(chunk(coeff, (2 * m + 1) * 16 + 8)),
            );
        }
        pass1_64x16_even_qtr(coeff);
        pass1_64x16_odd_qtr_n8(coeff, 0, &odd_lo);
        pass1_64x16_odd_qtr_n8(coeff, 4, &odd_hi);
        dct_dct_64x16_gecis2(dst, stride, coeff, false);
    } else {
        let mut odd_lo = [0i32; 64];
        let mut odd_hi = [0i32; 64];
        for m in 0..16 {
            sakla_i32x4(
                chunk_mut(&mut odd_lo, m * 4),
                yukle_i16x8_128(chunk(coeff, (2 * m + 1) * 16)),
            );
            sakla_i32x4(
                chunk_mut(&mut odd_hi, m * 4),
                yukle_i16x8_128(chunk(coeff, (2 * m + 1) * 16 + 8)),
            );
        }
        pass1_64x16_even_full(coeff);
        pass1_64x16_odd_full_n8(coeff, 0, &odd_lo);
        pass1_64x16_odd_full_n8(coeff, 4, &odd_hi);
        dct_dct_64x16_gecis2(dst, stride, coeff, false);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_32x64_qtr(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    macro_rules! ld {
        ($k:expr) => {
            x16::mul_r2(yukle_i16x16(chunk(coeff, $k * 32)))
        };
    }
    idct16_half_v!(x16, ld!(0), ld!(2), ld!(4), ld!(6), ld!(8), ld!(10), ld!(12), ld!(14)
        => e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15);
    sakla_ymm16!(
        coeff, 512, 8, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    );
    idct32_half_odd_v!(x16, ld!(1), ld!(3), ld!(5), ld!(7), ld!(9), ld!(11), ld!(13), ld!(15)
        => o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15);
    kelebek_sakla_lo_hi16!(
        x16, 1, coeff, 512, 8, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15
    );

    transpoz_cift_16x16(coeff, 0, 8, 128);
    idct64_yaz_w16_qtr_from_coeff(dst, stride, 0, coeff, 0, 256);
    idct64_yaz_w16_qtr_from_coeff(dst, stride, 16, coeff, 128, 256);
    sifirla_katsayi::<1024>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_32x64_full(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    pass1_ve_transpoz_32x32_z32::<1, true>(coeff);

    for half_x in [0usize, 16] {
        let sub = if half_x == 0 { 0usize } else { 8 };
        for y in 0..16 {
            let vt = yukle_i32x8(chunk(coeff, 512 + y * 16 + sub));
            let vb = yukle_i32x8(chunk(coeff, 768 + y * 16 + sub));
            store_i32x8(chunk_mut(coeff, y * 8), vt);
            store_i32x8(chunk_mut(coeff, 128 + y * 8), vb);
        }
        idct64_yaz_w16_tam_from_coeff(dst, stride, half_x, coeff);
    }
    sifirla_katsayi::<1024>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn dct_dct_32x64(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    if eob < 136 {
        dct_dct_32x64_qtr(dst, stride, coeff);
    } else {
        dct_dct_32x64_full(dst, stride, coeff);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn pass1_64_qtr_even<const RECT2: bool>(coeff: &mut [i32; 1024]) {
    let z = _mm256_setzero_si256();
    macro_rules! ld16 {
        ($off:expr) => {{
            let v = yukle_i16x16(chunk(coeff, $off));
            if RECT2 { x16::mul_r2(v) } else { v }
        }};
    }
    idct16_qtr_v!(x16, ld16!(0), ld16!(4 * 32), ld16!(8 * 32), ld16!(12 * 32)
        => ee0, ee1, ee2, ee3, ee4, ee5, ee6, ee7, ee8, ee9, ee10, ee11, ee12, ee13, ee14, ee15);
    sakla_ymm16!(
        coeff, 512, 8, ee0, ee1, ee2, ee3, ee4, ee5, ee6, ee7, ee8, ee9, ee10, ee11, ee12, ee13,
        ee14, ee15
    );

    idct32_half_odd_v!(x16, ld16!(2 * 32), ld16!(6 * 32), ld16!(10 * 32), ld16!(14 * 32), z, z, z, z
        => eo0, eo1, eo2, eo3, eo4, eo5, eo6, eo7, eo8, eo9, eo10, eo11, eo12, eo13, eo14, eo15);
    macro_rules! st_ev {
        ($k:literal, $ok:expr) => {{
            let ek = yukle_i32x8(chunk(coeff, 512 + $k * 8));
            let ok_v = $ok;
            store_i32x8(chunk_mut(coeff, 512 + $k * 8), x16::adds(ek, ok_v));
            store_i32x8(chunk_mut(coeff, 640 + $k * 8), x16::subs(ek, ok_v));
        }};
    }
    st_ev!(0, eo0);
    st_ev!(1, eo1);
    st_ev!(2, eo2);
    st_ev!(3, eo3);
    st_ev!(4, eo4);
    st_ev!(5, eo5);
    st_ev!(6, eo6);
    st_ev!(7, eo7);
    st_ev!(8, eo8);
    st_ev!(9, eo9);
    st_ev!(10, eo10);
    st_ev!(11, eo11);
    st_ev!(12, eo12);
    st_ev!(13, eo13);
    st_ev!(14, eo14);
    st_ev!(15, eo15);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn pass1_64_qtr_odd_n8<const SHIFT: u32, const RECT2: bool>(coeff: &mut [i32; 1024]) {
    // Önce `lane = 8` (`y = 8..16`) tek sütunlarını `coeff[768..800]` içine kopyala ki `s4 = 0` yazarken ezilmesin!
    for m in 0..8 {
        let v = yukle_i16x8_128(chunk(coeff, (2 * m + 1) * 32 + 8));
        let v = if RECT2 { n8::mul_r2(v) } else { v };
        sakla_i32x4(chunk_mut(coeff, 768 + m * 4), v);
    }
    for m in 0..8 {
        let v = yukle_i16x8_128(chunk(coeff, (2 * m + 1) * 32));
        let v = if RECT2 { n8::mul_r2(v) } else { v };
        sakla_i32x4(chunk_mut(coeff, 800 + m * 4), v);
    }
    for (s4, obase) in [(0usize, 800usize), (4, 768)] {
        idct64_odd_qtr_v!(n8,
            yukle_i32x4(chunk(coeff, obase)),
            yukle_i32x4(chunk(coeff, obase + 4)),
            yukle_i32x4(chunk(coeff, obase + 8)),
            yukle_i32x4(chunk(coeff, obase + 12)),
            yukle_i32x4(chunk(coeff, obase + 16)),
            yukle_i32x4(chunk(coeff, obase + 20)),
            yukle_i32x4(chunk(coeff, obase + 24)),
            yukle_i32x4(chunk(coeff, obase + 28))
            => lo0, lo1, lo2, lo3, lo4, lo5, lo6, lo7, lo8, lo9, lo10, lo11, lo12, lo13, lo14, lo15;
               hi0, hi1, hi2, hi3, hi4, hi5, hi6, hi7, hi8, hi9, hi10, hi11, hi12, hi13, hi14, hi15);
        birlestir_64_serit4_n8!(
            SHIFT, coeff, s4, 512, 8, 640, 8, lo0, lo1, lo2, lo3, lo4, lo5, lo6, lo7, lo8, lo9,
            lo10, lo11, lo12, lo13, lo14, lo15, hi0, hi1, hi2, hi3, hi4, hi5, hi6, hi7, hi8, hi9,
            hi10, hi11, hi12, hi13, hi14, hi15
        );
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline]
fn pass1_64_qtr_to_coeff<const SHIFT: u32, const RECT2: bool>(coeff: &mut [i32; 1024]) {
    pass1_64_qtr_even::<RECT2>(coeff);
    pass1_64_qtr_odd_n8::<SHIFT, RECT2>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn idct64_pass1_z32_even_ee<const RECT2: bool>(coeff: &mut [i32; 1024]) {
    macro_rules! colz {
        ($off:expr) => {{
            let v = yukle_z32(chunk(coeff, $off));
            if RECT2 { z32::mul_r2(v) } else { v }
        }};
    }
    idct16_half_v!(z32,
        colz!(0), colz!(2 * 64), colz!(4 * 64), colz!(6 * 64),
        colz!(8 * 64), colz!(10 * 64), colz!(12 * 64), colz!(14 * 64)
        => ee0, ee1, ee2, ee3, ee4, ee5, ee6, ee7, ee8, ee9, ee10, ee11, ee12, ee13, ee14, ee15);
    // Tüketilmiş `2 * q * 64 .. +32` (`q = 0..8`) yuvalarına `ee0..ee7` (`+0`) ve `ee8..ee15` (`+16`) sakla!
    sakla_zmm(chunk_mut(coeff, 0), ee0);
    sakla_zmm(chunk_mut(coeff, 2 * 64), ee1);
    sakla_zmm(chunk_mut(coeff, 4 * 64), ee2);
    sakla_zmm(chunk_mut(coeff, 6 * 64), ee3);
    sakla_zmm(chunk_mut(coeff, 8 * 64), ee4);
    sakla_zmm(chunk_mut(coeff, 10 * 64), ee5);
    sakla_zmm(chunk_mut(coeff, 12 * 64), ee6);
    sakla_zmm(chunk_mut(coeff, 14 * 64), ee7);
    sakla_zmm(chunk_mut(coeff, 16), ee8);
    sakla_zmm(chunk_mut(coeff, 2 * 64 + 16), ee9);
    sakla_zmm(chunk_mut(coeff, 4 * 64 + 16), ee10);
    sakla_zmm(chunk_mut(coeff, 6 * 64 + 16), ee11);
    sakla_zmm(chunk_mut(coeff, 8 * 64 + 16), ee12);
    sakla_zmm(chunk_mut(coeff, 10 * 64 + 16), ee13);
    sakla_zmm(chunk_mut(coeff, 12 * 64 + 16), ee14);
    sakla_zmm(chunk_mut(coeff, 14 * 64 + 16), ee15);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn duzle_pass1_even_z32(coeff: &mut [i32; 1024], off: usize) {
    let v1 = oku_zmm(chunk(coeff, 2 * 64 + off));
    let v2 = oku_zmm(chunk(coeff, 4 * 64 + off));
    let v3 = oku_zmm(chunk(coeff, 6 * 64 + off));
    let v4 = oku_zmm(chunk(coeff, 8 * 64 + off));
    let v5 = oku_zmm(chunk(coeff, 10 * 64 + off));
    let v6 = oku_zmm(chunk(coeff, 12 * 64 + off));
    let v7 = oku_zmm(chunk(coeff, 14 * 64 + off));
    let v8 = oku_zmm(chunk(coeff, 64 + off));
    let v9 = oku_zmm(chunk(coeff, 3 * 64 + off));
    let v10 = oku_zmm(chunk(coeff, 5 * 64 + off));
    let v11 = oku_zmm(chunk(coeff, 7 * 64 + off));
    let v12 = oku_zmm(chunk(coeff, 9 * 64 + off));
    let v13 = oku_zmm(chunk(coeff, 11 * 64 + off));
    let v14 = oku_zmm(chunk(coeff, 13 * 64 + off));
    sakla_zmm(chunk_mut(coeff, 64 + off), v1);
    sakla_zmm(chunk_mut(coeff, 2 * 64 + off), v2);
    sakla_zmm(chunk_mut(coeff, 3 * 64 + off), v3);
    sakla_zmm(chunk_mut(coeff, 4 * 64 + off), v4);
    sakla_zmm(chunk_mut(coeff, 5 * 64 + off), v5);
    sakla_zmm(chunk_mut(coeff, 6 * 64 + off), v6);
    sakla_zmm(chunk_mut(coeff, 7 * 64 + off), v7);
    sakla_zmm(chunk_mut(coeff, 8 * 64 + off), v8);
    sakla_zmm(chunk_mut(coeff, 9 * 64 + off), v9);
    sakla_zmm(chunk_mut(coeff, 10 * 64 + off), v10);
    sakla_zmm(chunk_mut(coeff, 11 * 64 + off), v11);
    sakla_zmm(chunk_mut(coeff, 12 * 64 + off), v12);
    sakla_zmm(chunk_mut(coeff, 13 * 64 + off), v13);
    sakla_zmm(chunk_mut(coeff, 14 * 64 + off), v14);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn idct64_pass1_z32_even<const RECT2: bool>(coeff: &mut [i32; 1024]) {
    idct64_pass1_z32_even_ee::<RECT2>(coeff);

    macro_rules! colz {
        ($off:expr) => {{
            let v = yukle_z32(chunk(coeff, $off));
            if RECT2 { z32::mul_r2(v) } else { v }
        }};
    }
    idct32_half_odd_v!(z32,
        colz!(64), colz!(3 * 64), colz!(5 * 64), colz!(7 * 64),
        colz!(9 * 64), colz!(11 * 64), colz!(13 * 64), colz!(15 * 64)
        => eo0, eo1, eo2, eo3, eo4, eo5, eo6, eo7, eo8, eo9, eo10, eo11, eo12, eo13, eo14, eo15);
    macro_rules! st_pair {
        ($q:literal, $oa:expr, $ob:expr) => {{
            let eka = oku_zmm(chunk(coeff, 2 * $q * 64));
            let ekb = oku_zmm(chunk(coeff, 2 * $q * 64 + 16));
            let oa_v = $oa;
            let ob_v = $ob;
            sakla_zmm(chunk_mut(coeff, 2 * $q * 64), z32::adds(eka, oa_v));
            sakla_zmm(chunk_mut(coeff, 2 * $q * 64 + 16), z32::subs(eka, oa_v));
            sakla_zmm(chunk_mut(coeff, (2 * $q + 1) * 64), z32::adds(ekb, ob_v));
            sakla_zmm(
                chunk_mut(coeff, (2 * $q + 1) * 64 + 16),
                z32::subs(ekb, ob_v),
            );
        }};
    }
    st_pair!(0, eo0, eo8);
    st_pair!(1, eo1, eo9);
    st_pair!(2, eo2, eo10);
    st_pair!(3, eo3, eo11);
    st_pair!(4, eo4, eo12);
    st_pair!(5, eo5, eo13);
    st_pair!(6, eo6, eo14);
    st_pair!(7, eo7, eo15);
    duzle_pass1_even_z32(coeff, 0);
    duzle_pass1_even_z32(coeff, 16);
}

/// `m * 64 + sub_in` (`sub_in in {32, 48}`) içindeki 16 tek sütunun `8 × i16` şeridini (`lane8 in {0, 8}`)
/// `n8` (`__m128i`) ile işleyip `k * 64 + sub_in + lane8/2` (`lo`) ve `+8` (`hi`) yuvalarına yazar.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn idct64_pass1_odd_n8<const RECT2: bool>(
    coeff: &mut [i32; 1024],
    sub_out: usize,
    s4: usize,
    odd_lane: &[i32; 64],
) {
    let _ = RECT2;
    idct64_odd_full_v!(n8,
        yukle_i32x4(chunk(odd_lane, 0)),
        yukle_i32x4(chunk(odd_lane, 4)),
        yukle_i32x4(chunk(odd_lane, 8)),
        yukle_i32x4(chunk(odd_lane, 12)),
        yukle_i32x4(chunk(odd_lane, 16)),
        yukle_i32x4(chunk(odd_lane, 20)),
        yukle_i32x4(chunk(odd_lane, 24)),
        yukle_i32x4(chunk(odd_lane, 28)),
        yukle_i32x4(chunk(odd_lane, 32)),
        yukle_i32x4(chunk(odd_lane, 36)),
        yukle_i32x4(chunk(odd_lane, 40)),
        yukle_i32x4(chunk(odd_lane, 44)),
        yukle_i32x4(chunk(odd_lane, 48)),
        yukle_i32x4(chunk(odd_lane, 52)),
        yukle_i32x4(chunk(odd_lane, 56)),
        yukle_i32x4(chunk(odd_lane, 60))
        => lo0, lo1, lo2, lo3, lo4, lo5, lo6, lo7, lo8, lo9, lo10, lo11, lo12, lo13, lo14, lo15;
           hi0, hi1, hi2, hi3, hi4, hi5, hi6, hi7, hi8, hi9, hi10, hi11, hi12, hi13, hi14, hi15);
    macro_rules! stp {
        ($k:literal, $ol:expr, $oh:expr) => {{
            sakla_i32x4(chunk_mut(coeff, $k * 64 + sub_out + s4), $ol);
            sakla_i32x4(chunk_mut(coeff, $k * 64 + sub_out + 8 + s4), $oh);
        }};
    }
    stp!(0, lo0, hi15);
    stp!(1, lo1, hi14);
    stp!(2, lo2, hi13);
    stp!(3, lo3, hi12);
    stp!(4, lo4, hi11);
    stp!(5, lo5, hi10);
    stp!(6, lo6, hi9);
    stp!(7, lo7, hi8);
    stp!(8, lo8, hi7);
    stp!(9, lo9, hi6);
    stp!(10, lo10, hi5);
    stp!(11, lo11, hi4);
    stp!(12, lo12, hi3);
    stp!(13, lo13, hi2);
    stp!(14, lo14, hi1);
    stp!(15, lo15, hi0);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn idct64_pass1_z32_tam<const SHIFT: u32, const RECT2: bool>(coeff: &mut [i32; 1024]) {
    idct64_pass1_z32_even::<RECT2>(coeff);

    let mut odd_a = [0i32; 64];
    let mut odd_b = [0i32; 64];
    for sub in [32usize, 48] {
        for m in 0..16 {
            let v0 = yukle_i16x8_128(chunk(coeff, m * 64 + sub));
            let v1 = yukle_i16x8_128(chunk(coeff, m * 64 + sub + 8));
            let v0 = if RECT2 { n8::mul_r2(v0) } else { v0 };
            let v1 = if RECT2 { n8::mul_r2(v1) } else { v1 };
            sakla_i32x4(chunk_mut(&mut odd_a, m * 4), v0);
            sakla_i32x4(chunk_mut(&mut odd_b, m * 4), v1);
        }
        idct64_pass1_odd_n8::<RECT2>(coeff, sub, 0, &odd_a);
        idct64_pass1_odd_n8::<RECT2>(coeff, sub, 4, &odd_b);
    }

    for k in 0..16 {
        let e_lo = oku_zmm(chunk(coeff, k * 64));
        let e_hi = oku_zmm(chunk(coeff, k * 64 + 16));
        let ok_lo = birlestir_zmm(
            yukle_i32x8(chunk(coeff, k * 64 + 32)),
            yukle_i32x8(chunk(coeff, k * 64 + 48)),
        );
        let ok_hi = birlestir_zmm(
            yukle_i32x8(chunk(coeff, k * 64 + 40)),
            yukle_i32x8(chunk(coeff, k * 64 + 56)),
        );
        sakla_zmm(
            chunk_mut(coeff, k * 64),
            z32::round_one::<SHIFT>(z32::adds(e_lo, ok_lo)),
        );
        sakla_zmm(
            chunk_mut(coeff, k * 64 + 16),
            z32::round_one::<SHIFT>(z32::adds(e_hi, ok_hi)),
        );
        sakla_zmm(
            chunk_mut(coeff, k * 64 + 32),
            z32::round_one::<SHIFT>(z32::subs(e_hi, ok_hi)),
        );
        sakla_zmm(
            chunk_mut(coeff, k * 64 + 48),
            z32::round_one::<SHIFT>(z32::subs(e_lo, ok_lo)),
        );
    }
}

/// `idct64_pass1_z32_tam` sonrasında `s` şeridinin (`k*64 + s*16 .. +16`) `16×32` verisini yerinde
/// iki adet `16×16` YMM transpozuyla çevirir (`coeff[y * 64 + s * 16 .. +8]` = `top[y]`, `+8..+16` = `bot[y]`).
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn transpozla_serit16_yerinde(coeff: &mut [i32; 1024], s: usize) {
    let sub = s * 16;
    for half_y in [0usize, 8] {
        let base = sub + half_y;
        yukle_ymm16!(coeff, base, 64 => m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15);
        if s & 1 != 0 {
            core::mem::swap(&mut m0, &mut m15);
            core::mem::swap(&mut m1, &mut m14);
            core::mem::swap(&mut m2, &mut m13);
            core::mem::swap(&mut m3, &mut m12);
            core::mem::swap(&mut m4, &mut m11);
            core::mem::swap(&mut m5, &mut m10);
            core::mem::swap(&mut m6, &mut m9);
            core::mem::swap(&mut m7, &mut m8);
        }
        transpose_16x16_ymm_v!(
            m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15
        );
        sakla_ymm16!(
            coeff, base, 64, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15
        );
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_64x32_qtr(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    pass1_64_qtr_to_coeff::<1, true>(coeff);
    for half in 0..2usize {
        let s0 = half * 2;
        let s1 = s0 + 1;
        yukle_ymm16!(coeff, s0 * 8, 32
            => l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15);
        transpose_16x16_ymm_v!(
            l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15
        );
        sakla_ymm16!(
            coeff, 512, 16, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15
        );

        let (mut h0, mut h1, mut h2, mut h3, mut h4, mut h5, mut h6, mut h7) = (
            yukle_i32x8(chunk(coeff, 15 * 32 + s1 * 8)),
            yukle_i32x8(chunk(coeff, 14 * 32 + s1 * 8)),
            yukle_i32x8(chunk(coeff, 13 * 32 + s1 * 8)),
            yukle_i32x8(chunk(coeff, 12 * 32 + s1 * 8)),
            yukle_i32x8(chunk(coeff, 11 * 32 + s1 * 8)),
            yukle_i32x8(chunk(coeff, 10 * 32 + s1 * 8)),
            yukle_i32x8(chunk(coeff, 9 * 32 + s1 * 8)),
            yukle_i32x8(chunk(coeff, 8 * 32 + s1 * 8)),
        );
        let (mut h8, mut h9, mut h10, mut h11, mut h12, mut h13, mut h14, mut h15) = (
            yukle_i32x8(chunk(coeff, 7 * 32 + s1 * 8)),
            yukle_i32x8(chunk(coeff, 6 * 32 + s1 * 8)),
            yukle_i32x8(chunk(coeff, 5 * 32 + s1 * 8)),
            yukle_i32x8(chunk(coeff, 4 * 32 + s1 * 8)),
            yukle_i32x8(chunk(coeff, 3 * 32 + s1 * 8)),
            yukle_i32x8(chunk(coeff, 2 * 32 + s1 * 8)),
            yukle_i32x8(chunk(coeff, 32 + s1 * 8)),
            yukle_i32x8(chunk(coeff, s1 * 8)),
        );
        transpose_16x16_ymm_v!(
            h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15
        );
        sakla_ymm16!(
            coeff, 520, 16, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15
        );

        idct32_yaz_z32_half_from_coeff(dst, stride, half * 32, coeff, 512, 768);
    }
    sifirla_katsayi::<1024>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_64x32_serit_yaz(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], s: usize) {
    transpozla_serit16_yerinde(coeff, s);
    let sub = s * 16;
    let x_off = s * 16;
    let (mut e0, mut e1, mut e2, mut e3, mut e4, mut e5, mut e6, mut e7) = (
        yukle_i32x8(chunk(coeff, sub)),
        yukle_i32x8(chunk(coeff, 2 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 4 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 6 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 8 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 10 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 12 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 14 * 64 + sub)),
    );
    let (mut e8, mut e9, mut e10, mut e11, mut e12, mut e13, mut e14, mut e15) = (
        yukle_i32x8(chunk(coeff, sub + 8)),
        yukle_i32x8(chunk(coeff, 2 * 64 + sub + 8)),
        yukle_i32x8(chunk(coeff, 4 * 64 + sub + 8)),
        yukle_i32x8(chunk(coeff, 6 * 64 + sub + 8)),
        yukle_i32x8(chunk(coeff, 8 * 64 + sub + 8)),
        yukle_i32x8(chunk(coeff, 10 * 64 + sub + 8)),
        yukle_i32x8(chunk(coeff, 12 * 64 + sub + 8)),
        yukle_i32x8(chunk(coeff, 14 * 64 + sub + 8)),
    );
    idct16_v!(
        x16, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15
    );
    store_i32x8(chunk_mut(coeff, sub), e0);
    store_i32x8(chunk_mut(coeff, 2 * 64 + sub), e1);
    store_i32x8(chunk_mut(coeff, 4 * 64 + sub), e2);
    store_i32x8(chunk_mut(coeff, 6 * 64 + sub), e3);
    store_i32x8(chunk_mut(coeff, 8 * 64 + sub), e4);
    store_i32x8(chunk_mut(coeff, 10 * 64 + sub), e5);
    store_i32x8(chunk_mut(coeff, 12 * 64 + sub), e6);
    store_i32x8(chunk_mut(coeff, 14 * 64 + sub), e7);
    store_i32x8(chunk_mut(coeff, sub + 8), e8);
    store_i32x8(chunk_mut(coeff, 2 * 64 + sub + 8), e9);
    store_i32x8(chunk_mut(coeff, 4 * 64 + sub + 8), e10);
    store_i32x8(chunk_mut(coeff, 6 * 64 + sub + 8), e11);
    store_i32x8(chunk_mut(coeff, 8 * 64 + sub + 8), e12);
    store_i32x8(chunk_mut(coeff, 10 * 64 + sub + 8), e13);
    store_i32x8(chunk_mut(coeff, 12 * 64 + sub + 8), e14);
    store_i32x8(chunk_mut(coeff, 14 * 64 + sub + 8), e15);

    idct32_full_odd_v!(x16,
        yukle_i32x8(chunk(coeff, 64 + sub)),
        yukle_i32x8(chunk(coeff, 3 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 5 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 7 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 9 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 11 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 13 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 15 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 64 + sub + 8)),
        yukle_i32x8(chunk(coeff, 3 * 64 + sub + 8)),
        yukle_i32x8(chunk(coeff, 5 * 64 + sub + 8)),
        yukle_i32x8(chunk(coeff, 7 * 64 + sub + 8)),
        yukle_i32x8(chunk(coeff, 9 * 64 + sub + 8)),
        yukle_i32x8(chunk(coeff, 11 * 64 + sub + 8)),
        yukle_i32x8(chunk(coeff, 13 * 64 + sub + 8)),
        yukle_i32x8(chunk(coeff, 15 * 64 + sub + 8))
        => o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15);

    ekle_yaz_w16_kelebek32(dst, stride, x_off, 0, yukle_i32x8(chunk(coeff, sub)), o0);
    ekle_yaz_w16_kelebek32(
        dst,
        stride,
        x_off,
        1,
        yukle_i32x8(chunk(coeff, 2 * 64 + sub)),
        o1,
    );
    ekle_yaz_w16_kelebek32(
        dst,
        stride,
        x_off,
        2,
        yukle_i32x8(chunk(coeff, 4 * 64 + sub)),
        o2,
    );
    ekle_yaz_w16_kelebek32(
        dst,
        stride,
        x_off,
        3,
        yukle_i32x8(chunk(coeff, 6 * 64 + sub)),
        o3,
    );
    ekle_yaz_w16_kelebek32(
        dst,
        stride,
        x_off,
        4,
        yukle_i32x8(chunk(coeff, 8 * 64 + sub)),
        o4,
    );
    ekle_yaz_w16_kelebek32(
        dst,
        stride,
        x_off,
        5,
        yukle_i32x8(chunk(coeff, 10 * 64 + sub)),
        o5,
    );
    ekle_yaz_w16_kelebek32(
        dst,
        stride,
        x_off,
        6,
        yukle_i32x8(chunk(coeff, 12 * 64 + sub)),
        o6,
    );
    ekle_yaz_w16_kelebek32(
        dst,
        stride,
        x_off,
        7,
        yukle_i32x8(chunk(coeff, 14 * 64 + sub)),
        o7,
    );
    ekle_yaz_w16_kelebek32(
        dst,
        stride,
        x_off,
        8,
        yukle_i32x8(chunk(coeff, sub + 8)),
        o8,
    );
    ekle_yaz_w16_kelebek32(
        dst,
        stride,
        x_off,
        9,
        yukle_i32x8(chunk(coeff, 2 * 64 + sub + 8)),
        o9,
    );
    ekle_yaz_w16_kelebek32(
        dst,
        stride,
        x_off,
        10,
        yukle_i32x8(chunk(coeff, 4 * 64 + sub + 8)),
        o10,
    );
    ekle_yaz_w16_kelebek32(
        dst,
        stride,
        x_off,
        11,
        yukle_i32x8(chunk(coeff, 6 * 64 + sub + 8)),
        o11,
    );
    ekle_yaz_w16_kelebek32(
        dst,
        stride,
        x_off,
        12,
        yukle_i32x8(chunk(coeff, 8 * 64 + sub + 8)),
        o12,
    );
    ekle_yaz_w16_kelebek32(
        dst,
        stride,
        x_off,
        13,
        yukle_i32x8(chunk(coeff, 10 * 64 + sub + 8)),
        o13,
    );
    ekle_yaz_w16_kelebek32(
        dst,
        stride,
        x_off,
        14,
        yukle_i32x8(chunk(coeff, 12 * 64 + sub + 8)),
        o14,
    );
    ekle_yaz_w16_kelebek32(
        dst,
        stride,
        x_off,
        15,
        yukle_i32x8(chunk(coeff, 14 * 64 + sub + 8)),
        o15,
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn dct_dct_64x32(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    if eob < 136 {
        dct_dct_64x32_qtr(dst, stride, coeff);
        return;
    }
    idct64_pass1_z32_tam::<1, true>(coeff);
    for s in 0..4usize {
        dct_dct_64x32_serit_yaz(dst, stride, coeff, s);
    }
    sifirla_katsayi::<1024>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_64x64_qtr_transpoz(coeff: &mut [i32; 1024], s: usize) {
    let base = s * 8;
    yukle_ymm16!(coeff, base, 32 => m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15);
    if s & 1 != 0 {
        core::mem::swap(&mut m0, &mut m15);
        core::mem::swap(&mut m1, &mut m14);
        core::mem::swap(&mut m2, &mut m13);
        core::mem::swap(&mut m3, &mut m12);
        core::mem::swap(&mut m4, &mut m11);
        core::mem::swap(&mut m5, &mut m10);
        core::mem::swap(&mut m6, &mut m9);
        core::mem::swap(&mut m7, &mut m8);
    }
    transpose_16x16_ymm_v!(
        m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15
    );
    sakla_ymm16!(
        coeff, 512, 8, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn dct_dct_64x64_qtr(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024]) {
    pass1_64_qtr_to_coeff::<2, false>(coeff);
    for s in 0..4usize {
        dct_dct_64x64_qtr_transpoz(coeff, s);
        idct64_yaz_w16_qtr_from_coeff(dst, stride, s * 16, coeff, 512, 640);
    }
    sifirla_katsayi::<1024>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn idct64_serit_even_ee_w16(coeff: &[i32; 1024], sub: usize, hi_buf: &mut [i32; 128]) {
    idct16_half_v!(x16,
        yukle_i32x8(chunk(coeff, sub)),
        yukle_i32x8(chunk(coeff, 4 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 8 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 12 * 64 + sub)),
        yukle_i32x8(chunk(coeff, sub + 8)),
        yukle_i32x8(chunk(coeff, 4 * 64 + sub + 8)),
        yukle_i32x8(chunk(coeff, 8 * 64 + sub + 8)),
        yukle_i32x8(chunk(coeff, 12 * 64 + sub + 8))
        => ee0, ee1, ee2, ee3, ee4, ee5, ee6, ee7, ee8, ee9, ee10, ee11, ee12, ee13, ee14, ee15);
    sakla_ymm16!(
        hi_buf, 0, 8, ee0, ee1, ee2, ee3, ee4, ee5, ee6, ee7, ee8, ee9, ee10, ee11, ee12, ee13,
        ee14, ee15
    );
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn idct64_serit_even_w16(coeff: &mut [i32; 1024], sub: usize, hi_buf: &mut [i32; 128]) {
    idct64_serit_even_ee_w16(coeff, sub, hi_buf);

    idct32_half_odd_v!(x16,
        yukle_i32x8(chunk(coeff, 2 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 6 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 10 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 14 * 64 + sub)),
        yukle_i32x8(chunk(coeff, 2 * 64 + sub + 8)),
        yukle_i32x8(chunk(coeff, 6 * 64 + sub + 8)),
        yukle_i32x8(chunk(coeff, 10 * 64 + sub + 8)),
        yukle_i32x8(chunk(coeff, 14 * 64 + sub + 8))
        => eo0, eo1, eo2, eo3, eo4, eo5, eo6, eo7, eo8, eo9, eo10, eo11, eo12, eo13, eo14, eo15);

    // `hi_buf[k*8]` ve `hi_buf[(15-k)*8]` simetrik çiftlerini çakışmasız güncelle!
    macro_rules! st_sym {
        ($k:literal, $m_lo:literal, $off_lo:expr, $ok_lo:expr, $rk:literal, $m_hi:literal, $off_hi:expr, $ok_hi:expr) => {{
            let ek_a = yukle_i32x8(chunk(hi_buf, $k * 8));
            let ek_b = yukle_i32x8(chunk(hi_buf, $rk * 8));
            let oa = $ok_lo;
            let ob = $ok_hi;
            store_i32x8(
                chunk_mut(coeff, 2 * $m_lo * 64 + sub + $off_lo),
                x16::adds(ek_a, oa),
            );
            store_i32x8(
                chunk_mut(coeff, 2 * $m_hi * 64 + sub + $off_hi),
                x16::adds(ek_b, ob),
            );
            store_i32x8(chunk_mut(hi_buf, $rk * 8), x16::subs(ek_a, oa));
            store_i32x8(chunk_mut(hi_buf, $k * 8), x16::subs(ek_b, ob));
        }};
    }
    st_sym!(0, 0, 0, eo0, 15, 7, 8, eo15);
    st_sym!(1, 1, 0, eo1, 14, 6, 8, eo14);
    st_sym!(2, 2, 0, eo2, 13, 5, 8, eo13);
    st_sym!(3, 3, 0, eo3, 12, 4, 8, eo12);
    st_sym!(4, 4, 0, eo4, 11, 3, 8, eo11);
    st_sym!(5, 5, 0, eo5, 10, 2, 8, eo10);
    st_sym!(6, 6, 0, eo6, 9, 1, 8, eo9);
    st_sym!(7, 7, 0, eo7, 8, 0, 8, eo8);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn idct64_serit_odd_yaz_n8(
    dst: &mut [u8],
    stride: usize,
    s: usize,
    half_x: usize,
    coeff: &[i32; 1024],
    hi_buf: &[i32; 128],
) {
    let sub = s * 16;
    let xo = s * 16 + half_x;
    let s4 = half_x / 2;
    idct64_odd_full_v!(n8,
        yukle_i32x4(chunk(coeff, 64 + sub + s4)),
        yukle_i32x4(chunk(coeff, 3 * 64 + sub + s4)),
        yukle_i32x4(chunk(coeff, 5 * 64 + sub + s4)),
        yukle_i32x4(chunk(coeff, 7 * 64 + sub + s4)),
        yukle_i32x4(chunk(coeff, 9 * 64 + sub + s4)),
        yukle_i32x4(chunk(coeff, 11 * 64 + sub + s4)),
        yukle_i32x4(chunk(coeff, 13 * 64 + sub + s4)),
        yukle_i32x4(chunk(coeff, 15 * 64 + sub + s4)),
        yukle_i32x4(chunk(coeff, 64 + sub + 8 + s4)),
        yukle_i32x4(chunk(coeff, 3 * 64 + sub + 8 + s4)),
        yukle_i32x4(chunk(coeff, 5 * 64 + sub + 8 + s4)),
        yukle_i32x4(chunk(coeff, 7 * 64 + sub + 8 + s4)),
        yukle_i32x4(chunk(coeff, 9 * 64 + sub + 8 + s4)),
        yukle_i32x4(chunk(coeff, 11 * 64 + sub + 8 + s4)),
        yukle_i32x4(chunk(coeff, 13 * 64 + sub + 8 + s4)),
        yukle_i32x4(chunk(coeff, 15 * 64 + sub + 8 + s4))
        => lo0, lo1, lo2, lo3, lo4, lo5, lo6, lo7, lo8, lo9, lo10, lo11, lo12, lo13, lo14, lo15;
           hi0, hi1, hi2, hi3, hi4, hi5, hi6, hi7, hi8, hi9, hi10, hi11, hi12, hi13, hi14, hi15);

    ekle_yaz_w8_kelebek64(dst, stride, xo, 0, yukle_i32x4(chunk(coeff, sub + s4)), lo0);
    ekle_yaz_w8_kelebek64(
        dst,
        stride,
        xo,
        1,
        yukle_i32x4(chunk(coeff, 2 * 64 + sub + s4)),
        lo1,
    );
    ekle_yaz_w8_kelebek64(
        dst,
        stride,
        xo,
        2,
        yukle_i32x4(chunk(coeff, 4 * 64 + sub + s4)),
        lo2,
    );
    ekle_yaz_w8_kelebek64(
        dst,
        stride,
        xo,
        3,
        yukle_i32x4(chunk(coeff, 6 * 64 + sub + s4)),
        lo3,
    );
    ekle_yaz_w8_kelebek64(
        dst,
        stride,
        xo,
        4,
        yukle_i32x4(chunk(coeff, 8 * 64 + sub + s4)),
        lo4,
    );
    ekle_yaz_w8_kelebek64(
        dst,
        stride,
        xo,
        5,
        yukle_i32x4(chunk(coeff, 10 * 64 + sub + s4)),
        lo5,
    );
    ekle_yaz_w8_kelebek64(
        dst,
        stride,
        xo,
        6,
        yukle_i32x4(chunk(coeff, 12 * 64 + sub + s4)),
        lo6,
    );
    ekle_yaz_w8_kelebek64(
        dst,
        stride,
        xo,
        7,
        yukle_i32x4(chunk(coeff, 14 * 64 + sub + s4)),
        lo7,
    );
    ekle_yaz_w8_kelebek64(
        dst,
        stride,
        xo,
        8,
        yukle_i32x4(chunk(coeff, sub + 8 + s4)),
        lo8,
    );
    ekle_yaz_w8_kelebek64(
        dst,
        stride,
        xo,
        9,
        yukle_i32x4(chunk(coeff, 2 * 64 + sub + 8 + s4)),
        lo9,
    );
    ekle_yaz_w8_kelebek64(
        dst,
        stride,
        xo,
        10,
        yukle_i32x4(chunk(coeff, 4 * 64 + sub + 8 + s4)),
        lo10,
    );
    ekle_yaz_w8_kelebek64(
        dst,
        stride,
        xo,
        11,
        yukle_i32x4(chunk(coeff, 6 * 64 + sub + 8 + s4)),
        lo11,
    );
    ekle_yaz_w8_kelebek64(
        dst,
        stride,
        xo,
        12,
        yukle_i32x4(chunk(coeff, 8 * 64 + sub + 8 + s4)),
        lo12,
    );
    ekle_yaz_w8_kelebek64(
        dst,
        stride,
        xo,
        13,
        yukle_i32x4(chunk(coeff, 10 * 64 + sub + 8 + s4)),
        lo13,
    );
    ekle_yaz_w8_kelebek64(
        dst,
        stride,
        xo,
        14,
        yukle_i32x4(chunk(coeff, 12 * 64 + sub + 8 + s4)),
        lo14,
    );
    ekle_yaz_w8_kelebek64(
        dst,
        stride,
        xo,
        15,
        yukle_i32x4(chunk(coeff, 14 * 64 + sub + 8 + s4)),
        lo15,
    );

    macro_rules! whs {
        ($k:literal, $ok:expr) => {
            ekle_yaz_w8_kelebek64(
                dst,
                stride,
                xo,
                16 + $k,
                yukle_i32x4(chunk(hi_buf, $k * 8 + s4)),
                $ok,
            )
        };
    }
    whs!(0, hi0);
    whs!(1, hi1);
    whs!(2, hi2);
    whs!(3, hi3);
    whs!(4, hi4);
    whs!(5, hi5);
    whs!(6, hi6);
    whs!(7, hi7);
    whs!(8, hi8);
    whs!(9, hi9);
    whs!(10, hi10);
    whs!(11, hi11);
    whs!(12, hi12);
    whs!(13, hi13);
    whs!(14, hi14);
    whs!(15, hi15);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
fn dct_dct_64x64(dst: &mut [u8], stride: usize, coeff: &mut [i32; 1024], eob: usize) {
    if eob < 136 {
        dct_dct_64x64_qtr(dst, stride, coeff);
        return;
    }

    idct64_pass1_z32_tam::<2, false>(coeff);
    let mut hi_buf = [0i32; 128];
    for s in 0..4usize {
        transpozla_serit16_yerinde(coeff, s);
        idct64_serit_even_w16(coeff, s * 16, &mut hi_buf);
        idct64_serit_odd_yaz_n8(dst, stride, s, 0, coeff, &hi_buf);
        idct64_serit_odd_yaz_n8(dst, stride, s, 8, coeff, &hi_buf);
    }
    sifirla_katsayi::<1024>(coeff);
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
#[inline(never)]
fn itx8_buyuk_avx512_cekirdek<const TX: u8, const TXTP: u8>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32; 1024],
    eob: usize,
) {
    let tx = const { TxSize::from_index(TX as usize).unwrap() };
    let txtp = const { TxType::from_index(TXTP as usize).unwrap() };

    if matches!(txtp, TxType::DctDct) && eob == 0 {
        match tx {
            TxSize::Rtx8x32 => dc_only::<8, 32, 2, false>(dst, dst_stride, coeff),
            TxSize::Rtx32x8 => dc_only::<32, 8, 2, false>(dst, dst_stride, coeff),
            TxSize::Rtx16x32 => dc_only::<16, 32, 1, true>(dst, dst_stride, coeff),
            TxSize::Rtx32x16 => dc_only::<32, 16, 1, true>(dst, dst_stride, coeff),
            TxSize::Tx32x32 => dc_only::<32, 32, 2, false>(dst, dst_stride, coeff),
            TxSize::Rtx16x64 => dc_only::<16, 64, 2, false>(dst, dst_stride, coeff),
            TxSize::Rtx64x16 => dc_only::<64, 16, 2, false>(dst, dst_stride, coeff),
            TxSize::Rtx32x64 => dc_only::<32, 64, 1, true>(dst, dst_stride, coeff),
            TxSize::Rtx64x32 => dc_only::<64, 32, 1, true>(dst, dst_stride, coeff),
            TxSize::Tx64x64 => dc_only::<64, 64, 2, false>(dst, dst_stride, coeff),
            _ => unreachable!(),
        }
        return;
    }

    match (tx, txtp) {
        (TxSize::Rtx8x32, TxType::DctDct) => dct_dct_8x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx8x32, TxType::Idtx) => idtx_8x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x8, TxType::DctDct) => dct_dct_32x8(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x8, TxType::Idtx) => idtx_32x8(dst, dst_stride, coeff, eob),
        (TxSize::Rtx16x32, TxType::DctDct) => dct_dct_16x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx16x32, TxType::Idtx) => idtx_16x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x16, TxType::DctDct) => dct_dct_32x16(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x16, TxType::Idtx) => idtx_32x16(dst, dst_stride, coeff, eob),
        (TxSize::Tx32x32, TxType::DctDct) => dct_dct_32x32(dst, dst_stride, coeff, eob),
        (TxSize::Tx32x32, TxType::Idtx) => idtx_32x32(dst, dst_stride, coeff, eob),
        (TxSize::Rtx16x64, TxType::DctDct) => dct_dct_16x64(dst, dst_stride, coeff, eob),
        (TxSize::Rtx64x16, TxType::DctDct) => dct_dct_64x16(dst, dst_stride, coeff, eob),
        (TxSize::Rtx32x64, TxType::DctDct) => dct_dct_32x64(dst, dst_stride, coeff, eob),
        (TxSize::Rtx64x32, TxType::DctDct) => dct_dct_64x32(dst, dst_stride, coeff, eob),
        (TxSize::Tx64x64, TxType::DctDct) => dct_dct_64x64(dst, dst_stride, coeff, eob),
        _ => unreachable!(),
    }
}

/// 8 bpc AVX512ICL büyük (`max(w, h) ∈ {32, 64}`) `ItxFn` giriş sarmalayıcısı (monomorfize `(TX, TXTP)`).
fn itx8_buyuk_avx512_giris<const TX: u8, const TXTP: u8>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
    bit_depth: BitDepth,
) {
    if bit_depth != BitDepth::Eight
        || tx as u8 != TX
        || txtp as u8 != TXTP
        || matches!(txtp, TxType::WhtWht)
        || !itx8_sozlesmesi(dst.len(), dst_stride, coeff.len(), eob, tx, txtp)
    {
        itx_skaler_devret::<8>(dst, dst_stride, coeff, eob, tx, txtp);
        return;
    }
    let coeff1024 = chunk_mut::<i32, 1024>(coeff, 0);
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Avx512IclToken` ile verilir
    // (`itx8_buyuk_avx512_fn`); belirteç AVX512ICL (F/BW/VL/VBMI/VNNI) tespiti + maske
    // izninin kanıtıdır. (2) sınır: `itx8_sozlesmesi` `dst`'nin `(h − 1)·dst_stride + w`
    // baytı ve `coeff`'un `1024` öğeyi kapsadığını garanti eder; çekirdek içindeki bütün
    // erişimler sabit boyutlu `chunk`/`chunk_mut` dizileri üzerindendir. (3) hizalama/takma
    // ad: `dst` ve `coeff1024` ayrık `&mut` referanslardır, yalnız hizasız erişim kullanılır.
    unsafe {
        itx8_buyuk_avx512_cekirdek::<TX, TXTP>(dst, dst_stride, coeff1024, eob);
    }
}

/// 15 adet `max(w, h) ∈ {32, 64}` 8 bpc AVX512ICL `ItxFn` tablosu (yalnız [`Avx512IclToken`] ile).
#[must_use]
pub fn itx8_buyuk_avx512_fn(_token: Avx512IclToken, tx: TxSize, txtp: TxType) -> Option<ItxFn> {
    match (tx, txtp) {
        (TxSize::Tx32x32, TxType::DctDct) => Some(itx8_buyuk_avx512_giris::<3, 0>),
        (TxSize::Tx32x32, TxType::Idtx) => Some(itx8_buyuk_avx512_giris::<3, 9>),
        (TxSize::Tx64x64, TxType::DctDct) => Some(itx8_buyuk_avx512_giris::<4, 0>),
        (TxSize::Rtx16x32, TxType::DctDct) => Some(itx8_buyuk_avx512_giris::<9, 0>),
        (TxSize::Rtx16x32, TxType::Idtx) => Some(itx8_buyuk_avx512_giris::<9, 9>),
        (TxSize::Rtx32x16, TxType::DctDct) => Some(itx8_buyuk_avx512_giris::<10, 0>),
        (TxSize::Rtx32x16, TxType::Idtx) => Some(itx8_buyuk_avx512_giris::<10, 9>),
        (TxSize::Rtx32x64, TxType::DctDct) => Some(itx8_buyuk_avx512_giris::<11, 0>),
        (TxSize::Rtx64x32, TxType::DctDct) => Some(itx8_buyuk_avx512_giris::<12, 0>),
        (TxSize::Rtx8x32, TxType::DctDct) => Some(itx8_buyuk_avx512_giris::<15, 0>),
        (TxSize::Rtx8x32, TxType::Idtx) => Some(itx8_buyuk_avx512_giris::<15, 9>),
        (TxSize::Rtx32x8, TxType::DctDct) => Some(itx8_buyuk_avx512_giris::<16, 0>),
        (TxSize::Rtx32x8, TxType::Idtx) => Some(itx8_buyuk_avx512_giris::<16, 9>),
        (TxSize::Rtx16x64, TxType::DctDct) => Some(itx8_buyuk_avx512_giris::<17, 0>),
        (TxSize::Rtx64x16, TxType::DctDct) => Some(itx8_buyuk_avx512_giris::<18, 0>),
        _ => None,
    }
}
