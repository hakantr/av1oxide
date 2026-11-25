//! x86_64 AVX2 MSAC 16-sembollü entropi çözücü çekirdeği (P47 / `F4-MSAC`).
//!
//! Referans: `dav1d` `src/x86/msac.h` + `src/x86/msac.asm:616-671` (`99f3354`).
//! `dav1d_msac_decode_symbol_adapt16_avx2` (`x86-msac-0005`) işlevini tek 256-bit
//! YMM yazmacı üzerinden saf `core::arch::x86_64` AVX2 intrinsics ile gerçekler
//! (`objdump -d`: bu modülde `0` EVEX komutu).

#![allow(unsafe_code)]

use crate::cpu::Avx2Token;
use crate::dsp::x86::mem::{load_u16x16, store_u16x16};
use crate::dsp::x86::msac::{MASK_LT, MIN_PROB};
use crate::entropy::msac::{MsacContext, ctx_norm_fast};
use core::arch::x86_64::{
    _mm_cvtsi32_si128, _mm256_add_epi16, _mm256_and_si256, _mm256_avg_epu16, _mm256_cmpeq_epi16,
    _mm256_max_epu16, _mm256_movemask_epi8, _mm256_mulhi_epu16, _mm256_set1_epi16,
    _mm256_slli_epi16, _mm256_sra_epi16, _mm256_srli_epi16, _mm256_sub_epi16,
};

#[target_feature(enable = "avx2")]
#[inline]
fn adapt16_avx2(s: &mut MsacContext<'_>, cdf: &mut [u16; 16], n_symbols: usize) -> usize {
    let nsym = n_symbols.min(15);
    let m0 = load_u16x16(cdf);
    let r_vec = _mm256_set1_epi16((s.rng & 0xff00) as i16);
    let c_vec = _mm256_set1_epi16((s.dif >> 48) as i16);
    let min_p = load_u16x16(&MIN_PROB[nsym]);
    let q = _mm256_slli_epi16(_mm256_srli_epi16(m0, 6), 7);
    let v = _mm256_add_epi16(_mm256_mulhi_epu16(q, r_vec), min_p);
    let eq = _mm256_cmpeq_epi16(_mm256_max_epu16(v, c_vec), c_vec);
    let mask = _mm256_movemask_epi8(eq) as u32;
    let val = ((mask.trailing_zeros() >> 1) as usize).min(nsym);

    let mut v_arr = [0u16; 16];
    store_u16x16(&mut v_arr, v);
    let v_val = u32::from(v_arr[val]);
    let u_val = if val == 0 {
        s.rng
    } else {
        u32::from(v_arr[val - 1])
    };

    if s.allow_update_cdf {
        let count = u32::from(cdf[nsym]);
        let rate = 4 + (count >> 4) + u32::from(nsym > 2);
        let rate_v = _mm_cvtsi32_si128(rate as i32);
        let ones = _mm256_set1_epi16(-1);
        let target = _mm256_avg_epu16(ones, eq);
        let diff = _mm256_sra_epi16(_mm256_sub_epi16(target, m0), rate_v);
        let delta = _mm256_sub_epi16(diff, eq);
        let mask_lt = load_u16x16(&MASK_LT[nsym]);
        let m0_new = _mm256_add_epi16(m0, _mm256_and_si256(delta, mask_lt));
        store_u16x16(cdf, m0_new);
        cdf[nsym] = (count + u32::from(count < 32)) as u16;
    }

    let new_dif = s.dif.wrapping_sub(u64::from(v_val) << 48);
    let new_rng = u_val.wrapping_sub(v_val);
    ctx_norm_fast(
        s.buf, &mut s.pos, &mut s.dif, &mut s.rng, &mut s.cnt, new_dif, new_rng,
    );
    val
}

/// `dav1d_msac_decode_symbol_adapt16_avx2` (`x86-msac-0005`: `1 <= n_symbols <= 15`, `cdf.len() >= 16`).
#[inline(always)]
pub fn decode_symbol_adapt16_avx2(
    _token: Avx2Token,
    s: &mut MsacContext<'_>,
    cdf: &mut [u16; 16],
    n_symbols: usize,
) -> usize {
    // SAFETY: (1) ISA: `_token` (`Avx2Token`) bu işlemcide AVX2'nin
    // doğrulandığını ve maskeyle kapatılmadığını kanıtlar; `adapt16_avx2`
    // yalnız AVX2 `#[target_feature(enable = "avx2")]` ister.
    // (2) sınır: `cdf` sabit boyutlu `&mut [u16; 16]` dizisidir ve `nsym = n_symbols.min(15) <= 15`
    // ile tüm dizinlemeler `[0..16)` aralığındadır.
    // (3) hizalama/takma ad: yükleme ve saklama `load_u16x16`/`store_u16x16` ile
    // hizasız ve tek `&mut` referans üzerinden yapılır.
    unsafe { adapt16_avx2(s, cdf, n_symbols) }
}
