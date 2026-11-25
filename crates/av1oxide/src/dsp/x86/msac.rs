//! x86_64 SSE2 MSAC entropi çözücü çekirdekleri (P47 / `F4-MSAC`).
//!
//! Referans: `dav1d` `src/x86/msac.h` + `src/x86/msac.asm` (`99f3354`).
//! Bu modül 7 SSE2 MSAC işlevini (`decode_symbol_adapt4_sse2`,
//! `decode_symbol_adapt8_sse2`, `decode_symbol_adapt16_sse2`,
//! `decode_hi_tok_sse2`, `decode_bool_adapt_sse2`, `decode_bool_equi_sse2`,
//! `decode_bool_sse2`) saf `core::arch::x86_64` SSE2 intrinsics ile gerçekler.
//! P34 modül izolasyonu gereği AVX2 varyantı (`decode_symbol_adapt16_avx2`)
//! ayrı `dsp::x86::msac_avx2` modülündedir (`objdump -d`: bu modülde `0`
//! SSSE3+/VEX/EVEX komutu).

#![allow(unsafe_code)]

use crate::cpu::Sse2Token;
use crate::dsp::x86::mem::{chunk, chunk_mut, load_u16x4, load_u16x8, store_u16x4, store_u16x8};
use crate::entropy::msac::{MsacContext, ctx_norm_fast, ctx_refill_fast};
use core::arch::x86_64::{
    _mm_add_epi16, _mm_and_si128, _mm_avg_epu16, _mm_cmpeq_epi16, _mm_cvtsi32_si128,
    _mm_cvtsi128_si64, _mm_movemask_epi8, _mm_mulhi_epu16, _mm_packs_epi16, _mm_set1_epi16,
    _mm_setzero_si128, _mm_slli_epi16, _mm_sra_epi16, _mm_srli_epi16, _mm_sub_epi16,
    _mm_subs_epu16,
};

/// `n_symbols` (`0..=15`) için `EC_MIN_PROB * (n_symbols - i)` (`i < n_symbols`,
/// `i >= n_symbols` için `0`). `i == n_symbols` şeridinde `min_prob == 0` ve
/// `cdf[n_symbols] <= 32` olduğundan `v[n_symbols] == 0` olur; `c >= 0` her
/// zaman doğru olduğu için `tzcnt` en geç `n_symbols` şeridinde durur.
pub(crate) const MIN_PROB: [[u16; 16]; 16] = {
    let mut t = [[0u16; 16]; 16];
    let mut n = 1usize;
    while n <= 15 {
        let mut i = 0usize;
        while i < n {
            t[n][i] = ((n - i) * 4) as u16;
            i += 1;
        }
        n += 1;
    }
    t
};

/// `n_symbols` (`0..=15`) için aktif CDF olasılık şeritlerinin maskesi
/// (`i < n_symbols ? 0xFFFF : 0x0000`). `cdf[n_symbols]` sayaç hücresinin ve
/// `cdf[n_symbols + 1 ..]` dolgu hücrelerinin vektörel güncellemede
/// kirlenmesini önler (`msac.c` / `coef_oracle` / `msacdump` tam CDF paritesi).
pub(crate) const MASK_LT: [[u16; 16]; 16] = {
    let mut t = [[0u16; 16]; 16];
    let mut n = 1usize;
    while n <= 15 {
        let mut i = 0usize;
        while i < n {
            t[n][i] = 0xFFFF;
            i += 1;
        }
        n += 1;
    }
    t
};

#[target_feature(enable = "sse2")]
#[inline]
fn adapt4_sse2(s: &mut MsacContext<'_>, cdf: &mut [u16; 4], n_symbols: usize) -> usize {
    let nsym = n_symbols.min(3);
    let m0 = load_u16x4(cdf);
    let r_vec = _mm_set1_epi16((s.rng & 0xff00) as i16);
    let c_vec = _mm_set1_epi16((s.dif >> 48) as i16);
    let min_p = load_u16x4(chunk::<u16, 4>(&MIN_PROB[nsym], 0));
    let q = _mm_slli_epi16(_mm_srli_epi16(m0, 6), 7);
    let v = _mm_add_epi16(_mm_mulhi_epu16(q, r_vec), min_p);
    let eq = _mm_cmpeq_epi16(_mm_subs_epu16(v, c_vec), _mm_setzero_si128());
    let mask = _mm_movemask_epi8(eq) as u32;
    let val = ((mask.trailing_zeros() >> 1) as usize).min(nsym);

    let v64 = _mm_cvtsi128_si64(v) as u64;
    let (u_val, v_val) = if val == 0 {
        (s.rng, (v64 & 0xffff) as u32)
    } else {
        let pair = (v64 >> ((val - 1) * 16)) as u32;
        (pair & 0xffff, pair >> 16)
    };

    if s.allow_update_cdf {
        let count = u32::from(cdf[nsym]);
        let rate = 4 + (count >> 4) + u32::from(nsym > 2);
        let rate_v = _mm_cvtsi32_si128(rate as i32);
        let ones = _mm_set1_epi16(-1);
        let target = _mm_avg_epu16(ones, eq);
        let diff = _mm_sra_epi16(_mm_sub_epi16(target, m0), rate_v);
        let delta = _mm_sub_epi16(diff, eq);
        let mask_lt = load_u16x4(chunk::<u16, 4>(&MASK_LT[nsym], 0));
        let m0_new = _mm_add_epi16(m0, _mm_and_si128(delta, mask_lt));
        store_u16x4(cdf, m0_new);
        cdf[nsym] = (count + u32::from(count < 32)) as u16;
    }

    let new_dif = s.dif.wrapping_sub(u64::from(v_val) << 48);
    let new_rng = u_val.wrapping_sub(v_val);
    ctx_norm_fast(
        s.buf, &mut s.pos, &mut s.dif, &mut s.rng, &mut s.cnt, new_dif, new_rng,
    );
    val
}

#[target_feature(enable = "sse2")]
#[inline]
fn adapt8_sse2(s: &mut MsacContext<'_>, cdf: &mut [u16; 8], n_symbols: usize) -> usize {
    let nsym = n_symbols.min(7);
    let m0 = load_u16x8(cdf);
    let r_vec = _mm_set1_epi16((s.rng & 0xff00) as i16);
    let c_vec = _mm_set1_epi16((s.dif >> 48) as i16);
    let min_p = load_u16x8(chunk::<u16, 8>(&MIN_PROB[nsym], 0));
    let q = _mm_slli_epi16(_mm_srli_epi16(m0, 6), 7);
    let v = _mm_add_epi16(_mm_mulhi_epu16(q, r_vec), min_p);
    let eq = _mm_cmpeq_epi16(_mm_subs_epu16(v, c_vec), _mm_setzero_si128());
    let mask = _mm_movemask_epi8(eq) as u32;
    let val = ((mask.trailing_zeros() >> 1) as usize).min(nsym);

    let mut v_arr = [0u16; 8];
    store_u16x8(&mut v_arr, v);
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
        let ones = _mm_set1_epi16(-1);
        let target = _mm_avg_epu16(ones, eq);
        let diff = _mm_sra_epi16(_mm_sub_epi16(target, m0), rate_v);
        let delta = _mm_sub_epi16(diff, eq);
        let mask_lt = load_u16x8(chunk::<u16, 8>(&MASK_LT[nsym], 0));
        let m0_new = _mm_add_epi16(m0, _mm_and_si128(delta, mask_lt));
        store_u16x8(cdf, m0_new);
        cdf[nsym] = (count + u32::from(count < 32)) as u16;
    }

    let new_dif = s.dif.wrapping_sub(u64::from(v_val) << 48);
    let new_rng = u_val.wrapping_sub(v_val);
    ctx_norm_fast(
        s.buf, &mut s.pos, &mut s.dif, &mut s.rng, &mut s.cnt, new_dif, new_rng,
    );
    val
}

#[target_feature(enable = "sse2")]
#[inline]
fn adapt16_sse2(s: &mut MsacContext<'_>, cdf: &mut [u16; 16], n_symbols: usize) -> usize {
    let nsym = n_symbols.min(15);
    let m0 = load_u16x8(chunk::<u16, 8>(cdf, 0));
    let m1 = load_u16x8(chunk::<u16, 8>(cdf, 8));
    let r_vec = _mm_set1_epi16((s.rng & 0xff00) as i16);
    let c_vec = _mm_set1_epi16((s.dif >> 48) as i16);
    let min_p0 = load_u16x8(chunk::<u16, 8>(&MIN_PROB[nsym], 0));
    let min_p1 = load_u16x8(chunk::<u16, 8>(&MIN_PROB[nsym], 8));
    let q0 = _mm_slli_epi16(_mm_srli_epi16(m0, 6), 7);
    let q1 = _mm_slli_epi16(_mm_srli_epi16(m1, 6), 7);
    let v0 = _mm_add_epi16(_mm_mulhi_epu16(q0, r_vec), min_p0);
    let v1 = _mm_add_epi16(_mm_mulhi_epu16(q1, r_vec), min_p1);
    let zero = _mm_setzero_si128();
    let eq0 = _mm_cmpeq_epi16(_mm_subs_epu16(v0, c_vec), zero);
    let eq1 = _mm_cmpeq_epi16(_mm_subs_epu16(v1, c_vec), zero);
    let packed = _mm_packs_epi16(eq0, eq1);
    let mask = _mm_movemask_epi8(packed) as u32;
    let val = (mask.trailing_zeros() as usize).min(nsym);

    let mut v_arr = [0u16; 16];
    store_u16x8(chunk_mut::<u16, 8>(&mut v_arr, 0), v0);
    store_u16x8(chunk_mut::<u16, 8>(&mut v_arr, 8), v1);
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
        let ones = _mm_set1_epi16(-1);
        let target0 = _mm_avg_epu16(ones, eq0);
        let target1 = _mm_avg_epu16(ones, eq1);
        let diff0 = _mm_sra_epi16(_mm_sub_epi16(target0, m0), rate_v);
        let diff1 = _mm_sra_epi16(_mm_sub_epi16(target1, m1), rate_v);
        let delta0 = _mm_sub_epi16(diff0, eq0);
        let delta1 = _mm_sub_epi16(diff1, eq1);
        let mask_lt0 = load_u16x8(chunk::<u16, 8>(&MASK_LT[nsym], 0));
        let mask_lt1 = load_u16x8(chunk::<u16, 8>(&MASK_LT[nsym], 8));
        let m0_new = _mm_add_epi16(m0, _mm_and_si128(delta0, mask_lt0));
        let m1_new = _mm_add_epi16(m1, _mm_and_si128(delta1, mask_lt1));
        store_u16x8(chunk_mut::<u16, 8>(cdf, 0), m0_new);
        store_u16x8(chunk_mut::<u16, 8>(cdf, 8), m1_new);
        cdf[nsym] = (count + u32::from(count < 32)) as u16;
    }

    let new_dif = s.dif.wrapping_sub(u64::from(v_val) << 48);
    let new_rng = u_val.wrapping_sub(v_val);
    ctx_norm_fast(
        s.buf, &mut s.pos, &mut s.dif, &mut s.rng, &mut s.cnt, new_dif, new_rng,
    );
    val
}

#[target_feature(enable = "sse2")]
#[inline]
fn hi_tok_sse2(s: &mut MsacContext<'_>, cdf: &mut [u16; 4]) -> usize {
    let mut m0 = load_u16x4(cdf);
    let mut count = u32::from(cdf[3]);
    let min_p = load_u16x4(&[12, 8, 4, 0]);
    let mask_lt = load_u16x4(&[0xFFFF, 0xFFFF, 0xFFFF, 0]);
    let ones = _mm_set1_epi16(-1);
    let zero = _mm_setzero_si128();
    let update = s.allow_update_cdf;
    let mut tok = 0usize;

    let mut round = 0usize;
    while round < 4 {
        let r_vec = _mm_set1_epi16((s.rng & 0xff00) as i16);
        let c_vec = _mm_set1_epi16((s.dif >> 48) as i16);
        let q = _mm_slli_epi16(_mm_srli_epi16(m0, 6), 7);
        let v = _mm_add_epi16(_mm_mulhi_epu16(q, r_vec), min_p);
        let eq = _mm_cmpeq_epi16(_mm_subs_epu16(v, c_vec), zero);
        let mask = _mm_movemask_epi8(eq) as u32;
        let val = ((mask.trailing_zeros() >> 1) as usize).min(3);

        // GPR üzerinden 4×u16 şerit çekimi (store-to-load stall önlemi).
        let v_u64 = _mm_cvtsi128_si64(v) as u64;
        let v_val = ((v_u64 >> (val * 16)) & 0xffff) as u32;
        let u_val = if val == 0 {
            s.rng
        } else {
            ((v_u64 >> ((val - 1) * 16)) & 0xffff) as u32
        };

        if update {
            let rate = 5 + (count >> 4);
            let rate_v = _mm_cvtsi32_si128(rate as i32);
            let target = _mm_avg_epu16(ones, eq);
            let diff = _mm_sra_epi16(_mm_sub_epi16(target, m0), rate_v);
            let delta = _mm_sub_epi16(diff, eq);
            m0 = _mm_add_epi16(m0, _mm_and_si128(delta, mask_lt));
            count += u32::from(count < 32);
        }

        let new_dif = s.dif.wrapping_sub(u64::from(v_val) << 48);
        let new_rng = u_val.wrapping_sub(v_val);
        ctx_norm_fast(
            s.buf, &mut s.pos, &mut s.dif, &mut s.rng, &mut s.cnt, new_dif, new_rng,
        );

        tok = 3 * (round + 1) + val;
        if val < 3 {
            break;
        }
        round += 1;
    }

    if update {
        store_u16x4(cdf, m0);
        cdf[3] = count as u16;
    }
    tok
}

#[target_feature(enable = "sse2")]
#[inline]
fn bool_adapt_sse2(s: &mut MsacContext<'_>, cdf: &mut [u16; 2]) -> u32 {
    let r = s.rng;
    let dif = s.dif;
    let c0 = u32::from(cdf[0]);
    let v = (((r >> 8).wrapping_mul(c0 >> 6)) >> 1) + 4;
    let vw = u64::from(v) << 48;
    let ge = dif >= vw;
    let new_dif = if ge { dif.wrapping_sub(vw) } else { dif };
    let new_rng = if ge { r.wrapping_sub(v) } else { v };
    let bit = u32::from(!ge);

    if s.allow_update_cdf {
        let count = u32::from(cdf[1]);
        let rate = 4 + (count >> 4);
        let c0_i = c0 as i32;
        cdf[0] = if bit != 0 {
            (c0_i + ((32768 - c0_i) >> rate)) as u16
        } else {
            (c0_i - (c0_i >> rate)) as u16
        };
        cdf[1] = (count + u32::from(count < 32)) as u16;
    }

    ctx_norm_fast(
        s.buf, &mut s.pos, &mut s.dif, &mut s.rng, &mut s.cnt, new_dif, new_rng,
    );
    bit
}

#[target_feature(enable = "sse2")]
#[inline]
fn bool_equi_sse2(s: &mut MsacContext<'_>) -> u32 {
    let r = s.rng;
    let dif = s.dif;
    let v = ((r >> 8) << 7) + 4;
    let vw = u64::from(v) << 48;
    let ge = dif >= vw;
    let new_dif = if ge { dif.wrapping_sub(vw) } else { dif };
    let new_rng = if ge { r.wrapping_sub(v) } else { v };
    let bit = u32::from(!ge);

    // `msac.asm:422-427`: `16 - clz(new_rng) = (0xbfff - new_rng) >> 14` (`0 < d <= 2`).
    let d = 0xbfffu32.wrapping_sub(new_rng) >> 14;
    debug_assert_eq!(d, 15 ^ (31 ^ new_rng.leading_zeros()));
    let old_cnt = s.cnt;
    s.dif = new_dif << d;
    s.rng = new_rng << d;
    s.cnt = old_cnt - d as i32;
    if (old_cnt as u32) < d {
        ctx_refill_fast(s.buf, &mut s.pos, &mut s.dif, &mut s.cnt);
    }
    bit
}

#[target_feature(enable = "sse2")]
#[inline]
fn bool_sse2(s: &mut MsacContext<'_>, f: u32) -> u32 {
    let r = s.rng;
    let dif = s.dif;
    let v = (((r >> 8).wrapping_mul(f >> 6)) >> 1) + 4;
    let vw = u64::from(v) << 48;
    let ge = dif >= vw;
    let new_dif = if ge { dif.wrapping_sub(vw) } else { dif };
    let new_rng = if ge { r.wrapping_sub(v) } else { v };
    let bit = u32::from(!ge);

    ctx_norm_fast(
        s.buf, &mut s.pos, &mut s.dif, &mut s.rng, &mut s.cnt, new_dif, new_rng,
    );
    bit
}

/// `dav1d_msac_decode_symbol_adapt4_sse2` (`x86-msac-0007`: `1 <= n_symbols <= 3`, `cdf.len() >= 4`).
#[inline(always)]
pub fn decode_symbol_adapt4_sse2(
    _token: Sse2Token,
    s: &mut MsacContext<'_>,
    cdf: &mut [u16; 4],
    n_symbols: usize,
) -> usize {
    // SAFETY: (1) ISA: `_token` (`Sse2Token`) bu işlemcide SSE2'nin
    // doğrulandığını ve maskeyle kapatılmadığını kanıtlar; `adapt4_sse2`
    // yalnız SSE2 `#[target_feature(enable = "sse2")]` ister.
    // (2) sınır: `cdf` sabit boyutlu `&mut [u16; 4]` dizisidir ve `nsym = n_symbols.min(3) <= 3`
    // ile tüm dizinlemeler `[0..4)` aralığındadır.
    // (3) hizalama/takma ad: yükleme ve saklama `load_u16x4`/`store_u16x4` ile
    // hizasız ve tek `&mut` referans üzerinden yapılır.
    unsafe { adapt4_sse2(s, cdf, n_symbols) }
}

/// `dav1d_msac_decode_symbol_adapt8_sse2` (`x86-msac-0008`: `1 <= n_symbols <= 7`, `cdf.len() >= 8`).
#[inline(always)]
pub fn decode_symbol_adapt8_sse2(
    _token: Sse2Token,
    s: &mut MsacContext<'_>,
    cdf: &mut [u16; 8],
    n_symbols: usize,
) -> usize {
    // SAFETY: (1) ISA: `_token` (`Sse2Token`) bu işlemcide SSE2'nin
    // doğrulandığını kanıtlar; `adapt8_sse2` yalnız SSE2 `#[target_feature]` ister.
    // (2) sınır: `cdf` sabit boyutlu `&mut [u16; 8]` dizisidir ve `nsym = n_symbols.min(7) <= 7`
    // ile tüm dizinlemeler `[0..8)` aralığındadır.
    // (3) hizalama/takma ad: yükleme ve saklama `load_u16x8`/`store_u16x8` ile
    // hizasız ve tek `&mut` referans üzerinden yapılır.
    unsafe { adapt8_sse2(s, cdf, n_symbols) }
}

/// `dav1d_msac_decode_symbol_adapt16_sse2` (`x86-msac-0006`: `1 <= n_symbols <= 15`, `cdf.len() >= 16`).
#[inline(always)]
pub fn decode_symbol_adapt16_sse2(
    _token: Sse2Token,
    s: &mut MsacContext<'_>,
    cdf: &mut [u16; 16],
    n_symbols: usize,
) -> usize {
    // SAFETY: (1) ISA: `_token` (`Sse2Token`) bu işlemcide SSE2'nin
    // doğrulandığını kanıtlar; `adapt16_sse2` yalnız SSE2 `#[target_feature]` ister.
    // (2) sınır: `cdf` sabit boyutlu `&mut [u16; 16]` dizisidir ve `nsym = n_symbols.min(15) <= 15`
    // ile tüm dizinlemeler `[0..16)` aralığındadır.
    // (3) hizalama/takma ad: yükleme ve saklama `load_u16x8`/`store_u16x8` ile
    // hizasız ve tek `&mut` referans üzerinden yapılır.
    unsafe { adapt16_sse2(s, cdf, n_symbols) }
}

/// `dav1d_msac_decode_hi_tok_sse2` (`x86-msac-0004`: `cdf.len() >= 4`).
#[inline(always)]
pub fn decode_hi_tok_sse2(_token: Sse2Token, s: &mut MsacContext<'_>, cdf: &mut [u16; 4]) -> usize {
    // SAFETY: (1) ISA: `_token` (`Sse2Token`) bu işlemcide SSE2'nin
    // doğrulandığını kanıtlar; `hi_tok_sse2` yalnız SSE2 `#[target_feature]` ister.
    // (2) sınır: `cdf` sabit boyutlu `&mut [u16; 4]` dizisidir ve tüm erişimler
    // `[0..4)` aralığındadır.
    // (3) hizalama/takma ad: yükleme ve saklama `load_u16x4`/`store_u16x4` ile
    // hizasız ve tek `&mut` referans üzerinden yapılır.
    unsafe { hi_tok_sse2(s, cdf) }
}

/// `dav1d_msac_decode_bool_adapt_sse2` (`x86-msac-0001`: `cdf.len() >= 2`).
#[inline(always)]
pub fn decode_bool_adapt_sse2(
    _token: Sse2Token,
    s: &mut MsacContext<'_>,
    cdf: &mut [u16; 2],
) -> u32 {
    // SAFETY: (1) ISA: `_token` (`Sse2Token`) bu işlemcide SSE2'nin
    // doğrulandığını kanıtlar; `bool_adapt_sse2` yalnız SSE2 `#[target_feature]` ister.
    // (2) sınır: `cdf` sabit boyutlu `&mut [u16; 2]` dizisidir ve yalnız `cdf[0..2]`
    // hücrelerine erişilir.
    // (3) hizalama/takma ad: doğal `u16` hizalamalı `&mut [u16; 2]` referansı tek
    // erişim yoludur.
    unsafe { bool_adapt_sse2(s, cdf) }
}

/// `dav1d_msac_decode_bool_equi_sse2` (`x86-msac-0002`).
#[inline(always)]
pub fn decode_bool_equi_sse2(_token: Sse2Token, s: &mut MsacContext<'_>) -> u32 {
    // SAFETY: (1) ISA: `_token` (`Sse2Token`) bu işlemcide SSE2'nin
    // doğrulandığını kanıtlar; `bool_equi_sse2` yalnız SSE2 `#[target_feature]` ister.
    // (2) sınır: tampon erişimi `ctx_refill_fast` içinde `pos + 8 <= buf.len()`
    // kontrolüyle sınırlandırılmıştır, tampon dışına okuma yapılmaz.
    // (3) hizalama/takma ad: `&mut MsacContext` tek erişim yoludur.
    unsafe { bool_equi_sse2(s) }
}

/// `dav1d_msac_decode_bool_sse2` (`x86-msac-0003`).
#[inline(always)]
pub fn decode_bool_sse2(_token: Sse2Token, s: &mut MsacContext<'_>, f: u32) -> u32 {
    // SAFETY: (1) ISA: `_token` (`Sse2Token`) bu işlemcide SSE2'nin
    // doğrulandığını kanıtlar; `bool_sse2` yalnız SSE2 `#[target_feature]` ister.
    // (2) sınır: tampon erişimi `ctx_norm_fast` / `ctx_refill_fast` içinde
    // `pos + 8 <= buf.len()` kontrolüyle sınırlandırılmıştır.
    // (3) hizalama/takma ad: `&mut MsacContext` tek erişim yoludur.
    unsafe { bool_sse2(s, f) }
}
