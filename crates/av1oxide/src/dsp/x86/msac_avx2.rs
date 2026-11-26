//! x86_64 AVX2+BMI1+BMI2 MSAC entropi çözücü çekirdekleri (P47 / `F4-MSAC`, 5.D.1).
//!
//! Referans: `dav1d` `src/x86/msac.h` + `src/x86/msac.asm` (`99f3354`).
//! `Avx2Token`, `CPUID 7 EBX` üzerinden `AVX2 + BMI1 + BMI2` (ve `OSXSAVE/XCR0`)
//! yetkisini birlikte kanıtlar (`cpu/x86.rs:LEAF7_EBX_AVX2`). Bu modül
//! `adapt4/8/16`, `hi_tok` ve birleşik 2B AC/dequant çekirdeklerini
//! 3-operand VEX + BMI1 (`tzcnt`) + BMI2 (`shlx`/`shrx`) kod üretimiyle
//! gerçekler (`objdump -d`: bu modülde `0` EVEX komutu).

#![allow(unsafe_code)]

use crate::cpu::Avx2Token;
use crate::dsp::x86::mem::{
    chunk, load_u16x4, load_u16x8, load_u16x16, store_u16x4, store_u16x8, store_u16x16,
};
#[cfg(not(feature = "msac-trace"))]
use crate::dsp::x86::mem::{chunk_mut, store_u8x32};
use crate::dsp::x86::msac::{MASK_LT, MIN_PROB};
#[cfg(not(feature = "msac-trace"))]
use crate::entropy::msac::ctx_refill_fast;
use crate::entropy::msac::{MsacContext, ctx_norm_ctx};
#[cfg(not(feature = "msac-trace"))]
use core::arch::x86_64::_mm256_setzero_si256;
use core::arch::x86_64::{
    _mm_add_epi16, _mm_and_si128, _mm_avg_epu16, _mm_cmpeq_epi16, _mm_cvtsi32_si128,
    _mm_cvtsi128_si64, _mm_max_epu16, _mm_movemask_epi8, _mm_mulhi_epu16, _mm_set1_epi16,
    _mm_slli_epi16, _mm_sra_epi16, _mm_srli_epi16, _mm_sub_epi16, _mm256_add_epi16,
    _mm256_and_si256, _mm256_avg_epu16, _mm256_cmpeq_epi16, _mm256_max_epu16, _mm256_movemask_epi8,
    _mm256_mulhi_epu16, _mm256_set1_epi16, _mm256_slli_epi16, _mm256_sra_epi16, _mm256_srli_epi16,
    _mm256_sub_epi16,
};

#[target_feature(enable = "avx2,bmi1,bmi2")]
#[inline]
fn adapt4_avx2(s: &mut MsacContext<'_>, cdf: &mut [u16; 4], n_symbols: usize) -> usize {
    let nsym = n_symbols.min(3);
    let m0 = load_u16x4(cdf);
    let r_vec = _mm_set1_epi16((s.rng & 0xff00) as i16);
    let c_vec = _mm_set1_epi16((s.dif >> 48) as i16);
    let min_p = load_u16x4(chunk::<u16, 4>(&MIN_PROB[nsym], 0));
    let q = _mm_slli_epi16(_mm_srli_epi16(m0, 6), 7);
    let v = _mm_add_epi16(_mm_mulhi_epu16(q, r_vec), min_p);
    let eq = _mm_cmpeq_epi16(_mm_max_epu16(v, c_vec), c_vec);
    let mask = _mm_movemask_epi8(eq) as u32;
    let val = ((mask.trailing_zeros() >> 1) as usize).min(nsym);

    let v64 = _mm_cvtsi128_si64(v) as u64;
    let (u_val, v_val) = if val == 0 {
        (s.rng, (v64 & 0xffff) as u32)
    } else {
        let pair = (v64 >> ((val - 1) * 16)) as u32;
        (pair & 0xffff, pair >> 16)
    };

    let new_dif = s.dif.wrapping_sub(u64::from(v_val) << 48);
    let new_rng = u_val.wrapping_sub(v_val);
    let update = s.allow_update_cdf;
    ctx_norm_ctx(s, new_dif, new_rng);

    if update {
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

    val
}

#[target_feature(enable = "avx2,bmi1,bmi2")]
#[inline]
fn adapt8_avx2(s: &mut MsacContext<'_>, cdf: &mut [u16; 8], n_symbols: usize) -> usize {
    let nsym = n_symbols.min(7);
    let m0 = load_u16x8(cdf);
    let r_vec = _mm_set1_epi16((s.rng & 0xff00) as i16);
    let c_vec = _mm_set1_epi16((s.dif >> 48) as i16);
    let min_p = load_u16x8(chunk::<u16, 8>(&MIN_PROB[nsym], 0));
    let q = _mm_slli_epi16(_mm_srli_epi16(m0, 6), 7);
    let v = _mm_add_epi16(_mm_mulhi_epu16(q, r_vec), min_p);
    let eq = _mm_cmpeq_epi16(_mm_max_epu16(v, c_vec), c_vec);
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

    let new_dif = s.dif.wrapping_sub(u64::from(v_val) << 48);
    let new_rng = u_val.wrapping_sub(v_val);
    let update = s.allow_update_cdf;
    ctx_norm_ctx(s, new_dif, new_rng);

    if update {
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

    val
}

#[target_feature(enable = "avx2,bmi1,bmi2")]
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

    let new_dif = s.dif.wrapping_sub(u64::from(v_val) << 48);
    let new_rng = u_val.wrapping_sub(v_val);
    let update = s.allow_update_cdf;
    ctx_norm_ctx(s, new_dif, new_rng);

    if update {
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

    val
}

#[target_feature(enable = "avx2,bmi1,bmi2")]
#[inline]
fn hi_tok_avx2(s: &mut MsacContext<'_>, cdf: &mut [u16; 4]) -> usize {
    let mut m0 = load_u16x4(cdf);
    let mut count = u32::from(cdf[3]);
    let min_p = load_u16x4(&[12, 8, 4, 0]);
    let mask_lt = load_u16x4(&[0xFFFF, 0xFFFF, 0xFFFF, 0]);
    let ones = _mm_set1_epi16(-1);
    let update = s.allow_update_cdf;
    let mut tok = 0usize;

    let mut round = 0usize;
    while round < 4 {
        let r_vec = _mm_set1_epi16((s.rng & 0xff00) as i16);
        let c_vec = _mm_set1_epi16((s.dif >> 48) as i16);
        let q = _mm_slli_epi16(_mm_srli_epi16(m0, 6), 7);
        let v = _mm_add_epi16(_mm_mulhi_epu16(q, r_vec), min_p);
        let eq = _mm_cmpeq_epi16(_mm_max_epu16(v, c_vec), c_vec);
        let mask = _mm_movemask_epi8(eq) as u32;
        let val = ((mask.trailing_zeros() >> 1) as usize).min(3);

        let v64 = _mm_cvtsi128_si64(v) as u64;
        let (u_val, v_val) = if val == 0 {
            (s.rng, (v64 & 0xffff) as u32)
        } else {
            let pair = (v64 >> ((val - 1) * 16)) as u32;
            (pair & 0xffff, pair >> 16)
        };

        let new_dif = s.dif.wrapping_sub(u64::from(v_val) << 48);
        let new_rng = u_val.wrapping_sub(v_val);
        ctx_norm_ctx(s, new_dif, new_rng);

        if update {
            let rate = 5 + (count >> 4);
            let rate_v = _mm_cvtsi32_si128(rate as i32);
            let target = _mm_avg_epu16(ones, eq);
            let diff = _mm_sra_epi16(_mm_sub_epi16(target, m0), rate_v);
            let delta = _mm_sub_epi16(diff, eq);
            m0 = _mm_add_epi16(m0, _mm_and_si128(delta, mask_lt));
            count += u32::from(count < 32);
        }

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

/// `dav1d_msac_decode_symbol_adapt4` (AVX2+BMI1+BMI2, `1 <= n_symbols <= 3`, `cdf.len() >= 4`).
#[inline(always)]
pub fn decode_symbol_adapt4_avx2(
    _token: Avx2Token,
    s: &mut MsacContext<'_>,
    cdf: &mut [u16; 4],
    n_symbols: usize,
) -> usize {
    // SAFETY: (1) ISA: `_token` (`Avx2Token`) `AVX2 + BMI1 + BMI2` yetkisinin
    // CPUID ve XCR0 üzerinden doğrulandığını kanıtlar; `adapt4_avx2` yalnız
    // `avx2,bmi1,bmi2` ister.
    // (2) sınır: `cdf` sabit boyutlu `&mut [u16; 4]` dizisidir ve `nsym = n_symbols.min(3) <= 3`
    // ile tüm dizinlemeler `[0..4)` aralığındadır.
    // (3) hizalama/takma ad: yükleme/saklama `load_u16x4`/`store_u16x4` ile
    // hizasız ve tek `&mut` referans üzerinden yapılır.
    unsafe { adapt4_avx2(s, cdf, n_symbols) }
}

/// `dav1d_msac_decode_symbol_adapt8` (AVX2+BMI1+BMI2, `1 <= n_symbols <= 7`, `cdf.len() >= 8`).
#[inline(always)]
pub fn decode_symbol_adapt8_avx2(
    _token: Avx2Token,
    s: &mut MsacContext<'_>,
    cdf: &mut [u16; 8],
    n_symbols: usize,
) -> usize {
    // SAFETY: (1) ISA: `_token` (`Avx2Token`) `AVX2 + BMI1 + BMI2` yetkisini
    // kanıtlar; `adapt8_avx2` yalnız `avx2,bmi1,bmi2` ister.
    // (2) sınır: `cdf` sabit boyutlu `&mut [u16; 8]` dizisidir ve `nsym = n_symbols.min(7) <= 7`
    // ile tüm dizinlemeler `[0..8)` aralığındadır.
    // (3) hizalama/takma ad: yükleme/saklama `load_u16x8`/`store_u16x8` ile
    // hizasız ve tek `&mut` referans üzerinden yapılır.
    unsafe { adapt8_avx2(s, cdf, n_symbols) }
}

/// `dav1d_msac_decode_symbol_adapt16_avx2` (`x86-msac-0005`: `1 <= n_symbols <= 15`, `cdf.len() >= 16`).
#[inline(always)]
pub fn decode_symbol_adapt16_avx2(
    _token: Avx2Token,
    s: &mut MsacContext<'_>,
    cdf: &mut [u16; 16],
    n_symbols: usize,
) -> usize {
    // SAFETY: (1) ISA: `_token` (`Avx2Token`) `AVX2 + BMI1 + BMI2` yetkisini
    // kanıtlar; `adapt16_avx2` yalnız `avx2,bmi1,bmi2` ister.
    // (2) sınır: `cdf` sabit boyutlu `&mut [u16; 16]` dizisidir ve `nsym = n_symbols.min(15) <= 15`
    // ile tüm dizinlemeler `[0..16)` aralığındadır.
    // (3) hizalama/takma ad: yükleme ve saklama `load_u16x16`/`store_u16x16` ile
    // hizasız ve tek `&mut` referans üzerinden yapılır.
    unsafe { adapt16_avx2(s, cdf, n_symbols) }
}

/// `dav1d_msac_decode_hi_tok` (AVX2+BMI1+BMI2, `cdf.len() >= 4`).
#[inline(always)]
pub fn decode_hi_tok_avx2(_token: Avx2Token, s: &mut MsacContext<'_>, cdf: &mut [u16; 4]) -> usize {
    // SAFETY: (1) ISA: `_token` (`Avx2Token`) `AVX2 + BMI1 + BMI2` yetkisini
    // kanıtlar; `hi_tok_avx2` yalnız `avx2,bmi1,bmi2` ister.
    // (2) sınır: `cdf` sabit boyutlu `&mut [u16; 4]` dizisidir ve tüm erişimler
    // `[0..4)` aralığındadır.
    // (3) hizalama/takma ad: yükleme/saklama `load_u16x4`/`store_u16x4` ile
    // hizasız ve tek `&mut` referans üzerinden yapılır.
    unsafe { hi_tok_avx2(s, cdf) }
}

/// 2B AC katsayı döngüsü + `eob` + `dc_tok` çözümü (AVX2+BMI1+BMI2 birleşik çekirdek).
///
/// `MsacContext` alanları (`dif`, `rng`, `cnt`, `pos`) döngü boyunca yerel
/// yazmaçlarda tutulur; `n_symbols == 3` ve `UPDATE` derleme zamanında sabittir.
#[cfg(not(feature = "msac-trace"))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,bmi1,bmi2")]
fn decode_ac_2d_avx2_core<const SLH: usize, const UPDATE: bool>(
    s: &mut MsacContext<'_>,
    lo_cdf: &mut [[u16; 4]; 41],
    hi_cdf: &mut [[u16; 4]; 21],
    levels: &mut [u8; 32 * 34 + 64],
    cf: &mut [i32; 1024],
    scan: &[u16],
    lo_ctx_offsets: &[[u8; 5]; 5],
    slw: usize,
    eob_idx: usize,
    eob_tok: usize,
) -> (usize, usize) {
    let shift = SLH + 2;
    let stride = 4usize << SLH;
    let mask = stride - 1;
    let scan = &scan[..=eob_idx];

    // `levels` temizleme: `memset` çağrısı yerine 256-bit AVX2 sıfırlama.
    // `clear_bytes <= 32 * 34 = 1088 <= 1152 = 36 * 32`.
    let clear_bytes = stride * ((4usize << slw) + 2);
    let n32 = ((clear_bytes + 31) >> 5).min(36);
    let zero256 = _mm256_setzero_si256();
    let mut off = 0usize;
    let end = n32 * 32;
    while off < end {
        store_u8x32(chunk_mut::<u8, 32>(levels, off), zero256);
        off += 32;
    }

    let buf = s.buf;
    let mut pos = s.pos;
    let mut dif = s.dif;
    let mut rng = s.rng;
    let mut cnt = s.cnt;

    let min_p = load_u16x4(&[12, 8, 4, 0]);
    let mask_lt = load_u16x4(&[0xFFFF, 0xFFFF, 0xFFFF, 0]);
    let ones = _mm_set1_epi16(-1);

    let mut rc = usize::from(scan[eob_idx]) & 0x3ff;
    let x0 = rc >> shift;
    let y0 = rc & mask;
    let mut tok = eob_tok + 1;
    let mut level_tok = tok * 0x41;
    if eob_tok == 2 {
        let hi_ctx = if x0 | y0 > 1 { 14 } else { 7 };
        tok = hi_tok_regs::<UPDATE>(
            buf,
            &mut pos,
            &mut dif,
            &mut rng,
            &mut cnt,
            &mut hi_cdf[hi_ctx],
            min_p,
            mask_lt,
            ones,
        );
        level_tok = tok + (3 << 6);
    }
    cf[rc] = (tok << 11) as i32;
    levels[rc] = level_tok as u8;

    for &scan_rc in scan[1..eob_idx].iter().rev() {
        let rc_i = usize::from(scan_rc) & 0x3ff;
        let x = rc_i >> shift;
        let y = rc_i & mask;

        let mut mag = u32::from(levels[rc_i + 1])
            + u32::from(levels[rc_i + stride])
            + u32::from(levels[rc_i + stride + 1]);
        let hi_mag = mag;
        mag += u32::from(levels[rc_i + 2]) + u32::from(levels[rc_i + 2 * stride]);
        let offset = usize::from(lo_ctx_offsets[y.min(4)][x.min(4)]);
        let add = (((mag + 64) >> 7) as usize).min(4);
        let lo_ctx = offset + add;

        let cdf = &mut lo_cdf[lo_ctx];
        let m0 = load_u16x4(cdf);
        let r_vec = _mm_set1_epi16((rng & 0xff00) as i16);
        let c_vec = _mm_set1_epi16((dif >> 48) as i16);
        let q = _mm_slli_epi16(_mm_srli_epi16(m0, 6), 7);
        let v = _mm_add_epi16(_mm_mulhi_epu16(q, r_vec), min_p);
        let eq = _mm_cmpeq_epi16(_mm_max_epu16(v, c_vec), c_vec);
        let mask_b = _mm_movemask_epi8(eq) as u32;
        let val = ((mask_b.trailing_zeros() >> 1) as usize).min(3);

        let v64 = _mm_cvtsi128_si64(v) as u64;
        let (u_val, v_val) = if val == 0 {
            (rng, (v64 & 0xffff) as u32)
        } else {
            let pair = (v64 >> ((val - 1) * 16)) as u32;
            (pair & 0xffff, pair >> 16)
        };

        let new_dif = dif.wrapping_sub(u64::from(v_val) << 48);
        let new_rng = u_val.wrapping_sub(v_val);
        let d = 15 ^ (31 ^ new_rng.leading_zeros());
        let old_cnt = cnt;
        dif = new_dif << d;
        rng = new_rng << d;
        cnt = old_cnt - d as i32;
        if (old_cnt as u32) < d {
            ctx_refill_fast(buf, &mut pos, &mut dif, &mut cnt);
        }

        if UPDATE {
            let count = u32::from(cdf[3]);
            let rate = 5 + (count >> 4);
            let rate_v = _mm_cvtsi32_si128(rate as i32);
            let target = _mm_avg_epu16(ones, eq);
            let diff = _mm_sra_epi16(_mm_sub_epi16(target, m0), rate_v);
            let delta = _mm_sub_epi16(diff, eq);
            let m0_new = _mm_add_epi16(m0, _mm_and_si128(delta, mask_lt));
            store_u16x4(cdf, m0_new);
            cdf[3] = (count + u32::from(count < 32)) as u16;
        }

        if val == 3 {
            let mag = hi_mag & 63;
            let base_ctx = if (x | y) > 1 { 14 } else { 7 };
            let hi_add = (if mag > 12 { 6 } else { (mag + 1) >> 1 }) as usize;
            let hi_ctx = base_ctx + hi_add;
            let htok = hi_tok_regs::<UPDATE>(
                buf,
                &mut pos,
                &mut dif,
                &mut rng,
                &mut cnt,
                &mut hi_cdf[hi_ctx],
                min_p,
                mask_lt,
                ones,
            );
            levels[rc_i] = (htok + (3 << 6)) as u8;
            cf[rc_i] = ((htok << 11) | rc) as i32;
            rc = rc_i;
        } else {
            let t = (val as u32) * 0x17ff41;
            levels[rc_i] = t as u8;
            let v_cf = (t >> 9) & (rc as u32).wrapping_add(!0x7ffu32);
            if val != 0 {
                rc = rc_i;
            }
            cf[rc_i] = v_cf as i32;
        }
    }

    // `dc_tok` (`dc_ctx == 0` 2B sınıfında) + `dc_hi_tok` sıcak yazmaçlarla.
    let dc_cdf = &mut lo_cdf[0];
    let m0 = load_u16x4(dc_cdf);
    let r_vec = _mm_set1_epi16((rng & 0xff00) as i16);
    let c_vec = _mm_set1_epi16((dif >> 48) as i16);
    let q = _mm_slli_epi16(_mm_srli_epi16(m0, 6), 7);
    let v = _mm_add_epi16(_mm_mulhi_epu16(q, r_vec), min_p);
    let eq = _mm_cmpeq_epi16(_mm_max_epu16(v, c_vec), c_vec);
    let mask_b = _mm_movemask_epi8(eq) as u32;
    let mut dc_tok = ((mask_b.trailing_zeros() >> 1) as usize).min(3);

    let v64 = _mm_cvtsi128_si64(v) as u64;
    let (u_val, v_val) = if dc_tok == 0 {
        (rng, (v64 & 0xffff) as u32)
    } else {
        let pair = (v64 >> ((dc_tok - 1) * 16)) as u32;
        (pair & 0xffff, pair >> 16)
    };
    let new_dif = dif.wrapping_sub(u64::from(v_val) << 48);
    let new_rng = u_val.wrapping_sub(v_val);
    let d = 15 ^ (31 ^ new_rng.leading_zeros());
    let old_cnt = cnt;
    dif = new_dif << d;
    rng = new_rng << d;
    cnt = old_cnt - d as i32;
    if (old_cnt as u32) < d {
        ctx_refill_fast(buf, &mut pos, &mut dif, &mut cnt);
    }
    if UPDATE {
        let count = u32::from(dc_cdf[3]);
        let rate = 5 + (count >> 4);
        let rate_v = _mm_cvtsi32_si128(rate as i32);
        let target = _mm_avg_epu16(ones, eq);
        let diff = _mm_sra_epi16(_mm_sub_epi16(target, m0), rate_v);
        let delta = _mm_sub_epi16(diff, eq);
        let m0_new = _mm_add_epi16(m0, _mm_and_si128(delta, mask_lt));
        store_u16x4(dc_cdf, m0_new);
        dc_cdf[3] = (count + u32::from(count < 32)) as u16;
    }
    if dc_tok == 3 {
        let mag =
            (u32::from(levels[1]) + u32::from(levels[stride]) + u32::from(levels[stride + 1])) & 63;
        let hi_ctx = if mag > 12 { 6 } else { (mag as usize + 1) >> 1 };
        dc_tok = hi_tok_regs::<UPDATE>(
            buf,
            &mut pos,
            &mut dif,
            &mut rng,
            &mut cnt,
            &mut hi_cdf[hi_ctx],
            min_p,
            mask_lt,
            ones,
        );
    }

    s.pos = pos;
    s.dif = dif;
    s.rng = rng;
    s.cnt = cnt;
    (rc, dc_tok)
}

#[cfg(not(feature = "msac-trace"))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,bmi1,bmi2")]
#[inline]
fn hi_tok_regs<const UPDATE: bool>(
    buf: &[u8],
    pos: &mut usize,
    dif: &mut u64,
    rng: &mut u32,
    cnt: &mut i32,
    cdf: &mut [u16; 4],
    min_p: core::arch::x86_64::__m128i,
    mask_lt: core::arch::x86_64::__m128i,
    ones: core::arch::x86_64::__m128i,
) -> usize {
    let mut m0 = load_u16x4(cdf);
    let mut count = u32::from(cdf[3]);
    let mut tok = 0usize;
    let mut round = 0usize;
    while round < 4 {
        let r_vec = _mm_set1_epi16((*rng & 0xff00) as i16);
        let c_vec = _mm_set1_epi16((*dif >> 48) as i16);
        let q = _mm_slli_epi16(_mm_srli_epi16(m0, 6), 7);
        let v = _mm_add_epi16(_mm_mulhi_epu16(q, r_vec), min_p);
        let eq = _mm_cmpeq_epi16(_mm_max_epu16(v, c_vec), c_vec);
        let mask_b = _mm_movemask_epi8(eq) as u32;
        let val = ((mask_b.trailing_zeros() >> 1) as usize).min(3);

        let v64 = _mm_cvtsi128_si64(v) as u64;
        let (u_val, v_val) = if val == 0 {
            (*rng, (v64 & 0xffff) as u32)
        } else {
            let pair = (v64 >> ((val - 1) * 16)) as u32;
            (pair & 0xffff, pair >> 16)
        };

        let new_dif = (*dif).wrapping_sub(u64::from(v_val) << 48);
        let new_rng = u_val.wrapping_sub(v_val);
        let d = 15 ^ (31 ^ new_rng.leading_zeros());
        let old_cnt = *cnt;
        *dif = new_dif << d;
        *rng = new_rng << d;
        *cnt = old_cnt - d as i32;
        if (old_cnt as u32) < d {
            ctx_refill_fast(buf, pos, dif, cnt);
        }

        if UPDATE {
            let rate = 5 + (count >> 4);
            let rate_v = _mm_cvtsi32_si128(rate as i32);
            let target = _mm_avg_epu16(ones, eq);
            let diff = _mm_sra_epi16(_mm_sub_epi16(target, m0), rate_v);
            let delta = _mm_sub_epi16(diff, eq);
            m0 = _mm_add_epi16(m0, _mm_and_si128(delta, mask_lt));
            count += u32::from(count < 32);
        }

        tok = 3 * (round + 1) + val;
        if val < 3 {
            break;
        }
        round += 1;
    }

    if UPDATE {
        store_u16x4(cdf, m0);
        cdf[3] = count as u16;
    }
    tok
}

#[cfg(not(feature = "msac-trace"))]
#[cold]
#[inline(never)]
#[target_feature(enable = "avx2,bmi1,bmi2")]
fn read_golomb_regs(
    buf: &[u8],
    pos: &mut usize,
    dif: &mut u64,
    rng: &mut u32,
    cnt: &mut i32,
) -> u32 {
    let mut len = 0u32;
    let mut val = 1u32;
    loop {
        let bit = bool_equi_regs(buf, pos, dif, rng, cnt);
        if bit != 0 || len >= 32 {
            break;
        }
        len += 1;
    }
    while len > 0 {
        len -= 1;
        val = val
            .wrapping_shl(1)
            .wrapping_add(bool_equi_regs(buf, pos, dif, rng, cnt));
    }
    val.wrapping_sub(1)
}

#[cfg(not(feature = "msac-trace"))]
#[target_feature(enable = "avx2,bmi1,bmi2")]
#[inline]
fn bool_equi_regs(buf: &[u8], pos: &mut usize, dif: &mut u64, rng: &mut u32, cnt: &mut i32) -> u32 {
    let r = *rng;
    let df = *dif;
    let v = ((r >> 8) << 7) + 4;
    let vw = u64::from(v) << 48;
    let ge = df >= vw;
    let new_dif = if ge { df.wrapping_sub(vw) } else { df };
    let new_rng = if ge { r.wrapping_sub(v) } else { v };
    let bit = u32::from(!ge);
    let d = 0xbfffu32.wrapping_sub(new_rng) >> 14;
    let old_cnt = *cnt;
    *dif = new_dif << d;
    *rng = new_rng << d;
    *cnt = old_cnt - d as i32;
    if (old_cnt as u32) < d {
        ctx_refill_fast(buf, pos, dif, cnt);
    }
    bit
}

#[cfg(not(feature = "msac-trace"))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,bmi1,bmi2")]
fn dequant_no_qm_avx2_core(
    s: &mut MsacContext<'_>,
    cf: &mut [i32; 1024],
    mut rc: usize,
    ac_dq: u32,
    dq_shift: u32,
    cf_max: u32,
    mut cul_level: u32,
) -> u32 {
    let buf = s.buf;
    let mut pos = s.pos;
    let mut dif = s.dif;
    let mut rng = s.rng;
    let mut cnt = s.cnt;

    while rc != 0 {
        let sign = bool_equi_regs(buf, &mut pos, &mut dif, &mut rng, &mut cnt);
        let rc_tok = cf[rc] as u32;
        let dq = if rc_tok >= (15 << 11) {
            let tok = read_golomb_regs(buf, &mut pos, &mut dif, &mut rng, &mut cnt)
                .wrapping_add(15)
                & 0xfffff;
            cul_level += tok;
            ((ac_dq.wrapping_mul(tok) & 0xffffff) >> dq_shift).min(cf_max.wrapping_add(sign))
        } else {
            let tok = rc_tok >> 11;
            cul_level += tok;
            (ac_dq * tok) >> dq_shift
        };
        cf[rc] = if sign != 0 { -(dq as i32) } else { dq as i32 };
        rc = (rc_tok & 0x3ff) as usize;
    }

    s.pos = pos;
    s.dif = dif;
    s.rng = rng;
    s.cnt = cnt;
    cul_level
}

#[cfg(not(feature = "msac-trace"))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,bmi1,bmi2")]
fn decode_ac_2d_avx2_entry<const SLH: usize>(
    s: &mut MsacContext<'_>,
    lo_cdf: &mut [[u16; 4]; 41],
    hi_cdf: &mut [[u16; 4]; 21],
    levels: &mut [u8; 32 * 34 + 64],
    cf: &mut [i32; 1024],
    scan: &[u16],
    lo_ctx_offsets: &[[u8; 5]; 5],
    slw: usize,
    eob_idx: usize,
    eob_tok: usize,
) -> (usize, usize) {
    if s.allow_update_cdf {
        decode_ac_2d_avx2_core::<SLH, true>(
            s,
            lo_cdf,
            hi_cdf,
            levels,
            cf,
            scan,
            lo_ctx_offsets,
            slw,
            eob_idx,
            eob_tok,
        )
    } else {
        decode_ac_2d_avx2_core::<SLH, false>(
            s,
            lo_cdf,
            hi_cdf,
            levels,
            cf,
            scan,
            lo_ctx_offsets,
            slw,
            eob_idx,
            eob_tok,
        )
    }
}

/// 2B AC katsayı döngüsü + `dc_tok` güvenli AVX2+BMI1+BMI2 girişi (`decode/coef.rs`).
#[cfg(not(feature = "msac-trace"))]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn decode_ac_2d_avx2<const SLH: usize>(
    _token: Avx2Token,
    s: &mut MsacContext<'_>,
    lo_cdf: &mut [[u16; 4]; 41],
    hi_cdf: &mut [[u16; 4]; 21],
    levels: &mut [u8; 32 * 34 + 64],
    cf: &mut [i32; 1024],
    scan: &[u16],
    lo_ctx_offsets: &[[u8; 5]; 5],
    slw: usize,
    eob_idx: usize,
    eob_tok: usize,
) -> (usize, usize) {
    // SAFETY: (1) ISA: `_token` (`Avx2Token`) `AVX2 + BMI1 + BMI2` yetkisini
    // kanıtlar. (2) sınır: `lo_cdf`, `hi_cdf`, `levels`, `cf` sabit boyutlu
    // referanslardır; tüm dizinlemeler güvenli Rust dilim denetimi altındadır.
    // (3) hizalama/takma ad: tüm SIMD erişimleri `load_u16x4`/`store_u16x4`/
    // `store_u8x32` ile hizasız ve ayrık `&mut` referanslar üzerinden yapılır.
    unsafe {
        decode_ac_2d_avx2_entry::<SLH>(
            s,
            lo_cdf,
            hi_cdf,
            levels,
            cf,
            scan,
            lo_ctx_offsets,
            slw,
            eob_idx,
            eob_tok,
        )
    }
}

/// QM'siz AC işaret + dequant döngüsü güvenli AVX2+BMI1+BMI2 girişi (`decode/coef.rs`).
#[cfg(not(feature = "msac-trace"))]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn dequant_no_qm_avx2(
    _token: Avx2Token,
    s: &mut MsacContext<'_>,
    cf: &mut [i32; 1024],
    rc: usize,
    ac_dq: u32,
    dq_shift: u32,
    cf_max: u32,
    cul_level: u32,
) -> u32 {
    // SAFETY: (1) ISA: `_token` (`Avx2Token`) `AVX2 + BMI1 + BMI2` yetkisini
    // kanıtlar. (2) sınır: `cf` sabit 1024 elemanlı dizidir ve `rc = rc_tok & 0x3ff`.
    // (3) hizalama/takma ad: `s` ve `cf` ayrık `&mut` referanslardır.
    unsafe { dequant_no_qm_avx2_core(s, cf, rc, ac_dq, dq_shift, cf_max, cul_level) }
}
