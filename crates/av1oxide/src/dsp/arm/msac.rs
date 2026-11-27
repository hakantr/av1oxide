//! `aarch64` MSAC entropi çözücü hızlı çekirdekleri (`P54` / `F5d` — `F5-MSAC`, `P55` / `F5e`, `Faz P4`).
//!
//! Referans: `dav1d` `src/arm/msac.h` + `src/arm/64/msac.S` (`99f3354`).
//! Bu modül 7 ARM64 MSAC işlevini (`decode_symbol_adapt4_neon`,
//! `decode_symbol_adapt8_neon`, `decode_symbol_adapt16_neon`,
//! `decode_hi_tok_neon`, `decode_bool_adapt_neon`, `decode_bool_equi_neon`,
//! `decode_bool_neon`) `%100 #![forbid(unsafe_code)]` ve `#[inline(always)]`
//! olarak gerçekler:
//! - Sembol arama kritik yolu (`s.rng` / `s.dif` seri bağımlılık zinciri) AArch64
//!   GPR yazmaçlarında kalır (`GPR ↔ NEON` `umov` gecikmesi = `0` çevrim) ve
//!   çağıran `decode_coefs` / `decode_b` döngülerine doğrudan satır içi açılır.
//! - CDF olasılık adaptasyonu sabit boyutlu `&mut [u16; N]` dilimi üzerinde
//!   16-bit yükleme/saklama genişliğini korur (Apple M4 LSU'da 128b vektör saklama
//!   → 16b skaler yükleme genişlik uyumsuzluğu stall'unu önler) ve yeniden
//!   ölçeklemede 64-bit tek yüklemeli [`ctx_norm_fast`] / [`ctx_refill_fast`] kullanır.

#![forbid(unsafe_code)]

use crate::entropy::msac::{MsacContext, ctx_norm_fast, ctx_refill_fast};

#[inline(always)]
fn adapt_symbol_fixed<const N: usize, const MAX_SYM: usize>(
    s: &mut MsacContext<'_>,
    cdf: &mut [u16; N],
    n_symbols: usize,
) -> usize {
    let nsym = n_symbols.min(MAX_SYM);
    let c = (s.dif >> 48) as u32;
    let r = s.rng >> 8;
    let mut v = s.rng;
    let mut u = v;
    let mut val = 0usize;
    let mut min_p = (nsym as u32) << 2;

    while val < nsym {
        u = v;
        v = ((r.wrapping_mul(u32::from(cdf[val] >> 6))) >> 1).wrapping_add(min_p);
        if c >= v {
            break;
        }
        min_p -= 4;
        val += 1;
    }
    if val == nsym {
        u = v;
        v = 0;
    }

    let new_dif = s.dif.wrapping_sub(u64::from(v) << 48);
    let new_rng = u.wrapping_sub(v);
    ctx_norm_fast(
        s.buf, &mut s.pos, &mut s.dif, &mut s.rng, &mut s.cnt, new_dif, new_rng,
    );

    if s.allow_update_cdf {
        let count = u32::from(cdf[nsym]);
        let rate = 4 + (count >> 4) + u32::from(nsym > 2);
        let mut i = 0usize;
        while i < val {
            let m = i32::from(cdf[i]);
            cdf[i] = (m + ((32768 - m) >> rate)) as u16;
            i += 1;
        }
        while i < nsym {
            let m = i32::from(cdf[i]);
            cdf[i] = (m - (m >> rate)) as u16;
            i += 1;
        }
        cdf[nsym] = (count + u32::from(count < 32)) as u16;
    }
    val
}

#[inline(always)]
pub(super) fn adapt4_neon(s: &mut MsacContext<'_>, cdf: &mut [u16; 4], n_symbols: usize) -> usize {
    adapt_symbol_fixed::<4, 3>(s, cdf, n_symbols)
}

#[inline(always)]
pub(super) fn adapt8_neon(s: &mut MsacContext<'_>, cdf: &mut [u16; 8], n_symbols: usize) -> usize {
    adapt_symbol_fixed::<8, 7>(s, cdf, n_symbols)
}

#[inline(always)]
pub(super) fn adapt16_neon(
    s: &mut MsacContext<'_>,
    cdf: &mut [u16; 16],
    n_symbols: usize,
) -> usize {
    adapt_symbol_fixed::<16, 15>(s, cdf, n_symbols)
}

#[inline(always)]
pub(super) fn hi_tok_neon(s: &mut MsacContext<'_>, cdf: &mut [u16; 4]) -> usize {
    let mut tok_br = adapt_symbol_fixed::<4, 3>(s, cdf, 3);
    let mut tok = 3 + tok_br;
    if tok_br == 3 {
        tok_br = adapt_symbol_fixed::<4, 3>(s, cdf, 3);
        tok = 6 + tok_br;
        if tok_br == 3 {
            tok_br = adapt_symbol_fixed::<4, 3>(s, cdf, 3);
            tok = 9 + tok_br;
            if tok_br == 3 {
                tok = 12 + adapt_symbol_fixed::<4, 3>(s, cdf, 3);
            }
        }
    }
    tok
}

#[inline(always)]
pub(super) fn bool_adapt_neon(s: &mut MsacContext<'_>, cdf: &mut [u16; 2]) -> u32 {
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

#[inline(always)]
pub(super) fn bool_equi_neon(s: &mut MsacContext<'_>) -> u32 {
    let r = s.rng;
    let dif = s.dif;
    let v = ((r >> 8) << 7) + 4;
    let vw = u64::from(v) << 48;
    let ge = dif >= vw;
    let new_dif = if ge { dif.wrapping_sub(vw) } else { dif };
    let new_rng = if ge { r.wrapping_sub(v) } else { v };
    let bit = u32::from(!ge);

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

#[inline(always)]
pub(super) fn bool_neon(s: &mut MsacContext<'_>, f: u32) -> u32 {
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
