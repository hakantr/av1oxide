//! SSE4.1 `load_tmvs` çekirdeği (`D03042` + `D03043`; P46 Adım 1).
//!
//! ## Kaynak eşlemesi (`dav1d` @ `99f3354`, salt-okunur)
//!
//! | Sembol | Kaynak |
//! |---|---|
//! | `dav1d_load_tmvs_sse4` | `src/x86/refmvs.asm:362-596`, `src/x86/refmvs.h:31,54` |
//!
//! ## Yalıtım ve güvenlik (P34 / GUV-02 / GUV-04)
//!
//! - Bu modülde bütün işlevler yalnız `#[target_feature(enable = "sse4.1")]`
//!   taşır; VEX ve EVEX komutu üretilmez (`0` VEX/EVEX).
//! - Kopyasız O(1) sözleşme: girişte [`load_tmvs_sozlesmesi`] denetlenir;
//!   sözleşme dışı girdide [`load_tmvs_skaler_devret`] ile skaler gövdeye
//!   devredilir.
//! - Tek `unsafe` blok [`load_tmvs_sse41`] içindeki `#[target_feature]` giriş
//!   çağrısıdır; dosya `audit::UNSAFE_EXEMPTIONS` defterinde kayıtlıdır.

#![allow(unsafe_code)]

use core::arch::x86_64::{
    __m128i, _mm_abs_epi32, _mm_add_epi32, _mm_cvtsi128_si64, _mm_mul_epi32, _mm_mullo_epi32,
    _mm_packs_epi32, _mm_packus_epi16, _mm_set_epi64x, _mm_set1_epi32, _mm_setr_epi32,
    _mm_shuffle_epi32, _mm_sign_epi32, _mm_srai_epi32, _mm_srli_epi32, _mm_xor_si128,
};

use crate::cpu::Sse41Token;
use crate::decode::refmvs::Mv;
use crate::decode::refmvs_tmvs::{INVALID_REF2CUR, MfmvPlan, TemporalBlock};
use crate::dsp::refmvs::{LoadTmvsFn, load_tmvs_skaler_devret, load_tmvs_sozlesmesi};

/// `mv_proj` tablosu (`src/x86/refmvs.asm:50-53`, `src/refmvs.c:177-181`).
const MV_PROJ: [i32; 32] = [
    0, 16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638, 1489, 1365, 1260, 1170, 1092,
    1024, 963, 910, 862, 819, 780, 744, 712, 682, 655, 630, 606, 585, 564, 546, 528,
];

/// `load_tmvs` SSE4.1 yuva işaretçisi (`D03042` / `D03043`).
#[must_use]
pub const fn load_tmvs_sse41_fn(_token: Sse41Token) -> LoadTmvsFn {
    load_tmvs_sse41
}

#[allow(clippy::too_many_arguments)]
fn load_tmvs_sse41(
    rp_proj: &mut [TemporalBlock],
    stride: usize,
    iw8: usize,
    ih8: usize,
    plan: &MfmvPlan,
    refs: &[Option<&[TemporalBlock]>; 7],
    tile_row_idx: usize,
    col_start8: usize,
    col_end8: usize,
    row_start8: usize,
    row_end8: usize,
) {
    if !load_tmvs_sozlesmesi(
        rp_proj.len(),
        stride,
        iw8,
        ih8,
        plan,
        refs,
        tile_row_idx,
        col_start8,
        col_end8,
        row_start8,
        row_end8,
    ) {
        load_tmvs_skaler_devret(
            rp_proj,
            stride,
            iw8,
            ih8,
            plan,
            refs,
            tile_row_idx,
            col_start8,
            col_end8,
            row_start8,
            row_end8,
        );
        return;
    }
    // SAFETY: (1) ISA: bu işlevin adresi yalnız `Sse41Token` ile verilir
    // (`load_tmvs_sse41_fn`); belirteç SSE4.1 tespiti + maske izninin kanıtıdır.
    // (2) sınır: `load_tmvs_sozlesmesi` `rp_proj`, `refs` ve `plan` sınırlarını
    // O(1) doğruladı. (3) hizalama/takma ad: tüm erişimler sınır denetimli dilim
    // indekslemesiyle ve yazmaç içi SSE4.1 işlemleriyle yapılır; `rp_proj` `&mut`,
    // `refs` ve `plan` `&` ayrık referanslardır.
    unsafe {
        load_tmvs_sse41_giris(
            rp_proj,
            stride,
            iw8,
            ih8,
            plan,
            refs,
            tile_row_idx,
            col_start8,
            col_end8,
            row_start8,
            row_end8,
        )
    }
}

#[target_feature(enable = "sse4.1")]
#[inline]
fn precomp_frac_fast(ref2cur: i32) -> [__m128i; 32] {
    let m4 = _mm_set1_epi32(ref2cur);
    let mut out = [_mm_set1_epi32(0); 32];
    let mut g = 0usize;
    while g < 8 {
        let base = g * 4;
        let v = _mm_setr_epi32(
            MV_PROJ[base],
            MV_PROJ[base + 1],
            MV_PROJ[base + 2],
            MV_PROJ[base + 3],
        );
        let prod = _mm_mullo_epi32(v, m4);
        out[base] = _mm_shuffle_epi32::<0x00>(prod);
        out[base + 1] = _mm_shuffle_epi32::<0x55>(prod);
        out[base + 2] = _mm_shuffle_epi32::<0xAA>(prod);
        out[base + 3] = _mm_shuffle_epi32::<0xFF>(prod);
        g += 1;
    }
    out
}

#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
fn load_tmvs_sse41_giris(
    rp_proj: &mut [TemporalBlock],
    stride: usize,
    iw8: usize,
    ih8: usize,
    plan: &MfmvPlan,
    refs: &[Option<&[TemporalBlock]>; 7],
    tile_row_idx: usize,
    col_start8: usize,
    col_end8: usize,
    row_start8: usize,
    row_end8: usize,
) {
    let row_end8 = row_end8.min(ih8);
    if row_end8 <= row_start8 {
        return;
    }
    let col_start8i = col_start8.saturating_sub(8);
    let col_end8i = (col_end8 + 8).min(iw8);
    let base = 16 * stride * tile_row_idx;
    let row_off = (row_start8 & 15) * stride;

    // `src/x86/refmvs.asm:422-437`: bölgeyi geçersiz kıl (`mv = INVALID_MV`, `ref_` korunur).
    for y in row_start8..row_end8 {
        let off = base + row_off + (y - row_start8) * stride;
        for slot in &mut rp_proj[off + col_start8..off + col_end8] {
            slot.mv = Mv::INVALID;
        }
    }

    let m3 = _mm_set_epi64x(8192, 8192);
    for n in 0..plan.n_mfmvs.min(3) {
        let ref2cur = plan.mfmv_ref2cur[n];
        if ref2cur == INVALID_REF2CUR {
            continue;
        }
        let ref_idx = usize::from(plan.mfmv_ref[n]);
        let Some(r) = refs[ref_idx] else {
            continue;
        };
        let fracs = precomp_frac_fast(i32::from(ref2cur));
        let ref_sign = (ref_idx as i64) - 4;
        let m2 = _mm_set_epi64x(0, ref_sign);

        for y in row_start8..row_end8 {
            let y_sb_align = y & !7;
            let y_proj_start = y_sb_align.max(row_start8) as i32;
            let y_proj_end = (y_sb_align + 8).min(row_end8) as i32;
            let r_row = &r[y * stride..y * stride + col_end8i];
            let mut x = col_start8i;
            while x < col_end8i {
                let rb = r_row[x];
                let b_ref = rb.ref_;
                if b_ref == 0 || b_ref > 7 {
                    x += 1;
                    continue;
                }
                let r2r = plan.mfmv_ref2ref[n][usize::from(b_ref) - 1];
                if r2r == 0 {
                    x += 1;
                    continue;
                }
                let b_mv = rb.mv;
                // `src/x86/refmvs.asm:504-520`:
                // `pmovsxwq m0, [rbq]` + `pmuldq m0, m1` + `(xy + (xy >> 31) + 8192) >> 14`
                // + `packssdw` (±0x3fff doğal `i16` doygunluğu, `>> 6` sonrası özdeş)
                // + `pabsd` + `psrld 6` + `packuswb` + `pxor m0, m2` + `psignd m1, m0`.
                let m0_in = _mm_set_epi64x(i64::from(b_mv.x), i64::from(b_mv.y));
                let m1_frac = fracs[usize::from(r2r)];
                let prod = _mm_mul_epi32(m0_in, m1_frac);
                let sign_hi = _mm_shuffle_epi32::<0xF5>(prod);
                let sum = _mm_add_epi32(_mm_add_epi32(prod, m3), sign_hi);
                let mut m0 = _mm_srai_epi32::<14>(sum);
                let mut m1 = _mm_abs_epi32(m0);
                m0 = _mm_packs_epi32(m0, m0);
                m1 = _mm_srli_epi32::<6>(m1);
                m1 = _mm_packus_epi16(m1, m1);
                m0 = _mm_xor_si128(m0, m2);
                m1 = _mm_sign_epi32(m1, m0);
                let mvxy = _mm_cvtsi128_si64(m1);
                let off_y = mvxy as i32;
                let off_x = (mvxy >> 32) as i32;
                let pos_y = (y as i32).wrapping_add(off_y);

                let mut run = 1usize;
                while x + run < col_end8i {
                    let next = r_row[x + run];
                    if next.ref_ != b_ref || next.mv != b_mv {
                        break;
                    }
                    run += 1;
                }

                if pos_y >= y_proj_start && pos_y < y_proj_end {
                    let pos = base + ((pos_y as usize) & 15) * stride;
                    let pos_x0 = (x as i32).wrapping_add(off_x);
                    let tmpl = TemporalBlock {
                        mv: b_mv,
                        ref_: r2r,
                    };
                    for k in 0..run {
                        let xx = x + k;
                        let x_sb_align = xx & !7;
                        let px = pos_x0 + (k as i32);
                        let lo = x_sb_align.saturating_sub(8).max(col_start8) as i32;
                        let hi = (x_sb_align + 16).min(col_end8) as i32;
                        if px >= lo && px < hi {
                            rp_proj[pos + (px as usize)] = tmpl;
                        }
                    }
                }
                x += run;
            }
        }
    }
}
