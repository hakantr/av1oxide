//! `aarch64` (`ARM64` NEON / DOTPROD) ISA backend'i (`F5a` / `P51` — `WP-08`).
//!
//! ## Mimari ve Güvenlik Sözleşmesi
//!
//! - `ARM_ISA_CANDIDATES` tablosu tüm hedeflerde derlenir (böylece `x86_64`
//!   üzerinde koşan `av1oxide-harness dispatch-map` `arm64` yuva eşlemesini
//!   denetleyebilir).
//! - Gerçek `core::arch::aarch64` NEON çekirdekleri (`cdef`, `cdef16`, `mem`,
//!   `pal`, `refmvs`) `#[cfg(target_arch = "aarch64")]` altındadır.
//! - Yalnız `mem` ve `giris` modülleri sabit boyutlu dizi referansları (`&[T; N]`) üzerinde
//!   `vld1*`/`vst1*` ve `NeonToken`-kapılı `#[target_feature(enable = "neon")]`
//!   giriş köprüleri için `unsafe` içerir (`audit::UNSAFE_EXEMPTIONS` kaydı);
//!   bu nedenle bu modülde `forbid` yerine `#![deny(unsafe_code)]` vardır.
//!   `cdef`, `cdef16`, `pal` ve `refmvs` modülleri `%100 #![forbid(unsafe_code)]`
//!   ile derlenir ve [`NeonToken`](crate::cpu::NeonToken) yetki belirteciyle kapılanır.

#![deny(unsafe_code)]

#[cfg(target_arch = "aarch64")]
pub mod cdef;
#[cfg(target_arch = "aarch64")]
pub mod cdef16;
#[cfg(target_arch = "aarch64")]
pub mod fg;
#[cfg(target_arch = "aarch64")]
pub mod giris;
#[cfg(target_arch = "aarch64")]
pub mod ipred;
#[cfg(target_arch = "aarch64")]
pub mod itx;
#[cfg(target_arch = "aarch64")]
pub mod itx_1d;
pub mod itx_alanlari;
#[cfg(target_arch = "aarch64")]
pub mod lf;
#[cfg(target_arch = "aarch64")]
pub mod lr;
#[cfg(target_arch = "aarch64")]
pub mod mc;
#[cfg(target_arch = "aarch64")]
pub mod mc_olcekli;
#[cfg(target_arch = "aarch64")]
pub mod mem;
#[cfg(target_arch = "aarch64")]
pub mod msac;
#[cfg(target_arch = "aarch64")]
pub mod pal;
#[cfg(target_arch = "aarch64")]
pub mod refmvs;
#[cfg(target_arch = "aarch64")]
pub mod resize;
#[cfg(all(target_arch = "aarch64", test))]
mod sozluk;

use crate::cpu::CpuFlags;
use crate::dsp::IsaCandidate;
#[cfg(target_arch = "aarch64")]
use crate::{
    cpu::{DotprodToken, NeonToken},
    dsp::IsaEntry,
};

const ARM_NEON_KARAR: &str = "varsayılan (P51 / F5a WP-08, aarch64 neon): %100 #![forbid(unsafe_code)] kararlı core::arch::aarch64 NEON çekirdeği; NeonToken yetki belirteciyle kapılanır (rust_method: core::arch)";

const fn arm_neon_aday(family: &'static str, field: &'static str, bit_depth: u8) -> IsaCandidate {
    IsaCandidate {
        family,
        field,
        bit_depth,
        implementation: "neon",
        requires: CpuFlags::NEON,
        default: true,
        decision: ARM_NEON_KARAR,
    }
}

const ARM_DOTPROD_KARAR: &str = "varsayılan (P52 / F5b WP-08, aarch64 neon_dotprod): 8 bpc 8-tap mc/mct yatay geçişi kararlı vdotq_s32 (sdot) ile; DotprodToken yetki belirteciyle kapılanır, isa_rank 2 > neon 1 (rust_method: core::arch)";

const fn arm_dotprod_aday(field: &'static str) -> IsaCandidate {
    IsaCandidate {
        family: "mc",
        field,
        bit_depth: 8,
        implementation: "neon_dotprod",
        requires: CpuFlags::NEON.union(CpuFlags::DOTPROD),
        default: true,
        decision: ARM_DOTPROD_KARAR,
    }
}

/// `mc` NEON adayları (bir bit derinliği için `52` alan: `32` temel + `20` ölçekli).
macro_rules! mc_neon_adaylari {
    ($bd:literal) => {
        [
            arm_neon_aday("mc", "mc[FILTER_2D_8TAP_REGULAR]", $bd),
            arm_neon_aday("mc", "mc[FILTER_2D_8TAP_REGULAR_SMOOTH]", $bd),
            arm_neon_aday("mc", "mc[FILTER_2D_8TAP_REGULAR_SHARP]", $bd),
            arm_neon_aday("mc", "mc[FILTER_2D_8TAP_SHARP_REGULAR]", $bd),
            arm_neon_aday("mc", "mc[FILTER_2D_8TAP_SHARP_SMOOTH]", $bd),
            arm_neon_aday("mc", "mc[FILTER_2D_8TAP_SHARP]", $bd),
            arm_neon_aday("mc", "mc[FILTER_2D_8TAP_SMOOTH_REGULAR]", $bd),
            arm_neon_aday("mc", "mc[FILTER_2D_8TAP_SMOOTH]", $bd),
            arm_neon_aday("mc", "mc[FILTER_2D_8TAP_SMOOTH_SHARP]", $bd),
            arm_neon_aday("mc", "mc[FILTER_2D_BILINEAR]", $bd),
            arm_neon_aday("mc", "mct[FILTER_2D_8TAP_REGULAR]", $bd),
            arm_neon_aday("mc", "mct[FILTER_2D_8TAP_REGULAR_SMOOTH]", $bd),
            arm_neon_aday("mc", "mct[FILTER_2D_8TAP_REGULAR_SHARP]", $bd),
            arm_neon_aday("mc", "mct[FILTER_2D_8TAP_SHARP_REGULAR]", $bd),
            arm_neon_aday("mc", "mct[FILTER_2D_8TAP_SHARP_SMOOTH]", $bd),
            arm_neon_aday("mc", "mct[FILTER_2D_8TAP_SHARP]", $bd),
            arm_neon_aday("mc", "mct[FILTER_2D_8TAP_SMOOTH_REGULAR]", $bd),
            arm_neon_aday("mc", "mct[FILTER_2D_8TAP_SMOOTH]", $bd),
            arm_neon_aday("mc", "mct[FILTER_2D_8TAP_SMOOTH_SHARP]", $bd),
            arm_neon_aday("mc", "mct[FILTER_2D_BILINEAR]", $bd),
            arm_neon_aday("mc", "mc_scaled[FILTER_2D_8TAP_REGULAR]", $bd),
            arm_neon_aday("mc", "mc_scaled[FILTER_2D_8TAP_REGULAR_SMOOTH]", $bd),
            arm_neon_aday("mc", "mc_scaled[FILTER_2D_8TAP_REGULAR_SHARP]", $bd),
            arm_neon_aday("mc", "mc_scaled[FILTER_2D_8TAP_SHARP_REGULAR]", $bd),
            arm_neon_aday("mc", "mc_scaled[FILTER_2D_8TAP_SHARP_SMOOTH]", $bd),
            arm_neon_aday("mc", "mc_scaled[FILTER_2D_8TAP_SHARP]", $bd),
            arm_neon_aday("mc", "mc_scaled[FILTER_2D_8TAP_SMOOTH_REGULAR]", $bd),
            arm_neon_aday("mc", "mc_scaled[FILTER_2D_8TAP_SMOOTH]", $bd),
            arm_neon_aday("mc", "mc_scaled[FILTER_2D_8TAP_SMOOTH_SHARP]", $bd),
            arm_neon_aday("mc", "mc_scaled[FILTER_2D_BILINEAR]", $bd),
            arm_neon_aday("mc", "mct_scaled[FILTER_2D_8TAP_REGULAR]", $bd),
            arm_neon_aday("mc", "mct_scaled[FILTER_2D_8TAP_REGULAR_SMOOTH]", $bd),
            arm_neon_aday("mc", "mct_scaled[FILTER_2D_8TAP_REGULAR_SHARP]", $bd),
            arm_neon_aday("mc", "mct_scaled[FILTER_2D_8TAP_SHARP_REGULAR]", $bd),
            arm_neon_aday("mc", "mct_scaled[FILTER_2D_8TAP_SHARP_SMOOTH]", $bd),
            arm_neon_aday("mc", "mct_scaled[FILTER_2D_8TAP_SHARP]", $bd),
            arm_neon_aday("mc", "mct_scaled[FILTER_2D_8TAP_SMOOTH_REGULAR]", $bd),
            arm_neon_aday("mc", "mct_scaled[FILTER_2D_8TAP_SMOOTH]", $bd),
            arm_neon_aday("mc", "mct_scaled[FILTER_2D_8TAP_SMOOTH_SHARP]", $bd),
            arm_neon_aday("mc", "mct_scaled[FILTER_2D_BILINEAR]", $bd),
            arm_neon_aday("mc", "avg", $bd),
            arm_neon_aday("mc", "w_avg", $bd),
            arm_neon_aday("mc", "mask", $bd),
            arm_neon_aday("mc", "w_mask[0]", $bd),
            arm_neon_aday("mc", "w_mask[1]", $bd),
            arm_neon_aday("mc", "w_mask[2]", $bd),
            arm_neon_aday("mc", "blend", $bd),
            arm_neon_aday("mc", "blend_v", $bd),
            arm_neon_aday("mc", "blend_h", $bd),
            arm_neon_aday("mc", "warp8x8", $bd),
            arm_neon_aday("mc", "warp8x8t", $bd),
            arm_neon_aday("mc", "emu_edge", $bd),
        ]
    };
}

/// P51 (`F5a`) adayları: CDEF 12 + REFMVS 3 + PAL 1.
const F5A_ADAYLARI: [IsaCandidate; 16] = [
    // CDEF (4 işlev × 3 bit derinliği = 12 yuva)
    arm_neon_aday("cdef", "dir", 8),
    arm_neon_aday("cdef", "fb[0]", 8),
    arm_neon_aday("cdef", "fb[1]", 8),
    arm_neon_aday("cdef", "fb[2]", 8),
    arm_neon_aday("cdef", "dir", 10),
    arm_neon_aday("cdef", "fb[0]", 10),
    arm_neon_aday("cdef", "fb[1]", 10),
    arm_neon_aday("cdef", "fb[2]", 10),
    arm_neon_aday("cdef", "dir", 12),
    arm_neon_aday("cdef", "fb[0]", 12),
    arm_neon_aday("cdef", "fb[1]", 12),
    arm_neon_aday("cdef", "fb[2]", 12),
    // REFMVS (3 yuva, bit derinliğinden bağımsız)
    arm_neon_aday("refmvs", "load_tmvs", 0),
    arm_neon_aday("refmvs", "save_tmvs", 0),
    arm_neon_aday("refmvs", "splat_mv", 0),
    // PAL (1 yuva, bit derinliğinden bağımsız)
    arm_neon_aday("pal", "pal_idx_finish", 0),
];

/// MC NEON adayları: `3 × 52 = 156` NEON + `18` DOTPROD.
const F5B_MC_NEON: [[IsaCandidate; 52]; 3] = [
    mc_neon_adaylari!(8),
    mc_neon_adaylari!(10),
    mc_neon_adaylari!(12),
];

/// SuperRes `resize/resize` NEON adayları (`8 / 10 / 12` bpc, `3` yuva).
const RESIZE_NEON: [IsaCandidate; 3] = [
    arm_neon_aday("resize", "resize", 8),
    arm_neon_aday("resize", "resize", 10),
    arm_neon_aday("resize", "resize", 12),
];

const F5B_MC_DOTPROD: [IsaCandidate; 18] = [
    arm_dotprod_aday("mc[FILTER_2D_8TAP_REGULAR]"),
    arm_dotprod_aday("mc[FILTER_2D_8TAP_REGULAR_SMOOTH]"),
    arm_dotprod_aday("mc[FILTER_2D_8TAP_REGULAR_SHARP]"),
    arm_dotprod_aday("mc[FILTER_2D_8TAP_SHARP_REGULAR]"),
    arm_dotprod_aday("mc[FILTER_2D_8TAP_SHARP_SMOOTH]"),
    arm_dotprod_aday("mc[FILTER_2D_8TAP_SHARP]"),
    arm_dotprod_aday("mc[FILTER_2D_8TAP_SMOOTH_REGULAR]"),
    arm_dotprod_aday("mc[FILTER_2D_8TAP_SMOOTH]"),
    arm_dotprod_aday("mc[FILTER_2D_8TAP_SMOOTH_SHARP]"),
    arm_dotprod_aday("mct[FILTER_2D_8TAP_REGULAR]"),
    arm_dotprod_aday("mct[FILTER_2D_8TAP_REGULAR_SMOOTH]"),
    arm_dotprod_aday("mct[FILTER_2D_8TAP_REGULAR_SHARP]"),
    arm_dotprod_aday("mct[FILTER_2D_8TAP_SHARP_REGULAR]"),
    arm_dotprod_aday("mct[FILTER_2D_8TAP_SHARP_SMOOTH]"),
    arm_dotprod_aday("mct[FILTER_2D_8TAP_SHARP]"),
    arm_dotprod_aday("mct[FILTER_2D_8TAP_SMOOTH_REGULAR]"),
    arm_dotprod_aday("mct[FILTER_2D_8TAP_SMOOTH]"),
    arm_dotprod_aday("mct[FILTER_2D_8TAP_SMOOTH_SHARP]"),
];

/// P53 (`F5c`) ITX adayı (`neon`, alan `itx_alanlari::ITX_NEON_ALANLARI`).
const fn itx_aday(field: &'static str, bit_depth: u8) -> IsaCandidate {
    arm_neon_aday("itx", field, bit_depth)
}

/// `ipred` NEON adayları (bir bit derinliği için `22` alan; dav1d `src/arm/ipred.h`).
macro_rules! ipred_neon_adaylari {
    ($bd:literal) => {
        [
            arm_neon_aday("ipred", "intra_pred[DC_PRED]", $bd),
            arm_neon_aday("ipred", "intra_pred[VERT_PRED]", $bd),
            arm_neon_aday("ipred", "intra_pred[HOR_PRED]", $bd),
            arm_neon_aday("ipred", "intra_pred[LEFT_DC_PRED]", $bd),
            arm_neon_aday("ipred", "intra_pred[TOP_DC_PRED]", $bd),
            arm_neon_aday("ipred", "intra_pred[DC_128_PRED]", $bd),
            arm_neon_aday("ipred", "intra_pred[Z1_PRED]", $bd),
            arm_neon_aday("ipred", "intra_pred[Z2_PRED]", $bd),
            arm_neon_aday("ipred", "intra_pred[Z3_PRED]", $bd),
            arm_neon_aday("ipred", "intra_pred[SMOOTH_PRED]", $bd),
            arm_neon_aday("ipred", "intra_pred[SMOOTH_V_PRED]", $bd),
            arm_neon_aday("ipred", "intra_pred[SMOOTH_H_PRED]", $bd),
            arm_neon_aday("ipred", "intra_pred[PAETH_PRED]", $bd),
            arm_neon_aday("ipred", "intra_pred[FILTER_PRED]", $bd),
            arm_neon_aday("ipred", "cfl_ac[DAV1D_PIXEL_LAYOUT_I420-1]", $bd),
            arm_neon_aday("ipred", "cfl_ac[DAV1D_PIXEL_LAYOUT_I422-1]", $bd),
            arm_neon_aday("ipred", "cfl_ac[DAV1D_PIXEL_LAYOUT_I444-1]", $bd),
            arm_neon_aday("ipred", "cfl_pred[DC_PRED]", $bd),
            arm_neon_aday("ipred", "cfl_pred[LEFT_DC_PRED]", $bd),
            arm_neon_aday("ipred", "cfl_pred[TOP_DC_PRED]", $bd),
            arm_neon_aday("ipred", "cfl_pred[DC_128_PRED]", $bd),
            arm_neon_aday("ipred", "pal_pred", $bd),
        ]
    };
}

/// `filmgrain` NEON adayları (bir bit derinliği için `8` alan; dav1d `src/arm/filmgrain.h`).
macro_rules! fg_neon_adaylari {
    ($bd:literal) => {
        [
            arm_neon_aday("filmgrain", "generate_grain_y", $bd),
            arm_neon_aday(
                "filmgrain",
                "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I420-1]",
                $bd,
            ),
            arm_neon_aday(
                "filmgrain",
                "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I422-1]",
                $bd,
            ),
            arm_neon_aday(
                "filmgrain",
                "generate_grain_uv[DAV1D_PIXEL_LAYOUT_I444-1]",
                $bd,
            ),
            arm_neon_aday("filmgrain", "fgy_32x32xn", $bd),
            arm_neon_aday("filmgrain", "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I420-1]", $bd),
            arm_neon_aday("filmgrain", "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I422-1]", $bd),
            arm_neon_aday("filmgrain", "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I444-1]", $bd),
        ]
    };
}

/// `loopfilter` NEON adayları (bir bit derinliği için `4` alan; dav1d `src/arm/loopfilter.h`).
macro_rules! lf_neon_adaylari {
    ($bd:literal) => {
        [
            arm_neon_aday("loopfilter", "loop_filter_sb[0][0]", $bd),
            arm_neon_aday("loopfilter", "loop_filter_sb[0][1]", $bd),
            arm_neon_aday("loopfilter", "loop_filter_sb[1][0]", $bd),
            arm_neon_aday("loopfilter", "loop_filter_sb[1][1]", $bd),
        ]
    };
}

const F5D_IPRED_NEON: [[IsaCandidate; 22]; 3] = [
    ipred_neon_adaylari!(8),
    ipred_neon_adaylari!(10),
    ipred_neon_adaylari!(12),
];

const F5D_FG_NEON: [[IsaCandidate; 8]; 3] = [
    fg_neon_adaylari!(8),
    fg_neon_adaylari!(10),
    fg_neon_adaylari!(12),
];

const F5D_LF_NEON: [[IsaCandidate; 4]; 3] = [
    lf_neon_adaylari!(8),
    lf_neon_adaylari!(10),
    lf_neon_adaylari!(12),
];

/// LR NEON adayları: Wiener @ `8/10/12` (`6` yuva) + SGR @ `8/10/12` (`9` yuva; toplam `15`).
const F5D_LR_NEON: [IsaCandidate; 15] = [
    arm_neon_aday("looprestoration", "wiener[0]", 8),
    arm_neon_aday("looprestoration", "wiener[1]", 8),
    arm_neon_aday("looprestoration", "sgr[0]", 8),
    arm_neon_aday("looprestoration", "sgr[1]", 8),
    arm_neon_aday("looprestoration", "sgr[2]", 8),
    arm_neon_aday("looprestoration", "wiener[0]", 10),
    arm_neon_aday("looprestoration", "wiener[1]", 10),
    arm_neon_aday("looprestoration", "sgr[0]", 10),
    arm_neon_aday("looprestoration", "sgr[1]", 10),
    arm_neon_aday("looprestoration", "sgr[2]", 10),
    arm_neon_aday("looprestoration", "wiener[0]", 12),
    arm_neon_aday("looprestoration", "wiener[1]", 12),
    arm_neon_aday("looprestoration", "sgr[0]", 12),
    arm_neon_aday("looprestoration", "sgr[1]", 12),
    arm_neon_aday("looprestoration", "sgr[2]", 12),
];

/// P54 (`F5d`) MSAC NEON adayları (`7` yuva, `bit_depth == 0`; `src/arm/64/msac.S`).
const F5D_MSAC_NEON: [IsaCandidate; 7] = [
    arm_neon_aday("msac", "decode_symbol_adapt4", 0),
    arm_neon_aday("msac", "decode_symbol_adapt8", 0),
    arm_neon_aday("msac", "decode_symbol_adapt16", 0),
    arm_neon_aday("msac", "decode_bool_adapt", 0),
    arm_neon_aday("msac", "decode_bool_equi", 0),
    arm_neon_aday("msac", "decode_bool", 0),
    arm_neon_aday("msac", "decode_hi_tok", 0),
];

/// Sabit dizileri tek bir `'static` dilime birleştirir (derleme zamanı).
const fn birlestir<const N: usize>() -> [IsaCandidate; N] {
    let mut out = [F5A_ADAYLARI[0]; N];
    let mut i = 0;
    let mut j = 0;
    while j < F5A_ADAYLARI.len() {
        out[i] = F5A_ADAYLARI[j];
        i += 1;
        j += 1;
    }
    let mut b = 0;
    while b < 3 {
        let mut k = 0;
        while k < 52 {
            out[i] = F5B_MC_NEON[b][k];
            i += 1;
            k += 1;
        }
        b += 1;
    }
    let mut r_i = 0;
    while r_i < RESIZE_NEON.len() {
        out[i] = RESIZE_NEON[r_i];
        i += 1;
        r_i += 1;
    }
    let mut d = 0;
    while d < F5B_MC_DOTPROD.len() {
        out[i] = F5B_MC_DOTPROD[d];
        i += 1;
        d += 1;
    }
    let bdler = [8u8, 10, 12];
    let mut b2 = 0;
    while b2 < 3 {
        let mut f = 0;
        while f < itx_alanlari::ITX_NEON_ALANLARI.len() {
            out[i] = itx_aday(itx_alanlari::ITX_NEON_ALANLARI[f], bdler[b2]);
            i += 1;
            f += 1;
        }
        b2 += 1;
    }
    let mut b3 = 0;
    while b3 < 3 {
        let mut k = 0;
        while k < 22 {
            out[i] = F5D_IPRED_NEON[b3][k];
            i += 1;
            k += 1;
        }
        b3 += 1;
    }
    let mut b4 = 0;
    while b4 < 3 {
        let mut k = 0;
        while k < 8 {
            out[i] = F5D_FG_NEON[b4][k];
            i += 1;
            k += 1;
        }
        b4 += 1;
    }
    let mut b5 = 0;
    while b5 < 3 {
        let mut k = 0;
        while k < 4 {
            out[i] = F5D_LF_NEON[b5][k];
            i += 1;
            k += 1;
        }
        b5 += 1;
    }
    let mut lr_i = 0;
    while lr_i < F5D_LR_NEON.len() {
        out[i] = F5D_LR_NEON[lr_i];
        i += 1;
        lr_i += 1;
    }
    let mut ms_i = 0;
    while ms_i < F5D_MSAC_NEON.len() {
        out[i] = F5D_MSAC_NEON[ms_i];
        i += 1;
        ms_i += 1;
    }
    assert!(i == N, "ARM_ISA_CANDIDATES uzunluğu");
    out
}

/// Toplam `aarch64` aday sayısı (`16` F5a + `156` MC NEON + `3` Resize NEON + `18` MC DOTPROD
/// + `468` ITX NEON + `66` IPRED + `24` FG + `12` LF + `15` LR + `7` MSAC = `785` aday, `767` tekil yuva).
const ARM_ADAY_SAYISI: usize = 16 + 3 * 52 + 3 + 18 + 3 * 156 + 3 * 22 + 3 * 8 + 3 * 4 + 15 + 7;
static ARM_ADAYLAR: [IsaCandidate; ARM_ADAY_SAYISI] = birlestir::<ARM_ADAY_SAYISI>();

/// `aarch64` ISA adayları (`785` aday / `767` tekil `DspDispatch` yuvası):
/// - `F5a` / `P51`: `F5-CDEF` (`12` yuva), `F5-REFMVS` (`3`), `F5-PAL` (`1`)
/// - `F5b` / `P52` + `4.A.3`: `F5-MC` `neon` (`156` yuva: `96` temel + `60` ölçekli)
///   + `resize` (`3` yuva) + `neon_dotprod` (`18` üst kademe adayı)
/// - `F5c` / `P53`: `F5-ITX` `neon` (`468` yuva, `WHT_WHT` dâhil)
/// - `F5d` / `P54` + `4.A.3`: `F5-IPRED` (`66`), `F5-FG` (`24`), `F5-LF` (`12`), `F5-LR` (`15`), `F5-MSAC` (`7`)
pub const ARM_ISA_CANDIDATES: &[IsaCandidate] = &ARM_ADAYLAR;

/// `DspDispatch::candidate_entry` için `aarch64` NEON yuva eşlemesi.
#[cfg(target_arch = "aarch64")]
#[must_use]
pub(super) fn candidate_entry(
    candidate: &IsaCandidate,
    neon: NeonToken,
    dotprod: Option<DotprodToken>,
) -> Option<IsaEntry> {
    if candidate.implementation == "neon_dotprod" {
        return mc_dotprod_entry(candidate, dotprod?);
    }
    if candidate.implementation != "neon" {
        return None;
    }
    match candidate.family {
        "mc" => return mc_neon_entry(candidate, neon),
        "resize" if candidate.field == "resize" => {
            let f = match candidate.bit_depth {
                8 => resize::resize_fn::<8>(neon),
                10 => resize::resize_fn::<10>(neon),
                12 => resize::resize_fn::<12>(neon),
                _ => return None,
            };
            return Some(IsaEntry::Resize(f));
        }
        "itx" => return itx_neon_entry(candidate, neon),
        "ipred" => return ipred_neon_entry(candidate, neon),
        "filmgrain" => return fg_neon_entry(candidate, neon),
        "loopfilter" => return lf_neon_entry(candidate, neon),
        "looprestoration" => return lr_neon_entry(candidate, neon),
        "msac" if candidate.bit_depth == 0 => {
            return match candidate.field {
                "decode_symbol_adapt4"
                | "decode_symbol_adapt8"
                | "decode_symbol_adapt16"
                | "decode_bool_adapt"
                | "decode_bool_equi"
                | "decode_bool"
                | "decode_hi_tok" => Some(IsaEntry::Msac),
                _ => None,
            };
        }
        _ => {}
    }
    match (candidate.family, candidate.field, candidate.bit_depth) {
        ("cdef", "dir", 8) => Some(IsaEntry::CdefDir(cdef::find_dir_neon_fn(neon))),
        ("cdef", "fb[0]", 8) => Some(IsaEntry::CdefFb(0, cdef::filter_8x8_neon_fn(neon))),
        ("cdef", "fb[1]", 8) => Some(IsaEntry::CdefFb(1, cdef::filter_4x8_neon_fn(neon))),
        ("cdef", "fb[2]", 8) => Some(IsaEntry::CdefFb(2, cdef::filter_4x4_neon_fn(neon))),
        ("cdef", "dir", 10) => Some(IsaEntry::CdefDir(cdef16::find_dir_neon_fn::<10>(neon))),
        ("cdef", "fb[0]", 10) => Some(IsaEntry::CdefFb(0, cdef16::filter_8x8_neon_fn::<10>(neon))),
        ("cdef", "fb[1]", 10) => Some(IsaEntry::CdefFb(1, cdef16::filter_4x8_neon_fn::<10>(neon))),
        ("cdef", "fb[2]", 10) => Some(IsaEntry::CdefFb(2, cdef16::filter_4x4_neon_fn::<10>(neon))),
        ("cdef", "dir", 12) => Some(IsaEntry::CdefDir(cdef16::find_dir_neon_fn::<12>(neon))),
        ("cdef", "fb[0]", 12) => Some(IsaEntry::CdefFb(0, cdef16::filter_8x8_neon_fn::<12>(neon))),
        ("cdef", "fb[1]", 12) => Some(IsaEntry::CdefFb(1, cdef16::filter_4x8_neon_fn::<12>(neon))),
        ("cdef", "fb[2]", 12) => Some(IsaEntry::CdefFb(2, cdef16::filter_4x4_neon_fn::<12>(neon))),
        ("refmvs", "load_tmvs", 0) => {
            Some(IsaEntry::RefmvsLoadTmvs(refmvs::load_tmvs_neon_fn(neon)))
        }
        ("refmvs", "save_tmvs", 0) => {
            Some(IsaEntry::RefmvsSaveTmvs(refmvs::save_tmvs_neon_fn(neon)))
        }
        ("refmvs", "splat_mv", 0) => Some(IsaEntry::RefmvsSplatMv(refmvs::splat_mv_neon_fn(neon))),
        ("pal", "pal_idx_finish", 0) => {
            Some(IsaEntry::PalIdxFinish(pal::pal_idx_finish_neon_fn(neon)))
        }
        _ => None,
    }
}

/// `mc[...]` / `mct[...]` alan adından `(prep, filtre dizini)`.
#[cfg(target_arch = "aarch64")]
fn mc_alan(field: &str) -> Option<(bool, usize)> {
    let (prep, ad) = if let Some(r) = field.strip_prefix("mct[") {
        (true, r)
    } else {
        (false, field.strip_prefix("mc[")?)
    };
    let ad = ad.strip_suffix(']')?;
    let i = super::FILTER_2D_NAMES.iter().position(|n| *n == ad)?;
    Some((prep, i))
}

/// `mc_scaled[...]` / `mct_scaled[...]` alan adından `(prep, filtre dizini)`.
#[cfg(target_arch = "aarch64")]
fn mc_olcekli_alan(field: &str) -> Option<(bool, usize)> {
    let (prep, ad) = if let Some(r) = field.strip_prefix("mct_scaled[") {
        (true, r)
    } else {
        (false, field.strip_prefix("mc_scaled[")?)
    };
    let ad = ad.strip_suffix(']')?;
    let i = super::FILTER_2D_NAMES.iter().position(|n| *n == ad)?;
    Some((prep, i))
}

/// `mc` NEON yuva eşlemesi (`8 / 10 / 12` bpc, P52 + 4.A.3).
#[cfg(target_arch = "aarch64")]
fn mc_neon_entry(candidate: &IsaCandidate, neon: NeonToken) -> Option<IsaEntry> {
    macro_rules! bpc {
        ($b:literal) => {{
            Some(match candidate.field {
                "avg" => IsaEntry::Avg(mc::avg_fn::<$b>(neon)),
                "w_avg" => IsaEntry::WAvg(mc::w_avg_fn::<$b>(neon)),
                "mask" => IsaEntry::Mask(mc::mask_fn::<$b>(neon)),
                "blend" => IsaEntry::Blend(mc::blend_fn::<$b>(neon)),
                "blend_v" => IsaEntry::BlendV(mc::blend_v_fn::<$b>(neon)),
                "blend_h" => IsaEntry::BlendH(mc::blend_h_fn::<$b>(neon)),
                "warp8x8" => IsaEntry::Warp(mc::warp_fn::<$b>(neon)),
                "warp8x8t" => IsaEntry::WarpT(mc::warp_t_fn::<$b>(neon)),
                "emu_edge" => IsaEntry::EmuEdge(mc::emu_edge_fn::<$b>(neon)),
                "w_mask[0]" => IsaEntry::WMask(0, mc::w_mask_fn::<$b>(neon, 0)),
                "w_mask[1]" => IsaEntry::WMask(1, mc::w_mask_fn::<$b>(neon, 1)),
                "w_mask[2]" => IsaEntry::WMask(2, mc::w_mask_fn::<$b>(neon, 2)),
                f => {
                    if let Some((prep, i)) = mc_olcekli_alan(f) {
                        if prep {
                            IsaEntry::MctScaled(i, mc_olcekli::prep_scaled_fn::<$b>(neon, i))
                        } else {
                            IsaEntry::McScaled(i, mc_olcekli::put_scaled_fn::<$b>(neon, i))
                        }
                    } else {
                        let (prep, i) = mc_alan(f)?;
                        if prep {
                            IsaEntry::Mct(i, mc::prep_fn::<$b>(neon, i))
                        } else {
                            IsaEntry::Mc(i, mc::put_fn::<$b>(neon, i))
                        }
                    }
                }
            })
        }};
    }
    match candidate.bit_depth {
        8 => bpc!(8),
        10 => bpc!(10),
        12 => bpc!(12),
        _ => None,
    }
}

/// `mc` 8 bpc DOTPROD yuva eşlemesi (yalnız 8-tap `mc[0..8]` / `mct[0..8]`, P52).
#[cfg(target_arch = "aarch64")]
fn mc_dotprod_entry(candidate: &IsaCandidate, dot: DotprodToken) -> Option<IsaEntry> {
    if candidate.family != "mc" || candidate.bit_depth != 8 {
        return None;
    }
    let (prep, i) = mc_alan(candidate.field)?;
    if i >= mc::BILIN {
        return None;
    }
    Some(if prep {
        IsaEntry::Mct(i, mc::prep_dot_fn(dot, i))
    } else {
        IsaEntry::Mc(i, mc::put_dot_fn(dot, i))
    })
}

/// `itx` NEON yuva eşlemesi (`itxfm_add[TX_..][..]`, P53).
#[cfg(target_arch = "aarch64")]
fn itx_neon_entry(candidate: &IsaCandidate, neon: NeonToken) -> Option<IsaEntry> {
    use crate::dsp::itx::{TX_SIZES, TX_TYPES, TxType};
    let ic = candidate
        .field
        .strip_prefix("itxfm_add[")?
        .strip_suffix(']')?;
    let (tx_ad, tur_ad) = ic.split_once("][")?;
    let tx = TX_SIZES.iter().find(|t| t.dav1d_name() == tx_ad)?;
    let tur = TX_TYPES
        .iter()
        .copied()
        .chain(core::iter::once(TxType::WhtWht))
        .find(|t| t.dav1d_name() == tur_ad)?;
    let f = match candidate.bit_depth {
        8 => itx::itx_fn::<8>(neon),
        10 => itx::itx_fn::<10>(neon),
        12 => itx::itx_fn::<12>(neon),
        _ => return None,
    };
    Some(IsaEntry::Itx(tx.index(), tur.index(), f))
}

/// `[DAV1D_PIXEL_LAYOUT_I4xx-1]` dizin çözücüsü (`0..3`).
#[cfg(target_arch = "aarch64")]
fn arm_duzen_indeksi(field: &str, onek: &str) -> Option<usize> {
    let ic = field
        .strip_prefix(onek)?
        .strip_prefix('[')?
        .strip_suffix(']')?;
    super::LAYOUT_INDEX.iter().position(|&l| l == ic)
}

/// `ipred` NEON yuva eşlemesi (`8 / 10 / 12` bpc, P54).
#[cfg(target_arch = "aarch64")]
fn ipred_neon_entry(candidate: &IsaCandidate, neon: NeonToken) -> Option<IsaEntry> {
    use crate::dsp::ipred::{CflPredMode, IntraPredMode};
    macro_rules! bpc {
        ($b:literal) => {{
            if candidate.field == "pal_pred" {
                return Some(IsaEntry::IpredPalPred(ipred::pal_pred_fn::<$b>(neon)));
            }
            if let Some(ad) = candidate
                .field
                .strip_prefix("intra_pred[")
                .and_then(|s| s.strip_suffix(']'))
            {
                let idx = super::INTRA_PRED_NAMES.iter().position(|&n| n == ad)?;
                let mode = IntraPredMode::from_index(idx)?;
                return Some(IsaEntry::IpredAngular(
                    mode.index(),
                    ipred::intra_pred_fn::<$b>(neon, mode),
                ));
            }
            if let Some(layout) = arm_duzen_indeksi(candidate.field, "cfl_ac") {
                return Some(IsaEntry::IpredCflAc(
                    layout,
                    ipred::cfl_ac_fn::<$b>(neon, layout)?,
                ));
            }
            if let Some(ad) = candidate
                .field
                .strip_prefix("cfl_pred[")
                .and_then(|s| s.strip_suffix(']'))
            {
                let idx = super::CFL_PRED_NAMES.iter().position(|&n| n == ad)?;
                let mode = *CflPredMode::ALL.get(idx)?;
                return Some(IsaEntry::IpredCflPred(
                    mode.index(),
                    ipred::cfl_pred_fn::<$b>(neon, mode),
                ));
            }
            None
        }};
    }
    match candidate.bit_depth {
        8 => bpc!(8),
        10 => bpc!(10),
        12 => bpc!(12),
        _ => None,
    }
}

/// `filmgrain` NEON yuva eşlemesi (`8 / 10 / 12` bpc, P54).
#[cfg(target_arch = "aarch64")]
fn fg_neon_entry(candidate: &IsaCandidate, neon: NeonToken) -> Option<IsaEntry> {
    macro_rules! bpc {
        ($b:literal) => {{
            if candidate.field == "generate_grain_y" {
                return Some(IsaEntry::FgGrainY(fg::generate_grain_y_fn::<$b>(neon)));
            }
            if candidate.field == "fgy_32x32xn" {
                return Some(IsaEntry::Fgy(fg::fgy_fn::<$b>(neon)));
            }
            if let Some(d) = arm_duzen_indeksi(candidate.field, "generate_grain_uv") {
                return Some(IsaEntry::FgGrainUv(
                    d,
                    fg::generate_grain_uv_fn::<$b>(neon, d)?,
                ));
            }
            if let Some(d) = arm_duzen_indeksi(candidate.field, "fguv_32x32xn") {
                return Some(IsaEntry::Fguv(d, fg::fguv_fn::<$b>(neon, d)?));
            }
            None
        }};
    }
    match candidate.bit_depth {
        8 => bpc!(8),
        10 => bpc!(10),
        12 => bpc!(12),
        _ => None,
    }
}

/// `loopfilter` NEON yuva eşlemesi (`8 / 10 / 12` bpc, P54).
#[cfg(target_arch = "aarch64")]
fn lf_neon_entry(candidate: &IsaCandidate, neon: NeonToken) -> Option<IsaEntry> {
    let (plane, dir) = match candidate.field {
        "loop_filter_sb[0][0]" => (0, 0),
        "loop_filter_sb[0][1]" => (0, 1),
        "loop_filter_sb[1][0]" => (1, 0),
        "loop_filter_sb[1][1]" => (1, 1),
        _ => return None,
    };
    let f = match candidate.bit_depth {
        8 => lf::loop_filter_sb_fn::<8>(neon, plane, dir)?,
        10 => lf::loop_filter_sb_fn::<10>(neon, plane, dir)?,
        12 => lf::loop_filter_sb_fn::<12>(neon, plane, dir)?,
        _ => return None,
    };
    Some(IsaEntry::LfSb(plane, dir, f))
}

/// `looprestoration` NEON yuva eşlemesi (`wiener` ve `sgr` @ `8/10/12`, P54 + 4.A.3).
#[cfg(target_arch = "aarch64")]
fn lr_neon_entry(candidate: &IsaCandidate, neon: NeonToken) -> Option<IsaEntry> {
    match candidate.field {
        "wiener[0]" | "wiener[1]" => {
            let tap = usize::from(candidate.field == "wiener[1]");
            let f = match candidate.bit_depth {
                8 => lr::wiener_fn::<8>(neon, tap)?,
                10 => lr::wiener_fn::<10>(neon, tap)?,
                12 => lr::wiener_fn::<12>(neon, tap)?,
                _ => return None,
            };
            Some(IsaEntry::LrWiener(tap, f))
        }
        "sgr[0]" | "sgr[1]" | "sgr[2]" => {
            let idx = match candidate.field {
                "sgr[0]" => 0,
                "sgr[1]" => 1,
                _ => 2,
            };
            let f = match candidate.bit_depth {
                8 => lr::sgr_fn::<8>(neon, idx)?,
                10 => lr::sgr_fn::<10>(neon, idx)?,
                12 => lr::sgr_fn::<12>(neon, idx)?,
                _ => return None,
            };
            Some(IsaEntry::LrSgr(idx, f))
        }
        _ => None,
    }
}
