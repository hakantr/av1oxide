//! WebAssembly 128-bit sabit genişlikli SIMD (`simd128`, `core::arch::wasm32`)
//! DSP çekirdekleri (`F6` / `P50` — `WP-10`, `DSP-07`; `v0.5.0` Adım `4.A.2`).
//!
//! ## Mimari ve Güvenlik Sözleşmesi (`GUV-08`, `GUV-10`, `DSP-07`)
//!
//! - `%100 #![forbid(unsafe_code)]`: `core::arch::wasm32` içindeki değer-tabanlı
//!   vektör intrinsikleri (`v128`, `i8x16_*`, `u8x16_*`, `i16x8_*`, `u16x8_*`,
//!   `i32x4_*`, `u32x4_*`, `i64x2_*`, `v128_*`) kararlı Rust'ta güvenli işlevlerdir.
//! - Sıfır yeni `unsafe` blok (`audit::UNSAFE_EXEMPTIONS` değişmez).
//! - Yalnız `#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]`
//!   koşulunda derlenir; skaler WASM artefaktına (`not(target_feature = "simd128")`)
//!   hiçbir `v128` / `simd128` komutu sızmaz (`wasm-objdump -d` ile doğrulanır).
//!
//! ## Kapsanan `DspDispatch` Yuvaları (`682 / 767` yuva)
//!
//! - **MC (`96` yuva, `8 / 10 / 12` bpc):** `mc[0..9]`, `mct[0..9]`, `avg`,
//!   `w_avg`, `mask`, `w_mask[0..2]`, `blend`, `blend_v`, `blend_h`, `warp8x8`,
//!   `warp8x8t`, `emu_edge` ([`mc`])
//! - **CDEF (`12` yuva, `8 / 10 / 12` bpc):** `dir`, `fb[0..2]`
//! - **PAL (`1` yuva, `0` bpc):** `pal/pal_idx_finish`
//! - **IPRED (`66` yuva, `8 / 10 / 12` bpc):** `intra_pred[0..13]`, `cfl_ac[0..2]`,
//!   `cfl_pred[0..3]`, `pal_pred` ([`ipred`])
//! - **ITX (`468` yuva, `8 / 10 / 12` bpc):** desteklenen tüm `156` `itxfm_add[tx][txtp]`
//!   hücresi (`WHT_WHT` ve DC-only hızlı yolu dahil) ([`itx`])
//! - **Loop Filter (`12` yuva, `8 / 10 / 12` bpc):** `loop_filter_sb[0..1][0..1]` ([`lf`])
//! - **Loop Restoration (`15` yuva, `8 / 10 / 12` bpc):** `wiener[0..1]`, `sgr[0..2]` ([`lr`])
//! - **Film Grain (`12` yuva, `8 / 10 / 12` bpc):** `fgy_32x32xn`, `fguv_32x32xn[0..2]` ([`fg`])

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
#![forbid(unsafe_code)]

mod fg;
mod ipred;
mod itx;
mod lf;
mod lr;
mod mc;

use core::arch::wasm32::{
    i8x16_shuffle, i16x8, i16x8_add, i16x8_extract_lane, i16x8_max, i16x8_min, i16x8_narrow_i32x4,
    i16x8_splat, i16x8_sub, i32x4, i32x4_abs, i32x4_add, i32x4_dot_i16x8, i32x4_extend_low_i16x8,
    i32x4_extract_lane, i32x4_lt, i32x4_max, i32x4_min, i32x4_mul, i32x4_neg, i32x4_shr,
    i32x4_splat, i32x4_sub, i64x2, i64x2_extract_lane, u8x16_narrow_i16x8, u8x16_splat,
    u16x8_extend_low_u8x16, u16x8_shl, u16x8_shr, u32x4_min, v128, v128_and, v128_bitselect,
    v128_or,
};

use crate::cpu::CpuFlags;
use crate::dsp::arm::itx_alanlari::ITX_NEON_ALANLARI;
use crate::dsp::cdef::{
    CDEF_DIRECTIONS, CdefBitDepth, CdefDirResult, CdefEdgeFlags, CdefFilterParams,
    DIR_CENTER_WEIGHT, DIV_TABLE, TMP_BUF_LEN, TMP_FILL, clamped_shift, px_at, px_put, tmp_index,
    ulog2,
};
use crate::dsp::ipred::{CflPredMode, IntraPredMode};
use crate::dsp::itx::{TX_SIZES, TX_TYPES, TxType};
use crate::dsp::{IsaCandidate, IsaEntry};

const WASM_SIMD128_KARAR: &str = "varsayılan (wasm32 simd128): %100 #![forbid(unsafe_code)] kararlı core::arch::wasm32 v128 çekirdeği; derleme zamanı +simd128 artefaktında etkin, skaler WASM artefaktına 0 sızıntı (rust_method: core::arch)";

const fn wasm_aday(family: &'static str, field: &'static str, bit_depth: u8) -> IsaCandidate {
    IsaCandidate {
        family,
        field,
        bit_depth,
        implementation: "simd128",
        requires: CpuFlags::WASM_SIMD128,
        default: true,
        decision: WASM_SIMD128_KARAR,
    }
}

macro_rules! mc_wasm_adaylari {
    ($bd:literal) => {
        [
            wasm_aday("mc", "mc[FILTER_2D_8TAP_REGULAR]", $bd),
            wasm_aday("mc", "mc[FILTER_2D_8TAP_REGULAR_SMOOTH]", $bd),
            wasm_aday("mc", "mc[FILTER_2D_8TAP_REGULAR_SHARP]", $bd),
            wasm_aday("mc", "mc[FILTER_2D_8TAP_SHARP_REGULAR]", $bd),
            wasm_aday("mc", "mc[FILTER_2D_8TAP_SHARP_SMOOTH]", $bd),
            wasm_aday("mc", "mc[FILTER_2D_8TAP_SHARP]", $bd),
            wasm_aday("mc", "mc[FILTER_2D_8TAP_SMOOTH_REGULAR]", $bd),
            wasm_aday("mc", "mc[FILTER_2D_8TAP_SMOOTH]", $bd),
            wasm_aday("mc", "mc[FILTER_2D_8TAP_SMOOTH_SHARP]", $bd),
            wasm_aday("mc", "mc[FILTER_2D_BILINEAR]", $bd),
            wasm_aday("mc", "mct[FILTER_2D_8TAP_REGULAR]", $bd),
            wasm_aday("mc", "mct[FILTER_2D_8TAP_REGULAR_SMOOTH]", $bd),
            wasm_aday("mc", "mct[FILTER_2D_8TAP_REGULAR_SHARP]", $bd),
            wasm_aday("mc", "mct[FILTER_2D_8TAP_SHARP_REGULAR]", $bd),
            wasm_aday("mc", "mct[FILTER_2D_8TAP_SHARP_SMOOTH]", $bd),
            wasm_aday("mc", "mct[FILTER_2D_8TAP_SHARP]", $bd),
            wasm_aday("mc", "mct[FILTER_2D_8TAP_SMOOTH_REGULAR]", $bd),
            wasm_aday("mc", "mct[FILTER_2D_8TAP_SMOOTH]", $bd),
            wasm_aday("mc", "mct[FILTER_2D_8TAP_SMOOTH_SHARP]", $bd),
            wasm_aday("mc", "mct[FILTER_2D_BILINEAR]", $bd),
            wasm_aday("mc", "avg", $bd),
            wasm_aday("mc", "w_avg", $bd),
            wasm_aday("mc", "mask", $bd),
            wasm_aday("mc", "w_mask[0]", $bd),
            wasm_aday("mc", "w_mask[1]", $bd),
            wasm_aday("mc", "w_mask[2]", $bd),
            wasm_aday("mc", "blend", $bd),
            wasm_aday("mc", "blend_v", $bd),
            wasm_aday("mc", "blend_h", $bd),
            wasm_aday("mc", "warp8x8", $bd),
            wasm_aday("mc", "warp8x8t", $bd),
            wasm_aday("mc", "emu_edge", $bd),
        ]
    };
}

macro_rules! ipred_wasm_adaylari {
    ($bd:literal) => {
        [
            wasm_aday("ipred", "intra_pred[DC_PRED]", $bd),
            wasm_aday("ipred", "intra_pred[VERT_PRED]", $bd),
            wasm_aday("ipred", "intra_pred[HOR_PRED]", $bd),
            wasm_aday("ipred", "intra_pred[LEFT_DC_PRED]", $bd),
            wasm_aday("ipred", "intra_pred[TOP_DC_PRED]", $bd),
            wasm_aday("ipred", "intra_pred[DC_128_PRED]", $bd),
            wasm_aday("ipred", "intra_pred[Z1_PRED]", $bd),
            wasm_aday("ipred", "intra_pred[Z2_PRED]", $bd),
            wasm_aday("ipred", "intra_pred[Z3_PRED]", $bd),
            wasm_aday("ipred", "intra_pred[SMOOTH_PRED]", $bd),
            wasm_aday("ipred", "intra_pred[SMOOTH_V_PRED]", $bd),
            wasm_aday("ipred", "intra_pred[SMOOTH_H_PRED]", $bd),
            wasm_aday("ipred", "intra_pred[PAETH_PRED]", $bd),
            wasm_aday("ipred", "intra_pred[FILTER_PRED]", $bd),
            wasm_aday("ipred", "cfl_ac[DAV1D_PIXEL_LAYOUT_I420-1]", $bd),
            wasm_aday("ipred", "cfl_ac[DAV1D_PIXEL_LAYOUT_I422-1]", $bd),
            wasm_aday("ipred", "cfl_ac[DAV1D_PIXEL_LAYOUT_I444-1]", $bd),
            wasm_aday("ipred", "cfl_pred[DC_PRED]", $bd),
            wasm_aday("ipred", "cfl_pred[LEFT_DC_PRED]", $bd),
            wasm_aday("ipred", "cfl_pred[TOP_DC_PRED]", $bd),
            wasm_aday("ipred", "cfl_pred[DC_128_PRED]", $bd),
            wasm_aday("ipred", "pal_pred", $bd),
        ]
    };
}

macro_rules! lf_wasm_adaylari {
    ($bd:literal) => {
        [
            wasm_aday("loopfilter", "loop_filter_sb[0][0]", $bd),
            wasm_aday("loopfilter", "loop_filter_sb[0][1]", $bd),
            wasm_aday("loopfilter", "loop_filter_sb[1][0]", $bd),
            wasm_aday("loopfilter", "loop_filter_sb[1][1]", $bd),
        ]
    };
}

macro_rules! lr_wasm_adaylari {
    ($bd:literal) => {
        [
            wasm_aday("looprestoration", "wiener[0]", $bd),
            wasm_aday("looprestoration", "wiener[1]", $bd),
            wasm_aday("looprestoration", "sgr[0]", $bd),
            wasm_aday("looprestoration", "sgr[1]", $bd),
            wasm_aday("looprestoration", "sgr[2]", $bd),
        ]
    };
}

macro_rules! fg_wasm_adaylari {
    ($bd:literal) => {
        [
            wasm_aday("filmgrain", "fgy_32x32xn", $bd),
            wasm_aday("filmgrain", "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I420-1]", $bd),
            wasm_aday("filmgrain", "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I422-1]", $bd),
            wasm_aday("filmgrain", "fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I444-1]", $bd),
        ]
    };
}

const WASM_CDEF_PAL: [IsaCandidate; 13] = [
    wasm_aday("cdef", "dir", 8),
    wasm_aday("cdef", "fb[0]", 8),
    wasm_aday("cdef", "fb[1]", 8),
    wasm_aday("cdef", "fb[2]", 8),
    wasm_aday("cdef", "dir", 10),
    wasm_aday("cdef", "fb[0]", 10),
    wasm_aday("cdef", "fb[1]", 10),
    wasm_aday("cdef", "fb[2]", 10),
    wasm_aday("cdef", "dir", 12),
    wasm_aday("cdef", "fb[0]", 12),
    wasm_aday("cdef", "fb[1]", 12),
    wasm_aday("cdef", "fb[2]", 12),
    wasm_aday("pal", "pal_idx_finish", 0),
];

const WASM_MC: [[IsaCandidate; 32]; 3] = [
    mc_wasm_adaylari!(8),
    mc_wasm_adaylari!(10),
    mc_wasm_adaylari!(12),
];

const WASM_IPRED: [[IsaCandidate; 22]; 3] = [
    ipred_wasm_adaylari!(8),
    ipred_wasm_adaylari!(10),
    ipred_wasm_adaylari!(12),
];

const WASM_LF: [[IsaCandidate; 4]; 3] = [
    lf_wasm_adaylari!(8),
    lf_wasm_adaylari!(10),
    lf_wasm_adaylari!(12),
];

const WASM_LR: [[IsaCandidate; 5]; 3] = [
    lr_wasm_adaylari!(8),
    lr_wasm_adaylari!(10),
    lr_wasm_adaylari!(12),
];

const WASM_FG: [[IsaCandidate; 4]; 3] = [
    fg_wasm_adaylari!(8),
    fg_wasm_adaylari!(10),
    fg_wasm_adaylari!(12),
];

const fn birlestir<const N: usize>() -> [IsaCandidate; N] {
    let mut out = [WASM_CDEF_PAL[0]; N];
    let mut i = 0usize;

    let mut j = 0usize;
    while j < WASM_CDEF_PAL.len() {
        out[i] = WASM_CDEF_PAL[j];
        i += 1;
        j += 1;
    }

    let mut b = 0usize;
    while b < 3 {
        let mut k = 0usize;
        while k < 32 {
            out[i] = WASM_MC[b][k];
            i += 1;
            k += 1;
        }
        b += 1;
    }

    let mut b = 0usize;
    while b < 3 {
        let mut k = 0usize;
        while k < 22 {
            out[i] = WASM_IPRED[b][k];
            i += 1;
            k += 1;
        }
        b += 1;
    }

    let bdler = [8u8, 10, 12];
    let mut b = 0usize;
    while b < 3 {
        let mut f = 0usize;
        while f < ITX_NEON_ALANLARI.len() {
            out[i] = wasm_aday("itx", ITX_NEON_ALANLARI[f], bdler[b]);
            i += 1;
            f += 1;
        }
        b += 1;
    }

    let mut b = 0usize;
    while b < 3 {
        let mut k = 0usize;
        while k < 4 {
            out[i] = WASM_LF[b][k];
            i += 1;
            k += 1;
        }
        b += 1;
    }

    let mut b = 0usize;
    while b < 3 {
        let mut k = 0usize;
        while k < 5 {
            out[i] = WASM_LR[b][k];
            i += 1;
            k += 1;
        }
        b += 1;
    }

    let mut b = 0usize;
    while b < 3 {
        let mut k = 0usize;
        while k < 4 {
            out[i] = WASM_FG[b][k];
            i += 1;
            k += 1;
        }
        b += 1;
    }

    assert!(i == N, "WASM_ISA_CANDIDATES uzunluğu uyuşmalı");
    out
}

/// Toplam `wasm32 +simd128` aday sayısı:
/// `13` (CDEF 12 + PAL 1) + `96` (MC) + `66` (IPRED) + `468` (ITX)
/// + `12` (LF) + `15` (LR) + `12` (FG) = `682` tekil `DspDispatch` yuvası.
const WASM_ADAY_SAYISI: usize = 13 + 3 * 32 + 3 * 22 + 3 * 156 + 3 * 4 + 3 * 5 + 3 * 4;
static WASM_ADAYLAR: [IsaCandidate; WASM_ADAY_SAYISI] = birlestir::<WASM_ADAY_SAYISI>();

/// `wasm32` (`target_feature = "simd128"`) `DspDispatch` ISA aday tablosu (`682` yuva).
pub const WASM_ISA_CANDIDATES: &[IsaCandidate] = &WASM_ADAYLAR;

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

fn duzen_indeksi(field: &str, onek: &str) -> Option<usize> {
    let ic = field
        .strip_prefix(onek)?
        .strip_prefix('[')?
        .strip_suffix(']')?;
    super::LAYOUT_INDEX.iter().position(|&l| l == ic)
}

fn mc_entry(candidate: &IsaCandidate) -> Option<IsaEntry> {
    macro_rules! bpc {
        ($b:literal) => {{
            Some(match candidate.field {
                "avg" => IsaEntry::Avg(mc::avg::<$b>),
                "w_avg" => IsaEntry::WAvg(mc::w_avg::<$b>),
                "mask" => IsaEntry::Mask(mc::mask::<$b>),
                "w_mask[0]" => IsaEntry::WMask(0, mc::w_mask::<$b, 0>),
                "w_mask[1]" => IsaEntry::WMask(1, mc::w_mask::<$b, 1>),
                "w_mask[2]" => IsaEntry::WMask(2, mc::w_mask::<$b, 2>),
                "blend" => IsaEntry::Blend(mc::blend::<$b>),
                "blend_v" => IsaEntry::BlendV(mc::blend_v::<$b>),
                "blend_h" => IsaEntry::BlendH(mc::blend_h::<$b>),
                "warp8x8" => IsaEntry::Warp(mc::warp8x8::<$b>),
                "warp8x8t" => IsaEntry::WarpT(mc::warp8x8t::<$b>),
                "emu_edge" => IsaEntry::EmuEdge(mc::emu_edge::<$b>),
                f => {
                    let (prep, i) = mc_alan(f)?;
                    if prep {
                        IsaEntry::Mct(i, mc::prep_fn::<$b>(i))
                    } else {
                        IsaEntry::Mc(i, mc::put_fn::<$b>(i))
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

fn ipred_entry(candidate: &IsaCandidate) -> Option<IsaEntry> {
    macro_rules! bpc {
        ($b:literal) => {{
            if candidate.field == "pal_pred" {
                return Some(IsaEntry::IpredPalPred(ipred::pal_pred::<$b>));
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
                    ipred::intra_pred_fn::<$b>(mode),
                ));
            }
            if let Some(layout) = duzen_indeksi(candidate.field, "cfl_ac") {
                return Some(IsaEntry::IpredCflAc(
                    layout,
                    ipred::cfl_ac_fn::<$b>(layout)?,
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
                    ipred::cfl_pred_fn::<$b>(mode),
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

fn itx_entry(candidate: &IsaCandidate) -> Option<IsaEntry> {
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
        8 => itx::itx_add::<8>,
        10 => itx::itx_add::<10>,
        12 => itx::itx_add::<12>,
        _ => return None,
    };
    Some(IsaEntry::Itx(tx.index(), tur.index(), f))
}

fn lf_entry(candidate: &IsaCandidate) -> Option<IsaEntry> {
    let (plane, dir) = match candidate.field {
        "loop_filter_sb[0][0]" => (0, 0),
        "loop_filter_sb[0][1]" => (0, 1),
        "loop_filter_sb[1][0]" => (1, 0),
        "loop_filter_sb[1][1]" => (1, 1),
        _ => return None,
    };
    let f = match candidate.bit_depth {
        8 => lf::loop_filter_sb_fn::<8>(plane, dir)?,
        10 => lf::loop_filter_sb_fn::<10>(plane, dir)?,
        12 => lf::loop_filter_sb_fn::<12>(plane, dir)?,
        _ => return None,
    };
    Some(IsaEntry::LfSb(plane, dir, f))
}

fn lr_entry(candidate: &IsaCandidate) -> Option<IsaEntry> {
    match candidate.field {
        "wiener[0]" | "wiener[1]" => {
            let tap = usize::from(candidate.field == "wiener[1]");
            let f = match candidate.bit_depth {
                8 => lr::wiener_fn::<8>(tap)?,
                10 => lr::wiener_fn::<10>(tap)?,
                12 => lr::wiener_fn::<12>(tap)?,
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
                8 => lr::sgr_fn::<8>(idx)?,
                10 => lr::sgr_fn::<10>(idx)?,
                12 => lr::sgr_fn::<12>(idx)?,
                _ => return None,
            };
            Some(IsaEntry::LrSgr(idx, f))
        }
        _ => None,
    }
}

fn fg_entry(candidate: &IsaCandidate) -> Option<IsaEntry> {
    macro_rules! bpc {
        ($b:literal) => {{
            if candidate.field == "fgy_32x32xn" {
                return Some(IsaEntry::Fgy(fg::fgy_32x32xn::<$b>));
            }
            if let Some(d) = duzen_indeksi(candidate.field, "fguv_32x32xn") {
                return Some(IsaEntry::Fguv(d, fg::fguv_fn::<$b>(d)?));
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

/// `DspDispatch::candidate_entry` için `wasm32` `simd128` yuva eşlemesi.
#[must_use]
pub(super) fn candidate_entry(candidate: &IsaCandidate) -> Option<IsaEntry> {
    if candidate.implementation != "simd128" {
        return None;
    }
    match candidate.family {
        "mc" => return mc_entry(candidate),
        "ipred" => return ipred_entry(candidate),
        "itx" => return itx_entry(candidate),
        "loopfilter" => return lf_entry(candidate),
        "looprestoration" => return lr_entry(candidate),
        "filmgrain" => return fg_entry(candidate),
        _ => {}
    }
    match (candidate.family, candidate.field, candidate.bit_depth) {
        ("cdef", "dir", 8) => Some(IsaEntry::CdefDir(cdef_find_dir::<8>)),
        ("cdef", "dir", 10) => Some(IsaEntry::CdefDir(cdef_find_dir::<10>)),
        ("cdef", "dir", 12) => Some(IsaEntry::CdefDir(cdef_find_dir::<12>)),
        ("cdef", "fb[0]", 8) => Some(IsaEntry::CdefFb(0, cdef_filter::<8>)),
        ("cdef", "fb[1]", 8) => Some(IsaEntry::CdefFb(1, cdef_filter::<8>)),
        ("cdef", "fb[2]", 8) => Some(IsaEntry::CdefFb(2, cdef_filter::<8>)),
        ("cdef", "fb[0]", 10) => Some(IsaEntry::CdefFb(0, cdef_filter::<10>)),
        ("cdef", "fb[1]", 10) => Some(IsaEntry::CdefFb(1, cdef_filter::<10>)),
        ("cdef", "fb[2]", 10) => Some(IsaEntry::CdefFb(2, cdef_filter::<10>)),
        ("cdef", "fb[0]", 12) => Some(IsaEntry::CdefFb(0, cdef_filter::<12>)),
        ("cdef", "fb[1]", 12) => Some(IsaEntry::CdefFb(1, cdef_filter::<12>)),
        ("cdef", "fb[2]", 12) => Some(IsaEntry::CdefFb(2, cdef_filter::<12>)),
        ("pal", "pal_idx_finish", 0) => Some(IsaEntry::PalIdxFinish(pal_idx_finish)),
        _ => None,
    }
}

// ============================================================================
// Güvenli `v128` yükleme / saklama yardımcıları (`#![forbid(unsafe_code)]`)
// ============================================================================

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn load_v128_u8(src: &[u8], off: usize) -> v128 {
    let b: &[u8; 16] = src[off..off + 16].try_into().unwrap();
    let lo = i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
    let hi = i64::from_le_bytes([b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]]);
    i64x2(lo, hi)
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn load_low64_u8(src: &[u8], off: usize) -> v128 {
    let b0: [u8; 8] = src[off..off + 8].try_into().unwrap();
    i64x2(i64::from_le_bytes(b0), 0)
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn load_low32_u8(src: &[u8], off: usize) -> v128 {
    let b0: [u8; 4] = src[off..off + 4].try_into().unwrap();
    i64x2(i64::from(u32::from_le_bytes(b0)), 0)
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn load_i16x8(src: &[i16], off: usize) -> v128 {
    let s: &[i16; 8] = src[off..off + 8].try_into().unwrap();
    i16x8(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7])
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn load_i16x4(src: &[i16], off: usize) -> v128 {
    let s: &[i16; 4] = src[off..off + 4].try_into().unwrap();
    i16x8(s[0], s[1], s[2], s[3], 0, 0, 0, 0)
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn load_u16x8(src: &[u16], off: usize) -> v128 {
    let s: &[u16; 8] = src[off..off + 8].try_into().unwrap();
    i16x8(
        s[0] as i16,
        s[1] as i16,
        s[2] as i16,
        s[3] as i16,
        s[4] as i16,
        s[5] as i16,
        s[6] as i16,
        s[7] as i16,
    )
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn load_i32x4(src: &[i32], off: usize) -> v128 {
    let s: &[i32; 4] = src[off..off + 4].try_into().unwrap();
    i32x4(s[0], s[1], s[2], s[3])
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn store_v128_u8(dst: &mut [u8], off: usize, v: v128) {
    let lo = i64x2_extract_lane::<0>(v).to_le_bytes();
    let hi = i64x2_extract_lane::<1>(v).to_le_bytes();
    let out: &mut [u8; 16] = (&mut dst[off..off + 16]).try_into().unwrap();
    out[0..8].copy_from_slice(&lo);
    out[8..16].copy_from_slice(&hi);
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn store_low64_u8(dst: &mut [u8], off: usize, v: v128) {
    let out: &mut [u8; 8] = (&mut dst[off..off + 8]).try_into().unwrap();
    *out = i64x2_extract_lane::<0>(v).to_le_bytes();
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn store_low32_u8(dst: &mut [u8], off: usize, v: v128) {
    let out: &mut [u8; 4] = (&mut dst[off..off + 4]).try_into().unwrap();
    *out = i32x4_extract_lane::<0>(v).to_le_bytes();
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn store_i16x8(dst: &mut [i16], off: usize, v: v128) {
    let out: &mut [i16; 8] = (&mut dst[off..off + 8]).try_into().unwrap();
    out[0] = i16x8_extract_lane::<0>(v);
    out[1] = i16x8_extract_lane::<1>(v);
    out[2] = i16x8_extract_lane::<2>(v);
    out[3] = i16x8_extract_lane::<3>(v);
    out[4] = i16x8_extract_lane::<4>(v);
    out[5] = i16x8_extract_lane::<5>(v);
    out[6] = i16x8_extract_lane::<6>(v);
    out[7] = i16x8_extract_lane::<7>(v);
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn store_u16x8(dst: &mut [u16], off: usize, v: v128) {
    let out: &mut [u16; 8] = (&mut dst[off..off + 8]).try_into().unwrap();
    out[0] = i16x8_extract_lane::<0>(v) as u16;
    out[1] = i16x8_extract_lane::<1>(v) as u16;
    out[2] = i16x8_extract_lane::<2>(v) as u16;
    out[3] = i16x8_extract_lane::<3>(v) as u16;
    out[4] = i16x8_extract_lane::<4>(v) as u16;
    out[5] = i16x8_extract_lane::<5>(v) as u16;
    out[6] = i16x8_extract_lane::<6>(v) as u16;
    out[7] = i16x8_extract_lane::<7>(v) as u16;
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn store_i32x4(dst: &mut [i32], off: usize, v: v128) {
    let out: &mut [i32; 4] = (&mut dst[off..off + 4]).try_into().unwrap();
    out[0] = i32x4_extract_lane::<0>(v);
    out[1] = i32x4_extract_lane::<1>(v);
    out[2] = i32x4_extract_lane::<2>(v);
    out[3] = i32x4_extract_lane::<3>(v);
}

#[inline(always)]
pub(super) const fn bpp_of(bits: u32) -> usize {
    if bits == 8 { 1 } else { 2 }
}

#[inline(always)]
pub(super) const fn intermediate_bits_of(bits: u32) -> u32 {
    if bits == 8 { 4 } else { 14 - bits }
}

#[inline(always)]
pub(super) const fn prep_bias_of(bits: u32) -> i32 {
    if bits == 8 { 0 } else { 8192 }
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn load_pixels_8<const BPC: u32>(buf: &[u8], off: usize) -> v128 {
    if BPC == 8 {
        u16x8_extend_low_u8x16(load_low64_u8(buf, off))
    } else {
        load_v128_u8(buf, off)
    }
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn load_pixels_4<const BPC: u32>(buf: &[u8], off: usize) -> v128 {
    if BPC == 8 {
        u16x8_extend_low_u8x16(load_low32_u8(buf, off))
    } else {
        load_low64_u8(buf, off)
    }
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn store_pixels_8_i16<const BPC: u32>(dst: &mut [u8], off: usize, p16: v128) {
    if BPC == 8 {
        store_low64_u8(dst, off, u8x16_narrow_i16x8(p16, p16));
    } else {
        store_v128_u8(dst, off, p16);
    }
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn store_pixels_4_i16<const BPC: u32>(dst: &mut [u8], off: usize, p16: v128) {
    if BPC == 8 {
        store_low32_u8(dst, off, u8x16_narrow_i16x8(p16, p16));
    } else {
        store_low64_u8(dst, off, p16);
    }
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn pack_and_store_8<const BPC: u32>(
    dst: &mut [u8],
    off: usize,
    lo: v128,
    hi: v128,
    v_max: v128,
) {
    let p16 = i16x8_min(i16x8_max(i16x8_narrow_i32x4(lo, hi), i16x8_splat(0)), v_max);
    store_pixels_8_i16::<BPC>(dst, off, p16);
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn pack_and_store_4<const BPC: u32>(dst: &mut [u8], off: usize, lo: v128, v_max: v128) {
    let p16 = i16x8_min(i16x8_max(i16x8_narrow_i32x4(lo, lo), i16x8_splat(0)), v_max);
    store_pixels_4_i16::<BPC>(dst, off, p16);
}

#[inline(always)]
pub(super) fn load_scalar<const BPC: u32>(buf: &[u8], off: usize) -> i32 {
    if BPC == 8 {
        i32::from(buf[off])
    } else {
        i32::from(u16::from_le_bytes([buf[off], buf[off + 1]]))
    }
}

#[inline(always)]
pub(super) fn store_scalar<const BPC: u32>(buf: &mut [u8], off: usize, value: i32) {
    if BPC == 8 {
        buf[off] = value as u8;
    } else {
        buf[off..off + 2].copy_from_slice(&(value as u16).to_le_bytes());
    }
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn transpose_4x4_i32(r0: v128, r1: v128, r2: v128, r3: v128) -> [v128; 4] {
    let t0 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(r0, r1);
    let t1 = i8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(r0, r1);
    let t2 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(r2, r3);
    let t3 = i8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(r2, r3);
    let c0 = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(t0, t2);
    let c1 = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(t0, t2);
    let c2 = i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(t1, t3);
    let c3 = i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(t1, t3);
    [c0, c1, c2, c3]
}

#[inline]
#[target_feature(enable = "simd128")]
pub(super) fn hsum_i32x4(v: v128) -> i32 {
    i32x4_extract_lane::<0>(v)
        .wrapping_add(i32x4_extract_lane::<1>(v))
        .wrapping_add(i32x4_extract_lane::<2>(v))
        .wrapping_add(i32x4_extract_lane::<3>(v))
}

// ============================================================================
// CDEF `dir` ve `fb[0..2]` (12 yuva)
// ============================================================================

#[target_feature(enable = "simd128")]
fn cdef_find_dir_simd128<const BPC: u32>(img: &[u8], stride: usize) -> CdefDirResult {
    let bd_min_8 = BPC - 8;
    let v_128 = i16x8_splat(128);

    let mut partial_sum_hv0 = [0i16; 8];
    let mut v_hv1 = i16x8_splat(0);
    let mut partial_sum_diag = [[0i32; 15]; 2];
    let mut partial_sum_alt = [[0i32; 11]; 4];

    for y in 0..8usize {
        let raw = load_pixels_8::<BPC>(img, y * stride);
        let row_vec = i16x8_sub(u16x8_shr(raw, bd_min_8), v_128);
        v_hv1 = i16x8_add(v_hv1, row_vec);

        let r = [
            i32::from(i16x8_extract_lane::<0>(row_vec)),
            i32::from(i16x8_extract_lane::<1>(row_vec)),
            i32::from(i16x8_extract_lane::<2>(row_vec)),
            i32::from(i16x8_extract_lane::<3>(row_vec)),
            i32::from(i16x8_extract_lane::<4>(row_vec)),
            i32::from(i16x8_extract_lane::<5>(row_vec)),
            i32::from(i16x8_extract_lane::<6>(row_vec)),
            i32::from(i16x8_extract_lane::<7>(row_vec)),
        ];
        let mut row_sum = 0i32;
        for x in 0..8usize {
            let px = r[x];
            row_sum += px;
            partial_sum_diag[0][y + x] += px;
            partial_sum_alt[0][y + (x >> 1)] += px;
            partial_sum_alt[1][3 + y - (x >> 1)] += px;
            partial_sum_diag[1][7 + y - x] += px;
            partial_sum_alt[2][3 - (y >> 1) + x] += px;
            partial_sum_alt[3][(y >> 1) + x] += px;
        }
        partial_sum_hv0[y] = row_sum as i16;
    }

    let v_hv0 = load_i16x8(&partial_sum_hv0, 0);
    let mut cost = [0u32; 8];
    cost[2] = (hsum_i32x4(i32x4_dot_i16x8(v_hv0, v_hv0)) as u32).wrapping_mul(DIR_CENTER_WEIGHT);
    cost[6] = (hsum_i32x4(i32x4_dot_i16x8(v_hv1, v_hv1)) as u32).wrapping_mul(DIR_CENTER_WEIGHT);

    for n in 0..7usize {
        let d = DIV_TABLE[n];
        let a0 = partial_sum_diag[0][n];
        let b0 = partial_sum_diag[0][14 - n];
        cost[0] = cost[0].wrapping_add(
            (a0.wrapping_mul(a0).wrapping_add(b0.wrapping_mul(b0)) as u32).wrapping_mul(d),
        );
        let a1 = partial_sum_diag[1][n];
        let b1 = partial_sum_diag[1][14 - n];
        cost[4] = cost[4].wrapping_add(
            (a1.wrapping_mul(a1).wrapping_add(b1.wrapping_mul(b1)) as u32).wrapping_mul(d),
        );
    }
    let c0 = partial_sum_diag[0][7];
    let c1 = partial_sum_diag[1][7];
    cost[0] = cost[0].wrapping_add((c0.wrapping_mul(c0) as u32).wrapping_mul(DIR_CENTER_WEIGHT));
    cost[4] = cost[4].wrapping_add((c1.wrapping_mul(c1) as u32).wrapping_mul(DIR_CENTER_WEIGHT));

    for n in 0..4usize {
        let mut acc = 0u32;
        for m in 0..5usize {
            let v = partial_sum_alt[n][3 + m];
            acc = acc.wrapping_add(v.wrapping_mul(v) as u32);
        }
        acc = acc.wrapping_mul(DIR_CENTER_WEIGHT);
        for m in 0..3usize {
            let d = DIV_TABLE[2 * m + 1];
            let a = partial_sum_alt[n][m];
            let b = partial_sum_alt[n][10 - m];
            acc = acc.wrapping_add(
                (a.wrapping_mul(a).wrapping_add(b.wrapping_mul(b)) as u32).wrapping_mul(d),
            );
        }
        cost[n * 2 + 1] = acc;
    }

    let mut best_dir = 0usize;
    let mut best_cost = cost[0];
    for n in 1..8usize {
        if cost[n] > best_cost {
            best_cost = cost[n];
            best_dir = n;
        }
    }

    CdefDirResult {
        dir: best_dir as u8,
        var: best_cost.wrapping_sub(cost[best_dir ^ 4]) >> 10,
    }
}

/// `cdef/dir` WASM `simd128` yuva girişi.
pub fn cdef_find_dir<const BPC: u32>(
    img: &[u8],
    stride: usize,
    bit_depth: CdefBitDepth,
) -> CdefDirResult {
    debug_assert_eq!(bit_depth.bits(), BPC);
    cdef_find_dir_simd128::<BPC>(img, stride)
}

#[inline]
#[target_feature(enable = "simd128")]
fn constrain_i32x4(diff: v128, v_thresh: v128, shift: u32) -> v128 {
    let adiff = i32x4_abs(diff);
    let shifted = i32x4_shr(adiff, shift);
    let limit = i32x4_max(i32x4_sub(v_thresh, shifted), i32x4_splat(0));
    let mag = i32x4_min(adiff, limit);
    let neg_mask = i32x4_lt(diff, i32x4_splat(0));
    v128_bitselect(i32x4_neg(mag), mag, neg_mask)
}

#[inline]
#[target_feature(enable = "simd128")]
fn round_shift_i32x4(sum: v128) -> v128 {
    let neg = i32x4_lt(sum, i32x4_splat(0));
    i32x4_shr(i32x4_add(i32x4_add(sum, neg), i32x4_splat(8)), 4)
}

#[inline]
#[target_feature(enable = "simd128")]
fn load_tmp_i32x4(tmp: &[i16; TMP_BUF_LEN], base_idx: usize) -> v128 {
    i32x4_extend_low_i16x8(load_i16x4(tmp, base_idx))
}

#[target_feature(enable = "simd128")]
fn cdef_filter_simd128<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    let bpp = bpp_of(BPC);
    let w = params.size.w();
    let h = params.size.h();

    let has_left = params.edges.contains(CdefEdgeFlags::HAVE_LEFT);
    let has_right = params.edges.contains(CdefEdgeFlags::HAVE_RIGHT);
    let has_top = params.edges.contains(CdefEdgeFlags::HAVE_TOP);
    let has_bottom = params.edges.contains(CdefEdgeFlags::HAVE_BOTTOM);

    let x_start: i32 = if has_left { -2 } else { 0 };
    let x_end: i32 = if has_right { w as i32 + 2 } else { w as i32 };

    let mut tmp = [TMP_FILL; TMP_BUF_LEN];
    let top_rows = if has_top { 2usize } else { 0 };
    let bottom_rows = if has_bottom { 2usize } else { 0 };

    for row in 0..top_rows {
        let y = -2 + row as i32;
        for x in x_start..x_end {
            let v = px_at::<BPC>(top, row * stride + (x + 2) as usize * bpp);
            tmp[tmp_index(x, y)] = v as i16;
        }
    }
    if has_left {
        for y in 0..h {
            for x in -2i32..0 {
                let v = px_at::<BPC>(left, y * 2 * bpp + (x + 2) as usize * bpp);
                tmp[tmp_index(x, y as i32)] = v as i16;
            }
        }
    }
    for y in 0..h {
        for x in 0..x_end {
            let v = px_at::<BPC>(dst, y * stride + x as usize * bpp);
            tmp[tmp_index(x, y as i32)] = v as i16;
        }
    }
    for row in 0..bottom_rows {
        let y = h as i32 + row as i32;
        for x in x_start..x_end {
            let v = px_at::<BPC>(bottom, row * stride + (x + 2) as usize * bpp);
            tmp[tmp_index(x, y)] = v as i16;
        }
    }

    let dir = usize::from(params.dir & 7);
    let pri_strength = i32::from(params.pri_strength);
    let sec_strength = i32::from(params.sec_strength);
    if pri_strength == 0 && sec_strength == 0 {
        return;
    }

    let v_pri_thresh = i32x4_splat(pri_strength);
    let v_sec_thresh = i32x4_splat(sec_strength);
    let bd_min_8 = BPC - 8;
    let pri_tap0 = 4 - (((params.pri_strength as u32) >> bd_min_8) & 1) as i32;
    let pri_tap1 = (pri_tap0 & 3) | 2;
    let pri_taps = [i32x4_splat(pri_tap0), i32x4_splat(pri_tap1)];
    let sec_taps = [i32x4_splat(2), i32x4_splat(1)];
    let pri_shift = if pri_strength != 0 {
        clamped_shift(params.damping as i32 - ulog2(params.pri_strength as u32) as i32)
    } else {
        0
    };
    let sec_shift = if sec_strength != 0 {
        clamped_shift(params.damping as i32 - ulog2(params.sec_strength as u32) as i32)
    } else {
        0
    };

    for y in 0..h {
        let mut x = 0usize;
        while x + 4 <= w {
            let base = tmp_index(x as i32, y as i32);
            let px = load_tmp_i32x4(&tmp, base);
            let mut sum = i32x4_splat(0);
            let mut min_v = px;
            let mut max_v = px;

            if pri_strength != 0 {
                for k in 0..2usize {
                    let off1 = isize::from(CDEF_DIRECTIONS[dir + 2][k]);
                    let p0 = load_tmp_i32x4(&tmp, (base as isize + off1) as usize);
                    let p1 = load_tmp_i32x4(&tmp, (base as isize - off1) as usize);
                    let c0 = constrain_i32x4(i32x4_sub(p0, px), v_pri_thresh, pri_shift);
                    let c1 = constrain_i32x4(i32x4_sub(p1, px), v_pri_thresh, pri_shift);
                    sum = i32x4_add(sum, i32x4_mul(pri_taps[k], i32x4_add(c0, c1)));
                    if sec_strength != 0 {
                        min_v = u32x4_min(min_v, p0);
                        max_v = i32x4_max(max_v, p0);
                        min_v = u32x4_min(min_v, p1);
                        max_v = i32x4_max(max_v, p1);
                    }
                }
            }

            if sec_strength != 0 {
                for k in 0..2usize {
                    let off2 = isize::from(CDEF_DIRECTIONS[dir + 4][k]);
                    let off3 = isize::from(CDEF_DIRECTIONS[dir][k]);
                    let s0 = load_tmp_i32x4(&tmp, (base as isize + off2) as usize);
                    let s1 = load_tmp_i32x4(&tmp, (base as isize - off2) as usize);
                    let s2 = load_tmp_i32x4(&tmp, (base as isize + off3) as usize);
                    let s3 = load_tmp_i32x4(&tmp, (base as isize - off3) as usize);
                    let cs0 = constrain_i32x4(i32x4_sub(s0, px), v_sec_thresh, sec_shift);
                    let cs1 = constrain_i32x4(i32x4_sub(s1, px), v_sec_thresh, sec_shift);
                    let cs2 = constrain_i32x4(i32x4_sub(s2, px), v_sec_thresh, sec_shift);
                    let cs3 = constrain_i32x4(i32x4_sub(s3, px), v_sec_thresh, sec_shift);
                    let cs_tot = i32x4_add(i32x4_add(cs0, cs1), i32x4_add(cs2, cs3));
                    sum = i32x4_add(sum, i32x4_mul(sec_taps[k], cs_tot));
                    if pri_strength != 0 {
                        min_v = u32x4_min(u32x4_min(min_v, s0), u32x4_min(s1, u32x4_min(s2, s3)));
                        max_v = i32x4_max(i32x4_max(max_v, s0), i32x4_max(s1, i32x4_max(s2, s3)));
                    }
                }
            }

            let mut out = i32x4_add(px, round_shift_i32x4(sum));
            if pri_strength != 0 && sec_strength != 0 {
                out = i32x4_min(i32x4_max(out, min_v), max_v);
            }
            let drow = y * stride;
            px_put::<BPC>(dst, drow + x * bpp, i32x4_extract_lane::<0>(out) as u16);
            px_put::<BPC>(
                dst,
                drow + (x + 1) * bpp,
                i32x4_extract_lane::<1>(out) as u16,
            );
            px_put::<BPC>(
                dst,
                drow + (x + 2) * bpp,
                i32x4_extract_lane::<2>(out) as u16,
            );
            px_put::<BPC>(
                dst,
                drow + (x + 3) * bpp,
                i32x4_extract_lane::<3>(out) as u16,
            );
            x += 4;
        }
    }
}

/// `cdef/fb[0..2]` WASM `simd128` yuva girişi.
pub fn cdef_filter<const BPC: u32>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    debug_assert_eq!(params.bit_depth.bits(), BPC);
    cdef_filter_simd128::<BPC>(dst, stride, left, top, bottom, params);
}

// ============================================================================
// PAL `pal_idx_finish` (1 yuva)
// ============================================================================

#[inline]
#[target_feature(enable = "simd128")]
fn pack_pal_16(a: v128) -> v128 {
    let even = i8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14>(a, a);
    let odd = i8x16_shuffle::<1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15>(a, a);
    let odd_shl4 = v128_and(u16x8_shl(odd, 4), u8x16_splat(0xF0));
    v128_or(even, odd_shl4)
}

#[target_feature(enable = "simd128")]
fn pal_idx_finish_simd128(dst: &mut [u8], src: &[u8], bw: usize, bh: usize, w: usize, h: usize) {
    assert!(
        (4..=64).contains(&bw) && bw.is_power_of_two(),
        "pal_idx_finish bw 4..=64 ve 2'nin kuvveti olmalı: {bw}"
    );
    assert!(
        (4..=64).contains(&bh) && bh.is_power_of_two(),
        "pal_idx_finish bh 4..=64 ve 2'nin kuvveti olmalı: {bh}"
    );
    assert!(
        (4..=w).contains(&w) && w <= bw && w.is_multiple_of(4),
        "pal_idx_finish w 4..=bw ve 4'ün katı olmalı: {w}"
    );
    assert!(
        (4..=h).contains(&h) && h <= bh && h.is_multiple_of(4),
        "pal_idx_finish h 4..=bh ve 4'ün katı olmalı: {h}"
    );
    assert!(
        src.len() >= bw * bh,
        "pal_idx_finish src dilimi kısa: {} < {}",
        src.len(),
        bw * bh
    );
    let dst_width = w / 2;
    let dst_bw = bw / 2;
    assert!(
        dst.len() >= dst_bw * bh,
        "pal_idx_finish dst dilimi kısa: {} < {}",
        dst.len(),
        dst_bw * bh
    );

    for y in 0..h {
        let src_row = y * bw;
        let dst_row = y * dst_bw;
        let mut sx = 0usize;
        while sx + 16 <= w {
            let a = load_v128_u8(src, src_row + sx);
            let packed = pack_pal_16(a);
            store_low64_u8(dst, dst_row + sx / 2, packed);
            sx += 16;
        }
        while sx + 8 <= w {
            let a = load_low64_u8(src, src_row + sx);
            let packed = pack_pal_16(a);
            store_low32_u8(dst, dst_row + sx / 2, packed);
            sx += 8;
        }
        while sx < w {
            dst[dst_row + sx / 2] = src[src_row + sx] | (src[src_row + sx + 1] << 4);
            sx += 2;
        }
        if dst_width < dst_bw {
            let fill = src[src_row + w - 1].wrapping_mul(0x11);
            dst[dst_row + dst_width..dst_row + dst_bw].fill(fill);
        }
    }

    if h < bh {
        let last_row = (h - 1) * dst_bw;
        for y in h..bh {
            let dst_row = y * dst_bw;
            for x in 0..dst_bw {
                dst[dst_row + x] = dst[last_row + x];
            }
        }
    }
}

/// `pal/pal_idx_finish` WASM `simd128` yuva girişi.
pub fn pal_idx_finish(dst: &mut [u8], src: &[u8], bw: usize, bh: usize, w: usize, h: usize) {
    pal_idx_finish_simd128(dst, src, bw, bh, w, h);
}
