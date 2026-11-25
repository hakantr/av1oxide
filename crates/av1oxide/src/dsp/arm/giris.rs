//! `aarch64` NEON / DOTPROD yuva giriş köprüleri ve ek sabit boyutlu bellek yardımcıları
//! (`F5b`–`F5d`, `P52`–).
//!
//! [`super::mem`] (P51) ile aynı politika: `dsp/arm/` altındaki `unsafe` yalnız
//! **sınır dosyalarında** durur (`audit::UNSAFE_EXEMPTIONS`). Bu dosya iki tür
//! `unsafe` bloğu taşır:
//!
//! 1. Sabit boyutlu dizi referansı (`&mut [T; N]`) üzerinde `vst1*` saklama
//!    yardımcıları (sınır dizinin türüyle kanıtlıdır).
//! 2. Yuva giriş köprüleri: her köprü **güvenli** kodla O(1) sözleşmeyi denetler
//!    (sözleşme dışı girdi skaler gövdeye devreder ve sayılır), sonra `%100
//!    #![forbid(unsafe_code)]` `#[target_feature]` gövdesini çağırır. Köprülerin
//!    adresi yalnız [`NeonToken`](crate::cpu::NeonToken) /
//!    [`DotprodToken`](crate::cpu::DotprodToken) isteyen `*_fn` işlevleriyle
//!    (`super::mc`) dışarı verilir; belirteç, ilgili ISA'nın çalışma zamanı
//!    tespitinin kanıtıdır.
//!
//! P51 köprüleri (`super::mem`) satır numaraları sabit kalsın diye yerinde bırakıldı;
//! yeni köprüler bu dosyanın **sonuna** eklenir.

#![allow(unsafe_code)]
#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]

use core::arch::aarch64::{int16x8_t, int32x4_t, vld1q_s32, vst1q_s16, vst1q_s32};

use super::mc as k;
use crate::bitstream::frame_header::FilmGrainData;
use crate::cpu::NeonToken;
use crate::dsp::fg::{
    FgBitDepth, FgDsp, GrainLut, fg_skaler_devri_say, fguv_sozlesmesi, fguv16_sozlesmesi,
    fgy_sozlesmesi, fgy16_sozlesmesi,
};
use crate::dsp::ipred::{
    CflPredMode, Edge, IntraPredMode, angular_skaler_devret, angular_sozlesmesi,
    cfl_ac_skaler_devret, cfl_ac_sozlesmesi, cfl_pred_skaler_devret, cfl_pred_sozlesmesi,
    pal_pred_skaler_devret, pal_pred_sozlesmesi,
};
use crate::dsp::itx::{TxSize, TxType, itx_skaler_devret, itx8_sozlesmesi, itx16_sozlesmesi};
use crate::dsp::lf::{LimLut, lpf_sb_skaler_devret, lpf_sb_sozlesmesi};
use crate::dsp::lr::{
    LrBitDepth, LrEdgeFlags, LrParams, sgr_skaler_devret, sgr_sozlesmesi, wiener_skaler_devret,
    wiener_sozlesmesi,
};
use crate::dsp::mc::{McDsp, skaler_devri_say};
use crate::entropy::msac::MsacContext;
use crate::frame::BitDepth;

/// 8 adet `i16` (`&mut [i16; 8]`, 16 bayt) saklar (`vst1q_s16`).
#[target_feature(enable = "neon")]
#[inline]
pub fn store_i16x8(dst: &mut [i16; 8], v: int16x8_t) {
    // SAFETY: (1) ISA: `vst1q_s16` ARM64 NEON komutudur; işlev `#[target_feature(enable = "neon")]` taşır. (2) sınır: `dst` sabit boyutlu `&mut [i16; 8]` (16 bayt) referansıdır ve komut tam 16 bayt yazar. (3) hizalama/takma ad: `vst1q_s16` `&mut [i16; 8]` üzerinde geçerlidir; `&mut` tek erişim yoludur.
    unsafe { vst1q_s16(dst.as_mut_ptr(), v) }
}

// ============================================================================
// F5b / P52 — MC giriş köprüleri
// ============================================================================

/// `mc[FI]` NEON girişi (`BPC` = 8 / 10 / 12).
pub(super) fn mc_put<const BPC: u32, const FI: usize>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bd: BitDepth,
    bdmax: u16,
) {
    if !k::put_kabul::<BPC>(dst.len(), ds, src.len(), ss, w, h, mx, my, bd, bdmax) {
        return k::put_skaler::<FI>(dst, ds, src, ss, w, h, mx, my, bd, bdmax);
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`mc::put_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `put_kabul` kaynak bölgesini + `MC_KUYRUK`'u ve hedefte `h` satır × `w` pikseli O(1) denetledi; gövde erişimleri dilim/dizi denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`; ödünç kuralları çakışmayı dışlar.
    unsafe { k::put_neon::<BPC, FI>(dst, ds, src, ss, w, h, mx, my) }
}

/// `mct[FI]` NEON girişi.
pub(super) fn mc_prep<const BPC: u32, const FI: usize>(
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bd: BitDepth,
    bdmax: u16,
) {
    if !k::prep_kabul::<BPC>(tmp.len(), src.len(), ss, w, h, mx, my, bd, bdmax) {
        return k::prep_skaler::<FI>(tmp, src, ss, w, h, mx, my, bd, bdmax);
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`mc::prep_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `prep_kabul` kaynak bölgesini + `MC_KUYRUK`'u ve `tmp`'de `w·h` öğeyi O(1) denetledi; erişimler denetimlidir. (3) hizalama/takma ad: `tmp` tek `&mut`, `src` `&`.
    unsafe { k::prep_neon::<BPC, FI>(tmp, src, ss, w, h, mx, my) }
}

/// `mc[FI]` 8 bpc DOTPROD girişi.
pub(super) fn mc_put_dot<const FI: usize>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bd: BitDepth,
    bdmax: u16,
) {
    if !k::put_kabul::<8>(dst.len(), ds, src.len(), ss, w, h, mx, my, bd, bdmax) {
        return k::put_skaler::<FI>(dst, ds, src, ss, w, h, mx, my, bd, bdmax);
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `DotprodToken` ile (`mc::put_dot_fn`) verilir; belirteç NEON + DOTPROD tespitinin kanıtıdır. (2) sınır: `put_kabul` kaynak bölgesini + `MC_KUYRUK`'u ve hedefi O(1) denetledi; erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
    unsafe { k::put_dot::<FI>(dst, ds, src, ss, w, h, mx, my) }
}

/// `mct[FI]` 8 bpc DOTPROD girişi.
pub(super) fn mc_prep_dot<const FI: usize>(
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bd: BitDepth,
    bdmax: u16,
) {
    if !k::prep_kabul::<8>(tmp.len(), src.len(), ss, w, h, mx, my, bd, bdmax) {
        return k::prep_skaler::<FI>(tmp, src, ss, w, h, mx, my, bd, bdmax);
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `DotprodToken` ile (`mc::prep_dot_fn`) verilir; belirteç NEON + DOTPROD tespitinin kanıtıdır. (2) sınır: `prep_kabul` kaynak bölgesini + `MC_KUYRUK`'u ve `tmp`'yi O(1) denetledi; erişimler denetimlidir. (3) hizalama/takma ad: `tmp` tek `&mut`, `src` `&`.
    unsafe { k::prep_dot::<FI>(tmp, src, ss, w, h, mx, my) }
}

/// `avg` NEON girişi.
pub(super) fn mc_avg<const BPC: u32>(
    dst: &mut [u8],
    ds: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    let t = k::Bilesik::Avg;
    if !k::bilesik_kabul::<BPC>(
        t,
        dst.len(),
        ds,
        tmp1.len(),
        tmp2.len(),
        w,
        h,
        0,
        0,
        bd,
        bdmax,
    ) {
        skaler_devri_say();
        return (McDsp::scalar_for(bd).avg)(dst, ds, tmp1, tmp2, w, h, bd, bdmax);
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`mc::avg_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `bilesik_kabul` `tmp1`/`tmp2`'nin `w·h` öğesini ve hedefte `h` satır × `w` pikseli denetledi; erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, girdiler `&`.
    unsafe { k::bilesik_neon::<BPC>(t, dst, ds, tmp1, tmp2, w, h, 0, &[]) }
}

/// `w_avg` NEON girişi.
pub(super) fn mc_w_avg<const BPC: u32>(
    dst: &mut [u8],
    ds: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
    bd: BitDepth,
    bdmax: u16,
) {
    let t = k::Bilesik::WAvg;
    if !k::bilesik_kabul::<BPC>(
        t,
        dst.len(),
        ds,
        tmp1.len(),
        tmp2.len(),
        w,
        h,
        weight,
        0,
        bd,
        bdmax,
    ) {
        skaler_devri_say();
        return (McDsp::scalar_for(bd).w_avg)(dst, ds, tmp1, tmp2, w, h, weight, bd, bdmax);
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`mc::w_avg_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `bilesik_kabul` `tmp1`/`tmp2`'nin `w·h` öğesini, hedefi ve `weight ∈ [0, 16]`'yı denetledi; erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, girdiler `&`.
    unsafe { k::bilesik_neon::<BPC>(t, dst, ds, tmp1, tmp2, w, h, weight, &[]) }
}

/// `mask` NEON girişi.
pub(super) fn mc_mask<const BPC: u32>(
    dst: &mut [u8],
    ds: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &[u8],
    bd: BitDepth,
    bdmax: u16,
) {
    let t = k::Bilesik::Mask;
    if !k::bilesik_kabul::<BPC>(
        t,
        dst.len(),
        ds,
        tmp1.len(),
        tmp2.len(),
        w,
        h,
        0,
        mask.len(),
        bd,
        bdmax,
    ) {
        skaler_devri_say();
        return (McDsp::scalar_for(bd).mask)(dst, ds, tmp1, tmp2, w, h, mask, bd, bdmax);
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`mc::mask_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `bilesik_kabul` `tmp1`/`tmp2`/`mask`'ın `w·h` öğesini ve hedefi denetledi; erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, girdiler `&`.
    unsafe { k::bilesik_neon::<BPC>(t, dst, ds, tmp1, tmp2, w, h, 0, mask) }
}

/// `w_mask[SUB]` NEON girişi.
pub(super) fn mc_w_mask<const BPC: u32, const SUB: usize>(
    dst: &mut [u8],
    ds: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &mut [u8],
    sign: i32,
    ss_hor: bool,
    ss_ver: bool,
    bd: BitDepth,
    bdmax: u16,
) {
    if !k::w_mask_kabul::<BPC, SUB>(
        dst.len(),
        ds,
        tmp1.len(),
        tmp2.len(),
        w,
        h,
        mask.len(),
        sign,
        ss_hor,
        ss_ver,
        bd,
        bdmax,
    ) {
        skaler_devri_say();
        return (McDsp::scalar_for(bd).w_mask[SUB.min(2)])(
            dst, ds, tmp1, tmp2, w, h, mask, sign, ss_hor, ss_ver, bd, bdmax,
        );
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`mc::w_mask_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `w_mask_kabul` `tmp` öğelerini, hedefi ve alt türün maske boyunu denetledi; erişimler denetimlidir. (3) hizalama/takma ad: `dst` ve `mask` ayrı `&mut`, girdiler `&`.
    unsafe { k::w_mask_neon::<BPC, SUB>(dst, ds, tmp1, tmp2, w, h, mask, sign) }
}

/// `blend` NEON girişi.
pub(super) fn mc_blend<const BPC: u32>(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    mask: &[u8],
    bd: BitDepth,
    bdmax: u16,
) {
    let t = k::Karisim::Blend;
    if !k::karisim_kabul::<BPC>(t, dst.len(), ds, tmp.len(), w, h, mask.len(), bd, bdmax) {
        skaler_devri_say();
        return (McDsp::scalar_for(bd).blend)(dst, ds, tmp, w, h, mask, bd, bdmax);
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`mc::blend_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `karisim_kabul` `tmp`/`mask`'ın `w·h` öğesini ve hedefte `h` satır × `w` pikseli denetledi; erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, girdiler `&`.
    unsafe { k::karisim_neon::<BPC>(t, dst, ds, tmp, w, h, mask) }
}

/// `blend_v` NEON girişi.
pub(super) fn mc_blend_v<const BPC: u32>(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    let t = k::Karisim::V;
    if !k::karisim_kabul::<BPC>(t, dst.len(), ds, tmp.len(), w, h, 0, bd, bdmax) {
        skaler_devri_say();
        return (McDsp::scalar_for(bd).blend_v)(dst, ds, tmp, w, h, bd, bdmax);
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`mc::blend_v_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `karisim_kabul` `tmp`'nin `w·h` öğesini ve hedefi denetledi; erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `tmp` `&`.
    unsafe { k::karisim_neon::<BPC>(t, dst, ds, tmp, w, h, &[]) }
}

/// `blend_h` NEON girişi.
pub(super) fn mc_blend_h<const BPC: u32>(
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    let t = k::Karisim::H;
    if !k::karisim_kabul::<BPC>(t, dst.len(), ds, tmp.len(), w, h, 0, bd, bdmax) {
        skaler_devri_say();
        return (McDsp::scalar_for(bd).blend_h)(dst, ds, tmp, w, h, bd, bdmax);
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`mc::blend_h_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `karisim_kabul` `tmp`'nin `w·h` öğesini ve yazılan `(h·3) >> 2` satırı denetledi; erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `tmp` `&`.
    unsafe { k::karisim_neon::<BPC>(t, dst, ds, tmp, w, h, &[]) }
}

/// `warp8x8` NEON girişi.
pub(super) fn mc_warp<const BPC: u32>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bd: BitDepth,
    bdmax: u16,
) {
    if !k::warp_kabul::<BPC, false>(src.len(), ss, abcd, mx, my, dst.len(), ds, bd, bdmax) {
        skaler_devri_say();
        return (McDsp::scalar_for(bd).warp8x8)(dst, ds, src, ss, abcd, mx, my, bd, bdmax);
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`mc::warp_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `warp_kabul` 15×15 okuma bölgesini + `MC_KUYRUK`'u, 8×8 hedefi ve filtre dizinlerinin `[0, 192]` aralığını denetledi; erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
    unsafe { k::warp_neon::<BPC, false>(dst, &mut [], ds, src, ss, abcd, mx, my) }
}

/// `warp8x8t` NEON girişi.
pub(super) fn mc_warp_t<const BPC: u32>(
    tmp: &mut [i16],
    ts: usize,
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    bd: BitDepth,
    bdmax: u16,
) {
    if !k::warp_kabul::<BPC, true>(src.len(), ss, abcd, mx, my, tmp.len(), ts, bd, bdmax) {
        skaler_devri_say();
        return (McDsp::scalar_for(bd).warp8x8t)(tmp, ts, src, ss, abcd, mx, my, bd, bdmax);
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`mc::warp_t_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `warp_kabul` 15×15 okuma bölgesini + `MC_KUYRUK`'u, 8×8 `tmp` hedefini ve filtre dizinlerini denetledi; erişimler denetimlidir. (3) hizalama/takma ad: `tmp` tek `&mut`, `src` `&`.
    unsafe { k::warp_neon::<BPC, true>(&mut [], tmp, ts, src, ss, abcd, mx, my) }
}

/// `emu_edge` NEON girişi.
pub(super) fn mc_emu_edge<const BPC: u32>(
    bw: usize,
    bh: usize,
    iw: usize,
    ih: usize,
    x: i32,
    y: i32,
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    bd: BitDepth,
    bdmax: u16,
) {
    if !k::emu_edge_kabul::<BPC>(
        bw,
        bh,
        iw,
        ih,
        x,
        y,
        dst.len(),
        ds,
        src.len(),
        ss,
        bd,
        bdmax,
    ) {
        skaler_devri_say();
        return (McDsp::scalar_for(bd).emu_edge)(bw, bh, iw, ih, x, y, dst, ds, src, ss, bd, bdmax);
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`mc::emu_edge_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `emu_edge_kabul` kırpılmış okuma penceresini kaynakta, `bh` satır × `bw` pikseli hedefte denetledi; erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `src` `&`.
    unsafe { k::emu_edge_neon::<BPC>(bw, bh, iw, ih, x, y, dst, ds, src, ss) }
}

// ============================================================================
// F5c / P53 — ITX giriş köprüsü ve yükleme yardımcısı
// ============================================================================

/// 4 adet `i32` (`&[i32; 4]`, 16 bayt) `int32x4_t` olarak yükler (`vld1q_s32`).
#[target_feature(enable = "neon")]
#[inline]
#[must_use]
pub fn load_i32x4(src: &[i32; 4]) -> int32x4_t {
    // SAFETY: (1) ISA: `vld1q_s32` ARM64 NEON komutudur; işlev `#[target_feature(enable = "neon")]` taşır. (2) sınır: `src` sabit boyutlu `&[i32; 4]` (16 bayt) referansıdır ve komut tam 16 bayt okur. (3) hizalama/takma ad: `vld1q_s32` `&[i32; 4]` üzerinde geçerlidir; paylaşılan referanstan salt okuma yapılır.
    unsafe { vld1q_s32(src.as_ptr()) }
}

/// `itxfm_add[tx][txtp]` NEON girişi (`BPC` = 8 / 10 / 12; bütün hücreler).
pub(super) fn itx_add<const BPC: u32>(
    dst: &mut [u8],
    ds: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
    bd: BitDepth,
) {
    let ok = bd.bits() == BPC
        && if BPC == 8 {
            itx8_sozlesmesi(dst.len(), ds, coeff.len(), eob, tx, txtp)
        } else {
            itx16_sozlesmesi(dst.len(), ds, coeff.len(), eob, tx, txtp)
        };
    if !ok {
        return itx_skaler_devret::<BPC>(dst, ds, coeff, eob, tx, txtp);
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`itx::itx_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `itx8_sozlesmesi`/`itx16_sozlesmesi` desteklenen hücreyi, `coeff ≥ 1024`, `eob < sw·sh` ve hedefte `h` satır × `w` pikseli O(1) denetledi; gövde erişimleri dilim/dizi denetimlidir. (3) hizalama/takma ad: `dst` ve `coeff` ayrı `&mut`; ödünç kuralları çakışmayı dışlar.
    unsafe { super::itx::itx_neon::<BPC>(dst, ds, coeff, eob, tx, txtp) }
}

// ============================================================================
// F5d / P54 — IPRED, FG, LF, LR, MSAC giriş köprüleri ve `store_i32x4`
// ============================================================================

/// 4 adet `i32` (`&mut [i32; 4]`, 16 bayt) saklar (`vst1q_s32`).
#[target_feature(enable = "neon")]
#[inline]
pub fn store_i32x4(dst: &mut [i32; 4], v: int32x4_t) {
    // SAFETY: (1) ISA: `vst1q_s32` ARM64 NEON komutudur; işlev `#[target_feature(enable = "neon")]` taşır. (2) sınır: `dst` sabit boyutlu `&mut [i32; 4]` (16 bayt) referansıdır ve komut tam 16 bayt yazar. (3) hizalama/takma ad: `vst1q_s32` `&mut [i32; 4]` üzerinde geçerlidir; `&mut` tek erişim yoludur.
    unsafe { vst1q_s32(dst.as_mut_ptr(), v) }
}

/// `intra_pred[MODE]` NEON girişi (`BPC` = 8 / 10 / 12).
pub(super) fn ipred_angular<const BPC: u32, const MODE: usize>(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    angle: usize,
    max_width: usize,
    max_height: usize,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    let Some(mode) = IntraPredMode::from_index(MODE) else {
        return;
    };
    if !angular_sozlesmesi::<BPC>(
        mode,
        dst.len(),
        dst_stride,
        edge,
        width,
        height,
        angle,
        bit_depth,
        bitdepth_max,
    ) {
        return angular_skaler_devret::<BPC>(
            mode,
            dst,
            dst_stride,
            edge,
            width,
            height,
            angle,
            max_width,
            max_height,
            bit_depth,
            bitdepth_max,
        );
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`ipred::intra_pred_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `angular_sozlesmesi` blok geometrisini, `edge` okuma aralığını ve hedef tamponunu O(1) denetledi; erişimler dilim/dizi denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `edge` `&`.
    unsafe {
        super::ipred::angular_neon::<BPC, MODE>(
            dst, dst_stride, edge, width, height, angle, max_width, max_height,
        )
    }
}

/// `cfl_ac[LAYOUT]` NEON girişi (`BPC` = 8 / 10 / 12).
pub(super) fn ipred_cfl_ac<const BPC: u32, const LAYOUT: usize>(
    ac: &mut [i16],
    y: &[u8],
    y_stride: usize,
    w_pad: usize,
    h_pad: usize,
    cw: usize,
    ch: usize,
    bit_depth: BitDepth,
) {
    if !cfl_ac_sozlesmesi::<BPC>(
        LAYOUT,
        ac.len(),
        y.len(),
        y_stride,
        w_pad,
        h_pad,
        cw,
        ch,
        bit_depth,
    ) {
        return cfl_ac_skaler_devret::<BPC>(
            LAYOUT, ac, y, y_stride, w_pad, h_pad, cw, ch, bit_depth,
        );
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`ipred::cfl_ac_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `cfl_ac_sozlesmesi` `ac` (`cw·ch`) ve `y` luma bölgesini O(1) denetledi; erişimler denetimlidir. (3) hizalama/takma ad: `ac` tek `&mut`, `y` `&`.
    unsafe { super::ipred::cfl_ac_neon::<BPC, LAYOUT>(ac, y, y_stride, w_pad, h_pad, cw, ch) }
}

/// `cfl_pred[MODE]` NEON girişi (`BPC` = 8 / 10 / 12).
pub(super) fn ipred_cfl_pred<const BPC: u32, const MODE: usize>(
    dst: &mut [u8],
    dst_stride: usize,
    edge: Edge<'_>,
    width: usize,
    height: usize,
    ac: &[i16],
    alpha: i32,
    bit_depth: BitDepth,
    bitdepth_max: u16,
) {
    let mode = CflPredMode::ALL[MODE.min(3)];
    if !cfl_pred_sozlesmesi::<BPC>(
        mode,
        dst.len(),
        dst_stride,
        edge,
        width,
        height,
        ac.len(),
        alpha,
        bit_depth,
        bitdepth_max,
    ) {
        return cfl_pred_skaler_devret::<BPC>(
            mode,
            dst,
            dst_stride,
            edge,
            width,
            height,
            ac,
            alpha,
            bit_depth,
            bitdepth_max,
        );
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`ipred::cfl_pred_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `cfl_pred_sozlesmesi` `ac`, `edge`, `alpha` ve `dst` bölgesini O(1) denetledi; erişimler denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `edge` ve `ac` `&`.
    unsafe {
        super::ipred::cfl_pred_neon::<BPC, MODE>(dst, dst_stride, edge, width, height, ac, alpha)
    }
}

/// `pal_pred` NEON girişi (`BPC` = 8 / 10 / 12).
pub(super) fn ipred_pal_pred<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    pal: &[u8],
    idx: &[u8],
    w: usize,
    h: usize,
    bit_depth: BitDepth,
) {
    if !pal_pred_sozlesmesi::<BPC>(dst.len(), dst_stride, pal.len(), idx.len(), w, h, bit_depth) {
        return pal_pred_skaler_devret::<BPC>(dst, dst_stride, pal, idx, w, h, bit_depth);
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`ipred::pal_pred_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `pal_pred_sozlesmesi` `idx.len() >= (w·h)/2` ve `pal`/`dst` boyutlarını O(1) denetledi. (3) hizalama/takma ad: `idx` salt okunur `&` dilimdir.
    let gecerli = unsafe { super::ipred::pal_idx_gecerli_neon(idx, (w * h) / 2) };
    if !gecerli {
        return pal_pred_skaler_devret::<BPC>(dst, dst_stride, pal, idx, w, h, bit_depth);
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`ipred::pal_pred_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `pal_pred_sozlesmesi` ve `pal_idx_gecerli_neon` tüm palet indekslerini ve tampon sınırlarını doğruladı. (3) hizalama/takma ad: `dst` tek `&mut`, `pal` ve `idx` `&`.
    unsafe { super::ipred::pal_pred_neon::<BPC>(dst, dst_stride, pal, idx, w, h) }
}

/// `generate_grain_y` NEON girişi (`BPC` = 8 / 10 / 12).
pub(super) fn fg_grain_y<const BPC: u32>(buf: &mut GrainLut, data: &FilmGrainData, bd: FgBitDepth) {
    let ok = bd.bits() == BPC
        && data.seed <= 0xFFFF
        && (0..=3).contains(&data.ar_coeff_lag)
        && (0..=3).contains(&data.grain_scale_shift)
        && (6..=9).contains(&data.ar_coeff_shift);
    if !ok {
        fg_skaler_devri_say();
        return (FgDsp::scalar_for(bd).generate_grain_y)(buf, data, bd);
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`fg::generate_grain_y_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `GrainLut` sabit boyutlu `[[i16; 82]; 73]` türdür ve kaydırma aralıkları denetlendi. (3) hizalama/takma ad: `buf` tek `&mut`, `data` `&`.
    unsafe { super::fg::generate_grain_y_neon::<BPC>(buf, data) }
}

/// `generate_grain_uv[LAYOUT]` NEON girişi (`BPC` = 8 / 10 / 12).
pub(super) fn fg_grain_uv<const BPC: u32, const LAYOUT: usize>(
    buf: &mut GrainLut,
    buf_y: &GrainLut,
    data: &FilmGrainData,
    uv: usize,
    bd: FgBitDepth,
) {
    let ok = bd.bits() == BPC
        && uv < 2
        && data.seed <= 0xFFFF
        && (0..=3).contains(&data.ar_coeff_lag)
        && (0..=3).contains(&data.grain_scale_shift)
        && (6..=9).contains(&data.ar_coeff_shift);
    if !ok {
        fg_skaler_devri_say();
        return (FgDsp::scalar_for(bd).generate_grain_uv[LAYOUT.min(2)])(buf, buf_y, data, uv, bd);
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`fg::generate_grain_uv_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `GrainLut` sabit boyutlu dizidir, `uv < 2` ve parametre aralıkları denetlendi. (3) hizalama/takma ad: `buf` tek `&mut`, `buf_y` ve `data` `&`.
    unsafe {
        match LAYOUT {
            0 => super::fg::generate_grain_uv_neon::<BPC, 1, 1>(buf, buf_y, data, uv),
            1 => super::fg::generate_grain_uv_neon::<BPC, 1, 0>(buf, buf_y, data, uv),
            _ => super::fg::generate_grain_uv_neon::<BPC, 0, 0>(buf, buf_y, data, uv),
        }
    }
}

/// `fgy_32x32xn` NEON girişi (`BPC` = 8 / 10 / 12).
pub(super) fn fg_fgy<const BPC: u32>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8],
    grain_lut: &GrainLut,
    bh: usize,
    row_num: usize,
    bd: FgBitDepth,
) {
    let ok = bd.bits() == BPC
        && if BPC == 8 {
            fgy_sozlesmesi(dst.len(), src.len(), stride, pw, bh, data)
        } else {
            fgy16_sozlesmesi(dst.len(), src.len(), stride, pw, bh, data, bd)
        };
    if ok {
        if BPC == 8
            && let Some(sc) = scaling.first_chunk::<256>()
        {
            // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`fg::fgy_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `fgy_sozlesmesi` `src`/`dst` boyutlarını ve `first_chunk` 256 baytlık ölçekleme tablosunu denetledi. (3) hizalama/takma ad: `dst` tek `&mut`, girdiler `&`.
            return unsafe {
                super::fg::fgy_neon::<8, 256>(
                    dst, src, stride, data, pw, sc, grain_lut, bh, row_num,
                )
            };
        } else if BPC == 10
            && let Some(sc) = scaling.first_chunk::<1024>()
        {
            // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`fg::fgy_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `fgy16_sozlesmesi` `src`/`dst` boyutlarını ve `first_chunk` 1024 baytlık ölçekleme tablosunu denetledi. (3) hizalama/takma ad: `dst` tek `&mut`, girdiler `&`.
            return unsafe {
                super::fg::fgy_neon::<10, 1024>(
                    dst, src, stride, data, pw, sc, grain_lut, bh, row_num,
                )
            };
        } else if BPC == 12
            && let Some(sc) = scaling.first_chunk::<4096>()
        {
            // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`fg::fgy_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `fgy16_sozlesmesi` `src`/`dst` boyutlarını ve `first_chunk` 4096 baytlık ölçekleme tablosunu denetledi. (3) hizalama/takma ad: `dst` tek `&mut`, girdiler `&`.
            return unsafe {
                super::fg::fgy_neon::<12, 4096>(
                    dst, src, stride, data, pw, sc, grain_lut, bh, row_num,
                )
            };
        }
    }
    fg_skaler_devri_say();
    (FgDsp::scalar_for(bd).fgy_32x32xn)(
        dst, src, stride, data, pw, scaling, grain_lut, bh, row_num, bd,
    );
}

/// `fguv_32x32xn[LAYOUT]` NEON girişi (`BPC` = 8 / 10 / 12).
pub(super) fn fg_fguv<const BPC: u32, const LAYOUT: usize>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    data: &FilmGrainData,
    pw: usize,
    scaling: &[u8],
    grain_lut: &GrainLut,
    bh: usize,
    row_num: usize,
    luma: &[u8],
    luma_stride: usize,
    uv: usize,
    is_id: bool,
    bd: FgBitDepth,
) {
    let sx = usize::from(LAYOUT != 2);
    let sy = usize::from(LAYOUT == 0);
    let ok = bd.bits() == BPC
        && if BPC == 8 {
            fguv_sozlesmesi(
                dst.len(),
                src.len(),
                stride,
                pw,
                bh,
                luma.len(),
                luma_stride,
                uv,
                sx,
                sy,
                data,
            )
        } else {
            fguv16_sozlesmesi(
                dst.len(),
                src.len(),
                stride,
                pw,
                bh,
                luma.len(),
                luma_stride,
                uv,
                sx,
                sy,
                data,
                bd,
            )
        };
    if ok {
        if BPC == 8
            && let Some(sc) = scaling.first_chunk::<256>()
        {
            // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`fg::fguv_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `fguv_sozlesmesi` `src`/`dst`/`luma` boyutlarını ve `first_chunk` 256 baytlık tabloyu denetledi. (3) hizalama/takma ad: `dst` tek `&mut`, girdiler `&`.
            return unsafe {
                match LAYOUT {
                    0 => super::fg::fguv_neon::<8, 1, 1, 256>(
                        dst,
                        src,
                        stride,
                        data,
                        pw,
                        sc,
                        grain_lut,
                        bh,
                        row_num,
                        luma,
                        luma_stride,
                        uv,
                        is_id,
                    ),
                    1 => super::fg::fguv_neon::<8, 1, 0, 256>(
                        dst,
                        src,
                        stride,
                        data,
                        pw,
                        sc,
                        grain_lut,
                        bh,
                        row_num,
                        luma,
                        luma_stride,
                        uv,
                        is_id,
                    ),
                    _ => super::fg::fguv_neon::<8, 0, 0, 256>(
                        dst,
                        src,
                        stride,
                        data,
                        pw,
                        sc,
                        grain_lut,
                        bh,
                        row_num,
                        luma,
                        luma_stride,
                        uv,
                        is_id,
                    ),
                }
            };
        } else if BPC == 10
            && let Some(sc) = scaling.first_chunk::<1024>()
        {
            // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`fg::fguv_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `fguv16_sozlesmesi` `src`/`dst`/`luma` boyutlarını ve `first_chunk` 1024 baytlık tabloyu denetledi. (3) hizalama/takma ad: `dst` tek `&mut`, girdiler `&`.
            return unsafe {
                match LAYOUT {
                    0 => super::fg::fguv_neon::<10, 1, 1, 1024>(
                        dst,
                        src,
                        stride,
                        data,
                        pw,
                        sc,
                        grain_lut,
                        bh,
                        row_num,
                        luma,
                        luma_stride,
                        uv,
                        is_id,
                    ),
                    1 => super::fg::fguv_neon::<10, 1, 0, 1024>(
                        dst,
                        src,
                        stride,
                        data,
                        pw,
                        sc,
                        grain_lut,
                        bh,
                        row_num,
                        luma,
                        luma_stride,
                        uv,
                        is_id,
                    ),
                    _ => super::fg::fguv_neon::<10, 0, 0, 1024>(
                        dst,
                        src,
                        stride,
                        data,
                        pw,
                        sc,
                        grain_lut,
                        bh,
                        row_num,
                        luma,
                        luma_stride,
                        uv,
                        is_id,
                    ),
                }
            };
        } else if BPC == 12
            && let Some(sc) = scaling.first_chunk::<4096>()
        {
            // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`fg::fguv_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `fguv16_sozlesmesi` `src`/`dst`/`luma` boyutlarını ve `first_chunk` 4096 baytlık tabloyu denetledi. (3) hizalama/takma ad: `dst` tek `&mut`, girdiler `&`.
            return unsafe {
                match LAYOUT {
                    0 => super::fg::fguv_neon::<12, 1, 1, 4096>(
                        dst,
                        src,
                        stride,
                        data,
                        pw,
                        sc,
                        grain_lut,
                        bh,
                        row_num,
                        luma,
                        luma_stride,
                        uv,
                        is_id,
                    ),
                    1 => super::fg::fguv_neon::<12, 1, 0, 4096>(
                        dst,
                        src,
                        stride,
                        data,
                        pw,
                        sc,
                        grain_lut,
                        bh,
                        row_num,
                        luma,
                        luma_stride,
                        uv,
                        is_id,
                    ),
                    _ => super::fg::fguv_neon::<12, 0, 0, 4096>(
                        dst,
                        src,
                        stride,
                        data,
                        pw,
                        sc,
                        grain_lut,
                        bh,
                        row_num,
                        luma,
                        luma_stride,
                        uv,
                        is_id,
                    ),
                }
            };
        }
    }
    fg_skaler_devri_say();
    (FgDsp::scalar_for(bd).fguv_32x32xn[LAYOUT.min(2)])(
        dst,
        src,
        stride,
        data,
        pw,
        scaling,
        grain_lut,
        bh,
        row_num,
        luma,
        luma_stride,
        uv,
        is_id,
        bd,
    );
}

/// `loop_filter_sb[PLANE][DIR]` NEON girişi (`BPC` = 8 / 10 / 12).
pub(super) fn lf_sb<const BPC: u32, const PLANE: usize, const DIR: usize>(
    dst: &mut [u8],
    stride: usize,
    mask: &[u8],
    mask_stride: usize,
    level: &[[u8; 4]],
    lvl_stride: usize,
    lvl_idx: usize,
    lut: &LimLut,
    w4: usize,
    y0: usize,
    count: usize,
) {
    if !lpf_sb_sozlesmesi::<BPC>(
        PLANE,
        DIR,
        dst.len(),
        stride,
        mask.len(),
        mask_stride,
        level.len(),
        lvl_stride,
        lvl_idx,
        w4,
        y0,
        count,
    ) {
        return lpf_sb_skaler_devret::<BPC>(
            PLANE,
            DIR,
            dst,
            stride,
            mask,
            mask_stride,
            level,
            lvl_stride,
            lvl_idx,
            lut,
            w4,
            y0,
            count,
        );
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`lf::loop_filter_sb_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `lpf_sb_sozlesmesi` `dst`, `mask` ve `level` boyutlarını O(1) denetledi; erişimler dilim denetimlidir. (3) hizalama/takma ad: `dst` tek `&mut`, `mask`/`level`/`lut` `&`.
    unsafe {
        super::lf::lpf_sb_neon::<BPC, PLANE, DIR>(
            dst,
            stride,
            mask,
            mask_stride,
            level,
            lvl_stride,
            lvl_idx,
            lut,
            w4,
            y0,
            count,
        )
    }
}

/// `wiener[TAP]` NEON girişi (`BPC` = 8 / 10 / 12; `TAP`: 0 = 7-tap, 1 = 5-tap).
pub(super) fn lr_wiener<const BPC: u32, const TAP: usize>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    lpf: &[u8],
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    params: &LrParams,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
) {
    if !wiener_sozlesmesi::<BPC>(
        TAP,
        dst.len(),
        stride,
        left.len(),
        lpf.len(),
        lpf_stride,
        lpf_x,
        w,
        h,
        params,
        edges,
        bd,
    ) {
        return wiener_skaler_devret::<BPC>(
            TAP, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
        );
    }
    let LrParams::Wiener { filter } = params else {
        return;
    };
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`lr::wiener_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `wiener_sozlesmesi` `dst`, `left`, `lpf`, `w ∈ 1..=384` ve kenar erişimlerini O(1) denetledi. (3) hizalama/takma ad: `dst` tek `&mut`, `left` ve `lpf` `&`.
    unsafe {
        if TAP == 0 {
            super::lr::wiener_neon::<BPC, false>(
                dst, stride, left, lpf, lpf_stride, lpf_x, w, h, filter, edges,
            );
        } else {
            super::lr::wiener_neon::<BPC, true>(
                dst, stride, left, lpf, lpf_stride, lpf_x, w, h, filter, edges,
            );
        }
    }
}

/// `sgr[SGR_IDX]` NEON girişi (`BPC` = 8 / 10; `SGR_IDX`: 0 = 5×5, 1 = 3×3, 2 = mix).
pub(super) fn lr_sgr<const BPC: u32, const SGR_IDX: usize>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    lpf: &[u8],
    lpf_stride: usize,
    lpf_x: usize,
    w: usize,
    h: usize,
    params: &LrParams,
    edges: LrEdgeFlags,
    bd: LrBitDepth,
) {
    if !sgr_sozlesmesi::<BPC>(
        SGR_IDX,
        dst.len(),
        stride,
        left.len(),
        lpf.len(),
        lpf_stride,
        lpf_x,
        w,
        h,
        params,
        edges,
        bd,
    ) {
        return sgr_skaler_devret::<BPC>(
            SGR_IDX, dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges, bd,
        );
    }
    // SAFETY: (1) ISA: Bu köprünün adresi yalnız `NeonToken` ile (`lr::sgr_fn`) verilir; belirteç NEON tespitinin kanıtıdır. (2) sınır: `sgr_sozlesmesi` `dst`, `left`, `lpf`, `w ∈ 1..=384` ve `reach` kenarlarını O(1) denetledi. (3) hizalama/takma ad: `dst` tek `&mut`, `left` ve `lpf` `&`.
    unsafe {
        super::lr::sgr_neon::<BPC, SGR_IDX>(
            dst, stride, left, lpf, lpf_stride, lpf_x, w, h, params, edges,
        )
    }
}

/// `dav1d_msac_decode_symbol_adapt4_neon` (`1 <= n_symbols <= 3`, `cdf.len() >= 4`).
#[inline(always)]
pub fn msac_decode_symbol_adapt4(
    _token: NeonToken,
    s: &mut MsacContext<'_>,
    cdf: &mut [u16; 4],
    n_symbols: usize,
) -> usize {
    super::msac::adapt4_neon(s, cdf, n_symbols)
}

/// `dav1d_msac_decode_symbol_adapt8_neon` (`1 <= n_symbols <= 7`, `cdf.len() >= 8`).
#[inline(always)]
pub fn msac_decode_symbol_adapt8(
    _token: NeonToken,
    s: &mut MsacContext<'_>,
    cdf: &mut [u16; 8],
    n_symbols: usize,
) -> usize {
    super::msac::adapt8_neon(s, cdf, n_symbols)
}

/// `dav1d_msac_decode_symbol_adapt16_neon` (`1 <= n_symbols <= 15`, `cdf.len() >= 16`).
#[inline(always)]
pub fn msac_decode_symbol_adapt16(
    _token: NeonToken,
    s: &mut MsacContext<'_>,
    cdf: &mut [u16; 16],
    n_symbols: usize,
) -> usize {
    super::msac::adapt16_neon(s, cdf, n_symbols)
}

/// `dav1d_msac_decode_hi_tok_neon` (`cdf.len() >= 4`).
#[inline(always)]
pub fn msac_decode_hi_tok(_token: NeonToken, s: &mut MsacContext<'_>, cdf: &mut [u16; 4]) -> usize {
    super::msac::hi_tok_neon(s, cdf)
}

/// `dav1d_msac_decode_bool_adapt_neon` (`cdf.len() >= 2`).
#[inline(always)]
pub fn msac_decode_bool_adapt(
    _token: NeonToken,
    s: &mut MsacContext<'_>,
    cdf: &mut [u16; 2],
) -> u32 {
    super::msac::bool_adapt_neon(s, cdf)
}

/// `dav1d_msac_decode_bool_equi_neon`.
#[inline(always)]
pub fn msac_decode_bool_equi(_token: NeonToken, s: &mut MsacContext<'_>) -> u32 {
    super::msac::bool_equi_neon(s)
}

/// `dav1d_msac_decode_bool_neon`.
#[inline(always)]
pub fn msac_decode_bool(_token: NeonToken, s: &mut MsacContext<'_>, f: u32) -> u32 {
    super::msac::bool_neon(s, f)
}

#[cfg(test)]
pub(super) fn sozluk_neon(
    _token: NeonToken,
    kip: super::sozluk::NeonKip,
    a: &[u8; 16],
    b: &[u8; 16],
) -> [u8; 16] {
    // SAFETY: (1) ISA: `_token` (`NeonToken`) bu işlemcide NEON etkinliğini kanıtlar; `neon_govde` yalnız NEON ister. (2) sınır: `a`, `b` ve dönüş değeri sabit boyutlu `[u8; 16]` dizileridir. (3) hizalama/takma ad: paylaşılan `&[u8; 16]` referansları yalnız okunur.
    unsafe { super::sozluk::neon_govde(kip, a, b) }
}

#[cfg(test)]
pub(super) fn sozluk_dotprod(
    _token: crate::cpu::DotprodToken,
    acc: &[u8; 16],
    a: &[u8; 16],
    b: &[u8; 16],
    signed: bool,
) -> [u8; 16] {
    // SAFETY: (1) ISA: `_token` (`DotprodToken`) bu işlemcide NEON ve DOTPROD etkinliğini kanıtlar; `dotprod_govde` yalnız `neon,dotprod` ister. (2) sınır: `acc`, `a`, `b` ve dönüş değeri sabit boyutlu `[u8; 16]` dizileridir. (3) hizalama/takma ad: paylaşılan `&[u8; 16]` referansları yalnız okunur.
    unsafe { super::sozluk::dotprod_govde(acc, a, b, signed) }
}
