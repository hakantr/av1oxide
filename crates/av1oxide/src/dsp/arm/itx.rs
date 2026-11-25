//! `aarch64` NEON ters dönüşüm + piksel ekleme (`F5c` / `P53` — `F5-ITX`).
//!
//! ## Kapsam
//!
//! dav1d `src/arm/itx.h` NEON kayıtları: `8 / 10 / 12` bpc × `156` `itxfm_add[tx][txtp]`
//! alanı (`WHT_WHT` dâhil) = **`468` yuva**. Tek bir genel gövde ([`itx_neon`]) bütün
//! hücrelere hizmet eder: tx boyutu/türü çağrı bağımsız değişkenidir
//! ([`crate::dsp::itx::ItxFn`]), monomorfizasyon yalnız `BPC` ve blok boyutu
//! (`const W`, `const H`) üzerinedir.
//!
//! ## Yöntem (skaler `dsp/scalar/itx.rs` ile aynı işlem sırası)
//!
//! - **1B dönüşümler** [`super::itx_1d`]'dedir: skaler 1B kodun mekanik `int32x4_t`
//!   çevirisi (`scripts/uretec/itx1d_neon.py`); bir vektörün 4 şeridi 4 bağımsız
//!   satır (ilk geçiş) ya da 4 bağımsız sütundur (ikinci geçiş).
//! - **İlk geçiş:** `last_nonzero_col`'a kadar 4'lü satır grupları; katsayılar
//!   `coeff[x·sh + y]` yerleşiminde olduğundan 4 satır tek `ld1` ile okunur;
//!   `last_nonzero_col` ötesindeki şeritler sıfırlanır (skaler o satırları işlemez).
//!   Ardından `rect2` ölçeği, satır 1B dönüşümü (satır kırpma), `(x + rnd) >> shift`
//!   (`srshl`) ve sütun kırpması; 4×4 bloklar devrilerek sütun tamponuna yazılır.
//! - **İkinci geçiş:** 4'lü sütun grupları üzerinde sütun 1B dönüşümü (sütun kırpma),
//!   `(x + 8) >> 4` (`srshr`), piksele ekleme ve `[0, 2^BPC − 1]` kırpması.
//! - **DC-only** (`DCT_DCT`, `eob = 0`) ve **WHT** yolları skalerle aynı aritmetikle.
//!
//! ## Aritmetik ve bellek (SMD-01/SMD-02, ADR-0003)
//!
//! `i32` modüler aritmetik; geçerli AV1 katsayı aralığında (`|c| ≤ cf_max`) skaler
//! `i64` ile bit-birebir (gerekçe üreteç belgesinde). Sözleşme dışı girdi
//! ([`crate::dsp::itx::itx8_sozlesmesi`] / `itx16_sozlesmesi`) skalere devreder ve
//! sayılır. Erişimler dilim/dizi denetimlidir. `%100 #![forbid(unsafe_code)]`.

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use core::arch::aarch64::*;

use super::giris::load_i32x4;
use super::itx_1d as k;
use crate::cpu::NeonToken;
use crate::dsp::itx::{ItxFn, TxSize, TxType};
use crate::dsp::itx_tables::LNZC_TABLES;
use crate::dsp::scalar::itx_1d::Tx1dType;

/// `itxfm_add[..][..]` NEON yuvası (`BPC` = 8 / 10 / 12); yalnız [`NeonToken`] ile.
#[must_use]
pub const fn itx_fn<const BPC: u32>(_neon: NeonToken) -> ItxFn {
    super::giris::itx_add::<BPC>
}

/// 1B dönüşümü uygular (`log2` boyut dizini, tür; `c` vektör dizisi, adım 1).
#[target_feature(enable = "neon")]
#[inline]
fn uygula_1d(lsz: u8, tur: Tx1dType, c: &mut [int32x4_t], min: int32x4_t, max: int32x4_t) {
    match (lsz, tur) {
        (0, Tx1dType::Dct) => k::inv_dct4_1d_c(c, 1, min, max),
        (0, Tx1dType::Adst) => k::inv_adst4_1d_c(c, 1, min, max),
        (0, Tx1dType::FlipAdst) => k::inv_flipadst4_1d_c(c, 1, min, max),
        (0, Tx1dType::Identity) => k::inv_identity4_1d_c(c, 1, min, max),
        (1, Tx1dType::Dct) => k::inv_dct8_1d_c(c, 1, min, max),
        (1, Tx1dType::Adst) => k::inv_adst8_1d_c(c, 1, min, max),
        (1, Tx1dType::FlipAdst) => k::inv_flipadst8_1d_c(c, 1, min, max),
        (1, Tx1dType::Identity) => k::inv_identity8_1d_c(c, 1, min, max),
        (2, Tx1dType::Dct) => k::inv_dct16_1d_c(c, 1, min, max),
        (2, Tx1dType::Adst) => k::inv_adst16_1d_c(c, 1, min, max),
        (2, Tx1dType::FlipAdst) => k::inv_flipadst16_1d_c(c, 1, min, max),
        (2, Tx1dType::Identity) => k::inv_identity16_1d_c(c, 1, min, max),
        (3, Tx1dType::Dct) => k::inv_dct32_1d_c(c, 1, min, max),
        (3, Tx1dType::Identity) => k::inv_identity32_1d_c(c, 1, min, max),
        (4, Tx1dType::Dct) => k::inv_dct64_1d_c(c, 1, min, max),
        // Desteklenmeyen hücre sözleşmede elenir (`is_supported`).
        _ => unreachable!("desteklenmeyen 1B hücre"),
    }
}

/// 4×4 `i32` devrik.
#[target_feature(enable = "neon")]
#[inline]
fn devrik4(a: int32x4_t, b: int32x4_t, c: int32x4_t, d: int32x4_t) -> [int32x4_t; 4] {
    let t0 = vtrn1q_s32(a, b);
    let t1 = vtrn2q_s32(a, b);
    let t2 = vtrn1q_s32(c, d);
    let t3 = vtrn2q_s32(c, d);
    let s = vreinterpretq_s64_s32;
    let r = vreinterpretq_s32_s64;
    [
        r(vtrn1q_s64(s(t0), s(t2))),
        r(vtrn1q_s64(s(t1), s(t3))),
        r(vtrn2q_s64(s(t0), s(t2))),
        r(vtrn2q_s64(s(t1), s(t3))),
    ]
}

/// 4 pikseli okur (`i32` şeritler).
#[target_feature(enable = "neon")]
#[inline]
fn oku4<const BPC: u32>(dst: &[u8], o: usize) -> int32x4_t {
    if BPC == 8 {
        let mut b = [0u8; 4];
        b.copy_from_slice(&dst[o..o + 4]);
        let v = vreinterpret_u8_u32(vdup_n_u32(u32::from_le_bytes(b)));
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(v))))
    } else {
        let mut b = [0u8; 8];
        b.copy_from_slice(&dst[o..o + 8]);
        vreinterpretq_s32_u32(vmovl_u16(vcreate_u16(u64::from_le_bytes(b))))
    }
}

/// `piksel + d` toplamını `[0, maxv]`'ye kırpıp 4 pikseli yazar.
#[target_feature(enable = "neon")]
#[inline]
fn yaz4<const BPC: u32>(dst: &mut [u8], o: usize, d: int32x4_t, maxv: int32x4_t) {
    let p = vminq_s32(
        vmaxq_s32(vaddq_s32(oku4::<BPC>(dst, o), d), vdupq_n_s32(0)),
        maxv,
    );
    let u = vqmovun_s32(p);
    if BPC == 8 {
        let b = vqmovn_u16(vcombine_u16(u, u));
        let w = vget_lane_u32::<0>(vreinterpret_u32_u8(b));
        dst[o..o + 4].copy_from_slice(&w.to_le_bytes());
    } else {
        let w = vget_lane_u64::<0>(vreinterpret_u64_u16(u));
        dst[o..o + 8].copy_from_slice(&w.to_le_bytes());
    }
}

/// Genel gövde (sözleşme girişte denetlendi).
#[target_feature(enable = "neon")]
pub(super) fn itx_neon<const BPC: u32>(
    dst: &mut [u8],
    ds: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
) {
    let maxv = vdupq_n_s32((1 << BPC) - 1);
    let bpp = if BPC == 8 { 1 } else { 2 };
    if txtp == TxType::WhtWht {
        return wht::<BPC>(dst, ds, coeff, maxv);
    }
    let w = tx.width();
    let h = tx.height();
    let shift = tx.shift() as i32;
    let rect2 = w * 2 == h || h * 2 == w;
    if txtp == TxType::DctDct && eob == 0 {
        let rnd = (1i64 << shift) >> 1;
        let mut dc = i64::from(coeff[0]);
        coeff[0] = 0;
        if rect2 {
            dc = (dc * 181 + 128) >> 8;
        }
        dc = (dc * 181 + 128) >> 8;
        dc = (dc + rnd) >> shift;
        dc = (dc * 181 + 128 + 2048) >> 12;
        let d = vdupq_n_s32(dc as i32);
        for y in 0..h {
            let mut x = 0;
            while x < w {
                yaz4::<BPC>(dst, y * ds + x * bpp, d, maxv);
                x += 4;
            }
        }
        return;
    }
    macro_rules! boyut {
        ($($t:ident => ($w:literal, $h:literal)),* $(,)?) => {
            match tx {
                $(TxSize::$t => genel::<BPC, $w, $h, { $w / 4 }>(dst, ds, coeff, eob, tx, txtp, rect2, shift, maxv),)*
            }
        };
    }
    boyut!(
        Tx4x4 => (4, 4), Tx8x8 => (8, 8), Tx16x16 => (16, 16), Tx32x32 => (32, 32),
        Tx64x64 => (64, 64), Rtx4x8 => (4, 8), Rtx8x4 => (8, 4), Rtx8x16 => (8, 16),
        Rtx16x8 => (16, 8), Rtx16x32 => (16, 32), Rtx32x16 => (32, 16), Rtx32x64 => (32, 64),
        Rtx64x32 => (64, 32), Rtx4x16 => (4, 16), Rtx16x4 => (16, 4), Rtx8x32 => (8, 32),
        Rtx32x8 => (32, 8), Rtx16x64 => (16, 64), Rtx64x16 => (64, 16),
    );
}

/// WHT (yalnız 4×4, kayıpsız) — dav1d `inv_txfm_add_wht_wht_4x4_c`.
#[target_feature(enable = "neon")]
fn wht<const BPC: u32>(dst: &mut [u8], ds: usize, coeff: &mut [i32], maxv: int32x4_t) {
    // v[x] şeridi y: coeff[x·4 + y] >> 2.
    let mut v = [vdupq_n_s32(0); 4];
    for (x, vx) in v.iter_mut().enumerate() {
        *vx = vshrq_n_s32::<2>(load_i32x4(crate::dsp::arm::mem::chunk(coeff, x * 4)));
    }
    k::dav1d_inv_wht4_1d_c(&mut v, 1);
    coeff[..16].fill(0);
    let mut u = devrik4(v[0], v[1], v[2], v[3]);
    k::dav1d_inv_wht4_1d_c(&mut u, 1);
    for (y, uy) in u.iter().enumerate() {
        yaz4::<BPC>(dst, y * ds, *uy, maxv);
    }
}

/// Genel iki geçişli gövde (`W × H` blok).
#[target_feature(enable = "neon")]
fn genel<const BPC: u32, const W: usize, const H: usize, const WG: usize>(
    dst: &mut [u8],
    ds: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
    rect2: bool,
    shift: i32,
    maxv: int32x4_t,
) {
    let bpp = if BPC == 8 { 1 } else { 2 };
    let (rmin, cmin) = if BPC == 8 {
        (-32768i32, -32768i32)
    } else {
        (-((1i32 << BPC) * 128), -((1i32 << BPC) * 32))
    };
    let (rmin_v, rmax_v) = (vdupq_n_s32(rmin), vdupq_n_s32(-rmin - 1));
    let (cmin_v, cmax_v) = (vdupq_n_s32(cmin), vdupq_n_s32(-cmin - 1));
    let info = tx.info();
    let sh = H.min(32);
    let sw = W.min(32);
    let (yatay, dikey) = txtp.tx1d();
    let lnzc = if dikey == Tx1dType::Identity && yatay != Tx1dType::Identity {
        (sh - 1).min(eob)
    } else if yatay == Tx1dType::Identity && dikey != Tx1dType::Identity {
        eob >> (info.lw as usize + 2)
    } else {
        usize::from(LNZC_TABLES[tx.index()][eob])
    };
    let nshift = vdupq_n_s32(-shift);
    let seritler = load_i32x4(&[0, 1, 2, 3]);

    let mut sutun = [[vdupq_n_s32(0); H]; WG];
    let mut satir = [vdupq_n_s32(0); W];
    let mut y0 = 0;
    while y0 <= lnzc {
        // Şerit `i` = satır `y0 + i`; `lnzc` ötesi sıfır.
        let gecerli = vcleq_s32(seritler, vdupq_n_s32((lnzc - y0) as i32));
        for (x, r) in satir.iter_mut().enumerate() {
            if x < sw {
                let c = load_i32x4(crate::dsp::arm::mem::chunk(coeff, x * sh + y0));
                let c = vreinterpretq_s32_u32(vandq_u32(vreinterpretq_u32_s32(c), gecerli));
                *r = if rect2 {
                    vrshrq_n_s32::<8>(vmulq_n_s32(c, 181))
                } else {
                    c
                };
            } else {
                *r = vdupq_n_s32(0);
            }
        }
        uygula_1d(info.lw, yatay, &mut satir, rmin_v, rmax_v);
        for r in &mut satir {
            *r = vminq_s32(vmaxq_s32(vrshlq_s32(*r, nshift), cmin_v), cmax_v);
        }
        for (xg, s) in sutun.iter_mut().enumerate() {
            let t = devrik4(
                satir[4 * xg],
                satir[4 * xg + 1],
                satir[4 * xg + 2],
                satir[4 * xg + 3],
            );
            s[y0..y0 + 4].copy_from_slice(&t);
        }
        y0 += 4;
    }
    coeff[..sw * sh].fill(0);
    for (xg, s) in sutun.iter_mut().enumerate() {
        uygula_1d(info.lh, dikey, s, cmin_v, cmax_v);
        for (y, v) in s.iter().enumerate() {
            yaz4::<BPC>(dst, y * ds + 4 * xg * bpp, vrshrq_n_s32::<4>(*v), maxv);
        }
    }
}
