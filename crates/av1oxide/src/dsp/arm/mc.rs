//! `aarch64` NEON (+ 8-bpc DOTPROD) motion compensation çekirdekleri (`F5b` / `P52` — `F5-MC`).
//!
//! ## Kapsam (`docs/dispatch-slot-map-arm64.csv`, `algorithm = mc`)
//!
//! - `neon` (`8 / 10 / 12` bpc, `32 + 32` kayıt = `96` yuva): `mc[0..9]`, `mct[0..9]`
//!   (8-tap regular/smooth/sharp birleşimleri + bilinear), `avg`, `w_avg`, `mask`,
//!   `w_mask[0..2]`, `blend`, `blend_v`, `blend_h`, `warp8x8`, `warp8x8t`, `emu_edge`
//!   (dav1d `src/arm/mc.h:71-124`, `src/arm/64/{mc.S,mc16.S}`).
//! - `neon_dotprod` (`8` bpc, `18` kayıt): `mc[0..8]`, `mct[0..8]` 8-tap birleşimleri;
//!   yatay geçiş `sdot` (`vdotq_s32`) ile: pikseller `p ^ 0x80 = p − 128` işaretli
//!   bayta çevrilir ve toplamaya `128 · Σc = 8192` sapması eklenir (dav1d
//!   `src/arm/64/mc_dotprod.S` yöntemi; asm kopyalanmadı).
//! - dav1d `src/arm/mc.h` ARM64'te `mc_scaled`, `mct_scaled` ve `resize` için NEON
//!   gövdesi **tanımlamaz**; bu yuvalar ARM64'te skaler kalır (parite).
//!
//! ## Bellek sözleşmesi (kopyasız, [`crate::dsp::mc`] ile aynı)
//!
//! `put`/`prep` kaynağı dolgu hücresinden (blok satırı −3, sütunu −3) başlar; giriş
//! [`crate::dsp::mc::simd_sozlesmesi`] / [`crate::dsp::mc::simd16_sozlesmesi`] ile
//! O(1) denetlenir (bölge sonu + [`crate::dsp::mc::MC_KUYRUK`] bayt). Çekirdekler 8
//! sütunluk şeritler hâlinde çalışır; dar bloklarda (`w < 8`) yüklemeler bölgenin en
//! çok 16 bayt ötesine (kuyruk) uzanır, saklamalar yalnız `w` pikseldir (yerel dizi +
//! dilim kopyası). Bütün erişimler dilim/dizi denetimlidir: sözleşme dışı bir okuma
//! paniğe yol açar, bellek dışına erişim olmaz. Sözleşme dışı girdi skaler gövdeye
//! devreder ve [`crate::dsp::mc::skaler_devir_sayisi`] ile sayılır.
//!
//! ## Aritmetik (SMD-01/SMD-02; dav1d `src/mc_tmpl.c`)
//!
//! - 8 bpc yatay/dikey 8-tap toplamı `i16`'ya sığar: `Σc = 64`, pozitif katsayı
//!   toplamı en çok 92 → `255 · 92 = 23 460`, negatif en çok `−7 140` (birim testi
//!   `sekiz_bpc_i16_toplam_siniri`). Ara (`mid`) değer `(Σ + 2) >> 2` (`srshr #2`),
//!   dikey geçiş `i32` (`smull`/`smlal`).
//! - 10/12 bpc yatay toplamı `i32`; `mid = (Σ + r) >> (BPC − 8)` `i16`'ya sığar.
//! - Yuvarlamalı kaydırmalar `srshl`/`srshr`/`sqrshrun` ile **taşmasız** yapılır;
//!   kırpma `sqxtun` + `umin` (`iclip_pixel`) ile. Bileşik işlevler (`avg`, `w_avg`,
//!   `mask`, `w_mask`) her bit derinliğinde `i32` ara genişlikle çalışır.
//!
//! `%100 #![forbid(unsafe_code)]`: `#[target_feature]` gövdelerine giriş yalnız
//! [`super::giris`] köprüleri üzerinden ve yalnız [`NeonToken`] / [`DotprodToken`]
//! ile alınan işlev işaretçileriyle olur.

#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]

use core::arch::aarch64::*;

use crate::cpu::{DotprodToken, NeonToken};
use crate::dsp::arm::giris::{self, store_i16x8};
use crate::dsp::arm::mem::{
    chunk, chunk_mut, load_i16x8, load_u8x8, load_u8x16, store_u8x8, store_u8x16,
};
use crate::dsp::mc::{
    AvgFn, BlendDirFn, BlendFn, EmuEdgeFn, Filter2d, MaskFn, McDsp, McPrepFn, McPutFn, WAvgFn,
    WMaskFn, Warp8x8Fn, Warp8x8tFn, bilesik_sozlesmesi, bilesik16_sozlesmesi, karisim_sozlesmesi,
    karisim16_sozlesmesi, kenar_sozlesmesi, kenar16_sozlesmesi, simd_sozlesmesi, simd16_sozlesmesi,
    skaler_devri_say, warp_sozlesmesi, warp16_sozlesmesi,
};
use crate::dsp::mc_tables::{MC_SUBPEL_FILTERS, MC_WARP_FILTER, OBMC_MASKS};
use crate::frame::BitDepth;

/// Bilinear yuvasının filtre dizini (`FILTER_2D_BILINEAR`).
pub(super) const BILIN: usize = 9;

/// `mc[fi]` / `mct[fi]` dizini → [`Filter2d`].
const fn filtre(fi: usize) -> Filter2d {
    match Filter2d::from_index(fi) {
        Some(f) => f,
        None => Filter2d::Bilinear,
    }
}

/// dav1d `GET_H_FILTER` / `GET_V_FILTER`: boyut ≤ 4'te 4-tap tablosu.
#[inline]
fn katsayi(tablo: usize, boyut: usize, m: u8) -> [i16; 8] {
    let t = if boyut > 4 { tablo } else { 3 + (tablo & 1) };
    MC_SUBPEL_FILTERS[t][usize::from(m) - 1].map(i16::from)
}

/// Yuvanın bit derinliği ile çağrının uyumu (10 ve 12 bit farklı kaydırma kullanır).
#[inline]
fn bd_uyumlu<const BPC: u32>(bd: BitDepth, bdmax: u16) -> bool {
    bd.bits() == BPC && u32::from(bdmax) == (1u32 << BPC) - 1
}

// ---------------------------------------------------------------------------
// Yükleme / saklama yardımcıları
// ---------------------------------------------------------------------------

/// 8 baytı `i16`'ya genişleterek okur (8 bpc pikseller).
#[target_feature(enable = "neon")]
#[inline]
fn oku_p8(s: &[u8], off: usize) -> int16x8_t {
    vreinterpretq_s16_u16(vmovl_u8(load_u8x8(chunk(s, off))))
}

/// 8 adet 16 bpc pikseli (16 bayt, yerel bayt sırası) `i16` olarak okur (değerler ≤ 4095).
#[target_feature(enable = "neon")]
#[inline]
fn oku_p16(s: &[u8], off: usize) -> int16x8_t {
    vreinterpretq_s16_u8(load_u8x16(chunk(s, off)))
}

/// 8 adet 16 bpc pikseli `u16` olarak okur.
#[target_feature(enable = "neon")]
#[inline]
fn oku_u16(s: &[u8], off: usize) -> uint16x8_t {
    vreinterpretq_u16_u8(load_u8x16(chunk(s, off)))
}

/// `n ≤ 8` adet 8 bpc pikseli yazar (`_memcpy` çağrısız sabit boyutlu kollar).
#[target_feature(enable = "neon")]
#[inline]
fn yaz_u8(dst: &mut [u8], off: usize, v: uint8x8_t, n: usize) {
    if n >= 8 {
        store_u8x8(chunk_mut(dst, off), v);
    } else {
        let w = vget_lane_u64::<0>(vreinterpret_u64_u8(v));
        match n {
            6 => {
                *chunk_mut::<u8, 4>(dst, off) = (w as u32).to_ne_bytes();
                *chunk_mut::<u8, 2>(dst, off + 4) = ((w >> 32) as u16).to_ne_bytes();
            }
            4 => {
                *chunk_mut::<u8, 4>(dst, off) = (w as u32).to_ne_bytes();
            }
            3 => {
                *chunk_mut::<u8, 2>(dst, off) = (w as u16).to_ne_bytes();
                dst[off + 2] = (w >> 16) as u8;
            }
            2 => {
                *chunk_mut::<u8, 2>(dst, off) = (w as u16).to_ne_bytes();
            }
            1 => {
                dst[off] = w as u8;
            }
            _ => {
                let t = w.to_ne_bytes();
                dst[off..off + n].copy_from_slice(&t[..n]);
            }
        }
    }
}

/// `n ≤ 8` adet 16 bpc pikseli (`2n` bayt) yazar (`_memcpy` çağrısız sabit boyutlu kollar).
#[target_feature(enable = "neon")]
#[inline]
fn yaz_u16(dst: &mut [u8], off: usize, v: uint16x8_t, n: usize) {
    if n >= 8 {
        store_u8x16(chunk_mut(dst, off), vreinterpretq_u8_u16(v));
    } else {
        let lo = vget_lane_u64::<0>(vreinterpret_u64_u16(vget_low_u16(v)));
        match n {
            6 => {
                let hi = vget_lane_u64::<0>(vreinterpret_u64_u16(vget_high_u16(v)));
                *chunk_mut::<u8, 8>(dst, off) = lo.to_ne_bytes();
                *chunk_mut::<u8, 4>(dst, off + 8) = (hi as u32).to_ne_bytes();
            }
            4 => {
                *chunk_mut::<u8, 8>(dst, off) = lo.to_ne_bytes();
            }
            3 => {
                *chunk_mut::<u8, 4>(dst, off) = (lo as u32).to_ne_bytes();
                *chunk_mut::<u8, 2>(dst, off + 4) = ((lo >> 32) as u16).to_ne_bytes();
            }
            2 => {
                *chunk_mut::<u8, 4>(dst, off) = (lo as u32).to_ne_bytes();
            }
            1 => {
                *chunk_mut::<u8, 2>(dst, off) = (lo as u16).to_ne_bytes();
            }
            _ => {
                let mut t = [0u8; 16];
                store_u8x16(&mut t, vreinterpretq_u8_u16(v));
                dst[off..off + 2 * n].copy_from_slice(&t[..2 * n]);
            }
        }
    }
}

/// `n ≤ 8` adet `i16` ara değeri yazar.
#[target_feature(enable = "neon")]
#[inline]
fn yaz_i16(tmp: &mut [i16], off: usize, v: int16x8_t, n: usize) {
    if n >= 8 {
        store_i16x8(chunk_mut(tmp, off), v);
    } else if n == 4 {
        let lo = vget_lane_u64::<0>(vreinterpret_u64_u16(vreinterpret_u16_s16(vget_low_s16(v))));
        let d = chunk_mut::<i16, 4>(tmp, off);
        d[0] = lo as i16;
        d[1] = (lo >> 16) as i16;
        d[2] = (lo >> 32) as i16;
        d[3] = (lo >> 48) as i16;
    } else {
        let mut t = [0i16; 8];
        store_i16x8(&mut t, v);
        tmp[off..off + n].copy_from_slice(&t[..n]);
    }
}

/// `n ≤ 8` öğeyi sıfır dolgulu okur (`_memcpy` çağrısız; dar bloklarda okuma `n`'de kalır).
#[target_feature(enable = "neon")]
#[inline]
fn oku_kismi_u8(s: &[u8], off: usize, n: usize) -> uint8x8_t {
    if n >= 8 {
        load_u8x8(chunk(s, off))
    } else {
        let w = match n {
            6 => {
                let a = u32::from_ne_bytes(*chunk::<u8, 4>(s, off));
                let b = u16::from_ne_bytes(*chunk::<u8, 2>(s, off + 4));
                u64::from(a) | (u64::from(b) << 32)
            }
            4 => u64::from(u32::from_ne_bytes(*chunk::<u8, 4>(s, off))),
            3 => {
                let a = u16::from_ne_bytes(*chunk::<u8, 2>(s, off));
                u64::from(a) | (u64::from(s[off + 2]) << 16)
            }
            2 => u64::from(u16::from_ne_bytes(*chunk::<u8, 2>(s, off))),
            1 => u64::from(s[off]),
            _ => {
                let mut t = [0u8; 8];
                t[..n].copy_from_slice(&s[off..off + n]);
                u64::from_ne_bytes(t)
            }
        };
        load_u8x8(&w.to_ne_bytes())
    }
}

/// `n ≤ 8` adet 16 bpc pikseli sıfır dolgulu okur (`_memcpy` çağrısız).
#[target_feature(enable = "neon")]
#[inline]
fn oku_kismi_u16(s: &[u8], off: usize, n: usize) -> uint16x8_t {
    if n >= 8 {
        oku_u16(s, off)
    } else {
        match n {
            6 => {
                let lo = u64::from_ne_bytes(*chunk::<u8, 8>(s, off));
                let hi = u64::from(u32::from_ne_bytes(*chunk::<u8, 4>(s, off + 8)));
                vcombine_u16(vcreate_u16(lo), vcreate_u16(hi))
            }
            4 => {
                let lo = u64::from_ne_bytes(*chunk::<u8, 8>(s, off));
                vcombine_u16(vcreate_u16(lo), vdup_n_u16(0))
            }
            3 => {
                let a = u32::from_ne_bytes(*chunk::<u8, 4>(s, off));
                let b = u16::from_ne_bytes(*chunk::<u8, 2>(s, off + 4));
                vcombine_u16(
                    vcreate_u16(u64::from(a) | (u64::from(b) << 32)),
                    vdup_n_u16(0),
                )
            }
            2 => {
                let lo = u64::from(u32::from_ne_bytes(*chunk::<u8, 4>(s, off)));
                vcombine_u16(vcreate_u16(lo), vdup_n_u16(0))
            }
            1 => {
                let lo = u64::from(u16::from_ne_bytes(*chunk::<u8, 2>(s, off)));
                vcombine_u16(vcreate_u16(lo), vdup_n_u16(0))
            }
            _ => {
                let mut t = [0u8; 16];
                t[..2 * n].copy_from_slice(&s[off..off + 2 * n]);
                vreinterpretq_u16_u8(load_u8x16(&t))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 8-tap filtre yardımcıları
// ---------------------------------------------------------------------------

macro_rules! yatay_adim8 {
    ($t:ident, $a:ident, $b:ident, $f:ident, $($k:literal)*) => {
        $( $t = vmlaq_n_s16($t, vextq_s16::<$k>($a, $b), $f[$k]); )*
    };
}

/// 8 bpc yatay 8-tap: `off` 0. çıktının 0. tap'idir; 8 çıktı (`i16`, taşmasız).
#[target_feature(enable = "neon")]
#[inline]
fn yatay8(s: &[u8], off: usize, f: &[i16; 8]) -> int16x8_t {
    let v = load_u8x16(chunk(s, off));
    let a = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v)));
    let b = vreinterpretq_s16_u16(vmovl_high_u8(v));
    let mut t = vmulq_n_s16(a, f[0]);
    yatay_adim8!(t, a, b, f, 1 2 3 4 5 6 7);
    t
}

macro_rules! yatay_adim16 {
    ($lo:ident, $hi:ident, $a:ident, $b:ident, $f:ident, $($k:literal)*) => {
        $(
            let e = vextq_s16::<$k>($a, $b);
            $lo = vmlal_n_s16($lo, vget_low_s16(e), $f[$k]);
            $hi = vmlal_high_n_s16($hi, e, $f[$k]);
        )*
    };
}

/// 10/12 bpc yatay 8-tap: 8 çıktı, `i32` (alt 4, üst 4).
#[target_feature(enable = "neon")]
#[inline]
fn yatay16(s: &[u8], off: usize, f: &[i16; 8]) -> (int32x4_t, int32x4_t) {
    let a = oku_p16(s, off);
    let b = oku_p16(s, off + 16);
    let mut lo = vmull_n_s16(vget_low_s16(a), f[0]);
    let mut hi = vmull_high_n_s16(a, f[0]);
    yatay_adim16!(lo, hi, a, b, f, 1 2 3 4 5 6 7);
    (lo, hi)
}

/// Dikey 8-tap (`i16` toplam; 8 bpc piksel penceresi).
#[target_feature(enable = "neon")]
#[inline]
fn dikey_i16(p: &[int16x8_t; 8], f: &[i16; 8]) -> int16x8_t {
    let mut t = vmulq_n_s16(p[0], f[0]);
    for k in 1..8 {
        t = vmlaq_n_s16(t, p[k], f[k]);
    }
    t
}

/// Dikey 8-tap (`i32` toplam; `i16` ara satır ya da 16 bpc piksel penceresi).
#[target_feature(enable = "neon")]
#[inline]
fn dikey_i32(p: &[int16x8_t; 8], f: &[i16; 8]) -> (int32x4_t, int32x4_t) {
    let mut lo = vmull_n_s16(vget_low_s16(p[0]), f[0]);
    let mut hi = vmull_high_n_s16(p[0], f[0]);
    for k in 1..8 {
        lo = vmlal_n_s16(lo, vget_low_s16(p[k]), f[k]);
        hi = vmlal_high_n_s16(hi, p[k], f[k]);
    }
    (lo, hi)
}

/// Pencereyi bir satır kaydırır (`p[0..7] ← p[1..8]`).
#[inline]
fn kaydir(p: &mut [int16x8_t; 8]) {
    p.copy_within(1.., 0);
}

/// `put` kopyası (`put_c`): `h` satır × `w·bpp` bayt.
#[inline]
fn kopya(dst: &mut [u8], ds: usize, src: &[u8], ss: usize, w: usize, h: usize, bpp: usize) {
    let n = w * bpp;
    for y in 0..h {
        let s = (y + 3) * ss + 3 * bpp;
        let d = y * ds;
        dst[d..d + n].copy_from_slice(&src[s..s + n]);
    }
}

/// `i32` alt/üst yarıyı 10/12 bpc piksele kırpar (`sqxtun` + `umin`).
#[target_feature(enable = "neon")]
#[inline]
fn kirp16(lo: int32x4_t, hi: int32x4_t, maxv: uint16x8_t) -> uint16x8_t {
    vminq_u16(vcombine_u16(vqmovun_s32(lo), vqmovun_s32(hi)), maxv)
}

/// `i32` alt/üst yarıyı `i16`'ya daraltır (`xtn`, değer aralığı sözleşmeli).
#[target_feature(enable = "neon")]
#[inline]
fn dar16(lo: int32x4_t, hi: int32x4_t) -> int16x8_t {
    vcombine_s16(vmovn_s32(lo), vmovn_s32(hi))
}

// ---------------------------------------------------------------------------
// DOTPROD yatay 8-tap (8 bpc)
// ---------------------------------------------------------------------------

/// `sdot` dizin tabloları: çıktı `j` için tap `0..3` / `4..7` baytları.
const DOT_IDX: [[u8; 16]; 3] = [
    [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6],
    [4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10],
    [8, 9, 10, 11, 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14],
];

/// DOTPROD yatay katsayı/dizin kümesi.
#[derive(Clone, Copy)]
struct DotFiltre {
    lo: int8x16_t,
    hi: int8x16_t,
    ia: uint8x16_t,
    ib: uint8x16_t,
    ic: uint8x16_t,
}

#[target_feature(enable = "neon")]
#[inline]
fn dot_filtre(f: &[i16; 8]) -> DotFiltre {
    let mut lo = [0u8; 16];
    let mut hi = [0u8; 16];
    for j in 0..4 {
        for k in 0..4 {
            lo[4 * j + k] = f[k] as i8 as u8;
            hi[4 * j + k] = f[4 + k] as i8 as u8;
        }
    }
    DotFiltre {
        lo: vreinterpretq_s8_u8(load_u8x16(&lo)),
        hi: vreinterpretq_s8_u8(load_u8x16(&hi)),
        ia: load_u8x16(&DOT_IDX[0]),
        ib: load_u8x16(&DOT_IDX[1]),
        ic: load_u8x16(&DOT_IDX[2]),
    }
}

/// 8 bpc yatay 8-tap, `sdot` ile (sonuç [`yatay8`] ile bit-birebir aynı).
#[target_feature(enable = "neon,dotprod")]
#[inline]
fn yatay8_dot(s: &[u8], off: usize, d: &DotFiltre) -> int16x8_t {
    let v = vreinterpretq_s8_u8(veorq_u8(load_u8x16(chunk(s, off)), vdupq_n_u8(0x80)));
    let pa = vqtbl1q_s8(v, d.ia);
    let pb = vqtbl1q_s8(v, d.ib);
    let pc = vqtbl1q_s8(v, d.ic);
    let sapma = vdupq_n_s32(8192);
    let lo = vdotq_s32(vdotq_s32(sapma, pa, d.lo), pb, d.hi);
    let hi = vdotq_s32(vdotq_s32(sapma, pb, d.lo), pc, d.hi);
    dar16(lo, hi)
}

// ---------------------------------------------------------------------------
// 8 bpc `put` / `prep` (8-tap)
// ---------------------------------------------------------------------------

/// 8 bpc 8-tap `put`/`prep` DOTPROD gövdesi (`PREP`: `tmp`'ye): yatay geçiş `sdot`,
/// dikey geçiş NEON `smlal` (dav1d `mc_dotprod.S` ile aynı iş bölümü).
#[target_feature(enable = "neon,dotprod")]
fn sekiz_tap8_dot<const PREP: bool>(
    dst: &mut [u8],
    ds: usize,
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    filt: Filter2d,
) {
    if mx == 0 {
        return sekiz_tap8_dikey_kopya::<PREP>(dst, ds, tmp, src, ss, w, h, my, filt);
    }
    let d = dot_filtre(&katsayi(filt.horizontal_table(), w, mx));
    if my == 0 {
        for y in 0..h {
            let so = (y + 3) * ss;
            let mut x = 0;
            while x < w {
                let t = yatay8_dot(src, so + x, &d);
                if PREP {
                    yaz_i16(tmp, y * w + x, vrshrq_n_s16::<2>(t), w - x);
                } else {
                    let r = vqrshrun_n_s16::<6>(vaddq_s16(t, vdupq_n_s16(2)));
                    yaz_u8(dst, y * ds + x, r, w - x);
                }
                x += 8;
            }
        }
        return;
    }
    let fv = katsayi(filt.vertical_table(), h, my);
    let mut x = 0;
    while x < w {
        let mut p = [vdupq_n_s16(0); 8];
        for (k, pk) in p.iter_mut().take(7).enumerate() {
            *pk = vrshrq_n_s16::<2>(yatay8_dot(src, k * ss + x, &d));
        }
        for y in 0..h {
            p[7] = vrshrq_n_s16::<2>(yatay8_dot(src, (y + 7) * ss + x, &d));
            let (lo, hi) = dikey_i32(&p, &fv);
            if PREP {
                let t = vcombine_s16(vrshrn_n_s32::<6>(lo), vrshrn_n_s32::<6>(hi));
                yaz_i16(tmp, y * w + x, t, w - x);
            } else {
                let u = vcombine_u16(vqrshrun_n_s32::<10>(lo), vqrshrun_n_s32::<10>(hi));
                yaz_u8(dst, y * ds + x, vqmovn_u16(u), w - x);
            }
            kaydir(&mut p);
        }
        x += 8;
    }
}

/// 8 bpc yalnız-dikey ve kopya yolları (DOTPROD ve NEON ortak; `sdot` kullanmaz).
#[target_feature(enable = "neon")]
fn sekiz_tap8_dikey_kopya<const PREP: bool>(
    dst: &mut [u8],
    ds: usize,
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    my: u8,
    filt: Filter2d,
) {
    if my == 0 {
        if PREP {
            for y in 0..h {
                let so = (y + 3) * ss + 3;
                let mut x = 0;
                while x < w {
                    let p = vshlq_n_s16::<4>(oku_p8(src, so + x));
                    yaz_i16(tmp, y * w + x, p, w - x);
                    x += 8;
                }
            }
        } else {
            kopya(dst, ds, src, ss, w, h, 1);
        }
        return;
    }
    let f = katsayi(filt.vertical_table(), h, my);
    let mut x = 0;
    while x < w {
        let mut p = [vdupq_n_s16(0); 8];
        for (k, pk) in p.iter_mut().take(7).enumerate() {
            *pk = oku_p8(src, k * ss + x + 3);
        }
        for y in 0..h {
            p[7] = oku_p8(src, (y + 7) * ss + x + 3);
            let t = dikey_i16(&p, &f);
            if PREP {
                yaz_i16(tmp, y * w + x, vrshrq_n_s16::<2>(t), w - x);
            } else {
                yaz_u8(dst, y * ds + x, vqrshrun_n_s16::<6>(t), w - x);
            }
            kaydir(&mut p);
        }
        x += 8;
    }
}

/// 8 bpc 8-tap gövdesi (yalnız NEON).
#[target_feature(enable = "neon")]
fn sekiz_tap8<const PREP: bool>(
    dst: &mut [u8],
    ds: usize,
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    filt: Filter2d,
) {
    if mx == 0 {
        return sekiz_tap8_dikey_kopya::<PREP>(dst, ds, tmp, src, ss, w, h, my, filt);
    }
    let fh = katsayi(filt.horizontal_table(), w, mx);
    if my == 0 {
        for y in 0..h {
            let so = (y + 3) * ss;
            let mut x = 0;
            while x < w {
                let t = yatay8(src, so + x, &fh);
                if PREP {
                    yaz_i16(tmp, y * w + x, vrshrq_n_s16::<2>(t), w - x);
                } else {
                    let r = vqrshrun_n_s16::<6>(vaddq_s16(t, vdupq_n_s16(2)));
                    yaz_u8(dst, y * ds + x, r, w - x);
                }
                x += 8;
            }
        }
        return;
    }
    let fv = katsayi(filt.vertical_table(), h, my);
    let mut x = 0;
    while x < w {
        let mut p = [vdupq_n_s16(0); 8];
        for (k, pk) in p.iter_mut().take(7).enumerate() {
            *pk = vrshrq_n_s16::<2>(yatay8(src, k * ss + x, &fh));
        }
        for y in 0..h {
            p[7] = vrshrq_n_s16::<2>(yatay8(src, (y + 7) * ss + x, &fh));
            let (lo, hi) = dikey_i32(&p, &fv);
            if PREP {
                let t = vcombine_s16(vrshrn_n_s32::<6>(lo), vrshrn_n_s32::<6>(hi));
                yaz_i16(tmp, y * w + x, t, w - x);
            } else {
                let u = vcombine_u16(vqrshrun_n_s32::<10>(lo), vqrshrun_n_s32::<10>(hi));
                yaz_u8(dst, y * ds + x, vqmovn_u16(u), w - x);
            }
            kaydir(&mut p);
        }
        x += 8;
    }
}

// ---------------------------------------------------------------------------
// 8 bpc bilinear
// ---------------------------------------------------------------------------

/// `(16 − m)·a + m·b` (8 piksel, `u16`; `a` = `off`, `b` = `off + 1`).
#[target_feature(enable = "neon")]
#[inline]
fn bilin_yatay8(s: &[u8], off: usize, m: u8) -> uint16x8_t {
    let v = load_u8x16(chunk(s, off));
    let a = vget_low_u8(v);
    let b = vget_low_u8(vextq_u8::<1>(v, v));
    vmlal_u8(vmull_u8(a, vdup_n_u8(16 - m)), b, vdup_n_u8(m))
}

/// `(16 − m)·a + m·b` iki satır (8 bpc, `u16`).
#[target_feature(enable = "neon")]
#[inline]
fn bilin_dikey8(s: &[u8], off: usize, ss: usize, m: u8) -> uint16x8_t {
    let a = load_u8x8(chunk(s, off));
    let b = load_u8x8(chunk(s, off + ss));
    vmlal_u8(vmull_u8(a, vdup_n_u8(16 - m)), b, vdup_n_u8(m))
}

/// 8 bpc bilinear `put`/`prep` (dav1d `put_bilin_c` / `prep_bilin_c`).
#[target_feature(enable = "neon")]
fn bilin8<const PREP: bool>(
    dst: &mut [u8],
    ds: usize,
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    match (mx != 0, my != 0) {
        (false, false) => {
            sekiz_tap8_dikey_kopya::<PREP>(dst, ds, tmp, src, ss, w, h, 0, Filter2d::Regular)
        }
        (true, false) => {
            for y in 0..h {
                let so = (y + 3) * ss + 3;
                let mut x = 0;
                while x < w {
                    let v = bilin_yatay8(src, so + x, mx);
                    if PREP {
                        yaz_i16(tmp, y * w + x, vreinterpretq_s16_u16(v), w - x);
                    } else {
                        yaz_u8(dst, y * ds + x, vrshrn_n_u16::<4>(v), w - x);
                    }
                    x += 8;
                }
            }
        }
        (false, true) => {
            for y in 0..h {
                let so = (y + 3) * ss + 3;
                let mut x = 0;
                while x < w {
                    let v = bilin_dikey8(src, so + x, ss, my);
                    if PREP {
                        yaz_i16(tmp, y * w + x, vreinterpretq_s16_u16(v), w - x);
                    } else {
                        yaz_u8(dst, y * ds + x, vrshrn_n_u16::<4>(v), w - x);
                    }
                    x += 8;
                }
            }
        }
        (true, true) => {
            let (a, b) = (u16::from(16 - my), u16::from(my));
            let mut x = 0;
            while x < w {
                let mut m0 = bilin_yatay8(src, 3 * ss + 3 + x, mx);
                for y in 0..h {
                    let m1 = bilin_yatay8(src, (y + 4) * ss + 3 + x, mx);
                    let v = vmlaq_n_u16(vmulq_n_u16(m0, a), m1, b);
                    if PREP {
                        yaz_i16(
                            tmp,
                            y * w + x,
                            vreinterpretq_s16_u16(vrshrq_n_u16::<4>(v)),
                            w - x,
                        );
                    } else {
                        yaz_u8(dst, y * ds + x, vmovn_u16(vrshrq_n_u16::<8>(v)), w - x);
                    }
                    m0 = m1;
                }
                x += 8;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 10/12 bpc `put` / `prep` (8-tap + bilinear)
// ---------------------------------------------------------------------------

/// 10/12 bpc 8-tap `put`/`prep` gövdesi.
#[target_feature(enable = "neon")]
fn sekiz_tap16<const BPC: u32, const PREP: bool>(
    dst: &mut [u8],
    ds: usize,
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    filt: Filter2d,
) {
    let sh = BPC as i32 - 8; // 6 − intermediate_bits
    let ib = 14 - BPC as i32;
    let maxv = vdupq_n_u16(((1u32 << BPC) - 1) as u16);
    let bias = vdupq_n_s32(8192);
    let nsh = vdupq_n_s32(-sh);
    match (mx != 0, my != 0) {
        (false, false) => {
            if PREP {
                let ibv = vdupq_n_s16(ib as i16);
                let b16 = vdupq_n_s16(8192);
                for y in 0..h {
                    let so = (y + 3) * ss + 6;
                    let mut x = 0;
                    while x < w {
                        let p = oku_p16(src, so + 2 * x);
                        yaz_i16(tmp, y * w + x, vsubq_s16(vshlq_s16(p, ibv), b16), w - x);
                        x += 8;
                    }
                }
            } else {
                kopya(dst, ds, src, ss, w, h, 2);
            }
        }
        (true, false) => {
            let f = katsayi(filt.horizontal_table(), w, mx);
            let c = vdupq_n_s32(32 + ((1 << sh) >> 1));
            for y in 0..h {
                let so = (y + 3) * ss;
                let mut x = 0;
                while x < w {
                    let (lo, hi) = yatay16(src, so + 2 * x, &f);
                    if PREP {
                        let lo = vsubq_s32(vrshlq_s32(lo, nsh), bias);
                        let hi = vsubq_s32(vrshlq_s32(hi, nsh), bias);
                        yaz_i16(tmp, y * w + x, dar16(lo, hi), w - x);
                    } else {
                        let lo = vshrq_n_s32::<6>(vaddq_s32(lo, c));
                        let hi = vshrq_n_s32::<6>(vaddq_s32(hi, c));
                        yaz_u16(dst, y * ds + 2 * x, kirp16(lo, hi, maxv), w - x);
                    }
                    x += 8;
                }
            }
        }
        (false, true) => {
            let f = katsayi(filt.vertical_table(), h, my);
            let mut x = 0;
            while x < w {
                let mut p = [vdupq_n_s16(0); 8];
                for (k, pk) in p.iter_mut().take(7).enumerate() {
                    *pk = oku_p16(src, k * ss + 2 * (x + 3));
                }
                for y in 0..h {
                    p[7] = oku_p16(src, (y + 7) * ss + 2 * (x + 3));
                    let (lo, hi) = dikey_i32(&p, &f);
                    if PREP {
                        let lo = vsubq_s32(vrshlq_s32(lo, nsh), bias);
                        let hi = vsubq_s32(vrshlq_s32(hi, nsh), bias);
                        yaz_i16(tmp, y * w + x, dar16(lo, hi), w - x);
                    } else {
                        let v = kirp16(vrshrq_n_s32::<6>(lo), vrshrq_n_s32::<6>(hi), maxv);
                        yaz_u16(dst, y * ds + 2 * x, v, w - x);
                    }
                    kaydir(&mut p);
                }
                x += 8;
            }
        }
        (true, true) => {
            let fh = katsayi(filt.horizontal_table(), w, mx);
            let fv = katsayi(filt.vertical_table(), h, my);
            let nsv = vdupq_n_s32(-(6 + ib));
            let mut x = 0;
            while x < w {
                let mut p = [vdupq_n_s16(0); 8];
                for (k, pk) in p.iter_mut().take(7).enumerate() {
                    let (lo, hi) = yatay16(src, k * ss + 2 * x, &fh);
                    *pk = dar16(vrshlq_s32(lo, nsh), vrshlq_s32(hi, nsh));
                }
                for y in 0..h {
                    let (lo, hi) = yatay16(src, (y + 7) * ss + 2 * x, &fh);
                    p[7] = dar16(vrshlq_s32(lo, nsh), vrshlq_s32(hi, nsh));
                    let (lo, hi) = dikey_i32(&p, &fv);
                    if PREP {
                        let lo = vsubq_s32(vrshrq_n_s32::<6>(lo), bias);
                        let hi = vsubq_s32(vrshrq_n_s32::<6>(hi), bias);
                        yaz_i16(tmp, y * w + x, dar16(lo, hi), w - x);
                    } else {
                        let v = kirp16(vrshlq_s32(lo, nsv), vrshlq_s32(hi, nsv), maxv);
                        yaz_u16(dst, y * ds + 2 * x, v, w - x);
                    }
                    kaydir(&mut p);
                }
                x += 8;
            }
        }
    }
}

/// `(16 − m)·a + m·b` (8 adet 16 bpc piksel, `u16`; `16·4095 < 2¹⁶`).
#[target_feature(enable = "neon")]
#[inline]
fn bilin_yatay16(s: &[u8], off: usize, m: u8) -> uint16x8_t {
    let a = oku_u16(s, off);
    let b = vextq_u16::<1>(a, oku_u16(s, off + 16));
    vmlaq_n_u16(vmulq_n_u16(a, u16::from(16 - m)), b, u16::from(m))
}

/// 10/12 bpc bilinear `put`/`prep`.
#[target_feature(enable = "neon")]
fn bilin16<const BPC: u32, const PREP: bool>(
    dst: &mut [u8],
    ds: usize,
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    let ib = 14 - BPC as i16;
    let hs = vdupq_n_s16(-(4 - ib)); // ara kaydırma `4 − ib`
    let maxv = vdupq_n_u16(((1u32 << BPC) - 1) as u16);
    let b16 = vdupq_n_s16(8192);
    match (mx != 0, my != 0) {
        (false, false) => {
            sekiz_tap16::<BPC, PREP>(dst, ds, tmp, src, ss, w, h, 0, 0, Filter2d::Regular)
        }
        (true, false) => {
            let ibn = vdupq_n_s16(-ib);
            for y in 0..h {
                let so = (y + 3) * ss + 6;
                let mut x = 0;
                while x < w {
                    let px = vrshlq_u16(bilin_yatay16(src, so + 2 * x, mx), hs);
                    if PREP {
                        yaz_i16(
                            tmp,
                            y * w + x,
                            vsubq_s16(vreinterpretq_s16_u16(px), b16),
                            w - x,
                        );
                    } else {
                        yaz_u16(
                            dst,
                            y * ds + 2 * x,
                            vminq_u16(vrshlq_u16(px, ibn), maxv),
                            w - x,
                        );
                    }
                    x += 8;
                }
            }
        }
        (false, true) => {
            let (a, b) = (u16::from(16 - my), u16::from(my));
            for y in 0..h {
                let so = (y + 3) * ss + 6;
                let mut x = 0;
                while x < w {
                    let p0 = oku_u16(src, so + 2 * x);
                    let p1 = oku_u16(src, so + ss + 2 * x);
                    let v = vmlaq_n_u16(vmulq_n_u16(p0, a), p1, b);
                    if PREP {
                        let t = vreinterpretq_s16_u16(vrshlq_u16(v, hs));
                        yaz_i16(tmp, y * w + x, vsubq_s16(t, b16), w - x);
                    } else {
                        yaz_u16(
                            dst,
                            y * ds + 2 * x,
                            vminq_u16(vrshrq_n_u16::<4>(v), maxv),
                            w - x,
                        );
                    }
                    x += 8;
                }
            }
        }
        (true, true) => {
            let (a, b) = (u16::from(16 - my), u16::from(my));
            let nsv = vdupq_n_s32(-(4 + i32::from(ib)));
            let bias = vdupq_n_s32(8192);
            let mut x = 0;
            while x < w {
                let mut m0 = vrshlq_u16(bilin_yatay16(src, 3 * ss + 6 + 2 * x, mx), hs);
                for y in 0..h {
                    let m1 = vrshlq_u16(bilin_yatay16(src, (y + 4) * ss + 6 + 2 * x, mx), hs);
                    let lo = vmlal_n_u16(vmull_n_u16(vget_low_u16(m0), a), vget_low_u16(m1), b);
                    let hi = vmlal_high_n_u16(vmull_high_n_u16(m0, a), m1, b);
                    if PREP {
                        let lo = vsubq_s32(vreinterpretq_s32_u32(vrshrq_n_u32::<4>(lo)), bias);
                        let hi = vsubq_s32(vreinterpretq_s32_u32(vrshrq_n_u32::<4>(hi)), bias);
                        yaz_i16(tmp, y * w + x, dar16(lo, hi), w - x);
                    } else {
                        let v = vcombine_u16(
                            vqmovn_u32(vrshlq_u32(lo, nsv)),
                            vqmovn_u32(vrshlq_u32(hi, nsv)),
                        );
                        yaz_u16(dst, y * ds + 2 * x, vminq_u16(v, maxv), w - x);
                    }
                    m0 = m1;
                }
                x += 8;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// `put` / `prep` girişleri (köprüler [`giris`] içinde)
// ---------------------------------------------------------------------------

/// `put` sözleşmesi (8 / 10 / 12 bpc).
#[inline]
pub(super) fn put_kabul<const BPC: u32>(
    dst_len: usize,
    ds: usize,
    src_len: usize,
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bd: BitDepth,
    bdmax: u16,
) -> bool {
    let bpp = if BPC == 8 { 1 } else { 2 };
    bd_uyumlu::<BPC>(bd, bdmax)
        && h >= 1
        && ds >= w * bpp
        && if BPC == 8 {
            simd_sozlesmesi(src_len, ss, w, h, mx, my, dst_len, (h - 1) * ds + w)
        } else {
            simd16_sozlesmesi(
                src_len,
                ss,
                w,
                h,
                mx,
                my,
                bd,
                bdmax,
                dst_len,
                (h - 1) * ds + 2 * w,
            )
        }
}

/// `prep` sözleşmesi (8 / 10 / 12 bpc).
#[inline]
pub(super) fn prep_kabul<const BPC: u32>(
    tmp_len: usize,
    src_len: usize,
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
    bd: BitDepth,
    bdmax: u16,
) -> bool {
    bd_uyumlu::<BPC>(bd, bdmax)
        && if BPC == 8 {
            simd_sozlesmesi(src_len, ss, w, h, mx, my, tmp_len, w * h)
        } else {
            simd16_sozlesmesi(src_len, ss, w, h, mx, my, bd, bdmax, tmp_len, w * h)
        }
}

/// Sözleşme dışı `put`: skaler devir.
#[cold]
pub(super) fn put_skaler<const FI: usize>(
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
    skaler_devri_say();
    (McDsp::scalar_for(bd).mc[FI])(dst, ds, src, ss, w, h, mx, my, bd, bdmax);
}

/// Sözleşme dışı `prep`: skaler devir.
#[cold]
pub(super) fn prep_skaler<const FI: usize>(
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
    skaler_devri_say();
    (McDsp::scalar_for(bd).mct[FI])(tmp, src, ss, w, h, mx, my, bd, bdmax);
}

/// NEON `put` gövdesi (sözleşme girişte denetlendi).
#[target_feature(enable = "neon")]
pub(super) fn put_neon<const BPC: u32, const FI: usize>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    let mut bos: [i16; 0] = [];
    if BPC == 8 {
        if FI == BILIN {
            bilin8::<false>(dst, ds, &mut bos, src, ss, w, h, mx, my);
        } else {
            sekiz_tap8::<false>(dst, ds, &mut bos, src, ss, w, h, mx, my, filtre(FI));
        }
    } else if FI == BILIN {
        bilin16::<BPC, false>(dst, ds, &mut bos, src, ss, w, h, mx, my);
    } else {
        sekiz_tap16::<BPC, false>(dst, ds, &mut bos, src, ss, w, h, mx, my, filtre(FI));
    }
}

/// NEON `prep` gövdesi (sözleşme girişte denetlendi).
#[target_feature(enable = "neon")]
pub(super) fn prep_neon<const BPC: u32, const FI: usize>(
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    let mut bos: [u8; 0] = [];
    if BPC == 8 {
        if FI == BILIN {
            bilin8::<true>(&mut bos, 0, tmp, src, ss, w, h, mx, my);
        } else {
            sekiz_tap8::<true>(&mut bos, 0, tmp, src, ss, w, h, mx, my, filtre(FI));
        }
    } else if FI == BILIN {
        bilin16::<BPC, true>(&mut bos, 0, tmp, src, ss, w, h, mx, my);
    } else {
        sekiz_tap16::<BPC, true>(&mut bos, 0, tmp, src, ss, w, h, mx, my, filtre(FI));
    }
}

/// DOTPROD `put` gövdesi (8 bpc, 8-tap; sözleşme girişte denetlendi).
#[target_feature(enable = "neon,dotprod")]
pub(super) fn put_dot<const FI: usize>(
    dst: &mut [u8],
    ds: usize,
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    let mut bos: [i16; 0] = [];
    sekiz_tap8_dot::<false>(dst, ds, &mut bos, src, ss, w, h, mx, my, filtre(FI));
}

/// DOTPROD `prep` gövdesi (8 bpc, 8-tap; sözleşme girişte denetlendi).
#[target_feature(enable = "neon,dotprod")]
pub(super) fn prep_dot<const FI: usize>(
    tmp: &mut [i16],
    src: &[u8],
    ss: usize,
    w: usize,
    h: usize,
    mx: u8,
    my: u8,
) {
    let mut bos: [u8; 0] = [];
    sekiz_tap8_dot::<true>(&mut bos, 0, tmp, src, ss, w, h, mx, my, filtre(FI));
}

macro_rules! fi_tablosu {
    ($fi:expr, $f:ident, [$($g:ident),*]) => {
        match $fi {
            0 => giris::$f::<$($g,)* 0>,
            1 => giris::$f::<$($g,)* 1>,
            2 => giris::$f::<$($g,)* 2>,
            3 => giris::$f::<$($g,)* 3>,
            4 => giris::$f::<$($g,)* 4>,
            5 => giris::$f::<$($g,)* 5>,
            6 => giris::$f::<$($g,)* 6>,
            7 => giris::$f::<$($g,)* 7>,
            8 => giris::$f::<$($g,)* 8>,
            _ => giris::$f::<$($g,)* 9>,
        }
    };
}

/// `mc[fi]` NEON yuvası (`BPC` = 8 / 10 / 12); yalnız [`NeonToken`] ile.
#[must_use]
pub const fn put_fn<const BPC: u32>(_neon: NeonToken, fi: usize) -> McPutFn {
    fi_tablosu!(fi, mc_put, [BPC])
}

/// `mct[fi]` NEON yuvası; yalnız [`NeonToken`] ile.
#[must_use]
pub const fn prep_fn<const BPC: u32>(_neon: NeonToken, fi: usize) -> McPrepFn {
    fi_tablosu!(fi, mc_prep, [BPC])
}

/// `mc[fi]` 8 bpc DOTPROD yuvası (`fi < 9`); yalnız [`DotprodToken`] ile.
#[must_use]
pub const fn put_dot_fn(_dot: DotprodToken, fi: usize) -> McPutFn {
    fi_tablosu!(fi, mc_put_dot, [])
}

/// `mct[fi]` 8 bpc DOTPROD yuvası (`fi < 9`); yalnız [`DotprodToken`] ile.
#[must_use]
pub const fn prep_dot_fn(_dot: DotprodToken, fi: usize) -> McPrepFn {
    fi_tablosu!(fi, mc_prep_dot, [])
}

// ---------------------------------------------------------------------------
// Bileşik: `avg`, `w_avg`, `mask`, `w_mask`
// ---------------------------------------------------------------------------

/// Bileşik işlev türü (`avg` / `w_avg` / `mask`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Bilesik {
    Avg,
    WAvg,
    Mask,
}

/// 8 öğelik bileşik toplamı (`i32` alt/üst; yuvarlama sabiti ve kaydırmadan önce).
#[target_feature(enable = "neon")]
#[inline]
fn bilesik_toplam(
    tur: Bilesik,
    a: int16x8_t,
    b: int16x8_t,
    weight: i16,
    m: int16x8_t,
) -> (int32x4_t, int32x4_t) {
    match tur {
        Bilesik::Avg => (
            vaddl_s16(vget_low_s16(a), vget_low_s16(b)),
            vaddl_high_s16(a, b),
        ),
        Bilesik::WAvg => {
            let wb = 16 - weight;
            (
                vmlal_n_s16(vmull_n_s16(vget_low_s16(a), weight), vget_low_s16(b), wb),
                vmlal_high_n_s16(vmull_high_n_s16(a, weight), b, wb),
            )
        }
        Bilesik::Mask => {
            let mb = vsubq_s16(vdupq_n_s16(64), m);
            (
                vmlal_s16(
                    vmull_s16(vget_low_s16(a), vget_low_s16(m)),
                    vget_low_s16(b),
                    vget_low_s16(mb),
                ),
                vmlal_high_s16(vmull_high_s16(a, m), b, mb),
            )
        }
    }
}

/// `avg` / `w_avg` / `mask` gövdesi (sözleşme girişte denetlendi). `tmp` satırları
/// bitişiktir; `w ≥ 4` ve `w·h ≥ 16` olduğundan 8 öğelik doğrusal öbekler en çok iki
/// satıra yayılır (`w = 4`).
#[target_feature(enable = "neon")]
pub(super) fn bilesik_neon<const BPC: u32>(
    tur: Bilesik,
    dst: &mut [u8],
    ds: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    weight: i32,
    mask: &[u8],
) {
    let ib = if BPC == 8 { 4 } else { 14 - BPC as i32 };
    let (kaydirma, sapma) = match tur {
        Bilesik::Avg => (ib + 1, if BPC == 8 { 0 } else { 8192 * 2 }),
        Bilesik::WAvg => (ib + 4, if BPC == 8 { 0 } else { 8192 * 16 }),
        Bilesik::Mask => (ib + 6, if BPC == 8 { 0 } else { 8192 * 64 }),
    };
    let nk = vdupq_n_s32(-kaydirma);
    let sap = vdupq_n_s32(sapma);
    let maxv = vdupq_n_u16(((1u32 << BPC) - 1) as u16);
    let bpp = if BPC == 8 { 1 } else { 2 };
    let n = w * h;
    let mut i = 0;
    while i < n {
        let a = load_i16x8(chunk(tmp1, i));
        let b = load_i16x8(chunk(tmp2, i));
        let m = if tur == Bilesik::Mask {
            vreinterpretq_s16_u16(vmovl_u8(load_u8x8(chunk(mask, i))))
        } else {
            vdupq_n_s16(0)
        };
        let (lo, hi) = bilesik_toplam(tur, a, b, weight as i16, m);
        // `(x + sapma + (1 << (k−1))) >> k` = `srshl(x + sapma, −k)` (taşmasız).
        let lo = vrshlq_s32(vaddq_s32(lo, sap), nk);
        let hi = vrshlq_s32(vaddq_s32(hi, sap), nk);
        let px = kirp16(lo, hi, maxv);
        let (y, x) = (i / w, i % w);
        if BPC == 8 {
            let p8 = vmovn_u16(px);
            if w >= 8 {
                store_u8x8(chunk_mut(dst, y * ds + x), p8);
            } else {
                let mut t = [0u8; 8];
                store_u8x8(&mut t, p8);
                dst[y * ds..y * ds + 4].copy_from_slice(&t[..4]);
                dst[(y + 1) * ds..(y + 1) * ds + 4].copy_from_slice(&t[4..]);
            }
        } else if w >= 8 {
            yaz_u16(dst, y * ds + bpp * x, px, 8);
        } else {
            let mut t = [0u8; 16];
            store_u8x16(&mut t, vreinterpretq_u8_u16(px));
            dst[y * ds..y * ds + 8].copy_from_slice(&t[..8]);
            dst[(y + 1) * ds..(y + 1) * ds + 8].copy_from_slice(&t[8..]);
        }
        i += 8;
    }
}

/// `avg`/`w_avg`/`mask` sözleşmesi.
#[inline]
pub(super) fn bilesik_kabul<const BPC: u32>(
    tur: Bilesik,
    dst_len: usize,
    ds: usize,
    t1: usize,
    t2: usize,
    w: usize,
    h: usize,
    weight: i32,
    mask_len: usize,
    bd: BitDepth,
    bdmax: u16,
) -> bool {
    let ek = match tur {
        Bilesik::Avg => true,
        Bilesik::WAvg => (0..=16).contains(&weight),
        Bilesik::Mask => mask_len >= w * h,
    };
    ek && bd_uyumlu::<BPC>(bd, bdmax)
        && if BPC == 8 {
            bilesik_sozlesmesi(w, h, 4, t1, t2, w * h, dst_len, ds)
        } else {
            bilesik16_sozlesmesi(w, h, 4, t1, t2, dst_len, ds, bd, bdmax)
        }
}

/// `w_mask[SUB]` gövdesi (dav1d `w_mask_c`; `SUB` 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0).
#[target_feature(enable = "neon")]
pub(super) fn w_mask_neon<const BPC: u32, const SUB: usize>(
    dst: &mut [u8],
    ds: usize,
    tmp1: &[i16],
    tmp2: &[i16],
    w: usize,
    h: usize,
    mask: &mut [u8],
    sign: i32,
) {
    let ib = if BPC == 8 { 4 } else { 14 - BPC as i32 };
    let kaydirma = ib + 6;
    let sapma = if BPC == 8 { 0 } else { 8192 * 64 };
    let mask_sh = BPC as i32 + ib - 4;
    let mask_rnd = vdupq_n_s32(1 << (mask_sh - 5));
    let nms = vdupq_n_s32(-mask_sh);
    let nk = vdupq_n_s32(-kaydirma);
    let sap = vdupq_n_s32(sapma);
    let c38 = vdupq_n_s32(38);
    let c64 = vdupq_n_s32(64);
    let maxv = vdupq_n_u16(((1u32 << BPC) - 1) as u16);
    let ss_hor = SUB != 0;
    let ss_ver = SUB == 2;
    let mask_row = w >> usize::from(ss_hor);
    let mut mask_off = 0usize;
    for y in 0..h {
        let rows_tek = (h - y) & 1 != 0;
        let mut x = 0;
        while x < w {
            let i = y * w + x;
            let t1 = load_i16x8(chunk(tmp1, i));
            let t2 = load_i16x8(chunk(tmp2, i));
            let dlo = vsubl_s16(vget_low_s16(t1), vget_low_s16(t2));
            let dhi = vsubl_high_s16(t1, t2);
            let mlo = vminq_s32(
                vaddq_s32(vshlq_s32(vaddq_s32(vabsq_s32(dlo), mask_rnd), nms), c38),
                c64,
            );
            let mhi = vminq_s32(
                vaddq_s32(vshlq_s32(vaddq_s32(vabsq_s32(dhi), mask_rnd), nms), c38),
                c64,
            );
            let vlo = vaddq_s32(
                vmulq_s32(dlo, mlo),
                vshlq_n_s32::<6>(vmovl_s16(vget_low_s16(t2))),
            );
            let vhi = vaddq_s32(vmulq_s32(dhi, mhi), vshlq_n_s32::<6>(vmovl_high_s16(t2)));
            let vlo = vrshlq_s32(vaddq_s32(vlo, sap), nk);
            let vhi = vrshlq_s32(vaddq_s32(vhi, sap), nk);
            let px = kirp16(vlo, vhi, maxv);
            if BPC == 8 {
                store_u8x8(chunk_mut(dst, y * ds + x), vmovn_u16(px));
            } else {
                yaz_u16(dst, y * ds + 2 * x, px, 8);
            }
            // m ∈ [38, 64] → u16.
            let m16 = vcombine_u16(
                vmovn_u32(vreinterpretq_u32_s32(mlo)),
                vmovn_u32(vreinterpretq_u32_s32(mhi)),
            );
            if !ss_hor {
                store_u8x8(chunk_mut(mask, mask_off + x), vmovn_u16(m16));
            } else {
                let cift = vpaddq_u16(m16, m16); // ilk 4 şerit: m₀+m₁, m₂+m₃, …
                let idx = mask_off + (x >> 1);
                let deger = if !ss_ver {
                    vshrq_n_u16::<1>(vaddq_u16(cift, vdupq_n_u16((1 - sign) as u16)))
                } else if rows_tek {
                    let mut onceki = [0u8; 8];
                    onceki[..4].copy_from_slice(&mask[idx..idx + 4]);
                    let p = vmovl_u8(load_u8x8(&onceki));
                    vshrq_n_u16::<2>(vaddq_u16(
                        vaddq_u16(cift, p),
                        vdupq_n_u16((2 - sign) as u16),
                    ))
                } else {
                    cift
                };
                let mut t = [0u8; 8];
                store_u8x8(&mut t, vmovn_u16(deger));
                mask[idx..idx + 4].copy_from_slice(&t[..4]);
            }
            x += 8;
        }
        if !ss_ver || rows_tek {
            mask_off += mask_row;
        }
    }
}

/// `w_mask[SUB]` sözleşmesi.
#[inline]
pub(super) fn w_mask_kabul<const BPC: u32, const SUB: usize>(
    dst_len: usize,
    ds: usize,
    t1: usize,
    t2: usize,
    w: usize,
    h: usize,
    mask_len: usize,
    sign: i32,
    ss_hor: bool,
    ss_ver: bool,
    bd: BitDepth,
    bdmax: u16,
) -> bool {
    (ss_hor, ss_ver) == (SUB != 0, SUB == 2)
        && (0..=1).contains(&sign)
        && mask_len >= (w >> usize::from(ss_hor)) * (h >> usize::from(ss_ver))
        && bd_uyumlu::<BPC>(bd, bdmax)
        && if BPC == 8 {
            bilesik_sozlesmesi(w, h, 8, t1, t2, w * h, dst_len, ds)
        } else {
            bilesik16_sozlesmesi(w, h, 8, t1, t2, dst_len, ds, bd, bdmax)
        }
}

/// `avg` NEON yuvası; yalnız [`NeonToken`] ile.
#[must_use]
pub const fn avg_fn<const BPC: u32>(_neon: NeonToken) -> AvgFn {
    giris::mc_avg::<BPC>
}

/// `w_avg` NEON yuvası; yalnız [`NeonToken`] ile.
#[must_use]
pub const fn w_avg_fn<const BPC: u32>(_neon: NeonToken) -> WAvgFn {
    giris::mc_w_avg::<BPC>
}

/// `mask` NEON yuvası; yalnız [`NeonToken`] ile.
#[must_use]
pub const fn mask_fn<const BPC: u32>(_neon: NeonToken) -> MaskFn {
    giris::mc_mask::<BPC>
}

/// `w_mask[sub]` NEON yuvası; yalnız [`NeonToken`] ile.
#[must_use]
pub const fn w_mask_fn<const BPC: u32>(_neon: NeonToken, sub: usize) -> WMaskFn {
    match sub {
        0 => giris::mc_w_mask::<BPC, 0>,
        1 => giris::mc_w_mask::<BPC, 1>,
        _ => giris::mc_w_mask::<BPC, 2>,
    }
}

// ---------------------------------------------------------------------------
// Karışım: `blend`, `blend_v`, `blend_h`
// ---------------------------------------------------------------------------

/// Karışım türü.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Karisim {
    Blend,
    V,
    H,
}

/// 8 piksel: `(a·(64 − m) + b·m + 32) >> 6`.
#[target_feature(enable = "neon")]
#[inline]
fn karis8(a: uint8x8_t, b: uint8x8_t, m: uint8x8_t) -> uint8x8_t {
    vrshrn_n_u16::<6>(vmlal_u8(vmull_u8(a, vsub_u8(vdup_n_u8(64), m)), b, m))
}

/// 8 adet 16 bpc piksel: `(a·(64 − m) + b·m + 32) >> 6` (`u32`).
#[target_feature(enable = "neon")]
#[inline]
fn karis16(a: uint16x8_t, b: uint16x8_t, m: uint16x8_t) -> uint16x8_t {
    let mb = vsubq_u16(vdupq_n_u16(64), m);
    let lo = vmlal_u16(
        vmull_u16(vget_low_u16(a), vget_low_u16(mb)),
        vget_low_u16(b),
        vget_low_u16(m),
    );
    let hi = vmlal_high_u16(vmull_high_u16(a, mb), b, m);
    vcombine_u16(vrshrn_n_u32::<6>(lo), vrshrn_n_u32::<6>(hi))
}

/// `blend` / `blend_v` / `blend_h` gövdesi (sözleşme girişte denetlendi).
#[target_feature(enable = "neon")]
pub(super) fn karisim_neon<const BPC: u32>(
    tur: Karisim,
    dst: &mut [u8],
    ds: usize,
    tmp: &[u8],
    w: usize,
    h: usize,
    mask: &[u8],
) {
    match tur {
        Karisim::V => {
            let sutun = (w * 3) >> 2;
            let rem = sutun & 7;
            let mv0 = if sutun >= 8 {
                load_u8x8(chunk(&OBMC_MASKS, w))
            } else {
                vdup_n_u8(0)
            };
            let mv1 = if sutun >= 16 {
                load_u8x8(chunk(&OBMC_MASKS, w + 8))
            } else {
                vdup_n_u8(0)
            };
            let mv2 = if sutun >= 24 {
                load_u8x8(chunk(&OBMC_MASKS, w + 16))
            } else {
                vdup_n_u8(0)
            };
            let mv_rem = if rem != 0 {
                oku_kismi_u8(&OBMC_MASKS, w + (sutun & !7), rem)
            } else {
                vdup_n_u8(0)
            };
            let m16_0 = vmovl_u8(mv0);
            let m16_1 = vmovl_u8(mv1);
            let m16_2 = vmovl_u8(mv2);
            let m16_rem = vmovl_u8(mv_rem);
            let rem_x = sutun & !7;

            for y in 0..h {
                if BPC == 8 {
                    if sutun >= 8 {
                        let a0 = load_u8x8(chunk(dst, y * ds));
                        let b0 = load_u8x8(chunk(tmp, y * w));
                        store_u8x8(chunk_mut(dst, y * ds), karis8(a0, b0, mv0));
                    }
                    if sutun >= 16 {
                        let a1 = load_u8x8(chunk(dst, y * ds + 8));
                        let b1 = load_u8x8(chunk(tmp, y * w + 8));
                        store_u8x8(chunk_mut(dst, y * ds + 8), karis8(a1, b1, mv1));
                    }
                    if sutun >= 24 {
                        let a2 = load_u8x8(chunk(dst, y * ds + 16));
                        let b2 = load_u8x8(chunk(tmp, y * w + 16));
                        store_u8x8(chunk_mut(dst, y * ds + 16), karis8(a2, b2, mv2));
                    }
                    if rem != 0 {
                        let ar = oku_kismi_u8(dst, y * ds + rem_x, rem);
                        let br = oku_kismi_u8(tmp, y * w + rem_x, rem);
                        yaz_u8(dst, y * ds + rem_x, karis8(ar, br, mv_rem), rem);
                    }
                } else {
                    if sutun >= 8 {
                        let a0 = oku_u16(dst, y * ds);
                        let b0 = oku_u16(tmp, 2 * y * w);
                        store_u8x16(
                            chunk_mut(dst, y * ds),
                            vreinterpretq_u8_u16(karis16(a0, b0, m16_0)),
                        );
                    }
                    if sutun >= 16 {
                        let a1 = oku_u16(dst, y * ds + 16);
                        let b1 = oku_u16(tmp, 2 * (y * w + 8));
                        store_u8x16(
                            chunk_mut(dst, y * ds + 16),
                            vreinterpretq_u8_u16(karis16(a1, b1, m16_1)),
                        );
                    }
                    if sutun >= 24 {
                        let a2 = oku_u16(dst, y * ds + 32);
                        let b2 = oku_u16(tmp, 2 * (y * w + 16));
                        store_u8x16(
                            chunk_mut(dst, y * ds + 32),
                            vreinterpretq_u8_u16(karis16(a2, b2, m16_2)),
                        );
                    }
                    if rem != 0 {
                        let ar = oku_kismi_u16(dst, y * ds + 2 * rem_x, rem);
                        let br = oku_kismi_u16(tmp, 2 * (y * w + rem_x), rem);
                        yaz_u16(dst, y * ds + 2 * rem_x, karis16(ar, br, m16_rem), rem);
                    }
                }
            }
        }
        Karisim::H => {
            let satir = (h * 3) >> 2;
            if BPC == 8 && w >= 16 {
                for y in 0..satir {
                    let m8 = vdup_n_u8(OBMC_MASKS[h + y]);
                    let mut x = 0;
                    while x < w {
                        let a = load_u8x16(chunk(dst, y * ds + x));
                        let b = load_u8x16(chunk(tmp, y * w + x));
                        let out = vcombine_u8(
                            karis8(vget_low_u8(a), vget_low_u8(b), m8),
                            karis8(vget_high_u8(a), vget_high_u8(b), m8),
                        );
                        store_u8x16(chunk_mut(dst, y * ds + x), out);
                        x += 16;
                    }
                }
            } else if w >= 8 {
                for y in 0..satir {
                    let m8 = vdup_n_u8(OBMC_MASKS[h + y]);
                    let m16 = vmovl_u8(m8);
                    let mut x = 0;
                    while x < w {
                        if BPC == 8 {
                            let a = load_u8x8(chunk(dst, y * ds + x));
                            let b = load_u8x8(chunk(tmp, y * w + x));
                            store_u8x8(chunk_mut(dst, y * ds + x), karis8(a, b, m8));
                        } else {
                            let a = oku_u16(dst, y * ds + 2 * x);
                            let b = oku_u16(tmp, 2 * (y * w + x));
                            store_u8x16(
                                chunk_mut(dst, y * ds + 2 * x),
                                vreinterpretq_u8_u16(karis16(a, b, m16)),
                            );
                        }
                        x += 8;
                    }
                }
            } else {
                for y in 0..satir {
                    let m8 = vdup_n_u8(OBMC_MASKS[h + y]);
                    let m16 = vmovl_u8(m8);
                    if BPC == 8 {
                        let a = oku_kismi_u8(dst, y * ds, w);
                        let b = oku_kismi_u8(tmp, y * w, w);
                        yaz_u8(dst, y * ds, karis8(a, b, m8), w);
                    } else {
                        let a = oku_kismi_u16(dst, y * ds, w);
                        let b = oku_kismi_u16(tmp, 2 * y * w, w);
                        yaz_u16(dst, y * ds, karis16(a, b, m16), w);
                    }
                }
            }
        }
        Karisim::Blend => {
            if BPC == 8 && w >= 16 {
                for y in 0..h {
                    let mut x = 0;
                    while x < w {
                        let a = load_u8x16(chunk(dst, y * ds + x));
                        let b = load_u8x16(chunk(tmp, y * w + x));
                        let m = load_u8x16(chunk(mask, y * w + x));
                        let out = vcombine_u8(
                            karis8(vget_low_u8(a), vget_low_u8(b), vget_low_u8(m)),
                            karis8(vget_high_u8(a), vget_high_u8(b), vget_high_u8(m)),
                        );
                        store_u8x16(chunk_mut(dst, y * ds + x), out);
                        x += 16;
                    }
                }
            } else if w >= 8 {
                for y in 0..h {
                    let mut x = 0;
                    while x < w {
                        let m8 = load_u8x8(chunk(mask, y * w + x));
                        if BPC == 8 {
                            let a = load_u8x8(chunk(dst, y * ds + x));
                            let b = load_u8x8(chunk(tmp, y * w + x));
                            store_u8x8(chunk_mut(dst, y * ds + x), karis8(a, b, m8));
                        } else {
                            let a = oku_u16(dst, y * ds + 2 * x);
                            let b = oku_u16(tmp, 2 * (y * w + x));
                            store_u8x16(
                                chunk_mut(dst, y * ds + 2 * x),
                                vreinterpretq_u8_u16(karis16(a, b, vmovl_u8(m8))),
                            );
                        }
                        x += 8;
                    }
                }
            } else {
                for y in 0..h {
                    let m8 = oku_kismi_u8(mask, y * w, w);
                    if BPC == 8 {
                        let a = oku_kismi_u8(dst, y * ds, w);
                        let b = oku_kismi_u8(tmp, y * w, w);
                        yaz_u8(dst, y * ds, karis8(a, b, m8), w);
                    } else {
                        let a = oku_kismi_u16(dst, y * ds, w);
                        let b = oku_kismi_u16(tmp, 2 * y * w, w);
                        yaz_u16(dst, y * ds, karis16(a, b, vmovl_u8(m8)), w);
                    }
                }
            }
        }
    }
}

/// `blend*` sözleşmesi (x86 girişleriyle aynı aralıklar).
#[inline]
pub(super) fn karisim_kabul<const BPC: u32>(
    tur: Karisim,
    dst_len: usize,
    ds: usize,
    tmp_len: usize,
    w: usize,
    h: usize,
    mask_len: usize,
    bd: BitDepth,
    bdmax: u16,
) -> bool {
    let (wa, ha, yazilan) = match tur {
        Karisim::Blend => ((4, 32), (4, 32), h),
        Karisim::V => ((2, 32), (2, 128), h),
        Karisim::H => ((2, 128), (2, 32), (h * 3) >> 2),
    };
    let ek = tur != Karisim::Blend || mask_len >= w * h;
    ek && bd_uyumlu::<BPC>(bd, bdmax)
        && if BPC == 8 {
            karisim_sozlesmesi(w, wa, h, ha, yazilan, tmp_len, dst_len, ds)
        } else {
            karisim16_sozlesmesi(w, wa, h, ha, yazilan, tmp_len, dst_len, ds, bd, bdmax)
        }
}

/// `blend` NEON yuvası; yalnız [`NeonToken`] ile.
#[must_use]
pub const fn blend_fn<const BPC: u32>(_neon: NeonToken) -> BlendFn {
    giris::mc_blend::<BPC>
}

/// `blend_v` NEON yuvası; yalnız [`NeonToken`] ile.
#[must_use]
pub const fn blend_v_fn<const BPC: u32>(_neon: NeonToken) -> BlendDirFn {
    giris::mc_blend_v::<BPC>
}

/// `blend_h` NEON yuvası; yalnız [`NeonToken`] ile.
#[must_use]
pub const fn blend_h_fn<const BPC: u32>(_neon: NeonToken) -> BlendDirFn {
    giris::mc_blend_h::<BPC>
}

// ---------------------------------------------------------------------------
// Warp: `warp8x8`, `warp8x8t`
// ---------------------------------------------------------------------------

/// İki `MC_WARP_FILTER` satırını (`[i8; 8]`) işaretli `int16x8_t` çiftine genişletir:
/// `(b ^ 0x80) − 128 == b` (`i8 ∈ [-128, 127]` için taşmasız özdeşlik).
macro_rules! warp_iki_satir_s16 {
    ($i0:expr, $i1:expr, $c80:expr, $c128:expr) => {{
        let f0 = MC_WARP_FILTER[$i0].map(|b| b as u8);
        let f1 = MC_WARP_FILTER[$i1].map(|b| b as u8);
        let raw = vcombine_u8(load_u8x8(&f0), load_u8x8(&f1));
        let biased = veorq_u8(raw, $c80);
        (
            vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(biased), $c128)),
            vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(biased), $c128)),
        )
    }};
}

/// 8 sütunun filtre katsayılarını yazmaç içi `8×8` `vuzp1q_s16`/`vuzp2q_s16` devriğiyle dizer
/// (ikili yerleşiminden bağımsız kesin satır içi açılım için makro).
macro_rules! warp_katsayi_vec {
    ($t:expr, $a:expr, $c80:expr, $c128:expr) => {{
        let t_val: i32 = $t;
        let a_val: i32 = $a;
        if a_val == 0 {
            let f = &MC_WARP_FILTER[(64 + ((t_val + 512) >> 10)) as usize];
            [
                vdupq_n_s16(i16::from(f[0])),
                vdupq_n_s16(i16::from(f[1])),
                vdupq_n_s16(i16::from(f[2])),
                vdupq_n_s16(i16::from(f[3])),
                vdupq_n_s16(i16::from(f[4])),
                vdupq_n_s16(i16::from(f[5])),
                vdupq_n_s16(i16::from(f[6])),
                vdupq_n_s16(i16::from(f[7])),
            ]
        } else {
            let idx0 = (64 + ((t_val + 512) >> 10)) as usize;
            let idx1 = (64 + ((t_val + a_val + 512) >> 10)) as usize;
            let idx2 = (64 + ((t_val + 2 * a_val + 512) >> 10)) as usize;
            let idx3 = (64 + ((t_val + 3 * a_val + 512) >> 10)) as usize;
            let idx4 = (64 + ((t_val + 4 * a_val + 512) >> 10)) as usize;
            let idx5 = (64 + ((t_val + 5 * a_val + 512) >> 10)) as usize;
            let idx6 = (64 + ((t_val + 6 * a_val + 512) >> 10)) as usize;
            let idx7 = (64 + ((t_val + 7 * a_val + 512) >> 10)) as usize;
            let (s0, s1) = warp_iki_satir_s16!(idx0, idx1, $c80, $c128);
            let (s2, s3) = warp_iki_satir_s16!(idx2, idx3, $c80, $c128);
            let (s4, s5) = warp_iki_satir_s16!(idx4, idx5, $c80, $c128);
            let (s6, s7) = warp_iki_satir_s16!(idx6, idx7, $c80, $c128);

            let u0 = vuzp1q_s16(s0, s1);
            let u1 = vuzp2q_s16(s0, s1);
            let u2 = vuzp1q_s16(s2, s3);
            let u3 = vuzp2q_s16(s2, s3);
            let u4 = vuzp1q_s16(s4, s5);
            let u5 = vuzp2q_s16(s4, s5);
            let u6 = vuzp1q_s16(s6, s7);
            let u7 = vuzp2q_s16(s6, s7);

            let v0 = vuzp1q_s16(u0, u2);
            let v2 = vuzp2q_s16(u0, u2);
            let v4 = vuzp1q_s16(u4, u6);
            let v6 = vuzp2q_s16(u4, u6);
            let v1 = vuzp1q_s16(u1, u3);
            let v3 = vuzp2q_s16(u1, u3);
            let v5 = vuzp1q_s16(u5, u7);
            let v7 = vuzp2q_s16(u5, u7);

            [
                vuzp1q_s16(v0, v4),
                vuzp1q_s16(v1, v5),
                vuzp1q_s16(v2, v6),
                vuzp1q_s16(v3, v7),
                vuzp2q_s16(v0, v4),
                vuzp2q_s16(v1, v5),
                vuzp2q_s16(v2, v6),
                vuzp2q_s16(v3, v7),
            ]
        }
    }};
}

/// Warp gövdesi (`T`: `warp8x8t`, `i16` çıktı). Sözleşme girişte denetlendi.
#[target_feature(enable = "neon")]
pub(super) fn warp_neon<const BPC: u32, const T: bool>(
    dst: &mut [u8],
    tmp: &mut [i16],
    ds: usize,
    src: &[u8],
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
) {
    let ib = if BPC == 8 { 4 } else { 14 - BPC as i32 };
    let [a, b, c, d] = abcd.map(i32::from);
    let nh = vdupq_n_s32(-(7 - ib));
    let c80 = vdupq_n_u8(0x80);
    let c128 = vdup_n_u8(128);
    let mut mid = [vdupq_n_s16(0); 15];
    for (r, m) in mid.iter_mut().enumerate() {
        let k = warp_katsayi_vec!(mx + r as i32 * b, a, c80, c128);
        let (pa, pb) = if BPC == 8 {
            let v = load_u8x16(chunk(src, r * ss));
            (
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v))),
                vreinterpretq_s16_u16(vmovl_high_u8(v)),
            )
        } else {
            (oku_p16(src, r * ss), oku_p16(src, r * ss + 16))
        };
        let mut lo = vmull_s16(vget_low_s16(pa), vget_low_s16(k[0]));
        let mut hi = vmull_high_s16(pa, k[0]);
        macro_rules! adim {
            ($($j:literal)*) => {$(
                let e = vextq_s16::<$j>(pa, pb);
                lo = vmlal_s16(lo, vget_low_s16(e), vget_low_s16(k[$j]));
                hi = vmlal_high_s16(hi, e, k[$j]);
            )*};
        }
        adim!(1 2 3 4 5 6 7);
        *m = dar16(vrshlq_s32(lo, nh), vrshlq_s32(hi, nh));
    }
    let maxv = vdupq_n_u16(((1u32 << BPC) - 1) as u16);
    let nv = vdupq_n_s32(-(7 + ib));
    let bias = vdupq_n_s32(if BPC == 8 { 0 } else { 8192 });
    for y in 0..8 {
        let k = warp_katsayi_vec!(my + y as i32 * d, c, c80, c128);
        let mut lo = vmull_s16(vget_low_s16(mid[y]), vget_low_s16(k[0]));
        let mut hi = vmull_high_s16(mid[y], k[0]);
        for j in 1..8 {
            lo = vmlal_s16(lo, vget_low_s16(mid[y + j]), vget_low_s16(k[j]));
            hi = vmlal_high_s16(hi, mid[y + j], k[j]);
        }
        if T {
            let lo = vsubq_s32(vrshrq_n_s32::<7>(lo), bias);
            let hi = vsubq_s32(vrshrq_n_s32::<7>(hi), bias);
            store_i16x8(chunk_mut(tmp, y * ds), dar16(lo, hi));
        } else if BPC == 8 {
            let u = vcombine_u16(vqrshrun_n_s32::<11>(lo), vqrshrun_n_s32::<11>(hi));
            store_u8x8(chunk_mut(dst, y * ds), vqmovn_u16(u));
        } else {
            let v = kirp16(vrshlq_s32(lo, nv), vrshlq_s32(hi, nv), maxv);
            yaz_u16(dst, y * ds, v, 8);
        }
    }
}

/// `warp8x8` / `warp8x8t` sözleşmesi (`oge_bayt`: `warp8x8` 16 bpc'de 2, `warp8x8t` 1).
#[inline]
pub(super) fn warp_kabul<const BPC: u32, const T: bool>(
    src_len: usize,
    ss: usize,
    abcd: &[i16; 4],
    mx: i32,
    my: i32,
    dst_len: usize,
    ds: usize,
    bd: BitDepth,
    bdmax: u16,
) -> bool {
    bd_uyumlu::<BPC>(bd, bdmax)
        && if BPC == 8 {
            warp_sozlesmesi(src_len, ss, abcd, mx, my, dst_len, ds)
        } else {
            warp16_sozlesmesi(
                src_len,
                ss,
                abcd,
                mx,
                my,
                dst_len,
                ds,
                if T { 1 } else { 2 },
                bd,
                bdmax,
            )
        }
}

/// `warp8x8` NEON yuvası; yalnız [`NeonToken`] ile.
#[must_use]
pub const fn warp_fn<const BPC: u32>(_neon: NeonToken) -> Warp8x8Fn {
    giris::mc_warp::<BPC>
}

/// `warp8x8t` NEON yuvası; yalnız [`NeonToken`] ile.
#[must_use]
pub const fn warp_t_fn<const BPC: u32>(_neon: NeonToken) -> Warp8x8tFn {
    giris::mc_warp_t::<BPC>
}

// ---------------------------------------------------------------------------
// `emu_edge`
// ---------------------------------------------------------------------------

/// `_memcpy` çağrısız NEON bayt kopyası (`dst[d_off..d_off+len] <- src[s_off..s_off+len]`).
#[target_feature(enable = "neon")]
#[inline]
fn kopyala_bayt_neon(dst: &mut [u8], d_off: usize, src: &[u8], s_off: usize, len: usize) {
    let mut i = 0usize;
    while i + 16 <= len {
        let v = load_u8x16(chunk(src, s_off + i));
        store_u8x16(chunk_mut(dst, d_off + i), v);
        i += 16;
    }
    if i + 8 <= len {
        let v = load_u8x8(chunk(src, s_off + i));
        store_u8x8(chunk_mut(dst, d_off + i), v);
        i += 8;
    }
    if i + 4 <= len {
        *chunk_mut::<u8, 4>(dst, d_off + i) = *chunk::<u8, 4>(src, s_off + i);
        i += 4;
    }
    if i + 2 <= len {
        *chunk_mut::<u8, 2>(dst, d_off + i) = *chunk::<u8, 2>(src, s_off + i);
        i += 2;
    }
    if i < len {
        dst[d_off + i] = src[s_off + i];
    }
}

/// Referans satırı `[src_row .. src_row + n)` bloğunu `dst` içinde `rows` ardışık satıra çoğaltır.
#[target_feature(enable = "neon")]
#[inline]
fn cogalt_satirlar_neon(
    dst: &mut [u8],
    src_row: usize,
    dst_first_row: usize,
    ds: usize,
    n: usize,
    rows: usize,
) {
    if rows == 0 {
        return;
    }
    if n <= 32 {
        // Sık karşılaşılan `bw * bpp <= 32` durumunda referans satırı yazmaçlarda tut (`0` `memmove`).
        let mut buf = [0u8; 32];
        kopyala_bayt_neon(&mut buf, 0, dst, src_row, n);
        let v0 = load_u8x16(chunk(&buf, 0));
        let v1 = load_u8x16(chunk(&buf, 16));
        for r in 0..rows {
            let d = dst_first_row + r * ds;
            if n >= 16 {
                store_u8x16(chunk_mut(dst, d), v0);
            }
            if n == 32 {
                store_u8x16(chunk_mut(dst, d + 16), v1);
            } else {
                let rem_i = n & !15;
                let mut i = rem_i;
                if i + 8 <= n {
                    store_u8x8(chunk_mut(dst, d + i), load_u8x8(chunk(&buf, i)));
                    i += 8;
                }
                if i + 4 <= n {
                    *chunk_mut::<u8, 4>(dst, d + i) = *chunk::<u8, 4>(&buf, i);
                    i += 4;
                }
                if i + 2 <= n {
                    *chunk_mut::<u8, 2>(dst, d + i) = *chunk::<u8, 2>(&buf, i);
                    i += 2;
                }
                if i < n {
                    dst[d + i] = buf[i];
                }
            }
        }
        return;
    }
    for r in 0..rows {
        let d = dst_first_row + r * ds;
        let mut i = 0usize;
        while i + 16 <= n {
            let v = load_u8x16(chunk(dst, src_row + i));
            store_u8x16(chunk_mut(dst, d + i), v);
            i += 16;
        }
        while i < n {
            dst[d + i] = dst[src_row + i];
            i += 1;
        }
    }
}

/// Bir satırın `[bas, bas + n)` piksellerini `p` ile doldurur (`_memcpy` çağrısız NEON saklamaları).
#[target_feature(enable = "neon")]
#[inline]
fn doldur<const BPC: u32>(dst: &mut [u8], bas: usize, n: usize, p: u16) {
    let bpp = if BPC == 8 { 1 } else { 2 };
    let v = if BPC == 8 {
        vdupq_n_u8(p as u8)
    } else {
        vreinterpretq_u8_u16(vdupq_n_u16(p))
    };
    let w64 = vget_lane_u64::<0>(vreinterpret_u64_u8(vget_low_u8(v)));
    let son = bas + n * bpp;
    let mut o = bas;
    while o + 16 <= son {
        store_u8x16(chunk_mut(dst, o), v);
        o += 16;
    }
    if o + 8 <= son {
        *chunk_mut::<u8, 8>(dst, o) = w64.to_ne_bytes();
        o += 8;
    }
    if o + 4 <= son {
        *chunk_mut::<u8, 4>(dst, o) = (w64 as u32).to_ne_bytes();
        o += 4;
    }
    if o + 2 <= son {
        *chunk_mut::<u8, 2>(dst, o) = (w64 as u16).to_ne_bytes();
        o += 2;
    }
    if o < son {
        dst[o] = w64 as u8;
    }
}

/// `emu_edge` gövdesi (`_memcpy` / `_memmove` çağrısız NEON satır kopyalama ve çoğaltma).
#[target_feature(enable = "neon")]
pub(super) fn emu_edge_neon<const BPC: u32>(
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
) {
    let bpp = if BPC == 8 { 1 } else { 2 };
    let piksel = |s: &[u8], o: usize| -> u16 {
        if BPC == 8 {
            u16::from(s[o])
        } else {
            u16::from_ne_bytes([s[o], s[o + 1]])
        }
    };
    let ref_row =
        y.clamp(0, ih as i32 - 1) as usize * ss + x.clamp(0, iw as i32 - 1) as usize * bpp;
    let sol = (-x).clamp(0, bw as i32 - 1) as usize;
    let sag = (x + bw as i32 - iw as i32).clamp(0, bw as i32 - 1) as usize;
    let ust = (-y).clamp(0, bh as i32 - 1) as usize;
    let alt = (y + bh as i32 - ih as i32).clamp(0, bh as i32 - 1) as usize;
    let cw = bw - sol - sag;
    let ch = bh - ust - alt;
    for row in 0..ch {
        let s = ref_row + row * ss;
        let d = (ust + row) * ds;
        kopyala_bayt_neon(dst, d + sol * bpp, src, s, cw * bpp);
        if sol > 0 {
            let p = piksel(dst, d + sol * bpp);
            doldur::<BPC>(dst, d, sol, p);
        }
        if sag > 0 {
            let p = piksel(dst, d + (sol + cw - 1) * bpp);
            doldur::<BPC>(dst, d + (sol + cw) * bpp, sag, p);
        }
    }
    let n = bw * bpp;
    cogalt_satirlar_neon(dst, ust * ds, 0, ds, n, ust);
    let last = (ust + ch - 1) * ds;
    cogalt_satirlar_neon(dst, last, (ust + ch) * ds, ds, n, alt);
}

/// `emu_edge` sözleşmesi.
#[inline]
pub(super) fn emu_edge_kabul<const BPC: u32>(
    bw: usize,
    bh: usize,
    iw: usize,
    ih: usize,
    x: i32,
    y: i32,
    dst_len: usize,
    ds: usize,
    src_len: usize,
    ss: usize,
    bd: BitDepth,
    bdmax: u16,
) -> bool {
    bd_uyumlu::<BPC>(bd, bdmax)
        && if BPC == 8 {
            kenar_sozlesmesi(bw, bh, iw, ih, x, y, dst_len, ds, src_len, ss)
        } else {
            kenar16_sozlesmesi(bw, bh, iw, ih, x, y, dst_len, ds, src_len, ss, bd, bdmax)
        }
}

/// `emu_edge` NEON yuvası; yalnız [`NeonToken`] ile.
#[must_use]
pub const fn emu_edge_fn<const BPC: u32>(_neon: NeonToken) -> EmuEdgeFn {
    giris::mc_emu_edge::<BPC>
}

#[cfg(test)]
mod tests {
    use crate::dsp::mc_tables::MC_SUBPEL_FILTERS;

    /// 8 bpc 8-tap toplamının `i16`'ya sığdığının kanıtı (modül belgesi, SMD-02).
    #[test]
    fn sekiz_bpc_i16_toplam_siniri() {
        for tablo in &MC_SUBPEL_FILTERS {
            for f in tablo {
                let pozitif: i32 = f.iter().map(|&c| i32::from(c).max(0)).sum();
                let negatif: i32 = f.iter().map(|&c| i32::from(c).min(0)).sum();
                assert_eq!(f.iter().map(|&c| i32::from(c)).sum::<i32>(), 64);
                assert!(255 * pozitif + 34 <= i32::from(i16::MAX), "{f:?}");
                assert!(255 * negatif >= i32::from(i16::MIN), "{f:?}");
            }
        }
    }
}
