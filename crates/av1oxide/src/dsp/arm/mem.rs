//! Bellek erişimli `aarch64` NEON intrinsic'lerinin sabit boyutlu dilim sarmalayıcıları (`F5a` / `P51`).
//!
//! `rustc 1.100.0`'da işaretçisiz `core::arch::aarch64` NEON intrinsic'leri
//! `#[target_feature(enable = "neon")]` bağlamında **güvenlidir** (`target_feature 1.1`).
//! Yalnız ham işaretçi alan yükleme/saklama (`vld1*` / `vst1*`) intrinsic'leri
//! `unsafe` kalır; bu modül `dsp/arm/` altındaki `unsafe` bellek erişimlerinin
//! **tek** yeridir. Her yardımcı sabit boyutlu bir dizi referansı (`&[T; N]` veya
//! `&mut [T; N]`) alır; sınır denetimi çağıranın dilimden diziyi ayırdığı yerde
//! ([`chunk`], [`chunk_mut`]) bir kez yapılır.

#![allow(unsafe_code, clippy::missing_safety_doc)]

use core::arch::aarch64::{
    int16x4_t, int16x8_t, uint8x8_t, uint8x16_t, uint16x4_t, uint16x8_t, vld1_s16, vld1_u8,
    vld1_u16, vld1q_s16, vld1q_u8, vld1q_u16, vst1_u8, vst1_u16, vst1q_u8, vst1q_u16,
};

use crate::decode::refmvs::RefBlock;
use crate::decode::refmvs_tmvs::TemporalBlock;

/// `s[at .. at + N]` dilimini sabit boyutlu dizi referansına ayırır; aşımda panik.
#[inline(always)]
#[must_use]
pub fn chunk<T, const N: usize>(s: &[T], at: usize) -> &[T; N] {
    s[at..]
        .first_chunk::<N>()
        .expect("ARM64 NEON yüklemesi dilim sınırını aşıyor")
}

/// `s[at .. at + N]` dilimini değiştirilebilir sabit boyutlu dizi referansına ayırır; aşımda panik.
#[inline(always)]
#[must_use]
pub fn chunk_mut<T, const N: usize>(s: &mut [T], at: usize) -> &mut [T; N] {
    s[at..]
        .first_chunk_mut::<N>()
        .expect("ARM64 NEON saklaması dilim sınırını aşıyor")
}

/// 16 baytı (`&[u8; 16]`) hizasız `uint8x16_t` olarak yükler (`vld1q_u8`).
#[target_feature(enable = "neon")]
#[inline]
#[must_use]
pub fn load_u8x16(src: &[u8; 16]) -> uint8x16_t {
    // SAFETY: (1) ISA: `vld1q_u8` ARM64 NEON komutudur; işlev `#[target_feature(enable = "neon")]` taşır. (2) sınır: `src` sabit boyutlu `&[u8; 16]` referansıdır ve komut tam 16 bayt okur. (3) hizalama/takma ad: `vld1q_u8` hizalama şartı koşmaz; paylaşılan referanstan salt okuma yapılır.
    unsafe { vld1q_u8(src.as_ptr()) }
}

/// 16 baytı (`&mut [u8; 16]`) hizasız saklar (`vst1q_u8`).
#[target_feature(enable = "neon")]
#[inline]
pub fn store_u8x16(dst: &mut [u8; 16], v: uint8x16_t) {
    // SAFETY: (1) ISA: `vst1q_u8` ARM64 NEON komutudur; işlev `#[target_feature(enable = "neon")]` taşır. (2) sınır: `dst` sabit boyutlu `&mut [u8; 16]` referansıdır ve komut tam 16 bayt yazar. (3) hizalama/takma ad: `vst1q_u8` hizalama şartı koşmaz; `&mut` tek erişim yoludur.
    unsafe { vst1q_u8(dst.as_mut_ptr(), v) }
}

/// 8 baytı (`&[u8; 8]`) hizasız `uint8x8_t` olarak yükler (`vld1_u8`).
#[target_feature(enable = "neon")]
#[inline]
#[must_use]
pub fn load_u8x8(src: &[u8; 8]) -> uint8x8_t {
    // SAFETY: (1) ISA: `vld1_u8` ARM64 NEON komutudur; işlev `#[target_feature(enable = "neon")]` taşır. (2) sınır: `src` sabit boyutlu `&[u8; 8]` referansıdır ve komut tam 8 bayt okur. (3) hizalama/takma ad: `vld1_u8` hizalama şartı koşmaz; paylaşılan referanstan salt okuma yapılır.
    unsafe { vld1_u8(src.as_ptr()) }
}

/// 8 baytı (`&mut [u8; 8]`) hizasız saklar (`vst1_u8`).
#[target_feature(enable = "neon")]
#[inline]
pub fn store_u8x8(dst: &mut [u8; 8], v: uint8x8_t) {
    // SAFETY: (1) ISA: `vst1_u8` ARM64 NEON komutudur; işlev `#[target_feature(enable = "neon")]` taşır. (2) sınır: `dst` sabit boyutlu `&mut [u8; 8]` referansıdır ve komut tam 8 bayt yazar. (3) hizalama/takma ad: `vst1_u8` hizalama şartı koşmaz; `&mut` tek erişim yoludur.
    unsafe { vst1_u8(dst.as_mut_ptr(), v) }
}

/// 8 adet `u16` (`&[u16; 8]`, 16 bayt) hizasız `uint16x8_t` olarak yükler (`vld1q_u16`).
#[target_feature(enable = "neon")]
#[inline]
#[must_use]
pub fn load_u16x8(src: &[u16; 8]) -> uint16x8_t {
    // SAFETY: (1) ISA: `vld1q_u16` ARM64 NEON komutudur; işlev `#[target_feature(enable = "neon")]` taşır. (2) sınır: `src` sabit boyutlu `&[u16; 8]` (16 bayt) referansıdır ve komut tam 16 bayt okur. (3) hizalama/takma ad: `vld1q_u16` `&[u16; 8]` üzerinde geçerlidir; paylaşılan referanstan salt okuma yapılır.
    unsafe { vld1q_u16(src.as_ptr()) }
}

/// 8 adet `u16` (`&mut [u16; 8]`, 16 bayt) hizasız saklar (`vst1q_u16`).
#[target_feature(enable = "neon")]
#[inline]
pub fn store_u16x8(dst: &mut [u16; 8], v: uint16x8_t) {
    // SAFETY: (1) ISA: `vst1q_u16` ARM64 NEON komutudur; işlev `#[target_feature(enable = "neon")]` taşır. (2) sınır: `dst` sabit boyutlu `&mut [u16; 8]` (16 bayt) referansıdır ve komut tam 16 bayt yazar. (3) hizalama/takma ad: `vst1q_u16` `&mut [u16; 8]` üzerinde geçerlidir; `&mut` tek erişim yoludur.
    unsafe { vst1q_u16(dst.as_mut_ptr(), v) }
}

/// 4 adet `u16` (`&[u16; 4]`, 8 bayt) hizasız `uint16x4_t` olarak yükler (`vld1_u16`).
#[target_feature(enable = "neon")]
#[inline]
#[must_use]
pub fn load_u16x4(src: &[u16; 4]) -> uint16x4_t {
    // SAFETY: (1) ISA: `vld1_u16` ARM64 NEON komutudur; işlev `#[target_feature(enable = "neon")]` taşır. (2) sınır: `src` sabit boyutlu `&[u16; 4]` (8 bayt) referansıdır ve komut tam 8 bayt okur. (3) hizalama/takma ad: `vld1_u16` `&[u16; 4]` üzerinde geçerlidir; paylaşılan referanstan salt okuma yapılır.
    unsafe { vld1_u16(src.as_ptr()) }
}

/// 4 adet `u16` (`&mut [u16; 4]`, 8 bayt) hizasız saklar (`vst1_u16`).
#[target_feature(enable = "neon")]
#[inline]
pub fn store_u16x4(dst: &mut [u16; 4], v: uint16x4_t) {
    // SAFETY: (1) ISA: `vst1_u16` ARM64 NEON komutudur; işlev `#[target_feature(enable = "neon")]` taşır. (2) sınır: `dst` sabit boyutlu `&mut [u16; 4]` (8 bayt) referansıdır ve komut tam 8 bayt yazar. (3) hizalama/takma ad: `vst1_u16` `&mut [u16; 4]` üzerinde geçerlidir; `&mut` tek erişim yoludur.
    unsafe { vst1_u16(dst.as_mut_ptr(), v) }
}

/// 8 adet `i16` (`&[i16; 8]`, 16 bayt) `int16x8_t` olarak yükler (`vld1q_s16`).
#[target_feature(enable = "neon")]
#[inline]
#[must_use]
pub fn load_i16x8(src: &[i16; 8]) -> int16x8_t {
    // SAFETY: (1) ISA: `vld1q_s16` ARM64 NEON komutudur; işlev `#[target_feature(enable = "neon")]` taşır. (2) sınır: `src` sabit boyutlu `&[i16; 8]` (16 bayt) referansıdır ve komut tam 16 bayt okur. (3) hizalama/takma ad: `vld1q_s16` `&[i16; 8]` üzerinde geçerlidir; paylaşılan referanstan salt okuma yapılır.
    unsafe { vld1q_s16(src.as_ptr()) }
}

/// 4 adet `i16` (`&[i16; 4]`, 8 bayt) `int16x4_t` olarak yükler (`vld1_s16`).
#[target_feature(enable = "neon")]
#[inline]
#[must_use]
pub fn load_i16x4(src: &[i16; 4]) -> int16x4_t {
    // SAFETY: (1) ISA: `vld1_s16` ARM64 NEON komutudur; işlev `#[target_feature(enable = "neon")]` taşır. (2) sınır: `src` sabit boyutlu `&[i16; 4]` (8 bayt) referansıdır ve komut tam 8 bayt okur. (3) hizalama/takma ad: `vld1_s16` `&[i16; 4]` üzerinde geçerlidir; paylaşılan referanstan salt okuma yapılır.
    unsafe { vld1_s16(src.as_ptr()) }
}

/// 4 adet `#[repr(C)]` [`RefBlock`] (`4 × 12 = 48` bayt) öğesini 3 × 16 baytlık NEON yazmacıyla saklar.
#[target_feature(enable = "neon")]
#[inline]
pub fn store_refblock_x4(dst: &mut [RefBlock; 4], m0: uint8x16_t, m1: uint8x16_t, m2: uint8x16_t) {
    const _: () = assert!(core::mem::size_of::<[RefBlock; 4]>() == 48);
    let ptr = dst.as_mut_ptr().cast::<u8>();
    // SAFETY: (1) ISA: `vst1q_u8` ARM64 NEON komutudur; işlev `#[target_feature(enable = "neon")]` taşır. (2) sınır: `[RefBlock; 4]` derleme zamanı kanıtıyla tam 48 bayttır; `0`, `16`, `32` ofsetlerine 3 × 16 = 48 bayt yazılır. (3) hizalama/takma ad: `vst1q_u8` hizasız `u8` işaretçisinde geçerlidir, `RefBlock` tüm bit desenleri geçerli `#[repr(C)]` tam sayı alanlarından oluşur ve `&mut` tek erişim yoludur.
    unsafe {
        vst1q_u8(ptr, m0);
        vst1q_u8(ptr.add(16), m1);
        vst1q_u8(ptr.add(32), m2);
    }
}

/// 4 adet `#[repr(C)]` [`TemporalBlock`] (`4 × 6 = 24` bayt) öğesini `16 + 8` baytlık NEON yazmaçlarıyla saklar.
#[target_feature(enable = "neon")]
#[inline]
pub fn store_temporal_x4(dst: &mut [TemporalBlock; 4], v0: uint8x16_t, v1: uint8x8_t) {
    const _: () = assert!(core::mem::size_of::<[TemporalBlock; 4]>() == 24);
    let ptr = dst.as_mut_ptr().cast::<u8>();
    // SAFETY: (1) ISA: `vst1q_u8` ve `vst1_u8` ARM64 NEON komutlarıdır; işlev `#[target_feature(enable = "neon")]` taşır. (2) sınır: `[TemporalBlock; 4]` derleme zamanı kanıtıyla tam 24 bayttır; `0..16` ve `16..24` aralıklarına toplam 24 bayt yazılır. (3) hizalama/takma ad: `vst1*_u8` hizasız `u8` işaretçisinde geçerlidir, `TemporalBlock` tüm bit desenleri geçerli `#[repr(C)]` tam sayı alanlarından oluşur ve `&mut` tek erişim yoludur.
    unsafe {
        vst1q_u8(ptr, v0);
        vst1_u8(ptr.add(16), v1);
    }
}

/// 8 adet `#[repr(C)]` [`TemporalBlock`] (`8 × 6 = 48` bayt) öğesini 3 × 16 baytlık NEON yazmacıyla saklar.
#[target_feature(enable = "neon")]
#[inline]
pub fn store_temporal_x8(
    dst: &mut [TemporalBlock; 8],
    v0: uint8x16_t,
    v1: uint8x16_t,
    v2: uint8x16_t,
) {
    const _: () = assert!(core::mem::size_of::<[TemporalBlock; 8]>() == 48);
    let ptr = dst.as_mut_ptr().cast::<u8>();
    // SAFETY: (1) ISA: `vst1q_u8` ARM64 NEON komutudur; işlev `#[target_feature(enable = "neon")]` taşır. (2) sınır: `[TemporalBlock; 8]` derleme zamanı kanıtıyla tam 48 bayttır; `0`, `16`, `32` ofsetlerine 3 × 16 = 48 bayt yazılır. (3) hizalama/takma ad: `vst1q_u8` hizasız `u8` işaretçisinde geçerlidir, `TemporalBlock` tüm bit desenleri geçerli `#[repr(C)]` tam sayı alanlarından oluşur ve `&mut` tek erişim yoludur.
    unsafe {
        vst1q_u8(ptr, v0);
        vst1q_u8(ptr.add(16), v1);
        vst1q_u8(ptr.add(32), v2);
    }
}

// ============================================================================
// `NeonToken`-kapılı `fn(...)` giriş köprüleri (`cdef`, `cdef16`, `pal`, `refmvs`)
// ============================================================================

use crate::decode::refmvs_tmvs::MfmvPlan;
use crate::dsp::cdef::{CdefBitDepth, CdefDirResult, CdefFilterParams};

/// 8-bpc `cdef/dir` giriş köprüsü (yalnız `cdef::find_dir_neon_fn(NeonToken)` ile verilir).
pub(super) fn find_dir_8bpc_entry(
    img: &[u8],
    stride: usize,
    bit_depth: CdefBitDepth,
) -> CdefDirResult {
    // SAFETY: (1) ISA: Bu işlev işaretçisi yalnız `NeonToken` kanıtıyla (`cdef::find_dir_neon_fn`) dışarı verilir; `aarch64` üzerinde NEON etkinliği doğrulanmıştır. (2) sınır: `find_dir_8bpc_neon` `%100 #![forbid(unsafe_code)]` güvenli Rust dilim sözleşmesiyle çalışır. (3) hizalama/takma ad: Yalnız güvenli `&[u8]` referansı iletilir.
    unsafe { super::cdef::find_dir_8bpc_neon(img, stride, bit_depth) }
}

/// 8-bpc `cdef/fb[0..2]` giriş köprüsü (yalnız `cdef::filter_*_neon_fn(NeonToken)` ile verilir).
pub(super) fn filter_8bpc_entry<const W: usize, const H: usize>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    // SAFETY: (1) ISA: Bu işlev işaretçisi yalnız `NeonToken` kanıtıyla (`cdef::filter_*_neon_fn`) dışarı verilir; `aarch64` üzerinde NEON etkinliği doğrulanmıştır. (2) sınır: `filter_8bpc_dispatch_neon` `%100 #![forbid(unsafe_code)]` güvenli Rust dilim sözleşmesiyle çalışır. (3) hizalama/takma ad: Yalnız güvenli `&mut [u8]` ve `&[u8]` referansları iletilir.
    unsafe {
        super::cdef::filter_8bpc_dispatch_neon::<W, H>(dst, stride, left, top, bottom, params);
    }
}

/// 16-bpc `cdef/dir` giriş köprüsü (yalnız `cdef16::find_dir_neon_fn(NeonToken)` ile verilir).
pub(super) fn find_dir_16bpc_entry<const BPC: u32>(
    img: &[u8],
    stride: usize,
    bit_depth: CdefBitDepth,
) -> CdefDirResult {
    // SAFETY: (1) ISA: Bu işlev işaretçisi yalnız `NeonToken` kanıtıyla (`cdef16::find_dir_neon_fn`) dışarı verilir; `aarch64` üzerinde NEON etkinliği doğrulanmıştır. (2) sınır: `find_dir_16bpc_neon` `%100 #![forbid(unsafe_code)]` güvenli Rust dilim sözleşmesiyle çalışır. (3) hizalama/takma ad: Yalnız güvenli `&[u8]` referansı iletilir.
    unsafe { super::cdef16::find_dir_16bpc_neon::<BPC>(img, stride, bit_depth) }
}

/// 16-bpc `cdef/fb[0..2]` giriş köprüsü (yalnız `cdef16::filter_*_neon_fn(NeonToken)` ile verilir).
pub(super) fn filter_16bpc_entry<const BPC: u32, const W: usize, const H: usize>(
    dst: &mut [u8],
    stride: usize,
    left: &[u8],
    top: &[u8],
    bottom: &[u8],
    params: CdefFilterParams,
) {
    // SAFETY: (1) ISA: Bu işlev işaretçisi yalnız `NeonToken` kanıtıyla (`cdef16::filter_*_neon_fn`) dışarı verilir; `aarch64` üzerinde NEON etkinliği doğrulanmıştır. (2) sınır: `filter_16bpc_dispatch_neon` `%100 #![forbid(unsafe_code)]` güvenli Rust dilim sözleşmesiyle çalışır. (3) hizalama/takma ad: Yalnız güvenli `&mut [u8]` ve `&[u8]` referansları iletilir.
    unsafe {
        super::cdef16::filter_16bpc_dispatch_neon::<BPC, W, H>(
            dst, stride, left, top, bottom, params,
        );
    }
}

/// `pal/pal_idx_finish` giriş köprüsü (yalnız `pal::pal_idx_finish_neon_fn(NeonToken)` ile verilir).
pub(super) fn pal_idx_finish_entry(
    dst: &mut [u8],
    src: &[u8],
    bw: usize,
    bh: usize,
    w: usize,
    h: usize,
) {
    // SAFETY: (1) ISA: Bu işlev işaretçisi yalnız `NeonToken` kanıtıyla (`pal::pal_idx_finish_neon_fn`) dışarı verilir; `aarch64` üzerinde NEON etkinliği doğrulanmıştır. (2) sınır: `pal_idx_finish_dispatch_neon` `%100 #![forbid(unsafe_code)]` güvenli Rust dilim sözleşmesiyle çalışır. (3) hizalama/takma ad: Yalnız güvenli `&mut [u8]` ve `&[u8]` referansları iletilir.
    unsafe {
        super::pal::pal_idx_finish_dispatch_neon(dst, src, bw, bh, w, h);
    }
}

/// `refmvs/splat_mv` giriş köprüsü (yalnız `refmvs::splat_mv_neon_fn(NeonToken)` ile verilir).
pub(super) fn splat_mv_entry(
    grid: &mut [RefBlock],
    iw4: usize,
    rmv: RefBlock,
    bx4: usize,
    by4: usize,
    bw4: usize,
    bh4: usize,
) {
    // SAFETY: (1) ISA: Bu işlev işaretçisi yalnız `NeonToken` kanıtıyla (`refmvs::splat_mv_neon_fn`) dışarı verilir; `aarch64` üzerinde NEON etkinliği doğrulanmıştır. (2) sınır: `splat_mv_dispatch_neon` `%100 #![forbid(unsafe_code)]` güvenli Rust dilim sözleşmesiyle çalışır. (3) hizalama/takma ad: Yalnız güvenli `&mut [RefBlock]` referansı iletilir.
    unsafe {
        super::refmvs::splat_mv_dispatch_neon(grid, iw4, rmv, bx4, by4, bw4, bh4);
    }
}

/// `refmvs/save_tmvs` giriş köprüsü (yalnız `refmvs::save_tmvs_neon_fn(NeonToken)` ile verilir).
#[allow(clippy::too_many_arguments)]
pub(super) fn save_tmvs_entry(
    rp: &mut [TemporalBlock],
    stride: usize,
    grid: &[RefBlock],
    iw4: usize,
    ref_sign: &[u8; 7],
    col_end8: usize,
    row_end8: usize,
    col_start8: usize,
    row_start8: usize,
) {
    // SAFETY: (1) ISA: Bu işlev işaretçisi yalnız `NeonToken` kanıtıyla (`refmvs::save_tmvs_neon_fn`) dışarı verilir; `aarch64` üzerinde NEON etkinliği doğrulanmıştır. (2) sınır: `save_tmvs_dispatch_neon` `%100 #![forbid(unsafe_code)]` güvenli Rust dilim sözleşmesiyle çalışır. (3) hizalama/takma ad: Yalnız güvenli `&mut [TemporalBlock]` ve `&[RefBlock]` referansları iletilir.
    unsafe {
        super::refmvs::save_tmvs_dispatch_neon(
            rp, stride, grid, iw4, ref_sign, col_end8, row_end8, col_start8, row_start8,
        );
    }
}

/// `refmvs/load_tmvs` giriş köprüsü (yalnız `refmvs::load_tmvs_neon_fn(NeonToken)` ile verilir).
#[allow(clippy::too_many_arguments)]
pub(super) fn load_tmvs_entry(
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
    // SAFETY: (1) ISA: Bu işlev işaretçisi yalnız `NeonToken` kanıtıyla (`refmvs::load_tmvs_neon_fn`) dışarı verilir; `aarch64` üzerinde NEON etkinliği doğrulanmıştır. (2) sınır: `load_tmvs_dispatch_neon` `%100 #![forbid(unsafe_code)]` güvenli Rust dilim sözleşmesiyle çalışır. (3) hizalama/takma ad: Yalnız güvenli dilim referansları iletilir.
    unsafe {
        super::refmvs::load_tmvs_dispatch_neon(
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
    }
}
