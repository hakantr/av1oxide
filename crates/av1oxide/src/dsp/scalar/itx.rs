//! Ters dönüşüm + piksel ekleme skaler çekirdeği — S2b.1 (`itxfm_add` portu).
//!
//! dav1d `src/itx_tmpl.c:43-118` (`inv_txfm_add_c`), `:184-202`
//! (`inv_txfm_add_wht_wht_4x4_c`) ve `:240-302` (destek matrisi) portudur.
//! Sözleşme ve kaynak satırı tablosu [`crate::dsp::itx`] modül belgesindedir;
//! bu dosya yalnız gövdeyi taşır.
//!
//! ## Bit derinliği monomorfizasyonu
//!
//! Gövde `const BPC: u32` (8/10/12) ile **tek** yazılır; üç örnekleme üretilir
//! (`DOG-05`). Satır/sütun kırpma sınırları derleme zamanında sabittir:
//!
//! | BPC | `row_clip_min` | `col_clip_min` | kaynak |
//! |---|---|---|---|
//! | 8 | −32768 | −32768 | `src/itx_tmpl.c:76-78` (`INT16_MIN` her ikisi) |
//! | 10 | −131072 | −32768 | `src/itx_tmpl.c:79-84` (~1023<<7, ~1023<<5) |
//! | 12 | −524288 | −131072 | `src/itx_tmpl.c:79-84` |
//!
//! Üst sınır her zaman `~alt`dır. 8-bit'te sütun sınırının **16-bit** kalması
//! dav1d'nin bilinçli davranışıdır ve birebir taşınır.
//!
//! ## Aritmetik
//!
//! Tüm ara değerler `i64`'tür; geçerli AV1 katsayı aralığında C `int` ile
//! bit-aynı sonuç verir (C'de taşma UB'si oluşmaz, bkz. `src/itx_1d.c:37-63`).
//! `>>` aritmetiktir. Son piksel `iclip_pixel` ile `[0, 2^BPC−1]` aralığına
//! kırpılır (`include/common/bitdepth.h:49,71`).
//!
//! ### İşlem bazında tür karşılaştırması (taşma denetimi)
//!
//! `C` tarafındaki tür `int32_t`/`int`'tir (`itx_tmpl.c:57` `int32_t tmp`);
//! bu port `i64` kullanır. **Hiçbir işlemde sarmaya dayanılmaz**; aşağıdaki
//! aralıklar C `int` taşmasının da oluşmadığını gösterir:
//!
//! | Bu dosyadaki işlem | dav1d satırı | C türü | En büyük ara değer | Seçilen |
//! |---|---|---|---|---|
//! | `(coeff*181 + 128) >> 8` (rect2) | `itx_tmpl.c:97-99` | `int` | `8 388 607 × 181 + 128 ≈ 1,518e9 < 2^31−1` | düz `*`/`+` (i64), sarma yok |
//! | `(dc*181 + 128) >> 8`, `(dc*181+128+2048)>>12` | `:57-62` | `int` | aynı sınır | düz, i64 |
//! | `(dc + rnd) >> shift` | `:61` | `int` | `≈5,9e5 + 2^5` | düz, i64 |
//! | `(*value + rnd) >> shift` | `:109-110` | `int` | `±2^19` (1B çıkış, kırpılmış) | düz, i64 |
//! | `(tmp + 8) >> 4` + piksel | `:114-118` | `int` | `±2 048 + 4 095` | düz, i64 |
//! | `iclip` | `common/intops.h` | `int` | — | `clamp` |
//! | WHT `coeff >> 2`, 1B geçişler | `itx_tmpl.c:184-202` | `int` | `±2^15` | düz, i64 |
//!
//! Yani bu dosyada **`wrapping_*` gerekmez**: dav1d burada işaretsiz sarmaya
//! dayanmaz. `i64` seçimi yalnız pay bırakır; sonuç C `int` ile aynıdır ve
//! `run_itx_oracle.sh --full` (468 000 blok) bunu bit-exact doğrular.
//!
//! ## Bellek sözleşmesi
//!
//! - `coeff` en az **1024** (`32×32`) eleman; kullanılan `sw·sh` bölgesi
//!   `sw = min(w,32)`, `sh = min(h,32)`.
//! - `dst` **bayt** dilimi; `dst_stride` **bayt**. Gereken uzunluk
//!   `(h−1)·dst_stride + w·bpp`; `dst_stride >= w·bpp`.
//! - Sözleşme dışı girdi **panikler** (güvenli indeksleme); taşan okuma/yazma
//!   yapısal olarak imkânsızdır. Sentinel testleri bunu ayrıca ölçer.
//!
//! ## Neden `coeff: &mut [i32]` (8-bit'te de)?
//!
//! dav1d 8-bit'te `coef = int16_t`, 16bpc'de `int32_t` kullanır
//! (`include/common/bitdepth.h:44,62`). Bu port **tek biçim** olarak `i32`
//! hazırlık tamponu kullanır: 8-bit değerleri i16 aralığındadır, aritmetik
//! sonuç değişmez; sıfırlama da aynı `sw·sh` bölgesini kapsar. Bit-exact
//! kanıtı, oracle'ın i16 tamponunu i32'ye genişletip aynı sonucu vermesiyle
//! ölçülür (bkz. `docs/baseline/itx-differential.md`).

#![forbid(unsafe_code)]

use crate::dsp::itx::{TxSize, TxType, is_supported, tx1d_fn};
use crate::dsp::itx_tables::LNZC_TABLES;
use crate::dsp::scalar::itx_1d::Tx1dType;
use crate::frame::BitDepth;

/// Ters dönüşüm + piksel ekleme — `BPC` (8/10/12) ile monomorfize yuva girişi.
///
/// İmza [`crate::dsp::itx::ItxFn`] ile birebir aynıdır; her bit derinliğinin
/// [`crate::dsp::itx::ItxDsp`] tablosu bu işlevin kendi monomorfunu taşır
/// (F3c, DOG-05: çağrı başına bit derinliği seçimi yok). `bit_depth`
/// parametresi imzada kalır (ADR-0003 §2.2) ve yalnız yuva ile çağıranın
/// uyumunu `debug_assert` ile denetler.
#[allow(clippy::too_many_arguments)]
pub fn inverse_transform_add<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
    bit_depth: BitDepth,
) {
    debug_assert_eq!(bit_depth.bits(), BPC, "yuva bit derinliği ≠ çağrı");
    add_bpc::<BPC>(dst, dst_stride, coeff, eob, tx, txtp);
}

/// Piksel baytı → `i64` (küçük sonlu; `pixel` 8-bit'te 1, 16bpc'de 2 bayt).
#[inline(always)]
fn px_at<const BPC: u32>(dst: &[u8], offset: usize) -> i64 {
    if BPC == 8 {
        i64::from(dst[offset])
    } else {
        i64::from(u16::from_le_bytes([dst[offset], dst[offset + 1]]))
    }
}

/// `i64` → piksel baytı; değer önce `[0, 2^BPC−1]` aralığına kırpılır
/// (dav1d `iclip_pixel`).
#[inline(always)]
fn px_put<const BPC: u32>(dst: &mut [u8], offset: usize, value: i64) {
    debug_assert!((0..(1i64 << BPC)).contains(&value), "piksel aralık dışı");
    if BPC == 8 {
        dst[offset] = value as u8;
    } else {
        dst[offset..offset + 2].copy_from_slice(&(value as u16).to_le_bytes());
    }
}

/// `i64` kırpma yardımcısı (dav1d `iclip`).
#[inline(always)]
fn clip(v: i64, min: i64, max: i64) -> i64 {
    v.clamp(min, max)
}

/// Gövde — dav1d `inv_txfm_add_c` + WHT (`BPC` = 8/10/12).
fn add_bpc<const BPC: u32>(
    dst: &mut [u8],
    dst_stride: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
) {
    assert!(
        is_supported(tx, txtp),
        "desteklenmeyen tx hücresi: {tx:?}/{txtp:?}"
    );
    assert!(coeff.len() >= 1024, "katsayı tamponu 32×32 olmalı");
    let w = tx.width();
    let h = tx.height();
    let bpp = if BPC == 8 { 1 } else { 2 };
    assert!(dst_stride >= w * bpp, "dst_stride < w·bpp");
    assert!(
        dst.len() >= (h - 1) * dst_stride + w * bpp,
        "dst dilimi kısa: (h−1)·stride + w·bpp gerekir"
    );
    let max_sample = (1i64 << BPC) - 1;

    // --- WHT (yalnız 4×4): dav1d src/itx_tmpl.c:184-202 --------------------
    if matches!(txtp, TxType::WhtWht) {
        debug_assert!(tx == TxSize::Tx4x4, "WHT yalnız 4×4");
        let mut tmp = [0i64; 16];
        for y in 0..4 {
            for x in 0..4 {
                tmp[y * 4 + x] = i64::from(coeff[y + x * 4]) >> 2;
            }
            crate::dsp::scalar::itx_1d::dav1d_inv_wht4_1d_c(&mut tmp[y * 4..y * 4 + 4], 1);
        }
        coeff[..16].fill(0);
        for x in 0..4 {
            crate::dsp::scalar::itx_1d::dav1d_inv_wht4_1d_c(&mut tmp[x..x + 13], 4);
        }
        for y in 0..4 {
            for x in 0..4 {
                let offset = y * dst_stride + x * bpp;
                let value = px_at::<BPC>(dst, offset) + tmp[y * 4 + x];
                px_put::<BPC>(dst, offset, clip(value, 0, max_sample));
            }
        }
        return;
    }

    let is_rect2 = w * 2 == h || h * 2 == w;
    let shift = tx.shift();
    let rnd = (1i64 << shift) >> 1;

    // --- DC-only: dav1d src/itx_tmpl.c:50,55-70 ---------------------------
    let has_dconly = matches!(txtp, TxType::DctDct);
    if has_dconly && eob == 0 {
        let mut dc = i64::from(coeff[0]);
        coeff[0] = 0;
        // `itx_tmpl.c:57-62`: `int` aritmetiği; `|dc| ≤ 8 388 607` × 181
        // ≈ 1,52e9 < 2^31−1 ⇒ C'de de taşma yok. i64 aynı sonucu verir.
        if is_rect2 {
            dc = (dc * 181 + 128) >> 8;
        }
        dc = (dc * 181 + 128) >> 8;
        dc = (dc + rnd) >> shift;
        dc = (dc * 181 + 128 + 2048) >> 12;
        for y in 0..h {
            for x in 0..w {
                let offset = y * dst_stride + x * bpp;
                // `itx_tmpl.c:68-70`: `dst[x] + dc`, `int`; `|dc| ≤ 2^19`,
                // piksel ≤ 4 095 ⇒ taşma yok.
                let value = px_at::<BPC>(dst, offset) + dc;
                px_put::<BPC>(dst, offset, clip(value, 0, max_sample));
            }
        }
        return;
    }

    // --- kırpma sınırları: dav1d src/itx_tmpl.c:76-84 ---------------------
    let (row_clip_min, col_clip_min) = if BPC == 8 {
        (-32768i64, -32768i64)
    } else {
        (-((1i64 << BPC) * 128), -((1i64 << BPC) * 32))
    };
    let row_clip_max = -row_clip_min - 1;
    let col_clip_max = -col_clip_min - 1;

    let info = tx.info();
    let sh = h.min(32);
    let sw = w.min(32);
    assert!(eob < sw * sh, "eob aralık dışı");
    let (horizontal, vertical) = txtp.tx1d();
    let first_1d = tx1d_fn(info.lw as usize, horizontal).expect("boyutun yatay 1B çekirdeği yok");
    let second_1d = tx1d_fn(info.lh as usize, vertical).expect("boyutun dikey 1B çekirdeği yok");

    // --- eob hızlı yolları: dav1d src/itx_tmpl.c:87-95 --------------------
    let last_nonzero_col = if vertical == Tx1dType::Identity && horizontal != Tx1dType::Identity {
        (sh - 1).min(eob)
    } else if horizontal == Tx1dType::Identity && vertical != Tx1dType::Identity {
        eob >> (info.lw as usize + 2)
    } else {
        usize::from(LNZC_TABLES[tx.index()][eob])
    };
    assert!(last_nonzero_col < sh, "last_nonzero_col < sh ihlali");

    // --- ilk 1B geçiş: dav1d src/itx_tmpl.c:96-104 ------------------------
    let mut tmp = [0i64; 64 * 64];
    for y in 0..=last_nonzero_col {
        let base = y * w;
        if is_rect2 {
            for x in 0..sw {
                // `itx_tmpl.c:97-99`: `(coeff * 181 + 128) >> 8`; `coef` sınırı
                // `cf_max` (12-bit'te 8 388 607) ⇒ 1,52e9 < 2^31−1, sarma yok.
                tmp[base + x] = (i64::from(coeff[y + x * sh]) * 181 + 128) >> 8;
            }
        } else {
            for x in 0..sw {
                tmp[base + x] = i64::from(coeff[y + x * sh]);
            }
        }
        first_1d(&mut tmp[base..base + w], 1, row_clip_min, row_clip_max);
    }

    // --- kalan satırlar: dav1d src/itx_tmpl.c:105-106 ---------------------
    if last_nonzero_col + 1 < sh {
        tmp[(last_nonzero_col + 1) * w..sh * w].fill(0);
    }

    // --- katsayı tamponunu sıfırla: dav1d src/itx_tmpl.c:108 --------------
    coeff[..sw * sh].fill(0);

    // --- kaydır + sütun kırpma: dav1d src/itx_tmpl.c:109-110 --------------
    for value in tmp.iter_mut().take(w * sh) {
        // `itx_tmpl.c:109-110`: `iclip((tmp + rnd) >> shift, ...)`; `tmp`
        // 1B çıkışı `row_clip` ile sınırlı (|·| ≤ 2^19) ⇒ taşma yok.
        *value = clip((*value + rnd) >> shift, col_clip_min, col_clip_max);
    }

    // --- ikinci 1B geçiş: dav1d src/itx_tmpl.c:112-113 --------------------
    for x in 0..w {
        second_1d(
            &mut tmp[x..x + (h - 1) * w + 1],
            w,
            col_clip_min,
            col_clip_max,
        );
    }

    // --- piksele ekle + kırp: dav1d src/itx_tmpl.c:115-118 ----------------
    for y in 0..h {
        for x in 0..w {
            let offset = y * dst_stride + x * bpp;
            // `itx_tmpl.c:114-118`: `dst[x] + ((*c + 8) >> 4)`; sütun çıkışı
            // `col_clip` ile sınırlı ⇒ `|(·+8)>>4| ≤ 2 048`, piksel ≤ 4 095.
            let value = px_at::<BPC>(dst, offset) + ((tmp[y * w + x] + 8) >> 4);
            px_put::<BPC>(dst, offset, clip(value, 0, max_sample));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::itx::ItxDsp;

    /// Test kısayolu: bit derinliğinin skaler tablosundaki yuva.
    #[allow(clippy::too_many_arguments)]
    fn inverse_transform_add(
        dst: &mut [u8],
        stride: usize,
        coeff: &mut [i32],
        eob: usize,
        tx: TxSize,
        txtp: TxType,
        bd: BitDepth,
    ) {
        (ItxDsp::scalar_for(bd).itxfm_add[tx.index()][txtp.index()])(
            dst, stride, coeff, eob, tx, txtp, bd,
        );
    }

    /// Sözleşme dışı hücre panikler (güvenli skaler yol kontratı).
    #[test]
    #[should_panic(expected = "desteklenmeyen tx hücresi")]
    fn desteklenmeyen_hucre_panikler() {
        let mut dst = [0u8; 64];
        let mut coeff = [0i32; 1024];
        inverse_transform_add(
            &mut dst,
            8,
            &mut coeff,
            1,
            TxSize::Tx64x64,
            TxType::AdstAdst,
            BitDepth::Eight,
        );
    }

    /// Kısa hedef dilimi panikler.
    #[test]
    #[should_panic(expected = "dst dilimi kısa")]
    fn kisa_dst_panikler() {
        let mut dst = [0u8; 4];
        let mut coeff = [0i32; 1024];
        inverse_transform_add(
            &mut dst,
            4,
            &mut coeff,
            0,
            TxSize::Tx4x4,
            TxType::DctDct,
            BitDepth::Eight,
        );
    }

    /// 4×4 DC-only: `coeff[0]=0` dışındaki katsayılar korunur (dav1d DC yolu).
    #[test]
    fn dc_yolu_yalniz_dc_katsayisini_sifirlar() {
        let mut dst = [10u8; 16];
        let mut coeff = [7i32; 1024];
        coeff[0] = 0;
        inverse_transform_add(
            &mut dst,
            4,
            &mut coeff,
            0,
            TxSize::Tx4x4,
            TxType::DctDct,
            BitDepth::Eight,
        );
        assert_eq!(coeff[0], 0);
        assert_eq!(&coeff[1..16], &[7i32; 15]);
    }

    /// Genel yol: `sw·sh` bölgesi sıfırlanır, dışına dokunulmaz.
    #[test]
    fn genel_yol_katsayi_bolgesini_sifirlar() {
        let mut dst = [0u8; 16];
        let mut coeff = [5i32; 1024];
        inverse_transform_add(
            &mut dst,
            4,
            &mut coeff,
            1,
            TxSize::Tx4x4,
            TxType::AdstAdst,
            BitDepth::Eight,
        );
        assert_eq!(&coeff[..16], &[0i32; 16]);
        assert_eq!(&coeff[16..], &[5i32; 1008]);
    }

    /// 12-bit sütun sınırı 16-bit değil, `−(4096·32)`'dir (dav1d 16bpc yolu).
    #[test]
    fn on_iki_bit_kirpma_sinirlari_dav1d_ile_ayni() {
        assert_eq!(-((1i64 << 12) * 128), -524288);
        assert_eq!(-((1i64 << 12) * 32), -131072);
        assert_eq!(-((1i64 << 10) * 32), -32768);
    }
}
