//! Temporal hareket vektörü projeksiyonu — dav1d `src/refmvs.c` veri modeli.
//!
//! **F3c:** `Dav1dRefmvsDSPContext`'in üç çekirdeği ([`save_tmvs`],
//! [`load_tmvs`], [`splat_mv`]) yuvalı DSP ağacına taşındı:
//! [`crate::dsp::refmvs`] (sözleşme, `RefmvsDsp`) ve
//! [`crate::dsp::scalar::refmvs`] (gövdeler). Bu modülde veri türleri,
//! [`mv_projection`] ve `mfmv` planı kalır; aşağıdaki tablo kaynak eşlemesidir.
//!
//! [`save_tmvs`]: crate::dsp::scalar::refmvs::save_tmvs
//! [`load_tmvs`]: crate::dsp::scalar::refmvs::load_tmvs
//! [`splat_mv`]: crate::dsp::scalar::refmvs::splat_mv
//!
//! ## Sorumluluk
//!
//! Bu modül `Dav1dRefmvsDSPContext`'in üç çekirdeğini birebir gerçekleştirir:
//!
//! | Rust | dav1d | Görev |
//! |---|---|---|
//! | [`save_tmvs`] | `save_tmvs_c` (`refmvs.c:792-830`) | kare sonunda 4×4 ızgarayı 8×8 temporal bloğa indirger |
//! | [`load_tmvs`] | `load_tmvs_c` (`refmvs.c:719-790`) | önceki karelerin MV'lerini geçerli kareye projekte eder |
//! | [`splat_mv`] | `splat_mv_c` (`refmvs.c:933-941`) | çözülen bloğun MV'sini ızgaraya yazar |
//! | [`mv_projection`] | `mv_projection` (`refmvs.c:176-192`) | AV1 §7.9.3 projeksiyonu |
//!
//! ## Neden ayrı modül?
//!
//! Uzamsal aday taraması ([`crate::decode::refmvs`]) kare **içi** durumu
//! kullanır; buradaki üç çekirdek ise **kareler arası** durumu (kaydedilmiş
//! MV ızgarası, `mfmv` planı) işler ve dav1d'de ayrı bir DSP tablosudur
//! (`Dav1dRefmvsDSPContext`). Ayrı tutmak, çekirdeklerin dav1d C'siyle
//! **hücre düzeyinde** karşılaştırılmasını (bkz.
//! `scripts/oracle/refmvs_dsp_oracle.c`,
//! `crates/av1oxide/tests/refmvs_dsp_oracle.rs`) ve kare düzeyi koddan
//! bağımsız sınanmasını sağlar.
//!
//! ## Veri modeli
//!
//! - [`TemporalBlock`] = dav1d `refmvs_temporal_block` (PACKED, 5 bayt):
//!   `mv` + `ref` (0 = kayıt yok).
//! - `rp`: kare başına bir kez ayrılan, `rp_stride` sütunlu 8×8 ızgarası;
//!   `save_tmvs` buraya yazar, sonraki kare `load_tmvs` ile **okur**.
//! - `rp_proj`: geçerli kare için projeksiyon hedefi (tile satırı başına
//!   16 satır).
//!
//! ## Tuzaklar (dav1d kaynağından)
//!
//! 1. `save_tmvs` her 8×8 hücrede **tek** 4×4 bloğu okur: `(2y+1, 2x+1)` —
//!    hücrenin sağ-alt 4×4'ü (`refmvs.c:798-800`).
//! 2. `save_tmvs` **ters yönlü** tercih yapar: önce `ref[1]` (compound'un
//!    ikinci referansı), sonra `ref[0]`; her ikisi de `ref_sign` ile
//!    süzülür (`refmvs.c:803-824`).
//! 3. `load_tmvs` yalnız `[row_start8, row_end8)` satırlarını **temizler**
//!    ve yazar; `col_start8`/`col_end8` dışına taşmaz (`refmvs.c:735-741`).
//! 4. `load_tmvs`'in koşu (run) döngüsü `x_sb_align`'ı **her** hücre için
//!    yeniden hesaplar (`refmvs.c:770`); tek seferlik önbellek yanlış olur.
//! 5. `mv_projection` çarpımı 32-bit **sarmalı** yapar: `num * div_mult[den]`
//!    ile `mv * frac` C'de taşabilir; iki taraf da aynı sarmayı üretmelidir.
//!
//! ## Kapsam
//!
//! Çekirdekler `#![forbid(unsafe_code)]` altında **düz dilim** API'siyle
//! yazılmıştır; üretim crate'ine bağımlılık yoktur, tek iş parçacıklıdır.

#![forbid(unsafe_code)]

use crate::decode::refmvs::Mv;

/// `INVALID_MV` (`refmvs.h:38`) — geçersiz MV işareti.
pub const INVALID_MV: i16 = i16::MIN;
/// `INVALID_REF2CUR` (`refmvs.h:39`).
pub const INVALID_REF2CUR: i8 = -32;
/// `refmvs_temporal_block` — 8×8 hücre başına kayıt (dav1d'de PACKED 5 bayt, Rust'ta `#[repr(C)]` 6 bayt).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TemporalBlock {
    /// Hareket vektörü (1/8 piksel).
    pub mv: Mv,
    /// Referans indeksi + 1 (`0` = kayıt yok).
    pub ref_: u8,
}

const _: () = assert!(core::mem::size_of::<TemporalBlock>() == 6);

impl TemporalBlock {
    /// Kayıt yok mu? (dav1d'de `mv.n == INVALID_MV` ya da `ref == 0`.)
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.ref_ == 0 || self.mv.y == INVALID_MV
    }
}

/// `mv_projection` (`refmvs.c:176-192`) — AV1 §7.9.3.
///
/// `den` 1..=31, `num` -31..=31 olmalıdır. Çarpımlar C'deki 32-bit sarmalı
/// aritmetiğiyle **birebir** aynıdır ([`i32::wrapping_mul`]).
#[must_use]
pub fn mv_projection(mv: Mv, num: i32, den: i32) -> Mv {
    /// `div_mult[32]` (`refmvs.c:177-181`).
    const DIV_MULT: [i32; 32] = [
        0, 16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638, 1489, 1365, 1260, 1170,
        1092, 1024, 963, 910, 862, 819, 780, 744, 712, 682, 655, 630, 606, 585, 564, 546, 528,
    ];
    debug_assert!((1..32).contains(&den));
    debug_assert!((-32..32).contains(&num));
    let frac = num.wrapping_mul(DIV_MULT[den as usize]);
    let y = i32::from(mv.y).wrapping_mul(frac);
    let x = i32::from(mv.x).wrapping_mul(frac);
    // `(y + 8192 + (y >> 31)) >> 14` — C'de `>>` negatifte aritmetiktir.
    let ry = (y.wrapping_add(8192).wrapping_add(y >> 31)) >> 14;
    let rx = (x.wrapping_add(8192).wrapping_add(x >> 31)) >> 14;
    Mv {
        y: ry.clamp(-0x3fff, 0x3fff) as i16,
        x: rx.clamp(-0x3fff, 0x3fff) as i16,
    }
}

/// `mfmv` planı — `dav1d_refmvs_init_frame`'in ürettiği türetilmiş alanlar.
///
/// `load_tmvs` yalnız bu alanları ve referans ızgaralarını kullanır
/// (`refmvs.c:744-749`).
#[derive(Clone, Copy, Debug, Default)]
pub struct MfmvPlan {
    /// `n_mfmvs` — projeksiyon yapılacak referans sayısı (0..=3).
    pub n_mfmvs: usize,
    /// `mfmv_ref[n]` — 0..6 referans indeksi.
    pub mfmv_ref: [u8; 3],
    /// `mfmv_ref2cur[n]` — `INVALID_REF2CUR` ise atlanır.
    pub mfmv_ref2cur: [i8; 3],
    /// `mfmv_ref2ref[n][m]` — referansın referansına POC farkı (0 = yok).
    pub mfmv_ref2ref: [[u8; 7]; 3],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::refmvs::RefBlock;
    use crate::dsp::scalar::refmvs::{load_tmvs, save_tmvs, splat_mv};

    #[test]
    fn projeksiyon_sinir_degerleri() {
        // num = 0 → sıfır.
        assert_eq!(mv_projection(Mv { y: 1234, x: -99 }, 0, 5), Mv::ZERO);
        // den = 1, num = 1 → 16384/16384 = 1 → aynı MV (yuvarlanmış).
        let mv = Mv { y: 100, x: -100 };
        assert_eq!(mv_projection(mv, 1, 1), mv);
        // num == den → aynı MV (yuvarlama ile).
        assert_eq!(
            mv_projection(Mv { y: 8000, x: -8000 }, 16, 16),
            Mv { y: 8000, x: -8000 }
        );
        // Kırpma: ±0x3fff (taşma olmadan).
        let big = mv_projection(
            Mv {
                y: 16000,
                x: -16000,
            },
            31,
            16,
        );
        assert_eq!(big.y, 0x3fff);
        assert_eq!(big.x, -0x3fff);
    }

    #[test]
    fn save_tmvs_ref1_oncelikli() {
        let iw4 = 4;
        let mut grid = vec![RefBlock::default(); iw4 * 4];
        let b = RefBlock {
            mv: [Mv { y: 8, x: 16 }, Mv { y: -24, x: 32 }],
            ref_pair: [1, 2],
            bs: 4, // BS_8x8 → bw8 = (2+1)>>1 = 1
            ..RefBlock::default()
        };
        // y=0, x=0 hücresi → mutlak 4×4 satırı 2*0+1 = 1, sütunu 1.
        grid[iw4 + 1] = b;
        let mut rp = vec![TemporalBlock::default(); 8 * 4];
        let sign = [1u8, 1, 0, 0, 0, 0, 0];
        save_tmvs(&mut rp, 8, &grid, iw4, &sign, 2, 1, 0, 0);
        assert_eq!(rp[0].ref_, 2);
        assert_eq!(rp[0].mv, Mv { y: -24, x: 32 });
    }

    #[test]
    fn save_tmvs_isaretsiz_ve_buyuk_mv_yazilmaz() {
        let iw4 = 4;
        let mut grid = vec![RefBlock::default(); iw4 * 4];
        let mut b = RefBlock {
            mv: [Mv { y: 4096, x: 0 }, Mv::ZERO],
            ref_pair: [1, -1],
            bs: 4,
            ..RefBlock::default()
        };
        grid[iw4 + 1] = b;
        let mut rp = vec![TemporalBlock::default(); 8 * 4];
        let sign = [1u8, 1, 0, 0, 0, 0, 0];
        // |mv| >= 4096 → kayıt yok, sıfır yazılır.
        save_tmvs(&mut rp, 8, &grid, iw4, &sign, 2, 1, 0, 0);
        assert_eq!(rp[0], TemporalBlock::default());
        // ref_sign = 0 → yine yazılmaz.
        b.mv = [Mv { y: 8, x: 8 }, Mv::ZERO];
        grid[iw4 + 1] = b;
        let sign0 = [0u8; 7];
        save_tmvs(&mut rp, 8, &grid, iw4, &sign0, 2, 1, 0, 0);
        assert_eq!(rp[0], TemporalBlock::default());
    }

    #[test]
    fn load_tmvs_bolge_temizler_ve_projekte_eder() {
        let stride = 8;
        let iw8 = 8;
        let ih8 = 8;
        // Referans ızgarası: (y=0,x=0) → mv (64,0), ref 1.
        let mut r = vec![TemporalBlock::default(); stride * ih8];
        r[0] = TemporalBlock {
            mv: Mv { y: 64, x: 0 },
            ref_: 1,
        };
        let mut proj = vec![
            TemporalBlock {
                mv: Mv { y: 7, x: 7 },
                ref_: 3
            };
            16 * stride
        ];
        let mut plan = MfmvPlan {
            n_mfmvs: 1,
            ..MfmvPlan::default()
        };
        plan.mfmv_ref[0] = 4;
        plan.mfmv_ref2cur[0] = 8;
        plan.mfmv_ref2ref[0][0] = 8;
        let refs: [Option<&[TemporalBlock]>; 7] = [None, None, None, None, Some(&r), None, None];
        load_tmvs(&mut proj, stride, iw8, ih8, &plan, &refs, 0, 0, iw8, 0, 8);
        // mv_projection(mv_y=64, num=8, den=8): frac = 8*2048 = 16384;
        // y = 64*16384 = 1048576; (1048576+8192)>>14 = 64;
        // offset.y = 64 >> 6 = 1 → pos_y = 0 + 1 = 1.
        assert_eq!(proj[stride].ref_, 8);
        assert_eq!(proj[stride].mv, Mv { y: 64, x: 0 });
        // Bölge dışı (satır 8) dokunulmamış olmalı.
        assert_eq!(proj[8 * stride].ref_, 3);
    }

    #[test]
    fn splat_mv_blogu_doldurur() {
        let iw4 = 8;
        let ih4 = 8;
        let mut grid = vec![RefBlock::default(); iw4 * ih4];
        let b = RefBlock {
            mv: [Mv { y: 3, x: 4 }, Mv::ZERO],
            ref_pair: [2, -1],
            bs: 4,
            ..RefBlock::default()
        };
        splat_mv(&mut grid, iw4, b, 2, 3, 2, 3);
        for y in 3..6 {
            for x in 2..4 {
                assert_eq!(grid[y * iw4 + x].ref_pair, [2, -1]);
            }
        }
        // Dışarısı değişmedi.
        assert_eq!(grid[2 * iw4 + 2].ref_pair, [0, -1]);
    }
}
