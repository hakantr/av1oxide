//! `aarch64` CPU özellik tespiti ve `dav1d` `src/arm/cpu.{h,c}` bayrak paritesi (F5a / P51).
//!
//! ## `dav1d` Paritesi (`src/arm/cpu.h:31-37`, `src/arm/cpu.c:38-96`)
//!
//! | `dav1d` sabiti | Bit değeri (`dav1d_arm_bits`) | [`CpuFlags`] karşılığı | `rustc 1.100.0` `core::arch::aarch64` durumu |
//! |---|---:|---|---|
//! | `DAV1D_ARM_CPU_FLAG_NEON` | `1 << 0` (`0x01`) | [`CpuFlags::NEON`] | **Kararlı** ([`NeonToken`](super::NeonToken)) |
//! | `DAV1D_ARM_CPU_FLAG_DOTPROD` | `1 << 1` (`0x02`) | [`CpuFlags::DOTPROD`] | **Kararlı** ([`DotprodToken`](super::DotprodToken)) |
//! | `DAV1D_ARM_CPU_FLAG_I8MM` | `1 << 2` (`0x04`) | [`CpuFlags::I8MM`] | Kararsız (`E0658: stdarch_neon_i8mm`, `BLOCKED`) |
//! | `DAV1D_ARM_CPU_FLAG_SVE` | `1 << 3` (`0x08`) | [`CpuFlags::SVE`] | Kararsız (`E0658: stdarch_aarch64_sve`, `BLOCKED`) |
//! | `DAV1D_ARM_CPU_FLAG_SVE2` | `1 << 4` (`0x10`) | [`CpuFlags::SVE2`] | Kararsız (`E0658: stdarch_aarch64_sve2`, `BLOCKED`) |
//!
//! - `aarch64` hedeflerinde (`aarch64-unknown-linux-gnu`, `aarch64-apple-darwin` vb.)
//!   `std::arch::is_aarch64_feature_detected!` (derleme zamanı + işletim sistemi
//!   `AT_HWCAP`/`AT_HWCAP2` veya `sysctlbyname`) kullanılır.
//! - `arm32` (`armv7-unknown-linux-gnueabihf`) üzerinde `core::arch::arm` NEON
//!   kararsız (`stdarch_arm_neon_intrinsics`) olduğundan tespit devreye girmez ve
//!   [`CpuFlags::NONE`] döner (`KARAR 1` / `DSP-06`).

#![forbid(unsafe_code)]

use super::CpuFlags;

/// `dav1d` `src/arm/cpu.h:31-37` `DAV1D_ARM_CPU_FLAG_NEON` biti.
pub const DAV1D_ARM_CPU_FLAG_NEON: u32 = 1 << 0;
/// `dav1d` `src/arm/cpu.h:31-37` `DAV1D_ARM_CPU_FLAG_DOTPROD` biti.
pub const DAV1D_ARM_CPU_FLAG_DOTPROD: u32 = 1 << 1;
/// `dav1d` `src/arm/cpu.h:31-37` `DAV1D_ARM_CPU_FLAG_I8MM` biti.
pub const DAV1D_ARM_CPU_FLAG_I8MM: u32 = 1 << 2;
/// `dav1d` `src/arm/cpu.h:31-37` `DAV1D_ARM_CPU_FLAG_SVE` biti.
pub const DAV1D_ARM_CPU_FLAG_SVE: u32 = 1 << 3;
/// `dav1d` `src/arm/cpu.h:31-37` `DAV1D_ARM_CPU_FLAG_SVE2` biti.
pub const DAV1D_ARM_CPU_FLAG_SVE2: u32 = 1 << 4;
/// `dav1d` ARM bayraklarının tamamı (`0x1f`).
pub const DAV1D_ARM_CPU_MASK_ALL: u32 = 0x1f;

/// Saf hesaplama için soyutlanmış `aarch64` özellik yapısı.
///
/// Birim testleri gerçek donanım olmadan tüm kombinasyonları doğrular.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArmCpuId {
    /// ARMv8-A Advanced SIMD (NEON).
    pub neon: bool,
    /// ARMv8.2-A Dot Product (`FEAT_DotProd`, `udot`/`sdot`).
    pub dotprod: bool,
    /// ARMv8.6-A Int8 Matrix Multiplication (`FEAT_I8MM`, `usdot`/`smmla`).
    pub i8mm: bool,
    /// Scalable Vector Extension (`FEAT_SVE`).
    pub sve: bool,
    /// Scalable Vector Extension 2 (`FEAT_SVE2`).
    pub sve2: bool,
}

impl ArmCpuId {
    /// Çalışma zamanında `aarch64` özelliklerini okur.
    #[cfg(all(feature = "std", target_arch = "aarch64", not(miri)))]
    #[must_use]
    pub fn read() -> Self {
        let neon = std::arch::is_aarch64_feature_detected!("neon");
        let dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
        let i8mm = std::arch::is_aarch64_feature_detected!("i8mm");
        let sve = std::arch::is_aarch64_feature_detected!("sve");
        let sve2 = std::arch::is_aarch64_feature_detected!("sve2");
        Self {
            neon,
            dotprod,
            i8mm,
            sve,
            sve2,
        }
    }

    /// Ham özelliklerden [`CpuFlags`] üretir (`dav1d` `dav1d_get_cpu_flags_arm` paritesi).
    ///
    /// NEON olmadan hiçbir üst ARM SIMD uzantısı (`DOTPROD`, `I8MM`, `SVE`, `SVE2`)
    /// etkinleştirilmez; `SVE` olmadan `SVE2` etkinleştirilmez (`dav1d`
    /// `src/arm/mc_init_tmpl.c:302-304`: `SVE2` dalı `NEON` bloğunun içindedir).
    #[must_use]
    pub const fn flags(self) -> CpuFlags {
        if !self.neon {
            return CpuFlags::NONE;
        }
        let mut out = CpuFlags::NEON;
        if self.dotprod {
            out = out.union(CpuFlags::DOTPROD);
        }
        if self.i8mm {
            out = out.union(CpuFlags::I8MM);
        }
        if self.sve {
            out = out.union(CpuFlags::SVE);
            if self.sve2 {
                out = out.union(CpuFlags::SVE2);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ArmCpuId, DAV1D_ARM_CPU_FLAG_DOTPROD, DAV1D_ARM_CPU_FLAG_I8MM, DAV1D_ARM_CPU_FLAG_NEON,
        DAV1D_ARM_CPU_FLAG_SVE, DAV1D_ARM_CPU_FLAG_SVE2, DAV1D_ARM_CPU_MASK_ALL,
    };
    use crate::cpu::CpuFlags;

    #[test]
    fn arm_bitleri_dav1d_cpu_h_ile_birebir_ayni() {
        assert_eq!(CpuFlags::NEON.dav1d_arm_bits(), DAV1D_ARM_CPU_FLAG_NEON);
        assert_eq!(
            CpuFlags::DOTPROD.dav1d_arm_bits(),
            DAV1D_ARM_CPU_FLAG_DOTPROD
        );
        assert_eq!(CpuFlags::I8MM.dav1d_arm_bits(), DAV1D_ARM_CPU_FLAG_I8MM);
        assert_eq!(CpuFlags::SVE.dav1d_arm_bits(), DAV1D_ARM_CPU_FLAG_SVE);
        assert_eq!(CpuFlags::SVE2.dav1d_arm_bits(), DAV1D_ARM_CPU_FLAG_SVE2);
        assert_eq!(CpuFlags::SSE2.dav1d_arm_bits(), 0);
        assert_eq!(CpuFlags::AVX2.dav1d_arm_bits(), 0);

        for bits in 0..=DAV1D_ARM_CPU_MASK_ALL {
            let flags = CpuFlags::from_dav1d_arm_bits(bits);
            assert_eq!(flags.dav1d_arm_bits(), bits);
            assert_eq!(flags.dav1d_x86_bits(), 0);
        }
        assert_eq!(CpuFlags::from_dav1d_arm_bits(u32::MAX), CpuFlags::ARM_DAV1D);
    }

    #[test]
    fn arm_cpuid_bayrak_hiyerarsisi() {
        // NEON kapalıysa üst uzantılar işaretlenmez.
        let no_neon = ArmCpuId {
            neon: false,
            dotprod: true,
            i8mm: true,
            sve: true,
            sve2: true,
        };
        assert_eq!(no_neon.flags(), CpuFlags::NONE);

        // Yalnız NEON (standart ARMv8.0-A tabanı).
        let neon_only = ArmCpuId {
            neon: true,
            ..ArmCpuId::default()
        };
        assert_eq!(neon_only.flags(), CpuFlags::NEON);
        assert_eq!(neon_only.flags().dav1d_arm_bits(), 0x01);

        // Apple Silicon / ARMv8.2+ (NEON + DOTPROD + I8MM, SVE yok).
        let apple_m = ArmCpuId {
            neon: true,
            dotprod: true,
            i8mm: true,
            sve: false,
            sve2: true, // SVE olmadan SVE2 açılmaz
        };
        assert_eq!(
            apple_m.flags(),
            CpuFlags::NEON | CpuFlags::DOTPROD | CpuFlags::I8MM
        );
        assert_eq!(apple_m.flags().dav1d_arm_bits(), 0x07);

        // Tam ARMv9-A (NEON + DOTPROD + I8MM + SVE + SVE2).
        let armv9 = ArmCpuId {
            neon: true,
            dotprod: true,
            i8mm: true,
            sve: true,
            sve2: true,
        };
        assert_eq!(armv9.flags(), CpuFlags::ARM_DAV1D);
        assert_eq!(armv9.flags().dav1d_arm_bits(), 0x1f);
    }
}
