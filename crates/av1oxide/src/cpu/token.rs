//! ISA yetki belirteçleri (F3b / P25, ADR-0003 §2.3).
//!
//! Bir belirteç, "bu süreçte bu ISA'nın gövdeleri çalıştırılabilir" kanıtıdır.
//! Yalnız iki koşul birlikte sağlandığında üretilir:
//!
//! 1. Çağıranın verdiği **izinli** küme (tespit ∩ kullanıcı maskesi,
//!    [`Config::cpu_mask`](crate::api::Config::cpu_mask)) ilgili bayrağı içerir;
//! 2. Bu makinenin **gerçek** tespiti ([`CpuFlags::detect`], dav1d anlamıyla
//!    işletim sistemi durumu dahil) aynı bayrağı içerir.
//!
//! İkinci koşul, belirtecin sahte bir bayrak kümesiyle üretilememesini sağlar:
//! crate içinden bile `Avx2Token::new(CpuFlags::AVX2)` AVX2'siz bir makinede
//! `None` döner. Kurucu `pub(crate)`'tir ve alanı gizlidir; crate dışından
//! belirteç üretilemez. Belirteçler `DspDispatch::select` kurulumunda üretilir
//! ve [`DspDispatch::avx2_token`](crate::dsp::DspDispatch::avx2_token) ile
//! yalnız okunur.
//!
//! ```compile_fail,E0423
//! // Crate dışından belirteç kurulamaz (gizli alan).
//! let _ = av1oxide::cpu::Avx2Token(());
//! ```
//!
//! ```compile_fail,E0624
//! // Kurucu crate dışına açık değildir.
//! let _ = av1oxide::cpu::Avx2Token::new(av1oxide::cpu::CpuFlags::AVX2);
//! ```
//!
//! ```compile_fail,E0624
//! // AVX-512 belirteci de aynı kurala bağlıdır.
//! let _ = av1oxide::cpu::Avx512IclToken::new(av1oxide::cpu::CpuFlags::AVX512ICL);
//! ```
//!
//! P30: SSSE3 ve SSE4.1 belirteçleri aynı kurala bağlıdır.
//!
//! ```compile_fail,E0423
//! let _ = av1oxide::cpu::Ssse3Token(());
//! ```
//!
//! ```compile_fail,E0624
//! let _ = av1oxide::cpu::Ssse3Token::new(av1oxide::cpu::CpuFlags::SSSE3);
//! ```
//!
//! ```compile_fail,E0423
//! let _ = av1oxide::cpu::Sse41Token(());
//! ```
//!
//! ```compile_fail,E0624
//! let _ = av1oxide::cpu::Sse41Token::new(av1oxide::cpu::CpuFlags::SSE4_1);
//! ```
//!
//! P43: SSE2 belirteci de aynı kurala bağlıdır.
//!
//! ```compile_fail,E0423
//! let _ = av1oxide::cpu::Sse2Token(());
//! ```
//!
//! ```compile_fail,E0624
//! let _ = av1oxide::cpu::Sse2Token::new(av1oxide::cpu::CpuFlags::SSE2);
//! ```
//!
//! P51 (F5a): ARM64 `NeonToken` ve `DotprodToken` belirteçleri de aynı kurala bağlıdır.
//!
//! ```compile_fail,E0423
//! let _ = av1oxide::cpu::NeonToken(());
//! ```
//!
//! ```compile_fail,E0624
//! let _ = av1oxide::cpu::NeonToken::new(av1oxide::cpu::CpuFlags::NEON);
//! ```
//!
//! ```compile_fail,E0423
//! let _ = av1oxide::cpu::DotprodToken(());
//! ```
//!
//! ```compile_fail,E0624
//! let _ = av1oxide::cpu::DotprodToken::new(av1oxide::cpu::CpuFlags::DOTPROD);
//! ```
//!
//! Belirteç sıfır boyutludur, `Copy`'dir ve iş parçacıkları arasında
//! taşınabilir (`Send` + `Sync`): kanıt süreç düzeyindedir.

#![forbid(unsafe_code)]

use super::CpuFlags;

/// SSE2 yetki belirteci (P43): dav1d anlamıyla SSE2 (CMOV + SSE + SSE2)
/// tespit edildi ve maske izin veriyor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sse2Token(());

impl Sse2Token {
    /// `allowed` SSE2 içeriyorsa **ve** bu makinede SSE2 tespit edildiyse.
    #[must_use]
    pub(crate) fn new(allowed: CpuFlags) -> Option<Self> {
        let ok = allowed.contains(CpuFlags::SSE2) && CpuFlags::detect().contains(CpuFlags::SSE2);
        ok.then_some(Self(()))
    }
}

/// SSSE3 yetki belirteci (P30): dav1d anlamıyla SSSE3 (SSE2 + SSE3 + SSSE3)
/// tespit edildi ve maske izin veriyor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ssse3Token(());

impl Ssse3Token {
    /// `allowed` SSSE3 içeriyorsa **ve** bu makinede SSSE3 tespit edildiyse.
    #[must_use]
    pub(crate) fn new(allowed: CpuFlags) -> Option<Self> {
        let ok = allowed.contains(CpuFlags::SSSE3) && CpuFlags::detect().contains(CpuFlags::SSSE3);
        ok.then_some(Self(()))
    }
}

/// SSE4.1 yetki belirteci (P30): dav1d anlamıyla SSE41 (SSSE3 + SSE4.1)
/// tespit edildi ve maske izin veriyor. dav1d düzey anlamında SSE41 düzeyi
/// SSSE3'ü içerir; tespit SSE41 bitini yalnız SSSE3 de varken verir
/// (`cpu::x86`), maske düzeyleri de SSSE3'ü birlikte açar ([`CpuLevel`]).
///
/// [`CpuLevel`]: super::CpuLevel
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sse41Token(());

impl Sse41Token {
    /// `allowed` SSE4.1 **ve** SSSE3 içeriyorsa ve bu makinede ikisi de
    /// tespit edildiyse.
    #[must_use]
    pub(crate) fn new(allowed: CpuFlags) -> Option<Self> {
        let gerek = CpuFlags::SSE4_1.union(CpuFlags::SSSE3);
        let ok = allowed.contains(gerek) && CpuFlags::detect().contains(gerek);
        ok.then_some(Self(()))
    }
}

/// AVX2 yetki belirteci: dav1d anlamıyla AVX2 (AVX + BMI1 + BMI2 + XCR0 YMM)
/// tespit edildi ve maske izin veriyor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Avx2Token(());

impl Avx2Token {
    /// `allowed` AVX2 içeriyorsa **ve** bu makinede AVX2 tespit edildiyse.
    #[must_use]
    pub(crate) fn new(allowed: CpuFlags) -> Option<Self> {
        let ok = allowed.contains(CpuFlags::AVX2) && CpuFlags::detect().contains(CpuFlags::AVX2);
        ok.then_some(Self(()))
    }
}

/// AVX512ICL yetki belirteci (F4 için yalnız tür; bu turda kullanıcısı yok).
///
/// dav1d anlamıyla AVX512ICL: AVX2 + XCR0 ZMM/OPMASK + 14 alt özellik.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Avx512IclToken(());

impl Avx512IclToken {
    /// `allowed` AVX512ICL içeriyorsa **ve** bu makinede AVX512ICL tespit edildiyse.
    #[must_use]
    pub(crate) fn new(allowed: CpuFlags) -> Option<Self> {
        let ok = allowed.contains(CpuFlags::AVX512ICL)
            && CpuFlags::detect().contains(CpuFlags::AVX512ICL);
        ok.then_some(Self(()))
    }
}

/// ARM64 NEON yetki belirteci (P51 / F5a): `aarch64` üzerinde NEON tespit
/// edildi ve maske izin veriyor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NeonToken(());

impl NeonToken {
    /// `allowed` NEON içeriyorsa **ve** bu makinede NEON tespit edildiyse.
    #[must_use]
    pub(crate) fn new(allowed: CpuFlags) -> Option<Self> {
        let ok = allowed.contains(CpuFlags::NEON) && CpuFlags::detect().contains(CpuFlags::NEON);
        ok.then_some(Self(()))
    }
}

/// ARM64 DOTPROD (`neon + dotprod`) yetki belirteci (P51 / F5a): `aarch64`
/// üzerinde hem NEON hem DOTPROD tespit edildi ve maske izin veriyor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DotprodToken(());

impl DotprodToken {
    /// `allowed` NEON **ve** DOTPROD içeriyorsa ve bu makinede ikisi de
    /// tespit edildiyse.
    #[must_use]
    pub(crate) fn new(allowed: CpuFlags) -> Option<Self> {
        let gerek = CpuFlags::NEON.union(CpuFlags::DOTPROD);
        let ok = allowed.contains(gerek) && CpuFlags::detect().contains(gerek);
        ok.then_some(Self(()))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Avx2Token, Avx512IclToken, DotprodToken, NeonToken, Sse2Token, Sse41Token, Ssse3Token,
    };
    use crate::cpu::{CpuFlags, CpuLevel};

    fn send_sync_copy<T: Send + Sync + Copy>() {}

    #[test]
    fn belirtecler_sifir_boyutlu_copy_send_sync() {
        assert_eq!(core::mem::size_of::<Sse2Token>(), 0);
        assert_eq!(core::mem::size_of::<Avx2Token>(), 0);
        assert_eq!(core::mem::size_of::<Avx512IclToken>(), 0);
        assert_eq!(core::mem::size_of::<Ssse3Token>(), 0);
        assert_eq!(core::mem::size_of::<Sse41Token>(), 0);
        assert_eq!(core::mem::size_of::<NeonToken>(), 0);
        assert_eq!(core::mem::size_of::<DotprodToken>(), 0);
        assert_eq!(core::mem::size_of::<Option<Avx2Token>>(), 1);
        assert_eq!(core::mem::size_of::<Option<NeonToken>>(), 1);
        assert_eq!(core::mem::size_of::<Option<DotprodToken>>(), 1);
        send_sync_copy::<Sse2Token>();
        send_sync_copy::<Avx2Token>();
        send_sync_copy::<Avx512IclToken>();
        send_sync_copy::<Ssse3Token>();
        send_sync_copy::<Sse41Token>();
        send_sync_copy::<NeonToken>();
        send_sync_copy::<DotprodToken>();
    }

    /// Altı dav1d düzeyi × bu makinenin tespiti: belirteç yalnız tespit ve
    /// düzey ilgili bayrağa izin verdiğinde üretilir.
    #[test]
    fn belirtecler_tespit_ve_maskeyle_uretilir() {
        let detected = CpuFlags::detect();
        for level in CpuLevel::ALL {
            let allowed = detected.intersection(level.allowed());
            assert_eq!(
                Sse2Token::new(allowed).is_some(),
                detected.contains(CpuFlags::SSE2) && level.allowed().contains(CpuFlags::SSE2),
                "SSE2 belirteci, düzey {}",
                level.name()
            );
            assert_eq!(
                Avx2Token::new(allowed).is_some(),
                detected.contains(CpuFlags::AVX2) && level.allowed().contains(CpuFlags::AVX2),
                "AVX2 belirteci, düzey {}",
                level.name()
            );
            assert_eq!(
                Avx512IclToken::new(allowed).is_some(),
                detected.contains(CpuFlags::AVX512ICL) && level == CpuLevel::Avx512Icl,
                "AVX512ICL belirteci, düzey {}",
                level.name()
            );
            assert_eq!(
                Ssse3Token::new(allowed).is_some(),
                detected.contains(CpuFlags::SSSE3) && level.allowed().contains(CpuFlags::SSSE3),
                "SSSE3 belirteci, düzey {}",
                level.name()
            );
            assert_eq!(
                Sse41Token::new(allowed).is_some(),
                detected.contains(CpuFlags::SSE4_1) && level.allowed().contains(CpuFlags::SSE4_1),
                "SSE4.1 belirteci, düzey {}",
                level.name()
            );
        }
        // Maske yok (dav1d `-1`) = tespit edilen küme.
        assert_eq!(
            Avx2Token::new(detected).is_some(),
            detected.contains(CpuFlags::AVX2)
        );
    }

    /// Sahte bayrak kümesi belirteç üretmez: gerçek tespitte olmayan bayrak
    /// verilse de sonuç tespite bağlıdır.
    #[test]
    fn sahte_bayrakla_belirtec_uretilmez() {
        let detected = CpuFlags::detect();
        let fake = CpuFlags::X86_DAV1D;
        assert_eq!(
            Avx2Token::new(fake).is_some(),
            detected.contains(CpuFlags::AVX2)
        );
        assert_eq!(
            Avx512IclToken::new(fake).is_some(),
            detected.contains(CpuFlags::AVX512ICL)
        );
        assert!(Avx2Token::new(CpuFlags::NONE).is_none());
        assert!(Avx512IclToken::new(CpuFlags::AVX2).is_none());
        // Kapatma maskesi AVX2'yi kapatıp AVX512ICL bitini bıraksa bile AVX2
        // belirteci üretilmez (her belirteç kendi bayrağına bakar).
        let only_icl = detected.difference(CpuFlags::AVX2);
        assert!(Avx2Token::new(only_icl).is_none());
    }

    /// P30 — zorlanmış düzeyler: maskede SSE4.1 kapalıyken [`Sse41Token`]
    /// üretilmez; SSSE3 kapalıyken iki belirteç de üretilmez. `allowed`'ı yok
    /// sayan (yalnız tespite bakan) bir kurucu bu testte düşer.
    #[test]
    fn p30_sse_belirtecleri_maskeye_uyar() {
        let detected = CpuFlags::detect();
        let sse41_kapali = detected.difference(CpuFlags::SSE4_1);
        assert!(Sse41Token::new(sse41_kapali).is_none());
        assert_eq!(
            Ssse3Token::new(sse41_kapali).is_some(),
            detected.contains(CpuFlags::SSSE3)
        );
        let ssse3_kapali = detected.difference(CpuFlags::SSSE3);
        assert!(Ssse3Token::new(ssse3_kapali).is_none());
        assert!(Sse41Token::new(ssse3_kapali).is_none());
        for level in [CpuLevel::None, CpuLevel::Sse2] {
            let allowed = detected.intersection(level.allowed());
            assert!(Ssse3Token::new(allowed).is_none(), "{}", level.name());
            assert!(Sse41Token::new(allowed).is_none(), "{}", level.name());
        }
        assert!(Sse2Token::new(detected.intersection(CpuLevel::None.allowed())).is_none());
        assert_eq!(
            Sse2Token::new(detected.intersection(CpuLevel::Sse2.allowed())).is_some(),
            detected.contains(CpuFlags::SSE2)
        );
        let ssse3_duzeyi = detected.intersection(CpuLevel::Ssse3.allowed());
        assert!(Sse41Token::new(ssse3_duzeyi).is_none());
        assert_eq!(
            Ssse3Token::new(ssse3_duzeyi).is_some(),
            detected.contains(CpuFlags::SSSE3)
        );
    }

    #[test]
    fn p51_arm_belirtecleri_tespit_ve_maskeye_uyar() {
        let detected = CpuFlags::detect();
        let fake_arm = CpuFlags::ARM_DAV1D;
        assert_eq!(
            NeonToken::new(fake_arm).is_some(),
            detected.contains(CpuFlags::NEON)
        );
        assert_eq!(
            DotprodToken::new(fake_arm).is_some(),
            detected.contains(CpuFlags::NEON.union(CpuFlags::DOTPROD))
        );
        assert!(NeonToken::new(CpuFlags::NONE).is_none());
        assert!(DotprodToken::new(CpuFlags::NEON).is_none());
        assert!(DotprodToken::new(CpuFlags::DOTPROD).is_none());
        for level in CpuLevel::ALL_ARM {
            let allowed = detected.intersection(level.allowed());
            assert_eq!(
                NeonToken::new(allowed).is_some(),
                detected.contains(CpuFlags::NEON) && level.allowed().contains(CpuFlags::NEON),
                "NeonToken, düzey {}",
                level.name()
            );
            assert_eq!(
                DotprodToken::new(allowed).is_some(),
                detected.contains(CpuFlags::NEON.union(CpuFlags::DOTPROD))
                    && level
                        .allowed()
                        .contains(CpuFlags::NEON.union(CpuFlags::DOTPROD)),
                "DotprodToken, düzey {}",
                level.name()
            );
        }
    }
}
