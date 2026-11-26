//! CPU özellik tespiti, özellik maskesi ve CPU quirks tablosu (F3a / P24).
//!
//! ## Sözleşme
//!
//! - Tespit **çalışma zamanında bir kez** yapılır ve [`CpuFlags`] olarak taşınır;
//!   piksel/işlem başına CPU tespiti **yasaktır** (DSP-01/4).
//! - x86 bayrakları dav1d bayraklarının **anlamını ve bit değerini** taşır
//!   (`src/x86/cpu.h:32-39`): `SSE2 = 1 << 0`, `SSSE3 = 1 << 1`,
//!   `SSE41 = 1 << 2`, `AVX2 = 1 << 3`, `AVX512ICL = 1 << 4`,
//!   `SLOW_GATHER = 1 << 5`. Bileşik koşullar [`x86::X86CpuId::flags`]
//!   içindedir (dav1d `dav1d_get_cpu_flags_x86`, `src/x86/cpu.c:47-98`):
//!   SSSE3 = SSE3 + SSSE3, AVX2 = AVX + BMI1 + BMI2 + işletim sistemi YMM
//!   durumu, AVX512ICL = 14 alt özellik + işletim sistemi ZMM/OPMASK durumu.
//!   Ham CPUID bitleri (AVX, BMI1, BMI2 …) yalnız [`x86::X86CpuId`] içinde iç
//!   ayrıntıdır; dispatch kararı bu modülün bayraklarıyla verilir.
//! - Mikro-mimari kısıtları ([`CpuFlags::SLOW_GATHER`]) veri olarak tutulan
//!   [`quirks::CPU_QUIRKS`] tablosundan gelir (ADR-0002 §4.2, S0.8).
//! - Kullanıcı, benchmark ve doğrulama için yolları **maskeleyerek**
//!   kapatabilir (DSP-01/5): [`Config::cpu_mask`](crate::api::Config::cpu_mask)
//!   **kapatma** maskesidir (`detected.difference(cpu_mask)`); dav1d'nin
//!   `--cpumask` düzeyleri ise **izin** maskesidir (`flags & mask`,
//!   `src/cpu.h:124-125`). İki gösterim arasındaki dönüşüm [`CpuFlags::complement`]
//!   ve [`CpuLevel`] ile açıktır.
//! - Doğrulanamayan bir bayrak **asla** işaretlenmez; yanlış tespit illegal
//!   instruction riski taşır ve `SIGILL` ile kurtarma **tasarlanmaz** (DSP-01/7).
//!
//! ## Kapsam
//!
//! | Alan | Durum |
//! |---|---|
//! | x86_64: SSE2, SSSE3, SSE4.1, AVX2, AVX512ICL | ✅ dav1d koşullarıyla ([`x86`]) |
//! | `SLOW_GATHER` | ✅ [`quirks::CPU_QUIRKS`] (AMD, aile ≤ 0x19, AVX2) |
//! | İşletim sistemi durumu (XCR0) | ✅ std `is_x86_feature_detected!` üzerinden ([`x86`] belgesi) |
//! | ARM/RISC-V/LoongArch | ⬜ **F5/F6**: o hedefler derlenmeye başladığında eklenir; o zamana kadar bayrak kümesi **boş** döner ve yalnız skaler yol seçilir |
//!
//! ## `unsafe` durumu (GUV-02, GUV-08)
//!
//! Bu modül ve alt modülleri `#![forbid(unsafe_code)]` taşır ve **hiç `unsafe`
//! blok içermez**. `std::arch::x86_64::__cpuid` (Rust 1.99.1'de güvenli) ve
//! `std::arch::is_x86_feature_detected!` güvenli yüzeylerdir. Dürüstlük notu
//! (GUV-08): bunlar `std` içinde `unsafe` kullanır — "kendi kodumuzda unsafe
//! yok" demek, "binary'de hiç unsafe yok" demek **değildir**.

#![forbid(unsafe_code)]

pub mod arm;
pub mod quirks;
pub mod token;
pub mod x86;

pub use arm::ArmCpuId;
pub use token::{
    Avx2Token, Avx512IclToken, DotprodToken, NeonToken, Sse2Token, Sse41Token, Ssse3Token,
};

use alloc::vec::Vec;
use core::fmt;

/// CPU özellik ve mikro-mimari kısıt bit kümesi (DSP-01, DSP-04).
///
/// x86 bitleri dav1d'nin `enum CpuFlags` değerleriyle **aynıdır**
/// (`src/x86/cpu.h:32-39`), böylece [`CpuFlags::dav1d_x86_bits`] dav1d'nin
/// `unsigned` bayrak değerini doğrudan verir. Bilinmeyen bitler
/// [`CpuFlags::from_bits`] tarafından reddedilir.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CpuFlags(u64);

impl CpuFlags {
    /// Boş küme: hiçbir optimize yol seçilemez, yalnız skaler yol.
    pub const NONE: Self = Self(0);
    /// dav1d `DAV1D_X86_CPU_FLAG_SSE2`: CPUID 1 EDX CMOV + SSE + SSE2.
    pub const SSE2: Self = Self(1 << 0);
    /// dav1d `DAV1D_X86_CPU_FLAG_SSSE3`: SSE2 + CPUID 1 ECX SSE3 + SSSE3.
    pub const SSSE3: Self = Self(1 << 1);
    /// dav1d `DAV1D_X86_CPU_FLAG_SSE41`: SSSE3 + CPUID 1 ECX SSE4.1.
    pub const SSE4_1: Self = Self(1 << 2);
    /// dav1d `DAV1D_X86_CPU_FLAG_AVX2`: OSXSAVE + AVX + XCR0 XMM/YMM +
    /// CPUID 7 EBX BMI1 + AVX2 + BMI2 (yalnız x86_64).
    pub const AVX2: Self = Self(1 << 3);
    /// dav1d `DAV1D_X86_CPU_FLAG_AVX512ICL`: AVX2 + XCR0 ZMM/OPMASK + 14 alt
    /// özellik (F, CD, BW, DQ, VL, IFMA, VBMI, VBMI2, VNNI, BITALG,
    /// VPOPCNTDQ, GFNI, VAES, VPCLMULQDQ).
    pub const AVX512ICL: Self = Self(1 << 4);
    /// **Quirk:** toplama (gather) komutları yavaş — dav1d
    /// `DAV1D_X86_CPU_FLAG_SLOW_GATHER` ([`quirks::CPU_QUIRKS`]).
    pub const SLOW_GATHER: Self = Self(1 << 5);
    /// ARM NEON (F5).
    pub const NEON: Self = Self(1 << 8);
    /// ARM DOTPROD (F5; stable `core::arch` desteği doğrulandı — S0.6.1).
    pub const DOTPROD: Self = Self(1 << 9);
    /// ARM I8MM (F5; stable intrinsic'i **yok**, skaler yol devrede — S0.6.1).
    pub const I8MM: Self = Self(1 << 10);
    /// ARM SVE2 (F5).
    pub const SVE2: Self = Self(1 << 11);
    /// ARM SVE (`DSP-10` parite bayrağı; `dav1d` `DAV1D_ARM_CPU_FLAG_SVE`).
    pub const SVE: Self = Self(1 << 12);
    /// RISC-V Vector (`V`, `DSP-10` / `F6`; `dav1d` `DAV1D_RISCV_CPU_FLAG_V`;
    /// `rustc 1.100.0` `riscv_ext_intrinsics` kararsız → skaler fallback, `KARAR 1`).
    pub const RVV: Self = Self(1 << 16);
    /// PowerPC VSX (`DSP-10` / `F6`; `dav1d` `DAV1D_PPC_CPU_FLAG_VSX`;
    /// `rustc 1.100.0` `stdarch_powerpc` kararsız → skaler fallback, `KARAR 1`).
    pub const VSX: Self = Self(1 << 20);
    /// PowerPC POWER9 (`DSP-10` / `F6`; `dav1d` `DAV1D_PPC_CPU_FLAG_PWR9`;
    /// `rustc 1.100.0` `stdarch_powerpc` kararsız → skaler fallback, `KARAR 1`).
    pub const PWR9: Self = Self(1 << 21);
    /// LoongArch 128-bit SIMD (`LSX`, `DSP-10` / `F6`; `dav1d` `DAV1D_LOONGARCH_CPU_FLAG_LSX`;
    /// `rustc 1.100.0` `stdarch_loongarch` kararsız → skaler fallback, `KARAR 1`).
    pub const LSX: Self = Self(1 << 24);
    /// LoongArch 256-bit SIMD (`LASX`, `DSP-10` / `F6`; `dav1d` `DAV1D_LOONGARCH_CPU_FLAG_LASX`;
    /// `rustc 1.100.0` `stdarch_loongarch` kararsız → skaler fallback, `KARAR 1`).
    pub const LASX: Self = Self(1 << 25);
    /// WebAssembly 128-bit sabit genişlikli SIMD (`simd128`, `DSP-07` / `WP-10` / `F6`;
    /// çalışma zamanı CPUID yoktur, `-C target-feature=+simd128` derleme zamanı seçimi).
    pub const WASM_SIMD128: Self = Self(1 << 28);

    /// dav1d x86 bayraklarının birleşimi (`src/x86/cpu.h:32-39`).
    pub const X86_DAV1D: Self = Self(0x3f);

    /// dav1d ARM bayraklarının birleşimi (`src/arm/cpu.h:31-37`).
    pub const ARM_DAV1D: Self =
        Self(Self::NEON.0 | Self::DOTPROD.0 | Self::I8MM.0 | Self::SVE2.0 | Self::SVE.0);

    /// Tanımlı tüm bitlerin birleşimi (`DSP-10` + `DSP-07` tam bayrak kümesi).
    pub const ALL_KNOWN: Self =
        Self(Self::X86_DAV1D.0 | (0x1f << 8) | (1 << 16) | (0x3 << 20) | (0x3 << 24) | (1 << 28));

    /// Ham bit deseni.
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Bilinen bitlerden küme kurar; bilinmeyen bit varsa `None` (DSP-01/7).
    #[must_use]
    pub const fn from_bits(bits: u64) -> Option<Self> {
        if bits & !Self::ALL_KNOWN.0 == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    /// dav1d'nin x86 `unsigned` bayrak değeri (`dav1d_get_cpu_flags_x86`).
    #[must_use]
    pub const fn dav1d_x86_bits(self) -> u32 {
        (self.0 & Self::X86_DAV1D.0) as u32
    }

    /// dav1d'nin x86 bayrak değerinden küme kurar; tanımsız bitler atılır.
    #[must_use]
    pub const fn from_dav1d_x86_bits(bits: u32) -> Self {
        Self(bits as u64 & Self::X86_DAV1D.0)
    }

    /// dav1d'nin ARM `unsigned` bayrak değeri (`dav1d_get_cpu_flags_arm`,
    /// `src/arm/cpu.h:31-37`: `NEON = 1<<0`, `DOTPROD = 1<<1`, `I8MM = 1<<2`,
    /// `SVE = 1<<3`, `SVE2 = 1<<4`).
    #[must_use]
    pub const fn dav1d_arm_bits(self) -> u32 {
        let mut out = 0u32;
        if self.contains(Self::NEON) {
            out |= arm::DAV1D_ARM_CPU_FLAG_NEON;
        }
        if self.contains(Self::DOTPROD) {
            out |= arm::DAV1D_ARM_CPU_FLAG_DOTPROD;
        }
        if self.contains(Self::I8MM) {
            out |= arm::DAV1D_ARM_CPU_FLAG_I8MM;
        }
        if self.contains(Self::SVE) {
            out |= arm::DAV1D_ARM_CPU_FLAG_SVE;
        }
        if self.contains(Self::SVE2) {
            out |= arm::DAV1D_ARM_CPU_FLAG_SVE2;
        }
        out
    }

    /// dav1d'nin ARM bayrak değerinden (`src/arm/cpu.h:31-37`) küme kurar;
    /// tanımsız bitler atılır.
    #[must_use]
    pub const fn from_dav1d_arm_bits(bits: u32) -> Self {
        let mut out = Self::NONE;
        if bits & arm::DAV1D_ARM_CPU_FLAG_NEON != 0 {
            out = out.union(Self::NEON);
        }
        if bits & arm::DAV1D_ARM_CPU_FLAG_DOTPROD != 0 {
            out = out.union(Self::DOTPROD);
        }
        if bits & arm::DAV1D_ARM_CPU_FLAG_I8MM != 0 {
            out = out.union(Self::I8MM);
        }
        if bits & arm::DAV1D_ARM_CPU_FLAG_SVE != 0 {
            out = out.union(Self::SVE);
        }
        if bits & arm::DAV1D_ARM_CPU_FLAG_SVE2 != 0 {
            out = out.union(Self::SVE2);
        }
        out
    }

    /// Küme boş mu? (yalnız skaler yol anlamına gelir.)
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// `other`'ın **tüm** bitleri bu kümede var mı?
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// İki kümenin en az bir ortak biti var mı?
    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// Birleşim (bayrak ekleme).
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Kesişim: dav1d'nin **izin** maskesi böyle uygulanır (`flags & mask`,
    /// `src/cpu.h:124-125`).
    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Fark: `other`'daki bitler bu kümeden çıkarılır.
    ///
    /// [`Config::cpu_mask`](crate::api::Config::cpu_mask) (**kapatma** maskesi)
    /// bu yolla uygulanır (DSP-01/5): `detected.difference(mask)`.
    #[must_use]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Bilinen bitler içinde tümleyen: izin maskesi ↔ kapatma maskesi.
    ///
    /// Her `f` ve izin maskesi `a` için
    /// `f.difference(a.complement()) == f.intersection(a)` (bilinen bitlerde);
    /// tersine kapatma maskesi `m` için izin maskesi `m.complement()`'tir.
    #[must_use]
    pub const fn complement(self) -> Self {
        Self(Self::ALL_KNOWN.0 & !self.0)
    }

    /// İşaretli bayrakların adları (raporlama için).
    #[must_use]
    pub fn names(self) -> Vec<&'static str> {
        let mut names = Vec::new();
        for (flag, name) in Self::named() {
            if self.contains(flag) {
                names.push(name);
            }
        }
        names
    }

    /// (bayrak, ad) çiftleri — [`CpuFlags::names`] ve [`fmt::Debug`] için tek
    /// kaynak. Adlar dav1d kaynak adları ve WASM `simd128` (`DSP-07`, `DSP-10`) adlarıdır.
    #[must_use]
    pub const fn named() -> [(Self, &'static str); 17] {
        [
            (Self::SSE2, "SSE2"),
            (Self::SSSE3, "SSSE3"),
            (Self::SSE4_1, "SSE41"),
            (Self::AVX2, "AVX2"),
            (Self::AVX512ICL, "AVX512ICL"),
            (Self::SLOW_GATHER, "SLOW_GATHER"),
            (Self::NEON, "NEON"),
            (Self::DOTPROD, "DOTPROD"),
            (Self::I8MM, "I8MM"),
            (Self::SVE2, "SVE2"),
            (Self::SVE, "SVE"),
            (Self::RVV, "V"),
            (Self::VSX, "VSX"),
            (Self::PWR9, "PWR9"),
            (Self::LSX, "LSX"),
            (Self::LASX, "LASX"),
            (Self::WASM_SIMD128, "SIMD128"),
        ]
    }

    /// Çalışma zamanı tespiti — süreç başına **bir kez** yapılır ve önbelleğe
    /// alınır (DSP-01/4); sonraki çağrılar aynı değeri döner.
    ///
    /// x86_64'te [`x86::X86CpuId::read`] ham CPUID ve işletim sistemi
    /// durumunu okur, [`x86::X86CpuId::flags`] dav1d koşullarını ve quirks
    /// tablosunu uygular. `wasm32`'de çalışma zamanı CPUID olmadığı için
    /// (`DSP-07`) `-C target-feature=+simd128` derleme zamanı özelliği etkinse
    /// [`CpuFlags::WASM_SIMD128`], değilse [`CpuFlags::NONE`] döner. Diğer
    /// hedeflerde boş küme döner (skaler yol).
    /// ISA belirteçleri ([`Avx2Token`], [`Avx512IclToken`]) bu değere bakar.
    ///
    /// P38: Miri (`cfg(miri)`, yalnız araç kanalı) CPUID'nin inline
    /// assembly'sini yorumlayamaz; orada da boş küme döner (skaler yol), böylece
    /// API yaşam döngüsü testleri Miri'de koşar. Normal derlemeye etkisi yoktur.
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(feature = "std")]
        {
            static DETECTED: crate::sync::OnceLock<CpuFlags> = crate::sync::OnceLock::new();
            *DETECTED.get_or_init(|| {
                #[cfg(all(target_arch = "x86_64", not(miri)))]
                {
                    x86::X86CpuId::read().flags()
                }
                #[cfg(all(target_arch = "aarch64", not(miri)))]
                {
                    arm::ArmCpuId::read().flags()
                }
                #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
                {
                    Self::WASM_SIMD128
                }
                #[cfg(any(
                    miri,
                    not(any(
                        target_arch = "x86_64",
                        target_arch = "aarch64",
                        all(target_arch = "wasm32", target_feature = "simd128")
                    ))
                ))]
                {
                    Self::NONE
                }
            })
        }
        #[cfg(not(feature = "std"))]
        {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            {
                let mut flags = Self::NONE;
                if cfg!(target_feature = "sse2") {
                    flags = flags.union(Self::SSE2);
                    if cfg!(all(target_feature = "sse3", target_feature = "ssse3")) {
                        flags = flags.union(Self::SSSE3);
                        if cfg!(target_feature = "sse4.1") {
                            flags = flags.union(Self::SSE4_1);
                            if cfg!(all(
                                target_feature = "avx",
                                target_feature = "avx2",
                                target_feature = "bmi1",
                                target_feature = "bmi2"
                            )) {
                                flags = flags.union(Self::AVX2);
                                if cfg!(all(
                                    target_feature = "avx512f",
                                    target_feature = "avx512dq",
                                    target_feature = "avx512ifma",
                                    target_feature = "avx512cd",
                                    target_feature = "avx512bw",
                                    target_feature = "avx512vl",
                                    target_feature = "avx512vbmi",
                                    target_feature = "avx512vbmi2",
                                    target_feature = "gfni",
                                    target_feature = "vaes",
                                    target_feature = "vpclmulqdq",
                                    target_feature = "avx512vnni",
                                    target_feature = "avx512bitalg",
                                    target_feature = "avx512vpopcntdq"
                                )) {
                                    flags = flags.union(Self::AVX512ICL);
                                }
                            }
                        }
                    }
                }
                flags
            }
            #[cfg(all(target_arch = "aarch64", not(miri)))]
            {
                let mut flags = Self::NONE;
                if cfg!(target_feature = "neon") {
                    flags = flags.union(Self::NEON);
                    if cfg!(target_feature = "dotprod") {
                        flags = flags.union(Self::DOTPROD);
                    }
                }
                flags
            }
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128", not(miri)))]
            {
                Self::WASM_SIMD128
            }
            #[cfg(any(
                miri,
                not(any(
                    target_arch = "x86_64",
                    target_arch = "aarch64",
                    all(target_arch = "wasm32", target_feature = "simd128")
                ))
            ))]
            {
                Self::NONE
            }
        }
    }
}

impl fmt::Debug for CpuFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CpuFlags({self})")
    }
}

impl fmt::Display for CpuFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names = self.names();
        if names.is_empty() {
            f.write_str("(boş)")
        } else {
            f.write_str(&names.join("|"))
        }
    }
}

impl core::ops::BitOr for CpuFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl core::ops::BitOrAssign for CpuFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.union(rhs);
    }
}

/// dav1d `--cpumask` düzeyleri (x86; `tools/dav1d_cli_parse.c:211-217`,
/// tablo `258-264`).
///
/// Her düzey bir **izin** maskesidir ([`CpuLevel::allowed`]); dav1d'deki
/// gibi `SLOW_GATHER` hiçbir düzeyin içinde değildir, yani bir düzey
/// seçmek bu quirk'ü de düşürür (dav1d `flags & mask`). Varsayılan (maske
/// yok, dav1d `-1`) bir düzey değildir: [`Config::cpu_mask`] =
/// [`CpuFlags::NONE`].
///
/// [`Config::cpu_mask`] için hazır kapatma değeri [`CpuLevel::cpu_mask`]'tir:
///
/// ```
/// # use av1oxide::api::Config;
/// # use av1oxide::cpu::CpuLevel;
/// let skaler = Config {
///     cpu_mask: CpuLevel::None.cpu_mask(),
///     ..Config::default()
/// };
/// assert!(av1oxide::cpu::CpuFlags::detect()
///     .difference(skaler.cpu_mask)
///     .is_empty());
/// ```
///
/// [`Config::cpu_mask`]: crate::api::Config::cpu_mask
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CpuLevel {
    /// `none` (0): yalnız skaler yol.
    None,
    /// `sse2`: ≤ SSE2.
    Sse2,
    /// `ssse3`: ≤ SSSE3.
    Ssse3,
    /// `sse41`: ≤ SSE4.1.
    Sse41,
    /// `avx2`: ≤ AVX2.
    Avx2,
    /// `avx512icl`: ≤ AVX512ICL (bütün x86 SIMD düzeyleri; `SLOW_GATHER` hariç).
    Avx512Icl,
    /// `neon`: ARM64 NEON (`CpuFlags::NEON`).
    Neon,
    /// `dotprod`: ARM64 NEON + DOTPROD (`CpuFlags::NEON | CpuFlags::DOTPROD`).
    Dotprod,
}

impl CpuLevel {
    /// Bütün x86 düzeyleri, düşükten yükseğe (`docs/cpu-dispatch.json` şeması).
    pub const ALL: [Self; 6] = [
        Self::None,
        Self::Sse2,
        Self::Ssse3,
        Self::Sse41,
        Self::Avx2,
        Self::Avx512Icl,
    ];

    /// Bütün kararlı ARM64 düzeyleri, düşükten yükseğe (`none`, `neon`, `dotprod`).
    pub const ALL_ARM: [Self; 3] = [Self::None, Self::Neon, Self::Dotprod];

    /// dav1d CLI adı (`--cpumask <ad>`).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Sse2 => "sse2",
            Self::Ssse3 => "ssse3",
            Self::Sse41 => "sse41",
            Self::Avx2 => "avx2",
            Self::Avx512Icl => "avx512icl",
            Self::Neon => "neon",
            Self::Dotprod => "dotprod",
        }
    }

    /// dav1d CLI adından düzey; bilinmeyen ad `None`.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "none" => Some(Self::None),
            "sse2" => Some(Self::Sse2),
            "ssse3" => Some(Self::Ssse3),
            "sse41" => Some(Self::Sse41),
            "avx2" => Some(Self::Avx2),
            "avx512icl" => Some(Self::Avx512Icl),
            "neon" => Some(Self::Neon),
            "dotprod" => Some(Self::Dotprod),
            _ => None,
        }
    }

    /// `CpuLevel::parse` için eşanlamlı yardımcı (`--cpu-mask` adından düzey).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::parse(name)
    }

    /// İzin verilen bayraklar — dav1d `X86_CPU_MASK_*` ve ARM `NEON`/`DOTPROD` değerleri.
    #[must_use]
    pub const fn allowed(self) -> CpuFlags {
        match self {
            Self::None => CpuFlags::NONE,
            Self::Sse2 => CpuFlags::SSE2,
            Self::Ssse3 => CpuFlags::SSSE3.union(Self::Sse2.allowed()),
            Self::Sse41 => CpuFlags::SSE4_1.union(Self::Ssse3.allowed()),
            Self::Avx2 => CpuFlags::AVX2.union(Self::Sse41.allowed()),
            Self::Avx512Icl => CpuFlags::AVX512ICL.union(Self::Avx2.allowed()),
            Self::Neon => CpuFlags::NEON,
            Self::Dotprod => CpuFlags::DOTPROD.union(Self::Neon.allowed()),
        }
    }

    /// [`Config::cpu_mask`](crate::api::Config::cpu_mask) için **kapatma**
    /// maskesi: bilinen bitlerden izin verilmeyenler.
    #[must_use]
    pub const fn cpu_mask(self) -> CpuFlags {
        self.allowed().complement()
    }

    /// dav1d'nin `dav1d_set_cpu_flags_mask` değeri (`unsigned`).
    #[must_use]
    pub const fn dav1d_mask(self) -> u32 {
        self.allowed().dav1d_x86_bits()
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::{CpuFlags, CpuLevel};

    #[test]
    fn bit_kumesi_islemleri() {
        let set = CpuFlags::SSE2.union(CpuFlags::AVX2);
        assert!(set.contains(CpuFlags::SSE2));
        assert!(set.contains(CpuFlags::AVX2));
        assert!(!set.contains(CpuFlags::SSSE3));
        assert!(set.intersects(CpuFlags::SSE2));
        assert_eq!(set.difference(CpuFlags::SSE2), CpuFlags::AVX2);
        assert_eq!(set.intersection(CpuFlags::AVX2), CpuFlags::AVX2);
        assert!(CpuFlags::NONE.is_empty());
        assert!(!set.is_empty());
    }

    #[test]
    fn bilinmeyen_bit_reddedilir() {
        assert_eq!(
            CpuFlags::from_bits(CpuFlags::ALL_KNOWN.bits()),
            Some(CpuFlags::ALL_KNOWN)
        );
        assert_eq!(CpuFlags::from_bits(1 << 6), None);
        assert_eq!(CpuFlags::from_bits(1 << 13), None);
        assert_eq!(CpuFlags::from_bits(1 << 17), None);
        assert_eq!(CpuFlags::from_bits(1 << 22), None);
        assert_eq!(CpuFlags::from_bits(1 << 26), None);
        assert_eq!(CpuFlags::from_bits(1 << 29), None);
        assert_eq!(CpuFlags::from_bits(u64::MAX), None);
    }

    #[test]
    fn dsp10_tum_mimari_bayrak_adlari_tanimlidir() {
        let f6_flags = CpuFlags::SVE
            | CpuFlags::RVV
            | CpuFlags::VSX
            | CpuFlags::PWR9
            | CpuFlags::LSX
            | CpuFlags::LASX
            | CpuFlags::WASM_SIMD128;
        assert_eq!(
            f6_flags.names(),
            vec!["SVE", "V", "VSX", "PWR9", "LSX", "LASX", "SIMD128"]
        );
        assert_eq!(f6_flags.dav1d_x86_bits(), 0);
    }

    /// x86 bitleri dav1d `enum CpuFlags` değerleriyle aynıdır
    /// (`src/x86/cpu.h:32-39`).
    #[test]
    fn x86_bitleri_dav1d_degerleridir() {
        assert_eq!(CpuFlags::SSE2.dav1d_x86_bits(), 1 << 0);
        assert_eq!(CpuFlags::SSSE3.dav1d_x86_bits(), 1 << 1);
        assert_eq!(CpuFlags::SSE4_1.dav1d_x86_bits(), 1 << 2);
        assert_eq!(CpuFlags::AVX2.dav1d_x86_bits(), 1 << 3);
        assert_eq!(CpuFlags::AVX512ICL.dav1d_x86_bits(), 1 << 4);
        assert_eq!(CpuFlags::SLOW_GATHER.dav1d_x86_bits(), 1 << 5);
        assert_eq!(CpuFlags::NEON.dav1d_x86_bits(), 0);
        for bits in 0..64u32 {
            assert_eq!(CpuFlags::from_dav1d_x86_bits(bits).dav1d_x86_bits(), bits);
        }
        assert_eq!(CpuFlags::from_dav1d_x86_bits(u32::MAX), CpuFlags::X86_DAV1D);
    }

    #[test]
    fn adlar_ve_gorunum() {
        assert!(CpuFlags::NONE.names().is_empty());
        assert_eq!(CpuFlags::NONE.to_string(), "(boş)");
        let set = CpuFlags::SSE2 | CpuFlags::SLOW_GATHER;
        assert_eq!(set.names(), vec!["SSE2", "SLOW_GATHER"]);
        assert_eq!(set.to_string(), "SSE2|SLOW_GATHER");
        assert_eq!(format!("{set:?}"), "CpuFlags(SSE2|SLOW_GATHER)");
        assert_eq!(
            CpuFlags::from_dav1d_x86_bits(0x1f).to_string(),
            "SSE2|SSSE3|SSE41|AVX2|AVX512ICL"
        );
    }

    #[test]
    fn ad_tablosu_tum_bilinen_bitleri_kapsar() {
        let mut covered = CpuFlags::NONE;
        for (flag, name) in CpuFlags::named() {
            assert!(flag.bits().is_power_of_two(), "tek bit değil: {name}");
            assert!(!covered.intersects(flag), "bit tekrarı: {name}");
            covered |= flag;
        }
        assert_eq!(covered, CpuFlags::ALL_KNOWN);
    }

    /// dav1d `X86_CPU_MASK_*` (`tools/dav1d_cli_parse.c:212-216`) ve `none`
    /// (264): her düzey bir öncekini kapsar; `SLOW_GATHER` hiçbirinde yok.
    #[test]
    fn duzeyler_dav1d_maskeleridir() {
        let want = [0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f];
        let names = ["none", "sse2", "ssse3", "sse41", "avx2", "avx512icl"];
        for ((level, mask), name) in CpuLevel::ALL.into_iter().zip(want).zip(names) {
            assert_eq!(level.dav1d_mask(), mask, "{name}");
            assert_eq!(level.name(), name);
            assert_eq!(CpuLevel::parse(name), Some(level));
            assert!(!level.allowed().contains(CpuFlags::SLOW_GATHER));
        }
        assert_eq!(CpuLevel::parse("avx"), None);
        assert_eq!(CpuLevel::parse(""), None);
        for pair in CpuLevel::ALL.windows(2) {
            assert!(pair[1].allowed().contains(pair[0].allowed()));
            assert_ne!(pair[1].allowed(), pair[0].allowed());
        }
    }

    /// İzin maskesi (dav1d `flags & mask`) ile kapatma maskesi
    /// (`Config::cpu_mask`, `difference`) bütün bilinen alt kümelerde aynı
    /// etkin kümeyi verir.
    #[test]
    fn izin_ve_kapatma_maskesi_esdegerdir() {
        let known: Vec<CpuFlags> = CpuFlags::named().iter().map(|(f, _)| *f).collect();
        for subset in 0u32..(1 << known.len()) {
            let mut flags = CpuFlags::NONE;
            for (i, flag) in known.iter().enumerate() {
                if subset & (1 << i) != 0 {
                    flags |= *flag;
                }
            }
            for level in CpuLevel::ALL {
                assert_eq!(
                    flags.difference(level.cpu_mask()),
                    flags.intersection(level.allowed()),
                    "{flags} / {}",
                    level.name()
                );
            }
            assert_eq!(flags.complement().complement(), flags);
            assert_eq!(CpuFlags::ALL_KNOWN.difference(flags.complement()), flags);
        }
        assert_eq!(CpuLevel::None.cpu_mask(), CpuFlags::ALL_KNOWN);
    }

    #[test]
    fn arm_duzeyleri_neon_ve_dotprod() {
        assert_eq!(CpuLevel::from_name("neon"), Some(CpuLevel::Neon));
        assert_eq!(CpuLevel::from_name("dotprod"), Some(CpuLevel::Dotprod));
        assert_eq!(CpuLevel::Neon.name(), "neon");
        assert_eq!(CpuLevel::Dotprod.name(), "dotprod");
        assert_eq!(CpuLevel::Neon.allowed(), CpuFlags::NEON);
        assert_eq!(
            CpuLevel::Dotprod.allowed(),
            CpuFlags::NEON.union(CpuFlags::DOTPROD)
        );
        assert_eq!(CpuLevel::Neon.allowed().dav1d_arm_bits(), 0x01);
        assert_eq!(CpuLevel::Dotprod.allowed().dav1d_arm_bits(), 0x03);
        for pair in CpuLevel::ALL_ARM.windows(2) {
            assert!(pair[1].allowed().contains(pair[0].allowed()));
            assert_ne!(pair[1].allowed(), pair[0].allowed());
        }
    }

    #[test]
    fn tespit_x86_64_tabanini_icerir() {
        let detected = CpuFlags::detect();
        #[cfg(target_arch = "x86_64")]
        {
            assert!(detected.contains(CpuFlags::SSE2), "x86_64'te SSE2 tabandır");
            #[cfg(feature = "std")]
            assert_eq!(detected, super::x86::X86CpuId::read().flags());
            assert!(CpuFlags::X86_DAV1D.contains(detected));
        }
        #[cfg(target_arch = "aarch64")]
        {
            assert!(CpuFlags::ARM_DAV1D.contains(detected));
        }
        assert_eq!(detected.difference(detected), CpuFlags::NONE);
        assert!(
            detected.difference(CpuLevel::None.cpu_mask()).is_empty(),
            "skaler düzey bütün bayrakları kapatır"
        );
    }
}
