//! x86 CPU tespiti — dav1d `dav1d_get_cpu_flags_x86` karşılığı
//! (`src/x86/cpu.c:47-98`).
//!
//! Tespit iki parçadır:
//!
//! 1. [`X86CpuId::read`] ham CPUID yapraklarını (0, 1, 7) güvenli
//!    `std::arch::x86_64::__cpuid` ile ve işletim sistemi durumunu (XCR0)
//!    std `is_x86_feature_detected!` üzerinden okur.
//! 2. [`X86CpuId::flags`] saf bir işlevdir: dav1d'nin koşullarını aynı
//!    maskelerle ve aynı iç içe sırayla uygular, sonra
//!    [`super::quirks::CPU_QUIRKS`] tablosunu ekler. Saf olduğu için bilinen
//!    işlemcilerin CPUID değerleriyle birim testine açıktır.
//!
//! ## İşletim sistemi durumu (DSP-04, ADR-0003 İ-8)
//!
//! dav1d XCR0'ı doğrudan okur (`dav1d_cpu_xgetbv(0)`, `src/x86/cpu.c:74`).
//! `xgetbv` Rust'ta `unsafe`'tir; bu modül onun yerine std'nin aynı
//! denetimini kullanır (Rust 1.99.1,
//! `library/std_detect/src/detect/os/x86.rs`):
//!
//! | dav1d | std (1.99.1) | Bu modül |
//! |---|---|---|
//! | `X(ecx, 0x18000000)` OSXSAVE + AVX | `cpu_xsave` (ecx 26, satır 168) → `cpu_osxsave` (ecx 27, satır 181) | ham ecx (dav1d maskesi) |
//! | `X(xcr0, 0x6)` XMM/YMM | `os_avx_support = xcr0 & 6 == 6` (satır 199); `xsave`, `avx`, `avx2` yalnız bu durumda açılır (satır 230, 241, 258-259) | `is_x86_feature_detected!("xsave")` |
//! | `X(xcr0, 0xe0)` ZMM/OPMASK | `os_avx512_support = xcr0 & 0xe0 == 0xe0` (satır 201); `avx512f` yalnız `os_avx512_support && f16c && fma` iken (satır 279-280) | `is_x86_feature_detected!("avx512f")` |
//!
//! **Bilinçli fark:** std'nin `avx512f` koşulu ayrıca FMA ve F16C ister; dav1d
//! istemez. Bu yalnız AVX-512'yi FMA/F16C'siz bildiren (bilinen örneği
//! olmayan) bir işlemcide AVX512ICL'nin **eksik** raporlanmasına yol açabilir;
//! fazla raporlama yolu yoktur (DSP-01/7).

#![forbid(unsafe_code)]

use alloc::string::String;

use super::{CpuFlags, quirks};

/// `X(reg, mask)`: maskedeki bütün bitler var mı (`src/x86/cpu.c:45`).
const fn has(reg: u32, mask: u32) -> bool {
    reg & mask == mask
}

/// CPUID 1 EDX: CMOV/SSE/SSE2 (`src/x86/cpu.c:63`).
pub const LEAF1_EDX_SSE2: u32 = 0x0600_8000;
/// CPUID 1 ECX: SSE3/SSSE3 (`src/x86/cpu.c:65`).
pub const LEAF1_ECX_SSSE3: u32 = 0x0000_0201;
/// CPUID 1 ECX: SSE4.1 (`src/x86/cpu.c:67`).
pub const LEAF1_ECX_SSE41: u32 = 0x0008_0000;
/// CPUID 1 ECX: OSXSAVE/AVX (`src/x86/cpu.c:73`).
pub const LEAF1_ECX_OSXSAVE_AVX: u32 = 0x1800_0000;
/// CPUID 7 EBX: BMI1/BMI2/AVX2 (`src/x86/cpu.c:78`).
pub const LEAF7_EBX_AVX2: u32 = 0x0000_0128;
/// CPUID 7 EBX: AVX-512 F/DQ/IFMA/CD/BW/VL (`src/x86/cpu.c:81`).
pub const LEAF7_EBX_AVX512ICL: u32 = 0xd023_0000;
/// CPUID 7 ECX: VBMI/VBMI2/GFNI/VAES/VPCLMULQDQ/VNNI/BITALG/VPOPCNTDQ
/// (`src/x86/cpu.c:81`).
pub const LEAF7_ECX_AVX512ICL: u32 = 0x0000_5f42;

/// AVX512ICL'nin 14 alt özelliği: (std `is_x86_feature_detected!` adı,
/// CPUID 7 yazmacı, bit).
pub const AVX512ICL_FEATURES: [(&str, Leaf7Reg, u32); 14] = [
    ("avx512f", Leaf7Reg::Ebx, 16),
    ("avx512dq", Leaf7Reg::Ebx, 17),
    ("avx512ifma", Leaf7Reg::Ebx, 21),
    ("avx512cd", Leaf7Reg::Ebx, 28),
    ("avx512bw", Leaf7Reg::Ebx, 30),
    ("avx512vl", Leaf7Reg::Ebx, 31),
    ("avx512vbmi", Leaf7Reg::Ecx, 1),
    ("avx512vbmi2", Leaf7Reg::Ecx, 6),
    ("gfni", Leaf7Reg::Ecx, 8),
    ("vaes", Leaf7Reg::Ecx, 9),
    ("vpclmulqdq", Leaf7Reg::Ecx, 10),
    ("avx512vnni", Leaf7Reg::Ecx, 11),
    ("avx512bitalg", Leaf7Reg::Ecx, 12),
    ("avx512vpopcntdq", Leaf7Reg::Ecx, 14),
];

/// CPUID 7 alt yaprak 0 yazmacı.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Leaf7Reg {
    /// EBX.
    Ebx,
    /// ECX.
    Ecx,
}

/// CPUID ve işletim sistemi durumunun ham görüntüsü.
///
/// Alanlar dav1d'nin okuduğu yazmaçlardır; okunmayan yapraklar sıfırdır
/// (`max_leaf < 7` ise yaprak 7 okunmaz, dav1d gibi).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct X86CpuId {
    /// CPUID 0 EAX: en büyük temel yaprak.
    pub max_leaf: u32,
    /// CPUID 0 EBX/EDX/ECX: üretici dizesi.
    pub vendor: [u8; 12],
    /// CPUID 1 EAX: aile/model/stepping.
    pub leaf1_eax: u32,
    /// CPUID 1 ECX.
    pub leaf1_ecx: u32,
    /// CPUID 1 EDX.
    pub leaf1_edx: u32,
    /// CPUID 7.0 EBX.
    pub leaf7_ebx: u32,
    /// CPUID 7.0 ECX.
    pub leaf7_ecx: u32,
    /// İşletim sistemi XMM/YMM durumunu kaydediyor (dav1d `X(xcr0, 0x6)`).
    pub os_xmm_ymm: bool,
    /// İşletim sistemi ZMM/OPMASK durumunu kaydediyor (dav1d `X(xcr0, 0xe0)`).
    pub os_zmm_opmask: bool,
}

impl X86CpuId {
    /// Bu makinenin CPUID ve işletim sistemi durumunu okur (güvenli kod).
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    #[must_use]
    pub fn read() -> Self {
        use std::arch::x86_64::{__cpuid, __cpuid_count};

        let leaf0 = __cpuid(0);
        let mut vendor = [0u8; 12];
        vendor[..4].copy_from_slice(&leaf0.ebx.to_le_bytes());
        vendor[4..8].copy_from_slice(&leaf0.edx.to_le_bytes());
        vendor[8..].copy_from_slice(&leaf0.ecx.to_le_bytes());
        let mut id = Self {
            max_leaf: leaf0.eax,
            vendor,
            leaf1_eax: 0,
            leaf1_ecx: 0,
            leaf1_edx: 0,
            leaf7_ebx: 0,
            leaf7_ecx: 0,
            os_xmm_ymm: std::arch::is_x86_feature_detected!("xsave"),
            os_zmm_opmask: std::arch::is_x86_feature_detected!("avx512f"),
        };
        if id.max_leaf >= 1 {
            let leaf1 = __cpuid(1);
            id.leaf1_eax = leaf1.eax;
            id.leaf1_ecx = leaf1.ecx;
            id.leaf1_edx = leaf1.edx;
        }
        if id.max_leaf >= 7 {
            let leaf7 = __cpuid_count(7, 0);
            id.leaf7_ebx = leaf7.ebx;
            id.leaf7_ecx = leaf7.ecx;
        }
        id
    }

    /// dav1d aile hesabı: temel + genişletilmiş aile (`src/x86/cpu.c:61`).
    #[must_use]
    pub const fn family(&self) -> u32 {
        ((self.leaf1_eax >> 8) & 0x0f) + ((self.leaf1_eax >> 20) & 0xff)
    }

    /// Model: temel aile 0x6 ya da 0xF ise genişletilmiş model eklenir
    /// (Intel SDM / AMD APM CPUID 1 EAX tanımı; dav1d modeli kullanmaz).
    #[must_use]
    pub const fn model(&self) -> u32 {
        let base_family = (self.leaf1_eax >> 8) & 0x0f;
        let model = (self.leaf1_eax >> 4) & 0x0f;
        if base_family == 0x6 || base_family == 0xf {
            model | (((self.leaf1_eax >> 16) & 0x0f) << 4)
        } else {
            model
        }
    }

    /// Üretici dizesi (ASCII dışı bayt `?` olur).
    #[must_use]
    pub fn vendor_str(&self) -> String {
        self.vendor
            .iter()
            .map(|&b| if b.is_ascii_graphic() { b as char } else { '?' })
            .collect()
    }

    /// Derleme zamanı varsayılan bayrakları — dav1d `dav1d_get_default_cpu_flags`
    /// (`src/cpu.h:58-122`, x86 dalı 90-118): x86_64 tabanı SSE2'dir; `-C target-feature`
    /// ile daha yüksek düzey derlenirse dav1d gibi o düzey ve altı eklenir.
    #[must_use]
    pub const fn default_flags() -> CpuFlags {
        if cfg!(all(
            target_feature = "avx512f",
            target_feature = "avx512cd",
            target_feature = "avx512bw",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            target_feature = "avx512vnni",
            target_feature = "avx512ifma",
            target_feature = "avx512vbmi",
            target_feature = "avx512vbmi2",
            target_feature = "avx512vpopcntdq",
            target_feature = "avx512bitalg",
            target_feature = "gfni",
            target_feature = "vaes",
            target_feature = "vpclmulqdq"
        )) {
            CpuFlags::from_dav1d_x86_bits(0x1f)
        } else if cfg!(target_feature = "avx2") {
            CpuFlags::from_dav1d_x86_bits(0x0f)
        } else if cfg!(any(target_feature = "sse4.1", target_feature = "avx")) {
            CpuFlags::from_dav1d_x86_bits(0x07)
        } else if cfg!(target_feature = "ssse3") {
            CpuFlags::from_dav1d_x86_bits(0x03)
        } else if cfg!(any(target_arch = "x86_64", target_feature = "sse2")) {
            CpuFlags::SSE2
        } else {
            CpuFlags::NONE
        }
    }

    /// dav1d bayrakları: [`X86CpuId::default_flags`] + çalışma zamanı.
    #[must_use]
    pub const fn flags(&self) -> CpuFlags {
        self.flags_from(Self::default_flags())
    }

    /// dav1d `dav1d_get_cpu_flags_x86` (`src/x86/cpu.c:47-98`), başlangıç
    /// kümesi `initial` (dav1d'de derleme zamanı varsayılanı, satır 56).
    #[must_use]
    pub const fn flags_from(&self, initial: CpuFlags) -> CpuFlags {
        let mut flags = initial;
        if self.max_leaf < 1 {
            return flags;
        }
        if has(self.leaf1_edx, LEAF1_EDX_SSE2) {
            flags = flags.union(CpuFlags::SSE2);
            if has(self.leaf1_ecx, LEAF1_ECX_SSSE3) {
                flags = flags.union(CpuFlags::SSSE3);
                if has(self.leaf1_ecx, LEAF1_ECX_SSE41) {
                    flags = flags.union(CpuFlags::SSE4_1);
                }
            }
        }
        // dav1d >128-bit SIMD'i yalnız x86-64'te destekler (satır 71-88); bu
        // modül yalnız x86_64'te `read` eder, saf hesap her hedefte aynıdır.
        if has(self.leaf1_ecx, LEAF1_ECX_OSXSAVE_AVX)
            && self.os_xmm_ymm
            && self.max_leaf >= 7
            && has(self.leaf7_ebx, LEAF7_EBX_AVX2)
        {
            flags = flags.union(CpuFlags::AVX2);
            if self.os_zmm_opmask
                && has(self.leaf7_ebx, LEAF7_EBX_AVX512ICL)
                && has(self.leaf7_ecx, LEAF7_ECX_AVX512ICL)
            {
                flags = flags.union(CpuFlags::AVX512ICL);
            }
        }
        quirks::apply(&self.vendor, self.family(), self.model(), flags)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AVX512ICL_FEATURES, LEAF1_ECX_OSXSAVE_AVX, LEAF1_ECX_SSE41, LEAF1_ECX_SSSE3,
        LEAF1_EDX_SSE2, LEAF7_EBX_AVX2, LEAF7_EBX_AVX512ICL, LEAF7_ECX_AVX512ICL, Leaf7Reg,
        X86CpuId,
    };
    use crate::cpu::CpuFlags;

    /// Bu projenin ölçüm makinesi (AMD Ryzen 9 9950X3D, Zen 5) — ham CPUID
    /// değerleri `docs/cpu-dispatch.json` `makine.cpuid` alanıyla aynıdır.
    const ZEN5: X86CpuId = X86CpuId {
        max_leaf: 0x10,
        vendor: *b"AuthenticAMD",
        leaf1_eax: 0x00b4_0f40,
        leaf1_ecx: 0x7ed8_320b,
        leaf1_edx: 0x178b_fbff,
        leaf7_ebx: 0xf1bf_97ab,
        leaf7_ecx: 0x1940_5fde,
        os_xmm_ymm: true,
        os_zmm_opmask: true,
    };

    /// Aynı özellik bitleriyle başka aile/model/üretici.
    const fn with_cpu(vendor: [u8; 12], eax: u32) -> X86CpuId {
        let mut id = ZEN5;
        id.vendor = vendor;
        id.leaf1_eax = eax;
        id
    }

    /// AVX-512 bitleri silinmiş (Zen 2/3, Haswell, Skylake istemci).
    const fn without_avx512(mut id: X86CpuId) -> X86CpuId {
        id.leaf7_ebx &= !LEAF7_EBX_AVX512ICL;
        id.leaf7_ecx &= !LEAF7_ECX_AVX512ICL;
        id
    }

    fn bits(id: &X86CpuId) -> u32 {
        id.flags_from(CpuFlags::NONE).dav1d_x86_bits()
    }

    #[test]
    fn zen5_dav1d_ile_ayni_bayraklar() {
        assert_eq!(ZEN5.family(), 0x1a);
        assert_eq!(ZEN5.model(), 0x44);
        assert_eq!(ZEN5.vendor_str(), "AuthenticAMD");
        // SSE2 | SSSE3 | SSE41 | AVX2 | AVX512ICL; SLOW_GATHER yok (aile 0x1A > 0x19).
        assert_eq!(bits(&ZEN5), 0x1f);
        assert_eq!(ZEN5.flags().dav1d_x86_bits(), 0x1f);
    }

    /// Bilinen işlemciler: (ad, üretici, CPUID 1 EAX, AVX-512 var mı, beklenen
    /// dav1d bayrakları, aile, model).
    #[test]
    fn bilinen_islemciler() {
        let amd = *b"AuthenticAMD";
        let intel = *b"GenuineIntel";
        let cases = [
            ("Zen 4 (Raphael)", amd, 0x00a6_0f12, true, 0x3f, 0x19, 0x61),
            ("Zen 3 (Vermeer)", amd, 0x00a2_0f10, false, 0x2f, 0x19, 0x21),
            ("Zen 2 (Matisse)", amd, 0x0087_0f10, false, 0x2f, 0x17, 0x71),
            (
                "Zen (Summit Ridge)",
                amd,
                0x0080_0f11,
                false,
                0x2f,
                0x17,
                0x01,
            ),
            ("Excavator", amd, 0x0066_0f01, false, 0x2f, 0x15, 0x60),
            ("Ice Lake", intel, 0x0007_06e5, true, 0x1f, 0x06, 0x7e),
            ("Haswell", intel, 0x0003_06c3, false, 0x0f, 0x06, 0x3c),
        ];
        for (name, vendor, eax, avx512, want, family, model) in cases {
            let mut id = with_cpu(vendor, eax);
            if !avx512 {
                id = without_avx512(id);
            }
            assert_eq!(id.family(), family, "{name} aile");
            assert_eq!(id.model(), model, "{name} model");
            assert_eq!(bits(&id), want, "{name}");
        }
    }

    /// dav1d AVX2 bayrağı BMI1 ve BMI2'yi de ister (`src/x86/cpu.c:78`):
    /// BMI'siz sanal makinede AVX2 (ve AVX512ICL) yoktur.
    #[test]
    fn avx2_bmi1_bmi2_ister() {
        for (name, bit) in [("BMI1", 3), ("AVX2", 5), ("BMI2", 8)] {
            let mut id = ZEN5;
            id.leaf7_ebx &= !(1 << bit);
            let got = id.flags_from(CpuFlags::NONE);
            assert!(!got.contains(CpuFlags::AVX2), "{name} yokken AVX2 olmamalı");
            assert!(
                !got.contains(CpuFlags::AVX512ICL),
                "{name} yokken AVX512ICL olmamalı"
            );
            assert!(got.contains(CpuFlags::SSE4_1));
        }
        assert_eq!(LEAF7_EBX_AVX2, (1 << 3) | (1 << 5) | (1 << 8));
    }

    /// OSXSAVE/AVX biti, XCR0 XMM/YMM ve `max_leaf >= 7` ayrı ayrı gereklidir.
    #[test]
    fn avx2_isletim_sistemi_durumu_ister() {
        for bit in [27, 28] {
            let mut id = ZEN5;
            id.leaf1_ecx &= !(1 << bit);
            assert_eq!(bits(&id), 0x07, "ecx {bit}");
        }
        assert_eq!(LEAF1_ECX_OSXSAVE_AVX, (1 << 27) | (1 << 28));
        let mut id = ZEN5;
        id.os_xmm_ymm = false;
        assert_eq!(bits(&id), 0x07, "XCR0 XMM/YMM yok");
        let mut id = ZEN5;
        id.max_leaf = 6;
        assert_eq!(bits(&id), 0x07, "yaprak 7 yok");
        let mut id = ZEN5;
        id.os_zmm_opmask = false;
        assert_eq!(bits(&id), 0x0f, "XCR0 ZMM/OPMASK yok");
    }

    /// AVX512ICL'nin 14 alt özelliğinden herhangi biri eksikse bayrak düşer,
    /// AVX2 kalır.
    #[test]
    fn avx512icl_on_dort_alt_ozellik_ister() {
        let mut ebx = 0;
        let mut ecx = 0;
        for (name, reg, bit) in AVX512ICL_FEATURES {
            let mut id = ZEN5;
            match reg {
                Leaf7Reg::Ebx => {
                    id.leaf7_ebx &= !(1 << bit);
                    ebx |= 1 << bit;
                }
                Leaf7Reg::Ecx => {
                    id.leaf7_ecx &= !(1 << bit);
                    ecx |= 1 << bit;
                }
            }
            assert_eq!(bits(&id), 0x0f, "{name} yokken AVX512ICL olmamalı");
        }
        assert_eq!(ebx, LEAF7_EBX_AVX512ICL, "EBX maskesi 6 alt özellik");
        assert_eq!(ecx, LEAF7_ECX_AVX512ICL, "ECX maskesi 8 alt özellik");
    }

    /// SSSE3 = SSE3 + SSSE3; SSE4.1 SSSE3'e bağlıdır; hepsi SSE2'ye bağlıdır
    /// (`src/x86/cpu.c:63-70`). AVX2 bu zincirden bağımsızdır (satır 73).
    #[test]
    fn sse_zinciri_dav1d_sirasiyla() {
        assert_eq!(LEAF1_ECX_SSSE3, (1 << 0) | (1 << 9));
        assert_eq!(LEAF1_ECX_SSE41, 1 << 19);
        assert_eq!(LEAF1_EDX_SSE2, (1 << 15) | (1 << 25) | (1 << 26));
        for (name, bit, want) in [("SSE3", 0, 0x09), ("SSSE3", 9, 0x09), ("SSE4.1", 19, 0x0b)] {
            let mut id = without_avx512(ZEN5);
            id.leaf1_ecx &= !(1 << bit);
            assert_eq!(bits(&id), want, "{name}");
        }
        for (name, bit) in [("CMOV", 15), ("SSE", 25), ("SSE2", 26)] {
            let mut id = without_avx512(ZEN5);
            id.leaf1_edx &= !(1 << bit);
            assert_eq!(bits(&id), 0x08, "{name}");
            // Derleme zamanı varsayılanı (x86_64: SSE2) korunur (`src/x86/cpu.c:56`).
            assert!(id.flags_from(CpuFlags::SSE2).contains(CpuFlags::SSE2));
        }
        let mut id = ZEN5;
        id.max_leaf = 0;
        assert_eq!(bits(&id), 0);
        assert_eq!(id.flags_from(CpuFlags::SSE2), CpuFlags::SSE2);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn varsayilan_bayraklar_x86_64_tabani() {
        assert!(X86CpuId::default_flags().contains(CpuFlags::SSE2));
        assert!(CpuFlags::from_dav1d_x86_bits(0x1f).contains(X86CpuId::default_flags()));
    }

    /// DSP-04 / ADR-0003 İ-8: std `is_x86_feature_detected!`'ın XCR0 denetimi.
    ///
    /// std (1.99.1) AVX ailesini ve `xsave`'i yalnız `XCR0 & 6 == 6` iken,
    /// AVX-512'yi yalnız ayrıca `XCR0 & 0xe0 == 0xe0` (+ FMA, F16C) iken açar.
    /// Bu makinede hangi değer dönerse dönsün aşağıdaki çıkarımlar sağlanır;
    /// ham CPUID biti olup std'nin bildirmediği bir AVX özelliği işletim
    /// sisteminin durumu kaydetmediğini gösterir ve tespit onu kullanmaz.
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    #[test]
    fn std_tespiti_isletim_sistemi_durumunu_icerir() {
        use std::arch::is_x86_feature_detected as d;
        let id = X86CpuId::read();
        assert_eq!(id.os_xmm_ymm, d!("xsave"));
        for (name, on) in [("avx", d!("avx")), ("avx2", d!("avx2")), ("fma", d!("fma"))] {
            assert!(!on || d!("xsave"), "{name} XCR0 XMM/YMM olmadan açık");
        }
        let avx512 = [
            d!("avx512f"),
            d!("avx512cd"),
            d!("avx512bw"),
            d!("avx512dq"),
            d!("avx512vl"),
            d!("avx512ifma"),
            d!("avx512vbmi"),
            d!("avx512vbmi2"),
            d!("avx512vnni"),
            d!("avx512bitalg"),
            d!("avx512vpopcntdq"),
        ];
        for on in avx512 {
            assert!(!on || (d!("xsave") && d!("avx512f") && d!("fma") && d!("f16c")));
        }
        // Ham bitle std aynı yöndedir: std açıksa ham bit de vardır.
        assert!(!d!("avx2") || id.leaf7_ebx & (1 << 5) != 0);
        assert!(!d!("avx") || id.leaf1_ecx & (1 << 28) != 0);
        assert!(!d!("avx512f") || id.leaf7_ebx & (1 << 16) != 0);
        // Ham bit var ama işletim sistemi kaydetmiyorsa std kapalıdır ve
        // dav1d eşdeğeri bayrak da düşer.
        if id.leaf1_ecx & (1 << 28) != 0 && !d!("xsave") {
            assert!(!id.flags().contains(CpuFlags::AVX2));
        }
    }

    /// Tespit edilen dav1d bayrakları std'nin bildirdiği özelliklerle
    /// tutarlıdır (fazla raporlama yok).
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    #[test]
    fn tespit_std_ile_tutarli() {
        use std::arch::is_x86_feature_detected as d;
        let flags = X86CpuId::read().flags();
        if flags.contains(CpuFlags::SSSE3) {
            assert!(d!("sse3") && d!("ssse3"));
        }
        if flags.contains(CpuFlags::SSE4_1) {
            assert!(d!("sse4.1"));
        }
        if flags.contains(CpuFlags::AVX2) {
            assert!(d!("avx") && d!("avx2") && d!("bmi1") && d!("bmi2"));
        }
        if flags.contains(CpuFlags::AVX512ICL) {
            assert!(
                d!("avx512f")
                    && d!("avx512cd")
                    && d!("avx512bw")
                    && d!("avx512dq")
                    && d!("avx512vl")
                    && d!("avx512ifma")
                    && d!("avx512vbmi")
                    && d!("avx512vbmi2")
                    && d!("avx512vnni")
                    && d!("avx512bitalg")
                    && d!("avx512vpopcntdq")
                    && d!("gfni")
                    && d!("vaes")
                    && d!("vpclmulqdq")
            );
        }
    }
}
