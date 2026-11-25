//! CPU quirks tablosu (S0.8, ADR-0002 §4.2, DSP-01/2).
//!
//! Mikro-mimari kısıtları **veri** olarak tutulur: her kayıt üretici, aile ve
//! model aralığı, ön koşul bayrakları, işaretlenecek quirk bayrağı ve dav1d
//! kaynak satırını taşır. [`apply`] tabloyu dav1d'nin bayrak hesabından sonra
//! uygular ([`super::x86::X86CpuId::flags`]); dav1d'de bu kurallar
//! `dav1d_get_cpu_flags_x86` içine gömülüdür (`src/x86/cpu.c:89-94`).
//!
//! İlk ve tek kayıt `SLOW_GATHER`'dır: AuthenticAMD, AVX2 ve aile ≤ 0x19
//! (Excavator, Zen, Zen+, Zen 2, Zen 3, Zen 3+, Zen 4). Aile hesabı dav1d
//! gibidir: `((eax >> 8) & 0xf) + ((eax >> 20) & 0xff)` (`src/x86/cpu.c:61`).
//! Zen 5 (aile 0x1A) ve Intel bu kısıtı almaz. Kanıt düzeyi dav1d kaynağıdır
//! (ADR-0002 §4.5.4 E3). P36'dan beri bayrak dav1d `mc_init_x86` gibi
//! AVX512ICL `warp8x8`, `warp8x8t` ve `resize` adaylarının kurulmasını engeller
//! ([`crate::dsp::IsaCandidate::yavas_toplamada_kurulmaz`]).

#![forbid(unsafe_code)]

use super::CpuFlags;

/// ADR-0002 §4.2 şemasının bu projede tutulan alanları.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuQuirk {
    /// Kalıcı kimlik (`cpu_quirk_id`).
    pub id: &'static str,
    /// CPUID 0 üretici dizesi (`vendor`), 12 bayt.
    pub vendor: [u8; 12],
    /// Aile aralığı, kapalı (`family`): dav1d aile hesabıyla.
    pub family: (u32, u32),
    /// Model aralığı, kapalı (`model_mask`); `(0, 0xff)` = bütün modeller.
    pub model: (u32, u32),
    /// Kısıtın uygulanması için gereken bayraklar (dav1d: `flags & AVX2`).
    pub requires: CpuFlags,
    /// İşaretlenecek quirk bayrağı (`quirk_flag`).
    pub flag: CpuFlags,
    /// Etkilenen işlevler (`affected_functions`).
    pub affected: &'static str,
    /// Gerekçe (`rationale`).
    pub rationale: &'static str,
    /// Kanıt düzeyi ve kaynağı (`evidence_level`, `evidence_source`).
    pub evidence: &'static str,
    /// dav1d karşılığı (`dav1d_equivalent`): kaynak satırı.
    pub dav1d_source: &'static str,
    /// Sapma gerekçesi (`deviation_reason`); sapma yoksa açıkça "yok".
    pub deviation: &'static str,
}

impl CpuQuirk {
    /// Kayıt bu işlemciye ve bayrak kümesine uyuyor mu?
    #[must_use]
    pub const fn matches(
        &self,
        vendor: &[u8; 12],
        family: u32,
        model: u32,
        flags: CpuFlags,
    ) -> bool {
        let mut i = 0;
        while i < 12 {
            if self.vendor[i] != vendor[i] {
                return false;
            }
            i += 1;
        }
        family >= self.family.0
            && family <= self.family.1
            && model >= self.model.0
            && model <= self.model.1
            && flags.contains(self.requires)
    }
}

/// CPU quirks tablosu.
pub const CPU_QUIRKS: [CpuQuirk; 1] = [CpuQuirk {
    id: "amd-slow-gather",
    vendor: *b"AuthenticAMD",
    family: (0, 0x19),
    model: (0, 0xff),
    requires: CpuFlags::AVX2,
    flag: CpuFlags::SLOW_GATHER,
    affected: "AVX512ICL warp8x8, warp8x8t ve resize (P36; dav1d mc_init_x86)",
    rationale: "Excavator, Zen, Zen+, Zen 2, Zen 3, Zen 3+, Zen 4: gather komutları gerileme yaratacak kadar yavaş",
    evidence: "E3: dav1d kaynağı; bu projede ölçülmedi",
    dav1d_source: "src/x86/cpu.c:89-94",
    deviation: "yok (dav1d kuralıyla birebir)",
}];

/// Tablodaki uyan her kaydın bayrağını ekler.
#[must_use]
pub const fn apply(vendor: &[u8; 12], family: u32, model: u32, flags: CpuFlags) -> CpuFlags {
    let mut out = flags;
    let mut i = 0;
    while i < CPU_QUIRKS.len() {
        if CPU_QUIRKS[i].matches(vendor, family, model, flags) {
            out = out.union(CPU_QUIRKS[i].flag);
        }
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{CPU_QUIRKS, apply};
    use crate::cpu::CpuFlags;

    const AMD: [u8; 12] = *b"AuthenticAMD";
    const INTEL: [u8; 12] = *b"GenuineIntel";
    const HYGON: [u8; 12] = *b"HygonGenuine";

    fn slow_gather(vendor: &[u8; 12], family: u32, model: u32, flags: CpuFlags) -> bool {
        apply(vendor, family, model, flags).contains(CpuFlags::SLOW_GATHER)
    }

    /// Bilinen aile/model değerleri (AMD aile = temel 0xF + genişletilmiş).
    #[test]
    fn slow_gather_bilinen_ailelerde_dav1d_kurali() {
        let avx2 = CpuFlags::SSE2 | CpuFlags::AVX2;
        // (ad, aile, model, beklenen)
        let cases = [
            ("Excavator", 0x15, 0x60, true),
            ("Zen", 0x17, 0x01, true),
            ("Zen+", 0x17, 0x08, true),
            ("Zen 2", 0x17, 0x71, true),
            ("Zen 3", 0x19, 0x21, true),
            ("Zen 3+", 0x19, 0x44, true),
            ("Zen 4", 0x19, 0x61, true),
            ("Zen 5", 0x1a, 0x44, false),
            ("Zen 5 (mobil)", 0x1a, 0x24, false),
            ("gelecek aile", 0x1b, 0x00, false),
        ];
        for (name, family, model, want) in cases {
            assert_eq!(slow_gather(&AMD, family, model, avx2), want, "{name}");
        }
    }

    #[test]
    fn slow_gather_avx2_ve_uretici_ister() {
        // AVX2'siz AMD (ör. Bulldozer, aile 0x15) kısıt almaz (dav1d `flags & AVX2`).
        assert!(!slow_gather(
            &AMD,
            0x15,
            0x01,
            CpuFlags::SSE2 | CpuFlags::SSE4_1
        ));
        // Intel ve Hygon (aile 0x18, AMD türevi) dav1d'de AuthenticAMD değildir.
        let avx2 = CpuFlags::SSE2 | CpuFlags::AVX2;
        assert!(!slow_gather(&INTEL, 0x06, 0x7e, avx2));
        assert!(!slow_gather(&INTEL, 0x06, 0x3c, avx2));
        assert!(!slow_gather(&HYGON, 0x18, 0x00, avx2));
        // Quirk yalnız bayrak ekler; diğer bitlere dokunmaz.
        assert_eq!(apply(&AMD, 0x17, 0x71, avx2), avx2 | CpuFlags::SLOW_GATHER);
        assert_eq!(apply(&AMD, 0x1a, 0x44, avx2), avx2);
    }

    #[test]
    fn tablo_alanlari_dolu() {
        for quirk in CPU_QUIRKS {
            assert!(!quirk.id.is_empty());
            assert!(quirk.family.0 <= quirk.family.1);
            assert!(quirk.model.0 <= quirk.model.1);
            assert!(quirk.flag.bits().is_power_of_two());
            assert!(!quirk.requires.contains(quirk.flag));
            for text in [
                quirk.affected,
                quirk.rationale,
                quirk.evidence,
                quirk.dav1d_source,
                quirk.deviation,
            ] {
                assert!(!text.is_empty(), "{}: boş alan", quirk.id);
            }
        }
    }
}
