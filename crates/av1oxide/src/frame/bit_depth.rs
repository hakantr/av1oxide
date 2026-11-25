//! Bit derinliği tipi (BLK-04).
//!
//! AV1 üç bit derinliği tanımlar: **8, 10 ve 12**. Bu tip, bit derinliğini
//! `u32`/`usize` gibi çıplak bir sayı olarak taşımak yerine **tiple** ifade eder;
//! böylece "kaç bayt/örnek", "en büyük geçerli örnek değeri" gibi soruların
//! cevabı tek yerde durur.
//!
//! F2'de sıcak yollar `const BITDEPTH` generics ile monomorfize edilecektir
//! (DOG-05: LLVM çıktısına bakılarak doğrulanır). Bu tip, o
//! generiklere kaynaklık eden **runtime** temsildir.

#![forbid(unsafe_code)]

use core::fmt;

/// AV1'in geçerli bit derinlikleri (BLK-04).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BitDepth {
    /// 8 bit — örnek başına 1 bayt depolama.
    Eight,
    /// 10 bit — örnek başına 2 bayt depolama (`u16` kapsayıcı).
    Ten,
    /// 12 bit — örnek başına 2 bayt depolama (`u16` kapsayıcı).
    Twelve,
}

impl BitDepth {
    /// Tüm geçerli bit derinlikleri, artan sırada.
    pub const ALL: [Self; 3] = [Self::Eight, Self::Ten, Self::Twelve];

    /// Bit cinsinden derinlik: 8, 10 veya 12.
    #[must_use]
    pub const fn bits(self) -> u32 {
        match self {
            Self::Eight => 8,
            Self::Ten => 10,
            Self::Twelve => 12,
        }
    }

    /// Bit derinliğinden tip üretir; 8/10/12 dışındaki değerler `None`'dur.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Option<Self> {
        match bits {
            8 => Some(Self::Eight),
            10 => Some(Self::Ten),
            12 => Some(Self::Twelve),
            _ => None,
        }
    }

    /// Örnek başına **depolama** baytı: 8-bit'te 1, 10/12-bit'te 2.
    #[must_use]
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::Eight => 1,
            Self::Ten | Self::Twelve => 2,
        }
    }

    /// Örnek kapsayıcısının ham bit genişliği: 8 veya 16.
    ///
    /// **Uyarı (BLK-04):** 16 bitlik kapsayıcı, 16-bit görüntü demek **değildir**;
    /// bkz. [`BitDepth::max_sample`].
    #[must_use]
    pub const fn container_bits(self) -> u32 {
        match self {
            Self::Eight => 8,
            Self::Ten | Self::Twelve => 16,
        }
    }

    /// Depolamanın `u16` hücre mi kullandığı (`bytes_per_sample() == 2` ile aynı bilgi).
    #[must_use]
    pub const fn uses_u16_storage(self) -> bool {
        matches!(self, Self::Ten | Self::Twelve)
    }

    /// En büyük **geçerli** örnek değeri: `2^bit_depth − 1` (255 / 1023 / 4095).
    ///
    /// Bu değer 10/12-bit'te `u16::MAX`'ten küçüktür: üst bitler sıfır kalır ve
    /// `u16::MAX` geçerli bir örnek değildir (BLK-04).
    #[must_use]
    pub const fn max_sample(self) -> u32 {
        (1u32 << self.bits()) - 1
    }
}

impl fmt::Display for BitDepth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-bit", self.bits())
    }
}

#[cfg(test)]
mod tests {
    use super::BitDepth;

    #[test]
    fn bits_ve_from_bits_tur_tamligi() {
        for depth in BitDepth::ALL {
            assert_eq!(BitDepth::from_bits(depth.bits()), Some(depth));
        }
        for invalid in [0u32, 1, 9, 11, 13, 16, 32] {
            assert_eq!(BitDepth::from_bits(invalid), None, "geçersiz: {invalid}");
        }
    }

    #[test]
    fn depolama_bayti_ve_kapsayici() {
        assert_eq!(BitDepth::Eight.bytes_per_sample(), 1);
        assert_eq!(BitDepth::Ten.bytes_per_sample(), 2);
        assert_eq!(BitDepth::Twelve.bytes_per_sample(), 2);

        assert!(!BitDepth::Eight.uses_u16_storage());
        assert!(BitDepth::Ten.uses_u16_storage());
        assert!(BitDepth::Twelve.uses_u16_storage());
    }

    #[test]
    fn u16_depolama_16_bit_goruntu_degildir() {
        // BLK-04: kapsayıcı 16 bit olsa da geçerli örnek aralığı bit derinliğiyle sınırlıdır.
        assert_eq!(BitDepth::Eight.max_sample(), 255);
        assert_eq!(BitDepth::Ten.max_sample(), 1023);
        assert_eq!(BitDepth::Twelve.max_sample(), 4095);

        for depth in [BitDepth::Ten, BitDepth::Twelve] {
            assert_eq!(depth.container_bits(), 16);
            assert!(
                depth.max_sample() < u32::from(u16::MAX),
                "{} için en büyük örnek u16::MAX'ten küçük olmalı",
                depth
            );
        }
    }

    #[test]
    fn gorunum_metni() {
        assert_eq!(BitDepth::Ten.to_string(), "10-bit");
    }
}
