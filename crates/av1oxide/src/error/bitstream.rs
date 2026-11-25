//! Bit akışı hata modeli (F2a / S2a.1).
//!
//! AV1 bit akışı **güvenilmeyen girdidir** (GUV-01): her okuma sınır denetimlidir
//! ve bozuk/kesik veri panik değil [`Error::Bitstream`](crate::error::Error::Bitstream)
//! üretir. Hata, **bit konumu** ve **neden** taşır; böylece hangi alanın
//! bozuk olduğu kaynak satırına kadar izlenebilir (DOG-01).
//!
//! ## Neden ayrı tip?
//!
//! [`Error::InvalidData`](crate::error::Error::InvalidData) yalnızca sabit bir
//! metin taşır; bit okuyucu/OBU/sequence header katmanları ise hata ayıklaması
//! için **offset** ve **ayrıştırma kuralı** bilgisine ihtiyaç duyar. Bu tip o
//! bilgiyi taşır, `Copy`'dir ve `Result` yolunda ek tahsis yapmaz.
//!
//! ## Upstream karşılığı
//!
//! dav1d `src/getbits.c` hataları `GetBits::error` bayrağıyla, `src/obu.c`
//! `parse_seq_hdr` ise `DAV1D_ERR(EINVAL)` ile bildirir (satır 72–300, 83–84,
//! 302–339). Buradaki model aynı reddetme noktalarını **konumlu** hata olarak
//! temsil eder; davranış farkı yoktur.

#![forbid(unsafe_code)]

use core::fmt;

/// Bit akışı hatasının nedeni.
///
/// Her varyant, AV1 söz dizimindeki tek bir reddetme noktasına karşılık gelir;
/// hangi dav1d satırıyla eşleştiği [`super::Error::Bitstream`] dokümanındadır.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BitstreamErrorKind {
    /// Okuma tamponun sonunu aştı (`f(n)` için yeterli bit yok).
    ///
    /// dav1d: `refill()` `c->error = 1` (getbits.c 65–78) / `dav1d_get_bit`
    /// (getbits.c 47–63).
    UnexpectedEnd,
    /// `obu_forbidden_bit` 1; AV1 §5.3.2'ye göre 0 olmalıdır.
    ForbiddenBit,
    /// OBU boyut alanı, pakette kalan bayttan büyük (dav1d `obu.c` 1179–1183).
    ObuSizeOverflow {
        /// Bildirilen boyut (bayt).
        declared: usize,
        /// Pakette kalan bayt.
        remaining: usize,
    },
    /// `leb128()` 8 bayttan uzun ya da son bayrak biti hâlâ 1 (dav1d getbits.c 95–112).
    Leb128TooLong,
    /// `leb128()` değeri `u32` aralığını aştı (dav1d getbits.c 95–112).
    Leb128Overflow,
    /// `uvlc()` 32 sıfır bitiyle taştı (dav1d getbits.c 125–136, `UINT32_MAX` sentinel).
    UvlcOverflow,
    /// `f(n)` için `n` 0–32 dışında (dav1d `assert(n > 0 && n <= 32)` getbits.c 80–93).
    InvalidBitCount(u32),
    /// `ns(max)` için `max < 2` (dav1d `assert(max > 1)` getbits.c 114–123).
    InvalidUniformMax(u32),
    /// `su(ref, n)` parametreleri taşma üretecek kadar büyük (savunmacı sınır).
    InvalidSubexp,
    /// Katı kipte `trailing_bits` kuralı ihlal edildi (dav1d obu.c 48–70).
    TrailingBits,
    /// Sequence header alanı, dav1d'nin reddettiği bir değer taşıyor.
    ///
    /// Metin, ilgili kuralı ve dav1d kaynak satırını taşır (ör. `profile > 2`,
    /// `reduced_still_picture_header && !still_picture`).
    InvalidSequenceHeader(&'static str),
    /// Frame header alanı, dav1d `parse_frame_hdr` (`src/obu.c` 409–1166)
    /// içindeki bir reddetme kuralını ihlal ediyor.
    ///
    /// Metin, ilgili kuralı ve dav1d kaynak satırını taşır.
    InvalidFrameHeader(&'static str),
    /// Tile group başlığı/verisi dav1d `parse_tile_hdr` (`src/obu.c`
    /// 1157–1166) veya AV1 §5.10.1 sınır kurallarını ihlal ediyor.
    InvalidTileGroup(&'static str),
}

impl BitstreamErrorKind {
    /// Kararlı, Türkçe tanı adı (günlük/rapor çıktıları için).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnexpectedEnd => "beklenmedik veri sonu",
            Self::ForbiddenBit => "obu_forbidden_bit 1",
            Self::ObuSizeOverflow { .. } => "OBU boyut alanı kalan baytı aşıyor",
            Self::Leb128TooLong => "leb128 8 bayttan uzun",
            Self::Leb128Overflow => "leb128 değeri u32 aralığını aştı",
            Self::UvlcOverflow => "uvlc 32 sıfır bitiyle taştı",
            Self::InvalidBitCount(_) => "f(n) bit sayısı 0..=32 dışında",
            Self::InvalidUniformMax(_) => "ns(max) için max < 2",
            Self::InvalidSubexp => "su(ref, n) parametreleri geçersiz",
            Self::TrailingBits => "trailing_bits kuralı ihlal edildi",
            Self::InvalidSequenceHeader(_) => "geçersiz sequence header alanı",
            Self::InvalidFrameHeader(_) => "geçersiz frame header alanı",
            Self::InvalidTileGroup(_) => "geçersiz tile group",
        }
    }
}

impl fmt::Display for BitstreamErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ObuSizeOverflow {
                declared,
                remaining,
            } => {
                write!(f, "OBU boyutu {declared} bayt, kalan {remaining} bayt")
            }
            Self::InvalidBitCount(n) => write!(f, "f({n}) geçersiz (0..=32)"),
            Self::InvalidUniformMax(max) => write!(f, "ns({max}) geçersiz (max > 1 olmalı)"),
            Self::InvalidSequenceHeader(why) => f.write_str(why),
            Self::InvalidFrameHeader(why) => f.write_str(why),
            Self::InvalidTileGroup(why) => f.write_str(why),
            other => f.write_str(other.as_str()),
        }
    }
}

/// Konumlu bit akışı hatası: neden + bit konumu.
///
/// `bit_offset`, okumanın yapıldığı tamponun başından itibaren **bitten**
/// sayılır (dav1d `dav1d_get_bits_pos`, getbits.h 69–71 ile aynı taban).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BitstreamError {
    /// Hatanın nedeni.
    pub kind: BitstreamErrorKind,
    /// Hatanın oluştuğu bit konumu.
    pub bit_offset: usize,
}

impl BitstreamError {
    /// Yeni konumlu hata.
    #[must_use]
    pub const fn new(kind: BitstreamErrorKind, bit_offset: usize) -> Self {
        Self { kind, bit_offset }
    }
}

impl fmt::Display for BitstreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (bit {})", self.kind, self.bit_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::{BitstreamError, BitstreamErrorKind};

    #[test]
    fn her_tur_metin_ve_konum_tasir() {
        let cases = [
            BitstreamErrorKind::UnexpectedEnd,
            BitstreamErrorKind::ForbiddenBit,
            BitstreamErrorKind::ObuSizeOverflow {
                declared: 9,
                remaining: 3,
            },
            BitstreamErrorKind::Leb128TooLong,
            BitstreamErrorKind::Leb128Overflow,
            BitstreamErrorKind::UvlcOverflow,
            BitstreamErrorKind::InvalidBitCount(33),
            BitstreamErrorKind::InvalidUniformMax(0),
            BitstreamErrorKind::InvalidSubexp,
            BitstreamErrorKind::TrailingBits,
            BitstreamErrorKind::InvalidSequenceHeader("profile > 2"),
        ];
        for kind in cases {
            let err = BitstreamError::new(kind, 123);
            let text = err.to_string();
            assert!(!text.is_empty(), "boş hata metni: {kind:?}");
            assert!(text.contains("bit 123"), "konum yok: {text}");
            assert!(!kind.as_str().is_empty());
        }
    }

    #[test]
    fn esitlik_neden_ve_konumu_birlikte_karsilastirir() {
        let a = BitstreamError::new(BitstreamErrorKind::UnexpectedEnd, 8);
        let b = BitstreamError::new(BitstreamErrorKind::UnexpectedEnd, 8);
        let c = BitstreamError::new(BitstreamErrorKind::UnexpectedEnd, 9);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
