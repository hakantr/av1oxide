//! Chroma biçimleri ve alt örnekleme (BLK-04).
//!
//! BLK-04, 4:2:0 / 4:2:2 / 4:4:4 ve **monochrome** için chroma stride'ının ve
//! kenar/yuvarlama davranışının **veri tipinde ifade edilmesini** ister. Bu modül
//! bunu [`ChromaFormat`] ve onun boyut türetme işleviyle yapar:
//!
//! - Tek sayılı luma boyutlarında chroma **yukarı yuvarlanır** (`⌈w/2⌉`), çünkü
//!   AV1'de tek satır/sütun kalan kenar blokları da kodlanır.
//! - Türetme **checked aritmetiği** kullanır (BLK-05): `usize` taşması panik
//!   değil, [`Error::SizeOverflow`](crate::error::Error::SizeOverflow) üretir.

#![forbid(unsafe_code)]

use crate::error::{Error, Result, SizeStage};

/// AV1'in desteklenen chroma biçimleri (BLK-04).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChromaFormat {
    /// Chroma düzlemi yok; yalnızca Y düzlemi taşınır.
    Monochrome,
    /// 4:2:0 — chroma hem yatay hem dikey yönde yarıya iner.
    C420,
    /// 4:2:2 — chroma yalnızca yatay yönde yarıya iner.
    C422,
    /// 4:4:4 — chroma luma ile aynı çözünürlükte.
    C444,
}

impl ChromaFormat {
    /// Çerçevedeki düzlem sayısı: monochrome'da 1, diğerlerinde 3 (Y, U, V).
    #[must_use]
    pub const fn num_planes(self) -> usize {
        match self {
            Self::Monochrome => 1,
            Self::C420 | Self::C422 | Self::C444 => 3,
        }
    }

    /// Chroma düzlemi var mı?
    #[must_use]
    pub const fn has_chroma(self) -> bool {
        self.num_planes() == 3
    }

    /// Chroma alt örnekleme kaydırması `(log2_yatay, log2_dikey)`.
    ///
    /// Monochrome'da chroma düzlemi olmadığı için `(0, 0)` döner; bu değer
    /// "chroma luma ile aynı" anlamına **gelmez**, yalnızca kaydırma yokluğudur.
    /// Boyut türetmek için [`ChromaFormat::chroma_dimensions`] kullanılmalıdır.
    #[must_use]
    pub const fn chroma_shift(self) -> (u32, u32) {
        match self {
            Self::Monochrome | Self::C444 => (0, 0),
            Self::C422 => (1, 0),
            Self::C420 => (1, 1),
        }
    }

    /// Luma boyutlarından chroma **örnek** boyutlarını türetir (checked).
    ///
    /// - 4:2:0 → `(⌈w/2⌉, ⌈h/2⌉)`
    /// - 4:2:2 → `(⌈w/2⌉, h)`
    /// - 4:4:4 → `(w, h)`
    /// - monochrome → `None` (chroma düzlemi yok)
    ///
    /// # Hatalar
    ///
    /// `w` veya `h` sıfırsa [`Error::InvalidArgument`]; `w + 1` taşarsa
    /// [`Error::SizeOverflow`] (`SizeStage::ChromaSubsample`).
    pub fn chroma_dimensions(
        self,
        luma_width: usize,
        luma_height: usize,
    ) -> Result<Option<(usize, usize)>> {
        if luma_width == 0 || luma_height == 0 {
            return Err(Error::InvalidArgument(
                "chroma boyutu türetilemez: luma genişlik/yükseklik sıfır",
            ));
        }
        if !self.has_chroma() {
            return Ok(None);
        }

        let (h_shift, v_shift) = self.chroma_shift();
        let width = if h_shift == 0 {
            luma_width
        } else {
            luma_width.checked_add(1).ok_or(Error::SizeOverflow {
                stage: SizeStage::ChromaSubsample,
                lhs: luma_width,
                rhs: 1,
            })? >> h_shift
        };
        let height = if v_shift == 0 {
            luma_height
        } else {
            luma_height.checked_add(1).ok_or(Error::SizeOverflow {
                stage: SizeStage::ChromaSubsample,
                lhs: luma_height,
                rhs: 1,
            })? >> v_shift
        };

        Ok(Some((width, height)))
    }
}

#[cfg(test)]
mod tests {
    use super::ChromaFormat;
    use crate::error::{Error, SizeStage};

    #[test]
    fn duzlem_sayilari() {
        assert_eq!(ChromaFormat::Monochrome.num_planes(), 1);
        assert_eq!(ChromaFormat::C420.num_planes(), 3);
        assert_eq!(ChromaFormat::C422.num_planes(), 3);
        assert_eq!(ChromaFormat::C444.num_planes(), 3);
        assert!(!ChromaFormat::Monochrome.has_chroma());
        assert!(ChromaFormat::C420.has_chroma());
    }

    #[test]
    fn chroma_boyutlari_tek_sayilarda_yukari_yuvarlanir() {
        assert_eq!(
            ChromaFormat::C420.chroma_dimensions(1920, 1080),
            Ok(Some((960, 540)))
        );
        assert_eq!(
            ChromaFormat::C420.chroma_dimensions(7, 5),
            Ok(Some((4, 3))),
            "tek boyutlar yukarı yuvarlanmalı (⌈7/2⌉=4, ⌈5/2⌉=3)"
        );
        assert_eq!(
            ChromaFormat::C422.chroma_dimensions(7, 5),
            Ok(Some((4, 5))),
            "4:2:2 yalnızca yatayda yarıya iner"
        );
        assert_eq!(ChromaFormat::C444.chroma_dimensions(7, 5), Ok(Some((7, 5))));
        assert_eq!(ChromaFormat::Monochrome.chroma_dimensions(7, 5), Ok(None));
    }

    #[test]
    fn sifir_boyut_reddedilir() {
        assert_eq!(
            ChromaFormat::C420.chroma_dimensions(0, 4),
            Err(Error::InvalidArgument(
                "chroma boyutu türetilemez: luma genişlik/yükseklik sıfır"
            ))
        );
    }

    #[test]
    fn tasma_panik_uretmez() {
        let result = ChromaFormat::C420.chroma_dimensions(usize::MAX, 4);
        assert_eq!(
            result,
            Err(Error::SizeOverflow {
                stage: SizeStage::ChromaSubsample,
                lhs: usize::MAX,
                rhs: 1,
            })
        );
    }
}
