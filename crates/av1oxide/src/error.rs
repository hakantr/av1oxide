//! Hata modeli (F1.5 / S15.2).
//!
//! Bu modül, crate'in **tek hata tipini** ([`Error`]) ve kısa yolu ([`Result`])
//! tanımlar. Tasarım kuralları:
//!
//! - **Panik yok (BLK-05):** Boyut/stride/tahsis aritmetiğinde taşma, doyma veya
//!   negatif sonuç **`Result` ile açıklayıcı hata** döner; `unwrap`/`expect`/panik
//!   üretim yolunda kullanılmaz. Taşmanın hangi adımda olduğu [`SizeStage`] ile
//!   bildirilir.
//! - **Kaynak bütçesi (BLK-07):** Kullanıcı tanımlı bellek sınırı aşılırsa
//!   [`Error::BudgetExceeded`] döner; AV1'in **normatif** sınırları ile
//!   uygulamanın **kullanıcı tanımlı** sınırları ayrı kavramlardır. Büyük
//!   ayırmalar düşebilirdir: ayırıcı başarısız olursa
//!   [`Error::AllocationFailed`] döner (P38).
//! - **Bozuk veri (GUV-01):** Güvenilmeyen AV1 verisi [`Error::InvalidData`] ile
//!   reddedilir; bit okuyucu/OBU/sequence header katmanı ise konumlu
//!   [`Error::Bitstream`] hatası üretir (F2a / S2a.1).
//! - **Durum ihlali (BLK-01):** Çağrı sırası sözleşmeye uymuyorsa
//!   [`Error::InvalidState`] döner (örn. `flush` sonrası `send_packet`).
//!
//! Hata tipi `#[non_exhaustive]`'tir: yeni varyant eklemek kırıcı değişiklik
//! sayılmaz, çağıran taraf `_ => ...` kolu bırakmalıdır.

#![forbid(unsafe_code)]

mod bitstream;

pub use bitstream::{BitstreamError, BitstreamErrorKind};

use core::fmt;

/// Boyut hesabı zincirinin hangi adımında taşma oluştuğu (BLK-05).
///
/// BLK-05'in zorunlu kıldığı zincir:
/// `genişlik × yükseklik → satır stride → toplam tahsis (padding/align dâhil)`.
/// Chroma alt örneklemesi bu zincirin önüne, uygulamanın kendi sınır kontrolü
/// ise sonuna eklenir.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SizeStage {
    /// Chroma düzlemi boyutları luma'dan türetilirken (`⌈genişlik/2⌉` vb.).
    ChromaSubsample,
    /// Örnek başına bayt ile çarpım: `genişlik × bayt/örnek`.
    RowBytes,
    /// Satır baytının hizalamaya yuvarlanması (stride hesabı).
    Stride,
    /// Düzlem alanı: `stride × yükseklik`.
    PlaneArea,
    /// Padding ve düzlemlerin toplamı: son tahsis boyutu.
    TotalAllocation,
}

impl SizeStage {
    /// Aşamanın kararlı, Türkçe tanı adı (günlük/rapor çıktıları için).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ChromaSubsample => "chroma alt örnekleme",
            Self::RowBytes => "satır baytı (genişlik × bayt/örnek)",
            Self::Stride => "stride hizalama",
            Self::PlaneArea => "düzlem alanı (stride × yükseklik)",
            Self::TotalAllocation => "toplam tahsis",
        }
    }
}

impl fmt::Display for SizeStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// av1oxide'in tek hata tipi.
///
/// Varyantlar, ilgili ilkeye göre gruplanmıştır; hiçbiri panik üretmez.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// İstenen işlev bu sürümde/derlemede yok.
    ///
    /// F1.5 iskeletinde gerçek çözme yolları **bilinçli olarak** bu hatayı döner;
    /// metin, işlevin hangi fazda geleceğini söyler (`todo!()` yerine sözleşme).
    Unsupported(&'static str),
    /// Çağrı sırası sözleşmeye uymuyor (BLK-01 durum makinesi).
    ///
    /// Örnek: `flush()` sonrası `send_packet()` çağrılması.
    InvalidState(&'static str),
    /// Çağrı parametresi geçersiz (sıfır boyut, 2'nin kuvveti olmayan hizalama…).
    InvalidArgument(&'static str),
    /// Girdi bit akışı AV1 sözleşmesine uymuyor (F2'de ayrıştırıcıdan gelir, GUV-01).
    InvalidData(&'static str),
    /// Bit okuyucu/OBU/sequence header katmanının konumlu hatası (F2a / S2a.1).
    ///
    /// [`BitstreamError`] neden ve **bit konumu** taşır; böylece bozuk alan
    /// dav1d kaynak satırına kadar izlenebilir. Her okuma sınır denetimlidir
    /// ve bozuk girdi panik üretmez (GUV-01, BLK-05).
    Bitstream(BitstreamError),
    /// Boyut/stride/tahsis aritmetiği taştı (BLK-05). Hangi adımda olduğu
    /// [`SizeStage`] ile bildirilir; `lhs`/`rhs` taşan işlemin operandlarıdır.
    SizeOverflow {
        /// Taşmanın oluştuğu hesap adımı.
        stage: SizeStage,
        /// İşlemin sol operandı (ör. genişlik veya stride).
        lhs: usize,
        /// İşlemin sağ operandı (ör. bayt/örnek veya yükseklik).
        rhs: usize,
    },
    /// Kullanıcı tanımlı tahsis bütçesi aşıldı (BLK-07). Panik yok, `Result` var.
    BudgetExceeded {
        /// İstenen toplam bayt.
        requested: usize,
        /// Kullanıcının verdiği üst sınır.
        limit: usize,
    },
    /// Bellek ayırıcısı isteği karşılayamadı (P38, GUV-09): düşebilir ayırma
    /// (`try_reserve`) başarısız oldu; panik/abort yerine bu hata döner.
    AllocationFailed {
        /// İstenen ayırma (bayt).
        requested: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(what) => write!(f, "desteklenmeyen işlev: {what}"),
            Self::InvalidState(what) => write!(f, "geçersiz çağrı sırası: {what}"),
            Self::InvalidArgument(what) => write!(f, "geçersiz parametre: {what}"),
            Self::InvalidData(what) => write!(f, "bozuk/geçersiz AV1 verisi: {what}"),
            Self::Bitstream(err) => write!(f, "bit akışı hatası: {err}"),
            Self::SizeOverflow { stage, lhs, rhs } => write!(
                f,
                "boyut taşması ({stage}): {lhs} × {rhs} usize aralığına sığmıyor"
            ),
            Self::BudgetExceeded { requested, limit } => write!(
                f,
                "tahsis bütçesi aşıldı: istenen {requested} bayt, sınır {limit} bayt"
            ),
            Self::AllocationFailed { requested } => {
                write!(f, "bellek ayrılamadı: istenen {requested} bayt")
            }
        }
    }
}

impl core::error::Error for Error {}

impl From<BitstreamError> for Error {
    fn from(err: BitstreamError) -> Self {
        Self::Bitstream(err)
    }
}

/// av1oxide sonuç tipi: [`Error`] ile kısa yol.
pub type Result<T> = core::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;
    use alloc::string::ToString;

    use super::{Error, Result, SizeStage};

    #[test]
    fn boyut_asamasi_adlari_bos_degil() {
        for stage in [
            SizeStage::ChromaSubsample,
            SizeStage::RowBytes,
            SizeStage::Stride,
            SizeStage::PlaneArea,
            SizeStage::TotalAllocation,
        ] {
            assert!(!stage.as_str().is_empty(), "aşama adı boş: {stage:?}");
            assert_eq!(stage.to_string(), stage.as_str());
        }
    }

    #[test]
    fn hata_mesajlari_baglami_tasir() {
        let overflow = Error::SizeOverflow {
            stage: SizeStage::PlaneArea,
            lhs: 1 << 40,
            rhs: 1 << 40,
        };
        let text = overflow.to_string();
        assert!(text.contains("boyut taşması"), "metin: {text}");
        assert!(
            text.contains(SizeStage::PlaneArea.as_str()),
            "metin: {text}"
        );

        let budget = Error::BudgetExceeded {
            requested: 4096,
            limit: 1024,
        };
        assert!(budget.to_string().contains("4096"));
        assert!(budget.to_string().contains("1024"));

        let alloc = Error::AllocationFailed { requested: 8192 };
        assert!(alloc.to_string().contains("8192"));
    }

    #[test]
    fn hata_std_error_olarak_kullanilabilir() {
        let boxed: Box<dyn core::error::Error> = Box::new(Error::Unsupported("test"));
        assert_eq!(boxed.to_string(), "desteklenmeyen işlev: test");
    }

    #[test]
    fn sonuc_tipi_kisa_yol() {
        let ok: Result<u8> = Ok(7);
        assert_eq!(ok, Ok(7));
    }
}
