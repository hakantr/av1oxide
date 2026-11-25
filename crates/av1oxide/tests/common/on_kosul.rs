//! Ön koşul yokluğu sessiz geçmez (P29, TST-01).
//!
//! Bir testin ön koşulu (dav1d oracle verisi, AVX2 / x86 işlemci) yoksa test
//! eskiden `println!(… atlandı); return;` ile "geçti" görünüyordu. Bütün bu
//! noktalar [`eksik`] çağırır; çağıran ardından döner (ya da döngüde
//! `continue` eder).
//!
//! Davranış üç ortam değişkenine bağlıdır (kapılar üçünü de ayarlar):
//!
//! - [`ZORUNLU`] (`AV1OXIDE_ORACLES_REQUIRED`) ayarlıysa eksik ön koşul **FAIL**
//!   olur (panik) — yalnız `(test, neden)` çifti [`BEKLENEN`] listesinde
//!   yazılıysa atlamaya izin verilir.
//! - Ayarlı değilse makinece okunur `ATLANDI <test> <neden>` satırı basılır.
//! - [`KAYIT`] (`AV1OXIDE_ATLAMA_KAYDI`) bir dosya yolu ise her atlama oraya
//!   da eklenir; kapı atlamaları sayar ve beklenen listeyle karşılaştırır.
//!   Zorunlu kipte beklenmeyen eksiklik paniklemeden önce `BEKLENMEYEN <test>
//!   <neden>` satırı yazar (P30 düzeltme 1).
//!
//! Beklenen atlama listeleri `docs/beklenen-atlamalar/<hedef>.txt`: satır başına
//! `<test> <neden>`; `#` yorum. Neden anahtarları sabit kümedir ([`Neden`]).

use std::io::Write as _;

/// Eksik ön koşulu FAIL yapan değişken.
pub const ZORUNLU: &str = "AV1OXIDE_ORACLES_REQUIRED";
/// Bu makinenin beklenen atlama listesinin yolu.
pub const BEKLENEN: &str = "AV1OXIDE_BEKLENEN_ATLAMALAR";
/// Atlamaların eklendiği kayıt dosyası.
pub const KAYIT: &str = "AV1OXIDE_ATLAMA_KAYDI";

/// Atlama nedeni (beklenen atlama listesinin ikinci sütunu).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Neden {
    /// İşlemci AVX2 yolunu açmıyor (x86_64'te).
    Avx2Yok,
    /// dav1d oracle verisi (`.cache/…` ya da `FOO_ORACLE_OUT`) yok.
    OracleVerisiYok,
    /// x86_64 dışı hedef: x86 gövdesi derlenmez.
    X86Degil,
    /// İşlemci (ya da maske) bir ISA düzeyini açmıyor (P30: SSSE3, SSE4.1,
    /// AVX512ICL gövdeleri; x86_64'te).
    IsaYok,
}

impl Neden {
    /// AVX2 eksikliğinin hedefe göre anahtarı: x86_64'te `avx2-yok`, başka
    /// hedefte `x86-degil`.
    #[must_use]
    pub const fn avx2() -> Self {
        if cfg!(target_arch = "x86_64") {
            Self::Avx2Yok
        } else {
            Self::X86Degil
        }
    }

    /// Bir ISA düzeyinin eksikliğinin hedefe göre anahtarı: x86_64'te
    /// `isa-yok`, başka hedefte `x86-degil`.
    #[must_use]
    pub const fn isa() -> Self {
        if cfg!(target_arch = "x86_64") {
            Self::IsaYok
        } else {
            Self::X86Degil
        }
    }

    /// Listede kullanılan sabit anahtar.
    #[must_use]
    pub const fn anahtar(self) -> &'static str {
        match self {
            Self::Avx2Yok => "avx2-yok",
            Self::OracleVerisiYok => "oracle-verisi-yok",
            Self::X86Degil => "x86-degil",
            Self::IsaYok => "isa-yok",
        }
    }
}

/// Ön koşul davranışının ayarı. Testler [`Ayar::ortamdan`] ile ortam
/// değişkenlerinden kurar; sınama ortamı değiştirmeden açıkça kurar
/// (`tests/on_kosul_kaydi.rs`; P30 düzeltme 1).
#[derive(Clone, Debug, Default)]
pub struct Ayar {
    /// [`ZORUNLU`] ayarlı mı?
    pub zorunlu: bool,
    /// [`BEKLENEN`] listesinin yolu.
    pub beklenen: Option<std::path::PathBuf>,
    /// [`KAYIT`] dosyasının yolu.
    pub kayit: Option<std::path::PathBuf>,
}

impl Ayar {
    /// Ortam değişkenlerinden.
    #[must_use]
    pub fn ortamdan() -> Self {
        Self {
            zorunlu: std::env::var_os(ZORUNLU).is_some(),
            beklenen: std::env::var_os(BEKLENEN).map(Into::into),
            kayit: std::env::var_os(KAYIT).map(Into::into),
        }
    }
}

fn beklenen_mi(liste: Option<&std::path::Path>, test: &str, neden: Neden) -> bool {
    let Some(yol) = liste else {
        return false;
    };
    let metin = std::fs::read_to_string(yol)
        .unwrap_or_else(|e| panic!("{BEKLENEN}={}: okunamadı: {e}", yol.display()));
    metin.lines().any(|satir| {
        let mut s = satir.split('#').next().unwrap_or("").split_whitespace();
        s.next() == Some(test) && s.next() == Some(neden.anahtar())
    })
}

/// Ön koşul eksik: [`ZORUNLU`] ayarlıysa ve atlama beklenmiyorsa panik verir;
/// aksi hâlde `ATLANDI <test> <neden>` basar ve [`KAYIT`] dosyasına ekler.
/// `test` biçimi `<test ikilisi>::<işlev>` (ör. `cdef_avx2::yedi_b_dir_…`).
#[track_caller]
pub fn eksik(test: &str, neden: Neden, ayrinti: &str) {
    eksik_ayarla(&Ayar::ortamdan(), test, neden, ayrinti);
}

/// [`eksik`]in açık ayarlı biçimi. Zorunlu kipte beklenmeyen eksiklik
/// paniklemeden **önce** kayda `BEKLENMEYEN <test> <neden>` yazar (P30
/// düzeltme 1, bulgu c): kapı özeti (`on_kosul_ozet`) bunu beklenmeyen sayar
/// ve 1 döndürür; önceden panik kayda düşmediği için özet "beklenmeyen 0"
/// gösteriyordu.
#[track_caller]
pub fn eksik_ayarla(ayar: &Ayar, test: &str, neden: Neden, ayrinti: &str) {
    if ayar.zorunlu && !beklenen_mi(ayar.beklenen.as_deref(), test, neden) {
        if let Some(yol) = &ayar.kayit {
            kayda_ekle(yol, &format!("BEKLENMEYEN {test} {}", neden.anahtar()));
        }
        panic!(
            "ÖN KOŞUL EKSİK {test} {} — {ayrinti} ({ZORUNLU} ayarlı, beklenen atlama listesinde yok)",
            neden.anahtar()
        );
    }
    let satir = format!("ATLANDI {test} {}", neden.anahtar());
    println!("{satir} — {ayrinti}");
    if let Some(yol) = &ayar.kayit {
        kayda_ekle(yol, &satir);
    }
}

/// Kayıt dosyasına satırı ekler: satır ve satır sonu **tek bir `write`**
/// çağrısıyla, `O_APPEND` dosyaya yazılır (P30, bulgu d). Paralel test iş
/// parçacıklarının satırları birleşemez: POSIX'te `O_APPEND` düzenli dosyaya
/// tek yazma, konumlandırma ve yazmayı bölünmez yapar. Önceki `writeln!`
/// satırı ve `\n`'yi ayrı `write(2)` çağrılarıyla yazabiliyordu (macOS'ta
/// "…x86-degilATLANDI …" birleşmesi).
pub fn kayda_ekle(yol: &std::path::Path, satir: &str) {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(yol)
        .unwrap_or_else(|e| panic!("{KAYIT}={}: açılamadı: {e}", yol.display()));
    let mut tampon = String::with_capacity(satir.len() + 1);
    tampon.push_str(satir);
    tampon.push('\n');
    let n = f
        .write(tampon.as_bytes())
        .unwrap_or_else(|e| panic!("{KAYIT}: yazılamadı: {e}"));
    assert_eq!(
        n,
        tampon.len(),
        "{KAYIT}: satır tek yazmada bütün yazılamadı"
    );
}
