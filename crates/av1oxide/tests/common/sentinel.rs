//! Sentinel (koruyucu bellek) ve padding over-read test yardımcıları — SMD-07.
//!
//! Kernel testlerinin üç zorunlu aracı (SMD-07, TST-01):
//!
//! | Araç | Ne yapar | Neyi yakalar |
//! |---|---|---|
//! | [`SentinelBuffer`] | Verinin **önüne ve arkasına** bilinen desende koruyucu bölge koyar | Tampon **taşması** (over-write) |
//! | [`OverReadProbe`] | Girdiyi **sıralı/padding'li** yerleştirir ve çekirdeği **iki farklı padding deseniyle** koşturur | Veri sınırının ötesini **okuma** (over-read) |
//! | [`regresyon_kaydet`] | Hata veren girdiyi diske yazar | Regresyon korpusu kaybı |
//!
//! ## Over-read nasıl ölçülür? (dürüstlük notu)
//!
//! Güvenli Rust'ta **okuma izlenemez**; sentinel yalnız *yazma* taşmasını
//! görebilir. Bu yüzden over-read, dolaylı ama deterministik bir yöntemle
//! ölçülür: aynı veri iki kez, **padding baytları farklı** olacak şekilde
//! girdiye yerleştirilir. Çekirdek yalnız veri aralığını okuyorsa iki koşunun
//! çıktısı **bit-aynı** olur; padding'den bir bayt bile okursa çıktı değişir.
//! Değişim, ilk farklı baytın indeksiyle birlikte [`OverRead`] olarak döner.
//!
//! ## Neden `Result`, neden panik yok?
//!
//! Tespit sonucu **veri** olarak döner (BLK-05 ruhu): test, ihlalin *hangi
//! tarafta* ve *hangi indekste* olduğunu görebilir; böylece hata mesajı kanıt
//! taşır ("bir yerde taştı" değil, "arka koruyucunun 0. baytına yazıldı").

#![forbid(unsafe_code)]

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Varsayılan sentinel deseni (`0xA5` — bit deseni 1010_0101, boş bellekte
/// sık görülen `0x00`/`0xFF` değerlerinden ayırt edilebilir).
pub const VARSAYILAN_SENTINEL: u8 = 0xA5;

/// Varsayılan koruyucu bölge uzunluğu (bayt).
///
/// En geniş desteklenen SIMD erişiminden (AVX-512'de 64 bayt) büyük seçilir;
/// böylece tek komutluk taşmalar koruyucunun içinde kalır ve **yakalanır**.
pub const VARSAYILAN_KORUMA: usize = 64;

/// Over-read ölçümünde kullanılan iki padding deseni.
const PADDING_A: u8 = 0x00;
const PADDING_B: u8 = 0xFF;

/// Koruyucu bölgenin hangi tarafı.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GuardSide {
    /// Verinin önündeki koruyucu (negatif yön taşması).
    Front,
    /// Verinin arkasındaki koruyucu (pozitif yön taşması).
    Back,
}

/// Sentinel ihlali: koruyucu bölgeye yazılmış baytın kanıtı.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SentinelViolation {
    /// İhlalin görüldüğü koruyucu bölge.
    pub side: GuardSide,
    /// İhlalin **ham tampon** içindeki indeksi (0 = ham tamponun ilk baytı).
    pub index: usize,
    /// Koruyucuda beklenen sentinel değeri.
    pub expected: u8,
    /// Koruyucuda bulunan gerçek değer.
    pub found: u8,
}

/// Ön/arka koruyucu bölgeli tampon (SMD-07: yazma taşması tespiti).
///
/// Yerleşim:
///
/// ```text
/// [ ön koruyucu: guard bayt ][ veri: data_len bayt ][ arka koruyucu: guard bayt ]
/// ```
///
/// Koruyucular [`SentinelBuffer::sentinel`] deseniyle doldurulur ve
/// [`SentinelBuffer::verify`] ile denetlenir. `raw*` erişimleri ham tamponu
/// verir; **negatif kontroller** (kasıtlı taşma) bu yolla yapılır.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SentinelBuffer {
    bytes: Vec<u8>,
    data_len: usize,
    guard: usize,
    sentinel: u8,
}

impl SentinelBuffer {
    /// `data_len` baytlık veri + her iki yanda `guard` baytlık koruyucu ayırır.
    #[must_use]
    pub fn new(data_len: usize, guard: usize, sentinel: u8) -> Self {
        let total = guard + data_len + guard;
        Self {
            bytes: vec![sentinel; total],
            data_len,
            guard,
            sentinel,
        }
    }

    /// Varsayılan koruma uzunluğu ve sentinel deseniyle ayırır.
    #[must_use]
    pub fn with_defaults(data_len: usize) -> Self {
        Self::new(data_len, VARSAYILAN_KORUMA, VARSAYILAN_SENTINEL)
    }

    /// Veri uzunluğu (koruyucular hariç).
    #[must_use]
    pub const fn data_len(&self) -> usize {
        self.data_len
    }

    /// Tek taraftaki koruyucu uzunluğu.
    #[must_use]
    pub const fn guard_len(&self) -> usize {
        self.guard
    }

    /// Koruyucularda beklenen desen.
    #[must_use]
    pub const fn sentinel(&self) -> u8 {
        self.sentinel
    }

    /// Verinin ham tampondaki başlangıç indeksi (ön koruyucunun uzunluğu).
    #[must_use]
    pub const fn data_offset(&self) -> usize {
        self.guard
    }

    /// Ham tamponun tamamı: `ön koruyucu + veri + arka koruyucu`.
    #[must_use]
    pub fn raw(&self) -> &[u8] {
        &self.bytes
    }

    /// Ham tampon (değiştirilebilir).
    ///
    /// **Kasıtlı taşma negatif kontrolleri** bu erişimle yazılır; üretim
    /// çekirdekleri yalnız [`SentinelBuffer::data_mut`] görmelidir.
    #[must_use]
    pub fn raw_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    /// Yalnız veri bölgesi.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.bytes[self.guard..self.guard + self.data_len]
    }

    /// Yalnız veri bölgesi (değiştirilebilir) — çekirdeklere verilecek görünüm.
    #[must_use]
    pub fn data_mut(&mut self) -> &mut [u8] {
        let start = self.guard;
        &mut self.bytes[start..start + self.data_len]
    }

    /// Veri bölgesini tek bir değerle doldurur.
    pub fn fill_data(&mut self, value: u8) {
        self.data_mut().fill(value);
    }

    /// Veriyi baştan yazar.
    ///
    /// `src` veri alanına **tam** sığmıyorsa hiçbir şey yazılmaz ve `false`
    /// döner (panik yok).
    pub fn write_data(&mut self, src: &[u8]) -> bool {
        if src.len() != self.data_len {
            return false;
        }
        self.data_mut().copy_from_slice(src);
        true
    }

    /// Koruyucuları denetler; ilk ihlali döner.
    ///
    /// # Hatalar
    ///
    /// Koruyucu baytlardan biri sentinel deseninden farklıysa
    /// [`SentinelViolation`] döner (ön koruyucu önce denetlenir).
    pub fn verify(&self) -> Result<(), SentinelViolation> {
        for (index, &found) in self.bytes.iter().enumerate().take(self.guard) {
            if found != self.sentinel {
                return Err(SentinelViolation {
                    side: GuardSide::Front,
                    index,
                    expected: self.sentinel,
                    found,
                });
            }
        }
        let back_start = self.guard + self.data_len;
        for (offset, &found) in self.bytes[back_start..].iter().enumerate() {
            if found != self.sentinel {
                return Err(SentinelViolation {
                    side: GuardSide::Back,
                    index: back_start + offset,
                    expected: self.sentinel,
                    found,
                });
            }
        }
        Ok(())
    }

    /// Hata veren tamponu diske kaydeder (SMD-07: regresyon korpusu).
    ///
    /// # Hatalar
    ///
    /// Girdi/çıktı hatası [`io::Error`] olarak döner (panik yok).
    pub fn dump(&self, dir: &Path, name: &str) -> io::Result<PathBuf> {
        regresyon_kaydet(dir, name, self.raw())
    }
}

/// Over-read tespiti: çekirdek, veri aralığının dışındaki padding'i okudu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverRead {
    /// İki koşunun çıktısında ilk farklı baytın indeksi.
    pub first_diff: usize,
    /// Birinci koşunun padding deseni.
    pub padding_a: u8,
    /// İkinci koşunun padding deseni.
    pub padding_b: u8,
}

/// Padding'li girdiyle çekirdeği iki kez koşturup over-read ölçen düzenek.
///
/// Girdi yerleşimi:
///
/// ```text
/// [ padding ][ veri: data_len bayt ][ padding ]
/// ```
///
/// Veri bölgesi **iki koşuda da aynıdır** ([`OverReadProbe::payload`]); değişen
/// tek şey padding baytlarıdır. Bu yüzden çıktı farkı, padding'in okunduğunun
/// kanıtıdır.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverReadProbe {
    data_len: usize,
    padding: usize,
    out_len: usize,
}

impl OverReadProbe {
    /// `data_len` baytlık veri, her iki yanda `padding` bayt dolgu,
    /// `out_len` baytlık çıktı ile düzenek kurar.
    #[must_use]
    pub const fn new(data_len: usize, padding: usize, out_len: usize) -> Self {
        Self {
            data_len,
            padding,
            out_len,
        }
    }

    /// Veri uzunluğu.
    #[must_use]
    pub const fn data_len(&self) -> usize {
        self.data_len
    }

    /// Tek taraftaki padding uzunluğu.
    #[must_use]
    pub const fn padding(&self) -> usize {
        self.padding
    }

    /// Çıktı uzunluğu.
    #[must_use]
    pub const fn out_len(&self) -> usize {
        self.out_len
    }

    /// Verinin padding'li girdi içindeki başlangıç indeksi.
    #[must_use]
    pub const fn data_offset(&self) -> usize {
        self.padding
    }

    /// Toplam girdi uzunluğu: `padding + data_len + padding`.
    #[must_use]
    pub const fn input_len(&self) -> usize {
        self.padding + self.data_len + self.padding
    }

    /// Veri bölgesinin deterministik içeriği (her koşuda aynı).
    ///
    /// Desen: `(i × 31 + 7) mod 256` — ardışık baytlar farklıdır, böylece
    /// yanlış ofsetten yapılan okumalar çıktıyı değiştirir.
    #[must_use]
    pub fn payload(&self) -> Vec<u8> {
        (0..self.data_len).map(payload_bayti).collect()
    }

    /// Padding'i `padding_byte` ile doldurulmuş girdi üretir.
    #[must_use]
    fn girdi(&self, padding_byte: u8) -> Vec<u8> {
        let mut input = vec![padding_byte; self.input_len()];
        for (i, slot) in input[self.data_offset()..self.data_offset() + self.data_len]
            .iter_mut()
            .enumerate()
        {
            *slot = payload_bayti(i);
        }
        input
    }

    /// Çekirdeği iki padding deseniyle koşturur ve **kendi ayırdığı** çıktıları
    /// karşılaştırır.
    ///
    /// Başarıda birinci koşunun çıktısını döner (iki koşu bit-aynıdır).
    ///
    /// # Hatalar
    ///
    /// Çıktılar farklıysa [`OverRead`] döner: çekirdek padding'den okumuştur.
    pub fn run<F>(&self, kernel: F) -> Result<Vec<u8>, OverRead>
    where
        F: Fn(&[u8], &mut [u8]),
    {
        let mut out_a = vec![0u8; self.out_len];
        let mut out_b = vec![0u8; self.out_len];
        self.run_pair(&mut out_a, &mut out_b, kernel)?;
        Ok(out_a)
    }

    /// Çekirdeği iki padding deseniyle koşturur; çıktı tamponlarını **çağıran**
    /// verir (sentinel'li çıktı için: `out.data_mut()`).
    ///
    /// İki çıktı tamponu eşit uzunlukta olmalıdır; aksi hâlde test yardımcısı
    /// sözleşmesi ihlal edilmiş olur ve panikler (bu, üretim kodu değildir).
    ///
    /// # Hatalar
    ///
    /// Çıktılar farklıysa [`OverRead`] döner.
    pub fn run_pair<F>(&self, out_a: &mut [u8], out_b: &mut [u8], kernel: F) -> Result<(), OverRead>
    where
        F: Fn(&[u8], &mut [u8]),
    {
        assert_eq!(
            out_a.len(),
            out_b.len(),
            "OverReadProbe::run_pair: iki çıktı tamponu eşit uzunlukta olmalı"
        );
        let input_a = self.girdi(PADDING_A);
        kernel(&input_a, out_a);
        let input_b = self.girdi(PADDING_B);
        kernel(&input_b, out_b);

        match out_a.iter().zip(out_b.iter()).position(|(a, b)| a != b) {
            None => Ok(()),
            Some(first_diff) => Err(OverRead {
                first_diff,
                padding_a: PADDING_A,
                padding_b: PADDING_B,
            }),
        }
    }
}

/// Girdi verisinin deterministik bayt deseni: `(i × 31 + 7) mod 256`.
fn payload_bayti(index: usize) -> u8 {
    (index.wrapping_mul(31).wrapping_add(7) & 0xFF) as u8
}

/// Varsayılan regresyon dizini: `<workspace>/target/sentinel-regressions`.
///
/// `target/` `.gitignore` içindedir; regresyon korpusu **üretim ağacına
/// girmez**, ancak hata ayıklama için diskte kalır (SMD-07).
#[must_use]
pub fn varsayilan_regresyon_dizini() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("target")
        .join("sentinel-regressions")
}

/// Hata veren girdiyi diske kaydeder (SMD-07) ve yazılan yolu döner.
///
/// Dizin yoksa oluşturulur.
///
/// # Hatalar
///
/// Girdi/çıktı hatası [`io::Error`] olarak döner (panik yok).
pub fn regresyon_kaydet(dir: &Path, name: &str, bytes: &[u8]) -> io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let path = dir.join(name);
    fs::write(&path, bytes)?;
    Ok(path)
}
