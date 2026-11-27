//! Çözücü yapılandırması (BLK-01, BLK-07, DSP-01/5).
//!
//! `Config` çözücünün **sözleşme yüzeyidir**: kaç iş parçacığı kullanılacağı,
//! toplam bellek bütçesi ve hangi CPU yollarının **kapatılacağı** burada
//! bildirilir. Doğrulama [`Config::validate`] ile `Decoder::new` sırasında
//! yapılır; geçersiz değer panik değil
//! [`Error::InvalidArgument`](crate::error::Error::InvalidArgument) üretir.

#![forbid(unsafe_code)]

use crate::cpu::CpuFlags;
use crate::error::{Error, Result};

/// Yapılandırmada kabul edilen en büyük iş parçacığı sayısı.
///
/// BLK-07: thread sayısı **sınırlanabilir** olmalıdır. F1.5'te bu üst sınır
/// yer tutucudur; F7'de ölçümle (görev grafiği + bellek) kesinleştirilir.
pub const MAX_THREADS: usize = 256;

/// Yapılandırmada kabul edilen en büyük kare gecikmesi (`dav1d` `DAV1D_MAX_FRAME_DELAY`).
pub const MAX_FRAME_DELAY: usize = 256;

/// Çözücü yapılandırması.
///
/// ```
/// # use av1oxide::api::Config;
/// let config = Config {
///     n_threads: 4,
///     max_frame_delay: 0,
///     allocation_limit: Some(256 * 1024 * 1024),
///     cpu_mask: av1oxide::cpu::CpuFlags::AVX512ICL,
///     operating_point: 0,
///     frame_size_limit: 0,
///     apply_grain: true,
///     output_invisible_frames: false,
///     strict_std_compliance: false,
/// };
/// assert_eq!(config.n_threads, 4);
/// assert_eq!(config.effective_frame_delay(), 2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// Kullanılacak iş parçacığı sayısı.
    ///
    /// - `0` → **otomatik** (`std::thread::available_parallelism()` ile tespit
    ///   edilen çekirdek sayısı, `1..=MAX_THREADS` aralığına kıstırılmış).
    /// - `1` (varsayılan) → tek iş parçacığı (`0` yardımcı iş parçacığı açılır).
    /// - `2..=MAX_THREADS` → kare-içi tile/post-filter ve kareler arası paralellik
    ///   için etkin iş parçacığı sayısı (`n - 1` yardımcı + çağıran iş parçacığı).
    pub n_threads: usize,
    /// Eşzamanlı kare bağlamı (`n_fc`, frame-level threading) üst sınırı
    /// (dav1d `Dav1dSettings.max_frame_delay`, `include/dav1d/dav1d.h:76`).
    ///
    /// - `0` (varsayılan) → **otomatik**: `effective_threads()` değerinden
    ///   dav1d `fc_lut` (`ceil(sqrt(n_tc))`, `1..=8`) ile türetilir
    ///   (`n_threads == 1` iken `1`).
    /// - `1` → düşük gecikme kipi: kareler arası boru hattı kapalıdır (`0`
    ///   kare gecikmesi), yalnız kare-içi tile ve post-filter paralelliği çalışır.
    /// - `2..=MAX_FRAME_DELAY` → en çok `min(max_frame_delay, effective_threads())`
    ///   kare eşzamanlı uçuşta (`in_flight`) işlenir.
    pub max_frame_delay: usize,
    /// Kullanıcı tanımlı **toplam** bellek bütçesi (bayt); `None` = sınırsız.
    ///
    /// BLK-07: Bu, AV1'in **normatif** boyut sınırlarından **ayrı** bir
    /// uygulama sınırıdır. Aşımda `Result` ile hata döner, panik olmaz.
    pub allocation_limit: Option<usize>,
    /// Kapatılacak CPU yollarının maskesi (DSP-01/5).
    ///
    /// Tespit edilen bayraklardan bu küme **çıkarılır**
    /// (`detected.difference(cpu_mask)`), böylece benchmark ve doğrulama
    /// koşularında ISA yolları zorla devre dışı bırakılabilir. Varsayılan
    /// [`CpuFlags::NONE`] = hiçbir şey kapatılmaz (dav1d `--cpumask` verilmemiş,
    /// `-1`).
    ///
    /// dav1d `--cpumask` düzeylerinin hazır değerleri
    /// [`CpuLevel::cpu_mask`](crate::cpu::CpuLevel::cpu_mask)'tir: dav1d maskesi
    /// **izin** maskesidir (`flags & mask`), bu alan **kapatma** maskesidir;
    /// dönüşüm `allowed.complement()` ↔ `cpu_mask.complement()`'tir
    /// ([`CpuFlags::complement`]). Örnek: `CpuLevel::None.cpu_mask()` bütün
    /// ISA yollarını kapatır (zorlanmış skaler), `CpuLevel::Avx2.cpu_mask()`
    /// AVX2 üstünü ve `SLOW_GATHER`'ı kapatır (dav1d `--cpumask avx2`).
    pub cpu_mask: CpuFlags,
    /// Sequence header'daki operating point seçimi (0–31; dav1d
    /// `Dav1dSettings.operating_point`, `include/dav1d/dav1d.h` 82).
    ///
    /// Sequence header'da bu dizin yoksa dav1d gibi **0** kullanılır
    /// (`src/obu.c` 1224–1226). Seçilen operating point'in `operating_point_idc`
    /// değeri, sonraki OBU'ların temporal/spatial katman süzgecini belirler
    /// (`src/obu.c` 1201–1208).
    pub operating_point: u8,
    /// Kare alanı üst sınırı (piksel); `0` = **sınırsız** (dav1d
    /// `Dav1dSettings.frame_size_limit`, `include/dav1d/dav1d.h`; varsayılan
    /// `src/lib.c:82`).
    ///
    /// dav1d semantiği (`src/obu.c:1300-1304`): sıfırdan farklıysa
    /// `width[1] * height` bu değeri aşan kare başlığı **reddedilir**
    /// (`DAV1D_ERR(ERANGE)`), kare çözülmez. Varsayılan `0` üretim
    /// davranışını değiştirmez; yalnız bozuk-girdi koşularında (dav1d
    /// fuzzer sınırı `4096 * 4096`) kullanılır.
    pub frame_size_limit: u64,
    /// Film grain çıktıya uygulansın mı (dav1d `Dav1dSettings.apply_grain`,
    /// varsayılan `1`, `src/lib.c:74`).
    ///
    /// Açıkken (varsayılan) grain parametreli her çıktı karesinin
    /// **kopyasına** grain uygulanır (`src/lib.c:311-330`); referans kareler
    /// her durumda grain'siz kalır. Kapalıyken çıktı grain'siz karedir
    /// (dav1d `--filmgrain 0`).
    pub apply_grain: bool,
    /// Görünmez kareler (`show_frame = 0`) de çıktı verilsin mi (dav1d
    /// `Dav1dSettings.output_invisible_frames`, varsayılan `0`, `src/lib.c:84`).
    ///
    /// Kapalıyken (varsayılan) yalnız gösterilen kareler (`show_frame = 1`) ve
    /// `show_existing_frame` çıktıları [`crate::Decoder::receive_frame`]'den
    /// döner (`src/decode.c:3551`); açıkken görünmez kareler de çözüm sırasıyla
    /// döner (kare differential'ı).
    pub output_invisible_frames: bool,
    /// Katı uyum denetimleri (dav1d `Dav1dSettings.strict_std_compliance`,
    /// kütüphane varsayılanı `0`, `src/lib.c:83`; CLI `1`).
    ///
    /// Açıkken gösterilemeyen kareyi gösteren `show_existing_frame` hata verir
    /// (`src/obu.c:1545-1548`).
    pub strict_std_compliance: bool,
}

impl Default for Config {
    /// Varsayılanlar: tek iş parçacığı, sınırsız bellek, maske yok, OP 0,
    /// film grain uygulanır; dav1d `dav1d_default_settings` gibi görünmez kare
    /// çıktısı ve katı uyum kapalı.
    fn default() -> Self {
        Self {
            n_threads: 1,
            max_frame_delay: 0,
            allocation_limit: None,
            cpu_mask: CpuFlags::NONE,
            operating_point: 0,
            frame_size_limit: 0,
            apply_grain: true,
            output_invisible_frames: false,
            strict_std_compliance: false,
        }
    }
}

impl Config {
    /// Değerleri sözleşmeye göre doğrular.
    ///
    /// # Hatalar
    ///
    /// - [`Error::InvalidArgument`]: `n_threads > `[`MAX_THREADS`],
    ///   `max_frame_delay > `[`MAX_FRAME_DELAY`],
    ///   `allocation_limit == Some(0)` (sıfır baytlık bütçe hiçbir kareyi
    ///   çözemez; sessiz kabul etmek yanıltıcı olurdu) veya
    ///   `operating_point > 31` (dav1d `Dav1dSettings.operating_point` aralığı).
    pub fn validate(&self) -> Result<()> {
        if self.n_threads > MAX_THREADS {
            return Err(Error::InvalidArgument(
                "n_threads üst sınırı aştı (bkz. MAX_THREADS)",
            ));
        }
        if self.max_frame_delay > MAX_FRAME_DELAY {
            return Err(Error::InvalidArgument(
                "max_frame_delay üst sınırı aştı (bkz. MAX_FRAME_DELAY)",
            ));
        }
        if self.allocation_limit == Some(0) {
            return Err(Error::InvalidArgument(
                "allocation_limit 0 olamaz (sıfır bütçe hiçbir kareyi çözemez)",
            ));
        }
        if self.operating_point > 31 {
            return Err(Error::InvalidArgument(
                "operating_point 0..=31 aralığında olmalı (dav1d Dav1dSettings)",
            ));
        }
        Ok(())
    }

    /// Etkin iş parçacığı sayısını (`1..=MAX_THREADS`) döndürür (dav1d `src/lib.c:120-165`).
    ///
    /// - `n_threads == 0`: `std::thread::available_parallelism()` ile tespit
    ///   edilen çekirdek sayısı (`1..=MAX_THREADS` aralığına kıstırılır).
    /// - `n_threads >= 1`: `n_threads.clamp(1, MAX_THREADS)`.
    /// - `wasm32` (`DSP-07`): varsayılan WASM ortamında işletim sistemi iş
    ///   parçacığı (`std::thread::spawn`) desteklenmediğinden her zaman `1`
    ///   döndürür.
    #[must_use]
    pub fn effective_threads(&self) -> usize {
        #[cfg(any(not(feature = "std"), target_arch = "wasm32"))]
        {
            1
        }
        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        {
            if self.n_threads == 0 {
                std::thread::available_parallelism()
                    .map(core::num::NonZeroUsize::get)
                    .unwrap_or(1)
                    .clamp(1, MAX_THREADS)
            } else {
                self.n_threads.clamp(1, MAX_THREADS)
            }
        }
    }

    /// Etkin eşzamanlı kare bağlamı sayısını (`n_fc`, `1..=MAX_THREADS`) döndürür
    /// (dav1d `src/lib.c:110-126` `fc_lut` + `get_num_threads` ile birebir aynı).
    ///
    /// - `max_frame_delay == 0`: `n_tc = effective_threads()` için
    ///   `n_tc < 50 ? FC_LUT[n_tc - 1] : 8` (`ceil(sqrt(n_tc))`, `1..=8`).
    /// - `max_frame_delay >= 1`: `max_frame_delay.min(effective_threads()).max(1)`.
    #[must_use]
    pub fn effective_frame_delay(&self) -> usize {
        const FC_LUT: [usize; 49] = [
            1, // 1
            2, 2, 2, // 2..=4
            3, 3, 3, 3, 3, // 5..=9
            4, 4, 4, 4, 4, 4, 4, // 10..=16
            5, 5, 5, 5, 5, 5, 5, 5, 5, // 17..=25
            6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, // 26..=36
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, // 37..=49
        ];
        let n_tc = self.effective_threads();
        if self.max_frame_delay == 0 {
            if n_tc < 50 { FC_LUT[n_tc - 1] } else { 8 }
        } else {
            self.max_frame_delay.min(n_tc).max(1)
        }
    }
}

/// Halka içi filtre bit maskesi (`dav1d` `enum Dav1dInloopFilterType` uyumlu).
///
/// Varsayılan değer [`InloopFilters::ALL`] olup deblocking, CDEF ve Loop Restoration
/// aşamalarının tamamını etkinleştirir. `SuperRes` ölçekleme aşaması bu bayraktan
/// bağımsız olarak her zaman uygulanır.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InloopFilters(u8);

impl InloopFilters {
    /// Hiçbir halka içi filtre uygulanmaz (`DAV1D_INLOOPFILTER_NONE`).
    pub const NONE: Self = Self(0);
    /// Deblocking (halka süzgeci) etkin (`DAV1D_INLOOPFILTER_DEBLOCK`).
    pub const DEBLOCK: Self = Self(1 << 0);
    /// CDEF yönlü kenar süzgeci etkin (`DAV1D_INLOOPFILTER_CDEF`).
    pub const CDEF: Self = Self(1 << 1);
    /// Loop Restoration (Wiener / SGR) etkin (`DAV1D_INLOOPFILTER_RESTORATION`).
    pub const RESTORATION: Self = Self(1 << 2);
    /// Tüm halka içi filtreler etkin (`DAV1D_INLOOPFILTER_ALL`).
    pub const ALL: Self = Self((1 << 0) | (1 << 1) | (1 << 2));

    /// Ham bit maskesini döndürür.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Yalnızca geçerli bitleri koruyarak maskedeki değerden `InloopFilters` oluşturur.
    #[must_use]
    pub const fn from_bits_truncate(bits: u8) -> Self {
        Self(bits & Self::ALL.0)
    }

    /// Verilen bayrakların tamamı bu maskede etkin mi?
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl Default for InloopFilters {
    fn default() -> Self {
        Self::ALL
    }
}

impl core::ops::BitOr for InloopFilters {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitAnd for InloopFilters {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

/// Çözülecek kare türlerini belirleyen filtre (`dav1d` `enum Dav1dDecodeFrameType` uyumlu).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DecodeFrameType {
    /// Tüm kareler çözülür (`DAV1D_DECODEFRAMETYPE_ALL`).
    #[default]
    All,
    /// Yalnızca referans olarak saklanan kareler çözülür (`DAV1D_DECODEFRAMETYPE_REFERENCE`).
    Reference,
    /// Yalnızca intra (Key + Intra-only) kareler çözülür (`DAV1D_DECODEFRAMETYPE_INTRA`).
    Intra,
    /// Yalnızca anahtar (`Key`) kareler çözülür (`DAV1D_DECODEFRAMETYPE_KEY`).
    Key,
}

#[cfg(test)]
mod tests {
    use super::{Config, MAX_FRAME_DELAY, MAX_THREADS};
    use crate::cpu::CpuFlags;
    use crate::error::Error;

    #[test]
    fn varsayilanlar_belgelenen_degerlerdir() {
        let config = Config::default();
        assert_eq!(config.n_threads, 1);
        assert_eq!(config.max_frame_delay, 0);
        assert_eq!(config.effective_frame_delay(), 1);
        assert_eq!(config.allocation_limit, None);
        assert!(config.cpu_mask.is_empty());
        assert_eq!(config.operating_point, 0);
        assert_eq!(config.frame_size_limit, 0);
        assert!(config.apply_grain);
        assert!(!config.output_invisible_frames);
        assert!(!config.strict_std_compliance);
        assert_eq!(config.validate(), Ok(()));
    }

    #[test]
    fn gecersiz_degerler_reddedilir() {
        let too_many = Config {
            n_threads: MAX_THREADS + 1,
            ..Config::default()
        };
        assert!(matches!(
            too_many.validate(),
            Err(Error::InvalidArgument(_))
        ));

        let zero_budget = Config {
            allocation_limit: Some(0),
            ..Config::default()
        };
        assert!(matches!(
            zero_budget.validate(),
            Err(Error::InvalidArgument(_))
        ));

        let bad_op = Config {
            operating_point: 32,
            ..Config::default()
        };
        assert!(matches!(bad_op.validate(), Err(Error::InvalidArgument(_))));

        let at_limit = Config {
            n_threads: MAX_THREADS,
            max_frame_delay: MAX_FRAME_DELAY,
            ..Config::default()
        };
        assert_eq!(at_limit.validate(), Ok(()));

        let too_much_delay = Config {
            max_frame_delay: MAX_FRAME_DELAY + 1,
            ..Config::default()
        };
        assert!(matches!(
            too_much_delay.validate(),
            Err(Error::InvalidArgument(_))
        ));
    }

    #[test]
    fn cpu_maskesi_kapatma_maskesidir() {
        let config = Config {
            cpu_mask: CpuFlags::AVX512ICL | CpuFlags::SLOW_GATHER,
            ..Config::default()
        };
        let detected = CpuFlags::SSE2 | CpuFlags::AVX512ICL;
        assert_eq!(detected.difference(config.cpu_mask), CpuFlags::SSE2);
    }

    #[test]
    fn effective_threads_sifir_ve_acik_degerleri_cozumler() {
        assert_eq!(Config::default().effective_threads(), 1);
        let auto = Config {
            n_threads: 0,
            ..Config::default()
        };
        assert_eq!(auto.validate(), Ok(()));
        let eff = auto.effective_threads();
        assert!((1..=MAX_THREADS).contains(&eff));
        for n in [1usize, 2, 4, 8, 16, MAX_THREADS] {
            let cfg = Config {
                n_threads: n,
                ..Config::default()
            };
            #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
            assert_eq!(cfg.effective_threads(), n);
            #[cfg(any(not(feature = "std"), target_arch = "wasm32"))]
            assert_eq!(cfg.effective_threads(), 1);
        }
    }

    #[test]
    fn effective_frame_delay_dav1d_fc_lut_ile_birebir_ayni() {
        let expected = [
            (1usize, 1usize),
            (2, 2),
            (4, 2),
            (5, 3),
            (8, 3),
            (9, 3),
            (10, 4),
            (16, 4),
            (17, 5),
            (25, 5),
            (26, 6),
            (36, 6),
            (37, 7),
            (49, 7),
            (50, 8),
            (256, 8),
        ];
        for (n_threads, exp_fc) in expected {
            let cfg = Config {
                n_threads,
                max_frame_delay: 0,
                ..Config::default()
            };
            #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
            assert_eq!(cfg.effective_frame_delay(), exp_fc, "n_threads={n_threads}");
            #[cfg(any(not(feature = "std"), target_arch = "wasm32"))]
            {
                let _ = exp_fc;
                assert_eq!(cfg.effective_frame_delay(), 1, "n_threads={n_threads}");
            }
        }
        // Açık `max_frame_delay` değeri `effective_threads()` ile üstten sınırlanır.
        let clamped = Config {
            n_threads: 4,
            max_frame_delay: 8,
            ..Config::default()
        };
        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        assert_eq!(clamped.effective_frame_delay(), 4);
        #[cfg(any(not(feature = "std"), target_arch = "wasm32"))]
        assert_eq!(clamped.effective_frame_delay(), 1);
        let single_fc = Config {
            n_threads: 8,
            max_frame_delay: 1,
            ..Config::default()
        };
        assert_eq!(single_fc.effective_frame_delay(), 1);
    }
}
