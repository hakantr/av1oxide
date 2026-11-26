//! Tahsis bütçesi defteri (BLK-07, P38 / GUV-09).
//!
//! [`Config::allocation_limit`](crate::api::Config::allocation_limit) çözücünün
//! **canlı büyük ayırmalarının** toplamına uygulanır. Defter iki parçadan
//! oluşur:
//!
//! - **taban:** kare dışında yaşayan ayırmalar — referans yuvaları (benzersiz
//!   kareler), çıktı kuyruğu, bekleyen tile verisi ve tanılama kuyrukları.
//!   Çözücü bunları her denetim noktasında gerçek uzunluklardan yeniden sayar
//!   (`Decoder::kalici_bayt`).
//! - **kare içi:** çözme sırasında ayrılan kare/tile/blok tamponları ve filtre
//!   tamponları. Kare boyunca yaşayanlar [`Butce::vec`] ile eklenir; bir işlev
//!   içinde açılıp kapanan geçici tamponlar [`Butce::gecici`] ile alınır ve
//!   düşürülünce defterden çıkar.
//!
//! Her ayırmadan **önce** `canlı + istenen ≤ sınır` denetlenir; aşımda
//! [`Error::BudgetExceeded`] döner ve ayırma yapılmaz. Bütçe içindeki ayırma
//! da düşebilirdir (`try_reserve_exact`): ayırıcı başarısız olursa
//! [`Error::AllocationFailed`] döner, panik/abort yoktur. Doldurma değeri
//! `vec![değer; n]` ile aynıdır (sıfır doldurma anlamı korunur).
//!
//! Sayılmayan küçük yapılar (başlıklar, CDF kopyaları yığında, tanılama
//! izleri, DSP içi sabit ara tamponlar) `docs/baseline/kaynak-siniri.md`'de
//! üst sınırlarıyla listelenir.

#![forbid(unsafe_code)]

use crate::error::{Error, Result, SizeStage};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicUsize, Ordering};
use crossbeam_utils::{Backoff, CachePadded};

/// Canlı ayırma defteri (atomik sayaçlarla iş parçacıkları arasında `&Butce` veya `Arc<Butce>` ile paylaşılır; BLK-08).
#[derive(Debug)]
pub(crate) struct Butce {
    sinir: Option<usize>,
    canli: CachePadded<AtomicUsize>,
    tepe: AtomicUsize,
    taban: AtomicUsize,
    ust: Option<Arc<Butce>>,
}

/// Paylaşılan kök `Arc<Butce>` üzerinde tutulan ve düşürüldüğünde (`Drop`)
/// `bayt` miktarını otomatik olarak serbest bırakan RAII bütçe bekçisi (`F7b`).
#[derive(Debug)]
pub(crate) struct ButceTutucu {
    ust: Arc<Butce>,
    bayt: usize,
}

impl Drop for ButceTutucu {
    fn drop(&mut self) {
        if self.bayt > 0 {
            self.ust.birak(self.bayt);
        }
    }
}

impl Drop for Butce {
    fn drop(&mut self) {
        if let Some(ref ust) = self.ust {
            // ORDERING: Alt defterin ömrü bittiğinde üzerinde kalan kare-içi
            // canlı baytları üst (paylaşılan boru hattı) defterinden düşer.
            let kalan = self.canli.swap(0, Ordering::Acquire);
            if kalan > 0 {
                ust.birak(kalan);
            }
        }
    }
}

impl Butce {
    /// `taban` bayt canlıyken yeni defter.
    pub(crate) fn yeni(sinir: Option<usize>, taban: usize) -> Self {
        Self {
            sinir,
            canli: CachePadded::new(AtomicUsize::new(taban)),
            tepe: AtomicUsize::new(0),
            taban: AtomicUsize::new(taban),
            ust: None,
        }
    }

    /// Paylaşılan `ust` boru hattı defterine bağlı kare-içi alt defter oluşturur (`F7b`).
    ///
    /// Alt defter üzerinden yapılan tüm `ayir`/`birak` çağrıları atomik olarak
    /// hem alt deftere hem `ust` deftere yansır; alt defter düşürüldüğünde
    /// (`Drop`) `devret` ile aktarılmamış tüm kare-içi geçici baytlar `ust`
    /// defterden otomatik olarak düşülür.
    pub(crate) fn alt_defter(ust: &Arc<Self>) -> Self {
        Self {
            sinir: ust.sinir,
            canli: CachePadded::new(AtomicUsize::new(0)),
            tepe: AtomicUsize::new(0),
            taban: AtomicUsize::new(0),
            ust: Some(Arc::clone(ust)),
        }
    }

    /// Kök defterin kalıcı taban baytını (`Decoder::kalici_bayt`) günceller (`F7b`).
    pub(crate) fn taban_guncelle(&self, yeni_taban: usize) {
        // ORDERING: Ana iş parçacığı tarafından kalıcı referans/çıktı kuyruğu
        // değiştiğinde çağrılır.
        let eski = self.taban.swap(yeni_taban, Ordering::AcqRel);
        if yeni_taban > eski {
            let fark = yeni_taban - eski;
            // ORDERING: Taban artışını canlı toplama atomik ekler.
            let yeni_canli = self
                .canli
                .fetch_add(fark, Ordering::AcqRel)
                .saturating_add(fark);
            self.tepe.fetch_max(yeni_canli, Ordering::Relaxed);
        } else if eski > yeni_taban {
            self.birak(eski - yeni_taban);
        }
    }

    /// Kök `ust` defterinde `bayt` ayırıp RAII [`ButceTutucu`] döndürür.
    pub(crate) fn tut(ust: &Arc<Self>, bayt: usize) -> Result<ButceTutucu> {
        if bayt > 0 {
            ust.ayir(bayt)?;
        }
        Ok(ButceTutucu {
            ust: Arc::clone(ust),
            bayt,
        })
    }

    /// Alt defterde zaten ayrılmış olan `bayt` miktarının sahipliğini (`ust`
    /// üzerinde yeniden ayırma/bırakma yapmadan) [`ButceTutucu`]'ya devreder.
    pub(crate) fn devret(&self, bayt: usize) -> Option<ButceTutucu> {
        let ust = self.ust.as_ref()?;
        let backoff = Backoff::new();
        let mut canli = self.canli.load(Ordering::Relaxed);
        let aktarilan = loop {
            let a = bayt.min(canli);
            let yeni = canli - a;
            match self
                .canli
                .compare_exchange_weak(canli, yeni, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => break a,
                Err(guncel) => {
                    canli = guncel;
                    backoff.spin();
                }
            }
        };
        Some(ButceTutucu {
            ust: Arc::clone(ust),
            bayt: aktarilan,
        })
    }

    /// Sınırsız, boş defter (birim testleri).
    #[cfg(test)]
    pub(crate) fn sinirsiz() -> Self {
        Self::yeni(None, 0)
    }

    /// Şu an sayılan canlı bayt (birim testleri).
    #[cfg(test)]
    pub(crate) fn canli(&self) -> usize {
        // ORDERING: Test gözlemi; önceki `ayir`/`birak` atomik yazımlarının
        // en son değerini aynı atomik değişken üzerinde okur.
        self.canli.load(Ordering::Acquire)
    }

    /// Başarılı denetimlerde görülen en büyük `canlı + istenen` bayt: bu
    /// defterle aynı ayırma dizisini geçiren en küçük sınır.
    pub(crate) fn tepe(&self) -> usize {
        if let Some(ref ust) = self.ust {
            return ust.tepe();
        }
        // ORDERING: Bariyer sonrası `Decoder::tepe_kat` okumasında tüm işçi
        // `fetch_max(..., Relaxed)` güncellemelerinin birleşik sonucunu okur.
        self.tepe.load(Ordering::Acquire)
    }

    /// `bayt` kadar yeni ayırmayı denetleyip deftere yazar.
    ///
    /// # Hatalar
    ///
    /// [`Error::BudgetExceeded`]: `canlı + bayt` sınırı aşıyor;
    /// [`Error::SizeOverflow`]: toplam `usize` aralığını aşıyor.
    pub(crate) fn ayir(&self, bayt: usize) -> Result<()> {
        if let Some(ref ust) = self.ust {
            ust.ayir(bayt)?;
        }
        // ORDERING: CAS döngüsünün başlangıç okuması; doğrulama CAS adımında
        // atomik olarak yapılır.
        let backoff = Backoff::new();
        let mut canli = self.canli.load(Ordering::Relaxed);
        loop {
            let Some(istenen) = canli.checked_add(bayt) else {
                if let Some(ref ust) = self.ust {
                    ust.birak(bayt);
                }
                return Err(Error::SizeOverflow {
                    stage: SizeStage::TotalAllocation,
                    lhs: canli,
                    rhs: bayt,
                });
            };
            if self.ust.is_none()
                && let Some(sinir) = self.sinir
                && istenen > sinir
            {
                return Err(Error::BudgetExceeded {
                    requested: istenen,
                    limit: sinir,
                });
            }
            // ORDERING: Başarılı `compare_exchange_weak` diğer iş parçacıklarının
            // önceki `birak` (`Release`) işlemlerini görür ve yeni canlı toplamı
            // yayınlar (`AcqRel`), böylece `canli <= sinir` değişmezi hiçbir
            // yarışta delinmez (ADR-0004 §4, BLK-08).
            match self.canli.compare_exchange_weak(
                canli,
                istenen,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // ORDERING: `tepe` monoton artan bir istatistiktir;
                    // `fetch_max` atomik olarak en büyük `istenen` değerini korur.
                    self.tepe.fetch_max(istenen, Ordering::Relaxed);
                    return Ok(());
                }
                Err(guncel) => {
                    canli = guncel;
                    backoff.spin();
                }
            }
        }
    }

    /// Daha önce [`Butce::ayir`] ile yazılmış `bayt`'ı defterden düşer.
    pub(crate) fn birak(&self, bayt: usize) {
        // ORDERING: Bırakılan bütçenin sonraki `ayir` CAS okumalarına
        // görünür olmasını sağlar (`Release`).
        let backoff = Backoff::new();
        let mut canli = self.canli.load(Ordering::Relaxed);
        let dusulen = loop {
            let d = bayt.min(canli);
            let yeni = canli - d;
            match self.canli.compare_exchange_weak(
                canli,
                yeni,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break d,
                Err(guncel) => {
                    canli = guncel;
                    backoff.spin();
                }
            }
        };
        if let Some(ref ust) = self.ust
            && dusulen > 0
        {
            ust.birak(dusulen);
        }
    }

    /// Defterde etkin bir bütçe sınırı (`Config::allocation_limit`) var mı?
    #[inline]
    pub(crate) const fn has_limit(&self) -> bool {
        self.sinir.is_some()
    }

    /// `n` öğelik, `deger` ile dolu vektör (`vec![deger; n]` eşdeğeri);
    /// defterde kalır (kare boyunca yaşayan tamponlar).
    #[allow(dead_code)]
    pub(crate) fn vec<T: Clone>(&self, n: usize, deger: T) -> Result<Vec<T>> {
        let bayt = dizi_bayti::<T>(n)?;
        self.ayir(bayt)?;
        dolu_vec(n, deger).inspect_err(|_| self.birak(bayt))
    }

    /// `old` vektörünün kapasitesi `[n, 2*n]` aralığındaysa (ve `allocation_limit`
    /// kapalıysa) belleği işletim sistemine geri vermeden `deger` ile sıfırlayıp
    /// yeniden kullanır; aksi hâlde `old` düşürülüp yeni `Vec<T>` ayrılır.
    pub(crate) fn reuse_vec<T: Clone>(
        &self,
        mut old: Vec<T>,
        n: usize,
        deger: T,
    ) -> Result<Vec<T>> {
        let bayt = dizi_bayti::<T>(n)?;
        self.ayir(bayt)?;
        if n == 0 {
            drop(old);
            return Ok(Vec::new());
        }
        if !self.has_limit() && old.capacity() >= n && old.capacity() <= n.saturating_mul(2) {
            if old.len() == n {
                old.fill(deger);
            } else {
                old.clear();
                old.resize(n, deger);
            }
            Ok(old)
        } else {
            drop(old);
            dolu_vec(n, deger).inspect_err(|_| self.birak(bayt))
        }
    }

    /// Dilimin düşebilir kopyası (`to_vec` eşdeğeri); defterde kalır.
    pub(crate) fn kopya<T: Clone>(&self, kaynak: &[T]) -> Result<Vec<T>> {
        let bayt = dizi_bayti::<T>(kaynak.len())?;
        self.ayir(bayt)?;
        kopya_vec(kaynak).inspect_err(|_| self.birak(bayt))
    }

    /// `old` vektörünün kapasitesi `[kaynak.len(), 2*kaynak.len()]` aralığındaysa
    /// belleği işletim sistemine geri vermeden `kaynak` dilimini kopyalar; aksi
    /// hâlde `old` düşürülüp yeni `Vec<T>` ayrılır.
    pub(crate) fn reuse_kopya<T: Clone>(&self, mut old: Vec<T>, kaynak: &[T]) -> Result<Vec<T>> {
        let n = kaynak.len();
        let bayt = dizi_bayti::<T>(n)?;
        self.ayir(bayt)?;
        if n == 0 {
            drop(old);
            return Ok(Vec::new());
        }
        if !self.has_limit() && old.capacity() >= n && old.capacity() <= n.saturating_mul(2) {
            old.clear();
            old.extend_from_slice(kaynak);
            Ok(old)
        } else {
            drop(old);
            kopya_vec(kaynak).inspect_err(|_| self.birak(bayt))
        }
    }

    /// `n` öğe kapasiteli boş vektör (`Vec::with_capacity` eşdeğeri);
    /// kapasite defterde kalır.
    pub(crate) fn kapasite<T>(&self, n: usize) -> Result<Vec<T>> {
        let bayt = dizi_bayti::<T>(n)?;
        self.ayir(bayt)?;
        let mut v = Vec::new();
        if v.try_reserve_exact(n).is_err() {
            self.birak(bayt);
            return Err(Error::AllocationFailed { requested: bayt });
        }
        Ok(v)
    }

    /// İşlev içi geçici tampon: düşürülünce defterden çıkar.
    pub(crate) fn gecici<T: Clone>(&self, n: usize, deger: T) -> Result<Gecici<'_, T>> {
        let bayt = dizi_bayti::<T>(n)?;
        self.ayir(bayt)?;
        match dolu_vec(n, deger) {
            Ok(v) => Ok(Gecici {
                v,
                bayt,
                butce: self,
            }),
            Err(err) => {
                self.birak(bayt);
                Err(err)
            }
        }
    }
}

/// [`Butce::gecici`] tamponu; `Vec<T>` gibi kullanılır.
pub(crate) struct Gecici<'b, T> {
    v: Vec<T>,
    bayt: usize,
    butce: &'b Butce,
}

impl<T> Deref for Gecici<'_, T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Vec<T> {
        &self.v
    }
}

impl<T> DerefMut for Gecici<'_, T> {
    fn deref_mut(&mut self) -> &mut Vec<T> {
        &mut self.v
    }
}

impl<T> Drop for Gecici<'_, T> {
    fn drop(&mut self) {
        self.butce.birak(self.bayt);
    }
}

/// `n` öğenin bayt sayısı (taşma denetimli).
pub(crate) fn dizi_bayti<T>(n: usize) -> Result<usize> {
    n.checked_mul(core::mem::size_of::<T>())
        .ok_or(Error::SizeOverflow {
            stage: SizeStage::TotalAllocation,
            lhs: n,
            rhs: core::mem::size_of::<T>(),
        })
}

/// Düşebilir `vec![deger; n]` (deftere yazmaz; bütçesi önceden denetlenen
/// ayırmalar için).
pub(crate) fn dolu_vec<T: Clone>(n: usize, deger: T) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| Error::AllocationFailed {
            requested: n.saturating_mul(core::mem::size_of::<T>()),
        })?;
    v.resize(n, deger);
    Ok(v)
}

/// Düşebilir `kaynak.to_vec()` (deftere yazmaz).
pub(crate) fn kopya_vec<T: Clone>(kaynak: &[T]) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(kaynak.len())
        .map_err(|_| Error::AllocationFailed {
            requested: core::mem::size_of_val(kaynak),
        })?;
    v.extend_from_slice(kaynak);
    Ok(v)
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    #[cfg(feature = "std")]
    use alloc::vec::Vec;

    use super::Butce;
    use crate::error::Error;

    #[test]
    fn sinir_canli_toplama_uygulanir_ve_tepe_basarili_denetimdir() {
        let b = Butce::yeni(Some(100), 40);
        let v = b.vec(50, 0u8).expect("40 + 50 ≤ 100");
        assert_eq!(v, vec![0u8; 50]);
        assert_eq!(b.canli(), 90);
        // 90 + 11 > 100: ayırma yapılmaz, defter değişmez.
        assert_eq!(
            b.vec(11, 7u8),
            Err(Error::BudgetExceeded {
                requested: 101,
                limit: 100
            })
        );
        assert_eq!(b.canli(), 90);
        assert_eq!(b.tepe(), 90);
        // `gecici` 20 bayt ister (u16): 110 > 100.
        assert!(matches!(
            b.gecici(10, 1u16),
            Err(Error::BudgetExceeded {
                requested: 110,
                limit: 100
            })
        ));
        assert_eq!(b.canli(), 90);
        let g = b.gecici(5, 1u16).expect("90 + 10 ≤ 100");
        assert_eq!(g.len(), 5);
        assert_eq!(b.canli(), 100);
        drop(g);
        assert_eq!(b.canli(), 90);
        assert_eq!(b.tepe(), 100);
    }

    #[test]
    fn tasan_boyut_ve_ayirici_hatasi_panik_degildir() {
        let b = Butce::sinirsiz();
        assert!(matches!(
            b.vec(usize::MAX, 0u64),
            Err(Error::SizeOverflow { .. })
        ));
        // `isize::MAX` baytı aşan istek ayırıcıya ulaşmadan hata verir; defter
        // geri alınır.
        assert!(matches!(
            b.vec(usize::MAX / 2, 0u16),
            Err(Error::AllocationFailed { .. })
        ));
        assert_eq!(b.canli(), 0);
        assert_eq!(b.kopya(&[1u8, 2, 3]).expect("küçük kopya"), vec![1, 2, 3]);
        assert_eq!(b.kapasite::<u32>(4).expect("kapasite").capacity(), 4);
    }

    #[cfg(feature = "std")]
    #[test]
    fn eszamanli_ayir_ve_birak_siniri_asmaz_ve_sizdirmaz() {
        use std::sync::{Arc, Barrier};
        let taban = 128usize;
        let sinir = 1024usize;
        let butce = Arc::new(Butce::yeni(Some(sinir), taban));
        let n_threads = 8usize;
        let barrier = Arc::new(Barrier::new(n_threads));
        let mut handles = Vec::with_capacity(n_threads);
        for tid in 0..n_threads {
            let b = Arc::clone(&butce);
            let bar = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                bar.wait();
                for step in 0..256usize {
                    let req = ((tid + step) % 16 + 1) * 16;
                    match b.gecici(req, 0u8) {
                        Ok(g) => {
                            assert_eq!(g.len(), req);
                            assert!(b.canli() <= sinir);
                        }
                        Err(Error::BudgetExceeded { limit, .. }) => {
                            assert_eq!(limit, sinir);
                        }
                        Err(other) => panic!("beklenmeyen hata: {other:?}"),
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("iş parçacığı paniklememeli");
        }
        assert_eq!(butce.canli(), taban);
        assert!(butce.tepe() <= sinir);
        assert!(butce.tepe() > taban);
    }
}
