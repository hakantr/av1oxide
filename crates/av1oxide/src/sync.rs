//! Platforma duyarlı senkronizasyon sarmalayıcısı (`Mutex`, `MutexGuard`, `Condvar`).
//!
//! - `not(target_arch = "wasm32")` hedeflerinde `parking_lot` (`0.12`,
//!   `default-features = false`) kullanarak zehirlenme (`PoisonError`) yükünü
//!   ve küçük kilitlerde işletim sistemi `futex` sistem çağrısı maliyetini
//!   ortadan kaldırır.
//! - `target_arch = "wasm32"` hedeflerinde `std::sync` üzerine şeffaf biçimde
//!   geri düşer.

#![forbid(unsafe_code)]

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    /// Karşılıklı dışlama kilidi (`parking_lot::Mutex` sarmalayıcısı).
    #[derive(Debug, Default)]
    pub(crate) struct Mutex<T>(parking_lot::Mutex<T>);

    /// Kilit bekçisi (`parking_lot::MutexGuard`).
    pub(crate) type MutexGuard<'a, T> = parking_lot::MutexGuard<'a, T>;

    impl<T> Mutex<T> {
        /// Yeni bir `Mutex` oluşturur.
        #[inline]
        pub(crate) const fn new(val: T) -> Self {
            Self(parking_lot::Mutex::new(val))
        }

        /// Kilidi alır ve zehirlenme sarmalaması olmadan bekçi döndürür.
        #[inline]
        pub(crate) fn lock(&self) -> MutexGuard<'_, T> {
            self.0.lock()
        }
    }

    /// Koşul değişkeni (`parking_lot::Condvar` sarmalayıcısı).
    #[derive(Debug, Default)]
    pub(crate) struct Condvar(parking_lot::Condvar);

    impl Condvar {
        /// Yeni bir `Condvar` oluşturur.
        #[inline]
        pub(crate) const fn new() -> Self {
            Self(parking_lot::Condvar::new())
        }

        /// Bildirim gelene kadar bekçiyi serbest bırakıp uyur ve yeniden kilitler.
        #[inline]
        pub(crate) fn wait<'a, T>(&self, mut guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
            self.0.wait(&mut guard);
            guard
        }

        /// Bekleyen tek bir iş parçacığını uyandırır.
        #[inline]
        #[allow(dead_code)]
        pub(crate) fn notify_one(&self) {
            self.0.notify_one();
        }

        /// Bekleyen tüm iş parçacıklarını uyandırır.
        #[inline]
        pub(crate) fn notify_all(&self) {
            self.0.notify_all();
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    /// Karşılıklı dışlama kilidi (`std::sync::Mutex` sarmalayıcısı).
    #[derive(Debug, Default)]
    pub(crate) struct Mutex<T>(std::sync::Mutex<T>);

    /// Kilit bekçisi (`std::sync::MutexGuard`).
    pub(crate) type MutexGuard<'a, T> = std::sync::MutexGuard<'a, T>;

    impl<T> Mutex<T> {
        /// Yeni bir `Mutex` oluşturur.
        #[inline]
        pub(crate) const fn new(val: T) -> Self {
            Self(std::sync::Mutex::new(val))
        }

        /// Kilidi alır ve zehirlenme durumunda iç veriyi kurtararak bekçi döndürür.
        #[inline]
        pub(crate) fn lock(&self) -> MutexGuard<'_, T> {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        }
    }

    /// Koşul değişkeni (`std::sync::Condvar` sarmalayıcısı).
    #[derive(Debug, Default)]
    pub(crate) struct Condvar(std::sync::Condvar);

    impl Condvar {
        /// Yeni bir `Condvar` oluşturur.
        #[inline]
        pub(crate) const fn new() -> Self {
            Self(std::sync::Condvar::new())
        }

        /// Bildirim gelene kadar bekçiyi serbest bırakıp uyur ve yeniden kilitler.
        #[inline]
        pub(crate) fn wait<'a, T>(&self, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
            self.0
                .wait(guard)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        }

        /// Bekleyen tek bir iş parçacığını uyandırır.
        #[inline]
        #[allow(dead_code)]
        pub(crate) fn notify_one(&self) {
            self.0.notify_one();
        }

        /// Bekleyen tüm iş parçacıklarını uyandırır.
        #[inline]
        pub(crate) fn notify_all(&self) {
            self.0.notify_all();
        }
    }
}

#[allow(unused_imports)]
pub(crate) use imp::{Condvar, Mutex, MutexGuard};
