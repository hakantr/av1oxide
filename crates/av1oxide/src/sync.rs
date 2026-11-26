//! Platforma ve özellik kipine duyarlı senkronizasyon sarmalayıcısı
//! (`Mutex`, `MutexGuard`, `Condvar`, `OnceLock`).
//!
//! - `feature = "std"` + `not(target_arch = "wasm32")`: `parking_lot` (`0.12`,
//!   `default-features = false`) ve `std::sync::OnceLock`.
//! - `feature = "std"` + `target_arch = "wasm32"`: `std::sync` ilkelleri.
//! - `not(feature = "std")` (`#![no_std] + alloc`): `%100 #![forbid(unsafe_code)]`
//!   `crossbeam_utils::atomic::AtomicCell` + `Backoff` tabanlı `Send + Sync` kilit ve
//!   tek seferlik kurulum hücresi.

#![forbid(unsafe_code)]

#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
mod imp {
    pub(crate) use std::sync::OnceLock;

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

#[cfg(all(feature = "std", target_arch = "wasm32"))]
mod imp {
    pub(crate) use std::sync::OnceLock;

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

#[cfg(not(feature = "std"))]
mod imp {
    use alloc::boxed::Box;
    use core::fmt;
    use core::ops::{Deref, DerefMut};
    use core::sync::atomic::{AtomicBool, Ordering};
    use crossbeam_utils::Backoff;
    use crossbeam_utils::atomic::AtomicCell;

    /// `%100 #![forbid(unsafe_code)]`, `Send + Sync` `no_std` karşılıklı dışlama kilidi.
    pub(crate) struct Mutex<T> {
        cell: AtomicCell<Option<T>>,
    }

    impl<T: Default> Default for Mutex<T> {
        fn default() -> Self {
            Self::new(T::default())
        }
    }

    impl<T: fmt::Debug> fmt::Debug for Mutex<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Mutex").finish_non_exhaustive()
        }
    }

    /// `no_std` [`Mutex`] kilit bekçisi; düşürüldüğünde değeri atomik olarak geri bırakır.
    pub(crate) struct MutexGuard<'a, T> {
        mutex: &'a Mutex<T>,
        val: Option<T>,
    }

    impl<T> Deref for MutexGuard<'_, T> {
        type Target = T;

        #[inline]
        fn deref(&self) -> &Self::Target {
            self.val.as_ref().expect("MutexGuard invariant")
        }
    }

    impl<T> DerefMut for MutexGuard<'_, T> {
        #[inline]
        fn deref_mut(&mut self) -> &mut Self::Target {
            self.val.as_mut().expect("MutexGuard invariant")
        }
    }

    impl<T> Drop for MutexGuard<'_, T> {
        #[inline]
        fn drop(&mut self) {
            if let Some(val) = self.val.take() {
                self.mutex.cell.store(Some(val));
            }
        }
    }

    impl<T> Mutex<T> {
        /// Yeni bir `Mutex` oluşturur.
        #[inline]
        pub(crate) const fn new(val: T) -> Self {
            Self {
                cell: AtomicCell::new(Some(val)),
            }
        }

        /// Kilidi alır ve RAII bekçisi döndürür.
        #[inline]
        pub(crate) fn lock(&self) -> MutexGuard<'_, T> {
            let backoff = Backoff::new();
            loop {
                if let Some(val) = self.cell.take() {
                    return MutexGuard {
                        mutex: self,
                        val: Some(val),
                    };
                }
                backoff.snooze();
            }
        }
    }

    /// Tek iş parçacıklı `no_std` koşul değişkeni geri-düşüşü.
    #[derive(Debug, Default)]
    pub(crate) struct Condvar;

    impl Condvar {
        /// Yeni bir `Condvar` oluşturur.
        #[inline]
        pub(crate) const fn new() -> Self {
            Self
        }

        /// Tek iş parçacıklı kipte uyuma olmaksızın bekçiyi geri döndürür.
        #[inline]
        pub(crate) fn wait<'a, T>(&self, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
            guard
        }

        /// Tek iş parçacıklı kipte boş işlemdir.
        #[inline]
        #[allow(dead_code)]
        pub(crate) fn notify_one(&self) {}

        /// Tek iş parçacıklı kipte boş işlemdir.
        #[inline]
        pub(crate) fn notify_all(&self) {}
    }

    /// `%100 #![forbid(unsafe_code)]`, kilitsiz hızlı yollu `no_std` tek seferlik hücre.
    pub(crate) struct OnceLock<T: 'static> {
        cell: AtomicCell<Option<&'static T>>,
        init_lock: AtomicBool,
    }

    impl<T: 'static> OnceLock<T> {
        /// Boş `OnceLock` oluşturur.
        #[inline]
        pub(crate) const fn new() -> Self {
            Self {
                cell: AtomicCell::new(None),
                init_lock: AtomicBool::new(false),
            }
        }

        /// Hücre boşsa `f` ile tek seferlik başlatır ve `'static` referansını döndürür.
        pub(crate) fn get_or_init<F: FnOnce() -> T>(&self, f: F) -> &'static T {
            if let Some(val) = self.cell.load() {
                return val;
            }
            let backoff = Backoff::new();
            while self
                .init_lock
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                if let Some(val) = self.cell.load() {
                    return val;
                }
                backoff.snooze();
            }
            struct Unlock<'a>(&'a AtomicBool);
            impl Drop for Unlock<'_> {
                fn drop(&mut self) {
                    self.0.store(false, Ordering::Release);
                }
            }
            let _unlock = Unlock(&self.init_lock);
            if let Some(val) = self.cell.load() {
                return val;
            }
            let leaked: &'static T = Box::leak(Box::new(f()));
            self.cell.store(Some(leaked));
            leaked
        }
    }
}

#[allow(unused_imports)]
pub(crate) use imp::{Condvar, Mutex, MutexGuard, OnceLock};
