//! Threading: kare-içi görev grafiği, `WorkerPool` ve `ScratchPool` (F7a / ADR-0004, BLK-08, BLK-09).
//!
//! ## Tasarım Özeti (`ADR-0004-threading-ve-kaynak-yonetimi.md`)
//!
//! - `%100 #![forbid(unsafe_code)]`, sıfır üçüncü taraf medya/kodek bağımlılığı
//!   (`std::thread`, `std::sync::Arc`, `crate::sync::{Mutex, Condvar}`,
//!   `core::sync::atomic`, `crossbeam_utils::CachePadded`).
//! - `effective_threads == 1` (`Config::default()`): **0** yardımcı iş parçacığı
//!   açılır; görevler çağıran iş parçacığında sırayla ve sıfır senkronizasyon
//!   yüküyle çalışır.
//! - `effective_threads >= 2`: `Decoder::new` sırasında `n - 1` kalıcı yardımcı
//!   iş parçacığı (`2 MiB` yığın boyutlu) bir kez başlatılır; çağıran iş
//!   parçacığı (`worker_id = 0`) her paralel aşamada (`execute_tasks`) kendi
//!   payına düşen görevleri bizzat yürüterek toplam `n` yürütücüyü tamamlar.
//! - Boşta bekleyen yardımcı iş parçacıkları `Condvar::wait` üzerinde uyur
//!   (spin-wait yoktur, `%0` CPU tüketimi).
//! - Deterministik hata ve erken iptal (`first_error_idx: AtomicUsize`):
//!   `k` indeksli görev hata ürettiğinde `first_error_idx.fetch_min(k, AcqRel)`
//!   güncellenir. `idx > first_error_idx` olan başlamamış görevler atlanır;
//!   `idx < k` olan görevler tamamlanır ve bariyer sonunda **en küçük indeksli**
//!   görevin hatası döndürülür (`n_threads = 1` ile %100 aynı `Err`).
//! - Panik güvenliği (`TaskDoneGuard`): bir görev beklenmedik biçimde
//!   unwind etse bile RAII bekçisi `remaining` sayacını düşürür ve bariyer
//!   `Condvar`'ını uyandırır; hiçbir durumda sonsuz bekleme (deadlock) oluşmaz.

#![forbid(unsafe_code)]

use crate::decode::intra::{DecodeBufPool, RefFrame};
use crate::error::{Error, Result};
use crate::frame::{BitDepth, ChromaFormat};
use crate::sync::{Condvar, Mutex};
use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use crossbeam_utils::CachePadded;
#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
use std::thread;
#[cfg(feature = "std")]
use std::thread::JoinHandle;

/// Yardımcı iş parçacığı yığın boyutu (`2 MiB`; derin özyinelemeli
/// `decode_sb` + 128×128 blok yığın çerçeveleri için güvenli sınır).
pub const WORKER_STACK_SIZE: usize = 2 * 1024 * 1024;

/// İşçi başına yeniden kullanılan geçici çalışma tamponu (`ScratchPool` yuvası).
#[derive(Clone, Debug, Default)]
pub struct WorkerScratch {
    /// Genel amaçlı geçici `u8` çalışma tamponu (ör. paralel CDEF şerit kopyası).
    pub bytes: Vec<u8>,
}

impl WorkerScratch {
    /// En az `len` baytlık sıfırlanmış veya yeniden boyutlandırılmış dilim verir.
    pub fn ensure_bytes(&mut self, len: usize) -> Result<&mut [u8]> {
        if self.bytes.len() < len {
            let add = len - self.bytes.len();
            self.bytes
                .try_reserve(add)
                .map_err(|_| Error::AllocationFailed { requested: len })?;
            self.bytes.resize(len, 0);
        }
        Ok(&mut self.bytes[..len])
    }
}

/// İşçi iş parçacığı başına bir adet `WorkerScratch` tutan havuz (BLK-07).
#[derive(Debug)]
pub struct ScratchPool {
    slots: Vec<CachePadded<Mutex<WorkerScratch>>>,
}

impl ScratchPool {
    /// `n_workers` işçi için boş `ScratchPool` oluşturur.
    #[must_use]
    pub fn new(n_workers: usize) -> Self {
        let count = n_workers.max(1);
        let mut slots = Vec::with_capacity(count);
        for _ in 0..count {
            slots.push(CachePadded::new(Mutex::new(WorkerScratch::default())));
        }
        Self { slots }
    }

    /// İşçi sayısı.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Havuz boş mu?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// `worker_id` yuvasındaki geçici tamponla `f` kapanışını çalıştırır.
    pub fn with_scratch<R, F>(&self, worker_id: usize, f: F) -> R
    where
        F: FnOnce(&mut WorkerScratch) -> R,
    {
        let idx = worker_id % self.slots.len().max(1);
        let mut guard = self.slots[idx].lock();
        f(&mut guard)
    }
}

type TaskRunnerFn = Box<dyn Fn(usize, usize, &TaskBatch) + Send + Sync>;
type TaskSlotList<T> = Arc<Vec<Mutex<Option<T>>>>;
type FirstErrorSlot = Arc<Mutex<Option<(usize, Error)>>>;
type AsyncTaskFn = Box<dyn FnOnce() + Send + 'static>;

#[inline]
fn record_first_error(
    slot: &Mutex<Option<(usize, Error)>>,
    batch_ref: &TaskBatch,
    idx: usize,
    err: Error,
) {
    // ORDERING: Hata veren en küçük görev indeksini atomik olarak kaydeder;
    // `AcqRel` ile diğer işçilerin erken iptal (`Acquire`) okumasını sıralar.
    // Yalnızca bu görev mevcut en küçük hata indeksini iyileştirdiğinde kilit
    // alınır; mutlu yolda (hata yokken) kilit hiç alınmaz.
    let prev = batch_ref.first_error_idx.fetch_min(idx, Ordering::AcqRel);
    if idx < prev {
        let mut guard = slot.lock();
        let replace = match &*guard {
            Some((cur_idx, _)) => idx < *cur_idx,
            None => true,
        };
        if replace {
            *guard = Some((idx, err));
        }
    }
}

/// `std::thread::scope` tabanlı sıfır-kopya paralel şerit aşamaları için
/// eşzamanlı ek iş parçacığı bütçesi tutan RAII bekçisi (`5.C.1`).
///
/// Birden çok kare `max_frame_delay > 1` altında eşzamanlı çözüldüğünde
/// toplam aktif ek iş parçacığı sayısını `n_threads - 1` sınırında tutarak
/// aşırı abonelik (`oversubscription`) ve bağlam değiştirme maliyetini önler.
#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
pub(crate) struct ScopeWorkersGuard<'a> {
    counter: Option<&'a AtomicUsize>,
    extra_workers: usize,
}

#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
impl ScopeWorkersGuard<'_> {
    /// Çağıran iş parçacığı dâhil toplam paralel yürütücü sayısı (`>= 1`).
    #[inline]
    pub(crate) const fn total_threads(&self) -> usize {
        1 + self.extra_workers
    }
}

#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
impl Drop for ScopeWorkersGuard<'_> {
    fn drop(&mut self) {
        if let Some(counter) = self.counter
            && self.extra_workers > 0
        {
            // ORDERING: Serbest bırakılan işçi bütçesini diğer karelere yayınlar.
            counter.fetch_sub(self.extra_workers, Ordering::Release);
        }
    }
}

/// Tek bir `execute_tasks` / `execute_indexed` çağrısına (kuşağına) özel,
/// diğer kuşaklardan tamamen yalıtılmış görev partisi durumu.
///
/// Kuşaklar arası atomik sayaç (`next_task`, `remaining`, `first_error_idx`)
/// paylaşımı yapılmaz; böylece önceki kuşaktan geç uyanan bir işçi hiçbir
/// koşulda yeni kuşağın sayaçlarını eksiltemez (`underflow` imkânsızdır).
struct TaskBatch {
    /// Bu partideki toplam görev sayısı.
    n_tasks: usize,
    /// Sıradaki görev indeksi (`0..n_tasks`).
    next_task: CachePadded<AtomicUsize>,
    /// Henüz tamamlanmamış görev sayısı.
    remaining: CachePadded<AtomicUsize>,
    /// Şimdiye kadar hata veren en küçük görev indeksi (`usize::MAX` = hata yok).
    first_error_idx: CachePadded<AtomicUsize>,
    /// En küçük indeksli hatayı (veya panik hatasını) tutan paylaşılan yuva.
    first_error: FirstErrorSlot,
    /// Görev çalıştırıcısı (`(worker_id, task_idx, &TaskBatch)`).
    runner: TaskRunnerFn,
    /// Parti tamamlanma koşul değişkeni kilidi.
    done_lock: Mutex<bool>,
    /// Parti tamamlandığında çağıran iş parçacığını uyandıran koşul değişkeni.
    done_cv: Condvar,
}

/// Görev tamamlandığında (veya panikle unwind ettiğinde) `batch.remaining`
/// sayacını düşüren ve son görevse çağıran iş parçacığını uyandıran RAII bekçisi.
struct TaskDoneGuard<'a> {
    batch: &'a TaskBatch,
    idx: usize,
    completed: bool,
}

impl Drop for TaskDoneGuard<'_> {
    fn drop(&mut self) {
        if !self.completed {
            record_first_error(
                &self.batch.first_error,
                self.batch,
                self.idx,
                Error::InvalidState("paralel işçi görevi panikledi"),
            );
        }
        // ORDERING: Görevin çıktı dilimine yaptığı tüm yazımları (`Release`)
        // bariyer sonrasındaki ana iş parçacığına yayınlar; son görev
        // `Acquire` ile önceki tüm görevlerin tamamlamalarını birleştirir
        // (`AcqRel`, ADR-0004 §4).
        if self.batch.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            let mut done = self.batch.done_lock.lock();
            *done = true;
            self.batch.done_cv.notify_all();
        }
    }
}

impl TaskBatch {
    /// Bu partideki görevleri atomik sayaçtan çekerek çalıştırır.
    fn drain_tasks(&self, worker_id: usize) {
        loop {
            // ORDERING: Bu partideki her görevi tam bir işçiye atomik olarak atar.
            let idx = self.next_task.fetch_add(1, Ordering::Relaxed);
            if idx >= self.n_tasks {
                break;
            }
            let mut done_guard = TaskDoneGuard {
                batch: self,
                idx,
                completed: false,
            };
            // ORDERING: Daha küçük indeksli bir görev zaten hata verdiyse,
            // deterministik ilk-hata sonucu (`first_error_idx`) değişmeyeceği
            // için bu görevi erken atla (ADR-0004 §4-5).
            if idx > self.first_error_idx.load(Ordering::Acquire) {
                done_guard.completed = true;
                continue;
            }
            #[cfg(feature = "std")]
            {
                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    (self.runner)(worker_id, idx, self);
                }));
                if res.is_ok() {
                    done_guard.completed = true;
                }
            }
            #[cfg(not(feature = "std"))]
            {
                (self.runner)(worker_id, idx, self);
                done_guard.completed = true;
            }
        }
    }
}

/// Yardımcı iş parçacıkları arasında paylaşılan kuyruk durumu.
struct SharedQueue {
    state: Mutex<QueueState>,
    work_cv: Condvar,
    /// Etkin `std::thread::scope` şerit aşamalarının tuttuğu ek iş parçacığı sayısı.
    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    scoped_active: CachePadded<AtomicUsize>,
}

struct QueueState {
    /// Her yeni iş eklendiğinde (`execute_tasks` veya `spawn_async`) artan kuşak sayacı.
    generation: u64,
    /// Havuz kapanıyor mu (`Drop`).
    #[cfg(feature = "std")]
    shutdown: bool,
    /// Etkin paralel alt-görev partileri (`execute_tasks` / `execute_indexed`).
    active_batches: Vec<Arc<TaskBatch>>,
    /// Kareler arası boru hattının (`FramePipeline`) FIFO asenkron kare görev kuyruğu.
    async_tasks: VecDeque<AsyncTaskFn>,
}

/// Kareler arası boru hattında (`F7b`, `max_frame_delay > 1`) bir referans
/// yuvasının durumunu temsil eden eşzamanlı vaat (`Promise`) yapısı (`ADR-0004`).
enum SlotState {
    /// Kare henüz uçuştadır (`in_flight`).
    Pending,
    /// Kare `Loop Restoration` adımını tamamlamış ve grain'siz `RefFrame`'i yayınlamıştır.
    Ready(Arc<RefFrame>),
    /// Kare geç hata üretmiş veya `reset()` / `drop()` ile iptal edilmiştir.
    Failed,
}

/// Referans yuvası vaadi (`pixel_refs` / `slot_promises` elemanı).
pub(crate) struct FrameSlot {
    /// Başlık zamanında bilinen superres-sonrası genişlik (`hdr.width[1]`).
    pub width: usize,
    /// Başlık zamanında bilinen yükseklik (`hdr.height`).
    pub height: usize,
    /// Başlık zamanında bilinen kroma düzeni.
    pub chroma: ChromaFormat,
    /// Başlık zamanında bilinen bit derinliği.
    pub bd: BitDepth,
    state: Mutex<SlotState>,
    cv: Condvar,
}

impl FrameSlot {
    /// Zaten tamamlanmış bir `Arc<RefFrame>` için hazır `FrameSlot` oluşturur.
    pub(crate) fn ready(rf: Arc<RefFrame>) -> Arc<Self> {
        let width = rf.pic.width;
        let height = rf.pic.height;
        let chroma = rf.chroma;
        let bd = rf.bd;
        Arc::new(Self {
            width,
            height,
            chroma,
            bd,
            state: Mutex::new(SlotState::Ready(rf)),
            cv: Condvar::new(),
        })
    }

    /// Uçuşa yeni gönderilen bir kare için `Pending` durumunda `FrameSlot` oluşturur.
    pub(crate) fn pending(
        width: usize,
        height: usize,
        chroma: ChromaFormat,
        bd: BitDepth,
    ) -> Arc<Self> {
        Arc::new(Self {
            width,
            height,
            chroma,
            bd,
            state: Mutex::new(SlotState::Pending),
            cv: Condvar::new(),
        })
    }

    /// `Loop Restoration` biter bitmez (gecikmeli Film Grain'den **önce**)
    /// grain'siz `Arc<RefFrame>`'i yayınlar ve bekleyen bağımlı kareleri uyandırır.
    pub(crate) fn publish_ready(&self, rf: Arc<RefFrame>) {
        let mut st = self.state.lock();
        if matches!(*st, SlotState::Pending) {
            *st = SlotState::Ready(rf);
            self.cv.notify_all();
        }
    }

    /// Geç hata veya iptal durumunda yuvayı `Failed` olarak işaretler ve
    /// bekleyen bağımlı kareleri kilitlenmeden uyandırır.
    pub(crate) fn mark_failed(&self) {
        let mut st = self.state.lock();
        if matches!(*st, SlotState::Pending) {
            *st = SlotState::Failed;
            self.cv.notify_all();
        }
    }

    /// Yuvadaki `Ready(Arc<RefFrame>)` referansını serbest bırakır (`F8b-1`:
    /// `slot_promises` içinde artık tutulmayan yuvaların `RefFrame` tamponlarının
    /// `Arc::try_unwrap` ile havuza geri dönüştürülebilmesi için).
    pub(crate) fn clear_ready(&self) {
        let mut st = self.state.lock();
        if matches!(*st, SlotState::Ready(_)) {
            *st = SlotState::Failed;
        }
    }

    /// Bloklamadan yuvanın `Failed` durumunda olup olmadığını döndürür.
    pub(crate) fn is_failed_nonblocking(&self) -> bool {
        let st = self.state.lock();
        matches!(*st, SlotState::Failed)
    }

    /// Bloklamadan yuvanın `Ready` veya `Failed` (yani `Pending` dışı) olup olmadığını döndürür.
    pub(crate) fn is_done_nonblocking(&self) -> bool {
        let st = self.state.lock();
        !matches!(*st, SlotState::Pending)
    }

    /// Yuvadaki `RefFrame` tamamlanana (veya hata/iptal olana) kadar `Condvar`
    /// üzerinde bekler (`spin-wait = 0`).
    pub(crate) fn wait(&self, cancelled: &AtomicBool) -> Option<Arc<RefFrame>> {
        let mut st = self.state.lock();
        loop {
            // ORDERING: İptal bayrağı (`Release`) gözlemi.
            if cancelled.load(Ordering::Acquire) {
                return None;
            }
            match &*st {
                SlotState::Ready(rf) => return Some(Arc::clone(rf)),
                SlotState::Failed => return None,
                SlotState::Pending => {
                    st = self.cv.wait(st);
                }
            }
        }
    }
}

/// Kare-içi paralel görev havuzu (`ADR-0004`).
///
/// - `n_threads == 1` iken işletim sistemi iş parçacığı açmaz (`workers` boştur).
/// - `n_threads >= 2` iken `n_threads - 1` yardımcı iş parçacığı açar; çağıran
///   iş parçacığı (`worker_id = 0`) görevlere katılır.
/// - `Clone`: aynı `n_threads` kapasiteli **taze** bir `WorkerPool` kurar
///   (klonlanan `Decoder`'ların iş parçacığı havuzları birbirinden bağımsızdır).
pub struct WorkerPool {
    n_threads: usize,
    scratch: Arc<ScratchPool>,
    buf_pool: Arc<DecodeBufPool>,
    shared: Option<Arc<SharedQueue>>,
    #[cfg(feature = "std")]
    workers: Vec<JoinHandle<()>>,
}

impl core::fmt::Debug for WorkerPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WorkerPool")
            .field("n_threads", &self.n_threads)
            .field("spawned_workers", &self.spawned_workers())
            .finish()
    }
}

impl Clone for WorkerPool {
    fn clone(&self) -> Self {
        Self::new(self.n_threads)
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        #[cfg(feature = "std")]
        {
            if !self.workers.is_empty()
                && let Some(shared) = &self.shared
            {
                {
                    let mut st = shared.state.lock();
                    st.shutdown = true;
                    st.active_batches.clear();
                    st.async_tasks.clear();
                }
                shared.work_cv.notify_all();
            }
            for handle in self.workers.drain(..) {
                let _ = handle.join();
            }
        }
    }
}

impl WorkerPool {
    /// `n_threads` (`Config::effective_threads()`) için iş parçacığı havuzu kurar.
    ///
    /// `n_threads <= 1` ise (veya `target_arch = "wasm32"` / `not(feature = "std")`
    /// altında) hiçbir işletim sistemi iş parçacığı başlatılmaz. İşletim sistemi yeni
    /// iş parçacığı açamazsa havuz güvenli biçimde daha az yardımcı iş parçacığına
    /// (en kötü durumda tek iş parçacığına) düşer; panik üretmez.
    #[must_use]
    pub fn new(n_threads: usize) -> Self {
        #[cfg(any(not(feature = "std"), target_arch = "wasm32"))]
        {
            let _ = n_threads;
            let scratch = Arc::new(ScratchPool::new(1));
            let buf_pool = Arc::new(DecodeBufPool::new(1));
            Self {
                n_threads: 1,
                scratch,
                buf_pool,
                shared: None,
                #[cfg(feature = "std")]
                workers: Vec::new(),
            }
        }
        #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
        {
            let n_threads = n_threads.max(1);
            let scratch = Arc::new(ScratchPool::new(n_threads));
            let buf_pool = Arc::new(DecodeBufPool::new(n_threads));
            if n_threads == 1 {
                return Self {
                    n_threads: 1,
                    scratch,
                    buf_pool,
                    shared: None,
                    workers: Vec::new(),
                };
            }
            let shared = Arc::new(SharedQueue {
                state: Mutex::new(QueueState {
                    generation: 0,
                    shutdown: false,
                    active_batches: Vec::new(),
                    async_tasks: VecDeque::new(),
                }),
                work_cv: Condvar::new(),
                scoped_active: CachePadded::new(AtomicUsize::new(0)),
            });
            let mut workers = Vec::with_capacity(n_threads - 1);
            for worker_id in 1..n_threads {
                let q = Arc::clone(&shared);
                let spawn_res = thread::Builder::new()
                    .name(alloc::format!("av1oxide-w{worker_id}"))
                    .stack_size(WORKER_STACK_SIZE)
                    .spawn(move || worker_loop(worker_id, &q));
                match spawn_res {
                    Ok(handle) => workers.push(handle),
                    Err(_) => break,
                }
            }
            if workers.is_empty() {
                return Self {
                    n_threads: 1,
                    scratch: Arc::new(ScratchPool::new(1)),
                    buf_pool,
                    shared: None,
                    workers: Vec::new(),
                };
            }
            let active_threads = workers.len() + 1;
            Self {
                n_threads: active_threads,
                scratch,
                buf_pool,
                shared: Some(shared),
                workers,
            }
        }
    }

    /// Havuzdaki etkin iş parçacığı sayısı (`1` = tek iş parçacığı).
    #[must_use]
    pub const fn n_threads(&self) -> usize {
        self.n_threads
    }

    /// Havuzda başlatılmış yardımcı iş parçacığı sayısı (`n_threads == 1` için `0`).
    #[must_use]
    pub fn spawned_workers(&self) -> usize {
        #[cfg(feature = "std")]
        {
            self.workers.len()
        }
        #[cfg(not(feature = "std"))]
        {
            0
        }
    }

    /// Paralel yürütme etkin mi (`n_threads > 1`)?
    #[must_use]
    pub const fn is_parallel(&self) -> bool {
        self.n_threads > 1
    }

    /// İşçi başına geçici tampon havuzuna erişim.
    #[must_use]
    pub fn scratch(&self) -> &ScratchPool {
        &self.scratch
    }

    /// Kareler arası çalışma ve referans tamponu yeniden kullanım havuzuna erişim (`F8b-1`).
    #[must_use]
    pub(crate) fn buf_pool(&self) -> &DecodeBufPool {
        &self.buf_pool
    }

    /// `Decoder::reset()` sırasında kareler arası tampon havuzunu tamamen boşaltır (`F8b-1`).
    pub(crate) fn clear_buf_pool(&self) {
        self.buf_pool.clear();
    }

    /// Aynı `SharedQueue` ve `ScratchPool` örneğini paylaşan, yeni işletim
    /// sistemi iş parçacığı açmayan ve düşürüldüğünde havuzu kapatmayan hafif
    /// tutamak döndürür (asenkron kare görevlerinin kare-içi `execute_tasks`
    /// çağrıları için).
    pub(crate) fn handle(&self) -> Self {
        Self {
            n_threads: self.n_threads,
            scratch: Arc::clone(&self.scratch),
            buf_pool: Arc::clone(&self.buf_pool),
            shared: self.shared.as_ref().map(Arc::clone),
            #[cfg(feature = "std")]
            workers: Vec::new(),
        }
    }

    /// Aynı `ScratchPool` örneğini paylaşan ancak görev dağıtımı yapmayan
    /// (`n_threads == 1`, `shared == None`) hafif tutamak döndürür.
    ///
    /// Kareler arası paralel boru hattında (`n_fc >= 2`), bir işçi iş parçacığı
    /// gecikmeli `apply_grain`'i kendi üzerinde yerinde (`in-place`) yürütürken
    /// diğer işçi iş parçacıklarının sonraki kareyi çekişmesiz çözebilmesi için
    /// kullanılır.
    pub(crate) fn single_thread_handle(&self) -> Self {
        Self {
            n_threads: 1,
            scratch: Arc::clone(&self.scratch),
            buf_pool: Arc::clone(&self.buf_pool),
            shared: None,
            #[cfg(feature = "std")]
            workers: Vec::new(),
        }
    }

    /// Kareler arası boru hattı (`FramePipeline`) için FIFO sırasıyla asenkron
    /// görev ekler.
    pub(crate) fn spawn_async<F>(&self, task: F)
    where
        F: FnOnce() + Send + 'static,
    {
        if let Some(shared) = &self.shared {
            {
                let mut st = shared.state.lock();
                st.generation = st.generation.wrapping_add(1);
                st.async_tasks.push_back(Box::new(task));
            }
            shared.work_cv.notify_all();
        } else {
            task();
        }
    }

    /// FIFO asenkron görev kuyruğunun başındaki görevi (henüz bir işçi tarafından
    /// alınmamışsa) çekip döndürür.
    pub(crate) fn try_pop_async_task(&self) -> Option<AsyncTaskFn> {
        let shared = self.shared.as_ref()?;
        let mut st = shared.state.lock();
        st.async_tasks.pop_front()
    }

    /// Etkin alt-görev partilerinden (`active_batches`) birine bir görevlik
    /// yardım eder; yardım edilecek parti varsa `true` döner.
    pub(crate) fn help_active_batch(&self) -> bool {
        let Some(shared) = self.shared.as_ref() else {
            return false;
        };
        let batch = {
            let st = shared.state.lock();
            st.active_batches
                .iter()
                .find(|b| b.next_task.load(Ordering::Relaxed) < b.n_tasks)
                .map(Arc::clone)
        };
        if let Some(b) = batch {
            b.drain_tasks(0);
            true
        } else {
            false
        }
    }

    /// `std::thread::scope` şerit aşaması için en çok `max_extra` ek iş parçacığı
    /// rezerve eder; toplam eşzamanlı ek işçi sayısını `n_threads - 1` sınırında tutar.
    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    pub(crate) fn acquire_scope_workers(&self, max_extra: usize) -> ScopeWorkersGuard<'_> {
        let budget = self.n_threads.saturating_sub(1);
        let cap = max_extra.min(budget);
        if cap == 0 {
            return ScopeWorkersGuard {
                counter: None,
                extra_workers: 0,
            };
        }
        let Some(shared) = &self.shared else {
            return ScopeWorkersGuard {
                counter: None,
                extra_workers: 0,
            };
        };
        let counter = &*shared.scoped_active;
        let mut cur = counter.load(Ordering::Relaxed);
        loop {
            let avail = budget.saturating_sub(cur);
            let take = avail.min(cap);
            if take == 0 {
                return ScopeWorkersGuard {
                    counter: None,
                    extra_workers: 0,
                };
            }
            match counter.compare_exchange_weak(
                cur,
                cur + take,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return ScopeWorkersGuard {
                        counter: Some(counter),
                        extra_workers: take,
                    };
                }
                Err(actual) => cur = actual,
            }
        }
    }

    /// `tasks` vektöründeki her `&mut T` öğesini kalıcı `WorkerPool` iş
    /// parçacıkları üzerinde paralel (veya `n_threads == 1` / `tasks.len() <= 1`
    /// iken doğrudan çağıran iş parçacığında) `op(worker_id, task_idx, &mut T)`
    /// ile işler.
    ///
    /// ## Deterministik Hata Sözleşmesi (`BLK-08`)
    ///
    /// Birden çok görev `Err` döndürse bile her zaman **en küçük `task_idx`**
    /// indeksli görevin hatası döner (`n_threads = 1` sırasıyla %100 özdeş).
    pub fn execute_tasks<T, F>(&self, tasks: &mut Vec<T>, op: F) -> Result<()>
    where
        T: Send + 'static,
        F: Fn(usize, usize, &mut T) -> Result<()> + Send + Sync + 'static,
    {
        let n_tasks = tasks.len();
        if n_tasks == 0 {
            return Ok(());
        }
        if n_tasks == 1 || self.n_threads <= 1 {
            for (idx, item) in tasks.iter_mut().enumerate() {
                op(0, idx, item)?;
            }
            return Ok(());
        }
        let Some(shared) = &self.shared else {
            for (idx, item) in tasks.iter_mut().enumerate() {
                op(0, idx, item)?;
            }
            return Ok(());
        };

        let slots: TaskSlotList<T> =
            Arc::new(tasks.drain(..).map(|item| Mutex::new(Some(item))).collect());
        let slots_worker = Arc::clone(&slots);
        let first_error: FirstErrorSlot = Arc::new(Mutex::new(None));
        let first_error_worker = Arc::clone(&first_error);

        let batch = Arc::new(TaskBatch {
            n_tasks,
            next_task: CachePadded::new(AtomicUsize::new(0)),
            remaining: CachePadded::new(AtomicUsize::new(n_tasks)),
            first_error_idx: CachePadded::new(AtomicUsize::new(usize::MAX)),
            first_error: Arc::clone(&first_error),
            runner: Box::new(move |worker_id: usize, idx: usize, batch_ref: &TaskBatch| {
                let mut guard = slots_worker[idx].lock();
                if let Some(ref mut item) = *guard
                    && let Err(err) = op(worker_id, idx, item)
                {
                    record_first_error(&first_error_worker, batch_ref, idx, err);
                }
            }),
            done_lock: Mutex::new(false),
            done_cv: Condvar::new(),
        });

        self.dispatch_and_wait(shared, &batch);
        let had_error = batch.first_error_idx.load(Ordering::Acquire) != usize::MAX;
        drop(batch);

        // Bariyer tamamlandı: öğeleri orijinal sırada `tasks` vektörüne geri
        // koy; tek sahip isek (`Arc::try_unwrap`) hiç kilit almadan taşı.
        match Arc::try_unwrap(slots) {
            Ok(slot_vec) => {
                for slot in slot_vec {
                    if let Some(item) = slot.into_inner() {
                        tasks.push(item);
                    }
                }
            }
            Err(slots_arc) => {
                for slot in slots_arc.iter() {
                    if let Some(item) = slot.lock().take() {
                        tasks.push(item);
                    }
                }
            }
        }
        if had_error && let Some((_, err)) = first_error.lock().take() {
            return Err(err);
        }
        Ok(())
    }

    /// Statik ömürlü (`'static`) veya `Arc` ile paylaşılan veriler üzerinde
    /// `0..n_tasks` indeksli görevleri kalıcı `WorkerPool` üzerinde paralel çalıştırır.
    pub fn execute_indexed<F>(&self, n_tasks: usize, op: F) -> Result<()>
    where
        F: Fn(usize, usize) -> Result<()> + Send + Sync + 'static,
    {
        if n_tasks == 0 {
            return Ok(());
        }
        if n_tasks == 1 || self.n_threads <= 1 {
            for idx in 0..n_tasks {
                op(0, idx)?;
            }
            return Ok(());
        }
        let Some(shared) = &self.shared else {
            for idx in 0..n_tasks {
                op(0, idx)?;
            }
            return Ok(());
        };

        let first_error: FirstErrorSlot = Arc::new(Mutex::new(None));
        let first_error_worker = Arc::clone(&first_error);

        let batch = Arc::new(TaskBatch {
            n_tasks,
            next_task: CachePadded::new(AtomicUsize::new(0)),
            remaining: CachePadded::new(AtomicUsize::new(n_tasks)),
            first_error_idx: CachePadded::new(AtomicUsize::new(usize::MAX)),
            first_error: Arc::clone(&first_error),
            runner: Box::new(move |worker_id: usize, idx: usize, batch_ref: &TaskBatch| {
                if let Err(err) = op(worker_id, idx) {
                    record_first_error(&first_error_worker, batch_ref, idx, err);
                }
            }),
            done_lock: Mutex::new(false),
            done_cv: Condvar::new(),
        });

        self.dispatch_and_wait(shared, &batch);
        let had_error = batch.first_error_idx.load(Ordering::Acquire) != usize::MAX;
        drop(batch);

        if had_error && let Some((_, err)) = first_error.lock().take() {
            return Err(err);
        }
        Ok(())
    }

    fn dispatch_and_wait(&self, shared: &Arc<SharedQueue>, batch: &Arc<TaskBatch>) {
        {
            let mut st = shared.state.lock();
            st.generation = st.generation.wrapping_add(1);
            st.active_batches.push(Arc::clone(batch));
        }
        shared.work_cv.notify_all();

        // Çağıran iş parçacığı (`worker_id = 0`) kendi partisinin görevlerine
        // doğrudan katılır; böylece havuzdaki diğer tüm işçiler başka karelerle
        // meşgul olsa bile alt parti asla açlıktan kilitlenmez (deadlock = 0).
        batch.drain_tasks(0);

        // Bu partideki tüm görevlerin bitmesini bekle.
        if batch.remaining.load(Ordering::Acquire) != 0 {
            let mut done = batch.done_lock.lock();
            while !*done && batch.remaining.load(Ordering::Acquire) != 0 {
                done = batch.done_cv.wait(done);
            }
        }

        let mut st = shared.state.lock();
        st.active_batches.retain(|b| !Arc::ptr_eq(b, batch));
    }
}

#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
fn worker_loop(worker_id: usize, shared: &SharedQueue) {
    enum WorkItem {
        Batch(Arc<TaskBatch>),
        Async(AsyncTaskFn),
    }
    let mut seen_gen = 0u64;
    loop {
        let item = {
            let mut st = shared.state.lock();
            loop {
                if st.shutdown {
                    return;
                }
                if let Some(batch) = st
                    .active_batches
                    .iter()
                    .find(|b| b.next_task.load(Ordering::Relaxed) < b.n_tasks)
                    .map(Arc::clone)
                {
                    seen_gen = st.generation;
                    break WorkItem::Batch(batch);
                }
                if let Some(task) = st.async_tasks.pop_front() {
                    seen_gen = st.generation;
                    break WorkItem::Async(task);
                }
                if st.generation != seen_gen {
                    seen_gen = st.generation;
                    continue;
                }
                st = shared.work_cv.wait(st);
            }
        };
        match item {
            WorkItem::Batch(batch) => {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    batch.drain_tasks(worker_id);
                }));
            }
            WorkItem::Async(task) => {
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    #[cfg(feature = "std")]
    use alloc::vec::Vec;

    use super::WorkerPool;
    #[cfg(feature = "std")]
    use crate::error::Error;

    #[test]
    fn tek_thread_havuzu_yardimci_thread_acmaz() {
        let pool = WorkerPool::new(1);
        assert_eq!(pool.n_threads(), 1);
        assert_eq!(pool.spawned_workers(), 0);
        assert!(!pool.is_parallel());
        let mut items = vec![1usize, 2, 3, 4];
        pool.execute_tasks(&mut items, |_wid, idx, val| {
            *val += idx * 10;
            Ok(())
        })
        .expect("tek thread çalışmalı");
        assert_eq!(items, vec![1, 12, 23, 34]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn cok_thread_havuzu_tum_gorevleri_ve_en_kucuk_indeksli_hatayi_belirlenimli_dondurur() {
        let pool = WorkerPool::new(4);
        assert_eq!(pool.n_threads(), 4);
        assert_eq!(pool.spawned_workers(), 3);
        assert!(pool.is_parallel());

        let mut items: Vec<usize> = (0..32).collect();
        pool.execute_tasks(&mut items, |wid, idx, val| {
            assert!(wid < 4);
            *val = idx * 3 + 1;
            Ok(())
        })
        .expect("paralel görevler tamamlanmalı");
        for (idx, &v) in items.iter().enumerate() {
            assert_eq!(v, idx * 3 + 1);
        }

        // Birden fazla görev hata verdiğinde her zaman en küçük indeksli (idx=3)
        // görevin hatası dönmelidir.
        let tekrar = if cfg!(miri) { 4 } else { 16 };
        for _ in 0..tekrar {
            let mut err_items: Vec<usize> = (0..24).collect();
            let res = pool.execute_tasks(&mut err_items, |_wid, idx, _val| {
                if idx == 7 {
                    return Err(Error::InvalidData("hata-7"));
                }
                if idx == 3 {
                    return Err(Error::InvalidData("hata-3"));
                }
                if idx == 15 {
                    return Err(Error::InvalidData("hata-15"));
                }
                Ok(())
            });
            assert_eq!(res, Err(Error::InvalidData("hata-3")));
            assert_eq!(err_items.len(), 24);
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn ardisik_hizli_kucuk_parti_kusaklar_arasi_yaris_ve_underflow_uretmez() {
        let pool = WorkerPool::new(if cfg!(miri) { 4 } else { 8 });
        let turlar = if cfg!(miri) { 32usize } else { 5_000usize };
        for round in 0..turlar {
            let n_tasks = 2 + (round % 16);
            let mut items: Vec<usize> = (0..n_tasks).collect();
            pool.execute_tasks(&mut items, move |_wid, idx, val| {
                *val = round.wrapping_mul(31).wrapping_add(idx);
                Ok(())
            })
            .expect("ardışık küçük parti tamamlanmalı");
            assert_eq!(items.len(), n_tasks);
            for (idx, &v) in items.iter().enumerate() {
                assert_eq!(v, round.wrapping_mul(31).wrapping_add(idx));
            }
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn ic_ice_paralel_kare_ve_alt_parti_gorevleri_kilitlenmeden_tamamlanir() {
        use std::sync::{Arc, Condvar, Mutex};
        let thread_list: &[usize] = if cfg!(miri) { &[2, 4] } else { &[2, 4, 8] };
        let n_frames = if cfg!(miri) { 8usize } else { 32usize };
        for &n_threads in thread_list {
            let pool = WorkerPool::new(n_threads);
            let done_count = Arc::new((Mutex::new(0usize), Condvar::new()));
            let sums = Arc::new(Mutex::new(vec![0usize; n_frames]));
            for f_idx in 0..n_frames {
                let sub_pool = pool.handle();
                let dc = Arc::clone(&done_count);
                let sums_ref = Arc::clone(&sums);
                pool.spawn_async(move || {
                    let mut tiles: Vec<usize> = (0..12).collect();
                    sub_pool
                        .execute_tasks(&mut tiles, move |_wid, t_idx, val| {
                            *val = (f_idx + 1) * 100 + t_idx;
                            Ok(())
                        })
                        .expect("iç içe alt parti çalışmalı");
                    let total: usize = tiles.into_iter().sum();
                    sums_ref.lock().unwrap()[f_idx] = total;
                    let mut c = dc.0.lock().unwrap();
                    *c += 1;
                    dc.1.notify_all();
                });
            }
            let mut c = done_count.0.lock().unwrap();
            while *c < n_frames {
                c = done_count.1.wait(c).unwrap();
            }
            let results = sums.lock().unwrap();
            for (f_idx, &total) in results.iter().enumerate() {
                let expected: usize = (0..12).map(|t| (f_idx + 1) * 100 + t).sum();
                assert_eq!(total, expected);
            }
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn execute_indexed_en_kucuk_indeksli_hatayi_belirlenimli_dondurur() {
        let pool = WorkerPool::new(4);
        let tekrar = if cfg!(miri) { 4 } else { 16 };
        for _ in 0..tekrar {
            let res = pool.execute_indexed(24, |_wid, idx| {
                if idx == 11 {
                    return Err(Error::InvalidData("idx-11"));
                }
                if idx == 4 {
                    return Err(Error::InvalidData("idx-4"));
                }
                if idx == 19 {
                    return Err(Error::InvalidData("idx-19"));
                }
                Ok(())
            });
            assert_eq!(res, Err(Error::InvalidData("idx-4")));
        }
    }

    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    #[test]
    fn scope_workers_butcesi_n_threads_sinirini_asmaz_ve_drop_ile_iade_edilir() {
        let pool = WorkerPool::new(4);
        let g1 = pool.acquire_scope_workers(2);
        assert_eq!(g1.total_threads(), 3);
        let g2 = pool.acquire_scope_workers(3);
        assert_eq!(g2.total_threads(), 2);
        let g3 = pool.acquire_scope_workers(2);
        assert_eq!(g3.total_threads(), 1);
        drop(g2);
        let g4 = pool.acquire_scope_workers(2);
        assert_eq!(g4.total_threads(), 2);
        drop(g1);
        drop(g3);
        drop(g4);
        let g_full = pool.acquire_scope_workers(8);
        assert_eq!(g_full.total_threads(), 4);
    }

    #[cfg(feature = "std")]
    #[test]
    fn panikleyen_paralel_gorev_hata_dondurur_ve_havuzu_saglam_birakir() {
        let pool = WorkerPool::new(2);
        let mut items = vec![10usize, 20usize];
        let res = pool.execute_tasks(&mut items, |_wid, idx, val| {
            if idx == 1 {
                panic!("test-paralel-panik");
            }
            *val += 1;
            Ok(())
        });
        assert_eq!(
            res,
            Err(Error::InvalidState("paralel işçi görevi panikledi"))
        );

        let res_idx = pool.execute_indexed(2, |_wid, idx| {
            if idx == 0 {
                panic!("test-indexed-panik");
            }
            Ok(())
        });
        assert_eq!(
            res_idx,
            Err(Error::InvalidState("paralel işçi görevi panikledi"))
        );

        // Panik sonrasında havuz işçileri canlı kalmalı ve sonraki partiyi normal çözmelidir.
        let mut healthy = vec![1usize, 2, 3, 4];
        pool.execute_tasks(&mut healthy, |_wid, idx, val| {
            *val = idx + 100;
            Ok(())
        })
        .expect("panik sonrası havuz sağlıklı çalışmalı");
        assert_eq!(healthy, vec![100, 101, 102, 103]);
    }
}
