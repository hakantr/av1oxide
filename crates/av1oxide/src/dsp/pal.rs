//! Palette yardımcı işlevleri — S2b.2.
//!
//! Bu modül dav1d `Dav1dPalDSPContext` (`src/pal.h:44-47`) karşılığının
//! tiplerini tanımlar. `pal_pred` intra tahmin bağlamındadır
//! ([`crate::dsp::ipred`]); burada yalnız **`pal_idx_finish`** vardır.
//!
//! ## Kaynak eşlemesi (dav1d @ `99f3354`, salt-okunur)
//!
//! | Kural | Kaynak |
//! |---|---|
//! | `pal_idx_finish_c` gövdesi: görünmez kenarı doldur, 4-bit paketle | `src/pal.c:37-57` |
//! | Paketleme: `dst[x] = src[2x] \| (src[2x+1] << 4)` | `src/pal.c:46-47` |
//! | Görünür genişlikten sonrası: `src[w-1] * 0x11` | `src/pal.c:48-49` |
//! | Görünür yükseklikten sonrası: son paketlenmiş satırın kopyası | `src/pal.c:52-55` |
//! | Boyut kısıtları: `bw,bh` 4..64 ve 2'nin kuvveti; `w,h` 4'ün katı | `src/pal.c:40-43` |
//!
//! ## Bellek sözleşmesi (ADR-0003)
//!
//! - `src`: `bw × bh` bayt; her bayt 0..7 indeksidir.
//! - `dst`: `(bw/2) × bh` bayt; çekirdek bu alanın **tamamını** yazar
//!   (görünür bölge + görünmez kenar dolgusu).
//! - Hizalama varsayımı yoktur.
//!
//! Bit derinliğinden **bağımsızdır** (indeksler her zaman 8-bit bayttır), bu
//! yüzden dispatch yuvası bit derinliği başına ayrı tutulur ama aynı işlevi
//! gösterir (`DspDispatch::pal`, ADR-0003 §2.1: bit derinliği yuvanın
//! parçasıdır; F3/F4'te ISA yolu bit derinliğine göre ayrışabilir).

#![forbid(unsafe_code)]

use crate::dsp::scalar::pal as scalar;

/// Bir `pal_idx_finish` çekirdeğinin imzası — dav1d `decl_pal_idx_finish_fn`
/// (`src/pal.h:34-37`).
///
/// `bw`/`bh` bloğun (görünmez kenar dâhil) tam boyutu, `w`/`h` görünür
/// boyuttur; `w = bw`, `h = bh` olabilir.
pub type PalIdxFinishFn = fn(dst: &mut [u8], src: &[u8], bw: usize, bh: usize, w: usize, h: usize);

/// Palet çekirdek ailesi — dav1d `Dav1dPalDSPContext`.
#[derive(Clone, Copy)]
pub struct PalDsp {
    /// `pal_idx_finish` — indeksleri 4-bit paketler ve görünmez kenarı doldurur.
    pub pal_idx_finish: PalIdxFinishFn,
}

impl PalDsp {
    /// Skaler referans tablo (DSP-01/6: `None` → bu yol).
    pub const SCALAR: Self = Self {
        pal_idx_finish: scalar::pal_idx_finish,
    };
}

impl std::fmt::Debug for PalDsp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PalDsp { pal_idx_finish: fn }")
    }
}

/// P45: SIMD `pal_idx_finish` girişlerinin sözleşme dışı girdide skaler gövdeye
/// devrettiği çağrı sayısı (süreç başından beri). Çözme hattında 0 olmalıdır.
static PAL_SKALER_DEVIR: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// [`PAL_SKALER_DEVIR`] değeri.
#[must_use]
pub fn pal_skaler_devir_sayisi() -> u64 {
    PAL_SKALER_DEVIR.load(core::sync::atomic::Ordering::Relaxed)
}

/// Skaler devri sayar (yalnız SIMD girişlerinin soğuk yolu).
#[cold]
pub fn pal_skaler_devri_say() {
    PAL_SKALER_DEVIR.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
}

/// P45: `pal_idx_finish` SIMD girişinin O(1) sözleşmesi:
/// - `bw, bh ∈ {4, 8, 16, 32, 64}` (`4..=64` ve 2'nin kuvveti)
/// - `w ∈ 4..=bw` ve `w % 4 == 0`
/// - `h ∈ 4..=bh` ve `h % 4 == 0`
/// - `src_len >= bw * bh`
/// - `dst_len >= (bw / 2) * bh`
#[must_use]
pub const fn pal_idx_finish_sozlesmesi(
    dst_len: usize,
    src_len: usize,
    bw: usize,
    bh: usize,
    w: usize,
    h: usize,
) -> bool {
    bw >= 4
        && bw <= 64
        && bw.is_power_of_two()
        && bh >= 4
        && bh <= 64
        && bh.is_power_of_two()
        && w >= 4
        && w <= bw
        && w.is_multiple_of(4)
        && h >= 4
        && h <= bh
        && h.is_multiple_of(4)
        && src_len >= bw * bh
        && dst_len >= (bw / 2) * bh
}

/// P45: Sözleşme dışı `pal_idx_finish` girdisinde skaler devir sayacını artırır;
/// skaler gövdenin önkoşulları sağlanıyorsa [`scalar::pal_idx_finish`] çağırır,
/// değilse paniklemeden döner.
#[cold]
pub fn pal_idx_finish_skaler_devret(
    dst: &mut [u8],
    src: &[u8],
    bw: usize,
    bh: usize,
    w: usize,
    h: usize,
) {
    pal_skaler_devri_say();
    if pal_idx_finish_sozlesmesi(dst.len(), src.len(), bw, bh, w, h) {
        scalar::pal_idx_finish(dst, src, bw, bh, w, h);
    }
}

#[cfg(test)]
mod tests {
    use super::{pal_idx_finish_skaler_devret, pal_idx_finish_sozlesmesi, pal_skaler_devir_sayisi};

    #[test]
    fn p45_pal_sozlesmesi_ve_paniksiz_skaler_devir() {
        assert!(pal_idx_finish_sozlesmesi(8, 16, 4, 4, 4, 4));
        assert!(pal_idx_finish_sozlesmesi(32 * 64, 64 * 64, 64, 64, 12, 20));
        // Kısa dst / kısa src / 4'ün katı olmayan w,h / 2'nin kuvveti olmayan bw,bh reddedilir.
        assert!(!pal_idx_finish_sozlesmesi(7, 16, 4, 4, 4, 4));
        assert!(!pal_idx_finish_sozlesmesi(8, 15, 4, 4, 4, 4));
        assert!(!pal_idx_finish_sozlesmesi(32, 64, 8, 8, 6, 8));
        assert!(!pal_idx_finish_sozlesmesi(32, 64, 8, 8, 8, 6));
        assert!(!pal_idx_finish_sozlesmesi(18, 36, 6, 6, 4, 4));
        assert!(!pal_idx_finish_sozlesmesi(32, 64, 8, 8, 12, 8));

        let once = pal_skaler_devir_sayisi();
        let mut kisa_dst = [0u8; 4];
        let src = [0u8; 16];
        pal_idx_finish_skaler_devret(&mut kisa_dst, &src, 4, 4, 4, 4);
        assert!(pal_skaler_devir_sayisi() > once);
    }
}
