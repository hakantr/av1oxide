//! CDF bağlamları ve varsayılan tablolar (S2a.3).
//!
//! Bu modül dav1d `src/cdf.c` / `src/cdf.h` dosyalarının C yolunun skaler
//! Rust karşılığıdır: **dört kuantizasyon kategorisi** için `init_static`,
//! `copy` ve `update` semantiği. Tablo **değerleri** üretilmiş dosyadadır
//! ([`crate::tables::cdf`]); bu modül yalnız yerleşim ve semantiği taşır.
//!
//! ## Kaynak izlenebilirliği (SRC-07)
//!
//! | Öğe | dav1d kaynağı (@ `99f3354`) | AV1 spec |
//! |---|---|---|
//! | `default_cdf` (mode + mv + kfym) | `src/cdf.c:77-697` | 8.2.6, 8.3.2 |
//! | `default_coef_cdf[4]` | `src/cdf.c:699-3913` | 8.2.6 |
//! | `dav1d_cdf_thread_update` | `src/cdf.c:3915-4004` | 7.20 (CDF güncellemesi) |
//! | `dav1d_cdf_thread_init_static` | `src/cdf.c:4023-4026` | — |
//! | `dav1d_cdf_thread_copy` | `src/cdf.c:4028-4039` | — |
//! | `CdfModeContext`, `CdfCoefContext`, `CdfMvContext` | `src/cdf.h:37-122` | 8.2.6 |
//! | `update_cdf_1d..4d` sayaç sıfırlama | `src/cdf.c:3918-3937` | 7.20 |
//!
//! ## Yerleşim
//!
//! [`CdfContext`] hücreleri (`u16`) dav1d `CdfContext` yapısının **bellek
//! yerleşiminin** aynısıdır (x86_64, `sizeof = 13472` bayt = 6736 hücre):
//!
//! | Bölge | Hücre aralığı | dav1d alanı |
//! |---|---|---|
//! | `coef` | `0 .. 3056` | `CdfCoefContext` (kategoriye göre değişir) |
//! | `mode` | `3056 .. 6192` | `CdfModeContext` |
//! | `mv` | `6192 .. 6336` | `CdfMvContext` |
//! | `kfym` | `6336 .. 6736` | `kfym[5][5][16]` |
//!
//! Araya giren hizalama dolgusu da hücre olarak yer alır ve sıfırdır; böylece
//! [`CdfContext::write_le_bytes`] çıktısı dav1d'nin ham belleğiyle **bayt
//! bayt** karşılaştırılabilir. Sayaç hücreleri (`cdf[n_symbols]`) dizilerin
//! içindedir (dav1d düzeni), ayrı bir alan değildir.
//!
//! ## `update` semantiği (dav1d `cdf_thread_update`)
//!
//! `update`, tile sonunda adapte olmuş CDF'leri bir sonraki kareye taşır ve
//! **sayaç hücrelerini sıfırlar**. Kare türüne göre kopyalama **kısmidir**:
//!
//! - anahtar/kamera içi (`frame_type & 1 == 0`): yalnız `[0, 5458)` hücreleri
//!   kaynaktan kopyalanır; `intrabc` (5458..5472), `y_mode..obmc` ve `mv`
//!   (5472..6336) ile `kfym` (6336..6736) **dst'de korunur**;
//! - inter/switch (`frame_type & 1 == 1`): ek olarak `[5472, 6336)` kopyalanır;
//! - her durumda `intrabc` ve `kfym` kopyalanmaz.
//!
//! Kopyalanan bölgede sayaç hücreleri sıfırlanır ([`RESET_KEY`],
//! [`RESET_INTER`]); bu listeler **ölçümle** üretilir, elle yazılmaz
//! (`scripts/oracle/cdf_oracle.c`).
//!
//! ## Kapsam sınırı
//!
//! Çok iş parçacıklı CDF paylaşımı (dav1d `ref`/`progress` alanları) **F7**
//! kapsamındadır; burada `init_static` + `copy` (statik tablo) yolu ve
//! `update` vardır. Kare içi tile/blok sözdizimi F2b'dir.

#![forbid(unsafe_code)]

use crate::tables::cdf::{
    CDF_CELLS, COEF_CELLS, COPY_INTER_END, COPY_INTER_START, COPY_KEY_CELLS, DEFAULT_COEF,
    DEFAULT_MODE_MV_KFYM, KFYM_OFF, MODE_OFF, MV_OFF, RESET_INTER, RESET_KEY,
};

/// `CdfContext` hücre sayısı (dav1d `sizeof(CdfContext)/2`).
pub const CDF_CELL_COUNT: usize = CDF_CELLS;
/// `CdfContext` bayt sayısı (dav1d `sizeof(CdfContext)`; 13472).
pub const CDF_BYTE_COUNT: usize = CDF_CELLS * 2;
/// `coef` bölgesinin hücre sayısı (`CdfCoefContext`).
pub const CDF_COEF_CELL_COUNT: usize = COEF_CELLS;
/// `mode` bölgesinin hücre başlangıcı.
pub const CDF_MODE_OFFSET: usize = MODE_OFF;
/// `mv` bölgesinin hücre başlangıcı.
pub const CDF_MV_OFFSET: usize = MV_OFF;
/// `kfym` bölgesinin hücre başlangıcı.
pub const CDF_KFYM_OFFSET: usize = KFYM_OFF;
/// `update` kopyalama aralığı: anahtar/kamera içi üst sınır (hücre).
pub const CDF_COPY_KEY_CELLS: usize = COPY_KEY_CELLS;
/// `update` kopyalama aralığı: inter/switch başlangıcı (hücre).
pub const CDF_COPY_INTER_START: usize = COPY_INTER_START;
/// `update` kopyalama aralığı: inter/switch bitişi (hücre).
pub const CDF_COPY_INTER_END: usize = COPY_INTER_END;
/// `update` (anahtar/kamera içi) sıfırlanan hücreler — ölçümle üretilir.
pub const CDF_RESET_KEY: &[u32] = &RESET_KEY;
/// `update` (inter/switch) ek olarak sıfırlanan hücreler.
pub const CDF_RESET_INTER: &[u32] = &RESET_INTER;

/// Kare türü sınıfı: dav1d `IS_KEY_OR_INTRA` / `IS_INTER_OR_SWITCH`.
///
/// `frame_type & 1` ayrımı AV1 spec §6.8.2 ile aynıdır: `KEY = 0`,
/// `INTER = 1`, `INTRA = 2`, `SWITCH = 3` (`dav1d/headers.h:116-119`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// `KEY` veya `INTRA`: `frame_type & 1 == 0`.
    KeyOrIntra,
    /// `INTER` veya `SWITCH`: `frame_type & 1 == 1`.
    InterOrSwitch,
}

impl FrameKind {
    /// dav1d `IS_KEY_OR_INTRA`: `!(frame_type & 1)`.
    pub fn from_frame_type(frame_type: u8) -> Self {
        if frame_type & 1 == 0 {
            Self::KeyOrIntra
        } else {
            Self::InterOrSwitch
        }
    }
}

/// `dav1d_cdf_thread_init_static` (`cdf.c:4023`): `qidx` → kategori.
///
/// `qcat = (qidx > 20) + (qidx > 60) + (qidx > 120)`.
pub fn qcat_for_qidx(qidx: u8) -> u8 {
    u8::from(qidx > 20) + u8::from(qidx > 60) + u8::from(qidx > 120)
}

/// dav1d `CdfThreadContext` (statik tablo yolu; F7'de sahipli `ref` yolu eklenir).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CdfThreadContext {
    qcat: u8,
}

impl CdfThreadContext {
    /// `dav1d_cdf_thread_init_static`: referans yok, yalnız kategori.
    pub fn init_static(qidx: u8) -> Self {
        Self {
            qcat: qcat_for_qidx(qidx),
        }
    }

    /// Seçilen kuantizasyon kategorisi (0..=3).
    pub fn qcat(&self) -> u8 {
        self.qcat
    }

    /// `dav1d_cdf_thread_copy` (`cdf.c:4028`), statik dal: dört kategoriden
    /// birinin varsayılan tablolarını `dst`'ye kopyalar.
    pub fn copy_into(&self, dst: &mut CdfContext) {
        *dst = CdfContext::from_default(self.qcat);
    }
}

/// dav1d `CdfContext`: `coef` + `mode` + `mv` + `kfym` hücreleri.
///
/// Hücreler dav1d bellek yerleşimini birebir izler; erişimciler bölge
/// dilimleri döndürür. Sayaç hücreleri dizilerin sonundadır (dav1d düzeni).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CdfContext {
    cells: [u16; CDF_CELLS],
}

impl Default for CdfContext {
    fn default() -> Self {
        Self::from_default(0)
    }
}

impl CdfContext {
    /// Kategori `qcat` (0..=3) için varsayılan bağlam — `dav1d_cdf_thread_copy`
    /// ile aynı sonuç.
    ///
    /// # Panics
    /// `qcat > 3` ise (dav1d'de dizi taşması; burada açık hata).
    pub fn from_default(qcat: u8) -> Self {
        assert!(qcat <= 3, "qcat 0..=3 olmalı: {qcat}");
        let mut cells = [0u16; CDF_CELLS];
        cells[..COEF_CELLS].copy_from_slice(&DEFAULT_COEF[qcat as usize]);
        cells[COEF_CELLS..].copy_from_slice(&DEFAULT_MODE_MV_KFYM);
        Self { cells }
    }

    /// `coef` bölgesi (`CdfCoefContext`).
    pub fn coef(&self) -> &[u16] {
        &self.cells[..COEF_CELLS]
    }

    /// `mode` bölgesi (`CdfModeContext`).
    pub fn mode(&self) -> &[u16] {
        &self.cells[MODE_OFF..MV_OFF]
    }

    /// `mv` bölgesi (`CdfMvContext`).
    pub fn mv(&self) -> &[u16] {
        &self.cells[MV_OFF..KFYM_OFF]
    }

    /// `kfym` bölgesi (`kfym[5][5][16]`).
    pub fn kfym(&self) -> &[u16] {
        &self.cells[KFYM_OFF..]
    }

    /// Bütün hücreler (dav1d belleği sırasıyla).
    pub fn cells(&self) -> &[u16; CDF_CELLS] {
        &self.cells
    }

    /// Bütün hücreler, değiştirilebilir (F2b alan erişimi için).
    pub fn cells_mut(&mut self) -> &mut [u16; CDF_CELLS] {
        &mut self.cells
    }

    /// `coef` bölgesi, değiştirilebilir.
    pub fn coef_mut(&mut self) -> &mut [u16] {
        &mut self.cells[..COEF_CELLS]
    }

    /// `mode` bölgesi, değiştirilebilir.
    pub fn mode_mut(&mut self) -> &mut [u16] {
        &mut self.cells[MODE_OFF..MV_OFF]
    }

    /// `mv` bölgesi, değiştirilebilir.
    pub fn mv_mut(&mut self) -> &mut [u16] {
        &mut self.cells[MV_OFF..KFYM_OFF]
    }

    /// Hücreleri little-endian bayt akışına yazar (`CDF_CELLS * 2` bayt).
    ///
    /// # Panics
    /// `out` uzunluğu `CDF_CELLS * 2` değilse.
    pub fn write_le_bytes(&self, out: &mut [u8]) {
        assert_eq!(out.len(), CDF_CELLS * 2, "çıktı tamponu yanlış boyutta");
        for (i, cell) in self.cells.iter().enumerate() {
            out[2 * i..2 * i + 2].copy_from_slice(&cell.to_le_bytes());
        }
    }

    /// Little-endian bayt akışından bağlam kurar (`write_le_bytes` tersi).
    ///
    /// # Panics
    /// `bytes` uzunluğu `CDF_CELLS * 2` değilse.
    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), CDF_CELLS * 2, "girdi tamponu yanlış boyutta");
        let mut cells = [0u16; CDF_CELLS];
        for (i, cell) in cells.iter_mut().enumerate() {
            *cell = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
        }
        Self { cells }
    }

    /// `dav1d_cdf_thread_update` (`cdf.c:3915`).
    ///
    /// Kare türüne göre kısmi kopyalama + sayaç sıfırlama. `src` adapte olmuş
    /// tile CDF'leridir; `self` bir sonraki karenin bağlamıdır ve kopyalanmayan
    /// bölgelerde **eski değerlerini korur** (dav1d'de `memcpy` kapsamı budur).
    pub fn update(&mut self, src: &CdfContext, kind: FrameKind) {
        // `memcpy(dst, src, offsetof(CdfContext, m.intrabc))`
        self.cells[..COPY_KEY_CELLS].copy_from_slice(&src.cells[..COPY_KEY_CELLS]);
        // Sayaç sıfırlama (kopyalanan bölgede).
        for &i in &RESET_KEY {
            self.cells[i as usize] = 0;
        }
        if kind == FrameKind::InterOrSwitch {
            // `memcpy(dst->m.y_mode, src->m.y_mode, offsetof(kfym) - offsetof(y_mode))`
            self.cells[COPY_INTER_START..COPY_INTER_END]
                .copy_from_slice(&src.cells[COPY_INTER_START..COPY_INTER_END]);
            for &i in &RESET_INTER {
                self.cells[i as usize] = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    /// `init_static` dört kategoriyi dav1d sınırlarıyla seçer.
    #[test]
    fn qcat_sinirlari() {
        for (qidx, beklenen) in [
            (0u8, 0u8),
            (1, 0),
            (20, 0),
            (21, 1),
            (60, 1),
            (61, 2),
            (120, 2),
            (121, 3),
            (200, 3),
            (255, 3),
        ] {
            assert_eq!(qcat_for_qidx(qidx), beklenen, "qidx={qidx}");
            assert_eq!(CdfThreadContext::init_static(qidx).qcat(), beklenen);
        }
    }

    /// `copy` dört kategoride farklı `coef`, ortak `mode/mv/kfym` üretir.
    #[test]
    fn copy_kategorileri() {
        let mut ctx = CdfContext::default();
        CdfThreadContext::init_static(61).copy_into(&mut ctx);
        assert_eq!(ctx, CdfContext::from_default(2));
        let a = CdfContext::from_default(0);
        let b = CdfContext::from_default(3);
        assert_ne!(a.coef(), b.coef(), "kategoriler coef'te farklı olmalı");
        assert_eq!(a.mode(), b.mode());
        assert_eq!(a.mv(), b.mv());
        assert_eq!(a.kfym(), b.kfym());
    }

    /// Sayaç sıfırlama listeleri kopyalama bölgelerinin içindedir ve
    /// birbirinden ayrıktır (dav1d `update` sırası ile tutarlı).
    #[test]
    fn reset_listeleri_tutarli() {
        assert!(RESET_KEY.iter().all(|&i| (i as usize) < COPY_KEY_CELLS));
        assert!(
            RESET_INTER
                .iter()
                .all(|&i| (i as usize) >= COPY_INTER_START && (i as usize) < COPY_INTER_END)
        );
        assert!(RESET_KEY.windows(2).all(|w| w[0] < w[1]), "sıralı olmalı");
        assert!(RESET_INTER.windows(2).all(|w| w[0] < w[1]), "sıralı olmalı");
        const {
            assert!(COPY_KEY_CELLS <= COPY_INTER_START);
            assert!(COPY_INTER_END <= CDF_CELLS);
        }
        assert_eq!(MODE_OFF, COEF_CELLS);
        assert_eq!(KFYM_OFF, MV_OFF + 144);
    }

    /// `update` sayaç hücrelerini sıfırlar, kopyalanmayan bölgeleri korur.
    #[test]
    fn update_sayaclari_sifirlar_ve_korur() {
        let mut src = CdfContext::from_default(1);
        // Bütün hücreleri 0xFFFF yap (sayaçlar dâhil): update onları sıfırlamalı.
        for cell in src.cells_mut().iter_mut() {
            *cell = 0xFFFF;
        }
        let mut dst = CdfContext::from_default(1);
        let once = dst.clone();
        dst.update(&src, FrameKind::KeyOrIntra);
        for &i in &RESET_KEY {
            assert_eq!(dst.cells()[i as usize], 0, "sayaç sıfırlanmadı: {i}");
        }
        // Kopyalanan bölgede sayaç dışındaki bütün hücreler kaynaktan gelir.
        for i in 0..COPY_KEY_CELLS {
            let beklenen = if RESET_KEY.contains(&(i as u32)) {
                0
            } else {
                src.cells()[i]
            };
            assert_eq!(dst.cells()[i], beklenen, "anahtar hücresi: {i}");
        }
        // intrabc (5458..5472) ve kfym (6336..) korunur.
        assert_eq!(
            &dst.cells()[COPY_KEY_CELLS..COPY_INTER_START],
            &once.cells()[COPY_KEY_CELLS..COPY_INTER_START]
        );
        assert_eq!(&dst.cells()[KFYM_OFF..], &once.cells()[KFYM_OFF..]);
        // Inter: y_mode..mv bölgesi de kopyalanır (sayaçlar sıfırlanarak).
        let mut dst = once.clone();
        dst.update(&src, FrameKind::InterOrSwitch);
        for i in COPY_INTER_START..COPY_INTER_END {
            let beklenen = if RESET_INTER.contains(&(i as u32)) {
                0
            } else {
                src.cells()[i]
            };
            assert_eq!(dst.cells()[i], beklenen, "inter hücresi: {i}");
        }
        for &i in &RESET_INTER {
            assert_eq!(dst.cells()[i as usize], 0, "inter sayacı sıfırlanmadı: {i}");
        }
        assert_eq!(&dst.cells()[KFYM_OFF..], &once.cells()[KFYM_OFF..]);
    }

    /// Bayt dönüşümü gidiş-dönüş kayıpsızdır ve 13472 bayttır.
    #[test]
    fn bayt_donusumu() {
        for q in 0..4u8 {
            let ctx = CdfContext::from_default(q);
            let mut buf = vec![0u8; CDF_CELLS * 2];
            ctx.write_le_bytes(&mut buf);
            assert_eq!(buf.len(), 13472);
            assert_eq!(CdfContext::from_le_bytes(&buf), ctx);
        }
        assert_eq!(FrameKind::from_frame_type(0), FrameKind::KeyOrIntra);
        assert_eq!(FrameKind::from_frame_type(1), FrameKind::InterOrSwitch);
        assert_eq!(FrameKind::from_frame_type(2), FrameKind::KeyOrIntra);
        assert_eq!(FrameKind::from_frame_type(3), FrameKind::InterOrSwitch);
    }
}
