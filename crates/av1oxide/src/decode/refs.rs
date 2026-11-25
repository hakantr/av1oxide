//! Başlık düzeyi referans durumu: 8 yuva ve `show_existing_frame` (F2a / S2a.2).
//!
//! ## Sorumluluk
//!
//! AV1 §7.8 "Reference frame update process"in **başlık düzeyi** karşılığı:
//!
//! - [`HeaderReferenceState::refs`] — 8 referans yuvasının `FrameHeader`'ları
//!   (dav1d `c->refs[i].p.p.frame_hdr`),
//! - [`HeaderReferenceState::set_current`] — bir frame header OBU'sunun
//!   ayrıştırılması; tile group'lar tamamlanana kadar bekleyen kare,
//! - [`HeaderReferenceState::add_tiles`] ve
//!   [`HeaderReferenceState::complete_frame`] — tile sayısı `TileCols*TileRows`
//!   değerine ulaşınca kare "çözülmüş" sayılır, `refresh_frame_flags` bitleri
//!   referans yuvalarını günceller (dav1d `dav1d_submit_frame`, `src/obu.c`
//!   1650–1680),
//! - [`HeaderReferenceState::show_existing`] — `show_existing_frame`:
//!   gösterilen referans karenin başlığı döner; dav1d `src/obu.c` 1580–1640
//!   ile aynı biçimde, gösterilen kare **KEY** ise diğer 7 yuva da bu kareye
//!   yenilenir.
//!
//! ## Neden yalnız başlık?
//!
//! F2a'da kare üretimi yoktur (F2b). Ancak frame header ayrıştırması referans
//! karelerin **başlıklarına** bağımlıdır: `frame_size_with_refs()`,
//! `delta_frame_id` doğrulaması, segmentasyon/loop filter delta devralma,
//! skip mode ve global motion önceki karelerin alanlarını okur. Bu modül o
//! durumu piksel tamponu olmadan tutar.
//!
//! ## Upstream karşılığı (dav1d `99f3354`, salt-okunur)
//!
//! | Rust | dav1d kaynağı | AV1 spesifikasyonu |
//! |---|---|---|
//! | `refs[8]` | `src/internal.h` (`c->refs`), `src/obu.c` 1583–1642 | §7.8 referans güncelleme |
//! | `set_current` | `src/obu.c` 1283–1300 (`parse_frame_hdr` sonucu) | §5.9 frame header |
//! | `complete_frame` | `src/obu.c` 1653–1685 (`dav1d_submit_frame`) | §7.8, §6.10 |
//! | `show_existing` | `src/obu.c` 1581–1641 | §7.8, §5.9.2 |
//!
//! ## Bilinçli fark
//!
//! dav1d yuvalarda `Dav1dThreadPicture` tutar (piksel + CDF + segment haritası);
//! burada yalnız `FrameHeader` kopyası tutulur. Başlık düzeyi karşılaştırması
//! için bu yeterlidir; piksel tamponları F2b'nin sorumluluğundadır.

#![forbid(unsafe_code)]

use crate::bitstream::frame_header::{FrameHeader, NUM_REF_FRAMES};

/// 8 referans yuvası + bekleyen kare (başlık düzeyi).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HeaderReferenceState {
    refs: [Option<FrameHeader>; NUM_REF_FRAMES],
    /// Yuva başına `Dav1dPicture.showable` (dav1d `src/picture.c:205`,
    /// `src/obu.c:1544-1548`, `src/obu.c:1619-1621`). `FrameHeader.showable_frame`
    /// **başlık alanıdır**, bu ise yuvanın o anki gösterilebilirlik durumudur:
    /// kare çözülünce `showable_frame` yazılır, anahtar kare `show_existing` ile
    /// gösterilince bütün yuvalarda `0`lanır.
    showable: [bool; NUM_REF_FRAMES],
    current: Option<FrameHeader>,
    n_tiles: usize,
}

/// Yuva anlık görüntüsü: başlıklar + `showable` (bkz.
/// [`HeaderReferenceState::slots`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RefSlots {
    refs: [Option<FrameHeader>; NUM_REF_FRAMES],
    showable: [bool; NUM_REF_FRAMES],
}

impl HeaderReferenceState {
    /// Yuvaların anlık görüntüsü. dav1d `dav1d_submit_frame` erken hatalarında
    /// (eksik/uyumsuz referans, `src/decode.c:3452-3478`) yuvalar **hiç**
    /// yenilenmez; Rust başlık yuvalarını kare tamamlanınca yenilediği için
    /// bu görüntüden geri yükler ([`HeaderReferenceState::restore_slots`]).
    #[must_use]
    pub fn slots(&self) -> RefSlots {
        RefSlots {
            refs: self.refs.clone(),
            showable: self.showable,
        }
    }

    /// `mask` bitlerindeki yuvaları anlık görüntüden geri yükler.
    pub fn restore_slots(&mut self, slots: &RefSlots, mask: u8) {
        for slot in 0..NUM_REF_FRAMES {
            if mask & (1 << slot) != 0 {
                self.refs[slot] = slots.refs[slot].clone();
                self.showable[slot] = slots.showable[slot];
            }
        }
    }

    /// `mask` bitlerindeki yuvaları boşaltır (dav1d `dav1d_decode_frame`
    /// hatası: yenilenen yuvalar `unref` edilir, `src/decode.c:3704-3716`).
    pub fn clear_slots(&mut self, mask: u8) {
        for slot in 0..NUM_REF_FRAMES {
            if mask & (1 << slot) != 0 {
                self.refs[slot] = None;
                self.showable[slot] = false;
            }
        }
    }

    /// Tile birikimini sıfırlar; bekleyen kare kalır (dav1d tile group sıra
    /// hatası, `src/obu.c:1343-1349`).
    pub fn reset_tiles(&mut self) {
        self.n_tiles = 0;
    }

    /// Bütün yuvaları ve bekleyen kareyi temizler (yeni sequence header).
    pub fn reset(&mut self) {
        self.refs = Default::default();
        self.showable = [false; NUM_REF_FRAMES];
        self.current = None;
        self.n_tiles = 0;
    }

    /// Referans yuvaları (dav1d `c->refs[i].p.p.frame_hdr`).
    #[must_use]
    pub const fn refs(&self) -> &[Option<FrameHeader>; NUM_REF_FRAMES] {
        &self.refs
    }

    /// Tek yuva (indeks 0–7).
    #[must_use]
    pub fn reference(&self, slot: usize) -> Option<&FrameHeader> {
        self.refs.get(slot).and_then(Option::as_ref)
    }

    /// Tile group'ları bekleyen kare.
    #[must_use]
    pub const fn current(&self) -> Option<&FrameHeader> {
        self.current.as_ref()
    }

    /// Bekleyen kareyi (yeni frame header) atar ve tile sayacını sıfırlar.
    pub fn set_current(&mut self, header: FrameHeader) {
        self.current = Some(header);
        self.n_tiles = 0;
    }

    /// Bekleyen kareyi ve tile sayacını temizler.
    pub fn clear_current(&mut self) {
        self.current = None;
        self.n_tiles = 0;
    }

    /// Şimdiye kadar gelen tile sayısı.
    #[must_use]
    pub const fn n_tiles(&self) -> usize {
        self.n_tiles
    }

    /// Gelen tile sayısını artırır (dav1d `c->n_tiles += end - start + 1`).
    pub fn add_tiles(&mut self, count: usize) {
        self.n_tiles = self.n_tiles.saturating_add(count);
    }

    /// Bütün tile'lar geldiyse kareyi tamamlar: `refresh_frame_flags` bitleri
    /// için yuvaları günceller ve kare başlığını döndürür.
    ///
    /// `None`: bekleyen kare yok ya da tile'lar henüz tamamlanmadı.
    pub fn complete_frame(&mut self) -> Option<FrameHeader> {
        let header = self.current.as_ref()?;
        let expected = usize::from(header.tiling.cols) * usize::from(header.tiling.rows);
        if expected == 0 || self.n_tiles != expected {
            return None;
        }
        let header = self.current.take()?;
        self.n_tiles = 0;
        for slot in 0..NUM_REF_FRAMES {
            if header.refresh_frame_flags & (1 << slot) != 0 {
                self.refs[slot] = Some(header.clone());
                // dav1d `src/picture.c:205`: `p->showable = frame_hdr->showable_frame`.
                self.showable[slot] = header.showable_frame;
            }
        }
        Some(header)
    }

    /// Yuvanın gösterilebilirlik durumu (dav1d `c->refs[i].p.showable`).
    #[must_use]
    pub fn showable(&self, slot: usize) -> bool {
        self.showable.get(slot).copied().unwrap_or(false)
    }

    /// `show_existing_frame` işlemi: gösterilen referans karenin başlığını
    /// döndürür; kare **KEY** ise diğer 7 yuva da bu kareye yenilenir.
    ///
    /// Referans yuvası boşsa `None` (dav1d `src/obu.c` 1582–1584 hata yolu).
    pub fn show_existing(&mut self, slot: usize) -> Option<FrameHeader> {
        let displayed = self.refs.get(slot)?.as_ref()?.clone();
        if displayed.is_key() {
            for (index, reference) in self.refs.iter_mut().enumerate() {
                if index != slot {
                    *reference = Some(displayed.clone());
                }
            }
            // dav1d `src/obu.c:1621`: anahtar kare gösterildikten sonra yuvanın
            // `showable` değeri 0'lanır; diğer yuvalar da aynı resme referans
            // verdiği için (`dav1d_thread_picture_ref`) onlar da 0 olur.
            self.showable = [false; NUM_REF_FRAMES];
        }
        self.current = None;
        self.n_tiles = 0;
        Some(displayed)
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::HeaderReferenceState;
    use crate::bitstream::frame_header::{FrameHeader, FrameType, TilingInfo};

    fn header(frame_type: FrameType, refresh: u8, cols: u8, rows: u8) -> FrameHeader {
        let mut h = FrameHeader {
            frame_type,
            refresh_frame_flags: refresh,
            ..FrameHeader::default()
        };
        h.tiling = TilingInfo {
            cols,
            rows,
            ..TilingInfo::default()
        };
        h
    }

    #[test]
    fn refresh_bayraklari_yalniz_secili_yuvalari_gunceller() {
        let mut state = HeaderReferenceState::default();
        state.set_current(header(FrameType::Key, 0b0000_0101, 1, 1));
        state.add_tiles(1);
        assert_eq!(
            state.complete_frame(),
            Some(header(FrameType::Key, 0b0000_0101, 1, 1))
        );
        assert!(state.reference(0).is_some());
        assert!(state.reference(2).is_some());
        assert!(state.reference(1).is_none());
        assert_eq!(
            state.reference(0).expect("yuva 0").frame_type,
            FrameType::Key
        );
    }

    #[test]
    fn tile_sayisi_tamamlanmadan_kare_cikmaz() {
        let mut state = HeaderReferenceState::default();
        state.set_current(header(FrameType::Inter, 0xff, 2, 2));
        assert_eq!(state.complete_frame(), None);
        state.add_tiles(3);
        assert_eq!(state.complete_frame(), None);
        state.add_tiles(1);
        assert!(state.complete_frame().is_some());
        assert_eq!(
            state.reference(7).expect("yuva 7").frame_type,
            FrameType::Inter
        );
    }

    #[test]
    fn show_existing_anahtar_kare_butun_yuvalari_yeniler() {
        let mut state = HeaderReferenceState::default();
        state.set_current(header(FrameType::Key, 0b0000_0001, 1, 1));
        state.add_tiles(1);
        let key = state.complete_frame().expect("anahtar kare");
        assert!(state.reference(1).is_none());
        let shown = state.show_existing(0).expect("gösterim");
        assert_eq!(shown, key);
        for slot in 0..8 {
            assert_eq!(state.reference(slot), Some(&key), "yuva {slot}");
        }
    }

    /// dav1d `src/picture.c:205`: yuvaya kare çözülünce `showable_frame`
    /// yazılır; `show_existing` anahtar kareyi gösterince yuvanın `showable`
    /// değeri 0'lanır (`src/obu.c:1544-1548`, `src/obu.c:1619-1621`).
    #[test]
    fn show_existing_anahtar_kare_gosterilebilirligi_sifirlar() {
        let mut state = HeaderReferenceState::default();
        let mut key = header(FrameType::Key, 0xff, 1, 1);
        key.show_frame = false;
        key.showable_frame = true;
        state.set_current(key);
        state.add_tiles(1);
        assert!(state.complete_frame().is_some());
        for slot in 0..8 {
            assert!(state.showable(slot), "yuva {slot} gösterilebilir olmalı");
        }
        assert!(state.show_existing(0).is_some());
        for slot in 0..8 {
            assert!(
                !state.showable(slot),
                "anahtar kare gösterildikten sonra yuva {slot} gösterilemez"
            );
        }

        // Anahtar olmayan karede `showable` **korunur** (dav1d yalnız KEY
        // dalında 0'lar); ikinci gösterim de geçerlidir.
        let mut state = HeaderReferenceState::default();
        let mut inter = header(FrameType::Inter, 0b0000_0001, 1, 1);
        inter.show_frame = false;
        inter.showable_frame = true;
        state.set_current(inter);
        state.add_tiles(1);
        assert!(state.complete_frame().is_some());
        assert!(state.showable(0));
        assert!(state.show_existing(0).is_some());
        assert!(
            state.showable(0),
            "anahtar olmayan kare gösterildikten sonra da gösterilebilir kalır"
        );
    }

    #[test]
    fn show_existing_ara_kare_yalnizca_dondurur() {
        let mut state = HeaderReferenceState::default();
        state.set_current(header(FrameType::Inter, 0b0000_0010, 1, 1));
        state.add_tiles(1);
        let inter = state.complete_frame().expect("ara kare");
        let shown = state.show_existing(1).expect("gösterim");
        assert_eq!(shown, inter);
        assert!(state.reference(0).is_none());
        assert!(state.reference(1).is_some());
        assert!(state.show_existing(0).is_none());
    }

    /// Erken çözme hatası: yuvalar anlık görüntüden geri yüklenir; geç hata:
    /// yenilenen yuvalar boşalır (dav1d `src/decode.c:3446-3713`).
    #[test]
    fn yuva_goruntusu_geri_yukleme_ve_bosaltma() {
        let mut state = HeaderReferenceState::default();
        state.set_current(header(FrameType::Key, 0xFF, 1, 1));
        state.add_tiles(1);
        assert!(state.complete_frame().is_some());
        let before = state.slots();
        state.set_current(header(FrameType::Inter, 0b0000_0011, 1, 1));
        state.add_tiles(1);
        assert!(state.complete_frame().is_some());
        assert_eq!(
            state.reference(0).map(|h| h.frame_type),
            Some(FrameType::Inter)
        );
        state.restore_slots(&before, 0b0000_0011);
        assert_eq!(
            state.reference(0).map(|h| h.frame_type),
            Some(FrameType::Key)
        );
        assert_eq!(
            state.reference(1).map(|h| h.frame_type),
            Some(FrameType::Key)
        );
        state.clear_slots(0b0000_0001);
        assert!(state.reference(0).is_none());
        assert!(!state.showable(0));
        assert!(state.reference(1).is_some());
        // Tile birikimi sıfırlanır, bekleyen kare kalır.
        state.set_current(header(FrameType::Inter, 0, 2, 1));
        state.add_tiles(1);
        state.reset_tiles();
        assert_eq!(state.n_tiles(), 0);
        assert!(state.current().is_some());
    }
}
