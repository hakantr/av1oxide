//! Çerçeve (frame) ve çerçeve metaverisi (BLK-03).
//!
//! [`Frame`], decoder çıktısının **tek taşıyıcısıdır**: Y/U/V düzlemleri,
//! bit derinliği, chroma biçimi, boyutlar ve metaveri. BLK-03 gereği:
//!
//! - çıktı **YUV düzlemleri + metaveri**dir; RGB'ye çevirme veya renk yönetimi
//!   bu crate'in işi **değildir** (CEK-10),
//! - film grain durumu metaveride **isteğe bağlı** olarak taşınır,
//! - tüm boyut hesapları [`check`](Frame::new_bounded) aşamasında **tahsis
//!   öncesi** yapılır (BLK-05, BLK-07).
//!
//! ## Tahsis bütçesi (BLK-07)
//!
//! [`Frame::new_bounded`], kullanıcı tanımlı bayt sınırını **tüm düzlemlerin
//! toplamı** üzerinden, tahsis yapılmadan **önce** denetler. Aşımda
//! [`Error::BudgetExceeded`] döner; kısmi tahsis oluşmaz.
//!
//! > AV1'in **normatif** boyut sınırları ile uygulamanın **kullanıcı tanımlı**
//! > kaynak sınırları ayrı kavramlardır ve karıştırılmaz (BLK-07).

#![forbid(unsafe_code)]

use crate::error::{Error, Result, SizeStage};
use crate::frame::butce::Butce;
use crate::frame::{BitDepth, ChromaFormat, FramePlane, PlaneLayout, PlaneType};

/// Çerçeve metaverisi (BLK-03: kayıpsız iletilen yan bilgi).
///
/// Renk/metaveri bilgisi burada taşınır, **yorumlanmaz**: dönüşüm üst katmanın
/// işidir (CEK-10).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameMetadata {
    /// Sunum zaman damgası (mikrosaniye); kaynak pakette yoksa `None`.
    pub timestamp: Option<i64>,
    /// Film grain bu kareye uygulandı mı? (F2d'de doldurulur.)
    pub film_grain_applied: bool,
}

impl FrameMetadata {
    /// Zaman damgası olmayan, film grain'siz varsayılan metaveri.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            timestamp: None,
            film_grain_applied: false,
        }
    }
}

/// Çözülmüş (veya ayrılmış) bir kare: düzlemler + metaveri.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    planes: Vec<FramePlane>,
    chroma_format: ChromaFormat,
    bit_depth: BitDepth,
    width: usize,
    height: usize,
    metadata: FrameMetadata,
}

impl Frame {
    /// Kareyi varsayılan hizalama/padding ile ayırır (bütçe sınırsız).
    ///
    /// # Hatalar
    ///
    /// Bkz. [`Frame::new_bounded`].
    pub fn new(
        width: usize,
        height: usize,
        chroma_format: ChromaFormat,
        bit_depth: BitDepth,
    ) -> Result<Self> {
        Self::new_bounded(width, height, chroma_format, bit_depth, None)
    }

    /// Kareyi ayırır; `limit` verilmişse **tüm düzlemlerin toplamı** bu sınırı
    /// aşamaz (BLK-07).
    ///
    /// Boyut zinciri ve bütçe denetimi tahsisten **önce** yapılır; bu yüzden
    /// hata durumunda yarım ayrılmış kare oluşmaz.
    ///
    /// # Hatalar
    ///
    /// - [`Error::InvalidArgument`]: genişlik/yükseklik sıfır.
    /// - [`Error::SizeOverflow`]: chroma türetimi, satır baytı, stride, düzlem
    ///   alanı veya toplam tahsis `usize` aralığını aştı (BLK-05).
    /// - [`Error::BudgetExceeded`]: toplam tahsis `limit` baytı aştı (BLK-07).
    /// - [`Error::AllocationFailed`]: ayırıcı isteği karşılayamadı (P38).
    pub fn new_bounded(
        width: usize,
        height: usize,
        chroma_format: ChromaFormat,
        bit_depth: BitDepth,
        limit: Option<usize>,
    ) -> Result<Self> {
        let butce = Butce::yeni(limit, 0);
        Self::new_in(width, height, chroma_format, bit_depth, &butce)
    }

    fn compute_layouts_and_total(
        width: usize,
        height: usize,
        chroma_format: ChromaFormat,
        bit_depth: BitDepth,
    ) -> Result<(Vec<(PlaneType, PlaneLayout)>, usize)> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidArgument(
                "kare boyutu sıfır olamaz (genişlik ve yükseklik ≥ 1)",
            ));
        }

        let align = PlaneLayout::DEFAULT_ALIGN;
        let padding = PlaneLayout::DEFAULT_PADDING;

        let mut layouts: Vec<(PlaneType, PlaneLayout)> =
            Vec::with_capacity(chroma_format.num_planes());
        layouts.push((
            PlaneType::Y,
            PlaneLayout::compute(width, height, bit_depth, align, padding)?,
        ));
        if let Some((chroma_width, chroma_height)) =
            chroma_format.chroma_dimensions(width, height)?
        {
            for plane in [PlaneType::U, PlaneType::V] {
                layouts.push((
                    plane,
                    PlaneLayout::compute(chroma_width, chroma_height, bit_depth, align, padding)?,
                ));
            }
        }

        let mut total: usize = 0;
        for (_, layout) in &layouts {
            total = total
                .checked_add(layout.total())
                .ok_or(Error::SizeOverflow {
                    stage: SizeStage::TotalAllocation,
                    lhs: total,
                    rhs: layout.total(),
                })?;
        }
        Ok((layouts, total))
    }

    /// Verilen boyut ve biçimdeki karenin düzlem yerleşim bayt uzunlukları (`[Y, U, V]`)
    /// ve toplam tahsis baytı.
    pub(crate) fn plane_byte_lens_and_total(
        width: usize,
        height: usize,
        chroma_format: ChromaFormat,
        bit_depth: BitDepth,
    ) -> Result<([usize; 3], usize)> {
        let (layouts, total) =
            Self::compute_layouts_and_total(width, height, chroma_format, bit_depth)?;
        let mut lens = [0usize; 3];
        for (i, (_, layout)) in layouts.iter().enumerate() {
            lens[i] = layout.total();
        }
        Ok((lens, total))
    }

    /// [`Frame::new_bounded`]'ın defterli biçimi: kare toplamı tek parça olarak
    /// `butce`'ye yazılır (aşımda hiçbir düzlem ayrılmaz).
    pub(crate) fn new_in(
        width: usize,
        height: usize,
        chroma_format: ChromaFormat,
        bit_depth: BitDepth,
        butce: &Butce,
    ) -> Result<Self> {
        let (layouts, total) =
            Self::compute_layouts_and_total(width, height, chroma_format, bit_depth)?;
        butce.ayir(total)?;

        let planes = layouts
            .into_iter()
            .map(|(plane, layout)| FramePlane::from_layout(plane, layout))
            .collect::<Result<Vec<_>>>()
            .inspect_err(|_| butce.birak(total))?;

        Ok(Self {
            planes,
            chroma_format,
            bit_depth,
            width,
            height,
            metadata: FrameMetadata::new(),
        })
    }

    /// Referans olarak saklanmayan (`refresh_frame_flags == 0`) bir `PicBuf`
    /// üçlüsünü sıfır ek tahsisle yerinde (`in-place`) [`Frame`]'e dönüştürür (`4.A.1`).
    /// Bütçe muhasebesi çağıran tarafından yönetilir.
    pub(crate) fn from_owned_pic(
        width: usize,
        height: usize,
        chroma_format: ChromaFormat,
        bit_depth: BitDepth,
        mut pic_data: [Vec<u8>; 3],
        pic_stride: [usize; 3],
        allow_reuse: bool,
    ) -> Result<Self> {
        let (layouts, _) =
            Self::compute_layouts_and_total(width, height, chroma_format, bit_depth)?;
        let mut planes = Vec::with_capacity(layouts.len());
        for (i, (plane, layout)) in layouts.into_iter().enumerate() {
            let buf = core::mem::take(&mut pic_data[i]);
            planes.push(FramePlane::from_owned_pic_buf(
                plane,
                layout,
                buf,
                pic_stride[i],
                allow_reuse,
            )?);
        }
        Ok(Self {
            planes,
            chroma_format,
            bit_depth,
            width,
            height,
            metadata: FrameMetadata::new(),
        })
    }

    /// Yeniden kurulum düzlemlerinden (`pic_data`, `pic_stride`) ve varsa havuzdan
    /// geri dönüştürülmüş düzlem tamponlarından (`recycled`) çıktı [`Frame`]'ini
    /// tek geçişte (gereksiz tam-düzlem sıfırlaması yapmadan; büyük karelerde
    /// `WorkerPool` üzerinde düzlemler arası paralel) oluşturur (`F8b-1`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_from_pic_in(
        width: usize,
        height: usize,
        chroma_format: ChromaFormat,
        bit_depth: BitDepth,
        pic_data: &mut [Vec<u8>; 3],
        pic_stride: [usize; 3],
        mut recycled: [Vec<u8>; 3],
        butce: &Butce,
        pool: &crate::thread::WorkerPool,
    ) -> Result<Self> {
        let (layouts, total) =
            Self::compute_layouts_and_total(width, height, chroma_format, bit_depth)?;
        butce.ayir(total)?;

        let allow_reuse = !butce.has_limit();
        let mut planes = Vec::with_capacity(layouts.len());
        if pool.is_parallel() && layouts.len() > 1 && width.saturating_mul(height) >= 1920 * 1080 {
            struct PicCopyJob {
                idx: usize,
                plane: PlaneType,
                layout: PlaneLayout,
                recycled: Vec<u8>,
                src: Vec<u8>,
                src_stride: usize,
                out: Option<FramePlane>,
            }
            let mut jobs: Vec<PicCopyJob> = layouts
                .into_iter()
                .enumerate()
                .map(|(i, (plane, layout))| PicCopyJob {
                    idx: i,
                    plane,
                    layout,
                    recycled: core::mem::take(&mut recycled[i]),
                    src: core::mem::take(&mut pic_data[i]),
                    src_stride: pic_stride[i],
                    out: None,
                })
                .collect();
            let res = pool.execute_tasks(&mut jobs, move |_wid, _idx, job: &mut PicCopyJob| {
                let buf = core::mem::take(&mut job.recycled);
                let p = FramePlane::from_layout_with_pic(
                    job.plane,
                    job.layout,
                    buf,
                    &job.src,
                    job.src_stride,
                    allow_reuse,
                )?;
                job.out = Some(p);
                Ok(())
            });
            for job in &mut jobs {
                pic_data[job.idx] = core::mem::take(&mut job.src);
            }
            if let Err(err) = res {
                butce.birak(total);
                return Err(err);
            }
            for job in jobs {
                if let Some(p) = job.out {
                    planes.push(p);
                }
            }
        } else {
            for (i, (plane, layout)) in layouts.into_iter().enumerate() {
                let buf = core::mem::take(&mut recycled[i]);
                match FramePlane::from_layout_with_pic(
                    plane,
                    layout,
                    buf,
                    &pic_data[i],
                    pic_stride[i],
                    allow_reuse,
                ) {
                    Ok(p) => planes.push(p),
                    Err(err) => {
                        butce.birak(total);
                        return Err(err);
                    }
                }
            }
        }

        Ok(Self {
            planes,
            chroma_format,
            bit_depth,
            width,
            height,
            metadata: FrameMetadata::new(),
        })
    }

    /// Paylaşılan (`Arc<P>`) salt-okunur yeniden kurulum düzlemlerinden çıktı
    /// [`Frame`]'ini oluşturur (`show_existing_frame` ve boru hattı referans yolu).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_from_shared_pic_in<P>(
        width: usize,
        height: usize,
        chroma_format: ChromaFormat,
        bit_depth: BitDepth,
        holder: &std::sync::Arc<P>,
        get_plane: fn(&P, usize) -> (&[u8], usize),
        mut recycled: [Vec<u8>; 3],
        butce: &Butce,
        pool: &crate::thread::WorkerPool,
    ) -> Result<Self>
    where
        P: Send + Sync + 'static,
    {
        let (layouts, total) =
            Self::compute_layouts_and_total(width, height, chroma_format, bit_depth)?;
        butce.ayir(total)?;

        let allow_reuse = !butce.has_limit();
        let mut planes = Vec::with_capacity(layouts.len());
        if pool.is_parallel() && layouts.len() > 1 && width.saturating_mul(height) >= 1920 * 1080 {
            struct SharedCopyJob {
                idx: usize,
                plane: PlaneType,
                layout: PlaneLayout,
                recycled: Vec<u8>,
                out: Option<FramePlane>,
            }
            let mut jobs: Vec<SharedCopyJob> = layouts
                .into_iter()
                .enumerate()
                .map(|(i, (plane, layout))| SharedCopyJob {
                    idx: i,
                    plane,
                    layout,
                    recycled: core::mem::take(&mut recycled[i]),
                    out: None,
                })
                .collect();
            let holder_w = std::sync::Arc::clone(holder);
            let res = pool.execute_tasks(&mut jobs, move |_wid, _idx, job: &mut SharedCopyJob| {
                let buf = core::mem::take(&mut job.recycled);
                let (src, src_stride) = get_plane(&holder_w, job.idx);
                let p = FramePlane::from_layout_with_pic(
                    job.plane,
                    job.layout,
                    buf,
                    src,
                    src_stride,
                    allow_reuse,
                )?;
                job.out = Some(p);
                Ok(())
            });
            if let Err(err) = res {
                butce.birak(total);
                return Err(err);
            }
            for job in jobs {
                if let Some(p) = job.out {
                    planes.push(p);
                }
            }
        } else {
            for (i, (plane, layout)) in layouts.into_iter().enumerate() {
                let buf = core::mem::take(&mut recycled[i]);
                let (src, src_stride) = get_plane(holder, i);
                match FramePlane::from_layout_with_pic(
                    plane,
                    layout,
                    buf,
                    src,
                    src_stride,
                    allow_reuse,
                ) {
                    Ok(p) => planes.push(p),
                    Err(err) => {
                        butce.birak(total);
                        return Err(err);
                    }
                }
            }
        }

        Ok(Self {
            planes,
            chroma_format,
            bit_depth,
            width,
            height,
            metadata: FrameMetadata::new(),
        })
    }

    /// Doğrudan `PicBuf` kaynaklı Film Grain sentezi için çıktı [`Frame`]'i hazırlar:
    /// `write_mask[i] == true` olan düzlemlerde kaynak kopyalaması yapmadan yalnız
    /// satır/kuyruk dolgusunu sıfırlar; `write_mask[i] == false` olan düzlemleri
    /// `pic_data[i]`'den kopyalar (`4.A.1`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_for_grain_from_pic_in(
        width: usize,
        height: usize,
        chroma_format: ChromaFormat,
        bit_depth: BitDepth,
        pic_data: &[Vec<u8>; 3],
        pic_stride: [usize; 3],
        write_mask: [bool; 3],
        mut recycled: [Vec<u8>; 3],
        butce: &Butce,
    ) -> Result<Self> {
        let (layouts, total) =
            Self::compute_layouts_and_total(width, height, chroma_format, bit_depth)?;
        butce.ayir(total)?;

        let allow_reuse = !butce.has_limit();
        let mut planes = Vec::with_capacity(layouts.len());
        for (i, (plane, layout)) in layouts.into_iter().enumerate() {
            let buf = core::mem::take(&mut recycled[i]);
            let res = if write_mask[i] {
                FramePlane::from_layout_for_grain(plane, layout, buf, allow_reuse)
            } else {
                FramePlane::from_layout_with_pic(
                    plane,
                    layout,
                    buf,
                    &pic_data[i],
                    pic_stride[i],
                    allow_reuse,
                )
            };
            match res {
                Ok(p) => planes.push(p),
                Err(err) => {
                    butce.birak(total);
                    return Err(err);
                }
            }
        }

        Ok(Self {
            planes,
            chroma_format,
            bit_depth,
            width,
            height,
            metadata: FrameMetadata::new(),
        })
    }

    /// Luma genişliği (örnek).
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Luma yüksekliği (satır).
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Bit derinliği (BLK-04).
    #[must_use]
    pub const fn bit_depth(&self) -> BitDepth {
        self.bit_depth
    }

    /// Chroma biçimi (BLK-04).
    #[must_use]
    pub const fn chroma_format(&self) -> ChromaFormat {
        self.chroma_format
    }

    /// Düzlem sayısı: monochrome'da 1, diğerlerinde 3.
    #[must_use]
    pub fn plane_count(&self) -> usize {
        self.planes.len()
    }

    /// Düzlemler, Y-U-V sırasında.
    #[must_use]
    pub fn planes(&self) -> &[FramePlane] {
        &self.planes
    }

    /// Düzlemler, Y-U-V sırasında (değiştirilebilir, iç şerit işlemleri için).
    pub(crate) fn planes_mut(&mut self) -> &mut [FramePlane] {
        &mut self.planes
    }

    /// Tek düzlem; biçimde yoksa (monochrome'da U/V) `None`.
    #[must_use]
    pub fn plane(&self, plane: PlaneType) -> Option<&FramePlane> {
        self.planes.get(plane.index())
    }

    /// Tek düzlem (değiştirilebilir); biçimde yoksa `None`.
    #[must_use]
    pub fn plane_mut(&mut self, plane: PlaneType) -> Option<&mut FramePlane> {
        self.planes.get_mut(plane.index())
    }

    /// Metaveri (zaman damgası, film grain durumu).
    #[must_use]
    pub const fn metadata(&self) -> &FrameMetadata {
        &self.metadata
    }

    /// Metaveri (değiştirilebilir) — F2'de çözme hattı doldurur.
    #[must_use]
    pub const fn metadata_mut(&mut self) -> &mut FrameMetadata {
        &mut self.metadata
    }

    /// Tüm düzlemlerin tahsis ettiği toplam bayt.
    #[must_use]
    pub fn allocation_bytes(&self) -> usize {
        self.planes.iter().map(FramePlane::allocation_bytes).sum()
    }

    /// Karenin düşebilir kopyası (`clone` eşdeğeri): toplam tek parça olarak
    /// `butce`'ye yazılır, sonra düzlemler kopyalanır (P38).
    #[allow(dead_code)]
    pub(crate) fn try_clone(&self, butce: &Butce) -> Result<Self> {
        butce.ayir(self.allocation_bytes())?;
        Ok(Self {
            planes: self
                .planes
                .iter()
                .map(FramePlane::try_clone)
                .collect::<Result<Vec<_>>>()?,
            chroma_format: self.chroma_format,
            bit_depth: self.bit_depth,
            width: self.width,
            height: self.height,
            metadata: self.metadata,
        })
    }

    /// Havuzdan geri dönüştürülmüş düzlem tamponlarıyla (`recycled`) düşebilir
    /// kare kopyası (`F8b-1`).
    pub(crate) fn try_clone_with_bufs(
        &self,
        mut recycled: [Vec<u8>; 3],
        butce: &Butce,
    ) -> Result<Self> {
        let total = self.allocation_bytes();
        butce.ayir(total)?;
        let allow_reuse = !butce.has_limit();
        let mut planes = Vec::with_capacity(self.planes.len());
        for (i, plane) in self.planes.iter().enumerate() {
            let buf = recycled.get_mut(i).map(core::mem::take).unwrap_or_default();
            match plane.try_clone_with_buf(buf, allow_reuse) {
                Ok(p) => planes.push(p),
                Err(err) => {
                    butce.birak(total);
                    return Err(err);
                }
            }
        }
        Ok(Self {
            planes,
            chroma_format: self.chroma_format,
            bit_depth: self.bit_depth,
            width: self.width,
            height: self.height,
            metadata: self.metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Frame;
    use crate::error::{Error, SizeStage};
    use crate::frame::{BitDepth, ChromaFormat, PlaneType};

    #[test]
    fn dort_iki_sifir_uc_duzlem_ayirir() {
        let frame =
            Frame::new(1920, 1080, ChromaFormat::C420, BitDepth::Eight).expect("kare ayrılmalı");
        assert_eq!(frame.width(), 1920);
        assert_eq!(frame.height(), 1080);
        assert_eq!(frame.bit_depth(), BitDepth::Eight);
        assert_eq!(frame.chroma_format(), ChromaFormat::C420);
        assert_eq!(frame.plane_count(), 3);

        let y = frame.plane(PlaneType::Y).expect("Y düzlemi");
        let u = frame.plane(PlaneType::U).expect("U düzlemi");
        let v = frame.plane(PlaneType::V).expect("V düzlemi");
        assert_eq!((y.width(), y.height()), (1920, 1080));
        assert_eq!((u.width(), u.height()), (960, 540));
        assert_eq!((v.width(), v.height()), (960, 540));
        assert_eq!(
            frame.allocation_bytes(),
            y.allocation_bytes() + 2 * u.allocation_bytes()
        );
    }

    #[test]
    fn monochrome_tek_duzlem_tasir() {
        let frame =
            Frame::new(64, 64, ChromaFormat::Monochrome, BitDepth::Ten).expect("kare ayrılmalı");
        assert_eq!(frame.plane_count(), 1);
        assert!(frame.plane(PlaneType::U).is_none());
        assert!(frame.plane(PlaneType::V).is_none());
        assert!(frame.plane(PlaneType::Y).is_some());
    }

    #[test]
    fn dort_iki_iki_yalnizca_yatayda_kucultur() {
        let frame = Frame::new(7, 5, ChromaFormat::C422, BitDepth::Eight).expect("kare ayrılmalı");
        let u = frame.plane(PlaneType::U).expect("U düzlemi");
        assert_eq!((u.width(), u.height()), (4, 5));
    }

    #[test]
    fn butce_asimi_tahsis_yapmadan_reddedilir() {
        let total = Frame::new(64, 64, ChromaFormat::C444, BitDepth::Eight)
            .expect("kare ayrılmalı")
            .allocation_bytes();
        let err = Frame::new_bounded(64, 64, ChromaFormat::C444, BitDepth::Eight, Some(total - 1))
            .expect_err("bütçe aşılmalı");
        assert_eq!(
            err,
            Error::BudgetExceeded {
                requested: total,
                limit: total - 1,
            }
        );

        // Tam sınır kabul edilir (aşım yok).
        assert!(
            Frame::new_bounded(64, 64, ChromaFormat::C444, BitDepth::Eight, Some(total)).is_ok()
        );
    }

    #[test]
    fn sifir_boyut_ve_tasma_reddedilir() {
        assert!(matches!(
            Frame::new(0, 64, ChromaFormat::C420, BitDepth::Eight),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            Frame::new(64, usize::MAX, ChromaFormat::C444, BitDepth::Twelve),
            Err(Error::SizeOverflow {
                stage: SizeStage::PlaneArea | SizeStage::TotalAllocation,
                ..
            })
        ));
    }

    #[test]
    fn metaveri_tasima_ve_degistirme() {
        let mut frame =
            Frame::new(8, 8, ChromaFormat::Monochrome, BitDepth::Eight).expect("kare ayrılmalı");
        assert_eq!(frame.metadata().timestamp, None);
        assert!(!frame.metadata().film_grain_applied);

        frame.metadata_mut().timestamp = Some(1234);
        frame.metadata_mut().film_grain_applied = true;
        assert_eq!(frame.metadata().timestamp, Some(1234));
        assert!(frame.metadata().film_grain_applied);
    }
}
