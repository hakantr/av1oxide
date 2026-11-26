//! Düzlem (plane) tipi, yerleşim hesabı ve düzlem deposu.
//!
//! ## Boyut zinciri (BLK-05)
//!
//! [`PlaneLayout::compute`] BLK-05'in zorunlu kıldığı zinciri **checked**
//! aritmetikle yürütür:
//!
//! ```text
//! genişlik × bayt/örnek  →  satır baytı (row_bytes)
//! row_bytes → hizalanmış stride
//! stride × yükseklik     →  düzlem alanı (area)
//! area + padding         →  toplam tahsis (total)
//! ```
//!
//! Taşma, doyma veya geçersiz parametre **panik değil**
//! [`Error`](crate::error::Error) üretir; hatanın hangi adımda olduğu
//! [`SizeStage`](crate::error::SizeStage) ile bildirilir.
//!
//! ## Bellek yerleşimi ve SIMD over-read payı (BLK-06, SMD-05)
//!
//! ```text
//! satır 0 : [ row_bytes ][ stride − row_bytes = satır dolgusu ]
//! satır 1 : [ row_bytes ][ satır dolgusu ]
//! ...
//! satır n-1: [ row_bytes ][ satır dolgusu ]
//! sonra    : [ padding baytı — SIMD çekirdeklerinin taşan okuması için ]
//! ```
//!
//! `stride` **bayt** cinsindendir (örnek değil): 10/12-bit'te `stride ≈ 2 × genişlik`.
//! `padding` payı, çekirdeklerin son satırın ötesine birkaç bayt taşan okumasına
//! izin verir; bu payın **sözleşmede yazılı** olması BLK-06'nın şartıdır ve
//! sentinel testleriyle doğrulanır (S15.3, `tests/common/sentinel.rs`).

#![forbid(unsafe_code)]

use alloc::vec::Vec;

use crate::error::{Error, Result, SizeStage};
use crate::frame::BitDepth;

/// Çerçeve içindeki düzlem kimliği (BLK-03: YUV düzlemleri).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlaneType {
    /// Parlaklık düzlemi (her zaman vardır).
    Y,
    /// Chroma U (Cb) düzlemi.
    U,
    /// Chroma V (Cr) düzlemi.
    V,
}

impl PlaneType {
    /// Düzlemlerin depodaki sabit sırası: Y=0, U=1, V=2.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Y => 0,
            Self::U => 1,
            Self::V => 2,
        }
    }

    /// Chroma düzlemi mi? (U veya V)
    #[must_use]
    pub const fn is_chroma(self) -> bool {
        !matches!(self, Self::Y)
    }
}

/// Bir düzlemin **hesaplanmış** yerleşimi (tahsis yapılmadan).
///
/// Boyut sözleşmesini test etmenin ve bütçe kontrolünü tahsisten **önce**
/// yapmanın yoludur (BLK-05, BLK-07).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneLayout {
    row_bytes: usize,
    stride: usize,
    area: usize,
    total: usize,
    width: usize,
    height: usize,
    bit_depth: BitDepth,
    align: usize,
    padding: usize,
}

impl PlaneLayout {
    /// Varsayılan hizalama: 64 bayt.
    ///
    /// F1.5 varsayılanıdır; kesin değer S15.4 pilotu ve S15.6 karar kaydı ile
    /// ölçüm üzerinden dondurulur (PRF/SMD).
    pub const DEFAULT_ALIGN: usize = 64;

    /// Varsayılan over-read payı: 64 bayt (BLK-06).
    ///
    /// En geniş desteklenen SIMD erişiminin (AVX-512'de 64 bayt) bir satır
    /// ötesine taşan okumasını karşılar.
    pub const DEFAULT_PADDING: usize = 64;

    /// BLK-05 zincirini varsayılan hizalama ve padding ile yürütür.
    ///
    /// # Hatalar
    ///
    /// - [`Error::InvalidArgument`]: genişlik veya yükseklik sıfır, ya da
    ///   `align` sıfır / 2'nin kuvveti değil.
    /// - [`Error::SizeOverflow`]: zincirin ilgili adımı `usize` aralığını aştı.
    pub fn compute(
        width: usize,
        height: usize,
        bit_depth: BitDepth,
        align: usize,
        padding: usize,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidArgument(
                "düzlem boyutu sıfır olamaz (genişlik ve yükseklik ≥ 1)",
            ));
        }
        if align == 0 || !align.is_power_of_two() {
            return Err(Error::InvalidArgument(
                "hizalama sıfırdan farklı ve 2'nin kuvveti olmalı",
            ));
        }

        let bytes_per_sample = bit_depth.bytes_per_sample();
        let row_bytes = width
            .checked_mul(bytes_per_sample)
            .ok_or(Error::SizeOverflow {
                stage: SizeStage::RowBytes,
                lhs: width,
                rhs: bytes_per_sample,
            })?;

        // Hizalama: `align` 2'nin kuvveti olduğu için bit maskesi yeterlidir.
        let stride = row_bytes
            .checked_add(align - 1)
            .ok_or(Error::SizeOverflow {
                stage: SizeStage::Stride,
                lhs: row_bytes,
                rhs: align - 1,
            })?
            & !(align - 1);

        let area = stride.checked_mul(height).ok_or(Error::SizeOverflow {
            stage: SizeStage::PlaneArea,
            lhs: stride,
            rhs: height,
        })?;

        let total = area.checked_add(padding).ok_or(Error::SizeOverflow {
            stage: SizeStage::TotalAllocation,
            lhs: area,
            rhs: padding,
        })?;

        Ok(Self {
            row_bytes,
            stride,
            area,
            total,
            width,
            height,
            bit_depth,
            align,
            padding,
        })
    }

    /// Bir satırın **anlamlı** bayt sayısı: `genişlik × bayt/örnek`.
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Satırlar arası bayt adımı (hizalanmış, `≥ row_bytes`).
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Satır dolgusu: `stride − row_bytes`.
    #[must_use]
    pub const fn row_padding(&self) -> usize {
        self.stride - self.row_bytes
    }

    /// `stride × yükseklik` — satırların toplam baytı.
    #[must_use]
    pub const fn area(&self) -> usize {
        self.area
    }

    /// Tahsis edilecek toplam bayt: `area + padding`.
    #[must_use]
    pub const fn total(&self) -> usize {
        self.total
    }

    /// Düzlem genişliği (örnek).
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Düzlem yüksekliği (satır).
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Bit derinliği.
    #[must_use]
    pub const fn bit_depth(&self) -> BitDepth {
        self.bit_depth
    }

    /// Kullanılan hizalama (bayt).
    #[must_use]
    pub const fn align(&self) -> usize {
        self.align
    }

    /// Over-read payı (bayt).
    #[must_use]
    pub const fn padding(&self) -> usize {
        self.padding
    }
}

/// Tek bir görüntü düzlemi: hizalı, padding'li, sahiplenilmiş bayt deposu.
///
/// Depo **bayt** (`Vec<u8>`) cinsindendir. 10/12-bit örnekler `u16` hücrelerde
/// **küçük-endian** olarak durur; `u16` depolama 16-bit görüntü anlamına gelmez
/// (BLK-04, bkz. [`BitDepth`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FramePlane {
    plane: PlaneType,
    layout: PlaneLayout,
    data: Vec<u8>,
}

impl FramePlane {
    /// Düzlemi varsayılan hizalama ve over-read payıyla ayırır.
    ///
    /// # Hatalar
    ///
    /// Bkz. [`PlaneLayout::compute`]; ayırıcı başarısız olursa
    /// [`Error::AllocationFailed`](crate::error::Error::AllocationFailed)
    /// (düşebilir ayırma, P38). Bütçe denetimi için
    /// [`Frame::new_bounded`](crate::frame::Frame::new_bounded) kullanılır.
    pub fn new(plane: PlaneType, width: usize, height: usize, bit_depth: BitDepth) -> Result<Self> {
        Self::new_aligned(
            plane,
            width,
            height,
            bit_depth,
            PlaneLayout::DEFAULT_ALIGN,
            PlaneLayout::DEFAULT_PADDING,
        )
    }

    /// Düzlemi **açık** hizalama ve over-read payıyla ayırır.
    ///
    /// # Hatalar
    ///
    /// Bkz. [`PlaneLayout::compute`].
    pub fn new_aligned(
        plane: PlaneType,
        width: usize,
        height: usize,
        bit_depth: BitDepth,
        align: usize,
        padding: usize,
    ) -> Result<Self> {
        let layout = PlaneLayout::compute(width, height, bit_depth, align, padding)?;
        Self::from_layout(plane, layout)
    }

    /// Hesaplanmış yerleşimden düzlem üretir (tahsis burada yapılır; düşebilir,
    /// sıfır dolu). Bütçe denetimi çağıranındır.
    pub(crate) fn from_layout(plane: PlaneType, layout: PlaneLayout) -> Result<Self> {
        Ok(Self {
            plane,
            layout,
            data: crate::frame::butce::dolu_vec(layout.total(), 0u8)?,
        })
    }

    /// Havuzdan geri dönüştürülmüş `buf` kapasitesi `[total, 2*total]` aralığındaysa
    /// yeniden tahsis yapmadan `src` (`src_stride`) satırlarını kopyalayıp satır/kuyruk
    /// dolgusunu sıfırlayarak düzlem üretir (`F8b-1`). Bütçe denetimi çağıranındır.
    pub(crate) fn from_layout_with_pic(
        plane: PlaneType,
        layout: PlaneLayout,
        mut buf: Vec<u8>,
        src: &[u8],
        src_stride: usize,
        allow_reuse: bool,
    ) -> Result<Self> {
        let total = layout.total();
        let row_bytes = layout.row_bytes();
        let dst_stride = layout.stride();
        let height = layout.height();
        let data = if allow_reuse
            && total > 0
            && buf.capacity() >= total
            && buf.capacity() <= total.saturating_mul(2)
        {
            if buf.len() != total {
                buf.resize(total, 0);
            }
            for y in 0..height {
                let dst_off = y * dst_stride;
                let src_off = y * src_stride;
                buf[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&src[src_off..src_off + row_bytes]);
                if dst_stride > row_bytes {
                    buf[dst_off + row_bytes..dst_off + dst_stride].fill(0);
                }
            }
            let area = layout.area();
            if total > area {
                buf[area..total].fill(0);
            }
            buf
        } else {
            drop(buf);
            let mut v = crate::frame::butce::dolu_vec(total, 0u8)?;
            for y in 0..height {
                let dst_off = y * dst_stride;
                let src_off = y * src_stride;
                v[dst_off..dst_off + row_bytes].copy_from_slice(&src[src_off..src_off + row_bytes]);
            }
            v
        };
        Ok(Self {
            plane,
            layout,
            data,
        })
    }

    /// Görünür alanı tamamen Film Grain çekirdeklerince (`fgy`/`fguv`) yazılacak
    /// düzlem için tampon hazırlar: havuzdan dönen `buf` kapasitesi uygunsa yalnız
    /// satır/kuyruk dolgusunu sıfırlar (tam düzlem kopyası/sıfırlaması yapmaz).
    pub(crate) fn from_layout_for_grain(
        plane: PlaneType,
        layout: PlaneLayout,
        mut buf: Vec<u8>,
        allow_reuse: bool,
    ) -> Result<Self> {
        let total = layout.total();
        let row_bytes = layout.row_bytes();
        let dst_stride = layout.stride();
        let height = layout.height();
        let data = if allow_reuse
            && total > 0
            && buf.capacity() >= total
            && buf.capacity() <= total.saturating_mul(2)
        {
            if buf.len() != total {
                buf.resize(total, 0);
            }
            if dst_stride > row_bytes {
                for y in 0..height {
                    let dst_off = y * dst_stride;
                    buf[dst_off + row_bytes..dst_off + dst_stride].fill(0);
                }
            }
            let area = layout.area();
            if total > area {
                buf[area..total].fill(0);
            }
            buf
        } else {
            drop(buf);
            crate::frame::butce::dolu_vec(total, 0u8)?
        };
        Ok(Self {
            plane,
            layout,
            data,
        })
    }

    /// Referans olarak saklanmayan (`refresh_frame_flags == 0`) bir `PicBuf`
    /// düzlem vektörünü sıfır ek tahsisle yerinde (`in-place`) [`FramePlane`]
    /// düzenine sıkıştırır (`4.A.1`).
    pub(crate) fn from_owned_pic_buf(
        plane: PlaneType,
        layout: PlaneLayout,
        mut buf: Vec<u8>,
        src_stride: usize,
        allow_reuse: bool,
    ) -> Result<Self> {
        let total = layout.total();
        let row_bytes = layout.row_bytes();
        let dst_stride = layout.stride();
        let height = layout.height();
        if src_stride >= dst_stride && buf.len() >= total {
            if src_stride > dst_stride {
                for y in 0..height {
                    let dst_off = y * dst_stride;
                    let src_off = y * src_stride;
                    if dst_off != src_off {
                        buf.copy_within(src_off..src_off + row_bytes, dst_off);
                    }
                    if dst_stride > row_bytes {
                        buf[dst_off + row_bytes..dst_off + dst_stride].fill(0);
                    }
                }
            } else if dst_stride > row_bytes {
                for y in 0..height {
                    let dst_off = y * dst_stride;
                    buf[dst_off + row_bytes..dst_off + dst_stride].fill(0);
                }
            }
            let area = layout.area();
            if total > area {
                buf[area..total].fill(0);
            }
            buf.truncate(total);
            if !allow_reuse {
                buf.shrink_to_fit();
            }
            return Ok(Self {
                plane,
                layout,
                data: buf,
            });
        }
        Self::from_layout_with_pic(plane, layout, Vec::new(), &buf, src_stride, false)
    }

    /// Düzlemin düşebilir kopyası (`clone` eşdeğeri; bütçe denetimi
    /// çağıranındır).
    #[allow(dead_code)]
    pub(crate) fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            plane: self.plane,
            layout: self.layout,
            data: crate::frame::butce::kopya_vec(&self.data)?,
        })
    }

    /// Havuzdan geri dönüştürülmüş `buf` kapasitesi `[n, 2*n]` aralığındaysa yeniden
    /// tahsis yapmadan düzlemi kopyalar (`F8b-1`). Bütçe denetimi çağıranındır.
    #[cfg_attr(not(feature = "std"), allow(dead_code))]
    pub(crate) fn try_clone_with_buf(&self, mut buf: Vec<u8>, allow_reuse: bool) -> Result<Self> {
        let n = self.data.len();
        let data =
            if allow_reuse && n > 0 && buf.capacity() >= n && buf.capacity() <= n.saturating_mul(2)
            {
                if buf.len() == n {
                    buf.copy_from_slice(&self.data);
                } else {
                    buf.clear();
                    buf.extend_from_slice(&self.data);
                }
                buf
            } else {
                drop(buf);
                crate::frame::butce::kopya_vec(&self.data)?
            };
        Ok(Self {
            plane: self.plane,
            layout: self.layout,
            data,
        })
    }

    /// Bu düzlemin kimliği (Y/U/V).
    #[must_use]
    pub const fn plane(&self) -> PlaneType {
        self.plane
    }

    /// Hesaplanmış yerleşim (stride, padding, toplam tahsis).
    #[must_use]
    pub const fn layout(&self) -> &PlaneLayout {
        &self.layout
    }

    /// Düzlem genişliği (örnek).
    #[must_use]
    pub const fn width(&self) -> usize {
        self.layout.width()
    }

    /// Düzlem yüksekliği (satır).
    #[must_use]
    pub const fn height(&self) -> usize {
        self.layout.height()
    }

    /// Bit derinliği.
    #[must_use]
    pub const fn bit_depth(&self) -> BitDepth {
        self.layout.bit_depth()
    }

    /// Satır adımı (bayt).
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.layout.stride()
    }

    /// Örnek sayısı: `genişlik × yükseklik` (tahsis kararı bu değere dayanır).
    #[must_use]
    pub const fn sample_count(&self) -> usize {
        self.layout.width() * self.layout.height()
    }

    /// Tahsis edilen toplam bayt (`stride × yükseklik + padding`).
    #[must_use]
    pub fn allocation_bytes(&self) -> usize {
        self.data.len()
    }

    /// Ham depo: `stride × yükseklik` satır baytı + son padding.
    ///
    /// Satırlara [`FramePlane::row`] ile erişmek tercih edilir; bu görünüm
    /// çekirdeklerin (SMD-05) taşan okumasına izin veren tam aralıktır.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Ham depo (değiştirilebilir).
    #[must_use]
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Tek bir satırın **anlamlı** baytları (`row_bytes`), satır dolgusu hariç.
    ///
    /// `y` aralık dışındaysa `None` döner (panik yok).
    #[must_use]
    pub fn row(&self, y: usize) -> Option<&[u8]> {
        if y >= self.height() {
            return None;
        }
        let start = y * self.stride();
        Some(&self.data[start..start + self.layout.row_bytes()])
    }

    /// Tek bir satırın anlamlı baytları (değiştirilebilir).
    ///
    /// `y` aralık dışındaysa `None` döner (panik yok).
    #[must_use]
    pub fn row_mut(&mut self, y: usize) -> Option<&mut [u8]> {
        let row_bytes = self.layout.row_bytes();
        let stride = self.stride();
        if y >= self.height() {
            return None;
        }
        let start = y * stride;
        Some(&mut self.data[start..start + row_bytes])
    }
}

#[cfg(test)]
mod tests {
    use super::{FramePlane, PlaneLayout, PlaneType};
    use crate::error::{Error, SizeStage};
    use crate::frame::BitDepth;

    #[test]
    fn duzlem_kimligi_sirasi() {
        assert_eq!(PlaneType::Y.index(), 0);
        assert_eq!(PlaneType::U.index(), 1);
        assert_eq!(PlaneType::V.index(), 2);
        assert!(!PlaneType::Y.is_chroma());
        assert!(PlaneType::U.is_chroma());
        assert!(PlaneType::V.is_chroma());
    }

    #[test]
    fn stride_hizalanir_ve_bayt_cinsindendir() {
        let layout =
            PlaneLayout::compute(100, 8, BitDepth::Eight, 64, 64).expect("yerleşim hesaplanmalı");
        assert_eq!(layout.row_bytes(), 100);
        assert_eq!(layout.stride(), 128, "100 bayt → 64'ün katı 128");
        assert_eq!(layout.row_padding(), 28);
        assert_eq!(layout.area(), 128 * 8);
        assert_eq!(layout.total(), 128 * 8 + 64);

        // 10-bit: stride **bayt** cinsindendir → 2 bayt/örnek.
        let ten_bit =
            PlaneLayout::compute(100, 8, BitDepth::Ten, 64, 64).expect("yerleşim hesaplanmalı");
        assert_eq!(ten_bit.row_bytes(), 200);
        assert_eq!(ten_bit.stride(), 256);
    }

    #[test]
    fn hizali_yerlesimde_dolgu_yok() {
        let layout =
            PlaneLayout::compute(64, 4, BitDepth::Eight, 64, 16).expect("yerleşim hesaplanmalı");
        assert_eq!(layout.stride(), 64);
        assert_eq!(layout.row_padding(), 0);
        assert_eq!(layout.total(), 64 * 4 + 16);
    }

    #[test]
    fn gecersiz_parametreler_reddedilir() {
        assert!(matches!(
            PlaneLayout::compute(0, 4, BitDepth::Eight, 64, 0),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            PlaneLayout::compute(4, 0, BitDepth::Eight, 64, 0),
            Err(Error::InvalidArgument(_))
        ));
        for bad_align in [0usize, 3, 6, 100] {
            assert!(
                matches!(
                    PlaneLayout::compute(4, 4, BitDepth::Eight, bad_align, 0),
                    Err(Error::InvalidArgument(_))
                ),
                "hizalama {bad_align} reddedilmeli"
            );
        }
    }

    #[test]
    fn tasma_her_asamada_yakalanir_ve_panik_yok() {
        // RowBytes: usize::MAX × 2
        assert_eq!(
            PlaneLayout::compute(usize::MAX, 1, BitDepth::Ten, 64, 0),
            Err(Error::SizeOverflow {
                stage: SizeStage::RowBytes,
                lhs: usize::MAX,
                rhs: 2,
            })
        );
        // Stride: row_bytes + (align - 1)
        assert_eq!(
            PlaneLayout::compute(usize::MAX, 1, BitDepth::Eight, 64, 0),
            Err(Error::SizeOverflow {
                stage: SizeStage::Stride,
                lhs: usize::MAX,
                rhs: 63,
            })
        );
        // PlaneArea: stride × height
        assert_eq!(
            PlaneLayout::compute(64, usize::MAX, BitDepth::Eight, 64, 0),
            Err(Error::SizeOverflow {
                stage: SizeStage::PlaneArea,
                lhs: 64,
                rhs: usize::MAX,
            })
        );
        // TotalAllocation: area + padding
        assert_eq!(
            PlaneLayout::compute(64, 1, BitDepth::Eight, 64, usize::MAX),
            Err(Error::SizeOverflow {
                stage: SizeStage::TotalAllocation,
                lhs: 64,
                rhs: usize::MAX,
            })
        );
    }

    #[test]
    fn duzlem_deposu_dogru_boyutta_ayrilir() {
        let plane = FramePlane::new(PlaneType::Y, 100, 8, BitDepth::Ten).expect("düzlem ayrılmalı");
        assert_eq!(plane.width(), 100);
        assert_eq!(plane.height(), 8);
        assert_eq!(plane.stride(), 256);
        assert_eq!(plane.sample_count(), 800);
        assert_eq!(plane.allocation_bytes(), 256 * 8 + 64);
        assert_eq!(plane.data().len(), plane.allocation_bytes());
        assert!(plane.data().iter().all(|&b| b == 0), "depo sıfırlanmalı");
    }

    #[test]
    fn satir_erisimi_sinirlari_korur() {
        let mut plane =
            FramePlane::new_aligned(PlaneType::U, 4, 2, BitDepth::Eight, 8, 8).expect("ayrılmalı");
        assert_eq!(plane.stride(), 8);
        assert_eq!(plane.row(0).map(<[u8]>::len), Some(4));
        assert_eq!(plane.row(1).map(<[u8]>::len), Some(4));
        assert_eq!(plane.row(2), None, "aralık dışı satır panik üretmemeli");

        if let Some(row) = plane.row_mut(1) {
            row.fill(9);
        }
        assert_eq!(plane.row(1), Some(&[9u8; 4][..]));
        assert_eq!(plane.row(0), Some(&[0u8; 4][..]));
    }

    #[test]
    fn tasmada_tahsis_yapilmaz() {
        assert!(FramePlane::new(PlaneType::Y, usize::MAX, 2, BitDepth::Twelve).is_err());
    }
}
