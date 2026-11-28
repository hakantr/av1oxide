//! IVF (`DKIF` / `AV01`) kapsayıcı ayrıştırıcısı (`#![forbid(unsafe_code)]`).
//!
//! - [`IvfParser`]: Bellek içi `&[u8]` dilimleri üzerinde sıfır kopyalı kare çıkarımı
//!   (`no_std` + `alloc` uyumlu).
//! - [`IvfReader`]: `std` özelliği etkinken [`std::io::Read`] akışlarından yeniden
//!   kullanılan tamponla ek tahsissiz kare okuması.

#![forbid(unsafe_code)]

use crate::error::{Error, Result};

/// Standart IVF dosya başlığı boyutu (`32` bayt).
pub const IVF_HEADER_MIN_LEN: usize = 32;
/// Standart IVF kare başlığı boyutu (`12` bayt: `u32` boyut + `i64` PTS).
pub const IVF_FRAME_HEADER_LEN: usize = 12;

/// [`IvfReader::read_packet_into`] tamponunun, mevcut kapasite yetmediğinde ilk büyüme adımı
/// (`64 KiB`); sonraki adımlar okunan veriyle ikiye katlanır.
#[cfg(feature = "std")]
pub(crate) const IVF_READ_CHUNK: usize = 64 * 1024;

/// Ayrıştırılmış IVF dosya başlığı (`32` bayt).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IvfHeader {
    /// IVF sürüm numarası (standart dosyalarda `0`).
    pub version: u16,
    /// Dosya başlığı uzunluğu (bayt; en az `32`).
    pub header_len: u16,
    /// Dört karakterlik kodek kimliği (AV1 akışları için `*b"AV01"`).
    pub fourcc: [u8; 4],
    /// Görüntü genişliği (piksel).
    pub width: u16,
    /// Görüntü yüksekliği (piksel).
    pub height: u16,
    /// Zaman tabanı paydası (`timebase_den` / `rate`).
    pub timebase_den: u32,
    /// Zaman tabanı payı (`timebase_num` / `scale`).
    pub timebase_num: u32,
    /// Başlıkta bildirilen kare sayısı (`0` olabilir).
    pub frame_count: u32,
}

impl IvfHeader {
    /// İlk `32` bayttan [`IvfHeader`] ayrıştırır ve `DKIF` / `header_len >= 32` / `AV01`
    /// sözleşmesini doğrular.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < IVF_HEADER_MIN_LEN {
            return Err(Error::InvalidData("kesik IVF dosya başlığı (< 32 bayt)"));
        }
        if &data[0..4] != b"DKIF" {
            return Err(Error::InvalidData("IVF dosya imzası 'DKIF' değil"));
        }
        let version = u16::from_le_bytes([data[4], data[5]]);
        let header_len = u16::from_le_bytes([data[6], data[7]]);
        if usize::from(header_len) < IVF_HEADER_MIN_LEN {
            return Err(Error::InvalidData("IVF başlık uzunluğu 32 bayttan küçük"));
        }
        let fourcc = [data[8], data[9], data[10], data[11]];
        if fourcc != *b"AV01" {
            return Err(Error::InvalidData(
                "IVF kodek kimliği desteklenmiyor ('AV01' bekleniyor)",
            ));
        }
        let width = u16::from_le_bytes([data[12], data[13]]);
        let height = u16::from_le_bytes([data[14], data[15]]);
        let timebase_den = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        let timebase_num = u32::from_le_bytes([data[20], data[21], data[22], data[23]]);
        let frame_count = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);

        Ok(Self {
            version,
            header_len,
            fourcc,
            width,
            height,
            timebase_den,
            timebase_num,
            frame_count,
        })
    }
}

/// IVF kapsayıcısından çıkarılmış sıfır kopyalı tekil AV1 paketi.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IvfPacket<'a> {
    /// Sunum zaman damgası (`64-bit` küçük-endian tamsayı).
    pub pts: i64,
    /// Paket yük baytları (doğrudan [`crate::Packet::with_timestamp`] ile kullanılabilir).
    pub data: &'a [u8],
}

/// Bellek içi `&[u8]` diliminden sıfır kopyalı IVF paket ayrıştırıcısı.
#[derive(Clone, Debug)]
pub struct IvfParser<'a> {
    data: &'a [u8],
    offset: usize,
    header: IvfHeader,
    finished: bool,
}

impl<'a> IvfParser<'a> {
    /// Verilen bayt diliminin IVF başlığını doğrular ve yeni bir [`IvfParser`] kurar.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let header = IvfHeader::parse(data)?;
        let header_len = usize::from(header.header_len);
        if header_len > data.len() {
            return Err(Error::InvalidData(
                "IVF başlık uzunluğu girdi boyutunu aşıyor",
            ));
        }
        Ok(Self {
            data,
            offset: header_len,
            header,
            finished: false,
        })
    }

    /// Ayrıştırılmış IVF dosya başlığını döndürür.
    #[must_use]
    pub const fn header(&self) -> &IvfHeader {
        &self.header
    }

    /// Sıradaki IVF paketini sıfır kopyayla döndürür; akış sonuna ulaşıldıysa `Ok(None)` döner.
    ///
    /// Kesik kare başlığı (`< 12` bayt) veya kesik kare yükü tespit edilirse panik
    /// üretmeden [`Error::InvalidData`] döndürür.
    pub fn next_packet(&mut self) -> Result<Option<IvfPacket<'a>>> {
        if self.finished || self.offset == self.data.len() {
            self.finished = true;
            return Ok(None);
        }
        let rem = self.data.len() - self.offset;
        if rem < IVF_FRAME_HEADER_LEN {
            self.finished = true;
            return Err(Error::InvalidData("kesik IVF kare başlığı (< 12 bayt)"));
        }
        let hdr = &self.data[self.offset..self.offset + IVF_FRAME_HEADER_LEN];
        let frame_size = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as usize;
        let pts = i64::from_le_bytes([
            hdr[4], hdr[5], hdr[6], hdr[7], hdr[8], hdr[9], hdr[10], hdr[11],
        ]);
        let payload_start = self.offset + IVF_FRAME_HEADER_LEN;
        let payload_end = payload_start.checked_add(frame_size).ok_or_else(|| {
            self.finished = true;
            Error::InvalidData("IVF kare boyutu taşması")
        })?;
        if payload_end > self.data.len() {
            self.finished = true;
            return Err(Error::InvalidData(
                "kesik IVF kare yükü (bildirilen boyut girdi sonunu aşıyor)",
            ));
        }
        let payload = &self.data[payload_start..payload_end];
        self.offset = payload_end;
        Ok(Some(IvfPacket { pts, data: payload }))
    }
}

impl<'a> Iterator for IvfParser<'a> {
    type Item = Result<IvfPacket<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_packet().transpose()
    }
}

/// Akış (`std::io::Read`) tabanlı IVF okuyucusu (`std` özelliği etkinken kullanılabilir).
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct IvfReader<R: std::io::Read> {
    reader: R,
    header: IvfHeader,
    max_frame_bytes: usize,
    pending_skip: usize,
    hdr_buf: [u8; IVF_FRAME_HEADER_LEN],
    hdr_filled: usize,
}

#[cfg(feature = "std")]
impl<R: std::io::Read> IvfReader<R> {
    /// Varsayılan tekil IVF kare boyutu üst sınırı (`256 MiB`).
    pub const DEFAULT_MAX_FRAME_BYTES: usize = 256 * 1024 * 1024;

    /// Akış başından IVF dosya başlığını okur ve doğrular.
    pub fn new(reader: R) -> Result<Self> {
        Self::with_max_frame_bytes(reader, Self::DEFAULT_MAX_FRAME_BYTES)
    }

    /// Tekil kare boyutu üst sınırını belirterek yeni bir [`IvfReader`] oluşturur.
    pub fn with_max_frame_bytes(mut reader: R, max_frame_bytes: usize) -> Result<Self> {
        let mut hdr_buf = [0u8; IVF_HEADER_MIN_LEN];
        read_exact_mapped(&mut reader, &mut hdr_buf, "kesik IVF dosya başlığı")?;
        let header = IvfHeader::parse(&hdr_buf)?;
        let extra = usize::from(header.header_len) - IVF_HEADER_MIN_LEN;
        if extra > 0 {
            let mut skip_buf = [0u8; 64];
            let mut left = extra;
            while left > 0 {
                let n = left.min(skip_buf.len());
                read_exact_mapped(
                    &mut reader,
                    &mut skip_buf[..n],
                    "kesik IVF genişletilmiş başlık",
                )?;
                left -= n;
            }
        }
        Ok(Self {
            reader,
            header,
            max_frame_bytes,
            pending_skip: 0,
            hdr_buf: [0u8; IVF_FRAME_HEADER_LEN],
            hdr_filled: 0,
        })
    }

    /// Ayrıştırılmış IVF dosya başlığını döndürür.
    #[must_use]
    pub const fn header(&self) -> &IvfHeader {
        &self.header
    }

    /// Sıradaki IVF karesini `buf` içine okur ve PTS değerini döndürür.
    ///
    /// `buf` kapasitesi zaten yeterli olduğunda ek yığın tahsisi yapılmaz. Yetmediğinde
    /// tampon beyan edilen kare boyutuna bir kerede değil, gelen veriyle orantılı
    /// (`64 KiB`'tan başlayıp ikiye katlanarak) büyür; böylece kesik bir
    /// akış büyük bir kare boyutu beyan ederek okunmamış veri için bellek ayırtamaz.
    /// Akış temiz biçimde bittiğinde `Ok(None)` döndürülür.
    pub fn read_packet_into(&mut self, buf: &mut alloc::vec::Vec<u8>) -> Result<Option<i64>> {
        if self.pending_skip > 0 && !drain_pending_skip(&mut self.reader, &mut self.pending_skip) {
            buf.clear();
            return Err(Error::InvalidData("IVF atlanan kare yükü okunamadı"));
        }

        while self.hdr_filled < IVF_FRAME_HEADER_LEN {
            match self.reader.read(&mut self.hdr_buf[self.hdr_filled..]) {
                Ok(0) => {
                    if self.hdr_filled == 0 {
                        buf.clear();
                        return Ok(None);
                    }
                    self.hdr_filled = 0;
                    buf.clear();
                    return Err(Error::InvalidData("kesik IVF kare başlığı (< 12 bayt)"));
                }
                Ok(n) => self.hdr_filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => {
                    buf.clear();
                    return Err(Error::InvalidData("IVF kare başlığı okunamadı"));
                }
            }
        }

        let frame_hdr = self.hdr_buf;
        self.hdr_filled = 0;
        let frame_size =
            u32::from_le_bytes([frame_hdr[0], frame_hdr[1], frame_hdr[2], frame_hdr[3]]) as usize;
        let pts = i64::from_le_bytes([
            frame_hdr[4],
            frame_hdr[5],
            frame_hdr[6],
            frame_hdr[7],
            frame_hdr[8],
            frame_hdr[9],
            frame_hdr[10],
            frame_hdr[11],
        ]);
        if frame_size > self.max_frame_bytes {
            buf.clear();
            self.pending_skip = frame_size;
            let _ = drain_pending_skip(&mut self.reader, &mut self.pending_skip);
            return Err(Error::BudgetExceeded {
                requested: frame_size,
                limit: self.max_frame_bytes,
            });
        }
        buf.clear();
        let mut read_so_far = 0usize;
        while read_so_far < frame_size {
            if read_so_far == buf.len() {
                // Mevcut kapasite kadar bir kerede, ötesinde okunan veriyle orantılı büyü.
                let hedef = read_so_far
                    .saturating_mul(2)
                    .max(IVF_READ_CHUNK)
                    .max(buf.capacity())
                    .min(frame_size);
                if buf.capacity() < hedef && buf.try_reserve(hedef - buf.len()).is_err() {
                    // Okuma hatası koluyla aynı: kalan yük sonraki çağrıda atlanır ki akış
                    // kare sınırında hizalı kalsın.
                    self.pending_skip = frame_size - read_so_far;
                    buf.clear();
                    return Err(Error::AllocationFailed { requested: hedef });
                }
                buf.resize(hedef, 0);
            }
            match self.reader.read(&mut buf[read_so_far..]) {
                Ok(0) => {
                    buf.clear();
                    return Err(Error::InvalidData("kesik IVF kare yükü"));
                }
                Ok(n) => read_so_far += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => {
                    self.pending_skip = frame_size - read_so_far;
                    buf.clear();
                    return Err(Error::InvalidData("IVF kare yükü okunamadı"));
                }
            }
        }
        Ok(Some(pts))
    }
}

/// Kalan `pending_skip` baytlarını `[0u8; 4096]` yığın tamponuyla tüketir.
/// Tamamlandığında (veya EOF'ta) `true`, `Interrupted` dışı geçici G/Ç hatasında
/// kalan `pending_skip` değerini koruyarak `false` döndürür.
#[cfg(feature = "std")]
fn drain_pending_skip<R: std::io::Read>(reader: &mut R, pending_skip: &mut usize) -> bool {
    let mut scratch = [0u8; 4096];
    while *pending_skip > 0 {
        let to_read = (*pending_skip).min(scratch.len());
        match reader.read(&mut scratch[..to_read]) {
            Ok(0) => {
                *pending_skip = 0;
                return true;
            }
            Ok(n) => *pending_skip -= n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return false,
        }
    }
    true
}

#[cfg(feature = "std")]
fn read_exact_mapped<R: std::io::Read>(
    reader: &mut R,
    mut buf: &mut [u8],
    eof_msg: &'static str,
) -> Result<()> {
    while !buf.is_empty() {
        match reader.read(buf) {
            Ok(0) => return Err(Error::InvalidData(eof_msg)),
            Ok(n) => buf = &mut buf[n..],
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Err(Error::InvalidData(eof_msg)),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "std")]
    use alloc::vec;
    use alloc::vec::Vec;

    use super::{IvfHeader, IvfParser};
    use crate::error::Error;

    fn make_ivf(frames: &[(i64, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"DKIF");
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&32u16.to_le_bytes());
        out.extend_from_slice(b"AV01");
        out.extend_from_slice(&320u16.to_le_bytes());
        out.extend_from_slice(&240u16.to_le_bytes());
        out.extend_from_slice(&30u32.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(frames.len() as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        for &(pts, payload) in frames {
            out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            out.extend_from_slice(&pts.to_le_bytes());
            out.extend_from_slice(payload);
        }
        out
    }

    #[test]
    fn ivf_parser_gecerli_ve_kesik_akislari_dogru_isler() {
        let ivf = make_ivf(&[(0, &[1, 2, 3]), (42, &[4, 5])]);
        let mut parser = IvfParser::new(&ivf).expect("geçerli IVF");
        assert_eq!(
            *parser.header(),
            IvfHeader {
                version: 0,
                header_len: 32,
                fourcc: *b"AV01",
                width: 320,
                height: 240,
                timebase_den: 30,
                timebase_num: 1,
                frame_count: 2,
            }
        );
        let p0 = parser.next_packet().unwrap().unwrap();
        assert_eq!(p0.pts, 0);
        assert_eq!(p0.data, &[1, 2, 3]);
        let p1 = parser.next_packet().unwrap().unwrap();
        assert_eq!(p1.pts, 42);
        assert_eq!(p1.data, &[4, 5]);
        assert_eq!(parser.next_packet().unwrap(), None);

        // Hatalı imza ve kesik başlık reddedilir.
        assert!(matches!(
            IvfParser::new(&ivf[..16]),
            Err(Error::InvalidData(_))
        ));
        let mut bad_magic = ivf.clone();
        bad_magic[0] = b'X';
        assert!(matches!(
            IvfParser::new(&bad_magic),
            Err(Error::InvalidData(_))
        ));
        let mut bad_fourcc = ivf.clone();
        bad_fourcc[8..12].copy_from_slice(b"VP90");
        assert!(matches!(
            IvfParser::new(&bad_fourcc),
            Err(Error::InvalidData(_))
        ));

        // Kesik kare yükü.
        let mut trunc = IvfParser::new(&ivf[..ivf.len() - 1]).expect("başlık geçerli");
        assert!(trunc.next_packet().unwrap().is_some());
        assert!(matches!(trunc.next_packet(), Err(Error::InvalidData(_))));
        assert_eq!(trunc.next_packet().unwrap(), None);

        // Kesik kare başlığı.
        let mut trunc_hdr = make_ivf(&[]);
        trunc_hdr.extend_from_slice(&[0u8; 5]);
        let mut p = IvfParser::new(&trunc_hdr).unwrap();
        assert!(matches!(p.next_packet(), Err(Error::InvalidData(_))));
    }

    #[cfg(feature = "std")]
    #[test]
    fn ivf_reader_tampon_geri_donusumu_ve_hata_denetimi() {
        use super::IvfReader;

        let ivf = make_ivf(&[(10, &[9, 8, 7, 6]), (20, &[1, 2])]);
        let mut reader = IvfReader::new(ivf.as_slice()).expect("başlık geçerli");
        assert_eq!(reader.header().width, 320);
        let mut buf = vec![0u8; 16];
        let cap_before = buf.capacity();
        assert_eq!(reader.read_packet_into(&mut buf).unwrap(), Some(10));
        assert_eq!(buf, &[9, 8, 7, 6]);
        assert_eq!(reader.read_packet_into(&mut buf).unwrap(), Some(20));
        assert_eq!(buf, &[1, 2]);
        assert_eq!(buf.capacity(), cap_before);
        assert_eq!(reader.read_packet_into(&mut buf).unwrap(), None);
    }

    /// Kesik bir akış büyük kare boyutu beyan ettiğinde tampon beyan edilen boyuta değil,
    /// gelen veriye göre büyümeli; tam kare ise baytları bozulmadan okunmalı.
    #[cfg(feature = "std")]
    #[test]
    fn ivf_reader_kesik_buyuk_kare_beyaninda_bellegi_veriyle_orantili_ayirir() {
        use super::{IVF_READ_CHUNK, IvfReader};

        let declared: u32 = 200 * 1024 * 1024;
        let mut ivf = make_ivf(&[]);
        ivf.extend_from_slice(&declared.to_le_bytes());
        ivf.extend_from_slice(&7i64.to_le_bytes());
        ivf.extend_from_slice(&[0xAB; 100]);
        let mut reader = IvfReader::new(ivf.as_slice()).expect("başlık geçerli");
        let mut buf = Vec::new();
        assert!(matches!(
            reader.read_packet_into(&mut buf),
            Err(Error::InvalidData(_))
        ));
        assert!(buf.is_empty());
        assert!(
            buf.capacity() <= 2 * IVF_READ_CHUNK,
            "kesik kare için ayrılan kapasite beyan edilen boyutla değil veriyle orantılı olmalı: {}",
            buf.capacity()
        );

        let payload: Vec<u8> = (0..(3 * IVF_READ_CHUNK + 17)).map(|i| i as u8).collect();
        let ivf = make_ivf(&[(5, &payload)]);
        let mut reader = IvfReader::new(ivf.as_slice()).expect("başlık geçerli");
        let mut buf = Vec::new();
        assert_eq!(reader.read_packet_into(&mut buf).unwrap(), Some(5));
        assert_eq!(buf, payload);
        assert_eq!(reader.read_packet_into(&mut buf).unwrap(), None);
    }
}
