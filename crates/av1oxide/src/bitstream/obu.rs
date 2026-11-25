//! OBU (Open Bitstream Unit) başlığı, uzantısı ve sınır ayrıştırması (F2a / S2a.1).
//!
//! ## Sorumluluk
//!
//! - OBU başlığı: `obu_forbidden_bit`, `obu_type`, `obu_extension_flag`,
//!   `obu_has_size_field`, `obu_reserved_1bit` (AV1 §5.3.2, §6.4.1),
//! - uzantı başlığı: `temporal_id` (3 bit) + `spatial_id` (2 bit) + reserved
//!   (3 bit) (AV1 §5.3.2, §6.4.1),
//! - boyut alanı: `obu_has_size_field = 1` iken `leb128()` (§5.3.3,
//!   §6.4.2); `obu_has_size_field = 0` iken OBU **paketin kalanını** kaplar
//!   (dav1d `obu.c` 1179–1184; Annex B'de dış `obu_unit_size` vardır),
//! - OBU tiplerini **tanıma ve sınırlarıyla atlama**: bu adımda yalnız
//!   `sequence header` ayrıştırılır; `frame header`, `frame`, `tile group`,
//!   `tile list` ve `metadata` yalnız tanınır ve atlanır (P8/P9/F2b).
//!
//! ## Upstream karşılığı (dav1d `99f3354`, salt-okunur)
//!
//! | Rust | dav1d kaynağı | AV1 spesifikasyonu |
//! |---|---|---|
//! | [`ObuHeader`] / [`split_next`] | `src/obu.c` 1169–1197 (`dav1d_parse_obus` başlığı) | §5.3.2, §6.4.1 |
//! | [`ObuExtension`] | `src/obu.c` 1186–1192 | §5.3.2, §6.4.1 |
//! | boyut alanı | `src/obu.c` 1193–1198 | §5.3.3, §6.4.2 |
//! | [`ObuType`] | `include/dav1d/headers.h` 49–58 (`enum Dav1dObuType`) | §5.3.2 (Tablo 4), §6.4.1 |
//! | [`parse_sequence_header_obus`] | `src/obu.c` 302–339 (`dav1d_parse_sequence_header`) | §5.5, §6.5 |
//!
//! ## Bilinçli fark
//!
//! dav1d `obu_forbidden_bit`'i **yalnız** `strict_std_compliance` açıkken
//! reddeder (`obu.c` 1176–1177). [`split_next`] (araçlar, demuxer) her zaman
//! reddeder; [`split_next_with`] katı uyumu parametre alır ve çözücü
//! (`Config::strict_std_compliance`, varsayılan kapalı) dav1d genel API'siyle
//! aynı davranır (S2d.2 yaşam döngüsü senaryosu d).

#![forbid(unsafe_code)]

use crate::bitstream::reader::BitReader;
use crate::bitstream::seq_header::{SequenceHeader, SequenceHeaderParseOptions};
use crate::error::{BitstreamError, BitstreamErrorKind, Error, Result};

/// AV1 OBU tipi (4 bit alan; AV1 §5.3.2 Tablo 4, dav1d `headers.h` 49–58).
///
/// Bilinmeyen/ayrılmış tipler [`ObuType::Reserved`] olarak **korunur**: sınırları
/// bilindiği için atlanabilir, ama asla ayrıştırılmaz.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ObuType {
    /// 1: sequence header OBU — bu adımda tam ayrıştırılır.
    SequenceHeader,
    /// 2: temporal delimiter OBU — yükü boştur, atlanır.
    TemporalDelimiter,
    /// 3: frame header OBU — P8'de ayrıştırılacak, şimdi atlanır.
    FrameHeader,
    /// 4: tile group OBU — F2b'de ayrıştırılacak, şimdi atlanır.
    TileGroup,
    /// 5: metadata OBU — şimdi atlanır (sınırlarıyla tanınır).
    Metadata,
    /// 6: frame OBU — P8/F2b, şimdi atlanır.
    Frame,
    /// 7: redundant frame header OBU — P8, şimdi atlanır.
    RedundantFrameHeader,
    /// 8: tile list OBU — büyük ölçekli tile (F2b), şimdi atlanır.
    TileList,
    /// 15: padding OBU — atlanır.
    Padding,
    /// 0 ve 9–14: ayrılmış tipler — sınırıyla atlanır.
    Reserved(u8),
}

impl ObuType {
    /// 4 bit ham değerden tip (dav1d `headers.h` 49–58 eşlemesi).
    #[must_use]
    pub const fn from_raw(raw: u8) -> Self {
        match raw & 0x0f {
            1 => Self::SequenceHeader,
            2 => Self::TemporalDelimiter,
            3 => Self::FrameHeader,
            4 => Self::TileGroup,
            5 => Self::Metadata,
            6 => Self::Frame,
            7 => Self::RedundantFrameHeader,
            8 => Self::TileList,
            15 => Self::Padding,
            other => Self::Reserved(other),
        }
    }

    /// 4 bit ham değer.
    #[must_use]
    pub const fn raw(self) -> u8 {
        match self {
            Self::SequenceHeader => 1,
            Self::TemporalDelimiter => 2,
            Self::FrameHeader => 3,
            Self::TileGroup => 4,
            Self::Metadata => 5,
            Self::Frame => 6,
            Self::RedundantFrameHeader => 7,
            Self::TileList => 8,
            Self::Padding => 15,
            Self::Reserved(raw) => raw & 0x0f,
        }
    }

    /// İnsan okunur ad (günlük/rapor çıktıları için).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SequenceHeader => "sequence header",
            Self::TemporalDelimiter => "temporal delimiter",
            Self::FrameHeader => "frame header",
            Self::TileGroup => "tile group",
            Self::Metadata => "metadata",
            Self::Frame => "frame",
            Self::RedundantFrameHeader => "redundant frame header",
            Self::TileList => "tile list",
            Self::Padding => "padding",
            Self::Reserved(_) => "reserved",
        }
    }
}

/// OBU uzantı başlığı: katman kimlikleri (AV1 §5.3.2, §6.4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ObuExtension {
    /// Zamansal katman kimliği (3 bit, 0–7).
    pub temporal_id: u8,
    /// Uzamsal katman kimliği (2 bit, 0–3).
    pub spatial_id: u8,
}

/// Ayrıştırılmış OBU başlığı.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ObuHeader {
    /// OBU tipi.
    pub obu_type: ObuType,
    /// Uzantı başlığı (varsa).
    pub extension: Option<ObuExtension>,
    /// `obu_has_size_field`: boyut alanı var mı?
    pub has_size_field: bool,
}

/// Paket içinde bir OBU: başlık + yük dilimi + konum bilgisi.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Obu<'a> {
    /// Ayrıştırılmış başlık.
    pub header: ObuHeader,
    /// OBU yükü (başlık ve boyut alanı hariç).
    pub payload: &'a [u8],
    /// Yükün paket içindeki bayt ofseti.
    pub payload_offset: usize,
    /// OBU'nun paket içindeki toplam bayt uzunluğu (başlık + boyut + yük).
    pub total_len: usize,
}

impl Obu<'_> {
    /// Yükün paket içindeki bit ofseti (hata konumlarını eşlemek için).
    #[must_use]
    pub const fn payload_offset_bits(&self) -> usize {
        self.payload_offset * 8
    }
}

/// Hatayı paket tabanına taşır (OBU dilimine göre üretilen konum + taban).
fn rebase(err: Error, base_bits: usize) -> Error {
    match err {
        Error::Bitstream(e) => Error::Bitstream(BitstreamError::new(
            e.kind,
            e.bit_offset.saturating_add(base_bits),
        )),
        other => other,
    }
}

/// Paketin `offset` konumundaki sıradaki OBU'yu ayrıştırır.
///
/// `offset` başarılı ayrıştırmada OBU'nun sonuna ilerletilir; paket sonunda
/// `Ok(None)` döner. `obu_has_size_field = 0` ise OBU **paketin kalanını**
/// kaplar (dav1d `obu.c` 1193–1197 aynı davranış).
///
/// # Hatalar
///
/// - [`BitstreamErrorKind::UnexpectedEnd`]: başlık/uzantı/boyut alanı kesik.
/// - [`BitstreamErrorKind::ForbiddenBit`]: `obu_forbidden_bit` 1.
/// - [`BitstreamErrorKind::ObuSizeOverflow`]: boyut alanı kalan baytı aşıyor.
pub fn split_next<'a>(packet: &'a [u8], offset: &mut usize) -> Result<Option<Obu<'a>>> {
    split_next_with(packet, offset, true)
}

/// [`split_next`]; `strict` dav1d `strict_std_compliance`'tır:
/// `obu_forbidden_bit = 1` yalnız katı uyumda reddedilir (`src/obu.c:1176-1177`;
/// dav1d kütüphane varsayılanı 0). [`crate::Decoder`] bu yolu
/// `Config::strict_std_compliance` ile kullanır.
///
/// # Hatalar
///
/// [`split_next`] ile aynı (yasak bit yalnız `strict` iken).
pub fn split_next_with<'a>(
    packet: &'a [u8],
    offset: &mut usize,
    strict: bool,
) -> Result<Option<Obu<'a>>> {
    if *offset > packet.len() {
        return Err(BitstreamError::new(BitstreamErrorKind::UnexpectedEnd, *offset * 8).into());
    }
    if *offset == packet.len() {
        return Ok(None);
    }
    let base = *offset;
    let mut br = BitReader::new(&packet[base..]);
    let forbidden = br.read_bit().map_err(|err| rebase(err, base * 8))?;
    if forbidden != 0 && strict {
        return Err(BitstreamError::new(BitstreamErrorKind::ForbiddenBit, base * 8).into());
    }
    let raw_type = br.read_bits(4).map_err(|err| rebase(err, base * 8))? as u8;
    let obu_type = ObuType::from_raw(raw_type);
    let has_extension = br.read_bit().map_err(|err| rebase(err, base * 8))? != 0;
    let has_size_field = br.read_bit().map_err(|err| rebase(err, base * 8))? != 0;
    // obu_reserved_1bit: 0 olmalıdır ama dav1d gibi yok sayılır (§5.3.2).
    br.read_bit().map_err(|err| rebase(err, base * 8))?;
    let extension = if has_extension {
        let temporal_id = br.read_bits(3).map_err(|err| rebase(err, base * 8))? as u8;
        let spatial_id = br.read_bits(2).map_err(|err| rebase(err, base * 8))? as u8;
        // Uzantının son 3 biti ayrılmıştır; dav1d gibi yok sayılır.
        br.read_bits(3).map_err(|err| rebase(err, base * 8))?;
        Some(ObuExtension {
            temporal_id,
            spatial_id,
        })
    } else {
        None
    };
    let declared_len = if has_size_field {
        Some(br.read_uleb128().map_err(|err| rebase(err, base * 8))? as usize)
    } else {
        None
    };
    if !br.is_byte_aligned() {
        // Başlık (1/2 bayt) + leb128 (tam baytlar) daima bayt hizalıdır; bu
        // noktaya yalnız kod değişikliğiyle gelinebilir. Panik yerine hata.
        return Err(BitstreamError::new(BitstreamErrorKind::UnexpectedEnd, base * 8).into());
    }
    let header_len = br.byte_position();
    let payload_offset = base + header_len;
    let remaining = packet.len() - payload_offset;
    let payload_len = match declared_len {
        Some(declared) => {
            if declared > remaining {
                return Err(BitstreamError::new(
                    BitstreamErrorKind::ObuSizeOverflow {
                        declared,
                        remaining,
                    },
                    base * 8,
                )
                .into());
            }
            declared
        }
        None => remaining,
    };
    *offset = payload_offset + payload_len;
    Ok(Some(Obu {
        header: ObuHeader {
            obu_type,
            extension,
            has_size_field,
        },
        payload: &packet[payload_offset..payload_offset + payload_len],
        payload_offset,
        total_len: *offset - base,
    }))
}

/// Bir tampondaki OBU'ları gezer ve **son** sequence header'ı döndürür.
///
/// dav1d'nin genel API'si `dav1d_parse_sequence_header` (obu.c 302–339) ile
/// aynı sözleşme: sequence header dışındaki OBU'lar yok sayılır; birden çok
/// sequence header varsa **sonuncusu** döner; hiç yoksa `Ok(None)`.
///
/// # Hatalar
///
/// OBU sınırı ya da sequence header alanı geçersizse ilgili [`Error`].
pub fn parse_sequence_header_obus(data: &[u8]) -> Result<Option<SequenceHeader>> {
    let mut offset = 0usize;
    let mut result = None;
    while let Some(obu) = split_next(data, &mut offset)? {
        if obu.header.obu_type != ObuType::SequenceHeader {
            continue;
        }
        let header = SequenceHeader::parse_with(obu.payload, SequenceHeaderParseOptions::default())
            .map_err(|err| rebase(err, obu.payload_offset_bits()))?;
        result = Some(header);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{ObuType, parse_sequence_header_obus, split_next};
    use crate::error::{BitstreamErrorKind, Error};

    fn kind(err: Error) -> BitstreamErrorKind {
        match err {
            Error::Bitstream(err) => err.kind,
            other => panic!("bit akışı hatası bekleniyordu: {other:?}"),
        }
    }

    #[test]
    fn obu_tipi_ham_deger_eslemesi() {
        assert_eq!(ObuType::from_raw(1), ObuType::SequenceHeader);
        assert_eq!(ObuType::from_raw(2), ObuType::TemporalDelimiter);
        assert_eq!(ObuType::from_raw(8), ObuType::TileList);
        assert_eq!(ObuType::from_raw(15), ObuType::Padding);
        assert_eq!(ObuType::from_raw(9), ObuType::Reserved(9));
        for raw in 0..=15u8 {
            assert_eq!(ObuType::from_raw(raw).raw(), raw, "ham değer {raw}");
        }
        assert_eq!(ObuType::Reserved(10).name(), "reserved");
        assert_eq!(ObuType::SequenceHeader.name(), "sequence header");
    }

    #[test]
    fn uzantisiz_baslik_ve_boyut_alani() {
        // 0x0A: forbidden=0, type=1 (seq hdr), ext=0, has_size=1, reserved=0
        // 0x02: boyut 2; yük 0xAA 0xBB
        let data = [0x0A, 0x02, 0xAA, 0xBB];
        let mut offset = 0usize;
        let obu = split_next(&data, &mut offset)
            .expect("ayrıştırma")
            .expect("OBU var");
        assert_eq!(obu.header.obu_type, ObuType::SequenceHeader);
        assert_eq!(obu.header.extension, None);
        assert!(obu.header.has_size_field);
        assert_eq!(obu.payload, &[0xAA, 0xBB]);
        assert_eq!(obu.payload_offset, 2);
        assert_eq!(obu.total_len, 4);
        assert_eq!(offset, 4);
        assert!(split_next(&data, &mut offset).expect("son").is_none());
    }

    #[test]
    fn uzanti_basligi_okunur() {
        // 0x0E = 0 0001 1 1 0: seq hdr, uzantı var, boyut alanı var.
        // Uzantı baytı: temporal_id=3 (011), spatial_id=2 (10), reserved=000.
        let data = [0x0E, 0b0111_0000, 0x01, 0x80];
        let mut offset = 0usize;
        let obu = split_next(&data, &mut offset)
            .expect("ayrıştırma")
            .expect("OBU var");
        let ext = obu.header.extension.expect("uzantı");
        assert_eq!(ext.temporal_id, 3);
        assert_eq!(ext.spatial_id, 2);
        assert!(obu.header.has_size_field);
        assert_eq!(obu.payload, &[0x80]);
        assert_eq!(obu.payload_offset, 3);
    }

    #[test]
    fn boyut_alani_yoksa_paketin_kalani_obu_yukudur() {
        // 0x08: seq hdr, uzantısız, boyut alanı YOK.
        let data = [0x08, 0xAA, 0xBB, 0xCC];
        let mut offset = 0usize;
        let obu = split_next(&data, &mut offset)
            .expect("ayrıştırma")
            .expect("OBU var");
        assert!(!obu.header.has_size_field);
        assert_eq!(obu.payload, &[0xAA, 0xBB, 0xCC]);
        assert_eq!(offset, 4);
        assert!(split_next(&data, &mut offset).expect("son").is_none());
    }

    #[test]
    fn boyut_asimi_ve_yasak_bit_hata_verir() {
        let data = [0x0A, 0x05, 0x00];
        let mut offset = 0usize;
        let err = split_next(&data, &mut offset).expect_err("boyut aşımı");
        assert!(matches!(
            kind(err),
            BitstreamErrorKind::ObuSizeOverflow {
                declared: 5,
                remaining: 1
            }
        ));

        let data = [0x80, 0x00];
        let mut offset = 0usize;
        let err = split_next(&data, &mut offset).expect_err("yasak bit");
        assert_eq!(kind(err), BitstreamErrorKind::ForbiddenBit);

        // Kesik başlık.
        let data = [0x0A];
        let mut offset = 0usize;
        let err = split_next(&data, &mut offset).expect_err("kesik");
        assert_eq!(kind(err), BitstreamErrorKind::UnexpectedEnd);
    }

    #[test]
    fn birden_cok_obu_ve_ayrilmis_tip_atlanir() {
        // TD (boyut 0) + ayrılmış tip 10 (boyut 1) + padding (boyut 0)
        let data = [0x12, 0x00, 0x52, 0x01, 0xFF, 0x7A, 0x00];
        let mut offset = 0usize;
        let mut types = Vec::new();
        while let Some(obu) = split_next(&data, &mut offset).expect("ayrıştırma") {
            types.push(obu.header.obu_type);
            if obu.header.obu_type == ObuType::Reserved(10) {
                assert_eq!(obu.payload, &[0xFF]);
            }
        }
        assert_eq!(
            types,
            vec![
                ObuType::TemporalDelimiter,
                ObuType::Reserved(10),
                ObuType::Padding
            ]
        );
    }

    #[test]
    fn sequence_header_yardimcisi_sonuncuyu_dondurur() {
        // Sequence header içermeyen tampon -> None.
        assert_eq!(parse_sequence_header_obus(&[0x12, 0x00]).expect("TD"), None);
        // Boş tampon.
        assert_eq!(parse_sequence_header_obus(&[]).expect("boş"), None);
        // Bozuk sequence header yükü hatayı yukarı taşır.
        assert!(parse_sequence_header_obus(&[0x0A, 0x00]).is_err());
    }

    #[test]
    fn bos_paket_none_doner() {
        let mut offset = 0usize;
        assert!(split_next(&[], &mut offset).expect("boş").is_none());
        assert_eq!(offset, 0);
    }
}
