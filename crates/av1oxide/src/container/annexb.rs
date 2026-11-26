//! AV1 Spesifikasyonu Annex B uzunluk-ön-ekli bit akışı ayrıştırıcısı (`#![forbid(unsafe_code)]`).
//!
//! AV1 Annex B biçimi her `temporal_unit`, `frame_unit` ve `obu_unit` önüne
//! `leb128` uzunluk alanı koyar ve OBU başlıklarında `obu_has_size_field` bitinin
//! `0` olmasına izin verir. [`AnnexBDemuxer`], her `temporal_unit` birimini
//! [`crate::Decoder::send_packet`] tarafından doğrudan çözülebilen standart
//! Section 5 düşük-ek-yüklü OBU paketine (`obu_has_size_field = 1` + kanonik
//! `leb128` yük uzunluğu) dönüştürür.

#![forbid(unsafe_code)]

use alloc::vec::Vec;

use crate::error::{Error, Result};

/// AV1 spesifikasyonu `leb128()` için izin verilen azami bayt uzunluğu.
const LEB128_MAX_BYTES: usize = 8;

/// AV1 Spesifikasyonu §4.10.5 uyarınca `leb128` değerini okur.
///
/// En fazla `8` bayt okur ve çözülen değerin `u32::MAX` (`(1 << 32) - 1`) sınırını
/// aşmadığını doğrular. Dönüş değeri `(değer, tüketilen_bayt_sayısı)` çiftidir.
pub fn read_leb128(data: &[u8]) -> Result<(u32, usize)> {
    let mut value = 0u64;
    for i in 0..LEB128_MAX_BYTES {
        let &byte = data
            .get(i)
            .ok_or(Error::InvalidData("kesik Annex B leb128 alanı"))?;
        let chunk = u64::from(byte & 0x7f);
        value |= chunk << (7 * i);
        if (byte & 0x80) == 0 {
            let val32 = u32::try_from(value).map_err(|_| {
                Error::InvalidData("Annex B leb128 değeri u32::MAX sınırını aşıyor")
            })?;
            return Ok((val32, i + 1));
        }
    }
    Err(Error::InvalidData(
        "Annex B leb128 alanı 8 bayt sınırını aşıyor",
    ))
}

/// `u32` değerini kanonik (en kısa) `leb128` baytları olarak `out` sonuna yazar.
fn write_leb128(out: &mut Vec<u8>, mut value: u32) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
            out.push(byte);
        } else {
            out.push(byte);
            break;
        }
    }
}

/// AV1 Annex B (`temporal_unit` → `frame_unit` → `obu_unit`) ayrıştırıcısı.
#[derive(Clone, Debug)]
pub struct AnnexBDemuxer<'a> {
    data: &'a [u8],
    offset: usize,
    allocation_limit: Option<usize>,
    finished: bool,
}

impl<'a> AnnexBDemuxer<'a> {
    /// Verilen Annex B bit akışı dilimi için yeni bir [`AnnexBDemuxer`] oluşturur.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            offset: 0,
            allocation_limit: None,
            finished: false,
        }
    }

    /// Tek bir çözümlenen `temporal_unit` çıktısı için isteğe bağlı bayt üst sınırı belirler.
    #[must_use]
    pub const fn with_allocation_limit(mut self, limit: Option<usize>) -> Self {
        self.allocation_limit = limit;
        self
    }

    /// Sıradaki `temporal_unit` birimini standart Section 5 OBU paketi olarak döndürür.
    ///
    /// Akış temiz biçimde sonlandığında `Ok(None)` döner. Bozuk/taşan `leb128` veya
    /// kesik uzunluklarda panik üretmeden [`Error::InvalidData`] / [`Error::BudgetExceeded`]
    /// döndürür.
    pub fn next_temporal_unit(&mut self) -> Result<Option<Vec<u8>>> {
        if self.finished || self.offset == self.data.len() {
            self.finished = true;
            return Ok(None);
        }

        let (tu_size_u32, tu_leb_len) = match read_leb128(&self.data[self.offset..]) {
            Ok(v) => v,
            Err(err) => {
                self.finished = true;
                return Err(err);
            }
        };
        let tu_size = tu_size_u32 as usize;
        let tu_start = match self.offset.checked_add(tu_leb_len) {
            Some(v) => v,
            None => {
                self.finished = true;
                return Err(Error::InvalidData("Annex B temporal_unit ofset taşması"));
            }
        };
        let tu_end = match tu_start.checked_add(tu_size) {
            Some(v) if v <= self.data.len() => v,
            _ => {
                self.finished = true;
                return Err(Error::InvalidData(
                    "kesik Annex B temporal_unit (bildirilen boyut girdi sonunu aşıyor)",
                ));
            }
        };
        if tu_size == 0 {
            self.finished = true;
            return Err(Error::InvalidData("boş Annex B temporal_unit (boyut 0)"));
        }

        // Her OBU en az 1 bayt başlık taşır; `obu_has_size_field = 1` eklemesi
        // OBU başına en çok 5 bayt kanonik leb128 ekler. Çıktı tamponu girdi
        // uzunluğuyla orantılı olarak üstten sınırlanır.
        let max_output_len = self
            .allocation_limit
            .unwrap_or(usize::MAX)
            .min(tu_size.saturating_mul(6));
        let initial_cap = tu_size.min(max_output_len);
        let mut out = Vec::new();
        out.try_reserve(initial_cap).map_err(|_| {
            self.finished = true;
            Error::AllocationFailed {
                requested: initial_cap,
            }
        })?;

        let tu_slice = &self.data[tu_start..tu_end];
        if let Err(err) =
            demux_temporal_unit_slice(tu_slice, max_output_len, self.allocation_limit, &mut out)
        {
            self.finished = true;
            return Err(err);
        }

        self.offset = tu_end;
        Ok(Some(out))
    }

    /// Tüm Annex B akışını baştan sona okuyarak Section 5 OBU paketleri listesi döndürür.
    pub fn demux_all(data: &[u8]) -> Result<Vec<Vec<u8>>> {
        let mut demuxer = AnnexBDemuxer::new(data);
        let mut units = Vec::new();
        while let Some(unit) = demuxer.next_temporal_unit()? {
            units.push(unit);
        }
        Ok(units)
    }
}

impl Iterator for AnnexBDemuxer<'_> {
    type Item = Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_temporal_unit().transpose()
    }
}

fn demux_temporal_unit_slice(
    tu_slice: &[u8],
    max_output_len: usize,
    allocation_limit: Option<usize>,
    out: &mut Vec<u8>,
) -> Result<()> {
    let mut tu_pos = 0usize;
    while tu_pos < tu_slice.len() {
        let (fu_size_u32, fu_leb_len) = read_leb128(&tu_slice[tu_pos..])?;
        let fu_size = fu_size_u32 as usize;
        if fu_size == 0 {
            return Err(Error::InvalidData("boş Annex B frame_unit (boyut 0)"));
        }
        let fu_start = tu_pos
            .checked_add(fu_leb_len)
            .ok_or(Error::InvalidData("Annex B frame_unit ofset taşması"))?;
        let fu_end = fu_start
            .checked_add(fu_size)
            .ok_or(Error::InvalidData("Annex B frame_unit boyut taşması"))?;
        if fu_end > tu_slice.len() {
            return Err(Error::InvalidData(
                "Annex B frame_unit boyutu temporal_unit sınırını aşıyor",
            ));
        }

        let fu_slice = &tu_slice[fu_start..fu_end];
        let mut fu_pos = 0usize;
        while fu_pos < fu_slice.len() {
            let (obu_len_u32, obu_leb_len) = read_leb128(&fu_slice[fu_pos..])?;
            let obu_len = obu_len_u32 as usize;
            if obu_len == 0 {
                return Err(Error::InvalidData("boş Annex B OBU birimi (boyut 0)"));
            }
            let obu_start = fu_pos
                .checked_add(obu_leb_len)
                .ok_or(Error::InvalidData("Annex B OBU ofset taşması"))?;
            let obu_end = obu_start
                .checked_add(obu_len)
                .ok_or(Error::InvalidData("Annex B OBU boyut taşması"))?;
            if obu_end > fu_slice.len() {
                return Err(Error::InvalidData(
                    "Annex B OBU boyutu frame_unit sınırını aşıyor",
                ));
            }

            emit_section5_obu(
                &fu_slice[obu_start..obu_end],
                max_output_len,
                allocation_limit,
                out,
            )?;
            fu_pos = obu_end;
        }
        tu_pos = fu_end;
    }
    Ok(())
}

fn emit_section5_obu(
    obu: &[u8],
    max_output_len: usize,
    allocation_limit: Option<usize>,
    out: &mut Vec<u8>,
) -> Result<()> {
    let hdr = *obu
        .first()
        .ok_or(Error::InvalidData("Annex B OBU başlığı eksik"))?;
    if (hdr & 0x80) != 0 {
        return Err(Error::InvalidData("Annex B OBU forbidden biti sıfır değil"));
    }
    let has_extension = (hdr & 0x04) != 0;
    let has_size_field = (hdr & 0x02) != 0;
    let hdr_len = 1 + usize::from(has_extension);
    if obu.len() < hdr_len {
        return Err(Error::InvalidData(
            "kesik Annex B OBU genişletilmiş başlığı",
        ));
    }

    if has_size_field {
        // OBU zaten kendi `obu_size` alanını taşıyor: uzunluk tutarlılığını doğrula ve aktar.
        let (embedded_size, size_len) = read_leb128(&obu[hdr_len..])?;
        let expected = hdr_len
            .checked_add(size_len)
            .and_then(|v| v.checked_add(embedded_size as usize))
            .ok_or(Error::InvalidData("Annex B OBU gömülü boyut taşması"))?;
        if expected > obu.len() {
            return Err(Error::InvalidData(
                "Annex B OBU gömülü obu_size değeri obu_length sınırını aşıyor",
            ));
        }
        ensure_output_capacity(out, obu.len(), max_output_len, allocation_limit)?;
        out.extend_from_slice(obu);
    } else {
        let payload = &obu[hdr_len..];
        let payload_len = u32::try_from(payload.len())
            .map_err(|_| Error::InvalidData("Annex B OBU yükü u32::MAX sınırını aşıyor"))?;
        let leb_len = canonical_leb128_len(payload_len);
        let needed = hdr_len
            .checked_add(leb_len)
            .and_then(|v| v.checked_add(payload.len()))
            .ok_or(Error::InvalidData("Section 5 OBU boyut taşması"))?;
        ensure_output_capacity(out, needed, max_output_len, allocation_limit)?;
        out.push(hdr | 0x02);
        if has_extension {
            out.push(obu[1]);
        }
        write_leb128(out, payload_len);
        out.extend_from_slice(payload);
    }
    Ok(())
}

fn canonical_leb128_len(mut value: u32) -> usize {
    let mut len = 1usize;
    while value >= 0x80 {
        value >>= 7;
        len += 1;
    }
    len
}

fn ensure_output_capacity(
    out: &mut Vec<u8>,
    add: usize,
    max_output_len: usize,
    allocation_limit: Option<usize>,
) -> Result<()> {
    let new_len = out
        .len()
        .checked_add(add)
        .ok_or(Error::InvalidData("Section 5 paket uzunluğu taşması"))?;
    if new_len > max_output_len {
        return Err(Error::BudgetExceeded {
            requested: new_len,
            limit: allocation_limit.unwrap_or(max_output_len),
        });
    }
    out.try_reserve(add)
        .map_err(|_| Error::AllocationFailed { requested: new_len })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{AnnexBDemuxer, read_leb128, write_leb128};
    use crate::error::Error;

    fn wrap_annexb_tu(obus: &[&[u8]]) -> Vec<u8> {
        let mut fu = Vec::new();
        for &obu in obus {
            write_leb128(&mut fu, obu.len() as u32);
            fu.extend_from_slice(obu);
        }
        let mut tu = Vec::new();
        write_leb128(&mut tu, fu.len() as u32);
        tu.extend_from_slice(&fu);

        let mut stream = Vec::new();
        write_leb128(&mut stream, tu.len() as u32);
        stream.extend_from_slice(&tu);
        stream
    }

    #[test]
    fn leb128_sinir_ve_hata_kosullari() {
        assert_eq!(read_leb128(&[0x00]), Ok((0, 1)));
        assert_eq!(read_leb128(&[0x7f]), Ok((127, 1)));
        assert_eq!(read_leb128(&[0x80, 0x01]), Ok((128, 2)));
        // 8 baytlık en küçük olmayan (non-minimal) geçerli kodlama (Argon akışları).
        assert_eq!(
            read_leb128(&[0x85, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00]),
            Ok((5, 8))
        );
        // u32::MAX (5 bayt) geçerlidir.
        assert_eq!(
            read_leb128(&[0xff, 0xff, 0xff, 0xff, 0x0f]),
            Ok((u32::MAX, 5))
        );
        // u32::MAX + 1 taşması reddedilir.
        assert!(matches!(
            read_leb128(&[0x80, 0x80, 0x80, 0x80, 0x10]),
            Err(Error::InvalidData(_))
        ));
        // 8 bayttan uzun leb128 reddedilir.
        assert!(matches!(
            read_leb128(&[0x80; 9]),
            Err(Error::InvalidData(_))
        ));
        // Kesik leb128 reddedilir.
        assert!(matches!(read_leb128(&[0x81]), Err(Error::InvalidData(_))));
    }

    #[test]
    fn annexb_demuxer_has_size_field_ekler_ve_bozuk_girdiyi_reddeder() {
        // obu_has_size_field = 0 (0x08 = Sequence Header, 3 bayt yük) + uzantılı OBU (0x0c, ext=0x18).
        let obu_no_size = [0x08u8, 0xaa, 0xbb, 0xcc];
        let obu_ext_no_size = [0x0cu8, 0x18, 0xde, 0xad];
        // Zaten obu_has_size_field = 1 taşıyan Temporal Delimiter.
        let obu_with_size = [0x12u8, 0x00];

        let stream = wrap_annexb_tu(&[&obu_with_size, &obu_no_size, &obu_ext_no_size]);
        let units = AnnexBDemuxer::demux_all(&stream).expect("geçerli Annex B");
        assert_eq!(units.len(), 1);
        assert_eq!(
            units[0],
            &[
                0x12, 0x00, // TD korundu
                0x0a, 0x03, 0xaa, 0xbb, 0xcc, // has_size_field=1 + len=3
                0x0e, 0x18, 0x02, 0xde, 0xad, // has_size_field=1 + ext + len=2
            ]
        );

        // Kesik stream reddedilir.
        assert!(matches!(
            AnnexBDemuxer::demux_all(&stream[..stream.len() - 1]),
            Err(Error::InvalidData(_))
        ));

        // Bütçe aşımı temiz BudgetExceeded döner.
        let mut limited = AnnexBDemuxer::new(&stream).with_allocation_limit(Some(4));
        assert!(matches!(
            limited.next_temporal_unit(),
            Err(Error::BudgetExceeded { .. })
        ));
    }
}
