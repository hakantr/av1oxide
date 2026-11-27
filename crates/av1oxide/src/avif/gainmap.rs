//! ISO 21496-1 / Ultra HDR (`tmap`) Kazanç Haritası (Gain-Map) üst veri ve doğrulama katmanı.

#![forbid(unsafe_code)]

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::avif::image::{AvifImage, ColorInfo, ContentLightLevel, MasteringDisplay};
use crate::error::{Error, Result};

/// İşaretli kesirli (`numerator / denominator`) kazanç haritası parametresi (`int32 / uint32`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GainMapFractionI32 {
    /// İşaretli pay.
    pub numerator: i32,
    /// İşaretsiz payda (`0` olmamalıdır).
    pub denominator: u32,
}

impl GainMapFractionI32 {
    /// Kesirli değeri `f64` olarak döndürür (`denominator == 0` ise `0.0`).
    #[must_use]
    pub fn as_f64(self) -> f64 {
        if self.denominator == 0 {
            0.0
        } else {
            f64::from(self.numerator) / f64::from(self.denominator)
        }
    }

    /// Kesirli değeri `f32` olarak döndürür (`denominator == 0` ise `0.0`).
    #[must_use]
    pub fn as_f32(self) -> f32 {
        self.as_f64() as f32
    }
}

/// İşaretsiz kesirli (`numerator / denominator`) kazanç haritası parametresi (`uint32 / uint32`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GainMapFractionU32 {
    /// İşaretsiz pay.
    pub numerator: u32,
    /// İşaretsiz payda (`0` olmamalıdır).
    pub denominator: u32,
}

impl GainMapFractionU32 {
    /// Kesirli değeri `f64` olarak döndürür (`denominator == 0` ise `0.0`).
    #[must_use]
    pub fn as_f64(self) -> f64 {
        if self.denominator == 0 {
            0.0
        } else {
            f64::from(self.numerator) / f64::from(self.denominator)
        }
    }

    /// Kesirli değeri `f32` olarak döndürür (`denominator == 0` ise `0.0`).
    #[must_use]
    pub fn as_f32(self) -> f32 {
        self.as_f64() as f32
    }
}

/// ISO 21496-1 `tmap` (Tone Map) türetilmiş görüntü öğesinden ayrıştırılan kazanç haritası
/// üst verisi ve alternatif görüntü renk/HDR tanımlayıcıları.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GainMapMetadata {
    /// Kanal başına (`R, G, B`) log2 tabanlı en küçük kazanç değeri.
    pub gain_map_min: [GainMapFractionI32; 3],
    /// Kanal başına (`R, G, B`) log2 tabanlı en büyük kazanç değeri.
    pub gain_map_max: [GainMapFractionI32; 3],
    /// Kanal başına (`R, G, B`) kazanç haritası gama üssü (`numerator > 0`).
    pub gamma: [GainMapFractionU32; 3],
    /// Kanal başına (`R, G, B`) taban görüntü ofset sabiti.
    pub base_offset: [GainMapFractionI32; 3],
    /// Kanal başına (`R, G, B`) alternatif görüntü ofset sabiti.
    pub alternate_offset: [GainMapFractionI32; 3],
    /// Taban görüntünün log2 tabanlı HDR üst boşluğu (`base_hdr_headroom`).
    pub base_hdr_headroom: GainMapFractionU32,
    /// Alternatif görüntünün log2 tabanlı HDR üst boşluğu (`alternate_hdr_headroom`).
    pub alternate_hdr_headroom: GainMapFractionU32,
    /// Ton eşleme matematiğinin taban görüntünün renk uzayında yürütülüp yürütülmeyeceği.
    pub use_base_color_space: bool,
    /// Varsa `tmap` öğesine bağlı alternatif görüntünün CICP (`colr` `nclx`) bilgisi.
    pub alt_color: Option<ColorInfo>,
    /// Varsa `tmap` öğesine bağlı alternatif görüntünün ICC profili (`colr` `rICC`/`prof`).
    pub alt_icc_profile: Option<Vec<u8>>,
    /// Varsa `tmap` `pixi` özelliğinden alınan alternatif görüntü bit derinliği (`0` = belirtilmemiş).
    pub alt_depth: u8,
    /// Varsa `tmap` `pixi` özelliğinden alınan alternatif görüntü kanal sayısı (`0` = belirtilmemiş).
    pub alt_plane_count: u8,
    /// Varsa `tmap` öğesine bağlı alternatif görüntünün içerik ışık düzeyi (`clli`).
    pub alt_clli: Option<ContentLightLevel>,
    /// Varsa `tmap` öğesine bağlı alternatif görüntünün ekran renk hacmi (`mdcv`).
    pub alt_mdcv: Option<MasteringDisplay>,
}

/// Çözülmüş kazanç haritası görüntüsü ve ISO 21496-1 üst verisi.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GainMap {
    /// ISO 21496-1 kazanç haritası üst verisi ve alternatif görüntü özellikleri.
    pub metadata: GainMapMetadata,
    /// Çözülmüş ve kendi `clap`/`irot`/`imir` dönüşümleri uygulanmış kazanç haritası görüntüsü.
    pub image: Box<AvifImage>,
}

fn read_u16_be(buf: &[u8], pos: &mut usize) -> Result<u16> {
    let end = pos
        .checked_add(2)
        .ok_or(Error::InvalidData("tmap yük ofset taşması"))?;
    let slice = buf
        .get(*pos..end)
        .ok_or(Error::InvalidData("kesik tmap ikili yükü"))?;
    *pos = end;
    Ok(u16::from_be_bytes([slice[0], slice[1]]))
}

fn read_u32_be(buf: &[u8], pos: &mut usize) -> Result<u32> {
    let end = pos
        .checked_add(4)
        .ok_or(Error::InvalidData("tmap yük ofset taşması"))?;
    let slice = buf
        .get(*pos..end)
        .ok_or(Error::InvalidData("kesik tmap ikili yükü"))?;
    *pos = end;
    Ok(u32::from_be_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_i32_be(buf: &[u8], pos: &mut usize) -> Result<i32> {
    Ok(read_u32_be(buf, pos)? as i32)
}

/// `GainMapMetadata` alanlarını ISO 21496-1 / `libavif` `avifGainMapValidateMetadata`
/// kurallarına göre doğrular.
pub(crate) fn validate_gain_map_metadata(meta: &GainMapMetadata) -> Result<()> {
    if meta.base_hdr_headroom.denominator == 0 || meta.alternate_hdr_headroom.denominator == 0 {
        return Err(Error::InvalidData(
            "tmap headroom kesrinde sıfır payda (denominator == 0)",
        ));
    }
    for c in 0..3 {
        if meta.gain_map_min[c].denominator == 0
            || meta.gain_map_max[c].denominator == 0
            || meta.gamma[c].denominator == 0
            || meta.base_offset[c].denominator == 0
            || meta.alternate_offset[c].denominator == 0
        {
            return Err(Error::InvalidData(
                "tmap kanal kesrinde sıfır payda (denominator == 0)",
            ));
        }
        let max_cross =
            i64::from(meta.gain_map_max[c].numerator) * i64::from(meta.gain_map_min[c].denominator);
        let min_cross =
            i64::from(meta.gain_map_min[c].numerator) * i64::from(meta.gain_map_max[c].denominator);
        if max_cross < min_cross {
            return Err(Error::InvalidData(
                "tmap gain_map_max değeri gain_map_min değerinden küçük",
            ));
        }
        if meta.gamma[c].numerator == 0 {
            return Err(Error::InvalidData("tmap gamma payı sıfır olamaz"));
        }
    }
    Ok(())
}

/// ISO 21496-1 `ToneMapImage` (`tmap` öğesi) ikili yükünü yığın tahsisi yapmadan ayrıştırır ve doğrular.
///
/// Desteklenmeyen `version != 0` veya `minimum_version > 0` durumunda `libavif` uyumu gereği
/// `Ok(None)` döndürülür ve birincil görüntü çözülmeye devam edilir.
pub(crate) fn parse_tone_mapped_image_box(raw: &[u8]) -> Result<Option<GainMapMetadata>> {
    const SUPPORTED_METADATA_VERSION: u16 = 0;

    if raw.is_empty() {
        return Err(Error::InvalidData("boş tmap ikili yükü"));
    }
    let version = raw[0];
    if version != 0 {
        return Ok(None);
    }

    let mut pos = 1usize;
    let minimum_version = read_u16_be(raw, &mut pos)?;
    if minimum_version > SUPPORTED_METADATA_VERSION {
        return Ok(None);
    }
    let writer_version = read_u16_be(raw, &mut pos)?;
    if writer_version < minimum_version {
        return Err(Error::InvalidData("tmap writer_version < minimum_version"));
    }

    let flags = *raw
        .get(pos)
        .ok_or(Error::InvalidData("kesik tmap bayrak baytı"))?;
    pos += 1;
    let is_multichannel = (flags & 0x80) != 0;
    let use_base_color_space = (flags & 0x40) != 0;
    let channel_count = if is_multichannel { 3usize } else { 1usize };

    let base_hdr_headroom = GainMapFractionU32 {
        numerator: read_u32_be(raw, &mut pos)?,
        denominator: read_u32_be(raw, &mut pos)?,
    };
    let alternate_hdr_headroom = GainMapFractionU32 {
        numerator: read_u32_be(raw, &mut pos)?,
        denominator: read_u32_be(raw, &mut pos)?,
    };

    let zero_i = GainMapFractionI32 {
        numerator: 0,
        denominator: 1,
    };
    let one_u = GainMapFractionU32 {
        numerator: 1,
        denominator: 1,
    };
    let mut gain_map_min = [zero_i; 3];
    let mut gain_map_max = [zero_i; 3];
    let mut gamma = [one_u; 3];
    let mut base_offset = [zero_i; 3];
    let mut alternate_offset = [zero_i; 3];

    for c in 0..channel_count {
        gain_map_min[c] = GainMapFractionI32 {
            numerator: read_i32_be(raw, &mut pos)?,
            denominator: read_u32_be(raw, &mut pos)?,
        };
        gain_map_max[c] = GainMapFractionI32 {
            numerator: read_i32_be(raw, &mut pos)?,
            denominator: read_u32_be(raw, &mut pos)?,
        };
        gamma[c] = GainMapFractionU32 {
            numerator: read_u32_be(raw, &mut pos)?,
            denominator: read_u32_be(raw, &mut pos)?,
        };
        base_offset[c] = GainMapFractionI32 {
            numerator: read_i32_be(raw, &mut pos)?,
            denominator: read_u32_be(raw, &mut pos)?,
        };
        alternate_offset[c] = GainMapFractionI32 {
            numerator: read_i32_be(raw, &mut pos)?,
            denominator: read_u32_be(raw, &mut pos)?,
        };
    }

    if channel_count == 1 {
        for c in 1..3 {
            gain_map_min[c] = gain_map_min[0];
            gain_map_max[c] = gain_map_max[0];
            gamma[c] = gamma[0];
            base_offset[c] = base_offset[0];
            alternate_offset[c] = alternate_offset[0];
        }
    }

    if writer_version == SUPPORTED_METADATA_VERSION && pos != raw.len() {
        return Err(Error::InvalidData(
            "tmap writer_version == 0 yükünde beklenmeyen kuyruk baytları",
        ));
    }

    let metadata = GainMapMetadata {
        gain_map_min,
        gain_map_max,
        gamma,
        base_offset,
        alternate_offset,
        base_hdr_headroom,
        alternate_hdr_headroom,
        use_base_color_space,
        alt_color: None,
        alt_icc_profile: None,
        alt_depth: 0,
        alt_plane_count: 0,
        alt_clli: None,
        alt_mdcv: None,
    };
    validate_gain_map_metadata(&metadata)?;
    Ok(Some(metadata))
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{GainMapFractionI32, GainMapFractionU32, parse_tone_mapped_image_box};
    use crate::error::Error;

    fn make_tmap_payload(
        version: u8,
        min_ver: u16,
        writer_ver: u16,
        multichannel: bool,
        use_base_cs: bool,
        gamma_n: u32,
        extra: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(version);
        out.extend_from_slice(&min_ver.to_be_bytes());
        out.extend_from_slice(&writer_ver.to_be_bytes());
        let flags = (u8::from(multichannel) << 7) | (u8::from(use_base_cs) << 6);
        out.push(flags);
        // base_hdr_headroom = 0/1, alternate_hdr_headroom = 2/1
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(&1u32.to_be_bytes());
        out.extend_from_slice(&2u32.to_be_bytes());
        out.extend_from_slice(&1u32.to_be_bytes());
        let ch = if multichannel { 3 } else { 1 };
        for c in 0..ch {
            out.extend_from_slice(&(-1i32 - c).to_be_bytes());
            out.extend_from_slice(&1u32.to_be_bytes());
            out.extend_from_slice(&(2i32 + c).to_be_bytes());
            out.extend_from_slice(&1u32.to_be_bytes());
            out.extend_from_slice(&gamma_n.to_be_bytes());
            out.extend_from_slice(&1u32.to_be_bytes());
            out.extend_from_slice(&1i32.to_be_bytes());
            out.extend_from_slice(&64u32.to_be_bytes());
            out.extend_from_slice(&1i32.to_be_bytes());
            out.extend_from_slice(&64u32.to_be_bytes());
        }
        out.extend_from_slice(extra);
        out
    }

    #[test]
    fn tmap_valid_single_and_multichannel_and_forward_compat() {
        let single = make_tmap_payload(0, 0, 0, false, true, 1, &[]);
        let meta = parse_tone_mapped_image_box(&single)
            .unwrap()
            .expect("Some(GainMapMetadata) bekleniyordu");
        assert!(meta.use_base_color_space);
        assert_eq!(
            meta.gain_map_min,
            [GainMapFractionI32 {
                numerator: -1,
                denominator: 1
            }; 3]
        );
        assert_eq!(
            meta.gamma,
            [GainMapFractionU32 {
                numerator: 1,
                denominator: 1
            }; 3]
        );
        assert_eq!(meta.alternate_hdr_headroom.as_f64(), 2.0);

        // writer_version > 0 fazladan kuyruk baytları ileri uyumluluk gereği kabul edilmeli.
        let forward = make_tmap_payload(0, 0, 1, true, false, 2, &[0xDE, 0xAD, 0xBE, 0xEF]);
        let meta_mc = parse_tone_mapped_image_box(&forward)
            .unwrap()
            .expect("Some(GainMapMetadata) bekleniyordu");
        assert!(!meta_mc.use_base_color_space);
        assert_eq!(meta_mc.gain_map_max[2].numerator, 4);
    }

    #[test]
    fn tmap_version_and_validation_rules() {
        // version != 0 veya minimum_version > 0 -> None (yoksayılır).
        let bad_ver = make_tmap_payload(1, 0, 0, false, true, 1, &[]);
        assert_eq!(parse_tone_mapped_image_box(&bad_ver).unwrap(), None);
        let bad_min = make_tmap_payload(0, 1, 1, false, true, 1, &[]);
        assert_eq!(parse_tone_mapped_image_box(&bad_min).unwrap(), None);

        // writer_version == 0 + fazladan kuyruk baytı -> InvalidData.
        let extra_v0 = make_tmap_payload(0, 0, 0, false, true, 1, &[0x00]);
        assert!(matches!(
            parse_tone_mapped_image_box(&extra_v0),
            Err(Error::InvalidData(_))
        ));

        // gamma == 0 -> InvalidData.
        let gamma0 = make_tmap_payload(0, 0, 0, false, true, 0, &[]);
        assert!(matches!(
            parse_tone_mapped_image_box(&gamma0),
            Err(Error::InvalidData(_))
        ));
    }
}
