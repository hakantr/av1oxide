//! ISOBMFF (`ISO/IEC 14496-12`), HEIF/MIAF (`ISO/IEC 23008-12`) ve AVIF (`v1.1.0`)
//! kutu ayrıştırıcısı.

#![forbid(unsafe_code)]

use std::borrow::Cow;

use crate::avif::image::{
    CleanAperture, ColorInfo, ContentLightLevel, MasteringDisplay, PixelAspectRatio,
};
use crate::error::{Error, Result};

/// AVIF / HEIF dosya imzasında kabul edilen `ftyp` marka kodları.
const SUPPORTED_BRANDS: &[[u8; 4]] = &[
    *b"avif", *b"avis", *b"mif1", *b"msf1", *b"miaf", *b"MA1B", *b"MA1A",
];

/// Alfa yardımcı öğesi için standart URN tanımlayıcıları.
pub(crate) const ALPHA_AUX_URN_CICP: &str = "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha";
pub(crate) const ALPHA_AUX_URN_HEVC: &str = "urn:mpeg:hevc:2015:auxid:1";

/// Ham ISOBMFF kutusu (`Box`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct RawBox<'a> {
    pub(crate) box_type: [u8; 4],
    pub(crate) payload: &'a [u8],
    pub(crate) next_offset: usize,
}

/// `av1C` (`AV1CodecConfigurationRecord`) özelliği.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Av1CProperty {
    pub(crate) seq_profile: u8,
    pub(crate) seq_level_idx_0: u8,
    pub(crate) seq_tier_0: u8,
    pub(crate) high_bitdepth: bool,
    pub(crate) twelve_bit: bool,
    pub(crate) monochrome: bool,
    pub(crate) chroma_subsampling_x: u8,
    pub(crate) chroma_subsampling_y: u8,
    pub(crate) chroma_sample_position: u8,
    pub(crate) config_obus: Vec<u8>,
}

/// `ipco` kutusu içindeki tekil özellik girişi.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ItemProperty {
    /// `ispe`: görüntü genişliği ve yüksekliği.
    Ispe { width: u32, height: u32 },
    /// `pixi`: kanal başına bit derinlikleri.
    Pixi { bits_per_channel: Vec<u8> },
    /// `av1C`: AV1 codec yapılandırma kaydı.
    Av1C(Av1CProperty),
    /// `colr` (`nclx`): CICP / H.273 renk tanımlayıcıları.
    ColrNclx(ColorInfo),
    /// `colr` (`rICC` veya `prof`): ham ICC renk profili baytları.
    ColrIcc(Vec<u8>),
    /// `clap`: kesirli temiz açıklık (kırpma) bölgesi.
    Clap(CleanAperture),
    /// `irot`: saat yönünün tersine döndürme açısı (`0`, `1` = 90°, `2` = 180°, `3` = 270°).
    Irot(u8),
    /// `imir`: aynalama ekseni (`0` = dikey eksen / üst-alt, `1` = yatay eksen / sol-sağ).
    Imir(u8),
    /// `pasp`: piksel en-boy oranı.
    Pasp(PixelAspectRatio),
    /// `clli`: içerik ışık düzeyi bilgisi.
    Clli(ContentLightLevel),
    /// `mdcv`: referans ekran renk hacmi metaverisi.
    Mdcv(MasteringDisplay),
    /// `a1op`: AV1 çalışma noktası (`operating_point`) seçimi.
    A1op(u8),
    /// `lsel`: AV1 uzamsal katman (`layer_id`) seçimi.
    Lsel(u16),
    /// `auxC`: yardımcı görüntü türü URN dizesi ve alt tür verisi.
    AuxC { aux_type: String, subtype: Vec<u8> },
    /// Tanınmayan özellik kutusu.
    Unknown([u8; 4]),
}

/// `ipma` öğe-özellik ilişkilendirme kaydı.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PropertyAssoc {
    /// 1 tabanlı `ipco` özellik indeksi (`0` = ilişkilendirme yok).
    pub(crate) property_index: u16,
    /// Özelliğin zorunlu (`essential`) olup olmadığı.
    pub(crate) essential: bool,
}

/// `infe` öğe bilgi kaydı.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ItemInfoEntry {
    pub(crate) item_id: u32,
    pub(crate) item_type: [u8; 4],
    pub(crate) item_name: String,
    pub(crate) content_type: String,
    pub(crate) hidden: bool,
}

/// `iloc` içindeki tekil veri parçası (`extent`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IlocExtent {
    pub(crate) extent_index: u64,
    pub(crate) extent_offset: u64,
    pub(crate) extent_length: u64,
}

/// `iloc` öğe konum kaydı.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct IlocItem {
    pub(crate) item_id: u32,
    pub(crate) construction_method: u8,
    pub(crate) data_reference_index: u16,
    pub(crate) base_offset: u64,
    pub(crate) extents: Vec<IlocExtent>,
}

/// `iref` öğe referans kaydı.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ItemReference {
    pub(crate) ref_type: [u8; 4],
    pub(crate) from_item_id: u32,
    pub(crate) to_item_ids: Vec<u32>,
}

/// Ayrıştırılmış AVIF / ISOBMFF kapsayıcı yapısı.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct AvifContainer<'a> {
    pub(crate) file_data: &'a [u8],
    pub(crate) major_brand: [u8; 4],
    pub(crate) primary_item_id: u32,
    pub(crate) items: Vec<ItemInfoEntry>,
    pub(crate) properties: Vec<ItemProperty>,
    pub(crate) associations: Vec<(u32, Vec<PropertyAssoc>)>,
    pub(crate) locations: Vec<IlocItem>,
    pub(crate) references: Vec<ItemReference>,
    pub(crate) idat: &'a [u8],
}

fn read_u16_be(buf: &[u8], offset: usize) -> Option<u16> {
    let bytes = buf.get(offset..offset.checked_add(2)?)?;
    Some(u16::from_be_bytes([bytes[0], bytes[1]]))
}

fn read_u32_be(buf: &[u8], offset: usize) -> Option<u32> {
    let bytes = buf.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_i32_be(buf: &[u8], offset: usize) -> Option<i32> {
    let bytes = buf.get(offset..offset.checked_add(4)?)?;
    Some(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_u64_be(buf: &[u8], offset: usize) -> Option<u64> {
    let bytes = buf.get(offset..offset.checked_add(8)?)?;
    Some(u64::from_be_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

fn read_uint_n(buf: &[u8], offset: usize, size: usize) -> Option<u64> {
    match size {
        0 => Some(0),
        4 => read_u32_be(buf, offset).map(u64::from),
        8 => read_u64_be(buf, offset),
        _ => None,
    }
}

fn read_cstring(buf: &[u8], offset: usize) -> Result<(String, usize)> {
    let tail = buf
        .get(offset..)
        .ok_or(Error::InvalidData("ISOBMFF dize ofseti sınır dışında"))?;
    let nul_pos = tail
        .iter()
        .position(|&b| b == 0)
        .ok_or(Error::InvalidData("ISOBMFF dizesi NUL ile sonlanmıyor"))?;
    let s = core::str::from_utf8(&tail[..nul_pos])
        .map_err(|_| Error::InvalidData("ISOBMFF dizesi geçerli UTF-8 değil"))?;
    Ok((s.to_owned(), offset + nul_pos + 1))
}

/// Verilen dilim içindeki sıradaki ISOBMFF kutusunu okur.
pub(crate) fn read_box_header(buf: &[u8], offset: usize) -> Result<Option<RawBox<'_>>> {
    if offset == buf.len() {
        return Ok(None);
    }
    if buf.len().saturating_sub(offset) < 8 {
        return Err(Error::InvalidData("kesik ISOBMFF kutu başlığı (< 8 bayt)"));
    }
    let size32 = read_u32_be(buf, offset).ok_or(Error::InvalidData("kutu boyutu okunamadı"))?;
    let t_slice = buf
        .get(offset + 4..offset + 8)
        .ok_or(Error::InvalidData("kutu türü okunamadı"))?;
    let box_type = [t_slice[0], t_slice[1], t_slice[2], t_slice[3]];

    let (mut header_len, total_size): (usize, usize) = match size32 {
        0 => (8, buf.len() - offset),
        1 => {
            let size64 = read_u64_be(buf, offset + 8)
                .ok_or(Error::InvalidData("kesik 64-bit largesize kutu başlığı"))?;
            if size64 < 16 {
                return Err(Error::InvalidData("ISOBMFF largesize < 16"));
            }
            let sz = usize::try_from(size64)
                .map_err(|_| Error::InvalidData("ISOBMFF kutu boyutu usize sınırını aşıyor"))?;
            (16, sz)
        }
        s => {
            let sz = s as usize;
            if sz < 8 {
                return Err(Error::InvalidData("ISOBMFF kutu boyutu < 8"));
            }
            (8, sz)
        }
    };

    if box_type == *b"uuid" {
        header_len = header_len
            .checked_add(16)
            .ok_or(Error::InvalidData("uuid kutu başlığı taşması"))?;
        if total_size < header_len {
            return Err(Error::InvalidData(
                "uuid kutu boyutu başlık boyutundan küçük",
            ));
        }
    }

    let next_offset = offset
        .checked_add(total_size)
        .ok_or(Error::InvalidData("ISOBMFF kutu bitiş ofseti taşması"))?;
    if next_offset > buf.len() {
        return Err(Error::InvalidData(
            "ISOBMFF kutu boyutu tampon sınırını aşıyor",
        ));
    }
    let payload_start = offset
        .checked_add(header_len)
        .ok_or(Error::InvalidData("ISOBMFF kutu yük başlangıcı taşması"))?;
    let payload = &buf[payload_start..next_offset];
    Ok(Some(RawBox {
        box_type,
        payload,
        next_offset,
    }))
}

/// `FullBox` (`version: u8`, `flags: u24`) başlığını ayrıştırır.
pub(crate) fn read_full_box(payload: &[u8]) -> Result<(u8, u32, &[u8])> {
    if payload.len() < 4 {
        return Err(Error::InvalidData(
            "kesik ISOBMFF FullBox başlığı (< 4 bayt)",
        ));
    }
    let version = payload[0];
    let flags = ((payload[1] as u32) << 16) | ((payload[2] as u32) << 8) | (payload[3] as u32);
    Ok((version, flags, &payload[4..]))
}

fn is_supported_brand(brand: &[u8; 4]) -> bool {
    SUPPORTED_BRANDS.contains(brand)
}

/// Verilen bayt diliminin geçerli bir AVIF (`ftyp`) başlığı taşıyıp taşımadığını sınar.
#[must_use]
pub fn probe(data: &[u8]) -> bool {
    let Ok(Some(first)) = read_box_header(data, 0) else {
        return false;
    };
    if first.box_type != *b"ftyp" || first.payload.len() < 8 {
        return false;
    }
    let major = [
        first.payload[0],
        first.payload[1],
        first.payload[2],
        first.payload[3],
    ];
    if is_supported_brand(&major) {
        return true;
    }
    for compat in first.payload[8..].as_chunks::<4>().0 {
        if is_supported_brand(compat) {
            return true;
        }
    }
    false
}

/// ISOBMFF / AVIF kapsayıcı yapısını ayrıştırır.
pub(crate) fn parse_container(data: &[u8]) -> Result<AvifContainer<'_>> {
    let mut offset = 0usize;
    let mut major_brand: Option<[u8; 4]> = None;
    let mut has_avif_brand = false;
    let mut meta_payload: Option<&[u8]> = None;

    while let Some(raw_box) = read_box_header(data, offset)? {
        offset = raw_box.next_offset;
        match &raw_box.box_type {
            b"ftyp" => {
                if raw_box.payload.len() < 8 {
                    return Err(Error::InvalidData("ftyp kutusu 8 bayttan kısa"));
                }
                let major = [
                    raw_box.payload[0],
                    raw_box.payload[1],
                    raw_box.payload[2],
                    raw_box.payload[3],
                ];
                major_brand = Some(major);
                if is_supported_brand(&major) {
                    has_avif_brand = true;
                }
                for compat in raw_box.payload[8..].as_chunks::<4>().0 {
                    if is_supported_brand(compat) {
                        has_avif_brand = true;
                    }
                }
            }
            b"meta" => {
                if meta_payload.is_some() {
                    return Err(Error::InvalidData("birden fazla üst düzey meta kutusu"));
                }
                meta_payload = Some(raw_box.payload);
            }
            _ => {}
        }
    }

    let major_brand = major_brand.ok_or(Error::InvalidData("AVIF dosyasında ftyp kutusu yok"))?;
    if !has_avif_brand {
        return Err(Error::InvalidData(
            "ftyp kutusu desteklenen bir AVIF/MIAF markası içermiyor",
        ));
    }
    let meta_raw = meta_payload.ok_or(Error::InvalidData("AVIF dosyasında meta kutusu yok"))?;
    let (meta_version, _meta_flags, meta_children) = read_full_box(meta_raw)?;
    if meta_version != 0 {
        return Err(Error::Unsupported(
            "meta kutusu sürümü 0 dışında desteklenmiyor",
        ));
    }

    let mut primary_item_id: Option<u32> = None;
    let mut items = Vec::new();
    let mut properties = Vec::new();
    let mut associations: Vec<(u32, Vec<PropertyAssoc>)> = Vec::new();
    let mut locations = Vec::new();
    let mut references = Vec::new();
    let mut idat: &[u8] = &[];
    let mut has_hdlr = false;

    let mut m_off = 0usize;
    while let Some(child) = read_box_header(meta_children, m_off)? {
        m_off = child.next_offset;
        match &child.box_type {
            b"hdlr" => {
                let (ver, _flags, body) = read_full_box(child.payload)?;
                if ver != 0 || body.len() < 8 {
                    return Err(Error::InvalidData("geçersiz hdlr kutusu"));
                }
                let handler_type = &body[4..8];
                if handler_type != b"pict" {
                    return Err(Error::InvalidData("meta hdlr türü 'pict' değil"));
                }
                has_hdlr = true;
            }
            b"pitm" => {
                let (ver, _flags, body) = read_full_box(child.payload)?;
                let id = match ver {
                    0 => u32::from(
                        read_u16_be(body, 0).ok_or(Error::InvalidData("kesik pitm v0 kutusu"))?,
                    ),
                    1 => read_u32_be(body, 0).ok_or(Error::InvalidData("kesik pitm v1 kutusu"))?,
                    _ => return Err(Error::Unsupported("desteklenmeyen pitm sürümü")),
                };
                primary_item_id = Some(id);
            }
            b"iinf" => {
                parse_iinf(child.payload, &mut items)?;
            }
            b"iprp" => {
                parse_iprp(child.payload, &mut properties, &mut associations)?;
            }
            b"iloc" => {
                parse_iloc(child.payload, &mut locations)?;
            }
            b"iref" => {
                parse_iref(child.payload, &mut references)?;
            }
            b"idat" => {
                idat = child.payload;
            }
            _ => {}
        }
    }

    if !has_hdlr {
        return Err(Error::InvalidData("meta kutusunda hdlr alt kutusu eksik"));
    }
    let primary_item_id =
        primary_item_id.ok_or(Error::InvalidData("meta kutusunda pitm alt kutusu eksik"))?;

    Ok(AvifContainer {
        file_data: data,
        major_brand,
        primary_item_id,
        items,
        properties,
        associations,
        locations,
        references,
        idat,
    })
}

fn parse_iinf(payload: &[u8], items: &mut Vec<ItemInfoEntry>) -> Result<()> {
    let (ver, _flags, body) = read_full_box(payload)?;
    let (entry_count, mut offset) = match ver {
        0 => (
            u32::from(
                read_u16_be(body, 0).ok_or(Error::InvalidData("kesik iinf v0 giriş sayısı"))?,
            ),
            2usize,
        ),
        1 => (
            read_u32_be(body, 0).ok_or(Error::InvalidData("kesik iinf v1 giriş sayısı"))?,
            4usize,
        ),
        _ => return Err(Error::Unsupported("desteklenmeyen iinf sürümü")),
    };

    let mut parsed = 0u32;
    while let Some(child) = read_box_header(body, offset)? {
        offset = child.next_offset;
        if &child.box_type == b"infe" {
            let entry = parse_infe(child.payload)?;
            items.push(entry);
            parsed += 1;
        }
    }
    if parsed < entry_count {
        return Err(Error::InvalidData(
            "iinf içindeki infe sayısı bildirilen değerden az",
        ));
    }
    Ok(())
}

fn parse_infe(payload: &[u8]) -> Result<ItemInfoEntry> {
    let (ver, flags, body) = read_full_box(payload)?;
    if !(2..=3).contains(&ver) {
        return Err(Error::Unsupported("yalnızca infe v2 ve v3 desteklenir"));
    }
    let hidden = (flags & 1) != 0;
    let (item_id, mut pos) = if ver == 2 {
        (
            u32::from(read_u16_be(body, 0).ok_or(Error::InvalidData("kesik infe v2 item_id"))?),
            2usize,
        )
    } else {
        (
            read_u32_be(body, 0).ok_or(Error::InvalidData("kesik infe v3 item_id"))?,
            4usize,
        )
    };
    // item_protection_index (u16) + item_type ([u8; 4])
    if body.len().saturating_sub(pos) < 6 {
        return Err(Error::InvalidData("kesik infe başlığı"));
    }
    pos += 2;
    let item_type = [body[pos], body[pos + 1], body[pos + 2], body[pos + 3]];
    pos += 4;

    let (item_name, next_pos) = if pos < body.len() {
        read_cstring(body, pos)?
    } else {
        (String::new(), pos)
    };
    pos = next_pos;

    let content_type = if item_type == *b"mime" && pos < body.len() {
        let (ct, _) = read_cstring(body, pos)?;
        ct
    } else {
        String::new()
    };

    Ok(ItemInfoEntry {
        item_id,
        item_type,
        item_name,
        content_type,
        hidden,
    })
}

fn parse_iprp(
    payload: &[u8],
    properties: &mut Vec<ItemProperty>,
    associations: &mut Vec<(u32, Vec<PropertyAssoc>)>,
) -> Result<()> {
    let mut offset = 0usize;
    let mut seen_ipco = false;
    while let Some(child) = read_box_header(payload, offset)? {
        offset = child.next_offset;
        match &child.box_type {
            b"ipco" => {
                if seen_ipco {
                    return Err(Error::InvalidData("iprp içinde birden fazla ipco kutusu"));
                }
                seen_ipco = true;
                parse_ipco(child.payload, properties)?;
            }
            b"ipma" => {
                parse_ipma(child.payload, associations)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_ipco(payload: &[u8], properties: &mut Vec<ItemProperty>) -> Result<()> {
    let mut offset = 0usize;
    while let Some(prop_box) = read_box_header(payload, offset)? {
        offset = prop_box.next_offset;
        let p = match &prop_box.box_type {
            b"ispe" => {
                let (ver, _flags, body) = read_full_box(prop_box.payload)?;
                if ver != 0 || body.len() < 8 {
                    return Err(Error::InvalidData("geçersiz ispe kutusu"));
                }
                let width = read_u32_be(body, 0).unwrap_or(0);
                let height = read_u32_be(body, 4).unwrap_or(0);
                ItemProperty::Ispe { width, height }
            }
            b"pixi" => {
                let (ver, _flags, body) = read_full_box(prop_box.payload)?;
                if ver != 0 || body.is_empty() {
                    return Err(Error::InvalidData("geçersiz pixi kutusu"));
                }
                let num_channels = body[0] as usize;
                if body.len() < 1 + num_channels {
                    return Err(Error::InvalidData("kesik pixi kanal listesi"));
                }
                ItemProperty::Pixi {
                    bits_per_channel: body[1..1 + num_channels].to_vec(),
                }
            }
            b"av1C" => {
                if prop_box.payload.len() < 4 {
                    return Err(Error::InvalidData("kesik av1C kutusu (< 4 bayt)"));
                }
                let b0 = prop_box.payload[0];
                if (b0 & 0x80) == 0 || (b0 & 0x7F) != 1 {
                    return Err(Error::InvalidData("geçersiz av1C marker veya sürüm"));
                }
                let b1 = prop_box.payload[1];
                let b2 = prop_box.payload[2];
                let seq_profile = (b1 >> 5) & 0x07;
                let seq_level_idx_0 = b1 & 0x1F;
                let seq_tier_0 = (b2 >> 7) & 0x01;
                let high_bitdepth = ((b2 >> 6) & 0x01) != 0;
                let twelve_bit = ((b2 >> 5) & 0x01) != 0;
                let monochrome = ((b2 >> 4) & 0x01) != 0;
                let chroma_subsampling_x = (b2 >> 3) & 0x01;
                let chroma_subsampling_y = (b2 >> 2) & 0x01;
                let chroma_sample_position = b2 & 0x03;
                let config_obus = prop_box.payload[4..].to_vec();
                ItemProperty::Av1C(Av1CProperty {
                    seq_profile,
                    seq_level_idx_0,
                    seq_tier_0,
                    high_bitdepth,
                    twelve_bit,
                    monochrome,
                    chroma_subsampling_x,
                    chroma_subsampling_y,
                    chroma_sample_position,
                    config_obus,
                })
            }
            b"colr" => {
                if prop_box.payload.len() < 4 {
                    return Err(Error::InvalidData("kesik colr kutusu"));
                }
                let colour_type = &prop_box.payload[0..4];
                if colour_type == b"nclx" {
                    if prop_box.payload.len() < 11 {
                        return Err(Error::InvalidData("kesik colr nclx kutusu (< 11 bayt)"));
                    }
                    let color_primaries = read_u16_be(prop_box.payload, 4).unwrap_or(2);
                    let transfer_characteristics = read_u16_be(prop_box.payload, 6).unwrap_or(2);
                    let matrix_coefficients = read_u16_be(prop_box.payload, 8).unwrap_or(2);
                    let full_range = (prop_box.payload[10] & 0x80) != 0;
                    ItemProperty::ColrNclx(ColorInfo {
                        color_primaries,
                        transfer_characteristics,
                        matrix_coefficients,
                        full_range,
                    })
                } else if colour_type == b"rICC" || colour_type == b"prof" {
                    ItemProperty::ColrIcc(prop_box.payload[4..].to_vec())
                } else {
                    ItemProperty::Unknown(*b"colr")
                }
            }
            b"clap" => {
                if prop_box.payload.len() < 32 {
                    return Err(Error::InvalidData("kesik clap kutusu (< 32 bayt)"));
                }
                let width_n = read_u32_be(prop_box.payload, 0).unwrap_or(0);
                let width_d = read_u32_be(prop_box.payload, 4).unwrap_or(0);
                let height_n = read_u32_be(prop_box.payload, 8).unwrap_or(0);
                let height_d = read_u32_be(prop_box.payload, 12).unwrap_or(0);
                let horiz_off_n = read_i32_be(prop_box.payload, 16).unwrap_or(0);
                let horiz_off_d = read_u32_be(prop_box.payload, 20).unwrap_or(0);
                let vert_off_n = read_i32_be(prop_box.payload, 24).unwrap_or(0);
                let vert_off_d = read_u32_be(prop_box.payload, 28).unwrap_or(0);
                if width_d == 0 || height_d == 0 || horiz_off_d == 0 || vert_off_d == 0 {
                    return Err(Error::InvalidData("clap kutusunda sıfır payda"));
                }
                ItemProperty::Clap(CleanAperture {
                    width_n,
                    width_d,
                    height_n,
                    height_d,
                    horiz_off_n,
                    horiz_off_d,
                    vert_off_n,
                    vert_off_d,
                })
            }
            b"irot" => {
                if prop_box.payload.is_empty() {
                    return Err(Error::InvalidData("kesik irot kutusu"));
                }
                ItemProperty::Irot(prop_box.payload[0] & 0x03)
            }
            b"imir" => {
                if prop_box.payload.is_empty() {
                    return Err(Error::InvalidData("kesik imir kutusu"));
                }
                ItemProperty::Imir(prop_box.payload[0] & 0x01)
            }
            b"pasp" => {
                if prop_box.payload.len() < 8 {
                    return Err(Error::InvalidData("kesik pasp kutusu"));
                }
                let h_spacing = read_u32_be(prop_box.payload, 0).unwrap_or(1);
                let v_spacing = read_u32_be(prop_box.payload, 4).unwrap_or(1);
                if h_spacing == 0 || v_spacing == 0 {
                    return Err(Error::InvalidData("pasp kutusunda sıfır aralık değeri"));
                }
                ItemProperty::Pasp(PixelAspectRatio {
                    h_spacing,
                    v_spacing,
                })
            }
            b"clli" => {
                if prop_box.payload.len() < 4 {
                    return Err(Error::InvalidData("kesik clli kutusu"));
                }
                let max_content_light_level = read_u16_be(prop_box.payload, 0).unwrap_or(0);
                let max_pic_average_light_level = read_u16_be(prop_box.payload, 2).unwrap_or(0);
                ItemProperty::Clli(ContentLightLevel {
                    max_content_light_level,
                    max_pic_average_light_level,
                })
            }
            b"mdcv" => {
                if prop_box.payload.len() < 24 {
                    return Err(Error::InvalidData("kesik mdcv kutusu (< 24 bayt)"));
                }
                let primaries = [
                    [
                        read_u16_be(prop_box.payload, 0).unwrap_or(0),
                        read_u16_be(prop_box.payload, 2).unwrap_or(0),
                    ],
                    [
                        read_u16_be(prop_box.payload, 4).unwrap_or(0),
                        read_u16_be(prop_box.payload, 6).unwrap_or(0),
                    ],
                    [
                        read_u16_be(prop_box.payload, 8).unwrap_or(0),
                        read_u16_be(prop_box.payload, 10).unwrap_or(0),
                    ],
                ];
                let white_point = [
                    read_u16_be(prop_box.payload, 12).unwrap_or(0),
                    read_u16_be(prop_box.payload, 14).unwrap_or(0),
                ];
                let max_luminance = read_u32_be(prop_box.payload, 16).unwrap_or(0);
                let min_luminance = read_u32_be(prop_box.payload, 20).unwrap_or(0);
                ItemProperty::Mdcv(MasteringDisplay {
                    primaries,
                    white_point,
                    max_luminance,
                    min_luminance,
                })
            }
            b"a1op" => {
                if prop_box.payload.is_empty() {
                    return Err(Error::InvalidData("kesik a1op kutusu"));
                }
                let op_index = prop_box.payload[0];
                if op_index > 31 {
                    return Err(Error::InvalidData("a1op operating_point_index > 31"));
                }
                ItemProperty::A1op(op_index)
            }
            b"lsel" => {
                if prop_box.payload.len() < 2 {
                    return Err(Error::InvalidData("kesik lsel kutusu"));
                }
                let layer_id = read_u16_be(prop_box.payload, 0).unwrap_or(0);
                ItemProperty::Lsel(layer_id)
            }
            b"auxC" => {
                let (ver, _flags, body) = read_full_box(prop_box.payload)?;
                if ver != 0 {
                    return Err(Error::Unsupported("auxC sürümü 0 dışında desteklenmiyor"));
                }
                let (aux_type, next_pos) = read_cstring(body, 0)?;
                let subtype = body.get(next_pos..).unwrap_or(&[]).to_vec();
                ItemProperty::AuxC { aux_type, subtype }
            }
            other => ItemProperty::Unknown(*other),
        };
        properties.push(p);
    }
    Ok(())
}

fn parse_ipma(payload: &[u8], associations: &mut Vec<(u32, Vec<PropertyAssoc>)>) -> Result<()> {
    let (ver, flags, body) = read_full_box(payload)?;
    if ver > 1 {
        return Err(Error::Unsupported("ipma sürümü > 1 desteklenmiyor"));
    }
    let entry_count = read_u32_be(body, 0).ok_or(Error::InvalidData("kesik ipma entry_count"))?;
    let mut pos = 4usize;
    let large_index = (flags & 1) != 0;

    for _ in 0..entry_count {
        let item_id = if ver == 0 {
            let id = read_u16_be(body, pos).ok_or(Error::InvalidData("kesik ipma v0 item_id"))?;
            pos += 2;
            u32::from(id)
        } else {
            let id = read_u32_be(body, pos).ok_or(Error::InvalidData("kesik ipma v1 item_id"))?;
            pos += 4;
            id
        };
        let assoc_count = *body
            .get(pos)
            .ok_or(Error::InvalidData("kesik ipma association_count"))?
            as usize;
        pos += 1;

        let mut list = Vec::with_capacity(assoc_count);
        for _ in 0..assoc_count {
            if large_index {
                let raw =
                    read_u16_be(body, pos).ok_or(Error::InvalidData("kesik ipma 16-bit indeks"))?;
                pos += 2;
                let essential = (raw & 0x8000) != 0;
                let property_index = raw & 0x7FFF;
                list.push(PropertyAssoc {
                    property_index,
                    essential,
                });
            } else {
                let raw = *body
                    .get(pos)
                    .ok_or(Error::InvalidData("kesik ipma 8-bit indeks"))?;
                pos += 1;
                let essential = (raw & 0x80) != 0;
                let property_index = u16::from(raw & 0x7F);
                list.push(PropertyAssoc {
                    property_index,
                    essential,
                });
            }
        }

        if let Some((_, existing)) = associations.iter_mut().find(|(id, _)| *id == item_id) {
            existing.extend_from_slice(&list);
        } else {
            associations.push((item_id, list));
        }
    }
    Ok(())
}

fn parse_iloc(payload: &[u8], locations: &mut Vec<IlocItem>) -> Result<()> {
    let (ver, _flags, body) = read_full_box(payload)?;
    if ver > 2 || body.len() < 2 {
        return Err(Error::Unsupported(
            "geçersiz veya desteklenmeyen iloc sürümü",
        ));
    }
    let offset_size = ((body[0] >> 4) & 0x0F) as usize;
    let length_size = (body[0] & 0x0F) as usize;
    let base_offset_size = ((body[1] >> 4) & 0x0F) as usize;
    let index_size = if ver == 1 || ver == 2 {
        (body[1] & 0x0F) as usize
    } else {
        0
    };
    for &sz in &[offset_size, length_size, base_offset_size, index_size] {
        if sz != 0 && sz != 4 && sz != 8 {
            return Err(Error::InvalidData("iloc alan boyutu 0, 4 veya 8 değil"));
        }
    }

    let (item_count, mut pos) = if ver < 2 {
        (
            u32::from(
                read_u16_be(body, 2).ok_or(Error::InvalidData("kesik iloc v0/v1 item_count"))?,
            ),
            4usize,
        )
    } else {
        (
            read_u32_be(body, 2).ok_or(Error::InvalidData("kesik iloc v2 item_count"))?,
            6usize,
        )
    };

    for _ in 0..item_count {
        let item_id = if ver < 2 {
            let id = read_u16_be(body, pos).ok_or(Error::InvalidData("kesik iloc item_id"))?;
            pos += 2;
            u32::from(id)
        } else {
            let id = read_u32_be(body, pos).ok_or(Error::InvalidData("kesik iloc v2 item_id"))?;
            pos += 4;
            id
        };

        let construction_method = if ver == 1 || ver == 2 {
            let tmp = read_u16_be(body, pos)
                .ok_or(Error::InvalidData("kesik iloc construction_method"))?;
            pos += 2;
            (tmp & 0x0F) as u8
        } else {
            0
        };
        let data_reference_index =
            read_u16_be(body, pos).ok_or(Error::InvalidData("kesik iloc data_reference_index"))?;
        pos += 2;

        let base_offset = read_uint_n(body, pos, base_offset_size)
            .ok_or(Error::InvalidData("kesik iloc base_offset"))?;
        pos += base_offset_size;

        let extent_count =
            read_u16_be(body, pos).ok_or(Error::InvalidData("kesik iloc extent_count"))? as usize;
        pos += 2;

        let mut extents = Vec::with_capacity(extent_count);
        for _ in 0..extent_count {
            let extent_index = if (ver == 1 || ver == 2) && index_size > 0 {
                let idx = read_uint_n(body, pos, index_size)
                    .ok_or(Error::InvalidData("kesik iloc extent_index"))?;
                pos += index_size;
                idx
            } else {
                0
            };
            let extent_offset = read_uint_n(body, pos, offset_size)
                .ok_or(Error::InvalidData("kesik iloc extent_offset"))?;
            pos += offset_size;
            let extent_length = read_uint_n(body, pos, length_size)
                .ok_or(Error::InvalidData("kesik iloc extent_length"))?;
            pos += length_size;
            extents.push(IlocExtent {
                extent_index,
                extent_offset,
                extent_length,
            });
        }

        locations.push(IlocItem {
            item_id,
            construction_method,
            data_reference_index,
            base_offset,
            extents,
        });
    }
    Ok(())
}

fn parse_iref(payload: &[u8], references: &mut Vec<ItemReference>) -> Result<()> {
    let (ver, _flags, body) = read_full_box(payload)?;
    if ver > 1 {
        return Err(Error::Unsupported("iref sürümü > 1 desteklenmiyor"));
    }
    let mut offset = 0usize;
    while let Some(ref_box) = read_box_header(body, offset)? {
        offset = ref_box.next_offset;
        let rbody = ref_box.payload;
        let mut pos = 0usize;
        let from_item_id = if ver == 0 {
            let id =
                read_u16_be(rbody, pos).ok_or(Error::InvalidData("kesik iref v0 from_item_id"))?;
            pos += 2;
            u32::from(id)
        } else {
            let id =
                read_u32_be(rbody, pos).ok_or(Error::InvalidData("kesik iref v1 from_item_id"))?;
            pos += 4;
            id
        };
        let ref_count = read_u16_be(rbody, pos)
            .ok_or(Error::InvalidData("kesik iref reference_count"))?
            as usize;
        pos += 2;

        let mut to_item_ids = Vec::with_capacity(ref_count);
        for _ in 0..ref_count {
            let to_id = if ver == 0 {
                let id = read_u16_be(rbody, pos)
                    .ok_or(Error::InvalidData("kesik iref v0 to_item_id"))?;
                pos += 2;
                u32::from(id)
            } else {
                let id = read_u32_be(rbody, pos)
                    .ok_or(Error::InvalidData("kesik iref v1 to_item_id"))?;
                pos += 4;
                id
            };
            to_item_ids.push(to_id);
        }
        references.push(ItemReference {
            ref_type: ref_box.box_type,
            from_item_id,
            to_item_ids,
        });
    }
    Ok(())
}

impl<'a> AvifContainer<'a> {
    /// Belirtilen `item_id` için `infe` kaydını döndürür.
    pub(crate) fn item_info(&self, item_id: u32) -> Option<&ItemInfoEntry> {
        self.items.iter().find(|i| i.item_id == item_id)
    }

    /// Belirtilen `item_id` ile ilişkilendirilmiş özellikleri (`ipma` sırasıyla) döndürür.
    pub(crate) fn item_properties(&self, item_id: u32) -> Result<Vec<(&ItemProperty, bool)>> {
        let Some((_, assocs)) = self.associations.iter().find(|(id, _)| *id == item_id) else {
            return Ok(Vec::new());
        };
        let mut out = Vec::with_capacity(assocs.len());
        for assoc in assocs {
            if assoc.property_index == 0 {
                continue;
            }
            let idx = (assoc.property_index as usize) - 1;
            let prop = self.properties.get(idx).ok_or(Error::InvalidData(
                "ipma özellik indeksi ipco sınırları dışında",
            ))?;
            if assoc.essential && matches!(prop, ItemProperty::Unknown(_)) {
                return Err(Error::Unsupported(
                    "öğede desteklenmeyen zorunlu (essential) ipco özelliği bulunuyor",
                ));
            }
            out.push((prop, assoc.essential));
        }
        Ok(out)
    }

    /// Belirtilen `from_item_id` ve `ref_type` için hedef öğe kimliklerini döndürür.
    pub(crate) fn references_from(&self, from_item_id: u32, ref_type: &[u8; 4]) -> Vec<u32> {
        let mut targets = Vec::new();
        for r in &self.references {
            if r.from_item_id == from_item_id && &r.ref_type == ref_type {
                targets.extend_from_slice(&r.to_item_ids);
            }
        }
        targets
    }

    /// Belirtilen `to_item_id` ve `ref_type` için kaynak öğe kimliklerini döndürür.
    pub(crate) fn references_to(&self, to_item_id: u32, ref_type: &[u8; 4]) -> Vec<u32> {
        let mut sources = Vec::new();
        for r in &self.references {
            if &r.ref_type == ref_type && r.to_item_ids.contains(&to_item_id) {
                sources.push(r.from_item_id);
            }
        }
        sources
    }

    /// Belirtilen öğenin bayt yükünü `iloc` üzerinden çözümler.
    pub(crate) fn resolve_item_data(
        &self,
        item_id: u32,
        allocation_limit: Option<usize>,
    ) -> Result<Cow<'a, [u8]>> {
        self.resolve_item_data_inner(item_id, allocation_limit, 0)
    }

    fn resolve_item_data_inner(
        &self,
        item_id: u32,
        allocation_limit: Option<usize>,
        depth: usize,
    ) -> Result<Cow<'a, [u8]>> {
        if depth > 4 {
            return Err(Error::InvalidData(
                "iloc item_offset zinciri derinlik sınırını aştı",
            ));
        }
        let loc = self
            .locations
            .iter()
            .find(|l| l.item_id == item_id)
            .ok_or(Error::InvalidData("öğe için iloc konum kaydı bulunamadı"))?;
        if loc.data_reference_index != 0 {
            return Err(Error::Unsupported(
                "harici dosya referanslı (data_reference_index != 0) iloc desteklenmiyor",
            ));
        }
        if loc.extents.is_empty() {
            return Err(Error::InvalidData("iloc kaydında extent bulunmuyor"));
        }

        match loc.construction_method {
            0 | 1 => {
                let source = if loc.construction_method == 0 {
                    self.file_data
                } else {
                    self.idat
                };
                if loc.extents.len() == 1 {
                    let ext = &loc.extents[0];
                    let start64 = loc
                        .base_offset
                        .checked_add(ext.extent_offset)
                        .ok_or(Error::InvalidData("iloc ofset taşması"))?;
                    let start = usize::try_from(start64)
                        .map_err(|_| Error::InvalidData("iloc ofset usize sınırını aşıyor"))?;
                    let len = usize::try_from(ext.extent_length)
                        .map_err(|_| Error::InvalidData("iloc uzunluk usize sınırını aşıyor"))?;
                    let end = start
                        .checked_add(len)
                        .ok_or(Error::InvalidData("iloc bitiş ofseti taşması"))?;
                    let slice = source.get(start..end).ok_or(Error::InvalidData(
                        "iloc extent dosya/idat sınırları dışında",
                    ))?;
                    Ok(Cow::Borrowed(slice))
                } else {
                    let mut total_len = 0usize;
                    for ext in &loc.extents {
                        let len = usize::try_from(ext.extent_length).map_err(|_| {
                            Error::InvalidData("iloc extent uzunluğu usize sınırını aşıyor")
                        })?;
                        total_len = total_len
                            .checked_add(len)
                            .ok_or(Error::InvalidData("iloc toplam extent uzunluğu taşması"))?;
                    }
                    if let Some(limit) = allocation_limit
                        && total_len > limit
                    {
                        return Err(Error::BudgetExceeded {
                            requested: total_len,
                            limit,
                        });
                    }
                    let mut buf = Vec::new();
                    buf.try_reserve_exact(total_len)
                        .map_err(|_| Error::AllocationFailed {
                            requested: total_len,
                        })?;
                    for ext in &loc.extents {
                        let start64 = loc
                            .base_offset
                            .checked_add(ext.extent_offset)
                            .ok_or(Error::InvalidData("iloc ofset taşması"))?;
                        let start = usize::try_from(start64)
                            .map_err(|_| Error::InvalidData("iloc ofset usize sınırını aşıyor"))?;
                        let len = usize::try_from(ext.extent_length).map_err(|_| {
                            Error::InvalidData("iloc uzunluk usize sınırını aşıyor")
                        })?;
                        let end = start
                            .checked_add(len)
                            .ok_or(Error::InvalidData("iloc bitiş ofseti taşması"))?;
                        let slice = source.get(start..end).ok_or(Error::InvalidData(
                            "iloc extent dosya/idat sınırları dışında",
                        ))?;
                        buf.extend_from_slice(slice);
                    }
                    Ok(Cow::Owned(buf))
                }
            }
            2 => {
                let dimg_targets = self.references_from(item_id, b"iloc");
                let mut total_len = 0usize;
                for ext in &loc.extents {
                    let len = usize::try_from(ext.extent_length).map_err(|_| {
                        Error::InvalidData("iloc extent uzunluğu usize sınırını aşıyor")
                    })?;
                    total_len = total_len
                        .checked_add(len)
                        .ok_or(Error::InvalidData("iloc toplam extent uzunluğu taşması"))?;
                }
                if let Some(limit) = allocation_limit
                    && total_len > limit
                {
                    return Err(Error::BudgetExceeded {
                        requested: total_len,
                        limit,
                    });
                }
                let mut buf = Vec::new();
                buf.try_reserve_exact(total_len)
                    .map_err(|_| Error::AllocationFailed {
                        requested: total_len,
                    })?;
                for ext in &loc.extents {
                    let idx = if ext.extent_index > 0 {
                        usize::try_from(ext.extent_index - 1)
                            .map_err(|_| Error::InvalidData("geçersiz iloc extent_index"))?
                    } else {
                        0
                    };
                    let target_id = *dimg_targets.get(idx).ok_or(Error::InvalidData(
                        "iloc construction_method=2 hedef öğesi yok",
                    ))?;
                    let target_data =
                        self.resolve_item_data_inner(target_id, allocation_limit, depth + 1)?;
                    let start64 = loc
                        .base_offset
                        .checked_add(ext.extent_offset)
                        .ok_or(Error::InvalidData("iloc ofset taşması"))?;
                    let start = usize::try_from(start64)
                        .map_err(|_| Error::InvalidData("iloc ofset usize sınırını aşıyor"))?;
                    let len = usize::try_from(ext.extent_length)
                        .map_err(|_| Error::InvalidData("iloc uzunluk usize sınırını aşıyor"))?;
                    let end = start
                        .checked_add(len)
                        .ok_or(Error::InvalidData("iloc bitiş ofseti taşması"))?;
                    let slice = target_data.get(start..end).ok_or(Error::InvalidData(
                        "iloc item_offset hedef öğe sınırları dışında",
                    ))?;
                    buf.extend_from_slice(slice);
                }
                Ok(Cow::Owned(buf))
            }
            _ => Err(Error::Unsupported(
                "desteklenmeyen iloc construction_method değeri",
            )),
        }
    }
}
