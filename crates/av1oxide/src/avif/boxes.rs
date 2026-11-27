//! ISOBMFF (`ISO/IEC 14496-12`), HEIF/MIAF (`ISO/IEC 23008-12`) ve AVIF (`v1.1.0`)
//! kutu ayrıştırıcısı.

#![forbid(unsafe_code)]

use alloc::borrow::{Cow, ToOwned};
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::io::Read;

#[cfg(feature = "std")]
use flate2::read::{DeflateDecoder, GzDecoder, ZlibDecoder};

use crate::avif::image::{
    CleanAperture, ColorInfo, ContentLightLevel, MasteringDisplay, PixelAspectRatio,
};
use crate::error::{Error, Result};
use crate::frame::butce::{Butce, ButceTutucu, kopya_vec};

/// AVIF / HEIF dosya imzasında kabul edilen `ftyp` marka kodları.
const SUPPORTED_BRANDS: &[[u8; 4]] = &[
    *b"avif", *b"avis", *b"avio", *b"mif1", *b"msf1", *b"miaf", *b"MA1B", *b"MA1A",
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
    /// `cmpC`: ISO/IEC 23008-12 sıkıştırma yapılandırma özelliği (`defl`, `zlib`, `gzip`).
    CmpC { compression_type: [u8; 4] },
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
    pub(crate) content_encoding: String,
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

/// `grpl` kutusu içindeki varlık grubu (`EntityToGroupBox`) kaydı.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EntityToGroup {
    pub(crate) grouping_type: [u8; 4],
    pub(crate) group_id: u32,
    pub(crate) entity_ids: Vec<u32>,
}

/// Ekleme sırasını koruyan deterministik `FxBuildHasher` tabanlı harita.
pub(crate) type FxIndexMap<K, V> = indexmap::IndexMap<K, V, rustc_hash::FxBuildHasher>;

/// Ekleme sırasını koruyan deterministik `FxBuildHasher` tabanlı küme.
pub(crate) type FxIndexSet<T> = indexmap::IndexSet<T, rustc_hash::FxBuildHasher>;

/// `idat` kutusu depolaması: normal `meta` kutusunda dosya içi ofset aralığı (`0` kopya,
/// öz-referanssız), `cmin` sıkıştırılmış `meta` kutusunda sahipli bayt vektörü.
#[derive(Clone, Debug)]
pub(crate) enum IdatStorage {
    Empty,
    FileRange(core::ops::Range<usize>),
    Owned(Vec<u8>),
}

/// Ayrıştırılmış AVIF / ISOBMFF kapsayıcı yapısı.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct AvifContainer<'a> {
    pub(crate) file_data: Cow<'a, [u8]>,
    pub(crate) major_brand: [u8; 4],
    pub(crate) compatible_brands: Vec<[u8; 4]>,
    pub(crate) primary_item_id: u32,
    pub(crate) items: FxIndexMap<u32, ItemInfoEntry>,
    pub(crate) properties: Vec<ItemProperty>,
    pub(crate) associations: FxIndexMap<u32, Vec<PropertyAssoc>>,
    pub(crate) locations: FxIndexMap<u32, IlocItem>,
    pub(crate) references: Vec<ItemReference>,
    pub(crate) entity_groups: Vec<EntityToGroup>,
    pub(crate) idat: IdatStorage,
    pub(crate) owned_bytes: usize,
    pub(crate) _guard: Arc<ButceTutucu>,
}

/// Tek bir sıkıştırılmış HEIF öğesi veya `cmin` kutusu için azami açılmış boyut (`64 MiB`).
#[cfg(feature = "std")]
const MAX_DECOMPRESSED_BYTES: usize = 64 * 1024 * 1024;
/// Sıkıştırma oranı denetiminin devreye girdiği alt eşik (`64 KiB`).
#[cfg(feature = "std")]
const ZIP_BOMB_THRESHOLD_BYTES: usize = 64 * 1024;
/// `ZIP_BOMB_THRESHOLD_BYTES` üzerinde izin verilen azami genişleme oranı (`1024:1`).
#[cfg(feature = "std")]
const MAX_EXPANSION_RATIO: usize = 1024;

#[cfg(feature = "std")]
fn decompress_bounded<R: Read>(
    mut reader: R,
    compressed_len: usize,
    butce: &Arc<Butce>,
) -> Result<(Vec<u8>, ButceTutucu)> {
    let ratio_limit = compressed_len
        .saturating_mul(MAX_EXPANSION_RATIO)
        .max(ZIP_BOMB_THRESHOLD_BYTES);
    let hard_cap = MAX_DECOMPRESSED_BYTES.min(ratio_limit);
    let mut out = Vec::new();
    let mut guard = ButceTutucu::bos(butce);
    let mut chunk = [0u8; 8192];

    loop {
        let budget_rem = butce.kalan();
        let cap_rem = hard_cap.saturating_sub(out.len());
        let step_limit = budget_rem.map_or(cap_rem, |r| cap_rem.min(r));
        if step_limit == 0 {
            let mut probe = [0u8; 1];
            let extra = reader.read(&mut probe).map_err(|_| {
                Error::InvalidData("sıkıştırılmış HEIF verisi bozuk veya sağlama özeti hatalı")
            })?;
            if extra > 0 {
                if budget_rem == Some(0) {
                    guard.ekle(1)?;
                }
                return Err(Error::BudgetExceeded {
                    requested: hard_cap.saturating_add(1),
                    limit: hard_cap,
                });
            }
            break;
        }

        let to_read = chunk.len().min(step_limit);
        let n = reader.read(&mut chunk[..to_read]).map_err(|_| {
            Error::InvalidData("sıkıştırılmış HEIF verisi bozuk veya sağlama özeti hatalı")
        })?;
        if n == 0 {
            break;
        }
        guard.ekle(n)?;
        let new_len = out.len().saturating_add(n);
        out.try_reserve(n)
            .map_err(|_| Error::AllocationFailed { requested: new_len })?;
        out.extend_from_slice(&chunk[..n]);
    }
    Ok((out, guard))
}

#[cfg(feature = "std")]
fn decompress_zlib(
    data: &[u8],
    orig_compressed_len: usize,
    butce: &Arc<Butce>,
) -> Result<(Vec<u8>, ButceTutucu)> {
    decompress_bounded(ZlibDecoder::new(data), orig_compressed_len, butce)
}

#[cfg(feature = "std")]
fn decompress_deflate(
    data: &[u8],
    orig_compressed_len: usize,
    butce: &Arc<Butce>,
) -> Result<(Vec<u8>, ButceTutucu)> {
    decompress_bounded(DeflateDecoder::new(data), orig_compressed_len, butce)
}

#[cfg(feature = "std")]
fn decompress_gzip(
    data: &[u8],
    orig_compressed_len: usize,
    butce: &Arc<Butce>,
) -> Result<(Vec<u8>, ButceTutucu)> {
    decompress_bounded(GzDecoder::new(data), orig_compressed_len, butce)
}

pub(crate) fn decompress_by_fourcc(
    compression_type: &[u8; 4],
    data: &[u8],
    orig_compressed_len: usize,
    butce: &Arc<Butce>,
) -> Result<(Vec<u8>, ButceTutucu)> {
    #[cfg(feature = "std")]
    {
        match compression_type {
            b"defl" => decompress_deflate(data, orig_compressed_len, butce),
            b"zlib" => decompress_zlib(data, orig_compressed_len, butce),
            b"gzip" => decompress_gzip(data, orig_compressed_len, butce),
            _ => Err(Error::Unsupported(
                "desteklenmeyen cmpC sıkıştırma algoritması",
            )),
        }
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = (compression_type, data, orig_compressed_len, butce);
        Err(Error::Unsupported(
            "compressed HEIF boxes require the std feature",
        ))
    }
}

pub(crate) fn read_u16_be(buf: &[u8], offset: usize) -> Option<u16> {
    let bytes = buf.get(offset..offset.checked_add(2)?)?;
    Some(u16::from_be_bytes([bytes[0], bytes[1]]))
}

pub(crate) fn read_u32_be(buf: &[u8], offset: usize) -> Option<u32> {
    let bytes = buf.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

pub(crate) fn read_i32_be(buf: &[u8], offset: usize) -> Option<i32> {
    let bytes = buf.get(offset..offset.checked_add(4)?)?;
    Some(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

pub(crate) fn read_u64_be(buf: &[u8], offset: usize) -> Option<u64> {
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
    let nul_pos =
        memchr::memchr(0, tail).ok_or(Error::InvalidData("ISOBMFF dizesi NUL ile sonlanmıyor"))?;
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

pub(crate) fn is_supported_brand(brand: &[u8; 4]) -> bool {
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

enum MetaSource {
    FileRange(core::ops::Range<usize>),
    Owned(Vec<u8>),
}

/// Ödünç alınan dilim üzerinden ISOBMFF / AVIF kapsayıcı yapısını ayrıştırır.
pub(crate) fn parse_container<'a>(data: &'a [u8], butce: &Arc<Butce>) -> Result<AvifContainer<'a>> {
    parse_container_cow(Cow::Borrowed(data), butce)
}

/// Sahipli `Vec<u8>` tamponu üzerinden `'static` ömürlü AVIF kapsayıcı yapısını tek geçişte ayrıştırır.
pub(crate) fn parse_container_owned(
    data: Vec<u8>,
    butce: &Arc<Butce>,
) -> Result<AvifContainer<'static>> {
    parse_container_cow(Cow::Owned(data), butce)
}

fn parse_container_cow<'a>(data: Cow<'a, [u8]>, butce: &Arc<Butce>) -> Result<AvifContainer<'a>> {
    let mut offset = 0usize;
    let mut major_brand: Option<[u8; 4]> = None;
    let mut compatible_brands: Vec<[u8; 4]> = Vec::new();
    let mut ftyp_guard = ButceTutucu::bos(butce);
    let mut has_avif_brand = false;
    let mut meta_source: Option<MetaSource> = None;
    let mut _cmin_guard: Option<ButceTutucu> = None;

    {
        let slice: &[u8] = &data;
        let base_ptr = slice.as_ptr() as usize;
        while let Some(raw_box) = read_box_header(slice, offset)? {
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
                    let compat_chunks = raw_box.payload[8..].as_chunks::<4>().0;
                    let compat_bytes = compat_chunks
                        .len()
                        .saturating_mul(core::mem::size_of::<[u8; 4]>());
                    ftyp_guard.ekle(compat_bytes)?;
                    compatible_brands
                        .try_reserve(compat_chunks.len())
                        .map_err(|_| Error::AllocationFailed {
                            requested: compat_bytes,
                        })?;
                    for &compat in compat_chunks {
                        compatible_brands.push(compat);
                        if is_supported_brand(&compat) {
                            has_avif_brand = true;
                        }
                    }
                }
                b"meta" => {
                    if meta_source.is_some() {
                        return Err(Error::InvalidData("birden fazla üst düzey meta kutusu"));
                    }
                    let start = (raw_box.payload.as_ptr() as usize) - base_ptr;
                    let end = start + raw_box.payload.len();
                    meta_source = Some(MetaSource::FileRange(start..end));
                }
                b"cmin" => {
                    if meta_source.is_some() {
                        return Err(Error::InvalidData("birden fazla üst düzey meta kutusu"));
                    }
                    let (ver, _flags, body) = read_full_box(raw_box.payload)?;
                    if ver != 0 || body.len() < 4 {
                        return Err(Error::InvalidData("geçersiz cmin kutusu"));
                    }
                    let comp_type = [body[0], body[1], body[2], body[3]];
                    let payload_slice = &body[4..];
                    let (decompressed, guard) = decompress_by_fourcc(
                        &comp_type,
                        payload_slice,
                        payload_slice.len(),
                        butce,
                    )?;
                    meta_source = Some(MetaSource::Owned(decompressed));
                    _cmin_guard = Some(guard);
                }
                _ => {}
            }
        }
    }

    let major_brand = major_brand.ok_or(Error::InvalidData("AVIF dosyasında ftyp kutusu yok"))?;
    if !has_avif_brand {
        return Err(Error::InvalidData(
            "ftyp kutusu desteklenen bir AVIF/MIAF markası içermiyor",
        ));
    }
    let meta_src = meta_source.ok_or(Error::InvalidData("AVIF dosyasında meta kutusu yok"))?;
    parse_meta_box_cow(
        data,
        meta_src,
        major_brand,
        compatible_brands,
        Some(ftyp_guard),
        true,
        butce,
    )
}

/// Bir `meta` kutusu yükünü (üst düzey veya `trak` içi gömülü) [`AvifContainer`] olarak ayrıştırır.
pub(crate) fn parse_meta_box<'a>(
    file_data: &'a [u8],
    meta_raw: Cow<'a, [u8]>,
    major_brand: [u8; 4],
    compatible_brands: Vec<[u8; 4]>,
    require_pitm: bool,
    butce: &Arc<Butce>,
) -> Result<AvifContainer<'a>> {
    let meta_src = match meta_raw {
        Cow::Borrowed(s) => {
            let base_ptr = file_data.as_ptr() as usize;
            let start = (s.as_ptr() as usize) - base_ptr;
            MetaSource::FileRange(start..start + s.len())
        }
        Cow::Owned(v) => MetaSource::Owned(v),
    };
    parse_meta_box_cow(
        Cow::Borrowed(file_data),
        meta_src,
        major_brand,
        compatible_brands,
        None,
        require_pitm,
        butce,
    )
}

fn parse_meta_box_cow<'a>(
    file_data: Cow<'a, [u8]>,
    meta_src: MetaSource,
    major_brand: [u8; 4],
    compatible_brands: Vec<[u8; 4]>,
    initial_guard: Option<ButceTutucu>,
    require_pitm: bool,
    butce: &Arc<Butce>,
) -> Result<AvifContainer<'a>> {
    let mut primary_item_id: Option<u32> = None;
    let mut items: FxIndexMap<u32, ItemInfoEntry> =
        FxIndexMap::with_hasher(rustc_hash::FxBuildHasher);
    let mut properties: Vec<ItemProperty> = Vec::new();
    let mut associations: FxIndexMap<u32, Vec<PropertyAssoc>> =
        FxIndexMap::with_hasher(rustc_hash::FxBuildHasher);
    let mut locations: FxIndexMap<u32, IlocItem> =
        FxIndexMap::with_hasher(rustc_hash::FxBuildHasher);
    let mut references: Vec<ItemReference> = Vec::new();
    let mut entity_groups: Vec<EntityToGroup> = Vec::new();
    let mut seen_grpl = false;
    let mut idat = IdatStorage::Empty;
    let mut idat_owned_bytes = 0usize;
    let mut container_guard = match initial_guard {
        Some(g) => g,
        None => {
            let mut g = ButceTutucu::bos(butce);
            let brand_bytes = compatible_brands
                .len()
                .saturating_mul(core::mem::size_of::<[u8; 4]>());
            g.ekle(brand_bytes)?;
            g
        }
    };
    let mut has_hdlr = false;

    {
        let (meta_slice, is_borrowed): (&[u8], bool) = match &meta_src {
            MetaSource::FileRange(r) => (&file_data[r.clone()], true),
            MetaSource::Owned(v) => (v.as_slice(), false),
        };
        let (meta_version, _meta_flags, meta_children) = read_full_box(meta_slice)?;
        if meta_version != 0 {
            return Err(Error::Unsupported(
                "meta kutusu sürümü 0 dışında desteklenmiyor",
            ));
        }

        let base_ptr = file_data.as_ptr() as usize;
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
                    if require_pitm && handler_type != b"pict" {
                        return Err(Error::InvalidData("meta hdlr türü 'pict' değil"));
                    }
                    has_hdlr = true;
                }
                b"pitm" => {
                    let (ver, _flags, body) = read_full_box(child.payload)?;
                    let id = match ver {
                        0 => u32::from(
                            read_u16_be(body, 0)
                                .ok_or(Error::InvalidData("kesik pitm v0 kutusu"))?,
                        ),
                        1 => read_u32_be(body, 0)
                            .ok_or(Error::InvalidData("kesik pitm v1 kutusu"))?,
                        _ => return Err(Error::Unsupported("desteklenmeyen pitm sürümü")),
                    };
                    primary_item_id = Some(id);
                }
                b"iinf" => {
                    parse_iinf(child.payload, &mut items, &mut container_guard)?;
                }
                b"iprp" => {
                    parse_iprp(
                        child.payload,
                        &mut properties,
                        &mut associations,
                        &mut container_guard,
                    )?;
                }
                b"iloc" => {
                    parse_iloc(child.payload, &mut locations, &mut container_guard)?;
                }
                b"iref" => {
                    parse_iref(child.payload, &mut references, &mut container_guard)?;
                }
                b"grpl" => {
                    if seen_grpl {
                        return Err(Error::InvalidData("meta içinde birden fazla grpl kutusu"));
                    }
                    seen_grpl = true;
                    parse_grpl(child.payload, &mut entity_groups, &mut container_guard)?;
                }
                b"idat" => {
                    if idat_owned_bytes > 0 {
                        container_guard.dusur(idat_owned_bytes);
                        idat_owned_bytes = 0;
                    }
                    idat = if is_borrowed {
                        let child_ptr = child.payload.as_ptr() as usize;
                        let off = child_ptr - base_ptr;
                        IdatStorage::FileRange(off..off + child.payload.len())
                    } else {
                        let len = child.payload.len();
                        container_guard.ekle(len)?;
                        let v = kopya_vec(child.payload).inspect_err(|_| {
                            container_guard.dusur(len);
                        })?;
                        idat_owned_bytes = len;
                        IdatStorage::Owned(v)
                    };
                }
                _ => {}
            }
        }
    }

    if !has_hdlr {
        return Err(Error::InvalidData("meta kutusunda hdlr alt kutusu eksik"));
    }
    let primary_item_id = if require_pitm {
        primary_item_id.ok_or(Error::InvalidData("meta kutusunda pitm alt kutusu eksik"))?
    } else {
        primary_item_id.unwrap_or(0)
    };
    let owned_bytes = container_guard.bayt();

    Ok(AvifContainer {
        file_data,
        major_brand,
        compatible_brands,
        primary_item_id,
        items,
        properties,
        associations,
        locations,
        references,
        entity_groups,
        idat,
        owned_bytes,
        _guard: Arc::new(container_guard),
    })
}

fn parse_grpl(
    payload: &[u8],
    groups: &mut Vec<EntityToGroup>,
    guard: &mut ButceTutucu,
) -> Result<()> {
    let mut offset = 0usize;
    while let Some(child) = read_box_header(payload, offset)? {
        offset = child.next_offset;
        let (_ver, _flags, body) = read_full_box(child.payload)?;
        let group_id = read_u32_be(body, 0).ok_or(Error::InvalidData("kesik grpl group_id"))?;
        let num_entities = read_u32_be(body, 4)
            .ok_or(Error::InvalidData("kesik grpl num_entities_in_group"))?
            as usize;
        let rem_bytes = body.len().saturating_sub(8);
        let needed_bytes = num_entities
            .checked_mul(4)
            .ok_or(Error::InvalidData("grpl entity sayısı taşması"))?;
        if needed_bytes > rem_bytes {
            return Err(Error::InvalidData("kesik grpl varlık kimlik listesi"));
        }
        let alloc_bytes = core::mem::size_of::<EntityToGroup>()
            .saturating_add(num_entities.saturating_mul(core::mem::size_of::<u32>()));
        guard.ekle(alloc_bytes)?;
        let mut entity_ids = Vec::new();
        entity_ids
            .try_reserve_exact(num_entities)
            .map_err(|_| Error::AllocationFailed {
                requested: alloc_bytes,
            })?;
        for i in 0..num_entities {
            let eid =
                read_u32_be(body, 8 + i * 4).ok_or(Error::InvalidData("kesik grpl entity_id"))?;
            entity_ids.push(eid);
        }
        groups.push(EntityToGroup {
            grouping_type: child.box_type,
            group_id,
            entity_ids,
        });
    }
    Ok(())
}

fn parse_iinf(
    payload: &[u8],
    items: &mut FxIndexMap<u32, ItemInfoEntry>,
    guard: &mut ButceTutucu,
) -> Result<()> {
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

    // Her `infe` (v2/v3) kutusu en az 8 (box) + 4 (fullbox) + 2 (id) + 2 (prot) + 4 (type) = 20 bayttır.
    let max_possible = body.len().saturating_sub(offset) / 20;
    if (entry_count as usize) > max_possible {
        return Err(Error::InvalidData(
            "iinf içindeki infe sayısı kutu boyutunu aşıyor",
        ));
    }
    let reserve_n = (entry_count as usize).min(256);
    items
        .try_reserve(reserve_n)
        .map_err(|_| Error::AllocationFailed {
            requested: reserve_n.saturating_mul(core::mem::size_of::<(u32, ItemInfoEntry)>()),
        })?;
    let mut parsed = 0u32;
    while let Some(child) = read_box_header(body, offset)? {
        offset = child.next_offset;
        if &child.box_type == b"infe" {
            let entry = parse_infe(child.payload)?;
            let entry_bytes = core::mem::size_of::<(u32, ItemInfoEntry)>()
                .saturating_add(entry.item_name.len())
                .saturating_add(entry.content_type.len())
                .saturating_add(entry.content_encoding.len());
            guard.ekle(entry_bytes)?;
            if let Some(prev) = items.insert(entry.item_id, entry) {
                let prev_bytes = core::mem::size_of::<(u32, ItemInfoEntry)>()
                    .saturating_add(prev.item_name.len())
                    .saturating_add(prev.content_type.len())
                    .saturating_add(prev.content_encoding.len());
                guard.dusur(prev_bytes);
            }
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

    let (content_type, content_encoding) = if item_type == *b"mime" && pos < body.len() {
        let (ct, next_pos) = read_cstring(body, pos)?;
        let ce = if next_pos < body.len() {
            let (enc, _) = read_cstring(body, next_pos)?;
            enc
        } else {
            String::new()
        };
        (ct, ce)
    } else {
        (String::new(), String::new())
    };

    Ok(ItemInfoEntry {
        item_id,
        item_type,
        item_name,
        content_type,
        content_encoding,
        hidden,
    })
}

fn parse_iprp(
    payload: &[u8],
    properties: &mut Vec<ItemProperty>,
    associations: &mut FxIndexMap<u32, Vec<PropertyAssoc>>,
    guard: &mut ButceTutucu,
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
                parse_ipco(child.payload, properties, guard)?;
            }
            b"ipma" => {
                parse_ipma(child.payload, associations, guard)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_ipco(
    payload: &[u8],
    properties: &mut Vec<ItemProperty>,
    guard: &mut ButceTutucu,
) -> Result<()> {
    parse_property_boxes(payload, properties, false, guard)
}

/// `ipco` veya `stsd` `av01` örnek girişi içindeki alt özellik kutularını ayrıştırır.
pub(crate) fn parse_property_boxes(
    payload: &[u8],
    properties: &mut Vec<ItemProperty>,
    is_track: bool,
    guard: &mut ButceTutucu,
) -> Result<()> {
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
                let config_slice = &prop_box.payload[4..];
                let config_len = config_slice.len();
                guard.ekle(config_len)?;
                let config_obus = kopya_vec(config_slice).inspect_err(|_| {
                    guard.dusur(config_len);
                })?;
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
                    let icc_slice = &prop_box.payload[4..];
                    let icc_len = icc_slice.len();
                    guard.ekle(icc_len)?;
                    let icc_vec = kopya_vec(icc_slice).inspect_err(|_| {
                        guard.dusur(icc_len);
                    })?;
                    ItemProperty::ColrIcc(icc_vec)
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
            b"auxC" if !is_track => {
                let (ver, _flags, body) = read_full_box(prop_box.payload)?;
                if ver != 0 {
                    return Err(Error::Unsupported("auxC sürümü 0 dışında desteklenmiyor"));
                }
                let (aux_type, next_pos) = read_cstring(body, 0)?;
                let subtype = body.get(next_pos..).unwrap_or(&[]).to_vec();
                ItemProperty::AuxC { aux_type, subtype }
            }
            b"auxi" if is_track => {
                let (ver, _flags, body) = read_full_box(prop_box.payload)?;
                if ver != 0 {
                    return Err(Error::Unsupported("auxi sürümü 0 dışında desteklenmiyor"));
                }
                let (aux_type, next_pos) = read_cstring(body, 0)?;
                let subtype = body.get(next_pos..).unwrap_or(&[]).to_vec();
                ItemProperty::AuxC { aux_type, subtype }
            }
            b"cmpC" => {
                let (ver, _flags, body) = read_full_box(prop_box.payload)?;
                if ver != 0 || body.len() < 4 {
                    return Err(Error::InvalidData("geçersiz cmpC kutusu"));
                }
                let compression_type = [body[0], body[1], body[2], body[3]];
                ItemProperty::CmpC { compression_type }
            }
            other => ItemProperty::Unknown(*other),
        };
        let prop_bytes = core::mem::size_of::<ItemProperty>().saturating_add(match &p {
            ItemProperty::Pixi { bits_per_channel } => bits_per_channel.len(),
            ItemProperty::AuxC { aux_type, subtype } => {
                aux_type.len().saturating_add(subtype.len())
            }
            _ => 0,
        });
        guard.ekle(prop_bytes)?;
        properties.push(p);
    }
    Ok(())
}

fn parse_ipma(
    payload: &[u8],
    associations: &mut FxIndexMap<u32, Vec<PropertyAssoc>>,
    guard: &mut ButceTutucu,
) -> Result<()> {
    let (ver, flags, body) = read_full_box(payload)?;
    if ver > 1 {
        return Err(Error::Unsupported("ipma sürümü > 1 desteklenmiyor"));
    }
    let entry_count = read_u32_be(body, 0).ok_or(Error::InvalidData("kesik ipma entry_count"))?;
    let mut pos = 4usize;
    let min_entry_bytes = if ver == 0 { 3usize } else { 5usize };
    let max_entries = body.len().saturating_sub(pos) / min_entry_bytes;
    if (entry_count as usize) > max_entries {
        return Err(Error::InvalidData("ipma entry_count kutu boyutunu aşıyor"));
    }
    let reserve_n = (entry_count as usize).min(256);
    associations
        .try_reserve(reserve_n)
        .map_err(|_| Error::AllocationFailed {
            requested: reserve_n.saturating_mul(core::mem::size_of::<(u32, Vec<PropertyAssoc>)>()),
        })?;
    let large_index = (flags & 1) != 0;
    let assoc_stride = if large_index { 2usize } else { 1usize };

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

        let needed = assoc_count
            .checked_mul(assoc_stride)
            .ok_or(Error::InvalidData("ipma özellik sayısı taşması"))?;
        if needed > body.len().saturating_sub(pos) {
            return Err(Error::InvalidData("kesik ipma özellik listesi"));
        }
        let is_new_key = !associations.contains_key(&item_id);
        let entry_bytes = (if is_new_key {
            core::mem::size_of::<(u32, Vec<PropertyAssoc>)>()
        } else {
            0
        })
        .saturating_add(assoc_count.saturating_mul(core::mem::size_of::<PropertyAssoc>()));
        guard.ekle(entry_bytes)?;

        let mut list: Vec<PropertyAssoc> = Vec::new();
        list.try_reserve_exact(assoc_count)
            .map_err(|_| Error::AllocationFailed {
                requested: assoc_count.saturating_mul(core::mem::size_of::<PropertyAssoc>()),
            })?;
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

        associations.entry(item_id).or_default().extend(list);
    }
    Ok(())
}

fn parse_iloc(
    payload: &[u8],
    locations: &mut FxIndexMap<u32, IlocItem>,
    guard: &mut ButceTutucu,
) -> Result<()> {
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

    let min_item_bytes = (if ver < 2 { 2usize } else { 4usize })
        + (if ver == 1 || ver == 2 { 2usize } else { 0usize })
        + 2
        + base_offset_size
        + 2;
    let max_items = body.len().saturating_sub(pos) / min_item_bytes;
    if (item_count as usize) > max_items {
        return Err(Error::InvalidData("iloc item_count kutu boyutunu aşıyor"));
    }
    let reserve_n = (item_count as usize).min(256);
    locations
        .try_reserve(reserve_n)
        .map_err(|_| Error::AllocationFailed {
            requested: reserve_n.saturating_mul(core::mem::size_of::<(u32, IlocItem)>()),
        })?;
    let ext_stride = (if ver == 1 || ver == 2 { index_size } else { 0 })
        .saturating_add(offset_size)
        .saturating_add(length_size);

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

        if ext_stride > 0 {
            let needed = extent_count
                .checked_mul(ext_stride)
                .ok_or(Error::InvalidData("iloc extent boyutu taşması"))?;
            if needed > body.len().saturating_sub(pos) {
                return Err(Error::InvalidData("kesik iloc extent dizisi"));
            }
        }
        let item_bytes = core::mem::size_of::<(u32, IlocItem)>()
            .saturating_add(extent_count.saturating_mul(core::mem::size_of::<IlocExtent>()));
        guard.ekle(item_bytes)?;

        let mut extents: Vec<IlocExtent> = Vec::new();
        extents
            .try_reserve_exact(extent_count)
            .map_err(|_| Error::AllocationFailed {
                requested: extent_count.saturating_mul(core::mem::size_of::<IlocExtent>()),
            })?;
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

        if let Some(prev) = locations.insert(
            item_id,
            IlocItem {
                item_id,
                construction_method,
                data_reference_index,
                base_offset,
                extents,
            },
        ) {
            let prev_bytes = core::mem::size_of::<(u32, IlocItem)>().saturating_add(
                prev.extents
                    .len()
                    .saturating_mul(core::mem::size_of::<IlocExtent>()),
            );
            guard.dusur(prev_bytes);
        }
    }
    Ok(())
}

fn parse_iref(
    payload: &[u8],
    references: &mut Vec<ItemReference>,
    guard: &mut ButceTutucu,
) -> Result<()> {
    let (ver, _flags, body) = read_full_box(payload)?;
    if ver > 1 {
        return Err(Error::Unsupported("iref sürümü > 1 desteklenmiyor"));
    }
    let id_size = if ver == 0 { 2usize } else { 4usize };
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

        let needed = ref_count
            .checked_mul(id_size)
            .ok_or(Error::InvalidData("iref referans boyutu taşması"))?;
        if needed > rbody.len().saturating_sub(pos) {
            return Err(Error::InvalidData("kesik iref referans listesi"));
        }

        let is_dimg = &ref_box.box_type == b"dimg";
        let to_bytes = ref_count.saturating_mul(core::mem::size_of::<u32>());
        let ref_bytes = core::mem::size_of::<ItemReference>().saturating_add(to_bytes);
        let seen_bytes = if is_dimg && ref_count > 8 {
            to_bytes
        } else {
            0
        };
        guard.ekle(ref_bytes.saturating_add(seen_bytes))?;

        let mut seen_dimg = if is_dimg && ref_count > 8 {
            let mut set = FxIndexSet::with_hasher(rustc_hash::FxBuildHasher);
            set.try_reserve(ref_count).map_err(|_| {
                guard.dusur(ref_bytes.saturating_add(seen_bytes));
                Error::AllocationFailed {
                    requested: seen_bytes,
                }
            })?;
            Some(set)
        } else {
            None
        };
        let mut to_item_ids: Vec<u32> = Vec::new();
        to_item_ids.try_reserve_exact(ref_count).map_err(|_| {
            guard.dusur(ref_bytes.saturating_add(seen_bytes));
            Error::AllocationFailed {
                requested: to_bytes,
            }
        })?;
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
            if is_dimg {
                let is_dup = if let Some(ref mut set) = seen_dimg {
                    !set.insert(to_id)
                } else {
                    to_item_ids.contains(&to_id)
                };
                if is_dup {
                    guard.dusur(seen_bytes);
                    return Err(Error::InvalidData(
                        "dimg referans dizisinde yinelenen hedef öğe kimliği",
                    ));
                }
            }
            to_item_ids.push(to_id);
        }
        drop(seen_dimg);
        if seen_bytes > 0 {
            guard.dusur(seen_bytes);
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
    /// `ftyp` uyumlu markalar listesinde verilen 4 baytlık markanın bulunup bulunmadığını döndürür.
    pub(crate) fn has_compatible_brand(&self, brand: &[u8; 4]) -> bool {
        self.compatible_brands.contains(brand)
    }

    /// `grpl` / `altr` varlık grubunda `id1`'in `id2`'den önce listelenip listelenmediğini döndürür.
    pub(crate) fn is_preferred_alternative(&self, id1: u32, id2: u32) -> bool {
        for group in &self.entity_groups {
            if &group.grouping_type != b"altr" {
                continue;
            }
            let mut id1_found = false;
            for &eid in &group.entity_ids {
                if eid == id1 {
                    id1_found = true;
                } else if eid == id2 {
                    return id1_found;
                }
            }
        }
        false
    }

    /// Belirtilen `item_id` için `iloc` üzerinde sıfırdan büyük veri boyutu tanımlanıp tanımlanmadığı.
    pub(crate) fn item_has_data(&self, item_id: u32) -> bool {
        self.locations
            .get(&item_id)
            .is_some_and(|loc| loc.extents.iter().any(|e| e.extent_length > 0))
    }

    /// Belirtilen `item_id` üzerinde desteklenmeyen zorunlu (`essential`) bir `ipco` özelliği olup olmadığı.
    pub(crate) fn has_unsupported_essential_property(&self, item_id: u32) -> Result<bool> {
        let Some(assocs) = self.associations.get(&item_id) else {
            return Ok(false);
        };
        for assoc in assocs {
            if assoc.property_index == 0 {
                continue;
            }
            let idx = (assoc.property_index as usize) - 1;
            let prop = self.properties.get(idx).ok_or(Error::InvalidData(
                "ipma özellik indeksi ipco sınırları dışında",
            ))?;
            if assoc.essential && matches!(prop, ItemProperty::Unknown(_)) {
                return Ok(true);
            }
            if let ItemProperty::CmpC { compression_type } = prop
                && !matches!(compression_type, b"defl" | b"zlib" | b"gzip")
                && assoc.essential
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Belirtilen `item_id` için `infe` kaydını döndürür.
    pub(crate) fn item_info(&self, item_id: u32) -> Option<&ItemInfoEntry> {
        self.items.get(&item_id)
    }

    /// Belirtilen `item_id` ile ilişkilendirilmiş özellikleri (`ipma` sırasıyla) döndürür.
    pub(crate) fn item_properties(&self, item_id: u32) -> Result<Vec<(&ItemProperty, bool)>> {
        let Some(assocs) = self.associations.get(&item_id) else {
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
            if let ItemProperty::CmpC { compression_type } = prop
                && !matches!(compression_type, b"defl" | b"zlib" | b"gzip")
                && assoc.essential
            {
                return Err(Error::Unsupported(
                    "desteklenmeyen zorunlu (essential) cmpC sıkıştırma algoritması",
                ));
            }
            if !assoc.essential
                && matches!(
                    prop,
                    ItemProperty::Clap(_) | ItemProperty::Irot(_) | ItemProperty::Imir(_)
                )
            {
                return Err(Error::InvalidData(
                    "dönüşüm özellikleri (clap, irot, imir) ipma içinde zorunlu (essential) olmalıdır",
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

    /// İçeride tutulan ham dosya bayt diliminin uzunluğu.
    #[inline]
    pub(crate) fn raw_len(&self) -> usize {
        self.file_data.len()
    }

    /// Kapsayıcı içinde sahipli tutulan (`cmin` `idat`, `ColrIcc`, `Av1C.config_obus`) bayt toplamı.
    #[inline]
    pub(crate) const fn owned_bytes(&self) -> usize {
        self.owned_bytes
    }

    #[inline]
    fn idat_slice(&self) -> &[u8] {
        match &self.idat {
            IdatStorage::Empty => &[],
            IdatStorage::FileRange(r) => &self.file_data[r.clone()],
            IdatStorage::Owned(v) => v.as_slice(),
        }
    }

    /// Belirtilen öğenin bayt yükünü `iloc` üzerinden çözümler.
    pub(crate) fn resolve_item_data<'b>(
        &'b self,
        item_id: u32,
        butce: &Arc<Butce>,
    ) -> Result<(Cow<'b, [u8]>, ButceTutucu)> {
        self.resolve_item_data_inner(item_id, butce, 0)
    }

    fn resolve_item_data_inner<'b>(
        &'b self,
        item_id: u32,
        butce: &Arc<Butce>,
        depth: usize,
    ) -> Result<(Cow<'b, [u8]>, ButceTutucu)> {
        if depth > 4 {
            return Err(Error::InvalidData(
                "iloc item_offset zinciri derinlik sınırını aştı",
            ));
        }
        let loc = self
            .locations
            .get(&item_id)
            .ok_or(Error::InvalidData("öğe için iloc konum kaydı bulunamadı"))?;
        if loc.data_reference_index != 0 {
            return Err(Error::Unsupported(
                "harici dosya referanslı (data_reference_index != 0) iloc desteklenmiyor",
            ));
        }
        if loc.extents.is_empty() {
            return Err(Error::InvalidData("iloc kaydında extent bulunmuyor"));
        }

        let (raw, guard) = match loc.construction_method {
            0 | 1 => {
                let source: &'b [u8] = if loc.construction_method == 0 {
                    &self.file_data
                } else {
                    self.idat_slice()
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
                    (Cow::Borrowed(slice), ButceTutucu::bos(butce))
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
                    let g = Butce::tut(butce, total_len)?;
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
                    (Cow::Owned(buf), g)
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
                let g = Butce::tut(butce, total_len)?;
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
                    let (target_data, _target_guard) =
                        self.resolve_item_data_inner(target_id, butce, depth + 1)?;
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
                (Cow::Owned(buf), g)
            }
            _ => {
                return Err(Error::Unsupported(
                    "desteklenmeyen iloc construction_method değeri",
                ));
            }
        };

        self.decompress_item_if_needed(item_id, raw, guard, butce)
    }

    fn decompress_item_if_needed<'b>(
        &self,
        item_id: u32,
        mut data: Cow<'b, [u8]>,
        mut guard: ButceTutucu,
        butce: &Arc<Butce>,
    ) -> Result<(Cow<'b, [u8]>, ButceTutucu)> {
        let orig_compressed_len = data.len();
        if let Some(info) = self.items.get(&item_id)
            && !info.content_encoding.is_empty()
            && info.content_encoding != "identity"
        {
            let (decoded, new_guard) = match info.content_encoding.as_str() {
                "deflate" | "zlib" => {
                    decompress_by_fourcc(b"zlib", &data, orig_compressed_len, butce)?
                }
                "gzip" | "x-gzip" => {
                    decompress_by_fourcc(b"gzip", &data, orig_compressed_len, butce)?
                }
                _ => {
                    return Err(Error::Unsupported(
                        "desteklenmeyen infe content_encoding algoritması",
                    ));
                }
            };
            data = Cow::Owned(decoded);
            guard = new_guard;
        }
        if let Some(assocs) = self.associations.get(&item_id) {
            for assoc in assocs {
                if assoc.property_index == 0 {
                    continue;
                }
                let idx = (assoc.property_index as usize) - 1;
                if let Some(ItemProperty::CmpC { compression_type }) = self.properties.get(idx) {
                    let (decoded, new_guard) =
                        decompress_by_fourcc(compression_type, &data, orig_compressed_len, butce)?;
                    data = Cow::Owned(decoded);
                    guard = new_guard;
                }
            }
        }
        Ok((data, guard))
    }
}
