//! AVIF görüntü çözme, `grid` karo birleştirme, alfa kanalı ve `clap`/`irot`/`imir` dönüşüm boru hattı.

#![forbid(unsafe_code)]

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::api::{Config, Decoder, Packet};
use crate::avif::boxes::{
    ALPHA_AUX_URN_CICP, ALPHA_AUX_URN_HEVC, Av1CProperty, AvifContainer, ItemProperty,
    parse_container, parse_container_owned,
};
use crate::avif::gainmap::{GainMap, GainMapMetadata, parse_tone_mapped_image_box};
use crate::avif::image::{
    AvifDecodeConfig, AvifImage, AvifInfo, CleanAperture, ColorInfo, PlaneStorage, RgbImage,
    RgbaImage,
};
use crate::bitstream::obu::parse_sequence_header_obus;
use crate::bitstream::seq_header::PixelLayout;
use crate::error::{Error, Result, SizeStage};
use crate::frame::butce::{Butce, ButceTutucu, dolu_vec, kopya_vec};
use crate::frame::{BitDepth, ChromaFormat, Frame, PlaneType};

/// İç düzlemsel çalışma tamponu.
#[derive(Debug)]
pub(crate) struct PlanarBuffer {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) chroma_width: u32,
    pub(crate) chroma_height: u32,
    pub(crate) bit_depth: BitDepth,
    pub(crate) chroma_format: ChromaFormat,
    pub(crate) y_plane: PlaneStorage,
    pub(crate) u_plane: PlaneStorage,
    pub(crate) v_plane: PlaneStorage,
    pub(crate) seq_color: Option<ColorInfo>,
    pub(crate) budget_guard: ButceTutucu,
}

#[inline]
pub(crate) fn plane_storage_bytes(storage: &PlaneStorage) -> usize {
    match storage {
        PlaneStorage::U8(v) => v.len(),
        PlaneStorage::U16(v) => v.len().saturating_mul(2),
    }
}

pub(crate) fn copy_bytes_budgeted(
    src: &[u8],
    butce: &Arc<Butce>,
) -> Result<(Vec<u8>, ButceTutucu)> {
    let guard = Butce::tut(butce, src.len())?;
    let v = kopya_vec(src)?;
    Ok((v, guard))
}

/// Yapılandırılmış iş parçacığı sayısını etkin çalışma zamanı sayısına çözümler.
pub(crate) fn resolve_effective_threads(n_threads: usize) -> usize {
    #[cfg(any(not(feature = "std"), target_arch = "wasm32"))]
    {
        let _ = n_threads;
        1
    }
    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    {
        if n_threads == 0 {
            std::thread::available_parallelism()
                .map(core::num::NonZeroUsize::get)
                .unwrap_or(1)
        } else {
            n_threads
        }
    }
}

/// `grid` (`ImageGrid`) tanımlayıcısı.
#[derive(Clone, Copy, Debug)]
struct GridDescriptor {
    rows: u32,
    cols: u32,
    output_width: u32,
    output_height: u32,
}

/// Yardımcı alfa kanalının kapsayıcı içindeki temsil biçimi.
#[derive(Clone, Debug)]
enum AlphaSource {
    /// Birincil öğeye doğrudan `auxl` ile bağlı tekil (`av01` veya `grid`) alfa öğesi.
    Direct(u32),
    /// Renk `grid` öğesinin her bir karosuna ayrı `auxl` ile bağlı alfa karolarından oluşan sentetik ızgara.
    GridTiles {
        grid_item_id: u32,
        alpha_tile_ids: Vec<u32>,
    },
}

fn parse_grid_descriptor(data: &[u8]) -> Result<GridDescriptor> {
    if data.len() < 8 {
        return Err(Error::InvalidData("kesik grid tanımlayıcısı (< 8 bayt)"));
    }
    let version = data[0];
    if version != 0 {
        return Err(Error::Unsupported("grid sürümü 0 dışında desteklenmiyor"));
    }
    let flags = data[1];
    let rows = u32::from(data[2]) + 1;
    let cols = u32::from(data[3]) + 1;
    let (output_width, output_height) = if (flags & 1) != 0 {
        if data.len() < 12 {
            return Err(Error::InvalidData(
                "kesik 32-bit grid tanımlayıcısı (< 12 bayt)",
            ));
        }
        let w = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        let h = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
        (w, h)
    } else {
        let w = u32::from(u16::from_be_bytes([data[4], data[5]]));
        let h = u32::from(u16::from_be_bytes([data[6], data[7]]));
        (w, h)
    };
    if output_width == 0 || output_height == 0 {
        return Err(Error::InvalidData("grid çıktı boyutları sıfır olamaz"));
    }
    Ok(GridDescriptor {
        rows,
        cols,
        output_width,
        output_height,
    })
}

pub(crate) fn av1c_to_format(av1c: &Av1CProperty) -> Result<(BitDepth, ChromaFormat)> {
    let bit_depth = match (av1c.high_bitdepth, av1c.twelve_bit) {
        (false, _) => BitDepth::Eight,
        (true, false) => BitDepth::Ten,
        (true, true) => BitDepth::Twelve,
    };
    let chroma_format = if av1c.monochrome {
        ChromaFormat::Monochrome
    } else {
        match (av1c.chroma_subsampling_x, av1c.chroma_subsampling_y) {
            (1, 1) => ChromaFormat::C420,
            (1, 0) => ChromaFormat::C422,
            (0, 0) => ChromaFormat::C444,
            _ => {
                return Err(Error::InvalidData(
                    "geçersiz av1C kroma alt örnekleme kombinasyonu",
                ));
            }
        }
    };
    Ok((bit_depth, chroma_format))
}

pub(crate) fn compute_clap_rect(
    w: u32,
    h: u32,
    clap: &CleanAperture,
) -> Result<(u32, u32, u32, u32)> {
    if clap.width_d == 0 || clap.height_d == 0 || clap.horiz_off_d == 0 || clap.vert_off_d == 0 {
        return Err(Error::InvalidData("clap paydası sıfır olamaz"));
    }
    if !clap.width_n.is_multiple_of(clap.width_d) || !clap.height_n.is_multiple_of(clap.height_d) {
        return Err(Error::InvalidData(
            "clap genişlik veya yüksekliği tam sayı değil",
        ));
    }
    let crop_w = clap.width_n / clap.width_d;
    let crop_h = clap.height_n / clap.height_d;
    if crop_w == 0 || crop_h == 0 || crop_w > w || crop_h > h {
        return Err(Error::InvalidData(
            "clap kırpma boyutları görüntü sınırları dışında",
        ));
    }

    // cropX = w/2 + horizOffN/horizOffD - cropW/2
    //       = ((w - cropW) * horizOffD + 2 * horizOffN) / (2 * horizOffD)
    // Ara çarpımlarda `i64` taşmasını önlemek için `i128` aritmetiği kullanılır.
    let hd = i128::from(clap.horiz_off_d);
    let vd = i128::from(clap.vert_off_d);
    let num_x = (i128::from(w) - i128::from(crop_w)) * hd + 2 * i128::from(clap.horiz_off_n);
    let den_x = 2 * hd;
    let num_y = (i128::from(h) - i128::from(crop_h)) * vd + 2 * i128::from(clap.vert_off_n);
    let den_y = 2 * vd;

    if (num_x % den_x) != 0 || (num_y % den_y) != 0 {
        return Err(Error::InvalidData(
            "clap sol-üst kırpma ofseti tam sayı değil",
        ));
    }
    let x0_i128 = num_x / den_x;
    let y0_i128 = num_y / den_y;
    if x0_i128 < 0
        || y0_i128 < 0
        || x0_i128 + i128::from(crop_w) > i128::from(w)
        || y0_i128 + i128::from(crop_h) > i128::from(h)
    {
        return Err(Error::InvalidData(
            "clap kırpma bölgesi görüntü sınırları dışında",
        ));
    }
    Ok((x0_i128 as u32, y0_i128 as u32, crop_w, crop_h))
}

/// Bir öğenin dönüşüm öncesi (`raw_w`, `raw_h`) ve dönüşümler (`clap`, `irot`, `imir`) sonrası
/// boyutlarını ve temel özelliklerini çözümler.
struct InspectedItem {
    width: u32,
    height: u32,
    raw_width: u32,
    raw_height: u32,
    bit_depth: BitDepth,
    chroma_format: ChromaFormat,
    seq_color: Option<ColorInfo>,
}

fn inspect_item_dimensions(
    container: &AvifContainer<'_>,
    item_id: u32,
    depth: usize,
    butce: &Arc<Butce>,
    config: &AvifDecodeConfig,
) -> Result<InspectedItem> {
    if depth > 8 {
        return Err(Error::InvalidData(
            "AVIF öğe referans zinciri derinlik sınırını aştı",
        ));
    }
    let info = container
        .item_info(item_id)
        .ok_or(Error::InvalidData("AVIF öğesi iinf içinde bulunamadı"))?;
    let props = container.item_properties(item_id)?;

    let mut ispe_dims: Option<(u32, u32)> = None;
    let mut bit_depth: Option<BitDepth> = None;
    let mut chroma_format: Option<ChromaFormat> = None;
    let mut seq_color: Option<ColorInfo> = None;
    let mut clap_opt: Option<&CleanAperture> = None;
    let mut irot_opt: Option<u8> = None;

    for (p, _) in props {
        match p {
            ItemProperty::Ispe { width, height } => {
                if *width == 0 || *height == 0 {
                    return Err(Error::InvalidData("ispe genişlik veya yüksekliği sıfır"));
                }
                ispe_dims = Some((*width, *height));
            }
            ItemProperty::Av1C(av1c) => {
                let (bd, cf) = av1c_to_format(av1c)?;
                bit_depth = Some(bd);
                chroma_format = Some(cf);
                if !av1c.config_obus.is_empty()
                    && let Ok(Some(seq)) = parse_sequence_header_obus(&av1c.config_obus)
                {
                    seq_color = Some(ColorInfo {
                        color_primaries: u16::from(seq.pri),
                        transfer_characteristics: u16::from(seq.trc),
                        matrix_coefficients: u16::from(seq.mtrx),
                        full_range: seq.color_range != 0,
                    });
                }
            }
            ItemProperty::Pixi { bits_per_channel } => {
                if bit_depth.is_none()
                    && let Some(&first_bpc) = bits_per_channel.first()
                {
                    bit_depth = BitDepth::from_bits(u32::from(first_bpc));
                }
            }
            ItemProperty::Clap(c) => clap_opt = Some(c),
            ItemProperty::Irot(a) => irot_opt = Some(*a),
            _ => {}
        }
    }

    let (mut w, mut h, bd, cf) = match &info.item_type {
        b"av01" => {
            if ispe_dims.is_none() || bit_depth.is_none() || chroma_format.is_none() {
                match container.resolve_item_data(item_id, butce) {
                    Ok((payload, _guard)) => {
                        if let Ok(Some(seq)) = parse_sequence_header_obus(&payload) {
                            if ispe_dims.is_none() {
                                ispe_dims = Some((
                                    seq.max_width.max(1) as u32,
                                    seq.max_height.max(1) as u32,
                                ));
                            }
                            if bit_depth.is_none() {
                                bit_depth = match seq.hbd {
                                    0 => Some(BitDepth::Eight),
                                    1 => Some(BitDepth::Ten),
                                    _ => Some(BitDepth::Twelve),
                                };
                            }
                            if chroma_format.is_none() {
                                chroma_format = Some(match seq.layout {
                                    PixelLayout::I400 => ChromaFormat::Monochrome,
                                    PixelLayout::I420 => ChromaFormat::C420,
                                    PixelLayout::I422 => ChromaFormat::C422,
                                    PixelLayout::I444 => ChromaFormat::C444,
                                });
                            }
                            if seq_color.is_none() {
                                seq_color = Some(ColorInfo {
                                    color_primaries: u16::from(seq.pri),
                                    transfer_characteristics: u16::from(seq.trc),
                                    matrix_coefficients: u16::from(seq.mtrx),
                                    full_range: seq.color_range != 0,
                                });
                            }
                        }
                    }
                    Err(err @ (Error::BudgetExceeded { .. } | Error::AllocationFailed { .. })) => {
                        return Err(err);
                    }
                    Err(_) => {}
                }
            }
            let (iw, ih) =
                ispe_dims.ok_or(Error::InvalidData("av01 öğesinde boyut bilgisi eksik"))?;
            let bd = bit_depth.unwrap_or(BitDepth::Eight);
            let cf = chroma_format.unwrap_or(ChromaFormat::C420);
            (iw, ih, bd, cf)
        }
        b"grid" => {
            let (grid_bytes, grid_guard) = container.resolve_item_data(item_id, butce)?;
            let grid = parse_grid_descriptor(&grid_bytes)?;
            drop((grid_bytes, grid_guard));
            let mut tiles = container.references_from(item_id, b"dimg");
            let tile_count = tiles.clone().count();
            let expected = (grid.rows as usize)
                .checked_mul(grid.cols as usize)
                .ok_or(Error::InvalidData("grid karo sayısı taşması"))?;
            if tile_count != expected || tile_count == 0 {
                return Err(Error::InvalidData(
                    "grid dimg referans sayısı rows*cols ile uyuşmuyor",
                ));
            }
            ensure_grid_tiles_coded(container, tiles.clone())?;
            let first_tile = tiles.next().unwrap();
            let tile_info =
                inspect_item_dimensions(container, first_tile, depth + 1, butce, config)?;
            if seq_color.is_none() {
                seq_color = tile_info.seq_color;
            }
            let (gw, gh) = ispe_dims.unwrap_or((grid.output_width, grid.output_height));
            (
                gw,
                gh,
                bit_depth.unwrap_or(tile_info.bit_depth),
                chroma_format.unwrap_or(tile_info.chroma_format),
            )
        }
        b"iden" => {
            let mut targets = container.references_from(item_id, b"dimg");
            let (Some(target_0), None) = (targets.next(), targets.next()) else {
                return Err(Error::InvalidData(
                    "iden öğesi tek bir dimg referansı içermelidir",
                ));
            };
            let target_info =
                inspect_item_dimensions(container, target_0, depth + 1, butce, config)?;
            if seq_color.is_none() {
                seq_color = target_info.seq_color;
            }
            (
                target_info.width,
                target_info.height,
                target_info.bit_depth,
                target_info.chroma_format,
            )
        }
        _ => return Err(Error::Unsupported("desteklenmeyen birincil AVIF öğe türü")),
    };

    let raw_width = w;
    let raw_height = h;
    check_pixel_and_byte_budget(raw_width, raw_height, 0, config)?;

    // MIAF dönüşüm sırası: önce clap, sonra irot, sonra imir.
    if let Some(clap) = clap_opt {
        let (_, _, cw, ch) = compute_clap_rect(w, h, clap)?;
        w = cw;
        h = ch;
    }
    if let Some(angle) = irot_opt
        && (angle & 1) != 0
    {
        core::mem::swap(&mut w, &mut h);
    }
    check_pixel_and_byte_budget(w, h, 0, config)?;

    Ok(InspectedItem {
        width: w,
        height: h,
        raw_width,
        raw_height,
        bit_depth: bd,
        chroma_format: cf,
        seq_color,
    })
}

fn find_direct_alpha_item(container: &AvifContainer<'_>, target_id: u32) -> Result<Option<u32>> {
    let aux_candidates = container.references_to(target_id, b"auxl");
    let mut found: Option<u32> = None;
    for aux_id in aux_candidates {
        let props = container.item_properties(aux_id)?;
        for (p, _) in props {
            if let ItemProperty::AuxC { aux_type, .. } = p
                && (aux_type == ALPHA_AUX_URN_CICP || aux_type == ALPHA_AUX_URN_HEVC)
            {
                if found.is_some() {
                    return Err(Error::InvalidData(
                        "bir öğeye birden fazla yardımcı alfa öğesi bağlanamaz",
                    ));
                }
                found = Some(aux_id);
            }
        }
    }
    Ok(found)
}

fn find_alpha_source(
    container: &AvifContainer<'_>,
    primary_id: u32,
    butce: &Arc<Butce>,
) -> Result<Option<(AlphaSource, ButceTutucu)>> {
    if let Some(aid) = find_direct_alpha_item(container, primary_id)? {
        return Ok(Some((AlphaSource::Direct(aid), ButceTutucu::bos(butce))));
    }
    let Some(info) = container.item_info(primary_id) else {
        return Ok(None);
    };
    if &info.item_type == b"grid" {
        let tile_ids = container.references_from(primary_id, b"dimg");
        let tile_count = tile_ids.clone().count();
        if tile_count == 0 {
            return Ok(None);
        }
        let bytes =
            tile_count
                .checked_mul(core::mem::size_of::<u32>())
                .ok_or(Error::SizeOverflow {
                    stage: SizeStage::TotalAllocation,
                    lhs: tile_count,
                    rhs: core::mem::size_of::<u32>(),
                })?;
        let guard = Butce::tut(butce, bytes)?;
        let mut alpha_tiles: Vec<u32> = Vec::new();
        alpha_tiles
            .try_reserve_exact(tile_count)
            .map_err(|_| Error::AllocationFailed { requested: bytes })?;
        for tid in tile_ids {
            if let Some(aid) = find_direct_alpha_item(container, tid)? {
                alpha_tiles.push(aid);
            } else {
                return Ok(None);
            }
        }
        return Ok(Some((
            AlphaSource::GridTiles {
                grid_item_id: primary_id,
                alpha_tile_ids: alpha_tiles,
            },
            guard,
        )));
    }
    Ok(None)
}

fn is_premultiplied_alpha(
    container: &AvifContainer<'_>,
    primary_id: u32,
    alpha_src: Option<&AlphaSource>,
) -> bool {
    let Some(alpha_src) = alpha_src else {
        return false;
    };
    let alpha_id = match alpha_src {
        AlphaSource::Direct(id) => *id,
        AlphaSource::GridTiles { alpha_tile_ids, .. } => {
            let Some(&first) = alpha_tile_ids.first() else {
                return false;
            };
            first
        }
    };
    if container
        .references_from(primary_id, b"prem")
        .any(|id| id == alpha_id)
    {
        return true;
    }
    container
        .references_to(primary_id, b"prem")
        .any(|id| id == alpha_id)
}

pub(crate) type MetadataBundle = (Option<Vec<u8>>, Option<Vec<u8>>, ButceTutucu);

pub(crate) fn extract_metadata_items(
    container: &AvifContainer<'_>,
    primary_id: u32,
    butce: &Arc<Butce>,
) -> Result<MetadataBundle> {
    let mut exif: Option<Vec<u8>> = None;
    let mut xmp: Option<Vec<u8>> = None;
    let mut meta_guard = ButceTutucu::bos(butce);

    let meta_ids = container.references_to(primary_id, b"cdsc");
    for mid in meta_ids {
        let Some(info) = container.item_info(mid) else {
            continue;
        };
        if &info.item_type == b"Exif" && exif.is_none() {
            let (raw, raw_guard) = container.resolve_item_data(mid, butce)?;
            if raw.len() < 4 {
                return Err(Error::InvalidData("Exif öğesi 4 bayttan kısa"));
            }
            let tiff_offset = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
            let start = 4usize
                .checked_add(tiff_offset)
                .ok_or(Error::InvalidData("Exif TIFF başlık ofseti taşması"))?;
            let slice = raw.get(start..).ok_or(Error::InvalidData(
                "Exif TIFF başlık ofseti veri sınırları dışında",
            ))?;
            let (v, g) = copy_bytes_budgeted(slice, butce)?;
            drop((raw, raw_guard));
            meta_guard.birlestir(g);
            exif = Some(v);
        } else if &info.item_type == b"mime"
            && info.content_type == "application/rdf+xml"
            && xmp.is_none()
        {
            let (raw, raw_guard) = container.resolve_item_data(mid, butce)?;
            let (v, g) = match raw {
                Cow::Owned(v) => (v, raw_guard),
                Cow::Borrowed(s) => copy_bytes_budgeted(s, butce)?,
            };
            meta_guard.birlestir(g);
            xmp = Some(v);
        }
    }
    Ok((exif, xmp, meta_guard))
}

/// `trak` içine gömülü `meta` kutusundan tüm ilişkili `Exif`/`XMP` öğelerini çıkarır
/// (`libavif` `avifDecoderFindMetadata(..., colorId = 0)` uyumlu).
pub(crate) fn extract_track_metadata_items(
    container: &AvifContainer<'_>,
    butce: &Arc<Butce>,
) -> Result<MetadataBundle> {
    let mut exif: Option<Vec<u8>> = None;
    let mut xmp: Option<Vec<u8>> = None;
    let mut meta_guard = ButceTutucu::bos(butce);

    for info in container.items.values() {
        if !container.item_has_data(info.item_id)
            || container.has_unsupported_essential_property(info.item_id)?
        {
            continue;
        }
        if &info.item_type == b"Exif" && exif.is_none() {
            let (raw, raw_guard) = container.resolve_item_data(info.item_id, butce)?;
            if raw.len() < 4 {
                return Err(Error::InvalidData("Exif öğesi 4 bayttan kısa"));
            }
            let tiff_offset = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
            let start = 4usize
                .checked_add(tiff_offset)
                .ok_or(Error::InvalidData("Exif TIFF başlık ofseti taşması"))?;
            let slice = raw.get(start..).ok_or(Error::InvalidData(
                "Exif TIFF başlık ofseti veri sınırları dışında",
            ))?;
            let (v, g) = copy_bytes_budgeted(slice, butce)?;
            drop((raw, raw_guard));
            meta_guard.birlestir(g);
            exif = Some(v);
        } else if &info.item_type == b"mime"
            && info.content_type == "application/rdf+xml"
            && xmp.is_none()
        {
            let (raw, raw_guard) = container.resolve_item_data(info.item_id, butce)?;
            let (v, g) = match raw {
                Cow::Owned(v) => (v, raw_guard),
                Cow::Borrowed(s) => copy_bytes_budgeted(s, butce)?,
            };
            meta_guard.birlestir(g);
            xmp = Some(v);
        }
    }
    Ok((exif, xmp, meta_guard))
}

/// ISO 21496-1 `tmap` öğesini `libavif` (`avifDecoderFindGainMapItem`) kurallarına göre bulur,
/// ikili `ToneMapImage` yükünü ve `tmap` özelliklerini doğrular.
fn find_and_validate_tmap(
    container: &AvifContainer<'_>,
    primary_id: u32,
    base_raw_w: u32,
    base_raw_h: u32,
    butce: &Arc<Butce>,
) -> Result<Option<(u32, GainMapMetadata, ButceTutucu)>> {
    // ISO/IEC 23008-12:2024/AMD 1 §10.2.6 ve libavif uyumu: `tmap` uyumlu markası yoksa
    // `tmap` öğeleri yoksayılır.
    if !container.has_compatible_brand(b"tmap") {
        return Ok(None);
    }

    let mut found: Option<(u32, u32)> = None;
    for item in container.items.values() {
        if &item.item_type != b"tmap" {
            continue;
        }
        if !container.item_has_data(item.item_id)
            || container.has_unsupported_essential_property(item.item_id)?
            || container
                .references_from(item.item_id, b"thmb")
                .next()
                .is_some()
        {
            continue;
        }
        let mut dimg_ids = container.references_from(item.item_id, b"dimg");
        let (Some(dimg_0), Some(dimg_1), None) =
            (dimg_ids.next(), dimg_ids.next(), dimg_ids.next())
        else {
            return Err(Error::InvalidData(
                "tmap dimg referansları tam olarak 2 farklı geçerli öğe içermelidir",
            ));
        };
        if dimg_0 == dimg_1
            || container.item_info(dimg_0).is_none()
            || container.item_info(dimg_1).is_none()
        {
            return Err(Error::InvalidData(
                "tmap dimg referansları tam olarak 2 farklı geçerli öğe içermelidir",
            ));
        }
        if dimg_0 != primary_id {
            continue;
        }
        found = Some((item.item_id, dimg_1));
        break;
    }

    let Some((tmap_id, gain_map_item_id)) = found else {
        return Ok(None);
    };

    // `grpl` / `altr` varlık grubunda `tmap` öğesi birincil öğeden önce tercih edilmelidir.
    if !container.is_preferred_alternative(tmap_id, primary_id) {
        return Ok(None);
    }

    let (tmap_data, _tmap_data_guard) = container.resolve_item_data(tmap_id, butce)?;
    let Some(mut metadata) = parse_tone_mapped_image_box(&tmap_data)? else {
        return Ok(None);
    };
    drop((tmap_data, _tmap_data_guard));

    // Kazanç haritası öğesi temel geçerlilik kontrolü (desteklenmeyen tür/özellik -> yoksay).
    let Some(gm_info) = container.item_info(gain_map_item_id) else {
        return Ok(None);
    };
    if !matches!(&gm_info.item_type, b"av01" | b"grid" | b"iden")
        || container.has_unsupported_essential_property(gain_map_item_id)?
    {
        return Ok(None);
    }

    // `tmap` öğesinin özelliklerini (`ispe`, `colr`, `clli`, `mdcv`, `pixi` ve yasak dönüşümler) doğrula.
    let tmap_props = container.item_properties(tmap_id)?;
    let mut tmap_ispe: Option<(u32, u32)> = None;
    let mut seen_alt_nclx = false;
    let mut seen_alt_icc = false;
    let mut tmap_guard = ButceTutucu::bos(butce);
    for (p, _) in tmap_props {
        match p {
            ItemProperty::Ispe { width, height } => {
                tmap_ispe = Some((*width, *height));
            }
            ItemProperty::ColrNclx(c) => {
                if seen_alt_nclx {
                    return Err(Error::InvalidData(
                        "tmap öğesinde birden fazla colr nclx özelliği",
                    ));
                }
                seen_alt_nclx = true;
                metadata.alt_color = Some(*c);
            }
            ItemProperty::ColrIcc(icc) => {
                if seen_alt_icc {
                    return Err(Error::InvalidData(
                        "tmap öğesinde birden fazla colr ICC özelliği",
                    ));
                }
                seen_alt_icc = true;
                let (icc_vec, g) = copy_bytes_budgeted(icc, butce)?;
                tmap_guard.birlestir(g);
                metadata.alt_icc_profile = Some(icc_vec);
            }
            ItemProperty::Clli(clli) => {
                metadata.alt_clli = Some(*clli);
            }
            ItemProperty::Mdcv(mdcv) => {
                metadata.alt_mdcv = Some(*mdcv);
            }
            ItemProperty::Pixi { bits_per_channel } => {
                metadata.alt_plane_count = bits_per_channel.len().min(usize::from(u8::MAX)) as u8;
                metadata.alt_depth = bits_per_channel.first().copied().unwrap_or(0);
            }
            ItemProperty::Pasp(_)
            | ItemProperty::Clap(_)
            | ItemProperty::Irot(_)
            | ItemProperty::Imir(_) => {
                return Err(Error::InvalidData(
                    "tmap öğesine pasp, clap, irot veya imir özelliği doğrudan bağlanamaz",
                ));
            }
            _ => {}
        }
    }

    let Some((tmap_w, tmap_h)) = tmap_ispe else {
        return Err(Error::InvalidData(
            "tmap öğesinde zorunlu ispe özelliği eksik",
        ));
    };
    if tmap_w != base_raw_w || tmap_h != base_raw_h {
        return Err(Error::InvalidData(
            "tmap ispe boyutları taban görüntü boyutlarıyla eşleşmiyor",
        ));
    }

    // Kazanç haritası öğesinde birden fazla `colr nclx` olmamalı (`libavif` `avifReadColorNclxProperty`).
    let gm_props = container.item_properties(gain_map_item_id)?;
    let mut seen_gm_nclx = false;
    for (p, _) in gm_props {
        if matches!(p, ItemProperty::ColrNclx(_)) {
            if seen_gm_nclx {
                return Err(Error::InvalidData(
                    "kazanç haritası öğesinde birden fazla colr nclx özelliği",
                ));
            }
            seen_gm_nclx = true;
        }
    }

    Ok(Some((gain_map_item_id, metadata, tmap_guard)))
}

#[derive(PartialEq, Eq)]
struct ItemTransformTuple {
    pasp: Option<crate::avif::image::PixelAspectRatio>,
    clap: Option<CleanAperture>,
    irot: Option<u8>,
    imir: Option<u8>,
}

/// Kazanç haritası öğesi ile birincil görüntü öğesinin `pasp`/`clap`/`irot`/`imir`
/// özelliklerinin eşleştiğini doğrular (`libavif` `avifDecoderCheckGainMapProperties`).
fn validate_gain_map_transform_parity(
    container: &AvifContainer<'_>,
    primary_id: u32,
    gain_map_item_id: u32,
) -> Result<()> {
    let extract_xforms = |item_id: u32| -> Result<ItemTransformTuple> {
        let mut out = ItemTransformTuple {
            pasp: None,
            clap: None,
            irot: None,
            imir: None,
        };
        for (p, _) in container.item_properties(item_id)? {
            match p {
                ItemProperty::Pasp(pa) => out.pasp = Some(*pa),
                ItemProperty::Clap(c) => out.clap = Some(*c),
                ItemProperty::Irot(a) => out.irot = Some(*a),
                ItemProperty::Imir(m) => out.imir = Some(*m),
                _ => {}
            }
        }
        Ok(out)
    };

    let primary_x = extract_xforms(primary_id)?;
    let gm_x = extract_xforms(gain_map_item_id)?;
    if primary_x != gm_x {
        return Err(Error::InvalidData(
            "kazanç haritası öğesi ile taban görüntünün pasp/clap/irot/imir özellikleri uyumsuz",
        ));
    }
    Ok(())
}

/// Piksel çözme işlemi yapmadan AVIF kapsayıcısının başlık, boyut, renk ve metaveri bilgilerini ayrıştırır.
pub fn parse_info(data: &[u8]) -> Result<AvifInfo> {
    parse_info_with(data, &AvifDecodeConfig::default())
}

/// Belirtilen [`AvifDecodeConfig`] bütçe sınırları (`allocation_limit`) altında piksel çözme
/// yapmadan AVIF kapsayıcısının başlık, boyut, renk ve metaveri bilgilerini ayrıştırır.
///
/// `cmin`/`infe` sıkıştırma açma, `idat`, `ICC`, `Exif`, `XMP` ve `tmap` üst verisi ortak
/// bellek bütçesi üzerinden denetlenir.
pub fn parse_info_with(data: &[u8], config: &AvifDecodeConfig) -> Result<AvifInfo> {
    let butce = Butce::yeni_arc(config.allocation_limit, 0)?;
    parse_info_with_budget(data, &butce, config)
}

pub(crate) fn parse_info_with_budget(
    data: &[u8],
    butce: &Arc<Butce>,
    config: &AvifDecodeConfig,
) -> Result<AvifInfo> {
    let container = parse_container(data, butce)?;
    parse_info_from_container(&container, butce, config)
}

fn parse_info_from_container(
    container: &AvifContainer<'_>,
    butce: &Arc<Butce>,
    config: &AvifDecodeConfig,
) -> Result<AvifInfo> {
    let primary_id = container.primary_item_id;
    let inspected = inspect_item_dimensions(container, primary_id, 0, butce, config)?;

    let props = container.item_properties(primary_id)?;
    let mut color = inspected.seq_color.unwrap_or_default();
    let mut icc_profile: Option<Vec<u8>> = None;
    let mut _info_guard = ButceTutucu::bos(butce);
    let mut clli = None;
    let mut mdcv = None;
    let mut pasp = None;

    for (p, _) in props {
        match p {
            ItemProperty::ColrNclx(c) => color = *c,
            ItemProperty::ColrIcc(icc) => {
                if let Some(ref prev) = icc_profile {
                    _info_guard.dusur(prev.len());
                }
                let (v, g) = copy_bytes_budgeted(icc, butce)?;
                _info_guard.birlestir(g);
                icc_profile = Some(v);
            }
            ItemProperty::Clli(c) => clli = Some(*c),
            ItemProperty::Mdcv(m) => mdcv = Some(*m),
            ItemProperty::Pasp(pa) => pasp = Some(*pa),
            _ => {}
        }
    }

    let alpha_src_opt = find_alpha_source(container, primary_id, butce)?;
    let alpha_src = alpha_src_opt.as_ref().map(|(s, _)| s);
    let has_alpha = alpha_src.is_some();
    let premultiplied_alpha = is_premultiplied_alpha(container, primary_id, alpha_src);
    drop(alpha_src_opt);
    let (exif, xmp, meta_guard) = extract_metadata_items(container, primary_id, butce)?;
    _info_guard.birlestir(meta_guard);
    let gain_map_metadata = find_and_validate_tmap(
        container,
        primary_id,
        inspected.raw_width,
        inspected.raw_height,
        butce,
    )?
    .map(|(_, meta, gm_guard)| {
        _info_guard.birlestir(gm_guard);
        meta
    });

    Ok(AvifInfo {
        width: inspected.width,
        height: inspected.height,
        bit_depth: inspected.bit_depth,
        chroma_format: inspected.chroma_format,
        has_alpha,
        premultiplied_alpha,
        color,
        icc_profile,
        exif,
        xmp,
        clli,
        mdcv,
        pasp,
        has_gain_map: gain_map_metadata.is_some(),
        gain_map_metadata,
    })
}

fn info_metadata_bytes(info: &AvifInfo) -> usize {
    info.icc_profile
        .as_ref()
        .map_or(0, Vec::len)
        .saturating_add(info.exif.as_ref().map_or(0, Vec::len))
        .saturating_add(info.xmp.as_ref().map_or(0, Vec::len))
        .saturating_add(
            info.gain_map_metadata
                .as_ref()
                .and_then(|g| g.alt_icc_profile.as_ref())
                .map_or(0, Vec::len),
        )
}

/// Tek geçişte ayrıştırılmış AVIF kapsayıcısı ve üst bilgisini birlikte tutan hazırlıklı çözücü girdisi.
///
/// Başlık incelemesi (`info`) ile piksel çözümünü (`decode_with`) ayırmak isteyen ancak kutu
/// ağacını (`cmin` açma dâhil) ikinci kez ayrıştırmak istemeyen tüketiciler (ör. `av1oxide-image`)
/// için tasarlanmıştır. `%100 #![forbid(unsafe_code)]` altında öz-referanssız ofset tabanlı
/// depolama kullanır.
#[derive(Clone, Debug)]
pub struct AvifPrepared {
    container: AvifContainer<'static>,
    info: AvifInfo,
}

impl AvifPrepared {
    /// Sahipli AVIF bayt vektörünü tek seferde ayrıştırıp başlık/üst bilgisini çıkarır.
    ///
    /// `raw_bytes` tamponu çağıran tarafından önceden ayrıldığı için `config.allocation_limit`
    /// kapsayıcı ve başlık tahsislerini (`cmin`/`infe` açma, `idat`, `ICC`/`Exif`/`XMP`, `tmap`)
    /// sınırlar; toplam uçtan uca tavan isteniyorsa `limit - raw_bytes.len()` geçirilmelidir.
    pub fn from_vec(raw_bytes: Vec<u8>, config: &AvifDecodeConfig) -> Result<Self> {
        let butce = Butce::yeni_arc(config.allocation_limit, 0)?;
        let container = parse_container_owned(raw_bytes, &butce)?;
        let info = parse_info_from_container(&container, &butce, config)?;
        Ok(Self { container, info })
    }

    /// Önceden ayrıştırılmış AVIF üst verisine salt-okunur erişim döndürür.
    #[must_use]
    pub const fn info(&self) -> &AvifInfo {
        &self.info
    }

    /// İçeride tutulan ham AVIF girdi vektörünün bayt uzunluğu.
    #[must_use]
    pub fn raw_len(&self) -> usize {
        self.container.raw_len()
    }

    /// İçeride tutulan ham AVIF girdi vektörünün gerçek yığın kapasitesi (bayt).
    #[must_use]
    pub fn raw_capacity(&self) -> usize {
        self.container.raw_capacity()
    }

    /// İçeride tutulan ham girdi (`capacity`) + sahipli kapsayıcı (`cmin`/`idat`/`ipco`) + üst veri bayt toplamı.
    #[must_use]
    pub fn held_bytes(&self) -> usize {
        self.container
            .raw_capacity()
            .saturating_add(self.container.owned_bytes())
            .saturating_add(info_metadata_bytes(&self.info))
    }

    /// Önceden ayrıştırılmış kapsayıcıyı yeniden ayrıştırmadan pikselleri çözer ve ham girdiyi bırakır.
    pub fn decode_with(self, config: &AvifDecodeConfig) -> Result<AvifImage> {
        let butce = Butce::yeni_arc(config.allocation_limit, 0)?;
        // Halihazırda kapsayıcıda canlı tutulan sahipli baytları (`cmin`/`idat`/`ipco`/tablolar)
        // ve çözme boyunca canlı kalacak üst veri baytlarını (`ICC`/`Exif`/`XMP`/`alt_icc`)
        // piksel çözümü başlamadan önce ortak bütçe tavanına kaydet.
        let _container_carry = Butce::tut(&butce, self.container.owned_bytes())?;
        let info_carry = Butce::tut(&butce, info_metadata_bytes(&self.info))?;
        let img = decode_from_container(
            &self.container,
            config,
            Arc::clone(&butce),
            Some((self.info, info_carry)),
        )?;
        drop((_container_carry, self.container));
        butce.taban_guncelle(0);
        Ok(img)
    }
}

pub(crate) fn check_pixel_and_byte_budget(
    width: u32,
    height: u32,
    bytes_needed: usize,
    config: &AvifDecodeConfig,
) -> Result<()> {
    let pixels = u64::from(width) * u64::from(height);
    if config.max_pixels > 0 && pixels > config.max_pixels {
        let req = usize::try_from(pixels).unwrap_or(usize::MAX);
        let lim = usize::try_from(config.max_pixels).unwrap_or(usize::MAX);
        return Err(Error::BudgetExceeded {
            requested: req,
            limit: lim,
        });
    }
    if let Some(limit) = config.allocation_limit
        && bytes_needed > limit
    {
        return Err(Error::BudgetExceeded {
            requested: bytes_needed,
            limit,
        });
    }
    Ok(())
}

/// 8-bit sınırlı aralık (`[16, 235]`) → tam aralık (`[0, 255]`) alfa genişletme arama tablosu.
pub(crate) const LIMITED_TO_FULL_Y_8BIT: [u8; 256] = {
    let mut lut = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        let expanded = (((i as i32 - 16) * 255) + (219 / 2)) / 219;
        lut[i] = if expanded < 0 {
            0
        } else if expanded > 255 {
            255
        } else {
            expanded as u8
        };
        i += 1;
    }
    lut
};

#[inline]
pub(crate) fn limited_to_full_y_u16(bit_depth: BitDepth, v: u16) -> u16 {
    let (min_y, max_y, full_y): (i32, i32, i32) = match bit_depth {
        BitDepth::Eight => (16, 235, 255),
        BitDepth::Ten => (64, 940, 1023),
        BitDepth::Twelve => (256, 3760, 4095),
    };
    let span = max_y - min_y;
    let expanded = (((i32::from(v) - min_y) * full_y) + (span / 2)) / span;
    expanded.clamp(0, full_y) as u16
}

pub(crate) fn extract_plane(
    frame: &Frame,
    plane_type: PlaneType,
    butce: &Arc<Butce>,
) -> Result<(u32, u32, PlaneStorage, ButceTutucu)> {
    let bit_depth = frame.bit_depth();
    let Some(plane) = frame.plane(plane_type) else {
        return Ok((
            0,
            0,
            PlaneStorage::empty_for(bit_depth),
            ButceTutucu::bos(butce),
        ));
    };
    let w = plane.width();
    let h = plane.height();
    let count = w.checked_mul(h).ok_or(Error::SizeOverflow {
        stage: SizeStage::PlaneArea,
        lhs: w,
        rhs: h,
    })?;
    let bps = PlaneStorage::bytes_per_sample(bit_depth);
    let bytes_req = count.checked_mul(bps).ok_or(Error::SizeOverflow {
        stage: SizeStage::TotalAllocation,
        lhs: count,
        rhs: bps,
    })?;
    let guard = Butce::tut(butce, bytes_req)?;

    match bit_depth {
        BitDepth::Eight => {
            let mut out = Vec::new();
            out.try_reserve_exact(count)
                .map_err(|_| Error::AllocationFailed {
                    requested: bytes_req,
                })?;
            for y in 0..h {
                let row = plane.row(y).ok_or(Error::InvalidData(
                    "çözülen karede eksik 8-bit düzlem satırı",
                ))?;
                out.extend_from_slice(&row[..w]);
            }
            Ok((w as u32, h as u32, PlaneStorage::U8(out), guard))
        }
        BitDepth::Ten | BitDepth::Twelve => {
            let mut out = Vec::new();
            out.try_reserve_exact(count)
                .map_err(|_| Error::AllocationFailed {
                    requested: bytes_req,
                })?;
            let row_bytes_len = w * 2;
            for y in 0..h {
                let row = plane.row(y).ok_or(Error::InvalidData(
                    "çözülen karede eksik 16-bit düzlem satırı",
                ))?;
                out.extend(
                    row[..row_bytes_len]
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|&pair| u16::from_le_bytes(pair)),
                );
            }
            Ok((w as u32, h as u32, PlaneStorage::U16(out), guard))
        }
    }
}

fn decode_av01_item(
    container: &AvifContainer<'_>,
    item_id: u32,
    config: &AvifDecodeConfig,
    threads_for_item: usize,
    is_alpha: bool,
    butce: &Arc<Butce>,
) -> Result<PlanarBuffer> {
    let props = container.item_properties(item_id)?;
    let mut av1c_opt: Option<&Av1CProperty> = None;
    let mut operating_point: u8 = 0;
    let mut layer_sel: Option<u16> = None;
    let mut a1lx_opt: Option<[usize; 3]> = None;
    let mut has_ispe = false;

    for (p, _) in props {
        match p {
            ItemProperty::Av1C(av1c) => av1c_opt = Some(av1c),
            ItemProperty::A1op(op) => operating_point = *op,
            ItemProperty::A1lx(sizes) => a1lx_opt = Some(*sizes),
            ItemProperty::Lsel(ls) => layer_sel = Some(*ls),
            ItemProperty::Ispe { width, height } => {
                has_ispe = true;
                check_pixel_and_byte_budget(*width, *height, 0, config)?;
            }
            _ => {}
        }
    }
    if config.strict && !has_ispe {
        return Err(Error::InvalidData(
            "katı kipte (strict) görüntü öğesinde ispe özelliği zorunludur",
        ));
    }

    let (payload, payload_guard) = container.resolve_item_data(item_id, butce)?;

    // `lsel` (0..=3) ile birlikte `a1lx` mevcutsa her `a1lx` katman diliminin taşıdığı gerçek
    // OBU `spatial_id` değeri taranır; böylece bitişik olmayan uzamsal katman kimliklerinde
    // (ör. `spatial_id = {0, 2}`, `lsel = 2`) doğru önek dilimi seçilir.
    let target_layer_id = layer_sel.filter(|&ls| ls != 0xFFFF);
    let decode_payload: &[u8] = if let Some(target_layer) = target_layer_id {
        if let Some(a1lx) = a1lx_opt {
            let mut layer_ranges = [(0usize, 0usize); 4];
            let mut layer_count = 0usize;
            let mut sum = 0usize;
            for &sz in &a1lx {
                if sz == 0 {
                    break;
                }
                let end = sum
                    .checked_add(sz)
                    .ok_or(Error::InvalidData("a1lx katman boyut toplamı taşması"))?;
                if end > payload.len() {
                    return Err(Error::InvalidData(
                        "a1lx katman boyutları öğe boyutunu aşıyor",
                    ));
                }
                layer_ranges[layer_count] = (sum, end);
                layer_count += 1;
                sum = end;
            }
            if sum < payload.len() && layer_count < 4 {
                layer_ranges[layer_count] = (sum, payload.len());
                layer_count += 1;
            }
            let mut matched_prefix_end: Option<usize> = None;
            for &(chunk_start, chunk_end) in &layer_ranges[..layer_count] {
                let chunk = &payload[chunk_start..chunk_end];
                let mut off = 0usize;
                while let Some(obu) =
                    crate::bitstream::split_next_with(chunk, &mut off, config.strict)?
                {
                    if matches!(
                        obu.header.obu_type,
                        crate::bitstream::ObuType::Frame
                            | crate::bitstream::ObuType::FrameHeader
                            | crate::bitstream::ObuType::RedundantFrameHeader
                    ) {
                        let sid = obu.header.extension.map_or(0u8, |e| e.spatial_id);
                        if u16::from(sid) == target_layer {
                            matched_prefix_end = Some(chunk_end);
                            break;
                        }
                    }
                }
                if matched_prefix_end.is_some() {
                    break;
                }
            }
            let prefix_len = matched_prefix_end.ok_or(Error::InvalidData(
                "lsel hedef katmanı a1lx katman dilimlerinde bulunamadı",
            ))?;
            &payload[..prefix_len]
        } else {
            &payload
        }
    } else {
        &payload
    };

    let dec_config = Config {
        n_threads: threads_for_item,
        max_frame_delay: 1,
        apply_grain: config.apply_grain,
        strict_std_compliance: config.strict,
        frame_size_limit: config.max_pixels,
        allocation_limit: config.allocation_limit,
        operating_point,
        ..Config::default()
    };
    const AV1_DECODER_INSTANCE_OVERHEAD_BYTES: usize = 4096;
    let _decoder_guard = Butce::tut(
        butce,
        AV1_DECODER_INSTANCE_OVERHEAD_BYTES.saturating_add(decode_payload.len()),
    )?;
    let mut decoder = Decoder::new_with_shared_budget(dec_config, Arc::clone(butce))?;

    if let Some(av1c) = av1c_opt
        && !av1c.config_obus.is_empty()
    {
        decoder.send_packet(&Packet::new(&av1c.config_obus))?;
    }
    decoder.send_packet(&Packet::new(decode_payload))?;
    decoder.flush();

    let mut chosen_frame: Option<Frame> = None;
    while let Some((frame, meta)) = decoder.receive_frame_capi(true)? {
        if let Some(target_layer) = target_layer_id {
            let spatial_id = meta.frame_hdr.as_ref().map_or(0u8, |h| h.spatial_id);
            if u16::from(spatial_id) == target_layer {
                chosen_frame = Some(frame);
                break;
            }
        } else {
            chosen_frame = Some(frame);
        }
    }

    let frame = chosen_frame.ok_or(Error::InvalidData(if target_layer_id.is_some() {
        "lsel ile seçilen uzamsal katman AV1 öğesinde bulunamadı"
    } else {
        "AV1 öğesinden görünür kare üretilemedi"
    }))?;
    let seq_color = decoder.sequence_header().map(|seq| ColorInfo {
        color_primaries: u16::from(seq.pri),
        transfer_characteristics: u16::from(seq.trc),
        matrix_coefficients: u16::from(seq.mtrx),
        full_range: seq.color_range != 0,
    });
    // Çözücü ve girdi yükü artık gerekli değil; düzlem kopyalamasından önce bütçeyi iade et.
    drop(decoder);
    drop(_decoder_guard);
    drop((payload, payload_guard));
    let _frame_guard = Butce::tut(butce, frame.allocation_bytes())?;

    let bit_depth = frame.bit_depth();
    let (w, h, mut y_plane, mut planar_guard) = extract_plane(&frame, PlaneType::Y, butce)?;
    let (cw, ch, u_plane, v_plane, chroma_format) = if is_alpha {
        (
            0,
            0,
            PlaneStorage::empty_for(bit_depth),
            PlaneStorage::empty_for(bit_depth),
            ChromaFormat::Monochrome,
        )
    } else {
        let (cw, ch, u, ug) = extract_plane(&frame, PlaneType::U, butce)?;
        planar_guard.birlestir(ug);
        let (_, _, v, vg) = extract_plane(&frame, PlaneType::V, butce)?;
        planar_guard.birlestir(vg);
        (cw, ch, u, v, frame.chroma_format())
    };

    // AVIF v1.0.0 geriye dönük uyumluluğu (`libavif` `avifImageLimitedToFullAlpha`):
    // Alfa düzlemi sınırlı aralıkta (`color_range == 0`) kodlanmışsa tam aralığa genişletilir.
    if is_alpha && seq_color.is_some_and(|c| !c.full_range) {
        match &mut y_plane {
            PlaneStorage::U8(samples) => {
                for sample in samples {
                    *sample = LIMITED_TO_FULL_Y_8BIT[usize::from(*sample)];
                }
            }
            PlaneStorage::U16(samples) => {
                for sample in samples {
                    *sample = limited_to_full_y_u16(bit_depth, *sample);
                }
            }
        }
    }

    Ok(PlanarBuffer {
        width: w,
        height: h,
        chroma_width: cw,
        chroma_height: ch,
        bit_depth,
        chroma_format,
        y_plane,
        u_plane,
        v_plane,
        seq_color,
        budget_guard: planar_guard,
    })
}

fn crop_slice<T: Copy>(
    src: &[T],
    old_w: usize,
    x0: usize,
    y0: usize,
    new_w: usize,
    new_h: usize,
) -> Result<Vec<T>> {
    let count = new_w.checked_mul(new_h).ok_or(Error::SizeOverflow {
        stage: SizeStage::PlaneArea,
        lhs: new_w,
        rhs: new_h,
    })?;
    let mut dst = Vec::new();
    dst.try_reserve_exact(count)
        .map_err(|_| Error::AllocationFailed {
            requested: count.saturating_mul(core::mem::size_of::<T>()),
        })?;
    for r in 0..new_h {
        let src_start = (y0 + r) * old_w + x0;
        dst.extend_from_slice(&src[src_start..src_start + new_w]);
    }
    Ok(dst)
}

fn crop_storage(
    storage: &PlaneStorage,
    old_w: usize,
    x0: usize,
    y0: usize,
    new_w: usize,
    new_h: usize,
    butce: &Arc<Butce>,
) -> Result<(PlaneStorage, ButceTutucu)> {
    let count = new_w.checked_mul(new_h).ok_or(Error::SizeOverflow {
        stage: SizeStage::PlaneArea,
        lhs: new_w,
        rhs: new_h,
    })?;
    match storage {
        PlaneStorage::U8(src) => {
            let guard = Butce::tut(butce, count)?;
            Ok((
                PlaneStorage::U8(crop_slice(src, old_w, x0, y0, new_w, new_h)?),
                guard,
            ))
        }
        PlaneStorage::U16(src) => {
            let bytes = count.checked_mul(2).ok_or(Error::SizeOverflow {
                stage: SizeStage::TotalAllocation,
                lhs: count,
                rhs: 2,
            })?;
            let guard = Butce::tut(butce, bytes)?;
            Ok((
                PlaneStorage::U16(crop_slice(src, old_w, x0, y0, new_w, new_h)?),
                guard,
            ))
        }
    }
}

pub(crate) fn crop_planar(
    buf: &mut PlanarBuffer,
    x0: u32,
    y0: u32,
    crop_w: u32,
    crop_h: u32,
    butce: &Arc<Butce>,
) -> Result<()> {
    if x0 == 0 && y0 == 0 && crop_w == buf.width && crop_h == buf.height {
        return Ok(());
    }
    if x0.checked_add(crop_w).is_none_or(|end| end > buf.width)
        || y0.checked_add(crop_h).is_none_or(|end| end > buf.height)
    {
        return Err(Error::InvalidData(
            "kırpma bölgesi düzlem sınırları dışında",
        ));
    }

    let old_w = buf.width as usize;
    let new_w = crop_w as usize;
    let new_h = crop_h as usize;

    let old_y_bytes = plane_storage_bytes(&buf.y_plane);
    let (new_y, new_y_guard) = crop_storage(
        &buf.y_plane,
        old_w,
        x0 as usize,
        y0 as usize,
        new_w,
        new_h,
        butce,
    )?;
    buf.y_plane = new_y;
    buf.budget_guard.dusur(old_y_bytes);
    buf.budget_guard.birlestir(new_y_guard);
    buf.width = crop_w;
    buf.height = crop_h;

    if buf.chroma_format != ChromaFormat::Monochrome {
        let sub_x = usize::from(matches!(
            buf.chroma_format,
            ChromaFormat::C420 | ChromaFormat::C422
        ));
        let sub_y = usize::from(matches!(buf.chroma_format, ChromaFormat::C420));
        let cx0 = (x0 as usize) >> sub_x;
        let cy0 = (y0 as usize) >> sub_y;
        let (new_cw, new_ch) = buf
            .chroma_format
            .chroma_dimensions(new_w, new_h)?
            .unwrap_or((0, 0));
        let old_cw = buf.chroma_width as usize;
        if cx0 + new_cw > old_cw || cy0 + new_ch > (buf.chroma_height as usize) {
            return Err(Error::InvalidData(
                "kroma kırpma bölgesi düzlem sınırları dışında",
            ));
        }
        let old_u_bytes = plane_storage_bytes(&buf.u_plane);
        let (new_u, new_u_guard) =
            crop_storage(&buf.u_plane, old_cw, cx0, cy0, new_cw, new_ch, butce)?;
        buf.u_plane = new_u;
        buf.budget_guard.dusur(old_u_bytes);
        buf.budget_guard.birlestir(new_u_guard);

        let old_v_bytes = plane_storage_bytes(&buf.v_plane);
        let (new_v, new_v_guard) =
            crop_storage(&buf.v_plane, old_cw, cx0, cy0, new_cw, new_ch, butce)?;
        buf.v_plane = new_v;
        buf.budget_guard.dusur(old_v_bytes);
        buf.budget_guard.birlestir(new_v_guard);

        buf.chroma_width = new_cw as u32;
        buf.chroma_height = new_ch as u32;
    }
    Ok(())
}

fn rotate_slice_ccw<T: Copy + Default>(
    src: &[T],
    w: usize,
    h: usize,
    angle: u8,
) -> Result<(usize, usize, Vec<T>)> {
    match angle & 3 {
        1 => {
            let mut dst = dolu_vec(w * h, T::default())?;
            for ny in 0..w {
                let sx = w - 1 - ny;
                for nx in 0..h {
                    let sy = nx;
                    dst[ny * h + nx] = src[sy * w + sx];
                }
            }
            Ok((h, w, dst))
        }
        2 => {
            let mut dst = dolu_vec(w * h, T::default())?;
            for ny in 0..h {
                let sy = h - 1 - ny;
                for nx in 0..w {
                    let sx = w - 1 - nx;
                    dst[ny * w + nx] = src[sy * w + sx];
                }
            }
            Ok((w, h, dst))
        }
        3 => {
            let mut dst = dolu_vec(w * h, T::default())?;
            for ny in 0..w {
                let sx = ny;
                for nx in 0..h {
                    let sy = h - 1 - nx;
                    dst[ny * h + nx] = src[sy * w + sx];
                }
            }
            Ok((h, w, dst))
        }
        _ => Ok((w, h, kopya_vec(src)?)),
    }
}

fn rotate_plane_ccw(
    src: &PlaneStorage,
    w: usize,
    h: usize,
    angle: u8,
    butce: &Arc<Butce>,
) -> Result<(usize, usize, PlaneStorage, ButceTutucu)> {
    let bytes = plane_storage_bytes(src);
    let guard = Butce::tut(butce, bytes)?;
    match src {
        PlaneStorage::U8(s) => {
            let (nw, nh, dst) = rotate_slice_ccw(s, w, h, angle)?;
            Ok((nw, nh, PlaneStorage::U8(dst), guard))
        }
        PlaneStorage::U16(s) => {
            let (nw, nh, dst) = rotate_slice_ccw(s, w, h, angle)?;
            Ok((nw, nh, PlaneStorage::U16(dst), guard))
        }
    }
}

fn upsample_slice_c422_to_c444<T: Copy>(
    src: &[T],
    w: usize,
    h: usize,
    cw: usize,
) -> Result<Vec<T>> {
    let count = w.checked_mul(h).ok_or(Error::SizeOverflow {
        stage: SizeStage::PlaneArea,
        lhs: w,
        rhs: h,
    })?;
    let mut dst = Vec::new();
    dst.try_reserve_exact(count)
        .map_err(|_| Error::AllocationFailed {
            requested: count.saturating_mul(core::mem::size_of::<T>()),
        })?;
    let max_cx = cw.saturating_sub(1);
    for y in 0..h {
        let row_off = y * cw;
        for x in 0..w {
            let cx = (x / 2).min(max_cx);
            dst.push(src[row_off + cx]);
        }
    }
    Ok(dst)
}

fn upsample_c422_to_c444(buf: &mut PlanarBuffer, butce: &Arc<Butce>) -> Result<()> {
    if buf.chroma_format != ChromaFormat::C422 {
        return Ok(());
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let cw = buf.chroma_width as usize;
    let new_bytes = w
        .saturating_mul(h)
        .saturating_mul(PlaneStorage::bytes_per_sample(buf.bit_depth));

    let old_u_bytes = plane_storage_bytes(&buf.u_plane);
    let ug = Butce::tut(butce, new_bytes)?;
    let new_u = match &buf.u_plane {
        PlaneStorage::U8(s) => PlaneStorage::U8(upsample_slice_c422_to_c444(s, w, h, cw)?),
        PlaneStorage::U16(s) => PlaneStorage::U16(upsample_slice_c422_to_c444(s, w, h, cw)?),
    };
    buf.u_plane = new_u;
    buf.budget_guard.dusur(old_u_bytes);
    buf.budget_guard.birlestir(ug);

    let old_v_bytes = plane_storage_bytes(&buf.v_plane);
    let vg = Butce::tut(butce, new_bytes)?;
    let new_v = match &buf.v_plane {
        PlaneStorage::U8(s) => PlaneStorage::U8(upsample_slice_c422_to_c444(s, w, h, cw)?),
        PlaneStorage::U16(s) => PlaneStorage::U16(upsample_slice_c422_to_c444(s, w, h, cw)?),
    };
    buf.v_plane = new_v;
    buf.budget_guard.dusur(old_v_bytes);
    buf.budget_guard.birlestir(vg);

    buf.chroma_width = buf.width;
    buf.chroma_height = buf.height;
    buf.chroma_format = ChromaFormat::C444;
    Ok(())
}

pub(crate) fn apply_irot(buf: &mut PlanarBuffer, angle: u8, butce: &Arc<Butce>) -> Result<()> {
    let a = angle & 3;
    if a == 0 {
        return Ok(());
    }
    if (a & 1) != 0 && buf.chroma_format == ChromaFormat::C422 {
        upsample_c422_to_c444(buf, butce)?;
    }
    let old_y_bytes = plane_storage_bytes(&buf.y_plane);
    let (nw, nh, ny, yg) = rotate_plane_ccw(
        &buf.y_plane,
        buf.width as usize,
        buf.height as usize,
        a,
        butce,
    )?;
    buf.width = nw as u32;
    buf.height = nh as u32;
    buf.y_plane = ny;
    buf.budget_guard.dusur(old_y_bytes);
    buf.budget_guard.birlestir(yg);

    if buf.chroma_format != ChromaFormat::Monochrome {
        let old_cw = buf.chroma_width as usize;
        let old_ch = buf.chroma_height as usize;

        let old_u_bytes = plane_storage_bytes(&buf.u_plane);
        let (ncw, nch, nu, ug) = rotate_plane_ccw(&buf.u_plane, old_cw, old_ch, a, butce)?;
        buf.u_plane = nu;
        buf.budget_guard.dusur(old_u_bytes);
        buf.budget_guard.birlestir(ug);

        let old_v_bytes = plane_storage_bytes(&buf.v_plane);
        let (_, _, nv, vg) = rotate_plane_ccw(&buf.v_plane, old_cw, old_ch, a, butce)?;
        buf.v_plane = nv;
        buf.budget_guard.dusur(old_v_bytes);
        buf.budget_guard.birlestir(vg);

        buf.chroma_width = ncw as u32;
        buf.chroma_height = nch as u32;
    }
    Ok(())
}

fn mirror_slice<T>(plane: &mut [T], w: usize, h: usize, axis: u8) {
    if w == 0 || h == 0 {
        return;
    }
    if (axis & 1) == 0 {
        for y in 0..(h / 2) {
            let top_start = y * w;
            let bot_start = (h - 1 - y) * w;
            let (head, tail) = plane.split_at_mut(bot_start);
            head[top_start..top_start + w].swap_with_slice(&mut tail[..w]);
        }
    } else {
        for y in 0..h {
            let row = &mut plane[y * w..(y + 1) * w];
            row.reverse();
        }
    }
}

fn mirror_plane(plane: &mut PlaneStorage, w: usize, h: usize, axis: u8) {
    match plane {
        PlaneStorage::U8(s) => mirror_slice(s, w, h, axis),
        PlaneStorage::U16(s) => mirror_slice(s, w, h, axis),
    }
}

pub(crate) fn apply_imir(buf: &mut PlanarBuffer, axis: u8) {
    mirror_plane(
        &mut buf.y_plane,
        buf.width as usize,
        buf.height as usize,
        axis,
    );
    if buf.chroma_format != ChromaFormat::Monochrome {
        let cw = buf.chroma_width as usize;
        let ch = buf.chroma_height as usize;
        mirror_plane(&mut buf.u_plane, cw, ch, axis);
        mirror_plane(&mut buf.v_plane, cw, ch, axis);
    }
}

/// MIAF (`ISO/IEC 23000-22 §7.3.6.7`) normatif sırasıyla (`clap` → `irot` → `imir`)
/// öğe dönüşümlerini uygular.
fn apply_item_transforms(
    container: &AvifContainer<'_>,
    item_id: u32,
    buf: &mut PlanarBuffer,
    butce: &Arc<Butce>,
) -> Result<()> {
    let props = container.item_properties(item_id)?;
    let mut clap_opt: Option<&CleanAperture> = None;
    let mut irot_opt: Option<u8> = None;
    let mut imir_opt: Option<u8> = None;

    for (p, _) in props {
        match p {
            ItemProperty::Clap(c) => clap_opt = Some(c),
            ItemProperty::Irot(a) => irot_opt = Some(*a),
            ItemProperty::Imir(m) => imir_opt = Some(*m),
            _ => {}
        }
    }
    if let Some(clap) = clap_opt {
        let (x0, y0, cw, ch) = compute_clap_rect(buf.width, buf.height, clap)?;
        crop_planar(buf, x0, y0, cw, ch, butce)?;
    }
    if let Some(angle) = irot_opt {
        apply_irot(buf, angle, butce)?;
    }
    if let Some(axis) = imir_opt {
        apply_imir(buf, axis);
    }
    Ok(())
}

/// Grid karolarının yalnız kodlanmış `av01` öğeleri olmasını zorunlu kılar (libavif de grid
/// karosu olarak türetilmiş öğe kabul etmez).
///
/// Karo olarak `grid` ya da `iden` kabul edilseydi her düzey çözme işini karo sayısıyla
/// çarpardı: birkaç KB'lık bir dosya derinlik sınırına kadar `M^düzey` yaprak çözümü ve
/// her düzeyde yeni iş parçacıkları isteyebilirdi. Bu denetim tek bir grid katmanı bırakır;
/// toplam çözme işi böylece dosyadaki `dimg` referans sayısıyla doğrusal kalır.
fn ensure_grid_tiles_coded(
    container: &AvifContainer<'_>,
    tile_ids: impl IntoIterator<Item = u32>,
) -> Result<()> {
    for tid in tile_ids {
        let info = container
            .item_info(tid)
            .ok_or(Error::InvalidData("grid karosu iinf içinde bulunamadı"))?;
        if &info.item_type != b"av01" {
            return Err(Error::InvalidData(
                "grid karoları yalnız kodlanmış av01 öğeleri olabilir",
            ));
        }
    }
    Ok(())
}

fn decode_grid_tiles_into_canvas(
    container: &AvifContainer<'_>,
    grid: GridDescriptor,
    tile_ids: &[u32],
    config: &AvifDecodeConfig,
    is_alpha: bool,
    depth: usize,
    butce: &Arc<Butce>,
) -> Result<PlanarBuffer> {
    check_pixel_and_byte_budget(grid.output_width, grid.output_height, 0, config)?;
    let rows = grid.rows as usize;
    let cols = grid.cols as usize;
    let num_tiles = rows
        .checked_mul(cols)
        .ok_or(Error::InvalidData("grid karo sayısı taşması"))?;
    if tile_ids.len() != num_tiles || num_tiles == 0 {
        return Err(Error::InvalidData(
            "grid dimg referans sayısı rows*cols ile uyuşmuyor",
        ));
    }
    ensure_grid_tiles_coded(container, tile_ids.iter().copied())?;

    let has_dups = is_alpha && (1..tile_ids.len()).any(|i| tile_ids[..i].contains(&tile_ids[i]));
    let mut grid_plan_guard = ButceTutucu::bos(butce);
    let mut dedup_unique_ids: Vec<u32> = Vec::new();
    let mut cell_head: Vec<u32> = Vec::new();
    let mut cell_next: Vec<u32> = Vec::new();

    if has_dups {
        let pairs_bytes = num_tiles
            .checked_mul(core::mem::size_of::<(u32, u32)>())
            .ok_or(Error::SizeOverflow {
                stage: SizeStage::TotalAllocation,
                lhs: num_tiles,
                rhs: core::mem::size_of::<(u32, u32)>(),
            })?;
        let flat_bytes = num_tiles
            .checked_mul(core::mem::size_of::<u32>() * 3)
            .ok_or(Error::SizeOverflow {
                stage: SizeStage::TotalAllocation,
                lhs: num_tiles,
                rhs: core::mem::size_of::<u32>() * 3,
            })?;
        let total_plan_bytes = pairs_bytes
            .checked_add(flat_bytes)
            .ok_or(Error::SizeOverflow {
                stage: SizeStage::TotalAllocation,
                lhs: pairs_bytes,
                rhs: flat_bytes,
            })?;
        grid_plan_guard.ekle(total_plan_bytes)?;

        let mut pairs: Vec<(u32, u32)> = Vec::new();
        pairs
            .try_reserve_exact(num_tiles)
            .map_err(|_| Error::AllocationFailed {
                requested: pairs_bytes,
            })?;
        for (cell_idx, &tid) in tile_ids.iter().enumerate() {
            pairs.push((tid, cell_idx as u32));
        }
        pairs.sort_unstable();

        dedup_unique_ids
            .try_reserve_exact(num_tiles)
            .map_err(|_| Error::AllocationFailed {
                requested: num_tiles * core::mem::size_of::<u32>(),
            })?;
        cell_head
            .try_reserve_exact(num_tiles)
            .map_err(|_| Error::AllocationFailed {
                requested: num_tiles * core::mem::size_of::<u32>(),
            })?;
        cell_next
            .try_reserve_exact(num_tiles)
            .map_err(|_| Error::AllocationFailed {
                requested: num_tiles * core::mem::size_of::<u32>(),
            })?;
        cell_next.resize(num_tiles, u32::MAX);

        let mut p_idx = 0usize;
        while p_idx < pairs.len() {
            let tid = pairs[p_idx].0;
            let run_start = p_idx;
            while p_idx < pairs.len() && pairs[p_idx].0 == tid {
                p_idx += 1;
            }
            // Ters sırada zincirle ki okuma sırasında hücreler artan `(r, c)` sırasıyla gezilsin.
            let mut head = u32::MAX;
            for k in (run_start..p_idx).rev() {
                let c_idx = pairs[k].1;
                cell_next[c_idx as usize] = head;
                head = c_idx;
            }
            dedup_unique_ids.push(tid);
            cell_head.push(head);
        }
        drop(pairs);
        grid_plan_guard.dusur(pairs_bytes);
    }

    let unique_ids: &[u32] = if has_dups {
        &dedup_unique_ids
    } else {
        tile_ids
    };
    let num_unique = unique_ids.len();
    let total_threads = resolve_effective_threads(config.n_threads);

    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    let wave_size = total_threads.max(1).min(num_unique);
    #[cfg(not(all(feature = "std", not(target_arch = "wasm32"))))]
    let wave_size = 1usize;

    // 1. İlk dalgayı (`0..first_end`, en fazla `worker_count` karo) tuval ayrılmadan önce çöz;
    //    böylece ilk dalganın AV1 çözücüleri tuvalle çakışmaz ve `num_unique <= wave_size`
    //    durumunda seri ilk-karo darboğazı oluşmaz.
    let first_end = wave_size.min(num_unique);
    let first_wave_threads = if num_unique == 1 { total_threads } else { 1 };
    let mut first_wave = decode_grid_wave(
        container,
        &unique_ids[..first_end],
        config,
        first_wave_threads,
        is_alpha,
        depth,
        butce,
    )?;

    let first_tile = &first_wave[0];
    let tile_w = first_tile.width as usize;
    let tile_h = first_tile.height as usize;
    let tile_cw = first_tile.chroma_width as usize;
    let tile_ch = first_tile.chroma_height as usize;
    let bit_depth = first_tile.bit_depth;
    let chroma_format = first_tile.chroma_format;
    let seq_color = first_tile.seq_color;

    if tile_w == 0 || tile_h == 0 {
        return Err(Error::InvalidData("grid karo boyutu sıfır olamaz"));
    }
    for tile in &first_wave[1..] {
        validate_grid_tile(tile, tile_w, tile_h, bit_depth, chroma_format, is_alpha)?;
    }

    let out_w = grid.output_width as usize;
    let out_h = grid.output_height as usize;
    if cols.checked_mul(tile_w).is_none_or(|cw| cw < out_w)
        || rows.checked_mul(tile_h).is_none_or(|ch| ch < out_h)
    {
        return Err(Error::InvalidData(
            "grid toplam karo alanı output_width x output_height değerinden küçük",
        ));
    }

    let (out_cw, out_ch) = if is_alpha {
        (0, 0)
    } else {
        chroma_format
            .chroma_dimensions(out_w, out_h)?
            .unwrap_or((0, 0))
    };
    let y_samples = out_w.checked_mul(out_h).ok_or(Error::SizeOverflow {
        stage: SizeStage::PlaneArea,
        lhs: out_w,
        rhs: out_h,
    })?;
    let c_samples = out_cw.checked_mul(out_ch).ok_or(Error::SizeOverflow {
        stage: SizeStage::PlaneArea,
        lhs: out_cw,
        rhs: out_ch,
    })?;
    let bps = PlaneStorage::bytes_per_sample(bit_depth);
    let total_samples = y_samples
        .checked_add(c_samples.checked_mul(2).ok_or(Error::SizeOverflow {
            stage: SizeStage::TotalAllocation,
            lhs: c_samples,
            rhs: 2,
        })?)
        .ok_or(Error::SizeOverflow {
            stage: SizeStage::TotalAllocation,
            lhs: y_samples,
            rhs: c_samples,
        })?;
    let total_bytes = total_samples.checked_mul(bps).ok_or(Error::SizeOverflow {
        stage: SizeStage::TotalAllocation,
        lhs: total_samples,
        rhs: bps,
    })?;
    check_pixel_and_byte_budget(grid.output_width, grid.output_height, 0, config)?;
    let canvas_guard = Butce::tut(butce, total_bytes)?;

    let copy_chroma = !is_alpha && chroma_format != ChromaFormat::Monochrome;
    let mut y_canvas = match bit_depth {
        BitDepth::Eight => PlaneStorage::U8(dolu_vec(y_samples, 0u8)?),
        BitDepth::Ten | BitDepth::Twelve => PlaneStorage::U16(dolu_vec(y_samples, 0u16)?),
    };
    let mut u_canvas = match bit_depth {
        BitDepth::Eight => PlaneStorage::U8(dolu_vec(c_samples, 0u8)?),
        BitDepth::Ten | BitDepth::Twelve => PlaneStorage::U16(dolu_vec(c_samples, 0u16)?),
    };
    let mut v_canvas = match bit_depth {
        BitDepth::Eight => PlaneStorage::U8(dolu_vec(c_samples, 0u8)?),
        BitDepth::Ten | BitDepth::Twelve => PlaneStorage::U16(dolu_vec(c_samples, 0u16)?),
    };

    let geom = GridCanvasGeom {
        tile_w,
        tile_h,
        tile_cw,
        tile_ch,
        out_w,
        out_h,
        out_cw,
        out_ch,
        copy_chroma,
    };

    // İlk dalgadaki tüm karoları tuvale yaz ve tamponlarını bir sonraki dalgadan önce serbest bırak.
    for (i, tile) in first_wave.drain(..).enumerate() {
        copy_tile_for_unique_index(
            &tile,
            i,
            cols,
            has_dups,
            &cell_head,
            &cell_next,
            &geom,
            &mut y_canvas,
            &mut u_canvas,
            &mut v_canvas,
        );
        drop(tile);
    }
    drop(first_wave);

    // 2. Kalan karoları en fazla `wave_size` (worker_count) büyüklüğünde dalgalar halinde çöz,
    //    tuvale yaz ve bir sonraki dalgaya geçmeden tüm dalga tamponlarını iade et.
    let mut offset = first_end;
    while offset < num_unique {
        let end = (offset + wave_size).min(num_unique);
        let mut wave_tiles = decode_grid_wave(
            container,
            &unique_ids[offset..end],
            config,
            1,
            is_alpha,
            depth,
            butce,
        )?;
        for (i, tile) in wave_tiles.drain(..).enumerate() {
            validate_grid_tile(&tile, tile_w, tile_h, bit_depth, chroma_format, is_alpha)?;
            copy_tile_for_unique_index(
                &tile,
                offset + i,
                cols,
                has_dups,
                &cell_head,
                &cell_next,
                &geom,
                &mut y_canvas,
                &mut u_canvas,
                &mut v_canvas,
            );
            drop(tile);
        }
        offset = end;
    }
    drop((dedup_unique_ids, cell_head, cell_next, grid_plan_guard));

    Ok(PlanarBuffer {
        width: grid.output_width,
        height: grid.output_height,
        chroma_width: out_cw as u32,
        chroma_height: out_ch as u32,
        bit_depth,
        chroma_format: if is_alpha {
            ChromaFormat::Monochrome
        } else {
            chroma_format
        },
        y_plane: y_canvas,
        u_plane: u_canvas,
        v_plane: v_canvas,
        seq_color,
        budget_guard: canvas_guard,
    })
}

fn decode_grid_wave(
    container: &AvifContainer<'_>,
    wave_ids: &[u32],
    config: &AvifDecodeConfig,
    single_item_threads: usize,
    is_alpha: bool,
    depth: usize,
    butce: &Arc<Butce>,
) -> Result<Vec<PlanarBuffer>> {
    let wave_vec_bytes = wave_ids
        .len()
        .checked_mul(core::mem::size_of::<PlanarBuffer>())
        .ok_or(Error::SizeOverflow {
            stage: SizeStage::TotalAllocation,
            lhs: wave_ids.len(),
            rhs: core::mem::size_of::<PlanarBuffer>(),
        })?;
    let _wave_guard = Butce::tut(butce, wave_vec_bytes)?;

    #[cfg(all(feature = "std", not(target_arch = "wasm32")))]
    if wave_ids.len() > 1 {
        let mut wave_tiles: Vec<PlanarBuffer> = Vec::new();
        wave_tiles
            .try_reserve_exact(wave_ids.len())
            .map_err(|_| Error::AllocationFailed {
                requested: wave_vec_bytes,
            })?;
        /// Dalgadaki bir karonun çözüleceği yer: açılabilen iş parçacığı ya da (iş parçacığı
        /// açılamadıysa) çağıran iş parçacığında seri çözüm.
        enum KaroIsi<'scope> {
            Paralel(std::thread::ScopedJoinHandle<'scope, Result<PlanarBuffer>>),
            Seri(u32),
        }
        std::thread::scope(|s| {
            let mut handles: Vec<KaroIsi<'_>> = Vec::new();
            handles
                .try_reserve_exact(wave_ids.len())
                .map_err(|_| Error::AllocationFailed {
                    requested: wave_ids.len() * core::mem::size_of::<KaroIsi<'_>>(),
                })?;
            for &tid in wave_ids {
                let butce_w = Arc::clone(butce);
                // `Scope::spawn` iş parçacığı açılamazsa panikler; `Builder::spawn_scoped`
                // hatayı döndürür ve karo seri yola düşer.
                let is = std::thread::Builder::new()
                    .spawn_scoped(s, move || {
                        decode_item_planar(container, tid, config, 1, is_alpha, depth + 1, &butce_w)
                    })
                    .map_or(KaroIsi::Seri(tid), KaroIsi::Paralel);
                handles.push(is);
            }
            for is in handles {
                let tile = match is {
                    KaroIsi::Paralel(h) => h
                        .join()
                        .map_err(|_| Error::InvalidState("grid karo iş parçacığı sonlandı"))??,
                    KaroIsi::Seri(tid) => {
                        decode_item_planar(container, tid, config, 1, is_alpha, depth + 1, butce)?
                    }
                };
                wave_tiles.push(tile);
            }
            Ok::<(), Error>(())
        })?;
        return Ok(wave_tiles);
    }

    let mut wave_tiles: Vec<PlanarBuffer> = Vec::new();
    wave_tiles
        .try_reserve_exact(wave_ids.len())
        .map_err(|_| Error::AllocationFailed {
            requested: wave_vec_bytes,
        })?;
    for &tid in wave_ids {
        wave_tiles.push(decode_item_planar(
            container,
            tid,
            config,
            single_item_threads,
            is_alpha,
            depth + 1,
            butce,
        )?);
    }
    Ok(wave_tiles)
}

struct GridCanvasGeom {
    tile_w: usize,
    tile_h: usize,
    tile_cw: usize,
    tile_ch: usize,
    out_w: usize,
    out_h: usize,
    out_cw: usize,
    out_ch: usize,
    copy_chroma: bool,
}

#[inline]
fn validate_grid_tile(
    tile: &PlanarBuffer,
    tile_w: usize,
    tile_h: usize,
    bit_depth: BitDepth,
    chroma_format: ChromaFormat,
    is_alpha: bool,
) -> Result<()> {
    if tile.width as usize != tile_w
        || tile.height as usize != tile_h
        || tile.bit_depth != bit_depth
        || (!is_alpha && tile.chroma_format != chroma_format)
    {
        return Err(Error::InvalidData(
            "grid karolarının boyut veya biçimleri uyumsuz",
        ));
    }
    Ok(())
}

fn copy_tile_slice<T: Copy>(
    src: (&[T], &[T], &[T]),
    cells: &[(usize, usize)],
    g: &GridCanvasGeom,
    dst: (&mut [T], &mut [T], &mut [T]),
) {
    let (ty_src, tu_src, tv_src) = src;
    let (y_canvas, u_canvas, v_canvas) = dst;
    for &(r, c) in cells {
        let dst_y0 = r * g.tile_h;
        let dst_x0 = c * g.tile_w;
        if dst_y0 < g.out_h && dst_x0 < g.out_w {
            let copy_h = g.tile_h.min(g.out_h - dst_y0);
            let copy_w = g.tile_w.min(g.out_w - dst_x0);
            for ty in 0..copy_h {
                let src_off = ty * g.tile_w;
                let dst_off = (dst_y0 + ty) * g.out_w + dst_x0;
                y_canvas[dst_off..dst_off + copy_w]
                    .copy_from_slice(&ty_src[src_off..src_off + copy_w]);
            }
        }
        if g.copy_chroma {
            let dst_cy0 = r * g.tile_ch;
            let dst_cx0 = c * g.tile_cw;
            if dst_cy0 < g.out_ch && dst_cx0 < g.out_cw {
                let copy_ch = g.tile_ch.min(g.out_ch - dst_cy0);
                let copy_cw = g.tile_cw.min(g.out_cw - dst_cx0);
                for tcy in 0..copy_ch {
                    let src_off = tcy * g.tile_cw;
                    let dst_off = (dst_cy0 + tcy) * g.out_cw + dst_cx0;
                    u_canvas[dst_off..dst_off + copy_cw]
                        .copy_from_slice(&tu_src[src_off..src_off + copy_cw]);
                    v_canvas[dst_off..dst_off + copy_cw]
                        .copy_from_slice(&tv_src[src_off..src_off + copy_cw]);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn copy_tile_for_unique_index(
    tile: &PlanarBuffer,
    u_idx: usize,
    cols: usize,
    has_dups: bool,
    cell_head: &[u32],
    cell_next: &[u32],
    g: &GridCanvasGeom,
    y_canvas: &mut PlaneStorage,
    u_canvas: &mut PlaneStorage,
    v_canvas: &mut PlaneStorage,
) {
    if !has_dups {
        let cell = [(u_idx / cols, u_idx % cols)];
        copy_tile_into_canvas(tile, &cell, g, y_canvas, u_canvas, v_canvas);
        return;
    }
    let mut cur = cell_head[u_idx];
    while cur != u32::MAX {
        let idx = cur as usize;
        let cell = [(idx / cols, idx % cols)];
        copy_tile_into_canvas(tile, &cell, g, y_canvas, u_canvas, v_canvas);
        cur = cell_next[idx];
    }
}

fn copy_tile_into_canvas(
    tile: &PlanarBuffer,
    cells: &[(usize, usize)],
    g: &GridCanvasGeom,
    y_canvas: &mut PlaneStorage,
    u_canvas: &mut PlaneStorage,
    v_canvas: &mut PlaneStorage,
) {
    match (y_canvas, u_canvas, v_canvas) {
        (PlaneStorage::U8(yc), PlaneStorage::U8(uc), PlaneStorage::U8(vc)) => {
            copy_tile_slice(
                (
                    tile.y_plane.as_u8().unwrap_or(&[]),
                    tile.u_plane.as_u8().unwrap_or(&[]),
                    tile.v_plane.as_u8().unwrap_or(&[]),
                ),
                cells,
                g,
                (yc, uc, vc),
            );
        }
        (PlaneStorage::U16(yc), PlaneStorage::U16(uc), PlaneStorage::U16(vc)) => {
            copy_tile_slice(
                (
                    tile.y_plane.as_u16().unwrap_or(&[]),
                    tile.u_plane.as_u16().unwrap_or(&[]),
                    tile.v_plane.as_u16().unwrap_or(&[]),
                ),
                cells,
                g,
                (yc, uc, vc),
            );
        }
        _ => {}
    }
}

fn decode_grid_item(
    container: &AvifContainer<'_>,
    item_id: u32,
    config: &AvifDecodeConfig,
    is_alpha: bool,
    depth: usize,
    butce: &Arc<Butce>,
) -> Result<PlanarBuffer> {
    let (grid_bytes, grid_guard) = container.resolve_item_data(item_id, butce)?;
    let grid = parse_grid_descriptor(&grid_bytes)?;
    drop((grid_bytes, grid_guard));
    let (tile_ids, tile_ids_guard) = container.references_from_slice(item_id, b"dimg", butce)?;
    let res =
        decode_grid_tiles_into_canvas(container, grid, &tile_ids, config, is_alpha, depth, butce);
    drop((tile_ids, tile_ids_guard));
    res
}

fn decode_item_planar(
    container: &AvifContainer<'_>,
    item_id: u32,
    config: &AvifDecodeConfig,
    threads_for_item: usize,
    is_alpha: bool,
    depth: usize,
    butce: &Arc<Butce>,
) -> Result<PlanarBuffer> {
    if depth > 8 {
        return Err(Error::InvalidData(
            "AVIF öğe referans zinciri derinlik sınırını aştı",
        ));
    }
    let info = container
        .item_info(item_id)
        .ok_or(Error::InvalidData("AVIF öğesi iinf içinde bulunamadı"))?;

    let mut buf = match &info.item_type {
        b"av01" => decode_av01_item(
            container,
            item_id,
            config,
            threads_for_item,
            is_alpha,
            butce,
        )?,
        b"grid" => decode_grid_item(container, item_id, config, is_alpha, depth, butce)?,
        b"iden" => {
            let mut targets = container.references_from(item_id, b"dimg");
            let (Some(target_0), None) = (targets.next(), targets.next()) else {
                return Err(Error::InvalidData(
                    "iden öğesi tek bir dimg referansı içermelidir",
                ));
            };
            decode_item_planar(
                container,
                target_0,
                config,
                threads_for_item,
                is_alpha,
                depth + 1,
                butce,
            )?
        }
        _ => return Err(Error::Unsupported("desteklenmeyen AVIF görüntü öğesi türü")),
    };

    let props = container.item_properties(item_id)?;
    for (p, _) in props {
        if let ItemProperty::Ispe { width, height } = p
            && (*width < buf.width || *height < buf.height)
        {
            let target_w = (*width).min(buf.width);
            let target_h = (*height).min(buf.height);
            crop_planar(&mut buf, 0, 0, target_w, target_h, butce)?;
        }
    }

    apply_item_transforms(container, item_id, &mut buf, butce)?;
    Ok(buf)
}

/// Varsayılan yapılandırma ([`AvifDecodeConfig::default`]) ile bir `.avif` bayt dilimini çözer.
pub fn decode(data: &[u8]) -> Result<AvifImage> {
    decode_with(data, &AvifDecodeConfig::default())
}

/// Belirtilen [`AvifDecodeConfig`] yapılandırması ile bir `.avif` bayt dilimini çözer.
pub fn decode_with(data: &[u8], config: &AvifDecodeConfig) -> Result<AvifImage> {
    let butce = Butce::yeni_arc(config.allocation_limit, 0)?;
    decode_with_budget(data, config, butce)
}

pub(crate) fn decode_with_budget(
    data: &[u8],
    config: &AvifDecodeConfig,
    butce: Arc<Butce>,
) -> Result<AvifImage> {
    let container = parse_container(data, &butce)?;
    let img = decode_from_container(&container, config, Arc::clone(&butce), None)?;
    drop(container);
    butce.taban_guncelle(0);
    Ok(img)
}

fn decode_from_container(
    container: &AvifContainer<'_>,
    config: &AvifDecodeConfig,
    butce: Arc<Butce>,
    mut cached_info: Option<(AvifInfo, ButceTutucu)>,
) -> Result<AvifImage> {
    let primary_id = container.primary_item_id;

    let color_buf = decode_item_planar(
        container,
        primary_id,
        config,
        config.n_threads,
        false,
        0,
        &butce,
    )?;
    let mut img_guard = color_buf.budget_guard;

    let props = container.item_properties(primary_id)?;
    let mut color = color_buf.seq_color.unwrap_or_default();
    let mut icc_profile: Option<Vec<u8>> = None;
    let mut clli = None;
    let mut mdcv = None;
    let mut pasp = None;

    if let Some((ref mut ci, _)) = cached_info {
        for (p, _) in props {
            match p {
                ItemProperty::ColrNclx(c) => color = *c,
                ItemProperty::Clli(c) => clli = Some(*c),
                ItemProperty::Mdcv(m) => mdcv = Some(*m),
                ItemProperty::Pasp(pa) => pasp = Some(*pa),
                _ => {}
            }
        }
        icc_profile = ci.icc_profile.take();
    } else {
        for (p, _) in props {
            match p {
                ItemProperty::ColrNclx(c) => color = *c,
                ItemProperty::ColrIcc(icc) => {
                    if let Some(ref prev) = icc_profile {
                        img_guard.dusur(prev.len());
                    }
                    let (v, g) = copy_bytes_budgeted(icc, &butce)?;
                    img_guard.birlestir(g);
                    icc_profile = Some(v);
                }
                ItemProperty::Clli(c) => clli = Some(*c),
                ItemProperty::Mdcv(m) => mdcv = Some(*m),
                ItemProperty::Pasp(pa) => pasp = Some(*pa),
                _ => {}
            }
        }
    }

    let alpha_src_opt = find_alpha_source(container, primary_id, &butce)?;
    let alpha_src = alpha_src_opt.as_ref().map(|(s, _)| s);
    let premultiplied_alpha = is_premultiplied_alpha(container, primary_id, alpha_src);

    let alpha_plane = if let Some(src) = alpha_src {
        let alpha_buf = match src {
            AlphaSource::Direct(aid) => {
                let mut ab =
                    decode_item_planar(container, *aid, config, config.n_threads, true, 0, &butce)?;
                // libavif uyumu: alfa öğesinin kendi irot/imir/clap dönüşümleri dikkate alınmaz,
                // birincil öğenin dönüşümleri birincil görüntü ile birlikte alfa düzlemine de uygulanır.
                let alpha_has_transforms = container.item_properties(*aid)?.any(|(p, _)| {
                    matches!(
                        p,
                        ItemProperty::Clap(_) | ItemProperty::Irot(_) | ItemProperty::Imir(_)
                    )
                });
                if !alpha_has_transforms {
                    apply_item_transforms(container, primary_id, &mut ab, &butce)?;
                }
                ab
            }
            AlphaSource::GridTiles {
                grid_item_id,
                alpha_tile_ids,
            } => {
                let (grid_bytes, grid_guard) =
                    container.resolve_item_data(*grid_item_id, &butce)?;
                let grid = parse_grid_descriptor(&grid_bytes)?;
                drop((grid_bytes, grid_guard));
                let mut ab = decode_grid_tiles_into_canvas(
                    container,
                    grid,
                    alpha_tile_ids,
                    config,
                    true,
                    0,
                    &butce,
                )?;
                apply_item_transforms(container, *grid_item_id, &mut ab, &butce)?;
                ab
            }
        };
        if alpha_buf.width != color_buf.width || alpha_buf.height != color_buf.height {
            return Err(Error::InvalidData(
                "alfa düzlemi boyutları birincil görüntü boyutlarıyla eşleşmiyor",
            ));
        }
        if alpha_buf.bit_depth != color_buf.bit_depth {
            return Err(Error::InvalidData(
                "alfa düzlemi bit derinliği birincil görüntü bit derinliğiyle eşleşmiyor",
            ));
        }
        img_guard.birlestir(alpha_buf.budget_guard);
        Some(alpha_buf.y_plane)
    } else {
        None
    };
    drop(alpha_src_opt);

    let (exif, xmp) = if let Some((ref mut ci, _)) = cached_info {
        (ci.exif.take(), ci.xmp.take())
    } else {
        let (ex, xm, meta_guard) = extract_metadata_items(container, primary_id, &butce)?;
        img_guard.birlestir(meta_guard);
        (ex, xm)
    };
    let effective_threads = resolve_effective_threads(config.n_threads);

    // ISO 21496-1 `tmap` üst veri ayrıştırması ve normatif doğrulaması her zaman yürütülür;
    // kazanç haritası pikselleri yalnızca `config.decode_gain_map == true` olduğunda çözülür.
    let (gain_map_metadata, gain_map) = if !config.decode_gain_map
        && let Some((ref mut ci, _)) = cached_info
    {
        (ci.gain_map_metadata.take(), None)
    } else {
        if let Some((ref mut ci, ref mut info_carry)) = cached_info
            && let Some(dropped_gm) = ci.gain_map_metadata.take()
            && let Some(ref alt_icc) = dropped_gm.alt_icc_profile
        {
            info_carry.dusur(alt_icc.len());
        }
        let (base_raw_w, base_raw_h) =
            match inspect_item_dimensions(container, primary_id, 0, &butce, config) {
                Ok(ins) => (ins.raw_width, ins.raw_height),
                Err(err @ (Error::BudgetExceeded { .. } | Error::AllocationFailed { .. })) => {
                    return Err(err);
                }
                Err(_) => (color_buf.width, color_buf.height),
            };
        let tmap_opt =
            find_and_validate_tmap(container, primary_id, base_raw_w, base_raw_h, &butce)?;

        if let Some((gm_item_id, gm_meta, tmap_guard)) = tmap_opt {
            img_guard.birlestir(tmap_guard);
            let decoded_gm = if config.decode_gain_map {
                validate_gain_map_transform_parity(container, primary_id, gm_item_id)?;
                let gm_buf = decode_item_planar(
                    container,
                    gm_item_id,
                    config,
                    config.n_threads,
                    false,
                    0,
                    &butce,
                )?;
                let gm_props = container.item_properties(gm_item_id)?;
                let mut gm_color = gm_buf.seq_color.unwrap_or(ColorInfo {
                    color_primaries: 2,
                    transfer_characteristics: 2,
                    matrix_coefficients: 2,
                    full_range: true,
                });
                let mut gm_pasp = None;
                for (p, _) in gm_props {
                    match p {
                        ItemProperty::ColrNclx(c) => gm_color = *c,
                        ItemProperty::Pasp(pa) => gm_pasp = Some(*pa),
                        _ => {}
                    }
                }
                // İkinci `alt_icc_profile` kopyası ve `Box<AvifImage>` tahsisini klonlamadan
                // önce bütçeye kaydet (`gm_image._butce_tutucu` her ikisini de taşır).
                let mut gm_guard = gm_buf.budget_guard;
                let extra_gm_bytes = gm_meta
                    .alt_icc_profile
                    .as_ref()
                    .map_or(0, Vec::len)
                    .saturating_add(core::mem::size_of::<AvifImage>());
                gm_guard.ekle(extra_gm_bytes)?;
                let gm_meta_copy = gm_meta.clone();
                let gm_image = AvifImage {
                    width: gm_buf.width,
                    height: gm_buf.height,
                    chroma_width: gm_buf.chroma_width,
                    chroma_height: gm_buf.chroma_height,
                    bit_depth: gm_buf.bit_depth,
                    chroma_format: gm_buf.chroma_format,
                    y_plane: gm_buf.y_plane,
                    u_plane: gm_buf.u_plane,
                    v_plane: gm_buf.v_plane,
                    alpha_plane: None,
                    premultiplied_alpha: false,
                    color: gm_color,
                    icc_profile: None,
                    exif: None,
                    xmp: None,
                    clli: None,
                    mdcv: None,
                    pasp: gm_pasp,
                    gain_map_metadata: None,
                    gain_map: None,
                    effective_threads,
                    butce: Some(Arc::clone(&butce)),
                    _butce_tutucu: Some(Arc::new(gm_guard)),
                };
                Some(GainMap {
                    metadata: gm_meta_copy,
                    image: Box::new(gm_image),
                })
            } else {
                None
            };
            (Some(gm_meta), decoded_gm)
        } else {
            (None, None)
        }
    };

    if let Some((_, info_carry)) = cached_info {
        img_guard.birlestir(info_carry);
    }

    Ok(AvifImage {
        width: color_buf.width,
        height: color_buf.height,
        chroma_width: color_buf.chroma_width,
        chroma_height: color_buf.chroma_height,
        bit_depth: color_buf.bit_depth,
        chroma_format: color_buf.chroma_format,
        y_plane: color_buf.y_plane,
        u_plane: color_buf.u_plane,
        v_plane: color_buf.v_plane,
        alpha_plane,
        premultiplied_alpha,
        color,
        icc_profile,
        exif,
        xmp,
        clli,
        mdcv,
        pasp,
        gain_map_metadata,
        gain_map,
        effective_threads,
        butce: Some(butce),
        _butce_tutucu: Some(Arc::new(img_guard)),
    })
}

/// Bir `.avif` bayt dilimini çözüp doğrudan 8-bit RGB ([`RgbImage`]) tamponuna dönüştürür.
pub fn decode_rgb8(data: &[u8]) -> Result<RgbImage> {
    let img = decode(data)?;
    img.to_rgb8()
}

/// Bir `.avif` bayt dilimini çözüp doğrudan 8-bit RGBA ([`RgbaImage`]) tamponuna dönüştürür.
pub fn decode_rgba8(data: &[u8]) -> Result<RgbaImage> {
    let img = decode(data)?;
    img.to_rgba8()
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;
    use alloc::vec;

    use super::{LIMITED_TO_FULL_Y_8BIT, PlanarBuffer, crop_planar, limited_to_full_y_u16};
    use crate::avif::image::PlaneStorage;
    use crate::error::Error;
    use crate::frame::butce::{Butce, ButceTutucu};
    use crate::frame::{BitDepth, ChromaFormat};

    #[test]
    fn limited_to_full_y_8bit_lut_matches_formula() {
        for v in 0u16..=255 {
            assert_eq!(
                u16::from(LIMITED_TO_FULL_Y_8BIT[usize::from(v)]),
                limited_to_full_y_u16(BitDepth::Eight, v)
            );
        }
    }

    #[test]
    fn allocation_limit_accounts_for_1_byte_8bit_and_2_byte_10bit() {
        // 32x32 4:2:0 8-bit canlı = 1024 + 256 + 256 = 1536 bayt;
        // 20x20 kırpmada ilk adım (Y): 1536 + 400 = 1936 bayt tepe gerektirir (sonra eski Y düşer).
        let b8_low = Arc::new(Butce::yeni(Some(1536 + 399), 0));
        let mut buf8 = PlanarBuffer {
            width: 32,
            height: 32,
            chroma_width: 16,
            chroma_height: 16,
            bit_depth: BitDepth::Eight,
            chroma_format: ChromaFormat::C420,
            y_plane: PlaneStorage::U8(vec![128u8; 32 * 32]),
            u_plane: PlaneStorage::U8(vec![128u8; 16 * 16]),
            v_plane: PlaneStorage::U8(vec![128u8; 16 * 16]),
            seq_color: None,
            budget_guard: Butce::tut(&b8_low, 1536).unwrap(),
        };
        assert_eq!(
            crop_planar(&mut buf8, 0, 0, 20, 20, &b8_low),
            Err(Error::BudgetExceeded {
                requested: 1936,
                limit: 1935,
            })
        );
        let b8_ok = Arc::new(Butce::yeni(Some(1536 + 400), 0));
        buf8.budget_guard = Butce::tut(&b8_ok, 1536).unwrap();
        assert_eq!(crop_planar(&mut buf8, 0, 0, 20, 20, &b8_ok), Ok(()));
        assert_eq!(buf8.y_plane.as_u8().map(<[u8]>::len), Some(400));
        assert_eq!(buf8.budget_guard.bayt(), 400 + 100 + 100);
        drop(buf8);
        assert_eq!(b8_ok.canli(), 0);

        // 32x32 4:2:0 10-bit canlı = 2 * 1536 = 3072 bayt;
        // 20x20 kırpmada ilk adım (Y): 3072 + 800 = 3872 bayt tepe gerektirir.
        let b10_low = Arc::new(Butce::yeni(Some(3072 + 799), 0));
        let mut buf10 = PlanarBuffer {
            width: 32,
            height: 32,
            chroma_width: 16,
            chroma_height: 16,
            bit_depth: BitDepth::Ten,
            chroma_format: ChromaFormat::C420,
            y_plane: PlaneStorage::U16(vec![512u16; 32 * 32]),
            u_plane: PlaneStorage::U16(vec![512u16; 16 * 16]),
            v_plane: PlaneStorage::U16(vec![512u16; 16 * 16]),
            seq_color: None,
            budget_guard: Butce::tut(&b10_low, 3072).unwrap(),
        };
        assert_eq!(
            crop_planar(&mut buf10, 0, 0, 20, 20, &b10_low),
            Err(Error::BudgetExceeded {
                requested: 3872,
                limit: 3871,
            })
        );
        let b10_ok = Arc::new(Butce::yeni(Some(3072 + 800), 0));
        buf10.budget_guard = Butce::tut(&b10_ok, 3072).unwrap();
        assert_eq!(crop_planar(&mut buf10, 0, 0, 20, 20, &b10_ok), Ok(()));
        assert_eq!(buf10.y_plane.as_u16().map(<[u16]>::len), Some(400));
        assert_eq!(buf10.budget_guard.bayt(), 800 + 200 + 200);
        drop(buf10);
        assert_eq!(b10_ok.canli(), 0);
        let _ = ButceTutucu::bos(&b10_ok);
    }

    #[test]
    fn compute_clap_rect_extreme_values_do_not_overflow() {
        use super::compute_clap_rect;
        use crate::avif::image::CleanAperture;

        let extreme_max = CleanAperture {
            width_n: 1,
            width_d: 1,
            height_n: 1,
            height_d: 1,
            horiz_off_n: i32::MAX,
            horiz_off_d: u32::MAX,
            vert_off_n: i32::MIN,
            vert_off_d: u32::MAX,
        };
        assert!(matches!(
            compute_clap_rect(u32::MAX, u32::MAX, &extreme_max),
            Err(Error::InvalidData(_))
        ));

        let exact_center = CleanAperture {
            width_n: 1,
            width_d: 1,
            height_n: 1,
            height_d: 1,
            horiz_off_n: 0,
            horiz_off_d: u32::MAX,
            vert_off_n: 0,
            vert_off_d: u32::MAX,
        };
        let rect = compute_clap_rect(u32::MAX, u32::MAX, &exact_center)
            .expect("u32::MAX × u32::MAX taşmadan hesaplanmalı");
        assert_eq!(rect, ((u32::MAX - 1) / 2, (u32::MAX - 1) / 2, 1, 1));
    }
}
