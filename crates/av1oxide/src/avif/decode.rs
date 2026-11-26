//! AVIF görüntü çözme, `grid` karo birleştirme, alfa kanalı ve `clap`/`irot`/`imir` dönüşüm boru hattı.

#![forbid(unsafe_code)]

use crate::api::{Config, Decoder, Packet};
use crate::avif::boxes::{
    ALPHA_AUX_URN_CICP, ALPHA_AUX_URN_HEVC, Av1CProperty, AvifContainer, ItemProperty,
    parse_container,
};
use crate::avif::gainmap::{
    GainMap, GainMapMetadata, ToneMapParseOutcome, parse_tone_mapped_image_box,
};
use crate::avif::image::{
    AvifDecodeConfig, AvifImage, AvifInfo, CleanAperture, ColorInfo, PlaneStorage, RgbImage,
    RgbaImage,
};
use crate::bitstream::obu::parse_sequence_header_obus;
use crate::bitstream::seq_header::PixelLayout;
use crate::error::{Error, Result, SizeStage};
use crate::frame::{BitDepth, ChromaFormat, Frame, PlaneType};

/// İç düzlemsel çalışma tamponu.
#[derive(Clone, Debug)]
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
}

/// Yapılandırılmış iş parçacığı sayısını etkin çalışma zamanı sayısına çözümler.
pub(crate) fn resolve_effective_threads(n_threads: usize) -> usize {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = n_threads;
        1
    }
    #[cfg(not(target_arch = "wasm32"))]
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
    let hd = i64::from(clap.horiz_off_d);
    let vd = i64::from(clap.vert_off_d);
    let num_x = (i64::from(w) - i64::from(crop_w)) * hd + 2 * i64::from(clap.horiz_off_n);
    let den_x = 2 * hd;
    let num_y = (i64::from(h) - i64::from(crop_h)) * vd + 2 * i64::from(clap.vert_off_n);
    let den_y = 2 * vd;

    if (num_x % den_x) != 0 || (num_y % den_y) != 0 {
        return Err(Error::InvalidData(
            "clap sol-üst kırpma ofseti tam sayı değil",
        ));
    }
    let x0_i64 = num_x / den_x;
    let y0_i64 = num_y / den_y;
    if x0_i64 < 0
        || y0_i64 < 0
        || x0_i64 + i64::from(crop_w) > i64::from(w)
        || y0_i64 + i64::from(crop_h) > i64::from(h)
    {
        return Err(Error::InvalidData(
            "clap kırpma bölgesi görüntü sınırları dışında",
        ));
    }
    Ok((x0_i64 as u32, y0_i64 as u32, crop_w, crop_h))
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

    for (p, _) in &props {
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
            if (ispe_dims.is_none() || bit_depth.is_none() || chroma_format.is_none())
                && let Ok(payload) = container.resolve_item_data(item_id, None)
                && let Ok(Some(seq)) = parse_sequence_header_obus(&payload)
            {
                if ispe_dims.is_none() {
                    ispe_dims = Some((seq.max_width.max(1) as u32, seq.max_height.max(1) as u32));
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
            let (iw, ih) =
                ispe_dims.ok_or(Error::InvalidData("av01 öğesinde boyut bilgisi eksik"))?;
            let bd = bit_depth.unwrap_or(BitDepth::Eight);
            let cf = chroma_format.unwrap_or(ChromaFormat::C420);
            (iw, ih, bd, cf)
        }
        b"grid" => {
            let grid_bytes = container.resolve_item_data(item_id, None)?;
            let grid = parse_grid_descriptor(&grid_bytes)?;
            let tiles = container.references_from(item_id, b"dimg");
            let expected = (grid.rows as usize)
                .checked_mul(grid.cols as usize)
                .ok_or(Error::InvalidData("grid karo sayısı taşması"))?;
            if tiles.len() != expected || tiles.is_empty() {
                return Err(Error::InvalidData(
                    "grid dimg referans sayısı rows*cols ile uyuşmuyor",
                ));
            }
            let tile_info = inspect_item_dimensions(container, tiles[0], depth + 1)?;
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
            let targets = container.references_from(item_id, b"dimg");
            if targets.len() != 1 {
                return Err(Error::InvalidData(
                    "iden öğesi tek bir dimg referansı içermelidir",
                ));
            }
            let target_info = inspect_item_dimensions(container, targets[0], depth + 1)?;
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
) -> Result<Option<AlphaSource>> {
    if let Some(aid) = find_direct_alpha_item(container, primary_id)? {
        return Ok(Some(AlphaSource::Direct(aid)));
    }
    let Some(info) = container.item_info(primary_id) else {
        return Ok(None);
    };
    if &info.item_type == b"grid" {
        let tile_ids = container.references_from(primary_id, b"dimg");
        if tile_ids.is_empty() {
            return Ok(None);
        }
        let mut alpha_tiles: Vec<u32> = Vec::with_capacity(tile_ids.len());
        for &tid in &tile_ids {
            if let Some(aid) = find_direct_alpha_item(container, tid)? {
                alpha_tiles.push(aid);
            } else {
                return Ok(None);
            }
        }
        return Ok(Some(AlphaSource::GridTiles {
            grid_item_id: primary_id,
            alpha_tile_ids: alpha_tiles,
        }));
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
    let from_primary = container.references_from(primary_id, b"prem");
    if from_primary.contains(&alpha_id) {
        return true;
    }
    let to_primary = container.references_to(primary_id, b"prem");
    to_primary.contains(&alpha_id)
}

pub(crate) type MetadataPair = (Option<Vec<u8>>, Option<Vec<u8>>);

pub(crate) fn extract_metadata_items(
    container: &AvifContainer<'_>,
    primary_id: u32,
    allocation_limit: Option<usize>,
) -> Result<MetadataPair> {
    let mut exif: Option<Vec<u8>> = None;
    let mut xmp: Option<Vec<u8>> = None;

    let meta_ids = container.references_to(primary_id, b"cdsc");
    for mid in meta_ids {
        let Some(info) = container.item_info(mid) else {
            continue;
        };
        if &info.item_type == b"Exif" && exif.is_none() {
            let raw = container.resolve_item_data(mid, allocation_limit)?;
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
            exif = Some(slice.to_vec());
        } else if &info.item_type == b"mime"
            && info.content_type == "application/rdf+xml"
            && xmp.is_none()
        {
            let raw = container.resolve_item_data(mid, allocation_limit)?;
            xmp = Some(raw.into_owned());
        }
    }
    Ok((exif, xmp))
}

/// `trak` içine gömülü `meta` kutusundan tüm ilişkili `Exif`/`XMP` öğelerini çıkarır
/// (`libavif` `avifDecoderFindMetadata(..., colorId = 0)` uyumlu).
pub(crate) fn extract_track_metadata_items(
    container: &AvifContainer<'_>,
    allocation_limit: Option<usize>,
) -> Result<MetadataPair> {
    let mut exif: Option<Vec<u8>> = None;
    let mut xmp: Option<Vec<u8>> = None;

    for info in container.items.values() {
        if !container.item_has_data(info.item_id)
            || container.has_unsupported_essential_property(info.item_id)?
        {
            continue;
        }
        if &info.item_type == b"Exif" && exif.is_none() {
            let raw = container.resolve_item_data(info.item_id, allocation_limit)?;
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
            exif = Some(slice.to_vec());
        } else if &info.item_type == b"mime"
            && info.content_type == "application/rdf+xml"
            && xmp.is_none()
        {
            let raw = container.resolve_item_data(info.item_id, allocation_limit)?;
            xmp = Some(raw.into_owned());
        }
    }
    Ok((exif, xmp))
}

/// ISO 21496-1 `tmap` öğesini `libavif` (`avifDecoderFindGainMapItem`) kurallarına göre bulur,
/// ikili `ToneMapImage` yükünü ve `tmap` özelliklerini doğrular.
fn find_and_validate_tmap(
    container: &AvifContainer<'_>,
    primary_id: u32,
    base_raw_w: u32,
    base_raw_h: u32,
    allocation_limit: Option<usize>,
) -> Result<Option<(u32, GainMapMetadata)>> {
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
            || !container.references_from(item.item_id, b"thmb").is_empty()
        {
            continue;
        }
        let dimg_ids = container.references_from(item.item_id, b"dimg");
        if dimg_ids.len() != 2
            || dimg_ids[0] == dimg_ids[1]
            || container.item_info(dimg_ids[0]).is_none()
            || container.item_info(dimg_ids[1]).is_none()
        {
            return Err(Error::InvalidData(
                "tmap dimg referansları tam olarak 2 farklı geçerli öğe içermelidir",
            ));
        }
        if dimg_ids[0] != primary_id {
            continue;
        }
        found = Some((item.item_id, dimg_ids[1]));
        break;
    }

    let Some((tmap_id, gain_map_item_id)) = found else {
        return Ok(None);
    };

    // `grpl` / `altr` varlık grubunda `tmap` öğesi birincil öğeden önce tercih edilmelidir.
    if !container.is_preferred_alternative(tmap_id, primary_id) {
        return Ok(None);
    }

    let tmap_data = container.resolve_item_data(tmap_id, allocation_limit)?;
    let mut metadata = match parse_tone_mapped_image_box(&tmap_data)? {
        ToneMapParseOutcome::Parsed(m) => *m,
        ToneMapParseOutcome::UnsupportedVersion => return Ok(None),
    };

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
    for (p, _) in &tmap_props {
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
                metadata.alt_icc_profile = Some(icc.clone());
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
    for (p, _) in &gm_props {
        if matches!(p, ItemProperty::ColrNclx(_)) {
            if seen_gm_nclx {
                return Err(Error::InvalidData(
                    "kazanç haritası öğesinde birden fazla colr nclx özelliği",
                ));
            }
            seen_gm_nclx = true;
        }
    }

    Ok(Some((gain_map_item_id, metadata)))
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
    let container = parse_container(data)?;
    let primary_id = container.primary_item_id;
    let inspected = inspect_item_dimensions(&container, primary_id, 0)?;

    let props = container.item_properties(primary_id)?;
    let mut color = inspected.seq_color.unwrap_or_default();
    let mut icc_profile = None;
    let mut clli = None;
    let mut mdcv = None;
    let mut pasp = None;

    for (p, _) in props {
        match p {
            ItemProperty::ColrNclx(c) => color = *c,
            ItemProperty::ColrIcc(icc) => icc_profile = Some(icc.clone()),
            ItemProperty::Clli(c) => clli = Some(*c),
            ItemProperty::Mdcv(m) => mdcv = Some(*m),
            ItemProperty::Pasp(pa) => pasp = Some(*pa),
            _ => {}
        }
    }

    let alpha_src = find_alpha_source(&container, primary_id)?;
    let premultiplied_alpha = is_premultiplied_alpha(&container, primary_id, alpha_src.as_ref());
    let (exif, xmp) = extract_metadata_items(&container, primary_id, None)?;
    let gain_map_metadata = find_and_validate_tmap(
        &container,
        primary_id,
        inspected.raw_width,
        inspected.raw_height,
        None,
    )?
    .map(|(_, meta)| meta);

    Ok(AvifInfo {
        width: inspected.width,
        height: inspected.height,
        bit_depth: inspected.bit_depth,
        chroma_format: inspected.chroma_format,
        has_alpha: alpha_src.is_some(),
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
    allocation_limit: Option<usize>,
) -> Result<(u32, u32, PlaneStorage)> {
    let bit_depth = frame.bit_depth();
    let Some(plane) = frame.plane(plane_type) else {
        return Ok((0, 0, PlaneStorage::empty_for(bit_depth)));
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
    if let Some(limit) = allocation_limit
        && bytes_req > limit
    {
        return Err(Error::BudgetExceeded {
            requested: bytes_req,
            limit,
        });
    }

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
            Ok((w as u32, h as u32, PlaneStorage::U8(out)))
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
            Ok((w as u32, h as u32, PlaneStorage::U16(out)))
        }
    }
}

fn decode_av01_item(
    container: &AvifContainer<'_>,
    item_id: u32,
    config: &AvifDecodeConfig,
    threads_for_item: usize,
    is_alpha: bool,
) -> Result<PlanarBuffer> {
    let props = container.item_properties(item_id)?;
    let mut av1c_opt: Option<&Av1CProperty> = None;
    let mut operating_point: u8 = 0;
    let mut layer_sel: Option<u16> = None;
    let mut has_ispe = false;

    for (p, _) in &props {
        match p {
            ItemProperty::Av1C(av1c) => av1c_opt = Some(av1c),
            ItemProperty::A1op(op) => operating_point = *op,
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

    let payload = container.resolve_item_data(item_id, config.allocation_limit)?;

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
    let mut decoder = Decoder::new(dec_config)?;

    if let Some(av1c) = av1c_opt
        && !av1c.config_obus.is_empty()
    {
        decoder.send_packet(&Packet::new(&av1c.config_obus))?;
    }
    decoder.send_packet(&Packet::new(&payload))?;
    decoder.flush();

    let mut chosen_frame: Option<Frame> = None;
    let mut idx = 0u16;
    while let Some(frame) = decoder.receive_frame()? {
        if let Some(target_layer) = layer_sel {
            if idx == target_layer {
                chosen_frame = Some(frame);
                break;
            }
            chosen_frame = Some(frame);
        } else {
            chosen_frame = Some(frame);
        }
        idx = idx.saturating_add(1);
    }

    let frame = chosen_frame.ok_or(Error::InvalidData("AV1 öğesinden görünür kare üretilemedi"))?;
    let seq_color = decoder.sequence_header().map(|seq| ColorInfo {
        color_primaries: u16::from(seq.pri),
        transfer_characteristics: u16::from(seq.trc),
        matrix_coefficients: u16::from(seq.mtrx),
        full_range: seq.color_range != 0,
    });

    let bit_depth = frame.bit_depth();
    let bps = PlaneStorage::bytes_per_sample(bit_depth);
    let y_samples = frame
        .plane(PlaneType::Y)
        .map_or(0usize, |p| p.width().saturating_mul(p.height()));
    let c_samples = if is_alpha {
        0usize
    } else {
        frame
            .plane(PlaneType::U)
            .map_or(0usize, |p| p.width().saturating_mul(p.height()))
            .saturating_add(
                frame
                    .plane(PlaneType::V)
                    .map_or(0usize, |p| p.width().saturating_mul(p.height())),
            )
    };
    let total_planar_bytes =
        y_samples
            .saturating_add(c_samples)
            .checked_mul(bps)
            .ok_or(Error::SizeOverflow {
                stage: SizeStage::TotalAllocation,
                lhs: y_samples.saturating_add(c_samples),
                rhs: bps,
            })?;
    if let Some(limit) = config.allocation_limit
        && total_planar_bytes > limit
    {
        return Err(Error::BudgetExceeded {
            requested: total_planar_bytes,
            limit,
        });
    }

    let (w, h, mut y_plane) = extract_plane(&frame, PlaneType::Y, config.allocation_limit)?;
    let (cw, ch, u_plane, v_plane, chroma_format) = if is_alpha {
        (
            0,
            0,
            PlaneStorage::empty_for(bit_depth),
            PlaneStorage::empty_for(bit_depth),
            ChromaFormat::Monochrome,
        )
    } else {
        let (cw, ch, u) = extract_plane(&frame, PlaneType::U, config.allocation_limit)?;
        let (_, _, v) = extract_plane(&frame, PlaneType::V, config.allocation_limit)?;
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
    })
}

fn crop_slice<T: Copy>(
    src: &[T],
    old_w: usize,
    x0: usize,
    y0: usize,
    new_w: usize,
    new_h: usize,
) -> Vec<T> {
    let mut dst = Vec::with_capacity(new_w * new_h);
    for r in 0..new_h {
        let src_start = (y0 + r) * old_w + x0;
        dst.extend_from_slice(&src[src_start..src_start + new_w]);
    }
    dst
}

fn crop_storage(
    storage: &PlaneStorage,
    old_w: usize,
    x0: usize,
    y0: usize,
    new_w: usize,
    new_h: usize,
) -> PlaneStorage {
    match storage {
        PlaneStorage::U8(src) => PlaneStorage::U8(crop_slice(src, old_w, x0, y0, new_w, new_h)),
        PlaneStorage::U16(src) => PlaneStorage::U16(crop_slice(src, old_w, x0, y0, new_w, new_h)),
    }
}

pub(crate) fn crop_planar(
    buf: &mut PlanarBuffer,
    x0: u32,
    y0: u32,
    crop_w: u32,
    crop_h: u32,
    allocation_limit: Option<usize>,
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
    let y_count = new_w * new_h;
    let bps = PlaneStorage::bytes_per_sample(buf.bit_depth);
    let y_bytes = y_count.checked_mul(bps).ok_or(Error::SizeOverflow {
        stage: SizeStage::TotalAllocation,
        lhs: y_count,
        rhs: bps,
    })?;
    if let Some(limit) = allocation_limit
        && y_bytes > limit
    {
        return Err(Error::BudgetExceeded {
            requested: y_bytes,
            limit,
        });
    }
    buf.y_plane = crop_storage(&buf.y_plane, old_w, x0 as usize, y0 as usize, new_w, new_h);
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
        buf.u_plane = crop_storage(&buf.u_plane, old_cw, cx0, cy0, new_cw, new_ch);
        buf.v_plane = crop_storage(&buf.v_plane, old_cw, cx0, cy0, new_cw, new_ch);
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
) -> (usize, usize, Vec<T>) {
    match angle & 3 {
        1 => {
            let mut dst = vec![T::default(); w * h];
            for ny in 0..w {
                let sx = w - 1 - ny;
                for nx in 0..h {
                    let sy = nx;
                    dst[ny * h + nx] = src[sy * w + sx];
                }
            }
            (h, w, dst)
        }
        2 => {
            let mut dst = vec![T::default(); w * h];
            for ny in 0..h {
                let sy = h - 1 - ny;
                for nx in 0..w {
                    let sx = w - 1 - nx;
                    dst[ny * w + nx] = src[sy * w + sx];
                }
            }
            (w, h, dst)
        }
        3 => {
            let mut dst = vec![T::default(); w * h];
            for ny in 0..w {
                let sx = ny;
                for nx in 0..h {
                    let sy = h - 1 - nx;
                    dst[ny * h + nx] = src[sy * w + sx];
                }
            }
            (h, w, dst)
        }
        _ => (w, h, src.to_vec()),
    }
}

fn rotate_plane_ccw(
    src: &PlaneStorage,
    w: usize,
    h: usize,
    angle: u8,
) -> (usize, usize, PlaneStorage) {
    match src {
        PlaneStorage::U8(s) => {
            let (nw, nh, dst) = rotate_slice_ccw(s, w, h, angle);
            (nw, nh, PlaneStorage::U8(dst))
        }
        PlaneStorage::U16(s) => {
            let (nw, nh, dst) = rotate_slice_ccw(s, w, h, angle);
            (nw, nh, PlaneStorage::U16(dst))
        }
    }
}

fn upsample_slice_c422_to_c444<T: Copy>(src: &[T], w: usize, h: usize, cw: usize) -> Vec<T> {
    let mut dst = Vec::with_capacity(w * h);
    let max_cx = cw.saturating_sub(1);
    for y in 0..h {
        let row_off = y * cw;
        for x in 0..w {
            let cx = (x / 2).min(max_cx);
            dst.push(src[row_off + cx]);
        }
    }
    dst
}

fn upsample_c422_to_c444(buf: &mut PlanarBuffer) {
    if buf.chroma_format != ChromaFormat::C422 {
        return;
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let cw = buf.chroma_width as usize;
    buf.u_plane = match &buf.u_plane {
        PlaneStorage::U8(s) => PlaneStorage::U8(upsample_slice_c422_to_c444(s, w, h, cw)),
        PlaneStorage::U16(s) => PlaneStorage::U16(upsample_slice_c422_to_c444(s, w, h, cw)),
    };
    buf.v_plane = match &buf.v_plane {
        PlaneStorage::U8(s) => PlaneStorage::U8(upsample_slice_c422_to_c444(s, w, h, cw)),
        PlaneStorage::U16(s) => PlaneStorage::U16(upsample_slice_c422_to_c444(s, w, h, cw)),
    };
    buf.chroma_width = buf.width;
    buf.chroma_height = buf.height;
    buf.chroma_format = ChromaFormat::C444;
}

pub(crate) fn apply_irot(buf: &mut PlanarBuffer, angle: u8) {
    let a = angle & 3;
    if a == 0 {
        return;
    }
    if (a & 1) != 0 && buf.chroma_format == ChromaFormat::C422 {
        upsample_c422_to_c444(buf);
    }
    let (nw, nh, ny) = rotate_plane_ccw(&buf.y_plane, buf.width as usize, buf.height as usize, a);
    buf.width = nw as u32;
    buf.height = nh as u32;
    buf.y_plane = ny;

    if buf.chroma_format != ChromaFormat::Monochrome {
        let (ncw, nch, nu) = rotate_plane_ccw(
            &buf.u_plane,
            buf.chroma_width as usize,
            buf.chroma_height as usize,
            a,
        );
        let (_, _, nv) = rotate_plane_ccw(
            &buf.v_plane,
            buf.chroma_width as usize,
            buf.chroma_height as usize,
            a,
        );
        buf.chroma_width = ncw as u32;
        buf.chroma_height = nch as u32;
        buf.u_plane = nu;
        buf.v_plane = nv;
    }
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
    allocation_limit: Option<usize>,
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
        crop_planar(buf, x0, y0, cw, ch, allocation_limit)?;
    }
    if let Some(angle) = irot_opt {
        apply_irot(buf, angle);
    }
    if let Some(axis) = imir_opt {
        apply_imir(buf, axis);
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

    let has_dups = is_alpha && (1..tile_ids.len()).any(|i| tile_ids[..i].contains(&tile_ids[i]));
    let mut unique_ids: Vec<u32> = Vec::with_capacity(num_tiles);
    let mut tile_to_unique: Vec<usize> = Vec::with_capacity(num_tiles);
    if has_dups {
        let mut unique_set = crate::avif::boxes::FxIndexSet::with_capacity_and_hasher(
            num_tiles,
            rustc_hash::FxBuildHasher,
        );
        for &tid in tile_ids {
            let (u_idx, inserted) = unique_set.insert_full(tid);
            if inserted {
                unique_ids.push(tid);
            }
            tile_to_unique.push(u_idx);
        }
    } else {
        unique_ids.extend_from_slice(tile_ids);
        for i in 0..num_tiles {
            tile_to_unique.push(i);
        }
    }
    let num_unique = unique_ids.len();

    let total_threads = resolve_effective_threads(config.n_threads);

    let decoded_tiles: Vec<PlanarBuffer> = if total_threads > 1 && num_unique > 1 {
        let worker_count = total_threads.min(num_unique);
        let chunk_size = num_unique.div_ceil(worker_count);
        let mut results: Vec<Result<Vec<PlanarBuffer>>> = Vec::with_capacity(worker_count);
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(worker_count);
            for chunk in unique_ids.chunks(chunk_size) {
                handles.push(s.spawn(move || {
                    let mut out = Vec::with_capacity(chunk.len());
                    for &tid in chunk {
                        out.push(decode_item_planar(
                            container,
                            tid,
                            config,
                            1,
                            is_alpha,
                            depth + 1,
                        )?);
                    }
                    Ok(out)
                }));
            }
            for h in handles {
                let res = h
                    .join()
                    .map_err(|_| Error::InvalidState("grid karo iş parçacığı sonlandı"))?;
                results.push(res);
            }
            Ok::<(), Error>(())
        })?;
        let mut flat = Vec::with_capacity(num_unique);
        for group in results {
            flat.extend(group?);
        }
        flat
    } else {
        let mut flat = Vec::with_capacity(num_unique);
        for &tid in &unique_ids {
            flat.push(decode_item_planar(
                container,
                tid,
                config,
                total_threads,
                is_alpha,
                depth + 1,
            )?);
        }
        flat
    };

    let first = &decoded_tiles[0];
    let tile_w = first.width as usize;
    let tile_h = first.height as usize;
    let tile_cw = first.chroma_width as usize;
    let tile_ch = first.chroma_height as usize;
    let bit_depth = first.bit_depth;
    let chroma_format = first.chroma_format;
    let seq_color = first.seq_color;

    if tile_w == 0 || tile_h == 0 {
        return Err(Error::InvalidData("grid karo boyutu sıfır olamaz"));
    }
    for t in &decoded_tiles[1..] {
        if t.width as usize != tile_w
            || t.height as usize != tile_h
            || t.bit_depth != bit_depth
            || (!is_alpha && t.chroma_format != chroma_format)
        {
            return Err(Error::InvalidData(
                "grid karolarının boyut veya biçimleri uyumsuz",
            ));
        }
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
    check_pixel_and_byte_budget(grid.output_width, grid.output_height, total_bytes, config)?;

    let copy_chroma = !is_alpha && chroma_format != ChromaFormat::Monochrome;
    let (y_canvas, u_canvas, v_canvas) = match bit_depth {
        BitDepth::Eight => {
            let (yc, uc, vc) = stitch_grid_canvases::<u8>(
                &decoded_tiles,
                &tile_to_unique,
                rows,
                cols,
                tile_w,
                tile_h,
                tile_cw,
                tile_ch,
                out_w,
                out_h,
                out_cw,
                out_ch,
                y_samples,
                c_samples,
                copy_chroma,
                |t| t.y_plane.as_u8().unwrap_or(&[]),
                |t| t.u_plane.as_u8().unwrap_or(&[]),
                |t| t.v_plane.as_u8().unwrap_or(&[]),
            );
            (
                PlaneStorage::U8(yc),
                PlaneStorage::U8(uc),
                PlaneStorage::U8(vc),
            )
        }
        BitDepth::Ten | BitDepth::Twelve => {
            let (yc, uc, vc) = stitch_grid_canvases::<u16>(
                &decoded_tiles,
                &tile_to_unique,
                rows,
                cols,
                tile_w,
                tile_h,
                tile_cw,
                tile_ch,
                out_w,
                out_h,
                out_cw,
                out_ch,
                y_samples,
                c_samples,
                copy_chroma,
                |t| t.y_plane.as_u16().unwrap_or(&[]),
                |t| t.u_plane.as_u16().unwrap_or(&[]),
                |t| t.v_plane.as_u16().unwrap_or(&[]),
            );
            (
                PlaneStorage::U16(yc),
                PlaneStorage::U16(uc),
                PlaneStorage::U16(vc),
            )
        }
    };

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
    })
}

#[allow(clippy::too_many_arguments)]
fn stitch_grid_canvases<T: Copy + Default>(
    decoded_tiles: &[PlanarBuffer],
    tile_to_unique: &[usize],
    rows: usize,
    cols: usize,
    tile_w: usize,
    tile_h: usize,
    tile_cw: usize,
    tile_ch: usize,
    out_w: usize,
    out_h: usize,
    out_cw: usize,
    out_ch: usize,
    y_samples: usize,
    c_samples: usize,
    copy_chroma: bool,
    get_y: impl Fn(&PlanarBuffer) -> &[T],
    get_u: impl Fn(&PlanarBuffer) -> &[T],
    get_v: impl Fn(&PlanarBuffer) -> &[T],
) -> (Vec<T>, Vec<T>, Vec<T>) {
    let mut y_canvas = vec![T::default(); y_samples];
    let mut u_canvas = vec![T::default(); c_samples];
    let mut v_canvas = vec![T::default(); c_samples];

    for r in 0..rows {
        let dst_y0 = r * tile_h;
        if dst_y0 >= out_h {
            continue;
        }
        let copy_h = tile_h.min(out_h - dst_y0);

        let dst_cy0 = r * tile_ch;
        let copy_ch = if out_ch > dst_cy0 {
            tile_ch.min(out_ch - dst_cy0)
        } else {
            0
        };

        for c in 0..cols {
            let tile = &decoded_tiles[tile_to_unique[r * cols + c]];
            let dst_x0 = c * tile_w;
            if dst_x0 >= out_w {
                continue;
            }
            let copy_w = tile_w.min(out_w - dst_x0);

            let ty_src = get_y(tile);
            for ty in 0..copy_h {
                let src_off = ty * tile_w;
                let dst_off = (dst_y0 + ty) * out_w + dst_x0;
                y_canvas[dst_off..dst_off + copy_w]
                    .copy_from_slice(&ty_src[src_off..src_off + copy_w]);
            }

            if copy_chroma && copy_ch > 0 {
                let dst_cx0 = c * tile_cw;
                if dst_cx0 < out_cw {
                    let copy_cw = tile_cw.min(out_cw - dst_cx0);
                    let tu_src = get_u(tile);
                    let tv_src = get_v(tile);
                    for tcy in 0..copy_ch {
                        let src_off = tcy * tile_cw;
                        let dst_off = (dst_cy0 + tcy) * out_cw + dst_cx0;
                        u_canvas[dst_off..dst_off + copy_cw]
                            .copy_from_slice(&tu_src[src_off..src_off + copy_cw]);
                        v_canvas[dst_off..dst_off + copy_cw]
                            .copy_from_slice(&tv_src[src_off..src_off + copy_cw]);
                    }
                }
            }
        }
    }

    (y_canvas, u_canvas, v_canvas)
}

fn decode_grid_item(
    container: &AvifContainer<'_>,
    item_id: u32,
    config: &AvifDecodeConfig,
    is_alpha: bool,
    depth: usize,
) -> Result<PlanarBuffer> {
    let grid_bytes = container.resolve_item_data(item_id, config.allocation_limit)?;
    let grid = parse_grid_descriptor(&grid_bytes)?;
    let tile_ids = container.references_from(item_id, b"dimg");
    decode_grid_tiles_into_canvas(container, grid, &tile_ids, config, is_alpha, depth)
}

fn decode_item_planar(
    container: &AvifContainer<'_>,
    item_id: u32,
    config: &AvifDecodeConfig,
    threads_for_item: usize,
    is_alpha: bool,
    depth: usize,
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
        b"av01" => decode_av01_item(container, item_id, config, threads_for_item, is_alpha)?,
        b"grid" => decode_grid_item(container, item_id, config, is_alpha, depth)?,
        b"iden" => {
            let targets = container.references_from(item_id, b"dimg");
            if targets.len() != 1 {
                return Err(Error::InvalidData(
                    "iden öğesi tek bir dimg referansı içermelidir",
                ));
            }
            decode_item_planar(
                container,
                targets[0],
                config,
                threads_for_item,
                is_alpha,
                depth + 1,
            )?
        }
        _ => return Err(Error::Unsupported("desteklenmeyen AVIF görüntü öğesi türü")),
    };

    let props = container.item_properties(item_id)?;
    for (p, _) in &props {
        if let ItemProperty::Ispe { width, height } = p
            && (*width < buf.width || *height < buf.height)
        {
            let target_w = (*width).min(buf.width);
            let target_h = (*height).min(buf.height);
            crop_planar(&mut buf, 0, 0, target_w, target_h, config.allocation_limit)?;
        }
    }

    apply_item_transforms(container, item_id, &mut buf, config.allocation_limit)?;
    Ok(buf)
}

/// Varsayılan yapılandırma ([`AvifDecodeConfig::default`]) ile bir `.avif` bayt dilimini çözer.
pub fn decode(data: &[u8]) -> Result<AvifImage> {
    decode_with(data, &AvifDecodeConfig::default())
}

/// Belirtilen [`AvifDecodeConfig`] yapılandırması ile bir `.avif` bayt dilimini çözer.
pub fn decode_with(data: &[u8], config: &AvifDecodeConfig) -> Result<AvifImage> {
    let container = parse_container(data)?;
    let primary_id = container.primary_item_id;

    let color_buf = decode_item_planar(&container, primary_id, config, config.n_threads, false, 0)?;

    let props = container.item_properties(primary_id)?;
    let mut color = color_buf.seq_color.unwrap_or_default();
    let mut icc_profile = None;
    let mut clli = None;
    let mut mdcv = None;
    let mut pasp = None;

    for (p, _) in props {
        match p {
            ItemProperty::ColrNclx(c) => color = *c,
            ItemProperty::ColrIcc(icc) => icc_profile = Some(icc.clone()),
            ItemProperty::Clli(c) => clli = Some(*c),
            ItemProperty::Mdcv(m) => mdcv = Some(*m),
            ItemProperty::Pasp(pa) => pasp = Some(*pa),
            _ => {}
        }
    }

    let alpha_src = find_alpha_source(&container, primary_id)?;
    let premultiplied_alpha = is_premultiplied_alpha(&container, primary_id, alpha_src.as_ref());

    let alpha_plane = if let Some(ref src) = alpha_src {
        let alpha_buf = match src {
            AlphaSource::Direct(aid) => {
                let mut ab =
                    decode_item_planar(&container, *aid, config, config.n_threads, true, 0)?;
                // libavif uyumu: alfa öğesinin kendi irot/imir/clap dönüşümleri dikkate alınmaz,
                // birincil öğenin dönüşümleri birincil görüntü ile birlikte alfa düzlemine de uygulanır.
                let alpha_has_transforms = container.item_properties(*aid)?.iter().any(|(p, _)| {
                    matches!(
                        p,
                        ItemProperty::Clap(_) | ItemProperty::Irot(_) | ItemProperty::Imir(_)
                    )
                });
                if !alpha_has_transforms {
                    apply_item_transforms(
                        &container,
                        primary_id,
                        &mut ab,
                        config.allocation_limit,
                    )?;
                }
                ab
            }
            AlphaSource::GridTiles {
                grid_item_id,
                alpha_tile_ids,
            } => {
                let grid_bytes =
                    container.resolve_item_data(*grid_item_id, config.allocation_limit)?;
                let grid = parse_grid_descriptor(&grid_bytes)?;
                let mut ab = decode_grid_tiles_into_canvas(
                    &container,
                    grid,
                    alpha_tile_ids,
                    config,
                    true,
                    0,
                )?;
                apply_item_transforms(&container, *grid_item_id, &mut ab, config.allocation_limit)?;
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
        Some(alpha_buf.y_plane)
    } else {
        None
    };

    let (exif, xmp) = extract_metadata_items(&container, primary_id, config.allocation_limit)?;
    let effective_threads = resolve_effective_threads(config.n_threads);

    // ISO 21496-1 `tmap` üst veri ayrıştırması ve normatif doğrulaması her zaman yürütülür;
    // kazanç haritası pikselleri yalnızca `config.decode_gain_map == true` olduğunda çözülür.
    let (base_raw_w, base_raw_h) = match inspect_item_dimensions(&container, primary_id, 0) {
        Ok(ins) => (ins.raw_width, ins.raw_height),
        Err(_) => (color_buf.width, color_buf.height),
    };
    let tmap_opt = find_and_validate_tmap(
        &container,
        primary_id,
        base_raw_w,
        base_raw_h,
        config.allocation_limit,
    )?;

    let (gain_map_metadata, gain_map) = if let Some((gm_item_id, gm_meta)) = tmap_opt {
        let decoded_gm = if config.decode_gain_map {
            validate_gain_map_transform_parity(&container, primary_id, gm_item_id)?;
            let gm_buf =
                decode_item_planar(&container, gm_item_id, config, config.n_threads, false, 0)?;
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
            };
            Some(GainMap {
                metadata: gm_meta.clone(),
                image: Box::new(gm_image),
            })
        } else {
            None
        };
        (Some(gm_meta), decoded_gm)
    } else {
        (None, None)
    };

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
    use super::{LIMITED_TO_FULL_Y_8BIT, PlanarBuffer, crop_planar, limited_to_full_y_u16};
    use crate::avif::image::PlaneStorage;
    use crate::error::Error;
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
        };
        // 20 x 20 kırpma: 8-bit için 400 bayt (399 baytta reddedilmeli, 400 baytta geçmeli).
        assert_eq!(
            crop_planar(&mut buf8, 0, 0, 20, 20, Some(399)),
            Err(Error::BudgetExceeded {
                requested: 400,
                limit: 399,
            })
        );
        assert_eq!(crop_planar(&mut buf8, 0, 0, 20, 20, Some(400)), Ok(()));
        assert_eq!(buf8.y_plane.as_u8().map(<[u8]>::len), Some(400));

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
        };
        // 20 x 20 kırpma: 10-bit için 800 bayt (799 baytta reddedilmeli, 800 baytta geçmeli).
        assert_eq!(
            crop_planar(&mut buf10, 0, 0, 20, 20, Some(799)),
            Err(Error::BudgetExceeded {
                requested: 800,
                limit: 799,
            })
        );
        assert_eq!(crop_planar(&mut buf10, 0, 0, 20, 20, Some(800)), Ok(()));
        assert_eq!(buf10.y_plane.as_u16().map(<[u16]>::len), Some(400));
    }
}
