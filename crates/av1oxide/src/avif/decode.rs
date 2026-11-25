//! AVIF görüntü çözme, `grid` karo birleştirme, alfa kanalı ve `clap`/`irot`/`imir` dönüşüm boru hattı.

#![forbid(unsafe_code)]

use crate::api::{Config, Decoder, Packet};
use crate::avif::boxes::{
    ALPHA_AUX_URN_CICP, ALPHA_AUX_URN_HEVC, Av1CProperty, AvifContainer, ItemProperty,
    parse_container,
};
use crate::avif::image::{
    AvifDecodeConfig, AvifImage, AvifInfo, CleanAperture, ColorInfo, RgbImage, RgbaImage,
};
use crate::bitstream::obu::parse_sequence_header_obus;
use crate::bitstream::seq_header::PixelLayout;
use crate::error::{Error, Result, SizeStage};
use crate::frame::{BitDepth, ChromaFormat, Frame, PlaneType};

/// İç düzlemsel çalışma tamponu.
#[derive(Clone, Debug)]
struct PlanarBuffer {
    width: u32,
    height: u32,
    chroma_width: u32,
    chroma_height: u32,
    bit_depth: BitDepth,
    chroma_format: ChromaFormat,
    y_plane: Vec<u16>,
    u_plane: Vec<u16>,
    v_plane: Vec<u16>,
    seq_color: Option<ColorInfo>,
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

fn av1c_to_format(av1c: &Av1CProperty) -> Result<(BitDepth, ChromaFormat)> {
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

fn compute_clap_rect(w: u32, h: u32, clap: &CleanAperture) -> Result<(u32, u32, u32, u32)> {
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

/// Bir öğenin dönüşümler (`clap`, `irot`, `imir`) sonrası boyutlarını ve temel özelliklerini çözümler.
fn inspect_item_dimensions(
    container: &AvifContainer<'_>,
    item_id: u32,
    depth: usize,
) -> Result<(u32, u32, BitDepth, ChromaFormat, Option<ColorInfo>)> {
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
            let (_, _, t_bd, t_cf, t_col) =
                inspect_item_dimensions(container, tiles[0], depth + 1)?;
            if seq_color.is_none() {
                seq_color = t_col;
            }
            let (gw, gh) = ispe_dims.unwrap_or((grid.output_width, grid.output_height));
            (
                gw,
                gh,
                bit_depth.unwrap_or(t_bd),
                chroma_format.unwrap_or(t_cf),
            )
        }
        b"iden" => {
            let targets = container.references_from(item_id, b"dimg");
            if targets.len() != 1 {
                return Err(Error::InvalidData(
                    "iden öğesi tek bir dimg referansı içermelidir",
                ));
            }
            let (tw, th, t_bd, t_cf, t_col) =
                inspect_item_dimensions(container, targets[0], depth + 1)?;
            if seq_color.is_none() {
                seq_color = t_col;
            }
            (tw, th, t_bd, t_cf)
        }
        _ => return Err(Error::Unsupported("desteklenmeyen birincil AVIF öğe türü")),
    };

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

    Ok((w, h, bd, cf, seq_color))
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
        let mut alpha_tiles = Vec::with_capacity(tile_ids.len());
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

type MetadataPair = (Option<Vec<u8>>, Option<Vec<u8>>);

fn extract_metadata_items(
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

/// Piksel çözme işlemi yapmadan AVIF kapsayıcısının başlık, boyut, renk ve metaveri bilgilerini ayrıştırır.
pub fn parse_info(data: &[u8]) -> Result<AvifInfo> {
    let container = parse_container(data)?;
    let primary_id = container.primary_item_id;
    let (width, height, bit_depth, chroma_format, seq_color) =
        inspect_item_dimensions(&container, primary_id, 0)?;

    let props = container.item_properties(primary_id)?;
    let mut color = seq_color.unwrap_or_default();
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

    Ok(AvifInfo {
        width,
        height,
        bit_depth,
        chroma_format,
        has_alpha: alpha_src.is_some(),
        premultiplied_alpha,
        color,
        icc_profile,
        exif,
        xmp,
        clli,
        mdcv,
        pasp,
    })
}

fn check_pixel_and_byte_budget(
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

#[inline]
fn limited_to_full_y(bit_depth: BitDepth, v: u16) -> u16 {
    let (min_y, max_y, full_y): (i32, i32, i32) = match bit_depth {
        BitDepth::Eight => (16, 235, 255),
        BitDepth::Ten => (64, 940, 1023),
        BitDepth::Twelve => (256, 3760, 4095),
    };
    let span = max_y - min_y;
    let expanded = (((i32::from(v) - min_y) * full_y) + (span / 2)) / span;
    expanded.clamp(0, full_y) as u16
}

fn extract_plane_u16(
    frame: &Frame,
    plane_type: PlaneType,
    allocation_limit: Option<usize>,
) -> Result<(u32, u32, Vec<u16>)> {
    let Some(plane) = frame.plane(plane_type) else {
        return Ok((0, 0, Vec::new()));
    };
    let w = plane.width();
    let h = plane.height();
    let count = w.checked_mul(h).ok_or(Error::SizeOverflow {
        stage: SizeStage::PlaneArea,
        lhs: w,
        rhs: h,
    })?;
    let bytes_req = count
        .checked_mul(core::mem::size_of::<u16>())
        .ok_or(Error::SizeOverflow {
            stage: SizeStage::TotalAllocation,
            lhs: count,
            rhs: core::mem::size_of::<u16>(),
        })?;
    if let Some(limit) = allocation_limit
        && bytes_req > limit
    {
        return Err(Error::BudgetExceeded {
            requested: bytes_req,
            limit,
        });
    }
    let mut out = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|_| Error::AllocationFailed {
            requested: bytes_req,
        })?;

    match frame.bit_depth() {
        BitDepth::Eight => {
            for y in 0..h {
                let row = plane.row(y).ok_or(Error::InvalidData(
                    "çözülen karede eksik 8-bit düzlem satırı",
                ))?;
                for &b in &row[..w] {
                    out.push(u16::from(b));
                }
            }
        }
        BitDepth::Ten | BitDepth::Twelve => {
            let row_bytes_len = w * 2;
            for y in 0..h {
                let row = plane.row(y).ok_or(Error::InvalidData(
                    "çözülen karede eksik 16-bit düzlem satırı",
                ))?;
                for &pair in row[..row_bytes_len].as_chunks::<2>().0 {
                    out.push(u16::from_le_bytes(pair));
                }
            }
        }
    }
    Ok((w as u32, h as u32, out))
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

    let (w, h, mut y_plane) = extract_plane_u16(&frame, PlaneType::Y, config.allocation_limit)?;
    let (cw, ch, u_plane) = extract_plane_u16(&frame, PlaneType::U, config.allocation_limit)?;
    let (_, _, v_plane) = extract_plane_u16(&frame, PlaneType::V, config.allocation_limit)?;

    // AVIF v1.0.0 geriye dönük uyumluluğu (`libavif` `avifImageLimitedToFullAlpha`):
    // Alfa düzlemi sınırlı aralıkta (`color_range == 0`) kodlanmışsa tam aralığa genişletilir.
    if is_alpha && seq_color.is_some_and(|c| !c.full_range) {
        let bd = frame.bit_depth();
        for sample in &mut y_plane {
            *sample = limited_to_full_y(bd, *sample);
        }
    }

    Ok(PlanarBuffer {
        width: w,
        height: h,
        chroma_width: cw,
        chroma_height: ch,
        bit_depth: frame.bit_depth(),
        chroma_format: frame.chroma_format(),
        y_plane,
        u_plane,
        v_plane,
        seq_color,
    })
}

fn crop_planar(
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
    if let Some(limit) = allocation_limit
        && y_count * 2 > limit
    {
        return Err(Error::BudgetExceeded {
            requested: y_count * 2,
            limit,
        });
    }
    let mut new_y = Vec::with_capacity(y_count);
    for r in 0..new_h {
        let src_start = ((y0 as usize) + r) * old_w + (x0 as usize);
        new_y.extend_from_slice(&buf.y_plane[src_start..src_start + new_w]);
    }
    buf.y_plane = new_y;
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
        let c_count = new_cw * new_ch;
        let mut new_u = Vec::with_capacity(c_count);
        let mut new_v = Vec::with_capacity(c_count);
        for r in 0..new_h.div_ceil(1 << sub_y) {
            let src_start = (cy0 + r) * old_cw + cx0;
            new_u.extend_from_slice(&buf.u_plane[src_start..src_start + new_cw]);
            new_v.extend_from_slice(&buf.v_plane[src_start..src_start + new_cw]);
        }
        buf.u_plane = new_u;
        buf.v_plane = new_v;
        buf.chroma_width = new_cw as u32;
        buf.chroma_height = new_ch as u32;
    }
    Ok(())
}

fn rotate_plane_ccw(src: &[u16], w: usize, h: usize, angle: u8) -> (usize, usize, Vec<u16>) {
    match angle & 3 {
        1 => {
            let mut dst = vec![0u16; w * h];
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
            let mut dst = vec![0u16; w * h];
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
            let mut dst = vec![0u16; w * h];
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

fn upsample_c422_to_c444(buf: &mut PlanarBuffer) {
    if buf.chroma_format != ChromaFormat::C422 {
        return;
    }
    let w = buf.width as usize;
    let h = buf.height as usize;
    let cw = buf.chroma_width as usize;
    let mut new_u = Vec::with_capacity(w * h);
    let mut new_v = Vec::with_capacity(w * h);
    for y in 0..h {
        let row_off = y * cw;
        for x in 0..w {
            let cx = (x / 2).min(cw.saturating_sub(1));
            new_u.push(buf.u_plane[row_off + cx]);
            new_v.push(buf.v_plane[row_off + cx]);
        }
    }
    buf.u_plane = new_u;
    buf.v_plane = new_v;
    buf.chroma_width = buf.width;
    buf.chroma_height = buf.height;
    buf.chroma_format = ChromaFormat::C444;
}

fn apply_irot(buf: &mut PlanarBuffer, angle: u8) {
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

fn mirror_plane(plane: &mut [u16], w: usize, h: usize, axis: u8) {
    if w == 0 || h == 0 {
        return;
    }
    if (axis & 1) == 0 {
        for y in 0..(h / 2) {
            let top_start = y * w;
            let bot_start = (h - 1 - y) * w;
            for x in 0..w {
                plane.swap(top_start + x, bot_start + x);
            }
        }
    } else {
        for y in 0..h {
            let row = &mut plane[y * w..(y + 1) * w];
            row.reverse();
        }
    }
}

fn apply_imir(buf: &mut PlanarBuffer, axis: u8) {
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

    let mut unique_ids = Vec::with_capacity(num_tiles);
    let mut tile_to_unique = Vec::with_capacity(num_tiles);
    let mut id_map =
        std::collections::HashMap::with_capacity_and_hasher(num_tiles, rustc_hash::FxBuildHasher);
    for &tid in tile_ids {
        let u_idx = *id_map.entry(tid).or_insert_with(|| {
            let idx = unique_ids.len();
            unique_ids.push(tid);
            idx
        });
        tile_to_unique.push(u_idx);
    }
    let num_unique = unique_ids.len();

    let total_threads = if config.n_threads == 0 {
        std::thread::available_parallelism()
            .map(core::num::NonZeroUsize::get)
            .unwrap_or(1)
    } else {
        config.n_threads
    };

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
    let total_bytes = (y_samples + 2 * c_samples) * core::mem::size_of::<u16>();
    check_pixel_and_byte_budget(grid.output_width, grid.output_height, total_bytes, config)?;

    let mut y_canvas = vec![0u16; y_samples];
    let mut u_canvas = vec![0u16; c_samples];
    let mut v_canvas = vec![0u16; c_samples];

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

            for ty in 0..copy_h {
                let src_off = ty * tile_w;
                let dst_off = (dst_y0 + ty) * out_w + dst_x0;
                y_canvas[dst_off..dst_off + copy_w]
                    .copy_from_slice(&tile.y_plane[src_off..src_off + copy_w]);
            }

            if !is_alpha && chroma_format != ChromaFormat::Monochrome && copy_ch > 0 {
                let dst_cx0 = c * tile_cw;
                if dst_cx0 < out_cw {
                    let copy_cw = tile_cw.min(out_cw - dst_cx0);
                    for tcy in 0..copy_ch {
                        let src_off = tcy * tile_cw;
                        let dst_off = (dst_cy0 + tcy) * out_cw + dst_cx0;
                        u_canvas[dst_off..dst_off + copy_cw]
                            .copy_from_slice(&tile.u_plane[src_off..src_off + copy_cw]);
                        v_canvas[dst_off..dst_off + copy_cw]
                            .copy_from_slice(&tile.v_plane[src_off..src_off + copy_cw]);
                    }
                }
            }
        }
    }

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

    let mut color_buf =
        decode_item_planar(&container, primary_id, config, config.n_threads, false, 0)?;

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
        let mut alpha_buf = match src {
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
        Some(core::mem::take(&mut alpha_buf.y_plane))
    } else {
        None
    };

    let (exif, xmp) = extract_metadata_items(&container, primary_id, config.allocation_limit)?;

    Ok(AvifImage {
        width: color_buf.width,
        height: color_buf.height,
        chroma_width: color_buf.chroma_width,
        chroma_height: color_buf.chroma_height,
        bit_depth: color_buf.bit_depth,
        chroma_format: color_buf.chroma_format,
        y_plane: core::mem::take(&mut color_buf.y_plane),
        u_plane: core::mem::take(&mut color_buf.u_plane),
        v_plane: core::mem::take(&mut color_buf.v_plane),
        alpha_plane,
        premultiplied_alpha,
        color,
        icc_profile,
        exif,
        xmp,
        clli,
        mdcv,
        pasp,
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
