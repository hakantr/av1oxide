//! `av1oxide::avif` (`.avif` ISOBMFF/HEIF/MIAF görüntü okuyucu) entegrasyon testleri.

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use av1oxide::avif::{
    AvifDecodeConfig, AvifPlane, AvifPrepared, AvifSequenceDecoder, CleanAperture, ColorInfo,
    ContentLightLevel, LoopCount, MasteringDisplay, PixelAspectRatio, decode, decode_rgb8,
    decode_rgba8, decode_with, parse_info, parse_info_with, parse_sequence_info,
    parse_sequence_info_with, probe,
};
use av1oxide::{BitDepth, ChromaFormat, Error};

/// Gömülü 1×1, 8-bit, 4:2:0 AV1 anahtar kare OBU yükü (`Y=190, U=128, V=128`).
const SAMPLE_1X1_AV1_OBU: &[u8] = &[
    0x12, 0x00, 0x0A, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7D, 0xFC, 0xC0, 0x20, 0x32, 0x17, 0x10,
    0x01, 0x91, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0xD2, 0x16, 0x8C, 0x64,
    0xBE, 0x2E, 0xC7, 0xDA, 0xEE, 0x96,
];

fn make_box(box_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let size = (8 + payload.len()) as u32;
    let mut out = Vec::with_capacity(size as usize);
    out.extend_from_slice(&size.to_be_bytes());
    out.extend_from_slice(box_type);
    out.extend_from_slice(payload);
    out
}

fn make_large_box(box_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let size = (16 + payload.len()) as u64;
    let mut out = Vec::with_capacity(size as usize);
    out.extend_from_slice(&1u32.to_be_bytes());
    out.extend_from_slice(box_type);
    out.extend_from_slice(&size.to_be_bytes());
    out.extend_from_slice(payload);
    out
}

fn make_full_box(box_type: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(4 + body.len());
    payload.push(version);
    payload.push(((flags >> 16) & 0xFF) as u8);
    payload.push(((flags >> 8) & 0xFF) as u8);
    payload.push((flags & 0xFF) as u8);
    payload.extend_from_slice(body);
    make_box(box_type, &payload)
}

fn make_ftyp(major: &[u8; 4], compat: &[[u8; 4]]) -> Vec<u8> {
    let mut body = Vec::with_capacity(8 + compat.len() * 4);
    body.extend_from_slice(major);
    body.extend_from_slice(&0u32.to_be_bytes());
    for c in compat {
        body.extend_from_slice(c);
    }
    make_box(b"ftyp", &body)
}

fn make_hdlr_pict() -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&0u32.to_be_bytes());
    body.extend_from_slice(b"pict");
    body.extend_from_slice(&[0u8; 13]);
    make_full_box(b"hdlr", 0, 0, &body)
}

fn make_infe_v2(item_id: u16, item_type: &[u8; 4], content_type: Option<&str>) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&item_id.to_be_bytes());
    body.extend_from_slice(&0u16.to_be_bytes());
    body.extend_from_slice(item_type);
    body.push(0); // item_name = ""
    if let Some(ct) = content_type {
        body.extend_from_slice(ct.as_bytes());
        body.push(0);
    }
    make_full_box(b"infe", 2, 0, &body)
}

fn make_ispe(width: u32, height: u32) -> Vec<u8> {
    let mut body = Vec::with_capacity(8);
    body.extend_from_slice(&width.to_be_bytes());
    body.extend_from_slice(&height.to_be_bytes());
    make_full_box(b"ispe", 0, 0, &body)
}

fn first_ivf_packet(bytes: &[u8]) -> &[u8] {
    assert!(bytes.len() >= 32 + 12 && &bytes[0..4] == b"DKIF");
    let frame_size = u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]) as usize;
    &bytes[44..44 + frame_size]
}

fn read_oddheight_keyframe_obu() -> Vec<u8> {
    let candidates = [
        "../dav1d-test-data/8-bit/features/422_oddheight.ivf",
        "../../dav1d-test-data/8-bit/features/422_oddheight.ivf",
    ];
    for p in candidates {
        if let Ok(bytes) = fs::read(p) {
            return first_ivf_packet(&bytes).to_vec();
        }
    }
    SAMPLE_1X1_AV1_OBU.to_vec()
}

#[test]
fn avif_single_item_metadata_and_color_matrices() {
    assert!(!probe(&[]));
    assert!(!probe(b"not_an_avif_file"));
    assert!(!probe(&make_ftyp(b"mp42", &[*b"isom"])));

    // Sequence Header (13 bayt = 2 bayt TD + 11 bayt Sequence Header OBU) av1C.configOBUs içinde,
    // Frame OBU (25 bayt) ise iki parçalı (multi-extent) idat içinde.
    let seq_only = &SAMPLE_1X1_AV1_OBU[0..13];
    let frame_only = &SAMPLE_1X1_AV1_OBU[13..];

    let ftyp = make_ftyp(b"mif1", &[*b"avif", *b"miaf"]);
    let hdlr = make_hdlr_pict();
    let pitm = make_full_box(b"pitm", 0, 0, &1u16.to_be_bytes());

    // Öğeler: 1 = av01, 2 = Exif, 3 = XMP (mime)
    let infe1 = make_infe_v2(1, b"av01", None);
    let infe2 = make_infe_v2(2, b"Exif", None);
    let infe3 = make_infe_v2(3, b"mime", Some("application/rdf+xml"));
    let mut iinf_body = Vec::new();
    iinf_body.extend_from_slice(&3u16.to_be_bytes());
    iinf_body.extend_from_slice(&infe1);
    iinf_body.extend_from_slice(&infe2);
    iinf_body.extend_from_slice(&infe3);
    let iinf = make_full_box(b"iinf", 0, 0, &iinf_body);

    // ipco özellikleri:
    // 1: ispe (1x1)
    // 2: pixi (8, 8, 8)
    // 3: av1C (configOBUs = seq_only)
    // 4: colr nclx (Identity matrix = 0, full_range = true)
    // 5: colr rICC (ICC profili)
    // 6: clli (1000, 400)
    // 7: mdcv
    // 8: pasp (1, 1)
    let ispe = make_ispe(1, 1);
    let pixi = make_full_box(b"pixi", 0, 0, &[3, 8, 8, 8]);
    let mut av1c_payload = vec![0x81, 0x00, 0x0C, 0x00];
    av1c_payload.extend_from_slice(seq_only);
    let av1c = make_box(b"av1C", &av1c_payload);
    let colr_nclx = make_box(b"colr", &[b'n', b'c', b'l', b'x', 0, 1, 0, 13, 0, 0, 0x80]);
    let icc_bytes = b"MOCK_ICC_PROFILE";
    let mut colr_icc_payload = b"rICC".to_vec();
    colr_icc_payload.extend_from_slice(icc_bytes);
    let colr_icc = make_box(b"colr", &colr_icc_payload);
    let clli = make_box(b"clli", &[0x03, 0xE8, 0x01, 0x90]); // 1000, 400
    let mut mdcv_payload = Vec::new();
    for coord in [13250u16, 34500, 7500, 3000, 34000, 16000, 15635, 16450] {
        mdcv_payload.extend_from_slice(&coord.to_be_bytes());
    }
    mdcv_payload.extend_from_slice(&10_000_000u32.to_be_bytes());
    mdcv_payload.extend_from_slice(&50u32.to_be_bytes());
    let mdcv = make_box(b"mdcv", &mdcv_payload);
    let pasp = make_box(b"pasp", &[0, 0, 0, 1, 0, 0, 0, 1]);

    let mut ipco_body = Vec::new();
    for p in [
        &ispe, &pixi, &av1c, &colr_nclx, &colr_icc, &clli, &mdcv, &pasp,
    ] {
        ipco_body.extend_from_slice(p);
    }
    let ipco = make_box(b"ipco", &ipco_body);

    let mut ipma_body = Vec::new();
    ipma_body.extend_from_slice(&1u32.to_be_bytes());
    ipma_body.extend_from_slice(&1u16.to_be_bytes());
    ipma_body.push(8);
    for idx in 1u8..=8 {
        ipma_body.push(idx);
    }
    let ipma = make_full_box(b"ipma", 0, 0, &ipma_body);

    let mut iprp_body = Vec::new();
    iprp_body.extend_from_slice(&ipco);
    iprp_body.extend_from_slice(&ipma);
    let iprp = make_box(b"iprp", &iprp_body);

    // iref: cdsc (2 -> 1 Exif, 3 -> 1 XMP)
    let mut cdsc_exif = Vec::new();
    cdsc_exif.extend_from_slice(&2u16.to_be_bytes());
    cdsc_exif.extend_from_slice(&1u16.to_be_bytes());
    cdsc_exif.extend_from_slice(&1u16.to_be_bytes());
    let mut cdsc_xmp = Vec::new();
    cdsc_xmp.extend_from_slice(&3u16.to_be_bytes());
    cdsc_xmp.extend_from_slice(&1u16.to_be_bytes());
    cdsc_xmp.extend_from_slice(&1u16.to_be_bytes());
    let mut iref_body = Vec::new();
    iref_body.extend_from_slice(&make_box(b"cdsc", &cdsc_exif));
    iref_body.extend_from_slice(&make_box(b"cdsc", &cdsc_xmp));
    let iref = make_full_box(b"iref", 0, 0, &iref_body);

    // idat: frame_only (2 extent olarak bölünmüş) + Exif (4-bayt ofset = 2 + 2 dolgu + "II*\0") + XMP
    let split = frame_only.len() / 2;
    let ext1 = &frame_only[..split];
    let ext2 = &frame_only[split..];
    let exif_raw = b"\x00\x00\x00\x02XXII*\x00EXIF_PAYLOAD";
    let xmp_raw = b"<x:xmpmeta>AV1OXIDE</x:xmpmeta>";

    let mut idat_payload = Vec::new();
    idat_payload.extend_from_slice(ext1);
    let ext2_off = idat_payload.len() as u32;
    idat_payload.extend_from_slice(ext2);
    let exif_off = idat_payload.len() as u32;
    idat_payload.extend_from_slice(exif_raw);
    let xmp_off = idat_payload.len() as u32;
    idat_payload.extend_from_slice(xmp_raw);
    let idat = make_large_box(b"idat", &idat_payload);

    // iloc v1: 3 öğe, hepsi construction_method = 1 (idat)
    let mut iloc_body = Vec::new();
    iloc_body.push(0x44); // offset_size=4, length_size=4
    iloc_body.push(0x00);
    iloc_body.extend_from_slice(&3u16.to_be_bytes());
    // item 1: 2 extent
    iloc_body.extend_from_slice(&1u16.to_be_bytes());
    iloc_body.extend_from_slice(&1u16.to_be_bytes());
    iloc_body.extend_from_slice(&0u16.to_be_bytes());
    iloc_body.extend_from_slice(&2u16.to_be_bytes());
    iloc_body.extend_from_slice(&0u32.to_be_bytes());
    iloc_body.extend_from_slice(&(ext1.len() as u32).to_be_bytes());
    iloc_body.extend_from_slice(&ext2_off.to_be_bytes());
    iloc_body.extend_from_slice(&(ext2.len() as u32).to_be_bytes());
    // item 2: Exif
    iloc_body.extend_from_slice(&2u16.to_be_bytes());
    iloc_body.extend_from_slice(&1u16.to_be_bytes());
    iloc_body.extend_from_slice(&0u16.to_be_bytes());
    iloc_body.extend_from_slice(&1u16.to_be_bytes());
    iloc_body.extend_from_slice(&exif_off.to_be_bytes());
    iloc_body.extend_from_slice(&(exif_raw.len() as u32).to_be_bytes());
    // item 3: XMP
    iloc_body.extend_from_slice(&3u16.to_be_bytes());
    iloc_body.extend_from_slice(&1u16.to_be_bytes());
    iloc_body.extend_from_slice(&0u16.to_be_bytes());
    iloc_body.extend_from_slice(&1u16.to_be_bytes());
    iloc_body.extend_from_slice(&xmp_off.to_be_bytes());
    iloc_body.extend_from_slice(&(xmp_raw.len() as u32).to_be_bytes());
    let iloc = make_full_box(b"iloc", 1, 0, &iloc_body);

    let mut meta_children = Vec::new();
    for b in [&hdlr, &pitm, &iinf, &iprp, &iref, &iloc, &idat] {
        meta_children.extend_from_slice(b);
    }
    let meta = make_full_box(b"meta", 0, 0, &meta_children);

    // Son kutu size == 0 (EOF kutusu)
    let mut free_eof = Vec::new();
    free_eof.extend_from_slice(&0u32.to_be_bytes());
    free_eof.extend_from_slice(b"free");
    free_eof.extend_from_slice(&[0u8; 4]);

    let mut avif_file = Vec::new();
    avif_file.extend_from_slice(&ftyp);
    avif_file.extend_from_slice(&meta);
    avif_file.extend_from_slice(&free_eof);

    assert!(probe(&avif_file));
    let info = parse_info(&avif_file).expect("parse_info başarılı olmalı");
    assert_eq!((info.width, info.height), (1, 1));
    assert_eq!(info.bit_depth, BitDepth::Eight);
    assert_eq!(info.chroma_format, ChromaFormat::C420);
    assert!(!info.has_alpha);
    assert!(!info.premultiplied_alpha);
    assert_eq!(
        info.color,
        ColorInfo {
            color_primaries: 1,
            transfer_characteristics: 13,
            matrix_coefficients: 0,
            full_range: true,
        }
    );
    assert_eq!(info.icc_profile.as_deref(), Some(&icc_bytes[..]));
    assert_eq!(info.exif.as_deref(), Some(&b"II*\x00EXIF_PAYLOAD"[..]));
    assert_eq!(info.xmp.as_deref(), Some(&xmp_raw[..]));
    assert_eq!(
        info.clli,
        Some(ContentLightLevel {
            max_content_light_level: 1000,
            max_pic_average_light_level: 400,
        })
    );
    assert_eq!(
        info.mdcv,
        Some(MasteringDisplay {
            primaries: [[13250, 34500], [7500, 3000], [34000, 16000]],
            white_point: [15635, 16450],
            max_luminance: 10_000_000,
            min_luminance: 50,
        })
    );
    assert_eq!(
        info.pasp,
        Some(PixelAspectRatio {
            h_spacing: 1,
            v_spacing: 1,
        })
    );

    // Identity (matrix_coefficients == 0): R = V, G = Y, B = U
    let img = decode(&avif_file).expect("decode başarılı olmalı");
    assert_eq!(img.y_plane(), AvifPlane::U8(&[190]));
    assert_eq!(img.y_plane_u8(), Some(&[190u8][..]));
    assert_eq!(img.y_plane_u16(), None);
    let u0 = img.u_plane_u8().unwrap()[0];
    let v0 = img.v_plane_u8().unwrap()[0];
    assert_eq!(u0, 1);
    let rgb8 = img.to_rgb8().unwrap();
    assert_eq!(rgb8.pixels, vec![v0, 190, u0]);
    let rgba8 = img.to_rgba8().unwrap();
    assert_eq!(rgba8.pixels, vec![v0, 190, u0, 255]);
    let rgb16 = img.to_rgb16().unwrap();
    assert_eq!(rgb16.pixels, vec![u16::from(v0), 190, u16::from(u0)]);
    let rgba16 = img.to_rgba16().unwrap();
    assert_eq!(rgba16.pixels, vec![u16::from(v0), 190, u16::from(u0), 255]);
    assert_eq!(img.y_row(0), Some(AvifPlane::U8(&[190u8][..])));
    assert_eq!(img.y_row_u8(0), Some(&[190u8][..]));
    assert_eq!(img.y_row_u16(0), None);
    assert_eq!(img.u_row(0), Some(AvifPlane::U8(&[u0][..])));
    assert_eq!(img.u_row_u8(0), Some(&[u0][..]));
    assert_eq!(img.v_row(0), Some(AvifPlane::U8(&[v0][..])));
    assert_eq!(img.v_row_u8(0), Some(&[v0][..]));
}

/// `2×2` `grid` karolu + yardımcı alfa (`auxl`/`auxC` + `prem`) kanallı `.avif` oluşturur.
fn build_grid_and_alpha_avif(
    tile_obu: &[u8],
    tile_w: u32,
    tile_h: u32,
    out_w: u32,
    out_h: u32,
) -> Vec<u8> {
    let ftyp = make_ftyp(b"avif", &[*b"mif1", *b"miaf"]);
    let hdlr = make_hdlr_pict();
    // Birincil öğe: 10 (iden -> 1 grid); karolar: 2, 3, 4, 5 (av01); alfa: 6 (av01)
    let pitm = make_full_box(b"pitm", 0, 0, &10u16.to_be_bytes());

    let mut iinf_body = Vec::new();
    iinf_body.extend_from_slice(&7u16.to_be_bytes());
    iinf_body.extend_from_slice(&make_infe_v2(10, b"iden", None));
    iinf_body.extend_from_slice(&make_infe_v2(1, b"grid", None));
    for tid in 2u16..=5 {
        iinf_body.extend_from_slice(&make_infe_v2(tid, b"av01", None));
    }
    iinf_body.extend_from_slice(&make_infe_v2(6, b"av01", None));
    let iinf = make_full_box(b"iinf", 0, 0, &iinf_body);

    // ipco:
    // 1: ispe (out_w, out_h) -> grid & iden & alpha
    // 2: ispe (tile_w, tile_h) -> tiles 2..=5
    // 3: auxC (alpha URN) -> item 6
    let ispe_out = make_ispe(out_w, out_h);
    let ispe_tile = make_ispe(tile_w, tile_h);
    let mut auxc_body = b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\0".to_vec();
    auxc_body.push(0);
    let auxc = make_full_box(b"auxC", 0, 0, &auxc_body);

    let mut ipco_body = Vec::new();
    ipco_body.extend_from_slice(&ispe_out);
    ipco_body.extend_from_slice(&ispe_tile);
    ipco_body.extend_from_slice(&auxc);
    let ipco = make_box(b"ipco", &ipco_body);

    let mut ipma_body = Vec::new();
    ipma_body.extend_from_slice(&7u32.to_be_bytes());
    // item 10 (iden) -> prop 1
    ipma_body.extend_from_slice(&10u16.to_be_bytes());
    ipma_body.push(1);
    ipma_body.push(1);
    // item 1 (grid) -> prop 1
    ipma_body.extend_from_slice(&1u16.to_be_bytes());
    ipma_body.push(1);
    ipma_body.push(1);
    // tiles 2..=5 -> prop 2
    for tid in 2u16..=5 {
        ipma_body.extend_from_slice(&tid.to_be_bytes());
        ipma_body.push(1);
        ipma_body.push(2);
    }
    // item 6 (alpha) -> prop 1, prop 3
    ipma_body.extend_from_slice(&6u16.to_be_bytes());
    ipma_body.push(2);
    ipma_body.push(1);
    ipma_body.push(0x80 | 3);
    let ipma = make_full_box(b"ipma", 0, 0, &ipma_body);

    let mut iprp_body = Vec::new();
    iprp_body.extend_from_slice(&ipco);
    iprp_body.extend_from_slice(&ipma);
    let iprp = make_box(b"iprp", &iprp_body);

    // iref:
    // - dimg: 10 -> [1]
    // - dimg: 1 -> [2, 3, 4, 5]
    // - auxl: 6 -> [10]
    // - prem: 10 -> [6]
    let mut dimg_iden = Vec::new();
    dimg_iden.extend_from_slice(&10u16.to_be_bytes());
    dimg_iden.extend_from_slice(&1u16.to_be_bytes());
    dimg_iden.extend_from_slice(&1u16.to_be_bytes());

    let mut dimg_grid = Vec::new();
    dimg_grid.extend_from_slice(&1u16.to_be_bytes());
    dimg_grid.extend_from_slice(&4u16.to_be_bytes());
    for tid in 2u16..=5 {
        dimg_grid.extend_from_slice(&tid.to_be_bytes());
    }

    let mut auxl_box = Vec::new();
    auxl_box.extend_from_slice(&6u16.to_be_bytes());
    auxl_box.extend_from_slice(&1u16.to_be_bytes());
    auxl_box.extend_from_slice(&10u16.to_be_bytes());

    let mut prem_box = Vec::new();
    prem_box.extend_from_slice(&10u16.to_be_bytes());
    prem_box.extend_from_slice(&1u16.to_be_bytes());
    prem_box.extend_from_slice(&6u16.to_be_bytes());

    let mut iref_body = Vec::new();
    iref_body.extend_from_slice(&make_box(b"dimg", &dimg_iden));
    iref_body.extend_from_slice(&make_box(b"dimg", &dimg_grid));
    iref_body.extend_from_slice(&make_box(b"auxl", &auxl_box));
    iref_body.extend_from_slice(&make_box(b"prem", &prem_box));
    let iref = make_full_box(b"iref", 0, 0, &iref_body);

    // ImageGrid descriptor (8 bayt: version=0, flags=0, rows_minus_one=1, cols_minus_one=1, out_w, out_h)
    let mut grid_desc = vec![0u8, 0u8, 1u8, 1u8];
    grid_desc.extend_from_slice(&(out_w as u16).to_be_bytes());
    grid_desc.extend_from_slice(&(out_h as u16).to_be_bytes());

    let mut idat_payload = Vec::new();
    idat_payload.extend_from_slice(&grid_desc);
    let tile_off = idat_payload.len() as u32;
    idat_payload.extend_from_slice(tile_obu);
    let idat = make_box(b"idat", &idat_payload);

    // iloc v1: item 1 (grid_desc), items 2..=5 (tile_obu), item 6 (alpha = tile_obu)
    let mut iloc_body = Vec::new();
    iloc_body.push(0x44);
    iloc_body.push(0x00);
    iloc_body.extend_from_slice(&6u16.to_be_bytes());
    // item 1
    iloc_body.extend_from_slice(&1u16.to_be_bytes());
    iloc_body.extend_from_slice(&1u16.to_be_bytes());
    iloc_body.extend_from_slice(&0u16.to_be_bytes());
    iloc_body.extend_from_slice(&1u16.to_be_bytes());
    iloc_body.extend_from_slice(&0u32.to_be_bytes());
    iloc_body.extend_from_slice(&(grid_desc.len() as u32).to_be_bytes());
    // items 2..=6
    for id in 2u16..=6 {
        iloc_body.extend_from_slice(&id.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&0u16.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&tile_off.to_be_bytes());
        iloc_body.extend_from_slice(&(tile_obu.len() as u32).to_be_bytes());
    }
    let iloc = make_full_box(b"iloc", 1, 0, &iloc_body);

    let mut meta_children = Vec::new();
    for b in [&hdlr, &pitm, &iinf, &iprp, &iref, &iloc, &idat] {
        meta_children.extend_from_slice(b);
    }
    let meta = make_full_box(b"meta", 0, 0, &meta_children);

    let mut file = Vec::new();
    file.extend_from_slice(&ftyp);
    file.extend_from_slice(&meta);
    file
}

/// Birincil öğesi `2×2` grid olan `.avif` oluşturur; karoların türü `karo_turu` ile seçilir:
/// `av01` doğrudan kodlanmış karo, `grid` her karosu yine `2×2` av01 grid'i olan iç içe grid,
/// `iden` her biri tek bir av01 öğesine bakan türetilmiş karo. Tüm öğeler `1×1`'dir.
fn build_grid_with_tile_kind(tile_obu: &[u8], karo_turu: &[u8; 4]) -> Vec<u8> {
    // (öğe kimliği, tür, idat verisi grid tanımı mı, dimg referansları)
    let mut items: Vec<(u16, [u8; 4], bool, Vec<u16>)> = Vec::new();
    let mut next_id = 1u16;
    let mut leaf = || {
        let id = next_id;
        next_id += 1;
        id
    };
    let mut leaves: Vec<u16> = Vec::new();
    for _ in 0..4 {
        let id = leaf();
        items.push((id, *b"av01", false, Vec::new()));
        leaves.push(id);
    }
    let tiles: Vec<u16> = match karo_turu {
        b"av01" => leaves,
        b"grid" => (0..4)
            .map(|_| {
                let id = leaf();
                items.push((id, *b"grid", true, leaves.clone()));
                id
            })
            .collect(),
        b"iden" => leaves
            .iter()
            .map(|&target| {
                let id = leaf();
                items.push((id, *b"iden", false, vec![target]));
                id
            })
            .collect(),
        _ => panic!("desteklenmeyen karo türü"),
    };
    let primary = leaf();
    items.push((primary, *b"grid", true, tiles));

    let ftyp = make_ftyp(b"avif", &[*b"mif1", *b"miaf"]);
    let hdlr = make_hdlr_pict();
    let pitm = make_full_box(b"pitm", 0, 0, &primary.to_be_bytes());

    let mut iinf_body = (items.len() as u16).to_be_bytes().to_vec();
    for (id, typ, _, _) in &items {
        iinf_body.extend_from_slice(&make_infe_v2(*id, typ, None));
    }
    let iinf = make_full_box(b"iinf", 0, 0, &iinf_body);

    let ipco = make_box(b"ipco", &make_ispe(1, 1));
    let mut ipma_body = (items.len() as u32).to_be_bytes().to_vec();
    for (id, _, _, _) in &items {
        ipma_body.extend_from_slice(&id.to_be_bytes());
        ipma_body.push(1);
        ipma_body.push(1);
    }
    let ipma = make_full_box(b"ipma", 0, 0, &ipma_body);
    let iprp = make_box(b"iprp", &[ipco, ipma].concat());

    let mut iref_body = Vec::new();
    for (id, _, _, refs) in &items {
        if refs.is_empty() {
            continue;
        }
        let mut dimg = id.to_be_bytes().to_vec();
        dimg.extend_from_slice(&(refs.len() as u16).to_be_bytes());
        for r in refs {
            dimg.extend_from_slice(&r.to_be_bytes());
        }
        iref_body.extend_from_slice(&make_box(b"dimg", &dimg));
    }
    let iref = make_full_box(b"iref", 0, 0, &iref_body);

    // ImageGrid: version=0, flags=0, rows_minus_one=1, cols_minus_one=1, 1×1 çıktı.
    let grid_desc = [0u8, 0, 1, 1, 0, 1, 0, 1];
    let mut idat_payload = grid_desc.to_vec();
    let tile_off = idat_payload.len() as u32;
    idat_payload.extend_from_slice(tile_obu);
    let idat = make_box(b"idat", &idat_payload);

    // iloc v1, construction_method=1 (idat); `iden` öğelerinin verisi yoktur.
    let located: Vec<_> = items.iter().filter(|(_, t, _, _)| t != b"iden").collect();
    let mut iloc_body = vec![0x44, 0x00];
    iloc_body.extend_from_slice(&(located.len() as u16).to_be_bytes());
    for (id, _, is_grid, _) in located {
        let (off, len) = if *is_grid {
            (0u32, grid_desc.len() as u32)
        } else {
            (tile_off, tile_obu.len() as u32)
        };
        iloc_body.extend_from_slice(&id.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&0u16.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&off.to_be_bytes());
        iloc_body.extend_from_slice(&len.to_be_bytes());
    }
    let iloc = make_full_box(b"iloc", 1, 0, &iloc_body);

    let meta = make_full_box(
        b"meta",
        0,
        0,
        &[hdlr, pitm, iinf, iprp, iref, iloc, idat].concat(),
    );
    [ftyp, meta].concat()
}

/// Grid karoları yalnız kodlanmış `av01` öğeleri olabilir. İç içe grid (ya da `iden` karo)
/// kabul edilseydi her düzey çözme işini karo sayısıyla çarpar ve birkaç KB'lık bir dosya
/// dakikalarca CPU ve GB'larca bellek isteyebilirdi; `parse_info` ile `decode` aynı kararı
/// vermelidir.
#[test]
fn avif_grid_tiles_must_be_coded_items() {
    let duz = build_grid_with_tile_kind(SAMPLE_1X1_AV1_OBU, b"av01");
    let info = parse_info(&duz).expect("av01 karolu grid parse_info başarılı olmalı");
    assert_eq!((info.width, info.height), (1, 1));
    for n_threads in [1, 4] {
        let img = decode_with(
            &duz,
            &AvifDecodeConfig {
                n_threads,
                ..AvifDecodeConfig::default()
            },
        )
        .expect("av01 karolu grid çözülmeli");
        assert_eq!((img.width(), img.height()), (1, 1));
    }

    for karo_turu in [b"grid", b"iden"] {
        let avif = build_grid_with_tile_kind(SAMPLE_1X1_AV1_OBU, karo_turu);
        assert!(
            matches!(parse_info(&avif), Err(Error::InvalidData(_))),
            "{} karolu grid parse_info tarafından reddedilmeli",
            String::from_utf8_lossy(karo_turu)
        );
        for n_threads in [1, 4] {
            let res = decode_with(
                &avif,
                &AvifDecodeConfig {
                    n_threads,
                    ..AvifDecodeConfig::default()
                },
            );
            assert!(
                matches!(res, Err(Error::InvalidData(_))),
                "{} karolu grid decode tarafından reddedilmeli (n_threads={n_threads})",
                String::from_utf8_lossy(karo_turu)
            );
        }
    }
}

#[test]
fn avif_grid_2x2_single_and_multithread_parity_and_alpha() {
    // 1x1 karolardan oluşan 2x2 grid -> 1x1 ispe kırpmasıyla alfa (1x1) ile eşleşir;
    // ayrıca 2x2 tam ızgarayı tek ve çok-thread altında doğrularız.
    let avif_with_alpha = build_grid_and_alpha_avif(SAMPLE_1X1_AV1_OBU, 1, 1, 1, 1);
    let info = parse_info(&avif_with_alpha).expect("grid+alpha parse_info başarılı olmalı");
    assert_eq!((info.width, info.height), (1, 1));
    assert!(info.has_alpha);
    assert!(info.premultiplied_alpha);

    let img_t1 = decode_with(
        &avif_with_alpha,
        &AvifDecodeConfig {
            n_threads: 1,
            ..AvifDecodeConfig::default()
        },
    )
    .expect("grid+alpha t=1 çözülmeli");
    let img_t4 = decode_with(
        &avif_with_alpha,
        &AvifDecodeConfig {
            n_threads: 4,
            ..AvifDecodeConfig::default()
        },
    )
    .expect("grid+alpha t=4 çözülmeli");

    assert_eq!(img_t1, img_t4);
    assert!(img_t1.has_alpha());
    assert!(img_t1.premultiplied_alpha());
    // SAMPLE_1X1_AV1_OBU limited-range (color_range=0) Y=190 -> tam aralık alfa = 203
    assert_eq!(img_t1.alpha_plane(), Some(AvifPlane::U8(&[203u8][..])));
    assert_eq!(img_t1.alpha_plane_u8(), Some(&[203u8][..]));
    assert_eq!(img_t1.alpha_plane_u16(), None);
    assert_eq!(img_t1.alpha_row(0), Some(AvifPlane::U8(&[203u8][..])));
    assert_eq!(img_t1.alpha_row_u8(0), Some(&[203u8][..]));
    assert_eq!(img_t1.alpha_row_u16(0), None);
    let rgba = img_t1.to_rgba8().unwrap();
    assert_eq!(rgba.pixels[3], 203);
    assert_eq!(rgba, img_t4.to_rgba8().unwrap());

    // 64x63 gerçek AV1 anahtar karesinden 2x2 grid (128x126 -> 120x110 kırpılmış tuval)
    let odd_obu = read_oddheight_keyframe_obu();
    if odd_obu.len() > SAMPLE_1X1_AV1_OBU.len() {
        let ftyp = make_ftyp(b"avif", &[*b"mif1"]);
        let hdlr = make_hdlr_pict();
        let pitm = make_full_box(b"pitm", 0, 0, &1u16.to_be_bytes());
        let mut iinf_body = Vec::new();
        iinf_body.extend_from_slice(&5u16.to_be_bytes());
        iinf_body.extend_from_slice(&make_infe_v2(1, b"grid", None));
        for tid in 2u16..=5 {
            iinf_body.extend_from_slice(&make_infe_v2(tid, b"av01", None));
        }
        let iinf = make_full_box(b"iinf", 0, 0, &iinf_body);

        let ispe_grid = make_ispe(120, 110);
        let ispe_tile = make_ispe(64, 63);
        let mut ipco_body = Vec::new();
        ipco_body.extend_from_slice(&ispe_grid);
        ipco_body.extend_from_slice(&ispe_tile);
        let ipco = make_box(b"ipco", &ipco_body);

        let mut ipma_body = Vec::new();
        ipma_body.extend_from_slice(&5u32.to_be_bytes());
        ipma_body.extend_from_slice(&1u16.to_be_bytes());
        ipma_body.push(1);
        ipma_body.push(1);
        for tid in 2u16..=5 {
            ipma_body.extend_from_slice(&tid.to_be_bytes());
            ipma_body.push(1);
            ipma_body.push(2);
        }
        let ipma = make_full_box(b"ipma", 0, 0, &ipma_body);
        let mut iprp_body = Vec::new();
        iprp_body.extend_from_slice(&ipco);
        iprp_body.extend_from_slice(&ipma);
        let iprp = make_box(b"iprp", &iprp_body);

        let mut dimg_grid = Vec::new();
        dimg_grid.extend_from_slice(&1u16.to_be_bytes());
        dimg_grid.extend_from_slice(&4u16.to_be_bytes());
        for tid in 2u16..=5 {
            dimg_grid.extend_from_slice(&tid.to_be_bytes());
        }
        let iref = make_full_box(b"iref", 0, 0, &make_box(b"dimg", &dimg_grid));

        // 32-bit flags=1 ImageGrid descriptor (12 bayt)
        let mut grid_desc = vec![0u8, 1u8, 1u8, 1u8];
        grid_desc.extend_from_slice(&120u32.to_be_bytes());
        grid_desc.extend_from_slice(&110u32.to_be_bytes());

        let mut idat_payload = Vec::new();
        idat_payload.extend_from_slice(&grid_desc);
        let tile_off = idat_payload.len() as u32;
        idat_payload.extend_from_slice(&odd_obu);
        let idat = make_box(b"idat", &idat_payload);

        let mut iloc_body = Vec::new();
        iloc_body.push(0x44);
        iloc_body.push(0x00);
        iloc_body.extend_from_slice(&5u16.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&0u16.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&0u32.to_be_bytes());
        iloc_body.extend_from_slice(&(grid_desc.len() as u32).to_be_bytes());
        for tid in 2u16..=5 {
            iloc_body.extend_from_slice(&tid.to_be_bytes());
            iloc_body.extend_from_slice(&1u16.to_be_bytes());
            iloc_body.extend_from_slice(&0u16.to_be_bytes());
            iloc_body.extend_from_slice(&1u16.to_be_bytes());
            iloc_body.extend_from_slice(&tile_off.to_be_bytes());
            iloc_body.extend_from_slice(&(odd_obu.len() as u32).to_be_bytes());
        }
        let iloc = make_full_box(b"iloc", 1, 0, &iloc_body);

        let mut meta_children = Vec::new();
        for b in [&hdlr, &pitm, &iinf, &iprp, &iref, &iloc, &idat] {
            meta_children.extend_from_slice(b);
        }
        let meta = make_full_box(b"meta", 0, 0, &meta_children);
        let mut grid_file = Vec::new();
        grid_file.extend_from_slice(&ftyp);
        grid_file.extend_from_slice(&meta);

        let g_t1 = decode_with(
            &grid_file,
            &AvifDecodeConfig {
                n_threads: 1,
                ..AvifDecodeConfig::default()
            },
        )
        .unwrap();
        let g_t4 = decode_with(
            &grid_file,
            &AvifDecodeConfig {
                n_threads: 4,
                ..AvifDecodeConfig::default()
            },
        )
        .unwrap();
        assert_eq!((g_t1.width(), g_t1.height()), (120, 110));
        assert_eq!(
            g_t1, g_t4,
            "tek-thread ve çok-thread grid çıktısı bit-birebir aynı olmalı"
        );
    }
}

fn build_transformed_avif(
    obu: &[u8],
    w: u32,
    h: u32,
    clap: Option<CleanAperture>,
    irot: Option<u8>,
    imir: Option<u8>,
    unknown_essential: bool,
) -> Vec<u8> {
    let ftyp = make_ftyp(b"avif", &[*b"mif1"]);
    let hdlr = make_hdlr_pict();
    let pitm = make_full_box(b"pitm", 0, 0, &1u16.to_be_bytes());
    let mut iinf_body = Vec::new();
    iinf_body.extend_from_slice(&1u16.to_be_bytes());
    iinf_body.extend_from_slice(&make_infe_v2(1, b"av01", None));
    let iinf = make_full_box(b"iinf", 0, 0, &iinf_body);

    let mut ipco_body = Vec::new();
    ipco_body.extend_from_slice(&make_ispe(w, h));
    let mut prop_indices: Vec<(u8, bool)> = vec![(1, false)];
    let mut next_idx = 2u8;

    if let Some(c) = clap {
        let mut cb = Vec::with_capacity(32);
        cb.extend_from_slice(&c.width_n.to_be_bytes());
        cb.extend_from_slice(&c.width_d.to_be_bytes());
        cb.extend_from_slice(&c.height_n.to_be_bytes());
        cb.extend_from_slice(&c.height_d.to_be_bytes());
        cb.extend_from_slice(&c.horiz_off_n.to_be_bytes());
        cb.extend_from_slice(&c.horiz_off_d.to_be_bytes());
        cb.extend_from_slice(&c.vert_off_n.to_be_bytes());
        cb.extend_from_slice(&c.vert_off_d.to_be_bytes());
        ipco_body.extend_from_slice(&make_box(b"clap", &cb));
        prop_indices.push((next_idx, true));
        next_idx += 1;
    }
    if let Some(angle) = irot {
        ipco_body.extend_from_slice(&make_box(b"irot", &[angle]));
        prop_indices.push((next_idx, true));
        next_idx += 1;
    }
    if let Some(axis) = imir {
        ipco_body.extend_from_slice(&make_box(b"imir", &[axis]));
        prop_indices.push((next_idx, true));
        next_idx += 1;
    }
    if unknown_essential {
        ipco_body.extend_from_slice(&make_box(b"zzzz", &[1, 2, 3, 4]));
        prop_indices.push((next_idx, true));
    }

    let ipco = make_box(b"ipco", &ipco_body);
    let mut ipma_body = Vec::new();
    ipma_body.extend_from_slice(&1u32.to_be_bytes());
    ipma_body.extend_from_slice(&1u16.to_be_bytes());
    ipma_body.push(prop_indices.len() as u8);
    for (idx, ess) in prop_indices {
        ipma_body.push(if ess { 0x80 | idx } else { idx });
    }
    let ipma = make_full_box(b"ipma", 0, 0, &ipma_body);

    let mut iprp_body = Vec::new();
    iprp_body.extend_from_slice(&ipco);
    iprp_body.extend_from_slice(&ipma);
    let iprp = make_box(b"iprp", &iprp_body);

    // iloc v0: construction_method = 0 (dosya sonundaki mdat ofseti)
    // Önce iloc'u yer tutucu ofsetle oluşturup meta boyutuna göre mdat ofsetini hesapla.
    let make_iloc_v0 = |mdat_payload_offset: u32| -> Vec<u8> {
        let mut ib = Vec::new();
        ib.push(0x44); // offset_size=4, length_size=4
        ib.push(0x00);
        ib.extend_from_slice(&1u16.to_be_bytes());
        ib.extend_from_slice(&1u16.to_be_bytes());
        ib.extend_from_slice(&0u16.to_be_bytes());
        ib.extend_from_slice(&1u16.to_be_bytes());
        ib.extend_from_slice(&mdat_payload_offset.to_be_bytes());
        ib.extend_from_slice(&(obu.len() as u32).to_be_bytes());
        make_full_box(b"iloc", 0, 0, &ib)
    };

    let dummy_iloc = make_iloc_v0(0);
    let meta_len = 12 + hdlr.len() + pitm.len() + iinf.len() + iprp.len() + dummy_iloc.len();
    let mdat_payload_offset = (ftyp.len() + meta_len + 8) as u32;
    let iloc = make_iloc_v0(mdat_payload_offset);

    let mut meta_children = Vec::new();
    for b in [&hdlr, &pitm, &iinf, &iprp, &iloc] {
        meta_children.extend_from_slice(b);
    }
    let meta = make_full_box(b"meta", 0, 0, &meta_children);
    let mdat = make_box(b"mdat", obu);

    let mut file = Vec::new();
    file.extend_from_slice(&ftyp);
    file.extend_from_slice(&meta);
    file.extend_from_slice(&mdat);
    file
}

#[test]
fn avif_transforms_clap_irot_imir_and_budget_rejection() {
    let odd_obu = read_oddheight_keyframe_obu();
    let (base_w, base_h) = if odd_obu.len() > SAMPLE_1X1_AV1_OBU.len() {
        (64u32, 63u32)
    } else {
        (1u32, 1u32)
    };

    let base_avif = build_transformed_avif(&odd_obu, base_w, base_h, None, None, None, false);
    let base_img = decode(&base_avif).expect("temel AVIF çözülmeli");
    assert_eq!((base_img.width(), base_img.height()), (base_w, base_h));

    // 1. irot = 1 (90° CCW): boyut (base_h, base_w), dst(0, 0) == src(base_w - 1, 0)
    let rot90_avif = build_transformed_avif(&odd_obu, base_w, base_h, None, Some(1), None, false);
    let rot90_info = parse_info(&rot90_avif).unwrap();
    assert_eq!((rot90_info.width, rot90_info.height), (base_h, base_w));
    let rot90_img = decode(&rot90_avif).unwrap();
    assert_eq!((rot90_img.width(), rot90_img.height()), (base_h, base_w));
    assert_eq!(
        rot90_img.y_row_u8(0).unwrap()[0],
        base_img.y_row_u8(0).unwrap()[(base_w - 1) as usize]
    );

    // 2. irot = 2 (180° CCW): boyut (base_w, base_h), dst(0, 0) == src(base_w - 1, base_h - 1)
    let rot180_avif = build_transformed_avif(&odd_obu, base_w, base_h, None, Some(2), None, false);
    let rot180_img = decode(&rot180_avif).unwrap();
    assert_eq!(
        rot180_img.y_row_u8(0).unwrap()[0],
        base_img.y_row_u8((base_h - 1) as usize).unwrap()[(base_w - 1) as usize]
    );

    // 3. irot = 3 (270° CCW): boyut (base_h, base_w), dst(0, 0) == src(0, base_h - 1)
    let rot270_avif = build_transformed_avif(&odd_obu, base_w, base_h, None, Some(3), None, false);
    let rot270_img = decode(&rot270_avif).unwrap();
    assert_eq!(
        rot270_img.y_row_u8(0).unwrap()[0],
        base_img.y_row_u8((base_h - 1) as usize).unwrap()[0]
    );

    // 4. imir axis = 0 (üst-alt) ve axis = 1 (sol-sağ)
    let imir0_avif = build_transformed_avif(&odd_obu, base_w, base_h, None, None, Some(0), false);
    let imir0_img = decode(&imir0_avif).unwrap();
    assert_eq!(
        imir0_img.y_row_u8(0).unwrap()[0],
        base_img.y_row_u8((base_h - 1) as usize).unwrap()[0]
    );

    let imir1_avif = build_transformed_avif(&odd_obu, base_w, base_h, None, None, Some(1), false);
    let imir1_img = decode(&imir1_avif).unwrap();
    assert_eq!(
        imir1_img.y_row_u8(0).unwrap()[0],
        base_img.y_row_u8(0).unwrap()[(base_w - 1) as usize]
    );

    // 5. clap kırpma (merkez 32x31 veya 1x1)
    let crop_w = (base_w / 2).max(1);
    let crop_h = (base_h / 2).max(1);
    let clap = CleanAperture {
        width_n: crop_w,
        width_d: 1,
        height_n: crop_h,
        height_d: 1,
        horiz_off_n: 0,
        horiz_off_d: 1,
        vert_off_n: 0,
        vert_off_d: 1,
    };
    let clap_avif = build_transformed_avif(&odd_obu, base_w, base_h, Some(clap), None, None, false);
    let clap_info = parse_info(&clap_avif).unwrap();
    assert_eq!((clap_info.width, clap_info.height), (crop_w, crop_h));
    let clap_img = decode(&clap_avif).unwrap();
    assert_eq!((clap_img.width(), clap_img.height()), (crop_w, crop_h));

    // 6. Bilinmeyen zorunlu (essential) ipco özelliği reddedilmeli
    let unk_avif = build_transformed_avif(&odd_obu, base_w, base_h, None, None, None, true);
    assert!(matches!(decode(&unk_avif), Err(Error::Unsupported(_))));

    // 7. Bütçe aşımı (max_pixels ve allocation_limit)
    if base_w > 1 {
        let res_px = decode_with(
            &base_avif,
            &AvifDecodeConfig {
                max_pixels: 16,
                ..AvifDecodeConfig::default()
            },
        );
        assert!(matches!(res_px, Err(Error::BudgetExceeded { .. })));
    }
    let res_mem = decode_with(
        &base_avif,
        &AvifDecodeConfig {
            allocation_limit: Some(8),
            ..AvifDecodeConfig::default()
        },
    );
    assert!(matches!(res_mem, Err(Error::BudgetExceeded { .. })));
}

#[test]
fn avif_ffmpeg_cross_validation_if_available() {
    let Ok(out) = Command::new("ffmpeg").arg("-version").output() else {
        return;
    };
    if !out.status.success() {
        return;
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let work_dir = workspace_root.join(".cache").join("avif_ffmpeg_test");
    let _ = fs::create_dir_all(&work_dir);
    let avif_path = work_dir.join("sample_ffmpeg.avif");

    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=32x32:d=0.04",
            "-frames:v",
            "1",
            "-c:v",
            "libaom-av1",
            "-still-picture",
            "1",
        ])
        .arg(&avif_path)
        .status();

    if let Ok(st) = status
        && st.success()
        && let Ok(bytes) = fs::read(&avif_path)
    {
        assert!(probe(&bytes));
        let info = parse_info(&bytes).expect("ffmpeg AVIF parse_info başarılı olmalı");
        assert_eq!((info.width, info.height), (32, 32));
        let rgb = decode_rgb8(&bytes).expect("ffmpeg AVIF decode_rgb8 başarılı olmalı");
        assert_eq!((rgb.width, rgb.height), (32, 32));
        assert_eq!(rgb.pixels.len(), 32 * 32 * 3);
        let rgba = decode_rgba8(&bytes).expect("ffmpeg AVIF decode_rgba8 başarılı olmalı");
        assert_eq!(rgba.pixels.len(), 32 * 32 * 4);
    }

    let _ = fs::remove_dir_all(&work_dir);
}

fn make_infe_v2_with_encoding(
    item_id: u16,
    item_type: &[u8; 4],
    content_type: Option<&str>,
    content_encoding: Option<&str>,
) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&item_id.to_be_bytes());
    body.extend_from_slice(&0u16.to_be_bytes());
    body.extend_from_slice(item_type);
    body.push(0); // item_name = ""
    if let Some(ct) = content_type {
        body.extend_from_slice(ct.as_bytes());
        body.push(0);
        if let Some(ce) = content_encoding {
            body.extend_from_slice(ce.as_bytes());
            body.push(0);
        }
    }
    make_full_box(b"infe", 2, 0, &body)
}

fn first_ivf_packet_00001105() -> (Vec<u8>, u32, u32) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/streams/00001105.ivf");
    let raw = std::fs::read(&path).expect("00001105.ivf okunmalı");
    assert!(raw.len() > 44 && &raw[..4] == b"DKIF");
    let w = u16::from_le_bytes([raw[12], raw[13]]) as u32;
    let h = u16::from_le_bytes([raw[14], raw[15]]) as u32;
    let sz = u32::from_le_bytes([raw[32], raw[33], raw[34], raw[35]]) as usize;
    (raw[44..44 + sz].to_vec(), w, h)
}

fn build_compressed_meta_avif(
    xmp_bytes: &[u8],
    xmp_encoding: Option<&str>,
    exif_bytes: &[u8],
    exif_cmpc: Option<&[u8; 4]>,
    use_cmin: bool,
) -> Vec<u8> {
    build_compressed_meta_avif_with_frame(
        SAMPLE_1X1_AV1_OBU,
        1,
        1,
        xmp_bytes,
        xmp_encoding,
        exif_bytes,
        exif_cmpc,
        use_cmin,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_compressed_meta_avif_with_frame(
    frame_obu: &[u8],
    width: u32,
    height: u32,
    xmp_bytes: &[u8],
    xmp_encoding: Option<&str>,
    exif_bytes: &[u8],
    exif_cmpc: Option<&[u8; 4]>,
    use_cmin: bool,
) -> Vec<u8> {
    let ftyp = make_ftyp(b"avif", &[*b"mif1", *b"miaf"]);
    let hdlr = make_hdlr_pict();
    let pitm = make_full_box(b"pitm", 0, 0, &1u16.to_be_bytes());

    let infe1 = make_infe_v2(1, b"av01", None);
    let infe2 = make_infe_v2(2, b"Exif", None);
    let infe3 = make_infe_v2_with_encoding(3, b"mime", Some("application/rdf+xml"), xmp_encoding);
    let mut iinf_body = Vec::new();
    iinf_body.extend_from_slice(&3u16.to_be_bytes());
    iinf_body.extend_from_slice(&infe1);
    iinf_body.extend_from_slice(&infe2);
    iinf_body.extend_from_slice(&infe3);
    let iinf = make_full_box(b"iinf", 0, 0, &iinf_body);

    let ispe = make_ispe(width, height);
    let av1c = make_box(b"av1C", &[0x81, 0x00, 0x0C, 0x00]);
    let mut ipco_body = Vec::new();
    ipco_body.extend_from_slice(&ispe);
    ipco_body.extend_from_slice(&av1c);
    if let Some(comp_fourcc) = exif_cmpc {
        let cmpc = make_full_box(b"cmpC", 0, 0, comp_fourcc);
        ipco_body.extend_from_slice(&cmpc);
    }
    let ipco = make_box(b"ipco", &ipco_body);

    let mut ipma_body = Vec::new();
    let entry_count: u32 = if exif_cmpc.is_some() { 2 } else { 1 };
    ipma_body.extend_from_slice(&entry_count.to_be_bytes());
    ipma_body.extend_from_slice(&1u16.to_be_bytes());
    ipma_body.push(2);
    ipma_body.push(0x81); // prop 1 essential
    ipma_body.push(0x82); // prop 2 essential
    if exif_cmpc.is_some() {
        ipma_body.extend_from_slice(&2u16.to_be_bytes());
        ipma_body.push(1);
        ipma_body.push(0x83); // prop 3 (cmpC) essential
    }
    let ipma = make_full_box(b"ipma", 0, 0, &ipma_body);

    let mut iprp_body = Vec::new();
    iprp_body.extend_from_slice(&ipco);
    iprp_body.extend_from_slice(&ipma);
    let iprp = make_box(b"iprp", &iprp_body);

    let mut cdsc_exif = Vec::new();
    cdsc_exif.extend_from_slice(&2u16.to_be_bytes());
    cdsc_exif.extend_from_slice(&1u16.to_be_bytes());
    cdsc_exif.extend_from_slice(&1u16.to_be_bytes());
    let mut cdsc_xmp = Vec::new();
    cdsc_xmp.extend_from_slice(&3u16.to_be_bytes());
    cdsc_xmp.extend_from_slice(&1u16.to_be_bytes());
    cdsc_xmp.extend_from_slice(&1u16.to_be_bytes());
    let mut iref_body = Vec::new();
    iref_body.extend_from_slice(&make_box(b"cdsc", &cdsc_exif));
    iref_body.extend_from_slice(&make_box(b"cdsc", &cdsc_xmp));
    let iref = make_full_box(b"iref", 0, 0, &iref_body);

    let mut idat_payload = Vec::new();
    idat_payload.extend_from_slice(frame_obu);
    let exif_off = idat_payload.len() as u32;
    idat_payload.extend_from_slice(exif_bytes);
    let xmp_off = idat_payload.len() as u32;
    idat_payload.extend_from_slice(xmp_bytes);
    let idat = make_box(b"idat", &idat_payload);

    let mut iloc_body = Vec::new();
    iloc_body.push(0x44);
    iloc_body.push(0x00);
    iloc_body.extend_from_slice(&3u16.to_be_bytes());
    for (id, off, len) in [
        (1u16, 0u32, frame_obu.len() as u32),
        (2u16, exif_off, exif_bytes.len() as u32),
        (3u16, xmp_off, xmp_bytes.len() as u32),
    ] {
        iloc_body.extend_from_slice(&id.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes()); // construction_method = 1 (idat)
        iloc_body.extend_from_slice(&0u16.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&off.to_be_bytes());
        iloc_body.extend_from_slice(&len.to_be_bytes());
    }
    let iloc = make_full_box(b"iloc", 1, 0, &iloc_body);

    let mut meta_children = Vec::new();
    for b in [&hdlr, &pitm, &iinf, &iprp, &iref, &iloc, &idat] {
        meta_children.extend_from_slice(b);
    }
    let meta = make_full_box(b"meta", 0, 0, &meta_children);

    let mut file = Vec::new();
    file.extend_from_slice(&ftyp);
    if use_cmin {
        use std::io::Write as _;
        let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(&meta[8..]).unwrap();
        let comp_meta = enc.finish().unwrap();
        let mut cmin_body = Vec::with_capacity(4 + comp_meta.len());
        cmin_body.extend_from_slice(b"zlib");
        cmin_body.extend_from_slice(&comp_meta);
        file.extend_from_slice(&make_full_box(b"cmin", 0, 0, &cmin_body));
    } else {
        file.extend_from_slice(&meta);
    }
    file
}

#[test]
fn avif_compressed_heif_metadata_cmin_and_security_limits() {
    use flate2::Compression;
    use flate2::write::{DeflateEncoder, GzEncoder, ZlibEncoder};
    use std::io::Write as _;

    let orig_xmp = b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">COMPRESSED_HEIF_XMP</x:xmpmeta>";
    let orig_exif_block = b"\x00\x00\x00\x00II*\x00EXIF_COMPRESSED_TIFF_BODY";

    // 1. content_encoding = "deflate" (RFC 1950 Zlib) + cmpC = "defl" (RFC 1951 raw deflate) + cmin
    let mut zlib_enc = ZlibEncoder::new(Vec::new(), Compression::default());
    zlib_enc.write_all(orig_xmp).unwrap();
    let xmp_zlib = zlib_enc.finish().unwrap();

    let mut defl_enc = DeflateEncoder::new(Vec::new(), Compression::default());
    defl_enc.write_all(orig_exif_block).unwrap();
    let exif_defl = defl_enc.finish().unwrap();

    let avif_zlib_defl =
        build_compressed_meta_avif(&xmp_zlib, Some("deflate"), &exif_defl, Some(b"defl"), true);
    let img1 = decode(&avif_zlib_defl).expect("cmin + deflate XMP + cmpC defl Exif çözülmeli");
    assert_eq!(img1.xmp(), Some(orig_xmp.as_slice()));
    assert_eq!(img1.exif(), Some(&orig_exif_block[4..]));

    // 2. content_encoding = "gzip" (RFC 1952) + cmpC = "zlib" (RFC 1950)
    let mut gz_enc = GzEncoder::new(Vec::new(), Compression::default());
    gz_enc.write_all(orig_xmp).unwrap();
    let xmp_gz = gz_enc.finish().unwrap();

    let mut exif_z_enc = ZlibEncoder::new(Vec::new(), Compression::default());
    exif_z_enc.write_all(orig_exif_block).unwrap();
    let exif_zlib = exif_z_enc.finish().unwrap();

    let avif_gz_zlib =
        build_compressed_meta_avif(&xmp_gz, Some("gzip"), &exif_zlib, Some(b"zlib"), false);
    let info2 = parse_info(&avif_gz_zlib).expect("gzip XMP + cmpC zlib Exif parse_info");
    assert_eq!(info2.xmp.as_deref(), Some(orig_xmp.as_slice()));
    assert_eq!(info2.exif.as_deref(), Some(&orig_exif_block[4..]));

    // 3. Bölüm 10.3 + 5.A.1: Ortak Arc<Butce> altında canlı düzlemler (1x1 YUV420 = 3 bayt) +
    //    Exif (27 bayt) + sıkıştırılmış XMP (600_000 bayt) toplamının tam sınırında kabul vs
    //    limit - 1 / XMP + 1 reddi.
    let exact_xmp_len = 600_000usize;
    let live_planes_and_exif = 3usize + (orig_exif_block.len() - 4);
    let payload_exact: Vec<u8> = (0..exact_xmp_len)
        .map(|i| ((i.wrapping_mul(73) ^ (i >> 5)) & 0xFF) as u8)
        .collect();
    let mut enc_exact = ZlibEncoder::new(Vec::new(), Compression::fast());
    enc_exact.write_all(&payload_exact).unwrap();
    let comp_exact = enc_exact.finish().unwrap();
    let avif_exact =
        build_compressed_meta_avif(&comp_exact, Some("deflate"), orig_exif_block, None, false);
    let container_overhead =
        AvifPrepared::from_vec(avif_exact.clone(), &AvifDecodeConfig::default())
            .unwrap()
            .held_bytes()
            - avif_exact.len()
            - exact_xmp_len
            - (orig_exif_block.len() - 4);
    const FLATE2_DECOMPRESS_STATE_BYTES: usize = 48 * 1024;
    let exact_total_limit =
        exact_xmp_len + live_planes_and_exif + container_overhead + FLATE2_DECOMPRESS_STATE_BYTES;
    let img_exact = decode_with(
        &avif_exact,
        &AvifDecodeConfig {
            allocation_limit: Some(exact_total_limit),
            ..AvifDecodeConfig::default()
        },
    )
    .expect("tam canlı toplam allocation_limit sınırındaki sıkıştırılmış öğe kabul edilmeli");
    assert_eq!(img_exact.xmp(), Some(payload_exact.as_slice()));

    let res_under_by_one = decode_with(
        &avif_exact,
        &AvifDecodeConfig {
            allocation_limit: Some(exact_total_limit - 1),
            ..AvifDecodeConfig::default()
        },
    );
    assert_eq!(
        res_under_by_one.unwrap_err(),
        Error::BudgetExceeded {
            requested: exact_total_limit,
            limit: exact_total_limit - 1,
        }
    );

    let mut payload_plus_one = payload_exact;
    payload_plus_one.push(0x42);
    let mut enc_over = ZlibEncoder::new(Vec::new(), Compression::fast());
    enc_over.write_all(&payload_plus_one).unwrap();
    let comp_over = enc_over.finish().unwrap();
    let avif_over =
        build_compressed_meta_avif(&comp_over, Some("deflate"), orig_exif_block, None, false);
    let res_over = decode_with(
        &avif_over,
        &AvifDecodeConfig {
            allocation_limit: Some(exact_total_limit),
            ..AvifDecodeConfig::default()
        },
    );
    assert_eq!(
        res_over.unwrap_err(),
        Error::BudgetExceeded {
            requested: exact_total_limit + 1,
            limit: exact_total_limit,
        }
    );

    // 4. Bölüm 10.3: Yüksek oranlı (> 1024:1 ve > 64 KiB) zip-bombası reddi
    let zip_bomb_uncompressed = vec![0u8; 300 * 1024];
    let mut enc_bomb_inner = DeflateEncoder::new(Vec::new(), Compression::best());
    enc_bomb_inner.write_all(&zip_bomb_uncompressed).unwrap();
    let bomb_stage1 = enc_bomb_inner.finish().unwrap();
    let mut enc_bomb_outer = ZlibEncoder::new(Vec::new(), Compression::best());
    enc_bomb_outer.write_all(&bomb_stage1).unwrap();
    let comp_bomb = enc_bomb_outer.finish().unwrap();
    let avif_bomb = build_compressed_meta_avif(orig_xmp, None, &comp_bomb, Some(b"zlib"), false);
    // Aynı Exif öğesine ikinci cmpC ("defl") özelliğini bağlayan veya XMP öğesine
    // content_encoding="deflate" + cmpC="defl" bağlayan >1024:1 zip-bombası:
    let ftyp = make_ftyp(b"avif", &[*b"mif1", *b"miaf"]);
    let hdlr = make_hdlr_pict();
    let pitm = make_full_box(b"pitm", 0, 0, &1u16.to_be_bytes());
    let infe1 = make_infe_v2(1, b"av01", None);
    let infe3 =
        make_infe_v2_with_encoding(3, b"mime", Some("application/rdf+xml"), Some("deflate"));
    let mut iinf_body = Vec::new();
    iinf_body.extend_from_slice(&2u16.to_be_bytes());
    iinf_body.extend_from_slice(&infe1);
    iinf_body.extend_from_slice(&infe3);
    let iinf = make_full_box(b"iinf", 0, 0, &iinf_body);
    let ispe = make_ispe(1, 1);
    let av1c = make_box(b"av1C", &[0x81, 0x00, 0x0C, 0x00]);
    let cmpc = make_full_box(b"cmpC", 0, 0, b"defl");
    let mut ipco_body = Vec::new();
    ipco_body.extend_from_slice(&ispe);
    ipco_body.extend_from_slice(&av1c);
    ipco_body.extend_from_slice(&cmpc);
    let ipco = make_box(b"ipco", &ipco_body);
    let mut ipma_body = Vec::new();
    ipma_body.extend_from_slice(&2u32.to_be_bytes());
    ipma_body.extend_from_slice(&1u16.to_be_bytes());
    ipma_body.push(2);
    ipma_body.push(0x81);
    ipma_body.push(0x82);
    ipma_body.extend_from_slice(&3u16.to_be_bytes());
    ipma_body.push(1);
    ipma_body.push(0x83);
    let ipma = make_full_box(b"ipma", 0, 0, &ipma_body);
    let mut iprp_body = Vec::new();
    iprp_body.extend_from_slice(&ipco);
    iprp_body.extend_from_slice(&ipma);
    let iprp = make_box(b"iprp", &iprp_body);
    let mut cdsc_xmp = Vec::new();
    cdsc_xmp.extend_from_slice(&3u16.to_be_bytes());
    cdsc_xmp.extend_from_slice(&1u16.to_be_bytes());
    cdsc_xmp.extend_from_slice(&1u16.to_be_bytes());
    let iref = make_full_box(b"iref", 0, 0, &make_box(b"cdsc", &cdsc_xmp));
    let mut idat_payload = Vec::new();
    idat_payload.extend_from_slice(SAMPLE_1X1_AV1_OBU);
    let bomb_off = idat_payload.len() as u32;
    idat_payload.extend_from_slice(&comp_bomb);
    let idat = make_box(b"idat", &idat_payload);
    let mut iloc_body = Vec::new();
    iloc_body.push(0x44);
    iloc_body.push(0x00);
    iloc_body.extend_from_slice(&2u16.to_be_bytes());
    for (id, off, len) in [
        (1u16, 0u32, SAMPLE_1X1_AV1_OBU.len() as u32),
        (3u16, bomb_off, comp_bomb.len() as u32),
    ] {
        iloc_body.extend_from_slice(&id.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&0u16.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&off.to_be_bytes());
        iloc_body.extend_from_slice(&len.to_be_bytes());
    }
    let iloc = make_full_box(b"iloc", 1, 0, &iloc_body);
    let mut meta_children = Vec::new();
    for b in [&hdlr, &pitm, &iinf, &iprp, &iref, &iloc, &idat] {
        meta_children.extend_from_slice(b);
    }
    let mut avif_layered_bomb = Vec::new();
    avif_layered_bomb.extend_from_slice(&ftyp);
    avif_layered_bomb.extend_from_slice(&make_full_box(b"meta", 0, 0, &meta_children));
    let _ = avif_bomb;
    assert!(matches!(
        parse_info(&avif_layered_bomb),
        Err(Error::BudgetExceeded { .. })
    ));

    // 5. Bozuk Adler-32 kuyruğu -> Error::InvalidData
    let mut corrupted_zlib = xmp_zlib;
    let last_idx = corrupted_zlib.len() - 1;
    corrupted_zlib[last_idx] ^= 0xFF;
    let avif_corrupt = build_compressed_meta_avif(
        &corrupted_zlib,
        Some("deflate"),
        orig_exif_block,
        None,
        false,
    );
    assert!(matches!(
        parse_info(&avif_corrupt),
        Err(Error::InvalidData(_))
    ));
}

#[allow(clippy::too_many_arguments)]
fn build_tmap_gainmap_avif(
    include_tmap_brand: bool,
    altr_order: &[u32],
    tmap_version: u8,
    min_version: u16,
    writer_version: u16,
    gamma_n: u32,
    extra_trailing: &[u8],
    colr_mtrx: u16,
    colr_trc: u16,
) -> Vec<u8> {
    let mut compat = vec![*b"avif", *b"miaf"];
    if include_tmap_brand {
        compat.push(*b"tmap");
    }
    let ftyp = make_ftyp(b"mif1", &compat);
    let hdlr = make_hdlr_pict();
    let pitm = make_full_box(b"pitm", 0, 0, &1u16.to_be_bytes());

    // Öğeler: 1 = taban (av01), 2 = kazanç haritası (av01), 3 = tmap
    let mut iinf_body = Vec::new();
    iinf_body.extend_from_slice(&3u16.to_be_bytes());
    iinf_body.extend_from_slice(&make_infe_v2(1, b"av01", None));
    iinf_body.extend_from_slice(&make_infe_v2(2, b"av01", None));
    iinf_body.extend_from_slice(&make_infe_v2(3, b"tmap", None));
    let iinf = make_full_box(b"iinf", 0, 0, &iinf_body);

    // ipco:
    // 1: ispe (1x1)
    // 2: av1C
    // 3: colr nclx (taban görüntü)
    // 4: colr nclx (alternatif HDR: BT.2020 + PQ + BT.2020 NCL)
    // 5: colr rICC (alternatif ICC)
    // 6: clli (1000, 400)
    // 7: pixi (10, 10, 10)
    let ispe = make_ispe(1, 1);
    let av1c = make_box(b"av1C", &[0x81, 0x00, 0x0C, 0x00]);
    let colr_base = make_box(
        b"colr",
        &[
            b'n',
            b'c',
            b'l',
            b'x',
            0,
            9,
            (colr_trc >> 8) as u8,
            (colr_trc & 0xFF) as u8,
            (colr_mtrx >> 8) as u8,
            (colr_mtrx & 0xFF) as u8,
            0x80,
        ],
    );
    let colr_alt = make_box(b"colr", &[b'n', b'c', b'l', b'x', 0, 9, 0, 16, 0, 9, 0x80]);
    let mut colr_alt_icc = b"rICC".to_vec();
    colr_alt_icc.extend_from_slice(b"ALT_HDR_ICC");
    let colr_icc_box = make_box(b"colr", &colr_alt_icc);
    let clli = make_box(b"clli", &[0x03, 0xE8, 0x01, 0x90]);
    let pixi_alt = make_full_box(b"pixi", 0, 0, &[3, 10, 10, 10]);

    let mut ipco_body = Vec::new();
    for p in [
        &ispe,
        &av1c,
        &colr_base,
        &colr_alt,
        &colr_icc_box,
        &clli,
        &pixi_alt,
    ] {
        ipco_body.extend_from_slice(p);
    }
    let ipco = make_box(b"ipco", &ipco_body);

    let mut ipma_body = Vec::new();
    ipma_body.extend_from_slice(&3u32.to_be_bytes());
    // item 1 (base) -> 1, 2, 3
    ipma_body.extend_from_slice(&1u16.to_be_bytes());
    ipma_body.push(3);
    ipma_body.extend_from_slice(&[1, 0x82, 3]);
    // item 2 (gain map) -> 1, 2
    ipma_body.extend_from_slice(&2u16.to_be_bytes());
    ipma_body.push(2);
    ipma_body.extend_from_slice(&[1, 0x82]);
    // item 3 (tmap) -> 1 (ispe), 4 (alt nclx), 5 (alt icc), 6 (clli), 7 (pixi)
    ipma_body.extend_from_slice(&3u16.to_be_bytes());
    ipma_body.push(5);
    ipma_body.extend_from_slice(&[1, 4, 5, 6, 7]);
    let ipma = make_full_box(b"ipma", 0, 0, &ipma_body);

    let mut iprp_body = Vec::new();
    iprp_body.extend_from_slice(&ipco);
    iprp_body.extend_from_slice(&ipma);
    let iprp = make_box(b"iprp", &iprp_body);

    // iref: dimg 3 -> [1, 2]
    let mut dimg_body = Vec::new();
    dimg_body.extend_from_slice(&3u16.to_be_bytes());
    dimg_body.extend_from_slice(&2u16.to_be_bytes());
    dimg_body.extend_from_slice(&1u16.to_be_bytes());
    dimg_body.extend_from_slice(&2u16.to_be_bytes());
    let iref = make_full_box(b"iref", 0, 0, &make_box(b"dimg", &dimg_body));

    // grpl -> altr
    let mut altr_body = Vec::new();
    altr_body.extend_from_slice(&100u32.to_be_bytes()); // group_id
    altr_body.extend_from_slice(&(altr_order.len() as u32).to_be_bytes());
    for &eid in altr_order {
        altr_body.extend_from_slice(&eid.to_be_bytes());
    }
    let grpl = make_box(b"grpl", &make_full_box(b"altr", 0, 0, &altr_body));

    // ISO 21496-1 tmap ikili yükü (tek kanallı)
    let mut tmap_payload = Vec::new();
    tmap_payload.push(tmap_version);
    tmap_payload.extend_from_slice(&min_version.to_be_bytes());
    tmap_payload.extend_from_slice(&writer_version.to_be_bytes());
    tmap_payload.push(0x40); // is_multichannel=0, use_base_color_space=1
    // base_hdr_headroom = 0/1, alternate_hdr_headroom = 2/1
    tmap_payload.extend_from_slice(&0u32.to_be_bytes());
    tmap_payload.extend_from_slice(&1u32.to_be_bytes());
    tmap_payload.extend_from_slice(&2u32.to_be_bytes());
    tmap_payload.extend_from_slice(&1u32.to_be_bytes());
    // min=-1/1, max=3/1, gamma=gamma_n/1, base_off=1/64, alt_off=1/64
    tmap_payload.extend_from_slice(&(-1i32).to_be_bytes());
    tmap_payload.extend_from_slice(&1u32.to_be_bytes());
    tmap_payload.extend_from_slice(&3i32.to_be_bytes());
    tmap_payload.extend_from_slice(&1u32.to_be_bytes());
    tmap_payload.extend_from_slice(&gamma_n.to_be_bytes());
    tmap_payload.extend_from_slice(&1u32.to_be_bytes());
    tmap_payload.extend_from_slice(&1i32.to_be_bytes());
    tmap_payload.extend_from_slice(&64u32.to_be_bytes());
    tmap_payload.extend_from_slice(&1i32.to_be_bytes());
    tmap_payload.extend_from_slice(&64u32.to_be_bytes());
    tmap_payload.extend_from_slice(extra_trailing);

    let mut idat_payload = Vec::new();
    idat_payload.extend_from_slice(SAMPLE_1X1_AV1_OBU);
    let tmap_off = idat_payload.len() as u32;
    idat_payload.extend_from_slice(&tmap_payload);
    let idat = make_box(b"idat", &idat_payload);

    let mut iloc_body = Vec::new();
    iloc_body.push(0x44);
    iloc_body.push(0x00);
    iloc_body.extend_from_slice(&3u16.to_be_bytes());
    for (id, off, len) in [
        (1u16, 0u32, SAMPLE_1X1_AV1_OBU.len() as u32),
        (2u16, 0u32, SAMPLE_1X1_AV1_OBU.len() as u32),
        (3u16, tmap_off, tmap_payload.len() as u32),
    ] {
        iloc_body.extend_from_slice(&id.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&0u16.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&off.to_be_bytes());
        iloc_body.extend_from_slice(&len.to_be_bytes());
    }
    let iloc = make_full_box(b"iloc", 1, 0, &iloc_body);

    let mut meta_children = Vec::new();
    for b in [&hdlr, &pitm, &iinf, &iprp, &iref, &grpl, &iloc, &idat] {
        meta_children.extend_from_slice(b);
    }
    let meta = make_full_box(b"meta", 0, 0, &meta_children);

    let mut file = Vec::new();
    file.extend_from_slice(&ftyp);
    file.extend_from_slice(&meta);
    file
}

#[test]
fn avif_tmap_gain_map_and_h273_nonlinear_matrices() {
    // 1. Geçerli tmap: parse_info, decode_gain_map=false ve decode_gain_map=true
    let valid_tmap = build_tmap_gainmap_avif(true, &[3, 1], 0, 0, 0, 1, &[], 1, 13);
    let info = parse_info(&valid_tmap).expect("geçerli tmap parse_info");
    assert!(info.has_gain_map);
    let gm_meta = info
        .gain_map_metadata
        .as_ref()
        .expect("gain_map_metadata mevcut olmalı");
    assert!(gm_meta.use_base_color_space);
    assert_eq!(gm_meta.alternate_hdr_headroom.numerator, 2);
    assert_eq!(gm_meta.alt_depth, 10);
    assert_eq!(gm_meta.alt_plane_count, 3);
    assert_eq!(
        gm_meta.alt_icc_profile.as_deref(),
        Some(&b"ALT_HDR_ICC"[..])
    );
    assert_eq!(
        gm_meta.alt_clli,
        Some(ContentLightLevel {
            max_content_light_level: 1000,
            max_pic_average_light_level: 400,
        })
    );

    let img_no_gm = decode(&valid_tmap).expect("decode_gain_map=false çözülmeli");
    assert!(img_no_gm.has_gain_map());
    assert!(img_no_gm.gain_map_metadata().is_some());
    assert!(img_no_gm.gain_map().is_none());

    let img_with_gm = decode_with(
        &valid_tmap,
        &AvifDecodeConfig {
            decode_gain_map: true,
            ..AvifDecodeConfig::default()
        },
    )
    .expect("decode_gain_map=true çözülmeli");
    let gm = img_with_gm.gain_map().expect("kazanç haritası çözülmeli");
    assert_eq!((gm.image.width(), gm.image.height()), (1, 1));
    assert_eq!(gm.image.y_plane_u8(), Some(&[190u8][..]));

    // 2. tmap markası eksikse veya altr sırası tersse yoksayılır (libavif uyumu)
    let no_brand = build_tmap_gainmap_avif(false, &[3, 1], 0, 0, 0, 1, &[], 1, 13);
    assert!(!parse_info(&no_brand).unwrap().has_gain_map);
    let wrong_altr = build_tmap_gainmap_avif(true, &[1, 3], 0, 0, 0, 1, &[], 1, 13);
    assert!(!parse_info(&wrong_altr).unwrap().has_gain_map);

    // 3. Desteklenmeyen version / minimum_version -> yoksayılır; writer_version > 0 fazladan bayt -> kabul
    let unsup_ver = build_tmap_gainmap_avif(true, &[3, 1], 1, 0, 0, 1, &[], 1, 13);
    assert!(!parse_info(&unsup_ver).unwrap().has_gain_map);
    let unsup_min = build_tmap_gainmap_avif(true, &[3, 1], 0, 1, 1, 1, &[], 1, 13);
    assert!(!parse_info(&unsup_min).unwrap().has_gain_map);
    let forward_extra = build_tmap_gainmap_avif(true, &[3, 1], 0, 0, 1, 1, &[0xAA, 0xBB], 1, 13);
    assert!(parse_info(&forward_extra).unwrap().has_gain_map);

    // 4. writer_version == 0 + fazladan bayt ve gamma == 0 -> reddedilmeli
    let bad_extra = build_tmap_gainmap_avif(true, &[3, 1], 0, 0, 0, 1, &[0x00], 1, 13);
    assert!(matches!(decode(&bad_extra), Err(Error::InvalidData(_))));
    let gamma_zero = build_tmap_gainmap_avif(true, &[3, 1], 0, 0, 0, 0, &[], 1, 13);
    assert!(matches!(decode(&gamma_zero), Err(Error::InvalidData(_))));

    // 5. Doğrusal olmayan ITU-T H.273 matrisleri: ICtCp (14, PQ=16 ve HLG=18) ve BT.2020 CL (10)
    for (mtrx, trc) in [
        (14u16, 16u16),
        (14u16, 18u16),
        (10u16, 14u16),
        (10u16, 16u16),
    ] {
        let avif_hdr = build_tmap_gainmap_avif(false, &[], 0, 0, 0, 1, &[], mtrx, trc);
        let img = decode(&avif_hdr).expect("HDR matrisli AVIF çözülmeli");
        assert_eq!(img.color().matrix_coefficients, mtrx);
        assert_eq!(img.to_rgb8().unwrap().pixels.len(), 3);
        assert_eq!(img.to_rgba8().unwrap().pixels.len(), 4);
        assert_eq!(img.to_rgb16().unwrap().pixels.len(), 3);
        assert_eq!(img.to_rgba16().unwrap().pixels.len(), 4);
    }
}

fn build_synthetic_avis_with_alpha(
    samples: &[&[u8]],
    alpha_samples: Option<&[&[u8]]>,
    infinite_loop: bool,
) -> Vec<u8> {
    let ftyp = make_ftyp(b"avis", &[*b"msf1", *b"avif", *b"miaf"]);

    // mvhd v0 (96 bayt gövde)
    let mut mvhd_body = vec![0u8; 96];
    mvhd_body[8..12].copy_from_slice(&30u32.to_be_bytes()); // timescale = 30
    mvhd_body[12..16].copy_from_slice(&(samples.len() as u32).to_be_bytes());
    let mvhd = make_full_box(b"mvhd", 0, 0, &mvhd_body);

    let make_track_box = |track_id: u32,
                          handler: &[u8; 4],
                          auxl_target: Option<u32>,
                          sample_sizes: &[u32],
                          chunk_offset: u32,
                          keyframes_1based: &[u32]|
     -> Vec<u8> {
        // tkhd v0: 80 bayt gövde (20 + 52 + 8) + standart birim matris
        let mut tkhd_body = vec![0u8; 80];
        tkhd_body[8..12].copy_from_slice(&track_id.to_be_bytes());
        let track_dur = if infinite_loop {
            u32::MAX
        } else {
            (sample_sizes.len() as u32) * 3 // 3 oynatma = 2 ek tekrar
        };
        tkhd_body[16..20].copy_from_slice(&track_dur.to_be_bytes());
        tkhd_body[36..40].copy_from_slice(&0x0001_0000i32.to_be_bytes()); // a = 1.0
        tkhd_body[52..56].copy_from_slice(&0x0001_0000i32.to_be_bytes()); // d = 1.0
        tkhd_body[68..72].copy_from_slice(&0x4000_0000i32.to_be_bytes()); // w = 1.0 (2.30)
        // width = 1 << 16, height = 1 << 16
        tkhd_body[72..76].copy_from_slice(&(1u32 << 16).to_be_bytes());
        tkhd_body[76..80].copy_from_slice(&(1u32 << 16).to_be_bytes());
        let tkhd = make_full_box(b"tkhd", 0, 3, &tkhd_body);

        // edts -> elst v0 flags=1 (tekrarlayan)
        let mut elst_body = Vec::new();
        elst_body.extend_from_slice(&1u32.to_be_bytes()); // entry_count = 1
        elst_body.extend_from_slice(&(sample_sizes.len() as u32).to_be_bytes()); // segment_duration
        elst_body.extend_from_slice(&0i32.to_be_bytes()); // media_time
        elst_body.extend_from_slice(&1i16.to_be_bytes()); // media_rate_integer
        elst_body.extend_from_slice(&0i16.to_be_bytes()); // media_rate_fraction
        let edts = make_box(b"edts", &make_full_box(b"elst", 0, 1, &elst_body));

        // mdhd v0
        let mut mdhd_body = vec![0u8; 20];
        mdhd_body[8..12].copy_from_slice(&30u32.to_be_bytes()); // timescale = 30
        mdhd_body[12..16].copy_from_slice(&(sample_sizes.len() as u32).to_be_bytes());
        let mdhd = make_full_box(b"mdhd", 0, 0, &mdhd_body);

        // hdlr
        let mut hdlr_body = Vec::new();
        hdlr_body.extend_from_slice(&0u32.to_be_bytes());
        hdlr_body.extend_from_slice(handler);
        hdlr_body.extend_from_slice(&[0u8; 13]);
        let hdlr = make_full_box(b"hdlr", 0, 0, &hdlr_body);

        // stsd -> av01 (78 bayt VisualSampleEntry + av1C [+ auxi])
        let mut av01_payload = vec![0u8; 78];
        av01_payload[24..26].copy_from_slice(&1u16.to_be_bytes());
        av01_payload[26..28].copy_from_slice(&1u16.to_be_bytes());
        let av1c = make_box(b"av1C", &[0x81, 0x00, 0x0C, 0x00]);
        av01_payload.extend_from_slice(&av1c);
        if auxl_target.is_some() {
            let auxi = make_full_box(
                b"auxi",
                0,
                0,
                b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\0",
            );
            av01_payload.extend_from_slice(&auxi);
        }
        let mut stsd_body = Vec::new();
        stsd_body.extend_from_slice(&1u32.to_be_bytes());
        stsd_body.extend_from_slice(&make_box(b"av01", &av01_payload));
        let stsd = make_full_box(b"stsd", 0, 0, &stsd_body);

        // stts: her örnek delta = 1
        let mut stts_body = Vec::new();
        stts_body.extend_from_slice(&1u32.to_be_bytes());
        stts_body.extend_from_slice(&(sample_sizes.len() as u32).to_be_bytes());
        stts_body.extend_from_slice(&1u32.to_be_bytes());
        let stts = make_full_box(b"stts", 0, 0, &stts_body);

        // stsc: 1 yığında tüm örnekler
        let mut stsc_body = Vec::new();
        stsc_body.extend_from_slice(&1u32.to_be_bytes());
        stsc_body.extend_from_slice(&1u32.to_be_bytes());
        stsc_body.extend_from_slice(&(sample_sizes.len() as u32).to_be_bytes());
        stsc_body.extend_from_slice(&1u32.to_be_bytes());
        let stsc = make_full_box(b"stsc", 0, 0, &stsc_body);

        // stsz
        let mut stsz_body = Vec::new();
        stsz_body.extend_from_slice(&0u32.to_be_bytes());
        stsz_body.extend_from_slice(&(sample_sizes.len() as u32).to_be_bytes());
        for &sz in sample_sizes {
            stsz_body.extend_from_slice(&sz.to_be_bytes());
        }
        let stsz = make_full_box(b"stsz", 0, 0, &stsz_body);

        // stco
        let mut stco_body = Vec::new();
        stco_body.extend_from_slice(&1u32.to_be_bytes());
        stco_body.extend_from_slice(&chunk_offset.to_be_bytes());
        let stco = make_full_box(b"stco", 0, 0, &stco_body);

        // stss
        let mut stss_body = Vec::new();
        stss_body.extend_from_slice(&(keyframes_1based.len() as u32).to_be_bytes());
        for &kf in keyframes_1based {
            stss_body.extend_from_slice(&kf.to_be_bytes());
        }
        let stss = make_full_box(b"stss", 0, 0, &stss_body);

        let mut stbl_body = Vec::new();
        for b in [&stsd, &stts, &stsc, &stsz, &stco, &stss] {
            stbl_body.extend_from_slice(b);
        }
        let stbl = make_box(b"stbl", &stbl_body);
        let minf = make_box(b"minf", &stbl);
        let mut mdia_body = Vec::new();
        for b in [&mdhd, &hdlr, &minf] {
            mdia_body.extend_from_slice(b);
        }
        let mdia = make_box(b"mdia", &mdia_body);

        let mut trak_body = Vec::new();
        trak_body.extend_from_slice(&tkhd);
        trak_body.extend_from_slice(&edts);
        if let Some(target) = auxl_target {
            let tref = make_box(b"tref", &make_box(b"auxl", &target.to_be_bytes()));
            trak_body.extend_from_slice(&tref);
        }
        trak_body.extend_from_slice(&mdia);
        make_box(b"trak", &trak_body)
    };

    let color_sizes: Vec<u32> = samples.iter().map(|s| s.len() as u32).collect();
    let alpha_sizes: Vec<u32> = alpha_samples
        .map(|asmp| asmp.iter().map(|s| s.len() as u32).collect())
        .unwrap_or_default();

    let mut mdat_payload = Vec::new();
    for s in samples {
        mdat_payload.extend_from_slice(s);
    }
    let alpha_rel_off = mdat_payload.len() as u32;
    if let Some(asmp) = alpha_samples {
        for s in asmp {
            mdat_payload.extend_from_slice(s);
        }
    }

    // Yer tutucu ofsetle moov boyutunu ölç, ardından gerçek mdat ofsetiyle yeniden kur.
    let build_moov = |color_off: u32, alpha_off: u32| -> Vec<u8> {
        let mut moov_body = Vec::new();
        moov_body.extend_from_slice(&mvhd);
        moov_body.extend_from_slice(&make_track_box(
            1,
            b"pict",
            None,
            &color_sizes,
            color_off,
            &[1, 3],
        ));
        if alpha_samples.is_some() {
            moov_body.extend_from_slice(&make_track_box(
                2,
                b"auxv",
                Some(1),
                &alpha_sizes,
                alpha_off,
                &[1, 3],
            ));
        }
        make_box(b"moov", &moov_body)
    };

    let dummy_moov = build_moov(0, 0);
    let mdat_data_off = (ftyp.len() + dummy_moov.len() + 8) as u32;
    let moov = build_moov(mdat_data_off, mdat_data_off + alpha_rel_off);
    let mdat = make_box(b"mdat", &mdat_payload);

    let mut file = Vec::new();
    file.extend_from_slice(&ftyp);
    file.extend_from_slice(&moov);
    file.extend_from_slice(&mdat);
    file
}

#[test]
fn avif_sequence_avis_streaming_alpha_loop_and_seek() {
    let samples = [SAMPLE_1X1_AV1_OBU, SAMPLE_1X1_AV1_OBU, SAMPLE_1X1_AV1_OBU];
    let avis_inf = build_synthetic_avis_with_alpha(&samples, Some(&samples), true);
    assert!(probe(&avis_inf));

    let info = parse_sequence_info(&avis_inf).expect("parse_sequence_info başarılı olmalı");
    assert_eq!((info.width, info.height), (1, 1));
    assert_eq!(info.frame_count, 3);
    assert_eq!(info.timescale, 30);
    assert_eq!(info.duration_in_timescale, 3);
    assert_eq!(info.repetition_count, LoopCount::Infinite);
    assert!(info.has_alpha);

    let mut seq = AvifSequenceDecoder::new(&avis_inf).expect("AvifSequenceDecoder kurulmalı");
    assert!(seq.is_keyframe(0));
    assert!(!seq.is_keyframe(1));
    assert!(seq.is_keyframe(2));
    assert_eq!(seq.nearest_keyframe(1), 0);
    assert_eq!(seq.nearest_keyframe(2), 2);

    let mut frames = Vec::new();
    while let Some(f) = seq.next_frame().expect("next_frame başarılı olmalı") {
        assert_eq!(f.image.y_plane_u8(), Some(&[190u8][..]));
        assert_eq!(f.image.alpha_plane_u8(), Some(&[203u8][..]));
        frames.push(f);
    }
    assert_eq!(frames.len(), 3);
    assert_eq!(frames[0].pts_in_timescale, 0);
    assert_eq!(frames[1].pts_in_timescale, 1);
    assert_eq!(frames[2].pts_in_timescale, 2);
    assert!(frames[0].is_keyframe && !frames[1].is_keyframe && frames[2].is_keyframe);

    // Geriye ve ileriye rastgele erişim (nth_frame) sıralı çözmeyle birebir aynı olmalı.
    let f2 = seq.nth_frame(2).expect("nth_frame(2)");
    assert_eq!(f2.image, frames[2].image);
    let f1 = seq.nth_frame(1).expect("nth_frame(1)");
    assert_eq!(f1.image, frames[1].image);
    let f0 = seq.nth_frame(0).expect("nth_frame(0)");
    assert_eq!(f0.image, frames[0].image);
    assert!(matches!(seq.nth_frame(3), Err(Error::InvalidData(_))));

    // Sonlu döngü sayısı: 3 oynatma -> LoopCount::Count(2)
    let avis_finite = build_synthetic_avis_with_alpha(&samples, None, false);
    let info_fin = parse_sequence_info(&avis_finite).unwrap();
    assert_eq!(info_fin.repetition_count, LoopCount::Count(2));
    assert!(!info_fin.has_alpha);

    // Hareketsiz AVIF dosyası da AvifSequenceDecoder üzerinden 1 karelik dizi olarak açılabilmeli.
    let still = build_transformed_avif(SAMPLE_1X1_AV1_OBU, 1, 1, None, None, None, false);
    let mut still_seq = AvifSequenceDecoder::new(&still).expect("hareketsiz AVIF dizi kipi");
    assert_eq!(still_seq.frame_count(), 1);
    let sf = still_seq.next_frame().unwrap().unwrap();
    assert_eq!(sf.index, 0);
    assert_eq!(sf.image.y_plane_u8(), Some(&[190u8][..]));
    assert!(still_seq.next_frame().unwrap().is_none());
}

/// `5.A.1`: Ortak uçtan uca `Arc<Butce>` denetiminin `cmin`, çoklu karo `grid` + `alpha`
/// (`t=1` ve `t=4`), `tmap` kazanç haritası, `AvifImage::to_rgba8` renk dönüşümü ve
/// `parse_sequence_info_with` / `AvifSequenceDecoder::with_config` üzerinde işlediğini doğrular.
#[test]
fn avif_shared_budget_cmin_grid_alpha_gainmap_color_and_sequence() {
    // 1. `cmin` sıkıştırılmış meta kutusu: hem `parse_info_with` hem `decode_with` dar bütçede reddetmeli.
    let big_xmp: Vec<u8> = (0usize..48 * 1024)
        .map(|i| b'A' + ((i.wrapping_mul(31) ^ (i >> 3)) % 26) as u8)
        .collect();
    let exif_block = b"\x00\x00\x00\x00II*\x00EXIF_CMIN_BUDGET";
    let avif_cmin = build_compressed_meta_avif(&big_xmp, None, exif_block, None, true);

    let tight_cmin_cfg = AvifDecodeConfig {
        allocation_limit: Some(8 * 1024),
        ..AvifDecodeConfig::default()
    };
    assert!(matches!(
        parse_info_with(&avif_cmin, &tight_cmin_cfg),
        Err(Error::BudgetExceeded { .. })
    ));
    assert!(matches!(
        decode_with(&avif_cmin, &tight_cmin_cfg),
        Err(Error::BudgetExceeded { .. })
    ));

    let generous_cmin_cfg = AvifDecodeConfig {
        allocation_limit: Some(512 * 1024),
        ..AvifDecodeConfig::default()
    };
    let cmin_info = parse_info_with(&avif_cmin, &generous_cmin_cfg)
        .expect("yeterli bütçe ile cmin parse_info_with");
    assert_eq!(cmin_info.xmp.as_deref(), Some(big_xmp.as_slice()));
    let cmin_img =
        decode_with(&avif_cmin, &generous_cmin_cfg).expect("yeterli bütçe ile cmin decode_with");
    assert_eq!(cmin_img.xmp(), Some(big_xmp.as_slice()));

    // 2. Çoklu karo `2x2` `grid` + `alpha`:
    //    Tekil bir karo çözümü minimum bütçeye sığarken, 2x2 grid + tuval + alfa toplamı
    //    hem tek iş parçacıklı (`n_threads = 1`) hem paralel (`n_threads = 4`) kipte `BudgetExceeded`
    //    döndürmelidir.
    let tile_obu = read_oddheight_keyframe_obu();
    let (tile_w, tile_h) = if tile_obu.len() > SAMPLE_1X1_AV1_OBU.len() {
        (64u32, 63u32)
    } else {
        (1u32, 1u32)
    };
    let single_tile_avif =
        build_transformed_avif(&tile_obu, tile_w, tile_h, None, None, None, false);
    let grid_alpha_avif = build_grid_and_alpha_avif(&tile_obu, tile_w, tile_h, tile_w, tile_h);

    // Tekil karo için minimum başarılı bütçeyi ikili aramayla tam belirle.
    let mut lo = 1usize;
    let mut hi = 4 * 1024 * 1024usize;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let cfg = AvifDecodeConfig {
            n_threads: 1,
            allocation_limit: Some(mid),
            ..AvifDecodeConfig::default()
        };
        if decode_with(&single_tile_avif, &cfg).is_ok() {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    let single_tile_min = lo;
    // Tekil karo tam bu bütçeyle çözülür, ancak 2x2 grid + tuval + alfa toplamı sığmaz.
    assert!(
        decode_with(
            &single_tile_avif,
            &AvifDecodeConfig {
                n_threads: 1,
                allocation_limit: Some(single_tile_min),
                ..AvifDecodeConfig::default()
            }
        )
        .is_ok()
    );
    for threads in [1usize, 4usize] {
        let res = decode_with(
            &grid_alpha_avif,
            &AvifDecodeConfig {
                n_threads: threads,
                allocation_limit: Some(single_tile_min),
                ..AvifDecodeConfig::default()
            },
        );
        assert!(
            matches!(res, Err(Error::BudgetExceeded { .. })),
            "grid+alpha (t={threads}) tekil karo bütçesini ({single_tile_min}) aşmalı, sonuç: {res:?}"
        );
    }

    let grid_img_ok = decode_with(
        &grid_alpha_avif,
        &AvifDecodeConfig {
            n_threads: 4,
            allocation_limit: Some(single_tile_min * 8 + 512 * 1024),
            ..AvifDecodeConfig::default()
        },
    )
    .expect("geniş bütçede t=4 grid+alpha çözülmeli");
    let rgba = grid_img_ok.to_rgba8().expect("geniş bütçede to_rgba8");
    assert_eq!((rgba.width, rgba.height), (tile_w, tile_h));
    assert_eq!(rgba.pixels.len(), (tile_w * tile_h * 4) as usize);

    // 3. `AvifImage` canlıyken `to_rgb8()` / `to_rgba8()` renk dönüşümü ortak `Arc<Butce>` paylaşır:
    //    352×288 4:2:0 + 2 MiB XMP + Exif toplamında tek bayt hassasiyeti.
    let (obu_352, w352, h352) = first_ivf_packet_00001105();
    let color_xmp: Vec<u8> = (0usize..2 * 1024 * 1024)
        .map(|i| b'a' + ((i.wrapping_mul(17) ^ (i >> 4)) % 26) as u8)
        .collect();
    let avif_color_budget = build_compressed_meta_avif_with_frame(
        &obu_352, w352, h352, &color_xmp, None, exif_block, None, false,
    );
    // 00001105.ivf 352×288 8-bit Monochrome (yalnızca Y düzlemi: 352*288 = 101 376 bayt).
    let yuv_bytes = (w352 as usize) * (h352 as usize);
    let rgb_bytes = (w352 as usize) * (h352 as usize) * 3;
    let rgba_bytes = (w352 as usize) * (h352 as usize) * 4;
    let live_img_bytes = yuv_bytes + (exif_block.len() - 4) + color_xmp.len();
    let container_overhead =
        AvifPrepared::from_vec(avif_color_budget.clone(), &AvifDecodeConfig::default())
            .unwrap()
            .held_bytes()
            - avif_color_budget.len()
            - color_xmp.len()
            - (exif_block.len() - 4);
    let decode_exact_limit = live_img_bytes + container_overhead;

    let img_exact_only = decode_with(
        &avif_color_budget,
        &AvifDecodeConfig {
            allocation_limit: Some(decode_exact_limit),
            ..AvifDecodeConfig::default()
        },
    )
    .expect("tam canlı görüntü + kapsayıcı tablosu bütçesinde decode_with geçmeli");
    assert_eq!(
        img_exact_only.to_rgb8().unwrap_err(),
        Error::BudgetExceeded {
            requested: live_img_bytes + rgb_bytes,
            limit: decode_exact_limit,
        }
    );

    let img_rgb_exact = decode_with(
        &avif_color_budget,
        &AvifDecodeConfig {
            allocation_limit: Some(live_img_bytes + rgb_bytes),
            ..AvifDecodeConfig::default()
        },
    )
    .expect("canlı görüntü + RGB8 bütçesinde decode_with geçmeli");
    assert_eq!(
        img_rgb_exact
            .to_rgb8()
            .expect("to_rgb8 tam sınırda geçmeli")
            .pixels
            .len(),
        rgb_bytes
    );
    assert_eq!(
        img_rgb_exact.to_rgba8().unwrap_err(),
        Error::BudgetExceeded {
            requested: live_img_bytes + rgba_bytes,
            limit: live_img_bytes + rgb_bytes,
        }
    );

    let img_rgba_exact = decode_with(
        &avif_color_budget,
        &AvifDecodeConfig {
            allocation_limit: Some(live_img_bytes + rgba_bytes),
            ..AvifDecodeConfig::default()
        },
    )
    .expect("canlı görüntü + RGBA8 bütçesinde decode_with geçmeli");
    assert_eq!(
        img_rgba_exact
            .to_rgba8()
            .expect("to_rgba8 tam sınırda geçmeli")
            .pixels
            .len(),
        rgba_bytes
    );

    // 4. `tmap` kazanç haritası ortak bütçesi (`decode_gain_map = false` vs `true`):
    let tmap_avif = build_tmap_gainmap_avif(true, &[3, 1], 0, 0, 0, 1, &[], 9, 13);
    let mut tlo = 1usize;
    let mut thi = 2 * 1024 * 1024usize;
    while tlo < thi {
        let mid = tlo + (thi - tlo) / 2;
        let cfg = AvifDecodeConfig {
            n_threads: 1,
            decode_gain_map: false,
            allocation_limit: Some(mid),
            ..AvifDecodeConfig::default()
        };
        if decode_with(&tmap_avif, &cfg).is_ok() {
            thi = mid;
        } else {
            tlo = mid + 1;
        }
    }
    let base_only_min = tlo;
    // `decode_gain_map = true` olduğunda taban düzlemler + alt_icc canlıyken kazanç haritası
    // çözücüsü aynı `Arc<Butce>` üzerinde çalışır ve ek bellek gerektirir.
    let gm_tight = decode_with(
        &tmap_avif,
        &AvifDecodeConfig {
            n_threads: 1,
            decode_gain_map: true,
            allocation_limit: Some(base_only_min),
            ..AvifDecodeConfig::default()
        },
    );
    assert!(
        matches!(gm_tight, Err(Error::BudgetExceeded { .. })),
        "taban minimum bütçesinde decode_gain_map=true BudgetExceeded dönmeli, sonuç: {gm_tight:?}"
    );
    let gm_ok = decode_with(
        &tmap_avif,
        &AvifDecodeConfig {
            n_threads: 1,
            decode_gain_map: true,
            allocation_limit: Some(base_only_min + 256 * 1024),
            ..AvifDecodeConfig::default()
        },
    )
    .expect("yeterli bütçede tmap + gain_map çözülmeli");
    assert!(gm_ok.gain_map().is_some());

    // 5. `parse_sequence_info_with` ve `AvifSequenceDecoder::with_config` ortak bütçesi:
    let samples = [SAMPLE_1X1_AV1_OBU, SAMPLE_1X1_AV1_OBU, SAMPLE_1X1_AV1_OBU];
    let avis = build_synthetic_avis_with_alpha(&samples, Some(&samples), true);
    let tiny_seq_cfg = AvifDecodeConfig {
        allocation_limit: Some(16),
        ..AvifDecodeConfig::default()
    };
    assert!(matches!(
        parse_sequence_info_with(&avis, &tiny_seq_cfg),
        Err(Error::BudgetExceeded { .. })
    ));
    assert!(matches!(
        AvifSequenceDecoder::with_config(&avis, tiny_seq_cfg),
        Err(Error::BudgetExceeded { .. })
    ));
    let ok_seq_cfg = AvifDecodeConfig {
        allocation_limit: Some(512 * 1024),
        ..AvifDecodeConfig::default()
    };
    let sinfo = parse_sequence_info_with(&avis, &ok_seq_cfg)
        .expect("yeterli bütçede parse_sequence_info_with");
    assert_eq!(sinfo.frame_count, 3);
    let mut seq = AvifSequenceDecoder::with_config(&avis, ok_seq_cfg)
        .expect("yeterli bütçede sequence decoder");
    let frame0 = seq.next_frame().expect("ilk kare").expect("Some");
    assert_eq!(frame0.image.y_plane_u8(), Some(&[190u8][..]));
    assert_eq!(frame0.image.alpha_plane_u8(), Some(&[203u8][..]));
}

/// `5.B.1` + `5.B.2`: Doğrudan dilime yazan `write_*_into` metodları (`0` çıktı bütçesi + boyut
/// denetimi + `to_*` eşitliği) ve `AvifPrepared` tek geçişli ayrıştırma doğrulaması.
#[test]
fn avif_direct_slice_writers_and_prepared_single_parse() {
    // 1. `write_*_into` ile `to_*` bit-birebir eşitliği ve yanlış boyut reddi
    let grid_alpha = build_grid_and_alpha_avif(SAMPLE_1X1_AV1_OBU, 1, 1, 1, 1);
    let img = decode(&grid_alpha).expect("grid+alpha çözülmeli");

    let exp_rgb8 = img.to_rgb8().unwrap();
    let exp_rgba8 = img.to_rgba8().unwrap();
    let exp_rgb16 = img.to_rgb16().unwrap();
    let exp_rgba16 = img.to_rgba16().unwrap();

    let mut rgb8_buf = vec![0u8; exp_rgb8.pixels.len()];
    let mut rgba8_buf = vec![0u8; exp_rgba8.pixels.len()];
    let mut rgb16_buf = vec![0u16; exp_rgb16.pixels.len()];
    let mut rgba16_buf = vec![0u16; exp_rgba16.pixels.len()];
    let mut rgb16_ne_buf = vec![0u8; exp_rgb16.pixels.len() * 2];
    let mut rgba16_ne_buf = vec![0u8; exp_rgba16.pixels.len() * 2];

    img.write_rgb8_into(&mut rgb8_buf).unwrap();
    img.write_rgba8_into(&mut rgba8_buf).unwrap();
    img.write_rgb16_into(&mut rgb16_buf).unwrap();
    img.write_rgba16_into(&mut rgba16_buf).unwrap();
    img.write_rgb16_scaled_ne_into(&mut rgb16_ne_buf).unwrap();
    img.write_rgba16_scaled_ne_into(&mut rgba16_ne_buf).unwrap();

    assert_eq!(rgb8_buf, exp_rgb8.pixels);
    assert_eq!(rgba8_buf, exp_rgba8.pixels);
    assert_eq!(rgb16_buf, exp_rgb16.pixels);
    assert_eq!(rgba16_buf, exp_rgba16.pixels);

    let exp_rgb16_ne: Vec<u8> = exp_rgb16
        .pixels
        .iter()
        .flat_map(|&v| (v * 257).to_ne_bytes())
        .collect();
    let exp_rgba16_ne: Vec<u8> = exp_rgba16
        .pixels
        .iter()
        .flat_map(|&v| (v * 257).to_ne_bytes())
        .collect();
    assert_eq!(rgb16_ne_buf, exp_rgb16_ne);
    assert_eq!(rgba16_ne_buf, exp_rgba16_ne);

    assert!(matches!(
        img.write_rgb8_into(&mut [0u8; 2]),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        img.write_rgba8_into(&mut [0u8; 3]),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        img.write_rgb16_into(&mut [0u16; 2]),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        img.write_rgba16_into(&mut [0u16; 3]),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        img.write_rgb16_scaled_ne_into(&mut [0u8; 5]),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        img.write_rgba16_scaled_ne_into(&mut [0u8; 7]),
        Err(Error::InvalidArgument(_))
    ));

    // 2. Harici `dst` dilimi bütçe tüketmez: tam canlı görüntü + kapsayıcı sınırında `to_rgb8()`
    //    reddederken `write_rgb8_into` sıfır ek bütçe ile başarıyla tamamlanmalı.
    let (obu_352, w352, h352) = first_ivf_packet_00001105();
    let xmp_pad = vec![b'X'; 2 * 1024 * 1024];
    let exif_block = b"\x00\x00\x00\x00II*\x00EXIF_DIRECT_WRITER";
    let avif_tight = build_compressed_meta_avif_with_frame(
        &obu_352, w352, h352, &xmp_pad, None, exif_block, None, false,
    );
    let yuv_bytes = (w352 as usize) * (h352 as usize);
    let container_overhead =
        AvifPrepared::from_vec(avif_tight.clone(), &AvifDecodeConfig::default())
            .unwrap()
            .held_bytes()
            - avif_tight.len()
            - xmp_pad.len()
            - (exif_block.len() - 4);
    let live_plus_container =
        yuv_bytes + (exif_block.len() - 4) + xmp_pad.len() + container_overhead;
    let img_tight = decode_with(
        &avif_tight,
        &AvifDecodeConfig {
            allocation_limit: Some(live_plus_container),
            ..AvifDecodeConfig::default()
        },
    )
    .expect("tam canlı sınırda çözülmeli");
    assert!(matches!(
        img_tight.to_rgb8(),
        Err(Error::BudgetExceeded { .. })
    ));
    let mut direct_rgb = vec![0u8; (w352 as usize) * (h352 as usize) * 3];
    img_tight
        .write_rgb8_into(&mut direct_rgb)
        .expect("dış dilime doğrudan yazım ek bütçe tüketmeden geçmeli");

    // 3. `AvifPrepared` tek ayrıştırma (`cmin` sıkıştırılmış meta dâhil) paritesi
    let avif_cmin = build_compressed_meta_avif(&xmp_pad[..4096], None, exif_block, None, true);
    let ref_info = parse_info(&avif_cmin).unwrap();
    let ref_img = decode(&avif_cmin).unwrap();

    let prepared = AvifPrepared::from_vec(avif_cmin, &AvifDecodeConfig::default())
        .expect("AvifPrepared::from_vec cmin");
    assert_eq!(prepared.info(), &ref_info);
    let prep_img = prepared
        .decode_with(&AvifDecodeConfig::default())
        .expect("AvifPrepared::decode_with");
    assert_eq!(prep_img, ref_img);
}

#[test]
fn avif_container_tables_fake_counts_and_prepared_metadata_respect_budget() {
    // 1. Sahte dev iloc extent_count (offset_size=4, length_size=4 ama 0 bayt kalmış)
    //    veya sahte dev iref ref_count ön-tahsis yapmadan InvalidData ile reddedilmeli.
    let ftyp = make_ftyp(b"avif", &[*b"mif1"]);
    let hdlr = make_hdlr_pict();
    let pitm = make_full_box(b"pitm", 0, 0, &1u16.to_be_bytes());
    let mut bad_iloc_body = vec![0x44u8, 0x00];
    bad_iloc_body.extend_from_slice(&1u16.to_be_bytes()); // item_count = 1
    bad_iloc_body.extend_from_slice(&1u16.to_be_bytes()); // item_id = 1
    bad_iloc_body.extend_from_slice(&0u16.to_be_bytes()); // data_reference_index = 0
    bad_iloc_body.extend_from_slice(&u16::MAX.to_be_bytes()); // extent_count = 65535 (sahte)
    let bad_iloc = make_full_box(b"iloc", 0, 0, &bad_iloc_body);
    let mut meta_children = Vec::new();
    meta_children.extend_from_slice(&hdlr);
    meta_children.extend_from_slice(&pitm);
    meta_children.extend_from_slice(&bad_iloc);
    let mut bad_iloc_file = ftyp.clone();
    bad_iloc_file.extend_from_slice(&make_full_box(b"meta", 0, 0, &meta_children));
    let err = parse_info_with(
        &bad_iloc_file,
        &AvifDecodeConfig {
            allocation_limit: Some(4096),
            ..AvifDecodeConfig::default()
        },
    )
    .expect_err("sahte dev iloc extent_count reddedilmeli");
    assert!(matches!(err, Error::InvalidData(_)));

    // 2. Sahte dev iref ref_count (0 bayt kalmış) ön-tahsis yapmadan reddedilmeli.
    let mut bad_dimg = Vec::new();
    bad_dimg.extend_from_slice(&1u16.to_be_bytes()); // from_item_id
    bad_dimg.extend_from_slice(&u16::MAX.to_be_bytes()); // ref_count = 65535 (sahte)
    let bad_iref = make_full_box(b"iref", 0, 0, &make_box(b"dimg", &bad_dimg));
    let mut meta_iref = Vec::new();
    meta_iref.extend_from_slice(&hdlr);
    meta_iref.extend_from_slice(&pitm);
    meta_iref.extend_from_slice(&bad_iref);
    let mut bad_iref_file = ftyp;
    bad_iref_file.extend_from_slice(&make_full_box(b"meta", 0, 0, &meta_iref));
    let err_iref = parse_info_with(
        &bad_iref_file,
        &AvifDecodeConfig {
            allocation_limit: Some(4096),
            ..AvifDecodeConfig::default()
        },
    )
    .expect_err("sahte dev iref ref_count reddedilmeli");
    assert!(matches!(err_iref, Error::InvalidData(_)));

    // 3. `AvifPrepared::decode_with` çözme başlamadan önce canlı ICC/Exif/XMP metaveri baytlarını
    //    bütçeye kaydetmeli (limit < canlı metaveri ise doğrudan BudgetExceeded).
    let big_xmp = vec![b'X'; 64 * 1024];
    let exif_block = b"\x00\x00\x00\x00II*\x00EXIF_META_BUDGET";
    let avif_meta = build_compressed_meta_avif(&big_xmp, None, exif_block, None, false);
    let prepared = AvifPrepared::from_vec(avif_meta, &AvifDecodeConfig::default())
        .expect("sınırsız AvifPrepared");
    let err_prep = prepared
        .clone()
        .decode_with(&AvifDecodeConfig {
            allocation_limit: Some(32 * 1024),
            ..AvifDecodeConfig::default()
        })
        .expect_err("64 KiB canlı XMP taşıyan AvifPrepared 32 KiB bütçede reddedilmeli");
    assert!(matches!(err_prep, Error::BudgetExceeded { .. }));

    // 4. `parse_info_with` erken `max_pixels` sınırı.
    let err_px = parse_info_with(
        &build_compressed_meta_avif(b"<x/>", None, exif_block, None, false),
        &AvifDecodeConfig {
            max_pixels: 0, // 0 = sınırsız, bu geçmeli
            ..AvifDecodeConfig::default()
        },
    );
    assert!(err_px.is_ok());
    const AV1_1X1_TEST: &[u8] = &[
        0x12, 0x00, 0x0A, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7D, 0xFC, 0xC0, 0x20, 0x32, 0x17,
        0x10, 0x01, 0x91, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0xD2, 0x16,
        0x8C, 0x64, 0xBE, 0x2E, 0xC7, 0xDA, 0xEE, 0x96,
    ];
    let grid_avif = build_grid_and_alpha_avif(AV1_1X1_TEST, 1, 1, 2, 2);
    let err_grid_px = parse_info_with(
        &grid_avif,
        &AvifDecodeConfig {
            max_pixels: 3, // 2x2 = 4 > 3
            ..AvifDecodeConfig::default()
        },
    )
    .expect_err("2x2 AVIF max_pixels=3 ile parse_info_with aşamasında reddedilmeli");
    assert!(matches!(err_grid_px, Error::BudgetExceeded { .. }));
}

/// `6.B.2`: `lsel` uzamsal katman (`spatial_id`) seçimi ve `a1lx` önek/katman doğrulaması.
#[test]
fn avif_lsel_spatial_id_and_a1lx_validation() {
    // 2 katmanlı sentetik 1x1 AV1 akışı:
    // - TD + SequenceHeader (13 bayt)
    // - Katman 0: spatial_id = 0, show_frame = 0 (gizli anahtar kare)
    // - Katman 1: spatial_id = 1, show_frame = 1 (görünür anahtar kare, Y=190)
    let td_and_seq: &[u8] = &SAMPLE_1X1_AV1_OBU[..13];
    const LAYER0_FRAME_PAYLOAD: &[u8] = &[
        0x00, 0x00, 0xFF, 0x64, 0x40, 0x00, 0x00, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0xD2,
        0x16, 0x8C, 0x64, 0xBE, 0x2E, 0xC7, 0xDA, 0xEE, 0x96,
    ];
    let layer1_frame_payload: &[u8] = &SAMPLE_1X1_AV1_OBU[15..];

    let mut layered_obu = Vec::new();
    layered_obu.extend_from_slice(td_and_seq);
    // OBU_FRAME (6) + extension_flag(1) + has_size_field(1) = 0x36; ext spatial_id=0 -> 0x00
    layered_obu.push(0x36);
    layered_obu.push(0x00);
    layered_obu.push(LAYER0_FRAME_PAYLOAD.len() as u8);
    layered_obu.extend_from_slice(LAYER0_FRAME_PAYLOAD);
    let layer0_total_len = layered_obu.len();
    // OBU_FRAME (6) + extension_flag(1) + has_size_field(1) = 0x36; ext spatial_id=1 -> 0x08
    layered_obu.push(0x36);
    layered_obu.push(0x08);
    layered_obu.push(layer1_frame_payload.len() as u8);
    layered_obu.extend_from_slice(layer1_frame_payload);

    let build_layered_avif = |lsel: Option<u16>, a1lx: Option<[u16; 3]>| -> Vec<u8> {
        let ftyp = make_ftyp(b"avif", &[*b"mif1"]);
        let hdlr = make_hdlr_pict();
        let pitm = make_full_box(b"pitm", 0, 0, &1u16.to_be_bytes());
        let mut iinf_body = Vec::new();
        iinf_body.extend_from_slice(&1u16.to_be_bytes());
        iinf_body.extend_from_slice(&make_infe_v2(1, b"av01", None));
        let iinf = make_full_box(b"iinf", 0, 0, &iinf_body);

        let mut ipco_body = Vec::new();
        ipco_body.extend_from_slice(&make_ispe(1, 1));
        let mut props: Vec<(u8, bool)> = vec![(1, false)];
        let mut next_idx = 2u8;
        if let Some(layer_id) = lsel {
            ipco_body.extend_from_slice(&make_box(b"lsel", &layer_id.to_be_bytes()));
            props.push((next_idx, true));
            next_idx += 1;
        }
        if let Some(sizes) = a1lx {
            let mut a1lx_body = vec![0u8]; // large_size = 0 (16-bit)
            for s in sizes {
                a1lx_body.extend_from_slice(&s.to_be_bytes());
            }
            ipco_body.extend_from_slice(&make_box(b"a1lx", &a1lx_body));
            props.push((next_idx, false));
        }
        let ipco = make_box(b"ipco", &ipco_body);
        let mut ipma_body = Vec::new();
        ipma_body.extend_from_slice(&1u32.to_be_bytes());
        ipma_body.extend_from_slice(&1u16.to_be_bytes());
        ipma_body.push(props.len() as u8);
        for (idx, ess) in props {
            ipma_body.push(if ess { 0x80 | idx } else { idx });
        }
        let ipma = make_full_box(b"ipma", 0, 0, &ipma_body);
        let mut iprp_body = Vec::new();
        iprp_body.extend_from_slice(&ipco);
        iprp_body.extend_from_slice(&ipma);
        let iprp = make_box(b"iprp", &iprp_body);

        let idat = make_box(b"idat", &layered_obu);
        let mut iloc_body = vec![0x44u8, 0x00];
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes()); // construction_method = 1 (idat)
        iloc_body.extend_from_slice(&0u16.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&0u32.to_be_bytes());
        iloc_body.extend_from_slice(&(layered_obu.len() as u32).to_be_bytes());
        let iloc = make_full_box(b"iloc", 1, 0, &iloc_body);

        let mut meta_children = Vec::new();
        for b in [&hdlr, &pitm, &iinf, &iprp, &iloc, &idat] {
            meta_children.extend_from_slice(b);
        }
        let mut file = ftyp;
        file.extend_from_slice(&make_full_box(b"meta", 0, 0, &meta_children));
        file
    };

    let valid_a1lx = Some([layer0_total_len as u16, 0, 0]);

    // 1. lsel = 0 -> alt katman show_frame = 0 olduğundan görünür katman 0 karesi yok -> Err.
    assert!(matches!(
        decode(&build_layered_avif(Some(0), valid_a1lx)),
        Err(Error::InvalidData(_))
    ));
    assert!(matches!(
        decode(&build_layered_avif(Some(0), None)),
        Err(Error::InvalidData(_))
    ));

    // 2. lsel = 1 -> uzamsal katman 1 (show_frame = 1) seçilir ve Y=190 çözülür.
    let img_l1 = decode(&build_layered_avif(Some(1), valid_a1lx))
        .expect("lsel=1 + a1lx ile katman 1 çözülmeli");
    assert_eq!(img_l1.y_plane_u8(), Some(&[190u8][..]));
    let img_l1_no_a1lx =
        decode(&build_layered_avif(Some(1), None)).expect("lsel=1 (a1lx olmadan) çözülmeli");
    assert_eq!(img_l1_no_a1lx.y_plane_u8(), Some(&[190u8][..]));

    // 3. lsel = 0xFFFF (tüm katmanlar) -> son görünür katman 1 çözülür.
    let img_all =
        decode(&build_layered_avif(Some(0xFFFF), valid_a1lx)).expect("lsel=0xFFFF ile çözülmeli");
    assert_eq!(img_all.y_plane_u8(), Some(&[190u8][..]));

    // 4. lsel = 2 -> katman 2 mevcut değil (hem a1lx sınırında hem spatial_id aramasında Err).
    assert!(matches!(
        decode(&build_layered_avif(Some(2), valid_a1lx)),
        Err(Error::InvalidData(_))
    ));
    assert!(matches!(
        decode(&build_layered_avif(Some(2), None)),
        Err(Error::InvalidData(_))
    ));

    // 5. lsel = 4 -> geçersiz katman kimliği (AV1 en fazla 4 uzamsal katman: 0..=3) -> Err.
    assert!(matches!(
        decode(&build_layered_avif(Some(4), valid_a1lx)),
        Err(Error::InvalidData(_))
    ));
}

/// `6.B.1`: `AvifSequenceDecoder` içinde `ctts` v0 (işaretsiz) ve v1 (işaretli) PTS hesaplaması.
#[test]
fn avif_sequence_ctts_v0_and_signed_v1_pts() {
    let build_avis_with_ctts = |ctts_ver: u8, ctts_entries: &[(u32, u32)]| -> Vec<u8> {
        let samples = [SAMPLE_1X1_AV1_OBU, SAMPLE_1X1_AV1_OBU, SAMPLE_1X1_AV1_OBU];
        let ftyp = make_ftyp(b"avis", &[*b"msf1", *b"avif", *b"miaf"]);

        let mut mvhd_body = vec![0u8; 96];
        mvhd_body[8..12].copy_from_slice(&30u32.to_be_bytes());
        mvhd_body[12..16].copy_from_slice(&30u32.to_be_bytes());
        let mvhd = make_full_box(b"mvhd", 0, 0, &mvhd_body);

        let make_trak = |chunk_offset: u32| -> Vec<u8> {
            let mut tkhd_body = vec![0u8; 80];
            tkhd_body[8..12].copy_from_slice(&1u32.to_be_bytes());
            tkhd_body[16..20].copy_from_slice(&30u32.to_be_bytes());
            tkhd_body[36..40].copy_from_slice(&0x0001_0000i32.to_be_bytes()); // a = 1.0
            tkhd_body[52..56].copy_from_slice(&0x0001_0000i32.to_be_bytes()); // d = 1.0
            tkhd_body[68..72].copy_from_slice(&0x4000_0000i32.to_be_bytes()); // w = 1.0 (2.30)
            tkhd_body[72..76].copy_from_slice(&(1u32 << 16).to_be_bytes());
            tkhd_body[76..80].copy_from_slice(&(1u32 << 16).to_be_bytes());
            let tkhd = make_full_box(b"tkhd", 0, 3, &tkhd_body);

            let mut mdhd_body = vec![0u8; 20];
            mdhd_body[8..12].copy_from_slice(&30u32.to_be_bytes());
            mdhd_body[12..16].copy_from_slice(&30u32.to_be_bytes());
            let mdhd = make_full_box(b"mdhd", 0, 0, &mdhd_body);

            let mut hdlr_body = Vec::new();
            hdlr_body.extend_from_slice(&0u32.to_be_bytes());
            hdlr_body.extend_from_slice(b"pict");
            hdlr_body.extend_from_slice(&[0u8; 13]);
            let hdlr = make_full_box(b"hdlr", 0, 0, &hdlr_body);

            let mut av01_payload = vec![0u8; 78];
            av01_payload[24..26].copy_from_slice(&1u16.to_be_bytes());
            av01_payload[26..28].copy_from_slice(&1u16.to_be_bytes());
            av01_payload.extend_from_slice(&make_box(b"av1C", &[0x81, 0x00, 0x0C, 0x00]));
            let mut stsd_body = Vec::new();
            stsd_body.extend_from_slice(&1u32.to_be_bytes());
            stsd_body.extend_from_slice(&make_box(b"av01", &av01_payload));
            let stsd = make_full_box(b"stsd", 0, 0, &stsd_body);

            // stts: 3 örnek, her biri delta = 10 -> dts = [0, 10, 20]
            let mut stts_body = Vec::new();
            stts_body.extend_from_slice(&1u32.to_be_bytes());
            stts_body.extend_from_slice(&3u32.to_be_bytes());
            stts_body.extend_from_slice(&10u32.to_be_bytes());
            let stts = make_full_box(b"stts", 0, 0, &stts_body);

            // ctts
            let mut ctts_body = Vec::new();
            ctts_body.extend_from_slice(&(ctts_entries.len() as u32).to_be_bytes());
            for &(cnt, off) in ctts_entries {
                ctts_body.extend_from_slice(&cnt.to_be_bytes());
                ctts_body.extend_from_slice(&off.to_be_bytes());
            }
            let ctts = make_full_box(b"ctts", ctts_ver, 0, &ctts_body);

            let mut stsc_body = Vec::new();
            stsc_body.extend_from_slice(&1u32.to_be_bytes());
            stsc_body.extend_from_slice(&1u32.to_be_bytes());
            stsc_body.extend_from_slice(&3u32.to_be_bytes());
            stsc_body.extend_from_slice(&1u32.to_be_bytes());
            let stsc = make_full_box(b"stsc", 0, 0, &stsc_body);

            let mut stsz_body = Vec::new();
            stsz_body.extend_from_slice(&0u32.to_be_bytes());
            stsz_body.extend_from_slice(&3u32.to_be_bytes());
            for s in &samples {
                stsz_body.extend_from_slice(&(s.len() as u32).to_be_bytes());
            }
            let stsz = make_full_box(b"stsz", 0, 0, &stsz_body);

            let mut stco_body = Vec::new();
            stco_body.extend_from_slice(&1u32.to_be_bytes());
            stco_body.extend_from_slice(&chunk_offset.to_be_bytes());
            let stco = make_full_box(b"stco", 0, 0, &stco_body);

            let mut stbl_body = Vec::new();
            for b in [&stsd, &stts, &ctts, &stsc, &stsz, &stco] {
                stbl_body.extend_from_slice(b);
            }
            let stbl = make_box(b"stbl", &stbl_body);
            let minf = make_box(b"minf", &stbl);
            let mut mdia_body = Vec::new();
            for b in [&mdhd, &hdlr, &minf] {
                mdia_body.extend_from_slice(b);
            }
            let mdia = make_box(b"mdia", &mdia_body);
            let mut trak_body = Vec::new();
            trak_body.extend_from_slice(&tkhd);
            trak_body.extend_from_slice(&mdia);
            make_box(b"trak", &trak_body)
        };

        let mut mdat_payload = Vec::new();
        for s in &samples {
            mdat_payload.extend_from_slice(s);
        }
        let dummy_moov = make_box(b"moov", &[mvhd.as_slice(), &make_trak(0)].concat());
        let data_off = (ftyp.len() + dummy_moov.len() + 8) as u32;
        let moov = make_box(b"moov", &[mvhd.as_slice(), &make_trak(data_off)].concat());
        let mdat = make_box(b"mdat", &mdat_payload);

        let mut file = ftyp;
        file.extend_from_slice(&moov);
        file.extend_from_slice(&mdat);
        file
    };

    // 1. ctts v0 (işaretsiz ofsetler): dts=[0, 10, 20] + [20, 0, 5] -> pts=[20, 10, 25]
    let avis_v0 = build_avis_with_ctts(0, &[(1, 20), (1, 0), (1, 5)]);
    let mut dec_v0 = AvifSequenceDecoder::new(&avis_v0).expect("ctts v0 açılmalı");
    let mut pts_v0 = Vec::new();
    while let Some(f) = dec_v0.next_frame().expect("ctts v0 kare") {
        pts_v0.push(f.pts_in_timescale);
    }
    assert_eq!(pts_v0, vec![20, 10, 25]);

    // 2. ctts v1 (işaretli i32 ofsetler): dts=[0, 10, 20] + [10, -5, 0] -> pts=[10, 5, 20]
    let avis_v1 = build_avis_with_ctts(1, &[(1, 10), (1, (-5i32) as u32), (1, 0)]);
    let mut dec_v1 = AvifSequenceDecoder::new(&avis_v1).expect("ctts v1 açılmalı");
    let mut pts_v1 = Vec::new();
    while let Some(f) = dec_v1.next_frame().expect("ctts v1 kare") {
        pts_v1.push(f.pts_in_timescale);
    }
    assert_eq!(pts_v1, vec![10, 5, 20]);

    // 3. ctts v1 negatif PTS (dts=0 + (-1) = -1 < 0) reddedilmeli.
    let avis_neg = build_avis_with_ctts(1, &[(1, (-1i32) as u32), (2, 0)]);
    assert!(matches!(
        AvifSequenceDecoder::new(&avis_neg),
        Err(Error::InvalidData(_))
    ));
}

/// `6.E.4` (`P2-4`): Bitişik olmayan uzamsal katman kimliklerinde (`spatial_id = {0, 2}`)
/// `a1lx` + `lsel = 2` eşleşmesi ve mevcut olmayan `lsel = 1` reddi.
#[test]
fn avif_lsel_non_contiguous_spatial_id_with_a1lx() {
    let td_and_seq: &[u8] = &SAMPLE_1X1_AV1_OBU[..13];
    const LAYER0_FRAME_PAYLOAD: &[u8] = &[
        0x00, 0x00, 0xFF, 0x64, 0x40, 0x00, 0x00, 0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0xD2,
        0x16, 0x8C, 0x64, 0xBE, 0x2E, 0xC7, 0xDA, 0xEE, 0x96,
    ];
    let layer2_frame_payload: &[u8] = &SAMPLE_1X1_AV1_OBU[15..];

    let mut layered_obu = Vec::new();
    layered_obu.extend_from_slice(td_and_seq);
    // Katman dilimi 0: spatial_id = 0 (ext = 0x00)
    layered_obu.push(0x36);
    layered_obu.push(0x00);
    layered_obu.push(LAYER0_FRAME_PAYLOAD.len() as u8);
    layered_obu.extend_from_slice(LAYER0_FRAME_PAYLOAD);
    let layer0_total_len = layered_obu.len();
    // Katman dilimi 1: spatial_id = 2 (bit [4:3] = 2 -> ext = 0x10)
    layered_obu.push(0x36);
    layered_obu.push(0x10);
    layered_obu.push(layer2_frame_payload.len() as u8);
    layered_obu.extend_from_slice(layer2_frame_payload);

    let build_avif = |lsel: u16| -> Vec<u8> {
        let ftyp = make_ftyp(b"avif", &[*b"mif1"]);
        let hdlr = make_hdlr_pict();
        let pitm = make_full_box(b"pitm", 0, 0, &1u16.to_be_bytes());
        let mut iinf_body = 1u16.to_be_bytes().to_vec();
        iinf_body.extend_from_slice(&make_infe_v2(1, b"av01", None));
        let iinf = make_full_box(b"iinf", 0, 0, &iinf_body);

        let mut a1lx_body = vec![0u8];
        a1lx_body.extend_from_slice(&(layer0_total_len as u16).to_be_bytes());
        a1lx_body.extend_from_slice(&0u16.to_be_bytes());
        a1lx_body.extend_from_slice(&0u16.to_be_bytes());

        let mut ipco_body = make_ispe(1, 1);
        ipco_body.extend_from_slice(&make_box(b"lsel", &lsel.to_be_bytes()));
        ipco_body.extend_from_slice(&make_box(b"a1lx", &a1lx_body));
        let ipco = make_box(b"ipco", &ipco_body);

        let mut ipma_body = Vec::new();
        ipma_body.extend_from_slice(&1u32.to_be_bytes());
        ipma_body.extend_from_slice(&1u16.to_be_bytes());
        ipma_body.push(3);
        ipma_body.extend_from_slice(&[0x01, 0x82, 0x03]);
        let ipma = make_full_box(b"ipma", 0, 0, &ipma_body);
        let mut iprp_body = ipco;
        iprp_body.extend_from_slice(&ipma);
        let iprp = make_box(b"iprp", &iprp_body);

        let idat = make_box(b"idat", &layered_obu);
        let mut iloc_body = vec![0x44u8, 0x00];
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&0u16.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes());
        iloc_body.extend_from_slice(&0u32.to_be_bytes());
        iloc_body.extend_from_slice(&(layered_obu.len() as u32).to_be_bytes());
        let iloc = make_full_box(b"iloc", 1, 0, &iloc_body);

        let mut meta_children = Vec::new();
        for b in [&hdlr, &pitm, &iinf, &iprp, &iloc, &idat] {
            meta_children.extend_from_slice(b);
        }
        let mut file = ftyp;
        file.extend_from_slice(&make_full_box(b"meta", 0, 0, &meta_children));
        file
    };

    // lsel = 2: a1lx'te 2 dilim olsa da 2. dilim spatial_id=2 taşıdığı için başarıyla çözülmeli.
    let img_l2 = decode(&build_avif(2)).expect("bitişik olmayan spatial_id=2 lsel=2 ile çözülmeli");
    assert_eq!(img_l2.y_plane_u8(), Some(&[190u8][..]));

    // lsel = 1: hiçbir a1lx diliminde spatial_id=1 yok -> InvalidData.
    assert!(matches!(decode(&build_avif(1)), Err(Error::InvalidData(_))));

    // lsel = 0xFFFF: tüm katmanlar -> spatial_id=2 çözülür.
    let img_all = decode(&build_avif(0xFFFF)).expect("lsel=0xFFFF ile çözülmeli");
    assert_eq!(img_all.y_plane_u8(), Some(&[190u8][..]));
}

/// `6.E.4` (`P2-5`): AVIS `tkhd` 3×3 yönelim matrisi (90°/180°/ayna + geçersiz matris reddi)
/// ve `flags = 0` `elst` zaman çizelgesi (`media_time = -1` boş düzenleme + `media_time > 0`
/// `KEY -> INTER` sessiz `pre-roll` ve `next_frame`/`nth_frame` eşliği).
#[test]
fn avif_sequence_tkhd_orientation_and_elst_timeline() {
    const P: i32 = 0x0001_0000;
    const N: i32 = -0x0001_0000;
    const W: i32 = 0x4000_0000;

    // 432x240 gerçek KEY -> INTER1 -> INTER2 IVF fikstürünün ilk 3 paketini oku.
    let ivf_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_002_vq_aom_ctest_4.2.stripped.ivf",
    );
    let raw_ivf = fs::read(&ivf_path).expect("IVF fikstürü okunmalı");
    let mut ivf_pkts: Vec<Vec<u8>> = Vec::new();
    let mut pos = 32usize;
    while pos + 12 <= raw_ivf.len() && ivf_pkts.len() < 3 {
        let sz = u32::from_le_bytes(raw_ivf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 12;
        ivf_pkts.push(raw_ivf[pos..pos + sz].to_vec());
        pos += sz;
    }
    assert_eq!(ivf_pkts.len(), 3);

    type ElstEntryV0 = (u32, i32, i16, i16);

    #[allow(clippy::too_many_arguments)]
    let build_avis = |samples: &[&[u8]],
                      w: u32,
                      h: u32,
                      matrix: [i32; 9],
                      movie_ts: u32,
                      media_ts: u32,
                      sample_delta: u32,
                      elst_v0: Option<(u32, &[ElstEntryV0])>,
                      keyframes_1based: &[u32]|
     -> Vec<u8> {
        let ftyp = make_ftyp(b"avis", &[*b"msf1", *b"avif", *b"miaf"]);

        let mut mvhd_body = vec![0u8; 96];
        mvhd_body[8..12].copy_from_slice(&movie_ts.to_be_bytes());
        mvhd_body[12..16].copy_from_slice(&((samples.len() as u32) * sample_delta).to_be_bytes());
        let mvhd = make_full_box(b"mvhd", 0, 0, &mvhd_body);

        let make_trak = |chunk_offset: u32| -> Vec<u8> {
            let mut tkhd_body = vec![0u8; 80];
            tkhd_body[8..12].copy_from_slice(&1u32.to_be_bytes());
            tkhd_body[16..20]
                .copy_from_slice(&((samples.len() as u32) * sample_delta).to_be_bytes());
            for (i, &val) in matrix.iter().enumerate() {
                tkhd_body[36 + i * 4..40 + i * 4].copy_from_slice(&val.to_be_bytes());
            }
            tkhd_body[72..76].copy_from_slice(&(w << 16).to_be_bytes());
            tkhd_body[76..80].copy_from_slice(&(h << 16).to_be_bytes());
            let tkhd = make_full_box(b"tkhd", 0, 3, &tkhd_body);

            let edts_opt = elst_v0.map(|(flags, entries)| {
                let mut elst_body = Vec::new();
                elst_body.extend_from_slice(&(entries.len() as u32).to_be_bytes());
                for &(sd, mt, r_int, r_frac) in entries {
                    elst_body.extend_from_slice(&sd.to_be_bytes());
                    elst_body.extend_from_slice(&mt.to_be_bytes());
                    elst_body.extend_from_slice(&r_int.to_be_bytes());
                    elst_body.extend_from_slice(&r_frac.to_be_bytes());
                }
                make_box(b"edts", &make_full_box(b"elst", 0, flags, &elst_body))
            });

            let mut mdhd_body = vec![0u8; 20];
            mdhd_body[8..12].copy_from_slice(&media_ts.to_be_bytes());
            mdhd_body[12..16]
                .copy_from_slice(&((samples.len() as u32) * sample_delta).to_be_bytes());
            let mdhd = make_full_box(b"mdhd", 0, 0, &mdhd_body);

            let mut hdlr_body = Vec::new();
            hdlr_body.extend_from_slice(&0u32.to_be_bytes());
            hdlr_body.extend_from_slice(b"pict");
            hdlr_body.extend_from_slice(&[0u8; 13]);
            let hdlr = make_full_box(b"hdlr", 0, 0, &hdlr_body);

            let mut av01_payload = vec![0u8; 78];
            av01_payload[24..26].copy_from_slice(&(w as u16).to_be_bytes());
            av01_payload[26..28].copy_from_slice(&(h as u16).to_be_bytes());
            av01_payload.extend_from_slice(&make_box(b"av1C", &[0x81, 0x00, 0x0C, 0x00]));
            let mut stsd_body = 1u32.to_be_bytes().to_vec();
            stsd_body.extend_from_slice(&make_box(b"av01", &av01_payload));
            let stsd = make_full_box(b"stsd", 0, 0, &stsd_body);

            let mut stts_body = Vec::new();
            stts_body.extend_from_slice(&1u32.to_be_bytes());
            stts_body.extend_from_slice(&(samples.len() as u32).to_be_bytes());
            stts_body.extend_from_slice(&sample_delta.to_be_bytes());
            let stts = make_full_box(b"stts", 0, 0, &stts_body);

            let mut stsc_body = Vec::new();
            stsc_body.extend_from_slice(&1u32.to_be_bytes());
            stsc_body.extend_from_slice(&1u32.to_be_bytes());
            stsc_body.extend_from_slice(&(samples.len() as u32).to_be_bytes());
            stsc_body.extend_from_slice(&1u32.to_be_bytes());
            let stsc = make_full_box(b"stsc", 0, 0, &stsc_body);

            let mut stsz_body = Vec::new();
            stsz_body.extend_from_slice(&0u32.to_be_bytes());
            stsz_body.extend_from_slice(&(samples.len() as u32).to_be_bytes());
            for s in samples {
                stsz_body.extend_from_slice(&(s.len() as u32).to_be_bytes());
            }
            let stsz = make_full_box(b"stsz", 0, 0, &stsz_body);

            let mut stco_body = 1u32.to_be_bytes().to_vec();
            stco_body.extend_from_slice(&chunk_offset.to_be_bytes());
            let stco = make_full_box(b"stco", 0, 0, &stco_body);

            let mut stss_body = (keyframes_1based.len() as u32).to_be_bytes().to_vec();
            for &kf in keyframes_1based {
                stss_body.extend_from_slice(&kf.to_be_bytes());
            }
            let stss = make_full_box(b"stss", 0, 0, &stss_body);

            let mut stbl_body = Vec::new();
            for b in [&stsd, &stts, &stsc, &stsz, &stco, &stss] {
                stbl_body.extend_from_slice(b);
            }
            let stbl = make_box(b"stbl", &stbl_body);
            let minf = make_box(b"minf", &stbl);
            let mut mdia_body = Vec::new();
            for b in [&mdhd, &hdlr, &minf] {
                mdia_body.extend_from_slice(b);
            }
            let mdia = make_box(b"mdia", &mdia_body);

            let mut trak_body = tkhd;
            if let Some(edts) = edts_opt {
                trak_body.extend_from_slice(&edts);
            }
            trak_body.extend_from_slice(&mdia);
            make_box(b"trak", &trak_body)
        };

        let mut mdat_payload = Vec::new();
        for s in samples {
            mdat_payload.extend_from_slice(s);
        }
        let dummy_moov = make_box(b"moov", &[mvhd.as_slice(), &make_trak(0)].concat());
        let data_off = (ftyp.len() + dummy_moov.len() + 8) as u32;
        let moov = make_box(b"moov", &[mvhd.as_slice(), &make_trak(data_off)].concat());
        let mdat = make_box(b"mdat", &mdat_payload);

        let mut file = ftyp;
        file.extend_from_slice(&moov);
        file.extend_from_slice(&mdat);
        file
    };

    let pkt_refs: Vec<&[u8]> = ivf_pkts.iter().map(Vec::as_slice).collect();
    let identity = [P, 0, 0, 0, P, 0, 0, 0, W];

    // Referans: elst'siz ve birim matrisli 3 ham kare (KEY, INTER1, INTER2).
    let base_avis = build_avis(&pkt_refs, 432, 240, identity, 30, 30, 10, None, &[1]);
    let mut base_dec = AvifSequenceDecoder::new(&base_avis).unwrap();
    let mut ref_frames = Vec::new();
    while let Some(f) = base_dec.next_frame().unwrap() {
        ref_frames.push(f.image);
    }
    assert_eq!(ref_frames.len(), 3);
    assert_eq!((ref_frames[0].width(), ref_frames[0].height()), (432, 240));

    // 1. `tkhd` 90° CCW (`[0, -P, P, 0]`, ötelemeli `x=100, y=200`): boyut 432x240 -> 240x432
    //    ve piksel (x=0, y=0) == eski (x=431, y=0).
    let rot90_ccw = [0, N, 0, P, 0, 0, 100, 200, W];
    let avis_rot90 = build_avis(&pkt_refs[..1], 432, 240, rot90_ccw, 30, 30, 10, None, &[1]);
    let mut dec_rot90 = AvifSequenceDecoder::new(&avis_rot90).unwrap();
    assert_eq!(
        (dec_rot90.info().width, dec_rot90.info().height),
        (240, 432)
    );
    let f_rot90 = dec_rot90.next_frame().unwrap().unwrap();
    assert_eq!((f_rot90.image.width(), f_rot90.image.height()), (240, 432));
    let ref_y = ref_frames[0].y_plane_u8().unwrap();
    let rot90_y = f_rot90.image.y_plane_u8().unwrap();
    assert_eq!(rot90_y[0], ref_y[431]);
    assert_eq!(rot90_y[431 * 240 + 239], ref_y[239 * 432]);

    // 2. `tkhd` 180° (`[-P, 0, 0, -P]`): boyut 432x240, piksel (0,0) == eski (431, 239).
    let rot180 = [N, 0, 0, 0, N, 0, 0, 0, W];
    let avis_rot180 = build_avis(&pkt_refs[..1], 432, 240, rot180, 30, 30, 10, None, &[1]);
    let f_rot180 = AvifSequenceDecoder::new(&avis_rot180)
        .unwrap()
        .next_frame()
        .unwrap()
        .unwrap();
    let rot180_y = f_rot180.image.y_plane_u8().unwrap();
    assert_eq!(rot180_y[0], ref_y[239 * 432 + 431]);

    // 3. `tkhd` yatay ayna (`[-P, 0, 0, P]`): piksel (0,0) == eski (431, 0).
    let mir_h = [N, 0, 0, 0, P, 0, 0, 0, W];
    let avis_mir = build_avis(&pkt_refs[..1], 432, 240, mir_h, 30, 30, 10, None, &[1]);
    let f_mir = AvifSequenceDecoder::new(&avis_mir)
        .unwrap()
        .next_frame()
        .unwrap()
        .unwrap();
    let mir_y = f_mir.image.y_plane_u8().unwrap();
    assert_eq!(mir_y[0], ref_y[431]);
    assert_eq!(mir_y[431], ref_y[0]);

    // 4. Geçersiz (ölçeklemeli / eğimli) `tkhd` matrisi -> Error::Unsupported.
    let bad_mat = [2 * P, 0, 0, 0, 2 * P, 0, 0, 0, W];
    let avis_bad_mat = build_avis(&pkt_refs[..1], 432, 240, bad_mat, 30, 30, 10, None, &[1]);
    assert!(matches!(
        AvifSequenceDecoder::new(&avis_bad_mat),
        Err(Error::Unsupported(_))
    ));

    // 5. `flags = 0` `elst`: 5 birimlik boş düzenleme (`media_time = -1`) + `media_time = 10`,
    //    `segment_duration = 20` (farklı `movie_ts = 60`, `media_ts = 30` ölçeklemesi:
    //    movie 10 -> media 5, movie 40 -> media 20).
    //    Ham 0 (KEY, t=0..10) gizli pre-roll olarak çözülüp Ham 1 (INTER1) ve Ham 2 (INTER2) sunulmalı!
    let elst_entries = [(10u32, -1i32, 1i16, 0i16), (40u32, 10i32, 1i16, 0i16)];
    let avis_elst = build_avis(
        &pkt_refs,
        432,
        240,
        identity,
        60,
        30,
        10,
        Some((0, &elst_entries)),
        &[1],
    );
    let mut seq = AvifSequenceDecoder::new(&avis_elst).expect("elst AVIS açılmalı");
    assert_eq!(seq.info().frame_count, 2);
    assert_eq!(seq.info().timescale, 30);
    assert_eq!(seq.info().duration_in_timescale, 25);

    let f0 = seq.next_frame().unwrap().expect("sunum karesi 0 (INTER1)");
    assert_eq!(f0.index, 0);
    assert_eq!(f0.pts_in_timescale, 5);
    assert_eq!(f0.duration_in_timescale, 10);
    assert!(!f0.is_keyframe);
    assert_eq!(f0.image, ref_frames[1]);

    let f1 = seq.next_frame().unwrap().expect("sunum karesi 1 (INTER2)");
    assert_eq!(f1.index, 1);
    assert_eq!(f1.pts_in_timescale, 15);
    assert_eq!(f1.duration_in_timescale, 10);
    assert!(!f1.is_keyframe);
    assert_eq!(f1.image, ref_frames[2]);
    assert!(seq.next_frame().unwrap().is_none());

    // Geriye ve ileriye `nth_frame` de pre-roll ile birebir aynı pikselleri vermeli.
    let seek0 = seq
        .nth_frame(0)
        .expect("nth_frame(0) pre-roll ile çözülmeli");
    assert_eq!(seek0.pts_in_timescale, 5);
    assert_eq!(seek0.image, ref_frames[1]);
    let seek1 = seq.nth_frame(1).expect("nth_frame(1) çözülmeli");
    assert_eq!(seek1.pts_in_timescale, 15);
    assert_eq!(seek1.image, ref_frames[2]);
}

/// `6.F.3`: AVIS `elst` `O((S + E + K) log S)` aralık ağacı (128k × 128k `< 250 ms`) ve
/// `lcm(media_ts, movie_ts)` kesin kesişimi (`movie_ts=1000, media_ts=30` üzerinde `16 ms` / `17 ms`).
#[test]
fn avif_sequence_elst_128k_perf_and_sub_media_tick_precision() {
    use std::time::Instant;

    const P: i32 = 0x0001_0000;
    const W: i32 = 0x4000_0000;
    let identity = [P, 0, 0, 0, P, 0, 0, 0, W];

    let build_synthetic_avis = |sample_count: u32,
                                movie_ts: u32,
                                media_ts: u32,
                                sample_delta: u32,
                                elst_entries: &[(u32, i32)]|
     -> Vec<u8> {
        let ftyp = make_ftyp(b"avis", &[*b"msf1", *b"avif", *b"miaf"]);

        let mut mvhd_body = vec![0u8; 96];
        mvhd_body[8..12].copy_from_slice(&movie_ts.to_be_bytes());
        mvhd_body[12..16].copy_from_slice(&sample_count.saturating_mul(sample_delta).to_be_bytes());
        let mvhd = make_full_box(b"mvhd", 0, 0, &mvhd_body);

        let make_trak = |chunk_offset: u32| -> Vec<u8> {
            let mut tkhd_body = vec![0u8; 80];
            tkhd_body[8..12].copy_from_slice(&1u32.to_be_bytes());
            tkhd_body[16..20]
                .copy_from_slice(&sample_count.saturating_mul(sample_delta).to_be_bytes());
            for (i, &val) in identity.iter().enumerate() {
                tkhd_body[36 + i * 4..40 + i * 4].copy_from_slice(&val.to_be_bytes());
            }
            tkhd_body[72..76].copy_from_slice(&(1u32 << 16).to_be_bytes());
            tkhd_body[76..80].copy_from_slice(&(1u32 << 16).to_be_bytes());
            let tkhd = make_full_box(b"tkhd", 0, 3, &tkhd_body);

            let mut elst_body = Vec::with_capacity(4 + elst_entries.len() * 12);
            elst_body.extend_from_slice(&(elst_entries.len() as u32).to_be_bytes());
            for &(sd, mt) in elst_entries {
                elst_body.extend_from_slice(&sd.to_be_bytes());
                elst_body.extend_from_slice(&mt.to_be_bytes());
                elst_body.extend_from_slice(&1i16.to_be_bytes());
                elst_body.extend_from_slice(&0i16.to_be_bytes());
            }
            let edts = make_box(b"edts", &make_full_box(b"elst", 0, 0, &elst_body));

            let mut mdhd_body = vec![0u8; 20];
            mdhd_body[8..12].copy_from_slice(&media_ts.to_be_bytes());
            mdhd_body[12..16]
                .copy_from_slice(&sample_count.saturating_mul(sample_delta).to_be_bytes());
            let mdhd = make_full_box(b"mdhd", 0, 0, &mdhd_body);

            let mut hdlr_body = Vec::new();
            hdlr_body.extend_from_slice(&0u32.to_be_bytes());
            hdlr_body.extend_from_slice(b"pict");
            hdlr_body.extend_from_slice(&[0u8; 13]);
            let hdlr = make_full_box(b"hdlr", 0, 0, &hdlr_body);

            let mut av01_payload = vec![0u8; 78];
            av01_payload[24..26].copy_from_slice(&1u16.to_be_bytes());
            av01_payload[26..28].copy_from_slice(&1u16.to_be_bytes());
            av01_payload.extend_from_slice(&make_box(b"av1C", &[0x81, 0x00, 0x0C, 0x00]));
            let mut stsd_body = 1u32.to_be_bytes().to_vec();
            stsd_body.extend_from_slice(&make_box(b"av01", &av01_payload));
            let stsd = make_full_box(b"stsd", 0, 0, &stsd_body);

            let mut stts_body = Vec::new();
            stts_body.extend_from_slice(&1u32.to_be_bytes());
            stts_body.extend_from_slice(&sample_count.to_be_bytes());
            stts_body.extend_from_slice(&sample_delta.to_be_bytes());
            let stts = make_full_box(b"stts", 0, 0, &stts_body);

            let mut stsc_body = Vec::new();
            stsc_body.extend_from_slice(&1u32.to_be_bytes());
            stsc_body.extend_from_slice(&1u32.to_be_bytes());
            stsc_body.extend_from_slice(&1u32.to_be_bytes());
            stsc_body.extend_from_slice(&1u32.to_be_bytes());
            let stsc = make_full_box(b"stsc", 0, 0, &stsc_body);

            // Sabit `sample_size` ile 128k örnek aynı `SAMPLE_1X1_AV1_OBU` yığınını paylaşır.
            let mut stsz_body = Vec::new();
            stsz_body.extend_from_slice(&(SAMPLE_1X1_AV1_OBU.len() as u32).to_be_bytes());
            stsz_body.extend_from_slice(&sample_count.to_be_bytes());
            let stsz = make_full_box(b"stsz", 0, 0, &stsz_body);

            let mut stco_body = Vec::with_capacity(4 + (sample_count as usize) * 4);
            stco_body.extend_from_slice(&sample_count.to_be_bytes());
            for _ in 0..sample_count {
                stco_body.extend_from_slice(&chunk_offset.to_be_bytes());
            }
            let stco = make_full_box(b"stco", 0, 0, &stco_body);

            let mut stbl_body = Vec::new();
            for b in [&stsd, &stts, &stsc, &stsz, &stco] {
                stbl_body.extend_from_slice(b);
            }
            let stbl = make_box(b"stbl", &stbl_body);
            let minf = make_box(b"minf", &stbl);
            let mut mdia_body = Vec::new();
            for b in [&mdhd, &hdlr, &minf] {
                mdia_body.extend_from_slice(b);
            }
            let mdia = make_box(b"mdia", &mdia_body);

            let mut trak_body = tkhd;
            trak_body.extend_from_slice(&edts);
            trak_body.extend_from_slice(&mdia);
            make_box(b"trak", &trak_body)
        };

        let dummy_moov = make_box(b"moov", &[mvhd.as_slice(), &make_trak(0)].concat());
        let data_off = (ftyp.len() + dummy_moov.len() + 8) as u32;
        let moov = make_box(b"moov", &[mvhd.as_slice(), &make_trak(data_off)].concat());
        let mdat = make_box(b"mdat", SAMPLE_1X1_AV1_OBU);

        let mut file = ftyp;
        file.extend_from_slice(&moov);
        file.extend_from_slice(&mdat);
        file
    };

    // 1. Alt-medya-birimi kesin kesişimi: `movie_ts = 1000`, `media_ts = 30`, 1 örnek `[0, 1)`.
    //    `segment_duration = 16` (16 ms) ve `17` (17 ms) her ikisi de `1` kare üretmeli ve
    //    `eff_ts = 3000` ölçeğinde tam `0.016` ve `0.017` saniye döndürmelidir.
    let avis_16ms = build_synthetic_avis(1, 1000, 30, 1, &[(16, 0)]);
    let mut dec_16 = AvifSequenceDecoder::new(&avis_16ms).expect("16 ms elst geçerli olmalı");
    assert_eq!(dec_16.info().frame_count, 1);
    assert_eq!(dec_16.info().timescale, 3000);
    assert_eq!(dec_16.info().duration_in_timescale, 48);
    assert!((dec_16.info().duration_sec - 0.016).abs() < 1e-12);
    let f16 = dec_16.next_frame().unwrap().unwrap();
    assert_eq!(f16.duration_in_timescale, 48);
    assert!((f16.duration_sec - 0.016).abs() < 1e-12);

    let avis_17ms = build_synthetic_avis(1, 1000, 30, 1, &[(17, 0)]);
    let mut dec_17 = AvifSequenceDecoder::new(&avis_17ms).expect("17 ms elst geçerli olmalı");
    assert_eq!(dec_17.info().frame_count, 1);
    assert_eq!(dec_17.info().timescale, 3000);
    assert_eq!(dec_17.info().duration_in_timescale, 51);
    assert!((dec_17.info().duration_sec - 0.017).abs() < 1e-12);
    let f17 = dec_17.next_frame().unwrap().unwrap();
    assert_eq!(f17.duration_in_timescale, 51);
    assert!((f17.duration_sec - 0.017).abs() < 1e-12);

    // 2. Tek girdili, kırpmasız ve ötelemesiz `flags = 0` `elst` (`media_time = 0`,
    //    `segment_duration` tam medya süresine eşit: `5/30 s = 100/600 s`):
    //    `lcm(30, 600) = 600` yerine ham `mdhd.timescale == 30` korunmalıdır (`6.G.2`).
    let avis_exact_full = build_synthetic_avis(5, 600, 30, 1, &[(100, 0)]);
    let info_full = av1oxide::avif::parse_sequence_info(&avis_exact_full)
        .expect("kırpmasız flags=0 elst AVIS açılmalı");
    assert_eq!(info_full.frame_count, 5);
    assert_eq!(info_full.timescale, 30);
    assert_eq!(info_full.duration_in_timescale, 5);

    // 2b. Tam medya birimi hizalı kırpmalı `elst` (`movie_ts = 1000`, `media_ts = 30`,
    //     5 kare, `segment_duration = 100` ms = tam `3 / 30 s`): tüm sunum PTS/süreleri
    //     `m_mul = 100`'e tam bölündüğü için `timescale == 30`, `dur == 3`, kareler
    //     `(0, 1), (1, 1), (2, 1)` olarak sadeleştirilmelidir (`6.H.5`).
    let avis_100ms = build_synthetic_avis(5, 1000, 30, 1, &[(100, 0)]);
    let mut dec_100 = AvifSequenceDecoder::new(&avis_100ms).expect("100 ms elst geçerli olmalı");
    assert_eq!(dec_100.info().frame_count, 3);
    assert_eq!(dec_100.info().timescale, 30);
    assert_eq!(dec_100.info().duration_in_timescale, 3);
    for expected_pts in 0..3u64 {
        let f = dec_100
            .next_frame()
            .unwrap()
            .expect("100 ms kırpma karesi gelmeli");
        assert_eq!(f.pts_in_timescale, expected_pts);
        assert_eq!(f.duration_in_timescale, 1);
    }
    assert!(dec_100.next_frame().unwrap().is_none());

    // 3. 128 000 örnek × 128 000 `elst` girdisi: `O((S + E + K) log S)` aralık ağacı ile
    //    `parse_sequence_info` 250 ms altında (pratikte birkaç ms) tamamlanmalıdır.
    let mut edits_128k: Vec<(u32, i32)> = Vec::with_capacity(128_000);
    for i in 0..128_000u32 {
        edits_128k.push((1, i as i32));
    }
    let avis_128k = build_synthetic_avis(128_000, 1000, 1000, 1, &edits_128k);
    let t0 = Instant::now();
    let info_128k = av1oxide::avif::parse_sequence_info(&avis_128k)
        .expect("128k × 128k AVIS parse_sequence_info başarılı olmalı");
    let elapsed = t0.elapsed();
    assert_eq!(info_128k.frame_count, 128_000);
    assert!(
        elapsed.as_millis() < 250,
        "128k × 128k elst ayrıştırması < 250 ms olmalı, ölçülen: {elapsed:?}"
    );

    // 4. `6.I.3` (`P3-2`): Geniş ortak-asal zaman tabanlarında (`movie_ts = 2_147_483_629`,
    //    `media_ts = 2_147_483_587`, 5 kare × 2 s) `elst` v1 ile `3 s` ve `5 s` kırpmaları:
    //    `eff_ts = lcm(movie_ts, media_ts)` uzayında `5 s` toplam süre `u64::MAX` sınırını aşsa da
    //    `m_mul` sadeleştirmesi `u128` uzayında `u64::try_from` öncesinde yapıldığından
    //    `media_ts` ölçeğinde (`10_737_417_935` tik, `3` kare) taşmasız açılmalıdır.
    let build_wide_elst_v1_avis =
        |movie_ts: u32, media_ts: u32, sample_delta: u32, seg_dur_movie: u64| -> Vec<u8> {
            let ftyp = make_ftyp(b"avis", &[*b"msf1", *b"avif", *b"miaf"]);
            let sample_count = 5u32;
            let total_media_dur = u64::from(sample_count) * u64::from(sample_delta);

            let mut mvhd_body = vec![0u8; 108];
            mvhd_body[16..20].copy_from_slice(&movie_ts.to_be_bytes());
            mvhd_body[20..28].copy_from_slice(&seg_dur_movie.to_be_bytes());
            let mvhd = make_full_box(b"mvhd", 1, 0, &mvhd_body);

            let make_trak = |chunk_offset: u32| -> Vec<u8> {
                let mut tkhd_body = vec![0u8; 92];
                tkhd_body[16..20].copy_from_slice(&1u32.to_be_bytes());
                tkhd_body[24..32].copy_from_slice(&seg_dur_movie.to_be_bytes());
                for (i, &val) in identity.iter().enumerate() {
                    tkhd_body[48 + i * 4..52 + i * 4].copy_from_slice(&val.to_be_bytes());
                }
                tkhd_body[84..88].copy_from_slice(&(1u32 << 16).to_be_bytes());
                tkhd_body[88..92].copy_from_slice(&(1u32 << 16).to_be_bytes());
                let tkhd = make_full_box(b"tkhd", 1, 3, &tkhd_body);

                let mut elst_body = Vec::with_capacity(24);
                elst_body.extend_from_slice(&1u32.to_be_bytes());
                elst_body.extend_from_slice(&seg_dur_movie.to_be_bytes());
                elst_body.extend_from_slice(&0i64.to_be_bytes());
                elst_body.extend_from_slice(&1i16.to_be_bytes());
                elst_body.extend_from_slice(&0i16.to_be_bytes());
                let edts = make_box(b"edts", &make_full_box(b"elst", 1, 0, &elst_body));

                let mut mdhd_body = vec![0u8; 32];
                mdhd_body[16..20].copy_from_slice(&media_ts.to_be_bytes());
                mdhd_body[20..28].copy_from_slice(&total_media_dur.to_be_bytes());
                let mdhd = make_full_box(b"mdhd", 1, 0, &mdhd_body);

                let mut hdlr_body = Vec::new();
                hdlr_body.extend_from_slice(&0u32.to_be_bytes());
                hdlr_body.extend_from_slice(b"pict");
                hdlr_body.extend_from_slice(&[0u8; 13]);
                let hdlr = make_full_box(b"hdlr", 0, 0, &hdlr_body);

                let mut av01_payload = vec![0u8; 78];
                av01_payload[24..26].copy_from_slice(&1u16.to_be_bytes());
                av01_payload[26..28].copy_from_slice(&1u16.to_be_bytes());
                av01_payload.extend_from_slice(&make_box(b"av1C", &[0x81, 0x00, 0x0C, 0x00]));
                let mut stsd_body = 1u32.to_be_bytes().to_vec();
                stsd_body.extend_from_slice(&make_box(b"av01", &av01_payload));
                let stsd = make_full_box(b"stsd", 0, 0, &stsd_body);

                let mut stts_body = Vec::new();
                stts_body.extend_from_slice(&1u32.to_be_bytes());
                stts_body.extend_from_slice(&sample_count.to_be_bytes());
                stts_body.extend_from_slice(&sample_delta.to_be_bytes());
                let stts = make_full_box(b"stts", 0, 0, &stts_body);

                let mut stsc_body = Vec::new();
                stsc_body.extend_from_slice(&1u32.to_be_bytes());
                stsc_body.extend_from_slice(&1u32.to_be_bytes());
                stsc_body.extend_from_slice(&1u32.to_be_bytes());
                stsc_body.extend_from_slice(&1u32.to_be_bytes());
                let stsc = make_full_box(b"stsc", 0, 0, &stsc_body);

                let mut stsz_body = Vec::new();
                stsz_body.extend_from_slice(&(SAMPLE_1X1_AV1_OBU.len() as u32).to_be_bytes());
                stsz_body.extend_from_slice(&sample_count.to_be_bytes());
                let stsz = make_full_box(b"stsz", 0, 0, &stsz_body);

                let mut stco_body = Vec::with_capacity(4 + (sample_count as usize) * 4);
                stco_body.extend_from_slice(&sample_count.to_be_bytes());
                for _ in 0..sample_count {
                    stco_body.extend_from_slice(&chunk_offset.to_be_bytes());
                }
                let stco = make_full_box(b"stco", 0, 0, &stco_body);

                let mut stbl_body = Vec::new();
                for b in [&stsd, &stts, &stsc, &stsz, &stco] {
                    stbl_body.extend_from_slice(b);
                }
                let stbl = make_box(b"stbl", &stbl_body);
                let minf = make_box(b"minf", &stbl);
                let mut mdia_body = Vec::new();
                for b in [&mdhd, &hdlr, &minf] {
                    mdia_body.extend_from_slice(b);
                }
                let mdia = make_box(b"mdia", &mdia_body);

                let mut trak_body = tkhd;
                trak_body.extend_from_slice(&edts);
                trak_body.extend_from_slice(&mdia);
                make_box(b"trak", &trak_body)
            };

            let dummy_moov = make_box(b"moov", &[mvhd.as_slice(), &make_trak(0)].concat());
            let data_off = (ftyp.len() + dummy_moov.len() + 8) as u32;
            let moov = make_box(b"moov", &[mvhd.as_slice(), &make_trak(data_off)].concat());
            let mdat = make_box(b"mdat", SAMPLE_1X1_AV1_OBU);

            let mut file = ftyp;
            file.extend_from_slice(&moov);
            file.extend_from_slice(&mdat);
            file
        };

    let movie_ts = 2_147_483_629u32;
    let media_ts = 2_147_483_587u32;
    let two_sec_media = media_ts * 2; // 4_294_967_174 <= u32::MAX

    // 3 saniyelik elst: 2 kare (2 s + 1 s kırpılmış), toplam 3 * media_ts = 6_442_450_761 tik.
    let avis_3s =
        build_wide_elst_v1_avis(movie_ts, media_ts, two_sec_media, u64::from(movie_ts) * 3);
    let info_3s =
        av1oxide::avif::parse_sequence_info(&avis_3s).expect("3s geniş zaman tabanlı AVIS");
    assert_eq!(info_3s.frame_count, 2);
    assert_eq!(info_3s.timescale, u64::from(media_ts));
    assert_eq!(info_3s.duration_in_timescale, u64::from(media_ts) * 3);

    // 5 saniyelik elst: 3 kare (2 s + 2 s + 1 s kırpılmış), toplam 5 * media_ts = 10_737_417_935 tik.
    // (`10_737_417_935 * 2_147_483_629 > u64::MAX`, ancak `m_mul` sadeleştirmesi sonrası `u64`'e sığar!)
    let avis_5s =
        build_wide_elst_v1_avis(movie_ts, media_ts, two_sec_media, u64::from(movie_ts) * 5);
    let mut dec_5s =
        AvifSequenceDecoder::new(&avis_5s).expect("5s geniş zaman tabanlı AVIS taşmasız açılmalı");
    assert_eq!(dec_5s.info().frame_count, 3);
    assert_eq!(dec_5s.info().timescale, u64::from(media_ts));
    assert_eq!(dec_5s.info().duration_in_timescale, 10_737_417_935);
    let f0 = dec_5s.next_frame().unwrap().unwrap();
    assert_eq!(f0.pts_in_timescale, 0);
    assert_eq!(f0.duration_in_timescale, u64::from(two_sec_media));
    let f1 = dec_5s.next_frame().unwrap().unwrap();
    assert_eq!(f1.pts_in_timescale, u64::from(two_sec_media));
    assert_eq!(f1.duration_in_timescale, u64::from(two_sec_media));
    let f2 = dec_5s.next_frame().unwrap().unwrap();
    assert_eq!(f2.pts_in_timescale, u64::from(two_sec_media) * 2);
    assert_eq!(f2.duration_in_timescale, u64::from(media_ts));
    assert!(dec_5s.next_frame().unwrap().is_none());
}
