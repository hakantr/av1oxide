//! `av1oxide::avif` (`.avif` ISOBMFF/HEIF/MIAF görüntü okuyucu) entegrasyon testleri.

#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use av1oxide::avif::{
    AvifDecodeConfig, CleanAperture, ColorInfo, ContentLightLevel, MasteringDisplay,
    PixelAspectRatio, decode, decode_rgb8, decode_rgba8, decode_with, parse_info, probe,
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
    assert_eq!(img.y_plane(), &[190]);
    let u0 = img.u_plane()[0];
    let v0 = img.v_plane()[0];
    assert_eq!(u0, 1);
    let rgb8 = img.to_rgb8().unwrap();
    assert_eq!(rgb8.pixels, vec![v0 as u8, 190, u0 as u8]);
    let rgba8 = img.to_rgba8().unwrap();
    assert_eq!(rgba8.pixels, vec![v0 as u8, 190, u0 as u8, 255]);
    let rgb16 = img.to_rgb16().unwrap();
    assert_eq!(rgb16.pixels, vec![v0, 190, u0]);
    let rgba16 = img.to_rgba16().unwrap();
    assert_eq!(rgba16.pixels, vec![v0, 190, u0, 255]);
    assert_eq!(img.y_row(0), Some(&[190u16][..]));
    assert_eq!(img.u_row(0), Some(&[u0][..]));
    assert_eq!(img.v_row(0), Some(&[v0][..]));
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
    assert_eq!(img_t1.alpha_plane(), Some(&[190u16][..]));
    assert_eq!(img_t1.alpha_row(0), Some(&[190u16][..]));
    let rgba = img_t1.to_rgba8().unwrap();
    assert_eq!(rgba.pixels[3], 190);

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
        rot90_img.y_row(0).unwrap()[0],
        base_img.y_row(0).unwrap()[(base_w - 1) as usize]
    );

    // 2. irot = 2 (180° CCW): boyut (base_w, base_h), dst(0, 0) == src(base_w - 1, base_h - 1)
    let rot180_avif = build_transformed_avif(&odd_obu, base_w, base_h, None, Some(2), None, false);
    let rot180_img = decode(&rot180_avif).unwrap();
    assert_eq!(
        rot180_img.y_row(0).unwrap()[0],
        base_img.y_row((base_h - 1) as usize).unwrap()[(base_w - 1) as usize]
    );

    // 3. irot = 3 (270° CCW): boyut (base_h, base_w), dst(0, 0) == src(0, base_h - 1)
    let rot270_avif = build_transformed_avif(&odd_obu, base_w, base_h, None, Some(3), None, false);
    let rot270_img = decode(&rot270_avif).unwrap();
    assert_eq!(
        rot270_img.y_row(0).unwrap()[0],
        base_img.y_row((base_h - 1) as usize).unwrap()[0]
    );

    // 4. imir axis = 0 (üst-alt) ve axis = 1 (sol-sağ)
    let imir0_avif = build_transformed_avif(&odd_obu, base_w, base_h, None, None, Some(0), false);
    let imir0_img = decode(&imir0_avif).unwrap();
    assert_eq!(
        imir0_img.y_row(0).unwrap()[0],
        base_img.y_row((base_h - 1) as usize).unwrap()[0]
    );

    let imir1_avif = build_transformed_avif(&odd_obu, base_w, base_h, None, None, Some(1), false);
    let imir1_img = decode(&imir1_avif).unwrap();
    assert_eq!(
        imir1_img.y_row(0).unwrap()[0],
        base_img.y_row(0).unwrap()[(base_w - 1) as usize]
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
