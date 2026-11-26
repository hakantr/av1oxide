//! `av1oxide` `.avif` görüntü inceleme ve çözme örneği (`v0.4.0`).
//!
//! Çalıştırma:
//!   cargo run --release -p av1oxide --example decode_avif
//!   cargo run --release -p av1oxide --example decode_avif -- <girdi.avif>

#![forbid(unsafe_code)]

use std::env;
use std::fs;

use av1oxide::BitDepth;
use av1oxide::avif::{decode, decode_rgba8, parse_info, probe};

/// Gömülü 1×1, 8-bit, 4:2:0 AV1 anahtar kare OBU yükü (TD + Sequence Header + Keyframe).
const SAMPLE_AV1_OBU: &[u8] = &[
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

fn make_full_box(box_type: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(4 + body.len());
    payload.push(version);
    payload.push(((flags >> 16) & 0xFF) as u8);
    payload.push(((flags >> 8) & 0xFF) as u8);
    payload.push((flags & 0xFF) as u8);
    payload.extend_from_slice(body);
    make_box(box_type, &payload)
}

/// Gömülü OBU yükünü standart bir ISOBMFF / AVIF kapsayıcısına sarar.
fn build_embedded_avif() -> Vec<u8> {
    let mut ftyp_body = Vec::new();
    ftyp_body.extend_from_slice(b"avif");
    ftyp_body.extend_from_slice(&0u32.to_be_bytes());
    ftyp_body.extend_from_slice(b"avifmif1miaf");
    let ftyp = make_box(b"ftyp", &ftyp_body);

    let mut hdlr_body = Vec::new();
    hdlr_body.extend_from_slice(&0u32.to_be_bytes());
    hdlr_body.extend_from_slice(b"pict");
    hdlr_body.extend_from_slice(&[0u8; 13]);
    let hdlr = make_full_box(b"hdlr", 0, 0, &hdlr_body);

    let pitm = make_full_box(b"pitm", 0, 0, &1u16.to_be_bytes());

    let mut infe_body = Vec::new();
    infe_body.extend_from_slice(&1u16.to_be_bytes());
    infe_body.extend_from_slice(&0u16.to_be_bytes());
    infe_body.extend_from_slice(b"av01");
    infe_body.push(0);
    let infe = make_full_box(b"infe", 2, 0, &infe_body);

    let mut iinf_body = Vec::new();
    iinf_body.extend_from_slice(&1u16.to_be_bytes());
    iinf_body.extend_from_slice(&infe);
    let iinf = make_full_box(b"iinf", 0, 0, &iinf_body);

    let mut ispe_body = Vec::new();
    ispe_body.extend_from_slice(&1u32.to_be_bytes());
    ispe_body.extend_from_slice(&1u32.to_be_bytes());
    let ispe = make_full_box(b"ispe", 0, 0, &ispe_body);

    let pixi = make_full_box(b"pixi", 0, 0, &[3, 8, 8, 8]);
    let av1c = make_box(b"av1C", &[0x81, 0x00, 0x0C, 0x00]);
    let colr = make_box(b"colr", &[b'n', b'c', b'l', b'x', 0, 1, 0, 13, 0, 6, 0x80]);

    let mut ipco_body = Vec::new();
    ipco_body.extend_from_slice(&ispe);
    ipco_body.extend_from_slice(&pixi);
    ipco_body.extend_from_slice(&av1c);
    ipco_body.extend_from_slice(&colr);
    let ipco = make_box(b"ipco", &ipco_body);

    let mut ipma_body = Vec::new();
    ipma_body.extend_from_slice(&1u32.to_be_bytes());
    ipma_body.extend_from_slice(&1u16.to_be_bytes());
    ipma_body.push(4);
    ipma_body.push(1);
    ipma_body.push(2);
    ipma_body.push(0x80 | 3);
    ipma_body.push(4);
    let ipma = make_full_box(b"ipma", 0, 0, &ipma_body);

    let mut iprp_body = Vec::new();
    iprp_body.extend_from_slice(&ipco);
    iprp_body.extend_from_slice(&ipma);
    let iprp = make_box(b"iprp", &iprp_body);

    let idat = make_box(b"idat", SAMPLE_AV1_OBU);

    // iloc v1: construction_method = 1 (idat ofseti = 0, uzunluk = SAMPLE_AV1_OBU.len())
    let mut iloc_body = Vec::new();
    iloc_body.push(0x44); // offset_size=4, length_size=4
    iloc_body.push(0x00); // base_offset_size=0, index_size=0
    iloc_body.extend_from_slice(&1u16.to_be_bytes()); // item_count = 1
    iloc_body.extend_from_slice(&1u16.to_be_bytes()); // item_id = 1
    iloc_body.extend_from_slice(&1u16.to_be_bytes()); // construction_method = 1 (idat)
    iloc_body.extend_from_slice(&0u16.to_be_bytes()); // data_reference_index = 0
    iloc_body.extend_from_slice(&1u16.to_be_bytes()); // extent_count = 1
    iloc_body.extend_from_slice(&0u32.to_be_bytes()); // extent_offset = 0
    iloc_body.extend_from_slice(&(SAMPLE_AV1_OBU.len() as u32).to_be_bytes());
    let iloc = make_full_box(b"iloc", 1, 0, &iloc_body);

    let mut meta_children = Vec::new();
    meta_children.extend_from_slice(&hdlr);
    meta_children.extend_from_slice(&pitm);
    meta_children.extend_from_slice(&iinf);
    meta_children.extend_from_slice(&iprp);
    meta_children.extend_from_slice(&iloc);
    meta_children.extend_from_slice(&idat);
    let meta = make_full_box(b"meta", 0, 0, &meta_children);

    let mut file = Vec::new();
    file.extend_from_slice(&ftyp);
    file.extend_from_slice(&meta);
    file
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let (source_label, avif_bytes, is_embedded) = if let Some(path) = args.get(1) {
        let data = fs::read(path).expect("AVIF dosyası okunamadı");
        (path.clone(), data, false)
    } else {
        (
            String::from("<gömülü 1x1 AVIF>"),
            build_embedded_avif(),
            true,
        )
    };

    assert!(probe(&avif_bytes), "geçerli AVIF ftyp imzası bekleniyor");
    let info = parse_info(&avif_bytes).expect("AVIF meta kutusu ayrıştırılmalı");
    println!(
        "AVIF başlığı ({source_label}): {}x{}, {:?}, {:?}, alfa={}, renk=(pri={}, trc={}, mtrx={}, full={})",
        info.width,
        info.height,
        info.bit_depth,
        info.chroma_format,
        info.has_alpha,
        info.color.color_primaries,
        info.color.transfer_characteristics,
        info.color.matrix_coefficients,
        info.color.full_range
    );

    let img = decode(&avif_bytes).expect("AVIF düzlemsel görüntü çözülmeli");
    let rgba = decode_rgba8(&avif_bytes).expect("AVIF RGBA8 görüntü çözülmeli");
    let first_y = img.y_plane().get(0).unwrap_or(0);
    println!(
        "Çözülen AVIF: {}x{}, Y[0]={}, RGBA[0..4]={:?}",
        rgba.width,
        rgba.height,
        first_y,
        &rgba.pixels[0..4]
    );

    if is_embedded {
        assert_eq!((info.width, info.height), (1, 1));
        assert_eq!(info.bit_depth, BitDepth::Eight);
        assert_eq!(img.y_plane_u8(), Some(&[190u8][..]));
        assert_eq!(first_y, 190);
        assert_eq!(rgba.pixels.len(), 4);
        assert_eq!(rgba.pixels[3], 255);
    }
}
