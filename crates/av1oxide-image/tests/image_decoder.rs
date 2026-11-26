//! `av1oxide-image` bütünleştirme testleri (`Rgb8`, `Rgba8`, `Rgb16`, `Rgba16`, `ICC`, `Exif`, `Limits`).

#![forbid(unsafe_code)]

use std::io::{self, Cursor, Read};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use av1oxide::avif::AvifDecodeConfig;
use av1oxide_image::{AvifDecoder, decode_avif_bytes, register_decoding_hook};
use image::error::{ImageError, LimitErrorKind};
use image::{ColorType, DynamicImage, ImageDecoder, Limits};

/// 1×1, 8-bit 4:2:0 AV1 anahtar kare OBU'su.
const AV1_1X1_8BIT: &[u8] = &[
    0x12, 0x00, 0x0A, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7D, 0xFC, 0xC0, 0x20, 0x32, 0x17, 0x10,
    0x01, 0x91, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0xD2, 0x16, 0x8C, 0x64,
    0xBE, 0x2E, 0xC7, 0xDA, 0xEE, 0x96,
];

fn make_box(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + payload.len());
    out.extend_from_slice(&((8 + payload.len()) as u32).to_be_bytes());
    out.extend_from_slice(fourcc);
    out.extend_from_slice(payload);
    out
}

fn make_full_box(fourcc: &[u8; 4], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(4 + payload.len());
    body.push(version);
    body.push(((flags >> 16) & 0xff) as u8);
    body.push(((flags >> 8) & 0xff) as u8);
    body.push((flags & 0xff) as u8);
    body.extend_from_slice(payload);
    make_box(fourcc, &body)
}

fn build_synthetic_avif(
    color_obu: &[u8],
    alpha_obu: Option<&[u8]>,
    width: u32,
    height: u32,
    bit_depth: u8,
    icc: Option<&[u8]>,
    exif_tiff: Option<&[u8]>,
) -> Vec<u8> {
    let mut ftyp_payload = Vec::new();
    ftyp_payload.extend_from_slice(b"avif");
    ftyp_payload.extend_from_slice(&0u32.to_be_bytes());
    ftyp_payload.extend_from_slice(b"avifmif1miaf");
    let ftyp = make_box(b"ftyp", &ftyp_payload);

    let mut hdlr_body = vec![0u8; 4];
    hdlr_body.extend_from_slice(b"pict");
    hdlr_body.extend_from_slice(&[0u8; 12]);
    hdlr_body.push(0);
    let hdlr = make_full_box(b"hdlr", 0, 0, &hdlr_body);

    let pitm = make_full_box(b"pitm", 0, 0, &1u16.to_be_bytes());

    let mut infe_entries = Vec::new();
    let mut item_count = 1u16;
    // item 1: color av01
    {
        let mut b = Vec::new();
        b.extend_from_slice(&1u16.to_be_bytes());
        b.extend_from_slice(&0u16.to_be_bytes());
        b.extend_from_slice(b"av01");
        b.push(0);
        infe_entries.extend_from_slice(&make_full_box(b"infe", 2, 0, &b));
    }
    let alpha_id = if alpha_obu.is_some() {
        item_count += 1;
        let id = item_count;
        let mut b = Vec::new();
        b.extend_from_slice(&id.to_be_bytes());
        b.extend_from_slice(&0u16.to_be_bytes());
        b.extend_from_slice(b"av01");
        b.push(0);
        infe_entries.extend_from_slice(&make_full_box(b"infe", 2, 1, &b));
        Some(id)
    } else {
        None
    };
    let exif_id = if exif_tiff.is_some() {
        item_count += 1;
        let id = item_count;
        let mut b = Vec::new();
        b.extend_from_slice(&id.to_be_bytes());
        b.extend_from_slice(&0u16.to_be_bytes());
        b.extend_from_slice(b"Exif");
        b.push(0);
        infe_entries.extend_from_slice(&make_full_box(b"infe", 2, 0, &b));
        Some(id)
    } else {
        None
    };

    let mut iinf_body = Vec::new();
    iinf_body.extend_from_slice(&item_count.to_be_bytes());
    iinf_body.extend_from_slice(&infe_entries);
    let iinf = make_full_box(b"iinf", 0, 0, &iinf_body);

    // ipco properties:
    // 1: ispe, 2: pixi (3ch), 3: av1C, [4: colr prof], [5: auxC]
    let mut ipco_body = Vec::new();
    let mut ispe_b = Vec::new();
    ispe_b.extend_from_slice(&width.to_be_bytes());
    ispe_b.extend_from_slice(&height.to_be_bytes());
    ipco_body.extend_from_slice(&make_full_box(b"ispe", 0, 0, &ispe_b));
    ipco_body.extend_from_slice(&make_full_box(
        b"pixi",
        0,
        0,
        &[3, bit_depth, bit_depth, bit_depth],
    ));
    let mut av1c = [0x81u8, 0x00, 0x0c, 0x00];
    if bit_depth == 10 {
        av1c[2] |= 0x40;
    } else if bit_depth == 12 {
        av1c[1] = 2 << 5; // profile 2
        av1c[2] |= 0x60;
    }
    ipco_body.extend_from_slice(&make_box(b"av1C", &av1c));
    let mut next_prop = 4u8;
    let icc_prop = if let Some(profile) = icc {
        let mut colr_b = Vec::with_capacity(4 + profile.len());
        colr_b.extend_from_slice(b"prof");
        colr_b.extend_from_slice(profile);
        ipco_body.extend_from_slice(&make_box(b"colr", &colr_b));
        let idx = next_prop;
        next_prop += 1;
        Some(idx)
    } else {
        None
    };
    let auxc_prop = if alpha_id.is_some() {
        let mut aux_b = b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha".to_vec();
        aux_b.push(0);
        ipco_body.extend_from_slice(&make_full_box(b"auxC", 0, 0, &aux_b));
        let idx = next_prop;
        Some(idx)
    } else {
        None
    };
    let ipco = make_box(b"ipco", &ipco_body);

    let mut ipma_body = Vec::new();
    let assoc_items = 1u32 + u32::from(alpha_id.is_some());
    ipma_body.extend_from_slice(&assoc_items.to_be_bytes());
    // item 1
    ipma_body.extend_from_slice(&1u16.to_be_bytes());
    let mut color_props = vec![1u8, 2u8, 0x80 | 3u8];
    if let Some(p) = icc_prop {
        color_props.push(p);
    }
    ipma_body.push(color_props.len() as u8);
    ipma_body.extend_from_slice(&color_props);
    if let (Some(aid), Some(auxc_idx)) = (alpha_id, auxc_prop) {
        ipma_body.extend_from_slice(&aid.to_be_bytes());
        let alpha_props = [1u8, 0x80 | 3u8, 0x80 | auxc_idx];
        ipma_body.push(alpha_props.len() as u8);
        ipma_body.extend_from_slice(&alpha_props);
    }
    let ipma = make_full_box(b"ipma", 0, 0, &ipma_body);

    let mut iprp_body = Vec::new();
    iprp_body.extend_from_slice(&ipco);
    iprp_body.extend_from_slice(&ipma);
    let iprp = make_box(b"iprp", &iprp_body);

    let mut iref_children = Vec::new();
    if let Some(aid) = alpha_id {
        let mut b = Vec::new();
        b.extend_from_slice(&aid.to_be_bytes());
        b.extend_from_slice(&1u16.to_be_bytes());
        b.extend_from_slice(&1u16.to_be_bytes());
        iref_children.extend_from_slice(&make_box(b"auxl", &b));
    }
    if let Some(eid) = exif_id {
        let mut b = Vec::new();
        b.extend_from_slice(&eid.to_be_bytes());
        b.extend_from_slice(&1u16.to_be_bytes());
        b.extend_from_slice(&1u16.to_be_bytes());
        iref_children.extend_from_slice(&make_box(b"cdsc", &b));
    }
    let iref = if iref_children.is_empty() {
        Vec::new()
    } else {
        make_full_box(b"iref", 0, 0, &iref_children)
    };

    // Pack all payloads in idat (construction_method = 1) so offsets are independent of meta size!
    let mut idat_payload = Vec::new();
    let mut iloc_entries = Vec::new();
    // item 1
    let off1 = idat_payload.len() as u32;
    idat_payload.extend_from_slice(color_obu);
    iloc_entries.push((1u16, off1, color_obu.len() as u32));
    if let (Some(aid), Some(aobu)) = (alpha_id, alpha_obu) {
        let off = idat_payload.len() as u32;
        idat_payload.extend_from_slice(aobu);
        iloc_entries.push((aid, off, aobu.len() as u32));
    }
    if let (Some(eid), Some(tiff)) = (exif_id, exif_tiff) {
        let off = idat_payload.len() as u32;
        // Exif item payload = 4-byte offset to TIFF header (0) + TIFF bytes
        idat_payload.extend_from_slice(&0u32.to_be_bytes());
        idat_payload.extend_from_slice(tiff);
        iloc_entries.push((eid, off, (4 + tiff.len()) as u32));
    }

    let mut iloc_body = Vec::new();
    iloc_body.push(0x44); // offset_size=4, length_size=4
    iloc_body.push(0x00); // base_offset_size=0, index_size=0
    iloc_body.extend_from_slice(&(iloc_entries.len() as u16).to_be_bytes());
    for (id, off, len) in iloc_entries {
        iloc_body.extend_from_slice(&id.to_be_bytes());
        iloc_body.extend_from_slice(&1u16.to_be_bytes()); // construction_method = 1 (idat)
        iloc_body.extend_from_slice(&0u16.to_be_bytes()); // data_reference_index = 0
        iloc_body.extend_from_slice(&1u16.to_be_bytes()); // extent_count = 1
        iloc_body.extend_from_slice(&off.to_be_bytes());
        iloc_body.extend_from_slice(&len.to_be_bytes());
    }
    let iloc = make_full_box(b"iloc", 1, 0, &iloc_body);
    let idat = make_box(b"idat", &idat_payload);

    let mut meta_children = Vec::new();
    meta_children.extend_from_slice(&hdlr);
    meta_children.extend_from_slice(&pitm);
    meta_children.extend_from_slice(&iinf);
    meta_children.extend_from_slice(&iprp);
    meta_children.extend_from_slice(&iref);
    meta_children.extend_from_slice(&iloc);
    meta_children.extend_from_slice(&idat);
    let meta = make_full_box(b"meta", 0, 0, &meta_children);

    let mut file = Vec::with_capacity(ftyp.len() + meta.len());
    file.extend_from_slice(&ftyp);
    file.extend_from_slice(&meta);
    file
}

#[test]
fn rgb8_rgba8_icc_exif_and_decode_bytes_helper() {
    let fake_icc = b"FAKE_ICC_PROFILE_BYTES_1234";
    let fake_exif = b"MM\x00\x2a\x00\x00\x00\x08";

    // 1. 8-bit RGB + ICC + Exif
    let avif_rgb =
        build_synthetic_avif(AV1_1X1_8BIT, None, 1, 1, 8, Some(fake_icc), Some(fake_exif));
    let mut dec = AvifDecoder::new(Cursor::new(&avif_rgb)).expect("RGB8 decoder");
    assert_eq!(dec.dimensions(), (1, 1));
    assert_eq!(dec.color_type(), ColorType::Rgb8);
    assert_eq!(
        dec.icc_profile().unwrap().as_deref(),
        Some(fake_icc.as_slice())
    );
    assert_eq!(
        dec.exif_metadata().unwrap().as_deref(),
        Some(fake_exif.as_slice())
    );
    let img = DynamicImage::from_decoder(dec).expect("DynamicImage RGB8");
    assert!(matches!(img, DynamicImage::ImageRgb8(_)));
    assert_eq!((img.width(), img.height()), (1, 1));

    let via_helper = decode_avif_bytes(&avif_rgb).expect("decode_avif_bytes");
    assert_eq!(via_helper.to_rgb8(), img.to_rgb8());

    // 2. 8-bit RGBA (with alpha)
    let avif_rgba = build_synthetic_avif(AV1_1X1_8BIT, Some(AV1_1X1_8BIT), 1, 1, 8, None, None);
    let dec_a = AvifDecoder::new(Cursor::new(&avif_rgba)).expect("RGBA8 decoder");
    assert_eq!(dec_a.color_type(), ColorType::Rgba8);
    let img_a = DynamicImage::from_decoder(dec_a).expect("DynamicImage RGBA8");
    assert!(matches!(img_a, DynamicImage::ImageRgba8(_)));
    assert_eq!((img_a.width(), img_a.height()), (1, 1));

    // 3. Hook registration does not panic.
    let _ = register_decoding_hook();
}

#[test]
fn limits_reject_oversized_dimensions_and_allocation() {
    let avif = build_synthetic_avif(AV1_1X1_8BIT, None, 1, 1, 8, None, None);

    // Dimension limit exceeded
    let mut dec = AvifDecoder::new(Cursor::new(&avif)).expect("decoder");
    let mut lim = Limits::no_limits();
    lim.max_image_width = Some(0);
    let err = dec.set_limits(lim).expect_err("genişlik sınırı aşılmalı");
    assert!(matches!(
        err,
        ImageError::Limits(ref e) if e.kind() == LimitErrorKind::DimensionError
    ));

    // Allocation limit exceeded (output is 3 bytes for 1x1 RGB8, limit = 1 byte)
    let mut dec2 = AvifDecoder::new(Cursor::new(&avif)).expect("decoder");
    let mut lim2 = Limits::no_limits();
    lim2.max_alloc = Some(1);
    let err2 = dec2.set_limits(lim2).expect_err("tahsis sınırı aşılmalı");
    assert!(matches!(
        err2,
        ImageError::Limits(ref e) if e.kind() == LimitErrorKind::InsufficientMemory
    ));
}

#[test]
fn real_10bit_and_12bit_avif_scale_to_full_u16_range_when_corpus_present() {
    let base = std::path::Path::new("/home/hakan/github/av1oxide_planlar/avif_dogrulama/kaynaklar");
    if !base.exists() {
        return;
    }

    let cases = [
        (
            base.join("av1-avif/testFiles/Link-U/fox.profile0.10bpc.yuv420.avif"),
            ColorType::Rgb16,
            10u32,
        ),
        (
            base.join("av1-avif/testFiles/Link-U/fox.profile2.12bpc.yuv420.avif"),
            ColorType::Rgb16,
            12u32,
        ),
    ];

    let mut tested = 0usize;
    for (path, expected_ct, depth_bits) in cases {
        if !path.exists() {
            continue;
        }
        let bytes = std::fs::read(&path).expect("test avif okunmalı");
        let av1_img = av1oxide::avif::decode(&bytes).expect("av1oxide decode");
        let dec = AvifDecoder::new(Cursor::new(&bytes)).expect("AvifDecoder");
        let expected_color = if av1_img.has_alpha() {
            ColorType::Rgba16
        } else {
            expected_ct
        };
        assert_eq!(dec.color_type(), expected_color);
        assert_eq!(dec.dimensions(), (av1_img.width(), av1_img.height()));

        let dyn_img = DynamicImage::from_decoder(dec).expect("DynamicImage 16-bit");
        let max_src = (1u32 << depth_bits) - 1;
        let half = max_src / 2;
        if av1_img.has_alpha() {
            let expected = av1_img.to_rgba16().unwrap();
            let actual = dyn_img.into_rgba16();
            for (&s, &a) in expected.pixels.iter().zip(actual.as_raw().iter()) {
                let scaled = ((u32::from(s) * 65_535 + half) / max_src) as u16;
                assert_eq!(a, scaled);
            }
        } else {
            let expected = av1_img.to_rgb16().unwrap();
            let actual = dyn_img.into_rgb16();
            for (&s, &a) in expected.pixels.iter().zip(actual.as_raw().iter()) {
                let scaled = ((u32::from(s) * 65_535 + half) / max_src) as u16;
                assert_eq!(a, scaled);
            }
        }
        tested += 1;
    }
    assert!(tested >= 1, "en az bir 10/12-bit AVIF test edilmeli");
}

#[derive(Debug)]
struct CountingReader<R> {
    inner: R,
    bytes_read: Arc<AtomicUsize>,
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.bytes_read.fetch_add(n, Ordering::Relaxed);
        Ok(n)
    }
}

#[test]
fn allocation_limit_enforced_before_full_read_and_across_header_and_decode() {
    let big_icc = vec![0x5Au8; 12 * 1024];
    let mut big_exif = b"MM\x00\x2a\x00\x00\x00\x08".to_vec();
    big_exif.resize(12 * 1024, 0xA5);
    let avif = build_synthetic_avif(AV1_1X1_8BIT, None, 1, 1, 8, Some(&big_icc), Some(&big_exif));
    assert!(avif.len() > 24 * 1024);

    // 1. allocation_limit = 1 bayt: akış tamamen okunmadan en fazla limit + 1 = 2 baytta reddedilmeli.
    let read_1 = Arc::new(AtomicUsize::new(0));
    let err_1 = AvifDecoder::with_config(
        CountingReader {
            inner: Cursor::new(&avif),
            bytes_read: Arc::clone(&read_1),
        },
        AvifDecodeConfig {
            allocation_limit: Some(1),
            ..AvifDecodeConfig::default()
        },
    )
    .expect_err("allocation_limit=1 tam okumadan önce reddedilmeli");
    assert!(matches!(
        err_1,
        ImageError::Limits(ref e) if e.kind() == LimitErrorKind::InsufficientMemory
    ));
    assert!(
        read_1.load(Ordering::Relaxed) <= 2,
        "en fazla limit+1 bayt okunmalıydı, okunan: {}",
        read_1.load(Ordering::Relaxed)
    );

    // 2. allocation_limit = avif.len() - 1: tam ham dosya sınırı 1 bayt aşıldığında reddedilmeli.
    let read_edge = Arc::new(AtomicUsize::new(0));
    let err_edge = AvifDecoder::with_config(
        CountingReader {
            inner: Cursor::new(&avif),
            bytes_read: Arc::clone(&read_edge),
        },
        AvifDecodeConfig {
            allocation_limit: Some(avif.len() - 1),
            ..AvifDecodeConfig::default()
        },
    )
    .expect_err("avif.len() - 1 sınırı reddedilmeli");
    assert!(matches!(
        err_edge,
        ImageError::Limits(ref e) if e.kind() == LimitErrorKind::InsufficientMemory
    ));
    assert_eq!(read_edge.load(Ordering::Relaxed), avif.len());

    // 3. Ham dosya sığar ancak raw_bytes + ICC (12 KiB) + Exif (12 KiB) üst bilgi kopyaları sığmaz:
    //    parse_info_with kalan bütçe (limit - raw_bytes.len()) üzerinden reddetmeli.
    let err_hdr = AvifDecoder::with_config(
        Cursor::new(&avif),
        AvifDecodeConfig {
            allocation_limit: Some(avif.len() + 1024),
            ..AvifDecodeConfig::default()
        },
    )
    .expect_err("raw_bytes + ICC/Exif üst bilgi tahsisleri bütçeyi aşmalı");
    assert!(matches!(
        err_hdr,
        ImageError::Limits(ref e) if e.kind() == LimitErrorKind::InsufficientMemory
    ));

    // 4. set_limits(max_alloc) halihazırda bellekte tutulan raw_bytes + ICC + Exif toplamından
    //    küçükse doğrudan InsufficientMemory döndürmeli.
    let mut dec_limits = AvifDecoder::new(Cursor::new(&avif)).expect("sınırsız kurulum");
    let mut lim = Limits::no_limits();
    lim.max_alloc = Some(avif.len() as u64);
    let err_lim = dec_limits
        .set_limits(lim)
        .expect_err("tutulan üst bilgi + ham bayt toplamı max_alloc'u aşmalı");
    assert!(matches!(
        err_lim,
        ImageError::Limits(ref e) if e.kind() == LimitErrorKind::InsufficientMemory
    ));

    // 5. Yeterli bütçede uçtan uca çözümleme başarılı ve `decode_avif_bytes` ile birebir aynı olmalı.
    let dec_ok = AvifDecoder::with_config(
        Cursor::new(&avif),
        AvifDecodeConfig {
            allocation_limit: Some(512 * 1024),
            ..AvifDecodeConfig::default()
        },
    )
    .expect("yeterli bütçe ile decoder kurulmalı");
    let img_ok = DynamicImage::from_decoder(dec_ok).expect("DynamicImage çözülmeli");
    let ref_img = decode_avif_bytes(&avif).expect("decode_avif_bytes");
    assert_eq!(img_ok.to_rgb8(), ref_img.to_rgb8());

    // 6. Gerçek lisans-onaylı korpus dosyası (>21 KiB) mevcutsa allocation_limit=1 anında reddetmeli.
    let fox_path = std::path::Path::new(
        "/home/hakan/github/av1oxide_planlar/avif_dogrulama/kaynaklar/av1-avif/testFiles/Link-U/fox.profile0.10bpc.yuv420.avif",
    );
    if fox_path.exists() {
        let fox_bytes = std::fs::read(fox_path).expect("fox avif okunmalı");
        assert!(fox_bytes.len() > 21 * 1024);
        let fox_read = Arc::new(AtomicUsize::new(0));
        let fox_err = AvifDecoder::with_config(
            CountingReader {
                inner: Cursor::new(&fox_bytes),
                bytes_read: Arc::clone(&fox_read),
            },
            AvifDecodeConfig {
                allocation_limit: Some(1),
                ..AvifDecodeConfig::default()
            },
        )
        .expect_err("gerçek >21 KiB AVIF allocation_limit=1 ile okunmadan reddedilmeli");
        assert!(matches!(
            fox_err,
            ImageError::Limits(ref e) if e.kind() == LimitErrorKind::InsufficientMemory
        ));
        assert!(fox_read.load(Ordering::Relaxed) <= 2);
    }
}
