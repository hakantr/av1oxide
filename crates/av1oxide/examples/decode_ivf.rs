//! `av1oxide` çok-thread IVF akışı çözme ve YUV düzlem okuma örneği (`v0.1.0`).
//!
//! Çalıştırma:
//!   cargo run --release -p av1oxide --example decode_ivf
//!   cargo run --release -p av1oxide --example decode_ivf -- <girdi.ivf> [cikti.yuv]

#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use av1oxide::{Config, Decoder, Frame, Packet, PlaneType, Result};

/// Basit IVF kapsayıcı ayrıştırıcısı (32 bayt dosya başlığı + kare başına 12 bayt başlık).
fn parse_ivf_packets(data: &[u8]) -> Vec<(i64, &[u8])> {
    assert!(
        data.len() >= 32 && &data[0..4] == b"DKIF" && &data[8..12] == b"AV01",
        "geçerli bir AV1 IVF dosyası bekleniyor"
    );
    let mut offset = 32usize;
    let mut packets = Vec::new();
    while offset + 12 <= data.len() {
        let frame_size =
            u32::from_le_bytes(data[offset..offset + 4].try_into().expect("4 bayt")) as usize;
        let pts = i64::from_le_bytes(data[offset + 4..offset + 12].try_into().expect("8 bayt"));
        offset += 12;
        if offset + frame_size > data.len() {
            break;
        }
        packets.push((pts, &data[offset..offset + frame_size]));
        offset += frame_size;
    }
    packets
}

/// Çözülen bir karenin Y, U, V düzlemlerini satır satır (stride dolgusunu atlayarak) yazar.
///
/// `plane.row(y)` 8-bit için `width` bayt, 10/12-bit için `width * 2` (küçük-endian `u16`)
/// anlamlı satır baytlarını döndürür.
fn write_planar_yuv(writer: &mut impl Write, frame: &Frame) -> std::io::Result<()> {
    for plane_type in [PlaneType::Y, PlaneType::U, PlaneType::V] {
        let Some(plane) = frame.plane(plane_type) else {
            continue;
        };
        for y in 0..plane.height() {
            if let Some(row) = plane.row(y) {
                writer.write_all(row)?;
            }
        }
    }
    Ok(())
}

fn default_ivf_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "tests/fixtures/streams/Syntax_AV1_mainb8ss420_432x240_002_vq_aom_ctest_4.2.stripped.ivf",
    )
}

/// `.crate` arşivinden bağımsız çalıştırıldığında (`tests/fixtures/` paket dışında olduğunda)
/// kullanılan gömülü 2 karelik (1×1, 8-bit) geçerli AV1 IVF akışı.
fn embedded_fallback_ivf() -> Vec<u8> {
    const FRAME_0: &[u8] = &[
        0x12, 0x00, 0x0A, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7D, 0xFC, 0xC0, 0x20, 0x32, 0x17,
        0x10, 0x01, 0x91, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0xD2, 0x16,
        0x8C, 0x64, 0xBE, 0x2E, 0xC7, 0xDA, 0xEE, 0x96,
    ];
    const FRAME_1: &[u8] = &[
        0x12, 0x00, 0x32, 0x1B, 0x30, 0x02, 0x01, 0x00, 0xA7, 0x2E, 0x47, 0xB0, 0x80, 0x00, 0x02,
        0x40, 0x00, 0x00, 0x80, 0x00, 0x00, 0x03, 0x5A, 0xF0, 0x1B, 0x0B, 0xA1, 0xEF, 0xD4, 0x62,
        0xC0,
    ];
    let mut ivf = Vec::with_capacity(32 + (12 + FRAME_0.len()) + (12 + FRAME_1.len()));
    ivf.extend_from_slice(b"DKIF");
    ivf.extend_from_slice(&0u16.to_le_bytes()); // version
    ivf.extend_from_slice(&32u16.to_le_bytes()); // header length
    ivf.extend_from_slice(b"AV01");
    ivf.extend_from_slice(&1u16.to_le_bytes()); // width
    ivf.extend_from_slice(&1u16.to_le_bytes()); // height
    ivf.extend_from_slice(&30u32.to_le_bytes()); // timebase den
    ivf.extend_from_slice(&1u32.to_le_bytes()); // timebase num
    ivf.extend_from_slice(&2u32.to_le_bytes()); // num frames
    ivf.extend_from_slice(&0u32.to_le_bytes()); // unused
    for (pts, frame) in [(0u64, FRAME_0), (1u64, FRAME_1)] {
        ivf.extend_from_slice(&(frame.len() as u32).to_le_bytes());
        ivf.extend_from_slice(&pts.to_le_bytes());
        ivf.extend_from_slice(frame);
    }
    ivf
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let cli_ivf_path = args.next().map(PathBuf::from);
    let yuv_out_path = args.next().map(PathBuf::from);

    let (source_label, ivf_bytes) = if let Some(path) = cli_ivf_path {
        let label = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("akis.ivf")
            .to_string();
        let bytes = std::fs::read(&path).expect("IVF dosyası okunamadı");
        (label, bytes)
    } else {
        let default_path = default_ivf_path();
        if let Ok(bytes) = std::fs::read(&default_path) {
            let label = default_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("akis.ivf")
                .to_string();
            (label, bytes)
        } else {
            ("embedded_1x1.ivf".to_string(), embedded_fallback_ivf())
        }
    };
    let packets = parse_ivf_packets(&ivf_bytes);

    // Çok-thread çözücü yapılandırması:
    // - n_threads: 4 işçi thread'i
    // - max_frame_delay: 2 eşzamanlı kare yuvası (kareler arası + kare içi paralellik)
    let config = Config {
        n_threads: 4,
        max_frame_delay: 2,
        ..Config::default()
    };
    let mut decoder = Decoder::new(config)?;
    let mut yuv_writer = yuv_out_path
        .as_ref()
        .map(|p| BufWriter::new(File::create(p).expect("YUV çıktı dosyası oluşturulamadı")));

    let mut frame_count = 0usize;
    let mut last_dims = (0usize, 0usize);

    for (pts, payload) in packets {
        decoder.send_packet(&Packet::with_timestamp(payload, pts))?;
        while let Some(frame) = decoder.receive_frame()? {
            last_dims = (frame.width(), frame.height());
            if let Some(w) = yuv_writer.as_mut() {
                write_planar_yuv(w, &frame).expect("YUV yazma hatası");
            }
            frame_count += 1;
        }
    }

    // Boru hattında bekleyen son kareleri boşalt.
    decoder.flush();
    while let Some(frame) = decoder.receive_frame()? {
        last_dims = (frame.width(), frame.height());
        if let Some(w) = yuv_writer.as_mut() {
            write_planar_yuv(w, &frame).expect("YUV yazma hatası");
        }
        frame_count += 1;
    }

    if let Some(mut w) = yuv_writer {
        w.flush().expect("YUV tampon boşaltma hatası");
    }

    println!(
        "av1oxide v{}: '{}' içinden {} kare çözüldü ({}x{}, tepe tahsis: {} bayt).",
        av1oxide::VERSION,
        source_label,
        frame_count,
        last_dims.0,
        last_dims.1,
        decoder.allocation_peak()
    );
    assert!(frame_count > 0, "en az bir kare çözülmeli");
    Ok(())
}
