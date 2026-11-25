//! `av1oxide` uçtan uca OBU paket çözme örneği (`v0.4.0`).
//!
//! Çalıştırma:
//!   cargo run --release -p av1oxide --example decode_obu
//!   cargo run --release -p av1oxide --example decode_obu -- <dosya.obu>

#![forbid(unsafe_code)]

use av1oxide::{Config, Decoder, Frame, Packet, PlaneType, Result};

/// Gömülü 2 paketlik (1×1, 8-bit, 4:2:0) örnek AV1 OBU akışı (`00000000.ivf` paketleri).
const EMBEDDED_OBU_PACKETS: &[&[u8]] = &[
    // Paket 0: Temporal Delimiter + Sequence Header + Keyframe (38 bayt)
    &[
        0x12, 0x00, 0x0A, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7D, 0xFC, 0xC0, 0x20, 0x32, 0x17,
        0x10, 0x01, 0x91, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0xD2, 0x16,
        0x8C, 0x64, 0xBE, 0x2E, 0xC7, 0xDA, 0xEE, 0x96,
    ],
    // Paket 1: Temporal Delimiter + Inter Frame (31 bayt)
    &[
        0x12, 0x00, 0x32, 0x1B, 0x30, 0x02, 0x01, 0x00, 0xA7, 0x2E, 0x47, 0xB0, 0x80, 0x00, 0x02,
        0x40, 0x00, 0x00, 0x80, 0x00, 0x00, 0x03, 0x5A, 0xF0, 0x1B, 0x0B, 0xA1, 0xEF, 0xD4, 0x62,
        0xC0,
    ],
];

fn inspect_frame(index: usize, frame: &Frame) {
    let y = frame.plane(PlaneType::Y).expect("Y düzlemi");
    let first_sample = y.row(0).and_then(|r| r.first()).copied().unwrap_or(0);
    println!(
        "kare #{index}: {}x{}, {:?}, {:?}, pts={:?}, düzlem={}, Y_stride={}, Y[0,0]={}",
        frame.width(),
        frame.height(),
        frame.bit_depth(),
        frame.chroma_format(),
        frame.metadata().timestamp,
        frame.plane_count(),
        y.stride(),
        first_sample
    );
}

fn main() -> Result<()> {
    let config = Config {
        n_threads: 2,
        max_frame_delay: 2,
        ..Config::default()
    };
    let mut decoder = Decoder::new(config)?;
    let mut decoded_count = 0usize;

    if let Some(path) = std::env::args().nth(1) {
        let bytes = std::fs::read(&path).expect("OBU dosyası okunamadı");
        decoder.send_packet(&Packet::new(&bytes))?;
        while let Some(frame) = decoder.receive_frame()? {
            inspect_frame(decoded_count, &frame);
            decoded_count += 1;
        }
    } else {
        for (pts, payload) in EMBEDDED_OBU_PACKETS.iter().enumerate() {
            let packet = Packet::with_timestamp(payload, pts as i64);
            decoder.send_packet(&packet)?;
            while let Some(frame) = decoder.receive_frame()? {
                inspect_frame(decoded_count, &frame);
                decoded_count += 1;
            }
        }
    }

    decoder.flush();
    while let Some(frame) = decoder.receive_frame()? {
        inspect_frame(decoded_count, &frame);
        decoded_count += 1;
    }

    println!(
        "av1oxide v{}: toplam {decoded_count} kare çözüldü.",
        av1oxide::VERSION
    );
    assert!(decoded_count > 0, "en az bir kare çözülmeli");
    Ok(())
}
