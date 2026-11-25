//! `av1oxide` güvenilmeyen girdi için kaynak bütçesi, hata yakalama ve `reset()` örneği (`v0.1.0`).
//!
//! Çalıştırma:
//!   cargo run --release -p av1oxide --example bounded_decoder

#![forbid(unsafe_code)]

use av1oxide::{Config, Decoder, DecoderState, Error, Packet, Result};

/// Geçerli 1×1 8-bit AV1 OBU paketi (Temporal Delimiter + Sequence Header + Keyframe).
const VALID_OBU: &[u8] = &[
    0x12, 0x00, 0x0A, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7D, 0xFC, 0xC0, 0x20, 0x32, 0x17, 0x10,
    0x01, 0x91, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x68, 0xD2, 0x16, 0x8C, 0x64,
    0xBE, 0x2E, 0xC7, 0xDA, 0xEE, 0x96,
];

/// Kesilmiş / bozuk OBU paketi (boyut alanı kalan bayttan büyük).
const CORRUPTED_OBU: &[u8] = &[0x0A, 0x7F, 0x00, 0x01];

fn main() -> Result<()> {
    // 1. Aşırı düşük bellek bütçesi (64 bayt) ile tahsis öncesi bütçe korumasını göster.
    let tiny_budget_config = Config {
        n_threads: 1,
        max_frame_delay: 1,
        frame_size_limit: 1920 * 1080,
        allocation_limit: Some(64),
        ..Config::default()
    };
    let mut tiny_decoder = Decoder::new(tiny_budget_config)?;
    match tiny_decoder.send_packet(&Packet::new(VALID_OBU)) {
        Err(Error::BudgetExceeded { requested, limit }) => {
            println!(
                "[bütçe koruması] Tahsis öncesi engellendi: istenen={requested} B, sınır={limit} B"
            );
        }
        other => panic!("BudgetExceeded bekleniyordu, alınan: {other:?}"),
    }

    // 2. Üretim için sınırlandırılmış çözücü (Full HD piksel sınırı + 32 MiB bellek tavanı).
    let prod_config = Config {
        n_threads: 2,
        max_frame_delay: 1,
        frame_size_limit: 1920 * 1080,
        allocation_limit: Some(32 * 1024 * 1024),
        ..Config::default()
    };
    let mut decoder = Decoder::new(prod_config)?;

    // Bozuk paket gönderildiğinde panik oluşmaz; yapılandırılmış `Error::Bitstream` döner.
    match decoder.send_packet(&Packet::new(CORRUPTED_OBU)) {
        Err(Error::Bitstream(msg)) => {
            println!("[bozuk paket yakalandı] Error::Bitstream({msg:?})");
        }
        other => panic!("Error::Bitstream bekleniyordu, alınan: {other:?}"),
    }

    // `reset()` ile aynı çözücü örneğini yeniden tahsis etmeden temiz duruma döndür.
    decoder.reset();
    assert_eq!(decoder.state(), DecoderState::Ready);

    // Sıfırlama sonrası geçerli akışı çözmeye devam et.
    decoder.send_packet(&Packet::with_timestamp(VALID_OBU, 42))?;
    let frame = decoder
        .receive_frame()?
        .expect("geçerli paketten kare üretilmeli");
    println!(
        "[kurtarma başarılı] Çözülen kare: {}x{}, pts={:?}, tepe tahsis={} B",
        frame.width(),
        frame.height(),
        frame.metadata().timestamp,
        decoder.allocation_peak()
    );

    Ok(())
}
