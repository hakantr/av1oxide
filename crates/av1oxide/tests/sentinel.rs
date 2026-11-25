//! S15.3 — Sentinel ve padding over-read altyapısının **kendi testi** (SMD-07).
//!
//! Bu dosya yalnız "mutlu yol"u değil, **tespitin gerçekten çalıştığını** da
//! kanıtlar: her pozitif kontrolün yanında, kasıtlı bir taşma/over-read üreten
//! bir **negatif kontrol** vardır ve ihlalin *tarafı + indeksi* tam olarak
//! doğrulanır. Negatif kontrolü olmayan bir sentinel testi, hiçbir şeyi
//! kanıtlamaz.

#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
mod common;

use common::sentinel::{
    GuardSide, OverReadProbe, SentinelBuffer, VARSAYILAN_KORUMA, VARSAYILAN_SENTINEL,
    regresyon_kaydet, varsayilan_regresyon_dizini,
};
use std::fs;

// ---------------------------------------------------------------- pozitif ---

#[test]
fn koruyucular_dogru_yerlesimle_ayrilir() {
    let buffer = SentinelBuffer::with_defaults(16);
    assert_eq!(buffer.data_len(), 16);
    assert_eq!(buffer.guard_len(), VARSAYILAN_KORUMA);
    assert_eq!(buffer.data_offset(), VARSAYILAN_KORUMA);
    assert_eq!(
        buffer.raw().len(),
        16 + 2 * VARSAYILAN_KORUMA,
        "ham tampon = ön koruyucu + veri + arka koruyucu"
    );
    assert_eq!(buffer.sentinel(), VARSAYILAN_SENTINEL);
    assert_eq!(buffer.verify(), Ok(()), "başlangıçta koruyucular temiz");
}

#[test]
fn temiz_yazma_koruyuculari_ihlal_etmez() {
    let mut buffer = SentinelBuffer::with_defaults(16);
    assert!(buffer.write_data(&[7u8; 16]));
    buffer.fill_data(3);
    assert_eq!(buffer.data(), &[3u8; 16]);
    assert_eq!(
        buffer.verify(),
        Ok(()),
        "yalnız veriye yazmak ihlal değildir"
    );
}

#[test]
fn yanlis_uzunlukta_yazma_reddedilir_ve_veri_degismez() {
    let mut buffer = SentinelBuffer::with_defaults(4);
    let before = buffer.data().to_vec();
    assert!(
        !buffer.write_data(&[1, 2, 3]),
        "eksik veri kabul edilmemeli"
    );
    assert!(
        !buffer.write_data(&[1, 2, 3, 4, 5]),
        "fazla veri kabul edilmemeli"
    );
    assert_eq!(buffer.data(), before.as_slice());
    assert_eq!(buffer.verify(), Ok(()));
}

#[test]
fn sifir_uzunluklu_veri_de_gecerlidir() {
    let buffer = SentinelBuffer::with_defaults(0);
    assert_eq!(buffer.data_len(), 0);
    assert!(buffer.data().is_empty());
    assert_eq!(buffer.raw().len(), 2 * VARSAYILAN_KORUMA);
    assert_eq!(buffer.verify(), Ok(()));
}

#[test]
fn veri_araligini_okuyan_cekirdek_over_read_sayilmaz() {
    let probe = OverReadProbe::new(16, 8, 2);
    assert_eq!(probe.input_len(), 32);
    assert_eq!(probe.data_offset(), 8);
    assert_eq!(probe.payload()[0], 7, "payload(i) = i×31 + 7 (mod 256)");

    let output = probe
        .run(|input, out| {
            let start = probe.data_offset();
            let data = &input[start..start + probe.data_len()];
            out[0] = data.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
            out[1] = u8::try_from(data.len()).expect("16 ≤ 255");
        })
        .expect("veri aralığını okumak over-read değildir");

    assert_eq!(output[1], 16);
}

// ------------------------------------------------- negatif kontroller ------

#[test]
fn arka_koruyucuya_bir_bayt_tasma_yakalanir() {
    let mut buffer = SentinelBuffer::with_defaults(16);
    let tasan_indeks = buffer.data_offset() + buffer.data_len();
    buffer.raw_mut()[tasan_indeks] = 0x5A; // kasıtlı taşma: verinin bir bayt ötesi

    let violation = buffer
        .verify()
        .expect_err("kasıtlı taşma yakalanmalıydı (negatif kontrol)");
    assert_eq!(violation.side, GuardSide::Back);
    assert_eq!(violation.index, tasan_indeks);
    assert_eq!(violation.expected, VARSAYILAN_SENTINEL);
    assert_eq!(violation.found, 0x5A);
}

#[test]
fn on_koruyucuya_bir_bayt_tasma_yakalanir() {
    let mut buffer = SentinelBuffer::with_defaults(16);
    let tasan_indeks = buffer.data_offset() - 1;
    buffer.raw_mut()[tasan_indeks] = 0x00; // kasıtlı negatif yön taşması

    let violation = buffer
        .verify()
        .expect_err("kasıtlı taşma yakalanmalıydı (negatif kontrol)");
    assert_eq!(violation.side, GuardSide::Front);
    assert_eq!(violation.index, tasan_indeks);
    assert_eq!(violation.expected, VARSAYILAN_SENTINEL);
    assert_eq!(violation.found, 0x00);
}

#[test]
fn tum_girdiyi_okuyan_cekirdek_over_read_olarak_yakalanir() {
    let probe = OverReadProbe::new(16, 8, 1);

    // Kasıtlı hata: veri aralığı yerine **tüm** tamponu (padding dâhil) toplar.
    let over = probe
        .run(|input, out| {
            out[0] = input.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        })
        .expect_err("padding okuması yakalanmalıydı (negatif kontrol)");

    assert_eq!(over.first_diff, 0, "ilk bayttan itibaren fark oluşur");
    assert_eq!(over.padding_a, 0x00);
    assert_eq!(over.padding_b, 0xFF);
}

#[test]
fn veri_baslangicindan_onceki_padding_okumasi_yakalanir() {
    let probe = OverReadProbe::new(24, 8, 1);

    let over = probe
        .run(|input, out| {
            // Kasıtlı hata: verinin bir bayt öncesinden (ön padding) okur.
            out[0] = input[probe.data_offset() - 1];
        })
        .expect_err("ön padding okuması yakalanmalıydı (negatif kontrol)");

    assert_eq!(over.first_diff, 0);
}

#[test]
fn padding_deseni_degismeden_sonuc_degismez() {
    let probe = OverReadProbe::new(32, 4, 4);

    // İki koşuyu ayrı ayrı çalıştırıp çıktıların gerçekten bit-aynı olduğunu
    // gösterir: tespit mekanizmasının "yanlış pozitif" üretmediğinin kanıtı.
    let mut out_a = [0u8; 4];
    let mut out_b = [0u8; 4];
    probe
        .run_pair(&mut out_a, &mut out_b, |input, out| {
            let start = probe.data_offset();
            let data = &input[start..start + probe.data_len()];
            out.copy_from_slice(&data[..4]);
        })
        .expect("temiz çekirdek iki koşuda da aynı sonucu vermeli");
    assert_eq!(out_a, out_b);
}

// --------------------------------------------- birlikte kullanım + dump -----

#[test]
fn sentinel_ve_probe_birlikte_kullanilir() {
    let probe = OverReadProbe::new(16, 8, 8);
    let mut out_a = SentinelBuffer::with_defaults(probe.out_len());
    let mut out_b = SentinelBuffer::with_defaults(probe.out_len());

    let mut sentinel_ihlali = None;
    let sonuc = probe.run_pair(out_a.data_mut(), out_b.data_mut(), |input, out| {
        let start = probe.data_offset();
        out.copy_from_slice(&input[start..start + out.len()]);
    });
    if let Err(violation) = out_a.verify() {
        sentinel_ihlali = Some(violation);
    }
    if let Err(violation) = out_b.verify() {
        sentinel_ihlali = Some(violation);
    }

    assert!(sonuc.is_ok(), "girdi over-read edilmedi: {sonuc:?}");
    assert!(sentinel_ihlali.is_none(), "çıktı taşmadı");
    assert_eq!(out_a.data(), out_b.data());
}

#[test]
fn hata_veren_girdi_diske_kaydedilir() {
    let mut buffer = SentinelBuffer::new(8, 4, 0x3C);
    buffer.fill_data(0x11);
    let tasan_indeks = buffer.data_offset() + buffer.data_len();
    buffer.raw_mut()[tasan_indeks] = 0x99; // hata üreten girdi

    let dir = varsayilan_regresyon_dizini();
    let path = buffer
        .dump(&dir, "sentinel-negatif-kontrol.bin")
        .expect("regresyon kaydı yazılmalı");

    assert!(path.exists(), "kayıt diskte olmalı: {}", path.display());
    assert_eq!(
        fs::read(&path).expect("kayıt okunmalı"),
        buffer.raw(),
        "diske yazılan içerik ham tamponun aynısı olmalı"
    );
    assert!(
        path.to_string_lossy()
            .contains("target/sentinel-regressions"),
        "regresyon dizini target/ altındadır (üretim ağacına girmez): {}",
        path.display()
    );

    // Yardımcı doğrudan da kullanılabilir.
    let ikinci =
        regresyon_kaydet(&dir, "dogrudan-kayit.bin", &[1, 2, 3]).expect("doğrudan kayıt yazılmalı");
    assert_eq!(fs::read(&ikinci).expect("okunmalı"), vec![1, 2, 3]);

    // Test kendi çöpünü temizler.
    fs::remove_file(&path).expect("kayıt silinmeli");
    fs::remove_file(&ikinci).expect("kayıt silinmeli");
    assert!(!path.exists());
}
