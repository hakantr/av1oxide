//! Kaynak ağacındaki `unsafe` blok ve işlev sayısını denetleyen birim testleri.
//!
//! Anahtar sözcük kendi kaynak taramasını tetiklememek için `["uns", "afe"].concat()`
//! ile çalışma zamanında oluşturulur.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
enum Tur {
    Blok,
    Islev,
}

impl Tur {
    fn sonek(self) -> &'static str {
        match self {
            Tur::Blok => " {",
            Tur::Islev => " fn",
        }
    }
}

fn kelime() -> String {
    ["uns", "afe"].concat()
}

fn kaynak_koku() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn kaynak_dosyalari(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).expect("kaynak dizini okunmalı");
    for entry in entries {
        let path = entry.expect("dizin girdisi okunmalı").path();
        if path.is_dir() {
            kaynak_dosyalari(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

fn kaynak_kumesi(tur: Tur) -> BTreeSet<(String, usize)> {
    let src = kaynak_koku();
    let mut dosyalar = Vec::new();
    kaynak_dosyalari(&src, &mut dosyalar);
    assert!(!dosyalar.is_empty(), "kaynak ağacı boş olamaz");

    let aranan = format!("{}{}", kelime(), tur.sonek());
    let mut kayitlar = BTreeSet::new();
    for dosya in dosyalar {
        let rel = dosya
            .strip_prefix(&src)
            .expect("kaynak kökü altında olmalı")
            .to_string_lossy()
            .replace('\\', "/");
        let metin = fs::read_to_string(&dosya).expect("kaynak okunmalı");
        for (index, satir) in metin.lines().enumerate() {
            if satir.contains(&aranan) {
                kayitlar.insert((rel.clone(), index + 1));
            }
        }
    }
    kayitlar
}

#[test]
fn envanter_bloklari_kaynakla_uyumlu() {
    let bloklar = kaynak_kumesi(Tur::Blok);
    assert_eq!(
        bloklar.len(),
        399,
        "beklenen SIMD + C-ABI sınır bloğu sayısı 399 olmalı"
    );
}

#[test]
fn envanter_fnleri_kaynakla_uyumlu() {
    let islevler = kaynak_kumesi(Tur::Islev);
    assert!(
        islevler.is_empty(),
        "kaynak ağacında güvensiz işlev imzası bulunmamalı: {islevler:?}"
    );
}
