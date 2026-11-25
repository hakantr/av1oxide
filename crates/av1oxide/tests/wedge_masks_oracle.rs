//! Wedge / interintra maske oracle'ı — Rust ↔ dav1d C **bayt-eşit**.
//!
//! dav1d `dav1d_init_ii_wedge_masks()` (`src/wedge.c:207-299`) bütün wedge ve
//! interintra maske tablolarını üretir. `scripts/oracle/wedge_masks_oracle.c`
//! bunları **kanonik sırayla** döker; bu test üretim crate'inin
//! [`av1oxide::decode::wedge::canonical_mask_bytes`] çıktısını aynı sırayla
//! üretip bayt bayt karşılaştırır ve Rust dökümünü yanına yazar (kabuk testi
//! `cmp` + `sha256sum` ile bağımsız olarak da doğrular).
//!
//! Kanonik sıra: layout (444, 422, 420) → blok boyutu (BS_32x32..BS_8x8) →
//! işaret → wedge indeksi; ardından aynı sırayla interintra modları.
//!
//! Yeniden üretim (tek komut): `bash scripts/oracle/run_wedge_oracle.sh`

#![forbid(unsafe_code)]

mod common;

use av1oxide::decode::wedge::canonical_mask_bytes;
use common::on_kosul::{self, Neden};
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn find(name: &str) -> Option<PathBuf> {
    let cache = root().join(".cache/wedge").join(name);
    if cache.is_file() {
        return Some(cache);
    }
    let fixture = root()
        .join("crates/av1oxide/tests/fixtures/wedge")
        .join(name);
    fixture.is_file().then_some(fixture)
}

#[test]
fn wedge_ve_interintra_maskeleri_dav1d_ile_bayt_esit() {
    let Some(c_path) = find("wedge-masks-c.bin") else {
        on_kosul::eksik(
            "wedge_masks_oracle::wedge_ve_interintra_maskeleri_dav1d_ile_bayt_esit",
            Neden::OracleVerisiYok,
            "wedge oracle çıktısı yok (bash scripts/oracle/run_wedge_oracle.sh)",
        );
        return;
    };
    let c = std::fs::read(&c_path).unwrap();
    let rust = canonical_mask_bytes();
    assert_eq!(
        rust.len(),
        c.len(),
        "maske bayt sayısı farklı: Rust {} != C {}",
        rust.len(),
        c.len()
    );
    if let Some(pos) = rust.iter().zip(c.iter()).position(|(a, b)| a != b) {
        panic!(
            "maske baytları farklı: ilk fark {pos} (Rust {} != C {})",
            rust[pos], c[pos]
        );
    }
    // Rust dökümünü kabuk testinin `cmp`/`sha256sum` doğrulaması için yaz.
    let out_dir = root().join(".cache/wedge");
    if out_dir.is_dir() {
        let _ = std::fs::write(out_dir.join("wedge-masks-rust.bin"), &rust);
    }
    eprintln!(
        "wedge oracle: {} bayt bayt-eşit (kanonik sıra: 3 layout × 9 blok × 2 işaret × 16 idx \
         + 3 layout × 7 blok × 4 interintra modu)",
        rust.len()
    );
}
