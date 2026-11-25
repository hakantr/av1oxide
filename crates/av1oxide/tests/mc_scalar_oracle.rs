//! S15.5 — 8(a): skaler Rust ↔ dav1d C **bit-exact** (oracle karşılaştırması).
//!
//! Oracle, dav1d'nin C yolunu (`mc_tmpl.c`, asm = OFF derlemesi) **ayrı bir
//! süreçte** çalıştırır ve beklenen çıktıyı fixture dosyasına yazar; bu test
//! üretim crate'inin skaler çekirdeğini aynı girdiyle koşturup bayt bayt
//! karşılaştırır. Üretim crate'i oracle'a **bağlanmaz** (SRC-11).
//!
//! Fixture'lar:
//!
//! | Küme | Konum | Git |
//! |---|---|---|
//! | Küçük sınır kümesi (1512 kayıt) | `tests/fixtures/mc/` | ✅ girer |
//! | Büyük rastgele küme (42 000 kayıt, hücre başına ≥ 2 000) | `.cache/mc-oracle/` | ⬜ girmez |
//!
//! Büyük küme yoksa (taze klon) yalnız küçük küme koşar ve bu **açıkça**
//! raporlanır; küme varsa hücre başına ≥ 2 000 şartı **test edilir**.
//!
//! Yeniden üretim (tek komut):
//! `bash scripts/oracle/run_mc_oracle.sh`

#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::dsp::scalar::mc;
use common::mc_fixture::{Case, Coverage, cell_counts, read_cases, run_case};
use common::on_kosul::{self, Neden};
use std::fs;
use std::path::{Path, PathBuf};

/// Çalışma alanı kökü (`crates/av1oxide/../..`).
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .to_path_buf()
}

/// Git'e giren küçük sınır kümesinin dizini.
fn small_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("mc")
}

/// `.cache/` altındaki büyük küme dizini.
fn large_dir() -> PathBuf {
    workspace_root().join(".cache").join("mc-oracle")
}

fn load(dir: &Path, name: &str) -> Option<Vec<Case>> {
    let input = fs::read(dir.join(format!("mc-{name}-input.bin"))).ok()?;
    let expect = fs::read(dir.join(format!("mc-{name}-expect.bin"))).ok()?;
    Some(read_cases(&input, &expect))
}

/// Küçük sınır kümesini yükler ve **kapsam** şartlarını doğrular.
fn small_cases() -> Vec<Case> {
    let cases = load(&small_dir(), "small").expect("küçük fixture git'te olmalı");
    assert!(
        cases.len() >= 1400,
        "küçük sınır kümesi beklenenden az: {}",
        cases.len()
    );
    let coverage = Coverage::of(&cases);
    let missing = coverage.missing();
    assert!(
        missing.is_empty(),
        "küçük fixture kapsamı eksik: {missing:?}"
    );
    cases
}

/// Fixture'ın kapsam özetini ekrana basar (kanıt izi).
fn report(label: &str, cases: &[Case]) {
    let cov = Coverage::of(cases);
    let cells = cell_counts(cases);
    println!("[{label}] {} kayıt", cases.len());
    for ((bd, w), n) in &cells {
        println!("  {bd:2}-bit w={w:3} : {n}");
    }
    println!(
        "  kapsam: derinlik={}/3 genişlik={}/7 mx={}/16 my={}/16 yol={}/4 maxpiksel={}/3 stride-farkı={} eksik={:?}",
        cov.depths.iter().filter(|x| **x).count(),
        cov.widths.iter().filter(|x| **x).count(),
        cov.mx.iter().filter(|x| **x).count(),
        cov.my.iter().filter(|x| **x).count(),
        cov.paths.iter().filter(|x| **x).count(),
        cov.max_pixel.iter().filter(|x| **x).count(),
        cov.strides_differ,
        cov.missing(),
    );
}

/// Tüm kayıtları skaler çekirdekle koşar; ilk uyuşmazlıkta panikler.
fn compare_all(label: &str, cases: &[Case]) {
    let mut bad = 0usize;
    let mut first: Option<(usize, Vec<u8>, Vec<u8>)> = None;
    for (index, case) in cases.iter().enumerate() {
        let got = run_case(mc::put_8tap_regular, case);
        if got != case.expect {
            bad += 1;
            if first.is_none() {
                first = Some((index, got, case.expect.clone()));
            }
        }
    }
    if let Some((index, got, want)) = first {
        let case = &cases[index];
        let first_diff = got
            .iter()
            .zip(want.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "{label}: {bad}/{} kayıt uyuşmadı; ilk fark kayıt {index} \
             (bd={}, w={}, h={}, mx={}, my={}, src_stride={}, dst_stride={})\n  \
             ilk fark baytı {first_diff}: skaler={:02x?} dav1d={:02x?}",
            cases.len(),
            case.bit_depth,
            case.w,
            case.h,
            case.mx,
            case.my,
            case.src_stride,
            case.dst_stride,
            &got[first_diff.saturating_sub(4)..(first_diff + 8).min(got.len())],
            &want[first_diff.saturating_sub(4)..(first_diff + 8).min(want.len())],
        );
    }
}

/// 8(a) — küçük, git'e giren sınır kümesi (her koşuda).
#[test]
fn sekiz_a_skaler_kucuk_sinir_kumesinde_dav1d_c_ile_bit_ayni() {
    let cases = small_cases();
    report("küçük sınır kümesi", &cases);
    compare_all("küçük sınır kümesi", &cases);
}

/// 8(a) — `.cache/` altındaki büyük rastgele küme (varsa).
#[test]
fn sekiz_a_skaler_buyuk_rastgele_kumesinde_dav1d_c_ile_bit_ayni() {
    let Some(cases) = load(&large_dir(), "large") else {
        on_kosul::eksik(
            "mc_scalar_oracle::sekiz_a_skaler_buyuk_rastgele_kumesinde_dav1d_c_ile_bit_ayni",
            Neden::OracleVerisiYok,
            ".cache/mc-oracle büyük küme; üretmek için: bash scripts/oracle/run_mc_oracle.sh",
        );
        return;
    };
    let cells = cell_counts(&cases);
    assert!(!cells.is_empty(), "büyük küme boş");
    let mut eksik = Vec::new();
    for bd in [8u8, 10, 12] {
        for w in [2usize, 4, 8, 16, 32, 64, 128] {
            let n = cells.get(&(bd, w)).copied().unwrap_or(0);
            if n < 2000 {
                eksik.push(format!("{bd}-bit w={w}: {n} < 2000"));
            }
        }
    }
    assert!(
        eksik.is_empty(),
        "büyük küme hücre başına ≥ 2000 şartını sağlamıyor: {eksik:?}"
    );
    report("büyük rastgele küme", &cases);
    compare_all("büyük rastgele küme", &cases);
}
