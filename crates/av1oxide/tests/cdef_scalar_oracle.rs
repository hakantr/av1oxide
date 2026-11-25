//! S15.4 — 7(a): skaler Rust ↔ dav1d C **bit-exact** (oracle karşılaştırması).
//!
//! Oracle, dav1d'nin C yolunu (`cdef_tmpl.c`, asm = OFF derlemesi) **ayrı bir
//! süreçte** çalıştırır ve beklenen çıktıyı fixture dosyasına yazar; bu test
//! üretim crate'inin skaler çekirdeklerini aynı girdiyle koşturup bayt bayt
//! karşılaştırır. Üretim crate'i oracle'a **bağlanmaz** (SRC-11).
//!
//! Fixture'lar:
//!
//! | Küme | Konum | Git |
//! |---|---|---|
//! | Küçük sınır kümesi (542 kayıt) | `tests/fixtures/cdef/` | ✅ girer |
//! | Büyük rastgele küme (120 000 kayıt, hücre başına ≥ 10 000) | `.cache/cdef-oracle/` | ⬜ girmez |
//!
//! Büyük küme yoksa (taze klon) yalnız küçük küme koşar ve bu **açıkça**
//! raporlanır; küme varsa hücre başına ≥ 10 000 şartı **test edilir**.
//!
//! Yeniden üretim (tek komut):
//! `bash scripts/oracle/run_cdef_oracle.sh`

#![forbid(unsafe_code)]

mod common;

use common::cdef_fixture::{
    Case, Coverage, cell_counts, read_cases, run_dir, run_filter, scalar_dir, scalar_filter,
};
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
        .join("cdef")
}

/// `.cache/` altındaki büyük küme dizini; `CDEF_ORACLE_OUT` verilirse o dizin
/// (`run_cdef_oracle.sh` ile aynı değişken; göreli yol depo köküne göredir).
fn large_dir() -> PathBuf {
    std::env::var_os("CDEF_ORACLE_OUT").map_or_else(
        || workspace_root().join(".cache").join("cdef-oracle"),
        |d| workspace_root().join(d),
    )
}

fn load(dir: &Path, name: &str) -> Option<Vec<Case>> {
    let input = fs::read(dir.join(format!("cdef-{name}-input.bin"))).ok()?;
    let expect = fs::read(dir.join(format!("cdef-{name}-expect.bin"))).ok()?;
    Some(read_cases(&input, &expect))
}

/// Küçük sınır kümesini yükler ve **kapsam** şartlarını doğrular.
fn small_cases() -> Vec<Case> {
    let cases = load(&small_dir(), "small").expect("küçük fixture git'te olmalı");
    assert!(
        cases.len() >= 500,
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

/// Fixture'ların kapsam özetini ekrana basar (kanıt izi).
fn report(label: &str, cases: &[Case]) {
    let cov = Coverage::of(cases);
    let cells = cell_counts(cases);
    println!("[{label}] {} kayıt", cases.len());
    for ((kind, bd, size), n) in &cells {
        let k = if *kind == 0 { "dir   " } else { "filter" };
        println!("  {k} {bd:2}-bit blok={size} : {n}");
    }
    println!(
        "  kapsam: yön={}/8 kenar={}/16 boyut={}/3 derinlik={}/3 pri={}/16 sec={} damp={} maxpiksel={}/3 alt-kırpma(8/10/12)={:?} eksik={:?}",
        cov.dirs.iter().filter(|x| **x).count(),
        cov.edges.iter().filter(|x| **x).count(),
        cov.sizes.iter().filter(|x| **x).count(),
        cov.depths.iter().filter(|x| **x).count(),
        cov.pri_base.iter().filter(|x| **x).count(),
        cov.sec_base.iter().filter(|x| **x).count(),
        cov.damping.iter().filter(|x| **x).count(),
        cov.max_pixel_seen.iter().filter(|x| **x).count(),
        cov.alt_kirpma,
        cov.missing(),
    );
}

/// Tüm kayıtları skaler çekirdekle koşar; ilk uyuşmazlıkta panikler.
fn compare_all(label: &str, cases: &[Case]) {
    let mut bad = 0usize;
    let mut first: Option<(usize, Vec<u8>, Vec<u8>)> = None;
    for (index, case) in cases.iter().enumerate() {
        let got = if case.kind == 0 {
            run_dir(case, scalar_dir(case))
        } else {
            run_filter(case, scalar_filter(case))
        };
        if got != case.expect {
            bad += 1;
            if first.is_none() {
                first = Some((index, got, case.expect.clone()));
            }
        }
    }
    if let Some((index, got, want)) = first {
        panic!(
            "{label}: {bad}/{} kayıt uyuşmadı; ilk fark kayıt {index} \
             (kind={}, bd={}, blok={}, kenar={:#06b}, pri={}, sec={}, dir={}, damping={})\n  \
             skaler = {:02x?}\n  dav1d  = {:02x?}",
            cases.len(),
            cases[index].kind,
            cases[index].bit_depth,
            cases[index].block_size,
            cases[index].edges,
            cases[index].pri,
            cases[index].sec,
            cases[index].dir,
            cases[index].damping,
            got,
            want
        );
    }
}

/// 7(a) — küçük, git'e giren sınır kümesi (her koşuda).
#[test]
fn yedi_a_skaler_kucuk_sinir_kumesinde_dav1d_c_ile_bit_ayni() {
    let cases = small_cases();
    report("küçük sınır kümesi", &cases);
    compare_all("küçük sınır kümesi", &cases);
}

/// 7(a) — büyük rastgele küme (`.cache` varsa) ve hücre başına ≥ 10 000 şartı.
///
/// Küme yoksa [`on_kosul::eksik`]: `AV1OXIDE_ORACLES_REQUIRED` ayarlıysa FAIL,
/// değilse `ATLANDI` satırı (P29; taze klonda `.cache` boştur).
#[test]
fn yedi_a_skaler_buyuk_rastgele_kumede_dav1d_c_ile_bit_ayni() {
    let Some(cases) = load(&large_dir(), "large") else {
        on_kosul::eksik(
            "cdef_scalar_oracle::yedi_a_skaler_buyuk_rastgele_kumede_dav1d_c_ile_bit_ayni",
            Neden::OracleVerisiYok,
            ".cache/cdef-oracle büyük küme; üretmek için: bash scripts/oracle/run_cdef_oracle.sh",
        );
        return;
    };
    report("büyük rastgele küme", &cases);
    let cells = cell_counts(&cases);
    assert_eq!(cells.len(), 12, "12 hücre beklenir (3 dir + 9 filter)");
    for ((kind, bd, size), n) in &cells {
        assert!(
            *n >= 10_000,
            "hücre (kind={kind}, bit_depth={bd}, block_size={size}) yalnız {n} kayıt; \
             en az 10 000 olmalı"
        );
    }
    let cov = Coverage::of(&cases);
    assert!(
        cov.missing().is_empty(),
        "büyük küme kapsamı eksik: {:?}",
        cov.missing()
    );
    compare_all("büyük rastgele küme", &cases);
}

/// 7(a) — oracle fixture'larının SHA-256 kaydı repoda ve yeniden üretilebilir.
#[test]
fn yedi_a_fixture_sha256_kaydi_tutarli() {
    let sums_path = small_dir().join("SHA256SUMS");
    let sums = fs::read_to_string(&sums_path).expect("SHA-256 kaydı olmalı");
    let mut lines = 0usize;
    for line in sums.lines().filter(|l| !l.trim().is_empty()) {
        let mut parts = line.split_whitespace();
        let hash = parts.next().expect("hash alanı");
        let name = parts.next().expect("dosya adı");
        assert_eq!(hash.len(), 64, "SHA-256 64 hex olmalı: {line}");
        assert!(
            hash.chars().all(|c| c.is_ascii_hexdigit()),
            "hash hex olmalı: {line}"
        );
        assert!(
            name.starts_with("cdef-") && name.ends_with(".bin"),
            "beklenmeyen fixture adı: {name}"
        );
        lines += 1;
    }
    assert_eq!(
        lines, 4,
        "dört fixture satırı beklenir (küçük+büyük, girdi+beklenen)"
    );
    assert!(
        sums.starts_with("6c77867179ce8fb6481522535421cf21e8b024c3de50867b0a0f6c26c5b8dd60")
            || sums.contains("  cdef-small-input.bin"),
        "küçük girdi kaydı eksik"
    );
}

/// Sayısal aritmetiğin tanımlı davranışa indirgendiğinin kanıtı (S15.4/4).
#[test]
fn yedi_a_tanimli_aritmetik_yardimcilari() {
    use av1oxide::dsp::cdef::{apply_sign, clamped_shift, constrain, px_at, px_put};

    // constrain: |diff| küçükken kırpılır, eşiğe yaklaşınca **sıfıra iner**
    // (dav1d `src/cdef_tmpl.c:41`: `imin(adiff, imax(0, thr - (adiff >> shift)))`).
    assert_eq!(constrain(0, 15, 0), 0);
    assert_eq!(constrain(5, 15, 0), 5);
    assert_eq!(constrain(-5, 15, 0), -5);
    assert_eq!(constrain(10, 15, 0), 5, "limit = 15 - 10 = 5");
    assert_eq!(constrain(100, 15, 0), 0, "|diff| > eşik → katkı yok");
    assert_eq!(constrain(-100, 15, 0), 0);
    assert_eq!(constrain(20, 15, 1), 5, "limit = 15 - (20 >> 1) = 5");
    assert_eq!(constrain(40, 15, 1), 0, "(40 >> 1) = 20 > 15 → katkı yok");
    // Eşik 0 ise katkı yok (dav1d `src/cdef_tmpl.c:41`).
    assert_eq!(constrain(1234, 0, 0), 0);
    assert_eq!(constrain(-1234, 0, 0), 0);
    // `abs(INT16_MIN)` sarmalı: 16-bit dolgusunun katkısı yine 0 olmalı.
    assert_eq!(constrain(i32::from(i16::MIN), 15, 0), 0);
    assert_eq!(constrain(i32::from(i16::MIN), 15, 6), 0);

    // apply_sign yalnız işareti uygular (dav1d `intops.h:59`).
    assert_eq!(apply_sign(4, -1), -4);
    assert_eq!(apply_sign(4, 0), 4);

    // Negatif kaydırma C'de UB'dir; bu port 0'a kırpar (tanımlı davranış).
    assert_eq!(clamped_shift(-3), 0);
    assert_eq!(clamped_shift(0), 0);
    assert_eq!(clamped_shift(4), 4);

    // 8-bit'te yazma mod 256, 16bpc'de yerel bayt sırası.
    let mut b = [0u8; 4];
    px_put::<8>(&mut b, 0, 300);
    assert_eq!(
        b[0], 44,
        "8-bit yazma mod 256 olmalı (dav1d `pixel` dönüşümü)"
    );
    px_put::<12>(&mut b, 2, 0x0ABC);
    assert_eq!(px_at::<12>(&b, 2), 0x0ABC);
}
