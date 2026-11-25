//! S2a.3 — CDF varsayılanları ve `update` semantiği: dav1d C oracle fixture'ları.
//!
//! Oracle, dav1d `src/cdf.c` **C yolunu** (asm = OFF) ayrı bir süreçte koşturur
//! ve beklenen bağlam baytlarını fixture'a yazar; bu test üretim crate'inin
//! `entropy::cdf` modülünü aynı girdiyle koşturup **bayt bayt** karşılaştırır.
//! Üretim crate'i oracle'a **bağlanmaz** (SRC-11).
//!
//! Fixture'lar `scripts/oracle/run_cdf_oracle.sh --update-fixtures` ile
//! yenilenir; canlı differential `python3 scripts/oracle/cdf_diff.py`.
//!
//! | Dosya | İçerik |
//! |---|---|
//! | `cdf-default.bin` | 4 kategori × `copy` sonrası bağlam (4 × 13472 bayt) |
//! | `cdf-update.bin`  | 4 kategori × 2 kare türü × `update` sonrası bağlam (8 × 13472) |
//!
//! `update`, yabancı bir `dst` (deterministik desen) üzerine uygulanır; böylece
//! kopyalanmayan bölgelerin (`intrabc`, `kfym`, anahtar karede `y_mode..mv`)
//! korunduğu da doğrulanır.

#![forbid(unsafe_code)]

use av1oxide::entropy::cdf::{CDF_BYTE_COUNT, CdfContext, FrameKind, qcat_for_qidx};
use std::fs;
use std::path::PathBuf;

/// Fixture dizini.
fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("entropy")
}

/// xorshift64* — `scripts/oracle/entropy_common.h` ile aynı (fixture üreteci).
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Oracle'daki `pattern_fill`: bütün hücreler (dolgu dâhil) PRNG ile dolar.
fn pattern_fill(seed: u64) -> CdfContext {
    let mut rng = Rng::new(seed);
    let mut ctx = CdfContext::from_default(0);
    for cell in ctx.cells_mut().iter_mut() {
        *cell = (rng.next() & 0xffff) as u16;
    }
    ctx
}

fn bytes_of(ctx: &CdfContext) -> Vec<u8> {
    let mut buf = vec![0u8; CDF_BYTE_COUNT];
    ctx.write_le_bytes(&mut buf);
    buf
}

fn read_fixture(name: &str) -> Vec<u8> {
    let path = fixture_dir().join(name);
    fs::read(&path).unwrap_or_else(|err| panic!("fixture okunamadı: {} ({err})", path.display()))
}

/// Dört kuantizasyon kategorisi için `copy` (dav1d `cdf_thread_copy`) bayt eşit.
#[test]
fn varsayilan_baglamlar_bayt_esit() {
    let data = read_fixture("cdf-default.bin");
    assert_eq!(data.len(), 4 * CDF_BYTE_COUNT, "fixture boyutu");
    for qcat in 0..4u8 {
        let ctx = CdfContext::from_default(qcat);
        let beklenen = &data[qcat as usize * CDF_BYTE_COUNT..(qcat as usize + 1) * CDF_BYTE_COUNT];
        assert_eq!(
            bytes_of(&ctx),
            beklenen,
            "qcat {qcat}: varsayılan bağlam baytları dav1d ile farklı"
        );
    }
}

/// `update` sonrası bağlamlar (anahtar/kamera içi + inter/switch) bayt eşit.
#[test]
fn update_baglamlari_bayt_esit() {
    let data = read_fixture("cdf-update.bin");
    assert_eq!(data.len(), 8 * CDF_BYTE_COUNT, "fixture boyutu");
    for qcat in 0..4u8 {
        let src_seed = 0x5EED_0000u64 + u64::from(qcat) * 0x101;
        let dst_seed = 0xC0DE_0000u64 + u64::from(qcat) * 0x101;
        let src = pattern_fill(src_seed);

        for (index, kind) in [FrameKind::KeyOrIntra, FrameKind::InterOrSwitch]
            .into_iter()
            .enumerate()
        {
            let mut dst = pattern_fill(dst_seed);
            dst.update(&src, kind);
            let offset = (usize::from(qcat) * 2 + index) * CDF_BYTE_COUNT;
            let beklenen = &data[offset..offset + CDF_BYTE_COUNT];
            assert_eq!(
                bytes_of(&dst),
                beklenen,
                "qcat {qcat} · {kind:?}: update sonrası baytlar farklı"
            );
        }
    }
}

/// `init_static(qidx)` kategori seçimi dav1d sınırlarıyla aynı (0/20/60/120).
#[test]
fn qcat_secimi_sinirlarda() {
    for (qidx, beklenen) in [
        (0u8, 0u8),
        (20, 0),
        (21, 1),
        (60, 1),
        (61, 2),
        (120, 2),
        (121, 3),
        (255, 3),
    ] {
        assert_eq!(qcat_for_qidx(qidx), beklenen, "qidx={qidx}");
    }
}
