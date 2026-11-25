//! S2b.1 — skaler Rust ↔ dav1d C **bit-exact** ters dönüşüm differential'ı.
//!
//! Oracle, dav1d'nin C yolunu (`itx_tmpl.c` + `itx_1d.c`, asm = OFF) ayrı bir
//! süreçte çalıştırır (`scripts/oracle/itx_oracle.c`) ve beklenen dosyaları
//! üretir; bu test aynı girdilerle üretim crate'inin skaler çekirdeğini
//! koşturur, **hedef pikselleri** ve **katsayı tamponunun çağrı sonrası
//! durumunu** karşılaştırır. Üretim crate'i oracle'a **bağlanmaz** (SRC-11).
//!
//! Fixture'lar:
//!
//! | Küme | Konum | Git |
//! |---|---|---|
//! | Küçük sınır kümesi (156 hücre × 6 kayıt × 3 bit derinliği = 2 808) | `tests/fixtures/itx/` | ✅ girer |
//! | Büyük rastgele küme (156 hücre × 1 000 kayıt × 3 = 468 000) | `.cache/itx-oracle/` | ⬜ girmez |
//!
//! Yeniden üretim (tek komut): `bash scripts/oracle/run_itx_oracle.sh`.
//! Büyük küme `--full` ile üretilir; testi
//! `cargo test --release -p av1oxide --test itx_scalar_oracle -- --ignored`
//! koşar.
//!
//! ## Doğrulanan sözleşme (her kayıt için)
//!
//! 1. **Hedef pikseller** (w×h, u16) dav1d C ile bit-aynı.
//! 2. **Katsayı tamponu**: dav1d'nin yazdığı `sw·sh` bölgesi tamamen sıfır;
//!    `sw·sh..1024` kuyruğu **değişmemiş** (dav1d bu bölgeye dokunmaz).
//! 3. Rust, C ile **aynı** beklenen dosya biçimini üretir; SHA-256
//!    karşılaştırması `scripts/oracle/run_itx_oracle.sh` tarafından yapılır.
//!
//! Piksel başlangıç değerleri iki tarafta da aynı LCG ile üretilir:
//! `state = state·6364136223846793005 + 1442695040888963407`,
//! `px = (state >> 33) & bitdepth_max`. Tohum
//! `0x9E3779B97F4A7C15·(kayıt+1) ^ (bpc << 51)`.

#![forbid(unsafe_code)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::dsp::itx::{TX_SIZES, TX_TYPES, TxSize, TxType, is_supported};
use av1oxide::frame::BitDepth;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// F3c: bit derinliği tablosunun skaler yuvası (`itxfm_add[tx][txtp]`).
#[allow(clippy::too_many_arguments)]
fn inverse_transform_add(
    dst: &mut [u8],
    stride: usize,
    coeff: &mut [i32],
    eob: usize,
    tx: TxSize,
    txtp: TxType,
    bd: av1oxide::frame::BitDepth,
) {
    let f = av1oxide::dsp::itx::ItxDsp::scalar_for(bd).itxfm_add[tx.index()][txtp.index()];
    f(dst, stride, coeff, eob, tx, txtp, bd);
}

const MAX_COEF: usize = 1024;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .to_path_buf()
}

fn small_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("itx")
}

fn cache_dir() -> PathBuf {
    workspace_root().join(".cache").join("itx-oracle")
}

fn bit_depth(bpc: u8) -> BitDepth {
    BitDepth::from_bits(u32::from(bpc)).expect("bpc 8/10/12")
}

fn read_u8<R: Read>(r: &mut R) -> u8 {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).expect("okuma");
    b[0]
}

fn read_u16<R: Read>(r: &mut R) -> u16 {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).expect("okuma");
    u16::from_le_bytes(b)
}

fn read_u32<R: Read>(r: &mut R) -> u32 {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).expect("okuma");
    u32::from_le_bytes(b)
}

fn read_i16<R: Read>(r: &mut R) -> i16 {
    i16::from_le_bytes([read_u8(r), read_u8(r)])
}

fn read_i32<R: Read>(r: &mut R) -> i32 {
    i32::from_le_bytes([read_u8(r), read_u8(r), read_u8(r), read_u8(r)])
}

fn write_u16<W: Write>(w: &mut W, v: u16) {
    w.write_all(&v.to_le_bytes()).expect("yazma");
}

fn write_u32<W: Write>(w: &mut W, v: u32) {
    w.write_all(&v.to_le_bytes()).expect("yazma");
}

/// LCG: hedef piksel başlangıç değerleri (C oracle ile birebir).
fn pix_seed(record_index: u64, bpc: u8) -> u64 {
    0x9E37_79B9_7F4A_7C15u64.wrapping_mul(record_index + 1) ^ (u64::from(bpc) << 51)
}

fn pix_next(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Katsayı kuyruğu (`sw·sh..1024`) — dav1d bu bölgeye dokunmaz. C oracle ile
/// **birebir aynı** deterministik desen; böylece çağrı sonrası **tam 1024**
/// girişlik tampon durumu iki tarafta da aynıdır (yazılan bölge sıfır, kuyruk
/// değişmemiş).
fn tail_value(index: usize, record_index: u64, bpc: u8) -> i32 {
    let x = (index as u32)
        .wrapping_mul(2_654_435_761)
        .wrapping_add((record_index as u32).wrapping_mul(40_503))
        .wrapping_add(u32::from(bpc) * 97);
    if bpc == 8 {
        i32::from(x as i16)
    } else {
        x as i32
    }
}

fn fill_tail(coeff: &mut [i32; MAX_COEF], start: usize, record_index: u64, bpc: u8) {
    for (index, value) in coeff.iter_mut().enumerate().skip(start) {
        *value = tail_value(index, record_index, bpc);
    }
}

struct Stats {
    records: BTreeMap<(u8, u8, u8), (u64, u64)>,
    pixels_total: u64,
    coeff_cells_total: u64,
}

impl Stats {
    fn new() -> Self {
        Self {
            records: BTreeMap::new(),
            pixels_total: 0,
            coeff_cells_total: 0,
        }
    }

    fn record(&mut self, tx: u8, txtp: u8, bpc: u8, pixels: u64, coeff_cells: u64, ok: bool) {
        let entry = self.records.entry((tx, txtp, bpc)).or_insert((0, 0));
        entry.0 += 1;
        if ok {
            entry.1 += 1;
            self.pixels_total += pixels;
            self.coeff_cells_total += coeff_cells;
        }
    }
}

struct Header {
    bpc: u8,
    profile: u8,
    count: u32,
}

fn read_header<R: Read>(r: &mut R, expected_magic: &[u8; 4]) -> Header {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).expect("başlık okunamadı");
    assert_eq!(&magic, expected_magic, "dosya büyüsü yanlış");
    let version = read_u32(r);
    assert_eq!(version, 1, "bilinmeyen sürüm");
    let bpc = read_u8(r);
    let profile = read_u8(r);
    let pad = read_u16(r);
    assert_eq!(pad, 0);
    let count = read_u32(r);
    Header {
        bpc,
        profile,
        count,
    }
}

fn write_header<W: Write>(w: &mut W, magic: &[u8; 4], header: &Header) {
    w.write_all(magic).expect("yazma");
    write_u32(w, 1);
    w.write_all(&[header.bpc, header.profile]).expect("yazma");
    write_u16(w, 0);
    write_u32(w, header.count);
}

fn write_record_header<W: Write>(w: &mut W, tx: u8, txtp: u8, bpc: u8, eob: u16) {
    w.write_all(&[tx, txtp, bpc, 0]).expect("yazma");
    write_u16(w, eob);
    write_u16(w, 0);
}

/// Bir kümenin tüm kayıtlarını koşar; ilk uyuşmazlıkta ayrıntılı panikler.
fn run_set(profile_name: &str, input_dir: &Path, rust_out_dir: &Path) -> Stats {
    fs::create_dir_all(rust_out_dir).expect("çıktı dizini");
    let mut stats = Stats::new();
    let mut grand_ok = 0u64;
    let mut grand_total = 0u64;
    for bpc in [8u8, 10, 12] {
        let in_path = input_dir.join(format!("itx-{profile_name}-{bpc}-input.bin"));
        let exp_path = input_dir.join(format!("itx-{profile_name}-{bpc}-expect.bin"));
        let rust_path = rust_out_dir.join(format!("itx-{profile_name}-{bpc}-rust.bin"));
        let (input, expect) = match (File::open(&in_path), File::open(&exp_path)) {
            (Ok(i), Ok(e)) => (i, e),
            _ => panic!(
                "fixture yok: {} — üretmek için: bash scripts/oracle/run_itx_oracle.sh{}",
                in_path.display(),
                if profile_name == "large" {
                    " --full"
                } else {
                    ""
                }
            ),
        };
        let mut input = BufReader::new(input);
        let mut expect = BufReader::new(expect);
        let in_header = read_header(&mut input, b"ITXI");
        let exp_header = read_header(&mut expect, b"ITXE");
        assert_eq!(in_header.bpc, bpc);
        assert_eq!(exp_header.bpc, bpc);
        assert_eq!(in_header.profile, exp_header.profile);
        assert_eq!(in_header.count, exp_header.count);
        let profile = in_header.profile;

        let rust_file = File::create(&rust_path).expect("rust çıktısı");
        let mut rust_file = BufWriter::new(rust_file);
        write_header(&mut rust_file, b"ITXE", &in_header);

        let mut coeff = [0i32; MAX_COEF];
        let mut local_pixels = 0u64;
        let mut local_coeff_cells = 0u64;
        let mut cell_block = 0u32;
        let mut cell_block_ok = 0u32;
        let mut current_cell: Option<(u8, u8)> = None;
        let mut first_mismatch: Option<(u64, u8, u8, u8, usize, i32, i32)> = None;

        for record_index in 0..u64::from(in_header.count) {
            let tx = read_u8(&mut input);
            let txtp = read_u8(&mut input);
            let rec_bpc = read_u8(&mut input);
            let pad = read_u8(&mut input);
            assert_eq!(pad, 0);
            let eob = read_u16(&mut input);
            let pad = read_u16(&mut input);
            assert_eq!(pad, 0);
            assert_eq!(rec_bpc, bpc);

            let et_tx = read_u8(&mut expect);
            let et_txtp = read_u8(&mut expect);
            let et_bpc = read_u8(&mut expect);
            let epad = read_u8(&mut expect);
            assert_eq!((et_tx, et_txtp, et_bpc, epad), (tx, txtp, rec_bpc, 0));
            let et_eob = read_u16(&mut expect);
            let epad = read_u16(&mut expect);
            assert_eq!((et_eob, epad), (eob, 0));

            let tx_size = TxSize::from_index(usize::from(tx)).expect("tx");
            let txtp_ty = TxType::from_index(usize::from(txtp)).expect("txtp");
            assert!(
                is_supported(tx_size, txtp_ty),
                "desteklenmeyen hücre: {tx_size:?}/{txtp_ty:?}"
            );
            let w = tx_size.width();
            let h = tx_size.height();
            let sw = w.min(32);
            let sh = h.min(32);
            let bpp = if bpc == 8 { 1 } else { 2 };
            let stride = w * bpp;

            // --- girdi katsayıları (dav1d'nin okuduğu sw·sh bölgesi) -------
            for slot in coeff.iter_mut().take(sw * sh) {
                *slot = if bpc == 8 {
                    i32::from(read_i16(&mut input))
                } else {
                    read_i32(&mut input)
                };
            }
            fill_tail(&mut coeff, sw * sh, record_index, bpc);

            // --- hedef piksel başlangıcı (C ile aynı LCG) ------------------
            let max_sample = (1u32 << bpc) - 1;
            let mut dst = vec![0u8; w * h * bpp];
            let mut state = pix_seed(record_index, bpc);
            let variant = if profile == 0 {
                (record_index % 6) as u8
            } else {
                0xff
            };
            for p in 0..w * h {
                let v = pix_next(&mut state) & max_sample;
                let value = if profile == 0 && (variant == 1 || variant == 5) {
                    max_sample
                } else if profile == 0 && (variant == 0 || variant == 2) {
                    0
                } else {
                    v
                };
                if bpc == 8 {
                    dst[p] = value as u8;
                } else {
                    dst[2 * p..2 * p + 2].copy_from_slice(&(value as u16).to_le_bytes());
                }
            }

            // --- skaler çekirdek ------------------------------------------
            inverse_transform_add(
                &mut dst,
                stride,
                &mut coeff,
                usize::from(eob),
                tx_size,
                txtp_ty,
                bit_depth(bpc),
            );

            // --- doğrulama 1: katsayı tamponu ------------------------------
            let mut ok = true;
            for (i, value) in coeff.iter().enumerate().take(sw * sh) {
                if *value != 0 {
                    ok = false;
                    if first_mismatch.is_none() {
                        first_mismatch = Some((record_index, tx, txtp, bpc, i, *value, 0));
                    }
                }
            }
            let mut tail = [0i32; MAX_COEF];
            fill_tail(&mut tail, sw * sh, record_index, bpc);
            for i in sw * sh..MAX_COEF {
                if coeff[i] != tail[i] {
                    ok = false;
                    if first_mismatch.is_none() {
                        first_mismatch = Some((record_index, tx, txtp, bpc, i, coeff[i], tail[i]));
                    }
                }
            }

            // --- Rust beklenen dosyasına kayıt başlığı -------------------
            write_record_header(&mut rust_file, tx, txtp, bpc, eob);

            // --- doğrulama 2: hedef pikseller -----------------------------
            let mut pixels = 0u64;
            for p in 0..w * h {
                let got = if bpc == 8 {
                    i32::from(dst[p])
                } else {
                    i32::from(u16::from_le_bytes([dst[2 * p], dst[2 * p + 1]]))
                };
                let want = i32::from(read_u16(&mut expect));
                if got != want {
                    ok = false;
                    if first_mismatch.is_none() {
                        first_mismatch = Some((record_index, tx, txtp, bpc, p, got, want));
                    }
                }
                pixels += 1;
                write_u16(&mut rust_file, got as u16);
            }

            if current_cell != Some((tx, txtp)) {
                if let Some((ctx_tx, ctx_txtp)) = current_cell {
                    assert_eq!(
                        cell_block, cell_block_ok,
                        "hücre içi uyuşmazlık: tx={ctx_tx} txtp={ctx_txtp}"
                    );
                }
                current_cell = Some((tx, txtp));
                cell_block = 0;
                cell_block_ok = 0;
            }
            cell_block += 1;
            if ok {
                cell_block_ok += 1;
            }
            stats.record(tx, txtp, bpc, pixels, (sw * sh) as u64, ok);
            local_pixels += pixels;
            local_coeff_cells += (sw * sh) as u64;
            grand_total += 1;
            if ok {
                grand_ok += 1;
            }
        }
        if let Some((ctx_tx, ctx_txtp)) = current_cell {
            assert_eq!(
                cell_block, cell_block_ok,
                "hücre içi uyuşmazlık: tx={ctx_tx} txtp={ctx_txtp}"
            );
        }
        // Beklenen dosyada fazladan bayt olmamalı.
        let mut extra = [0u8; 1];
        assert_eq!(
            expect.read(&mut extra).expect("okuma"),
            0,
            "beklenen dosyada fazladan veri var"
        );
        rust_file.flush().expect("flush");
        let got = fs::metadata(&rust_path).expect("rust çıktısı").len();
        let want = fs::metadata(&exp_path).expect("beklenen").len();
        assert_eq!(got, want, "rust ve C beklenen dosya boyutları farklı");

        if let Some((idx, tx, txtp, bpc, pos, got, want)) = first_mismatch {
            panic!(
                "{profile_name}: ilk uyuşmazlık kayıt {idx} (tx={tx} txtp={txtp} {bpc}bpc) \
                 indeks {pos}: rust={got} dav1d={want}"
            );
        }
        println!(
            "[{profile_name} {bpc}-bit] kayıt {}/{} · hedef piksel {} · katsayı hücresi {}",
            in_header.count, in_header.count, local_pixels, local_coeff_cells
        );
    }
    // Kapsam: 156 hücre × 3 bit derinliği.
    let cells = stats.records.len();
    assert_eq!(cells, 156 * 3, "hücre sayısı 156×3 olmalı");
    for ((tx, txtp, bpc), (total, ok)) in &stats.records {
        assert_eq!(total, ok, "hücre uyuşmazlığı: tx={tx} txtp={txtp} {bpc}bpc");
        assert!(*ok > 0);
    }
    println!(
        "[{profile_name}] TOPLAM kayıt {grand_ok}/{grand_total} eşleşti · hücre {}",
        stats.records.len()
    );
    // Kapsam CSV'si (belge tablosu için).
    let mut csv = String::from("tx,txtp,bpc,kayit,eslesen\n");
    for ((tx, txtp, bpc), (total, ok)) in &stats.records {
        csv.push_str(&format!("{tx},{txtp},{bpc},{total},{ok}\n"));
    }
    fs::write(
        rust_out_dir.join(format!("itx-kapsam-{profile_name}.csv")),
        csv,
    )
    .expect("kapsam csv");
    stats
}

#[test]
fn kucuk_sinir_kumesi_bit_exact() {
    let dir = small_dir();
    assert!(
        dir.join("itx-small-8-input.bin").exists(),
        "küçük fixture git'te olmalı: {}",
        dir.display()
    );
    let stats = run_set("small", &dir, &cache_dir());
    for ((tx, txtp, bpc), (total, ok)) in &stats.records {
        assert_eq!(total, ok);
        assert_eq!(
            *total, 6,
            "küçük kümede hücre başına 6 kayıt: {tx}/{txtp}/{bpc}"
        );
    }
    // Küçük kümede 156 hücre × 6 = 936 kayıt (bit derinliği başına).
    assert_eq!(stats.records.len(), 468);
}

#[test]
#[ignore = "468 000 blok (hücre başına 1 000) — scripts/oracle/run_itx_oracle.sh --full"]
fn buyuk_kume_bit_exact() {
    let dir = cache_dir();
    assert!(
        dir.join("itx-large-8-input.bin").exists(),
        "büyük küme yok: bash scripts/oracle/run_itx_oracle.sh --full"
    );
    let stats = run_set("large", &dir, &dir);
    for ((tx, txtp, bpc), (total, ok)) in &stats.records {
        assert_eq!(total, ok);
        assert_eq!(
            *total, 1000,
            "büyük kümede hücre başına 1 000 kayıt: {tx}/{txtp}/{bpc}"
        );
    }
    assert_eq!(stats.records.len(), 468);
    // Hücre türü kapsamı: bütün (tx,txtp,bpc) hücreleri tabloda.
    for tx in TX_SIZES {
        for txtp in TX_TYPES {
            for bpc in [8u8, 10, 12] {
                if is_supported(tx, txtp) {
                    assert!(
                        stats
                            .records
                            .contains_key(&(tx.index() as u8, txtp.index() as u8, bpc)),
                        "eksik hücre: {tx:?}/{txtp:?}/{bpc}"
                    );
                }
            }
        }
        for bpc in [8u8, 10, 12] {
            if is_supported(tx, TxType::WhtWht) {
                assert!(stats.records.contains_key(&(
                    tx.index() as u8,
                    TxType::WhtWht.index() as u8,
                    bpc
                )));
            }
        }
    }
}

/// Uç katsayı hücreleri — `cf_max = (1 << (bpc+7)) - 1` sınırına kadar
/// (`dav1d src/recon_tmpl.c:599`), geçerli eob dağılımı; satır ve sütun
/// kırpma yollarını tetikler (`src/itx_tmpl.c:76-84`). Hücre başına 1 000 blok
/// (468 000). Üretim: `bash scripts/oracle/run_itx_oracle.sh --full`.
#[test]
#[ignore = "uç katsayı kümesi: 468 000 blok — scripts/oracle/run_itx_oracle.sh --full"]
fn uc_kume_bit_exact() {
    let dir = cache_dir();
    assert!(
        dir.join("itx-extreme-8-input.bin").exists(),
        "uç küme yok: bash scripts/oracle/run_itx_oracle.sh --full"
    );
    let stats = run_set("extreme", &dir, &dir);
    for ((tx, txtp, bpc), (total, ok)) in &stats.records {
        assert_eq!(
            total, ok,
            "uç hücre uyuşmazlığı: tx={tx} txtp={txtp} {bpc}bpc"
        );
        assert_eq!(*total, 1000, "uç kümede hücre başına 1 000 kayıt");
    }
    assert_eq!(stats.records.len(), 468);
}
