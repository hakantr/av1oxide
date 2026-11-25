//! S2b.2 — skaler Rust ↔ dav1d C **bit-exact** intra tahmin differential'ı.
//!
//! Oracle, dav1d'nin C yolunu (`src/ipred_tmpl.c`, `src/pal.c`, asm = OFF) ayrı
//! bir süreçte çalıştırır (`scripts/oracle/ipred_oracle.c`) ve beklenen
//! dosyaları üretir; bu test aynı girdilerle üretim crate'inin skaler
//! çekirdeklerini koşturur, **çıktı piksellerini** (ve `cfl_ac` /
//! `pal_idx_finish` çıktı tamponlarını) dav1d C ile karşılaştırır. Üretim
//! crate'i oracle'a **bağlanmaz** (SRC-11).
//!
//! Fixture'lar:
//!
//! | Küme | Konum | Git |
//! |---|---|---|
//! | Küçük sınır kümesi (389 hücre/bit derinliği × 2–108 kayıt × 3) | `tests/fixtures/ipred/` | ✅ girer |
//! | Büyük rastgele küme (389 hücre/bit derinliği × 1 000 × 3 = 1 167 000) | `.cache/ipred-oracle/` | ⬜ girmez |
//!
//! Yeniden üretim (tek komut): `bash scripts/oracle/run_ipred_oracle.sh`.
//! Büyük küme `--full` ile üretilir; testi
//! `cargo test --release -p av1oxide --test ipred_scalar_oracle -- --ignored`
//! koşar.
//!
//! ## Kapsam (kabul ölçütü 1)
//!
//! Hücre anahtarı **gerçek `bitdepth_max`**'ı içerir: oracle 10-bit koşuda
//! yalnız 0x3ff, 12-bit koşuda yalnız 0xfff üretir. Her (işlev/mod, blok
//! boyutu, bit derinliği) hücresi en az **1 000** çağrı alır; yönlü modlarda
//! (Z1/Z2/Z3) her hücrede **27 açı × 4 bayrak = 108** birleşim en az bir kez
//! kapsanır. `pal_idx_finish` bit derinliğinden bağımsızdır ve **ayrı satırda**
//! raporlanır. Hücre sayısı, bit derinliği başına en küçük/en büyük çağrı ve
//! `pal_idx_finish` satırı test çıktısında yazdırılır ve makinece doğrulanır.
//!
//! Mutasyon kanıtı için `IPRED_BPC=10` / `IPRED_BPC=12` yalnız o bit
//! derinliğini koşar (varsayılan: üçü birden).
//!
//! ## Dosya biçimi
//!
//! `scripts/oracle/ipred_oracle.h` ile birebir (40 baytlık kayıt başlığı).
//! Piksel değerleri dosyada u16 (küçük sonlu); `cfl_ac` çıktısı i16.

#![forbid(unsafe_code)]
// Piksel indeksleri uyuşmazlık raporunda konum olarak kullanılır.
#![allow(clippy::needless_range_loop)]

#[path = "common/helpers.rs"]
mod common;

use av1oxide::dsp::ipred::{CflPredMode, Edge, IntraPredMode};
use av1oxide::dsp::scalar::pal::pal_idx_finish;
use av1oxide::frame::BitDepth;
use common::ipred_slots::{
    cfl_ac_420, cfl_ac_422, cfl_ac_444, cfl_pred_dc, cfl_pred_dc_128, cfl_pred_left_dc,
    cfl_pred_top_dc, intra_pred, pal_pred,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const KIND_COUNT: usize = 23;
const MAX_RECORD_PAYLOAD: usize = 64 * 64 * 2 + 257 * 2;

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
        .join("ipred")
}

fn cache_dir() -> PathBuf {
    workspace_root().join(".cache").join("ipred-oracle")
}

fn bit_depth(bpc: u8) -> BitDepth {
    BitDepth::from_bits(u32::from(bpc)).expect("bpc 8/10/12")
}

// ---------------------------------------------------------------- okuma ---

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

fn write_u16<W: Write>(w: &mut W, v: u16) {
    w.write_all(&v.to_le_bytes()).expect("yazma");
}

fn write_u32<W: Write>(w: &mut W, v: u32) {
    w.write_all(&v.to_le_bytes()).expect("yazma");
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Header {
    bpc: u8,
    count: u32,
}

fn read_header<R: Read>(r: &mut R, magic: &[u8; 4]) -> Header {
    let mut found = [0u8; 4];
    r.read_exact(&mut found).expect("başlık");
    assert_eq!(&found, magic, "dosya büyüsü yanlış");
    assert_eq!(read_u32(r), 1, "bilinmeyen sürüm");
    let bpc = read_u8(r);
    let mut pad = [0u8; 3];
    r.read_exact(&mut pad).expect("pad");
    assert_eq!(pad, [0u8; 3]);
    Header {
        bpc,
        count: read_u32(r),
    }
}

fn write_header<W: Write>(w: &mut W, header: &Header) {
    w.write_all(b"IPRE").expect("yazma");
    write_u32(w, 1);
    w.write_all(&[header.bpc, 0, 0, 0]).expect("yazma");
    write_u32(w, header.count);
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Record {
    kind: u8,
    aux: u8,
    w: usize,
    h: usize,
    bw: usize,
    bh: usize,
    back: usize,
    front: usize,
    angle: usize,
    max_width: usize,
    max_height: usize,
    bitdepth_max: u16,
    alpha: i8,
    w_pad: usize,
    h_pad: usize,
    pal_sz: usize,
    payload_len: usize,
}

fn read_record<R: Read>(r: &mut R) -> Record {
    let kind = read_u8(r);
    let aux = read_u8(r);
    assert_eq!(read_u8(r), 0, "kind pad");
    assert_eq!(read_u8(r), 0, "kind pad");
    let w = usize::from(read_u16(r));
    let h = usize::from(read_u16(r));
    let bw = usize::from(read_u16(r));
    let bh = usize::from(read_u16(r));
    let back = usize::from(read_u16(r));
    let front = usize::from(read_u16(r));
    let angle = read_u32(r) as usize;
    let max_width = read_u32(r) as usize;
    let max_height = read_u32(r) as usize;
    let bitdepth_max = read_u16(r);
    let alpha = read_u8(r) as i8;
    let w_pad = usize::from(read_u8(r));
    let h_pad = usize::from(read_u8(r));
    let pal_sz = usize::from(read_u8(r));
    assert_eq!(read_u16(r), 0, "record pad");
    let payload_len = read_u32(r) as usize;
    Record {
        kind,
        aux,
        w,
        h,
        bw,
        bh,
        back,
        front,
        angle,
        max_width,
        max_height,
        bitdepth_max,
        alpha,
        w_pad,
        h_pad,
        pal_sz,
        payload_len,
    }
}

fn write_record<W: Write>(w: &mut W, record: &Record) {
    w.write_all(&[record.kind, record.aux, 0, 0])
        .expect("yazma");
    for value in [
        record.w,
        record.h,
        record.bw,
        record.bh,
        record.back,
        record.front,
    ] {
        write_u16(w, value as u16);
    }
    write_u32(w, record.angle as u32);
    write_u32(w, record.max_width as u32);
    write_u32(w, record.max_height as u32);
    write_u16(w, record.bitdepth_max);
    w.write_all(&[
        record.alpha as u8,
        record.w_pad as u8,
        record.h_pad as u8,
        record.pal_sz as u8,
    ])
    .expect("yazma");
    write_u16(w, 0);
    write_u32(w, record.payload_len as u32);
}

// ------------------------------------------------------------ hücreler ---

/// (kind, aux, w, h, **gerçek `bitdepth_max`**): 10-bit ve 12-bit hücreleri
/// ayrı sayılır; `pal_idx_finish` 0xff ile ayrı anahtar taşır.
type CellKey = (u8, u8, usize, usize, u16);

fn cells_for_mode(mode: u8) -> Vec<(usize, usize)> {
    let w_max = if mode == 13 { 32 } else { 64 };
    let mut cells = Vec::new();
    let mut w = 4;
    while w <= w_max {
        let h_min = (w / 4).max(4);
        let h_max = (w * 4).min(w_max);
        let mut h = 4;
        while h <= h_max {
            if h >= h_min {
                cells.push((w, h));
            }
            h <<= 1;
        }
        w <<= 1;
    }
    cells
}

fn cells_for_cfl_ac(layout: u8) -> Vec<(usize, usize)> {
    let ss_hor = layout != 2;
    let ss_ver = layout == 0;
    let w_max = if ss_hor { 16 } else { 32 };
    let h_max = if ss_ver { 16 } else { 32 };
    let mut cells = Vec::new();
    let mut w = 4;
    while w <= w_max {
        let h_min = (w / 4).max(4);
        let h_max2 = (w * 4).min(h_max);
        let mut h = 4;
        while h <= h_max2 {
            if h >= h_min {
                cells.push((w, h));
            }
            h <<= 1;
        }
        w <<= 1;
    }
    cells
}

fn cells_for_cfl_pred() -> Vec<(usize, usize)> {
    let mut cells = Vec::new();
    let mut w = 4;
    while w <= 32 {
        let h_min = (w / 4).max(4);
        let h_max = (w * 4).min(32);
        let mut h = 4;
        while h <= h_max {
            if h >= h_min {
                cells.push((w, h));
            }
            h <<= 1;
        }
        w <<= 1;
    }
    cells
}

fn cells_for_pal() -> Vec<(usize, usize)> {
    cells_for_mode(0)
}

fn cells_for_pal_idx() -> Vec<(usize, usize)> {
    let mut cells = Vec::new();
    let mut bw = 4;
    while bw <= 64 {
        let bh_min = (bw / 4).max(4);
        let bh_max = (bw * 4).min(64);
        let mut bh = 4;
        while bh <= bh_max {
            if bh >= bh_min {
                cells.push((bw, bh));
            }
            bh <<= 1;
        }
        bw <<= 1;
    }
    cells
}

/// `(1 << bpc) - 1` — oracle'ın `bpc_bitdepth_max()` üretimiyle **aynı** sabit.
fn expected_bitdepth_max(bpc: u8) -> u16 {
    (1u16 << bpc) - 1
}

/// Beklenen hücre listesi ve hücre başına çağrı sayısı.
///
/// Hücre anahtarı **gerçek `bitdepth_max`**'ı içerir: 10-bit hücresi (0x3ff)
/// ile 12-bit hücresi (0xfff) ayrı sayılır. `pal_idx_finish` (kind 22) bit
/// derinliğinden bağımsızdır; her dosyada 0xff ile görünür, bu yüzden tek anahtar
/// altında dosya sayısıyla çarpılır ve ayrı raporlanır.
fn expected_cells(profile: &str, bpcs: &[u8]) -> BTreeMap<CellKey, usize> {
    let mut expected = BTreeMap::new();
    let calls_for = |kind: u8, w: usize, h: usize| -> usize {
        if profile == "large" {
            1000
        } else if (6..=8).contains(&kind) && w == 8 && h == 8 {
            108
        } else {
            2
        }
    };
    for &bpc in bpcs {
        let bdmax = expected_bitdepth_max(bpc);
        let add =
            |map: &mut BTreeMap<CellKey, usize>, kind: u8, aux: u8, cells: &[(usize, usize)]| {
                for &(w, h) in cells {
                    assert!(
                        map.insert((kind, aux, w, h, bdmax), calls_for(kind, w, h))
                            .is_none(),
                        "hücre tekrarı"
                    );
                }
            };
        for mode in 0..14u8 {
            add(&mut expected, mode, 0, &cells_for_mode(mode));
        }
        for layout in 0..3u8 {
            add(
                &mut expected,
                14 + layout,
                layout,
                &cells_for_cfl_ac(layout),
            );
        }
        for mode in 0..4u8 {
            add(&mut expected, 17 + mode, mode, &cells_for_cfl_pred());
        }
        add(&mut expected, 21, 0, &cells_for_pal());
    }
    for &(w, h) in &cells_for_pal_idx() {
        assert!(
            expected
                .insert((22, 0, w, h, 0xff), calls_for(22, w, h) * bpcs.len())
                .is_none(),
            "pal_idx hücre tekrarı"
        );
    }
    expected
}

// ------------------------------------------------------------- çalışma ---

#[derive(Default)]
struct CellStat {
    calls: usize,
    mismatches: usize,
    combos: BTreeSet<(usize, bool, bool)>,
}

struct RunStats {
    cells: BTreeMap<CellKey, CellStat>,
    pixels: u64,
    ac_cells: u64,
    records: u64,
    /// Bu koşuda işlenen bit derinlikleri (varsayılan 8, 10, 12).
    bpcs: Vec<u8>,
}

impl RunStats {
    fn new(bpcs: Vec<u8>) -> Self {
        Self {
            cells: BTreeMap::new(),
            pixels: 0,
            ac_cells: 0,
            records: 0,
            bpcs,
        }
    }

    fn entry(&mut self, key: CellKey) -> &mut CellStat {
        self.cells.entry(key).or_default()
    }
}

/// Aynı LCG (C oracle'ın `fill_dst`ıyla birebir): hedef tamponun başlangıç
/// pikselleri. Çıktı tüm hücrelerde tüm pikselleri yazdığı için değerler
/// karşılaştırmaya girmez; yine de iki taraf aynı başlangıç durumundan başlar.
fn fill_initial_dst(
    dst: &mut [u8],
    stride: usize,
    w: usize,
    h: usize,
    bpc: u8,
    bitdepth_max: u16,
    record_index: u64,
) {
    let bytes = if bpc == 8 { 1 } else { 2 };
    dst.fill(0);
    let mut state =
        0x9E37_79B9_7F4A_7C15u64.wrapping_mul(record_index + 1) ^ (u64::from(bpc) << 51);
    for y in 0..h {
        for x in 0..w {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let value = ((state >> 33) as u16) & bitdepth_max;
            let offset = y * stride + x * bytes;
            if bytes == 1 {
                dst[offset] = value as u8;
            } else {
                dst[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
            }
        }
    }
}

fn edge_bytes(values: &[u16], back: usize, bpc: u8) -> (Vec<u8>, usize) {
    let bytes = if bpc == 8 { 1 } else { 2 };
    let mut buffer = vec![0u8; values.len() * bytes];
    for (index, value) in values.iter().enumerate() {
        if bytes == 1 {
            buffer[index] = *value as u8;
        } else {
            buffer[index * 2..index * 2 + 2].copy_from_slice(&value.to_le_bytes());
        }
    }
    (buffer, back * bytes)
}

/// `IPRED_BPC=10` / `IPRED_BPC=10,12` yalnız seçilen bit derinliklerini koşar
/// (mutasyon kanıtında 10-bit ve 12-bit koşularını **ayrı ayrı** göstermek
/// için). Değişken yoksa üçü de koşulur — varsayılan davranış değişmez.
fn selected_bpcs() -> Vec<u8> {
    let Ok(raw) = std::env::var("IPRED_BPC") else {
        return vec![8, 10, 12];
    };
    if raw.trim().is_empty() {
        return vec![8, 10, 12];
    }
    let bpcs: Vec<u8> = raw
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<u8>()
                .unwrap_or_else(|_| panic!("IPRED_BPC geçersiz: {raw}"))
        })
        .collect();
    for &bpc in &bpcs {
        assert!(
            matches!(bpc, 8 | 10 | 12),
            "IPRED_BPC yalnız 8/10/12 olabilir: {bpc}"
        );
    }
    bpcs
}

fn run_set(profile: &str, input_dir: &Path, rust_out_dir: &Path) -> RunStats {
    fs::create_dir_all(rust_out_dir).expect("çıktı dizini");
    let bpcs = selected_bpcs();
    let mut stats = RunStats::new(bpcs.clone());
    for bpc in bpcs {
        let in_path = input_dir.join(format!("ipred-{profile}-{bpc}-input.bin"));
        let exp_path = input_dir.join(format!("ipred-{profile}-{bpc}-expect.bin"));
        let rust_path = rust_out_dir.join(format!("ipred-{profile}-{bpc}-rust.bin"));
        let (input, expect) = match (File::open(&in_path), File::open(&exp_path)) {
            (Ok(input), Ok(expect)) => (input, expect),
            _ => panic!(
                "fixture yok: {} — üretmek için: bash scripts/oracle/run_ipred_oracle.sh{}",
                in_path.display(),
                if profile == "large" { " --full" } else { "" }
            ),
        };
        let mut input = BufReader::new(input);
        let mut expect = BufReader::new(expect);
        let input_header = read_header(&mut input, b"IPRI");
        let expect_header = read_header(&mut expect, b"IPRE");
        assert_eq!(input_header, expect_header, "başlıklar farklı");
        assert_eq!(input_header.bpc, bpc);
        let header = input_header;
        let mut rust_file = BufWriter::new(File::create(&rust_path).expect("rust çıktısı"));
        write_header(&mut rust_file, &header);

        let mut cell_counters: BTreeMap<CellKey, u64> = BTreeMap::new();
        let mut first_mismatch: Option<(u64, Record, usize, i32, i32)> = None;
        for record_index in 0..u64::from(header.count) {
            let record = read_record(&mut input);
            let expect_record = read_record(&mut expect);
            assert!((record.kind as usize) < KIND_COUNT, "geçersiz kind");
            assert_eq!(record, expect_record, "kayıt başlıkları farklı");
            assert!(
                record.payload_len <= MAX_RECORD_PAYLOAD,
                "kayıt yükü çok büyük: {}",
                record.payload_len
            );

            // Rust tarafı da aynı üretimi doğrular: 16bpc'de bitdepth_max
            // koşunun bit derinliğine sabittir (10-bit 0x3ff, 12-bit 0xfff);
            // pal_idx_finish bit derinliğinden bağımsız olarak 0xff'tir.
            let expected_bdmax = if record.kind == 22 {
                0xff
            } else {
                expected_bitdepth_max(bpc)
            };
            assert_eq!(
                record.bitdepth_max, expected_bdmax,
                "bitdepth_max üretimi: bpc={bpc} kind={} beklenen 0x{expected_bdmax:x} gelen 0x{:x}",
                record.kind, record.bitdepth_max
            );

            let key = if record.kind == 22 {
                (record.kind, 0, record.bw, record.bh, record.bitdepth_max)
            } else {
                (
                    record.kind,
                    record.aux,
                    record.w,
                    record.h,
                    record.bitdepth_max,
                )
            };
            let counter = cell_counters.entry(key).or_insert(0);
            let per_cell_index = *counter;
            *counter += 1;

            let bytes = if bpc == 8 { 1 } else { 2 };
            let bd = bit_depth(bpc);
            let stride = (record.w + 8) * bytes;
            let mut got: Vec<u32> = Vec::new();
            let mut ok = true;
            let mut output_bytes: Vec<u8> = Vec::new();
            match record.kind {
                0..=13 => {
                    let mode =
                        IntraPredMode::from_index(usize::from(record.kind)).expect("intra modu");
                    let count = record.back + record.front + 1;
                    assert_eq!(record.payload_len, count * 2);
                    let mut values = vec![0u16; count];
                    for value in &mut values {
                        *value = read_u16(&mut input);
                    }
                    let (edge_buffer, edge_offset) = edge_bytes(&values, record.back, bpc);
                    let mut dst = vec![0u8; (record.h - 1) * stride + record.w * bytes];
                    fill_initial_dst(
                        &mut dst,
                        stride,
                        record.w,
                        record.h,
                        bpc,
                        record.bitdepth_max,
                        per_cell_index,
                    );
                    let edge = Edge::new(&edge_buffer, edge_offset);
                    intra_pred(
                        mode,
                        &mut dst,
                        stride,
                        edge,
                        record.w,
                        record.h,
                        record.angle,
                        record.max_width,
                        record.max_height,
                        bd,
                        record.bitdepth_max,
                    );
                    for y in 0..record.h {
                        for x in 0..record.w {
                            let offset = y * stride + x * bytes;
                            let value = if bytes == 1 {
                                u32::from(dst[offset])
                            } else {
                                u32::from(u16::from_le_bytes([dst[offset], dst[offset + 1]]))
                            };
                            got.push(value);
                            output_bytes.extend_from_slice(&(value as u16).to_le_bytes());
                        }
                    }
                    let base = match mode {
                        IntraPredMode::Z1 => 0,
                        IntraPredMode::Z2 => 90,
                        IntraPredMode::Z3 => 180,
                        _ => 0,
                    };
                    if mode.is_directional() {
                        let stat = stats.entry(key);
                        stat.combos.insert((
                            (record.angle & 511) - base,
                            (record.angle >> 9) & 1 == 1,
                            record.angle >> 10 != 0,
                        ));
                    }
                    stats.pixels += (record.w * record.h) as u64;
                }
                14..=16 => {
                    let layout = record.kind - 14;
                    let (ss_hor, ss_ver) = match layout {
                        0 => (true, true),
                        1 => (true, false),
                        _ => (false, false),
                    };
                    let luma_w = record.w << usize::from(ss_hor);
                    let luma_h = record.h << usize::from(ss_ver);
                    assert_eq!(record.payload_len, luma_w * luma_h * 2);
                    let luma_stride = (luma_w + 4) * bytes;
                    let mut luma = vec![0u8; luma_stride * luma_h];
                    for y in 0..luma_h {
                        for x in 0..luma_w {
                            let value = read_u16(&mut input);
                            let offset = y * luma_stride + x * bytes;
                            if bytes == 1 {
                                luma[offset] = value as u8;
                            } else {
                                luma[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
                            }
                        }
                    }
                    let mut ac = vec![0i16; record.w * record.h];
                    match layout {
                        0 => cfl_ac_420(
                            &mut ac,
                            &luma,
                            luma_stride,
                            record.w_pad,
                            record.h_pad,
                            record.w,
                            record.h,
                            bd,
                        ),
                        1 => cfl_ac_422(
                            &mut ac,
                            &luma,
                            luma_stride,
                            record.w_pad,
                            record.h_pad,
                            record.w,
                            record.h,
                            bd,
                        ),
                        _ => cfl_ac_444(
                            &mut ac,
                            &luma,
                            luma_stride,
                            record.w_pad,
                            record.h_pad,
                            record.w,
                            record.h,
                            bd,
                        ),
                    }
                    for value in &ac {
                        got.push(u32::from(*value as u16));
                        output_bytes.extend_from_slice(&value.to_le_bytes());
                    }
                    stats.ac_cells += (record.w * record.h) as u64;
                }
                17..=20 => {
                    let mode = match record.kind - 17 {
                        0 => CflPredMode::Dc,
                        1 => CflPredMode::LeftDc,
                        2 => CflPredMode::TopDc,
                        _ => CflPredMode::Dc128,
                    };
                    let count = record.back + record.front + 1;
                    assert_eq!(record.payload_len, count * 2 + record.w * record.h * 2);
                    let mut values = vec![0u16; count];
                    for value in &mut values {
                        *value = read_u16(&mut input);
                    }
                    let mut ac = vec![0i16; record.w * record.h];
                    for value in &mut ac {
                        *value = read_i16(&mut input);
                    }
                    let (edge_buffer, edge_offset) = edge_bytes(&values, record.back, bpc);
                    let mut dst = vec![0u8; (record.h - 1) * stride + record.w * bytes];
                    fill_initial_dst(
                        &mut dst,
                        stride,
                        record.w,
                        record.h,
                        bpc,
                        record.bitdepth_max,
                        per_cell_index,
                    );
                    let edge = Edge::new(&edge_buffer, edge_offset);
                    match mode {
                        CflPredMode::Dc => cfl_pred_dc(
                            &mut dst,
                            stride,
                            edge,
                            record.w,
                            record.h,
                            &ac,
                            i32::from(record.alpha),
                            bd,
                            record.bitdepth_max,
                        ),
                        CflPredMode::LeftDc => cfl_pred_left_dc(
                            &mut dst,
                            stride,
                            edge,
                            record.w,
                            record.h,
                            &ac,
                            i32::from(record.alpha),
                            bd,
                            record.bitdepth_max,
                        ),
                        CflPredMode::TopDc => cfl_pred_top_dc(
                            &mut dst,
                            stride,
                            edge,
                            record.w,
                            record.h,
                            &ac,
                            i32::from(record.alpha),
                            bd,
                            record.bitdepth_max,
                        ),
                        CflPredMode::Dc128 => cfl_pred_dc_128(
                            &mut dst,
                            stride,
                            edge,
                            record.w,
                            record.h,
                            &ac,
                            i32::from(record.alpha),
                            bd,
                            record.bitdepth_max,
                        ),
                    }
                    for y in 0..record.h {
                        for x in 0..record.w {
                            let offset = y * stride + x * bytes;
                            let value = if bytes == 1 {
                                u32::from(dst[offset])
                            } else {
                                u32::from(u16::from_le_bytes([dst[offset], dst[offset + 1]]))
                            };
                            got.push(value);
                            output_bytes.extend_from_slice(&(value as u16).to_le_bytes());
                        }
                    }
                    stats.pixels += (record.w * record.h) as u64;
                }
                21 => {
                    assert_eq!(record.payload_len, 16 + record.w * record.h / 2);
                    let mut pal = vec![0u8; 16];
                    for slot in 0..8 {
                        let value = read_u16(&mut input);
                        if bytes == 1 {
                            pal[slot] = value as u8;
                        } else {
                            pal[slot * 2..slot * 2 + 2].copy_from_slice(&value.to_le_bytes());
                        }
                    }
                    let mut idx = vec![0u8; record.w * record.h / 2];
                    input.read_exact(&mut idx).expect("indeks");
                    let mut dst = vec![0u8; (record.h - 1) * stride + record.w * bytes];
                    fill_initial_dst(
                        &mut dst,
                        stride,
                        record.w,
                        record.h,
                        bpc,
                        record.bitdepth_max,
                        per_cell_index,
                    );
                    pal_pred(&mut dst, stride, &pal, &idx, record.w, record.h, bd);
                    assert!(record.pal_sz >= 2 && record.pal_sz <= 8);
                    for y in 0..record.h {
                        for x in 0..record.w {
                            let offset = y * stride + x * bytes;
                            let value = if bytes == 1 {
                                u32::from(dst[offset])
                            } else {
                                u32::from(u16::from_le_bytes([dst[offset], dst[offset + 1]]))
                            };
                            got.push(value);
                            output_bytes.extend_from_slice(&(value as u16).to_le_bytes());
                        }
                    }
                    stats.pixels += (record.w * record.h) as u64;
                }
                22 => {
                    assert_eq!(record.payload_len, record.bw * record.bh);
                    let mut src = vec![0u8; record.bw * record.bh];
                    input.read_exact(&mut src).expect("src");
                    let mut dst = vec![0u8; (record.bw / 2) * record.bh];
                    pal_idx_finish(&mut dst, &src, record.bw, record.bh, record.w, record.h);
                    output_bytes.extend_from_slice(&dst);
                    stats.ac_cells += dst.len() as u64;
                }
                _ => panic!("bilinmeyen kind: {}", record.kind),
            }
            write_record(&mut rust_file, &record);
            rust_file.write_all(&output_bytes).expect("yazma");

            // Beklenen dosyayla karşılaştır.
            if record.kind == 22 {
                let count = (record.bw / 2) * record.bh;
                for p in 0..count {
                    let want = read_u8(&mut expect);
                    let have = output_bytes[p];
                    if have != want {
                        ok = false;
                        if first_mismatch.is_none() {
                            first_mismatch = Some((
                                record_index,
                                record.clone(),
                                p,
                                i32::from(have),
                                i32::from(want),
                            ));
                        }
                    }
                }
            } else if record.kind >= 14 && record.kind <= 16 {
                for p in 0..got.len() {
                    let want = read_u16(&mut expect);
                    let have = got[p] as u16;
                    if have != want {
                        ok = false;
                        if first_mismatch.is_none() {
                            first_mismatch = Some((
                                record_index,
                                record.clone(),
                                p,
                                i32::from(have as i16),
                                i32::from(want as i16),
                            ));
                        }
                    }
                }
            } else {
                for p in 0..got.len() {
                    let want = read_u16(&mut expect);
                    let have = got[p] as u16;
                    if have != want {
                        ok = false;
                        if first_mismatch.is_none() {
                            first_mismatch = Some((
                                record_index,
                                record.clone(),
                                p,
                                i32::from(have),
                                i32::from(want),
                            ));
                        }
                    }
                }
            }
            let stat = stats.entry(key);
            stat.calls += 1;
            if !ok {
                stat.mismatches += 1;
            }
            stats.records += 1;
        }
        // Fazladan bayt olmamalı.
        let mut extra = [0u8; 1];
        assert_eq!(input.read(&mut extra).expect("okuma"), 0, "girdi fazla");
        assert_eq!(expect.read(&mut extra).expect("okuma"), 0, "beklenen fazla");
        rust_file.flush().expect("flush");
        let got_size = fs::metadata(&rust_path).expect("rust").len();
        let want_size = fs::metadata(&exp_path).expect("beklenen").len();
        assert_eq!(got_size, want_size, "Rust ve C dosya boyutları farklı");

        if let Some((index, record, position, have, want)) = first_mismatch {
            panic!(
                "{profile} {bpc}-bit: ilk uyuşmazlık kayıt {index} \
                 (kind={} aux={} {}x{}) indeks {position}: rust={have} dav1d={want}",
                record.kind, record.aux, record.w, record.h
            );
        }
        println!(
            "[{profile} {bpc}-bit] kayıt {}/{} · piksel {} · ac/idx hücresi {}",
            header.count, header.count, stats.pixels, stats.ac_cells
        );
    }
    stats
}

fn assert_coverage(profile: &str, stats: &RunStats) {
    let expected = expected_cells(profile, &stats.bpcs);
    assert_eq!(
        stats.cells.len(),
        expected.len(),
        "hücre sayısı beklenenden farklı"
    );
    for (key, calls) in &expected {
        let stat = stats.cells.get(key).unwrap_or_else(|| {
            panic!(
                "hücre eksik: kind={} aux={} {}x{} bdmax=0x{:x}",
                key.0, key.1, key.2, key.3, key.4
            )
        });
        // Anahtar gerçek bitdepth_max'ı içerir: 10-bit ve 12-bit hücreleri
        // ayrı sayılır; hücre başına çağrı = o bit derinliğinin çağrısı.
        assert_eq!(stat.calls, *calls, "hücre çağrı sayısı: {key:?}");
        assert_eq!(stat.mismatches, 0, "hücre uyuşmazlığı: {key:?}");
        if (6..=8).contains(&key.0) {
            let required = if profile == "large" || (key.2 == 8 && key.3 == 8) {
                108
            } else {
                0
            };
            if required > 0 {
                assert_eq!(
                    stat.combos.len(),
                    108,
                    "açı/bayrak kapsamı eksik: kind={} {}x{} → {}",
                    key.0,
                    key.2,
                    key.3,
                    stat.combos.len()
                );
            }
        }
    }
    // Kabul ölçütü 1'in Rust ayağı: gerçek `bitdepth_max` başına hücre sayısı
    // ve hücre başına çağrı. Büyük kümede **bütün** hücreler ≥ 1000 olmalı.
    for &bpc in &stats.bpcs {
        let bdmax = expected_bitdepth_max(bpc);
        let counts: Vec<usize> = stats
            .cells
            .iter()
            .filter(|(key, _)| key.0 != 22 && key.4 == bdmax)
            .map(|(_, stat)| stat.calls)
            .collect();
        let expected_per_bd = expected
            .keys()
            .filter(|key| key.0 != 22 && key.4 == bdmax)
            .count();
        assert_eq!(
            counts.len(),
            expected_per_bd,
            "{profile} {bpc}-bit: hücre sayısı"
        );
        let min = *counts.iter().min().expect("hücre yok");
        let max = *counts.iter().max().expect("hücre yok");
        if profile == "large" {
            assert!(
                min >= 1000,
                "{profile} {bpc}-bit: hücre başına en küçük çağrı {min} < 1000"
            );
        }
        println!(
            "[{profile} {bpc}-bit] hücre {expected_per_bd} · en küçük {min} · en büyük {max} (bitdepth_max 0x{bdmax:x})"
        );
    }
    let pal_cells = stats.cells.keys().filter(|key| key.0 == 22).count();
    let pal_calls: usize = stats
        .cells
        .iter()
        .filter(|(key, _)| key.0 == 22)
        .map(|(_, stat)| stat.calls)
        .sum();
    println!(
        "[{profile}] pal_idx_finish: hücre {pal_cells} · çağrı {pal_calls} · bit derinliğinden bağımsız (ayrı satır)"
    );
}

#[test]
fn kucuk_kume_bit_exact() {
    let stats = run_set("small", &small_dir(), &cache_dir());
    assert_coverage("small", &stats);
    // Özet tablo: işlev ailesi başına hücre/çağrı/kapsam.
    let mut families: BTreeMap<u8, (usize, usize, usize)> = BTreeMap::new();
    for (key, stat) in &stats.cells {
        let family = match key.0 {
            0..=13 => key.0,
            14..=16 => 14,
            17..=20 => 17,
            21 => 21,
            _ => 22,
        };
        let entry = families.entry(family).or_insert((0, 0, 0));
        entry.0 += 1;
        entry.1 += stat.calls;
        entry.2 += stat.combos.len();
    }
    println!(
        "[small] TOPLAM kayıt {} · hücre {}",
        stats.records,
        stats.cells.len()
    );
    for (family, (cells, calls, combos)) in families {
        println!(
            "[small] aile {family:2}: hücre {cells:3} · çağrı {calls:5} · açı/bayrak birleşimi {combos}"
        );
    }
}

#[test]
#[ignore = "büyük küme: bash scripts/oracle/run_ipred_oracle.sh --full"]
fn buyuk_kume_bit_exact() {
    let stats = run_set("large", &cache_dir(), &cache_dir());
    assert_coverage("large", &stats);
    println!(
        "[large] TOPLAM kayıt {} · hücre {} · piksel {} · ac/idx hücresi {}",
        stats.records,
        stats.cells.len(),
        stats.pixels,
        stats.ac_cells
    );
}

/// Kabul ölçütü 2: üç tablo dav1d derlemesinden dökülen baytlarla **bayt-eşit**.
///
/// Fixture `bash scripts/oracle/run_ipred_oracle.sh --update` ile üretilir;
/// dav1d @ 99f3354 `src/tables.c` (ARCH_X86=1, HAVE_ASM=0) dökümüdür.
#[test]
fn tablolar_dav1d_ile_bayt_esit() {
    use av1oxide::dsp::ipred_tables::{DR_INTRA_DERIVATIVE, FILTER_INTRA_TAPS, SM_WEIGHTS};

    let path = small_dir().join("ipred-tables.bin");
    let bytes =
        fs::read(&path).unwrap_or_else(|_| panic!("tablo fixture'ı yok: {}", path.display()));
    assert_eq!(bytes.len(), 128 + 88 + 320, "tablo dökümü 536 bayt olmalı");
    assert_eq!(&bytes[..128], &SM_WEIGHTS, "sm_weights farklı");
    let mut derivative = Vec::with_capacity(88);
    for value in DR_INTRA_DERIVATIVE {
        derivative.extend_from_slice(&value.to_le_bytes());
    }
    assert_eq!(
        &bytes[128..216],
        derivative.as_slice(),
        "dr_intra_derivative farklı"
    );
    let mut taps = Vec::with_capacity(320);
    for table in FILTER_INTRA_TAPS {
        for value in table {
            taps.push(value as u8);
        }
    }
    assert_eq!(
        &bytes[216..536],
        taps.as_slice(),
        "filter_intra_taps farklı"
    );
}
