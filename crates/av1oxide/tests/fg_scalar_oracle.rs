//! S2d.1 film grain çekirdek oracle'ı (Rust tarafı).
//!
//! `scripts/oracle/run_fg_oracle.sh` dav1d C (`filmgrain_tmpl.c` DSP tablosu +
//! `fg_apply_tmpl.c` `generate_scaling`, asm=OFF) ile durum dosyası ve çağrı
//! başına SHA-256 listesi üretir (`fg-<küme>-cases.bin`, `-expect.bin`). Bu
//! test aynı durumlardan aynı girdiyi yeniden üretir (`fg_oracle.c`
//! `sm64`/`val` ile birebir), [`av1oxide::dsp::fg::FgDsp::scalar_for`] ve
//! [`av1oxide::decode::filmgrain::generate_scaling`] ile koşar ve
//! `fg-<küme>-rust.bin` yazar; betik iki listeyi karşılaştırır. Sözleşme dışı
//! bölgeler C'dekinden **farklı** bir değerle (0x3C) doldurulur: kenar ötesi
//! okuma SHA farkı ya da panik üretir; hedefin bölge dışı baytları denetlenir.
//!
//! Differential betik kipindedir (`#[ignore]`; betik `--include-ignored` ve
//! `FG_ORACLE_OUT` / `FG_ORACLE_SET` ile koşturur). Tablo testi ortam
//! değişkeni yoksa varsayılan `.cache/fg-oracle` dökümünü okur; döküm yoksa
//! ön koşul eksiktir (`on_kosul::eksik`, P29) — sessizce geçmez.

#![forbid(unsafe_code)]

mod common;

use av1oxide::bitstream::frame_header::FilmGrainData;
use av1oxide::decode::filmgrain::generate_scaling;
use av1oxide::dsp::fg::{
    EMPTY_LUT, FgBitDepth, FgDsp, GRAIN_HEIGHT, GRAIN_WIDTH, GrainLut, SUB_GRAIN_HEIGHT,
    SUB_GRAIN_WIDTH,
};
use av1oxide::dsp::fg_tables::GAUSSIAN_SEQUENCE;
use common::on_kosul::{self, Neden};
use std::fmt::Write as _;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4) — test yardımcısı; harici crate yok.
// ---------------------------------------------------------------------------

struct Sha256 {
    state: [u32; 8],
    len: u64,
    buf: [u8; 64],
    fill: usize,
}

const SHA_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            len: 0,
            buf: [0u8; 64],
            fill: 0,
        }
    }

    fn block(&mut self, p: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, word) in w.iter_mut().take(16).enumerate() {
            *word = u32::from_be_bytes([p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, v) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h].iter()) {
            *slot = slot.wrapping_add(*v);
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.len += data.len() as u64;
        if self.fill > 0 {
            let take = (64 - self.fill).min(data.len());
            self.buf[self.fill..self.fill + take].copy_from_slice(&data[..take]);
            self.fill += take;
            data = &data[take..];
            if self.fill == 64 {
                let b = self.buf;
                self.block(&b);
                self.fill = 0;
            }
        }
        while data.len() >= 64 {
            let mut b = [0u8; 64];
            b.copy_from_slice(&data[..64]);
            self.block(&b);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.fill = data.len();
        }
    }

    fn finish(mut self) -> [u8; 32] {
        let bitlen = self.len * 8;
        self.update(&[0x80]);
        while self.fill != 56 {
            self.update(&[0]);
        }
        self.update(&bitlen.to_be_bytes());
        let mut out = [0u8; 32];
        for i in 0..8 {
            out[4 * i..4 * i + 4].copy_from_slice(&self.state[i].to_be_bytes());
        }
        out
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn sha(data: &[u8]) -> [u8; 32] {
    let mut s = Sha256::new();
    s.update(data);
    s.finish()
}

// ---------------------------------------------------------------------------
// Girdi üreteci — `scripts/oracle/fg_oracle.c` `sm64`/`val` ile birebir.
// ---------------------------------------------------------------------------

fn sm64(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[allow(clippy::too_many_arguments)]
fn val(pat: i32, x: i32, y: i32, r: u64, lo: i32, hi: i32, bd: i32) -> i32 {
    let range = hi - lo + 1;
    match pat {
        0 => lo,
        1 => hi,
        2 => {
            if (x ^ y) & 1 != 0 {
                hi
            } else {
                lo
            }
        }
        3 => lo + (r % range as u64) as i32,
        4 => lo + (x * 37 + y * 11) % range,
        5 => {
            if r & 1 != 0 {
                hi
            } else {
                lo
            }
        }
        _ => {
            const BASE: [i32; 8] = [15, 16, 17, 234, 235, 236, 240, 241];
            let v = (BASE[(r % 8) as usize] << (bd - 8)) + ((r >> 8) % 3) as i32 - 1;
            v.clamp(lo, hi)
        }
    }
}

const REC_SIZE: usize = 15 * 4 + 207;
const KIND_NAMES: [&str; 5] = ["grain_y", "grain_uv", "fgy", "fguv", "scaling"];
const LAYOUTS: [&str; 3] = ["420", "422", "444"];

struct Case {
    kind: usize,
    bd: i32,
    layout: usize,
    uv: usize,
    pw: usize,
    bh: usize,
    row_num: usize,
    is_id: bool,
    stride_px: usize,
    luma_stride_px: usize,
    pix_pat: i32,
    lut_pat: i32,
    scal_pat: i32,
    seed: u64,
    d: FilmGrainData,
}

struct Rd<'a> {
    b: &'a [u8],
    at: usize,
}

impl Rd<'_> {
    fn i32(&mut self) -> i32 {
        let v = i32::from_le_bytes(self.b[self.at..self.at + 4].try_into().expect("4"));
        self.at += 4;
        v
    }
    fn u8(&mut self) -> u8 {
        let v = self.b[self.at];
        self.at += 1;
        v
    }
}

/// `fg_oracle.c` `put_case` sırası.
fn parse_case(rec: &[u8]) -> Case {
    let mut r = Rd { b: rec, at: 0 };
    let h: Vec<i32> = (0..13).map(|_| r.i32()).collect();
    let lo = r.i32() as u32;
    let hi = r.i32() as u32;
    let mut d = FilmGrainData {
        seed: r.i32() as u32,
        num_y_points: r.i32(),
        ..FilmGrainData::default()
    };
    for p in &mut d.y_points {
        p[0] = r.u8();
        p[1] = r.u8();
    }
    d.chroma_scaling_from_luma = r.u8() != 0;
    d.num_uv_points = [r.i32(), r.i32()];
    for pl in &mut d.uv_points {
        for p in pl.iter_mut() {
            p[0] = r.u8();
            p[1] = r.u8();
        }
    }
    d.scaling_shift = r.i32();
    d.ar_coeff_lag = r.i32();
    for c in &mut d.ar_coeffs_y {
        *c = r.u8() as i8;
    }
    for pl in &mut d.ar_coeffs_uv {
        for c in pl.iter_mut() {
            *c = r.u8() as i8;
        }
    }
    d.ar_coeff_shift = r.i32() as u64;
    d.grain_scale_shift = r.i32();
    d.uv_mult = [r.i32(), r.i32()];
    d.uv_luma_mult = [r.i32(), r.i32()];
    d.uv_offset = [r.i32(), r.i32()];
    d.overlap_flag = r.u8() != 0;
    d.clip_to_restricted_range = r.u8() != 0;
    assert_eq!(r.at, REC_SIZE, "kayıt boyu");
    Case {
        kind: h[0] as usize,
        bd: h[1],
        layout: h[2] as usize,
        uv: h[3] as usize,
        pw: h[4] as usize,
        bh: h[5] as usize,
        row_num: h[6] as usize,
        is_id: h[7] != 0,
        stride_px: h[8] as usize,
        luma_stride_px: h[9] as usize,
        pix_pat: h[10],
        lut_pat: h[11],
        scal_pat: h[12],
        seed: u64::from(lo) | (u64::from(hi) << 32),
        d,
    }
}

fn store_px(buf: &mut [u8], idx: usize, bpp: usize, v: i32) {
    if bpp == 1 {
        buf[idx] = v as u8;
    } else {
        buf[2 * idx..2 * idx + 2].copy_from_slice(&(v as u16).to_le_bytes());
    }
}

fn lut_bytes(h: &mut Sha256, lut: &GrainLut, w: usize, hgt: usize) {
    for row in lut.iter().take(hgt) {
        for v in row.iter().take(w) {
            h.update(&v.to_le_bytes());
        }
    }
}

/// Tablo dökümünün dizini: `FG_ORACLE_OUT` ya da varsayılan `.cache/fg-oracle`.
fn tablo_dizini() -> PathBuf {
    std::env::var_os("FG_ORACLE_OUT").map_or_else(
        || {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../.cache/fg-oracle"
            ))
        },
        PathBuf::from,
    )
}

fn oracle_dir() -> Option<(PathBuf, String)> {
    let dir = std::env::var_os("FG_ORACLE_OUT")?;
    let set = std::env::var("FG_ORACLE_SET").ok()?;
    Some((PathBuf::from(dir), set))
}

#[test]
fn sha256_sabit_vektor() {
    assert_eq!(
        hex(&sha(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

/// `GAUSSIAN_SEQUENCE` dav1d `dav1d_gaussian_sequence` dökümüyle bayt bayt aynı.
#[test]
fn gauss_tablosu_dav1d_ile_bayt_esit() {
    assert_eq!(GAUSSIAN_SEQUENCE.len(), 2048);
    let yol = tablo_dizini().join("fg-tables.bin");
    let Ok(c) = std::fs::read(&yol) else {
        on_kosul::eksik(
            "fg_scalar_oracle::gauss_tablosu_dav1d_ile_bayt_esit",
            Neden::OracleVerisiYok,
            "dav1d tablo dökümü yok; üretmek için: bash scripts/oracle/run_fg_oracle.sh",
        );
        return;
    };
    let rust: Vec<u8> = GAUSSIAN_SEQUENCE
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(c, rust, "dav1d_gaussian_sequence farklı");
    println!("[fg tablo] 2048 × i16 = {} bayt dav1d = Rust", rust.len());
}

#[test]
#[ignore = "betik kipi: scripts/oracle/run_fg_oracle.sh koşturur (FG_ORACLE_OUT/FG_ORACLE_SET)"]
fn fg_oracle_differential() {
    let Some((dir, set)) = oracle_dir() else {
        on_kosul::eksik(
            "fg_scalar_oracle::fg_oracle_differential",
            Neden::OracleVerisiYok,
            "FG_ORACLE_OUT/FG_ORACLE_SET yok (betik kipi)",
        );
        return;
    };
    let cases = std::fs::read(dir.join(format!("fg-{set}-cases.bin"))).expect("cases");
    assert_eq!(cases.len() % REC_SIZE, 0);
    let mut out = Vec::with_capacity(cases.len() / REC_SIZE * 32);
    let mut cell: Vec<Vec<Vec<Sha256>>> = (0..5)
        .map(|_| {
            (0..3)
                .map(|_| (0..3).map(|_| Sha256::new()).collect())
                .collect()
        })
        .collect();
    let mut cell_n = [[[0u32; 3]; 3]; 5];
    let mut lut: Box<GrainLut> = Box::new(EMPTY_LUT);
    let mut lut2: Box<GrainLut> = Box::new(EMPTY_LUT);
    let mut calls = 0usize;
    // P39 / P41 / P42: 8/10/12 bpc AVX2, SSSE3 ve AVX512ICL gövdeleri aynı
    // durumlarda skaler (dolayısıyla dav1d C) özetiyle karşılaştırılır.
    let avx2 = avx2_tablolari();
    let ssse3 = ssse3_tablolari();
    let avx512 = avx512_tablolari();
    let mut avx2_calls = [0usize; 3];
    let mut ssse3_calls = [0usize; 3];
    let mut avx512_calls = [0usize; 3];
    for (ci, rec) in cases.chunks(REC_SIZE).enumerate() {
        let e = parse_case(rec);
        let bd = match e.bd {
            8 => FgBitDepth::B8,
            10 => FgBitDepth::B10,
            _ => FgBitDepth::B12,
        };
        let bpp = bd.bytes();
        let max = bd.bitdepth_max();
        let gctr = 128i32 << (e.bd - 8);
        // F3c: kaydın bit derinliği tablosu.
        let dsp = FgDsp::scalar_for(bd);
        let mut s = e.seed;
        let mut h = Sha256::new();
        let d = &e.d;
        match e.kind {
            0 | 1 => {
                *lut = EMPTY_LUT;
                (dsp.generate_grain_y)(&mut lut, d, bd);
                if e.kind == 0 {
                    lut_bytes(&mut h, &lut, GRAIN_WIDTH, GRAIN_HEIGHT);
                } else {
                    *lut2 = EMPTY_LUT;
                    (dsp.generate_grain_uv[e.layout])(&mut lut2, &lut, d, e.uv, bd);
                    let cw = if e.layout != 2 {
                        SUB_GRAIN_WIDTH
                    } else {
                        GRAIN_WIDTH
                    };
                    let ch = if e.layout == 0 {
                        SUB_GRAIN_HEIGHT
                    } else {
                        GRAIN_HEIGHT
                    };
                    lut_bytes(&mut h, &lut2, cw, ch);
                }
            }
            4 => {
                let mut scaling = vec![0x3Cu8; 4096];
                let num = d.num_y_points as usize;
                generate_scaling(bd, &d.y_points[..num], &mut scaling);
                let size = 1usize << e.bd;
                assert!(
                    scaling[size..].iter().all(|&b| b == 0x3C),
                    "generate_scaling bölge dışı yazdı (durum {ci})"
                );
                h.update(&scaling[..size]);
            }
            _ => {
                let (sx, sy) = if e.kind == 3 {
                    (usize::from(e.layout != 2), usize::from(e.layout == 0))
                } else {
                    (0, 0)
                };
                let stride = e.stride_px * bpp;
                let lstride = e.luma_stride_px * bpp;
                let luma_w = ((e.pw - 1) << sx) + sx + 1;
                let luma_rows = ((e.bh - 1) << sy) + 1;
                // Kaynak ve luma dilimleri tam son pikselde biter.
                let mut src = vec![0x3Cu8; (e.bh - 1) * stride + e.pw * bpp];
                let mut dst = vec![0x5Au8; stride * e.bh];
                let mut luma = vec![0x3Cu8; (luma_rows - 1) * lstride + luma_w * bpp];
                let mut scaling = vec![0x3Cu8; max as usize + 1];
                for (y, row) in lut.iter_mut().enumerate() {
                    for (x, v) in row.iter_mut().enumerate() {
                        *v = val(
                            e.lut_pat,
                            x as i32,
                            y as i32,
                            sm64(&mut s),
                            -gctr,
                            gctr - 1,
                            e.bd,
                        ) as i16;
                    }
                }
                for (i, v) in scaling.iter_mut().enumerate() {
                    *v = val(e.scal_pat, i as i32, 0, sm64(&mut s), 0, 255, 8) as u8;
                }
                for y in 0..e.bh {
                    for x in 0..e.pw {
                        let v = val(e.pix_pat, x as i32, y as i32, sm64(&mut s), 0, max, e.bd);
                        store_px(&mut src, y * e.stride_px + x, bpp, v);
                    }
                }
                for y in 0..luma_rows {
                    for x in 0..luma_w {
                        let v = val(
                            e.pix_pat,
                            x as i32 + 1,
                            y as i32,
                            sm64(&mut s),
                            0,
                            max,
                            e.bd,
                        );
                        store_px(&mut luma, y * e.luma_stride_px + x, bpp, v);
                    }
                }
                if e.kind == 2 {
                    (dsp.fgy_32x32xn)(
                        &mut dst, &src, stride, d, e.pw, &scaling, &lut, e.bh, e.row_num, bd,
                    );
                } else {
                    (dsp.fguv_32x32xn[e.layout])(
                        &mut dst, &src, stride, d, e.pw, &scaling, &lut, e.bh, e.row_num, &luma,
                        lstride, e.uv, e.is_id, bd,
                    );
                }
                for (i, &b) in dst.iter().enumerate() {
                    let inside = i / stride < e.bh && i % stride < e.pw * bpp;
                    assert!(
                        inside || b == 0x5A,
                        "{} bölge dışı yazdı (durum {ci})",
                        KIND_NAMES[e.kind]
                    );
                }
                for y in 0..e.bh {
                    h.update(&dst[y * stride..y * stride + e.pw * bpp]);
                }
            }
        }
        let dg = h.finish();
        let bi = match e.bd {
            8 => 0,
            10 => 1,
            _ => 2,
        };
        if let Some(a2) = &avx2 {
            let dg2 = simd_ozet(&a2[bi], &e, &lut, &mut s2_tohum(&e));
            assert_eq!(
                hex(&dg2),
                hex(&dg),
                "AVX2 ({} bpc) ≠ skaler: {} durum {ci}",
                e.bd,
                KIND_NAMES[e.kind]
            );
            avx2_calls[bi] += usize::from(e.kind != 4);
        }
        if let Some(s3) = &ssse3 {
            let dg3 = simd_ozet(&s3[bi], &e, &lut, &mut s2_tohum(&e));
            assert_eq!(
                hex(&dg3),
                hex(&dg),
                "SSSE3 ({} bpc) ≠ skaler: {} durum {ci}",
                e.bd,
                KIND_NAMES[e.kind]
            );
            ssse3_calls[bi] += usize::from(e.kind != 4);
        }
        if let (Some(z), 2 | 3) = (&avx512, e.kind) {
            let dgz = simd_ozet(&z[bi], &e, &lut, &mut s2_tohum(&e));
            assert_eq!(
                hex(&dgz),
                hex(&dg),
                "AVX512ICL ({} bpc) ≠ skaler: {} durum {ci}",
                e.bd,
                KIND_NAMES[e.kind]
            );
            avx512_calls[bi] += 1;
        }
        out.extend_from_slice(&dg);
        cell[e.kind][bi][e.layout].update(&dg);
        cell_n[e.kind][bi][e.layout] += 1;
        calls += 1;
    }
    std::fs::write(dir.join(format!("fg-{set}-rust.bin")), &out).expect("rust.bin");
    let mut txt = String::new();
    for (k, name) in KIND_NAMES.iter().enumerate() {
        for (bi, bdv) in [8u32, 10, 12].iter().enumerate() {
            for l in 0..3 {
                if cell_n[k][bi][l] == 0 {
                    continue;
                }
                let c = std::mem::replace(&mut cell[k][bi][l], Sha256::new());
                let lay = if k == 1 || k == 3 { LAYOUTS[l] } else { "-" };
                let _ = writeln!(
                    txt,
                    "{name} {bdv} {lay} {} {}",
                    cell_n[k][bi][l],
                    hex(&c.finish())
                );
            }
        }
    }
    std::fs::write(dir.join(format!("fg-{set}-cells-rust.txt")), txt).expect("cells");
    println!(
        "[fg {set}] {calls} çağrı, çıktı SHA-256 {}",
        hex(&sha(&out))
    );
    if avx2.is_some() {
        println!(
            "[fg {set}] AVX2 (8/10/12 bpc): {}/{}/{} çağrı, özet = skaler = dav1d C",
            avx2_calls[0], avx2_calls[1], avx2_calls[2]
        );
    }
    if ssse3.is_some() {
        println!(
            "[fg {set}] SSSE3 (8/10/12 bpc): {}/{}/{} çağrı, özet = skaler = dav1d C",
            ssse3_calls[0], ssse3_calls[1], ssse3_calls[2]
        );
    }
    if avx512.is_some() {
        println!(
            "[fg {set}] AVX512ICL (8/10/12 bpc): {}/{}/{} çağrı, özet = skaler = dav1d C",
            avx512_calls[0], avx512_calls[1], avx512_calls[2]
        );
    }
}

/// P41 / P42: 8, 10 ve 12 bpc AVX512ICL film grain yuvaları (`fgy_32x32xn` ve
/// `fguv_32x32xn[0..3]`; AVX512ICL yoksa ön koşul ilkesiyle `None`).
fn avx512_tablolari() -> Option<[FgDsp; 3]> {
    #[cfg(target_arch = "x86_64")]
    {
        use av1oxide::dsp::{DspDispatch, ISA_CANDIDATES, IsaEntry};
        let d = DspDispatch::select(av1oxide::cpu::CpuFlags::detect());
        if let Some(tk) = d.avx512icl_token() {
            let mut out = [
                FgDsp::scalar_for(FgBitDepth::B8),
                FgDsp::scalar_for(FgBitDepth::B10),
                FgDsp::scalar_for(FgBitDepth::B12),
            ];
            for (bi, bdv) in [8u8, 10, 12].into_iter().enumerate() {
                let mut n = 0;
                for c in ISA_CANDIDATES.iter().filter(|c| {
                    c.family == "filmgrain" && c.bit_depth == bdv && c.implementation == "avx512icl"
                }) {
                    match d.candidate_entry(c) {
                        Some(IsaEntry::Fgy(f)) => out[bi].fgy_32x32xn = f,
                        Some(IsaEntry::Fguv(i, f)) => out[bi].fguv_32x32xn[i] = f,
                        _ => continue,
                    }
                    n += 1;
                }
                // `SLOW_GATHER`'lı işlemcide 10/12 bpc `fgy`/`fguv` AVX512ICL adayı
                // yoktur: gövdeler yine sınansın diye doğrudan belirteçle alınır.
                if n < 4 {
                    assert_ne!(bdv, 8, "8 bpc AVX512ICL filmgrain 4 aday olmalı");
                    out[bi].fgy_32x32xn = av1oxide::dsp::x86::fg16_avx512::fgy_fn(tk);
                    out[bi].fguv_32x32xn =
                        [0, 1, 2].map(|i| av1oxide::dsp::x86::fg16_avx512::fguv_fn(tk, i));
                }
            }
            return Some(out);
        }
    }
    on_kosul::eksik(
        "fg_scalar_oracle::fg_oracle_differential[avx512icl]",
        Neden::isa(),
        "[P41/P42] 8/10/12 bpc AVX512ICL film grain gövdeleri AVX512ICL ister (skaler kısım koşar)",
    );
    None
}

/// P41 / P42: 8, 10 ve 12 bpc SSSE3 film grain yuvaları (`DspDispatch` adayları;
/// SSSE3 yoksa ön koşul ilkesiyle `None`).
fn ssse3_tablolari() -> Option<[FgDsp; 3]> {
    #[cfg(target_arch = "x86_64")]
    {
        use av1oxide::dsp::{DspDispatch, ISA_CANDIDATES, IsaEntry};
        let d = DspDispatch::select(av1oxide::cpu::CpuFlags::detect());
        if d.ssse3_token().is_some() {
            let mut out = [
                FgDsp::scalar_for(FgBitDepth::B8),
                FgDsp::scalar_for(FgBitDepth::B10),
                FgDsp::scalar_for(FgBitDepth::B12),
            ];
            for (bi, bdv) in [8u8, 10, 12].into_iter().enumerate() {
                let mut n = 0;
                for c in ISA_CANDIDATES.iter().filter(|c| {
                    c.family == "filmgrain" && c.bit_depth == bdv && c.implementation == "ssse3"
                }) {
                    match d.candidate_entry(c) {
                        Some(IsaEntry::FgGrainY(f)) => out[bi].generate_grain_y = f,
                        Some(IsaEntry::FgGrainUv(i, f)) => out[bi].generate_grain_uv[i] = f,
                        Some(IsaEntry::Fgy(f)) => out[bi].fgy_32x32xn = f,
                        Some(IsaEntry::Fguv(i, f)) => out[bi].fguv_32x32xn[i] = f,
                        _ => continue,
                    }
                    n += 1;
                }
                assert_eq!(n, 8, "SSSE3 filmgrain ({bdv} bpc) 8 aday olmalı");
            }
            return Some(out);
        }
    }
    on_kosul::eksik(
        "fg_scalar_oracle::fg_oracle_differential[ssse3]",
        Neden::isa(),
        "[P41/P42] 8/10/12 bpc SSSE3 film grain gövdeleri SSSE3 ister (skaler kısım koşar)",
    );
    None
}

/// P39 / P42: 8, 10 ve 12 bpc AVX2 film grain yuvaları (`DspDispatch` adayları;
/// AVX2 yoksa ön koşul ilkesiyle `None`).
fn avx2_tablolari() -> Option<[FgDsp; 3]> {
    #[cfg(target_arch = "x86_64")]
    {
        use av1oxide::dsp::{DspDispatch, ISA_CANDIDATES, IsaEntry};
        let d = DspDispatch::select(av1oxide::cpu::CpuFlags::detect());
        if let Some(tk) = d.avx2_token() {
            let mut out = [
                FgDsp::scalar_for(FgBitDepth::B8),
                FgDsp::scalar_for(FgBitDepth::B10),
                FgDsp::scalar_for(FgBitDepth::B12),
            ];
            for (bi, bdv) in [8u8, 10, 12].into_iter().enumerate() {
                let mut n = 0;
                for c in ISA_CANDIDATES.iter().filter(|c| {
                    c.family == "filmgrain" && c.bit_depth == bdv && c.implementation == "avx2"
                }) {
                    match d.candidate_entry(c) {
                        Some(IsaEntry::FgGrainY(f)) => out[bi].generate_grain_y = f,
                        Some(IsaEntry::FgGrainUv(i, f)) => out[bi].generate_grain_uv[i] = f,
                        Some(IsaEntry::Fgy(f)) => out[bi].fgy_32x32xn = f,
                        Some(IsaEntry::Fguv(i, f)) => out[bi].fguv_32x32xn[i] = f,
                        _ => continue,
                    }
                    n += 1;
                }
                // `SLOW_GATHER`'lı işlemcide `fgy`/`fguv` adayı yoktur: gövdeler
                // yine sınansın diye doğrudan belirteçle alınır.
                if n < 8 {
                    if bdv == 8 {
                        out[bi].fgy_32x32xn = av1oxide::dsp::x86::fg::fgy_fn(tk);
                        out[bi].fguv_32x32xn =
                            [0, 1, 2].map(|i| av1oxide::dsp::x86::fg::fguv_fn(tk, i));
                    } else {
                        out[bi].fgy_32x32xn = av1oxide::dsp::x86::fg16::fgy_fn(tk);
                        out[bi].fguv_32x32xn =
                            [0, 1, 2].map(|i| av1oxide::dsp::x86::fg16::fguv_fn(tk, i));
                    }
                }
            }
            return Some(out);
        }
    }
    on_kosul::eksik(
        "fg_scalar_oracle::fg_oracle_differential[avx2]",
        Neden::avx2(),
        "[P39/P42] 8/10/12 bpc AVX2 film grain gövdeleri AVX2 ister (skaler kısım koşar)",
    );
    None
}

/// Durumun girdi tohumu (fgy/fguv girdileri `e.seed`'den yeniden üretilir).
fn s2_tohum(e: &Case) -> u64 {
    e.seed
}

/// P39 / P41 / P42: durumu SIMD tablosuyla koşar ve skalerle aynı biçimde özetler
/// (8 bpc AVX2 toplaması için `SCALING_SIZE_MAX`, 10/12 bpc için `max + 1`;
/// `dst` bölge dışı yazma denetimi aynı).
fn simd_ozet(a: &FgDsp, e: &Case, lut_y: &GrainLut, s: &mut u64) -> [u8; 32] {
    let bd = match e.bd {
        8 => FgBitDepth::B8,
        10 => FgBitDepth::B10,
        _ => FgBitDepth::B12,
    };
    let bpp = bd.bytes();
    let max = bd.bitdepth_max();
    let d = &e.d;
    let mut h = Sha256::new();
    match e.kind {
        0 | 1 => {
            let mut ly: Box<GrainLut> = Box::new(EMPTY_LUT);
            (a.generate_grain_y)(&mut ly, d, bd);
            if e.kind == 0 {
                lut_bytes(&mut h, &ly, GRAIN_WIDTH, GRAIN_HEIGHT);
            } else {
                let mut l2: Box<GrainLut> = Box::new(EMPTY_LUT);
                (a.generate_grain_uv[e.layout])(&mut l2, &ly, d, e.uv, bd);
                let cw = if e.layout != 2 {
                    SUB_GRAIN_WIDTH
                } else {
                    GRAIN_WIDTH
                };
                let ch = if e.layout == 0 {
                    SUB_GRAIN_HEIGHT
                } else {
                    GRAIN_HEIGHT
                };
                lut_bytes(&mut h, &l2, cw, ch);
            }
        }
        4 => {
            let mut scaling = vec![0x3Cu8; 4096];
            let num = d.num_y_points as usize;
            generate_scaling(bd, &d.y_points[..num], &mut scaling);
            h.update(&scaling[..1usize << e.bd]);
        }
        _ => {
            let (sx, sy) = if e.kind == 3 {
                (usize::from(e.layout != 2), usize::from(e.layout == 0))
            } else {
                (0, 0)
            };
            let stride = e.stride_px * bpp;
            let lstride = e.luma_stride_px * bpp;
            let luma_w = ((e.pw - 1) << sx) + sx + 1;
            let luma_rows = ((e.bh - 1) << sy) + 1;
            let mut src = vec![0x3Cu8; (e.bh - 1) * stride + e.pw * bpp];
            let mut dst = vec![0x5Au8; stride * e.bh];
            let mut luma = vec![0x3Cu8; (luma_rows - 1) * lstride + luma_w * bpp];
            let mut scaling = vec![0x3Cu8; max as usize + 1];
            // Girdiler skaler koşudaki sırayla aynı üreteçten (`lut_y` skaler
            // koşunun doldurduğu LUT'tur; aynı değerler).
            for _ in 0..GRAIN_HEIGHT * GRAIN_WIDTH {
                sm64(s);
            }
            for (i, v) in scaling.iter_mut().enumerate() {
                *v = val(e.scal_pat, i as i32, 0, sm64(s), 0, 255, 8) as u8;
            }
            if e.bd == 8 {
                scaling.resize(av1oxide::dsp::fg::SCALING_SIZE_MAX, 0x3C);
            }
            for y in 0..e.bh {
                for x in 0..e.pw {
                    let v = val(e.pix_pat, x as i32, y as i32, sm64(s), 0, max, e.bd);
                    store_px(&mut src, y * e.stride_px + x, bpp, v);
                }
            }
            for y in 0..luma_rows {
                for x in 0..luma_w {
                    let v = val(e.pix_pat, x as i32 + 1, y as i32, sm64(s), 0, max, e.bd);
                    store_px(&mut luma, y * e.luma_stride_px + x, bpp, v);
                }
            }
            if e.kind == 2 {
                (a.fgy_32x32xn)(
                    &mut dst, &src, stride, d, e.pw, &scaling, lut_y, e.bh, e.row_num, bd,
                );
            } else {
                (a.fguv_32x32xn[e.layout])(
                    &mut dst, &src, stride, d, e.pw, &scaling, lut_y, e.bh, e.row_num, &luma,
                    lstride, e.uv, e.is_id, bd,
                );
            }
            for (i, &b) in dst.iter().enumerate() {
                let inside = i / stride < e.bh && i % stride < e.pw * bpp;
                assert!(
                    inside || b == 0x5A,
                    "SIMD {} bölge dışı yazdı",
                    KIND_NAMES[e.kind]
                );
            }
            for y in 0..e.bh {
                h.update(&dst[y * stride..y * stride + e.pw * bpp]);
            }
        }
    }
    h.finish()
}
